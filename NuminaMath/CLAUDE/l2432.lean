import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2432_243201

theorem sqrt_equation_solution (x : ℝ) :
  (3 * x - 2 > 0) →
  (Real.sqrt (3 * x - 2) + 9 / Real.sqrt (3 * x - 2) = 6) ↔
  (x = 11 / 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2432_243201


namespace NUMINAMATH_CALUDE_train_length_l2432_243272

/-- Given a bridge and a train, prove the length of the train -/
theorem train_length 
  (bridge_length : ℝ) 
  (train_cross_time : ℝ) 
  (man_cross_time : ℝ) 
  (train_speed : ℝ) 
  (h1 : bridge_length = 180) 
  (h2 : train_cross_time = 20) 
  (h3 : man_cross_time = 8) 
  (h4 : train_speed = 15) : 
  ∃ train_length : ℝ, train_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2432_243272


namespace NUMINAMATH_CALUDE_women_average_age_l2432_243202

theorem women_average_age 
  (n : Nat) 
  (initial_avg : ℝ) 
  (age_increase : ℝ) 
  (man1_age : ℝ) 
  (man2_age : ℝ) 
  (h1 : n = 7) 
  (h2 : age_increase = 4) 
  (h3 : man1_age = 26) 
  (h4 : man2_age = 30) 
  (h5 : n * (initial_avg + age_increase) = n * initial_avg - man1_age - man2_age + (women_avg * 2)) : 
  women_avg = 42 := by
  sorry

#check women_average_age

end NUMINAMATH_CALUDE_women_average_age_l2432_243202


namespace NUMINAMATH_CALUDE_red_balls_count_l2432_243273

/-- Given a bag of balls with red and yellow colors, prove that the number of red balls is 6 -/
theorem red_balls_count (total_balls : ℕ) (prob_red : ℚ) : 
  total_balls = 15 → prob_red = 2/5 → (prob_red * total_balls : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2432_243273


namespace NUMINAMATH_CALUDE_canteen_distance_l2432_243218

/-- Given a right triangle with one leg of length 400 rods and hypotenuse of length 700 rods,
    the point on the other leg that is equidistant from both endpoints of the hypotenuse
    is approximately 1711 rods from each endpoint. -/
theorem canteen_distance (a b c : ℝ) (h1 : a = 400) (h2 : c = 700) (h3 : a^2 + b^2 = c^2) :
  let x := (2 * a^2 + 2 * b^2) / (2 * b)
  ∃ ε > 0, abs (x - 1711) < ε :=
sorry

end NUMINAMATH_CALUDE_canteen_distance_l2432_243218


namespace NUMINAMATH_CALUDE_max_blocks_fit_l2432_243280

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the problem of fitting smaller blocks into a larger box -/
structure BlockFittingProblem where
  largeBox : BoxDimensions
  smallBlock : BoxDimensions

/-- Calculates the maximum number of blocks that can fit based on volume -/
def maxBlocksByVolume (p : BlockFittingProblem) : ℕ :=
  (boxVolume p.largeBox) / (boxVolume p.smallBlock)

/-- Calculates the maximum number of blocks that can fit based on physical arrangement -/
def maxBlocksByArrangement (p : BlockFittingProblem) : ℕ :=
  (p.largeBox.length / p.smallBlock.length) *
  (p.largeBox.width / p.smallBlock.width) *
  (p.largeBox.height / p.smallBlock.height)

/-- The main theorem stating that the maximum number of blocks that can fit is 6 -/
theorem max_blocks_fit (p : BlockFittingProblem) 
    (h1 : p.largeBox = ⟨4, 3, 2⟩) 
    (h2 : p.smallBlock = ⟨3, 1, 1⟩) : 
    min (maxBlocksByVolume p) (maxBlocksByArrangement p) = 6 := by
  sorry


end NUMINAMATH_CALUDE_max_blocks_fit_l2432_243280


namespace NUMINAMATH_CALUDE_parabola_focus_l2432_243246

/-- A parabola is defined by its equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of a parabola is a point -/
def Focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola y^2 = 4x is (1, 0) -/
theorem parabola_focus :
  ∀ (p : ℝ × ℝ), p ∈ Parabola → Focus = (1, 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l2432_243246


namespace NUMINAMATH_CALUDE_large_circle_diameter_l2432_243224

/-- The diameter of a circle that encompasses six smaller tangent circles -/
theorem large_circle_diameter (r : ℝ) (offset : ℝ) : 
  r = 4 ∧ 
  offset = 1 → 
  2 * (Real.sqrt 17 + 4) = 
    2 * (Real.sqrt ((r - offset)^2 + (2*r/2)^2) + r) :=
by sorry

end NUMINAMATH_CALUDE_large_circle_diameter_l2432_243224


namespace NUMINAMATH_CALUDE_fraction_inequality_l2432_243228

theorem fraction_inequality (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hm : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2432_243228


namespace NUMINAMATH_CALUDE_f_properties_l2432_243200

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - a) - Real.log (x + a)

theorem f_properties :
  (∀ x > -1/2, f (1/2) x < f (1/2) (1/2)) ∧ 
  (∀ x > 1/2, f (1/2) x > f (1/2) (1/2)) ∧
  (f (1/2) (1/2) = 1) ∧
  (∀ a ≤ 1, ∀ x > -a, f a x > 0) := by sorry

end NUMINAMATH_CALUDE_f_properties_l2432_243200


namespace NUMINAMATH_CALUDE_solution_ranges_l2432_243231

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2*m - 2

-- Define the conditions
def has_solution_in_closed_interval (m : ℝ) : Prop :=
  ∃ x, x ∈ Set.Icc 0 (3/2) ∧ quadratic m x = 0

def has_solution_in_open_interval (m : ℝ) : Prop :=
  ∃ x, x ∈ Set.Ioo 0 (3/2) ∧ quadratic m x = 0

def has_exactly_one_solution_in_open_interval (m : ℝ) : Prop :=
  ∃! x, x ∈ Set.Ioo 0 (3/2) ∧ quadratic m x = 0

def has_two_solutions_in_closed_interval (m : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ x ∈ Set.Icc 0 (3/2) ∧ y ∈ Set.Icc 0 (3/2) ∧ quadratic m x = 0 ∧ quadratic m y = 0

-- Theorem statements
theorem solution_ranges :
  (∀ m, has_solution_in_closed_interval m ↔ m ∈ Set.Icc (-1/2) (4 - 2*Real.sqrt 2)) ∧
  (∀ m, has_solution_in_open_interval m ↔ m ∈ Set.Ico (-1/2) (4 - 2*Real.sqrt 2)) ∧
  (∀ m, has_exactly_one_solution_in_open_interval m ↔ m ∈ Set.Ioc (-1/2) 1) ∧
  (∀ m, has_two_solutions_in_closed_interval m ↔ m ∈ Set.Ioo 1 (4 - 2*Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_solution_ranges_l2432_243231


namespace NUMINAMATH_CALUDE_circle_radius_condition_l2432_243230

theorem circle_radius_condition (c : ℝ) : 
  (∀ x y : ℝ, x^2 - 4*x + y^2 + 2*y + c = 0 ↔ (x - 2)^2 + (y + 1)^2 = 5^2) → 
  c = -20 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_condition_l2432_243230


namespace NUMINAMATH_CALUDE_pizza_area_increase_l2432_243206

/-- Given that the radius of a large pizza is 40% larger than the radius of a medium pizza,
    prove that the percent increase in area between a medium and a large pizza is 96%. -/
theorem pizza_area_increase (r : ℝ) (h : r > 0) : 
  let large_radius := 1.4 * r
  let medium_area := Real.pi * r^2
  let large_area := Real.pi * large_radius^2
  (large_area - medium_area) / medium_area * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l2432_243206


namespace NUMINAMATH_CALUDE_bug_probability_after_10_moves_l2432_243242

/-- Probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n+1 => (1/3) * (1 - Q n)

/-- The probability of the bug returning to its starting vertex on a square after 10 moves is 3431/19683 -/
theorem bug_probability_after_10_moves :
  Q 10 = 3431 / 19683 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_after_10_moves_l2432_243242


namespace NUMINAMATH_CALUDE_no_money_left_l2432_243222

theorem no_money_left (total_money : ℝ) (total_items : ℝ) (h1 : total_money > 0) (h2 : total_items > 0) :
  (1 / 3 : ℝ) * total_money = (1 / 3 : ℝ) * total_items * (total_money / total_items) →
  total_money - total_items * (total_money / total_items) = 0 := by
sorry

end NUMINAMATH_CALUDE_no_money_left_l2432_243222


namespace NUMINAMATH_CALUDE_right_triangle_sine_cosine_sum_equality_l2432_243284

theorem right_triangle_sine_cosine_sum_equality (A B C : ℝ) (x y : ℝ) 
  (h1 : A + B + C = π / 2)  -- ∠C is a right angle
  (h2 : 0 ≤ A ∧ A ≤ π / 2)  -- A is an angle in the right triangle
  (h3 : 0 ≤ B ∧ B ≤ π / 2)  -- B is an angle in the right triangle
  (h4 : x = Real.sin A + Real.cos A)  -- Definition of x
  (h5 : y = Real.sin B + Real.cos B)  -- Definition of y
  : x = y := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sine_cosine_sum_equality_l2432_243284


namespace NUMINAMATH_CALUDE_power_of_power_l2432_243288

theorem power_of_power (a : ℝ) : (a^2)^10 = a^20 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l2432_243288


namespace NUMINAMATH_CALUDE_total_movies_in_five_years_l2432_243269

-- Define the number of movies L&J Productions makes per year
def lj_movies_per_year : ℕ := 220

-- Define the percentage increase for Johnny TV
def johnny_tv_increase_percent : ℕ := 25

-- Define the number of years
def years : ℕ := 5

-- Statement to prove
theorem total_movies_in_five_years :
  (lj_movies_per_year + (lj_movies_per_year * johnny_tv_increase_percent) / 100 + lj_movies_per_year) * years = 2475 := by
  sorry

end NUMINAMATH_CALUDE_total_movies_in_five_years_l2432_243269


namespace NUMINAMATH_CALUDE_min_cube_sum_l2432_243267

theorem min_cube_sum (a b t : ℝ) (h : a + b = t) :
  ∃ (min : ℝ), min = t^3 / 4 ∧ ∀ (x y : ℝ), x + y = t → x^3 + y^3 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_cube_sum_l2432_243267


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2432_243214

/-- The length of the diagonal of a rectangle with length 20√5 and width 10√3 is 10√23 -/
theorem rectangle_diagonal (length width diagonal : ℝ) 
  (h_length : length = 20 * Real.sqrt 5)
  (h_width : width = 10 * Real.sqrt 3)
  (h_diagonal : diagonal^2 = length^2 + width^2) : 
  diagonal = 10 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2432_243214


namespace NUMINAMATH_CALUDE_num_women_is_sixteen_l2432_243254

-- Define the number of men
def num_men : ℕ := 24

-- Define the daily wage of a man
def man_wage : ℕ := 350

-- Define the total daily wage
def total_wage : ℕ := 11600

-- Define the number of women in the second condition
def women_in_second_condition : ℕ := 37

-- Define the function to calculate the number of women
def calculate_women : ℕ := 16

-- Theorem statement
theorem num_women_is_sixteen :
  ∃ (women_wage : ℕ),
    -- Condition 1: Total wage equation
    num_men * man_wage + calculate_women * women_wage = total_wage ∧
    -- Condition 2: Half men and 37 women earn the same as all men and all women
    (num_men / 2) * man_wage + women_in_second_condition * women_wage = num_men * man_wage + calculate_women * women_wage :=
by
  sorry


end NUMINAMATH_CALUDE_num_women_is_sixteen_l2432_243254


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l2432_243266

theorem integer_pairs_satisfying_equation : 
  {(x, y) : ℤ × ℤ | (y - 2) * x^2 + (y^2 - 6*y + 8) * x = y^2 - 5*y + 62} = 
  {(8, 3), (2, 9), (-7, 9), (-7, 3), (2, -6), (8, -6)} := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l2432_243266


namespace NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l2432_243281

theorem fourth_root_256_times_cube_root_64_times_sqrt_16 :
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 64 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l2432_243281


namespace NUMINAMATH_CALUDE_distribute_spots_correct_l2432_243215

/-- The number of ways to distribute 8 spots among 6 classes with at least one spot per class -/
def distribute_spots : ℕ := 21

/-- The number of senior classes -/
def num_classes : ℕ := 6

/-- The total number of spots to be distributed -/
def total_spots : ℕ := 8

/-- The minimum number of spots per class -/
def min_spots_per_class : ℕ := 1

theorem distribute_spots_correct :
  distribute_spots = 
    (num_classes.choose 2) + num_classes ∧
  num_classes * min_spots_per_class ≤ total_spots ∧
  total_spots - num_classes * min_spots_per_class = 2 := by
  sorry

end NUMINAMATH_CALUDE_distribute_spots_correct_l2432_243215


namespace NUMINAMATH_CALUDE_additional_money_needed_l2432_243208

def dictionary_cost : ℕ := 11
def dinosaur_book_cost : ℕ := 19
def cookbook_cost : ℕ := 7
def savings : ℕ := 8

theorem additional_money_needed : 
  dictionary_cost + dinosaur_book_cost + cookbook_cost - savings = 29 := by
sorry

end NUMINAMATH_CALUDE_additional_money_needed_l2432_243208


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l2432_243289

theorem consecutive_integers_average (c d : ℝ) : 
  (c ≥ 1) →  -- Ensure c is positive
  (d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5)) / 6 = c + 5) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l2432_243289


namespace NUMINAMATH_CALUDE_books_read_l2432_243268

/-- The number of books read in the 'crazy silly school' series -/
theorem books_read (total_books : ℕ) (unread_books : ℕ) (h1 : total_books = 20) (h2 : unread_books = 5) :
  total_books - unread_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_read_l2432_243268


namespace NUMINAMATH_CALUDE_division_problem_l2432_243250

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 1565 → divisor = 24 → remainder = 5 → quotient = 65 →
  dividend = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l2432_243250


namespace NUMINAMATH_CALUDE_unique_base_thirteen_l2432_243263

/-- Converts a digit character to its numeric value -/
def char_to_digit (c : Char) : ℕ :=
  if c.isDigit then c.toNat - '0'.toNat
  else if c = 'A' then 10
  else if c = 'B' then 11
  else if c = 'C' then 12
  else 0

/-- Converts a string representation of a number in base a to its decimal value -/
def to_decimal (s : String) (a : ℕ) : ℕ :=
  s.foldr (fun c acc => char_to_digit c + a * acc) 0

/-- Checks if the equation 375_a + 592_a = 9C7_a is satisfied for a given base a -/
def equation_satisfied (a : ℕ) : Prop :=
  to_decimal "375" a + to_decimal "592" a = to_decimal "9C7" a

theorem unique_base_thirteen :
  ∃! a : ℕ, a > 12 ∧ equation_satisfied a ∧ char_to_digit 'C' = 12 :=
sorry

end NUMINAMATH_CALUDE_unique_base_thirteen_l2432_243263


namespace NUMINAMATH_CALUDE_orange_price_is_60_l2432_243295

/-- The price of an orange in cents, given the conditions of the fruit stand problem -/
def orange_price : ℕ :=
  let apple_price : ℕ := 40
  let total_fruits : ℕ := 15
  let initial_avg_price : ℕ := 48
  let final_avg_price : ℕ := 45
  let removed_oranges : ℕ := 3
  60

/-- Theorem stating that the price of an orange is 60 cents -/
theorem orange_price_is_60 :
  orange_price = 60 := by sorry

end NUMINAMATH_CALUDE_orange_price_is_60_l2432_243295


namespace NUMINAMATH_CALUDE_average_temperature_proof_l2432_243248

/-- Given the average temperature for four days and individual temperatures for two days,
    prove that the average temperature for a different set of four days is as calculated. -/
theorem average_temperature_proof
  (avg_mon_to_thu : ℝ)
  (temp_mon : ℝ)
  (temp_fri : ℝ)
  (h1 : avg_mon_to_thu = 48)
  (h2 : temp_mon = 40)
  (h3 : temp_fri = 32) :
  (4 * avg_mon_to_thu - temp_mon + temp_fri) / 4 = 46 := by
  sorry


end NUMINAMATH_CALUDE_average_temperature_proof_l2432_243248


namespace NUMINAMATH_CALUDE_range_of_a_l2432_243265

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ≤ -1
def q (a x : ℝ) : Prop := a ≤ x ∧ x < a + 2

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q a x → p x) ∧ ¬(∀ x, p x → q a x)

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, sufficient_not_necessary a ↔ a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2432_243265


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2432_243279

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  (8 - i) / (2 + i) = 3 - 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2432_243279


namespace NUMINAMATH_CALUDE_triangle_area_l2432_243278

theorem triangle_area (x : ℝ) (α : ℝ) : 
  let BC := 4*x
  let CD := x
  let AC := 8*x*(Real.sqrt 2/Real.sqrt 3)
  let AD := (3/4 : ℝ)
  let cos_α := Real.sqrt 2/Real.sqrt 3
  let sin_α := 1/Real.sqrt 3
  (AD^2 = 33*x^2) →
  (1/2 * AC * BC * sin_α = Real.sqrt 2/11) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2432_243278


namespace NUMINAMATH_CALUDE_candy_problem_l2432_243237

theorem candy_problem (n : ℕ) (x : ℕ) (h1 : n > 1) (h2 : x > 1) 
  (h3 : ∀ i : ℕ, i < n → x = (n - 1) * x - 7) : 
  n * x = 21 := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l2432_243237


namespace NUMINAMATH_CALUDE_batting_highest_score_l2432_243255

-- Define the given conditions
def total_innings : ℕ := 46
def overall_average : ℚ := 60
def score_difference : ℕ := 180
def average_excluding_extremes : ℚ := 58
def min_half_centuries : ℕ := 15
def min_centuries : ℕ := 10

-- Define the function to calculate the highest score
def highest_score : ℕ := 194

-- Theorem statement
theorem batting_highest_score :
  (total_innings : ℚ) * overall_average = 
    (total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + (highest_score - score_difference) ∧
  highest_score ≥ 100 ∧
  min_half_centuries + min_centuries ≤ total_innings - 2 :=
by sorry

end NUMINAMATH_CALUDE_batting_highest_score_l2432_243255


namespace NUMINAMATH_CALUDE_fraction_of_25_l2432_243210

theorem fraction_of_25 : 
  ∃ (x : ℚ), x * 25 + 8 = 70 * 40 / 100 ∧ x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_25_l2432_243210


namespace NUMINAMATH_CALUDE_hip_size_conversion_l2432_243213

/-- Converts inches to millimeters given the conversion factors -/
def inches_to_mm (inches_per_foot : ℚ) (mm_per_foot : ℚ) (inches : ℚ) : ℚ :=
  inches * (mm_per_foot / inches_per_foot)

/-- Proves that 42 inches is equivalent to 1067.5 millimeters -/
theorem hip_size_conversion (inches_per_foot mm_per_foot : ℚ) 
  (h1 : inches_per_foot = 12)
  (h2 : mm_per_foot = 305) : 
  inches_to_mm inches_per_foot mm_per_foot 42 = 1067.5 := by
  sorry

#eval inches_to_mm 12 305 42

end NUMINAMATH_CALUDE_hip_size_conversion_l2432_243213


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l2432_243212

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l2432_243212


namespace NUMINAMATH_CALUDE_x_neg_one_is_local_minimum_l2432_243275

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

theorem x_neg_one_is_local_minimum :
  ∃ δ > 0, ∀ x : ℝ, x ≠ -1 ∧ |x - (-1)| < δ → f x ≥ f (-1) := by
  sorry

end NUMINAMATH_CALUDE_x_neg_one_is_local_minimum_l2432_243275


namespace NUMINAMATH_CALUDE_set_A_is_open_interval_zero_two_l2432_243232

-- Define the function f(x) = x^3 - x
def f (x : ℝ) : ℝ := x^3 - x

-- Define the set A
def A : Set ℝ := {a : ℝ | a > 0 ∧ ∃ x : ℝ, f (x + a) = f x}

-- Theorem statement
theorem set_A_is_open_interval_zero_two :
  A = Set.Ioo 0 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_set_A_is_open_interval_zero_two_l2432_243232


namespace NUMINAMATH_CALUDE_total_container_weight_l2432_243226

def container_weight (steel_weight tin_weight copper_weight aluminum_weight : ℝ) : ℝ :=
  10 * steel_weight + 15 * tin_weight + 12 * copper_weight + 8 * aluminum_weight

theorem total_container_weight :
  ∀ (steel_weight tin_weight copper_weight aluminum_weight : ℝ),
    steel_weight = 2 * tin_weight →
    steel_weight = copper_weight + 20 →
    copper_weight = 90 →
    aluminum_weight = tin_weight + 10 →
    container_weight steel_weight tin_weight copper_weight aluminum_weight = 3525 := by
  sorry

end NUMINAMATH_CALUDE_total_container_weight_l2432_243226


namespace NUMINAMATH_CALUDE_necessary_condition_for_124_l2432_243238

/-- A line in the form y = (m/n)x - 1/n -/
structure Line where
  m : ℝ
  n : ℝ
  n_nonzero : n ≠ 0

/-- Predicate for a line passing through the first, second, and fourth quadrants -/
def passes_through_124 (l : Line) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₄ y₄ : ℝ),
    x₁ > 0 ∧ y₁ > 0 ∧  -- First quadrant
    x₂ < 0 ∧ y₂ > 0 ∧  -- Second quadrant
    x₄ > 0 ∧ y₄ < 0 ∧  -- Fourth quadrant
    y₁ = (l.m / l.n) * x₁ - 1 / l.n ∧
    y₂ = (l.m / l.n) * x₂ - 1 / l.n ∧
    y₄ = (l.m / l.n) * x₄ - 1 / l.n

/-- Theorem stating the necessary condition -/
theorem necessary_condition_for_124 (l : Line) :
  passes_through_124 l → l.m > 0 ∧ l.n < 0 :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_for_124_l2432_243238


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2432_243225

theorem complex_number_quadrant (z : ℂ) (h : z * (2 - Complex.I) = 1) :
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2432_243225


namespace NUMINAMATH_CALUDE_total_borrowed_by_lunchtime_l2432_243235

/-- Represents the number of books borrowed from shelf A by lunchtime -/
def x : ℕ := 50

/-- Represents the number of books borrowed from shelf B by lunchtime -/
def y : ℕ := 60

/-- Represents the number of books borrowed from shelf C by lunchtime -/
def z : ℕ := 85

/-- The initial number of books on shelf A -/
def shelf_a_initial : ℕ := 100

/-- The number of books added to shelf A after lunchtime -/
def shelf_a_added : ℕ := 40

/-- The number of books borrowed from shelf A after lunchtime -/
def shelf_a_borrowed_after : ℕ := 30

/-- The number of books remaining on shelf A by evening -/
def shelf_a_remaining : ℕ := 60

/-- The initial number of books on shelf B -/
def shelf_b_initial : ℕ := 150

/-- The number of books added to shelf B -/
def shelf_b_added : ℕ := 20

/-- The number of books borrowed from shelf B after morning -/
def shelf_b_borrowed_after : ℕ := 30

/-- The number of books remaining on shelf B by evening -/
def shelf_b_remaining : ℕ := 80

/-- The initial number of books on shelf C -/
def shelf_c_initial : ℕ := 200

/-- The total number of books borrowed from shelf C throughout the day -/
def shelf_c_borrowed_total : ℕ := 130

/-- The number of books borrowed from shelf C after lunchtime -/
def shelf_c_borrowed_after : ℕ := 45

theorem total_borrowed_by_lunchtime :
  x + y + z = 195 ∧
  shelf_a_initial - x + shelf_a_added - shelf_a_borrowed_after = shelf_a_remaining ∧
  shelf_b_initial - y + shelf_b_added - shelf_b_borrowed_after = shelf_b_remaining ∧
  shelf_c_initial - shelf_c_borrowed_total = shelf_c_initial - z - shelf_c_borrowed_after :=
by sorry

end NUMINAMATH_CALUDE_total_borrowed_by_lunchtime_l2432_243235


namespace NUMINAMATH_CALUDE_night_rides_calculation_wills_ferris_wheel_rides_l2432_243244

/-- Calculates the number of night rides on a Ferris wheel -/
def night_rides (total_rides day_rides : ℕ) : ℕ :=
  total_rides - day_rides

/-- Theorem: The number of night rides is equal to the total rides minus the day rides -/
theorem night_rides_calculation (total_rides day_rides : ℕ) 
  (h : day_rides ≤ total_rides) : 
  night_rides total_rides day_rides = total_rides - day_rides := by
  sorry

/-- Given Will's specific scenario -/
theorem wills_ferris_wheel_rides : 
  night_rides 13 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_night_rides_calculation_wills_ferris_wheel_rides_l2432_243244


namespace NUMINAMATH_CALUDE_best_distribution_for_1_l2432_243296

/-- Represents a distribution of pearls among 4 people -/
def Distribution := Fin 4 → ℕ

/-- The total number of pearls to be distributed -/
def totalPearls : ℕ := 10

/-- A valid distribution must sum to the total number of pearls -/
def isValidDistribution (d : Distribution) : Prop :=
  (Finset.univ.sum d) = totalPearls

/-- A distribution passes if it has at least half of the votes -/
def passes (d : Distribution) : Prop :=
  2 * (Finset.filter (fun i => d i > 0) Finset.univ).card ≥ 4

/-- The best distribution for person 3 if 1 and 2 are eliminated -/
def bestFor3 : Distribution :=
  fun i => if i = 2 then 10 else 0

/-- The proposed best distribution for person 1 -/
def proposedBest : Distribution :=
  fun i => match i with
  | 0 => 9
  | 2 => 1
  | _ => 0

theorem best_distribution_for_1 :
  isValidDistribution proposedBest ∧
  passes proposedBest ∧
  ∀ d : Distribution, isValidDistribution d ∧ passes d → proposedBest 0 ≥ d 0 :=
by sorry

end NUMINAMATH_CALUDE_best_distribution_for_1_l2432_243296


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2432_243298

theorem quadrilateral_diagonal_length 
  (offset1 : ℝ) (offset2 : ℝ) (area : ℝ) (diagonal : ℝ) :
  offset1 = 10 →
  offset2 = 6 →
  area = 240 →
  area = (1 / 2) * diagonal * (offset1 + offset2) →
  diagonal = 30 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2432_243298


namespace NUMINAMATH_CALUDE_no_linear_term_implies_equal_coefficients_l2432_243219

/-- Given a polynomial (x+p)(x-q) with no linear term in x, prove that p = q -/
theorem no_linear_term_implies_equal_coefficients (p q : ℝ) :
  (∀ x : ℝ, ∃ a b : ℝ, (x + p) * (x - q) = a * x^2 + b) → p = q := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_equal_coefficients_l2432_243219


namespace NUMINAMATH_CALUDE_valentines_theorem_l2432_243239

theorem valentines_theorem (boys girls : ℕ) : 
  (boys * girls = boys + girls + 46) → (boys * girls = 96) := by
  sorry

end NUMINAMATH_CALUDE_valentines_theorem_l2432_243239


namespace NUMINAMATH_CALUDE_discriminant_of_5x2_minus_3x_plus_4_l2432_243253

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The discriminant of the quadratic equation 5x² - 3x + 4 is -71 -/
theorem discriminant_of_5x2_minus_3x_plus_4 :
  discriminant 5 (-3) 4 = -71 := by sorry

end NUMINAMATH_CALUDE_discriminant_of_5x2_minus_3x_plus_4_l2432_243253


namespace NUMINAMATH_CALUDE_expand_expression_l2432_243207

theorem expand_expression (x : ℝ) : (x - 3) * (4 * x + 12) = 4 * x^2 - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2432_243207


namespace NUMINAMATH_CALUDE_sara_team_wins_l2432_243270

/-- Represents a basketball team's game statistics -/
structure TeamStats where
  total_games : ℕ
  lost_games : ℕ

/-- Calculates the number of games won by a team -/
def games_won (stats : TeamStats) : ℕ :=
  stats.total_games - stats.lost_games

/-- Theorem: For Sara's team, the number of games won is 12 -/
theorem sara_team_wins (sara_team : TeamStats) 
  (h1 : sara_team.total_games = 16) 
  (h2 : sara_team.lost_games = 4) : 
  games_won sara_team = 12 := by
  sorry

end NUMINAMATH_CALUDE_sara_team_wins_l2432_243270


namespace NUMINAMATH_CALUDE_negation_of_p_l2432_243258

open Real

def p : Prop := ∃ x : ℚ, 2^(x : ℝ) - log x < 2

theorem negation_of_p : ¬p ↔ ∀ x : ℚ, 2^(x : ℝ) - log x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l2432_243258


namespace NUMINAMATH_CALUDE_complex_square_plus_self_l2432_243203

theorem complex_square_plus_self (z : ℂ) (h : z = -1/2 + (Real.sqrt 3)/2 * Complex.I) : z^2 + z = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_plus_self_l2432_243203


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2432_243256

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 + X^2 + 1 : Polynomial ℝ) = q * (X^2 - 4*X + 7) + (12*X - 69) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2432_243256


namespace NUMINAMATH_CALUDE_catch_up_time_is_55_minutes_l2432_243221

/-- The time it takes for Bob to catch up with John -/
def catch_up_time (john_speed bob_speed initial_distance stop_time : ℚ) : ℚ :=
  let relative_speed := bob_speed - john_speed
  let time_without_stop := initial_distance / relative_speed
  (time_without_stop + stop_time / 60) * 60

theorem catch_up_time_is_55_minutes :
  catch_up_time 2 6 3 10 = 55 := by sorry

end NUMINAMATH_CALUDE_catch_up_time_is_55_minutes_l2432_243221


namespace NUMINAMATH_CALUDE_set_union_problem_l2432_243293

theorem set_union_problem (m : ℝ) : 
  let A : Set ℝ := {1, 2^m}
  let B : Set ℝ := {0, 2}
  A ∪ B = {0, 1, 2, 8} → m = 3 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l2432_243293


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2432_243227

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x^2 - x - 6 < 0 ↔ x ∈ Set.Ioo (-2 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2432_243227


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l2432_243271

/-- A regular polygon with side length 5 units and exterior angle 120 degrees has a perimeter of 15 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n ≥ 3 →
  side_length = 5 →
  exterior_angle = 120 →
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 15 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l2432_243271


namespace NUMINAMATH_CALUDE_triangle_side_length_l2432_243291

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : Real) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = π/3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2432_243291


namespace NUMINAMATH_CALUDE_students_not_reading_l2432_243294

theorem students_not_reading (total : ℕ) (three_or_more : ℚ) (two : ℚ) (one : ℚ) :
  total = 240 →
  three_or_more = 1 / 6 →
  two = 35 / 100 →
  one = 5 / 12 →
  ↑total - (↑total * (three_or_more + two + one)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_not_reading_l2432_243294


namespace NUMINAMATH_CALUDE_smallest_three_digit_prime_with_composite_reverse_l2432_243243

/-- A function that reverses the digits of a three-digit number -/
def reverseDigits (n : Nat) : Nat :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- A predicate that checks if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- A predicate that checks if a number is composite -/
def isComposite (n : Nat) : Prop :=
  n > 1 ∧ ∃ d : Nat, d > 1 ∧ d < n ∧ n % d = 0

theorem smallest_three_digit_prime_with_composite_reverse :
  ∃ (p : Nat),
    p = 103 ∧
    isPrime p ∧
    100 ≤ p ∧ p < 1000 ∧
    isComposite (reverseDigits p) ∧
    ∀ (q : Nat),
      isPrime q ∧
      100 ≤ q ∧ q < p →
      ¬(isComposite (reverseDigits q)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_prime_with_composite_reverse_l2432_243243


namespace NUMINAMATH_CALUDE_product_eval_l2432_243290

theorem product_eval (m : ℤ) (h : m = 3) : (m-2) * (m-1) * m * (m+1) * (m+2) * (m+3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_eval_l2432_243290


namespace NUMINAMATH_CALUDE_fraction_equality_l2432_243276

theorem fraction_equality (p r s u : ℝ) 
  (h1 : p / r = 4)
  (h2 : s / r = 8)
  (h3 : s / u = 1 / 4) :
  u / p = 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2432_243276


namespace NUMINAMATH_CALUDE_value_of_y_l2432_243245

theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2432_243245


namespace NUMINAMATH_CALUDE_total_sales_equals_205_l2432_243247

def apple_price : ℝ := 1.50
def orange_price : ℝ := 1.00

def morning_apples : ℕ := 40
def morning_oranges : ℕ := 30
def afternoon_apples : ℕ := 50
def afternoon_oranges : ℕ := 40

def total_sales : ℝ :=
  apple_price * (morning_apples + afternoon_apples) +
  orange_price * (morning_oranges + afternoon_oranges)

theorem total_sales_equals_205 : total_sales = 205 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_equals_205_l2432_243247


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2432_243233

theorem complex_magnitude_problem (Z : ℂ) (h : (2 + Complex.I) * Z = 3 - Complex.I) :
  Complex.abs Z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2432_243233


namespace NUMINAMATH_CALUDE_additional_plates_count_l2432_243251

/-- Represents the number of choices for each position in a license plate. -/
structure LicensePlateChoices where
  first : Nat
  second : Nat
  third : Nat
  fourth : Nat

/-- Calculates the total number of possible license plates. -/
def totalPlates (choices : LicensePlateChoices) : Nat :=
  choices.first * choices.second * choices.third * choices.fourth

/-- The original choices for each position in TriCity license plates. -/
def originalChoices : LicensePlateChoices :=
  { first := 3, second := 4, third := 2, fourth := 5 }

/-- The new choices after adding two new letters. -/
def newChoices : LicensePlateChoices :=
  { first := originalChoices.first + 1,
    second := originalChoices.second,
    third := originalChoices.third + 1,
    fourth := originalChoices.fourth }

/-- Theorem stating the number of additional license plates after the change. -/
theorem additional_plates_count :
  totalPlates newChoices - totalPlates originalChoices = 120 := by
  sorry

end NUMINAMATH_CALUDE_additional_plates_count_l2432_243251


namespace NUMINAMATH_CALUDE_sin_cos_45_sum_l2432_243209

theorem sin_cos_45_sum : Real.sin (π / 4) + Real.cos (π / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_45_sum_l2432_243209


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_equation_l2432_243264

theorem purely_imaginary_complex_equation (z : ℂ) (a : ℝ) : 
  (z.re = 0) → ((2 - I) * z = a + I) → (a = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_equation_l2432_243264


namespace NUMINAMATH_CALUDE_system_solution_l2432_243241

theorem system_solution (a b c x y z : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (1 / x + 1 / y + 1 / z = 1) ∧ (a * x = b * y) ∧ (b * y = c * z) →
  (x = (a + b + c) / a) ∧ (y = (a + b + c) / b) ∧ (z = (a + b + c) / c) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2432_243241


namespace NUMINAMATH_CALUDE_sum_of_squares_near_n_l2432_243240

theorem sum_of_squares_near_n (n : ℕ) (h : n > 10000) :
  ∃ m : ℕ, ∃ x y : ℕ, 
    m = x^2 + y^2 ∧ 
    0 < m - n ∧
    (m - n : ℝ) < 3 * Real.sqrt (n : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_near_n_l2432_243240


namespace NUMINAMATH_CALUDE_randy_money_problem_l2432_243261

def randy_initial_money (randy_received : ℕ) (randy_gave : ℕ) (randy_left : ℕ) : Prop :=
  ∃ (initial : ℕ), initial + randy_received - randy_gave = randy_left

theorem randy_money_problem :
  randy_initial_money 200 1200 2000 → ∃ (initial : ℕ), initial = 3000 := by
  sorry

end NUMINAMATH_CALUDE_randy_money_problem_l2432_243261


namespace NUMINAMATH_CALUDE_midpoint_between_fractions_l2432_243292

theorem midpoint_between_fractions :
  (1 / 12 + 1 / 20) / 2 = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_between_fractions_l2432_243292


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a6_l2432_243249

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b * b = a * c

theorem arithmetic_geometric_sequence_a6 (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a6_l2432_243249


namespace NUMINAMATH_CALUDE_apple_preference_percentage_l2432_243283

/-- Represents the frequencies of fruits in a survey --/
structure FruitSurvey where
  apples : ℕ
  bananas : ℕ
  cherries : ℕ
  oranges : ℕ
  grapes : ℕ

/-- Calculates the total number of responses in the survey --/
def totalResponses (survey : FruitSurvey) : ℕ :=
  survey.apples + survey.bananas + survey.cherries + survey.oranges + survey.grapes

/-- Calculates the percentage of respondents who preferred apples --/
def applePercentage (survey : FruitSurvey) : ℚ :=
  (survey.apples : ℚ) / (totalResponses survey : ℚ) * 100

/-- The given survey results --/
def givenSurvey : FruitSurvey :=
  { apples := 70
  , bananas := 50
  , cherries := 30
  , oranges := 50
  , grapes := 40 }

theorem apple_preference_percentage :
  applePercentage givenSurvey = 29 := by
  sorry

end NUMINAMATH_CALUDE_apple_preference_percentage_l2432_243283


namespace NUMINAMATH_CALUDE_line_equation_slope_intercept_l2432_243286

/-- Given a line equation, prove its slope and y-intercept -/
theorem line_equation_slope_intercept :
  let line_eq : ℝ × ℝ → ℝ := λ p => 3 * (p.1 + 2) + (-7) * (p.2 - 4)
  ∃ m b : ℝ, m = 3 / 7 ∧ b = -34 / 7 ∧
    ∀ x y : ℝ, line_eq (x, y) = 0 ↔ y = m * x + b := by
  sorry

end NUMINAMATH_CALUDE_line_equation_slope_intercept_l2432_243286


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l2432_243285

theorem bobby_candy_problem (C : ℕ) : 
  (C + 36 = 16 + 58) → C = 38 := by
sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l2432_243285


namespace NUMINAMATH_CALUDE_gcd_of_390_455_546_l2432_243205

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_390_455_546_l2432_243205


namespace NUMINAMATH_CALUDE_rainfall_difference_l2432_243252

def camping_days : ℕ := 14
def rainy_days : ℕ := 7
def friend_rainfall : ℕ := 65

def greg_rainfall : List ℕ := [3, 6, 5, 7, 4, 8, 9]

theorem rainfall_difference :
  friend_rainfall - (greg_rainfall.sum) = 23 :=
by sorry

end NUMINAMATH_CALUDE_rainfall_difference_l2432_243252


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2432_243259

/-- The value of k for which the line y = kx (k > 0) is tangent to the circle (x-√3)^2 + y^2 = 1 -/
theorem tangent_line_to_circle (k : ℝ) : 
  k > 0 ∧ 
  (∃ (x y : ℝ), y = k * x ∧ (x - Real.sqrt 3)^2 + y^2 = 1) ∧
  (∀ (x y : ℝ), y = k * x → (x - Real.sqrt 3)^2 + y^2 ≥ 1) →
  k = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2432_243259


namespace NUMINAMATH_CALUDE_jeans_extra_trips_l2432_243297

theorem jeans_extra_trips (total_trips : ℕ) (jeans_trips : ℕ) 
  (h1 : total_trips = 40) 
  (h2 : jeans_trips = 23) : 
  jeans_trips - (total_trips - jeans_trips) = 6 := by
  sorry

end NUMINAMATH_CALUDE_jeans_extra_trips_l2432_243297


namespace NUMINAMATH_CALUDE_polygon_with_150_degree_angles_is_12_gon_l2432_243282

theorem polygon_with_150_degree_angles_is_12_gon (n : ℕ) 
  (h : n ≥ 3) 
  (interior_angle : ℝ) 
  (h_angle : interior_angle = 150) 
  (h_sum : (n - 2) * 180 = n * interior_angle) : n = 12 := by
sorry

end NUMINAMATH_CALUDE_polygon_with_150_degree_angles_is_12_gon_l2432_243282


namespace NUMINAMATH_CALUDE_distance_AB_is_550_l2432_243220

/-- The distance between points A and B --/
def distance_AB : ℝ := 550

/-- Xiaodong's speed in meters per minute --/
def speed_Xiaodong : ℝ := 50

/-- Xiaorong's speed in meters per minute --/
def speed_Xiaorong : ℝ := 60

/-- Time taken for Xiaodong and Xiaorong to meet, in minutes --/
def meeting_time : ℝ := 10

/-- Theorem stating that the distance between points A and B is 550 meters --/
theorem distance_AB_is_550 :
  distance_AB = (speed_Xiaodong + speed_Xiaorong) * meeting_time / 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_AB_is_550_l2432_243220


namespace NUMINAMATH_CALUDE_triangle_area_is_12_l2432_243262

/-- The area of a triangular region bounded by the x-axis, y-axis, and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

/-- The x-intercept of the line -/
def xIntercept : ℝ := 4

/-- The y-intercept of the line -/
def yIntercept : ℝ := 6

theorem triangle_area_is_12 :
  triangleArea = (1 / 2) * xIntercept * yIntercept :=
sorry

end NUMINAMATH_CALUDE_triangle_area_is_12_l2432_243262


namespace NUMINAMATH_CALUDE_sqrt_five_addition_l2432_243287

theorem sqrt_five_addition : 2 * Real.sqrt 5 + 3 * Real.sqrt 5 = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_addition_l2432_243287


namespace NUMINAMATH_CALUDE_treasure_burial_year_l2432_243277

def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8^i)) 0

theorem treasure_burial_year : 
  octal_to_decimal [1, 7, 6, 2] = 1465 := by
  sorry

end NUMINAMATH_CALUDE_treasure_burial_year_l2432_243277


namespace NUMINAMATH_CALUDE_prob_at_least_two_evens_eq_247_256_l2432_243299

/-- Probability of getting an even number on a single roll of a standard die -/
def p_even : ℚ := 1/2

/-- Number of rolls -/
def n : ℕ := 8

/-- Probability of getting exactly k even numbers in n rolls -/
def prob_k_evens (k : ℕ) : ℚ :=
  (n.choose k) * (p_even ^ k) * ((1 - p_even) ^ (n - k))

/-- Probability of getting at least two even numbers in n rolls -/
def prob_at_least_two_evens : ℚ :=
  1 - (prob_k_evens 0 + prob_k_evens 1)

theorem prob_at_least_two_evens_eq_247_256 :
  prob_at_least_two_evens = 247/256 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_evens_eq_247_256_l2432_243299


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2432_243229

/-- Given (1 + 2i)a + b = 2i, where a and b are real numbers, prove that a = 1 and b = -1 -/
theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2432_243229


namespace NUMINAMATH_CALUDE_sum_of_seven_odds_mod_twelve_l2432_243223

theorem sum_of_seven_odds_mod_twelve (n : ℕ) (h : n = 10331) : 
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) + (n + 12)) % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_odds_mod_twelve_l2432_243223


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l2432_243217

/-- A linear function y = kx - k where k ≠ 0 and k < 0 does not pass through the third quadrant -/
theorem linear_function_not_in_third_quadrant (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ∀ x y : ℝ, y = k * x - k → ¬(x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l2432_243217


namespace NUMINAMATH_CALUDE_new_person_weight_l2432_243216

/-- Given a group of 6 people, if replacing one person weighing 75 kg with a new person
    increases the average weight by 4.5 kg, then the weight of the new person is 102 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 6 →
  weight_increase = 4.5 →
  replaced_weight = 75 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 102 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2432_243216


namespace NUMINAMATH_CALUDE_new_salary_calculation_l2432_243274

def current_salary : ℝ := 10000
def increase_percentage : ℝ := 0.02

theorem new_salary_calculation :
  current_salary * (1 + increase_percentage) = 10200 := by
  sorry

end NUMINAMATH_CALUDE_new_salary_calculation_l2432_243274


namespace NUMINAMATH_CALUDE_leo_assignment_time_theorem_l2432_243260

theorem leo_assignment_time_theorem :
  ∀ (first_part second_part third_part first_break second_break total_time : ℕ),
    first_part = 25 →
    second_part = 2 * first_part →
    first_break = 10 →
    second_break = 15 →
    total_time = 150 →
    total_time = first_part + second_part + third_part + first_break + second_break →
    third_part = 50 := by
  sorry

end NUMINAMATH_CALUDE_leo_assignment_time_theorem_l2432_243260


namespace NUMINAMATH_CALUDE_blake_initial_milk_l2432_243211

/-- The amount of milk needed for one milkshake in ounces -/
def milk_per_milkshake : ℕ := 4

/-- The amount of ice cream needed for one milkshake in ounces -/
def ice_cream_per_milkshake : ℕ := 12

/-- The total amount of ice cream available in ounces -/
def total_ice_cream : ℕ := 192

/-- The amount of milk left over after making milkshakes in ounces -/
def milk_leftover : ℕ := 8

/-- The initial amount of milk Blake had -/
def initial_milk : ℕ := total_ice_cream / ice_cream_per_milkshake * milk_per_milkshake + milk_leftover

theorem blake_initial_milk :
  initial_milk = 72 :=
sorry

end NUMINAMATH_CALUDE_blake_initial_milk_l2432_243211


namespace NUMINAMATH_CALUDE_circle_line_intersection_l2432_243234

/-- Given a circle and a line, prove the value of a when the chord length is known -/
theorem circle_line_intersection (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*y = 0) →
  (∀ x y : ℝ, x + y = 0) →
  (∃ x1 y1 x2 y2 : ℝ, 
    x1^2 + y1^2 - 2*a*y1 = 0 ∧
    x2^2 + y2^2 - 2*a*y2 = 0 ∧
    x1 + y1 = 0 ∧
    x2 + y2 = 0 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 8) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l2432_243234


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2432_243204

theorem two_numbers_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2432_243204


namespace NUMINAMATH_CALUDE_perpendicular_lines_in_parallel_planes_l2432_243236

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (lies_in : Line → Plane → Prop)
variable (not_lies_on : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_in_parallel_planes
  (α β : Plane) (l m : Line)
  (h1 : lies_in l α)
  (h2 : not_lies_on m α)
  (h3 : parallel α β)
  (h4 : perpendicular m β) :
  line_perpendicular l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_in_parallel_planes_l2432_243236


namespace NUMINAMATH_CALUDE_chocolate_price_after_discount_l2432_243257

/-- The final price of a chocolate after discount -/
def final_price (original_cost discount : ℚ) : ℚ :=
  original_cost - discount

/-- Theorem: The final price of a chocolate with original cost $2 and discount $0.57 is $1.43 -/
theorem chocolate_price_after_discount :
  final_price 2 0.57 = 1.43 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_price_after_discount_l2432_243257
