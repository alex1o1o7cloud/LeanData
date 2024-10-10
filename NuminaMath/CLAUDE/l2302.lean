import Mathlib

namespace problem_1_problem_2_problem_3_problem_4_l2302_230200

-- Problem 1
theorem problem_1 : (-3/7) + 1/5 + 2/7 + (-6/5) = -8/7 := by sorry

-- Problem 2
theorem problem_2 : -(-1) + 3^2 / (1-4) * 2 = -5 := by sorry

-- Problem 3
theorem problem_3 : (-1/6)^2 / ((1/2 - 1/3)^2) / |(-6)|^2 = 1/36 := by sorry

-- Problem 4
theorem problem_4 : (-1)^1000 - 2.45 * 8 + 2.55 * (-8) = -39 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2302_230200


namespace f_not_tangent_to_x_axis_max_a_for_monotone_g_l2302_230272

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x - (a / 2) * x^2

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - a * x

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2 * x

-- Define the derivative of g(x)
def g_deriv (a : ℝ) (x : ℝ) : ℝ := f_deriv a x + 2

-- Theorem 1: f(x) cannot be tangent to the x-axis for any a
theorem f_not_tangent_to_x_axis (a : ℝ) : ¬∃ x : ℝ, f a x = 0 ∧ f_deriv a x = 0 := by
  sorry

-- Theorem 2: The maximum integer value of a for which g(x) is monotonically increasing is 1
theorem max_a_for_monotone_g : 
  ∀ a : ℤ, (∀ x : ℝ, g_deriv a x ≥ 0) → a ≤ 1 := by
  sorry

end

end f_not_tangent_to_x_axis_max_a_for_monotone_g_l2302_230272


namespace initial_gasohol_volume_l2302_230202

/-- Represents the composition of a fuel mixture -/
structure FuelMixture where
  ethanol : ℝ
  gasoline : ℝ
  valid : ethanol + gasoline = 1

/-- Represents the state of the fuel tank -/
structure FuelTank where
  volume : ℝ
  mixture : FuelMixture

def initial_mixture : FuelMixture := {
  ethanol := 0.05,
  gasoline := 0.95,
  valid := by norm_num
}

def desired_mixture : FuelMixture := {
  ethanol := 0.1,
  gasoline := 0.9,
  valid := by norm_num
}

def ethanol_added : ℝ := 2

theorem initial_gasohol_volume (initial : FuelTank) :
  initial.mixture = initial_mixture →
  (∃ (final : FuelTank), 
    final.volume = initial.volume + ethanol_added ∧
    final.mixture = desired_mixture) →
  initial.volume = 36 := by
  sorry

end initial_gasohol_volume_l2302_230202


namespace steamer_problem_l2302_230294

theorem steamer_problem :
  ∃ (a b c n k p x : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    1 ≤ n ∧ n ≤ 31 ∧
    1 ≤ k ∧ k ≤ 12 ∧
    p ≥ 0 ∧
    a * b * c * n * k * p + x^3 = 4752862 := by
  sorry

end steamer_problem_l2302_230294


namespace adlai_animal_legs_l2302_230258

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of dogs Adlai has -/
def adlai_dogs : ℕ := 2

/-- The number of chickens Adlai has -/
def adlai_chickens : ℕ := 1

/-- The total number of animal legs Adlai has -/
def total_legs : ℕ := adlai_dogs * dog_legs + adlai_chickens * chicken_legs

theorem adlai_animal_legs : total_legs = 10 := by
  sorry

end adlai_animal_legs_l2302_230258


namespace oxen_count_l2302_230207

/-- The number of oxen in the first group that can plough 1/7th of a field in 2 days. -/
def first_group : ℕ := sorry

/-- The time it takes for the first group to plough the entire field. -/
def total_time : ℕ := 14

/-- The fraction of the field ploughed by the first group. -/
def ploughed_fraction : ℚ := 1/7

/-- The number of oxen in the second group. -/
def second_group : ℕ := 18

/-- The time it takes for the second group to plough the remaining field. -/
def remaining_time : ℕ := 20

/-- The fraction of the field ploughed by the second group. -/
def remaining_fraction : ℚ := 6/7

theorem oxen_count :
  (first_group * total_time) / 1 = (second_group * remaining_time) / remaining_fraction →
  first_group = 30 := by sorry

end oxen_count_l2302_230207


namespace common_remainder_problem_l2302_230271

theorem common_remainder_problem (n : ℕ) : 
  n > 1 ∧ 
  n % 25 = n % 7 ∧ 
  n = 175 ∧ 
  ∀ m : ℕ, (m > 1 ∧ m % 25 = m % 7) → m ≥ n → 
  n % 25 = 0 :=
by sorry

end common_remainder_problem_l2302_230271


namespace point_C_coordinates_l2302_230274

-- Define the translation function
def translate (x y dx : ℝ) : ℝ × ℝ := (x + dx, y)

-- Define the symmetric point with respect to x-axis
def symmetricX (x y : ℝ) : ℝ × ℝ := (x, -y)

theorem point_C_coordinates :
  let A : ℝ × ℝ := (-1, 2)
  let B := translate A.1 A.2 2
  let C := symmetricX B.1 B.2
  C = (1, -2) := by sorry

end point_C_coordinates_l2302_230274


namespace intersection_point_l2302_230220

def f (x : ℝ) : ℝ := x^3 + 6*x^2 + 16*x + 28

theorem intersection_point :
  ∃! (a b : ℝ), (f a = b ∧ f b = a) ∧ a = -4 ∧ b = -4 := by sorry

end intersection_point_l2302_230220


namespace snowdrift_solution_l2302_230228

def snowdrift_problem (initial_depth : ℝ) : Prop :=
  let day2_depth := initial_depth / 2
  let day3_depth := day2_depth + 6
  let day4_depth := day3_depth + 18
  day4_depth = 34 ∧ initial_depth = 20

theorem snowdrift_solution :
  ∃ (initial_depth : ℝ), snowdrift_problem initial_depth :=
sorry

end snowdrift_solution_l2302_230228


namespace sheela_bank_deposit_l2302_230249

theorem sheela_bank_deposit (monthly_income : ℝ) (deposit_percentage : ℝ) (deposit_amount : ℝ) :
  monthly_income = 11875 →
  deposit_percentage = 32 →
  deposit_amount = (deposit_percentage / 100) * monthly_income →
  deposit_amount = 3796 := by
  sorry

end sheela_bank_deposit_l2302_230249


namespace spade_combination_l2302_230265

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_combination : spade 5 (spade 2 3) = 0 := by
  sorry

end spade_combination_l2302_230265


namespace line_slope_equidistant_points_l2302_230213

/-- The slope of a line passing through (4, 4) and equidistant from points (0, 2) and (12, 8) is -2 -/
theorem line_slope_equidistant_points : 
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y - 4 = m * (x - 4) → 
      (x - 0)^2 + (y - 2)^2 = (x - 12)^2 + (y - 8)^2) → 
    m = -2 :=
by sorry

end line_slope_equidistant_points_l2302_230213


namespace cos_105_degrees_l2302_230260

theorem cos_105_degrees : Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_degrees_l2302_230260


namespace magic_8_ball_probability_l2302_230277

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem magic_8_ball_probability : 
  binomial_probability 7 3 (3/7) = 241920/823543 := by sorry

end magic_8_ball_probability_l2302_230277


namespace isosceles_triangle_base_angle_l2302_230245

theorem isosceles_triangle_base_angle (apex_angle : ℝ) (base_angle : ℝ) :
  apex_angle = 100 → -- The apex angle is 100°
  apex_angle + 2 * base_angle = 180 → -- Sum of angles in a triangle is 180°
  base_angle = 40 := by sorry

end isosceles_triangle_base_angle_l2302_230245


namespace square_value_l2302_230285

theorem square_value : ∃ (square : ℝ), 
  ((11.2 - 1.2 * square) / 4 + 51.2 * square) * 0.1 = 9.1 ∧ square = 1.568 := by
  sorry

end square_value_l2302_230285


namespace intersection_equality_l2302_230204

def M : Set ℝ := {x : ℝ | x < 2012}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_equality : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_equality_l2302_230204


namespace cow_count_l2302_230242

/-- The number of days over which the husk consumption is measured -/
def days : ℕ := 50

/-- The number of bags of husk consumed by the group of cows -/
def group_consumption : ℕ := 50

/-- The number of bags of husk consumed by one cow -/
def single_cow_consumption : ℕ := 1

/-- The number of cows in the farm -/
def num_cows : ℕ := group_consumption

theorem cow_count : num_cows = 50 := by
  sorry

end cow_count_l2302_230242


namespace michael_has_52_robots_l2302_230206

/-- The number of flying robots Tom has -/
def tom_robots : ℕ := 12

/-- The ratio of Michael's robots to Tom's robots -/
def michael_to_tom_ratio : ℕ := 4

/-- The number of robots Tom gives away for every group of robots he has -/
def tom_giveaway_ratio : ℕ := 1

/-- The size of the group of robots Tom considers when giving away -/
def tom_group_size : ℕ := 3

/-- Calculates the number of flying robots Michael has in total -/
def michael_total_robots : ℕ :=
  (michael_to_tom_ratio * tom_robots) + (tom_robots / tom_group_size)

/-- Theorem stating that Michael has 52 flying robots in total -/
theorem michael_has_52_robots : michael_total_robots = 52 := by
  sorry

end michael_has_52_robots_l2302_230206


namespace cube_coplanar_probability_l2302_230235

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of vertices we choose -/
def chosen_vertices : ℕ := 4

/-- The number of ways to choose 4 vertices that lie in the same plane -/
def coplanar_choices : ℕ := 12

/-- The total number of ways to choose 4 vertices from 8 -/
def total_choices : ℕ := Nat.choose cube_vertices chosen_vertices

/-- The probability that 4 randomly chosen vertices of a cube lie in the same plane -/
theorem cube_coplanar_probability : 
  (coplanar_choices : ℚ) / total_choices = 6 / 35 := by sorry

end cube_coplanar_probability_l2302_230235


namespace mixed_decimal_to_vulgar_fraction_l2302_230289

theorem mixed_decimal_to_vulgar_fraction :
  (4 + 13 / 50 : ℚ) = 4.26 ∧
  (1 + 3 / 20 : ℚ) = 1.15 ∧
  (3 + 2 / 25 : ℚ) = 3.08 ∧
  (2 + 37 / 100 : ℚ) = 2.37 := by
  sorry

end mixed_decimal_to_vulgar_fraction_l2302_230289


namespace pizza_toppings_l2302_230292

/-- Given a pizza with 24 slices, where 15 slices have pepperoni and 14 slices have mushrooms,
    prove that 5 slices have both pepperoni and mushrooms. -/
theorem pizza_toppings (total : ℕ) (pepperoni : ℕ) (mushrooms : ℕ) 
  (h_total : total = 24)
  (h_pepperoni : pepperoni = 15)
  (h_mushrooms : mushrooms = 14) :
  pepperoni + mushrooms - total = 5 := by
  sorry

end pizza_toppings_l2302_230292


namespace triangle_cut_theorem_l2302_230280

theorem triangle_cut_theorem : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → 
    ((9 - y : ℤ) + (12 - y) ≤ (20 - y) ∧
     (9 - y : ℤ) + (20 - y) ≤ (12 - y) ∧
     (12 - y : ℤ) + (20 - y) ≤ (9 - y)) → 
    y ≥ x) ∧
  (9 - x : ℤ) + (12 - x) ≤ (20 - x) ∧
  (9 - x : ℤ) + (20 - x) ≤ (12 - x) ∧
  (12 - x : ℤ) + (20 - x) ≤ (9 - x) ∧
  x = 17 :=
by sorry

end triangle_cut_theorem_l2302_230280


namespace train_length_l2302_230299

theorem train_length (crossing_time : ℝ) (speed_kmh : ℝ) : 
  crossing_time = 20 → speed_kmh = 36 → 
  (speed_kmh * 1000 / 3600) * crossing_time = 200 := by
  sorry

end train_length_l2302_230299


namespace max_crayfish_revenue_l2302_230234

/-- The revenue function for selling crayfish -/
def revenue (total : ℕ) (sold : ℕ) : ℝ :=
  (total - sold : ℝ) * ((total - sold : ℝ) - 4.5) * sold

/-- The statement that proves the maximum revenue and number of crayfish sold -/
theorem max_crayfish_revenue :
  let total := 32
  ∃ (max_sold : ℕ) (max_revenue : ℝ),
    max_sold = 14 ∧
    max_revenue = 189 ∧
    ∀ (sold : ℕ), sold ≤ total → revenue total sold ≤ max_revenue :=
by sorry

end max_crayfish_revenue_l2302_230234


namespace exists_four_digit_sum_21_div_14_l2302_230290

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem exists_four_digit_sum_21_div_14 : 
  ∃ (n : ℕ), is_four_digit n ∧ digit_sum n = 21 ∧ n % 14 = 0 := by
  sorry

end exists_four_digit_sum_21_div_14_l2302_230290


namespace carpet_rearrangement_l2302_230273

/-- Represents a piece of carpet with a given length -/
structure CarpetPiece where
  length : ℝ
  length_pos : length > 0

/-- Represents a corridor covered by carpet pieces -/
structure CarpetedCorridor where
  length : ℝ
  length_pos : length > 0
  pieces : List CarpetPiece
  covers_corridor : (pieces.map CarpetPiece.length).sum ≥ length

theorem carpet_rearrangement (corridor : CarpetedCorridor) :
  ∃ (subset : List CarpetPiece), subset ⊆ corridor.pieces ∧
    (subset.map CarpetPiece.length).sum ≥ corridor.length ∧
    (subset.map CarpetPiece.length).sum < 2 * corridor.length :=
by sorry

end carpet_rearrangement_l2302_230273


namespace not_equal_to_seven_thirds_l2302_230216

theorem not_equal_to_seven_thirds : ∃ x, x ≠ 7/3 ∧ 
  (x = 3 + 1/9) ∧ 
  (14/6 = 7/3) ∧ 
  (2 + 1/3 = 7/3) ∧ 
  (2 + 4/12 = 7/3) := by
  sorry

end not_equal_to_seven_thirds_l2302_230216


namespace parallel_vectors_x_values_l2302_230208

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_values :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (x - 1, 2)
  are_parallel a b → x = -1 ∨ x = 2 := by
sorry

end parallel_vectors_x_values_l2302_230208


namespace tangent_line_sin_at_pi_l2302_230279

theorem tangent_line_sin_at_pi (x y : ℝ) :
  let f : ℝ → ℝ := λ t => Real.sin t
  let f' : ℝ → ℝ := λ t => Real.cos t
  let tangent_point : ℝ × ℝ := (π, 0)
  let slope : ℝ := f' tangent_point.1
  x + y - π = 0 ↔ y - tangent_point.2 = slope * (x - tangent_point.1) :=
by sorry

end tangent_line_sin_at_pi_l2302_230279


namespace remainder_n_cubed_plus_three_l2302_230287

theorem remainder_n_cubed_plus_three (n : ℕ) (h : n > 2) :
  (n^3 + 3) % (n + 1) = 2 := by
  sorry

end remainder_n_cubed_plus_three_l2302_230287


namespace billys_restaurant_bill_l2302_230259

/-- Calculates the total bill for a group at Billy's Restaurant -/
theorem billys_restaurant_bill (adults children meal_cost : ℕ) : 
  adults = 2 → children = 5 → meal_cost = 3 → 
  (adults + children) * meal_cost = 21 := by
sorry

end billys_restaurant_bill_l2302_230259


namespace initial_typists_count_l2302_230250

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 20

/-- The number of letters typed by the initial group in 20 minutes -/
def letters_20min : ℕ := 60

/-- The number of typists in the second group -/
def second_group_typists : ℕ := 30

/-- The number of letters typed by the second group in 1 hour -/
def letters_1hour : ℕ := 270

/-- The time ratio between 1 hour and 20 minutes -/
def time_ratio : ℚ := 3

theorem initial_typists_count :
  initial_typists * second_group_typists * letters_20min * time_ratio = letters_1hour * initial_typists :=
by sorry

end initial_typists_count_l2302_230250


namespace melissa_oranges_l2302_230266

theorem melissa_oranges (initial_oranges : ℕ) (taken_oranges : ℕ) 
  (h1 : initial_oranges = 70) (h2 : taken_oranges = 19) : 
  initial_oranges - taken_oranges = 51 := by
  sorry

end melissa_oranges_l2302_230266


namespace new_average_weight_l2302_230256

/-- Given a group of students and a new student joining, calculate the new average weight -/
theorem new_average_weight 
  (initial_count : ℕ) 
  (initial_average : ℝ) 
  (new_student_weight : ℝ) : 
  initial_count = 29 → 
  initial_average = 28 → 
  new_student_weight = 7 → 
  let total_weight := initial_count * initial_average + new_student_weight
  let new_count := initial_count + 1
  (total_weight / new_count : ℝ) = 27.3 := by
  sorry

end new_average_weight_l2302_230256


namespace unique_function_property_l2302_230224

theorem unique_function_property (f : ℚ → ℚ) :
  (f 1 = 2) →
  (∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) →
  (∀ x : ℚ, f x = x + 1) :=
by sorry

end unique_function_property_l2302_230224


namespace library_tables_count_l2302_230215

/-- The number of pupils that can be seated at a rectangular table -/
def rectangular_table_capacity : ℕ := 10

/-- The number of pupils that can be seated at a square table -/
def square_table_capacity : ℕ := 4

/-- The number of square tables needed in the library -/
def square_tables_needed : ℕ := 5

/-- The total number of pupils that need to be seated -/
def total_pupils : ℕ := 90

/-- The number of rectangular tables in the library -/
def rectangular_tables : ℕ := 7

theorem library_tables_count :
  rectangular_tables * rectangular_table_capacity +
  square_tables_needed * square_table_capacity = total_pupils :=
by sorry

end library_tables_count_l2302_230215


namespace polynomial_equality_implies_specific_a_l2302_230219

theorem polynomial_equality_implies_specific_a (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 15) + 4 = (x + b) * (x + c)) →
  (a = 10 ∨ a = 25) := by
  sorry

end polynomial_equality_implies_specific_a_l2302_230219


namespace arrangement_count_l2302_230239

/-- The number of distinct arrangements of 8 indistinguishable items and 2 other indistinguishable items in a row of 10 slots -/
def distinct_arrangements : ℕ := 45

/-- The total number of slots available -/
def total_slots : ℕ := 10

/-- The number of the first type of indistinguishable items -/
def first_item_count : ℕ := 8

/-- The number of the second type of indistinguishable items -/
def second_item_count : ℕ := 2

theorem arrangement_count :
  distinct_arrangements = (total_slots.choose second_item_count) :=
by sorry

end arrangement_count_l2302_230239


namespace third_number_is_one_l2302_230241

/-- Define a sequence where each segment starts with 1 and counts up by one more number than the previous segment -/
def special_sequence : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => 
  let segment := n / 2 + 1
  let position := n % (segment + 1)
  if position = 0 then 1 else position + 1

/-- The third number in the special sequence is 1 -/
theorem third_number_is_one : special_sequence 2 = 1 := by
  sorry

end third_number_is_one_l2302_230241


namespace arithmetic_sequence_sum_l2302_230236

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a, if a₁ + a₂ + a₃ = 32 and a₁₁ + a₁₂ + a₁₃ = 118, then a₄ + a₁₀ = 50. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_sum1 : a 1 + a 2 + a 3 = 32) 
    (h_sum2 : a 11 + a 12 + a 13 = 118) : 
  a 4 + a 10 = 50 := by
  sorry


end arithmetic_sequence_sum_l2302_230236


namespace period_2_gym_class_size_l2302_230262

theorem period_2_gym_class_size : ℕ → Prop :=
  fun x => (2 * x - 5 = 11) → x = 8

#check period_2_gym_class_size

end period_2_gym_class_size_l2302_230262


namespace gcd_9009_13860_l2302_230230

theorem gcd_9009_13860 : Nat.gcd 9009 13860 = 1 := by
  sorry

end gcd_9009_13860_l2302_230230


namespace devin_teaching_years_difference_l2302_230288

/-- Proves that Devin has been teaching for 5 years less than half of Tom's teaching years -/
theorem devin_teaching_years_difference (total_years devin_years tom_years : ℕ) : 
  total_years = 70 → tom_years = 50 → devin_years = total_years - tom_years → 
  tom_years / 2 - devin_years = 5 := by
  sorry

end devin_teaching_years_difference_l2302_230288


namespace consecutive_integers_cube_sum_l2302_230229

theorem consecutive_integers_cube_sum : 
  ∀ n : ℕ, 
    n > 2 → 
    (n - 2) * (n - 1) * n = 15 * (3 * n - 3) → 
    (n - 2)^3 + (n - 1)^3 + n^3 = 216 := by
  sorry

end consecutive_integers_cube_sum_l2302_230229


namespace odd_times_abs_even_is_odd_l2302_230264

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function g is even if g(-x) = g(x) for all x in its domain -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

/-- The product of an odd function and the absolute value of an even function is odd -/
theorem odd_times_abs_even_is_odd (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  IsOdd (fun x ↦ f x * |g x|) := by sorry

end odd_times_abs_even_is_odd_l2302_230264


namespace proposition_analysis_l2302_230222

theorem proposition_analysis :
  (¬(∀ x y a : ℝ, 1 < y ∧ y < x ∧ 0 < a ∧ a < 1 → a^(1/x) < a^(1/y))) ∧
  (∀ x y a : ℝ, 1 < y ∧ y < x ∧ a < 0 → x^a < y^a) ∧
  ((¬(∀ x y a : ℝ, 1 < y ∧ y < x ∧ 0 < a ∧ a < 1 → a^(1/x) < a^(1/y))) ∨
   (∀ x y a : ℝ, 1 < y ∧ y < x ∧ a < 0 → x^a < y^a)) ∧
  (¬(∀ x y a : ℝ, 1 < y ∧ y < x ∧ 0 < a ∧ a < 1 → a^(1/x) < a^(1/y))) :=
by sorry

end proposition_analysis_l2302_230222


namespace set_b_forms_triangle_l2302_230263

/-- Triangle inequality theorem: A set of three line segments can form a triangle if and only if
    the sum of the lengths of any two sides is greater than the length of the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that line segments of lengths 8, 6, and 4 can form a triangle. -/
theorem set_b_forms_triangle : can_form_triangle 8 6 4 := by
  sorry

end set_b_forms_triangle_l2302_230263


namespace arctan_sum_roots_cubic_l2302_230231

theorem arctan_sum_roots_cubic (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 10*x₁ + 11 = 0 → 
  x₂^3 - 10*x₂ + 11 = 0 → 
  x₃^3 - 10*x₃ + 11 = 0 → 
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = π/4 := by
  sorry

end arctan_sum_roots_cubic_l2302_230231


namespace rationalize_denominator_l2302_230251

theorem rationalize_denominator : 35 / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end rationalize_denominator_l2302_230251


namespace total_revenue_equals_4452_4_l2302_230255

def calculate_revenue (price : ℝ) (quantity : ℕ) (discount : ℝ) (tax : ℝ) (surcharge : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  let final_price := taxed_price * (1 + surcharge)
  final_price * quantity

def total_revenue : ℝ :=
  calculate_revenue 25 60 0.1 0.05 0 +
  calculate_revenue 25 10 0 0 0.03 +
  calculate_revenue 25 20 0.05 0.02 0 +
  calculate_revenue 25 44 0.15 0 0.04 +
  calculate_revenue 25 66 0.2 0 0

theorem total_revenue_equals_4452_4 :
  total_revenue = 4452.4 := by
  sorry

end total_revenue_equals_4452_4_l2302_230255


namespace square_root_problem_l2302_230201

theorem square_root_problem (a b : ℝ) 
  (h1 : Real.sqrt (a + 3) = 3) 
  (h2 : (3 * b - 2) ^ (1/3 : ℝ) = 2) : 
  Real.sqrt (a + 3*b) = 6 := by
sorry

end square_root_problem_l2302_230201


namespace hyperbola_and_angle_bisector_l2302_230232

-- Define the hyperbola Γ
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the line l
def line (x y : ℝ) : Prop :=
  x + y - 2 = 0

-- Define that l is parallel to one of the asymptotes and passes through a focus
def line_properties (a b : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), line x₀ y₀ ∧ 
  ((x₀ = a ∧ y₀ = 0) ∨ (x₀ = -a ∧ y₀ = 0)) ∧
  (∀ x y : ℝ, line x y → y = x ∨ y = -x)

-- Main theorem
theorem hyperbola_and_angle_bisector 
  (a b : ℝ) 
  (h : line_properties a b) :
  (∀ x y : ℝ, hyperbola a b x y ↔ x^2 - y^2 = 2) ∧
  (∃ P : ℝ × ℝ, 
    hyperbola a b P.1 P.2 ∧ 
    line P.1 P.2 ∧
    ∀ x y : ℝ, 3*x - y - 4 = 0 ↔ 
      (∃ t : ℝ, x = t*P.1 + (1-t)*(-2) ∧ y = t*P.2) ∨
      (∃ t : ℝ, x = t*P.1 + (1-t)*2 ∧ y = t*P.2)) :=
by sorry

end hyperbola_and_angle_bisector_l2302_230232


namespace f_monotone_and_max_a_l2302_230293

noncomputable def f (a x : ℝ) : ℝ := (x - a - 1) * Real.exp (x - 1) - (1/2) * x^2 + a * x

theorem f_monotone_and_max_a :
  (∀ x y : ℝ, x < y → f 1 x < f 1 y) ∧
  (∃ a : ℝ, a = Real.exp 1 / 2 - 1 ∧
    (∀ b : ℝ, (∃ x : ℝ, x > 0 ∧ f b x = -1/2) →
      (∀ y : ℝ, y > 0 → f b y ≥ -1/2) →
      b ≤ a)) :=
by sorry

end f_monotone_and_max_a_l2302_230293


namespace square_sum_identity_l2302_230237

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end square_sum_identity_l2302_230237


namespace sum_of_real_and_imag_parts_l2302_230214

theorem sum_of_real_and_imag_parts : ∃ (z : ℂ), z = (Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I)) ∧ z.re + z.im = 3/2 := by
  sorry

end sum_of_real_and_imag_parts_l2302_230214


namespace triangle_line_equation_l2302_230210

/-- A line passing through a point and forming a triangle with coordinate axes -/
structure TriangleLine where
  -- Coefficients of the line equation ax + by = c
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through the point (-2, 2)
  passes_through_point : a * (-2) + b * 2 = c
  -- The line forms a triangle with area 1
  triangle_area : |a * b| / 2 = 1

/-- The equation of the line is either x + 2y - 2 = 0 or 2x + y + 2 = 0 -/
theorem triangle_line_equation (l : TriangleLine) : 
  (l.a = 1 ∧ l.b = 2 ∧ l.c = 2) ∨ (l.a = 2 ∧ l.b = 1 ∧ l.c = -2) :=
sorry

end triangle_line_equation_l2302_230210


namespace integer_solutions_system_l2302_230233

theorem integer_solutions_system :
  ∀ x y z : ℤ,
  (x^2 - y^2 = z ∧ 3*x*y + (x-y)*z = z^2) →
  ((x = 2 ∧ y = 1 ∧ z = 3) ∨
   (x = 1 ∧ y = 2 ∧ z = -3) ∨
   (x = 1 ∧ y = 0 ∧ z = 1) ∨
   (x = 0 ∧ y = 1 ∧ z = -1) ∨
   (x = 0 ∧ y = 0 ∧ z = 0)) :=
by sorry

end integer_solutions_system_l2302_230233


namespace radical_product_equals_64_l2302_230203

theorem radical_product_equals_64 : 
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 64 := by
  sorry

end radical_product_equals_64_l2302_230203


namespace parallel_vectors_implies_m_eq_neg_one_l2302_230291

/-- Two 2D vectors are parallel if the cross product of their components is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_implies_m_eq_neg_one (m : ℝ) :
  let a : ℝ × ℝ := (m, -1)
  let b : ℝ × ℝ := (1, m + 2)
  parallel a b → m = -1 := by
sorry

end parallel_vectors_implies_m_eq_neg_one_l2302_230291


namespace equation_solution_l2302_230238

theorem equation_solution (x y z : ℝ) 
  (eq1 : 4*x - 5*y - z = 0)
  (eq2 : x + 5*y - 18*z = 0)
  (h : z ≠ 0) :
  (x^2 + 4*x*y) / (y^2 + z^2) = 3622 / 9256 := by
  sorry

end equation_solution_l2302_230238


namespace hyperbolas_same_asymptotes_l2302_230298

/-- Given two hyperbolas with equations (x²/9) - (y²/16) = 1 and (y²/25) - ((x-4)²/M) = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - (x - 4)^2 / M = 1) →
  (∀ x y, y = (4/3) * x ↔ y = (5/Real.sqrt M) * (x - 4)) →
  M = 225 / 16 := by
  sorry

end hyperbolas_same_asymptotes_l2302_230298


namespace ellipse_foci_distance_l2302_230275

/-- An ellipse with axes parallel to the coordinate axes, tangent to the x-axis at (6, 0) and tangent to the y-axis at (0, 2) -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ -- semi-major axis
  b : ℝ -- semi-minor axis
  h_tangent_x : center.1 - a = 6
  h_tangent_y : center.2 - b = 2
  h_a_gt_b : a > b
  h_positive : a > 0 ∧ b > 0

/-- The distance between the foci of the ellipse is 8√2 -/
theorem ellipse_foci_distance (e : Ellipse) : Real.sqrt 128 = 8 * Real.sqrt 2 := by
  sorry

end ellipse_foci_distance_l2302_230275


namespace number_properties_l2302_230257

theorem number_properties :
  (∃! x : ℝ, -x = x) ∧
  (∀ x : ℝ, x ≠ 0 → (1 / x = x ↔ x = 1 ∨ x = -1)) ∧
  (∀ x : ℝ, x < -1 → 1 / x > x) ∧
  (∀ y : ℝ, y > 1 → 1 / y < y) ∧
  (∃ n : ℕ, ∀ m : ℕ, n ≤ m) :=
by sorry

end number_properties_l2302_230257


namespace positive_intervals_l2302_230254

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 2)

theorem positive_intervals (x : ℝ) : 
  f x > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 2 :=
sorry

end positive_intervals_l2302_230254


namespace sandwich_fraction_proof_l2302_230240

theorem sandwich_fraction_proof (total : ℚ) (ticket : ℚ) (book : ℚ) (leftover : ℚ) 
  (h_total : total = 150)
  (h_ticket : ticket = 1 / 6)
  (h_book : book = 1 / 2)
  (h_leftover : leftover = 20)
  (h_spent : total - leftover = ticket * total + book * total + (total - leftover - ticket * total - book * total)) :
  (total - leftover - ticket * total - book * total) / total = 1 / 5 := by
sorry

end sandwich_fraction_proof_l2302_230240


namespace banana_problem_solution_l2302_230218

/-- Represents the banana purchase and sale problem --/
def banana_problem (purchase_pounds : ℚ) (purchase_price : ℚ) 
                   (sale_pounds : ℚ) (sale_price : ℚ) 
                   (profit : ℚ) (total_pounds : ℚ) : Prop :=
  -- Cost price per pound
  let cp_per_pound := purchase_price / purchase_pounds
  -- Selling price per pound
  let sp_per_pound := sale_price / sale_pounds
  -- Total cost
  let total_cost := total_pounds * cp_per_pound
  -- Total revenue
  let total_revenue := total_pounds * sp_per_pound
  -- Profit calculation
  (total_revenue - total_cost = profit) ∧
  -- Ensure the total pounds is positive
  (total_pounds > 0)

/-- Theorem stating the solution to the banana problem --/
theorem banana_problem_solution :
  banana_problem 3 0.5 4 1 6 432 := by
  sorry

end banana_problem_solution_l2302_230218


namespace base_10_648_equals_base_7_1614_l2302_230268

/-- Converts a base-10 integer to its representation in base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to its decimal representation --/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldr (λ d acc => d + 7 * acc) 0

theorem base_10_648_equals_base_7_1614 :
  fromBase7 [4, 1, 6, 1] = 648 :=
by sorry

end base_10_648_equals_base_7_1614_l2302_230268


namespace quadratic_roots_property_l2302_230223

theorem quadratic_roots_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  ∃ (x₁ x₂ : ℝ), 
    (a * x₁^2 + b * x₁ + c = 0) ∧ 
    (a * x₂^2 + b * x₂ + c = 0) ∧ 
    (x₁ > 0) ∧ 
    (x₂ < 0) ∧ 
    (|x₂| > |x₁|) := by
  sorry

end quadratic_roots_property_l2302_230223


namespace circle_center_and_radius_l2302_230225

/-- Given a circle with equation x^2 + y^2 - 2x + 6y + 9 = 0, prove that its center is at (1, -3) and its radius is 1 -/
theorem circle_center_and_radius :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 2*x + 6*y + 9 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧ radius = 1 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l2302_230225


namespace intuitive_diagram_area_l2302_230209

/-- The area of the intuitive diagram of a square in oblique axonometric drawing -/
theorem intuitive_diagram_area (a : ℝ) (h : a > 0) :
  let planar_area := a^2
  let ratio := 2 * Real.sqrt 2
  let intuitive_area := planar_area / ratio
  intuitive_area = (Real.sqrt 2 / 4) * a^2 := by
  sorry

end intuitive_diagram_area_l2302_230209


namespace square_difference_from_sum_and_product_l2302_230283

theorem square_difference_from_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
  sorry

end square_difference_from_sum_and_product_l2302_230283


namespace number_multiplied_by_three_twice_l2302_230278

theorem number_multiplied_by_three_twice (x : ℝ) : (3 * (3 * x) = 18) → x = 2 := by
  sorry

end number_multiplied_by_three_twice_l2302_230278


namespace bicycle_inventory_solution_l2302_230247

/-- Represents the bicycle inventory changes in Hank's store over three days -/
def bicycle_inventory_problem (initial_stock : ℕ) (saturday_bought : ℕ) : Prop :=
  let friday_change : ℤ := 15 - 10
  let saturday_change : ℤ := saturday_bought - 12
  let sunday_change : ℤ := 11 - 9
  (friday_change + saturday_change + sunday_change : ℤ) = 3

/-- The solution to the bicycle inventory problem -/
theorem bicycle_inventory_solution :
  ∃ (initial_stock : ℕ), bicycle_inventory_problem initial_stock 8 :=
sorry

end bicycle_inventory_solution_l2302_230247


namespace stratified_sample_medium_supermarkets_l2302_230243

/-- Given a population of supermarkets with the following properties:
  * total_supermarkets: The total number of supermarkets
  * medium_supermarkets: The number of medium-sized supermarkets
  * sample_size: The size of the stratified sample to be taken
  
  This theorem proves that the number of medium-sized supermarkets
  to be selected in the sample is equal to the expected value. -/
theorem stratified_sample_medium_supermarkets
  (total_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (sample_size : ℕ)
  (h1 : total_supermarkets = 2000)
  (h2 : medium_supermarkets = 400)
  (h3 : sample_size = 100)
  : (medium_supermarkets * sample_size) / total_supermarkets = 20 := by
  sorry

#check stratified_sample_medium_supermarkets

end stratified_sample_medium_supermarkets_l2302_230243


namespace interest_problem_solution_l2302_230211

/-- Given conditions for the interest problem -/
structure InterestProblem where
  P : ℝ  -- Principal amount
  r : ℝ  -- Interest rate (as a decimal)
  t : ℝ  -- Time period in years
  diff : ℝ  -- Difference between compound and simple interest

/-- Theorem statement for the interest problem -/
theorem interest_problem_solution (prob : InterestProblem) 
  (h1 : prob.r = 0.1)  -- 10% interest rate
  (h2 : prob.t = 2)  -- 2 years time period
  (h3 : prob.diff = 631)  -- Difference between compound and simple interest is $631
  : prob.P = 63100 := by
  sorry

end interest_problem_solution_l2302_230211


namespace sequence_fifth_term_l2302_230226

/-- Given a positive sequence {a_n}, prove that a_5 = 3 -/
theorem sequence_fifth_term (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_1 : a 1 = 1)
  (h_2 : a 2 = Real.sqrt 3)
  (h_rec : ∀ n ≥ 2, 2 * (a n)^2 = (a (n+1))^2 + (a (n-1))^2) :
  a 5 = 3 := by
sorry

end sequence_fifth_term_l2302_230226


namespace apartment_length_l2302_230221

/-- Proves that the length of an apartment with given specifications is 16 feet -/
theorem apartment_length : 
  ∀ (width : ℝ) (total_rooms : ℕ) (living_room_size : ℝ),
    width = 10 →
    total_rooms = 6 →
    living_room_size = 60 →
    ∃ (room_size : ℝ),
      room_size = living_room_size / 3 ∧
      width * 16 = living_room_size + (total_rooms - 1) * room_size :=
by sorry

end apartment_length_l2302_230221


namespace quadratic_equation_solution_l2302_230244

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 - 4*x - 5 = 0} = {-1, 5} := by sorry

end quadratic_equation_solution_l2302_230244


namespace max_a_for_monotone_cubic_l2302_230252

/-- Given a > 0 and f(x) = x^3 - ax is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotone_cubic (a : ℝ) (h1 : a > 0) :
  (∀ x ≥ 1, Monotone (fun x => x^3 - a*x)) →
  a ≤ 3 ∧ ∀ ε > 0, ∃ x ≥ 1, ¬Monotone (fun x => x^3 - (3 + ε)*x) := by
  sorry

end max_a_for_monotone_cubic_l2302_230252


namespace triangle_cosine_proof_l2302_230253

/-- Given a triangle ABC with A = 2B, a = 6, and b = 4, prove that cos B = 3/4 -/
theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) : 
  A = 2 * B → 
  a = 6 → 
  b = 4 → 
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  Real.cos B = 3 / 4 := by
sorry

end triangle_cosine_proof_l2302_230253


namespace square_area_from_diagonal_l2302_230248

/-- Given a square with diagonal length 14√2 cm, its area is 196 cm² -/
theorem square_area_from_diagonal : ∀ s : ℝ,
  s > 0 →
  s * s * 2 = (14 * Real.sqrt 2) ^ 2 →
  s * s = 196 := by
  sorry

end square_area_from_diagonal_l2302_230248


namespace repeating_decimal_subtraction_l2302_230295

theorem repeating_decimal_subtraction : 
  ∃ (a b c : ℚ), 
    (∀ n : ℕ, a = (234 / 10^3 + 234 / (10^3 * (1000^n - 1)))) ∧
    (∀ n : ℕ, b = (567 / 10^3 + 567 / (10^3 * (1000^n - 1)))) ∧
    (∀ n : ℕ, c = (891 / 10^3 + 891 / (10^3 * (1000^n - 1)))) ∧
    a - b - c = -408 / 333 := by
  sorry

end repeating_decimal_subtraction_l2302_230295


namespace triangle_angle_c_l2302_230217

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_c (t : Triangle) :
  t.a = 1 ∧ t.A = Real.pi / 3 ∧ t.c = Real.sqrt 3 / 3 → t.C = Real.pi / 6 := by
  sorry

end triangle_angle_c_l2302_230217


namespace larger_tv_diagonal_l2302_230267

theorem larger_tv_diagonal (d : ℝ) : 
  d > 0 → 
  (d / Real.sqrt 2) ^ 2 = (25 / Real.sqrt 2) ^ 2 + 79.5 → 
  d = 28 :=
by
  sorry

end larger_tv_diagonal_l2302_230267


namespace percentage_of_men_employees_l2302_230269

theorem percentage_of_men_employees (men_attendance : ℝ) (women_attendance : ℝ) (total_attendance : ℝ) :
  men_attendance = 0.2 →
  women_attendance = 0.4 →
  total_attendance = 0.34 →
  ∃ (men_percentage : ℝ),
    men_percentage + (1 - men_percentage) = 1 ∧
    men_attendance * men_percentage + women_attendance * (1 - men_percentage) = total_attendance ∧
    men_percentage = 0.3 := by
  sorry

end percentage_of_men_employees_l2302_230269


namespace certain_value_proof_l2302_230286

theorem certain_value_proof (n : ℝ) (v : ℝ) (h1 : n = 45) (h2 : (1/3) * n - v = 10) : v = 5 := by
  sorry

end certain_value_proof_l2302_230286


namespace decimal_difference_l2302_230296

theorem decimal_difference : (8.1 : ℝ) - (8.01 : ℝ) ≠ 0.1 := by sorry

end decimal_difference_l2302_230296


namespace parallel_line_equation_l2302_230261

/-- The equation of a line passing through (3, 2) and parallel to 4x + y - 2 = 0 -/
theorem parallel_line_equation : 
  ∀ (x y : ℝ), 
  (∃ (m b : ℝ), 
    -- The line passes through (3, 2)
    2 = m * 3 + b ∧ 
    -- The line is parallel to 4x + y - 2 = 0
    m = -4 ∧ 
    -- The equation of the line
    y = m * x + b) 
  ↔ 
  -- The resulting equation
  4 * x + y - 14 = 0 := by sorry

end parallel_line_equation_l2302_230261


namespace complex_number_in_fourth_quadrant_l2302_230205

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - I) / (2 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l2302_230205


namespace cube_root_of_product_powers_l2302_230246

theorem cube_root_of_product_powers (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (m^4 * n^4)^(1/3) = (m * n)^(4/3) := by sorry

end cube_root_of_product_powers_l2302_230246


namespace absolute_value_inequality_l2302_230297

theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, |x - 4| - |x + 5| ≤ m) → m ≥ 9 := by
  sorry

end absolute_value_inequality_l2302_230297


namespace joy_tonight_outcomes_l2302_230270

/-- The number of letters in mailbox A -/
def mailbox_A : Nat := 30

/-- The number of letters in mailbox B -/
def mailbox_B : Nat := 20

/-- The total number of different outcomes for selecting a lucky star and two lucky partners -/
def total_outcomes : Nat := mailbox_A * (mailbox_A - 1) * mailbox_B + mailbox_B * (mailbox_B - 1) * mailbox_A

theorem joy_tonight_outcomes : total_outcomes = 28800 := by
  sorry

end joy_tonight_outcomes_l2302_230270


namespace correct_calculation_l2302_230282

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * x^2 * y = x^2 * y := by
  sorry

end correct_calculation_l2302_230282


namespace log_x2y2_value_l2302_230284

-- Define the logarithm function (assuming it's the natural logarithm)
noncomputable def log : ℝ → ℝ := Real.log

-- Define the main theorem
theorem log_x2y2_value (x y : ℝ) (h1 : log (x * y^4) = 1) (h2 : log (x^3 * y) = 1) :
  log (x^2 * y^2) = 10/11 := by
  sorry

end log_x2y2_value_l2302_230284


namespace weight_gain_ratio_l2302_230276

/-- The weight gain problem at the family reunion -/
theorem weight_gain_ratio (jose_gain orlando_gain fernando_gain : ℚ) : 
  orlando_gain = 5 →
  fernando_gain = jose_gain / 2 - 3 →
  jose_gain + orlando_gain + fernando_gain = 20 →
  jose_gain / orlando_gain = 12 / 5 := by
sorry

end weight_gain_ratio_l2302_230276


namespace m_range_l2302_230227

def f (x : ℝ) : ℝ := x^5 + x^3

theorem m_range (m : ℝ) (h1 : m ∈ Set.Icc (-2 : ℝ) 2) 
  (h2 : (m - 1) ∈ Set.Icc (-2 : ℝ) 2) (h3 : f m + f (m - 1) > 0) : 
  m ∈ Set.Ioo (1/2 : ℝ) 2 := by
sorry

end m_range_l2302_230227


namespace max_square_side_length_is_correct_l2302_230212

/-- The width of the blackboard in centimeters. -/
def blackboardWidth : ℕ := 120

/-- The length of the blackboard in centimeters. -/
def blackboardLength : ℕ := 96

/-- The maximum side length of a square picture that can fit on the blackboard without remainder. -/
def maxSquareSideLength : ℕ := 24

/-- Theorem stating that the maximum side length of a square that can fit both the width and length of the blackboard without remainder is 24 cm. -/
theorem max_square_side_length_is_correct :
  maxSquareSideLength = Nat.gcd blackboardWidth blackboardLength ∧
  blackboardWidth % maxSquareSideLength = 0 ∧
  blackboardLength % maxSquareSideLength = 0 ∧
  ∀ n : ℕ, n > maxSquareSideLength →
    (blackboardWidth % n ≠ 0 ∨ blackboardLength % n ≠ 0) :=
by sorry

end max_square_side_length_is_correct_l2302_230212


namespace right_triangle_area_l2302_230281

theorem right_triangle_area (p q r : ℝ) : 
  p > 0 → q > 0 → r > 0 →
  p + q + r = 16 →
  p^2 + q^2 + r^2 = 98 →
  p^2 + q^2 = r^2 →
  (1/2) * p * q = 8 :=
by sorry

end right_triangle_area_l2302_230281
