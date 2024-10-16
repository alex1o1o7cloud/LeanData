import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3924_392470

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3924_392470


namespace NUMINAMATH_CALUDE_water_remaining_after_required_pourings_l3924_392416

/-- Represents the fraction of water remaining after n pourings -/
def waterRemaining (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- The number of pourings required to reach exactly one-fifth of the original water -/
def requiredPourings : ℕ := 8

theorem water_remaining_after_required_pourings :
  waterRemaining requiredPourings = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_water_remaining_after_required_pourings_l3924_392416


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3924_392410

theorem quadratic_roots_sum (a b : ℝ) : 
  (∃ x : ℝ, x^2 + x - 2023 = 0) → 
  (a^2 + a - 2023 = 0) → 
  (b^2 + b - 2023 = 0) → 
  (a ≠ b) →
  a^2 + 2*a + b = 2022 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3924_392410


namespace NUMINAMATH_CALUDE_circle_center_correct_l3924_392479

/-- The center of a circle given by the equation x^2 + y^2 - 2x + 4y = 0 --/
def circle_center : ℝ × ℝ := sorry

/-- The equation of the circle --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y = 0

theorem circle_center_correct :
  let (h, k) := circle_center
  ∀ x y, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 1) ∧ 
  h = 1 ∧ k = -2 := by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l3924_392479


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3924_392439

theorem election_votes_theorem (candidate1_percent : ℝ) (candidate2_percent : ℝ) 
  (candidate3_percent : ℝ) (candidate4_percent : ℝ) (candidate4_votes : ℕ) :
  candidate1_percent = 42 →
  candidate2_percent = 30 →
  candidate3_percent = 20 →
  candidate4_percent = 8 →
  candidate4_votes = 720 →
  (candidate1_percent + candidate2_percent + candidate3_percent + candidate4_percent = 100) →
  ∃ (total_votes : ℕ), total_votes = 9000 ∧ 
    (candidate4_percent / 100 * total_votes : ℝ) = candidate4_votes :=
by sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3924_392439


namespace NUMINAMATH_CALUDE_tournament_prize_orderings_l3924_392481

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := 5

/-- Represents the number of possible outcomes for each game -/
def outcomes_per_game : ℕ := 2

/-- Theorem stating the number of possible prize orderings in the tournament -/
theorem tournament_prize_orderings :
  (outcomes_per_game ^ num_games : ℕ) = 32 := by sorry

end NUMINAMATH_CALUDE_tournament_prize_orderings_l3924_392481


namespace NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l3924_392447

/-- A fair die with sides numbered from 1 to 6 -/
def Die : Type := Fin 6

/-- The set of possible outcomes when rolling two dice -/
def TwoRolls : Type := Die × Die

/-- Function to check if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The sum of two dice rolls -/
def rollSum (roll : TwoRolls) : ℕ := sorry

/-- The set of all possible outcomes when rolling two dice -/
def allOutcomes : Finset TwoRolls := sorry

/-- The set of outcomes where the sum is prime -/
def primeOutcomes : Finset TwoRolls := sorry

/-- Theorem: The probability of rolling a prime sum with two fair dice is 5/12 -/
theorem probability_prime_sum_two_dice : 
  (Finset.card primeOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l3924_392447


namespace NUMINAMATH_CALUDE_tetrahedron_subdivision_existence_l3924_392411

theorem tetrahedron_subdivision_existence : ∃ k : ℕ, (1 / 2 : ℝ) ^ k < (1 / 100 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_subdivision_existence_l3924_392411


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3924_392450

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ x : ℂ, (3 - 2 * i * x = 6 + i * x) ∧ (x = i) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3924_392450


namespace NUMINAMATH_CALUDE_number_times_one_fourth_squared_l3924_392491

theorem number_times_one_fourth_squared (x : ℝ) : x * (1/4)^2 = 4^3 → x = 1024 := by
  sorry

end NUMINAMATH_CALUDE_number_times_one_fourth_squared_l3924_392491


namespace NUMINAMATH_CALUDE_three_fourths_of_45_l3924_392430

theorem three_fourths_of_45 : (3 : ℚ) / 4 * 45 = 33 + 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_of_45_l3924_392430


namespace NUMINAMATH_CALUDE_area_third_polygon_l3924_392453

/-- Given three regular polygons inscribed in a circle, where each subsequent polygon
    has twice as many sides as the previous one, the area of the third polygon can be
    expressed in terms of the areas of the first two polygons. -/
theorem area_third_polygon (S₁ S₂ : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) :
  ∃ (S : ℝ), S = Real.sqrt (2 * S₂^3 / (S₁ + S₂)) :=
sorry

end NUMINAMATH_CALUDE_area_third_polygon_l3924_392453


namespace NUMINAMATH_CALUDE_student_count_l3924_392458

theorem student_count : ℕ :=
  let avg_age : ℕ := 20
  let group1_count : ℕ := 5
  let group1_avg : ℕ := 14
  let group2_count : ℕ := 9
  let group2_avg : ℕ := 16
  let last_student_age : ℕ := 186
  let total_students : ℕ := group1_count + group2_count + 1
  let total_age : ℕ := group1_count * group1_avg + group2_count * group2_avg + last_student_age
  have h1 : avg_age * total_students = total_age := by sorry
  20

end NUMINAMATH_CALUDE_student_count_l3924_392458


namespace NUMINAMATH_CALUDE_power_of_three_difference_l3924_392473

theorem power_of_three_difference : 3^(1+3+4) - (3^1 + 3^3 + 3^4) = 6450 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_difference_l3924_392473


namespace NUMINAMATH_CALUDE_function_value_problem_l3924_392421

theorem function_value_problem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (2 * x - 1) = 3 * x + a) →
  f 3 = 2 →
  a = -4 :=
by sorry

end NUMINAMATH_CALUDE_function_value_problem_l3924_392421


namespace NUMINAMATH_CALUDE_gmat_scores_l3924_392462

theorem gmat_scores (u v : ℝ) (h1 : u > v) (h2 : u - v = (u + v) / 2) : v / u = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gmat_scores_l3924_392462


namespace NUMINAMATH_CALUDE_jerrys_average_increase_l3924_392480

theorem jerrys_average_increase (initial_average : ℝ) (fourth_test_score : ℝ) : 
  initial_average = 85 → fourth_test_score = 93 → 
  (4 * (initial_average + 2) = 3 * initial_average + fourth_test_score) := by
sorry

end NUMINAMATH_CALUDE_jerrys_average_increase_l3924_392480


namespace NUMINAMATH_CALUDE_clothing_cost_price_l3924_392428

theorem clothing_cost_price 
  (marked_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (h1 : marked_price = 132)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.1)
  : ∃ (cost_price : ℝ), 
    cost_price = 108 ∧ 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_clothing_cost_price_l3924_392428


namespace NUMINAMATH_CALUDE_worker_original_wage_l3924_392408

/-- Calculates the worker's original daily wage given the conditions of the problem -/
def calculate_original_wage (increase_percentage : ℚ) (final_take_home : ℚ) 
  (tax_rate : ℚ) (fixed_deduction : ℚ) : ℚ :=
  let increased_wage := (1 + increase_percentage) * (final_take_home + fixed_deduction) / (1 - tax_rate)
  increased_wage / (1 + increase_percentage)

/-- Theorem stating that the worker's original daily wage is $37.50 -/
theorem worker_original_wage :
  calculate_original_wage (1/2) 42 (1/5) 3 = 75/2 :=
sorry

end NUMINAMATH_CALUDE_worker_original_wage_l3924_392408


namespace NUMINAMATH_CALUDE_translation_equivalence_l3924_392449

noncomputable def original_function (x : ℝ) : ℝ :=
  Real.sin (2 * x + Real.pi / 6)

noncomputable def translated_function (x : ℝ) : ℝ :=
  original_function (x + Real.pi / 6)

theorem translation_equivalence :
  ∀ x : ℝ, translated_function x = Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_translation_equivalence_l3924_392449


namespace NUMINAMATH_CALUDE_intersection_distance_squared_specific_circles_l3924_392414

/-- Two circles in a 2D plane -/
structure TwoCircles where
  center1 : ℝ × ℝ
  radius1 : ℝ
  center2 : ℝ × ℝ
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionDistanceSquared (circles : TwoCircles) : ℝ := sorry

theorem intersection_distance_squared_specific_circles :
  let circles : TwoCircles := {
    center1 := (3, -2),
    radius1 := 5,
    center2 := (3, 4),
    radius2 := 3
  }
  intersectionDistanceSquared circles = 224 / 9 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_specific_circles_l3924_392414


namespace NUMINAMATH_CALUDE_abe_age_problem_l3924_392445

/-- The number of years ago that satisfies the equation for Abe's ages -/
def years_ago : ℕ := 21

/-- Abe's present age -/
def present_age : ℕ := 25

/-- The sum of Abe's present age and his age a certain number of years ago -/
def age_sum : ℕ := 29

theorem abe_age_problem :
  present_age + (present_age - years_ago) = age_sum :=
by sorry

end NUMINAMATH_CALUDE_abe_age_problem_l3924_392445


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3924_392405

/-- The line (a+1)x - y - 2a + 1 = 0 passes through the point (2,3) for all real a -/
theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), (a + 1) * 2 - 3 - 2 * a + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3924_392405


namespace NUMINAMATH_CALUDE_bananas_per_box_l3924_392498

/-- Given 40 bananas and 10 boxes, prove that the number of bananas per box is 4. -/
theorem bananas_per_box (total_bananas : ℕ) (total_boxes : ℕ) (h1 : total_bananas = 40) (h2 : total_boxes = 10) :
  total_bananas / total_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_box_l3924_392498


namespace NUMINAMATH_CALUDE_sqrt_12_times_sqrt_75_l3924_392400

theorem sqrt_12_times_sqrt_75 : Real.sqrt 12 * Real.sqrt 75 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_times_sqrt_75_l3924_392400


namespace NUMINAMATH_CALUDE_max_product_of_digits_optimal_solution_l3924_392478

-- Define a structure for a four-digit number of the form (abab)
structure FourDigitNumber (a b : Nat) : Type :=
  (a_valid : a > 0 ∧ a < 10)
  (b_valid : b ≥ 0 ∧ b < 10)

def value (n : FourDigitNumber a b) : Nat :=
  1000 * a + 100 * b + 10 * a + b

-- Define the problem
theorem max_product_of_digits (m : FourDigitNumber a b) (n : FourDigitNumber c d) :
  (∃ (t : Nat), value m + value n = t * t) →
  a * b * c * d ≤ 600 := by
  sorry

-- Define the optimality
theorem optimal_solution :
  ∃ (m : FourDigitNumber a b) (n : FourDigitNumber c d),
    (∃ (t : Nat), value m + value n = t * t) ∧
    a * b * c * d = 600 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_digits_optimal_solution_l3924_392478


namespace NUMINAMATH_CALUDE_angle4_is_35_degrees_l3924_392442

-- Define angles as real numbers (in degrees)
variable (angle1 angle2 angle3 angle4 : ℝ)

-- State the theorem
theorem angle4_is_35_degrees
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4) :
  angle4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_angle4_is_35_degrees_l3924_392442


namespace NUMINAMATH_CALUDE_white_balls_count_l3924_392469

theorem white_balls_count (total : ℕ) (p_yellow : ℚ) : 
  total = 10 → p_yellow = 6/10 → (total : ℚ) * (1 - p_yellow) = 4 := by sorry

end NUMINAMATH_CALUDE_white_balls_count_l3924_392469


namespace NUMINAMATH_CALUDE_oabc_shape_oabc_not_rhombus_l3924_392463

/-- Given distinct points A, B, and C on a coordinate plane with origin O,
    prove that OABC can form either a parallelogram or a straight line, but not a rhombus. -/
theorem oabc_shape (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2*x₁ - x₂, 2*y₁ - y₂) ∧ (x₂, y₂) ≠ (2*x₁ - x₂, 2*y₁ - y₂)) :
  (∃ (k : ℝ), k ≠ 0 ∧ k ≠ 1 ∧ x₂ = k * x₁ ∧ y₂ = k * y₁) ∨ 
  (x₁ + x₂ = 2*x₁ - x₂ ∧ y₁ + y₂ = 2*y₁ - y₂) :=
by sorry

/-- The figure OABC cannot form a rhombus. -/
theorem oabc_not_rhombus (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2*x₁ - x₂, 2*y₁ - y₂) ∧ (x₂, y₂) ≠ (2*x₁ - x₂, 2*y₁ - y₂)) :
  ¬(x₁^2 + y₁^2 = x₂^2 + y₂^2 ∧ 
    x₁^2 + y₁^2 = (2*x₁ - x₂)^2 + (2*y₁ - y₂)^2 ∧ 
    x₂^2 + y₂^2 = (2*x₁ - x₂)^2 + (2*y₁ - y₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_oabc_shape_oabc_not_rhombus_l3924_392463


namespace NUMINAMATH_CALUDE_equation_transformation_correct_l3924_392484

theorem equation_transformation_correct (x : ℝ) :
  (x + 1) / 2 - 1 = (2 * x - 1) / 3 ↔ 3 * (x + 1) - 6 = 2 * (2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_correct_l3924_392484


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3924_392493

theorem power_of_two_equality (M : ℕ) : (32^3) * (16^3) = 2^M → M = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3924_392493


namespace NUMINAMATH_CALUDE_odd_plus_one_even_implies_f_four_zero_l3924_392431

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_plus_one_even_implies_f_four_zero (f : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even (fun x ↦ f (x + 1))) : f 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_plus_one_even_implies_f_four_zero_l3924_392431


namespace NUMINAMATH_CALUDE_sin_shift_l3924_392461

theorem sin_shift (x : ℝ) : 
  Real.sin (4 * x - π / 3) = Real.sin (4 * (x - π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l3924_392461


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l3924_392423

theorem sum_of_coefficients_zero (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 2) = 2 * x^2 + 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l3924_392423


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3924_392436

theorem imaginary_part_of_z (z : ℂ) : z = (1 + 2*I) / ((1 - I)^2) → z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3924_392436


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l3924_392452

/-- Represents a participant's score in a math competition --/
structure Score where
  points : ℕ
  total : ℕ
  h_positive : points > 0
  h_valid : points ≤ total

/-- Represents a participant's scores over three days --/
structure ThreeDayScore where
  day1 : Score
  day2 : Score
  day3 : Score
  h_total : day1.total + day2.total + day3.total = 600

def success_ratio (s : Score) : ℚ :=
  s.points / s.total

def three_day_success_ratio (s : ThreeDayScore) : ℚ :=
  (s.day1.points + s.day2.points + s.day3.points) / 600

theorem delta_max_success_ratio 
  (charlie : ThreeDayScore)
  (delta : ThreeDayScore)
  (h_charlie_day1 : charlie.day1 = ⟨210, 350, by sorry, by sorry⟩)
  (h_charlie_day2 : charlie.day2 = ⟨170, 250, by sorry, by sorry⟩)
  (h_charlie_day3 : charlie.day3 = ⟨0, 0, by sorry, by sorry⟩)
  (h_delta_day1 : success_ratio delta.day1 < success_ratio charlie.day1)
  (h_delta_day2 : success_ratio delta.day2 < success_ratio charlie.day2)
  (h_delta_day3_positive : delta.day3.points > 0) :
  three_day_success_ratio delta ≤ 378 / 600 :=
sorry

end NUMINAMATH_CALUDE_delta_max_success_ratio_l3924_392452


namespace NUMINAMATH_CALUDE_pear_price_is_correct_l3924_392477

/-- The price of a pear in won -/
def pear_price : ℕ := 6300

/-- The price of an apple in won -/
def apple_price : ℕ := pear_price + 2400

/-- The sum of the prices of an apple and a pear in won -/
def total_price : ℕ := 15000

theorem pear_price_is_correct : pear_price = 6300 := by
  have h1 : apple_price + pear_price = total_price := by sorry
  have h2 : apple_price = pear_price + 2400 := by sorry
  sorry

end NUMINAMATH_CALUDE_pear_price_is_correct_l3924_392477


namespace NUMINAMATH_CALUDE_center_coordinate_sum_l3924_392499

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4*x + 12*y - 39

/-- The center of a circle given by its equation -/
def CenterOfCircle (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 1

/-- Theorem: The sum of coordinates of the center of the given circle is 8 -/
theorem center_coordinate_sum :
  ∃ h k : ℝ, CenterOfCircle h k ∧ h + k = 8 := by sorry

end NUMINAMATH_CALUDE_center_coordinate_sum_l3924_392499


namespace NUMINAMATH_CALUDE_rahims_average_book_price_l3924_392487

/-- Calculates the average price per book given two purchases -/
def average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating the average price per book for Rahim's purchases -/
theorem rahims_average_book_price :
  let books1 : ℕ := 65
  let books2 : ℕ := 50
  let price1 : ℚ := 1160
  let price2 : ℚ := 920
  let avg_price := average_price_per_book books1 books2 price1 price2
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |avg_price - 18.09| < ε :=
sorry

end NUMINAMATH_CALUDE_rahims_average_book_price_l3924_392487


namespace NUMINAMATH_CALUDE_min_y_value_l3924_392425

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 56*y) : 
  y ≥ 28 - 2 * Real.sqrt 212 := by
sorry

end NUMINAMATH_CALUDE_min_y_value_l3924_392425


namespace NUMINAMATH_CALUDE_initial_erasers_count_l3924_392422

/-- The number of scissors initially in the drawer -/
def initial_scissors : ℕ := 118

/-- The number of erasers Jason placed in the drawer -/
def erasers_added : ℕ := 131

/-- The total number of erasers after Jason added some -/
def total_erasers : ℕ := 270

/-- The initial number of erasers in the drawer -/
def initial_erasers : ℕ := total_erasers - erasers_added

theorem initial_erasers_count : initial_erasers = 139 := by
  sorry

end NUMINAMATH_CALUDE_initial_erasers_count_l3924_392422


namespace NUMINAMATH_CALUDE_sand_lost_during_journey_l3924_392417

theorem sand_lost_during_journey (initial_sand final_sand : ℝ) 
  (h1 : initial_sand = 4.1)
  (h2 : final_sand = 1.7) :
  initial_sand - final_sand = 2.4 := by
sorry

end NUMINAMATH_CALUDE_sand_lost_during_journey_l3924_392417


namespace NUMINAMATH_CALUDE_maintenance_team_journey_l3924_392402

/-- Represents the direction of travel --/
inductive Direction
  | East
  | West

/-- Represents a segment of the journey --/
structure Segment where
  distance : ℝ
  direction : Direction

/-- Calculates the net distance traveled given a list of segments --/
def netDistance (journey : List Segment) : ℝ := sorry

/-- Calculates the total distance traveled given a list of segments --/
def totalDistance (journey : List Segment) : ℝ := sorry

/-- Theorem: The maintenance team's final position and fuel consumption --/
theorem maintenance_team_journey 
  (journey : List Segment)
  (fuel_rate : ℝ)
  (h1 : journey = [
    ⟨12, Direction.East⟩, 
    ⟨6, Direction.West⟩, 
    ⟨4, Direction.East⟩, 
    ⟨2, Direction.West⟩, 
    ⟨8, Direction.West⟩, 
    ⟨13, Direction.East⟩, 
    ⟨2, Direction.West⟩
  ])
  (h2 : fuel_rate = 0.2) :
  netDistance journey = 11 ∧ 
  totalDistance journey * fuel_rate * 2 = 11.6 := by sorry

end NUMINAMATH_CALUDE_maintenance_team_journey_l3924_392402


namespace NUMINAMATH_CALUDE_triangle_properties_l3924_392492

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * (Real.cos ((t.C - t.A) / 2))^2 * Real.cos t.A - 
        Real.sin (t.C - t.A) * Real.sin t.A + 
        Real.cos (t.B + t.C) = 1/3)
  (h2 : t.c = 2 * Real.sqrt 2) : 
  Real.sin t.C = (2 * Real.sqrt 2) / 3 ∧ 
  ∃ (max_area : ℝ), max_area = 2 * Real.sqrt 2 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3924_392492


namespace NUMINAMATH_CALUDE_debby_flour_purchase_l3924_392401

/-- Given that Debby initially had 12 pounds of flour and ended up with 16 pounds in total,
    prove that she bought 4 pounds of flour. -/
theorem debby_flour_purchase (initial_flour : ℕ) (total_flour : ℕ) (purchased_flour : ℕ) :
  initial_flour = 12 →
  total_flour = 16 →
  total_flour = initial_flour + purchased_flour →
  purchased_flour = 4 := by
  sorry

end NUMINAMATH_CALUDE_debby_flour_purchase_l3924_392401


namespace NUMINAMATH_CALUDE_chord_length_l3924_392420

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by sorry

end NUMINAMATH_CALUDE_chord_length_l3924_392420


namespace NUMINAMATH_CALUDE_min_value_of_product_l3924_392412

-- Define the quadratic function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem min_value_of_product (a b c : ℝ) (x₁ x₂ x₃ : ℝ) :
  a ≠ 0 →
  f a b c (-1) = 0 →
  (∀ x : ℝ, f a b c x ≥ x) →
  (∀ x ∈ Set.Ioo 0 2, f a b c x ≤ (x + 1)^2 / 4) →
  x₁ ∈ Set.Ioo 0 2 →
  x₂ ∈ Set.Ioo 0 2 →
  x₃ ∈ Set.Ioo 0 2 →
  1 / x₁ + 1 / x₂ + 1 / x₃ = 3 →
  (f a b c x₁) * (f a b c x₂) * (f a b c x₃) ≥ 1 :=
by sorry

#check min_value_of_product

end NUMINAMATH_CALUDE_min_value_of_product_l3924_392412


namespace NUMINAMATH_CALUDE_smallest_value_l3924_392433

theorem smallest_value (y : ℝ) (h : y = 8) :
  let a := 5 / (y - 1)
  let b := 5 / (y + 1)
  let c := 5 / y
  let d := (5 + y) / 10
  let e := y - 5
  b < a ∧ b < c ∧ b < d ∧ b < e :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_l3924_392433


namespace NUMINAMATH_CALUDE_right_triangle_set_l3924_392496

/-- Checks if a set of three numbers can form a right-angled triangle --/
def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The sets of sticks given in the problem --/
def stickSets : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7)]

theorem right_triangle_set :
  ∃! (a b c : ℕ), (a, b, c) ∈ stickSets ∧ isRightTriangle a b c :=
by
  sorry

#check right_triangle_set

end NUMINAMATH_CALUDE_right_triangle_set_l3924_392496


namespace NUMINAMATH_CALUDE_simplify_fraction_l3924_392489

theorem simplify_fraction (a : ℝ) (ha : a ≠ 0) :
  1 - (1 / (1 + (a - 1) / (a + 1))) = (a - 1) / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3924_392489


namespace NUMINAMATH_CALUDE_dennis_purchase_cost_l3924_392413

/-- Calculates the total cost after discount for Dennis's purchase -/
def total_cost_after_discount (pants_price : ℝ) (socks_price : ℝ) (pants_quantity : ℕ) (socks_quantity : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_before_discount := pants_price * pants_quantity + socks_price * socks_quantity
  total_before_discount * (1 - discount_rate)

/-- Proves that the total cost after discount for Dennis's purchase is $392 -/
theorem dennis_purchase_cost :
  total_cost_after_discount 110 60 4 2 0.3 = 392 := by
  sorry

end NUMINAMATH_CALUDE_dennis_purchase_cost_l3924_392413


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3924_392494

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 6) 
  (h_sum : a 3 + a 5 = 0) :
  ∀ n : ℕ, a n = 8 - 2 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3924_392494


namespace NUMINAMATH_CALUDE_mean_calculation_l3924_392456

theorem mean_calculation (x : ℝ) : 
  (28 + x + 70 + 88 + 104) / 5 = 67 →
  (50 + 62 + 97 + 124 + x) / 5 = 75.6 :=
by sorry

end NUMINAMATH_CALUDE_mean_calculation_l3924_392456


namespace NUMINAMATH_CALUDE_two_inequalities_for_real_numbers_l3924_392448

theorem two_inequalities_for_real_numbers (a b c : ℝ) : 
  (a * b / c^2 + b * c / a^2 + a * c / b^2 ≥ a / c + b / a + c / b) ∧
  (a^2 / b^2 + b^2 / c^2 + c^2 / a^2 ≥ a / b + b / c + c / a) := by
  sorry

end NUMINAMATH_CALUDE_two_inequalities_for_real_numbers_l3924_392448


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l3924_392435

/-- Two points are symmetric with respect to the x-axis if they have the same x-coordinate
    and their y-coordinates are negatives of each other -/
def symmetric_wrt_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- The theorem states that if point A(m,n) is symmetric to point B(1,-2) with respect to the x-axis,
    then m = 1 and n = 2 -/
theorem symmetry_coordinates :
  ∀ m n : ℝ, symmetric_wrt_x_axis (m, n) (1, -2) → m = 1 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l3924_392435


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3924_392472

theorem inequality_system_solution (p : ℝ) : 19 * p < 10 ∧ p > (1/2 : ℝ) → (1/2 : ℝ) < p ∧ p < 10/19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3924_392472


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3924_392407

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 5 = 3 →
  a 6 = -2 →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3924_392407


namespace NUMINAMATH_CALUDE_circle_symmetry_l3924_392438

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), 
    (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
      symmetry_line ((x + x₀)/2) ((y + y₀)/2)) →
    symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3924_392438


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l3924_392454

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l3924_392454


namespace NUMINAMATH_CALUDE_problem_solution_l3924_392437

theorem problem_solution (m : ℝ) (h : m + 1/m = 5) : m^2 + 1/m^2 + 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3924_392437


namespace NUMINAMATH_CALUDE_total_license_plates_l3924_392429

/-- The number of vowels in the license plate system -/
def num_vowels : ℕ := 8

/-- The number of consonants in the license plate system -/
def num_consonants : ℕ := 26 - num_vowels

/-- The number of even digits (0, 2, 4, 6, 8) -/
def num_even_digits : ℕ := 5

/-- The structure of a license plate: consonant, vowel, consonant, even digit, even digit -/
def license_plate_structure := 
  num_consonants * num_vowels * num_consonants * num_even_digits * num_even_digits

/-- The total number of possible license plates -/
theorem total_license_plates : license_plate_structure = 25920 := by
  sorry

end NUMINAMATH_CALUDE_total_license_plates_l3924_392429


namespace NUMINAMATH_CALUDE_football_field_length_proof_l3924_392475

/-- The length of a football field in yards -/
def football_field_length : ℝ := 200

/-- The number of football fields a potato is launched across -/
def fields_crossed : ℕ := 6

/-- The speed of the dog in feet per minute -/
def dog_speed : ℝ := 400

/-- The time taken by the dog to fetch the potato in minutes -/
def fetch_time : ℝ := 9

/-- The number of feet in a yard -/
def feet_per_yard : ℝ := 3

theorem football_field_length_proof :
  football_field_length = 
    (dog_speed / feet_per_yard * fetch_time) / fields_crossed := by
  sorry

end NUMINAMATH_CALUDE_football_field_length_proof_l3924_392475


namespace NUMINAMATH_CALUDE_calculation_proof_l3924_392434

theorem calculation_proof : (Real.sqrt 2 - Real.sqrt 3) * (Real.sqrt 2 + Real.sqrt 3) + (2 * Real.sqrt 2 - 1)^2 = 8 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3924_392434


namespace NUMINAMATH_CALUDE_function_satisfies_equation_value_at_sqrt_2014_l3924_392415

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that the function f(x) = 1 - x²/2 satisfies the functional equation -/
theorem function_satisfies_equation :
    ∃ f : ℝ → ℝ, FunctionalEquation f ∧ ∀ x : ℝ, f x = 1 - x^2 / 2 := by
  sorry

/-- The value of f(√2014) -/
theorem value_at_sqrt_2014 :
    ∀ f : ℝ → ℝ, FunctionalEquation f → f (Real.sqrt 2014) = -1006 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_value_at_sqrt_2014_l3924_392415


namespace NUMINAMATH_CALUDE_power_of_power_l3924_392468

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3924_392468


namespace NUMINAMATH_CALUDE_two_m_squared_eq_three_n_cubed_l3924_392467

theorem two_m_squared_eq_three_n_cubed (m n : ℕ+) :
  2 * m ^ 2 = 3 * n ^ 3 ↔ ∃ k : ℕ+, m = 18 * k ^ 3 ∧ n = 6 * k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_two_m_squared_eq_three_n_cubed_l3924_392467


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l3924_392495

theorem sphere_radius_ratio (V₁ V₂ : ℝ) (h₁ : V₁ = 512 * Real.pi) (h₂ : V₂ = 32 * Real.pi) :
  (V₂ / V₁) ^ (1/3 : ℝ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l3924_392495


namespace NUMINAMATH_CALUDE_min_value_expression_l3924_392482

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 2) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 2 ∧
  (∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 2 → a' + b' = 2 →
    (a' * c' / b' + c' / (a' * b') - c' / 2 + Real.sqrt 5 / (c' - 2) ≥ Real.sqrt 10 + Real.sqrt 5)) ∧
  (x * (2 + Real.sqrt 2) / y + (2 + Real.sqrt 2) / (x * y) - (2 + Real.sqrt 2) / 2 + Real.sqrt 5 / Real.sqrt 2 = Real.sqrt 10 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3924_392482


namespace NUMINAMATH_CALUDE_investment_sum_l3924_392427

/-- 
Given a sum P invested at simple interest for two years, 
if the difference in interest between 18% p.a. and 12% p.a. is Rs. 840, 
then P = 7000.
-/
theorem investment_sum (P : ℝ) : 
  (P * (18 / 100) * 2 - P * (12 / 100) * 2 = 840) → P = 7000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l3924_392427


namespace NUMINAMATH_CALUDE_least_number_divisible_by_53_and_71_l3924_392404

theorem least_number_divisible_by_53_and_71 (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((1357 + y) % 53 = 0 ∧ (1357 + y) % 71 = 0)) ∧ 
  (1357 + x) % 53 = 0 ∧ (1357 + x) % 71 = 0 → 
  x = 2406 := by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_53_and_71_l3924_392404


namespace NUMINAMATH_CALUDE_books_divided_l3924_392464

theorem books_divided (num_girls num_boys : ℕ) (total_girls_books : ℕ) : 
  num_girls = 15 →
  num_boys = 10 →
  total_girls_books = 225 →
  (total_girls_books / num_girls) * (num_girls + num_boys) = 375 :=
by sorry

end NUMINAMATH_CALUDE_books_divided_l3924_392464


namespace NUMINAMATH_CALUDE_m_range_when_p_or_q_false_l3924_392497

theorem m_range_when_p_or_q_false (m : ℝ) :
  (¬ ((∃ x : ℝ, m * x^2 + 1 ≤ 0) ∨ (∀ x : ℝ, x^2 + m * x + 1 > 0))) →
  m ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_when_p_or_q_false_l3924_392497


namespace NUMINAMATH_CALUDE_scale_division_l3924_392446

/-- The length of the scale in inches -/
def scale_length : ℕ := 80

/-- The length of each part in inches -/
def part_length : ℕ := 20

/-- The number of equal parts the scale is divided into -/
def num_parts : ℕ := scale_length / part_length

theorem scale_division :
  scale_length % part_length = 0 ∧ num_parts = 4 :=
by sorry

end NUMINAMATH_CALUDE_scale_division_l3924_392446


namespace NUMINAMATH_CALUDE_integer_solutions_l3924_392444

/-- The equation whose solutions we're interested in -/
def equation (k x : ℝ) : Prop :=
  (k^2 - 2*k)*x^2 - (6*k - 4)*x + 8 = 0

/-- Predicate to check if a number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- The main theorem stating the conditions for integer solutions -/
theorem integer_solutions (k : ℝ) :
  (∀ x : ℝ, equation k x → isInteger x) ↔ (k = 1 ∨ k = -2 ∨ k = 2/3) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_l3924_392444


namespace NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l3924_392471

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 2

-- State the theorem
theorem range_of_a_for_decreasing_f :
  (∀ a : ℝ, (∀ x : ℝ, (∀ y : ℝ, x < y → f a x > f a y)) ↔ a ∈ Set.Iic (-3)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l3924_392471


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3924_392466

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - 5 * x + 2 > 0) ↔ (x < 2/3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3924_392466


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3924_392490

theorem perfect_square_trinomial : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3924_392490


namespace NUMINAMATH_CALUDE_average_hamburgers_per_day_l3924_392459

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7

theorem average_hamburgers_per_day :
  (total_hamburgers : ℚ) / days_in_week = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_hamburgers_per_day_l3924_392459


namespace NUMINAMATH_CALUDE_population_growth_problem_l3924_392406

theorem population_growth_problem (x y z : ℕ) : 
  (3/2)^x * (128/225)^y * (5/6)^z = 2 ↔ x = 4 ∧ y = 1 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_problem_l3924_392406


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3924_392441

noncomputable def f (x : ℝ) : ℝ := -Real.cos x + Real.log x

theorem f_derivative_at_one : 
  deriv f 1 = 1 + Real.sin 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3924_392441


namespace NUMINAMATH_CALUDE_grandfather_cake_blue_candles_l3924_392443

/-- The number of blue candles on Caleb's grandfather's birthday cake -/
def blue_candles (total_candles yellow_candles red_candles : ℕ) : ℕ :=
  total_candles - (yellow_candles + red_candles)

/-- Theorem stating the number of blue candles on the cake -/
theorem grandfather_cake_blue_candles :
  blue_candles 79 27 14 = 38 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_cake_blue_candles_l3924_392443


namespace NUMINAMATH_CALUDE_binary_1010101_to_decimal_l3924_392419

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number 1010101₂ -/
def binary_1010101 : List Nat := [1, 0, 1, 0, 1, 0, 1]

/-- Theorem: The decimal equivalent of 1010101₂ is 85 -/
theorem binary_1010101_to_decimal :
  binary_to_decimal binary_1010101.reverse = 85 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010101_to_decimal_l3924_392419


namespace NUMINAMATH_CALUDE_circle_radius_theorem_l3924_392465

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if two circles touch externally -/
def touch_externally (c1 c2 : Circle) : Prop := sorry

/-- Checks if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop := sorry

theorem circle_radius_theorem (C1 C2 : Circle) (P Q R : Point) :
  C1.radius = 12 →
  touch_externally C1 C2 →
  on_circle P C1 →
  on_circle P C2 →
  on_circle Q C1 →
  on_circle R C2 →
  collinear P Q R →
  distance P Q = 7 →
  distance P R = 17 →
  C2.radius = 10 := by sorry

end NUMINAMATH_CALUDE_circle_radius_theorem_l3924_392465


namespace NUMINAMATH_CALUDE_chest_to_treadmill_ratio_l3924_392483

/-- The price of the treadmill in dollars -/
def treadmill_price : ℝ := 100

/-- The price of the television in dollars -/
def tv_price : ℝ := 3 * treadmill_price

/-- The total sum of money from the sale in dollars -/
def total_sum : ℝ := 600

/-- The price of the chest of drawers in dollars -/
def chest_price : ℝ := total_sum - treadmill_price - tv_price

/-- The theorem stating that the ratio of the chest price to the treadmill price is 2:1 -/
theorem chest_to_treadmill_ratio :
  chest_price / treadmill_price = 2 := by sorry

end NUMINAMATH_CALUDE_chest_to_treadmill_ratio_l3924_392483


namespace NUMINAMATH_CALUDE_smallest_3digit_base6_divisible_by_7_l3924_392455

/-- Converts a base 6 number to decimal --/
def base6ToDecimal (n : ℕ) : ℕ :=
  sorry

/-- Converts a decimal number to base 6 --/
def decimalToBase6 (n : ℕ) : ℕ :=
  sorry

/-- Checks if a number is a 3-digit base 6 number --/
def isThreeDigitBase6 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem smallest_3digit_base6_divisible_by_7 :
  ∃ (n : ℕ), isThreeDigitBase6 n ∧ 
             base6ToDecimal n % 7 = 0 ∧
             decimalToBase6 (base6ToDecimal n) = 110 ∧
             ∀ (m : ℕ), isThreeDigitBase6 m ∧ base6ToDecimal m % 7 = 0 → base6ToDecimal n ≤ base6ToDecimal m :=
by sorry

end NUMINAMATH_CALUDE_smallest_3digit_base6_divisible_by_7_l3924_392455


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l3924_392488

-- Define the cubic polynomial
def cubic_poly (x : ℝ) : ℝ := x^3 - 7*x^2 + 3*x + 4

-- Define the roots of the polynomial
axiom a : ℝ
axiom b : ℝ
axiom c : ℝ

-- State that a, b, c are roots of the polynomial
axiom a_root : cubic_poly a = 0
axiom b_root : cubic_poly b = 0
axiom c_root : cubic_poly c = 0

-- State that a, b, c are distinct
axiom roots_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Theorem to prove
theorem sum_of_reciprocal_squares : 
  1/a^2 + 1/b^2 + 1/c^2 = 65/16 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l3924_392488


namespace NUMINAMATH_CALUDE_product_sum_fractions_l3924_392409

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l3924_392409


namespace NUMINAMATH_CALUDE_derivative_of_2x_sin_2x_plus_5_l3924_392460

/-- The derivative of f(x) = 2x sin(2x + 5) is f'(x) = 2 sin(2x + 5) + 4x cos(2x + 5) -/
theorem derivative_of_2x_sin_2x_plus_5 (x : ℝ) :
  deriv (fun x => 2 * x * Real.sin (2 * x + 5)) x =
    2 * Real.sin (2 * x + 5) + 4 * x * Real.cos (2 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_2x_sin_2x_plus_5_l3924_392460


namespace NUMINAMATH_CALUDE_document_word_count_l3924_392418

/-- Calculates the number of words in a document based on typing speed and time --/
def document_words (original_speed : ℕ) (speed_reduction : ℕ) (typing_time : ℕ) : ℕ :=
  (original_speed - speed_reduction) * typing_time

/-- Proves that the number of words in the document is 810 --/
theorem document_word_count : document_words 65 20 18 = 810 := by
  sorry

end NUMINAMATH_CALUDE_document_word_count_l3924_392418


namespace NUMINAMATH_CALUDE_orange_count_l3924_392476

/-- Represents the count of fruits in a box -/
structure FruitBox where
  apples : ℕ
  pears : ℕ
  oranges : ℕ

/-- The properties of the fruit box as described in the problem -/
def is_valid_fruit_box (box : FruitBox) : Prop :=
  box.apples + box.pears + box.oranges = 60 ∧
  box.apples = 3 * (box.pears + box.oranges) ∧
  box.pears * 5 = box.apples + box.oranges

/-- Theorem stating that a valid fruit box has 5 oranges -/
theorem orange_count (box : FruitBox) (h : is_valid_fruit_box box) : box.oranges = 5 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_l3924_392476


namespace NUMINAMATH_CALUDE_circle_circumference_area_relation_l3924_392432

theorem circle_circumference_area_relation : 
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 12 → π * d < 10 * (π * d^2 / 4) := by
  sorry

#check circle_circumference_area_relation

end NUMINAMATH_CALUDE_circle_circumference_area_relation_l3924_392432


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3924_392474

theorem circle_equation_proof (x y : ℝ) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 4*x + 6*y - 3 = 0
  let M : ℝ × ℝ := (-1, 1)
  let new_circle : ℝ → ℝ → Prop := λ x y => (x - 2)^2 + (y + 3)^2 = 25
  (∃ h k r : ℝ, ∀ x y : ℝ, C x y ↔ (x - h)^2 + (y - k)^2 = r^2) →
  (new_circle M.1 M.2) ∧
  (∀ x y : ℝ, C x y ↔ (x - 2)^2 + (y + 3)^2 = r^2) →
  (∀ x y : ℝ, new_circle x y ↔ (x - 2)^2 + (y + 3)^2 = 25) :=
by sorry


end NUMINAMATH_CALUDE_circle_equation_proof_l3924_392474


namespace NUMINAMATH_CALUDE_number_puzzle_l3924_392440

theorem number_puzzle : ∃ x : ℚ, x = (3/5) * (2*x) + 238 ∧ x = 1190 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3924_392440


namespace NUMINAMATH_CALUDE_parallel_lines_set_l3924_392403

-- Define the plane
variable (Plane : Type)

-- Define points on the plane
variable (D E P : Plane)

-- Define the distance between two points
variable (distance : Plane → Plane → ℝ)

-- Define the area of a triangle given three points
variable (triangle_area : Plane → Plane → Plane → ℝ)

-- Define a set of points
variable (T : Set Plane)

-- State the theorem
theorem parallel_lines_set (h_distinct : D ≠ E) :
  T = {P | triangle_area D E P = 0.5} →
  ∃ (l₁ l₂ : Set Plane), 
    (∀ X ∈ l₁, ∀ Y ∈ l₂, distance X Y = 2 / distance D E) ∧
    T = l₁ ∪ l₂ :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_set_l3924_392403


namespace NUMINAMATH_CALUDE_system_solutions_correct_l3924_392424

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℚ, x - y = 2 ∧ 2*x + y = 7 ∧ x = 3 ∧ y = 1) ∧
  -- System 2
  (∃ x y : ℚ, x - 2*y = 3 ∧ (1/2)*x + (3/4)*y = 13/4 ∧ x = 5 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l3924_392424


namespace NUMINAMATH_CALUDE_no_quadratic_function_exists_l3924_392451

theorem no_quadratic_function_exists : 
  ¬ ∃ (b c : ℝ), 
    ((-4)^2 + b*(-4) + c = 1) ∧ 
    (∀ x : ℝ, 6*x ≤ 3*x^2 + 3 ∧ 3*x^2 + 3 ≤ x^2 + b*x + c) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_function_exists_l3924_392451


namespace NUMINAMATH_CALUDE_doughnuts_left_l3924_392457

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of doughnuts in the box -/
def boxDozens : ℕ := 2

/-- The number of doughnuts eaten -/
def eatenDoughnuts : ℕ := 8

/-- Theorem: Given a box with 2 dozen doughnuts and 8 doughnuts eaten, 
    the number of doughnuts left is 16 -/
theorem doughnuts_left : 
  boxDozens * dozen - eatenDoughnuts = 16 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_left_l3924_392457


namespace NUMINAMATH_CALUDE_statement_d_is_incorrect_l3924_392485

theorem statement_d_is_incorrect : ∃ (a b : ℝ), a^2 > b^2 ∧ a * b > 0 ∧ 1 / a ≥ 1 / b := by
  sorry

end NUMINAMATH_CALUDE_statement_d_is_incorrect_l3924_392485


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3924_392426

/-- A geometric sequence with the given property -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  arithmetic_property : 3 * a 1 + 2 * a 2 = a 3

/-- The main theorem -/
theorem geometric_sequence_property (seq : GeometricSequence) :
  (seq.a 9 + seq.a 10) / (seq.a 7 + seq.a 8) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3924_392426


namespace NUMINAMATH_CALUDE_continuous_functional_equation_solution_l3924_392486

/-- A function that satisfies the given functional equation and condition -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y, 4 * f (x + y) = f x * f y) ∧ f 1 = 12

theorem continuous_functional_equation_solution 
  (f : ℝ → ℝ) 
  (hf : Continuous f) 
  (heq : FunctionalEquation f) : 
  ∀ x, f x = 4 * (3 : ℝ) ^ x := by
  sorry

end NUMINAMATH_CALUDE_continuous_functional_equation_solution_l3924_392486
