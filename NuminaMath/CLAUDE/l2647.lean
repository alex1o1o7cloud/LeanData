import Mathlib

namespace NUMINAMATH_CALUDE_new_light_wattage_l2647_264708

theorem new_light_wattage (original_wattage : ℝ) (percentage_increase : ℝ) :
  original_wattage = 80 →
  percentage_increase = 25 →
  original_wattage * (1 + percentage_increase / 100) = 100 := by
  sorry

end NUMINAMATH_CALUDE_new_light_wattage_l2647_264708


namespace NUMINAMATH_CALUDE_stream_speed_l2647_264719

/-- The speed of the stream given rowing conditions -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ)
                     (downstream_time : ℝ) (upstream_time : ℝ)
                     (h1 : downstream_distance = 120)
                     (h2 : upstream_distance = 90)
                     (h3 : downstream_time = 4)
                     (h4 : upstream_time = 6) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2647_264719


namespace NUMINAMATH_CALUDE_tangent_line_implies_b_minus_a_equals_three_l2647_264755

-- Define the function f(x) = ax³ + bx - 1
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 1

-- Define the derivative of f
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

theorem tangent_line_implies_b_minus_a_equals_three (a b : ℝ) :
  f_derivative a b 1 = 1 ∧ f a b 1 = 1 → b - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_b_minus_a_equals_three_l2647_264755


namespace NUMINAMATH_CALUDE_six_ronna_scientific_notation_l2647_264769

/-- Represents the number of zeros after the number for the "ronna" prefix -/
def ronna_zeros : ℕ := 27

/-- Converts a number with the "ronna" prefix to scientific notation -/
def ronna_to_scientific (n : ℝ) : ℝ := n * (10 ^ ronna_zeros)

/-- Theorem stating that 6 ronna is equal to 6 × 10^27 -/
theorem six_ronna_scientific_notation : ronna_to_scientific 6 = 6 * (10 ^ 27) := by
  sorry

end NUMINAMATH_CALUDE_six_ronna_scientific_notation_l2647_264769


namespace NUMINAMATH_CALUDE_comic_book_arrangements_l2647_264729

def batman_comics : ℕ := 5
def superman_comics : ℕ := 3
def xmen_comics : ℕ := 6
def ironman_comics : ℕ := 4

def total_arrangements : ℕ := 2987520000

theorem comic_book_arrangements :
  (batman_comics.factorial * superman_comics.factorial * xmen_comics.factorial * ironman_comics.factorial) *
  (batman_comics + superman_comics + xmen_comics + ironman_comics).factorial =
  total_arrangements := by sorry

end NUMINAMATH_CALUDE_comic_book_arrangements_l2647_264729


namespace NUMINAMATH_CALUDE_money_problem_l2647_264760

theorem money_problem (a b : ℝ) 
  (eq_condition : 6 * a + b = 66)
  (ineq_condition : 4 * a + b < 48) :
  a > 9 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_money_problem_l2647_264760


namespace NUMINAMATH_CALUDE_cos_42_cos_18_minus_cos_48_sin_18_l2647_264746

theorem cos_42_cos_18_minus_cos_48_sin_18 :
  Real.cos (42 * π / 180) * Real.cos (18 * π / 180) -
  Real.cos (48 * π / 180) * Real.sin (18 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_42_cos_18_minus_cos_48_sin_18_l2647_264746


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2647_264705

theorem smallest_n_congruence : ∃! n : ℕ+, (3 * n : ℤ) ≡ 568 [ZMOD 34] ∧ 
  ∀ m : ℕ+, (3 * m : ℤ) ≡ 568 [ZMOD 34] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2647_264705


namespace NUMINAMATH_CALUDE_kates_wand_cost_l2647_264790

/-- Proves that the original cost of each wand is $60 given the conditions of Kate's wand purchase and sale. -/
theorem kates_wand_cost (total_wands : ℕ) (kept_wands : ℕ) (sold_wands : ℕ) 
  (price_increase : ℕ) (total_collected : ℕ) : ℕ :=
  by
  have h1 : total_wands = 3 := by sorry
  have h2 : kept_wands = 1 := by sorry
  have h3 : sold_wands = 2 := by sorry
  have h4 : price_increase = 5 := by sorry
  have h5 : total_collected = 130 := by sorry
  
  have h6 : sold_wands = total_wands - kept_wands := by sorry
  
  have h7 : total_collected / sold_wands - price_increase = 60 := by sorry
  
  exact 60

end NUMINAMATH_CALUDE_kates_wand_cost_l2647_264790


namespace NUMINAMATH_CALUDE_sum_of_integers_l2647_264728

theorem sum_of_integers (m n : ℕ+) (h1 : m * n = 2 * (m + n)) (h2 : m * n = 6 * (m - n)) :
  m + n = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2647_264728


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_over_f_at_zero_l2647_264724

noncomputable def f (x : ℝ) : ℝ := (x + Real.sqrt (1 + x^2))^10

theorem f_derivative_at_zero_over_f_at_zero : 
  (deriv f 0) / (f 0) = 10 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_over_f_at_zero_l2647_264724


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2647_264762

theorem quadratic_factorization (x : ℝ) :
  -x^2 + 4*x - 4 = -(x - 2)^2 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2647_264762


namespace NUMINAMATH_CALUDE_hyperbola_foci_l2647_264723

/-- The foci of the hyperbola y²/16 - x²/9 = 1 are located at (0, ±5) -/
theorem hyperbola_foci : 
  ∀ (x y : ℝ), (y^2 / 16 - x^2 / 9 = 1) → 
  ∃ (c : ℝ), c = 5 ∧ ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l2647_264723


namespace NUMINAMATH_CALUDE_frenchBulldogRatioIsTwo_l2647_264759

/-- The ratio of French Bulldogs Peter wants to Sam's -/
def frenchBulldogRatio (samGermanShepherds samFrenchBulldogs peterTotalDogs : ℕ) : ℚ :=
  let peterGermanShepherds := 3 * samGermanShepherds
  let peterFrenchBulldogs := peterTotalDogs - peterGermanShepherds
  (peterFrenchBulldogs : ℚ) / samFrenchBulldogs

/-- The ratio of French Bulldogs Peter wants to Sam's is 2:1 -/
theorem frenchBulldogRatioIsTwo :
  frenchBulldogRatio 3 4 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_frenchBulldogRatioIsTwo_l2647_264759


namespace NUMINAMATH_CALUDE_brenda_remaining_mice_l2647_264794

def total_baby_mice : ℕ := 3 * 8

def mice_given_to_robbie : ℕ := total_baby_mice / 6

def mice_sold_to_pet_store : ℕ := 3 * mice_given_to_robbie

def remaining_after_pet_store : ℕ := total_baby_mice - (mice_given_to_robbie + mice_sold_to_pet_store)

def mice_sold_as_feeder : ℕ := remaining_after_pet_store / 2

theorem brenda_remaining_mice :
  total_baby_mice - (mice_given_to_robbie + mice_sold_to_pet_store + mice_sold_as_feeder) = 4 := by
  sorry

end NUMINAMATH_CALUDE_brenda_remaining_mice_l2647_264794


namespace NUMINAMATH_CALUDE_newton_6_years_or_more_percentage_l2647_264796

/-- Represents the number of marks for each year range on the graph --/
structure EmployeeDistribution :=
  (less_than_1_year : ℕ)
  (one_to_2_years : ℕ)
  (two_to_3_years : ℕ)
  (three_to_4_years : ℕ)
  (four_to_5_years : ℕ)
  (five_to_6_years : ℕ)
  (six_to_7_years : ℕ)
  (seven_to_8_years : ℕ)
  (eight_to_9_years : ℕ)
  (nine_to_10_years : ℕ)

/-- Calculates the percentage of employees who have worked for 6 years or more --/
def percentage_6_years_or_more (dist : EmployeeDistribution) : ℚ :=
  let total_marks := dist.less_than_1_year + dist.one_to_2_years + dist.two_to_3_years +
                     dist.three_to_4_years + dist.four_to_5_years + dist.five_to_6_years +
                     dist.six_to_7_years + dist.seven_to_8_years + dist.eight_to_9_years +
                     dist.nine_to_10_years
  let marks_6_plus := dist.six_to_7_years + dist.seven_to_8_years + dist.eight_to_9_years +
                      dist.nine_to_10_years
  (marks_6_plus : ℚ) / (total_marks : ℚ) * 100

/-- The given distribution of marks on the graph --/
def newton_distribution : EmployeeDistribution :=
  { less_than_1_year := 6,
    one_to_2_years := 6,
    two_to_3_years := 7,
    three_to_4_years := 4,
    four_to_5_years := 3,
    five_to_6_years := 3,
    six_to_7_years := 3,
    seven_to_8_years := 1,
    eight_to_9_years := 1,
    nine_to_10_years := 1 }

theorem newton_6_years_or_more_percentage :
  percentage_6_years_or_more newton_distribution = 17.14 := by
  sorry

end NUMINAMATH_CALUDE_newton_6_years_or_more_percentage_l2647_264796


namespace NUMINAMATH_CALUDE_supplementary_angle_of_60_degrees_l2647_264797

theorem supplementary_angle_of_60_degrees (α : Real) : 
  α = 60 → 180 - α = 120 := by sorry

end NUMINAMATH_CALUDE_supplementary_angle_of_60_degrees_l2647_264797


namespace NUMINAMATH_CALUDE_elberta_amount_l2647_264749

/-- The amount of money Granny Smith has -/
def granny_smith : ℕ := 120

/-- The amount of money Anjou has -/
def anjou : ℕ := granny_smith / 2

/-- The amount of money Elberta has -/
def elberta : ℕ := anjou + 5

/-- Theorem stating that Elberta has $65 -/
theorem elberta_amount : elberta = 65 := by
  sorry

end NUMINAMATH_CALUDE_elberta_amount_l2647_264749


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l2647_264738

/-- Calculates the average speed of a car trip given specific conditions -/
theorem car_trip_average_speed :
  let total_time : ℝ := 6
  let first_part_time : ℝ := 4
  let first_part_speed : ℝ := 35
  let second_part_speed : ℝ := 44
  let total_distance : ℝ := first_part_speed * first_part_time + 
                             second_part_speed * (total_time - first_part_time)
  total_distance / total_time = 38 := by
sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l2647_264738


namespace NUMINAMATH_CALUDE_pauls_new_books_l2647_264782

theorem pauls_new_books (initial_books sold_books current_books : ℕ) : 
  initial_books = 2 → 
  sold_books = 94 → 
  current_books = 58 → 
  current_books = initial_books - sold_books + (sold_books - initial_books + current_books) :=
by
  sorry

end NUMINAMATH_CALUDE_pauls_new_books_l2647_264782


namespace NUMINAMATH_CALUDE_distinct_roots_condition_roots_when_k_is_one_l2647_264751

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := x^2 + (2*k + 3)*x + k^2 + 5*k

-- Theorem for part 1
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
   quadratic_equation k x = 0 ∧ 
   quadratic_equation k y = 0) →
  k < 9/8 :=
sorry

-- Theorem for part 2
theorem roots_when_k_is_one :
  quadratic_equation 1 (-2) = 0 ∧ 
  quadratic_equation 1 (-3) = 0 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_condition_roots_when_k_is_one_l2647_264751


namespace NUMINAMATH_CALUDE_goods_payment_calculation_l2647_264725

/-- Calculates the final amount to be paid for goods after rebate and sales tax. -/
def final_amount (total_cost rebate_percent sales_tax_percent : ℚ) : ℚ :=
  let rebate_amount := (rebate_percent / 100) * total_cost
  let amount_after_rebate := total_cost - rebate_amount
  let sales_tax := (sales_tax_percent / 100) * amount_after_rebate
  amount_after_rebate + sales_tax

/-- Proves that given a total cost of 6650, a rebate of 6%, and a sales tax of 10%,
    the final amount to be paid is 6876.10. -/
theorem goods_payment_calculation :
  final_amount 6650 6 10 = 6876.1 := by
  sorry

end NUMINAMATH_CALUDE_goods_payment_calculation_l2647_264725


namespace NUMINAMATH_CALUDE_increasing_function_property_l2647_264799

-- Define a function f on positive real numbers
variable (f : ℝ → ℝ)

-- Define the property of being increasing for positive real numbers
def IncreasingOnPositive (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → g x < g y

-- State the theorem
theorem increasing_function_property
  (h1 : IncreasingOnPositive (fun x => f x - x))
  (h2 : IncreasingOnPositive (fun x => f (x^2) - x^6)) :
  IncreasingOnPositive (fun x => f (x^3) - (Real.sqrt 3 / 2) * x^6) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_property_l2647_264799


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l2647_264707

theorem final_sum_after_operations (S x k : ℝ) (a b : ℝ) (h : a + b = S) :
  k * (a + x) + k * (b + x) = k * S + 2 * k * x := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l2647_264707


namespace NUMINAMATH_CALUDE_skater_race_solution_l2647_264757

/-- Represents the speeds and times of two speed skaters in a race --/
structure SkaterRace where
  v : ℝ  -- Speed of the second skater in m/s
  t1 : ℝ  -- Time for the first skater to complete 10000 m in seconds
  t2 : ℝ  -- Time for the second skater to complete 10000 m in seconds

/-- The speeds and times of the skaters satisfy the race conditions --/
def satisfies_conditions (race : SkaterRace) : Prop :=
  let v1 := race.v + 1/3  -- Speed of the first skater
  (v1 * 600 - race.v * 600 = 200) ∧  -- Overtaking condition
  (400 / race.v - 400 / v1 = 2) ∧  -- Lap time difference
  (10000 / v1 = race.t1) ∧  -- First skater's total time
  (10000 / race.v = race.t2)  -- Second skater's total time

/-- The theorem stating the correct speeds and times for the skaters --/
theorem skater_race_solution :
  ∃ (race : SkaterRace),
    satisfies_conditions race ∧
    race.v = 8 ∧
    race.t1 = 1200 ∧
    race.t2 = 1250 :=
sorry

end NUMINAMATH_CALUDE_skater_race_solution_l2647_264757


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l2647_264702

-- Define the inequality function
def f (m : ℝ) (x : ℝ) : Prop := m * x^2 + (2 * m - 1) * x - 2 > 0

-- Define the solution set for each case
def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 then { x | x < -2 }
  else if m > 0 then { x | x < -2 ∨ x > 1/m }
  else if -1/2 < m ∧ m < 0 then { x | 1/m < x ∧ x < -2 }
  else if m = -1/2 then ∅
  else { x | -2 < x ∧ x < 1/m }

-- State the theorem
theorem inequality_solution_sets (m : ℝ) :
  { x : ℝ | f m x } = solution_set m := by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l2647_264702


namespace NUMINAMATH_CALUDE_olaf_game_score_l2647_264718

theorem olaf_game_score (dad_score : ℕ) : 
  (3 * dad_score + dad_score = 28) → dad_score = 7 := by
  sorry

end NUMINAMATH_CALUDE_olaf_game_score_l2647_264718


namespace NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_l2647_264711

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem minimum_value_of_f_plus_f'_derivative (a : ℝ) :
  (∃ x, f' a x = 0 ∧ x = 2) →  -- f has an extremum at x = 2
  (∃ m n : ℝ, m ∈ Set.Icc (-1) 1 ∧ n ∈ Set.Icc (-1) 1 ∧
    ∀ p q : ℝ, p ∈ Set.Icc (-1) 1 → q ∈ Set.Icc (-1) 1 →
      f a m + f' a n ≤ f a p + f' a q) →
  (∃ m n : ℝ, m ∈ Set.Icc (-1) 1 ∧ n ∈ Set.Icc (-1) 1 ∧
    f a m + f' a n = -13) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_l2647_264711


namespace NUMINAMATH_CALUDE_apples_per_box_l2647_264713

theorem apples_per_box (total_apples : ℕ) (rotten_apples : ℕ) (num_boxes : ℕ) 
  (h1 : total_apples = 40)
  (h2 : rotten_apples = 4)
  (h3 : num_boxes = 4)
  (h4 : rotten_apples < total_apples) :
  (total_apples - rotten_apples) / num_boxes = 9 := by
sorry

end NUMINAMATH_CALUDE_apples_per_box_l2647_264713


namespace NUMINAMATH_CALUDE_product_of_two_digit_numbers_l2647_264779

theorem product_of_two_digit_numbers (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) → 
  (10 ≤ b ∧ b < 100) → 
  a * b = 4680 → 
  min a b = 40 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_digit_numbers_l2647_264779


namespace NUMINAMATH_CALUDE_square_sum_seventeen_l2647_264740

theorem square_sum_seventeen (x y : ℝ) 
  (h1 : y + 7 = (x - 3)^2) 
  (h2 : x + 7 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 17 := by
sorry

end NUMINAMATH_CALUDE_square_sum_seventeen_l2647_264740


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l2647_264752

theorem conic_section_eccentricity (m : ℝ) : 
  (m^2 = 2 * 8) →
  (∃ (e : ℝ), (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧
    ((m > 0 → e = Real.sqrt 3 / 2) ∧
     (m < 0 → e = Real.sqrt 5)) ∧
    (∀ (x y : ℝ), x^2 + y^2 / m = 1 → 
      (m > 0 → e^2 = 1 - (1 / m)) ∧
      (m < 0 → e^2 = 1 + (1 / m)))) :=
by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l2647_264752


namespace NUMINAMATH_CALUDE_total_tasters_l2647_264767

/-- Represents the number of apple pies Sedrach has -/
def num_pies : ℕ := 13

/-- Represents the number of halves each pie can be divided into -/
def halves_per_pie : ℕ := 2

/-- Represents the number of bite-size samples each half can be split into -/
def samples_per_half : ℕ := 5

/-- Theorem stating the total number of people who can taste Sedrach's apple pies -/
theorem total_tasters : num_pies * halves_per_pie * samples_per_half = 130 := by
  sorry

end NUMINAMATH_CALUDE_total_tasters_l2647_264767


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l2647_264739

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l2647_264739


namespace NUMINAMATH_CALUDE_knights_problem_l2647_264747

/-- Represents the arrangement of knights -/
structure KnightArrangement where
  total : ℕ
  rows : ℕ
  cols : ℕ
  knights_per_row : ℕ
  knights_per_col : ℕ

/-- The conditions of the problem -/
def problem_conditions (k : KnightArrangement) : Prop :=
  k.total = k.rows * k.cols ∧
  k.total - 2 * k.knights_per_row = 24 ∧
  k.total - 2 * k.knights_per_col = 18

/-- The theorem to be proved -/
theorem knights_problem :
  ∀ k : KnightArrangement, problem_conditions k → k.total = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_knights_problem_l2647_264747


namespace NUMINAMATH_CALUDE_five_hundred_billion_scientific_notation_l2647_264773

/-- Express 500 billion in scientific notation -/
theorem five_hundred_billion_scientific_notation :
  (500000000000 : ℝ) = 5 * 10^11 := by
  sorry

end NUMINAMATH_CALUDE_five_hundred_billion_scientific_notation_l2647_264773


namespace NUMINAMATH_CALUDE_min_distance_complex_l2647_264710

theorem min_distance_complex (Z : ℂ) (h : Complex.abs (Z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (W : ℂ), Complex.abs (W + 2 - 2*I) = 1 → Complex.abs (W - 2 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l2647_264710


namespace NUMINAMATH_CALUDE_angle_equality_in_right_triangle_l2647_264741

theorem angle_equality_in_right_triangle (D E F : Real) (angle_D angle_E angle_3 angle_4 : Real) :
  angle_E = 90 →
  angle_D = 70 →
  angle_3 = angle_4 →
  angle_3 + angle_4 = 180 - angle_E - angle_D →
  angle_4 = 45 := by
sorry

end NUMINAMATH_CALUDE_angle_equality_in_right_triangle_l2647_264741


namespace NUMINAMATH_CALUDE_candidate_votes_l2647_264700

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15/100 →
  candidate_percent = 65/100 →
  (1 - invalid_percent) * candidate_percent * total_votes = 309400 := by
  sorry

end NUMINAMATH_CALUDE_candidate_votes_l2647_264700


namespace NUMINAMATH_CALUDE_planar_cube_area_is_600_l2647_264784

/-- The side length of each square in centimeters -/
def side_length : ℝ := 10

/-- The number of faces in a cube -/
def cube_faces : ℕ := 6

/-- The area of the planar figure of a cube in square centimeters -/
def planar_cube_area : ℝ := side_length^2 * cube_faces

/-- Theorem: The area of a planar figure representing a cube, 
    made up of squares with side length 10 cm, is 600 cm² -/
theorem planar_cube_area_is_600 : planar_cube_area = 600 := by
  sorry

end NUMINAMATH_CALUDE_planar_cube_area_is_600_l2647_264784


namespace NUMINAMATH_CALUDE_spheres_in_cone_radius_l2647_264722

/-- A right circular cone -/
structure Cone :=
  (base_radius : ℝ)
  (height : ℝ)

/-- A sphere -/
structure Sphere :=
  (radius : ℝ)

/-- Configuration of four spheres in a cone -/
structure SpheresInCone :=
  (cone : Cone)
  (sphere : Sphere)
  (tangent_to_base : Prop)
  (tangent_to_each_other : Prop)
  (tangent_to_side : Prop)

/-- Theorem stating the radius of spheres in the given configuration -/
theorem spheres_in_cone_radius 
  (config : SpheresInCone)
  (h_base_radius : config.cone.base_radius = 6)
  (h_height : config.cone.height = 15)
  (h_tangent_base : config.tangent_to_base)
  (h_tangent_each_other : config.tangent_to_each_other)
  (h_tangent_side : config.tangent_to_side) :
  config.sphere.radius = 15 / 11 :=
sorry

end NUMINAMATH_CALUDE_spheres_in_cone_radius_l2647_264722


namespace NUMINAMATH_CALUDE_math_team_selection_count_l2647_264780

theorem math_team_selection_count : 
  let total_boys : ℕ := 7
  let total_girls : ℕ := 10
  let team_size : ℕ := 5
  let boys_in_team : ℕ := 2
  let girls_in_team : ℕ := 3
  (Nat.choose total_boys boys_in_team) * (Nat.choose total_girls girls_in_team) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_math_team_selection_count_l2647_264780


namespace NUMINAMATH_CALUDE_june_christopher_difference_l2647_264795

/-- The length of Christopher's sword in inches -/
def christopher_sword : ℕ := 15

/-- The length of Jameson's sword in inches -/
def jameson_sword : ℕ := 2 * christopher_sword + 3

/-- The length of June's sword in inches -/
def june_sword : ℕ := jameson_sword + 5

/-- Theorem: June's sword is 23 inches longer than Christopher's sword -/
theorem june_christopher_difference : june_sword - christopher_sword = 23 := by
  sorry

end NUMINAMATH_CALUDE_june_christopher_difference_l2647_264795


namespace NUMINAMATH_CALUDE_notebook_final_price_l2647_264737

def initial_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

def price_after_first_discount : ℝ := initial_price * (1 - first_discount_rate)
def final_price : ℝ := price_after_first_discount * (1 - second_discount_rate)

theorem notebook_final_price : final_price = 9 := by
  sorry

end NUMINAMATH_CALUDE_notebook_final_price_l2647_264737


namespace NUMINAMATH_CALUDE_painting_fraction_l2647_264730

def total_students : ℕ := 50
def field_fraction : ℚ := 1 / 5
def classroom_left : ℕ := 10

theorem painting_fraction :
  (total_students - (field_fraction * total_students).num - classroom_left) / total_students = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_painting_fraction_l2647_264730


namespace NUMINAMATH_CALUDE_correct_sunset_time_l2647_264788

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  ⟨newHours % 24, newMinutes, by sorry, by sorry⟩

theorem correct_sunset_time :
  let sunrise : Time := ⟨7, 12, by sorry, by sorry⟩
  let incorrectDaylight : Nat := 11 * 60 + 15 -- 11 hours and 15 minutes in minutes
  let calculatedSunset := addMinutes sunrise incorrectDaylight
  calculatedSunset.hours = 18 ∧ calculatedSunset.minutes = 27 :=
by sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l2647_264788


namespace NUMINAMATH_CALUDE_truth_values_equivalence_l2647_264791

theorem truth_values_equivalence (p q : Prop) 
  (h1 : p ∨ q) (h2 : ¬(p ∧ q)) : p ↔ ¬q := by
  sorry

end NUMINAMATH_CALUDE_truth_values_equivalence_l2647_264791


namespace NUMINAMATH_CALUDE_vector_operation_proof_l2647_264744

/-- Prove that the vector operation (3, -8) - 3(2, -5) + (-1, 4) equals (-4, 11) -/
theorem vector_operation_proof :
  let v1 : Fin 2 → ℝ := ![3, -8]
  let v2 : Fin 2 → ℝ := ![2, -5]
  let v3 : Fin 2 → ℝ := ![-1, 4]
  v1 - 3 • v2 + v3 = ![-4, 11] := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l2647_264744


namespace NUMINAMATH_CALUDE_cross_in_square_side_length_l2647_264734

/-- Represents a cross shape inside a square -/
structure CrossInSquare where
  a : ℝ  -- Side length of the largest square
  area_cross : ℝ -- Area of the cross

/-- The area of the cross is equal to the sum of areas of its component squares -/
def cross_area_equation (c : CrossInSquare) : Prop :=
  c.area_cross = 2 * (c.a / 2)^2 + 2 * (c.a / 4)^2

/-- Theorem stating that if the area of the cross is 810 cm², then the side length of the largest square is 36 cm -/
theorem cross_in_square_side_length 
  (c : CrossInSquare) 
  (h1 : c.area_cross = 810) 
  (h2 : cross_area_equation c) : 
  c.a = 36 := by
sorry

end NUMINAMATH_CALUDE_cross_in_square_side_length_l2647_264734


namespace NUMINAMATH_CALUDE_equation_solution_l2647_264701

theorem equation_solution : ∃! x : ℚ, (7 * x - 2) / (x + 4) - 4 / (x + 4) = 2 / (x + 4) ∧ x = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2647_264701


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2647_264766

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2647_264766


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l2647_264761

theorem unique_two_digit_integer (u : ℕ) : 
  (u ≥ 10 ∧ u ≤ 99) ∧ (17 * u) % 100 = 45 ↔ u = 85 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l2647_264761


namespace NUMINAMATH_CALUDE_water_scooped_out_l2647_264792

theorem water_scooped_out (total_weight : ℝ) (alcohol_concentration : ℝ) :
  total_weight = 10 ∧ alcohol_concentration = 0.75 →
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 10 ∧ x / total_weight = alcohol_concentration ∧ x = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_water_scooped_out_l2647_264792


namespace NUMINAMATH_CALUDE_least_perimeter_of_special_triangle_l2647_264727

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The condition for a triangle to be non-equilateral -/
def is_non_equilateral (t : IntTriangle) : Prop :=
  t.a ≠ t.b ∨ t.b ≠ t.c ∨ t.c ≠ t.a

/-- The condition for points D, C, E, G to be concyclic -/
def is_concyclic (t : IntTriangle) : Prop :=
  -- This is a placeholder for the actual concyclic condition
  -- In reality, this would involve more complex geometric relations
  true

/-- The perimeter of a triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

theorem least_perimeter_of_special_triangle :
  ∃ (t : IntTriangle),
    is_non_equilateral t ∧
    is_concyclic t ∧
    (∀ (s : IntTriangle), is_non_equilateral s → is_concyclic s → perimeter t ≤ perimeter s) ∧
    perimeter t = 37 := by
  sorry

end NUMINAMATH_CALUDE_least_perimeter_of_special_triangle_l2647_264727


namespace NUMINAMATH_CALUDE_angle_covered_in_three_layers_l2647_264721

theorem angle_covered_in_three_layers 
  (total_angle : ℝ) 
  (sum_of_angles : ℝ) 
  (angle_three_layers : ℝ) 
  (h1 : total_angle = 90) 
  (h2 : sum_of_angles = 290) 
  (h3 : angle_three_layers * 3 + (total_angle - angle_three_layers) * 2 = sum_of_angles) :
  angle_three_layers = 20 := by
sorry

end NUMINAMATH_CALUDE_angle_covered_in_three_layers_l2647_264721


namespace NUMINAMATH_CALUDE_work_completion_time_l2647_264732

-- Define the work rate of A
def work_rate_A : ℚ := 1 / 60

-- Define the work done by A in 15 days
def work_done_A : ℚ := 15 * work_rate_A

-- Define the remaining work after A's 15 days
def remaining_work : ℚ := 1 - work_done_A

-- Define B's work rate based on completing the remaining work in 30 days
def work_rate_B : ℚ := remaining_work / 30

-- Define the combined work rate of A and B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Theorem to prove
theorem work_completion_time : (1 : ℚ) / combined_work_rate = 24 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2647_264732


namespace NUMINAMATH_CALUDE_gold_beads_undetermined_l2647_264736

/-- Represents the types of beads used in the corset --/
inductive BeadType
  | Purple
  | Blue
  | Gold

/-- Represents a row of beads --/
structure BeadRow where
  beadType : BeadType
  beadsPerRow : ℕ
  rowCount : ℕ

/-- Represents the corset design --/
structure CorsetDesign where
  purpleRows : BeadRow
  blueRows : BeadRow
  goldBeads : ℕ
  totalCost : ℚ

def carlyDesign : CorsetDesign :=
  { purpleRows := { beadType := BeadType.Purple, beadsPerRow := 20, rowCount := 50 }
  , blueRows := { beadType := BeadType.Blue, beadsPerRow := 18, rowCount := 40 }
  , goldBeads := 0  -- This is what we're trying to determine
  , totalCost := 180 }

/-- The theorem stating that the number of gold beads cannot be determined --/
theorem gold_beads_undetermined (design : CorsetDesign) : 
  design.purpleRows.beadsPerRow = carlyDesign.purpleRows.beadsPerRow ∧ 
  design.purpleRows.rowCount = carlyDesign.purpleRows.rowCount ∧
  design.blueRows.beadsPerRow = carlyDesign.blueRows.beadsPerRow ∧
  design.blueRows.rowCount = carlyDesign.blueRows.rowCount ∧
  design.totalCost = carlyDesign.totalCost →
  ∃ (x y : ℕ), x ≠ y ∧ 
    (∃ (design1 design2 : CorsetDesign), 
      design1.goldBeads = x ∧ 
      design2.goldBeads = y ∧
      design1.purpleRows = design.purpleRows ∧
      design1.blueRows = design.blueRows ∧
      design1.totalCost = design.totalCost ∧
      design2.purpleRows = design.purpleRows ∧
      design2.blueRows = design.blueRows ∧
      design2.totalCost = design.totalCost) :=
by
  sorry

end NUMINAMATH_CALUDE_gold_beads_undetermined_l2647_264736


namespace NUMINAMATH_CALUDE_negative_comparison_l2647_264781

theorem negative_comparison : -0.5 > -0.7 := by
  sorry

end NUMINAMATH_CALUDE_negative_comparison_l2647_264781


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l2647_264774

theorem quadratic_function_m_value :
  ∃! m : ℝ, (abs (m - 1) = 2) ∧ (m - 3 ≠ 0) ∧ (m = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_value_l2647_264774


namespace NUMINAMATH_CALUDE_window_purchase_savings_l2647_264703

/-- Represents the window purchase scenario --/
structure WindowPurchase where
  regularPrice : ℕ
  freeWindows : ℕ
  purchaseThreshold : ℕ
  daveNeeds : ℕ
  dougNeeds : ℕ

/-- Calculates the cost for a given number of windows --/
def calculateCost (wp : WindowPurchase) (windows : ℕ) : ℕ :=
  let freeGroups := windows / wp.purchaseThreshold
  let paidWindows := windows - (freeGroups * wp.freeWindows)
  paidWindows * wp.regularPrice

/-- Calculates the savings when purchasing together vs separately --/
def calculateSavings (wp : WindowPurchase) : ℕ :=
  let separateCost := calculateCost wp wp.daveNeeds + calculateCost wp wp.dougNeeds
  let jointCost := calculateCost wp (wp.daveNeeds + wp.dougNeeds)
  separateCost - jointCost

/-- The main theorem stating the savings amount --/
theorem window_purchase_savings (wp : WindowPurchase) 
  (h1 : wp.regularPrice = 120)
  (h2 : wp.freeWindows = 2)
  (h3 : wp.purchaseThreshold = 6)
  (h4 : wp.daveNeeds = 12)
  (h5 : wp.dougNeeds = 9) :
  calculateSavings wp = 360 := by
  sorry


end NUMINAMATH_CALUDE_window_purchase_savings_l2647_264703


namespace NUMINAMATH_CALUDE_indeterminate_equation_solution_l2647_264763

theorem indeterminate_equation_solution (a b : ℤ) :
  ∃ (x y z u v w t : ℤ),
    x^4 + y^4 + z^4 = u^2 + v^2 + w^2 + t^2 ∧
    x = a ∧
    y = b ∧
    z = a + b ∧
    u = a^2 + a*b + b^2 ∧
    v = a*b ∧
    w = a*b*(a + b) ∧
    t = b*(a + b) := by
  sorry

end NUMINAMATH_CALUDE_indeterminate_equation_solution_l2647_264763


namespace NUMINAMATH_CALUDE_opposite_of_one_seventh_l2647_264704

theorem opposite_of_one_seventh :
  ∀ x : ℚ, x + (1 / 7) = 0 ↔ x = -(1 / 7) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_seventh_l2647_264704


namespace NUMINAMATH_CALUDE_tournament_handshakes_count_l2647_264776

/-- The number of unique handshakes in a tournament with 4 teams of 2 players each,
    where each player shakes hands once with every other player except their partner. -/
def tournament_handshakes : ℕ :=
  let total_players : ℕ := 4 * 2
  let handshakes_per_player : ℕ := total_players - 2
  (total_players * handshakes_per_player) / 2

theorem tournament_handshakes_count : tournament_handshakes = 24 := by
  sorry

end NUMINAMATH_CALUDE_tournament_handshakes_count_l2647_264776


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2647_264777

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2647_264777


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2647_264772

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1/x + 4/y ≥ 9 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2647_264772


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2647_264715

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2647_264715


namespace NUMINAMATH_CALUDE_race_coin_problem_l2647_264716

theorem race_coin_problem (x y : ℕ) (h1 : x > y) (h2 : y > 0) : 
  (∃ n : ℕ, n > 2 ∧ 
   (n - 2) * x + 2 * y = 42 ∧ 
   2 * x + (n - 2) * y = 35) → 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_race_coin_problem_l2647_264716


namespace NUMINAMATH_CALUDE_xyz_product_l2647_264798

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 168)
  (h2 : y * (z + x) = 186)
  (h3 : z * (x + y) = 194) :
  x * y * z = 860 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l2647_264798


namespace NUMINAMATH_CALUDE_car_resale_gain_percentage_car_resale_specific_case_l2647_264720

/-- Calculates the gain percentage when reselling a car --/
theorem car_resale_gain_percentage 
  (original_price : ℝ) 
  (loss_percentage : ℝ) 
  (resale_price : ℝ) : ℝ :=
  let first_sale_price := original_price * (1 - loss_percentage / 100)
  let gain := resale_price - first_sale_price
  let gain_percentage := (gain / first_sale_price) * 100
  gain_percentage

/-- Proves that the gain percentage is approximately 3.55% for the given scenario --/
theorem car_resale_specific_case : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |car_resale_gain_percentage 52941.17647058824 15 54000 - 3.55| < ε :=
sorry

end NUMINAMATH_CALUDE_car_resale_gain_percentage_car_resale_specific_case_l2647_264720


namespace NUMINAMATH_CALUDE_emily_furniture_assembly_time_l2647_264758

/-- Calculates the total assembly time for furniture -/
def total_assembly_time (
  num_chairs : ℕ) (chair_time : ℕ)
  (num_tables : ℕ) (table_time : ℕ)
  (num_shelves : ℕ) (shelf_time : ℕ)
  (num_wardrobes : ℕ) (wardrobe_time : ℕ) : ℕ :=
  num_chairs * chair_time +
  num_tables * table_time +
  num_shelves * shelf_time +
  num_wardrobes * wardrobe_time

/-- Proves that the total assembly time for Emily's furniture is 137 minutes -/
theorem emily_furniture_assembly_time :
  total_assembly_time 4 8 2 15 3 10 1 45 = 137 := by
  sorry


end NUMINAMATH_CALUDE_emily_furniture_assembly_time_l2647_264758


namespace NUMINAMATH_CALUDE_largest_gold_coins_distribution_l2647_264748

theorem largest_gold_coins_distribution (n : ℕ) : 
  n > 50 ∧ n < 150 ∧ 
  ∃ (k : ℕ), n = 7 * k + 2 ∧
  (∀ m : ℕ, m > n → ¬(∃ j : ℕ, m = 7 * j + 2)) →
  n = 149 := by
sorry

end NUMINAMATH_CALUDE_largest_gold_coins_distribution_l2647_264748


namespace NUMINAMATH_CALUDE_factorial_divisibility_l2647_264753

theorem factorial_divisibility (m : ℕ) (h : m > 1) :
  (m - 1).factorial % m = 0 ↔ ¬ Nat.Prime m := by sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l2647_264753


namespace NUMINAMATH_CALUDE_distribute_18_balls_5_boxes_l2647_264778

/-- The number of ways to distribute n identical balls into k distinct boxes,
    with each box containing at least m balls. -/
def distribute_balls (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

theorem distribute_18_balls_5_boxes :
  distribute_balls 18 5 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_distribute_18_balls_5_boxes_l2647_264778


namespace NUMINAMATH_CALUDE_function_value_at_negative_half_l2647_264742

theorem function_value_at_negative_half (a : ℝ) (f : ℝ → ℝ) :
  0 < a →
  a ≠ 1 →
  (∀ x, f x = a^x) →
  f 2 = 81 →
  f (-1/2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_half_l2647_264742


namespace NUMINAMATH_CALUDE_estate_division_l2647_264764

theorem estate_division (E : ℝ) 
  (h1 : ∃ (x : ℝ), 6 * x = 2/3 * E)  -- Two sons and daughter receive 2/3 of estate in 3:2:1 ratio
  (h2 : ∃ (x : ℝ), 3 * x = E - (9 * x + 750))  -- Wife's share is 3x, where x is daughter's share
  (h3 : 750 ≤ E)  -- Butler's share is $750
  : E = 7500 := by
  sorry

end NUMINAMATH_CALUDE_estate_division_l2647_264764


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2647_264731

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2647_264731


namespace NUMINAMATH_CALUDE_power_and_arithmetic_equality_l2647_264775

theorem power_and_arithmetic_equality : (-1)^100 * 5 + (-2)^3 / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_and_arithmetic_equality_l2647_264775


namespace NUMINAMATH_CALUDE_equation_describes_ellipse_l2647_264771

def is_ellipse (f₁ f₂ : ℝ × ℝ) (c : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
               Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = c

theorem equation_describes_ellipse :
  is_ellipse (0, 2) (6, -4) 12 := by sorry

end NUMINAMATH_CALUDE_equation_describes_ellipse_l2647_264771


namespace NUMINAMATH_CALUDE_unique_pair_satisfying_conditions_l2647_264765

theorem unique_pair_satisfying_conditions :
  ∃! (n p : ℕ+), 
    (Nat.Prime p.val) ∧ 
    (-↑n : ℤ) ≤ 2 * ↑p ∧
    (↑p - 1 : ℤ) ^ n.val + 1 ∣ ↑n ^ (p.val - 1) ∧
    n = 3 ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_satisfying_conditions_l2647_264765


namespace NUMINAMATH_CALUDE_b_equals_one_l2647_264743

-- Define the variables
variable (a b y : ℝ)

-- Define the conditions
def condition1 : Prop := |b - y| = b + y - a
def condition2 : Prop := |b + y| = b + a

-- State the theorem
theorem b_equals_one (h1 : condition1 a b y) (h2 : condition2 a b y) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_b_equals_one_l2647_264743


namespace NUMINAMATH_CALUDE_unknown_number_is_nine_l2647_264756

def first_number : ℝ := 4.2

def second_number : ℝ := first_number + 2

def third_number : ℝ := first_number + 4

def unknown_number : ℝ := 9 * first_number - 2 * third_number - 2 * second_number

theorem unknown_number_is_nine : unknown_number = 9 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_is_nine_l2647_264756


namespace NUMINAMATH_CALUDE_carol_meets_alice_l2647_264783

/-- Alice's speed in miles per hour -/
def alice_speed : ℝ := 4

/-- Carol's speed in miles per hour -/
def carol_speed : ℝ := 6

/-- Initial distance between Carol and Alice in miles -/
def initial_distance : ℝ := 5

/-- Time in minutes for Carol to meet Alice -/
def meeting_time : ℝ := 30

theorem carol_meets_alice : 
  initial_distance / (alice_speed + carol_speed) * 60 = meeting_time := by
  sorry

end NUMINAMATH_CALUDE_carol_meets_alice_l2647_264783


namespace NUMINAMATH_CALUDE_triangle_area_determines_p_l2647_264733

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    if the area of triangle ABC is 36, then p = 12.75 -/
theorem triangle_area_determines_p :
  ∀ p : ℝ,
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  triangle_area = 36 → p = 12.75 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_determines_p_l2647_264733


namespace NUMINAMATH_CALUDE_orange_weight_problem_l2647_264712

theorem orange_weight_problem (initial_water_concentration : Real)
                               (water_decrease : Real)
                               (new_weight : Real) :
  initial_water_concentration = 0.95 →
  water_decrease = 0.05 →
  new_weight = 25 →
  ∃ (initial_weight : Real),
    initial_weight = 50 ∧
    (1 - initial_water_concentration) * initial_weight =
    (1 - (initial_water_concentration - water_decrease)) * new_weight :=
by sorry

end NUMINAMATH_CALUDE_orange_weight_problem_l2647_264712


namespace NUMINAMATH_CALUDE_mrs_hilt_daily_reading_l2647_264789

/-- The number of books Mrs. Hilt read in one week -/
def books_per_week : ℕ := 14

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of books Mrs. Hilt read per day -/
def books_per_day : ℚ := books_per_week / days_in_week

theorem mrs_hilt_daily_reading : books_per_day = 2 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_daily_reading_l2647_264789


namespace NUMINAMATH_CALUDE_octagon_diagonals_l2647_264717

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- The number of vertices in an octagon -/
def octagon_vertices : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_vertices = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l2647_264717


namespace NUMINAMATH_CALUDE_marble_probability_l2647_264714

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 50) (h2 : blue = 5) (h3 : red = 9) :
  (red + (total - blue - red)) / total = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l2647_264714


namespace NUMINAMATH_CALUDE_angle_B_is_140_degrees_l2647_264793

/-- A quadrilateral with angles A, B, C, and D -/
structure Quadrilateral where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angleD : ℝ

/-- The theorem stating that if the sum of angles A, B, and C in a quadrilateral is 220°, 
    then angle B is 140° -/
theorem angle_B_is_140_degrees (q : Quadrilateral) 
    (h : q.angleA + q.angleB + q.angleC = 220) : q.angleB = 140 := by
  sorry


end NUMINAMATH_CALUDE_angle_B_is_140_degrees_l2647_264793


namespace NUMINAMATH_CALUDE_inverse_equals_k_times_self_l2647_264726

def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, d]

theorem inverse_equals_k_times_self (d k : ℝ) :
  A d * (A d)⁻¹ = 1 ∧ (A d)⁻¹ = k • (A d) → d = -3 ∧ k = 1/33 := by
  sorry

end NUMINAMATH_CALUDE_inverse_equals_k_times_self_l2647_264726


namespace NUMINAMATH_CALUDE_system_solution_l2647_264770

theorem system_solution : 
  ∀ x y z : ℕ, 
    (2 * x^2 + 30 * y^2 + 3 * z^2 + 12 * x * y + 12 * y * z = 308 ∧
     2 * x^2 + 6 * y^2 - 3 * z^2 + 12 * x * y - 12 * y * z = 92) →
    ((x = 7 ∧ y = 1 ∧ z = 4) ∨ (x = 4 ∧ y = 2 ∧ z = 2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2647_264770


namespace NUMINAMATH_CALUDE_vincent_outer_space_books_l2647_264768

/-- The number of books about outer space Vincent bought -/
def outer_space_books : ℕ := 1

/-- The number of books about animals Vincent bought -/
def animal_books : ℕ := 10

/-- The number of books about trains Vincent bought -/
def train_books : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 16

/-- The total amount spent on books in dollars -/
def total_spent : ℕ := 224

theorem vincent_outer_space_books :
  outer_space_books = 1 ∧
  animal_books * book_cost + outer_space_books * book_cost + train_books * book_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_vincent_outer_space_books_l2647_264768


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2647_264706

/-- Given a quadratic inequality x^2 + bx - a < 0 with solution set {x | -2 < x < 3}, prove that a + b = 5 -/
theorem quadratic_inequality_solution (a b : ℝ) 
  (h : ∀ x, x^2 + b*x - a < 0 ↔ -2 < x ∧ x < 3) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2647_264706


namespace NUMINAMATH_CALUDE_min_value_of_f_l2647_264750

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2024)^2

-- State the theorem
theorem min_value_of_f :
  (∀ x : ℝ, f (x + 2023) = x^2 - 2*x + 1) →
  (∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, f x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2647_264750


namespace NUMINAMATH_CALUDE_unique_stamp_denomination_l2647_264754

/-- A function that determines if a postage value can be formed with given stamp denominations -/
def can_form_postage (n : ℕ) (postage : ℕ) : Prop :=
  ∃ (a b c : ℕ), postage = 10 * a + n * b + (n + 1) * c

/-- The main theorem stating that 16 is the unique positive integer satisfying the conditions -/
theorem unique_stamp_denomination : 
  ∃! (n : ℕ), n > 0 ∧ 
    (¬ can_form_postage n 120) ∧ 
    (∀ m > 120, can_form_postage n m) ∧
    n = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_stamp_denomination_l2647_264754


namespace NUMINAMATH_CALUDE_inverse_variation_doubling_inverse_variation_example_l2647_264785

/-- Given two quantities that vary inversely, if one quantity doubles, the other halves -/
theorem inverse_variation_doubling (a b c d : ℝ) (h1 : a * b = c * d) (h2 : c = 2 * a) :
  d = b / 2 := by
  sorry

/-- When a and b vary inversely, if b = 0.5 when a = 800, then b = 0.25 when a = 1600 -/
theorem inverse_variation_example :
  ∃ (k : ℝ), (800 * 0.5 = k) ∧ (1600 * 0.25 = k) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_doubling_inverse_variation_example_l2647_264785


namespace NUMINAMATH_CALUDE_vasya_gift_choices_l2647_264786

theorem vasya_gift_choices (n_cars : ℕ) (n_sets : ℕ) : 
  n_cars = 7 → n_sets = 5 → (n_cars.choose 2) + (n_sets.choose 2) + n_cars * n_sets = 66 :=
by
  sorry

end NUMINAMATH_CALUDE_vasya_gift_choices_l2647_264786


namespace NUMINAMATH_CALUDE_total_sleep_week_is_366_l2647_264735

def connor_sleep : ℕ := 6
def luke_sleep : ℕ := connor_sleep + 2
def emma_sleep : ℕ := connor_sleep - 1
def ava_sleep (day : ℕ) : ℕ := 5 + (day - 1) / 2
def puppy_sleep : ℕ := 2 * luke_sleep
def cat_sleep : ℕ := 4 + 7

def total_sleep_week : ℕ :=
  7 * connor_sleep +
  7 * luke_sleep +
  7 * emma_sleep +
  (ava_sleep 1 + ava_sleep 2 + ava_sleep 3 + ava_sleep 4 + ava_sleep 5 + ava_sleep 6 + ava_sleep 7) +
  7 * puppy_sleep +
  7 * cat_sleep

theorem total_sleep_week_is_366 : total_sleep_week = 366 := by
  sorry

end NUMINAMATH_CALUDE_total_sleep_week_is_366_l2647_264735


namespace NUMINAMATH_CALUDE_ap_square_cube_implies_sixth_power_l2647_264709

/-- An arithmetic progression is represented by its first term and common difference -/
structure ArithmeticProgression where
  first_term : ℕ
  common_diff : ℕ

/-- Check if a number is in the arithmetic progression -/
def ArithmeticProgression.contains (ap : ArithmeticProgression) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = ap.first_term + k * ap.common_diff

/-- An arithmetic progression contains a perfect square -/
def contains_square (ap : ArithmeticProgression) : Prop :=
  ∃ x : ℕ, ap.contains (x^2)

/-- An arithmetic progression contains a perfect cube -/
def contains_cube (ap : ArithmeticProgression) : Prop :=
  ∃ y : ℕ, ap.contains (y^3)

/-- An arithmetic progression contains a sixth power -/
def contains_sixth_power (ap : ArithmeticProgression) : Prop :=
  ∃ z : ℕ, ap.contains (z^6)

/-- Main theorem: If an AP contains a square and a cube, it contains a sixth power -/
theorem ap_square_cube_implies_sixth_power (ap : ArithmeticProgression) 
  (h1 : ap.first_term > 0) 
  (h2 : contains_square ap) 
  (h3 : contains_cube ap) : 
  contains_sixth_power ap := by
  sorry

end NUMINAMATH_CALUDE_ap_square_cube_implies_sixth_power_l2647_264709


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2647_264787

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2*y + 2*x*y = 8) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + 2*b + 2*a*b = 8 → x + 2*y ≤ a + 2*b ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ + 2*x₀*y₀ = 8 ∧ x₀ + 2*y₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2647_264787


namespace NUMINAMATH_CALUDE_gcf_360_150_l2647_264745

theorem gcf_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_360_150_l2647_264745
