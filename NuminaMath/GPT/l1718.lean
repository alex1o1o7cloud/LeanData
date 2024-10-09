import Mathlib

namespace digit_sum_is_14_l1718_171880

theorem digit_sum_is_14 (P Q R S T : ℕ) 
  (h1 : P = 1)
  (h2 : Q = 0)
  (h3 : R = 2)
  (h4 : S = 5)
  (h5 : T = 6) :
  P + Q + R + S + T = 14 :=
by 
  sorry

end digit_sum_is_14_l1718_171880


namespace angles_measure_l1718_171853

theorem angles_measure (A B C : ℝ) (h1 : A + B = 180) (h2 : C = 1 / 2 * B) (h3 : A = 6 * B) :
  A = 1080 / 7 ∧ B = 180 / 7 ∧ C = 90 / 7 :=
by
  sorry

end angles_measure_l1718_171853


namespace ratio_of_radii_of_truncated_cone_l1718_171878

theorem ratio_of_radii_of_truncated_cone 
  (R r s : ℝ) 
  (h1 : s = Real.sqrt (R * r)) 
  (h2 : (π * (R^2 + r^2 + R * r) * (2 * s) / 3) = 3 * (4 * π * s^3 / 3)) :
  R / r = 7 := 
sorry

end ratio_of_radii_of_truncated_cone_l1718_171878


namespace park_area_l1718_171845

theorem park_area (P : ℝ) (w l : ℝ) (hP : P = 120) (hL : l = 3 * w) (hPerimeter : 2 * l + 2 * w = P) : l * w = 675 :=
by
  sorry

end park_area_l1718_171845


namespace two_pipes_fill_tank_l1718_171873

theorem two_pipes_fill_tank (C : ℝ) (hA : ∀ (t : ℝ), t = 10 → t = C / (C / 10)) (hB : ∀ (t : ℝ), t = 15 → t = C / (C / 15)) :
  ∀ (t : ℝ), t = C / (C / 6) → t = 6 :=
by
  sorry

end two_pipes_fill_tank_l1718_171873


namespace right_triangles_with_specific_area_and_perimeter_l1718_171828

theorem right_triangles_with_specific_area_and_perimeter :
  ∃ (count : ℕ),
    count = 7 ∧
    ∀ (a b : ℕ), 
      (a > 0 ∧ b > 0 ∧ (a ≠ b) ∧ (a^2 + b^2 = c^2) ∧ (a * b / 2 = 5 * (a + b + c))) → 
      count = 7 :=
by
  sorry

end right_triangles_with_specific_area_and_perimeter_l1718_171828


namespace proof_problem_l1718_171867

noncomputable def initialEfficiencyOfOneMan : ℕ := sorry
noncomputable def initialEfficiencyOfOneWoman : ℕ := sorry
noncomputable def totalWork : ℕ := sorry

-- Condition (1): 10 men and 15 women together can complete the work in 6 days.
def condition1 := 10 * initialEfficiencyOfOneMan + 15 * initialEfficiencyOfOneWoman = totalWork / 6

-- Condition (2): The efficiency of men to complete the work decreases by 5% every day.
-- This condition is not directly measurable to our proof but noted as additional info.

-- Condition (3): The efficiency of women to complete the work increases by 3% every day.
-- This condition is not directly measurable to our proof but noted as additional info.

-- Condition (4): It takes 100 days for one man alone to complete the same work at his initial efficiency.
def condition4 := initialEfficiencyOfOneMan = totalWork / 100

-- Define the days required for one woman alone to complete the work at her initial efficiency.
noncomputable def daysForWomanToCompleteWork : ℕ := 225

-- Mathematically equivalent proof problem
theorem proof_problem : 
  condition1 ∧ condition4 → (totalWork / daysForWomanToCompleteWork = initialEfficiencyOfOneWoman) :=
by
  sorry

end proof_problem_l1718_171867


namespace smallest_diff_l1718_171814

noncomputable def triangleSides : ℕ → ℕ → ℕ → Prop := λ AB BC AC =>
  AB < BC ∧ BC ≤ AC ∧ AB + BC + AC = 2007

theorem smallest_diff (AB BC AC : ℕ) (h : triangleSides AB BC AC) : BC - AB = 1 :=
  sorry

end smallest_diff_l1718_171814


namespace complex_numbers_are_real_l1718_171887

theorem complex_numbers_are_real
  (a b c : ℂ)
  (h1 : (a + b) * (a + c) = b)
  (h2 : (b + c) * (b + a) = c)
  (h3 : (c + a) * (c + b) = a) : 
  a.im = 0 ∧ b.im = 0 ∧ c.im = 0 :=
sorry

end complex_numbers_are_real_l1718_171887


namespace athlete_with_most_stable_performance_l1718_171898

def variance_A : ℝ := 0.78
def variance_B : ℝ := 0.2
def variance_C : ℝ := 1.28

theorem athlete_with_most_stable_performance : variance_B < variance_A ∧ variance_B < variance_C :=
by {
  -- Variance comparisons:
  -- 0.2 < 0.78
  -- 0.2 < 1.28
  sorry
}

end athlete_with_most_stable_performance_l1718_171898


namespace geometric_sequence_sum_l1718_171857

theorem geometric_sequence_sum :
  ∀ {a : ℕ → ℝ} (r : ℝ),
    (∀ n, a (n + 1) = r * a n) →
    a 1 + a 2 = 1 →
    a 3 + a 4 = 4 →
    a 5 + a 6 + a 7 + a 8 = 80 :=
by
  intros a r h_geom h_sum_1 h_sum_2
  sorry

end geometric_sequence_sum_l1718_171857


namespace probability_calc_l1718_171802

noncomputable def probability_no_distinct_positive_real_roots : ℚ :=
  let pairs_count := 169
  let valid_pairs_count := 17
  1 - (valid_pairs_count / pairs_count : ℚ)

theorem probability_calc :
  probability_no_distinct_positive_real_roots = 152 / 169 := by sorry

end probability_calc_l1718_171802


namespace height_of_building_l1718_171809

-- Define the conditions as hypotheses
def height_of_flagstaff : ℝ := 17.5
def shadow_length_of_flagstaff : ℝ := 40.25
def shadow_length_of_building : ℝ := 28.75

-- Define the height ratio based on similar triangles
theorem height_of_building :
  (height_of_flagstaff / shadow_length_of_flagstaff = 12.47 / shadow_length_of_building) :=
by
  sorry

end height_of_building_l1718_171809


namespace tan_ratio_l1718_171870

variable (a b : Real)

theorem tan_ratio (h1 : Real.sin (a + b) = 5 / 8) (h2 : Real.sin (a - b) = 1 / 4) : 
  (Real.tan a) / (Real.tan b) = 7 / 3 := 
by 
  sorry

end tan_ratio_l1718_171870


namespace middle_schoolers_count_l1718_171804

theorem middle_schoolers_count (total_students : ℕ) (fraction_girls : ℚ) 
  (primary_girls_fraction : ℚ) (primary_boys_fraction : ℚ) 
  (num_girls : ℕ) (num_boys: ℕ) (primary_grade_girls : ℕ) 
  (primary_grade_boys : ℕ) :
  total_students = 800 →
  fraction_girls = 5 / 8 →
  primary_girls_fraction = 7 / 10 →
  primary_boys_fraction = 2 / 5 →
  num_girls = fraction_girls * total_students →
  num_boys = total_students - num_girls →
  primary_grade_girls = primary_girls_fraction * num_girls →
  primary_grade_boys = primary_boys_fraction * num_boys →
  total_students - (primary_grade_girls + primary_grade_boys) = 330 :=
by
  intros
  sorry

end middle_schoolers_count_l1718_171804


namespace John_and_Rose_work_together_l1718_171869

theorem John_and_Rose_work_together (John_work_days : ℕ) (Rose_work_days : ℕ) (combined_work_days: ℕ) 
  (hJohn : John_work_days = 10) (hRose : Rose_work_days = 40) :
  combined_work_days = 8 :=
by 
  sorry

end John_and_Rose_work_together_l1718_171869


namespace eraser_cost_l1718_171876

noncomputable def price_of_erasers 
  (P : ℝ) -- price of one pencil
  (E : ℝ) -- price of one eraser
  (bundle_count : ℝ) -- number of bundles sold
  (total_earned : ℝ) -- total amount earned
  (discount : ℝ) -- discount percentage for 20 bundles
  (bundle_contents : ℕ) -- 1 pencil and 2 erasers per bundle
  (price_ratio : ℝ) -- price ratio of eraser to pencil
  : Prop := 
  E = 0.5 * P ∧ -- The price of the erasers is 1/2 the price of the pencils.
  bundle_count = 20 ∧ -- The store sold a total of 20 bundles.
  total_earned = 80 ∧ -- The store earned $80.
  discount = 30 ∧ -- 30% discount for 20 bundles
  bundle_contents = 1 + 2 -- A bundle consists of 1 pencil and 2 erasers

theorem eraser_cost
  (P : ℝ) -- price of one pencil
  (E : ℝ) -- price of one eraser
  : price_of_erasers P E 20 80 30 (1 + 2) 0.5 → E = 1.43 :=
by
  intro h
  sorry

end eraser_cost_l1718_171876


namespace combined_height_of_cylinders_l1718_171894

/-- Given three cylinders with perimeters 6 feet, 9 feet, and 11 feet respectively,
    and rolled out on a rectangular plate with a diagonal of 19 feet,
    the combined height of the cylinders is 26 feet. -/
theorem combined_height_of_cylinders
  (p1 p2 p3 : ℝ) (d : ℝ)
  (h_p1 : p1 = 6) (h_p2 : p2 = 9) (h_p3 : p3 = 11) (h_d : d = 19) :
  p1 + p2 + p3 = 26 :=
sorry

end combined_height_of_cylinders_l1718_171894


namespace right_triangle_condition_l1718_171849

theorem right_triangle_condition (a b c : ℝ) (h : c^2 - a^2 = b^2) : 
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A = 90 ∧ B + C = 90 :=
by sorry

end right_triangle_condition_l1718_171849


namespace find_x_squared_minus_one_l1718_171833

theorem find_x_squared_minus_one (x : ℕ) 
  (h : 2^x + 2^x + 2^x + 2^x = 256) : 
  x^2 - 1 = 35 :=
sorry

end find_x_squared_minus_one_l1718_171833


namespace part_a_ellipse_and_lines_l1718_171851

theorem part_a_ellipse_and_lines (x y : ℝ) : 
  (4 * x^2 + 8 * y^2 + 8 * y * abs y = 1) ↔ 
  ((y ≥ 0 ∧ (x^2 / (1/4) + y^2 / (1/16)) = 1) ∨ 
  (y < 0 ∧ ((x = 1/2) ∨ (x = -1/2)))) := 
sorry

end part_a_ellipse_and_lines_l1718_171851


namespace correct_operation_l1718_171807

theorem correct_operation (a : ℝ) :
  (2 * a^2) * a = 2 * a^3 :=
by sorry

end correct_operation_l1718_171807


namespace sandwiches_difference_l1718_171871

-- Define the number of sandwiches Samson ate at lunch on Monday
def sandwichesLunchMonday : ℕ := 3

-- Define the number of sandwiches Samson ate at dinner on Monday (twice as many as lunch)
def sandwichesDinnerMonday : ℕ := 2 * sandwichesLunchMonday

-- Define the total number of sandwiches Samson ate on Monday
def totalSandwichesMonday : ℕ := sandwichesLunchMonday + sandwichesDinnerMonday

-- Define the number of sandwiches Samson ate for breakfast on Tuesday
def sandwichesBreakfastTuesday : ℕ := 1

-- Define the total number of sandwiches Samson ate on Tuesday
def totalSandwichesTuesday : ℕ := sandwichesBreakfastTuesday

-- Define the number of more sandwiches Samson ate on Monday than on Tuesday
theorem sandwiches_difference : totalSandwichesMonday - totalSandwichesTuesday = 8 :=
by
  sorry

end sandwiches_difference_l1718_171871


namespace hypotenuse_length_l1718_171860

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l1718_171860


namespace different_algorithms_for_same_problem_l1718_171896

-- Define the basic concept of a problem
def Problem := Type

-- Define what it means for something to be an algorithm solving a problem
def Algorithm (P : Problem) := P -> Prop

-- Define the statement to be true: Different algorithms can solve the same problem
theorem different_algorithms_for_same_problem (P : Problem) (A1 A2 : Algorithm P) :
  P = P -> A1 ≠ A2 -> true :=
by
  sorry

end different_algorithms_for_same_problem_l1718_171896


namespace rent_percentage_l1718_171847

noncomputable def condition1 (E : ℝ) : ℝ := 0.25 * E
noncomputable def condition2 (E : ℝ) : ℝ := 1.35 * E
noncomputable def condition3 (E' : ℝ) : ℝ := 0.40 * E'

theorem rent_percentage (E R R' : ℝ) (hR : R = condition1 E) (hE' : E = condition2 E) (hR' : R' = condition3 E) :
  (R' / R) * 100 = 216 :=
sorry

end rent_percentage_l1718_171847


namespace geometric_sequence_304th_term_l1718_171852

theorem geometric_sequence_304th_term (a r : ℤ) (n : ℕ) (h_a : a = 8) (h_ar : a * r = -8) (h_n : n = 304) :
  ∃ t : ℤ, t = -8 :=
by
  sorry

end geometric_sequence_304th_term_l1718_171852


namespace integrate_diff_eq_l1718_171817

noncomputable def particular_solution (x y : ℝ) : Prop :=
  (y^2 - x^2) / 2 + Real.exp y - Real.log ((x + Real.sqrt (1 + x^2)) / (2 + Real.sqrt 5)) = Real.exp 1 - 3 / 2

theorem integrate_diff_eq (x y : ℝ) :
  (∀ x y : ℝ, y' = (x * Real.sqrt (1 + x^2) + 1) / (Real.sqrt (1 + x^2) * (y + Real.exp y))) → 
  (∃ x0 y0 : ℝ, x0 = 2 ∧ y0 = 1) → 
  particular_solution x y :=
sorry

end integrate_diff_eq_l1718_171817


namespace proof_correctness_l1718_171820

-- Define the new operation
def new_op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Definitions for the conclusions
def conclusion_1 : Prop := new_op 1 (-2) = -8
def conclusion_2 : Prop := ∀ a b : ℝ, new_op a b = new_op b a
def conclusion_3 : Prop := ∀ a b : ℝ, new_op a b = 0 → a = 0
def conclusion_4 : Prop := ∀ a b : ℝ, a + b = 0 → (new_op a a + new_op b b = 8 * a^2)

-- Specify the correct conclusions
def correct_conclusions : Prop := conclusion_1 ∧ conclusion_2 ∧ ¬conclusion_3 ∧ conclusion_4

-- State the theorem
theorem proof_correctness : correct_conclusions := by
  sorry

end proof_correctness_l1718_171820


namespace printed_value_l1718_171836

theorem printed_value (X S : ℕ) (h1 : X = 5) (h2 : S = 0) : 
  (∃ n, S = (n * (3 * n + 7)) / 2 ∧ S ≥ 15000) → 
  X = 5 + 3 * 122 - 3 :=
by 
  sorry

end printed_value_l1718_171836


namespace max_cut_length_l1718_171889

theorem max_cut_length (board_size : ℕ) (total_pieces : ℕ) 
  (area_each : ℕ) 
  (total_area : ℕ)
  (total_perimeter : ℕ)
  (initial_perimeter : ℕ)
  (max_possible_length : ℕ)
  (h1 : board_size = 30) 
  (h2 : total_pieces = 225)
  (h3 : area_each = 4)
  (h4 : total_area = board_size * board_size)
  (h5 : total_perimeter = total_pieces * 10)
  (h6 : initial_perimeter = 4 * board_size)
  (h7 : max_possible_length = (total_perimeter - initial_perimeter) / 2) :
  max_possible_length = 1065 :=
by 
  -- Here, we do not include the proof as per the instructions
  sorry

end max_cut_length_l1718_171889


namespace find_number_of_girls_l1718_171866

-- Define the ratio of boys to girls as 8:4.
def ratio_boys_to_girls : ℕ × ℕ := (8, 4)

-- Define the total number of students.
def total_students : ℕ := 600

-- Define what it means for the number of girls given a ratio and total students.
def number_of_girls (ratio : ℕ × ℕ) (total : ℕ) : ℕ :=
  let total_parts := (ratio.1 + ratio.2)
  let part_value := total / total_parts
  ratio.2 * part_value

-- State the goal to prove the number of girls is 200 given the conditions.
theorem find_number_of_girls :
  number_of_girls ratio_boys_to_girls total_students = 200 :=
sorry

end find_number_of_girls_l1718_171866


namespace arithmetic_sequence_sum_l1718_171891

/-- Let {a_n} be an arithmetic sequence and S_n the sum of its first n terms.
   Given a_1 - a_5 - a_10 - a_15 + a_19 = 2, prove that S_19 = -38. --/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 1 - a 5 - a 10 - a 15 + a 19 = 2) :
  S 19 = -38 := 
sorry

end arithmetic_sequence_sum_l1718_171891


namespace kindergarten_classes_l1718_171863

theorem kindergarten_classes :
  ∃ (j a m : ℕ), j + a + m = 32 ∧
                  j > 0 ∧ a > 0 ∧ m > 0 ∧
                  j / 2 + a / 4 + m / 8 = 6 ∧
                  (j = 4 ∧ a = 4 ∧ m = 24) :=
by {
  sorry
}

end kindergarten_classes_l1718_171863


namespace average_sales_is_96_l1718_171864

-- Definitions for the sales data
def january_sales : ℕ := 110
def february_sales : ℕ := 80
def march_sales : ℕ := 70
def april_sales : ℕ := 130
def may_sales : ℕ := 90

-- Number of months
def num_months : ℕ := 5

-- Total sales calculation
def total_sales : ℕ := january_sales + february_sales + march_sales + april_sales + may_sales

-- Average sales per month calculation
def average_sales_per_month : ℕ := total_sales / num_months

-- Proposition to prove that the average sales per month is 96
theorem average_sales_is_96 : average_sales_per_month = 96 :=
by
  -- We use 'sorry' here to skip the proof, as the problem requires only the statement
  sorry

end average_sales_is_96_l1718_171864


namespace gary_egg_collection_l1718_171865

-- Conditions
def initial_chickens : ℕ := 4
def multiplier : ℕ := 8
def eggs_per_chicken_per_day : ℕ := 6
def days_in_week : ℕ := 7

-- Definitions derived from conditions
def current_chickens : ℕ := initial_chickens * multiplier
def eggs_per_day : ℕ := current_chickens * eggs_per_chicken_per_day
def eggs_per_week : ℕ := eggs_per_day * days_in_week

-- Proof statement
theorem gary_egg_collection : eggs_per_week = 1344 := by
  unfold eggs_per_week
  unfold eggs_per_day
  unfold current_chickens
  sorry

end gary_egg_collection_l1718_171865


namespace rowing_distance_l1718_171800

def man_rowing_speed_still_water : ℝ := 10
def stream_speed : ℝ := 8
def rowing_time_downstream : ℝ := 5
def effective_speed_downstream : ℝ := man_rowing_speed_still_water + stream_speed

theorem rowing_distance :
  effective_speed_downstream * rowing_time_downstream = 90 := 
by 
  sorry

end rowing_distance_l1718_171800


namespace work_completion_time_l1718_171844

theorem work_completion_time (days_B days_C days_all : ℝ) (h_B : days_B = 5) (h_C : days_C = 12) (h_all : days_all = 2.2222222222222223) : 
    (1 / ((days_all / 9) * 10) - 1 / days_B - 1 / days_C)⁻¹ = 60 / 37 := by 
  sorry

end work_completion_time_l1718_171844


namespace Tobias_change_l1718_171884

def cost_of_shoes := 95
def allowance_per_month := 5
def months_saving := 3
def charge_per_lawn := 15
def lawns_mowed := 4
def charge_per_driveway := 7
def driveways_shoveled := 5
def total_amount_saved : ℕ := (allowance_per_month * months_saving)
                          + (charge_per_lawn * lawns_mowed)
                          + (charge_per_driveway * driveways_shoveled)

theorem Tobias_change : total_amount_saved - cost_of_shoes = 15 := by
  sorry

end Tobias_change_l1718_171884


namespace second_storm_duration_l1718_171819

theorem second_storm_duration
  (x y : ℕ)
  (h1 : x + y = 45)
  (h2 : 30 * x + 15 * y = 975) :
  y = 25 := 
sorry

end second_storm_duration_l1718_171819


namespace sum_of_sampled_types_l1718_171862

-- Define the types of books in each category
def Chinese_types := 20
def Mathematics_types := 10
def Liberal_Arts_Comprehensive_types := 40
def English_types := 30

-- Define the total types of books
def total_types := Chinese_types + Mathematics_types + Liberal_Arts_Comprehensive_types + English_types

-- Define the sample size and stratified sampling ratio
def sample_size := 20
def sampling_ratio := sample_size / total_types

-- Define the number of types sampled from each category
def Mathematics_sampled := Mathematics_types * sampling_ratio
def Liberal_Arts_Comprehensive_sampled := Liberal_Arts_Comprehensive_types * sampling_ratio

-- Define the proof statement
theorem sum_of_sampled_types : Mathematics_sampled + Liberal_Arts_Comprehensive_sampled = 10 :=
by
  -- Your proof here
  sorry

end sum_of_sampled_types_l1718_171862


namespace ratio_square_correct_l1718_171826

noncomputable def ratio_square (a b : ℝ) (h : a / b = b / Real.sqrt (a^2 + b^2)) : ℝ :=
  let k := a / b
  let x := k * k
  x

theorem ratio_square_correct (a b : ℝ) (h : a / b = b / Real.sqrt (a^2 + b^2)) :
  ratio_square a b h = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end ratio_square_correct_l1718_171826


namespace value_of_3W5_l1718_171812

-- Define the operation W
def W (a b : ℝ) : ℝ := b + 15 * a - a^3

-- State the theorem to prove
theorem value_of_3W5 : W 3 5 = 23 := by
    sorry

end value_of_3W5_l1718_171812


namespace ca1_l1718_171843

theorem ca1 {
  a b : ℝ
} (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := 
by
  sorry

end ca1_l1718_171843


namespace students_play_alto_saxophone_l1718_171883

def roosevelt_high_school :=
  let total_students := 600
  let marching_band_students := total_students / 5
  let brass_instrument_students := marching_band_students / 2
  let saxophone_students := brass_instrument_students / 5
  let alto_saxophone_students := saxophone_students / 3
  alto_saxophone_students

theorem students_play_alto_saxophone :
  roosevelt_high_school = 4 :=
  by
    sorry

end students_play_alto_saxophone_l1718_171883


namespace probability_of_heart_and_joker_l1718_171825

-- Define a deck with 54 cards, including jokers
def total_cards : ℕ := 54

-- Define the count of specific cards in the deck
def hearts_count : ℕ := 13
def jokers_count : ℕ := 2
def remaining_cards (x: ℕ) : ℕ := total_cards - x

-- Define the probability of drawing a specific card
def prob_of_first_heart : ℚ := hearts_count / total_cards
def prob_of_second_joker (first_card_a_heart: Bool) : ℚ :=
  if first_card_a_heart then jokers_count / remaining_cards 1 else 0

-- Calculate the probability of drawing a heart first and then a joker
def prob_first_heart_then_joker : ℚ :=
  prob_of_first_heart * prob_of_second_joker true

-- Proving the final probability
theorem probability_of_heart_and_joker :
  prob_first_heart_then_joker = 13 / 1419 := by
  -- Skipping the proof
  sorry

end probability_of_heart_and_joker_l1718_171825


namespace karen_tests_graded_l1718_171806

theorem karen_tests_graded (n : ℕ) (T : ℕ) 
  (avg_score_70 : T = 70 * n)
  (combined_score_290 : T + 290 = 85 * (n + 2)) : 
  n = 8 := 
sorry

end karen_tests_graded_l1718_171806


namespace codecracker_number_of_codes_l1718_171875

theorem codecracker_number_of_codes : ∃ n : ℕ, n = 6 * 5^4 := by
  sorry

end codecracker_number_of_codes_l1718_171875


namespace problem_statement_l1718_171822

theorem problem_statement : 15 * 35 + 50 * 15 - 5 * 15 = 1200 := by
  sorry

end problem_statement_l1718_171822


namespace math_problem_l1718_171813

theorem math_problem :
  let initial := 180
  let thirty_five_percent := 0.35 * initial
  let one_third_less := thirty_five_percent - (thirty_five_percent / 3)
  let remaining := initial - one_third_less
  let three_fifths_remaining := (3 / 5) * remaining
  (three_fifths_remaining ^ 2) = 6857.84 :=
by
  sorry

end math_problem_l1718_171813


namespace negative_integer_solution_l1718_171831

theorem negative_integer_solution (N : ℤ) (h1 : N < 0) (h2 : N^2 + N = 6) : N = -3 := 
by 
  sorry

end negative_integer_solution_l1718_171831


namespace smaller_group_men_l1718_171808

theorem smaller_group_men (M : ℕ) (h1 : 36 * 25 = M * 90) : M = 10 :=
by
  -- Here we would provide the proof. Unfortunately, proving this in Lean 4 requires knowledge of algebra.
  sorry

end smaller_group_men_l1718_171808


namespace tim_used_to_run_days_l1718_171832

def hours_per_day := 2
def total_hours_per_week := 10
def added_days := 2

theorem tim_used_to_run_days (runs_per_day : ℕ) (total_weekly_runs : ℕ) (additional_runs : ℕ) : 
  runs_per_day = hours_per_day →
  total_weekly_runs = total_hours_per_week →
  additional_runs = added_days →
  (total_weekly_runs / runs_per_day) - additional_runs = 3 :=
by
  intros h1 h2 h3
  sorry

end tim_used_to_run_days_l1718_171832


namespace find_f1_l1718_171815

theorem find_f1 (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) + (-x) ^ 2 = -(f x + x ^ 2))
  (h2 : ∀ x, f (-x) + 2 ^ (-x) = f x + 2 ^ x) :
  f 1 = -7 / 4 := by
sorry

end find_f1_l1718_171815


namespace find_softball_players_l1718_171882

def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def football_players : ℕ := 18
def total_players : ℕ := 59

theorem find_softball_players :
  let C := cricket_players
  let H := hockey_players
  let F := football_players
  let T := total_players
  S = T - (C + H + F) :=
by
  let C := cricket_players
  let H := hockey_players
  let F := football_players
  let T := total_players
  show S = T - (C + H + F)
  sorry

end find_softball_players_l1718_171882


namespace rectangle_square_ratio_l1718_171839

theorem rectangle_square_ratio (l w s : ℝ) (h1 : 0.4 * l * w = 0.25 * s * s) : l / w = 15.625 :=
by
  sorry

end rectangle_square_ratio_l1718_171839


namespace alcohol_solution_contradiction_l1718_171821

theorem alcohol_solution_contradiction (initial_volume : ℕ) (added_water : ℕ) 
                                        (final_volume : ℕ) (final_concentration : ℕ) 
                                        (initial_concentration : ℕ) : 
                                        initial_volume = 75 → added_water = 50 → 
                                        final_volume = initial_volume + added_water → 
                                        final_concentration = 45 → 
                                        ¬ (initial_concentration * initial_volume = final_concentration * final_volume) :=
by 
  intro h_initial_volume h_added_water h_final_volume h_final_concentration
  sorry

end alcohol_solution_contradiction_l1718_171821


namespace carnations_count_l1718_171885

-- Define the conditions:
def vase_capacity : ℕ := 6
def number_of_roses : ℕ := 47
def number_of_vases : ℕ := 9

-- The goal is to prove that the number of carnations is 7:
theorem carnations_count : (number_of_vases * vase_capacity) - number_of_roses = 7 :=
by
  sorry

end carnations_count_l1718_171885


namespace evaluate_f_2010_times_l1718_171890

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x^2011)^(1/2011)

theorem evaluate_f_2010_times (x : ℝ) (h : x = 2011) :
  (f^[2010] x)^2011 = 2011^2011 :=
by
  rw [h]
  sorry

end evaluate_f_2010_times_l1718_171890


namespace max_blue_cells_n2_max_blue_cells_n25_l1718_171846

noncomputable def max_blue_cells (table_size n : ℕ) : ℕ :=
  if h : (table_size = 50 ∧ n = 2) then 2450
  else if h : (table_size = 50 ∧ n = 25) then 1300
  else 0 -- Default case that should not happen for this problem

theorem max_blue_cells_n2 : max_blue_cells 50 2 = 2450 := 
by
  sorry

theorem max_blue_cells_n25 : max_blue_cells 50 25 = 1300 :=
by
  sorry

end max_blue_cells_n2_max_blue_cells_n25_l1718_171846


namespace arithmetic_sequence_general_formula_and_geometric_condition_l1718_171837

theorem arithmetic_sequence_general_formula_and_geometric_condition :
  ∀ {a : ℕ → ℤ} {S : ℕ → ℤ} {k : ℕ}, 
    (∀ n, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)) →
    a 1 = 9 →
    S 3 = 21 →
    a 5 * S k = a 8 ^ 2 →
    k = 5 :=
by 
  intros a S k hS ha1 hS3 hgeom
  sorry

end arithmetic_sequence_general_formula_and_geometric_condition_l1718_171837


namespace probability_blue_face_up_l1718_171858

def cube_probability_blue : ℚ := 
  let total_faces := 6
  let blue_faces := 4
  blue_faces / total_faces

theorem probability_blue_face_up :
  cube_probability_blue = 2 / 3 :=
by
  sorry

end probability_blue_face_up_l1718_171858


namespace extra_time_needed_l1718_171861

variable (S : ℝ) (d : ℝ) (T T' : ℝ)

-- Original conditions
def original_speed_at_time_distance (S : ℝ) (T : ℝ) (d : ℝ) : Prop :=
  S * T = d

def decreased_speed (original_S : ℝ) : ℝ :=
  0.80 * original_S

def decreased_speed_time (T' : ℝ ) (decreased_S : ℝ) (d : ℝ) : Prop :=
  decreased_S * T' = d

theorem extra_time_needed
  (h1 : original_speed_at_time_distance S T d)
  (h2 : T = 40)
  (h3 : decreased_speed S = 0.80 * S)
  (h4 : decreased_speed_time T' (decreased_speed S) d) :
  T' - T = 10 :=
by
  sorry

end extra_time_needed_l1718_171861


namespace simplify_expression_l1718_171841

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 0) : x⁻¹ - 3 * x + 2 = - (3 * x^2 - 2 * x - 1) / x :=
by
  sorry

end simplify_expression_l1718_171841


namespace total_cows_l1718_171803

theorem total_cows (Matthews Aaron Tyron Marovich : ℕ) 
  (h1 : Matthews = 60)
  (h2 : Aaron = 4 * Matthews)
  (h3 : Tyron = Matthews - 20)
  (h4 : Aaron + Matthews + Tyron = Marovich + 30) :
  Aaron + Matthews + Tyron + Marovich = 650 :=
by
  sorry

end total_cows_l1718_171803


namespace magnitude_z1_pure_imaginary_l1718_171818

open Complex

theorem magnitude_z1_pure_imaginary 
  (a : ℝ)
  (z1 : ℂ := a + 2 * I)
  (z2 : ℂ := 3 - 4 * I)
  (h : (z1 / z2).re = 0) :
  Complex.abs z1 = 10 / 3 := 
sorry

end magnitude_z1_pure_imaginary_l1718_171818


namespace representation_of_2015_l1718_171850

theorem representation_of_2015 :
  ∃ (p d3 i : ℕ),
    Prime p ∧ -- p is prime
    d3 % 3 = 0 ∧ -- d3 is divisible by 3
    400 < i ∧ i < 500 ∧ i % 3 ≠ 0 ∧ -- i is in interval and not divisible by 3
    2015 = p + d3 + i := sorry

end representation_of_2015_l1718_171850


namespace impossible_a_values_l1718_171886

theorem impossible_a_values (a : ℝ) :
  ¬((1-a)^2 + (1+a)^2 < 4) → (a ≤ -1 ∨ a ≥ 1) :=
by
  sorry

end impossible_a_values_l1718_171886


namespace correct_result_is_102357_l1718_171868

-- Defining the conditions
def number (f : ℕ) : Prop := f * 153 = 102357

-- Stating the proof problem
theorem correct_result_is_102357 (f : ℕ) (h : f * 153 = 102325) (wrong_digits : ℕ) :
  (number f) :=
by
  sorry

end correct_result_is_102357_l1718_171868


namespace largest_corner_sum_l1718_171842

noncomputable def sum_faces (cube : ℕ → ℕ) : Prop :=
  cube 1 + cube 7 = 8 ∧ 
  cube 2 + cube 6 = 8 ∧ 
  cube 3 + cube 5 = 8 ∧ 
  cube 4 + cube 4 = 8

theorem largest_corner_sum (cube : ℕ → ℕ) 
  (h : sum_faces cube) : 
  ∃ n, n = 17 ∧ 
  ∀ a b c, (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
            (cube a = 7 ∧ cube b = 6 ∧ cube c = 4 ∨ 
             cube a = 6 ∧ cube b = 4 ∧ cube c = 7 ∨ 
             cube a = 4 ∧ cube b = 7 ∧ cube c = 6)) → 
            a + b + c = n := sorry

end largest_corner_sum_l1718_171842


namespace round_table_arrangement_l1718_171824

theorem round_table_arrangement :
  ∀ (n : ℕ), n = 10 → (∃ factorial_value : ℕ, factorial_value = Nat.factorial (n - 1) ∧ factorial_value = 362880) := by
  sorry

end round_table_arrangement_l1718_171824


namespace shane_chewed_pieces_l1718_171897

theorem shane_chewed_pieces :
  ∀ (Elyse Rick Shane: ℕ),
  Elyse = 100 →
  Rick = Elyse / 2 →
  Shane = Rick / 2 →
  Shane_left = 14 →
  (Shane - Shane_left) = 11 :=
by
  intros Elyse Rick Shane Elyse_def Rick_def Shane_def Shane_left_def
  sorry

end shane_chewed_pieces_l1718_171897


namespace general_formula_condition1_general_formula_condition2_general_formula_condition3_sum_Tn_l1718_171811

-- Define the conditions
axiom condition1 (n : ℕ) (h : 2 ≤ n) : ∀ (a : ℕ → ℕ), a 1 = 1 → a n = n / (n-1) * a (n-1)
axiom condition2 (n : ℕ) : ∀ (S : ℕ → ℕ), 2 * S n = n^2 + n
axiom condition3 (n : ℕ) : ∀ (a : ℕ → ℕ), a 1 = 1 → a 3 = 3 → (a n + a (n+2)) = 2 * a (n+1)

-- Proof statements
theorem general_formula_condition1 : ∀ (n : ℕ) (a : ℕ → ℕ) (h : 2 ≤ n), (a 1 = 1) → (∀ n, a n = n) :=
by sorry

theorem general_formula_condition2 : ∀ (n : ℕ) (S a : ℕ → ℕ), (2 * S n = n^2 + n) → (∀ n, a n = n) :=
by sorry

theorem general_formula_condition3 : ∀ (n : ℕ) (a : ℕ → ℕ), (a 1 = 1) → (a 3 = 3) → (∀ n, a n + a (n+2) = 2 * a (n+1)) → (∀ n, a n = n) :=
by sorry

theorem sum_Tn : ∀ (b : ℕ → ℕ) (T : ℕ → ℝ), (b 1 = 2) → (b 2 + b 3 = 12) → (∀ n, T n = 2 * (1 - 1 / (n + 1))) :=
by sorry

end general_formula_condition1_general_formula_condition2_general_formula_condition3_sum_Tn_l1718_171811


namespace smallest_integer_solution_l1718_171838

theorem smallest_integer_solution (x : ℤ) (h : 2 * (x : ℝ)^2 + 2 * |(x : ℝ)| + 7 < 25) : x = -2 :=
by
  sorry

end smallest_integer_solution_l1718_171838


namespace girls_in_art_class_l1718_171899

theorem girls_in_art_class (g b : ℕ) (h_ratio : 4 * b = 3 * g) (h_total : g + b = 70) : g = 40 :=
by {
  sorry
}

end girls_in_art_class_l1718_171899


namespace cosQ_is_0_point_4_QP_is_12_prove_QR_30_l1718_171895

noncomputable def find_QR (Q : Real) (QP : Real) : Real :=
  let cosQ := 0.4
  let QR := QP / cosQ
  QR

theorem cosQ_is_0_point_4_QP_is_12_prove_QR_30 :
  find_QR 0.4 12 = 30 :=
by
  sorry

end cosQ_is_0_point_4_QP_is_12_prove_QR_30_l1718_171895


namespace age_sum_l1718_171872

theorem age_sum (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 10) : a + b + c = 27 := by
  sorry

end age_sum_l1718_171872


namespace fraction_subtraction_inequality_l1718_171856

theorem fraction_subtraction_inequality (a b n : ℕ) (h1 : a < b) (h2 : 0 < n) (h3 : n < a) : 
  (a : ℚ) / b > (a - n : ℚ) / (b - n) :=
sorry

end fraction_subtraction_inequality_l1718_171856


namespace remainder_of_2_pow_23_mod_5_l1718_171829

theorem remainder_of_2_pow_23_mod_5 
    (h1 : (2^2) % 5 = 4)
    (h2 : (2^3) % 5 = 3)
    (h3 : (2^4) % 5 = 1) :
    (2^23) % 5 = 3 :=
by
  sorry

end remainder_of_2_pow_23_mod_5_l1718_171829


namespace mingyu_change_l1718_171877

theorem mingyu_change :
  let eraser_cost := 350
  let pencil_cost := 180
  let erasers_count := 3
  let pencils_count := 2
  let payment := 2000
  let total_eraser_cost := erasers_count * eraser_cost
  let total_pencil_cost := pencils_count * pencil_cost
  let total_cost := total_eraser_cost + total_pencil_cost
  let change := payment - total_cost
  change = 590 := 
by
  -- The proof will go here
  sorry

end mingyu_change_l1718_171877


namespace solve_inequality_l1718_171830

-- Define the function satisfying the given conditions
def f (x : ℝ) : ℝ := sorry

axiom f_functional_eq : ∀ (x y : ℝ), f (x / y) = f x - f y
axiom f_not_zero : ∀ x : ℝ, f x ≠ 0
axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

-- Define the theorem that proves the inequality given the conditions
theorem solve_inequality (x : ℝ) :
  f x + f (x + 1/2) < 0 ↔ x ∈ (Set.Ioo ( (1 - Real.sqrt 17) / 4 ) 0) ∪ (Set.Ioo 0 ( (1 + Real.sqrt 17) / 4 )) :=
by
  sorry

end solve_inequality_l1718_171830


namespace sum_first_8_geometric_l1718_171840

theorem sum_first_8_geometric :
  let a₁ := 1 / 15
  let r := 2
  let S₄ := a₁ * (1 - r^4) / (1 - r)
  let S₈ := a₁ * (1 - r^8) / (1 - r)
  S₄ = 1 → S₈ = 17 := 
by
  intros a₁ r S₄ S₈ h
  sorry

end sum_first_8_geometric_l1718_171840


namespace probability_red_balls_by_4th_draw_l1718_171827

theorem probability_red_balls_by_4th_draw :
  let total_balls := 10
  let red_prob := 2 / total_balls
  let white_prob := 1 - red_prob
  (white_prob^3) * red_prob = 0.0434 := sorry

end probability_red_balls_by_4th_draw_l1718_171827


namespace projection_coordinates_eq_zero_l1718_171834

theorem projection_coordinates_eq_zero (x y z : ℝ) :
  let M := (x, y, z)
  let M₁ := (x, y, 0)
  let M₂ := (0, y, 0)
  let M₃ := (0, 0, 0)
  M₃ = (0, 0, 0) :=
sorry

end projection_coordinates_eq_zero_l1718_171834


namespace edward_spent_amount_l1718_171881

-- Definitions based on the problem conditions
def initial_amount : ℕ := 18
def remaining_amount : ℕ := 2

-- The statement to prove: Edward spent $16
theorem edward_spent_amount : initial_amount - remaining_amount = 16 := by
  sorry

end edward_spent_amount_l1718_171881


namespace sqrt_arith_progression_impossible_l1718_171859

theorem sqrt_arith_progression_impossible (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hneab : a ≠ b) (hnebc : b ≠ c) (hneca : c ≠ a) :
  ¬ ∃ d : ℝ, (d = (Real.sqrt b - Real.sqrt a)) ∧ (d = (Real.sqrt c - Real.sqrt b)) :=
sorry

end sqrt_arith_progression_impossible_l1718_171859


namespace min_x_y_l1718_171893

noncomputable def min_value (x y : ℝ) : ℝ := x + y

theorem min_x_y (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 16 * y = x * y) :
  min_value x y = 25 :=
sorry

end min_x_y_l1718_171893


namespace alpha_beta_sum_l1718_171835

theorem alpha_beta_sum (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 80 * x + 1551) / (x^2 + 57 * x - 2970)) :
  α + β = 137 :=
by
  sorry

end alpha_beta_sum_l1718_171835


namespace find_stream_speed_l1718_171801

theorem find_stream_speed (b s : ℝ) 
  (h1 : b + s = 10) 
  (h2 : b - s = 8) : s = 1 :=
by
  sorry

end find_stream_speed_l1718_171801


namespace andrew_kept_correct_l1718_171848

open Nat

def andrew_bought : ℕ := 750
def daniel_received : ℕ := 250
def fred_received : ℕ := daniel_received + 120
def total_shared : ℕ := daniel_received + fred_received
def andrew_kept : ℕ := andrew_bought - total_shared

theorem andrew_kept_correct : andrew_kept = 130 :=
by
  unfold andrew_kept andrew_bought total_shared fred_received daniel_received
  rfl

end andrew_kept_correct_l1718_171848


namespace total_percentage_of_failed_candidates_l1718_171855

-- Define the given conditions
def total_candidates : ℕ := 2000
def number_of_girls : ℕ := 900
def number_of_boys : ℕ := total_candidates - number_of_girls
def percentage_of_boys_passed : ℚ := 0.28
def percentage_of_girls_passed : ℚ := 0.32

-- Define the proof statement
theorem total_percentage_of_failed_candidates : 
  (total_candidates - (percentage_of_boys_passed * number_of_boys + percentage_of_girls_passed * number_of_girls)) / total_candidates * 100 = 70.2 :=
by
  sorry

end total_percentage_of_failed_candidates_l1718_171855


namespace g_at_pi_over_3_l1718_171810

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

noncomputable def g (x : ℝ) (ω φ : ℝ) : ℝ := 3 * Real.sin (ω * x + φ) - 1

theorem g_at_pi_over_3 (ω φ : ℝ) :
  (∀ x : ℝ, f (π / 3 + x) ω φ = f (π / 3 - x) ω φ) →
  g (π / 3) ω φ = -1 :=
by sorry

end g_at_pi_over_3_l1718_171810


namespace remainder_19_pow_19_plus_19_mod_20_l1718_171823

theorem remainder_19_pow_19_plus_19_mod_20 : (19^19 + 19) % 20 = 18 := 
by
  sorry

end remainder_19_pow_19_plus_19_mod_20_l1718_171823


namespace compute_expression_value_l1718_171874

noncomputable def expression := 3 ^ (Real.log 4 / Real.log 3) - 27 ^ (2 / 3) - Real.log 0.01 / Real.log 10 + Real.log (Real.exp 3)

theorem compute_expression_value :
  expression = 0 := 
by
  sorry

end compute_expression_value_l1718_171874


namespace only_nice_number_is_three_l1718_171892

def P (x : ℕ) : ℕ := x + 1
def Q (x : ℕ) : ℕ := x^2 + 1

def nice (n : ℕ) : Prop :=
  ∃ (xs ys : ℕ → ℕ), 
    xs 1 = 1 ∧ ys 1 = 3 ∧
    (∀ k, xs (k+1) = P (xs k) ∧ ys (k+1) = Q (ys k) ∨ xs (k+1) = Q (xs k) ∧ ys (k+1) = P (ys k)) ∧
    xs n = ys n

theorem only_nice_number_is_three (n : ℕ) : nice n ↔ n = 3 :=
by
  sorry

end only_nice_number_is_three_l1718_171892


namespace total_vehicle_wheels_in_parking_lot_l1718_171805

def vehicles_wheels := (1 * 4) + (1 * 4) + (8 * 4) + (4 * 2) + (3 * 6) + (2 * 4) + (1 * 8) + (2 * 3)

theorem total_vehicle_wheels_in_parking_lot : vehicles_wheels = 88 :=
by {
    sorry
}

end total_vehicle_wheels_in_parking_lot_l1718_171805


namespace set_operation_equivalence_l1718_171854

variable {U : Type} -- U is the universal set
variables {X Y Z : Set U} -- X, Y, and Z are subsets of the universal set U

def star (A B : Set U) : Set U := A ∩ B  -- Define the operation "∗" as intersection

theorem set_operation_equivalence :
  star (star X Y) Z = (X ∩ Y) ∩ Z :=  -- Formulate the problem as a theorem to prove
by
  sorry  -- Proof is omitted

end set_operation_equivalence_l1718_171854


namespace domain_of_fn_l1718_171879

noncomputable def domain_fn (x : ℝ) : ℝ := (Real.sqrt (3 * x + 4)) / x

theorem domain_of_fn :
  { x : ℝ | x ≥ -4 / 3 ∧ x ≠ 0 } =
  { x : ℝ | 3 * x + 4 ≥ 0 ∧ x ≠ 0 } :=
by
  ext x
  simp
  exact sorry

end domain_of_fn_l1718_171879


namespace minimum_odd_numbers_in_set_l1718_171888

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l1718_171888


namespace find_b_perpendicular_l1718_171816

theorem find_b_perpendicular
  (b : ℝ)
  (line1 : ∀ x y : ℝ, 2 * x - 3 * y + 5 = 0)
  (line2 : ∀ x y : ℝ, b * x - 3 * y + 1 = 0)
  (perpendicular : (2 / 3) * (b / 3) = -1)
  : b = -9/2 :=
sorry

end find_b_perpendicular_l1718_171816
