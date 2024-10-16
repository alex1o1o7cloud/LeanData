import Mathlib

namespace NUMINAMATH_CALUDE_dogs_not_eating_l1838_183825

theorem dogs_not_eating (total : ℕ) (like_apples : ℕ) (like_chicken : ℕ) (like_both : ℕ) :
  total = 75 →
  like_apples = 18 →
  like_chicken = 55 →
  like_both = 10 →
  total - (like_apples + like_chicken - like_both) = 12 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_eating_l1838_183825


namespace NUMINAMATH_CALUDE_min_force_to_prevent_slipping_l1838_183860

/-- The minimum force needed to keep a book from slipping -/
theorem min_force_to_prevent_slipping 
  (M : ℝ) -- Mass of the book
  (g : ℝ) -- Acceleration due to gravity
  (μs : ℝ) -- Coefficient of static friction
  (h1 : M > 0) -- Mass is positive
  (h2 : g > 0) -- Gravity is positive
  (h3 : μs > 0) -- Coefficient of static friction is positive
  : 
  ∃ (F : ℝ), F = M * g / μs ∧ F ≥ M * g ∧ ∀ (F' : ℝ), F' < F → F' * μs < M * g :=
sorry

end NUMINAMATH_CALUDE_min_force_to_prevent_slipping_l1838_183860


namespace NUMINAMATH_CALUDE_employment_percentage_l1838_183804

theorem employment_percentage
  (population : ℝ)
  (employed_males_percentage : ℝ)
  (employed_females_percentage : ℝ)
  (h1 : employed_males_percentage = 48 / 100)
  (h2 : employed_females_percentage = 25 / 100)
  : (employed_males_percentage * population) / ((1 - employed_females_percentage) * population) = 64 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_employment_percentage_l1838_183804


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l1838_183896

theorem min_sum_of_squares (x₁ x₂ x₃ : ℝ) 
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0)
  (sum_constraint : x₁ + 3*x₂ + 5*x₃ = 100) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 2000/7 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l1838_183896


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_two_thirds_l1838_183814

/-- The infinite repeating decimal 0.666... -/
def repeating_decimal : ℚ := 0.6666666666666667

/-- The theorem stating that the repeating decimal 0.666... is equal to 2/3 -/
theorem repeating_decimal_equals_two_thirds : repeating_decimal = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_two_thirds_l1838_183814


namespace NUMINAMATH_CALUDE_study_time_for_average_score_l1838_183858

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  studyTime : ℝ
  score : ℝ
  ratio : ℝ
  direct_relation : ratio = score / studyTime

/-- The problem setup and solution -/
theorem study_time_for_average_score
  (first_exam : StudyScoreRelation)
  (h_first_exam : first_exam.studyTime = 3 ∧ first_exam.score = 60)
  (target_average : ℝ)
  (h_target_average : target_average = 75)
  : ∃ (second_exam : StudyScoreRelation),
    second_exam.ratio = first_exam.ratio ∧
    (first_exam.score + second_exam.score) / 2 = target_average ∧
    second_exam.studyTime = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_study_time_for_average_score_l1838_183858


namespace NUMINAMATH_CALUDE_fourth_cd_cost_l1838_183882

theorem fourth_cd_cost (initial_avg_cost : ℝ) (new_avg_cost : ℝ) (initial_cd_count : ℕ) :
  initial_avg_cost = 15 →
  new_avg_cost = 16 →
  initial_cd_count = 3 →
  (initial_cd_count * initial_avg_cost + (new_avg_cost * (initial_cd_count + 1) - initial_cd_count * initial_avg_cost)) = 19 := by
  sorry

end NUMINAMATH_CALUDE_fourth_cd_cost_l1838_183882


namespace NUMINAMATH_CALUDE_pumpkin_count_l1838_183810

/-- The number of pumpkins grown by Sandy -/
def sandy_pumpkins : ℕ := 51

/-- The number of pumpkins grown by Mike -/
def mike_pumpkins : ℕ := 23

/-- The total number of pumpkins grown by Sandy and Mike -/
def total_pumpkins : ℕ := sandy_pumpkins + mike_pumpkins

theorem pumpkin_count : total_pumpkins = 74 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_count_l1838_183810


namespace NUMINAMATH_CALUDE_semicircle_distance_l1838_183815

theorem semicircle_distance (r : ℝ) (h : r > 0) : 
  (π * r) = (1 / 2) * (2 * π * r) := by sorry

#check semicircle_distance

end NUMINAMATH_CALUDE_semicircle_distance_l1838_183815


namespace NUMINAMATH_CALUDE_walking_speed_l1838_183801

theorem walking_speed (x : ℝ) : 
  let tom_speed := x^2 - 14*x - 48
  let jerry_distance := x^2 - 5*x - 84
  let jerry_time := x + 8
  let jerry_speed := jerry_distance / jerry_time
  x ≠ -8 → tom_speed = jerry_speed → tom_speed = 6 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_l1838_183801


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l1838_183888

/-- For a normal distribution with mean 16.5 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 13.5. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (h_μ : μ = 16.5) (h_σ : σ = 1.5) :
  μ - 2 * σ = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l1838_183888


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l1838_183846

/-- For a normal distribution with given properties, prove the standard deviation --/
theorem normal_distribution_std_dev (μ σ : ℝ) (h1 : μ = 17.5) (h2 : μ - 2 * σ = 12.5) : σ = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l1838_183846


namespace NUMINAMATH_CALUDE_rachels_brownies_l1838_183851

/-- Rachel's brownie problem -/
theorem rachels_brownies (total : ℕ) (left_at_home : ℕ) (brought_to_school : ℕ) : 
  total = 40 → left_at_home = 24 → brought_to_school = total - left_at_home → brought_to_school = 16 := by
  sorry

#check rachels_brownies

end NUMINAMATH_CALUDE_rachels_brownies_l1838_183851


namespace NUMINAMATH_CALUDE_no_double_application_function_l1838_183871

theorem no_double_application_function : ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l1838_183871


namespace NUMINAMATH_CALUDE_new_girl_weight_l1838_183873

/-- Given a group of 10 girls, if replacing one girl weighing 50 kg with a new girl
    increases the average weight by 5 kg, then the new girl weighs 100 kg. -/
theorem new_girl_weight (initial_weight : ℝ) (new_weight : ℝ) :
  (initial_weight - 50 + new_weight) / 10 = initial_weight / 10 + 5 →
  new_weight = 100 := by
sorry

end NUMINAMATH_CALUDE_new_girl_weight_l1838_183873


namespace NUMINAMATH_CALUDE_certain_number_equals_sixteen_l1838_183819

theorem certain_number_equals_sixteen : ∃ x : ℝ, x^5 = 4^10 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equals_sixteen_l1838_183819


namespace NUMINAMATH_CALUDE_andrews_eggs_l1838_183809

-- Define the costs
def toast_cost : ℕ := 1
def egg_cost : ℕ := 3

-- Define Dale's breakfast
def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

-- Define Andrew's breakfast
def andrew_toast : ℕ := 1

-- Define the total cost
def total_cost : ℕ := 15

-- Theorem to prove
theorem andrews_eggs :
  ∃ (andrew_eggs : ℕ),
    toast_cost * (dale_toast + andrew_toast) +
    egg_cost * (dale_eggs + andrew_eggs) = total_cost ∧
    andrew_eggs = 2 := by
  sorry

end NUMINAMATH_CALUDE_andrews_eggs_l1838_183809


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1838_183808

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a - 1)*x + (a - 1) > 0) ↔ (1 < a ∧ a < 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1838_183808


namespace NUMINAMATH_CALUDE_vector_perpendicular_value_l1838_183875

-- Define the vectors
def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the perpendicularity condition
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Theorem statement
theorem vector_perpendicular_value (x : ℝ) :
  perpendicular a (a.1 - (b x).1, a.2 - (b x).2) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_value_l1838_183875


namespace NUMINAMATH_CALUDE_pillar_base_side_length_l1838_183834

theorem pillar_base_side_length (string_length : ℝ) (side_length : ℝ) : 
  string_length = 78 → 
  string_length = 3 * side_length → 
  side_length = 26 := by
  sorry

#check pillar_base_side_length

end NUMINAMATH_CALUDE_pillar_base_side_length_l1838_183834


namespace NUMINAMATH_CALUDE_no_integer_solution_l1838_183865

theorem no_integer_solution : ¬∃ (x : ℝ), 
  (∃ (a b c : ℤ), (x - 1/x = a) ∧ (1/x - 1/(x^2 + 1) = b) ∧ (1/(x^2 + 1) - 2*x = c)) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1838_183865


namespace NUMINAMATH_CALUDE_brownies_in_box_l1838_183802

-- Define the total number of brownies
def total_brownies : ℕ := 349

-- Define the number of full boxes
def full_boxes : ℕ := 49

-- Define the number of brownies per box
def brownies_per_box : ℕ := total_brownies / full_boxes

-- Theorem statement
theorem brownies_in_box : brownies_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_brownies_in_box_l1838_183802


namespace NUMINAMATH_CALUDE_smallest_B_for_2020_sum_l1838_183803

/-- The sum of consecutive integers from a to b, inclusive -/
def sum_consecutive (a b : ℤ) : ℤ := (b - a + 1) * (a + b) / 2

/-- Proposition: -2019 is the smallest integer B such that there exists a sequence
    of consecutive integers, including B, that sums up to 2020 -/
theorem smallest_B_for_2020_sum : 
  (∀ B < -2019, ¬∃ n : ℕ, sum_consecutive B (B + n) = 2020) ∧
  (∃ n : ℕ, sum_consecutive (-2019) (-2019 + n) = 2020) :=
sorry

end NUMINAMATH_CALUDE_smallest_B_for_2020_sum_l1838_183803


namespace NUMINAMATH_CALUDE_car_meeting_problem_l1838_183862

theorem car_meeting_problem (S : ℝ) 
  (h1 : S > 0) 
  (h2 : 60 < S) 
  (h3 : 50 < S) 
  (h4 : (60 / (S - 60)) = ((S - 60 + 50) / (60 + S - 50))) : S = 130 := by
  sorry

end NUMINAMATH_CALUDE_car_meeting_problem_l1838_183862


namespace NUMINAMATH_CALUDE_sector_area_l1838_183877

theorem sector_area (diameter : ℝ) (central_angle : ℝ) :
  diameter = 6 →
  central_angle = 120 →
  (π * (diameter / 2)^2 * central_angle / 360) = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l1838_183877


namespace NUMINAMATH_CALUDE_negation_existence_quadratic_l1838_183889

theorem negation_existence_quadratic (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_quadratic_l1838_183889


namespace NUMINAMATH_CALUDE_tax_free_amount_correct_l1838_183876

/-- The tax-free amount for goods purchased in country B -/
def tax_free_amount : ℝ := 600

/-- The total value of goods purchased -/
def total_value : ℝ := 1720

/-- The tax rate applied to the portion exceeding the tax-free amount -/
def tax_rate : ℝ := 0.08

/-- The amount of tax paid -/
def tax_paid : ℝ := 89.6

/-- Theorem stating that the tax-free amount satisfies the given conditions -/
theorem tax_free_amount_correct : 
  tax_rate * (total_value - tax_free_amount) = tax_paid := by sorry

end NUMINAMATH_CALUDE_tax_free_amount_correct_l1838_183876


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1838_183822

def p (x : ℝ) : ℝ := 4*x^4 - 5*x^3 - 30*x^2 + 40*x + 24

theorem roots_of_polynomial :
  {x : ℝ | p x = 0} = {3, -1, -2, 1} :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1838_183822


namespace NUMINAMATH_CALUDE_min_value_fraction_l1838_183870

theorem min_value_fraction (a : ℝ) (h1 : 0 < a) (h2 : a < 3) :
  1 / a + 9 / (3 - a) ≥ 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1838_183870


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1838_183864

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 171 → n = 19 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l1838_183864


namespace NUMINAMATH_CALUDE_intersection_P_Q_l1838_183874

def P : Set ℝ := {x | x^2 - 16 < 0}
def Q : Set ℝ := {x | ∃ n : ℤ, x = 2 * n}

theorem intersection_P_Q : P ∩ Q = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l1838_183874


namespace NUMINAMATH_CALUDE_expression_equality_l1838_183898

theorem expression_equality (a b : ℝ) :
  (-a * b^2)^3 + a * b^2 * (a * b)^2 * (-2 * b)^2 = 3 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1838_183898


namespace NUMINAMATH_CALUDE_point_b_position_l1838_183853

theorem point_b_position (a b : ℤ) : 
  a = -5 → (b = a + 4 ∨ b = a - 4) → (b = -1 ∨ b = -9) := by
  sorry

end NUMINAMATH_CALUDE_point_b_position_l1838_183853


namespace NUMINAMATH_CALUDE_arun_weight_average_l1838_183842

def arun_weight_lower_bound : ℝ := 66
def arun_weight_upper_bound : ℝ := 72
def brother_lower_bound : ℝ := 60
def brother_upper_bound : ℝ := 70
def mother_upper_bound : ℝ := 69

theorem arun_weight_average :
  let lower := max arun_weight_lower_bound brother_lower_bound
  let upper := min (min arun_weight_upper_bound brother_upper_bound) mother_upper_bound
  (lower + upper) / 2 = 67.5 := by sorry

end NUMINAMATH_CALUDE_arun_weight_average_l1838_183842


namespace NUMINAMATH_CALUDE_mixture_volume_calculation_l1838_183848

/-- Proves that given a mixture with 20% water content, if adding 8.333333333333334 gallons
    of water increases the water percentage to 25%, then the initial volume of the mixture
    is 125 gallons. -/
theorem mixture_volume_calculation (initial_volume : ℝ) : 
  (0.20 * initial_volume + 8.333333333333334) / (initial_volume + 8.333333333333334) = 0.25 →
  initial_volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_calculation_l1838_183848


namespace NUMINAMATH_CALUDE_inequality_holds_on_interval_largest_interval_l1838_183879

theorem inequality_holds_on_interval (x : ℝ) (h : x ∈ Set.Icc 0 4) :
  ∀ y : ℝ, y > 0 → (5 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y :=
sorry

theorem largest_interval :
  ∀ a : ℝ, a > 4 → ∃ y : ℝ, y > 0 ∧ (5 * (a * y^2 + a^2 * y + 4 * y^2 + 4 * a * y)) / (a + y) ≤ 3 * a^2 * y :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_on_interval_largest_interval_l1838_183879


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1838_183844

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1838_183844


namespace NUMINAMATH_CALUDE_second_smallest_coprime_to_210_l1838_183890

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem second_smallest_coprime_to_210 :
  ∃ (x : ℕ), x > 1 ∧ 
  is_relatively_prime x 210 ∧
  (∃ (y : ℕ), y > 1 ∧ y < x ∧ is_relatively_prime y 210) ∧
  (∀ (z : ℕ), z > 1 ∧ z < x ∧ is_relatively_prime z 210 → z = 11) ∧
  x = 13 := by
sorry

end NUMINAMATH_CALUDE_second_smallest_coprime_to_210_l1838_183890


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l1838_183884

theorem ratio_a_to_b (a b c d : ℚ) 
  (h1 : b / c = 7 / 9)
  (h2 : c / d = 5 / 7)
  (h3 : a / d = 5 / 12) :
  a / b = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l1838_183884


namespace NUMINAMATH_CALUDE_bisection_method_theorem_l1838_183828

/-- The bisection method theorem -/
theorem bisection_method_theorem 
  (f : ℝ → ℝ) (a b : ℝ) 
  (h_continuous : Continuous f) 
  (h_unique_zero : ∃! x, x ∈ Set.Ioo a b ∧ f x = 0) 
  (h_interval : b - a = 0.1) :
  ∃ n : ℕ, n ≤ 10 ∧ (0.1 / 2^n : ℝ) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_bisection_method_theorem_l1838_183828


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1838_183827

theorem quadratic_inequality_solution_range (d : ℝ) : 
  (d > 0 ∧ ∃ x : ℝ, x^2 - 8*x + d < 0) ↔ 0 < d ∧ d < 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1838_183827


namespace NUMINAMATH_CALUDE_square_difference_l1838_183813

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1838_183813


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1838_183837

theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
    sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1838_183837


namespace NUMINAMATH_CALUDE_probability_shirt_shorts_hat_l1838_183817

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 7

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 6

/-- The number of hats in the drawer -/
def num_hats : ℕ := 3

/-- The total number of articles of clothing in the drawer -/
def total_articles : ℕ := num_shirts + num_shorts + num_socks + num_hats

/-- The number of articles to be chosen -/
def num_chosen : ℕ := 3

theorem probability_shirt_shorts_hat : 
  (num_shirts.choose 1 * num_shorts.choose 1 * num_hats.choose 1 : ℚ) / 
  (total_articles.choose num_chosen) = 63 / 770 :=
sorry

end NUMINAMATH_CALUDE_probability_shirt_shorts_hat_l1838_183817


namespace NUMINAMATH_CALUDE_shakes_undetermined_l1838_183885

/-- Represents the price of a burger -/
def burger_price : ℝ := sorry

/-- Represents the price of a shake -/
def shake_price : ℝ := sorry

/-- Represents the price of a cola -/
def cola_price : ℝ := sorry

/-- Represents the number of shakes in the second purchase -/
def num_shakes_second : ℝ := sorry

/-- The total cost of the first purchase -/
def first_purchase : Prop :=
  3 * burger_price + 7 * shake_price + cola_price = 120

/-- The total cost of the second purchase -/
def second_purchase : Prop :=
  4 * burger_price + num_shakes_second * shake_price + cola_price = 164.5

/-- Theorem stating that the number of shakes in the second purchase cannot be uniquely determined -/
theorem shakes_undetermined (h1 : first_purchase) (h2 : second_purchase) :
  ∃ (x y : ℝ), x ≠ y ∧ 
    (4 * burger_price + x * shake_price + cola_price = 164.5) ∧
    (4 * burger_price + y * shake_price + cola_price = 164.5) :=
  sorry

end NUMINAMATH_CALUDE_shakes_undetermined_l1838_183885


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1838_183800

theorem cubic_equation_roots (a b : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (1 - 2 * Complex.I : ℂ) ^ 3 + a * (1 - 2 * Complex.I : ℂ) ^ 2 - (1 - 2 * Complex.I : ℂ) + b = 0 →
  a = 1 ∧ b = 15 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1838_183800


namespace NUMINAMATH_CALUDE_article_cost_l1838_183866

theorem article_cost (selling_price_high : ℝ) (selling_price_low : ℝ) (cost : ℝ) :
  selling_price_high = 600 →
  selling_price_low = 580 →
  selling_price_high - cost = 1.05 * (selling_price_low - cost) →
  cost = 180 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1838_183866


namespace NUMINAMATH_CALUDE_trig_identity_l1838_183854

theorem trig_identity : 
  (Real.cos (70 * π / 180) + Real.cos (50 * π / 180)) * 
  (Real.cos (310 * π / 180) + Real.cos (290 * π / 180)) + 
  (Real.cos (40 * π / 180) + Real.cos (160 * π / 180)) * 
  (Real.cos (320 * π / 180) - Real.cos (380 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1838_183854


namespace NUMINAMATH_CALUDE_right_handed_players_l1838_183883

/-- The number of right-handed players on a cricket team -/
theorem right_handed_players (total : ℕ) (throwers : ℕ) : 
  total = 55 →
  throwers = 37 →
  throwers ≤ total →
  (total - throwers) % 3 = 0 →  -- Ensures one-third of non-throwers can be left-handed
  49 = throwers + (total - throwers) - (total - throwers) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_l1838_183883


namespace NUMINAMATH_CALUDE_wendy_flowers_proof_l1838_183894

/-- The number of flowers in each bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- The number of flowers that wilted -/
def wilted_flowers : ℕ := 35

/-- The number of bouquets that can be made after some flowers wilted -/
def remaining_bouquets : ℕ := 2

/-- The initial number of flowers Wendy picked -/
def initial_flowers : ℕ := wilted_flowers + remaining_bouquets * flowers_per_bouquet

theorem wendy_flowers_proof : initial_flowers = 45 := by
  sorry

end NUMINAMATH_CALUDE_wendy_flowers_proof_l1838_183894


namespace NUMINAMATH_CALUDE_tetrahedra_arrangement_exists_l1838_183895

/-- A type representing a regular tetrahedron -/
structure Tetrahedron where
  -- Add necessary fields

/-- A type representing the arrangement of tetrahedra -/
structure Arrangement where
  tetrahedra : Set Tetrahedron
  lower_plane : Set (ℝ × ℝ × ℝ)
  upper_plane : Set (ℝ × ℝ × ℝ)

/-- Predicate to check if two planes are parallel -/
def are_parallel (plane1 plane2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a tetrahedron is between two planes -/
def is_between_planes (t : Tetrahedron) (lower upper : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a tetrahedron can be removed without moving others -/
def can_be_removed (t : Tetrahedron) (arr : Arrangement) : Prop :=
  sorry

/-- The main theorem statement -/
theorem tetrahedra_arrangement_exists :
  ∃ (arr : Arrangement),
    (∀ t ∈ arr.tetrahedra, is_between_planes t arr.lower_plane arr.upper_plane) ∧
    (are_parallel arr.lower_plane arr.upper_plane) ∧
    (Set.Infinite arr.tetrahedra) ∧
    (∀ t ∈ arr.tetrahedra, ¬can_be_removed t arr) :=
  sorry

end NUMINAMATH_CALUDE_tetrahedra_arrangement_exists_l1838_183895


namespace NUMINAMATH_CALUDE_time_difference_is_six_minutes_l1838_183868

/-- The time difference between walking and biking to work -/
def time_difference (blocks : ℕ) (walk_time_per_block : ℚ) (bike_time_per_block : ℚ) : ℚ :=
  blocks * (walk_time_per_block - bike_time_per_block)

/-- Proof that the time difference is 6 minutes -/
theorem time_difference_is_six_minutes :
  time_difference 9 1 (20 / 60) = 6 := by
  sorry

#eval time_difference 9 1 (20 / 60)

end NUMINAMATH_CALUDE_time_difference_is_six_minutes_l1838_183868


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l1838_183824

/-- The line equation 4y - 3x = 16 intersects the x-axis at (-16/3, 0) -/
theorem line_intersects_x_axis :
  let line := λ x y : ℚ => 4 * y - 3 * x = 16
  let x_axis := λ x y : ℚ => y = 0
  let intersection_point := (-16/3, 0)
  line intersection_point.1 intersection_point.2 ∧ x_axis intersection_point.1 intersection_point.2 := by
  sorry


end NUMINAMATH_CALUDE_line_intersects_x_axis_l1838_183824


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_1_range_of_a_when_not_p_implies_not_q_l1838_183881

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem range_of_x_when_a_is_1 :
  ∀ x : ℝ, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem range_of_a_when_not_p_implies_not_q :
  (∀ x a : ℝ, ¬(p x a) → ¬(q x)) ∧ 
  (∃ x a : ℝ, ¬(q x) ∧ p x a) →
  ∀ a : ℝ, 1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_1_range_of_a_when_not_p_implies_not_q_l1838_183881


namespace NUMINAMATH_CALUDE_expression_evaluation_l1838_183855

theorem expression_evaluation (b x : ℝ) (h : x = b + 4) :
  2*x - b + 5 = b + 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1838_183855


namespace NUMINAMATH_CALUDE_sum_of_twos_and_threes_1800_l1838_183897

/-- The number of ways to represent a positive integer as a sum of 2s and 3s -/
def waysToSum (n : ℕ) : ℕ :=
  (n / 6 + 1)

/-- 1800 can be represented as a sum of 2s and 3s in 301 ways -/
theorem sum_of_twos_and_threes_1800 : waysToSum 1800 = 301 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_twos_and_threes_1800_l1838_183897


namespace NUMINAMATH_CALUDE_volleyball_scoring_l1838_183859

/-- Volleyball team scoring problem -/
theorem volleyball_scoring
  (lizzie_score : ℕ)
  (nathalie_score : ℕ)
  (aimee_score : ℕ)
  (teammates_score : ℕ)
  (total_score : ℕ)
  (h1 : lizzie_score = 4)
  (h2 : nathalie_score > lizzie_score)
  (h3 : aimee_score = 2 * (lizzie_score + nathalie_score))
  (h4 : total_score = 50)
  (h5 : teammates_score = 17)
  (h6 : lizzie_score + nathalie_score + aimee_score + teammates_score = total_score) :
  nathalie_score = lizzie_score + 3 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_scoring_l1838_183859


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1838_183856

open Complex

theorem complex_exponential_sum (α β γ : ℝ) :
  exp (I * α) + exp (I * β) + exp (I * γ) = 1 + I →
  exp (-I * α) + exp (-I * β) + exp (-I * γ) = 1 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1838_183856


namespace NUMINAMATH_CALUDE_max_x_given_lcm_l1838_183836

theorem max_x_given_lcm (x : ℕ) : 
  (Nat.lcm x (Nat.lcm 15 21) = 105) → x ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_max_x_given_lcm_l1838_183836


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1838_183867

theorem smallest_positive_solution :
  let f : ℝ → ℝ := λ x => Real.sqrt (3 * x) - 5 * x
  ∃ x : ℝ, x > 0 ∧ f x = 0 ∧ ∀ y : ℝ, y > 0 ∧ f y = 0 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1838_183867


namespace NUMINAMATH_CALUDE_consecutive_even_product_divisible_l1838_183857

theorem consecutive_even_product_divisible (n : ℕ) : 
  ∃ k : ℕ, (2*n) * (2*n + 2) * (2*n + 4) * (2*n + 6) = 240 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_product_divisible_l1838_183857


namespace NUMINAMATH_CALUDE_no_integer_roots_l1838_183863

theorem no_integer_roots : ∀ x : ℤ, x^2 + 2^2018 * x + 2^2019 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l1838_183863


namespace NUMINAMATH_CALUDE_paint_cube_cost_l1838_183845

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem paint_cube_cost 
  (paint_cost : ℝ)       -- Cost of paint per kg in Rs
  (paint_coverage : ℝ)   -- Area covered by 1 kg of paint in sq. ft
  (cube_side : ℝ)        -- Length of cube side in feet
  (h1 : paint_cost = 20) -- Paint costs 20 Rs per kg
  (h2 : paint_coverage = 15) -- 1 kg of paint covers 15 sq. ft
  (h3 : cube_side = 5)   -- Cube side is 5 feet
  : ℝ :=
by
  -- The cost to paint the cube is 200 Rs
  sorry

#check paint_cube_cost

end NUMINAMATH_CALUDE_paint_cube_cost_l1838_183845


namespace NUMINAMATH_CALUDE_radio_survey_female_nonlisteners_l1838_183806

theorem radio_survey_female_nonlisteners (total_surveyed : ℕ) 
  (males_listen females_dont_listen total_listen total_dont_listen : ℕ) :
  total_surveyed = total_listen + total_dont_listen →
  males_listen ≤ total_listen →
  females_dont_listen ≤ total_dont_listen →
  total_surveyed = 255 →
  males_listen = 45 →
  total_listen = 120 →
  total_dont_listen = 135 →
  females_dont_listen = 87 →
  females_dont_listen = 87 :=
by sorry

end NUMINAMATH_CALUDE_radio_survey_female_nonlisteners_l1838_183806


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1838_183838

theorem decimal_to_fraction : (0.38 : ℚ) = 19 / 50 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1838_183838


namespace NUMINAMATH_CALUDE_max_value_of_x_l1838_183850

theorem max_value_of_x (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (prod_sum_eq : x * y + x * z + y * z = 3) : 
  x ≤ 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_x_l1838_183850


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l1838_183829

theorem sum_of_x_and_y_equals_two (x y : ℝ) 
  (h : 2 * x^2 + 2 * y^2 = 20 * x - 12 * y + 68) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l1838_183829


namespace NUMINAMATH_CALUDE_sqrt_one_fourth_l1838_183821

theorem sqrt_one_fourth : Real.sqrt (1 / 4) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_fourth_l1838_183821


namespace NUMINAMATH_CALUDE_exists_irrational_less_than_four_l1838_183818

theorem exists_irrational_less_than_four : ∃ x : ℝ, Irrational x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_irrational_less_than_four_l1838_183818


namespace NUMINAMATH_CALUDE_atomic_mass_scientific_notation_l1838_183861

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem atomic_mass_scientific_notation :
  toScientificNotation 0.00001992 = ScientificNotation.mk 1.992 (-5) sorry := by
  sorry

end NUMINAMATH_CALUDE_atomic_mass_scientific_notation_l1838_183861


namespace NUMINAMATH_CALUDE_sin_300_degrees_l1838_183893

theorem sin_300_degrees : Real.sin (300 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l1838_183893


namespace NUMINAMATH_CALUDE_polynomial_comparison_l1838_183839

theorem polynomial_comparison : ∀ x : ℝ, (x - 3) * (x - 2) > (x + 1) * (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_comparison_l1838_183839


namespace NUMINAMATH_CALUDE_min_value_x_plus_3y_l1838_183812

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y + x*y = 9) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3*b + a*b = 9 → x + 3*y ≤ a + 3*b :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_3y_l1838_183812


namespace NUMINAMATH_CALUDE_total_jellybeans_needed_l1838_183872

/-- The number of jellybeans needed to fill a large glass -/
def large_glass_jellybeans : ℕ := 50

/-- The number of large glasses to be filled -/
def num_large_glasses : ℕ := 5

/-- The number of small glasses to be filled -/
def num_small_glasses : ℕ := 3

/-- The number of jellybeans needed to fill a small glass -/
def small_glass_jellybeans : ℕ := large_glass_jellybeans / 2

/-- Theorem: The total number of jellybeans needed to fill all glasses is 325 -/
theorem total_jellybeans_needed : 
  num_large_glasses * large_glass_jellybeans + num_small_glasses * small_glass_jellybeans = 325 := by
  sorry

end NUMINAMATH_CALUDE_total_jellybeans_needed_l1838_183872


namespace NUMINAMATH_CALUDE_percentage_equality_l1838_183835

-- Define variables
variable (j k l m : ℝ)

-- Define the conditions
def condition1 : Prop := 1.25 * j = 0.25 * k
def condition2 : Prop := 1.5 * k = 0.5 * l
def condition3 : Prop := 0.2 * m = 7 * j

-- Theorem statement
theorem percentage_equality 
  (h1 : condition1 j k) 
  (h2 : condition2 k l) 
  (h3 : condition3 j m) : 
  1.75 * l = 0.75 * m := by sorry

end NUMINAMATH_CALUDE_percentage_equality_l1838_183835


namespace NUMINAMATH_CALUDE_unique_circle_construction_l1838_183826

/-- A line in a plane -/
structure Line : Type :=
  (l : Set (Real × Real))

/-- A point in a plane -/
structure Point : Type :=
  (x : Real) (y : Real)

/-- A circle in a plane -/
structure Circle : Type :=
  (center : Point) (radius : Real)

/-- Predicate to check if a point belongs to a line -/
def PointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Predicate to check if a circle passes through a point -/
def CirclePassesThrough (c : Circle) (p : Point) : Prop := sorry

/-- Predicate to check if a circle is tangent to a line at a point -/
def CircleTangentToLineAt (c : Circle) (l : Line) (p : Point) : Prop := sorry

/-- Main theorem: Existence and uniqueness of a circle passing through B and tangent to l at A -/
theorem unique_circle_construction (l : Line) (A B : Point) 
  (h1 : PointOnLine A l) 
  (h2 : ¬PointOnLine B l) : 
  ∃! k : Circle, CirclePassesThrough k B ∧ CircleTangentToLineAt k l A := by
  sorry

end NUMINAMATH_CALUDE_unique_circle_construction_l1838_183826


namespace NUMINAMATH_CALUDE_mollys_present_age_l1838_183849

def mollys_age_equation (x : ℕ) : Prop :=
  x + 18 = 5 * (x - 6)

theorem mollys_present_age : 
  ∃ (x : ℕ), mollys_age_equation x ∧ x = 12 :=
by sorry

end NUMINAMATH_CALUDE_mollys_present_age_l1838_183849


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l1838_183820

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 
  ∃ (d : ℕ), d > 0 ∧ (∀ (m : ℤ), (m^4 - m^2) % d = 0) ∧ 
  (∀ (k : ℕ), k > d → ∃ (l : ℤ), (l^4 - l^2) % k ≠ 0) ∧ d = 6 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l1838_183820


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l1838_183831

theorem min_perimeter_triangle (a b c : ℕ) : 
  a = 47 → b = 53 → c > 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  a + b + c ≥ 107 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l1838_183831


namespace NUMINAMATH_CALUDE_max_value_sqrt_x_1_minus_9x_l1838_183807

theorem max_value_sqrt_x_1_minus_9x (x : ℝ) (h1 : 0 < x) (h2 : x < 1/9) :
  ∃ (max_val : ℝ), max_val = 1/6 ∧ ∀ y, 0 < y ∧ y < 1/9 → Real.sqrt (y * (1 - 9*y)) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_x_1_minus_9x_l1838_183807


namespace NUMINAMATH_CALUDE_circular_garden_radius_l1838_183892

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 6) * π * r^2 → r = 12 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l1838_183892


namespace NUMINAMATH_CALUDE_consecutive_sum_39_largest_l1838_183869

theorem consecutive_sum_39_largest (n m : ℕ) : 
  n + 1 = m → n + m = 39 → m = 20 := by
sorry

end NUMINAMATH_CALUDE_consecutive_sum_39_largest_l1838_183869


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1838_183832

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + a + 3 ≥ 0) ↔ a ∈ Set.Ici (0 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1838_183832


namespace NUMINAMATH_CALUDE_colonization_combinations_l1838_183891

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 8

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 7

/-- Represents the colonization effort required for an Earth-like planet -/
def earth_like_effort : ℕ := 2

/-- Represents the colonization effort required for a Mars-like planet -/
def mars_like_effort : ℕ := 1

/-- Represents the total available colonization effort -/
def total_effort : ℕ := 18

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Theorem stating the number of distinct combinations of planets that can be fully colonized -/
theorem colonization_combinations : 
  (choose earth_like_planets 8 * choose mars_like_planets 2) +
  (choose earth_like_planets 7 * choose mars_like_planets 4) +
  (choose earth_like_planets 6 * choose mars_like_planets 6) = 497 := by
  sorry

end NUMINAMATH_CALUDE_colonization_combinations_l1838_183891


namespace NUMINAMATH_CALUDE_unit_digit_of_15_100_pow_20_l1838_183830

-- Define a function to get the unit digit of a natural number
def unitDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem unit_digit_of_15_100_pow_20 :
  unitDigit ((15^100)^20) = 5 := by sorry

end NUMINAMATH_CALUDE_unit_digit_of_15_100_pow_20_l1838_183830


namespace NUMINAMATH_CALUDE_probability_of_selection_l1838_183823

/-- The number of students -/
def n : ℕ := 5

/-- The number of students to be chosen -/
def k : ℕ := 2

/-- Binomial coefficient -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The probability of selecting 2 students out of 5, where student A is selected and student B is not -/
theorem probability_of_selection : 
  (choose (n - 2) (k - 1) : ℚ) / (choose n k) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_l1838_183823


namespace NUMINAMATH_CALUDE_percentage_of_360_is_180_l1838_183899

theorem percentage_of_360_is_180 : 
  let whole : ℝ := 360
  let part : ℝ := 180
  let percentage : ℝ := (part / whole) * 100
  percentage = 50 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_is_180_l1838_183899


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1838_183833

theorem simple_interest_problem (P R : ℚ) : 
  P + (P * R * 2) / 100 = 820 → 
  P + (P * R * 6) / 100 = 1020 → 
  P = 720 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1838_183833


namespace NUMINAMATH_CALUDE_n_value_is_six_l1838_183886

/-- The cost of a water bottle in cents -/
def water_cost : ℕ := 50

/-- The cost of a fruit in cents -/
def fruit_cost : ℕ := 25

/-- The cost of a snack in cents -/
def snack_cost : ℕ := 100

/-- The number of water bottles in a bundle -/
def water_in_bundle : ℕ := 1

/-- The number of snacks in a bundle -/
def snacks_in_bundle : ℕ := 3

/-- The number of fruits in a bundle -/
def fruits_in_bundle : ℕ := 2

/-- The regular selling price of a bundle in cents -/
def bundle_price : ℕ := 460

/-- The special price for every nth bundle in cents -/
def special_price : ℕ := 200

/-- The function to calculate the cost of a regular bundle in cents -/
def bundle_cost : ℕ := 
  water_cost * water_in_bundle + 
  snack_cost * snacks_in_bundle + 
  fruit_cost * fruits_in_bundle

/-- The function to calculate the profit from a regular bundle in cents -/
def bundle_profit : ℕ := bundle_price - bundle_cost

/-- The function to calculate the cost of a special bundle in cents -/
def special_bundle_cost : ℕ := bundle_cost + snack_cost

/-- The function to calculate the loss from a special bundle in cents -/
def special_bundle_loss : ℕ := special_bundle_cost - special_price

/-- Theorem stating that the value of n is 6 -/
theorem n_value_is_six : 
  ∃ n : ℕ, n > 0 ∧ (n - 1) * bundle_profit = special_bundle_loss ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_n_value_is_six_l1838_183886


namespace NUMINAMATH_CALUDE_bono_jelly_beans_l1838_183887

/-- Given the number of jelly beans for Alida, Bono, and Cate, prove that Bono has 4t - 1 jelly beans. -/
theorem bono_jelly_beans (t : ℕ) (A B C : ℕ) : 
  A + B = 6 * t + 3 →
  A + C = 4 * t + 5 →
  B + C = 6 * t →
  B = 4 * t - 1 := by
  sorry

end NUMINAMATH_CALUDE_bono_jelly_beans_l1838_183887


namespace NUMINAMATH_CALUDE_money_distribution_theorem_l1838_183840

/-- Represents the money distribution problem --/
structure MoneyDistribution where
  total : ℚ
  first_share : ℚ
  second_share : ℚ
  third_share : ℚ

/-- Checks if the given distribution satisfies the initial conditions --/
def valid_initial_distribution (d : MoneyDistribution) : Prop :=
  d.first_share = d.total / 2 ∧
  d.second_share = d.total / 3 ∧
  d.third_share = d.total / 6

/-- Calculates the amount each person saves --/
def savings (d : MoneyDistribution) : (ℚ × ℚ × ℚ) :=
  (d.first_share / 2, d.second_share / 3, d.third_share / 6)

/-- Calculates the total amount saved --/
def total_savings (d : MoneyDistribution) : ℚ :=
  let (s1, s2, s3) := savings d
  s1 + s2 + s3

/-- Checks if the final distribution is equal for all three people --/
def equal_final_distribution (d : MoneyDistribution) : Prop :=
  let total_saved := total_savings d
  d.first_share + total_saved / 3 =
  d.second_share + total_saved / 3 ∧
  d.second_share + total_saved / 3 =
  d.third_share + total_saved / 3

/-- The main theorem stating the existence of a valid solution --/
theorem money_distribution_theorem :
  ∃ (d : MoneyDistribution),
    valid_initial_distribution d ∧
    equal_final_distribution d ∧
    d.first_share = 23.5 ∧
    d.second_share = 15 + 2/3 ∧
    d.third_share = 7 + 5/6 :=
sorry

end NUMINAMATH_CALUDE_money_distribution_theorem_l1838_183840


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l1838_183852

theorem trigonometric_calculations :
  (2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180) + Real.sqrt 9 = Real.sqrt 3) ∧
  (Real.cos (30 * π / 180) / (1 + Real.sin (30 * π / 180)) + Real.tan (60 * π / 180) = 4 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_calculations_l1838_183852


namespace NUMINAMATH_CALUDE_money_distribution_l1838_183880

theorem money_distribution (a b : ℚ) : 
  (a + b / 2 = 50) → 
  (b + 2 * a / 3 = 50) → 
  (a = 37.5 ∧ b = 25) := by sorry

end NUMINAMATH_CALUDE_money_distribution_l1838_183880


namespace NUMINAMATH_CALUDE_group_intersection_theorem_l1838_183878

theorem group_intersection_theorem (n : ℕ) (groups : Finset (Finset (Fin n))) :
  n = 1997 →
  (∀ g ∈ groups, Finset.card g = 3) →
  Finset.card groups ≥ 1998 →
  ∃ g₁ g₂ : Finset (Fin n), g₁ ∈ groups ∧ g₂ ∈ groups ∧ g₁ ≠ g₂ ∧ Finset.card (g₁ ∩ g₂) = 1 :=
by sorry


end NUMINAMATH_CALUDE_group_intersection_theorem_l1838_183878


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1838_183843

theorem smallest_number_with_remainders : ∃! n : ℕ,
  (∀ k ∈ Finset.range 10, n % (k + 3) = k + 2) ∧
  (∀ m : ℕ, m < n → ∃ k ∈ Finset.range 10, m % (k + 3) ≠ k + 2) :=
by
  use 27719
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1838_183843


namespace NUMINAMATH_CALUDE_debate_students_difference_l1838_183805

theorem debate_students_difference (s1 s2 s3 : ℕ) : 
  s1 = 2 * s2 →
  s3 = 200 →
  s1 + s2 + s3 = 920 →
  s2 - s3 = 40 := by
sorry

end NUMINAMATH_CALUDE_debate_students_difference_l1838_183805


namespace NUMINAMATH_CALUDE_birds_meeting_point_l1838_183847

/-- Theorem: Meeting point of two birds flying towards each other --/
theorem birds_meeting_point 
  (total_distance : ℝ) 
  (speed_bird1 : ℝ) 
  (speed_bird2 : ℝ) 
  (h1 : total_distance = 20)
  (h2 : speed_bird1 = 4)
  (h3 : speed_bird2 = 1) :
  (speed_bird1 * total_distance) / (speed_bird1 + speed_bird2) = 16 := by
  sorry

#check birds_meeting_point

end NUMINAMATH_CALUDE_birds_meeting_point_l1838_183847


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1838_183816

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x^2 - x - 2 < 0 → |x| < 2) ∧
  (∃ x : ℝ, |x| < 2 ∧ x^2 - x - 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1838_183816


namespace NUMINAMATH_CALUDE_octagon_diagonals_l1838_183811

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Proof that an octagon has 20 diagonals -/
theorem octagon_diagonals : num_diagonals 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l1838_183811


namespace NUMINAMATH_CALUDE_point_on_line_point_40_161_on_line_l1838_183841

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Given three points on a line, check if a fourth point is on the same line -/
theorem point_on_line (p1 p2 p3 p4 : Point)
  (h1 : collinear p1 p2 p3) : 
  collinear p1 p2 p4 ∧ collinear p2 p3 p4 → collinear p1 p3 p4 := by sorry

/-- The main theorem to prove -/
theorem point_40_161_on_line : 
  let p1 : Point := ⟨2, 9⟩
  let p2 : Point := ⟨6, 25⟩
  let p3 : Point := ⟨10, 41⟩
  let p4 : Point := ⟨40, 161⟩
  collinear p1 p2 p3 → collinear p1 p2 p4 ∧ collinear p2 p3 p4 := by sorry

end NUMINAMATH_CALUDE_point_on_line_point_40_161_on_line_l1838_183841
