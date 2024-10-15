import Mathlib

namespace NUMINAMATH_CALUDE_jonathan_first_name_length_l1199_119998

/-- The number of letters in Jonathan's first name -/
def jonathan_first_name : ℕ := by sorry

/-- The number of letters in Jonathan's surname -/
def jonathan_surname : ℕ := 10

/-- The number of letters in Jonathan's sister's first name -/
def sister_first_name : ℕ := 5

/-- The number of letters in Jonathan's sister's surname -/
def sister_surname : ℕ := 10

/-- The total number of letters in both their names -/
def total_letters : ℕ := 33

theorem jonathan_first_name_length :
  jonathan_first_name = 8 :=
by
  have h1 : jonathan_first_name + jonathan_surname + sister_first_name + sister_surname = total_letters := by sorry
  sorry

end NUMINAMATH_CALUDE_jonathan_first_name_length_l1199_119998


namespace NUMINAMATH_CALUDE_maxim_birth_probability_l1199_119914

/-- The year Maxim starts first grade -/
def start_year : ℕ := 2014

/-- Maxim's age when starting first grade -/
def start_age : ℕ := 6

/-- The day of the year when Maxim starts first grade (1st September) -/
def start_day : ℕ := 244

/-- The number of days in a year (assuming non-leap year) -/
def days_in_year : ℕ := 365

/-- The year we're interested in for Maxim's birth -/
def birth_year_of_interest : ℕ := 2008

/-- The number of days from 1st January to 31st August in 2008 (leap year) -/
def days_in_2008_until_august : ℕ := 244

theorem maxim_birth_probability :
  let total_possible_days := days_in_year
  let favorable_days := days_in_2008_until_august
  (favorable_days : ℚ) / total_possible_days = 244 / 365 := by
  sorry

end NUMINAMATH_CALUDE_maxim_birth_probability_l1199_119914


namespace NUMINAMATH_CALUDE_ellipse_m_value_l1199_119951

/-- Given an ellipse with equation x²/10 + y²/m = 1, foci on the y-axis, and major axis 8, prove that m = 16 -/
theorem ellipse_m_value (x y m : ℝ) : 
  (∀ x y, x^2 / 10 + y^2 / m = 1) →  -- Ellipse equation
  (∃ c, c > 0 ∧ ∀ x, x^2 / 10 + (y + c)^2 / m = 1 ∧ x^2 / 10 + (y - c)^2 / m = 1) →  -- Foci on y-axis
  (∃ y, y^2 / m = 1 ∧ y = 4) →  -- Major axis is 8 (semi-major axis is 4)
  m = 16 := by
sorry


end NUMINAMATH_CALUDE_ellipse_m_value_l1199_119951


namespace NUMINAMATH_CALUDE_school_sports_probabilities_l1199_119993

/-- Represents a school with boys and girls, some of whom like sports -/
structure School where
  girls : ℕ
  boys : ℕ
  boys_like_sports : ℕ
  girls_like_sports : ℕ
  boys_ratio : boys = 3 * girls / 2
  boys_sports_ratio : boys_like_sports = 2 * boys / 5
  girls_sports_ratio : girls_like_sports = girls / 5

/-- The probability that a randomly selected student likes sports -/
def prob_likes_sports (s : School) : ℚ :=
  (s.boys_like_sports + s.girls_like_sports : ℚ) / (s.boys + s.girls)

/-- The probability that a randomly selected student who likes sports is a boy -/
def prob_boy_given_sports (s : School) : ℚ :=
  (s.boys_like_sports : ℚ) / (s.boys_like_sports + s.girls_like_sports)

theorem school_sports_probabilities (s : School) :
  prob_likes_sports s = 8/25 ∧ prob_boy_given_sports s = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_school_sports_probabilities_l1199_119993


namespace NUMINAMATH_CALUDE_nancys_weight_calculation_l1199_119991

/-- Nancy's weight in pounds -/
def nancys_weight : ℝ := 90

/-- Nancy's daily water intake as a percentage of her body weight -/
def water_intake_percentage : ℝ := 60

/-- Nancy's daily water intake in pounds -/
def daily_water_intake : ℝ := 54

theorem nancys_weight_calculation :
  nancys_weight * (water_intake_percentage / 100) = daily_water_intake :=
by sorry

end NUMINAMATH_CALUDE_nancys_weight_calculation_l1199_119991


namespace NUMINAMATH_CALUDE_trees_planted_total_l1199_119933

/-- Calculates the total number of trees planted given the number of apricot trees and the ratio of peach to apricot trees. -/
def total_trees (apricot_trees : ℕ) (peach_to_apricot_ratio : ℕ) : ℕ :=
  apricot_trees + peach_to_apricot_ratio * apricot_trees

/-- Theorem stating that given the specific conditions, the total number of trees planted is 232. -/
theorem trees_planted_total : total_trees 58 3 = 232 := by
  sorry

#eval total_trees 58 3

end NUMINAMATH_CALUDE_trees_planted_total_l1199_119933


namespace NUMINAMATH_CALUDE_division_remainder_l1199_119958

theorem division_remainder : ∃ (q : ℕ), 37 = 8 * q + 5 ∧ 5 < 8 := by sorry

end NUMINAMATH_CALUDE_division_remainder_l1199_119958


namespace NUMINAMATH_CALUDE_sets_with_property_P_l1199_119988

-- Define property P
def property_P (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ) (k : ℝ), (x, y) ∈ M → 0 < k → k < 1 → (k * x, k * y) ∈ M

-- Define the four sets
def set1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 ≥ p.2}
def set2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 < 1}
def set3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 + 2 * p.2 = 0}
def set4 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^3 + p.2^3 - p.1^2 * p.2 = 0}

-- Theorem stating which sets possess property P
theorem sets_with_property_P :
  property_P set2 ∧ property_P set4 ∧ ¬property_P set1 ∧ ¬property_P set3 := by
  sorry

end NUMINAMATH_CALUDE_sets_with_property_P_l1199_119988


namespace NUMINAMATH_CALUDE_total_earned_is_144_l1199_119932

/-- Calculates the total money earned from selling milk and butter --/
def total_money_earned (milk_price : ℚ) (butter_conversion : ℚ) (butter_price : ℚ) 
  (num_cows : ℕ) (milk_per_cow : ℚ) (num_customers : ℕ) (milk_per_customer : ℚ) : ℚ :=
  let total_milk := num_cows * milk_per_cow
  let sold_milk := min total_milk (num_customers * milk_per_customer)
  let remaining_milk := total_milk - sold_milk
  let butter_sticks := remaining_milk * butter_conversion
  milk_price * sold_milk + butter_price * butter_sticks

/-- Theorem stating that the total money earned is $144 given the problem conditions --/
theorem total_earned_is_144 :
  total_money_earned 3 2 (3/2) 12 4 6 6 = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_earned_is_144_l1199_119932


namespace NUMINAMATH_CALUDE_apples_left_is_ten_l1199_119977

/-- Represents the number of apples picked by Mike -/
def mike_apples : ℕ := 12

/-- Represents the number of apples eaten by Nancy -/
def nancy_apples : ℕ := 7

/-- Represents the number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- Represents the number of pears picked by Keith -/
def keith_pears : ℕ := 4

/-- Represents the number of apples picked by Christine -/
def christine_apples : ℕ := 10

/-- Represents the number of pears picked by Christine -/
def christine_pears : ℕ := 3

/-- Represents the number of bananas picked by Christine -/
def christine_bananas : ℕ := 5

/-- Represents the number of apples eaten by Greg -/
def greg_apples : ℕ := 9

/-- Represents the number of peaches picked by an unknown person -/
def unknown_peaches : ℕ := 14

/-- Represents the number of plums picked by an unknown person -/
def unknown_plums : ℕ := 7

/-- Represents the ratio of pears picked to apples disappeared -/
def pears_per_apple : ℕ := 3

/-- Theorem stating that the number of apples left is 10 -/
theorem apples_left_is_ten : 
  mike_apples + keith_apples + christine_apples - 
  nancy_apples - greg_apples - 
  ((keith_pears + christine_pears) / pears_per_apple) = 10 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_is_ten_l1199_119977


namespace NUMINAMATH_CALUDE_amanda_to_kimberly_distance_l1199_119907

/-- The distance between two houses given a constant speed and time -/
def distance_between_houses (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: Amanda's house is 6 miles away from Kimberly's house -/
theorem amanda_to_kimberly_distance :
  distance_between_houses 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_amanda_to_kimberly_distance_l1199_119907


namespace NUMINAMATH_CALUDE_expression_evaluation_l1199_119902

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  (1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1199_119902


namespace NUMINAMATH_CALUDE_square_side_length_l1199_119904

/-- Given a square ABCD with side length x, prove that x = 12 under the given conditions --/
theorem square_side_length (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive
  (h2 : x^2 - (1/2) * ((x-5) * (x-4)) - (7/2) * (x-7) - 2*(x-1) - 3.5 = 78) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1199_119904


namespace NUMINAMATH_CALUDE_percentage_of_x_minus_y_l1199_119967

theorem percentage_of_x_minus_y (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (20 / 100) * (x + y) →
  y = (20 / 100) * x →
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_x_minus_y_l1199_119967


namespace NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_cubed_l1199_119942

theorem ceiling_negative_seven_fourths_cubed : ⌈(-7/4)^3⌉ = -5 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_cubed_l1199_119942


namespace NUMINAMATH_CALUDE_outer_circle_radius_l1199_119996

/-- Given a circular race track with an inner circumference of 440 meters and a width of 14 meters,
    the radius of the outer circle is equal to (440 / (2 * π)) + 14. -/
theorem outer_circle_radius (inner_circumference : ℝ) (track_width : ℝ)
    (h1 : inner_circumference = 440)
    (h2 : track_width = 14) :
    (inner_circumference / (2 * Real.pi) + track_width) = (440 / (2 * Real.pi) + 14) := by
  sorry

end NUMINAMATH_CALUDE_outer_circle_radius_l1199_119996


namespace NUMINAMATH_CALUDE_ocean_area_scientific_notation_l1199_119923

theorem ocean_area_scientific_notation : 
  361000000 = 3.61 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_ocean_area_scientific_notation_l1199_119923


namespace NUMINAMATH_CALUDE_f_x_plus_3_l1199_119987

/-- Given a function f: ℝ → ℝ defined as f(x) = x^2 for all real numbers x,
    prove that f(x + 3) = x^2 + 6x + 9 for all real numbers x. -/
theorem f_x_plus_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x^2) :
  ∀ x : ℝ, f (x + 3) = x^2 + 6*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_f_x_plus_3_l1199_119987


namespace NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l1199_119920

def consecutive_integers (start : Int) (n : Nat) : List Int :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List Int) : List Int :=
  l.filter (λ x => x > 0)

def range (l : List Int) : Int :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_list (k : List Int) :
  k = consecutive_integers (-4) 10 →
  range (positive_integers k) = 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l1199_119920


namespace NUMINAMATH_CALUDE_orange_ribbons_l1199_119979

theorem orange_ribbons (total : ℕ) (yellow purple orange silver : ℕ) : 
  yellow + purple + orange + silver = total →
  4 * yellow = total →
  3 * purple = total →
  6 * orange = total →
  silver = 40 →
  orange = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_ribbons_l1199_119979


namespace NUMINAMATH_CALUDE_equation_solutions_l1199_119905

theorem equation_solutions : 
  (∃ x : ℝ, x^2 - 2*x - 8 = 0 ↔ x = -2 ∨ x = 4) ∧
  (∃ x : ℝ, (x + 1)^2 = 4*x^2 ↔ x = -1/3 ∨ x = 1) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1199_119905


namespace NUMINAMATH_CALUDE_correct_num_kettles_l1199_119909

/-- The number of kettles of hawks the ornithologists are tracking -/
def num_kettles : ℕ := 6

/-- The average number of pregnancies per kettle -/
def pregnancies_per_kettle : ℕ := 15

/-- The number of babies per pregnancy -/
def babies_per_pregnancy : ℕ := 4

/-- The survival rate of babies -/
def survival_rate : ℚ := 3/4

/-- The total number of expected babies this season -/
def total_babies : ℕ := 270

/-- Theorem stating that the number of kettles is correct given the conditions -/
theorem correct_num_kettles : 
  num_kettles = total_babies / (pregnancies_per_kettle * babies_per_pregnancy * survival_rate) :=
sorry

end NUMINAMATH_CALUDE_correct_num_kettles_l1199_119909


namespace NUMINAMATH_CALUDE_vector_to_line_parallel_to_direction_l1199_119941

/-- A line parameterized by x = 3t + 3, y = 2t + 3 -/
def parametric_line (t : ℝ) : ℝ × ℝ := (3*t + 3, 2*t + 3)

/-- The vector we want to prove is correct -/
def vector : ℝ × ℝ := (9, 6)

/-- The direction vector -/
def direction : ℝ × ℝ := (3, 2)

theorem vector_to_line_parallel_to_direction :
  ∃ (t : ℝ), parametric_line t = vector ∧ 
  ∃ (k : ℝ), vector.1 = k * direction.1 ∧ vector.2 = k * direction.2 := by
sorry

end NUMINAMATH_CALUDE_vector_to_line_parallel_to_direction_l1199_119941


namespace NUMINAMATH_CALUDE_blue_marbles_count_l1199_119961

theorem blue_marbles_count (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) 
  (h1 : total = 20)
  (h2 : red = 9)
  (h3 : prob_red_or_white = 3/4) :
  ∃ blue : ℕ, blue = 5 ∧ 
    (blue + red : ℚ) / total + prob_red_or_white = 1 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l1199_119961


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l1199_119968

theorem sin_cos_sum_equals_one : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l1199_119968


namespace NUMINAMATH_CALUDE_article_price_proof_l1199_119908

/-- The normal price of an article before discounts -/
def normal_price : ℝ := 150

/-- The final price after discounts -/
def final_price : ℝ := 108

/-- The first discount rate -/
def discount1 : ℝ := 0.1

/-- The second discount rate -/
def discount2 : ℝ := 0.2

theorem article_price_proof :
  normal_price * (1 - discount1) * (1 - discount2) = final_price := by
  sorry

end NUMINAMATH_CALUDE_article_price_proof_l1199_119908


namespace NUMINAMATH_CALUDE_difference_h_f_l1199_119955

theorem difference_h_f (e f g h : ℕ+) 
  (he : e^5 = f^4)
  (hg : g^3 = h^2)
  (hge : g - e = 31) : 
  h - f = 971 := by sorry

end NUMINAMATH_CALUDE_difference_h_f_l1199_119955


namespace NUMINAMATH_CALUDE_intersection_M_N_l1199_119970

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1199_119970


namespace NUMINAMATH_CALUDE_smallest_solution_of_quadratic_l1199_119990

theorem smallest_solution_of_quadratic : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 10*x - 24
  ∃ (x : ℝ), f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = -12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_quadratic_l1199_119990


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1199_119917

theorem condition_sufficient_not_necessary (a b : ℝ) :
  ((1 < b) ∧ (b < a)) → (a - 1 > |b - 1|) ∧
  ¬(∀ a b : ℝ, (a - 1 > |b - 1|) → ((1 < b) ∧ (b < a))) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1199_119917


namespace NUMINAMATH_CALUDE_perpendicular_vectors_acute_angle_vectors_l1199_119906

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (2, 1)

/-- Theorem for part 1 -/
theorem perpendicular_vectors (m : ℝ) : 
  (((1/2 : ℝ) • a.1 + b.1, (1/2 : ℝ) • a.2 + b.2) • (a.1 + m * b.1, a.2 + m * b.2) = 0) ↔ 
  (m = -5/12) :=
sorry

/-- Theorem for part 2 -/
theorem acute_angle_vectors (m : ℝ) :
  (((1/2 : ℝ) • a.1 + b.1, (1/2 : ℝ) • a.2 + b.2) • (a.1 + m * b.1, a.2 + m * b.2) > 0 ∧
   ((1/2 : ℝ) • a.1 + b.1) / ((1/2 : ℝ) • a.2 + b.2) ≠ (a.1 + m * b.1) / (a.2 + m * b.2)) ↔
  (m > -5/12 ∧ m ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_acute_angle_vectors_l1199_119906


namespace NUMINAMATH_CALUDE_sam_current_age_l1199_119934

/-- Sam's current age -/
def sam_age : ℕ := 46

/-- Drew's current age -/
def drew_age : ℕ := 12

/-- Theorem stating Sam's current age is 46, given the conditions -/
theorem sam_current_age :
  (sam_age + 5 = 3 * (drew_age + 5)) → sam_age = 46 := by
  sorry

end NUMINAMATH_CALUDE_sam_current_age_l1199_119934


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l1199_119940

/-- Calculates the profit percentage for a shopkeeper who sold 30 articles at the cost price of 35 articles -/
theorem shopkeeper_profit_percentage :
  let articles_sold : ℕ := 30
  let cost_price_articles : ℕ := 35
  let profit_ratio : ℚ := (cost_price_articles - articles_sold) / articles_sold
  profit_ratio * 100 = 5 / 30 * 100 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l1199_119940


namespace NUMINAMATH_CALUDE_f_zeros_l1199_119960

noncomputable def f (x : ℝ) : ℝ := (1/3) * x - Real.log x

theorem f_zeros (h : ∀ x, x > 0 → f x = (1/3) * x - Real.log x) :
  (∀ x, 1/Real.exp 1 < x ∧ x < 1 → f x ≠ 0) ∧
  (∃ x, 1 < x ∧ x < Real.exp 1 ∧ f x = 0) :=
sorry

end NUMINAMATH_CALUDE_f_zeros_l1199_119960


namespace NUMINAMATH_CALUDE_janine_read_five_books_last_month_l1199_119995

/-- The number of books Janine read last month -/
def last_month_books : ℕ := sorry

/-- The number of books Janine read this month -/
def this_month_books : ℕ := 2 * last_month_books

/-- The number of pages in each book -/
def pages_per_book : ℕ := 10

/-- The total number of pages Janine read in two months -/
def total_pages : ℕ := 150

theorem janine_read_five_books_last_month :
  last_month_books = 5 :=
by sorry

end NUMINAMATH_CALUDE_janine_read_five_books_last_month_l1199_119995


namespace NUMINAMATH_CALUDE_unique_solution_for_f_equals_two_l1199_119971

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x - 4
  else if x ≤ 2 then x^2 - 1
  else x/3 + 2

theorem unique_solution_for_f_equals_two :
  ∃! x : ℝ, f x = 2 ∧ x = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_f_equals_two_l1199_119971


namespace NUMINAMATH_CALUDE_game_points_difference_l1199_119935

theorem game_points_difference (eric_points mark_points samanta_points : ℕ) : 
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points > mark_points →
  samanta_points + mark_points + eric_points = 32 →
  samanta_points - mark_points = 8 :=
by sorry

end NUMINAMATH_CALUDE_game_points_difference_l1199_119935


namespace NUMINAMATH_CALUDE_nth_equation_l1199_119976

theorem nth_equation (n : ℕ) (hn : n > 0) :
  1 + 1 / n - 2 / (2 * n + 1) = (2 * n^2 + n + 1) / (n * (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l1199_119976


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l1199_119946

/-- The volume of a tetrahedron with given edge lengths -/
def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ := sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths -/
theorem volume_of_specific_tetrahedron :
  tetrahedron_volume 3 4 6 5 (Real.sqrt 37) (2 * Real.sqrt 10) = (4 * Real.sqrt 77) / 3 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l1199_119946


namespace NUMINAMATH_CALUDE_b_is_ten_l1199_119964

/-- The base of the number system that satisfies the given equation -/
def b : ℕ := sorry

/-- The equation that b must satisfy -/
axiom eq_condition : (3 * b + 5)^2 = 1 * b^3 + 2 * b^2 + 2 * b + 5

/-- Proof that b is the only positive integer solution -/
theorem b_is_ten : b = 10 := by sorry

end NUMINAMATH_CALUDE_b_is_ten_l1199_119964


namespace NUMINAMATH_CALUDE_extended_segment_vector_representation_l1199_119950

/-- Given a line segment AB extended past B to Q with AQ:QB = 7:2,
    prove that Q = (2/9)A + (7/9)B -/
theorem extended_segment_vector_representation 
  (A B Q : ℝ × ℝ) -- Points in 2D plane
  (h : (dist A Q) / (dist Q B) = 7 / 2) -- AQ:QB = 7:2
  : ∃ (x y : ℝ), x = 2/9 ∧ y = 7/9 ∧ Q = x • A + y • B :=
by sorry


end NUMINAMATH_CALUDE_extended_segment_vector_representation_l1199_119950


namespace NUMINAMATH_CALUDE_valid_student_counts_exists_valid_distributions_l1199_119944

/-- Represents the distribution of students in groups -/
structure StudentDistribution where
  total_groups : ℕ
  groups_with_13 : ℕ
  total_students : ℕ

/-- Checks if a given distribution satisfies the problem conditions -/
def is_valid_distribution (d : StudentDistribution) : Prop :=
  d.total_groups = 6 ∧
  d.groups_with_13 = 4 ∧
  (d.total_students = 76 ∨ d.total_students = 80)

/-- Theorem stating the only valid total numbers of students -/
theorem valid_student_counts :
  ∀ d : StudentDistribution,
    is_valid_distribution d →
    (d.total_students = 76 ∨ d.total_students = 80) :=
by
  sorry

/-- Theorem proving the existence of valid distributions -/
theorem exists_valid_distributions :
  ∃ d₁ d₂ : StudentDistribution,
    is_valid_distribution d₁ ∧
    is_valid_distribution d₂ ∧
    d₁.total_students = 76 ∧
    d₂.total_students = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_valid_student_counts_exists_valid_distributions_l1199_119944


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1199_119915

/-- Given a group of 9 persons where the average weight increases by 1.5 kg
    after replacing one person with a new person weighing 78.5 kg,
    prove that the weight of the replaced person was 65 kg. -/
theorem weight_of_replaced_person
  (n : ℕ) -- number of persons in the group
  (avg_increase : ℝ) -- increase in average weight
  (new_weight : ℝ) -- weight of the new person
  (h1 : n = 9) -- there are 9 persons in the group
  (h2 : avg_increase = 1.5) -- average weight increases by 1.5 kg
  (h3 : new_weight = 78.5) -- new person weighs 78.5 kg
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1199_119915


namespace NUMINAMATH_CALUDE_triangle_max_third_side_l1199_119969

theorem triangle_max_third_side (a b : ℝ) (ha : a = 4) (hb : b = 9) :
  ∃ (c : ℕ), c ≤ 12 ∧ 
  (∀ (d : ℕ), d > c → ¬(a + b > d ∧ a + d > b ∧ b + d > a)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_third_side_l1199_119969


namespace NUMINAMATH_CALUDE_car_hire_total_amount_l1199_119943

/-- Represents the hire charges for a car -/
structure CarHire where
  hourly_rate : ℕ
  hours_a : ℕ
  hours_b : ℕ
  hours_c : ℕ
  amount_b : ℕ

/-- Calculates the total amount paid for hiring the car -/
def total_amount (hire : CarHire) : ℕ :=
  hire.hourly_rate * (hire.hours_a + hire.hours_b + hire.hours_c)

/-- Theorem stating the total amount paid for the car hire -/
theorem car_hire_total_amount (hire : CarHire)
  (h1 : hire.hours_a = 7)
  (h2 : hire.hours_b = 8)
  (h3 : hire.hours_c = 11)
  (h4 : hire.amount_b = 160)
  (h5 : hire.hourly_rate = hire.amount_b / hire.hours_b) :
  total_amount hire = 520 := by
  sorry

#check car_hire_total_amount

end NUMINAMATH_CALUDE_car_hire_total_amount_l1199_119943


namespace NUMINAMATH_CALUDE_special_function_value_l1199_119939

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  f 0 = 1008 ∧
  (∀ x : ℝ, f (x + 4) - f x ≤ 2 * (x + 1)) ∧
  (∀ x : ℝ, f (x + 12) - f x ≥ 6 * (x + 5))

/-- The main theorem -/
theorem special_function_value (f : ℝ → ℝ) (h : SpecialFunction f) :
  f 2016 / 2016 = 504 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l1199_119939


namespace NUMINAMATH_CALUDE_three_numbers_ratio_l1199_119948

theorem three_numbers_ratio (a b c : ℝ) : 
  (a : ℝ) / 2 = (b : ℝ) / 3 ∧ (b : ℝ) / 3 = (c : ℝ) / 4 →
  a^2 + b^2 + c^2 = 725 →
  (a = 10 ∧ b = 15 ∧ c = 20) ∨ (a = -10 ∧ b = -15 ∧ c = -20) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_ratio_l1199_119948


namespace NUMINAMATH_CALUDE_cube_edge_length_l1199_119949

theorem cube_edge_length (x : ℝ) : x > 0 → 6 * x^2 = 1014 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1199_119949


namespace NUMINAMATH_CALUDE_youngest_child_age_proof_l1199_119924

def youngest_child_age (n : ℕ) (interval : ℕ) (total_age : ℕ) : ℕ :=
  (total_age - (n - 1) * n * interval / 2) / n

theorem youngest_child_age_proof (n : ℕ) (interval : ℕ) (total_age : ℕ) 
  (h1 : n = 5)
  (h2 : interval = 3)
  (h3 : total_age = 50)
  (h4 : youngest_child_age n interval total_age * 2 = youngest_child_age n interval total_age + (n - 1) * interval) :
  youngest_child_age n interval total_age = 4 := by
sorry

#eval youngest_child_age 5 3 50

end NUMINAMATH_CALUDE_youngest_child_age_proof_l1199_119924


namespace NUMINAMATH_CALUDE_ball_return_ways_formula_l1199_119973

/-- The number of ways a ball can return to the starting person after n passes among 7m people. -/
def ball_return_ways (m n : ℕ) : ℚ :=
  (1 / m : ℚ) * ((m - 1 : ℚ)^n + (m - 1 : ℚ) * (-1)^n)

/-- Theorem stating the formula for the number of ways a ball can return to the starting person. -/
theorem ball_return_ways_formula {m n : ℕ} (hm : m ≥ 3) (hn : n ≥ 2) :
  ∃ (c : ℕ → ℚ), c n = ball_return_ways m n :=
by sorry

end NUMINAMATH_CALUDE_ball_return_ways_formula_l1199_119973


namespace NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l1199_119954

theorem distance_to_point : ℝ → ℝ → ℝ
  | x, y => Real.sqrt (x^2 + y^2)

theorem distance_from_origin_to_point :
  distance_to_point 8 (-15) = 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l1199_119954


namespace NUMINAMATH_CALUDE_max_distance_C_D_l1199_119921

open Complex

/-- The set of solutions to z^4 - 16 = 0 -/
def C : Set ℂ := {z : ℂ | z^4 - 16 = 0}

/-- The set of solutions to z^4 - 16z^3 + 48z^2 - 64z + 64 = 0 -/
def D : Set ℂ := {z : ℂ | z^4 - 16*z^3 + 48*z^2 - 64*z + 64 = 0}

/-- The maximum distance between any point in C and any point in D is 2 -/
theorem max_distance_C_D : 
  ∃ (c : C) (d : D), ∀ (c' : C) (d' : D), abs (c - d) ≥ abs (c' - d') ∧ abs (c - d) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_C_D_l1199_119921


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1199_119985

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 1)
  let b : ℝ × ℝ := (x, 2)
  are_parallel a b → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1199_119985


namespace NUMINAMATH_CALUDE_apple_eating_time_l1199_119916

theorem apple_eating_time (apples_per_hour : ℕ) (total_apples : ℕ) (h1 : apples_per_hour = 5) (h2 : total_apples = 15) :
  total_apples / apples_per_hour = 3 := by
sorry

end NUMINAMATH_CALUDE_apple_eating_time_l1199_119916


namespace NUMINAMATH_CALUDE_max_distinct_ten_blocks_l1199_119910

/-- Represents a binary string of length 10^4 -/
def BinaryString := Fin 10000 → Bool

/-- A k-block is a contiguous substring of length k -/
def kBlock (s : BinaryString) (start : Fin 10000) (k : Nat) : Fin k → Bool :=
  fun i => s ⟨start + i, sorry⟩

/-- Two k-blocks are identical if all their corresponding elements are equal -/
def kBlocksEqual (b1 b2 : Fin k → Bool) : Prop :=
  ∀ i : Fin k, b1 i = b2 i

/-- Count the number of distinct 3-blocks in a binary string -/
def distinctThreeBlocks (s : BinaryString) : Nat :=
  sorry

/-- Count the number of distinct 10-blocks in a binary string -/
def distinctTenBlocks (s : BinaryString) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem max_distinct_ten_blocks :
  ∀ s : BinaryString,
    distinctThreeBlocks s ≤ 7 →
    distinctTenBlocks s ≤ 504 :=
  sorry

end NUMINAMATH_CALUDE_max_distinct_ten_blocks_l1199_119910


namespace NUMINAMATH_CALUDE_binomial_expectation_and_variance_l1199_119974

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  ξ : ℝ → ℝ  -- The random variable

/-- The expected value of a random variable -/
def expectation (X : ℝ → ℝ) : ℝ := sorry

/-- The variance of a random variable -/
def variance (X : ℝ → ℝ) : ℝ := sorry

theorem binomial_expectation_and_variance 
  (ξ : BinomialDistribution 5 (1/2)) 
  (η : ℝ → ℝ) 
  (h : η = λ x => 5 * ξ.ξ x) : 
  expectation η = 25/2 ∧ variance η = 125/4 := by sorry

end NUMINAMATH_CALUDE_binomial_expectation_and_variance_l1199_119974


namespace NUMINAMATH_CALUDE_a_finishes_in_eight_days_l1199_119919

/-- Given two workers A and B who can finish a job together in a certain number of days,
    this function calculates how long it takes for A to finish the job alone. -/
def time_for_a_alone (total_time_together : ℚ) (days_worked_together : ℚ) (days_a_alone : ℚ) : ℚ :=
  let work_rate_together := 1 / total_time_together
  let work_done_together := work_rate_together * days_worked_together
  let remaining_work := 1 - work_done_together
  let work_rate_a := remaining_work / days_a_alone
  1 / work_rate_a

/-- Theorem stating that under the given conditions, A can finish the job alone in 8 days. -/
theorem a_finishes_in_eight_days :
  time_for_a_alone 40 10 6 = 8 := by sorry

end NUMINAMATH_CALUDE_a_finishes_in_eight_days_l1199_119919


namespace NUMINAMATH_CALUDE_system_solution_l1199_119947

theorem system_solution (x y : ℝ) : 
  (x + y = 1 ∧ x - y = 3) → (x = 2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1199_119947


namespace NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l1199_119901

theorem greatest_common_multiple_under_120 : ∃ (n : ℕ), n = 90 ∧ 
  (∀ m : ℕ, m < 120 → m % 10 = 0 → m % 15 = 0 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l1199_119901


namespace NUMINAMATH_CALUDE_second_win_proof_l1199_119986

/-- Represents the financial transactions of a man and calculates the amount won in the second round --/
def calculate_second_win (initial_amount : ℚ) (first_win : ℚ) : ℚ :=
  let after_first_loss := initial_amount * (2/3)
  let after_first_win := after_first_loss + first_win
  let after_second_loss := after_first_win * (2/3)
  initial_amount - after_second_loss

/-- Proves that the calculated second win amount results in the initial amount --/
theorem second_win_proof (initial_amount : ℚ) (first_win : ℚ) :
  let second_win := calculate_second_win initial_amount first_win
  let final_amount := (((initial_amount * (2/3) + first_win) * (2/3)) + second_win)
  initial_amount = 48.00000000000001 ∧ first_win = 10 →
  final_amount = initial_amount ∧ second_win = 20 := by
  sorry

#eval calculate_second_win 48.00000000000001 10

end NUMINAMATH_CALUDE_second_win_proof_l1199_119986


namespace NUMINAMATH_CALUDE_younger_person_age_l1199_119938

/-- Given two people's ages, proves that the younger person is 12 years old --/
theorem younger_person_age
  (total_age : ℕ)
  (age_difference : ℕ)
  (h1 : total_age = 30)
  (h2 : age_difference = 6) :
  (total_age - age_difference) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_younger_person_age_l1199_119938


namespace NUMINAMATH_CALUDE_cube_monotone_l1199_119936

theorem cube_monotone (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_monotone_l1199_119936


namespace NUMINAMATH_CALUDE_exists_starting_station_l1199_119962

/-- Represents a gasoline station with its fuel amount -/
structure GasStation where
  fuel : ℝ

/-- Represents a circular highway with gasoline stations -/
structure CircularHighway where
  stations : List GasStation
  length : ℝ
  h_positive_length : length > 0

/-- The total fuel available in all stations -/
def total_fuel (highway : CircularHighway) : ℝ :=
  (highway.stations.map (fun s => s.fuel)).sum

/-- Checks if it's possible to complete a lap starting from a given station index -/
def can_complete_lap (highway : CircularHighway) (start_index : ℕ) : Prop :=
  ∃ (direction : Bool), 
    let station_sequence := if direction then 
        highway.stations.rotateLeft start_index
      else 
        (highway.stations.rotateLeft start_index).reverse
    station_sequence.foldl 
      (fun (acc : ℝ) (station : GasStation) => 
        acc + station.fuel - (highway.length / highway.stations.length))
      0 
    ≥ 0

/-- The main theorem to be proved -/
theorem exists_starting_station (highway : CircularHighway) 
  (h_fuel : total_fuel highway = 2 * highway.length) :
  ∃ (i : ℕ), i < highway.stations.length ∧ can_complete_lap highway i :=
sorry

end NUMINAMATH_CALUDE_exists_starting_station_l1199_119962


namespace NUMINAMATH_CALUDE_lorenzo_thumbtacks_l1199_119929

/-- The number of cans of thumbtacks Lorenzo had -/
def number_of_cans : ℕ := sorry

/-- The number of boards Lorenzo tested -/
def boards_tested : ℕ := 120

/-- The number of tacks remaining in each can at the end of the day -/
def tacks_remaining : ℕ := 30

/-- The total combined number of thumbtacks from the full cans -/
def total_thumbtacks : ℕ := 450

theorem lorenzo_thumbtacks :
  (number_of_cans * (boards_tested + tacks_remaining) = total_thumbtacks) →
  number_of_cans = 3 := by sorry

end NUMINAMATH_CALUDE_lorenzo_thumbtacks_l1199_119929


namespace NUMINAMATH_CALUDE_candy_distribution_l1199_119966

theorem candy_distribution (e : ℚ) 
  (frank_candies : ℚ) (gail_candies : ℚ) (hank_candies : ℚ) : 
  frank_candies = 4 * e →
  gail_candies = 4 * frank_candies →
  hank_candies = 6 * gail_candies →
  e + frank_candies + gail_candies + hank_candies = 876 →
  e = 7.5 := by
sorry


end NUMINAMATH_CALUDE_candy_distribution_l1199_119966


namespace NUMINAMATH_CALUDE_remaining_trees_l1199_119972

/-- Given a park with an initial number of trees, some of which die and others are cut,
    this theorem proves the number of remaining trees. -/
theorem remaining_trees (initial : ℕ) (dead : ℕ) (cut : ℕ) : 
  initial = 86 → dead = 15 → cut = 23 → initial - (dead + cut) = 48 := by
  sorry

end NUMINAMATH_CALUDE_remaining_trees_l1199_119972


namespace NUMINAMATH_CALUDE_six_star_three_equals_three_l1199_119926

-- Define the * operation
def star (a b : ℤ) : ℤ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem six_star_three_equals_three : star 6 3 = 3 := by sorry

end NUMINAMATH_CALUDE_six_star_three_equals_three_l1199_119926


namespace NUMINAMATH_CALUDE_keith_pears_l1199_119952

theorem keith_pears (mike_pears : ℕ) (keith_gave_away : ℕ) (total_left : ℕ) 
  (h1 : mike_pears = 12)
  (h2 : keith_gave_away = 46)
  (h3 : total_left = 13) :
  ∃ keith_initial : ℕ, 
    keith_initial = 47 ∧ 
    keith_initial - keith_gave_away + mike_pears = total_left :=
by sorry

end NUMINAMATH_CALUDE_keith_pears_l1199_119952


namespace NUMINAMATH_CALUDE_friendship_subset_exists_l1199_119965

/-- Represents a friendship relation between students -/
def FriendshipRelation (S : Type) := S → S → Prop

/-- A school is valid if it satisfies the friendship condition -/
def ValidSchool (S : Type) (friendship : FriendshipRelation S) (students : Finset S) : Prop :=
  ∀ s ∈ students, ∃ t ∈ students, s ≠ t ∧ friendship s t

theorem friendship_subset_exists 
  (S : Type) 
  (friendship : FriendshipRelation S) 
  (students : Finset S) 
  (h_valid : ValidSchool S friendship students)
  (h_count : students.card = 101) :
  ∀ n : ℕ, 1 < n → n < 101 → 
    ∃ subset : Finset S, subset.card = n ∧ subset ⊆ students ∧
      ∀ s ∈ subset, ∃ t ∈ subset, s ≠ t ∧ friendship s t :=
by
  sorry


end NUMINAMATH_CALUDE_friendship_subset_exists_l1199_119965


namespace NUMINAMATH_CALUDE_shorter_leg_length_is_five_l1199_119999

/-- A right triangle that can be cut and reassembled into a square -/
structure CuttableRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  is_right_triangle : shorter_leg^2 + longer_leg^2 = hypotenuse^2
  can_form_square : hypotenuse = 2 * shorter_leg

/-- The theorem stating that if a right triangle with longer leg 10 can be cut and
    reassembled into a square, then its shorter leg has length 5 -/
theorem shorter_leg_length_is_five
  (triangle : CuttableRightTriangle)
  (h : triangle.longer_leg = 10) :
  triangle.shorter_leg = 5 := by
  sorry


end NUMINAMATH_CALUDE_shorter_leg_length_is_five_l1199_119999


namespace NUMINAMATH_CALUDE_green_hats_count_l1199_119925

theorem green_hats_count (total_hats : ℕ) (blue_price green_price total_price : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_price = 6)
  (h3 : green_price = 7)
  (h4 : total_price = 530) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_price * blue_hats + green_price * green_hats = total_price ∧
    green_hats = 20 :=
by sorry

end NUMINAMATH_CALUDE_green_hats_count_l1199_119925


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l1199_119981

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 - 0.30) * (1 + 0.5873) = P → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l1199_119981


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1199_119959

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + 2*m - 4 = 0 ∧ x₂^2 - x₂ + 2*m - 4 = 0) →
  (m ≤ 17/8 ∧
   (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + 2*m - 4 = 0 ∧ x₂^2 - x₂ + 2*m - 4 = 0 →
    (x₁ - 3) * (x₂ - 3) = m^2 - 1 → m = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1199_119959


namespace NUMINAMATH_CALUDE_range_of_a_l1199_119994

/-- Given propositions p and q, where p is a necessary but not sufficient condition for q,
    prove that the range of real number a is [-1/2, 1]. -/
theorem range_of_a (x a : ℝ) : 
  (∀ x, x^2 - ax - 2*a^2 < 0 → x^2 - 2*x - 3 < 0) ∧ 
  (∃ x, x^2 - 2*x - 3 < 0 ∧ x^2 - ax - 2*a^2 ≥ 0) →
  -1/2 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1199_119994


namespace NUMINAMATH_CALUDE_unique_number_property_l1199_119992

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Reverse of a natural number -/
def reverseNum (n : ℕ) : ℕ := sorry

/-- Prime factors of a natural number -/
def primeFactors (n : ℕ) : List ℕ := sorry

/-- Remove zeros from a natural number -/
def removeZeros (n : ℕ) : ℕ := sorry

theorem unique_number_property : ∃! n : ℕ, 
  n > 0 ∧ 
  n = sumOfDigits n * reverseNum (sumOfDigits n) ∧ 
  n = removeZeros ((List.sum (List.map (λ x => x^2) (primeFactors n))) / 2) ∧
  n = 1729 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_property_l1199_119992


namespace NUMINAMATH_CALUDE_floor_abs_neg_57_6_l1199_119912

theorem floor_abs_neg_57_6 : ⌊|(-57.6 : ℝ)|⌋ = 57 := by sorry

end NUMINAMATH_CALUDE_floor_abs_neg_57_6_l1199_119912


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1199_119900

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h1 : seq.a 2 + seq.a 6 = 2)
    (h2 : seq.S 9 = -18) :
    (∀ n, seq.a n = 13 - 3*n) ∧
    (∀ n, |seq.S n| ≥ |seq.S 8|) ∧
    (|seq.S 8| = 4) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1199_119900


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1199_119927

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1199_119927


namespace NUMINAMATH_CALUDE_unique_prime_pair_divisibility_l1199_119978

theorem unique_prime_pair_divisibility : 
  ∃! (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (3 * p^(q-1) + 1) ∣ (11^p + 17^p) ∧
    p = 3 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_divisibility_l1199_119978


namespace NUMINAMATH_CALUDE_range_of_a_l1199_119975

def point_P (a : ℝ) : ℝ × ℝ := (3*a - 9, a + 2)

def on_terminal_side (p : ℝ × ℝ) (α : ℝ) : Prop :=
  (p.1 ≥ 0 ∧ p.2 ≥ 0) ∨ (p.1 ≤ 0 ∧ p.2 ≥ 0) ∨ (p.1 ≤ 0 ∧ p.2 ≤ 0) ∨ (p.1 ≥ 0 ∧ p.2 ≤ 0)

theorem range_of_a (α : ℝ) :
  (∀ a : ℝ, on_terminal_side (point_P a) α ∧ Real.cos α ≤ 0 ∧ Real.sin α > 0) →
  (∀ a : ℝ, a ∈ Set.Ioc (-2) 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1199_119975


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1199_119997

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to calculate the units digit of a^n
def powerUnitsDigit (a n : ℕ) : ℕ :=
  unitsDigit ((unitsDigit a)^n)

theorem units_digit_sum_of_powers : 
  unitsDigit ((35 : ℕ)^87 + (93 : ℕ)^53) = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1199_119997


namespace NUMINAMATH_CALUDE_trio_selection_l1199_119945

theorem trio_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 3) :
  Nat.choose n k = 220 := by
  sorry

end NUMINAMATH_CALUDE_trio_selection_l1199_119945


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l1199_119989

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ := sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- Finds the point E on CD that is one-fourth the way from C to D -/
def findPointE (C D : Point) : Point := sorry

/-- Finds the intersection point F of BE and AC -/
def findIntersectionF (A B C E : Point) : Point := sorry

theorem rectangle_area_theorem (ABCD : Rectangle) :
  let E := findPointE ABCD.C ABCD.D
  let F := findIntersectionF ABCD.A ABCD.B ABCD.C E
  quadrilateralArea ABCD.A F E ABCD.D = 36 →
  rectangleArea ABCD = 144 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l1199_119989


namespace NUMINAMATH_CALUDE_num_small_squares_seven_l1199_119931

/-- The number of small squares formed when a square is divided into n equal parts on each side and the points are joined -/
def num_small_squares (n : ℕ) : ℕ := 4 * (n * (n - 1) / 2)

/-- Theorem stating that the number of small squares is 84 when n = 7 -/
theorem num_small_squares_seven : num_small_squares 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_num_small_squares_seven_l1199_119931


namespace NUMINAMATH_CALUDE_polynomial_square_l1199_119928

theorem polynomial_square (a b : ℚ) : 
  (∃ q₀ q₁ : ℚ, ∀ x, x^4 + 3*x^3 + x^2 + a*x + b = (x^2 + q₁*x + q₀)^2) → 
  b = 25/64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_square_l1199_119928


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1199_119903

theorem decimal_to_fraction :
  (0.32 : ℚ) = 8 / 25 := by
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1199_119903


namespace NUMINAMATH_CALUDE_cello_practice_time_l1199_119930

/-- Given a total practice time of 7.5 hours in a week, with 86 minutes of practice on each of 2 days,
    the remaining practice time on the other days is 278 minutes. -/
theorem cello_practice_time (total_hours : ℝ) (practice_minutes_per_day : ℕ) (practice_days : ℕ) :
  total_hours = 7.5 ∧ practice_minutes_per_day = 86 ∧ practice_days = 2 →
  (total_hours * 60 : ℝ) - (practice_minutes_per_day * practice_days : ℕ) = 278 := by
  sorry

end NUMINAMATH_CALUDE_cello_practice_time_l1199_119930


namespace NUMINAMATH_CALUDE_infinite_square_divisibility_l1199_119956

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | (n + 4) => 2 * a (n + 3) + a (n + 2) - 2 * a (n + 1) - a n

theorem infinite_square_divisibility :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (n : ℤ)^2 ∣ a n := by sorry

end NUMINAMATH_CALUDE_infinite_square_divisibility_l1199_119956


namespace NUMINAMATH_CALUDE_meaningful_sqrt_over_x_l1199_119957

theorem meaningful_sqrt_over_x (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / x) ↔ x ≥ -3 ∧ x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_over_x_l1199_119957


namespace NUMINAMATH_CALUDE_factorial_345_trailing_zeros_l1199_119982

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_345_trailing_zeros :
  trailingZeros 345 = 84 :=
sorry

end NUMINAMATH_CALUDE_factorial_345_trailing_zeros_l1199_119982


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l1199_119918

theorem polynomial_multiplication (y : ℝ) :
  (3*y - 2 + 4) * (2*y^12 + 3*y^11 - y^9 - y^8) =
  6*y^13 + 13*y^12 + 6*y^11 - 3*y^10 - 5*y^9 - 2*y^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l1199_119918


namespace NUMINAMATH_CALUDE_diego_orange_weight_l1199_119983

/-- Given Diego's fruit purchases, prove the weight of oranges he bought. -/
theorem diego_orange_weight (total_capacity : ℕ) (watermelon_weight : ℕ) (grape_weight : ℕ) (apple_weight : ℕ) 
  (h1 : total_capacity = 20)
  (h2 : watermelon_weight = 1)
  (h3 : grape_weight = 1)
  (h4 : apple_weight = 17) :
  total_capacity - (watermelon_weight + grape_weight + apple_weight) = 1 := by
  sorry

end NUMINAMATH_CALUDE_diego_orange_weight_l1199_119983


namespace NUMINAMATH_CALUDE_tuesday_flower_sales_ratio_l1199_119984

/-- Represents the number of flowers sold -/
structure FlowerSales where
  roses : ℕ
  lilacs : ℕ
  gardenias : ℕ

/-- Represents the ratio of two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of roses to lilacs -/
def roseToLilacRatio (sales : FlowerSales) : Ratio :=
  { numerator := sales.roses, denominator := sales.lilacs }

theorem tuesday_flower_sales_ratio : 
  ∀ (sales : FlowerSales), 
    sales.lilacs = 10 →
    sales.gardenias = sales.lilacs / 2 →
    sales.roses + sales.lilacs + sales.gardenias = 45 →
    (roseToLilacRatio sales).numerator = 3 ∧ (roseToLilacRatio sales).denominator = 1 := by
  sorry


end NUMINAMATH_CALUDE_tuesday_flower_sales_ratio_l1199_119984


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l1199_119937

/-- A geometric sequence with first term 2 and fifth term 4 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) / a n = a 2 / a 1) ∧ a 1 = 2 ∧ a 5 = 4

theorem geometric_sequence_ninth_term (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l1199_119937


namespace NUMINAMATH_CALUDE_triangle_proof_l1199_119953

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = t.a * Real.cos t.B ∧ Real.sin t.C = 1/3

-- State the theorem
theorem triangle_proof (t : Triangle) (h : triangle_conditions t) :
  t.A = Real.pi/2 ∧ Real.cos (Real.pi + t.B) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l1199_119953


namespace NUMINAMATH_CALUDE_principal_calculation_l1199_119911

theorem principal_calculation (P R : ℝ) 
  (h1 : P + (P * R * 2) / 100 = 660)
  (h2 : P + (P * R * 7) / 100 = 1020) : 
  P = 516 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1199_119911


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l1199_119922

theorem tangent_sum_simplification :
  (Real.tan (10 * π / 180) + Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (40 * π / 180)) / Real.cos (10 * π / 180) =
  (1/2 + Real.cos (20 * π / 180)^2) / (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * Real.cos (40 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l1199_119922


namespace NUMINAMATH_CALUDE_tan_equality_implies_specific_angles_l1199_119963

theorem tan_equality_implies_specific_angles (m : ℤ) :
  -180 < m ∧ m < 180 →
  Real.tan (↑m * π / 180) = Real.tan (405 * π / 180) →
  m = 45 ∨ m = -135 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_implies_specific_angles_l1199_119963


namespace NUMINAMATH_CALUDE_A_B_mutually_exclusive_A_C_mutually_exclusive_C_D_complementary_l1199_119913

-- Define the sample space for a six-sided die
def Ω : Type := Fin 6

-- Define the probability measure
variable (P : Ω → ℝ)

-- Assume the die is fair
axiom fair_die : ∀ x : Ω, P x = 1 / 6

-- Define events
def A (x : Ω) : Prop := x.val + 1 = 4
def B (x : Ω) : Prop := x.val % 2 = 0
def C (x : Ω) : Prop := x.val + 1 < 4
def D (x : Ω) : Prop := x.val + 1 > 3

-- Theorem statements
theorem A_B_mutually_exclusive : ∀ x : Ω, ¬(A x ∧ B x) := by sorry

theorem A_C_mutually_exclusive : ∀ x : Ω, ¬(A x ∧ C x) := by sorry

theorem C_D_complementary : ∀ x : Ω, C x ↔ ¬(D x) := by sorry

end NUMINAMATH_CALUDE_A_B_mutually_exclusive_A_C_mutually_exclusive_C_D_complementary_l1199_119913


namespace NUMINAMATH_CALUDE_grid_triangle_square_l1199_119980

/-- A point on a 2D grid represented by integer coordinates -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- The area of a triangle formed by three grid points -/
def triangleArea (A B C : GridPoint) : ℚ := sorry

/-- The squared distance between two grid points -/
def squaredDistance (A B : GridPoint) : ℤ := sorry

/-- Predicate to check if three grid points form three vertices of a square -/
def formSquareVertices (A B C : GridPoint) : Prop := sorry

theorem grid_triangle_square (A B C : GridPoint) :
  let T := triangleArea A B C
  (squaredDistance A B + squaredDistance B C)^2 < 8 * T + 1 →
  formSquareVertices A B C := by
  sorry

end NUMINAMATH_CALUDE_grid_triangle_square_l1199_119980
