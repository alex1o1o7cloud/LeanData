import Mathlib

namespace houses_visited_per_day_l3923_392331

-- Define the parameters
def buyerPercentage : Real := 0.2
def cheapKnivesPrice : Real := 50
def expensiveKnivesPrice : Real := 150
def weeklyRevenue : Real := 5000
def workDaysPerWeek : Nat := 5

-- Define the theorem
theorem houses_visited_per_day :
  ∃ (housesPerDay : Nat),
    (housesPerDay : Real) * buyerPercentage * 
    ((cheapKnivesPrice + expensiveKnivesPrice) / 2) * 
    (workDaysPerWeek : Real) = weeklyRevenue ∧
    housesPerDay = 50 := by
  sorry

end houses_visited_per_day_l3923_392331


namespace original_price_calculation_l3923_392369

/-- Proves that if an article is sold for $120 with a 20% gain, its original price was $100. -/
theorem original_price_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 120 ∧ gain_percent = 20 → 
  selling_price = (100 : ℝ) * (1 + gain_percent / 100) :=
by
  sorry

end original_price_calculation_l3923_392369


namespace problem_solution_l3923_392376

theorem problem_solution (x y z : ℝ) : 
  3 * x = 0.75 * y → 
  x + z = 24 → 
  z = 8 → 
  y = 64 := by
sorry

end problem_solution_l3923_392376


namespace integral_nonnegative_function_integral_positive_at_point_l3923_392330

open MeasureTheory
open Measure
open Set
open Interval

theorem integral_nonnegative_function
  {a b : ℝ} (hab : a ≤ b)
  {f : ℝ → ℝ} (hf : ContinuousOn f (Icc a b))
  (hfnonneg : ∀ x ∈ Icc a b, f x ≥ 0) :
  ∫ x in a..b, f x ≥ 0 :=
sorry

theorem integral_positive_at_point
  {a b : ℝ} (hab : a ≤ b)
  {f : ℝ → ℝ} (hf : ContinuousOn f (Icc a b))
  (hfnonneg : ∀ x ∈ Icc a b, f x ≥ 0)
  (x₀ : ℝ) (hx₀ : x₀ ∈ Icc a b) (hfx₀ : f x₀ > 0) :
  ∫ x in a..b, f x > 0 :=
sorry

end integral_nonnegative_function_integral_positive_at_point_l3923_392330


namespace calculate_savings_l3923_392397

/-- Given total expenses and savings rate, calculate the amount saved -/
theorem calculate_savings (total_expenses : ℝ) (savings_rate : ℝ) : 
  total_expenses = 24150 ∧ savings_rate = 0.1 → 
  ∃ amount_saved : ℝ, abs (amount_saved - 2683.33) < 0.01 := by
  sorry

end calculate_savings_l3923_392397


namespace not_always_sufficient_condition_l3923_392344

theorem not_always_sufficient_condition : 
  ¬(∀ (a b c : ℝ), a > b → a * c^2 > b * c^2) :=
by sorry

end not_always_sufficient_condition_l3923_392344


namespace delta_max_success_ratio_l3923_392322

/-- Represents a contestant's score for a single day -/
structure DailyScore where
  scored : ℚ
  attempted : ℚ

/-- Represents a contestant's scores for the three-day contest -/
structure ContestScore where
  day1 : DailyScore
  day2 : DailyScore
  day3 : DailyScore

def Charlie : ContestScore :=
  { day1 := { scored := 200, attempted := 300 },
    day2 := { scored := 160, attempted := 200 },
    day3 := { scored := 90, attempted := 100 } }

def totalAttempted (score : ContestScore) : ℚ :=
  score.day1.attempted + score.day2.attempted + score.day3.attempted

def totalScored (score : ContestScore) : ℚ :=
  score.day1.scored + score.day2.scored + score.day3.scored

def successRatio (score : ContestScore) : ℚ :=
  totalScored score / totalAttempted score

def dailySuccessRatio (day : DailyScore) : ℚ :=
  day.scored / day.attempted

theorem delta_max_success_ratio :
  ∀ delta : ContestScore,
    totalAttempted delta = 600 →
    dailySuccessRatio delta.day1 < dailySuccessRatio Charlie.day1 →
    dailySuccessRatio delta.day2 < dailySuccessRatio Charlie.day2 →
    dailySuccessRatio delta.day3 < dailySuccessRatio Charlie.day3 →
    delta.day1.attempted ≠ Charlie.day1.attempted →
    delta.day2.attempted ≠ Charlie.day2.attempted →
    delta.day3.attempted ≠ Charlie.day3.attempted →
    successRatio delta ≤ 399 / 600 :=
by sorry

#check delta_max_success_ratio

end delta_max_success_ratio_l3923_392322


namespace triangle_side_length_l3923_392320

/-- In a triangle ABC, given side lengths a and c, and angle A, prove that side length b has a specific value. -/
theorem triangle_side_length (a c b : ℝ) (A : ℝ) : 
  a = 3 → c = Real.sqrt 3 → A = π / 3 → 
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A → 
  b = 2 * Real.sqrt 3 := by sorry

end triangle_side_length_l3923_392320


namespace oil_price_reduction_60_percent_l3923_392340

/-- The percentage reduction in oil price -/
def oil_price_reduction (original_price reduced_price : ℚ) : ℚ :=
  (original_price - reduced_price) / original_price * 100

/-- The amount of oil that can be bought with a fixed amount of money -/
def oil_amount (price : ℚ) (money : ℚ) : ℚ := money / price

theorem oil_price_reduction_60_percent 
  (reduced_price : ℚ) 
  (additional_amount : ℚ) 
  (fixed_money : ℚ) :
  reduced_price = 30 →
  additional_amount = 10 →
  fixed_money = 1500 →
  oil_amount reduced_price fixed_money = oil_amount reduced_price (fixed_money / 2) + additional_amount →
  oil_price_reduction ((fixed_money / 2) / additional_amount) reduced_price = 60 := by
sorry

end oil_price_reduction_60_percent_l3923_392340


namespace dihedral_angle_inscribed_spheres_l3923_392398

/-- Given two spheres inscribed in a dihedral angle, this theorem proves
    the relationship between the spheres' radii, their position, and the
    measure of the dihedral angle. -/
theorem dihedral_angle_inscribed_spheres 
  (R₁ R₂ : ℝ) -- Radii of the two spheres
  (h_touch : R₁ + R₂ > 0) -- The spheres touch (implied by positive sum of radii)
  (h_ratio : R₁ = 1.5 * R₂) -- Ratio of radii
  (h_angle : Real.cos (45 * π / 180) = Real.sqrt (1 / 2)) -- 45° angle with edge
  : Real.cos (θ / 2) = Real.sqrt ((1 + Real.sqrt (1 / 2)) / 2) :=
by sorry

end dihedral_angle_inscribed_spheres_l3923_392398


namespace oranges_sold_count_l3923_392347

/-- Given information about oranges on a truck -/
structure OrangeTruck where
  bags : Nat
  oranges_per_bag : Nat
  rotten : Nat
  for_juice : Nat

/-- Calculate the number of oranges to be sold -/
def oranges_to_sell (truck : OrangeTruck) : Nat :=
  truck.bags * truck.oranges_per_bag - (truck.rotten + truck.for_juice)

/-- Theorem stating the number of oranges to be sold -/
theorem oranges_sold_count (truck : OrangeTruck) 
  (h1 : truck.bags = 10)
  (h2 : truck.oranges_per_bag = 30)
  (h3 : truck.rotten = 50)
  (h4 : truck.for_juice = 30) :
  oranges_to_sell truck = 220 := by
  sorry

#eval oranges_to_sell { bags := 10, oranges_per_bag := 30, rotten := 50, for_juice := 30 }

end oranges_sold_count_l3923_392347


namespace parabola_equation_l3923_392329

/-- The equation of a parabola with vertex at the origin and focus at (2, 0) -/
theorem parabola_equation : ∀ x y : ℝ, 
  (∃ p : ℝ, p > 0 ∧ x = p ∧ y = 0) →  -- focus at (p, 0)
  (∀ a b : ℝ, (a - x)^2 + (b - y)^2 = (a - 0)^2 + b^2) →  -- definition of parabola
  y^2 = 4 * 2 * x :=
sorry

end parabola_equation_l3923_392329


namespace find_certain_number_l3923_392356

theorem find_certain_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + x + 45) / 3) + 5 → x = 50 := by
  sorry

end find_certain_number_l3923_392356


namespace table_loss_percentage_l3923_392335

theorem table_loss_percentage 
  (cost_price selling_price : ℝ) 
  (h1 : 15 * cost_price = 20 * selling_price) 
  (discount_rate : ℝ) (h2 : discount_rate = 0.1)
  (tax_rate : ℝ) (h3 : tax_rate = 0.08) : 
  (cost_price * (1 - discount_rate) - selling_price * (1 + tax_rate)) / cost_price = 0.09 := by
sorry

end table_loss_percentage_l3923_392335


namespace workshop_production_balance_l3923_392384

theorem workshop_production_balance :
  let total_workers : ℕ := 85
  let type_a_rate : ℕ := 16
  let type_b_rate : ℕ := 10
  let set_a_parts : ℕ := 2
  let set_b_parts : ℕ := 3
  let workers_a : ℕ := 25
  let workers_b : ℕ := 60
  (total_workers = workers_a + workers_b) ∧
  ((type_a_rate * workers_a) / set_a_parts = (type_b_rate * workers_b) / set_b_parts) := by
  sorry

end workshop_production_balance_l3923_392384


namespace cindys_calculation_l3923_392321

theorem cindys_calculation (x : ℝ) : (x - 10) / 5 = 50 → (x - 5) / 10 = 25.5 := by
  sorry

end cindys_calculation_l3923_392321


namespace exists_student_won_all_l3923_392363

/-- Represents a competition --/
def Competition := Fin 44

/-- Represents a student --/
structure Student where
  id : ℕ

/-- The set of students who won a given competition --/
def winners : Competition → Finset Student :=
  sorry

/-- The number of competitions a student has won --/
def wins (s : Student) : ℕ :=
  sorry

/-- Statement: There exists a student who won all competitions --/
theorem exists_student_won_all :
  (∀ c : Competition, (winners c).card = 7) →
  (∀ c₁ c₂ : Competition, c₁ ≠ c₂ → ∃! s : Student, s ∈ winners c₁ ∧ s ∈ winners c₂) →
  ∃ s : Student, ∀ c : Competition, s ∈ winners c :=
sorry

end exists_student_won_all_l3923_392363


namespace date_sum_equality_l3923_392392

/-- Represents a calendar date sequence -/
structure DateSequence where
  x : ℕ  -- Date behind C
  dateA : ℕ := x + 2  -- Date behind A
  dateB : ℕ := x + 11  -- Date behind B
  dateP : ℕ := x + 13  -- Date behind P

/-- Theorem: The sum of dates behind C and P equals the sum of dates behind A and B -/
theorem date_sum_equality (d : DateSequence) : 
  d.x + d.dateP = d.dateA + d.dateB := by
  sorry

end date_sum_equality_l3923_392392


namespace simplify_and_evaluate_l3923_392389

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -5) (hy : y = 2) :
  ((x + 2*y)^2 - (x - 2*y)*(2*y + x)) / (4*y) = -1 := by
  sorry

end simplify_and_evaluate_l3923_392389


namespace direction_vector_b_l3923_392324

def point_1 : ℝ × ℝ := (-3, 4)
def point_2 : ℝ × ℝ := (2, -1)

theorem direction_vector_b (b : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ (point_2.1 - point_1.1, point_2.2 - point_1.2) = (k * b, k * (-1))) → 
  b = 1 := by
sorry

end direction_vector_b_l3923_392324


namespace decagon_adjacent_vertex_probability_l3923_392381

/-- A decagon is a polygon with 10 vertices -/
def Decagon : ℕ := 10

/-- The number of vertices adjacent to any given vertex in a decagon -/
def AdjacentVertices : ℕ := 2

/-- The probability of selecting two adjacent vertices when choosing two distinct vertices at random from a decagon -/
theorem decagon_adjacent_vertex_probability : 
  (AdjacentVertices : ℚ) / (Decagon - 1 : ℚ) = 2 / 9 := by sorry

end decagon_adjacent_vertex_probability_l3923_392381


namespace new_person_age_l3923_392301

theorem new_person_age (initial_group_size : ℕ) (age_decrease : ℕ) (replaced_person_age : ℕ) :
  initial_group_size = 10 →
  age_decrease = 3 →
  replaced_person_age = 42 →
  ∃ (new_person_age : ℕ),
    new_person_age = initial_group_size * age_decrease + replaced_person_age - initial_group_size * age_decrease :=
by
  sorry

end new_person_age_l3923_392301


namespace trim_length_calculation_oliver_trim_purchase_l3923_392339

theorem trim_length_calculation (table_area : Real) (pi_approx : Real) (extra_trim : Real) : Real :=
  let radius := Real.sqrt (table_area / pi_approx)
  let circumference := 2 * pi_approx * radius
  circumference + extra_trim

theorem oliver_trim_purchase :
  trim_length_calculation 616 (22/7) 5 = 93 :=
by sorry

end trim_length_calculation_oliver_trim_purchase_l3923_392339


namespace files_remaining_l3923_392311

theorem files_remaining (music_files : ℕ) (video_files : ℕ) (deleted_files : ℕ)
  (h1 : music_files = 27)
  (h2 : video_files = 42)
  (h3 : deleted_files = 11) :
  music_files + video_files - deleted_files = 58 :=
by sorry

end files_remaining_l3923_392311


namespace intersection_point_product_l3923_392348

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := y^2 / 16 + x^2 / 9 = 1

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 / 5 = 1

-- Define the common foci
def common_foci (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ (a b c d : ℝ), 
    a^2 / 16 + b^2 / 9 = 1 ∧ 
    c^2 / 4 - d^2 / 5 = 1 ∧
    F1 = (b, a) ∧ F2 = (-b, -a)

-- Define the point of intersection
def is_intersection_point (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P.1 P.2 ∧ is_on_hyperbola P.1 P.2

-- The theorem
theorem intersection_point_product (F1 F2 P : ℝ × ℝ) :
  common_foci F1 F2 → is_intersection_point P →
  (P.1 - F1.1)^2 + (P.2 - F1.2)^2 * ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 144 :=
by sorry

end intersection_point_product_l3923_392348


namespace danai_decorations_l3923_392377

/-- The total number of decorations Danai will put up -/
def total_decorations (skulls broomsticks spiderwebs cauldrons additional_budget left_to_put_up : ℕ) : ℕ :=
  skulls + broomsticks + spiderwebs + (spiderwebs * 2) + cauldrons + additional_budget + left_to_put_up

/-- Theorem stating the total number of decorations Danai will put up -/
theorem danai_decorations : 
  total_decorations 12 4 12 1 20 10 = 83 := by
  sorry

end danai_decorations_l3923_392377


namespace plot_length_is_100_l3923_392382

/-- Proves that the length of a rectangular plot is 100 meters given specific conditions. -/
theorem plot_length_is_100 (width : ℝ) (path_width : ℝ) (gravel_cost_per_sqm : ℝ) (total_gravel_cost : ℝ) :
  width = 65 →
  path_width = 2.5 →
  gravel_cost_per_sqm = 0.4 →
  total_gravel_cost = 340 →
  ∃ (length : ℝ),
    ((length + 2 * path_width) * (width + 2 * path_width) - length * width) * gravel_cost_per_sqm = total_gravel_cost ∧
    length = 100 := by
  sorry

end plot_length_is_100_l3923_392382


namespace map_distance_calculation_map_distance_proof_l3923_392312

theorem map_distance_calculation (scale_map : Real) (scale_actual : Real) (actual_distance : Real) : Real :=
  let scale_factor := scale_actual / scale_map
  let map_distance := actual_distance / scale_factor
  map_distance

theorem map_distance_proof (h1 : map_distance_calculation 0.4 5.3 848 = 64) : 
  ∃ (d : Real), map_distance_calculation 0.4 5.3 848 = d ∧ d = 64 := by
  sorry

end map_distance_calculation_map_distance_proof_l3923_392312


namespace problem_statement_l3923_392385

theorem problem_statement (a b x y : ℕ+) (P : ℕ) 
  (h1 : ∃ k : ℕ, a * x + b * y = k * (a^2 + b^2))
  (h2 : P = x^2 + y^2)
  (h3 : Nat.Prime P) :
  (P ∣ (a^2 + b^2)) ∧ (a = x ∧ b = y) := by
sorry

end problem_statement_l3923_392385


namespace girls_in_group_l3923_392327

theorem girls_in_group (n : ℕ) : 
  (4 : ℝ) + n > 0 → -- ensure the total number of students is positive
  (((n + 4) * (n + 3) / 2 - 6) / ((n + 4) * (n + 3) / 2) = 5 / 6) →
  n = 5 := by
  sorry


end girls_in_group_l3923_392327


namespace sum_and_simplest_form_l3923_392383

theorem sum_and_simplest_form :
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (2 : ℚ) / 3 + (7 : ℚ) / 8 = (n : ℚ) / d ∧ 
  ∀ (n' d' : ℕ), n' > 0 → d' > 0 → (n' : ℚ) / d' = (n : ℚ) / d → n' ≥ n ∧ d' ≥ d :=
by
  sorry

end sum_and_simplest_form_l3923_392383


namespace hidden_digit_problem_l3923_392366

theorem hidden_digit_problem :
  ∃! (x : ℕ), x ≠ 0 ∧ x < 10 ∧ ((10 * x + x) + (10 * x + x) + 1) * x = 100 * x + 10 * x + x :=
by
  sorry

end hidden_digit_problem_l3923_392366


namespace max_value_3m_plus_4n_l3923_392315

theorem max_value_3m_plus_4n (m n : ℕ) (even_nums : Finset ℕ) (odd_nums : Finset ℕ) : 
  m = 15 →
  even_nums.card = m →
  odd_nums.card = n →
  (∀ x ∈ even_nums, x % 2 = 0 ∧ x > 0) →
  (∀ x ∈ odd_nums, x % 2 = 1 ∧ x > 0) →
  (even_nums.sum id + odd_nums.sum id = 1987) →
  (3 * m + 4 * n ≤ 221) :=
by sorry

end max_value_3m_plus_4n_l3923_392315


namespace sarah_max_correct_l3923_392342

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ

/-- Represents a student's exam results. -/
structure ExamResult where
  exam : Exam
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  total_score : ℤ

/-- Checks if the exam result is valid according to the exam rules. -/
def is_valid_result (result : ExamResult) : Prop :=
  result.correct + result.incorrect + result.unanswered = result.exam.total_questions ∧
  result.correct * result.exam.correct_score + result.incorrect * result.exam.incorrect_score = result.total_score

/-- The specific exam Sarah took. -/
def sarah_exam : Exam :=
  { total_questions := 25
  , correct_score := 4
  , incorrect_score := -3 }

/-- Sarah's exam result. -/
def sarah_result (correct : ℕ) : ExamResult :=
  { exam := sarah_exam
  , correct := correct
  , incorrect := (4 * correct - 40) / 3
  , unanswered := 25 - correct - (4 * correct - 40) / 3
  , total_score := 40 }

theorem sarah_max_correct :
  ∀ c : ℕ, c > 13 → ¬(is_valid_result (sarah_result c)) ∧
  is_valid_result (sarah_result 13) :=
sorry

end sarah_max_correct_l3923_392342


namespace equation_equivalent_to_line_segments_l3923_392365

def satisfies_equation (x y : ℝ) : Prop :=
  3 * |x - 1| + 2 * |y + 2| = 6

def within_rectangle (x y : ℝ) : Prop :=
  -1 ≤ x ∧ x ≤ 3 ∧ -5 ≤ y ∧ y ≤ 1

def on_line_segments (x y : ℝ) : Prop :=
  (3*x + 2*y = 5 ∨ -3*x + 2*y = -1 ∨ 3*x - 2*y = 13 ∨ -3*x - 2*y = 7) ∧ within_rectangle x y

theorem equation_equivalent_to_line_segments :
  ∀ x y : ℝ, satisfies_equation x y ↔ on_line_segments x y :=
sorry

end equation_equivalent_to_line_segments_l3923_392365


namespace systematic_sampling_40th_number_l3923_392360

/-- Given a systematic sample of 50 students from 1000, with the first number drawn being 0015,
    prove that the 40th number drawn is 0795. -/
theorem systematic_sampling_40th_number
  (total_students : Nat)
  (sample_size : Nat)
  (first_number : Nat)
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_number = 15)
  : (first_number + (39 * (total_students / sample_size))) % total_students = 795 := by
  sorry

#eval (15 + (39 * (1000 / 50))) % 1000  -- Should output 795

end systematic_sampling_40th_number_l3923_392360


namespace third_month_sale_l3923_392371

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def sale_month1 : ℕ := 6535
def sale_month2 : ℕ := 6927
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 4891

theorem third_month_sale :
  ∃ (sale_month3 : ℕ),
    sale_month3 = average_sale * num_months - (sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month3 = 6855 := by
  sorry

end third_month_sale_l3923_392371


namespace ladder_distance_l3923_392361

theorem ladder_distance (c a b : ℝ) : 
  c = 25 → a = 20 → c^2 = a^2 + b^2 → b = 15 :=
by sorry

end ladder_distance_l3923_392361


namespace probability_yellow_or_green_l3923_392314

def yellow_marbles : ℕ := 4
def green_marbles : ℕ := 3
def red_marbles : ℕ := 2
def blue_marbles : ℕ := 1

def total_marbles : ℕ := yellow_marbles + green_marbles + red_marbles + blue_marbles
def favorable_marbles : ℕ := yellow_marbles + green_marbles

theorem probability_yellow_or_green : 
  (favorable_marbles : ℚ) / total_marbles = 7 / 10 := by
  sorry

end probability_yellow_or_green_l3923_392314


namespace repeating_decimal_as_fraction_l3923_392373

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 8.137137137... -/
def x : RepeatingDecimal :=
  { integerPart := 8, repeatingPart := 137 }

theorem repeating_decimal_as_fraction :
  toRational x = 2709 / 333 := by
  sorry

end repeating_decimal_as_fraction_l3923_392373


namespace least_three_digit_7_heavy_l3923_392367

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

theorem least_three_digit_7_heavy : 
  (∀ m : ℕ, 100 ≤ m ∧ m < 104 → ¬ is_7_heavy m) ∧ 
  is_7_heavy 104 := by
  sorry

end least_three_digit_7_heavy_l3923_392367


namespace solve_for_C_l3923_392351

theorem solve_for_C : ∃ C : ℝ, (4 * C - 5 = 23) ∧ (C = 7) := by
  sorry

end solve_for_C_l3923_392351


namespace smallest_integer_with_remainder_two_l3923_392386

theorem smallest_integer_with_remainder_two : ∃! m : ℕ,
  m > 1 ∧
  m % 13 = 2 ∧
  m % 5 = 2 ∧
  m % 3 = 2 ∧
  ∀ n : ℕ, n > 1 ∧ n % 13 = 2 ∧ n % 5 = 2 ∧ n % 3 = 2 → m ≤ n :=
by
  use 197
  sorry

end smallest_integer_with_remainder_two_l3923_392386


namespace ppf_combination_l3923_392380

/-- Production Possibility Frontier (PPF) for a single female -/
def individual_ppf (K : ℝ) : ℝ := 40 - 2 * K

/-- Combined Production Possibility Frontier (PPF) for two females -/
def combined_ppf (K : ℝ) : ℝ := 80 - 2 * K

theorem ppf_combination (K : ℝ) (h : K ≤ 40) :
  combined_ppf K = individual_ppf (K / 2) + individual_ppf (K / 2) :=
by sorry

#check ppf_combination

end ppf_combination_l3923_392380


namespace vector_at_negative_two_l3923_392372

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  /-- The vector on the line at parameter t -/
  vector_at : ℝ → ℝ × ℝ

/-- Theorem: Given a parameterized line with specific points, the vector at t = -2 can be determined -/
theorem vector_at_negative_two
  (line : ParameterizedLine)
  (h1 : line.vector_at 1 = (2, 5))
  (h4 : line.vector_at 4 = (8, -7)) :
  line.vector_at (-2) = (-4, 17) := by
  sorry

end vector_at_negative_two_l3923_392372


namespace exists_valid_configuration_l3923_392337

/-- Represents a point on the chessboard -/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- A configuration of 16 points on the chessboard -/
def Configuration := Fin 16 → Point

/-- Checks if a configuration is valid (no three points are collinear) -/
def valid_configuration (config : Configuration) : Prop :=
  ∀ i j k, i < j → j < k → ¬collinear (config i) (config j) (config k)

/-- Theorem: There exists a valid configuration of 16 points on an 8x8 chessboard -/
theorem exists_valid_configuration : ∃ (config : Configuration), valid_configuration config := by
  sorry

end exists_valid_configuration_l3923_392337


namespace e_general_term_l3923_392302

/-- A sequence is a DQ sequence if it can be expressed as the sum of an arithmetic sequence
and a geometric sequence, both with positive integer terms. -/
def is_dq_sequence (e : ℕ → ℕ) : Prop :=
  ∃ (a b : ℕ → ℕ) (d q : ℕ),
    (∀ n, a n = a 1 + (n - 1) * d) ∧
    (∀ n, b n = b 1 * q^(n - 1)) ∧
    (∀ n, e n = a n + b n) ∧
    (∀ n, a n > 0 ∧ b n > 0)

/-- The sequence e_n satisfies the given conditions -/
def e_satisfies_conditions (e : ℕ → ℕ) : Prop :=
  is_dq_sequence e ∧
  e 1 = 3 ∧ e 2 = 6 ∧ e 3 = 11 ∧ e 4 = 20 ∧ e 5 = 37

theorem e_general_term (e : ℕ → ℕ) (h : e_satisfies_conditions e) :
  ∀ n : ℕ, e n = n + 2^n :=
by sorry

end e_general_term_l3923_392302


namespace new_rectangle_area_l3923_392354

/-- Given a rectangle with sides a and b, prove the area of a new rectangle constructed from it. -/
theorem new_rectangle_area (a b : ℝ) (h : 0 < a ∧ a < b) :
  (b + 2 * a) * (b - a) = b^2 + a * b - 2 * a^2 := by
  sorry

end new_rectangle_area_l3923_392354


namespace new_weekly_earnings_l3923_392375

-- Define the original weekly earnings
def original_earnings : ℝ := 60

-- Define the percentage increase
def percentage_increase : ℝ := 0.30

-- Theorem to prove the new weekly earnings
theorem new_weekly_earnings :
  original_earnings * (1 + percentage_increase) = 78 := by
  sorry

end new_weekly_earnings_l3923_392375


namespace probability_two_black_balls_l3923_392378

/-- The probability of drawing two black balls from a box containing 8 white balls and 7 black balls, without replacement. -/
theorem probability_two_black_balls (white_balls black_balls : ℕ) (h1 : white_balls = 8) (h2 : black_balls = 7) :
  (black_balls.choose 2 : ℚ) / ((white_balls + black_balls).choose 2) = 1 / 5 := by
  sorry


end probability_two_black_balls_l3923_392378


namespace parabola_coefficient_ratio_l3923_392326

/-- A parabola with equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given two points on a parabola with the same y-coordinate and x-coordinates
    equidistant from x = 1, the ratio a/b of the parabola coefficients is -1/2 -/
theorem parabola_coefficient_ratio 
  (p : Parabola) 
  (A B : Point) 
  (h1 : A.x = -1 ∧ A.y = 2) 
  (h2 : B.x = 3 ∧ B.y = 2) 
  (h3 : A.y = p.a * A.x^2 + p.b * A.x + p.c) 
  (h4 : B.y = p.a * B.x^2 + p.b * B.x + p.c) :
  p.a / p.b = -1/2 := by sorry

end parabola_coefficient_ratio_l3923_392326


namespace second_half_revenue_l3923_392355

/-- Represents the ticket categories --/
inductive TicketCategory
  | A
  | B
  | C

/-- Calculates the total revenue from ticket sales --/
def calculate_revenue (tickets : Nat) (price : Nat) : Nat :=
  tickets * price

/-- Represents the ticket sales data for Richmond Tigers --/
structure TicketSalesData where
  total_tickets : Nat
  first_half_total : Nat
  first_half_A : Nat
  first_half_B : Nat
  first_half_C : Nat
  price_A : Nat
  price_B : Nat
  price_C : Nat

/-- Theorem: The total revenue from the second half of the season is $154,510 --/
theorem second_half_revenue (data : TicketSalesData) 
  (h1 : data.total_tickets = 9570)
  (h2 : data.first_half_total = 3867)
  (h3 : data.first_half_A = 1350)
  (h4 : data.first_half_B = 1150)
  (h5 : data.first_half_C = 1367)
  (h6 : data.price_A = 50)
  (h7 : data.price_B = 40)
  (h8 : data.price_C = 30) :
  calculate_revenue data.first_half_A data.price_A + 
  calculate_revenue data.first_half_B data.price_B + 
  calculate_revenue data.first_half_C data.price_C = 154510 := by
  sorry


end second_half_revenue_l3923_392355


namespace low_key_function_m_range_l3923_392338

def is_t_degree_low_key (f : ℝ → ℝ) (t : ℝ) (C : Set ℝ) : Prop :=
  ∀ x ∈ C, f (x + t) ≤ f x

def f (m : ℝ) (x : ℝ) : ℝ := -|m * x - 3|

theorem low_key_function_m_range :
  ∀ m : ℝ, (is_t_degree_low_key (f m) 6 (Set.Ici 0)) →
    (m ≤ 0 ∨ m ≥ 1) :=
by sorry

end low_key_function_m_range_l3923_392338


namespace amount_ratio_l3923_392394

def total : ℕ := 1210
def r_amount : ℕ := 400

theorem amount_ratio (p q r : ℕ) 
  (h1 : p + q + r = total)
  (h2 : r = r_amount)
  (h3 : 9 * r = 10 * q) :
  5 * q = 4 * p := by sorry

end amount_ratio_l3923_392394


namespace determinant_of_cubic_roots_l3923_392358

theorem determinant_of_cubic_roots (s p q : ℝ) (a b c : ℝ) : 
  (a^3 - s*a^2 + p*a + q = 0) →
  (b^3 - s*b^2 + p*b + q = 0) →
  (c^3 - s*c^2 + p*c + q = 0) →
  (a + b + c = s) →
  (a*b + b*c + a*c = p) →
  (a*b*c = -q) →
  Matrix.det !![1 + a, 1, 1; 1, 1 + b, 1; 1, 1, 1 + c] = p + 3*s := by
sorry

end determinant_of_cubic_roots_l3923_392358


namespace cookie_difference_l3923_392334

/-- The number of chocolate chip cookies Helen baked yesterday -/
def helen_choc_yesterday : ℕ := 19

/-- The number of raisin cookies Helen baked this morning -/
def helen_raisin_today : ℕ := 231

/-- The number of chocolate chip cookies Helen baked this morning -/
def helen_choc_today : ℕ := 237

/-- The number of oatmeal cookies Helen baked this morning -/
def helen_oatmeal_today : ℕ := 107

/-- The number of chocolate chip cookies Giselle baked -/
def giselle_choc : ℕ := 156

/-- The number of raisin cookies Giselle baked -/
def giselle_raisin : ℕ := 89

/-- The number of chocolate chip cookies Timmy baked -/
def timmy_choc : ℕ := 135

/-- The number of oatmeal cookies Timmy baked -/
def timmy_oatmeal : ℕ := 246

theorem cookie_difference : 
  (helen_choc_yesterday + helen_choc_today + giselle_choc + timmy_choc) - 
  (helen_raisin_today + giselle_raisin) = 227 := by
  sorry

end cookie_difference_l3923_392334


namespace inscribed_circles_theorem_l3923_392399

theorem inscribed_circles_theorem (N : ℕ) (r : ℝ) (h_pos : r > 0) : 
  let R := N * r
  let area_small_circles := N * Real.pi * r^2
  let area_large_circle := Real.pi * R^2
  let area_remaining := area_large_circle - area_small_circles
  (area_small_circles / area_remaining = 1 / 3) → N = 4 := by
sorry

end inscribed_circles_theorem_l3923_392399


namespace parallel_vectors_magnitude_l3923_392391

/-- Given vectors a and b in ℝ², if a is parallel to b, then the magnitude of b is √5. -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.1 = -1 →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  ‖b‖ = Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l3923_392391


namespace pencil_transfer_l3923_392357

/-- Given that Gloria has 2 pencils and Lisa has 99 pencils, 
    if Lisa gives all of her pencils to Gloria, 
    then Gloria will have 101 pencils. -/
theorem pencil_transfer (gloria_initial : ℕ) (lisa_initial : ℕ) 
  (h1 : gloria_initial = 2) 
  (h2 : lisa_initial = 99) : 
  gloria_initial + lisa_initial = 101 := by
  sorry

end pencil_transfer_l3923_392357


namespace white_surface_fraction_is_seven_eighths_l3923_392328

/-- Represents a cube with white and black smaller cubes -/
structure ColoredCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  white_cubes : ℕ
  black_cubes : ℕ

/-- Calculates the fraction of white surface area for a colored cube -/
def white_surface_fraction (c : ColoredCube) : ℚ :=
  sorry

/-- Theorem: The fraction of white surface area for the given cube configuration is 7/8 -/
theorem white_surface_fraction_is_seven_eighths :
  let c : ColoredCube := {
    edge_length := 4,
    total_small_cubes := 64,
    white_cubes := 48,
    black_cubes := 16
  }
  white_surface_fraction c = 7/8 := by
  sorry

end white_surface_fraction_is_seven_eighths_l3923_392328


namespace function_identity_l3923_392388

def is_positive_integer (n : ℤ) : Prop := 0 < n

structure PositiveInteger where
  val : ℤ
  pos : is_positive_integer val

def PositiveIntegerFunction := PositiveInteger → PositiveInteger

theorem function_identity (f : PositiveIntegerFunction) : 
  (∀ (a b : PositiveInteger), ∃ (k : ℤ), a.val ^ 2 + (f a).val * (f b).val = ((f a).val + b.val) * k) →
  (∀ (n : PositiveInteger), (f n).val = n.val) := by
  sorry

end function_identity_l3923_392388


namespace binary_111011_equals_59_l3923_392306

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of the number we're converting -/
def binary_111011 : List Bool := [true, true, true, false, true, true]

/-- Theorem stating that the decimal representation of 111011(2) is 59 -/
theorem binary_111011_equals_59 : binary_to_decimal binary_111011 = 59 := by
  sorry

end binary_111011_equals_59_l3923_392306


namespace soda_preference_result_l3923_392300

/-- The number of people who prefer calling soft drinks "Soda" in a survey. -/
def soda_preference (total_surveyed : ℕ) (central_angle : ℕ) : ℕ :=
  (total_surveyed * central_angle) / 360

/-- Theorem stating that 330 people prefer calling soft drinks "Soda" in the given survey. -/
theorem soda_preference_result : soda_preference 600 198 = 330 := by
  sorry

end soda_preference_result_l3923_392300


namespace percentage_problem_l3923_392319

theorem percentage_problem (P : ℝ) : P = 20 → (P / 100) * 680 = 0.4 * 140 + 80 := by
  sorry

end percentage_problem_l3923_392319


namespace stock_value_ordering_l3923_392350

def initial_investment : ℝ := 200

def alpha_year1 : ℝ := 1.30
def beta_year1 : ℝ := 0.80
def gamma_year1 : ℝ := 1.10
def delta_year1 : ℝ := 0.90

def alpha_year2 : ℝ := 0.85
def beta_year2 : ℝ := 1.30
def gamma_year2 : ℝ := 0.95
def delta_year2 : ℝ := 1.20

def final_alpha : ℝ := initial_investment * alpha_year1 * alpha_year2
def final_beta : ℝ := initial_investment * beta_year1 * beta_year2
def final_gamma : ℝ := initial_investment * gamma_year1 * gamma_year2
def final_delta : ℝ := initial_investment * delta_year1 * delta_year2

theorem stock_value_ordering :
  final_delta < final_beta ∧ final_beta < final_gamma ∧ final_gamma < final_alpha :=
by sorry

end stock_value_ordering_l3923_392350


namespace zunyi_conference_highest_temp_l3923_392370

/-- Given the lowest temperature and maximum temperature difference of a day,
    calculate the highest temperature of that day. -/
def highest_temperature (lowest_temp max_diff : ℝ) : ℝ :=
  lowest_temp + max_diff

/-- Theorem stating that given the specific conditions of the problem,
    the highest temperature of the day is 22°C. -/
theorem zunyi_conference_highest_temp :
  highest_temperature 18 4 = 22 := by
  sorry

end zunyi_conference_highest_temp_l3923_392370


namespace digits_zeros_equality_l3923_392364

/-- Count the number of digits in a natural number -/
def countDigits (n : ℕ) : ℕ := sorry

/-- Count the number of zeros in a natural number -/
def countZeros (n : ℕ) : ℕ := sorry

/-- Sum of digits in a sequence from 1 to n -/
def sumDigits (n : ℕ) : ℕ := (Finset.range n).sum (λ i => countDigits (i + 1))

/-- Sum of zeros in a sequence from 1 to n -/
def sumZeros (n : ℕ) : ℕ := (Finset.range n).sum (λ i => countZeros (i + 1))

/-- Theorem: For any natural number k, the number of all digits in the sequence
    1, 2, 3, ..., 10^k is equal to the number of all zeros in the sequence
    1, 2, 3, ..., 10^(k+1) -/
theorem digits_zeros_equality (k : ℕ) :
  sumDigits (10^k) = sumZeros (10^(k+1)) := by sorry

end digits_zeros_equality_l3923_392364


namespace volume_of_special_parallelepiped_l3923_392345

/-- A rectangular parallelepiped with specific properties -/
structure RectangularParallelepiped where
  /-- Side length of the square face -/
  a : ℝ
  /-- Height perpendicular to the square face -/
  b : ℝ
  /-- The diagonal length is 1 -/
  diagonal_eq_one : 2 * a^2 + b^2 = 1
  /-- The surface area is 1 -/
  surface_area_eq_one : 4 * a * b + 2 * a^2 = 1
  /-- Ensure a and b are positive -/
  a_pos : 0 < a
  b_pos : 0 < b

/-- The volume of a rectangular parallelepiped with the given properties is √2/27 -/
theorem volume_of_special_parallelepiped (p : RectangularParallelepiped) :
  p.a^2 * p.b = Real.sqrt 2 / 27 := by
  sorry

end volume_of_special_parallelepiped_l3923_392345


namespace find_a_l3923_392368

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => 
  if x > 0 then a^x else -a^(-x)

-- State the theorem
theorem find_a : 
  ∀ a : ℝ, 
  (a > 0) → 
  (a ≠ 1) → 
  (∀ x : ℝ, f a x = -(f a (-x))) → 
  (f a (Real.log 4 / Real.log (1/2)) = -3) → 
  a = Real.sqrt 3 := by
sorry

end find_a_l3923_392368


namespace direct_square_variation_problem_l3923_392304

/-- A function representing direct variation with the square of x -/
def direct_square_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem direct_square_variation_problem (y : ℝ → ℝ) :
  (∃ k : ℝ, ∀ x, y x = direct_square_variation k x) →  -- y varies directly as the square of x
  y 3 = 18 →  -- y = 18 when x = 3
  y 6 = 72 :=  -- y = 72 when x = 6
by
  sorry

end direct_square_variation_problem_l3923_392304


namespace non_adjacent_arrangement_count_l3923_392343

/-- The number of ways to arrange 4 boys and 2 girls in a row such that the 2 girls are not adjacent --/
def non_adjacent_arrangements : ℕ := 480

/-- The number of boys --/
def num_boys : ℕ := 4

/-- The number of girls --/
def num_girls : ℕ := 2

/-- The number of spaces available for girls (including ends) --/
def num_spaces : ℕ := num_boys + 1

theorem non_adjacent_arrangement_count :
  non_adjacent_arrangements = num_boys.factorial * (num_spaces.choose num_girls) := by
  sorry

end non_adjacent_arrangement_count_l3923_392343


namespace rectangle_circle_union_area_l3923_392303

/-- The area of the union of a rectangle and a circle -/
theorem rectangle_circle_union_area :
  let rectangle_width : ℝ := 8
  let rectangle_height : ℝ := 12
  let circle_radius : ℝ := 8
  let rectangle_area : ℝ := rectangle_width * rectangle_height
  let circle_area : ℝ := π * circle_radius^2
  let overlap_area : ℝ := (1/4) * circle_area
  rectangle_area + circle_area - overlap_area = 96 + 48 * π :=
by sorry

end rectangle_circle_union_area_l3923_392303


namespace student_average_less_than_true_average_l3923_392313

theorem student_average_less_than_true_average 
  (w x y z : ℝ) (hw : w < x) (hx : x < y) (hy : y < z) :
  (w + x + (y + z) / 2) / 3 < (w + x + y + z) / 4 := by
sorry

end student_average_less_than_true_average_l3923_392313


namespace camel_cost_proof_l3923_392359

/-- The cost of a camel in rupees -/
def camel_cost : ℝ := 5200

/-- The cost of a horse in rupees -/
def horse_cost : ℝ := 2166.67

/-- The cost of an ox in rupees -/
def ox_cost : ℝ := 8666.67

/-- The cost of an elephant in rupees -/
def elephant_cost : ℝ := 13000

theorem camel_cost_proof :
  (10 * camel_cost = 24 * horse_cost) ∧
  (16 * horse_cost = 4 * ox_cost) ∧
  (6 * ox_cost = 4 * elephant_cost) ∧
  (10 * elephant_cost = 130000) →
  camel_cost = 5200 := by
sorry

end camel_cost_proof_l3923_392359


namespace sin_three_pi_half_plus_alpha_l3923_392390

theorem sin_three_pi_half_plus_alpha (α : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) →
  Real.sin (3 * Real.pi / 2 + α) = -4/5 := by
  sorry

end sin_three_pi_half_plus_alpha_l3923_392390


namespace arithmetic_mean_inequality_and_minimum_t_l3923_392308

theorem arithmetic_mean_inequality_and_minimum_t :
  (∀ a b c : ℝ, (((a + b + c) / 3) ^ 2 ≤ (a ^ 2 + b ^ 2 + c ^ 2) / 3) ∧
    (((a + b + c) / 3) ^ 2 = (a ^ 2 + b ^ 2 + c ^ 2) / 3 ↔ a = b ∧ b = c)) ∧
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (∀ t : ℝ, Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ t * Real.sqrt (x + y + z) →
      t ≥ Real.sqrt 3) ∧
    ∃ t : ℝ, t = Real.sqrt 3 ∧
      Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ t * Real.sqrt (x + y + z)) :=
by sorry

end arithmetic_mean_inequality_and_minimum_t_l3923_392308


namespace greatest_integer_solution_l3923_392393

theorem greatest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, y > x → 7 - 5 * y + y^2 ≥ 28) ∧ 
  (7 - 5 * x + x^2 < 28) → 
  x = 7 := by
sorry

end greatest_integer_solution_l3923_392393


namespace smallest_five_digit_congruent_to_3_mod_17_l3923_392323

theorem smallest_five_digit_congruent_to_3_mod_17 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit positive integer
  (n % 17 = 3) ∧              -- congruent to 3 modulo 17
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 3 → m ≥ n) ∧  -- smallest such integer
  n = 10012 := by
sorry

end smallest_five_digit_congruent_to_3_mod_17_l3923_392323


namespace thirty_five_million_scientific_notation_l3923_392379

-- Define the value of one million
def million : ℝ := 10^6

-- Theorem statement
theorem thirty_five_million_scientific_notation :
  35 * million = 3.5 * 10^7 := by
  sorry

end thirty_five_million_scientific_notation_l3923_392379


namespace infinite_slips_with_same_number_l3923_392333

-- Define a type for slip numbers
def SlipNumber : Type := ℕ

-- Define the set of all slips
def AllSlips : Set SlipNumber := Set.univ

-- Define the property that any infinite subset has at least two slips with the same number
def HasDuplicatesInInfiniteSubsets (S : Set SlipNumber) : Prop :=
  ∀ (T : Set SlipNumber), T ⊆ S → T.Infinite → ∃ (n : SlipNumber), (∃ (s t : SlipNumber), s ∈ T ∧ t ∈ T ∧ s ≠ t ∧ n = s ∧ n = t)

-- State the theorem
theorem infinite_slips_with_same_number :
  AllSlips.Infinite →
  HasDuplicatesInInfiniteSubsets AllSlips →
  ∃ (n : SlipNumber), {s : SlipNumber | s ∈ AllSlips ∧ n = s}.Infinite :=
by sorry

end infinite_slips_with_same_number_l3923_392333


namespace middle_number_problem_l3923_392310

theorem middle_number_problem (x y z : ℤ) : 
  x < y ∧ y < z →
  x + y = 18 →
  x + z = 25 →
  y + z = 27 →
  y = 10 := by
sorry

end middle_number_problem_l3923_392310


namespace sin_two_alpha_zero_l3923_392325

theorem sin_two_alpha_zero (α : Real) (f : Real → Real)
  (h1 : ∀ x, f x = Real.sin x - Real.cos x)
  (h2 : f α = 1) : Real.sin (2 * α) = 0 := by
  sorry

end sin_two_alpha_zero_l3923_392325


namespace min_max_abs_quadratic_minus_linear_exists_y_for_min_max_abs_quadratic_minus_linear_l3923_392387

theorem min_max_abs_quadratic_minus_linear (y : ℝ) :
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2*x*y| = 4) ∧
  (∀ (x : ℝ), 0 ≤ x → x ≤ 2 → |x^2 - 2*x*y| ≤ 4) :=
by sorry

theorem exists_y_for_min_max_abs_quadratic_minus_linear :
  ∃ (y : ℝ), ∀ (x : ℝ), 0 ≤ x → x ≤ 2 → |x^2 - 2*x*y| ≤ 4 :=
by sorry

end min_max_abs_quadratic_minus_linear_exists_y_for_min_max_abs_quadratic_minus_linear_l3923_392387


namespace completing_square_result_l3923_392362

theorem completing_square_result (x : ℝ) : 
  (x^2 - 6*x - 8 = 0) ↔ ((x - 3)^2 = 17) := by
  sorry

end completing_square_result_l3923_392362


namespace parallel_vectors_not_always_same_direction_l3923_392352

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (v w : V) : Prop := ∃ k : ℝ, v = k • w

theorem parallel_vectors_not_always_same_direction :
  ∃ (v w : V), parallel v w ∧ ¬(∃ k : ℝ, k > 0 ∧ v = k • w) :=
sorry

end parallel_vectors_not_always_same_direction_l3923_392352


namespace temperature_data_inconsistency_l3923_392307

theorem temperature_data_inconsistency 
  (x_bar : ℝ) 
  (m : ℝ) 
  (S_squared : ℝ) 
  (h_x_bar : x_bar = 0) 
  (h_m : m = 4) 
  (h_S_squared : S_squared = 15.917) : 
  ¬(|x_bar - m| ≤ Real.sqrt S_squared) := by
  sorry

end temperature_data_inconsistency_l3923_392307


namespace fraction_product_theorem_l3923_392305

theorem fraction_product_theorem : 
  (7 / 4 : ℚ) * (8 / 16 : ℚ) * (21 / 14 : ℚ) * (15 / 25 : ℚ) * 
  (28 / 21 : ℚ) * (20 / 40 : ℚ) * (49 / 28 : ℚ) * (25 / 50 : ℚ) = 147 / 320 := by
  sorry

end fraction_product_theorem_l3923_392305


namespace min_value_theorem_l3923_392374

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  2 ≤ b / a + 3 / (b + 1) :=
by sorry

end min_value_theorem_l3923_392374


namespace min_sum_values_l3923_392318

theorem min_sum_values (a b x y : ℝ) : 
  a > 0 → b > 0 → x > 0 → y > 0 →
  a + b = 10 →
  a / x + b / y = 1 →
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → a / x' + b / y' = 1 → x' + y' ≥ 16) →
  x + y = 16 →
  ((a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1)) := by
sorry

end min_sum_values_l3923_392318


namespace unique_triple_solution_l3923_392353

theorem unique_triple_solution : 
  ∃! (x y z : ℝ), x + y = 4 ∧ x * y - z^2 = 4 :=
by
  sorry

end unique_triple_solution_l3923_392353


namespace polygon_area_is_787_5_l3923_392396

/-- The area of a triangle given its vertices -/
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The vertices of the polygon -/
def vertices : List (ℝ × ℝ) :=
  [(0, 0), (15, 0), (45, 30), (45, 45), (30, 45), (0, 15)]

/-- The area of the polygon -/
def polygon_area : ℝ :=
  triangle_area 0 0 15 0 0 15 +
  triangle_area 15 0 45 30 0 15 +
  triangle_area 45 30 45 45 30 45

theorem polygon_area_is_787_5 :
  polygon_area = 787.5 := by
  sorry

end polygon_area_is_787_5_l3923_392396


namespace fraction_problem_l3923_392341

theorem fraction_problem (x : ℚ) : x * 8 + 2 = 8 → x = 3/4 := by
  sorry

end fraction_problem_l3923_392341


namespace factorization_cubic_quadratic_l3923_392395

theorem factorization_cubic_quadratic (x y : ℝ) : x^3*y - 4*x*y = x*y*(x-2)*(x+2) := by
  sorry

end factorization_cubic_quadratic_l3923_392395


namespace parallelogram_area_36_18_l3923_392317

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 36 cm and height 18 cm is 648 square centimeters -/
theorem parallelogram_area_36_18 : parallelogram_area 36 18 = 648 := by
  sorry

end parallelogram_area_36_18_l3923_392317


namespace regression_analysis_relationship_l3923_392316

/-- Represents a statistical relationship between two variables -/
inductive StatisticalRelationship
| Correlation

/-- Represents a method of statistical analysis -/
inductive StatisticalAnalysisMethod
| RegressionAnalysis

/-- The relationship between variables in regression analysis -/
def relationship_in_regression_analysis : StatisticalRelationship := StatisticalRelationship.Correlation

theorem regression_analysis_relationship :
  relationship_in_regression_analysis = StatisticalRelationship.Correlation := by
  sorry

end regression_analysis_relationship_l3923_392316


namespace infinite_fraction_value_l3923_392332

theorem infinite_fraction_value : 
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (3 + Real.sqrt 69) / 2 := by
sorry

end infinite_fraction_value_l3923_392332


namespace pyramid_max_volume_l3923_392336

/-- The maximum volume of a pyramid with given base side lengths and angle constraints. -/
theorem pyramid_max_volume (AB AC : ℝ) (sin_BAC : ℝ) (max_lateral_angle : ℝ) :
  AB = 3 →
  AC = 5 →
  sin_BAC = 3/5 →
  max_lateral_angle = 60 * π / 180 →
  ∃ (V : ℝ), V = (5 * Real.sqrt 174) / 4 ∧ 
    ∀ (V' : ℝ), V' ≤ V := by
  sorry

end pyramid_max_volume_l3923_392336


namespace arithmetic_fraction_subtraction_l3923_392349

theorem arithmetic_fraction_subtraction :
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12 := by
  sorry

end arithmetic_fraction_subtraction_l3923_392349


namespace cyclists_speed_cyclists_speed_is_10_l3923_392346

/-- Two cyclists traveling in opposite directions for 2.5 hours end up 50 km apart. -/
theorem cyclists_speed : ℝ → Prop :=
  fun speed : ℝ =>
    let time : ℝ := 2.5
    let distance : ℝ := 50
    2 * speed * time = distance

/-- The speed of each cyclist is 10 km/h. -/
theorem cyclists_speed_is_10 : cyclists_speed 10 := by
  sorry

end cyclists_speed_cyclists_speed_is_10_l3923_392346


namespace limit_fraction_three_n_l3923_392309

/-- The limit of (3^n - 1) / (3^(n+1) + 1) as n approaches infinity is 1/3 -/
theorem limit_fraction_three_n (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |((3^n - 1) / (3^(n+1) + 1)) - 1/3| < ε :=
sorry

end limit_fraction_three_n_l3923_392309
