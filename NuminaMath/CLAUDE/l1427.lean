import Mathlib

namespace NUMINAMATH_CALUDE_probability_one_common_number_l1427_142707

/-- The number of numbers in the lottery -/
def totalNumbers : ℕ := 45

/-- The number of numbers each participant chooses -/
def chosenNumbers : ℕ := 6

/-- The probability of exactly one common number between two independently chosen combinations -/
def probabilityOneCommon : ℚ :=
  (chosenNumbers : ℚ) * (Nat.choose (totalNumbers - chosenNumbers) (chosenNumbers - 1) : ℚ) /
  (Nat.choose totalNumbers chosenNumbers : ℚ)

/-- Theorem stating the probability of exactly one common number -/
theorem probability_one_common_number :
  probabilityOneCommon = (6 : ℚ) * (Nat.choose 39 5 : ℚ) / (Nat.choose 45 6 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_probability_one_common_number_l1427_142707


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1427_142702

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 4}
def Q : Set ℝ := {x : ℝ | x ≥ 3}

-- State the theorem
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1427_142702


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_l1427_142710

theorem smallest_four_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 1000 ∧ n < 10000) ∧ 
    (75 * n) % 345 = 225 ∧
    (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ (75 * m) % 345 = 225 → m ≥ n) ∧
    n = 1015 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_l1427_142710


namespace NUMINAMATH_CALUDE_karlee_grapes_l1427_142779

theorem karlee_grapes (G : ℚ) : 
  (G * 3/5 * 3/5 + G * 3/5) = 96 → G = 100 := by
  sorry

end NUMINAMATH_CALUDE_karlee_grapes_l1427_142779


namespace NUMINAMATH_CALUDE_division_problem_l1427_142752

theorem division_problem (n : ℕ) : n / 14 = 9 ∧ n % 14 = 1 → n = 127 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1427_142752


namespace NUMINAMATH_CALUDE_chicken_surprise_serving_weight_l1427_142758

/-- Represents the recipe for Chicken Surprise -/
structure ChickenSurprise where
  servings : ℕ
  chickenPounds : ℚ
  stuffingOunces : ℕ

/-- Calculates the weight of one serving of Chicken Surprise in ounces -/
def servingWeight (recipe : ChickenSurprise) : ℚ :=
  let totalOunces := recipe.chickenPounds * 16 + recipe.stuffingOunces
  totalOunces / recipe.servings

/-- Theorem stating that one serving of Chicken Surprise is 8 ounces -/
theorem chicken_surprise_serving_weight :
  let recipe := ChickenSurprise.mk 12 (9/2) 24
  servingWeight recipe = 8 := by
  sorry

end NUMINAMATH_CALUDE_chicken_surprise_serving_weight_l1427_142758


namespace NUMINAMATH_CALUDE_largest_gold_coins_distribution_l1427_142755

theorem largest_gold_coins_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 15 * k + 3) → 
  n < 150 → 
  (∀ m : ℕ, (∃ j : ℕ, m = 15 * j + 3) → m < 150 → m ≤ n) →
  n = 138 := by
sorry

end NUMINAMATH_CALUDE_largest_gold_coins_distribution_l1427_142755


namespace NUMINAMATH_CALUDE_tims_initial_amount_l1427_142790

/-- Tim's candy bar purchase scenario -/
def candy_bar_purchase (initial_amount paid change : ℕ) : Prop :=
  initial_amount = paid + change

/-- Theorem: Tim's initial amount before buying the candy bar -/
theorem tims_initial_amount : ∃ (initial_amount : ℕ), 
  candy_bar_purchase initial_amount 45 5 ∧ initial_amount = 50 := by
  sorry

end NUMINAMATH_CALUDE_tims_initial_amount_l1427_142790


namespace NUMINAMATH_CALUDE_sterling_total_questions_l1427_142730

/-- Represents the candy reward system and Sterling's performance --/
structure CandyReward where
  correct_reward : ℕ
  incorrect_penalty : ℕ
  correct_answers : ℕ
  total_questions : ℕ
  hypothetical_candy : ℕ

/-- Theorem stating that Sterling answered 9 questions in total --/
theorem sterling_total_questions 
  (reward : CandyReward) 
  (h1 : reward.correct_reward = 3)
  (h2 : reward.incorrect_penalty = 2)
  (h3 : reward.correct_answers = 7)
  (h4 : reward.hypothetical_candy = 31)
  (h5 : reward.hypothetical_candy = 
    (reward.correct_answers + 2) * reward.correct_reward - 
    (reward.total_questions - reward.correct_answers - 2) * reward.incorrect_penalty) : 
  reward.total_questions = 9 := by
  sorry

end NUMINAMATH_CALUDE_sterling_total_questions_l1427_142730


namespace NUMINAMATH_CALUDE_quadratic_value_relation_l1427_142744

theorem quadratic_value_relation (x : ℝ) (h : x^2 + x + 1 = 8) : 4*x^2 + 4*x + 9 = 37 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_relation_l1427_142744


namespace NUMINAMATH_CALUDE_blake_purchase_change_l1427_142795

/-- The amount of change Blake will receive after his purchase -/
def blakes_change (lollipop_count : ℕ) (chocolate_pack_count : ℕ) (lollipop_price : ℕ) (bill_count : ℕ) (bill_value : ℕ) : ℕ :=
  let chocolate_pack_price := 4 * lollipop_price
  let total_cost := lollipop_count * lollipop_price + chocolate_pack_count * chocolate_pack_price
  let amount_paid := bill_count * bill_value
  amount_paid - total_cost

theorem blake_purchase_change :
  blakes_change 4 6 2 6 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_blake_purchase_change_l1427_142795


namespace NUMINAMATH_CALUDE_f_g_inequality_implies_a_range_l1427_142742

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

def g (x : ℝ) : ℝ := Real.exp x

def H (a : ℝ) (x : ℝ) : ℝ := f a x / g x

theorem f_g_inequality_implies_a_range :
  ∀ a : ℝ,
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 ∧ x₁ > x₂ →
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔
  -1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2 :=
by sorry

end

end NUMINAMATH_CALUDE_f_g_inequality_implies_a_range_l1427_142742


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1427_142760

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of the 1st, 7th, and 13th terms equals 8 -/
def product_condition (a : ℕ → ℝ) : Prop :=
  a 1 * a 7 * a 13 = 8

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : product_condition a) : 
  a 3 * a 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1427_142760


namespace NUMINAMATH_CALUDE_divisibility_statements_l1427_142788

theorem divisibility_statements :
  (12 % 2 = 0) ∧
  (123 % 3 = 0) ∧
  (1234 % 4 ≠ 0) ∧
  (12345 % 5 = 0) ∧
  (123456 % 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_statements_l1427_142788


namespace NUMINAMATH_CALUDE_sophomore_sample_count_l1427_142765

/-- Given a school with 1000 students, including 320 sophomores,
    prove that a random sample of 200 students will contain 64 sophomores. -/
theorem sophomore_sample_count (total_students : ℕ) (sophomores : ℕ) (sample_size : ℕ) :
  total_students = 1000 →
  sophomores = 320 →
  sample_size = 200 →
  (sophomores : ℚ) / total_students * sample_size = 64 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_sample_count_l1427_142765


namespace NUMINAMATH_CALUDE_triangle_angle_120_l1427_142774

/-- In a triangle ABC with side lengths a, b, and c, if a^2 = b^2 + bc + c^2, then angle A is 120° -/
theorem triangle_angle_120 (a b c : ℝ) (h : a^2 = b^2 + b*c + c^2) :
  let A := Real.arccos ((c^2 + b^2 - a^2) / (2*b*c))
  A = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_120_l1427_142774


namespace NUMINAMATH_CALUDE_height_to_sphere_ratio_l1427_142789

/-- A truncated right circular cone with an inscribed sphere -/
structure TruncatedConeWithSphere where
  R : ℝ  -- radius of the larger base
  r : ℝ  -- radius of the smaller base
  H : ℝ  -- height of the truncated cone
  s : ℝ  -- radius of the inscribed sphere
  R_positive : R > 0
  r_positive : r > 0
  H_positive : H > 0
  s_positive : s > 0
  sphere_inscribed : s = Real.sqrt (R * r)
  volume_relation : π * H * (R^2 + R*r + r^2) / 3 = 4 * π * s^3

/-- The ratio of the height of the truncated cone to the radius of the sphere is 4 -/
theorem height_to_sphere_ratio (cone : TruncatedConeWithSphere) : 
  cone.H / cone.s = 4 := by
  sorry

end NUMINAMATH_CALUDE_height_to_sphere_ratio_l1427_142789


namespace NUMINAMATH_CALUDE_det_B_is_one_l1427_142725

def B (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]

theorem det_B_is_one (a d : ℝ) (h : B a d + (B a d)⁻¹ = 0) : 
  Matrix.det (B a d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_B_is_one_l1427_142725


namespace NUMINAMATH_CALUDE_triple_hash_40_l1427_142798

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem triple_hash_40 : hash (hash (hash 40)) = 12.56 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_40_l1427_142798


namespace NUMINAMATH_CALUDE_carries_hourly_wage_l1427_142727

/-- Carrie's work and savings scenario --/
theorem carries_hourly_wage (hours_per_week : ℕ) (weeks : ℕ) (bike_cost : ℕ) (leftover : ℕ) :
  hours_per_week = 35 →
  weeks = 4 →
  bike_cost = 400 →
  leftover = 720 →
  ∃ (hourly_wage : ℚ), hourly_wage = 8 ∧ 
    (hourly_wage * (hours_per_week * weeks : ℚ) : ℚ) = (bike_cost + leftover : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_carries_hourly_wage_l1427_142727


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1427_142739

/-- An arithmetic sequence with first term 2 and the sum of the second and fourth terms equal to the sixth term has a common difference of 2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 1 = 2)  -- First term is 2
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  (h3 : a 2 + a 4 = a 6)  -- Sum of second and fourth terms equals sixth term
  : d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1427_142739


namespace NUMINAMATH_CALUDE_flowerbed_count_l1427_142731

theorem flowerbed_count (total_seeds : ℕ) (seeds_per_bed : ℕ) (h1 : total_seeds = 32) (h2 : seeds_per_bed = 4) :
  total_seeds / seeds_per_bed = 8 := by
  sorry

end NUMINAMATH_CALUDE_flowerbed_count_l1427_142731


namespace NUMINAMATH_CALUDE_max_value_a_l1427_142786

theorem max_value_a (a b c d : ℝ) 
  (h1 : b + c + d = 3 - a) 
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) : 
  a ≤ 2 ∧ ∃ (b c d : ℝ), b + c + d = 3 - 2 ∧ 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - 2^2 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l1427_142786


namespace NUMINAMATH_CALUDE_gluten_free_pasta_cost_l1427_142734

theorem gluten_free_pasta_cost 
  (mustard_oil_quantity : ℕ)
  (mustard_oil_price : ℚ)
  (pasta_quantity : ℕ)
  (pasta_sauce_quantity : ℕ)
  (pasta_sauce_price : ℚ)
  (initial_money : ℚ)
  (remaining_money : ℚ)
  (h1 : mustard_oil_quantity = 2)
  (h2 : mustard_oil_price = 13)
  (h3 : pasta_quantity = 3)
  (h4 : pasta_sauce_quantity = 1)
  (h5 : pasta_sauce_price = 5)
  (h6 : initial_money = 50)
  (h7 : remaining_money = 7) :
  (initial_money - remaining_money - 
   (mustard_oil_quantity * mustard_oil_price + pasta_sauce_quantity * pasta_sauce_price)) / pasta_quantity = 4 := by
  sorry

end NUMINAMATH_CALUDE_gluten_free_pasta_cost_l1427_142734


namespace NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l1427_142770

theorem triangle_sine_sum_inequality (A B C : Real) : 
  A + B + C = π → 0 < A → 0 < B → 0 < C → 
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l1427_142770


namespace NUMINAMATH_CALUDE_day_relationship_l1427_142715

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to determine the day of the week for a given day number -/
def dayOfWeek (dayNumber : ℕ) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- Theorem stating the relationship between days in different years -/
theorem day_relationship (N : ℕ) :
  dayOfWeek 290 = DayOfWeek.Wednesday →
  dayOfWeek 210 = DayOfWeek.Wednesday →
  dayOfWeek 110 = DayOfWeek.Wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_day_relationship_l1427_142715


namespace NUMINAMATH_CALUDE_miriam_marbles_l1427_142751

theorem miriam_marbles (initial_marbles current_marbles brother_marbles sister_marbles friend_marbles : ℕ) :
  initial_marbles = 300 →
  current_marbles = 30 →
  sister_marbles = 2 * brother_marbles →
  friend_marbles = 90 →
  initial_marbles = current_marbles + brother_marbles + sister_marbles + friend_marbles →
  brother_marbles = 60 := by
sorry

end NUMINAMATH_CALUDE_miriam_marbles_l1427_142751


namespace NUMINAMATH_CALUDE_pipe_a_fills_in_12_hours_l1427_142703

/-- Represents the time (in hours) taken by pipe A to fill the cistern -/
def pipe_a_time : ℝ := 12

/-- Represents the time (in hours) taken by pipe B to leak out the cistern -/
def pipe_b_time : ℝ := 18

/-- Represents the time (in hours) taken to fill the cistern when both pipes are open -/
def both_pipes_time : ℝ := 36

/-- Proves that pipe A fills the cistern in 12 hours given the conditions -/
theorem pipe_a_fills_in_12_hours :
  (1 / pipe_a_time) - (1 / pipe_b_time) = (1 / both_pipes_time) :=
by sorry

end NUMINAMATH_CALUDE_pipe_a_fills_in_12_hours_l1427_142703


namespace NUMINAMATH_CALUDE_pencil_sharpening_l1427_142780

/-- The length sharpened off a pencil is equal to the difference between its initial and final lengths. -/
theorem pencil_sharpening (initial_length final_length : ℝ) :
  initial_length ≥ final_length →
  initial_length - final_length = initial_length - final_length :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_l1427_142780


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1427_142764

theorem arithmetic_equality : 3 * 9 + 4 * 10 + 11 * 3 + 3 * 8 = 124 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1427_142764


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l1427_142723

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  heart_rate * pace * race_distance

/-- Proves that the athlete's heart beats 21600 times during the 30-mile race. -/
theorem athlete_heartbeats :
  let heart_rate : ℕ := 120  -- heartbeats per minute
  let pace : ℕ := 6          -- minutes per mile
  let race_distance : ℕ := 30 -- miles
  total_heartbeats heart_rate pace race_distance = 21600 := by
  sorry

#eval total_heartbeats 120 6 30

end NUMINAMATH_CALUDE_athlete_heartbeats_l1427_142723


namespace NUMINAMATH_CALUDE_smallest_x_for_540x_perfect_square_l1427_142776

theorem smallest_x_for_540x_perfect_square :
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), ∃ (M : ℤ), 540 * y = M^2 → x ≤ y) ∧
    (∃ (M : ℤ), 540 * x = M^2) ∧
    x = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_540x_perfect_square_l1427_142776


namespace NUMINAMATH_CALUDE_horner_V₁_equals_22_l1427_142767

-- Define the polynomial coefficients
def a₅ : ℝ := 4
def a₄ : ℝ := 2
def a₃ : ℝ := 3.5
def a₂ : ℝ := -2.6
def a₁ : ℝ := 1.7
def a₀ : ℝ := -0.8

-- Define the polynomial
def f (x : ℝ) : ℝ := a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- Define Horner's method for this polynomial
def horner (x : ℝ) : ℝ := ((((a₅ * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀

-- Define V₁ in Horner's method
def V₁ (x : ℝ) : ℝ := a₅ * x + a₄

-- Theorem statement
theorem horner_V₁_equals_22 : V₁ 5 = 22 := by sorry

end NUMINAMATH_CALUDE_horner_V₁_equals_22_l1427_142767


namespace NUMINAMATH_CALUDE_problem_solution_l1427_142713

theorem problem_solution (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 1/2) : m = 100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1427_142713


namespace NUMINAMATH_CALUDE_jan_extra_miles_l1427_142711

/-- Represents the driving scenario of Ian, Han, and Jan -/
structure DrivingScenario where
  ian_speed : ℝ
  ian_time : ℝ
  han_speed : ℝ
  han_time : ℝ
  jan_speed : ℝ
  jan_time : ℝ

/-- The conditions of the driving scenario -/
def scenario_conditions (s : DrivingScenario) : Prop :=
  s.han_speed = s.ian_speed + 10 ∧
  s.han_time = s.ian_time ∧
  s.jan_time = s.ian_time + 3 ∧
  s.jan_speed = s.ian_speed + 15 ∧
  s.han_speed * s.han_time = s.ian_speed * s.ian_time + 90

/-- The theorem to be proved -/
theorem jan_extra_miles (s : DrivingScenario) :
  scenario_conditions s →
  s.jan_speed * s.jan_time - s.ian_speed * s.ian_time = 210 := by
  sorry


end NUMINAMATH_CALUDE_jan_extra_miles_l1427_142711


namespace NUMINAMATH_CALUDE_triangle_properties_l1427_142724

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  2 * c * Real.sin A = Real.sqrt 3 * a →
  b = 2 →
  c = Real.sqrt 7 →
  C = π/3 ∧ (1/2 * a * b * Real.sin C = (3 * Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1427_142724


namespace NUMINAMATH_CALUDE_integral_x_plus_sqrt_4_minus_x_squared_l1427_142726

open Set
open MeasureTheory
open Interval

/-- The definite integral of x + √(4 - x^2) from -2 to 2 equals 2π -/
theorem integral_x_plus_sqrt_4_minus_x_squared : 
  ∫ x in (-2)..2, (x + Real.sqrt (4 - x^2)) = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_integral_x_plus_sqrt_4_minus_x_squared_l1427_142726


namespace NUMINAMATH_CALUDE_solve_equation_l1427_142743

theorem solve_equation (x : ℝ) (number : ℝ) : 
  x = 32 → 
  35 - (23 - (15 - x)) = number * 2 / (1 / 2) → 
  number = -1.25 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1427_142743


namespace NUMINAMATH_CALUDE_animal_video_ratio_l1427_142701

theorem animal_video_ratio :
  ∀ (total_time cat_time dog_time gorilla_time : ℝ),
    total_time = 36 →
    cat_time = 4 →
    gorilla_time = 2 * (cat_time + dog_time) →
    total_time = cat_time + dog_time + gorilla_time →
    dog_time / cat_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_animal_video_ratio_l1427_142701


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1427_142781

theorem simplify_and_evaluate : 
  let x : ℝ := -3
  3 * (2 * x^2 - 5 * x) - 2 * (-3 * x - 2 + 3 * x^2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1427_142781


namespace NUMINAMATH_CALUDE_laptop_price_l1427_142733

theorem laptop_price (sticker_price : ℝ) : 
  (sticker_price * 0.8 - 100 = sticker_price * 0.7 - 25) → sticker_price = 750 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_l1427_142733


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l1427_142773

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l1427_142773


namespace NUMINAMATH_CALUDE_room_width_calculation_l1427_142756

theorem room_width_calculation (length width total_area : ℝ) 
  (h1 : length = 4)
  (h2 : total_area = 80)
  (h3 : total_area = length * width) :
  width = 20 := by
sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1427_142756


namespace NUMINAMATH_CALUDE_october_birth_percentage_l1427_142700

def total_people : ℕ := 100
def october_births : ℕ := 6

theorem october_birth_percentage :
  (october_births : ℚ) / total_people * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_october_birth_percentage_l1427_142700


namespace NUMINAMATH_CALUDE_tower_comparison_l1427_142721

def tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (tower base n)

theorem tower_comparison (n : ℕ) : ∃ m : ℕ, ∀ k ≥ m,
  (tower 3 k > tower 2 (k + 1)) ∧ (tower 4 k > tower 3 k) := by
  sorry

#check tower_comparison

end NUMINAMATH_CALUDE_tower_comparison_l1427_142721


namespace NUMINAMATH_CALUDE_statements_equivalent_l1427_142763

-- Define the propositions P and Q
variable (P Q : Prop)

-- Define the two logical statements
def statement1 : Prop := ¬P → Q
def statement2 : Prop := ¬Q → P

-- Theorem stating the logical equivalence of the two statements
theorem statements_equivalent : statement1 P Q ↔ statement2 P Q := by
  sorry

end NUMINAMATH_CALUDE_statements_equivalent_l1427_142763


namespace NUMINAMATH_CALUDE_negation_equivalence_l1427_142708

theorem negation_equivalence :
  (¬ ∃ x : ℤ, 7 ∣ x ∧ ¬ Odd x) ↔ (∀ x : ℤ, 7 ∣ x → Odd x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1427_142708


namespace NUMINAMATH_CALUDE_problem_solution_l1427_142759

theorem problem_solution (a b : ℝ) (h : a^2 - 2*b^2 - 2 = 0) :
  -3*a^2 + 6*b^2 + 2023 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1427_142759


namespace NUMINAMATH_CALUDE_triangle_problem_l1427_142768

theorem triangle_problem (A B C a b c : Real) : 
  -- Given conditions
  (Real.sqrt 3 * Real.cos (2 * A) + 1 = 4 * Real.sin (π / 6 + A) * Real.sin (π / 3 - A)) →
  (a = Real.sqrt 2) →
  (b ≥ a) →
  -- Triangle properties
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Conclusions
  (A = π / 4) ∧
  (0 ≤ Real.sqrt 2 * b - c ∧ Real.sqrt 2 * b - c < 2) := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l1427_142768


namespace NUMINAMATH_CALUDE_negation_equivalence_l1427_142794

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 2 * x - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1427_142794


namespace NUMINAMATH_CALUDE_molecular_weight_CaO_is_56_l1427_142736

/-- The molecular weight of CaO in grams per mole -/
def molecular_weight_CaO : ℝ := 56

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 7

/-- The total weight of the given moles of CaO in grams -/
def given_weight : ℝ := 392

/-- Theorem stating that the molecular weight of CaO is 56 grams/mole -/
theorem molecular_weight_CaO_is_56 :
  molecular_weight_CaO = given_weight / given_moles :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_CaO_is_56_l1427_142736


namespace NUMINAMATH_CALUDE_john_hearing_aid_cost_l1427_142754

/-- Calculates the personal cost for replacing hearing aids given insurance conditions --/
def personal_cost_hearing_aids (
  cost_two_aids : ℕ)  -- Cost of two hearing aids
  (cost_third_aid : ℕ) -- Cost of the third hearing aid
  (deductible : ℕ) -- Insurance deductible
  (coverage_rate : ℚ) -- Insurance coverage rate after deductible
  (coverage_limit : ℕ) -- Insurance coverage limit
  : ℕ :=
  sorry

/-- Theorem stating the personal cost for John's hearing aids --/
theorem john_hearing_aid_cost :
  personal_cost_hearing_aids 5000 3000 500 (4/5) 3500 = 4500 :=
sorry

end NUMINAMATH_CALUDE_john_hearing_aid_cost_l1427_142754


namespace NUMINAMATH_CALUDE_second_recipe_amount_is_one_l1427_142793

/-- The amount of lower sodium soy sauce in the second recipe -/
def second_recipe_amount : ℚ :=
  let bottle_ounces : ℚ := 16
  let ounces_per_cup : ℚ := 8
  let first_recipe_cups : ℚ := 2
  let third_recipe_cups : ℚ := 3
  let total_bottles : ℚ := 3
  let total_ounces : ℚ := total_bottles * bottle_ounces
  let total_cups : ℚ := total_ounces / ounces_per_cup
  total_cups - first_recipe_cups - third_recipe_cups

theorem second_recipe_amount_is_one :
  second_recipe_amount = 1 := by sorry

end NUMINAMATH_CALUDE_second_recipe_amount_is_one_l1427_142793


namespace NUMINAMATH_CALUDE_innings_played_l1427_142748

/-- Represents the number of innings played by a cricket player. -/
def innings : ℕ := sorry

/-- Represents the current average runs of the player. -/
def currentAverage : ℕ := 24

/-- Represents the runs needed in the next innings. -/
def nextInningsRuns : ℕ := 96

/-- Represents the increase in average after the next innings. -/
def averageIncrease : ℕ := 8

/-- Theorem stating that the number of innings played is 8. -/
theorem innings_played : innings = 8 := by sorry

end NUMINAMATH_CALUDE_innings_played_l1427_142748


namespace NUMINAMATH_CALUDE_gina_initial_amount_l1427_142716

def initial_amount (remaining : ℚ) (fraction_given : ℚ) : ℚ :=
  remaining / (1 - fraction_given)

theorem gina_initial_amount :
  let fraction_to_mom : ℚ := 1/4
  let fraction_for_clothes : ℚ := 1/8
  let fraction_to_charity : ℚ := 1/5
  let total_fraction_given := fraction_to_mom + fraction_for_clothes + fraction_to_charity
  let remaining_amount : ℚ := 170
  initial_amount remaining_amount total_fraction_given = 400 := by
  sorry

end NUMINAMATH_CALUDE_gina_initial_amount_l1427_142716


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_two_range_of_a_for_inequality_l1427_142753

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3| - |x + a|

-- Theorem for part I
theorem solution_set_when_a_eq_two :
  {x : ℝ | f 2 x < 1} = {x : ℝ | x > 0} := by sorry

-- Theorem for part II
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x, f a x ≤ 2*a} = {a : ℝ | a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_two_range_of_a_for_inequality_l1427_142753


namespace NUMINAMATH_CALUDE_unique_root_P_l1427_142706

-- Define the polynomial sequence
def P : ℕ → ℝ → ℝ
  | 0, x => 0
  | 1, x => x
  | (n+2), x => x * P (n+1) x + (1 - x) * P n x

-- State the theorem
theorem unique_root_P (n : ℕ) (hn : n ≥ 1) : 
  ∀ x : ℝ, P n x = 0 ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_root_P_l1427_142706


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1427_142714

theorem modulus_of_complex_number (z : ℂ) :
  z = Complex.mk (Real.sqrt 3 / 2) (-3 / 2) →
  Complex.abs z = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1427_142714


namespace NUMINAMATH_CALUDE_students_yes_R_is_400_l1427_142732

/-- Given information about student responses to subjects M and R -/
structure StudentResponses where
  total : Nat
  yes_only_M : Nat
  no_both : Nat

/-- Calculate the number of students who answered yes for subject R -/
def students_yes_R (responses : StudentResponses) : Nat :=
  responses.total - responses.yes_only_M - responses.no_both

/-- Theorem stating that the number of students who answered yes for R is 400 -/
theorem students_yes_R_is_400 (responses : StudentResponses)
  (h1 : responses.total = 800)
  (h2 : responses.yes_only_M = 170)
  (h3 : responses.no_both = 230) :
  students_yes_R responses = 400 := by
  sorry

#eval students_yes_R ⟨800, 170, 230⟩

end NUMINAMATH_CALUDE_students_yes_R_is_400_l1427_142732


namespace NUMINAMATH_CALUDE_simplify_expression_l1427_142772

theorem simplify_expression (r : ℝ) : 120*r - 38*r + 25*r = 107*r := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1427_142772


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1427_142791

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    (pointOnLine ⟨2, 3⟩ l1 ∧ equalIntercepts l1) ∧
    (pointOnLine ⟨2, 3⟩ l2 ∧ equalIntercepts l2) ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -5) ∨ (l2.a = 3 ∧ l2.b = -2 ∧ l2.c = 0)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1427_142791


namespace NUMINAMATH_CALUDE_tangent_line_intercept_l1427_142797

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * exp x - 3 * x + 1

theorem tangent_line_intercept (a : ℝ) :
  let f' := fun x => a * exp x - 3
  (f' 0 = 1) →
  (f a 0 = 5) →
  (∃ b, ∀ x, f a 0 + f' 0 * x = x + b) →
  ∃ b, f a 0 + f' 0 * 0 = 0 + b ∧ b = 5 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_intercept_l1427_142797


namespace NUMINAMATH_CALUDE_f_composition_value_l1427_142720

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_value : f (f (f (-1))) = Real.pi + 1 := by sorry

end NUMINAMATH_CALUDE_f_composition_value_l1427_142720


namespace NUMINAMATH_CALUDE_calculate_expression_l1427_142747

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1427_142747


namespace NUMINAMATH_CALUDE_set_a_constraint_l1427_142769

theorem set_a_constraint (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 2*x + a ≥ 0}
  1 ∉ A → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_set_a_constraint_l1427_142769


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l1427_142777

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- The sum of two repeating decimals 0.̅6 and 0.̅2 minus 0.̅4 equals 4/9 -/
theorem repeating_decimal_sum_diff :
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l1427_142777


namespace NUMINAMATH_CALUDE_current_speed_l1427_142749

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 16)
  (h2 : speed_against_current = 9.6) :
  ∃ (current_speed : ℝ), current_speed = 3.2 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by
  sorry

#check current_speed

end NUMINAMATH_CALUDE_current_speed_l1427_142749


namespace NUMINAMATH_CALUDE_ellipse_and_circle_properties_l1427_142746

-- Define the points and shapes
structure Point where
  x : ℝ
  y : ℝ

def F : Point := ⟨0, -1⟩
def A : Point := ⟨0, 2⟩
def O : Point := ⟨0, 0⟩

structure Circle where
  center : Point
  radius : ℝ

structure Ellipse where
  center : Point
  a : ℝ
  b : ℝ

def Line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Define the problem conditions
def is_on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

def is_tangent_to_line (c : Circle) (l : ℝ → ℝ) : Prop :=
  ∃ p : Point, is_on_circle p c ∧ p.y = l p.x ∧
  ∀ q : Point, q ≠ p → is_on_circle q c → q.y ≠ l q.x

def is_on_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.b^2 + (p.y - e.center.y)^2 / e.a^2 = 1

def is_focus_of_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 + (p.y - e.center.y)^2 = e.a^2 - e.b^2

-- Define the theorem
theorem ellipse_and_circle_properties :
  ∀ (Q : Circle) (N : Ellipse) (M : ℝ → ℝ) (m : ℝ → ℝ → ℝ) (Z : ℝ → ℝ),
  (∀ x : ℝ, is_on_circle F Q) →
  (is_tangent_to_line Q (Line 0 1)) →
  (N.center = O) →
  (is_focus_of_ellipse F N) →
  (is_on_ellipse A N) →
  (∀ k : ℝ, ∃ B C D E : Point,
    is_on_ellipse B N ∧ is_on_ellipse C N ∧
    B.y = m k B.x ∧ C.y = m k C.x ∧
    D.x^2 = -4 * D.y ∧ E.x^2 = -4 * E.y ∧
    D.y = m k D.x ∧ E.y = m k E.x) →
  (∀ x : ℝ, M x = -x^2 / 4) →
  (N.a = 2 ∧ N.b = Real.sqrt 3) →
  (∀ k : ℝ, 9 ≤ Z k ∧ Z k < 12) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_properties_l1427_142746


namespace NUMINAMATH_CALUDE_helen_laundry_time_l1427_142750

def wash_silk : ℕ := 30
def wash_wool : ℕ := 45
def wash_cashmere : ℕ := 15

def dry_silk : ℕ := 20
def dry_wool : ℕ := 30
def dry_cashmere : ℕ := 10

def fold_silk : ℕ := 10
def fold_wool : ℕ := 15
def fold_cashmere : ℕ := 5

def iron_silk : ℕ := 5
def iron_wool : ℕ := 20
def iron_cashmere : ℕ := 10

def days_in_leap_year : ℕ := 366
def days_in_regular_year : ℕ := 365
def num_regular_years : ℕ := 3
def days_between_washes : ℕ := 28

def total_days : ℕ := days_in_leap_year + num_regular_years * days_in_regular_year

def num_wash_sessions : ℕ := total_days / days_between_washes

def time_per_session : ℕ := 
  (wash_silk + dry_silk + fold_silk + iron_silk) +
  (wash_wool + dry_wool + fold_wool + iron_wool) +
  (wash_cashmere + dry_cashmere + fold_cashmere + iron_cashmere)

theorem helen_laundry_time : 
  num_wash_sessions * time_per_session = 11180 :=
sorry

end NUMINAMATH_CALUDE_helen_laundry_time_l1427_142750


namespace NUMINAMATH_CALUDE_final_nickel_count_is_45_l1427_142709

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- Represents a transaction of coins -/
structure CoinTransaction where
  nickels : ℤ
  dimes : ℤ
  quarters : ℤ
  half_dollars : ℤ

def initial_coins : CoinCount := {
  pennies := 45,
  nickels := 29,
  dimes := 16,
  quarters := 8,
  half_dollars := 4
}

def dad_gives : CoinTransaction := {
  nickels := 24,
  dimes := 15,
  quarters := 12,
  half_dollars := 6
}

def dad_takes : CoinTransaction := {
  nickels := -13,
  dimes := -9,
  quarters := -5,
  half_dollars := 0
}

def additional_percentage : ℚ := 20 / 100

/-- Applies a transaction to the coin count -/
def apply_transaction (coins : CoinCount) (transaction : CoinTransaction) : CoinCount :=
  { coins with
    nickels := (coins.nickels : ℤ) + transaction.nickels |>.toNat,
    dimes := (coins.dimes : ℤ) + transaction.dimes |>.toNat,
    quarters := (coins.quarters : ℤ) + transaction.quarters |>.toNat,
    half_dollars := (coins.half_dollars : ℤ) + transaction.half_dollars |>.toNat
  }

/-- Calculates the final number of nickels Sam has -/
def final_nickel_count : ℕ :=
  let after_first_transaction := apply_transaction initial_coins dad_gives
  let after_second_transaction := apply_transaction after_first_transaction dad_takes
  let additional_nickels := (dad_gives.nickels : ℚ) * additional_percentage |>.ceil.toNat
  after_second_transaction.nickels + additional_nickels

theorem final_nickel_count_is_45 : final_nickel_count = 45 := by
  sorry

end NUMINAMATH_CALUDE_final_nickel_count_is_45_l1427_142709


namespace NUMINAMATH_CALUDE_min_c_value_l1427_142735

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c) (h_sum : a + b = c)
  (h_unique_solution : ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1501 ∧ ∃ (a' b' c' : ℕ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' < b' ∧ b' < c' ∧ a' + b' = c' ∧ c' = 1501 ∧
    ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a'| + |x - b'| + |x - c'| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l1427_142735


namespace NUMINAMATH_CALUDE_sum_of_angles_of_roots_l1427_142704

def isRoot (z : ℂ) : Prop := z^5 = -Complex.I

def angleInRange (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi

theorem sum_of_angles_of_roots (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) 
  (h₁ : isRoot (Complex.exp (Complex.I * θ₁)))
  (h₂ : isRoot (Complex.exp (Complex.I * θ₂)))
  (h₃ : isRoot (Complex.exp (Complex.I * θ₃)))
  (h₄ : isRoot (Complex.exp (Complex.I * θ₄)))
  (h₅ : isRoot (Complex.exp (Complex.I * θ₅)))
  (r₁ : angleInRange θ₁)
  (r₂ : angleInRange θ₂)
  (r₃ : angleInRange θ₃)
  (r₄ : angleInRange θ₄)
  (r₅ : angleInRange θ₅) :
  θ₁ + θ₂ + θ₃ + θ₄ + θ₅ = 5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_of_roots_l1427_142704


namespace NUMINAMATH_CALUDE_number_of_houses_street_houses_l1427_142771

/-- Given a street with clotheslines, prove the number of houses -/
theorem number_of_houses (children : ℕ) (adults : ℕ) (child_items : ℕ) (adult_items : ℕ) 
  (items_per_line : ℕ) (lines_per_house : ℕ) : ℕ :=
  let total_items := children * child_items + adults * adult_items
  let total_lines := total_items / items_per_line
  total_lines / lines_per_house

/-- Prove that there are 26 houses on the street -/
theorem street_houses : 
  number_of_houses 11 20 4 3 2 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_of_houses_street_houses_l1427_142771


namespace NUMINAMATH_CALUDE_smallest_lpm_l1427_142766

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number with repeating digits -/
def TwoDigitRepeating (d : Digit) := 10 * d.val + d.val

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds.val + 10 * n.tens.val + n.ones.val

/-- Checks if a natural number is a valid result of the multiplication -/
def isValidResult (l : Digit) (result : ThreeDigitNumber) : Prop :=
  (TwoDigitRepeating l) * l.val = result.toNat ∧
  result.hundreds = l

theorem smallest_lpm :
  ∃ (result : ThreeDigitNumber),
    (∃ (l : Digit), isValidResult l result) ∧
    (∀ (other : ThreeDigitNumber),
      (∃ (l : Digit), isValidResult l other) →
      result.toNat ≤ other.toNat) ∧
    result.toNat = 275 := by
  sorry

end NUMINAMATH_CALUDE_smallest_lpm_l1427_142766


namespace NUMINAMATH_CALUDE_angle_DAC_measure_l1427_142719

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the point D
def D (t : Triangle) : Point := sorry

-- Define the angles
def angle_BAC (t : Triangle) : ℝ := sorry
def angle_ABC (t : Triangle) : ℝ := sorry
def angle_DAC (t : Triangle) : ℝ := sorry

-- Define the lengths
def length_DA (t : Triangle) : ℝ := sorry
def length_CB (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_DAC_measure (t : Triangle) 
  (h1 : length_DA t = length_CB t)
  (h2 : angle_BAC t = 70)
  (h3 : angle_ABC t = 55) :
  angle_DAC t = 100 := by sorry

end NUMINAMATH_CALUDE_angle_DAC_measure_l1427_142719


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1427_142775

theorem log_sum_equals_two : 2 * Real.log 63 + Real.log 64 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1427_142775


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1427_142796

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  (∀ x, 2*x^2 + b*x + a < 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1427_142796


namespace NUMINAMATH_CALUDE_cyclic_fraction_theorem_l1427_142783

theorem cyclic_fraction_theorem (x y z k : ℝ) :
  (x / (y + z) = k ∧ y / (z + x) = k ∧ z / (x + y) = k) →
  (k = 1/2 ∨ k = -1) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_fraction_theorem_l1427_142783


namespace NUMINAMATH_CALUDE_simplify_expression_l1427_142740

theorem simplify_expression (x y : ℝ) : (5 - 2*x) - (8 - 6*x + 3*y) = -3 + 4*x - 3*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1427_142740


namespace NUMINAMATH_CALUDE_machine_production_time_l1427_142737

/-- Proves that a machine producing 360 items in 2 hours takes 1/3 minute to produce one item. -/
theorem machine_production_time 
  (items_produced : ℕ) 
  (production_hours : ℕ) 
  (minutes_per_hour : ℕ) 
  (h1 : items_produced = 360)
  (h2 : production_hours = 2)
  (h3 : minutes_per_hour = 60) :
  (production_hours * minutes_per_hour) / items_produced = 1 / 3 := by
  sorry

#check machine_production_time

end NUMINAMATH_CALUDE_machine_production_time_l1427_142737


namespace NUMINAMATH_CALUDE_rope_length_proof_l1427_142717

/-- The length of the rope in meters -/
def rope_length : ℝ := 1.15

/-- The fraction of the rope that was used -/
def used_fraction : ℝ := 0.4

/-- The remaining length of the rope in meters -/
def remaining_length : ℝ := 0.69

theorem rope_length_proof : 
  rope_length * (1 - used_fraction) = remaining_length :=
by sorry

end NUMINAMATH_CALUDE_rope_length_proof_l1427_142717


namespace NUMINAMATH_CALUDE_distance_between_points_l1427_142761

theorem distance_between_points (m : ℝ) :
  let P : ℝ × ℝ × ℝ := (m, 0, 0)
  let P₁ : ℝ × ℝ × ℝ := (4, 1, 2)
  (m - 4)^2 + 1^2 + 2^2 = 30 → m = 9 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1427_142761


namespace NUMINAMATH_CALUDE_max_value_of_function_l1427_142738

theorem max_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 5/4) :
  x * (5 - 4*x) ≤ 25/16 ∧ ∃ x₀, x₀ * (5 - 4*x₀) = 25/16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1427_142738


namespace NUMINAMATH_CALUDE_closest_to_fraction_l1427_142792

def options : List ℝ := [4000, 5000, 6000, 7000, 8000]

theorem closest_to_fraction (x : ℝ) (h : x = 510 / 0.125) :
  4000 ∈ options ∧ ∀ y ∈ options, |x - 4000| ≤ |x - y| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l1427_142792


namespace NUMINAMATH_CALUDE_exist_non_congruent_polyhedra_with_same_views_l1427_142782

/-- Represents a polyhedron --/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))

/-- Represents a 2D view of a polyhedron --/
structure View where
  points : Set (Fin 2 → ℝ)
  edges : Set (Fin 2 → ℝ) × (Fin 2 → ℝ)

/-- Checks if two polyhedra are congruent --/
def are_congruent (p1 p2 : Polyhedron) : Prop :=
  sorry

/-- Gets the front view of a polyhedron --/
def front_view (p : Polyhedron) : View :=
  sorry

/-- Gets the top view of a polyhedron --/
def top_view (p : Polyhedron) : View :=
  sorry

/-- Checks if a view has an internal intersection point at the center of the square --/
def has_center_intersection (v : View) : Prop :=
  sorry

/-- Checks if all segments of the squares in a view are visible edges --/
def all_segments_visible (v : View) : Prop :=
  sorry

/-- Checks if a view has no hidden edges --/
def no_hidden_edges (v : View) : Prop :=
  sorry

/-- The main theorem stating the existence of two non-congruent polyhedra with the given properties --/
theorem exist_non_congruent_polyhedra_with_same_views : 
  ∃ (p1 p2 : Polyhedron), 
    front_view p1 = front_view p2 ∧
    top_view p1 = top_view p2 ∧
    has_center_intersection (front_view p1) ∧
    has_center_intersection (top_view p1) ∧
    all_segments_visible (front_view p1) ∧
    all_segments_visible (top_view p1) ∧
    no_hidden_edges (front_view p1) ∧
    no_hidden_edges (top_view p1) ∧
    ¬(are_congruent p1 p2) :=
  sorry

end NUMINAMATH_CALUDE_exist_non_congruent_polyhedra_with_same_views_l1427_142782


namespace NUMINAMATH_CALUDE_average_salary_proof_l1427_142757

theorem average_salary_proof (salary_a salary_b salary_c salary_d salary_e : ℕ)
  (h1 : salary_a = 10000)
  (h2 : salary_b = 5000)
  (h3 : salary_c = 11000)
  (h4 : salary_d = 7000)
  (h5 : salary_e = 9000) :
  (salary_a + salary_b + salary_c + salary_d + salary_e) / 5 = 8600 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_proof_l1427_142757


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1427_142799

/-- The area of a square with adjacent points (1,2) and (4,6) on a Cartesian coordinate plane is 25. -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1427_142799


namespace NUMINAMATH_CALUDE_fraction_product_equality_l1427_142784

theorem fraction_product_equality : (3 / 4) * (5 / 9) * (8 / 13) * (3 / 7) = 10 / 91 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l1427_142784


namespace NUMINAMATH_CALUDE_children_outnumber_parents_l1427_142762

/-- Represents a family unit in the apartment block -/
structure Family where
  parents : Nat
  boys : Nat
  girls : Nat

/-- Represents the apartment block -/
structure ApartmentBlock where
  families : List Family

/-- Every couple has at least one child -/
axiom at_least_one_child (f : Family) : f.boys + f.girls ≥ 1

/-- Every child has exactly two parents -/
axiom two_parents (f : Family) : f.parents = 2

/-- Every little boy has a sister -/
axiom boys_have_sisters (f : Family) : f.boys > 0 → f.girls > 0

/-- Among the children, there are more boys than girls -/
axiom more_boys_than_girls (ab : ApartmentBlock) :
  (ab.families.map (λ f => f.boys)).sum > (ab.families.map (λ f => f.girls)).sum

/-- There are no grandparents living in the building -/
axiom no_grandparents (ab : ApartmentBlock) :
  (ab.families.map (λ f => f.parents)).sum = 2 * ab.families.length

theorem children_outnumber_parents (ab : ApartmentBlock) :
  (ab.families.map (λ f => f.boys + f.girls)).sum > (ab.families.map (λ f => f.parents)).sum :=
sorry

end NUMINAMATH_CALUDE_children_outnumber_parents_l1427_142762


namespace NUMINAMATH_CALUDE_complex_product_sum_l1427_142718

theorem complex_product_sum (a b : ℝ) : 
  (1 + Complex.I) * (2 + Complex.I) = Complex.mk a b → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_sum_l1427_142718


namespace NUMINAMATH_CALUDE_parallelogram_properties_l1427_142729

def A : ℂ := Complex.I
def B : ℂ := 1
def C : ℂ := 4 + 2 * Complex.I

theorem parallelogram_properties :
  let D : ℂ := 4 + 3 * Complex.I
  let diagonal_BD : ℂ := D - B
  (A + C = B + D) ∧ 
  (Complex.abs diagonal_BD = 3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_properties_l1427_142729


namespace NUMINAMATH_CALUDE_feb_2_is_tuesday_l1427_142705

-- Define the days of the week
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define a function to get the day of the week given a number of days before Sunday
def daysBefore (n : Nat) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Saturday
  | 2 => DayOfWeek.Friday
  | 3 => DayOfWeek.Thursday
  | 4 => DayOfWeek.Wednesday
  | 5 => DayOfWeek.Tuesday
  | _ => DayOfWeek.Monday

-- Theorem statement
theorem feb_2_is_tuesday (h : DayOfWeek.Sunday = daysBefore 0) :
  DayOfWeek.Tuesday = daysBefore 12 := by
  sorry


end NUMINAMATH_CALUDE_feb_2_is_tuesday_l1427_142705


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1427_142712

theorem sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / b + b / c + c / a + b / a + c / b + a / c = 9) :
  a / b + b / c + c / a = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1427_142712


namespace NUMINAMATH_CALUDE_wheel_probability_l1427_142741

theorem wheel_probability (p_D p_E p_F : ℚ) : 
  p_D = 3/8 → p_E = 1/4 → p_D + p_E + p_F = 1 → p_F = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l1427_142741


namespace NUMINAMATH_CALUDE_lottery_probability_l1427_142785

theorem lottery_probability : 
  let powerball_count : ℕ := 30
  let luckyball_count : ℕ := 49
  let luckyball_picks : ℕ := 6
  let powerball_prob : ℚ := 1 / powerball_count
  let luckyball_prob : ℚ := 1 / (Nat.choose luckyball_count luckyball_picks)
  powerball_prob * luckyball_prob = 1 / 419512480 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l1427_142785


namespace NUMINAMATH_CALUDE_jellybean_ratio_l1427_142722

/-- Proves that the ratio of jellybeans Shannon refilled to the total taken out
    by Samantha and Shelby is 1/2, given the initial count, the amounts taken
    by Samantha and Shelby, and the final count. -/
theorem jellybean_ratio (initial : ℕ) (samantha_taken : ℕ) (shelby_taken : ℕ) (final : ℕ)
  (h1 : initial = 90)
  (h2 : samantha_taken = 24)
  (h3 : shelby_taken = 12)
  (h4 : final = 72) :
  (final - (initial - (samantha_taken + shelby_taken))) / (samantha_taken + shelby_taken) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_ratio_l1427_142722


namespace NUMINAMATH_CALUDE_solve_for_c_l1427_142745

theorem solve_for_c (c d : ℚ) 
  (eq1 : (c - 34) / 2 = (2 * d - 8) / 7)
  (eq2 : d = c + 9) : 
  c = 86 := by
sorry

end NUMINAMATH_CALUDE_solve_for_c_l1427_142745


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l1427_142778

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -5 ∨ 1 < x} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (11/2)*t} = {t : ℝ | 1/2 ≤ t ∧ t ≤ 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l1427_142778


namespace NUMINAMATH_CALUDE_corveus_sleep_lack_l1427_142728

/-- The number of hours Corveus lacks sleep in a week -/
def sleep_lack_per_week (actual_sleep : ℕ) (recommended_sleep : ℕ) (days_in_week : ℕ) : ℕ :=
  (recommended_sleep - actual_sleep) * days_in_week

/-- Theorem stating that Corveus lacks 14 hours of sleep in a week -/
theorem corveus_sleep_lack :
  sleep_lack_per_week 4 6 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_corveus_sleep_lack_l1427_142728


namespace NUMINAMATH_CALUDE_car_banker_speed_ratio_l1427_142787

/-- The ratio of car speed to banker speed given specific timing conditions -/
theorem car_banker_speed_ratio :
  ∀ (T : ℝ) (Vb Vc : ℝ) (d : ℝ),
    Vb > 0 →
    Vc > 0 →
    d > 0 →
    (Vb * 60 = Vc * 5) →
    (Vc / Vb = 12) := by
  sorry

end NUMINAMATH_CALUDE_car_banker_speed_ratio_l1427_142787
