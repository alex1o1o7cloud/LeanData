import Mathlib

namespace NUMINAMATH_CALUDE_pages_read_on_thursday_l3967_396793

theorem pages_read_on_thursday (wednesday_pages friday_pages total_pages : ℕ) 
  (h1 : wednesday_pages = 18)
  (h2 : friday_pages = 23)
  (h3 : total_pages = 60) :
  total_pages - (wednesday_pages + friday_pages) = 19 := by
sorry

end NUMINAMATH_CALUDE_pages_read_on_thursday_l3967_396793


namespace NUMINAMATH_CALUDE_triangle_properties_l3967_396750

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  c * Real.sin B + (a + c^2 / a - b^2 / a) * Real.sin C = 2 * c * Real.sin A →
  (a * b * Real.sin C) / 2 = Real.sqrt 3 →
  a * Real.sin A / 2 = Real.sqrt 3 →
  b * Real.sin B / 2 = Real.sqrt 3 →
  c * Real.sin C / 2 = Real.sqrt 3 →
  C = π / 3 ∧ Real.cos (2 * A) - 2 * Real.sin B ^ 2 + 1 = -1 / 6 := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l3967_396750


namespace NUMINAMATH_CALUDE_runner_distance_at_click_l3967_396753

/-- The time in seconds for which the camera timer is set -/
def timer_setting : ℝ := 45

/-- The runner's speed in yards per second -/
def runner_speed : ℝ := 10

/-- The speed of sound in feet per second without headwind -/
def sound_speed : ℝ := 1100

/-- The reduction factor of sound speed due to headwind -/
def sound_speed_reduction : ℝ := 0.1

/-- The effective speed of sound in feet per second with headwind -/
def effective_sound_speed : ℝ := sound_speed * (1 - sound_speed_reduction)

/-- The distance the runner travels in feet at time t -/
def runner_distance (t : ℝ) : ℝ := runner_speed * 3 * t

/-- The distance sound travels in feet at time t after the camera click -/
def sound_distance (t : ℝ) : ℝ := effective_sound_speed * (t - timer_setting)

/-- The time when the runner hears the camera click -/
noncomputable def hearing_time : ℝ := 
  (effective_sound_speed * timer_setting) / (effective_sound_speed - runner_speed * 3)

theorem runner_distance_at_click : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |runner_distance hearing_time / 3 - 464| < ε :=
sorry

end NUMINAMATH_CALUDE_runner_distance_at_click_l3967_396753


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l3967_396703

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_second_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 3 = 2) :
  a 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l3967_396703


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3967_396796

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 5 * n % 26 = 2024 % 26 ∧ ∀ (m : ℕ), m > 0 ∧ m < n → 5 * m % 26 ≠ 2024 % 26 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3967_396796


namespace NUMINAMATH_CALUDE_student_average_age_l3967_396725

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (avg_increase : ℝ) : 
  num_students = 22 → 
  teacher_age = 44 → 
  avg_increase = 1 → 
  (((num_students : ℝ) * x + teacher_age) / (num_students + 1) = x + avg_increase) → 
  x = 21 :=
by sorry

end NUMINAMATH_CALUDE_student_average_age_l3967_396725


namespace NUMINAMATH_CALUDE_base_for_888_l3967_396712

theorem base_for_888 :
  ∃! b : ℕ,
    (b > 1) ∧
    (∃ a B : ℕ,
      a ≠ B ∧
      a < b ∧
      B < b ∧
      888 = a * b^3 + a * b^2 + B * b + B) ∧
    (b^3 ≤ 888) ∧
    (888 < b^4) :=
by sorry

end NUMINAMATH_CALUDE_base_for_888_l3967_396712


namespace NUMINAMATH_CALUDE_problem_solution_l3967_396761

theorem problem_solution (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : 
  x^2 + y^2 = 6 ∧ (x - y)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3967_396761


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3967_396724

theorem quadratic_inequality (x : ℝ) : 3 * x^2 - 8 * x - 3 > 0 ↔ x < -1/3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3967_396724


namespace NUMINAMATH_CALUDE_abc_fraction_l3967_396731

theorem abc_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a * b / (a + b) = 3)
  (hbc : b * c / (b + c) = 4)
  (hca : c * a / (c + a) = 5) :
  a * b * c / (a * b + b * c + c * a) = 120 / 47 := by
sorry

end NUMINAMATH_CALUDE_abc_fraction_l3967_396731


namespace NUMINAMATH_CALUDE_anthony_pencil_count_l3967_396794

/-- Anthony's initial pencil count -/
def initial_pencils : ℝ := 56.0

/-- Number of pencils Anthony gives away -/
def pencils_given : ℝ := 9.0

/-- Anthony's final pencil count -/
def final_pencils : ℝ := 47.0

theorem anthony_pencil_count : initial_pencils - pencils_given = final_pencils := by
  sorry

end NUMINAMATH_CALUDE_anthony_pencil_count_l3967_396794


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l3967_396737

-- Problem 1
theorem simplify_fraction_1 (x y : ℝ) (h : y ≠ 0) :
  (x^2 - 1) / y / ((x + 1) / y^2) = y * (x - 1) := by sorry

-- Problem 2
theorem simplify_fraction_2 (m n : ℝ) (h1 : m ≠ n) (h2 : m ≠ -n) :
  m / (m + n) + n / (m - n) - 2 * m^2 / (m^2 - n^2) = -1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l3967_396737


namespace NUMINAMATH_CALUDE_FMF_better_than_MFM_l3967_396756

/-- Represents the probability of winning a tennis match against a parent. -/
structure ParentProbability where
  /-- The probability of winning against the parent. -/
  prob : ℝ
  /-- The probability is between 0 and 1. -/
  prob_between_zero_and_one : 0 ≤ prob ∧ prob ≤ 1

/-- Calculates the probability of winning in a Father-Mother-Father (FMF) sequence. -/
def prob_win_FMF (p q : ParentProbability) : ℝ :=
  2 * p.prob * q.prob - p.prob * q.prob^2

/-- Calculates the probability of winning in a Mother-Father-Mother (MFM) sequence. -/
def prob_win_MFM (p q : ParentProbability) : ℝ :=
  2 * p.prob * q.prob - p.prob^2 * q.prob

/-- 
Theorem: The probability of winning in the Father-Mother-Father (FMF) sequence
is higher than the probability of winning in the Mother-Father-Mother (MFM) sequence,
given that the probability of winning against the father is less than
the probability of winning against the mother.
-/
theorem FMF_better_than_MFM (p q : ParentProbability) 
  (h : p.prob < q.prob) : prob_win_FMF p q > prob_win_MFM p q := by
  sorry


end NUMINAMATH_CALUDE_FMF_better_than_MFM_l3967_396756


namespace NUMINAMATH_CALUDE_interview_probability_l3967_396765

/-- The number of students enrolled in at least one foreign language class -/
def total_students : ℕ := 25

/-- The number of students in the French class -/
def french_students : ℕ := 18

/-- The number of students in the Spanish class -/
def spanish_students : ℕ := 21

/-- The number of students to be chosen -/
def chosen_students : ℕ := 2

/-- The probability of selecting at least one student from French class
    and at least one student from Spanish class -/
def probability_both_classes : ℚ := 91 / 100

theorem interview_probability :
  let students_in_both := french_students + spanish_students - total_students
  let only_french := french_students - students_in_both
  let only_spanish := spanish_students - students_in_both
  probability_both_classes = 1 - (Nat.choose only_french chosen_students + Nat.choose only_spanish chosen_students : ℚ) / Nat.choose total_students chosen_students :=
by sorry

end NUMINAMATH_CALUDE_interview_probability_l3967_396765


namespace NUMINAMATH_CALUDE_equation_solution_l3967_396700

theorem equation_solution :
  ∃ x : ℝ, (x^2 - x + 2) / (x - 1) = x + 3 ∧ x = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3967_396700


namespace NUMINAMATH_CALUDE_frank_bought_five_chocolates_l3967_396787

-- Define the cost of items
def chocolate_cost : ℕ := 2
def chips_cost : ℕ := 3

-- Define the number of bags of chips
def chips_count : ℕ := 2

-- Define the total amount spent
def total_spent : ℕ := 16

-- Define the function to calculate the number of chocolate bars
def chocolate_bars : ℕ → Prop
  | n => chocolate_cost * n + chips_cost * chips_count = total_spent

-- Theorem statement
theorem frank_bought_five_chocolates : 
  ∃ (n : ℕ), chocolate_bars n ∧ n = 5 := by sorry

end NUMINAMATH_CALUDE_frank_bought_five_chocolates_l3967_396787


namespace NUMINAMATH_CALUDE_moses_esther_difference_l3967_396780

theorem moses_esther_difference (total : ℝ) (moses_percentage : ℝ) : 
  total = 50 →
  moses_percentage = 0.4 →
  let moses_share := moses_percentage * total
  let remainder := total - moses_share
  let esther_share := remainder / 2
  moses_share - esther_share = 5 := by
  sorry

end NUMINAMATH_CALUDE_moses_esther_difference_l3967_396780


namespace NUMINAMATH_CALUDE_final_savings_calculation_correct_l3967_396734

/-- Calculates the final savings given initial savings, monthly income, monthly expenses, and number of months. -/
def calculate_final_savings (initial_savings monthly_income monthly_expenses : ℕ) (num_months : ℕ) : ℕ :=
  initial_savings + num_months * monthly_income - num_months * monthly_expenses

/-- Theorem stating that the final savings calculation is correct for the given problem. -/
theorem final_savings_calculation_correct :
  let initial_savings : ℕ := 849400
  let monthly_income : ℕ := 45000 + 35000 + 7000 + 10000 + 13000
  let monthly_expenses : ℕ := 30000 + 10000 + 5000 + 4500 + 9000
  let num_months : ℕ := 5
  calculate_final_savings initial_savings monthly_income monthly_expenses num_months = 1106900 := by
  sorry

#eval calculate_final_savings 849400 110000 58500 5

end NUMINAMATH_CALUDE_final_savings_calculation_correct_l3967_396734


namespace NUMINAMATH_CALUDE_seventeen_factorial_minus_fifteen_factorial_prime_divisors_l3967_396717

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of prime divisors of a natural number -/
def numPrimeDivisors (n : ℕ) : ℕ := (Nat.factors n).length

/-- The main theorem -/
theorem seventeen_factorial_minus_fifteen_factorial_prime_divisors :
  numPrimeDivisors (factorial 17 - factorial 15) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_factorial_minus_fifteen_factorial_prime_divisors_l3967_396717


namespace NUMINAMATH_CALUDE_class_ratio_theorem_l3967_396740

-- Define the class structure
structure ClassComposition where
  total_students : ℕ
  girls : ℕ
  boys : ℕ

-- Define the condition given in the problem
def satisfies_condition (c : ClassComposition) : Prop :=
  2 * c.girls * 5 = 3 * c.total_students

-- Define the property we want to prove
def has_correct_ratio (c : ClassComposition) : Prop :=
  7 * c.girls = 3 * c.boys

-- The theorem to prove
theorem class_ratio_theorem (c : ClassComposition) 
  (h1 : c.total_students = c.girls + c.boys)
  (h2 : satisfies_condition c) : 
  has_correct_ratio c := by
  sorry

#check class_ratio_theorem

end NUMINAMATH_CALUDE_class_ratio_theorem_l3967_396740


namespace NUMINAMATH_CALUDE_foma_cannot_guarantee_win_l3967_396790

/-- Represents a player in the coin game -/
inductive Player : Type
| Foma : Player
| Yerema : Player

/-- Represents the state of the game -/
structure GameState :=
(coins : List Nat)  -- List of remaining coin values
(foma_coins : Nat)  -- Total value of Foma's coins
(yerema_coins : Nat)  -- Total value of Yerema's coins
(last_selector : Player)  -- Player who made the last selection

/-- Function to determine the next selector based on current game state -/
def next_selector (state : GameState) : Player :=
  if state.foma_coins > state.yerema_coins then Player.Foma
  else if state.yerema_coins > state.foma_coins then Player.Yerema
  else state.last_selector

/-- Theorem stating that Foma cannot guarantee winning -/
theorem foma_cannot_guarantee_win :
  ∀ (initial_state : GameState),
    initial_state.coins = List.range 25
    → initial_state.foma_coins = 0
    → initial_state.yerema_coins = 0
    → initial_state.last_selector = Player.Foma
    → ¬ (∀ (strategy : GameState → Nat),
         ∃ (final_state : GameState),
           final_state.coins = []
           ∧ final_state.foma_coins > final_state.yerema_coins) :=
sorry

end NUMINAMATH_CALUDE_foma_cannot_guarantee_win_l3967_396790


namespace NUMINAMATH_CALUDE_xy_squared_l3967_396777

theorem xy_squared (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y - x - y = 2) : 
  x^2 * y^2 = 1/4 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_l3967_396777


namespace NUMINAMATH_CALUDE_gcf_78_104_l3967_396732

theorem gcf_78_104 : Nat.gcd 78 104 = 26 := by
  sorry

end NUMINAMATH_CALUDE_gcf_78_104_l3967_396732


namespace NUMINAMATH_CALUDE_intersection_count_is_four_l3967_396707

/-- The number of distinct intersection points between two curves -/
def intersection_count (f g : ℝ × ℝ → ℝ) : ℕ :=
  sorry

/-- First equation: (x + 2y - 6)(3x - y + 4) = 0 -/
def f (p : ℝ × ℝ) : ℝ :=
  (p.1 + 2*p.2 - 6) * (3*p.1 - p.2 + 4)

/-- Second equation: (2x - 3y + 1)(x + y - 2) = 0 -/
def g (p : ℝ × ℝ) : ℝ :=
  (2*p.1 - 3*p.2 + 1) * (p.1 + p.2 - 2)

/-- Theorem stating that the two curves intersect at exactly 4 points -/
theorem intersection_count_is_four :
  intersection_count f g = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_is_four_l3967_396707


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3967_396768

-- Define the sets A and B
def A : Set ℕ := {2, 1, 3}
def B : Set ℕ := {2, 3, 5}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3967_396768


namespace NUMINAMATH_CALUDE_carlos_jogging_distance_l3967_396762

/-- Calculates the distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Given Carlos' jogging speed and time, prove the distance he jogged -/
theorem carlos_jogging_distance :
  let jogging_speed : ℝ := 4
  let jogging_time : ℝ := 2
  distance jogging_speed jogging_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_carlos_jogging_distance_l3967_396762


namespace NUMINAMATH_CALUDE_value_of_b_plus_a_l3967_396755

theorem value_of_b_plus_a (a b : ℝ) : 
  (abs a = 8) → 
  (abs b = 2) → 
  (abs (a - b) = b - a) → 
  ((b + a = -6) ∨ (b + a = -10)) := by
sorry

end NUMINAMATH_CALUDE_value_of_b_plus_a_l3967_396755


namespace NUMINAMATH_CALUDE_y_days_to_finish_work_l3967_396747

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 18

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 10

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 6

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

/-- Theorem stating that given the conditions, y needs 15 days to finish the work alone -/
theorem y_days_to_finish_work : 
  (1 / x_days) * x_remaining = 1 - (y_worked / y_days) := by sorry

end NUMINAMATH_CALUDE_y_days_to_finish_work_l3967_396747


namespace NUMINAMATH_CALUDE_set_A_determination_l3967_396704

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}

theorem set_A_determination (A : Set ℕ) (h : (U \ A) = {2, 3}) : A = {1, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_A_determination_l3967_396704


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3967_396748

theorem complex_equation_solution (z : ℂ) :
  (2 - 3*I)*z = 5 - I → z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3967_396748


namespace NUMINAMATH_CALUDE_max_k_value_l3967_396769

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ (3/2) ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 = (3/2)^2 * ((x^2 / y^2) + (y^2 / x^2)) + (3/2) * ((x / y) + (y / x)) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3967_396769


namespace NUMINAMATH_CALUDE_ounces_per_pound_l3967_396709

def cat_food_bags : ℕ := 2
def cat_food_weight : ℕ := 3
def dog_food_bags : ℕ := 2
def dog_food_extra_weight : ℕ := 2
def total_ounces : ℕ := 256

theorem ounces_per_pound :
  ∃ (x : ℕ),
    x * (cat_food_bags * cat_food_weight + 
         dog_food_bags * (cat_food_weight + dog_food_extra_weight)) = total_ounces ∧
    x = 16 := by
  sorry

end NUMINAMATH_CALUDE_ounces_per_pound_l3967_396709


namespace NUMINAMATH_CALUDE_max_product_price_for_given_conditions_l3967_396744

/-- Represents a company's product line -/
structure ProductLine where
  numProducts : ℕ
  averagePrice : ℝ
  minPrice : ℝ
  numLowPriced : ℕ
  lowPriceThreshold : ℝ

/-- The greatest possible selling price of the most expensive product -/
def maxProductPrice (pl : ProductLine) : ℝ :=
  sorry

/-- Theorem stating the maximum product price for the given conditions -/
theorem max_product_price_for_given_conditions :
  let pl : ProductLine := {
    numProducts := 25,
    averagePrice := 1200,
    minPrice := 400,
    numLowPriced := 10,
    lowPriceThreshold := 1000
  }
  maxProductPrice pl = 12000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_price_for_given_conditions_l3967_396744


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l3967_396795

theorem green_shirt_pairs (blue_students : ℕ) (green_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 65 →
  green_students = 95 →
  total_students = 160 →
  total_pairs = 80 →
  blue_blue_pairs = 25 →
  blue_students + green_students = total_students →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l3967_396795


namespace NUMINAMATH_CALUDE_resort_tips_multiple_l3967_396714

theorem resort_tips_multiple (total_months : ℕ) (august_ratio : ℝ) : 
  total_months = 7 → 
  august_ratio = 0.25 → 
  (7 * august_ratio) / (1 - august_ratio) = 1.75 := by
sorry

end NUMINAMATH_CALUDE_resort_tips_multiple_l3967_396714


namespace NUMINAMATH_CALUDE_min_queries_for_parity_l3967_396772

/-- Represents a query about the parity of balls in 15 bags -/
def Query := Fin 100 → Bool

/-- Represents the state of all bags -/
def BagState := Fin 100 → Bool

/-- The result of a query given a bag state -/
def queryResult (q : Query) (s : BagState) : Bool :=
  (List.filter (fun i => q i) (List.range 100)).foldl (fun acc i => acc ≠ s i) false

/-- A set of queries is sufficient if it can determine the parity of bag 1 -/
def isSufficient (qs : List Query) : Prop :=
  ∀ s1 s2 : BagState, (∀ q ∈ qs, queryResult q s1 = queryResult q s2) → s1 0 = s2 0

theorem min_queries_for_parity : 
  (∃ qs : List Query, qs.length = 3 ∧ isSufficient qs) ∧
  (∀ qs : List Query, qs.length < 3 → ¬isSufficient qs) := by
  sorry

end NUMINAMATH_CALUDE_min_queries_for_parity_l3967_396772


namespace NUMINAMATH_CALUDE_farm_tax_problem_l3967_396718

/-- The farm tax problem -/
theorem farm_tax_problem (total_cultivated_land : ℝ) (tax_rate : ℝ) :
  let total_taxable_land := 0.6 * total_cultivated_land
  let williams_taxable_land := 0.16 * total_taxable_land
  let williams_tax := 480
  williams_tax = williams_taxable_land * tax_rate →
  total_taxable_land * tax_rate = 3000 :=
by sorry

end NUMINAMATH_CALUDE_farm_tax_problem_l3967_396718


namespace NUMINAMATH_CALUDE_triangle_inequality_l3967_396758

/-- For any triangle ABC with semiperimeter p and inradius r, 
    the sum of the reciprocals of the square roots of twice the sines of its angles 
    is less than or equal to the square root of the ratio of its semiperimeter to its inradius. -/
theorem triangle_inequality (A B C : Real) (p r : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧ 0 < p ∧ 0 < r → 
  1 / Real.sqrt (2 * Real.sin A) + 1 / Real.sqrt (2 * Real.sin B) + 1 / Real.sqrt (2 * Real.sin C) ≤ Real.sqrt (p / r) := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequality_l3967_396758


namespace NUMINAMATH_CALUDE_teacher_health_survey_l3967_396710

theorem teacher_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 120)
  (h_high_bp : high_bp = 70)
  (h_heart_trouble : heart_trouble = 40)
  (h_both : both = 20) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 25 :=
by sorry

end NUMINAMATH_CALUDE_teacher_health_survey_l3967_396710


namespace NUMINAMATH_CALUDE_essay_word_ratio_l3967_396715

def johnny_words : ℕ := 150
def timothy_words (madeline_words : ℕ) : ℕ := madeline_words + 30
def total_pages : ℕ := 3
def words_per_page : ℕ := 260

theorem essay_word_ratio (madeline_words : ℕ) :
  (johnny_words + madeline_words + timothy_words madeline_words = total_pages * words_per_page) →
  (madeline_words : ℚ) / johnny_words = 2 := by
  sorry

end NUMINAMATH_CALUDE_essay_word_ratio_l3967_396715


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3967_396767

/-- Given a geometric sequence {aₙ} where a₁ = -2 and a₅ = -4, prove that a₃ = -2√2 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = -2) (h_a5 : a 5 = -4) : a 3 = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3967_396767


namespace NUMINAMATH_CALUDE_hat_markup_price_l3967_396735

theorem hat_markup_price (P : ℝ) (h : 2 * P - 1.6 * P = 6) : 1.6 * P = 24 := by
  sorry

end NUMINAMATH_CALUDE_hat_markup_price_l3967_396735


namespace NUMINAMATH_CALUDE_line_equation_l3967_396766

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 2*y + 4 = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := x + 3*y + 5 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ :=
  (0, 2)

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Theorem statement
theorem line_equation : 
  ∃ (A B C : ℝ), 
    (A ≠ 0 ∨ B ≠ 0) ∧ 
    (∀ x y : ℝ, A*x + B*y + C = 0 ↔ 
      (line1 x y ∧ line2 x y) ∨
      (x = intersection_point.1 ∧ y = intersection_point.2) ∨
      (∃ m : ℝ, perpendicular m (-1/3) ∧ y - intersection_point.2 = m * (x - intersection_point.1))) ∧
    A = 3 ∧ B = -1 ∧ C = 2 :=
  sorry

end NUMINAMATH_CALUDE_line_equation_l3967_396766


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_lower_bound_achievable_l3967_396745

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / x + 1 / (2 * y) = 1) : 
  x + 4 * y ≥ 3 + 2 * Real.sqrt 2 := by
sorry

theorem lower_bound_achievable : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1 / x + 1 / (2 * y) = 1 ∧ 
  x + 4 * y = 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_lower_bound_achievable_l3967_396745


namespace NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l3967_396784

/-- The perimeter of a rectangular garden with length 100 m and breadth 200 m is 600 m. -/
theorem garden_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun length breadth perimeter =>
    length = 100 ∧ 
    breadth = 200 ∧ 
    perimeter = 2 * (length + breadth) →
    perimeter = 600

-- The proof is omitted
theorem garden_perimeter_proof : garden_perimeter 100 200 600 := by sorry

end NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l3967_396784


namespace NUMINAMATH_CALUDE_ellas_food_calculation_l3967_396713

/-- The amount of food Ella eats each day, in pounds -/
def ellas_daily_food : ℝ := 20

/-- The number of days considered -/
def days : ℕ := 10

/-- The total amount of food Ella and her dog eat in the given number of days, in pounds -/
def total_food : ℝ := 1000

/-- The ratio of food Ella's dog eats compared to Ella -/
def dog_food_ratio : ℝ := 4

theorem ellas_food_calculation :
  ellas_daily_food * (1 + dog_food_ratio) * days = total_food :=
by sorry

end NUMINAMATH_CALUDE_ellas_food_calculation_l3967_396713


namespace NUMINAMATH_CALUDE_cubic_repeated_root_l3967_396746

/-- The cubic equation has a repeated root iff p = 5 or p = -7 -/
theorem cubic_repeated_root (p : ℝ) : 
  (∃ x : ℝ, (3 * x^3 - (p + 1) * x^2 + 4 * x - 12 = 0) ∧ 
   (6 * x^2 - 2 * (p + 1) * x + 4 = 0)) ↔ 
  (p = 5 ∨ p = -7) :=
sorry

end NUMINAMATH_CALUDE_cubic_repeated_root_l3967_396746


namespace NUMINAMATH_CALUDE_negation_of_all_x_squared_plus_one_positive_l3967_396749

theorem negation_of_all_x_squared_plus_one_positive :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_x_squared_plus_one_positive_l3967_396749


namespace NUMINAMATH_CALUDE_expansion_properties_l3967_396727

theorem expansion_properties (n : ℕ) (hn : n ≥ 3) :
  (∃ k : ℕ, k ≥ 1 ∧ k ≤ n ∧ (14 - 3 * k) % 4 = 0) ∧
  (∀ k : ℕ, k ≥ 0 → k ≤ n → Nat.choose n k * ((-1/2)^k : ℚ) ≤ 21/4) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l3967_396727


namespace NUMINAMATH_CALUDE_factor_expression_l3967_396798

theorem factor_expression (x y : ℝ) : x^2 * y - x * y^2 = x * y * (x - y) := by sorry

end NUMINAMATH_CALUDE_factor_expression_l3967_396798


namespace NUMINAMATH_CALUDE_days_without_calls_is_250_l3967_396764

/-- Represents the frequency of calls from each grandchild -/
def call_frequency₁ : ℕ := 5
def call_frequency₂ : ℕ := 7

/-- Represents the number of days in the year -/
def days_in_year : ℕ := 365

/-- Calculates the number of days without calls -/
def days_without_calls : ℕ :=
  days_in_year - (days_in_year / call_frequency₁ + days_in_year / call_frequency₂ - days_in_year / (call_frequency₁ * call_frequency₂))

/-- Theorem stating that there are 250 days without calls -/
theorem days_without_calls_is_250 : days_without_calls = 250 := by
  sorry

end NUMINAMATH_CALUDE_days_without_calls_is_250_l3967_396764


namespace NUMINAMATH_CALUDE_swap_values_l3967_396789

theorem swap_values (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  ∃ c : ℕ, (c = b) ∧ (b = a) ∧ (a = c) → a = 2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_swap_values_l3967_396789


namespace NUMINAMATH_CALUDE_sum_a_d_l3967_396752

theorem sum_a_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 42)
  (h2 : b + c = 6) : 
  a + d = 7 := by sorry

end NUMINAMATH_CALUDE_sum_a_d_l3967_396752


namespace NUMINAMATH_CALUDE_triangle_angle_value_l3967_396701

theorem triangle_angle_value (A B C : ℝ) (a b c : ℝ) :
  0 < B → B < π →
  0 < C → C < π →
  b * Real.cos C + c * Real.sin B = 0 →
  C = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l3967_396701


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3967_396739

/-- The solution set of the inequality |x| + |x - 1| < 2 -/
theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x| + |x - 1| < 2} = Set.Ioo (-1/2 : ℝ) (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3967_396739


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3967_396781

theorem trigonometric_product_equals_one :
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3967_396781


namespace NUMINAMATH_CALUDE_triangle_side_length_l3967_396721

theorem triangle_side_length (a b c : ℝ) (A : Real) :
  A = π / 3 →  -- Angle A = 60°
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →  -- Area of triangle = √3
  b + c = 6 →  -- Given condition
  a = 2 * Real.sqrt 6 := by  -- Prove that a = 2√6
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3967_396721


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3967_396770

theorem smallest_fraction_between (p q : ℕ+) : 
  (5 : ℚ) / 9 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (4 : ℚ) / 7 ∧ 
  (∀ p' q' : ℕ+, (5 : ℚ) / 9 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (4 : ℚ) / 7 → q ≤ q') →
  q - p = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3967_396770


namespace NUMINAMATH_CALUDE_greatest_gcd_square_successor_l3967_396751

theorem greatest_gcd_square_successor (n : ℕ+) : 
  ∃ (k : ℕ+), Nat.gcd (6 * n^2) (n + 1) ≤ 6 ∧ 
  Nat.gcd (6 * k^2) (k + 1) = 6 :=
sorry

end NUMINAMATH_CALUDE_greatest_gcd_square_successor_l3967_396751


namespace NUMINAMATH_CALUDE_product_calculation_l3967_396791

theorem product_calculation : 
  (1 / 3) * 6 * (1 / 12) * 24 * (1 / 48) * 96 * (1 / 192) * 384 = 16 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l3967_396791


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_75_7350_l3967_396743

theorem gcd_lcm_sum_75_7350 : Nat.gcd 75 7350 + Nat.lcm 75 7350 = 3225 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_75_7350_l3967_396743


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l3967_396797

theorem rowing_time_ratio (man_speed : ℝ) (current_speed : ℝ) 
  (h1 : man_speed = 3.3) (h2 : current_speed = 1.1) :
  (man_speed - current_speed) / (man_speed + current_speed) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_ratio_l3967_396797


namespace NUMINAMATH_CALUDE_end_of_year_deposits_l3967_396722

/-- Accumulated capital for end-of-year deposits given beginning-of-year deposits -/
theorem end_of_year_deposits (P r : ℝ) (n : ℕ) (K : ℝ) :
  P > 0 → r > 0 → n > 0 →
  K = P * ((1 + r/100)^n - 1) / (r/100) * (1 + r/100) →
  ∃ K', K' = P * ((1 + r/100)^n - 1) / (r/100) ∧ K' = K / (1 + r/100) := by
  sorry

end NUMINAMATH_CALUDE_end_of_year_deposits_l3967_396722


namespace NUMINAMATH_CALUDE_cubic_factorization_l3967_396788

theorem cubic_factorization (x : ℝ) : -3*x + 6*x^2 - 3*x^3 = -3*x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3967_396788


namespace NUMINAMATH_CALUDE_eccentricity_conic_sections_l3967_396773

theorem eccentricity_conic_sections : ∃ (e₁ e₂ : ℝ), 
  e₁^2 - 5*e₁ + 1 = 0 ∧ 
  e₂^2 - 5*e₂ + 1 = 0 ∧ 
  (0 < e₁ ∧ e₁ < 1) ∧ 
  (e₂ > 1) := by sorry

end NUMINAMATH_CALUDE_eccentricity_conic_sections_l3967_396773


namespace NUMINAMATH_CALUDE_nonzero_matrix_squared_zero_l3967_396782

theorem nonzero_matrix_squared_zero : 
  ∃ (A : Matrix (Fin 2) (Fin 2) ℝ), A ≠ 0 ∧ A ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_matrix_squared_zero_l3967_396782


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3967_396741

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 2 * Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = Real.sqrt 76 ∧ θ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3967_396741


namespace NUMINAMATH_CALUDE_velocity_at_5_seconds_l3967_396759

-- Define the position function
def s (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 4

-- Define the velocity function as the derivative of the position function
def v (t : ℝ) : ℝ := 6 * t - 2

-- Theorem statement
theorem velocity_at_5_seconds :
  v 5 = 28 := by
  sorry

end NUMINAMATH_CALUDE_velocity_at_5_seconds_l3967_396759


namespace NUMINAMATH_CALUDE_sin_70_deg_l3967_396706

theorem sin_70_deg (k : ℝ) (h : Real.sin (10 * π / 180) = k) : 
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_70_deg_l3967_396706


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3967_396720

/-- A circle with center (a, 2a) and radius √5 is tangent to the line 2x + y + 1 = 0 
    if and only if its equation is (x-1)² + (y-2)² = 5 -/
theorem circle_tangent_to_line (x y : ℝ) : 
  (∃ a : ℝ, (x - a)^2 + (y - 2*a)^2 = 5 ∧ 
   (|2*a + 2*a + 1| / Real.sqrt 5 = Real.sqrt 5)) ↔ 
  (x - 1)^2 + (y - 2)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3967_396720


namespace NUMINAMATH_CALUDE_hall_length_proof_l3967_396786

theorem hall_length_proof (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = breadth + 5 →
  area = length * breadth →
  area = 750 →
  length = 30 := by
sorry

end NUMINAMATH_CALUDE_hall_length_proof_l3967_396786


namespace NUMINAMATH_CALUDE_five_ruble_coins_l3967_396729

theorem five_ruble_coins (total_coins : ℕ) 
  (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ) :
  total_coins = 25 →
  not_two_ruble = 19 →
  not_ten_ruble = 20 →
  not_one_ruble = 16 →
  total_coins - (total_coins - not_two_ruble + total_coins - not_ten_ruble + total_coins - not_one_ruble) = 5 :=
by sorry

end NUMINAMATH_CALUDE_five_ruble_coins_l3967_396729


namespace NUMINAMATH_CALUDE_double_room_percentage_l3967_396723

theorem double_room_percentage (total_students : ℝ) (h : total_students > 0) :
  let students_in_double_rooms := 0.75 * total_students
  let double_rooms := students_in_double_rooms / 2
  let students_in_single_rooms := 0.25 * total_students
  let single_rooms := students_in_single_rooms
  let total_rooms := double_rooms + single_rooms
  (double_rooms / total_rooms) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_double_room_percentage_l3967_396723


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3967_396763

/-- Given a line and a circle, prove that the coefficient of x in the line equation is 2 when the chord length is 4 -/
theorem line_circle_intersection (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 4*x - 2*y + 1 = 0 ∧ a*x + y - 5 = 0) → -- Circle and line intersect
  (∃ x1 y1 x2 y2 : ℝ, 
    x1^2 + y1^2 - 4*x1 - 2*y1 + 1 = 0 ∧ 
    x2^2 + y2^2 - 4*x2 - 2*y2 + 1 = 0 ∧ 
    a*x1 + y1 - 5 = 0 ∧ 
    a*x2 + y2 - 5 = 0 ∧ 
    (x1 - x2)^2 + (y1 - y2)^2 = 16) → -- Chord length is 4
  a = 2 := by
  sorry

#check line_circle_intersection

end NUMINAMATH_CALUDE_line_circle_intersection_l3967_396763


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l3967_396771

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) = (4 * Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l3967_396771


namespace NUMINAMATH_CALUDE_non_real_roots_quadratic_l3967_396754

theorem non_real_roots_quadratic (b : ℝ) :
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_non_real_roots_quadratic_l3967_396754


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l3967_396783

/-- The function f(x) = x³ --/
def f (x : ℝ) : ℝ := x^3

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_line_at_x_1 :
  let m := f 1
  let slope := f' 1
  (fun x y => y - m = slope * (x - 1)) = (fun x y => y = 3 * x - 2) := by
    sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l3967_396783


namespace NUMINAMATH_CALUDE_connie_watch_savings_l3967_396711

/-- The amount of money Connie needs to buy a watch -/
theorem connie_watch_savings (saved : ℕ) (watch_cost : ℕ) (h1 : saved = 39) (h2 : watch_cost = 55) :
  watch_cost - saved = 16 := by
  sorry

end NUMINAMATH_CALUDE_connie_watch_savings_l3967_396711


namespace NUMINAMATH_CALUDE_biquadratic_root_negation_l3967_396730

/-- A biquadratic equation is of the form ax^4 + bx^2 + c = 0 -/
def BiquadraticEquation (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^4 + b * x^2 + c = 0

/-- If α is a root of a biquadratic equation, then -α is also a root -/
theorem biquadratic_root_negation (a b c α : ℝ) :
  BiquadraticEquation a b c α → BiquadraticEquation a b c (-α) :=
by sorry

end NUMINAMATH_CALUDE_biquadratic_root_negation_l3967_396730


namespace NUMINAMATH_CALUDE_abs_diff_eq_diff_abs_condition_l3967_396733

theorem abs_diff_eq_diff_abs_condition (a b : ℝ) :
  (∀ a b : ℝ, |a - b| = |a| - |b| → a * b ≥ 0) ∧
  (∃ a b : ℝ, a * b ≥ 0 ∧ |a - b| ≠ |a| - |b|) :=
by sorry

end NUMINAMATH_CALUDE_abs_diff_eq_diff_abs_condition_l3967_396733


namespace NUMINAMATH_CALUDE_composition_value_l3967_396775

-- Define the functions h and j
def h (x : ℝ) : ℝ := 4 * x + 5
def j (x : ℝ) : ℝ := 6 * x - 11

-- State the theorem
theorem composition_value : j (h 5) = 139 := by sorry

end NUMINAMATH_CALUDE_composition_value_l3967_396775


namespace NUMINAMATH_CALUDE_length_of_AB_l3967_396785

/-- Given two points P and Q on a line segment AB, prove that AB has length 35 -/
theorem length_of_AB (A B P Q : ℝ) : 
  (0 < A ∧ A < P ∧ P < Q ∧ Q < B) →  -- P and Q are on AB and on the same side of midpoint
  (P - A) / (B - P) = 1 / 4 →        -- P divides AB in ratio 1:4
  (Q - A) / (B - Q) = 2 / 5 →        -- Q divides AB in ratio 2:5
  Q - P = 3 →                        -- Distance PQ = 3
  B - A = 35 := by                   -- Length of AB is 35
sorry


end NUMINAMATH_CALUDE_length_of_AB_l3967_396785


namespace NUMINAMATH_CALUDE_insurance_coverage_percentage_l3967_396708

theorem insurance_coverage_percentage
  (total_cost : ℝ)
  (out_of_pocket : ℝ)
  (h1 : total_cost = 500)
  (h2 : out_of_pocket = 200) :
  (total_cost - out_of_pocket) / total_cost * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_insurance_coverage_percentage_l3967_396708


namespace NUMINAMATH_CALUDE_company_contracts_probability_l3967_396705

theorem company_contracts_probability
  (p_hardware : ℝ)
  (p_not_software : ℝ)
  (p_network : ℝ)
  (p_maintenance : ℝ)
  (p_at_least_one : ℝ)
  (h_hardware : p_hardware = 3/4)
  (h_not_software : p_not_software = 3/5)
  (h_network : p_network = 2/3)
  (h_maintenance : p_maintenance = 1/2)
  (h_at_least_one : p_at_least_one = 7/8) :
  p_hardware * (1 - p_not_software) * p_network * p_maintenance = 1/10 :=
sorry

end NUMINAMATH_CALUDE_company_contracts_probability_l3967_396705


namespace NUMINAMATH_CALUDE_field_trip_van_occupancy_l3967_396738

theorem field_trip_van_occupancy (num_vans num_buses people_per_bus total_people : ℕ) 
  (h1 : num_vans = 9)
  (h2 : num_buses = 10)
  (h3 : people_per_bus = 27)
  (h4 : total_people = 342) :
  (total_people - num_buses * people_per_bus) / num_vans = 8 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_van_occupancy_l3967_396738


namespace NUMINAMATH_CALUDE_fish_tanks_theorem_l3967_396776

/-- The total number of fish in three tanks, where one tank has a given number of fish
    and the other two have twice as many fish each as the first. -/
def total_fish (first_tank_fish : ℕ) : ℕ :=
  first_tank_fish + 2 * (2 * first_tank_fish)

/-- Theorem stating that with 3 fish tanks, where one tank has 20 fish and the other two
    have twice as many fish each as the first, the total number of fish is 100. -/
theorem fish_tanks_theorem : total_fish 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_fish_tanks_theorem_l3967_396776


namespace NUMINAMATH_CALUDE_perpendicular_line_parallel_line_l3967_396736

-- Define the types for our points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ → Prop

-- Define the intersection point of two lines
def intersection (l1 l2 : Line) : Point :=
  let x := -1
  let y := 2
  (x, y)

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop :=
  ∃ (m1 m2 : ℝ), m1 * m2 = -1 ∧
    (∀ x y c, l1 x y c ↔ m1 * x + y = c) ∧
    (∀ x y c, l2 x y c ↔ m2 * x + y = c)

-- Define parallelism of two lines
def parallel (l1 l2 : Line) : Prop :=
  ∃ (m c1 c2 : ℝ), 
    (∀ x y c, l1 x y c ↔ m * x + y = c1) ∧
    (∀ x y c, l2 x y c ↔ m * x + y = c2)

-- Define the given lines
def line1 : Line := λ x y c => 3 * x + 4 * y - 5 = c
def line2 : Line := λ x y c => 2 * x + y = c
def line3 : Line := λ x y c => 3 * x - 2 * y - 1 = c

-- State the theorems
theorem perpendicular_line :
  let p := intersection line1 line2
  ∃ (l : Line), l p.1 p.2 (-4) ∧ perpendicular l line3 ∧ 
    ∀ x y c, l x y c ↔ 2 * x + 3 * y - 4 = c := by sorry

theorem parallel_line :
  let p := intersection line1 line2
  ∃ (l : Line), l p.1 p.2 7 ∧ parallel l line3 ∧ 
    ∀ x y c, l x y c ↔ 3 * x - 2 * y + 7 = c := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_parallel_line_l3967_396736


namespace NUMINAMATH_CALUDE_flower_shop_sales_l3967_396702

theorem flower_shop_sales (lilacs : ℕ) : 
  (3 * lilacs) + lilacs + (lilacs / 2) = 45 → lilacs = 10 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_sales_l3967_396702


namespace NUMINAMATH_CALUDE_present_age_of_R_l3967_396774

-- Define the present ages of P, Q, and R
variable (Pp Qp Rp : ℝ)

-- Define the conditions
def condition1 : Prop := Pp - 8 = (1/2) * (Qp - 8)
def condition2 : Prop := Qp - 8 = (2/3) * (Rp - 8)
def condition3 : Prop := Qp = 2 * Real.sqrt Rp
def condition4 : Prop := Pp / Qp = 3/5

-- Theorem statement
theorem present_age_of_R 
  (h1 : condition1 Pp Qp)
  (h2 : condition2 Qp Rp)
  (h3 : condition3 Qp Rp)
  (h4 : condition4 Pp Qp) :
  Rp = 400 := by
  sorry

end NUMINAMATH_CALUDE_present_age_of_R_l3967_396774


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l3967_396799

theorem smaller_number_in_ratio (a b : ℝ) : 
  a / b = 3 / 4 → a + b = 420 → a = 180 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l3967_396799


namespace NUMINAMATH_CALUDE_min_value_sum_equality_condition_l3967_396719

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (2 * c^2) + c^2 / (9 * a) ≥ 3 / Real.rpow 54 (1/3) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (2 * c^2) + c^2 / (9 * a) = 3 / Real.rpow 54 (1/3) ↔
  a = 6 * c^2 ∧ b = 2 * c^2 * Real.rpow 54 (1/3) ∧ c = Real.rpow 54 (1/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_equality_condition_l3967_396719


namespace NUMINAMATH_CALUDE_triangle_properties_l3967_396778

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are opposite sides to A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Area condition
  (b^2 / (3 * Real.sin B)) = (1/2) * a * c * Real.sin B →
  -- Given condition
  Real.cos A * Real.cos C = 1/6 →
  -- Prove these
  Real.sin A * Real.sin C = 2/3 ∧ B = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3967_396778


namespace NUMINAMATH_CALUDE_max_sum_of_first_two_l3967_396726

theorem max_sum_of_first_two (a b c d e : ℕ) : 
  a < b → b < c → c < d → d < e → 
  a + 2*b + 3*c + 4*d + 5*e = 300 → 
  a + b ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_first_two_l3967_396726


namespace NUMINAMATH_CALUDE_area_between_circle_and_square_l3967_396792

/-- Given a square with side length 2 and a circle with radius √2 sharing the same center,
    the area inside the circle but outside the square is equal to 2π - 4. -/
theorem area_between_circle_and_square :
  let square_side : ℝ := 2
  let circle_radius : ℝ := Real.sqrt 2
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  circle_area - square_area = 2 * π - 4 := by
  sorry

end NUMINAMATH_CALUDE_area_between_circle_and_square_l3967_396792


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l3967_396716

theorem power_of_three_mod_five : 3^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l3967_396716


namespace NUMINAMATH_CALUDE_average_weight_a_b_l3967_396757

/-- Given three weights a, b, and c, proves that the average of a and b is 40,
    under certain conditions. -/
theorem average_weight_a_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 31 →
  (a + b) / 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_weight_a_b_l3967_396757


namespace NUMINAMATH_CALUDE_f_is_odd_l3967_396779

-- Define the function F
def F (f : ℝ → ℝ) (x : ℝ) : ℝ := (x^3 - 2*x) * f x

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define what it means for a function to be odd
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- State the theorem
theorem f_is_odd (f : ℝ → ℝ) (h1 : is_even (F f)) (h2 : ∃ x, f x ≠ 0) : is_odd f := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_l3967_396779


namespace NUMINAMATH_CALUDE_possible_d_values_l3967_396742

theorem possible_d_values : 
  ∀ d : ℤ, (∃ e f : ℤ, ∀ x : ℤ, (x - d) * (x - 12) + 1 = (x + e) * (x + f)) → (d = 22 ∨ d = 26) :=
by sorry

end NUMINAMATH_CALUDE_possible_d_values_l3967_396742


namespace NUMINAMATH_CALUDE_polynomial_decomposition_l3967_396760

/-- The set of valid n values for which the polynomial decomposition is possible -/
def valid_n : Set ℕ :=
  {0, 1, 3, 7, 15, 12, 18, 25, 37, 51, 75, 151, 246, 493, 987, 1975}

/-- Predicate to check if a list of coefficients is valid for a given n -/
def valid_coefficients (n : ℕ) (coeffs : List ℕ) : Prop :=
  coeffs.length = n ∧
  coeffs.Nodup ∧
  ∀ a ∈ coeffs, 0 < a ∧ a ≤ n

/-- The main theorem stating the condition for valid polynomial decomposition -/
theorem polynomial_decomposition (n : ℕ) :
  (∃ coeffs : List ℕ, valid_coefficients n coeffs) ↔ n ∈ valid_n := by
  sorry

#check polynomial_decomposition

end NUMINAMATH_CALUDE_polynomial_decomposition_l3967_396760


namespace NUMINAMATH_CALUDE_inequality_proof_l3967_396728

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄)
  (h4 : x₂ + x₃ + x₄ ≥ x₁) : 
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3967_396728
