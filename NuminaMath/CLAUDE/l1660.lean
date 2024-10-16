import Mathlib

namespace NUMINAMATH_CALUDE_go_game_probabilities_l1660_166071

/-- Represents the probability of a player winning a single game -/
structure GameProbability where
  player_a : ℝ
  player_b : ℝ
  sum_to_one : player_a + player_b = 1

/-- Represents the state of the game after the first two games -/
structure InitialState where
  a_wins : ℕ
  b_wins : ℕ
  total_games : a_wins + b_wins = 2

/-- Calculates the probability of player A winning the competition -/
def probability_a_wins (p : GameProbability) (init : InitialState) : ℝ :=
  p.player_a * p.player_a +
  p.player_b * p.player_a * p.player_a +
  p.player_a * p.player_b * p.player_a

/-- Calculates the probability of the competition ending after 5 games -/
def probability_end_after_five (p : GameProbability) (init : InitialState) : ℝ :=
  p.player_b * p.player_a * p.player_a +
  p.player_a * p.player_b * p.player_a +
  p.player_a * p.player_b * p.player_b +
  p.player_b * p.player_a * p.player_b

/-- The main theorem stating the probabilities for the Go game competition -/
theorem go_game_probabilities 
  (p : GameProbability) 
  (init : InitialState) 
  (h_p : p.player_a = 0.6 ∧ p.player_b = 0.4) 
  (h_init : init.a_wins = 1 ∧ init.b_wins = 1) : 
  probability_a_wins p init = 0.648 ∧ 
  probability_end_after_five p init = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_go_game_probabilities_l1660_166071


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l1660_166033

theorem max_digits_product_5_4 : 
  ∀ a b : ℕ, 
  10000 ≤ a ∧ a < 100000 → 
  1000 ≤ b ∧ b < 10000 → 
  a * b < 1000000000 :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l1660_166033


namespace NUMINAMATH_CALUDE_doughnuts_left_l1660_166042

theorem doughnuts_left (total_doughnuts : ℕ) (staff_count : ℕ) (doughnuts_per_staff : ℕ) :
  total_doughnuts = 50 →
  staff_count = 19 →
  doughnuts_per_staff = 2 →
  total_doughnuts - (staff_count * doughnuts_per_staff) = 12 :=
by sorry

end NUMINAMATH_CALUDE_doughnuts_left_l1660_166042


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l1660_166012

theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 4 * a + b = a * b) :
  (∃ (min : ℝ), min = 9 ∧ ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = x * y → x + y ≥ min) ∧
  (∀ x t : ℝ, t ∈ Set.Icc (-1) 3 → |x - a| + |x - b| ≥ t^2 - 2*t) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l1660_166012


namespace NUMINAMATH_CALUDE_elderly_workers_in_sample_l1660_166007

/-- Represents the composition of workers in a company --/
structure WorkforceComposition where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents a stratified sample from the workforce --/
structure StratifiedSample where
  youngInSample : ℕ
  elderlyInSample : ℕ

/-- Theorem stating the number of elderly workers in the stratified sample --/
theorem elderly_workers_in_sample 
  (wc : WorkforceComposition) 
  (sample : StratifiedSample) : 
  wc.total = 430 →
  wc.young = 160 →
  wc.middleAged = 2 * wc.elderly →
  sample.youngInSample = 32 →
  sample.elderlyInSample = 18 := by
  sorry

end NUMINAMATH_CALUDE_elderly_workers_in_sample_l1660_166007


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1660_166003

-- Define repeating decimals
def repeating_decimal_8 : ℚ := 8/9
def repeating_decimal_2 : ℚ := 2/9

-- Theorem statement
theorem sum_of_repeating_decimals : 
  repeating_decimal_8 + repeating_decimal_2 = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1660_166003


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l1660_166070

/-- Proves that the area of a rectangular garden with length three times its width and width of 12 meters is 432 square meters. -/
theorem rectangular_garden_area :
  ∀ (length width area : ℝ),
    width = 12 →
    length = 3 * width →
    area = length * width →
    area = 432 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l1660_166070


namespace NUMINAMATH_CALUDE_girls_in_class_l1660_166040

theorem girls_in_class (boys : ℕ) (ways : ℕ) : boys = 15 → ways = 1050 → ∃ girls : ℕ,
  girls * (boys.choose 2) = ways ∧ girls = 10 := by sorry

end NUMINAMATH_CALUDE_girls_in_class_l1660_166040


namespace NUMINAMATH_CALUDE_carters_baseball_cards_l1660_166081

/-- Given that Marcus has 210 baseball cards and 58 more than Carter,
    prove that Carter has 152 baseball cards. -/
theorem carters_baseball_cards :
  ∀ (marcus_cards carter_cards : ℕ),
    marcus_cards = 210 →
    marcus_cards = carter_cards + 58 →
    carter_cards = 152 :=
by sorry

end NUMINAMATH_CALUDE_carters_baseball_cards_l1660_166081


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1660_166049

/-- 
A proposition stating that "a>2 and b>2" is a sufficient but not necessary condition 
for "a+b>4 and ab>4" for real numbers a and b.
-/
theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∃ x y : ℝ, x > 2 ∧ y > 2 → x + y > 4 ∧ x * y > 4) ∧ 
  (∃ p q : ℝ, p + q > 4 ∧ p * q > 4 ∧ ¬(p > 2 ∧ q > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1660_166049


namespace NUMINAMATH_CALUDE_largest_solution_proof_l1660_166025

/-- The equation from the problem -/
def equation (x : ℝ) : Prop :=
  4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 12*x - 5

/-- The largest real solution to the equation -/
def largest_solution : ℝ := 20

/-- The representation of the solution in the form d + √(e + √f) -/
def solution_form (d e f : ℕ) (x : ℝ) : Prop :=
  x = d + Real.sqrt (e + Real.sqrt f)

theorem largest_solution_proof :
  equation largest_solution ∧
  ∃ (d e f : ℕ), solution_form d e f largest_solution ∧
  ∀ (x : ℝ), equation x → x ≤ largest_solution :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_proof_l1660_166025


namespace NUMINAMATH_CALUDE_mirror_area_l1660_166086

/-- Given a rectangular frame with outer dimensions 100 cm by 140 cm and a uniform frame width of 15 cm,
    the area of the rectangular mirror that fits exactly inside the frame is 7700 cm². -/
theorem mirror_area (frame_width : ℝ) (frame_height : ℝ) (frame_thickness : ℝ) : 
  frame_width = 100 ∧ frame_height = 140 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 7700 := by
  sorry

end NUMINAMATH_CALUDE_mirror_area_l1660_166086


namespace NUMINAMATH_CALUDE_even_function_intersection_l1660_166004

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem even_function_intersection (ω φ : ℝ) :
  (0 < φ) → (φ < π) →
  (∀ x, f ω φ x = f ω φ (-x)) →
  (∃ x₁ x₂, f ω φ x₁ = 2 ∧ f ω φ x₂ = 2 ∧ |x₁ - x₂| = π) →
  ω = 2 ∧ φ = π/2 := by
sorry

end NUMINAMATH_CALUDE_even_function_intersection_l1660_166004


namespace NUMINAMATH_CALUDE_m_range_characterization_l1660_166018

def f (x : ℝ) : ℝ := x^2 + 3

theorem m_range_characterization (m : ℝ) : 
  (∀ x ≥ 1, f x + m^2 * f x ≥ f (x - 1) + 3 * f m) ↔ 
  (m ≤ -1 ∨ m ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_m_range_characterization_l1660_166018


namespace NUMINAMATH_CALUDE_range_of_a_l1660_166027

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*|x - a| ≥ a^2) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1660_166027


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sign_l1660_166034

theorem quadratic_coefficient_sign 
  (a b c : ℝ) 
  (h1 : a + b + c < 0) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) : 
  c < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sign_l1660_166034


namespace NUMINAMATH_CALUDE_sticker_distribution_l1660_166041

/-- Represents the share of stickers each winner should receive -/
structure Share where
  al : Rat
  bert : Rat
  carl : Rat
  dan : Rat

/-- Calculates the remaining fraction of stickers after all winners have taken their perceived shares -/
def remaining_stickers (s : Share) : Rat :=
  let total := 1
  let bert_sees := total - s.al
  let carl_sees := bert_sees - (s.bert * bert_sees)
  let dan_sees := carl_sees - (s.carl * carl_sees)
  total - (s.al + s.bert * bert_sees + s.carl * carl_sees + s.dan * dan_sees)

/-- The theorem to be proved -/
theorem sticker_distribution (s : Share) 
  (h1 : s.al = 4/10)
  (h2 : s.bert = 3/10)
  (h3 : s.carl = 2/10)
  (h4 : s.dan = 1/10) :
  remaining_stickers s = 2844/10000 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1660_166041


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1660_166087

theorem largest_angle_in_triangle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = 105 →      -- Sum of two angles is 7/6 of a right angle (90° * 7/6 = 105°)
  β = α + 20 →       -- One angle is 20° larger than the other
  max α (max β γ) = 75 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1660_166087


namespace NUMINAMATH_CALUDE_shelter_ratio_l1660_166024

/-- 
Given a shelter with dogs and cats, prove that if there are 75 dogs, 
and adding 20 cats would make the ratio of dogs to cats 15:11, 
then the initial ratio of dogs to cats is 15:7.
-/
theorem shelter_ratio (initial_cats : ℕ) : 
  (75 : ℚ) / (initial_cats + 20) = 15 / 11 → 
  75 / initial_cats = 15 / 7 := by
sorry

end NUMINAMATH_CALUDE_shelter_ratio_l1660_166024


namespace NUMINAMATH_CALUDE_prob_different_numbers_l1660_166014

/-- The number of balls in the bag -/
def num_balls : ℕ := 6

/-- The probability of drawing different numbers -/
def prob_different : ℚ := 5/6

/-- Theorem stating the probability of drawing different numbers -/
theorem prob_different_numbers :
  (num_balls - 1 : ℚ) / num_balls = prob_different :=
sorry

end NUMINAMATH_CALUDE_prob_different_numbers_l1660_166014


namespace NUMINAMATH_CALUDE_cubic_function_three_zeros_l1660_166085

/-- A cubic function with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - k

/-- The derivative of f with respect to x -/
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem cubic_function_three_zeros (k : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f k x = 0 ∧ f k y = 0 ∧ f k z = 0) →
  -4 < k ∧ k < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_three_zeros_l1660_166085


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1660_166022

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2015) + 2015
  f 2015 = 2016 ∧ ∃ (x : ℝ), f x = x := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1660_166022


namespace NUMINAMATH_CALUDE_least_divisible_by_three_smallest_primes_gt_7_l1660_166005

def smallest_prime_greater_than_7 : ℕ := 11
def second_smallest_prime_greater_than_7 : ℕ := 13
def third_smallest_prime_greater_than_7 : ℕ := 17

theorem least_divisible_by_three_smallest_primes_gt_7 :
  ∃ n : ℕ, n > 0 ∧ 
  smallest_prime_greater_than_7 ∣ n ∧
  second_smallest_prime_greater_than_7 ∣ n ∧
  third_smallest_prime_greater_than_7 ∣ n ∧
  ∀ m : ℕ, m > 0 → 
    smallest_prime_greater_than_7 ∣ m →
    second_smallest_prime_greater_than_7 ∣ m →
    third_smallest_prime_greater_than_7 ∣ m →
    n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_three_smallest_primes_gt_7_l1660_166005


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1660_166077

/-- A geometric sequence with first four terms 25, -50, 100, -200 has a common ratio of -2. -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℚ), 
    (a 0 = 25) → 
    (a 1 = -50) → 
    (a 2 = 100) → 
    (a 3 = -200) → 
    (∀ n : ℕ, a (n + 1) = a n * (-2)) → 
    (∀ n : ℕ, a (n + 1) / a n = -2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1660_166077


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l1660_166031

theorem max_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = x + 2*y ∧ ∀ (w : ℝ), w = x + 2*y → w ≤ z ∧ z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l1660_166031


namespace NUMINAMATH_CALUDE_binomial_26_6_l1660_166088

theorem binomial_26_6 (h1 : Nat.choose 23 5 = 33649) 
                       (h2 : Nat.choose 23 6 = 33649)
                       (h3 : Nat.choose 25 5 = 53130) : 
  Nat.choose 26 6 = 163032 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l1660_166088


namespace NUMINAMATH_CALUDE_probability_ten_people_no_adjacent_standing_l1660_166090

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing --/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing in a circular arrangement of n people --/
def probabilityNoAdjacentStanding (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n : ℚ)

theorem probability_ten_people_no_adjacent_standing :
  probabilityNoAdjacentStanding 10 = 123 / 1024 := by
  sorry


end NUMINAMATH_CALUDE_probability_ten_people_no_adjacent_standing_l1660_166090


namespace NUMINAMATH_CALUDE_chocolate_candy_cost_l1660_166015

/-- The cost of purchasing a given number of chocolate candies, given the cost and quantity of a box. -/
theorem chocolate_candy_cost (box_quantity : ℕ) (box_cost : ℚ) (total_quantity : ℕ) : 
  (total_quantity / box_quantity : ℚ) * box_cost = 72 :=
by
  sorry

#check chocolate_candy_cost 40 8 360

end NUMINAMATH_CALUDE_chocolate_candy_cost_l1660_166015


namespace NUMINAMATH_CALUDE_notebook_increase_correct_l1660_166045

/-- Calculates the increase in Jimin's notebook count -/
def notebook_increase (initial : ℕ) (father_bought : ℕ) (mother_bought : ℕ) : ℕ :=
  father_bought + mother_bought

theorem notebook_increase_correct (initial : ℕ) (father_bought : ℕ) (mother_bought : ℕ) :
  notebook_increase initial father_bought mother_bought = father_bought + mother_bought :=
by sorry

end NUMINAMATH_CALUDE_notebook_increase_correct_l1660_166045


namespace NUMINAMATH_CALUDE_guinea_pig_food_theorem_l1660_166082

/-- The amount of food eaten by the first guinea pig -/
def first_guinea_pig_food : ℝ := 2

/-- The amount of food eaten by the second guinea pig -/
def second_guinea_pig_food : ℝ := 2 * first_guinea_pig_food

/-- The amount of food eaten by the third guinea pig -/
def third_guinea_pig_food : ℝ := second_guinea_pig_food + 3

/-- The total amount of food eaten by all three guinea pigs -/
def total_food : ℝ := first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food

theorem guinea_pig_food_theorem :
  first_guinea_pig_food = 2 ∧ total_food = 13 :=
sorry

end NUMINAMATH_CALUDE_guinea_pig_food_theorem_l1660_166082


namespace NUMINAMATH_CALUDE_min_cos_C_in_special_triangle_l1660_166026

theorem min_cos_C_in_special_triangle (A B C : ℝ) (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π)
  (h5 : ∃ k : ℝ, (1 / Real.tan A) + k = 2 / Real.tan C ∧
                 2 / Real.tan C + k = 1 / Real.tan B) :
  ∃ (cosC : ℝ), cosC = Real.cos C ∧ cosC ≥ 1/3 ∧
  ∀ (cosC' : ℝ), cosC' = Real.cos C → cosC' ≥ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_min_cos_C_in_special_triangle_l1660_166026


namespace NUMINAMATH_CALUDE_smallest_y_in_arithmetic_sequence_l1660_166008

theorem smallest_y_in_arithmetic_sequence (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →  -- x, y, z are positive
  ∃ d : ℝ, x = y - d ∧ z = y + d →  -- x, y, z form an arithmetic sequence
  x * y * z = 216 →  -- product condition
  y ≥ 6 ∧ (∀ w : ℝ, w > 0 ∧ (∃ d' : ℝ, (w - d') * w * (w + d') = 216) → w ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_in_arithmetic_sequence_l1660_166008


namespace NUMINAMATH_CALUDE_tony_mileage_milestone_l1660_166083

/-- Represents the distances for Tony's errands -/
structure ErrandDistances where
  groceries : ℕ
  haircut : ℕ
  doctor : ℕ

/-- Calculates the point at which Tony has driven exactly 15 miles -/
def mileageMilestone (distances : ErrandDistances) : ℕ :=
  if distances.groceries ≥ 15 then 15
  else distances.groceries + min (15 - distances.groceries) distances.haircut

/-- Theorem stating that Tony will have driven exactly 15 miles after completing
    his grocery trip and driving partially towards his haircut destination -/
theorem tony_mileage_milestone (distances : ErrandDistances)
    (h1 : distances.groceries = 10)
    (h2 : distances.haircut = 15)
    (h3 : distances.doctor = 5) :
    mileageMilestone distances = 15 :=
  sorry

#eval mileageMilestone ⟨10, 15, 5⟩

end NUMINAMATH_CALUDE_tony_mileage_milestone_l1660_166083


namespace NUMINAMATH_CALUDE_value_of_a_l1660_166073

theorem value_of_a (a : ℝ) (h : a = -a) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1660_166073


namespace NUMINAMATH_CALUDE_emma_remaining_time_l1660_166079

-- Define the wrapping rates and initial joint work time
def emma_rate : ℚ := 1 / 6
def troy_rate : ℚ := 1 / 8
def joint_work_time : ℚ := 2

-- Define the function to calculate the remaining time for Emma
def remaining_time_for_emma (emma_rate troy_rate joint_work_time : ℚ) : ℚ :=
  let joint_completion := (emma_rate + troy_rate) * joint_work_time
  let remaining_work := 1 - joint_completion
  remaining_work / emma_rate

-- Theorem statement
theorem emma_remaining_time :
  remaining_time_for_emma emma_rate troy_rate joint_work_time = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_emma_remaining_time_l1660_166079


namespace NUMINAMATH_CALUDE_pauls_garage_sale_l1660_166032

/-- The number of books Paul sold in the garage sale -/
def books_sold (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : ℕ :=
  initial - given_away - remaining

/-- Proof that Paul sold 27 books in the garage sale -/
theorem pauls_garage_sale : books_sold 134 39 68 = 27 := by
  sorry

end NUMINAMATH_CALUDE_pauls_garage_sale_l1660_166032


namespace NUMINAMATH_CALUDE_ava_finishes_on_monday_l1660_166039

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def days_after (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => next_day (days_after d n)

def reading_time (n : ℕ) : ℕ := 2 * n - 1

def total_reading_time (n : ℕ) : ℕ := 
  (List.range n).map reading_time |>.sum

theorem ava_finishes_on_monday : 
  days_after DayOfWeek.Sunday (total_reading_time 20) = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_ava_finishes_on_monday_l1660_166039


namespace NUMINAMATH_CALUDE_range_of_a_l1660_166097

def A (a : ℝ) : Set ℝ := {x | |x - 1| ≤ a ∧ a > 0}

def B : Set ℝ := {x | x^2 - 6*x - 7 > 0}

theorem range_of_a (a : ℝ) :
  (A a ∩ B = ∅) → (0 < a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1660_166097


namespace NUMINAMATH_CALUDE_baseball_game_opponent_score_l1660_166021

theorem baseball_game_opponent_score :
  let team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let total_games : ℕ := team_scores.length
  let lost_games : ℕ := 8
  let won_games : ℕ := total_games - lost_games
  let lost_score_diff : ℕ := 2
  let won_score_ratio : ℕ := 3
  ∃ (opponent_scores : List ℕ),
    opponent_scores.length = total_games ∧
    (∀ i ∈ Finset.range lost_games,
      opponent_scores[i]! = team_scores[i]! + lost_score_diff) ∧
    (∀ i ∈ Finset.range won_games,
      team_scores[lost_games + i]! = won_score_ratio * opponent_scores[lost_games + i]!) ∧
    opponent_scores.sum = 78 :=
by sorry

end NUMINAMATH_CALUDE_baseball_game_opponent_score_l1660_166021


namespace NUMINAMATH_CALUDE_find_x_value_l1660_166019

theorem find_x_value (numbers : List ℕ) (x : ℕ) : 
  numbers = [54, 55, 57, 58, 59, 62, 62, 63, 65] →
  numbers.length = 9 →
  (numbers.sum + x) / 10 = 60 →
  x = 65 := by
sorry

end NUMINAMATH_CALUDE_find_x_value_l1660_166019


namespace NUMINAMATH_CALUDE_fifth_root_of_unity_l1660_166094

theorem fifth_root_of_unity (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * u^4 + q * u^3 + r * u^2 + s * u + t = 0)
  (h2 : q * u^4 + r * u^3 + s * u^2 + t * u + p = 0) :
  u^5 = 1 :=
sorry

end NUMINAMATH_CALUDE_fifth_root_of_unity_l1660_166094


namespace NUMINAMATH_CALUDE_third_number_in_second_set_l1660_166016

theorem third_number_in_second_set (x y : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 90 →
  (128 + 255 + y + 1023 + x) / 5 = 423 →
  y = 511 := by sorry

end NUMINAMATH_CALUDE_third_number_in_second_set_l1660_166016


namespace NUMINAMATH_CALUDE_stating_same_suit_selections_standard_deck_l1660_166028

/-- Represents a standard deck of cards. -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h1 : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards with 4 suits and 13 cards per suit. -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h1 := rfl }

/-- 
The number of ways to select two different cards from the same suit in a standard deck,
where order matters.
-/
def same_suit_selections (d : Deck) : Nat :=
  d.num_suits * (d.cards_per_suit * (d.cards_per_suit - 1))

/-- 
Theorem stating that the number of ways to select two different cards 
from the same suit in a standard deck, where order matters, is 624.
-/
theorem same_suit_selections_standard_deck : 
  same_suit_selections standard_deck = 624 := by
  sorry


end NUMINAMATH_CALUDE_stating_same_suit_selections_standard_deck_l1660_166028


namespace NUMINAMATH_CALUDE_initial_ratio_problem_l1660_166076

theorem initial_ratio_problem (a b : ℕ) : 
  b = 6 → 
  (a + 2 : ℚ) / (b + 2 : ℚ) = 3 / 2 → 
  (a : ℚ) / b = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_initial_ratio_problem_l1660_166076


namespace NUMINAMATH_CALUDE_expression_evaluation_l1660_166067

theorem expression_evaluation (a : ℝ) (h : a = 3) : 
  (3 * a⁻¹ + (2 * a⁻¹) / 3) / (2 * a) = 11 / 54 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1660_166067


namespace NUMINAMATH_CALUDE_impossibleCubeLabeling_l1660_166084

-- Define a cube type
structure Cube where
  vertices : Fin 8 → ℕ

-- Define the property of being an odd number between 1 and 600
def isValidNumber (n : ℕ) : Prop :=
  n % 2 = 1 ∧ 1 ≤ n ∧ n ≤ 600

-- Define adjacency in a cube
def isAdjacent (i j : Fin 8) : Prop :=
  (i.val + j.val) % 2 = 1 ∧ i ≠ j

-- Define the property of having a common divisor greater than 1
def hasCommonDivisor (a b : ℕ) : Prop :=
  ∃ (d : ℕ), d > 1 ∧ a % d = 0 ∧ b % d = 0

-- Main theorem
theorem impossibleCubeLabeling :
  ¬∃ (c : Cube),
    (∀ i : Fin 8, isValidNumber (c.vertices i)) ∧
    (∀ i j : Fin 8, i ≠ j → c.vertices i ≠ c.vertices j) ∧
    (∀ i j : Fin 8, isAdjacent i j → hasCommonDivisor (c.vertices i) (c.vertices j)) ∧
    (∀ i j : Fin 8, ¬isAdjacent i j → ¬hasCommonDivisor (c.vertices i) (c.vertices j)) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibleCubeLabeling_l1660_166084


namespace NUMINAMATH_CALUDE_custom_op_inequality_l1660_166052

/-- Custom operation ⊗ on ℝ -/
def custom_op (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem statement -/
theorem custom_op_inequality (a : ℝ) :
  (∀ x > 2, custom_op (x - a) x ≤ a + 2) →
  a ∈ Set.Iic 7 :=
by sorry

end NUMINAMATH_CALUDE_custom_op_inequality_l1660_166052


namespace NUMINAMATH_CALUDE_distance_to_midpoint_l1660_166080

/-- Right triangle with inscribed circle -/
structure RightTriangleWithInscribedCircle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Points where circle touches sides
  d : ℝ  -- Distance from B to D on AB
  e : ℝ  -- Distance from B to E on BC
  f : ℝ  -- Distance from C to F on AC
  -- Conditions
  ab_positive : ab > 0
  bc_positive : bc > 0
  d_in_range : 0 < d ∧ d < ab
  e_in_range : 0 < e ∧ e < bc
  f_in_range : 0 < f ∧ f < (ab^2 + bc^2).sqrt
  circle_tangent : d + e + f = (ab^2 + bc^2).sqrt

/-- The main theorem -/
theorem distance_to_midpoint
  (t : RightTriangleWithInscribedCircle)
  (h_ab : t.ab = 6)
  (h_bc : t.bc = 8) :
  t.ab / 2 - t.d = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_midpoint_l1660_166080


namespace NUMINAMATH_CALUDE_stating_calculate_downstream_speed_l1660_166060

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- 
Theorem stating that given a man's upstream and still water speeds, 
his downstream speed can be calculated.
-/
theorem calculate_downstream_speed (speed : RowingSpeed) 
  (h1 : speed.upstream = 15)
  (h2 : speed.stillWater = 20) :
  speed.downstream = 25 := by
  sorry

#check calculate_downstream_speed

end NUMINAMATH_CALUDE_stating_calculate_downstream_speed_l1660_166060


namespace NUMINAMATH_CALUDE_triangle_area_difference_l1660_166046

/-- Given a square with side length 10 meters, divided by three straight line segments,
    where P and Q are the areas of two triangles formed by these segments, 
    prove that P - Q = 0 -/
theorem triangle_area_difference (P Q : ℝ) : 
  (∃ R : ℝ, P + R = 50 ∧ Q + R = 50) → P - Q = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_difference_l1660_166046


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1660_166075

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 150 = longer_base
  midline_ratio_condition : (shorter_base + (shorter_base + 150) / 2) / 
    ((shorter_base + 150 + (shorter_base + 150)) / 2) = 3 / 4
  equal_area_condition : ∃ h₁ : ℝ, 
    2 * (1/2 * h₁ * (shorter_base + equal_area_segment)) = 
    1/2 * height * (shorter_base + longer_base)

/-- The main theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) : 
  ⌊t.equal_area_segment^2 / 150⌋ = 300 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l1660_166075


namespace NUMINAMATH_CALUDE_pond_animals_l1660_166056

/-- Given a pond with snails and frogs, calculate the total number of animals -/
theorem pond_animals (num_snails num_frogs : ℕ) : num_snails = 5 → num_frogs = 2 → num_snails + num_frogs = 7 := by
  sorry

end NUMINAMATH_CALUDE_pond_animals_l1660_166056


namespace NUMINAMATH_CALUDE_good_couples_parity_l1660_166054

/-- Represents the color of a grid on the chess board -/
inductive Color
| Red
| Blue

/-- Converts a Color to an integer label -/
def color_to_label (c : Color) : Int :=
  match c with
  | Color.Red => 1
  | Color.Blue => -1

/-- Represents a chess board with m rows and n columns -/
structure ChessBoard (m n : Nat) where
  grid : Fin m → Fin n → Color
  m_ge_three : m ≥ 3
  n_ge_three : n ≥ 3

/-- Counts the number of "good couples" on the chess board -/
def count_good_couples (board : ChessBoard m n) : Nat :=
  sorry

/-- Calculates the product of labels for border grids (excluding corners) -/
def border_product (board : ChessBoard m n) : Int :=
  sorry

/-- Main theorem: The parity of good couples is determined by the border product -/
theorem good_couples_parity (m n : Nat) (board : ChessBoard m n) :
  Even (count_good_couples board) ↔ border_product board = 1 :=
  sorry

end NUMINAMATH_CALUDE_good_couples_parity_l1660_166054


namespace NUMINAMATH_CALUDE_man_speed_man_speed_specific_case_l1660_166091

/-- Calculates the speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / time_to_pass
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- Proves that the speed of the man is approximately 6 km/hr given the specific conditions. -/
theorem man_speed_specific_case : 
  ∃ ε > 0, |man_speed 110 84 4.399648028157747 - 6| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_specific_case_l1660_166091


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1660_166051

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the nth terms of two arithmetic sequences -/
def SumOfTerms (a b : ℕ → ℝ) (n : ℕ) : ℝ := a n + b n

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  ArithmeticSequence a → ArithmeticSequence b →
  SumOfTerms a b 1 = 7 → SumOfTerms a b 3 = 21 →
  SumOfTerms a b 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1660_166051


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l1660_166017

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_seven_balls_three_boxes : 
  distribute_balls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l1660_166017


namespace NUMINAMATH_CALUDE_constant_term_proof_l1660_166059

/-- The constant term in the expansion of (√x + 1/(3x))^10 -/
def constant_term : ℕ := 210

/-- The index of the term with the maximum coefficient -/
def max_coeff_index : ℕ := 6

theorem constant_term_proof (h : max_coeff_index = 6) : constant_term = 210 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_proof_l1660_166059


namespace NUMINAMATH_CALUDE_fraction_equality_l1660_166057

theorem fraction_equality : (1000^2 : ℚ) / (252^2 - 248^2) = 500 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1660_166057


namespace NUMINAMATH_CALUDE_simplify_and_sum_coefficients_l1660_166011

theorem simplify_and_sum_coefficients (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ), (15*d + 11 + 18*d^2) + (3*d + 2) = a*d^2 + b*d + c ∧ a + b + c = 49 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_coefficients_l1660_166011


namespace NUMINAMATH_CALUDE_probability_10_heads_in_12_flips_l1660_166089

/-- The probability of getting exactly 10 heads in 12 flips of a fair coin -/
theorem probability_10_heads_in_12_flips : 
  (Nat.choose 12 10 : ℚ) / 2^12 = 66 / 4096 := by sorry

end NUMINAMATH_CALUDE_probability_10_heads_in_12_flips_l1660_166089


namespace NUMINAMATH_CALUDE_autumn_pencils_l1660_166096

theorem autumn_pencils (initial : ℕ) 
  (misplaced : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) (final : ℕ) : 
  misplaced = 7 → broken = 3 → found = 4 → bought = 2 → final = 16 →
  initial - misplaced - broken + found + bought = final →
  initial = 22 := by
sorry

end NUMINAMATH_CALUDE_autumn_pencils_l1660_166096


namespace NUMINAMATH_CALUDE_porch_length_calculation_l1660_166064

/-- Given the dimensions of a house and porch, and the total area needing shingles,
    calculate the length of the porch. -/
theorem porch_length_calculation
  (house_length : ℝ)
  (house_width : ℝ)
  (porch_width : ℝ)
  (total_area : ℝ)
  (h1 : house_length = 20.5)
  (h2 : house_width = 10)
  (h3 : porch_width = 4.5)
  (h4 : total_area = 232) :
  (total_area - house_length * house_width) / porch_width = 6 := by
  sorry

end NUMINAMATH_CALUDE_porch_length_calculation_l1660_166064


namespace NUMINAMATH_CALUDE_iesha_book_count_l1660_166092

/-- The number of school books Iesha has -/
def school_books : ℕ := 136

/-- The number of sports books Iesha has -/
def sports_books : ℕ := 208

/-- The total number of books Iesha has -/
def total_books : ℕ := school_books + sports_books

theorem iesha_book_count : total_books = 344 := by
  sorry

end NUMINAMATH_CALUDE_iesha_book_count_l1660_166092


namespace NUMINAMATH_CALUDE_fifteenth_row_seats_l1660_166029

/-- Represents the number of seats in a row of an auditorium -/
def seats (n : ℕ) : ℕ :=
  5 + 2 * (n - 1)

/-- Theorem: The fifteenth row of the auditorium has 33 seats -/
theorem fifteenth_row_seats : seats 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_row_seats_l1660_166029


namespace NUMINAMATH_CALUDE_students_registered_l1660_166066

theorem students_registered (students_yesterday : ℕ) (students_today : ℕ) : ℕ :=
  let students_registered := 156
  let students_absent := 30
  have h1 : students_today + students_absent = students_registered := by sorry
  have h2 : students_today = (2 * students_yesterday * 9) / 10 := by sorry
  students_registered

#check students_registered

end NUMINAMATH_CALUDE_students_registered_l1660_166066


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1660_166023

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a * b * c + a * b * d + a * c * d + b * c * d) / 4 ≤ 
    ((a * b + a * c + a * d + b * c + b * d + c * d) / 6) ^ (3/2) ∧
  ((a * b * c + a * b * d + a * c * d + b * c * d) / 4 = 
    ((a * b + a * c + a * d + b * c + b * d + c * d) / 6) ^ (3/2) ↔ 
      a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1660_166023


namespace NUMINAMATH_CALUDE_square_side_length_l1660_166010

/-- Given a regular triangle and a square with specific perimeter conditions, 
    prove that the side length of the square is 8 cm. -/
theorem square_side_length 
  (triangle_perimeter : ℝ) 
  (total_perimeter : ℝ) 
  (h1 : triangle_perimeter = 46) 
  (h2 : total_perimeter = 78) : 
  (total_perimeter - triangle_perimeter) / 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1660_166010


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l1660_166058

theorem number_puzzle_solution : ∃ x : ℝ, x^2 + 85 = (x - 17)^2 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l1660_166058


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1660_166047

theorem quadratic_rewrite (b : ℝ) (m : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/6 = (x+m)^2 + 1/18) → 
  b = -2/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1660_166047


namespace NUMINAMATH_CALUDE_evaluate_expression_l1660_166063

theorem evaluate_expression (x : ℝ) (h : x = 2) : x^3 + x^2 + x + Real.exp x = 14 + Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1660_166063


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l1660_166001

theorem missing_fraction_sum (x : ℚ) : 
  1/3 + 1/2 + (-5/6) + 1/5 + 1/4 + (-5/6) + x = 5/6 → x = 73/60 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l1660_166001


namespace NUMINAMATH_CALUDE_journey_theorem_l1660_166002

/-- Represents a two-segment journey with different speeds -/
structure Journey where
  time_at_5mph : ℝ
  time_at_15mph : ℝ
  total_time : ℝ
  total_distance : ℝ

/-- The average speed of the entire journey is 10 mph -/
def average_speed (j : Journey) : Prop :=
  j.total_distance / j.total_time = 10

/-- The total time is the sum of time spent at each speed -/
def total_time_sum (j : Journey) : Prop :=
  j.total_time = j.time_at_5mph + j.time_at_15mph

/-- The total distance is the sum of distances covered at each speed -/
def total_distance_sum (j : Journey) : Prop :=
  j.total_distance = 5 * j.time_at_5mph + 15 * j.time_at_15mph

/-- The fraction of time spent at 15 mph is half of the total time -/
def half_time_at_15mph (j : Journey) : Prop :=
  j.time_at_15mph / j.total_time = 1 / 2

theorem journey_theorem (j : Journey) 
  (h1 : average_speed j) 
  (h2 : total_time_sum j) 
  (h3 : total_distance_sum j) : 
  half_time_at_15mph j := by
  sorry

end NUMINAMATH_CALUDE_journey_theorem_l1660_166002


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l1660_166069

/-- Represents the sampling methods --/
inductive SamplingMethod
  | StratifiedSampling
  | LotteryMethod
  | SystematicSampling
  | RandomNumberTableMethod

/-- Represents a school structure --/
structure School where
  num_classes : Nat
  students_per_class : Nat
  student_numbering : Nat → Nat → Nat  -- Class number → Student number → Assigned number

/-- Represents a selection method --/
structure SelectionMethod where
  selected_number : Nat

/-- Determines the sampling method based on school structure and selection method --/
def determineSamplingMethod (school : School) (selection : SelectionMethod) : SamplingMethod :=
  sorry

/-- Theorem stating that the given conditions result in Systematic Sampling --/
theorem systematic_sampling_proof (school : School) (selection : SelectionMethod) :
  school.num_classes = 18 ∧
  school.students_per_class = 56 ∧
  (∀ c s, school.student_numbering c s = s) ∧
  selection.selected_number = 14 →
  determineSamplingMethod school selection = SamplingMethod.SystematicSampling :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l1660_166069


namespace NUMINAMATH_CALUDE_binomial_coefficient_max_l1660_166030

theorem binomial_coefficient_max (n : ℕ) (h : 2^n = 256) : 
  Finset.sup (Finset.range (n + 1)) (fun k => Nat.choose n k) = 70 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_max_l1660_166030


namespace NUMINAMATH_CALUDE_power_of_power_product_simplification_expression_simplification_division_simplification_l1660_166009

-- Problem 1
theorem power_of_power : (3^3)^2 = 3^6 := by sorry

-- Problem 2
theorem product_simplification (x y : ℝ) : (-4*x*y^3)*(-2*x^2) = 8*x^3*y^3 := by sorry

-- Problem 3
theorem expression_simplification (x y : ℝ) : 2*x*(3*y-x^2)+2*x*x^2 = 6*x*y := by sorry

-- Problem 4
theorem division_simplification (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : 
  (20*x^3*y^5-10*x^4*y^4-20*x^3*y^2) / (-5*x^3*y^2) = -4*y^3 + 2*x*y^2 + 4 := by sorry

end NUMINAMATH_CALUDE_power_of_power_product_simplification_expression_simplification_division_simplification_l1660_166009


namespace NUMINAMATH_CALUDE_system_solution_l1660_166035

theorem system_solution (x y z : ℚ) : 
  x = 2/7 ∧ y = 2/5 ∧ z = 2/3 ↔ 
  1/x + 1/y = 6 ∧ 1/y + 1/z = 4 ∧ 1/z + 1/x = 5 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1660_166035


namespace NUMINAMATH_CALUDE_max_product_constrained_sum_l1660_166037

theorem max_product_constrained_sum (a b : ℝ) : 
  a + b = 1 → (∀ x y : ℝ, x + y = 1 → a * b ≥ x * y) → a * b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_sum_l1660_166037


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1660_166038

/-- Calculate the profit percentage given the selling price and profit -/
theorem profit_percentage_calculation
  (selling_price : ℝ)
  (profit : ℝ)
  (h1 : selling_price = 900)
  (h2 : profit = 100) :
  (profit / (selling_price - profit)) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1660_166038


namespace NUMINAMATH_CALUDE_art_club_collection_l1660_166050

/-- Calculates the total number of artworks collected by an art club over multiple school years. -/
def total_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (years : ℕ) : ℕ :=
  students * artworks_per_student_per_quarter * quarters_per_year * years

/-- Proves that an art club with 15 students, each making 2 artworks per quarter, 
    collects 240 artworks in 2 school years with 4 quarters per year. -/
theorem art_club_collection : total_artworks 15 2 4 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_art_club_collection_l1660_166050


namespace NUMINAMATH_CALUDE_broken_sign_distance_l1660_166074

/-- Represents a town in the problem -/
inductive Town
| Atown
| Betown
| Cetown

/-- Represents a signpost along the path -/
structure Signpost where
  distanceToAtown : ℕ
  distanceToCetown : ℕ

/-- The problem setup -/
axiom signpostA : Signpost
axiom signpostB : Signpost
axiom path_through_Betown : True
axiom signpostA_distances : signpostA.distanceToAtown = 7 ∧ signpostA.distanceToCetown = 2
axiom signpostB_distances : signpostB.distanceToAtown = 9 ∧ signpostB.distanceToCetown = 4

/-- The theorem to be proved -/
theorem broken_sign_distance : ∃ (d : ℕ), d = 1 ∧ 
  (d = signpostA.distanceToAtown - signpostB.distanceToAtown + signpostB.distanceToCetown) :=
sorry

end NUMINAMATH_CALUDE_broken_sign_distance_l1660_166074


namespace NUMINAMATH_CALUDE_solutions_equation_1_solutions_equation_2_l1660_166048

-- Equation 1
theorem solutions_equation_1 : 
  ∀ x : ℝ, x^2 - 2*x - 8 = 0 ↔ x = -2 ∨ x = 4 := by sorry

-- Equation 2
theorem solutions_equation_2 : 
  ∀ x : ℝ, x*(x-1) + 2*(x-1) = 0 ↔ x = 1 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_solutions_equation_1_solutions_equation_2_l1660_166048


namespace NUMINAMATH_CALUDE_construct_triangle_from_excenters_l1660_166095

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the concept of an excenter
def is_excenter (P : Point) (T : Triangle) : Prop :=
  sorry -- Definition of excenter

-- Define the concept of altitude foot
def is_altitude_foot (P : Point) (T : Triangle) : Prop :=
  sorry -- Definition of altitude foot

-- Theorem statement
theorem construct_triangle_from_excenters 
  (A₁ B₁ C₁ : Point) 
  (h_excenters : is_excenter A₁ T ∧ is_excenter B₁ T ∧ is_excenter C₁ T) :
  ∃ (T : Triangle),
    is_altitude_foot T.A (Triangle.mk A₁ B₁ C₁) ∧
    is_altitude_foot T.B (Triangle.mk A₁ B₁ C₁) ∧
    is_altitude_foot T.C (Triangle.mk A₁ B₁ C₁) :=
by
  sorry

end NUMINAMATH_CALUDE_construct_triangle_from_excenters_l1660_166095


namespace NUMINAMATH_CALUDE_lana_extra_tickets_l1660_166098

/-- Calculates the number of extra tickets bought given the ticket price, number of tickets for friends, and total amount spent. -/
def extra_tickets (ticket_price : ℕ) (friends_tickets : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent - friends_tickets * ticket_price) / ticket_price

/-- Proves that Lana bought 2 extra tickets given the problem conditions. -/
theorem lana_extra_tickets :
  let ticket_price : ℕ := 6
  let friends_tickets : ℕ := 8
  let total_spent : ℕ := 60
  extra_tickets ticket_price friends_tickets total_spent = 2 := by
  sorry

end NUMINAMATH_CALUDE_lana_extra_tickets_l1660_166098


namespace NUMINAMATH_CALUDE_equation_system_solution_l1660_166044

theorem equation_system_solution (n p : ℕ) :
  (∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x + p * y = n ∧ x + y = p^2) ↔
  (p > 1 ∧ (n - 1) % (p - 1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l1660_166044


namespace NUMINAMATH_CALUDE_basketball_tournament_handshakes_l1660_166072

/-- Calculates the total number of handshakes in a basketball tournament --/
def total_handshakes (num_teams : ℕ) (players_per_team : ℕ) (num_referees : ℕ) : ℕ :=
  let total_players := num_teams * players_per_team
  let player_handshakes := (total_players * (total_players - players_per_team)) / 2
  let referee_handshakes := total_players * num_referees
  player_handshakes + referee_handshakes

/-- Theorem stating the total number of handshakes in the specific basketball tournament scenario --/
theorem basketball_tournament_handshakes :
  total_handshakes 3 5 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tournament_handshakes_l1660_166072


namespace NUMINAMATH_CALUDE_hypotenuse_length_of_area_49_l1660_166065

/-- An isosceles right triangle with given area and hypotenuse length -/
structure IsoscelesRightTriangle where
  -- The length of a leg
  leg : ℝ
  -- The area of the triangle
  area : ℝ
  -- The hypotenuse length
  hypotenuse : ℝ
  -- Condition: The area is (1/2) * leg^2
  area_eq : area = (1/2) * leg^2
  -- Condition: The hypotenuse is √2 times the leg
  hypotenuse_eq : hypotenuse = Real.sqrt 2 * leg

/-- The main theorem: If the area is 49, then the hypotenuse length is 14 -/
theorem hypotenuse_length_of_area_49 (t : IsoscelesRightTriangle) (h : t.area = 49) :
  t.hypotenuse = 14 := by
  sorry


end NUMINAMATH_CALUDE_hypotenuse_length_of_area_49_l1660_166065


namespace NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l1660_166061

theorem ceiling_negative_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l1660_166061


namespace NUMINAMATH_CALUDE_probability_green_second_is_three_fifths_l1660_166078

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  white : ℕ := 0
  black : ℕ := 0
  red : ℕ := 0
  green : ℕ := 0

/-- Calculate the probability of drawing a green marble as the second marble -/
def probabilityGreenSecond (bagX bagY bagZ : BagContents) : ℚ :=
  let probWhiteX := bagX.white / (bagX.white + bagX.black)
  let probBlackX := bagX.black / (bagX.white + bagX.black)
  let probGreenY := bagY.green / (bagY.red + bagY.green)
  let probGreenZ := bagZ.green / (bagZ.red + bagZ.green)
  probWhiteX * probGreenY + probBlackX * probGreenZ

/-- The main theorem to prove -/
theorem probability_green_second_is_three_fifths :
  let bagX := BagContents.mk 5 5 0 0
  let bagY := BagContents.mk 0 0 7 8
  let bagZ := BagContents.mk 0 0 3 6
  probabilityGreenSecond bagX bagY bagZ = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_probability_green_second_is_three_fifths_l1660_166078


namespace NUMINAMATH_CALUDE_last_car_speed_l1660_166062

theorem last_car_speed (n : ℕ) (first_speed last_speed : ℕ) : 
  n = 31 ∧ 
  first_speed = 61 ∧ 
  last_speed = 91 ∧ 
  last_speed - first_speed + 1 = n → 
  first_speed + ((n + 1) / 2 - 1) = 76 :=
by sorry

end NUMINAMATH_CALUDE_last_car_speed_l1660_166062


namespace NUMINAMATH_CALUDE_cooper_remaining_pies_l1660_166099

/-- The number of apple pies Cooper makes per day -/
def pies_per_day : ℕ := 7

/-- The number of days Cooper makes pies -/
def days_making_pies : ℕ := 12

/-- The number of pies Ashley eats -/
def pies_eaten : ℕ := 50

/-- The number of pies remaining with Cooper -/
def remaining_pies : ℕ := pies_per_day * days_making_pies - pies_eaten

theorem cooper_remaining_pies : remaining_pies = 34 := by sorry

end NUMINAMATH_CALUDE_cooper_remaining_pies_l1660_166099


namespace NUMINAMATH_CALUDE_arithmetic_mean_theorem_l1660_166020

theorem arithmetic_mean_theorem (x a b : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) :
  (1 / 2) * ((x * b + a) / x + (x * b - a) / x) = b := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_theorem_l1660_166020


namespace NUMINAMATH_CALUDE_sum_of_coefficients_factorization_l1660_166006

theorem sum_of_coefficients_factorization (x y : ℝ) : 
  ∃ (a b c d e f g h j k : ℤ),
    27 * x^6 - 512 * y^6 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2) ∧
    a + b + c + d + e + f + g + h + j + k = 55 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_factorization_l1660_166006


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l1660_166068

theorem cylinder_height_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_cylinder_radius : r_cylinder = 3)
  (h_hemisphere_radius : r_hemisphere = 7) (h_inscribed : r_cylinder ≤ r_hemisphere) :
  let height := Real.sqrt (r_hemisphere^2 - r_cylinder^2)
  height = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l1660_166068


namespace NUMINAMATH_CALUDE_six_by_six_checkerboard_half_shaded_l1660_166093

/-- Represents a square grid with checkerboard shading -/
structure CheckerboardGrid :=
  (size : ℕ)
  (startUnshaded : Bool)

/-- Calculates the fraction of shaded squares in a checkerboard grid -/
def shadedFraction (grid : CheckerboardGrid) : ℚ :=
  1/2

/-- Theorem: In a 6x6 checkerboard grid starting with an unshaded square,
    half of the squares are shaded -/
theorem six_by_six_checkerboard_half_shaded :
  let grid : CheckerboardGrid := ⟨6, true⟩
  shadedFraction grid = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_six_by_six_checkerboard_half_shaded_l1660_166093


namespace NUMINAMATH_CALUDE_wayne_shrimp_appetizer_cost_l1660_166053

/-- Calculates the total cost of shrimp for an appetizer given the number of shrimp per guest,
    number of guests, cost per pound of shrimp, and number of shrimp per pound. -/
def shrimp_appetizer_cost (shrimp_per_guest : ℕ) (num_guests : ℕ) (cost_per_pound : ℚ) (shrimp_per_pound : ℕ) : ℚ :=
  (shrimp_per_guest * num_guests : ℚ) / shrimp_per_pound * cost_per_pound

/-- Proves that Wayne's shrimp appetizer will cost $170 given the specified conditions. -/
theorem wayne_shrimp_appetizer_cost :
  shrimp_appetizer_cost 5 40 17 20 = 170 := by
  sorry

end NUMINAMATH_CALUDE_wayne_shrimp_appetizer_cost_l1660_166053


namespace NUMINAMATH_CALUDE_megan_remaining_acorns_l1660_166036

def initial_acorns : ℕ := 16
def acorns_given : ℕ := 7

theorem megan_remaining_acorns :
  initial_acorns - acorns_given = 9 := by
  sorry

end NUMINAMATH_CALUDE_megan_remaining_acorns_l1660_166036


namespace NUMINAMATH_CALUDE_max_vector_sum_on_unit_circle_l1660_166055

theorem max_vector_sum_on_unit_circle :
  let A : ℝ × ℝ := (Real.sqrt 3, 1)
  let O : ℝ × ℝ := (0, 0)
  ∃ (max : ℝ), max = 3 ∧ 
    ∀ (B : ℝ × ℝ), (B.1 - O.1)^2 + (B.2 - O.2)^2 = 1 →
      Real.sqrt ((A.1 - O.1 + B.1 - O.1)^2 + (A.2 - O.2 + B.2 - O.2)^2) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_vector_sum_on_unit_circle_l1660_166055


namespace NUMINAMATH_CALUDE_june_upload_total_l1660_166013

/-- Represents the upload schedule for a YouTuber in June --/
structure UploadSchedule where
  early_june : Nat  -- videos per day from June 1st to June 15th
  mid_june : Nat    -- videos per day from June 16th to June 23rd
  late_june : Nat   -- videos per day from June 24th to June 30th

/-- Calculates the total number of video hours uploaded in June --/
def total_video_hours (schedule : UploadSchedule) : Nat :=
  schedule.early_june * 15 + schedule.mid_june * 8 + schedule.late_june * 7

/-- Theorem stating that the given upload schedule results in 480 total video hours --/
theorem june_upload_total (schedule : UploadSchedule) 
  (h1 : schedule.early_june = 10)
  (h2 : schedule.mid_june = 15)
  (h3 : schedule.late_june = 30) : 
  total_video_hours schedule = 480 := by
  sorry

#eval total_video_hours { early_june := 10, mid_june := 15, late_june := 30 }

end NUMINAMATH_CALUDE_june_upload_total_l1660_166013


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_pi_3_l1660_166043

theorem cos_2alpha_minus_pi_3 (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 4) :
  Real.cos (2 * α - π / 3) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_pi_3_l1660_166043


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1660_166000

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (2, -1)
  parallel a b → m = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1660_166000
