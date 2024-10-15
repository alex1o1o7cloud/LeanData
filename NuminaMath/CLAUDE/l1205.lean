import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1205_120575

theorem quadratic_form_equivalence (x : ℝ) : 
  (2*x - 1) * (x + 2) + 1 = 2*(x + 3/4)^2 - 17/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1205_120575


namespace NUMINAMATH_CALUDE_factorial_divisibility_l1205_120543

theorem factorial_divisibility (n : ℕ) (h : 1 ≤ n ∧ n ≤ 100) :
  ∃ k : ℕ, (Nat.factorial (n^3 - 1)) = k * (Nat.factorial n)^(n + 1) :=
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l1205_120543


namespace NUMINAMATH_CALUDE_darcies_age_l1205_120503

/-- Darcie's age problem -/
theorem darcies_age :
  ∀ (darcie_age mother_age father_age : ℚ),
    darcie_age = (1 / 6 : ℚ) * mother_age →
    mother_age = (4 / 5 : ℚ) * father_age →
    father_age = 30 →
    darcie_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_darcies_age_l1205_120503


namespace NUMINAMATH_CALUDE_joe_paint_usage_l1205_120590

theorem joe_paint_usage (total_paint : ℝ) (second_week_fraction : ℝ) (total_used : ℝ) :
  total_paint = 360 →
  second_week_fraction = 1 / 7 →
  total_used = 128.57 →
  ∃ (first_week_fraction : ℝ),
    first_week_fraction * total_paint +
    second_week_fraction * (total_paint - first_week_fraction * total_paint) = total_used ∧
    first_week_fraction = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_joe_paint_usage_l1205_120590


namespace NUMINAMATH_CALUDE_prime_power_plus_one_mod_240_l1205_120567

theorem prime_power_plus_one_mod_240 (n : ℕ+) (h : Nat.Prime (2^n.val + 1)) :
  (2^n.val + 1) % 240 = 17 ∨ (2^n.val + 1) % 240 = 3 ∨ (2^n.val + 1) % 240 = 5 :=
by sorry

end NUMINAMATH_CALUDE_prime_power_plus_one_mod_240_l1205_120567


namespace NUMINAMATH_CALUDE_distance_to_point_l1205_120582

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 2*y + 6

/-- The center of the circle -/
def circle_center : ℝ × ℝ := sorry

/-- The distance between two points in 2D space -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distance between the center of the circle and (10, 3) is √68 -/
theorem distance_to_point : distance circle_center (10, 3) = Real.sqrt 68 := by sorry

end NUMINAMATH_CALUDE_distance_to_point_l1205_120582


namespace NUMINAMATH_CALUDE_classroom_students_l1205_120554

theorem classroom_students (T : ℕ) (S : ℕ) (n : ℕ) : 
  (T = S / n + 24) →  -- Teacher's age is 24 years more than average student age
  (T = (T + S) / (n + 1) + 20) →  -- Teacher's age is 20 years more than average age of everyone
  (n = 5) := by  -- Number of students is 5
sorry

end NUMINAMATH_CALUDE_classroom_students_l1205_120554


namespace NUMINAMATH_CALUDE_pyramid_block_count_l1205_120520

/-- 
Represents a four-layer pyramid where each layer has three times as many blocks 
as the layer above it, with the top layer being a single block.
-/
def PyramidBlocks : ℕ → ℕ
| 0 => 1  -- Top layer
| n + 1 => 3 * PyramidBlocks n  -- Each subsequent layer

/-- The total number of blocks in the four-layer pyramid -/
def TotalBlocks : ℕ := 
  PyramidBlocks 0 + PyramidBlocks 1 + PyramidBlocks 2 + PyramidBlocks 3

theorem pyramid_block_count : TotalBlocks = 40 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_block_count_l1205_120520


namespace NUMINAMATH_CALUDE_g_g_2_equals_263_l1205_120528

def g (x : ℝ) : ℝ := 2 * x^2 + 2 * x - 1

theorem g_g_2_equals_263 : g (g 2) = 263 := by
  sorry

end NUMINAMATH_CALUDE_g_g_2_equals_263_l1205_120528


namespace NUMINAMATH_CALUDE_counterexample_exists_l1205_120544

theorem counterexample_exists : ∃ n : ℕ, 
  2 ∣ n ∧ ¬ Nat.Prime n ∧ Nat.Prime (n - 3) := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1205_120544


namespace NUMINAMATH_CALUDE_food_relation_values_l1205_120535

def A : Set ℝ := {-1, 1/2, 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 1 ∧ a ≥ 0}

def is_full_food (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def is_partial_food (X Y : Set ℝ) : Prop :=
  (∃ x, x ∈ X ∧ x ∈ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

theorem food_relation_values :
  ∀ a : ℝ, (is_full_food A (B a) ∨ is_partial_food A (B a)) ↔ (a = 0 ∨ a = 1 ∨ a = 4) :=
sorry

end NUMINAMATH_CALUDE_food_relation_values_l1205_120535


namespace NUMINAMATH_CALUDE_work_completion_time_l1205_120593

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 40

/-- The number of days it takes A and B to complete the work together -/
def ab_days : ℝ := 24

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 60

/-- Theorem stating that if A can do the work in 40 days and A and B together can do it in 24 days, 
    then B can do the work alone in 60 days -/
theorem work_completion_time : 
  (1 / a_days + 1 / b_days = 1 / ab_days) ∧ (b_days = 60) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1205_120593


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_neg_two_l1205_120591

theorem fraction_zero_implies_x_neg_two (x : ℝ) :
  (|x| - 2) / (x^2 - x - 2) = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_neg_two_l1205_120591


namespace NUMINAMATH_CALUDE_sufficient_condition_absolute_value_l1205_120525

theorem sufficient_condition_absolute_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_absolute_value_l1205_120525


namespace NUMINAMATH_CALUDE_max_ab_is_nine_l1205_120537

/-- The function f(x) defined in the problem -/
def f (a b : ℝ) (x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 12 * x^2 - 2 * a * x - 2 * b

/-- The second derivative of f(x) -/
def f'' (a : ℝ) (x : ℝ) : ℝ := 24 * x - 2 * a

theorem max_ab_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f' a b 1 = 0) : 
  (∃ (max_ab : ℝ), max_ab = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → f' a' b' 1 = 0 → a' * b' ≤ max_ab) :=
sorry

end NUMINAMATH_CALUDE_max_ab_is_nine_l1205_120537


namespace NUMINAMATH_CALUDE_nonzero_even_from_second_step_l1205_120572

/-- Represents a bi-infinite sequence of integers -/
def BiInfiniteSequence := ℤ → ℤ

/-- The initial sequence with one 1 and all other elements 0 -/
def initial_sequence : BiInfiniteSequence :=
  fun i => if i = 0 then 1 else 0

/-- The next sequence after one step of evolution -/
def next_sequence (s : BiInfiniteSequence) : BiInfiniteSequence :=
  fun i => s (i - 1) + s i + s (i + 1)

/-- The sequence after n steps of evolution -/
def evolved_sequence (n : ℕ) : BiInfiniteSequence :=
  match n with
  | 0 => initial_sequence
  | m + 1 => next_sequence (evolved_sequence m)

/-- Predicate to check if a sequence contains a non-zero even number -/
def contains_nonzero_even (s : BiInfiniteSequence) : Prop :=
  ∃ i : ℤ, s i ≠ 0 ∧ s i % 2 = 0

/-- The main theorem to be proved -/
theorem nonzero_even_from_second_step :
  ∀ n : ℕ, n ≥ 2 → contains_nonzero_even (evolved_sequence n) :=
sorry

end NUMINAMATH_CALUDE_nonzero_even_from_second_step_l1205_120572


namespace NUMINAMATH_CALUDE_positive_x_axis_line_m_range_l1205_120556

/-- A line passing through the positive half-axis of the x-axis -/
structure PositiveXAxisLine where
  m : ℝ
  equation : ℝ → ℝ
  equation_def : ∀ x, equation x = 2 * x + m - 3
  passes_positive_x : ∃ x > 0, equation x = 0

/-- The range of m for a line passing through the positive half-axis of the x-axis -/
theorem positive_x_axis_line_m_range (line : PositiveXAxisLine) : line.m < 3 := by
  sorry


end NUMINAMATH_CALUDE_positive_x_axis_line_m_range_l1205_120556


namespace NUMINAMATH_CALUDE_pyramid_max_volume_l1205_120566

theorem pyramid_max_volume (a b c h : ℝ) (angle : ℝ) :
  a = 5 ∧ b = 12 ∧ c = 13 →
  a^2 + b^2 = c^2 →
  angle ≥ 30 * π / 180 →
  (∃ (height : ℝ), height > 0 ∧
    (∀ (face_height : ℝ), face_height > 0 →
      Real.cos (Real.arccos (height / face_height)) ≥ Real.cos angle)) →
  (1/3 : ℝ) * (1/2 * a * b) * h ≤ 150 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_max_volume_l1205_120566


namespace NUMINAMATH_CALUDE_no_natural_solution_l1205_120564

theorem no_natural_solution : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l1205_120564


namespace NUMINAMATH_CALUDE_math_team_combinations_l1205_120580

theorem math_team_combinations : 
  let total_girls : ℕ := 4
  let total_boys : ℕ := 6
  let girls_on_team : ℕ := 3
  let boys_on_team : ℕ := 2
  (total_girls.choose girls_on_team) * (total_boys.choose boys_on_team) = 60 := by
sorry

end NUMINAMATH_CALUDE_math_team_combinations_l1205_120580


namespace NUMINAMATH_CALUDE_vertex_not_zero_l1205_120522

/-- The vertex of a quadratic function y = x^2 - (m-2)x + 4 lies on a coordinate axis if and only if
    m = 2 or m = -2 or m = 6 -/
def vertex_on_axis (m : ℝ) : Prop :=
  m = 2 ∨ m = -2 ∨ m = 6

/-- If the vertex of the quadratic function y = x^2 - (m-2)x + 4 lies on a coordinate axis,
    then m ≠ 0 -/
theorem vertex_not_zero (m : ℝ) (h : vertex_on_axis m) : m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_vertex_not_zero_l1205_120522


namespace NUMINAMATH_CALUDE_shopkeeper_decks_l1205_120552

theorem shopkeeper_decks (total_red_cards : ℕ) (cards_per_deck : ℕ) (colors_per_deck : ℕ) (red_suits_per_deck : ℕ) (cards_per_suit : ℕ) : 
  total_red_cards = 182 →
  cards_per_deck = 52 →
  colors_per_deck = 2 →
  red_suits_per_deck = 2 →
  cards_per_suit = 13 →
  (total_red_cards / (red_suits_per_deck * cards_per_suit) : ℕ) = 7 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_decks_l1205_120552


namespace NUMINAMATH_CALUDE_gcf_and_lcm_of_numbers_l1205_120549

def numbers : List Nat := [42, 126, 105]

theorem gcf_and_lcm_of_numbers :
  (Nat.gcd (Nat.gcd 42 126) 105 = 21) ∧
  (Nat.lcm (Nat.lcm 42 126) 105 = 630) := by
  sorry

end NUMINAMATH_CALUDE_gcf_and_lcm_of_numbers_l1205_120549


namespace NUMINAMATH_CALUDE_yellow_red_paper_area_comparison_l1205_120584

theorem yellow_red_paper_area_comparison (x : ℝ) (h : x > 0) :
  let yellow_area := 2 * x
  let larger_part := x / (1 - 0.25)
  let smaller_part := yellow_area - larger_part
  (x - smaller_part) / smaller_part = 0.5
  := by sorry

end NUMINAMATH_CALUDE_yellow_red_paper_area_comparison_l1205_120584


namespace NUMINAMATH_CALUDE_fraction_ordering_l1205_120594

theorem fraction_ordering : (7 : ℚ) / 29 < 11 / 33 ∧ 11 / 33 < 13 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1205_120594


namespace NUMINAMATH_CALUDE_grading_multiple_l1205_120589

theorem grading_multiple (total_questions : ℕ) (score : ℕ) (correct_responses : ℕ) :
  total_questions = 100 →
  score = 70 →
  correct_responses = 90 →
  ∃ m : ℕ, score = correct_responses - m * (total_questions - correct_responses) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_grading_multiple_l1205_120589


namespace NUMINAMATH_CALUDE_daves_monday_hours_l1205_120542

/-- 
Given:
- Dave's hourly rate is $6
- Dave worked on Monday and Tuesday
- On Tuesday, Dave worked 2 hours
- Dave made $48 in total for both days

Prove: Dave worked 6 hours on Monday
-/
theorem daves_monday_hours 
  (hourly_rate : ℕ) 
  (tuesday_hours : ℕ) 
  (total_earnings : ℕ) 
  (h1 : hourly_rate = 6)
  (h2 : tuesday_hours = 2)
  (h3 : total_earnings = 48) : 
  ∃ (monday_hours : ℕ), 
    hourly_rate * (monday_hours + tuesday_hours) = total_earnings ∧ 
    monday_hours = 6 := by
  sorry

#check daves_monday_hours

end NUMINAMATH_CALUDE_daves_monday_hours_l1205_120542


namespace NUMINAMATH_CALUDE_outfits_count_l1205_120555

/-- The number of shirts available --/
def num_shirts : ℕ := 8

/-- The number of pairs of pants available --/
def num_pants : ℕ := 5

/-- The number of ties available --/
def num_ties : ℕ := 4

/-- The number of hats available --/
def num_hats : ℕ := 2

/-- The total number of outfit combinations --/
def total_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_hats + 1)

/-- Theorem stating that the total number of outfits is 600 --/
theorem outfits_count : total_outfits = 600 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1205_120555


namespace NUMINAMATH_CALUDE_bird_families_left_l1205_120512

theorem bird_families_left (initial_families : ℕ) (families_flown_away : ℕ) : 
  initial_families = 67 → families_flown_away = 32 → initial_families - families_flown_away = 35 :=
by sorry

end NUMINAMATH_CALUDE_bird_families_left_l1205_120512


namespace NUMINAMATH_CALUDE_billie_bakes_three_pies_l1205_120541

/-- The number of pies Billie bakes per day -/
def pies_per_day : ℕ := sorry

/-- The number of days Billie bakes pies -/
def baking_days : ℕ := 11

/-- The number of cans of whipped cream needed to cover one pie -/
def cans_per_pie : ℕ := 2

/-- The number of pies Tiffany eats -/
def pies_eaten : ℕ := 4

/-- The number of cans of whipped cream needed for the remaining pies -/
def cans_needed : ℕ := 58

theorem billie_bakes_three_pies : 
  pies_per_day * baking_days = pies_eaten + cans_needed / cans_per_pie ∧ 
  pies_per_day = 3 := by sorry

end NUMINAMATH_CALUDE_billie_bakes_three_pies_l1205_120541


namespace NUMINAMATH_CALUDE_sets_intersection_and_union_l1205_120547

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x | (x+2)*(x-3) < 0}

theorem sets_intersection_and_union :
  (A ∩ B = {x : ℝ | -2 < x ∧ x < 1}) ∧
  (A ∪ B = {x : ℝ | -3 < x ∧ x < 3}) := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_and_union_l1205_120547


namespace NUMINAMATH_CALUDE_correct_average_after_error_l1205_120521

theorem correct_average_after_error (n : ℕ) (initial_avg : ℚ) (wrong_mark correct_mark : ℚ) :
  n = 10 →
  initial_avg = 100 →
  wrong_mark = 50 →
  correct_mark = 10 →
  (n : ℚ) * initial_avg - wrong_mark + correct_mark = (n : ℚ) * 96 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_error_l1205_120521


namespace NUMINAMATH_CALUDE_roots_order_l1205_120562

variables (a b m n : ℝ)

-- Define the equation
def f (x : ℝ) : ℝ := 1 - (x - a) * (x - b)

theorem roots_order (h1 : f m = 0) (h2 : f n = 0) (h3 : m < n) (h4 : a < b) :
  m < a ∧ a < b ∧ b < n := by
  sorry

end NUMINAMATH_CALUDE_roots_order_l1205_120562


namespace NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l1205_120546

theorem definite_integral_exp_plus_2x : ∫ x in (0:ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 - 1 := by sorry

end NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l1205_120546


namespace NUMINAMATH_CALUDE_quadratic_roots_equal_irrational_l1205_120501

theorem quadratic_roots_equal_irrational (d : ℝ) :
  let a : ℝ := 3
  let b : ℝ := -4 * Real.pi
  let c : ℝ := d
  let discriminant := b^2 - 4*a*c
  discriminant = 16 →
  ∃ (x : ℝ), (a*x^2 + b*x + c = 0 ∧ 
              ∀ (y : ℝ), a*y^2 + b*y + c = 0 → y = x) ∧
             (¬ ∃ (p q : ℤ), x = p / q) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_equal_irrational_l1205_120501


namespace NUMINAMATH_CALUDE_find_principal_amount_l1205_120578

/-- Given compound and simple interest for 2 years, find the principal amount -/
theorem find_principal_amount (compound_interest simple_interest : ℚ) : 
  compound_interest = 11730 → 
  simple_interest = 10200 → 
  ∃ (principal rate : ℚ), 
    principal > 0 ∧ 
    rate > 0 ∧ 
    rate < 100 ∧
    compound_interest = principal * ((1 + rate / 100) ^ 2 - 1) ∧
    simple_interest = principal * rate * 2 / 100 ∧
    principal = 1700 :=
by sorry

end NUMINAMATH_CALUDE_find_principal_amount_l1205_120578


namespace NUMINAMATH_CALUDE_total_weight_moved_l1205_120526

/-- Represents an exercise with weight, reps, and sets -/
structure Exercise where
  weight : Nat
  reps : Nat
  sets : Nat

/-- Calculates the total weight moved for a single exercise -/
def totalWeightForExercise (e : Exercise) : Nat :=
  e.weight * e.reps * e.sets

/-- John's workout routine -/
def workoutRoutine : List Exercise := [
  { weight := 15, reps := 10, sets := 3 },  -- Bench press
  { weight := 12, reps := 8,  sets := 4 },  -- Bicep curls
  { weight := 50, reps := 12, sets := 3 },  -- Squats
  { weight := 80, reps := 6,  sets := 2 }   -- Deadlift
]

/-- Theorem stating the total weight John moves during his workout -/
theorem total_weight_moved : 
  (workoutRoutine.map totalWeightForExercise).sum = 3594 := by
  sorry


end NUMINAMATH_CALUDE_total_weight_moved_l1205_120526


namespace NUMINAMATH_CALUDE_hotdog_cost_l1205_120573

/-- Represents the cost of Sara's lunch items in dollars -/
structure LunchCost where
  total : ℝ
  salad : ℝ
  hotdog : ℝ

/-- Theorem stating that given the total lunch cost and salad cost, the hotdog cost can be determined -/
theorem hotdog_cost (lunch : LunchCost) 
  (h1 : lunch.total = 10.46)
  (h2 : lunch.salad = 5.1)
  (h3 : lunch.total = lunch.salad + lunch.hotdog) : 
  lunch.hotdog = 5.36 := by
  sorry

#check hotdog_cost

end NUMINAMATH_CALUDE_hotdog_cost_l1205_120573


namespace NUMINAMATH_CALUDE_isabel_afternoon_runs_l1205_120506

/-- Calculates the number of afternoon runs given circuit length, morning runs, and total weekly distance -/
def afternoon_runs (circuit_length : ℕ) (morning_runs : ℕ) (total_weekly_distance : ℕ) : ℕ :=
  (total_weekly_distance - 7 * morning_runs * circuit_length) / circuit_length

/-- Proves that Isabel runs the circuit 21 times in the afternoon during a week -/
theorem isabel_afternoon_runs : 
  afternoon_runs 365 7 25550 = 21 := by
  sorry

end NUMINAMATH_CALUDE_isabel_afternoon_runs_l1205_120506


namespace NUMINAMATH_CALUDE_length_of_AB_l1205_120500

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 6 = 1

-- Define the line with slope tan(30°) passing through (3, 0)
def line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - 3)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (16 / 5) * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l1205_120500


namespace NUMINAMATH_CALUDE_weekly_calorie_allowance_l1205_120581

/-- The number of calories to reduce from the average daily allowance to hypothetically live to 100 years old -/
def calorie_reduction : ℕ := 500

/-- The average daily calorie allowance for a person in their 60's -/
def average_daily_allowance : ℕ := 2000

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The weekly calorie allowance for a person in their 60's to hypothetically live to 100 years old -/
theorem weekly_calorie_allowance :
  (average_daily_allowance - calorie_reduction) * days_in_week = 10500 := by
  sorry

end NUMINAMATH_CALUDE_weekly_calorie_allowance_l1205_120581


namespace NUMINAMATH_CALUDE_phone_bill_calculation_l1205_120583

def initial_balance : ℚ := 800
def rent_payment : ℚ := 450
def paycheck_deposit : ℚ := 1500
def electricity_bill : ℚ := 117
def internet_bill : ℚ := 100
def final_balance : ℚ := 1563

theorem phone_bill_calculation : 
  initial_balance - rent_payment + paycheck_deposit - electricity_bill - internet_bill - final_balance = 70 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_calculation_l1205_120583


namespace NUMINAMATH_CALUDE_ellipse_param_sum_l1205_120596

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to foci -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  distance_sum : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, compute its parameters -/
def compute_ellipse_params (e : Ellipse) : EllipseParams :=
  sorry

/-- The main theorem: sum of ellipse parameters equals 14 -/
theorem ellipse_param_sum (e : Ellipse) : 
  let ep := compute_ellipse_params e
  e.F₁ = (0, 2) → e.F₂ = (6, 2) → e.distance_sum = 10 →
  ep.h + ep.k + ep.a + ep.b = 14 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_param_sum_l1205_120596


namespace NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l1205_120563

/-- Represents a chessboard configuration -/
structure ChessBoard where
  size : Nat
  rooks : Nat

/-- Calculates the minimum number of attacking rook pairs on a chessboard -/
def minAttackingPairs (board : ChessBoard) : Nat :=
  sorry

/-- Theorem: The minimum number of attacking rook pairs on an 8x8 board with 16 rooks is 16 -/
theorem min_attacking_pairs_8x8_16rooks :
  ∀ (board : ChessBoard),
    board.size = 8 ∧ board.rooks = 16 →
    minAttackingPairs board = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l1205_120563


namespace NUMINAMATH_CALUDE_solution_exists_l1205_120561

def f (x : ℝ) : ℝ := 2 * x - 3

def d : ℝ := 2

theorem solution_exists : ∃ x : ℝ, 2 * (f x) - 11 = f (x - d) :=
  sorry

end NUMINAMATH_CALUDE_solution_exists_l1205_120561


namespace NUMINAMATH_CALUDE_max_container_volume_l1205_120504

/-- The volume of an open-top container made from a rectangular sheet metal --/
def container_volume (l w h : ℝ) : ℝ := h * (l - 2*h) * (w - 2*h)

/-- The theorem stating the maximum volume of the container --/
theorem max_container_volume :
  let l : ℝ := 90
  let w : ℝ := 48
  ∃ (h : ℝ), 
    (h > 0) ∧ 
    (h < w/2) ∧ 
    (h < l/2) ∧
    (∀ (x : ℝ), x > 0 → x < w/2 → x < l/2 → container_volume l w h ≥ container_volume l w x) ∧
    (container_volume l w h = 16848) ∧
    (h = 6) :=
sorry

end NUMINAMATH_CALUDE_max_container_volume_l1205_120504


namespace NUMINAMATH_CALUDE_sum_of_divisors_143_l1205_120557

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_143 : sum_of_divisors 143 = 168 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_143_l1205_120557


namespace NUMINAMATH_CALUDE_triangle_count_after_12_iterations_l1205_120515

/-- The number of triangles after n iterations of the division process -/
def num_triangles (n : ℕ) : ℕ := 3^n

/-- The side length of triangles after n iterations -/
def side_length (n : ℕ) : ℚ := 1 / 2^n

theorem triangle_count_after_12_iterations :
  num_triangles 12 = 531441 ∧ side_length 12 = 1 / 2^12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_after_12_iterations_l1205_120515


namespace NUMINAMATH_CALUDE_count_random_events_l1205_120516

-- Define the type for events
inductive Event
  | throwDice : Event
  | pearFall : Event
  | winLottery : Event
  | haveBoy : Event
  | waterBoil : Event

-- Define a function to determine if an event is random
def isRandom (e : Event) : Bool :=
  match e with
  | Event.throwDice => true
  | Event.pearFall => false
  | Event.winLottery => true
  | Event.haveBoy => true
  | Event.waterBoil => false

-- Define the list of all events
def allEvents : List Event :=
  [Event.throwDice, Event.pearFall, Event.winLottery, Event.haveBoy, Event.waterBoil]

-- State the theorem
theorem count_random_events :
  (allEvents.filter isRandom).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_random_events_l1205_120516


namespace NUMINAMATH_CALUDE_principal_sum_from_interest_difference_l1205_120568

/-- Proves that for a given interest rate and time period, if the difference between
    compound interest and simple interest is 41, then the principal sum is 4100. -/
theorem principal_sum_from_interest_difference
  (rate : ℝ) (time : ℝ) (diff : ℝ) (p : ℝ) :
  rate = 10 →
  time = 2 →
  diff = 41 →
  diff = p * ((1 + rate / 100) ^ time - 1) - p * (rate * time / 100) →
  p = 4100 := by
  sorry

#check principal_sum_from_interest_difference

end NUMINAMATH_CALUDE_principal_sum_from_interest_difference_l1205_120568


namespace NUMINAMATH_CALUDE_parallel_lines_exist_points_not_on_line_l1205_120508

-- Define the line equation
def line_equation (α x y : ℝ) : Prop :=
  Real.cos α * (x - 2) + Real.sin α * (y + 1) = 1

-- Statement ②: There exist different real numbers α₁, α₂, such that the corresponding lines l₁, l₂ are parallel
theorem parallel_lines_exist : ∃ α₁ α₂ : ℝ, α₁ ≠ α₂ ∧
  ∀ x y : ℝ, line_equation α₁ x y ↔ line_equation α₂ x y :=
sorry

-- Statement ③: There are at least two points in the coordinate plane that are not on the line l
theorem points_not_on_line : ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
  (∀ α : ℝ, ¬line_equation α x₁ y₁ ∧ ¬line_equation α x₂ y₂) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_exist_points_not_on_line_l1205_120508


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_sum_l1205_120530

theorem seventh_root_of_unity_sum (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  ∃ (sign : Bool), z + z^2 + z^4 = (-1 + (if sign then 1 else -1) * Complex.I * Real.sqrt 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_sum_l1205_120530


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_l1205_120588

def sum_odd (n : ℕ) : ℕ := n^2

def sum_even (n : ℕ) : ℕ := n * (n + 1)

theorem odd_even_sum_difference : 
  let n_odd : ℕ := (2023 - 1) / 2 + 1
  let n_even : ℕ := (2022 - 2) / 2 + 1
  sum_odd n_odd - sum_even n_even + 7 - 8 = 47 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_l1205_120588


namespace NUMINAMATH_CALUDE_odd_monotone_function_range_theorem_l1205_120597

/-- A function that is odd and monotonically increasing on non-negative reals -/
def OddMonotoneFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f x < f y)

/-- The theorem statement -/
theorem odd_monotone_function_range_theorem (f : ℝ → ℝ) (h : OddMonotoneFunction f) :
  {x : ℝ | f (x^2 - x - 1) < f 5} = Set.Ioo (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_monotone_function_range_theorem_l1205_120597


namespace NUMINAMATH_CALUDE_business_income_calculation_l1205_120527

theorem business_income_calculation (spending income : ℚ) (profit : ℚ) : 
  spending / income = 5 / 9 →
  profit = income - spending →
  profit = 48000 →
  income = 108000 := by
sorry

end NUMINAMATH_CALUDE_business_income_calculation_l1205_120527


namespace NUMINAMATH_CALUDE_polynomial_composition_l1205_120514

/-- Given f(x) = x² and f(h(x)) = 9x² + 6x + 1, prove that h(x) = 3x + 1 or h(x) = -3x - 1 -/
theorem polynomial_composition (f h : ℝ → ℝ) : 
  (∀ x, f x = x^2) → 
  (∀ x, f (h x) = 9*x^2 + 6*x + 1) → 
  (∀ x, h x = 3*x + 1 ∨ h x = -3*x - 1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_composition_l1205_120514


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1205_120586

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = Complex.abs (3 - 4*I)) :
  z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1205_120586


namespace NUMINAMATH_CALUDE_lcm_gcf_relation_l1205_120574

theorem lcm_gcf_relation (n : ℕ) :
  (Nat.lcm n 12 = 42) ∧ (Nat.gcd n 12 = 6) → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relation_l1205_120574


namespace NUMINAMATH_CALUDE_popcorn_profit_30_bags_l1205_120511

/-- Calculates the profit from selling popcorn bags -/
def popcorn_profit (buy_price sell_price : ℕ) (num_bags : ℕ) : ℕ :=
  (sell_price - buy_price) * num_bags

theorem popcorn_profit_30_bags :
  popcorn_profit 4 8 30 = 120 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_profit_30_bags_l1205_120511


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l1205_120587

/-- Given two integers u and v with 0 < v < u, and points A, B, C, D, E defined as follows:
    A = (u,v)
    B is the reflection of A across y = x
    C is the reflection of B across y = -x
    D is the reflection of C across the x-axis
    E is the reflection of D across the y-axis
    If the area of pentagon ABCDE is 615, then u + v = 45. -/
theorem pentagon_area_sum (u v : ℤ) (hu : u > 0) (hv : v > 0) (huv : u > v) : 
  let A := (u, v)
  let B := (v, u)
  let C := (-u, v)
  let D := (-u, -v)
  let E := (u, -v)
  let area := u^2 + 3*u*v
  area = 615 → u + v = 45 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l1205_120587


namespace NUMINAMATH_CALUDE_profit_and_maximum_l1205_120533

noncomputable section

-- Define the sales volume function
def p (x : ℝ) : ℝ := 3 - 2 / (x + 1)

-- Define the profit function
def y (x : ℝ) : ℝ := 16 - 4 / (x + 1) - x

-- Theorem for the profit function and its maximum
theorem profit_and_maximum (a : ℝ) (h_a : a > 0) :
  -- The profit function
  (∀ x, 0 ≤ x ∧ x ≤ a → y x = 16 - 4 / (x + 1) - x) ∧
  -- Maximum profit when a ≥ 1
  (a ≥ 1 → ∃ x, 0 ≤ x ∧ x ≤ a ∧ y x = 13 ∧ ∀ x', 0 ≤ x' ∧ x' ≤ a → y x' ≤ y x) ∧
  -- Maximum profit when a < 1
  (a < 1 → ∃ x, 0 ≤ x ∧ x ≤ a ∧ y x = 16 - 4 / (a + 1) - a ∧ ∀ x', 0 ≤ x' ∧ x' ≤ a → y x' ≤ y x) :=
sorry

end

end NUMINAMATH_CALUDE_profit_and_maximum_l1205_120533


namespace NUMINAMATH_CALUDE_rounding_accuracy_l1205_120507

-- Define the rounded number
def rounded_number : ℝ := 5.8 * 10^5

-- Define the accuracy levels
inductive AccuracyLevel
  | Tenth
  | Hundredth
  | Thousandth
  | TenThousandth
  | HundredThousandth

-- Define a function to determine the accuracy level
def determine_accuracy (x : ℝ) : AccuracyLevel :=
  match x with
  | _ => AccuracyLevel.TenThousandth -- We know this is the correct answer from the problem

-- State the theorem
theorem rounding_accuracy :
  determine_accuracy rounded_number = AccuracyLevel.TenThousandth :=
by sorry

end NUMINAMATH_CALUDE_rounding_accuracy_l1205_120507


namespace NUMINAMATH_CALUDE_count_divisors_of_360_l1205_120569

theorem count_divisors_of_360 : Finset.card (Nat.divisors 360) = 24 := by
  sorry

end NUMINAMATH_CALUDE_count_divisors_of_360_l1205_120569


namespace NUMINAMATH_CALUDE_mingyoungs_math_score_l1205_120509

theorem mingyoungs_math_score 
  (korean : ℝ) 
  (english : ℝ) 
  (math : ℝ) 
  (h1 : (korean + english) / 2 = 89) 
  (h2 : (korean + english + math) / 3 = 91) : 
  math = 95 :=
sorry

end NUMINAMATH_CALUDE_mingyoungs_math_score_l1205_120509


namespace NUMINAMATH_CALUDE_work_completion_time_l1205_120510

/-- The number of days it takes for person A to complete the work -/
def days_A : ℝ := 18

/-- The fraction of work completed by A and B together in 2 days -/
def work_completed_2_days : ℝ := 0.19444444444444442

/-- The number of days it takes for person B to complete the work -/
def days_B : ℝ := 24

theorem work_completion_time :
  (1 / days_A + 1 / days_B) * 2 = work_completed_2_days :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1205_120510


namespace NUMINAMATH_CALUDE_minimal_point_in_rectangle_l1205_120592

/-- Given positive real numbers a and b, the point (a/2, b/2) minimizes the sum of distances
    to the corners of the rectangle with vertices at (0,0), (a,0), (0,b), and (a,b). -/
theorem minimal_point_in_rectangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x y, 0 < x → x < a → 0 < y → y < b →
  Real.sqrt (x^2 + y^2) + Real.sqrt (x^2 + (b-y)^2) + 
  Real.sqrt ((a-x)^2 + y^2) + Real.sqrt ((a-x)^2 + (b-y)^2) ≥
  Real.sqrt ((a/2)^2 + (b/2)^2) + Real.sqrt ((a/2)^2 + (b/2)^2) + 
  Real.sqrt ((a/2)^2 + (b/2)^2) + Real.sqrt ((a/2)^2 + (b/2)^2) :=
by sorry


end NUMINAMATH_CALUDE_minimal_point_in_rectangle_l1205_120592


namespace NUMINAMATH_CALUDE_g_100_zeros_l1205_120571

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 150| - |x + 150|

-- Define gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem g_100_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ g 100 a = 0 ∧ g 100 b = 0 ∧
  ∀ (x : ℝ), g 100 x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_g_100_zeros_l1205_120571


namespace NUMINAMATH_CALUDE_perimeter_semicircular_square_l1205_120598

/-- The perimeter of a region bounded by semicircular arcs constructed on the sides of a square with side length 1/π is equal to 2. -/
theorem perimeter_semicircular_square : 
  let side_length : ℝ := 1 / Real.pi
  let semicircle_length : ℝ := Real.pi * side_length / 2
  let num_semicircles : ℕ := 4
  semicircle_length * num_semicircles = 2 := by sorry

end NUMINAMATH_CALUDE_perimeter_semicircular_square_l1205_120598


namespace NUMINAMATH_CALUDE_waiter_tip_earnings_l1205_120539

theorem waiter_tip_earnings (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) (total_earnings : ℕ) : 
  total_customers = 10 →
  non_tipping_customers = 5 →
  tip_amount = 3 →
  total_earnings = (total_customers - non_tipping_customers) * tip_amount →
  total_earnings = 15 :=
by sorry

end NUMINAMATH_CALUDE_waiter_tip_earnings_l1205_120539


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l1205_120540

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 7)) ↔ x ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l1205_120540


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l1205_120532

theorem difference_of_squares_example : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l1205_120532


namespace NUMINAMATH_CALUDE_min_value_expression_l1205_120517

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1205_120517


namespace NUMINAMATH_CALUDE_max_product_of_three_distinct_naturals_l1205_120585

theorem max_product_of_three_distinct_naturals (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c → a + b + c = 48 → a * b * c ≤ 4080 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_three_distinct_naturals_l1205_120585


namespace NUMINAMATH_CALUDE_equidistant_point_sum_of_distances_equidistant_times_l1205_120577

-- Define the points A and B
def A : ℝ := -2
def B : ℝ := 4

-- Define the moving point P
def P (x : ℝ) : ℝ := x

-- Define the distances from P to A and B
def distPA (x : ℝ) : ℝ := |x - A|
def distPB (x : ℝ) : ℝ := |x - B|

-- Define the positions of M and N after t seconds
def M (t : ℝ) : ℝ := A - t
def N (t : ℝ) : ℝ := B - 3*t

-- Define the origin O
def O : ℝ := 0

-- Theorem 1: The point equidistant from A and B
theorem equidistant_point : ∃ x : ℝ, distPA x = distPB x ∧ x = 1 := by sorry

-- Theorem 2: Points where sum of distances from A and B is 8
theorem sum_of_distances : ∃ x₁ x₂ : ℝ, 
  distPA x₁ + distPB x₁ = 8 ∧ 
  distPA x₂ + distPB x₂ = 8 ∧ 
  x₁ = -3 ∧ x₂ = 5 := by sorry

-- Theorem 3: Times when one point is equidistant from the other two
theorem equidistant_times : ∃ t₁ t₂ t₃ t₄ t₅ : ℝ,
  (|M t₁| = |N t₁| ∧ t₁ = 1/2) ∧
  (N t₂ = O ∧ t₂ = 4/3) ∧
  (|N t₃ - O| = |N t₃ - M t₃| ∧ t₃ = 2) ∧
  (M t₄ = N t₄ ∧ t₄ = 3) ∧
  (|M t₅ - O| = |M t₅ - N t₅| ∧ t₅ = 8) := by sorry

end NUMINAMATH_CALUDE_equidistant_point_sum_of_distances_equidistant_times_l1205_120577


namespace NUMINAMATH_CALUDE_angle_measure_l1205_120523

theorem angle_measure (m1 m2 m3 : ℝ) (h1 : m1 = 80) (h2 : m2 = 35) (h3 : m3 = 25) :
  ∃ m4 : ℝ, m4 = 140 ∧ m1 + m2 + m3 + (180 - m4) = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l1205_120523


namespace NUMINAMATH_CALUDE_jack_mopping_time_l1205_120545

/-- Calculates the total time Jack spends mopping and resting given the room sizes and mopping speeds -/
def total_mopping_time (bathroom_size kitchen_size living_room_size : ℕ) 
                       (bathroom_speed kitchen_speed living_room_speed : ℕ) : ℕ :=
  let bathroom_time := (bathroom_size + bathroom_speed - 1) / bathroom_speed
  let kitchen_time := (kitchen_size + kitchen_speed - 1) / kitchen_speed
  let living_room_time := (living_room_size + living_room_speed - 1) / living_room_speed
  let mopping_time := bathroom_time + kitchen_time + living_room_time
  let break_time := 3 * 5 + (bathroom_size + kitchen_size + living_room_size) / 40
  mopping_time + break_time

theorem jack_mopping_time :
  total_mopping_time 24 80 120 8 10 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_jack_mopping_time_l1205_120545


namespace NUMINAMATH_CALUDE_storeroom_items_proof_l1205_120579

/-- Calculates the number of items in the storeroom given the number of restocked items,
    sold items, and total items left in the store. -/
def items_in_storeroom (restocked : ℕ) (sold : ℕ) (total_left : ℕ) : ℕ :=
  total_left - (restocked - sold)

/-- Proves that the number of items in the storeroom is 575 given the specific conditions. -/
theorem storeroom_items_proof :
  items_in_storeroom 4458 1561 3472 = 575 := by
  sorry

#eval items_in_storeroom 4458 1561 3472

end NUMINAMATH_CALUDE_storeroom_items_proof_l1205_120579


namespace NUMINAMATH_CALUDE_product_of_numbers_l1205_120536

theorem product_of_numbers (x y : ℝ) : x + y = 25 ∧ x - y = 7 → x * y = 144 := by sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1205_120536


namespace NUMINAMATH_CALUDE_circle_radius_proof_l1205_120559

theorem circle_radius_proof (x y : ℝ) (h : x + y = 100 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l1205_120559


namespace NUMINAMATH_CALUDE_mosaic_length_l1205_120565

theorem mosaic_length 
  (height_feet : ℝ) 
  (tile_size_inch : ℝ) 
  (total_tiles : ℕ) : ℝ :=
  let height_inch : ℝ := height_feet * 12
  let area_inch_sq : ℝ := total_tiles * tile_size_inch ^ 2
  let length_inch : ℝ := area_inch_sq / height_inch
  let length_feet : ℝ := length_inch / 12
  by
    have h1 : height_feet = 10 := by sorry
    have h2 : tile_size_inch = 1 := by sorry
    have h3 : total_tiles = 21600 := by sorry
    sorry

#check mosaic_length

end NUMINAMATH_CALUDE_mosaic_length_l1205_120565


namespace NUMINAMATH_CALUDE_difference_of_squares_l1205_120553

theorem difference_of_squares (a : ℝ) : a^2 - 100 = (a + 10) * (a - 10) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1205_120553


namespace NUMINAMATH_CALUDE_trip_duration_is_six_hours_l1205_120550

/-- Represents the position of a clock hand in minutes (0-59) -/
def ClockPosition : Type := Fin 60

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Fin 24
  minutes : Fin 60

/-- Returns true if the hour and minute hands coincide at the given time -/
def hands_coincide (t : TimeOfDay) : Prop :=
  (t.hours.val * 5 + t.minutes.val / 12 : ℚ) = t.minutes.val

/-- Returns true if the hour and minute hands form a 180° angle at the given time -/
def hands_opposite (t : TimeOfDay) : Prop :=
  ((t.hours.val * 5 + t.minutes.val / 12 : ℚ) + 30) % 60 = t.minutes.val

/-- The start time of the trip -/
def start_time : TimeOfDay :=
  { hours := 8, minutes := 43 }

/-- The end time of the trip -/
def end_time : TimeOfDay :=
  { hours := 14, minutes := 43 }

theorem trip_duration_is_six_hours :
  hands_coincide start_time →
  hands_opposite end_time →
  start_time.hours.val < 9 →
  end_time.hours.val > 14 ∧ end_time.hours.val < 15 →
  (end_time.hours.val - start_time.hours.val : ℕ) = 6 :=
sorry

end NUMINAMATH_CALUDE_trip_duration_is_six_hours_l1205_120550


namespace NUMINAMATH_CALUDE_curve_is_two_semicircles_l1205_120576

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  |x| - 1 = Real.sqrt (1 - (y - 1)^2)

-- Define a semicircle
def is_semicircle (center_x center_y radius : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2 ∧ x ≥ center_x

-- Theorem statement
theorem curve_is_two_semicircles :
  ∀ x y : ℝ, curve_equation x y ↔
    (is_semicircle 1 1 1 x y ∨ is_semicircle (-1) 1 1 x y) :=
sorry

end NUMINAMATH_CALUDE_curve_is_two_semicircles_l1205_120576


namespace NUMINAMATH_CALUDE_distribute_4_2_l1205_120529

/-- The number of ways to distribute n indistinguishable objects into k indistinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 3 ways to distribute 4 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_4_2 : distribute 4 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distribute_4_2_l1205_120529


namespace NUMINAMATH_CALUDE_train_crossing_time_l1205_120513

/-- The time taken for two trains to cross each other -/
theorem train_crossing_time (length1 length2 speed1 speed2 : ℝ) : 
  length1 = 180 ∧ 
  length2 = 360 ∧ 
  speed1 = 60 * (1000 / 3600) ∧ 
  speed2 = 30 * (1000 / 3600) →
  (length1 + length2) / (speed1 + speed2) = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1205_120513


namespace NUMINAMATH_CALUDE_log_inequality_l1205_120534

theorem log_inequality : 
  Real.log 2 / Real.log 3 < 2/3 ∧ 
  2/3 < Real.log 75 / Real.log 625 ∧ 
  Real.log 75 / Real.log 625 < Real.log 3 / Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1205_120534


namespace NUMINAMATH_CALUDE_distance_to_school_proof_l1205_120519

/-- The distance from Xiaohong's home to school in meters -/
def distance_to_school : ℝ := 2720

/-- The distance dad drove Xiaohong towards school in meters -/
def distance_driven : ℝ := 1000

/-- Total travel time (drive + walk) in minutes -/
def total_travel_time : ℝ := 22.5

/-- Time taken to bike from home to school in minutes -/
def biking_time : ℝ := 40

/-- Xiaohong's walking speed in meters per minute -/
def walking_speed : ℝ := 80

/-- The difference between dad's driving speed and Xiaohong's biking speed in meters per minute -/
def speed_difference : ℝ := 800

theorem distance_to_school_proof :
  ∃ (driving_speed : ℝ),
    driving_speed > 0 ∧
    distance_to_school = distance_driven + walking_speed * (total_travel_time - distance_driven / driving_speed) ∧
    distance_to_school = biking_time * (driving_speed - speed_difference) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_school_proof_l1205_120519


namespace NUMINAMATH_CALUDE_chocolates_gain_percent_l1205_120505

/-- Calculates the gain percent given the number of chocolates at cost price and selling price that are equal in value -/
def gain_percent (cost_chocolates : ℕ) (sell_chocolates : ℕ) : ℚ :=
  ((cost_chocolates : ℚ) / sell_chocolates - 1) * 100

/-- Theorem stating that if the cost price of 81 chocolates equals the selling price of 45 chocolates, the gain percent is 80% -/
theorem chocolates_gain_percent :
  gain_percent 81 45 = 80 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_gain_percent_l1205_120505


namespace NUMINAMATH_CALUDE_shaded_area_of_rectangle_l1205_120518

/-- The area of the shaded part of a rectangle with specific properties -/
theorem shaded_area_of_rectangle (base height total_area : ℝ) : 
  base = 7 →
  height = 4 →
  total_area = 56 →
  total_area - 2 * (base * height / 2) = 28 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_rectangle_l1205_120518


namespace NUMINAMATH_CALUDE_systematic_sampling_in_school_l1205_120531

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | RandomDraw
  | RandomSampling
  | SystematicSampling

/-- Represents a school with classes and students -/
structure School where
  num_classes : Nat
  students_per_class : Nat
  student_numbers : Finset Nat

/-- Represents a sampling scenario -/
structure SamplingScenario where
  school : School
  selected_number : Nat

/-- Determines the sampling method used in a given scenario -/
def determineSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  sorry

theorem systematic_sampling_in_school (scenario : SamplingScenario) :
  scenario.school.num_classes = 35 →
  scenario.school.students_per_class = 56 →
  scenario.school.student_numbers = Finset.range 56 →
  scenario.selected_number = 14 →
  determineSamplingMethod scenario = SamplingMethod.SystematicSampling :=
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_in_school_l1205_120531


namespace NUMINAMATH_CALUDE_groups_formed_equals_seven_l1205_120570

/-- Given a class with boys and girls, and a group size, calculate the number of groups formed. -/
def calculateGroups (boys : ℕ) (girls : ℕ) (groupSize : ℕ) : ℕ :=
  (boys + girls) / groupSize

/-- Theorem: Given 9 boys, 12 girls, and groups of 3 members, 7 groups are formed. -/
theorem groups_formed_equals_seven :
  calculateGroups 9 12 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_groups_formed_equals_seven_l1205_120570


namespace NUMINAMATH_CALUDE_rebecca_pie_slices_l1205_120551

theorem rebecca_pie_slices (total_pies : ℕ) (slices_per_pie : ℕ) 
  (remaining_slices : ℕ) (rebecca_husband_slices : ℕ) 
  (family_friends_percent : ℚ) :
  total_pies = 2 →
  slices_per_pie = 8 →
  remaining_slices = 5 →
  rebecca_husband_slices = 2 →
  family_friends_percent = 1/2 →
  ∃ (rebecca_initial_slices : ℕ),
    rebecca_initial_slices = total_pies * slices_per_pie - 
      ((remaining_slices + rebecca_husband_slices) / family_friends_percent) :=
by sorry

end NUMINAMATH_CALUDE_rebecca_pie_slices_l1205_120551


namespace NUMINAMATH_CALUDE_sons_age_l1205_120502

/-- Proves that given the conditions, the son's present age is 22 years -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l1205_120502


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l1205_120524

theorem largest_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x^2 + y^2 = 1) :
  ∃ c : ℝ, c = 1/2 ∧ x^6 + y^6 ≥ c * x * y ∧ ∀ d : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = 1 → a^6 + b^6 ≥ d * a * b) → d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l1205_120524


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1205_120538

theorem train_speed_calculation (rail_length : Real) (time_period : Real) : 
  rail_length = 40 ∧ time_period = 30 / 60 →
  ∃ (ε : Real), ε > 0 ∧ ∀ (train_speed : Real),
    train_speed > 0 →
    |train_speed - (train_speed * 5280 / 60 / rail_length * time_period)| < ε :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1205_120538


namespace NUMINAMATH_CALUDE_smallest_angle_in_345_ratio_triangle_l1205_120548

theorem smallest_angle_in_345_ratio_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  b = (4/3) * a →
  c = (5/3) * a →
  a = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_345_ratio_triangle_l1205_120548


namespace NUMINAMATH_CALUDE_find_carols_number_l1205_120558

/-- A prime number between 10 and 99, inclusive. -/
def TwoDigitPrime := {p : Nat // p.Prime ∧ 10 ≤ p ∧ p ≤ 99}

/-- The problem statement -/
theorem find_carols_number 
  (a b c : TwoDigitPrime) 
  (h1 : b.val + c.val = 14)
  (h2 : a.val + c.val = 20)
  (h3 : a.val + b.val = 18)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  c.val = 11 := by
  sorry

#check find_carols_number

end NUMINAMATH_CALUDE_find_carols_number_l1205_120558


namespace NUMINAMATH_CALUDE_speed_ratio_inverse_of_time_ratio_l1205_120599

/-- Proves that the ratio of speeds for two runners completing the same race
    is the inverse of the ratio of their completion times. -/
theorem speed_ratio_inverse_of_time_ratio
  (total_time : ℝ)
  (rickey_time : ℝ)
  (prejean_time : ℝ)
  (h1 : total_time = rickey_time + prejean_time)
  (h2 : rickey_time = 40)
  (h3 : total_time = 70)
  : (prejean_time / rickey_time) = (3 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_inverse_of_time_ratio_l1205_120599


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_l1205_120560

theorem rectangular_box_dimensions (A B C : ℝ) : 
  A > 0 ∧ B > 0 ∧ C > 0 →
  A * B = 40 →
  A * C = 90 →
  B * C = 100 →
  A + B + C = 83 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_l1205_120560


namespace NUMINAMATH_CALUDE_start_with_any_digits_l1205_120595

theorem start_with_any_digits :
  ∀ (A : ℕ), ∃ (n m : ℕ), 10^m * A ≤ 2^n ∧ 2^n < 10^m * (A + 1) :=
sorry

end NUMINAMATH_CALUDE_start_with_any_digits_l1205_120595
