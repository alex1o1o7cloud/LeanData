import Mathlib

namespace NUMINAMATH_CALUDE_exists_points_with_longer_inner_vector_sum_l957_95712

/-- A regular polygon with 1976 sides -/
structure RegularPolygon1976 where
  vertices : Fin 1976 → ℝ × ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is inside the regular 1976-gon -/
def isInside (p : Point) (poly : RegularPolygon1976) : Prop :=
  sorry

/-- Checks if a point is outside the regular 1976-gon -/
def isOutside (p : Point) (poly : RegularPolygon1976) : Prop :=
  sorry

/-- Sum of vectors from a point to all vertices of the 1976-gon -/
def vectorSum (p : Point) (poly : RegularPolygon1976) : ℝ × ℝ :=
  sorry

/-- Length of a 2D vector -/
def vectorLength (v : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating the existence of points A and B satisfying the conditions -/
theorem exists_points_with_longer_inner_vector_sum (poly : RegularPolygon1976) :
  ∃ (A B : Point),
    isInside A poly ∧
    isOutside B poly ∧
    vectorLength (vectorSum A poly) > vectorLength (vectorSum B poly) :=
  sorry

end NUMINAMATH_CALUDE_exists_points_with_longer_inner_vector_sum_l957_95712


namespace NUMINAMATH_CALUDE_stating_bacteria_fill_time_l957_95751

/-- 
Represents the time (in minutes) it takes to fill a bottle with bacteria,
given the initial number of bacteria and their division rate.
-/
def fill_time (initial_bacteria : ℕ) (a : ℕ) : ℕ :=
  if initial_bacteria = 1 then a
  else a - 1

/-- 
Theorem stating that if one bacterium fills a bottle in 'a' minutes,
then two bacteria will fill the same bottle in 'a - 1' minutes,
given that each bacterium divides into two every minute.
-/
theorem bacteria_fill_time (a : ℕ) (h : a > 0) :
  fill_time 2 a = a - 1 :=
sorry

end NUMINAMATH_CALUDE_stating_bacteria_fill_time_l957_95751


namespace NUMINAMATH_CALUDE_card_area_problem_l957_95724

theorem card_area_problem (l w : ℝ) (h1 : l = 8) (h2 : w = 3) 
  (h3 : (l - 2) * w = 15) : (l * (w - 2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_card_area_problem_l957_95724


namespace NUMINAMATH_CALUDE_candy_bar_profit_l957_95720

theorem candy_bar_profit :
  let total_bars : ℕ := 1200
  let buy_price : ℚ := 3 / 4
  let sell_price : ℚ := 2 / 3
  let discount_threshold : ℕ := 1000
  let discount_per_bar : ℚ := 1 / 10
  let cost : ℚ := total_bars * buy_price
  let revenue_before_discount : ℚ := total_bars * sell_price
  let discounted_bars : ℕ := total_bars - discount_threshold
  let discount : ℚ := discounted_bars * discount_per_bar
  let revenue_after_discount : ℚ := revenue_before_discount - discount
  let profit : ℚ := revenue_after_discount - cost
  profit = -116
:= by sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l957_95720


namespace NUMINAMATH_CALUDE_ratio_problem_l957_95736

/-- Given two ratios a:b:c and c:d:e, prove that a:e is 3:10 -/
theorem ratio_problem (a b c d e : ℚ) 
  (h1 : a / b = 2 / 3 ∧ b / c = 3 / 4)
  (h2 : c / d = 3 / 4 ∧ d / e = 4 / 5) :
  a / e = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l957_95736


namespace NUMINAMATH_CALUDE_cos_420_plus_sin_330_eq_zero_l957_95765

theorem cos_420_plus_sin_330_eq_zero :
  Real.cos (420 * π / 180) + Real.sin (330 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_420_plus_sin_330_eq_zero_l957_95765


namespace NUMINAMATH_CALUDE_calculate_expression_l957_95755

theorem calculate_expression : (1/2)⁻¹ + (Real.pi - 3.14)^0 - |-3| + Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l957_95755


namespace NUMINAMATH_CALUDE_arctanSum_implies_powerSum_l957_95788

theorem arctanSum_implies_powerSum (x y z : ℝ) (n : ℕ) 
  (h1 : x + y + z = 1) 
  (h2 : Real.arctan x + Real.arctan y + Real.arctan z = π / 4) 
  (h3 : n > 0) : 
  x^(2*n+1) + y^(2*n+1) + z^(2*n+1) = 1 := by
sorry

end NUMINAMATH_CALUDE_arctanSum_implies_powerSum_l957_95788


namespace NUMINAMATH_CALUDE_negation_of_no_slow_learners_attend_l957_95773

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (slow_learner : U → Prop)
variable (attends_school : U → Prop)

-- State the theorem
theorem negation_of_no_slow_learners_attend (h : ¬∃ x, slow_learner x ∧ attends_school x) :
  ∃ x, slow_learner x ∧ attends_school x ↔ ¬(¬∃ x, slow_learner x ∧ attends_school x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_no_slow_learners_attend_l957_95773


namespace NUMINAMATH_CALUDE_kathryn_gave_skittles_l957_95772

def cheryl_start : ℕ := 8
def cheryl_end : ℕ := 97

theorem kathryn_gave_skittles : cheryl_end - cheryl_start = 89 := by
  sorry

end NUMINAMATH_CALUDE_kathryn_gave_skittles_l957_95772


namespace NUMINAMATH_CALUDE_sum_of_numbers_l957_95735

theorem sum_of_numbers : 2 * 2143 + 4321 + 3214 + 1432 = 13523 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l957_95735


namespace NUMINAMATH_CALUDE_sum_property_unique_l957_95771

/-- The property that the sum of the first n natural numbers can be written as n followed by three digits in base 10 -/
def sum_property (n : ℕ) : Prop :=
  ∃ k : ℕ, k < 1000 ∧ (n * (n + 1)) / 2 = 1000 * n + k

/-- Theorem stating that 1999 is the only natural number satisfying the sum property -/
theorem sum_property_unique : ∀ n : ℕ, sum_property n ↔ n = 1999 :=
sorry

end NUMINAMATH_CALUDE_sum_property_unique_l957_95771


namespace NUMINAMATH_CALUDE_road_project_solution_l957_95723

/-- Road construction project parameters -/
structure RoadProject where
  total_length : ℝ
  small_eq_rate : ℝ
  large_eq_rate : ℝ
  large_eq_time_ratio : ℝ
  length_increase : ℝ
  small_eq_time_increase : ℝ
  large_eq_rate_decrease : ℝ
  large_eq_time_increase : ℝ → ℝ

/-- Theorem stating the correct small equipment usage time and the value of m -/
theorem road_project_solution (project : RoadProject)
  (h1 : project.total_length = 39000)
  (h2 : project.small_eq_rate = 30)
  (h3 : project.large_eq_rate = 60)
  (h4 : project.large_eq_time_ratio = 5/3)
  (h5 : project.length_increase = 9000)
  (h6 : project.small_eq_time_increase = 18)
  (h7 : project.large_eq_time_increase = λ m => 150 + 2*m) :
  ∃ (small_eq_time m : ℝ),
    small_eq_time = 300 ∧
    m = 5 ∧
    project.small_eq_rate * small_eq_time +
    project.large_eq_rate * (project.large_eq_time_ratio * small_eq_time) = project.total_length ∧
    project.small_eq_rate * (small_eq_time + project.small_eq_time_increase) +
    (project.large_eq_rate - m) * (project.large_eq_time_ratio * small_eq_time + project.large_eq_time_increase m) =
    project.total_length + project.length_increase :=
sorry

end NUMINAMATH_CALUDE_road_project_solution_l957_95723


namespace NUMINAMATH_CALUDE_local_max_range_l957_95799

def f' (x a : ℝ) : ℝ := a * (x + 1) * (x - a)

theorem local_max_range (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, (deriv f) x = f' x a)
  (h2 : IsLocalMax f a) :
  -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_local_max_range_l957_95799


namespace NUMINAMATH_CALUDE_jessica_quarters_l957_95775

/-- Calculates the number of quarters Jessica has after her sister borrows some. -/
def quarters_remaining (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that if Jessica had 8 quarters initially and her sister borrowed 3,
    then Jessica now has 5 quarters. -/
theorem jessica_quarters :
  quarters_remaining 8 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_quarters_l957_95775


namespace NUMINAMATH_CALUDE_parabolas_intersection_l957_95764

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x_coords : Set ℝ :=
  {x | 2 * x^2 - 7 * x + 1 = 5 * x^2 - 2 * x - 2}

/-- The intersection points of two parabolas -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | p.1 ∈ intersection_x_coords ∧ p.2 = 2 * p.1^2 - 7 * p.1 + 1}

theorem parabolas_intersection :
  intersection_points = {((5 - Real.sqrt 61) / 6, 2 * ((5 - Real.sqrt 61) / 6)^2 - 7 * ((5 - Real.sqrt 61) / 6) + 1),
                         ((5 + Real.sqrt 61) / 6, 2 * ((5 + Real.sqrt 61) / 6)^2 - 7 * ((5 + Real.sqrt 61) / 6) + 1)} :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l957_95764


namespace NUMINAMATH_CALUDE_chord_length_l957_95730

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l957_95730


namespace NUMINAMATH_CALUDE_graveyard_bones_count_l957_95710

def total_skeletons : ℕ := 20
def adult_woman_bones : ℕ := 20

theorem graveyard_bones_count :
  let adult_women := total_skeletons / 2
  let adult_men := (total_skeletons - adult_women) / 2
  let children := (total_skeletons - adult_women) / 2
  let adult_man_bones := adult_woman_bones + 5
  let child_bones := adult_woman_bones / 2
  adult_women * adult_woman_bones +
  adult_men * adult_man_bones +
  children * child_bones = 375 := by
sorry

end NUMINAMATH_CALUDE_graveyard_bones_count_l957_95710


namespace NUMINAMATH_CALUDE_same_end_word_count_l957_95784

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- A four-letter word with the same first and last letter -/
structure SameEndWord :=
  (first : Fin alphabet_size)
  (second : Fin alphabet_size)
  (third : Fin alphabet_size)

/-- The count of all possible SameEndWords -/
def count_same_end_words : ℕ := alphabet_size * alphabet_size * alphabet_size

theorem same_end_word_count :
  count_same_end_words = 17576 :=
sorry

end NUMINAMATH_CALUDE_same_end_word_count_l957_95784


namespace NUMINAMATH_CALUDE_intersection_complement_empty_l957_95786

/-- The set of all non-zero real numbers -/
def P : Set ℝ := {x : ℝ | x ≠ 0}

/-- The set of all positive real numbers -/
def Q : Set ℝ := {x : ℝ | x > 0}

/-- Theorem stating that the intersection of Q and the complement of P in ℝ is empty -/
theorem intersection_complement_empty : Q ∩ (Set.univ \ P) = ∅ := by sorry

end NUMINAMATH_CALUDE_intersection_complement_empty_l957_95786


namespace NUMINAMATH_CALUDE_max_profit_theorem_l957_95727

/-- Represents the sales data for a week -/
structure WeekData where
  modelA : ℕ
  modelB : ℕ
  revenue : ℕ

/-- Represents the appliance models -/
inductive Model
| A
| B

def purchase_price (m : Model) : ℕ :=
  match m with
  | Model.A => 180
  | Model.B => 160

def selling_price (m : Model) : ℕ :=
  match m with
  | Model.A => 240
  | Model.B => 200

def profit (m : Model) : ℕ :=
  selling_price m - purchase_price m

def total_units : ℕ := 35

def max_budget : ℕ := 6000

def profit_goal : ℕ := 1750

def week1_data : WeekData := ⟨3, 2, 1120⟩

def week2_data : WeekData := ⟨4, 3, 1560⟩

/-- Calculates the total profit for a given number of units of each model -/
def total_profit (units_A units_B : ℕ) : ℕ :=
  units_A * profit Model.A + units_B * profit Model.B

/-- Calculates the total purchase cost for a given number of units of each model -/
def total_cost (units_A units_B : ℕ) : ℕ :=
  units_A * purchase_price Model.A + units_B * purchase_price Model.B

theorem max_profit_theorem :
  ∃ (units_A units_B : ℕ),
    units_A + units_B = total_units ∧
    total_cost units_A units_B ≤ max_budget ∧
    total_profit units_A units_B > profit_goal ∧
    total_profit units_A units_B = 1800 ∧
    ∀ (x y : ℕ), x + y = total_units → total_cost x y ≤ max_budget →
      total_profit x y ≤ total_profit units_A units_B :=
by sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l957_95727


namespace NUMINAMATH_CALUDE_only_B_is_random_l957_95701

-- Define the events
inductive Event
| A
| B
| C
| D

-- Define a function to check if an event is random
def isRandom (e : Event) : Prop :=
  match e with
  | Event.A => false  -- Water freezing is deterministic
  | Event.B => true   -- Bus arrival is random
  | Event.C => false  -- Sum of 13 is impossible with two dice
  | Event.D => false  -- Pigeonhole principle guarantees same month births

-- Theorem statement
theorem only_B_is_random :
  ∀ e : Event, isRandom e ↔ e = Event.B :=
sorry

end NUMINAMATH_CALUDE_only_B_is_random_l957_95701


namespace NUMINAMATH_CALUDE_distinct_grade_assignments_l957_95794

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of possible grades -/
def num_grades : ℕ := 4

/-- Theorem: The number of distinct ways to assign grades to all students -/
theorem distinct_grade_assignments :
  (num_grades : ℕ) ^ num_students = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_distinct_grade_assignments_l957_95794


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l957_95763

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 1}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l957_95763


namespace NUMINAMATH_CALUDE_at_least_two_positive_l957_95749

theorem at_least_two_positive (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c > 0) (h_prod : a * b + b * c + c * a > 0) :
  (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (c > 0 ∧ a > 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_positive_l957_95749


namespace NUMINAMATH_CALUDE_marbles_distribution_l957_95785

/-- Given 20 marbles distributed equally among 2 boys, prove that each boy receives 10 marbles. -/
theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 20 →
  num_boys = 2 →
  marbles_per_boy * num_boys = total_marbles →
  marbles_per_boy = 10 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l957_95785


namespace NUMINAMATH_CALUDE_divisibility_property_l957_95754

theorem divisibility_property (m : ℤ) (n : ℕ) :
  (10 ∣ (3^n + m)) → (10 ∣ (3^(n+4) + m)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l957_95754


namespace NUMINAMATH_CALUDE_nearest_integer_to_sum_l957_95792

def fraction1 : ℚ := 2007 / 2999
def fraction2 : ℚ := 8001 / 5998
def fraction3 : ℚ := 2001 / 3999

def sum : ℚ := fraction1 + fraction2 + fraction3

theorem nearest_integer_to_sum :
  round sum = 3 := by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_sum_l957_95792


namespace NUMINAMATH_CALUDE_roller_coaster_tickets_l957_95744

/-- Calculates the total number of tickets needed for a group of friends riding roller coasters -/
theorem roller_coaster_tickets (
  first_coaster_cost : ℕ)
  (discount_rate : ℚ)
  (discount_threshold : ℕ)
  (new_coaster_cost : ℕ)
  (num_friends : ℕ)
  (first_coaster_rides : ℕ)
  (new_coaster_rides : ℕ)
  (h1 : first_coaster_cost = 6)
  (h2 : discount_rate = 15 / 100)
  (h3 : discount_threshold = 10)
  (h4 : new_coaster_cost = 8)
  (h5 : num_friends = 8)
  (h6 : first_coaster_rides = 2)
  (h7 : new_coaster_rides = 1)
  : ℕ :=
  160

#check roller_coaster_tickets

end NUMINAMATH_CALUDE_roller_coaster_tickets_l957_95744


namespace NUMINAMATH_CALUDE_purely_imaginary_fraction_l957_95713

theorem purely_imaginary_fraction (a : ℝ) (z : ℂ) :
  z = (a^2 - 1 : ℂ) + (a - 1 : ℂ) * I →
  z.re = 0 →
  z.im ≠ 0 →
  (a + I^2024) / (1 - I) = 0 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_fraction_l957_95713


namespace NUMINAMATH_CALUDE_prob_sum_le_5_is_correct_l957_95731

/-- The probability of the sum of two fair six-sided dice being less than or equal to 5 -/
def prob_sum_le_5 : ℚ :=
  5 / 18

/-- The set of possible outcomes when rolling two dice -/
def dice_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6)

/-- The set of favorable outcomes (sum ≤ 5) when rolling two dice -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  dice_outcomes.filter (fun p => p.1 + p.2 + 2 ≤ 5)

/-- Theorem stating that the probability of the sum of two fair six-sided dice
    being less than or equal to 5 is 5/18 -/
theorem prob_sum_le_5_is_correct :
  (favorable_outcomes.card : ℚ) / dice_outcomes.card = prob_sum_le_5 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_le_5_is_correct_l957_95731


namespace NUMINAMATH_CALUDE_unique_solution_l957_95708

/-- The infinite series representation of the equation -/
def infiniteSeries (x : ℝ) : ℝ := 2 - x + x^2 - x^3 + x^4 - x^5

/-- The condition for series convergence -/
def seriesConverges (x : ℝ) : Prop := abs x < 1

theorem unique_solution : 
  ∃! x : ℝ, (x = infiniteSeries x) ∧ seriesConverges x ∧ x = -1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l957_95708


namespace NUMINAMATH_CALUDE_intersection_of_sets_l957_95798

open Set

theorem intersection_of_sets : 
  let A : Set ℝ := {x | x > 0}
  let B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
  A ∩ B = {x | 0 < x ∧ x ≤ 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l957_95798


namespace NUMINAMATH_CALUDE_cos_minus_sin_special_angle_l957_95717

/-- An angle whose initial side coincides with the non-negative x-axis
    and whose terminal side lies on the ray 4x - 3y = 0 (x ≤ 0) -/
def special_angle (α : Real) : Prop :=
  ∃ (x y : Real), x ≤ 0 ∧ 4 * x - 3 * y = 0 ∧
  Real.cos α = x / Real.sqrt (x^2 + y^2) ∧
  Real.sin α = y / Real.sqrt (x^2 + y^2)

/-- Theorem: For a special angle α, cos α - sin α = 1/5 -/
theorem cos_minus_sin_special_angle (α : Real) (h : special_angle α) :
  Real.cos α - Real.sin α = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_special_angle_l957_95717


namespace NUMINAMATH_CALUDE_x_minus_p_in_terms_of_p_l957_95762

theorem x_minus_p_in_terms_of_p (x p : ℝ) (h1 : |x - 2| = p) (h2 : x < 2) : x - p = 2 - 2*p := by
  sorry

end NUMINAMATH_CALUDE_x_minus_p_in_terms_of_p_l957_95762


namespace NUMINAMATH_CALUDE_pumpkin_weight_difference_l957_95779

/-- Given three pumpkin weights with specific relationships, 
    prove the difference between the heaviest and lightest is 81 pounds. -/
theorem pumpkin_weight_difference :
  ∀ (brad jessica betty : ℝ),
  brad = 54 →
  jessica = brad / 2 →
  betty = 4 * jessica →
  max brad (max jessica betty) - min brad (min jessica betty) = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_pumpkin_weight_difference_l957_95779


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l957_95715

/-- The y-intercept of the line 3x - 4y = 12 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 4 * y = 12 → x = 0 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l957_95715


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l957_95716

/-- 
Given an arithmetic sequence {a_n} with common difference 3,
where a_1, a_3, and a_4 form a geometric sequence,
prove that a_2 = -9
-/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 3)^2 = a 1 * a 4 →         -- a_1, a_3, a_4 form a geometric sequence
  a 2 = -9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l957_95716


namespace NUMINAMATH_CALUDE_eighth_group_selection_l957_95747

/-- Represents the systematic sampling method for a population --/
def systematicSampling (populationSize : Nat) (groupCount : Nat) (t : Nat) : Nat → Nat :=
  fun k => (t + k - 1) % 10 + (k - 1) * 10

/-- Theorem stating the correct number selected from the 8th group --/
theorem eighth_group_selection
  (populationSize : Nat)
  (groupCount : Nat)
  (t : Nat)
  (h1 : populationSize = 100)
  (h2 : groupCount = 10)
  (h3 : t = 7) :
  systematicSampling populationSize groupCount t 8 = 75 := by
  sorry

#check eighth_group_selection

end NUMINAMATH_CALUDE_eighth_group_selection_l957_95747


namespace NUMINAMATH_CALUDE_max_t_value_l957_95702

theorem max_t_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 18 →
  k < m → m < r → r < s → s < t →
  r ≤ 23 →
  t = 40 := by
sorry

end NUMINAMATH_CALUDE_max_t_value_l957_95702


namespace NUMINAMATH_CALUDE_triangle_angle_b_value_l957_95790

theorem triangle_angle_b_value 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : A + B + C = π) 
  (h3 : 0 < A ∧ A < π) 
  (h4 : 0 < B ∧ B < π) 
  (h5 : 0 < C ∧ C < π) 
  (h6 : (c - b) / (Real.sqrt 2 * c - a) = Real.sin A / (Real.sin B + Real.sin C)) : 
  B = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_b_value_l957_95790


namespace NUMINAMATH_CALUDE_new_profit_percentage_is_50_percent_l957_95711

/-- Represents the selling price of a key chain -/
def selling_price : ℝ := 100

/-- Represents the initial manufacturing cost -/
def initial_cost : ℝ := 65

/-- Represents the new manufacturing cost -/
def new_cost : ℝ := 50

/-- Represents the initial profit percentage -/
def initial_profit_percentage : ℝ := 0.35

/-- Theorem stating that the new profit percentage is 50% given the conditions -/
theorem new_profit_percentage_is_50_percent :
  let initial_profit := initial_profit_percentage * selling_price
  let initial_equation := initial_cost + initial_profit = selling_price
  let new_profit := selling_price - new_cost
  let new_profit_percentage := new_profit / selling_price
  initial_equation → new_profit_percentage = 0.5 := by sorry


end NUMINAMATH_CALUDE_new_profit_percentage_is_50_percent_l957_95711


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l957_95745

theorem f_derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*(deriv f 1)) :
  deriv f 0 = -4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l957_95745


namespace NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l957_95793

theorem range_of_a_when_p_is_false :
  (¬∃ (x : ℝ), x > 0 ∧ x + 1/x < a) ↔ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l957_95793


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l957_95733

theorem arithmetic_sequence_nth_term (x : ℝ) (n : ℕ) : 
  (2*x - 3 = (5*x - 11) - (3*x - 8)) → 
  (5*x - 11 = (3*x + 1) - (3*x - 8)) → 
  (1 + 4*n = 2009) → 
  n = 502 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l957_95733


namespace NUMINAMATH_CALUDE_smallest_x_composite_l957_95768

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def absolute_value (n : ℤ) : ℕ := Int.natAbs n

theorem smallest_x_composite : 
  (∀ x : ℤ, x < 5 → ¬ is_composite (absolute_value (5 * x^2 - 38 * x + 7))) ∧ 
  is_composite (absolute_value (5 * 5^2 - 38 * 5 + 7)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_composite_l957_95768


namespace NUMINAMATH_CALUDE_steven_shirt_count_l957_95797

def brian_shirts : ℕ := 3
def andrew_shirts : ℕ := 6 * brian_shirts
def steven_shirts : ℕ := 4 * andrew_shirts

theorem steven_shirt_count : steven_shirts = 72 := by
  sorry

end NUMINAMATH_CALUDE_steven_shirt_count_l957_95797


namespace NUMINAMATH_CALUDE_quadratic_sum_l957_95770

/-- A quadratic function with vertex (h, k) and passing through point (x₀, y₀) -/
def quadratic_function (a b c h k x₀ y₀ : ℝ) : Prop :=
  ∀ x, a * (x - h)^2 + k = a * x^2 + b * x + c ∧
  a * (x₀ - h)^2 + k = y₀

theorem quadratic_sum (a b c : ℝ) :
  quadratic_function a b c 2 5 1 8 →
  a - b + c = 32 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l957_95770


namespace NUMINAMATH_CALUDE_evaluate_expression_l957_95791

theorem evaluate_expression : 2 + 0 - 2 * 0 = 2 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l957_95791


namespace NUMINAMATH_CALUDE_woman_lawyer_probability_l957_95780

/-- Represents a study group with given proportions of women and lawyers --/
structure StudyGroup where
  total_members : ℕ
  women_percentage : ℝ
  lawyer_percentage : ℝ
  women_percentage_valid : 0 ≤ women_percentage ∧ women_percentage ≤ 1
  lawyer_percentage_valid : 0 ≤ lawyer_percentage ∧ lawyer_percentage ≤ 1

/-- Calculates the probability of selecting a woman lawyer from the study group --/
def probability_woman_lawyer (group : StudyGroup) : ℝ :=
  group.women_percentage * group.lawyer_percentage

/-- Theorem stating that the probability of selecting a woman lawyer is 0.32 
    given the specified conditions --/
theorem woman_lawyer_probability (group : StudyGroup) 
  (h1 : group.women_percentage = 0.8) 
  (h2 : group.lawyer_percentage = 0.4) : 
  probability_woman_lawyer group = 0.32 := by
  sorry


end NUMINAMATH_CALUDE_woman_lawyer_probability_l957_95780


namespace NUMINAMATH_CALUDE_discount_calculation_l957_95704

/-- Proves the true discount and the difference between claimed and true discount for a given discount scenario. -/
theorem discount_calculation (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) :
  initial_discount = 0.25 →
  additional_discount = 0.1 →
  claimed_discount = 0.4 →
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_additional := remaining_after_initial * (1 - additional_discount)
  let true_discount := 1 - remaining_after_additional
  true_discount = 0.325 ∧ claimed_discount - true_discount = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l957_95704


namespace NUMINAMATH_CALUDE_e_squared_f_2_gt_e_cubed_f_3_l957_95707

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the condition that f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define the given condition
axiom condition : ∀ x, f x + f' x < 0

-- State the theorem to be proved
theorem e_squared_f_2_gt_e_cubed_f_3 : e^2 * f 2 > e^3 * f 3 := by sorry

end NUMINAMATH_CALUDE_e_squared_f_2_gt_e_cubed_f_3_l957_95707


namespace NUMINAMATH_CALUDE_waiter_tip_calculation_l957_95703

/-- Waiter's tip calculation problem -/
theorem waiter_tip_calculation
  (total_customers : ℕ)
  (non_tipping_customers : ℕ)
  (total_tip_amount : ℕ)
  (h1 : total_customers = 9)
  (h2 : non_tipping_customers = 5)
  (h3 : total_tip_amount = 32) :
  total_tip_amount / (total_customers - non_tipping_customers) = 8 := by
  sorry

#check waiter_tip_calculation

end NUMINAMATH_CALUDE_waiter_tip_calculation_l957_95703


namespace NUMINAMATH_CALUDE_nina_savings_weeks_l957_95756

/-- The number of weeks Nina needs to save to buy a video game -/
def weeks_to_save (game_cost : ℚ) (tax_rate : ℚ) (weekly_allowance : ℚ) (savings_rate : ℚ) : ℚ :=
  let total_cost := game_cost * (1 + tax_rate)
  let weekly_savings := weekly_allowance * savings_rate
  total_cost / weekly_savings

/-- Theorem: Nina needs 11 weeks to save for the video game -/
theorem nina_savings_weeks :
  weeks_to_save 50 0.1 10 0.5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_nina_savings_weeks_l957_95756


namespace NUMINAMATH_CALUDE_fraction_simplification_l957_95767

theorem fraction_simplification (x : ℝ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l957_95767


namespace NUMINAMATH_CALUDE_expression_equality_l957_95750

theorem expression_equality : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l957_95750


namespace NUMINAMATH_CALUDE_vehicle_speeds_l957_95759

/-- Proves that given the conditions, the bus speed is 20 km/h and the car speed is 60 km/h. -/
theorem vehicle_speeds 
  (distance : ℝ) 
  (bus_delay : ℝ) 
  (car_arrival_delay : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : distance = 70) 
  (h2 : bus_delay = 3) 
  (h3 : car_arrival_delay = 2/3) 
  (h4 : speed_ratio = 3) : 
  ∃ (bus_speed car_speed : ℝ), 
    bus_speed = 20 ∧ 
    car_speed = 60 ∧ 
    distance / bus_speed = distance / car_speed + bus_delay - car_arrival_delay ∧
    car_speed = speed_ratio * bus_speed :=
by sorry

end NUMINAMATH_CALUDE_vehicle_speeds_l957_95759


namespace NUMINAMATH_CALUDE_vacuum_time_per_room_l957_95761

theorem vacuum_time_per_room 
  (battery_life : ℕ) 
  (num_rooms : ℕ) 
  (additional_charges : ℕ) 
  (h1 : battery_life = 10)
  (h2 : num_rooms = 5)
  (h3 : additional_charges = 2) :
  (battery_life * (additional_charges + 1)) / num_rooms = 6 := by
  sorry

end NUMINAMATH_CALUDE_vacuum_time_per_room_l957_95761


namespace NUMINAMATH_CALUDE_ciphertext_solution_l957_95734

theorem ciphertext_solution :
  ∃! (x₁ x₂ x₃ x₄ : ℕ),
    x₁ ≤ 25 ∧ x₂ ≤ 25 ∧ x₃ ≤ 25 ∧ x₄ ≤ 25 ∧
    (x₁ + 2*x₂) % 26 = 9 ∧
    (3*x₂) % 26 = 16 ∧
    (x₃ + 2*x₄) % 26 = 23 ∧
    (3*x₄) % 26 = 12 ∧
    x₁ = 7 ∧ x₂ = 14 ∧ x₃ = 15 ∧ x₄ = 4 :=
by sorry

end NUMINAMATH_CALUDE_ciphertext_solution_l957_95734


namespace NUMINAMATH_CALUDE_expansion_terms_count_expansion_terms_count_equals_66_l957_95737

theorem expansion_terms_count : Nat :=
  let n : Nat := 10  -- power in (a + b + c)^10
  let k : Nat := 3   -- number of variables (a, b, c)
  Nat.choose (n + k - 1) (k - 1)

theorem expansion_terms_count_equals_66 : expansion_terms_count = 66 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_expansion_terms_count_equals_66_l957_95737


namespace NUMINAMATH_CALUDE_bob_distance_when_meeting_l957_95778

/-- Prove that Bob walked 8 miles when he met Yolanda given the following conditions:
  - The total distance between X and Y is 17 miles
  - Yolanda starts walking from X to Y
  - Bob starts walking from Y to X one hour after Yolanda
  - Yolanda's walking rate is 3 miles per hour
  - Bob's walking rate is 4 miles per hour
-/
theorem bob_distance_when_meeting (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) 
  (h1 : total_distance = 17)
  (h2 : yolanda_rate = 3)
  (h3 : bob_rate = 4) :
  ∃ t : ℝ, t > 0 ∧ yolanda_rate * (t + 1) + bob_rate * t = total_distance ∧ bob_rate * t = 8 := by
  sorry


end NUMINAMATH_CALUDE_bob_distance_when_meeting_l957_95778


namespace NUMINAMATH_CALUDE_square_tile_side_length_l957_95766

theorem square_tile_side_length (side : ℝ) (area : ℝ) : 
  area = 49 ∧ area = side * side → side = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_tile_side_length_l957_95766


namespace NUMINAMATH_CALUDE_total_students_surveyed_l957_95760

theorem total_students_surveyed :
  let french_and_english : ℕ := 25
  let french_not_english : ℕ := 65
  let percent_not_french : ℚ := 55/100
  let total_students : ℕ := 200
  (french_and_english + french_not_english : ℚ) / total_students = 1 - percent_not_french :=
by sorry

end NUMINAMATH_CALUDE_total_students_surveyed_l957_95760


namespace NUMINAMATH_CALUDE_a_squared_plus_a_negative_l957_95787

theorem a_squared_plus_a_negative (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by
  sorry

end NUMINAMATH_CALUDE_a_squared_plus_a_negative_l957_95787


namespace NUMINAMATH_CALUDE_cube_midpoint_planes_l957_95796

-- Define a cube type
structure Cube where
  -- Add necessary properties of a cube

-- Define a plane type
structure Plane where
  -- Add necessary properties of a plane

-- Define a function to check if a plane contains a midpoint of a cube's edge
def containsMidpoint (p : Plane) (c : Cube) : Prop :=
  sorry

-- Define a function to count the number of midpoints a plane contains
def countMidpoints (p : Plane) (c : Cube) : ℕ :=
  sorry

-- Define a function to check if a plane contains at least 3 midpoints
def containsAtLeastThreeMidpoints (p : Plane) (c : Cube) : Prop :=
  countMidpoints p c ≥ 3

-- Define a function to count the number of planes containing at least 3 midpoints
def countPlanesWithAtLeastThreeMidpoints (c : Cube) : ℕ :=
  sorry

-- Theorem statement
theorem cube_midpoint_planes (c : Cube) :
  countPlanesWithAtLeastThreeMidpoints c = 81 :=
sorry

end NUMINAMATH_CALUDE_cube_midpoint_planes_l957_95796


namespace NUMINAMATH_CALUDE_mean_median_mode_equality_l957_95769

/-- Represents the days of the week -/
inductive Weekday
  | Saturday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- A month with its properties -/
structure Month where
  totalDays : Nat
  startDay : Weekday
  frequencies : Weekday → Nat

/-- Calculates the mean of the frequencies -/
def calculateMean (m : Month) : ℚ :=
  (m.frequencies Weekday.Saturday +
   m.frequencies Weekday.Sunday +
   m.frequencies Weekday.Monday +
   m.frequencies Weekday.Tuesday +
   m.frequencies Weekday.Wednesday +
   m.frequencies Weekday.Thursday +
   m.frequencies Weekday.Friday) / 7

/-- Calculates the median day -/
def calculateMedian (m : Month) : Weekday :=
  Weekday.Tuesday  -- Since the 15th day (median) is a Tuesday

/-- Calculates the median of the modes -/
def calculateMedianOfModes (m : Month) : ℚ := 4

/-- The theorem to be proved -/
theorem mean_median_mode_equality (m : Month)
  (h1 : m.totalDays = 29)
  (h2 : m.startDay = Weekday.Saturday)
  (h3 : m.frequencies Weekday.Saturday = 5)
  (h4 : m.frequencies Weekday.Sunday = 4)
  (h5 : m.frequencies Weekday.Monday = 4)
  (h6 : m.frequencies Weekday.Tuesday = 4)
  (h7 : m.frequencies Weekday.Wednesday = 4)
  (h8 : m.frequencies Weekday.Thursday = 4)
  (h9 : m.frequencies Weekday.Friday = 4) :
  calculateMean m = calculateMedianOfModes m ∧
  calculateMedianOfModes m = (calculateMedian m).rec 4 4 4 4 4 4 4 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_mode_equality_l957_95769


namespace NUMINAMATH_CALUDE_board_game_impossibility_l957_95746

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The operation of replacing two numbers with their difference -/
def replace_with_diff (s : ℤ) (a b : ℤ) : ℤ := s - 2 * min a b

/-- Theorem: It's impossible to reduce the sum of numbers from 1 to 1989 to zero
    by repeatedly replacing any two numbers with their difference -/
theorem board_game_impossibility :
  ∀ (ops : ℕ),
  ∃ (result : ℤ),
  result ≠ 0 ∧
  (∃ (numbers : List ℤ),
    numbers.sum = result ∧
    numbers.length + ops = 1989 ∧
    (∀ (x : ℤ), x ∈ numbers → x ≥ 0)) :=
by sorry


end NUMINAMATH_CALUDE_board_game_impossibility_l957_95746


namespace NUMINAMATH_CALUDE_distance_to_city_l957_95705

theorem distance_to_city (D : ℝ) 
  (h1 : D / 2 + D / 4 + 6 = D) : D = 24 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_city_l957_95705


namespace NUMINAMATH_CALUDE_abs_value_inequality_l957_95714

theorem abs_value_inequality (x : ℝ) : |x| < 5 ↔ -5 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_abs_value_inequality_l957_95714


namespace NUMINAMATH_CALUDE_regression_relationships_l957_95748

/-- Represents the possibility that x is not related to y -/
def notRelatedPossibility : ℝ → ℝ := sorry

/-- Represents the fitting effect of the regression line -/
def fittingEffect : ℝ → ℝ := sorry

/-- Represents the degree of fit -/
def degreeOfFit : ℝ → ℝ := sorry

theorem regression_relationships :
  (∀ k₁ k₂ : ℝ, k₁ < k₂ → notRelatedPossibility k₁ > notRelatedPossibility k₂) ∧
  (∀ s₁ s₂ : ℝ, s₁ < s₂ → fittingEffect s₁ > fittingEffect s₂) ∧
  (∀ r₁ r₂ : ℝ, r₁ < r₂ → degreeOfFit r₁ < degreeOfFit r₂) :=
by sorry

end NUMINAMATH_CALUDE_regression_relationships_l957_95748


namespace NUMINAMATH_CALUDE_f_at_4_l957_95740

def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

theorem f_at_4 : f 4 = 371 := by
  sorry

end NUMINAMATH_CALUDE_f_at_4_l957_95740


namespace NUMINAMATH_CALUDE_division_problem_l957_95795

theorem division_problem (a b q : ℕ) (h1 : a - b = 1365) (h2 : a = 1575) (h3 : a = b * q + 15) : q = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l957_95795


namespace NUMINAMATH_CALUDE_sin_translation_l957_95741

theorem sin_translation (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 3)
  let translation : ℝ := π / 6
  let result : ℝ → ℝ := λ x => Real.sin (2 * x + 2 * π / 3)
  (λ x => f (x + translation)) = result := by
sorry

end NUMINAMATH_CALUDE_sin_translation_l957_95741


namespace NUMINAMATH_CALUDE_inequality_solution_and_property_l957_95722

def f (x : ℝ) := |x - 2|

theorem inequality_solution_and_property (a : ℝ) (h : a > 2) :
  (∀ x, f (x + 1) + f (x + 2) < 4 ↔ x ∈ Set.Ioo (-3/2) (5/2)) ∧
  (∀ x, f (a * x) + a * f x > 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_and_property_l957_95722


namespace NUMINAMATH_CALUDE_complex_fraction_value_l957_95783

theorem complex_fraction_value : (Complex.I : ℂ) / (1 - Complex.I)^2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l957_95783


namespace NUMINAMATH_CALUDE_boat_distance_theorem_l957_95742

/-- The distance traveled by a boat against a water flow -/
def distance_traveled (a : ℝ) : ℝ :=
  3 * (a - 3)

/-- Theorem: The distance traveled by a boat against a water flow in 3 hours
    is 3(a-3) km, given that the boat's speed in still water is a km/h
    and the water flow speed is 3 km/h. -/
theorem boat_distance_theorem (a : ℝ) :
  let boat_speed := a
  let water_flow_speed := (3 : ℝ)
  let travel_time := (3 : ℝ)
  distance_traveled a = travel_time * (boat_speed - water_flow_speed) :=
by
  sorry


end NUMINAMATH_CALUDE_boat_distance_theorem_l957_95742


namespace NUMINAMATH_CALUDE_commodity_tax_consumption_l957_95752

theorem commodity_tax_consumption (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let new_tax := 0.8 * T
  let new_revenue := 0.92 * T * C
  ∃ new_consumption, 
    new_tax * new_consumption = new_revenue ∧ 
    new_consumption = 1.15 * C := by
sorry

end NUMINAMATH_CALUDE_commodity_tax_consumption_l957_95752


namespace NUMINAMATH_CALUDE_joes_trip_expenses_l957_95718

/-- Joe's trip expenses problem -/
theorem joes_trip_expenses (initial_savings : ℕ) (flight_cost : ℕ) (hotel_cost : ℕ) (remaining : ℕ) 
  (h1 : initial_savings = 6000)
  (h2 : flight_cost = 1200)
  (h3 : hotel_cost = 800)
  (h4 : remaining = 1000) :
  initial_savings - flight_cost - hotel_cost - remaining = 3000 := by
  sorry

end NUMINAMATH_CALUDE_joes_trip_expenses_l957_95718


namespace NUMINAMATH_CALUDE_acrobats_count_l957_95753

/-- Represents the number of acrobats, elephants, and clowns in a circus performance. -/
structure CircusPerformance where
  acrobats : ℕ
  elephants : ℕ
  clowns : ℕ

/-- Conditions for the circus performance. -/
def circusConditions (p : CircusPerformance) : Prop :=
  2 * p.acrobats + 4 * p.elephants + 2 * p.clowns = 58 ∧
  p.acrobats + p.elephants + p.clowns = 22

/-- Theorem stating that under the given conditions, the number of acrobats is 0. -/
theorem acrobats_count (p : CircusPerformance) : 
  circusConditions p → p.acrobats = 0 := by
  sorry


end NUMINAMATH_CALUDE_acrobats_count_l957_95753


namespace NUMINAMATH_CALUDE_circle_circumference_irrational_l957_95700

/-- A circle with rational diameter has irrational circumference -/
theorem circle_circumference_irrational (d : ℚ) :
  ∃ (C : ℝ), C = π * (d : ℝ) ∧ Irrational C := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_irrational_l957_95700


namespace NUMINAMATH_CALUDE_pillar_length_calculation_l957_95726

/-- Given a formula for L and specific values for T, H, and K, prove that L equals 100. -/
theorem pillar_length_calculation (T H K L : ℝ) : 
  T = 2 * Real.sqrt 5 →
  H = 10 →
  K = 2 →
  L = (50 * T^4) / (H^2 * K) →
  L = 100 := by
sorry

end NUMINAMATH_CALUDE_pillar_length_calculation_l957_95726


namespace NUMINAMATH_CALUDE_tetragon_diagonals_l957_95725

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A tetragon is a polygon with 4 sides -/
def tetragon_sides : ℕ := 4

/-- Theorem: The number of diagonals in a tetragon is 2 -/
theorem tetragon_diagonals : num_diagonals tetragon_sides = 2 := by
  sorry

end NUMINAMATH_CALUDE_tetragon_diagonals_l957_95725


namespace NUMINAMATH_CALUDE_quarterback_passes_l957_95721

theorem quarterback_passes (total passes_left passes_right passes_center : ℕ) : 
  total = 50 → 
  passes_left = 12 → 
  passes_right = 2 * passes_left → 
  total = passes_left + passes_right + passes_center → 
  passes_center - passes_left = 2 := by
  sorry

end NUMINAMATH_CALUDE_quarterback_passes_l957_95721


namespace NUMINAMATH_CALUDE_cost_price_calculation_l957_95732

theorem cost_price_calculation (cost_price : ℝ) : 
  cost_price * 1.20 * 0.91 = cost_price + 16 → cost_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l957_95732


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l957_95774

/-- Given two lines with the same non-zero y-intercept and different slopes,
    prove that the ratio of their x-intercepts is 1/2 -/
theorem x_intercept_ratio (b : ℝ) (u v : ℝ) : 
  b ≠ 0 →  -- The common y-intercept is non-zero
  0 = 8 * u + b →  -- First line equation at x-intercept
  0 = 4 * v + b →  -- Second line equation at x-intercept
  u / v = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l957_95774


namespace NUMINAMATH_CALUDE_factorization_proof_l957_95728

theorem factorization_proof (x : ℝ) : 
  3 * x^2 * (x - 5) + 4 * x * (x - 5) + 6 * (x - 5) = (3 * x^2 + 4 * x + 6) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l957_95728


namespace NUMINAMATH_CALUDE_parallel_lines_l957_95758

/-- Two lines are parallel if their normal vectors are proportional -/
def parallel (a b c d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a = k * d ∧ b = k * e

/-- Two lines coincide if their coefficients are proportional -/
def coincide (a b c d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a = k * d ∧ b = k * e ∧ c = k * f

theorem parallel_lines (a : ℝ) :
  parallel a (a + 2) 2 1 a 1 ∧ ¬coincide a (a + 2) 2 1 a 1 → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_l957_95758


namespace NUMINAMATH_CALUDE_square_root_pattern_square_root_ten_squared_minus_one_l957_95776

theorem square_root_pattern (n : ℕ) (hn : n ≥ 3) :
  ∀ m : ℕ, m ≥ 3 → m ≤ 5 →
  Real.sqrt (m^2 - 1) = Real.sqrt (m - 1) * Real.sqrt (m + 1) :=
  sorry

theorem square_root_ten_squared_minus_one :
  Real.sqrt (10^2 - 1) = 3 * Real.sqrt 11 :=
  sorry

end NUMINAMATH_CALUDE_square_root_pattern_square_root_ten_squared_minus_one_l957_95776


namespace NUMINAMATH_CALUDE_problem_solution_l957_95719

theorem problem_solution : 
  (Real.sqrt 27 + Real.sqrt 3 - 2 * Real.sqrt 12 = 0) ∧
  ((3 + 2 * Real.sqrt 2) * (3 - 2 * Real.sqrt 2) - Real.sqrt 54 / Real.sqrt 6 = -2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l957_95719


namespace NUMINAMATH_CALUDE_equal_angles_same_terminal_side_l957_95709

-- Define the angle type
def Angle : Type := ℝ

-- Define the terminal side of an angle
def terminalSide (α : Angle) : Set (ℝ × ℝ) := sorry

-- Theorem: If two angles are equal, they have the same terminal side
theorem equal_angles_same_terminal_side (α β : Angle) :
  α = β → terminalSide α = terminalSide β := by sorry

end NUMINAMATH_CALUDE_equal_angles_same_terminal_side_l957_95709


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l957_95743

/-- The number of ways to place n distinguishable objects into k distinguishable containers -/
def placement_count (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to place 5 distinguishable balls into 4 distinguishable boxes is 1024 -/
theorem five_balls_four_boxes : placement_count 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l957_95743


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l957_95789

/-- Represents a chicken farm with its population -/
structure Farm where
  population : ℕ

/-- Calculates the sample size for a farm given the total population and total sample size -/
def sampleSize (farm : Farm) (totalPopulation : ℕ) (totalSample : ℕ) : ℕ :=
  (farm.population * totalSample) / totalPopulation

theorem stratified_sampling_theorem (farmA farmB farmC : Farm) 
    (h1 : farmA.population = 12000)
    (h2 : farmB.population = 8000)
    (h3 : farmC.population = 4000)
    (totalSample : ℕ)
    (h4 : totalSample = 120) :
  let totalPopulation := farmA.population + farmB.population + farmC.population
  (sampleSize farmA totalPopulation totalSample = 60) ∧
  (sampleSize farmB totalPopulation totalSample = 40) ∧
  (sampleSize farmC totalPopulation totalSample = 20) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l957_95789


namespace NUMINAMATH_CALUDE_biased_coin_prob_sum_l957_95781

/-- The probability of getting heads for a biased coin -/
def h : ℚ :=
  3 / 7

/-- The condition that the probability of 2 heads equals the probability of 3 heads in 6 flips -/
axiom prob_equality : 15 * h^2 * (1 - h)^4 = 20 * h^3 * (1 - h)^3

/-- The probability of getting exactly 4 heads in 6 flips -/
def prob_4_heads : ℚ :=
  15 * h^4 * (1 - h)^2

/-- The numerator and denominator of prob_4_heads in lowest terms -/
def p : ℕ := 19440
def q : ℕ := 117649

theorem biased_coin_prob_sum :
  prob_4_heads = p / q ∧ p + q = 137089 := by sorry

end NUMINAMATH_CALUDE_biased_coin_prob_sum_l957_95781


namespace NUMINAMATH_CALUDE_angle_A_range_triangle_area_l957_95739

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a^2 + t.a * t.c = t.b^2

-- Theorem I
theorem angle_A_range (t : Triangle) (h : triangle_condition t) :
  0 < t.A ∧ t.A < π/3 := by sorry

-- Theorem II
theorem triangle_area (t : Triangle) (h : triangle_condition t) 
  (h_a : t.a = 2) (h_A : t.A = π/6) :
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_A_range_triangle_area_l957_95739


namespace NUMINAMATH_CALUDE_equation_root_interval_l957_95782

-- Define the function f(x) = lg(x+1) + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2 + x - 3

-- State the theorem
theorem equation_root_interval :
  ∃ (x : ℝ), x ∈ Set.Ioo 2 3 ∧ f x = 0 ∧
  ∀ (k : ℤ), (∃ (y : ℝ), y ∈ Set.Ioo (k : ℝ) (k + 1) ∧ f y = 0) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_interval_l957_95782


namespace NUMINAMATH_CALUDE_cubic_inequality_implies_a_range_l957_95738

theorem cubic_inequality_implies_a_range :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → a * x^3 - x^2 + 4*x + 3 ≥ 0) →
  a ∈ Set.Icc (-6) (-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_implies_a_range_l957_95738


namespace NUMINAMATH_CALUDE_first_group_size_l957_95706

/-- The number of persons in the first group that can repair a road -/
def first_group : ℕ :=
  let days : ℕ := 12
  let hours_per_day_first : ℕ := 5
  let second_group : ℕ := 30
  let hours_per_day_second : ℕ := 6
  (second_group * hours_per_day_second) / hours_per_day_first

theorem first_group_size :
  first_group = 36 :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l957_95706


namespace NUMINAMATH_CALUDE_triangle_inequality_squared_l957_95777

theorem triangle_inequality_squared (a b c : ℝ) 
  (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle : a < b + c ∧ b < a + c ∧ c < a + b) : 
  a^2 < a*b + a*c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_squared_l957_95777


namespace NUMINAMATH_CALUDE_stacy_height_last_year_l957_95757

/-- Represents Stacy's height measurements and growth --/
structure StacyHeight where
  current : ℕ
  brother_growth : ℕ
  growth_difference : ℕ

/-- Calculates Stacy's height last year given her current measurements --/
def height_last_year (s : StacyHeight) : ℕ :=
  s.current - (s.brother_growth + s.growth_difference)

/-- Theorem stating Stacy's height last year was 50 inches --/
theorem stacy_height_last_year :
  let s : StacyHeight := {
    current := 57,
    brother_growth := 1,
    growth_difference := 6
  }
  height_last_year s = 50 := by
  sorry

end NUMINAMATH_CALUDE_stacy_height_last_year_l957_95757


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l957_95729

theorem smallest_five_digit_divisible_by_53 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 → n ≥ 10017 := by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l957_95729
