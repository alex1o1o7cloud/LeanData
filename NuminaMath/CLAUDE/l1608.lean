import Mathlib

namespace NUMINAMATH_CALUDE_emily_marbles_l1608_160885

/-- Emily's marble problem -/
theorem emily_marbles :
  let initial_marbles : ℕ := 6
  let megan_gives := 2 * initial_marbles
  let emily_new_total := initial_marbles + megan_gives
  let emily_gives_back := emily_new_total / 2 + 1
  let emily_final := emily_new_total - emily_gives_back
  emily_final = 8 := by sorry

end NUMINAMATH_CALUDE_emily_marbles_l1608_160885


namespace NUMINAMATH_CALUDE_square_minus_equal_two_implies_sum_equal_one_l1608_160807

theorem square_minus_equal_two_implies_sum_equal_one (m : ℝ) 
  (h : m^2 - m = 2) : 
  (m - 1)^2 + (m + 2) * (m - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_equal_two_implies_sum_equal_one_l1608_160807


namespace NUMINAMATH_CALUDE_soda_survey_l1608_160843

/-- Given a survey of 600 people and a central angle of 108° for the "Soda" sector,
    prove that the number of people who chose "Soda" is 180. -/
theorem soda_survey (total_people : ℕ) (soda_angle : ℕ) :
  total_people = 600 →
  soda_angle = 108 →
  (total_people * soda_angle) / 360 = 180 := by
  sorry

end NUMINAMATH_CALUDE_soda_survey_l1608_160843


namespace NUMINAMATH_CALUDE_juice_mixture_solution_l1608_160849

/-- Represents the juice mixture problem -/
def JuiceMixture (super_cost mixed_cost acai_cost : ℝ) (acai_amount : ℝ) : Prop :=
  ∃ (mixed_amount : ℝ),
    mixed_amount ≥ 0 ∧
    super_cost * (mixed_amount + acai_amount) =
      mixed_cost * mixed_amount + acai_cost * acai_amount

/-- The solution to the juice mixture problem is approximately 35 litres -/
theorem juice_mixture_solution :
  JuiceMixture 1399.45 262.85 3104.35 23.333333333333336 →
  ∃ (mixed_amount : ℝ),
    mixed_amount ≥ 0 ∧
    abs (mixed_amount - 35) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_juice_mixture_solution_l1608_160849


namespace NUMINAMATH_CALUDE_science_fair_girls_fraction_l1608_160827

theorem science_fair_girls_fraction :
  let pine_grove_total : ℕ := 300
  let pine_grove_ratio_boys : ℕ := 3
  let pine_grove_ratio_girls : ℕ := 2
  let maple_town_total : ℕ := 240
  let maple_town_ratio_boys : ℕ := 5
  let maple_town_ratio_girls : ℕ := 3
  let total_students := pine_grove_total + maple_town_total
  let pine_grove_girls := (pine_grove_total * pine_grove_ratio_girls) / (pine_grove_ratio_boys + pine_grove_ratio_girls)
  let maple_town_girls := (maple_town_total * maple_town_ratio_girls) / (maple_town_ratio_boys + maple_town_ratio_girls)
  let total_girls := pine_grove_girls + maple_town_girls
  (total_girls : ℚ) / total_students = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_science_fair_girls_fraction_l1608_160827


namespace NUMINAMATH_CALUDE_speech_competition_orders_l1608_160870

/-- The number of different possible orders in a speech competition --/
def num_orders (original : ℕ) (new : ℕ) : ℕ :=
  (original + 1) * (original + 2)

/-- Theorem: For 5 original participants and 2 new participants,
    the number of different possible orders is 42 --/
theorem speech_competition_orders :
  num_orders 5 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_speech_competition_orders_l1608_160870


namespace NUMINAMATH_CALUDE_frequency_in_range_l1608_160835

/-- Represents an interval with its frequency -/
structure IntervalData where
  lower : ℝ
  upper : ℝ
  frequency : ℕ

/-- Calculates the frequency of a sample within a given range -/
def calculateFrequency (data : List IntervalData) (range_start range_end : ℝ) (sample_size : ℕ) : ℝ :=
  sorry

/-- The given data set -/
def sampleData : List IntervalData := [
  ⟨10, 20, 2⟩,
  ⟨20, 30, 3⟩,
  ⟨30, 40, 4⟩,
  ⟨40, 50, 5⟩,
  ⟨50, 60, 4⟩,
  ⟨60, 70, 2⟩
]

theorem frequency_in_range : calculateFrequency sampleData 15 50 20 = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_frequency_in_range_l1608_160835


namespace NUMINAMATH_CALUDE_rectangle_arrangement_perimeters_l1608_160834

/- Define the properties of the rectangles -/
def identical_rectangles (l w : ℝ) : Prop := l = 2 * w

/- Define the first arrangement's perimeter -/
def first_arrangement_perimeter (l w : ℝ) : ℝ := 3 * l + 4 * w

/- Define the second arrangement's perimeter -/
def second_arrangement_perimeter (l w : ℝ) : ℝ := 6 * l + 2 * w

/- Theorem statement -/
theorem rectangle_arrangement_perimeters (l w : ℝ) :
  identical_rectangles l w →
  first_arrangement_perimeter l w = 20 →
  second_arrangement_perimeter l w = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_perimeters_l1608_160834


namespace NUMINAMATH_CALUDE_video_distribution_solution_l1608_160820

/-- Represents the problem of distributing video content across discs -/
def VideoDistribution (total_minutes : ℝ) (max_capacity : ℝ) : Prop :=
  ∃ (num_discs : ℕ) (minutes_per_disc : ℝ),
    num_discs > 0 ∧
    num_discs = ⌈total_minutes / max_capacity⌉ ∧
    minutes_per_disc = total_minutes / num_discs ∧
    minutes_per_disc ≤ max_capacity

/-- Theorem stating the solution to the video distribution problem -/
theorem video_distribution_solution :
  VideoDistribution 495 65 →
  ∃ (num_discs : ℕ) (minutes_per_disc : ℝ),
    num_discs = 8 ∧ minutes_per_disc = 61.875 := by
  sorry

#check video_distribution_solution

end NUMINAMATH_CALUDE_video_distribution_solution_l1608_160820


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1608_160833

open Real

/-- Given f(x) = 2/x + a*ln(x) - 2 where a > 0, if f(x) > 2(a-1) for all x > 0, then 0 < a < 2/e -/
theorem function_inequality_implies_a_range (a : ℝ) (h_a_pos : a > 0) :
  (∀ x : ℝ, x > 0 → (2 / x + a * log x - 2 > 2 * (a - 1))) →
  0 < a ∧ a < 2 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1608_160833


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l1608_160806

theorem smaller_root_of_equation : 
  let f (x : ℚ) := (x - 3/5) * (x - 3/5) + 2 * (x - 3/5) * (x - 1/3)
  ∃ r : ℚ, f r = 0 ∧ r = 19/45 ∧ ∀ s : ℚ, f s = 0 → s ≠ r → r < s :=
by sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l1608_160806


namespace NUMINAMATH_CALUDE_vector_equation_l1608_160899

def a : ℝ × ℝ := (3, -2)
def b : ℝ × ℝ := (-2, 1)
def c : ℝ × ℝ := (-12, 7)

theorem vector_equation (m n : ℝ) (h : c = m • a + n • b) : m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l1608_160899


namespace NUMINAMATH_CALUDE_relay_race_time_l1608_160811

/-- Represents the time taken by each runner in the relay race -/
structure RelayTimes where
  mary : ℕ
  susan : ℕ
  jen : ℕ
  tiffany : ℕ

/-- Calculates the total time of the relay race -/
def total_time (times : RelayTimes) : ℕ :=
  times.mary + times.susan + times.jen + times.tiffany

/-- Theorem stating that the total time of the relay race is 223 seconds -/
theorem relay_race_time : ∃ (times : RelayTimes), 
  times.mary = 2 * times.susan ∧
  times.susan = times.jen + 10 ∧
  times.jen = 30 ∧
  times.tiffany = times.mary - 7 ∧
  total_time times = 223 := by
  sorry


end NUMINAMATH_CALUDE_relay_race_time_l1608_160811


namespace NUMINAMATH_CALUDE_force_for_18_inch_crowbar_l1608_160816

-- Define the inverse relationship between force and length
def inverse_relationship (force : ℝ) (length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

-- Define the given condition
def given_condition : Prop :=
  inverse_relationship 200 12

-- Define the theorem to be proved
theorem force_for_18_inch_crowbar :
  given_condition →
  ∃ force : ℝ, inverse_relationship force 18 ∧ 
    (force ≥ 133.33 ∧ force ≤ 133.34) :=
by
  sorry

end NUMINAMATH_CALUDE_force_for_18_inch_crowbar_l1608_160816


namespace NUMINAMATH_CALUDE_total_passengers_in_hour_l1608_160866

/-- Calculates the total number of different passengers stepping on and off trains at a station within an hour -/
def total_passengers (train_frequency : ℕ) (passengers_leaving : ℕ) (passengers_boarding : ℕ) : ℕ :=
  let trains_per_hour := 60 / train_frequency
  let passengers_per_train := passengers_leaving + passengers_boarding
  trains_per_hour * passengers_per_train

/-- Proves that given the specified conditions, the total number of different passengers in an hour is 6240 -/
theorem total_passengers_in_hour :
  total_passengers 5 200 320 = 6240 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_in_hour_l1608_160866


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1608_160839

-- Define the foci of the ellipse
def F1 : ℝ × ℝ := (3, 15)
def F2 : ℝ × ℝ := (28, 45)

-- Define the reflection of F1 over the y-axis
def F1_reflected : ℝ × ℝ := (-3, 15)

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) (k : ℝ) : Prop :=
  dist P F1 + dist P F2 = k

-- Define the tangency condition
def is_tangent_to_y_axis (k : ℝ) : Prop :=
  ∃ y : ℝ, is_on_ellipse (0, y) k ∧
    ∀ y' : ℝ, is_on_ellipse (0, y') k → y = y'

-- State the theorem
theorem ellipse_major_axis_length :
  ∃ k : ℝ, is_tangent_to_y_axis k ∧ k = dist F1_reflected F2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1608_160839


namespace NUMINAMATH_CALUDE_cross_country_race_winning_scores_l1608_160892

/-- Represents a cross-country race with two teams -/
structure CrossCountryRace where
  /-- The number of players in each team -/
  players_per_team : Nat
  /-- The total number of players in the race -/
  total_players : Nat
  /-- The sum of all possible scores in the race -/
  total_score : Nat

/-- Calculates the maximum possible score for the winning team -/
def max_winning_score (race : CrossCountryRace) : Nat :=
  race.total_score / 2

/-- Calculates the minimum possible score for any team -/
def min_team_score (race : CrossCountryRace) : Nat :=
  List.sum (List.range race.players_per_team)

/-- The number of possible scores for the winning team -/
def winning_score_count (race : CrossCountryRace) : Nat :=
  max_winning_score race - min_team_score race + 1

/-- Theorem stating the number of possible scores for the winning team in a specific cross-country race -/
theorem cross_country_race_winning_scores :
  ∃ (race : CrossCountryRace),
    race.players_per_team = 5 ∧
    race.total_players = 10 ∧
    race.total_score = (race.total_players * (race.total_players + 1)) / 2 ∧
    winning_score_count race = 13 := by
  sorry


end NUMINAMATH_CALUDE_cross_country_race_winning_scores_l1608_160892


namespace NUMINAMATH_CALUDE_events_A_B_independent_l1608_160865

structure GiftBox :=
  (chinese_knot : Bool)
  (notebook : Bool)
  (pencil_case : Bool)

def box1 : GiftBox := ⟨true, false, false⟩
def box2 : GiftBox := ⟨false, true, false⟩
def box3 : GiftBox := ⟨false, false, true⟩
def box4 : GiftBox := ⟨true, true, true⟩

def all_boxes : List GiftBox := [box1, box2, box3, box4]

def event_A (box : GiftBox) : Bool := box.chinese_knot
def event_B (box : GiftBox) : Bool := box.notebook

def prob_A : ℚ := (all_boxes.filter event_A).length / all_boxes.length
def prob_B : ℚ := (all_boxes.filter event_B).length / all_boxes.length
def prob_AB : ℚ := (all_boxes.filter (λ b => event_A b ∧ event_B b)).length / all_boxes.length

theorem events_A_B_independent : prob_A * prob_B = prob_AB := by sorry

end NUMINAMATH_CALUDE_events_A_B_independent_l1608_160865


namespace NUMINAMATH_CALUDE_pipe_stack_height_l1608_160883

theorem pipe_stack_height (d : ℝ) (h : ℝ) :
  d = 12 →
  h = 2 * d + d * Real.sqrt 3 →
  h = 24 + 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pipe_stack_height_l1608_160883


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l1608_160881

def f (x : ℝ) := -x^2 + 4*x - 2

theorem max_min_values_of_f :
  ∃ (x_max x_min : ℝ),
    x_max ∈ Set.Icc 0 3 ∧
    x_min ∈ Set.Icc 0 3 ∧
    (∀ x ∈ Set.Icc 0 3, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc 0 3, f x_min ≤ f x) ∧
    f x_max = 2 ∧
    f x_min = -2 :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l1608_160881


namespace NUMINAMATH_CALUDE_intersection_sum_l1608_160819

def M : Set ℝ := {x | |x - 4| + |x - 1| < 5}
def N (a : ℝ) : Set ℝ := {x | a < x ∧ x < 6}

theorem intersection_sum (a b : ℝ) : 
  M ∩ N a = {2, b} → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1608_160819


namespace NUMINAMATH_CALUDE_soup_distribution_l1608_160884

-- Define the total amount of soup
def total_soup : ℚ := 1

-- Define the number of grandchildren
def num_children : ℕ := 5

-- Define the amount taken by Ângela and Daniela
def angela_daniela_portion : ℚ := 2 / 5

-- Define Laura's division
def laura_division : ℕ := 5

-- Define João's division
def joao_division : ℕ := 4

-- Define the container size in ml
def container_size : ℕ := 100

-- Theorem statement
theorem soup_distribution (
  laura_portion : ℚ)
  (toni_portion : ℚ)
  (min_soup_amount : ℚ) :
  laura_portion = 3 / 25 ∧
  toni_portion = 9 / 25 ∧
  min_soup_amount = 5 / 2 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_soup_distribution_l1608_160884


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l1608_160897

theorem wire_ratio_proof (total_length shorter_length : ℝ) 
  (h1 : total_length = 35)
  (h2 : shorter_length = 10)
  (h3 : shorter_length < total_length) :
  shorter_length / (total_length - shorter_length) = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l1608_160897


namespace NUMINAMATH_CALUDE_percentage_calculation_l1608_160878

theorem percentage_calculation : 
  (0.60 * 4500 * 0.40 * 2800) - (0.80 * 1750 + 0.35 * 3000) = 3021550 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1608_160878


namespace NUMINAMATH_CALUDE_negation_of_exists_positive_power_l1608_160858

theorem negation_of_exists_positive_power (x : ℝ) : 
  (¬ (∃ x < 0, 2^x > 0)) ↔ (∀ x < 0, 2^x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_positive_power_l1608_160858


namespace NUMINAMATH_CALUDE_committee_selection_ways_l1608_160810

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_selection_ways : choose 12 5 = 792 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l1608_160810


namespace NUMINAMATH_CALUDE_hemisphere_on_cone_surface_area_l1608_160868

/-- The total surface area of a solid consisting of a hemisphere on top of a cone,
    where the area of the hemisphere's base is 144π and the height of the cone is
    twice the radius of the hemisphere. -/
theorem hemisphere_on_cone_surface_area :
  ∀ (r : ℝ),
  r > 0 →
  π * r^2 = 144 * π →
  let hemisphere_area := 2 * π * r^2
  let cone_height := 2 * r
  let cone_slant_height := Real.sqrt (r^2 + cone_height^2)
  let cone_area := π * r * cone_slant_height
  hemisphere_area + cone_area = 288 * π + 144 * Real.sqrt 5 * π :=
by
  sorry

end NUMINAMATH_CALUDE_hemisphere_on_cone_surface_area_l1608_160868


namespace NUMINAMATH_CALUDE_oatmeal_cookies_given_away_l1608_160829

/-- Represents the number of cookies in a dozen. -/
def dozen : ℕ := 12

/-- Represents the total number of cookies Ann baked. -/
def totalBaked : ℕ := 3 * dozen + 2 * dozen + 4 * dozen

/-- Represents the number of sugar cookies Ann gave away. -/
def sugarGivenAway : ℕ := (3 * dozen) / 2

/-- Represents the number of chocolate chip cookies Ann gave away. -/
def chocolateGivenAway : ℕ := (5 * dozen) / 2

/-- Represents the number of cookies Ann kept. -/
def cookiesKept : ℕ := 36

/-- Proves that Ann gave away 2 dozen oatmeal raisin cookies. -/
theorem oatmeal_cookies_given_away :
  ∃ (x : ℕ), x * dozen + sugarGivenAway + chocolateGivenAway + cookiesKept = totalBaked ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_oatmeal_cookies_given_away_l1608_160829


namespace NUMINAMATH_CALUDE_roots_equation_sum_l1608_160872

theorem roots_equation_sum (α β : ℝ) : 
  α^2 - 3*α - 4 = 0 → β^2 - 3*β - 4 = 0 → 3*α^3 + 7*β^4 = 1591 :=
by
  sorry

end NUMINAMATH_CALUDE_roots_equation_sum_l1608_160872


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l1608_160853

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x + 2|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ -x} = {x : ℝ | x ≤ -3 ∨ -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≤ a^2 + 1) → a ≤ -Real.sqrt 2 ∨ a ≥ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l1608_160853


namespace NUMINAMATH_CALUDE_largest_cube_surface_area_l1608_160857

/-- The surface area of the largest cube that can be cut from a cuboid -/
theorem largest_cube_surface_area (width length height : ℝ) 
  (hw : width = 12) (hl : length = 16) (hh : height = 14) : 
  let side_length := min width (min length height)
  6 * side_length^2 = 864 := by sorry

end NUMINAMATH_CALUDE_largest_cube_surface_area_l1608_160857


namespace NUMINAMATH_CALUDE_travel_probabilities_l1608_160832

/-- Represents a set of countries --/
structure CountrySet where
  asian : Finset Nat
  european : Finset Nat

/-- The probability of an event given the number of favorable outcomes and total outcomes --/
def probability (favorable : Nat) (total : Nat) : ℚ := favorable / total

/-- The total number of ways to choose 2 items from n items --/
def choose_two (n : Nat) : Nat := n * (n - 1) / 2

theorem travel_probabilities (countries : CountrySet) 
  (h1 : countries.asian.card = 3)
  (h2 : countries.european.card = 3) :
  (probability (choose_two 3) (choose_two 6) = 1 / 5) ∧ 
  (probability 2 9 = 2 / 9) := by
  sorry


end NUMINAMATH_CALUDE_travel_probabilities_l1608_160832


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l1608_160890

theorem sum_of_number_and_its_square (x : ℕ) : x = 14 → x + x^2 = 210 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l1608_160890


namespace NUMINAMATH_CALUDE_line_through_points_and_equal_intercepts_l1608_160894

-- Define points
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-2, 0)
def P : ℝ × ℝ := (-1, 3)

-- Define line equations
def line_eq_1 (x y : ℝ) : Prop := 2 * x - 5 * y + 4 = 0
def line_eq_2 (x y : ℝ) : Prop := x + y = 2

-- Define a function to check if a point lies on a line
def point_on_line (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line p.1 p.2

-- Define equal intercepts
def equal_intercepts (line : ℝ → ℝ → Prop) : Prop :=
  ∃ m : ℝ, line m 0 ∧ line 0 m

theorem line_through_points_and_equal_intercepts :
  (point_on_line A line_eq_1 ∧ point_on_line B line_eq_1) ∧
  (point_on_line P line_eq_2 ∧ equal_intercepts line_eq_2) :=
sorry

end NUMINAMATH_CALUDE_line_through_points_and_equal_intercepts_l1608_160894


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1608_160848

theorem simplify_square_roots : 
  (Real.sqrt 448 / Real.sqrt 128) + (Real.sqrt 98 / Real.sqrt 49) = 
  (Real.sqrt 14 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1608_160848


namespace NUMINAMATH_CALUDE_inequality_proof_l1608_160896

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π/2) : 
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1608_160896


namespace NUMINAMATH_CALUDE_max_a_value_l1608_160864

def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem max_a_value (a : ℝ) :
  (A ⊂ B a) → (∀ b, (A ⊂ B b) → a ≥ b) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_l1608_160864


namespace NUMINAMATH_CALUDE_students_taller_than_yoongi_l1608_160805

theorem students_taller_than_yoongi 
  (total_students : ℕ) 
  (shorter_than_yoongi : ℕ) 
  (h1 : total_students = 20) 
  (h2 : shorter_than_yoongi = 11) : 
  total_students - shorter_than_yoongi - 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_taller_than_yoongi_l1608_160805


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l1608_160814

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 9 ∧
  (∀ x : ℝ, x^2 - 12*x + 27 ≤ 0 → x ≤ x_max) ∧
  (x_max^2 - 12*x_max + 27 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l1608_160814


namespace NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l1608_160876

theorem gcd_seven_eight_factorial : 
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l1608_160876


namespace NUMINAMATH_CALUDE_pictures_in_first_album_l1608_160828

theorem pictures_in_first_album 
  (total_pictures : ℕ) 
  (num_albums : ℕ) 
  (pics_per_album : ℕ) 
  (h1 : total_pictures = 65)
  (h2 : num_albums = 6)
  (h3 : pics_per_album = 8) :
  total_pictures - (num_albums * pics_per_album) = 17 :=
by sorry

end NUMINAMATH_CALUDE_pictures_in_first_album_l1608_160828


namespace NUMINAMATH_CALUDE_root_sum_product_l1608_160821

theorem root_sum_product (p q r : ℝ) : 
  (5 * p^3 - 10 * p^2 + 17 * p - 7 = 0) ∧ 
  (5 * q^3 - 10 * q^2 + 17 * q - 7 = 0) ∧ 
  (5 * r^3 - 10 * r^2 + 17 * r - 7 = 0) → 
  p * q + p * r + q * r = 17 / 5 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l1608_160821


namespace NUMINAMATH_CALUDE_short_trees_after_planting_park_short_trees_l1608_160804

/-- The number of short trees in a park after planting new trees -/
def total_short_trees (initial_short_trees new_short_trees : ℕ) : ℕ :=
  initial_short_trees + new_short_trees

/-- Theorem: The total number of short trees after planting is the sum of initial and new short trees -/
theorem short_trees_after_planting 
  (initial_short_trees : ℕ) 
  (new_short_trees : ℕ) : 
  total_short_trees initial_short_trees new_short_trees = initial_short_trees + new_short_trees := by
  sorry

/-- Application to the specific problem -/
theorem park_short_trees : total_short_trees 3 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_after_planting_park_short_trees_l1608_160804


namespace NUMINAMATH_CALUDE_independence_day_bananas_l1608_160825

theorem independence_day_bananas (total_children : ℕ) 
  (present_children : ℕ) (absent_children : ℕ) (bananas : ℕ) : 
  total_children = 260 →
  bananas = 4 * present_children →
  bananas = 2 * total_children →
  present_children + absent_children = total_children →
  absent_children = 130 := by
sorry

end NUMINAMATH_CALUDE_independence_day_bananas_l1608_160825


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l1608_160889

/-- The number of walnut trees planted in the park -/
def trees_planted (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that 33 walnut trees were planted -/
theorem walnut_trees_planted :
  trees_planted 22 55 = 33 := by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l1608_160889


namespace NUMINAMATH_CALUDE_geoffrey_remaining_money_l1608_160850

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial_amount : ℕ) (num_items : ℕ) (item_cost : ℕ) : ℕ :=
  initial_amount - num_items * item_cost

/-- Proves that Geoffrey has €20 left after his purchase -/
theorem geoffrey_remaining_money :
  remaining_money 125 3 35 = 20 := by
  sorry

end NUMINAMATH_CALUDE_geoffrey_remaining_money_l1608_160850


namespace NUMINAMATH_CALUDE_sum_in_B_l1608_160802

def A : Set ℤ := {x | ∃ k, x = 2 * k}
def B : Set ℤ := {x | ∃ k, x = 2 * k + 1}
def C : Set ℤ := {x | ∃ k, x = 4 * k + 1}

theorem sum_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end NUMINAMATH_CALUDE_sum_in_B_l1608_160802


namespace NUMINAMATH_CALUDE_remainder_17_power_2023_mod_28_l1608_160830

theorem remainder_17_power_2023_mod_28 : 17^2023 % 28 = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_power_2023_mod_28_l1608_160830


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1608_160863

theorem quadratic_no_real_roots : ∀ x : ℝ, 3 * x^2 - 6 * x + 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1608_160863


namespace NUMINAMATH_CALUDE_inequality_proof_l1608_160867

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1608_160867


namespace NUMINAMATH_CALUDE_problem_solution_l1608_160809

theorem problem_solution (a b : ℝ) : 
  a = 105 ∧ a^3 = 21 * 49 * 45 * b → b = 12.5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1608_160809


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l1608_160898

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_sum_theorem (p q : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (h1 : is_prime (7*p + q)) 
  (h2 : is_prime (p*q + 11)) : 
  p^q + q^p = 17 := by sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l1608_160898


namespace NUMINAMATH_CALUDE_unique_square_numbers_l1608_160856

theorem unique_square_numbers : ∃! (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  ∃ (m n : ℕ), 
    (100 * a + b = m^2) ∧ 
    (201 * a + b = n^2) ∧ 
    1000 ≤ m^2 ∧ m^2 < 10000 ∧ 
    1000 ≤ n^2 ∧ n^2 < 10000 ∧
    a = 17 ∧ b = 64 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_numbers_l1608_160856


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1608_160844

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ a*x - b*y + 2 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁^2 + y₁^2 + 2*x₁ - 4*y₁ + 1 = 0 ∧ 
                         x₂^2 + y₂^2 + 2*x₂ - 4*y₂ + 1 = 0 ∧
                         a*x₁ - b*y₁ + 2 = 0 ∧ a*x₂ - b*y₂ + 2 = 0 ∧
                         (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16) →
  (∀ c d : ℝ, c > 0 → d > 0 → 
    (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ c*x - d*y + 2 = 0) →
    1/a + 1/b ≤ 1/c + 1/d) →
  1/a + 1/b = 3/2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1608_160844


namespace NUMINAMATH_CALUDE_marble_ratio_is_two_to_one_l1608_160886

/-- The ratio of Mary's blue marbles to Dan's blue marbles -/
def marble_ratio (dans_marbles marys_marbles : ℕ) : ℚ :=
  marys_marbles / dans_marbles

/-- Proof that the ratio of Mary's blue marbles to Dan's blue marbles is 2:1 -/
theorem marble_ratio_is_two_to_one :
  marble_ratio 5 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_is_two_to_one_l1608_160886


namespace NUMINAMATH_CALUDE_twelfth_even_multiple_of_5_l1608_160879

-- Define a function that represents the nth positive integer that is both even and a multiple of 5
def evenMultipleOf5 (n : ℕ) : ℕ := 10 * n

-- State the theorem
theorem twelfth_even_multiple_of_5 : evenMultipleOf5 12 = 120 := by sorry

end NUMINAMATH_CALUDE_twelfth_even_multiple_of_5_l1608_160879


namespace NUMINAMATH_CALUDE_fish_tank_problem_l1608_160860

/-- Represents the number of gallons needed for each of the smaller tanks -/
def smaller_tank_gallons (total_weekly_gallons : ℕ) : ℕ :=
  (total_weekly_gallons - 2 * 8) / 2

/-- Represents the difference in gallons between larger and smaller tanks -/
def gallon_difference (total_weekly_gallons : ℕ) : ℕ :=
  8 - smaller_tank_gallons total_weekly_gallons

theorem fish_tank_problem (total_gallons : ℕ) 
  (h1 : total_gallons = 112) 
  (h2 : total_gallons % 4 = 0) : 
  gallon_difference (total_gallons / 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l1608_160860


namespace NUMINAMATH_CALUDE_max_f_value_1997_l1608_160855

def f : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => f (n / 2) + (n + 2) - 2 * (n / 2)

theorem max_f_value_1997 :
  (∃ (n : ℕ), n ≤ 1997 ∧ f n = 10) ∧
  (∀ (n : ℕ), n ≤ 1997 → f n ≤ 10) :=
sorry

end NUMINAMATH_CALUDE_max_f_value_1997_l1608_160855


namespace NUMINAMATH_CALUDE_det_cofactor_matrix_cube_l1608_160891

/-- For a 4x4 matrix A, the determinant of its cofactor matrix B is equal to the cube of the determinant of A. -/
theorem det_cofactor_matrix_cube (A : Matrix (Fin 4) (Fin 4) ℝ) :
  let d := Matrix.det A
  let B := Matrix.adjugate A
  Matrix.det B = d^3 := by sorry

end NUMINAMATH_CALUDE_det_cofactor_matrix_cube_l1608_160891


namespace NUMINAMATH_CALUDE_divisibility_by_23_and_29_l1608_160842

theorem divisibility_by_23_and_29 (a b c : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) (hc : c ≤ 9) :
  ∃ (k m : ℕ), 200100 * a + 20010 * b + 2001 * c = 23 * k ∧ 200100 * a + 20010 * b + 2001 * c = 29 * m := by
  sorry

#check divisibility_by_23_and_29

end NUMINAMATH_CALUDE_divisibility_by_23_and_29_l1608_160842


namespace NUMINAMATH_CALUDE_paint_coverage_l1608_160854

/-- Proves that a gallon of paint covers 400 square feet given the problem conditions -/
theorem paint_coverage 
  (paint_cost : ℝ) 
  (wall_area : ℝ) 
  (coats : ℕ) 
  (contribution : ℝ) :
  paint_cost = 45 →
  wall_area = 1600 →
  coats = 2 →
  contribution = 180 →
  (2 * contribution) / paint_cost * wall_area * coats / ((2 * contribution) / paint_cost) = 400 :=
by
  sorry

#check paint_coverage

end NUMINAMATH_CALUDE_paint_coverage_l1608_160854


namespace NUMINAMATH_CALUDE_max_m_value_max_m_is_75_l1608_160812

theorem max_m_value (m n : ℕ+) (h : 8 * m + 9 * n = m * n + 6) : 
  ∀ k : ℕ+, 8 * k + 9 * n = k * n + 6 → k ≤ m :=
sorry

theorem max_m_is_75 : ∃ m n : ℕ+, 8 * m + 9 * n = m * n + 6 ∧ m = 75 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_max_m_is_75_l1608_160812


namespace NUMINAMATH_CALUDE_third_row_sum_is_226_l1608_160822

/-- Represents a position in the grid -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the spiral grid -/
def SpiralGrid :=
  Position → Nat

/-- The size of the grid -/
def gridSize : Nat := 13

/-- The starting number -/
def startNum : Nat := 100

/-- The ending number -/
def endNum : Nat := 268

/-- The center position of the grid -/
def centerPos : Position :=
  { row := 6, col := 6 }  -- 0-based index

/-- Generates the spiral grid -/
def generateSpiralGrid : SpiralGrid :=
  sorry

/-- Gets the numbers in the third row -/
def getThirdRowNumbers (grid : SpiralGrid) : List Nat :=
  sorry

/-- Theorem: The sum of the greatest and least numbers in the third row is 226 -/
theorem third_row_sum_is_226 (grid : SpiralGrid) :
  grid = generateSpiralGrid →
  let thirdRowNums := getThirdRowNumbers grid
  (List.maximum thirdRowNums).getD 0 + (List.minimum thirdRowNums).getD 0 = 226 :=
sorry

end NUMINAMATH_CALUDE_third_row_sum_is_226_l1608_160822


namespace NUMINAMATH_CALUDE_annika_hans_age_multiple_l1608_160826

/-- Proves that in four years, Annika's age will be 3 times Hans' age -/
theorem annika_hans_age_multiple :
  ∀ (hans_current_age annika_current_age years_elapsed : ℕ),
    hans_current_age = 8 →
    annika_current_age = 32 →
    years_elapsed = 4 →
    (annika_current_age + years_elapsed) = 3 * (hans_current_age + years_elapsed) :=
by sorry

end NUMINAMATH_CALUDE_annika_hans_age_multiple_l1608_160826


namespace NUMINAMATH_CALUDE_intersection_point_of_perpendicular_chords_l1608_160800

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line
def line (m b x y : ℝ) : Prop := x = m*y + b

-- Define perpendicularity of two points with respect to the origin
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

theorem intersection_point_of_perpendicular_chords :
  ∀ (m b x₁ y₁ x₂ y₂ : ℝ),
  parabola x₁ y₁ →
  parabola x₂ y₂ →
  line m b x₁ y₁ →
  line m b x₂ y₂ →
  perpendicular x₁ y₁ x₂ y₂ →
  ∃ (x y : ℝ), line m b x y ∧ x = 2 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_of_perpendicular_chords_l1608_160800


namespace NUMINAMATH_CALUDE_periodic_trig_function_value_l1608_160888

theorem periodic_trig_function_value (m n α₁ α₂ : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hα₁ : α₁ ≠ 0) (hα₂ : α₂ ≠ 0) :
  let f : ℝ → ℝ := λ x => m * Real.sin (π * x + α₁) + n * Real.cos (π * x + α₂)
  f 2011 = 1 → f 2012 = -1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_trig_function_value_l1608_160888


namespace NUMINAMATH_CALUDE_intersection_M_N_l1608_160824

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x > 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the interval [1, 2)
def interval_1_2 : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = interval_1_2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1608_160824


namespace NUMINAMATH_CALUDE_correct_prediction_probability_l1608_160846

def n_monday : ℕ := 5
def n_tuesday : ℕ := 6
def n_total : ℕ := n_monday + n_tuesday
def n_correct : ℕ := 7
def n_correct_monday : ℕ := 3

theorem correct_prediction_probability :
  let p : ℝ := 1 / 2
  (Nat.choose n_monday n_correct_monday * p ^ n_monday * (1 - p) ^ (n_monday - n_correct_monday)) *
  (Nat.choose n_tuesday (n_correct - n_correct_monday) * p ^ (n_correct - n_correct_monday) * (1 - p) ^ (n_tuesday - (n_correct - n_correct_monday))) /
  (Nat.choose n_total n_correct * p ^ n_correct * (1 - p) ^ (n_total - n_correct)) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_correct_prediction_probability_l1608_160846


namespace NUMINAMATH_CALUDE_car_speed_l1608_160845

/-- The speed of a car in km/h given the tire's rotation rate and circumference -/
theorem car_speed (revolutions_per_minute : ℝ) (tire_circumference : ℝ) : 
  revolutions_per_minute = 400 → 
  tire_circumference = 1 → 
  (revolutions_per_minute * tire_circumference * 60) / 1000 = 24 := by
sorry

end NUMINAMATH_CALUDE_car_speed_l1608_160845


namespace NUMINAMATH_CALUDE_triangle_area_form_l1608_160893

/-- The radius of each circle -/
def r : ℝ := 44

/-- The side length of the equilateral triangle -/
noncomputable def s : ℝ := 2 * r * Real.sqrt 3

/-- The area of the equilateral triangle -/
noncomputable def area : ℝ := (s^2 * Real.sqrt 3) / 4

/-- Theorem stating the form of the area -/
theorem triangle_area_form :
  ∃ (a b : ℕ), area = Real.sqrt a + Real.sqrt b :=
sorry

end NUMINAMATH_CALUDE_triangle_area_form_l1608_160893


namespace NUMINAMATH_CALUDE_quadratic_below_x_axis_iff_a_in_range_l1608_160840

/-- A quadratic function f(x) = ax^2 + 2ax - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x - 2

/-- The property that the graph of f is always below the x-axis -/
def always_below_x_axis (a : ℝ) : Prop :=
  ∀ x, f a x < 0

theorem quadratic_below_x_axis_iff_a_in_range :
  ∀ a : ℝ, always_below_x_axis a ↔ -2 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_below_x_axis_iff_a_in_range_l1608_160840


namespace NUMINAMATH_CALUDE_three_four_five_pythagorean_triple_l1608_160815

/-- A Pythagorean triple is a set of three positive integers a, b, and c that satisfy a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The set (3, 4, 5) is a Pythagorean triple -/
theorem three_four_five_pythagorean_triple : is_pythagorean_triple 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_three_four_five_pythagorean_triple_l1608_160815


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1608_160851

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal or both lines are vertical -/
def parallel (l1 l2 : Line) : Prop :=
  (l1.b ≠ 0 ∧ l2.b ≠ 0 ∧ l1.a / l1.b = l2.a / l2.b) ∨
  (l1.b = 0 ∧ l2.b = 0)

theorem parallel_lines_a_value (a : ℝ) :
  let l1 : Line := ⟨a, 2, a⟩
  let l2 : Line := ⟨3*a, a-1, 7⟩
  parallel l1 l2 → a = 0 ∨ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1608_160851


namespace NUMINAMATH_CALUDE_jelly_beans_distribution_l1608_160869

theorem jelly_beans_distribution (initial_beans : ℕ) (remaining_beans : ℕ) 
  (h1 : initial_beans = 8000)
  (h2 : remaining_beans = 1600) :
  ∃ (x : ℕ), 
    x = 400 ∧ 
    initial_beans - remaining_beans = 6 * (2 * x) + 4 * x :=
by sorry

end NUMINAMATH_CALUDE_jelly_beans_distribution_l1608_160869


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1608_160836

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1608_160836


namespace NUMINAMATH_CALUDE_mildred_oranges_l1608_160823

/-- The number of oranges Mildred's father ate -/
def fatherAte : ℕ := 2

/-- The number of oranges Mildred has now -/
def currentOranges : ℕ := 75

/-- The initial number of oranges Mildred collected -/
def initialOranges : ℕ := currentOranges + fatherAte

theorem mildred_oranges : initialOranges = 77 := by
  sorry

end NUMINAMATH_CALUDE_mildred_oranges_l1608_160823


namespace NUMINAMATH_CALUDE_census_suitability_l1608_160861

/-- Represents a survey --/
structure Survey where
  description : String
  population_size : Nat
  ease_of_survey : Bool

/-- Defines when a survey is suitable for a census --/
def suitable_for_census (s : Survey) : Prop :=
  s.population_size < 1000 ∧ s.ease_of_survey

/-- Theorem stating the condition for a survey to be suitable for a census --/
theorem census_suitability (s : Survey) :
  suitable_for_census s ↔ s.population_size < 1000 ∧ s.ease_of_survey := by sorry

end NUMINAMATH_CALUDE_census_suitability_l1608_160861


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1608_160801

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), (2*a*x - b*y + 2 = 0 ∧ 
                 x^2 + y^2 + 2*x - 4*y + 1 = 0) ∧
   (∃ (x1 y1 x2 y2 : ℝ), 
      (2*a*x1 - b*y1 + 2 = 0 ∧ x1^2 + y1^2 + 2*x1 - 4*y1 + 1 = 0) ∧
      (2*a*x2 - b*y2 + 2 = 0 ∧ x2^2 + y2^2 + 2*x2 - 4*y2 + 1 = 0) ∧
      ((x1 - x2)^2 + (y1 - y2)^2 = 16))) →
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 1/a' + 1/b' ≥ 1/a + 1/b) →
  1/a + 1/b = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1608_160801


namespace NUMINAMATH_CALUDE_min_value_theorem_l1608_160882

theorem min_value_theorem (a b : ℝ) (h : a * b > 0) :
  (a^4 + 4 * b^4 + 1) / (a * b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1608_160882


namespace NUMINAMATH_CALUDE_count_five_digit_numbers_with_one_odd_l1608_160831

/-- The count of five-digit numbers with exactly one odd digit -/
def five_digit_numbers_with_one_odd : ℕ :=
  let odd_digits := 5  -- Count of odd digits (1, 3, 5, 7, 9)
  let even_digits := 5  -- Count of even digits (0, 2, 4, 6, 8)
  let first_digit_odd := odd_digits * even_digits^4
  let other_digit_odd := 4 * odd_digits * (even_digits - 1) * even_digits^3
  first_digit_odd + other_digit_odd

theorem count_five_digit_numbers_with_one_odd :
  five_digit_numbers_with_one_odd = 10625 := by
  sorry

end NUMINAMATH_CALUDE_count_five_digit_numbers_with_one_odd_l1608_160831


namespace NUMINAMATH_CALUDE_teacher_earnings_five_weeks_l1608_160887

/-- Calculates the teacher's earnings for piano lessons over a given number of weeks -/
def teacher_earnings (rate_per_half_hour : ℕ) (lesson_duration_hours : ℕ) (weeks : ℕ) : ℕ :=
  rate_per_half_hour * 2 * lesson_duration_hours * weeks

/-- Proves that the teacher earns $100 in 5 weeks under the given conditions -/
theorem teacher_earnings_five_weeks :
  teacher_earnings 10 1 5 = 100 :=
by
  sorry

#eval teacher_earnings 10 1 5

end NUMINAMATH_CALUDE_teacher_earnings_five_weeks_l1608_160887


namespace NUMINAMATH_CALUDE_product_234_75_in_base5_l1608_160837

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of base 5 digits to a decimal number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

/-- Multiplies two numbers in base 5 representation -/
def multiplyBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem product_234_75_in_base5 :
  let a := toBase5 234
  let b := toBase5 75
  multiplyBase5 a b = [4, 5, 0, 6, 2, 0] :=
sorry

end NUMINAMATH_CALUDE_product_234_75_in_base5_l1608_160837


namespace NUMINAMATH_CALUDE_count_integers_with_conditions_l1608_160841

theorem count_integers_with_conditions : 
  ∃ (S : Finset ℤ), 
    (∀ n ∈ S, 150 < n ∧ n < 300 ∧ n % 7 = n % 9) ∧ 
    (∀ n, 150 < n → n < 300 → n % 7 = n % 9 → n ∈ S) ∧ 
    Finset.card S = 14 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_conditions_l1608_160841


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1608_160873

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x = 2 ↔ (x - 1)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1608_160873


namespace NUMINAMATH_CALUDE_polynomial_inequality_l1608_160875

theorem polynomial_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1/2) →
  (∀ x : ℝ, |x| ≥ 1 → |a * x^2 + b * x + c| ≤ x^2 - 1/2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l1608_160875


namespace NUMINAMATH_CALUDE_pineapple_sweets_count_l1608_160838

/-- Proves the number of initial pineapple-flavored sweets in a candy packet --/
theorem pineapple_sweets_count (cherry : ℕ) (strawberry : ℕ) (remaining : ℕ) : 
  cherry = 30 → 
  strawberry = 40 → 
  remaining = 55 → 
  ∃ (pineapple : ℕ), 
    pineapple + cherry + strawberry = 
    2 * remaining + 5 + (cherry / 2) + (strawberry / 2) ∧ 
    pineapple = 50 := by
  sorry

#check pineapple_sweets_count

end NUMINAMATH_CALUDE_pineapple_sweets_count_l1608_160838


namespace NUMINAMATH_CALUDE_largest_unattainable_integer_l1608_160808

/-- Given positive integers a, b, c with no pairwise common divisor greater than 1,
    2abc-ab-bc-ca is the largest integer that cannot be expressed as xbc+yca+zab
    for non-negative integers x, y, z -/
theorem largest_unattainable_integer (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : Nat.gcd a b = 1) (hbc : Nat.gcd b c = 1) (hac : Nat.gcd a c = 1) :
  (∀ x y z : ℕ, x * b * c + y * c * a + z * a * b ≠ 2 * a * b * c - a * b - b * c - c * a) ∧
  (∀ n : ℕ, n > 2 * a * b * c - a * b - b * c - c * a →
    ∃ x y z : ℕ, x * b * c + y * c * a + z * a * b = n) :=
by sorry

end NUMINAMATH_CALUDE_largest_unattainable_integer_l1608_160808


namespace NUMINAMATH_CALUDE_remaining_three_digit_numbers_l1608_160877

/-- The count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers where the first and last digits are the same
    and the middle digit is different -/
def excluded_numbers : ℕ := 81

/-- Theorem: The count of three-digit numbers excluding those where the first and last digits
    are the same and the middle digit is different is equal to 819 -/
theorem remaining_three_digit_numbers :
  total_three_digit_numbers - excluded_numbers = 819 := by
  sorry

end NUMINAMATH_CALUDE_remaining_three_digit_numbers_l1608_160877


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1608_160862

theorem root_sum_theorem (p q r s : ℂ) : 
  p^4 - 15*p^3 + 35*p^2 - 27*p + 9 = 0 →
  q^4 - 15*q^3 + 35*q^2 - 27*q + 9 = 0 →
  r^4 - 15*r^3 + 35*r^2 - 27*r + 9 = 0 →
  s^4 - 15*s^3 + 35*s^2 - 27*s + 9 = 0 →
  p / (1/p + q*r) + q / (1/q + r*s) + r / (1/r + s*p) + s / (1/s + p*q) = 155/123 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1608_160862


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1608_160859

def M : Set Int := {-1, 0, 1, 3, 5}
def N : Set Int := {-2, 1, 2, 3, 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1608_160859


namespace NUMINAMATH_CALUDE_rectangle_area_is_143_l1608_160852

/-- Represents a square in the rectangle --/
structure Square where
  sideLength : ℝ
  area : ℝ
  area_eq : area = sideLength ^ 2

/-- Represents the rectangle ABCD --/
structure Rectangle where
  squares : Fin 6 → Square
  smallestSquare : squares 0 = { sideLength := 1, area := 1, area_eq := by simp }
  unequal : ∀ i j, i ≠ j → squares i ≠ squares j
  width : ℝ
  height : ℝ
  area : ℝ
  area_eq : area = width * height
  width_eq : width = (squares 1).sideLength + (squares 2).sideLength + (squares 0).sideLength
  height_eq : height = (squares 3).sideLength + (squares 0).sideLength + 1

theorem rectangle_area_is_143 (rect : Rectangle) : rect.area = 143 := by
  sorry

#check rectangle_area_is_143

end NUMINAMATH_CALUDE_rectangle_area_is_143_l1608_160852


namespace NUMINAMATH_CALUDE_striped_shirts_difference_l1608_160880

theorem striped_shirts_difference (total : ℕ) (striped_ratio : ℚ) (checkered_ratio : ℚ) 
  (h_total : total = 120)
  (h_striped : striped_ratio = 3/5)
  (h_checkered : checkered_ratio = 1/4)
  (h_shorts_plain : ∃ (plain : ℕ) (shorts : ℕ), 
    plain = total - (striped_ratio * total).num - (checkered_ratio * total).num ∧
    shorts + 10 = plain) :
  ∃ (striped : ℕ) (shorts : ℕ),
    striped = (striped_ratio * total).num ∧
    striped - shorts = 44 :=
sorry

end NUMINAMATH_CALUDE_striped_shirts_difference_l1608_160880


namespace NUMINAMATH_CALUDE_cubic_identity_l1608_160871

theorem cubic_identity (a b c : ℝ) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l1608_160871


namespace NUMINAMATH_CALUDE_female_percentage_l1608_160847

/-- Represents a classroom with double desks -/
structure Classroom where
  male_students : ℕ
  female_students : ℕ
  male_with_male : ℕ
  female_with_female : ℕ

/-- All seats are occupied and the percentages of same-gender pairings are as given -/
def valid_classroom (c : Classroom) : Prop :=
  c.male_with_male = (6 * c.male_students) / 10 ∧
  c.female_with_female = (2 * c.female_students) / 10 ∧
  c.male_students - c.male_with_male = c.female_students - c.female_with_female

theorem female_percentage (c : Classroom) (h : valid_classroom c) :
  (c.female_students : ℚ) / (c.male_students + c.female_students) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_female_percentage_l1608_160847


namespace NUMINAMATH_CALUDE_friends_team_assignment_l1608_160895

theorem friends_team_assignment (n : ℕ) (k : ℕ) :
  (n = 8 ∧ k = 4) →
  (number_of_assignments : ℕ) →
  number_of_assignments = k^n :=
by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l1608_160895


namespace NUMINAMATH_CALUDE_min_value_theorem_l1608_160803

/-- An arithmetic sequence with given properties -/
structure ArithSeq where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  h1 : a 2 = 4
  h2 : S 10 = 110

/-- The main theorem -/
theorem min_value_theorem (seq : ArithSeq) :
  ∀ n : ℕ, n ≥ 1 → (seq.S n + 64) / seq.a n ≥ 17/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1608_160803


namespace NUMINAMATH_CALUDE_canning_box_theorem_l1608_160817

/-- Represents the solution to the canning box problem -/
def canning_box_solution (total_sheets : ℕ) (bodies_per_sheet : ℕ) (bottoms_per_sheet : ℕ) 
  (sheets_for_bodies : ℕ) (sheets_for_bottoms : ℕ) : Prop :=
  -- All sheets are used
  sheets_for_bodies + sheets_for_bottoms = total_sheets ∧
  -- Number of bodies matches half the number of bottoms
  bodies_per_sheet * sheets_for_bodies = (bottoms_per_sheet * sheets_for_bottoms) / 2 ∧
  -- Solution is optimal (no other solution exists)
  ∀ (x y : ℕ), 
    x + y = total_sheets ∧ 
    bodies_per_sheet * x = (bottoms_per_sheet * y) / 2 → 
    x ≤ sheets_for_bodies ∧ y ≤ sheets_for_bottoms

/-- The canning box theorem -/
theorem canning_box_theorem : 
  canning_box_solution 33 30 50 15 18 := by
  sorry

end NUMINAMATH_CALUDE_canning_box_theorem_l1608_160817


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_implies_m_l1608_160818

/-- Represents a hyperbola with equation mx^2 + y^2 = 1 -/
structure Hyperbola (m : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + y^2 = 1

/-- The length of the imaginary axis of the hyperbola -/
def imaginary_axis_length (h : Hyperbola m) : ℝ := sorry

/-- The length of the real axis of the hyperbola -/
def real_axis_length (h : Hyperbola m) : ℝ := sorry

/-- 
  Theorem: For a hyperbola with equation mx^2 + y^2 = 1, 
  if the length of the imaginary axis is twice the length of the real axis, 
  then m = -1/4
-/
theorem hyperbola_axis_ratio_implies_m (m : ℝ) (h : Hyperbola m) 
  (axis_ratio : imaginary_axis_length h = 2 * real_axis_length h) : 
  m = -1/4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_implies_m_l1608_160818


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l1608_160813

/-- The quadratic polynomial q(x) that satisfies specific conditions -/
def q (x : ℚ) : ℚ := -25/11 * x^2 + 75/11 * x + 450/11

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-3) = 0 ∧ q 6 = 0 ∧ q 8 = -50 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l1608_160813


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_zero_l1608_160874

theorem factorial_fraction_equals_zero : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 : ℚ) / 
  Nat.factorial (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) = 0 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_zero_l1608_160874
