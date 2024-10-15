import Mathlib

namespace NUMINAMATH_CALUDE_certain_number_equation_l3824_382413

theorem certain_number_equation (x : ℚ) : 4 / (1 + 3 / x) = 1 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3824_382413


namespace NUMINAMATH_CALUDE_number_of_boys_l3824_382426

theorem number_of_boys (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  girls = 120 →
  boys + girls = total →
  3 * total = 8 * boys →
  boys = 72 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l3824_382426


namespace NUMINAMATH_CALUDE_triangle_inequality_tangent_l3824_382487

theorem triangle_inequality_tangent (a b c α β : ℝ) 
  (h : a + b < 3 * c) : 
  Real.tan (α / 2) * Real.tan (β / 2) < 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_tangent_l3824_382487


namespace NUMINAMATH_CALUDE_caiden_roofing_cost_l3824_382410

/-- Calculates the cost of remaining metal roofing needed -/
def roofing_cost (total_required : ℕ) (free_provided : ℕ) (cost_per_foot : ℕ) : ℕ :=
  (total_required - free_provided) * cost_per_foot

/-- Theorem stating the cost calculation for Mr. Caiden's roofing -/
theorem caiden_roofing_cost :
  roofing_cost 300 250 8 = 400 := by
  sorry

end NUMINAMATH_CALUDE_caiden_roofing_cost_l3824_382410


namespace NUMINAMATH_CALUDE_subject_selection_methods_l3824_382462

/-- The number of subjects excluding the mandatory subject -/
def n : ℕ := 5

/-- The number of subjects to be chosen from the remaining subjects -/
def k : ℕ := 2

/-- Combination formula -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem subject_selection_methods :
  combination n k = 10 :=
by sorry

end NUMINAMATH_CALUDE_subject_selection_methods_l3824_382462


namespace NUMINAMATH_CALUDE_olivia_coin_device_l3824_382405

def coin_change (start : ℕ) (change : ℕ) (target : ℕ) : Prop :=
  ∃ k : ℕ, start + k * (change - 1) = target

theorem olivia_coin_device (targets : List ℕ := [492, 776, 1248, 1520, 1984]) :
  ∀ t ∈ targets, (coin_change 1 80 t ↔ t = 1984) := by sorry

end NUMINAMATH_CALUDE_olivia_coin_device_l3824_382405


namespace NUMINAMATH_CALUDE_set_equality_l3824_382447

open Set

def U : Set ℝ := univ
def E : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def F : Set ℝ := {x | -1 < x ∧ x < 5}

theorem set_equality : {x : ℝ | -1 < x ∧ x < 2} = (U \ E) ∩ F := by sorry

end NUMINAMATH_CALUDE_set_equality_l3824_382447


namespace NUMINAMATH_CALUDE_total_population_of_two_villages_l3824_382469

/-- Given two villages A and B with the following properties:
    - 90% of Village A's population is 23040
    - 80% of Village B's population is 17280
    - Village A has three times as many children as Village B
    - The adult population is equally distributed between the two villages
    Prove that the total population of both villages combined is 47,200 -/
theorem total_population_of_two_villages :
  ∀ (population_A population_B children_A children_B : ℕ),
    (population_A : ℚ) * (9 / 10) = 23040 →
    (population_B : ℚ) * (4 / 5) = 17280 →
    children_A = 3 * children_B →
    population_A - children_A = population_B - children_B →
    population_A + population_B = 47200 := by
  sorry

#eval 47200

end NUMINAMATH_CALUDE_total_population_of_two_villages_l3824_382469


namespace NUMINAMATH_CALUDE_street_trees_l3824_382407

theorem street_trees (road_length : ℝ) (tree_spacing : ℝ) (h1 : road_length = 268.8) (h2 : tree_spacing = 6.4) : 
  ⌊road_length / tree_spacing⌋ + 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_street_trees_l3824_382407


namespace NUMINAMATH_CALUDE_rhombus_side_length_l3824_382472

/-- A rhombus with area L and longer diagonal three times the shorter diagonal has side length √(5L/3) -/
theorem rhombus_side_length (L : ℝ) (h : L > 0) : 
  ∃ (short_diag long_diag side : ℝ),
    short_diag > 0 ∧
    long_diag = 3 * short_diag ∧
    L = (1/2) * short_diag * long_diag ∧
    side^2 = (short_diag/2)^2 + (long_diag/2)^2 ∧
    side = Real.sqrt ((5 * L) / 3) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l3824_382472


namespace NUMINAMATH_CALUDE_spring_work_compression_l3824_382452

/-- Given a spring that is compressed 1 cm by a 10 N force, 
    the work done to compress it by 10 cm is 5 J. -/
theorem spring_work_compression (k : ℝ) : 
  (10 : ℝ) = k * 1 → (∫ x in (0 : ℝ)..(10 : ℝ), k * x) = 5 := by
  sorry

end NUMINAMATH_CALUDE_spring_work_compression_l3824_382452


namespace NUMINAMATH_CALUDE_macys_weekly_goal_l3824_382497

/-- Macy's weekly running goal in miles -/
def weekly_goal : ℕ := 24

/-- Miles Macy runs per day -/
def miles_per_day : ℕ := 3

/-- Number of days Macy has run -/
def days_run : ℕ := 6

/-- Miles left to run after 6 days -/
def miles_left : ℕ := 6

/-- Theorem stating Macy's weekly running goal -/
theorem macys_weekly_goal : 
  weekly_goal = miles_per_day * days_run + miles_left := by sorry

end NUMINAMATH_CALUDE_macys_weekly_goal_l3824_382497


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l3824_382455

def is_perpendicular_bisector (a b c : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  a * midpoint.1 + b * midpoint.2 + c = 0

theorem perpendicular_bisector_equation (b : ℝ) :
  is_perpendicular_bisector 1 (-1) (-b) (2, 4) (10, -6) → b = 7 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l3824_382455


namespace NUMINAMATH_CALUDE_men_to_women_percentage_l3824_382442

theorem men_to_women_percentage (men women : ℕ) (h : women = men / 2) :
  (men : ℚ) / (women : ℚ) * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_men_to_women_percentage_l3824_382442


namespace NUMINAMATH_CALUDE_overtake_time_l3824_382430

/-- The time when b starts relative to a's start time. -/
def b_start_time : ℝ := 15

/-- The speed of person a in km/hr. -/
def speed_a : ℝ := 30

/-- The speed of person b in km/hr. -/
def speed_b : ℝ := 40

/-- The speed of person k in km/hr. -/
def speed_k : ℝ := 60

/-- The time when k starts relative to a's start time. -/
def k_start_time : ℝ := 10

theorem overtake_time (t : ℝ) : 
  speed_a * t = speed_b * (t - b_start_time) ∧ 
  speed_a * t = speed_k * (t - k_start_time) → 
  b_start_time = 15 := by sorry

end NUMINAMATH_CALUDE_overtake_time_l3824_382430


namespace NUMINAMATH_CALUDE_additional_passengers_proof_l3824_382466

/-- The number of carriages in a train -/
def carriages_per_train : ℕ := 4

/-- The number of seats in each carriage -/
def seats_per_carriage : ℕ := 25

/-- The total number of passengers that can be accommodated in 3 trains with additional capacity -/
def total_passengers : ℕ := 420

/-- The number of trains -/
def num_trains : ℕ := 3

/-- The additional number of passengers each carriage can accommodate -/
def additional_passengers : ℕ := 10

theorem additional_passengers_proof :
  additional_passengers = 
    (total_passengers - num_trains * carriages_per_train * seats_per_carriage) / 
    (num_trains * carriages_per_train) :=
by sorry

end NUMINAMATH_CALUDE_additional_passengers_proof_l3824_382466


namespace NUMINAMATH_CALUDE_solution_range_l3824_382423

theorem solution_range (k : ℝ) : 
  (∃ x : ℝ, x + 2*k = 4*(x + k) + 1 ∧ x < 0) → k > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l3824_382423


namespace NUMINAMATH_CALUDE_ralph_tv_watching_hours_l3824_382416

/-- The number of hours Ralph watches TV on a weekday -/
def weekday_hours : ℕ := 4

/-- The number of hours Ralph watches TV on a weekend day -/
def weekend_hours : ℕ := 6

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The total number of hours Ralph watches TV in a week -/
def total_hours : ℕ := weekday_hours * weekdays + weekend_hours * weekend_days

theorem ralph_tv_watching_hours :
  total_hours = 32 := by sorry

end NUMINAMATH_CALUDE_ralph_tv_watching_hours_l3824_382416


namespace NUMINAMATH_CALUDE_trig_ratios_for_point_on_terminal_side_l3824_382449

/-- Given a point P(3m, -2m) where m < 0 lying on the terminal side of angle α,
    prove the trigonometric ratios for α. -/
theorem trig_ratios_for_point_on_terminal_side (m : ℝ) (α : ℝ) 
  (h1 : m < 0) 
  (h2 : ∃ (r : ℝ), r > 0 ∧ 3 * m = r * Real.cos α ∧ -2 * m = r * Real.sin α) :
  Real.sin α = 2 * Real.sqrt 13 / 13 ∧ 
  Real.cos α = -3 * Real.sqrt 13 / 13 ∧ 
  Real.tan α = -2 / 3 := by
sorry


end NUMINAMATH_CALUDE_trig_ratios_for_point_on_terminal_side_l3824_382449


namespace NUMINAMATH_CALUDE_cube_properties_l3824_382490

-- Define the surface area of the cube
def surface_area : ℝ := 150

-- Define the relationship between surface area and edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the volume of a cube given its edge length
def volume (s : ℝ) : ℝ := s^3

-- Theorem statement
theorem cube_properties :
  ∃ (s : ℝ), edge_length s ∧ s = 5 ∧ volume s = 125 :=
sorry

end NUMINAMATH_CALUDE_cube_properties_l3824_382490


namespace NUMINAMATH_CALUDE_t_values_l3824_382406

theorem t_values (t : ℝ) : 
  let M : Set ℝ := {1, 3, t}
  let N : Set ℝ := {t^2 - t + 1}
  (M ∪ N = M) → (t = 0 ∨ t = 2 ∨ t = -1) := by
sorry

end NUMINAMATH_CALUDE_t_values_l3824_382406


namespace NUMINAMATH_CALUDE_smallest_7digit_binary_proof_l3824_382402

/-- The smallest positive integer with a 7-digit binary representation -/
def smallest_7digit_binary : ℕ := 64

/-- The binary representation of a natural number -/
def binary_representation (n : ℕ) : List Bool :=
  sorry

/-- The length of the binary representation of a natural number -/
def binary_length (n : ℕ) : ℕ :=
  (binary_representation n).length

theorem smallest_7digit_binary_proof :
  (∀ m : ℕ, m < smallest_7digit_binary → binary_length m < 7) ∧
  binary_length smallest_7digit_binary = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_7digit_binary_proof_l3824_382402


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3824_382453

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ,
  x^5 + 3*x^3 + 1 = (x - 3)^2 * q + (324*x - 488) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3824_382453


namespace NUMINAMATH_CALUDE_largest_number_less_than_two_l3824_382411

theorem largest_number_less_than_two : 
  let numbers : Finset ℝ := {0.8, 1/2, 0.5}
  ∀ x ∈ numbers, x < 2 → 
  ∃ max ∈ numbers, ∀ y ∈ numbers, y ≤ max ∧ max = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_less_than_two_l3824_382411


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3824_382484

theorem greatest_divisor_with_remainders : 
  let a := 6215 - 23
  let b := 7373 - 29
  let c := 8927 - 35
  Nat.gcd a (Nat.gcd b c) = 36 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3824_382484


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3824_382414

theorem isosceles_triangle_perimeter (m : ℝ) : 
  (2 : ℝ) ^ 2 - (5 + m) * 2 + 5 * m = 0 →
  ∃ (a b : ℝ), a ^ 2 - (5 + m) * a + 5 * m = 0 ∧
                b ^ 2 - (5 + m) * b + 5 * m = 0 ∧
                a ≠ b ∧
                (a = 2 ∨ b = 2) ∧
                (a + a + b = 12 ∨ a + b + b = 12) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3824_382414


namespace NUMINAMATH_CALUDE_point_on_y_axis_l3824_382425

/-- Given that point P(2-a, a-3) lies on the y-axis, prove that a = 2 -/
theorem point_on_y_axis (a : ℝ) : (2 - a = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3824_382425


namespace NUMINAMATH_CALUDE_linear_function_k_range_l3824_382485

theorem linear_function_k_range (k b : ℝ) :
  k ≠ 0 →
  (2 * k + b = -3) →
  (0 < b ∧ b < 1) →
  (-2 < k ∧ k < -3/2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_k_range_l3824_382485


namespace NUMINAMATH_CALUDE_greater_number_proof_l3824_382461

theorem greater_number_proof (x y : ℝ) (h_sum : x + y = 40) (h_diff : x - y = 12) (h_greater : x > y) : x = 26 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l3824_382461


namespace NUMINAMATH_CALUDE_percentage_not_working_l3824_382445

/-- Represents the employment status of a group --/
structure EmploymentStatus where
  fullTime : Rat
  partTime : Rat

/-- Represents the survey data --/
structure SurveyData where
  mothers : EmploymentStatus
  fathers : EmploymentStatus
  grandparents : EmploymentStatus
  womenPercentage : Rat
  menPercentage : Rat
  grandparentsPercentage : Rat

/-- Calculates the percentage of individuals not working in a given group --/
def notWorkingPercentage (status : EmploymentStatus) : Rat :=
  1 - status.fullTime - status.partTime

/-- Theorem stating the percentage of surveyed individuals not holding a job --/
theorem percentage_not_working (data : SurveyData) :
  data.mothers = { fullTime := 5/6, partTime := 1/6 } →
  data.fathers = { fullTime := 3/4, partTime := 1/8 } →
  data.grandparents = { fullTime := 1/2, partTime := 1/4 } →
  data.womenPercentage = 55/100 →
  data.menPercentage = 35/100 →
  data.grandparentsPercentage = 1/10 →
  (notWorkingPercentage data.mothers) * data.womenPercentage +
  (notWorkingPercentage data.fathers) * data.menPercentage +
  (notWorkingPercentage data.grandparents) * data.grandparentsPercentage =
  6875/100000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_working_l3824_382445


namespace NUMINAMATH_CALUDE_right_triangle_area_l3824_382448

/-- The area of a right triangle with hypotenuse 13 and shortest side 5 is 30 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) (h4 : a ≤ b) : (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3824_382448


namespace NUMINAMATH_CALUDE_divisibility_condition_iff_n_le_3_l3824_382483

/-- A complete graph with n vertices -/
structure CompleteGraph (n : ℕ) where
  vertices : Fin n
  edges : Fin (n.choose 2)

/-- A labeling of edges with consecutive natural numbers -/
def EdgeLabeling (n : ℕ) := Fin (n.choose 2) → ℕ

/-- Condition for divisibility in a path of length 3 -/
def DivisibilityCondition (g : CompleteGraph n) (l : EdgeLabeling n) : Prop :=
  ∀ (a b c : Fin (n.choose 2)),
    (l b) ∣ (Nat.gcd (l a) (l c))

/-- Main theorem: The divisibility condition can be satisfied if and only if n ≤ 3 -/
theorem divisibility_condition_iff_n_le_3 (n : ℕ) :
  (∃ (g : CompleteGraph n) (l : EdgeLabeling n),
    DivisibilityCondition g l ∧
    (∀ i : Fin (n.choose 2), l i = i.val + 1)) ↔
  n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_iff_n_le_3_l3824_382483


namespace NUMINAMATH_CALUDE_exists_number_satisfying_equation_l3824_382418

theorem exists_number_satisfying_equation : ∃ N : ℝ, (0.47 * N - 0.36 * 1412) + 66 = 6 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_satisfying_equation_l3824_382418


namespace NUMINAMATH_CALUDE_cyclic_win_sets_count_l3824_382441

/-- A round-robin tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (wins_per_team : ℕ)
  (losses_per_team : ℕ)
  (h_round_robin : wins_per_team + losses_per_team = num_teams - 1)
  (h_no_ties : True)

/-- The number of sets of three teams with cyclic wins -/
def cyclic_win_sets (t : Tournament) : ℕ := sorry

/-- The theorem to be proved -/
theorem cyclic_win_sets_count 
  (t : Tournament) 
  (h_num_teams : t.num_teams = 20) 
  (h_wins : t.wins_per_team = 12) 
  (h_losses : t.losses_per_team = 7) : 
  cyclic_win_sets t = 570 := by sorry

end NUMINAMATH_CALUDE_cyclic_win_sets_count_l3824_382441


namespace NUMINAMATH_CALUDE_sqrt_operations_l3824_382481

theorem sqrt_operations :
  (∀ x y : ℝ, x > 0 → y > 0 → (Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y)) ∧
  (Real.sqrt 12 / Real.sqrt 3 = 2) ∧
  (Real.sqrt 8 = 2 * Real.sqrt 2) ∧
  (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) ∧
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_operations_l3824_382481


namespace NUMINAMATH_CALUDE_brians_purchased_animals_ratio_l3824_382437

theorem brians_purchased_animals_ratio (initial_horses : ℕ) (initial_sheep : ℕ) (initial_chickens : ℕ) (gifted_goats : ℕ) (male_animals : ℕ) : 
  initial_horses = 100 →
  initial_sheep = 29 →
  initial_chickens = 9 →
  gifted_goats = 37 →
  male_animals = 53 →
  (initial_horses + initial_sheep + initial_chickens - (2 * male_animals - gifted_goats)) * 2 = initial_horses + initial_sheep + initial_chickens :=
by sorry

end NUMINAMATH_CALUDE_brians_purchased_animals_ratio_l3824_382437


namespace NUMINAMATH_CALUDE_flowerbed_count_l3824_382498

theorem flowerbed_count (total_seeds : ℕ) (seeds_per_flowerbed : ℕ) (h1 : total_seeds = 45) (h2 : seeds_per_flowerbed = 5) :
  total_seeds / seeds_per_flowerbed = 9 := by
  sorry

end NUMINAMATH_CALUDE_flowerbed_count_l3824_382498


namespace NUMINAMATH_CALUDE_reyansh_farm_water_ratio_l3824_382428

/-- Represents the farm owned by Mr. Reyansh -/
structure Farm where
  num_cows : ℕ
  cow_water_daily : ℕ
  sheep_cow_ratio : ℕ
  total_water_weekly : ℕ

/-- Calculates the ratio of daily water consumption of a sheep to a cow -/
def water_consumption_ratio (f : Farm) : Rat :=
  let cow_water_weekly := f.num_cows * f.cow_water_daily * 7
  let sheep_water_weekly := f.total_water_weekly - cow_water_weekly
  let num_sheep := f.num_cows * f.sheep_cow_ratio
  let sheep_water_daily := sheep_water_weekly / (7 * num_sheep)
  sheep_water_daily / f.cow_water_daily

/-- Theorem stating that the water consumption ratio for Mr. Reyansh's farm is 1:4 -/
theorem reyansh_farm_water_ratio :
  let f : Farm := {
    num_cows := 40,
    cow_water_daily := 80,
    sheep_cow_ratio := 10,
    total_water_weekly := 78400
  }
  water_consumption_ratio f = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_reyansh_farm_water_ratio_l3824_382428


namespace NUMINAMATH_CALUDE_function_satisfying_condition_l3824_382450

/-- A function that satisfies f(a f(b)) = a b for all a and b -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a * f b) = a * b

/-- The theorem stating that a function satisfying the condition must be either the identity function or its negation -/
theorem function_satisfying_condition (f : ℝ → ℝ) (h : SatisfiesCondition f) :
    (∀ x, f x = x) ∨ (∀ x, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_condition_l3824_382450


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3824_382422

theorem complementary_angles_difference (x y : ℝ) : 
  x + y = 90 → x / y = 5 → |x - y| = 60 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3824_382422


namespace NUMINAMATH_CALUDE_dimitri_burger_consumption_l3824_382459

/-- Dimitri's burger consumption problem -/
theorem dimitri_burger_consumption (burgers_per_day : ℕ) 
  (calories_per_burger : ℕ) (total_calories : ℕ) (days : ℕ) :
  burgers_per_day * calories_per_burger * days = total_calories →
  calories_per_burger = 20 →
  total_calories = 120 →
  days = 2 →
  burgers_per_day = 3 := by
sorry

end NUMINAMATH_CALUDE_dimitri_burger_consumption_l3824_382459


namespace NUMINAMATH_CALUDE_equations_solution_l3824_382470

def satisfies_equations (a b c d : ℕ) : Prop :=
  a + b = c * d ∧ c + d = a * b

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(2,2,2,2), (1,2,3,5), (2,1,3,5), (1,2,5,3), (2,1,5,3), (3,5,1,2), (5,3,1,2), (3,5,2,1), (5,3,2,1)}

theorem equations_solution :
  ∀ (a b c d : ℕ), satisfies_equations a b c d ↔ (a, b, c, d) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equations_solution_l3824_382470


namespace NUMINAMATH_CALUDE_triangle_side_length_l3824_382409

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2 / 3 →
  b^2 + c^2 - a^2 = 2 * b * c * Real.cos A →
  b = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3824_382409


namespace NUMINAMATH_CALUDE_exam_items_count_l3824_382404

theorem exam_items_count :
  ∀ (X : ℕ) (E M : ℕ),
    M = 24 →
    M = E / 2 + 6 →
    X = E + 4 →
    X = 40 := by
  sorry

end NUMINAMATH_CALUDE_exam_items_count_l3824_382404


namespace NUMINAMATH_CALUDE_expression_evaluation_l3824_382464

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := 2
  4 * (2 * a^2 * b - a * b^2) - (3 * a * b^2 + 2 * a^2 * b) = -11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3824_382464


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequalities_l3824_382424

theorem smallest_integer_satisfying_inequalities :
  ∀ x : ℤ, x < 3 * x - 12 ∧ x > 0 → x ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequalities_l3824_382424


namespace NUMINAMATH_CALUDE_inequality_range_m_l3824_382408

/-- The function f(x) defined in the problem -/
def f (x : ℝ) : ℝ := x^2 + |x - 1|

/-- The theorem stating the range of m for which the inequality always holds -/
theorem inequality_range_m :
  (∀ x : ℝ, f x ≥ (m + 2) * x - 1) ↔ m ∈ Set.Icc (-3 - 2 * Real.sqrt 2) 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_m_l3824_382408


namespace NUMINAMATH_CALUDE_pet_food_price_l3824_382477

theorem pet_food_price (regular_discount_min : ℝ) (regular_discount_max : ℝ) 
  (additional_discount : ℝ) (lowest_price : ℝ) :
  regular_discount_min = 0.1 →
  regular_discount_max = 0.3 →
  additional_discount = 0.2 →
  lowest_price = 16.8 →
  ∃ (original_price : ℝ), 
    original_price * (1 - regular_discount_max) * (1 - additional_discount) = lowest_price ∧
    original_price = 30 :=
by sorry

end NUMINAMATH_CALUDE_pet_food_price_l3824_382477


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3824_382475

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3824_382475


namespace NUMINAMATH_CALUDE_vector_operations_l3824_382415

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, -4)

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

def is_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem vector_operations (c : ℝ × ℝ) 
  (h1 : is_unit_vector c) 
  (h2 : is_perpendicular c (a.1 - b.1, a.2 - b.2)) : 
  (2 * a.1 + 3 * b.1, 2 * a.2 + 3 * b.2) = (-5, -10) ∧
  (a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2 = 145 ∧
  (c = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) ∨ c = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l3824_382415


namespace NUMINAMATH_CALUDE_total_tax_percentage_l3824_382471

/-- Calculates the total tax percentage given spending percentages and tax rates -/
theorem total_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.2)
  (h3 : other_percent = 0.3)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.05)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.1) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.055 := by
sorry


end NUMINAMATH_CALUDE_total_tax_percentage_l3824_382471


namespace NUMINAMATH_CALUDE_sum_of_divisors_143_l3824_382446

/-- The sum of all positive integer divisors of 143 is 168 -/
theorem sum_of_divisors_143 : (Finset.filter (· ∣ 143) (Finset.range 144)).sum id = 168 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_143_l3824_382446


namespace NUMINAMATH_CALUDE_coltons_remaining_stickers_l3824_382457

/-- The number of stickers Colton has left after giving some away to friends. -/
def stickers_left (initial_stickers : ℕ) (stickers_per_friend : ℕ) (num_friends : ℕ) 
  (extra_to_mandy : ℕ) (less_to_justin : ℕ) : ℕ :=
  let stickers_to_friends := stickers_per_friend * num_friends
  let stickers_to_mandy := stickers_to_friends + extra_to_mandy
  let stickers_to_justin := stickers_to_mandy - less_to_justin
  let total_given_away := stickers_to_friends + stickers_to_mandy + stickers_to_justin
  initial_stickers - total_given_away

/-- Theorem stating that Colton has 42 stickers left given the problem conditions. -/
theorem coltons_remaining_stickers : 
  stickers_left 72 4 3 2 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_coltons_remaining_stickers_l3824_382457


namespace NUMINAMATH_CALUDE_expression_evaluation_l3824_382403

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x = 789 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3824_382403


namespace NUMINAMATH_CALUDE_zain_total_coins_l3824_382473

/-- Represents the number of coins of each type --/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ
  halfDollars : ℕ

/-- Calculates the total number of coins --/
def totalCoins (coins : CoinCount) : ℕ :=
  coins.quarters + coins.dimes + coins.nickels + coins.pennies + coins.halfDollars

/-- Represents Emerie's coin count --/
def emerieCoins : CoinCount :=
  { quarters := 6
  , dimes := 7
  , nickels := 5
  , pennies := 10
  , halfDollars := 2 }

/-- Calculates Zain's coin count based on Emerie's --/
def zainCoins (emerie : CoinCount) : CoinCount :=
  { quarters := emerie.quarters + 10
  , dimes := emerie.dimes + 10
  , nickels := emerie.nickels + 10
  , pennies := emerie.pennies + 10
  , halfDollars := emerie.halfDollars + 10 }

/-- Theorem: Zain has 80 coins in total --/
theorem zain_total_coins : totalCoins (zainCoins emerieCoins) = 80 := by
  sorry

end NUMINAMATH_CALUDE_zain_total_coins_l3824_382473


namespace NUMINAMATH_CALUDE_tu_yuan_yuan_theorem_l3824_382420

/-- Represents the purchase and sale of "Tu Yuan Yuan" toys -/
structure ToyPurchase where
  first_cost : ℕ
  second_cost : ℕ
  price_increase : ℕ
  min_profit : ℕ

/-- Calculates the quantity of the first purchase -/
def first_quantity (tp : ToyPurchase) : ℕ :=
  sorry

/-- Calculates the minimum selling price -/
def min_selling_price (tp : ToyPurchase) : ℕ :=
  sorry

/-- Theorem stating the correct quantity and minimum selling price -/
theorem tu_yuan_yuan_theorem (tp : ToyPurchase) 
  (h1 : tp.first_cost = 1500)
  (h2 : tp.second_cost = 3500)
  (h3 : tp.price_increase = 5)
  (h4 : tp.min_profit = 1150) :
  first_quantity tp = 50 ∧ min_selling_price tp = 41 := by
  sorry

end NUMINAMATH_CALUDE_tu_yuan_yuan_theorem_l3824_382420


namespace NUMINAMATH_CALUDE_equation_four_solutions_l3824_382444

theorem equation_four_solutions :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x : ℝ, x ∈ s ↔ (x - 2) * (x + 1) * (x + 4) * (x + 7) = 19) ∧
  (s = {(-5 + Real.sqrt 85) / 2, (-5 - Real.sqrt 85) / 2, 
        (-5 + Real.sqrt 5) / 2, (-5 - Real.sqrt 5) / 2}) :=
by sorry

end NUMINAMATH_CALUDE_equation_four_solutions_l3824_382444


namespace NUMINAMATH_CALUDE_pencil_case_solution_l3824_382439

/-- Represents the cost and quantity of pencil cases --/
structure PencilCases where
  cost_A : ℚ
  cost_B : ℚ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Conditions for the pencil case problem --/
def PencilCaseProblem (p : PencilCases) : Prop :=
  p.cost_B = p.cost_A + 2 ∧
  800 / p.cost_A = 1000 / p.cost_B ∧
  p.quantity_A = 3 * p.quantity_B - 50 ∧
  p.quantity_A + p.quantity_B ≤ 910 ∧
  12 * p.quantity_A + 15 * p.quantity_B - 
  (p.cost_A * p.quantity_A + p.cost_B * p.quantity_B) > 3795

/-- The main theorem to prove --/
theorem pencil_case_solution (p : PencilCases) 
  (h : PencilCaseProblem p) : 
  p.cost_A = 8 ∧ 
  p.cost_B = 10 ∧ 
  p.quantity_B ≤ 240 ∧ 
  (∃ n : ℕ, n = 5 ∧ 
    ∀ m : ℕ, 236 ≤ m ∧ m ≤ 240 → 
      (12 * (3 * m - 50) + 15 * m - (8 * (3 * m - 50) + 10 * m) > 3795)) := by
  sorry


end NUMINAMATH_CALUDE_pencil_case_solution_l3824_382439


namespace NUMINAMATH_CALUDE_arrangements_with_separation_l3824_382467

/-- The number of ways to arrange n people in a row. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people are adjacent. -/
def adjacentArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of people in the problem. -/
def numberOfPeople : ℕ := 5

/-- Theorem stating that the number of arrangements with at least one person between A and B is 72. -/
theorem arrangements_with_separation :
  totalArrangements numberOfPeople - adjacentArrangements numberOfPeople = 72 := by
  sorry

#eval totalArrangements numberOfPeople - adjacentArrangements numberOfPeople

end NUMINAMATH_CALUDE_arrangements_with_separation_l3824_382467


namespace NUMINAMATH_CALUDE_rahul_savings_l3824_382496

/-- Proves that given the conditions on Rahul's savings, the total amount saved is 180,000 Rs. -/
theorem rahul_savings (nsc ppf : ℕ) : 
  (1 / 3 : ℚ) * nsc = (1 / 2 : ℚ) * ppf →
  ppf = 72000 →
  nsc + ppf = 180000 := by
sorry

end NUMINAMATH_CALUDE_rahul_savings_l3824_382496


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3824_382476

theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (total_length : ℝ) 
  (h1 : train_length = 130)
  (h2 : train_speed_kmh = 45)
  (h3 : total_length = 245) :
  let bridge_length := total_length - train_length
  let total_distance := total_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let crossing_time := total_distance / train_speed_ms
  crossing_time = 19.6 := by sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3824_382476


namespace NUMINAMATH_CALUDE_largest_factor_of_9975_l3824_382451

theorem largest_factor_of_9975 : 
  ∀ n : ℕ, n ∣ 9975 ∧ n < 10000 → n ≤ 4975 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_factor_of_9975_l3824_382451


namespace NUMINAMATH_CALUDE_james_lifting_weight_l3824_382494

/-- Calculates the weight James can lift with straps for 10 meters given initial conditions -/
def weight_with_straps (initial_weight : ℝ) (distance_increase : ℝ) (short_distance_factor : ℝ) (strap_factor : ℝ) : ℝ :=
  let base_weight := initial_weight + distance_increase
  let short_distance_weight := base_weight * (1 + short_distance_factor)
  short_distance_weight * (1 + strap_factor)

/-- Theorem stating the final weight James can lift with straps for 10 meters -/
theorem james_lifting_weight :
  weight_with_straps 300 50 0.3 0.2 = 546 := by
  sorry

#eval weight_with_straps 300 50 0.3 0.2

end NUMINAMATH_CALUDE_james_lifting_weight_l3824_382494


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l3824_382435

/-- Given a cone with base radius √2 and lateral surface that unfolds into a semicircle,
    the length of the generatrix is 2√2. -/
theorem cone_generatrix_length (r : ℝ) (l : ℝ) : 
  r = Real.sqrt 2 →  -- Base radius is √2
  2 * Real.pi * r = Real.pi * l →  -- Lateral surface unfolds into a semicircle
  l = 2 * Real.sqrt 2 :=  -- Length of generatrix is 2√2
by sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l3824_382435


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3824_382432

theorem arithmetic_mean_problem (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = c →
  b * c = 400 →
  d = 28 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3824_382432


namespace NUMINAMATH_CALUDE_problem_1_l3824_382465

theorem problem_1 : (-2)^0 + 1 / Real.sqrt 2 - Real.sqrt 9 = Real.sqrt 2 / 2 - 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_l3824_382465


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3824_382456

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

def is_perfect_cube (x : ℕ) : Prop := ∃ y : ℕ, x = y^3

theorem smallest_n_square_and_cube : 
  (∀ n : ℕ, n > 0 ∧ is_perfect_square (5*n) ∧ is_perfect_cube (4*n) → n ≥ 80) ∧
  (is_perfect_square (5*80) ∧ is_perfect_cube (4*80)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3824_382456


namespace NUMINAMATH_CALUDE_additional_toothpicks_for_8_steps_l3824_382429

/-- The number of toothpicks needed for a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 4
  else toothpicks (n - 1) + 2 + 4 * (n - 1)

theorem additional_toothpicks_for_8_steps :
  toothpicks 4 = 30 →
  toothpicks 8 - toothpicks 4 = 88 :=
by sorry

end NUMINAMATH_CALUDE_additional_toothpicks_for_8_steps_l3824_382429


namespace NUMINAMATH_CALUDE_locus_of_centers_l3824_382493

/-- The locus of centers of circles externally tangent to C1 and internally tangent to C2 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, 
    (a^2 + b^2 = (r + 1)^2) ∧ 
    ((a - 3)^2 + b^2 = (5 - r)^2)) ↔ 
  (4*a^2 + 4*b^2 - 6*a - 25 = 0) := by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l3824_382493


namespace NUMINAMATH_CALUDE_bookshop_inventory_bookshop_current_inventory_l3824_382400

/-- Calculates the current number of books in a bookshop after a weekend of sales and a new shipment. -/
theorem bookshop_inventory (
  initial_inventory : ℕ
  ) (saturday_in_store : ℕ) (saturday_online : ℕ)
  (sunday_in_store_multiplier : ℕ) (sunday_online_increase : ℕ)
  (new_shipment : ℕ) : ℕ :=
  let sunday_in_store := sunday_in_store_multiplier * saturday_in_store
  let sunday_online := saturday_online + sunday_online_increase
  let total_sold := saturday_in_store + saturday_online + sunday_in_store + sunday_online
  let net_change := new_shipment - total_sold
  initial_inventory + net_change

/-- The bookshop currently has 502 books. -/
theorem bookshop_current_inventory :
  bookshop_inventory 743 37 128 2 34 160 = 502 := by
  sorry

end NUMINAMATH_CALUDE_bookshop_inventory_bookshop_current_inventory_l3824_382400


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l3824_382488

theorem opposite_of_fraction (n : ℕ) (h : n ≠ 0) : 
  -(1 : ℚ) / n = -(1 / n) := by sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l3824_382488


namespace NUMINAMATH_CALUDE_basis_iff_not_parallel_l3824_382412

def is_basis (e₁ e₂ : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (v : ℝ × ℝ), v = (a * e₁.1 + b * e₂.1, a * e₁.2 + b * e₂.2)

def are_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 = v₁.2 * v₂.1

theorem basis_iff_not_parallel (e₁ e₂ : ℝ × ℝ) :
  is_basis e₁ e₂ ↔ ¬ are_parallel e₁ e₂ :=
sorry

end NUMINAMATH_CALUDE_basis_iff_not_parallel_l3824_382412


namespace NUMINAMATH_CALUDE_hyperbola_equation_proof_l3824_382474

/-- The focal length of the hyperbola -/
def focal_length : ℝ := 4

/-- The equation of circle C₂ -/
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

/-- The equation of hyperbola C₁ -/
def hyperbola_equation (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptote_equation (a b x y : ℝ) : Prop := b * x = a * y ∨ b * x = -a * y

/-- The asymptotes are tangent to the circle -/
def asymptotes_tangent_to_circle (a b : ℝ) : Prop :=
  ∀ x y, asymptote_equation a b x y → (abs (-2 * b) / Real.sqrt (a^2 + b^2) = 1)

theorem hyperbola_equation_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_focal : a^2 - b^2 = focal_length^2 / 4)
  (h_tangent : asymptotes_tangent_to_circle a b) :
  ∀ x y, hyperbola_equation a b x y ↔ x^2 / 3 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_proof_l3824_382474


namespace NUMINAMATH_CALUDE_divide_angle_19_degrees_l3824_382401

theorem divide_angle_19_degrees (angle : ℝ) (n : ℕ) : 
  angle = 19 ∧ n = 19 → (angle / n : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_divide_angle_19_degrees_l3824_382401


namespace NUMINAMATH_CALUDE_log_expression_equals_four_l3824_382454

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_four :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_four_l3824_382454


namespace NUMINAMATH_CALUDE_negation_equivalence_l3824_382499

theorem negation_equivalence (P Q : Prop) :
  ¬(P → ¬Q) ↔ (P ∧ Q) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3824_382499


namespace NUMINAMATH_CALUDE_inequality_of_powers_l3824_382482

theorem inequality_of_powers (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^(2*a) * b^(2*b) * c^(2*c) ≥ a^(b+c) * b^(a+c) * c^(a+b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l3824_382482


namespace NUMINAMATH_CALUDE_parabolic_trajectory_falls_within_interval_l3824_382438

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabolic trajectory -/
structure Trajectory where
  a : ℝ
  c : ℝ

/-- Check if a trajectory passes through a point -/
def passesThrough (t : Trajectory) (p : Point) : Prop :=
  p.y = t.a * p.x^2 + t.c

/-- Check if a trajectory intersects with a given x-coordinate at or below a certain y-coordinate -/
def intersectsAt (t : Trajectory) (x y : ℝ) : Prop :=
  t.a * x^2 + t.c ≤ y

theorem parabolic_trajectory_falls_within_interval 
  (t : Trajectory) 
  (A : Point) 
  (P : Point) 
  (D : Point) :
  t.a < 0 →
  A.x = 0 ∧ A.y = 9 →
  P.x = 2 ∧ P.y = 8.1 →
  D.x = 6 ∧ D.y = 7 →
  passesThrough t A →
  passesThrough t P →
  intersectsAt t D.x D.y :=
by sorry

end NUMINAMATH_CALUDE_parabolic_trajectory_falls_within_interval_l3824_382438


namespace NUMINAMATH_CALUDE_min_fencing_cost_problem_l3824_382492

/-- Represents the cost of fencing materials in rupees per meter -/
structure FencingMaterial where
  cost : ℚ

/-- Represents a rectangular field -/
structure RectangularField where
  length : ℚ
  width : ℚ
  area : ℚ

/-- Calculates the minimum fencing cost for a rectangular field -/
def minFencingCost (field : RectangularField) (materials : List FencingMaterial) : ℚ :=
  sorry

/-- Theorem stating the minimum fencing cost for the given problem -/
theorem min_fencing_cost_problem :
  let field : RectangularField := {
    length := 108,
    width := 81,
    area := 8748
  }
  let materials : List FencingMaterial := [
    { cost := 0.25 },
    { cost := 0.35 },
    { cost := 0.40 }
  ]
  minFencingCost field materials = 87.75 := by sorry

end NUMINAMATH_CALUDE_min_fencing_cost_problem_l3824_382492


namespace NUMINAMATH_CALUDE_total_tomatoes_l3824_382460

def tomato_problem (plant1 plant2 plant3 : ℕ) : Prop :=
  plant1 = 24 ∧
  plant2 = (plant1 / 2) + 5 ∧
  plant3 = plant2 + 2 ∧
  plant1 + plant2 + plant3 = 60

theorem total_tomatoes :
  ∃ plant1 plant2 plant3 : ℕ, tomato_problem plant1 plant2 plant3 :=
sorry

end NUMINAMATH_CALUDE_total_tomatoes_l3824_382460


namespace NUMINAMATH_CALUDE_units_digit_of_L_L15_l3824_382417

/-- Lucas numbers sequence -/
def Lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => Lucas (n + 1) + Lucas n

/-- The period of the units digit in the Lucas sequence -/
def LucasPeriod : ℕ := 12

theorem units_digit_of_L_L15 : 
  (Lucas (Lucas 15)) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_L_L15_l3824_382417


namespace NUMINAMATH_CALUDE_g_2016_equals_1_l3824_382433

-- Define the properties of function f
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧
  (∀ x : ℝ, f (x + 5) ≥ f x + 5) ∧
  (∀ x : ℝ, f (x + 1) ≤ f x + 1)

-- Define function g
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1 - x

-- Theorem statement
theorem g_2016_equals_1 (f : ℝ → ℝ) (h : satisfies_conditions f) :
  g f 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_2016_equals_1_l3824_382433


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3824_382491

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, (6 * x^2 + 7 = 5 * x - 11) ↔ (x = a + b * I ∨ x = a - b * I)) →
  a + b^2 = 467/144 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3824_382491


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l3824_382436

theorem gcd_lcm_product_90_150 : Nat.gcd 90 150 * Nat.lcm 90 150 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l3824_382436


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l3824_382479

theorem convex_polygon_sides (n : ℕ) (sum_except_one : ℝ) : 
  sum_except_one = 2190 → 
  (∃ (missing_angle : ℝ), 
    missing_angle > 0 ∧ 
    missing_angle < 180 ∧ 
    sum_except_one + missing_angle = 180 * (n - 2)) → 
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l3824_382479


namespace NUMINAMATH_CALUDE_vet_spay_ratio_l3824_382486

theorem vet_spay_ratio (total_animals : ℕ) (cats : ℕ) (dogs : ℕ) :
  total_animals = 21 →
  cats = 7 →
  dogs = total_animals - cats →
  (dogs : ℚ) / (cats : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_vet_spay_ratio_l3824_382486


namespace NUMINAMATH_CALUDE_parallel_tangents_and_zero_points_l3824_382478

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x - a) / (x^2)

theorem parallel_tangents_and_zero_points (a : ℝ) (h : a > 0) :
  -- Part 1: Parallel tangents imply a = 3.5
  (f_deriv a 3 = f_deriv a (3/2) → a = 3.5) ∧
  -- Part 2: Zero points imply 0 < a ≤ 1
  (∃ x, f a x = 0 → 0 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_parallel_tangents_and_zero_points_l3824_382478


namespace NUMINAMATH_CALUDE_square_of_sum_fifteen_three_l3824_382463

theorem square_of_sum_fifteen_three : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_fifteen_three_l3824_382463


namespace NUMINAMATH_CALUDE_sum_edge_face_angles_less_than_plane_angles_sum_edge_face_angles_greater_than_half_plane_angles_if_acute_l3824_382434

-- Define a trihedral angle
structure TrihedralAngle where
  -- Angles between edges and opposite faces
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Plane angles at vertex
  θ₁ : ℝ
  θ₂ : ℝ
  θ₃ : ℝ
  -- Ensure all angles are positive
  α_pos : 0 < α
  β_pos : 0 < β
  γ_pos : 0 < γ
  θ₁_pos : 0 < θ₁
  θ₂_pos : 0 < θ₂
  θ₃_pos : 0 < θ₃

-- Theorem 1: Sum of angles between edges and opposite faces is less than sum of plane angles
theorem sum_edge_face_angles_less_than_plane_angles (t : TrihedralAngle) :
  t.α + t.β + t.γ < t.θ₁ + t.θ₂ + t.θ₃ := by
  sorry

-- Theorem 2: If all plane angles are acute, sum of angles between edges and opposite faces 
-- is greater than half the sum of plane angles
theorem sum_edge_face_angles_greater_than_half_plane_angles_if_acute (t : TrihedralAngle)
  (h₁ : t.θ₁ < π/2) (h₂ : t.θ₂ < π/2) (h₃ : t.θ₃ < π/2) :
  t.α + t.β + t.γ > (t.θ₁ + t.θ₂ + t.θ₃) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_edge_face_angles_less_than_plane_angles_sum_edge_face_angles_greater_than_half_plane_angles_if_acute_l3824_382434


namespace NUMINAMATH_CALUDE_polygon_area_l3824_382489

structure Polygon :=
  (sides : ℕ)
  (side_length : ℝ)
  (perimeter : ℝ)
  (is_rectangular_with_removed_corners : Prop)

def area_of_polygon (p : Polygon) : ℝ :=
  20 * p.side_length^2

theorem polygon_area (p : Polygon) 
  (h1 : p.sides = 20)
  (h2 : p.perimeter = 60)
  (h3 : p.is_rectangular_with_removed_corners)
  (h4 : p.side_length = p.perimeter / p.sides) :
  area_of_polygon p = 180 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_l3824_382489


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3824_382443

-- Define the sets P and Q
def P : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the result set
def result : Set ℝ := Set.Icc 0 2

-- State the theorem
theorem intersection_complement_equality : P ∩ (Set.univ \ Q) = result := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3824_382443


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3824_382421

theorem inequality_solution_set : 
  {x : ℝ | x / (x - 1) + (x + 2) / (2 * x) ≥ 3} = 
  {x : ℝ | (0 < x ∧ x ≤ 1/3) ∨ (1 < x ∧ x ≤ 2)} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3824_382421


namespace NUMINAMATH_CALUDE_algebra_test_female_count_l3824_382427

theorem algebra_test_female_count :
  ∀ (total_average : ℝ) (male_count : ℕ) (male_average female_average : ℝ),
    total_average = 90 →
    male_count = 8 →
    male_average = 85 →
    female_average = 92 →
    ∃ (female_count : ℕ),
      (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
      female_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_algebra_test_female_count_l3824_382427


namespace NUMINAMATH_CALUDE_sqrt_fraction_eval_l3824_382431

theorem sqrt_fraction_eval (x : ℝ) (h : x < -1) :
  Real.sqrt (x / (1 - (2 * x - 3) / (x + 1))) = Complex.I * Real.sqrt (x^2 - 3*x - 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_eval_l3824_382431


namespace NUMINAMATH_CALUDE_floating_state_exists_l3824_382458

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  volume : ℝ
  surfaceArea : ℝ

/-- Represents the state of a floating polyhedron -/
structure FloatingState (p : ConvexPolyhedron) where
  submergedVolume : ℝ
  submergedSurfaceArea : ℝ
  volumeRatio : submergedVolume = 0.9 * p.volume
  surfaceAreaRatio : submergedSurfaceArea < 0.5 * p.surfaceArea

/-- Theorem stating that the described floating state is possible -/
theorem floating_state_exists : ∃ (p : ConvexPolyhedron), ∃ (s : FloatingState p), True := by
  sorry

end NUMINAMATH_CALUDE_floating_state_exists_l3824_382458


namespace NUMINAMATH_CALUDE_equation_solution_l3824_382468

theorem equation_solution :
  ∃ x : ℝ, x ≠ -2 ∧ (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 5 → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3824_382468


namespace NUMINAMATH_CALUDE_perpendicular_conditions_l3824_382419

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (line_perp_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (plane_perp_plane : Plane → Plane → Prop)
variable (plane_parallel_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_conditions 
  (a b : Line) (α β : Plane) :
  (line_perp_plane a α ∧ line_perp_plane b β ∧ plane_perp_plane α β → perpendicular a b) ∧
  (line_in_plane a α ∧ line_perp_plane b β ∧ plane_parallel_plane α β → perpendicular a b) ∧
  (line_perp_plane a α ∧ line_parallel_plane b β ∧ plane_parallel_plane α β → perpendicular a b) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_conditions_l3824_382419


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3824_382480

theorem right_triangle_sides : ∃ (a b c : ℝ), 
  a = 8 ∧ b = 15 ∧ c = 17 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3824_382480


namespace NUMINAMATH_CALUDE_bonsai_cost_proof_l3824_382440

/-- The cost of a small bonsai -/
def small_bonsai_cost : ℝ := 30

/-- The cost of a big bonsai -/
def big_bonsai_cost : ℝ := 20

/-- The number of small bonsai sold -/
def small_bonsai_sold : ℕ := 3

/-- The number of big bonsai sold -/
def big_bonsai_sold : ℕ := 5

/-- The total earnings -/
def total_earnings : ℝ := 190

theorem bonsai_cost_proof :
  small_bonsai_cost * small_bonsai_sold + big_bonsai_cost * big_bonsai_sold = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_bonsai_cost_proof_l3824_382440


namespace NUMINAMATH_CALUDE_two_approve_probability_l3824_382495

/-- The probability of a voter approving the mayor's work -/
def approval_rate : ℝ := 0.6

/-- The number of voters randomly selected -/
def sample_size : ℕ := 4

/-- The number of approving voters we're interested in -/
def target_approvals : ℕ := 2

/-- The probability of exactly two out of four randomly selected voters approving the mayor's work -/
def prob_two_approve : ℝ := Nat.choose sample_size target_approvals * approval_rate ^ target_approvals * (1 - approval_rate) ^ (sample_size - target_approvals)

theorem two_approve_probability :
  prob_two_approve = 0.864 := by sorry

end NUMINAMATH_CALUDE_two_approve_probability_l3824_382495
