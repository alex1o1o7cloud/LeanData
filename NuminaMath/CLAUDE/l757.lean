import Mathlib

namespace NUMINAMATH_CALUDE_doughnut_machine_completion_time_l757_75771

-- Define the start time and quarter completion time
def start_time : Nat := 7 * 60  -- 7:00 AM in minutes
def quarter_completion_time : Nat := 10 * 60  -- 10:00 AM in minutes

-- Define the maintenance break duration
def maintenance_break : Nat := 30  -- 30 minutes

-- Theorem to prove
theorem doughnut_machine_completion_time : 
  -- Given conditions
  (quarter_completion_time - start_time = 3 * 60) →  -- 3 hours to complete 1/4 of the job
  -- Conclusion
  (start_time + 4 * (quarter_completion_time - start_time) + maintenance_break = 19 * 60 + 30) :=  -- 7:30 PM in minutes
by
  sorry

end NUMINAMATH_CALUDE_doughnut_machine_completion_time_l757_75771


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l757_75714

/-- Proves that if a deposit of 5000 is 20% of a person's monthly income, then their monthly income is 25000. -/
theorem monthly_income_calculation (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 5000)
  (h2 : percentage = 20)
  (h3 : deposit = (percentage / 100) * monthly_income) :
  monthly_income = 25000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l757_75714


namespace NUMINAMATH_CALUDE_race_heartbeats_l757_75728

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  heart_rate * pace * race_distance

theorem race_heartbeats :
  total_heartbeats 140 6 30 = 25200 := by
  sorry

end NUMINAMATH_CALUDE_race_heartbeats_l757_75728


namespace NUMINAMATH_CALUDE_product_expansion_l757_75720

theorem product_expansion (x : ℝ) : (9*x + 2) * (4*x^2 + 3) = 36*x^3 + 8*x^2 + 27*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l757_75720


namespace NUMINAMATH_CALUDE_torn_sheets_count_l757_75782

/-- Represents a book with numbered pages -/
structure Book where
  /-- Each sheet contains two pages -/
  pages_per_sheet : Nat
  /-- The first torn-out page number -/
  first_torn_page : Nat
  /-- The last torn-out page number -/
  last_torn_page : Nat

/-- Check if two numbers have the same digits -/
def same_digits (a b : Nat) : Prop := sorry

/-- Calculate the number of torn-out sheets -/
def torn_sheets (book : Book) : Nat := sorry

/-- Main theorem -/
theorem torn_sheets_count (book : Book) :
  book.pages_per_sheet = 2 →
  book.first_torn_page = 185 →
  same_digits book.first_torn_page book.last_torn_page →
  Even book.last_torn_page →
  torn_sheets book = 167 := by
  sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l757_75782


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l757_75795

/-- The number of ways to arrange 8 students and 2 teachers in a line,
    where the teachers cannot stand next to each other -/
def arrangement_count : ℕ :=
  Nat.factorial 8 * 9 * 8

/-- Theorem stating that the number of valid arrangements is correct -/
theorem correct_arrangement_count :
  arrangement_count = Nat.factorial 8 * 9 * 8 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l757_75795


namespace NUMINAMATH_CALUDE_red_papers_count_l757_75754

theorem red_papers_count (papers_per_box : ℕ) (num_boxes : ℕ) : 
  papers_per_box = 2 → num_boxes = 2 → papers_per_box * num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_red_papers_count_l757_75754


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_conditions_l757_75747

/-- The analysis method in mathematical proofs --/
structure AnalysisMethod where
  conclusion : Prop
  seek_conditions : Prop → Prop

/-- Definition of sufficient conditions --/
def sufficient_conditions (am : AnalysisMethod) (conditions : Prop) : Prop :=
  conditions → am.conclusion

/-- Theorem stating that the analysis method seeks sufficient conditions --/
theorem analysis_method_seeks_sufficient_conditions (am : AnalysisMethod) :
  ∃ (conditions : Prop), sufficient_conditions am conditions :=
sorry

end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_conditions_l757_75747


namespace NUMINAMATH_CALUDE_expression_value_l757_75751

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |m| = 2) : 
  2 * m - (a + b - 1) + 3 * c * d = 8 ∨ 2 * m - (a + b - 1) + 3 * c * d = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l757_75751


namespace NUMINAMATH_CALUDE_point_transformation_theorem_l757_75783

-- Define the rotation and reflection transformations
def rotate90CounterClockwise (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (h, k) := center
  let (x, y) := point
  (h - (y - k), k + (x - h))

def reflectAboutYEqualsX (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point
  (y, x)

-- State the theorem
theorem point_transformation_theorem (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let rotated := rotate90CounterClockwise (2, 3) P
  let final := reflectAboutYEqualsX rotated
  final = (4, -5) → b - a = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_point_transformation_theorem_l757_75783


namespace NUMINAMATH_CALUDE_union_of_sets_l757_75787

theorem union_of_sets : 
  let A : Set ℕ := {1,2,3,4}
  let B : Set ℕ := {2,4,5}
  A ∪ B = {1,2,3,4,5} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l757_75787


namespace NUMINAMATH_CALUDE_equation_solution_l757_75775

theorem equation_solution (a b : ℝ) (h1 : 3 = (a + 5).sqrt) (h2 : 3 = (7 * a - 2 * b + 1)^(1/3)) :
  ∃ x : ℝ, (a * (x - 2)^2 - 9 * b = 0) ∧ (x = 7/2 ∨ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l757_75775


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l757_75763

theorem binomial_coefficient_divisibility (p n : ℕ) (hp : Prime p) (hn : n ≥ p) :
  p ∣ Nat.choose n p := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l757_75763


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l757_75742

theorem quadratic_equations_solutions : 
  ∃ (s : Set ℝ), s = {0, 2, (6:ℝ)/5, -(6:ℝ)/5, -3, -7, 3, 1} ∧
  (∀ x ∈ s, x^2 - 2*x = 0 ∨ 25*x^2 - 36 = 0 ∨ x^2 + 10*x + 21 = 0 ∨ (x-3)^2 + 2*x*(x-3) = 0) ∧
  (∀ x : ℝ, x^2 - 2*x = 0 ∨ 25*x^2 - 36 = 0 ∨ x^2 + 10*x + 21 = 0 ∨ (x-3)^2 + 2*x*(x-3) = 0 → x ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l757_75742


namespace NUMINAMATH_CALUDE_walter_bus_time_l757_75762

def minutes_in_hour : ℕ := 60

def walter_schedule : Prop :=
  let wake_up_time : ℕ := 6 * 60 + 15
  let bus_departure_time : ℕ := 7 * 60
  let class_duration : ℕ := 45
  let num_classes : ℕ := 8
  let lunch_duration : ℕ := 30
  let break_duration : ℕ := 15
  let additional_time : ℕ := 2 * 60
  let return_home_time : ℕ := 16 * 60 + 30
  let total_away_time : ℕ := return_home_time - bus_departure_time
  let school_activities_time : ℕ := num_classes * class_duration + lunch_duration + break_duration + additional_time
  let bus_time : ℕ := total_away_time - school_activities_time
  bus_time = 45

theorem walter_bus_time : walter_schedule := by
  sorry

end NUMINAMATH_CALUDE_walter_bus_time_l757_75762


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l757_75716

/-- A geometric sequence with positive terms and common ratio greater than 1 -/
structure GeometricSequence where
  b : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, b n > 0
  h_q_gt_one : q > 1
  h_geometric : ∀ n, b (n + 1) = b n * q

/-- In a geometric sequence with positive terms and common ratio greater than 1,
    the sum of the 6th and 7th terms is less than the sum of the 4th and 9th terms -/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  seq.b 6 + seq.b 7 < seq.b 4 + seq.b 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l757_75716


namespace NUMINAMATH_CALUDE_seashells_given_to_jessica_l757_75732

theorem seashells_given_to_jessica (original_seashells : ℕ) (seashells_left : ℕ) 
  (h1 : original_seashells = 56)
  (h2 : seashells_left = 22)
  (h3 : seashells_left < original_seashells) :
  original_seashells - seashells_left = 34 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_jessica_l757_75732


namespace NUMINAMATH_CALUDE_fraction_equality_l757_75777

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l757_75777


namespace NUMINAMATH_CALUDE_final_price_after_reductions_ball_price_reduction_l757_75729

/-- Calculates the final price of an item after two successive price reductions -/
theorem final_price_after_reductions (original_price : ℝ) 
  (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : 
  original_price * (1 - first_reduction_percent / 100) * (1 - second_reduction_percent / 100) = 8 :=
by
  sorry

/-- The specific case of the ball price reduction problem -/
theorem ball_price_reduction : 
  20 * (1 - 20 / 100) * (1 - 50 / 100) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_final_price_after_reductions_ball_price_reduction_l757_75729


namespace NUMINAMATH_CALUDE_local_extremum_properties_l757_75746

/-- A function with a local extremum -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - 1

/-- The derivative of f -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_properties (a b : ℝ) :
  (f' a b (-1) = 0 ∧ f a b (-1) = 4) →
  (a = -3 ∧ b = -9 ∧
   ∀ x ∈ Set.Icc (0 : ℝ) 4, f (-3) (-9) x ≤ -1 ∧
   ∃ y ∈ Set.Icc (0 : ℝ) 4, f (-3) (-9) y = -28 ∧
   ∀ z ∈ Set.Icc (0 : ℝ) 4, f (-3) (-9) z ≥ -28) := by sorry

end NUMINAMATH_CALUDE_local_extremum_properties_l757_75746


namespace NUMINAMATH_CALUDE_gary_remaining_money_l757_75778

/-- Calculates the remaining money after a purchase. -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Proves that Gary has 18 dollars left after his purchase. -/
theorem gary_remaining_money :
  remaining_money 73 55 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gary_remaining_money_l757_75778


namespace NUMINAMATH_CALUDE_field_trip_vans_l757_75760

theorem field_trip_vans (buses : ℝ) (people_per_van : ℝ) (people_per_bus : ℝ) 
  (extra_people_in_buses : ℝ) :
  buses = 8 →
  people_per_van = 6 →
  people_per_bus = 18 →
  extra_people_in_buses = 108 →
  ∃ (vans : ℝ), vans = 6 ∧ people_per_bus * buses - people_per_van * vans = extra_people_in_buses :=
by
  sorry

end NUMINAMATH_CALUDE_field_trip_vans_l757_75760


namespace NUMINAMATH_CALUDE_sin_max_min_difference_l757_75705

theorem sin_max_min_difference (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 9 → f x = 2 * Real.sin (π * x / 6 - π / 3)) →
  (⨆ x ∈ Set.Icc 0 9, f x) - (⨅ x ∈ Set.Icc 0 9, f x) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_max_min_difference_l757_75705


namespace NUMINAMATH_CALUDE_possible_values_of_a_l757_75702

theorem possible_values_of_a (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 10) + 1 = (x + b) * (x + c)) →
  (a = 8 ∨ a = 12) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l757_75702


namespace NUMINAMATH_CALUDE_jacobs_gift_budget_l757_75786

theorem jacobs_gift_budget (total_budget : ℕ) (num_friends : ℕ) (friend_gift_cost : ℕ) (num_parents : ℕ) :
  total_budget = 100 →
  num_friends = 8 →
  friend_gift_cost = 9 →
  num_parents = 2 →
  (total_budget - num_friends * friend_gift_cost) / num_parents = 14 := by
  sorry

end NUMINAMATH_CALUDE_jacobs_gift_budget_l757_75786


namespace NUMINAMATH_CALUDE_combined_large_cheese_volume_l757_75752

/-- The volume of a normal rectangular block of cheese in cubic feet -/
def normal_rectangular_volume : ℝ := 4

/-- The volume of a normal cylindrical block of cheese in cubic feet -/
def normal_cylindrical_volume : ℝ := 6

/-- The width multiplier for a large rectangular block -/
def large_rect_width_mult : ℝ := 1.5

/-- The depth multiplier for a large rectangular block -/
def large_rect_depth_mult : ℝ := 3

/-- The length multiplier for a large rectangular block -/
def large_rect_length_mult : ℝ := 2

/-- The radius multiplier for a large cylindrical block -/
def large_cyl_radius_mult : ℝ := 2

/-- The height multiplier for a large cylindrical block -/
def large_cyl_height_mult : ℝ := 3

/-- Theorem stating the combined volume of a large rectangular block and a large cylindrical block -/
theorem combined_large_cheese_volume :
  (normal_rectangular_volume * large_rect_width_mult * large_rect_depth_mult * large_rect_length_mult) +
  (normal_cylindrical_volume * large_cyl_radius_mult^2 * large_cyl_height_mult) = 108 := by
  sorry

end NUMINAMATH_CALUDE_combined_large_cheese_volume_l757_75752


namespace NUMINAMATH_CALUDE_root_product_theorem_l757_75794

theorem root_product_theorem (m p : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) →
  (r = 16/3) := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l757_75794


namespace NUMINAMATH_CALUDE_inequality_proof_l757_75731

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  a * (b/a)^(1/3 : ℝ) + b * (c/b)^(1/3 : ℝ) + c * (a/c)^(1/3 : ℝ) ≤ a*b + b*c + c*a + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l757_75731


namespace NUMINAMATH_CALUDE_spade_nested_operation_l757_75773

/-- The spade operation defined as the absolute difference between two numbers -/
def spade (a b : ℝ) : ℝ := |a - b|

/-- Theorem stating that 3 ♠ (5 ♠ (8 ♠ 11)) = 1 -/
theorem spade_nested_operation : spade 3 (spade 5 (spade 8 11)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spade_nested_operation_l757_75773


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l757_75784

/-- Proves the relationship between y-coordinates of three points on an inverse proportion function -/
theorem inverse_proportion_y_relationship :
  ∀ (y₁ y₂ y₃ : ℝ),
  (y₁ = -4 / (-1)) →  -- Point A(-1, y₁) lies on y = -4/x
  (y₂ = -4 / 2) →     -- Point B(2, y₂) lies on y = -4/x
  (y₃ = -4 / 3) →     -- Point C(3, y₃) lies on y = -4/x
  (y₁ > y₃ ∧ y₃ > y₂) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l757_75784


namespace NUMINAMATH_CALUDE_batsman_total_matches_l757_75745

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  initial_matches : ℕ
  initial_average : ℝ
  additional_matches : ℕ
  additional_average : ℝ
  overall_average : ℝ

/-- Calculates the total number of matches played by a batsman -/
def total_matches (performance : BatsmanPerformance) : ℕ :=
  performance.initial_matches + performance.additional_matches

/-- Theorem stating that given the specific performance, the total matches played is 30 -/
theorem batsman_total_matches (performance : BatsmanPerformance) 
  (h1 : performance.initial_matches = 20)
  (h2 : performance.initial_average = 40)
  (h3 : performance.additional_matches = 10)
  (h4 : performance.additional_average = 13)
  (h5 : performance.overall_average = 31) :
  total_matches performance = 30 := by
  sorry


end NUMINAMATH_CALUDE_batsman_total_matches_l757_75745


namespace NUMINAMATH_CALUDE_problem_solution_l757_75791

theorem problem_solution (p_A p_B : ℝ) (h1 : p_A = 0.4) (h2 : p_B = 0.5) 
  (h3 : 0 ≤ p_A ∧ p_A ≤ 1) (h4 : 0 ≤ p_B ∧ p_B ≤ 1) :
  1 - (1 - p_A) * (1 - p_B) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l757_75791


namespace NUMINAMATH_CALUDE_fraction_calculation_l757_75750

theorem fraction_calculation : 
  (1 / 5 + 1 / 7) / (3 / 8 - 1 / 9) = 864 / 665 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l757_75750


namespace NUMINAMATH_CALUDE_initial_population_village1_is_correct_l757_75792

/-- The initial population of the first village -/
def initial_population_village1 : ℕ := 78000

/-- The yearly decrease in population of the first village -/
def yearly_decrease_village1 : ℕ := 1200

/-- The initial population of the second village -/
def initial_population_village2 : ℕ := 42000

/-- The yearly increase in population of the second village -/
def yearly_increase_village2 : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 18

/-- Theorem stating that the initial population of the first village is correct -/
theorem initial_population_village1_is_correct :
  initial_population_village1 - years_until_equal * yearly_decrease_village1 =
  initial_population_village2 + years_until_equal * yearly_increase_village2 :=
by sorry

end NUMINAMATH_CALUDE_initial_population_village1_is_correct_l757_75792


namespace NUMINAMATH_CALUDE_binomial_12_3_l757_75739

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_3_l757_75739


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l757_75723

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- The sequence of integer pairs sorted by sum and then by first element -/
def sortedPairs : List IntPair := sorry

/-- The 60th element in the sortedPairs sequence -/
def sixtiethPair : IntPair := sorry

/-- Theorem stating that the 60th pair in the sequence is (5,7) -/
theorem sixtieth_pair_is_five_seven : 
  sixtiethPair = IntPair.mk 5 7 := by sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l757_75723


namespace NUMINAMATH_CALUDE_second_bus_ride_time_l757_75704

theorem second_bus_ride_time (waiting_time first_bus_time : ℕ) 
  (h1 : waiting_time = 12)
  (h2 : first_bus_time = 30)
  (h3 : ∀ x, x = (waiting_time + first_bus_time) / 2 → x = 21) :
  ∃ second_bus_time : ℕ, second_bus_time = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_second_bus_ride_time_l757_75704


namespace NUMINAMATH_CALUDE_max_positive_integers_l757_75709

/-- A circular arrangement of 100 nonzero integers -/
def CircularArrangement := Fin 100 → ℤ

/-- Predicate to check if an arrangement satisfies the given condition -/
def SatisfiesCondition (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 100, arr i ≠ 0 ∧ arr i > arr ((i + 1) % 100) * arr ((i + 2) % 100)

/-- Count of positive integers in an arrangement -/
def PositiveCount (arr : CircularArrangement) : ℕ :=
  (Finset.univ.filter (fun i => arr i > 0)).card

/-- Theorem stating the maximum number of positive integers possible -/
theorem max_positive_integers (arr : CircularArrangement) 
  (h : SatisfiesCondition arr) : PositiveCount arr ≤ 50 := by
  sorry

end NUMINAMATH_CALUDE_max_positive_integers_l757_75709


namespace NUMINAMATH_CALUDE_line_segments_form_triangle_l757_75799

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that the line segments 5, 6, and 10 can form a triangle. -/
theorem line_segments_form_triangle :
  can_form_triangle 5 6 10 := by
  sorry


end NUMINAMATH_CALUDE_line_segments_form_triangle_l757_75799


namespace NUMINAMATH_CALUDE_parallel_lines_sum_l757_75722

/-- Two parallel lines with a given distance -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  distance : ℝ
  is_parallel : m = 8
  satisfies_distance : distance = 3

/-- The sum m + n for parallel lines with the given properties is either 48 or -12 -/
theorem parallel_lines_sum (lines : ParallelLines) : lines.m + lines.n = 48 ∨ lines.m + lines.n = -12 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_sum_l757_75722


namespace NUMINAMATH_CALUDE_unique_n_l757_75749

def is_valid_n (n : ℕ+) : Prop :=
  ∃ (k : ℕ) (d : ℕ → ℕ+),
    k ≥ 6 ∧
    (∀ i ≤ k, d i ∣ n) ∧
    (∀ i j, i < j → d i < d j) ∧
    d 1 = 1 ∧
    d k = n ∧
    n = (d 5)^2 + (d 6)^2

theorem unique_n : ∀ n : ℕ+, is_valid_n n → n = 500 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_l757_75749


namespace NUMINAMATH_CALUDE_subtraction_of_squares_l757_75798

theorem subtraction_of_squares (a : ℝ) : 3 * a^2 - a^2 = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_squares_l757_75798


namespace NUMINAMATH_CALUDE_seven_digit_sum_2015_l757_75733

theorem seven_digit_sum_2015 :
  ∃ (a b c d e f g : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
    e ≠ f ∧ e ≠ g ∧
    f ≠ g ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧
    (1000 * a + 100 * b + 10 * c + d) + (10 * e + f) + g = 2015 :=
by sorry

end NUMINAMATH_CALUDE_seven_digit_sum_2015_l757_75733


namespace NUMINAMATH_CALUDE_shortest_distance_is_eight_fifths_l757_75767

/-- Square ABCD with side length 2 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : side_length = 2)

/-- Circular arc with center A from B to D -/
structure CircularArc (sq : Square) :=
  (center : ℝ × ℝ)
  (start_point : ℝ × ℝ)
  (end_point : ℝ × ℝ)
  (is_valid : center = sq.A ∧ start_point = sq.B ∧ end_point = sq.D)

/-- Semicircle with center at midpoint of CD, from C to D -/
structure Semicircle (sq : Square) :=
  (center : ℝ × ℝ)
  (start_point : ℝ × ℝ)
  (end_point : ℝ × ℝ)
  (is_valid : center = ((sq.C.1 + sq.D.1) / 2, (sq.C.2 + sq.D.2) / 2) ∧ 
              start_point = sq.C ∧ end_point = sq.D)

/-- Intersection point of the circular arc and semicircle -/
def intersectionPoint (sq : Square) (arc : CircularArc sq) (semi : Semicircle sq) : ℝ × ℝ := sorry

/-- Shortest distance from a point to a line segment -/
def shortestDistance (point : ℝ × ℝ) (segment_start : ℝ × ℝ) (segment_end : ℝ × ℝ) : ℝ := sorry

/-- Main theorem: The shortest distance from the intersection point to AD is 8/5 -/
theorem shortest_distance_is_eight_fifths (sq : Square) 
  (arc : CircularArc sq) (semi : Semicircle sq) :
  shortestDistance (intersectionPoint sq arc semi) sq.A sq.D = 8/5 := by sorry

end NUMINAMATH_CALUDE_shortest_distance_is_eight_fifths_l757_75767


namespace NUMINAMATH_CALUDE_state_quarters_fraction_l757_75781

theorem state_quarters_fraction :
  ∀ (total_quarters : ℕ) (states_in_decade : ℕ),
    total_quarters = 18 →
    states_in_decade = 5 →
    (states_in_decade : ℚ) / (total_quarters : ℚ) = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_state_quarters_fraction_l757_75781


namespace NUMINAMATH_CALUDE_triangle_property_l757_75727

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.c * Real.cos t.C = (t.a * Real.cos t.B + t.b * Real.cos t.A) / 2)
  (h2 : t.c = 2) :
  t.C = π / 3 ∧ 
  (∀ (t' : Triangle), t'.c = 2 → t.a + t.b + t.c ≥ t'.a + t'.b + t'.c) ∧
  t.a + t.b + t.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l757_75727


namespace NUMINAMATH_CALUDE_planes_perpendicular_from_lines_l757_75706

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the perpendicular relation between planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_from_lines
  (α β : Plane) (a b : Line)
  (h1 : perpendicular_line_plane a α)
  (h2 : perpendicular_line_plane b β)
  (h3 : perpendicular_line_line a b) :
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_from_lines_l757_75706


namespace NUMINAMATH_CALUDE_square_area_given_edge_expressions_l757_75769

theorem square_area_given_edge_expressions (x : ℚ) :
  (5 * x - 20 : ℚ) = (30 - 4 * x : ℚ) →
  (5 * x - 20 : ℚ) > 0 →
  (5 * x - 20 : ℚ)^2 = 4900 / 81 := by
  sorry

end NUMINAMATH_CALUDE_square_area_given_edge_expressions_l757_75769


namespace NUMINAMATH_CALUDE_exists_point_P_satisfying_condition_l757_75712

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with edge length 10 -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A' : Point3D
  B' : Point3D
  C' : Point3D
  D' : Point3D

/-- Represents the plane intersecting the cube -/
structure IntersectingPlane where
  P : Point3D
  Q : Point3D
  R : Point3D
  S : Point3D
  T : Point3D

/-- Function to calculate distance between two points -/
def distance (p1 p2 : Point3D) : ℝ :=
  sorry

/-- Theorem stating the existence of point P satisfying the condition -/
theorem exists_point_P_satisfying_condition 
  (cube : Cube) 
  (plane : IntersectingPlane) 
  (h1 : distance cube.A plane.R / distance plane.R cube.B = 7 / 3)
  (h2 : distance cube.C plane.S / distance plane.S cube.B = 7 / 3)
  (h3 : plane.P.x = cube.D.x ∧ plane.P.y = cube.D.y)
  (h4 : plane.Q.x = cube.A.x ∧ plane.Q.y = cube.A.y)
  (h5 : plane.R.z = cube.A.z ∧ plane.R.y = cube.A.y)
  (h6 : plane.S.z = cube.B.z ∧ plane.S.x = cube.B.x)
  (h7 : plane.T.x = cube.C.x ∧ plane.T.y = cube.C.y) :
  ∃ (P : Point3D), 
    P.x = cube.D.x ∧ P.y = cube.D.y ∧ 
    cube.D.z ≤ P.z ∧ P.z ≤ cube.D'.z ∧
    2 * distance plane.Q plane.R = distance P plane.Q + distance plane.R plane.S :=
sorry

end NUMINAMATH_CALUDE_exists_point_P_satisfying_condition_l757_75712


namespace NUMINAMATH_CALUDE_unique_solution_condition_l757_75759

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, a * x - 7 + (b + 2) * x = 3) ↔ a ≠ -b - 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l757_75759


namespace NUMINAMATH_CALUDE_compute_M_l757_75780

def M : ℕ → ℕ 
| 0 => 0
| (n + 1) => 
  let k := 4 * n + 2
  (k + 2)^2 + k^2 - 2*((k + 1)^2) - 2*((k - 1)^2) + M n

theorem compute_M : M 12 = 75 := by
  sorry

end NUMINAMATH_CALUDE_compute_M_l757_75780


namespace NUMINAMATH_CALUDE_counterexample_exists_l757_75793

theorem counterexample_exists : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^3 + b^3 < 2 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l757_75793


namespace NUMINAMATH_CALUDE_marigold_fraction_l757_75707

/-- Represents the composition of flowers in a bouquet -/
structure Bouquet where
  yellow_daisies : ℚ
  white_daisies : ℚ
  yellow_marigolds : ℚ
  white_marigolds : ℚ

/-- The conditions of the flower bouquet problem -/
def bouquet_conditions (b : Bouquet) : Prop :=
  -- Half of the yellow flowers are daisies
  b.yellow_daisies = b.yellow_marigolds ∧
  -- Two-thirds of the white flowers are marigolds
  b.white_marigolds = 2 * b.white_daisies ∧
  -- Four-sevenths of the flowers are yellow
  b.yellow_daisies + b.yellow_marigolds = (4:ℚ)/7 * (b.yellow_daisies + b.white_daisies + b.yellow_marigolds + b.white_marigolds) ∧
  -- All fractions are non-negative
  0 ≤ b.yellow_daisies ∧ 0 ≤ b.white_daisies ∧ 0 ≤ b.yellow_marigolds ∧ 0 ≤ b.white_marigolds ∧
  -- The sum of all fractions is 1
  b.yellow_daisies + b.white_daisies + b.yellow_marigolds + b.white_marigolds = 1

/-- The theorem stating that marigolds constitute 4/7 of the flowers -/
theorem marigold_fraction (b : Bouquet) (h : bouquet_conditions b) :
  b.yellow_marigolds + b.white_marigolds = (4:ℚ)/7 := by
  sorry

end NUMINAMATH_CALUDE_marigold_fraction_l757_75707


namespace NUMINAMATH_CALUDE_inequality_equivalence_l757_75711

theorem inequality_equivalence (x : ℝ) : 
  -2 < (x^2 - 18*x + 24) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 18*x + 24) / (x^2 - 4*x + 8) < 2 ↔ 
  -2 < x ∧ x < 10/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l757_75711


namespace NUMINAMATH_CALUDE_boat_payment_l757_75718

theorem boat_payment (total : ℚ) (a b c d e : ℚ) : 
  total = 120 →
  a = (1/3) * (b + c + d + e) →
  b = (1/4) * (a + c + d + e) →
  c = (1/5) * (a + b + d + e) →
  d = 2 * e →
  a + b + c + d + e = total →
  e = 40/3 := by
sorry

end NUMINAMATH_CALUDE_boat_payment_l757_75718


namespace NUMINAMATH_CALUDE_min_value_of_f_l757_75757

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l757_75757


namespace NUMINAMATH_CALUDE_chairs_built_in_ten_days_l757_75770

/-- Represents the time it takes to build different types of chairs -/
structure ChairTimes where
  rocking : ℕ
  dining : ℕ
  armchair : ℕ

/-- Represents the number of chairs built -/
structure ChairsBuilt where
  rocking : ℕ
  dining : ℕ
  armchair : ℕ

/-- Calculates the maximum number of chairs that can be built in a given number of days -/
def maxChairsBuilt (shiftLength : ℕ) (times : ChairTimes) (days : ℕ) : ChairsBuilt :=
  sorry

/-- Theorem stating the maximum number of chairs that can be built in 10 days -/
theorem chairs_built_in_ten_days :
  let times : ChairTimes := ⟨5, 3, 6⟩
  let result : ChairsBuilt := maxChairsBuilt 8 times 10
  result.rocking = 10 ∧ result.dining = 10 ∧ result.armchair = 0 := by
  sorry

end NUMINAMATH_CALUDE_chairs_built_in_ten_days_l757_75770


namespace NUMINAMATH_CALUDE_range_of_f_l757_75774

def f (x : Int) : Int := x + 1

def domain : Set Int := {-1, 0, 1, 2}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {0, 1, 2, 3} :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l757_75774


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l757_75764

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- For the quadratic equation 3x^2 - 2x - 1 = 0, the discriminant equals 16 -/
theorem quadratic_discriminant : discriminant 3 (-2) (-1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l757_75764


namespace NUMINAMATH_CALUDE_min_sum_of_product_l757_75776

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 1806) :
  ∃ (x y z : ℕ+), x * y * z = 1806 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 153 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l757_75776


namespace NUMINAMATH_CALUDE_product_of_powers_and_primes_l757_75753

theorem product_of_powers_and_primes :
  2^4 * 3 * 5^3 * 7 * 11 = 2310000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_and_primes_l757_75753


namespace NUMINAMATH_CALUDE_atop_distributive_laws_l757_75717

-- Define the @ operation
def atop (a b : ℝ) : ℝ := a + 2 * b

-- State the theorem
theorem atop_distributive_laws :
  (∀ x y z : ℝ, x * (atop y z) = atop (x * y) (x * z)) ∧
  (∃ x y z : ℝ, atop x (y * z) ≠ (atop x y) * (atop x z)) ∧
  (∃ x y z : ℝ, atop (atop x y) (atop x z) ≠ atop x (y * z)) := by
  sorry

end NUMINAMATH_CALUDE_atop_distributive_laws_l757_75717


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l757_75756

/-- A point (x, y) on the parabola y^2 = 4x that maintains equal distance from (1, 0) and the line x = -1 -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- A line passing through (-2, 0) with slope k -/
def Line (k x y : ℝ) : Prop := y = k*(x + 2)

/-- The set of k values for which the line intersects the parabola -/
def IntersectionSet : Set ℝ := {k : ℝ | ∃ x y : ℝ, Parabola x y ∧ Line k x y}

theorem parabola_line_intersection :
  IntersectionSet = {k : ℝ | k ∈ Set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2)} := by sorry

#check parabola_line_intersection

end NUMINAMATH_CALUDE_parabola_line_intersection_l757_75756


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l757_75719

theorem trigonometric_expression_equals_one (α : Real) (h : Real.tan α = 2) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / (Real.cos (3 * π / 2 + α) + Real.cos (π - α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l757_75719


namespace NUMINAMATH_CALUDE_weekly_pig_feed_l757_75730

def feed_per_pig_per_day : ℕ := 10
def number_of_pigs : ℕ := 2
def days_in_week : ℕ := 7

theorem weekly_pig_feed : 
  feed_per_pig_per_day * number_of_pigs * days_in_week = 140 := by
  sorry

end NUMINAMATH_CALUDE_weekly_pig_feed_l757_75730


namespace NUMINAMATH_CALUDE_water_in_first_tank_l757_75768

theorem water_in_first_tank (capacity : ℝ) (water_second : ℝ) (fill_percentage : ℝ) (additional_water : ℝ) :
  capacity > 0 →
  water_second = 450 →
  fill_percentage = 0.45 →
  water_second = fill_percentage * capacity →
  additional_water = 1250 →
  additional_water + water_second + (capacity - water_second) = 2 * capacity →
  capacity - (additional_water + water_second) = 300 :=
by sorry

end NUMINAMATH_CALUDE_water_in_first_tank_l757_75768


namespace NUMINAMATH_CALUDE_reflect_F_l757_75734

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Theorem: Reflecting point F(3, 3) over y-axis then x-axis results in F''(-3, -3) -/
theorem reflect_F : 
  let F : ℝ × ℝ := (3, 3)
  reflect_x (reflect_y F) = (-3, -3) := by
sorry

end NUMINAMATH_CALUDE_reflect_F_l757_75734


namespace NUMINAMATH_CALUDE_tire_usage_calculation_l757_75710

/-- Calculates the miles each tire is used given the total number of tires, 
    simultaneously used tires, and total miles driven. -/
def miles_per_tire (total_tires : ℕ) (used_tires : ℕ) (total_miles : ℕ) : ℚ :=
  (total_miles * used_tires : ℚ) / total_tires

theorem tire_usage_calculation :
  let total_tires : ℕ := 6
  let used_tires : ℕ := 5
  let total_miles : ℕ := 42000
  miles_per_tire total_tires used_tires total_miles = 35000 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_calculation_l757_75710


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l757_75779

theorem factorization_cubic_minus_linear (a : ℝ) : a^3 - 16*a = a*(a + 4)*(a - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l757_75779


namespace NUMINAMATH_CALUDE_rationalize_and_sum_l757_75725

theorem rationalize_and_sum : ∃ (A B C D E F : ℤ),
  (F > 0) ∧
  (∃ (k : ℚ), k ≠ 0 ∧ 
    k * (1 / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11)) = 
    (A * Real.sqrt 5 + B * Real.sqrt 3 + C * Real.sqrt 11 + D * Real.sqrt E) / F) ∧
  (A + B + C + D + E + F = 196) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_sum_l757_75725


namespace NUMINAMATH_CALUDE_yoga_to_exercise_ratio_l757_75788

/-- Represents the time spent on different activities in minutes -/
structure ExerciseTime where
  gym : ℕ
  bike : ℕ
  yoga : ℕ

/-- Calculates the total exercise time -/
def totalExerciseTime (t : ExerciseTime) : ℕ := t.gym + t.bike

/-- Represents the ratio of gym to bike time -/
def gymBikeRatio (t : ExerciseTime) : ℚ := t.gym / t.bike

theorem yoga_to_exercise_ratio (t : ExerciseTime) 
  (h1 : gymBikeRatio t = 2/3)
  (h2 : t.bike = 18) :
  ∃ (y : ℕ), t.yoga = y ∧ y / (totalExerciseTime t) = y / 30 := by
  sorry

end NUMINAMATH_CALUDE_yoga_to_exercise_ratio_l757_75788


namespace NUMINAMATH_CALUDE_fifth_month_sales_l757_75766

def sales_problem (month1 month2 month3 month4 month6 : ℕ) (target_average : ℕ) : ℕ :=
  6 * target_average - (month1 + month2 + month3 + month4 + month6)

theorem fifth_month_sales :
  sales_problem 6635 6927 6855 7230 4791 6500 = 6562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l757_75766


namespace NUMINAMATH_CALUDE_inverse_composition_l757_75713

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the given condition
axiom inverse_relation (x : ℝ) : (f⁻¹ ∘ g) x = 4 * x - 1

-- State the theorem to be proved
theorem inverse_composition : g⁻¹ (f 5) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_l757_75713


namespace NUMINAMATH_CALUDE_dvds_left_l757_75743

def debby_dvds : ℕ := 13
def sold_dvds : ℕ := 6

theorem dvds_left : debby_dvds - sold_dvds = 7 := by sorry

end NUMINAMATH_CALUDE_dvds_left_l757_75743


namespace NUMINAMATH_CALUDE_max_area_rectangle_l757_75726

/-- The maximum area of a rectangle with integer side lengths and perimeter 150 feet is 1406 square feet. -/
theorem max_area_rectangle (w h : ℕ) : 
  w + h = 75 → w * h ≤ 1406 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l757_75726


namespace NUMINAMATH_CALUDE_double_yolked_eggs_l757_75740

/-- Given a carton of eggs with some double-yolked eggs, calculate the number of double-yolked eggs. -/
theorem double_yolked_eggs (total_eggs : ℕ) (total_yolks : ℕ) (double_yolked : ℕ) : 
  total_eggs = 12 → total_yolks = 17 → double_yolked = 5 → 
  2 * double_yolked + (total_eggs - double_yolked) = total_yolks := by
  sorry

#check double_yolked_eggs

end NUMINAMATH_CALUDE_double_yolked_eggs_l757_75740


namespace NUMINAMATH_CALUDE_crypto_deg_is_69_l757_75790

/-- Represents the digits in the cryptographer's encoding -/
inductive CryptoDigit
| A | B | C | D | E | F | G

/-- Converts a CryptoDigit to its corresponding base 7 value -/
def cryptoToBase7 : CryptoDigit → Fin 7
| CryptoDigit.A => 0
| CryptoDigit.B => 1
| CryptoDigit.D => 3
| CryptoDigit.E => 2
| CryptoDigit.F => 5
| CryptoDigit.G => 6
| _ => 0  -- C is not used in this problem, so we assign it 0

/-- Represents a three-digit number in the cryptographer's encoding -/
structure CryptoNumber where
  hundreds : CryptoDigit
  tens : CryptoDigit
  ones : CryptoDigit

/-- Converts a CryptoNumber to its base 10 value -/
def cryptoToBase10 (n : CryptoNumber) : Nat :=
  (cryptoToBase7 n.hundreds).val * 49 +
  (cryptoToBase7 n.tens).val * 7 +
  (cryptoToBase7 n.ones).val

/-- The main theorem to prove -/
theorem crypto_deg_is_69 :
  let deg : CryptoNumber := ⟨CryptoDigit.D, CryptoDigit.E, CryptoDigit.G⟩
  cryptoToBase10 deg = 69 := by sorry

end NUMINAMATH_CALUDE_crypto_deg_is_69_l757_75790


namespace NUMINAMATH_CALUDE_water_experiment_proof_l757_75703

/-- Calculates the remaining amount of water after an experiment -/
def remaining_water (initial : ℚ) (used : ℚ) : ℚ :=
  initial - used

/-- Proves that given 3 gallons of water and using 5/4 gallons, the remaining amount is 7/4 gallons -/
theorem water_experiment_proof :
  remaining_water 3 (5/4) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_water_experiment_proof_l757_75703


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l757_75748

/-- The function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ f' x = 4 ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) := by
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l757_75748


namespace NUMINAMATH_CALUDE_labourer_savings_l757_75724

/-- Calculates the amount saved by a labourer after clearing debt -/
theorem labourer_savings (monthly_income : ℕ) (initial_expenditure : ℕ) (reduced_expenditure : ℕ) : 
  monthly_income = 78 → 
  initial_expenditure = 85 → 
  reduced_expenditure = 60 → 
  (4 * monthly_income - (4 * reduced_expenditure + (6 * initial_expenditure - 6 * monthly_income))) = 30 := by
sorry

end NUMINAMATH_CALUDE_labourer_savings_l757_75724


namespace NUMINAMATH_CALUDE_spelling_bee_contestants_l757_75765

theorem spelling_bee_contestants (initial_students : ℕ) : 
  (initial_students / 2 : ℚ) / 3 = 24 → initial_students = 144 := by
  sorry

end NUMINAMATH_CALUDE_spelling_bee_contestants_l757_75765


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l757_75736

theorem average_marks_combined_classes (n₁ n₂ : ℕ) (avg₁ avg₂ : ℝ) 
  (h₁ : n₁ = 20) (h₂ : n₂ = 50) (h₃ : avg₁ = 40) (h₄ : avg₂ = 60) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℝ) = 3800 / 70 :=
sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l757_75736


namespace NUMINAMATH_CALUDE_base5_product_theorem_l757_75700

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base-10 number to base-5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Multiplies two base-5 numbers --/
def multiplyBase5 (a b : List Nat) : List Nat :=
  base10ToBase5 (base5ToBase10 a * base5ToBase10 b)

theorem base5_product_theorem :
  multiplyBase5 [1, 3, 2] [3, 1] = [3, 0, 0, 1, 4] := by sorry

end NUMINAMATH_CALUDE_base5_product_theorem_l757_75700


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l757_75789

theorem no_positive_integer_solutions :
  ∀ x : ℕ+, ¬(15 < -3 * (x : ℤ) + 18) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l757_75789


namespace NUMINAMATH_CALUDE_expected_value_is_six_point_five_l757_75738

/-- A function representing a fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of a roll of the 12-sided die -/
def expected_value : ℚ := (twelve_sided_die.sum id + twelve_sided_die.card) / (2 * twelve_sided_die.card)

/-- Theorem stating that the expected value of a roll of the 12-sided die is 6.5 -/
theorem expected_value_is_six_point_five : expected_value = 13/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_six_point_five_l757_75738


namespace NUMINAMATH_CALUDE_initial_puppies_count_l757_75772

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℚ := 8.5

/-- The number of puppies Alyssa kept -/
def puppies_kept : ℚ := 12.5

/-- The total number of puppies Alyssa had initially -/
def total_puppies : ℚ := puppies_given_away + puppies_kept

theorem initial_puppies_count : total_puppies = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l757_75772


namespace NUMINAMATH_CALUDE_stratified_sampling_elderly_count_l757_75744

theorem stratified_sampling_elderly_count 
  (total_employees : ℕ) 
  (young_employees : ℕ) 
  (middle_aged_employees : ℕ) 
  (elderly_employees : ℕ) 
  (sample_size : ℕ) :
  total_employees = young_employees + middle_aged_employees + elderly_employees →
  total_employees = 550 →
  young_employees = 300 →
  middle_aged_employees = 150 →
  elderly_employees = 100 →
  sample_size = 33 →
  (elderly_employees * sample_size) / total_employees = 6 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_elderly_count_l757_75744


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l757_75737

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l757_75737


namespace NUMINAMATH_CALUDE_field_length_problem_l757_75708

theorem field_length_problem (w l : ℝ) (h1 : l = 2 * w) (h2 : 36 = (1 / 8) * (l * w)) : l = 24 :=
by sorry

end NUMINAMATH_CALUDE_field_length_problem_l757_75708


namespace NUMINAMATH_CALUDE_division_problem_l757_75715

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l757_75715


namespace NUMINAMATH_CALUDE_households_with_only_bike_l757_75796

theorem households_with_only_bike 
  (total : Nat) 
  (neither : Nat) 
  (both : Nat) 
  (with_car : Nat) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 22)
  (h4 : with_car = 44) :
  total - neither - with_car + both = 35 := by
sorry

end NUMINAMATH_CALUDE_households_with_only_bike_l757_75796


namespace NUMINAMATH_CALUDE_shopkeeper_theft_loss_l757_75741

/-- Calculates the total percentage loss due to thefts for a shopkeeper --/
theorem shopkeeper_theft_loss (X : ℝ) (h : X > 0) : 
  let remaining_after_first_theft := 0.7 * X
  let remaining_after_first_sale := 0.75 * remaining_after_first_theft
  let remaining_after_second_theft := 0.6 * remaining_after_first_sale
  let remaining_after_second_sale := 0.7 * remaining_after_second_theft
  let final_remaining := 0.8 * remaining_after_second_sale
  (X - final_remaining) / X * 100 = 82.36 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_theft_loss_l757_75741


namespace NUMINAMATH_CALUDE_second_set_length_is_twenty_l757_75701

/-- The length of the first set of wood in feet -/
def first_set_length : ℝ := 4

/-- The factor by which the second set is longer than the first set -/
def length_factor : ℝ := 5

/-- The length of the second set of wood in feet -/
def second_set_length : ℝ := first_set_length * length_factor

theorem second_set_length_is_twenty : second_set_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_set_length_is_twenty_l757_75701


namespace NUMINAMATH_CALUDE_fraction_of_time_at_4kmh_l757_75761

/-- Represents the walking scenario described in the problem -/
structure WalkScenario where
  totalTime : ℝ
  timeAt2kmh : ℝ
  timeAt3kmh : ℝ
  timeAt4kmh : ℝ
  distanceAt2kmh : ℝ
  distanceAt3kmh : ℝ
  distanceAt4kmh : ℝ

/-- Theorem stating the fraction of time walked at 4 km/h -/
theorem fraction_of_time_at_4kmh (w : WalkScenario) : 
  w.timeAt2kmh = w.totalTime / 2 →
  w.distanceAt3kmh = (w.distanceAt2kmh + w.distanceAt3kmh + w.distanceAt4kmh) / 2 →
  w.distanceAt2kmh = 2 * w.timeAt2kmh →
  w.distanceAt3kmh = 3 * w.timeAt3kmh →
  w.distanceAt4kmh = 4 * w.timeAt4kmh →
  w.totalTime = w.timeAt2kmh + w.timeAt3kmh + w.timeAt4kmh →
  w.timeAt4kmh / w.totalTime = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_time_at_4kmh_l757_75761


namespace NUMINAMATH_CALUDE_smaller_number_is_24_l757_75758

theorem smaller_number_is_24 (x y : ℝ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) :
  min x y = 24 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_is_24_l757_75758


namespace NUMINAMATH_CALUDE_reeya_average_score_l757_75797

def reeya_scores : List ℝ := [65, 67, 76, 80, 95]

theorem reeya_average_score :
  let total := reeya_scores.sum
  let count := reeya_scores.length
  total / count = 76.6 := by sorry

end NUMINAMATH_CALUDE_reeya_average_score_l757_75797


namespace NUMINAMATH_CALUDE_star_polygon_external_intersection_angle_l757_75785

/-- 
The angle at each intersection point outside a star-polygon with n points (n > 4) 
inscribed in a circle, given that each internal angle is (180(n-4))/n degrees.
-/
theorem star_polygon_external_intersection_angle (n : ℕ) (h : n > 4) : 
  let internal_angle := (180 * (n - 4)) / n
  (360 * (n - 4)) / n = 360 - 2 * (180 - internal_angle) := by
  sorry

#check star_polygon_external_intersection_angle

end NUMINAMATH_CALUDE_star_polygon_external_intersection_angle_l757_75785


namespace NUMINAMATH_CALUDE_sum_gcd_lcm_factorial_l757_75755

theorem sum_gcd_lcm_factorial : 
  Nat.gcd 48 180 + Nat.lcm 48 180 + Nat.factorial 4 = 756 := by
  sorry

end NUMINAMATH_CALUDE_sum_gcd_lcm_factorial_l757_75755


namespace NUMINAMATH_CALUDE_subtraction_result_l757_75735

theorem subtraction_result (chosen_number : ℕ) : 
  chosen_number = 127 → (2 * chosen_number) - 152 = 102 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l757_75735


namespace NUMINAMATH_CALUDE_duck_pond_problem_l757_75721

theorem duck_pond_problem (small_pond : ℕ) (large_pond : ℕ) : 
  large_pond = 80 →
  (small_pond * 20 : ℕ) / 100 + (large_pond * 15 : ℕ) / 100 = ((small_pond + large_pond) * 16 : ℕ) / 100 →
  small_pond = 20 := by
sorry

end NUMINAMATH_CALUDE_duck_pond_problem_l757_75721
