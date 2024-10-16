import Mathlib

namespace NUMINAMATH_CALUDE_base_difference_proof_l2140_214040

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem base_difference_proof :
  let base_8_num := to_base_10 [5, 4, 3, 2, 1, 0] 8
  let base_5_num := to_base_10 [4, 3, 2, 1, 0] 5
  base_8_num - base_5_num = 177966 := by
sorry

end NUMINAMATH_CALUDE_base_difference_proof_l2140_214040


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l2140_214035

theorem purely_imaginary_condition (a : ℝ) : 
  (∃ (y : ℝ), Complex.mk (a^2 - 4) (a + 2) = Complex.I * y) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l2140_214035


namespace NUMINAMATH_CALUDE_function_properties_l2140_214006

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x + c

theorem function_properties (a b c : ℝ) 
  (h1 : f a b c 0 = 0)
  (h2 : ∀ x : ℝ, f a b c x ≤ f a b c (Real.pi / 3))
  (h3 : ∃ x : ℝ, f a b c x = 1) :
  (∃ x : ℝ, f a b c x = 1) ∧ 
  (∀ x : ℝ, f a b c x ≤ f (Real.sqrt 3) 1 (-1) x) ∧
  (f a b c (b / a) > f a b c (c / a)) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_function_properties_l2140_214006


namespace NUMINAMATH_CALUDE_smallest_number_drawn_l2140_214055

/-- Represents a systematic sampling of classes -/
structure ClassSampling where
  total_classes : ℕ
  sample_size : ℕ
  sum_of_selected : ℕ

/-- Theorem: If we have 18 classes, sample 6 of them systematically, 
    and the sum of selected numbers is 57, then the smallest number drawn is 2 -/
theorem smallest_number_drawn (s : ClassSampling) 
  (h1 : s.total_classes = 18)
  (h2 : s.sample_size = 6)
  (h3 : s.sum_of_selected = 57) :
  ∃ x : ℕ, x = 2 ∧ 
    (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) + (x + 15) = s.sum_of_selected) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_drawn_l2140_214055


namespace NUMINAMATH_CALUDE_xunzi_wangzhi_interpretation_l2140_214097

/-- Represents the four seasonal agricultural activities -/
inductive SeasonalActivity
| SpringPlowing
| SummerWeeding
| AutumnHarvesting
| WinterStoring

/-- Represents the result of following the seasonal activities -/
def SurplusFood : Prop := True

/-- Represents the concept of objective laws in nature -/
def ObjectiveLaw : Prop := True

/-- Represents the concept of subjective initiative -/
def SubjectiveInitiative : Prop := True

/-- Represents the concept of expected results -/
def ExpectedResults : Prop := True

/-- The main theorem based on the given problem -/
theorem xunzi_wangzhi_interpretation 
  (seasonal_activities : List SeasonalActivity)
  (follow_activities_lead_to_surplus : seasonal_activities.length = 4 → SurplusFood) :
  (ObjectiveLaw → ExpectedResults) ∧ 
  (SubjectiveInitiative → ObjectiveLaw) := by
  sorry


end NUMINAMATH_CALUDE_xunzi_wangzhi_interpretation_l2140_214097


namespace NUMINAMATH_CALUDE_meeting_time_is_48_minutes_l2140_214029

/-- Represents the cycling scenario between Andrea and Lauren -/
structure CyclingScenario where
  total_distance : ℝ
  andrea_speed_ratio : ℝ
  distance_decrease_rate : ℝ
  andrea_stop_time : ℝ

/-- Calculates the total time for Lauren to meet Andrea -/
def total_meeting_time (scenario : CyclingScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, the total meeting time is 48 minutes -/
theorem meeting_time_is_48_minutes 
  (scenario : CyclingScenario)
  (h1 : scenario.total_distance = 30)
  (h2 : scenario.andrea_speed_ratio = 2)
  (h3 : scenario.distance_decrease_rate = 1.5)
  (h4 : scenario.andrea_stop_time = 6) :
  total_meeting_time scenario = 48 :=
sorry

end NUMINAMATH_CALUDE_meeting_time_is_48_minutes_l2140_214029


namespace NUMINAMATH_CALUDE_david_chemistry_marks_l2140_214023

def marks_problem (english math physics biology : ℕ) (average : ℚ) : Prop :=
  let total_known := english + math + physics + biology
  let total_all := average * 5
  let chemistry := total_all - total_known
  chemistry = 63

theorem david_chemistry_marks :
  marks_problem 70 63 80 65 (68.2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_david_chemistry_marks_l2140_214023


namespace NUMINAMATH_CALUDE_exists_piecewise_linear_involution_l2140_214057

/-- A piecewise-linear function is a function whose graph is a union of a finite number of points and line segments. -/
def PiecewiseLinear (f : ℝ → ℝ) : Prop := sorry

theorem exists_piecewise_linear_involution :
  ∃ (f : ℝ → ℝ), PiecewiseLinear f ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f x ∈ Set.Icc (-1) 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f (f x) = -x) :=
sorry

end NUMINAMATH_CALUDE_exists_piecewise_linear_involution_l2140_214057


namespace NUMINAMATH_CALUDE_p_money_calculation_l2140_214095

theorem p_money_calculation (p q r : ℝ) 
  (h1 : p = (1/7 * p + 1/7 * p) + 35)
  (h2 : q = 1/7 * p) 
  (h3 : r = 1/7 * p) : 
  p = 49 := by
  sorry

end NUMINAMATH_CALUDE_p_money_calculation_l2140_214095


namespace NUMINAMATH_CALUDE_tangent_line_at_1_0_l2140_214039

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_1_0 :
  let p : ℝ × ℝ := (1, 0)
  let m : ℝ := f' p.1
  let tangent_line (x : ℝ) : ℝ := m * (x - p.1) + p.2
  ∀ x, tangent_line x = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_1_0_l2140_214039


namespace NUMINAMATH_CALUDE_initial_apples_count_apple_problem_l2140_214089

theorem initial_apples_count (num_trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ) : ℕ :=
  num_trees * apples_per_tree + remaining_apples

theorem apple_problem : initial_apples_count 3 8 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_count_apple_problem_l2140_214089


namespace NUMINAMATH_CALUDE_two_books_from_different_genres_l2140_214084

theorem two_books_from_different_genres :
  let num_genres : ℕ := 3
  let books_per_genre : ℕ := 4
  let choose_genres : ℕ := 2
  (num_genres.choose choose_genres) * books_per_genre * books_per_genre = 48 :=
by sorry

end NUMINAMATH_CALUDE_two_books_from_different_genres_l2140_214084


namespace NUMINAMATH_CALUDE_distribute_four_balls_three_boxes_l2140_214019

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Stirling number of the second kind: number of ways to partition a set of n objects into k non-empty subsets -/
def stirling_second (n k : ℕ) : ℕ := sorry

theorem distribute_four_balls_three_boxes : 
  distribute_balls 4 3 = 14 := by sorry

end NUMINAMATH_CALUDE_distribute_four_balls_three_boxes_l2140_214019


namespace NUMINAMATH_CALUDE_annettes_miscalculation_l2140_214058

theorem annettes_miscalculation (x y x_rounded y_rounded : ℤ) : 
  x = 6 → y = 3 → x_rounded = 5 → y_rounded = 4 → x_rounded - y_rounded = 1 := by
  sorry

end NUMINAMATH_CALUDE_annettes_miscalculation_l2140_214058


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2140_214099

theorem least_addition_for_divisibility (n : ℕ) : 
  let x := Nat.minFac (9 - n % 9)
  x > 0 ∧ (4499 + x) % 9 = 0 ∧ ∀ y : ℕ, y < x → (4499 + y) % 9 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2140_214099


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2140_214001

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2140_214001


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2140_214076

/-- A quadratic function f(x) = ax^2 + bx + c satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, f x = a * x^2 + b * x + c) ∧
    f 1 = 7 ∧
    f 2 = 12 ∧
    f 3 = 19

/-- Theorem stating that if f is a QuadraticFunction, then f(4) = 28 -/
theorem quadratic_function_value (f : ℝ → ℝ) (hf : QuadraticFunction f) : f 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2140_214076


namespace NUMINAMATH_CALUDE_classroom_problem_l2140_214013

theorem classroom_problem (boys girls : ℕ) : 
  boys * 5 = girls * 3 →  -- Ratio of boys to girls is 3:5
  girls = boys + 4 →      -- There are 4 more girls than boys
  boys + girls = 16       -- Total number of students is 16
  := by sorry

end NUMINAMATH_CALUDE_classroom_problem_l2140_214013


namespace NUMINAMATH_CALUDE_reasoning_is_deductive_l2140_214074

-- Define the set of all substances
variable (Substance : Type)

-- Define the property of being a metal
variable (is_metal : Substance → Prop)

-- Define the property of conducting electricity
variable (conducts_electricity : Substance → Prop)

-- Define iron as a specific substance
variable (iron : Substance)

-- Theorem stating that the given reasoning is deductive
theorem reasoning_is_deductive 
  (h1 : ∀ x, is_metal x → conducts_electricity x)  -- All metals can conduct electricity
  (h2 : is_metal iron)                             -- Iron is a metal
  (h3 : conducts_electricity iron)                 -- Iron can conduct electricity
  : Prop :=
sorry

end NUMINAMATH_CALUDE_reasoning_is_deductive_l2140_214074


namespace NUMINAMATH_CALUDE_unique_solution_l2140_214092

/-- Represents the pictures in the table --/
inductive Picture : Type
| Cat : Picture
| Chicken : Picture
| Crab : Picture
| Bear : Picture
| Goat : Picture

/-- Assignment of digits to pictures --/
def PictureAssignment := Picture → Fin 10

/-- Checks if all pictures are assigned different digits --/
def is_valid_assignment (assignment : PictureAssignment) : Prop :=
  ∀ p q : Picture, p ≠ q → assignment p ≠ assignment q

/-- Checks if the assignment satisfies the row and column sums --/
def satisfies_sums (assignment : PictureAssignment) : Prop :=
  assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab = 10 ∧
  assignment Picture.Goat + assignment Picture.Goat + assignment Picture.Crab + assignment Picture.Bear + assignment Picture.Bear = 16 ∧
  assignment Picture.Cat + assignment Picture.Bear + assignment Picture.Goat + assignment Picture.Goat + assignment Picture.Crab = 13 ∧
  assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Chicken + assignment Picture.Chicken + assignment Picture.Goat = 17 ∧
  assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Crab + assignment Picture.Goat = 11

/-- The theorem to be proved --/
theorem unique_solution :
  ∃! assignment : PictureAssignment,
    is_valid_assignment assignment ∧
    satisfies_sums assignment ∧
    assignment Picture.Cat = 1 ∧
    assignment Picture.Chicken = 5 ∧
    assignment Picture.Crab = 2 ∧
    assignment Picture.Bear = 4 ∧
    assignment Picture.Goat = 3 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2140_214092


namespace NUMINAMATH_CALUDE_boat_distribution_problem_l2140_214063

/-- Represents the boat distribution problem from "Nine Chapters on the Mathematical Art" --/
theorem boat_distribution_problem (x : ℕ) : 
  (∀ (total_boats : ℕ) (large_boat_capacity : ℕ) (small_boat_capacity : ℕ) (total_students : ℕ),
    total_boats = 8 ∧ 
    large_boat_capacity = 6 ∧ 
    small_boat_capacity = 4 ∧ 
    total_students = 38 ∧ 
    x ≤ total_boats ∧
    x * small_boat_capacity + (total_boats - x) * large_boat_capacity = total_students) →
  4 * x + 6 * (8 - x) = 38 :=
by sorry

end NUMINAMATH_CALUDE_boat_distribution_problem_l2140_214063


namespace NUMINAMATH_CALUDE_binomial_20_19_l2140_214041

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_19_l2140_214041


namespace NUMINAMATH_CALUDE_luncheon_tables_l2140_214014

def tables_needed (invited : ℕ) (no_show : ℕ) (seats_per_table : ℕ) : ℕ :=
  ((invited - no_show) + seats_per_table - 1) / seats_per_table

theorem luncheon_tables :
  tables_needed 47 7 5 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_luncheon_tables_l2140_214014


namespace NUMINAMATH_CALUDE_intersection_A_M_range_of_b_l2140_214051

-- Define the sets A, B, M, and U
def A : Set ℝ := {x | -3 < x ∧ x ≤ 6}
def B (b : ℝ) : Set ℝ := {x | b - 3 < x ∧ x < b + 7}
def M : Set ℝ := {x | -4 ≤ x ∧ x < 5}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∩ M = {x | -3 < x < 5}
theorem intersection_A_M : A ∩ M = {x : ℝ | -3 < x ∧ x < 5} := by sorry

-- Theorem 2: If B ∪ (¬UM) = R, then -2 ≤ b < -1
theorem range_of_b (b : ℝ) (h : B b ∪ (Mᶜ) = Set.univ) : -2 ≤ b ∧ b < -1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_M_range_of_b_l2140_214051


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_powers_of_five_plus_one_l2140_214024

theorem prime_pairs_dividing_powers_of_five_plus_one :
  ∀ p q : ℕ, 
    Nat.Prime p → Nat.Prime q → 
    p ∣ (5^q + 1) → q ∣ (5^p + 1) → 
    ((p = 2 ∧ q = 2) ∨ 
     (p = 2 ∧ q = 13) ∨ 
     (p = 3 ∧ q = 3) ∨ 
     (p = 3 ∧ q = 7) ∨ 
     (p = 13 ∧ q = 2) ∨ 
     (p = 7 ∧ q = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_powers_of_five_plus_one_l2140_214024


namespace NUMINAMATH_CALUDE_jungkook_total_sheets_l2140_214034

/-- The number of sheets in a bundle of colored paper -/
def sheets_per_bundle : ℕ := 10

/-- The number of bundles Jungkook has -/
def bundles : ℕ := 3

/-- The number of additional individual sheets Jungkook has -/
def individual_sheets : ℕ := 8

/-- Theorem stating the total number of sheets Jungkook has -/
theorem jungkook_total_sheets :
  bundles * sheets_per_bundle + individual_sheets = 38 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_total_sheets_l2140_214034


namespace NUMINAMATH_CALUDE_rectangle_to_cylinders_volume_ratio_l2140_214094

/-- Given a rectangle with dimensions 6 × 10, prove that when rolled into two cylinders,
    the ratio of the larger cylinder volume to the smaller cylinder volume is 5/3. -/
theorem rectangle_to_cylinders_volume_ratio :
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 10
  let cylinder1_height : ℝ := rectangle_height
  let cylinder1_circumference : ℝ := rectangle_width
  let cylinder2_height : ℝ := rectangle_width
  let cylinder2_circumference : ℝ := rectangle_height
  let cylinder1_volume := π * (cylinder1_circumference / (2 * π))^2 * cylinder1_height
  let cylinder2_volume := π * (cylinder2_circumference / (2 * π))^2 * cylinder2_height
  let larger_volume := max cylinder1_volume cylinder2_volume
  let smaller_volume := min cylinder1_volume cylinder2_volume
  larger_volume / smaller_volume = 5 / 3 := by
sorry


end NUMINAMATH_CALUDE_rectangle_to_cylinders_volume_ratio_l2140_214094


namespace NUMINAMATH_CALUDE_last_digit_of_max_value_l2140_214098

/-- Operation that combines two numbers a and b into a * b + 1 -/
def combine (a b : ℕ) : ℕ := a * b + 1

/-- Type representing the state of the blackboard -/
def Blackboard := List ℕ

/-- Function to perform one step of the operation -/
def performStep (board : Blackboard) : Blackboard :=
  match board with
  | a :: b :: rest => combine a b :: rest
  | _ => board

/-- Function to perform n steps of the operation -/
def performNSteps (n : ℕ) (board : Blackboard) : Blackboard :=
  match n with
  | 0 => board
  | n + 1 => performNSteps n (performStep board)

/-- The maximum possible value after 127 operations -/
def maxFinalValue : ℕ :=
  let initialBoard : Blackboard := List.replicate 128 1
  (performNSteps 127 initialBoard).head!

theorem last_digit_of_max_value :
  maxFinalValue % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_max_value_l2140_214098


namespace NUMINAMATH_CALUDE_coloring_book_solution_l2140_214028

/-- Represents the problem of determining the initial stock of coloring books. -/
def ColoringBookProblem (initial_stock acquired_books books_per_shelf total_shelves : ℝ) : Prop :=
  initial_stock + acquired_books = books_per_shelf * total_shelves

/-- The theorem stating the solution to the coloring book problem. -/
theorem coloring_book_solution :
  ∃ (initial_stock : ℝ),
    ColoringBookProblem initial_stock 20 4 15 ∧
    initial_stock = 40 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_solution_l2140_214028


namespace NUMINAMATH_CALUDE_last_two_average_l2140_214079

theorem last_two_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 60 →
  ((list.take 3).sum / 3 : ℝ) = 50 →
  ((list.drop 3).take 2).sum / 2 = 70 →
  ((list.drop 5).sum / 2 : ℝ) = 65 := by
sorry

end NUMINAMATH_CALUDE_last_two_average_l2140_214079


namespace NUMINAMATH_CALUDE_hancho_drank_03L_l2140_214009

/-- The amount of milk Hancho drank -/
def hancho_consumption (initial_amount yeseul_consumption gayoung_extra remaining : ℝ) : ℝ :=
  initial_amount - (yeseul_consumption + (yeseul_consumption + gayoung_extra) + remaining)

/-- Theorem stating that Hancho drank 0.3 L of milk given the initial conditions -/
theorem hancho_drank_03L (initial_amount yeseul_consumption gayoung_extra remaining : ℝ) 
  (h1 : initial_amount = 1)
  (h2 : yeseul_consumption = 0.1)
  (h3 : gayoung_extra = 0.2)
  (h4 : remaining = 0.3) :
  hancho_consumption initial_amount yeseul_consumption gayoung_extra remaining = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_hancho_drank_03L_l2140_214009


namespace NUMINAMATH_CALUDE_sum_of_cubes_and_cube_of_sum_l2140_214017

theorem sum_of_cubes_and_cube_of_sum : (3 + 6 + 9)^3 + (3^3 + 6^3 + 9^3) = 6804 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_and_cube_of_sum_l2140_214017


namespace NUMINAMATH_CALUDE_farm_harvest_after_26_days_l2140_214053

/-- Represents the daily harvest rates for a fruit farm -/
structure HarvestRates where
  ripeOrangesOdd : ℕ
  unripeOrangesOdd : ℕ
  ripeOrangesEven : ℕ
  unripeOrangesEven : ℕ
  ripeApples : ℕ
  unripeApples : ℕ

/-- Calculates the total harvest for a given number of days -/
def totalHarvest (rates : HarvestRates) (days : ℕ) :
  ℕ × ℕ × ℕ × ℕ :=
  let oddDays := (days + 1) / 2
  let evenDays := days / 2
  ( oddDays * rates.ripeOrangesOdd + evenDays * rates.ripeOrangesEven
  , oddDays * rates.unripeOrangesOdd + evenDays * rates.unripeOrangesEven
  , days * rates.ripeApples
  , days * rates.unripeApples
  )

/-- The main theorem stating the total harvest after 26 days -/
theorem farm_harvest_after_26_days (rates : HarvestRates)
  (h1 : rates.ripeOrangesOdd = 32)
  (h2 : rates.unripeOrangesOdd = 46)
  (h3 : rates.ripeOrangesEven = 28)
  (h4 : rates.unripeOrangesEven = 52)
  (h5 : rates.ripeApples = 50)
  (h6 : rates.unripeApples = 30) :
  totalHarvest rates 26 = (780, 1274, 1300, 780) := by
  sorry

end NUMINAMATH_CALUDE_farm_harvest_after_26_days_l2140_214053


namespace NUMINAMATH_CALUDE_tiling_implies_divisible_by_six_l2140_214038

/-- Represents a square floor of size n × n -/
structure Floor (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Represents the two types of tiles -/
inductive Tile
  | square  : Tile  -- 2 × 2 tile
  | rect    : Tile  -- 2 × 1 tile

/-- Represents a tiling of the floor -/
structure Tiling (n : ℕ) where
  floor : Floor n
  num_square : ℕ  -- number of 2 × 2 tiles
  num_rect : ℕ    -- number of 2 × 1 tiles
  equal_num : num_square = num_rect
  covers_floor : 4 * num_square + 2 * num_rect = n * n
  no_overlap : 4 * num_square + 2 * num_rect ≤ n * n

/-- Theorem: If a floor of size n × n can be tiled with an equal number of 2 × 2 and 2 × 1 tiles,
    then n is divisible by 6 -/
theorem tiling_implies_divisible_by_six (n : ℕ) (t : Tiling n) : 
  6 ∣ n :=
sorry

end NUMINAMATH_CALUDE_tiling_implies_divisible_by_six_l2140_214038


namespace NUMINAMATH_CALUDE_trigonometric_equation_proof_l2140_214018

theorem trigonometric_equation_proof (α : ℝ) : 
  (Real.sin (2 * α) + Real.sin (5 * α) - Real.sin (3 * α)) / 
  (Real.cos α + 1 - 2 * (Real.sin (2 * α))^2) = 2 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_proof_l2140_214018


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_a_for_nonempty_solution_l2140_214037

-- Define the function f(x)
def f (x : ℝ) : ℝ := |4*x - 1| - |x + 2|

-- Theorem for the solution set of f(x) < 8
theorem solution_set_f_less_than_8 :
  {x : ℝ | f x < 8} = {x : ℝ | -9/5 < x ∧ x < 11/3} :=
sorry

-- Theorem for the range of a
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x + 5*|x + 2| < a^2 - 8*a) ↔ (a < -1 ∨ a > 9) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_a_for_nonempty_solution_l2140_214037


namespace NUMINAMATH_CALUDE_count_numbers_satisfying_condition_l2140_214090

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n < 2000 ∧ n = 9 * sum_of_digits n

theorem count_numbers_satisfying_condition :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_condition n) ∧ S.card = 4 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_satisfying_condition_l2140_214090


namespace NUMINAMATH_CALUDE_pie_arrangement_l2140_214070

/-- Given the number of pecan and apple pies, calculates the number of complete rows when arranged with a fixed number of pies per row. -/
def calculate_rows (pecan_pies apple_pies pies_per_row : ℕ) : ℕ :=
  (pecan_pies + apple_pies) / pies_per_row

/-- Theorem stating that with 16 pecan pies and 14 apple pies, arranged in rows of 5 pies each, there will be 6 complete rows. -/
theorem pie_arrangement : calculate_rows 16 14 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_arrangement_l2140_214070


namespace NUMINAMATH_CALUDE_meeting_at_64th_light_l2140_214085

/-- Represents the meeting point of Petya and Vasya on a street with streetlights -/
def meeting_point (total_lights : ℕ) (petya_start : ℕ) (vasya_start : ℕ) 
                  (petya_position : ℕ) (vasya_position : ℕ) : ℕ :=
  let total_intervals := total_lights - 1
  let petya_intervals := petya_position - petya_start
  let vasya_intervals := vasya_start - vasya_position
  let total_covered := petya_intervals + vasya_intervals
  petya_start + (petya_intervals * 3)

theorem meeting_at_64th_light :
  meeting_point 100 1 100 22 88 = 64 := by
  sorry

#eval meeting_point 100 1 100 22 88

end NUMINAMATH_CALUDE_meeting_at_64th_light_l2140_214085


namespace NUMINAMATH_CALUDE_subset_sum_indivisibility_implies_equality_l2140_214031

theorem subset_sum_indivisibility_implies_equality (m : ℕ) (a : Fin m → ℕ) :
  (∀ i, a i ∈ Finset.range m) →
  (∀ s : Finset (Fin m), (s.sum a) % (m + 1) ≠ 0) →
  ∀ i j, a i = a j :=
sorry

end NUMINAMATH_CALUDE_subset_sum_indivisibility_implies_equality_l2140_214031


namespace NUMINAMATH_CALUDE_workbook_selection_cases_l2140_214020

/-- The number of cases to choose either a Korean workbook or a math workbook -/
def total_cases (korean_books : ℕ) (math_books : ℕ) : ℕ :=
  korean_books + math_books

/-- Theorem: Given 2 types of Korean workbooks and 4 types of math workbooks,
    the total number of cases to choose either a Korean workbook or a math workbook is 6 -/
theorem workbook_selection_cases : total_cases 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_workbook_selection_cases_l2140_214020


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l2140_214021

/-- Proves that given two rectangles with equal area, where one rectangle has dimensions 12 inches 
    by W inches, and the other rectangle has dimensions 6 inches by 30 inches, the value of W is 15 inches. -/
theorem equal_area_rectangles_width (W : ℝ) : 
  (12 * W = 6 * 30) → W = 15 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l2140_214021


namespace NUMINAMATH_CALUDE_nickels_count_l2140_214078

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Calculates the number of nickels given the total value and the number of pennies and dimes -/
def calculate_nickels (total_value : ℕ) (num_pennies : ℕ) (num_dimes : ℕ) : ℕ :=
  let pennies_value := num_pennies * penny_value
  let dimes_value := num_dimes * dime_value
  let nickels_value := total_value - pennies_value - dimes_value
  nickels_value / nickel_value

theorem nickels_count (total_value : ℕ) (num_pennies : ℕ) (num_dimes : ℕ)
    (h1 : total_value = 59)
    (h2 : num_pennies = 9)
    (h3 : num_dimes = 3) :
    calculate_nickels total_value num_pennies num_dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_nickels_count_l2140_214078


namespace NUMINAMATH_CALUDE_journey_duration_l2140_214087

-- Define the distance covered by the train
def distance : ℝ := 80

-- Define the average speed of the train
def average_speed : ℝ := 10

-- Theorem: The duration of the journey is 8 seconds
theorem journey_duration : (distance / average_speed) = 8 := by
  sorry

end NUMINAMATH_CALUDE_journey_duration_l2140_214087


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l2140_214049

/-- The area of a regular octagon inscribed in a circle -/
theorem inscribed_octagon_area (r : ℝ) (h : r^2 = 256) :
  2 * (1 + Real.sqrt 2) * (r * Real.sqrt (2 - Real.sqrt 2))^2 = 512 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l2140_214049


namespace NUMINAMATH_CALUDE_complement_union_equals_set_l2140_214027

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_union_equals_set : 
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_set_l2140_214027


namespace NUMINAMATH_CALUDE_tangent_line_circle_sum_constraint_l2140_214005

theorem tangent_line_circle_sum_constraint (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ (x y : ℝ), (m + 1) * x + (n + 1) * y - 2 = 0 ∧ 
                (x - 1)^2 + (y - 1)^2 = 1 ∧
                ∀ (x' y' : ℝ), (x' - 1)^2 + (y' - 1)^2 ≤ 1 → 
                               (m + 1) * x' + (n + 1) * y' - 2 ≥ 0) →
  m + n ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_sum_constraint_l2140_214005


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2140_214086

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the equation of the curve -/
def curve_equation (m : ℝ) (p : Point) : Prop :=
  m * (p.x^2 + p.y^2 + 2*p.y + 1) = (p.x - 2*p.y + 3)^2

/-- Defines what it means for the curve to be an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (p : Point), curve_equation m p ↔ 
    (p.x - h)^2 / a^2 + (p.y - k)^2 / b^2 = 1
  where
    h := 0  -- center x-coordinate
    k := -1 -- center y-coordinate

/-- The main theorem stating the range of m for which the curve is an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ m > 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2140_214086


namespace NUMINAMATH_CALUDE_difference_product_sum_equals_difference_of_squares_l2140_214071

theorem difference_product_sum_equals_difference_of_squares (a b : ℝ) :
  (a - b) * (b + a) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_product_sum_equals_difference_of_squares_l2140_214071


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2140_214064

theorem sum_of_squares_of_roots (p q r : ℂ) : 
  (3 * p^3 - 3 * p^2 + 6 * p - 9 = 0) →
  (3 * q^3 - 3 * q^2 + 6 * q - 9 = 0) →
  (3 * r^3 - 3 * r^2 + 6 * r - 9 = 0) →
  p^2 + q^2 + r^2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2140_214064


namespace NUMINAMATH_CALUDE_test_score_after_5_hours_l2140_214082

/-- A student's test score is directly proportional to study time -/
structure TestScore where
  maxPoints : ℝ
  scoreAfter2Hours : ℝ
  hoursStudied : ℝ
  score : ℝ
  proportional : scoreAfter2Hours / 2 = score / hoursStudied

/-- The theorem to prove -/
theorem test_score_after_5_hours (test : TestScore) 
  (h1 : test.maxPoints = 150)
  (h2 : test.scoreAfter2Hours = 90)
  (h3 : test.hoursStudied = 5) : 
  test.score = 225 := by
sorry

end NUMINAMATH_CALUDE_test_score_after_5_hours_l2140_214082


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l2140_214067

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) :
  ∃ (q' r' : ℕ+), Nat.gcd q' r' = 70 ∧
    ∀ (q'' r'' : ℕ+), Nat.gcd p q'' = 210 → Nat.gcd p r'' = 770 → Nat.gcd q'' r'' ≥ 70 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l2140_214067


namespace NUMINAMATH_CALUDE_total_heads_calculation_l2140_214043

theorem total_heads_calculation (num_hens : ℕ) (total_feet : ℕ) : num_hens = 20 → total_feet = 200 → ∃ (num_cows : ℕ), num_hens + num_cows = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_heads_calculation_l2140_214043


namespace NUMINAMATH_CALUDE_base_ratio_in_special_isosceles_trapezoid_l2140_214046

/-- Represents an isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  smaller_base : ℝ
  larger_base : ℝ
  diagonal : ℝ
  altitude : ℝ
  sum_of_bases : smaller_base + larger_base = 10
  larger_base_prop : larger_base = 2 * diagonal
  smaller_base_prop : smaller_base = 2 * altitude

/-- Theorem stating the ratio of bases in the specific isosceles trapezoid -/
theorem base_ratio_in_special_isosceles_trapezoid (t : IsoscelesTrapezoid) :
  t.smaller_base / t.larger_base = (2 * Real.sqrt 2 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_base_ratio_in_special_isosceles_trapezoid_l2140_214046


namespace NUMINAMATH_CALUDE_set_equality_l2140_214091

def I : Set Char := {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
def M : Set Char := {'c', 'd', 'e'}
def N : Set Char := {'a', 'c', 'f'}

theorem set_equality : (I \ M) ∩ (I \ N) = {'b', 'g'} := by sorry

end NUMINAMATH_CALUDE_set_equality_l2140_214091


namespace NUMINAMATH_CALUDE_part1_part2_l2140_214052

-- Define the function y
def y (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x + a - 2

-- Part 1
theorem part1 : ∀ a : ℝ, (∀ x : ℝ, y a x ≥ -2) ↔ a ∈ Set.Ici (1/3) :=
sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then { x | -1/a < x ∧ x < 1 }
  else if a = 0 then { x | x < 1 }
  else if -1 < a ∧ a < 0 then { x | x < 1 ∨ x > -1/a }
  else if a = -1 then { x | x ≠ 1 }
  else { x | x < -1/a ∨ x > 1 }

theorem part2 : ∀ a : ℝ, ∀ x : ℝ, x ∈ solution_set a ↔ a * x^2 + (1 - a) * x - 1 < 0 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l2140_214052


namespace NUMINAMATH_CALUDE_johns_age_multiple_l2140_214033

/-- Given the ages and relationships described in the problem, prove that John's age 3 years ago
    was twice James' age 6 years from now. -/
theorem johns_age_multiple (john_current_age james_brother_age james_brother_age_diff : ℕ)
  (h1 : john_current_age = 39)
  (h2 : james_brother_age = 16)
  (h3 : james_brother_age_diff = 4) : 
  (john_current_age - 3) = 2 * (james_brother_age - james_brother_age_diff + 6) := by
  sorry

end NUMINAMATH_CALUDE_johns_age_multiple_l2140_214033


namespace NUMINAMATH_CALUDE_divisible_by_seven_count_l2140_214073

theorem divisible_by_seven_count : 
  (∃! (s : Finset Nat), 
    (∀ k ∈ s, k < 100 ∧ k > 0) ∧ 
    (∀ k ∈ s, ∀ n : Nat, n > 0 → (2 * (3^(6*n)) + k * (2^(3*n+1)) - 1) % 7 = 0) ∧
    s.card = 14) := by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_count_l2140_214073


namespace NUMINAMATH_CALUDE_swimming_practice_months_l2140_214093

def total_required_hours : ℕ := 4000
def completed_hours : ℕ := 460
def practice_hours_per_month : ℕ := 400

theorem swimming_practice_months : 
  ∃ (months : ℕ), 
    months * practice_hours_per_month ≥ total_required_hours - completed_hours ∧ 
    (months - 1) * practice_hours_per_month < total_required_hours - completed_hours ∧
    months = 9 := by
  sorry

end NUMINAMATH_CALUDE_swimming_practice_months_l2140_214093


namespace NUMINAMATH_CALUDE_inequality_proof_l2140_214007

theorem inequality_proof (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt ((w^2 + x^2 + y^2 + z^2) / 4) ≥ ((wxy + wxz + wyz + xyz) / 4)^(1/3) :=
by
  sorry

where
  wxy := w * x * y
  wxz := w * x * z
  wyz := w * y * z
  xyz := x * y * z

end NUMINAMATH_CALUDE_inequality_proof_l2140_214007


namespace NUMINAMATH_CALUDE_dante_coconuts_left_l2140_214015

theorem dante_coconuts_left (paolo_coconuts : ℕ) (dante_coconuts : ℕ) (sold_coconuts : ℕ) : 
  paolo_coconuts = 14 →
  dante_coconuts = 3 * paolo_coconuts →
  sold_coconuts = 10 →
  dante_coconuts - sold_coconuts = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_dante_coconuts_left_l2140_214015


namespace NUMINAMATH_CALUDE_pizza_dough_production_l2140_214044

-- Define the given conditions
def batches_per_sack : ℕ := 15
def sacks_per_day : ℕ := 5
def days_per_week : ℕ := 7

-- Define the theorem to be proved
theorem pizza_dough_production :
  batches_per_sack * sacks_per_day * days_per_week = 525 := by
  sorry

end NUMINAMATH_CALUDE_pizza_dough_production_l2140_214044


namespace NUMINAMATH_CALUDE_greater_fourteen_game_count_l2140_214002

/-- Represents a basketball league with two divisions -/
structure BasketballLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of scheduled games in the league -/
def total_games (league : BasketballLeague) : Nat :=
  let total_teams := league.divisions * league.teams_per_division
  let games_per_team := (league.teams_per_division - 1) * league.intra_division_games +
                        league.teams_per_division * league.inter_division_games
  total_teams * games_per_team / 2

/-- The Greater Fourteen Basketball League -/
def greater_fourteen : BasketballLeague :=
  { divisions := 2,
    teams_per_division := 7,
    intra_division_games := 2,
    inter_division_games := 2 }

theorem greater_fourteen_game_count :
  total_games greater_fourteen = 182 := by
  sorry

end NUMINAMATH_CALUDE_greater_fourteen_game_count_l2140_214002


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l2140_214047

/-- Represents a trapezoid with an inscribed circle -/
structure TrapezoidWithInscribedCircle where
  /-- Distance from the center of the inscribed circle to one end of a non-parallel side -/
  distance1 : ℝ
  /-- Distance from the center of the inscribed circle to the other end of the same non-parallel side -/
  distance2 : ℝ

/-- Theorem: If the center of the inscribed circle in a trapezoid is at distances 5 and 12
    from the ends of one non-parallel side, then the length of that side is 13. -/
theorem trapezoid_side_length (t : TrapezoidWithInscribedCircle)
    (h1 : t.distance1 = 5)
    (h2 : t.distance2 = 12) :
    Real.sqrt (t.distance1^2 + t.distance2^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l2140_214047


namespace NUMINAMATH_CALUDE_cube_root_cube_equality_l2140_214077

theorem cube_root_cube_equality (x : ℝ) : x = (x^3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_cube_equality_l2140_214077


namespace NUMINAMATH_CALUDE_tuesday_lesson_duration_is_one_hour_l2140_214062

/-- Represents the duration of each lesson on Tuesday in hours -/
def tuesday_lesson_duration : ℝ := 1

/-- The total number of hours Adam spent at school over the three days -/
def total_hours : ℝ := 12

/-- The number of lessons Adam had on Monday -/
def monday_lessons : ℕ := 6

/-- The duration of each lesson on Monday in hours -/
def monday_lesson_duration : ℝ := 0.5

/-- The number of lessons Adam had on Tuesday -/
def tuesday_lessons : ℕ := 3

/-- Theorem stating that the duration of each lesson on Tuesday is 1 hour -/
theorem tuesday_lesson_duration_is_one_hour :
  tuesday_lesson_duration = 1 ∧
  total_hours = (monday_lessons : ℝ) * monday_lesson_duration +
                (tuesday_lessons : ℝ) * tuesday_lesson_duration +
                2 * (tuesday_lessons : ℝ) * tuesday_lesson_duration :=
by sorry

end NUMINAMATH_CALUDE_tuesday_lesson_duration_is_one_hour_l2140_214062


namespace NUMINAMATH_CALUDE_barometric_pressure_proof_l2140_214003

/-- Represents the combined gas law equation -/
def combined_gas_law (p1 v1 T1 p2 v2 T2 : ℝ) : Prop :=
  p1 * v1 / T1 = p2 * v2 / T2

/-- Calculates the absolute temperature from Celsius -/
def absolute_temp (celsius : ℝ) : ℝ := celsius + 273

theorem barometric_pressure_proof 
  (well_functioning_pressure : ℝ) 
  (faulty_pressure_15C : ℝ) 
  (faulty_pressure_30C : ℝ) 
  (air_free_space : ℝ) :
  well_functioning_pressure = 762 →
  faulty_pressure_15C = 704 →
  faulty_pressure_30C = 692 →
  air_free_space = 143 →
  ∃ (true_pressure : ℝ),
    true_pressure = 748 ∧
    combined_gas_law 
      (well_functioning_pressure - faulty_pressure_15C) 
      air_free_space 
      (absolute_temp 15)
      (true_pressure - faulty_pressure_30C) 
      (air_free_space + (faulty_pressure_15C - faulty_pressure_30C)) 
      (absolute_temp 30) :=
by sorry

end NUMINAMATH_CALUDE_barometric_pressure_proof_l2140_214003


namespace NUMINAMATH_CALUDE_max_value_of_shui_l2140_214050

/-- Represents the digits assigned to each Chinese character -/
structure ChineseDigits where
  jin : Fin 8
  xin : Fin 8
  li : Fin 8
  ke : Fin 8
  ba : Fin 8
  shan : Fin 8
  qiong : Fin 8
  shui : Fin 8

/-- All digits are unique -/
def all_unique (d : ChineseDigits) : Prop :=
  d.jin ≠ d.xin ∧ d.jin ≠ d.li ∧ d.jin ≠ d.ke ∧ d.jin ≠ d.ba ∧ d.jin ≠ d.shan ∧ d.jin ≠ d.qiong ∧ d.jin ≠ d.shui ∧
  d.xin ≠ d.li ∧ d.xin ≠ d.ke ∧ d.xin ≠ d.ba ∧ d.xin ≠ d.shan ∧ d.xin ≠ d.qiong ∧ d.xin ≠ d.shui ∧
  d.li ≠ d.ke ∧ d.li ≠ d.ba ∧ d.li ≠ d.shan ∧ d.li ≠ d.qiong ∧ d.li ≠ d.shui ∧
  d.ke ≠ d.ba ∧ d.ke ≠ d.shan ∧ d.ke ≠ d.qiong ∧ d.ke ≠ d.shui ∧
  d.ba ≠ d.shan ∧ d.ba ≠ d.qiong ∧ d.ba ≠ d.shui ∧
  d.shan ≠ d.qiong ∧ d.shan ≠ d.shui ∧
  d.qiong ≠ d.shui

/-- The sum of digits in each phrase is 19 -/
def sum_is_19 (d : ChineseDigits) : Prop :=
  d.jin.val + d.jin.val + d.xin.val + d.li.val = 19 ∧
  d.ke.val + d.ba.val + d.shan.val = 19 ∧
  d.shan.val + d.qiong.val + d.shui.val + d.jin.val = 19

/-- The ordering constraint: 尽 > 山 > 力 -/
def ordering_constraint (d : ChineseDigits) : Prop :=
  d.jin > d.shan ∧ d.shan > d.li

theorem max_value_of_shui (d : ChineseDigits) 
  (h1 : all_unique d) 
  (h2 : sum_is_19 d) 
  (h3 : ordering_constraint d) : 
  d.shui.val ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_shui_l2140_214050


namespace NUMINAMATH_CALUDE_unique_symmetric_shape_l2140_214010

-- Define a type for the shapes
inductive Shape : Type
  | A | B | C | D | E

-- Define a function to represent symmetry with respect to the vertical line
def isSymmetric (s : Shape) : Prop :=
  match s with
  | Shape.D => True
  | _ => False

-- Theorem statement
theorem unique_symmetric_shape :
  ∃! s : Shape, isSymmetric s :=
by
  sorry

end NUMINAMATH_CALUDE_unique_symmetric_shape_l2140_214010


namespace NUMINAMATH_CALUDE_linear_function_segment_l2140_214011

-- Define the linear function
def f (x : ℝ) := -2 * x + 3

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem linear_function_segment :
  ∃ (A B : ℝ × ℝ), 
    (∀ x ∈ domain, ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
      (x, f x) = (1 - t) • A + t • B) ∧
    (∀ t : ℝ, 0 ≤ t → t ≤ 1 → 
      ((1 - t) • A.1 + t • B.1) ∈ domain) :=
by sorry


end NUMINAMATH_CALUDE_linear_function_segment_l2140_214011


namespace NUMINAMATH_CALUDE_digit_table_size_l2140_214069

/-- A table with digits -/
structure DigitTable where
  rows : ℕ
  cols : ℕ
  digits : Fin rows → Fin cols → Fin 10

/-- The property that for any row and any two columns, there exists another row
    that differs only in those two columns -/
def hasTwoColumnDifference (t : DigitTable) : Prop :=
  ∀ (r : Fin t.rows) (c₁ c₂ : Fin t.cols),
    c₁ ≠ c₂ →
    ∃ (r' : Fin t.rows),
      r' ≠ r ∧
      (∀ (c : Fin t.cols), c ≠ c₁ ∧ c ≠ c₂ → t.digits r c = t.digits r' c) ∧
      (t.digits r c₁ ≠ t.digits r' c₁ ∨ t.digits r c₂ ≠ t.digits r' c₂)

/-- The main theorem -/
theorem digit_table_size (t : DigitTable) (h : t.cols = 10) (p : hasTwoColumnDifference t) :
  t.rows ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_digit_table_size_l2140_214069


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l2140_214026

theorem angle_sum_theorem (α β : Real) : 
  0 < α ∧ α < π/2 → -- α is acute
  0 < β ∧ β < π/2 → -- β is acute
  |Real.sin α - 1/2| + Real.sqrt ((Real.tan β - 1)^2) = 0 →
  α + β = π/2.4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l2140_214026


namespace NUMINAMATH_CALUDE_no_significant_relationship_l2140_214080

-- Define the contingency table data
def boys_enthusiasts : ℕ := 45
def boys_non_enthusiasts : ℕ := 10
def girls_enthusiasts : ℕ := 30
def girls_non_enthusiasts : ℕ := 15

-- Define the total number of students
def total_students : ℕ := boys_enthusiasts + boys_non_enthusiasts + girls_enthusiasts + girls_non_enthusiasts

-- Define the K² calculation function
def calculate_k_squared (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n : ℚ) * (a * d - b * c : ℚ)^2 / ((a + b : ℚ) * (c + d : ℚ) * (a + c : ℚ) * (b + d : ℚ))

-- Define the critical value for 95% confidence
def critical_value : ℚ := 3841 / 1000

-- Theorem statement
theorem no_significant_relationship : 
  calculate_k_squared boys_enthusiasts boys_non_enthusiasts girls_enthusiasts girls_non_enthusiasts < critical_value := by
  sorry


end NUMINAMATH_CALUDE_no_significant_relationship_l2140_214080


namespace NUMINAMATH_CALUDE_function_machine_output_l2140_214016

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 30 then step1 - 4 else step1
  if step2 ≤ 20 then step2 + 8 else step2 - 5

theorem function_machine_output : function_machine 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_function_machine_output_l2140_214016


namespace NUMINAMATH_CALUDE_sony_johnny_fish_ratio_l2140_214030

def total_fishes : ℕ := 40
def johnny_fishes : ℕ := 8

theorem sony_johnny_fish_ratio :
  (total_fishes - johnny_fishes) / johnny_fishes = 4 := by
  sorry

end NUMINAMATH_CALUDE_sony_johnny_fish_ratio_l2140_214030


namespace NUMINAMATH_CALUDE_line_plane_relationship_l2140_214096

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (l : Line) (α : Plane) (m : Line)
  (h : parallel l m ∧ contained_in m α) :
  contained_in l α ∨ parallel_plane l α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l2140_214096


namespace NUMINAMATH_CALUDE_polynomial_equality_l2140_214004

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₁ + a₃ = -39 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2140_214004


namespace NUMINAMATH_CALUDE_distance_from_origin_l2140_214056

theorem distance_from_origin (x y n : ℝ) : 
  x = 8 →
  y > 10 →
  (x - 3)^2 + (y - 10)^2 = 15^2 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (364 + 200 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l2140_214056


namespace NUMINAMATH_CALUDE_circle_area_difference_l2140_214081

theorem circle_area_difference : 
  let d₁ : ℝ := 30
  let r₁ : ℝ := d₁ / 2
  let r₂ : ℝ := 10
  let r₃ : ℝ := 5
  (π * r₁^2) - (π * r₂^2) - (π * r₃^2) = 100 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l2140_214081


namespace NUMINAMATH_CALUDE_fox_initial_coins_l2140_214066

def bridge_crossings (initial_coins : ℕ) : ℕ := 
  let after_first := 2 * initial_coins - 50
  let after_second := 2 * after_first - 50
  let after_third := 2 * after_second - 50
  2 * after_third - 50

theorem fox_initial_coins : 
  ∃ (x : ℕ), bridge_crossings x = 0 ∧ x = 47 :=
sorry

end NUMINAMATH_CALUDE_fox_initial_coins_l2140_214066


namespace NUMINAMATH_CALUDE_cone_base_radius_l2140_214088

/-- Given a cone with slant height 5 cm and lateral area 15π cm², 
    the radius of its base circle is 3 cm. -/
theorem cone_base_radius (s : ℝ) (A : ℝ) (r : ℝ) : 
  s = 5 →  -- slant height is 5 cm
  A = 15 * Real.pi →  -- lateral area is 15π cm²
  A = Real.pi * r * s →  -- formula for lateral area of a cone
  r = 3 :=  -- radius of base circle is 3 cm
by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2140_214088


namespace NUMINAMATH_CALUDE_water_drip_relationship_faucet_left_on_time_l2140_214025

/-- Represents the water drip rate in mL per second -/
def drip_rate : ℝ := 2 * 0.05

/-- Represents the relationship between time (in hours) and water volume (in mL) -/
def water_volume (time : ℝ) : ℝ := (3600 * drip_rate) * time

theorem water_drip_relationship (time : ℝ) (volume : ℝ) (h : time ≥ 0) :
  water_volume time = 360 * time :=
sorry

theorem faucet_left_on_time (volume : ℝ) (h : volume = 1620) :
  ∃ (time : ℝ), water_volume time = volume ∧ time = 4.5 :=
sorry

end NUMINAMATH_CALUDE_water_drip_relationship_faucet_left_on_time_l2140_214025


namespace NUMINAMATH_CALUDE_penumbra_ring_area_l2140_214000

/-- Given the ratio of radii of umbra to penumbra and the radius of the umbra,
    calculate the area of the penumbra ring around the umbra. -/
theorem penumbra_ring_area (umbra_radius : ℝ) (ratio_umbra : ℝ) (ratio_penumbra : ℝ) : 
  umbra_radius = 40 →
  ratio_umbra = 2 →
  ratio_penumbra = 6 →
  (ratio_penumbra / ratio_umbra * umbra_radius)^2 * Real.pi - umbra_radius^2 * Real.pi = 12800 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_penumbra_ring_area_l2140_214000


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l2140_214075

/-- Given a canoe rowing upstream at 9 km/hr and a stream speed of 1.5 km/hr,
    the speed of the canoe when rowing downstream is 12 km/hr. -/
theorem canoe_downstream_speed :
  let upstream_speed : ℝ := 9
  let stream_speed : ℝ := 1.5
  let canoe_speed : ℝ := upstream_speed + stream_speed
  let downstream_speed : ℝ := canoe_speed + stream_speed
  downstream_speed = 12 := by sorry

end NUMINAMATH_CALUDE_canoe_downstream_speed_l2140_214075


namespace NUMINAMATH_CALUDE_pen_discount_problem_l2140_214060

/-- Proves that given a 12.5% discount on pens and the ability to buy 13 more pens
    after the discount, the original number of pens that could be bought before
    the discount is 91. -/
theorem pen_discount_problem (money : ℝ) (original_price : ℝ) 
  (original_price_positive : original_price > 0) :
  let discount_rate : ℝ := 0.125
  let discounted_price : ℝ := original_price * (1 - discount_rate)
  let original_quantity : ℝ := money / original_price
  let discounted_quantity : ℝ := money / discounted_price
  discounted_quantity - original_quantity = 13 →
  original_quantity = 91 := by
sorry


end NUMINAMATH_CALUDE_pen_discount_problem_l2140_214060


namespace NUMINAMATH_CALUDE_pen_cost_calculation_l2140_214022

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    calculate the cost of 12 pens -/
theorem pen_cost_calculation (total_cost : ℚ) (pen_cost : ℚ) (pencil_cost : ℚ) : 
  total_cost = 150 →
  3 * pen_cost + 5 * pencil_cost = total_cost →
  pen_cost = 5 * pencil_cost →
  12 * pen_cost = 450 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_calculation_l2140_214022


namespace NUMINAMATH_CALUDE_john_and_alice_money_l2140_214065

theorem john_and_alice_money : 5/8 + 7/20 = 0.975 := by
  sorry

end NUMINAMATH_CALUDE_john_and_alice_money_l2140_214065


namespace NUMINAMATH_CALUDE_probability_after_removal_l2140_214054

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset ℕ)
  (card_counts : ℕ → ℕ)
  (total_cards : ℕ)

/-- Initial deck configuration -/
def initial_deck : Deck :=
  { cards := Finset.range 13,
    card_counts := λ _ => 4,
    total_cards := 52 }

/-- Deck after removing two pairs -/
def deck_after_removal (d : Deck) : Deck :=
  { cards := d.cards,
    card_counts := λ n => if d.card_counts n ≥ 2 then d.card_counts n - 2 else d.card_counts n,
    total_cards := d.total_cards - 4 }

/-- Number of ways to choose 2 cards from n cards -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Probability of selecting a pair from the remaining deck -/
def pair_probability (d : Deck) : ℚ :=
  let total_choices := choose_two d.total_cards
  let pair_choices := d.cards.sum (λ n => choose_two (d.card_counts n))
  pair_choices / total_choices

/-- Main theorem -/
theorem probability_after_removal :
  pair_probability (deck_after_removal initial_deck) = 17 / 282 := by
  sorry

end NUMINAMATH_CALUDE_probability_after_removal_l2140_214054


namespace NUMINAMATH_CALUDE_exists_axisymmetric_capital_letter_l2140_214048

-- Define a type for capital letters
inductive CapitalLetter
  | A | B | C | D | E | F | G | H | I | J | K | L | M
  | N | O | P | Q | R | S | T | U | V | W | X | Y | Z

-- Define a predicate for axisymmetric figures
def isAxisymmetric (letter : CapitalLetter) : Prop :=
  sorry  -- The actual implementation would depend on how we define axisymmetry

-- Theorem statement
theorem exists_axisymmetric_capital_letter :
  ∃ (letter : CapitalLetter), 
    (letter = CapitalLetter.A ∨ 
     letter = CapitalLetter.B ∨ 
     letter = CapitalLetter.D ∨ 
     letter = CapitalLetter.E) ∧ 
    isAxisymmetric letter :=
by
  sorry


end NUMINAMATH_CALUDE_exists_axisymmetric_capital_letter_l2140_214048


namespace NUMINAMATH_CALUDE_min_value_of_f_zero_l2140_214061

/-- A quadratic function from reals to reals -/
def QuadraticFunction := ℝ → ℝ

/-- Predicate to check if a function is quadratic -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- Predicate to check if a function is ever more than another function -/
def EverMore (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x ≥ g x

/-- The theorem statement -/
theorem min_value_of_f_zero
  (f : QuadraticFunction)
  (hquad : IsQuadratic f)
  (hf1 : f 1 = 16)
  (hg : EverMore f (fun x ↦ (x + 3)^2))
  (hh : EverMore f (fun x ↦ x^2 + 9)) :
  ∃ (min_f0 : ℝ), min_f0 = 21/2 ∧ f 0 ≥ min_f0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_zero_l2140_214061


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l2140_214008

/-- Represents the watermelon vendor's business model -/
structure WatermelonVendor where
  initialPurchasePrice : ℝ
  initialSellingPrice : ℝ
  initialDailySales : ℝ
  salesIncreaseRate : ℝ
  fixedCosts : ℝ

/-- Calculates the daily sales volume based on price reduction -/
def dailySalesVolume (w : WatermelonVendor) (priceReduction : ℝ) : ℝ :=
  w.initialDailySales + w.salesIncreaseRate * priceReduction * 10

/-- Calculates the daily profit based on price reduction -/
def dailyProfit (w : WatermelonVendor) (priceReduction : ℝ) : ℝ :=
  (w.initialSellingPrice - priceReduction - w.initialPurchasePrice) * 
  (dailySalesVolume w priceReduction) - w.fixedCosts

/-- Theorem stating the optimal price reduction for maximum sales and 200 yuan profit -/
theorem optimal_price_reduction (w : WatermelonVendor) 
  (h1 : w.initialPurchasePrice = 2)
  (h2 : w.initialSellingPrice = 3)
  (h3 : w.initialDailySales = 200)
  (h4 : w.salesIncreaseRate = 40)
  (h5 : w.fixedCosts = 24) :
  ∃ (x : ℝ), x = 0.3 ∧ 
  dailyProfit w x = 200 ∧ 
  ∀ (y : ℝ), dailyProfit w y = 200 → dailySalesVolume w x ≥ dailySalesVolume w y := by
  sorry


end NUMINAMATH_CALUDE_optimal_price_reduction_l2140_214008


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2140_214072

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 - 3*x^2 - 8*x + 40 - 8*(4*x + 4)^(1/4) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2140_214072


namespace NUMINAMATH_CALUDE_penetrated_cubes_count_stating_penetrated_cubes_calculation_correct_l2140_214036

/-- 
Given a rectangular solid with dimensions 120 × 260 × 300,
the number of unit cubes penetrated by an internal diagonal is 520.
-/
theorem penetrated_cubes_count : ℕ → ℕ → ℕ → ℕ
  | 120, 260, 300 => 520
  | _, _, _ => 0

/-- Function to calculate the number of penetrated cubes -/
def calculate_penetrated_cubes (a b c : ℕ) : ℕ :=
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c)

/-- 
Theorem stating that the calculate_penetrated_cubes function 
correctly calculates the number of penetrated cubes for the given dimensions
-/
theorem penetrated_cubes_calculation_correct :
  calculate_penetrated_cubes 120 260 300 = penetrated_cubes_count 120 260 300 := by
  sorry

#eval calculate_penetrated_cubes 120 260 300

end NUMINAMATH_CALUDE_penetrated_cubes_count_stating_penetrated_cubes_calculation_correct_l2140_214036


namespace NUMINAMATH_CALUDE_sam_final_penny_count_l2140_214083

def initial_pennies : ℕ := 980
def found_pennies : ℕ := 930
def exchanged_pennies : ℕ := 725
def gift_pennies : ℕ := 250

theorem sam_final_penny_count :
  initial_pennies + found_pennies - exchanged_pennies + gift_pennies = 1435 :=
by sorry

end NUMINAMATH_CALUDE_sam_final_penny_count_l2140_214083


namespace NUMINAMATH_CALUDE_probability_no_defective_pencils_l2140_214012

def total_pencils : ℕ := 9
def defective_pencils : ℕ := 2
def selected_pencils : ℕ := 3

def non_defective_pencils : ℕ := total_pencils - defective_pencils

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def total_ways : ℕ := choose total_pencils selected_pencils
def non_defective_ways : ℕ := choose non_defective_pencils selected_pencils

theorem probability_no_defective_pencils :
  (non_defective_ways : ℚ) / total_ways = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_no_defective_pencils_l2140_214012


namespace NUMINAMATH_CALUDE_circle_inequality_l2140_214042

/-- Given three circles with centers P, Q, R and radii p, q, r respectively,
    where p > q > r, prove that p + q + r ≠ dist P Q + dist Q R -/
theorem circle_inequality (P Q R : EuclideanSpace ℝ (Fin 2))
    (p q r : ℝ) (hp : p > q) (hq : q > r) :
    p + q + r ≠ dist P Q + dist Q R := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_l2140_214042


namespace NUMINAMATH_CALUDE_triangle_centroid_length_l2140_214068

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right triangle condition
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- BC = 6
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 36 ∧
  -- AC = 8
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 64

-- Define the centroid
def Centroid (O A B C : ℝ × ℝ) : Prop :=
  O.1 = (A.1 + B.1 + C.1) / 3 ∧ O.2 = (A.2 + B.2 + C.2) / 3

-- Define the midpoint
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Main theorem
theorem triangle_centroid_length (A B C O P Q : ℝ × ℝ) :
  Triangle A B C →
  Centroid O A B C →
  Midpoint Q A B →
  Midpoint P B C →
  ((O.1 - P.1)^2 + (O.2 - P.2)^2) = (4/9) * 73 :=
by sorry

end NUMINAMATH_CALUDE_triangle_centroid_length_l2140_214068


namespace NUMINAMATH_CALUDE_percentage_difference_l2140_214032

theorem percentage_difference : (150 * 62 / 100) - (250 * 20 / 100) = 43 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2140_214032


namespace NUMINAMATH_CALUDE_factorize_difference_of_squares_l2140_214045

theorem factorize_difference_of_squares (a b : ℝ) : a^2 - 4*b^2 = (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorize_difference_of_squares_l2140_214045


namespace NUMINAMATH_CALUDE_march_pancake_expense_l2140_214059

/-- Given the total expense on pancakes in March and the number of days,
    calculate the daily expense assuming equal consumption each day. -/
def daily_pancake_expense (total_expense : ℕ) (days : ℕ) : ℕ :=
  total_expense / days

/-- Theorem stating that the daily pancake expense in March is 11 dollars -/
theorem march_pancake_expense :
  daily_pancake_expense 341 31 = 11 := by
  sorry

end NUMINAMATH_CALUDE_march_pancake_expense_l2140_214059
