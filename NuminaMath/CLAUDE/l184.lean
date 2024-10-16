import Mathlib

namespace NUMINAMATH_CALUDE_binary_linear_equation_l184_18452

theorem binary_linear_equation (x y : ℝ) : x + y = 5 → x = 3 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_linear_equation_l184_18452


namespace NUMINAMATH_CALUDE_quadratic_point_m_value_l184_18428

theorem quadratic_point_m_value (a m : ℝ) :
  a > 0 →
  m ≠ 0 →
  3 = -a * m^2 + 2 * a * m + 3 →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_m_value_l184_18428


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l184_18426

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / (3 - i)
  Complex.im z = -1/5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l184_18426


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_37_l184_18467

theorem modular_inverse_of_3_mod_37 : ∃ x : ℤ, 
  (x * 3) % 37 = 1 ∧ 
  0 ≤ x ∧ 
  x ≤ 36 ∧ 
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_37_l184_18467


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_least_people_l184_18447

/-- Cost function for the first caterer -/
def cost1 (n : ℕ) : ℚ := 120 + 18 * n

/-- Cost function for the second caterer -/
def cost2 (n : ℕ) : ℚ := 250 + 15 * n

/-- The least number of people for which the second caterer is cheaper -/
def least_people : ℕ := 44

theorem second_caterer_cheaper_at_least_people :
  cost2 least_people < cost1 least_people ∧
  cost1 (least_people - 1) ≤ cost2 (least_people - 1) :=
by sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_least_people_l184_18447


namespace NUMINAMATH_CALUDE_earthquake_damage_in_usd_l184_18418

/-- Converts Euros to US Dollars based on a given exchange rate -/
def euro_to_usd (euro_amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  euro_amount * exchange_rate

/-- Theorem: The earthquake damage in USD is $75,000,000 -/
theorem earthquake_damage_in_usd :
  let damage_in_euros : ℝ := 50000000
  let exchange_rate : ℝ := 3/2 -- 2 Euros = 3 USD, so 1 Euro = 3/2 USD
  euro_to_usd damage_in_euros exchange_rate = 75000000 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_damage_in_usd_l184_18418


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l184_18489

theorem smallest_solution_floor_equation :
  ∀ x : ℝ, (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 19 → x ≥ Real.sqrt 119 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l184_18489


namespace NUMINAMATH_CALUDE_parallel_subset_parallel_perpendicular_planes_parallel_l184_18480

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (lineparallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Theorem 1
theorem parallel_subset_parallel 
  (a : Line) (α β : Plane) :
  parallel α β → subset a α → lineparallel a β := by sorry

-- Theorem 2
theorem perpendicular_planes_parallel 
  (a : Line) (α β : Plane) :
  perpendicular a α → perpendicular a β → parallel α β := by sorry

end NUMINAMATH_CALUDE_parallel_subset_parallel_perpendicular_planes_parallel_l184_18480


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l184_18441

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) →
  Real.sqrt (1 + (b/a)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l184_18441


namespace NUMINAMATH_CALUDE_keyboard_mouse_cost_ratio_l184_18458

/-- Given a mouse cost and total expenditure, proves the ratio of keyboard to mouse cost -/
theorem keyboard_mouse_cost_ratio 
  (mouse_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : mouse_cost = 16) 
  (h2 : total_cost = 64) 
  (h3 : ∃ n : ℝ, total_cost = mouse_cost + n * mouse_cost) :
  ∃ n : ℝ, n = 3 ∧ total_cost = mouse_cost + n * mouse_cost :=
sorry

end NUMINAMATH_CALUDE_keyboard_mouse_cost_ratio_l184_18458


namespace NUMINAMATH_CALUDE_panda_weight_l184_18464

theorem panda_weight (monkey_weight : ℕ) (panda_weight : ℕ) : 
  monkey_weight = 25 →
  panda_weight = 6 * monkey_weight + 12 →
  panda_weight = 162 := by
sorry

end NUMINAMATH_CALUDE_panda_weight_l184_18464


namespace NUMINAMATH_CALUDE_range_of_a_l184_18412

theorem range_of_a (a : ℝ) : 
  (∀ x > a, 2 * x + 2 / (x - a) ≥ 5) → a ≥ (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l184_18412


namespace NUMINAMATH_CALUDE_bug_return_probability_l184_18445

/-- Probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The probability of the bug returning to its starting vertex on the twelfth move -/
theorem bug_return_probability : Q 12 = 44287 / 177147 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l184_18445


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l184_18483

theorem crayons_lost_or_given_away 
  (initial_box1 initial_box2 initial_box3 : ℕ)
  (final_box1 final_box2 final_box3 : ℕ)
  (h1 : initial_box1 = 479)
  (h2 : initial_box2 = 352)
  (h3 : initial_box3 = 621)
  (h4 : final_box1 = 134)
  (h5 : final_box2 = 221)
  (h6 : final_box3 = 487) :
  (initial_box1 - final_box1) + (initial_box2 - final_box2) + (initial_box3 - final_box3) = 610 :=
by sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l184_18483


namespace NUMINAMATH_CALUDE_proposition_truth_l184_18466

theorem proposition_truth : ∃ (a b : ℝ), (a * b = 0 ∧ a ≠ 0) ∧ (3 ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l184_18466


namespace NUMINAMATH_CALUDE_test_problems_count_l184_18409

theorem test_problems_count :
  let total_points : ℕ := 110
  let computation_problems : ℕ := 20
  let points_per_computation : ℕ := 3
  let points_per_word : ℕ := 5
  let word_problems : ℕ := (total_points - computation_problems * points_per_computation) / points_per_word
  computation_problems + word_problems = 30 := by
  sorry

end NUMINAMATH_CALUDE_test_problems_count_l184_18409


namespace NUMINAMATH_CALUDE_max_rented_trucks_theorem_l184_18482

/-- Represents the state of a truck rental lot over a week -/
structure TruckRentalLot where
  total_trucks : ℕ
  returned_ratio : ℚ
  saturday_trucks : ℕ

/-- The maximum number of different trucks that could have been rented out during the week -/
def max_rented_trucks (lot : TruckRentalLot) : ℕ :=
  min lot.total_trucks (2 * lot.saturday_trucks)

/-- Theorem stating the maximum number of different trucks that could have been rented out -/
theorem max_rented_trucks_theorem (lot : TruckRentalLot) 
  (h1 : lot.total_trucks = 24)
  (h2 : lot.returned_ratio = 1/2)
  (h3 : lot.saturday_trucks ≥ 12) :
  max_rented_trucks lot = 24 := by
  sorry

#eval max_rented_trucks { total_trucks := 24, returned_ratio := 1/2, saturday_trucks := 12 }

end NUMINAMATH_CALUDE_max_rented_trucks_theorem_l184_18482


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l184_18435

/-- Given the cost of two varieties of rice and their mixing ratio, 
    calculate the cost of the second variety. -/
theorem rice_mixture_cost 
  (cost_first : ℝ) 
  (cost_mixture : ℝ) 
  (ratio : ℝ) 
  (h1 : cost_first = 5.5)
  (h2 : cost_mixture = 7.50)
  (h3 : ratio = 0.625) :
  ∃ (cost_second : ℝ), 
    cost_second = 10.7 ∧ 
    (cost_first - cost_mixture) / (cost_mixture - cost_second) = ratio / 1 :=
by sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l184_18435


namespace NUMINAMATH_CALUDE_total_eyes_in_pond_l184_18451

/-- The number of snakes in the pond -/
def num_snakes : ℕ := 18

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

/-- The number of spiders in the pond -/
def num_spiders : ℕ := 5

/-- The number of snails in the pond -/
def num_snails : ℕ := 15

/-- The number of eyes a snake has -/
def snake_eyes : ℕ := 2

/-- The number of eyes an alligator has -/
def alligator_eyes : ℕ := 2

/-- The number of eyes a spider has -/
def spider_eyes : ℕ := 8

/-- The number of eyes a snail has -/
def snail_eyes : ℕ := 2

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := num_snakes * snake_eyes + num_alligators * alligator_eyes + 
                      num_spiders * spider_eyes + num_snails * snail_eyes

theorem total_eyes_in_pond : total_eyes = 126 := by sorry

end NUMINAMATH_CALUDE_total_eyes_in_pond_l184_18451


namespace NUMINAMATH_CALUDE_tenth_group_draw_l184_18405

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : Nat
  sample_size : Nat
  group_size : Nat
  first_draw : Nat

/-- Calculates the number drawn from a specific group in systematic sampling -/
def draw_from_group (s : SystematicSampling) (group : Nat) : Nat :=
  s.first_draw + s.group_size * (group - 1)

theorem tenth_group_draw (s : SystematicSampling) 
  (h1 : s.total_students = 1000)
  (h2 : s.sample_size = 100)
  (h3 : s.group_size = 10)
  (h4 : s.first_draw = 6) :
  draw_from_group s 10 = 96 := by
  sorry

end NUMINAMATH_CALUDE_tenth_group_draw_l184_18405


namespace NUMINAMATH_CALUDE_projectile_meeting_time_l184_18436

/-- Given two objects traveling towards each other, calculate the time it takes for them to meet. -/
theorem projectile_meeting_time 
  (initial_distance : ℝ) 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (h1 : initial_distance = 1182) 
  (h2 : speed1 = 460) 
  (h3 : speed2 = 525) : 
  (initial_distance / (speed1 + speed2)) * 60 = 72 := by
  sorry

end NUMINAMATH_CALUDE_projectile_meeting_time_l184_18436


namespace NUMINAMATH_CALUDE_go_match_results_l184_18420

structure GoMatch where
  redWinProb : ℝ
  mk_prob_valid : 0 ≤ redWinProb ∧ redWinProb ≤ 1

def RedTeam := Fin 3 → GoMatch

def atLeastTwoWins (team : RedTeam) : ℝ :=
  sorry

def expectedWins (team : RedTeam) : ℝ :=
  sorry

theorem go_match_results (team : RedTeam) 
  (h1 : team 0 = ⟨0.6, sorry⟩) 
  (h2 : team 1 = ⟨0.5, sorry⟩)
  (h3 : team 2 = ⟨0.5, sorry⟩) :
  atLeastTwoWins team = 0.55 ∧ expectedWins team = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_go_match_results_l184_18420


namespace NUMINAMATH_CALUDE_hadassah_painting_time_l184_18422

/-- Represents the time taken to paint paintings and take breaks -/
def total_time (small_paint_rate : ℝ) (large_paint_rate : ℝ) (small_count : ℕ) (large_count : ℕ) (break_duration : ℝ) (paintings_per_break : ℕ) : ℝ :=
  let small_time := small_paint_rate * small_count
  let large_time := large_paint_rate * large_count
  let total_paintings := small_count + large_count
  let break_count := total_paintings / paintings_per_break
  let break_time := break_count * break_duration
  small_time + large_time + break_time

/-- Theorem stating the total time Hadassah takes to finish all paintings -/
theorem hadassah_painting_time : 
  let small_paint_rate := 6 / 12
  let large_paint_rate := 8 / 6
  let small_count := 15
  let large_count := 10
  let break_duration := 0.5
  let paintings_per_break := 3
  total_time small_paint_rate large_paint_rate small_count large_count break_duration paintings_per_break = 24.8 := by
  sorry

end NUMINAMATH_CALUDE_hadassah_painting_time_l184_18422


namespace NUMINAMATH_CALUDE_fraction_inequality_l184_18473

theorem fraction_inequality (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0)
  (h1 : 1 / a > 1 / b) (h2 : x > y) : 
  x / (x + a) > y / (y + b) := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l184_18473


namespace NUMINAMATH_CALUDE_inequality_proof_l184_18400

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) :
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l184_18400


namespace NUMINAMATH_CALUDE_category_d_cost_after_discount_l184_18457

/-- Represents the cost and discount information for a category of items --/
structure Category where
  percentage : Real
  discount_rate : Real

/-- Calculates the cost of items in a category after applying the discount --/
def cost_after_discount (total_cost : Real) (category : Category) : Real :=
  let cost_before_discount := total_cost * category.percentage
  cost_before_discount * (1 - category.discount_rate)

/-- Theorem stating that the cost of category D items after discount is 562.5 --/
theorem category_d_cost_after_discount (total_cost : Real) (category_d : Category) :
  total_cost = 2500 →
  category_d.percentage = 0.25 →
  category_d.discount_rate = 0.10 →
  cost_after_discount total_cost category_d = 562.5 := by
  sorry

#check category_d_cost_after_discount

end NUMINAMATH_CALUDE_category_d_cost_after_discount_l184_18457


namespace NUMINAMATH_CALUDE_triangle_inequality_l184_18406

/-- 
Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if a^2 - √3ab + b^2 = 1 and c = 1, then 1 < √3a - b < √3.
-/
theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  a^2 - Real.sqrt 3 * a * b + b^2 = 1 ∧  -- Given condition
  c = 1 →  -- Given condition
  1 < Real.sqrt 3 * a - b ∧ Real.sqrt 3 * a - b < Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l184_18406


namespace NUMINAMATH_CALUDE_equation_solution_l184_18403

theorem equation_solution (x : ℝ) : 
  (8 / (Real.sqrt (x - 8) - 10) + 3 / (Real.sqrt (x - 8) - 5) + 
   4 / (Real.sqrt (x - 8) + 5) + 15 / (Real.sqrt (x - 8) + 10) = 0) ↔ 
  (x = 33 ∨ x = 108) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l184_18403


namespace NUMINAMATH_CALUDE_pool_capacity_l184_18434

theorem pool_capacity (C : ℝ) (h1 : C > 0) : 
  (0.4 * C + 300 = 0.8 * C) → C = 750 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l184_18434


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l184_18475

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 0 = 3 →
  a 1 = 10 →
  a 2 = 17 →
  a 5 = 32 →
  a 3 + a 4 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l184_18475


namespace NUMINAMATH_CALUDE_intersection_point_values_l184_18472

theorem intersection_point_values (m n : ℚ) : 
  (1 / 2 : ℚ) * 1 + n = -2 → -- y = x/2 + n at x = 1
  m * 1 - 1 = -2 →          -- y = mx - 1 at x = 1
  m = -1 ∧ n = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_values_l184_18472


namespace NUMINAMATH_CALUDE_class_ratio_proof_l184_18439

theorem class_ratio_proof (eduardo_classes : ℕ) (total_classes : ℕ) 
  (h1 : eduardo_classes = 3)
  (h2 : total_classes = 9) :
  (total_classes - eduardo_classes) / eduardo_classes = 2 := by
sorry

end NUMINAMATH_CALUDE_class_ratio_proof_l184_18439


namespace NUMINAMATH_CALUDE_cleaning_room_time_l184_18416

/-- Represents the time spent on various tasks in minutes -/
structure TaskTimes where
  total : ℕ
  laundry : ℕ
  bathroom : ℕ
  homework : ℕ

/-- Calculates the time spent cleaning the room given the other task times -/
def timeCleaningRoom (t : TaskTimes) : ℕ :=
  t.total - (t.laundry + t.bathroom + t.homework)

/-- Theorem stating that given the specific task times, the time spent cleaning the room is 35 minutes -/
theorem cleaning_room_time :
  let t : TaskTimes := {
    total := 120,
    laundry := 30,
    bathroom := 15,
    homework := 40
  }
  timeCleaningRoom t = 35 := by sorry

end NUMINAMATH_CALUDE_cleaning_room_time_l184_18416


namespace NUMINAMATH_CALUDE_expansion_properties_l184_18427

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Sum of odd-indexed binomial coefficients for (a + b)^n -/
def sumOddCoeffs (n : ℕ) : ℕ := sorry

/-- The constant term in the expansion of (x + 1/√x)^n -/
def constantTerm (n : ℕ) : ℕ := sorry

/-- Theorem about the expansion of (x + 1/√x)^9 -/
theorem expansion_properties :
  (sumOddCoeffs 9 = 256) ∧ (constantTerm 9 = 84) := by sorry

end NUMINAMATH_CALUDE_expansion_properties_l184_18427


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_for_f_geq_a_squared_minus_a_l184_18471

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for part I
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_geq_a_squared_minus_a :
  {a : ℝ | ∀ x : ℝ, f x ≥ a^2 - a} = {a : ℝ | -1 ≤ a ∧ a ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_for_f_geq_a_squared_minus_a_l184_18471


namespace NUMINAMATH_CALUDE_thousandth_digit_is_three_l184_18494

/-- The sequence of digits obtained by concatenating integers from 1 to 499 -/
def digit_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => if n + 1 < 499 then digit_sequence n * 10 + (n + 2) else digit_sequence n

/-- The nth digit in the sequence -/
def nth_digit (n : ℕ) : ℕ :=
  (digit_sequence (n / 9) / (10 ^ (n % 9))) % 10

/-- Theorem stating that the 1000th digit is 3 -/
theorem thousandth_digit_is_three : nth_digit 999 = 3 := by
  sorry

end NUMINAMATH_CALUDE_thousandth_digit_is_three_l184_18494


namespace NUMINAMATH_CALUDE_solution_comparison_l184_18431

theorem solution_comparison (c c' d d' : ℝ) (hc : c ≠ 0) (hc' : c' ≠ 0) :
  (-d / c > -d' / c') ↔ (d' / c' < d / c) := by sorry

end NUMINAMATH_CALUDE_solution_comparison_l184_18431


namespace NUMINAMATH_CALUDE_work_completion_time_l184_18499

/-- Given workers A, B, and C who can complete a work individually in 4, 8, and 8 days respectively,
    prove that they can complete the work together in 2 days. -/
theorem work_completion_time (work : ℝ) (days_A days_B days_C : ℝ) 
    (h_work : work > 0)
    (h_A : days_A = 4)
    (h_B : days_B = 8)
    (h_C : days_C = 8) :
    work / (work / days_A + work / days_B + work / days_C) = 2 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l184_18499


namespace NUMINAMATH_CALUDE_min_one_by_one_required_l184_18404

/-- Represents a square on the grid -/
inductive Square
  | one : Square  -- 1x1 square
  | two : Square  -- 2x2 square
  | three : Square -- 3x3 square

/-- The size of the grid -/
def gridSize : Nat := 23

/-- Represents a cell on the grid -/
structure Cell where
  row : Fin gridSize
  col : Fin gridSize

/-- A covering of the grid -/
def Covering := List (Square × Cell)

/-- Checks if a covering is valid (covers all cells except one) -/
def isValidCovering (c : Covering) : Prop := sorry

/-- Checks if a covering uses only 2x2 and 3x3 squares -/
def usesOnlyTwoAndThree (c : Covering) : Prop := sorry

theorem min_one_by_one_required :
  ¬∃ (c : Covering), isValidCovering c ∧ usesOnlyTwoAndThree c :=
sorry

end NUMINAMATH_CALUDE_min_one_by_one_required_l184_18404


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l184_18421

theorem solve_system_of_equations (x y : ℤ) 
  (h1 : x + y = 14) 
  (h2 : x - y = 60) : 
  x = 37 := by sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l184_18421


namespace NUMINAMATH_CALUDE_usual_time_calculation_l184_18493

theorem usual_time_calculation (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_time > 0) (h2 : usual_speed > 0) : 
  (usual_speed * usual_time = (usual_speed / 2) * (usual_time + 24)) → 
  usual_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_calculation_l184_18493


namespace NUMINAMATH_CALUDE_line_connecting_centers_l184_18437

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the centers of the circles
def center1 : ℝ × ℝ := (2, -3)
def center2 : ℝ × ℝ := (3, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem line_connecting_centers :
  ∀ (x y : ℝ),
  (x = center1.1 ∧ y = center1.2) ∨ (x = center2.1 ∧ y = center2.2) →
  line_equation x y :=
sorry

end NUMINAMATH_CALUDE_line_connecting_centers_l184_18437


namespace NUMINAMATH_CALUDE_complement_of_union_equals_set_l184_18478

/-- The universal set U -/
def U : Set Int := {-2, -1, 0, 1, 2, 3}

/-- Set A -/
def A : Set Int := {-1, 2}

/-- Set B -/
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

/-- The main theorem -/
theorem complement_of_union_equals_set (h : U = {-2, -1, 0, 1, 2, 3}) :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_set_l184_18478


namespace NUMINAMATH_CALUDE_little_john_initial_money_l184_18481

def sweets_cost : ℝ := 1.05
def friend_gift : ℝ := 1.00
def num_friends : ℕ := 2
def money_left : ℝ := 17.05

theorem little_john_initial_money :
  sweets_cost + friend_gift * num_friends + money_left = 20.10 := by
  sorry

end NUMINAMATH_CALUDE_little_john_initial_money_l184_18481


namespace NUMINAMATH_CALUDE_parallel_transitive_parallel_common_not_parallel_to_common_not_parallel_no_common_l184_18488

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop := sorry

theorem parallel_transitive (l1 l2 l3 : Line) :
  (parallel l1 l3 ∧ parallel l2 l3) → parallel l1 l2 :=
sorry

theorem parallel_common (l1 l2 : Line) :
  parallel l1 l2 → ∃ l3 : Line, parallel l1 l3 ∧ parallel l2 l3 :=
sorry

theorem not_parallel_to_common (l1 l2 l3 : Line) :
  (¬ parallel l1 l3 ∨ ¬ parallel l2 l3) → ¬ parallel l1 l2 :=
sorry

theorem not_parallel_no_common (l1 l2 : Line) :
  ¬ parallel l1 l2 → ¬ ∃ l3 : Line, parallel l1 l3 ∧ parallel l2 l3 :=
sorry

end NUMINAMATH_CALUDE_parallel_transitive_parallel_common_not_parallel_to_common_not_parallel_no_common_l184_18488


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l184_18497

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l184_18497


namespace NUMINAMATH_CALUDE_midpoint_locus_of_intersection_l184_18410

/-- Given an arithmetic sequence A, B, C, this function represents the line Ax + By + C = 0 --/
def line (A B C : ℝ) (x y : ℝ) : Prop :=
  A * x + B * y + C = 0

/-- The parabola y = -2x^2 --/
def parabola (x y : ℝ) : Prop :=
  y = -2 * x^2

/-- The locus of the midpoint --/
def midpoint_locus (x y : ℝ) : Prop :=
  y + 1 = -(2 * x - 1)^2

/-- The main theorem --/
theorem midpoint_locus_of_intersection
  (A B C : ℝ) -- A, B, C are real numbers
  (h_arithmetic : A - 2*B + C = 0) -- A, B, C form an arithmetic sequence
  (x y : ℝ) -- x and y are real numbers
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line A B C x₁ y₁ ∧ parabola x₁ y₁ ∧
    line A B C x₂ y₂ ∧ parabola x₂ y₂ ∧
    x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2) -- (x, y) is the midpoint of the chord of intersection
  : midpoint_locus x y :=
sorry

end NUMINAMATH_CALUDE_midpoint_locus_of_intersection_l184_18410


namespace NUMINAMATH_CALUDE_equal_roots_when_m_is_negative_half_l184_18474

theorem equal_roots_when_m_is_negative_half :
  let f (x m : ℝ) := (x * (x - 1) - (m^2 + m*x + 1)) / ((x - 1) * (m - 1)) - x / m
  ∀ x₁ x₂ : ℝ, f x₁ (-1/2) = 0 → f x₂ (-1/2) = 0 → x₁ = x₂ := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_when_m_is_negative_half_l184_18474


namespace NUMINAMATH_CALUDE_division_problem_l184_18461

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
    (h1 : dividend = 2944)
    (h2 : divisor = 72)
    (h3 : remainder = 64)
    (h4 : dividend = divisor * quotient + remainder) :
  quotient = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l184_18461


namespace NUMINAMATH_CALUDE_company_employees_count_l184_18460

theorem company_employees_count :
  let total_employees : ℝ := 140
  let prefer_x : ℝ := 0.6
  let prefer_y : ℝ := 0.4
  let max_satisfied : ℝ := 140
  prefer_x + prefer_y = 1 →
  prefer_x * total_employees + prefer_y * total_employees = max_satisfied →
  total_employees = 140 :=
by sorry

end NUMINAMATH_CALUDE_company_employees_count_l184_18460


namespace NUMINAMATH_CALUDE_student_congress_sample_size_l184_18424

theorem student_congress_sample_size 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (selected_students : ℕ) 
  (h1 : num_classes = 40) 
  (h2 : students_per_class = 50) 
  (h3 : selected_students = 150) : 
  selected_students = 150 := by
sorry

end NUMINAMATH_CALUDE_student_congress_sample_size_l184_18424


namespace NUMINAMATH_CALUDE_joans_books_l184_18491

theorem joans_books (tom_books : ℕ) (total_books : ℕ) (h1 : tom_books = 38) (h2 : total_books = 48) :
  total_books - tom_books = 10 := by
sorry

end NUMINAMATH_CALUDE_joans_books_l184_18491


namespace NUMINAMATH_CALUDE_binary_digits_difference_l184_18423

-- Define a function to calculate the number of digits in the binary representation of a number
def binaryDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

-- State the theorem
theorem binary_digits_difference :
  binaryDigits 1500 - binaryDigits 300 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digits_difference_l184_18423


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_seven_l184_18495

theorem sum_of_solutions_is_seven : 
  let f (x : ℝ) := |x^2 - 8*x + 12|
  let g (x : ℝ) := 35/4 - x
  ∃ (a b : ℝ), (f a = g a) ∧ (f b = g b) ∧ (a + b = 7) ∧ 
    (∀ (x : ℝ), (f x = g x) → (x = a ∨ x = b)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_seven_l184_18495


namespace NUMINAMATH_CALUDE_soda_quarters_needed_l184_18443

theorem soda_quarters_needed (total_amount : ℚ) (quarters_per_soda : ℕ) : 
  total_amount = 213.75 ∧ quarters_per_soda = 7 →
  (⌊(total_amount / 0.25) / quarters_per_soda⌋ + 1) * quarters_per_soda - 
  (total_amount / 0.25).floor = 6 := by
  sorry

end NUMINAMATH_CALUDE_soda_quarters_needed_l184_18443


namespace NUMINAMATH_CALUDE_ring_sector_area_proof_l184_18485

/-- The area of a ring-shaped sector formed by two concentric circles with radii 13 and 7, and a common central angle θ -/
def ring_sector_area (θ : Real) : Real :=
  60 * θ

/-- Theorem: The area of a ring-shaped sector formed by two concentric circles
    with radii 13 and 7, and a common central angle θ, is equal to 60θ -/
theorem ring_sector_area_proof (θ : Real) :
  ring_sector_area θ = 60 * θ := by
  sorry

#check ring_sector_area_proof

end NUMINAMATH_CALUDE_ring_sector_area_proof_l184_18485


namespace NUMINAMATH_CALUDE_smallest_integer_y_l184_18442

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y < 20) ∧ (∀ z : ℤ, z < y → ¬(7 - 3 * z < 20)) → y = -4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l184_18442


namespace NUMINAMATH_CALUDE_gcd_16_12_l184_18479

def operation_process : List (Nat × Nat) := [(16, 12), (4, 12), (4, 8), (4, 4)]

theorem gcd_16_12 : Nat.gcd 16 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_16_12_l184_18479


namespace NUMINAMATH_CALUDE_quadratic_trinomial_factorization_l184_18446

theorem quadratic_trinomial_factorization 
  (a b c x x₁ x₂ : ℝ) 
  (ha : a ≠ 0) 
  (hx₁ : a * x₁^2 + b * x₁ + c = 0) 
  (hx₂ : a * x₂^2 + b * x₂ + c = 0) : 
  a * x^2 + b * x + c = a * (x - x₁) * (x - x₂) := by
sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_factorization_l184_18446


namespace NUMINAMATH_CALUDE_second_next_perfect_square_l184_18449

theorem second_next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n^2 = x + 4 * (x : ℝ).sqrt + 4 :=
sorry

end NUMINAMATH_CALUDE_second_next_perfect_square_l184_18449


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_4_l184_18429

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + 2 * |x - a|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ x + 4} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 7/2} := by sorry

-- Part 2
theorem range_of_a_when_f_geq_4 :
  {a : ℝ | ∀ x, f a x ≥ 4} = {a : ℝ | a ≤ -5 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_4_l184_18429


namespace NUMINAMATH_CALUDE_number_of_candies_bought_l184_18419

/-- Given the cost of snacks and candies, the total number of items, and the total amount spent,
    prove that the number of candies bought is 3. -/
theorem number_of_candies_bought
  (snack_cost : ℕ)
  (candy_cost : ℕ)
  (total_items : ℕ)
  (total_spent : ℕ)
  (h1 : snack_cost = 300)
  (h2 : candy_cost = 500)
  (h3 : total_items = 8)
  (h4 : total_spent = 3000)
  : ∃ (num_candies : ℕ), num_candies = 3 ∧
    ∃ (num_snacks : ℕ),
      num_snacks + num_candies = total_items ∧
      num_snacks * snack_cost + num_candies * candy_cost = total_spent :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_candies_bought_l184_18419


namespace NUMINAMATH_CALUDE_tan_plus_4sin_20_deg_equals_sqrt3_l184_18459

theorem tan_plus_4sin_20_deg_equals_sqrt3 :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_4sin_20_deg_equals_sqrt3_l184_18459


namespace NUMINAMATH_CALUDE_natasha_dimes_l184_18462

theorem natasha_dimes : ∃ n : ℕ, 
  10 < n ∧ n < 100 ∧ 
  n % 3 = 1 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 1 ∧ 
  n = 61 := by
  sorry

end NUMINAMATH_CALUDE_natasha_dimes_l184_18462


namespace NUMINAMATH_CALUDE_profit_maximized_at_150_l184_18425

/-- The profit function for a company based on the number of machines -/
def profit (x : ℝ) : ℝ := -25 * x^2 + 7500 * x

/-- Theorem stating that the profit is maximized when x = 150 -/
theorem profit_maximized_at_150 :
  ∃ (x_max : ℝ), x_max = 150 ∧ ∀ (x : ℝ), profit x ≤ profit x_max :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_150_l184_18425


namespace NUMINAMATH_CALUDE_compound_carbon_atoms_l184_18476

/-- Represents a chemical compound --/
structure Compound where
  hydrogen : ℕ
  oxygen : ℕ
  molecular_weight : ℕ

/-- Atomic weights of elements --/
def atomic_weight : String → ℕ
  | "C" => 12
  | "H" => 1
  | "O" => 16
  | _ => 0

/-- Calculate the number of Carbon atoms in a compound --/
def carbon_atoms (c : Compound) : ℕ :=
  (c.molecular_weight - (c.hydrogen * atomic_weight "H" + c.oxygen * atomic_weight "O")) / atomic_weight "C"

/-- Theorem: The given compound has 4 Carbon atoms --/
theorem compound_carbon_atoms :
  let c : Compound := { hydrogen := 8, oxygen := 2, molecular_weight := 88 }
  carbon_atoms c = 4 := by
  sorry

end NUMINAMATH_CALUDE_compound_carbon_atoms_l184_18476


namespace NUMINAMATH_CALUDE_perimeter_of_larger_square_l184_18444

-- Define the side lengths of the small squares
def small_squares : List ℕ := [1, 1, 2, 3, 5, 8, 13]

-- Define the property that these squares form a larger square
def forms_larger_square (squares : List ℕ) : Prop := sorry

-- Define the perimeter calculation function
def calculate_perimeter (squares : List ℕ) : ℕ := sorry

-- Theorem statement
theorem perimeter_of_larger_square :
  forms_larger_square small_squares →
  calculate_perimeter small_squares = 68 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_larger_square_l184_18444


namespace NUMINAMATH_CALUDE_smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l184_18492

theorem smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11 :
  ∃ w : ℕ, w > 0 ∧ w % 13 = 0 ∧ (w + 3) % 11 = 0 ∧
  ∀ x : ℕ, x > 0 ∧ x % 13 = 0 ∧ (x + 3) % 11 = 0 → w ≤ x :=
by
  use 52
  sorry

end NUMINAMATH_CALUDE_smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l184_18492


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l184_18407

def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_of_M_and_N : M ∩ N = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l184_18407


namespace NUMINAMATH_CALUDE_probability_is_seventy_percent_l184_18448

/-- Represents a frequency interval with its lower bound, upper bound, and frequency count -/
structure FrequencyInterval where
  lower : ℝ
  upper : ℝ
  frequency : ℕ

/-- The sample data -/
def sample : List FrequencyInterval := [
  ⟨10, 20, 2⟩,
  ⟨20, 30, 3⟩,
  ⟨30, 40, 4⟩,
  ⟨40, 50, 5⟩,
  ⟨50, 60, 4⟩,
  ⟨60, 70, 2⟩
]

/-- The total sample size -/
def sampleSize : ℕ := 20

/-- The upper bound of the interval in question -/
def intervalUpperBound : ℝ := 50

/-- Calculates the probability of the sample data falling within (-∞, intervalUpperBound) -/
def probabilityWithinInterval (sample : List FrequencyInterval) (sampleSize : ℕ) (intervalUpperBound : ℝ) : ℚ :=
  sorry

/-- Theorem stating that the probability of the sample data falling within (-∞, 50) is 70% -/
theorem probability_is_seventy_percent :
  probabilityWithinInterval sample sampleSize intervalUpperBound = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_seventy_percent_l184_18448


namespace NUMINAMATH_CALUDE_smallest_value_absolute_equation_l184_18456

theorem smallest_value_absolute_equation :
  ∃ (x : ℝ), x = -5 ∧ |x - 4| = 9 ∧ ∀ (y : ℝ), |y - 4| = 9 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_absolute_equation_l184_18456


namespace NUMINAMATH_CALUDE_at_least_two_equations_have_solution_l184_18411

theorem at_least_two_equations_have_solution (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  let f₁ : ℝ → ℝ := λ x ↦ (x - b) * (x - c) - (x - a)
  let f₂ : ℝ → ℝ := λ x ↦ (x - c) * (x - a) - (x - b)
  let f₃ : ℝ → ℝ := λ x ↦ (x - a) * (x - b) - (x - c)
  ∃ (i j : Fin 3), i ≠ j ∧ (∃ x : ℝ, [f₁, f₂, f₃][i] x = 0) ∧ (∃ y : ℝ, [f₁, f₂, f₃][j] y = 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_equations_have_solution_l184_18411


namespace NUMINAMATH_CALUDE_intersecting_circles_values_l184_18454

/-- Two circles intersecting at points A and B, with centers on a line -/
structure IntersectingCircles where
  A : ℝ × ℝ
  B : ℝ × ℝ
  c : ℝ
  centers_on_line : ∀ (center : ℝ × ℝ), center.1 + center.2 + c = 0

/-- The theorem stating the values of m and c for the given configuration -/
theorem intersecting_circles_values (circles : IntersectingCircles) 
  (h1 : circles.A = (-1, 3))
  (h2 : circles.B.1 = -6) : 
  circles.B.2 = 3 ∧ circles.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_values_l184_18454


namespace NUMINAMATH_CALUDE_intersection_M_N_l184_18468

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {x | 2*x - x^2 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l184_18468


namespace NUMINAMATH_CALUDE_stratified_sample_size_l184_18496

/-- Represents the sample sizes of sedan models A, B, and C in a stratified sample. -/
structure SedanSample where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The production ratio of sedan models A, B, and C. -/
def productionRatio : Fin 3 → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 4

/-- The total of the production ratio values. -/
def ratioTotal : ℕ := (productionRatio 0) + (productionRatio 1) + (productionRatio 2)

/-- Theorem stating that if the number of model A sedans is 8 fewer than model B sedans
    in a stratified sample with the given production ratio, then the total sample size is 72. -/
theorem stratified_sample_size
  (sample : SedanSample)
  (h1 : sample.a + 8 = sample.b)
  (h2 : (sample.a : ℚ) / (sample.a + sample.b + sample.c : ℚ) = (productionRatio 0 : ℚ) / ratioTotal)
  (h3 : (sample.b : ℚ) / (sample.a + sample.b + sample.c : ℚ) = (productionRatio 1 : ℚ) / ratioTotal)
  (h4 : (sample.c : ℚ) / (sample.a + sample.b + sample.c : ℚ) = (productionRatio 2 : ℚ) / ratioTotal) :
  sample.a + sample.b + sample.c = 72 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_size_l184_18496


namespace NUMINAMATH_CALUDE_towel_price_problem_l184_18498

theorem towel_price_problem (x : ℚ) : 
  (3 * 100 + 5 * 150 + 2 * x) / 10 = 150 → x = 225 := by
  sorry

end NUMINAMATH_CALUDE_towel_price_problem_l184_18498


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_eight_l184_18455

/-- A geometric sequence with its sum function -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1
  sum_formula : ∀ n : ℕ, n > 0 → S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

/-- Theorem stating the conditions and conclusion about S₈ -/
theorem geometric_sequence_sum_eight
  (seq : GeometricSequence)
  (h1 : seq.S 4 = -5)
  (h2 : seq.S 6 = 21 * seq.S 2) :
  seq.S 8 = -85 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_eight_l184_18455


namespace NUMINAMATH_CALUDE_cos_equality_problem_l184_18450

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 270 → 
  Real.cos (n * π / 180) = Real.cos (962 * π / 180) →
  n = 118 := by sorry

end NUMINAMATH_CALUDE_cos_equality_problem_l184_18450


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l184_18438

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 6) ↔ ((-3 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 9)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l184_18438


namespace NUMINAMATH_CALUDE_goldfish_problem_l184_18469

theorem goldfish_problem (initial : ℕ) (final : ℕ) (birds : ℕ) (disease : ℕ) 
  (h1 : initial = 240)
  (h2 : final = 45)
  (h3 : birds = 15)
  (h4 : disease = 30) :
  let vanished := initial - final
  let heat := (vanished * 20) / 100
  let eaten := vanished - heat - disease - birds
  let raccoons := eaten / 3
  let cats := raccoons * 2
  cats = 64 ∧ raccoons = 32 ∧ heat = 39 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_problem_l184_18469


namespace NUMINAMATH_CALUDE_trader_markup_percentage_l184_18477

theorem trader_markup_percentage (discount : ℝ) (loss : ℝ) : 
  discount = 7.857142857142857 / 100 →
  loss = 1 / 100 →
  ∃ (markup : ℝ), 
    (1 + markup) * (1 - discount) = 1 - loss ∧ 
    abs (markup - 7.4285714285714 / 100) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_trader_markup_percentage_l184_18477


namespace NUMINAMATH_CALUDE_football_equipment_cost_l184_18470

/-- Given the cost of football equipment, prove the total cost relation -/
theorem football_equipment_cost (x : ℝ) 
  (h1 : x + x = 2 * x)           -- Shorts + T-shirt = 2x
  (h2 : x + 4 * x = 5 * x)       -- Shorts + boots = 5x
  (h3 : x + 2 * x = 3 * x)       -- Shorts + shin guards = 3x
  : x + x + 4 * x + 2 * x = 8 * x := by
  sorry


end NUMINAMATH_CALUDE_football_equipment_cost_l184_18470


namespace NUMINAMATH_CALUDE_homework_scenarios_count_l184_18484

/-- The number of subjects available for homework -/
def num_subjects : ℕ := 4

/-- The number of students doing homework -/
def num_students : ℕ := 3

/-- The number of possible scenarios for homework assignment -/
def num_scenarios : ℕ := num_subjects ^ num_students

/-- Theorem stating that the number of possible scenarios is 64 -/
theorem homework_scenarios_count : num_scenarios = 64 := by
  sorry

end NUMINAMATH_CALUDE_homework_scenarios_count_l184_18484


namespace NUMINAMATH_CALUDE_sum_of_coordinates_l184_18402

def point_C : ℝ × ℝ := (3, -2)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def point_D : ℝ × ℝ := reflect_over_y_axis point_C

theorem sum_of_coordinates : 
  point_C.1 + point_C.2 + point_D.1 + point_D.2 = -4 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_l184_18402


namespace NUMINAMATH_CALUDE_max_value_of_squared_differences_l184_18417

theorem max_value_of_squared_differences (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 10) : 
  (∃ (x : ℝ), (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ x) ∧ 
  (∀ (y : ℝ), (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ y → 40 ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_squared_differences_l184_18417


namespace NUMINAMATH_CALUDE_remainder_problem_l184_18430

theorem remainder_problem (x : ℤ) (h : x % 61 = 24) : x % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l184_18430


namespace NUMINAMATH_CALUDE_compound_vs_simple_interest_amount_l184_18414

/-- The amount of money (in rupees) that results in a difference of 8.000000000000227
    between 8% compound interest and 4% simple interest over 2 years -/
theorem compound_vs_simple_interest_amount : ℝ := by
  -- Define the compound interest rate
  let compound_rate : ℝ := 0.08
  -- Define the simple interest rate
  let simple_rate : ℝ := 0.04
  -- Define the time period in years
  let time : ℝ := 2
  -- Define the difference between compound and simple interest amounts
  let difference : ℝ := 8.000000000000227

  -- Define the function for compound interest
  let compound_interest (p : ℝ) : ℝ := p * (1 + compound_rate) ^ time

  -- Define the function for simple interest
  let simple_interest (p : ℝ) : ℝ := p * (1 + simple_rate * time)

  -- The amount p that satisfies the condition
  let p : ℝ := difference / (compound_interest 1 - simple_interest 1)

  -- Assert that p is approximately equal to 92.59
  sorry


end NUMINAMATH_CALUDE_compound_vs_simple_interest_amount_l184_18414


namespace NUMINAMATH_CALUDE_barber_total_loss_l184_18432

/-- Represents the barber's financial transactions and losses --/
def barber_loss : ℕ → Prop :=
  fun loss =>
    ∃ (haircut_cost change_given flower_shop_exchange bakery_exchange counterfeit_50 counterfeit_10 replacement_50 replacement_10 : ℕ),
      haircut_cost = 25 ∧
      change_given = 25 ∧
      flower_shop_exchange = 50 ∧
      bakery_exchange = 10 ∧
      counterfeit_50 = 50 ∧
      counterfeit_10 = 10 ∧
      replacement_50 = 50 ∧
      replacement_10 = 10 ∧
      loss = haircut_cost + change_given + counterfeit_50 + counterfeit_10 + replacement_50 + replacement_10 - flower_shop_exchange

theorem barber_total_loss :
  barber_loss 120 :=
sorry

end NUMINAMATH_CALUDE_barber_total_loss_l184_18432


namespace NUMINAMATH_CALUDE_integer_difference_l184_18408

theorem integer_difference (x y : ℤ) (h1 : x < y) (h2 : x + y = -9) (h3 : x = -5) (h4 : y = -4) : y - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_difference_l184_18408


namespace NUMINAMATH_CALUDE_unique_solution_of_equation_l184_18413

theorem unique_solution_of_equation :
  ∃! (x y z : ℝ), x^2 + 5*y^2 + 5*z^2 - 4*x*z - 2*y - 4*y*z + 1 = 0 ∧ x = 4 ∧ y = 1 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_of_equation_l184_18413


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_series_l184_18415

/-- Sum of an arithmetic series with given parameters -/
theorem sum_of_arithmetic_series : 
  ∀ (a l d : ℤ) (n : ℕ+),
  a = -48 →
  d = 4 →
  l = 0 →
  a + (n - 1 : ℤ) * d = l →
  (n : ℤ) * (a + l) / 2 = -312 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_series_l184_18415


namespace NUMINAMATH_CALUDE_simplify_fraction_l184_18401

theorem simplify_fraction (x y z : ℚ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  15 * x^2 * z^3 / (9 * x * y * z^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l184_18401


namespace NUMINAMATH_CALUDE_expression_value_l184_18463

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 4)  -- The absolute value of m is 4
  : m + c * d + (a + b) / m = 5 ∨ m + c * d + (a + b) / m = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l184_18463


namespace NUMINAMATH_CALUDE_complement_A_inter_B_when_m_3_A_inter_B_empty_iff_l184_18490

/-- The set A defined as {x | -1 ≤ x < 4} -/
def A : Set ℝ := {x | -1 ≤ x ∧ x < 4}

/-- The set B defined as {x | m ≤ x ≤ m+2} for a real number m -/
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2}

/-- Part 1: The complement of A ∩ B when m = 3 -/
theorem complement_A_inter_B_when_m_3 :
  (A ∩ B 3)ᶜ = {x | x < 3 ∨ x ≥ 4} := by sorry

/-- Part 2: Characterization of m when A ∩ B is empty -/
theorem A_inter_B_empty_iff (m : ℝ) :
  A ∩ B m = ∅ ↔ m < -3 ∨ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_when_m_3_A_inter_B_empty_iff_l184_18490


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l184_18453

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 27) 
  (h3 : S = a / (1 - r)) : 
  a = 36 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l184_18453


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l184_18440

/-- Calculates the profit percentage when selling n articles at the cost price of m articles -/
def profit_percentage (n m : ℕ) : ℚ :=
  (m - n) / n * 100

/-- Theorem: When a shopkeeper sells 10 articles at the cost price of 12 articles, the profit percentage is 20% -/
theorem shopkeeper_profit : profit_percentage 10 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l184_18440


namespace NUMINAMATH_CALUDE_keiko_walking_speed_l184_18486

/-- Keiko's walking speed around two rectangular tracks with semicircular ends -/
theorem keiko_walking_speed :
  ∀ (speed : ℝ) (width_A width_B time_diff_A time_diff_B : ℝ),
  width_A = 4 →
  width_B = 8 →
  time_diff_A = 48 →
  time_diff_B = 72 →
  (2 * π * width_A) / speed = time_diff_A →
  (2 * π * width_B) / speed = time_diff_B →
  speed = 2 * π / 5 :=
by sorry

end NUMINAMATH_CALUDE_keiko_walking_speed_l184_18486


namespace NUMINAMATH_CALUDE_sequence_not_in_interval_l184_18433

/-- Given an infinite sequence of real numbers {aₙ} where aₙ₊₁ = √(aₙ² + aₙ - 1) for all n ≥ 1,
    prove that a₁ ∉ (-2, 1). -/
theorem sequence_not_in_interval (a : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = Real.sqrt ((a n)^2 + a n - 1)) : 
    a 1 ∉ Set.Ioo (-2 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_not_in_interval_l184_18433


namespace NUMINAMATH_CALUDE_number_count_from_average_correction_l184_18465

/-- Given an initial average and a corrected average after fixing a misread number,
    calculate the number of numbers in the original set. -/
theorem number_count_from_average_correction (initial_avg : ℚ) (corrected_avg : ℚ) 
    (misread : ℚ) (correct : ℚ) (h1 : initial_avg = 16) (h2 : corrected_avg = 19) 
    (h3 : misread = 25) (h4 : correct = 55) : 
    ∃ n : ℕ, (n : ℚ) * initial_avg + misread = (n : ℚ) * corrected_avg + correct ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_count_from_average_correction_l184_18465


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_20_l184_18487

def digit_sum (n : Nat) : Nat :=
  Nat.rec 0 (fun n sum => sum + n % 10) n

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_digit_sum_20 :
  ∀ n : Nat, n < 389 → ¬(is_prime n ∧ digit_sum n = 20) ∧
  is_prime 389 ∧ digit_sum 389 = 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_20_l184_18487
