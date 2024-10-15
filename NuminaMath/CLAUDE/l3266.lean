import Mathlib

namespace NUMINAMATH_CALUDE_fill_time_correct_l3266_326639

/-- The time in seconds for eight faucets to fill a 30-gallon tub, given that four faucets can fill a 120-gallon tub in 8 minutes -/
def fill_time : ℝ := 60

/-- The volume of the large tub in gallons -/
def large_tub_volume : ℝ := 120

/-- The volume of the small tub in gallons -/
def small_tub_volume : ℝ := 30

/-- The time in minutes for four faucets to fill the large tub -/
def large_tub_fill_time : ℝ := 8

/-- The number of faucets used to fill the large tub -/
def large_tub_faucets : ℕ := 4

/-- The number of faucets used to fill the small tub -/
def small_tub_faucets : ℕ := 8

/-- Conversion factor from minutes to seconds -/
def minutes_to_seconds : ℝ := 60

theorem fill_time_correct : fill_time = 
  (small_tub_volume / large_tub_volume) * 
  (large_tub_faucets / small_tub_faucets) * 
  large_tub_fill_time * 
  minutes_to_seconds := by
  sorry

end NUMINAMATH_CALUDE_fill_time_correct_l3266_326639


namespace NUMINAMATH_CALUDE_cube_sum_divisible_implies_product_divisible_l3266_326660

theorem cube_sum_divisible_implies_product_divisible (a b c : ℤ) :
  7 ∣ (a^3 + b^3 + c^3) → 7 ∣ (a * b * c) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_divisible_implies_product_divisible_l3266_326660


namespace NUMINAMATH_CALUDE_car_speed_problem_l3266_326655

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
def second_hour_speed (first_hour_speed average_speed : ℝ) : ℝ :=
  2 * average_speed - first_hour_speed

/-- Theorem stating that if a car travels at 90 km/h for the first hour
    and has an average speed of 60 km/h over two hours,
    its speed in the second hour must be 30 km/h. -/
theorem car_speed_problem :
  second_hour_speed 90 60 = 30 := by
  sorry

#eval second_hour_speed 90 60

end NUMINAMATH_CALUDE_car_speed_problem_l3266_326655


namespace NUMINAMATH_CALUDE_art_club_collection_l3266_326632

/-- Calculates the total number of artworks collected by an art club over multiple school years. -/
def total_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (years : ℕ) : ℕ :=
  students * artworks_per_student_per_quarter * quarters_per_year * years

/-- Proves that an art club with 15 students, each making 2 artworks per quarter, 
    collects 240 artworks in 2 school years with 4 quarters per year. -/
theorem art_club_collection : total_artworks 15 2 4 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_art_club_collection_l3266_326632


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_for_school_scenario_l3266_326693

/-- Represents the different blood types --/
inductive BloodType
| O
| A
| B
| AB

/-- Represents the available sampling methods --/
inductive SamplingMethod
| Random
| Systematic
| Stratified

/-- Structure representing the school scenario --/
structure SchoolScenario where
  total_students : Nat
  blood_type_distribution : BloodType → Nat
  sample_size_blood_study : Nat
  soccer_team_size : Nat
  sample_size_soccer_study : Nat

/-- Determines the optimal sampling method for a given scenario and study type --/
def optimal_sampling_method (scenario : SchoolScenario) (is_blood_study : Bool) : SamplingMethod :=
  if is_blood_study then SamplingMethod.Stratified else SamplingMethod.Random

/-- Theorem stating the optimal sampling methods for the given school scenario --/
theorem optimal_sampling_methods_for_school_scenario 
  (scenario : SchoolScenario)
  (h1 : scenario.total_students = 500)
  (h2 : scenario.blood_type_distribution BloodType.O = 200)
  (h3 : scenario.blood_type_distribution BloodType.A = 125)
  (h4 : scenario.blood_type_distribution BloodType.B = 125)
  (h5 : scenario.blood_type_distribution BloodType.AB = 50)
  (h6 : scenario.sample_size_blood_study = 20)
  (h7 : scenario.soccer_team_size = 11)
  (h8 : scenario.sample_size_soccer_study = 2) :
  (optimal_sampling_method scenario true = SamplingMethod.Stratified) ∧
  (optimal_sampling_method scenario false = SamplingMethod.Random) :=
sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_for_school_scenario_l3266_326693


namespace NUMINAMATH_CALUDE_chessboard_tromino_coverage_l3266_326621

/-- Represents a chessboard with alternating colors and black corners -/
structure Chessboard (n : ℕ) :=
  (is_odd : n % 2 = 1)
  (ge_seven : n ≥ 7)

/-- Calculates the number of black squares on the chessboard -/
def black_squares (board : Chessboard n) : ℕ :=
  (n^2 + 1) / 2

/-- Calculates the minimum number of trominos needed -/
def min_trominos (board : Chessboard n) : ℕ :=
  (n^2 + 1) / 6

theorem chessboard_tromino_coverage (n : ℕ) (board : Chessboard n) :
  (black_squares board) % 3 = 0 ∧
  min_trominos board = (n^2 + 1) / 6 :=
sorry

end NUMINAMATH_CALUDE_chessboard_tromino_coverage_l3266_326621


namespace NUMINAMATH_CALUDE_center_is_ten_l3266_326696

/-- Represents a 4x4 array of integers -/
def Array4x4 := Fin 4 → Fin 4 → ℕ

/-- Checks if two positions in the array share an edge -/
def share_edge (p q : Fin 4 × Fin 4) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Defines a valid array according to the problem conditions -/
def valid_array (a : Array4x4) : Prop :=
  (∀ n : Fin 16, ∃ i j : Fin 4, a i j = n.val + 1) ∧
  (∀ n : Fin 15, ∃ i j k l : Fin 4, 
    a i j = n.val + 1 ∧ 
    a k l = n.val + 2 ∧ 
    share_edge (i, j) (k, l)) ∧
  (a 0 0 + a 0 3 + a 3 0 + a 3 3 = 34)

/-- The main theorem to prove -/
theorem center_is_ten (a : Array4x4) (h : valid_array a) : 
  a 1 1 = 10 ∨ a 1 2 = 10 ∨ a 2 1 = 10 ∨ a 2 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_center_is_ten_l3266_326696


namespace NUMINAMATH_CALUDE_apple_distribution_l3266_326641

theorem apple_distribution (total_apples : ℕ) (additional_people : ℕ) (apple_reduction : ℕ) :
  total_apples = 10000 →
  additional_people = 100 →
  apple_reduction = 15 →
  ∃ X : ℕ,
    (X * (total_apples / X) = total_apples) ∧
    ((X + additional_people) * (total_apples / X - apple_reduction) = total_apples) ∧
    X = 213 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3266_326641


namespace NUMINAMATH_CALUDE_AgOH_formation_l3266_326653

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  ratio : ℚ

-- Define the initial conditions
def initial_AgNO3 : ℚ := 3
def initial_NaOH : ℚ := 3

-- Define the reaction
def silver_hydroxide_reaction : Reaction := {
  reactant1 := "AgNO3"
  reactant2 := "NaOH"
  product1 := "AgOH"
  product2 := "NaNO3"
  ratio := 1
}

-- Theorem statement
theorem AgOH_formation (r : Reaction) (h1 : r = silver_hydroxide_reaction) 
  (h2 : initial_AgNO3 = initial_NaOH) : 
  let moles_AgOH := min initial_AgNO3 initial_NaOH
  moles_AgOH = 3 := by sorry

end NUMINAMATH_CALUDE_AgOH_formation_l3266_326653


namespace NUMINAMATH_CALUDE_last_car_speed_l3266_326640

theorem last_car_speed (n : ℕ) (first_speed last_speed : ℕ) : 
  n = 31 ∧ 
  first_speed = 61 ∧ 
  last_speed = 91 ∧ 
  last_speed - first_speed + 1 = n → 
  first_speed + ((n + 1) / 2 - 1) = 76 :=
by sorry

end NUMINAMATH_CALUDE_last_car_speed_l3266_326640


namespace NUMINAMATH_CALUDE_pool_filling_time_l3266_326634

theorem pool_filling_time (a b c d : ℝ) 
  (h1 : a + b = 1/2)
  (h2 : b + c = 1/3)
  (h3 : c + d = 1/4) :
  a + d = 5/12 :=
sorry

end NUMINAMATH_CALUDE_pool_filling_time_l3266_326634


namespace NUMINAMATH_CALUDE_complex_power_four_l3266_326643

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem: (1+i)^4 = -4 -/
theorem complex_power_four : (1 + i)^4 = -4 := by sorry

end NUMINAMATH_CALUDE_complex_power_four_l3266_326643


namespace NUMINAMATH_CALUDE_range_of_x_l3266_326658

def p (x : ℝ) : Prop := x^2 - 5*x + 6 ≥ 0

def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (h1 : p x ∨ q x) (h2 : ¬q x) :
  x ≤ 0 ∨ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l3266_326658


namespace NUMINAMATH_CALUDE_middle_guard_hours_l3266_326691

theorem middle_guard_hours (total_hours : ℕ) (num_guards : ℕ) (first_guard_hours : ℕ) (last_guard_hours : ℕ) :
  total_hours = 9 ∧ num_guards = 4 ∧ first_guard_hours = 3 ∧ last_guard_hours = 2 →
  (total_hours - first_guard_hours - last_guard_hours) / (num_guards - 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_middle_guard_hours_l3266_326691


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l3266_326679

/-- Two lines are parallel but not coincident -/
def parallel_not_coincident (a : ℝ) : Prop :=
  (a * 3 - 3 * (a - 1) = 0) ∧ (a * (a - 7) - 3 * (3 * a) ≠ 0)

/-- The condition a = 3 or a = -2 -/
def condition (a : ℝ) : Prop := a = 3 ∨ a = -2

theorem parallel_lines_condition :
  (∀ a : ℝ, parallel_not_coincident a → condition a) ∧
  ¬(∀ a : ℝ, condition a → parallel_not_coincident a) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l3266_326679


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_integers_l3266_326604

theorem sum_of_five_consecutive_integers (n : ℤ) : 
  (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 5 * n + 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_integers_l3266_326604


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3266_326615

theorem equation_solution_exists : ∃ x : ℝ, -x^3 + 555^3 = x^2 - x * 555 + 555^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3266_326615


namespace NUMINAMATH_CALUDE_parabola_properties_l3266_326699

/-- A parabola is defined by the equation y = -x^2 + 1 --/
def parabola (x : ℝ) : ℝ := -x^2 + 1

theorem parabola_properties :
  (∀ x : ℝ, parabola x ≤ parabola 0) ∧ 
  parabola 0 = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3266_326699


namespace NUMINAMATH_CALUDE_equation_solution_l3266_326603

theorem equation_solution :
  ∃! (a b c d : ℚ), 
    a^2 + b^2 + c^2 + d^2 - a*b - b*c - c*d - d + 2/5 = 0 ∧
    a = 1/5 ∧ b = 2/5 ∧ c = 3/5 ∧ d = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3266_326603


namespace NUMINAMATH_CALUDE_reciprocal_opposite_theorem_l3266_326683

theorem reciprocal_opposite_theorem (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  : (c + d)^2 - a * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_opposite_theorem_l3266_326683


namespace NUMINAMATH_CALUDE_f_properties_l3266_326684

-- Define the function f
def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

-- Theorem statement
theorem f_properties :
  (∀ x, f x 0 0 = -f (-x) 0 0) ∧
  (∀ x, f x 0 (0 : ℝ) = 0 → x = 0) ∧
  (∀ x, f (x - 0) b c = f (-x - 0) b c + 2 * c) ∧
  (∃ b c, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3266_326684


namespace NUMINAMATH_CALUDE_multiply_to_target_l3266_326695

theorem multiply_to_target (x : ℕ) : x * 586645 = 5865863355 → x = 9999 := by
  sorry

end NUMINAMATH_CALUDE_multiply_to_target_l3266_326695


namespace NUMINAMATH_CALUDE_floor_painting_two_solutions_l3266_326654

/-- The number of ordered pairs (a, b) satisfying the floor painting conditions -/
def floor_painting_solutions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let a := p.1
    let b := p.2
    b > a ∧ (a - 4) * (b - 4) = 2 * a * b / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- There are exactly two solutions to the floor painting problem -/
theorem floor_painting_two_solutions : floor_painting_solutions = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_two_solutions_l3266_326654


namespace NUMINAMATH_CALUDE_real_estate_investment_l3266_326602

theorem real_estate_investment
  (total_investment : ℝ)
  (real_estate_ratio : ℝ)
  (h1 : total_investment = 250000)
  (h2 : real_estate_ratio = 3)
  : real_estate_ratio * (total_investment / (1 + real_estate_ratio)) = 187500 :=
by
  sorry

end NUMINAMATH_CALUDE_real_estate_investment_l3266_326602


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l3266_326635

/-- The number of ways to seat n people in a row -/
def totalArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to seat n people in a row where m specific people sit together -/
def arrangementsWithGroupTogether (n m : ℕ) : ℕ := 
  (n - m + 1).factorial * m.factorial

/-- The number of ways to seat 10 people in a row where 4 specific people cannot sit in 4 consecutive seats -/
def seatingArrangements : ℕ := 
  totalArrangements 10 - arrangementsWithGroupTogether 10 4

theorem seating_arrangements_count : seatingArrangements = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l3266_326635


namespace NUMINAMATH_CALUDE_odd_integer_divisibility_l3266_326613

theorem odd_integer_divisibility (n : Int) (h : Odd n) :
  ∀ k : Nat, 2 ∣ k * (n - k) :=
sorry

end NUMINAMATH_CALUDE_odd_integer_divisibility_l3266_326613


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3266_326636

theorem fraction_product_simplification :
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3266_326636


namespace NUMINAMATH_CALUDE_files_remaining_l3266_326620

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 13)
  (h2 : video_files = 30)
  (h3 : deleted_files = 10) :
  music_files + video_files - deleted_files = 33 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l3266_326620


namespace NUMINAMATH_CALUDE_find_k_l3266_326609

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - k * x + 7

-- State the theorem
theorem find_k : ∃ k : ℝ, f 5 - g k 5 = 40 ∧ k = 1.4 := by sorry

end NUMINAMATH_CALUDE_find_k_l3266_326609


namespace NUMINAMATH_CALUDE_ratio_problem_l3266_326629

theorem ratio_problem (a b c : ℝ) 
  (h1 : b / c = 1 / 5)
  (h2 : a / c = 1 / 7.5) :
  a / b = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3266_326629


namespace NUMINAMATH_CALUDE_exists_valid_strategy_l3266_326645

/-- Represents the problem of the father and two sons visiting their grandmother --/
structure VisitProblem where
  distance : ℝ
  scooter_speed_alone : ℝ
  scooter_speed_with_passenger : ℝ
  walking_speed : ℝ

/-- Defines the specific problem instance --/
def problem : VisitProblem :=
  { distance := 33
  , scooter_speed_alone := 25
  , scooter_speed_with_passenger := 20
  , walking_speed := 5
  }

/-- Represents a solution strategy for the visit problem --/
structure Strategy where
  (p : VisitProblem)
  travel_time : ℝ

/-- Predicate to check if a strategy is valid --/
def is_valid_strategy (s : Strategy) : Prop :=
  s.travel_time ≤ 3 ∧
  ∃ (t1 t2 t3 : ℝ),
    t1 ≤ s.travel_time ∧
    t2 ≤ s.travel_time ∧
    t3 ≤ s.travel_time ∧
    s.p.distance / s.p.walking_speed ≤ t1 ∧
    s.p.distance / s.p.walking_speed ≤ t2 ∧
    s.p.distance / s.p.scooter_speed_alone ≤ t3

/-- Theorem stating that there exists a valid strategy for the given problem --/
theorem exists_valid_strategy :
  ∃ (s : Strategy), s.p = problem ∧ is_valid_strategy s :=
sorry


end NUMINAMATH_CALUDE_exists_valid_strategy_l3266_326645


namespace NUMINAMATH_CALUDE_expression_evaluation_l3266_326622

theorem expression_evaluation : (-4)^6 / 4^4 + 2^5 * 5 - 7^2 = 127 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3266_326622


namespace NUMINAMATH_CALUDE_douglas_county_y_percentage_l3266_326664

-- Define the ratio of voters in county X to county Y
def voter_ratio : ℚ := 2 / 1

-- Define the percentage of votes Douglas won in both counties combined
def total_vote_percentage : ℚ := 60 / 100

-- Define the percentage of votes Douglas won in county X
def county_x_percentage : ℚ := 72 / 100

-- Theorem to prove
theorem douglas_county_y_percentage :
  let total_voters := 3 -- represents the sum of parts in the ratio (2 + 1)
  let county_x_voters := 2 -- represents the larger part of the ratio
  let county_y_voters := 1 -- represents the smaller part of the ratio
  let total_douglas_votes := total_vote_percentage * total_voters
  let county_x_douglas_votes := county_x_percentage * county_x_voters
  let county_y_douglas_votes := total_douglas_votes - county_x_douglas_votes
  county_y_douglas_votes / county_y_voters = 36 / 100 :=
by sorry

end NUMINAMATH_CALUDE_douglas_county_y_percentage_l3266_326664


namespace NUMINAMATH_CALUDE_desired_average_sale_l3266_326617

theorem desired_average_sale (sales : List ℕ) (desired_sixth : ℕ) : 
  sales = [6235, 6927, 6855, 7230, 6562] → 
  desired_sixth = 5191 → 
  (sales.sum + desired_sixth) / 6 = 6500 := by
  sorry

end NUMINAMATH_CALUDE_desired_average_sale_l3266_326617


namespace NUMINAMATH_CALUDE_fn_solution_l3266_326625

-- Define the sequence of functions
def f : ℕ → ℝ → ℝ
| 0, x => |x|
| n + 1, x => |f n x - 2|

-- Define the set of solutions
def solution_set (n : ℕ) : Set ℝ :=
  {x | ∃ k : ℤ, x = 2*k + 1 ∨ x = -(2*k + 1) ∧ |2*k + 1| ≤ 2*n + 1}

-- Theorem statement
theorem fn_solution (n : ℕ+) :
  {x : ℝ | f n x = 1} = solution_set n :=
sorry

end NUMINAMATH_CALUDE_fn_solution_l3266_326625


namespace NUMINAMATH_CALUDE_competitive_examination_selection_l3266_326682

theorem competitive_examination_selection (total_candidates : ℕ) 
  (selection_rate_A : ℚ) (selection_rate_B : ℚ) : 
  total_candidates = 8000 →
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B * total_candidates : ℚ) - (selection_rate_A * total_candidates : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_competitive_examination_selection_l3266_326682


namespace NUMINAMATH_CALUDE_initial_birds_on_fence_l3266_326662

theorem initial_birds_on_fence (initial_birds additional_birds total_birds : ℕ) :
  initial_birds + additional_birds = total_birds →
  additional_birds = 4 →
  total_birds = 6 →
  initial_birds = 2 := by
sorry

end NUMINAMATH_CALUDE_initial_birds_on_fence_l3266_326662


namespace NUMINAMATH_CALUDE_printer_cost_l3266_326689

/-- The cost of a printer given the conditions of the merchant's purchase. -/
theorem printer_cost (total_cost : ℕ) (keyboard_cost : ℕ) (num_keyboards : ℕ) (num_printers : ℕ)
  (h1 : total_cost = 2050)
  (h2 : keyboard_cost = 20)
  (h3 : num_keyboards = 15)
  (h4 : num_printers = 25) :
  (total_cost - num_keyboards * keyboard_cost) / num_printers = 70 := by
  sorry

end NUMINAMATH_CALUDE_printer_cost_l3266_326689


namespace NUMINAMATH_CALUDE_josiah_cookie_expense_l3266_326649

/-- The amount Josiah spent on cookies in March -/
def cookie_expense : ℕ → ℕ → ℕ → ℕ
| cookies_per_day, cookie_cost, days_in_march =>
  cookies_per_day * cookie_cost * days_in_march

/-- Theorem: Josiah spent 992 dollars on cookies in March -/
theorem josiah_cookie_expense :
  cookie_expense 2 16 31 = 992 := by
  sorry

end NUMINAMATH_CALUDE_josiah_cookie_expense_l3266_326649


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3266_326647

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 19 * x^3 - 84 * y^2 = 1984 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3266_326647


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3266_326668

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  ∃ (a : ℝ), (a = 1 → |a| = 1) ∧ (|a| = 1 → ¬(a = 1 ↔ |a| = 1)) := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3266_326668


namespace NUMINAMATH_CALUDE_partnership_profit_theorem_l3266_326685

/-- Represents an investment in a partnership business -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the total profit of a partnership business -/
def calculateTotalProfit (investments : List Investment) (cProfit : ℕ) : ℕ :=
  let totalCapitalMonths := investments.foldl (fun acc inv => acc + inv.amount * inv.duration) 0
  let cCapitalMonths := (investments.find? (fun inv => inv.amount = 6000 ∧ inv.duration = 6)).map (fun inv => inv.amount * inv.duration)
  match cCapitalMonths with
  | some cm => totalCapitalMonths * cProfit / cm
  | none => 0

theorem partnership_profit_theorem (investments : List Investment) (cProfit : ℕ) :
  investments = [
    ⟨8000, 12⟩,  -- A's investment
    ⟨4000, 8⟩,   -- B's investment
    ⟨6000, 6⟩,   -- C's investment
    ⟨10000, 9⟩   -- D's investment
  ] ∧ cProfit = 36000 →
  calculateTotalProfit investments cProfit = 254000 := by
  sorry

#eval calculateTotalProfit [⟨8000, 12⟩, ⟨4000, 8⟩, ⟨6000, 6⟩, ⟨10000, 9⟩] 36000

end NUMINAMATH_CALUDE_partnership_profit_theorem_l3266_326685


namespace NUMINAMATH_CALUDE_system_solution_l3266_326670

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  3 * x + y = 5 ∧ x + 3 * y = 7

-- State the theorem
theorem system_solution :
  ∃! (x y : ℝ), system x y ∧ x = 1 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3266_326670


namespace NUMINAMATH_CALUDE_weight_replacement_l3266_326638

theorem weight_replacement (initial_total : ℝ) (replaced_weight : ℝ) :
  (∃ (average_increase : ℝ),
    average_increase = 1.5 ∧
    4 * (initial_total / 4 + average_increase) = initial_total - replaced_weight + 71) →
  replaced_weight = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l3266_326638


namespace NUMINAMATH_CALUDE_sally_walked_2540_miles_l3266_326688

/-- Calculates the total miles walked given pedometer resets, final reading, steps per mile, and additional steps --/
def total_miles_walked (resets : ℕ) (final_reading : ℕ) (steps_per_mile : ℕ) (additional_steps : ℕ) : ℕ :=
  let total_steps := resets * 100000 + final_reading + additional_steps
  (total_steps + steps_per_mile - 1) / steps_per_mile

/-- Theorem stating that Sally walked 2540 miles during the year --/
theorem sally_walked_2540_miles :
  total_miles_walked 50 30000 2000 50000 = 2540 := by
  sorry

end NUMINAMATH_CALUDE_sally_walked_2540_miles_l3266_326688


namespace NUMINAMATH_CALUDE_regression_coefficient_nonzero_l3266_326678

/-- Represents a regression line for two variables with a linear relationship -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- Theorem: The regression coefficient b in a regression line y = a + bx 
    for two variables with a linear relationship cannot be equal to 0 -/
theorem regression_coefficient_nonzero (line : RegressionLine) : 
  line.b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_regression_coefficient_nonzero_l3266_326678


namespace NUMINAMATH_CALUDE_gcd_288_123_l3266_326694

theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_288_123_l3266_326694


namespace NUMINAMATH_CALUDE_parking_spaces_available_l3266_326663

theorem parking_spaces_available (front_spaces back_spaces parked_cars : ℕ) :
  front_spaces = 52 →
  back_spaces = 38 →
  parked_cars = 39 →
  (front_spaces + back_spaces) - (parked_cars + back_spaces / 2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_parking_spaces_available_l3266_326663


namespace NUMINAMATH_CALUDE_company_employees_l3266_326624

/-- If a company has 15% more employees in December than in January,
    and it has 500 employees in December, then it had 435 employees in January. -/
theorem company_employees (january_employees : ℕ) (december_employees : ℕ) :
  december_employees = 500 →
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 435 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l3266_326624


namespace NUMINAMATH_CALUDE_existence_and_not_forall_l3266_326633

theorem existence_and_not_forall :
  (∃ x₀ : ℝ, x₀ - 2 > 0) ∧ ¬(∀ x : ℝ, 2^x > x^2) := by
  sorry

end NUMINAMATH_CALUDE_existence_and_not_forall_l3266_326633


namespace NUMINAMATH_CALUDE_total_legs_in_park_l3266_326612

/-- The total number of legs in a park with various animals, some with missing legs -/
def total_legs : ℕ :=
  let num_dogs : ℕ := 109
  let num_cats : ℕ := 37
  let num_birds : ℕ := 52
  let num_spiders : ℕ := 19
  let dogs_missing_legs : ℕ := 4
  let cats_missing_legs : ℕ := 3
  let spiders_missing_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let cat_legs : ℕ := 4
  let bird_legs : ℕ := 2
  let spider_legs : ℕ := 8
  (num_dogs * dog_legs - dogs_missing_legs) +
  (num_cats * cat_legs - cats_missing_legs) +
  (num_birds * bird_legs) +
  (num_spiders * spider_legs - 2 * spiders_missing_legs)

theorem total_legs_in_park : total_legs = 829 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_park_l3266_326612


namespace NUMINAMATH_CALUDE_right_triangle_area_l3266_326669

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) : (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3266_326669


namespace NUMINAMATH_CALUDE_favorite_numbers_sum_l3266_326687

/-- Given that Glory's favorite number is 450 and Misty's favorite number is 3 times smaller than Glory's,
    prove that the sum of their favorite numbers is 600. -/
theorem favorite_numbers_sum (glory_number : ℕ) (misty_number : ℕ)
    (h1 : glory_number = 450)
    (h2 : misty_number * 3 = glory_number) :
    misty_number + glory_number = 600 := by
  sorry

end NUMINAMATH_CALUDE_favorite_numbers_sum_l3266_326687


namespace NUMINAMATH_CALUDE_derek_dogs_at_16_l3266_326676

/-- Represents Derek's possessions at different ages --/
structure DereksPossessions where
  dogs_at_6 : ℕ
  cars_at_6 : ℕ
  dogs_at_16 : ℕ
  cars_at_16 : ℕ

/-- Theorem stating the number of dogs Derek has at age 16 --/
theorem derek_dogs_at_16 (d : DereksPossessions) 
  (h1 : d.dogs_at_6 = 3 * d.cars_at_6)
  (h2 : d.dogs_at_6 = 90)
  (h3 : d.cars_at_16 = d.cars_at_6 + 210)
  (h4 : d.cars_at_16 = 2 * d.dogs_at_16) :
  d.dogs_at_16 = 120 := by
  sorry

#check derek_dogs_at_16

end NUMINAMATH_CALUDE_derek_dogs_at_16_l3266_326676


namespace NUMINAMATH_CALUDE_cubic_function_extreme_value_l3266_326623

/-- Given a cubic function f(x) = ax³ + bx + c that reaches an extreme value of c-6 at x=2,
    prove that a = 3/8 and b = -9/2 -/
theorem cubic_function_extreme_value (a b c : ℝ) :
  (∀ x, ∃ f : ℝ → ℝ, f x = a * x^3 + b * x + c) →
  (∃ f : ℝ → ℝ, f 2 = c - 6 ∧ ∀ x, f x ≤ f 2 ∨ f x ≥ f 2) →
  a = 3/8 ∧ b = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extreme_value_l3266_326623


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3266_326672

-- Define the properties of the parallelogram
def parallelogram_area : ℝ := 200
def parallelogram_height : ℝ := 20

-- Theorem statement
theorem parallelogram_base_length :
  ∃ (base : ℝ), base * parallelogram_height = parallelogram_area ∧ base = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3266_326672


namespace NUMINAMATH_CALUDE_sequence_inequality_l3266_326606

theorem sequence_inequality (a : ℕ → ℕ) 
  (h1 : a 2 > a 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 2) = 3 * a (n + 1) - 2 * a n) : 
  a 2021 > 2^2019 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3266_326606


namespace NUMINAMATH_CALUDE_work_completion_time_l3266_326650

/-- The number of days it takes to complete a task when two people work together -/
def combined_work_time (john_rate : ℚ) (rose_rate : ℚ) : ℚ :=
  1 / (john_rate + rose_rate)

/-- Theorem: John and Rose complete the work in 8 days when working together -/
theorem work_completion_time :
  let john_rate : ℚ := 1 / 10
  let rose_rate : ℚ := 1 / 40
  combined_work_time john_rate rose_rate = 8 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3266_326650


namespace NUMINAMATH_CALUDE_more_subsets_gt_2009_l3266_326657

def M : Finset ℕ := {1, 2, 3, 4, 6, 8, 12, 16, 24, 48}

def product (s : Finset ℕ) : ℕ := s.prod id

def subsets_gt_2009 : Finset (Finset ℕ) :=
  M.powerset.filter (λ s => s.card = 4 ∧ product s > 2009)

def subsets_lt_2009 : Finset (Finset ℕ) :=
  M.powerset.filter (λ s => s.card = 4 ∧ product s < 2009)

theorem more_subsets_gt_2009 : subsets_gt_2009.card > subsets_lt_2009.card := by
  sorry

end NUMINAMATH_CALUDE_more_subsets_gt_2009_l3266_326657


namespace NUMINAMATH_CALUDE_remaining_concert_time_l3266_326605

def concert_duration : ℕ := 165 -- 2 hours and 45 minutes in minutes
def intermission1 : ℕ := 12
def intermission2 : ℕ := 10
def intermission3 : ℕ := 8
def regular_song_duration : ℕ := 4
def ballad_duration : ℕ := 7
def medley_duration : ℕ := 15
def num_regular_songs : ℕ := 15
def num_ballads : ℕ := 6

theorem remaining_concert_time : 
  concert_duration - 
  (intermission1 + intermission2 + intermission3 + 
   num_regular_songs * regular_song_duration + 
   num_ballads * ballad_duration + 
   medley_duration) = 18 := by sorry

end NUMINAMATH_CALUDE_remaining_concert_time_l3266_326605


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l3266_326626

theorem binomial_square_coefficient (b : ℚ) : 
  (∃ t u : ℚ, ∀ x, bx^2 + 18*x + 16 = (t*x + u)^2) → b = 81/16 := by
sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l3266_326626


namespace NUMINAMATH_CALUDE_lg_100_equals_2_l3266_326618

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_100_equals_2 : lg 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_100_equals_2_l3266_326618


namespace NUMINAMATH_CALUDE_square_perimeter_l3266_326686

/-- A square with area 484 cm² has a perimeter of 88 cm. -/
theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : s^2 = 484) : 4 * s = 88 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3266_326686


namespace NUMINAMATH_CALUDE_fred_gave_156_sheets_l3266_326628

/-- The number of sheets Fred gave to Charles -/
def sheets_given_to_charles (initial_sheets : ℕ) (received_sheets : ℕ) (final_sheets : ℕ) : ℕ :=
  initial_sheets + received_sheets - final_sheets

/-- Theorem stating that Fred gave 156 sheets to Charles -/
theorem fred_gave_156_sheets :
  sheets_given_to_charles 212 307 363 = 156 := by
  sorry

end NUMINAMATH_CALUDE_fred_gave_156_sheets_l3266_326628


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_property_l3266_326646

theorem opposite_numbers_sum_property (a b : ℝ) : 
  (∃ k : ℝ, a = k ∧ b = -k) → -5 * (a + b) = 0 := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_property_l3266_326646


namespace NUMINAMATH_CALUDE_gcd_10010_15015_l3266_326677

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10010_15015_l3266_326677


namespace NUMINAMATH_CALUDE_chocolate_candy_cost_l3266_326607

/-- The cost of purchasing a given number of chocolate candies, given the cost and quantity of a box. -/
theorem chocolate_candy_cost (box_quantity : ℕ) (box_cost : ℚ) (total_quantity : ℕ) : 
  (total_quantity / box_quantity : ℚ) * box_cost = 72 :=
by
  sorry

#check chocolate_candy_cost 40 8 360

end NUMINAMATH_CALUDE_chocolate_candy_cost_l3266_326607


namespace NUMINAMATH_CALUDE_donna_has_40_bananas_l3266_326674

/-- The number of bananas Donna has -/
def donnas_bananas (total : ℕ) (dawns_extra : ℕ) (lydias : ℕ) : ℕ :=
  total - (lydias + dawns_extra) - lydias

/-- Proof that Donna has 40 bananas -/
theorem donna_has_40_bananas :
  donnas_bananas 200 40 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_donna_has_40_bananas_l3266_326674


namespace NUMINAMATH_CALUDE_truck_driver_net_pay_rate_l3266_326637

/-- Calculates the net rate of pay for a truck driver given specific conditions --/
theorem truck_driver_net_pay_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (diesel_cost : ℝ)
  (h_travel_time : travel_time = 3)
  (h_speed : speed = 50)
  (h_fuel_efficiency : fuel_efficiency = 25)
  (h_pay_rate : pay_rate = 0.60)
  (h_diesel_cost : diesel_cost = 2.50)
  : (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * diesel_cost) / travel_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_truck_driver_net_pay_rate_l3266_326637


namespace NUMINAMATH_CALUDE_smallest_stairs_l3266_326601

theorem smallest_stairs (n : ℕ) : 
  n > 10 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 → 
  n ≥ 52 ∧ 
  ∃ (m : ℕ), m > 10 ∧ m % 6 = 4 ∧ m % 7 = 3 ∧ m = 52 := by
  sorry

end NUMINAMATH_CALUDE_smallest_stairs_l3266_326601


namespace NUMINAMATH_CALUDE_sphere_volume_in_cone_l3266_326698

/-- A right circular cone with a sphere inscribed inside it -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle in degrees -/
  vertex_angle : ℝ

/-- Calculate the volume of the inscribed sphere -/
def sphere_volume (cone : ConeWithSphere) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed sphere in the given cone -/
theorem sphere_volume_in_cone (cone : ConeWithSphere) 
  (h1 : cone.base_diameter = 24)
  (h2 : cone.vertex_angle = 90) : 
  sphere_volume cone = 2304 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_sphere_volume_in_cone_l3266_326698


namespace NUMINAMATH_CALUDE_initial_bird_families_l3266_326690

/-- The number of bird families that flew away for winter. -/
def flew_away : ℕ := 7

/-- The difference between the number of bird families that stayed and those that flew away. -/
def difference : ℕ := 73

/-- The total number of bird families initially living near the mountain. -/
def total_families : ℕ := flew_away + (flew_away + difference)

theorem initial_bird_families :
  total_families = 87 :=
sorry

end NUMINAMATH_CALUDE_initial_bird_families_l3266_326690


namespace NUMINAMATH_CALUDE_octal_calculation_l3266_326652

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Addition operation for octal numbers --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction operation for octal numbers --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Theorem: ($451_8 + 162_8) - 123_8 = 510_8$ in base 8 --/
theorem octal_calculation : 
  octal_sub (octal_add (to_octal 451) (to_octal 162)) (to_octal 123) = to_octal 510 :=
by sorry

end NUMINAMATH_CALUDE_octal_calculation_l3266_326652


namespace NUMINAMATH_CALUDE_classroom_boys_count_l3266_326610

/-- Represents the number of desks with one boy and one girl -/
def x : ℕ := 2

/-- The number of desks with two girls -/
def desks_two_girls : ℕ := 2 * x

/-- The number of desks with two boys -/
def desks_two_boys : ℕ := 2 * desks_two_girls

/-- The total number of girls in the classroom -/
def total_girls : ℕ := 10

/-- The total number of boys in the classroom -/
def total_boys : ℕ := 2 * desks_two_boys + x

theorem classroom_boys_count :
  total_girls = 5 * x ∧ total_boys = 18 := by
  sorry

#check classroom_boys_count

end NUMINAMATH_CALUDE_classroom_boys_count_l3266_326610


namespace NUMINAMATH_CALUDE_hexagon_triangle_quadrilateral_area_ratio_l3266_326681

/-- A regular hexagon with vertices labeled A to F. -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A quadrilateral -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- The area of a polygon -/
noncomputable def area {n : ℕ} (vertices : Fin n → ℝ × ℝ) : ℝ := sorry

theorem hexagon_triangle_quadrilateral_area_ratio
  (h : RegularHexagon)
  (triangles : Fin 6 → EquilateralTriangle)
  (quad : Quadrilateral) :
  (∀ i, area (triangles i).vertices = area (triangles 0).vertices) →
  (quad.vertices 0 = h.vertices 0) →
  (quad.vertices 1 = h.vertices 2) →
  (quad.vertices 2 = h.vertices 4) →
  (quad.vertices 3 = h.vertices 1) →
  area (triangles 0).vertices / area quad.vertices = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangle_quadrilateral_area_ratio_l3266_326681


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3266_326675

/-- Calculates the average speed of a car trip given odometer readings and time taken -/
theorem average_speed_calculation
  (initial_reading : ℝ)
  (lunch_reading : ℝ)
  (final_reading : ℝ)
  (total_time : ℝ)
  (h1 : initial_reading < lunch_reading)
  (h2 : lunch_reading < final_reading)
  (h3 : total_time > 0) :
  let total_distance := final_reading - initial_reading
  (total_distance / total_time) = (final_reading - initial_reading) / total_time :=
by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3266_326675


namespace NUMINAMATH_CALUDE_tallest_building_height_l3266_326644

theorem tallest_building_height :
  ∀ (h1 h2 h3 h4 : ℝ),
    h2 = h1 / 2 →
    h3 = h2 / 2 →
    h4 = h3 / 5 →
    h1 + h2 + h3 + h4 = 180 →
    h1 = 100 := by
  sorry

end NUMINAMATH_CALUDE_tallest_building_height_l3266_326644


namespace NUMINAMATH_CALUDE_pipe_fill_rate_l3266_326648

theorem pipe_fill_rate (slow_time fast_time combined_time : ℝ) 
  (h1 : slow_time = 160)
  (h2 : combined_time = 40)
  (h3 : slow_time > 0)
  (h4 : fast_time > 0)
  (h5 : combined_time > 0)
  (h6 : 1 / combined_time = 1 / fast_time + 1 / slow_time) :
  fast_time = slow_time / 3 :=
sorry

end NUMINAMATH_CALUDE_pipe_fill_rate_l3266_326648


namespace NUMINAMATH_CALUDE_plane_cost_calculation_l3266_326671

/-- The cost of taking a plane to the Island of Mysteries --/
def plane_cost : ℕ := 600

/-- The cost of taking a boat to the Island of Mysteries --/
def boat_cost : ℕ := 254

/-- The amount saved by taking the boat instead of the plane --/
def savings : ℕ := 346

/-- Theorem stating that the plane cost is equal to the boat cost plus the savings --/
theorem plane_cost_calculation : plane_cost = boat_cost + savings := by
  sorry

end NUMINAMATH_CALUDE_plane_cost_calculation_l3266_326671


namespace NUMINAMATH_CALUDE_problem_1_l3266_326666

theorem problem_1 (α : Real) (h : 2 * Real.sin α - Real.cos α = 0) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) + 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -10/3 := by
sorry

end NUMINAMATH_CALUDE_problem_1_l3266_326666


namespace NUMINAMATH_CALUDE_polynomial_identity_l3266_326614

/-- Given a polynomial function f such that f(x^2 + 1) = x^4 + 4x^2 for all x,
    prove that f(x^2 - 1) = x^4 - 4 for all x. -/
theorem polynomial_identity (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 4*x^2) :
  ∀ x : ℝ, f (x^2 - 1) = x^4 - 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3266_326614


namespace NUMINAMATH_CALUDE_backpack_store_theorem_l3266_326631

/-- Represents the backpack types --/
inductive BackpackType
| A
| B

/-- Represents a purchasing plan --/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

/-- Represents the backpack pricing and inventory --/
structure BackpackStore where
  sellingPriceA : ℕ
  sellingPriceB : ℕ
  costPriceA : ℕ
  costPriceB : ℕ
  inventory : PurchasePlan
  givenAwayA : ℕ
  givenAwayB : ℕ

/-- The main theorem to prove --/
theorem backpack_store_theorem (store : BackpackStore) : 
  (store.sellingPriceA = store.sellingPriceB + 12) →
  (2 * store.sellingPriceA + 3 * store.sellingPriceB = 264) →
  (store.inventory.typeA + store.inventory.typeB = 100) →
  (store.costPriceA * store.inventory.typeA + store.costPriceB * store.inventory.typeB ≤ 4550) →
  (store.inventory.typeA > 52) →
  (store.costPriceA = 50) →
  (store.costPriceB = 40) →
  (store.givenAwayA + store.givenAwayB = 5) →
  (store.sellingPriceA * (store.inventory.typeA - store.givenAwayA) + 
   store.sellingPriceB * (store.inventory.typeB - store.givenAwayB) - 
   store.costPriceA * store.inventory.typeA - 
   store.costPriceB * store.inventory.typeB = 658) →
  (store.sellingPriceA = 60 ∧ store.sellingPriceB = 48) ∧
  ((store.inventory.typeA = 53 ∧ store.inventory.typeB = 47) ∨
   (store.inventory.typeA = 54 ∧ store.inventory.typeB = 46) ∨
   (store.inventory.typeA = 55 ∧ store.inventory.typeB = 45)) ∧
  (store.givenAwayA = 1 ∧ store.givenAwayB = 4) :=
by sorry


end NUMINAMATH_CALUDE_backpack_store_theorem_l3266_326631


namespace NUMINAMATH_CALUDE_third_number_in_second_set_l3266_326608

theorem third_number_in_second_set (x y : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 90 →
  (128 + 255 + y + 1023 + x) / 5 = 423 →
  y = 511 := by sorry

end NUMINAMATH_CALUDE_third_number_in_second_set_l3266_326608


namespace NUMINAMATH_CALUDE_inequality_proof_l3266_326627

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := 2 * (a + 1) * Real.log x - a * x

def g (x : ℝ) : ℝ := (1 / 2) * x^2 - x

theorem inequality_proof (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : -1 < a ∧ a < 7) 
  (hx₁ : x₁ > 1) 
  (hx₂ : x₂ > 1) 
  (hne : x₁ ≠ x₂) : 
  (f a x₁ - f a x₂) / (g x₁ - g x₂) > -1 := by
  sorry

end

end NUMINAMATH_CALUDE_inequality_proof_l3266_326627


namespace NUMINAMATH_CALUDE_kathy_happy_probability_kathy_probability_sum_l3266_326673

def total_cards : ℕ := 10
def cards_laid_out : ℕ := 5
def red_cards : ℕ := 5
def green_cards : ℕ := 5

def happy_configurations : ℕ := 62
def total_configurations : ℕ := 30240

def probability_numerator : ℕ := 31
def probability_denominator : ℕ := 15120

theorem kathy_happy_probability :
  (happy_configurations : ℚ) / total_configurations = probability_numerator / probability_denominator :=
sorry

theorem kathy_probability_sum :
  probability_numerator + probability_denominator = 15151 :=
sorry

end NUMINAMATH_CALUDE_kathy_happy_probability_kathy_probability_sum_l3266_326673


namespace NUMINAMATH_CALUDE_product_105_95_l3266_326680

theorem product_105_95 : 105 * 95 = 9975 := by
  sorry

end NUMINAMATH_CALUDE_product_105_95_l3266_326680


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l3266_326692

def arithmetic_progression (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_sum (a : ℕ → ℝ) :
  arithmetic_progression a → a 5 = 5 → a 3 + a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l3266_326692


namespace NUMINAMATH_CALUDE_lcm_nine_six_l3266_326665

theorem lcm_nine_six : Nat.lcm 9 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_nine_six_l3266_326665


namespace NUMINAMATH_CALUDE_negation_equivalence_l3266_326600

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3266_326600


namespace NUMINAMATH_CALUDE_pens_left_in_jar_l3266_326619

/-- Represents the number of pens of each color in the jar -/
structure PenCount where
  blue : ℕ
  black : ℕ
  red : ℕ
  green : ℕ
  purple : ℕ

def initial_pens : PenCount :=
  { blue := 15, black := 27, red := 12, green := 10, purple := 8 }

def removed_pens : PenCount :=
  { blue := 8, black := 9, red := 3, green := 5, purple := 6 }

def remaining_pens (initial : PenCount) (removed : PenCount) : PenCount :=
  { blue := initial.blue - removed.blue,
    black := initial.black - removed.black,
    red := initial.red - removed.red,
    green := initial.green - removed.green,
    purple := initial.purple - removed.purple }

def total_pens (pens : PenCount) : ℕ :=
  pens.blue + pens.black + pens.red + pens.green + pens.purple

theorem pens_left_in_jar :
  total_pens (remaining_pens initial_pens removed_pens) = 41 := by
  sorry

end NUMINAMATH_CALUDE_pens_left_in_jar_l3266_326619


namespace NUMINAMATH_CALUDE_variableCostIncrease_is_ten_percent_l3266_326667

/-- Represents the annual breeding cost model for a certain breeder -/
structure BreedingCost where
  fixedCost : ℝ
  initialVariableCost : ℝ
  variableCostIncrease : ℝ

/-- Calculates the total breeding cost for a given year -/
def totalCost (model : BreedingCost) (year : ℕ) : ℝ :=
  model.fixedCost + model.initialVariableCost * (1 + model.variableCostIncrease) ^ (year - 1)

/-- Theorem stating that the percentage increase in variable costs is 10% -/
theorem variableCostIncrease_is_ten_percent (model : BreedingCost) :
  model.fixedCost = 40000 →
  model.initialVariableCost = 26000 →
  totalCost model 3 = 71460 →
  model.variableCostIncrease = 0.1 := by
  sorry


end NUMINAMATH_CALUDE_variableCostIncrease_is_ten_percent_l3266_326667


namespace NUMINAMATH_CALUDE_rectangle_perimeter_theorem_l3266_326651

theorem rectangle_perimeter_theorem (a b : ℕ) : 
  a ≠ b →                 -- non-square condition
  a * b = 4 * (2 * a + 2 * b) →  -- area equals four times perimeter
  2 * (a + b) = 66 :=     -- perimeter is 66
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_theorem_l3266_326651


namespace NUMINAMATH_CALUDE_water_flow_difference_l3266_326642

/-- Given a water flow restrictor problem, prove the difference between 0.6 times
    the original flow rate and the reduced flow rate. -/
theorem water_flow_difference (original_rate reduced_rate : ℝ) 
    (h1 : original_rate = 5)
    (h2 : reduced_rate = 2) :
  0.6 * original_rate - reduced_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_difference_l3266_326642


namespace NUMINAMATH_CALUDE_cubes_not_touching_foil_count_l3266_326661

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of cubes not touching tin foil in a prism -/
def cubes_not_touching_foil (outer_width : ℕ) (inner : PrismDimensions) : ℕ :=
  inner.length * inner.width * inner.height

/-- Theorem stating the number of cubes not touching tin foil -/
theorem cubes_not_touching_foil_count 
  (outer_width : ℕ) 
  (inner : PrismDimensions) 
  (h1 : outer_width = 10)
  (h2 : inner.width = 2 * inner.length)
  (h3 : inner.width = 2 * inner.height)
  (h4 : inner.width = outer_width - 4) :
  cubes_not_touching_foil outer_width inner = 54 := by
  sorry

#eval cubes_not_touching_foil 10 ⟨6, 3, 3⟩

end NUMINAMATH_CALUDE_cubes_not_touching_foil_count_l3266_326661


namespace NUMINAMATH_CALUDE_pq_equals_10_l3266_326697

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define properties of the triangle
def isRightAngled (t : Triangle) : Prop := sorry
def anglePRQ (t : Triangle) : ℝ := sorry
def lengthPR (t : Triangle) : ℝ := sorry
def lengthPQ (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem pq_equals_10 (t : Triangle) 
  (h1 : isRightAngled t) 
  (h2 : anglePRQ t = 45) 
  (h3 : lengthPR t = 10) : 
  lengthPQ t = 10 := by sorry

end NUMINAMATH_CALUDE_pq_equals_10_l3266_326697


namespace NUMINAMATH_CALUDE_two_solutions_characterization_l3266_326656

def has_two_solutions (a : ℕ) : Prop :=
  ∃ x y : ℕ, x < 2007 ∧ y < 2007 ∧ x ≠ y ∧
  x^2 + a ≡ 0 [ZMOD 2007] ∧ y^2 + a ≡ 0 [ZMOD 2007] ∧
  ∀ z : ℕ, z < 2007 → z^2 + a ≡ 0 [ZMOD 2007] → (z = x ∨ z = y)

theorem two_solutions_characterization :
  {a : ℕ | a < 2007 ∧ has_two_solutions a} = {446, 1115, 1784} := by sorry

end NUMINAMATH_CALUDE_two_solutions_characterization_l3266_326656


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l3266_326659

/-- The eccentricity of the conic section defined by 10x - 2xy - 2y + 1 = 0 is √2 -/
theorem conic_section_eccentricity :
  let P : ℝ × ℝ → Prop := λ (x, y) ↦ 10 * x - 2 * x * y - 2 * y + 1 = 0
  ∃ e : ℝ, e = Real.sqrt 2 ∧
    ∀ (x y : ℝ), P (x, y) →
      (Real.sqrt ((x - 2)^2 + (y - 2)^2)) / (|x - y + 3| / Real.sqrt 2) = e :=
by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l3266_326659


namespace NUMINAMATH_CALUDE_basketball_win_percentage_l3266_326630

theorem basketball_win_percentage (games_played : ℕ) (games_won : ℕ) (games_left : ℕ) 
  (target_percentage : ℚ) (h1 : games_played = 50) (h2 : games_won = 35) 
  (h3 : games_left = 25) (h4 : target_percentage = 64 / 100) : 
  ∃ (additional_wins : ℕ), 
    (games_won + additional_wins) / (games_played + games_left : ℚ) = target_percentage ∧ 
    additional_wins = 13 := by
  sorry

end NUMINAMATH_CALUDE_basketball_win_percentage_l3266_326630


namespace NUMINAMATH_CALUDE_finsler_hadwiger_inequality_l3266_326611

/-- The Finsler-Hadwiger inequality for triangles -/
theorem finsler_hadwiger_inequality (a b c : ℝ) (S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0) :
  a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end NUMINAMATH_CALUDE_finsler_hadwiger_inequality_l3266_326611


namespace NUMINAMATH_CALUDE_theater_seat_increment_l3266_326616

/-- Represents a theater with a specific seating arrangement -/
structure Theater where
  num_rows : ℕ
  first_row_seats : ℕ
  last_row_seats : ℕ
  total_seats : ℕ

/-- 
  Given a theater with 23 rows, where the first row has 14 seats, 
  the last row has 56 seats, and the total number of seats is 770, 
  prove that the number of additional seats in each row compared 
  to the previous row is 2.
-/
theorem theater_seat_increment (t : Theater) 
  (h1 : t.num_rows = 23)
  (h2 : t.first_row_seats = 14)
  (h3 : t.last_row_seats = 56)
  (h4 : t.total_seats = 770) : 
  (t.last_row_seats - t.first_row_seats) / (t.num_rows - 1) = 2 := by
  sorry


end NUMINAMATH_CALUDE_theater_seat_increment_l3266_326616
