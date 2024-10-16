import Mathlib

namespace NUMINAMATH_CALUDE_expected_value_of_three_from_seven_l1787_178763

/-- The number of marbles in the bag -/
def n : ℕ := 7

/-- The number of marbles drawn -/
def k : ℕ := 3

/-- The sum of numbers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The average value of a set of k elements from 1 to n -/
def avg_value (n k : ℕ) : ℚ := (sum_to_n n : ℚ) / n * k

/-- The expected value of the sum of k randomly chosen marbles from n marbles -/
def expected_value (n k : ℕ) : ℚ := avg_value n k

theorem expected_value_of_three_from_seven :
  expected_value n k = 12 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_three_from_seven_l1787_178763


namespace NUMINAMATH_CALUDE_race_distance_P_300_l1787_178733

/-- A race between two runners P and Q, where P is faster but Q gets a head start -/
structure Race where
  /-- The speed ratio of P to Q -/
  speed_ratio : ℝ
  /-- The head start given to Q in meters -/
  head_start : ℝ

/-- The distance run by P in the race -/
def distance_P (race : Race) : ℝ :=
  sorry

theorem race_distance_P_300 (race : Race) 
  (h_speed : race.speed_ratio = 1.25)
  (h_head_start : race.head_start = 60)
  (h_tie : distance_P race = distance_P race - race.head_start + race.head_start) :
  distance_P race = 300 :=
sorry

end NUMINAMATH_CALUDE_race_distance_P_300_l1787_178733


namespace NUMINAMATH_CALUDE_min_value_expression_l1787_178781

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (a + 1) + 4 / (b + 1) ≥ 9 / 4 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / (a₀ + 1) + 4 / (b₀ + 1) = 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1787_178781


namespace NUMINAMATH_CALUDE_elevator_exit_theorem_l1787_178702

/-- The number of ways 9 passengers can exit an elevator in groups of 2, 3, and 4 at any of 10 floors -/
def elevator_exit_ways : ℕ :=
  Nat.factorial 10 / Nat.factorial 4

/-- Theorem stating that the number of ways 9 passengers can exit an elevator
    in groups of 2, 3, and 4 at any of 10 floors is equal to 10! / 4! -/
theorem elevator_exit_theorem :
  elevator_exit_ways = Nat.factorial 10 / Nat.factorial 4 := by
  sorry

end NUMINAMATH_CALUDE_elevator_exit_theorem_l1787_178702


namespace NUMINAMATH_CALUDE_tan_angle_equality_l1787_178744

theorem tan_angle_equality (n : ℤ) : 
  -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (1600 * π / 180) → n = -20 :=
by sorry

end NUMINAMATH_CALUDE_tan_angle_equality_l1787_178744


namespace NUMINAMATH_CALUDE_ninth_term_value_l1787_178722

/-- An arithmetic sequence {aₙ} with sum Sₙ of first n terms -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ S n = n / 2 * (2 * a 1 + (n - 1) * d)

theorem ninth_term_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a S)
  (h_S8 : S 8 = 4 * a 1)
  (h_a7 : a 7 = -2)
  : a 9 = 2 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_value_l1787_178722


namespace NUMINAMATH_CALUDE_point_on_line_l1787_178728

/-- Given a point P(2, m) lying on the line 3x + y = 2, prove that m = -4 -/
theorem point_on_line (m : ℝ) : (3 * 2 + m = 2) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1787_178728


namespace NUMINAMATH_CALUDE_trip_duration_is_101_l1787_178700

/-- Calculates the total trip duration for Jill's journey to the library --/
def total_trip_duration (first_bus_wait : ℕ) (first_bus_ride : ℕ) (first_bus_delay : ℕ)
                        (walk_time : ℕ) (train_wait : ℕ) (train_ride : ℕ) (train_delay : ℕ)
                        (second_bus_wait_A : ℕ) (second_bus_ride_A : ℕ)
                        (second_bus_wait_B : ℕ) (second_bus_ride_B : ℕ) : ℕ :=
  let first_bus_total := first_bus_wait + first_bus_ride + first_bus_delay
  let train_total := walk_time + train_wait + train_ride + train_delay
  let second_bus_total := if second_bus_ride_A < second_bus_ride_B
                          then (second_bus_wait_A + second_bus_ride_A) / 2
                          else (second_bus_wait_B + second_bus_ride_B) / 2
  first_bus_total + train_total + second_bus_total

/-- Theorem stating that the total trip duration is 101 minutes --/
theorem trip_duration_is_101 :
  total_trip_duration 12 30 5 10 8 20 3 15 10 20 6 = 101 := by
  sorry

end NUMINAMATH_CALUDE_trip_duration_is_101_l1787_178700


namespace NUMINAMATH_CALUDE_vector_parallel_problem_l1787_178742

/-- Two 2D vectors are parallel if their components are proportional -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_parallel_problem (m : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-3, 0)
  parallel (2 • a + b) (a - m • b) →
  m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_problem_l1787_178742


namespace NUMINAMATH_CALUDE_ac_lt_zero_sufficient_not_necessary_l1787_178782

theorem ac_lt_zero_sufficient_not_necessary (a b c : ℝ) (h : c < b ∧ b < a) :
  (∀ x y z : ℝ, x < y ∧ y < z ∧ x*z < 0 → z*y > z*x) ∧
  (∃ x y z : ℝ, x < y ∧ y < z ∧ z*y > z*x ∧ x*z ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ac_lt_zero_sufficient_not_necessary_l1787_178782


namespace NUMINAMATH_CALUDE_new_train_distance_l1787_178743

theorem new_train_distance (old_distance : ℝ) (percentage_increase : ℝ) : 
  old_distance = 300 → percentage_increase = 30 → 
  old_distance * (1 + percentage_increase / 100) = 390 := by
  sorry

end NUMINAMATH_CALUDE_new_train_distance_l1787_178743


namespace NUMINAMATH_CALUDE_f_2018_eq_l1787_178765

open Real

/-- Sequence of functions defined recursively --/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => sin x - cos x
  | n + 1 => λ x => deriv (f n) x

/-- The 2018th function in the sequence equals -sin(x) + cos(x) --/
theorem f_2018_eq (x : ℝ) : f 2018 x = -sin x + cos x := by
  sorry

end NUMINAMATH_CALUDE_f_2018_eq_l1787_178765


namespace NUMINAMATH_CALUDE_dormitory_arrangements_l1787_178780

def num_students : ℕ := 7
def min_per_dorm : ℕ := 2

-- Function to calculate the number of arrangements
def calculate_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  sorry

theorem dormitory_arrangements :
  calculate_arrangements num_students min_per_dorm 2 = 60 :=
sorry

end NUMINAMATH_CALUDE_dormitory_arrangements_l1787_178780


namespace NUMINAMATH_CALUDE_bankers_discount_example_l1787_178762

/-- Calculates the banker's discount given the face value and true discount of a bill. -/
def bankers_discount (face_value : ℚ) (true_discount : ℚ) : ℚ :=
  let present_value := face_value - true_discount
  (true_discount / present_value) * face_value

/-- Theorem stating that for a bill with face value 2660 and true discount 360,
    the banker's discount is approximately 416.35. -/
theorem bankers_discount_example :
  ∃ ε > 0, |bankers_discount 2660 360 - 416.35| < ε :=
by
  sorry

#eval bankers_discount 2660 360

end NUMINAMATH_CALUDE_bankers_discount_example_l1787_178762


namespace NUMINAMATH_CALUDE_equipment_value_after_three_years_l1787_178767

/-- The value of equipment after n years, given an initial value and annual depreciation rate. -/
def equipment_value (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

/-- Theorem: The value of equipment initially worth 10,000 yuan, depreciating by 50% annually, will be 1,250 yuan after 3 years. -/
theorem equipment_value_after_three_years :
  equipment_value 10000 0.5 3 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_equipment_value_after_three_years_l1787_178767


namespace NUMINAMATH_CALUDE_sam_puppies_count_l1787_178719

def final_puppies (initial bought given_away sold : ℕ) : ℕ :=
  initial - given_away + bought - sold

theorem sam_puppies_count : final_puppies 72 25 18 13 = 66 := by
  sorry

end NUMINAMATH_CALUDE_sam_puppies_count_l1787_178719


namespace NUMINAMATH_CALUDE_earnings_calculation_l1787_178715

/-- If a person spends 10% of their earnings and is left with $405, prove their total earnings were $450. -/
theorem earnings_calculation (spent_percentage : Real) (remaining_amount : Real) (total_earnings : Real) : 
  spent_percentage = 0.1 →
  remaining_amount = 405 →
  remaining_amount = (1 - spent_percentage) * total_earnings →
  total_earnings = 450 := by
sorry

end NUMINAMATH_CALUDE_earnings_calculation_l1787_178715


namespace NUMINAMATH_CALUDE_consecutive_nonprime_integers_l1787_178732

theorem consecutive_nonprime_integers :
  ∃ n : ℕ,
    100 < n ∧
    n + 4 < 200 ∧
    (¬ Prime n) ∧
    (¬ Prime (n + 1)) ∧
    (¬ Prime (n + 2)) ∧
    (¬ Prime (n + 3)) ∧
    (¬ Prime (n + 4)) ∧
    n + 4 = 148 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_nonprime_integers_l1787_178732


namespace NUMINAMATH_CALUDE_magnitude_z_l1787_178792

theorem magnitude_z (w z : ℂ) (h1 : w * z = 16 - 30 * I) (h2 : Complex.abs w = 5) : 
  Complex.abs z = 6.8 := by
sorry

end NUMINAMATH_CALUDE_magnitude_z_l1787_178792


namespace NUMINAMATH_CALUDE_shaded_square_covering_all_columns_l1787_178754

def shaded_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => shaded_sequence n + (2 * (n + 1) - 1)

def column_position (n : ℕ) : ℕ :=
  (shaded_sequence n - 1) % 10 + 1

def all_columns_covered (n : ℕ) : Prop :=
  ∀ k : ℕ, k ∈ Finset.range 10 → ∃ i : ℕ, i ≤ n ∧ column_position i = k + 1

theorem shaded_square_covering_all_columns :
  all_columns_covered 20 ∧ ∀ m : ℕ, m < 20 → ¬ all_columns_covered m :=
sorry

end NUMINAMATH_CALUDE_shaded_square_covering_all_columns_l1787_178754


namespace NUMINAMATH_CALUDE_solution_sum_l1787_178760

theorem solution_sum (a b x y : ℝ) : 
  x = 2 ∧ y = -1 ∧ 
  a * x - 2 * y = 4 ∧ 
  3 * x + b * y = -7 →
  a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l1787_178760


namespace NUMINAMATH_CALUDE_chocolate_distribution_l1787_178759

theorem chocolate_distribution (pieces_per_bar : ℕ) 
  (girls_consumption_case1 girls_consumption_case2 : ℕ) 
  (boys_consumption_case1 boys_consumption_case2 : ℕ) 
  (bars_case1 bars_case2 : ℕ) : 
  pieces_per_bar = 12 →
  girls_consumption_case1 = 7 →
  boys_consumption_case1 = 2 →
  bars_case1 = 3 →
  girls_consumption_case2 = 8 →
  boys_consumption_case2 = 4 →
  bars_case2 = 4 →
  ∃ (girls boys : ℕ),
    girls_consumption_case1 * girls + boys_consumption_case1 * boys > pieces_per_bar * bars_case1 ∧
    girls_consumption_case2 * girls + boys_consumption_case2 * boys < pieces_per_bar * bars_case2 ∧
    girls = 5 ∧
    boys = 1 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l1787_178759


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1787_178786

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 5) * (x - 6) = -34 + k * x) ↔ 
  (k = -13 + 4 * Real.sqrt 3 ∨ k = -13 - 4 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1787_178786


namespace NUMINAMATH_CALUDE_largest_term_index_l1787_178725

def A (k : ℕ) : ℝ := (Nat.choose 2000 k) * (0.1 ^ k)

theorem largest_term_index : 
  ∃ (k : ℕ), k ≤ 2000 ∧ 
  (∀ (j : ℕ), j ≤ 2000 → A k ≥ A j) ∧
  k = 181 := by
  sorry

end NUMINAMATH_CALUDE_largest_term_index_l1787_178725


namespace NUMINAMATH_CALUDE_probability_four_white_balls_l1787_178796

/-- The probability of drawing 4 white balls from a box containing 7 white and 8 black balls -/
theorem probability_four_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) :
  total_balls = white_balls + black_balls →
  total_balls = 15 →
  white_balls = 7 →
  black_balls = 8 →
  drawn_balls = 4 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 39 :=
by sorry

end NUMINAMATH_CALUDE_probability_four_white_balls_l1787_178796


namespace NUMINAMATH_CALUDE_marissa_initial_ribbon_l1787_178752

/-- The amount of ribbon used per box in feet -/
def ribbon_per_box : ℝ := 0.7

/-- The number of boxes Marissa tied -/
def num_boxes : ℕ := 5

/-- The amount of ribbon left after tying all boxes in feet -/
def ribbon_left : ℝ := 1

/-- The initial amount of ribbon Marissa had in feet -/
def initial_ribbon : ℝ := ribbon_per_box * num_boxes + ribbon_left

theorem marissa_initial_ribbon :
  initial_ribbon = 4.5 := by sorry

end NUMINAMATH_CALUDE_marissa_initial_ribbon_l1787_178752


namespace NUMINAMATH_CALUDE_problem_statement_l1787_178705

theorem problem_statement :
  (∀ x : ℝ, (x + 8) * (x + 11) < (x + 9) * (x + 10)) ∧
  (Real.sqrt 5 - 2 > Real.sqrt 6 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1787_178705


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1787_178772

theorem interest_rate_calculation (P : ℝ) (R : ℝ) : 
  P * (1 + 5 * R / 100) = 9800 →
  P * (1 + 8 * R / 100) = 12005 →
  R = 12 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1787_178772


namespace NUMINAMATH_CALUDE_outfit_choices_l1787_178740

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 6

/-- The number of shirts available -/
def num_shirts : ℕ := num_colors

/-- The number of pants available -/
def num_pants : ℕ := num_colors

/-- The number of hats available -/
def num_hats : ℕ := num_colors

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Theorem: The number of outfit choices where not all items are the same color -/
theorem outfit_choices : 
  total_combinations - same_color_outfits = 210 :=
sorry

end NUMINAMATH_CALUDE_outfit_choices_l1787_178740


namespace NUMINAMATH_CALUDE_extra_flowers_l1787_178708

def tulips : ℕ := 36
def roses : ℕ := 37
def used_flowers : ℕ := 70

theorem extra_flowers :
  tulips + roses - used_flowers = 3 := by sorry

end NUMINAMATH_CALUDE_extra_flowers_l1787_178708


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1787_178724

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1787_178724


namespace NUMINAMATH_CALUDE_paint_remaining_l1787_178778

theorem paint_remaining (initial_paint : ℚ) : initial_paint = 2 →
  let day1_remaining := initial_paint / 2
  let day2_remaining := day1_remaining * 3 / 4
  let day3_remaining := day2_remaining * 2 / 3
  day3_remaining = initial_paint / 2 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_l1787_178778


namespace NUMINAMATH_CALUDE_factorization_cubic_l1787_178729

theorem factorization_cubic (a b : ℝ) : a^3 + 2*a^2*b + a*b^2 = a*(a+b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_l1787_178729


namespace NUMINAMATH_CALUDE_nh3_moles_produced_l1787_178704

structure Reaction where
  reactants : List (String × ℚ)
  products : List (String × ℚ)

def initial_moles : List (String × ℚ) := [
  ("NH4Cl", 3),
  ("KOH", 3),
  ("Na2CO3", 1),
  ("H3PO4", 1)
]

def reaction1 : Reaction := {
  reactants := [("NH4Cl", 2), ("Na2CO3", 1)],
  products := [("NH3", 2), ("CO2", 1), ("NaCl", 2), ("H2O", 1)]
}

def reaction2 : Reaction := {
  reactants := [("KOH", 2), ("H3PO4", 1)],
  products := [("K2HPO4", 1), ("H2O", 2)]
}

def limiting_reactant (reaction : Reaction) (available : List (String × ℚ)) : String :=
  sorry

def moles_produced (reaction : Reaction) (product : String) (limiting : String) : ℚ :=
  sorry

theorem nh3_moles_produced : 
  moles_produced reaction1 "NH3" (limiting_reactant reaction1 initial_moles) = 2 :=
sorry

end NUMINAMATH_CALUDE_nh3_moles_produced_l1787_178704


namespace NUMINAMATH_CALUDE_rearrangement_count_correct_l1787_178714

/-- The number of ways to rearrange 3 out of 8 people in a row, 
    while keeping the other 5 in their original positions. -/
def rearrangement_count : ℕ := Nat.choose 8 3 * 2

/-- Theorem stating that the number of rearrangements is correct. -/
theorem rearrangement_count_correct : 
  rearrangement_count = Nat.choose 8 3 * 2 := by sorry

end NUMINAMATH_CALUDE_rearrangement_count_correct_l1787_178714


namespace NUMINAMATH_CALUDE_christen_peeled_17_l1787_178757

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initial_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  homer_solo_time : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christens_potatoes (scenario : PotatoPeeling) : ℕ :=
  let potatoes_after_homer := scenario.initial_potatoes - scenario.homer_rate * scenario.homer_solo_time
  let combined_rate := scenario.homer_rate + scenario.christen_rate
  let remaining_time := potatoes_after_homer / combined_rate
  remaining_time * scenario.christen_rate

/-- Theorem stating that Christen peeled 17 potatoes -/
theorem christen_peeled_17 (scenario : PotatoPeeling) 
  (h1 : scenario.initial_potatoes = 58)
  (h2 : scenario.homer_rate = 4)
  (h3 : scenario.christen_rate = 4)
  (h4 : scenario.homer_solo_time = 6) :
  christens_potatoes scenario = 17 := by
  sorry

#eval christens_potatoes { initial_potatoes := 58, homer_rate := 4, christen_rate := 4, homer_solo_time := 6 }

end NUMINAMATH_CALUDE_christen_peeled_17_l1787_178757


namespace NUMINAMATH_CALUDE_tangent_line_parallel_point_l1787_178703

/-- The function f(x) = x^4 - x --/
def f (x : ℝ) : ℝ := x^4 - x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_line_parallel_point (P : ℝ × ℝ) :
  P.1 = 1 ∧ P.2 = 0 ↔
    f P.1 = P.2 ∧ f' P.1 = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_point_l1787_178703


namespace NUMINAMATH_CALUDE_square_area_ratio_l1787_178787

theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : a^2 / b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1787_178787


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_seventeen_l1787_178790

def sum_of_last_two_digits (n : ℕ) : ℕ :=
  (n % 100) / 10 + n % 10

theorem sum_of_digits_of_seven_to_seventeen (n : ℕ) (h : n = (3 + 4)^17) :
  sum_of_last_two_digits n = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_seventeen_l1787_178790


namespace NUMINAMATH_CALUDE_carols_blocks_l1787_178713

/-- Carol's block problem -/
theorem carols_blocks (initial_blocks lost_blocks : ℕ) :
  initial_blocks = 42 →
  lost_blocks = 25 →
  initial_blocks - lost_blocks = 17 :=
by sorry

end NUMINAMATH_CALUDE_carols_blocks_l1787_178713


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1787_178707

/-- A geometric sequence with the given first three terms has its fourth term equal to -24 -/
theorem geometric_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (x : ℝ) 
  (h1 : a 1 = x)
  (h2 : a 2 = 3*x + 3)
  (h3 : a 3 = 6*x + 6)
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n+1) / a n = a 2 / a 1) :
  a 4 = -24 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1787_178707


namespace NUMINAMATH_CALUDE_expression_value_l1787_178748

theorem expression_value :
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 1
  (x^2 * y * z - x * y * z^2) = 6 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1787_178748


namespace NUMINAMATH_CALUDE_remainder_thirteen_150_mod_11_l1787_178736

theorem remainder_thirteen_150_mod_11 : 13^150 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_thirteen_150_mod_11_l1787_178736


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l1787_178717

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 31)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0)
  : total_players - (total_players - throwers) / 3 = 57 := by
  sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l1787_178717


namespace NUMINAMATH_CALUDE_all_balls_are_red_l1787_178746

/-- 
Given a bag of 12 balls that are either red or blue, 
prove that if the probability of drawing two red balls simultaneously is 1/10, 
then all 12 balls must be red.
-/
theorem all_balls_are_red (total_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = 12)
  (h2 : red_balls ≤ total_balls)
  (h3 : (red_balls : ℚ) / total_balls * (red_balls - 1) / (total_balls - 1) = 1 / 10) :
  red_balls = total_balls :=
sorry

end NUMINAMATH_CALUDE_all_balls_are_red_l1787_178746


namespace NUMINAMATH_CALUDE_range_of_h_l1787_178756

noncomputable def h (t : ℝ) : ℝ := (t^2 + 5/4 * t) / (t^2 + 2)

theorem range_of_h :
  Set.range h = Set.Icc 0 (128/103) := by sorry

end NUMINAMATH_CALUDE_range_of_h_l1787_178756


namespace NUMINAMATH_CALUDE_bobby_shoe_cost_l1787_178735

/-- Calculates the total cost of Bobby's handmade shoes -/
def calculate_total_cost (mold_cost : ℝ) (material_cost : ℝ) (material_discount : ℝ) 
  (hourly_rate : ℝ) (rate_increase : ℝ) (work_hours : ℝ) (work_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_material := material_cost * (1 - material_discount)
  let new_hourly_rate := hourly_rate + rate_increase
  let work_cost := work_hours * new_hourly_rate * work_discount
  let subtotal := mold_cost + discounted_material + work_cost
  let total := subtotal * (1 + tax_rate)
  total

/-- Theorem stating that Bobby's total cost is $1005.40 -/
theorem bobby_shoe_cost : 
  calculate_total_cost 250 150 0.2 75 10 8 0.8 0.1 = 1005.40 := by
  sorry

end NUMINAMATH_CALUDE_bobby_shoe_cost_l1787_178735


namespace NUMINAMATH_CALUDE_cylinder_j_value_l1787_178720

/-- The value of J for a cylinder with specific properties -/
theorem cylinder_j_value (h d r : ℝ) (j : ℝ) : 
  h > 0 → d > 0 → r > 0 →
  h = d →  -- Cylinder height equals diameter
  r = d / 2 →  -- Radius is half the diameter
  6 * 3^2 = 2 * π * r^2 + π * d * h →  -- Surface area of cylinder equals surface area of cube
  j * π / 6 = π * r^2 * h →  -- Volume of cylinder
  j = 324 * Real.sqrt π := by
sorry

end NUMINAMATH_CALUDE_cylinder_j_value_l1787_178720


namespace NUMINAMATH_CALUDE_largest_digit_rounding_l1787_178777

def number (d : ℕ) : ℕ := 5400000000 + d * 10000000 + 9607502

def rounds_to_5_5_billion (n : ℕ) : Prop :=
  5450000000 ≤ n ∧ n < 5550000000

theorem largest_digit_rounding :
  ∀ d : ℕ, d ≤ 9 →
    (rounds_to_5_5_billion (number d) ↔ 5 ≤ d) ∧
    (d = 9 ↔ ∀ k : ℕ, k ≤ 9 ∧ rounds_to_5_5_billion (number k) → k ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_rounding_l1787_178777


namespace NUMINAMATH_CALUDE_smallest_fraction_l1787_178789

theorem smallest_fraction (x : ℝ) (h : x = 5) : 
  min (min (min (min (8/x) (8/(x+1))) (8/(x-1))) (x/8)) ((x+1)/8) = x/8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l1787_178789


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l1787_178764

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (middle : ℝ) :
  first_six_avg = 10.5 →
  last_six_avg = 11.4 →
  middle = 22.5 →
  (6 * first_six_avg + 6 * last_six_avg - middle) / 11 = 9.9 := by
sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l1787_178764


namespace NUMINAMATH_CALUDE_sin_150_degrees_l1787_178753

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l1787_178753


namespace NUMINAMATH_CALUDE_divisibility_by_eight_and_nine_l1787_178793

theorem divisibility_by_eight_and_nine (x y : Nat) : 
  x < 10 ∧ y < 10 →
  (1234 * 10 * x + 1234 * y) % 8 = 0 ∧ 
  (1234 * 10 * x + 1234 * y) % 9 = 0 ↔ 
  (x = 8 ∧ y = 0) ∨ (x = 0 ∧ y = 8) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_eight_and_nine_l1787_178793


namespace NUMINAMATH_CALUDE_g_of_4_l1787_178785

def g (x : ℝ) : ℝ := 5 * x - 2

theorem g_of_4 : g 4 = 18 := by sorry

end NUMINAMATH_CALUDE_g_of_4_l1787_178785


namespace NUMINAMATH_CALUDE_arc_length_theorem_l1787_178706

-- Define the curve
def curve (x y : ℝ) : Prop := Real.exp (2 * y) * (Real.exp (2 * x) - 1) = Real.exp (2 * x) + 1

-- Define the arc length function
noncomputable def arcLength (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (1 + (deriv f x) ^ 2)

-- State the theorem
theorem arc_length_theorem :
  ∃ f : ℝ → ℝ,
    (∀ x, curve x (f x)) ∧
    arcLength f 1 2 = (1 / 2) * Real.log (Real.exp 4 + 1) - 1 := by sorry

end NUMINAMATH_CALUDE_arc_length_theorem_l1787_178706


namespace NUMINAMATH_CALUDE_soccer_match_handshakes_l1787_178768

def soccer_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let player_handshakes := team_size * team_size * (num_teams - 1) / 2
  let referee_handshakes := team_size * num_teams * num_referees
  player_handshakes + referee_handshakes

theorem soccer_match_handshakes :
  soccer_handshakes 6 2 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_soccer_match_handshakes_l1787_178768


namespace NUMINAMATH_CALUDE_sam_placed_twelve_crayons_l1787_178799

/-- The number of crayons Sam placed in the drawer -/
def crayons_placed (initial_crayons final_crayons : ℕ) : ℕ :=
  final_crayons - initial_crayons

/-- Theorem: Sam placed 12 crayons in the drawer -/
theorem sam_placed_twelve_crayons :
  crayons_placed 41 53 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sam_placed_twelve_crayons_l1787_178799


namespace NUMINAMATH_CALUDE_hospital_staff_count_l1787_178723

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) 
  (h_total : total = 280)
  (h_ratio : doctor_ratio = 5 ∧ nurse_ratio = 9) :
  (nurse_ratio * total) / (doctor_ratio + nurse_ratio) = 180 :=
by sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l1787_178723


namespace NUMINAMATH_CALUDE_salary_proof_l1787_178734

/-- Represents the man's salary in dollars -/
def salary : ℝ := 190000

/-- Theorem stating that given the spending conditions, the salary is $190000 -/
theorem salary_proof :
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := salary - (food_expense + rent_expense + clothes_expense)
  remaining = 19000 := by sorry

end NUMINAMATH_CALUDE_salary_proof_l1787_178734


namespace NUMINAMATH_CALUDE_ln_geq_num_prime_factors_ln2_l1787_178773

/-- The number of prime factors of a positive integer -/
def num_prime_factors (n : ℕ+) : ℕ := sorry

/-- For any positive integer n, ln n ≥ p(n) ln 2, where p(n) is the number of prime factors of n -/
theorem ln_geq_num_prime_factors_ln2 (n : ℕ+) : Real.log n ≥ (num_prime_factors n : ℝ) * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ln_geq_num_prime_factors_ln2_l1787_178773


namespace NUMINAMATH_CALUDE_tom_candy_l1787_178741

def candy_problem (initial : ℕ) (from_friend : ℕ) (bought : ℕ) : Prop :=
  initial + from_friend + bought = 19

theorem tom_candy : candy_problem 2 7 10 := by sorry

end NUMINAMATH_CALUDE_tom_candy_l1787_178741


namespace NUMINAMATH_CALUDE_cubic_equation_has_real_root_l1787_178761

theorem cubic_equation_has_real_root :
  ∃ (x : ℝ), x^3 + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_has_real_root_l1787_178761


namespace NUMINAMATH_CALUDE_gcd_105_88_l1787_178727

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_88_l1787_178727


namespace NUMINAMATH_CALUDE_largest_possible_median_is_one_l1787_178775

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  s.card % 2 = 1 ∧ 
  (s.filter (· < m)).card = (s.filter (· > m)).card

theorem largest_possible_median_is_one (x : ℤ) (h : x < 0) :
  is_median 1 {x, 2*x, 4, 1, 7} :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_median_is_one_l1787_178775


namespace NUMINAMATH_CALUDE_negation_of_p_l1787_178771

-- Define the original proposition
def p : Prop := ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 3 > 0

-- State the theorem
theorem negation_of_p : 
  ¬p ↔ ∀ x : ℝ, x^2 + 2*x + 3 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l1787_178771


namespace NUMINAMATH_CALUDE_percentage_equality_l1787_178751

theorem percentage_equality (x : ℝ) (h : x > 0) :
  ∃ p : ℝ, p / 100 * (x + 20) = 0.3 * (0.6 * x) ∧ p = 1800 * x / (x + 20) := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1787_178751


namespace NUMINAMATH_CALUDE_art_gallery_theorem_l1787_178716

theorem art_gallery_theorem (T : ℕ) 
  (h1 : T / 3 = T - (2 * T / 3))  -- 1/3 of pieces are displayed
  (h2 : (T / 3) / 6 = T / 18)  -- 1/6 of displayed pieces are sculptures
  (h3 : (2 * T / 3) / 3 = 2 * T / 9)  -- 1/3 of non-displayed pieces are paintings
  (h4 : T / 18 + 400 = T / 18 + (T - (T / 3)) / 3)  -- 400 sculptures not on display
  (h5 : 3 * (T / 18) = T / 6)  -- 3 photographs for each displayed sculpture
  (h6 : 2 * (T / 18) = T / 9)  -- 2 installations for each displayed sculpture
  : T = 7200 := by
sorry

end NUMINAMATH_CALUDE_art_gallery_theorem_l1787_178716


namespace NUMINAMATH_CALUDE_managers_salary_l1787_178711

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 100 →
  (num_employees : ℝ) * avg_salary + (num_employees + 1 : ℝ) * salary_increase = 3600 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l1787_178711


namespace NUMINAMATH_CALUDE_sin_cos_sum_10_50_l1787_178758

theorem sin_cos_sum_10_50 : 
  Real.sin (10 * π / 180) * Real.cos (50 * π / 180) + 
  Real.cos (10 * π / 180) * Real.sin (50 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_10_50_l1787_178758


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l1787_178776

theorem quadratic_equation_transformation :
  ∀ x : ℝ, (2 * x^2 = -3 * x + 1) ↔ (2 * x^2 + 3 * x - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l1787_178776


namespace NUMINAMATH_CALUDE_mushroom_collection_l1787_178737

theorem mushroom_collection (a b v g : ℚ) 
  (eq1 : a / 2 + 2 * b = v + g) 
  (eq2 : a + b = v / 2 + 2 * g) : 
  v = 2 * b ∧ a = 2 * g := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_l1787_178737


namespace NUMINAMATH_CALUDE_robot_race_track_length_l1787_178788

/-- Represents the race between three robots A, B, and C --/
structure RobotRace where
  track_length : ℝ
  va : ℝ
  vb : ℝ
  vc : ℝ

/-- The conditions of the race --/
def race_conditions (race : RobotRace) : Prop :=
  race.track_length > 0 ∧
  race.va > 0 ∧ race.vb > 0 ∧ race.vc > 0 ∧
  race.track_length / race.va = (race.track_length - 1) / race.vb ∧
  race.track_length / race.va = (race.track_length - 2) / race.vc ∧
  race.track_length / race.vb = (race.track_length - 1.01) / race.vc

theorem robot_race_track_length (race : RobotRace) :
  race_conditions race → race.track_length = 101 := by
  sorry

#check robot_race_track_length

end NUMINAMATH_CALUDE_robot_race_track_length_l1787_178788


namespace NUMINAMATH_CALUDE_distribute_balls_into_boxes_l1787_178718

theorem distribute_balls_into_boxes (n : ℕ) (k : ℕ) : 
  n = 5 → k = 4 → (Nat.choose (n + k - 1) (k - 1)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_distribute_balls_into_boxes_l1787_178718


namespace NUMINAMATH_CALUDE_triangle_inequality_with_medians_l1787_178709

/-- Given a triangle with sides a, b, c and medians s_a, s_b, s_c, 
    prove the inequality a + b + c > s_a + s_b + s_c > 3/4 * (a + b + c) -/
theorem triangle_inequality_with_medians 
  (a b c s_a s_b s_c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : s_a = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2))
  (h_median_b : s_b = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2))
  (h_median_c : s_c = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)) :
  a + b + c > s_a + s_b + s_c ∧ s_a + s_b + s_c > (3/4) * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_medians_l1787_178709


namespace NUMINAMATH_CALUDE_min_value_expression_l1787_178783

theorem min_value_expression (x y : ℝ) (h : x^2 + y^2 ≤ 1) :
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 + b^2 ≤ 1 → |2*a + b - 2| + |6 - a - 3*b| ≥ m) ∧
             (∃ (c d : ℝ), c^2 + d^2 ≤ 1 ∧ |2*c + d - 2| + |6 - c - 3*d| = m) ∧
             m = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1787_178783


namespace NUMINAMATH_CALUDE_relationship_abc_l1787_178769

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.7 0.6
  let b : ℝ := Real.rpow 0.6 (-0.6)
  let c : ℝ := Real.rpow 0.6 0.7
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1787_178769


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1787_178774

-- Define the sets p and q
def p : Set ℝ := {x | |2*x - 3| > 1}
def q : Set ℝ := {x | x^2 + x - 6 > 0}

-- Define what it means for one set to be a sufficient condition for another
def is_sufficient_condition (A B : Set ℝ) : Prop := B ⊆ A

-- Define what it means for one set to be a necessary condition for another
def is_necessary_condition (A B : Set ℝ) : Prop := A ⊆ B

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  is_sufficient_condition (Set.univ \ p) (Set.univ \ q) ∧
  ¬ is_necessary_condition (Set.univ \ p) (Set.univ \ q) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1787_178774


namespace NUMINAMATH_CALUDE_marble_weight_proof_l1787_178784

/-- The weight of one marble in pounds -/
def marble_weight : ℚ := 100 / 9

/-- The weight of one waffle iron in pounds -/
def waffle_iron_weight : ℚ := 25

theorem marble_weight_proof :
  (9 * marble_weight = 4 * waffle_iron_weight) ∧
  (3 * waffle_iron_weight = 75) →
  marble_weight = 100 / 9 := by
sorry

end NUMINAMATH_CALUDE_marble_weight_proof_l1787_178784


namespace NUMINAMATH_CALUDE_square_point_configuration_l1787_178710

-- Define the square and points
def Square (A B C D : Point) : Prop := sorry

def OnSegment (P Q R : Point) : Prop := sorry

-- Define angle measurement
def AngleMeasure (P Q R : Point) : ℝ := sorry

-- Main theorem
theorem square_point_configuration 
  (A B C D M N P : Point) 
  (x : ℝ) 
  (h_square : Square A B C D)
  (h_M : OnSegment B M C)
  (h_N : OnSegment C N D)
  (h_P : OnSegment D P A)
  (h_angle_AM : AngleMeasure A B M = x)
  (h_angle_MN : AngleMeasure B C N = 2 * x)
  (h_angle_NP : AngleMeasure C D P = 3 * x)
  (h_x_range : 0 ≤ x ∧ x ≤ 22.5) :
  (∃! (M N P : Point), 
    OnSegment B M C ∧ 
    OnSegment C N D ∧ 
    OnSegment D P A ∧
    AngleMeasure A B M = x ∧
    AngleMeasure B C N = 2 * x ∧
    AngleMeasure C D P = 3 * x) ∧
  (∀ Q, OnSegment D Q A → ∃ x, 
    0 ≤ x ∧ x ≤ 22.5 ∧
    AngleMeasure A B M = x ∧
    AngleMeasure B C N = 2 * x ∧
    AngleMeasure C D P = 3 * x ∧
    Q = P) ∧
  (∃ S : Set ℝ, S.Infinite ∧ 
    ∀ y ∈ S, 0 ≤ y ∧ y ≤ 22.5 ∧ 
    AngleMeasure D A B = 4 * y) :=
sorry

end NUMINAMATH_CALUDE_square_point_configuration_l1787_178710


namespace NUMINAMATH_CALUDE_andy_final_position_l1787_178701

/-- Represents the position of Andy the Ant -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the direction Andy is facing -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Andy's state at any given moment -/
structure AntState :=
  (pos : Position)
  (dir : Direction)
  (moveCount : Nat)

/-- The movement function for Andy -/
def move (state : AntState) : AntState :=
  sorry

/-- The main theorem stating Andy's final position -/
theorem andy_final_position :
  let initialState : AntState :=
    { pos := { x := 30, y := -30 }
    , dir := Direction.North
    , moveCount := 0
    }
  let finalState := (move^[3030]) initialState
  finalState.pos = { x := 4573, y := -1546 } :=
sorry

end NUMINAMATH_CALUDE_andy_final_position_l1787_178701


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1787_178738

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 8 = 0) →
  (b^3 - 15*b^2 + 25*b - 8 = 0) →
  (c^3 - 15*c^2 + 25*c - 8 = 0) →
  (a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 175/9) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1787_178738


namespace NUMINAMATH_CALUDE_jonathan_weekly_deficit_l1787_178779

/-- Represents the days of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Jonathan's daily calorie intake -/
def calorie_intake (d : Day) : ℕ :=
  match d with
  | Day.Monday => 2500
  | Day.Tuesday => 2600
  | Day.Wednesday => 2400
  | Day.Thursday => 2700
  | Day.Friday => 2300
  | Day.Saturday => 3500
  | Day.Sunday => 2400

/-- Jonathan's daily calorie expenditure -/
def calorie_expenditure (d : Day) : ℕ :=
  match d with
  | Day.Monday => 3000
  | Day.Tuesday => 3200
  | Day.Wednesday => 2900
  | Day.Thursday => 3100
  | Day.Friday => 2800
  | Day.Saturday => 3000
  | Day.Sunday => 2700

/-- Calculate the daily caloric deficit -/
def daily_deficit (d : Day) : ℤ :=
  (calorie_expenditure d : ℤ) - (calorie_intake d : ℤ)

/-- The weekly caloric deficit -/
def weekly_deficit : ℤ :=
  (daily_deficit Day.Monday) +
  (daily_deficit Day.Tuesday) +
  (daily_deficit Day.Wednesday) +
  (daily_deficit Day.Thursday) +
  (daily_deficit Day.Friday) +
  (daily_deficit Day.Saturday) +
  (daily_deficit Day.Sunday)

/-- Theorem: Jonathan's weekly caloric deficit is 2800 calories -/
theorem jonathan_weekly_deficit : weekly_deficit = 2800 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_weekly_deficit_l1787_178779


namespace NUMINAMATH_CALUDE_min_value_of_ab_l1787_178721

theorem min_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 8) :
  a * b ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_ab_l1787_178721


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l1787_178731

theorem complex_square_one_plus_i (i : ℂ) : 
  i ^ 2 = -1 → (1 + i) ^ 2 = 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l1787_178731


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1787_178794

theorem complex_fraction_equality : (1 - Complex.I * Real.sqrt 3) / ((Real.sqrt 3 + Complex.I) ^ 2) = -1/4 - (Complex.I * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1787_178794


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1787_178766

theorem absolute_value_inequality (x : ℝ) : 
  |2*x - 1| - |x + 1| < 1 ↔ -1/3 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1787_178766


namespace NUMINAMATH_CALUDE_cost_of_500_sheets_l1787_178747

/-- The cost in dollars of a given number of sheets of paper. -/
def paper_cost (sheets : ℕ) : ℚ :=
  (sheets * 2 : ℚ) / 100

/-- Theorem stating that 500 sheets of paper cost $10.00. -/
theorem cost_of_500_sheets :
  paper_cost 500 = 10 := by sorry

end NUMINAMATH_CALUDE_cost_of_500_sheets_l1787_178747


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1787_178798

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0) → (-1 < m ∧ m < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1787_178798


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1787_178791

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with S₁₀ = 5 and a₇ = 1, a₁ = -1 -/
theorem arithmetic_sequence_first_term
  (seq : ArithmeticSequence)
  (h1 : seq.S 10 = 5)
  (h2 : seq.a 7 = 1) :
  seq.a 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1787_178791


namespace NUMINAMATH_CALUDE_tiffany_homework_l1787_178739

theorem tiffany_homework (math_pages : ℕ) (problems_per_page : ℕ) (total_problems : ℕ) 
  (h1 : math_pages = 6)
  (h2 : problems_per_page = 3)
  (h3 : total_problems = 30) :
  (total_problems - math_pages * problems_per_page) / problems_per_page = 4 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_homework_l1787_178739


namespace NUMINAMATH_CALUDE_unique_prime_pair_square_sum_l1787_178745

theorem unique_prime_pair_square_sum : 
  ∀ p q : ℕ, 
    Prime p → Prime q → p > 0 → q > 0 →
    (∃ n : ℕ, p^(q-1) + q^(p-1) = n^2) →
    p = 2 ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_square_sum_l1787_178745


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l1787_178750

theorem perfect_cube_units_digits : 
  ∃! (S : Finset ℕ), 
    (∀ n : ℕ, n ∈ S ↔ ∃ m : ℕ, m^3 % 10 = n) ∧ 
    S.card = 10 :=
sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l1787_178750


namespace NUMINAMATH_CALUDE_solution_set_empty_implies_a_range_main_theorem_l1787_178755

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + a * x + 3

-- State the theorem
theorem solution_set_empty_implies_a_range (a : ℝ) :
  (∀ x, f a x ≥ 0) → 0 ≤ a ∧ a ≤ 12 :=
by sorry

-- Define the range of a
def a_range : Set ℝ := {a | 0 ≤ a ∧ a ≤ 12}

-- State the main theorem
theorem main_theorem : 
  {a : ℝ | ∀ x, f a x ≥ 0} = a_range :=
by sorry

end NUMINAMATH_CALUDE_solution_set_empty_implies_a_range_main_theorem_l1787_178755


namespace NUMINAMATH_CALUDE_car_trip_speed_l1787_178797

theorem car_trip_speed (initial_speed initial_time total_speed total_time : ℝ) 
  (h1 : initial_speed = 45)
  (h2 : initial_time = 4)
  (h3 : total_speed = 65)
  (h4 : total_time = 12) :
  let remaining_time := total_time - initial_time
  let initial_distance := initial_speed * initial_time
  let total_distance := total_speed * total_time
  let remaining_distance := total_distance - initial_distance
  remaining_distance / remaining_time = 75 := by sorry

end NUMINAMATH_CALUDE_car_trip_speed_l1787_178797


namespace NUMINAMATH_CALUDE_total_people_in_program_l1787_178712

theorem total_people_in_program (parents : ℕ) (pupils : ℕ) 
  (h1 : parents = 22) (h2 : pupils = 654) : 
  parents + pupils = 676 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l1787_178712


namespace NUMINAMATH_CALUDE_initial_shoe_pairs_l1787_178795

/-- 
Given that a person loses 9 individual shoes and is left with a maximum of 20 matching pairs,
prove that the initial number of pairs of shoes was 25.
-/
theorem initial_shoe_pairs (lost_shoes : ℕ) (max_pairs_left : ℕ) : 
  lost_shoes = 9 →
  max_pairs_left = 20 →
  ∃ (initial_pairs : ℕ), initial_pairs = 25 ∧ 
    initial_pairs * 2 = max_pairs_left * 2 + lost_shoes :=
by sorry


end NUMINAMATH_CALUDE_initial_shoe_pairs_l1787_178795


namespace NUMINAMATH_CALUDE_total_notes_is_133_l1787_178770

/-- Calculates the sum of integers from 1 to n -/
def triangleSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the distribution of notes on the board -/
structure NoteDistribution where
  redRowCount : ℕ
  redPerRow : ℕ
  redScattered : ℕ
  blueRowCount : ℕ
  bluePerRow : ℕ
  blueScattered : ℕ
  greenTriangleBases : List ℕ
  yellowDiagonal1 : ℕ
  yellowDiagonal2 : ℕ
  yellowHexagon : ℕ

/-- Calculates the total number of notes based on the given distribution -/
def totalNotes (dist : NoteDistribution) : ℕ :=
  let redNotes := dist.redRowCount * dist.redPerRow + dist.redScattered
  let blueNotes := dist.blueRowCount * dist.bluePerRow + dist.blueScattered
  let greenNotes := (dist.greenTriangleBases.map triangleSum).sum
  let yellowNotes := dist.yellowDiagonal1 + dist.yellowDiagonal2 + dist.yellowHexagon
  redNotes + blueNotes + greenNotes + yellowNotes

/-- The actual distribution of notes on the board -/
def actualDistribution : NoteDistribution := {
  redRowCount := 5
  redPerRow := 6
  redScattered := 3
  blueRowCount := 4
  bluePerRow := 7
  blueScattered := 12
  greenTriangleBases := [4, 5, 6]
  yellowDiagonal1 := 5
  yellowDiagonal2 := 3
  yellowHexagon := 6
}

/-- Theorem stating that the total number of notes is 133 -/
theorem total_notes_is_133 : totalNotes actualDistribution = 133 := by
  sorry

end NUMINAMATH_CALUDE_total_notes_is_133_l1787_178770


namespace NUMINAMATH_CALUDE_clock_angle_at_3_30_clock_angle_at_3_30_is_75_l1787_178726

/-- The smaller angle between clock hands at 3:30 -/
theorem clock_angle_at_3_30 : ℝ :=
  let total_hours : ℕ := 12
  let total_degrees : ℝ := 360
  let degrees_per_hour : ℝ := total_degrees / total_hours
  let minute_hand_position : ℝ := 180
  let hour_hand_position : ℝ := 3 * degrees_per_hour + degrees_per_hour / 2
  let angle_difference : ℝ := |minute_hand_position - hour_hand_position|
  min angle_difference (total_degrees - angle_difference)

/-- Proof that the smaller angle between clock hands at 3:30 is 75 degrees -/
theorem clock_angle_at_3_30_is_75 : clock_angle_at_3_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_30_clock_angle_at_3_30_is_75_l1787_178726


namespace NUMINAMATH_CALUDE_batsman_highest_score_l1787_178749

def batting_problem (total_innings : ℕ) (overall_average : ℚ) (score_difference : ℕ) (average_excluding_extremes : ℚ) : Prop :=
  let total_runs := total_innings * overall_average
  let runs_excluding_extremes := (total_innings - 2) * average_excluding_extremes
  let sum_of_extremes := total_runs - runs_excluding_extremes
  let highest_score := (sum_of_extremes + score_difference) / 2
  highest_score = 199

theorem batsman_highest_score :
  batting_problem 46 60 190 58 := by sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l1787_178749


namespace NUMINAMATH_CALUDE_flower_count_l1787_178730

theorem flower_count (num_bees : ℕ) (num_flowers : ℕ) : 
  num_bees = 3 → num_bees = num_flowers - 2 → num_flowers = 5 := by
  sorry

end NUMINAMATH_CALUDE_flower_count_l1787_178730
