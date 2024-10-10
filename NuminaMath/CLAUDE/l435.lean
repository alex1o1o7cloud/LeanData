import Mathlib

namespace egg_weight_probability_l435_43555

theorem egg_weight_probability (p_less_than_30 p_30_to_40 : ℝ) 
  (h1 : p_less_than_30 = 0.30)
  (h2 : p_30_to_40 = 0.50) :
  1 - p_less_than_30 = 0.70 :=
by sorry

end egg_weight_probability_l435_43555


namespace evaluate_expression_l435_43564

theorem evaluate_expression : (800^2 : ℚ) / (300^2 - 296^2) = 640000 / 2384 := by
  sorry

end evaluate_expression_l435_43564


namespace workshop_selection_count_l435_43573

/-- The number of photography enthusiasts --/
def total_students : ℕ := 4

/-- The number of sessions in the workshop --/
def num_sessions : ℕ := 3

/-- The number of students who cannot participate in the first session --/
def restricted_students : ℕ := 2

/-- The number of different ways to select students for the workshop --/
def selection_methods : ℕ := (total_students - restricted_students) * (total_students - 1) * (total_students - 2)

theorem workshop_selection_count :
  selection_methods = 12 :=
sorry

end workshop_selection_count_l435_43573


namespace min_distance_to_line_l435_43559

/-- Given a line x + y + 1 = 0 in a 2D plane, the minimum distance from the point (-2, -3) to this line is 2√2 -/
theorem min_distance_to_line :
  ∀ x y : ℝ, x + y + 1 = 0 →
  (2 * Real.sqrt 2 : ℝ) ≤ Real.sqrt ((x + 2)^2 + (y + 3)^2) :=
by sorry

end min_distance_to_line_l435_43559


namespace work_completion_theorem_l435_43576

theorem work_completion_theorem (work : ℕ) (days1 days2 men1 : ℕ) 
  (h1 : work = men1 * days1)
  (h2 : work = 24 * (work / (men1 * days1) * men1))
  (h3 : men1 = 16)
  (h4 : days1 = 30)
  : work / (men1 * days1) * men1 = 20 := by
  sorry

end work_completion_theorem_l435_43576


namespace f_min_value_f_inequality_condition_l435_43587

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + abs (x - 2)

-- Theorem for the minimum value of f
theorem f_min_value : ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 7/4) :=
sorry

-- Theorem for the inequality condition
theorem f_inequality_condition (a b c : ℝ) :
  (∀ (x : ℝ), f x ≥ a^2 + 2*b^2 + 3*c^2) → a*c + 2*b*c ≤ 7/8 :=
sorry

end f_min_value_f_inequality_condition_l435_43587


namespace total_pumpkins_l435_43588

theorem total_pumpkins (sandy_pumpkins mike_pumpkins : ℕ) 
  (h1 : sandy_pumpkins = 51) 
  (h2 : mike_pumpkins = 23) : 
  sandy_pumpkins + mike_pumpkins = 74 := by
  sorry

end total_pumpkins_l435_43588


namespace greatest_divisible_power_of_three_l435_43521

theorem greatest_divisible_power_of_three (m : ℕ+) : 
  (∃ (k : ℕ), k = 2 ∧ (3^k : ℕ) ∣ (2^(3^m.val) + 1)) ∧
  (∀ (k : ℕ), k > 2 → ¬((3^k : ℕ) ∣ (2^(3^m.val) + 1))) :=
sorry

end greatest_divisible_power_of_three_l435_43521


namespace magical_stack_size_is_470_l435_43516

/-- A stack of cards is magical if it satisfies certain conditions --/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (retains_position : ℕ := 157)
  (is_magical : Prop)

/-- The number of cards in a magical stack where card 157 retains its position --/
def magical_stack_size (stack : MagicalStack) : ℕ := stack.total_cards

/-- Theorem stating the size of the magical stack --/
theorem magical_stack_size_is_470 (stack : MagicalStack) : 
  stack.retains_position = 157 → magical_stack_size stack = 470 := by
  sorry

#check magical_stack_size_is_470

end magical_stack_size_is_470_l435_43516


namespace range_of_a_l435_43583

def f (x a : ℝ) : ℝ := |2*x - a| + a

def g (x : ℝ) : ℝ := |2*x - 1|

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a + g x ≥ 2*a^2 - 13) → a ∈ Set.Icc (-Real.sqrt 7) 3 :=
by sorry

end range_of_a_l435_43583


namespace solution_range_l435_43511

-- Define the solution set A
def A : Set ℝ := {x | x^2 ≤ 5*x - 4}

-- Define the solution set M as a function of a
def M (a : ℝ) : Set ℝ := {x | (x - a) * (x - 2) ≤ 0}

-- State the theorem
theorem solution_range : 
  {a : ℝ | M a ⊆ A} = {a : ℝ | 1 ≤ a ∧ a ≤ 4} := by sorry

end solution_range_l435_43511


namespace monitoring_system_odd_agents_l435_43537

/-- Represents a cyclic monitoring system of agents -/
structure MonitoringSystem (n : ℕ) where
  -- The number of agents is positive
  agents_exist : 0 < n
  -- The monitoring function
  monitor : Fin n → Fin n
  -- The monitoring is cyclic
  cyclic : ∀ i : Fin n, monitor (monitor i) = i.succ

/-- Theorem: In a cyclic monitoring system, the number of agents is odd -/
theorem monitoring_system_odd_agents (n : ℕ) (sys : MonitoringSystem n) : 
  Odd n := by
  sorry


end monitoring_system_odd_agents_l435_43537


namespace coefficient_sum_l435_43508

theorem coefficient_sum (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end coefficient_sum_l435_43508


namespace complex_roots_of_equation_l435_43591

theorem complex_roots_of_equation : ∃ (z₁ z₂ : ℂ),
  z₁ = -1 + 2 * Real.sqrt 5 + (2 * Real.sqrt 5 / 5) * Complex.I ∧
  z₂ = -1 - 2 * Real.sqrt 5 - (2 * Real.sqrt 5 / 5) * Complex.I ∧
  z₁^2 + 2*z₁ = 16 + 8*Complex.I ∧
  z₂^2 + 2*z₂ = 16 + 8*Complex.I :=
by sorry

end complex_roots_of_equation_l435_43591


namespace exists_nonnegative_coeff_multiplier_l435_43536

/-- A polynomial with real coefficients that is positive for all nonnegative real numbers. -/
structure PositivePolynomial where
  P : Polynomial ℝ
  pos : ∀ x : ℝ, x ≥ 0 → P.eval x > 0

/-- The theorem stating that for any positive polynomial, there exists a positive integer n
    such that (1 + x)^n * P(x) has nonnegative coefficients. -/
theorem exists_nonnegative_coeff_multiplier (p : PositivePolynomial) :
  ∃ n : ℕ+, ∀ i : ℕ, ((1 + X : Polynomial ℝ)^(n : ℕ) * p.P).coeff i ≥ 0 := by
  sorry

end exists_nonnegative_coeff_multiplier_l435_43536


namespace sqrt_15_bounds_l435_43598

theorem sqrt_15_bounds : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 := by
  sorry

end sqrt_15_bounds_l435_43598


namespace intersection_count_is_four_l435_43541

/-- The number of intersection points between two curves -/
def intersection_count (C₁ C₂ : ℝ → ℝ → Prop) : ℕ :=
  sorry

/-- First curve: x² - y² + 4y - 3 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 - y^2 + 4*y - 3 = 0

/-- Second curve: y = ax², where a > 0 -/
def C₂ (a : ℝ) (x y : ℝ) : Prop :=
  y = a * x^2

/-- Theorem stating that the number of intersection points is 4 -/
theorem intersection_count_is_four (a : ℝ) (h : a > 0) :
  intersection_count C₁ (C₂ a) = 4 :=
sorry

end intersection_count_is_four_l435_43541


namespace elle_practice_time_l435_43546

/-- The number of minutes Elle practices piano on a weekday -/
def weekday_practice : ℕ := 30

/-- The number of weekdays Elle practices piano -/
def weekdays : ℕ := 5

/-- The factor by which Elle's Saturday practice is longer than a weekday -/
def saturday_factor : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the total number of hours Elle spends practicing piano each week -/
def total_practice_hours : ℚ :=
  let weekday_total := weekday_practice * weekdays
  let saturday_practice := weekday_practice * saturday_factor
  let total_minutes := weekday_total + saturday_practice
  (total_minutes : ℚ) / minutes_per_hour

theorem elle_practice_time : total_practice_hours = 4 := by
  sorry

end elle_practice_time_l435_43546


namespace second_player_wins_l435_43527

/-- Represents the game board -/
def Board := Fin 4 → Fin 2017 → Bool

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a position on the board -/
structure Position :=
  (row : Fin 4)
  (col : Fin 2017)

/-- Checks if a rook at the given position attacks an even number of other rooks -/
def attacksEven (board : Board) (pos : Position) : Bool :=
  sorry

/-- Checks if a rook at the given position attacks an odd number of other rooks -/
def attacksOdd (board : Board) (pos : Position) : Bool :=
  sorry

/-- Checks if the given move is valid for the current player -/
def isValidMove (board : Board) (player : Player) (pos : Position) : Prop :=
  match player with
  | Player.First => attacksEven board pos
  | Player.Second => attacksOdd board pos

/-- Represents a winning strategy for the second player -/
def secondPlayerStrategy (board : Board) (firstPlayerMove : Position) : Position :=
  sorry

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∀ (board : Board),
  ∀ (firstPlayerMove : Position),
  isValidMove board Player.First firstPlayerMove →
  isValidMove board Player.Second (secondPlayerStrategy board firstPlayerMove) :=
sorry

end second_player_wins_l435_43527


namespace two_and_one_third_symbiotic_neg_one_third_and_neg_two_symbiotic_symbiotic_pair_negation_l435_43597

-- Definition of symbiotic rational number pair
def is_symbiotic_pair (a b : ℚ) : Prop := a - b = a * b + 1

-- Theorem 1: (2, 1/3) is a symbiotic rational number pair
theorem two_and_one_third_symbiotic : is_symbiotic_pair 2 (1/3) := by sorry

-- Theorem 2: (-1/3, -2) is a symbiotic rational number pair
theorem neg_one_third_and_neg_two_symbiotic : is_symbiotic_pair (-1/3) (-2) := by sorry

-- Theorem 3: If (m, n) is a symbiotic rational number pair, then (-n, -m) is also a symbiotic rational number pair
theorem symbiotic_pair_negation (m n : ℚ) : 
  is_symbiotic_pair m n → is_symbiotic_pair (-n) (-m) := by sorry

end two_and_one_third_symbiotic_neg_one_third_and_neg_two_symbiotic_symbiotic_pair_negation_l435_43597


namespace line_tangent_to_parabola_l435_43506

/-- A parabola defined by y = 2x^2 -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2

/-- The point A on the parabola -/
def point_A : ℝ × ℝ := (-1, 2)

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop := 4 * x + y + 2 = 0

/-- Theorem stating that line l is tangent to the parabola at point A -/
theorem line_tangent_to_parabola :
  parabola (point_A.1) (point_A.2) ∧
  line_l (point_A.1) (point_A.2) ∧
  ∀ x y : ℝ, parabola x y ∧ line_l x y → (x, y) = point_A :=
sorry

end line_tangent_to_parabola_l435_43506


namespace tank_capacity_l435_43545

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 3/4 →
  added_amount = 8 →
  final_fraction = 7/8 →
  ∃ (total_capacity : Rat),
    initial_fraction * total_capacity + added_amount = final_fraction * total_capacity ∧
    total_capacity = 64 := by
  sorry

end tank_capacity_l435_43545


namespace percentage_calculation_l435_43503

theorem percentage_calculation : 
  (2 * (1/4 * (4/100))) + (3 * (15/100)) - (1/2 * (10/100)) = 0.42 := by
  sorry

end percentage_calculation_l435_43503


namespace custom_op_example_l435_43590

/-- Custom operation $\$$ defined for two integers -/
def custom_op (a b : Int) : Int := a * (b - 1) + a * b

/-- Theorem stating that 5 $\$$ (-3) = -35 -/
theorem custom_op_example : custom_op 5 (-3) = -35 := by
  sorry

end custom_op_example_l435_43590


namespace complement_of_A_l435_43522

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | (x + 2) / x < 0}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≥ 0 ∨ x ≤ -2} := by
  sorry

end complement_of_A_l435_43522


namespace mary_flour_calculation_l435_43501

/-- The number of cups of flour required by the recipe -/
def total_flour : ℕ := 7

/-- The number of cups of flour Mary has already added -/
def added_flour : ℕ := 2

/-- The number of cups of flour Mary needs to add -/
def flour_to_add : ℕ := total_flour - added_flour

theorem mary_flour_calculation :
  flour_to_add = 5 := by sorry

end mary_flour_calculation_l435_43501


namespace equal_passengers_after_changes_l435_43563

/-- Represents the number of passengers in a bus --/
structure BusPassengers where
  men : ℕ
  women : ℕ

/-- Calculates the total number of passengers --/
def BusPassengers.total (p : BusPassengers) : ℕ := p.men + p.women

/-- Represents the changes in passengers at a city --/
structure PassengerChanges where
  menLeaving : ℕ
  womenEntering : ℕ

/-- Applies changes to the passenger count --/
def applyChanges (p : BusPassengers) (c : PassengerChanges) : BusPassengers :=
  { men := p.men - c.menLeaving,
    women := p.women + c.womenEntering }

theorem equal_passengers_after_changes 
  (initialPassengers : BusPassengers)
  (changes : PassengerChanges) :
  initialPassengers.total = 72 →
  initialPassengers.women = initialPassengers.men / 2 →
  changes.menLeaving = 16 →
  changes.womenEntering = 8 →
  let finalPassengers := applyChanges initialPassengers changes
  finalPassengers.men = finalPassengers.women :=
by sorry

end equal_passengers_after_changes_l435_43563


namespace sarah_pencils_count_l435_43551

/-- The number of pencils Sarah buys on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah buys on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The number of pencils Sarah buys on Wednesday -/
def wednesday_pencils : ℕ := 3 * tuesday_pencils

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := monday_pencils + tuesday_pencils + wednesday_pencils

theorem sarah_pencils_count : total_pencils = 92 := by
  sorry

end sarah_pencils_count_l435_43551


namespace absolute_value_of_negative_four_squared_plus_six_l435_43512

theorem absolute_value_of_negative_four_squared_plus_six : 
  |(-4^2 + 6)| = 10 := by sorry

end absolute_value_of_negative_four_squared_plus_six_l435_43512


namespace barangay_speed_l435_43566

/-- Proves that the speed going to the barangay is 5 km/h given the problem conditions -/
theorem barangay_speed 
  (total_time : ℝ) 
  (distance : ℝ) 
  (rest_time : ℝ) 
  (return_speed : ℝ) 
  (h1 : total_time = 6)
  (h2 : distance = 7.5)
  (h3 : rest_time = 2)
  (h4 : return_speed = 3) : 
  distance / (total_time - rest_time - distance / return_speed) = 5 := by
sorry

end barangay_speed_l435_43566


namespace smallest_composite_no_small_factors_l435_43547

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 13 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 169 ∧ has_no_small_prime_factors 169) ∧ 
  (∀ m : ℕ, m < 169 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end smallest_composite_no_small_factors_l435_43547


namespace units_digit_sum_factorials_50_l435_43554

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials_50 : 
  units_digit (sum_factorials 50) = 3 := by sorry

end units_digit_sum_factorials_50_l435_43554


namespace sin_pi_plus_2alpha_l435_43509

theorem sin_pi_plus_2alpha (α : ℝ) (h : Real.sin (α - π/4) = 3/5) :
  Real.sin (π + 2*α) = -7/25 := by sorry

end sin_pi_plus_2alpha_l435_43509


namespace bilingual_point_part1_bilingual_points_part2_bilingual_point_part3_l435_43568

/-- Definition of a bilingual point -/
def is_bilingual_point (x y : ℝ) : Prop := y = 2 * x

/-- Part 1: Bilingual point of y = 3x + 1 -/
theorem bilingual_point_part1 : 
  ∃ x y : ℝ, is_bilingual_point x y ∧ y = 3 * x + 1 ∧ x = -1 ∧ y = -2 := by sorry

/-- Part 2: Bilingual points of y = k/x -/
theorem bilingual_points_part2 (k : ℝ) (h : k ≠ 0) :
  (∃ x y : ℝ, is_bilingual_point x y ∧ y = k / x) ↔ k > 0 := by sorry

/-- Part 3: Conditions for the function y = 1/4 * x^2 + (n-k-1)x + m+k+2 -/
theorem bilingual_point_part3 (n m k : ℝ) :
  (∃! x y : ℝ, is_bilingual_point x y ∧ 
    y = 1/4 * x^2 + (n - k - 1) * x + m + k + 2) ∧
  1 ≤ n ∧ n ≤ 3 ∧
  (∀ m' : ℝ, m' ≥ m → 
    ∃! x y : ℝ, is_bilingual_point x y ∧ 
      y = 1/4 * x^2 + (n - k - 1) * x + m' + k + 2) →
  k = 1 + Real.sqrt 3 ∨ k = -1 := by sorry

end bilingual_point_part1_bilingual_points_part2_bilingual_point_part3_l435_43568


namespace bill_equation_l435_43523

/-- Represents the monthly telephone bill calculation -/
def monthly_bill (rental_fee : ℝ) (per_call_cost : ℝ) (num_calls : ℝ) : ℝ :=
  rental_fee + per_call_cost * num_calls

/-- Theorem stating the relationship between monthly bill and number of calls -/
theorem bill_equation (x : ℝ) :
  monthly_bill 10 0.2 x = 10 + 0.2 * x :=
by sorry

end bill_equation_l435_43523


namespace interchange_relation_l435_43517

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The original number satisfies the given condition -/
def satisfies_condition (n : TwoDigitNumber) (c : Nat) : Prop :=
  10 * n.tens + n.ones = c * (n.tens + n.ones) + 3

/-- The number formed by interchanging digits -/
def interchange_digits (n : TwoDigitNumber) : Nat :=
  10 * n.ones + n.tens

/-- The main theorem to prove -/
theorem interchange_relation (n : TwoDigitNumber) (c : Nat) 
  (h : satisfies_condition n c) :
  interchange_digits n = (11 - c) * (n.tens + n.ones) := by
  sorry


end interchange_relation_l435_43517


namespace max_gcd_of_sequence_l435_43539

theorem max_gcd_of_sequence (n : ℕ+) :
  let a : ℕ+ → ℕ := fun k => 120 + k^2
  let d : ℕ+ → ℕ := fun k => Nat.gcd (a k) (a (k + 1))
  ∃ k : ℕ+, d k = 121 ∧ ∀ m : ℕ+, d m ≤ 121 :=
by sorry

end max_gcd_of_sequence_l435_43539


namespace expected_interval_is_three_minutes_l435_43585

/-- Represents the train system with given conditions -/
structure TrainSystem where
  northern_route_time : ℝ
  southern_route_time : ℝ
  arrival_time_difference : ℝ
  travel_time_difference : ℝ

/-- The expected interval between trains in one direction -/
def expected_interval (ts : TrainSystem) : ℝ :=
  3

/-- Theorem stating that the expected interval is 3 minutes -/
theorem expected_interval_is_three_minutes (ts : TrainSystem) 
  (h1 : ts.northern_route_time = 17)
  (h2 : ts.southern_route_time = 11)
  (h3 : ts.arrival_time_difference = 1.25)
  (h4 : ts.travel_time_difference = 1) :
  expected_interval ts = 3 := by
  sorry

#check expected_interval_is_three_minutes

end expected_interval_is_three_minutes_l435_43585


namespace product_of_smaller_numbers_l435_43567

theorem product_of_smaller_numbers (A B C : ℝ) : 
  B = 10 → 
  C - B = B - A → 
  B * C = 115 → 
  A * B = 85 := by
sorry

end product_of_smaller_numbers_l435_43567


namespace bees_second_day_l435_43534

def bees_first_day : ℕ := 144
def multiplier : ℕ := 3

theorem bees_second_day : bees_first_day * multiplier = 432 := by
  sorry

end bees_second_day_l435_43534


namespace matrix_power_sum_l435_43514

def A (a : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, 3, a],
    ![0, 1, 5],
    ![0, 0, 1]]

theorem matrix_power_sum (a : ℝ) (n : ℕ) :
  (A a)^n = ![![1, 27, 2883],
              ![0,  1,   45],
              ![0,  0,    1]] →
  a + n = 264 := by
  sorry

end matrix_power_sum_l435_43514


namespace b_2048_value_l435_43542

/-- A sequence of real numbers satisfying the given conditions -/
def special_sequence (b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≥ 2 → b n = b (n - 1) * b (n + 1)) ∧
  (b 1 = 3 + 2 * Real.sqrt 5) ∧
  (b 2023 = 23 + 10 * Real.sqrt 5)

/-- The theorem stating the value of b_2048 -/
theorem b_2048_value (b : ℕ → ℝ) (h : special_sequence b) :
  b 2048 = 19 + 6 * Real.sqrt 5 :=
sorry

end b_2048_value_l435_43542


namespace reciprocal_of_negative_three_l435_43553

theorem reciprocal_of_negative_three :
  ∀ x : ℚ, x * (-3) = 1 → x = -1/3 := by
  sorry

end reciprocal_of_negative_three_l435_43553


namespace work_completion_proof_l435_43549

/-- The number of days it takes p to complete the work alone -/
def p_days : ℕ := 80

/-- The number of days it takes q to complete the work alone -/
def q_days : ℕ := 48

/-- The total number of days the work lasted -/
def total_days : ℕ := 35

/-- The number of days after which q joined p -/
def q_join_day : ℕ := 8

/-- The work rate of p per day -/
def p_rate : ℚ := 1 / p_days

/-- The work rate of q per day -/
def q_rate : ℚ := 1 / q_days

/-- The total work completed is 1 (representing 100%) -/
def total_work : ℚ := 1

theorem work_completion_proof :
  p_rate * q_join_day + (p_rate + q_rate) * (total_days - q_join_day) = total_work :=
sorry

end work_completion_proof_l435_43549


namespace value_of_expression_l435_43538

theorem value_of_expression (a b x y : ℝ) 
  (h1 : a * x + b * y = 3) 
  (h2 : a * y - b * x = 5) : 
  (a^2 + b^2) * (x^2 + y^2) = 34 := by
sorry

end value_of_expression_l435_43538


namespace least_perimeter_triangle_l435_43525

theorem least_perimeter_triangle (d e f : ℕ) : 
  d > 0 → e > 0 → f > 0 →
  (d^2 + e^2 - f^2) / (2 * d * e : ℚ) = 24/25 →
  (d^2 + f^2 - e^2) / (2 * d * f : ℚ) = 3/5 →
  (e^2 + f^2 - d^2) / (2 * e * f : ℚ) = -2/5 →
  d + e + f ≥ 32 :=
by sorry

end least_perimeter_triangle_l435_43525


namespace rhombus_area_l435_43562

/-- The area of a rhombus with specific properties -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) (h_side : s = Real.sqrt 130) 
  (h_diag_diff : d₂ = d₁ + 4) (h_perp : d₁ * d₂ = 4 * s^2) : d₁ * d₂ / 2 = 126 := by
  sorry

end rhombus_area_l435_43562


namespace power_of_seven_mod_hundred_l435_43530

theorem power_of_seven_mod_hundred : ∃ (n : ℕ), n > 0 ∧ 7^n % 100 = 1 ∧ ∀ (k : ℕ), 0 < k → k < n → 7^k % 100 ≠ 1 :=
sorry

end power_of_seven_mod_hundred_l435_43530


namespace oranges_left_l435_43552

def initial_oranges : ℕ := 96
def taken_oranges : ℕ := 45

theorem oranges_left : initial_oranges - taken_oranges = 51 := by
  sorry

end oranges_left_l435_43552


namespace total_shares_sold_l435_43581

/-- Proves that the total number of shares sold is 300 given the specified conditions -/
theorem total_shares_sold (microtron_price dynaco_price avg_price : ℚ) (dynaco_shares : ℕ) : 
  microtron_price = 36 →
  dynaco_price = 44 →
  avg_price = 40 →
  dynaco_shares = 150 →
  ∃ (microtron_shares : ℕ), 
    (microtron_price * microtron_shares + dynaco_price * dynaco_shares) / (microtron_shares + dynaco_shares) = avg_price ∧
    microtron_shares + dynaco_shares = 300 := by
  sorry

end total_shares_sold_l435_43581


namespace square_perimeter_l435_43592

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 675 → perimeter = 60 * Real.sqrt 3 := by
  sorry

end square_perimeter_l435_43592


namespace bus_problem_l435_43531

/-- The number of students remaining on a bus after a given number of stops,
    where half the students get off at each stop. -/
def studentsRemaining (initial : ℕ) (stops : ℕ) : ℚ :=
  initial / (2 ^ stops)

/-- Theorem stating that if a bus starts with 48 students and half of the remaining
    students get off at each of three stops, then 6 students will remain after the third stop. -/
theorem bus_problem : studentsRemaining 48 3 = 6 := by
  sorry

end bus_problem_l435_43531


namespace quadratic_root_range_l435_43582

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - 1

theorem quadratic_root_range (a b : ℝ) :
  a > 0 →
  (∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  (∃ z : ℝ, 1 < z ∧ z < 2 ∧ f a b z = 0) →
  ∀ k : ℝ, -1 < k ∧ k < 1 ↔ ∃ a b : ℝ, a - b = k :=
by sorry

end quadratic_root_range_l435_43582


namespace min_value_reciprocal_plus_x_l435_43502

theorem min_value_reciprocal_plus_x (x : ℝ) (h : x > 0) : 
  4 / x + x ≥ 4 ∧ (4 / x + x = 4 ↔ x = 2) := by
  sorry

end min_value_reciprocal_plus_x_l435_43502


namespace rational_numbers_classification_l435_43593

theorem rational_numbers_classification (x : ℚ) : 
  ¬(∀ x : ℚ, x > 0 ∨ x < 0) :=
by
  sorry

end rational_numbers_classification_l435_43593


namespace expression_simplification_l435_43560

theorem expression_simplification :
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end expression_simplification_l435_43560


namespace parallel_line_equation_l435_43533

/-- A line passing through point (-2, 0) and parallel to 3x - y + 1 = 0 has equation y = 3x + 6 -/
theorem parallel_line_equation :
  let point : ℝ × ℝ := (-2, 0)
  let parallel_line (x y : ℝ) := 3 * x - y + 1 = 0
  let proposed_line (x y : ℝ) := y = 3 * x + 6
  (∀ x y, parallel_line x y ↔ y = 3 * x - 1) →
  (proposed_line point.1 point.2) ∧
  (∀ x₁ y₁ x₂ y₂, parallel_line x₁ y₁ → proposed_line x₂ y₂ →
    y₂ - y₁ = 3 * (x₂ - x₁)) :=
by sorry

end parallel_line_equation_l435_43533


namespace clock_second_sale_price_l435_43578

/-- Represents the clock sale scenario in the shop -/
structure ClockSale where
  originalCost : ℝ
  firstSalePrice : ℝ
  buyBackPrice : ℝ
  secondSalePrice : ℝ

/-- The conditions of the clock sale problem -/
def clockSaleProblem (sale : ClockSale) : Prop :=
  sale.firstSalePrice = 1.2 * sale.originalCost ∧
  sale.buyBackPrice = 0.5 * sale.firstSalePrice ∧
  sale.originalCost - sale.buyBackPrice = 100 ∧
  sale.secondSalePrice = sale.buyBackPrice * 1.8

/-- The theorem stating that under the given conditions, 
    the second sale price is 270 -/
theorem clock_second_sale_price (sale : ClockSale) :
  clockSaleProblem sale → sale.secondSalePrice = 270 := by
  sorry

end clock_second_sale_price_l435_43578


namespace complete_square_constant_l435_43580

theorem complete_square_constant (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end complete_square_constant_l435_43580


namespace sin_value_for_specific_tan_l435_43518

/-- Prove that for an acute angle α, if tan(π - α) + 3 = 0, then sinα = 3√10 / 10 -/
theorem sin_value_for_specific_tan (α : Real) : 
  0 < α ∧ α < π / 2 →  -- α is an acute angle
  Real.tan (π - α) + 3 = 0 → 
  Real.sin α = 3 * Real.sqrt 10 / 10 := by
  sorry

end sin_value_for_specific_tan_l435_43518


namespace coefficient_sum_l435_43565

-- Define the sets A and B
def A : Set ℝ := {x | x^3 + 3*x^2 + 2*x > 0}
def B : Set ℝ := {x | ∃ (a b : ℝ), x^2 + a*x + b ≤ 0}

-- Define the intersection and union of A and B
def intersection : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def union : Set ℝ := {x | x > -2}

-- State the theorem
theorem coefficient_sum (a b : ℝ) : 
  A ∩ B = intersection → A ∪ B = union → a + b = -3 := by sorry

end coefficient_sum_l435_43565


namespace percentage_difference_l435_43519

theorem percentage_difference (p t j : ℝ) : 
  t = 0.9375 * p →  -- t is 6.25% less than p
  j = 0.8 * t →     -- j is 20% less than t
  j = 0.75 * p :=   -- j is 25% less than p
by sorry

end percentage_difference_l435_43519


namespace volume_of_rotated_solid_l435_43579

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop := |x/3| + |y/3| = 2

/-- The region enclosed by the equation -/
def enclosed_region : Set (ℝ × ℝ) := {p : ℝ × ℝ | region_equation p.1 p.2}

/-- The volume of the solid generated by rotating the region around the x-axis -/
noncomputable def rotation_volume : ℝ := sorry

/-- Theorem stating that the volume of the rotated solid is equal to some value V -/
theorem volume_of_rotated_solid :
  ∃ V, rotation_volume = V :=
sorry

end volume_of_rotated_solid_l435_43579


namespace rectangle_validity_l435_43574

/-- A rectangle is valid if its area is less than or equal to the square of a quarter of its perimeter. -/
theorem rectangle_validity (S l : ℝ) (h_pos : S > 0 ∧ l > 0) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = S ∧ 2 * (x + y) = l) ↔ S ≤ (l / 4)^2 := by
sorry

end rectangle_validity_l435_43574


namespace lawn_width_calculation_l435_43505

/-- Calculates the width of a rectangular lawn given specific conditions. -/
theorem lawn_width_calculation (length width road_width cost_per_sqm total_cost : ℝ) 
  (h1 : length = 80)
  (h2 : road_width = 10)
  (h3 : cost_per_sqm = 3)
  (h4 : total_cost = 3900)
  (h5 : (road_width * width + road_width * length - road_width * road_width) * cost_per_sqm = total_cost) :
  width = 60 := by
sorry

end lawn_width_calculation_l435_43505


namespace cosine_sine_inequality_l435_43529

theorem cosine_sine_inequality (x : ℝ) : 
  (1 / 4 : ℝ) ≤ (Real.cos x)^6 + (Real.sin x)^6 ∧ (Real.cos x)^6 + (Real.sin x)^6 ≤ 1 :=
by
  sorry

#check cosine_sine_inequality

end cosine_sine_inequality_l435_43529


namespace net_difference_in_expenditure_l435_43504

/-- Represents the problem of calculating the net difference in expenditure after a price increase --/
theorem net_difference_in_expenditure
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percentage : ℝ)
  (budget : ℝ)
  (purchased_percentage : ℝ)
  (h1 : price_increase_percentage = 0.25)
  (h2 : budget = 150)
  (h3 : purchased_percentage = 0.64)
  (h4 : original_price * original_quantity = budget)
  (h5 : original_quantity ≤ 40) :
  original_price * original_quantity - (original_price * (1 + price_increase_percentage)) * (purchased_percentage * original_quantity) = 30 :=
by sorry

end net_difference_in_expenditure_l435_43504


namespace fraction_sign_l435_43594

theorem fraction_sign (a b : ℝ) (ha : a > 0) (hb : b < 0) : a / b < 0 := by
  sorry

end fraction_sign_l435_43594


namespace sequence_contradiction_l435_43520

theorem sequence_contradiction (s : Finset ℕ) (h1 : s.card = 5) 
  (h2 : 2 ∈ s) (h3 : 35 ∈ s) (h4 : 26 ∈ s) (h5 : ∃ x ∈ s, ∀ y ∈ s, y ≤ x) 
  (h6 : ∀ x ∈ s, x ≤ 25) : False := by
  sorry

end sequence_contradiction_l435_43520


namespace point_coordinates_l435_43548

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian coordinate system -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance of a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point in the fourth quadrant with distances 3 and 5 to x-axis and y-axis respectively has coordinates (5, -3) -/
theorem point_coordinates (p : Point) 
  (h1 : fourth_quadrant p) 
  (h2 : distance_to_x_axis p = 3) 
  (h3 : distance_to_y_axis p = 5) : 
  p = Point.mk 5 (-3) := by
  sorry

end point_coordinates_l435_43548


namespace blue_paper_side_length_l435_43513

theorem blue_paper_side_length (red_side : ℝ) (blue_side1 : ℝ) (blue_side2 : ℝ) : 
  red_side = 5 →
  blue_side1 = 4 →
  red_side * red_side = blue_side1 * blue_side2 →
  blue_side2 = 6.25 := by
  sorry

end blue_paper_side_length_l435_43513


namespace daily_sales_extrema_l435_43589

-- Define the sales volume function
def g (t : ℝ) : ℝ := 80 - 2 * t

-- Define the price function
def f (t : ℝ) : ℝ := 20 - abs (t - 10)

-- Define the daily sales function
def y (t : ℝ) : ℝ := g t * f t

-- Theorem statement
theorem daily_sales_extrema :
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 20 → y t ≤ 1200) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 20 ∧ y t = 1200) ∧
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 20 → y t ≥ 400) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 20 ∧ y t = 400) :=
by sorry

end daily_sales_extrema_l435_43589


namespace history_books_shelved_l435_43528

theorem history_books_shelved (total_books : ℕ) (romance_books : ℕ) (poetry_books : ℕ)
  (western_books : ℕ) (biography_books : ℕ) :
  total_books = 46 →
  romance_books = 8 →
  poetry_books = 4 →
  western_books = 5 →
  biography_books = 6 →
  ∃ (history_books : ℕ) (mystery_books : ℕ),
    history_books = 12 ∧
    mystery_books = western_books + biography_books ∧
    total_books = history_books + romance_books + poetry_books + western_books + biography_books + mystery_books :=
by
  sorry

end history_books_shelved_l435_43528


namespace sin_product_equals_one_eighth_l435_43558

theorem sin_product_equals_one_eighth : 
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * 
  Real.sin (72 * π / 180) * Real.sin (84 * π / 180) = 1 / 8 := by
  sorry

end sin_product_equals_one_eighth_l435_43558


namespace number_ratio_and_sum_of_squares_l435_43526

theorem number_ratio_and_sum_of_squares (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  x / y = 2 / (3/2) → x^2 + y^2 = 400 → x = 16 ∧ y = 12 := by
  sorry

end number_ratio_and_sum_of_squares_l435_43526


namespace calculation_proof_l435_43596

theorem calculation_proof : (1/4) * 6.16^2 - 4 * 1.04^2 = 5.16 := by
  sorry

end calculation_proof_l435_43596


namespace vector_sum_equality_l435_43550

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- For any three points A, B, and C in a vector space, 
    the sum of vectors AB, BC, and BA equals vector BC. -/
theorem vector_sum_equality (A B C : V) : 
  (B - A) + (C - B) + (A - B) = C - B := by sorry

end vector_sum_equality_l435_43550


namespace manuscript_fee_proof_l435_43510

/-- Calculates the tax payable for manuscript income not exceeding 4000 yuan -/
def tax_payable (income : ℝ) : ℝ := (income - 800) * 0.2 * 0.7

/-- The manuscript fee before tax deduction -/
def manuscript_fee : ℝ := 2800

theorem manuscript_fee_proof :
  manuscript_fee ≤ 4000 ∧
  tax_payable manuscript_fee = 280 :=
sorry

end manuscript_fee_proof_l435_43510


namespace sum_of_solutions_eq_23_20_l435_43532

theorem sum_of_solutions_eq_23_20 : 
  let f : ℝ → ℝ := λ x => (5*x + 3) * (4*x - 7)
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ x + y = 23/20) := by
sorry

end sum_of_solutions_eq_23_20_l435_43532


namespace stock_purchase_problem_l435_43515

/-- Mr. Wise's stock purchase problem -/
theorem stock_purchase_problem (total_value : ℝ) (price_type1 : ℝ) (total_shares : ℕ) (shares_type1 : ℕ) :
  total_value = 1950 →
  price_type1 = 3 →
  total_shares = 450 →
  shares_type1 = 400 →
  ∃ (price_type2 : ℝ),
    price_type2 * (total_shares - shares_type1) + price_type1 * shares_type1 = total_value ∧
    price_type2 = 15 :=
by sorry

end stock_purchase_problem_l435_43515


namespace sodas_drunk_equals_three_l435_43572

/-- The number of sodas Robin bought -/
def total_sodas : ℕ := 11

/-- The number of sodas left after drinking -/
def extras : ℕ := 8

/-- The number of sodas drunk -/
def sodas_drunk : ℕ := total_sodas - extras

theorem sodas_drunk_equals_three : sodas_drunk = 3 := by
  sorry

end sodas_drunk_equals_three_l435_43572


namespace largest_four_digit_divisible_by_6_l435_43595

def is_divisible_by_2 (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_6 (n : ℕ) : Prop := is_divisible_by_2 n ∧ is_divisible_by_3 n

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

theorem largest_four_digit_divisible_by_6 :
  ∀ n : ℕ, is_four_digit n → is_divisible_by_6 n → n ≤ 9996 :=
by sorry

end largest_four_digit_divisible_by_6_l435_43595


namespace lemon_bags_count_l435_43575

/-- The maximum load of the truck in kilograms -/
def max_load : ℕ := 900

/-- The mass of one bag of lemons in kilograms -/
def bag_mass : ℕ := 8

/-- The remaining capacity of the truck in kilograms -/
def remaining_capacity : ℕ := 100

/-- The number of bags of lemons on the truck -/
def num_bags : ℕ := (max_load - remaining_capacity) / bag_mass

theorem lemon_bags_count : num_bags = 100 := by
  sorry

end lemon_bags_count_l435_43575


namespace min_draw_correct_l435_43540

/-- The total number of balls in the bag -/
def total_balls : ℕ := 70

/-- The number of red balls in the bag -/
def red_balls : ℕ := 20

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 20

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 20

/-- The minimum number of balls that must be drawn to ensure at least 10 balls of one color -/
def min_draw : ℕ := 38

theorem min_draw_correct : 
  ∀ (draw : ℕ), draw ≥ min_draw → 
  ∃ (color : ℕ), color ≥ 10 ∧ 
  (color ≤ red_balls ∨ color ≤ blue_balls ∨ color ≤ yellow_balls ∨ color ≤ total_balls - red_balls - blue_balls - yellow_balls) :=
by sorry

end min_draw_correct_l435_43540


namespace train_crossing_time_l435_43561

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 50 ∧ train_speed_kmh = 60 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 3 := by
  sorry

end train_crossing_time_l435_43561


namespace tank_filling_time_l435_43577

theorem tank_filling_time (a b c : ℝ) (h1 : c = 2 * b) (h2 : b = 2 * a) (h3 : a + b + c = 1 / 8) :
  1 / a = 56 := by
  sorry

end tank_filling_time_l435_43577


namespace problem_1_l435_43544

theorem problem_1 : 4 * Real.sin (π / 3) + (1 / 3)⁻¹ + |-2| - Real.sqrt 12 = 5 := by
  sorry

end problem_1_l435_43544


namespace max_subjects_per_teacher_l435_43557

theorem max_subjects_per_teacher 
  (total_subjects : Nat) 
  (min_teachers : Nat) 
  (maths_teachers : Nat) 
  (physics_teachers : Nat) 
  (chemistry_teachers : Nat) 
  (h1 : total_subjects = maths_teachers + physics_teachers + chemistry_teachers)
  (h2 : maths_teachers = 4)
  (h3 : physics_teachers = 3)
  (h4 : chemistry_teachers = 3)
  (h5 : min_teachers = 5)
  : (total_subjects / min_teachers : Nat) = 2 := by
  sorry

end max_subjects_per_teacher_l435_43557


namespace inequality_solution_and_function_property_l435_43500

def f (x : ℝ) := |x - 1|

theorem inequality_solution_and_function_property :
  (∃ (S : Set ℝ), S = {x : ℝ | x ≤ -10/3 ∨ x ≥ 2} ∧
    ∀ x, x ∈ S ↔ f (2*x) + f (x + 4) ≥ 8) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → a ≠ 0 → f (a*b) / |a| > f (b/a)) :=
by sorry

end inequality_solution_and_function_property_l435_43500


namespace average_weight_increase_l435_43599

/-- Proves that the average weight increase is 200 grams when a 45 kg student leaves a group of 60 students, resulting in a new average of 57 kg for the remaining 59 students. -/
theorem average_weight_increase (initial_count : ℕ) (left_weight : ℝ) (remaining_count : ℕ) (new_average : ℝ) : 
  initial_count = 60 → 
  left_weight = 45 → 
  remaining_count = 59 → 
  new_average = 57 → 
  (new_average - (initial_count * new_average - left_weight) / initial_count) * 1000 = 200 := by
  sorry

#check average_weight_increase

end average_weight_increase_l435_43599


namespace prime_divisibility_l435_43586

theorem prime_divisibility (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r →
  (pqr : ℕ) = p * q * r →
  (pqr ∣ (p * q)^r + (q * r)^p + (r * p)^q - 1) →
  ((pqr)^3 ∣ 3 * ((p * q)^r + (q * r)^p + (r * p)^q - 1)) := by
sorry

end prime_divisibility_l435_43586


namespace jack_buttons_theorem_l435_43524

/-- The number of buttons Jack must use for all shirts -/
def total_buttons (num_kids : ℕ) (shirts_per_kid : ℕ) (buttons_per_shirt : ℕ) : ℕ :=
  num_kids * shirts_per_kid * buttons_per_shirt

/-- Theorem stating the total number of buttons Jack must use -/
theorem jack_buttons_theorem :
  total_buttons 3 3 7 = 63 := by
  sorry

end jack_buttons_theorem_l435_43524


namespace matt_keychains_purchase_l435_43570

/-- The number of key chains Matt buys -/
def num_keychains : ℕ := 10

/-- The price of a pack of 10 key chains -/
def price_pack_10 : ℚ := 20

/-- The price of a pack of 4 key chains -/
def price_pack_4 : ℚ := 12

/-- The amount Matt saves by choosing the cheaper option -/
def savings : ℚ := 20

theorem matt_keychains_purchase :
  num_keychains = 10 ∧
  (num_keychains : ℚ) * (price_pack_10 / 10) = 
    (num_keychains : ℚ) * (price_pack_4 / 4) - savings :=
by sorry

end matt_keychains_purchase_l435_43570


namespace canoe_rental_cost_l435_43569

/-- The cost of renting a canoe per day -/
def canoe_cost : ℚ := 9

/-- The cost of renting a kayak per day -/
def kayak_cost : ℚ := 12

/-- The ratio of canoes to kayaks rented -/
def canoe_kayak_ratio : ℚ := 4/3

/-- The number of additional canoes compared to kayaks -/
def additional_canoes : ℕ := 6

/-- The total revenue for the day -/
def total_revenue : ℚ := 432

theorem canoe_rental_cost :
  let kayaks : ℕ := 18
  let canoes : ℕ := kayaks + additional_canoes
  canoe_cost * canoes + kayak_cost * kayaks = total_revenue ∧
  (canoes : ℚ) / kayaks = canoe_kayak_ratio :=
by sorry

end canoe_rental_cost_l435_43569


namespace complex_equation_solution_l435_43571

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_equation_solution (z : ℂ) (a : ℝ) (h1 : is_pure_imaginary z) 
  (h2 : (2 - Complex.I) * z = a + Complex.I) : a = 1/2 := by
  sorry

end complex_equation_solution_l435_43571


namespace quadratic_root_implies_c_value_l435_43507

theorem quadratic_root_implies_c_value (c : ℝ) :
  (∀ x : ℝ, (3/2) * x^2 + 11*x + c = 0 ↔ x = (-11 + Real.sqrt 7) / 3 ∨ x = (-11 - Real.sqrt 7) / 3) →
  c = 19 := by
  sorry

end quadratic_root_implies_c_value_l435_43507


namespace amy_tips_calculation_l435_43556

/-- Calculates the amount of tips earned by Amy given her hourly wage, hours worked, and total earnings. -/
theorem amy_tips_calculation (hourly_wage : ℝ) (hours_worked : ℝ) (total_earnings : ℝ) : 
  hourly_wage = 2 → hours_worked = 7 → total_earnings = 23 → 
  total_earnings - (hourly_wage * hours_worked) = 9 := by
  sorry

end amy_tips_calculation_l435_43556


namespace equation_solution_l435_43543

theorem equation_solution : ∃ x : ℚ, (4/7 : ℚ) * (2/5 : ℚ) * x = 8 ∧ x = 35 := by
  sorry

end equation_solution_l435_43543


namespace paint_cube_cost_l435_43535

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem paint_cube_cost 
  (paint_cost : ℝ)        -- Cost of paint per kg in Rs
  (paint_coverage : ℝ)    -- Area covered by 1 kg of paint in sq. ft
  (cube_side : ℝ)         -- Length of cube side in feet
  (h1 : paint_cost = 20)  -- Paint costs Rs. 20 per kg
  (h2 : paint_coverage = 15) -- 1 kg of paint covers 15 sq. ft
  (h3 : cube_side = 5)    -- Cube has sides of 5 feet
  : ℝ :=
by
  -- The proof would go here
  sorry

#check paint_cube_cost

end paint_cube_cost_l435_43535


namespace bus_journey_distance_l435_43584

/-- Given a bus journey with two different speeds, prove the distance covered at the lower speed. -/
theorem bus_journey_distance (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5)
  (h5 : total_time = (distance1 / speed1) + ((total_distance - distance1) / speed2)) :
  distance1 = 100 := by sorry


end bus_journey_distance_l435_43584
