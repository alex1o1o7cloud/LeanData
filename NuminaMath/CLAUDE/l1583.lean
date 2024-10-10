import Mathlib

namespace container_volume_l1583_158341

theorem container_volume (x y z : ℝ) 
  (h_order : x < y ∧ y < z)
  (h_xy : 5 * x * y = 120)
  (h_xz : 3 * x * z = 120)
  (h_yz : 2 * y * z = 120) :
  x * y * z = 240 := by
sorry

end container_volume_l1583_158341


namespace candle_count_l1583_158313

/-- The number of candles Alex used -/
def used_candles : ℕ := 32

/-- The number of candles Alex has left -/
def leftover_candles : ℕ := 12

/-- The total number of candles Alex had initially -/
def initial_candles : ℕ := used_candles + leftover_candles

theorem candle_count : initial_candles = 44 := by
  sorry

end candle_count_l1583_158313


namespace range_of_a_l1583_158384

-- Define the system of inequalities
def inequality_system (x a : ℝ) : Prop :=
  3 * x - a > x + 1 ∧ (3 * x - 2) / 2 < 1 + x

-- Define the condition of having exactly 3 integer solutions
def has_three_integer_solutions (a : ℝ) : Prop :=
  ∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x ∈ s, inequality_system x a

-- The main theorem
theorem range_of_a (a : ℝ) :
  has_three_integer_solutions a → -1 ≤ a ∧ a < 1 :=
by sorry

end range_of_a_l1583_158384


namespace prime_factorization_sum_l1583_158338

theorem prime_factorization_sum (a b c : ℕ) : 
  2^a * 3^b * 7^c = 432 → a + b + c = 5 → 3*a + 2*b + 4*c = 18 := by
sorry

end prime_factorization_sum_l1583_158338


namespace circle_circumference_limit_l1583_158382

open Real

theorem circle_circumference_limit (C : ℝ) (h : C > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * π * (C / n) - C| < ε :=
by sorry

end circle_circumference_limit_l1583_158382


namespace fraction_sum_integer_implies_fractions_integer_l1583_158354

theorem fraction_sum_integer_implies_fractions_integer
  (a b c : ℤ) (h : ∃ (m : ℤ), (a * b) / c + (a * c) / b + (b * c) / a = m) :
  (∃ (k : ℤ), (a * b) / c = k) ∧
  (∃ (l : ℤ), (a * c) / b = l) ∧
  (∃ (n : ℤ), (b * c) / a = n) :=
by sorry

end fraction_sum_integer_implies_fractions_integer_l1583_158354


namespace xiaoliang_draw_probability_l1583_158349

/-- Represents the labels of balls in the box -/
inductive Label : Type
| one : Label
| two : Label
| three : Label
| four : Label

/-- The state of the box after Xiaoming's draw -/
structure BoxState :=
  (remaining_two : Nat)
  (remaining_three : Nat)
  (remaining_four : Nat)

/-- The initial state of the box -/
def initial_box : BoxState :=
  { remaining_two := 2
  , remaining_three := 1
  , remaining_four := 2 }

/-- The total number of balls remaining in the box -/
def total_remaining (box : BoxState) : Nat :=
  box.remaining_two + box.remaining_three + box.remaining_four

/-- The probability of drawing a ball with a specific label -/
def prob_draw (box : BoxState) (label : Label) : Rat :=
  match label with
  | Label.one => 0  -- No balls labeled 1 remaining
  | Label.two => box.remaining_two / (total_remaining box)
  | Label.three => box.remaining_three / (total_remaining box)
  | Label.four => box.remaining_four / (total_remaining box)

/-- The probability of drawing a ball matching Xiaoming's drawn balls -/
def prob_match_xiaoming (box : BoxState) : Rat :=
  prob_draw box Label.three

theorem xiaoliang_draw_probability :
  prob_match_xiaoming initial_box = 1/5 := by
  sorry

end xiaoliang_draw_probability_l1583_158349


namespace tournament_27_teams_26_games_l1583_158302

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- Number of games needed to determine a winner in a single-elimination tournament -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: A single-elimination tournament with 27 teams requires 26 games to determine a winner -/
theorem tournament_27_teams_26_games :
  ∀ (t : Tournament), t.num_teams = 27 → t.no_ties = true → games_to_winner t = 26 := by
  sorry

end tournament_27_teams_26_games_l1583_158302


namespace basketball_game_third_quarter_score_l1583_158355

/-- Represents the points scored by a team in each quarter -/
structure TeamScore :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if a TeamScore follows a geometric sequence -/
def isGeometric (s : TeamScore) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if a TeamScore follows an arithmetic sequence -/
def isArithmetic (s : TeamScore) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a TeamScore -/
def totalScore (s : TeamScore) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

theorem basketball_game_third_quarter_score :
  ∀ (teamA teamB : TeamScore),
    teamA.q1 = teamB.q1 →                        -- Tied at the end of first quarter
    isGeometric teamA →                          -- Team A follows geometric sequence
    isArithmetic teamB →                         -- Team B follows arithmetic sequence
    totalScore teamA = totalScore teamB + 3 →    -- Team A wins by 3 points
    totalScore teamA ≤ 100 →                     -- Team A's total score ≤ 100
    totalScore teamB ≤ 100 →                     -- Team B's total score ≤ 100
    teamA.q3 + teamB.q3 = 60                     -- Total score in third quarter is 60
  := by sorry

end basketball_game_third_quarter_score_l1583_158355


namespace rabbit_exchange_l1583_158389

/-- The exchange problem between two rabbits --/
theorem rabbit_exchange (white_carrots gray_cabbages : ℕ) 
  (h1 : white_carrots = 180) 
  (h2 : gray_cabbages = 120) : 
  ∃ (x : ℕ), x > 0 ∧ x < gray_cabbages ∧ 
  (gray_cabbages - x + 3 * x = (white_carrots + gray_cabbages) / 2) ∧
  (white_carrots - 3 * x + x = (white_carrots + gray_cabbages) / 2) := by
sorry

#eval (180 + 120) / 2  -- Expected output: 150

end rabbit_exchange_l1583_158389


namespace deal_or_no_deal_probability_l1583_158368

def box_values : List ℕ := [10, 50, 100, 500, 1000, 5000, 50000, 75000, 200000, 400000, 500000, 1000000]

def total_boxes : ℕ := 16

def high_value_boxes : ℕ := (box_values.filter (λ x => x ≥ 500000)).length

theorem deal_or_no_deal_probability (boxes_to_eliminate : ℕ) :
  boxes_to_eliminate = 10 ↔ 
  (high_value_boxes : ℚ) / (total_boxes - boxes_to_eliminate : ℚ) ≥ 1/2 ∧
  ∀ n : ℕ, n < boxes_to_eliminate → 
    (high_value_boxes : ℚ) / (total_boxes - n : ℚ) < 1/2 :=
sorry

end deal_or_no_deal_probability_l1583_158368


namespace root_in_interval_l1583_158332

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  ∃ r ∈ Set.Ioo 2 2.5, f r = 0 :=
by
  sorry

end root_in_interval_l1583_158332


namespace black_friday_tv_sales_increase_black_friday_tv_sales_increase_proof_l1583_158381

theorem black_friday_tv_sales_increase : ℕ → Prop :=
  fun increase =>
    ∃ (T : ℕ),
      T + increase = 327 ∧
      T + 3 * increase = 477 ∧
      increase = 75

-- The proof would go here, but we'll use sorry as instructed
theorem black_friday_tv_sales_increase_proof :
  ∃ increase, black_friday_tv_sales_increase increase :=
by sorry

end black_friday_tv_sales_increase_black_friday_tv_sales_increase_proof_l1583_158381


namespace hyperbola_asymptotes_l1583_158395

theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := fun x y => x^2 / 4 - y^2 / 9 = 1
  ∀ x y : ℝ, (∃ t : ℝ, t ≠ 0 ∧ h (t * x) (t * y)) ↔ y = (3/2) * x ∨ y = -(3/2) * x :=
sorry

end hyperbola_asymptotes_l1583_158395


namespace share_division_l1583_158324

theorem share_division (total : ℚ) (a b c : ℚ) 
  (h1 : total = 595)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) :
  c = 420 := by sorry

end share_division_l1583_158324


namespace fourth_game_shots_correct_l1583_158328

-- Define the initial conditions
def initial_shots : ℕ := 45
def initial_made : ℕ := 18
def initial_average : ℚ := 40 / 100
def fourth_game_shots : ℕ := 15
def new_average : ℚ := 55 / 100

-- Define the function to calculate the number of shots made in the fourth game
def fourth_game_made : ℕ := 15

-- Theorem statement
theorem fourth_game_shots_correct :
  (initial_made + fourth_game_made : ℚ) / (initial_shots + fourth_game_shots) = new_average :=
by sorry

end fourth_game_shots_correct_l1583_158328


namespace gcd_of_squares_sum_l1583_158345

theorem gcd_of_squares_sum : Nat.gcd (12^2 + 23^2 + 34^2) (13^2 + 22^2 + 35^2) = 1 := by
  sorry

end gcd_of_squares_sum_l1583_158345


namespace no_solution_exists_l1583_158325

theorem no_solution_exists :
  ¬∃ (B C : ℕ+), 
    (Nat.lcm 360 (Nat.lcm B C) = 55440) ∧ 
    (Nat.gcd 360 (Nat.gcd B C) = 15) ∧ 
    (B * C = 2316) := by
  sorry

end no_solution_exists_l1583_158325


namespace integral_sqrt_plus_x_plus_x_cubed_l1583_158306

theorem integral_sqrt_plus_x_plus_x_cubed (f : ℝ → ℝ) :
  (∫ x in (0)..(1), (Real.sqrt (1 - x^2) + x + x^3)) = (π + 3) / 4 := by
  sorry

end integral_sqrt_plus_x_plus_x_cubed_l1583_158306


namespace claire_photos_l1583_158356

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 24) :
  claire = 12 := by
  sorry

end claire_photos_l1583_158356


namespace max_q_minus_r_for_1073_l1583_158310

theorem max_q_minus_r_for_1073 :
  ∃ (q r : ℕ+), 1073 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1073 = 23 * q' + r' → q - r ≥ q' - r' :=
by
  sorry

end max_q_minus_r_for_1073_l1583_158310


namespace segment_ratio_l1583_158375

/-- Given two line segments a and b, where a is 2 meters and b is 40 centimeters,
    prove that the ratio of a to b is 5:1. -/
theorem segment_ratio (a b : ℝ) : a = 2 → b = 40 / 100 → a / b = 5 / 1 := by
  sorry

end segment_ratio_l1583_158375


namespace seven_times_coefficient_polynomials_l1583_158344

theorem seven_times_coefficient_polynomials (m n : ℤ) : 
  (∃ k : ℤ, 4 * m - n = 7 * k) → (∃ l : ℤ, 2 * m + 3 * n = 7 * l) := by
  sorry

end seven_times_coefficient_polynomials_l1583_158344


namespace problem_1_problem_2_l1583_158393

theorem problem_1 (m : ℤ) (h : m = -3) : 4 * (m + 1)^2 - (2*m + 5) * (2*m - 5) = 5 := by sorry

theorem problem_2 (x : ℚ) (h : x = 2) : (x^2 - 1) / (x^2 + 2*x) / ((x - 1) / x) = 3/4 := by sorry

end problem_1_problem_2_l1583_158393


namespace cubic_function_extremum_l1583_158351

/-- Given a cubic function f(x) = x³ + ax² + bx + a², 
    if f has an extremum at x = 1 and f(1) = 10, then a + b = -7 -/
theorem cubic_function_extremum (a b : ℝ) : 
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x + a^2
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  f 1 = 10 →
  a + b = -7 := by
sorry

end cubic_function_extremum_l1583_158351


namespace rectangle_area_l1583_158397

/-- Given a rectangle PQRS with specified coordinates, prove its area is 40400 -/
theorem rectangle_area (y : ℤ) : 
  let P : ℝ × ℝ := (10, -30)
  let Q : ℝ × ℝ := (2010, 170)
  let S : ℝ × ℝ := (12, y)
  let PQ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let PS := Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2)
  PQ * PS = 40400 := by
  sorry

end rectangle_area_l1583_158397


namespace daughters_to_sons_ratio_l1583_158376

theorem daughters_to_sons_ratio (total_children : ℕ) (sons : ℕ) (daughters : ℕ) : 
  total_children = 21 → sons = 3 → daughters = total_children - sons → 
  (daughters : ℚ) / (sons : ℚ) = 6 / 1 := by
  sorry

end daughters_to_sons_ratio_l1583_158376


namespace quadratic_real_root_condition_l1583_158359

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b is in the set (-∞, -10] ∪ [10, ∞). -/
theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry


end quadratic_real_root_condition_l1583_158359


namespace greatest_divisor_with_remainders_l1583_158383

theorem greatest_divisor_with_remainders : 
  Nat.gcd (690 - 10) (875 - 25) = 170 := by sorry

end greatest_divisor_with_remainders_l1583_158383


namespace sequential_search_comparisons_l1583_158386

/-- Represents a sequential search on an unordered array. -/
structure SequentialSearch where
  array_size : Nat
  element_not_present : Bool
  unordered : Bool

/-- The number of comparisons needed for a sequential search. -/
def comparisons_needed (search : SequentialSearch) : Nat :=
  search.array_size

/-- Theorem: The number of comparisons for a sequential search on an unordered array
    of 100 elements, where the element is not present, is 100. -/
theorem sequential_search_comparisons :
  ∀ (search : SequentialSearch),
    search.array_size = 100 →
    search.element_not_present = true →
    search.unordered = true →
    comparisons_needed search = 100 := by
  sorry

end sequential_search_comparisons_l1583_158386


namespace distinct_integer_quadruple_l1583_158371

theorem distinct_integer_quadruple : 
  ∀ a b c d : ℕ+, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    a + b = c * d →
    a * b = c + d →
    ((a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 2) ∨
     (a = 1 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
     (a = 5 ∧ b = 1 ∧ c = 3 ∧ d = 2) ∨
     (a = 5 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
     (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) ∨
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
     (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
     (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1)) :=
by
  sorry

end distinct_integer_quadruple_l1583_158371


namespace integral_tan_ln_cos_l1583_158311

theorem integral_tan_ln_cos (x : ℝ) :
  HasDerivAt (fun x => -1/2 * (Real.log (Real.cos x))^2) (Real.tan x * Real.log (Real.cos x)) x :=
by sorry

end integral_tan_ln_cos_l1583_158311


namespace infinite_representable_theorem_l1583_158369

-- Define an increasing sequence of positive integers
def IncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- Define the property we want to prove
def InfinitelyRepresentable (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, ∀ k : ℕ, ∃ n > k, ∃ j > i, ∃ r s : ℕ+, a n = r * a i + s * a j

-- State the theorem
theorem infinite_representable_theorem (a : ℕ → ℕ) (h : IncreasingSequence a) :
  InfinitelyRepresentable a := by
  sorry

end infinite_representable_theorem_l1583_158369


namespace some_number_value_l1583_158364

theorem some_number_value (some_number : ℝ) 
  (h1 : ∃ n : ℝ, (n / 18) * (n / some_number) = 1)
  (h2 : (54 / 18) * (54 / some_number) = 1) : 
  some_number = 162 := by
sorry

end some_number_value_l1583_158364


namespace imaginary_part_of_product_l1583_158323

def complex_mul (a b c d : ℝ) : ℂ := Complex.mk (a * c - b * d) (a * d + b * c)

theorem imaginary_part_of_product :
  (complex_mul 2 1 1 (-3)).im = -5 := by sorry

end imaginary_part_of_product_l1583_158323


namespace probability_mathematics_in_machine_l1583_158346

def mathematics_letters : Finset Char := {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S'}
def machine_letters : Finset Char := {'M', 'A', 'C', 'H', 'I', 'N', 'E'}

theorem probability_mathematics_in_machine :
  (mathematics_letters.filter (λ c => c ∈ machine_letters)).card / mathematics_letters.card = 7 / 11 := by
  sorry

end probability_mathematics_in_machine_l1583_158346


namespace brothers_age_sum_l1583_158360

theorem brothers_age_sum : 
  ∀ (older_age younger_age : ℕ),
  younger_age = 27 →
  younger_age = older_age / 3 + 10 →
  older_age + younger_age = 78 :=
by
  sorry

end brothers_age_sum_l1583_158360


namespace second_batch_average_l1583_158348

theorem second_batch_average (n1 n2 n3 : ℕ) (a1 a2 a3 overall_avg : ℝ) :
  n1 = 40 →
  n2 = 50 →
  n3 = 60 →
  a1 = 45 →
  a3 = 65 →
  overall_avg = 56.333333333333336 →
  (n1 * a1 + n2 * a2 + n3 * a3) / (n1 + n2 + n3) = overall_avg →
  a2 = 55 := by
  sorry

end second_batch_average_l1583_158348


namespace intersection_of_M_and_N_l1583_158373

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end intersection_of_M_and_N_l1583_158373


namespace running_speed_calculation_l1583_158322

/-- Proves that given the conditions, the running speed must be 8 km/hr -/
theorem running_speed_calculation (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) : 
  walking_speed = 4 →
  total_distance = 16 →
  total_time = 3 →
  (total_distance / 2) / walking_speed + (total_distance / 2) / 8 = total_time :=
by
  sorry

#check running_speed_calculation

end running_speed_calculation_l1583_158322


namespace not_necessarily_true_squared_l1583_158362

theorem not_necessarily_true_squared (x y : ℝ) (h : x > y) : 
  ¬ (∀ x y : ℝ, x > y → x^2 > y^2) :=
sorry

end not_necessarily_true_squared_l1583_158362


namespace bowknot_equation_solution_l1583_158342

-- Define the bowknot operation
noncomputable def bowknot (c d : ℝ) : ℝ :=
  c + Real.sqrt (d + Real.sqrt (d + Real.sqrt (d + Real.sqrt d)))

-- Theorem statement
theorem bowknot_equation_solution :
  ∃ x : ℝ, bowknot 3 x = 12 → x = 72 := by sorry

end bowknot_equation_solution_l1583_158342


namespace sin_range_theorem_l1583_158312

theorem sin_range_theorem (x : ℝ) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  Real.sin x ≥ Real.sqrt 2 / 2 → 
  x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) :=
by sorry

end sin_range_theorem_l1583_158312


namespace unique_four_letter_product_l1583_158321

def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5
  | 'F' => 6 | 'G' => 7 | 'H' => 8 | 'I' => 9 | 'J' => 10
  | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14 | 'O' => 15
  | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25
  | 'Z' => 26
  | _ => 0

def list_product (s : String) : ℕ :=
  s.foldl (fun acc c => acc * letter_value c) 1

def is_valid_four_letter_string (s : String) : Prop :=
  s.length = 4 ∧ s.all (fun c => 'A' ≤ c ∧ c ≤ 'Z')

theorem unique_four_letter_product :
  ∀ s : String, is_valid_four_letter_string s →
    list_product s = list_product "TUVW" →
    s = "TUVW" := by sorry

#check unique_four_letter_product

end unique_four_letter_product_l1583_158321


namespace negative_division_equals_nine_l1583_158367

theorem negative_division_equals_nine : (-81) / (-9) = 9 := by
  sorry

end negative_division_equals_nine_l1583_158367


namespace brookes_added_balloons_l1583_158365

/-- Prove that Brooke added 8 balloons to his collection -/
theorem brookes_added_balloons :
  ∀ (brooke_initial tracy_initial tracy_added total_after : ℕ) 
    (brooke_added : ℕ),
  brooke_initial = 12 →
  tracy_initial = 6 →
  tracy_added = 24 →
  total_after = 35 →
  brooke_initial + brooke_added + (tracy_initial + tracy_added) / 2 = total_after →
  brooke_added = 8 := by
sorry

end brookes_added_balloons_l1583_158365


namespace back_sides_average_l1583_158391

def is_prime_or_one (n : ℕ) : Prop := n = 1 ∨ Nat.Prime n

theorem back_sides_average (a b c : ℕ) : 
  is_prime_or_one a ∧ is_prime_or_one b ∧ is_prime_or_one c →
  28 + a = 40 + b ∧ 40 + b = 49 + c →
  (a + b + c) / 3 = 12 := by
  sorry

end back_sides_average_l1583_158391


namespace sequence_squared_l1583_158303

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ m n, m ≥ n → a (m + n) + a (m - n) = (a (2 * m) + a (2 * n)) / 2

theorem sequence_squared (a : ℕ → ℝ) (h : sequence_property a) (h1 : a 1 = 1) :
  ∀ n : ℕ, a n = n^2 := by
  sorry

#check sequence_squared

end sequence_squared_l1583_158303


namespace necessary_but_not_sufficient_condition_l1583_158320

theorem necessary_but_not_sufficient_condition 
  (A B C : Set α) 
  (hAnonempty : A.Nonempty) 
  (hBnonempty : B.Nonempty) 
  (hCnonempty : C.Nonempty) 
  (hUnion : A ∪ B = C) 
  (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ x, x ∈ C ∧ x ∉ A) := by
  sorry

end necessary_but_not_sufficient_condition_l1583_158320


namespace smallest_positive_integer_2016m_43200n_l1583_158331

theorem smallest_positive_integer_2016m_43200n :
  ∃ (k : ℕ+), (∀ (a : ℕ+), (∃ (m n : ℤ), a = 2016 * m + 43200 * n) → k ≤ a) ∧
  (∃ (m n : ℤ), (k : ℕ) = 2016 * m + 43200 * n) ∧
  k = 24 := by
  sorry

end smallest_positive_integer_2016m_43200n_l1583_158331


namespace probability_of_sunflower_seed_l1583_158357

def sunflower_seeds : ℕ := 2
def green_bean_seeds : ℕ := 3
def pumpkin_seeds : ℕ := 4

def total_seeds : ℕ := sunflower_seeds + green_bean_seeds + pumpkin_seeds

theorem probability_of_sunflower_seed :
  (sunflower_seeds : ℚ) / total_seeds = 2 / 9 := by sorry

end probability_of_sunflower_seed_l1583_158357


namespace initial_girls_count_l1583_158330

theorem initial_girls_count (p : ℕ) : 
  p > 0 →  -- Ensure p is positive
  (p : ℚ) / 2 - 3 = ((p : ℚ) * 2) / 5 → 
  (p : ℚ) / 2 = 15 :=
by
  sorry

#check initial_girls_count

end initial_girls_count_l1583_158330


namespace cricketer_wickets_after_match_l1583_158350

/-- Represents a cricketer's bowling statistics -/
structure Cricketer where
  initialAverage : ℝ
  initialWickets : ℕ
  matchWickets : ℕ
  matchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the total number of wickets taken by a cricketer after a match -/
def totalWicketsAfterMatch (c : Cricketer) : ℕ :=
  c.initialWickets + c.matchWickets

/-- Theorem stating that for a cricketer with given statistics, the total wickets after the match is 90 -/
theorem cricketer_wickets_after_match (c : Cricketer) 
  (h1 : c.initialAverage = 12.4)
  (h2 : c.matchWickets = 5)
  (h3 : c.matchRuns = 26)
  (h4 : c.averageDecrease = 0.4) :
  totalWicketsAfterMatch c = 90 := by
  sorry

end cricketer_wickets_after_match_l1583_158350


namespace conversion_equivalence_l1583_158343

/-- Conversion rates between different units --/
structure ConversionRates where
  knicks_to_knacks : ℚ  -- 5 knicks = 3 knacks
  knacks_to_knocks : ℚ  -- 2 knacks = 5 knocks
  knocks_to_kracks : ℚ  -- 4 knocks = 1 krack

/-- Calculate the equivalent number of knicks for a given number of knocks --/
def knocks_to_knicks (rates : ConversionRates) (knocks : ℚ) : ℚ :=
  knocks * rates.knacks_to_knocks * rates.knicks_to_knacks

/-- Calculate the equivalent number of kracks for a given number of knocks --/
def knocks_to_kracks (rates : ConversionRates) (knocks : ℚ) : ℚ :=
  knocks * rates.knocks_to_kracks

theorem conversion_equivalence (rates : ConversionRates) 
  (h1 : rates.knicks_to_knacks = 3 / 5)
  (h2 : rates.knacks_to_knocks = 5 / 2)
  (h3 : rates.knocks_to_kracks = 1 / 4) :
  knocks_to_knicks rates 50 = 100 / 3 ∧ knocks_to_kracks rates 50 = 25 / 3 := by
  sorry

#check conversion_equivalence

end conversion_equivalence_l1583_158343


namespace train_length_l1583_158379

/-- The length of a train crossing a bridge -/
theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 200 →
  crossing_time = 60 →
  train_speed = 5 →
  bridge_length + (train_speed * crossing_time - bridge_length) = 100 :=
by
  sorry

end train_length_l1583_158379


namespace mean_equality_implies_x_value_l1583_158378

theorem mean_equality_implies_x_value :
  let mean1 := (8 + 12 + 24) / 3
  let mean2 := (16 + x) / 2
  mean1 = mean2 → x = 40 / 3 := by
sorry

end mean_equality_implies_x_value_l1583_158378


namespace esme_school_non_pizza_eaters_l1583_158337

/-- The number of teachers at Esme's school -/
def num_teachers : ℕ := 30

/-- The number of staff members at Esme's school -/
def num_staff : ℕ := 45

/-- The fraction of teachers who ate pizza -/
def teacher_pizza_fraction : ℚ := 2/3

/-- The fraction of staff members who ate pizza -/
def staff_pizza_fraction : ℚ := 4/5

/-- The total number of non-pizza eaters at Esme's school -/
def non_pizza_eaters : ℕ := 19

theorem esme_school_non_pizza_eaters :
  (num_teachers - (num_teachers : ℚ) * teacher_pizza_fraction).floor +
  (num_staff - (num_staff : ℚ) * staff_pizza_fraction).floor = non_pizza_eaters := by
  sorry

end esme_school_non_pizza_eaters_l1583_158337


namespace constant_term_expansion_l1583_158385

theorem constant_term_expansion (x : ℝ) : ∃ c : ℝ, c = 24 ∧ 
  (∃ f : ℝ → ℝ, (λ x => (2*x + 1/x)^4) = f + λ _ => c) := by
  sorry

end constant_term_expansion_l1583_158385


namespace garden_area_l1583_158327

/-- Represents a rectangular garden with given properties. -/
structure Garden where
  length : ℝ
  width : ℝ
  length_walk : ℝ
  perimeter_walk : ℝ
  length_condition : length * 30 = length_walk
  perimeter_condition : (2 * length + 2 * width) * 12 = perimeter_walk
  walk_equality : length_walk = perimeter_walk
  length_walk_value : length_walk = 1500

/-- The area of the garden with the given conditions is 625 square meters. -/
theorem garden_area (g : Garden) : g.length * g.width = 625 := by
  sorry

end garden_area_l1583_158327


namespace sweet_potato_problem_l1583_158309

theorem sweet_potato_problem (total : ℕ) (sold_to_adams : ℕ) (sold_to_lenon : ℕ) 
  (h1 : total = 80) 
  (h2 : sold_to_adams = 20) 
  (h3 : sold_to_lenon = 15) : 
  total - (sold_to_adams + sold_to_lenon) = 45 := by
  sorry

end sweet_potato_problem_l1583_158309


namespace percentage_problem_l1583_158329

theorem percentage_problem (P : ℝ) (h : (P / 4) * 2 = 0.02) : P = 4 := by
  sorry

end percentage_problem_l1583_158329


namespace line_equation_proof_l1583_158387

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y + 3 = 0
def line2 (x y : ℝ) : Prop := 4*x + 3*y + 1 = 0
def line3 (x y : ℝ) : Prop := 2*x - 3*y + 4 = 0
def line_result (x y : ℝ) : Prop := 3*x + 2*y + 1 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_equation_proof :
  ∃ x y : ℝ, 
    intersection_point x y ∧ 
    line_result x y ∧
    perpendicular (3/2) (-2/3) :=
sorry

end line_equation_proof_l1583_158387


namespace expand_and_simplify_l1583_158336

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end expand_and_simplify_l1583_158336


namespace smallest_y_for_perfect_fourth_power_l1583_158307

def x : ℕ := 5 * 15 * 35

theorem smallest_y_for_perfect_fourth_power (y : ℕ) : 
  y = 46485 ↔ 
  (∀ z : ℕ, z < y → ¬∃ (n : ℕ), x * z = n^4) ∧
  ∃ (n : ℕ), x * y = n^4 :=
sorry

end smallest_y_for_perfect_fourth_power_l1583_158307


namespace birthday_crayons_l1583_158301

/-- The number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 291

/-- The number of crayons Paul lost or gave away -/
def crayons_lost_or_given : ℕ := 315

/-- The total number of crayons Paul got for his birthday -/
def total_crayons : ℕ := crayons_left + crayons_lost_or_given

theorem birthday_crayons : total_crayons = 606 := by
  sorry

end birthday_crayons_l1583_158301


namespace john_earnings_increase_l1583_158347

/-- Calculates the percentage increase in earnings -/
def percentage_increase (initial_earnings final_earnings : ℚ) : ℚ :=
  (final_earnings - initial_earnings) / initial_earnings * 100

/-- Represents John's weekly earnings from two jobs -/
structure WeeklyEarnings where
  job_a_initial : ℚ
  job_a_final : ℚ
  job_b_initial : ℚ
  job_b_final : ℚ

theorem john_earnings_increase (john : WeeklyEarnings)
  (h1 : john.job_a_initial = 60)
  (h2 : john.job_a_final = 78)
  (h3 : john.job_b_initial = 100)
  (h4 : john.job_b_final = 120) :
  percentage_increase (john.job_a_initial + john.job_b_initial)
                      (john.job_a_final + john.job_b_final) = 23.75 := by
  sorry

end john_earnings_increase_l1583_158347


namespace athletes_arrival_time_l1583_158380

/-- Proves that the number of hours new athletes arrived is 7, given the initial conditions and the final difference in the number of athletes. -/
theorem athletes_arrival_time (
  initial_athletes : ℕ)
  (leaving_rate : ℕ)
  (leaving_hours : ℕ)
  (arriving_rate : ℕ)
  (final_difference : ℕ)
  (h1 : initial_athletes = 300)
  (h2 : leaving_rate = 28)
  (h3 : leaving_hours = 4)
  (h4 : arriving_rate = 15)
  (h5 : final_difference = 7)
  : ∃ (x : ℕ), 
    initial_athletes - (leaving_rate * leaving_hours) + (arriving_rate * x) = 
    initial_athletes - final_difference ∧ x = 7 := by
  sorry

end athletes_arrival_time_l1583_158380


namespace sin_value_given_tan_and_range_l1583_158358

theorem sin_value_given_tan_and_range (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) (3 * π / 2)) 
  (h2 : Real.tan α = Real.sqrt 2) : 
  Real.sin α = -(Real.sqrt 6 / 3) := by
sorry

end sin_value_given_tan_and_range_l1583_158358


namespace line_not_in_third_quadrant_l1583_158305

/-- A line defined by y = (m-2)x + m, where 0 < m < 2, does not pass through the third quadrant. -/
theorem line_not_in_third_quadrant (m : ℝ) (h : 0 < m ∧ m < 2) :
  ∃ (x y : ℝ), y = (m - 2) * x + m → ¬(x < 0 ∧ y < 0) :=
by sorry

end line_not_in_third_quadrant_l1583_158305


namespace min_abs_sum_l1583_158399

theorem min_abs_sum (x : ℝ) : 
  ∃ (l : ℝ), l = 45 ∧ ∀ y : ℝ, |y - 2| + |y - 47| ≥ l :=
sorry

end min_abs_sum_l1583_158399


namespace shorter_side_length_l1583_158394

-- Define the circle and rectangle
def circle_radius : ℝ := 6

-- Define the relationship between circle and rectangle areas
def rectangle_area (circle_area : ℝ) : ℝ := 3 * circle_area

-- Define the theorem
theorem shorter_side_length (circle_area : ℝ) (rectangle_area : ℝ) 
  (h1 : circle_area = π * circle_radius ^ 2)
  (h2 : rectangle_area = 3 * circle_area)
  (h3 : rectangle_area = (2 * circle_radius) * shorter_side) :
  shorter_side = 9 * π := by
  sorry


end shorter_side_length_l1583_158394


namespace pyramid_volume_theorem_l1583_158326

/-- Represents a pyramid with a square base and a vertex -/
structure Pyramid where
  base_area : ℝ
  triangle_abe_area : ℝ
  triangle_cde_area : ℝ

/-- Calculate the volume of a pyramid -/
def pyramid_volume (p : Pyramid) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem pyramid_volume_theorem (p : Pyramid) 
  (h1 : p.base_area = 256)
  (h2 : p.triangle_abe_area = 120)
  (h3 : p.triangle_cde_area = 110) :
  pyramid_volume p = 1152 :=
sorry

end pyramid_volume_theorem_l1583_158326


namespace divisibility_relation_l1583_158366

theorem divisibility_relation (p q r s : ℤ) 
  (h_s : s % 5 ≠ 0)
  (h_a : ∃ a : ℤ, (p * a^3 + q * a^2 + r * a + s) % 5 = 0) :
  ∃ b : ℤ, (s * b^3 + r * b^2 + q * b + p) % 5 = 0 :=
sorry

end divisibility_relation_l1583_158366


namespace xyz_value_l1583_158334

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by
sorry

end xyz_value_l1583_158334


namespace playground_area_is_4200_l1583_158390

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ

/-- The landscape satisfies the given conditions -/
def is_valid_landscape (l : Landscape) : Prop :=
  l.breadth = 6 * l.length ∧
  l.breadth = 420 ∧
  l.playground_area = (1 / 7) * (l.length * l.breadth)

theorem playground_area_is_4200 (l : Landscape) (h : is_valid_landscape l) :
  l.playground_area = 4200 := by
  sorry

#check playground_area_is_4200

end playground_area_is_4200_l1583_158390


namespace ceiling_neg_sqrt_64_over_9_l1583_158308

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end ceiling_neg_sqrt_64_over_9_l1583_158308


namespace mans_rowing_speed_l1583_158392

/-- Represents the rowing scenario in a river with current --/
structure RowingScenario where
  stream_rate : ℝ
  rowing_speed : ℝ
  time_ratio : ℝ

/-- Checks if the rowing scenario satisfies the given conditions --/
def is_valid_scenario (s : RowingScenario) : Prop :=
  s.stream_rate = 18 ∧ 
  s.time_ratio = 3 ∧
  (1 / (s.rowing_speed - s.stream_rate)) = s.time_ratio * (1 / (s.rowing_speed + s.stream_rate))

/-- Theorem stating that the man's rowing speed in still water is 36 kmph --/
theorem mans_rowing_speed (s : RowingScenario) : 
  is_valid_scenario s → s.rowing_speed = 36 :=
by
  sorry


end mans_rowing_speed_l1583_158392


namespace remainder_calculation_l1583_158396

-- Define the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- State the theorem
theorem remainder_calculation : rem (-1/3) (4/7) = 5/21 := by
  sorry

end remainder_calculation_l1583_158396


namespace rope_cutting_l1583_158352

theorem rope_cutting (total_length : ℕ) (equal_pieces : ℕ) (equal_piece_length : ℕ) (remaining_piece_length : ℕ) : 
  total_length = 1165 ∧ 
  equal_pieces = 150 ∧ 
  equal_piece_length = 75 ∧ 
  remaining_piece_length = 100 → 
  (total_length * 10 - equal_pieces * equal_piece_length) / remaining_piece_length + equal_pieces = 154 :=
by sorry

end rope_cutting_l1583_158352


namespace arithmetic_operations_l1583_158361

theorem arithmetic_operations : 
  (24 - (-16) + (-25) - 32 = -17) ∧
  ((-1/2) * 2 / 2 * (-1/2) = 1/4) ∧
  (-2^2 * 5 - (-2)^3 * (1/8) + 1 = -18) ∧
  ((-1/4 - 5/6 + 8/9) / (-1/6)^2 + (-2)^2 * (-6) = -31) := by
  sorry

end arithmetic_operations_l1583_158361


namespace correct_calculation_l1583_158398

theorem correct_calculation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end correct_calculation_l1583_158398


namespace multiply_add_equality_l1583_158316

theorem multiply_add_equality : 45 * 27 + 18 * 45 = 2025 := by
  sorry

end multiply_add_equality_l1583_158316


namespace equation_solution_l1583_158319

theorem equation_solution : 
  ∃! x : ℚ, (x^2 - 4*x + 3)/(x^2 - 6*x + 5) = (x^2 - 3*x - 10)/(x^2 - 2*x - 15) ∧ x = -19/3 := by
  sorry

end equation_solution_l1583_158319


namespace greatest_k_value_l1583_158318

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 73) →
  k ≤ Real.sqrt 105 :=
sorry

end greatest_k_value_l1583_158318


namespace tower_surface_area_l1583_158388

-- Define the volumes of the cubes
def cube_volumes : List ℝ := [1, 8, 27, 64, 125, 216, 343]

-- Function to calculate the side length of a cube given its volume
def side_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Function to calculate the surface area of a cube given its side length
def surface_area (side : ℝ) : ℝ := 6 * side^2

-- Function to calculate the exposed surface area of a cube in the tower
def exposed_surface_area (side : ℝ) (is_bottom : Bool) : ℝ :=
  if is_bottom then surface_area side else surface_area side - side^2

-- Theorem statement
theorem tower_surface_area :
  let sides := cube_volumes.map side_length
  let exposed_areas := List.zipWith exposed_surface_area sides [true, false, false, false, false, false, false]
  exposed_areas.sum = 701 := by sorry

end tower_surface_area_l1583_158388


namespace min_value_a_squared_plus_b_squared_l1583_158314

theorem min_value_a_squared_plus_b_squared :
  ∀ a b : ℝ,
  ((-2)^2 + a*(-2) + 2*b = 0) →
  ∀ c d : ℝ,
  (c^2 + d^2 ≥ a^2 + b^2) →
  (a^2 + b^2 ≥ 2) :=
by sorry

end min_value_a_squared_plus_b_squared_l1583_158314


namespace rope_fraction_proof_l1583_158363

theorem rope_fraction_proof (total_ropes : ℕ) (avg_length total_length : ℝ) 
  (group1_avg group2_avg : ℝ) (f : ℝ) :
  total_ropes = 6 →
  avg_length = 80 →
  total_length = avg_length * total_ropes →
  group1_avg = 70 →
  group2_avg = 85 →
  total_length = group1_avg * (f * total_ropes) + group2_avg * ((1 - f) * total_ropes) →
  f = 1 / 3 := by
  sorry

#check rope_fraction_proof

end rope_fraction_proof_l1583_158363


namespace range_of_m_l1583_158339

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -Real.sqrt (4 - p.2^2)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 6}

-- Define the condition for points P and Q
def existsPQ (m : ℝ) : Prop :=
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ C ∧ Q ∈ l ∧
    (P.1 - m, P.2) + (Q.1 - m, Q.2) = (0, 0)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, existsPQ m → 2 ≤ m ∧ m ≤ 3 :=
sorry

end range_of_m_l1583_158339


namespace dog_weight_ratio_l1583_158372

/-- Given the weights of two dogs, prove the ratio of their weights -/
theorem dog_weight_ratio 
  (evan_dog_weight : ℕ) 
  (total_weight : ℕ) 
  (h1 : evan_dog_weight = 63)
  (h2 : total_weight = 72)
  (h3 : ∃ k : ℕ, k * (total_weight - evan_dog_weight) = evan_dog_weight) :
  evan_dog_weight / (total_weight - evan_dog_weight) = 7 := by
sorry

end dog_weight_ratio_l1583_158372


namespace sum_of_squares_of_roots_l1583_158377

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 5*x + 6 = 0 → ∃ s₁ s₂ : ℝ, s₁ + s₂ = 5 ∧ s₁ * s₂ = 6 ∧ s₁^2 + s₂^2 = 13 :=
by sorry

end sum_of_squares_of_roots_l1583_158377


namespace bus_schedule_hours_l1583_158333

/-- The number of hours per day that buses leave the station -/
def hours_per_day (total_buses : ℕ) (days : ℕ) (buses_per_hour : ℕ) : ℚ :=
  (total_buses : ℚ) / (days : ℚ) / (buses_per_hour : ℚ)

/-- Theorem stating that under given conditions, buses leave the station for 12 hours per day -/
theorem bus_schedule_hours (total_buses : ℕ) (days : ℕ) (buses_per_hour : ℕ)
    (h1 : total_buses = 120)
    (h2 : days = 5)
    (h3 : buses_per_hour = 2) :
    hours_per_day total_buses days buses_per_hour = 12 := by
  sorry

end bus_schedule_hours_l1583_158333


namespace garden_division_theorem_l1583_158374

/-- Represents a rectangular garden -/
structure Garden where
  width : ℕ
  height : ℕ
  trees : ℕ

/-- Represents a division of the garden -/
structure Division where
  parts : ℕ
  matches_used : ℕ
  trees_per_part : ℕ

/-- Checks if a division is valid for a given garden -/
def is_valid_division (g : Garden) (d : Division) : Prop :=
  d.parts = 4 ∧
  d.matches_used = 12 ∧
  d.trees_per_part * d.parts = g.trees ∧
  d.trees_per_part = 3

theorem garden_division_theorem (g : Garden) 
  (h1 : g.width = 4)
  (h2 : g.height = 3)
  (h3 : g.trees = 12) :
  ∃ d : Division, is_valid_division g d :=
sorry

end garden_division_theorem_l1583_158374


namespace min_value_theorem_l1583_158304

theorem min_value_theorem (x y : ℝ) (h : x - 2*y - 4 = 0) :
  ∃ (min : ℝ), min = 8 ∧ ∀ z, z = 2^x + 1/(4^y) → z ≥ min :=
sorry

end min_value_theorem_l1583_158304


namespace devin_initial_height_l1583_158340

/-- The chances of making the basketball team for a given height. -/
def chance_of_making_team (height : ℝ) : ℝ :=
  0.1 + (height - 66) * 0.1

/-- Devin's initial height before growth. -/
def initial_height : ℝ := 68

/-- The amount Devin grew in inches. -/
def growth : ℝ := 3

/-- Devin's final chance of making the team after growth. -/
def final_chance : ℝ := 0.3

theorem devin_initial_height :
  chance_of_making_team (initial_height + growth) = final_chance :=
by sorry

end devin_initial_height_l1583_158340


namespace kim_sweaters_theorem_l1583_158317

/-- The number of sweaters Kim knit on Monday -/
def monday_sweaters : ℕ := 8

/-- The number of sweaters Kim knit on Tuesday -/
def tuesday_sweaters : ℕ := monday_sweaters + 2

/-- The number of sweaters Kim knit on Wednesday -/
def wednesday_sweaters : ℕ := tuesday_sweaters - 4

/-- The number of sweaters Kim knit on Thursday -/
def thursday_sweaters : ℕ := tuesday_sweaters - 4

/-- The number of sweaters Kim knit on Friday -/
def friday_sweaters : ℕ := monday_sweaters / 2

/-- The total number of sweaters Kim knit in the week -/
def total_sweaters : ℕ := monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters

theorem kim_sweaters_theorem : total_sweaters = 34 := by
  sorry

end kim_sweaters_theorem_l1583_158317


namespace no_three_fractions_product_one_l1583_158315

theorem no_three_fractions_product_one :
  ¬ ∃ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 100 ∧
    (a : ℚ) / (101 - a) * (b : ℚ) / (101 - b) * (c : ℚ) / (101 - c) = 1 := by
  sorry

end no_three_fractions_product_one_l1583_158315


namespace solution_to_equation_l1583_158335

theorem solution_to_equation : ∃ x : ℕ, 
  (x = 10^2023 - 1) ∧ 
  (567 * x^3 + 171 * x^2 + 15 * x - (3 * x + 5 * x * 10^2023 + 7 * x * 10^(2*2023)) = 0) := by
  sorry

end solution_to_equation_l1583_158335


namespace A_inverse_correct_l1583_158300

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![3, -1, 3],
    ![2, -1, 4],
    ![1,  2, -3]]

def A_inv : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![ 1/2, -3/10,  1/10],
    ![-1,    6/5,   3/5],
    ![-1/2,  7/10,  1/10]]

theorem A_inverse_correct : A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end A_inverse_correct_l1583_158300


namespace investment_difference_is_1000_l1583_158353

/-- Represents the investment problem with three persons --/
structure InvestmentProblem where
  total_investment : ℕ
  total_gain : ℕ
  third_person_gain : ℕ

/-- Calculates the investment difference between the second and first person --/
def investment_difference (problem : InvestmentProblem) : ℕ :=
  let first_investment := problem.total_investment / 3
  let second_investment := first_investment + (problem.total_investment / 3 - first_investment)
  second_investment - first_investment

/-- Theorem stating that the investment difference is 1000 for the given problem --/
theorem investment_difference_is_1000 (problem : InvestmentProblem) 
  (h1 : problem.total_investment = 9000)
  (h2 : problem.total_gain = 1800)
  (h3 : problem.third_person_gain = 800) :
  investment_difference problem = 1000 := by
  sorry

#eval investment_difference ⟨9000, 1800, 800⟩

end investment_difference_is_1000_l1583_158353


namespace alcohol_concentration_second_vessel_l1583_158370

/-- Proves that the initial concentration of alcohol in the second vessel is 60% --/
theorem alcohol_concentration_second_vessel :
  let vessel1_capacity : ℝ := 2
  let vessel1_alcohol_percentage : ℝ := 40
  let vessel2_capacity : ℝ := 6
  let total_liquid : ℝ := 8
  let final_vessel_capacity : ℝ := 10
  let final_mixture_percentage : ℝ := 44
  let vessel2_alcohol_percentage : ℝ := 
    (final_mixture_percentage * final_vessel_capacity - vessel1_alcohol_percentage * vessel1_capacity) / vessel2_capacity
  vessel2_alcohol_percentage = 60 := by
sorry

end alcohol_concentration_second_vessel_l1583_158370
