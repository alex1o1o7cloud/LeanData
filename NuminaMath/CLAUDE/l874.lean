import Mathlib

namespace auction_result_l874_87461

def auction_total (tv_initial : ℝ) (tv_increase : ℝ) (phone_initial : ℝ) (phone_increase : ℝ) 
                  (laptop_initial : ℝ) (laptop_decrease : ℝ) (auction_fee_rate : ℝ) : ℝ :=
  let tv_final := tv_initial * (1 + tv_increase)
  let phone_final := phone_initial * (1 + phone_increase)
  let laptop_final := laptop_initial * (1 - laptop_decrease)
  let total_before_fee := tv_final + phone_final + laptop_final
  let fee := total_before_fee * auction_fee_rate
  total_before_fee - fee

theorem auction_result : 
  auction_total 500 (2/5) 400 0.4 800 0.15 0.05 = 1843 := by
  sorry

end auction_result_l874_87461


namespace division_problem_l874_87480

theorem division_problem (dividend divisor : ℕ) (h1 : dividend + divisor = 136) (h2 : dividend / divisor = 7) : divisor = 17 := by
  sorry

end division_problem_l874_87480


namespace last_segment_speed_l874_87429

def total_distance : ℝ := 120
def total_time : ℝ := 1.5
def segment_time : ℝ := 0.5
def speed_segment1 : ℝ := 50
def speed_segment2 : ℝ := 70

theorem last_segment_speed :
  ∃ (speed_segment3 : ℝ),
    (speed_segment1 * segment_time + speed_segment2 * segment_time + speed_segment3 * segment_time) / total_time = total_distance / total_time ∧
    speed_segment3 = 120 := by
  sorry

end last_segment_speed_l874_87429


namespace fraction_arrangement_l874_87449

theorem fraction_arrangement :
  (1 / 8 * 1 / 9 * 1 / 28 : ℚ) = 1 / 2016 ∨ ((1 / 8 - 1 / 9) * 1 / 28 : ℚ) = 1 / 2016 :=
by sorry

end fraction_arrangement_l874_87449


namespace blue_easter_eggs_fraction_l874_87491

theorem blue_easter_eggs_fraction 
  (purple_fraction : ℚ) 
  (purple_five_candy_ratio : ℚ) 
  (blue_five_candy_ratio : ℚ) 
  (five_candy_probability : ℚ) :
  purple_fraction = 1/5 →
  purple_five_candy_ratio = 1/2 →
  blue_five_candy_ratio = 1/4 →
  five_candy_probability = 3/10 →
  ∃ blue_fraction : ℚ, 
    blue_fraction = 4/5 ∧ 
    purple_fraction * purple_five_candy_ratio + blue_fraction * blue_five_candy_ratio = five_candy_probability :=
by sorry

end blue_easter_eggs_fraction_l874_87491


namespace arithmetic_sequence_25th_term_l874_87450

/-- An arithmetic sequence with first term 100 and common difference -4 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  100 - 4 * (n - 1)

theorem arithmetic_sequence_25th_term :
  arithmetic_sequence 25 = 4 :=
by sorry

end arithmetic_sequence_25th_term_l874_87450


namespace rectangle_diagonal_pythagorean_l874_87499

/-- A rectangle with side lengths a and b, and diagonal c -/
structure Rectangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- The Pythagorean theorem holds for the rectangle's diagonal -/
theorem rectangle_diagonal_pythagorean (rect : Rectangle) : 
  rect.c^2 = rect.a^2 + rect.b^2 := by
  sorry

#check rectangle_diagonal_pythagorean

end rectangle_diagonal_pythagorean_l874_87499


namespace inequality_and_equality_condition_l874_87478

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : ¬(a = b ∧ b = c)) : 
  (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ≥ 
    (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ∧
  ((a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 = 
    (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ↔ 
    ((a = 0 ∧ b = 0 ∧ c > 0) ∨ (a = 0 ∧ c = 0 ∧ b > 0) ∨ (b = 0 ∧ c = 0 ∧ a > 0))) :=
by sorry

end inequality_and_equality_condition_l874_87478


namespace time_after_elapsed_minutes_l874_87464

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting DateTime -/
def startTime : DateTime :=
  { year := 2015, month := 3, day := 3, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def elapsedMinutes : Nat := 4350

/-- The expected result DateTime -/
def expectedResult : DateTime :=
  { year := 2015, month := 3, day := 6, hour := 0, minute := 30 }

theorem time_after_elapsed_minutes :
  addMinutes startTime elapsedMinutes = expectedResult := by
  sorry

end time_after_elapsed_minutes_l874_87464


namespace log_inequality_l874_87417

theorem log_inequality (a : ℝ) (h : 0 < a ∧ a < 1/4) :
  ∀ x : ℝ, (0 < x ∧ x ≠ 1 ∧ x + a > 0 ∧ x + a ≠ 1) →
  (Real.log 2 / Real.log (x + a) < Real.log 4 / Real.log x ↔
    (0 < x ∧ x < 1/2 - a - Real.sqrt (1/4 - a)) ∨
    (1/2 - a + Real.sqrt (1/4 - a) < x ∧ x < 1 - a) ∨
    (1 < x)) :=
by sorry

end log_inequality_l874_87417


namespace infinite_prime_divisors_of_derived_set_l874_87407

/-- A subset of natural numbers with infinite members -/
def InfiniteNatSubset (S : Set ℕ) : Prop := Set.Infinite S

/-- The set S' derived from S -/
def DerivedSet (S : Set ℕ) : Set ℕ :=
  {n | ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ n = x^y + y^x}

/-- The set of prime divisors of a set of natural numbers -/
def PrimeDivisors (S : Set ℕ) : Set ℕ :=
  {p | Nat.Prime p ∧ ∃ n ∈ S, p ∣ n}

/-- Main theorem: The set of prime divisors of S' is infinite -/
theorem infinite_prime_divisors_of_derived_set (S : Set ℕ) 
  (h : InfiniteNatSubset S) : Set.Infinite (PrimeDivisors (DerivedSet S)) :=
sorry

end infinite_prime_divisors_of_derived_set_l874_87407


namespace smallest_n_divisible_l874_87490

theorem smallest_n_divisible (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < n → ¬(20 ∣ (25 * m) ∧ 18 ∣ (25 * m) ∧ 24 ∣ (25 * m))) →
  (20 ∣ (25 * n) ∧ 18 ∣ (25 * n) ∧ 24 ∣ (25 * n)) →
  n = 36 := by
sorry

end smallest_n_divisible_l874_87490


namespace last_term_value_l874_87446

-- Define the arithmetic sequence
def arithmetic_sequence (a b : ℝ) : ℕ → ℝ
  | 0 => a
  | 1 => b
  | 2 => 5 * a
  | 3 => 7
  | 4 => 3 * b
  | n + 5 => arithmetic_sequence a b n

-- Define the sum of the sequence
def sequence_sum (a b : ℝ) (n : ℕ) : ℝ :=
  (List.range n).map (arithmetic_sequence a b) |>.sum

-- Theorem statement
theorem last_term_value (a b : ℝ) (n : ℕ) :
  sequence_sum a b n = 2500 →
  ∃ c, arithmetic_sequence a b (n - 1) = c ∧ c = 99 := by
  sorry

end last_term_value_l874_87446


namespace school_ratio_problem_l874_87484

theorem school_ratio_problem (S T : ℕ) : 
  S / T = 50 →
  (S + 50) / (T + 5) = 25 →
  T = 3 :=
by sorry

end school_ratio_problem_l874_87484


namespace compare_numbers_l874_87493

-- Define the base conversion function
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => acc * base + digit) 0

-- Define the numbers in their respective bases
def num1 : List Nat := [8, 5]
def base1 : Nat := 9

def num2 : List Nat := [2, 1, 0]
def base2 : Nat := 6

def num3 : List Nat := [1, 0, 0, 0]
def base3 : Nat := 4

def num4 : List Nat := [1, 1, 1, 1, 1, 1]
def base4 : Nat := 2

-- State the theorem
theorem compare_numbers :
  to_decimal num2 base2 > to_decimal num1 base1 ∧
  to_decimal num1 base1 > to_decimal num3 base3 ∧
  to_decimal num3 base3 > to_decimal num4 base4 := by
  sorry

end compare_numbers_l874_87493


namespace lizas_account_balance_l874_87444

/-- Calculates the final balance in Liza's account after all transactions -/
def final_balance (initial_balance rent paycheck electricity internet phone : ℤ) : ℤ :=
  initial_balance - rent + paycheck - electricity - internet - phone

/-- Theorem stating that Liza's final account balance is correct -/
theorem lizas_account_balance :
  final_balance 800 450 1500 117 100 70 = 1563 := by
  sorry

end lizas_account_balance_l874_87444


namespace triangle_is_obtuse_l874_87412

/-- A triangle is obtuse if one of its angles is greater than 90 degrees -/
def IsObtuseTriangle (a b c : ℝ) : Prop :=
  a > 90 ∨ b > 90 ∨ c > 90

/-- Theorem: If A, B, and C are the interior angles of a triangle, 
    and A > 3B and C < 2B, then the triangle is obtuse -/
theorem triangle_is_obtuse (a b c : ℝ) 
    (angle_sum : a + b + c = 180)
    (h1 : a > 3 * b) 
    (h2 : c < 2 * b) : 
  IsObtuseTriangle a b c := by
sorry

end triangle_is_obtuse_l874_87412


namespace concert_admission_revenue_l874_87477

theorem concert_admission_revenue :
  let total_attendance : ℕ := 578
  let adult_price : ℚ := 2
  let child_price : ℚ := (3/2)
  let num_adults : ℕ := 342
  let num_children : ℕ := total_attendance - num_adults
  let total_revenue : ℚ := (num_adults : ℚ) * adult_price + (num_children : ℚ) * child_price
  total_revenue = 1038 :=
by sorry

end concert_admission_revenue_l874_87477


namespace tangent_parallel_points_l874_87496

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ (3 * x^2 + 1 = 4) ↔ (x = -1 ∧ y = -4) ∨ (x = 1 ∧ y = 0) :=
by sorry

end tangent_parallel_points_l874_87496


namespace unique_solution_sqrt_equation_l874_87476

theorem unique_solution_sqrt_equation :
  ∃! (x : ℝ), 2 * x + Real.sqrt (x - 3) = 7 :=
by
  -- The unique solution is x = 3.25
  use 3.25
  sorry

end unique_solution_sqrt_equation_l874_87476


namespace inequality_proof_l874_87460

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 - a^3) * (1 / b^2 - b^3) ≥ (31/8)^2 := by
  sorry

end inequality_proof_l874_87460


namespace flea_jump_rational_angle_l874_87418

/-- Represents a flea jumping between two intersecting lines -/
structure FleaJump where
  α : ℝ  -- Angle between the lines in radians
  jump_length : ℝ  -- Length of each jump
  returns_to_start : Prop  -- Flea eventually returns to the starting point

/-- Main theorem: If a flea jumps between two intersecting lines and returns to the start,
    the angle between the lines is a rational multiple of π -/
theorem flea_jump_rational_angle (fj : FleaJump) 
  (h1 : fj.jump_length = 1)
  (h2 : fj.returns_to_start)
  (h3 : fj.α > 0)
  (h4 : fj.α < π) :
  ∃ q : ℚ, fj.α = q * π :=
sorry

end flea_jump_rational_angle_l874_87418


namespace scientific_notation_of_6_1757_million_l874_87487

theorem scientific_notation_of_6_1757_million :
  let original_number : ℝ := 6.1757 * 1000000
  original_number = 6.1757 * (10 ^ 6) :=
by sorry

end scientific_notation_of_6_1757_million_l874_87487


namespace sum_of_repeating_decimals_l874_87400

def repeating_decimal_3 : ℚ := 1/3
def repeating_decimal_02 : ℚ := 2/99

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_02 = 35/99 := by
  sorry

end sum_of_repeating_decimals_l874_87400


namespace soda_cost_l874_87416

/-- Given the total cost of an order and the cost of sandwiches, 
    calculate the cost of each soda. -/
theorem soda_cost (total_cost sandwich_cost : ℚ) 
  (h1 : total_cost = 10.46)
  (h2 : sandwich_cost = 3.49)
  (h3 : 2 * sandwich_cost + 4 * (total_cost - 2 * sandwich_cost) / 4 = total_cost) :
  (total_cost - 2 * sandwich_cost) / 4 = 0.87 := by sorry

end soda_cost_l874_87416


namespace sales_tax_difference_example_l874_87489

/-- The difference between two sales tax amounts -/
def sales_tax_difference (price : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  price * rate1 - price * rate2

/-- Theorem: The difference between a 7.25% sales tax and a 7% sales tax on an item priced at $50 before tax is $0.125 -/
theorem sales_tax_difference_example : 
  sales_tax_difference 50 0.0725 0.07 = 0.125 := by
sorry

end sales_tax_difference_example_l874_87489


namespace arithmetic_evaluation_l874_87471

theorem arithmetic_evaluation : 8 / 2 - 3 * 2 + 5^2 / 5 = 3 := by
  sorry

end arithmetic_evaluation_l874_87471


namespace expected_defective_meters_l874_87453

/-- Proves that given a rejection rate of 1.5% and a sample size of 10,000 meters,
    the expected number of defective meters is 150. -/
theorem expected_defective_meters
  (rejection_rate : ℝ)
  (sample_size : ℕ)
  (h1 : rejection_rate = 0.015)
  (h2 : sample_size = 10000) :
  ↑sample_size * rejection_rate = 150 := by
  sorry

end expected_defective_meters_l874_87453


namespace rearrangements_of_13358_l874_87472

/-- The number of different five-digit numbers that can be formed by rearranging the digits in 13358 -/
def rearrangements : ℕ :=
  Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)

/-- Theorem stating that the number of rearrangements is 60 -/
theorem rearrangements_of_13358 : rearrangements = 60 := by
  sorry

end rearrangements_of_13358_l874_87472


namespace curve_tangent_to_line_l874_87479

/-- A curve y = e^x + a is tangent to the line y = x if and only if a = -1 -/
theorem curve_tangent_to_line (a : ℝ) : 
  (∃ x₀ : ℝ, (Real.exp x₀ + a = x₀) ∧ (Real.exp x₀ = 1)) ↔ a = -1 :=
sorry

end curve_tangent_to_line_l874_87479


namespace not_proportional_six_nine_nine_twelve_l874_87404

/-- Two ratios a:b and c:d are proportional if a/b = c/d -/
def proportional (a b c d : ℚ) : Prop := a / b = c / d

/-- The ratios 6:9 and 9:12 -/
def ratio1 : ℚ := 6 / 9
def ratio2 : ℚ := 9 / 12

/-- Theorem stating that 6:9 and 9:12 are not proportional -/
theorem not_proportional_six_nine_nine_twelve : ¬(proportional 6 9 9 12) := by
  sorry

end not_proportional_six_nine_nine_twelve_l874_87404


namespace number_operation_result_l874_87430

theorem number_operation_result : 
  let x : ℚ := 33
  (x / 4) + 9 = 17.25 := by sorry

end number_operation_result_l874_87430


namespace quadratic_minimum_l874_87426

theorem quadratic_minimum (x y : ℝ) : 
  y = x^2 + 16*x + 20 → (∀ z : ℝ, z = x^2 + 16*x + 20 → y ≤ z) → y = -44 :=
by sorry

end quadratic_minimum_l874_87426


namespace sams_carrots_l874_87437

theorem sams_carrots (sandy_carrots : ℕ) (total_carrots : ℕ) (h1 : sandy_carrots = 6) (h2 : total_carrots = 9) : 
  total_carrots - sandy_carrots = 3 := by
  sorry

end sams_carrots_l874_87437


namespace systematic_sampling_l874_87401

theorem systematic_sampling (population : ℕ) (sample_size : ℕ) 
  (h_pop : population = 1650) (h_sample : sample_size = 35) :
  ∃ (removed : ℕ) (segments : ℕ) (per_segment : ℕ),
    removed = 5 ∧ 
    segments = sample_size ∧
    per_segment = 47 ∧
    (population - removed) = segments * per_segment :=
by sorry

end systematic_sampling_l874_87401


namespace diagonal_length_l874_87454

/-- A quadrilateral with specific side lengths and an integer diagonal -/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  AC : ℤ
  h1 : AB = 9
  h2 : BC = 2
  h3 : CD = 14
  h4 : DA = 5

/-- The diagonal AC of the quadrilateral is 10 -/
theorem diagonal_length (q : Quadrilateral) : q.AC = 10 := by
  sorry

end diagonal_length_l874_87454


namespace divisibility_by_five_l874_87495

theorem divisibility_by_five (k m n : ℕ+) 
  (hk : ¬ 5 ∣ k.val) (hm : ¬ 5 ∣ m.val) (hn : ¬ 5 ∣ n.val) : 
  5 ∣ (k.val^2 - m.val^2) ∨ 5 ∣ (m.val^2 - n.val^2) ∨ 5 ∣ (n.val^2 - k.val^2) := by
  sorry

end divisibility_by_five_l874_87495


namespace sin_negative_780_degrees_l874_87422

theorem sin_negative_780_degrees : 
  Real.sin ((-780 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_negative_780_degrees_l874_87422


namespace trivia_team_score_l874_87498

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_points : ℕ) 
  (h1 : total_members = 15)
  (h2 : absent_members = 6)
  (h3 : total_points = 27) :
  total_points / (total_members - absent_members) = 3 := by
  sorry

end trivia_team_score_l874_87498


namespace object_max_height_time_l874_87486

/-- The height function of the thrown object -/
def h (t : ℝ) : ℝ := -15 * (t - 3)^2 + 150

/-- The time at which the object reaches its maximum height -/
def t_max : ℝ := 3

theorem object_max_height_time :
  (∀ t, h t ≤ h t_max) ∧ h (t_max + 2) = 90 :=
by sorry

end object_max_height_time_l874_87486


namespace dice_product_six_prob_l874_87481

/-- The probability of rolling a specific number on a standard die -/
def die_prob : ℚ := 1 / 6

/-- The set of all possible outcomes when rolling three dice -/
def all_outcomes : Finset (ℕ × ℕ × ℕ) := sorry

/-- The set of favorable outcomes where the product of the three numbers is 6 -/
def favorable_outcomes : Finset (ℕ × ℕ × ℕ) := sorry

/-- The probability of rolling three dice such that their product is 6 -/
theorem dice_product_six_prob : 
  (Finset.card favorable_outcomes : ℚ) / (Finset.card all_outcomes : ℚ) = 1 / 24 := by sorry

end dice_product_six_prob_l874_87481


namespace boat_upstream_distance_l874_87466

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (boat_speed : ℝ) (stream_speed : ℝ) : ℝ :=
  boat_speed + stream_speed

theorem boat_upstream_distance 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : boat_speed = 11)
  (h2 : boat_distance boat_speed (downstream_distance - boat_speed) = 13) :
  boat_distance boat_speed (boat_speed - downstream_distance) = 9 := by
sorry

end boat_upstream_distance_l874_87466


namespace smallest_number_of_students_l874_87452

/-- Represents the number of students in each grade --/
structure Students where
  ninth : ℕ
  eighth : ℕ
  seventh : ℕ

/-- The ratio of 9th-graders to 8th-graders is 7:4 --/
def ratio_9th_8th (s : Students) : Prop :=
  7 * s.eighth = 4 * s.ninth

/-- The ratio of 9th-graders to 7th-graders is 10:3 --/
def ratio_9th_7th (s : Students) : Prop :=
  10 * s.seventh = 3 * s.ninth

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.ninth + s.eighth + s.seventh

/-- The main theorem stating the smallest possible number of students --/
theorem smallest_number_of_students :
  ∃ (s : Students),
    ratio_9th_8th s ∧
    ratio_9th_7th s ∧
    total_students s = 131 ∧
    (∀ (t : Students),
      ratio_9th_8th t → ratio_9th_7th t →
      total_students t ≥ total_students s) :=
  sorry

end smallest_number_of_students_l874_87452


namespace jack_christina_lindy_problem_l874_87433

/-- The problem setup and solution for Jack, Christina, and Lindy's movement --/
theorem jack_christina_lindy_problem (
  initial_distance : ℝ) 
  (jack_speed christina_speed lindy_speed : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : jack_speed = 7)
  (h3 : christina_speed = 8)
  (h4 : lindy_speed = 10) :
  let meeting_time := initial_distance / (jack_speed + christina_speed)
  lindy_speed * meeting_time = 100 := by
  sorry


end jack_christina_lindy_problem_l874_87433


namespace jacob_shooting_improvement_l874_87435

/-- Represents the number of shots Jacob made in the fourth game -/
def shots_made_fourth_game : ℕ := 9

/-- Represents Jacob's initial number of shots -/
def initial_shots : ℕ := 45

/-- Represents Jacob's initial number of successful shots -/
def initial_successful_shots : ℕ := 18

/-- Represents the number of shots Jacob attempted in the fourth game -/
def fourth_game_attempts : ℕ := 15

/-- Represents Jacob's initial shooting average as a rational number -/
def initial_average : ℚ := 2/5

/-- Represents Jacob's final shooting average as a rational number -/
def final_average : ℚ := 9/20

theorem jacob_shooting_improvement :
  (initial_successful_shots + shots_made_fourth_game : ℚ) / (initial_shots + fourth_game_attempts) = final_average :=
sorry

end jacob_shooting_improvement_l874_87435


namespace least_divisible_by_second_smallest_consecutive_primes_l874_87406

def second_smallest_consecutive_primes : List Nat := [11, 13, 17, 19]

theorem least_divisible_by_second_smallest_consecutive_primes :
  (∀ n : Nat, n > 0 ∧ (∀ p ∈ second_smallest_consecutive_primes, p ∣ n) → n ≥ 46189) ∧
  (∀ p ∈ second_smallest_consecutive_primes, p ∣ 46189) :=
sorry

end least_divisible_by_second_smallest_consecutive_primes_l874_87406


namespace cubic_function_properties_l874_87408

/-- A cubic function with a linear term -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x

theorem cubic_function_properties (m : ℝ) (h : f m 1 = 5) :
  m = 4 ∧ ∀ x : ℝ, f m (-x) = -(f m x) := by sorry

end cubic_function_properties_l874_87408


namespace cosine_range_theorem_l874_87474

theorem cosine_range_theorem (f : ℝ → ℝ) (x : ℝ) :
  (f = λ x => Real.cos (x - π/3)) →
  (x ∈ Set.Icc 0 (π/2)) →
  (∀ y, y ∈ Set.range f ↔ y ∈ Set.Icc (1/2) 1) :=
sorry

end cosine_range_theorem_l874_87474


namespace rose_work_days_l874_87420

/-- Proves that if John completes a work in 320 days, and both John and Rose together complete
    the same work in 192 days, then Rose completes the work alone in 384 days. -/
theorem rose_work_days (john_days : ℕ) (together_days : ℕ) (rose_days : ℕ) : 
  john_days = 320 → together_days = 192 → 
  1 / john_days + 1 / rose_days = 1 / together_days → 
  rose_days = 384 := by
sorry

end rose_work_days_l874_87420


namespace lcm_of_20_25_30_l874_87457

theorem lcm_of_20_25_30 : Nat.lcm (Nat.lcm 20 25) 30 = 300 := by
  sorry

end lcm_of_20_25_30_l874_87457


namespace cookingAndYogaCount_l874_87488

/-- Represents a group of people participating in various curriculums -/
structure CurriculumGroup where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- The number of people who study both cooking and yoga -/
def bothCookingAndYoga (g : CurriculumGroup) : ℕ :=
  g.cooking - g.cookingOnly - g.cookingAndWeaving + g.allCurriculums

/-- Theorem stating the number of people who study both cooking and yoga -/
theorem cookingAndYogaCount (g : CurriculumGroup) 
  (h1 : g.yoga = 35)
  (h2 : g.cooking = 20)
  (h3 : g.weaving = 15)
  (h4 : g.cookingOnly = 7)
  (h5 : g.allCurriculums = 3)
  (h6 : g.cookingAndWeaving = 5) :
  bothCookingAndYoga g = 5 := by
  sorry

#eval bothCookingAndYoga { yoga := 35, cooking := 20, weaving := 15, cookingOnly := 7, allCurriculums := 3, cookingAndWeaving := 5 }

end cookingAndYogaCount_l874_87488


namespace trapezoid_area_in_circle_l874_87473

/-- The area of a trapezoid inscribed in a circle -/
theorem trapezoid_area_in_circle (R : ℝ) (α : ℝ) (h : 0 < α ∧ α < π) :
  let trapezoid_area := R^2 * (1 + Real.sin (α/2)) * Real.cos (α/2)
  let diameter := 2 * R
  let chord := 2 * R * Real.sin (α/2)
  let height := R * Real.cos (α/2)
  trapezoid_area = (diameter + chord) * height / 2 :=
by sorry

end trapezoid_area_in_circle_l874_87473


namespace two_solutions_only_l874_87497

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem two_solutions_only : 
  {k : ℕ | k > 0 ∧ digit_product k = (25 * k) / 8 - 211} = {72, 88} :=
by sorry

end two_solutions_only_l874_87497


namespace chipped_marbles_count_l874_87414

/-- Represents the number of marbles in each bag -/
def bags : List Nat := [20, 22, 25, 30, 32, 34, 36]

/-- Represents the number of bags Jane takes -/
def jane_bags : Nat := 3

/-- Represents the number of bags George takes -/
def george_bags : Nat := 3

/-- The number of chipped marbles -/
def chipped_marbles : Nat := 22

theorem chipped_marbles_count :
  ∃ (jane_selection george_selection : List Nat),
    jane_selection.length = jane_bags ∧
    george_selection.length = george_bags ∧
    (∀ x, x ∈ jane_selection ∨ x ∈ george_selection → x ∈ bags) ∧
    (∀ x, x ∈ jane_selection → x ∉ george_selection) ∧
    (∀ x, x ∈ george_selection → x ∉ jane_selection) ∧
    (∃ remaining, remaining ∈ bags ∧
      remaining ∉ jane_selection ∧
      remaining ∉ george_selection ∧
      remaining = chipped_marbles ∧
      (jane_selection.sum + george_selection.sum = 3 * remaining)) :=
sorry

end chipped_marbles_count_l874_87414


namespace perpendicular_line_l874_87462

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line (x y : ℝ) : 
  (∃ (m b : ℝ), (3 * x - 6 * y = 9) ∧ (y = m * x + b)) →  -- L1 equation
  (y = -2 * x + 1) →                                     -- L2 equation
  ((-2) * (1/2) = -1) →                                  -- Perpendicularity condition
  ((-3) = -2 * 2 + 1) →                                  -- Point P satisfies L2
  (∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = -3 ∧ y₀ = -2 * x₀ + 1) -- L2 passes through P
  := by sorry

end perpendicular_line_l874_87462


namespace expression_evaluation_l874_87463

theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = 1) :
  2 * (x - y) - 3 * (2 * x - y) + y = 10 := by
  sorry

end expression_evaluation_l874_87463


namespace megan_shirt_payment_l874_87439

/-- The amount Megan pays for a shirt after discount -/
def shirt_price (original_price discount : ℕ) : ℕ :=
  original_price - discount

/-- Theorem: Megan pays $16 for the shirt -/
theorem megan_shirt_payment : shirt_price 22 6 = 16 := by
  sorry

end megan_shirt_payment_l874_87439


namespace tshirt_sale_duration_l874_87434

/-- Calculates the duration of a t-shirt sale given the number of shirts sold,
    their prices, and the revenue rate per minute. -/
theorem tshirt_sale_duration
  (total_shirts : ℕ)
  (black_shirts : ℕ)
  (white_shirts : ℕ)
  (black_price : ℚ)
  (white_price : ℚ)
  (revenue_rate : ℚ)
  (h1 : total_shirts = 200)
  (h2 : black_shirts = total_shirts / 2)
  (h3 : white_shirts = total_shirts / 2)
  (h4 : black_price = 30)
  (h5 : white_price = 25)
  (h6 : revenue_rate = 220) :
  (black_shirts * black_price + white_shirts * white_price) / revenue_rate = 25 := by
  sorry

end tshirt_sale_duration_l874_87434


namespace toaster_cost_l874_87425

def amazon_purchase : ℝ := 3000
def tv_cost : ℝ := 700
def returned_bike_cost : ℝ := 500
def sold_bike_cost : ℝ := returned_bike_cost * 1.2
def sold_bike_price : ℝ := sold_bike_cost * 0.8
def total_out_of_pocket : ℝ := 2020

theorem toaster_cost :
  let total_return := tv_cost + returned_bike_cost
  let out_of_pocket_before_toaster := amazon_purchase - total_return + sold_bike_price
  let toaster_cost := out_of_pocket_before_toaster - total_out_of_pocket
  toaster_cost = 260 :=
by sorry

end toaster_cost_l874_87425


namespace trapezoid_area_l874_87465

/-- Trapezoid ABCD with diagonals AC and BD, and midpoints P and S of AD and BC respectively -/
structure Trapezoid :=
  (A B C D P S : ℝ × ℝ)
  (is_trapezoid : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1))
  (diag_AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 8)
  (diag_BD_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 6)
  (P_midpoint : P = ((A.1 + D.1) / 2, (A.2 + D.2) / 2))
  (S_midpoint : S = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (PS_length : Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) = 5)

/-- The area of the trapezoid ABCD is 24 -/
theorem trapezoid_area (t : Trapezoid) : 
  (1 / 2) * Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) * 
  Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2) = 24 := by
  sorry

end trapezoid_area_l874_87465


namespace proposition_relationship_l874_87431

theorem proposition_relationship :
  (∀ a b : ℝ, (a > b ∧ a⁻¹ > b⁻¹) → a > 0) ∧
  ¬(∀ a b : ℝ, a > 0 → (a > b ∧ a⁻¹ > b⁻¹)) :=
by sorry

end proposition_relationship_l874_87431


namespace mathcounts_teach_probability_l874_87409

def mathcounts_letters : Finset Char := {'M', 'A', 'T', 'H', 'C', 'O', 'U', 'N', 'T', 'S'}
def teach_letters : Finset Char := {'T', 'E', 'A', 'C', 'H'}

theorem mathcounts_teach_probability :
  let common_letters := mathcounts_letters ∩ teach_letters
  (common_letters.card : ℚ) / mathcounts_letters.card = 1 / 2 := by
sorry

end mathcounts_teach_probability_l874_87409


namespace fixed_point_of_logarithmic_function_l874_87443

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x+1) + 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) + 2

-- Theorem statement
theorem fixed_point_of_logarithmic_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 0 = 2 := by
  sorry

end fixed_point_of_logarithmic_function_l874_87443


namespace books_obtained_l874_87455

/-- The number of additional books obtained by the class -/
def additional_books (initial final : ℕ) : ℕ := final - initial

/-- Proves that the number of additional books is 23 given the initial and final counts -/
theorem books_obtained (initial final : ℕ) 
  (h_initial : initial = 54)
  (h_final : final = 77) :
  additional_books initial final = 23 := by
  sorry

end books_obtained_l874_87455


namespace problem_solution_l874_87483

theorem problem_solution (x y : ℝ) 
  (sum_eq : x + y = 360)
  (ratio_eq : x / y = 3 / 5) : 
  y - x = 90 := by
sorry

end problem_solution_l874_87483


namespace divisibility_3_power_l874_87405

theorem divisibility_3_power (n : ℕ) : 
  (∃ k : ℤ, 3^n + 1 = 10 * k) → (∃ m : ℤ, 3^(n+4) + 1 = 10 * m) := by
sorry

end divisibility_3_power_l874_87405


namespace tan_alpha_value_l874_87445

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.cos (α + π / 4) = -3 / 5) : Real.tan α = 7 := by
  sorry

end tan_alpha_value_l874_87445


namespace min_value_of_expression_l874_87482

theorem min_value_of_expression (a b : ℝ) (h : a ≠ -1) :
  |a + b| + |1 / (a + 1) - b| ≥ 1 := by sorry

end min_value_of_expression_l874_87482


namespace octal_to_decimal_conversion_coral_age_conversion_l874_87438

-- Define the octal age
def octal_age : ℕ := 753

-- Define the decimal age
def decimal_age : ℕ := 491

-- Theorem to prove the equivalence
theorem octal_to_decimal_conversion :
  (3 * 8^0 + 5 * 8^1 + 7 * 8^2 : ℕ) = decimal_age :=
by sorry

-- Theorem to prove that octal_age in decimal is equal to decimal_age
theorem coral_age_conversion :
  octal_age.digits 8 = [3, 5, 7] ∧
  (3 * 8^0 + 5 * 8^1 + 7 * 8^2 : ℕ) = decimal_age :=
by sorry

end octal_to_decimal_conversion_coral_age_conversion_l874_87438


namespace quarter_power_inequality_l874_87440

theorem quarter_power_inequality (x y : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : y < 1) :
  (1/4 : ℝ)^x > (1/4 : ℝ)^y := by
  sorry

end quarter_power_inequality_l874_87440


namespace brittany_vacation_duration_l874_87492

/-- The duration of Brittany's vacation --/
def vacation_duration (rebecca_age : ℕ) (age_difference : ℕ) (brittany_age_after : ℕ) : ℕ :=
  brittany_age_after - (rebecca_age + age_difference)

/-- Theorem stating that Brittany's vacation lasted 4 years --/
theorem brittany_vacation_duration :
  vacation_duration 25 3 32 = 4 := by
  sorry

end brittany_vacation_duration_l874_87492


namespace quadratic_root_range_l874_87421

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + 4*a + 1

theorem quadratic_root_range (a : ℝ) :
  (∃ r₁ r₂ : ℝ, r₁ < -1 ∧ r₂ > 3 ∧ f a r₁ = 0 ∧ f a r₂ = 0) →
  a > 4/5 ∧ a < 1 :=
by sorry

end quadratic_root_range_l874_87421


namespace project_completion_time_l874_87458

/-- Represents the number of days it takes to complete the project. -/
def total_days : ℕ := 21

/-- Represents the rate at which A completes the project per day. -/
def rate_A : ℚ := 1 / 20

/-- Represents the rate at which B completes the project per day. -/
def rate_B : ℚ := 1 / 30

/-- Represents the combined rate at which A and B complete the project per day when working together. -/
def combined_rate : ℚ := rate_A + rate_B

theorem project_completion_time (x : ℕ) :
  (↑(total_days - x) * combined_rate + ↑x * rate_B = 1) → x = 15 := by
  sorry

end project_completion_time_l874_87458


namespace systematic_sample_fourth_element_l874_87424

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Generates the nth element of a systematic sample -/
def SystematicSample.nth_element (s : SystematicSample) (n : ℕ) : ℕ :=
  s.start + (n - 1) * s.interval

/-- Theorem: In a systematic sample of size 4 from a population of 50,
    if students with ID numbers 6, 30, and 42 are included,
    then the fourth student in the sample must have ID number 18 -/
theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop : s.population_size = 50)
  (h_sample : s.sample_size = 4)
  (h_start : s.start = 6)
  (h_interval : s.interval = 12)
  (h_30 : s.nth_element 3 = 30)
  (h_42 : s.nth_element 4 = 42) :
  s.nth_element 2 = 18 := by
  sorry


end systematic_sample_fourth_element_l874_87424


namespace median_longest_side_right_triangle_l874_87470

theorem median_longest_side_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let median := (max a (max b c)) / 2
  median = 5 := by
  sorry

end median_longest_side_right_triangle_l874_87470


namespace probability_at_least_one_green_l874_87468

theorem probability_at_least_one_green (total : ℕ) (red : ℕ) (green : ℕ) (choose : ℕ) :
  total = red + green →
  total = 10 →
  red = 6 →
  green = 4 →
  choose = 3 →
  (1 : ℚ) - (Nat.choose red choose : ℚ) / (Nat.choose total choose : ℚ) = 5 / 6 :=
by sorry

end probability_at_least_one_green_l874_87468


namespace find_divisor_l874_87448

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 139 →
  quotient = 7 →
  remainder = 6 →
  dividend = divisor * quotient + remainder →
  divisor = 19 := by
sorry

end find_divisor_l874_87448


namespace fifth_largest_divisor_of_n_l874_87402

def n : ℕ := 5040000000

-- Define a function to get the kth largest divisor
def kth_largest_divisor (k : ℕ) (n : ℕ) : ℕ :=
  sorry

theorem fifth_largest_divisor_of_n :
  kth_largest_divisor 5 n = 315000000 :=
sorry

end fifth_largest_divisor_of_n_l874_87402


namespace ben_hit_seven_l874_87469

-- Define the set of friends
inductive Friend
| Alice | Ben | Cindy | Dave | Ellen | Frank

-- Define the scores for each friend
def score (f : Friend) : ℕ :=
  match f with
  | Friend.Alice => 18
  | Friend.Ben => 13
  | Friend.Cindy => 19
  | Friend.Dave => 16
  | Friend.Ellen => 20
  | Friend.Frank => 5

-- Define the set of possible target scores
def targetScores : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

-- Define a function to check if a pair of scores is valid
def validPair (a b : ℕ) : Prop :=
  a ∈ targetScores ∧ b ∈ targetScores ∧ a ≠ b ∧ a + b = score Friend.Ben

-- Theorem statement
theorem ben_hit_seven :
  ∃ (a b : ℕ), validPair a b ∧ (a = 7 ∨ b = 7) ∧
  (∀ (f : Friend), f ≠ Friend.Ben → ¬∃ (x y : ℕ), validPair x y ∧ (x = 7 ∨ y = 7)) :=
sorry

end ben_hit_seven_l874_87469


namespace difference_of_squares_65_35_l874_87436

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end difference_of_squares_65_35_l874_87436


namespace complex_abs_value_l874_87475

theorem complex_abs_value : Complex.abs (-3 - (8/5)*Complex.I) = 17/5 := by
  sorry

end complex_abs_value_l874_87475


namespace max_value_of_b_l874_87494

/-- Given functions f and g with a common point and tangent, prove the maximum value of b -/
theorem max_value_of_b (a : ℝ) (h_a : a > 0) : 
  let f := fun x : ℝ => (1/2) * x^2 + 2 * a * x
  let g := fun x b : ℝ => 3 * a^2 * Real.log x + b
  ∃ (x₀ b₀ : ℝ), 
    (f x₀ = g x₀ b₀) ∧ 
    (deriv f x₀ = deriv (fun x => g x b₀) x₀) →
  (∀ b : ℝ, ∃ (x : ℝ), (f x = g x b) ∧ (deriv f x = deriv (fun x => g x b) x) → b ≤ (3/2) * Real.exp ((2/3) : ℝ)) :=
by sorry

end max_value_of_b_l874_87494


namespace janine_reading_theorem_l874_87411

/-- The number of books Janine read last month -/
def books_last_month : ℕ := 5

/-- The number of books Janine read this month -/
def books_this_month : ℕ := 2 * books_last_month

/-- The number of pages in each book -/
def pages_per_book : ℕ := 10

/-- The total number of pages Janine read in two months -/
def total_pages : ℕ := (books_last_month + books_this_month) * pages_per_book

theorem janine_reading_theorem : total_pages = 150 := by
  sorry

end janine_reading_theorem_l874_87411


namespace original_mixture_composition_l874_87432

def original_mixture (acid water : ℝ) : Prop :=
  acid > 0 ∧ water > 0

def after_adding_water (acid water : ℝ) : Prop :=
  acid / (acid + water + 2) = 1/4

def after_adding_acid (acid water : ℝ) : Prop :=
  (acid + 3) / (acid + water + 5) = 2/5

theorem original_mixture_composition (acid water : ℝ) :
  original_mixture acid water →
  after_adding_water acid water →
  after_adding_acid acid water →
  acid / (acid + water) = 3/10 :=
by sorry

end original_mixture_composition_l874_87432


namespace probability_both_selected_l874_87456

theorem probability_both_selected (prob_ram : ℚ) (prob_ravi : ℚ) 
  (h1 : prob_ram = 5 / 7) 
  (h2 : prob_ravi = 1 / 5) : 
  prob_ram * prob_ravi = 1 / 7 := by
  sorry

end probability_both_selected_l874_87456


namespace hyperbola_intersection_length_l874_87447

/-- Given a hyperbola with imaginary axis length 4 and eccentricity √6/2,
    if a line through the left focus intersects the left branch at points A and B
    such that |AB| is the arithmetic mean of |AF₂| and |BF₂|, then |AB| = 8√2 -/
theorem hyperbola_intersection_length
  (b : ℝ) (e : ℝ) (A B F₁ F₂ : ℝ × ℝ)
  (h_b : b = 2)
  (h_e : e = Real.sqrt 6 / 2)
  (h_foci : F₁.1 < F₂.1)
  (h_left_branch : A.1 < F₁.1 ∧ B.1 < F₁.1)
  (h_line : ∃ (m k : ℝ), A.2 = m * A.1 + k ∧ B.2 = m * B.1 + k ∧ F₁.2 = m * F₁.1 + k)
  (h_arithmetic_mean : 2 * dist A B = dist A F₂ + dist B F₂)
  (h_hyperbola : dist A F₂ - dist A F₁ = dist B F₂ - dist B F₁) :
  dist A B = 8 * Real.sqrt 2 :=
sorry

end hyperbola_intersection_length_l874_87447


namespace pizza_combinations_l874_87403

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 92 :=
by sorry

end pizza_combinations_l874_87403


namespace tangent_circles_radius_l874_87451

/-- Two circles are tangent if their centers' distance equals the sum or difference of their radii -/
def are_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂ ∨ d = |r₁ - r₂|

theorem tangent_circles_radius (r₁ r₂ d : ℝ) (h₁ : r₁ = 2) (h₂ : d = 5) 
  (h₃ : are_tangent r₁ r₂ d) : r₂ = 3 ∨ r₂ = 7 := by
  sorry

end tangent_circles_radius_l874_87451


namespace pages_read_l874_87467

/-- Given a book with a total number of pages and the number of pages left to read,
    calculate the number of pages already read. -/
theorem pages_read (total_pages left_to_read : ℕ) : 
  total_pages = 17 → left_to_read = 6 → total_pages - left_to_read = 11 := by
  sorry

end pages_read_l874_87467


namespace debby_drinks_six_bottles_per_day_l874_87459

-- Define the total number of bottles
def total_bottles : ℕ := 12

-- Define the number of days the bottles last
def days_last : ℕ := 2

-- Define the function to calculate bottles per day
def bottles_per_day (total : ℕ) (days : ℕ) : ℚ :=
  (total : ℚ) / (days : ℚ)

-- Theorem statement
theorem debby_drinks_six_bottles_per_day :
  bottles_per_day total_bottles days_last = 6 := by
  sorry

end debby_drinks_six_bottles_per_day_l874_87459


namespace m_value_l874_87442

theorem m_value (a b : ℝ) (m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = 10 := by
  sorry

end m_value_l874_87442


namespace quadratic_root_in_interval_l874_87419

/-- A quadratic function f(x) = ax^2 + bx + c has a root in the interval (-2, 0),
    given that 2a + c/2 > b and c < 0. -/
theorem quadratic_root_in_interval (a b c : ℝ) (h1 : 2 * a + c / 2 > b) (h2 : c < 0) :
  ∃ x : ℝ, x ∈ Set.Ioo (-2 : ℝ) 0 ∧ a * x^2 + b * x + c = 0 := by
  sorry

end quadratic_root_in_interval_l874_87419


namespace integer_triple_solution_l874_87410

theorem integer_triple_solution (x y z : ℤ) :
  x * y * z + 4 * (x + y + z) = 2 * (x * y + x * z + y * z) + 7 ↔
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 3 ∧ y = 3 ∧ z = 1) ∨
  (x = 3 ∧ y = 1 ∧ z = 3) ∨
  (x = 1 ∧ y = 3 ∧ z = 3) :=
by sorry

end integer_triple_solution_l874_87410


namespace petya_wins_l874_87441

/-- Represents the game between Petya and Vasya -/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game with the given conditions -/
def game : CandyGame :=
  { total_candies := 25,
    prob_two_caramels := 0.54 }

/-- Theorem: Petya has a higher chance of winning -/
theorem petya_wins (g : CandyGame) 
  (h1 : g.total_candies = 25)
  (h2 : g.prob_two_caramels = 0.54) :
  g.prob_two_caramels > 1 - g.prob_two_caramels := by
  sorry

#check petya_wins game

end petya_wins_l874_87441


namespace unique_angle_solution_l874_87427

theorem unique_angle_solution :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
    Real.tan ((150 - x) * π / 180) = 
      (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
      (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
    x = 110 := by
  sorry

end unique_angle_solution_l874_87427


namespace impossible_equal_side_sums_l874_87423

/-- Represents the pattern of squares in the problem -/
structure SquarePattern :=
  (vertices : Fin 24 → ℕ)
  (is_consecutive : ∀ i : Fin 23, vertices i.succ = vertices i + 1)
  (is_bijective : Function.Bijective vertices)

/-- Represents a side of a square in the pattern -/
inductive Side : Type
| Top : Side
| Right : Side
| Bottom : Side
| Left : Side

/-- Gets the vertices on a given side of a square -/
def side_vertices (square : Fin 4) (side : Side) : Fin 24 → Prop :=
  sorry

/-- The sum of numbers on a side of a square -/
def side_sum (p : SquarePattern) (square : Fin 4) (side : Side) : ℕ :=
  sorry

/-- The theorem stating the impossibility of the required arrangement -/
theorem impossible_equal_side_sums :
  ¬ ∃ (p : SquarePattern),
    ∀ (s1 s2 : Fin 4) (side1 side2 : Side),
      side_sum p s1 side1 = side_sum p s2 side2 :=
sorry

end impossible_equal_side_sums_l874_87423


namespace fractional_equation_solution_range_l874_87428

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ m / (x - 1) + 3 / (1 - x) = 1) ↔ (m ≥ 2 ∧ m ≠ 3) :=
sorry

end fractional_equation_solution_range_l874_87428


namespace five_balls_four_boxes_l874_87415

/-- The number of ways to distribute n identical balls into k distinct boxes with at least one ball in each box -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 4 ways to distribute 5 identical balls into 4 distinct boxes with at least one ball in each box -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 4 := by sorry

end five_balls_four_boxes_l874_87415


namespace election_votes_theorem_l874_87413

theorem election_votes_theorem (total_votes : ℕ) : 
  (∃ (candidate_votes rival_votes : ℕ),
    candidate_votes = total_votes / 10 ∧
    rival_votes = candidate_votes + 16000 ∧
    candidate_votes + rival_votes = total_votes) →
  total_votes = 20000 := by
sorry

end election_votes_theorem_l874_87413


namespace exists_parallel_line_l874_87485

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersects : Plane → Plane → Prop)
variable (not_perpendicular : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem exists_parallel_line 
  (α β γ : Plane)
  (h1 : perpendicular β γ)
  (h2 : intersects α γ)
  (h3 : not_perpendicular α γ) :
  ∃ (a : Line), in_plane a α ∧ parallel a γ :=
sorry

end exists_parallel_line_l874_87485
