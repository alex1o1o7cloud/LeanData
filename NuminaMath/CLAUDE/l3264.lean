import Mathlib

namespace NUMINAMATH_CALUDE_farmers_herd_size_l3264_326495

theorem farmers_herd_size :
  ∀ (total : ℚ),
  (2 / 5 : ℚ) * total + (1 / 5 : ℚ) * total + (1 / 10 : ℚ) * total + 9 = total →
  total = 30 := by
sorry

end NUMINAMATH_CALUDE_farmers_herd_size_l3264_326495


namespace NUMINAMATH_CALUDE_set_intersection_problem_l3264_326456

theorem set_intersection_problem :
  let M : Set ℝ := {x | x^2 - x = 0}
  let N : Set ℝ := {-1, 0}
  M ∩ N = {0} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l3264_326456


namespace NUMINAMATH_CALUDE_extreme_values_and_tangent_lines_l3264_326425

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem extreme_values_and_tangent_lines :
  ∃ (a b : ℝ),
    (f' a b (-2/3) = 0 ∧ f' a b 1 = 0) ∧
    (a = -1/2 ∧ b = -2) ∧
    (∃ (t₁ t₂ : ℝ),
      (f a b t₁ - 1 = (f' a b t₁) * (-t₁) ∧ 2*t₁ + (f a b t₁) - 1 = 0) ∧
      (f a b t₂ - 1 = (f' a b t₂) * (-t₂) ∧ 33*t₂ + 16*(f a b t₂) - 16 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_tangent_lines_l3264_326425


namespace NUMINAMATH_CALUDE_youngest_boy_age_l3264_326483

/-- Given three boys whose ages are in proportion 2 : 6 : 8 and whose average age is 120 years,
    the age of the youngest boy is 45 years. -/
theorem youngest_boy_age (a b c : ℕ) : 
  a + b + c = 360 →  -- Sum of ages is 360 (3 * 120)
  3 * a = b →        -- b is 3 times a
  4 * a = c →        -- c is 4 times a
  a = 45 :=          -- The age of the youngest boy (a) is 45
by sorry

end NUMINAMATH_CALUDE_youngest_boy_age_l3264_326483


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3264_326480

/-- A line passing through (1, -1) and perpendicular to 3x - 2y = 0 has the equation 2x + 3y + 1 = 0 -/
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
  (2 * x + 3 * y + 1 = 0) ↔
  (∃ (m : ℝ), (y - (-1) = m * (x - 1)) ∧ 
              (m * 3 = -1/2) ∧
              (2 * 1 + 3 * (-1) + 1 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3264_326480


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l3264_326475

def initial_cards : ℕ := 573
def bought_cards : ℕ := 127
def cards_to_john : ℕ := 195
def cards_to_jimmy : ℕ := 75
def percentage_to_jeff : ℚ := 6 / 100
def final_cards : ℕ := 210

theorem cards_given_to_jeff :
  let total_cards := initial_cards + bought_cards
  let cards_after_john_jimmy := total_cards - (cards_to_john + cards_to_jimmy)
  let cards_to_jeff := (percentage_to_jeff * cards_after_john_jimmy).ceil
  cards_to_jeff = 26 ∧ 
  final_cards + cards_to_jeff = cards_after_john_jimmy :=
sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l3264_326475


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l3264_326464

/-- Function to transform a digit according to the problem rules -/
def transformDigit (d : Nat) : Nat :=
  match d with
  | 2 => 5
  | 5 => 2
  | _ => d

/-- Function to transform a five-digit number according to the problem rules -/
def transformNumber (n : Nat) : Nat :=
  let d1 := n / 10000
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  10000 * (transformDigit d1) + 1000 * (transformDigit d2) + 100 * (transformDigit d3) + 10 * (transformDigit d4) + (transformDigit d5)

/-- The main theorem statement -/
theorem unique_five_digit_number :
  ∃! x : Nat, 
    10000 ≤ x ∧ x < 100000 ∧  -- x is a five-digit number
    x % 2 = 1 ∧               -- x is odd
    transformNumber x = 2 * (x + 1) ∧ -- y = 2(x+1)
    x = 29995 := by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l3264_326464


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l3264_326403

/-- Calculates the distance traveled downstream by a boat -/
theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (travel_time : ℝ) 
  (h1 : boat_speed = 22) 
  (h2 : stream_speed = 5) 
  (h3 : travel_time = 4) : 
  boat_speed + stream_speed * travel_time = 108 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l3264_326403


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3264_326479

theorem modulus_of_complex_number : 
  Complex.abs (Complex.mk 1 (-2)) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3264_326479


namespace NUMINAMATH_CALUDE_sqrt_nine_plus_sixteen_l3264_326472

theorem sqrt_nine_plus_sixteen : Real.sqrt (9 + 16) = 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_nine_plus_sixteen_l3264_326472


namespace NUMINAMATH_CALUDE_peach_difference_l3264_326436

theorem peach_difference (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 17) 
  (h2 : green_peaches = 16) : 
  red_peaches - green_peaches = 1 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l3264_326436


namespace NUMINAMATH_CALUDE_unique_exam_scores_l3264_326494

def is_valid_score_set (scores : List Nat) : Prop :=
  scores.length = 5 ∧
  scores.all (λ x => x % 2 = 1 ∧ x < 100) ∧
  scores.Nodup ∧
  scores.sum / scores.length = 80 ∧
  [95, 85, 75, 65].all (λ x => x ∈ scores)

theorem unique_exam_scores :
  ∃! scores : List Nat, is_valid_score_set scores ∧ scores = [95, 85, 79, 75, 65] := by
  sorry

end NUMINAMATH_CALUDE_unique_exam_scores_l3264_326494


namespace NUMINAMATH_CALUDE_consecutive_tickets_product_l3264_326409

/-- Given 5 consecutive integers whose sum is 120, their product is 7893600 -/
theorem consecutive_tickets_product (x : ℤ) 
  (h_sum : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 120) : 
  x * (x + 1) * (x + 2) * (x + 3) * (x + 4) = 7893600 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_tickets_product_l3264_326409


namespace NUMINAMATH_CALUDE_cupcake_business_net_profit_l3264_326445

/-- Calculates the net profit from a cupcake business given the following conditions:
  * Each cupcake costs $0.75 to make
  * First 2 dozen cupcakes burnt and were thrown out
  * Next 2 dozen came out perfectly
  * 5 cupcakes were eaten right away
  * Later made 2 more dozen cupcakes
  * 4 more cupcakes were eaten
  * Remaining cupcakes are sold at $2.00 each
-/
theorem cupcake_business_net_profit :
  let cost_per_cupcake : ℚ := 75 / 100
  let sell_price : ℚ := 2
  let dozen : ℕ := 12
  let burnt_cupcakes : ℕ := 2 * dozen
  let eaten_cupcakes : ℕ := 5 + 4
  let total_cupcakes : ℕ := 6 * dozen
  let remaining_cupcakes : ℕ := total_cupcakes - burnt_cupcakes - eaten_cupcakes
  let revenue : ℚ := remaining_cupcakes * sell_price
  let total_cost : ℚ := total_cupcakes * cost_per_cupcake
  let net_profit : ℚ := revenue - total_cost
  net_profit = 24 := by sorry

end NUMINAMATH_CALUDE_cupcake_business_net_profit_l3264_326445


namespace NUMINAMATH_CALUDE_certain_number_proof_l3264_326454

theorem certain_number_proof : ∃! x : ℝ, x + (1/4 * 48) = 27 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3264_326454


namespace NUMINAMATH_CALUDE_count_integer_pairs_l3264_326422

theorem count_integer_pairs : ∃ (count : ℕ), 
  count = (Finset.filter (fun p : ℕ × ℕ => 
    let m := p.1
    let n := p.2
    1 ≤ m ∧ m ≤ 2887 ∧ 
    (7 : ℝ)^n < 3^m ∧ 3^m < 3^(m+3) ∧ 3^(m+3) < 7^(n+1))
  (Finset.product (Finset.range 2888) (Finset.range (3^2889 / 7^1233 + 1)))).card ∧
  3^2888 < 7^1233 ∧ 7^1233 < 3^2889 ∧
  count = 2466 :=
by sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l3264_326422


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l3264_326408

theorem two_digit_number_problem : ∃ (n : ℕ), 
  (n ≥ 10 ∧ n < 100) ∧  -- two-digit number
  (n % 10 = (n / 10) + 3) ∧  -- units digit is 3 greater than tens digit
  ((n % 10)^2 + (n / 10)^2 = (n % 10) + (n / 10) + 18) ∧  -- sum of squares condition
  n = 47 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l3264_326408


namespace NUMINAMATH_CALUDE_percentage_of_singles_is_70_percent_l3264_326497

def total_hits : ℕ := 50
def home_runs : ℕ := 3
def triples : ℕ := 2
def doubles : ℕ := 10

def singles : ℕ := total_hits - (home_runs + triples + doubles)

theorem percentage_of_singles_is_70_percent :
  (singles : ℚ) / total_hits * 100 = 70 := by sorry

end NUMINAMATH_CALUDE_percentage_of_singles_is_70_percent_l3264_326497


namespace NUMINAMATH_CALUDE_decimal_point_problem_l3264_326400

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 / x) : x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l3264_326400


namespace NUMINAMATH_CALUDE_equal_integers_from_gcd_l3264_326477

theorem equal_integers_from_gcd (a b : ℤ) 
  (h : ∀ (n : ℤ), n ≥ 1 → Nat.gcd (Int.natAbs (a + n)) (Int.natAbs (b + n)) > 1) : 
  a = b := by
  sorry

end NUMINAMATH_CALUDE_equal_integers_from_gcd_l3264_326477


namespace NUMINAMATH_CALUDE_no_sum_2017_double_digits_l3264_326466

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the impossibility of expressing 2017 as the sum of two natural numbers
    where the sum of digits of one is twice the sum of digits of the other -/
theorem no_sum_2017_double_digits : ¬ ∃ (A B : ℕ), 
  (A + B = 2017) ∧ (sumOfDigits A = 2 * sumOfDigits B) := by
  sorry

end NUMINAMATH_CALUDE_no_sum_2017_double_digits_l3264_326466


namespace NUMINAMATH_CALUDE_coins_collected_in_hours_2_3_l3264_326429

/-- Represents the number of coins collected in each hour -/
structure CoinCollection where
  hour1 : ℕ
  hour2_3 : ℕ
  hour4 : ℕ
  given_away : ℕ
  total : ℕ

/-- The coin collection scenario for Joanne -/
def joannes_collection : CoinCollection where
  hour1 := 15
  hour2_3 := 0  -- This is what we need to prove
  hour4 := 50
  given_away := 15
  total := 120

/-- Theorem stating that Joanne collected 70 coins in hours 2 and 3 -/
theorem coins_collected_in_hours_2_3 :
  joannes_collection.hour2_3 = 70 :=
by sorry

end NUMINAMATH_CALUDE_coins_collected_in_hours_2_3_l3264_326429


namespace NUMINAMATH_CALUDE_zeros_of_continuous_function_l3264_326447

theorem zeros_of_continuous_function 
  (f : ℝ → ℝ) (hf : Continuous f) 
  (a b c : ℝ) (hab : a < b) (hbc : b < c)
  (hab_sign : f a * f b < 0) (hbc_sign : f b * f c < 0) :
  ∃ (n : ℕ), n > 0 ∧ Even n ∧ 
  (∃ (S : Finset ℝ), S.card = n ∧ 
    (∀ x ∈ S, a < x ∧ x < c ∧ f x = 0)) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_continuous_function_l3264_326447


namespace NUMINAMATH_CALUDE_new_city_building_count_l3264_326432

/-- Represents the number of buildings of each type in Pittsburgh -/
structure PittsburghBuildings where
  stores : Nat
  hospitals : Nat
  schools : Nat
  police_stations : Nat

/-- Calculates the total number of buildings for the new city based on Pittsburgh's data -/
def new_city_buildings (p : PittsburghBuildings) : Nat :=
  p.stores / 2 + p.hospitals * 2 + (p.schools - 50) + (p.police_stations + 5)

/-- The theorem stating that given Pittsburgh's building numbers, the new city will require 2175 buildings -/
theorem new_city_building_count (p : PittsburghBuildings) 
  (h1 : p.stores = 2000)
  (h2 : p.hospitals = 500)
  (h3 : p.schools = 200)
  (h4 : p.police_stations = 20) :
  new_city_buildings p = 2175 := by
  sorry

end NUMINAMATH_CALUDE_new_city_building_count_l3264_326432


namespace NUMINAMATH_CALUDE_x_minus_q_equals_three_minus_two_q_l3264_326427

theorem x_minus_q_equals_three_minus_two_q (x q : ℝ) 
  (h1 : |x - 3| = q) 
  (h2 : x < 3) : 
  x - q = 3 - 2*q := by
  sorry

end NUMINAMATH_CALUDE_x_minus_q_equals_three_minus_two_q_l3264_326427


namespace NUMINAMATH_CALUDE_richard_twice_scott_age_l3264_326470

/-- The number of years until Richard is twice as old as Scott -/
def years_until_double : ℕ := 8

/-- David's current age -/
def david_age : ℕ := 14

/-- Richard's current age -/
def richard_age : ℕ := david_age + 6

/-- Scott's current age -/
def scott_age : ℕ := david_age - 8

theorem richard_twice_scott_age : 
  richard_age + years_until_double = 2 * (scott_age + years_until_double) :=
by sorry

end NUMINAMATH_CALUDE_richard_twice_scott_age_l3264_326470


namespace NUMINAMATH_CALUDE_bus_students_count_l3264_326413

/-- Calculates the number of students on the bus after all stops -/
def final_students (initial : ℕ) (second_on second_off third_on third_off : ℕ) : ℕ :=
  initial + second_on - second_off + third_on - third_off

/-- Theorem stating the final number of students on the bus -/
theorem bus_students_count :
  final_students 39 29 12 35 18 = 73 := by
  sorry

end NUMINAMATH_CALUDE_bus_students_count_l3264_326413


namespace NUMINAMATH_CALUDE_art_arrangement_count_l3264_326446

/-- Represents the number of calligraphy works -/
def calligraphy_count : ℕ := 2

/-- Represents the number of painting works -/
def painting_count : ℕ := 2

/-- Represents the number of architectural designs -/
def architecture_count : ℕ := 1

/-- Represents the total number of art pieces -/
def total_art_pieces : ℕ := calligraphy_count + painting_count + architecture_count

/-- Calculates the number of arrangements of art pieces -/
def calculate_arrangements : ℕ :=
  sorry

/-- Theorem stating that the number of arrangements is 24 -/
theorem art_arrangement_count : calculate_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_art_arrangement_count_l3264_326446


namespace NUMINAMATH_CALUDE_solve_equation_l3264_326440

theorem solve_equation (x y : ℝ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3264_326440


namespace NUMINAMATH_CALUDE_smallest_a_for_sqrt_50a_l3264_326489

theorem smallest_a_for_sqrt_50a (a : ℕ) : (∃ k : ℕ, k^2 = 50 * a) → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_sqrt_50a_l3264_326489


namespace NUMINAMATH_CALUDE_train_distance_l3264_326419

/-- Represents the speed of a train in miles per minute -/
def train_speed : ℚ := 3 / 2.25

/-- Represents the duration of the journey in minutes -/
def journey_duration : ℚ := 120

/-- Theorem stating that the train will travel 160 miles in 2 hours -/
theorem train_distance : train_speed * journey_duration = 160 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l3264_326419


namespace NUMINAMATH_CALUDE_age_inconsistency_l3264_326452

theorem age_inconsistency (a b c d : ℝ) : 
  (a + c + d) / 3 = 30 →
  (a + c) / 2 = 32 →
  (b + d) / 2 = 34 →
  ¬(0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :=
by
  sorry

#check age_inconsistency

end NUMINAMATH_CALUDE_age_inconsistency_l3264_326452


namespace NUMINAMATH_CALUDE_distance_is_sqrt_152_l3264_326418

/-- The distance between two adjacent parallel lines intersecting a circle -/
def distance_between_lines (r : ℝ) (d : ℝ) : Prop :=
  ∃ (chord1 chord2 chord3 : ℝ),
    chord1 = 40 ∧ chord2 = 36 ∧ chord3 = 34 ∧
    40 * r^2 = 800 + 10 * d^2 ∧
    36 * r^2 = 648 + 9 * d^2 ∧
    d = Real.sqrt 152

/-- Theorem stating that the distance between two adjacent parallel lines is √152 -/
theorem distance_is_sqrt_152 :
  ∃ (r : ℝ), distance_between_lines r (Real.sqrt 152) :=
sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_152_l3264_326418


namespace NUMINAMATH_CALUDE_sphere_equation_implies_zero_difference_l3264_326482

theorem sphere_equation_implies_zero_difference (x y z : ℝ) :
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 →
  (x - y - z)^2002 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sphere_equation_implies_zero_difference_l3264_326482


namespace NUMINAMATH_CALUDE_average_of_multiples_of_10_l3264_326476

def multiples_of_10 : List ℕ := [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

theorem average_of_multiples_of_10 : 
  (List.sum multiples_of_10) / (List.length multiples_of_10) = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_10_l3264_326476


namespace NUMINAMATH_CALUDE_percentage_problem_l3264_326451

theorem percentage_problem (P : ℝ) : 
  (0.15 * (P / 100) * 0.5 * 5200 = 117) → P = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3264_326451


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3264_326448

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 2 + a 8 = 10) : 
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3264_326448


namespace NUMINAMATH_CALUDE_box_office_scientific_notation_l3264_326428

theorem box_office_scientific_notation :
  let billion : ℝ := 10^9
  let box_office : ℝ := 40.25 * billion
  box_office = 4.025 * 10^9 := by sorry

end NUMINAMATH_CALUDE_box_office_scientific_notation_l3264_326428


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3264_326435

theorem arithmetic_calculations :
  ((1 : ℤ) - 4 + 8 - 5 = -1) ∧ 
  (24 / (-3 : ℤ) - (-2)^3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3264_326435


namespace NUMINAMATH_CALUDE_calculation_proof_l3264_326401

theorem calculation_proof : (40 * 1505 - 20 * 1505) / 5 = 6020 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3264_326401


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l3264_326421

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 400) (h2 : cat_owners = 80) : 
  (cat_owners : ℝ) / total_students * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l3264_326421


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l3264_326411

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ θ : ℝ) : Prop :=
  ρ^2 - 4*ρ*(Real.cos θ) - 4*ρ*(Real.sin θ) + 7 = 0

/-- Line C₂ in polar coordinates -/
def C₂ (θ : ℝ) : Prop :=
  Real.tan θ = Real.sqrt 3

/-- Theorem stating the sum of reciprocals of distances to intersection points -/
theorem intersection_distance_sum :
  ∀ ρ₁ ρ₂ θ : ℝ,
  C₁ ρ₁ θ → C₁ ρ₂ θ → C₂ θ →
  1 / ρ₁ + 1 / ρ₂ = (2 + 2 * Real.sqrt 3) / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l3264_326411


namespace NUMINAMATH_CALUDE_exactly_one_and_two_red_mutually_exclusive_non_opposing_l3264_326430

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of drawing three balls -/
structure DrawOutcome :=
  (red_count : Nat)
  (white_count : Nat)
  (h_total : red_count + white_count = 3)

/-- The bag of balls -/
def bag : Multiset BallColor :=
  Multiset.replicate 5 BallColor.Red + Multiset.replicate 3 BallColor.White

/-- The event of drawing exactly one red ball -/
def exactly_one_red (outcome : DrawOutcome) : Prop :=
  outcome.red_count = 1

/-- The event of drawing exactly two red balls -/
def exactly_two_red (outcome : DrawOutcome) : Prop :=
  outcome.red_count = 2

/-- Two events are mutually exclusive -/
def mutually_exclusive (e1 e2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

/-- Two events are non-opposing -/
def non_opposing (e1 e2 : DrawOutcome → Prop) : Prop :=
  ∃ outcome, e1 outcome ∨ e2 outcome

theorem exactly_one_and_two_red_mutually_exclusive_non_opposing :
  mutually_exclusive exactly_one_red exactly_two_red ∧
  non_opposing exactly_one_red exactly_two_red :=
sorry

end NUMINAMATH_CALUDE_exactly_one_and_two_red_mutually_exclusive_non_opposing_l3264_326430


namespace NUMINAMATH_CALUDE_inscribed_circle_ratio_l3264_326493

-- Define the quadrilateral ABCD
variable (A B C D : Point)

-- Define the inscribed circle P
variable (P : Point)

-- Define that P is the center of the inscribed circle
def is_inscribed_center (P : Point) (A B C D : Point) : Prop := sorry

-- Define the distance function
def distance (P Q : Point) : ℝ := sorry

-- State the theorem
theorem inscribed_circle_ratio 
  (h : is_inscribed_center P A B C D) :
  (distance P A)^2 / (distance P C)^2 = 
  (distance A B * distance A D) / (distance B C * distance C D) := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_ratio_l3264_326493


namespace NUMINAMATH_CALUDE_art_department_probability_l3264_326443

theorem art_department_probability : 
  let total_students : ℕ := 4
  let students_per_grade : ℕ := 2
  let selected_students : ℕ := 2
  let different_grade_selections : ℕ := students_per_grade * students_per_grade
  let total_selections : ℕ := Nat.choose total_students selected_students
  (different_grade_selections : ℚ) / total_selections = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_art_department_probability_l3264_326443


namespace NUMINAMATH_CALUDE_final_savings_after_expense_increase_l3264_326410

/-- Calculates the final savings after expense increase -/
def finalSavings (salary : ℝ) (initialSavingsRate : ℝ) (expenseIncreaseRate : ℝ) : ℝ :=
  let initialExpenses := salary * (1 - initialSavingsRate)
  let newExpenses := initialExpenses * (1 + expenseIncreaseRate)
  salary - newExpenses

/-- Theorem stating that given the problem conditions, the final savings is 250 -/
theorem final_savings_after_expense_increase :
  finalSavings 6250 0.2 0.2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_final_savings_after_expense_increase_l3264_326410


namespace NUMINAMATH_CALUDE_internal_borders_length_l3264_326433

/-- Represents a square garden bed with integer side length -/
structure SquareBed where
  side : ℕ

/-- Represents a rectangular garden divided into square beds -/
structure Garden where
  width : ℕ
  height : ℕ
  beds : List SquareBed

/-- Calculates the total area of the garden -/
def Garden.area (g : Garden) : ℕ := g.width * g.height

/-- Calculates the total area covered by the beds -/
def Garden.bedArea (g : Garden) : ℕ := g.beds.map (fun b => b.side * b.side) |>.sum

/-- Calculates the perimeter of the garden -/
def Garden.perimeter (g : Garden) : ℕ := 2 * (g.width + g.height)

/-- Calculates the sum of perimeters of all beds -/
def Garden.bedPerimeters (g : Garden) : ℕ := g.beds.map (fun b => 4 * b.side) |>.sum

/-- Theorem stating the length of internal borders in a specific garden configuration -/
theorem internal_borders_length (g : Garden) : 
  g.width = 6 ∧ 
  g.height = 7 ∧ 
  g.beds.length = 5 ∧ 
  g.area = g.bedArea →
  (g.bedPerimeters - g.perimeter) / 2 = 15 := by
  sorry


end NUMINAMATH_CALUDE_internal_borders_length_l3264_326433


namespace NUMINAMATH_CALUDE_range_of_a_l3264_326488

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ∈ Set.union (Set.Iic (-2)) {1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3264_326488


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3264_326492

theorem fifteenth_student_age 
  (total_students : Nat) 
  (group1_students : Nat) 
  (group2_students : Nat) 
  (total_average_age : ℝ) 
  (group1_average_age : ℝ) 
  (group2_average_age : ℝ) :
  total_students = 15 →
  group1_students = 5 →
  group2_students = 9 →
  total_average_age = 15 →
  group1_average_age = 14 →
  group2_average_age = 16 →
  (total_students * total_average_age) - 
    (group1_students * group1_average_age + group2_students * group2_average_age) = 11 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l3264_326492


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3264_326420

/-- An arithmetic sequence with sum of first 5 terms equal to 15 and second term equal to 5 has common difference -2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ)  -- arithmetic sequence
  (S : ℕ → ℝ)  -- sum function
  (h1 : S 5 = 15)  -- sum of first 5 terms is 15
  (h2 : a 2 = 5)   -- second term is 5
  (h3 : ∀ n, S n = n * (a 1 + a n) / 2)  -- sum formula for arithmetic sequence
  (h4 : ∀ n, a (n + 1) = a n + d)  -- definition of arithmetic sequence
  : d = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3264_326420


namespace NUMINAMATH_CALUDE_integer_roots_l3264_326465

/-- A polynomial of degree 3 with integer coefficients -/
def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂ * x^2 + a₁ * x - 13

/-- The set of possible integer roots -/
def possible_roots : Set ℤ := {-13, -1, 1, 13}

/-- Theorem stating that the possible integer roots of the polynomial are -13, -1, 1, and 13 -/
theorem integer_roots (a₂ a₁ : ℤ) :
  ∀ x : ℤ, polynomial a₂ a₁ x = 0 → x ∈ possible_roots :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_l3264_326465


namespace NUMINAMATH_CALUDE_radio_operator_distribution_probability_radio_operator_distribution_probability_proof_l3264_326478

/-- The probability of each group having exactly one radio operator when 12 soldiers 
    (including 3 radio operators) are randomly divided into groups of 3, 4, and 5 soldiers. -/
theorem radio_operator_distribution_probability : ℝ :=
  let total_soldiers : ℕ := 12
  let radio_operators : ℕ := 3
  let group_sizes : List ℕ := [3, 4, 5]
  3 / 11

/-- Proof of the radio operator distribution probability theorem -/
theorem radio_operator_distribution_probability_proof :
  radio_operator_distribution_probability = 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_radio_operator_distribution_probability_radio_operator_distribution_probability_proof_l3264_326478


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3264_326406

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 - 3*X^2 + 2 : Polynomial ℝ) = (X^2 - 3) * q + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3264_326406


namespace NUMINAMATH_CALUDE_trent_total_travel_l3264_326471

/-- Represents the number of blocks Trent traveled -/
def trent_travel (blocks_to_bus_stop : ℕ) (blocks_on_bus : ℕ) : ℕ :=
  2 * (blocks_to_bus_stop + blocks_on_bus)

/-- Proves that Trent's total travel is 22 blocks given the problem conditions -/
theorem trent_total_travel :
  trent_travel 4 7 = 22 := by
  sorry

end NUMINAMATH_CALUDE_trent_total_travel_l3264_326471


namespace NUMINAMATH_CALUDE_combined_average_l3264_326468

/-- Given two sets of results, one with 80 results averaging 32 and another with 50 results averaging 56,
    prove that the average of all results combined is (80 * 32 + 50 * 56) / (80 + 50) -/
theorem combined_average (set1_count : Nat) (set1_avg : ℚ) (set2_count : Nat) (set2_avg : ℚ)
    (h1 : set1_count = 80)
    (h2 : set1_avg = 32)
    (h3 : set2_count = 50)
    (h4 : set2_avg = 56) :
  (set1_count * set1_avg + set2_count * set2_avg) / (set1_count + set2_count) =
    (80 * 32 + 50 * 56) / (80 + 50) := by
  sorry

end NUMINAMATH_CALUDE_combined_average_l3264_326468


namespace NUMINAMATH_CALUDE_olympic_photo_arrangements_l3264_326486

/-- Represents the number of athletes -/
def num_athletes : ℕ := 5

/-- Represents the number of athletes that can occupy the leftmost position -/
def num_leftmost_athletes : ℕ := 2

/-- Represents whether athlete A can occupy the rightmost position -/
def a_can_be_rightmost : Bool := false

/-- The total number of different arrangement possibilities -/
def total_arrangements : ℕ := 42

/-- Theorem stating that the total number of arrangements is 42 -/
theorem olympic_photo_arrangements :
  (num_athletes = 5) →
  (num_leftmost_athletes = 2) →
  (a_can_be_rightmost = false) →
  (total_arrangements = 42) := by
  sorry

end NUMINAMATH_CALUDE_olympic_photo_arrangements_l3264_326486


namespace NUMINAMATH_CALUDE_vector_subtraction_l3264_326463

/-- Given plane vectors a and b, prove that a - 2b equals (7, 3) -/
theorem vector_subtraction (a b : ℝ × ℝ) (ha : a = (3, 5)) (hb : b = (-2, 1)) :
  a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3264_326463


namespace NUMINAMATH_CALUDE_regression_lines_intersect_l3264_326441

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point (s, t) represents the average values of x and y -/
structure AveragePoint where
  s : ℝ
  t : ℝ

/-- Theorem: Two regression lines with the same average point intersect at that point -/
theorem regression_lines_intersect (t₁ t₂ : RegressionLine) (avg : AveragePoint) :
  (avg.s * t₁.slope + t₁.intercept = avg.t) →
  (avg.s * t₂.slope + t₂.intercept = avg.t) →
  ∃ (x y : ℝ), x = avg.s ∧ y = avg.t ∧ 
    y = x * t₁.slope + t₁.intercept ∧
    y = x * t₂.slope + t₂.intercept := by
  sorry


end NUMINAMATH_CALUDE_regression_lines_intersect_l3264_326441


namespace NUMINAMATH_CALUDE_sara_pumpkins_left_l3264_326449

/-- Given that Sara grew 43 pumpkins and rabbits ate 23 pumpkins, 
    prove that Sara has 20 pumpkins left. -/
theorem sara_pumpkins_left : 
  let total_grown : ℕ := 43
  let eaten_by_rabbits : ℕ := 23
  let pumpkins_left := total_grown - eaten_by_rabbits
  pumpkins_left = 20 := by sorry

end NUMINAMATH_CALUDE_sara_pumpkins_left_l3264_326449


namespace NUMINAMATH_CALUDE_product_xyz_l3264_326450

theorem product_xyz (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) : 
  x * y * z = -1 := by sorry

end NUMINAMATH_CALUDE_product_xyz_l3264_326450


namespace NUMINAMATH_CALUDE_polygon_square_equal_area_l3264_326416

/-- Given a polygon with perimeter 800 cm and each side tangent to a circle of radius 100 cm,
    the side length of a square with equal area is 200 cm. -/
theorem polygon_square_equal_area (polygon_perimeter : ℝ) (circle_radius : ℝ) :
  polygon_perimeter = 800 ∧ circle_radius = 100 →
  ∃ (square_side : ℝ),
    square_side = 200 ∧
    square_side ^ 2 = (polygon_perimeter * circle_radius) / 2 := by
  sorry

end NUMINAMATH_CALUDE_polygon_square_equal_area_l3264_326416


namespace NUMINAMATH_CALUDE_exist_x_y_different_squares_no_x_y_different_squares_in_range_l3264_326455

-- Define the property for two numbers to be different perfect squares
def areDifferentPerfectSquares (a b : ℕ) : Prop :=
  ∃ m n : ℕ, m ≠ n ∧ a = m^2 ∧ b = n^2

-- Theorem 1: Existence of x and y satisfying the condition
theorem exist_x_y_different_squares :
  ∃ x y : ℕ, areDifferentPerfectSquares (x*y + x) (x*y + y) :=
sorry

-- Theorem 2: Non-existence of x and y between 988 and 1991 satisfying the condition
theorem no_x_y_different_squares_in_range :
  ¬∃ x y : ℕ, 988 ≤ x ∧ x ≤ 1991 ∧ 988 ≤ y ∧ y ≤ 1991 ∧
    areDifferentPerfectSquares (x*y + x) (x*y + y) :=
sorry

end NUMINAMATH_CALUDE_exist_x_y_different_squares_no_x_y_different_squares_in_range_l3264_326455


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3264_326405

variables (x y : ℝ)

theorem polynomial_multiplication :
  (2 * x^25 - 5 * x^8 + 2 * x * y^3 - 9) * (3 * x^7) =
  6 * x^32 - 15 * x^15 + 6 * x^8 * y^3 - 27 * x^7 := by sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3264_326405


namespace NUMINAMATH_CALUDE_terminating_decimal_expansion_13_200_l3264_326407

theorem terminating_decimal_expansion_13_200 : 
  ∃ (n : ℕ) (a : ℤ), (13 : ℚ) / 200 = (a : ℚ) / (10 ^ n) ∧ (a : ℚ) / (10 ^ n) = 0.052 :=
by
  sorry

end NUMINAMATH_CALUDE_terminating_decimal_expansion_13_200_l3264_326407


namespace NUMINAMATH_CALUDE_factor_count_8100_l3264_326496

def number_to_factor : ℕ := 8100

/-- The number of positive factors of a natural number n -/
def count_factors (n : ℕ) : ℕ := sorry

theorem factor_count_8100 : count_factors number_to_factor = 45 := by sorry

end NUMINAMATH_CALUDE_factor_count_8100_l3264_326496


namespace NUMINAMATH_CALUDE_walnuts_amount_l3264_326474

/-- The amount of walnuts in the trail mix -/
def walnuts : ℝ := sorry

/-- The total amount of nuts in the trail mix -/
def total_nuts : ℝ := 0.5

/-- The amount of almonds in the trail mix -/
def almonds : ℝ := 0.25

/-- Theorem stating that the amount of walnuts is 0.25 cups -/
theorem walnuts_amount : walnuts = 0.25 := by sorry

end NUMINAMATH_CALUDE_walnuts_amount_l3264_326474


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3264_326423

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, x > 0 → p x) ↔ (∃ x : ℝ, x > 0 ∧ ¬(p x)) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3264_326423


namespace NUMINAMATH_CALUDE_fraction_simplification_l3264_326415

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3264_326415


namespace NUMINAMATH_CALUDE_fifteenth_entry_is_22_l3264_326458

/-- r_7(n) represents the remainder when n is divided by 7 -/
def r_7 (n : ℕ) : ℕ := n % 7

/-- The list of nonnegative integers n that satisfy r_7(3n) ≤ 3 -/
def satisfying_list : List ℕ :=
  (List.range (100 : ℕ)).filter (fun n => r_7 (3 * n) ≤ 3)

theorem fifteenth_entry_is_22 : satisfying_list[14] = 22 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_entry_is_22_l3264_326458


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l3264_326459

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (7 + x) + Real.sqrt (25 - x) = 8) →
  ((7 + x) * (25 - x) = 256) := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l3264_326459


namespace NUMINAMATH_CALUDE_number_of_blue_balls_l3264_326412

/-- The number of blue balls originally in the box -/
def B : ℕ := sorry

/-- The number of red balls originally in the box -/
def R : ℕ := sorry

/-- Theorem stating the number of blue balls originally in the box -/
theorem number_of_blue_balls : 
  B = R + 17 ∧ 
  (B + 57) + (R + 18) - 44 = 502 → 
  B = 244 := by sorry

end NUMINAMATH_CALUDE_number_of_blue_balls_l3264_326412


namespace NUMINAMATH_CALUDE_unique_solution_l3264_326431

theorem unique_solution : ∃! n : ℕ, n > 0 ∧ Nat.lcm n 150 = Nat.gcd n 150 + 600 ∧ n = 675 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3264_326431


namespace NUMINAMATH_CALUDE_hiker_speed_l3264_326498

theorem hiker_speed (supplies_per_mile : Real) (first_pack : Real) (resupply_ratio : Real)
  (hours_per_day : Real) (num_days : Real) :
  supplies_per_mile = 0.5 →
  first_pack = 40 →
  resupply_ratio = 0.25 →
  hours_per_day = 8 →
  num_days = 5 →
  (first_pack + first_pack * resupply_ratio) / supplies_per_mile / (hours_per_day * num_days) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_hiker_speed_l3264_326498


namespace NUMINAMATH_CALUDE_garden_trees_l3264_326473

/-- The number of trees in a garden with specific planting conditions. -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  yard_length / tree_distance + 1

/-- Theorem stating that the number of trees in the garden is 26. -/
theorem garden_trees :
  number_of_trees 800 32 = 26 := by
  sorry

end NUMINAMATH_CALUDE_garden_trees_l3264_326473


namespace NUMINAMATH_CALUDE_nancy_coffee_consumption_l3264_326481

/-- Represents the daily coffee consumption and costs for Nancy --/
structure CoffeeConsumption where
  double_espresso_cost : ℝ
  iced_coffee_cost : ℝ
  total_spent : ℝ
  days : ℕ

/-- Calculates the number of coffees Nancy buys each day --/
def coffees_per_day (c : CoffeeConsumption) : ℕ :=
  2

/-- Theorem stating that Nancy buys 2 coffees per day given the conditions --/
theorem nancy_coffee_consumption (c : CoffeeConsumption) 
  (h1 : c.double_espresso_cost = 3)
  (h2 : c.iced_coffee_cost = 2.5)
  (h3 : c.total_spent = 110)
  (h4 : c.days = 20) :
  coffees_per_day c = 2 := by
  sorry

#check nancy_coffee_consumption

end NUMINAMATH_CALUDE_nancy_coffee_consumption_l3264_326481


namespace NUMINAMATH_CALUDE_blocks_in_prism_l3264_326414

/-- The number of unit blocks needed to fill a rectangular prism -/
def num_blocks (length width height : ℕ) : ℕ := length * width * height

/-- The dimensions of the rectangular prism -/
def prism_length : ℕ := 4
def prism_width : ℕ := 3
def prism_height : ℕ := 3

/-- Theorem: The number of 1 cm³ blocks needed to fill the given rectangular prism is 36 -/
theorem blocks_in_prism : 
  num_blocks prism_length prism_width prism_height = 36 := by
  sorry

end NUMINAMATH_CALUDE_blocks_in_prism_l3264_326414


namespace NUMINAMATH_CALUDE_syllogism_flaw_l3264_326462

theorem syllogism_flaw : ¬(∀ a : ℝ, a^2 > 0) := by sorry

end NUMINAMATH_CALUDE_syllogism_flaw_l3264_326462


namespace NUMINAMATH_CALUDE_sqrt_18_div_sqrt_8_l3264_326457

theorem sqrt_18_div_sqrt_8 : Real.sqrt 18 / Real.sqrt 8 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_div_sqrt_8_l3264_326457


namespace NUMINAMATH_CALUDE_probability_of_one_in_twenty_rows_l3264_326499

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) := sorry

/-- Counts the number of ones in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def countElements (n : ℕ) : ℕ := sorry

/-- The probability of randomly selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ :=
  (countOnes n : ℚ) / (countElements n : ℚ)

theorem probability_of_one_in_twenty_rows :
  probabilityOfOne 20 = 13 / 70 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_twenty_rows_l3264_326499


namespace NUMINAMATH_CALUDE_perpendicular_bisector_focus_condition_l3264_326442

/-- A point on a parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = 2 * x^2

/-- The perpendicular bisector of two points passes through the focus of the parabola -/
def perpendicular_bisector_passes_through_focus (A B : PointOnParabola) : Prop :=
  let midpoint := ((A.x + B.x) / 2, (A.y + B.y) / 2)
  let slope := if A.x = B.x then 0 else (B.y - A.y) / (B.x - A.x)
  let perp_slope := if slope = 0 then 0 else -1 / slope
  ∃ (t : ℝ), midpoint.1 + t * perp_slope = 0 ∧ midpoint.2 + t = 1/8

/-- Theorem: The perpendicular bisector passes through the focus iff x₁ + x₂ = 0 -/
theorem perpendicular_bisector_focus_condition (A B : PointOnParabola) :
  perpendicular_bisector_passes_through_focus A B ↔ A.x + B.x = 0 := by
  sorry

/-- The equation of the perpendicular bisector when x₁ = 1 and x₂ = -3 -/
def perpendicular_bisector_equation (A B : PointOnParabola) (h₁ : A.x = 1) (h₂ : B.x = -3) : 
  ∃ (a b c : ℝ), a * A.x + b * A.y + c = 0 ∧ a * B.x + b * B.y + c = 0 ∧ (a, b, c) = (1, -4, 41) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_focus_condition_l3264_326442


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3264_326453

theorem quadratic_inequality_solution_sets 
  (c b a : ℝ) 
  (h : Set.Ioo (-3 : ℝ) (1/2) = {x : ℝ | c * x^2 + b * x + a < 0}) : 
  {x : ℝ | a * x^2 + b * x + c ≥ 0} = Set.Icc (-1/3 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3264_326453


namespace NUMINAMATH_CALUDE_cube_lateral_surface_area_l3264_326402

theorem cube_lateral_surface_area (volume : ℝ) (lateral_surface_area : ℝ) :
  volume = 125 →
  lateral_surface_area = 4 * (volume ^ (1/3))^2 →
  lateral_surface_area = 100 := by
  sorry

end NUMINAMATH_CALUDE_cube_lateral_surface_area_l3264_326402


namespace NUMINAMATH_CALUDE_prime_triplet_l3264_326417

theorem prime_triplet (p : ℤ) : 
  Prime p ∧ Prime (p + 2) ∧ Prime (p + 4) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_triplet_l3264_326417


namespace NUMINAMATH_CALUDE_sum_of_w_and_y_is_three_l3264_326424

theorem sum_of_w_and_y_is_three :
  ∀ (W X Y Z : ℕ),
    W ∈ ({1, 2, 3, 4} : Set ℕ) →
    X ∈ ({1, 2, 3, 4} : Set ℕ) →
    Y ∈ ({1, 2, 3, 4} : Set ℕ) →
    Z ∈ ({1, 2, 3, 4} : Set ℕ) →
    W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
    (W : ℚ) / X + (Y : ℚ) / Z = 1 →
    W + Y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_w_and_y_is_three_l3264_326424


namespace NUMINAMATH_CALUDE_ellipse_properties_l3264_326439

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

-- Define a point on the ellipse
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

-- Define the foci
def foci (e : Ellipse) : ℝ × ℝ := sorry

-- Theorem statement
theorem ellipse_properties (e : Ellipse) (A : PointOnEllipse e)
  (h_A : A.x = 1 ∧ A.y = 1)
  (h_foci_dist : let (F1, F2) := foci e
                 Real.sqrt ((A.x - F1)^2 + (A.y - F1)^2) +
                 Real.sqrt ((A.x - F2)^2 + (A.y - F2)^2) = 4) :
  (e.a = 2 ∧ e.b^2 = 4/3) ∧
  (∀ x y, x + 3*y - 4 = 0 ↔ x^2/4 + 3*y^2/4 = 1) ∧
  (∀ C D : PointOnEllipse e,
    let k₁ := (C.y - A.y) / (C.x - A.x)
    let k₂ := (D.y - A.y) / (D.x - A.x)
    k₁ * k₂ = -1 →
    (D.y - C.y) / (D.x - C.x) = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3264_326439


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3264_326469

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem line_through_point_parallel_to_line 
  (M : Point) 
  (A B : Point) 
  (h_M : M.x = 2 ∧ M.y = -3)
  (h_A : A.x = 1 ∧ A.y = 2)
  (h_B : B.x = -1 ∧ B.y = -5) :
  ∃ (l : Line), 
    l.a = 7 ∧ l.b = -2 ∧ l.c = -20 ∧ 
    M.liesOn l ∧
    l.isParallelTo (Line.mk (B.y - A.y) (A.x - B.x) (B.x * A.y - A.x * B.y)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3264_326469


namespace NUMINAMATH_CALUDE_cow_selling_price_l3264_326491

/-- Calculates the selling price of a cow given the initial cost, daily food cost,
    vaccination and deworming cost, number of days, and profit made. -/
theorem cow_selling_price
  (initial_cost : ℕ)
  (daily_food_cost : ℕ)
  (vaccination_cost : ℕ)
  (num_days : ℕ)
  (profit : ℕ)
  (h1 : initial_cost = 600)
  (h2 : daily_food_cost = 20)
  (h3 : vaccination_cost = 500)
  (h4 : num_days = 40)
  (h5 : profit = 600) :
  initial_cost + num_days * daily_food_cost + vaccination_cost + profit = 2500 :=
by
  sorry

end NUMINAMATH_CALUDE_cow_selling_price_l3264_326491


namespace NUMINAMATH_CALUDE_colleen_pays_more_than_joy_l3264_326434

/-- Calculates the difference in cost between Colleen's and Joy's pencils -/
def pencil_cost_difference (joy_pencils colleen_pencils pencil_price : ℕ) : ℕ :=
  colleen_pencils * pencil_price - joy_pencils * pencil_price

theorem colleen_pays_more_than_joy :
  pencil_cost_difference 30 50 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_colleen_pays_more_than_joy_l3264_326434


namespace NUMINAMATH_CALUDE_binomial_sum_equals_31_l3264_326484

theorem binomial_sum_equals_31 (n : ℕ) : Nat.choose (2*n) (17-n) + Nat.choose (13+n) (3*n) = 31 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_31_l3264_326484


namespace NUMINAMATH_CALUDE_yadav_clothes_transport_expenditure_l3264_326438

/-- Represents Mr Yadav's monthly finances --/
structure YadavFinances where
  salary : ℝ
  consumable_percentage : ℝ
  clothes_transport_percentage : ℝ
  yearly_savings : ℝ

/-- Calculates the monthly amount spent on clothes and transport --/
def monthly_clothes_transport (y : YadavFinances) : ℝ :=
  y.salary * (1 - y.consumable_percentage) * y.clothes_transport_percentage

/-- Theorem stating the amount spent on clothes and transport --/
theorem yadav_clothes_transport_expenditure (y : YadavFinances) 
  (h1 : y.consumable_percentage = 0.6)
  (h2 : y.clothes_transport_percentage = 0.5)
  (h3 : y.yearly_savings = 19008)
  (h4 : y.yearly_savings = 12 * (y.salary * (1 - y.consumable_percentage) * (1 - y.clothes_transport_percentage))) :
  monthly_clothes_transport y = 1584 := by
  sorry

#eval monthly_clothes_transport { salary := 7920, consumable_percentage := 0.6, clothes_transport_percentage := 0.5, yearly_savings := 19008 }

end NUMINAMATH_CALUDE_yadav_clothes_transport_expenditure_l3264_326438


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3264_326437

theorem reciprocal_of_negative_two :
  ((-2 : ℝ)⁻¹ : ℝ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3264_326437


namespace NUMINAMATH_CALUDE_shopping_equation_system_l3264_326461

theorem shopping_equation_system (x y : ℤ) : 
  (∀ (coins_per_person excess : ℤ), coins_per_person * x - y = excess → 
    ((coins_per_person = 8 ∧ excess = 3) ∨ (coins_per_person = 7 ∧ excess = -4))) → 
  (8 * x - y = 3 ∧ y - 7 * x = 4) := by
sorry

end NUMINAMATH_CALUDE_shopping_equation_system_l3264_326461


namespace NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l3264_326404

theorem smallest_n_for_irreducible_fractions : ∃ (n : ℕ), 
  (n = 95) ∧ 
  (∀ (k : ℕ), 19 ≤ k ∧ k ≤ 91 → Nat.gcd k (n + k + 2) = 1) ∧
  (∀ (m : ℕ), m < n → ∃ (k : ℕ), 19 ≤ k ∧ k ≤ 91 ∧ Nat.gcd k (m + k + 2) ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l3264_326404


namespace NUMINAMATH_CALUDE_comics_bought_l3264_326444

theorem comics_bought (initial_amount remaining_amount cost_per_comic : ℕ) 
  (h1 : initial_amount = 87)
  (h2 : remaining_amount = 55)
  (h3 : cost_per_comic = 4) :
  (initial_amount - remaining_amount) / cost_per_comic = 8 := by
  sorry

end NUMINAMATH_CALUDE_comics_bought_l3264_326444


namespace NUMINAMATH_CALUDE_speed_limit_exceeders_l3264_326426

/-- The percentage of motorists who exceed the speed limit -/
def exceed_limit_percent : ℝ := 50

/-- The percentage of all motorists who receive speeding tickets -/
def receive_ticket_percent : ℝ := 40

/-- The percentage of speed limit exceeders who do not receive tickets -/
def no_ticket_percent : ℝ := 20

theorem speed_limit_exceeders :
  exceed_limit_percent = 50 :=
by
  sorry

#check speed_limit_exceeders

end NUMINAMATH_CALUDE_speed_limit_exceeders_l3264_326426


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3264_326487

theorem division_remainder_problem (x y u v : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + 3 * u * y + 4 * v) % y = 5 * v % y := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3264_326487


namespace NUMINAMATH_CALUDE_train_speed_problem_l3264_326467

/-- Represents a train with its speed and travel time after meeting another train -/
structure Train where
  speed : ℝ
  time_after_meeting : ℝ

/-- Proves that given the conditions of the problem, the speed of train B is 225 km/h -/
theorem train_speed_problem (train_A train_B : Train) 
  (h1 : train_A.speed = 100)
  (h2 : train_A.time_after_meeting = 9)
  (h3 : train_B.time_after_meeting = 4)
  (h4 : train_A.speed * train_A.time_after_meeting = train_B.speed * train_B.time_after_meeting) :
  train_B.speed = 225 := by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_train_speed_problem_l3264_326467


namespace NUMINAMATH_CALUDE_total_travel_ways_l3264_326490

-- Define the number of services for each mode of transportation
def bus_services : ℕ := 8
def train_services : ℕ := 3
def ferry_services : ℕ := 2

-- Theorem statement
theorem total_travel_ways : bus_services + train_services + ferry_services = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_ways_l3264_326490


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l3264_326460

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 < x ≤ 8}
theorem union_A_B : A ∪ B = {x | 1 < x ∧ x ≤ 8} := by sorry

-- Theorem 2: (∁ᵤA) ∩ B = {x | 1 < x < 2}
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x | 1 < x ∧ x < 2} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a < 8
theorem intersection_A_C_nonempty (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l3264_326460


namespace NUMINAMATH_CALUDE_function_equivalence_l3264_326485

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * (cos x)^2 - Real.sqrt 3 * sin (2 * x)

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x) + 1

theorem function_equivalence : ∀ x : ℝ, f x = g (x + 5 * π / 12) := by sorry

end NUMINAMATH_CALUDE_function_equivalence_l3264_326485
