import Mathlib

namespace NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l2404_240493

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the polynomial expansion function
def expandPolynomial (a b : ℝ) (n : ℕ) : (ℕ → ℝ) := sorry

-- Theorem statement
theorem coefficient_x_cubed_expansion :
  let expansion := expandPolynomial 1 (-1) 5
  let coefficient_x_cubed := (expansion 3) + (expansion 1)
  coefficient_x_cubed = -15 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l2404_240493


namespace NUMINAMATH_CALUDE_football_team_right_handed_count_l2404_240446

theorem football_team_right_handed_count 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (left_handed_fraction : ℚ) :
  total_players = 120 →
  throwers = 67 →
  left_handed_fraction = 2 / 5 →
  ∃ (right_handed : ℕ), 
    right_handed = throwers + (total_players - throwers - Int.floor ((total_players - throwers : ℚ) * left_handed_fraction)) ∧
    right_handed = 99 :=
by sorry

end NUMINAMATH_CALUDE_football_team_right_handed_count_l2404_240446


namespace NUMINAMATH_CALUDE_houses_built_l2404_240461

theorem houses_built (original : ℕ) (current : ℕ) (built : ℕ) : 
  original = 20817 → current = 118558 → built = current - original → built = 97741 := by
  sorry

end NUMINAMATH_CALUDE_houses_built_l2404_240461


namespace NUMINAMATH_CALUDE_sin_double_angle_special_case_l2404_240439

theorem sin_double_angle_special_case (θ : ℝ) (h : Real.tan θ + (Real.tan θ)⁻¹ = Real.sqrt 5) : 
  Real.sin (2 * θ) = (2 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_case_l2404_240439


namespace NUMINAMATH_CALUDE_quadratic_points_relation_l2404_240486

-- Define the quadratic function
def f (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Define the points A, B, and C
def A (m : ℝ) : ℝ × ℝ := (-4, f (-4) m)
def B (m : ℝ) : ℝ × ℝ := (0, f 0 m)
def C (m : ℝ) : ℝ × ℝ := (3, f 3 m)

-- State the theorem
theorem quadratic_points_relation (m : ℝ) :
  (B m).2 < (C m).2 ∧ (C m).2 < (A m).2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_relation_l2404_240486


namespace NUMINAMATH_CALUDE_investment_exceeds_4_million_in_2020_l2404_240495

/-- The year when the investment first exceeds 4 million CNY -/
def first_year_exceeding_4_million : ℕ :=
  2020

/-- The initial investment in 2010 in millions of CNY -/
def initial_investment : ℝ :=
  1.3

/-- The annual increase rate as a decimal -/
def annual_increase_rate : ℝ :=
  0.12

/-- The target investment in millions of CNY -/
def target_investment : ℝ :=
  4.0

theorem investment_exceeds_4_million_in_2020 :
  initial_investment * (1 + annual_increase_rate) ^ (first_year_exceeding_4_million - 2010) > target_investment ∧
  ∀ year : ℕ, year < first_year_exceeding_4_million →
    initial_investment * (1 + annual_increase_rate) ^ (year - 2010) ≤ target_investment :=
by sorry

end NUMINAMATH_CALUDE_investment_exceeds_4_million_in_2020_l2404_240495


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2404_240474

-- Define the conditions
def condition_p (m : ℝ) : Prop := ∃ x : ℝ, |x - 1| + |x - 3| < m

def condition_q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (7 - 3*m)^x > (7 - 3*m)^y

-- State the theorem
theorem necessary_but_not_sufficient :
  (∀ m : ℝ, condition_q m → condition_p m) ∧
  (∃ m : ℝ, condition_p m ∧ ¬condition_q m) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2404_240474


namespace NUMINAMATH_CALUDE_max_consecutive_interesting_l2404_240413

/-- A positive integer is interesting if it is a product of two prime numbers -/
def IsInteresting (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q

/-- The maximum number of consecutive interesting positive integers -/
theorem max_consecutive_interesting : 
  (∃ k : ℕ, k > 0 ∧ ∀ i : ℕ, i < k → IsInteresting (i + 1)) ∧ 
  (∀ k : ℕ, k > 3 → ∃ i : ℕ, i < k ∧ ¬IsInteresting (i + 1)) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_interesting_l2404_240413


namespace NUMINAMATH_CALUDE_andrey_gleb_distance_l2404_240453

/-- Represents the position of a home on a straight street -/
structure Home where
  position : ℝ

/-- The street with four homes -/
structure Street where
  andrey : Home
  borya : Home
  vova : Home
  gleb : Home

/-- The distance between two homes -/
def distance (h1 h2 : Home) : ℝ := |h1.position - h2.position|

/-- The conditions of the problem -/
def valid_street (s : Street) : Prop :=
  distance s.andrey s.borya = 600 ∧
  distance s.vova s.gleb = 600 ∧
  distance s.andrey s.gleb = 3 * distance s.borya s.vova

/-- The theorem to be proved -/
theorem andrey_gleb_distance (s : Street) :
  valid_street s →
  distance s.andrey s.gleb = 1500 ∨ distance s.andrey s.gleb = 1800 :=
sorry

end NUMINAMATH_CALUDE_andrey_gleb_distance_l2404_240453


namespace NUMINAMATH_CALUDE_quaternary_1010_equals_68_l2404_240479

/-- Converts a quaternary (base 4) digit to its decimal value --/
def quaternaryToDecimal (digit : Nat) : Nat :=
  if digit < 4 then digit else 0

/-- Calculates the decimal value of a quaternary number represented as a list of digits --/
def quaternaryListToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + quaternaryToDecimal d * (4 ^ (digits.length - 1 - i))) 0

/-- The quaternary representation of the number to be converted --/
def quaternaryNumber : List Nat := [1, 0, 1, 0]

/-- Statement: The quaternary number 1010₍₄₎ is equal to the decimal number 68 --/
theorem quaternary_1010_equals_68 : 
  quaternaryListToDecimal quaternaryNumber = 68 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_1010_equals_68_l2404_240479


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_l2404_240452

/-- Represents a club with members of different characteristics -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_non_jazz : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club)
  (h1 : c.total_members = 30)
  (h2 : c.left_handed = 11)
  (h3 : c.jazz_lovers = 20)
  (h4 : c.right_handed_non_jazz = 4)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members) :
  c.left_handed + c.jazz_lovers - c.total_members + c.right_handed_non_jazz = 5 := by
  sorry

#check left_handed_jazz_lovers

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_l2404_240452


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2404_240484

theorem inequality_solution_set (x : ℝ) :
  x ≠ 1 →
  (((x^2 - x - 6) / (x - 1) > 0) ↔ ((-2 < x ∧ x < 1) ∨ x > 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2404_240484


namespace NUMINAMATH_CALUDE_pentagon_coverage_is_62_5_percent_l2404_240475

/-- Represents a tiling of the plane with large squares and pentagons -/
structure PlaneTiling where
  /-- The number of smaller squares in each row/column of a large square -/
  grid_size : ℕ
  /-- The number of smaller squares that are part of pentagons in each large square -/
  pentagon_squares : ℕ

/-- The percentage of the plane enclosed by pentagons -/
def pentagon_percentage (t : PlaneTiling) : ℚ :=
  t.pentagon_squares / (t.grid_size ^ 2 : ℚ) * 100

/-- Theorem stating that the percentage of the plane enclosed by pentagons is 62.5% -/
theorem pentagon_coverage_is_62_5_percent (t : PlaneTiling) 
  (h1 : t.grid_size = 4)
  (h2 : t.pentagon_squares = 10) : 
  pentagon_percentage t = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_coverage_is_62_5_percent_l2404_240475


namespace NUMINAMATH_CALUDE_A_intersect_B_is_empty_l2404_240407

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}
def B : Set ℝ := {x | x - 1 > 0}

-- Statement to prove
theorem A_intersect_B_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_empty_l2404_240407


namespace NUMINAMATH_CALUDE_prob_ace_heart_queen_l2404_240444

-- Define the structure of a standard deck
def StandardDeck : Type := Unit

-- Define card types
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

inductive Suit
| Hearts | Diamonds | Clubs | Spades

structure Card where
  rank : Rank
  suit : Suit

-- Define the probability of drawing specific cards
def prob_first_ace (deck : StandardDeck) : ℚ := 4 / 52

def prob_second_heart (deck : StandardDeck) : ℚ := 13 / 51

def prob_third_queen (deck : StandardDeck) : ℚ := 4 / 50

-- State the theorem
theorem prob_ace_heart_queen (deck : StandardDeck) :
  prob_first_ace deck * prob_second_heart deck * prob_third_queen deck = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_heart_queen_l2404_240444


namespace NUMINAMATH_CALUDE_evaluate_expression_l2404_240420

theorem evaluate_expression : (24^18) / (72^9) = 8^9 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2404_240420


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l2404_240436

theorem no_solution_fractional_equation :
  ¬ ∃ (x : ℝ), x ≠ 5 ∧ (3 * x / (x - 5) + 15 / (5 - x) = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l2404_240436


namespace NUMINAMATH_CALUDE_sqrt_eight_times_half_minus_sqrt_three_power_zero_l2404_240431

theorem sqrt_eight_times_half_minus_sqrt_three_power_zero :
  Real.sqrt 8 * (1 / 2) - (Real.sqrt 3) ^ 0 = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_times_half_minus_sqrt_three_power_zero_l2404_240431


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2404_240442

theorem fraction_equals_zero (x : ℝ) :
  (x^2 - 1) / (1 - x) = 0 ∧ 1 - x ≠ 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2404_240442


namespace NUMINAMATH_CALUDE_perimeter_of_square_region_l2404_240425

theorem perimeter_of_square_region (total_area : ℝ) (num_squares : ℕ) (perimeter : ℝ) :
  total_area = 588 →
  num_squares = 14 →
  perimeter = 15 * Real.sqrt 42 :=
by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_square_region_l2404_240425


namespace NUMINAMATH_CALUDE_die_probability_l2404_240424

/-- The number of times the die is tossed -/
def n : ℕ := 30

/-- The number of faces on the die -/
def faces : ℕ := 6

/-- The number of favorable outcomes before the first six -/
def favorable_before : ℕ := 3

/-- Probability of the event: at least one six appears, and no five or four appears before the first six -/
def prob_event : ℚ :=
  1 / 3

theorem die_probability :
  prob_event = (favorable_before ^ (n - 1) * (2 ^ n - 1)) / (faces ^ n) :=
sorry

end NUMINAMATH_CALUDE_die_probability_l2404_240424


namespace NUMINAMATH_CALUDE_geometric_sum_5_quarters_l2404_240478

def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

theorem geometric_sum_5_quarters : 
  geometric_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_5_quarters_l2404_240478


namespace NUMINAMATH_CALUDE_new_year_party_arrangements_l2404_240462

/-- The number of ways to distribute n teachers between two classes,
    with each class having no more than m teachers. -/
def teacherDistributions (n m : ℕ) : ℕ :=
  sorry

theorem new_year_party_arrangements :
  teacherDistributions 6 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_new_year_party_arrangements_l2404_240462


namespace NUMINAMATH_CALUDE_donghwan_candies_l2404_240410

theorem donghwan_candies (total_candies bag_size : ℕ) 
  (h1 : total_candies = 138)
  (h2 : bag_size = 18) :
  total_candies % bag_size = 12 := by
  sorry

end NUMINAMATH_CALUDE_donghwan_candies_l2404_240410


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2404_240455

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x^2 = a - 2) ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2404_240455


namespace NUMINAMATH_CALUDE_computation_proof_l2404_240443

theorem computation_proof : 45 * (28 + 72) + 55 * 45 = 6975 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l2404_240443


namespace NUMINAMATH_CALUDE_die_roll_outcomes_l2404_240402

/-- The number of faces on a standard die -/
def numDieFaces : ℕ := 6

/-- The number of rolls before stopping -/
def numRolls : ℕ := 5

/-- The number of different outcomes when rolling a die continuously and stopping
    after exactly 5 rolls, with the condition that three different numbers appear
    on the fifth roll -/
def numOutcomes : ℕ := 840

/-- Theorem stating that the number of different outcomes is 840 -/
theorem die_roll_outcomes :
  (numDieFaces.choose 2) * ((numDieFaces - 2).choose 1) * (4 + 6 + 4) = numOutcomes := by
  sorry

end NUMINAMATH_CALUDE_die_roll_outcomes_l2404_240402


namespace NUMINAMATH_CALUDE_complex_function_evaluation_l2404_240492

theorem complex_function_evaluation : 
  let z : ℂ := (Complex.I + 1) / (Complex.I - 1)
  let f : ℂ → ℂ := fun x ↦ x^2 - x + 1
  f z = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_function_evaluation_l2404_240492


namespace NUMINAMATH_CALUDE_jones_pants_count_l2404_240471

/-- Represents the number of pants Mr. Jones has -/
def num_pants : ℕ := 40

/-- Represents the number of shirts Mr. Jones has for each pair of pants -/
def shirts_per_pants : ℕ := 6

/-- Represents the total number of pieces of clothes Mr. Jones owns -/
def total_clothes : ℕ := 280

/-- Theorem stating that the number of pants Mr. Jones has is 40 -/
theorem jones_pants_count :
  num_pants * (shirts_per_pants + 1) = total_clothes :=
by sorry

end NUMINAMATH_CALUDE_jones_pants_count_l2404_240471


namespace NUMINAMATH_CALUDE_fraction_simplification_l2404_240403

theorem fraction_simplification :
  (1 / 20 : ℚ) - (1 / 21 : ℚ) + (1 / (20 * 21) : ℚ) = (1 / 210 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2404_240403


namespace NUMINAMATH_CALUDE_coinciding_rest_days_count_l2404_240441

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 6

/-- Carol's schedule cycle length -/
def carol_cycle : ℕ := 6

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Al's rest days in a cycle -/
def al_rest_days : Finset ℕ := {5, 6}

/-- Carol's rest days in a cycle -/
def carol_rest_days : Finset ℕ := {6}

/-- The number of days both Al and Carol have rest-days on the same day -/
def coinciding_rest_days : ℕ := (al_rest_days ∩ carol_rest_days).card * (total_days / al_cycle)

theorem coinciding_rest_days_count : coinciding_rest_days = 166 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_count_l2404_240441


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l2404_240406

theorem pure_imaginary_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : ∃ (t : ℝ), (3 - 4 * Complex.I) * (x + y * Complex.I) = t * Complex.I) : 
  x / y = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l2404_240406


namespace NUMINAMATH_CALUDE_circle_area_reduction_l2404_240498

theorem circle_area_reduction (r : ℝ) (h : r > 0) :
  let new_r := 0.9 * r
  let original_area := π * r^2
  let new_area := π * new_r^2
  new_area = 0.81 * original_area := by
  sorry

end NUMINAMATH_CALUDE_circle_area_reduction_l2404_240498


namespace NUMINAMATH_CALUDE_squared_gt_implies_abs_gt_but_not_conversely_l2404_240433

theorem squared_gt_implies_abs_gt_but_not_conversely :
  (∀ a b : ℝ, a^2 > b^2 → |a| > b) ∧
  (∃ a b : ℝ, |a| > b ∧ a^2 ≤ b^2) :=
by sorry

end NUMINAMATH_CALUDE_squared_gt_implies_abs_gt_but_not_conversely_l2404_240433


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2404_240450

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2404_240450


namespace NUMINAMATH_CALUDE_grid_walk_probability_l2404_240451

def grid_walk (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * (grid_walk (x-1) y + grid_walk x (y-1) + grid_walk (x-1) (y-1))

theorem grid_walk_probability :
  ∃ (m n : ℕ), 
    m > 0 ∧ 
    n > 0 ∧ 
    ¬(3 ∣ m) ∧ 
    grid_walk 5 5 = m / (3^n : ℚ) ∧ 
    m + n = 1186 :=
sorry

end NUMINAMATH_CALUDE_grid_walk_probability_l2404_240451


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2404_240447

theorem remainder_divisibility (x y z p : ℕ) : 
  0 < x → 0 < y → 0 < z →  -- x, y, z are positive integers
  Nat.Prime p →            -- p is prime
  x < y → y < z → z < p →  -- x < y < z < p
  x^3 % p = y^3 % p →      -- x^3 and y^3 have the same remainder mod p
  y^3 % p = z^3 % p →      -- y^3 and z^3 have the same remainder mod p
  (x + y + z) ∣ (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2404_240447


namespace NUMINAMATH_CALUDE_equation_solution_l2404_240490

theorem equation_solution :
  ∃! x : ℚ, (4 * x^2 + 3 * x + 1) / (x - 2) = 4 * x + 5 :=
by
  use -11/6
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2404_240490


namespace NUMINAMATH_CALUDE_simplified_expression_and_evaluation_l2404_240405

theorem simplified_expression_and_evaluation (x : ℝ) 
  (h1 : x ≠ -3) (h2 : x ≠ 3) :
  (3 / (x - 3) - 3 * x / (x^2 - 9)) / ((3 * x - 9) / (x^2 - 6 * x + 9)) = 3 / (x + 3) ∧
  (3 / (1 + 3) = 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_and_evaluation_l2404_240405


namespace NUMINAMATH_CALUDE_floral_shop_sales_theorem_l2404_240467

/-- Represents the sales and prices of bouquets for a floral shop over three days -/
structure FloralShopSales where
  /-- Number of rose bouquets sold on Monday -/
  rose_monday : ℕ
  /-- Number of lily bouquets sold on Monday -/
  lily_monday : ℕ
  /-- Number of orchid bouquets sold on Monday -/
  orchid_monday : ℕ
  /-- Price of rose bouquets on Monday -/
  rose_price_monday : ℕ
  /-- Price of lily bouquets on Monday -/
  lily_price_monday : ℕ
  /-- Price of orchid bouquets on Monday -/
  orchid_price_monday : ℕ
  /-- Price of rose bouquets on Tuesday -/
  rose_price_tuesday : ℕ
  /-- Price of lily bouquets on Tuesday -/
  lily_price_tuesday : ℕ
  /-- Price of orchid bouquets on Tuesday -/
  orchid_price_tuesday : ℕ
  /-- Price of rose bouquets on Wednesday -/
  rose_price_wednesday : ℕ
  /-- Price of lily bouquets on Wednesday -/
  lily_price_wednesday : ℕ
  /-- Price of orchid bouquets on Wednesday -/
  orchid_price_wednesday : ℕ

/-- Calculates the total number and value of bouquets sold over three days -/
def calculate_total_sales (sales : FloralShopSales) : ℕ × ℕ :=
  let rose_tuesday := 3 * sales.rose_monday
  let lily_tuesday := 2 * sales.lily_monday
  let orchid_tuesday := sales.orchid_monday / 2
  let rose_wednesday := rose_tuesday / 3
  let lily_wednesday := lily_tuesday / 4
  let orchid_wednesday := (2 * orchid_tuesday) / 3
  
  let total_roses := sales.rose_monday + rose_tuesday + rose_wednesday
  let total_lilies := sales.lily_monday + lily_tuesday + lily_wednesday
  let total_orchids := sales.orchid_monday + orchid_tuesday + orchid_wednesday
  
  let total_bouquets := total_roses + total_lilies + total_orchids
  
  let rose_value := sales.rose_monday * sales.rose_price_monday + 
                    rose_tuesday * sales.rose_price_tuesday + 
                    rose_wednesday * sales.rose_price_wednesday
  let lily_value := sales.lily_monday * sales.lily_price_monday + 
                    lily_tuesday * sales.lily_price_tuesday + 
                    lily_wednesday * sales.lily_price_wednesday
  let orchid_value := sales.orchid_monday * sales.orchid_price_monday + 
                      orchid_tuesday * sales.orchid_price_tuesday + 
                      orchid_wednesday * sales.orchid_price_wednesday
  
  let total_value := rose_value + lily_value + orchid_value
  
  (total_bouquets, total_value)

theorem floral_shop_sales_theorem (sales : FloralShopSales) 
  (h1 : sales.rose_monday = 12)
  (h2 : sales.lily_monday = 8)
  (h3 : sales.orchid_monday = 6)
  (h4 : sales.rose_price_monday = 10)
  (h5 : sales.lily_price_monday = 15)
  (h6 : sales.orchid_price_monday = 20)
  (h7 : sales.rose_price_tuesday = 12)
  (h8 : sales.lily_price_tuesday = 18)
  (h9 : sales.orchid_price_tuesday = 22)
  (h10 : sales.rose_price_wednesday = 8)
  (h11 : sales.lily_price_wednesday = 12)
  (h12 : sales.orchid_price_wednesday = 16) :
  calculate_total_sales sales = (99, 1322) := by
  sorry


end NUMINAMATH_CALUDE_floral_shop_sales_theorem_l2404_240467


namespace NUMINAMATH_CALUDE_infinite_set_sum_of_digits_squared_equal_l2404_240416

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proposition: There exists an infinite set of natural numbers n, not ending in 0,
    such that the sum of digits of n^2 equals the sum of digits of n -/
theorem infinite_set_sum_of_digits_squared_equal :
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    (∀ n ∈ S, n % 10 ≠ 0 ∧ sum_of_digits (n^2) = sum_of_digits n) :=
sorry

end NUMINAMATH_CALUDE_infinite_set_sum_of_digits_squared_equal_l2404_240416


namespace NUMINAMATH_CALUDE_one_and_one_third_l2404_240411

theorem one_and_one_third : ∃ x : ℚ, (4 / 3) * x = 45 ∧ x = 135 / 4 := by
  sorry

end NUMINAMATH_CALUDE_one_and_one_third_l2404_240411


namespace NUMINAMATH_CALUDE_first_square_perimeter_l2404_240438

/-- Given two squares and a third square with specific properties, 
    prove that the perimeter of the first square is 24 meters. -/
theorem first_square_perimeter : 
  ∀ (s₁ s₂ s₃ : ℝ),
  (4 * s₂ = 32) →  -- Perimeter of second square is 32 m
  (4 * s₃ = 40) →  -- Perimeter of third square is 40 m
  (s₃^2 = s₁^2 + s₂^2) →  -- Area of third square equals sum of areas of first two squares
  (4 * s₁ = 24) :=  -- Perimeter of first square is 24 m
by
  sorry

#check first_square_perimeter

end NUMINAMATH_CALUDE_first_square_perimeter_l2404_240438


namespace NUMINAMATH_CALUDE_equation_has_real_root_l2404_240466

theorem equation_has_real_root (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 2) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l2404_240466


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2404_240434

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I + a) * (1 + Complex.I) = b * Complex.I →
  Complex.I * b + a = 1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2404_240434


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2404_240489

/-- Given that the solution set of ax² - bx - 1 ≥ 0 is [1/3, 1/2],
    prove that the solution set of x² - bx - a < 0 is (-3, -2) -/
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ 1/3 ≤ x ∧ x ≤ 1/2) →
  (∀ x, x^2 - b*x - a < 0 ↔ -3 < x ∧ x < -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2404_240489


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l2404_240456

/-- Calculates the total number of heartbeats during a race --/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  let race_duration := pace * race_distance
  race_duration * heart_rate

/-- Theorem: The athlete's heart beats 24300 times during the 30-mile race --/
theorem athlete_heartbeats :
  let heart_rate : ℕ := 135  -- beats per minute
  let pace : ℕ := 6          -- minutes per mile
  let race_distance : ℕ := 30 -- miles
  total_heartbeats heart_rate pace race_distance = 24300 :=
by
  sorry

end NUMINAMATH_CALUDE_athlete_heartbeats_l2404_240456


namespace NUMINAMATH_CALUDE_probability_sum_5_l2404_240488

def S : Finset ℕ := {1, 2, 3, 4, 5}

def pairs : Finset (ℕ × ℕ) := S.product S

def valid_pairs : Finset (ℕ × ℕ) := pairs.filter (fun p => p.1 < p.2)

def sum_5_pairs : Finset (ℕ × ℕ) := valid_pairs.filter (fun p => p.1 + p.2 = 5)

theorem probability_sum_5 : 
  (sum_5_pairs.card : ℚ) / valid_pairs.card = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_sum_5_l2404_240488


namespace NUMINAMATH_CALUDE_ratio_change_problem_l2404_240468

theorem ratio_change_problem (x y : ℝ) (h1 : y / x = 3 / 2) (h2 : y - x = 8) :
  ∃ z : ℝ, (y + z) / (x + z) = 7 / 5 ∧ z = 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_problem_l2404_240468


namespace NUMINAMATH_CALUDE_school_attendance_l2404_240404

/-- Represents the attendance schedule for a group of students -/
inductive AttendanceSchedule
  | A -- Attends Mondays and Wednesdays
  | B -- Attends Tuesdays and Thursdays
  | C -- Attends Fridays

/-- Represents a day of the week -/
inductive WeekDay
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- The school attendance problem -/
theorem school_attendance
  (total_students : Nat)
  (home_learning_percentage : Rat)
  (group_schedules : List AttendanceSchedule)
  (h1 : total_students = 1000)
  (h2 : home_learning_percentage = 60 / 100)
  (h3 : group_schedules = [AttendanceSchedule.A, AttendanceSchedule.B, AttendanceSchedule.C]) :
  ∃ (attendance : WeekDay → Nat),
    attendance WeekDay.Monday = 133 ∧
    attendance WeekDay.Tuesday = 133 ∧
    attendance WeekDay.Wednesday = 133 ∧
    attendance WeekDay.Thursday = 133 ∧
    attendance WeekDay.Friday = 134 := by
  sorry

end NUMINAMATH_CALUDE_school_attendance_l2404_240404


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l2404_240482

theorem circus_ticket_cost (ticket_price : ℝ) (num_tickets : ℕ) (total_cost : ℝ) : 
  ticket_price = 44 ∧ num_tickets = 7 ∧ total_cost = ticket_price * num_tickets → total_cost = 308 :=
by sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l2404_240482


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l2404_240465

theorem power_mod_thirteen : 6^2040 ≡ 1 [ZMOD 13] := by sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l2404_240465


namespace NUMINAMATH_CALUDE_exist_good_numbers_not_preserving_sum_of_digits_l2404_240419

/-- A natural number is "good" if its decimal representation contains only zeros and ones. -/
def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The sum of digits of a natural number in base 10. -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem stating that there exist good numbers whose product is good,
    but the sum of digits property doesn't hold. -/
theorem exist_good_numbers_not_preserving_sum_of_digits :
  ∃ (A B : ℕ), is_good A ∧ is_good B ∧ is_good (A * B) ∧
    sum_of_digits (A * B) ≠ sum_of_digits A * sum_of_digits B := by
  sorry

end NUMINAMATH_CALUDE_exist_good_numbers_not_preserving_sum_of_digits_l2404_240419


namespace NUMINAMATH_CALUDE_total_apples_eaten_l2404_240472

def simone_daily_consumption : ℚ := 1/2
def simone_days : ℕ := 16
def lauri_daily_consumption : ℚ := 1/3
def lauri_days : ℕ := 15

theorem total_apples_eaten :
  (simone_daily_consumption * simone_days + lauri_daily_consumption * lauri_days : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_eaten_l2404_240472


namespace NUMINAMATH_CALUDE_bivalent_metal_relative_atomic_mass_l2404_240429

-- Define the bivalent metal
structure BivalentMetal where
  relative_atomic_mass : ℝ

-- Define the reaction conditions
def hcl_moles : ℝ := 0.25

-- Define the reaction properties
def incomplete_reaction (m : BivalentMetal) : Prop :=
  3.5 / m.relative_atomic_mass > hcl_moles / 2

def complete_reaction (m : BivalentMetal) : Prop :=
  2.5 / m.relative_atomic_mass < hcl_moles / 2

-- Theorem to prove
theorem bivalent_metal_relative_atomic_mass :
  ∃ (m : BivalentMetal), 
    m.relative_atomic_mass = 24 ∧ 
    incomplete_reaction m ∧ 
    complete_reaction m :=
by
  sorry

end NUMINAMATH_CALUDE_bivalent_metal_relative_atomic_mass_l2404_240429


namespace NUMINAMATH_CALUDE_min_value_of_a_l2404_240412

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

-- Define the theorem
theorem min_value_of_a (h_a : ℝ) (h_a_pos : h_a > 0) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-1 : ℝ) 2, f x₁ = g h_a x₂) → 
  h_a ≥ 3 := by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_min_value_of_a_l2404_240412


namespace NUMINAMATH_CALUDE_total_students_l2404_240423

def line_up (students_between : ℕ) (right_of_hoseok : ℕ) (left_of_yoongi : ℕ) : ℕ :=
  2 + students_between + right_of_hoseok + left_of_yoongi

theorem total_students :
  line_up 5 9 6 = 22 :=
by sorry

end NUMINAMATH_CALUDE_total_students_l2404_240423


namespace NUMINAMATH_CALUDE_meaningful_fraction_l2404_240463

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / x) ↔ x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l2404_240463


namespace NUMINAMATH_CALUDE_batsman_average_l2404_240400

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman after their last innings -/
def calculateAverage (b : Batsman) : Nat :=
  (b.totalRuns + b.lastInningsScore) / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after the 12th innings is 82 runs -/
theorem batsman_average (b : Batsman)
  (h1 : b.innings = 12)
  (h2 : b.lastInningsScore = 115)
  (h3 : b.averageIncrease = 3)
  (h4 : calculateAverage b = calculateAverage { b with innings := b.innings - 1 } + b.averageIncrease) :
  calculateAverage b = 82 := by
  sorry

#check batsman_average

end NUMINAMATH_CALUDE_batsman_average_l2404_240400


namespace NUMINAMATH_CALUDE_even_monotonic_function_property_l2404_240494

def is_even_on (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, -a ≤ x ∧ x ≤ a → f x = f (-x)

def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → (f x ≤ f y ∨ f y ≤ f x)

theorem even_monotonic_function_property (f : ℝ → ℝ) 
  (h1 : is_even_on f 5)
  (h2 : is_monotonic_on f 0 5)
  (h3 : f (-3) < f 1) :
  f 1 < f 0 := by
  sorry

end NUMINAMATH_CALUDE_even_monotonic_function_property_l2404_240494


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l2404_240414

theorem cube_sum_divisibility (a b c : ℤ) : 
  (∃ k : ℤ, a + b + c = 6 * k) → (∃ m : ℤ, a^3 + b^3 + c^3 = 6 * m) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l2404_240414


namespace NUMINAMATH_CALUDE_le_zero_iff_lt_or_eq_l2404_240437

theorem le_zero_iff_lt_or_eq (x : ℝ) : x ≤ 0 ↔ x < 0 ∨ x = 0 := by sorry

end NUMINAMATH_CALUDE_le_zero_iff_lt_or_eq_l2404_240437


namespace NUMINAMATH_CALUDE_ring_area_between_circles_l2404_240440

theorem ring_area_between_circles (π : ℝ) (h : π > 0) :
  let r₁ : ℝ := 12
  let r₂ : ℝ := 7
  let area_larger := π * r₁^2
  let area_smaller := π * r₂^2
  area_larger - area_smaller = 95 * π :=
by sorry

end NUMINAMATH_CALUDE_ring_area_between_circles_l2404_240440


namespace NUMINAMATH_CALUDE_octagon_area_l2404_240448

theorem octagon_area (circle_area : ℝ) (h : circle_area = 256 * Real.pi) :
  ∃ (octagon_area : ℝ), octagon_area = 512 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l2404_240448


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2404_240418

theorem sin_cos_identity (α : ℝ) : (Real.sin α - Real.cos α)^2 + Real.sin (2 * α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2404_240418


namespace NUMINAMATH_CALUDE_boisjoli_farm_egg_production_l2404_240458

/-- The number of eggs each hen lays per day at Boisjoli farm -/
theorem boisjoli_farm_egg_production 
  (num_hens : ℕ) 
  (num_days : ℕ) 
  (num_boxes : ℕ) 
  (eggs_per_box : ℕ) 
  (h_hens : num_hens = 270) 
  (h_days : num_days = 7) 
  (h_boxes : num_boxes = 315) 
  (h_eggs_per_box : eggs_per_box = 6) : 
  (num_boxes * eggs_per_box) / (num_hens * num_days) = 1 := by
  sorry

#check boisjoli_farm_egg_production

end NUMINAMATH_CALUDE_boisjoli_farm_egg_production_l2404_240458


namespace NUMINAMATH_CALUDE_bobs_password_probability_l2404_240483

theorem bobs_password_probability :
  let single_digit_numbers : ℕ := 10
  let even_single_digit_numbers : ℕ := 5
  let alphabet_letters : ℕ := 26
  let vowels : ℕ := 5
  let non_zero_single_digit_numbers : ℕ := 9
  let prob_even : ℚ := even_single_digit_numbers / single_digit_numbers
  let prob_vowel : ℚ := vowels / alphabet_letters
  let prob_non_zero : ℚ := non_zero_single_digit_numbers / single_digit_numbers
  prob_even * prob_vowel * prob_non_zero = 9 / 52 := by
sorry

end NUMINAMATH_CALUDE_bobs_password_probability_l2404_240483


namespace NUMINAMATH_CALUDE_rs_length_l2404_240422

structure Tetrahedron where
  edges : Finset ℝ
  pq : ℝ

def valid_tetrahedron (t : Tetrahedron) : Prop :=
  t.edges.card = 6 ∧ 
  t.pq ∈ t.edges ∧
  ∀ e ∈ t.edges, e > 0

theorem rs_length (t : Tetrahedron) 
  (h_valid : valid_tetrahedron t)
  (h_edges : t.edges = {9, 16, 22, 31, 39, 48})
  (h_pq : t.pq = 48) :
  ∃ rs ∈ t.edges, rs = 9 :=
sorry

end NUMINAMATH_CALUDE_rs_length_l2404_240422


namespace NUMINAMATH_CALUDE_least_positive_angle_phi_l2404_240499

theorem least_positive_angle_phi : 
  ∃ φ : ℝ, φ > 0 ∧ φ ≤ π/2 ∧ 
  (∀ ψ : ℝ, ψ > 0 → ψ < φ → Real.cos (10 * π/180) ≠ Real.sin (15 * π/180) + Real.sin ψ) ∧
  Real.cos (10 * π/180) = Real.sin (15 * π/180) + Real.sin φ ∧
  φ = 42.5 * π/180 :=
sorry

end NUMINAMATH_CALUDE_least_positive_angle_phi_l2404_240499


namespace NUMINAMATH_CALUDE_pentagon_angles_l2404_240469

-- Define the angles of the pentagons
variable (A B C D E F G H : ℝ)

-- Define the properties of the pentagons
axiom pentagon_sum : A + B + C + D + E = 540
axiom pentagon_sum2 : B + C + F + G + H = 540

-- Define the conditions given in the problem
axiom congruent_ABC : A = B ∧ B = C
axiom congruent_BCF : B = C ∧ C = F
axiom A_less_D : A = D - 50
axiom F_greater_G : F = G + 10

-- Define the theorem to be proved
theorem pentagon_angles : D = 138 ∧ G = 118 :=
sorry

end NUMINAMATH_CALUDE_pentagon_angles_l2404_240469


namespace NUMINAMATH_CALUDE_bridget_profit_is_fifty_l2404_240426

/-- Calculates Bridget's profit from baking and selling bread --/
def bridget_profit (total_loaves : ℕ) (cost_per_loaf : ℚ) 
  (morning_price : ℚ) (late_afternoon_price : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3
  let morning_revenue := morning_sales * morning_price
  let afternoon_remaining := total_loaves - morning_sales
  let afternoon_sales := afternoon_remaining / 2
  let afternoon_revenue := afternoon_sales * (morning_price / 2)
  let late_afternoon_remaining := afternoon_remaining - afternoon_sales
  let late_afternoon_sales := (late_afternoon_remaining * 2) / 3
  let late_afternoon_revenue := late_afternoon_sales * late_afternoon_price
  let evening_sales := late_afternoon_remaining - late_afternoon_sales
  let evening_revenue := evening_sales * cost_per_loaf
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue + evening_revenue
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

/-- Bridget's profit is $50 --/
theorem bridget_profit_is_fifty :
  bridget_profit 60 1 3 1 = 50 :=
by sorry

end NUMINAMATH_CALUDE_bridget_profit_is_fifty_l2404_240426


namespace NUMINAMATH_CALUDE_count_valid_markings_l2404_240480

/-- Represents a valid marking of an 8x8 chessboard -/
def ValidMarking : Type := 
  { marking : Fin 8 → Fin 8 // 
    (∀ i j, i ≠ j → marking i ≠ marking j) ∧ 
    (∀ i, marking i ≠ 0 ∧ marking i ≠ 7) ∧
    (marking 0 ≠ 0 ∧ marking 0 ≠ 7) ∧
    (marking 7 ≠ 0 ∧ marking 7 ≠ 7) }

/-- The number of valid markings on an 8x8 chessboard -/
def numValidMarkings : ℕ := sorry

/-- The theorem stating the number of valid markings -/
theorem count_valid_markings : numValidMarkings = 21600 := by sorry

end NUMINAMATH_CALUDE_count_valid_markings_l2404_240480


namespace NUMINAMATH_CALUDE_circumcircle_radius_of_specific_triangle_l2404_240481

/-- The radius of the circumcircle of a triangle with side lengths 8, 15, and 17 is 8.5. -/
theorem circumcircle_radius_of_specific_triangle : 
  ∀ (a b c : ℝ) (r : ℝ),
    a = 8 → b = 15 → c = 17 →
    (a^2 + b^2 = c^2) →  -- right triangle condition
    r = c / 2 →          -- radius is half the hypotenuse
    r = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_radius_of_specific_triangle_l2404_240481


namespace NUMINAMATH_CALUDE_beatrice_auction_tvs_l2404_240487

/-- The number of TVs Beatrice looked at on the auction site -/
def auction_tvs (in_person : ℕ) (online_multiplier : ℕ) (total : ℕ) : ℕ :=
  total - (in_person + online_multiplier * in_person)

/-- Proof that Beatrice looked at 10 TVs on the auction site -/
theorem beatrice_auction_tvs :
  auction_tvs 8 3 42 = 10 := by
  sorry

end NUMINAMATH_CALUDE_beatrice_auction_tvs_l2404_240487


namespace NUMINAMATH_CALUDE_fraction_problem_l2404_240432

theorem fraction_problem (x : ℚ) : x = 4/5 ↔ 0.55 * 40 = x * 25 + 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2404_240432


namespace NUMINAMATH_CALUDE_closest_to_M_div_N_l2404_240497

-- Define the state space complexity of Go
def M : ℝ := 3^361

-- Define the number of atoms in the observable universe
def N : ℝ := 10^80

-- Define the options
def options : List ℝ := [10^33, 10^53, 10^73, 10^93]

-- Theorem statement
theorem closest_to_M_div_N :
  let ratio := M / N
  (∀ x ∈ options, |ratio - 10^93| ≤ |ratio - x|) ∧ (10^93 ∈ options) :=
by sorry

end NUMINAMATH_CALUDE_closest_to_M_div_N_l2404_240497


namespace NUMINAMATH_CALUDE_surprise_shop_revenue_l2404_240445

/-- Represents the daily potential revenue of a shop during Christmas holidays -/
def daily_potential_revenue (closed_days_per_year : ℕ) (total_years : ℕ) (total_revenue_loss : ℕ) : ℚ :=
  total_revenue_loss / (closed_days_per_year * total_years)

/-- Theorem stating that the daily potential revenue for the given conditions is 5000 dollars -/
theorem surprise_shop_revenue : 
  daily_potential_revenue 3 6 90000 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_surprise_shop_revenue_l2404_240445


namespace NUMINAMATH_CALUDE_probability_specific_case_l2404_240401

/-- The probability of drawing a white marble first and a red marble second -/
def probability_white_then_red (total_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) : ℚ :=
  (white_marbles : ℚ) / (total_marbles : ℚ) * (red_marbles : ℚ) / ((total_marbles - 1) : ℚ)

theorem probability_specific_case :
  probability_white_then_red 10 4 6 = 4 / 15 := by
  sorry

#eval probability_white_then_red 10 4 6

end NUMINAMATH_CALUDE_probability_specific_case_l2404_240401


namespace NUMINAMATH_CALUDE_rectangle_area_double_triangle_area_double_circle_area_quadruple_fraction_unchanged_triple_negative_more_negative_all_statements_correct_l2404_240460

-- Statement A
theorem rectangle_area_double (b h : ℝ) (h_pos : 0 < h) :
  2 * (b * h) = b * (2 * h) := by sorry

-- Statement B
theorem triangle_area_double (b h : ℝ) (h_pos : 0 < h) :
  2 * ((1/2) * b * h) = (1/2) * (2 * b) * h := by sorry

-- Statement C
theorem circle_area_quadruple (r : ℝ) (r_pos : 0 < r) :
  (π * (2 * r)^2) = 4 * (π * r^2) := by sorry

-- Statement D
theorem fraction_unchanged (a b : ℝ) (b_nonzero : b ≠ 0) :
  (2 * a) / (2 * b) = a / b := by sorry

-- Statement E
theorem triple_negative_more_negative (x : ℝ) (x_neg : x < 0) :
  3 * x < x := by sorry

-- All statements are correct
theorem all_statements_correct :
  (∀ b h, 0 < h → 2 * (b * h) = b * (2 * h)) ∧
  (∀ b h, 0 < h → 2 * ((1/2) * b * h) = (1/2) * (2 * b) * h) ∧
  (∀ r, 0 < r → (π * (2 * r)^2) = 4 * (π * r^2)) ∧
  (∀ a b, b ≠ 0 → (2 * a) / (2 * b) = a / b) ∧
  (∀ x, x < 0 → 3 * x < x) := by sorry

end NUMINAMATH_CALUDE_rectangle_area_double_triangle_area_double_circle_area_quadruple_fraction_unchanged_triple_negative_more_negative_all_statements_correct_l2404_240460


namespace NUMINAMATH_CALUDE_cone_slant_height_l2404_240457

/-- Given a cone with lateral area 10π cm² and base radius 2 cm, 
    the slant height of the cone is 5 cm. -/
theorem cone_slant_height (lateral_area base_radius : ℝ) : 
  lateral_area = 10 * Real.pi ∧ base_radius = 2 → 
  lateral_area = (1 / 2) * (2 * Real.pi * base_radius) * 5 := by
sorry

end NUMINAMATH_CALUDE_cone_slant_height_l2404_240457


namespace NUMINAMATH_CALUDE_value_of_a_l2404_240417

theorem value_of_a (x y a : ℝ) 
  (h1 : 3^x = a) 
  (h2 : 5^y = a) 
  (h3 : 1/x + 1/y = 2) : 
  a = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2404_240417


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2404_240491

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 - 6 * a^2 + 9 * a + 18 = 0) →
  (3 * b^3 - 6 * b^2 + 9 * b + 18 = 0) →
  (3 * c^3 - 6 * c^2 + 9 * c + 18 = 0) →
  a^2 + b^2 + c^2 = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2404_240491


namespace NUMINAMATH_CALUDE_x2_value_l2404_240476

def sequence_condition (x : ℕ → ℝ) : Prop :=
  x 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 10 → x (n + 2) = ((x (n + 1) + 1) * (x (n + 1) - 1)) / x n) ∧
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 11 → x n > 0) ∧
  x 12 = 0

theorem x2_value (x : ℕ → ℝ) (h : sequence_condition x) : x 2 = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x2_value_l2404_240476


namespace NUMINAMATH_CALUDE_dipolia_puzzle_solution_l2404_240464

-- Define the types of people in Dipolia
inductive PersonType
| Knight
| Liar

-- Define the possible meanings of "Irgo"
inductive IrgoMeaning
| Yes
| No

-- Define the properties of knights and liars
def always_truthful (p : PersonType) : Prop :=
  p = PersonType.Knight

def always_lies (p : PersonType) : Prop :=
  p = PersonType.Liar

-- Define the scenario
structure DipoliaScenario where
  inhabitant_type : PersonType
  irgo_meaning : IrgoMeaning
  guide_truthful : Prop

-- Theorem statement
theorem dipolia_puzzle_solution (scenario : DipoliaScenario) :
  scenario.guide_truthful →
  (scenario.irgo_meaning = IrgoMeaning.Yes ∧ scenario.inhabitant_type = PersonType.Liar) :=
by sorry

end NUMINAMATH_CALUDE_dipolia_puzzle_solution_l2404_240464


namespace NUMINAMATH_CALUDE_expression_simplification_l2404_240459

theorem expression_simplification (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) (h3 : x ≠ 3) :
  (((x^2 - 2*x) / (x^2 - 4*x + 4) - 3 / (x - 2)) / ((x - 3) / (x^2 - 4))) = x + 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2404_240459


namespace NUMINAMATH_CALUDE_roger_trays_capacity_l2404_240427

/-- The number of trays Roger can carry at a time -/
def trays_per_trip : ℕ := sorry

/-- The number of trips Roger made -/
def num_trips : ℕ := 3

/-- The total number of trays Roger picked up -/
def total_trays : ℕ := 12

theorem roger_trays_capacity :
  trays_per_trip = 4 ∧ 
  num_trips * trays_per_trip = total_trays :=
by sorry

end NUMINAMATH_CALUDE_roger_trays_capacity_l2404_240427


namespace NUMINAMATH_CALUDE_no_real_roots_m_range_l2404_240430

/-- A quadratic function with parameter m -/
def f (m x : ℝ) : ℝ := x^2 + m*x + (m+3)

/-- The discriminant of the quadratic function -/
def discriminant (m : ℝ) : ℝ := m^2 - 4*(m+3)

theorem no_real_roots_m_range (m : ℝ) :
  (∀ x, f m x ≠ 0) → m ∈ Set.Ioo (-2 : ℝ) 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_m_range_l2404_240430


namespace NUMINAMATH_CALUDE_cody_final_tickets_l2404_240477

/-- Calculates the final number of tickets Cody has after various transactions at the arcade. -/
def final_tickets (initial_tickets : ℕ) (won_tickets : ℕ) (beanie_cost : ℕ) (traded_tickets : ℕ) (games_played : ℕ) (tickets_per_game : ℕ) : ℕ :=
  initial_tickets + won_tickets - beanie_cost - traded_tickets + (games_played * tickets_per_game)

/-- Theorem stating that Cody ends up with 82 tickets given the specific conditions of the problem. -/
theorem cody_final_tickets :
  final_tickets 50 49 25 10 3 6 = 82 := by sorry

end NUMINAMATH_CALUDE_cody_final_tickets_l2404_240477


namespace NUMINAMATH_CALUDE_smallest_e_value_l2404_240408

theorem smallest_e_value (a b c d e : ℤ) :
  (∃ (x : ℝ), a * x^4 + b * x^3 + c * x^2 + d * x + e = 0) →
  (-3 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  (6 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  (10 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  (-1/4 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  e > 0 →
  e ≥ 180 :=
by sorry

end NUMINAMATH_CALUDE_smallest_e_value_l2404_240408


namespace NUMINAMATH_CALUDE_sum_of_first_fifteen_multiples_of_eight_l2404_240428

/-- The sum of the first n natural numbers -/
def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n positive multiples of m -/
def sum_of_multiples (m n : ℕ) : ℕ := m * sum_of_naturals n

theorem sum_of_first_fifteen_multiples_of_eight :
  sum_of_multiples 8 15 = 960 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_fifteen_multiples_of_eight_l2404_240428


namespace NUMINAMATH_CALUDE_sqrt_sum_zero_implies_y_minus_x_l2404_240409

theorem sqrt_sum_zero_implies_y_minus_x (x y : ℝ) :
  Real.sqrt (2 * x + y) + Real.sqrt (x^2 - 9) = 0 →
  (y - x = -9 ∨ y - x = 9) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_zero_implies_y_minus_x_l2404_240409


namespace NUMINAMATH_CALUDE_equation_solution_l2404_240496

theorem equation_solution : 
  ∀ s : ℝ, (s^2 - 3*s + 2) / (s^2 - 6*s + 5) = (s^2 - 4*s - 5) / (s^2 - 2*s - 15) ↔ s = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2404_240496


namespace NUMINAMATH_CALUDE_rd_scenario_theorem_l2404_240435

/-- Represents a firm in the R&D scenario -/
structure Firm where
  participates : Bool

/-- Represents the R&D scenario -/
structure RDScenario where
  V : ℝ  -- Revenue if successful
  α : ℝ  -- Probability of success
  IC : ℝ  -- Investment cost
  firms : Fin 2 → Firm

/-- Expected revenue for a firm when both participate -/
def expectedRevenueBoth (s : RDScenario) : ℝ :=
  s.α * (1 - s.α) * s.V + 0.5 * s.α^2 * s.V

/-- Expected revenue for a firm when only one participates -/
def expectedRevenueOne (s : RDScenario) : ℝ :=
  s.α * s.V

/-- Condition for both firms to participate -/
def bothParticipateCondition (s : RDScenario) : Prop :=
  s.V * s.α * (1 - 0.5 * s.α) ≥ s.IC

/-- Total profit when both firms participate -/
def totalProfitBoth (s : RDScenario) : ℝ :=
  2 * (expectedRevenueBoth s - s.IC)

/-- Total profit when only one firm participates -/
def totalProfitOne (s : RDScenario) : ℝ :=
  expectedRevenueOne s - s.IC

/-- The main theorem to prove -/
theorem rd_scenario_theorem (s : RDScenario) 
    (h1 : 0 < s.α ∧ s.α < 1) 
    (h2 : s.V > 0) 
    (h3 : s.IC > 0) : 
  (bothParticipateCondition s ↔ expectedRevenueBoth s ≥ s.IC) ∧
  (s.V = 16 ∧ s.α = 0.5 ∧ s.IC = 5 → bothParticipateCondition s) ∧
  (s.V = 16 ∧ s.α = 0.5 ∧ s.IC = 5 → totalProfitOne s > totalProfitBoth s) := by
  sorry

end NUMINAMATH_CALUDE_rd_scenario_theorem_l2404_240435


namespace NUMINAMATH_CALUDE_prove_a_equals_six_l2404_240470

/-- Given a function f' and a real number a, proves that a = 6 -/
theorem prove_a_equals_six (f' : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f' x = 2 * x^3 + a * x^2 + x) →
  f' 1 = 9 →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_prove_a_equals_six_l2404_240470


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l2404_240454

noncomputable def f (x : ℝ) : ℝ := 6 * (x^(1/3)) - (16/3) * (x^(1/4))

theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) → y = (2/3) * x :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l2404_240454


namespace NUMINAMATH_CALUDE_cost_to_fly_AB_l2404_240415

/-- The cost of flying between two cities -/
def flying_cost (distance : ℝ) : ℝ :=
  120 + 0.12 * distance

/-- The distance from A to B in kilometers -/
def distance_AB : ℝ := 4500

theorem cost_to_fly_AB : flying_cost distance_AB = 660 := by
  sorry

end NUMINAMATH_CALUDE_cost_to_fly_AB_l2404_240415


namespace NUMINAMATH_CALUDE_min_value_theorem_l2404_240473

theorem min_value_theorem (x y : ℝ) (h1 : x + y = 1) (h2 : x > 0) (h3 : y > 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 
    (1 / (2 * x) + x / (y + 1)) ≤ (1 / (2 * a) + a / (b + 1))) ∧
  (1 / (2 * x) + x / (y + 1) = 5/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2404_240473


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l2404_240485

theorem equilateral_triangle_area (perimeter : ℝ) (area : ℝ) :
  perimeter = 120 →
  area = (400 : ℝ) * Real.sqrt 3 →
  ∃ (side : ℝ), 
    side > 0 ∧
    perimeter = 3 * side ∧
    area = (Real.sqrt 3 / 4) * side^2 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l2404_240485


namespace NUMINAMATH_CALUDE_function_satisfies_differential_equation_l2404_240421

/-- Prove that the function y = x(c - ln x) satisfies the differential equation (x - y) dx + x · dy = 0 -/
theorem function_satisfies_differential_equation (x : ℝ) (c : ℝ) :
  let y := x * (c - Real.log x)
  (x - y) * 1 + x * (c - Real.log x - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_differential_equation_l2404_240421


namespace NUMINAMATH_CALUDE_total_weight_of_cans_l2404_240449

theorem total_weight_of_cans (weights : List ℕ) (h : weights = [444, 459, 454, 459, 454, 454, 449, 454, 459, 464]) : 
  weights.sum = 4550 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_cans_l2404_240449
