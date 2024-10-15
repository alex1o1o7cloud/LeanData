import Mathlib

namespace NUMINAMATH_CALUDE_coefficient_f_nonzero_l589_58992

-- Define the polynomial Q(x)
def Q (a b c d f : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + f

-- Define the theorem
theorem coefficient_f_nonzero 
  (a b c d f : ℝ) 
  (h1 : ∃ p q r s : ℝ, p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ 
                       Q a b c d f p = 0 ∧ Q a b c d f q = 0 ∧ Q a b c d f r = 0 ∧ Q a b c d f s = 0)
  (h2 : Q a b c d f 1 = 0) : 
  f ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_f_nonzero_l589_58992


namespace NUMINAMATH_CALUDE_pet_store_dogs_l589_58963

/-- The number of dogs in a pet store with dogs and parakeets -/
def num_dogs : ℕ := 6

/-- The number of parakeets in the pet store -/
def num_parakeets : ℕ := 15 - num_dogs

/-- The total number of heads in the pet store -/
def total_heads : ℕ := 15

/-- The total number of feet in the pet store -/
def total_feet : ℕ := 42

theorem pet_store_dogs :
  num_dogs + num_parakeets = total_heads ∧
  4 * num_dogs + 2 * num_parakeets = total_feet :=
by sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l589_58963


namespace NUMINAMATH_CALUDE_journey_length_l589_58993

theorem journey_length :
  ∀ (total : ℚ),
  (1 / 4 : ℚ) * total +  -- First part
  30 +                   -- Second part (city)
  (1 / 7 : ℚ) * total    -- Third part
  = total                -- Sum of all parts equals total
  →
  total = 840 / 17 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_length_l589_58993


namespace NUMINAMATH_CALUDE_estate_division_l589_58975

theorem estate_division (total_estate : ℚ) : 
  total_estate > 0 → 
  ∃ (son_share mother_share daughter_share : ℚ),
    son_share = (4 : ℚ) / 7 * total_estate ∧
    mother_share = (2 : ℚ) / 7 * total_estate ∧
    daughter_share = (1 : ℚ) / 7 * total_estate ∧
    son_share + mother_share + daughter_share = total_estate ∧
    son_share = 2 * mother_share ∧
    mother_share = 2 * daughter_share :=
by sorry

end NUMINAMATH_CALUDE_estate_division_l589_58975


namespace NUMINAMATH_CALUDE_compound_interest_rate_l589_58987

theorem compound_interest_rate : ∃ (r : ℝ), 
  (1 + r)^2 = 7/6 ∧ 
  0.0800 < r ∧ 
  r < 0.0802 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l589_58987


namespace NUMINAMATH_CALUDE_head_probability_l589_58994

/-- Represents the possible outcomes of a coin toss -/
inductive CoinOutcome
  | Head
  | Tail

/-- A fair coin toss -/
def FairCoin : Type := CoinOutcome

/-- The probability of an outcome in a fair coin toss -/
def prob (outcome : CoinOutcome) : ℚ :=
  1 / 2

theorem head_probability (c : FairCoin) :
  prob CoinOutcome.Head = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_head_probability_l589_58994


namespace NUMINAMATH_CALUDE_statement_is_proposition_l589_58952

-- Define what a proposition is
def is_proposition (s : Prop) : Prop := True

-- Define the statement we're examining
def statement : Prop := ∀ a : ℤ, Prime a → Odd a

-- Theorem stating that our statement is a proposition
theorem statement_is_proposition : is_proposition statement := by sorry

end NUMINAMATH_CALUDE_statement_is_proposition_l589_58952


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l589_58990

/-- The perimeter of a semicircle with radius 12 units is approximately 61.7 units. -/
theorem semicircle_perimeter_approx : 
  let r : ℝ := 12
  let perimeter := 2 * r + π * r
  ∃ ε > 0, abs (perimeter - 61.7) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l589_58990


namespace NUMINAMATH_CALUDE_tan_alpha_equals_one_l589_58940

theorem tan_alpha_equals_one (α : Real) 
  (h : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) : 
  Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_one_l589_58940


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l589_58933

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 2 * a 8 = a 4 * a 4) →     -- a_2, a_4, a_8 form a geometric sequence
  a 4 = 12 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l589_58933


namespace NUMINAMATH_CALUDE_inequality_proof_l589_58913

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l589_58913


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l589_58904

-- Define the colors
inductive Color
| Red
| Blue

-- Define the coloring function type
def ColoringFunction := ℕ → Color

-- Define an infinite arithmetic progression
def IsArithmeticProgression (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define a property that a coloring function contains both colors in any arithmetic progression
def ContainsBothColors (f : ColoringFunction) : Prop :=
  ∀ (a : ℕ → ℕ) (d : ℕ), 
    IsArithmeticProgression a d → 
    (∃ n : ℕ, f (a n) = Color.Red) ∧ (∃ m : ℕ, f (a m) = Color.Blue)

-- The main theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction, ContainsBothColors f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l589_58904


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l589_58920

theorem polynomial_value_theorem (x : ℝ) (h : x^2 - 8*x - 3 = 0) :
  (x - 1) * (x - 3) * (x - 5) * (x - 7) = 180 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l589_58920


namespace NUMINAMATH_CALUDE_cindy_earnings_l589_58949

/-- Calculates the earnings for teaching one math course in a month -/
def earnings_per_course (total_courses : ℕ) (total_hours_per_week : ℕ) (hourly_rate : ℕ) (weeks_per_month : ℕ) : ℕ :=
  (total_hours_per_week / total_courses) * weeks_per_month * hourly_rate

/-- Theorem: Cindy's earnings for one math course in a month is $1200 -/
theorem cindy_earnings : 
  earnings_per_course 4 48 25 4 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_cindy_earnings_l589_58949


namespace NUMINAMATH_CALUDE_fish_population_calculation_l589_58958

/-- Calculates the number of fish in a lake on May 1 based on sampling data --/
theorem fish_population_calculation (tagged_may : ℕ) (caught_sept : ℕ) (tagged_sept : ℕ)
  (death_rate : ℚ) (new_fish_rate : ℚ) 
  (h1 : tagged_may = 60)
  (h2 : caught_sept = 70)
  (h3 : tagged_sept = 3)
  (h4 : death_rate = 1/4)
  (h5 : new_fish_rate = 2/5)
  (h6 : tagged_sept ≤ caught_sept) :
  ∃ (fish_may : ℕ), fish_may = 840 := by
  sorry

end NUMINAMATH_CALUDE_fish_population_calculation_l589_58958


namespace NUMINAMATH_CALUDE_equation_solution_range_l589_58988

-- Define the equation
def equation (a m x : ℝ) : Prop :=
  a^(2*x) + (1 + 1/m)*a^x + 1 = 0

-- Define the conditions
def conditions (a m : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  -1/3 ≤ m ∧ m < 0

-- Theorem statement
theorem equation_solution_range (a m : ℝ) :
  conditions a m →
  (∃ x : ℝ, equation a m x ∧ a^x > 0) ↔
  m_range m :=
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l589_58988


namespace NUMINAMATH_CALUDE_fraction_subtraction_l589_58971

theorem fraction_subtraction : (18 : ℚ) / 42 - (3 : ℚ) / 8 = (3 : ℚ) / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l589_58971


namespace NUMINAMATH_CALUDE_suv_fuel_efficiency_l589_58973

theorem suv_fuel_efficiency (highway_mpg city_mpg : ℝ) (max_distance gallons : ℝ) 
  (h1 : highway_mpg = 12.2)
  (h2 : city_mpg = 7.6)
  (h3 : max_distance = 268.4)
  (h4 : gallons = 22)
  (h5 : max_distance = highway_mpg * gallons) :
  city_mpg * gallons = 167.2 := by
  sorry

end NUMINAMATH_CALUDE_suv_fuel_efficiency_l589_58973


namespace NUMINAMATH_CALUDE_oscar_cd_distribution_l589_58946

/-- Represents the number of CDs Oscar can pack in each box -/
def max_cds_per_box : ℕ := 2

/-- Represents the number of rock CDs Oscar needs to ship -/
def rock_cds : ℕ := 14

/-- Represents the number of pop CDs Oscar needs to ship -/
def pop_cds : ℕ := 8

/-- Theorem stating that for any non-negative integer n, if Oscar ships 2n classical CDs
    along with the rock and pop CDs, the total number of CDs can be evenly distributed
    into boxes of 2 CDs each -/
theorem oscar_cd_distribution (n : ℕ) :
  ∃ (total_boxes : ℕ), (rock_cds + 2*n + pop_cds) = max_cds_per_box * total_boxes :=
sorry

end NUMINAMATH_CALUDE_oscar_cd_distribution_l589_58946


namespace NUMINAMATH_CALUDE_largest_812_double_l589_58941

/-- Converts a base-10 number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Interprets a list of digits as a base-12 number -/
def fromBase12 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a number is an 8-12 double -/
def is812Double (n : ℕ) : Prop :=
  fromBase12 (toBase8 n) = 2 * n

theorem largest_812_double :
  (∀ m : ℕ, is812Double m → m ≤ 4032) ∧ is812Double 4032 :=
sorry

end NUMINAMATH_CALUDE_largest_812_double_l589_58941


namespace NUMINAMATH_CALUDE_value_of_expression_l589_58983

theorem value_of_expression (a b : ℝ) (h : a + 2*b - 1 = 0) : 3*a + 6*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l589_58983


namespace NUMINAMATH_CALUDE_jasons_correct_answers_l589_58909

theorem jasons_correct_answers
  (total_problems : ℕ)
  (points_for_correct : ℕ)
  (points_for_incorrect : ℕ)
  (final_score : ℕ)
  (h1 : total_problems = 12)
  (h2 : points_for_correct = 4)
  (h3 : points_for_incorrect = 1)
  (h4 : final_score = 33) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_problems ∧
    points_for_correct * correct_answers -
    points_for_incorrect * (total_problems - correct_answers) = final_score ∧
    correct_answers = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_jasons_correct_answers_l589_58909


namespace NUMINAMATH_CALUDE_sam_seashells_l589_58977

/-- The number of seashells Sam has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Sam has 17 seashells after giving away 18 from his initial 35 -/
theorem sam_seashells : remaining_seashells 35 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l589_58977


namespace NUMINAMATH_CALUDE_sequence_formulas_l589_58923

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = 1

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n, b (n + 1) = q * b n

theorem sequence_formulas
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a235 : a 2 + a 3 = a 5)
  (h_a4b12 : a 4 = 4 * b 1 - b 2)
  (h_b3a35 : b 3 = a 3 + a 5) :
  (∀ n, a n = n) ∧ (∀ n, b n = 2^n) :=
sorry

end NUMINAMATH_CALUDE_sequence_formulas_l589_58923


namespace NUMINAMATH_CALUDE_sqrt_sum_irrational_l589_58906

theorem sqrt_sum_irrational (a b : ℚ) 
  (ha : Irrational (Real.sqrt a)) 
  (hb : Irrational (Real.sqrt b)) : 
  Irrational (Real.sqrt a + Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_irrational_l589_58906


namespace NUMINAMATH_CALUDE_basketball_free_throws_l589_58918

/-- Represents the scoring of a basketball team -/
structure BasketballScore where
  two_pointers : ℕ
  three_pointers : ℕ
  free_throws : ℕ

/-- Checks if the given BasketballScore satisfies the problem conditions -/
def is_valid_score (score : BasketballScore) : Prop :=
  3 * score.three_pointers = 2 * 2 * score.two_pointers ∧
  score.free_throws = 2 * score.two_pointers - 3 ∧
  2 * score.two_pointers + 3 * score.three_pointers + score.free_throws = 73

theorem basketball_free_throws (score : BasketballScore) :
  is_valid_score score → score.free_throws = 21 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l589_58918


namespace NUMINAMATH_CALUDE_inequality_proof_l589_58903

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (b * (a + b)) + 1 / (c * (b + c)) + 1 / (a * (c + a)) ≥ 27 / (2 * (a + b + c)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l589_58903


namespace NUMINAMATH_CALUDE_subset_relation_l589_58974

theorem subset_relation (x : ℝ) : x > 1 → x^2 - x > 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_relation_l589_58974


namespace NUMINAMATH_CALUDE_function_property_l589_58967

/-- Given a function g : ℝ → ℝ satisfying g(x)g(y) - g(xy) = x - y for all real x and y,
    prove that g(3) = -2 -/
theorem function_property (g : ℝ → ℝ) 
    (h : ∀ x y : ℝ, g x * g y - g (x * y) = x - y) : 
    g 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l589_58967


namespace NUMINAMATH_CALUDE_johns_workday_end_l589_58912

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_def : minutes < 60

/-- Calculates the difference between two times in hours -/
def time_diff (t1 t2 : Time) : ℚ :=
  (t2.hours - t1.hours : ℚ) + (t2.minutes - t1.minutes : ℚ) / 60

/-- Adds hours and minutes to a given time -/
def add_time (t : Time) (h : ℕ) (m : ℕ) : Time :=
  let total_minutes := t.hours * 60 + t.minutes + h * 60 + m
  { hours := total_minutes / 60,
    minutes := total_minutes % 60,
    inv_def := by sorry }

theorem johns_workday_end (work_hours : ℕ) (lunch_break : Time) (start_time end_time : Time) :
  work_hours = 9 →
  lunch_break.hours = 1 ∧ lunch_break.minutes = 15 →
  start_time.hours = 6 ∧ start_time.minutes = 30 →
  time_diff start_time { hours := 11, minutes := 30, inv_def := by sorry } = 5 →
  add_time { hours := 11, minutes := 30, inv_def := by sorry } lunch_break.hours lunch_break.minutes = { hours := 12, minutes := 45, inv_def := by sorry } →
  add_time { hours := 12, minutes := 45, inv_def := by sorry } 4 0 = end_time →
  end_time.hours = 16 ∧ end_time.minutes = 45 :=
by sorry

end NUMINAMATH_CALUDE_johns_workday_end_l589_58912


namespace NUMINAMATH_CALUDE_allison_video_upload_ratio_l589_58938

/-- Represents the problem of calculating the ratio of days Allison uploaded videos at her initial pace to the total days in June. -/
theorem allison_video_upload_ratio :
  ∀ (x y : ℕ), 
    x + y = 30 →  -- Total days in June
    10 * x + 20 * y = 450 →  -- Total video hours uploaded
    (x : ℚ) / 30 = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_allison_video_upload_ratio_l589_58938


namespace NUMINAMATH_CALUDE_constant_b_value_l589_58915

theorem constant_b_value (a b : ℝ) : 
  (∀ x : ℝ, (x - 3) * (x - a) = x^2 - b*x - 10) → b = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_constant_b_value_l589_58915


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l589_58998

theorem baseball_card_value_decrease (initial_value : ℝ) (first_year_decrease : ℝ) (total_decrease : ℝ) 
  (h1 : first_year_decrease = 60)
  (h2 : total_decrease = 64)
  (h3 : initial_value > 0) :
  let value_after_first_year := initial_value * (1 - first_year_decrease / 100)
  let final_value := initial_value * (1 - total_decrease / 100)
  let second_year_decrease := (value_after_first_year - final_value) / value_after_first_year * 100
  second_year_decrease = 10 := by sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l589_58998


namespace NUMINAMATH_CALUDE_equivalent_ratios_l589_58905

theorem equivalent_ratios (x : ℚ) : (3 : ℚ) / 12 = x / 16 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_ratios_l589_58905


namespace NUMINAMATH_CALUDE_complex_root_modulus_l589_58981

theorem complex_root_modulus (z : ℂ) : z^2 - 2*z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_modulus_l589_58981


namespace NUMINAMATH_CALUDE_xiaoli_estimate_l589_58986

theorem xiaoli_estimate (x y z : ℝ) (hxy : x > y) (hy : y > 0) (hz : z > 0) :
  (x + z) + (y - z) = x + y := by
  sorry

end NUMINAMATH_CALUDE_xiaoli_estimate_l589_58986


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l589_58934

theorem certain_fraction_proof (x y : ℚ) : 
  x / y ≠ 0 → -- Ensure division by y is valid
  (x / y) / (1 / 5) = (3 / 4) / (2 / 5) →
  x / y = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l589_58934


namespace NUMINAMATH_CALUDE_max_n_value_l589_58979

theorem max_n_value (a b c : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c) 
  (h3 : (a - b)⁻¹ + (b - c)⁻¹ ≥ n / (a - c)) : n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_n_value_l589_58979


namespace NUMINAMATH_CALUDE_fourteen_trucks_sufficient_l589_58985

/-- Represents the number of packages of each size -/
structure PackageDistribution where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the capacity of a Type B truck for each package size -/
structure TruckCapacity where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the number of Type B trucks needed given a package distribution and truck capacity -/
def trucksNeeded (packages : PackageDistribution) (capacity : TruckCapacity) : ℕ :=
  let smallTrucks := (packages.small + capacity.small - 1) / capacity.small
  let mediumTrucks := (packages.medium + capacity.medium - 1) / capacity.medium
  let largeTrucks := (packages.large + capacity.large - 1) / capacity.large
  smallTrucks + mediumTrucks + largeTrucks

/-- Theorem stating that 14 Type B trucks are sufficient for the given package distribution -/
theorem fourteen_trucks_sufficient 
  (packages : PackageDistribution)
  (capacity : TruckCapacity)
  (h1 : packages.small + packages.medium + packages.large = 1000)
  (h2 : packages.small = 2 * packages.medium)
  (h3 : packages.medium = 3 * packages.large)
  (h4 : capacity.small = 90)
  (h5 : capacity.medium = 60)
  (h6 : capacity.large = 50) :
  trucksNeeded packages capacity ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_fourteen_trucks_sufficient_l589_58985


namespace NUMINAMATH_CALUDE_ln_graph_rotation_l589_58961

open Real

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := log x

-- Define the rotation angle
variable (θ : ℝ)

-- State the theorem
theorem ln_graph_rotation (h : ∃ x > 0, f x * cos θ + x * sin θ = 0) :
  sin θ = ℯ * cos θ :=
sorry

end NUMINAMATH_CALUDE_ln_graph_rotation_l589_58961


namespace NUMINAMATH_CALUDE_expression_simplification_l589_58953

theorem expression_simplification (m : ℝ) : 
  ((7*m + 3) - 3*m*2)*4 + (5 - 2/4)*(8*m - 12) = 40*m - 42 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l589_58953


namespace NUMINAMATH_CALUDE_ratio_multiple_choice_to_free_response_l589_58907

/-- Represents the number of problems of each type in Stacy's homework assignment --/
structure HomeworkAssignment where
  total : Nat
  truefalse : Nat
  freeresponse : Nat
  multiplechoice : Nat

/-- Conditions for Stacy's homework assignment --/
def stacysHomework : HomeworkAssignment where
  total := 45
  truefalse := 6
  freeresponse := 13  -- 6 + 7
  multiplechoice := 26 -- 45 - (13 + 6)

theorem ratio_multiple_choice_to_free_response :
  (stacysHomework.multiplechoice : ℚ) / stacysHomework.freeresponse = 2 / 1 := by
  sorry

#check ratio_multiple_choice_to_free_response

end NUMINAMATH_CALUDE_ratio_multiple_choice_to_free_response_l589_58907


namespace NUMINAMATH_CALUDE_cheryl_egg_difference_l589_58991

/-- The number of eggs found by Kevin -/
def kevin_eggs : ℕ := 5

/-- The number of eggs found by Bonnie -/
def bonnie_eggs : ℕ := 13

/-- The number of eggs found by George -/
def george_eggs : ℕ := 9

/-- The number of eggs found by Cheryl -/
def cheryl_eggs : ℕ := 56

/-- Theorem stating that Cheryl found 29 more eggs than the other three children combined -/
theorem cheryl_egg_difference : 
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_egg_difference_l589_58991


namespace NUMINAMATH_CALUDE_empty_can_weight_is_two_l589_58919

/-- Calculates the weight of each empty can given the total weight, number of soda cans, soda weight per can, and number of empty cans. -/
def empty_can_weight (total_weight : ℕ) (soda_cans : ℕ) (soda_weight_per_can : ℕ) (empty_cans : ℕ) : ℕ :=
  (total_weight - soda_cans * soda_weight_per_can) / (soda_cans + empty_cans)

/-- Proves that each empty can weighs 2 ounces given the problem conditions. -/
theorem empty_can_weight_is_two :
  empty_can_weight 88 6 12 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_empty_can_weight_is_two_l589_58919


namespace NUMINAMATH_CALUDE_annie_money_left_l589_58965

/-- Calculates the amount of money Annie has left after buying hamburgers and milkshakes. -/
def money_left (initial_money hamburger_price milkshake_price hamburger_count milkshake_count : ℕ) : ℕ :=
  initial_money - (hamburger_price * hamburger_count + milkshake_price * milkshake_count)

/-- Proves that Annie has $70 left after her purchases. -/
theorem annie_money_left :
  money_left 132 4 5 8 6 = 70 := by
  sorry

end NUMINAMATH_CALUDE_annie_money_left_l589_58965


namespace NUMINAMATH_CALUDE_subtract_preserves_inequality_l589_58966

theorem subtract_preserves_inequality (a b : ℝ) (h : a > b) : a - 3 > b - 3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_preserves_inequality_l589_58966


namespace NUMINAMATH_CALUDE_complex_root_quadratic_l589_58962

theorem complex_root_quadratic (b c : ℝ) : 
  (Complex.I * Real.sqrt 2 + 1) ^ 2 + b * (Complex.I * Real.sqrt 2 + 1) + c = 0 → 
  b = -2 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_quadratic_l589_58962


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l589_58935

def total_spent (robert_pens julia_pens_multiplier dorothy_pens_divisor : ℕ)
                (pen_cost : ℚ)
                (robert_pencils julia_pencils_difference dorothy_pencils_multiplier : ℕ)
                (pencil_cost : ℚ)
                (dorothy_notebooks julia_notebooks_addition robert_notebooks_divisor : ℕ)
                (notebook_cost : ℚ) : ℚ :=
  let julia_pens := julia_pens_multiplier * robert_pens
  let dorothy_pens := julia_pens / dorothy_pens_divisor
  let total_pens := robert_pens + julia_pens + dorothy_pens
  let pens_cost := (total_pens : ℚ) * pen_cost

  let julia_pencils := robert_pencils - julia_pencils_difference
  let dorothy_pencils := dorothy_pencils_multiplier * julia_pencils
  let total_pencils := robert_pencils + julia_pencils + dorothy_pencils
  let pencils_cost := (total_pencils : ℚ) * pencil_cost

  let julia_notebooks := dorothy_notebooks + julia_notebooks_addition
  let robert_notebooks := julia_notebooks / robert_notebooks_divisor
  let total_notebooks := dorothy_notebooks + julia_notebooks + robert_notebooks
  let notebooks_cost := (total_notebooks : ℚ) * notebook_cost

  pens_cost + pencils_cost + notebooks_cost

theorem total_spent_is_correct :
  total_spent 4 3 2 (3/2) 12 5 2 (3/4) 3 1 2 4 = 93.75 := by sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l589_58935


namespace NUMINAMATH_CALUDE_convex_nonagon_diagonals_l589_58937

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem convex_nonagon_diagonals : 
  nonagon_diagonals = 27 := by sorry

end NUMINAMATH_CALUDE_convex_nonagon_diagonals_l589_58937


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l589_58936

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- Theorem: The center coordinates of the circle x^2 + y^2 + 2x - 4y = 0 are (-1, 2) -/
theorem circle_center_coordinates :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l589_58936


namespace NUMINAMATH_CALUDE_product_expansion_evaluation_l589_58996

theorem product_expansion_evaluation :
  ∀ (a b c d : ℝ),
  (∀ x : ℝ, (4 * x^2 - 3 * x + 6) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 48 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_evaluation_l589_58996


namespace NUMINAMATH_CALUDE_total_books_l589_58911

-- Define the number of books for each person
def harry_books : ℕ := 50
def flora_books : ℕ := 2 * harry_books
def gary_books : ℕ := harry_books / 2

-- Theorem to prove
theorem total_books : harry_books + flora_books + gary_books = 175 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l589_58911


namespace NUMINAMATH_CALUDE_complement_of_M_l589_58932

/-- The complement of set M in the real numbers -/
theorem complement_of_M (x : ℝ) :
  x ∈ (Set.univ : Set ℝ) \ {x : ℝ | x^2 - 4 ≤ 0} ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l589_58932


namespace NUMINAMATH_CALUDE_savings_difference_l589_58917

def savings_problem : Prop :=
  let dick_1989 := 5000
  let jane_1989 := 5000
  let dick_1990 := dick_1989 * 1.10
  let jane_1990 := jane_1989 * 0.95
  let dick_1991 := dick_1990 * 1.07
  let jane_1991 := jane_1990 * 1.08
  let dick_1992 := dick_1991 * 0.88
  let jane_1992 := jane_1991 * 1.15
  let dick_total := dick_1989 + dick_1990 + dick_1991 + dick_1992
  let jane_total := jane_1989 + jane_1990 + jane_1991 + jane_1992
  dick_total - jane_total = 784.30

theorem savings_difference : savings_problem := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l589_58917


namespace NUMINAMATH_CALUDE_trisection_dot_product_l589_58968

/-- Given three points A, B, C in 2D space, and E, F as trisection points of BC,
    prove that the dot product of vectors AE and AF is 3. -/
theorem trisection_dot_product (A B C E F : ℝ × ℝ) : 
  A = (1, 2) →
  B = (2, -1) →
  C = (2, 2) →
  E = B + (1/3 : ℝ) • (C - B) →
  F = B + (2/3 : ℝ) • (C - B) →
  (E.1 - A.1) * (F.1 - A.1) + (E.2 - A.2) * (F.2 - A.2) = 3 := by
  sorry

#check trisection_dot_product

end NUMINAMATH_CALUDE_trisection_dot_product_l589_58968


namespace NUMINAMATH_CALUDE_arithmetic_sequence_61st_term_l589_58939

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a_15 : a 15 = 33
  a_45 : a 45 = 153

/-- Theorem: In the given arithmetic sequence, the 61st term is 217 -/
theorem arithmetic_sequence_61st_term (seq : ArithmeticSequence) : seq.a 61 = 217 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_61st_term_l589_58939


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l589_58995

/-- Given a line with equation 3x - y + 2 = 0, its symmetric line with respect to the y-axis has the equation 3x + y - 2 = 0 -/
theorem symmetric_line_equation (x y : ℝ) :
  (3 * x - y + 2 = 0) → 
  ∃ (x' y' : ℝ), (3 * x' + y' - 2 = 0 ∧ x' = -x ∧ y' = y) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l589_58995


namespace NUMINAMATH_CALUDE_class_test_percentages_l589_58943

theorem class_test_percentages
  (percent_first : ℝ)
  (percent_second : ℝ)
  (percent_both : ℝ)
  (h1 : percent_first = 75)
  (h2 : percent_second = 35)
  (h3 : percent_both = 30) :
  100 - (percent_first + percent_second - percent_both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_class_test_percentages_l589_58943


namespace NUMINAMATH_CALUDE_inequality_proof_l589_58956

theorem inequality_proof (a b c d : ℝ) (h1 : a * b > 0) (h2 : -c / a < -d / b) :
  b * c > a * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l589_58956


namespace NUMINAMATH_CALUDE_madeline_water_goal_l589_58902

/-- The amount of water Madeline wants to drink in a day -/
def waterGoal (bottleCapacity : ℕ) (refills : ℕ) (additionalWater : ℕ) : ℕ :=
  bottleCapacity * refills + additionalWater

/-- Proves that Madeline's water goal is 100 ounces -/
theorem madeline_water_goal :
  waterGoal 12 7 16 = 100 := by
  sorry

end NUMINAMATH_CALUDE_madeline_water_goal_l589_58902


namespace NUMINAMATH_CALUDE_cubic_minus_four_ab_squared_factorization_l589_58927

theorem cubic_minus_four_ab_squared_factorization (a b : ℝ) :
  a^3 - 4*a*b^2 = a*(a+2*b)*(a-2*b) := by sorry

end NUMINAMATH_CALUDE_cubic_minus_four_ab_squared_factorization_l589_58927


namespace NUMINAMATH_CALUDE_simplify_expression_l589_58984

theorem simplify_expression (y : ℝ) : 2 - (2 * (1 - (3 - (2 * (2 - y))))) = -2 + 4 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l589_58984


namespace NUMINAMATH_CALUDE_octal_724_equals_468_l589_58924

/-- Converts an octal number represented as a list of digits to its decimal equivalent. -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the number -/
def octal_number : List Nat := [4, 2, 7]

theorem octal_724_equals_468 :
  octal_to_decimal octal_number = 468 := by
  sorry

end NUMINAMATH_CALUDE_octal_724_equals_468_l589_58924


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_p_or_q_l589_58999

theorem not_p_necessary_not_sufficient_for_not_p_or_q (p q : Prop) :
  (∀ (h : ¬p ∨ q), ¬p) ∧ 
  ¬(∀ (h : ¬p), ¬(p ∨ q)) :=
sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_p_or_q_l589_58999


namespace NUMINAMATH_CALUDE_not_divides_power_diff_l589_58947

theorem not_divides_power_diff (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) 
  (hm_odd : Odd m) (hn_odd : Odd n) : 
  ¬ ((2^m - 1) ∣ (3^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_diff_l589_58947


namespace NUMINAMATH_CALUDE_limit_calculation_l589_58901

open Real

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

theorem limit_calculation :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f (1 + Δx) - f (1 - 2*Δx)) / Δx + 3/exp 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_calculation_l589_58901


namespace NUMINAMATH_CALUDE_point_lies_on_graph_l589_58970

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define a point lying on the graph of a function
def LiesOnGraph (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

-- Theorem statement
theorem point_lies_on_graph (f : ℝ → ℝ) (a : ℝ) 
  (h : EvenFunction f) : LiesOnGraph f (-a) (f a) := by
  sorry

end NUMINAMATH_CALUDE_point_lies_on_graph_l589_58970


namespace NUMINAMATH_CALUDE_fraction_problem_l589_58989

theorem fraction_problem (n d : ℚ) : 
  d = 2 * n - 1 → 
  (n + 1) / (d + 1) = 3 / 5 → 
  n / d = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l589_58989


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_def_l589_58928

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (a : ℝ) : ℝ := Real.sqrt a

-- State the theorem
theorem arithmetic_sqrt_def (a : ℝ) (h : 0 < a) : 
  arithmetic_sqrt a = Real.sqrt a := by sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_def_l589_58928


namespace NUMINAMATH_CALUDE_total_flowers_l589_58948

theorem total_flowers (class_a_students class_b_students flowers_per_student : ℕ) 
  (h1 : class_a_students = 48)
  (h2 : class_b_students = 48)
  (h3 : flowers_per_student = 16) :
  class_a_students * flowers_per_student + class_b_students * flowers_per_student = 1536 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l589_58948


namespace NUMINAMATH_CALUDE_rhombus_area_from_square_circumference_l589_58921

/-- The area of a rhombus formed by connecting the midpoints of a square's sides,
    given the square's circumference. -/
theorem rhombus_area_from_square_circumference (circumference : ℝ) :
  circumference = 96 →
  let square_side := circumference / 4
  let rhombus_area := square_side^2 / 2
  rhombus_area = 288 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_from_square_circumference_l589_58921


namespace NUMINAMATH_CALUDE_easter_eggs_per_basket_l589_58925

theorem easter_eggs_per_basket : ∃ (n : ℕ), n ≥ 5 ∧ n ∣ 30 ∧ n ∣ 42 ∧ ∀ (m : ℕ), m ≥ 5 ∧ m ∣ 30 ∧ m ∣ 42 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_easter_eggs_per_basket_l589_58925


namespace NUMINAMATH_CALUDE_jessica_purchases_total_cost_l589_58976

/-- The total cost of Jessica's purchases is $41.44 -/
theorem jessica_purchases_total_cost :
  let cat_toy := 10.22
  let cage := 11.73
  let cat_food := 8.15
  let collar := 4.35
  let litter_box := 6.99
  cat_toy + cage + cat_food + collar + litter_box = 41.44 := by
  sorry

end NUMINAMATH_CALUDE_jessica_purchases_total_cost_l589_58976


namespace NUMINAMATH_CALUDE_min_floor_sum_l589_58964

theorem min_floor_sum (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ⌊(a + b + d) / c⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + a + d) / b⌋ ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_floor_sum_l589_58964


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l589_58960

/-- An isosceles triangle with side lengths 5 and 11 has a perimeter of 27. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 5 ∧ b = 11 ∧ c = 11) ∨ (a = 11 ∧ b = 5 ∧ c = 11) →
    IsoscelesTriangle a b c →
    a + b + c = 27
  where
    IsoscelesTriangle a b c := (a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)

theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 5 11 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l589_58960


namespace NUMINAMATH_CALUDE_quadratic_inequality_l589_58900

/-- A quadratic function with a symmetry axis at x = 2 -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The symmetry axis of f is at x = 2 -/
def symmetry_axis (b c : ℝ) : Prop := ∀ x : ℝ, f b c (2 - x) = f b c (2 + x)

theorem quadratic_inequality (b c : ℝ) (h : symmetry_axis b c) : 
  f b c 2 > f b c 1 ∧ f b c 1 > f b c 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l589_58900


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l589_58969

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planeparallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_parallel_lines
  (m n : Line) (α β : Plane)
  (h1 : parallel m n)
  (h2 : perpendicular m α)
  (h3 : perpendicular n β) :
  planeparallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l589_58969


namespace NUMINAMATH_CALUDE_find_certain_number_l589_58951

theorem find_certain_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((x + 70 + 16) / 3) + 8 → x = 10 := by
sorry

end NUMINAMATH_CALUDE_find_certain_number_l589_58951


namespace NUMINAMATH_CALUDE_least_factorial_divisible_by_7350_l589_58916

theorem least_factorial_divisible_by_7350 : ∃ (n : ℕ), n > 0 ∧ 7350 ∣ n.factorial ∧ ∀ (m : ℕ), m > 0 → 7350 ∣ m.factorial → n ≤ m :=
  sorry

end NUMINAMATH_CALUDE_least_factorial_divisible_by_7350_l589_58916


namespace NUMINAMATH_CALUDE_expression_evaluation_l589_58982

theorem expression_evaluation (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (3 * x + 3)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l589_58982


namespace NUMINAMATH_CALUDE_transportation_cost_optimization_l589_58955

/-- The transportation cost problem -/
theorem transportation_cost_optimization (a : ℝ) :
  let distance : ℝ := 300
  let fuel_cost_constant : ℝ := 1/2
  let other_costs : ℝ := 800
  let cost_function (x : ℝ) : ℝ := 150 * (x + 1600 / x)
  let optimal_speed : ℝ := if a ≥ 40 then 40 else a
  0 < a →
  (∀ x > 0, x ≤ a → cost_function optimal_speed ≤ cost_function x) :=
by sorry

end NUMINAMATH_CALUDE_transportation_cost_optimization_l589_58955


namespace NUMINAMATH_CALUDE_contrapositive_truth_square_less_than_one_implies_absolute_less_than_one_contrapositive_of_square_less_than_one_is_true_l589_58910

theorem contrapositive_truth (P Q : Prop) :
  (P → Q) → (¬Q → ¬P) := by sorry

theorem square_less_than_one_implies_absolute_less_than_one :
  ∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1 := by sorry

theorem contrapositive_of_square_less_than_one_is_true :
  (∀ x : ℝ, ¬(-1 < x ∧ x < 1) → ¬(x^2 < 1)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_truth_square_less_than_one_implies_absolute_less_than_one_contrapositive_of_square_less_than_one_is_true_l589_58910


namespace NUMINAMATH_CALUDE_tuesday_boot_sales_l589_58997

/-- Represents the sales data for a day -/
structure DailySales where
  shoes : ℕ
  boots : ℕ
  total : ℚ

/-- Represents the pricing and sales data for the shoe store -/
structure ShoeStore where
  shoe_price : ℚ
  boot_price : ℚ
  monday : DailySales
  tuesday : DailySales

/-- The main theorem to prove -/
theorem tuesday_boot_sales (store : ShoeStore) : store.tuesday.boots = 24 :=
  by
  have price_difference : store.boot_price = store.shoe_price + 15 := by sorry
  have monday_equation : store.shoe_price * store.monday.shoes + store.boot_price * store.monday.boots = store.monday.total := by sorry
  have tuesday_equation : store.shoe_price * store.tuesday.shoes + store.boot_price * store.tuesday.boots = store.tuesday.total := by sorry
  have monday_sales : store.monday.shoes = 22 ∧ store.monday.boots = 16 ∧ store.monday.total = 460 := by sorry
  have tuesday_partial_sales : store.tuesday.shoes = 8 ∧ store.tuesday.total = 560 := by sorry
  sorry

end NUMINAMATH_CALUDE_tuesday_boot_sales_l589_58997


namespace NUMINAMATH_CALUDE_backpacks_sold_at_swap_meet_l589_58930

/-- Prove that 17 backpacks were sold at the swap meet given the problem conditions -/
theorem backpacks_sold_at_swap_meet :
  ∀ (x : ℕ),
  (
    -- Total number of backpacks
    48 : ℕ
  ) = (
    -- Backpacks sold at swap meet
    x
  ) + (
    -- Backpacks sold to department store
    10 : ℕ
  ) + (
    -- Remaining backpacks
    48 - x - 10
  ) ∧
  (
    -- Total revenue
    1018 : ℕ
  ) = (
    -- Revenue from swap meet
    18 * x
  ) + (
    -- Revenue from department store
    10 * 25
  ) + (
    -- Revenue from remaining backpacks
    22 * (48 - x - 10)
  ) ∧
  (
    -- Total revenue
    1018 : ℕ
  ) = (
    -- Cost of backpacks
    576 : ℕ
  ) + (
    -- Profit
    442 : ℕ
  ) →
  x = 17 := by
  sorry


end NUMINAMATH_CALUDE_backpacks_sold_at_swap_meet_l589_58930


namespace NUMINAMATH_CALUDE_mushroom_picking_ratio_l589_58929

/-- Proves the ratio of mushrooms picked on the last day to the second day -/
theorem mushroom_picking_ratio : 
  ∀ (total_mushrooms first_day_revenue second_day_picked price_per_mushroom : ℕ),
  total_mushrooms = 65 →
  first_day_revenue = 58 →
  second_day_picked = 12 →
  price_per_mushroom = 2 →
  (total_mushrooms - first_day_revenue / price_per_mushroom - second_day_picked) / second_day_picked = 2 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_picking_ratio_l589_58929


namespace NUMINAMATH_CALUDE_razorback_shop_tshirt_profit_l589_58922

/-- The amount of money made per t-shirt, given the number of t-shirts sold and the total revenue from t-shirt sales. -/
def amount_per_tshirt (num_tshirts : ℕ) (total_revenue : ℕ) : ℚ :=
  total_revenue / num_tshirts

/-- Theorem stating that the amount made per t-shirt is $215, given the conditions. -/
theorem razorback_shop_tshirt_profit :
  amount_per_tshirt 20 4300 = 215 := by
  sorry

end NUMINAMATH_CALUDE_razorback_shop_tshirt_profit_l589_58922


namespace NUMINAMATH_CALUDE_volume_ratio_volume_112oz_l589_58959

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℝ
  /-- Assumption that k is positive -/
  k_pos : k > 0

/-- Volume of the substance given its weight -/
def volume (s : Substance) (weight : ℝ) : ℝ :=
  s.k * weight

/-- Theorem stating the relationship between volumes of different weights -/
theorem volume_ratio (s : Substance) (w1 w2 v1 : ℝ) (hw1 : w1 > 0) (hw2 : w2 > 0) (hv1 : v1 > 0)
    (h : volume s w1 = v1) :
    volume s w2 = v1 * (w2 / w1) := by
  sorry

/-- Main theorem proving the volume for 112 ounces given the volume for 84 ounces -/
theorem volume_112oz (s : Substance) (h : volume s 84 = 36) :
    volume s 112 = 48 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_volume_112oz_l589_58959


namespace NUMINAMATH_CALUDE_ramesh_profit_share_l589_58914

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_profit_share (investment1 : ℕ) (investment2 : ℕ) (total_profit : ℕ) : ℕ :=
  (investment2 * total_profit) / (investment1 + investment2)

/-- Theorem stating that Ramesh's share of the profit is 11,875 --/
theorem ramesh_profit_share :
  calculate_profit_share 24000 40000 19000 = 11875 := by
  sorry

end NUMINAMATH_CALUDE_ramesh_profit_share_l589_58914


namespace NUMINAMATH_CALUDE_prime_plus_three_prime_l589_58945

theorem prime_plus_three_prime (p : ℕ) (hp : Nat.Prime p) (hp3 : Nat.Prime (p + 3)) :
  p^11 - 52 = 1996 := by
  sorry

end NUMINAMATH_CALUDE_prime_plus_three_prime_l589_58945


namespace NUMINAMATH_CALUDE_factors_of_M_l589_58926

/-- The number of natural-number factors of M, where M = 2^4 · 3^3 · 7^1 · 11^2 -/
def num_factors (M : ℕ) : ℕ :=
  if M = 2^4 * 3^3 * 7^1 * 11^2 then 120 else 0

/-- Theorem stating that the number of natural-number factors of M is 120 -/
theorem factors_of_M :
  ∀ M : ℕ, M = 2^4 * 3^3 * 7^1 * 11^2 → num_factors M = 120 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_M_l589_58926


namespace NUMINAMATH_CALUDE_largest_package_size_l589_58980

theorem largest_package_size (liam_markers zoe_markers : ℕ) 
  (h1 : liam_markers = 60) 
  (h2 : zoe_markers = 36) : 
  Nat.gcd liam_markers zoe_markers = 12 := by
sorry

end NUMINAMATH_CALUDE_largest_package_size_l589_58980


namespace NUMINAMATH_CALUDE_coffee_cost_theorem_l589_58978

/-- The cost of coffee A per kilogram -/
def coffee_A_cost : ℝ := 10

/-- The cost of coffee B per kilogram -/
def coffee_B_cost : ℝ := 12

/-- The selling price of the mixture per kilogram -/
def mixture_price : ℝ := 11

/-- The total weight of the mixture in kilograms -/
def total_mixture : ℝ := 480

/-- The weight of coffee A used in the mixture in kilograms -/
def coffee_A_weight : ℝ := 240

/-- The weight of coffee B used in the mixture in kilograms -/
def coffee_B_weight : ℝ := 240

theorem coffee_cost_theorem :
  coffee_A_weight * coffee_A_cost + coffee_B_weight * coffee_B_cost = total_mixture * mixture_price :=
by sorry

end NUMINAMATH_CALUDE_coffee_cost_theorem_l589_58978


namespace NUMINAMATH_CALUDE_baker_pastries_cakes_difference_l589_58931

theorem baker_pastries_cakes_difference (cakes pastries : ℕ) 
  (h1 : cakes = 19) 
  (h2 : pastries = 131) : 
  pastries - cakes = 112 := by
sorry

end NUMINAMATH_CALUDE_baker_pastries_cakes_difference_l589_58931


namespace NUMINAMATH_CALUDE_function_characterization_l589_58957

/-- A function from natural numbers to natural numbers. -/
def NatFunction := ℕ → ℕ

/-- The property that f(3x + 2y) = f(x)f(y) for all x, y ∈ ℕ. -/
def SatisfiesProperty (f : NatFunction) : Prop :=
  ∀ x y : ℕ, f (3 * x + 2 * y) = f x * f y

/-- The constant zero function. -/
def ZeroFunction : NatFunction := λ _ => 0

/-- The constant one function. -/
def OneFunction : NatFunction := λ _ => 1

/-- The function that is 1 at 0 and 0 elsewhere. -/
def ZeroOneFunction : NatFunction := λ n => if n = 0 then 1 else 0

/-- The main theorem stating that any function satisfying the property
    must be one of the three specified functions. -/
theorem function_characterization (f : NatFunction) 
  (h : SatisfiesProperty f) : 
  f = ZeroFunction ∨ f = OneFunction ∨ f = ZeroOneFunction :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l589_58957


namespace NUMINAMATH_CALUDE_annie_cookies_count_l589_58954

/-- The number of cookies Annie ate on Monday -/
def monday_cookies : ℕ := 5

/-- The number of cookies Annie ate on Tuesday -/
def tuesday_cookies : ℕ := 2 * monday_cookies

/-- The number of cookies Annie ate on Wednesday -/
def wednesday_cookies : ℕ := tuesday_cookies + (tuesday_cookies * 2 / 5)

/-- The total number of cookies Annie ate during the three days -/
def total_cookies : ℕ := monday_cookies + tuesday_cookies + wednesday_cookies

theorem annie_cookies_count : total_cookies = 29 := by
  sorry

end NUMINAMATH_CALUDE_annie_cookies_count_l589_58954


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l589_58908

theorem max_sum_arithmetic_sequence (x y : ℝ) (h : x^2 + y^2 = 4) :
  (∃ (z : ℝ), (3/4) * (x + 3*y) ≤ z) ∧ (∀ (z : ℝ), (3/4) * (x + 3*y) ≤ z → 3 * Real.sqrt 10 / 2 ≤ z) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l589_58908


namespace NUMINAMATH_CALUDE_bicycle_speed_correct_l589_58972

/-- The speed of bicycles that satisfies the given conditions -/
def bicycle_speed : ℝ := 15

theorem bicycle_speed_correct :
  let distance : ℝ := 10
  let car_speed : ℝ → ℝ := λ x => 2 * x
  let bicycle_time : ℝ → ℝ := λ x => distance / x
  let car_time : ℝ → ℝ := λ x => distance / (car_speed x)
  let time_difference : ℝ := 1 / 3
  bicycle_time bicycle_speed = car_time bicycle_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_bicycle_speed_correct_l589_58972


namespace NUMINAMATH_CALUDE_afternoon_bike_sales_l589_58942

theorem afternoon_bike_sales (morning_sales : ℕ) (total_clamps : ℕ) (clamps_per_bike : ℕ) :
  morning_sales = 19 →
  total_clamps = 92 →
  clamps_per_bike = 2 →
  ∃ (afternoon_sales : ℕ), 
    afternoon_sales = 27 ∧
    total_clamps = clamps_per_bike * (morning_sales + afternoon_sales) :=
by sorry

end NUMINAMATH_CALUDE_afternoon_bike_sales_l589_58942


namespace NUMINAMATH_CALUDE_route_redistribution_possible_l589_58950

/-- Represents an airline with its routes -/
structure Airline where
  id : Nat
  routes : Finset (Nat × Nat)

/-- Represents the initial configuration of airlines -/
def initial_airlines (k : Nat) : Finset Airline :=
  sorry

/-- Checks if an airline complies with the one-route-per-city law -/
def complies_with_law (a : Airline) : Prop :=
  sorry

/-- Checks if all airlines have the same number of routes -/
def equal_routes (airlines : Finset Airline) : Prop :=
  sorry

theorem route_redistribution_possible (k : Nat) :
  ∃ (new_airlines : Finset Airline),
    (∀ a ∈ new_airlines, complies_with_law a) ∧
    equal_routes new_airlines ∧
    new_airlines.card = (initial_airlines k).card :=
  sorry

end NUMINAMATH_CALUDE_route_redistribution_possible_l589_58950


namespace NUMINAMATH_CALUDE_integer_solution_2017_l589_58944

theorem integer_solution_2017 (x y z : ℤ) : 
  x + y + z + x*y + y*z + z*x + x*y*z = 2017 ↔ 
  ((x = 0 ∧ y = 1 ∧ z = 1008) ∨
   (x = 0 ∧ y = 1008 ∧ z = 1) ∨
   (x = 1 ∧ y = 0 ∧ z = 1008) ∨
   (x = 1 ∧ y = 1008 ∧ z = 0) ∨
   (x = 1008 ∧ y = 0 ∧ z = 1) ∨
   (x = 1008 ∧ y = 1 ∧ z = 0)) :=
by sorry

#check integer_solution_2017

end NUMINAMATH_CALUDE_integer_solution_2017_l589_58944
