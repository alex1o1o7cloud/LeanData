import Mathlib

namespace triangle_theorem_l2350_235025

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.c / (Real.sqrt 3 * Real.cos t.C) ∧
  t.a + t.b = 6 ∧
  t.a * t.b * Real.cos t.C = 4

-- State the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = π / 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end triangle_theorem_l2350_235025


namespace chicken_vaccine_probabilities_l2350_235049

theorem chicken_vaccine_probabilities :
  let n : ℕ := 5  -- number of chickens
  let p : ℝ := 0.8  -- probability of not being infected
  let q : ℝ := 1 - p  -- probability of being infected
  
  -- Probability of no chicken being infected
  (p ^ n : ℝ) = 1024 / 3125 ∧
  
  -- Probability of exactly one chicken being infected
  (n : ℝ) * (p ^ (n - 1)) * q = 256 / 625 :=
by sorry

end chicken_vaccine_probabilities_l2350_235049


namespace exam_probabilities_l2350_235089

/-- Represents the probability of passing the exam for each attempt -/
structure PassProbability where
  male : ℚ
  female : ℚ

/-- Represents the exam conditions -/
structure ExamConditions where
  pass_prob : PassProbability
  max_attempts : ℕ
  free_attempts : ℕ

/-- Calculates the probability of both passing within first two attempts -/
def prob_both_pass_free (conditions : ExamConditions) : ℚ :=
  sorry

/-- Calculates the probability of passing with one person requiring a third attempt -/
def prob_one_third_attempt (conditions : ExamConditions) : ℚ :=
  sorry

theorem exam_probabilities (conditions : ExamConditions) 
  (h1 : conditions.pass_prob.male = 3/4)
  (h2 : conditions.pass_prob.female = 2/3)
  (h3 : conditions.max_attempts = 5)
  (h4 : conditions.free_attempts = 2) :
  prob_both_pass_free conditions = 5/6 ∧ 
  prob_one_third_attempt conditions = 1/9 := by
  sorry

end exam_probabilities_l2350_235089


namespace festival_fruit_prices_l2350_235061

/-- Proves that given the conditions from the problem, the cost per kg of oranges is 2.2 yuan and the cost per kg of bananas is 5.4 yuan -/
theorem festival_fruit_prices :
  let orange_price : ℚ := x
  let pear_price : ℚ := x
  let apple_price : ℚ := y
  let banana_price : ℚ := y
  ∀ x y : ℚ,
  (9 * x + 10 * y = 73.8) →
  (17 * x + 6 * y = 69.8) →
  (x = 2.2 ∧ y = 5.4) :=
by
  sorry

end festival_fruit_prices_l2350_235061


namespace complement_of_union_l2350_235048

def U : Finset Nat := {1,2,3,4,5,6}
def M : Finset Nat := {1,3,6}
def N : Finset Nat := {2,3,4}

theorem complement_of_union : (U \ (M ∪ N)) = {5} := by sorry

end complement_of_union_l2350_235048


namespace negative_product_of_negatives_l2350_235032

theorem negative_product_of_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : -a * b < 0 := by
  sorry

end negative_product_of_negatives_l2350_235032


namespace log_expression_equals_negative_twenty_l2350_235005

theorem log_expression_equals_negative_twenty :
  (Real.log (1/4) - Real.log 25) / (100 ^ (-1/2 : ℝ)) = -20 := by
  sorry

end log_expression_equals_negative_twenty_l2350_235005


namespace correct_sampling_methods_l2350_235019

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics --/
structure Survey where
  population_size : ℕ
  sample_size : ℕ
  has_subgroups : Bool
  is_uniform : Bool

/-- Determines the most appropriate sampling method for a given survey --/
def best_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_subgroups then SamplingMethod.Stratified
  else if s.is_uniform && s.population_size > s.sample_size * 10 then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

/-- The three surveys described in the problem --/
def survey1 : Survey := { population_size := 10, sample_size := 3, has_subgroups := false, is_uniform := true }
def survey2 : Survey := { population_size := 32 * 40, sample_size := 32, has_subgroups := false, is_uniform := true }
def survey3 : Survey := { population_size := 160, sample_size := 20, has_subgroups := true, is_uniform := false }

theorem correct_sampling_methods :
  best_sampling_method survey1 = SamplingMethod.SimpleRandom ∧
  best_sampling_method survey2 = SamplingMethod.Systematic ∧
  best_sampling_method survey3 = SamplingMethod.Stratified :=
sorry

end correct_sampling_methods_l2350_235019


namespace mother_daughter_ages_l2350_235018

/-- Proves the ages of a mother and daughter given certain conditions --/
theorem mother_daughter_ages :
  ∀ (daughter_age mother_age : ℕ),
  mother_age = daughter_age + 22 →
  2 * (2 * daughter_age) = 2 * daughter_age + 22 →
  daughter_age = 11 ∧ mother_age = 33 :=
by
  sorry

#check mother_daughter_ages

end mother_daughter_ages_l2350_235018


namespace brownie_pieces_count_l2350_235065

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from a pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  area pan.panDimensions / area pan.pieceDimensions

/-- Theorem stating that a 24x15 inch pan can be divided into exactly 60 pieces of 3x2 inch brownies -/
theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end brownie_pieces_count_l2350_235065


namespace caging_theorem_l2350_235096

/-- The number of ways to cage 6 animals in 6 cages, where 4 cages are too small for 6 animals -/
def caging_arrangements : ℕ := 24

/-- The total number of animals -/
def total_animals : ℕ := 6

/-- The total number of cages -/
def total_cages : ℕ := 6

/-- The number of cages that are too small for most animals -/
def small_cages : ℕ := 4

/-- The number of animals that can't fit in the small cages -/
def large_animals : ℕ := 6

theorem caging_theorem : 
  caging_arrangements = 24 ∧ 
  total_animals = 6 ∧ 
  total_cages = 6 ∧ 
  small_cages = 4 ∧ 
  large_animals = 6 :=
sorry

end caging_theorem_l2350_235096


namespace specialPrimes_eq_l2350_235095

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function that checks if a number has all digits 0 to b-1 exactly once in base b -/
def hasAllDigitsOnce (n : ℕ) (b : ℕ) : Prop :=
  ∃ (digits : List ℕ), 
    (digits.length = b) ∧
    (∀ d, d ∈ digits → d < b) ∧
    (digits.toFinset = Finset.range b) ∧
    (n = digits.foldr (λ d acc => acc * b + d) 0)

/-- The set of prime numbers with the special digit property -/
def specialPrimes : Set ℕ :=
  {p | ∃ b : ℕ, isPrime p ∧ b > 1 ∧ hasAllDigitsOnce p b}

/-- The theorem stating that the set of special primes is equal to {2, 5, 7, 11, 19} -/
theorem specialPrimes_eq : specialPrimes = {2, 5, 7, 11, 19} := by sorry

end specialPrimes_eq_l2350_235095


namespace arithmetic_calculations_l2350_235001

theorem arithmetic_calculations :
  ((-6) - 3 + (-7) - (-2) = -14) ∧
  ((-1)^2023 + 5 * (-2) - 12 / (-4) = -8) := by
  sorry

end arithmetic_calculations_l2350_235001


namespace five_digit_numbers_without_specific_digits_l2350_235004

/-- The number of digits allowed in each place (excluding the first place) -/
def allowed_digits : ℕ := 8

/-- The number of digits allowed in the first place -/
def first_place_digits : ℕ := 7

/-- The total number of places in the number -/
def total_places : ℕ := 5

/-- The expected total count of valid numbers -/
def expected_total : ℕ := 28672

theorem five_digit_numbers_without_specific_digits (d : ℕ) (h : d ≠ 7) :
  first_place_digits * (allowed_digits ^ (total_places - 1)) = expected_total :=
sorry

end five_digit_numbers_without_specific_digits_l2350_235004


namespace simplify_sqrt_sum_l2350_235086

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l2350_235086


namespace unique_grid_solution_l2350_235028

-- Define the grid type
def Grid := Fin 3 → Fin 3 → ℕ

-- Define adjacency
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ 
  (i = k ∧ l.val + 1 = j.val) ∨ 
  (j = l ∧ i.val + 1 = k.val) ∨ 
  (j = l ∧ k.val + 1 = i.val)

-- Define the property of sum of adjacent cells being less than 12
def valid_sum (g : Grid) : Prop :=
  ∀ i j k l, adjacent i j k l → g i j + g k l < 12

-- Define the given partial grid
def partial_grid (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 2 = 9 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7

-- Define the property that all numbers from 1 to 9 are used
def all_numbers_used (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → ∃ i j, g i j = n

-- The main theorem
theorem unique_grid_solution :
  ∀ g : Grid, 
    valid_sum g → 
    partial_grid g → 
    all_numbers_used g → 
    g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end unique_grid_solution_l2350_235028


namespace star_3_5_l2350_235057

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 + 3*a*b + b^2

-- Theorem statement
theorem star_3_5 : star 3 5 = 79 := by sorry

end star_3_5_l2350_235057


namespace geometric_sequence_ratio_two_l2350_235035

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q ∧ a n > 0

/-- Theorem: For a geometric sequence with positive terms satisfying 2a_1 + a_2 = a_3, 
    the common ratio is 2 -/
theorem geometric_sequence_ratio_two (a : ℕ → ℝ) (q : ℝ) 
    (h_geom : GeometricSequence a q)
    (h_eq : 2 * a 1 + a 2 = a 3) : q = 2 := by
  sorry


end geometric_sequence_ratio_two_l2350_235035


namespace geometric_sequence_common_ratio_one_l2350_235072

/-- A geometric sequence with negative terms and a specific sum condition has a common ratio of 1. -/
theorem geometric_sequence_common_ratio_one 
  (a : ℕ+ → ℝ) 
  (h_geometric : ∀ n : ℕ+, a (n + 1) = a n * q) 
  (h_negative : ∀ n : ℕ+, a n < 0) 
  (h_sum : a 3 + a 7 ≥ 2 * a 5) : 
  q = 1 := by
  sorry

end geometric_sequence_common_ratio_one_l2350_235072


namespace sub_committee_count_l2350_235083

/-- The number of people in the committee -/
def totalPeople : ℕ := 8

/-- The size of each sub-committee -/
def subCommitteeSize : ℕ := 2

/-- The number of people who cannot be in the same sub-committee -/
def restrictedPair : ℕ := 1

/-- The number of valid two-person sub-committees -/
def validSubCommittees : ℕ := 27

theorem sub_committee_count :
  (Nat.choose totalPeople subCommitteeSize) - restrictedPair = validSubCommittees :=
sorry

end sub_committee_count_l2350_235083


namespace worker_hours_per_day_l2350_235097

/-- Represents a factory worker's productivity and work schedule -/
structure Worker where
  widgets_per_hour : ℕ
  days_per_week : ℕ
  widgets_per_week : ℕ

/-- Calculates the number of hours a worker works per day -/
def hours_per_day (w : Worker) : ℚ :=
  (w.widgets_per_week : ℚ) / (w.widgets_per_hour : ℚ) / (w.days_per_week : ℚ)

/-- Theorem stating that a worker with given productivity and output works 8 hours per day -/
theorem worker_hours_per_day (w : Worker)
    (h1 : w.widgets_per_hour = 20)
    (h2 : w.days_per_week = 5)
    (h3 : w.widgets_per_week = 800) :
    hours_per_day w = 8 := by
  sorry

end worker_hours_per_day_l2350_235097


namespace fraction_relation_l2350_235071

theorem fraction_relation (x : ℝ) (h : 1 / (x + 3) = 2) : 1 / (x + 5) = 2 / 5 := by
  sorry

end fraction_relation_l2350_235071


namespace unique_solution_system_l2350_235024

theorem unique_solution_system (x y z : ℝ) : 
  y^3 - 6*x^2 + 12*x - 8 = 0 ∧
  z^3 - 6*y^2 + 12*y - 8 = 0 ∧
  x^3 - 6*z^2 + 12*z - 8 = 0 →
  x = 2 ∧ y = 2 ∧ z = 2 := by
sorry

end unique_solution_system_l2350_235024


namespace sequence_sum_property_l2350_235069

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem sequence_sum_property (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → (sequence_sum a n - 1)^2 = a n * sequence_sum a n) →
  (∀ n : ℕ, n > 0 → sequence_sum a n = n / (n + 1)) :=
by sorry

end sequence_sum_property_l2350_235069


namespace parabola_focus_l2350_235026

/-- A parabola is defined by the equation x = -1/4 * y^2 -/
def parabola (x y : ℝ) : Prop := x = -1/4 * y^2

/-- The focus of a parabola is a point (f, 0) where f is a real number -/
def is_focus (f : ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, parabola x y → 
    ((x - f)^2 + y^2 = (x - (-f))^2) ∧ 
    (∀ g : ℝ, g ≠ f → ∃ x y : ℝ, parabola x y ∧ (x - g)^2 + y^2 ≠ (x - (-g))^2)

/-- The focus of the parabola x = -1/4 * y^2 is at the point (-1, 0) -/
theorem parabola_focus : is_focus (-1) parabola := by
  sorry

end parabola_focus_l2350_235026


namespace more_girls_than_boys_l2350_235073

/-- The number of girls in the school -/
def num_girls : ℕ := 739

/-- The number of boys in the school -/
def num_boys : ℕ := 337

/-- The difference between the number of girls and boys -/
def difference : ℕ := num_girls - num_boys

theorem more_girls_than_boys : difference = 402 := by
  sorry

end more_girls_than_boys_l2350_235073


namespace arithmetic_sequence_ratio_l2350_235009

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0
  h_a1_nonzero : a 1 ≠ 0
  h_geometric : (a 2) ^ 2 = a 1 * a 4

/-- The main theorem -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) :
  (seq.a 1 + seq.a 14) / seq.a 3 = 5 := by
  sorry

end arithmetic_sequence_ratio_l2350_235009


namespace floor_sum_existence_l2350_235012

theorem floor_sum_existence : ∃ (a b c : ℝ), 
  (⌊a⌋ + ⌊b⌋ = ⌊a + b⌋) ∧ (⌊a⌋ + ⌊c⌋ < ⌊a + c⌋) := by
  sorry

end floor_sum_existence_l2350_235012


namespace log_relation_l2350_235068

theorem log_relation (a b : ℝ) : 
  a = Real.log 400 / Real.log 16 → b = Real.log 20 / Real.log 2 → a = b / 2 := by
  sorry

end log_relation_l2350_235068


namespace count_integers_with_2_and_3_l2350_235078

def count_integers_with_digits (lower_bound upper_bound : ℕ) (digit1 digit2 : ℕ) : ℕ :=
  sorry

theorem count_integers_with_2_and_3 :
  count_integers_with_digits 1000 2000 2 3 = 108 :=
sorry

end count_integers_with_2_and_3_l2350_235078


namespace election_results_l2350_235042

/-- Election results theorem -/
theorem election_results 
  (vote_percentage_A : ℝ) 
  (vote_percentage_B : ℝ) 
  (vote_percentage_C : ℝ) 
  (vote_percentage_D : ℝ) 
  (majority_difference : ℕ) 
  (h1 : vote_percentage_A = 0.45) 
  (h2 : vote_percentage_B = 0.30) 
  (h3 : vote_percentage_C = 0.20) 
  (h4 : vote_percentage_D = 0.05) 
  (h5 : vote_percentage_A + vote_percentage_B + vote_percentage_C + vote_percentage_D = 1) 
  (h6 : majority_difference = 1620) : 
  ∃ (total_votes : ℕ), 
    total_votes = 10800 ∧ 
    (vote_percentage_A * total_votes : ℝ) = 4860 ∧ 
    (vote_percentage_B * total_votes : ℝ) = 3240 ∧ 
    (vote_percentage_C * total_votes : ℝ) = 2160 ∧ 
    (vote_percentage_D * total_votes : ℝ) = 540 ∧ 
    (vote_percentage_A * total_votes - vote_percentage_B * total_votes : ℝ) = majority_difference :=
by sorry


end election_results_l2350_235042


namespace parallel_tangents_ordinates_l2350_235030

/-- The curve function y = x³ - 3x² + 6x + 2 -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 6

theorem parallel_tangents_ordinates (P Q : ℝ × ℝ) :
  P.2 = f P.1 →
  Q.2 = f Q.1 →
  f' P.1 = f' Q.1 →
  P.2 = 1 →
  Q.2 = 11 := by
  sorry

end parallel_tangents_ordinates_l2350_235030


namespace sarah_copies_360_pages_l2350_235000

/-- The number of pages Sarah will copy for a meeting -/
def total_pages (num_people : ℕ) (copies_per_person : ℕ) (pages_per_contract : ℕ) : ℕ :=
  num_people * copies_per_person * pages_per_contract

/-- Proof that Sarah will copy 360 pages for the meeting -/
theorem sarah_copies_360_pages : 
  total_pages 9 2 20 = 360 := by
  sorry

end sarah_copies_360_pages_l2350_235000


namespace smallest_with_12_divisors_l2350_235003

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Check if a number has exactly 12 divisors -/
def has_12_divisors (n : ℕ+) : Prop := num_divisors n = 12

theorem smallest_with_12_divisors :
  ∃ (n : ℕ+), has_12_divisors n ∧ ∀ (m : ℕ+), has_12_divisors m → n ≤ m :=
by
  use 288
  sorry

end smallest_with_12_divisors_l2350_235003


namespace no_rational_solutions_for_equation_l2350_235074

theorem no_rational_solutions_for_equation :
  ¬ ∃ (x y z : ℚ), 11 = x^5 + 2*y^5 + 5*z^5 := by
  sorry

end no_rational_solutions_for_equation_l2350_235074


namespace no_infinite_prime_sequence_l2350_235091

theorem no_infinite_prime_sequence :
  ¬∃ (p : ℕ → ℕ), (∀ k, p (k + 1) = 5 * p k + 4) ∧ (∀ k, Nat.Prime (p k)) := by
  sorry

end no_infinite_prime_sequence_l2350_235091


namespace function_coefficient_sum_l2350_235046

theorem function_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 2) = 2 * x^2 + 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 0 := by sorry

end function_coefficient_sum_l2350_235046


namespace faster_walking_speed_l2350_235098

/-- Proves that given a person walks 50 km at 10 km/hr, if they walked at a faster speed
    for the same time and covered 70 km, the faster speed is 14 km/hr -/
theorem faster_walking_speed (actual_distance : ℝ) (original_speed : ℝ) (extra_distance : ℝ)
    (h1 : actual_distance = 50)
    (h2 : original_speed = 10)
    (h3 : extra_distance = 20) :
    let time := actual_distance / original_speed
    let total_distance := actual_distance + extra_distance
    let faster_speed := total_distance / time
    faster_speed = 14 := by sorry

end faster_walking_speed_l2350_235098


namespace triangle_height_l2350_235034

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 9.31 → base = 4.9 → height = (2 * area) / base → height = 3.8 := by
  sorry

end triangle_height_l2350_235034


namespace shopkeeper_total_amount_l2350_235084

/-- Represents the total amount a shopkeeper receives for selling cloth. -/
def totalAmount (totalMetres : ℕ) (costPrice : ℕ) (lossPerMetre : ℕ) : ℕ :=
  totalMetres * (costPrice - lossPerMetre)

/-- Proves that the shopkeeper's total amount is 18000 for the given conditions. -/
theorem shopkeeper_total_amount :
  totalAmount 600 35 5 = 18000 := by
  sorry

end shopkeeper_total_amount_l2350_235084


namespace inequality_solution_l2350_235052

theorem inequality_solution (x : ℝ) :
  (x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9) ↔ 
  (x > -9/2 ∧ x < -2) ∨ (x > (1 - Real.sqrt 5) / 2 ∧ x < (1 + Real.sqrt 5) / 2) :=
by sorry

end inequality_solution_l2350_235052


namespace connie_markers_count_l2350_235021

/-- The number of red markers Connie has -/
def red_markers : ℕ := 2315

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 1028

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers

theorem connie_markers_count : total_markers = 3343 := by
  sorry

end connie_markers_count_l2350_235021


namespace mouse_ratio_l2350_235013

/-- Represents the mouse distribution problem --/
def mouse_distribution (total_mice : ℕ) (robbie_fraction : ℚ) (store_multiple : ℕ) (feeder_fraction : ℚ) (remaining : ℕ) : Prop :=
  let robbie_mice := (total_mice : ℚ) * robbie_fraction
  let store_mice := (robbie_mice * store_multiple : ℚ)
  let before_feeder := (total_mice : ℚ) - robbie_mice - store_mice
  (before_feeder * feeder_fraction = (remaining : ℚ)) ∧
  (store_mice / robbie_mice = 3)

/-- Theorem stating the ratio of mice sold to pet store vs given to Robbie --/
theorem mouse_ratio :
  ∃ (store_multiple : ℕ),
    mouse_distribution 24 (1/6 : ℚ) store_multiple (1/2 : ℚ) 4 :=
sorry

end mouse_ratio_l2350_235013


namespace geometry_problem_l2350_235081

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 1)
def C : ℝ × ℝ := (-1, 0)

-- Define the line equation
def line_AB (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the circle equation
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 25/2

theorem geometry_problem :
  -- The equation of line AB
  (∀ x y : ℝ, (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) ↔ line_AB x y) ∧
  -- The circle with center C is tangent to line AB
  (∃ x y : ℝ, line_AB x y ∧ circle_C x y ∧
    ∀ x' y' : ℝ, line_AB x' y' → ((x' - C.1)^2 + (y' - C.2)^2 ≥ 25/2)) :=
by sorry

end geometry_problem_l2350_235081


namespace catalog_arrangements_l2350_235064

theorem catalog_arrangements : 
  let n : ℕ := 7  -- number of letters in "catalog"
  Nat.factorial n = 5040 := by sorry

end catalog_arrangements_l2350_235064


namespace intersection_implies_range_140_l2350_235076

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 6)^2 + (y - 3)^2 = 7^2
def circle2 (x y k : ℝ) : Prop := (x - 2)^2 + (y - 6)^2 = k + 40

-- Define the intersection condition
def intersect (k : ℝ) : Prop := ∃ x y : ℝ, circle1 x y ∧ circle2 x y k

-- Theorem statement
theorem intersection_implies_range_140 (a b : ℝ) :
  (∀ k : ℝ, a ≤ k ∧ k ≤ b → intersect k) → b - a = 140 :=
by sorry

end intersection_implies_range_140_l2350_235076


namespace sum_difference_theorem_l2350_235050

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def joe_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sarah_sum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => round_to_nearest_five (i + 1))

theorem sum_difference_theorem :
  joe_sum 60 - sarah_sum 60 = 270 := by
  sorry

end sum_difference_theorem_l2350_235050


namespace inequality_solution_set_l2350_235058

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 > x

-- Define the solution set
def solution_set : Set ℝ := {x | x < 0 ∨ x > 1}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end inequality_solution_set_l2350_235058


namespace regular_polygon_sides_l2350_235038

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end regular_polygon_sides_l2350_235038


namespace yogurt_topping_combinations_l2350_235051

/-- The number of yogurt flavors --/
def yogurt_flavors : ℕ := 6

/-- The number of available toppings --/
def toppings : ℕ := 8

/-- The number of toppings to choose --/
def choose_toppings : ℕ := 2

/-- Theorem stating the number of unique combinations --/
theorem yogurt_topping_combinations : 
  yogurt_flavors * Nat.choose toppings choose_toppings = 168 := by
  sorry

end yogurt_topping_combinations_l2350_235051


namespace cubic_root_sum_l2350_235062

theorem cubic_root_sum (α β γ : ℂ) : 
  α^3 - α - 1 = 0 → β^3 - β - 1 = 0 → γ^3 - γ - 1 = 0 →
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = 3 := by
sorry

end cubic_root_sum_l2350_235062


namespace sock_inventory_theorem_l2350_235020

/-- Represents the number of socks of a particular color --/
structure SockCount where
  pairs : ℕ
  singles : ℕ

/-- Represents the total sock inventory --/
structure SockInventory where
  blue : SockCount
  green : SockCount
  red : SockCount

def initial_inventory : SockInventory := {
  blue := { pairs := 20, singles := 0 },
  green := { pairs := 15, singles := 0 },
  red := { pairs := 15, singles := 0 }
}

def lost_socks : SockInventory := {
  blue := { pairs := 0, singles := 3 },
  green := { pairs := 0, singles := 2 },
  red := { pairs := 0, singles := 2 }
}

def donated_socks : SockInventory := {
  blue := { pairs := 0, singles := 10 },
  green := { pairs := 0, singles := 15 },
  red := { pairs := 0, singles := 10 }
}

def purchased_socks : SockInventory := {
  blue := { pairs := 5, singles := 0 },
  green := { pairs := 3, singles := 0 },
  red := { pairs := 2, singles := 0 }
}

def gifted_socks : SockInventory := {
  blue := { pairs := 2, singles := 0 },
  green := { pairs := 1, singles := 0 },
  red := { pairs := 0, singles := 0 }
}

def update_inventory (inv : SockInventory) (change : SockInventory) : SockInventory :=
  { blue := { pairs := inv.blue.pairs + change.blue.pairs - (inv.blue.singles + change.blue.singles) / 2,
              singles := (inv.blue.singles + change.blue.singles) % 2 },
    green := { pairs := inv.green.pairs + change.green.pairs - (inv.green.singles + change.green.singles) / 2,
               singles := (inv.green.singles + change.green.singles) % 2 },
    red := { pairs := inv.red.pairs + change.red.pairs - (inv.red.singles + change.red.singles) / 2,
             singles := (inv.red.singles + change.red.singles) % 2 } }

def total_pairs (inv : SockInventory) : ℕ :=
  inv.blue.pairs + inv.green.pairs + inv.red.pairs

theorem sock_inventory_theorem :
  let final_inventory := update_inventory 
                          (update_inventory 
                            (update_inventory 
                              (update_inventory initial_inventory lost_socks) 
                            donated_socks) 
                          purchased_socks) 
                        gifted_socks
  total_pairs final_inventory = 43 := by
  sorry

end sock_inventory_theorem_l2350_235020


namespace A_intersect_B_l2350_235022

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end A_intersect_B_l2350_235022


namespace f_satisfies_conditions_l2350_235067

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem f_satisfies_conditions :
  -- Condition 1: f is defined on {x ∈ ℝ | x ≠ 0}
  (∀ x : ℝ, x ≠ 0 → f x = Real.log (abs x)) ∧
  -- Condition 2: f is an even function
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  -- Condition 3: f is monotonically increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) ∧
  -- Condition 4: For any non-zero real numbers x and y, f(xy) = f(x) + f(y)
  (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → f (x * y) = f x + f y) := by
  sorry

end f_satisfies_conditions_l2350_235067


namespace rope_cutting_probability_l2350_235006

theorem rope_cutting_probability (rope_length : ℝ) (min_segment_length : ℝ) : 
  rope_length = 4 →
  min_segment_length = 1.5 →
  (rope_length - 2 * min_segment_length) / rope_length = 1 / 4 := by
sorry

end rope_cutting_probability_l2350_235006


namespace perception_arrangements_l2350_235056

/-- The number of distinct arrangements of letters in a word with specific letter frequencies -/
def word_arrangements (total : ℕ) (double_count : ℕ) (single_count : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial 2 ^ double_count)

/-- Theorem stating the number of arrangements for the given word structure -/
theorem perception_arrangements :
  word_arrangements 10 3 4 = 453600 := by
  sorry

#eval word_arrangements 10 3 4

end perception_arrangements_l2350_235056


namespace ludwig_daily_salary_l2350_235045

def weekly_salary : ℚ := 55
def full_days : ℕ := 4
def half_days : ℕ := 3

theorem ludwig_daily_salary : 
  ∃ (daily_salary : ℚ), 
    (daily_salary * full_days + daily_salary * half_days / 2 = weekly_salary) ∧
    daily_salary = 10 := by
sorry

end ludwig_daily_salary_l2350_235045


namespace initial_money_equals_spent_plus_left_l2350_235090

/-- The amount of money Trisha spent on meat -/
def meat_cost : ℕ := 17

/-- The amount of money Trisha spent on chicken -/
def chicken_cost : ℕ := 22

/-- The amount of money Trisha spent on veggies -/
def veggies_cost : ℕ := 43

/-- The amount of money Trisha spent on eggs -/
def eggs_cost : ℕ := 5

/-- The amount of money Trisha spent on dog food -/
def dog_food_cost : ℕ := 45

/-- The amount of money Trisha had left after shopping -/
def money_left : ℕ := 35

/-- The total amount of money Trisha spent -/
def total_spent : ℕ := meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost

/-- The theorem stating that the initial amount of money Trisha had
    is equal to the sum of all her expenses plus the amount left after shopping -/
theorem initial_money_equals_spent_plus_left :
  total_spent + money_left = 167 := by sorry

end initial_money_equals_spent_plus_left_l2350_235090


namespace four_digit_number_counts_l2350_235075

def digits : Finset ℕ := {1, 2, 3, 4, 5}

def four_digit_numbers_no_repetition : ℕ := sorry

def four_digit_numbers_with_repetition : ℕ := sorry

def odd_four_digit_numbers_no_repetition : ℕ := sorry

theorem four_digit_number_counts :
  four_digit_numbers_no_repetition = 120 ∧
  four_digit_numbers_with_repetition = 625 ∧
  odd_four_digit_numbers_no_repetition = 72 := by sorry

end four_digit_number_counts_l2350_235075


namespace distance_product_zero_l2350_235031

-- Define the curve C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 / 36 + y^2 / 16 = 1}

-- Define the line l
def l : Set (ℝ × ℝ) := {(x, y) | ∃ t : ℝ, x = 1 - t/2 ∧ y = 1 + (Real.sqrt 3 * t)/2}

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem distance_product_zero (A B : ℝ × ℝ) 
  (hA : A ∈ C ∩ l) (hB : B ∈ C ∩ l) (hAB : A ≠ B) :
  ‖P‖ * ‖P - B‖ = 0 := by
  sorry

end distance_product_zero_l2350_235031


namespace arithmetic_sequence_property_l2350_235029

/-- For an arithmetic sequence {a_n}, if a_3 + a_11 = 22, then a_7 = 11 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence property
  a 3 + a 11 = 22 →                                -- given condition
  a 7 = 11                                         -- conclusion to prove
:= by sorry

end arithmetic_sequence_property_l2350_235029


namespace similar_quadrilaterals_rectangle_areas_l2350_235063

/-- Given two similar quadrilaterals with sides (a, b, c, d) and (a', b', c', d') respectively,
    and similarity ratio k, prove that the areas of rectangles formed by opposite sides
    are proportional to k^2 -/
theorem similar_quadrilaterals_rectangle_areas
  (a b c d a' b' c' d' k : ℝ)
  (h_similar : a / a' = b / b' ∧ b / b' = c / c' ∧ c / c' = d / d' ∧ d / d' = k) :
  a * c / (a' * c') = k^2 ∧ b * d / (b' * d') = k^2 := by
  sorry

end similar_quadrilaterals_rectangle_areas_l2350_235063


namespace first_digit_389_base4_is_1_l2350_235060

-- Define a function to convert a number to its base-4 representation
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

-- Theorem statement
theorem first_digit_389_base4_is_1 :
  (toBase4 389).reverse.head? = some 1 := by
  sorry

end first_digit_389_base4_is_1_l2350_235060


namespace quadratic_diophantine_equation_solution_l2350_235070

theorem quadratic_diophantine_equation_solution 
  (a b c : ℕ+) 
  (h : (a * c : ℕ) = b^2 + b + 1) : 
  ∃ (x y : ℤ), (a : ℤ) * x^2 - (2 * (b : ℤ) + 1) * x * y + (c : ℤ) * y^2 = 1 := by
  sorry

end quadratic_diophantine_equation_solution_l2350_235070


namespace sandwich_bread_count_l2350_235016

/-- The number of pieces of bread needed for a given number of regular and double meat sandwiches -/
def breadNeeded (regularCount : ℕ) (doubleMeatCount : ℕ) : ℕ :=
  2 * regularCount + 3 * doubleMeatCount

/-- Theorem stating that 14 regular sandwiches and 12 double meat sandwiches require 64 pieces of bread -/
theorem sandwich_bread_count : breadNeeded 14 12 = 64 := by
  sorry

end sandwich_bread_count_l2350_235016


namespace hyperbola_foci_distance_l2350_235080

/-- The distance between the foci of a hyperbola defined by x^2 - y^2 = 1 is 2√2 -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x^2 - y^2 = 1 → (x - f₁.1)^2 + (y - f₁.2)^2 = (x - f₂.1)^2 + (y - f₂.2)^2) ∧
    dist f₁ f₂ = 2 * Real.sqrt 2 :=
by sorry

end hyperbola_foci_distance_l2350_235080


namespace modulo_13_residue_l2350_235099

theorem modulo_13_residue : (247 + 5 * 40 + 7 * 143 + 4 * (2^3 - 1)) % 13 = 7 := by
  sorry

end modulo_13_residue_l2350_235099


namespace extremum_implies_f_2_eq_18_l2350_235023

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2_eq_18 (a b : ℝ) :
  f' a b 1 = 0 →  -- f has a critical point at x = 1
  f a b 1 = 10 →  -- The value of f at x = 1 is 10
  f a b 2 = 18    -- Then f(2) = 18
:= by sorry

end extremum_implies_f_2_eq_18_l2350_235023


namespace geometric_sequence_sum_l2350_235008

/-- A geometric sequence with first term 1 and the sum of the third and fifth terms equal to 6 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  (∃ q : ℝ, ∀ n : ℕ, a n = q ^ (n - 1)) ∧
  a 3 + a 5 = 6

/-- The sum of the fifth and seventh terms of the geometric sequence is 12 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 5 + a 7 = 12 := by
  sorry

end geometric_sequence_sum_l2350_235008


namespace peach_problem_l2350_235007

theorem peach_problem (jake steven jill : ℕ) 
  (h1 : jake = steven - 6)
  (h2 : steven = jill + 18)
  (h3 : jake = 17) : 
  jill = 5 := by
sorry

end peach_problem_l2350_235007


namespace cylinder_volume_l2350_235040

/-- The volume of a cylinder with base radius 2 cm and height h cm is 4πh cm³ -/
theorem cylinder_volume (h : ℝ) : 
  let r : ℝ := 2
  let V : ℝ := π * r^2 * h
  V = 4 * π * h := by sorry

end cylinder_volume_l2350_235040


namespace max_notebooks_is_11_l2350_235033

/-- Represents the number of notebooks in a pack -/
inductive NotebookPack
  | Single
  | Pack4
  | Pack7

/-- The cost of a notebook pack in dollars -/
def cost (pack : NotebookPack) : ℕ :=
  match pack with
  | .Single => 2
  | .Pack4 => 6
  | .Pack7 => 9

/-- The number of notebooks in a pack -/
def notebooks (pack : NotebookPack) : ℕ :=
  match pack with
  | .Single => 1
  | .Pack4 => 4
  | .Pack7 => 7

/-- Maria's budget in dollars -/
def budget : ℕ := 15

/-- A purchase combination is valid if it doesn't exceed the budget -/
def isValidPurchase (singles pack4s pack7s : ℕ) : Prop :=
  singles * cost .Single + pack4s * cost .Pack4 + pack7s * cost .Pack7 ≤ budget

/-- The total number of notebooks for a given purchase combination -/
def totalNotebooks (singles pack4s pack7s : ℕ) : ℕ :=
  singles * notebooks .Single + pack4s * notebooks .Pack4 + pack7s * notebooks .Pack7

/-- Theorem: The maximum number of notebooks that can be purchased with the given budget is 11 -/
theorem max_notebooks_is_11 :
    ∀ singles pack4s pack7s : ℕ,
      isValidPurchase singles pack4s pack7s →
      totalNotebooks singles pack4s pack7s ≤ 11 ∧
      ∃ s p4 p7 : ℕ, isValidPurchase s p4 p7 ∧ totalNotebooks s p4 p7 = 11 :=
  sorry


end max_notebooks_is_11_l2350_235033


namespace counterexample_exists_l2350_235047

theorem counterexample_exists : ∃ a : ℝ, a^2 > 0 ∧ a ≤ 0 := by
  sorry

end counterexample_exists_l2350_235047


namespace show_revenue_l2350_235085

/-- Calculates the total revenue for two shows given the attendance of the first show,
    the multiplier for the second show's attendance, and the ticket price. -/
def totalRevenue (firstShowAttendance : ℕ) (secondShowMultiplier : ℕ) (ticketPrice : ℕ) : ℕ :=
  (firstShowAttendance + secondShowMultiplier * firstShowAttendance) * ticketPrice

/-- Theorem stating that the total revenue for both shows is $20,000 -/
theorem show_revenue : totalRevenue 200 3 25 = 20000 := by
  sorry

end show_revenue_l2350_235085


namespace cosine_equality_l2350_235088

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (980 * π / 180) → n = 100 := by
  sorry

end cosine_equality_l2350_235088


namespace max_two_digit_div_sum_of_digits_l2350_235054

theorem max_two_digit_div_sum_of_digits :
  ∀ a b : ℕ,
    1 ≤ a ∧ a ≤ 9 →
    0 ≤ b ∧ b ≤ 9 →
    ¬(a = 0 ∧ b = 0) →
    (10 * a + b) / (a + b) ≤ 10 ∧
    ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ ¬(a = 0 ∧ b = 0) ∧ (10 * a + b) / (a + b) = 10 :=
by sorry

end max_two_digit_div_sum_of_digits_l2350_235054


namespace square_plus_one_divides_l2350_235044

theorem square_plus_one_divides (n : ℕ) : (n^2 + 1) ∣ n ↔ n = 0 := by sorry

end square_plus_one_divides_l2350_235044


namespace scientific_notation_proof_l2350_235077

theorem scientific_notation_proof : 
  ∃ (a : ℝ) (n : ℤ), 680000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 6.8 ∧ n = 8 := by
  sorry

end scientific_notation_proof_l2350_235077


namespace cost_per_dozen_is_240_l2350_235093

/-- Calculates the cost per dozen donuts given the total number of donuts,
    selling price per donut, desired profit, and total number of dozens. -/
def cost_per_dozen (total_donuts : ℕ) (price_per_donut : ℚ) (desired_profit : ℚ) (total_dozens : ℕ) : ℚ :=
  let total_sales := total_donuts * price_per_donut
  let total_cost := total_sales - desired_profit
  total_cost / total_dozens

/-- Proves that the cost per dozen donuts is $2.40 given the specified conditions. -/
theorem cost_per_dozen_is_240 :
  cost_per_dozen 120 1 96 10 = 240 / 100 := by
  sorry

end cost_per_dozen_is_240_l2350_235093


namespace gcf_of_450_and_144_l2350_235094

theorem gcf_of_450_and_144 : Nat.gcd 450 144 = 18 := by
  sorry

end gcf_of_450_and_144_l2350_235094


namespace find_g_of_x_l2350_235066

theorem find_g_of_x (x : ℝ) (g : ℝ → ℝ) 
  (h : ∀ x, 4 * x^4 + 2 * x^2 - x + 7 + g x = x^3 - 4 * x^2 + 6) : 
  g = λ x => -4 * x^4 + x^3 - 6 * x^2 + x - 1 := by
  sorry

end find_g_of_x_l2350_235066


namespace max_projection_area_of_special_tetrahedron_l2350_235027

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  side_length : ℝ
  dihedral_angle : ℝ

/-- The area of the projection of the tetrahedron onto a plane -/
noncomputable def projection_area (t : Tetrahedron) (rotation_angle : ℝ) : ℝ :=
  sorry

/-- The maximum area of the projection over all rotation angles -/
noncomputable def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

theorem max_projection_area_of_special_tetrahedron :
  let t : Tetrahedron := { side_length := 1, dihedral_angle := π/3 }
  max_projection_area t = Real.sqrt 3 / 4 := by
  sorry

end max_projection_area_of_special_tetrahedron_l2350_235027


namespace miranda_savings_l2350_235011

theorem miranda_savings (total_cost sister_contribution saving_period : ℕ) 
  (h1 : total_cost = 260)
  (h2 : sister_contribution = 50)
  (h3 : saving_period = 3) :
  (total_cost - sister_contribution) / saving_period = 70 := by
  sorry

end miranda_savings_l2350_235011


namespace base_b_not_divisible_by_five_l2350_235015

def is_not_divisible_by_five (b : ℤ) : Prop :=
  ¬ (5 ∣ (2 * b^3 - 2 * b^2))

theorem base_b_not_divisible_by_five :
  ∀ b : ℤ, b ∈ ({4, 5, 7, 8, 10} : Set ℤ) →
    (is_not_divisible_by_five b ↔ b ∈ ({4, 7, 8} : Set ℤ)) :=
by sorry

end base_b_not_divisible_by_five_l2350_235015


namespace circle_properties_l2350_235037

/-- The equation of a circle in the form x^2 + y^2 + ax + by + c = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: Given the circle equation x^2 + y^2 + 4x - 6y - 3 = 0,
    prove that its center is (-2, 3) and its radius is 4 -/
theorem circle_properties :
  let eq : CircleEquation := ⟨4, -6, -3⟩
  let props : CircleProperties := ⟨(-2, 3), 4⟩
  (∀ x y : ℝ, x^2 + y^2 + eq.a * x + eq.b * y + eq.c = 0 ↔ 
    (x - props.center.1)^2 + (y - props.center.2)^2 = props.radius^2) :=
by sorry

end circle_properties_l2350_235037


namespace sum_outside_angles_inscribed_pentagon_l2350_235014

/-- A pentagon inscribed in a circle -/
structure InscribedPentagon where
  -- Define the circle
  circle : Set (ℝ × ℝ)
  -- Define the pentagon
  pentagon : Set (ℝ × ℝ)
  -- Ensure the pentagon is inscribed in the circle
  is_inscribed : pentagon ⊆ circle

/-- An angle inscribed in a segment outside the pentagon -/
def OutsideAngle (p : InscribedPentagon) : Type :=
  { θ : ℝ // 0 ≤ θ ∧ θ ≤ 2 * Real.pi }

/-- The theorem stating that the sum of angles inscribed in the five segments
    outside an inscribed pentagon is equal to 5π/2 radians (900°) -/
theorem sum_outside_angles_inscribed_pentagon (p : InscribedPentagon) 
  (α β γ δ ε : OutsideAngle p) : 
  α.val + β.val + γ.val + δ.val + ε.val = 5 * Real.pi / 2 := by
  sorry

end sum_outside_angles_inscribed_pentagon_l2350_235014


namespace min_k_for_inequality_l2350_235059

theorem min_k_for_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, (1 / a + 1 / b + k / (a + b) ≥ 0) → k ≥ -4) ∧
  (∃ k : ℝ, k = -4 ∧ 1 / a + 1 / b + k / (a + b) ≥ 0) :=
by sorry

end min_k_for_inequality_l2350_235059


namespace quadratic_inequality_range_l2350_235002

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
sorry

end quadratic_inequality_range_l2350_235002


namespace repeating_decimal_sum_l2350_235053

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.474747... -/
def x : ℚ := RepeatingDecimal 4 7

theorem repeating_decimal_sum : x = 47 / 99 ∧ 47 + 99 = 146 := by sorry

end repeating_decimal_sum_l2350_235053


namespace imaginary_part_of_z_l2350_235039

theorem imaginary_part_of_z (z : ℂ) (h : z / (2 - I) = I) : z.im = 2 := by sorry

end imaginary_part_of_z_l2350_235039


namespace digit_150_is_zero_l2350_235043

/-- The decimal representation of 16/81 -/
def decimal_rep : ℚ := 16 / 81

/-- The repeating cycle in the decimal representation of 16/81 -/
def cycle : List ℕ := [1, 9, 7, 5, 3, 0, 8, 6, 4]

/-- The length of the repeating cycle -/
def cycle_length : ℕ := 9

/-- The position of the 150th digit within the cycle -/
def position_in_cycle : ℕ := 150 % cycle_length

/-- The 150th digit after the decimal point in the decimal representation of 16/81 -/
def digit_150 : ℕ := cycle[position_in_cycle]

theorem digit_150_is_zero : digit_150 = 0 := by sorry

end digit_150_is_zero_l2350_235043


namespace largest_equal_sum_digits_l2350_235079

/-- The sum of decimal digits of a natural number -/
def sumDecimalDigits (n : ℕ) : ℕ := sorry

/-- The sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 503 is the largest number less than 1000 
    with equal sum of decimal and binary digits -/
theorem largest_equal_sum_digits : 
  ∀ n : ℕ, n < 1000 → n > 503 → 
    sumDecimalDigits n ≠ sumBinaryDigits n :=
by sorry

end largest_equal_sum_digits_l2350_235079


namespace unique_intersection_l2350_235092

/-- The value of a for which the graphs of y = ax² + 5x + 2 and y = -2x - 2 intersect at exactly one point -/
def intersection_value : ℚ := 49 / 16

/-- The first graph equation -/
def graph1 (a x : ℚ) : ℚ := a * x^2 + 5 * x + 2

/-- The second graph equation -/
def graph2 (x : ℚ) : ℚ := -2 * x - 2

/-- Theorem stating that the graphs intersect at exactly one point when a = 49/16 -/
theorem unique_intersection :
  ∃! x : ℚ, graph1 intersection_value x = graph2 x :=
sorry

end unique_intersection_l2350_235092


namespace a2b2_value_l2350_235036

def is_arithmetic_progression (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_progression (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem a2b2_value (a₁ a₂ b₁ b₂ b₃ : ℝ) 
  (h_arith : is_arithmetic_progression 1 a₁ a₂ 4)
  (h_geom : is_geometric_progression 1 b₁ b₂ b₃ 4) : 
  a₂ * b₂ = 6 := by
  sorry

end a2b2_value_l2350_235036


namespace units_digit_of_sum_of_cubes_l2350_235041

theorem units_digit_of_sum_of_cubes : 
  (42^3 + 24^3) % 10 = 2 := by sorry

end units_digit_of_sum_of_cubes_l2350_235041


namespace moving_circle_trajectory_l2350_235017

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- Define the moving circle
structure MovingCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency conditions
def internallyTangent (c : MovingCircle) : Prop :=
  C₁ (c.center.1 - c.radius) (c.center.2)

def externallyTangent (c : MovingCircle) : Prop :=
  C₂ (c.center.1 + c.radius) (c.center.2)

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := x^2 / 64 + y^2 / 48 = 1

-- The theorem to prove
theorem moving_circle_trajectory :
  ∀ (c : MovingCircle),
    internallyTangent c →
    externallyTangent c →
    trajectory c.center.1 c.center.2 :=
sorry

end moving_circle_trajectory_l2350_235017


namespace bird_migration_l2350_235010

/-- Bird migration problem -/
theorem bird_migration 
  (total_families : ℕ)
  (africa_families : ℕ)
  (asia_families : ℕ)
  (south_america_families : ℕ)
  (africa_days : ℕ)
  (asia_days : ℕ)
  (south_america_days : ℕ)
  (h1 : total_families = 200)
  (h2 : africa_families = 60)
  (h3 : asia_families = 95)
  (h4 : south_america_families = 30)
  (h5 : africa_days = 7)
  (h6 : asia_days = 14)
  (h7 : south_america_days = 10) :
  (total_families - (africa_families + asia_families + south_america_families) = 15) ∧
  (africa_families * africa_days + asia_families * asia_days + south_america_families * south_america_days = 2050) := by
  sorry


end bird_migration_l2350_235010


namespace union_condition_equiv_range_l2350_235082

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem union_condition_equiv_range (a : ℝ) :
  A a ∪ B = B ↔ a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 2) := by
  sorry

end union_condition_equiv_range_l2350_235082


namespace product_of_three_primes_l2350_235055

theorem product_of_three_primes : 
  ∃ (p q r : ℕ), 
    989 * 1001 * 1007 + 320 = p * q * r ∧ 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p < q ∧ q < r ∧
    p = 991 ∧ q = 997 ∧ r = 1009 := by
  sorry

end product_of_three_primes_l2350_235055


namespace gp_common_ratio_l2350_235087

/-- Given a geometric progression where the ratio of the sum of the first 6 terms
    to the sum of the first 3 terms is 217, prove that the common ratio is 6. -/
theorem gp_common_ratio (a r : ℝ) (h : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217 →
  r = 6 := by
sorry

end gp_common_ratio_l2350_235087
