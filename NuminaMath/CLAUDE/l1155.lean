import Mathlib

namespace fourth_term_of_arithmetic_sequence_l1155_115502

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fourth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_first : a 0 = 25) 
  (h_last : a 5 = 57) : 
  a 3 = 41 := by
sorry

end fourth_term_of_arithmetic_sequence_l1155_115502


namespace equivalent_angle_exists_l1155_115583

-- Define the angle in degrees
def angle : ℝ := -463

-- Theorem stating that there exists an equivalent angle in the form k·360° + 257°
theorem equivalent_angle_exists :
  ∃ (k : ℤ), (k : ℝ) * 360 + 257 = angle + 360 * ⌊angle / 360⌋ := by
  sorry

end equivalent_angle_exists_l1155_115583


namespace fraction_power_division_l1155_115538

theorem fraction_power_division :
  (1 / 3 : ℚ)^4 / (1 / 5 : ℚ) = 5 / 81 := by sorry

end fraction_power_division_l1155_115538


namespace jeremy_jerseys_l1155_115578

def jerseyProblem (initialAmount basketballCost shortsCost jerseyCost remainingAmount : ℕ) : Prop :=
  let totalSpent := initialAmount - remainingAmount
  let nonJerseyCost := basketballCost + shortsCost
  let jerseyTotalCost := totalSpent - nonJerseyCost
  jerseyTotalCost / jerseyCost = 5

theorem jeremy_jerseys :
  jerseyProblem 50 18 8 2 14 := by sorry

end jeremy_jerseys_l1155_115578


namespace car_distance_proof_l1155_115582

theorem car_distance_proof (speed1 speed2 speed3 : ℝ) 
  (h1 : speed1 = 180)
  (h2 : speed2 = 160)
  (h3 : speed3 = 220) :
  speed1 + speed2 + speed3 = 560 := by
  sorry

end car_distance_proof_l1155_115582


namespace complement_of_angle_A_l1155_115503

-- Define the angle A
def angle_A : ℝ := 42

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Theorem statement
theorem complement_of_angle_A :
  complement angle_A = 48 := by
  sorry

end complement_of_angle_A_l1155_115503


namespace expression_value_l1155_115573

theorem expression_value (x : ℝ) (h : x^2 - x - 3 = 0) :
  (x + 2) * (x - 2) - x * (2 - x) = 2 := by
sorry

end expression_value_l1155_115573


namespace no_nines_in_product_l1155_115594

def first_number : Nat := 123456789
def second_number : Nat := 999999999

theorem no_nines_in_product : 
  ∀ d : Nat, d ∈ (first_number * second_number).digits 10 → d ≠ 9 := by
  sorry

end no_nines_in_product_l1155_115594


namespace smallest_y_coordinate_on_ellipse_l1155_115593

/-- The ellipse is defined by the equation (x^2/49) + ((y-3)^2/25) = 1 -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2/49 + (y-3)^2/25 = 1

/-- The smallest y-coordinate of any point on the ellipse -/
def smallest_y_coordinate : ℝ := -2

/-- Theorem stating that the smallest y-coordinate of any point on the ellipse is -2 -/
theorem smallest_y_coordinate_on_ellipse :
  ∀ x y : ℝ, is_on_ellipse x y → y ≥ smallest_y_coordinate :=
by sorry

end smallest_y_coordinate_on_ellipse_l1155_115593


namespace paul_penny_count_l1155_115579

theorem paul_penny_count (k m : ℕ+) : ∃! k, ∃ m, 1 + 3 * (k - 1) = 2017 - 5 * (m - 1) := by
  sorry

end paul_penny_count_l1155_115579


namespace initial_markup_percentage_l1155_115547

theorem initial_markup_percentage (C : ℝ) (M : ℝ) : 
  (C * (1 + M) * 1.25 * 0.92 = C * 1.38) → M = 0.2 := by
  sorry

end initial_markup_percentage_l1155_115547


namespace equal_weight_partition_l1155_115586

theorem equal_weight_partition : ∃ (A B C : Finset Nat), 
  (A ∪ B ∪ C = Finset.range 556 \ {0}) ∧ 
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (A.sum id = B.sum id) ∧ (B.sum id = C.sum id) := by
  sorry

#check equal_weight_partition

end equal_weight_partition_l1155_115586


namespace geometric_sequence_sum_l1155_115523

theorem geometric_sequence_sum (a₁ a₂ a₃ a₆ a₇ a₈ : ℚ) :
  a₁ = 4096 →
  a₂ = 1024 →
  a₃ = 256 →
  a₆ = 4 →
  a₇ = 1 →
  a₈ = 1/4 →
  ∃ r : ℚ, r ≠ 0 ∧
    (∀ n : ℕ, n ≥ 1 → a₁ * r^(n-1) = a₁ * (a₂ / a₁)^(n-1)) →
    a₄ + a₅ = 80 :=
by sorry

end geometric_sequence_sum_l1155_115523


namespace triangle_properties_l1155_115526

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.c - t.a) * Real.cos t.B = t.b * Real.cos t.A)
  (h2 : t.b = 6)
  (h3 : t.c = 2 * t.a) :
  t.B = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.c * Real.sin t.B = 6 * Real.sqrt 3 := by
  sorry

end triangle_properties_l1155_115526


namespace greatest_integer_side_length_l1155_115527

theorem greatest_integer_side_length (area : ℝ) (h : area < 150) :
  ∃ (s : ℕ), s * s ≤ area ∧ ∀ (t : ℕ), t * t ≤ area → t ≤ s ∧ s = 12 :=
sorry

end greatest_integer_side_length_l1155_115527


namespace part_one_calculation_part_two_calculation_part_three_calculation_l1155_115557

-- Part 1
theorem part_one_calculation : -12 - (-18) + (-7) = -1 := by sorry

-- Part 2
theorem part_two_calculation : (4/7 - 1/9 + 2/21) * (-63) = -35 := by sorry

-- Part 3
theorem part_three_calculation : (-4)^2 / 2 + 9 * (-1/3) - |3 - 4| = 4 := by sorry

end part_one_calculation_part_two_calculation_part_three_calculation_l1155_115557


namespace min_value_theorem_l1155_115510

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_theorem_l1155_115510


namespace bracelet_sales_average_l1155_115551

theorem bracelet_sales_average (bike_cost : ℕ) (bracelet_price : ℕ) (selling_days : ℕ) 
  (h1 : bike_cost = 112)
  (h2 : bracelet_price = 1)
  (h3 : selling_days = 14) :
  (bike_cost / bracelet_price) / selling_days = 8 := by
  sorry

end bracelet_sales_average_l1155_115551


namespace jakes_weight_ratio_l1155_115563

/-- Proves that the ratio of Jake's weight after losing 20 pounds to his sister's weight is 2:1 -/
theorem jakes_weight_ratio (jake_weight sister_weight : ℕ) : 
  jake_weight = 156 →
  jake_weight + sister_weight = 224 →
  (jake_weight - 20) / sister_weight = 2 := by
sorry

end jakes_weight_ratio_l1155_115563


namespace alphazian_lost_words_l1155_115514

/-- The number of letters in the Alphazian alphabet -/
def alphabet_size : ℕ := 128

/-- The number of forbidden letters -/
def forbidden_letters : ℕ := 2

/-- The maximum word length in Alphazia -/
def max_word_length : ℕ := 2

/-- Calculates the number of lost words due to letter prohibition in Alphazia -/
def lost_words : ℕ :=
  forbidden_letters + (alphabet_size * forbidden_letters)

theorem alphazian_lost_words :
  lost_words = 258 := by sorry

end alphazian_lost_words_l1155_115514


namespace smallest_with_14_divisors_l1155_115516

/-- Count the number of positive divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number has exactly 14 positive divisors -/
def has_14_divisors (n : ℕ) : Prop :=
  count_divisors n = 14

/-- The theorem stating that 192 is the smallest positive integer with exactly 14 positive divisors -/
theorem smallest_with_14_divisors :
  (∀ m : ℕ, m > 0 → m < 192 → ¬(has_14_divisors m)) ∧ has_14_divisors 192 := by sorry

end smallest_with_14_divisors_l1155_115516


namespace set_of_values_for_a_l1155_115507

theorem set_of_values_for_a (a : ℝ) : 
  (2 ∉ {x : ℝ | x - a < 0}) ↔ a ≤ 2 := by sorry

end set_of_values_for_a_l1155_115507


namespace partitionWays_10_l1155_115519

/-- The number of ways to partition n ordered elements into 1 to n non-empty subsets,
    where the elements within each subset are contiguous. -/
def partitionWays (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => Nat.choose (n - 1) k)

/-- Theorem stating that for 10 elements, the number of partition ways is 512. -/
theorem partitionWays_10 : partitionWays 10 = 512 := by
  sorry

end partitionWays_10_l1155_115519


namespace spade_calculation_l1155_115555

def spade (k : ℕ) (x y : ℝ) : ℝ := (x + y + k) * (x - y + k)

theorem spade_calculation : 
  let k : ℕ := 2
  spade k 5 (spade k 3 2) = -392 := by
sorry

end spade_calculation_l1155_115555


namespace sum_of_roots_l1155_115598

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1716 := by
sorry

end sum_of_roots_l1155_115598


namespace distinct_digit_count_is_4032_l1155_115587

/-- The number of integers between 2000 and 9999 with four distinct digits -/
def distinct_digit_count : ℕ := sorry

/-- The range of possible first digits -/
def first_digit_range : List ℕ := List.range 8

/-- The range of possible second digits (including 0) -/
def second_digit_range : List ℕ := List.range 10

/-- The range of possible third digits -/
def third_digit_range : List ℕ := List.range 10

/-- The range of possible fourth digits -/
def fourth_digit_range : List ℕ := List.range 10

theorem distinct_digit_count_is_4032 :
  distinct_digit_count = first_digit_range.length *
                         (second_digit_range.length - 1) *
                         (third_digit_range.length - 2) *
                         (fourth_digit_range.length - 3) :=
by sorry

end distinct_digit_count_is_4032_l1155_115587


namespace additional_bags_capacity_plane_capacity_proof_l1155_115535

/-- Calculates the number of additional maximum-weight bags an airplane can hold -/
theorem additional_bags_capacity 
  (num_people : ℕ) 
  (bags_per_person : ℕ) 
  (bag_weight : ℕ) 
  (plane_capacity : ℕ) : ℕ :=
  let total_bags := num_people * bags_per_person
  let total_weight := total_bags * bag_weight
  let remaining_capacity := plane_capacity - total_weight
  remaining_capacity / bag_weight

/-- Proves that given the specific conditions, the plane can hold 90 more bags -/
theorem plane_capacity_proof :
  additional_bags_capacity 6 5 50 6000 = 90 := by
  sorry

end additional_bags_capacity_plane_capacity_proof_l1155_115535


namespace stating_transportation_equation_correct_l1155_115588

/-- Represents the rate at which Vehicle A transports goods per day -/
def vehicle_a_rate : ℚ := 1/4

/-- Represents the time Vehicle A works alone -/
def vehicle_a_solo_time : ℚ := 1

/-- Represents the time both vehicles work together -/
def combined_work_time : ℚ := 1/2

/-- Represents the total amount of goods (100%) -/
def total_goods : ℚ := 1

/-- 
Theorem stating that the equation correctly represents the transportation situation
given the conditions of the problem
-/
theorem transportation_equation_correct (x : ℚ) : 
  vehicle_a_rate * vehicle_a_solo_time + 
  combined_work_time * (vehicle_a_rate + 1/x) = total_goods := by
  sorry

end stating_transportation_equation_correct_l1155_115588


namespace max_stores_visited_is_four_l1155_115513

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  total_visits : Nat
  unique_shoppers : Nat
  two_store_visitors : Nat
  stores_in_town : Nat

/-- Calculates the maximum number of stores visited by any single person -/
def max_stores_visited (scenario : ShoppingScenario) : Nat :=
  let remaining_visits := scenario.total_visits - 2 * scenario.two_store_visitors
  let remaining_shoppers := scenario.unique_shoppers - scenario.two_store_visitors
  let extra_visits := remaining_visits - remaining_shoppers
  1 + extra_visits

/-- Theorem stating the maximum number of stores visited by any single person -/
theorem max_stores_visited_is_four (scenario : ShoppingScenario) : 
  scenario.total_visits = 23 →
  scenario.unique_shoppers = 12 →
  scenario.two_store_visitors = 8 →
  scenario.stores_in_town = 8 →
  max_stores_visited scenario = 4 := by
  sorry

#eval max_stores_visited ⟨23, 12, 8, 8⟩

end max_stores_visited_is_four_l1155_115513


namespace range_of_a_l1155_115537

/-- The function f(x) = a - x² -/
def f (a : ℝ) (x : ℝ) : ℝ := a - x^2

/-- The function g(x) = x + 2 -/
def g (x : ℝ) : ℝ := x + 2

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f a x = -g y) → 
  -2 ≤ a ∧ a ≤ 0 := by
  sorry

#check range_of_a

end range_of_a_l1155_115537


namespace unique_solution_l1155_115550

theorem unique_solution : ∃! x : ℝ, 3 * x + 3 * 12 + 3 * 16 + 11 = 134 := by
  sorry

end unique_solution_l1155_115550


namespace g_of_4_l1155_115545

/-- Given a function g: ℝ → ℝ satisfying g(x) + 3*g(2 - x) = 2*x^2 + x - 1 for all real x,
    prove that g(4) = -5/2 -/
theorem g_of_4 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 2 * x^2 + x - 1) : 
  g 4 = -5/2 := by
sorry

end g_of_4_l1155_115545


namespace even_function_f_2_l1155_115529

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * (x + 3)

-- State the theorem
theorem even_function_f_2 (a : ℝ) (h : ∀ x : ℝ, f a x = f a (-x)) : f a 2 = -5 := by
  sorry

end even_function_f_2_l1155_115529


namespace other_root_of_quadratic_l1155_115509

/-- Given that 2 is one root of the equation 5x^2 + kx = 4, prove that -2/5 is the other root -/
theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 5 * x^2 + k * x = 4 ∧ x = 2) → 
  (∃ x : ℝ, 5 * x^2 + k * x = 4 ∧ x = -2/5) :=
by sorry

end other_root_of_quadratic_l1155_115509


namespace theater_ticket_sales_l1155_115531

theorem theater_ticket_sales
  (total_tickets : ℕ)
  (adult_price senior_price : ℕ)
  (total_receipts : ℕ)
  (h1 : total_tickets = 510)
  (h2 : adult_price = 21)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 8748) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end theater_ticket_sales_l1155_115531


namespace sum_product_difference_l1155_115558

theorem sum_product_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 96) : 
  |x - y| = 4 := by sorry

end sum_product_difference_l1155_115558


namespace divisor_problem_l1155_115530

theorem divisor_problem (x d : ℕ) (h1 : x % d = 5) (h2 : (x + 13) % 41 = 18) : d = 41 := by
  sorry

end divisor_problem_l1155_115530


namespace range_of_a_l1155_115560

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) : 
  1 ≤ a ∧ a ≤ 9 := by
  sorry

end range_of_a_l1155_115560


namespace pages_left_to_write_l1155_115524

/-- Calculates the remaining pages to write given the daily page counts and total book length -/
theorem pages_left_to_write (total_pages day1 day2 day3 day4 day5 : ℝ) : 
  total_pages = 750 →
  day1 = 30 →
  day2 = 1.5 * day1 →
  day3 = 0.5 * day2 →
  day4 = 2.5 * day3 →
  day5 = 15 →
  total_pages - (day1 + day2 + day3 + day4 + day5) = 581.25 := by
  sorry

end pages_left_to_write_l1155_115524


namespace ap_has_ten_terms_l1155_115591

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  n : ℕ                  -- number of terms
  a : ℝ                  -- first term
  d : ℤ                  -- common difference
  n_even : Even n
  sum_odd : (n / 2) * (a + (a + (n - 2) * d)) = 56
  sum_even : (n / 2) * (a + d + (a + (n - 1) * d)) = 80
  last_minus_first : a + (n - 1) * d - a = 18

/-- The theorem stating that an arithmetic progression with the given properties has 10 terms -/
theorem ap_has_ten_terms (ap : ArithmeticProgression) : ap.n = 10 := by
  sorry

end ap_has_ten_terms_l1155_115591


namespace fred_marbles_count_l1155_115569

/-- Represents the number of marbles Fred has of each color -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  dark_blue : ℕ

/-- Calculates the total number of marbles -/
def total_marbles (m : MarbleCount) : ℕ :=
  m.red + m.green + m.dark_blue

/-- Theorem stating the total number of marbles Fred has -/
theorem fred_marbles_count :
  ∃ (m : MarbleCount),
    m.red = 38 ∧
    m.green = m.red / 2 ∧
    m.dark_blue = 6 ∧
    total_marbles m = 63 := by
  sorry

end fred_marbles_count_l1155_115569


namespace frank_has_twelve_cookies_l1155_115542

-- Define the number of cookies each person has
def lucy_cookies : ℕ := 5
def millie_cookies : ℕ := 2 * lucy_cookies
def mike_cookies : ℕ := 3 * millie_cookies
def frank_cookies : ℕ := mike_cookies / 2 - 3

-- Theorem to prove
theorem frank_has_twelve_cookies : frank_cookies = 12 := by
  sorry

end frank_has_twelve_cookies_l1155_115542


namespace product_of_tripled_numbers_l1155_115540

theorem product_of_tripled_numbers (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ + 1/x₁ = 3*x₁ ∧ x₂ + 1/x₂ = 3*x₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -1/2) := by
  sorry

end product_of_tripled_numbers_l1155_115540


namespace cheerful_not_green_l1155_115589

-- Define the universe of birds
variable (Bird : Type)

-- Define properties of birds
variable (green : Bird → Prop)
variable (cheerful : Bird → Prop)
variable (can_sing : Bird → Prop)
variable (can_dance : Bird → Prop)

-- Define Jen's collection of birds
variable (jen_birds : Set Bird)

-- State the theorem
theorem cheerful_not_green 
  (h1 : ∀ b ∈ jen_birds, cheerful b → can_sing b)
  (h2 : ∀ b ∈ jen_birds, green b → ¬can_dance b)
  (h3 : ∀ b ∈ jen_birds, ¬can_dance b → ¬can_sing b)
  : ∀ b ∈ jen_birds, cheerful b → ¬green b :=
by
  sorry


end cheerful_not_green_l1155_115589


namespace problem_odometer_miles_l1155_115549

/-- Represents a faulty odometer that skips certain digits --/
structure FaultyOdometer where
  skipped_digits : List Nat
  display : Nat

/-- Converts a faulty odometer reading to actual miles traveled --/
def actualMiles (o : FaultyOdometer) : Nat :=
  sorry

/-- The specific faulty odometer in the problem --/
def problemOdometer : FaultyOdometer :=
  { skipped_digits := [4, 7], display := 5006 }

/-- Theorem stating that the problemOdometer has traveled 1721 miles --/
theorem problem_odometer_miles :
  actualMiles problemOdometer = 1721 := by
  sorry

end problem_odometer_miles_l1155_115549


namespace height_survey_is_census_l1155_115599

/-- Represents a survey method --/
inductive SurveyMethod
| HeightOfStudents
| CarCrashResistance
| TVViewership
| ShoeSoleDurability

/-- Defines the properties of a census --/
structure Census where
  collectsAllData : Bool
  isFeasible : Bool

/-- Determines if a survey method is suitable for a census --/
def isSuitableForCensus (method : SurveyMethod) : Prop :=
  ∃ (c : Census), c.collectsAllData ∧ c.isFeasible

/-- The main theorem stating that measuring the height of all students is suitable for a census --/
theorem height_survey_is_census : isSuitableForCensus SurveyMethod.HeightOfStudents :=
  sorry

end height_survey_is_census_l1155_115599


namespace exponential_equation_solution_l1155_115571

theorem exponential_equation_solution :
  ∀ x : ℝ, (10 : ℝ)^x * (1000 : ℝ)^(2*x) = (100 : ℝ)^6 → x = 12/7 := by
  sorry

end exponential_equation_solution_l1155_115571


namespace permutations_of_377353752_div_by_5_l1155_115548

def original_number : ℕ := 377353752

-- Function to count occurrences of a digit in a number
def count_digit (n : ℕ) (d : ℕ) : ℕ := sorry

-- Function to calculate factorial
def factorial (n : ℕ) : ℕ := sorry

-- Function to calculate permutations of multiset
def permutations_multiset (n : ℕ) (counts : List ℕ) : ℕ := sorry

theorem permutations_of_377353752_div_by_5 :
  let digits := [3, 3, 3, 7, 7, 7, 5, 2]
  let n := digits.length
  let counts := [
    count_digit original_number 3,
    count_digit original_number 7,
    count_digit original_number 5,
    count_digit original_number 2
  ]
  permutations_multiset n counts = 1120 :=
by sorry

end permutations_of_377353752_div_by_5_l1155_115548


namespace range_of_m_l1155_115543

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 5*x - 6 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 6*x + 9 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, p x ∧ ¬(q x m)) →
  m ∈ Set.Ioo 0 3 := by
sorry

-- Note: Set.Ioo 0 3 represents the open interval (0, 3)

end range_of_m_l1155_115543


namespace max_sum_abcd_l1155_115561

theorem max_sum_abcd (a b c d : ℤ) 
  (b_pos : b > 0)
  (eq1 : a + b = c)
  (eq2 : b + c = d)
  (eq3 : c + d = a) :
  a + b + c + d ≤ -5 ∧ ∃ (a₀ b₀ c₀ d₀ : ℤ), 
    b₀ > 0 ∧ 
    a₀ + b₀ = c₀ ∧ 
    b₀ + c₀ = d₀ ∧ 
    c₀ + d₀ = a₀ ∧ 
    a₀ + b₀ + c₀ + d₀ = -5 :=
by sorry

end max_sum_abcd_l1155_115561


namespace no_wobbly_multiple_iff_div_10_or_25_l1155_115556

/-- A wobbly number is a positive integer whose digits in base 10 are alternatively non-zero and zero, with the units digit being non-zero. -/
def IsWobbly (n : ℕ) : Prop := sorry

/-- Theorem: A positive integer n does not divide any wobbly number if and only if n is divisible by 10 or 25. -/
theorem no_wobbly_multiple_iff_div_10_or_25 (n : ℕ) (hn : n > 0) :
  (∀ w : ℕ, IsWobbly w → ¬(w % n = 0)) ↔ (n % 10 = 0 ∨ n % 25 = 0) := by sorry

end no_wobbly_multiple_iff_div_10_or_25_l1155_115556


namespace jori_water_remaining_l1155_115577

/-- The amount of water remaining after usage -/
def water_remaining (initial : ℚ) (usage1 : ℚ) (usage2 : ℚ) : ℚ :=
  initial - usage1 - usage2

/-- Theorem stating the remaining water after Jori's usage -/
theorem jori_water_remaining :
  water_remaining 3 (5/4) (1/2) = 5/4 := by
  sorry

end jori_water_remaining_l1155_115577


namespace regular_polygon_exterior_angle_l1155_115584

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 60 → n = 6 := by
  sorry

end regular_polygon_exterior_angle_l1155_115584


namespace debbys_candy_l1155_115541

theorem debbys_candy (sister_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ)
  (h1 : sister_candy = 42)
  (h2 : eaten_candy = 35)
  (h3 : remaining_candy = 39) :
  ∃ (debby_candy : ℕ), debby_candy + sister_candy - eaten_candy = remaining_candy ∧ debby_candy = 32 :=
by sorry

end debbys_candy_l1155_115541


namespace peaches_left_l1155_115597

/-- Given baskets of peaches with specific initial conditions, proves the number of peaches left after removal. -/
theorem peaches_left (initial_baskets : Nat) (initial_peaches : Nat) (added_baskets : Nat) (added_peaches : Nat) (removed_peaches : Nat) : 
  initial_baskets = 5 →
  initial_peaches = 20 →
  added_baskets = 4 →
  added_peaches = 25 →
  removed_peaches = 10 →
  (initial_baskets * initial_peaches + added_baskets * added_peaches) - 
  ((initial_baskets + added_baskets) * removed_peaches) = 110 := by
  sorry

end peaches_left_l1155_115597


namespace charity_donation_l1155_115534

/-- The number of pennies collected by Cassandra -/
def cassandra_pennies : ℕ := 5000

/-- The difference in pennies collected between Cassandra and James -/
def difference : ℕ := 276

/-- The number of pennies collected by James -/
def james_pennies : ℕ := cassandra_pennies - difference

/-- The total number of pennies donated to charity -/
def total_donated : ℕ := cassandra_pennies + james_pennies

theorem charity_donation :
  total_donated = 9724 :=
sorry

end charity_donation_l1155_115534


namespace bus_departure_interval_l1155_115567

/-- Represents the scenario of Xiao Wang and the No. 18 buses -/
structure BusScenario where
  /-- Speed of Xiao Wang in meters per minute -/
  wang_speed : ℝ
  /-- Speed of the No. 18 buses in meters per minute -/
  bus_speed : ℝ
  /-- Distance between two adjacent buses traveling in the same direction in meters -/
  bus_distance : ℝ
  /-- Xiao Wang walks at a constant speed -/
  wang_constant_speed : wang_speed > 0
  /-- Buses travel at a constant speed -/
  bus_constant_speed : bus_speed > 0
  /-- A bus passes Xiao Wang from behind every 6 minutes -/
  overtake_condition : 6 * bus_speed - 6 * wang_speed = bus_distance
  /-- A bus comes towards Xiao Wang every 3 minutes -/
  approach_condition : 3 * bus_speed + 3 * wang_speed = bus_distance

/-- The interval between bus departures is 4 minutes -/
theorem bus_departure_interval (scenario : BusScenario) : 
  scenario.bus_distance = 4 * scenario.bus_speed := by
  sorry

#check bus_departure_interval

end bus_departure_interval_l1155_115567


namespace barbell_cost_l1155_115570

def number_of_barbells : ℕ := 3
def amount_given : ℕ := 850
def change_received : ℕ := 40

theorem barbell_cost :
  (amount_given - change_received) / number_of_barbells = 270 :=
by sorry

end barbell_cost_l1155_115570


namespace rectangle_ratio_l1155_115544

theorem rectangle_ratio (s w h : ℝ) (h1 : w > 0) (h2 : h > 0) (h3 : s > 0) : 
  (s + 2*w) * (s + h) = 3 * s^2 → h / w = 1 := by
  sorry

end rectangle_ratio_l1155_115544


namespace book_price_change_l1155_115595

theorem book_price_change (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 + 20 / 100) = P * (1 + 16 / 100) → 
  x = 10 / 3 := by
sorry

end book_price_change_l1155_115595


namespace triangle_side_length_l1155_115520

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x) + 2, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem triangle_side_length 
  (A B C : ℝ) 
  (h1 : f A = 4) 
  (h2 : 0 < A ∧ A < π) 
  (h3 : Real.sin A * 1 / 2 = Real.sqrt 3 / 2) :
  ∃ (a : ℝ), a^2 = 3 :=
sorry

end triangle_side_length_l1155_115520


namespace smallest_a_for_equation_l1155_115522

theorem smallest_a_for_equation : 
  ∀ a : ℕ, a ≥ 2 → 
  (∃ (p : ℕ) (b : ℕ), Prime p ∧ b ≥ 2 ∧ (a^p - a) / p = b^2) → 
  a ≥ 9 :=
sorry

end smallest_a_for_equation_l1155_115522


namespace alpha_value_l1155_115585

theorem alpha_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) 
  (h_min : ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → 1/m + 16/n ≤ 1/x + 16/y) 
  (h_curve : (m/5)^α = m/4) : α = 1/2 := by
  sorry

end alpha_value_l1155_115585


namespace talia_father_age_l1155_115533

-- Define Talia's current age
def talia_age : ℕ := 20 - 7

-- Define Talia's mom's current age
def mom_age : ℕ := 3 * talia_age

-- Define Talia's father's current age
def father_age : ℕ := mom_age - 3

-- Theorem statement
theorem talia_father_age : father_age = 36 := by
  sorry

end talia_father_age_l1155_115533


namespace max_prob_second_highest_l1155_115553

variable (p₁ p₂ p₃ : ℝ)

-- Define the conditions
axiom prob_order : 0 < p₁ ∧ p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ ≤ 1

-- Define the probability of winning two consecutive games for each scenario
def P_A := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

-- Theorem statement
theorem max_prob_second_highest :
  P_C p₁ p₂ p₃ > P_A p₁ p₂ p₃ ∧ P_C p₁ p₂ p₃ > P_B p₁ p₂ p₃ :=
sorry

end max_prob_second_highest_l1155_115553


namespace star_operation_value_l1155_115517

def star_operation (a b : ℚ) : ℚ := 1 / a + 1 / b

theorem star_operation_value (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  star_operation a b = 5 / 12 := by
  sorry

end star_operation_value_l1155_115517


namespace orangeade_price_day2_l1155_115580

/-- Represents the price and composition of orangeade on two consecutive days -/
structure Orangeade where
  orange_juice : ℝ
  water_day1 : ℝ
  water_day2 : ℝ
  price_day1 : ℝ
  price_day2 : ℝ

/-- Theorem stating the conditions and the result to be proven -/
theorem orangeade_price_day2 (o : Orangeade) 
  (h1 : o.orange_juice = o.water_day1)
  (h2 : o.water_day2 = 2 * o.water_day1)
  (h3 : o.price_day1 = 0.3)
  (h4 : (o.orange_juice + o.water_day1) * o.price_day1 = 
        (o.orange_juice + o.water_day2) * o.price_day2) :
  o.price_day2 = 0.2 := by
  sorry

#check orangeade_price_day2

end orangeade_price_day2_l1155_115580


namespace salary_change_calculation_salary_decrease_percentage_l1155_115505

/-- Given an initial salary increase followed by a decrease, 
    calculate the percentage of the decrease. -/
theorem salary_change_calculation (initial_increase : ℝ) (net_increase : ℝ) : ℝ :=
  let final_factor := 1 + net_increase / 100
  let increase_factor := 1 + initial_increase / 100
  100 * (1 - final_factor / increase_factor)

/-- The percentage decrease in salary after an initial 10% increase,
    resulting in a net 1% increase, is approximately 8.18%. -/
theorem salary_decrease_percentage : 
  ∃ ε > 0, |salary_change_calculation 10 1 - 8.18| < ε :=
sorry

end salary_change_calculation_salary_decrease_percentage_l1155_115505


namespace max_area_rectangle_142_perimeter_l1155_115500

/-- Represents a rectangle with integer side lengths. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle. -/
def Rectangle.area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem stating the maximum area of a rectangle with a perimeter of 142 feet. -/
theorem max_area_rectangle_142_perimeter :
  ∃ (r : Rectangle), r.perimeter = 142 ∧
    ∀ (s : Rectangle), s.perimeter = 142 → s.area ≤ r.area ∧
    r.area = 1260 := by
  sorry


end max_area_rectangle_142_perimeter_l1155_115500


namespace product_of_red_is_red_l1155_115572

-- Define the color type
inductive Color : Type
  | Red : Color
  | Blue : Color

-- Define the coloring function
def coloring : ℕ+ → Color := sorry

-- Define the conditions
axiom all_colored : ∀ n : ℕ+, (coloring n = Color.Red) ∨ (coloring n = Color.Blue)
axiom sum_different_colors : ∀ m n : ℕ+, coloring m ≠ coloring n → coloring (m + n) = Color.Blue
axiom product_different_colors : ∀ m n : ℕ+, coloring m ≠ coloring n → coloring (m * n) = Color.Red

-- State the theorem
theorem product_of_red_is_red :
  ∀ m n : ℕ+, coloring m = Color.Red → coloring n = Color.Red → coloring (m * n) = Color.Red :=
sorry

end product_of_red_is_red_l1155_115572


namespace correct_regression_sequence_l1155_115554

/-- A step in the linear regression analysis process -/
inductive RegressionStep
  | predict : RegressionStep
  | collectData : RegressionStep
  | deriveEquation : RegressionStep
  | plotScatter : RegressionStep

/-- The correct sequence of steps in linear regression analysis -/
def correctSequence : List RegressionStep :=
  [RegressionStep.collectData, RegressionStep.plotScatter, 
   RegressionStep.deriveEquation, RegressionStep.predict]

/-- Theorem stating that the given sequence is the correct order of steps -/
theorem correct_regression_sequence :
  correctSequence = [RegressionStep.collectData, RegressionStep.plotScatter, 
                     RegressionStep.deriveEquation, RegressionStep.predict] := by
  sorry

end correct_regression_sequence_l1155_115554


namespace negation_equivalence_l1155_115536

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ > 3 - x₀) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≤ 3 - x) := by
  sorry

end negation_equivalence_l1155_115536


namespace seventh_flip_probability_l1155_115575

/-- A fair coin is a coin where the probability of getting heads is 1/2. -/
def fair_coin (p : ℝ → ℝ) : Prop := p 1 = 1/2

/-- A sequence of coin flips is independent if the probability of any outcome
    is not affected by the previous flips. -/
def independent_flips (p : ℕ → ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∀ x : ℝ, p n x = p 0 x

/-- The probability of getting heads on the seventh flip of a fair coin is 1/2,
    regardless of the outcomes of the previous six flips. -/
theorem seventh_flip_probability (p : ℕ → ℝ → ℝ) :
  fair_coin (p 0) →
  independent_flips p →
  p 6 1 = 1/2 :=
by
  sorry

end seventh_flip_probability_l1155_115575


namespace modulo_residue_problem_l1155_115504

theorem modulo_residue_problem : (325 + 3 * 66 + 8 * 187 + 6 * 23) % 11 = 1 := by
  sorry

end modulo_residue_problem_l1155_115504


namespace solve_inequality_when_a_is_5_range_of_a_for_always_positive_l1155_115518

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 6

-- Statement for part I
theorem solve_inequality_when_a_is_5 :
  {x : ℝ | f 5 x < 0} = {x : ℝ | -3 < x ∧ x < -2} := by sorry

-- Statement for part II
theorem range_of_a_for_always_positive :
  ∀ a : ℝ, (∀ x : ℝ, f a x > 0) ↔ a ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) := by sorry

end solve_inequality_when_a_is_5_range_of_a_for_always_positive_l1155_115518


namespace hilt_fountain_distance_l1155_115564

/-- The total distance Mrs. Hilt walks to the water fountain -/
def total_distance (desk_to_fountain : ℕ) (num_trips : ℕ) : ℕ :=
  2 * desk_to_fountain * num_trips

/-- Theorem: Mrs. Hilt walks 240 feet given the problem conditions -/
theorem hilt_fountain_distance :
  total_distance 30 4 = 240 :=
by sorry

end hilt_fountain_distance_l1155_115564


namespace coloring_book_problem_l1155_115528

theorem coloring_book_problem (book1 : Nat) (book2 : Nat) (colored : Nat) : 
  book1 = 23 → book2 = 32 → colored = 44 → book1 + book2 - colored = 11 := by
  sorry

end coloring_book_problem_l1155_115528


namespace number_of_carnations_solve_carnation_problem_l1155_115501

/-- Proves the number of carnations given the problem conditions --/
theorem number_of_carnations : ℕ → Prop :=
  fun c =>
    let vase_capacity : ℕ := 9
    let num_roses : ℕ := 23
    let num_vases : ℕ := 3
    (c + num_roses = num_vases * vase_capacity) → c = 4

/-- The theorem statement --/
theorem solve_carnation_problem : number_of_carnations 4 := by
  sorry

end number_of_carnations_solve_carnation_problem_l1155_115501


namespace min_value_theorem_l1155_115568

/-- The function f(x) = x|x - a| has a minimum value of 2 on the interval [1, 2] when a = 3 -/
theorem min_value_theorem (a : ℝ) (h1 : a > 0) :
  (∀ x ∈ Set.Icc 1 2, x * |x - a| ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 1 2, x * |x - a| = 2) →
  a = 3 := by
  sorry

end min_value_theorem_l1155_115568


namespace symmetric_about_one_empty_solution_set_implies_a_leq_one_at_most_one_intersection_l1155_115559

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define evenness for a function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Proposition 2
theorem symmetric_about_one (h : is_even (fun x ↦ f (x + 1))) :
  ∀ x, f (1 + x) = f (1 - x) := by sorry

-- Proposition 3
theorem empty_solution_set_implies_a_leq_one (a : ℝ) :
  (∀ x, |x - 4| + |x - 3| ≥ a) → a ≤ 1 := by sorry

-- Proposition 4
theorem at_most_one_intersection (a : ℝ) :
  ∃! y, f a = y := by sorry

end symmetric_about_one_empty_solution_set_implies_a_leq_one_at_most_one_intersection_l1155_115559


namespace second_term_base_l1155_115511

theorem second_term_base (x y : ℕ) (base : ℝ) : 
  3^x * base^y = 19683 → x - y = 9 → x = 9 → base = 1 :=
by
  sorry

end second_term_base_l1155_115511


namespace F_opposite_A_l1155_115574

/-- Represents a face of a cube --/
inductive Face : Type
| A | B | C | D | E | F

/-- Represents a cube net that can be folded into a cube --/
structure CubeNet where
  faces : List Face
  can_fold : Bool

/-- Represents a folded cube --/
structure Cube where
  net : CubeNet
  bottom : Face

/-- Defines the opposite face relation in a cube --/
def opposite_face (c : Cube) (f1 f2 : Face) : Prop :=
  f1 ≠ f2 ∧ ∀ (f : Face), f ≠ f1 → f ≠ f2 → (f ∈ c.net.faces)

/-- Theorem: In a cube formed from a net where face F is the bottom, face F is opposite to face A --/
theorem F_opposite_A (c : Cube) (h : c.bottom = Face.F) : opposite_face c Face.A Face.F :=
sorry

end F_opposite_A_l1155_115574


namespace last_score_entered_last_score_is_95_l1155_115506

def scores : List ℕ := [75, 81, 85, 87, 95]

def is_integer_average (subset : List ℕ) : Prop :=
  ∃ n : ℕ, n * subset.length = subset.sum

theorem last_score_entered (last : ℕ) : Prop :=
  last ∈ scores ∧
  ∀ subset : List ℕ, subset ⊆ scores → last ∈ subset →
    is_integer_average subset

theorem last_score_is_95 : 
  ∃ last : ℕ, last_score_entered last ∧ last = 95 := by
  sorry

end last_score_entered_last_score_is_95_l1155_115506


namespace sqrt_equation_solution_l1155_115525

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 10 → x = 105 := by
  sorry

end sqrt_equation_solution_l1155_115525


namespace syrup_box_cost_l1155_115592

/-- Represents the cost of syrup boxes for a convenience store -/
def SyrupCost (total_soda : ℕ) (soda_per_box : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / (total_soda / soda_per_box)

/-- Theorem: The cost per box of syrup is $40 -/
theorem syrup_box_cost :
  SyrupCost 180 30 240 = 40 := by
  sorry

end syrup_box_cost_l1155_115592


namespace concert_duration_is_80_minutes_l1155_115521

/-- Calculates the total duration of a concert given the number of songs, 
    duration of regular songs, duration of the special song, and intermission time. -/
def concertDuration (numSongs : ℕ) (regularSongDuration : ℕ) (specialSongDuration : ℕ) (intermissionTime : ℕ) : ℕ :=
  (numSongs - 1) * regularSongDuration + specialSongDuration + intermissionTime

/-- Proves that the concert duration is 80 minutes given the specified conditions. -/
theorem concert_duration_is_80_minutes :
  concertDuration 13 5 10 10 = 80 := by
  sorry

end concert_duration_is_80_minutes_l1155_115521


namespace weather_conditions_on_july_15_l1155_115512

/-- Represents the weather conditions at the beach --/
structure WeatherCondition where
  temperature : ℝ
  sunny : Bool
  windSpeed : ℝ

/-- Predicate to determine if the beach is crowded based on weather conditions --/
def isCrowded (w : WeatherCondition) : Prop :=
  w.temperature ≥ 85 ∧ w.sunny ∧ w.windSpeed < 10

/-- Theorem: Given that the beach is not crowded on July 15, prove that the weather conditions
    must satisfy: temperature < 85°F or not sunny or wind speed ≥ 10 mph --/
theorem weather_conditions_on_july_15 (w : WeatherCondition) 
  (h : ¬isCrowded w) : 
  w.temperature < 85 ∨ ¬w.sunny ∨ w.windSpeed ≥ 10 := by
  sorry


end weather_conditions_on_july_15_l1155_115512


namespace line_through_circle_center_l1155_115539

/-- A line intersecting a circle -/
structure LineIntersectingCircle where
  /-- The slope of the line -/
  k : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The line intersects the circle at two points -/
  intersects : True
  /-- The distance between the intersection points -/
  chord_length : ℝ

/-- The theorem stating the value of k for a specific configuration -/
theorem line_through_circle_center (config : LineIntersectingCircle)
    (h1 : config.b = 2)
    (h2 : config.center = (1, 1))
    (h3 : config.radius = Real.sqrt 2)
    (h4 : config.chord_length = 2 * Real.sqrt 2) :
    config.k = -1 := by
  sorry

end line_through_circle_center_l1155_115539


namespace rockham_soccer_league_members_l1155_115562

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 4

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 5

/-- The total cost for all members in dollars -/
def total_cost : ℕ := 2366

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 2

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- Theorem: The number of members in the Rockham Soccer League is 91 -/
theorem rockham_soccer_league_members : 
  (total_cost / (socks_per_member * sock_cost + 
                 tshirts_per_member * (sock_cost + tshirt_additional_cost))) = 91 := by
  sorry

end rockham_soccer_league_members_l1155_115562


namespace equation_solution_l1155_115565

theorem equation_solution (x : ℝ) : 
  x ≠ 2 → ((4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 5) → x = -2 := by
sorry

end equation_solution_l1155_115565


namespace quadratic_even_iff_a_eq_zero_l1155_115576

/-- A quadratic function f(x) = x^2 + ax + b is even if and only if a = 0 -/
theorem quadratic_even_iff_a_eq_zero (a b : ℝ) :
  (∀ x : ℝ, x^2 + a*x + b = (-x)^2 + a*(-x) + b) ↔ a = 0 := by
  sorry

end quadratic_even_iff_a_eq_zero_l1155_115576


namespace picking_black_is_random_event_l1155_115546

/-- Represents a ball in the box -/
inductive Ball
| White
| Black

/-- Represents the box containing the balls -/
structure Box where
  white_balls : ℕ
  black_balls : ℕ

/-- Defines what a random event is -/
def is_random_event (box : Box) (pick : Ball → Prop) : Prop :=
  (∃ b : Ball, pick b) ∧ 
  (∃ b : Ball, ¬ pick b) ∧ 
  (box.white_balls + box.black_balls > 0)

/-- The main theorem to prove -/
theorem picking_black_is_random_event (box : Box) 
  (h1 : box.white_balls = 1) 
  (h2 : box.black_balls = 200) : 
  is_random_event box (λ b => b = Ball.Black) := by
  sorry


end picking_black_is_random_event_l1155_115546


namespace cosine_value_proof_l1155_115590

theorem cosine_value_proof (α : ℝ) (h : Real.sin (π/6 - α) = 4/5) : 
  Real.cos (π/3 + α) = 4/5 := by
  sorry

end cosine_value_proof_l1155_115590


namespace working_mom_time_allocation_l1155_115508

theorem working_mom_time_allocation :
  let total_hours_in_day : ℝ := 24
  let work_hours : ℝ := 8
  let daughter_care_hours : ℝ := 2.25
  let household_chores_hours : ℝ := 3.25
  let total_activity_hours : ℝ := work_hours + daughter_care_hours + household_chores_hours
  let percentage_of_day : ℝ := (total_activity_hours / total_hours_in_day) * 100
  percentage_of_day = 56.25 := by
sorry

end working_mom_time_allocation_l1155_115508


namespace triangle_tangent_l1155_115566

theorem triangle_tangent (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : Real.tan A = 1/2) (h3 : Real.cos B = (3 * Real.sqrt 10) / 10) : 
  Real.tan C = -1 := by
  sorry

end triangle_tangent_l1155_115566


namespace point_translation_to_origin_l1155_115581

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point by a given vector -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem point_translation_to_origin (A : Point) :
  translate A 3 2 = ⟨0, 0⟩ → A = ⟨-3, -2⟩ := by
  sorry

end point_translation_to_origin_l1155_115581


namespace mothers_age_problem_l1155_115532

theorem mothers_age_problem (x : ℕ) : 
  x + 3 * x = 40 → x = 10 := by sorry

end mothers_age_problem_l1155_115532


namespace jeff_donation_proof_l1155_115515

/-- The percentage of pencils Jeff donated -/
def jeff_donation_percentage : ℝ := 0.3

theorem jeff_donation_proof :
  let jeff_initial : ℕ := 300
  let vicki_initial : ℕ := 2 * jeff_initial
  let vicki_donation : ℝ := 3/4 * vicki_initial
  let total_remaining : ℕ := 360
  (jeff_initial - jeff_initial * jeff_donation_percentage) +
    (vicki_initial - vicki_donation) = total_remaining :=
by sorry

end jeff_donation_proof_l1155_115515


namespace hot_dog_problem_l1155_115552

theorem hot_dog_problem :
  let hot_dogs := 12
  let hot_dog_buns := 9
  let mustard := 18
  let ketchup := 24
  Nat.lcm (Nat.lcm (Nat.lcm hot_dogs hot_dog_buns) mustard) ketchup = 72 := by
  sorry

end hot_dog_problem_l1155_115552


namespace two_intersection_points_l1155_115596

/-- A line in the plane represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of intersection points between at least two of three given lines -/
def intersection_count (l1 l2 l3 : Line) : ℕ :=
  sorry

/-- The three lines from the problem -/
def line1 : Line := { a := -2, b := 3, c := 1 }
def line2 : Line := { a := 1, b := 2, c := 2 }
def line3 : Line := { a := 4, b := -6, c := 5 }

theorem two_intersection_points : intersection_count line1 line2 line3 = 2 := by
  sorry

end two_intersection_points_l1155_115596
