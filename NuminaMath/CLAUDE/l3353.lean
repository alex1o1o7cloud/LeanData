import Mathlib

namespace set_operations_l3353_335373

def U := Set ℝ
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem set_operations :
  (A ∩ B = {x : ℝ | 2 < x ∧ x < 3}) ∧
  (Set.compl B = {x : ℝ | x ≤ 1 ∨ x ≥ 3}) := by
  sorry

end set_operations_l3353_335373


namespace percent_of_x_is_y_l3353_335313

theorem percent_of_x_is_y (x y : ℝ) (h : 0.7 * (x - y) = 0.3 * (x + y)) : y = 0.4 * x := by
  sorry

end percent_of_x_is_y_l3353_335313


namespace gcd_seven_eight_factorial_l3353_335352

theorem gcd_seven_eight_factorial : 
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcd_seven_eight_factorial_l3353_335352


namespace spade_theorem_l3353_335376

-- Define the binary operation ◊
def spade (A B : ℚ) : ℚ := 4 * A + 3 * B - 2

-- Theorem statement
theorem spade_theorem (A : ℚ) : spade A 7 = 40 → A = 21 / 4 := by
  sorry

end spade_theorem_l3353_335376


namespace first_applicant_better_by_850_l3353_335317

/-- Represents an applicant for a job position -/
structure Applicant where
  salary : ℕ
  revenue : ℕ
  training_months : ℕ
  training_cost_per_month : ℕ
  hiring_bonus_percent : ℕ

/-- Calculates the net gain for the company from an applicant -/
def net_gain (a : Applicant) : ℤ :=
  a.revenue - a.salary - (a.training_months * a.training_cost_per_month) - (a.salary * a.hiring_bonus_percent / 100)

theorem first_applicant_better_by_850 :
  let first := Applicant.mk 42000 93000 3 1200 0
  let second := Applicant.mk 45000 92000 0 0 1
  net_gain first - net_gain second = 850 := by sorry

end first_applicant_better_by_850_l3353_335317


namespace line_through_points_l3353_335367

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a line passing through distinct vectors a and b, 
    if k*a + (5/6)*b lies on the same line, then k = 5/6 -/
theorem line_through_points (a b : V) (h : a ≠ b) :
  ∃ (t : ℝ), k • a + (5/6) • b = a + t • (b - a) → k = 5/6 :=
sorry

end line_through_points_l3353_335367


namespace sam_fish_count_l3353_335359

theorem sam_fish_count (harry joe sam : ℕ) 
  (harry_joe : harry = 4 * joe)
  (joe_sam : joe = 8 * sam)
  (harry_count : harry = 224) : 
  sam = 7 := by
sorry

end sam_fish_count_l3353_335359


namespace circumscribed_quadrilateral_sides_l3353_335369

/-- A circumscribed quadrilateral with perimeter 24 and three consecutive sides in ratio 1:2:3 has sides 3, 6, 9, and 6. -/
theorem circumscribed_quadrilateral_sides (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- sides are positive
  a + b + c + d = 24 →  -- perimeter is 24
  a + c = b + d →  -- circumscribed property
  ∃ (x : ℝ), a = x ∧ b = 2*x ∧ c = 3*x →  -- consecutive sides in ratio 1:2:3
  a = 3 ∧ b = 6 ∧ c = 9 ∧ d = 6 := by
  sorry

#check circumscribed_quadrilateral_sides

end circumscribed_quadrilateral_sides_l3353_335369


namespace f_max_value_l3353_335393

/-- The quadratic function f(x) = -3x^2 + 6x + 4 --/
def f (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 4

/-- The maximum value of f(x) over all real numbers x --/
def max_value : ℝ := 7

/-- Theorem stating that the maximum value of f(x) is 7 --/
theorem f_max_value : ∀ x : ℝ, f x ≤ max_value := by sorry

end f_max_value_l3353_335393


namespace other_endpoint_of_line_segment_l3353_335301

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -3), prove that the other endpoint is (-1, 5) -/
theorem other_endpoint_of_line_segment (x₂ y₂ : ℚ) : 
  (3 = (7 + x₂) / 2) ∧ (1 = (-3 + y₂) / 2) → (x₂ = -1 ∧ y₂ = 5) := by
sorry

end other_endpoint_of_line_segment_l3353_335301


namespace practicing_to_writing_ratio_l3353_335308

/-- Represents the time spent on different activities for a speech --/
structure SpeechTime where
  outlining : ℕ
  writing : ℕ
  practicing : ℕ
  total : ℕ

/-- Defines the conditions of Javier's speech preparation --/
def javierSpeechTime : SpeechTime where
  outlining := 30
  writing := 30 + 28
  practicing := 117 - (30 + 58)
  total := 117

/-- Theorem stating the ratio of practicing to writing time --/
theorem practicing_to_writing_ratio :
  (javierSpeechTime.practicing : ℚ) / javierSpeechTime.writing = 1 / 2 := by
  sorry

end practicing_to_writing_ratio_l3353_335308


namespace polynomial_root_implies_coefficients_l3353_335343

theorem polynomial_root_implies_coefficients : 
  ∀ (p q : ℝ), 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - 3 * Complex.I : ℂ) ^ 3 + p * (2 - 3 * Complex.I : ℂ) ^ 2 - 5 * (2 - 3 * Complex.I : ℂ) + q = 0 →
  p = 1/2 ∧ q = 117/2 := by
  sorry

end polynomial_root_implies_coefficients_l3353_335343


namespace graph_shift_l3353_335337

/-- Given a function f and real numbers a and b, 
    the graph of y = f(x - a) + b is obtained by shifting 
    the graph of y = f(x) a units right and b units up. -/
theorem graph_shift (f : ℝ → ℝ) (a b : ℝ) :
  ∀ x y, y = f (x - a) + b ↔ y - b = f (x - a) :=
by sorry

end graph_shift_l3353_335337


namespace sufficient_not_necessary_condition_l3353_335332

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem sufficient_not_necessary_condition
  (a₁ q : ℝ) :
  (a₁ < 0 ∧ 0 < q ∧ q < 1 →
    ∀ n : ℕ, n > 0 → geometric_sequence a₁ q (n + 1) > geometric_sequence a₁ q n) ∧
  (∃ a₁' q' : ℝ, (∀ n : ℕ, n > 0 → geometric_sequence a₁' q' (n + 1) > geometric_sequence a₁' q' n) ∧
    ¬(a₁' < 0 ∧ 0 < q' ∧ q' < 1)) :=
by sorry

end sufficient_not_necessary_condition_l3353_335332


namespace magician_earnings_l3353_335329

/-- Represents the sales and conditions of the magician's card deck business --/
structure MagicianSales where
  initialPrice : ℝ
  initialStock : ℕ
  finalStock : ℕ
  promotionPrice : ℝ
  initialExchangeRate : ℝ
  changedExchangeRate : ℝ
  foreignCustomersBulk : ℕ
  foreignCustomersSingle : ℕ
  domesticCustomers : ℕ

/-- Calculates the total earnings of the magician in dollars --/
def calculateEarnings (sales : MagicianSales) : ℝ :=
  sorry

/-- Theorem stating that the magician's earnings equal 11 dollars --/
theorem magician_earnings (sales : MagicianSales) 
  (h1 : sales.initialPrice = 2)
  (h2 : sales.initialStock = 5)
  (h3 : sales.finalStock = 3)
  (h4 : sales.promotionPrice = 3)
  (h5 : sales.initialExchangeRate = 1)
  (h6 : sales.changedExchangeRate = 1.5)
  (h7 : sales.foreignCustomersBulk = 2)
  (h8 : sales.foreignCustomersSingle = 1)
  (h9 : sales.domesticCustomers = 2) :
  calculateEarnings sales = 11 := by
  sorry

end magician_earnings_l3353_335329


namespace smallest_odd_with_five_prime_factors_l3353_335398

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def has_exactly_five_prime_factors (n : ℕ) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧ is_prime p₅ ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧ p₄ < p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅ ∧
    ∀ (q : ℕ), is_prime q → q ∣ n → (q = p₁ ∨ q = p₂ ∨ q = p₃ ∨ q = p₄ ∨ q = p₅)

theorem smallest_odd_with_five_prime_factors :
  (∀ n : ℕ, n < 15015 → ¬(n % 2 = 1 ∧ has_exactly_five_prime_factors n)) ∧
  15015 % 2 = 1 ∧ has_exactly_five_prime_factors 15015 := by
  sorry

end smallest_odd_with_five_prime_factors_l3353_335398


namespace work_completion_proof_l3353_335366

/-- The number of days x needs to finish the work alone -/
def x_days : ℕ := 20

/-- The number of days y worked before leaving -/
def y_worked : ℕ := 12

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℕ := 5

/-- The number of days y needs to finish the work alone -/
def y_days : ℕ := 16

theorem work_completion_proof :
  (1 : ℚ) / x_days * x_remaining + (1 : ℚ) / y_days * y_worked = 1 :=
sorry

end work_completion_proof_l3353_335366


namespace quadratic_equals_binomial_square_l3353_335341

theorem quadratic_equals_binomial_square (d : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 60*x + d = (a*x + b)^2) → d = 900 := by
sorry

end quadratic_equals_binomial_square_l3353_335341


namespace polynomial_expansions_l3353_335327

theorem polynomial_expansions (x y : ℝ) : 
  ((x - 3) * (x^2 + 4) = x^3 - 3*x^2 + 4*x - 12) ∧ 
  ((3*x^2 - y) * (x + 2*y) = 3*x^3 + 6*y*x^2 - x*y - 2*y^2) := by
  sorry

end polynomial_expansions_l3353_335327


namespace inequality_holds_iff_l3353_335390

theorem inequality_holds_iff (x : ℝ) :
  (∀ y : ℝ, y^2 - (5^x - 1)*(y - 1) > 0) ↔ (0 < x ∧ x < 1) :=
sorry

end inequality_holds_iff_l3353_335390


namespace part_to_third_ratio_l3353_335381

theorem part_to_third_ratio (N P : ℝ) (h1 : (1/4) * (1/3) * P = 14) (h2 : 0.40 * N = 168) :
  P / ((1/3) * N) = 6/5 := by
  sorry

end part_to_third_ratio_l3353_335381


namespace dice_sum_probabilities_l3353_335340

/-- The probability of rolling a sum of 15 with 3 dice -/
def p_sum_15 : ℚ := 10 / 216

/-- The probability of rolling a sum of at least 15 with 3 dice -/
def p_sum_at_least_15 : ℚ := 20 / 216

/-- The minimum number of trials to roll a sum of 15 exactly once with probability > 1/2 -/
def min_trials_sum_15 : ℕ := 15

/-- The minimum number of trials to roll a sum of at least 15 exactly once with probability > 1/2 -/
def min_trials_sum_at_least_15 : ℕ := 8

theorem dice_sum_probabilities :
  (1 - (1 - p_sum_15) ^ min_trials_sum_15 > 1/2) ∧
  (∀ n : ℕ, n < min_trials_sum_15 → 1 - (1 - p_sum_15) ^ n ≤ 1/2) ∧
  (1 - (1 - p_sum_at_least_15) ^ min_trials_sum_at_least_15 > 1/2) ∧
  (∀ n : ℕ, n < min_trials_sum_at_least_15 → 1 - (1 - p_sum_at_least_15) ^ n ≤ 1/2) := by
  sorry

end dice_sum_probabilities_l3353_335340


namespace line_relationships_l3353_335380

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation for planes and lines
variable (perp : Plane → Plane → Prop)
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_relationships
  (α β : Plane) (a b l : Line)
  (h1 : intersect α β = l)
  (h2 : subset a α)
  (h3 : subset b β)
  (h4 : ¬ perp α β)
  (h5 : ¬ perp_line a l)
  (h6 : ¬ perp_line b l) :
  (∃ (a' b' : Line), parallel a' b' ∧ a' = a ∧ b' = b) ∧
  (∃ (a'' b'' : Line), perp_line a'' b'' ∧ a'' = a ∧ b'' = b) :=
sorry

end line_relationships_l3353_335380


namespace roots_equation_sum_l3353_335322

theorem roots_equation_sum (a b : ℝ) : 
  a^2 - 6*a + 8 = 0 → b^2 - 6*b + 8 = 0 → a^4 + b^4 + a^3*b + a*b^3 = 432 := by
  sorry

end roots_equation_sum_l3353_335322


namespace rectangle_area_l3353_335349

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 49 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 147 := by
sorry

end rectangle_area_l3353_335349


namespace slope_range_l3353_335387

-- Define the line l passing through point P(-2, 2) with slope k
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y - 2 = k * (x + 2)}

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {(x, y) | x^2 + y^2 + 12*x + 35 = 0}

-- Define the condition that circles with centers on l and radius 1 have no common points with C
def no_common_points (k : ℝ) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ line_l k →
    ∀ (a b : ℝ), (a - x)^2 + (b - y)^2 ≤ 1 →
      (a, b) ∉ circle_C

-- State the theorem
theorem slope_range :
  ∀ k : ℝ, no_common_points k →
    k < 0 ∨ k > 4/3 :=
sorry

end slope_range_l3353_335387


namespace tv_cost_l3353_335389

theorem tv_cost (total_budget : ℕ) (computer_cost : ℕ) (fridge_extra_cost : ℕ) :
  total_budget = 1600 →
  computer_cost = 250 →
  fridge_extra_cost = 500 →
  ∃ tv_cost : ℕ, tv_cost = 600 ∧ 
    tv_cost + (computer_cost + fridge_extra_cost) + computer_cost = total_budget :=
by sorry

end tv_cost_l3353_335389


namespace f_monotone_increasing_range_l3353_335374

/-- The function f(x) defined on the interval [0,1] -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a - 1) * x + 3

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*a*x - (2*a - 1)

/-- Theorem stating the range of a for which f(x) is monotonically increasing on [0,1] -/
theorem f_monotone_increasing_range :
  {a : ℝ | ∀ x ∈ Set.Icc 0 1, f_derivative a x ≥ 0} = Set.Iic (1/2) :=
sorry

end f_monotone_increasing_range_l3353_335374


namespace billy_caught_three_fish_l3353_335363

/-- Represents the number of fish caught by each family member and other relevant information --/
structure FishingTrip where
  ben_fish : ℕ
  judy_fish : ℕ
  jim_fish : ℕ
  susie_fish : ℕ
  thrown_back : ℕ
  total_filets : ℕ
  filets_per_fish : ℕ

/-- Calculates the number of fish Billy caught given the fishing trip information --/
def billy_fish_count (trip : FishingTrip) : ℕ :=
  let total_kept := trip.total_filets / trip.filets_per_fish
  let total_caught := total_kept + trip.thrown_back
  total_caught - trip.ben_fish - trip.judy_fish - trip.jim_fish - trip.susie_fish

/-- Theorem stating that Billy caught 3 fish given the specific conditions of the fishing trip --/
theorem billy_caught_three_fish (trip : FishingTrip) 
  (h1 : trip.ben_fish = 4)
  (h2 : trip.judy_fish = 1)
  (h3 : trip.jim_fish = 2)
  (h4 : trip.susie_fish = 5)
  (h5 : trip.thrown_back = 3)
  (h6 : trip.total_filets = 24)
  (h7 : trip.filets_per_fish = 2) :
  billy_fish_count trip = 3 := by
  sorry

#eval billy_fish_count ⟨4, 1, 2, 5, 3, 24, 2⟩

end billy_caught_three_fish_l3353_335363


namespace gender_related_to_reading_l3353_335385

-- Define the survey data
def survey_data : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![30, 20],
    ![40, 10]]

-- Define the total number of observations
def N : ℕ := 100

-- Define the formula for calculating k^2
def calculate_k_squared (data : Matrix (Fin 2) (Fin 2) ℕ) (total : ℕ) : ℚ :=
  let O11 := data 0 0
  let O12 := data 0 1
  let O21 := data 1 0
  let O22 := data 1 1
  (total * (O11 * O22 - O12 * O21)^2 : ℚ) / 
  ((O11 + O12) * (O21 + O22) * (O11 + O21) * (O12 + O22) : ℚ)

-- Define the critical values
def critical_value_005 : ℚ := 3841 / 1000
def critical_value_001 : ℚ := 6635 / 1000

-- State the theorem
theorem gender_related_to_reading :
  let k_squared := calculate_k_squared survey_data N
  k_squared > critical_value_005 ∧ k_squared < critical_value_001 := by
  sorry

end gender_related_to_reading_l3353_335385


namespace johns_height_l3353_335312

theorem johns_height (building_height : ℝ) (building_shadow : ℝ) (johns_shadow_inches : ℝ) :
  building_height = 60 →
  building_shadow = 20 →
  johns_shadow_inches = 18 →
  ∃ (johns_height : ℝ), johns_height = 4.5 := by
  sorry

end johns_height_l3353_335312


namespace vegetable_baskets_weight_l3353_335333

def num_baskets : ℕ := 5
def standard_weight : ℕ := 50
def excess_deficiency : List ℤ := [3, -6, -4, 2, -1]

theorem vegetable_baskets_weight :
  (List.sum excess_deficiency = -6) ∧
  (num_baskets * standard_weight + List.sum excess_deficiency = 244) := by
sorry

end vegetable_baskets_weight_l3353_335333


namespace three_greater_than_sqrt_seven_l3353_335346

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end three_greater_than_sqrt_seven_l3353_335346


namespace polynomial_remainder_l3353_335344

theorem polynomial_remainder (p : ℝ → ℝ) (h1 : p 2 = 6) (h2 : p 4 = 10) :
  ∃ (q r : ℝ → ℝ), (∀ x, p x = q x * ((x - 2) * (x - 4)) + r x) ∧
                    (∃ a b : ℝ, ∀ x, r x = a * x + b) ∧
                    (∀ x, r x = 2 * x + 2) :=
by sorry

end polynomial_remainder_l3353_335344


namespace mrs_hilt_fountain_trips_l3353_335307

/-- Calculates the number of trips to the water fountain given the distance to the fountain and total distance walked. -/
def trips_to_fountain (distance_to_fountain : ℕ) (total_distance_walked : ℕ) : ℕ :=
  total_distance_walked / (2 * distance_to_fountain)

/-- Theorem stating that given a distance of 30 feet to the fountain and 120 feet walked, the number of trips is 2. -/
theorem mrs_hilt_fountain_trips :
  trips_to_fountain 30 120 = 2 := by
  sorry


end mrs_hilt_fountain_trips_l3353_335307


namespace taxi_fare_calculation_l3353_335353

/-- Represents the fare structure of a taxi service -/
structure TaxiFare where
  base_fare : ℝ
  per_mile_charge : ℝ

/-- Calculates the total fare for a given distance -/
def total_fare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.base_fare + tf.per_mile_charge * distance

theorem taxi_fare_calculation (tf : TaxiFare) 
  (h1 : tf.base_fare = 40)
  (h2 : total_fare tf 80 = 200) :
  total_fare tf 100 = 240 := by
sorry

end taxi_fare_calculation_l3353_335353


namespace inequality_proof_l3353_335303

theorem inequality_proof (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) 
  (h_sum : x * y + y * z + z * x = 1) : 
  x * y * z * (x + y + z) ≤ 1 / 3 := by
sorry

end inequality_proof_l3353_335303


namespace five_primes_in_valid_set_l3353_335328

/-- The set of digits to choose from -/
def digit_set : Finset Nat := {3, 5, 7, 8}

/-- Function to form a two-digit number from two digits -/
def form_number (tens units : Nat) : Nat := 10 * tens + units

/-- Predicate to check if a number is formed from two different digits in the set -/
def is_valid_number (n : Nat) : Prop :=
  ∃ (tens units : Nat), tens ∈ digit_set ∧ units ∈ digit_set ∧ tens ≠ units ∧ n = form_number tens units

/-- The set of all valid two-digit numbers formed from the digit set -/
def valid_numbers : Finset Nat := sorry

/-- The theorem stating that there are exactly 5 prime numbers in the valid set -/
theorem five_primes_in_valid_set : (valid_numbers.filter Nat.Prime).card = 5 := by sorry

end five_primes_in_valid_set_l3353_335328


namespace second_discount_percentage_l3353_335351

theorem second_discount_percentage (initial_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  initial_price = 150 →
  first_discount = 20 →
  final_price = 108 →
  ∃ (second_discount : ℝ),
    second_discount = 10 ∧
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end second_discount_percentage_l3353_335351


namespace cuboid_base_area_l3353_335347

theorem cuboid_base_area (volume : ℝ) (height : ℝ) (base_area : ℝ) :
  volume = 144 →
  height = 8 →
  volume = base_area * height →
  base_area = 18 :=
by
  sorry

end cuboid_base_area_l3353_335347


namespace donkey_mule_bags_l3353_335300

theorem donkey_mule_bags (x y : ℕ) (hx : x > 0) (hy : y > 0) : 
  (y + 1 = 2 * (x - 1) ∧ y - 1 = x + 1) ↔ 
  (∃ (d m : ℕ), d = x ∧ m = y ∧ 
    (m + 1 = 2 * (d - 1)) ∧ 
    (m - 1 = d + 1)) :=
by sorry

end donkey_mule_bags_l3353_335300


namespace arithmetic_sequence_equivalence_l3353_335311

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def is_arithmetic_seq (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

theorem arithmetic_sequence_equivalence
  (a b c : ℕ → ℝ)
  (h1 : ∀ n : ℕ, b n = a n - a (n + 2))
  (h2 : ∀ n : ℕ, c n = a n + 2 * a (n + 1) + 3 * a (n + 2)) :
  is_arithmetic_seq a ↔ is_arithmetic_seq c ∧ (∀ n : ℕ, b n ≤ b (n + 1)) :=
by sorry

end arithmetic_sequence_equivalence_l3353_335311


namespace function_simplification_l3353_335302

/-- Given f(x) = (2x + 1)^5 - 5(2x + 1)^4 + 10(2x + 1)^3 - 10(2x + 1)^2 + 5(2x + 1) - 1,
    prove that f(x) = 32x^5 for all real x -/
theorem function_simplification (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1
  f x = 32*x^5 := by
  sorry

end function_simplification_l3353_335302


namespace decimal_value_changes_when_removing_zeros_l3353_335365

theorem decimal_value_changes_when_removing_zeros : 7.0800 ≠ 7.8 := by sorry

end decimal_value_changes_when_removing_zeros_l3353_335365


namespace gustran_nails_cost_l3353_335391

structure Salon where
  name : String
  haircut : ℕ
  facial : ℕ
  nails : ℕ

def gustran_salon : Salon := {
  name := "Gustran Salon"
  haircut := 45
  facial := 22
  nails := 0  -- Unknown, to be proved
}

def barbaras_shop : Salon := {
  name := "Barbara's Shop"
  haircut := 30
  facial := 28
  nails := 40
}

def fancy_salon : Salon := {
  name := "The Fancy Salon"
  haircut := 34
  facial := 30
  nails := 20
}

def total_cost (s : Salon) : ℕ := s.haircut + s.facial + s.nails

theorem gustran_nails_cost :
  ∃ (x : ℕ), 
    gustran_salon.nails = x ∧ 
    total_cost gustran_salon = 84 ∧
    total_cost barbaras_shop ≥ 84 ∧
    total_cost fancy_salon = 84 ∧
    x = 17 := by sorry

end gustran_nails_cost_l3353_335391


namespace survey_preferences_l3353_335368

theorem survey_preferences (total : ℕ) (mac_pref : ℕ) (windows_pref : ℕ) : 
  total = 210 →
  mac_pref = 60 →
  windows_pref = 40 →
  ∃ (no_pref : ℕ),
    no_pref = total - (mac_pref + windows_pref + (mac_pref / 3)) ∧
    no_pref = 90 :=
by sorry

end survey_preferences_l3353_335368


namespace rap_song_requests_l3353_335361

/-- Represents the number of song requests for different genres --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rap song requests --/
theorem rap_song_requests (req : SongRequests) : req.rap = 2 :=
  by
  have h1 : req.total = 30 := by sorry
  have h2 : req.electropop = req.total / 2 := by sorry
  have h3 : req.dance = req.electropop / 3 := by sorry
  have h4 : req.rock = 5 := by sorry
  have h5 : req.oldies = req.rock - 3 := by sorry
  have h6 : req.dj_choice = req.oldies / 2 := by sorry
  have h7 : req.total = req.electropop + req.dance + req.rock + req.oldies + req.dj_choice + req.rap := by sorry
  sorry

end rap_song_requests_l3353_335361


namespace max_value_expression_l3353_335326

theorem max_value_expression (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  (x^2 - 2*x*y + y^2) * (x^2 - 2*x*z + z^2) * (y^2 - 2*y*z + z^2) ≤ 1 := by
  sorry

end max_value_expression_l3353_335326


namespace work_time_problem_l3353_335330

/-- The time taken to complete a work when multiple workers work together -/
def combined_work_time (work_rates : List ℚ) : ℚ :=
  1 / (work_rates.sum)

/-- The problem of finding the combined work time for A, B, and C -/
theorem work_time_problem :
  let a_rate : ℚ := 1 / 12
  let b_rate : ℚ := 1 / 24
  let c_rate : ℚ := 1 / 18
  combined_work_time [a_rate, b_rate, c_rate] = 72 / 13 := by
  sorry

#eval combined_work_time [1/12, 1/24, 1/18]

end work_time_problem_l3353_335330


namespace circle_intersection_range_l3353_335375

/-- The problem statement translated to Lean 4 --/
theorem circle_intersection_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (x - a)^2 + (y - (a - 3))^2 = 1) ↔ 0 ≤ a ∧ a ≤ 3 := by
  sorry

end circle_intersection_range_l3353_335375


namespace fourth_sample_number_l3353_335386

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun i => firstSample + i * (totalStudents / sampleSize)

theorem fourth_sample_number
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (h_total : totalStudents = 52)
  (h_sample : sampleSize = 4)
  (h_first : systematicSample totalStudents sampleSize 7 0 = 7)
  (h_second : systematicSample totalStudents sampleSize 7 1 = 33)
  (h_third : systematicSample totalStudents sampleSize 7 2 = 46) :
  systematicSample totalStudents sampleSize 7 3 = 20 :=
sorry

end fourth_sample_number_l3353_335386


namespace anns_shopping_cost_anns_shopping_proof_l3353_335342

theorem anns_shopping_cost (total_spent : ℕ) (shorts_quantity : ℕ) (shorts_price : ℕ) 
  (shoes_quantity : ℕ) (shoes_price : ℕ) (tops_quantity : ℕ) : ℕ :=
  let shorts_total := shorts_quantity * shorts_price
  let shoes_total := shoes_quantity * shoes_price
  let tops_total := total_spent - shorts_total - shoes_total
  tops_total / tops_quantity

theorem anns_shopping_proof :
  anns_shopping_cost 75 5 7 2 10 4 = 5 := by
  sorry

end anns_shopping_cost_anns_shopping_proof_l3353_335342


namespace marble_ratio_is_two_to_one_l3353_335395

/-- The ratio of Mary's blue marbles to Dan's blue marbles -/
def marble_ratio (dans_marbles marys_marbles : ℕ) : ℚ :=
  marys_marbles / dans_marbles

/-- Proof that the ratio of Mary's blue marbles to Dan's blue marbles is 2:1 -/
theorem marble_ratio_is_two_to_one :
  marble_ratio 5 10 = 2 := by
  sorry

end marble_ratio_is_two_to_one_l3353_335395


namespace range_of_k_l3353_335310

def system_of_inequalities (x k : ℝ) : Prop :=
  x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+7)*x + 7*k < 0

def integer_solutions (k : ℝ) : Prop :=
  ∀ x : ℤ, system_of_inequalities (x : ℝ) k ↔ x = -3 ∨ x = -2

theorem range_of_k :
  ∀ k : ℝ, integer_solutions k → k ∈ Set.Ici (-3) ∩ Set.Iio 2 :=
sorry

end range_of_k_l3353_335310


namespace racing_track_width_l3353_335371

theorem racing_track_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) : 
  r₁ - r₂ = 10 := by
  sorry

end racing_track_width_l3353_335371


namespace paint_cans_for_house_l3353_335388

/-- Represents the number of paint cans needed for a house painting job -/
def paint_cans_needed (num_bedrooms : ℕ) (paint_per_room : ℕ) 
  (color_can_size : ℕ) (white_can_size : ℕ) : ℕ :=
  let num_other_rooms := 2 * num_bedrooms
  let total_rooms := num_bedrooms + num_other_rooms
  let total_paint_needed := total_rooms * paint_per_room
  let color_cans := num_bedrooms * paint_per_room
  let white_paint_needed := num_other_rooms * paint_per_room
  let white_cans := (white_paint_needed + white_can_size - 1) / white_can_size
  color_cans + white_cans

/-- Theorem stating that the number of paint cans needed for the given conditions is 10 -/
theorem paint_cans_for_house : paint_cans_needed 3 2 1 3 = 10 := by
  sorry

end paint_cans_for_house_l3353_335388


namespace power_of_three_expression_l3353_335314

theorem power_of_three_expression : 3^3 - 3^2 + 3^1 - 3^0 = 20 := by
  sorry

end power_of_three_expression_l3353_335314


namespace convex_number_count_l3353_335345

/-- A function that checks if a three-digit number is convex -/
def isConvex (n : Nat) : Bool :=
  let a₁ := n / 100
  let a₂ := (n / 10) % 10
  let a₃ := n % 10
  100 ≤ n ∧ n < 1000 ∧ a₁ < a₂ ∧ a₃ < a₂

/-- The count of convex numbers -/
def convexCount : Nat :=
  (List.range 1000).filter isConvex |>.length

/-- Theorem stating that the count of convex numbers is 240 -/
theorem convex_number_count : convexCount = 240 := by
  sorry

end convex_number_count_l3353_335345


namespace inscribed_circle_radius_l3353_335364

/-- Given a triangle with exradii 2, 3, and 6 cm, the radius of the inscribed circle is 1 cm. -/
theorem inscribed_circle_radius (r₁ r₂ r₃ : ℝ) (hr₁ : r₁ = 2) (hr₂ : r₂ = 3) (hr₃ : r₃ = 6) :
  (1 / r₁ + 1 / r₂ + 1 / r₃)⁻¹ = 1 := by
  sorry

end inscribed_circle_radius_l3353_335364


namespace ohara_triple_49_64_l3353_335304

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ (Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x)

/-- Theorem: If (49, 64, x) is an O'Hara triple, then x = 15 -/
theorem ohara_triple_49_64 (x : ℕ) :
  is_ohara_triple 49 64 x → x = 15 := by
  sorry

end ohara_triple_49_64_l3353_335304


namespace regular_octahedron_faces_regular_octahedron_has_eight_faces_l3353_335394

/-- A regular octahedron is a Platonic solid with equilateral triangular faces. -/
structure RegularOctahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of faces of a regular octahedron is 8. -/
theorem regular_octahedron_faces (o : RegularOctahedron) : Nat :=
  8

/-- Prove that a regular octahedron has 8 faces. -/
theorem regular_octahedron_has_eight_faces (o : RegularOctahedron) :
  regular_octahedron_faces o = 8 := by
  sorry

end regular_octahedron_faces_regular_octahedron_has_eight_faces_l3353_335394


namespace sum_of_powers_of_i_l3353_335339

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i : i^2024 + i^2025 + i^2026 + i^2027 = 0 := by
  sorry

end sum_of_powers_of_i_l3353_335339


namespace janet_time_saved_l3353_335315

/-- The number of minutes Janet spends looking for her keys daily -/
def looking_time : ℕ := 8

/-- The number of minutes Janet spends complaining after finding her keys daily -/
def complaining_time : ℕ := 3

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total time Janet saves in a week if she stops losing her keys -/
def time_saved : ℕ := (looking_time + complaining_time) * days_in_week

theorem janet_time_saved : time_saved = 77 := by
  sorry

end janet_time_saved_l3353_335315


namespace convex_polygon_sides_l3353_335319

/-- The number of sides in a convex polygon where the sum of all angles except two is 3420 degrees. -/
def polygon_sides : ℕ := 22

/-- The sum of interior angles of a polygon with n sides. -/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The sum of all angles except two in the polygon. -/
def given_angle_sum : ℝ := 3420

theorem convex_polygon_sides :
  ∃ (missing_angles : ℝ), 
    missing_angles ≥ 0 ∧ 
    missing_angles < 360 ∧
    interior_angle_sum polygon_sides = given_angle_sum + missing_angles := by
  sorry

#check convex_polygon_sides

end convex_polygon_sides_l3353_335319


namespace octagon_diagonals_l3353_335320

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l3353_335320


namespace fans_with_all_items_l3353_335336

def stadium_capacity : ℕ := 4500
def tshirt_interval : ℕ := 60
def hat_interval : ℕ := 45
def keychain_interval : ℕ := 75

theorem fans_with_all_items :
  (stadium_capacity / (Nat.lcm tshirt_interval (Nat.lcm hat_interval keychain_interval))) = 5 := by
  sorry

end fans_with_all_items_l3353_335336


namespace min_value_expression_l3353_335384

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 4) * (b^2 + 4*b + 4) * (c^2 + 4*c + 4) / (a * b * c) ≥ 729 := by
  sorry

end min_value_expression_l3353_335384


namespace cone_lateral_surface_area_l3353_335383

theorem cone_lateral_surface_area 
  (r : ℝ) (h : ℝ) (l : ℝ) (A : ℝ) 
  (h_r : r = 2) 
  (h_h : h = Real.sqrt 5) 
  (h_l : l = Real.sqrt (r^2 + h^2)) 
  (h_A : A = π * r * l) : A = 6 * π := by
  sorry

end cone_lateral_surface_area_l3353_335383


namespace boat_animals_correct_number_of_dogs_l3353_335305

theorem boat_animals (sheep_initial : ℕ) (cows_initial : ℕ) (sheep_drowned : ℕ) (animals_survived : ℕ) : ℕ :=
  let cows_drowned := 2 * sheep_drowned
  let sheep_survived := sheep_initial - sheep_drowned
  let cows_survived := cows_initial - cows_drowned
  let dogs := animals_survived - sheep_survived - cows_survived
  dogs

theorem correct_number_of_dogs : 
  boat_animals 20 10 3 35 = 14 := by
  sorry

end boat_animals_correct_number_of_dogs_l3353_335305


namespace advertisement_length_main_theorem_l3353_335370

/-- Proves that the advertisement length is 20 minutes given the movie theater conditions -/
theorem advertisement_length : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (movie_length : ℕ) (replays : ℕ) (operating_time : ℕ) (ad_length : ℕ) =>
    movie_length = 90 ∧ 
    replays = 6 ∧ 
    operating_time = 660 ∧
    movie_length * replays + ad_length * replays = operating_time →
    ad_length = 20

/-- The main theorem stating the advertisement length -/
theorem main_theorem : advertisement_length 90 6 660 20 := by
  sorry

end advertisement_length_main_theorem_l3353_335370


namespace tractor_trailer_unloading_l3353_335392

theorem tractor_trailer_unloading (initial_load : ℝ) : 
  initial_load = 50000 → 
  let remaining_after_first := initial_load - 0.1 * initial_load
  let final_load := remaining_after_first - 0.2 * remaining_after_first
  final_load = 36000 := by
sorry

end tractor_trailer_unloading_l3353_335392


namespace purely_imaginary_complex_equation_l3353_335360

theorem purely_imaginary_complex_equation (a : ℝ) (z : ℂ) :
  z + 3 * Complex.I = a + a * Complex.I →
  z.re = 0 →
  a = 0 := by
sorry

end purely_imaginary_complex_equation_l3353_335360


namespace science_books_in_large_box_probability_l3353_335362

def total_textbooks : ℕ := 16
def science_textbooks : ℕ := 4
def box_capacities : List ℕ := [2, 4, 5, 5]

theorem science_books_in_large_box_probability :
  let total_ways := (total_textbooks.choose box_capacities[2]) * 
                    ((total_textbooks - box_capacities[2]).choose box_capacities[2]) * 
                    ((total_textbooks - box_capacities[2] - box_capacities[2]).choose box_capacities[1]) * 
                    1
  let favorable_ways := 2 * (box_capacities[2].choose science_textbooks) * 
                        (total_textbooks - science_textbooks).choose 1 * 
                        ((total_textbooks - box_capacities[2]).choose box_capacities[2]) * 
                        ((total_textbooks - box_capacities[2] - box_capacities[2]).choose box_capacities[1])
  (favorable_ways : ℚ) / total_ways = 5 / 182 := by
  sorry

end science_books_in_large_box_probability_l3353_335362


namespace george_socks_problem_l3353_335377

theorem george_socks_problem (initial_socks : ℕ) (new_socks : ℕ) (final_socks : ℕ) 
  (h1 : initial_socks = 28)
  (h2 : new_socks = 36)
  (h3 : final_socks = 60) :
  initial_socks - (initial_socks + new_socks - final_socks) + new_socks = final_socks ∧ 
  initial_socks + new_socks - final_socks = 4 :=
by sorry

end george_socks_problem_l3353_335377


namespace simplify_expression_l3353_335323

theorem simplify_expression : (3 / 4 : ℚ) * 60 - (8 / 5 : ℚ) * 60 + 63 = 12 := by
  sorry

end simplify_expression_l3353_335323


namespace spelling_bee_participants_l3353_335382

/-- In a competition, given a participant's ranking from best and worst, determine the total number of participants. -/
theorem spelling_bee_participants (n : ℕ) 
  (h_best : n = 75)  -- Priya is the 75th best
  (h_worst : n = 75) -- Priya is the 75th worst
  : (2 * n - 1 : ℕ) = 149 := by
  sorry

end spelling_bee_participants_l3353_335382


namespace simplify_fraction_l3353_335348

theorem simplify_fraction (x y : ℝ) (hxy : x ≠ y) (hxy_neg : x ≠ -y) (hx : x ≠ 0) :
  (1 / (x - y) - 1 / (x + y)) / (x * y / (x^2 - y^2)) = 2 / x :=
by sorry

end simplify_fraction_l3353_335348


namespace misha_older_than_tanya_l3353_335324

/-- Represents a person's age in years and months -/
structure Age where
  years : ℕ
  months : ℕ
  inv : months < 12

/-- Compares two Ages -/
def Age.lt (a b : Age) : Prop :=
  a.years < b.years ∨ (a.years = b.years ∧ a.months < b.months)

/-- Adds months to an Age -/
def Age.addMonths (a : Age) (m : ℕ) : Age :=
  { years := a.years + (a.months + m) / 12,
    months := (a.months + m) % 12,
    inv := by sorry }

/-- Subtracts months from an Age -/
def Age.subMonths (a : Age) (m : ℕ) : Age :=
  { years := a.years - (m + 11) / 12,
    months := (a.months + 12 - (m % 12)) % 12,
    inv := by sorry }

theorem misha_older_than_tanya (tanya_past misha_future : Age) :
  tanya_past.addMonths 19 = tanya_past.addMonths 19 →
  misha_future.subMonths 16 = misha_future.subMonths 16 →
  tanya_past.years = 16 →
  misha_future.years = 19 →
  Age.lt (tanya_past.addMonths 19) (misha_future.subMonths 16) := by
  sorry

end misha_older_than_tanya_l3353_335324


namespace horizontal_chord_theorem_l3353_335357

-- Define the set of valid d values
def ValidD : Set ℝ := {d | ∃ n : ℕ+, d = 1 / n}

theorem horizontal_chord_theorem (f : ℝ → ℝ) (h_cont : Continuous f) (h_end : f 0 = f 1) :
  ∀ d : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x + d ∈ Set.Icc 0 1 ∧ f x = f (x + d)) ↔ d ∈ ValidD :=
sorry

end horizontal_chord_theorem_l3353_335357


namespace seating_arrangements_with_restriction_l3353_335355

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def circular_permutations (n : ℕ) : ℕ := factorial (n - 1)

def adjacent_arrangements (n : ℕ) : ℕ := 2 * factorial (n - 2)

theorem seating_arrangements_with_restriction (total_people : ℕ) 
  (h1 : total_people = 8) :
  circular_permutations total_people - adjacent_arrangements total_people = 3600 :=
sorry

end seating_arrangements_with_restriction_l3353_335355


namespace shoe_price_calculation_l3353_335356

theorem shoe_price_calculation (initial_price : ℝ) : 
  initial_price = 50 →
  let wednesday_price := initial_price * (1 + 0.15)
  let thursday_price := wednesday_price * (1 - 0.05)
  let monday_price := thursday_price * (1 - 0.20)
  monday_price = 43.70 := by sorry

end shoe_price_calculation_l3353_335356


namespace first_class_product_rate_l3353_335334

/-- Given a product with a pass rate and a rate of first-class products among qualified products,
    calculate the overall rate of first-class products. -/
theorem first_class_product_rate
  (pass_rate : ℝ)
  (first_class_rate_among_qualified : ℝ)
  (h_pass_rate : pass_rate = 0.95)
  (h_first_class_rate_among_qualified : first_class_rate_among_qualified = 0.2) :
  pass_rate * first_class_rate_among_qualified = 0.19 := by
  sorry

end first_class_product_rate_l3353_335334


namespace paint_area_is_129_l3353_335338

/-- The area of the wall to be painted, given the dimensions of the wall, window, and door. -/
def areaToBePainted (wallHeight wallLength windowHeight windowLength doorHeight doorLength : ℕ) : ℕ :=
  wallHeight * wallLength - (windowHeight * windowLength + doorHeight * doorLength)

/-- Theorem stating that the area to be painted is 129 square feet. -/
theorem paint_area_is_129 :
  areaToBePainted 10 15 3 5 2 3 = 129 := by
  sorry

#eval areaToBePainted 10 15 3 5 2 3

end paint_area_is_129_l3353_335338


namespace parkway_elementary_boys_l3353_335321

theorem parkway_elementary_boys (total_students : ℕ) (soccer_players : ℕ) (girls_not_playing : ℕ)
  (h1 : total_students = 470)
  (h2 : soccer_players = 250)
  (h3 : girls_not_playing = 135)
  (h4 : (86 : ℚ) / 100 * soccer_players = ↑⌊(86 : ℚ) / 100 * soccer_players⌋) :
  total_students - (girls_not_playing + (soccer_players - ⌊(86 : ℚ) / 100 * soccer_players⌋)) = 300 :=
by sorry

end parkway_elementary_boys_l3353_335321


namespace factoring_expression_l3353_335350

theorem factoring_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) = (y + 2) * (5 * y + 9) := by
  sorry

end factoring_expression_l3353_335350


namespace m_range_characterization_l3353_335378

/-- Proposition P: The equation x^2 + mx + 1 = 0 has two distinct negative roots -/
def P (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Proposition Q: The equation 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- The range of values for m satisfying the given conditions -/
def m_range (m : ℝ) : Prop := m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3

theorem m_range_characterization :
  ∀ m : ℝ, ((P m ∨ Q m) ∧ ¬(P m ∧ Q m)) ↔ m_range m :=
sorry

end m_range_characterization_l3353_335378


namespace carnival_tickets_l3353_335309

theorem carnival_tickets (num_games : ℕ) (found_tickets : ℕ) (ticket_value : ℕ) (total_value : ℕ) :
  num_games = 5 →
  found_tickets = 5 →
  ticket_value = 3 →
  total_value = 30 →
  ∃ (tickets_per_game : ℕ),
    (tickets_per_game * num_games + found_tickets) * ticket_value = total_value ∧
    tickets_per_game = 1 := by
  sorry

end carnival_tickets_l3353_335309


namespace inverse_matrices_sum_l3353_335316

def A (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![a, 2, b; 3, 3, 4; c, 6, d]

def B (e f g h : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![-4, e, -12; f, -14, g; 3, h, 5]

theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  (A a b c d) * (B e f g h) = 1 →
  a + b + c + d + e + f + g + h = 47 := by
  sorry

end inverse_matrices_sum_l3353_335316


namespace jerry_feathers_l3353_335335

theorem jerry_feathers (x : ℕ) : 
  let hawk_feathers : ℕ := 6
  let eagle_feathers : ℕ := x * hawk_feathers
  let total_feathers : ℕ := hawk_feathers + eagle_feathers
  let remaining_after_gift : ℕ := total_feathers - 10
  let sold_feathers : ℕ := remaining_after_gift / 2
  let final_feathers : ℕ := remaining_after_gift - sold_feathers
  (final_feathers = 49) → (x = 17) :=
by sorry

end jerry_feathers_l3353_335335


namespace ship_passengers_l3353_335379

theorem ship_passengers : 
  ∀ (P : ℕ), 
    (P / 4 : ℚ) + (P / 8 : ℚ) + (P / 12 : ℚ) + (P / 6 : ℚ) + 36 = P → 
    P = 96 := by
  sorry

end ship_passengers_l3353_335379


namespace intersection_point_of_g_and_inverse_l3353_335399

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 15*x + 35

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! c : ℝ, g c = c ∧ c = -5 := by sorry

end intersection_point_of_g_and_inverse_l3353_335399


namespace range_of_p_l3353_335306

/-- The function p(x) = x^4 - 4x^2 + 4 -/
def p (x : ℝ) : ℝ := x^4 - 4*x^2 + 4

/-- The theorem stating that the range of p(x) over [0, ∞) is [0, ∞) -/
theorem range_of_p :
  ∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, x ≥ 0 ∧ p x = y :=
by sorry

end range_of_p_l3353_335306


namespace xy_max_value_l3353_335358

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  x * y ≤ 1/8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x * y = 1/8 :=
by sorry

end xy_max_value_l3353_335358


namespace shaded_probability_is_half_l3353_335354

/-- Represents an isosceles triangle with base 2 cm and height 4 cm -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  is_isosceles : base = 2 ∧ height = 4

/-- Represents the division of the triangle into 4 regions -/
structure TriangleDivision where
  triangle : IsoscelesTriangle
  num_regions : ℕ
  is_divided : num_regions = 4

/-- Represents the shading of two opposite regions -/
structure ShadedRegions where
  division : TriangleDivision
  num_shaded : ℕ
  are_opposite : num_shaded = 2

/-- The probability of the spinner landing on a shaded region -/
def shaded_probability (shaded : ShadedRegions) : ℚ :=
  shaded.num_shaded / shaded.division.num_regions

/-- Theorem stating that the probability of landing on a shaded region is 1/2 -/
theorem shaded_probability_is_half (shaded : ShadedRegions) :
  shaded_probability shaded = 1/2 := by
  sorry

end shaded_probability_is_half_l3353_335354


namespace odd_m_triple_g_35_l3353_335397

def g (n : Int) : Int :=
  if n % 2 = 1 then n + 5 else n / 2

theorem odd_m_triple_g_35 (m : Int) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 35) : m = 135 := by
  sorry

end odd_m_triple_g_35_l3353_335397


namespace longest_side_range_l3353_335318

-- Define an obtuse triangle
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_obtuse : ∃ angle, angle > π/2 ∧ angle < π

-- Theorem statement
theorem longest_side_range (triangle : ObtuseTriangle) 
  (ha : triangle.a = 1) 
  (hb : triangle.b = 2) : 
  (Real.sqrt 5 < triangle.c ∧ triangle.c < 3) ∨ triangle.c = 2 := by
  sorry

end longest_side_range_l3353_335318


namespace corn_syrup_amount_l3353_335331

/-- Represents the ratio of ingredients in a drink formulation -/
structure Ratio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- Represents a drink formulation -/
structure Formulation :=
  (ratio : Ratio)
  (water_amount : ℚ)

def standard_ratio : Ratio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

def sport_ratio (r : Ratio) : Ratio :=
  { flavoring := r.flavoring,
    corn_syrup := r.corn_syrup / 3,
    water := r.water * 2 }

def sport_formulation : Formulation :=
  { ratio := sport_ratio standard_ratio,
    water_amount := 120 }

theorem corn_syrup_amount :
  (sport_formulation.ratio.corn_syrup / sport_formulation.ratio.water) *
    sport_formulation.water_amount = 8 := by
  sorry

end corn_syrup_amount_l3353_335331


namespace student_committee_candidates_l3353_335325

theorem student_committee_candidates : 
  ∃ n : ℕ, 
    n > 0 ∧ 
    n * (n - 1) = 132 ∧ 
    n = 12 := by
  sorry

end student_committee_candidates_l3353_335325


namespace teacher_earnings_five_weeks_l3353_335396

/-- Calculates the teacher's earnings for piano lessons over a given number of weeks -/
def teacher_earnings (rate_per_half_hour : ℕ) (lesson_duration_hours : ℕ) (weeks : ℕ) : ℕ :=
  rate_per_half_hour * 2 * lesson_duration_hours * weeks

/-- Proves that the teacher earns $100 in 5 weeks under the given conditions -/
theorem teacher_earnings_five_weeks :
  teacher_earnings 10 1 5 = 100 :=
by
  sorry

#eval teacher_earnings 10 1 5

end teacher_earnings_five_weeks_l3353_335396


namespace nail_color_percentage_difference_l3353_335372

theorem nail_color_percentage_difference (total nails : ℕ) (purple blue : ℕ) :
  total = 20 →
  purple = 6 →
  blue = 8 →
  let striped := total - purple - blue
  let blue_percentage := (blue : ℚ) / (total : ℚ) * 100
  let striped_percentage := (striped : ℚ) / (total : ℚ) * 100
  blue_percentage - striped_percentage = 10 := by
sorry

end nail_color_percentage_difference_l3353_335372
