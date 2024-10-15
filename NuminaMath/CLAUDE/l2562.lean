import Mathlib

namespace NUMINAMATH_CALUDE_inverse_proportion_points_l2562_256200

/-- An inverse proportion function passing through (-6, 1) also passes through (2, -3) -/
theorem inverse_proportion_points :
  ∀ k : ℝ, k / (-6 : ℝ) = 1 → k / (2 : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_l2562_256200


namespace NUMINAMATH_CALUDE_olya_candies_l2562_256213

theorem olya_candies 
  (total : ℕ)
  (pasha masha tolya olya : ℕ)
  (h_total : pasha + masha + tolya + olya = total)
  (h_total_val : total = 88)
  (h_masha_tolya : masha + tolya = 57)
  (h_pasha_most : pasha > masha ∧ pasha > tolya ∧ pasha > olya)
  (h_at_least_one : pasha ≥ 1 ∧ masha ≥ 1 ∧ tolya ≥ 1 ∧ olya ≥ 1) :
  olya = 1 := by
sorry

end NUMINAMATH_CALUDE_olya_candies_l2562_256213


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l2562_256280

theorem wire_cut_ratio (a b : ℝ) : 
  a > 0 → b > 0 → (∃ (r : ℝ), a = 2 * Real.pi * r) → (∃ (s : ℝ), b = 4 * s) → a = b → a / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l2562_256280


namespace NUMINAMATH_CALUDE_additional_charge_per_segment_l2562_256217

/-- Proves that the additional charge per 2/5 of a mile is $0.40 --/
theorem additional_charge_per_segment (initial_fee : ℚ) (trip_distance : ℚ) (total_charge : ℚ) 
  (h1 : initial_fee = 9/4)  -- $2.25
  (h2 : trip_distance = 18/5)  -- 3.6 miles
  (h3 : total_charge = 117/20)  -- $5.85
  : (total_charge - initial_fee) / (trip_distance / (2/5)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_additional_charge_per_segment_l2562_256217


namespace NUMINAMATH_CALUDE_power_two_ge_two_times_l2562_256224

theorem power_two_ge_two_times (n : ℕ) : 2^n ≥ 2*n := by
  sorry

end NUMINAMATH_CALUDE_power_two_ge_two_times_l2562_256224


namespace NUMINAMATH_CALUDE_number_of_partitions_l2562_256265

-- Define the set A
def A : Set Nat := {1, 2}

-- Define what a partition is
def is_partition (A₁ A₂ : Set Nat) : Prop :=
  A₁ ∪ A₂ = A

-- Define when two partitions are considered the same
def same_partition (A₁ A₂ : Set Nat) : Prop :=
  A₁ = A₂

-- Define a function to count the number of different partitions
def count_partitions : Nat :=
  sorry

-- The theorem to prove
theorem number_of_partitions :
  count_partitions = 9 :=
sorry

end NUMINAMATH_CALUDE_number_of_partitions_l2562_256265


namespace NUMINAMATH_CALUDE_eighth_term_value_l2562_256242

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first six terms
  sum_six : ℚ
  -- The seventh term
  seventh_term : ℚ

/-- Theorem: Given an arithmetic sequence where the sum of the first six terms is 21
    and the seventh term is 8, the eighth term is 65/7 -/
theorem eighth_term_value (seq : ArithmeticSequence)
    (h1 : seq.sum_six = 21)
    (h2 : seq.seventh_term = 8) :
    ∃ (a d : ℚ), a + 7 * d = 65 / 7 ∧
                 6 * a + 15 * d = 21 ∧
                 a + 6 * d = 8 :=
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l2562_256242


namespace NUMINAMATH_CALUDE_slate_rock_count_l2562_256285

theorem slate_rock_count :
  let pumice_count : ℕ := 11
  let granite_count : ℕ := 4
  let total_count (slate_count : ℕ) : ℕ := slate_count + pumice_count + granite_count
  let prob_two_slate (slate_count : ℕ) : ℚ :=
    (slate_count : ℚ) / (total_count slate_count : ℚ) *
    ((slate_count - 1 : ℚ) / (total_count slate_count - 1 : ℚ))
  ∃ (slate_count : ℕ),
    prob_two_slate slate_count = 15 / 100 ∧
    slate_count = 10 :=
by sorry

end NUMINAMATH_CALUDE_slate_rock_count_l2562_256285


namespace NUMINAMATH_CALUDE_farm_animals_ratio_l2562_256201

theorem farm_animals_ratio :
  let goats : ℕ := 66
  let chickens : ℕ := 2 * goats
  let ducks : ℕ := (goats + chickens) / 2
  let pigs : ℕ := goats - 33
  (pigs : ℚ) / ducks = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_ratio_l2562_256201


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l2562_256240

/-- The probability of no rain on a given day -/
def prob_no_rain : ℝ := 0.3

/-- The probability of 5 inches of rain on a given day -/
def prob_5_inches : ℝ := 0.4

/-- The probability of 12 inches of rain on a given day -/
def prob_12_inches : ℝ := 0.3

/-- The amount of rainfall in inches when it rains 5 inches -/
def rain_5_inches : ℝ := 5

/-- The amount of rainfall in inches when it rains 12 inches -/
def rain_12_inches : ℝ := 12

/-- The number of days in the week -/
def days_in_week : ℕ := 7

/-- The expected rainfall for one day -/
def expected_daily_rainfall : ℝ :=
  prob_no_rain * 0 + prob_5_inches * rain_5_inches + prob_12_inches * rain_12_inches

/-- Theorem: The expected total rainfall for the week is 39.2 inches -/
theorem expected_weekly_rainfall :
  (days_in_week : ℝ) * expected_daily_rainfall = 39.2 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l2562_256240


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2562_256295

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse defined by its center and semi-axes lengths -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis length
  b : ℝ  -- semi-minor axis length

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- The given points -/
def points : List Point := [
  { x := 1, y := 1 },
  { x := 0, y := 0 },
  { x := 0, y := 3 },
  { x := 4, y := 0 },
  { x := 4, y := 3 }
]

theorem ellipse_minor_axis_length :
  ∃ (e : Ellipse),
    (∀ p ∈ points, pointOnEllipse p e) ∧
    (e.center.x = 2 ∧ e.center.y = 1.5) ∧
    (e.a = 2) ∧
    (e.b * 2 = 2 * Real.sqrt 3) :=
by sorry

#check ellipse_minor_axis_length

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2562_256295


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2562_256237

theorem arithmetic_mean_of_fractions :
  let a := 3 / 5
  let b := 5 / 7
  let c := 9 / 14
  let arithmetic_mean := (a + b) / 2
  arithmetic_mean = 23 / 35 ∧ arithmetic_mean ≠ c := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2562_256237


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l2562_256239

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 5*x^3 + 6*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 4 - 3*x - 7*x^3 + 10*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The theorem stating that c = -3/5 makes h(x) a polynomial of degree 3 -/
theorem degree_three_polynomial :
  ∃ (c : ℝ), c = -3/5 ∧ 
  (∀ (x : ℝ), h c x = 2 + (-15 - 3*c)*x + (4 - 0*c)*x^2 + (-5 - 7*c)*x^3) :=
sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l2562_256239


namespace NUMINAMATH_CALUDE_algebraic_identities_l2562_256227

theorem algebraic_identities (a b c : ℝ) : 
  (a^4 * (a^2)^3 = a^10) ∧ 
  (2*a^3*b^2*c / ((1/3)*a^2*b) = 6*a*b*c) ∧ 
  (6*a*((1/3)*a*b - b) - (2*a*b + b)*(a - 1) = -5*a*b + b) ∧ 
  ((a - 2)^2 - (3*a + 2*b)*(3*a - 2*b) = -8*a^2 - 4*a + 4 + 4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l2562_256227


namespace NUMINAMATH_CALUDE_cos_A_minus_B_l2562_256228

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2)
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = -3/8 := by
sorry

end NUMINAMATH_CALUDE_cos_A_minus_B_l2562_256228


namespace NUMINAMATH_CALUDE_purple_length_is_three_l2562_256206

/-- The length of the purple part of a pencil -/
def purple_length (total black blue : ℝ) : ℝ := total - black - blue

/-- Theorem stating that the length of the purple part of the pencil is 3 cm -/
theorem purple_length_is_three :
  let total := 6
  let black := 2
  let blue := 1
  purple_length total black blue = 3 := by
  sorry

end NUMINAMATH_CALUDE_purple_length_is_three_l2562_256206


namespace NUMINAMATH_CALUDE_logger_productivity_l2562_256277

/-- Represents the number of trees one logger can cut down per day -/
def trees_per_logger_per_day (forest_length : ℕ) (forest_width : ℕ) (trees_per_square_mile : ℕ) 
  (days_per_month : ℕ) (num_loggers : ℕ) (num_months : ℕ) : ℕ :=
  let total_trees := forest_length * forest_width * trees_per_square_mile
  let total_days := num_months * days_per_month
  total_trees / (num_loggers * total_days)

theorem logger_productivity : 
  trees_per_logger_per_day 4 6 600 30 8 10 = 6 := by
  sorry

#eval trees_per_logger_per_day 4 6 600 30 8 10

end NUMINAMATH_CALUDE_logger_productivity_l2562_256277


namespace NUMINAMATH_CALUDE_minimum_value_problem_l2562_256298

theorem minimum_value_problem (x y : ℝ) (h1 : 5 * x - x * y - y = -6) (h2 : x > -1) :
  ∃ (min : ℝ), min = 3 + 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2 * x + y ≥ z := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l2562_256298


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2562_256263

theorem arithmetic_equality : 2^2 * 7 + 5 * 12 + 7^2 * 2 + 6 * 3 = 212 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2562_256263


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l2562_256272

theorem junk_mail_distribution (total : ℕ) (blocks : ℕ) (first : ℕ) (second : ℕ) 
  (h1 : total = 2758)
  (h2 : blocks = 5)
  (h3 : first = 365)
  (h4 : second = 421) :
  (total - first - second) / (blocks - 2) = 657 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l2562_256272


namespace NUMINAMATH_CALUDE_original_ticket_price_l2562_256216

/-- Proves that the original ticket price is $7 given the problem conditions --/
theorem original_ticket_price (num_tickets : ℕ) (discount_percent : ℚ) (total_cost : ℚ) : 
  num_tickets = 24 → 
  discount_percent = 1/2 → 
  total_cost = 84 → 
  (1 - discount_percent) * (num_tickets : ℚ) * (7 : ℚ) = total_cost := by
sorry

end NUMINAMATH_CALUDE_original_ticket_price_l2562_256216


namespace NUMINAMATH_CALUDE_range_of_m_l2562_256244

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2562_256244


namespace NUMINAMATH_CALUDE_two_rectangle_formations_l2562_256256

def square_sides : List ℕ := [3, 5, 9, 11, 14, 19, 20, 24, 31, 33, 36, 39, 42]

def rectangle_width : ℕ := 75
def rectangle_height : ℕ := 112

def forms_rectangle (subset : List ℕ) : Prop :=
  (subset.map (λ x => x^2)).sum = rectangle_width * rectangle_height

theorem two_rectangle_formations :
  ∃ (subset1 subset2 : List ℕ),
    subset1 ⊆ square_sides ∧
    subset2 ⊆ square_sides ∧
    subset1 ∩ subset2 = ∅ ∧
    forms_rectangle subset1 ∧
    forms_rectangle subset2 :=
sorry

end NUMINAMATH_CALUDE_two_rectangle_formations_l2562_256256


namespace NUMINAMATH_CALUDE_equation_solutions_l2562_256218

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = -11 ∧ x₂ = 1 ∧ 
  (∀ x : ℝ, 4 * (2 * x + 1)^2 = 9 * (x - 3)^2 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2562_256218


namespace NUMINAMATH_CALUDE_complex_multiplication_l2562_256233

/-- Given that i is the imaginary unit, prove that (2+i)(1-3i) = 5-5i -/
theorem complex_multiplication (i : ℂ) (hi : i * i = -1) :
  (2 + i) * (1 - 3*i) = 5 - 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2562_256233


namespace NUMINAMATH_CALUDE_coin_puzzle_solution_l2562_256274

/-- Represents the number of coins in each pile -/
structure CoinPiles :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Represents the coin movement operation -/
def moveCoins (piles : CoinPiles) : CoinPiles :=
  { first := piles.first - piles.second + piles.third,
    second := 2 * piles.second - piles.third,
    third := piles.third + piles.second - piles.first }

/-- Theorem stating that if after moving coins each pile has 16 coins, 
    then the initial number in the first pile was 22 -/
theorem coin_puzzle_solution (initial : CoinPiles) :
  (moveCoins initial).first = 16 ∧
  (moveCoins initial).second = 16 ∧
  (moveCoins initial).third = 16 →
  initial.first = 22 :=
sorry

end NUMINAMATH_CALUDE_coin_puzzle_solution_l2562_256274


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_l2562_256270

theorem imaginary_part_of_i : Complex.im Complex.I = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_l2562_256270


namespace NUMINAMATH_CALUDE_a_8_equals_15_l2562_256243

/-- The sum of the first n terms of the sequence {aₙ} -/
def S (n : ℕ) : ℕ := n^2

/-- The n-th term of the sequence {aₙ} -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- Theorem stating that the 8th term of the sequence is 15 -/
theorem a_8_equals_15 : a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_a_8_equals_15_l2562_256243


namespace NUMINAMATH_CALUDE_largest_valid_number_l2562_256276

def is_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n % 100 = (n / 100 % 10 + n / 1000) % 10) ∧
  (n % 10 = (n / 10 % 10 + n / 100 % 10) % 10)

theorem largest_valid_number : 
  (∀ m : ℕ, is_valid m → m ≤ 9099) ∧ is_valid 9099 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l2562_256276


namespace NUMINAMATH_CALUDE_plastic_for_one_ruler_l2562_256230

/-- The amount of plastic needed to make one ruler, given the total amount of plastic and the number of rulers that can be made. -/
def plastic_per_ruler (total_plastic : ℕ) (num_rulers : ℕ) : ℚ :=
  (total_plastic : ℚ) / (num_rulers : ℚ)

/-- Theorem stating that 8 grams of plastic are needed to make one ruler. -/
theorem plastic_for_one_ruler :
  plastic_per_ruler 828 103 = 8 := by
  sorry

end NUMINAMATH_CALUDE_plastic_for_one_ruler_l2562_256230


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l2562_256262

theorem sin_cos_difference_equals_half : 
  Real.sin (-(10 * π / 180)) * Real.cos (160 * π / 180) - 
  Real.sin (80 * π / 180) * Real.sin (200 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l2562_256262


namespace NUMINAMATH_CALUDE_enrique_commission_l2562_256219

/-- Calculates the total commission for a salesperson given their sales and commission rate. -/
def calculate_commission (suit_price : ℚ) (suit_count : ℕ) 
                         (shirt_price : ℚ) (shirt_count : ℕ) 
                         (loafer_price : ℚ) (loafer_count : ℕ) 
                         (commission_rate : ℚ) : ℚ :=
  let total_sales := suit_price * suit_count + 
                     shirt_price * shirt_count + 
                     loafer_price * loafer_count
  total_sales * commission_rate

/-- Theorem stating that Enrique's commission is $300.00 given his sales and commission rate. -/
theorem enrique_commission :
  calculate_commission 700 2 50 6 150 2 (15/100) = 300 := by
  sorry

end NUMINAMATH_CALUDE_enrique_commission_l2562_256219


namespace NUMINAMATH_CALUDE_range_of_a_l2562_256236

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 1)^2 > 4
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(p x) ∧ (q x a))) →
  (∀ a : ℝ, a ≥ 1 ↔ (∀ x : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(p x) ∧ (q x a)))) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2562_256236


namespace NUMINAMATH_CALUDE_min_tan_half_angle_l2562_256250

theorem min_tan_half_angle (A B C : Real) (h1 : A + B + C = π) 
  (h2 : Real.tan (A/2) + Real.tan (B/2) = 1) :
  ∃ (m : Real), m = 3/4 ∧ ∀ x, x = Real.tan (C/2) → x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_tan_half_angle_l2562_256250


namespace NUMINAMATH_CALUDE_park_walk_time_l2562_256268

/-- Represents the time in minutes for various walks in the park. -/
structure ParkWalks where
  office_to_hidden_lake : ℕ
  hidden_lake_to_office : ℕ
  total_time : ℕ

/-- Calculates the time from Park Office to Lake Park restaurant. -/
def time_to_restaurant (w : ParkWalks) : ℕ :=
  w.total_time - (w.office_to_hidden_lake + w.hidden_lake_to_office)

/-- Theorem stating the time from Park Office to Lake Park restaurant is 10 minutes. -/
theorem park_walk_time (w : ParkWalks) 
  (h1 : w.office_to_hidden_lake = 15)
  (h2 : w.hidden_lake_to_office = 7)
  (h3 : w.total_time = 32) : 
  time_to_restaurant w = 10 := by
  sorry

#eval time_to_restaurant { office_to_hidden_lake := 15, hidden_lake_to_office := 7, total_time := 32 }

end NUMINAMATH_CALUDE_park_walk_time_l2562_256268


namespace NUMINAMATH_CALUDE_colored_tape_length_l2562_256259

theorem colored_tape_length : 
  ∀ (original_length : ℝ),
  (1 / 5 : ℝ) * original_length + -- Used for art
  (3 / 4 : ℝ) * (4 / 5 : ℝ) * original_length + -- Given away
  1.5 = original_length → -- Remaining length
  original_length = 7.5 := by
sorry

end NUMINAMATH_CALUDE_colored_tape_length_l2562_256259


namespace NUMINAMATH_CALUDE_relationship_abc_l2562_256264

theorem relationship_abc : 
  let a : ℝ := (1/3 : ℝ)^(2/3 : ℝ)
  let b : ℝ := (1/3 : ℝ)^(1/3 : ℝ)
  let c : ℝ := (2/3 : ℝ)^(1/3 : ℝ)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2562_256264


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l2562_256267

theorem arithmetic_expression_equals_24 : (2 + 4 / 10) * 10 = 24 := by
  sorry

#check arithmetic_expression_equals_24

end NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l2562_256267


namespace NUMINAMATH_CALUDE_sixth_triangular_number_l2562_256260

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := (n * (n + 1)) / 2

/-- The 6th triangular number is 21 -/
theorem sixth_triangular_number : triangular 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sixth_triangular_number_l2562_256260


namespace NUMINAMATH_CALUDE_irreducible_fractions_l2562_256221

theorem irreducible_fractions (n : ℕ) : 
  (Nat.gcd (2*n + 13) (n + 7) = 1) ∧ 
  (Nat.gcd (2*n^2 - 1) (n + 1) = 1) ∧ 
  (Nat.gcd (n^2 - n + 1) (n^2 + 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fractions_l2562_256221


namespace NUMINAMATH_CALUDE_two_vectors_basis_iff_linearly_independent_not_any_two_vectors_form_basis_l2562_256222

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

/-- Two vectors form a basis for a 2-dimensional vector space if and only if they are linearly independent. -/
theorem two_vectors_basis_iff_linearly_independent (v w : V) :
  Submodule.span ℝ {v, w} = ⊤ ↔ LinearIndependent ℝ ![v, w] :=
sorry

/-- It is not true that any two vectors in a 2-dimensional vector space form a basis. -/
theorem not_any_two_vectors_form_basis :
  ¬ ∀ (v w : V), Submodule.span ℝ {v, w} = ⊤ :=
sorry

end NUMINAMATH_CALUDE_two_vectors_basis_iff_linearly_independent_not_any_two_vectors_form_basis_l2562_256222


namespace NUMINAMATH_CALUDE_haircut_cost_l2562_256257

-- Define the constants
def hair_growth_rate : ℝ := 1.5
def max_hair_length : ℝ := 9
def min_hair_length : ℝ := 6
def tip_percentage : ℝ := 0.2
def annual_haircut_cost : ℝ := 324

-- Define the theorem
theorem haircut_cost (haircut_cost : ℝ) : 
  hair_growth_rate * 12 / (max_hair_length - min_hair_length) * 
  (haircut_cost * (1 + tip_percentage)) = annual_haircut_cost → 
  haircut_cost = 45 := by
sorry

end NUMINAMATH_CALUDE_haircut_cost_l2562_256257


namespace NUMINAMATH_CALUDE_stating_initial_order_correct_l2562_256286

/-- Represents the colors of the notebooks -/
inductive Color
  | Blue
  | Grey
  | Brown
  | Red
  | Yellow

/-- Represents a stack of notebooks -/
def Stack := List Color

/-- The first arrangement of notebooks -/
def first_arrangement : (Stack × Stack) :=
  ([Color.Red, Color.Yellow, Color.Grey], [Color.Brown, Color.Blue])

/-- The second arrangement of notebooks -/
def second_arrangement : (Stack × Stack) :=
  ([Color.Brown, Color.Red], [Color.Yellow, Color.Grey, Color.Blue])

/-- The hypothesized initial order of notebooks -/
def initial_order : Stack :=
  [Color.Brown, Color.Red, Color.Yellow, Color.Grey, Color.Blue]

/-- 
Theorem stating that the initial_order is correct given the two arrangements
-/
theorem initial_order_correct :
  ∃ (process : Stack → (Stack × Stack)),
    process initial_order = first_arrangement ∧
    process (initial_order.reverse.reverse) = second_arrangement :=
sorry

end NUMINAMATH_CALUDE_stating_initial_order_correct_l2562_256286


namespace NUMINAMATH_CALUDE_range_equality_odd_decreasing_function_l2562_256271

-- Statement 1
theorem range_equality (f : ℝ → ℝ) : Set.range f = Set.range (fun x ↦ f (x + 1)) := by sorry

-- Statement 3
theorem odd_decreasing_function (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_decreasing_neg : ∀ x y, x < y → y < 0 → f y < f x) : 
  ∀ x y, 0 < x → x < y → f y < f x := by sorry

end NUMINAMATH_CALUDE_range_equality_odd_decreasing_function_l2562_256271


namespace NUMINAMATH_CALUDE_robot_staircase_l2562_256281

theorem robot_staircase (a b : ℕ+) : 
  ∃ n : ℕ, n = a + b - Nat.gcd a b ∧ 
  (∀ m : ℕ, m < n → ¬∃ (k l : ℕ), k * a = m + l * b) ∧
  (∃ (k l : ℕ), k * a = n + l * b) := by
  sorry

end NUMINAMATH_CALUDE_robot_staircase_l2562_256281


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l2562_256249

/-- Given two similar right triangles, where the first triangle has a side of 15 units and a hypotenuse of 34 units, and the second triangle has a hypotenuse of 68 units, the shortest side of the second triangle is 2√931 units. -/
theorem similar_triangle_shortest_side :
  ∀ (a b c d e : ℝ),
  a^2 + 15^2 = 34^2 →  -- Pythagorean theorem for the first triangle
  a ≤ 15 →  -- a is the shortest side of the first triangle
  c^2 + d^2 = 68^2 →  -- Pythagorean theorem for the second triangle
  c / a = d / 15 →  -- triangles are similar
  c / a = 68 / 34 →  -- ratio of hypotenuses
  c = 2 * Real.sqrt 931 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l2562_256249


namespace NUMINAMATH_CALUDE_final_card_expectation_l2562_256209

/-- Represents a deck of cards -/
def Deck := List Nat

/-- The process of drawing two cards, discarding one, and reinserting the other -/
def drawDiscardReinsert (d : Deck) : Deck :=
  sorry

/-- The expected value of the label of the remaining card after the process -/
def expectedValue (d : Deck) : Rat :=
  sorry

/-- Theorem stating the expected value of the final card in a 100-card deck -/
theorem final_card_expectation :
  let initialDeck : Deck := List.range 100
  expectedValue initialDeck = 467 / 8 := by
  sorry

end NUMINAMATH_CALUDE_final_card_expectation_l2562_256209


namespace NUMINAMATH_CALUDE_noemi_initial_money_l2562_256255

/-- The amount of money Noemi lost on roulette -/
def roulette_loss : ℕ := 400

/-- The amount of money Noemi lost on blackjack -/
def blackjack_loss : ℕ := 500

/-- The amount of money Noemi still has in her purse -/
def remaining_money : ℕ := 800

/-- The initial amount of money Noemi had -/
def initial_money : ℕ := roulette_loss + blackjack_loss + remaining_money

theorem noemi_initial_money : 
  initial_money = roulette_loss + blackjack_loss + remaining_money := by
  sorry

end NUMINAMATH_CALUDE_noemi_initial_money_l2562_256255


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2562_256220

theorem inequality_system_solution (x : ℝ) :
  (4 * x + 5 > x - 1) ∧ ((3 * x - 1) / 2 < x) → -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2562_256220


namespace NUMINAMATH_CALUDE_square_perimeter_l2562_256253

/-- Given a rectangle with length 50 cm and width 10 cm, and a square with an area
    five times that of the rectangle, prove that the perimeter of the square is 200 cm. -/
theorem square_perimeter (rectangle_length : ℝ) (rectangle_width : ℝ) (square_area : ℝ) :
  rectangle_length = 50 ∧ 
  rectangle_width = 10 ∧ 
  square_area = 5 * (rectangle_length * rectangle_width) →
  4 * Real.sqrt square_area = 200 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l2562_256253


namespace NUMINAMATH_CALUDE_pencils_per_box_l2562_256210

/-- The number of pencils Louise has for each color and the number of boxes --/
structure PencilData where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  boxes : ℕ

/-- The conditions of Louise's pencil organization --/
def validPencilData (d : PencilData) : Prop :=
  d.red = 20 ∧
  d.blue = 2 * d.red ∧
  d.yellow = 40 ∧
  d.green = d.red + d.blue ∧
  d.boxes = 8

/-- The theorem stating that each box holds 20 pencils --/
theorem pencils_per_box (d : PencilData) (h : validPencilData d) :
  (d.red + d.blue + d.yellow + d.green) / d.boxes = 20 := by
  sorry

#check pencils_per_box

end NUMINAMATH_CALUDE_pencils_per_box_l2562_256210


namespace NUMINAMATH_CALUDE_student_pet_difference_l2562_256229

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 24

/-- The number of rabbits in each classroom -/
def rabbits_per_classroom : ℕ := 2

/-- The number of hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 3

/-- Theorem: The difference between the total number of students and the total number of pets
    in all fourth-grade classrooms is 95 -/
theorem student_pet_difference :
  num_classrooms * students_per_classroom - 
  (num_classrooms * rabbits_per_classroom + num_classrooms * hamsters_per_classroom) = 95 := by
  sorry

end NUMINAMATH_CALUDE_student_pet_difference_l2562_256229


namespace NUMINAMATH_CALUDE_melanie_missed_games_l2562_256202

theorem melanie_missed_games (total_games attended_games : ℕ) 
  (h1 : total_games = 64)
  (h2 : attended_games = 32) :
  total_games - attended_games = 32 := by
  sorry

end NUMINAMATH_CALUDE_melanie_missed_games_l2562_256202


namespace NUMINAMATH_CALUDE_angle_measure_l2562_256278

theorem angle_measure (PQR PQS : ℝ) (h1 : PQR = 40) (h2 : PQS = 15) : PQR - PQS = 25 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l2562_256278


namespace NUMINAMATH_CALUDE_simplify_expression_l2562_256292

theorem simplify_expression (x : ℝ) : 7*x + 9 - 2*x + 15 = 5*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2562_256292


namespace NUMINAMATH_CALUDE_scientific_notation_of_189100_l2562_256234

/-- The scientific notation of 189100 is 1.891 × 10^5 -/
theorem scientific_notation_of_189100 :
  (189100 : ℝ) = 1.891 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_189100_l2562_256234


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2562_256282

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2562_256282


namespace NUMINAMATH_CALUDE_six_and_negative_six_are_opposite_l2562_256296

/-- Two real numbers are opposite if one is the negative of the other -/
def are_opposite (a b : ℝ) : Prop := b = -a

/-- 6 and -6 are opposite numbers -/
theorem six_and_negative_six_are_opposite : are_opposite 6 (-6) := by
  sorry

end NUMINAMATH_CALUDE_six_and_negative_six_are_opposite_l2562_256296


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2562_256225

theorem sin_cos_identity : 
  Real.sin (160 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (200 * π / 180) * Real.cos (80 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2562_256225


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2562_256288

/-- Given two quadratic equations y² + dy + e = 0 and 4x² - ax - 12 = 0,
    where the roots of the first equation are each three more than 
    the roots of the second equation, prove that e = (3a + 24) / 4 -/
theorem quadratic_roots_relation (a d e : ℝ) : 
  (∀ x y : ℝ, (4 * x^2 - a * x - 12 = 0 → y^2 + d * y + e = 0 → y = x + 3)) →
  e = (3 * a + 24) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2562_256288


namespace NUMINAMATH_CALUDE_tan_equality_solution_l2562_256207

open Real

theorem tan_equality_solution (x : ℝ) : 
  0 ≤ x ∧ x ≤ 180 ∧ 
  tan (150 * π / 180 - x * π / 180) = 
    (sin (150 * π / 180) - sin (x * π / 180)) / 
    (cos (150 * π / 180) - cos (x * π / 180)) →
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_solution_l2562_256207


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l2562_256297

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := (m * (x^2 - 1)) / x - 2 * Real.log x

theorem f_has_three_zeros :
  ∃ (a b c : ℝ), a < b ∧ b < c ∧ 
  (∀ x, x > 0 → f (1/2) x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry


end NUMINAMATH_CALUDE_f_has_three_zeros_l2562_256297


namespace NUMINAMATH_CALUDE_expand_product_l2562_256247

theorem expand_product (x : ℝ) : (5*x^2 + 7) * (3*x^3 + 4*x + 1) = 15*x^5 + 41*x^3 + 5*x^2 + 28*x + 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2562_256247


namespace NUMINAMATH_CALUDE_speed_calculation_l2562_256289

theorem speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 50 → time = 2.5 → speed = distance / time → speed = 20 := by
sorry

end NUMINAMATH_CALUDE_speed_calculation_l2562_256289


namespace NUMINAMATH_CALUDE_max_vertex_product_sum_l2562_256293

/-- The set of numbers to be assigned to the cube faces -/
def CubeNumbers : Finset ℕ := {1, 2, 3, 8, 9, 10}

/-- A valid assignment of numbers to cube faces -/
structure CubeAssignment where
  assignment : Fin 6 → ℕ
  valid : ∀ i, assignment i ∈ CubeNumbers
  distinct : Function.Injective assignment

/-- The sum of products at vertices for a given assignment -/
def vertexProductSum (a : CubeAssignment) : ℕ :=
  let faces := a.assignment
  (faces 0 + faces 1) * (faces 2 + faces 3) * (faces 4 + faces 5)

/-- Theorem stating the maximum sum of vertex products -/
theorem max_vertex_product_sum :
  ∀ a : CubeAssignment, vertexProductSum a ≤ 1331 :=
sorry

end NUMINAMATH_CALUDE_max_vertex_product_sum_l2562_256293


namespace NUMINAMATH_CALUDE_smallest_area_triangle_l2562_256261

-- Define the angle XAY
def Angle (X A Y : Point) : Prop := sorry

-- Define a point O inside the angle XAY
def InsideAngle (O X A Y : Point) : Prop := sorry

-- Define symmetry of angles with respect to a point
def SymmetricAngle (X A Y X' A' Y' O : Point) : Prop := sorry

-- Define the intersection points B and C
def IntersectionPoints (B C X A Y X' A' Y' O : Point) : Prop := sorry

-- Define a line passing through three points
def LineThroughPoints (P Q R : Point) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem smallest_area_triangle 
  (X A Y O : Point) 
  (h1 : Angle X A Y) 
  (h2 : InsideAngle O X A Y) 
  (X' A' Y' : Point) 
  (h3 : SymmetricAngle X A Y X' A' Y' O) 
  (B C : Point) 
  (h4 : IntersectionPoints B C X A Y X' A' Y' O) 
  (h5 : LineThroughPoints B O C) :
  ∀ P Q : Point, 
    LineThroughPoints P O Q → 
    TriangleArea A P Q ≥ TriangleArea A B C := 
by sorry

end NUMINAMATH_CALUDE_smallest_area_triangle_l2562_256261


namespace NUMINAMATH_CALUDE_unique_perpendicular_to_skew_lines_l2562_256283

/-- A line in three-dimensional space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- A line is perpendicular to another line -/
def is_perpendicular (l1 l2 : Line3D) : Prop :=
  -- Definition of perpendicular lines
  sorry

theorem unique_perpendicular_to_skew_lines 
  (p : ℝ × ℝ × ℝ) (l1 l2 : Line3D) (h : are_skew l1 l2) :
  ∃! l : Line3D, l.point = p ∧ is_perpendicular l l1 ∧ is_perpendicular l l2 :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_to_skew_lines_l2562_256283


namespace NUMINAMATH_CALUDE_total_commission_proof_l2562_256232

def commission_rate : ℚ := 2 / 100

def house_prices : List ℚ := [157000, 499000, 125000]

def calculate_commission (price : ℚ) : ℚ :=
  price * commission_rate

theorem total_commission_proof :
  (house_prices.map calculate_commission).sum = 15620 := by
  sorry

end NUMINAMATH_CALUDE_total_commission_proof_l2562_256232


namespace NUMINAMATH_CALUDE_discounted_shirt_price_l2562_256245

/-- Given a shirt sold at a 30% discount for 560 units of currency,
    prove that the original price was 800 units of currency. -/
theorem discounted_shirt_price (discount_percent : ℝ) (discounted_price : ℝ) :
  discount_percent = 30 →
  discounted_price = 560 →
  (1 - discount_percent / 100) * 800 = discounted_price := by
sorry

end NUMINAMATH_CALUDE_discounted_shirt_price_l2562_256245


namespace NUMINAMATH_CALUDE_books_rearrangement_l2562_256287

theorem books_rearrangement (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 1500 → 
  books_per_initial_box = 42 → 
  books_per_new_box = 45 → 
  (initial_boxes * books_per_initial_box) % books_per_new_box = 0 :=
by sorry

end NUMINAMATH_CALUDE_books_rearrangement_l2562_256287


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2562_256204

theorem quadratic_equation_roots (k : ℤ) :
  let f := fun x : ℝ => k * x^2 - (4*k + 1) * x + 3*k + 3
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (x₁^2 + x₂^2 = (3 * Real.sqrt 5 / 2)^2 → k = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2562_256204


namespace NUMINAMATH_CALUDE_prime_squares_and_fourth_powers_l2562_256279

theorem prime_squares_and_fourth_powers (p : ℕ) : 
  Prime p ↔ 
  (p = 2 ∨ p = 3) ∧ 
  (∃ (a b c k : ℤ), a^2 + b^2 + c^2 = p ∧ a^4 + b^4 + c^4 = k * p) :=
sorry

end NUMINAMATH_CALUDE_prime_squares_and_fourth_powers_l2562_256279


namespace NUMINAMATH_CALUDE_overtime_rate_calculation_l2562_256246

/-- Calculate the overtime rate given the following conditions:
  * Regular hourly rate
  * Total weekly pay
  * Total hours worked
  * Overtime hours worked
-/
def calculate_overtime_rate (regular_rate : ℚ) (total_pay : ℚ) (total_hours : ℕ) (overtime_hours : ℕ) : ℚ :=
  let regular_hours := total_hours - overtime_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_hours

theorem overtime_rate_calculation :
  let regular_rate : ℚ := 60 / 100  -- 60 cents per hour
  let total_pay : ℚ := 3240 / 100   -- $32.40
  let total_hours : ℕ := 50
  let overtime_hours : ℕ := 8
  calculate_overtime_rate regular_rate total_pay total_hours overtime_hours = 90 / 100 := by
    sorry

#eval calculate_overtime_rate (60 / 100) (3240 / 100) 50 8

end NUMINAMATH_CALUDE_overtime_rate_calculation_l2562_256246


namespace NUMINAMATH_CALUDE_yellow_ball_players_l2562_256203

theorem yellow_ball_players (total : ℕ) (white : ℕ) (both : ℕ) (yellow : ℕ) : 
  total = 35 → white = 26 → both = 19 → yellow = 28 → 
  total = white + yellow - both :=
by sorry

end NUMINAMATH_CALUDE_yellow_ball_players_l2562_256203


namespace NUMINAMATH_CALUDE_smallest_z_for_inequality_l2562_256294

theorem smallest_z_for_inequality : ∃! z : ℕ, (∀ w : ℕ, 27^w > 3^24 → w ≥ z) ∧ 27^z > 3^24 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_z_for_inequality_l2562_256294


namespace NUMINAMATH_CALUDE_creature_dressing_order_l2562_256252

-- Define the number of arms
def num_arms : ℕ := 6

-- Define the number of items per arm
def items_per_arm : ℕ := 3

-- Define the total number of items
def total_items : ℕ := num_arms * items_per_arm

-- Define the number of valid permutations per arm (1 out of 6)
def valid_perm_per_arm : ℕ := 1

-- Define the total number of permutations per arm
def total_perm_per_arm : ℕ := Nat.factorial items_per_arm

-- Theorem statement
theorem creature_dressing_order :
  (Nat.factorial total_items) / (total_perm_per_arm ^ num_arms) =
  (Nat.factorial total_items) / (3 ^ num_arms) :=
sorry

end NUMINAMATH_CALUDE_creature_dressing_order_l2562_256252


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l2562_256269

/-- The focal length of an ellipse with equation 2x^2 + 3y^2 = 1 is √6/3 -/
theorem ellipse_focal_length : 
  let a : ℝ := 1 / Real.sqrt 2
  let b : ℝ := 1 / Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  c = Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l2562_256269


namespace NUMINAMATH_CALUDE_removed_player_height_l2562_256275

/-- The height of the removed player given the initial and final average heights -/
def height_of_removed_player (initial_avg : ℝ) (final_avg : ℝ) : ℝ :=
  11 * initial_avg - 10 * final_avg

/-- Theorem stating the height of the removed player -/
theorem removed_player_height :
  height_of_removed_player 182 181 = 192 := by
  sorry

#eval height_of_removed_player 182 181

end NUMINAMATH_CALUDE_removed_player_height_l2562_256275


namespace NUMINAMATH_CALUDE_sequence_properties_l2562_256266

def arithmetic_sequence (n : ℕ) : ℕ := n

def geometric_sequence (n : ℕ) : ℕ := 2^n

def S (n : ℕ) : ℚ := (n^2 + n) / 2

def T (n : ℕ) : ℕ := 2 * (2^n - 1)

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → arithmetic_sequence n = n) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sequence n = 2^n) ∧
  (∀ n : ℕ, n < 8 → T n + arithmetic_sequence n ≤ 300) ∧
  (T 8 + arithmetic_sequence 8 > 300) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2562_256266


namespace NUMINAMATH_CALUDE_benny_picked_two_l2562_256290

-- Define the total number of apples picked
def total_apples : ℕ := 11

-- Define the number of apples Dan picked
def dan_apples : ℕ := 9

-- Define Benny's apples as the difference between total and Dan's
def benny_apples : ℕ := total_apples - dan_apples

-- Theorem stating that Benny picked 2 apples
theorem benny_picked_two : benny_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_benny_picked_two_l2562_256290


namespace NUMINAMATH_CALUDE_mixed_committee_probability_l2562_256258

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 6

def probability_mixed_committee : ℚ :=
  1 - (2 * Nat.choose boys committee_size) / Nat.choose total_members committee_size

theorem mixed_committee_probability :
  probability_mixed_committee = 33187 / 33649 :=
sorry

end NUMINAMATH_CALUDE_mixed_committee_probability_l2562_256258


namespace NUMINAMATH_CALUDE_divisible_by_117_and_2_less_than_2011_l2562_256235

theorem divisible_by_117_and_2_less_than_2011 : 
  (Finset.filter (fun n => n < 2011 ∧ n % 117 = 0 ∧ n % 2 = 0) (Finset.range 2011)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_117_and_2_less_than_2011_l2562_256235


namespace NUMINAMATH_CALUDE_male_democrat_ratio_l2562_256226

/-- Proves the ratio of male democrats to total male participants in a meeting --/
theorem male_democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) :
  total_participants = 750 →
  female_democrats = 125 →
  female_democrats * 2 ≤ total_participants →
  3 * female_democrats * 2 = total_participants →
  (total_participants / 3 - female_democrats) * 4 = total_participants - female_democrats * 2 :=
by
  sorry

#check male_democrat_ratio

end NUMINAMATH_CALUDE_male_democrat_ratio_l2562_256226


namespace NUMINAMATH_CALUDE_teagan_total_payment_l2562_256214

def original_shirt_price : ℚ := 60
def original_jacket_price : ℚ := 90
def price_reduction : ℚ := 20 / 100
def num_shirts : ℕ := 5
def num_jackets : ℕ := 10

def reduced_price (original_price : ℚ) : ℚ :=
  original_price * (1 - price_reduction)

def total_cost (item_price : ℚ) (quantity : ℕ) : ℚ :=
  item_price * quantity

theorem teagan_total_payment :
  total_cost (reduced_price original_shirt_price) num_shirts +
  total_cost (reduced_price original_jacket_price) num_jackets = 960 := by
  sorry

end NUMINAMATH_CALUDE_teagan_total_payment_l2562_256214


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2562_256284

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, (1 < x ∧ x < 2) → x * (x - 3) < 0) ∧
  (∃ x, x * (x - 3) < 0 ∧ ¬(1 < x ∧ x < 2)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2562_256284


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2562_256251

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ 0}
def N : Set ℝ := {y | ∃ x, y = -x^2 + 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2562_256251


namespace NUMINAMATH_CALUDE_max_boxes_in_wooden_box_l2562_256211

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ :=
  m * 100

theorem max_boxes_in_wooden_box :
  let largeBox : BoxDimensions :=
    { length := metersToCentimeters 8
      width := metersToCentimeters 7
      height := metersToCentimeters 6 }
  let smallBox : BoxDimensions :=
    { length := 8
      width := 7
      height := 6 }
  (boxVolume largeBox) / (boxVolume smallBox) = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_in_wooden_box_l2562_256211


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2562_256212

theorem imaginary_part_of_z : Complex.im ((2 : ℂ) - Complex.I) ^ 2 = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2562_256212


namespace NUMINAMATH_CALUDE_spelling_badges_l2562_256299

theorem spelling_badges (H L C : ℕ) : 
  H + L + C = 83 → H = 14 → L = 17 → C = 52 := by
  sorry

end NUMINAMATH_CALUDE_spelling_badges_l2562_256299


namespace NUMINAMATH_CALUDE_base_three_five_digits_l2562_256231

theorem base_three_five_digits : ∃! b : ℕ, b ≥ 2 ∧ b^4 ≤ 200 ∧ 200 < b^5 := by sorry

end NUMINAMATH_CALUDE_base_three_five_digits_l2562_256231


namespace NUMINAMATH_CALUDE_apple_rate_problem_l2562_256273

theorem apple_rate_problem (apple_rate : ℕ) : 
  (8 * apple_rate + 9 * 75 = 1235) → apple_rate = 70 := by
  sorry

end NUMINAMATH_CALUDE_apple_rate_problem_l2562_256273


namespace NUMINAMATH_CALUDE_equation_value_l2562_256254

theorem equation_value (x y : ℝ) (h : x + 2 * y = 30) :
  x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l2562_256254


namespace NUMINAMATH_CALUDE_divisible_by_24_count_l2562_256208

theorem divisible_by_24_count :
  (∃! (s : Finset ℕ), 
    (∀ a ∈ s, 0 < a ∧ a < 100 ∧ 24 ∣ (a^3 + 23)) ∧ 
    s.card = 5) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_24_count_l2562_256208


namespace NUMINAMATH_CALUDE_distinct_arrangements_of_three_letters_l2562_256291

/-- The number of distinct arrangements of 3 unique letters -/
def arrangements_of_three_letters : ℕ := 6

/-- The word consists of 3 distinct letters -/
def number_of_letters : ℕ := 3

theorem distinct_arrangements_of_three_letters : 
  arrangements_of_three_letters = Nat.factorial number_of_letters := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_of_three_letters_l2562_256291


namespace NUMINAMATH_CALUDE_half_power_inequality_l2562_256241

theorem half_power_inequality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) :
  (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l2562_256241


namespace NUMINAMATH_CALUDE_norma_cards_l2562_256238

/-- 
Given that Norma loses 70 cards and has 18 cards left,
prove that she initially had 88 cards.
-/
theorem norma_cards : 
  ∀ (initial_cards : ℕ),
  (initial_cards - 70 = 18) → initial_cards = 88 := by
  sorry

end NUMINAMATH_CALUDE_norma_cards_l2562_256238


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_three_fifths_l2562_256223

theorem fraction_of_powers_equals_three_fifths :
  (3^2011 + 3^2011) / (3^2010 + 3^2012) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_three_fifths_l2562_256223


namespace NUMINAMATH_CALUDE_opera_ticket_price_increase_l2562_256248

theorem opera_ticket_price_increase (last_year_price this_year_price : ℝ) 
  (h1 : last_year_price = 85)
  (h2 : this_year_price = 102) :
  (this_year_price - last_year_price) / last_year_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_opera_ticket_price_increase_l2562_256248


namespace NUMINAMATH_CALUDE_ernies_original_income_l2562_256215

theorem ernies_original_income
  (ernies_original : ℝ)
  (ernies_current : ℝ)
  (jacks_current : ℝ)
  (h1 : ernies_current = 4/5 * ernies_original)
  (h2 : jacks_current = 2 * ernies_original)
  (h3 : ernies_current + jacks_current = 16800) :
  ernies_original = 6000 := by
sorry

end NUMINAMATH_CALUDE_ernies_original_income_l2562_256215


namespace NUMINAMATH_CALUDE_minimum_packages_l2562_256205

theorem minimum_packages (p : ℕ) : p > 0 → (∃ N : ℕ, N = 19 * p ∧ N % 7 = 4 ∧ N % 11 = 1) → p ≥ 40 :=
by sorry

end NUMINAMATH_CALUDE_minimum_packages_l2562_256205
