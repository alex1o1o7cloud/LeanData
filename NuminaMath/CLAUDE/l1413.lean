import Mathlib

namespace NUMINAMATH_CALUDE_calculation_proof_l1413_141352

theorem calculation_proof :
  ((-3)^2 - 60 / 10 * (1 / 10) - |(-2)|) = 32 / 5 ∧
  (-4 / 5 * (9 / 4) + (-1 / 4) * (4 / 5) - (3 / 2) * (-4 / 5)) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1413_141352


namespace NUMINAMATH_CALUDE_total_tiles_is_44_l1413_141336

-- Define the room dimensions
def room_length : ℕ := 20
def room_width : ℕ := 15

-- Define tile sizes
def border_tile_size : ℕ := 2
def inner_tile_size : ℕ := 3

-- Function to calculate the number of border tiles
def border_tiles : ℕ :=
  2 * (room_length / border_tile_size + room_width / border_tile_size) - 4

-- Function to calculate the inner area
def inner_area : ℕ :=
  (room_length - 2 * border_tile_size) * (room_width - 2 * border_tile_size)

-- Function to calculate the number of inner tiles
def inner_tiles : ℕ :=
  (inner_area + inner_tile_size^2 - 1) / inner_tile_size^2

-- Theorem stating the total number of tiles
theorem total_tiles_is_44 :
  border_tiles + inner_tiles = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_tiles_is_44_l1413_141336


namespace NUMINAMATH_CALUDE_simplify_like_terms_l1413_141366

theorem simplify_like_terms (x : ℝ) : 3 * x + 5 * x + 7 * x = 15 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_like_terms_l1413_141366


namespace NUMINAMATH_CALUDE_smallest_n_proof_l1413_141331

def has_digit_seven (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ d = 7 ∧ ∃ k m : ℕ, n = k * 10 + d + m * 100

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def smallest_n_with_properties : ℕ := 65536

theorem smallest_n_proof :
  (is_terminating_decimal smallest_n_with_properties) ∧
  (has_digit_seven smallest_n_with_properties) ∧
  (∀ m : ℕ, m < smallest_n_with_properties →
    ¬(is_terminating_decimal m ∧ has_digit_seven m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_proof_l1413_141331


namespace NUMINAMATH_CALUDE_square_sum_of_special_integers_l1413_141324

theorem square_sum_of_special_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 47)
  (h2 : x^2 * y + x * y^2 = 506) : 
  x^2 + y^2 = 101 := by sorry

end NUMINAMATH_CALUDE_square_sum_of_special_integers_l1413_141324


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1413_141377

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 > 0 →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 4 + a 7 + a 10 = -5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1413_141377


namespace NUMINAMATH_CALUDE_calculation_result_l1413_141340

theorem calculation_result (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : m = -2)  -- m is a negative number with an absolute value of 2
  : m + c * d + a + b + (c * d) ^ 2010 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l1413_141340


namespace NUMINAMATH_CALUDE_train_speed_problem_l1413_141370

/-- Proves that given a train journey of 3x km, where x km is traveled at 50 kmph
    and 2x km is traveled at speed v, and the average speed for the entire journey
    is 25 kmph, the speed v must be 20 kmph. -/
theorem train_speed_problem (x : ℝ) (v : ℝ) (h_x_pos : x > 0) :
  (x / 50 + 2 * x / v = 3 * x / 25) → v = 20 := by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_train_speed_problem_l1413_141370


namespace NUMINAMATH_CALUDE_infinitely_many_composite_mersenne_numbers_l1413_141371

theorem infinitely_many_composite_mersenne_numbers :
  ∀ k : ℕ, ∃ n : ℕ, 
    Odd n ∧ 
    ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2^n - 1 = a * b :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_composite_mersenne_numbers_l1413_141371


namespace NUMINAMATH_CALUDE_fraction_multiplication_equality_l1413_141376

theorem fraction_multiplication_equality : 
  (8 / 9)^2 * (1 / 3)^2 * (2 / 5) = 128 / 3645 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equality_l1413_141376


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_12_with_digit_sum_24_l1413_141318

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_12_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n % 12 = 0 → digit_sum n = 24 → n ≤ 888 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_12_with_digit_sum_24_l1413_141318


namespace NUMINAMATH_CALUDE_number_of_divisors_of_60_l1413_141350

theorem number_of_divisors_of_60 : Nat.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_60_l1413_141350


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1413_141367

-- Problem 1
theorem problem_1 : 8 * 77 * 125 = 77000 := by sorry

-- Problem 2
theorem problem_2 : 12 * 98 = 1176 := by sorry

-- Problem 3
theorem problem_3 : 6 * 321 + 6 * 179 = 3000 := by sorry

-- Problem 4
theorem problem_4 : 56 * 101 - 56 = 5600 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1413_141367


namespace NUMINAMATH_CALUDE_expression_evaluation_l1413_141310

theorem expression_evaluation (a b c d e : ℚ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) 
  (h3 : |e| = 2) : 
  (c + d) / 5 - (1 / 2) * a * b + e = 3 / 2 ∨ 
  (c + d) / 5 - (1 / 2) * a * b + e = -(5 / 2) :=
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1413_141310


namespace NUMINAMATH_CALUDE_complement_A_union_B_eq_univ_A_inter_B_ne_B_l1413_141323

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ 3 - 2*a}

-- Statement for part 1
theorem complement_A_union_B_eq_univ (a : ℝ) :
  (Set.univ \ A) ∪ B a = Set.univ ↔ a ≤ 0 :=
sorry

-- Statement for part 2
theorem A_inter_B_ne_B (a : ℝ) :
  A ∩ B a ≠ B a ↔ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_complement_A_union_B_eq_univ_A_inter_B_ne_B_l1413_141323


namespace NUMINAMATH_CALUDE_bert_stamp_cost_l1413_141383

/-- The total cost of stamps Bert purchased -/
def total_cost (type_a_count type_b_count type_c_count : ℕ) 
               (type_a_price type_b_price type_c_price : ℕ) : ℕ :=
  type_a_count * type_a_price + 
  type_b_count * type_b_price + 
  type_c_count * type_c_price

/-- Theorem stating the total cost of Bert's stamp purchase -/
theorem bert_stamp_cost : 
  total_cost 150 90 60 2 3 5 = 870 := by
  sorry

end NUMINAMATH_CALUDE_bert_stamp_cost_l1413_141383


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l1413_141341

theorem quadratic_root_sum (a b : ℝ) : 
  (1 : ℝ) ^ 2 * a + 1 * b - 3 = 0 → a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l1413_141341


namespace NUMINAMATH_CALUDE_reciprocal_of_point_three_l1413_141387

theorem reciprocal_of_point_three (h : (0.3 : ℚ) = 3/10) : 
  (0.3 : ℚ)⁻¹ = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_point_three_l1413_141387


namespace NUMINAMATH_CALUDE_range_of_a_l1413_141325

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 21 = 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | 5*x - a ≥ 3*x + 2}

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∪ B a = B a → a ≤ -8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1413_141325


namespace NUMINAMATH_CALUDE_willow_count_l1413_141368

theorem willow_count (total : ℕ) (diff : ℕ) : 
  total = 83 →
  diff = 11 →
  ∃ (willows oaks : ℕ),
    willows + oaks = total ∧
    oaks = willows + diff ∧
    willows = 36 := by
  sorry

end NUMINAMATH_CALUDE_willow_count_l1413_141368


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l1413_141315

/-- Proves that the upstream speed of a canoe is 9 km/hr given its downstream speed and the stream speed -/
theorem canoe_upstream_speed 
  (downstream_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : downstream_speed = 12) 
  (h2 : stream_speed = 1.5) : 
  downstream_speed - 2 * stream_speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_canoe_upstream_speed_l1413_141315


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1413_141379

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with common ratio q
  (a 1 + a 2 + a 3 + a 4 = 3) →  -- First condition
  (a 5 + a 6 + a 7 + a 8 = 48) →  -- Second condition
  (a 1 / (1 - q) = -1/5) :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1413_141379


namespace NUMINAMATH_CALUDE_remaining_ribbon_length_l1413_141334

/-- Calculates the remaining ribbon length after wrapping gifts -/
theorem remaining_ribbon_length
  (num_gifts : ℕ)
  (ribbon_per_gift : ℝ)
  (initial_ribbon_length : ℝ)
  (h1 : num_gifts = 8)
  (h2 : ribbon_per_gift = 1.5)
  (h3 : initial_ribbon_length = 15) :
  initial_ribbon_length - (↑num_gifts * ribbon_per_gift) = 3 :=
by sorry

end NUMINAMATH_CALUDE_remaining_ribbon_length_l1413_141334


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_four_l1413_141319

theorem largest_five_digit_divisible_by_four :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99996 ∧ n % 4 = 0 → n ≤ 99996 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_four_l1413_141319


namespace NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l1413_141312

/-- Given an angle α = -3000°, this theorem states that the smallest positive angle
    with the same terminal side as α is 240°. -/
theorem smallest_positive_angle_same_terminal_side :
  let α : ℝ := -3000
  ∃ (k : ℤ), α + k * 360 = 240 ∧
    ∀ (m : ℤ), α + m * 360 > 0 → α + m * 360 ≥ 240 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l1413_141312


namespace NUMINAMATH_CALUDE_ratio_of_arithmetic_sums_l1413_141311

def arithmetic_sum (a₁ d l : ℚ) : ℚ :=
  let n := (l - a₁) / d + 1
  n * (a₁ + l) / 2

theorem ratio_of_arithmetic_sums : 
  let sum1 := arithmetic_sum 5 3 59
  let sum2 := arithmetic_sum 4 4 64
  sum1 / sum2 = 19 / 17 := by sorry

end NUMINAMATH_CALUDE_ratio_of_arithmetic_sums_l1413_141311


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1413_141332

noncomputable def f (x : ℝ) := x^4 - 2*x^3

theorem tangent_line_at_one (x : ℝ) : 
  let p := (1, f 1)
  let m := deriv f 1
  (fun x => m * (x - p.1) + p.2) = (fun x => -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l1413_141332


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l1413_141381

theorem largest_prime_divisor_test (n : ℕ) (h1 : 1100 ≤ n) (h2 : n ≤ 1150) :
  Prime n → ∀ p, Prime p → p > 31 → ¬(p ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l1413_141381


namespace NUMINAMATH_CALUDE_exponent_division_l1413_141390

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1413_141390


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1413_141386

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 1
def q (x : ℝ) : Prop := 1/x < 1

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1413_141386


namespace NUMINAMATH_CALUDE_white_marbles_count_l1413_141314

-- Define the parameters
def total_marbles : ℕ := 60
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def prob_red_or_white : ℚ := 55 / 60

-- Theorem statement
theorem white_marbles_count :
  ∃ (white_marbles : ℕ),
    white_marbles = total_marbles - blue_marbles - red_marbles ∧
    (red_marbles + white_marbles : ℚ) / total_marbles = prob_red_or_white ∧
    white_marbles = 46 := by
  sorry

end NUMINAMATH_CALUDE_white_marbles_count_l1413_141314


namespace NUMINAMATH_CALUDE_probability_sum_less_than_product_l1413_141351

def valid_pairs : Finset (ℕ × ℕ) :=
  (Finset.range 6).product (Finset.range 6)

def satisfying_pairs : Finset (ℕ × ℕ) :=
  valid_pairs.filter (fun p => p.1 + p.2 < p.1 * p.2)

theorem probability_sum_less_than_product :
  (satisfying_pairs.card : ℚ) / valid_pairs.card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_less_than_product_l1413_141351


namespace NUMINAMATH_CALUDE_smallest_positive_c_inequality_l1413_141346

theorem smallest_positive_c_inequality (c : ℝ) : 
  (c > 0 ∧ 
   ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 
   (x^3 + y^3 - x) + c * |x - y| ≥ y - x^2) → 
  c ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_c_inequality_l1413_141346


namespace NUMINAMATH_CALUDE_count_with_3_or_6_in_base_7_eq_1776_l1413_141361

/-- The count of integers among the first 2401 positive integers in base 7 that use 3 or 6 as a digit -/
def count_with_3_or_6_in_base_7 : ℕ :=
  2401 - 5^4

theorem count_with_3_or_6_in_base_7_eq_1776 :
  count_with_3_or_6_in_base_7 = 1776 := by sorry

end NUMINAMATH_CALUDE_count_with_3_or_6_in_base_7_eq_1776_l1413_141361


namespace NUMINAMATH_CALUDE_nancy_jade_amount_l1413_141304

/-- The amount of jade (in grams) needed for a giraffe statue -/
def giraffe_jade : ℝ := 120

/-- The price (in dollars) of a giraffe statue -/
def giraffe_price : ℝ := 150

/-- The amount of jade (in grams) needed for an elephant statue -/
def elephant_jade : ℝ := 2 * giraffe_jade

/-- The price (in dollars) of an elephant statue -/
def elephant_price : ℝ := 350

/-- The additional revenue (in dollars) from making elephant statues instead of giraffe statues -/
def additional_revenue : ℝ := 400

/-- The theorem stating the amount of jade Nancy has -/
theorem nancy_jade_amount :
  ∃ (J : ℝ), J > 0 ∧
    (J / elephant_jade) * elephant_price - (J / giraffe_jade) * giraffe_price = additional_revenue ∧
    J = 1920 := by
  sorry

end NUMINAMATH_CALUDE_nancy_jade_amount_l1413_141304


namespace NUMINAMATH_CALUDE_tangent_lines_with_slope_one_l1413_141393

def f (x : ℝ) := x^3

theorem tangent_lines_with_slope_one :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, (deriv f) x = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_lines_with_slope_one_l1413_141393


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1413_141380

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  Real.sqrt (4 * x + 1) + Real.sqrt (4 * y + 1) + Real.sqrt (4 * z + 1) ≤ 3 * Real.sqrt (35 / 3) ∧
  ∃ x y z, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 8 ∧
    Real.sqrt (4 * x + 1) + Real.sqrt (4 * y + 1) + Real.sqrt (4 * z + 1) = 3 * Real.sqrt (35 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1413_141380


namespace NUMINAMATH_CALUDE_problem_solution_l1413_141374

theorem problem_solution (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1413_141374


namespace NUMINAMATH_CALUDE_derivative_neg_cos_l1413_141355

theorem derivative_neg_cos (x : ℝ) : deriv (fun x => -Real.cos x) x = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_neg_cos_l1413_141355


namespace NUMINAMATH_CALUDE_division_problem_l1413_141305

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) : 
  dividend = 127 → divisor = 14 → remainder = 1 → 
  dividend = divisor * quotient + remainder → quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1413_141305


namespace NUMINAMATH_CALUDE_gain_percentage_is_twenty_percent_l1413_141330

def selling_price : ℝ := 180
def gain : ℝ := 30

theorem gain_percentage_is_twenty_percent : 
  (gain / (selling_price - gain)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gain_percentage_is_twenty_percent_l1413_141330


namespace NUMINAMATH_CALUDE_mrs_heine_dogs_l1413_141333

/-- Given that Mrs. Heine buys 3 heart biscuits for each dog and needs to buy 6 biscuits in total,
    prove that she has 2 dogs. -/
theorem mrs_heine_dogs :
  ∀ (total_biscuits biscuits_per_dog : ℕ),
    total_biscuits = 6 →
    biscuits_per_dog = 3 →
    total_biscuits / biscuits_per_dog = 2 :=
by sorry

end NUMINAMATH_CALUDE_mrs_heine_dogs_l1413_141333


namespace NUMINAMATH_CALUDE_total_statues_l1413_141313

/-- The length of the street in meters -/
def street_length : ℕ := 1650

/-- The interval between statues in meters -/
def statue_interval : ℕ := 50

/-- The number of sides of the street with statues -/
def sides : ℕ := 2

theorem total_statues : 
  (street_length / statue_interval + 1) * sides = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_statues_l1413_141313


namespace NUMINAMATH_CALUDE_unique_solution_implies_n_l1413_141345

/-- Given a real number n, if the equation 9x^2 + nx + 36 = 0 has exactly one solution in x,
    then n = 36 or n = -36 -/
theorem unique_solution_implies_n (n : ℝ) :
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) → n = 36 ∨ n = -36 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_n_l1413_141345


namespace NUMINAMATH_CALUDE_binary_digit_difference_l1413_141321

/-- Returns the number of digits in the base-2 representation of a natural number -/
def numDigitsBinary (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

/-- The difference in the number of binary digits between 800 and 250 is 2 -/
theorem binary_digit_difference : numDigitsBinary 800 - numDigitsBinary 250 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l1413_141321


namespace NUMINAMATH_CALUDE_min_subset_size_for_sum_equation_l1413_141375

theorem min_subset_size_for_sum_equation (n : ℕ) (hn : n ≥ 2) :
  ∃ m : ℕ, m = 2 * n + 2 ∧
  (∀ S : Finset ℕ, S ⊆ Finset.range (3 * n + 1) → S.card ≥ m →
    ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      a = b + c + d) ∧
  (∀ k : ℕ, k < m →
    ∃ T : Finset ℕ, T ⊆ Finset.range (3 * n + 1) ∧ T.card = k ∧
      ∀ a b c d : ℕ, a ∈ T → b ∈ T → c ∈ T → d ∈ T →
        (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) → a ≠ b + c + d) :=
by sorry

end NUMINAMATH_CALUDE_min_subset_size_for_sum_equation_l1413_141375


namespace NUMINAMATH_CALUDE_stock_price_change_l1413_141396

/-- Calculates the net percentage change in stock price over three years -/
def netPercentageChange (year1Change : Real) (year2Change : Real) (year3Change : Real) : Real :=
  let price1 := 1 + year1Change
  let price2 := price1 * (1 + year2Change)
  let price3 := price2 * (1 + year3Change)
  (price3 - 1) * 100

/-- Theorem stating the net percentage change for the given scenario -/
theorem stock_price_change : 
  ∀ (ε : Real), ε > 0 → 
  |netPercentageChange (-0.08) 0.10 0.06 - 7.272| < ε :=
sorry

end NUMINAMATH_CALUDE_stock_price_change_l1413_141396


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l1413_141328

theorem trigonometric_expression_evaluation : 
  (2 * Real.sin (100 * π / 180) - Real.cos (70 * π / 180)) / Real.cos (20 * π / 180) = 2 * Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l1413_141328


namespace NUMINAMATH_CALUDE_line_BC_equation_l1413_141306

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  angle_bisector_B : ℝ → ℝ → Prop
  angle_bisector_C : ℝ → ℝ → Prop

-- Define the specific triangle from the problem
def triangle_ABC : Triangle where
  A := (1, 4)
  angle_bisector_B := λ x y => x - 2*y = 0
  angle_bisector_C := λ x y => x + y - 1 = 0

-- Define the equation of line BC
def line_BC (x y : ℝ) : Prop := 4*x + 17*y + 12 = 0

-- Theorem statement
theorem line_BC_equation (t : Triangle) (h1 : t = triangle_ABC) :
  ∀ x y, t.angle_bisector_B x y ∧ t.angle_bisector_C x y → line_BC x y :=
by sorry

end NUMINAMATH_CALUDE_line_BC_equation_l1413_141306


namespace NUMINAMATH_CALUDE_solve_baseball_card_problem_l1413_141397

def baseball_card_problem (initial_cards : ℕ) (final_cards : ℕ) : Prop :=
  let cards_after_maria := initial_cards - (initial_cards + 1) / 2
  let cards_after_peter := cards_after_maria - 1
  let cards_paul_added := final_cards - cards_after_peter
  cards_paul_added = 12

theorem solve_baseball_card_problem :
  baseball_card_problem 15 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_baseball_card_problem_l1413_141397


namespace NUMINAMATH_CALUDE_olivia_wallet_proof_l1413_141363

def initial_wallet_amount (amount_spent : ℕ) (amount_left : ℕ) : ℕ :=
  amount_spent + amount_left

theorem olivia_wallet_proof (amount_spent : ℕ) (amount_left : ℕ) 
  (h1 : amount_spent = 38) (h2 : amount_left = 90) :
  initial_wallet_amount amount_spent amount_left = 128 := by
  sorry

end NUMINAMATH_CALUDE_olivia_wallet_proof_l1413_141363


namespace NUMINAMATH_CALUDE_carol_invitations_proof_l1413_141339

/-- The number of invitations Carol is sending out -/
def total_invitations : ℕ := 12

/-- The number of packs Carol bought -/
def number_of_packs : ℕ := 3

/-- The number of invitations in each pack -/
def invitations_per_pack : ℕ := total_invitations / number_of_packs

theorem carol_invitations_proof :
  invitations_per_pack = 4 ∧
  total_invitations = number_of_packs * invitations_per_pack :=
by sorry

end NUMINAMATH_CALUDE_carol_invitations_proof_l1413_141339


namespace NUMINAMATH_CALUDE_unsafe_trip_probability_775km_l1413_141337

/-- The probability of not completing a trip safely given the probability of an accident per km and the total distance. -/
def unsafe_trip_probability (p : ℝ) (distance : ℕ) : ℝ :=
  1 - (1 - p) ^ distance

/-- Theorem stating that the probability of not completing a 775 km trip safely
    is equal to 1 - (1 - p)^775, where p is the probability of an accident per km. -/
theorem unsafe_trip_probability_775km (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  unsafe_trip_probability p 775 = 1 - (1 - p)^775 := by
  sorry

#check unsafe_trip_probability_775km

end NUMINAMATH_CALUDE_unsafe_trip_probability_775km_l1413_141337


namespace NUMINAMATH_CALUDE_gcd_102_238_l1413_141369

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l1413_141369


namespace NUMINAMATH_CALUDE_polyhedron_volume_l1413_141326

-- Define the polygons
def right_triangle (a b c : ℝ) := a^2 + b^2 = c^2
def rectangle (l w : ℝ) := l > 0 ∧ w > 0
def equilateral_triangle (s : ℝ) := s > 0

-- Define the polyhedron
def polyhedron (A E F : ℝ → ℝ → ℝ → Prop) 
               (B C D : ℝ → ℝ → Prop) 
               (G : ℝ → Prop) := 
  A 1 2 (Real.sqrt 5) ∧ 
  E 1 2 (Real.sqrt 5) ∧ 
  F 1 2 (Real.sqrt 5) ∧ 
  B 1 2 ∧ 
  C 2 3 ∧ 
  D 1 3 ∧ 
  G (Real.sqrt 5)

-- State the theorem
theorem polyhedron_volume 
  (A E F : ℝ → ℝ → ℝ → Prop) 
  (B C D : ℝ → ℝ → Prop) 
  (G : ℝ → Prop) : 
  polyhedron right_triangle right_triangle right_triangle 
              rectangle rectangle rectangle 
              equilateral_triangle → 
  ∃ v : ℝ, v = 6 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l1413_141326


namespace NUMINAMATH_CALUDE_count_not_divisible_9999_l1413_141302

def count_not_divisible (n : ℕ) : ℕ :=
  n + 1 - (
    (n / 3 + 1) + (n / 5 + 1) + (n / 7 + 1) -
    (n / 15 + 1) - (n / 21 + 1) - (n / 35 + 1) +
    (n / 105 + 1)
  )

theorem count_not_divisible_9999 :
  count_not_divisible 9999 = 4571 := by
sorry

end NUMINAMATH_CALUDE_count_not_divisible_9999_l1413_141302


namespace NUMINAMATH_CALUDE_consecutive_squares_equivalence_l1413_141384

theorem consecutive_squares_equivalence (n : ℤ) : 
  (∃ a : ℤ, n = a^2 + (a + 1)^2) ↔ (∃ b : ℤ, 2*n - 1 = b^2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_squares_equivalence_l1413_141384


namespace NUMINAMATH_CALUDE_tetrahedron_symmetry_l1413_141399

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- The center of mass of a tetrahedron -/
def centerOfMass (t : Tetrahedron) : Point3D := sorry

/-- The center of the circumscribed sphere of a tetrahedron -/
def circumCenter (t : Tetrahedron) : Point3D := sorry

/-- Check if a line intersects an edge of a tetrahedron -/
def intersectsEdge (l : Line3D) (p1 p2 : Point3D) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Theorem statement -/
theorem tetrahedron_symmetry (t : Tetrahedron) 
  (l : Line3D) 
  (h1 : l.point = centerOfMass t) 
  (h2 : l.point = circumCenter t) 
  (h3 : intersectsEdge l t.A t.B) 
  (h4 : intersectsEdge l t.C t.D) : 
  distance t.A t.C = distance t.B t.D ∧ 
  distance t.A t.D = distance t.B t.C := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_symmetry_l1413_141399


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_l1413_141378

/-- A function f(x) = e^x(x^2 + 2ax + 2) is monotonically increasing on R if and only if a is in the range [-1, 1] -/
theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => Real.exp x * (x^2 + 2*a*x + 2))) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_l1413_141378


namespace NUMINAMATH_CALUDE_toy_cost_l1413_141329

theorem toy_cost (saved : ℕ) (allowance : ℕ) (num_toys : ℕ) :
  saved = 21 →
  allowance = 15 →
  num_toys = 6 →
  (saved + allowance) / num_toys = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_cost_l1413_141329


namespace NUMINAMATH_CALUDE_tims_children_treats_l1413_141358

/-- The total number of treats Tim's children get while trick or treating --/
def total_treats (num_children : ℕ) (hours : ℕ) (houses_per_hour : ℕ) (treats_per_kid : ℕ) : ℕ :=
  num_children * hours * houses_per_hour * treats_per_kid

/-- Theorem stating that Tim's children get 180 treats in total --/
theorem tims_children_treats :
  total_treats 3 4 5 3 = 180 := by
  sorry

#eval total_treats 3 4 5 3

end NUMINAMATH_CALUDE_tims_children_treats_l1413_141358


namespace NUMINAMATH_CALUDE_function_equality_l1413_141349

theorem function_equality (f : ℕ+ → ℕ+) 
  (h : ∀ m n : ℕ+, m^2 + f n^2 + (m - f n)^2 ≥ f m^2 + n^2) : 
  ∀ n : ℕ+, f n = n :=
sorry

end NUMINAMATH_CALUDE_function_equality_l1413_141349


namespace NUMINAMATH_CALUDE_age_puzzle_l1413_141303

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 32) (h2 : 4 * (A + x) - 4 * (A - 4) = A) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l1413_141303


namespace NUMINAMATH_CALUDE_congruence_problem_l1413_141347

theorem congruence_problem (x : ℤ) : 
  (4 * x + 5) ≡ 3 [ZMOD 17] → (2 * x + 8) ≡ 7 [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1413_141347


namespace NUMINAMATH_CALUDE_program_output_is_66_l1413_141365

/-- A simplified representation of the program output -/
def program_output : ℕ := 66

/-- The theorem stating that the program output is 66 -/
theorem program_output_is_66 : program_output = 66 := by sorry

end NUMINAMATH_CALUDE_program_output_is_66_l1413_141365


namespace NUMINAMATH_CALUDE_decagon_triangles_l1413_141308

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices required to form a triangle -/
def r : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def num_triangles : ℕ := Nat.choose n r

/-- Theorem: The number of triangles that can be formed using the vertices of a regular decagon is 120 -/
theorem decagon_triangles : num_triangles = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l1413_141308


namespace NUMINAMATH_CALUDE_sin_960_degrees_l1413_141359

theorem sin_960_degrees : Real.sin (960 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_960_degrees_l1413_141359


namespace NUMINAMATH_CALUDE_joan_change_theorem_l1413_141389

def change_received (cat_toy_cost cage_cost amount_paid : ℚ) : ℚ :=
  amount_paid - (cat_toy_cost + cage_cost)

theorem joan_change_theorem (cat_toy_cost cage_cost amount_paid : ℚ) 
  (h1 : cat_toy_cost = 8.77)
  (h2 : cage_cost = 10.97)
  (h3 : amount_paid = 20) :
  change_received cat_toy_cost cage_cost amount_paid = 0.26 := by
  sorry

#eval change_received 8.77 10.97 20

end NUMINAMATH_CALUDE_joan_change_theorem_l1413_141389


namespace NUMINAMATH_CALUDE_largest_prime_to_check_primality_l1413_141382

theorem largest_prime_to_check_primality (n : ℕ) : 
  1000 ≤ n → n ≤ 1100 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  (∀ p : ℕ, p.Prime → p < n → n % p ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_to_check_primality_l1413_141382


namespace NUMINAMATH_CALUDE_garden_flower_ratio_l1413_141360

/-- Represents the number of flowers of each color in the garden -/
structure FlowerCounts where
  red : ℕ
  orange : ℕ
  yellow : ℕ
  pink : ℕ
  purple : ℕ

/-- The conditions of the garden problem -/
def gardenConditions (f : FlowerCounts) : Prop :=
  f.red + f.orange + f.yellow + f.pink + f.purple = 105 ∧
  f.orange = 10 ∧
  f.yellow = f.red - 5 ∧
  f.pink = f.purple ∧
  f.pink + f.purple = 30

theorem garden_flower_ratio : 
  ∀ f : FlowerCounts, gardenConditions f → (f.red : ℚ) / f.orange = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_flower_ratio_l1413_141360


namespace NUMINAMATH_CALUDE_lighter_cost_difference_l1413_141317

/-- Calculates the cost of buying lighters at the gas station with a "buy 4 get 1 free" offer -/
def gas_station_cost (price_per_lighter : ℚ) (num_lighters : ℕ) : ℚ :=
  let sets := (num_lighters + 4) / 5
  let lighters_to_pay := sets * 4
  lighters_to_pay * price_per_lighter

/-- Calculates the cost of buying lighters on Amazon including tax and shipping -/
def amazon_cost (price_per_pack : ℚ) (lighters_per_pack : ℕ) (num_lighters : ℕ) 
                (tax_rate : ℚ) (shipping_cost : ℚ) : ℚ :=
  let packs_needed := (num_lighters + lighters_per_pack - 1) / lighters_per_pack
  let subtotal := packs_needed * price_per_pack
  let tax := subtotal * tax_rate
  subtotal + tax + shipping_cost

theorem lighter_cost_difference : 
  gas_station_cost (175/100) 24 - amazon_cost 5 12 24 (5/100) (7/2) = 1925/100 := by
  sorry

end NUMINAMATH_CALUDE_lighter_cost_difference_l1413_141317


namespace NUMINAMATH_CALUDE_houses_per_block_l1413_141356

theorem houses_per_block (mail_per_block : ℕ) (mail_per_house : ℕ) 
  (h1 : mail_per_block = 32) (h2 : mail_per_house = 8) :
  mail_per_block / mail_per_house = 4 := by
  sorry

end NUMINAMATH_CALUDE_houses_per_block_l1413_141356


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1413_141391

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 200) → (∀ m : ℕ, m > n → m * (m + 1) ≥ 200) → n + (n + 1) = 27 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1413_141391


namespace NUMINAMATH_CALUDE_coal_piles_weights_l1413_141395

theorem coal_piles_weights (pile1 pile2 : ℕ) : 
  pile1 = pile2 + 80 →
  pile1 * 80 / 100 = pile2 - 50 →
  pile1 = 650 ∧ pile2 = 570 := by
sorry

end NUMINAMATH_CALUDE_coal_piles_weights_l1413_141395


namespace NUMINAMATH_CALUDE_scott_running_distance_l1413_141372

/-- Scott's running schedule and total distance for a month --/
theorem scott_running_distance :
  let miles_mon_to_wed : ℕ := 3 * 3
  let miles_thu_fri : ℕ := 2 * (2 * 3)
  let miles_per_week : ℕ := miles_mon_to_wed + miles_thu_fri
  let weeks_in_month : ℕ := 4
  miles_per_week * weeks_in_month = 84 := by
  sorry

end NUMINAMATH_CALUDE_scott_running_distance_l1413_141372


namespace NUMINAMATH_CALUDE_base_conversion_equality_l1413_141354

def base_8_to_10 (n : ℕ) : ℕ := sorry

def base_10_to_7 (n : ℕ) : ℕ := sorry

theorem base_conversion_equality :
  base_10_to_7 (base_8_to_10 5314) = 11026 := by sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l1413_141354


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1413_141344

-- Define an isosceles triangle with one angle of 80 degrees
structure IsoscelesTriangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)
  (is_isosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3))
  (has_80_degree : angle1 = 80 ∨ angle2 = 80 ∨ angle3 = 80)
  (sum_180 : angle1 + angle2 + angle3 = 180)

-- Theorem statement
theorem isosceles_triangle_base_angle (t : IsoscelesTriangle) :
  (t.angle1 = 50 ∨ t.angle1 = 80) ∨
  (t.angle2 = 50 ∨ t.angle2 = 80) ∨
  (t.angle3 = 50 ∨ t.angle3 = 80) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1413_141344


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l1413_141327

/-- Hexagon formed by two overlapping equilateral triangles -/
structure Hexagon where
  /-- Side length of the equilateral triangles -/
  side_length : ℝ
  /-- Rotation angle in radians -/
  rotation_angle : ℝ
  /-- The hexagon is symmetric about a central point -/
  symmetric : Bool
  /-- Points A and A' coincide -/
  coincident_points : Bool

/-- Calculate the area of the hexagon -/
def hexagon_area (h : Hexagon) : ℝ :=
  sorry

/-- Theorem stating the area of the specific hexagon -/
theorem specific_hexagon_area :
  let h : Hexagon := {
    side_length := 2,
    rotation_angle := Real.pi / 6,  -- 30 degrees in radians
    symmetric := true,
    coincident_points := true
  }
  hexagon_area h = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l1413_141327


namespace NUMINAMATH_CALUDE_p_true_q_false_l1413_141348

theorem p_true_q_false (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by sorry

end NUMINAMATH_CALUDE_p_true_q_false_l1413_141348


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1413_141307

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_intersect : ∃ x y : ℝ, y = 3*x ∧ x^2/a^2 - y^2/b^2 = 1) :
  let e := Real.sqrt (1 + (b/a)^2)
  e > Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1413_141307


namespace NUMINAMATH_CALUDE_marks_initial_money_l1413_141394

theorem marks_initial_money (initial_money : ℚ) : 
  (1/2 : ℚ) * initial_money + 14 + 
  (1/3 : ℚ) * initial_money + 16 + 
  (1/4 : ℚ) * initial_money + 18 = initial_money → 
  initial_money = 576 := by
sorry

end NUMINAMATH_CALUDE_marks_initial_money_l1413_141394


namespace NUMINAMATH_CALUDE_trapezoid_shaded_fraction_l1413_141309

/-- Represents a trapezoid divided into strips -/
structure StripedTrapezoid where
  num_strips : ℕ
  shaded_strips : ℕ

/-- The fraction of the trapezoid's area that is shaded -/
def shaded_fraction (t : StripedTrapezoid) : ℚ :=
  t.shaded_strips / t.num_strips

theorem trapezoid_shaded_fraction :
  ∀ t : StripedTrapezoid,
    t.num_strips = 7 →
    shaded_fraction t = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shaded_fraction_l1413_141309


namespace NUMINAMATH_CALUDE_area_of_triangle_l1413_141335

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the foci of the hyperbola
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let c := Real.sqrt 7
  F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

-- Define the angle between PF₁ and PF₂
def angle_F₁PF₂ (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let v₁ := (F₁.1 - P.1, F₁.2 - P.2)
  let v₂ := (F₂.1 - P.1, F₂.2 - P.2)
  let cos_angle := (v₁.1 * v₂.1 + v₁.2 * v₂.2) / 
    (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2))
  cos_angle = 1/2  -- cos 60° = 1/2

-- Theorem statement
theorem area_of_triangle (P F₁ F₂ : ℝ × ℝ) :
  point_on_hyperbola P →
  foci F₁ F₂ →
  angle_F₁PF₂ P F₁ F₂ →
  let a := Real.sqrt ((F₁.1 - P.1)^2 + (F₁.2 - P.2)^2)
  let b := Real.sqrt ((F₂.1 - P.1)^2 + (F₂.2 - P.2)^2)
  let s := (a + b + 2 * Real.sqrt 7) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - 2 * Real.sqrt 7)) = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_l1413_141335


namespace NUMINAMATH_CALUDE_circular_garden_area_l1413_141398

theorem circular_garden_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : 
  let AD := AB / 2
  let R := (AD ^ 2 + DC ^ 2).sqrt
  π * R ^ 2 = 244 * π := by sorry

end NUMINAMATH_CALUDE_circular_garden_area_l1413_141398


namespace NUMINAMATH_CALUDE_factorization_equality_l1413_141342

theorem factorization_equality (x y z : ℝ) : x^2 + x*y - x*z - y*z = (x + y)*(x - z) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1413_141342


namespace NUMINAMATH_CALUDE_house_transaction_loss_l1413_141357

theorem house_transaction_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 →
  loss_percent = 0.15 →
  gain_percent = 0.20 →
  let first_sale := initial_value * (1 - loss_percent)
  let second_sale := first_sale * (1 + gain_percent)
  second_sale - initial_value = 2040 :=
by sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l1413_141357


namespace NUMINAMATH_CALUDE_voldemort_cake_calories_l1413_141300

/-- Calculates the calories of a cake given daily calorie limit, consumed calories, and remaining allowed calories. -/
def cake_calories (daily_limit : ℕ) (breakfast : ℕ) (lunch : ℕ) (chips : ℕ) (coke : ℕ) (remaining : ℕ) : ℕ :=
  daily_limit - (breakfast + lunch + chips + coke) - remaining

/-- Proves that the cake has 110 calories given Voldemort's calorie intake information. -/
theorem voldemort_cake_calories :
  cake_calories 2500 560 780 310 215 525 = 110 := by
  sorry

#eval cake_calories 2500 560 780 310 215 525

end NUMINAMATH_CALUDE_voldemort_cake_calories_l1413_141300


namespace NUMINAMATH_CALUDE_triangle_side_length_l1413_141392

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  b = 2 * Real.sqrt 7 →
  B = π / 3 →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1413_141392


namespace NUMINAMATH_CALUDE_arithmetic_triangle_sum_l1413_141388

-- Define a triangle with angles in arithmetic progression and side lengths 6, 7, and y
structure ArithmeticTriangle where
  y : ℝ
  angle_progression : ℝ → ℝ → ℝ → Prop
  side_lengths : ℝ → ℝ → ℝ → Prop

-- Define the sum of possible y values
def sum_of_y_values (t : ArithmeticTriangle) : ℝ := sorry

-- Define positive integers a, b, and c
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

-- Theorem statement
theorem arithmetic_triangle_sum :
  ∃ (t : ArithmeticTriangle),
    t.angle_progression 60 60 60 ∧
    t.side_lengths 6 7 t.y ∧
    sum_of_y_values t = a + Real.sqrt b + Real.sqrt c ∧
    a + b + c = 68 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_triangle_sum_l1413_141388


namespace NUMINAMATH_CALUDE_susan_spending_l1413_141338

def carnival_spending (initial_budget : ℝ) (food_cost : ℝ) : Prop :=
  let ride_cost : ℝ := 3 * food_cost
  let game_cost : ℝ := (1 / 4) * initial_budget
  let total_spent : ℝ := food_cost + ride_cost + game_cost
  let remaining : ℝ := initial_budget - total_spent
  remaining = 0

theorem susan_spending :
  carnival_spending 80 15 := by
  sorry

end NUMINAMATH_CALUDE_susan_spending_l1413_141338


namespace NUMINAMATH_CALUDE_basketball_score_formula_l1413_141320

/-- Represents the total points scored in a basketball game with specific conditions -/
def total_points (x : ℝ) : ℝ :=
  0.2 * x + 50

theorem basketball_score_formula (x y : ℝ) 
  (h1 : x + y = 50) 
  (h2 : x ≥ 0) 
  (h3 : y ≥ 0) : 
  0.4 * x * 3 + 0.5 * y * 2 = total_points x :=
sorry

end NUMINAMATH_CALUDE_basketball_score_formula_l1413_141320


namespace NUMINAMATH_CALUDE_new_person_weight_l1413_141362

theorem new_person_weight (initial_count : ℕ) (initial_average : ℝ) (weight_decrease : ℝ) :
  initial_count = 20 →
  initial_average = 60 →
  weight_decrease = 5 →
  let total_weight := initial_count * initial_average
  let new_count := initial_count + 1
  let new_average := initial_average - weight_decrease
  let new_person_weight := new_count * new_average - total_weight
  new_person_weight = 55 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l1413_141362


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1413_141385

/-- Given a complex number z and a real number m, 
    if z = (2+3i)(1-mi) is a pure imaginary number, then m = -2/3 -/
theorem pure_imaginary_condition (z : ℂ) (m : ℝ) : 
  z = (2 + 3*Complex.I) * (1 - m*Complex.I) ∧ z.re = 0 → m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1413_141385


namespace NUMINAMATH_CALUDE_book_distribution_l1413_141353

theorem book_distribution (total : ℕ) (books_A books_B : ℕ) : 
  total = 282 → 
  4 * books_A = 3 * total → 
  9 * books_B = 5 * total → 
  books_A + books_B = total → 
  books_A = 120 ∧ books_B = 162 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_l1413_141353


namespace NUMINAMATH_CALUDE_heaviest_coin_weighings_l1413_141301

/-- Represents a coin with a unique mass -/
structure Coin where
  mass : ℝ

/-- Represents a scale that can be faulty or not -/
structure Scale where
  isFaulty : Bool

/-- The minimum number of weighings needed to find the heaviest coin -/
def minWeighings (n : ℕ) : ℕ := 2 * n - 1

/-- Theorem stating the minimum number of weighings needed to find the heaviest coin -/
theorem heaviest_coin_weighings (n : ℕ) (coins : Fin n → Coin) (scales : Fin n → Scale) :
  n > 2 →
  (∃ i : Fin n, (scales i).isFaulty) →
  (∀ i j : Fin n, i ≠ j → (coins i).mass ≠ (coins j).mass) →
  minWeighings n = 2 * n - 1 :=
sorry

end NUMINAMATH_CALUDE_heaviest_coin_weighings_l1413_141301


namespace NUMINAMATH_CALUDE_whole_number_between_values_l1413_141343

theorem whole_number_between_values (N : ℤ) : 
  (6.75 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 7.25) → N = 28 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_values_l1413_141343


namespace NUMINAMATH_CALUDE_root_relation_l1413_141316

theorem root_relation (a b c : ℂ) (p q u : ℂ) : 
  (a^3 + 2*a^2 + 5*a - 8 = 0) → 
  (b^3 + 2*b^2 + 5*b - 8 = 0) → 
  (c^3 + 2*c^2 + 5*c - 8 = 0) → 
  ((a+b)^3 + p*(a+b)^2 + q*(a+b) + u = 0) → 
  ((b+c)^3 + p*(b+c)^2 + q*(b+c) + u = 0) → 
  ((c+a)^3 + p*(c+a)^2 + q*(c+a) + u = 0) → 
  u = 18 := by
sorry

end NUMINAMATH_CALUDE_root_relation_l1413_141316


namespace NUMINAMATH_CALUDE_coupon_value_l1413_141364

theorem coupon_value (total_spent peaches_after_coupon cherries : ℚ) : 
  total_spent = 23.86 →
  peaches_after_coupon = 12.32 →
  cherries = 11.54 →
  total_spent = peaches_after_coupon + cherries →
  0 = total_spent - (peaches_after_coupon + cherries) := by
sorry

end NUMINAMATH_CALUDE_coupon_value_l1413_141364


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_2000_l1413_141373

theorem greatest_multiple_of_four_cubed_less_than_2000 :
  ∃ (x : ℕ), x % 4 = 0 ∧ x^3 < 2000 ∧ ∀ (y : ℕ), y % 4 = 0 ∧ y^3 < 2000 → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_2000_l1413_141373


namespace NUMINAMATH_CALUDE_constant_grid_function_l1413_141322

/-- A function from integer pairs to non-negative integers -/
def GridFunction := ℤ × ℤ → ℕ

/-- The property that each value is the average of its four neighbors -/
def IsAverageOfNeighbors (f : GridFunction) : Prop :=
  ∀ x y : ℤ, 4 * f (x, y) = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)

/-- Theorem stating that if a grid function satisfies the average property, it is constant -/
theorem constant_grid_function (f : GridFunction) (h : IsAverageOfNeighbors f) :
  ∀ x₁ y₁ x₂ y₂ : ℤ, f (x₁, y₁) = f (x₂, y₂) := by
  sorry


end NUMINAMATH_CALUDE_constant_grid_function_l1413_141322
