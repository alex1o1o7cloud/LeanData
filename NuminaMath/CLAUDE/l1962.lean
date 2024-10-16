import Mathlib

namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l1962_196213

theorem cone_sphere_ratio (r h : ℝ) (h_pos : 0 < r) : 
  (1 / 3 : ℝ) * (4 / 3 * π * r^3) = (1 / 3 : ℝ) * π * r^2 * h → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l1962_196213


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l1962_196256

/-- The number of red jelly beans -/
def red_beans : ℕ := 10

/-- The number of green jelly beans -/
def green_beans : ℕ := 12

/-- The number of yellow jelly beans -/
def yellow_beans : ℕ := 13

/-- The number of blue jelly beans -/
def blue_beans : ℕ := 15

/-- The number of purple jelly beans -/
def purple_beans : ℕ := 5

/-- The total number of jelly beans -/
def total_beans : ℕ := red_beans + green_beans + yellow_beans + blue_beans + purple_beans

/-- The number of blue and purple jelly beans combined -/
def blue_and_purple : ℕ := blue_beans + purple_beans

/-- The probability of selecting either a blue or purple jelly bean -/
def probability : ℚ := blue_and_purple / total_beans

theorem jelly_bean_probability : probability = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l1962_196256


namespace NUMINAMATH_CALUDE_program_result_l1962_196215

def double_n_times (initial : ℕ) (n : ℕ) : ℕ :=
  initial * (2^n)

theorem program_result :
  double_n_times 1 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_program_result_l1962_196215


namespace NUMINAMATH_CALUDE_apple_difference_l1962_196237

theorem apple_difference (total : ℕ) (red : ℕ) (h1 : total = 44) (h2 : red = 16) :
  total > red → total - red - red = 12 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l1962_196237


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l1962_196287

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The main theorem -/
theorem arithmetic_geometric_sequence_product (a : ℕ → ℝ) :
  ArithmeticGeometricSequence a →
  a 1 = 3 →
  a 1 + a 3 + a 5 = 21 →
  a 2 * a 4 = 36 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l1962_196287


namespace NUMINAMATH_CALUDE_algebraic_identity_l1962_196290

theorem algebraic_identity (a b c d : ℝ) :
  (a^2 + b^2) * (a*b + c*d) - a*b * (a^2 + b^2 - c^2 - d^2) = (a*c + b*d) * (a*d + b*c) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l1962_196290


namespace NUMINAMATH_CALUDE_repeating_decimal_417_equals_fraction_sum_of_numerator_and_denominator_l1962_196230

/-- Represents a repeating decimal with a 3-digit repetend -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_417_equals_fraction :
  RepeatingDecimal 4 1 7 = 46 / 111 := by sorry

theorem sum_of_numerator_and_denominator :
  46 + 111 = 157 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_417_equals_fraction_sum_of_numerator_and_denominator_l1962_196230


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4704_l1962_196210

theorem largest_prime_factor_of_4704 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4704 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4704 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4704_l1962_196210


namespace NUMINAMATH_CALUDE_projection_matrix_values_l1962_196251

def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  Q * Q = Q

theorem projection_matrix_values :
  ∀ (a c : ℝ),
  let Q : Matrix (Fin 2) (Fin 2) ℝ := !![a, 18/45; c, 27/45]
  is_projection_matrix Q →
  a = 2/5 ∧ c = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l1962_196251


namespace NUMINAMATH_CALUDE_compare_fractions_l1962_196240

theorem compare_fractions (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / (1 + x + y) < x / (1 + x) + y / (1 + y) := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l1962_196240


namespace NUMINAMATH_CALUDE_felix_tree_chopping_l1962_196222

theorem felix_tree_chopping (trees_per_sharpen : ℕ) (sharpen_cost : ℕ) (total_spent : ℕ) : 
  trees_per_sharpen = 13 → 
  sharpen_cost = 5 → 
  total_spent = 35 → 
  ∃ (trees_chopped : ℕ), trees_chopped ≥ 91 ∧ trees_chopped ≥ (total_spent / sharpen_cost) * trees_per_sharpen :=
by
  sorry

end NUMINAMATH_CALUDE_felix_tree_chopping_l1962_196222


namespace NUMINAMATH_CALUDE_fraction_simplification_l1962_196249

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1962_196249


namespace NUMINAMATH_CALUDE_larger_number_problem_l1962_196223

theorem larger_number_problem (x y : ℝ) (h1 : x - y = 3) (h2 : x + y = 47) :
  max x y = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1962_196223


namespace NUMINAMATH_CALUDE_problem_1_problem_2a_problem_2b_problem_2c_l1962_196220

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2

-- State the theorems
theorem problem_1 : 27^(2/3) + Real.log 5 / Real.log 10 - 2 * Real.log 3 / Real.log 2 + Real.log 2 / Real.log 10 + Real.log 9 / Real.log 2 = 10 := by sorry

theorem problem_2a : f (-Real.sqrt 2) = 8 + 5 * Real.sqrt 2 := by sorry

theorem problem_2b (a : ℝ) : f (-a) = 3 * a^2 + 5 * a + 2 := by sorry

theorem problem_2c (a : ℝ) : f (a + 3) = 3 * a^2 + 13 * a + 14 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2a_problem_2b_problem_2c_l1962_196220


namespace NUMINAMATH_CALUDE_alberts_to_angelas_marbles_ratio_l1962_196228

/-- Proves that the ratio of Albert's marbles to Angela's marbles is 3:1 -/
theorem alberts_to_angelas_marbles_ratio (allison_marbles : ℕ) (angela_more_than_allison : ℕ) 
  (albert_and_allison_total : ℕ) 
  (h1 : allison_marbles = 28)
  (h2 : angela_more_than_allison = 8)
  (h3 : albert_and_allison_total = 136) : 
  (albert_and_allison_total - allison_marbles) / (allison_marbles + angela_more_than_allison) = 3 := by
  sorry

#check alberts_to_angelas_marbles_ratio

end NUMINAMATH_CALUDE_alberts_to_angelas_marbles_ratio_l1962_196228


namespace NUMINAMATH_CALUDE_remaining_document_arrangements_l1962_196286

/-- Represents the number of documents --/
def total_documents : ℕ := 12

/-- Represents the number of the processed document --/
def processed_document : ℕ := 10

/-- Calculates the number of possible arrangements for the remaining documents --/
def possible_arrangements : ℕ :=
  2 * (Nat.factorial 9 + 2 * Nat.factorial 10 + Nat.factorial 11)

/-- Theorem stating the number of possible ways to handle the remaining documents --/
theorem remaining_document_arrangements :
  possible_arrangements = 95116960 := by sorry

end NUMINAMATH_CALUDE_remaining_document_arrangements_l1962_196286


namespace NUMINAMATH_CALUDE_number_of_unique_sums_l1962_196264

/-- The set of numbers on the balls -/
def ball_numbers : Finset ℕ := {1, 2, 3, 4, 5}

/-- The set of all possible sums when drawing two balls with replacement -/
def possible_sums : Finset ℕ :=
  (ball_numbers.product ball_numbers).image (λ (x, y) => x + y)

/-- Theorem: The number of unique sums is 9 -/
theorem number_of_unique_sums : possible_sums.card = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_unique_sums_l1962_196264


namespace NUMINAMATH_CALUDE_coalition_percentage_is_79_percent_l1962_196211

/-- Represents the election results and voter information -/
structure ElectionData where
  total_votes : ℕ
  invalid_vote_percentage : ℚ
  registered_voters : ℕ
  candidate_x_valid_percentage : ℚ
  candidate_y_valid_percentage : ℚ
  candidate_z_valid_percentage : ℚ

/-- Calculates the percentage of valid votes received by a coalition of two candidates -/
def coalition_percentage (data : ElectionData) : ℚ :=
  data.candidate_x_valid_percentage + data.candidate_y_valid_percentage

/-- Theorem stating that the coalition of candidates X and Y received 79% of the valid votes -/
theorem coalition_percentage_is_79_percent (data : ElectionData)
  (h1 : data.total_votes = 750000)
  (h2 : data.invalid_vote_percentage = 18 / 100)
  (h3 : data.registered_voters = 900000)
  (h4 : data.candidate_x_valid_percentage = 47 / 100)
  (h5 : data.candidate_y_valid_percentage = 32 / 100)
  (h6 : data.candidate_z_valid_percentage = 21 / 100) :
  coalition_percentage data = 79 / 100 := by
  sorry


end NUMINAMATH_CALUDE_coalition_percentage_is_79_percent_l1962_196211


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1962_196298

/-- Calculates the total amount spent at a restaurant given the prices of items,
    discount rates, service fee rates, and tipping percentage. -/
def restaurant_bill (seafood_price rib_eye_price wine_price dessert_price : ℚ)
                    (wine_quantity : ℕ)
                    (food_discount service_fee_low service_fee_high tip_rate : ℚ) : ℚ :=
  let food_cost := seafood_price + rib_eye_price + dessert_price
  let wine_cost := wine_price * wine_quantity
  let total_before_discount := food_cost + wine_cost
  let discounted_food_cost := food_cost * (1 - food_discount)
  let after_discount := discounted_food_cost + wine_cost
  let service_fee_rate := if after_discount > 80 then service_fee_high else service_fee_low
  let service_fee := after_discount * service_fee_rate
  let total_after_service := after_discount + service_fee
  let tip := total_after_service * tip_rate
  total_after_service + tip

/-- The theorem states that given the specific prices and rates from the problem,
    the total amount spent at the restaurant is $167.67. -/
theorem restaurant_bill_calculation :
  restaurant_bill 45 38 18 12 2 0.1 0.12 0.15 0.2 = 167.67 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1962_196298


namespace NUMINAMATH_CALUDE_sample_capacity_proof_l1962_196271

/-- The sample capacity for a population of 36 individuals -/
def sample_capacity : ℕ := 6

theorem sample_capacity_proof :
  let total_population : ℕ := 36
  (sample_capacity ∣ total_population) ∧
  (6 ∣ sample_capacity) ∧
  ((total_population - 1) % (sample_capacity + 1) = 0) →
  sample_capacity = 6 :=
by sorry

end NUMINAMATH_CALUDE_sample_capacity_proof_l1962_196271


namespace NUMINAMATH_CALUDE_potato_percentage_l1962_196292

/-- Proves that the percentage of cleared land planted with potato is 30% -/
theorem potato_percentage (total_land : ℝ) (cleared_land : ℝ) (grape_land : ℝ) (tomato_land : ℝ) 
  (h1 : total_land = 3999.9999999999995)
  (h2 : cleared_land = 0.9 * total_land)
  (h3 : grape_land = 0.6 * cleared_land)
  (h4 : tomato_land = 360)
  : (cleared_land - grape_land - tomato_land) / cleared_land = 0.3 := by
  sorry

#eval (3999.9999999999995 * 0.9 - 3999.9999999999995 * 0.9 * 0.6 - 360) / (3999.9999999999995 * 0.9)

end NUMINAMATH_CALUDE_potato_percentage_l1962_196292


namespace NUMINAMATH_CALUDE_shelby_buys_three_posters_l1962_196268

/-- Calculates the number of posters Shelby can buy after her initial purchases and taxes --/
def posters_shelby_can_buy (initial_amount : ℚ) (book1_price : ℚ) (book2_price : ℚ) 
  (bookmark_price : ℚ) (pencils_price : ℚ) (tax_rate : ℚ) (poster_price : ℚ) : ℕ :=
  let total_before_tax := book1_price + book2_price + bookmark_price + pencils_price
  let total_with_tax := total_before_tax * (1 + tax_rate)
  let money_left := initial_amount - total_with_tax
  (money_left / poster_price).floor.toNat

/-- Theorem stating that Shelby can buy exactly 3 posters --/
theorem shelby_buys_three_posters : 
  posters_shelby_can_buy 50 12.50 7.25 2.75 3.80 0.07 5.50 = 3 := by
  sorry

end NUMINAMATH_CALUDE_shelby_buys_three_posters_l1962_196268


namespace NUMINAMATH_CALUDE_only_negative_option_l1962_196252

theorem only_negative_option (x : ℝ) : 
  (|(-1)| < 0 ∨ (-2^2) < 0 ∨ ((-Real.sqrt 3)^2) < 0 ∨ ((-3)^0) < 0) ↔ 
  (-2^2) < 0 :=
by sorry

end NUMINAMATH_CALUDE_only_negative_option_l1962_196252


namespace NUMINAMATH_CALUDE_sammys_offer_per_record_l1962_196272

theorem sammys_offer_per_record (total_records : ℕ) 
  (bryans_offer_high : ℕ) (bryans_offer_low : ℕ) (profit_difference : ℕ) :
  total_records = 200 →
  bryans_offer_high = 6 →
  bryans_offer_low = 1 →
  profit_difference = 100 →
  (total_records / 2 * bryans_offer_high + total_records / 2 * bryans_offer_low + profit_difference) / total_records = 4 := by
  sorry

end NUMINAMATH_CALUDE_sammys_offer_per_record_l1962_196272


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1962_196214

/-- Given a line segment CD where C(2,10) and M(6,6) is the midpoint of CD, 
    the sum of the coordinates of point D is 12. -/
theorem midpoint_coordinate_sum : 
  ∀ (D : ℝ × ℝ), 
  let C : ℝ × ℝ := (2, 10)
  let M : ℝ × ℝ := (6, 6)
  (M.1 = (C.1 + D.1) / 2 ∧ M.2 = (C.2 + D.2) / 2) → 
  D.1 + D.2 = 12 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1962_196214


namespace NUMINAMATH_CALUDE_count_initials_sets_l1962_196266

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def initials_length : ℕ := 4

/-- The number of different four-letter sets of initials possible using letters A through J -/
theorem count_initials_sets : (num_letters ^ initials_length : ℕ) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_count_initials_sets_l1962_196266


namespace NUMINAMATH_CALUDE_lunch_cost_theorem_l1962_196267

/-- The cost of the Taco Grande Plate -/
def taco_grande_cost : ℝ := 8

/-- The cost of Mike's additional items -/
def mike_additional_cost : ℝ := 2 + 4 + 2

/-- Mike's total bill -/
def mike_bill : ℝ := taco_grande_cost + mike_additional_cost

/-- John's total bill -/
def john_bill : ℝ := taco_grande_cost

/-- The combined total cost of Mike and John's lunch -/
def combined_total_cost : ℝ := mike_bill + john_bill

theorem lunch_cost_theorem :
  (mike_bill = 2 * john_bill) → combined_total_cost = 24 := by
  sorry

#eval combined_total_cost

end NUMINAMATH_CALUDE_lunch_cost_theorem_l1962_196267


namespace NUMINAMATH_CALUDE_hiring_theorem_l1962_196241

/-- Given probabilities for hiring three students A, B, and C --/
structure HiringProbabilities where
  probA : ℝ
  probNeitherANorB : ℝ
  probBothBAndC : ℝ

/-- The hiring probabilities satisfy the given conditions --/
def ValidHiringProbabilities (h : HiringProbabilities) : Prop :=
  h.probA = 2/3 ∧ h.probNeitherANorB = 1/12 ∧ h.probBothBAndC = 3/8

/-- Individual probabilities for B and C, and the probability of at least two being hired --/
structure HiringResults where
  probB : ℝ
  probC : ℝ
  probAtLeastTwo : ℝ

/-- The main theorem: given the conditions, prove the results --/
theorem hiring_theorem (h : HiringProbabilities) 
  (hvalid : ValidHiringProbabilities h) : 
  ∃ (r : HiringResults), r.probB = 3/4 ∧ r.probC = 1/2 ∧ r.probAtLeastTwo = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_hiring_theorem_l1962_196241


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_360_l1962_196254

/-- A rectangular table with 6 seats -/
structure RectangularTable :=
  (total_seats : ℕ)
  (longer_side_seats : ℕ)
  (shorter_side_seats : ℕ)
  (h_total : total_seats = 2 * longer_side_seats + 2 * shorter_side_seats)
  (h_longer : longer_side_seats = 2)
  (h_shorter : shorter_side_seats = 1)

/-- The number of ways to seat 5 persons at a rectangular table with 6 seats -/
def seating_arrangements (table : RectangularTable) (persons : ℕ) : ℕ := 
  3 * Nat.factorial (table.total_seats - 1)

/-- Theorem stating that the number of seating arrangements for 5 persons
    at the specified rectangular table is 360 -/
theorem seating_arrangements_eq_360 (table : RectangularTable) :
  seating_arrangements table 5 = 360 := by
  sorry

#eval seating_arrangements ⟨6, 2, 1, rfl, rfl, rfl⟩ 5

end NUMINAMATH_CALUDE_seating_arrangements_eq_360_l1962_196254


namespace NUMINAMATH_CALUDE_lcm_225_624_l1962_196204

theorem lcm_225_624 : Nat.lcm 225 624 = 46800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_225_624_l1962_196204


namespace NUMINAMATH_CALUDE_second_projectile_speed_l1962_196293

/-- Given two projectiles launched simultaneously from a distance apart, 
    with one traveling at a known speed, and both meeting after a certain time, 
    this theorem proves the speed of the second projectile. -/
theorem second_projectile_speed 
  (initial_distance : ℝ) 
  (speed_first : ℝ) 
  (time_to_meet : ℝ) 
  (h1 : initial_distance = 1998) 
  (h2 : speed_first = 444) 
  (h3 : time_to_meet = 2) : 
  ∃ (speed_second : ℝ), speed_second = 555 :=
by
  sorry

end NUMINAMATH_CALUDE_second_projectile_speed_l1962_196293


namespace NUMINAMATH_CALUDE_angle_from_terminal_point_l1962_196201

/-- Given an angle α in degrees where 0 ≤ α < 360, if a point on its terminal side
    has coordinates (sin 150°, cos 150°), then α = 300°. -/
theorem angle_from_terminal_point : ∀ α : ℝ,
  0 ≤ α → α < 360 →
  (∃ (x y : ℝ), x = Real.sin (150 * π / 180) ∧ y = Real.cos (150 * π / 180) ∧
    x = Real.sin (α * π / 180) ∧ y = Real.cos (α * π / 180)) →
  α = 300 := by
  sorry

end NUMINAMATH_CALUDE_angle_from_terminal_point_l1962_196201


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1962_196297

theorem absolute_value_inequality (b : ℝ) :
  (∃ x : ℝ, |x - 5| + |x - 7| < b) ↔ b > 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1962_196297


namespace NUMINAMATH_CALUDE_glen_animals_theorem_l1962_196217

theorem glen_animals_theorem (f t c r : ℕ) : 
  f = (5 * t) / 2 → 
  c = 3 * f → 
  r = 4 * c → 
  ∀ t : ℕ, f + t + c + r ≠ 108 :=
by
  sorry

end NUMINAMATH_CALUDE_glen_animals_theorem_l1962_196217


namespace NUMINAMATH_CALUDE_total_pennies_thrown_l1962_196253

/-- The number of pennies thrown by each person and their total --/
def penny_throwing (R G X M T : ℚ) : Prop :=
  R = 1500 ∧
  G = (2/3) * R ∧
  X = (3/4) * G ∧
  M = (7/2) * X ∧
  T = (4/5) * M ∧
  R + G + X + M + T = 7975

/-- Theorem stating that the total number of pennies thrown is 7975 --/
theorem total_pennies_thrown :
  ∃ (R G X M T : ℚ), penny_throwing R G X M T :=
sorry

end NUMINAMATH_CALUDE_total_pennies_thrown_l1962_196253


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1962_196216

/-- Proves that the rate of interest is 8% given the problem conditions --/
theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) (rate : ℝ) :
  principal = 1100 →
  interest_paid = 704 →
  interest_paid = principal * rate * rate / 100 →
  rate = 8 :=
by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l1962_196216


namespace NUMINAMATH_CALUDE_susans_bread_profit_l1962_196275

/-- Susan's bread selling problem -/
theorem susans_bread_profit :
  let total_loaves : ℕ := 60
  let cost_per_loaf : ℚ := 1
  let morning_price : ℚ := 3
  let afternoon_price : ℚ := 2
  let evening_price : ℚ := 3/2
  let morning_fraction : ℚ := 1/3
  let afternoon_fraction : ℚ := 1/2

  let morning_sales : ℚ := morning_fraction * total_loaves * morning_price
  let afternoon_sales : ℚ := afternoon_fraction * (total_loaves - morning_fraction * total_loaves) * afternoon_price
  let evening_sales : ℚ := (total_loaves - morning_fraction * total_loaves - afternoon_fraction * (total_loaves - morning_fraction * total_loaves)) * evening_price

  let total_revenue : ℚ := morning_sales + afternoon_sales + evening_sales
  let total_cost : ℚ := total_loaves * cost_per_loaf
  let profit : ℚ := total_revenue - total_cost

  profit = 70 := by sorry

end NUMINAMATH_CALUDE_susans_bread_profit_l1962_196275


namespace NUMINAMATH_CALUDE_four_numbers_theorem_l1962_196291

theorem four_numbers_theorem (a b c d : ℕ) : 
  a + b + c = 17 ∧ 
  a + b + d = 21 ∧ 
  a + c + d = 25 ∧ 
  b + c + d = 30 → 
  (a = 14 ∧ b = 10 ∧ c = 6 ∧ d = 1) ∨
  (a = 14 ∧ b = 10 ∧ c = 1 ∧ d = 6) ∨
  (a = 14 ∧ b = 6 ∧ c = 10 ∧ d = 1) ∨
  (a = 14 ∧ b = 6 ∧ c = 1 ∧ d = 10) ∨
  (a = 14 ∧ b = 1 ∧ c = 10 ∧ d = 6) ∨
  (a = 14 ∧ b = 1 ∧ c = 6 ∧ d = 10) ∨
  (a = 10 ∧ b = 14 ∧ c = 6 ∧ d = 1) ∨
  (a = 10 ∧ b = 14 ∧ c = 1 ∧ d = 6) ∨
  (a = 10 ∧ b = 6 ∧ c = 14 ∧ d = 1) ∨
  (a = 10 ∧ b = 6 ∧ c = 1 ∧ d = 14) ∨
  (a = 10 ∧ b = 1 ∧ c = 14 ∧ d = 6) ∨
  (a = 10 ∧ b = 1 ∧ c = 6 ∧ d = 14) ∨
  (a = 6 ∧ b = 14 ∧ c = 10 ∧ d = 1) ∨
  (a = 6 ∧ b = 14 ∧ c = 1 ∧ d = 10) ∨
  (a = 6 ∧ b = 10 ∧ c = 14 ∧ d = 1) ∨
  (a = 6 ∧ b = 10 ∧ c = 1 ∧ d = 14) ∨
  (a = 6 ∧ b = 1 ∧ c = 14 ∧ d = 10) ∨
  (a = 6 ∧ b = 1 ∧ c = 10 ∧ d = 14) ∨
  (a = 1 ∧ b = 14 ∧ c = 10 ∧ d = 6) ∨
  (a = 1 ∧ b = 14 ∧ c = 6 ∧ d = 10) ∨
  (a = 1 ∧ b = 10 ∧ c = 14 ∧ d = 6) ∨
  (a = 1 ∧ b = 10 ∧ c = 6 ∧ d = 14) ∨
  (a = 1 ∧ b = 6 ∧ c = 14 ∧ d = 10) ∨
  (a = 1 ∧ b = 6 ∧ c = 10 ∧ d = 14) :=
by sorry


end NUMINAMATH_CALUDE_four_numbers_theorem_l1962_196291


namespace NUMINAMATH_CALUDE_ticket_cost_l1962_196270

/-- Given that 7 tickets were purchased for a total of $308, prove that each ticket costs $44. -/
theorem ticket_cost (num_tickets : ℕ) (total_cost : ℕ) (h1 : num_tickets = 7) (h2 : total_cost = 308) :
  total_cost / num_tickets = 44 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_l1962_196270


namespace NUMINAMATH_CALUDE_number_solution_l1962_196208

theorem number_solution : ∃ x : ℝ, (3034 - (1002 / x) = 2984) ∧ x = 20.04 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l1962_196208


namespace NUMINAMATH_CALUDE_monomial_sum_implies_m_pow_n_eq_nine_l1962_196226

/-- If the sum of a^(m-1)b^2 and (1/2)a^2b^n is a monomial, then m^n = 9 -/
theorem monomial_sum_implies_m_pow_n_eq_nine 
  (a b : ℝ) (m n : ℕ) 
  (h : ∃ (k : ℝ) (p q : ℕ), a^(m-1) * b^2 + (1/2) * a^2 * b^n = k * a^p * b^q) :
  m^n = 9 := by
sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_m_pow_n_eq_nine_l1962_196226


namespace NUMINAMATH_CALUDE_scout_troop_profit_l1962_196227

-- Define the parameters
def num_bars : ℕ := 1500
def purchase_rate : ℚ := 1 / 3
def selling_rate : ℚ := 3 / 4
def fixed_cost : ℚ := 50

-- Define the profit calculation
def profit : ℚ :=
  num_bars * selling_rate - (num_bars * purchase_rate + fixed_cost)

-- Theorem statement
theorem scout_troop_profit : profit = 575 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_l1962_196227


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l1962_196289

/-- The number of ways to partition n indistinguishable balls into k indistinguishable boxes,
    with at least one ball in each box. -/
def partition_count (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are exactly 2 ways to partition 6 indistinguishable balls into 4 indistinguishable boxes,
    with at least one ball in each box. -/
theorem six_balls_four_boxes : partition_count 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l1962_196289


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_ending_in_seven_l1962_196276

theorem sum_of_arithmetic_sequence_ending_in_seven : 
  ∀ (a : ℕ) (d : ℕ) (n : ℕ),
    a = 107 → d = 10 → n = 40 →
    (a + (n - 1) * d = 497) →
    (n * (a + (a + (n - 1) * d))) / 2 = 12080 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_ending_in_seven_l1962_196276


namespace NUMINAMATH_CALUDE_chord_length_in_circle_l1962_196288

theorem chord_length_in_circle (r d : ℝ) (hr : r = 5) (hd : d = 3) :
  let half_chord := Real.sqrt (r^2 - d^2)
  2 * half_chord = 8 := by sorry

end NUMINAMATH_CALUDE_chord_length_in_circle_l1962_196288


namespace NUMINAMATH_CALUDE_olivia_initial_wallet_l1962_196273

/-- The amount of money Olivia spent at the supermarket -/
def amount_spent : ℕ := 15

/-- The amount of money Olivia has left after spending -/
def amount_left : ℕ := 63

/-- The initial amount of money in Olivia's wallet -/
def initial_amount : ℕ := amount_spent + amount_left

theorem olivia_initial_wallet : initial_amount = 78 := by
  sorry

end NUMINAMATH_CALUDE_olivia_initial_wallet_l1962_196273


namespace NUMINAMATH_CALUDE_savings_calculation_l1962_196258

/-- Given a person's income and expenditure ratio, and their total income, calculate their savings. -/
def calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (total_income : ℕ) : ℕ :=
  let total_ratio := income_ratio + expenditure_ratio
  let expenditure := (expenditure_ratio * total_income) / total_ratio
  total_income - expenditure

/-- Theorem stating that given the specific income-expenditure ratio and total income, 
    the savings amount to 7000. -/
theorem savings_calculation :
  calculate_savings 3 2 21000 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l1962_196258


namespace NUMINAMATH_CALUDE_tournament_matches_l1962_196218

/-- Represents a single-elimination tournament -/
structure Tournament where
  initial_teams : ℕ
  matches_played : ℕ

/-- The number of teams remaining after playing a certain number of matches -/
def teams_remaining (t : Tournament) : ℕ :=
  t.initial_teams - t.matches_played

theorem tournament_matches (t : Tournament) 
  (h1 : t.initial_teams = 128)
  (h2 : teams_remaining t = 1) : 
  t.matches_played = 127 := by
sorry

end NUMINAMATH_CALUDE_tournament_matches_l1962_196218


namespace NUMINAMATH_CALUDE_games_attended_l1962_196202

theorem games_attended (total : ℕ) (missed : ℕ) (attended : ℕ) : 
  total = 12 → missed = 7 → attended = total - missed → attended = 5 := by
  sorry

end NUMINAMATH_CALUDE_games_attended_l1962_196202


namespace NUMINAMATH_CALUDE_x_fourth_power_zero_l1962_196262

theorem x_fourth_power_zero (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (1 - x^2) + Real.sqrt (1 + x^2) = 2) : x^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_power_zero_l1962_196262


namespace NUMINAMATH_CALUDE_hidden_primes_average_l1962_196225

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def card_sum (visible hidden : ℕ) : ℕ := visible + hidden

theorem hidden_primes_average (h1 h2 h3 : ℕ) :
  is_prime h1 →
  is_prime h2 →
  is_prime h3 →
  card_sum 44 h1 = card_sum 59 h2 →
  card_sum 44 h1 = card_sum 38 h3 →
  (h1 + h2 + h3) / 3 = 14 :=
by sorry

end NUMINAMATH_CALUDE_hidden_primes_average_l1962_196225


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l1962_196299

theorem gcd_lcm_problem (A B : ℕ) (hA : A = 8 * 6) (hB : B = 36 / 3) :
  Nat.gcd A B = 12 ∧ Nat.lcm A B = 48 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l1962_196299


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1962_196245

theorem quadratic_equation_solution (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 7 * x - 20 = 0) : x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1962_196245


namespace NUMINAMATH_CALUDE_james_total_score_l1962_196274

theorem james_total_score (field_goals : ℕ) (two_point_shots : ℕ) 
  (h1 : field_goals = 13) (h2 : two_point_shots = 20) : 
  field_goals * 3 + two_point_shots * 2 = 79 := by
sorry

end NUMINAMATH_CALUDE_james_total_score_l1962_196274


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l1962_196224

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = √2, b = 2, and sin B + cos B = √2, then the measure of angle A is π/6. -/
theorem angle_measure_in_triangle (A B C : ℝ) (a b c : ℝ) :
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l1962_196224


namespace NUMINAMATH_CALUDE_problem_statement_l1962_196277

theorem problem_statement (m n : ℤ) (h : m * n = m + 3) : 
  3 * m - 3 * (m * n) + 10 = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1962_196277


namespace NUMINAMATH_CALUDE_constant_c_value_l1962_196294

theorem constant_c_value : ∃ (d e c : ℝ), 
  (∀ x : ℝ, (6*x^2 - 2*x + 5/2)*(d*x^2 + e*x + c) = 18*x^4 - 9*x^3 + 13*x^2 - 7/2*x + 15/4) →
  c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_constant_c_value_l1962_196294


namespace NUMINAMATH_CALUDE_triangle_circle_radii_l1962_196269

theorem triangle_circle_radii (a b c : ℝ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 8) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * area)
  let r := area / s
  R = (7 * Real.sqrt 3) / 3 ∧ r = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_circle_radii_l1962_196269


namespace NUMINAMATH_CALUDE_expression_evaluation_l1962_196261

theorem expression_evaluation : 5 * 12 + 2 * 15 - (3 * 7 + 4 * 6) = 45 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1962_196261


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1962_196279

/-- The circumference of the base of a right circular cone formed from a circular piece of paper 
    with radius 6 inches, after removing a 180-degree sector, is equal to 6π inches. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) : 
  let full_circumference := 2 * π * r
  let removed_angle := π  -- 180 degrees in radians
  let remaining_angle := 2 * π - removed_angle
  let base_circumference := (remaining_angle / (2 * π)) * full_circumference
  base_circumference = 6 * π := by
sorry


end NUMINAMATH_CALUDE_cone_base_circumference_l1962_196279


namespace NUMINAMATH_CALUDE_quadratic_properties_l1962_196209

-- Define the quadratic function
def y (m x : ℝ) : ℝ := 2*m*x^2 + (1-m)*x - 1 - m

-- Theorem statement
theorem quadratic_properties :
  -- 1. When m = -1, the vertex of the graph is at (1/2, 1/2)
  (y (-1) (1/2) = 1/2) ∧
  -- 2. When m > 0, the length of the segment intercepted by the graph on the x-axis is greater than 3/2
  (∀ m > 0, ∃ x₁ x₂, y m x₁ = 0 ∧ y m x₂ = 0 ∧ |x₁ - x₂| > 3/2) ∧
  -- 3. When m ≠ 0, the graph always passes through the fixed points (1, 0) and (-1/2, -3/2)
  (∀ m ≠ 0, y m 1 = 0 ∧ y m (-1/2) = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1962_196209


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_5_4_l1962_196248

/-- Represents the financial state of a person --/
structure FinancialState where
  income : ℕ
  savings : ℕ

/-- Calculates the expenditure given income and savings --/
def expenditure (fs : FinancialState) : ℕ :=
  fs.income - fs.savings

/-- Represents a ratio as a pair of natural numbers --/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Simplifies a ratio by dividing both parts by their GCD --/
def simplifyRatio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd,
    denominator := r.denominator / gcd }

/-- Calculates the ratio of income to expenditure --/
def incomeToExpenditureRatio (fs : FinancialState) : Ratio :=
  simplifyRatio { numerator := fs.income, denominator := expenditure fs }

theorem income_expenditure_ratio_5_4 (fs : FinancialState) 
  (h1 : fs.income = 15000) (h2 : fs.savings = 3000) : 
  incomeToExpenditureRatio fs = { numerator := 5, denominator := 4 } := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_5_4_l1962_196248


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l1962_196238

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- Returns the four-digit number formed by removing the digit at position i -/
def removeDigit (n : FiveDigitNumber) (i : Fin 5) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem unique_five_digit_number :
  ∃! (n : FiveDigitNumber),
    ∃ (i : Fin 5),
      n.val + removeDigit n i = 54321 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l1962_196238


namespace NUMINAMATH_CALUDE_b_range_l1962_196250

def P : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

def Q (b : ℝ) : Set ℝ := {x | x^2 - (b+2)*x + 2*b ≤ 0}

theorem b_range (b : ℝ) : P ⊇ Q b ↔ b ∈ Set.Icc 1 4 := by
  sorry

end NUMINAMATH_CALUDE_b_range_l1962_196250


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1962_196257

theorem sufficient_not_necessary_condition (a : ℝ) (h : a > 0) :
  (∀ a, a ≥ 1 → a + 1/a ≥ 2) ∧
  (∃ a, 0 < a ∧ a < 1 ∧ a + 1/a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1962_196257


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_14_l1962_196235

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the vertex of the original parabola
def vertex : ℝ × ℝ := (3, 4)

-- Define the transformed parabola after rotation and translation
def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 1

-- Define the zeros of the transformed parabola
def p : ℝ := 6
def q : ℝ := 8

theorem sum_of_zeros_is_14 : p + q = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_14_l1962_196235


namespace NUMINAMATH_CALUDE_rehabilitation_centers_count_rehabilitation_centers_count_proof_l1962_196282

theorem rehabilitation_centers_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun lisa jude han jane =>
    lisa = 6 ∧
    jude = lisa / 2 ∧
    han = 2 * jude - 2 ∧
    jane = 2 * han + 6 →
    lisa + jude + han + jane = 27

#check rehabilitation_centers_count

-- The proof is omitted
theorem rehabilitation_centers_count_proof :
  ∃ (lisa jude han jane : ℕ),
    rehabilitation_centers_count lisa jude han jane :=
sorry

end NUMINAMATH_CALUDE_rehabilitation_centers_count_rehabilitation_centers_count_proof_l1962_196282


namespace NUMINAMATH_CALUDE_boxes_delivered_to_orphanage_l1962_196229

def total_lemon_cupcakes : ℕ := 53
def total_chocolate_cupcakes : ℕ := 76
def lemon_cupcakes_left_at_home : ℕ := 7
def chocolate_cupcakes_left_at_home : ℕ := 8
def cupcakes_per_box : ℕ := 5

def lemon_cupcakes_delivered : ℕ := total_lemon_cupcakes - lemon_cupcakes_left_at_home
def chocolate_cupcakes_delivered : ℕ := total_chocolate_cupcakes - chocolate_cupcakes_left_at_home

def total_cupcakes_delivered : ℕ := lemon_cupcakes_delivered + chocolate_cupcakes_delivered

theorem boxes_delivered_to_orphanage :
  (total_cupcakes_delivered / cupcakes_per_box : ℕ) +
  (if total_cupcakes_delivered % cupcakes_per_box > 0 then 1 else 0) = 23 :=
by sorry

end NUMINAMATH_CALUDE_boxes_delivered_to_orphanage_l1962_196229


namespace NUMINAMATH_CALUDE_toy_difference_l1962_196265

/-- The number of toys each person has -/
structure ToyCount where
  mandy : ℕ
  anna : ℕ
  amanda : ℕ

/-- The conditions of the problem -/
def ProblemConditions (tc : ToyCount) : Prop :=
  tc.mandy = 20 ∧
  tc.anna = 3 * tc.mandy ∧
  tc.mandy + tc.anna + tc.amanda = 142 ∧
  tc.amanda > tc.anna

/-- The theorem to be proved -/
theorem toy_difference (tc : ToyCount) (h : ProblemConditions tc) : 
  tc.amanda - tc.anna = 2 := by
  sorry

end NUMINAMATH_CALUDE_toy_difference_l1962_196265


namespace NUMINAMATH_CALUDE_polynomial_roots_l1962_196263

theorem polynomial_roots : ∃ (a b c d : ℂ),
  (a = (1 + Real.sqrt 5) / 2) ∧
  (b = (1 - Real.sqrt 5) / 2) ∧
  (c = (3 + Real.sqrt 13) / 6) ∧
  (d = (3 - Real.sqrt 13) / 6) ∧
  (∀ x : ℂ, 3 * x^4 - 4 * x^3 - 5 * x^2 - 4 * x + 3 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1962_196263


namespace NUMINAMATH_CALUDE_negation_of_implication_l1962_196232

theorem negation_of_implication (A B : Set α) (a b : α) :
  ¬(a ∉ A → b ∈ B) ↔ (a ∉ A ∧ b ∉ B) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1962_196232


namespace NUMINAMATH_CALUDE_walkway_time_proof_l1962_196219

theorem walkway_time_proof (walkway_length : ℝ) (time_against : ℝ) (time_stationary : ℝ)
  (h1 : walkway_length = 80)
  (h2 : time_against = 120)
  (h3 : time_stationary = 60) :
  let person_speed := walkway_length / time_stationary
  let walkway_speed := person_speed - walkway_length / time_against
  walkway_length / (person_speed + walkway_speed) = 40 := by
sorry

end NUMINAMATH_CALUDE_walkway_time_proof_l1962_196219


namespace NUMINAMATH_CALUDE_det_A_l1962_196296

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 0, 2; 8, 5, -1; 3, 3, 7]

theorem det_A : A.det = 132 := by sorry

end NUMINAMATH_CALUDE_det_A_l1962_196296


namespace NUMINAMATH_CALUDE_prob_three_same_color_l1962_196231

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- The probability of drawing three cards of the same color from a standard deck -/
theorem prob_three_same_color (d : Deck) (h : d = standard_deck) :
  (d.red_suits * d.cards_per_suit).choose 3 / d.total_cards.choose 3 +
  (d.black_suits * d.cards_per_suit).choose 3 / d.total_cards.choose 3 = 40 / 85 :=
sorry

end NUMINAMATH_CALUDE_prob_three_same_color_l1962_196231


namespace NUMINAMATH_CALUDE_ring_arrangements_value_l1962_196236

/-- The number of possible 6-ring arrangements on 4 fingers, given 10 distinguishable rings,
    with no more than 2 rings per finger. -/
def ring_arrangements : ℕ :=
  let total_rings : ℕ := 10
  let fingers : ℕ := 4
  let rings_to_arrange : ℕ := 6
  let max_rings_per_finger : ℕ := 2
  
  let ways_to_choose_rings : ℕ := Nat.choose total_rings rings_to_arrange
  let ways_to_distribute_rings : ℕ := Nat.choose (rings_to_arrange + fingers - 1) (fingers - 1) -
    fingers * Nat.choose (rings_to_arrange - max_rings_per_finger - 1 + fingers - 1) (fingers - 1)
  let ways_to_order_rings : ℕ := Nat.factorial rings_to_arrange

  ways_to_choose_rings * ways_to_distribute_rings * ways_to_order_rings

theorem ring_arrangements_value : ring_arrangements = 604800 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangements_value_l1962_196236


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1962_196280

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1962_196280


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1962_196221

theorem cubic_equation_roots (p q : ℝ) :
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
   ∀ x : ℝ, x^3 - 11*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  p + q = 78 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1962_196221


namespace NUMINAMATH_CALUDE_product_increase_by_three_times_l1962_196239

theorem product_increase_by_three_times : 
  ∃ (a b c d : ℕ), (a + 1) * (b + 1) * (c + 1) * (d + 1) = 3 * (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_product_increase_by_three_times_l1962_196239


namespace NUMINAMATH_CALUDE_transportation_theorem_l1962_196234

/-- Represents a type of transportation with its quantity and number of wheels -/
structure Transportation where
  name : String
  quantity : Nat
  wheels : Nat

/-- Calculates the total number of wheels for a given transportation -/
def totalWheels (t : Transportation) : Nat :=
  t.quantity * t.wheels

/-- Calculates the total number of wheels for a list of transportations -/
def sumWheels (ts : List Transportation) : Nat :=
  ts.foldl (fun acc t => acc + totalWheels t) 0

/-- Calculates the total quantity of all transportations -/
def totalQuantity (ts : List Transportation) : Nat :=
  ts.foldl (fun acc t => acc + t.quantity) 0

/-- Calculates the quantity of bicycles and tricycles -/
def bikeAndTricycleCount (ts : List Transportation) : Nat :=
  ts.filter (fun t => t.name = "bicycle" || t.name = "tricycle")
    |>.foldl (fun acc t => acc + t.quantity) 0

theorem transportation_theorem (observations : List Transportation) 
  (h1 : observations = [
    ⟨"car", 15, 4⟩, 
    ⟨"bicycle", 3, 2⟩, 
    ⟨"pickup truck", 8, 4⟩, 
    ⟨"tricycle", 1, 3⟩, 
    ⟨"motorcycle", 4, 2⟩, 
    ⟨"skateboard", 2, 4⟩, 
    ⟨"unicycle", 1, 1⟩
  ]) : 
  sumWheels observations = 118 ∧ 
  (bikeAndTricycleCount observations : Rat) / (totalQuantity observations : Rat) = 4/34 := by
  sorry

end NUMINAMATH_CALUDE_transportation_theorem_l1962_196234


namespace NUMINAMATH_CALUDE_tan_arccos_three_fifths_l1962_196242

theorem tan_arccos_three_fifths : Real.tan (Real.arccos (3/5)) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_arccos_three_fifths_l1962_196242


namespace NUMINAMATH_CALUDE_sphere_cube_intersection_areas_l1962_196207

/-- Given a cube with edge length a and a sphere circumscribed around it, 
    this theorem proves the areas of the sections formed by the intersection 
    of the sphere and the cube's faces. -/
theorem sphere_cube_intersection_areas (a : ℝ) (h : a > 0) :
  let R := a * Real.sqrt 3 / 2
  ∃ (bicorn_area curvilinear_quad_area : ℝ),
    bicorn_area = π * a^2 * (2 - Real.sqrt 3) / 4 ∧
    curvilinear_quad_area = π * a^2 * (Real.sqrt 3 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_cube_intersection_areas_l1962_196207


namespace NUMINAMATH_CALUDE_chess_pawns_remaining_l1962_196205

theorem chess_pawns_remaining (initial_pawns : ℕ) 
  (kennedy_lost : ℕ) (riley_lost : ℕ) : 
  initial_pawns = 8 → kennedy_lost = 4 → riley_lost = 1 →
  (initial_pawns - kennedy_lost) + (initial_pawns - riley_lost) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_pawns_remaining_l1962_196205


namespace NUMINAMATH_CALUDE_max_intersection_points_l1962_196203

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 20

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 10

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := 8550

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points :
  (num_x_points.choose 2) * (num_y_points.choose 2) = max_intersections :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_l1962_196203


namespace NUMINAMATH_CALUDE_carpet_cost_calculation_carpet_cost_result_l1962_196246

/-- Calculate the cost of a carpet with increased dimensions -/
theorem carpet_cost_calculation (breadth_1 : Real) (length_ratio : Real) 
  (length_increase : Real) (breadth_increase : Real) (rate : Real) : Real :=
  let length_1 := breadth_1 * length_ratio
  let breadth_2 := breadth_1 * (1 + breadth_increase)
  let length_2 := length_1 * (1 + length_increase)
  let area_2 := breadth_2 * length_2
  area_2 * rate

/-- The cost of the carpet with specified dimensions and rate -/
theorem carpet_cost_result : 
  carpet_cost_calculation 6 1.44 0.4 0.25 45 = 4082.4 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_calculation_carpet_cost_result_l1962_196246


namespace NUMINAMATH_CALUDE_equal_roots_condition_no_three_equal_values_same_solutions_condition_l1962_196295

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Statement ①
theorem equal_roots_condition (a b c : ℝ) (h : a ≠ 0) :
  b^2 - 4*a*c = 0 → ∃! x : ℝ, quadratic a b c x = 0 :=
sorry

-- Statement ②
theorem no_three_equal_values (a b c : ℝ) (h : a ≠ 0) :
  ¬∃ (m n s : ℝ), m ≠ n ∧ n ≠ s ∧ m ≠ s ∧
    quadratic a b c m = quadratic a b c n ∧
    quadratic a b c n = quadratic a b c s :=
sorry

-- Statement ③
theorem same_solutions_condition (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, quadratic a b c x + 2 = 0 ↔ (x + 2) * (x - 3) = 0) →
  4*a - 2*b + c = -2 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_no_three_equal_values_same_solutions_condition_l1962_196295


namespace NUMINAMATH_CALUDE_stone_piles_total_l1962_196243

theorem stone_piles_total (pile1 pile2 pile3 pile4 pile5 : ℕ) : 
  pile5 = 6 * pile3 →
  pile2 = 2 * (pile3 + pile5) →
  pile1 * 3 = pile5 →
  pile1 + 10 = pile4 →
  2 * pile4 = pile2 →
  pile1 + pile2 + pile3 + pile4 + pile5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stone_piles_total_l1962_196243


namespace NUMINAMATH_CALUDE_book_pages_calculation_l1962_196284

/-- The number of pages Sally reads on weekdays -/
def weekday_pages : ℕ := 10

/-- The number of pages Sally reads on weekends -/
def weekend_pages : ℕ := 20

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- The total number of pages in the book -/
def total_pages : ℕ := 180

theorem book_pages_calculation :
  total_pages = 
    weeks_to_finish * (weekdays_per_week * weekday_pages + weekend_days_per_week * weekend_pages) :=
by sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l1962_196284


namespace NUMINAMATH_CALUDE_area_enclosed_by_trajectory_l1962_196283

def f (x : ℝ) : ℝ := x^2 + 1

theorem area_enclosed_by_trajectory (a b : ℝ) (h1 : a < b) 
  (h2 : Set.range f = Set.Icc 1 5) 
  (h3 : Set.Icc a b = f⁻¹' (Set.Icc 1 5)) : 
  (b - a) * 1 = 4 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_trajectory_l1962_196283


namespace NUMINAMATH_CALUDE_tom_calorie_consumption_l1962_196260

/-- Calculates the total calories consumed by Tom given the weight and calorie content of carrots and broccoli. -/
def total_calories (carrot_weight : ℝ) (broccoli_weight : ℝ) (carrot_calories : ℝ) (broccoli_calories : ℝ) : ℝ :=
  carrot_weight * carrot_calories + broccoli_weight * broccoli_calories

/-- Theorem stating that Tom's total calorie consumption is 85 given the problem conditions. -/
theorem tom_calorie_consumption :
  let carrot_weight : ℝ := 1
  let broccoli_weight : ℝ := 2 * carrot_weight
  let carrot_calories : ℝ := 51
  let broccoli_calories : ℝ := (1/3) * carrot_calories
  total_calories carrot_weight broccoli_weight carrot_calories broccoli_calories = 85 := by
  sorry

end NUMINAMATH_CALUDE_tom_calorie_consumption_l1962_196260


namespace NUMINAMATH_CALUDE_nadia_mistakes_l1962_196278

/-- Represents Nadia's piano playing statistics -/
structure PianoStats where
  mistakes_per_40_notes : ℕ
  notes_per_minute : ℕ
  playing_time : ℕ

/-- Calculates the number of mistakes Nadia makes given her piano playing statistics -/
def calculate_mistakes (stats : PianoStats) : ℕ :=
  let total_notes := stats.notes_per_minute * stats.playing_time
  let blocks_of_40 := total_notes / 40
  blocks_of_40 * stats.mistakes_per_40_notes

/-- Theorem stating that Nadia makes 36 mistakes in 8 minutes of playing -/
theorem nadia_mistakes (stats : PianoStats)
  (h1 : stats.mistakes_per_40_notes = 3)
  (h2 : stats.notes_per_minute = 60)
  (h3 : stats.playing_time = 8) :
  calculate_mistakes stats = 36 := by
  sorry


end NUMINAMATH_CALUDE_nadia_mistakes_l1962_196278


namespace NUMINAMATH_CALUDE_workshop_probability_l1962_196244

def total_students : ℕ := 30
def painting_students : ℕ := 22
def sculpting_students : ℕ := 24

theorem workshop_probability : 
  let both_workshops := painting_students + sculpting_students - total_students
  let painting_only := painting_students - both_workshops
  let sculpting_only := sculpting_students - both_workshops
  let total_combinations := total_students.choose 2
  let not_both_workshops := (painting_only.choose 2) + (sculpting_only.choose 2)
  (total_combinations - not_both_workshops : ℚ) / total_combinations = 56 / 62 :=
by sorry

end NUMINAMATH_CALUDE_workshop_probability_l1962_196244


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1962_196200

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m * (m + 2)) / (m - 1) = 0 → m = 0 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1962_196200


namespace NUMINAMATH_CALUDE_egg_distribution_l1962_196212

theorem egg_distribution (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) : 
  total_eggs = 8 → num_groups = 4 → eggs_per_group = total_eggs / num_groups → eggs_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_egg_distribution_l1962_196212


namespace NUMINAMATH_CALUDE_gcd_50420_35313_l1962_196285

theorem gcd_50420_35313 : Nat.gcd 50420 35313 = 19 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50420_35313_l1962_196285


namespace NUMINAMATH_CALUDE_interest_difference_proof_l1962_196281

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates the difference between principal and interest -/
def principalInterestDifference (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal - simpleInterest principal rate time

theorem interest_difference_proof :
  let principal : ℝ := 1100
  let rate : ℝ := 0.06
  let time : ℝ := 8
  principalInterestDifference principal rate time = 572 := by
sorry

end NUMINAMATH_CALUDE_interest_difference_proof_l1962_196281


namespace NUMINAMATH_CALUDE_nine_appears_once_l1962_196255

def multiply_987654321_by_9 : ℕ := 987654321 * 9

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  n.digits 10
    |>.filter (· = d)
    |>.length

theorem nine_appears_once :
  count_digit multiply_987654321_by_9 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_nine_appears_once_l1962_196255


namespace NUMINAMATH_CALUDE_lexie_family_age_difference_l1962_196233

/-- Given Lexie's age, calculate the age difference between her brother and sister -/
def age_difference (lexie_age : ℕ) : ℕ :=
  let brother_age := lexie_age - 6
  let sister_age := lexie_age * 2
  sister_age - brother_age

/-- Theorem stating the age difference between Lexie's brother and sister -/
theorem lexie_family_age_difference :
  age_difference 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_lexie_family_age_difference_l1962_196233


namespace NUMINAMATH_CALUDE_cashew_price_in_mixture_l1962_196206

/-- The price per pound of cashews in a mixture with peanuts -/
def cashew_price (peanut_price : ℚ) (total_weight : ℚ) (total_value : ℚ) (cashew_weight : ℚ) : ℚ :=
  (total_value - (total_weight - cashew_weight) * peanut_price) / cashew_weight

/-- Theorem stating the price of cashews in the given mixture -/
theorem cashew_price_in_mixture :
  cashew_price 2 25 92 11 = 64/11 := by
  sorry

end NUMINAMATH_CALUDE_cashew_price_in_mixture_l1962_196206


namespace NUMINAMATH_CALUDE_dog_bones_problem_l1962_196259

theorem dog_bones_problem (buried_bones initial_bones final_bones : ℚ) : 
  buried_bones = 367.5 ∧ 
  final_bones = -860 ∧ 
  initial_bones - buried_bones = final_bones → 
  initial_bones = 367.5 := by
sorry


end NUMINAMATH_CALUDE_dog_bones_problem_l1962_196259


namespace NUMINAMATH_CALUDE_min_difference_is_one_l1962_196247

/-- Triangle with integer side lengths and specific properties -/
structure IntegerTriangle where
  DE : ℕ
  EF : ℕ
  FD : ℕ
  perimeter_eq : DE + EF + FD = 398
  side_order : DE < EF ∧ EF ≤ FD

/-- The minimum difference between EF and DE in an IntegerTriangle is 1 -/
theorem min_difference_is_one :
  ∀ t : IntegerTriangle, (∀ s : IntegerTriangle, t.EF - t.DE ≤ s.EF - s.DE) → t.EF - t.DE = 1 := by
  sorry

#check min_difference_is_one

end NUMINAMATH_CALUDE_min_difference_is_one_l1962_196247
