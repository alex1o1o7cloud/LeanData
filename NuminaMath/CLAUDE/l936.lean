import Mathlib

namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l936_93633

/-- Given a circle with circumference 24 cm, its area is 144/π square centimeters. -/
theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 24 → π * r^2 = 144 / π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l936_93633


namespace NUMINAMATH_CALUDE_brand_a_most_cost_effective_l936_93609

/-- Represents a chocolate bar brand with its price and s'mores per bar -/
structure ChocolateBar where
  price : ℝ
  smoresPerBar : ℕ

/-- Calculates the cost of chocolate bars for a given number of s'mores -/
def calculateCost (bar : ChocolateBar) (numSmores : ℕ) : ℝ :=
  let numBars := (numSmores + bar.smoresPerBar - 1) / bar.smoresPerBar
  let cost := numBars * bar.price
  if numBars ≥ 10 then cost * 0.85 else cost

/-- Proves that Brand A is the most cost-effective option for Ron's scout camp -/
theorem brand_a_most_cost_effective :
  let numScouts : ℕ := 15
  let smoresPerScout : ℕ := 2
  let brandA := ChocolateBar.mk 1.50 3
  let brandB := ChocolateBar.mk 2.10 4
  let brandC := ChocolateBar.mk 3.00 6
  let totalSmores := numScouts * smoresPerScout
  let costA := calculateCost brandA totalSmores
  let costB := calculateCost brandB totalSmores
  let costC := calculateCost brandC totalSmores
  (costA < costB ∧ costA < costC) ∧ costA = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_brand_a_most_cost_effective_l936_93609


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l936_93683

/-- Given a triangle with sides 9, 12, and 15 units and a rectangle with width 6 units
    and area equal to the triangle's area, the perimeter of the rectangle is 30 units. -/
theorem rectangle_perimeter (triangle_side1 triangle_side2 triangle_side3 rectangle_width : ℝ) :
  triangle_side1 = 9 ∧ triangle_side2 = 12 ∧ triangle_side3 = 15 ∧ rectangle_width = 6 ∧
  (1/2 * triangle_side1 * triangle_side2 = rectangle_width * (1/2 * triangle_side1 * triangle_side2 / rectangle_width)) →
  2 * (rectangle_width + (1/2 * triangle_side1 * triangle_side2 / rectangle_width)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l936_93683


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l936_93663

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 2*x < 3} = {x : ℝ | -3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l936_93663


namespace NUMINAMATH_CALUDE_division_problem_l936_93631

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 5 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 113 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l936_93631


namespace NUMINAMATH_CALUDE_vacation_probability_l936_93634

theorem vacation_probability (prob_A prob_B : ℝ) 
  (h1 : prob_A = 1/4)
  (h2 : prob_B = 1/5)
  (h3 : 0 ≤ prob_A ∧ prob_A ≤ 1)
  (h4 : 0 ≤ prob_B ∧ prob_B ≤ 1) :
  1 - (1 - prob_A) * (1 - prob_B) = 2/5 := by
sorry

end NUMINAMATH_CALUDE_vacation_probability_l936_93634


namespace NUMINAMATH_CALUDE_no_real_roots_l936_93637

theorem no_real_roots :
  ∀ x : ℝ, ¬(Real.sqrt (x + 9) - Real.sqrt (x - 5) + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_l936_93637


namespace NUMINAMATH_CALUDE_num_adoption_ways_l936_93693

/-- The number of parrots available for adoption -/
def num_parrots : ℕ := 20

/-- The number of snakes available for adoption -/
def num_snakes : ℕ := 10

/-- The number of rabbits available for adoption -/
def num_rabbits : ℕ := 12

/-- The set of possible animal types -/
inductive AnimalType
| Parrot
| Snake
| Rabbit

/-- A function representing Emily's constraint -/
def emily_constraint (a : AnimalType) : Prop :=
  a = AnimalType.Parrot ∨ a = AnimalType.Rabbit

/-- A function representing John's constraint (can adopt any animal) -/
def john_constraint (a : AnimalType) : Prop := True

/-- A function representing Susan's constraint -/
def susan_constraint (a : AnimalType) : Prop :=
  a = AnimalType.Snake

/-- The theorem stating the number of ways to adopt animals -/
theorem num_adoption_ways :
  (num_parrots * num_snakes * num_rabbits) +
  (num_rabbits * num_snakes * num_parrots) = 4800 := by
  sorry

end NUMINAMATH_CALUDE_num_adoption_ways_l936_93693


namespace NUMINAMATH_CALUDE_a_equals_2_sufficient_not_necessary_l936_93662

def third_term (a : ℝ) : ℝ → ℝ := λ x ↦ 15 * a^2 * x^4

theorem a_equals_2_sufficient_not_necessary :
  (∀ x, third_term 2 x = 60 * x^4) ∧
  (∃ a ≠ 2, ∀ x, third_term a x = 60 * x^4) :=
sorry

end NUMINAMATH_CALUDE_a_equals_2_sufficient_not_necessary_l936_93662


namespace NUMINAMATH_CALUDE_flour_weight_qualified_l936_93622

def is_qualified (weight : ℝ) : Prop :=
  24.75 ≤ weight ∧ weight ≤ 25.25

theorem flour_weight_qualified :
  is_qualified 24.80 := by sorry

end NUMINAMATH_CALUDE_flour_weight_qualified_l936_93622


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l936_93645

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ x₀^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l936_93645


namespace NUMINAMATH_CALUDE_cubic_factorization_l936_93696

theorem cubic_factorization :
  ∀ x : ℝ, 343 * x^3 + 125 = (7 * x + 5) * (49 * x^2 - 35 * x + 25) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l936_93696


namespace NUMINAMATH_CALUDE_devin_basketball_chance_l936_93638

/-- Represents the chance of making the basketball team based on height -/
def basketballChance (initialHeight : ℕ) (growth : ℕ) : ℝ :=
  let baseHeight : ℕ := 66
  let baseChance : ℝ := 0.1
  let chanceIncreasePerInch : ℝ := 0.1
  let finalHeight : ℕ := initialHeight + growth
  let additionalInches : ℕ := max (finalHeight - baseHeight) 0
  baseChance + (additionalInches : ℝ) * chanceIncreasePerInch

/-- Theorem stating Devin's chance of making the team after growing -/
theorem devin_basketball_chance :
  basketballChance 65 3 = 0.3 := by
  sorry

#eval basketballChance 65 3

end NUMINAMATH_CALUDE_devin_basketball_chance_l936_93638


namespace NUMINAMATH_CALUDE_dot_product_equals_eight_l936_93669

def a : Fin 2 → ℝ := ![0, 4]
def b : Fin 2 → ℝ := ![2, 2]

theorem dot_product_equals_eight :
  (Finset.univ.sum (λ i => a i * b i)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equals_eight_l936_93669


namespace NUMINAMATH_CALUDE_fourDigitNumbers_eq_14_l936_93686

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of 4-digit numbers formed using digits 2 and 3, where each number must include at least one occurrence of both digits -/
def fourDigitNumbers : ℕ :=
  choose 4 1 + choose 4 2 + choose 4 3

theorem fourDigitNumbers_eq_14 : fourDigitNumbers = 14 := by sorry

end NUMINAMATH_CALUDE_fourDigitNumbers_eq_14_l936_93686


namespace NUMINAMATH_CALUDE_min_value_theorem_l936_93635

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : (a + b) * b * c = 5) : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y) * y * z = 5 → 2 * a + b + c ≤ 2 * x + y + z ∧
  2 * a + b + c = 2 * Real.sqrt 5 :=
by sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_min_value_theorem_l936_93635


namespace NUMINAMATH_CALUDE_fried_chicken_dinner_orders_l936_93678

/-- Represents the number of pieces of chicken used in different order types -/
structure ChickenPieces where
  pasta : Nat
  barbecue : Nat
  friedDinner : Nat

/-- Represents the number of orders for each type -/
structure Orders where
  pasta : Nat
  barbecue : Nat
  friedDinner : Nat

/-- Calculates the total number of chicken pieces used -/
def totalChickenPieces (cp : ChickenPieces) (o : Orders) : Nat :=
  cp.pasta * o.pasta + cp.barbecue * o.barbecue + cp.friedDinner * o.friedDinner

/-- The main theorem to prove -/
theorem fried_chicken_dinner_orders
  (cp : ChickenPieces)
  (o : Orders)
  (h1 : cp.pasta = 2)
  (h2 : cp.barbecue = 3)
  (h3 : cp.friedDinner = 8)
  (h4 : o.pasta = 6)
  (h5 : o.barbecue = 3)
  (h6 : totalChickenPieces cp o = 37) :
  o.friedDinner = 2 := by
  sorry

end NUMINAMATH_CALUDE_fried_chicken_dinner_orders_l936_93678


namespace NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_13_l936_93607

theorem sum_seven_smallest_multiples_of_13 : 
  (Finset.range 7).sum (fun i => 13 * (i + 1)) = 364 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_13_l936_93607


namespace NUMINAMATH_CALUDE_inequality_equivalence_l936_93674

theorem inequality_equivalence (x : ℝ) : -3 * x^2 + 8 * x + 1 < 0 ↔ -1/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l936_93674


namespace NUMINAMATH_CALUDE_marcus_baseball_cards_l936_93665

theorem marcus_baseball_cards 
  (initial_cards : ℝ) 
  (cards_from_carter : ℝ) 
  (h1 : initial_cards = 210.0) 
  (h2 : cards_from_carter = 58.0) : 
  initial_cards + cards_from_carter = 268.0 := by
sorry

end NUMINAMATH_CALUDE_marcus_baseball_cards_l936_93665


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l936_93682

def z : ℂ := (3 - Complex.I) * (1 + Complex.I)

theorem z_in_first_quadrant : z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l936_93682


namespace NUMINAMATH_CALUDE_cone_volume_l936_93625

/-- Given a cone with slant height 3 and lateral surface area 3π, its volume is (2√2π)/3 -/
theorem cone_volume (l : ℝ) (A_L : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) : 
  l = 3 →
  A_L = 3 * Real.pi →
  A_L = Real.pi * r * l →
  l^2 = h^2 + r^2 →
  V = (1/3) * Real.pi * r^2 * h →
  V = (2 * Real.sqrt 2 * Real.pi) / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_l936_93625


namespace NUMINAMATH_CALUDE_common_root_sum_k_l936_93684

theorem common_root_sum_k : ∃ (k₁ k₂ : ℝ),
  (∃ x : ℝ, x^2 - 4*x + 3 = 0 ∧ x^2 - 6*x + k₁ = 0) ∧
  (∃ x : ℝ, x^2 - 4*x + 3 = 0 ∧ x^2 - 6*x + k₂ = 0) ∧
  k₁ ≠ k₂ ∧
  k₁ + k₂ = 14 :=
by sorry

end NUMINAMATH_CALUDE_common_root_sum_k_l936_93684


namespace NUMINAMATH_CALUDE_card_drawing_problem_l936_93601

theorem card_drawing_problem (n : Nat) (r y b g : Nat) (total_cards : Nat) (drawn_cards : Nat) : 
  n = 12 → r = 3 → y = 3 → b = 3 → g = 3 → 
  total_cards = n → 
  drawn_cards = 3 → 
  (Nat.choose total_cards drawn_cards) - 
  (4 * (Nat.choose r drawn_cards)) - 
  ((Nat.choose r 2) * (Nat.choose (y + b + g) 1)) = 189 := by
sorry

end NUMINAMATH_CALUDE_card_drawing_problem_l936_93601


namespace NUMINAMATH_CALUDE_curve_asymptotes_sum_l936_93677

/-- A curve with equation y = x / (x^3 + Ax^2 + Bx + C) where A, B, and C are integers -/
structure Curve where
  A : ℤ
  B : ℤ
  C : ℤ

/-- The denominator of the curve equation -/
def Curve.denominator (c : Curve) (x : ℝ) : ℝ :=
  x^3 + c.A * x^2 + c.B * x + c.C

/-- A curve has a vertical asymptote at x = a if its denominator is zero at x = a -/
def has_vertical_asymptote (c : Curve) (a : ℝ) : Prop :=
  c.denominator a = 0

theorem curve_asymptotes_sum (c : Curve) 
  (h1 : has_vertical_asymptote c (-1))
  (h2 : has_vertical_asymptote c 2)
  (h3 : has_vertical_asymptote c 3) :
  c.A + c.B + c.C = -3 := by
  sorry

end NUMINAMATH_CALUDE_curve_asymptotes_sum_l936_93677


namespace NUMINAMATH_CALUDE_incorrect_statement_l936_93615

theorem incorrect_statement : ¬(0 > |(-1)|) ∧ (-(-3) = 3) ∧ (|2| = |-2|) ∧ (-2 > -3) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l936_93615


namespace NUMINAMATH_CALUDE_max_box_volume_l936_93623

/-- The length of the cardboard in centimeters -/
def cardboard_length : ℝ := 30

/-- The width of the cardboard in centimeters -/
def cardboard_width : ℝ := 14

/-- The volume of the box as a function of the side length of the cut squares -/
def box_volume (x : ℝ) : ℝ := (cardboard_length - 2*x) * (cardboard_width - 2*x) * x

/-- The maximum volume of the box -/
def max_volume : ℝ := 576

theorem max_box_volume :
  ∃ x : ℝ, 0 < x ∧ x < cardboard_width / 2 ∧
  (∀ y : ℝ, 0 < y ∧ y < cardboard_width / 2 → box_volume y ≤ box_volume x) ∧
  box_volume x = max_volume :=
sorry

end NUMINAMATH_CALUDE_max_box_volume_l936_93623


namespace NUMINAMATH_CALUDE_spongebob_daily_earnings_l936_93651

/-- Spongebob's earnings for a day of work at the burger shop -/
def spongebob_earnings (num_burgers : ℕ) (price_burger : ℚ) (num_fries : ℕ) (price_fries : ℚ) : ℚ :=
  num_burgers * price_burger + num_fries * price_fries

/-- Theorem: Spongebob's earnings for the day are $78 -/
theorem spongebob_daily_earnings : 
  spongebob_earnings 30 2 12 (3/2) = 78 := by
sorry

end NUMINAMATH_CALUDE_spongebob_daily_earnings_l936_93651


namespace NUMINAMATH_CALUDE_max_erasable_digits_l936_93611

/-- Represents the number of digits in the original number -/
def total_digits : ℕ := 1000

/-- Represents the sum of digits we want to maintain after erasure -/
def target_sum : ℕ := 2018

/-- Represents the repetitive pattern in the original number -/
def pattern : List ℕ := [2, 0, 1, 8]

/-- Represents the sum of digits in one repetition of the pattern -/
def pattern_sum : ℕ := pattern.sum

/-- Represents the number of complete repetitions of the pattern in the original number -/
def repetitions : ℕ := total_digits / pattern.length

theorem max_erasable_digits : 
  ∃ (erasable : ℕ), 
    erasable = total_digits - (target_sum / pattern_sum * pattern.length + target_sum % pattern_sum) ∧
    erasable = 741 := by sorry

end NUMINAMATH_CALUDE_max_erasable_digits_l936_93611


namespace NUMINAMATH_CALUDE_x_squared_plus_four_y_squared_lt_one_l936_93641

theorem x_squared_plus_four_y_squared_lt_one
  (x y : ℝ)
  (x_pos : x > 0)
  (y_pos : y > 0)
  (h : x^3 + y^3 = x - y) :
  x^2 + 4*y^2 < 1 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_plus_four_y_squared_lt_one_l936_93641


namespace NUMINAMATH_CALUDE_greatest_m_for_ratio_bound_l936_93617

/-- The number of ordered m-coverings of a set with 2n elements -/
def a (m n : ℕ) : ℕ := (2^m - 1)^(2*n)

/-- The number of ordered m-coverings without pairs of a set with 2n elements -/
def b (m n : ℕ) : ℕ := (3^m - 2^(m+1) + 1)^n

/-- The ratio of a(m,n) to b(m,n) -/
def ratio (m n : ℕ) : ℚ := (a m n : ℚ) / (b m n : ℚ)

theorem greatest_m_for_ratio_bound :
  (∃ n : ℕ+, ratio 26 n ≤ 2021) ∧
  (∀ m > 26, ∀ n : ℕ+, ratio m n > 2021) :=
sorry

end NUMINAMATH_CALUDE_greatest_m_for_ratio_bound_l936_93617


namespace NUMINAMATH_CALUDE_square_root_of_25_l936_93624

theorem square_root_of_25 : ∀ x : ℝ, x^2 = 25 ↔ x = 5 ∨ x = -5 := by sorry

end NUMINAMATH_CALUDE_square_root_of_25_l936_93624


namespace NUMINAMATH_CALUDE_complex_equation_implies_unit_magnitude_l936_93699

theorem complex_equation_implies_unit_magnitude (z : ℂ) : 
  11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0 → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_unit_magnitude_l936_93699


namespace NUMINAMATH_CALUDE_no_cube_in_range_l936_93691

theorem no_cube_in_range : ¬ ∃ n : ℕ, 4 ≤ n ∧ n ≤ 12 ∧ ∃ k : ℕ, n^2 + 3*n + 1 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_no_cube_in_range_l936_93691


namespace NUMINAMATH_CALUDE_rowing_round_trip_time_l936_93689

/-- Proves that the total time to row to a place and back is 1 hour, given the specified conditions -/
theorem rowing_round_trip_time
  (rowing_speed : ℝ)
  (current_speed : ℝ)
  (distance : ℝ)
  (h1 : rowing_speed = 5)
  (h2 : current_speed = 1)
  (h3 : distance = 2.4)
  : (distance / (rowing_speed + current_speed)) + (distance / (rowing_speed - current_speed)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_rowing_round_trip_time_l936_93689


namespace NUMINAMATH_CALUDE_original_number_l936_93660

theorem original_number : ∃ x : ℝ, 10 * x = x + 81 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_original_number_l936_93660


namespace NUMINAMATH_CALUDE_parabola_through_point_l936_93644

/-- A parabola passing through the point (4, -2) has either the equation y² = x or x² = -8y -/
theorem parabola_through_point (P : ℝ × ℝ) (h : P = (4, -2)) :
  (∃ (x y : ℝ), y^2 = x ∧ P = (x, y)) ∨ (∃ (x y : ℝ), x^2 = -8*y ∧ P = (x, y)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l936_93644


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_11_l936_93621

theorem three_digit_number_divisible_by_11 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 6 ∧ (n / 100) % 10 = 3 ∧ n % 11 = 0 ∧ n = 396 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_11_l936_93621


namespace NUMINAMATH_CALUDE_k_minus_one_not_square_k_plus_one_not_square_l936_93648

/-- k is the product of several of the first prime numbers -/
def k : ℕ := sorry

/-- k is the product of at least two prime numbers -/
axiom k_def : ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p < q ∧ k = p * q

/-- k-1 is not a perfect square -/
theorem k_minus_one_not_square : ¬∃ (n : ℕ), n^2 = k - 1 := by sorry

/-- k+1 is not a perfect square -/
theorem k_plus_one_not_square : ¬∃ (n : ℕ), n^2 = k + 1 := by sorry

end NUMINAMATH_CALUDE_k_minus_one_not_square_k_plus_one_not_square_l936_93648


namespace NUMINAMATH_CALUDE_initial_population_is_10000_l936_93654

/-- Represents the annual population growth rate -/
def annual_growth_rate : ℝ := 0.1

/-- Represents the population after 2 years -/
def population_after_2_years : ℕ := 12100

/-- Calculates the population after n years given an initial population -/
def population_after_n_years (initial_population : ℝ) (n : ℕ) : ℝ :=
  initial_population * (1 + annual_growth_rate) ^ n

/-- Theorem stating that if a population grows by 10% annually and reaches 12100 after 2 years,
    the initial population was 10000 -/
theorem initial_population_is_10000 :
  ∃ (initial_population : ℕ),
    (population_after_n_years initial_population 2 = population_after_2_years) ∧
    (initial_population = 10000) := by
  sorry

end NUMINAMATH_CALUDE_initial_population_is_10000_l936_93654


namespace NUMINAMATH_CALUDE_sum_squared_equals_400_l936_93668

variable (a b c : ℝ)

theorem sum_squared_equals_400 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a*b + b*c + c*a = 5) : 
  (a + b + c)^2 = 400 := by
sorry

end NUMINAMATH_CALUDE_sum_squared_equals_400_l936_93668


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l936_93680

theorem greatest_prime_factor_of_341 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 341 → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l936_93680


namespace NUMINAMATH_CALUDE_zoo_visitors_l936_93619

/-- Represents the number of adults who went to the zoo on Monday -/
def adults_monday : ℕ := sorry

/-- The total revenue from both days -/
def total_revenue : ℕ := 61

/-- The cost of a child ticket -/
def child_ticket_cost : ℕ := 3

/-- The cost of an adult ticket -/
def adult_ticket_cost : ℕ := 4

/-- The number of children who went to the zoo on Monday -/
def children_monday : ℕ := 7

/-- The number of children who went to the zoo on Tuesday -/
def children_tuesday : ℕ := 4

/-- The number of adults who went to the zoo on Tuesday -/
def adults_tuesday : ℕ := 2

theorem zoo_visitors :
  adults_monday = 5 :=
by sorry

end NUMINAMATH_CALUDE_zoo_visitors_l936_93619


namespace NUMINAMATH_CALUDE_conference_languages_l936_93602

/-- The proportion of delegates who know both English and Spanish -/
def both_languages (p_english p_spanish : ℝ) : ℝ :=
  p_english + p_spanish - 1

theorem conference_languages :
  let p_english : ℝ := 0.85
  let p_spanish : ℝ := 0.75
  both_languages p_english p_spanish = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_conference_languages_l936_93602


namespace NUMINAMATH_CALUDE_three_lines_triangle_l936_93655

/-- A line in the 2D plane represented by its equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  ∃ x y : ℝ, l1.a * x + l1.b * y = l1.c ∧
             l2.a * x + l2.b * y = l2.c ∧
             l3.a * x + l3.b * y = l3.c

/-- The set of possible values for m -/
def possible_m_values : Set ℝ :=
  {m : ℝ | m = 4 ∨ m = -1/6 ∨ m = 1 ∨ m = -2/3}

theorem three_lines_triangle (m : ℝ) :
  let l1 : Line := ⟨4, 1, 4⟩
  let l2 : Line := ⟨m, 1, 0⟩
  let l3 : Line := ⟨2, -3*m, 4⟩
  (parallel l1 l2 ∨ parallel l1 l3 ∨ parallel l2 l3 ∨ intersect_at_point l1 l2 l3) →
  m ∈ possible_m_values :=
sorry

end NUMINAMATH_CALUDE_three_lines_triangle_l936_93655


namespace NUMINAMATH_CALUDE_pool_radius_l936_93695

/-- Proves that a circular pool with a surrounding concrete wall has a radius of 20 feet
    given specific conditions on the wall width and area ratio. -/
theorem pool_radius (r : ℝ) : 
  r > 0 → -- The radius is positive
  (π * ((r + 4)^2 - r^2) = (11/25) * π * r^2) → -- Area ratio condition
  r = 20 := by
  sorry

end NUMINAMATH_CALUDE_pool_radius_l936_93695


namespace NUMINAMATH_CALUDE_bank_account_final_amount_l936_93653

/-- Calculates the final amount in a bank account given initial savings, withdrawal, and deposit. -/
def final_amount (initial_savings withdrawal : ℕ) : ℕ :=
  initial_savings - withdrawal + 2 * withdrawal

/-- Theorem stating that given the specific conditions, the final amount is $290. -/
theorem bank_account_final_amount : 
  final_amount 230 60 = 290 := by
  sorry

end NUMINAMATH_CALUDE_bank_account_final_amount_l936_93653


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l936_93681

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- State the theorem
theorem intersection_complement_theorem :
  M ∩ (U \ N) = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l936_93681


namespace NUMINAMATH_CALUDE_house_problem_l936_93630

theorem house_problem (total garage pool both : ℕ) 
  (h_total : total = 65)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_both : both = 35) :
  total - (garage + pool - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_house_problem_l936_93630


namespace NUMINAMATH_CALUDE_new_person_weight_is_143_l936_93649

/-- Calculates the weight of a new person given the following conditions:
  * There are 15 people initially
  * The average weight increases by 5 kg when the new person replaces one person
  * The replaced person weighs 68 kg
-/
def new_person_weight (initial_count : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  initial_count * avg_increase + replaced_weight

/-- Proves that under the given conditions, the weight of the new person is 143 kg -/
theorem new_person_weight_is_143 :
  new_person_weight 15 5 68 = 143 := by
  sorry

#eval new_person_weight 15 5 68

end NUMINAMATH_CALUDE_new_person_weight_is_143_l936_93649


namespace NUMINAMATH_CALUDE_nine_digit_divisibility_l936_93640

theorem nine_digit_divisibility (a b c : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) (h4 : a ≠ 0) :
  ∃ k : ℕ, (a * 100000000 + b * 10000000 + c * 1000000 +
            a * 100000 + b * 10000 + c * 1000 +
            a * 100 + b * 10 + c) = k * 1001001 :=
by sorry

end NUMINAMATH_CALUDE_nine_digit_divisibility_l936_93640


namespace NUMINAMATH_CALUDE_gift_budget_calculation_l936_93620

theorem gift_budget_calculation (total_budget : ℕ) (num_friends : ℕ) (friend_gift_cost : ℕ) 
  (num_family : ℕ) : 
  total_budget = 200 → 
  num_friends = 12 → 
  friend_gift_cost = 15 → 
  num_family = 4 → 
  (total_budget - num_friends * friend_gift_cost) / num_family = 5 := by
sorry

end NUMINAMATH_CALUDE_gift_budget_calculation_l936_93620


namespace NUMINAMATH_CALUDE_N_subset_M_l936_93657

-- Define set M
def M : Set ℝ := {x | ∃ n : ℤ, x = n / 2 + 1}

-- Define set N
def N : Set ℝ := {y | ∃ m : ℤ, y = m + 1 / 2}

-- Theorem statement
theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l936_93657


namespace NUMINAMATH_CALUDE_tank_filling_l936_93692

theorem tank_filling (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 25 →
  capacity_ratio = 2 / 5 →
  ∃ (new_buckets : ℕ), 
    (↑new_buckets : ℚ) > (↑original_buckets / capacity_ratio) ∧ 
    (↑new_buckets : ℚ) ≤ (↑original_buckets / capacity_ratio + 1) ∧
    new_buckets = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_filling_l936_93692


namespace NUMINAMATH_CALUDE_high_school_students_l936_93618

theorem high_school_students (total : ℕ) 
  (h1 : (total : ℚ) / 2 = (total : ℚ) * (1 / 2))
  (h2 : (total : ℚ) / 2 * (1 / 5) = (total : ℚ) * (1 / 10))
  (h3 : (total : ℚ) * (1 / 10) + 160 = (total : ℚ) * (1 / 2)) : 
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_high_school_students_l936_93618


namespace NUMINAMATH_CALUDE_lynne_cat_books_l936_93606

def books_about_cats (x : ℕ) : Prop :=
  ∃ (total_spent : ℕ),
    let books_solar_system := 2
    let magazines := 3
    let book_cost := 7
    let magazine_cost := 4
    total_spent = 75 ∧
    total_spent = x * book_cost + books_solar_system * book_cost + magazines * magazine_cost

theorem lynne_cat_books : ∃ x : ℕ, books_about_cats x ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_lynne_cat_books_l936_93606


namespace NUMINAMATH_CALUDE_square_root_problem_l936_93679

theorem square_root_problem (x : ℝ) : (Real.sqrt x / 11 = 4) → x = 1936 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l936_93679


namespace NUMINAMATH_CALUDE_brianna_book_savings_l936_93632

theorem brianna_book_savings (m : ℚ) (p : ℚ) : 
  (1/4 : ℚ) * m = (1/2 : ℚ) * p → m - p = (1/2 : ℚ) * m :=
by
  sorry

end NUMINAMATH_CALUDE_brianna_book_savings_l936_93632


namespace NUMINAMATH_CALUDE_remainder_problem_l936_93650

theorem remainder_problem (y : ℤ) (h : y % 264 = 42) : y % 22 = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l936_93650


namespace NUMINAMATH_CALUDE_spinster_cat_problem_l936_93656

theorem spinster_cat_problem (spinsters cats : ℕ) : 
  (spinsters : ℚ) / cats = 2 / 9 →
  cats = spinsters + 63 →
  spinsters = 18 := by
sorry

end NUMINAMATH_CALUDE_spinster_cat_problem_l936_93656


namespace NUMINAMATH_CALUDE_tangent_line_at_P_l936_93639

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + 1

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the two possible tangent line equations
def tangent1 (x y : ℝ) : Prop := 3*x - y - 1 = 0
def tangent2 (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Theorem statement
theorem tangent_line_at_P :
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m*x + b ↔ (tangent1 x y ∨ tangent2 x y)) ∧
  (curve P.1 = P.2) ∧
  (∃ ε > 0, ∀ h ∈ Set.Ioo (-ε) ε, 
    (curve (P.1 + h) - curve P.1) / h - m < ε) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_l936_93639


namespace NUMINAMATH_CALUDE_tank_filling_l936_93688

theorem tank_filling (original_buckets : ℕ) (capacity_reduction : ℚ) : 
  original_buckets = 10 →
  capacity_reduction = 2/5 →
  ∃ (new_buckets : ℕ), new_buckets ≥ 25 ∧ 
    (new_buckets : ℚ) * capacity_reduction = original_buckets := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_l936_93688


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_distinct_primes_l936_93667

def is_divisible_by_four_distinct_primes (n : ℕ) : Prop :=
  ∃ p q r s : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  n % p = 0 ∧ n % q = 0 ∧ n % r = 0 ∧ n % s = 0

theorem least_positive_integer_divisible_by_four_distinct_primes :
  (∀ m : ℕ, m > 0 → is_divisible_by_four_distinct_primes m → m ≥ 210) ∧
  is_divisible_by_four_distinct_primes 210 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_distinct_primes_l936_93667


namespace NUMINAMATH_CALUDE_power_function_value_l936_93670

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x^a

-- State the theorem
theorem power_function_value 
  (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 4 = 1/2) : 
  f (1/16) = 4 := by
sorry

end NUMINAMATH_CALUDE_power_function_value_l936_93670


namespace NUMINAMATH_CALUDE_reciprocal_relationship_l936_93697

theorem reciprocal_relationship (a b : ℝ) : (a + b)^2 - (a - b)^2 = 4 → a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_relationship_l936_93697


namespace NUMINAMATH_CALUDE_theater_hall_seats_l936_93664

/-- Represents a theater hall with three categories of seats. -/
structure TheaterHall where
  totalSeats : ℕ
  categoryIPrice : ℕ
  categoryIIPrice : ℕ
  categoryIIIPrice : ℕ
  freeTickets : ℕ
  revenueDifference : ℕ

/-- Checks if the theater hall satisfies all given conditions. -/
def validTheaterHall (hall : TheaterHall) : Prop :=
  (hall.totalSeats % 5 = 0) ∧
  (hall.categoryIPrice = 220) ∧
  (hall.categoryIIPrice = 200) ∧
  (hall.categoryIIIPrice = 180) ∧
  (hall.freeTickets = 150) ∧
  (hall.revenueDifference = 4320)

/-- Theorem stating that a valid theater hall has 360 seats. -/
theorem theater_hall_seats (hall : TheaterHall) 
  (h : validTheaterHall hall) : hall.totalSeats = 360 := by
  sorry

#check theater_hall_seats

end NUMINAMATH_CALUDE_theater_hall_seats_l936_93664


namespace NUMINAMATH_CALUDE_eric_running_time_l936_93673

/-- Given Eric's trip to the park and back, prove the time he ran before jogging. -/
theorem eric_running_time (total_time_to_park : ℕ) (running_time : ℕ) : 
  total_time_to_park = running_time + 10 →
  90 = 3 * total_time_to_park →
  running_time = 20 := by
sorry

end NUMINAMATH_CALUDE_eric_running_time_l936_93673


namespace NUMINAMATH_CALUDE_jadens_car_count_l936_93608

/-- Calculates the final number of toy cars Jaden has after a series of transactions -/
def jadensFinalCarCount (initial bought birthday givenSister givenVinnie tradedAway tradedFor : ℕ) : ℕ :=
  initial + bought + birthday - givenSister - givenVinnie - tradedAway + tradedFor

/-- Theorem stating that Jaden ends up with 45 toy cars -/
theorem jadens_car_count : 
  jadensFinalCarCount 14 28 12 8 3 5 7 = 45 := by
  sorry

end NUMINAMATH_CALUDE_jadens_car_count_l936_93608


namespace NUMINAMATH_CALUDE_triangle_sine_squared_ratio_l936_93613

theorem triangle_sine_squared_ratio (a b c : ℝ) (A B C : Real) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  S > 0 →
  S = (1/2) * a * b * Real.sin C →
  (a^2 + b^2) * Real.tan C = 8 * S →
  (Real.sin A)^2 + (Real.sin B)^2 = 2 * (Real.sin C)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_squared_ratio_l936_93613


namespace NUMINAMATH_CALUDE_triangle_perimeter_l936_93694

theorem triangle_perimeter (a b c : ℕ) : 
  a = 7 → b = 2 → Odd c → a + b + c = 16 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l936_93694


namespace NUMINAMATH_CALUDE_quadratic_maximum_l936_93685

theorem quadratic_maximum (s : ℝ) : -7 * s^2 + 56 * s + 20 ≤ 132 ∧ ∃ t : ℝ, -7 * t^2 + 56 * t + 20 = 132 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l936_93685


namespace NUMINAMATH_CALUDE_circle_area_l936_93658

theorem circle_area (x y : ℝ) : 
  (2 * x^2 + 2 * y^2 + 10 * x - 6 * y - 18 = 0) → 
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    ((x - center.1)^2 + (y - center.2)^2 = r^2) ∧ 
    (π * r^2 = 35/2 * π)) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_l936_93658


namespace NUMINAMATH_CALUDE_value_of_expression_l936_93690

theorem value_of_expression (x : ℝ) (h : x = -3) : 3 * x^2 + 2 * x = 21 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l936_93690


namespace NUMINAMATH_CALUDE_division_remainder_proof_l936_93629

theorem division_remainder_proof :
  let dividend : ℕ := 165
  let divisor : ℕ := 18
  let quotient : ℕ := 9
  let remainder : ℕ := dividend - divisor * quotient
  remainder = 3 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l936_93629


namespace NUMINAMATH_CALUDE_uma_money_fraction_l936_93661

theorem uma_money_fraction (rita sam tina unknown : ℚ) : 
  rita > 0 ∧ sam > 0 ∧ tina > 0 ∧ unknown > 0 →
  rita / 6 = sam / 5 ∧ rita / 6 = tina / 7 ∧ rita / 6 = unknown / 8 →
  (rita / 6 + sam / 5 + tina / 7 + unknown / 8) / (rita + sam + tina + unknown) = 2 / 13 := by
sorry

end NUMINAMATH_CALUDE_uma_money_fraction_l936_93661


namespace NUMINAMATH_CALUDE_fair_attendance_l936_93672

/-- The total attendance at a fair over three years -/
def total_attendance (this_year : ℕ) : ℕ :=
  let next_year := 2 * this_year
  let last_year := next_year - 200
  last_year + this_year + next_year

/-- Theorem: The total attendance over three years is 2800 -/
theorem fair_attendance : total_attendance 600 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_l936_93672


namespace NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_unique_index_298_l936_93652

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℕ := 1 + (n - 1) * 3

/-- The theorem stating that the 100th term of the arithmetic sequence is 298 -/
theorem arithmetic_sequence_100th_term :
  arithmetic_sequence 100 = 298 := by sorry

/-- The theorem stating that 100 is the unique index for which the term is 298 -/
theorem unique_index_298 :
  ∀ n : ℕ, arithmetic_sequence n = 298 ↔ n = 100 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_unique_index_298_l936_93652


namespace NUMINAMATH_CALUDE_mary_remaining_money_l936_93676

def initial_money : ℕ := 58
def pie_cost : ℕ := 6

theorem mary_remaining_money :
  initial_money - pie_cost = 52 := by sorry

end NUMINAMATH_CALUDE_mary_remaining_money_l936_93676


namespace NUMINAMATH_CALUDE_square_difference_formula_l936_93671

theorem square_difference_formula : 15^2 - 2*(15*5) + 5^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l936_93671


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l936_93646

theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ x y : ℝ, y = (3/5) * x ∧ x^2 / a^2 - y^2 / 9 = 1) →
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l936_93646


namespace NUMINAMATH_CALUDE_angle_difference_range_l936_93612

theorem angle_difference_range (α β : ℝ) 
  (h1 : -π/2 < α) (h2 : α < 0) (h3 : 0 < β) (h4 : β < π/3) : 
  -5*π/6 < α - β ∧ α - β < 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_difference_range_l936_93612


namespace NUMINAMATH_CALUDE_complement_A_in_B_union_equality_implies_m_range_l936_93675

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | x > m}

-- Theorem 1: Complement of A in B when m = -1
theorem complement_A_in_B : 
  {x : ℝ | x ∈ B (-1) ∧ x ∉ A} = {x : ℝ | x ≥ 3} := by sorry

-- Theorem 2: Range of m when A ∪ B = B
theorem union_equality_implies_m_range (m : ℝ) : 
  A ∪ B m = B m → m ≤ -1 := by sorry

end NUMINAMATH_CALUDE_complement_A_in_B_union_equality_implies_m_range_l936_93675


namespace NUMINAMATH_CALUDE_expression_equality_l936_93666

theorem expression_equality : 
  (((3 + 5 + 7) / (2 + 4 + 6)) * 2 - ((2 + 4 + 6) / (3 + 5 + 7))) = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l936_93666


namespace NUMINAMATH_CALUDE_craigs_walk_distance_l936_93614

theorem craigs_walk_distance (distance_school_to_david : ℝ) (distance_david_to_home : ℝ)
  (h1 : distance_school_to_david = 0.27)
  (h2 : distance_david_to_home = 0.73) :
  distance_school_to_david + distance_david_to_home = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_craigs_walk_distance_l936_93614


namespace NUMINAMATH_CALUDE_eight_times_ten_y_plus_fourteen_sin_y_l936_93603

theorem eight_times_ten_y_plus_fourteen_sin_y (y : ℝ) (Q : ℝ) 
  (h : 4 * (5 * y + 7 * Real.sin y) = Q) : 
  8 * (10 * y + 14 * Real.sin y) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_eight_times_ten_y_plus_fourteen_sin_y_l936_93603


namespace NUMINAMATH_CALUDE_artist_june_pictures_l936_93698

/-- The number of pictures painted in June -/
def june_pictures : ℕ := sorry

/-- The number of pictures painted in July -/
def july_pictures : ℕ := june_pictures + 2

/-- The number of pictures painted in August -/
def august_pictures : ℕ := 9

/-- The total number of pictures painted over the three months -/
def total_pictures : ℕ := 13

theorem artist_june_pictures :
  june_pictures = 1 ∧
  june_pictures + july_pictures + august_pictures = total_pictures :=
sorry

end NUMINAMATH_CALUDE_artist_june_pictures_l936_93698


namespace NUMINAMATH_CALUDE_christen_peeled_21_potatoes_l936_93616

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initial_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  time_before_christen : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledPotatoes (scenario : PotatoPeeling) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that Christen peeled 21 potatoes in the given scenario -/
theorem christen_peeled_21_potatoes :
  let scenario : PotatoPeeling := {
    initial_potatoes := 60,
    homer_rate := 4,
    christen_rate := 6,
    time_before_christen := 6
  }
  christenPeeledPotatoes scenario = 21 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_21_potatoes_l936_93616


namespace NUMINAMATH_CALUDE_rhombus_area_l936_93626

/-- The area of a rhombus with side length √117 and diagonals differing by 10 units is 72 square units. -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) (h₁ : s = Real.sqrt 117) (h₂ : d₂ - d₁ = 10) 
  (h₃ : s^2 = (d₁/2)^2 + (d₂/2)^2) : d₁ * d₂ / 2 = 72 := by
  sorry

#check rhombus_area

end NUMINAMATH_CALUDE_rhombus_area_l936_93626


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l936_93628

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h1 : area = 120) (h2 : leg = 24) :
  ∃ (other_leg hypotenuse : ℝ),
    area = (1 / 2) * leg * other_leg ∧
    hypotenuse ^ 2 = leg ^ 2 + other_leg ^ 2 ∧
    leg + other_leg + hypotenuse = 60 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l936_93628


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l936_93642

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∃ x : ℝ, x ^ 2 = 1 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x ^ 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l936_93642


namespace NUMINAMATH_CALUDE_circle_through_three_points_l936_93627

/-- A circle passing through three points -/
structure Circle3Points where
  O : ℝ × ℝ
  M₁ : ℝ × ℝ
  M₂ : ℝ × ℝ

/-- The equation of a circle in standard form -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The circle passing through O(0,0), M₁(1,1), and M₂(4,2) 
    has the equation (x-4)² + (y+3)² = 25, with center (4, -3) and radius 5 -/
theorem circle_through_three_points :
  let c := Circle3Points.mk (0, 0) (1, 1) (4, 2)
  ∃ (h k r : ℝ),
    h = 4 ∧ k = -3 ∧ r = 5 ∧
    (∀ (x y : ℝ), CircleEquation h k r x y ↔
      ((x = c.O.1 ∧ y = c.O.2) ∨
       (x = c.M₁.1 ∧ y = c.M₁.2) ∨
       (x = c.M₂.1 ∧ y = c.M₂.2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_through_three_points_l936_93627


namespace NUMINAMATH_CALUDE_grapes_problem_l936_93610

theorem grapes_problem (bryce_grapes : ℚ) : 
  (∃ (carter_grapes : ℚ), 
    bryce_grapes = carter_grapes + 7 ∧ 
    carter_grapes = bryce_grapes / 3) → 
  bryce_grapes = 21 / 2 := by
sorry

end NUMINAMATH_CALUDE_grapes_problem_l936_93610


namespace NUMINAMATH_CALUDE_solution_form_and_sum_l936_93636

theorem solution_form_and_sum (x y : ℝ) : 
  (x + y = 7 ∧ 4 * x * y = 7) →
  ∃ (a b c d : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d ∧
    a = 7 ∧ b = 1 ∧ c = 42 ∧ d = 2 ∧
    a + b + c + d = 52 :=
by sorry

end NUMINAMATH_CALUDE_solution_form_and_sum_l936_93636


namespace NUMINAMATH_CALUDE_parabola_and_max_area_line_l936_93600

-- Define the parabola
def Parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define a point on the parabola
def PointOnParabola (p : ℝ) (x₀ : ℝ) : Prop := Parabola p x₀ 4

-- Define the distance from a point to the focus
def DistanceToFocus (p : ℝ) (x₀ : ℝ) : Prop := x₀ + p/2 = 4

-- Define the angle bisector condition
def AngleBisectorCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ (4 - y₁)/(2 - x₁) = -(4 - y₂)/(2 - x₂)

-- Main theorem
theorem parabola_and_max_area_line
  (p : ℝ) (x₀ : ℝ)
  (h₁ : PointOnParabola p x₀)
  (h₂ : DistanceToFocus p x₀) :
  (∀ x y, Parabola p x y ↔ y^2 = 8*x) ∧
  (∃ x₁ y₁ x₂ y₂,
    AngleBisectorCondition x₁ y₁ x₂ y₂ ∧
    Parabola p x₁ y₁ ∧ Parabola p x₂ y₂ ∧
    (∀ a b, (Parabola p a b ∧ b ≤ 0) →
      (x₁ - 2)*(y₂ - 4) - (x₂ - 2)*(y₁ - 4) ≤ (x₁ - 2)*(b - 4) - (a - 2)*(y₁ - 4)) ∧
    x₁ + y₁ = 0 ∧ x₂ + y₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_max_area_line_l936_93600


namespace NUMINAMATH_CALUDE_computer_profit_pricing_l936_93687

theorem computer_profit_pricing (cost selling_price_40 selling_price_60 : ℝ) :
  selling_price_40 = 2240 ∧
  selling_price_40 = cost * 1.4 →
  selling_price_60 = cost * 1.6 →
  selling_price_60 = 2560 := by
sorry

end NUMINAMATH_CALUDE_computer_profit_pricing_l936_93687


namespace NUMINAMATH_CALUDE_t_value_l936_93659

theorem t_value : 
  let t := 2 / (1 - Real.rpow 2 (1/3))
  t = -2 * (1 + Real.rpow 2 (1/3) + Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_t_value_l936_93659


namespace NUMINAMATH_CALUDE_casey_owns_five_hoodies_l936_93604

/-- The number of hoodies Fiona and Casey own together -/
def total_hoodies : ℕ := 8

/-- The number of hoodies Fiona owns -/
def fiona_hoodies : ℕ := 3

/-- The number of hoodies Casey owns -/
def casey_hoodies : ℕ := total_hoodies - fiona_hoodies

theorem casey_owns_five_hoodies : casey_hoodies = 5 := by
  sorry

end NUMINAMATH_CALUDE_casey_owns_five_hoodies_l936_93604


namespace NUMINAMATH_CALUDE_possible_values_of_a_l936_93647

theorem possible_values_of_a (a : ℝ) 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (non_neg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0)
  (eq1 : x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ = a)
  (eq2 : x₁ + 8*x₂ + 27*x₃ + 64*x₄ + 125*x₅ = a^2)
  (eq3 : x₁ + 32*x₂ + 243*x₃ + 1024*x₄ + 3125*x₅ = a^3) :
  a ∈ ({0, 1, 4, 9, 16, 25} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l936_93647


namespace NUMINAMATH_CALUDE_angle_inclination_range_l936_93643

theorem angle_inclination_range (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ((-a - 2 + 1) * (Real.sqrt 3 / 3 * a + 1) > 0)) :
  ∃ α : ℝ, (2 * Real.pi / 3 < α) ∧ (α < 3 * Real.pi / 4) ∧ 
  (a = Real.tan α) :=
sorry

end NUMINAMATH_CALUDE_angle_inclination_range_l936_93643


namespace NUMINAMATH_CALUDE_smallest_middle_term_l936_93605

theorem smallest_middle_term (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c → 
  (∃ d : ℝ, a = b - d ∧ c = b + d) → 
  a * b * c = 216 → 
  b ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_middle_term_l936_93605
