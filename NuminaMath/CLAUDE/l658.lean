import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_minimum_l658_65880

theorem quadratic_minimum (h : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → (x - h)^2 + 1 ≥ 10) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ (x - h)^2 + 1 = 10) →
  h = -2 ∨ h = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l658_65880


namespace NUMINAMATH_CALUDE_digit_difference_1234_l658_65817

/-- The number of digits in the base-b representation of a positive integer n -/
def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  if n < b then 1 else Nat.log b n + 1

/-- The difference in the number of digits between base-4 and base-9 representations of 1234 -/
theorem digit_difference_1234 :
  num_digits 1234 4 - num_digits 1234 9 = 2 := by sorry

end NUMINAMATH_CALUDE_digit_difference_1234_l658_65817


namespace NUMINAMATH_CALUDE_point_on_x_axis_l658_65829

/-- 
A point M with coordinates (m-1, 2m) lies on the x-axis if and only if 
its coordinates are (-1, 0).
-/
theorem point_on_x_axis (m : ℝ) : 
  (m - 1, 2 * m) ∈ {p : ℝ × ℝ | p.2 = 0} ↔ (m - 1, 2 * m) = (-1, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l658_65829


namespace NUMINAMATH_CALUDE_two_false_propositions_l658_65830

-- Define the original proposition
def original_prop (a : ℝ) : Prop := a > -3 → a > -6

-- Define the converse proposition
def converse_prop (a : ℝ) : Prop := a > -6 → a > -3

-- Define the inverse proposition
def inverse_prop (a : ℝ) : Prop := a ≤ -3 → a ≤ -6

-- Define the contrapositive proposition
def contrapositive_prop (a : ℝ) : Prop := a ≤ -6 → a ≤ -3

-- Theorem statement
theorem two_false_propositions :
  ∃ (f : Fin 4 → Prop), 
    (∀ a : ℝ, f 0 = original_prop a ∧ 
              f 1 = converse_prop a ∧ 
              f 2 = inverse_prop a ∧ 
              f 3 = contrapositive_prop a) ∧
    (∃! (i j : Fin 4), i ≠ j ∧ ¬(f i) ∧ ¬(f j) ∧ 
      ∀ (k : Fin 4), k ≠ i ∧ k ≠ j → f k) :=
by
  sorry

end NUMINAMATH_CALUDE_two_false_propositions_l658_65830


namespace NUMINAMATH_CALUDE_range_of_m_l658_65868

/-- The set A -/
def A : Set ℝ := {x | |x - 2| ≤ 4}

/-- The set B parameterized by m -/
def B (m : ℝ) : Set ℝ := {x | (x - 1 - m) * (x - 1 + m) ≤ 0}

/-- The proposition that ¬p is a necessary but not sufficient condition for ¬q -/
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  (∀ x, x ∉ B m → x ∉ A) ∧ ∃ x, x ∉ B m ∧ x ∈ A

/-- The theorem stating the range of m -/
theorem range_of_m :
  ∀ m : ℝ, m > 0 ∧ not_p_necessary_not_sufficient_for_not_q m ↔ m ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l658_65868


namespace NUMINAMATH_CALUDE_a_gt_6_sufficient_not_necessary_for_a_sq_gt_36_l658_65807

theorem a_gt_6_sufficient_not_necessary_for_a_sq_gt_36 :
  (∀ a : ℝ, a > 6 → a^2 > 36) ∧
  (∃ a : ℝ, a^2 > 36 ∧ a ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_6_sufficient_not_necessary_for_a_sq_gt_36_l658_65807


namespace NUMINAMATH_CALUDE_min_value_theorem_l658_65862

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y ≤ 2) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 4 ∧
  ∀ (z : ℝ), z = 2 / (x + 3 * y) + 1 / (x - y) → z ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l658_65862


namespace NUMINAMATH_CALUDE_total_tires_changed_l658_65867

/-- The number of tires on a motorcycle -/
def motorcycle_tires : ℕ := 2

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of motorcycles Mike changed tires on -/
def num_motorcycles : ℕ := 12

/-- The number of cars Mike changed tires on -/
def num_cars : ℕ := 10

/-- Theorem: The total number of tires Mike changed is 64 -/
theorem total_tires_changed : 
  num_motorcycles * motorcycle_tires + num_cars * car_tires = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_tires_changed_l658_65867


namespace NUMINAMATH_CALUDE_a_closed_form_l658_65860

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => (2 * a n + 6) / (a n + 1)

theorem a_closed_form (n : ℕ) :
  a n = (3 * 4^(n+1) + 2 * (-1)^(n+1)) / (4^(n+1) + (-1)^n) := by
  sorry

end NUMINAMATH_CALUDE_a_closed_form_l658_65860


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l658_65879

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l658_65879


namespace NUMINAMATH_CALUDE_trajectory_of_moving_point_l658_65839

/-- The trajectory of a point M(x, y) that is twice as far from A(-4, 0) as it is from B(2, 0) -/
theorem trajectory_of_moving_point (x y : ℝ) : 
  (((x + 4)^2 + y^2).sqrt = 2 * ((x - 2)^2 + y^2).sqrt) ↔ 
  (x^2 + y^2 - 8*x = 0) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_point_l658_65839


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l658_65885

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l658_65885


namespace NUMINAMATH_CALUDE_games_within_division_is_40_l658_65895

/-- Represents the structure of a baseball league --/
structure BaseballLeague where
  N : ℕ  -- Number of games played against each team in the same division
  M : ℕ  -- Number of games played against each team in other divisions
  total_games : ℕ  -- Total number of games played by each team
  h1 : N > 3 * M
  h2 : M > 5
  h3 : 2 * N + 6 * M = total_games
  h4 : total_games = 76

/-- The number of games a team plays within its own division --/
def games_within_division (league : BaseballLeague) : ℕ :=
  2 * league.N

/-- Theorem stating that the number of games played within a team's own division is 40 --/
theorem games_within_division_is_40 (league : BaseballLeague) :
  games_within_division league = 40 := by
  sorry

#check games_within_division_is_40

end NUMINAMATH_CALUDE_games_within_division_is_40_l658_65895


namespace NUMINAMATH_CALUDE_water_fraction_after_replacements_l658_65886

-- Define the radiator capacity
def radiator_capacity : ℚ := 20

-- Define the volume replaced each time
def replacement_volume : ℚ := 5

-- Define the number of replacements
def num_replacements : ℕ := 5

-- Define the fraction of liquid remaining after each replacement
def remaining_fraction : ℚ := (radiator_capacity - replacement_volume) / radiator_capacity

-- Statement of the problem
theorem water_fraction_after_replacements :
  (remaining_fraction ^ num_replacements : ℚ) = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_water_fraction_after_replacements_l658_65886


namespace NUMINAMATH_CALUDE_remainder_after_adding_4032_l658_65859

theorem remainder_after_adding_4032 (m : ℤ) (h : m % 8 = 3) :
  (m + 4032) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_4032_l658_65859


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l658_65801

-- Define the hyperbola equation
def hyperbola_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 16 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (4, 0)

-- Main theorem
theorem hyperbola_standard_equation 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_asymptote : ∃ x y, asymptote_equation x y ∧ hyperbola_equation a b x y)
  (h_focus : ∃ x y, hyperbola_equation a b x y ∧ (x, y) = parabola_focus) :
  a^2 = 4 ∧ b^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l658_65801


namespace NUMINAMATH_CALUDE_common_root_condition_l658_65821

theorem common_root_condition (m : ℝ) : 
  (∃ x : ℝ, m * x - 1000 = 1021 ∧ 1021 * x = m - 1000 * x) ↔ (m = 2021 ∨ m = -2021) := by
  sorry

end NUMINAMATH_CALUDE_common_root_condition_l658_65821


namespace NUMINAMATH_CALUDE_daria_credit_card_debt_l658_65836

def couch_price : ℝ := 800
def couch_discount : ℝ := 0.10
def table_price : ℝ := 120
def table_discount : ℝ := 0.05
def lamp_price : ℝ := 50
def rug_price : ℝ := 250
def rug_discount : ℝ := 0.20
def bookshelf_price : ℝ := 180
def bookshelf_discount : ℝ := 0.15
def artwork_price : ℝ := 100
def artwork_discount : ℝ := 0.25
def savings : ℝ := 500

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_cost : ℝ :=
  discounted_price couch_price couch_discount +
  discounted_price table_price table_discount +
  lamp_price +
  discounted_price rug_price rug_discount +
  discounted_price bookshelf_price bookshelf_discount +
  discounted_price artwork_price artwork_discount

theorem daria_credit_card_debt :
  total_cost - savings = 812 := by sorry

end NUMINAMATH_CALUDE_daria_credit_card_debt_l658_65836


namespace NUMINAMATH_CALUDE_max_cubes_is_117_l658_65827

/-- The maximum number of 64 cubic centimetre cubes that can fit in a 15 cm x 20 cm x 25 cm rectangular box -/
def max_cubes : ℕ :=
  let box_volume : ℕ := 15 * 20 * 25
  let cube_volume : ℕ := 64
  (box_volume / cube_volume : ℕ)

/-- Theorem stating that the maximum number of 64 cubic centimetre cubes
    that can fit in a 15 cm x 20 cm x 25 cm rectangular box is 117 -/
theorem max_cubes_is_117 : max_cubes = 117 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_is_117_l658_65827


namespace NUMINAMATH_CALUDE_compound_interest_rate_l658_65832

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r / 100)^2 = 2420)
  (h2 : P * (1 + r / 100)^3 = 2662) :
  r = 10 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l658_65832


namespace NUMINAMATH_CALUDE_michaels_pie_order_cost_l658_65873

/-- Calculate the total cost of fruit for Michael's pie order --/
theorem michaels_pie_order_cost :
  let peach_pies := 8
  let apple_pies := 6
  let blueberry_pies := 5
  let mixed_fruit_pies := 3
  let peach_per_pie := 4
  let apple_per_pie := 3
  let blueberry_per_pie := 3.5
  let mixed_fruit_per_pie := 3
  let apple_price := 1.25
  let blueberry_price := 0.90
  let peach_price := 2.50
  let mixed_fruit_per_type := mixed_fruit_per_pie / 3

  let total_peaches := peach_pies * peach_per_pie + mixed_fruit_pies * mixed_fruit_per_type
  let total_apples := apple_pies * apple_per_pie + mixed_fruit_pies * mixed_fruit_per_type
  let total_blueberries := blueberry_pies * blueberry_per_pie + mixed_fruit_pies * mixed_fruit_per_type

  let peach_cost := total_peaches * peach_price
  let apple_cost := total_apples * apple_price
  let blueberry_cost := total_blueberries * blueberry_price

  let total_cost := peach_cost + apple_cost + blueberry_cost

  total_cost = 132.20 := by
    sorry

end NUMINAMATH_CALUDE_michaels_pie_order_cost_l658_65873


namespace NUMINAMATH_CALUDE_total_amount_is_fifteen_l658_65893

/-- Represents the share of each person in Rupees -/
structure Share where
  w : ℚ
  x : ℚ
  y : ℚ

/-- The total amount of the sum -/
def total_amount (s : Share) : ℚ :=
  s.w + s.x + s.y

/-- The condition that for each rupee w gets, x gets 30 paisa and y gets 20 paisa -/
def share_ratio (s : Share) : Prop :=
  s.x = (3/10) * s.w ∧ s.y = (1/5) * s.w

/-- The theorem stating that if w's share is 10 rupees and the share ratio is maintained,
    then the total amount is 15 rupees -/
theorem total_amount_is_fifteen (s : Share) 
    (h1 : s.w = 10)
    (h2 : share_ratio s) : 
    total_amount s = 15 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_fifteen_l658_65893


namespace NUMINAMATH_CALUDE_flight_duration_is_two_hours_l658_65899

/-- Calculates the flight duration in hours given the number of peanut bags, 
    peanuts per bag, and consumption rate. -/
def flight_duration (bags : ℕ) (peanuts_per_bag : ℕ) (minutes_per_peanut : ℕ) : ℚ :=
  (bags * peanuts_per_bag * minutes_per_peanut) / 60

/-- Proves that the flight duration is 2 hours given the specified conditions. -/
theorem flight_duration_is_two_hours : 
  flight_duration 4 30 1 = 2 := by
  sorry

#eval flight_duration 4 30 1

end NUMINAMATH_CALUDE_flight_duration_is_two_hours_l658_65899


namespace NUMINAMATH_CALUDE_positive_polynomial_fraction_representation_l658_65811

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- A polynomial with non-negative coefficients -/
def NonNegativePolynomial (p : RealPolynomial) : Prop :=
  ∀ i, (p.coeff i) ≥ 0

/-- The theorem statement -/
theorem positive_polynomial_fraction_representation
  (P : RealPolynomial) (h : ∀ x : ℝ, x > 0 → P.eval x > 0) :
  ∃ (Q R : RealPolynomial), NonNegativePolynomial Q ∧ NonNegativePolynomial R ∧
    ∀ x : ℝ, x ≠ 0 → P.eval x = (Q.eval x) / (R.eval x) := by
  sorry

end NUMINAMATH_CALUDE_positive_polynomial_fraction_representation_l658_65811


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l658_65802

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.005 : ℝ)⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l658_65802


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l658_65889

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - k = 0 ∧ x = 3) → 
  (∃ y : ℝ, y^2 - 2*y - k = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l658_65889


namespace NUMINAMATH_CALUDE_circle_max_sum_squares_l658_65808

theorem circle_max_sum_squares :
  ∀ x y : ℝ, x^2 - 4*x - 4 + y^2 = 0 →
  x^2 + y^2 ≤ 12 + 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_sum_squares_l658_65808


namespace NUMINAMATH_CALUDE_softball_team_ratio_l658_65857

/-- Represents a co-ed softball team -/
structure SoftballTeam where
  men : ℕ
  women : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of men to women on the softball team -/
theorem softball_team_ratio (team : SoftballTeam) : 
  team.women = team.men + 4 → 
  team.men + team.women = 14 → 
  ∃ (r : Ratio), r.numerator = team.men ∧ r.denominator = team.women ∧ r.numerator = 5 ∧ r.denominator = 9 :=
by sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l658_65857


namespace NUMINAMATH_CALUDE_cube_root_four_solves_equation_l658_65891

theorem cube_root_four_solves_equation :
  let x : ℝ := (4 : ℝ) ^ (1/3)
  x^3 - ⌊x⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_four_solves_equation_l658_65891


namespace NUMINAMATH_CALUDE_no_two_digit_number_satisfies_conditions_l658_65815

theorem no_two_digit_number_satisfies_conditions : ¬ ∃ n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 2 = 0 ∧         -- even
  n % 13 = 0 ∧        -- multiple of 13
  ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  ∃ k : ℕ, a * b = k * k  -- product of digits is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_no_two_digit_number_satisfies_conditions_l658_65815


namespace NUMINAMATH_CALUDE_floor_product_inequality_l658_65865

theorem floor_product_inequality (m n : ℕ+) :
  ⌊Real.sqrt 2 * m⌋ * ⌊Real.sqrt 7 * n⌋ < ⌊Real.sqrt 14 * (m * n)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_product_inequality_l658_65865


namespace NUMINAMATH_CALUDE_word_count_between_czyeb_and_xceda_l658_65820

/-- Represents the set of available letters --/
inductive Letter : Type
  | A | B | C | D | E | X | Y | Z

/-- A word is a list of 5 letters --/
def Word := List Letter

/-- Convert a letter to its corresponding digit in base 8 --/
def letterToDigit (l : Letter) : Nat :=
  match l with
  | Letter.A => 0
  | Letter.B => 1
  | Letter.C => 2
  | Letter.D => 3
  | Letter.E => 4
  | Letter.X => 5
  | Letter.Y => 6
  | Letter.Z => 7

/-- Convert a word to its corresponding number in base 8 --/
def wordToNumber (w : Word) : Nat :=
  w.foldl (fun acc l => acc * 8 + letterToDigit l) 0

/-- The word CZYEB --/
def czyeb : Word := [Letter.C, Letter.Z, Letter.Y, Letter.E, Letter.B]

/-- The word XCEDA --/
def xceda : Word := [Letter.X, Letter.C, Letter.E, Letter.D, Letter.A]

/-- The theorem to be proved --/
theorem word_count_between_czyeb_and_xceda :
  (wordToNumber xceda) - (wordToNumber czyeb) - 1 = 9590 := by
  sorry

end NUMINAMATH_CALUDE_word_count_between_czyeb_and_xceda_l658_65820


namespace NUMINAMATH_CALUDE_average_monthly_growth_rate_l658_65883

theorem average_monthly_growth_rate 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (months : ℕ) 
  (h1 : initial_sales = 5000)
  (h2 : final_sales = 7200)
  (h3 : months = 2) :
  ∃ (rate : ℝ), 
    rate = 1/5 ∧ 
    initial_sales * (1 + rate) ^ months = final_sales :=
by sorry

end NUMINAMATH_CALUDE_average_monthly_growth_rate_l658_65883


namespace NUMINAMATH_CALUDE_max_sum_of_squared_unit_complex_l658_65878

theorem max_sum_of_squared_unit_complex (z : ℂ) (a b : ℝ) 
  (h1 : Complex.abs z = 1)
  (h2 : z^2 = Complex.mk a b) :
  ∃ (x y : ℝ), Complex.mk x y = z^2 ∧ x + y ≤ Real.sqrt 2 ∧
  ∀ (c d : ℝ), Complex.mk c d = z^2 → c + d ≤ x + y :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squared_unit_complex_l658_65878


namespace NUMINAMATH_CALUDE_smallest_possible_a_l658_65846

theorem smallest_possible_a (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (29 * ↑x)) :
  ∃ a_min : ℝ, a_min = 10 * Real.pi - 29 ∧ 
  (∀ a' : ℝ, (0 ≤ a' ∧ ∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (29 * ↑x)) → a_min ≤ a') :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l658_65846


namespace NUMINAMATH_CALUDE_food_lasts_fifty_days_l658_65833

/-- The number of days dog food will last given the number of dogs, meals per day, 
    food per meal, number of sacks, and weight per sack. -/
def days_food_lasts (num_dogs : ℕ) (meals_per_day : ℕ) (food_per_meal : ℕ) 
                    (num_sacks : ℕ) (weight_per_sack : ℕ) : ℕ :=
  (num_sacks * weight_per_sack * 1000) / (num_dogs * meals_per_day * food_per_meal)

/-- Proof that given the specific conditions, the food will last 50 days. -/
theorem food_lasts_fifty_days : 
  days_food_lasts 4 2 250 2 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_food_lasts_fifty_days_l658_65833


namespace NUMINAMATH_CALUDE_ricardo_coin_difference_l658_65894

theorem ricardo_coin_difference :
  ∀ (one_cent five_cent : ℕ),
    one_cent + five_cent = 2020 →
    one_cent ≥ 1 →
    five_cent ≥ 1 →
    (5 * 2019 + 1) - (2019 + 5) = 8072 :=
by
  sorry

end NUMINAMATH_CALUDE_ricardo_coin_difference_l658_65894


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_47_l658_65841

theorem smallest_four_digit_divisible_by_47 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 47 = 0 → 1034 ≤ n) ∧
  1000 ≤ 1034 ∧ 1034 < 10000 ∧ 1034 % 47 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_47_l658_65841


namespace NUMINAMATH_CALUDE_max_value_of_f_l658_65823

-- Define the function we want to maximize
def f (x : ℤ) : ℝ := 5 - |6 * x - 80|

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℤ), f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l658_65823


namespace NUMINAMATH_CALUDE_kerrys_age_l658_65854

/-- Given Kerry's birthday celebration setup, prove his age. -/
theorem kerrys_age :
  ∀ (num_cakes : ℕ) 
    (candles_per_box : ℕ) 
    (cost_per_box : ℚ) 
    (total_cost : ℚ),
  num_cakes = 3 →
  candles_per_box = 12 →
  cost_per_box = 5/2 →
  total_cost = 5 →
  ∃ (age : ℕ),
    age * num_cakes = (total_cost / cost_per_box) * candles_per_box ∧
    age = 8 := by
sorry

end NUMINAMATH_CALUDE_kerrys_age_l658_65854


namespace NUMINAMATH_CALUDE_f_convergence_and_g_continuity_l658_65838

noncomputable def f (x : ℝ) : ℕ → ℝ
  | 0 => Real.exp 1
  | n + 1 => Real.log x / Real.log (f x n)

def converges_to (s : ℕ → ℝ) (l : ℝ) :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |s n - l| < ε

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log (Real.log x)

theorem f_convergence_and_g_continuity (x : ℝ) (h : x > Real.exp (Real.exp 1)) :
  (∃ l, converges_to (f x) l) ∧ Continuous g :=
sorry

end NUMINAMATH_CALUDE_f_convergence_and_g_continuity_l658_65838


namespace NUMINAMATH_CALUDE_steps_to_rockefeller_center_l658_65847

theorem steps_to_rockefeller_center 
  (total_steps : ℕ) 
  (steps_to_times_square : ℕ) 
  (h1 : total_steps = 582) 
  (h2 : steps_to_times_square = 228) : 
  total_steps - steps_to_times_square = 354 := by
  sorry

end NUMINAMATH_CALUDE_steps_to_rockefeller_center_l658_65847


namespace NUMINAMATH_CALUDE_intersection_range_l658_65819

-- Define the points P and Q
def P : ℝ × ℝ := (-1, 1)
def Q : ℝ × ℝ := (2, 2)

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop := x + m * y + m = 0

-- Define the line PQ
def line_PQ (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem intersection_range :
  ∀ m : ℝ, 
  (∃ x y : ℝ, x > 2 ∧ line_equation m x y ∧ line_PQ x y) ↔ 
  -3 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l658_65819


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l658_65848

/-- 
Given a rectangular plot where:
- The area is 23 times its breadth
- The difference between the length and breadth is 10 metres

This theorem proves that the breadth of the plot is 13 metres.
-/
theorem rectangular_plot_breadth (length breadth : ℝ) 
  (h1 : length * breadth = 23 * breadth) 
  (h2 : length - breadth = 10) : 
  breadth = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l658_65848


namespace NUMINAMATH_CALUDE_exists_n_no_rational_roots_l658_65875

/-- A quadratic trinomial with real coefficients -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic trinomial at a given x -/
def QuadraticTrinomial.eval (p : QuadraticTrinomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Theorem: For any quadratic trinomial with real coefficients, 
    there exists a positive integer n such that p(x) = 1/n has no rational roots -/
theorem exists_n_no_rational_roots (p : QuadraticTrinomial) : 
  ∃ n : ℕ+, ¬∃ q : ℚ, p.eval q = (1 : ℝ) / n := by
  sorry

end NUMINAMATH_CALUDE_exists_n_no_rational_roots_l658_65875


namespace NUMINAMATH_CALUDE_stockholm_malmo_distance_l658_65890

/-- The scale factor of the map, representing kilometers per centimeter. -/
def scale : ℝ := 12

/-- The distance between Stockholm and Malmö on the map, in centimeters. -/
def map_distance : ℝ := 120

/-- The actual distance between Stockholm and Malmö, in kilometers. -/
def actual_distance : ℝ := map_distance * scale

/-- Theorem stating that the actual distance between Stockholm and Malmö is 1440 km. -/
theorem stockholm_malmo_distance : actual_distance = 1440 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_malmo_distance_l658_65890


namespace NUMINAMATH_CALUDE_product_factor_adjustment_l658_65850

theorem product_factor_adjustment (a b c : ℝ) (h1 : a * b = c) (h2 : a / 100 * (b * 100) = c) : 
  b * 100 = b * 100 := by sorry

end NUMINAMATH_CALUDE_product_factor_adjustment_l658_65850


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l658_65896

theorem inscribed_triangle_area (a b c : ℝ) (h : a + b + c = 12) :
  let r := 6 / Real.pi
  let s := (a * b * c * r) / (4 * 12)
  s = 9 / Real.pi^2 * (Real.sqrt 3 + 3) := by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l658_65896


namespace NUMINAMATH_CALUDE_perpendicular_iff_a_eq_neg_five_or_one_l658_65824

def line1 (a : ℝ) (x y : ℝ) : Prop := (2*a + 1)*x + (a + 5)*y - 6 = 0

def line2 (a : ℝ) (x y : ℝ) : Prop := (a + 5)*x + (a - 4)*y + 1 = 0

def perpendicular (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, line1 a x1 y1 ∧ line2 a x2 y2 →
    (2*a + 1)*(a + 5) + (a + 5)*(a - 4) = 0

theorem perpendicular_iff_a_eq_neg_five_or_one :
  ∀ a : ℝ, perpendicular a ↔ a = -5 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_iff_a_eq_neg_five_or_one_l658_65824


namespace NUMINAMATH_CALUDE_min_students_with_both_l658_65804

theorem min_students_with_both (n : ℕ) (glasses watches both : ℕ → ℕ) :
  (∀ m : ℕ, m ≥ n → glasses m = (3 * m) / 8) →
  (∀ m : ℕ, m ≥ n → watches m = (5 * m) / 6) →
  (∀ m : ℕ, m ≥ n → glasses m + watches m - both m = m) →
  (∃ m : ℕ, m ≥ n ∧ both m = 5 ∧ ∀ k, k < m → ¬(glasses k = (3 * k) / 8 ∧ watches k = (5 * k) / 6)) :=
sorry

end NUMINAMATH_CALUDE_min_students_with_both_l658_65804


namespace NUMINAMATH_CALUDE_tic_tac_toe_tournament_contradiction_l658_65855

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (participants : ℕ)

/-- Calculates the total number of matches in a single-elimination tournament -/
def total_matches (t : Tournament) : ℕ := t.participants - 1

/-- Represents the claims of some participants -/
structure Claims :=
  (num_claimants : ℕ)
  (matches_per_claimant : ℕ)

/-- Calculates the total number of matches implied by the claims -/
def implied_matches (c : Claims) : ℕ := c.num_claimants * c.matches_per_claimant / 2

theorem tic_tac_toe_tournament_contradiction (t : Tournament) (c : Claims) 
  (h1 : t.participants = 18)
  (h2 : c.num_claimants = 6)
  (h3 : c.matches_per_claimant = 4) :
  implied_matches c ≠ total_matches t :=
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tournament_contradiction_l658_65855


namespace NUMINAMATH_CALUDE_square_difference_equality_l658_65892

theorem square_difference_equality : (43 + 15)^2 - (43^2 + 15^2) = 1290 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l658_65892


namespace NUMINAMATH_CALUDE_wood_cutting_problem_l658_65834

theorem wood_cutting_problem (original_length : ℚ) (first_cut : ℚ) (second_cut : ℚ) :
  original_length = 35/8 ∧ first_cut = 5/3 ∧ second_cut = 9/4 →
  (original_length - first_cut - second_cut) / 3 = 11/72 := by
  sorry

end NUMINAMATH_CALUDE_wood_cutting_problem_l658_65834


namespace NUMINAMATH_CALUDE_room_053_selected_l658_65866

/-- Represents a room number in the range [1, 64] -/
def RoomNumber := Fin 64

/-- Systematic sampling function -/
def systematicSample (totalRooms sampleSize : ℕ) (firstSample : RoomNumber) : List RoomNumber :=
  let interval := totalRooms / sampleSize
  (List.range sampleSize).map (fun i => ⟨(firstSample.val + i * interval) % totalRooms + 1, sorry⟩)

theorem room_053_selected :
  let totalRooms := 64
  let sampleSize := 8
  let firstSample : RoomNumber := ⟨5, sorry⟩
  let sampledRooms := systematicSample totalRooms sampleSize firstSample
  53 ∈ sampledRooms.map (fun r => r.val) := by
  sorry

#eval systematicSample 64 8 ⟨5, sorry⟩

end NUMINAMATH_CALUDE_room_053_selected_l658_65866


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l658_65803

theorem partial_fraction_decomposition (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ 1 → 
    1 / (x^3 - 2*x^2 - 13*x + 10) = A / (x + 2) + B / (x - 1) + C / ((x - 1)^2)) →
  A = 1/9 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l658_65803


namespace NUMINAMATH_CALUDE_polynomial_factorization_l658_65828

def polynomial (n x y : ℤ) : ℤ := x^2 + 2*x*y + n*x^2 + y^2 + 2*y - n^2

def is_linear_factor (f : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b c : ℤ), ∀ (x y : ℤ), f x y = a*x + b*y + c

theorem polynomial_factorization (n : ℤ) :
  (∃ (f g : ℤ → ℤ → ℤ), is_linear_factor f ∧ is_linear_factor g ∧
    (∀ (x y : ℤ), polynomial n x y = f x y * g x y)) ↔ n = 0 ∨ n = 2 ∨ n = -2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l658_65828


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l658_65884

/-- Given two points C and D as endpoints of a diameter of a circle,
    calculate the area of the circle. -/
theorem circle_area_from_diameter_endpoints
  (C D : ℝ × ℝ) -- C and D are points in the real plane
  (h : C = (-2, 3) ∧ D = (4, -1)) -- C and D have specific coordinates
  : (π * ((C.1 - D.1)^2 + (C.2 - D.2)^2) / 4) = 13 * π := by
  sorry


end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l658_65884


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l658_65842

theorem gcd_from_lcm_and_ratio (A B : ℕ+) 
  (h_lcm : Nat.lcm A B = 180)
  (h_ratio : A * 6 = B * 5) :
  Nat.gcd A B = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l658_65842


namespace NUMINAMATH_CALUDE_tan_945_degrees_l658_65861

theorem tan_945_degrees (x : ℝ) : 
  (∀ x, Real.tan (x + 2 * Real.pi) = Real.tan x) → 
  Real.tan (945 * Real.pi / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_945_degrees_l658_65861


namespace NUMINAMATH_CALUDE_student_weight_l658_65844

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight - 5 = 2 * sister_weight)
  (h2 : student_weight + sister_weight = 116) :
  student_weight = 79 := by
sorry

end NUMINAMATH_CALUDE_student_weight_l658_65844


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l658_65898

theorem min_value_sum_reciprocals (a b : ℝ) (h : Real.log a + Real.log b = 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.log x + Real.log y = 0 ∧ 2/x + 1/y < 2/a + 1/b) ∨ 
  (2/a + 1/b = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l658_65898


namespace NUMINAMATH_CALUDE_coin_value_difference_max_value_achievable_min_value_achievable_l658_65840

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime

/-- The value of a coin in cents --/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10

/-- A distribution of coins --/
structure CoinDistribution where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  total_coins : pennies + nickels + dimes = 3030
  at_least_one : pennies ≥ 1 ∧ nickels ≥ 1 ∧ dimes ≥ 1

/-- The total value of a coin distribution in cents --/
def totalValue (d : CoinDistribution) : Nat :=
  d.pennies * coinValue CoinType.Penny +
  d.nickels * coinValue CoinType.Nickel +
  d.dimes * coinValue CoinType.Dime

/-- The maximum possible value for any valid coin distribution --/
def maxValue : Nat := 30286

/-- The minimum possible value for any valid coin distribution --/
def minValue : Nat := 3043

theorem coin_value_difference :
  maxValue - minValue = 27243 :=
by
  sorry

theorem max_value_achievable (d : CoinDistribution) :
  totalValue d ≤ maxValue :=
by
  sorry

theorem min_value_achievable (d : CoinDistribution) :
  totalValue d ≥ minValue :=
by
  sorry

end NUMINAMATH_CALUDE_coin_value_difference_max_value_achievable_min_value_achievable_l658_65840


namespace NUMINAMATH_CALUDE_m_range_l658_65837

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (2*m) - y^2 / (m-1) = 1 ∧ m > 0 ∧ m < 1/3

def q (m : ℝ) : Prop := ∃ (e : ℝ), ∃ (x y : ℝ), y^2 / 5 - x^2 / m = 1 ∧ 1 < e ∧ e < 2 ∧ m > 0 ∧ m < 15

-- State the theorem
theorem m_range :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (1/3 ≤ m ∧ m < 15) :=
by sorry

end NUMINAMATH_CALUDE_m_range_l658_65837


namespace NUMINAMATH_CALUDE_rachel_envelope_stuffing_l658_65825

/-- Rachel's envelope stuffing problem -/
theorem rachel_envelope_stuffing 
  (total_hours : ℕ) 
  (total_envelopes : ℕ) 
  (first_hour_envelopes : ℕ) 
  (h1 : total_hours = 8) 
  (h2 : total_envelopes = 1500) 
  (h3 : first_hour_envelopes = 135) :
  ∃ (second_hour_envelopes : ℕ),
    second_hour_envelopes = 195 ∧ 
    (total_envelopes - first_hour_envelopes - second_hour_envelopes) / (total_hours - 2) = 
    (total_envelopes - first_hour_envelopes - second_hour_envelopes) / (total_hours - 2) :=
by sorry


end NUMINAMATH_CALUDE_rachel_envelope_stuffing_l658_65825


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l658_65805

theorem x_gt_one_sufficient_not_necessary_for_abs_x_gt_one :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l658_65805


namespace NUMINAMATH_CALUDE_comparison_theorem_l658_65845

theorem comparison_theorem :
  (5.6 - 7/8 > 4.6) ∧ (638/81 < 271/29) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l658_65845


namespace NUMINAMATH_CALUDE_sum_product_equality_l658_65813

theorem sum_product_equality : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l658_65813


namespace NUMINAMATH_CALUDE_cyclic_equation_system_solution_l658_65800

theorem cyclic_equation_system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : (x₃ + x₄ + x₅)^5 = 3*x₁)
  (h₂ : (x₄ + x₅ + x₁)^5 = 3*x₂)
  (h₃ : (x₅ + x₁ + x₂)^5 = 3*x₃)
  (h₄ : (x₁ + x₂ + x₃)^5 = 3*x₄)
  (h₅ : (x₂ + x₃ + x₄)^5 = 3*x₅)
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0) (pos₄ : x₄ > 0) (pos₅ : x₅ > 0) :
  x₁ = 1/3 ∧ x₂ = 1/3 ∧ x₃ = 1/3 ∧ x₄ = 1/3 ∧ x₅ = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_equation_system_solution_l658_65800


namespace NUMINAMATH_CALUDE_b_95_mod_64_l658_65806

/-- The sequence b_n defined as 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The theorem stating that b_95 ≡ 48 (mod 64) -/
theorem b_95_mod_64 : b 95 ≡ 48 [ZMOD 64] := by
  sorry

end NUMINAMATH_CALUDE_b_95_mod_64_l658_65806


namespace NUMINAMATH_CALUDE_same_solution_systems_l658_65856

theorem same_solution_systems (m n : ℝ) : 
  (∃ x y : ℝ, 5*x - 2*y = 3 ∧ m*x + 5*y = 4 ∧ x - 4*y = -3 ∧ 5*x + n*y = 1) →
  m = -1 ∧ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_systems_l658_65856


namespace NUMINAMATH_CALUDE_profit_share_difference_l658_65877

/-- Represents the profit share of a party -/
structure ProfitShare where
  numerator : ℕ
  denominator : ℕ
  inv_pos : denominator > 0

/-- Calculates the profit for a given share and total profit -/
def calculate_profit (share : ProfitShare) (total_profit : ℚ) : ℚ :=
  total_profit * (share.numerator : ℚ) / (share.denominator : ℚ)

/-- The problem statement -/
theorem profit_share_difference 
  (total_profit : ℚ)
  (share_x share_y share_z : ProfitShare)
  (h_total : total_profit = 700)
  (h_x : share_x = ⟨1, 3, by norm_num⟩)
  (h_y : share_y = ⟨1, 4, by norm_num⟩)
  (h_z : share_z = ⟨1, 5, by norm_num⟩) :
  let profit_x := calculate_profit share_x total_profit
  let profit_y := calculate_profit share_y total_profit
  let profit_z := calculate_profit share_z total_profit
  let max_profit := max profit_x (max profit_y profit_z)
  let min_profit := min profit_x (min profit_y profit_z)
  ∃ (ε : ℚ), abs (max_profit - min_profit - 7148.93) < ε ∧ ε < 0.01 :=
sorry

end NUMINAMATH_CALUDE_profit_share_difference_l658_65877


namespace NUMINAMATH_CALUDE_first_fifth_mile_charge_l658_65814

/-- Represents the charge structure of a taxi company -/
structure TaxiCharge where
  first_fifth_mile : ℝ
  per_additional_fifth : ℝ

/-- Calculates the total charge for a given distance -/
def total_charge (c : TaxiCharge) (distance : ℝ) : ℝ :=
  c.first_fifth_mile + c.per_additional_fifth * (distance * 5 - 1)

/-- Theorem stating the charge for the first 1/5 mile -/
theorem first_fifth_mile_charge (c : TaxiCharge) :
  c.per_additional_fifth = 0.40 →
  total_charge c 8 = 18.10 →
  c.first_fifth_mile = 2.50 := by
sorry

end NUMINAMATH_CALUDE_first_fifth_mile_charge_l658_65814


namespace NUMINAMATH_CALUDE_all_turbans_zero_price_l658_65881

/-- Represents a servant's employment details -/
structure Servant where
  fullYearSalary : ℚ
  monthsWorked : ℚ
  actualPayment : ℚ

/-- Calculates the price of a turban given a servant's details -/
def turbanPrice (s : Servant) : ℚ :=
  s.actualPayment - (s.monthsWorked / 12) * s.fullYearSalary

/-- The main theorem proving that all turbans have zero price -/
theorem all_turbans_zero_price (servantA servantB servantC : Servant)
  (hA : servantA = { fullYearSalary := 120, monthsWorked := 8, actualPayment := 80 })
  (hB : servantB = { fullYearSalary := 150, monthsWorked := 7, actualPayment := 87.5 })
  (hC : servantC = { fullYearSalary := 180, monthsWorked := 10, actualPayment := 150 }) :
  turbanPrice servantA = 0 ∧ turbanPrice servantB = 0 ∧ turbanPrice servantC = 0 := by
  sorry


end NUMINAMATH_CALUDE_all_turbans_zero_price_l658_65881


namespace NUMINAMATH_CALUDE_planes_perpendicular_l658_65809

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem statement
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_parallel_α : line_parallel_plane m α)
  (h_n_perp_β : line_perpendicular_plane n β)
  (h_m_parallel_n : parallel m n) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l658_65809


namespace NUMINAMATH_CALUDE_job_completion_proof_l658_65870

/-- The number of days it takes the initial group of machines to finish the job -/
def initial_days : ℕ := 36

/-- The number of additional machines added -/
def additional_machines : ℕ := 5

/-- The number of days it takes after adding more machines -/
def reduced_days : ℕ := 27

/-- The number of machines initially working on the job -/
def initial_machines : ℕ := 20

theorem job_completion_proof :
  (initial_machines : ℚ) / initial_days = (initial_machines + additional_machines) / reduced_days :=
by sorry

end NUMINAMATH_CALUDE_job_completion_proof_l658_65870


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l658_65816

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    n = 100 ∧ 
    100 ≤ n ∧ n < 1000 ∧
    75 * n % 450 = 300 % 450 ∧
    (∀ m : ℕ, 100 ≤ m ∧ m < n → 75 * m % 450 ≠ 300 % 450) := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l658_65816


namespace NUMINAMATH_CALUDE_quadratic_maximum_value_l658_65888

theorem quadratic_maximum_value :
  ∃ (max : ℝ), max = 111 / 4 ∧ ∀ (x : ℝ), -3 * x^2 + 15 * x + 9 ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_value_l658_65888


namespace NUMINAMATH_CALUDE_power_of_five_times_112_l658_65872

theorem power_of_five_times_112 : (112 * 5^4) = 70000 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_times_112_l658_65872


namespace NUMINAMATH_CALUDE_compare_with_negative_three_l658_65858

theorem compare_with_negative_three : 
  let numbers : List ℝ := [-3, 2, 0, -4]
  numbers.filter (λ x => x < -3) = [-4] := by sorry

end NUMINAMATH_CALUDE_compare_with_negative_three_l658_65858


namespace NUMINAMATH_CALUDE_laptop_cost_proof_l658_65871

/-- The cost of the laptop satisfies the given conditions -/
theorem laptop_cost_proof (monthly_installment : ℝ) (down_payment_percent : ℝ) 
  (additional_down_payment : ℝ) (months_paid : ℕ) (remaining_balance : ℝ) :
  monthly_installment = 65 →
  down_payment_percent = 0.2 →
  additional_down_payment = 20 →
  months_paid = 4 →
  remaining_balance = 520 →
  ∃ (cost : ℝ), 
    cost - (down_payment_percent * cost + additional_down_payment + monthly_installment * months_paid) = remaining_balance ∧
    cost = 1000 := by
  sorry

end NUMINAMATH_CALUDE_laptop_cost_proof_l658_65871


namespace NUMINAMATH_CALUDE_max_value_K_l658_65874

/-- The maximum value of K for x₁, x₂, x₃, x₄ ∈ [0,1] --/
theorem max_value_K : 
  ∃ (K_max : ℝ), K_max = Real.sqrt 5 / 125 ∧ 
  ∀ (x₁ x₂ x₃ x₄ : ℝ), 
    0 ≤ x₁ ∧ x₁ ≤ 1 ∧ 
    0 ≤ x₂ ∧ x₂ ≤ 1 ∧ 
    0 ≤ x₃ ∧ x₃ ≤ 1 ∧ 
    0 ≤ x₄ ∧ x₄ ≤ 1 → 
    let K := |x₁ - x₂| * |x₁ - x₃| * |x₁ - x₄| * |x₂ - x₃| * |x₂ - x₄| * |x₃ - x₄|
    K ≤ K_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_K_l658_65874


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l658_65897

/-- A number is divisible by 8 if and only if its last three digits form a number divisible by 8 -/
axiom divisible_by_8 (n : ℕ) : n % 8 = 0 ↔ (n % 1000) % 8 = 0

/-- The largest five-digit number -/
def largest_five_digit : ℕ := 99999

/-- The largest five-digit number divisible by 8 -/
def largest_five_digit_div_8 : ℕ := 99992

theorem largest_five_digit_divisible_by_8 :
  largest_five_digit_div_8 % 8 = 0 ∧
  ∀ n : ℕ, n > largest_five_digit_div_8 → n ≤ largest_five_digit → n % 8 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l658_65897


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l658_65843

theorem sqrt_x_minus_one_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l658_65843


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_equation_l658_65810

theorem smallest_n_for_candy_equation : ∃ (n : ℕ), n > 0 ∧
  (∀ (r g b y : ℕ), r > 0 ∧ g > 0 ∧ b > 0 ∧ y > 0 →
    (10 * r = 8 * g ∧ 8 * g = 9 * b ∧ 9 * b = 12 * y ∧ 12 * y = 18 * n) →
    (∀ (m : ℕ), m > 0 ∧ m < n →
      ¬(∃ (r' g' b' y' : ℕ), r' > 0 ∧ g' > 0 ∧ b' > 0 ∧ y' > 0 ∧
        10 * r' = 8 * g' ∧ 8 * g' = 9 * b' ∧ 9 * b' = 12 * y' ∧ 12 * y' = 18 * m))) ∧
  n = 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_equation_l658_65810


namespace NUMINAMATH_CALUDE_mink_babies_problem_l658_65849

/-- Represents the problem of determining the number of babies each mink had --/
theorem mink_babies_problem (initial_minks : ℕ) (coats_made : ℕ) (skins_per_coat : ℕ) 
  (h1 : initial_minks = 30)
  (h2 : coats_made = 7)
  (h3 : skins_per_coat = 15) :
  ∃ babies_per_mink : ℕ, 
    (initial_minks + initial_minks * babies_per_mink) / 2 = coats_made * skins_per_coat ∧ 
    babies_per_mink = 6 := by
  sorry

end NUMINAMATH_CALUDE_mink_babies_problem_l658_65849


namespace NUMINAMATH_CALUDE_workshop_average_salary_l658_65863

/-- Proves that the average salary of all workers is 8000 Rs given the specified conditions -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technician_count : ℕ)
  (technician_avg_salary : ℕ)
  (non_technician_avg_salary : ℕ)
  (h1 : total_workers = 14)
  (h2 : technician_count = 7)
  (h3 : technician_avg_salary = 10000)
  (h4 : non_technician_avg_salary = 6000) :
  (technician_count * technician_avg_salary + (total_workers - technician_count) * non_technician_avg_salary) / total_workers = 8000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l658_65863


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l658_65852

/-- The rate per kg of grapes -/
def grape_rate : ℝ := 70

/-- The amount of grapes purchased in kg -/
def grape_amount : ℝ := 10

/-- The rate per kg of mangoes -/
def mango_rate : ℝ := 55

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount paid to the shopkeeper -/
def total_paid : ℝ := 1195

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l658_65852


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l658_65812

/-- Given that i is the imaginary unit, prove that (1+3i)/(1+i) = 2+i -/
theorem complex_fraction_equality : (1 + 3 * I) / (1 + I) = 2 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l658_65812


namespace NUMINAMATH_CALUDE_solution_set_f_geq_x_min_value_a_min_a_is_three_l658_65826

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| - |2*x + 3|

-- Theorem for part (I)
theorem solution_set_f_geq_x : 
  {x : ℝ | f x ≥ x} = {x : ℝ | x ≥ 4/5} := by sorry

-- Theorem for part (II)
theorem min_value_a (m : ℝ) (h : m > 0) :
  (∀ (x y : ℝ), f x ≤ m^y + a/m^y) → a ≥ 3 := by sorry

-- Theorem for the minimum value of a
theorem min_a_is_three :
  ∃ (a : ℝ), a = 3 ∧ 
  (∀ (m : ℝ), m > 0 → ∀ (x y : ℝ), f x ≤ m^y + a/m^y) ∧
  (∀ (a' : ℝ), (∀ (m : ℝ), m > 0 → ∀ (x y : ℝ), f x ≤ m^y + a'/m^y) → a' ≥ a) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_x_min_value_a_min_a_is_three_l658_65826


namespace NUMINAMATH_CALUDE_nanoseconds_to_scientific_notation_l658_65851

/-- Conversion factor from nanoseconds to seconds -/
def nanosecond_to_second : ℝ := 1e-9

/-- The number of nanoseconds we want to convert -/
def nanoseconds : ℝ := 20

/-- The expected result in scientific notation (in seconds) -/
def expected_result : ℝ := 2e-8

theorem nanoseconds_to_scientific_notation :
  nanoseconds * nanosecond_to_second = expected_result := by
  sorry

end NUMINAMATH_CALUDE_nanoseconds_to_scientific_notation_l658_65851


namespace NUMINAMATH_CALUDE_megan_spelling_problems_l658_65822

/-- The number of spelling problems Megan had to solve -/
def spelling_problems (math_problems : ℕ) (problems_per_hour : ℕ) (total_hours : ℕ) : ℕ :=
  problems_per_hour * total_hours - math_problems

theorem megan_spelling_problems :
  spelling_problems 36 8 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_megan_spelling_problems_l658_65822


namespace NUMINAMATH_CALUDE_exists_digit_sum_div_11_l658_65887

def digit_sum (n : ℕ) : ℕ := sorry

theorem exists_digit_sum_div_11 (n : ℕ) : ∃ k, n ≤ k ∧ k < n + 39 ∧ (digit_sum k) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_digit_sum_div_11_l658_65887


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l658_65835

/-- Given a box of 16 pens with 3 defective pens, prove that the probability
    of selecting 2 non-defective pens at random is 13/20. -/
theorem prob_two_non_defective_pens :
  let total_pens : ℕ := 16
  let defective_pens : ℕ := 3
  let non_defective_pens : ℕ := total_pens - defective_pens
  let prob_first_non_defective : ℚ := non_defective_pens / total_pens
  let prob_second_non_defective : ℚ := (non_defective_pens - 1) / (total_pens - 1)
  prob_first_non_defective * prob_second_non_defective = 13 / 20 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l658_65835


namespace NUMINAMATH_CALUDE_function_positivity_and_inequality_l658_65818

/-- The function f(x) = mx² + mx + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + m * x + 3

/-- The function g(x) = (3m-1)x + 5 -/
def g (m : ℝ) (x : ℝ) : ℝ := (3*m - 1) * x + 5

theorem function_positivity_and_inequality (m : ℝ) :
  (∀ x : ℝ, f m x > 0) ↔ (0 ≤ m ∧ m < 12) ∧
  (∀ x : ℝ, f m x > g m x ↔
    (m < -1/2 ∧ -1/m < x ∧ x < 2) ∨
    (m = -1/2 ∧ False) ∨
    (-1/2 < m ∧ m < 0 ∧ 2 < x ∧ x < -1/m) ∨
    (m = 0 ∧ x > 2) ∨
    (m > 0 ∧ (x < -1/m ∨ x > 2))) :=
sorry

end NUMINAMATH_CALUDE_function_positivity_and_inequality_l658_65818


namespace NUMINAMATH_CALUDE_correct_oranges_to_put_back_l658_65869

/-- The number of oranges Mary must put back to achieve the desired average price -/
def oranges_to_put_back (apple_price orange_price : ℚ) (total_fruit : ℕ) (initial_avg_price desired_avg_price : ℚ) : ℕ :=
  sorry

theorem correct_oranges_to_put_back :
  oranges_to_put_back (40/100) (60/100) 10 (54/100) (50/100) = 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_to_put_back_l658_65869


namespace NUMINAMATH_CALUDE_shekar_average_marks_l658_65882

def shekar_scores : List Nat := [76, 65, 82, 67, 55]

theorem shekar_average_marks :
  let total_marks := shekar_scores.sum
  let num_subjects := shekar_scores.length
  (total_marks / num_subjects : ℚ) = 69 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l658_65882


namespace NUMINAMATH_CALUDE_complex_integer_sum_of_squares_l658_65864

theorem complex_integer_sum_of_squares (x y : ℤ) :
  (∃ (a b c d : ℤ), x + y * I = (a + b * I)^2 + (c + d * I)^2) ↔ Even y := by
  sorry

end NUMINAMATH_CALUDE_complex_integer_sum_of_squares_l658_65864


namespace NUMINAMATH_CALUDE_diophantine_approximation_l658_65876

theorem diophantine_approximation (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℕ), 1 ≤ b ∧ b ≤ n ∧ |a - b * Real.sqrt 2| ≤ 1 / n := by
  sorry

end NUMINAMATH_CALUDE_diophantine_approximation_l658_65876


namespace NUMINAMATH_CALUDE_halloween_goodie_bags_cost_l658_65853

/-- Represents the minimum cost to purchase Halloween goodie bags --/
def minimum_cost (total_students : ℕ) (vampire_students : ℕ) (pumpkin_students : ℕ) 
  (package_size : ℕ) (package_cost : ℚ) (individual_cost : ℚ) : ℚ :=
  let vampire_packages := (vampire_students + package_size - 1) / package_size
  let pumpkin_packages := pumpkin_students / package_size
  let pumpkin_individual := pumpkin_students % package_size
  let base_cost := (vampire_packages + pumpkin_packages) * package_cost + 
                   pumpkin_individual * individual_cost
  let discounted_cost := if base_cost > 10 then base_cost * (1 - 0.1) else base_cost
  ⌈discounted_cost * 100⌉ / 100

/-- Theorem stating the minimum cost for Halloween goodie bags --/
theorem halloween_goodie_bags_cost :
  minimum_cost 25 11 14 5 3 1 = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_halloween_goodie_bags_cost_l658_65853


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l658_65831

theorem unique_solution_to_equation : ∃! t : ℝ, 4 * (4 : ℝ)^t + Real.sqrt (16 * 16^t) + 2^t = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l658_65831
