import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_monotonicity_like_property_l844_84441

/-- The negation of a monotonicity-like property for a real-valued function -/
theorem negation_of_monotonicity_like_property (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_monotonicity_like_property_l844_84441


namespace NUMINAMATH_CALUDE_max_xy_value_l844_84446

theorem max_xy_value (x y : ℝ) (h : x^2 + y^2 + 3*x*y = 2015) : 
  ∀ a b : ℝ, a^2 + b^2 + 3*a*b = 2015 → x*y ≤ 403 ∧ ∃ c d : ℝ, c^2 + d^2 + 3*c*d = 2015 ∧ c*d = 403 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l844_84446


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l844_84417

theorem unique_solution_floor_equation :
  ∃! b : ℝ, b + ⌊b⌋ = 14.3 ∧ b = 7.3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l844_84417


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l844_84468

theorem quadratic_equation_root : ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 9 = 0) ∧
  ((-3 : ℝ)^2 - 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l844_84468


namespace NUMINAMATH_CALUDE_jewelry_sales_fraction_l844_84402

theorem jewelry_sales_fraction (total_sales : ℕ) (stationery_sales : ℕ) :
  total_sales = 36 →
  stationery_sales = 15 →
  (total_sales : ℚ) / 3 + stationery_sales + (total_sales : ℚ) / 4 = total_sales :=
by
  sorry

end NUMINAMATH_CALUDE_jewelry_sales_fraction_l844_84402


namespace NUMINAMATH_CALUDE_saree_sale_price_l844_84482

/-- Calculates the final price of a saree after discounts and tax -/
def finalSalePrice (initialPrice : ℝ) (discount1 discount2 discount3 taxRate : ℝ) : ℝ :=
  let price1 := initialPrice * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let price3 := price2 * (1 - discount3)
  price3 * (1 + taxRate)

/-- The final sale price of a saree is approximately 298.55 Rs -/
theorem saree_sale_price :
  ∃ ε > 0, abs (finalSalePrice 560 0.2 0.3 0.15 0.12 - 298.55) < ε :=
sorry

end NUMINAMATH_CALUDE_saree_sale_price_l844_84482


namespace NUMINAMATH_CALUDE_band_problem_solution_l844_84475

def band_problem (num_flutes num_clarinets num_trumpets num_total : ℕ) 
  (flute_ratio clarinet_ratio trumpet_ratio pianist_ratio : ℚ) : Prop :=
  let flutes_in := (num_flutes : ℚ) * flute_ratio
  let clarinets_in := (num_clarinets : ℚ) * clarinet_ratio
  let trumpets_in := (num_trumpets : ℚ) * trumpet_ratio
  let non_pianists_in := flutes_in + clarinets_in + trumpets_in
  let pianists_in := (num_total : ℚ) - non_pianists_in
  ∃ (num_pianists : ℕ), (num_pianists : ℚ) * pianist_ratio = pianists_in ∧ num_pianists = 20

theorem band_problem_solution :
  band_problem 20 30 60 53 (4/5) (1/2) (1/3) (1/10) :=
sorry

end NUMINAMATH_CALUDE_band_problem_solution_l844_84475


namespace NUMINAMATH_CALUDE_factorization_of_16x_squared_minus_4_l844_84410

theorem factorization_of_16x_squared_minus_4 (x : ℝ) :
  16 * x^2 - 4 = 4 * (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_16x_squared_minus_4_l844_84410


namespace NUMINAMATH_CALUDE_people_per_column_l844_84428

theorem people_per_column (P : ℕ) (X : ℕ) : 
  P = 30 * 16 → P = X * 40 → X = 12 := by
  sorry

end NUMINAMATH_CALUDE_people_per_column_l844_84428


namespace NUMINAMATH_CALUDE_panda_babies_count_l844_84497

def total_couples : ℕ := 100
def young_percentage : ℚ := 1/5
def adult_percentage : ℚ := 3/5
def old_percentage : ℚ := 1/5

def young_pregnancy_chance : ℚ := 2/5
def adult_pregnancy_chance : ℚ := 1/4
def old_pregnancy_chance : ℚ := 1/10

def average_babies_per_pregnancy : ℚ := 3/2

def young_babies : ℕ := 12
def adult_babies : ℕ := 22
def old_babies : ℕ := 3

theorem panda_babies_count :
  young_babies + adult_babies + old_babies = 37 :=
by sorry

end NUMINAMATH_CALUDE_panda_babies_count_l844_84497


namespace NUMINAMATH_CALUDE_bookshelf_cost_price_l844_84426

/-- The cost price of a bookshelf sold at a loss and would have made a profit with additional revenue -/
theorem bookshelf_cost_price (C : ℝ) : C = 1071.43 :=
  let SP := 0.76 * C
  have h1 : SP = 0.76 * C := by rfl
  have h2 : SP + 450 = 1.18 * C := by sorry
  sorry

end NUMINAMATH_CALUDE_bookshelf_cost_price_l844_84426


namespace NUMINAMATH_CALUDE_password_count_l844_84449

/-- The number of possible values for the last two digits of a birth year. -/
def year_choices : ℕ := 100

/-- The number of possible values for the birth month. -/
def month_choices : ℕ := 12

/-- The number of possible values for the birth date. -/
def day_choices : ℕ := 31

/-- The total number of possible six-digit passwords. -/
def total_passwords : ℕ := year_choices * month_choices * day_choices

theorem password_count : total_passwords = 37200 := by
  sorry

end NUMINAMATH_CALUDE_password_count_l844_84449


namespace NUMINAMATH_CALUDE_conference_handshakes_l844_84469

theorem conference_handshakes (n : ℕ) (h : n = 12) : n.choose 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l844_84469


namespace NUMINAMATH_CALUDE_unique_angle_l844_84412

def is_valid_angle (a b c d e f : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  0 ≤ f ∧ f ≤ 9 ∧
  10 * a + b < 90 ∧
  10 * c + d < 60 ∧
  10 * e + f < 60

def is_complement (a b c d e f a1 b1 c1 d1 e1 f1 : ℕ) : Prop :=
  (10 * a + b) + (10 * a1 + b1) = 89 ∧
  (10 * c + d) + (10 * c1 + d1) = 59 ∧
  (10 * e + f) + (10 * e1 + f1) = 60

def is_rearrangement (a b c d e f a1 b1 c1 d1 e1 f1 : ℕ) : Prop :=
  ∃ (n m : ℕ), n + m = 6 ∧ n ≤ m ∧
  (10^n + 1) * (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) +
  (10^m + 1) * (100000 * a1 + 10000 * b1 + 1000 * c1 + 100 * d1 + 10 * e1 + f1) = 895960

theorem unique_angle :
  ∀ (a b c d e f : ℕ),
    is_valid_angle a b c d e f →
    (∃ (a1 b1 c1 d1 e1 f1 : ℕ),
      is_valid_angle a1 b1 c1 d1 e1 f1 ∧
      is_complement a b c d e f a1 b1 c1 d1 e1 f1 ∧
      is_rearrangement a b c d e f a1 b1 c1 d1 e1 f1) →
    a = 4 ∧ b = 5 ∧ c = 4 ∧ d = 4 ∧ e = 1 ∧ f = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_angle_l844_84412


namespace NUMINAMATH_CALUDE_hexagon_same_length_probability_l844_84401

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements (sides and diagonals) in a regular hexagon -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two segments of the same length from a regular hexagon -/
def prob_same_length : ℚ := 17/35

theorem hexagon_same_length_probability :
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) / (total_elements * (total_elements - 1)) = prob_same_length := by
  sorry

end NUMINAMATH_CALUDE_hexagon_same_length_probability_l844_84401


namespace NUMINAMATH_CALUDE_truck_rental_miles_driven_l844_84414

theorem truck_rental_miles_driven 
  (rental_fee : ℝ) 
  (charge_per_mile : ℝ) 
  (total_paid : ℝ) 
  (h1 : rental_fee = 20.99)
  (h2 : charge_per_mile = 0.25)
  (h3 : total_paid = 95.74) : 
  ⌊(total_paid - rental_fee) / charge_per_mile⌋ = 299 := by
sorry


end NUMINAMATH_CALUDE_truck_rental_miles_driven_l844_84414


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l844_84437

theorem complex_fraction_calculation : 
  (2 + 2/3 : ℚ) * ((1/3 - 1/11) / (1/11 + 1/5)) / (8/27) = 7 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l844_84437


namespace NUMINAMATH_CALUDE_power_of_81_l844_84419

theorem power_of_81 : 81^(8/3) = 59049 * (9^(1/3)) := by sorry

end NUMINAMATH_CALUDE_power_of_81_l844_84419


namespace NUMINAMATH_CALUDE_exponential_decrease_l844_84404

theorem exponential_decrease (x y a : Real) 
  (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : 
  a^x < a^y := by
  sorry

end NUMINAMATH_CALUDE_exponential_decrease_l844_84404


namespace NUMINAMATH_CALUDE_extreme_values_and_interval_extrema_l844_84472

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 3/2]
def interval : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3/2}

theorem extreme_values_and_interval_extrema :
  -- Global maximum
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ f x) ∧
  -- Global minimum
  (∃ (x : ℝ), f x = -2 ∧ ∀ (y : ℝ), f y ≥ f x) ∧
  -- Maximum on the interval
  (∃ (x : ℝ), x ∈ interval ∧ f x = 2 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x) ∧
  -- Minimum on the interval
  (∃ (x : ℝ), x ∈ interval ∧ f x = -18 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x) :=
by sorry


end NUMINAMATH_CALUDE_extreme_values_and_interval_extrema_l844_84472


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_5_256_minus_1_l844_84479

theorem largest_power_of_two_dividing_5_256_minus_1 :
  (∃ n : ℕ, 2^n ∣ (5^256 - 1) ∧ ∀ m : ℕ, 2^m ∣ (5^256 - 1) → m ≤ n) →
  (∃ n : ℕ, 2^n ∣ (5^256 - 1) ∧ ∀ m : ℕ, 2^m ∣ (5^256 - 1) → m ≤ n) ∧
  (∀ n : ℕ, 2^n ∣ (5^256 - 1) ∧ ∀ m : ℕ, 2^m ∣ (5^256 - 1) → m ≤ n → n = 10) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_5_256_minus_1_l844_84479


namespace NUMINAMATH_CALUDE_circle_area_l844_84433

theorem circle_area (x y : ℝ) : 
  (2 * x^2 + 2 * y^2 + 8 * x - 4 * y - 16 = 0) → 
  (∃ (center_x center_y radius : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = radius^2 ∧ 
    π * radius^2 = 13 * π) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_l844_84433


namespace NUMINAMATH_CALUDE_polynomial_factor_l844_84451

theorem polynomial_factor (x : ℝ) :
  ∃ (k : ℝ), (29 * 37 * x^4 + 2 * x^2 + 9) = k * (x^2 - 2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l844_84451


namespace NUMINAMATH_CALUDE_oranges_packed_in_week_l844_84440

/-- The number of oranges packed in a full week given the daily packing rate and box capacity -/
theorem oranges_packed_in_week
  (oranges_per_box : ℕ)
  (boxes_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : oranges_per_box = 15)
  (h2 : boxes_per_day = 2150)
  (h3 : days_in_week = 7) :
  oranges_per_box * boxes_per_day * days_in_week = 225750 := by
  sorry

end NUMINAMATH_CALUDE_oranges_packed_in_week_l844_84440


namespace NUMINAMATH_CALUDE_T4_championship_probability_l844_84411

/-- Represents a team in the playoffs -/
inductive Team : Type
| T1 : Team
| T2 : Team
| T3 : Team
| T4 : Team

/-- The probability of team i winning against team j -/
def winProbability (i j : Team) : ℚ :=
  match i, j with
  | Team.T1, Team.T2 => 1/3
  | Team.T1, Team.T3 => 1/4
  | Team.T1, Team.T4 => 1/5
  | Team.T2, Team.T1 => 2/3
  | Team.T2, Team.T3 => 2/5
  | Team.T2, Team.T4 => 1/3
  | Team.T3, Team.T1 => 3/4
  | Team.T3, Team.T2 => 3/5
  | Team.T3, Team.T4 => 3/7
  | Team.T4, Team.T1 => 4/5
  | Team.T4, Team.T2 => 2/3
  | Team.T4, Team.T3 => 4/7
  | _, _ => 1/2  -- This case should never occur in our scenario

/-- The probability of T4 winning the championship -/
def T4ChampionshipProbability : ℚ :=
  (winProbability Team.T4 Team.T1) * 
  ((winProbability Team.T3 Team.T2) * (winProbability Team.T4 Team.T3) +
   (winProbability Team.T2 Team.T3) * (winProbability Team.T4 Team.T2))

theorem T4_championship_probability :
  T4ChampionshipProbability = 256/525 := by
  sorry

#eval T4ChampionshipProbability

end NUMINAMATH_CALUDE_T4_championship_probability_l844_84411


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l844_84456

/-- The speed of the east-bound cyclist in mph -/
def east_speed : ℝ := 18

/-- The speed of the west-bound cyclist in mph -/
def west_speed : ℝ := east_speed + 4

/-- The time traveled in hours -/
def time : ℝ := 5

/-- The total distance between the cyclists after the given time -/
def total_distance : ℝ := 200

theorem cyclist_speed_proof :
  east_speed * time + west_speed * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l844_84456


namespace NUMINAMATH_CALUDE_complex_fraction_equality_complex_fraction_equality_proof_l844_84481

theorem complex_fraction_equality : ℂ → Prop :=
  λ i => (5 * i) / (2 - i) = -1 + 2 * i

-- The proof goes here
theorem complex_fraction_equality_proof : complex_fraction_equality I :=
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_complex_fraction_equality_proof_l844_84481


namespace NUMINAMATH_CALUDE_potatoes_per_bag_l844_84492

/-- Proves that the number of pounds of potatoes in one bag is 20 -/
theorem potatoes_per_bag (potatoes_per_person : ℝ) (num_people : ℕ) (cost_per_bag : ℝ) (total_cost : ℝ) :
  potatoes_per_person = 1.5 →
  num_people = 40 →
  cost_per_bag = 5 →
  total_cost = 15 →
  (num_people * potatoes_per_person) / (total_cost / cost_per_bag) = 20 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_per_bag_l844_84492


namespace NUMINAMATH_CALUDE_units_digit_sum_base9_l844_84478

/-- The units digit of a number in base 9 -/
def unitsDigitBase9 (n : ℕ) : ℕ := n % 9

/-- Addition in base 9 -/
def addBase9 (a b : ℕ) : ℕ := (a + b) % 9

theorem units_digit_sum_base9 :
  unitsDigitBase9 (addBase9 45 76) = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_base9_l844_84478


namespace NUMINAMATH_CALUDE_candy_distribution_properties_l844_84415

-- Define the people
inductive Person : Type
| Chun : Person
| Tian : Person
| Zhen : Person
| Mei : Person
| Li : Person

-- Define the order of taking candies
def Order := Fin 5 → Person

-- Define the number of candies taken by each person
def CandiesTaken := Person → ℕ

-- Define the properties of the candy distribution
structure CandyDistribution where
  order : Order
  candiesTaken : CandiesTaken
  initialCandies : ℕ
  allDifferent : ∀ (p q : Person), p ≠ q → candiesTaken p ≠ candiesTaken q
  tianHalf : candiesTaken Person.Tian = (initialCandies - candiesTaken Person.Chun) / 2
  zhenTwoThirds : candiesTaken Person.Zhen = 2 * (initialCandies - candiesTaken Person.Chun - candiesTaken Person.Tian - candiesTaken Person.Li) / 3
  meiAll : candiesTaken Person.Mei = initialCandies - candiesTaken Person.Chun - candiesTaken Person.Tian - candiesTaken Person.Zhen - candiesTaken Person.Li
  liHalf : candiesTaken Person.Li = (initialCandies - candiesTaken Person.Chun - candiesTaken Person.Tian) / 2

-- Theorem statement
theorem candy_distribution_properties (d : CandyDistribution) :
  (∃ i : Fin 5, d.order i = Person.Zhen ∧ i = 3) ∧
  d.initialCandies ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_properties_l844_84415


namespace NUMINAMATH_CALUDE_place_two_after_three_digit_number_l844_84424

/-- Given a three-digit number with hundreds digit a, tens digit b, and units digit c,
    prove that placing the digit 2 after this number results in 1000a + 100b + 10c + 2 -/
theorem place_two_after_three_digit_number (a b c : ℕ) :
  let original := 100 * a + 10 * b + c
  10 * original + 2 = 1000 * a + 100 * b + 10 * c + 2 := by
  sorry

end NUMINAMATH_CALUDE_place_two_after_three_digit_number_l844_84424


namespace NUMINAMATH_CALUDE_coupon_value_l844_84486

def vacuum_cost : ℝ := 250
def dishwasher_cost : ℝ := 450
def total_cost_after_coupon : ℝ := 625

theorem coupon_value : 
  vacuum_cost + dishwasher_cost - total_cost_after_coupon = 75 := by
  sorry

end NUMINAMATH_CALUDE_coupon_value_l844_84486


namespace NUMINAMATH_CALUDE_f_of_4_6_l844_84498

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

theorem f_of_4_6 : f (4, 6) = (10, -2) := by
  sorry

end NUMINAMATH_CALUDE_f_of_4_6_l844_84498


namespace NUMINAMATH_CALUDE_sallys_remaining_cards_l844_84425

/-- Given Sally's initial number of baseball cards, the number of torn cards, 
    and the number of cards Sara bought, prove the number of cards Sally has now. -/
theorem sallys_remaining_cards (initial_cards torn_cards cards_bought : ℕ) :
  initial_cards = 39 →
  torn_cards = 9 →
  cards_bought = 24 →
  initial_cards - torn_cards - cards_bought = 6 := by
  sorry

end NUMINAMATH_CALUDE_sallys_remaining_cards_l844_84425


namespace NUMINAMATH_CALUDE_total_legs_count_l844_84447

theorem total_legs_count (total_tables : ℕ) (four_leg_tables : ℕ) : 
  total_tables = 36 → four_leg_tables = 16 → 
  (∃ (three_leg_tables : ℕ), 
    three_leg_tables + four_leg_tables = total_tables ∧
    3 * three_leg_tables + 4 * four_leg_tables = 124) := by
  sorry

end NUMINAMATH_CALUDE_total_legs_count_l844_84447


namespace NUMINAMATH_CALUDE_max_value_ln_x_over_x_l844_84483

/-- The function f(x) = ln(x) / x attains its maximum value at e^(-1) for x > 0 -/
theorem max_value_ln_x_over_x : 
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → (Real.log x) / x ≥ (Real.log y) / y ∧ (Real.log x) / x = Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_max_value_ln_x_over_x_l844_84483


namespace NUMINAMATH_CALUDE_blanket_average_price_l844_84407

/-- Given the following conditions:
    - A man purchased 8 blankets in total
    - 1 blanket costs Rs. 100
    - 5 blankets cost Rs. 150 each
    - 2 blankets cost Rs. 650 in total
    Prove that the average price of all blankets is Rs. 187.50 -/
theorem blanket_average_price :
  let total_blankets : ℕ := 8
  let price_of_one : ℕ := 100
  let price_of_five : ℕ := 150
  let price_of_two : ℕ := 650
  let total_cost : ℕ := price_of_one + 5 * price_of_five + price_of_two
  (total_cost : ℚ) / total_blankets = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_blanket_average_price_l844_84407


namespace NUMINAMATH_CALUDE_breakfast_eggs_count_l844_84474

def total_eggs : ℕ := 6
def lunch_eggs : ℕ := 3
def dinner_eggs : ℕ := 1

theorem breakfast_eggs_count :
  total_eggs - lunch_eggs - dinner_eggs = 2 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_eggs_count_l844_84474


namespace NUMINAMATH_CALUDE_sixty_second_pair_l844_84458

/-- Definition of our sequence of pairs -/
def pair_sequence : ℕ → ℕ × ℕ
| 0 => (1, 1)
| n + 1 =>
  let (a, b) := pair_sequence n
  if a = 1 then (b + 1, 1)
  else (a - 1, b + 1)

/-- The 62nd pair in the sequence is (7,5) -/
theorem sixty_second_pair :
  pair_sequence 61 = (7, 5) :=
sorry

end NUMINAMATH_CALUDE_sixty_second_pair_l844_84458


namespace NUMINAMATH_CALUDE_set_A_equals_zero_one_l844_84434

def A : Set ℤ := {x | (2 * x - 3) / (x + 1) ≤ 0}

theorem set_A_equals_zero_one : A = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_A_equals_zero_one_l844_84434


namespace NUMINAMATH_CALUDE_select_and_order_two_from_five_eq_twenty_l844_84496

/-- The number of ways to select and order 2 items from a set of 5 distinct items -/
def select_and_order_two_from_five : ℕ :=
  5 * 4

/-- Theorem: The number of ways to select and order 2 items from a set of 5 distinct items is 20 -/
theorem select_and_order_two_from_five_eq_twenty :
  select_and_order_two_from_five = 20 := by
  sorry

end NUMINAMATH_CALUDE_select_and_order_two_from_five_eq_twenty_l844_84496


namespace NUMINAMATH_CALUDE_x_squared_plus_7x_plus_12_bounds_l844_84494

theorem x_squared_plus_7x_plus_12_bounds (x : ℝ) (h : x^2 - 7*x + 12 < 0) :
  42 < x^2 + 7*x + 12 ∧ x^2 + 7*x + 12 < 56 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_7x_plus_12_bounds_l844_84494


namespace NUMINAMATH_CALUDE_eggs_eaten_in_morning_l844_84454

theorem eggs_eaten_in_morning (initial_eggs : ℕ) (afternoon_eggs : ℕ) (remaining_eggs : ℕ) :
  initial_eggs = 20 →
  afternoon_eggs = 3 →
  remaining_eggs = 13 →
  initial_eggs - remaining_eggs - afternoon_eggs = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_eggs_eaten_in_morning_l844_84454


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l844_84450

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) ∧ 
  (b^3 - 2*b^2 + 3*b - 4 = 0) ∧ 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l844_84450


namespace NUMINAMATH_CALUDE_father_son_speed_ratio_l844_84438

/-- 
Given a hallway of length 16 meters where a father and son start walking from opposite ends 
at the same time and meet at a point 12 meters from the father's end, 
the ratio of the father's walking speed to the son's walking speed is 3:1.
-/
theorem father_son_speed_ratio 
  (hallway_length : ℝ) 
  (meeting_point : ℝ) 
  (father_speed : ℝ) 
  (son_speed : ℝ) 
  (h1 : hallway_length = 16)
  (h2 : meeting_point = 12)
  (h3 : father_speed > 0)
  (h4 : son_speed > 0)
  (h5 : meeting_point / father_speed = (hallway_length - meeting_point) / son_speed) :
  father_speed / son_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_father_son_speed_ratio_l844_84438


namespace NUMINAMATH_CALUDE_factorize_quadratic_minimum_value_quadratic_sum_abc_l844_84462

-- Problem 1
theorem factorize_quadratic (m : ℝ) : m^2 - 6*m + 5 = (m - 1)*(m - 5) := by sorry

-- Problem 2
theorem minimum_value_quadratic (a b : ℝ) :
  a^2 + b^2 - 4*a + 10*b + 33 ≥ 4 ∧
  (a^2 + b^2 - 4*a + 10*b + 33 = 4 ↔ a = 2 ∧ b = -5) := by sorry

-- Problem 3
theorem sum_abc (a b c : ℝ) (h1 : a - b = 8) (h2 : a*b + c^2 - 4*c + 20 = 0) :
  a + b + c = 2 := by sorry

end NUMINAMATH_CALUDE_factorize_quadratic_minimum_value_quadratic_sum_abc_l844_84462


namespace NUMINAMATH_CALUDE_product_1011_2_112_3_l844_84444

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.reverse.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a ternary number represented as a list of digits to its decimal equivalent -/
def ternary_to_decimal (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The main theorem stating that the product of 1011₂ and 112₃ in base 10 is 154 -/
theorem product_1011_2_112_3 : 
  (binary_to_decimal [true, true, false, true]) * 
  (ternary_to_decimal [2, 1, 1]) = 154 := by
  sorry

#eval binary_to_decimal [true, true, false, true]  -- Should output 11
#eval ternary_to_decimal [2, 1, 1]  -- Should output 14

end NUMINAMATH_CALUDE_product_1011_2_112_3_l844_84444


namespace NUMINAMATH_CALUDE_plane_equation_through_point_perpendicular_to_vector_l844_84431

/-- A plane passing through a point and perpendicular to a non-zero vector -/
theorem plane_equation_through_point_perpendicular_to_vector
  (x₀ y₀ z₀ : ℝ) (a b c : ℝ) (h : (a, b, c) ≠ (0, 0, 0)) :
  ∀ x y z : ℝ,
  (a * (x - x₀) + b * (y - y₀) + c * (z - z₀) = 0) ↔
  ((x, y, z) ∈ {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p = (x₀, y₀, z₀) + t • (a, b, c)}ᶜ) :=
by sorry


end NUMINAMATH_CALUDE_plane_equation_through_point_perpendicular_to_vector_l844_84431


namespace NUMINAMATH_CALUDE_inequality_solution_l844_84485

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 9 ↔ x < -3 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l844_84485


namespace NUMINAMATH_CALUDE_complex_number_properties_l844_84413

theorem complex_number_properties (z : ℂ) (h : (2 + Complex.I) * z = 1 + 3 * Complex.I) :
  (Complex.abs z = Real.sqrt 2) ∧ (z^2 - 2*z + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l844_84413


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_k_range_l844_84406

theorem quadratic_always_positive_implies_k_range (k : ℝ) :
  (∀ x : ℝ, x^2 + 2*k*x - (k - 2) > 0) → k ∈ Set.Ioo (-2 : ℝ) 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_k_range_l844_84406


namespace NUMINAMATH_CALUDE_condition_relationship_l844_84470

theorem condition_relationship : ¬(∀ x y : ℝ, (x > 1 ∧ y > 1) ↔ x + y > 3) :=
by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l844_84470


namespace NUMINAMATH_CALUDE_cake_muffin_mix_buyers_l844_84489

theorem cake_muffin_mix_buyers (total : ℕ) (cake : ℕ) (muffin : ℕ) (neither_prob : ℚ) : 
  total = 100 → cake = 50 → muffin = 40 → neither_prob = 29/100 → 
  ∃ both : ℕ, both = 19 ∧ 
    (total - (cake + muffin - both)) / total = neither_prob :=
by sorry

end NUMINAMATH_CALUDE_cake_muffin_mix_buyers_l844_84489


namespace NUMINAMATH_CALUDE_percentage_of_students_taking_music_l844_84430

/-- Calculates the percentage of students taking music in a school -/
theorem percentage_of_students_taking_music
  (total_students : ℕ)
  (dance_students : ℕ)
  (art_students : ℕ)
  (drama_students : ℕ)
  (h1 : total_students = 2000)
  (h2 : dance_students = 450)
  (h3 : art_students = 680)
  (h4 : drama_students = 370) :
  (total_students - (dance_students + art_students + drama_students)) / total_students * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_taking_music_l844_84430


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l844_84422

-- Define the binary number
def binary_number : List Bool := [true, false, true, true, true, false]

-- Define the octal number
def octal_number : Nat := 56

-- Theorem statement
theorem binary_to_octal_conversion :
  (binary_number.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0) = octal_number := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l844_84422


namespace NUMINAMATH_CALUDE_f_of_one_plus_g_of_two_l844_84465

def f (x : ℝ) : ℝ := 2 * x - 3

def g (x : ℝ) : ℝ := x + 1

theorem f_of_one_plus_g_of_two : f (1 + g 2) = 5 := by sorry

end NUMINAMATH_CALUDE_f_of_one_plus_g_of_two_l844_84465


namespace NUMINAMATH_CALUDE_four_digit_addition_l844_84491

theorem four_digit_addition (A B C D : ℕ) : 
  4000 * A + 500 * B + 100 * C + 20 * D + 7 = 8070 → C = 3 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_addition_l844_84491


namespace NUMINAMATH_CALUDE_flower_arrangement_count_l844_84476

def num_flowers : ℕ := 5
def num_vases : ℕ := 3

theorem flower_arrangement_count :
  (num_flowers * (num_flowers - 1) * (num_flowers - 2)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_flower_arrangement_count_l844_84476


namespace NUMINAMATH_CALUDE_pocket_money_calculation_l844_84484

def fifty_cent_coins : ℕ := 6
def twenty_cent_coins : ℕ := 6
def fifty_cent_value : ℚ := 0.5
def twenty_cent_value : ℚ := 0.2

theorem pocket_money_calculation :
  (fifty_cent_coins : ℚ) * fifty_cent_value + (twenty_cent_coins : ℚ) * twenty_cent_value = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_pocket_money_calculation_l844_84484


namespace NUMINAMATH_CALUDE_equal_quadratic_expressions_l844_84420

theorem equal_quadratic_expressions (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 6) :
  a * (a - 6) = b * (b - 6) ∧ a * (a - 6) = -9 := by
  sorry

end NUMINAMATH_CALUDE_equal_quadratic_expressions_l844_84420


namespace NUMINAMATH_CALUDE_cube_root_27_div_fourth_root_16_l844_84466

theorem cube_root_27_div_fourth_root_16 : (27 ^ (1/3)) / (16 ^ (1/4)) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_div_fourth_root_16_l844_84466


namespace NUMINAMATH_CALUDE_trajectory_of_P_l844_84442

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define the relationship between N, M, and P
def RelationNMP (xn yn xm ym xp yp : ℝ) : Prop :=
  C xn yn ∧ xm = 0 ∧ ym = yn ∧ xp = xn / 2 ∧ yp = yn

-- Theorem statement
theorem trajectory_of_P : 
  ∀ (x y : ℝ), (∃ (xn yn xm ym : ℝ), RelationNMP xn yn xm ym x y) → 
  x^2 / 2 + y^2 / 8 = 1 := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l844_84442


namespace NUMINAMATH_CALUDE_hospital_staff_count_l844_84423

theorem hospital_staff_count (total : ℕ) (doc_ratio nurse_ratio : ℕ) (nurse_count : ℕ) : 
  total = 280 → 
  doc_ratio = 5 →
  nurse_ratio = 9 →
  doc_ratio + nurse_ratio = (total / nurse_count) →
  nurse_count = 180 := by
sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l844_84423


namespace NUMINAMATH_CALUDE_quadratic_intercept_distance_l844_84436

/-- A quadratic function -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex (qf : QuadraticFunction) : ℝ := sorry

theorem quadratic_intercept_distance 
  (f g : QuadraticFunction)
  (h1 : ∀ x, g.f x = -f.f (120 - x))
  (h2 : ∃ v, g.f (vertex f) = v)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h3 : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
  (h4 : f.f x₁ = 0 ∨ g.f x₁ = 0)
  (h5 : f.f x₂ = 0 ∨ g.f x₂ = 0)
  (h6 : f.f x₃ = 0 ∨ g.f x₃ = 0)
  (h7 : f.f x₄ = 0 ∨ g.f x₄ = 0)
  (h8 : x₃ - x₂ = 120) :
  x₄ - x₁ = 360 + 240 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_intercept_distance_l844_84436


namespace NUMINAMATH_CALUDE_edwards_toy_purchase_l844_84403

/-- Proves that given an initial amount of $17.80, after purchasing 4 items at $0.95 each
    and one item at $6.00, the remaining amount is $8.00. -/
theorem edwards_toy_purchase (initial_amount : ℚ) (toy_car_price : ℚ) (race_track_price : ℚ)
    (num_toy_cars : ℕ) (h1 : initial_amount = 17.8)
    (h2 : toy_car_price = 0.95) (h3 : race_track_price = 6)
    (h4 : num_toy_cars = 4) : 
    initial_amount - (toy_car_price * num_toy_cars + race_track_price) = 8 := by
  sorry

end NUMINAMATH_CALUDE_edwards_toy_purchase_l844_84403


namespace NUMINAMATH_CALUDE_equation_solution_l844_84405

theorem equation_solution :
  ∀ x : ℝ, (Real.sqrt (5 * x^3 - 1) + Real.sqrt (x^3 - 1) = 4) ↔ 
  (x = Real.rpow 10 (1/3) ∨ x = Real.rpow 2 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l844_84405


namespace NUMINAMATH_CALUDE_bottle_cap_count_l844_84473

/-- Given the total cost of bottle caps and the cost per bottle cap,
    prove that the number of bottle caps is correct. -/
theorem bottle_cap_count 
  (total_cost : ℝ) 
  (cost_per_cap : ℝ) 
  (h1 : total_cost = 25) 
  (h2 : cost_per_cap = 5) : 
  total_cost / cost_per_cap = 5 := by
  sorry

#check bottle_cap_count

end NUMINAMATH_CALUDE_bottle_cap_count_l844_84473


namespace NUMINAMATH_CALUDE_chi_square_test_win_probability_not_C_given_not_win_l844_84429

-- Define the data from the problem
def flavor1_C : ℕ := 20
def flavor1_nonC : ℕ := 75
def flavor2_C : ℕ := 10
def flavor2_nonC : ℕ := 45
def total_samples : ℕ := 150

-- Define the chi-square test statistic function
def chi_square (a b c d n : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the proportions of card types
def prob_A : ℚ := 2 / 5
def prob_B : ℚ := 2 / 5
def prob_C : ℚ := 1 / 5

-- Theorem statements
theorem chi_square_test :
  chi_square flavor1_C flavor1_nonC flavor2_C flavor2_nonC total_samples < 6635 / 1000 :=
sorry

theorem win_probability :
  (3 * prob_A * prob_B * prob_C : ℚ) = 24 / 125 :=
sorry

theorem not_C_given_not_win :
  ((1 - prob_C)^3 : ℚ) / (1 - 3 * prob_A * prob_B * prob_C) = 64 / 101 :=
sorry

end NUMINAMATH_CALUDE_chi_square_test_win_probability_not_C_given_not_win_l844_84429


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l844_84443

theorem largest_prime_factor_of_12321 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 12321 → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l844_84443


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l844_84448

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation x^2 - 7x + 4 = 0 -/
def a : ℝ := 1
def b : ℝ := -7
def c : ℝ := 4

theorem quadratic_discriminant : discriminant a b c = 33 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l844_84448


namespace NUMINAMATH_CALUDE_tony_age_at_end_of_period_l844_84445

/-- Represents Tony's age at the start of the period -/
def initial_age : ℕ := 14

/-- Represents Tony's age at the end of the period -/
def final_age : ℕ := initial_age + 1

/-- Represents the number of days Tony worked at his initial age -/
def days_at_initial_age : ℕ := 46

/-- Represents the number of days Tony worked at his final age -/
def days_at_final_age : ℕ := 100 - days_at_initial_age

/-- Represents Tony's daily earnings at a given age -/
def daily_earnings (age : ℕ) : ℚ := 1.9 * age

/-- Represents Tony's total earnings during the period -/
def total_earnings : ℚ := 3750

theorem tony_age_at_end_of_period :
  final_age = 15 ∧
  days_at_initial_age + days_at_final_age = 100 ∧
  daily_earnings initial_age * days_at_initial_age +
  daily_earnings final_age * days_at_final_age = total_earnings :=
sorry

end NUMINAMATH_CALUDE_tony_age_at_end_of_period_l844_84445


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_quadratic_equation_distinct_roots_2_l844_84488

theorem quadratic_equation_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + (a + 1) * x - 2 = 0 ∧ a * y^2 + (a + 1) * y - 2 = 0) ↔
  (a < -5 - 2 * Real.sqrt 6 ∨ (-5 + 2 * Real.sqrt 6 < a ∧ a < 0) ∨ a > 0) :=
sorry

theorem quadratic_equation_distinct_roots_2 (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (1 - a) * x^2 + (a + 1) * x - 2 = 0 ∧ (1 - a) * y^2 + (a + 1) * y - 2 = 0) ↔
  (a < 1 ∨ (1 < a ∧ a < 3) ∨ a > 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_quadratic_equation_distinct_roots_2_l844_84488


namespace NUMINAMATH_CALUDE_average_problem_l844_84499

theorem average_problem (n₁ n₂ : ℕ) (avg_all avg₂ : ℚ) (h₁ : n₁ = 30) (h₂ : n₂ = 20) 
  (h₃ : avg₂ = 30) (h₄ : avg_all = 24) :
  let sum_all := (n₁ + n₂ : ℚ) * avg_all
  let sum₂ := n₂ * avg₂
  let sum₁ := sum_all - sum₂
  sum₁ / n₁ = 20 := by sorry

end NUMINAMATH_CALUDE_average_problem_l844_84499


namespace NUMINAMATH_CALUDE_difference_of_squares_l844_84455

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) :
  x^2 - y^2 = 160 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l844_84455


namespace NUMINAMATH_CALUDE_gumball_theorem_l844_84418

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  purple : Nat
  orange : Nat
  green : Nat
  yellow : Nat

/-- The minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFour (machine : GumballMachine) : Nat :=
  3 * 4 + 1

theorem gumball_theorem (machine : GumballMachine) 
  (h : machine = { purple := 12, orange := 6, green := 8, yellow := 5 }) : 
  minGumballsForFour machine = 13 := by
  sorry

#eval minGumballsForFour { purple := 12, orange := 6, green := 8, yellow := 5 }

end NUMINAMATH_CALUDE_gumball_theorem_l844_84418


namespace NUMINAMATH_CALUDE_product_closest_to_1200_l844_84490

def product : ℝ := 0.000315 * 3928500

def options : List ℝ := [1100, 1200, 1300, 1400]

theorem product_closest_to_1200 : 
  1200 ∈ options ∧ ∀ x ∈ options, |product - 1200| ≤ |product - x| :=
by sorry

end NUMINAMATH_CALUDE_product_closest_to_1200_l844_84490


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l844_84408

/-- The distance between the foci of the ellipse 9x^2 + 16y^2 = 144 is 2√7 -/
theorem ellipse_foci_distance :
  let a : ℝ := 4
  let b : ℝ := 3
  ∀ x y : ℝ, 9 * x^2 + 16 * y^2 = 144 →
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l844_84408


namespace NUMINAMATH_CALUDE_max_popsicles_for_10_dollars_l844_84452

/-- Represents the number of popsicles in a box -/
inductive BoxSize
  | Single : BoxSize
  | Three : BoxSize
  | Five : BoxSize
  | Seven : BoxSize

/-- Returns the cost of a box given its size -/
def boxCost (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 2
  | BoxSize.Five => 3
  | BoxSize.Seven => 4

/-- Returns the number of popsicles in a box given its size -/
def boxPopsicles (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 3
  | BoxSize.Five => 5
  | BoxSize.Seven => 7

/-- Represents a purchase of popsicle boxes -/
structure Purchase where
  single : ℕ
  three : ℕ
  five : ℕ
  seven : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.single * boxCost BoxSize.Single +
  p.three * boxCost BoxSize.Three +
  p.five * boxCost BoxSize.Five +
  p.seven * boxCost BoxSize.Seven

/-- Calculates the total number of popsicles in a purchase -/
def totalPopsicles (p : Purchase) : ℕ :=
  p.single * boxPopsicles BoxSize.Single +
  p.three * boxPopsicles BoxSize.Three +
  p.five * boxPopsicles BoxSize.Five +
  p.seven * boxPopsicles BoxSize.Seven

/-- Theorem: The maximum number of popsicles that can be bought with $10 is 17 -/
theorem max_popsicles_for_10_dollars :
  ∀ p : Purchase, totalCost p ≤ 10 → totalPopsicles p ≤ 17 ∧
  ∃ p : Purchase, totalCost p ≤ 10 ∧ totalPopsicles p = 17 :=
by sorry

end NUMINAMATH_CALUDE_max_popsicles_for_10_dollars_l844_84452


namespace NUMINAMATH_CALUDE_total_steel_parts_l844_84467

/-- Represents the number of machines of type A -/
def a : ℕ := sorry

/-- Represents the number of machines of type B -/
def b : ℕ := sorry

/-- The total number of machines -/
def total_machines : ℕ := 21

/-- The total number of chrome parts -/
def total_chrome_parts : ℕ := 66

/-- Steel parts in a type A machine -/
def steel_parts_A : ℕ := 3

/-- Chrome parts in a type A machine -/
def chrome_parts_A : ℕ := 2

/-- Steel parts in a type B machine -/
def steel_parts_B : ℕ := 2

/-- Chrome parts in a type B machine -/
def chrome_parts_B : ℕ := 4

theorem total_steel_parts :
  a + b = total_machines ∧
  chrome_parts_A * a + chrome_parts_B * b = total_chrome_parts →
  steel_parts_A * a + steel_parts_B * b = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_steel_parts_l844_84467


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l844_84477

theorem simplify_and_evaluate (a b : ℝ) : 
  a = 2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180) →
  b = 3 →
  1 - (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l844_84477


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l844_84493

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = (3/4) * x ∨ y = -(3/4) * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l844_84493


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l844_84427

/-- The surface area of a cylinder with height 2 and base circumference 2π is 6π -/
theorem cylinder_surface_area :
  ∀ (h : ℝ) (c : ℝ),
  h = 2 →
  c = 2 * Real.pi →
  2 * Real.pi * (c / (2 * Real.pi))^2 + c * h = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l844_84427


namespace NUMINAMATH_CALUDE_dance_team_initial_members_l844_84463

theorem dance_team_initial_members (initial_members quit_members new_members current_members : ℕ) 
  (h1 : quit_members = 8)
  (h2 : new_members = 13)
  (h3 : current_members = 30)
  (h4 : current_members = initial_members - quit_members + new_members) : 
  initial_members = 25 := by
  sorry

end NUMINAMATH_CALUDE_dance_team_initial_members_l844_84463


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l844_84464

theorem diophantine_equation_solution :
  ∃ (a b c d : ℕ+), 4^(a : ℕ) * 5^(b : ℕ) - 3^(c : ℕ) * 11^(d : ℕ) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l844_84464


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt_11_l844_84480

theorem closest_integer_to_sqrt_11 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 11| ≤ |m - Real.sqrt 11| ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt_11_l844_84480


namespace NUMINAMATH_CALUDE_find_y_l844_84459

theorem find_y (a b : ℝ) (y : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let s := (3 * a) ^ (2 * b)
  s = 5 * a^b * y^b →
  y = 9 * a / 5 := by
sorry

end NUMINAMATH_CALUDE_find_y_l844_84459


namespace NUMINAMATH_CALUDE_football_team_progress_l844_84460

/-- The progress of a football team after a series of gains and losses -/
theorem football_team_progress 
  (L1 G1 L2 G2 G3 : ℤ) 
  (hL1 : L1 = 17)
  (hG1 : G1 = 35)
  (hL2 : L2 = 22)
  (hG2 : G2 = 8) :
  (G1 + G2 - (L1 + L2)) + G3 = 4 + G3 :=
by sorry

end NUMINAMATH_CALUDE_football_team_progress_l844_84460


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l844_84400

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l844_84400


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_root_l844_84409

theorem cubic_polynomials_common_root :
  ∃ (c d : ℝ), c = -3 ∧ d = -4 ∧
  ∃ (x : ℝ), x^3 + c*x^2 + 15*x + 10 = 0 ∧ x^3 + d*x^2 + 17*x + 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_root_l844_84409


namespace NUMINAMATH_CALUDE_fraction_transformation_l844_84435

theorem fraction_transformation (a b c d x : ℤ) 
  (hb : b ≠ 0) 
  (hcd : c - d ≠ 0) 
  (h_simplest : ∀ k : ℤ, k ∣ c ∧ k ∣ d → k = 1 ∨ k = -1) 
  (h_eq : (2 * a + x) * d = (b - x) * c) : 
  x = (b * c - 2 * a * d) / (d + c) := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l844_84435


namespace NUMINAMATH_CALUDE_brothers_sisters_ratio_l844_84457

theorem brothers_sisters_ratio :
  ∀ (num_brothers : ℕ),
    (num_brothers + 2) * 2 = 12 →
    num_brothers / 2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_sisters_ratio_l844_84457


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l844_84421

/-- Given the cost of pencils and pens, calculate the cost of a specific combination -/
theorem pencil_pen_cost (pencil_cost pen_cost : ℝ) 
  (h1 : 4 * pencil_cost + pen_cost = 2.60)
  (h2 : pencil_cost + 3 * pen_cost = 2.15) :
  3 * pencil_cost + 2 * pen_cost = 2.63 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l844_84421


namespace NUMINAMATH_CALUDE_complex_absolute_value_l844_84432

theorem complex_absolute_value (z : ℂ) : z = 7 + 3*I → Complex.abs (z^2 + 8*z + 65) = Real.sqrt 30277 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l844_84432


namespace NUMINAMATH_CALUDE_square_of_negative_two_l844_84487

theorem square_of_negative_two : (-2)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_l844_84487


namespace NUMINAMATH_CALUDE_pages_copied_for_30_dollars_l844_84495

/-- Given a cost per page in cents and a budget in dollars, 
    calculate the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Theorem: With a cost of 3 cents per page and a budget of $30, 
    the maximum number of pages that can be copied is 1000. -/
theorem pages_copied_for_30_dollars : 
  max_pages_copied 3 30 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_30_dollars_l844_84495


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_open_2_closed_3_l844_84461

-- Define set A
def A : Set ℝ := {x | -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 5}

-- Define set B (domain of log(-x^2 + 6x - 8))
def B : Set ℝ := {x | -x^2 + 6*x - 8 > 0}

-- Theorem to prove
theorem A_intersect_B_equals_open_2_closed_3 : A ∩ B = {x | 2 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_open_2_closed_3_l844_84461


namespace NUMINAMATH_CALUDE_holds_age_ratio_l844_84439

/-- Proves that the ratio of Hold's age to her son's age today is 3:1 -/
theorem holds_age_ratio : 
  ∀ (hold_age_today hold_age_8_years_ago son_age_today son_age_8_years_ago : ℕ),
  hold_age_today = 36 →
  hold_age_8_years_ago = hold_age_today - 8 →
  son_age_8_years_ago = son_age_today - 8 →
  hold_age_8_years_ago = 7 * son_age_8_years_ago →
  (hold_age_today : ℚ) / son_age_today = 3 := by
sorry

end NUMINAMATH_CALUDE_holds_age_ratio_l844_84439


namespace NUMINAMATH_CALUDE_speed_to_arrive_on_time_l844_84453

/-- The speed required to arrive on time given late and early arrival conditions -/
theorem speed_to_arrive_on_time (d : ℝ) (t : ℝ) (h1 : d = 50 * (t + 1/12)) (h2 : d = 70 * (t - 1/12)) : 
  d / t = 58 := by
  sorry

end NUMINAMATH_CALUDE_speed_to_arrive_on_time_l844_84453


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l844_84471

theorem complex_absolute_value_product : 
  Complex.abs ((3 * Real.sqrt 5 - 5 * Complex.I) * (2 * Real.sqrt 2 + 4 * Complex.I)) = 12 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l844_84471


namespace NUMINAMATH_CALUDE_not_sufficient_and_necessary_condition_l844_84416

theorem not_sufficient_and_necessary_condition : ¬∀ (a b c : ℝ), (a * b > a * c ↔ b > c) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_and_necessary_condition_l844_84416
