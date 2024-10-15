import Mathlib

namespace NUMINAMATH_CALUDE_hourly_rate_approximation_l873_87356

/-- Calculates the hourly rate based on given salary information and work schedule. -/
def calculate_hourly_rate (base_salary : ℚ) (commission_rate : ℚ) (total_sales : ℚ) 
  (performance_bonus : ℚ) (deductions : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) 
  (weeks_per_month : ℕ) : ℚ :=
  let total_earnings := base_salary + (commission_rate * total_sales) + performance_bonus - deductions
  let total_hours := hours_per_day * days_per_week * weeks_per_month
  total_earnings / total_hours

/-- Proves that the hourly rate is approximately $3.86 given the specified conditions. -/
theorem hourly_rate_approximation :
  let base_salary : ℚ := 576
  let commission_rate : ℚ := 3 / 100
  let total_sales : ℚ := 4000
  let performance_bonus : ℚ := 75
  let deductions : ℚ := 30
  let hours_per_day : ℕ := 8
  let days_per_week : ℕ := 6
  let weeks_per_month : ℕ := 4
  let hourly_rate := calculate_hourly_rate base_salary commission_rate total_sales 
    performance_bonus deductions hours_per_day days_per_week weeks_per_month
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |hourly_rate - 386/100| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_hourly_rate_approximation_l873_87356


namespace NUMINAMATH_CALUDE_carmichael_function_properties_l873_87311

variable (a : ℕ)

theorem carmichael_function_properties (ha : a > 2) :
  (∃ n : ℕ, n > 1 ∧ ¬ Nat.Prime n ∧ a^n ≡ 1 [ZMOD n]) ∧
  (∀ p : ℕ, p > 1 → (∀ k : ℕ, 1 < k ∧ k < p → ¬(a^k ≡ 1 [ZMOD k])) → a^p ≡ 1 [ZMOD p] → Nat.Prime p) ∧
  ¬(∃ n : ℕ, n > 1 ∧ 2^n ≡ 1 [ZMOD n]) :=
by sorry

end NUMINAMATH_CALUDE_carmichael_function_properties_l873_87311


namespace NUMINAMATH_CALUDE_negation_of_proposition_l873_87382

theorem negation_of_proposition (m : ℝ) :
  (¬(m > 0 → ∃ x : ℝ, x^2 + x - m = 0)) ↔ (m ≤ 0 → ∀ x : ℝ, x^2 + x - m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l873_87382


namespace NUMINAMATH_CALUDE_correct_total_distance_l873_87377

/-- The total distance to fly from Germany to Russia and then return to Spain -/
def totalDistance (spainRussia : ℕ) (spainGermany : ℕ) : ℕ :=
  (spainRussia - spainGermany) + spainRussia

theorem correct_total_distance :
  totalDistance 7019 1615 = 12423 := by
  sorry

#eval totalDistance 7019 1615

end NUMINAMATH_CALUDE_correct_total_distance_l873_87377


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l873_87380

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∃ x, f x a < 2*a} = {a : ℝ | a > 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l873_87380


namespace NUMINAMATH_CALUDE_executive_committee_formation_l873_87318

theorem executive_committee_formation (total_members : ℕ) (experienced_members : ℕ) (committee_size : ℕ) : 
  total_members = 30 →
  experienced_members = 8 →
  committee_size = 5 →
  (Finset.sum (Finset.range (Nat.min committee_size experienced_members + 1))
    (λ k => Nat.choose experienced_members k * Nat.choose (total_members - experienced_members) (committee_size - k))) = 116172 := by
  sorry

end NUMINAMATH_CALUDE_executive_committee_formation_l873_87318


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l873_87390

/-- The number of ways to distribute books to people -/
def distribute_books (total_people : ℕ) (math_books : ℕ) (chinese_books : ℕ) : ℕ :=
  Nat.choose total_people chinese_books

/-- Theorem stating that the number of ways to distribute 6 math books and 3 Chinese books
    to 9 people is equal to C(9,3) -/
theorem book_distribution_theorem :
  distribute_books 9 6 3 = Nat.choose 9 3 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l873_87390


namespace NUMINAMATH_CALUDE_star_operation_result_l873_87335

def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

def star_operation (X Y : Set Nat) : Set Nat :=
  {x | x ∈ X ∧ x ∉ Y}

theorem star_operation_result :
  star_operation A B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_star_operation_result_l873_87335


namespace NUMINAMATH_CALUDE_common_solution_y_value_l873_87325

theorem common_solution_y_value (x y : ℝ) 
  (eq1 : x^2 + y^2 = 25) 
  (eq2 : x^2 + y = 10) : 
  y = (1 - Real.sqrt 61) / 2 := by
  sorry

end NUMINAMATH_CALUDE_common_solution_y_value_l873_87325


namespace NUMINAMATH_CALUDE_eggs_not_eaten_is_six_l873_87385

/-- Represents the number of eggs not eaten in a week given the following conditions:
  * Rhea buys 2 trays of eggs every week
  * Each tray has 24 eggs
  * Her son and daughter eat 2 eggs every morning
  * Rhea and her husband eat 4 eggs every night
  * There are 7 days in a week
-/
def eggs_not_eaten : ℕ :=
  let trays_per_week : ℕ := 2
  let eggs_per_tray : ℕ := 24
  let children_eggs_per_day : ℕ := 2
  let parents_eggs_per_day : ℕ := 4
  let days_per_week : ℕ := 7
  
  let total_eggs_bought := trays_per_week * eggs_per_tray
  let children_eggs_eaten := children_eggs_per_day * days_per_week
  let parents_eggs_eaten := parents_eggs_per_day * days_per_week
  let total_eggs_eaten := children_eggs_eaten + parents_eggs_eaten
  
  total_eggs_bought - total_eggs_eaten

theorem eggs_not_eaten_is_six : eggs_not_eaten = 6 := by
  sorry

end NUMINAMATH_CALUDE_eggs_not_eaten_is_six_l873_87385


namespace NUMINAMATH_CALUDE_f_value_at_2_l873_87364

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l873_87364


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_m_l873_87346

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, |m - 2| ≥ |f x|} = {m : ℝ | m ≥ 5 ∨ m ≤ -1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_m_l873_87346


namespace NUMINAMATH_CALUDE_goldbach_132_max_diff_l873_87358

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem goldbach_132_max_diff :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 132 ∧ p < q ∧
  ∀ (r s : ℕ), is_prime r → is_prime s → r + s = 132 → r < s →
  s - r ≤ q - p ∧
  q - p = 122 :=
sorry

end NUMINAMATH_CALUDE_goldbach_132_max_diff_l873_87358


namespace NUMINAMATH_CALUDE_function_value_at_two_l873_87399

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem function_value_at_two (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l873_87399


namespace NUMINAMATH_CALUDE_jessie_weight_before_jogging_l873_87363

def weight_before_jogging (weight_after_first_week weight_lost_first_week : ℕ) : ℕ :=
  weight_after_first_week + weight_lost_first_week

theorem jessie_weight_before_jogging 
  (weight_after_first_week : ℕ) 
  (weight_lost_first_week : ℕ) 
  (h1 : weight_after_first_week = 36)
  (h2 : weight_lost_first_week = 56) : 
  weight_before_jogging weight_after_first_week weight_lost_first_week = 92 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_before_jogging_l873_87363


namespace NUMINAMATH_CALUDE_gross_revenue_increase_l873_87326

theorem gross_revenue_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_reduction_percent : ℝ) 
  (quantity_increase_percent : ℝ) 
  (h1 : price_reduction_percent = 20) 
  (h2 : quantity_increase_percent = 50) :
  let new_price := original_price * (1 - price_reduction_percent / 100)
  let new_quantity := original_quantity * (1 + quantity_increase_percent / 100)
  let original_gross := original_price * original_quantity
  let new_gross := new_price * new_quantity
  (new_gross - original_gross) / original_gross * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_gross_revenue_increase_l873_87326


namespace NUMINAMATH_CALUDE_mary_nickels_l873_87312

theorem mary_nickels (initial_nickels : ℕ) (dad_gave : ℕ) (total_now : ℕ) : 
  dad_gave = 5 → total_now = 12 → initial_nickels + dad_gave = total_now → initial_nickels = 7 := by
sorry

end NUMINAMATH_CALUDE_mary_nickels_l873_87312


namespace NUMINAMATH_CALUDE_prob_red_ball_l873_87379

def urn1_red : ℚ := 3 / 8
def urn1_total : ℚ := 8
def urn2_red : ℚ := 1 / 2
def urn2_total : ℚ := 8
def urn3_red : ℚ := 0
def urn3_total : ℚ := 8

def prob_urn_selection : ℚ := 1 / 3

theorem prob_red_ball : 
  prob_urn_selection * (urn1_red * (urn1_total / urn1_total) + 
                        urn2_red * (urn2_total / urn2_total) + 
                        urn3_red * (urn3_total / urn3_total)) = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_ball_l873_87379


namespace NUMINAMATH_CALUDE_joes_fast_food_cost_l873_87362

theorem joes_fast_food_cost : 
  let sandwich_cost : ℕ := 4
  let soda_cost : ℕ := 3
  let num_sandwiches : ℕ := 3
  let num_sodas : ℕ := 5
  (num_sandwiches * sandwich_cost + num_sodas * soda_cost) = 27 := by
sorry

end NUMINAMATH_CALUDE_joes_fast_food_cost_l873_87362


namespace NUMINAMATH_CALUDE_martin_crayons_l873_87332

theorem martin_crayons (total_boxes : ℕ) (crayons_per_box : ℕ) (boxes_with_missing : ℕ) (missing_per_box : ℕ) :
  total_boxes = 8 →
  crayons_per_box = 7 →
  boxes_with_missing = 3 →
  missing_per_box = 2 →
  total_boxes * crayons_per_box - boxes_with_missing * missing_per_box = 50 :=
by sorry

end NUMINAMATH_CALUDE_martin_crayons_l873_87332


namespace NUMINAMATH_CALUDE_vampire_population_after_two_nights_l873_87301

def vampire_growth (initial_vampires : ℕ) (new_vampires_per_night : ℕ) (nights : ℕ) : ℕ :=
  initial_vampires * (new_vampires_per_night + 1)^nights

theorem vampire_population_after_two_nights :
  vampire_growth 3 7 2 = 192 :=
by sorry

end NUMINAMATH_CALUDE_vampire_population_after_two_nights_l873_87301


namespace NUMINAMATH_CALUDE_rational_function_positivity_l873_87381

theorem rational_function_positivity (x : ℝ) :
  (x^2 - 9) / (x^2 - 16) > 0 ↔ x < -4 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_positivity_l873_87381


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l873_87302

theorem quadratic_equation_root_zero (a : ℝ) :
  (∀ x, x^2 + x + a^2 - 1 = 0 → x = 0 ∨ x ≠ 0) →
  (∃ x, x^2 + x + a^2 - 1 = 0 ∧ x = 0) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l873_87302


namespace NUMINAMATH_CALUDE_lollipop_challenge_l873_87395

def joann_lollipops (n : ℕ) : ℕ := 8 + 2 * n

def tom_lollipops (n : ℕ) : ℕ := 5 * 2^(n - 1)

def total_lollipops : ℕ := 
  (Finset.range 7).sum joann_lollipops + (Finset.range 7).sum tom_lollipops

theorem lollipop_challenge : total_lollipops = 747 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_challenge_l873_87395


namespace NUMINAMATH_CALUDE_circle_equation_l873_87371

/-- A circle C with center (a,b) and radius 1 -/
structure Circle where
  a : ℝ
  b : ℝ
  radius : ℝ := 1

/-- The circle C is in the first quadrant -/
def in_first_quadrant (C : Circle) : Prop :=
  C.a > 0 ∧ C.b > 0

/-- The circle C is tangent to the line 4x-3y=0 -/
def tangent_to_line (C : Circle) : Prop :=
  abs (4 * C.a - 3 * C.b) / 5 = C.radius

/-- The circle C is tangent to the x-axis -/
def tangent_to_x_axis (C : Circle) : Prop :=
  C.b = C.radius

/-- The standard equation of the circle -/
def standard_equation (C : Circle) : Prop :=
  ∀ x y : ℝ, (x - C.a)^2 + (y - C.b)^2 = C.radius^2

theorem circle_equation (C : Circle) 
  (h1 : in_first_quadrant C)
  (h2 : tangent_to_line C)
  (h3 : tangent_to_x_axis C) :
  standard_equation { a := 2, b := 1, radius := 1 } :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l873_87371


namespace NUMINAMATH_CALUDE_quadratic_roots_existence_l873_87323

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation ax^2 + bx + c = 0 has real roots iff its discriminant is nonnegative -/
def has_real_roots (a b c : ℝ) : Prop := discriminant a b c ≥ 0

theorem quadratic_roots_existence :
  ¬(has_real_roots 1 1 1) ∧
  (has_real_roots 1 2 1) ∧
  (has_real_roots 1 (-2) (-1)) ∧
  (has_real_roots 1 (-1) (-2)) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_existence_l873_87323


namespace NUMINAMATH_CALUDE_ratio_B_to_C_l873_87375

def total_amount : ℕ := 578
def share_A : ℕ := 408
def share_B : ℕ := 102
def share_C : ℕ := 68

theorem ratio_B_to_C :
  (share_B : ℚ) / share_C = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_B_to_C_l873_87375


namespace NUMINAMATH_CALUDE_train_ride_nap_time_l873_87389

theorem train_ride_nap_time (total_time reading_time eating_time movie_time : ℕ) 
  (h1 : total_time = 9)
  (h2 : reading_time = 2)
  (h3 : eating_time = 1)
  (h4 : movie_time = 3) :
  total_time - (reading_time + eating_time + movie_time) = 3 :=
by sorry

end NUMINAMATH_CALUDE_train_ride_nap_time_l873_87389


namespace NUMINAMATH_CALUDE_quadratic_function_derivative_l873_87352

theorem quadratic_function_derivative (a c : ℝ) :
  (∀ x, deriv (fun x => a * x^2 + c) x = 2 * a * x) →
  deriv (fun x => a * x^2 + c) 1 = 2 →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_derivative_l873_87352


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l873_87339

/-- The quadratic polynomial q(x) that satisfies the given conditions -/
def q (x : ℚ) : ℚ := (17 * x^2 - 8 * x + 21) / 15

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q (-2) = 7 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l873_87339


namespace NUMINAMATH_CALUDE_inner_outer_hexagon_area_ratio_is_three_fourths_l873_87393

/-- A regular hexagon -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The ratio of the area of the inner hexagon to the area of the outer hexagon -/
def inner_outer_hexagon_area_ratio (h : RegularHexagon) : ℚ :=
  3 / 4

/-- The theorem stating that the ratio of the areas is 3/4 -/
theorem inner_outer_hexagon_area_ratio_is_three_fourths (h : RegularHexagon) :
  inner_outer_hexagon_area_ratio h = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inner_outer_hexagon_area_ratio_is_three_fourths_l873_87393


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l873_87343

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 ∧ 
  (∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 10 → a + b ≤ c + d) ∧
  a + b = 45 :=
sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l873_87343


namespace NUMINAMATH_CALUDE_troy_computer_worth_l873_87394

/-- The worth of Troy's new computer -/
def new_computer_worth (initial_savings selling_price additional_needed : ℕ) : ℕ :=
  initial_savings + selling_price + additional_needed

/-- Theorem: The worth of Troy's new computer is $80 -/
theorem troy_computer_worth :
  new_computer_worth 50 20 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_troy_computer_worth_l873_87394


namespace NUMINAMATH_CALUDE_infinite_non_representable_numbers_l873_87331

theorem infinite_non_representable_numbers : 
  ∃ S : Set ℕ, Set.Infinite S ∧ 
  ∀ k ∈ S, ∀ n : ℕ, ∀ p : ℕ, 
    Prime p → k ≠ n^2 + p := by
  sorry

end NUMINAMATH_CALUDE_infinite_non_representable_numbers_l873_87331


namespace NUMINAMATH_CALUDE_arthur_purchase_cost_l873_87308

/-- The cost of Arthur's purchases on two days -/
theorem arthur_purchase_cost
  (hamburger_cost : ℝ)
  (hot_dog_cost : ℝ)
  (day1_total : ℝ)
  (h_hot_dog_cost : hot_dog_cost = 1)
  (h_day1_equation : 3 * hamburger_cost + 4 * hot_dog_cost = day1_total)
  (h_day1_total : day1_total = 10) :
  2 * hamburger_cost + 3 * hot_dog_cost = 7 :=
by sorry

end NUMINAMATH_CALUDE_arthur_purchase_cost_l873_87308


namespace NUMINAMATH_CALUDE_turn_over_five_most_effective_l873_87383

-- Define the type for card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define a function to check if a character is a vowel
def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

-- Define a function to check if a number is even
def isEven (n : Nat) : Bool :=
  n % 2 = 0

-- Define Jane's claim as a function
def janesClaimHolds (card : Card) : Bool :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => ¬(isVowel c) ∨ isEven n
  | (CardSide.Number n, CardSide.Letter c) => ¬(isVowel c) ∨ isEven n
  | _ => true

-- Define the set of cards on the table
def cardsOnTable : List Card := [
  (CardSide.Letter 'A', CardSide.Number 0),  -- 0 is a placeholder
  (CardSide.Letter 'T', CardSide.Number 0),
  (CardSide.Letter 'U', CardSide.Number 0),
  (CardSide.Number 5, CardSide.Letter ' '),  -- ' ' is a placeholder
  (CardSide.Number 8, CardSide.Letter ' '),
  (CardSide.Number 10, CardSide.Letter ' '),
  (CardSide.Number 14, CardSide.Letter ' ')
]

-- Theorem: Turning over the card with 5 is the most effective way to potentially disprove Jane's claim
theorem turn_over_five_most_effective :
  ∃ (card : Card), card ∈ cardsOnTable ∧ 
  (∃ (c : Char), card = (CardSide.Number 5, CardSide.Letter c)) ∧
  (∀ (otherCard : Card), otherCard ∈ cardsOnTable → otherCard ≠ card →
    (∃ (possibleChar : Char), 
      ¬(janesClaimHolds (CardSide.Number 5, CardSide.Letter possibleChar)) →
      (janesClaimHolds otherCard ∨ 
       ∀ (possibleNum : Nat), janesClaimHolds (CardSide.Letter possibleChar, CardSide.Number possibleNum))))
  := by sorry


end NUMINAMATH_CALUDE_turn_over_five_most_effective_l873_87383


namespace NUMINAMATH_CALUDE_range_of_a_l873_87307

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 5, a * x^2 - x - 4 > 0) → a > 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l873_87307


namespace NUMINAMATH_CALUDE_initial_oranges_count_l873_87333

/-- The number of oranges initially in the bin -/
def initial_oranges : ℕ := sorry

/-- The number of oranges thrown away -/
def thrown_away : ℕ := 37

/-- The number of new oranges added -/
def new_oranges : ℕ := 7

/-- The final number of oranges in the bin -/
def final_oranges : ℕ := 10

/-- Theorem stating that the initial number of oranges was 40 -/
theorem initial_oranges_count : initial_oranges = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l873_87333


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l873_87368

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l873_87368


namespace NUMINAMATH_CALUDE_third_visit_next_month_l873_87350

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a schedule of pool visits -/
structure PoolSchedule :=
  (visit_days : List DayOfWeek)

/-- Represents a month's pool visits -/
structure MonthVisits :=
  (count : Nat)

/-- Function to calculate the date of the nth visit in the next month -/
def nextMonthVisitDate (schedule : PoolSchedule) (current_month : MonthVisits) (n : Nat) : Nat :=
  sorry

/-- Theorem statement -/
theorem third_visit_next_month 
  (schedule : PoolSchedule)
  (current_month : MonthVisits)
  (h1 : schedule.visit_days = [DayOfWeek.Wednesday, DayOfWeek.Friday])
  (h2 : current_month.count = 10) :
  nextMonthVisitDate schedule current_month 3 = 12 :=
sorry

end NUMINAMATH_CALUDE_third_visit_next_month_l873_87350


namespace NUMINAMATH_CALUDE_isabel_music_purchase_l873_87353

theorem isabel_music_purchase (country_albums : ℕ) (pop_albums : ℕ) (songs_per_album : ℕ) : 
  country_albums = 4 → pop_albums = 5 → songs_per_album = 8 → 
  (country_albums + pop_albums) * songs_per_album = 72 := by
sorry

end NUMINAMATH_CALUDE_isabel_music_purchase_l873_87353


namespace NUMINAMATH_CALUDE_unique_four_digit_int_l873_87319

/-- Represents a four-digit positive integer --/
structure FourDigitInt where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  positive : 1000 ≤ a * 1000 + b * 100 + c * 10 + d

/-- The conditions given in the problem --/
def satisfiesConditions (n : FourDigitInt) : Prop :=
  n.a + n.b + n.c + n.d = 17 ∧
  n.b + n.c = 9 ∧
  n.a - n.d = 2 ∧
  (n.a * 1000 + n.b * 100 + n.c * 10 + n.d) % 9 = 0

/-- The theorem to be proved --/
theorem unique_four_digit_int :
  ∃! (n : FourDigitInt), satisfiesConditions n ∧ n.a = 5 ∧ n.b = 4 ∧ n.c = 5 ∧ n.d = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_four_digit_int_l873_87319


namespace NUMINAMATH_CALUDE_linear_function_range_l873_87365

theorem linear_function_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (m + 2) * x₁ + (1 - m) > (m + 2) * x₂ + (1 - m)) →
  (∃ x : ℝ, x > 0 ∧ (m + 2) * x + (1 - m) = 0) →
  m < -2 := by
sorry

end NUMINAMATH_CALUDE_linear_function_range_l873_87365


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l873_87330

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def has_seven_consecutive_nonprimes (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ i : ℕ, i ≥ k ∧ i < k + 7 → ¬(is_prime i)

theorem smallest_prime_after_seven_nonprimes :
  (is_prime 97) ∧ 
  (has_seven_consecutive_nonprimes 90) ∧
  (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ has_seven_consecutive_nonprimes (p - 7))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l873_87330


namespace NUMINAMATH_CALUDE_cheaper_to_buy_more_count_l873_87373

def C (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 15 then 15 * n + 20
  else if 16 ≤ n ∧ n ≤ 30 then 13 * n
  else if 31 ≤ n ∧ n ≤ 45 then 11 * n + 50
  else 9 * n

theorem cheaper_to_buy_more_count :
  (∃ s : Finset ℕ, s.card = 4 ∧ ∀ n ∈ s, C (n + 1) < C n) ∧
  ¬(∃ s : Finset ℕ, s.card > 4 ∧ ∀ n ∈ s, C (n + 1) < C n) :=
by sorry

end NUMINAMATH_CALUDE_cheaper_to_buy_more_count_l873_87373


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l873_87305

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l873_87305


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l873_87336

theorem chicken_rabbit_problem (x y : ℕ) : 
  (x + y = 35 ∧ 2 * x + 4 * y = 94) ↔ 
  (x + y = 35 ∧ x * 2 + y * 4 = 94) := by sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l873_87336


namespace NUMINAMATH_CALUDE_expression_value_l873_87384

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 1) :
  3 * x - 2 * y + 4 * z = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l873_87384


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l873_87349

/-- 
Given a man's speed against the current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 12.4)
  (h2 : current_speed = 4.3) : 
  speed_against_current + 2 * current_speed = 21 := by
sorry

end NUMINAMATH_CALUDE_mans_speed_with_current_l873_87349


namespace NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_four_l873_87334

def numbers : List Nat := [4628, 4638, 4648, 4658, 4662]

theorem product_of_digits_not_divisible_by_four : 
  ∃ n ∈ numbers, 
    ¬(n % 4 = 0) ∧ 
    ((n % 100) % 10 * ((n % 100) / 10 % 10) = 24) := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_four_l873_87334


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_range_of_a_l873_87327

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

-- Part 1
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B 1 = {x : ℝ | -1 < x ∧ x < 1} := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  B a ∩ A = B a → a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_range_of_a_l873_87327


namespace NUMINAMATH_CALUDE_sequence_equality_l873_87344

-- Define the sequence S
def S (n : ℕ) : ℕ := (4 * n - 3)^2

-- Define the proposed form of S
def S_proposed (n : ℕ) (a b : ℤ) : ℤ := (4 * n - 3) * (a * n + b)

-- Theorem statement
theorem sequence_equality (a b : ℤ) :
  (∀ n : ℕ, n > 0 → S n = S_proposed n a b) →
  a^2 + b^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_sequence_equality_l873_87344


namespace NUMINAMATH_CALUDE_harriets_age_l873_87387

theorem harriets_age (peter_age harriet_age : ℕ) : 
  (peter_age + 4 = 2 * (harriet_age + 4)) →  -- Condition 1
  (peter_age = 60 / 2) →                     -- Conditions 2 and 3 combined
  harriet_age = 13 := by
sorry

end NUMINAMATH_CALUDE_harriets_age_l873_87387


namespace NUMINAMATH_CALUDE_minor_premise_identification_l873_87316

-- Define the type for functions
def Function := Type → Type

-- Define properties
def IsTrigonometric (f : Function) : Prop := sorry
def IsPeriodic (f : Function) : Prop := sorry

-- Define tan function
def tan : Function := sorry

-- Theorem statement
theorem minor_premise_identification :
  (∀ f : Function, IsTrigonometric f → IsPeriodic f) →  -- major premise
  (IsTrigonometric tan) →                               -- minor premise
  (IsPeriodic tan) →                                    -- conclusion
  (IsTrigonometric tan)                                 -- proves minor premise
  := by sorry

end NUMINAMATH_CALUDE_minor_premise_identification_l873_87316


namespace NUMINAMATH_CALUDE_compound_carbon_count_l873_87300

/-- Represents the number of atoms of a given element in a compound -/
structure AtomCount where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound given its atom count and atomic weights -/
def molecularWeight (count : AtomCount) (weights : AtomicWeights) : ℝ :=
  count.carbon * weights.carbon +
  count.hydrogen * weights.hydrogen +
  count.oxygen * weights.oxygen

/-- The theorem to be proved -/
theorem compound_carbon_count (weights : AtomicWeights)
    (h_carbon : weights.carbon = 12.01)
    (h_hydrogen : weights.hydrogen = 1.01)
    (h_oxygen : weights.oxygen = 16.00) :
    ∃ (count : AtomCount),
      count.hydrogen = 8 ∧
      count.oxygen = 2 ∧
      molecularWeight count weights = 88 ∧
      count.carbon = 4 := by
  sorry

end NUMINAMATH_CALUDE_compound_carbon_count_l873_87300


namespace NUMINAMATH_CALUDE_tangency_triangle_area_for_given_radii_l873_87374

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents three mutually externally tangent circles -/
structure TangentCircles where
  c1 : Circle
  c2 : Circle
  c3 : Circle

/-- The area of the triangle formed by the points of tangency of three mutually externally tangent circles -/
def tangencyTriangleArea (tc : TangentCircles) : ℝ := sorry

/-- Theorem stating that for the given radii, the area of the tangency triangle is 120/25 -/
theorem tangency_triangle_area_for_given_radii :
  ∀ tc : TangentCircles,
    tc.c1.radius = 5 ∧ tc.c2.radius = 12 ∧ tc.c3.radius = 13 →
    tangencyTriangleArea tc = 120 / 25 := by
  sorry

end NUMINAMATH_CALUDE_tangency_triangle_area_for_given_radii_l873_87374


namespace NUMINAMATH_CALUDE_plum_problem_l873_87392

theorem plum_problem (x : ℕ) : 
  (4 * x / 5 : ℚ) = (5 * x / 6 : ℚ) - 1 → 2 * x = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_plum_problem_l873_87392


namespace NUMINAMATH_CALUDE_proposition_is_true_l873_87357

theorem proposition_is_true : ∀ (x y : ℝ), x + 2*y ≠ 5 → x ≠ 1 ∨ y ≠ 2 := by sorry

end NUMINAMATH_CALUDE_proposition_is_true_l873_87357


namespace NUMINAMATH_CALUDE_balanced_leaving_probability_formula_l873_87320

/-- The probability that 3n students leaving from 3 rows of n students, one at a time
    with all leaving orders equally likely, such that there are never two rows where
    the number of students remaining differs by 2 or more. -/
def balanced_leaving_probability (n : ℕ) : ℚ :=
  (6 * n * (n.factorial ^ 3 : ℚ)) / ((3 * n).factorial : ℚ)

/-- Theorem stating that the probability of balanced leaving for 3n students
    in 3 rows of n is equal to (6n * (n!)^3) / (3n)! -/
theorem balanced_leaving_probability_formula (n : ℕ) (h : n ≥ 1) :
  balanced_leaving_probability n = (6 * n * (n.factorial ^ 3 : ℚ)) / ((3 * n).factorial : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_balanced_leaving_probability_formula_l873_87320


namespace NUMINAMATH_CALUDE_f_81_product_remainder_l873_87370

def p : ℕ := 2^16 + 1

-- S is implicitly defined as the set of positive integers not divisible by p

def is_in_S (x : ℕ) : Prop := x > 0 ∧ ¬(p ∣ x)

axiom p_is_prime : Nat.Prime p

axiom f_exists : ∃ (f : ℕ → ℕ), 
  (∀ x, is_in_S x → f x < p) ∧
  (∀ x y, is_in_S x → is_in_S y → (f x * f y) % p = (f (x * y) + f (x * y^(p-2))) % p) ∧
  (∀ x, is_in_S x → f (x + p) = f x)

def N : ℕ := sorry  -- Definition of N as the product of nonzero f(81) values

theorem f_81_product_remainder : N % p = 16384 := by sorry

end NUMINAMATH_CALUDE_f_81_product_remainder_l873_87370


namespace NUMINAMATH_CALUDE_solution_in_interval_l873_87369

def f (x : ℝ) := x^2 + 12*x - 15

theorem solution_in_interval :
  ∃ x : ℝ, x ∈ (Set.Ioo 1.1 1.2) ∧ f x = 0 :=
by
  have h1 : f 1.1 < 0 := by sorry
  have h2 : f 1.2 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l873_87369


namespace NUMINAMATH_CALUDE_complex_on_ray_unit_circle_l873_87388

theorem complex_on_ray_unit_circle (z : ℂ) (a b : ℝ) :
  z = a + b * I →
  a = b →
  a ≥ 0 →
  Complex.abs z = 1 →
  z = Complex.mk (Real.sqrt 2 / 2) (Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_on_ray_unit_circle_l873_87388


namespace NUMINAMATH_CALUDE_clock_problem_l873_87397

/-- Represents a clock with special striking properties -/
structure StrikingClock where
  /-- Time for each stroke and interval between strokes (in seconds) -/
  stroke_time : ℝ
  /-- Calculates the total time lapse for striking a given hour -/
  time_lapse : ℕ → ℝ
  /-- The time lapse is (2n - 1) * stroke_time, where n is the hour -/
  time_lapse_eq : ∀ (hour : ℕ), time_lapse hour = (2 * hour - 1) * stroke_time

/-- The theorem representing our clock problem -/
theorem clock_problem (clock : StrikingClock) 
    (h1 : clock.time_lapse 7 = 26) 
    (h2 : ∃ (hour : ℕ), clock.time_lapse hour = 22) : 
  ∃ (hour : ℕ), hour = 6 ∧ clock.time_lapse hour = 22 :=
sorry

end NUMINAMATH_CALUDE_clock_problem_l873_87397


namespace NUMINAMATH_CALUDE_line_equal_intercepts_l873_87378

/-- A line with equation ax + y - 2 - a = 0 has equal intercepts on the x-axis and y-axis if and only if a = -2 or a = 1 -/
theorem line_equal_intercepts (a : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ 
    (a * k = k - 2 + a) ∧ 
    (k = k - 2 + a)) ↔ 
  (a = -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_line_equal_intercepts_l873_87378


namespace NUMINAMATH_CALUDE_stella_annual_income_after_tax_l873_87337

/-- Calculates Stella's annual income after tax deduction --/
theorem stella_annual_income_after_tax :
  let base_salary : ℕ := 3500
  let bonuses : List ℕ := [1200, 600, 1500, 900, 1200]
  let paid_months : ℕ := 10
  let tax_rate : ℚ := 1 / 20

  let total_base_salary := base_salary * paid_months
  let total_bonuses := bonuses.sum
  let total_income := total_base_salary + total_bonuses
  let tax_deduction := (total_income : ℚ) * tax_rate
  let annual_income_after_tax := (total_income : ℚ) - tax_deduction

  annual_income_after_tax = 38380 := by
  sorry

end NUMINAMATH_CALUDE_stella_annual_income_after_tax_l873_87337


namespace NUMINAMATH_CALUDE_triangle_problem_l873_87306

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = π ∧
  Real.cos (B - C) - 2 * Real.sin B * Real.sin C = -1/2 →
  A = π/3 ∧
  (a = 5 ∧ b = 4 →
    a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
    (1/2) * b * c * Real.sin A = 2*Real.sqrt 3 + Real.sqrt 39) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l873_87306


namespace NUMINAMATH_CALUDE_unique_divisible_by_44_l873_87328

/-- Represents a six-digit number in the form 5n7264 where n is a single digit -/
def sixDigitNumber (n : Nat) : Nat :=
  500000 + 10000 * n + 7264

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : Nat) : Prop :=
  ∃ k, a = b * k

/-- Theorem stating that 517264 is the only number in the form 5n7264 
    (where n is a single digit) that is divisible by 44 -/
theorem unique_divisible_by_44 : 
  ∀ n : Nat, n < 10 → 
    (isDivisibleBy (sixDigitNumber n) 44 ↔ n = 1) := by
  sorry

#check unique_divisible_by_44

end NUMINAMATH_CALUDE_unique_divisible_by_44_l873_87328


namespace NUMINAMATH_CALUDE_pat_stickers_l873_87347

theorem pat_stickers (initial_stickers given_away_stickers : ℝ) 
  (h1 : initial_stickers = 39.0)
  (h2 : given_away_stickers = 22.0) :
  initial_stickers - given_away_stickers = 17.0 := by
  sorry

end NUMINAMATH_CALUDE_pat_stickers_l873_87347


namespace NUMINAMATH_CALUDE_right_triangle_cos_z_l873_87398

theorem right_triangle_cos_z (X Y Z : ℝ) : 
  -- Triangle XYZ is right-angled at X
  X + Y + Z = π →
  X = π / 2 →
  -- sin Y = 3/5
  Real.sin Y = 3 / 5 →
  -- Prove: cos Z = 3/5
  Real.cos Z = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_cos_z_l873_87398


namespace NUMINAMATH_CALUDE_fixed_point_quadratic_l873_87341

theorem fixed_point_quadratic (p : ℝ) : 
  9 * (5 : ℝ)^2 + p * 5 - 5 * p = 225 := by sorry

end NUMINAMATH_CALUDE_fixed_point_quadratic_l873_87341


namespace NUMINAMATH_CALUDE_max_value_expression_l873_87355

theorem max_value_expression :
  ∃ (x y : ℝ),
    ∀ (a b : ℝ),
      (Real.sqrt (9 - Real.sqrt 7) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) *
      (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y - Real.cos (2 * y)) ≤
      (Real.sqrt (9 - Real.sqrt 7) * Real.sin a - Real.sqrt (2 * (1 + Real.cos (2 * a))) - 1) *
      (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos b - Real.cos (2 * b)) →
      (Real.sqrt (9 - Real.sqrt 7) * Real.sin a - Real.sqrt (2 * (1 + Real.cos (2 * a))) - 1) *
      (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos b - Real.cos (2 * b)) = 24 - 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l873_87355


namespace NUMINAMATH_CALUDE_cost_equation_solution_l873_87322

/-- Given the cost equations for products A and B, prove that the solution (16, 4) satisfies both equations. -/
theorem cost_equation_solution :
  let x : ℚ := 16
  let y : ℚ := 4
  (20 * x + 15 * y = 380) ∧ (15 * x + 10 * y = 280) := by
  sorry

end NUMINAMATH_CALUDE_cost_equation_solution_l873_87322


namespace NUMINAMATH_CALUDE_fibonacci_sum_theorem_l873_87354

/-- Definition of the Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of the Fibonacci series divided by powers of 10 -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (10 : ℝ) ^ n

/-- Theorem stating that the sum of Fₙ/10ⁿ from n=0 to infinity equals 10/89 -/
theorem fibonacci_sum_theorem : fibSum = 10 / 89 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_sum_theorem_l873_87354


namespace NUMINAMATH_CALUDE_first_hour_distance_car_distance_problem_l873_87366

/-- Given a car with increasing speed, calculate the distance traveled in the first hour -/
theorem first_hour_distance (speed_increase : ℕ → ℕ) (total_distance : ℕ) : ℕ :=
  let first_hour_dist : ℕ := 55
  have speed_increase_def : ∀ n : ℕ, speed_increase n = 2 * n := by sorry
  have total_distance_def : total_distance = 792 := by sorry
  have sum_formula : total_distance = (12 : ℕ) * first_hour_dist + 11 * 12 := by sorry
  first_hour_dist

/-- The main theorem stating the distance traveled in the first hour -/
theorem car_distance_problem : first_hour_distance (λ n => 2 * n) 792 = 55 := by sorry

end NUMINAMATH_CALUDE_first_hour_distance_car_distance_problem_l873_87366


namespace NUMINAMATH_CALUDE_ratio_fraction_l873_87338

theorem ratio_fraction (x y : ℚ) (h : x / y = 4 / 5) : (x + y) / (x - y) = -9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_l873_87338


namespace NUMINAMATH_CALUDE_candy_store_sales_l873_87321

-- Define the quantities and prices
def fudge_pounds : ℕ := 20
def fudge_price : ℚ := 2.5
def truffle_dozens : ℕ := 5
def truffle_price : ℚ := 1.5
def pretzel_dozens : ℕ := 3
def pretzel_price : ℚ := 2

-- Define the calculation for total sales
def total_sales : ℚ :=
  fudge_pounds * fudge_price +
  truffle_dozens * 12 * truffle_price +
  pretzel_dozens * 12 * pretzel_price

-- Theorem statement
theorem candy_store_sales : total_sales = 212 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_sales_l873_87321


namespace NUMINAMATH_CALUDE_height_difference_l873_87376

/-- The height difference between the tallest and shortest players on a basketball team. -/
theorem height_difference (tallest_height shortest_height : ℝ) 
  (h_tallest : tallest_height = 77.75)
  (h_shortest : shortest_height = 68.25) : 
  tallest_height - shortest_height = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l873_87376


namespace NUMINAMATH_CALUDE_not_obtainable_2013201420152016_l873_87317

/-- Represents the state of the board -/
structure Board :=
  (left : ℕ)
  (right : ℕ)

/-- Represents a single operation on the board -/
def operate (b : Board) : Board :=
  { left := b.left * b.right,
    right := b.left^3 + b.right^3 }

/-- Checks if a number is obtainable on the board -/
def is_obtainable (n : ℕ) : Prop :=
  ∃ (b : Board), ∃ (k : ℕ), 
    (Nat.iterate operate k { left := 21, right := 8 }).left = n ∨
    (Nat.iterate operate k { left := 21, right := 8 }).right = n

/-- The main theorem stating that 2013201420152016 is not obtainable -/
theorem not_obtainable_2013201420152016 : 
  ¬ is_obtainable 2013201420152016 := by
  sorry


end NUMINAMATH_CALUDE_not_obtainable_2013201420152016_l873_87317


namespace NUMINAMATH_CALUDE_right_triangle_sides_l873_87340

theorem right_triangle_sides (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_ratio : ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k) 
  (h_area : a * b / 2 = 24) :
  a = 6 ∧ b = 8 ∧ c = 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l873_87340


namespace NUMINAMATH_CALUDE_cubic_solution_sum_l873_87348

theorem cubic_solution_sum (k : ℝ) (a b c : ℝ) : 
  (a^3 - 6*a^2 + 8*a + k = 0) →
  (b^3 - 6*b^2 + 8*b + k = 0) →
  (c^3 - 6*c^2 + 8*c + k = 0) →
  (k ≠ 0) →
  (a*b/c + b*c/a + c*a/b = 64/k - 12) := by sorry

end NUMINAMATH_CALUDE_cubic_solution_sum_l873_87348


namespace NUMINAMATH_CALUDE_sum_equals_221_2357_l873_87324

theorem sum_equals_221_2357 : 217 + 2.017 + 0.217 + 2.0017 = 221.2357 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_221_2357_l873_87324


namespace NUMINAMATH_CALUDE_expected_value_is_four_thirds_l873_87329

/-- The expected value of a biased coin flip --/
def expected_value_biased_coin : ℚ :=
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let value_heads : ℤ := 5
  let value_tails : ℤ := -6
  p_heads * value_heads + p_tails * value_tails

/-- Theorem: The expected value of the biased coin flip is 4/3 --/
theorem expected_value_is_four_thirds :
  expected_value_biased_coin = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_four_thirds_l873_87329


namespace NUMINAMATH_CALUDE_set_operations_l873_87303

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 4}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 4}) ∧
  (A ∪ B = {x | x > 1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l873_87303


namespace NUMINAMATH_CALUDE_carrots_grown_total_l873_87367

/-- The number of carrots grown by Joan -/
def joans_carrots : ℕ := 29

/-- The number of carrots grown by Jessica -/
def jessicas_carrots : ℕ := 11

/-- The total number of carrots grown by Joan and Jessica -/
def total_carrots : ℕ := joans_carrots + jessicas_carrots

theorem carrots_grown_total : total_carrots = 40 := by
  sorry

end NUMINAMATH_CALUDE_carrots_grown_total_l873_87367


namespace NUMINAMATH_CALUDE_waiter_tables_l873_87391

theorem waiter_tables (initial_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) : 
  initial_customers = 44 → left_customers = 12 → people_per_table = 8 → 
  (initial_customers - left_customers) / people_per_table = 4 := by
sorry

end NUMINAMATH_CALUDE_waiter_tables_l873_87391


namespace NUMINAMATH_CALUDE_ellipse_properties_l873_87351

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection of a line with the ellipse
def line_ellipse_intersection (k : ℝ) (a b : ℝ) (x : ℝ) : Prop :=
  (3 + 4*k^2) * x^2 - 8*k^2 * x + (4*k^2 - 12) = 0

-- Main theorem
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ (x y : ℝ), ellipse a b x y ∧ parabola x y) →
  (∃ (x1 y1 x2 y2 : ℝ), ellipse a b x1 y1 ∧ ellipse a b x2 y2 ∧ 
    parabola x1 y1 ∧ parabola x2 y2 ∧ 
    ((x2 - x1)^2 + (y2 - y1)^2)^(1/2 : ℝ) = 3) →
  (a = 2 ∧ b = (3 : ℝ)^(1/2 : ℝ)) ∧
  (∀ (k : ℝ), k ≠ 0 →
    (∃ (x1 x2 x3 x4 : ℝ), 
      line_ellipse_intersection k a b x1 ∧
      line_ellipse_intersection k a b x2 ∧
      line_ellipse_intersection (-1/k) a b x3 ∧
      line_ellipse_intersection (-1/k) a b x4 ∧
      (∃ (r : ℝ), 
        (x1 - 1)^2 + (k*(x1 - 1))^2 = r^2 ∧
        (x2 - 1)^2 + (k*(x2 - 1))^2 = r^2 ∧
        (x3 - 1)^2 + (-1/k*(x3 - 1))^2 = r^2 ∧
        (x4 - 1)^2 + (-1/k*(x4 - 1))^2 = r^2)) ↔
    (k = 1 ∨ k = -1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l873_87351


namespace NUMINAMATH_CALUDE_tina_crayon_selection_ways_l873_87386

/-- The number of different-colored crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of crayons Tina must select -/
def selected_crayons : ℕ := 6

/-- The number of mandatory crayons (red and blue) -/
def mandatory_crayons : ℕ := 2

/-- Computes the number of ways to select k items from n items -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The main theorem stating the number of ways Tina can select the crayons -/
theorem tina_crayon_selection_ways :
  combinations (total_crayons - mandatory_crayons) (selected_crayons - mandatory_crayons) = 715 := by
  sorry

end NUMINAMATH_CALUDE_tina_crayon_selection_ways_l873_87386


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l873_87313

theorem consecutive_integers_problem (x y z : ℤ) :
  (x = y + 1) →
  (y = z + 1) →
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 8) →
  z = 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l873_87313


namespace NUMINAMATH_CALUDE_prob_only_one_AB_qualifies_prob_at_least_one_qualifies_l873_87361

-- Define the probabilities for each student passing each round
def prob_written_A : ℚ := 2/3
def prob_written_B : ℚ := 1/2
def prob_written_C : ℚ := 3/4
def prob_interview_A : ℚ := 1/2
def prob_interview_B : ℚ := 2/3
def prob_interview_C : ℚ := 1/3

-- Define the probability of each student qualifying for the finals
def prob_qualify_A : ℚ := prob_written_A * prob_interview_A
def prob_qualify_B : ℚ := prob_written_B * prob_interview_B
def prob_qualify_C : ℚ := prob_written_C * prob_interview_C

-- Theorem for the first question
theorem prob_only_one_AB_qualifies :
  (prob_qualify_A * (1 - prob_qualify_B) + (1 - prob_qualify_A) * prob_qualify_B) = 4/9 := by
  sorry

-- Theorem for the second question
theorem prob_at_least_one_qualifies :
  (1 - (1 - prob_qualify_A) * (1 - prob_qualify_B) * (1 - prob_qualify_C)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_only_one_AB_qualifies_prob_at_least_one_qualifies_l873_87361


namespace NUMINAMATH_CALUDE_borrowed_amount_l873_87359

theorem borrowed_amount (X : ℝ) : 
  (X + 0.1 * X = 110) → X = 100 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amount_l873_87359


namespace NUMINAMATH_CALUDE_total_gumballs_l873_87342

def gumball_problem (total gumballs_todd gumballs_alisha gumballs_bobby remaining : ℕ) : Prop :=
  gumballs_todd = 4 ∧
  gumballs_alisha = 2 * gumballs_todd ∧
  gumballs_bobby = 4 * gumballs_alisha - 5 ∧
  total = gumballs_todd + gumballs_alisha + gumballs_bobby + remaining ∧
  remaining = 6

theorem total_gumballs : ∃ total : ℕ, gumball_problem total 4 8 27 6 ∧ total = 45 :=
  sorry

end NUMINAMATH_CALUDE_total_gumballs_l873_87342


namespace NUMINAMATH_CALUDE_gcd_840_1785_f_2_equals_62_l873_87304

-- Define the polynomial f(x) = 2x⁴ + 3x³ + 5x - 4
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Theorem for the GCD of 840 and 1785
theorem gcd_840_1785 : Nat.gcd 840 1785 = 105 := by sorry

-- Theorem for the value of f(2)
theorem f_2_equals_62 : f 2 = 62 := by sorry

end NUMINAMATH_CALUDE_gcd_840_1785_f_2_equals_62_l873_87304


namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l873_87360

theorem bumper_car_line_problem (initial_people : ℕ) (people_left : ℕ) (total_people : ℕ) : 
  initial_people = 9 →
  people_left = 6 →
  total_people = 18 →
  total_people - (initial_people - people_left) = 15 := by
sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l873_87360


namespace NUMINAMATH_CALUDE_arithmetic_square_root_property_l873_87314

theorem arithmetic_square_root_property (π : ℝ) : 
  Real.sqrt ((π - 4)^2) = 4 - π := by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_property_l873_87314


namespace NUMINAMATH_CALUDE_two_points_theorem_l873_87315

/-- Represents the three possible states of a point in the bun -/
inductive PointState
  | Type1
  | Type2
  | NoRaisin

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The bun as a bounded 3D space -/
def Bun : Set Point3D :=
  sorry

/-- Function that determines the state of a point in the bun -/
def pointState : Point3D → PointState :=
  sorry

/-- Distance between two points in 3D space -/
def distance (p q : Point3D) : ℝ :=
  sorry

theorem two_points_theorem :
  ∃ (p q : Point3D), p ∈ Bun ∧ q ∈ Bun ∧ distance p q = 1 ∧
    (pointState p = pointState q ∨ (pointState p = PointState.NoRaisin ∧ pointState q = PointState.NoRaisin)) :=
  sorry

end NUMINAMATH_CALUDE_two_points_theorem_l873_87315


namespace NUMINAMATH_CALUDE_find_A_l873_87309

theorem find_A : ∃ A : ℤ, A + 19 = 47 ∧ A = 28 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l873_87309


namespace NUMINAMATH_CALUDE_john_car_profit_l873_87345

/-- The money John made from fixing and racing a car -/
theorem john_car_profit (original_cost repair_discount prize_money prize_keep_percent : ℚ)
  (h1 : original_cost = 20000)
  (h2 : repair_discount = 20)
  (h3 : prize_money = 70000)
  (h4 : prize_keep_percent = 90) :
  let discounted_cost := original_cost * (1 - repair_discount / 100)
  let kept_prize := prize_money * (prize_keep_percent / 100)
  kept_prize - discounted_cost = 47000 :=
by sorry

end NUMINAMATH_CALUDE_john_car_profit_l873_87345


namespace NUMINAMATH_CALUDE_percentage_greater_l873_87310

theorem percentage_greater (A B : ℝ) (y : ℝ) (h1 : A > B) (h2 : B > 0) : 
  let C := A + B
  y = 100 * ((C - B) / B) → y = 100 * (A / B) := by
sorry

end NUMINAMATH_CALUDE_percentage_greater_l873_87310


namespace NUMINAMATH_CALUDE_no_equal_coin_exchange_l873_87372

theorem no_equal_coin_exchange : ¬ ∃ (n : ℕ), n > 0 ∧ n * (1 + 2 + 3 + 5) = 500 := by
  sorry

end NUMINAMATH_CALUDE_no_equal_coin_exchange_l873_87372


namespace NUMINAMATH_CALUDE_factorize_2x_squared_minus_18_l873_87396

theorem factorize_2x_squared_minus_18 (x : ℝ) :
  2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by sorry

end NUMINAMATH_CALUDE_factorize_2x_squared_minus_18_l873_87396
