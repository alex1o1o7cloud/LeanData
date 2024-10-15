import Mathlib

namespace NUMINAMATH_CALUDE_sabrina_basil_leaves_l1188_118893

/-- The number of basil leaves Sabrina needs -/
def basil : ℕ := 12

/-- The number of sage leaves Sabrina needs -/
def sage : ℕ := 6

/-- The number of verbena leaves Sabrina needs -/
def verbena : ℕ := 11

/-- Theorem stating the correct number of basil leaves Sabrina needs -/
theorem sabrina_basil_leaves :
  (basil = 2 * sage) ∧
  (sage = verbena - 5) ∧
  (basil + sage + verbena = 29) ∧
  (basil = 12) := by
  sorry

#check sabrina_basil_leaves

end NUMINAMATH_CALUDE_sabrina_basil_leaves_l1188_118893


namespace NUMINAMATH_CALUDE_fred_initial_cards_l1188_118804

theorem fred_initial_cards (cards_bought cards_left : ℕ) : 
  cards_bought = 3 → cards_left = 2 → cards_bought + cards_left = 5 :=
by sorry

end NUMINAMATH_CALUDE_fred_initial_cards_l1188_118804


namespace NUMINAMATH_CALUDE_clock_hands_straight_in_day_l1188_118860

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents when the clock hands are straight -/
inductive ClockHandsStraight
  | coinciding
  | opposite

/-- Represents the position of the minute hand when the clock hands are straight -/
inductive MinuteHandPosition
  | zero_minutes
  | thirty_minutes

/-- The number of times the clock hands are straight in a day -/
def straight_hands_count : ℕ := 44

/-- Theorem stating that the clock hands are straight 44 times in a day -/
theorem clock_hands_straight_in_day :
  straight_hands_count = 44 :=
by sorry

end NUMINAMATH_CALUDE_clock_hands_straight_in_day_l1188_118860


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1188_118828

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧ (∃ x : ℝ, x ≤ 1 ∧ x^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1188_118828


namespace NUMINAMATH_CALUDE_first_number_value_l1188_118827

-- Define the custom operation
def custom_op (m n : ℤ) : ℤ := n^2 - m

-- Theorem statement
theorem first_number_value :
  ∃ x : ℤ, custom_op x 3 = 6 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l1188_118827


namespace NUMINAMATH_CALUDE_arithmetic_sequence_probability_l1188_118882

/-- The set of numbers from which we select -/
def NumberSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

/-- A function to check if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop := a + c = 2 * b

/-- The total number of ways to choose 3 numbers from the set -/
def totalSelections : ℕ := Nat.choose 20 3

/-- The number of ways to choose 3 numbers that form an arithmetic sequence -/
def arithmeticSequenceSelections : ℕ := 90

/-- The probability of selecting 3 numbers that form an arithmetic sequence -/
def probability : ℚ := arithmeticSequenceSelections / totalSelections

theorem arithmetic_sequence_probability :
  probability = 3 / 38 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_probability_l1188_118882


namespace NUMINAMATH_CALUDE_mixed_fraction_product_l1188_118870

theorem mixed_fraction_product (X Y : ℤ) : 
  (5 + 1 / X : ℚ) * (Y + 1 / 2 : ℚ) = 43 →
  5 < (5 + 1 / X : ℚ) →
  (5 + 1 / X : ℚ) ≤ 5.5 →
  X = 17 ∧ Y = 8 := by
sorry

end NUMINAMATH_CALUDE_mixed_fraction_product_l1188_118870


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1188_118858

/-- The standard equation of a hyperbola sharing a focus with the parabola x² = 8y and having eccentricity 2 -/
theorem hyperbola_standard_equation :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ x₀ y₀ : ℝ, x₀^2 = 8*y₀ ∧ (x₀, y₀) = (0, 2)) →
  (a = 1 ∧ b^2 = 3) →
  ∀ x y : ℝ, y^2 - x^2 / 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1188_118858


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l1188_118820

/-- A circle inscribed in quadrilateral EFGH -/
structure InscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The circle is tangent to EF at R -/
  tangent_EF : True
  /-- The circle is tangent to GH at S -/
  tangent_GH : True
  /-- The circle is tangent to EH at T -/
  tangent_EH : True
  /-- ER = 25 -/
  ER : r = 25
  /-- RF = 35 -/
  RF : r = 35
  /-- GS = 40 -/
  GS : r = 40
  /-- SH = 20 -/
  SH : r = 20
  /-- ET = 45 -/
  ET : r = 45

/-- The square of the radius of the inscribed circle is 3600 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle) : c.r^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l1188_118820


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1188_118874

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1188_118874


namespace NUMINAMATH_CALUDE_height_difference_l1188_118863

/-- Proves that the difference between Ron's height and Dean's height is 8 feet -/
theorem height_difference (water_depth : ℝ) (ron_height : ℝ) (dean_height : ℝ)
  (h1 : water_depth = 2 * dean_height)
  (h2 : ron_height = 14)
  (h3 : water_depth = 12) :
  ron_height - dean_height = 8 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l1188_118863


namespace NUMINAMATH_CALUDE_M_divisible_by_52_l1188_118846

/-- The number formed by concatenating integers from 1 to 51 -/
def M : ℕ :=
  -- We don't actually compute M, just define it conceptually
  sorry

/-- M is divisible by 52 -/
theorem M_divisible_by_52 : 52 ∣ M := by
  sorry

end NUMINAMATH_CALUDE_M_divisible_by_52_l1188_118846


namespace NUMINAMATH_CALUDE_restaurant_group_size_restaurant_group_size_proof_l1188_118841

theorem restaurant_group_size (adult_meal_cost : ℕ) (kids_in_group : ℕ) (total_cost : ℕ) : ℕ :=
  let adults_in_group := total_cost / adult_meal_cost
  let total_people := adults_in_group + kids_in_group
  total_people

#check restaurant_group_size 8 2 72 = 11

theorem restaurant_group_size_proof 
  (adult_meal_cost : ℕ) 
  (kids_in_group : ℕ) 
  (total_cost : ℕ) 
  (h1 : adult_meal_cost = 8)
  (h2 : kids_in_group = 2)
  (h3 : total_cost = 72) :
  restaurant_group_size adult_meal_cost kids_in_group total_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_size_restaurant_group_size_proof_l1188_118841


namespace NUMINAMATH_CALUDE_base_eight_1563_to_ten_l1188_118879

def base_eight_to_ten (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem base_eight_1563_to_ten :
  base_eight_to_ten 1563 = 883 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_1563_to_ten_l1188_118879


namespace NUMINAMATH_CALUDE_fraction_equality_l1188_118850

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1188_118850


namespace NUMINAMATH_CALUDE_extra_bananas_distribution_l1188_118813

theorem extra_bananas_distribution (total_children absent_children : ℕ) 
  (original_distribution : ℕ) (h1 : total_children = 610) 
  (h2 : absent_children = 305) (h3 : original_distribution = 2) : 
  (total_children * original_distribution) / (total_children - absent_children) - 
   original_distribution = 2 := by
  sorry

end NUMINAMATH_CALUDE_extra_bananas_distribution_l1188_118813


namespace NUMINAMATH_CALUDE_min_electricity_price_l1188_118875

theorem min_electricity_price (a : ℝ) (h_a : a > 0) :
  let f (x : ℝ) := (a + 0.2 * a / (x - 0.4)) * (x - 0.3)
  ∃ x_min : ℝ, x_min = 0.6 ∧
    (∀ x : ℝ, 0.55 ≤ x ∧ x ≤ 0.75 ∧ f x ≥ 0.6 * a → x ≥ x_min) :=
by sorry

end NUMINAMATH_CALUDE_min_electricity_price_l1188_118875


namespace NUMINAMATH_CALUDE_solve_for_y_l1188_118800

theorem solve_for_y (x y : ℝ) (h1 : x + 2 * y = 20) (h2 : x = 10) : y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1188_118800


namespace NUMINAMATH_CALUDE_brand_d_highest_sales_l1188_118832

/-- Represents the sales volume of a brand -/
structure BrandSales where
  name : String
  sales : ℕ

/-- Theorem: Brand D has the highest sales volume -/
theorem brand_d_highest_sales (total : ℕ) (a b c d : BrandSales) :
  total = 100 ∧
  a.name = "A" ∧ a.sales = 15 ∧
  b.name = "B" ∧ b.sales = 30 ∧
  c.name = "C" ∧ c.sales = 12 ∧
  d.name = "D" ∧ d.sales = 43 →
  d.sales ≥ a.sales ∧ d.sales ≥ b.sales ∧ d.sales ≥ c.sales :=
by sorry

end NUMINAMATH_CALUDE_brand_d_highest_sales_l1188_118832


namespace NUMINAMATH_CALUDE_equation_solution_l1188_118852

theorem equation_solution :
  ∃ x : ℝ, (64 : ℝ) ^ (3 * x + 1) = (16 : ℝ) ^ (4 * x - 5) ∧ x = -13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1188_118852


namespace NUMINAMATH_CALUDE_orange_box_ratio_l1188_118814

theorem orange_box_ratio (total : ℕ) (given_to_mother : ℕ) (remaining : ℕ) :
  total = 9 →
  given_to_mother = 1 →
  remaining = 4 →
  (total - given_to_mother - remaining) * 2 = total - given_to_mother :=
by sorry

end NUMINAMATH_CALUDE_orange_box_ratio_l1188_118814


namespace NUMINAMATH_CALUDE_fish_population_estimate_l1188_118834

theorem fish_population_estimate 
  (initially_marked : ℕ) 
  (second_catch : ℕ) 
  (marked_in_second : ℕ) 
  (h1 : initially_marked = 30)
  (h2 : second_catch = 50)
  (h3 : marked_in_second = 2) :
  (initially_marked * second_catch) / marked_in_second = 750 :=
by
  sorry

#check fish_population_estimate

end NUMINAMATH_CALUDE_fish_population_estimate_l1188_118834


namespace NUMINAMATH_CALUDE_exists_x0_f_less_than_g_l1188_118877

noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin x ^ 2017

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2017 + 2017 ^ x

theorem exists_x0_f_less_than_g :
  ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > x₀, f x < g x := by sorry

end NUMINAMATH_CALUDE_exists_x0_f_less_than_g_l1188_118877


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1188_118815

theorem smallest_constant_inequality (D : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2 ≥ D * (x - y)) ↔ D ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1188_118815


namespace NUMINAMATH_CALUDE_apple_cost_l1188_118816

/-- The cost of an item given the amount paid and change received -/
def itemCost (amountPaid changeReceived : ℚ) : ℚ :=
  amountPaid - changeReceived

/-- Proof that the apple costs $0.75 -/
theorem apple_cost (amountPaid changeReceived : ℚ) 
  (h1 : amountPaid = 5)
  (h2 : changeReceived = 4.25) : 
  itemCost amountPaid changeReceived = 0.75 := by
  sorry

#check apple_cost

end NUMINAMATH_CALUDE_apple_cost_l1188_118816


namespace NUMINAMATH_CALUDE_projection_relations_l1188_118845

-- Define a plane
structure Plane where
  -- Add necessary fields

-- Define a line
structure Line where
  -- Add necessary fields

-- Define the projection of a line onto a plane
def project (l : Line) (p : Plane) : Line :=
  sorry

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  sorry

-- Define perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

-- Define coincident lines
def coincident (l1 l2 : Line) : Prop :=
  sorry

theorem projection_relations (α : Plane) (m n : Line) :
  let m1 := project m α
  let n1 := project n α
  -- All four propositions are false
  (¬ (parallel m1 n1 → parallel m n)) ∧
  (¬ (parallel m n → (parallel m1 n1 ∨ coincident m1 n1))) ∧
  (¬ (perpendicular m1 n1 → perpendicular m n)) ∧
  (¬ (perpendicular m n → perpendicular m1 n1)) :=
by
  sorry

end NUMINAMATH_CALUDE_projection_relations_l1188_118845


namespace NUMINAMATH_CALUDE_inequalities_proof_l1188_118891

theorem inequalities_proof (k : ℕ) (x : Fin k → ℝ) 
  (h_pos : ∀ i, x i > 0) (h_diff : ∀ i j, i ≠ j → x i ≠ x j) : 
  (Real.sqrt ((Finset.univ.sum (λ i => (x i)^2)) / k) > 
   (Finset.univ.sum (λ i => x i)) / k) ∧
  ((Finset.univ.sum (λ i => x i)) / k > 
   k / (Finset.univ.sum (λ i => 1 / (x i)))) := by
  sorry


end NUMINAMATH_CALUDE_inequalities_proof_l1188_118891


namespace NUMINAMATH_CALUDE_line_segment_ratio_l1188_118872

/-- Given points P, Q, R, and S on a straight line in that order,
    with PQ = 3, QR = 7, and PS = 20, prove that PR:QS = 1 -/
theorem line_segment_ratio (P Q R S : ℝ) 
  (h_order : P < Q ∧ Q < R ∧ R < S)
  (h_PQ : Q - P = 3)
  (h_QR : R - Q = 7)
  (h_PS : S - P = 20) :
  (R - P) / (S - Q) = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l1188_118872


namespace NUMINAMATH_CALUDE_prime_power_sum_l1188_118819

theorem prime_power_sum (p q : Nat) (m n : Nat) : 
  Nat.Prime p → Nat.Prime q → p < q →
  (∃ c : Nat, (p^(m+1) - 1) / (p - 1) = q^c) →
  (∃ d : Nat, (q^(n+1) - 1) / (q - 1) = p^d) →
  (p = 2 ∧ ∃ t : Nat, Nat.Prime t ∧ q = 2^t - 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1188_118819


namespace NUMINAMATH_CALUDE_relationship_abc_l1188_118873

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.rpow 0.8 0.8)
  (hb : b = Real.rpow 0.8 1.2)
  (hc : c = Real.rpow 1.2 0.8) : 
  c > a ∧ a > b :=
sorry

end NUMINAMATH_CALUDE_relationship_abc_l1188_118873


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l1188_118833

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l1188_118833


namespace NUMINAMATH_CALUDE_cost_of_750_apples_l1188_118855

/-- The cost of buying a given number of apples, given the price and quantity of a bag of apples -/
def cost_of_apples (apples_per_bag : ℕ) (price_per_bag : ℕ) (total_apples : ℕ) : ℕ :=
  (total_apples / apples_per_bag) * price_per_bag

/-- Theorem: The cost of 750 apples is $120, given that a bag of 50 apples costs $8 -/
theorem cost_of_750_apples :
  cost_of_apples 50 8 750 = 120 := by
  sorry

#eval cost_of_apples 50 8 750

end NUMINAMATH_CALUDE_cost_of_750_apples_l1188_118855


namespace NUMINAMATH_CALUDE_max_value_xy_difference_l1188_118889

theorem max_value_xy_difference (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  x^2 * y - y^2 * x ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xy_difference_l1188_118889


namespace NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l1188_118824

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Predicate for a number being even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem units_digit_of_p_plus_two (p : ℕ) 
  (h1 : p > 0)
  (h2 : isEven p)
  (h3 : unitsDigit p > 0)
  (h4 : unitsDigit (p^3) - unitsDigit (p^2) = 0) :
  unitsDigit (p + 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l1188_118824


namespace NUMINAMATH_CALUDE_relay_game_error_l1188_118886

def initial_equation (x : ℝ) : Prop :=
  3 / (x - 1) = 1 - x / (x + 1)

def step1 (x : ℝ) : Prop :=
  3 * (x + 1) = (x + 1) * (x - 1) - x * (x - 1)

def step2 (x : ℝ) : Prop :=
  3 * x + 3 = x^2 + 1 - x^2 + x

def step3 (x : ℝ) : Prop :=
  3 * x - x = 1 - 3

def step4 (x : ℝ) : Prop :=
  x = -1

theorem relay_game_error :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 →
    (initial_equation x ↔ step1 x) ∧
    ¬(initial_equation x ↔ step2 x) ∧
    (initial_equation x ↔ step3 x) ∧
    (initial_equation x ↔ step4 x) :=
by sorry

end NUMINAMATH_CALUDE_relay_game_error_l1188_118886


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l1188_118888

/-- The system of linear equations representing two lines -/
def line1 (x y : ℚ) : Prop := 12 * x - 5 * y = 40
def line2 (x y : ℚ) : Prop := 8 * x + 2 * y = 20

/-- The intersection point of the two lines -/
def intersection_point : ℚ × ℚ := (45/16, -5/4)

/-- Theorem stating that the intersection point satisfies both equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
sorry

/-- Theorem stating that the intersection point is unique -/
theorem intersection_point_unique (x y : ℚ) :
  line1 x y → line2 x y → (x, y) = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l1188_118888


namespace NUMINAMATH_CALUDE_largest_m_is_138_l1188_118853

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_valid_pair (x y : ℕ) : Prop :=
  x < 15 ∧ y < 15 ∧ x ≠ y ∧ is_prime x ∧ is_prime y ∧ is_prime (x + y) ∧ is_prime (10 * x + y)

def m (x y : ℕ) : ℕ := x * y * (10 * x + y)

theorem largest_m_is_138 :
  ∀ x y : ℕ, is_valid_pair x y → m x y ≤ 138 :=
sorry

end NUMINAMATH_CALUDE_largest_m_is_138_l1188_118853


namespace NUMINAMATH_CALUDE_john_travel_money_l1188_118806

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after buying a ticket -/
def remainingMoney (savings : ℕ) (ticketCost : ℕ) : ℕ :=
  base8ToBase10 savings - ticketCost

theorem john_travel_money :
  remainingMoney 5555 1200 = 1725 := by sorry

end NUMINAMATH_CALUDE_john_travel_money_l1188_118806


namespace NUMINAMATH_CALUDE_fly_revolutions_at_midnight_l1188_118898

/-- Represents a clock hand --/
inductive ClockHand
| Second
| Minute
| Hour

/-- Represents the state of the fly on the clock --/
structure FlyState where
  currentHand : ClockHand
  revolutions : ℕ

/-- The number of revolutions each hand makes in 12 hours --/
def handRevolutions (hand : ClockHand) : ℕ :=
  match hand with
  | ClockHand.Second => 720
  | ClockHand.Minute => 12
  | ClockHand.Hour => 1

/-- The total number of revolutions made by all hands in 12 hours --/
def totalRevolutions : ℕ :=
  (handRevolutions ClockHand.Second) +
  (handRevolutions ClockHand.Minute) +
  (handRevolutions ClockHand.Hour)

/-- Theorem stating that the fly makes 245 revolutions by midnight --/
theorem fly_revolutions_at_midnight :
  ∃ (finalState : FlyState),
    finalState.currentHand = ClockHand.Second →
    (∀ t, t ∈ Set.Icc (0 : ℝ) 12 →
      ¬ (∃ (h1 h2 h3 : ClockHand), h1 ≠ h2 ∧ h2 ≠ h3 ∧ h1 ≠ h3 ∧
        handRevolutions h1 * t = handRevolutions h2 * t ∧
        handRevolutions h2 * t = handRevolutions h3 * t)) →
    finalState.revolutions = 245 :=
sorry

end NUMINAMATH_CALUDE_fly_revolutions_at_midnight_l1188_118898


namespace NUMINAMATH_CALUDE_quadratic_root_discriminant_square_relation_l1188_118829

theorem quadratic_root_discriminant_square_relation 
  (a b c t : ℝ) (h1 : a ≠ 0) (h2 : a * t^2 + b * t + c = 0) :
  b^2 - 4*a*c = (2*a*t + b)^2 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_discriminant_square_relation_l1188_118829


namespace NUMINAMATH_CALUDE_largest_valid_number_nine_zero_nine_nine_is_valid_nine_zero_nine_nine_is_largest_l1188_118835

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n / 10) % 10 = (n / 1000) % 10 + (n / 100) % 10 ∧
  n % 10 = (n / 100) % 10 + (n / 10) % 10

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 9099 :=
by sorry

theorem nine_zero_nine_nine_is_valid :
  is_valid_number 9099 :=
by sorry

theorem nine_zero_nine_nine_is_largest :
  ∀ n : ℕ, is_valid_number n → n = 9099 ∨ n < 9099 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_nine_zero_nine_nine_is_valid_nine_zero_nine_nine_is_largest_l1188_118835


namespace NUMINAMATH_CALUDE_cruise_liner_travelers_l1188_118866

theorem cruise_liner_travelers :
  ∃ a : ℕ,
    250 ≤ a ∧ a ≤ 400 ∧
    a % 15 = 8 ∧
    a % 25 = 17 ∧
    (a = 292 ∨ a = 367) :=
by sorry

end NUMINAMATH_CALUDE_cruise_liner_travelers_l1188_118866


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l1188_118862

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- Define the open interval (2, 3)
def openInterval : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l1188_118862


namespace NUMINAMATH_CALUDE_difference_of_squares_l1188_118808

theorem difference_of_squares (x : ℝ) : x^2 - 25 = (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1188_118808


namespace NUMINAMATH_CALUDE_father_daughter_age_sum_l1188_118890

theorem father_daughter_age_sum :
  ∀ (father_age daughter_age : ℕ),
    father_age - daughter_age = 22 →
    daughter_age = 16 →
    father_age + daughter_age = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_father_daughter_age_sum_l1188_118890


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1188_118838

theorem quadratic_inequality_solution (a b : ℚ) : 
  (∀ x, ax^2 - (a+1)*x + b < 0 ↔ 1 < x ∧ x < 5) → a + b = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1188_118838


namespace NUMINAMATH_CALUDE_triangle_properties_l1188_118823

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- Define the altitude line
def altitude_line (x y : ℝ) : Prop := 2 * x - 3 * y + 14 = 0

-- Define the equidistant lines
def equidistant_line1 (x y : ℝ) : Prop := 7 * x - 6 * y + 4 = 0
def equidistant_line2 (x y : ℝ) : Prop := 3 * x + 2 * y - 44 = 0

-- Theorem statement
theorem triangle_properties :
  -- 1. The altitude from A to BC
  (∀ x y : ℝ, altitude_line x y ↔ 
    (x - A.1) * (B.1 - C.1) + (y - A.2) * (B.2 - C.2) = 0 ∧ 
    ∃ t : ℝ, x = A.1 + t * (B.2 - C.2) ∧ y = A.2 - t * (B.1 - C.1)) ∧
  -- 2. The lines through B equidistant from A and C
  (∀ x y : ℝ, (equidistant_line1 x y ∨ equidistant_line2 x y) ↔
    abs ((y - A.2) * (B.1 - A.1) - (x - A.1) * (B.2 - A.2)) = 
    abs ((y - C.2) * (B.1 - C.1) - (x - C.1) * (B.2 - C.2))) :=
sorry


end NUMINAMATH_CALUDE_triangle_properties_l1188_118823


namespace NUMINAMATH_CALUDE_age_group_problem_l1188_118896

theorem age_group_problem (n : ℕ) (A : ℝ) : 
  (n + 1) * (A + 7) = n * A + 39 →
  (n + 1) * (A - 1) = n * A + 15 →
  n = 3 := by sorry

end NUMINAMATH_CALUDE_age_group_problem_l1188_118896


namespace NUMINAMATH_CALUDE_six_matchsticks_remain_l1188_118876

/-- The number of matchsticks remaining in a box after Elvis and Ralph make squares -/
def remaining_matchsticks (total : ℕ) (elvis_squares : ℕ) (ralph_squares : ℕ) : ℕ :=
  total - (4 * elvis_squares + 8 * ralph_squares)

/-- Theorem stating that 6 matchsticks remain when Elvis makes 5 squares and Ralph makes 3 squares from a box of 50 matchsticks -/
theorem six_matchsticks_remain : remaining_matchsticks 50 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_matchsticks_remain_l1188_118876


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1188_118807

/-- An arithmetic sequence is a sequence where the difference between 
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1188_118807


namespace NUMINAMATH_CALUDE_anne_age_when_paul_is_38_l1188_118881

/-- Given the initial ages of Paul and Anne in 2015, this theorem proves
    Anne's age when Paul is 38 years old. -/
theorem anne_age_when_paul_is_38 (paul_age_2015 anne_age_2015 : ℕ) 
    (h1 : paul_age_2015 = 11) 
    (h2 : anne_age_2015 = 14) : 
    anne_age_2015 + (38 - paul_age_2015) = 41 := by
  sorry

#check anne_age_when_paul_is_38

end NUMINAMATH_CALUDE_anne_age_when_paul_is_38_l1188_118881


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l1188_118880

def smallest_five_digit_number_divisible_by_smallest_primes : ℕ := 11550

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def is_divisible_by_smallest_primes (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0

theorem smallest_five_digit_divisible_by_smallest_primes :
  (is_five_digit smallest_five_digit_number_divisible_by_smallest_primes) ∧
  (is_divisible_by_smallest_primes smallest_five_digit_number_divisible_by_smallest_primes) ∧
  (∀ m : ℕ, m < smallest_five_digit_number_divisible_by_smallest_primes →
    ¬(is_five_digit m ∧ is_divisible_by_smallest_primes m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l1188_118880


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1188_118865

theorem exam_maximum_marks (ashley_marks : ℕ) (ashley_percentage : ℚ) :
  ashley_marks = 332 →
  ashley_percentage = 83 / 100 →
  (ashley_marks : ℚ) / ashley_percentage = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1188_118865


namespace NUMINAMATH_CALUDE_degree_of_polynomial_l1188_118892

/-- The degree of the polynomial (5x^3 + 7)^10 is 30 -/
theorem degree_of_polynomial (x : ℝ) : Polynomial.degree ((5 * X ^ 3 + 7 : Polynomial ℝ) ^ 10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_l1188_118892


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l1188_118844

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l1188_118844


namespace NUMINAMATH_CALUDE_parabola_vertex_l1188_118869

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 3 * (x - 2)^2 - 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, -5)

/-- Theorem: The vertex of the parabola y = 3(x-2)^2 - 5 is (2, -5) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1188_118869


namespace NUMINAMATH_CALUDE_calculation_proof_l1188_118861

theorem calculation_proof : (-0.75) / 3 * (-2/5) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1188_118861


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1188_118897

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1188_118897


namespace NUMINAMATH_CALUDE_vector_representation_l1188_118805

-- Define points A, B, and Q in a 2D plane
variable (A B Q : ℝ × ℝ)

-- Define the ratio condition
def ratio_condition (A B Q : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ Q.1 - A.1 = 7*k ∧ Q.1 - B.1 = 2*k ∧
              Q.2 - A.2 = 7*k ∧ Q.2 - B.2 = 2*k

-- Define vector addition and scalar multiplication
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def scalar_mul (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Theorem statement
theorem vector_representation (A B Q : ℝ × ℝ) 
  (h : ratio_condition A B Q) :
  Q = vec_add (scalar_mul (-2/5) A) B :=
sorry

end NUMINAMATH_CALUDE_vector_representation_l1188_118805


namespace NUMINAMATH_CALUDE_candy_division_problem_l1188_118864

theorem candy_division_problem :
  ∃! x : ℕ, 120 ≤ x ∧ x ≤ 150 ∧ x % 5 = 2 ∧ x % 6 = 5 ∧ x = 137 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_problem_l1188_118864


namespace NUMINAMATH_CALUDE_inequality_proof_l1188_118868

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 2 * (x * y + y * z + z * x) = x * y * z) :
  (1 / ((x - 2) * (y - 2) * (z - 2))) + (8 / ((x + 2) * (y + 2) * (z + 2))) ≤ 1 / 32 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1188_118868


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1188_118867

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17)
  (h2 : a * b + c + d = 85)
  (h3 : a * d + b * c = 180)
  (h4 : c * d = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 934 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1188_118867


namespace NUMINAMATH_CALUDE_second_term_is_seven_general_formula_l1188_118849

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  monotone : Monotone a
  is_arithmetic : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d
  sum_first_three : a 1 + a 2 + a 3 = 21
  product_first_three : a 1 * a 2 * a 3 = 231

/-- The second term of the sequence is 7 -/
theorem second_term_is_seven (seq : ArithmeticSequence) : seq.a 2 = 7 := by
  sorry

/-- The general formula for the n-th term -/
theorem general_formula (seq : ArithmeticSequence) : ∀ n : ℕ, seq.a n = 4 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_second_term_is_seven_general_formula_l1188_118849


namespace NUMINAMATH_CALUDE_girls_attending_sports_event_l1188_118859

theorem girls_attending_sports_event (total_students : ℕ) (attending_students : ℕ) 
  (h1 : total_students = 1500)
  (h2 : attending_students = 900)
  (h3 : ∃ (girls boys : ℕ), girls + boys = total_students ∧ 
                             (girls / 2 : ℚ) + (3 * boys / 5 : ℚ) = attending_students) :
  ∃ (girls : ℕ), girls / 2 = 500 := by
sorry

end NUMINAMATH_CALUDE_girls_attending_sports_event_l1188_118859


namespace NUMINAMATH_CALUDE_cut_cube_total_count_l1188_118894

/-- Represents a cube that has been cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes along each edge of the original cube -/
  edge_count : ℕ
  /-- The number of smaller cubes painted on exactly two faces -/
  two_face_painted : ℕ

/-- Theorem stating that if a cube is cut such that 12 smaller cubes are painted on 2 faces,
    then the total number of smaller cubes is 27 -/
theorem cut_cube_total_count (c : CutCube) (h : c.two_face_painted = 12) : 
  c.edge_count ^ 3 = 27 := by
  sorry

#check cut_cube_total_count

end NUMINAMATH_CALUDE_cut_cube_total_count_l1188_118894


namespace NUMINAMATH_CALUDE_sum_inequality_l1188_118856

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) :
  a / (a^3 + b*c) + b / (b^3 + a*c) + c / (c^3 + a*b) > 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1188_118856


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1188_118817

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_remainder (a₁ d aₙ : ℕ) (h1 : a₁ = 3) (h2 : d = 6) (h3 : aₙ = 309) :
  arithmetic_sequence_sum a₁ d aₙ % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l1188_118817


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l1188_118851

theorem quadratic_roots_opposite_signs (p : ℝ) (hp : p > 2) :
  let f (x : ℝ) := 5 * x^2 - 4 * (p + 3) * x + 4 - p^2
  let x₁ := p + 2
  let x₂ := (-p + 2) / 5
  (f x₁ = 0) ∧ (f x₂ = 0) ∧ (x₁ * x₂ < 0) := by
  sorry

#check quadratic_roots_opposite_signs

end NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l1188_118851


namespace NUMINAMATH_CALUDE_sum_four_digit_ending_zero_value_l1188_118857

/-- The sum of all four-digit positive integers ending in 0 -/
def sum_four_digit_ending_zero : ℕ :=
  let first_term := 1000
  let last_term := 9990
  let common_difference := 10
  let num_terms := (last_term - first_term) / common_difference + 1
  num_terms * (first_term + last_term) / 2

theorem sum_four_digit_ending_zero_value : 
  sum_four_digit_ending_zero = 4945500 := by
  sorry

end NUMINAMATH_CALUDE_sum_four_digit_ending_zero_value_l1188_118857


namespace NUMINAMATH_CALUDE_total_lemons_l1188_118840

/-- The number of lemons each person has -/
structure LemonCounts where
  levi : ℕ
  jayden : ℕ
  eli : ℕ
  ian : ℕ

/-- The conditions of the lemon problem -/
def lemon_problem (c : LemonCounts) : Prop :=
  c.levi = 5 ∧
  c.jayden = c.levi + 6 ∧
  c.jayden * 3 = c.eli ∧
  c.eli * 2 = c.ian

/-- The theorem stating the total number of lemons -/
theorem total_lemons (c : LemonCounts) :
  lemon_problem c → c.levi + c.jayden + c.eli + c.ian = 115 := by
  sorry

end NUMINAMATH_CALUDE_total_lemons_l1188_118840


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l1188_118821

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of metallic crayons in the box -/
def metallic_crayons : ℕ := 2

/-- The number of crayons to be selected -/
def selection_size : ℕ := 5

/-- The number of ways to select crayons with the given conditions -/
def selection_ways : ℕ := metallic_crayons * choose (total_crayons - metallic_crayons) (selection_size - 1)

theorem crayon_selection_theorem : selection_ways = 1430 := by sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l1188_118821


namespace NUMINAMATH_CALUDE_prob_no_female_ends_correct_l1188_118885

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The probability that neither end is a female student when arranging the students in a row -/
def prob_no_female_ends : ℚ := 1 / 5

theorem prob_no_female_ends_correct :
  (num_male.choose 2 * (total_students - 2).factorial) / total_students.factorial = prob_no_female_ends :=
sorry

end NUMINAMATH_CALUDE_prob_no_female_ends_correct_l1188_118885


namespace NUMINAMATH_CALUDE_problem_solution_l1188_118801

theorem problem_solution (a b : ℚ) 
  (h1 : 5 + a = 3 - b) 
  (h2 : 3 + b = 8 + a) : 
  5 - a = 17 / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1188_118801


namespace NUMINAMATH_CALUDE_complex_modulus_cos_sin_three_l1188_118843

theorem complex_modulus_cos_sin_three : 
  let z : ℂ := Complex.mk (Real.cos 3) (Real.sin 3)
  |(Complex.abs z - 1)| < 1e-10 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_cos_sin_three_l1188_118843


namespace NUMINAMATH_CALUDE_ski_class_ratio_l1188_118809

theorem ski_class_ratio (b g : ℕ) : 
  b + g ≥ 66 →
  (b + 11 : ℤ) = (g - 13 : ℤ) →
  b ≠ 5 ∨ g ≠ 11 :=
by sorry

end NUMINAMATH_CALUDE_ski_class_ratio_l1188_118809


namespace NUMINAMATH_CALUDE_long_tennis_players_l1188_118811

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 38 →
  football = 26 →
  both = 17 →
  neither = 9 →
  ∃ long_tennis : ℕ, long_tennis = 20 ∧ total = football + long_tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_long_tennis_players_l1188_118811


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1188_118822

/-- Given an arithmetic sequence -1, a, b, m, 7, prove the eccentricity of x²/a² - y²/b² = 1 is √10 -/
theorem hyperbola_eccentricity (a b m : ℝ) : 
  (∃ d : ℝ, a = -1 + d ∧ b = a + d ∧ m = b + d ∧ 7 = m + d) →
  Real.sqrt ((b / a)^2 + 1) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1188_118822


namespace NUMINAMATH_CALUDE_smallest_stairs_solution_l1188_118884

theorem smallest_stairs_solution (n : ℕ) : 
  (n > 20 ∧ n % 6 = 5 ∧ n % 7 = 4) → n ≥ 53 :=
by sorry

end NUMINAMATH_CALUDE_smallest_stairs_solution_l1188_118884


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1188_118810

/-- The function f(x) = x^2 - 2x + 6 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 6

/-- Theorem stating that f(m+3) > f(2m) is equivalent to -1/3 < m < 3 -/
theorem inequality_equivalence (m : ℝ) : 
  f (m + 3) > f (2 * m) ↔ -1/3 < m ∧ m < 3 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1188_118810


namespace NUMINAMATH_CALUDE_karlson_expenditure_can_exceed_2000_l1188_118899

theorem karlson_expenditure_can_exceed_2000 :
  ∃ (n m : ℕ), 25 * n + 340 * m > 2000 :=
by sorry

end NUMINAMATH_CALUDE_karlson_expenditure_can_exceed_2000_l1188_118899


namespace NUMINAMATH_CALUDE_polynomial_identity_l1188_118895

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (2 - Real.sqrt 3 * x)^8 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  (a₀ + a₂ + a₄ + a₆ + a₈)^2 - (a₁ + a₃ + a₅ + a₇)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1188_118895


namespace NUMINAMATH_CALUDE_eighteen_team_tournament_games_l1188_118842

/-- Calculates the number of games in a knockout tournament with byes -/
def knockout_tournament_games (total_teams : ℕ) (bye_teams : ℕ) : ℕ :=
  total_teams - 1

/-- Theorem: A knockout tournament with 18 teams and 2 byes has 17 games -/
theorem eighteen_team_tournament_games :
  knockout_tournament_games 18 2 = 17 := by
  sorry

#eval knockout_tournament_games 18 2

end NUMINAMATH_CALUDE_eighteen_team_tournament_games_l1188_118842


namespace NUMINAMATH_CALUDE_mean_car_sales_l1188_118883

def car_sales : List Nat := [8, 3, 10, 4, 4, 4]

theorem mean_car_sales :
  (car_sales.sum : ℚ) / car_sales.length = 5.5 := by sorry

end NUMINAMATH_CALUDE_mean_car_sales_l1188_118883


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_21_l1188_118830

/-- A set of consecutive positive integers -/
def ConsecutiveSet (start : ℕ) (length : ℕ) : Set ℕ :=
  {n | start ≤ n ∧ n < start + length}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

/-- Predicate for a valid set of consecutive integers summing to 21 -/
def ValidSet (start : ℕ) (length : ℕ) : Prop :=
  length ≥ 2 ∧ ConsecutiveSum start length = 21

theorem unique_consecutive_sum_21 :
  ∃! p : ℕ × ℕ, ValidSet p.1 p.2 := by sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_21_l1188_118830


namespace NUMINAMATH_CALUDE_bacteria_growth_l1188_118871

-- Define the division rate of bacteria
def division_rate : ℕ := 10

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 1

-- Define the time passed in minutes
def time_passed : ℕ := 120

-- Define the function to calculate the number of bacteria
def num_bacteria (t : ℕ) : ℕ := 2 ^ (t / division_rate)

-- Theorem to prove
theorem bacteria_growth :
  num_bacteria time_passed = 2^12 :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1188_118871


namespace NUMINAMATH_CALUDE_better_scores_seventh_grade_l1188_118826

/-- Represents the grade level of students -/
inductive Grade
  | Seventh
  | Eighth

/-- Represents statistical measures for a set of scores -/
structure ScoreStatistics where
  mean : ℝ
  median : ℝ
  mode : ℝ
  variance : ℝ

/-- The test scores for a grade -/
def scores (g : Grade) : List ℝ :=
  match g with
  | Grade.Seventh => [96, 85, 90, 86, 93, 92, 95, 81, 75, 81]
  | Grade.Eighth => [68, 95, 83, 93, 94, 75, 85, 95, 95, 77]

/-- The statistical measures for a grade -/
def statistics (g : Grade) : ScoreStatistics :=
  match g with
  | Grade.Seventh => ⟨87.4, 88, 81, 43.44⟩
  | Grade.Eighth => ⟨86, 89, 95, 89.2⟩

/-- Maximum possible score -/
def maxScore : ℝ := 100

theorem better_scores_seventh_grade :
  (statistics Grade.Seventh).median = 88 ∧
  (statistics Grade.Eighth).mode = 95 ∧
  (statistics Grade.Seventh).mean > (statistics Grade.Eighth).mean ∧
  (statistics Grade.Seventh).variance < (statistics Grade.Eighth).variance :=
by sorry

end NUMINAMATH_CALUDE_better_scores_seventh_grade_l1188_118826


namespace NUMINAMATH_CALUDE_cereal_eating_time_l1188_118802

/-- The time taken for three people to eat a certain amount of cereal together -/
def time_to_eat (fat_rate thin_rate medium_rate total_cereal : ℚ) : ℚ :=
  total_cereal / (fat_rate + thin_rate + medium_rate)

theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15
  let thin_rate : ℚ := 1 / 35
  let medium_rate : ℚ := 1 / 25
  let total_cereal : ℚ := 5
  time_to_eat fat_rate thin_rate medium_rate total_cereal = 2625 / 71 :=
by sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l1188_118802


namespace NUMINAMATH_CALUDE_smallest_yellow_candy_quantity_l1188_118831

def red_candy_cost : ℕ := 8
def green_candy_cost : ℕ := 12
def blue_candy_cost : ℕ := 15
def yellow_candy_cost : ℕ := 24

def red_candy_quantity : ℕ := 10
def green_candy_quantity : ℕ := 18
def blue_candy_quantity : ℕ := 20

def red_total_cost : ℕ := red_candy_cost * red_candy_quantity
def green_total_cost : ℕ := green_candy_cost * green_candy_quantity
def blue_total_cost : ℕ := blue_candy_cost * blue_candy_quantity

theorem smallest_yellow_candy_quantity :
  ∃ (n : ℕ), n > 0 ∧
  (yellow_candy_cost * n) % red_total_cost = 0 ∧
  (yellow_candy_cost * n) % green_total_cost = 0 ∧
  (yellow_candy_cost * n) % blue_total_cost = 0 ∧
  ∀ (m : ℕ), m > 0 →
    (yellow_candy_cost * m) % red_total_cost = 0 →
    (yellow_candy_cost * m) % green_total_cost = 0 →
    (yellow_candy_cost * m) % blue_total_cost = 0 →
    m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_candy_quantity_l1188_118831


namespace NUMINAMATH_CALUDE_magician_marbles_problem_l1188_118836

theorem magician_marbles_problem (initial_red : ℕ) (initial_blue : ℕ) 
  (red_taken : ℕ) (blue_taken_multiplier : ℕ) :
  initial_red = 20 →
  initial_blue = 30 →
  red_taken = 3 →
  blue_taken_multiplier = 4 →
  (initial_red - red_taken) + (initial_blue - (blue_taken_multiplier * red_taken)) = 35 :=
by sorry

end NUMINAMATH_CALUDE_magician_marbles_problem_l1188_118836


namespace NUMINAMATH_CALUDE_point_C_coordinates_l1188_118887

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line that point C is on
def line_C (x y : ℝ) : Prop := 3 * x - y + 3 = 0

-- Define the area of triangle ABC
def triangle_area (C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
  line_C C.1 C.2 →
  triangle_area C = 10 →
  C = (-1, 0) ∨ C = (5/3, 8) :=
sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l1188_118887


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1188_118878

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℤ, (x < 0 ∧ -4 * x - k ≤ 0) ↔ (x = -1 ∨ x = -2)) →
  (8 ≤ k ∧ k < 12) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1188_118878


namespace NUMINAMATH_CALUDE_john_total_cost_l1188_118837

def nike_cost : ℝ := 150
def boot_cost : ℝ := 120
def tax_rate : ℝ := 0.1

def total_cost (nike : ℝ) (boot : ℝ) (tax : ℝ) : ℝ :=
  let subtotal := nike + boot
  let tax_amount := subtotal * tax
  subtotal + tax_amount

theorem john_total_cost :
  total_cost nike_cost boot_cost tax_rate = 297 :=
sorry

end NUMINAMATH_CALUDE_john_total_cost_l1188_118837


namespace NUMINAMATH_CALUDE_ratio_equality_implies_fraction_value_l1188_118812

theorem ratio_equality_implies_fraction_value
  (a b c : ℝ)
  (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b) / (b - c) = -7 :=
by sorry

end NUMINAMATH_CALUDE_ratio_equality_implies_fraction_value_l1188_118812


namespace NUMINAMATH_CALUDE_chess_game_theorem_l1188_118818

/-- Represents a three-player turn-based game system -/
structure GameSystem where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The game system satisfies the conditions of the problem -/
def valid_game_system (g : GameSystem) : Prop :=
  g.total_games = 27 ∧
  g.player1_games = 27 ∧
  g.player2_games = 13 ∧
  g.player3_games = g.total_games - g.player2_games

theorem chess_game_theorem (g : GameSystem) (h : valid_game_system g) :
  g.player3_games = 14 := by
  sorry


end NUMINAMATH_CALUDE_chess_game_theorem_l1188_118818


namespace NUMINAMATH_CALUDE_circle_sum_center_radius_l1188_118848

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 2*x - 8*y - 7 = -y^2 - 6*x

-- Define the center and radius
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_sum_center_radius :
  ∃ (a b r : ℝ), is_center_radius a b r ∧ a + b + r = Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_center_radius_l1188_118848


namespace NUMINAMATH_CALUDE_outfit_combinations_l1188_118839

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 6

/-- The number of different types of clothing items -/
def num_items : ℕ := 4

/-- The number of valid outfit combinations -/
def valid_combinations : ℕ := num_colors * (num_colors - 1) * (num_colors - 2) * (num_colors - 3)

theorem outfit_combinations :
  valid_combinations = 360 :=
sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1188_118839


namespace NUMINAMATH_CALUDE_distance_covered_l1188_118825

/-- Proves that the total distance covered is 6 km given the specified conditions -/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 2.25)
  (h4 : (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time)
  : total_distance = 6 :=
by
  sorry

#check distance_covered

end NUMINAMATH_CALUDE_distance_covered_l1188_118825


namespace NUMINAMATH_CALUDE_value_of_a_l1188_118854

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 9) 
  (eq3 : c = 4) : 
  a = -1 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l1188_118854


namespace NUMINAMATH_CALUDE_inequality_system_solution_condition_l1188_118847

theorem inequality_system_solution_condition (m : ℝ) :
  (∃ x : ℝ, x - 2 * m < 0 ∧ x + m > 2) → m > 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_condition_l1188_118847


namespace NUMINAMATH_CALUDE_total_players_count_l1188_118803

/-- The number of players who play kabadi -/
def kabadi_players : ℕ := 10

/-- The number of players who play kho kho only -/
def kho_kho_only_players : ℕ := 40

/-- The number of players who play both games -/
def both_games_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := kabadi_players + kho_kho_only_players - both_games_players

theorem total_players_count : total_players = 45 := by sorry

end NUMINAMATH_CALUDE_total_players_count_l1188_118803
