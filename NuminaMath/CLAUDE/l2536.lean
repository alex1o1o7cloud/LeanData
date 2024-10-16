import Mathlib

namespace NUMINAMATH_CALUDE_number_ordering_l2536_253690

theorem number_ordering : 
  (-1.1 : ℝ) < -0.75 ∧ 
  -0.75 < -2/3 ∧ 
  -2/3 < 1/200 ∧ 
  1/200 = (0.005 : ℝ) ∧ 
  0.005 < 4/6 ∧ 
  4/6 < 5/7 ∧ 
  5/7 < 11/15 ∧ 
  11/15 < 1 := by sorry

end NUMINAMATH_CALUDE_number_ordering_l2536_253690


namespace NUMINAMATH_CALUDE_alloy_composition_proof_l2536_253604

-- Define the initial alloy properties
def initial_weight : ℝ := 25
def initial_gold_percentage : ℝ := 0.5
def target_gold_percentage : ℝ := 0.9

-- Define the amount of pure gold to be added
def added_gold : ℝ := 100

-- Theorem statement
theorem alloy_composition_proof :
  let initial_gold := initial_weight * initial_gold_percentage
  let final_weight := initial_weight + added_gold
  let final_gold := initial_gold + added_gold
  (final_gold / final_weight) = target_gold_percentage := by
  sorry

end NUMINAMATH_CALUDE_alloy_composition_proof_l2536_253604


namespace NUMINAMATH_CALUDE_ndoti_winning_strategy_l2536_253677

/-- Represents a point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square on the plane -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Represents a quadrilateral on the plane -/
structure Quadrilateral where
  x : Point
  y : Point
  z : Point
  w : Point

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- Checks if a point is on a side of the square -/
def isOnSquareSide (p : Point) (s : Square) : Prop := sorry

/-- Ndoti's strategy function -/
def ndotiStrategy (s : Square) (x : Point) : Quadrilateral := sorry

/-- The main theorem stating Ndoti's winning strategy -/
theorem ndoti_winning_strategy (s : Square) :
  ∀ x : Point, isOnSquareSide x s →
    quadrilateralArea (ndotiStrategy s x) < (1/2) * squareArea s :=
by sorry

end NUMINAMATH_CALUDE_ndoti_winning_strategy_l2536_253677


namespace NUMINAMATH_CALUDE_gcd_of_large_numbers_l2536_253628

theorem gcd_of_large_numbers : Nat.gcd 1000000000 1000000005 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_large_numbers_l2536_253628


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l2536_253655

/-- Pentagon ABCDE with specified side lengths and relationships -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  AC : ℝ
  AD : ℝ
  AE : ℝ
  ab_eq_one : AB = 1
  bc_eq_one : BC = 1
  cd_eq_one : CD = 1
  de_eq_one : DE = 1
  ac_pythagoras : AC^2 = AB^2 + BC^2
  ad_pythagoras : AD^2 = AC^2 + CD^2
  ae_pythagoras : AE^2 = AD^2 + DE^2

/-- The perimeter of pentagon ABCDE is 6 -/
theorem pentagon_perimeter (p : Pentagon) : p.AB + p.BC + p.CD + p.DE + p.AE = 6 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_perimeter_l2536_253655


namespace NUMINAMATH_CALUDE_two_real_roots_l2536_253648

-- Define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem two_real_roots (a b c : ℝ) 
  (h : ∀ x ∈ Set.Icc (-1) 1, |f a b c x| < 1) :
  ∃ x y : ℝ, x ≠ y ∧ f a b c x = 2 * x^2 - 1 ∧ f a b c y = 2 * y^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_two_real_roots_l2536_253648


namespace NUMINAMATH_CALUDE_no_k_exists_product_odd_primes_minus_one_is_power_l2536_253602

/-- The nth odd prime number -/
def nthOddPrime (n : ℕ) : ℕ := sorry

/-- The product of the first k odd prime numbers -/
def productFirstKOddPrimes (k : ℕ) : ℕ := sorry

/-- Theorem: There does not exist a natural number k such that the product of the first k odd prime numbers minus 1 is an exact power of a natural number greater than one -/
theorem no_k_exists_product_odd_primes_minus_one_is_power :
  ¬ ∃ (k : ℕ), ∃ (a n : ℕ), n > 1 ∧ productFirstKOddPrimes k = a^n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_k_exists_product_odd_primes_minus_one_is_power_l2536_253602


namespace NUMINAMATH_CALUDE_different_meal_combinations_l2536_253685

theorem different_meal_combinations (n : ℕ) (h : n = 12) : n * (n - 1) = 132 := by
  sorry

end NUMINAMATH_CALUDE_different_meal_combinations_l2536_253685


namespace NUMINAMATH_CALUDE_minimum_discount_rate_correct_l2536_253660

/-- The minimum discount rate (as a percentage) that ensures a gross profit of at least 12.5% 
    for a product with a cost price of 400 yuan and a selling price of 500 yuan. -/
def minimum_discount_rate : ℝ := 9

/-- The cost price of the product in yuan. -/
def cost_price : ℝ := 400

/-- The selling price of the product in yuan. -/
def selling_price : ℝ := 500

/-- The minimum required gross profit as a percentage of the cost price. -/
def min_gross_profit_percentage : ℝ := 12.5

theorem minimum_discount_rate_correct :
  ∀ x : ℝ, x ≥ minimum_discount_rate → 
  (selling_price - selling_price * (x / 100) - cost_price ≥ cost_price * (min_gross_profit_percentage / 100)) ∧
  ∀ y : ℝ, y < minimum_discount_rate → 
  (selling_price - selling_price * (y / 100) - cost_price > cost_price * (min_gross_profit_percentage / 100)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_discount_rate_correct_l2536_253660


namespace NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l2536_253654

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (1/4 : ℚ) + (x : ℚ)/5 < 3/2 ↔ x ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l2536_253654


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2536_253657

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*a*x + 1 < 0) ↔ a ∈ Set.Ioi 1 ∪ Set.Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2536_253657


namespace NUMINAMATH_CALUDE_remainder_problem_l2536_253601

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 38 = 32 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2536_253601


namespace NUMINAMATH_CALUDE_n_squared_not_divides_n_factorial_l2536_253608

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem n_squared_not_divides_n_factorial (n : ℕ) :
  (¬ divides (n^2) (n!)) ↔ (Nat.Prime n ∨ n = 4) := by sorry

end NUMINAMATH_CALUDE_n_squared_not_divides_n_factorial_l2536_253608


namespace NUMINAMATH_CALUDE_area_of_special_trapezoid_l2536_253692

/-- An isosceles trapezoid with a circle inscribed in it -/
structure InscribedCircleTrapezoid where
  /-- Length of the shorter base -/
  a : ℝ
  /-- Length of the longer base -/
  b : ℝ
  /-- Length of the leg -/
  c : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : c > 0
  /-- A circle can be inscribed in the trapezoid -/
  hasInscribedCircle : a + b = 2 * c

/-- The area of an isosceles trapezoid with bases 2 and 8, in which a circle can be inscribed, is 20 -/
theorem area_of_special_trapezoid :
  ∀ t : InscribedCircleTrapezoid, t.a = 2 ∧ t.b = 8 → 
  (1/2 : ℝ) * (t.a + t.b) * Real.sqrt (t.c^2 - ((t.b - t.a)/2)^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_area_of_special_trapezoid_l2536_253692


namespace NUMINAMATH_CALUDE_flowers_left_l2536_253635

theorem flowers_left (alissa_flowers melissa_flowers given_flowers : ℕ) :
  alissa_flowers = 16 →
  melissa_flowers = 16 →
  given_flowers = 18 →
  alissa_flowers + melissa_flowers - given_flowers = 14 := by
sorry

end NUMINAMATH_CALUDE_flowers_left_l2536_253635


namespace NUMINAMATH_CALUDE_ten_point_circle_triangles_l2536_253653

/-- The number of triangles that can be formed from n distinct points on a circle's circumference -/
def total_triangles (n : ℕ) : ℕ := Nat.choose n 3

/-- The number of triangles where one side subtends an arc greater than 180 degrees -/
def long_arc_triangles (n : ℕ) : ℕ := 2 * n

/-- The number of valid triangles that can be formed from n distinct points on a circle's circumference,
    where no side subtends an arc greater than 180 degrees -/
def valid_triangles (n : ℕ) : ℕ := total_triangles n - long_arc_triangles n

theorem ten_point_circle_triangles :
  valid_triangles 10 = 100 := by sorry

end NUMINAMATH_CALUDE_ten_point_circle_triangles_l2536_253653


namespace NUMINAMATH_CALUDE_point_not_in_second_quadrant_l2536_253676

theorem point_not_in_second_quadrant (a : ℝ) :
  ¬(a < 0 ∧ 2*a - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_second_quadrant_l2536_253676


namespace NUMINAMATH_CALUDE_dance_event_relation_l2536_253619

/-- Represents a dance event with boys and girls -/
structure DanceEvent where
  b : ℕ  -- Total number of boys
  g : ℕ  -- Total number of girls

/-- The number of girls the nth boy dances with -/
def girlsForBoy (n : ℕ) : ℕ := 7 + 2 * (n - 1)

/-- Axiom: The last boy dances with all girls -/
axiom last_boy_dances_all (event : DanceEvent) : girlsForBoy event.b = event.g

/-- Theorem: The relationship between boys and girls in the dance event -/
theorem dance_event_relation (event : DanceEvent) : event.b = (event.g - 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_dance_event_relation_l2536_253619


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2536_253647

theorem largest_digit_divisible_by_six : 
  ∃ (M : ℕ), M ≤ 9 ∧ (45670 + M) % 6 = 0 ∧ 
  ∀ (N : ℕ), N ≤ 9 ∧ (45670 + N) % 6 = 0 → N ≤ M :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2536_253647


namespace NUMINAMATH_CALUDE_sum_odd_product_even_l2536_253695

theorem sum_odd_product_even (a b : ℤ) : 
  Odd (a + b) → Even (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_product_even_l2536_253695


namespace NUMINAMATH_CALUDE_exists_x0_where_P_less_than_Q_l2536_253641

/-- Given two polynomials P and Q, and an interval (r,s) satisfying certain conditions,
    there exists a real x₀ such that P(x₀) < Q(x₀) -/
theorem exists_x0_where_P_less_than_Q
  (a b c d p q r s : ℝ)
  (P : ℝ → ℝ)
  (Q : ℝ → ℝ)
  (h_P : ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + d)
  (h_Q : ∀ x, Q x = x^2 + p*x + q)
  (h_interval : s - r > 2)
  (h_negative : ∀ x, r < x ∧ x < s → P x < 0 ∧ Q x < 0)
  (h_positive_right : ∀ x, x > s → P x > 0 ∧ Q x > 0)
  (h_positive_left : ∀ x, x < r → P x > 0 ∧ Q x > 0) :
  ∃ x₀, P x₀ < Q x₀ :=
by sorry

end NUMINAMATH_CALUDE_exists_x0_where_P_less_than_Q_l2536_253641


namespace NUMINAMATH_CALUDE_unique_number_with_special_divisor_property_l2536_253605

theorem unique_number_with_special_divisor_property : 
  ∃! (N : ℕ), 
    N > 0 ∧ 
    (∃ (k : ℕ), 
      N + (Nat.factors N).foldl Nat.lcm 1 = 10^k) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_special_divisor_property_l2536_253605


namespace NUMINAMATH_CALUDE_probability_in_standard_deck_l2536_253645

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (number_cards : Nat)
  (face_cards : Nat)

/-- The probability of drawing a number card first and a face card second -/
def probability_number_then_face (d : Deck) : Rat :=
  (d.number_cards : Rat) / d.total_cards * d.face_cards / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a number card first and a face card second from a standard deck -/
theorem probability_in_standard_deck :
  let standard_deck : Deck := ⟨52, 36, 12⟩
  probability_number_then_face standard_deck = 36 / 221 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_standard_deck_l2536_253645


namespace NUMINAMATH_CALUDE_bus_journey_distance_l2536_253687

theorem bus_journey_distance :
  ∀ (D s : ℝ),
    -- Original expected travel time
    (D / s = 2 + 1 + (3 * (D - 2 * s)) / (2 * s)) →
    -- Actual travel time (6 hours late)
    (2 + 1 + (3 * (D - 2 * s)) / (2 * s) = D / s + 6) →
    -- Travel time if delay occurred 120 miles further (4 hours late)
    ((2 * s + 120) / s + 1 + (3 * (D - 2 * s - 120)) / (2 * s) = D / s + 4) →
    -- Bus continues at 2/3 of original speed after delay
    (D - 2 * s) / ((2/3) * s) = (3 * (D - 2 * s)) / (2 * s) →
    D = 720 :=
by sorry

end NUMINAMATH_CALUDE_bus_journey_distance_l2536_253687


namespace NUMINAMATH_CALUDE_initial_value_exists_l2536_253630

/-- Represents a small deviation from exactness -/
def small_deviation : ℝ := 0.5

/-- A number is approximately equal to another if their difference is less than a small deviation -/
def approx_equal (x y : ℝ) : Prop := |x - y| < small_deviation

theorem initial_value_exists (n : ℤ) : ∃ x : ℝ, approx_equal (x + 21) (136 * n) := by
  sorry

end NUMINAMATH_CALUDE_initial_value_exists_l2536_253630


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2536_253664

theorem greatest_integer_b_for_all_real_domain : ∃ b : ℤ,
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ 0) ∧
  (∀ c : ℤ, c > b → ∃ x : ℝ, x^2 + (c : ℝ) * x + 10 = 0) ∧
  b = 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2536_253664


namespace NUMINAMATH_CALUDE_no_prime_for_expression_l2536_253640

theorem no_prime_for_expression (p : ℕ) (hp : Nat.Prime p) : ¬ Nat.Prime (22 * p^2 + 23) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_for_expression_l2536_253640


namespace NUMINAMATH_CALUDE_article_profit_percentage_l2536_253673

theorem article_profit_percentage (CP : ℝ) (G : ℝ) : 
  CP = 800 →
  (CP * 0.95) * 1.1 = CP * (1 + G / 100) - 4 →
  G = 5 := by
sorry

end NUMINAMATH_CALUDE_article_profit_percentage_l2536_253673


namespace NUMINAMATH_CALUDE_sams_recycling_cans_l2536_253694

theorem sams_recycling_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : 
  saturday_bags = 4 → sunday_bags = 3 → cans_per_bag = 6 →
  (saturday_bags + sunday_bags) * cans_per_bag = 42 := by
  sorry

end NUMINAMATH_CALUDE_sams_recycling_cans_l2536_253694


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l2536_253617

-- Define i as a complex number with i² = -1
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i :
  (Finset.range 603).sum (fun k => i ^ k) = i - 1 := by
  sorry

-- Note: The proof is omitted as per your instructions.

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l2536_253617


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_t_l2536_253671

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 2*|x + 3|

-- Theorem for the first part of the problem
theorem solution_set_f (x : ℝ) :
  f x ≥ 2 ↔ -4 ≤ x ∧ x ≤ -8/3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_t :
  ∃ t₁ t₂ : ℝ, t₁ = -1/3 ∧ t₂ = 5/3 ∧
  (∀ t : ℝ, (∃ x : ℝ, f x - |3*t - 2| ≥ 0) ↔ t₁ ≤ t ∧ t ≤ t₂) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_t_l2536_253671


namespace NUMINAMATH_CALUDE_consecutive_three_digit_prime_factors_l2536_253638

theorem consecutive_three_digit_prime_factors :
  ∀ n : ℕ, 
    100 ≤ n ∧ n + 9 ≤ 999 →
    ∃ (S : Finset ℕ),
      (∀ p ∈ S, Nat.Prime p) ∧
      (Finset.card S ≤ 23) ∧
      (∀ k : ℕ, n ≤ k ∧ k ≤ n + 9 → ∀ p : ℕ, Nat.Prime p → p ∣ k → p ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_three_digit_prime_factors_l2536_253638


namespace NUMINAMATH_CALUDE_years_calculation_l2536_253666

/-- The number of years for which a sum was put at simple interest -/
def years_at_interest (principal : ℚ) (additional_interest : ℚ) : ℚ :=
  additional_interest * 100 / principal

theorem years_calculation (principal : ℚ) (additional_interest : ℚ) 
  (h1 : principal = 2600)
  (h2 : additional_interest = 78) :
  years_at_interest principal additional_interest = 3 := by
  sorry

#eval years_at_interest 2600 78

end NUMINAMATH_CALUDE_years_calculation_l2536_253666


namespace NUMINAMATH_CALUDE_product_of_decimals_l2536_253650

theorem product_of_decimals : (0.5 : ℝ) * 0.8 = 0.40 := by sorry

end NUMINAMATH_CALUDE_product_of_decimals_l2536_253650


namespace NUMINAMATH_CALUDE_buffalo_count_l2536_253611

/-- Represents the number of buffaloes in the group -/
def num_buffaloes : ℕ → ℕ → ℕ := sorry

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ → ℕ → ℕ := sorry

/-- The total number of legs in the group -/
def total_legs (b d : ℕ) : ℕ := 4 * b + 2 * d

/-- The total number of heads in the group -/
def total_heads (b d : ℕ) : ℕ := b + d

theorem buffalo_count (b d : ℕ) : 
  total_legs b d = 2 * total_heads b d + 24 → num_buffaloes b d = 12 := by
  sorry

end NUMINAMATH_CALUDE_buffalo_count_l2536_253611


namespace NUMINAMATH_CALUDE_inequality_proof_l2536_253689

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2536_253689


namespace NUMINAMATH_CALUDE_couples_in_club_is_three_l2536_253688

/-- Represents a book club with couples and single members -/
structure BookClub where
  weeksPerYear : ℕ
  ronPicksPerYear : ℕ
  singleMembers : ℕ

/-- Calculates the number of couples in the book club -/
def couplesInClub (club : BookClub) : ℕ :=
  (club.weeksPerYear - (2 * club.ronPicksPerYear + club.singleMembers * club.ronPicksPerYear)) / (2 * club.ronPicksPerYear)

/-- Theorem stating that the number of couples in the specified book club is 3 -/
theorem couples_in_club_is_three (club : BookClub) 
  (h1 : club.weeksPerYear = 52)
  (h2 : club.ronPicksPerYear = 4)
  (h3 : club.singleMembers = 5) : 
  couplesInClub club = 3 := by
  sorry

end NUMINAMATH_CALUDE_couples_in_club_is_three_l2536_253688


namespace NUMINAMATH_CALUDE_power_of_power_l2536_253649

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2536_253649


namespace NUMINAMATH_CALUDE_percentage_problem_l2536_253651

theorem percentage_problem (x : ℝ) (h : 150 = 250 / 100 * x) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2536_253651


namespace NUMINAMATH_CALUDE_elevator_problem_l2536_253616

theorem elevator_problem (initial_people : ℕ) (remaining_people : ℕ) 
  (h1 : initial_people = 18) 
  (h2 : remaining_people = 11) : 
  initial_people - remaining_people = 7 := by
  sorry

end NUMINAMATH_CALUDE_elevator_problem_l2536_253616


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2536_253606

theorem inscribed_cube_surface_area (outer_cube_area : ℝ) (h : outer_cube_area = 54) :
  let outer_side := Real.sqrt (outer_cube_area / 6)
  let sphere_diameter := outer_side
  let inner_side := Real.sqrt (sphere_diameter^2 / 3)
  let inner_cube_area := 6 * inner_side^2
  inner_cube_area = 18 := by sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2536_253606


namespace NUMINAMATH_CALUDE_cats_at_rescue_center_l2536_253684

/-- The number of puppies Hartley has -/
def num_puppies : ℕ := 4

/-- The weight of each puppy in kilograms -/
def puppy_weight : ℚ := 7.5

/-- The weight of each cat in kilograms -/
def cat_weight : ℚ := 2.5

/-- The difference in total weight between cats and puppies in kilograms -/
def weight_difference : ℚ := 5

/-- The number of cats at the rescue center -/
def num_cats : ℕ := 14

theorem cats_at_rescue_center : 
  (↑num_cats : ℚ) * cat_weight = 
    ↑num_puppies * puppy_weight + weight_difference := by
  sorry

end NUMINAMATH_CALUDE_cats_at_rescue_center_l2536_253684


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2536_253622

theorem minimum_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 - b + 4 ≤ 0) :
  ∃ (min : ℝ), min = 14/5 ∧ ∀ x, x = (2*a + 3*b)/(a + b) → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2536_253622


namespace NUMINAMATH_CALUDE_chess_competition_result_l2536_253636

/-- Represents the number of 8th-grade students in the chess competition. -/
def n : ℕ := sorry

/-- Represents the score of each 8th-grade student. -/
def σ : ℚ := sorry

/-- The total points scored by the two 7th-grade students. -/
def seventh_grade_total : ℕ := 8

/-- The theorem stating the conditions and the result of the chess competition. -/
theorem chess_competition_result :
  (∃ (n : ℕ) (σ : ℚ),
    n > 0 ∧
    σ = (2 * n - 7 : ℚ) / n ∧
    σ = 2 - 7 / n ∧
    (σ = 1 ∨ σ = (3 : ℚ) / 2) ∧
    n = 7) :=
sorry

end NUMINAMATH_CALUDE_chess_competition_result_l2536_253636


namespace NUMINAMATH_CALUDE_frank_uniform_number_l2536_253627

def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem frank_uniform_number 
  (d e f : ℕ) 
  (h1 : is_two_digit_prime d) 
  (h2 : is_two_digit_prime e) 
  (h3 : is_two_digit_prime f) 
  (h4 : d + f = 28) 
  (h5 : d + e = 24) 
  (h6 : e + f = 30) : 
  f = 17 := by
sorry

end NUMINAMATH_CALUDE_frank_uniform_number_l2536_253627


namespace NUMINAMATH_CALUDE_unique_valid_number_l2536_253633

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  n % 10 = (n / 10) % 10 ∧
  (n / 100) % 10 = n / 1000 ∧
  ∃ k : ℕ, n = k * k

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 7744 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2536_253633


namespace NUMINAMATH_CALUDE_number_relation_l2536_253669

theorem number_relation (first second multiple : ℕ) : 
  first = 15 →
  second = 55 →
  first + second = 70 →
  second = multiple * first + 10 →
  multiple = 3 := by
sorry

end NUMINAMATH_CALUDE_number_relation_l2536_253669


namespace NUMINAMATH_CALUDE_c_used_car_for_11_hours_l2536_253626

/-- Calculates the number of hours c used the car given the total cost,
    a's usage, b's usage and payment. -/
def calculate_c_hours (total_cost : ℕ) (a_hours : ℕ) (b_hours : ℕ) (b_payment : ℕ) : ℕ :=
  let hourly_rate := b_payment / b_hours
  let a_payment := a_hours * hourly_rate
  let c_payment := total_cost - a_payment - b_payment
  c_payment / hourly_rate

/-- Proves that given the problem conditions, c used the car for 11 hours -/
theorem c_used_car_for_11_hours :
  calculate_c_hours 520 7 8 160 = 11 := by
  sorry

end NUMINAMATH_CALUDE_c_used_car_for_11_hours_l2536_253626


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2536_253609

theorem partial_fraction_sum_zero : 
  ∃ (A B C D E F : ℝ), 
    (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
      1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) = 
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
    A + B + C + D + E + F = 0 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2536_253609


namespace NUMINAMATH_CALUDE_quadratic_sum_l2536_253658

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_sum (a b c : ℝ) :
  f a b c 0 = 8 → f a b c 1 = 9 → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2536_253658


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2536_253682

theorem simplify_trig_expression (θ : ℝ) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = 2 * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2536_253682


namespace NUMINAMATH_CALUDE_congruence_solution_a_l2536_253675

theorem congruence_solution_a (x : Int) : 
  (8 * x) % 13 = 3 % 13 ↔ x % 13 = 2 % 13 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_a_l2536_253675


namespace NUMINAMATH_CALUDE_existence_of_homomorphism_l2536_253697

variable {G : Type*} [Group G]

def special_function (φ : G → G) : Prop :=
  ∀ a b c d e f : G, a * b * c = 1 ∧ d * e * f = 1 → φ a * φ b * φ c = φ d * φ e * φ f

theorem existence_of_homomorphism (φ : G → G) (h : special_function φ) :
  ∃ k : G, ∀ x y : G, k * φ (x * y) = (k * φ x) * (k * φ y) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_homomorphism_l2536_253697


namespace NUMINAMATH_CALUDE_average_price_theorem_l2536_253614

def average_price_6_toys (num_initial_toys : ℕ) (initial_avg_cost : ℚ) 
  (sixth_toy_price : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let total_initial_cost := num_initial_toys * initial_avg_cost
  let sixth_toy_total_cost := sixth_toy_price * (1 + sales_tax_rate)
  let total_cost := total_initial_cost + sixth_toy_total_cost
  total_cost / (num_initial_toys + 1)

theorem average_price_theorem :
  average_price_6_toys 5 10 16 (1/10) = 67.6 / 6 :=
by sorry

end NUMINAMATH_CALUDE_average_price_theorem_l2536_253614


namespace NUMINAMATH_CALUDE_largest_divisor_of_p_squared_minus_q_squared_l2536_253674

theorem largest_divisor_of_p_squared_minus_q_squared (p q : ℤ) 
  (h_p_gt_q : p > q) 
  (h_p_odd : Odd p) 
  (h_q_even : Even q) : 
  (∀ (d : ℤ), d ∣ (p^2 - q^2) → d = 1 ∨ d = -1) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_p_squared_minus_q_squared_l2536_253674


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2536_253618

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * Complex.I
  let z₂ : ℂ := 4 - 7 * Complex.I
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2536_253618


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2536_253679

/-- Given non-zero vectors a and b in a real inner product space, 
    if |a| = √2|b| and (a - b) ⊥ (2a + 3b), 
    then the angle between a and b is 3π/4. -/
theorem angle_between_vectors 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : ‖a‖ = Real.sqrt 2 * ‖b‖) 
  (h2 : @inner ℝ V _ (a - b) (2 • a + 3 • b) = 0) : 
  Real.arccos ((@inner ℝ V _ a b) / (‖a‖ * ‖b‖)) = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2536_253679


namespace NUMINAMATH_CALUDE_triangle_bisector_angle_tangent_l2536_253691

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a line that bisects both perimeter and area of a triangle -/
structure BisectingLine where
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ

/-- The acute angle between two bisecting lines -/
def angleBetweenBisectors (l1 l2 : BisectingLine) : ℝ := sorry

/-- Checks if a line bisects both perimeter and area of a triangle -/
def isBisectingLine (t : Triangle) (l : BisectingLine) : Prop := sorry

theorem triangle_bisector_angle_tangent (t : Triangle) 
  (h1 : t.a = 13 ∧ t.b = 14 ∧ t.c = 15) : 
  ∃ (l1 l2 : BisectingLine) (θ : ℝ),
    isBisectingLine t l1 ∧ 
    isBisectingLine t l2 ∧ 
    θ = angleBetweenBisectors l1 l2 ∧ 
    0 < θ ∧ θ < π/2 ∧
    Real.tan θ = sorry -- This should be replaced with the actual value or expression
    := by sorry

end NUMINAMATH_CALUDE_triangle_bisector_angle_tangent_l2536_253691


namespace NUMINAMATH_CALUDE_wednesday_thursday_miles_l2536_253652

/-- Represents the mileage reimbursement rate in dollars per mile -/
def reimbursement_rate : ℚ := 36 / 100

/-- Represents the total reimbursement amount in dollars -/
def total_reimbursement : ℚ := 36

/-- Represents the miles driven on Monday -/
def monday_miles : ℕ := 18

/-- Represents the miles driven on Tuesday -/
def tuesday_miles : ℕ := 26

/-- Represents the miles driven on Friday -/
def friday_miles : ℕ := 16

/-- Theorem stating that the miles driven on Wednesday and Thursday combined is 40 -/
theorem wednesday_thursday_miles : 
  (total_reimbursement / reimbursement_rate : ℚ) - 
  (monday_miles + tuesday_miles + friday_miles : ℚ) = 40 := by sorry

end NUMINAMATH_CALUDE_wednesday_thursday_miles_l2536_253652


namespace NUMINAMATH_CALUDE_line_perpendicular_condition_l2536_253646

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpLP : Line → Plane → Prop)
variable (perpPP : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_condition 
  (a b : Line) (α β : Plane) :
  subset a α → perpLP b β → ¬ parallel α β → perp a b :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_condition_l2536_253646


namespace NUMINAMATH_CALUDE_z_greater_than_w_by_50_percent_l2536_253681

theorem z_greater_than_w_by_50_percent 
  (w e y z : ℝ) 
  (hw : w = 0.6 * e) 
  (he : e = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  z = 1.5 * w := by
sorry

end NUMINAMATH_CALUDE_z_greater_than_w_by_50_percent_l2536_253681


namespace NUMINAMATH_CALUDE_pentagon_angle_problem_l2536_253603

def pentagon_largest_angle (P Q R S T : ℝ) : Prop :=
  -- Sum of angles in a pentagon is 540°
  P + Q + R + S + T = 540 ∧
  -- Given conditions
  P = 55 ∧
  Q = 120 ∧
  R = S ∧
  T = 2 * R + 20 ∧
  -- Largest angle is 192.5°
  max P (max Q (max R (max S T))) = 192.5

theorem pentagon_angle_problem :
  ∃ (P Q R S T : ℝ), pentagon_largest_angle P Q R S T := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_problem_l2536_253603


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2536_253693

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 48 → (∀ x y : ℕ, x * y = 48 → heart + club ≥ x + y) → heart + club = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2536_253693


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2536_253607

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a n > 0) →  -- All terms are positive
  a 1 = 3 →  -- First term is 3
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 + a 2 + a 3 = 21 →  -- Sum of first three terms is 21
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2536_253607


namespace NUMINAMATH_CALUDE_expected_value_is_three_l2536_253662

/-- Represents the game with two balls --/
structure TwoBallGame where
  /-- Probability of drawing ball 1 --/
  p₁ : ℝ
  /-- Probability of drawing ball 2 --/
  p₂ : ℝ
  /-- Points earned for drawing ball 1 --/
  score₁ : ℝ
  /-- Points earned for drawing ball 2 --/
  score₂ : ℝ
  /-- The probabilities sum to 1 --/
  prob_sum : p₁ + p₂ = 1
  /-- Probabilities are non-negative --/
  prob_nonneg : 0 ≤ p₁ ∧ 0 ≤ p₂

/-- The expected value of the total score in the game --/
noncomputable def expectedValue (game : TwoBallGame) : ℝ :=
  (game.score₂ * game.p₂) / (1 - game.p₁)

/-- Theorem stating that the expected value of the game is 3 --/
theorem expected_value_is_three (game : TwoBallGame)
  (h₁ : game.p₁ = 1/2)
  (h₂ : game.p₂ = 1/2)
  (h₃ : game.score₁ = 1)
  (h₄ : game.score₂ = 2) :
  expectedValue game = 3 := by
  sorry


end NUMINAMATH_CALUDE_expected_value_is_three_l2536_253662


namespace NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l2536_253686

theorem integral_exp_plus_2x_equals_e :
  ∫ x in (0:ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l2536_253686


namespace NUMINAMATH_CALUDE_expected_sales_after_price_change_l2536_253661

/-- Represents the relationship between price and sales of blenders -/
structure BlenderSales where
  price : ℝ
  units : ℝ

/-- The constant of proportionality for the inverse relationship -/
def k : ℝ := 15 * 500

/-- The inverse proportionality relationship between price and sales -/
def inverse_proportional (bs : BlenderSales) : Prop :=
  bs.price * bs.units = k

/-- The new price after discount -/
def new_price : ℝ := 1000 * (1 - 0.1)

/-- Theorem stating the expected sales under the new pricing scheme -/
theorem expected_sales_after_price_change 
  (initial : BlenderSales) 
  (h_initial : initial.price = 500 ∧ initial.units = 15) 
  (h_inverse : inverse_proportional initial) :
  ∃ (new : BlenderSales), 
    new.price = new_price ∧ 
    inverse_proportional new ∧ 
    (8 ≤ new.units ∧ new.units < 9) := by
  sorry

end NUMINAMATH_CALUDE_expected_sales_after_price_change_l2536_253661


namespace NUMINAMATH_CALUDE_min_value_theorem_l2536_253667

theorem min_value_theorem (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  ∀ ε > 0, ∃ x₀ y₀ z₀ : ℝ, 
    x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀^2 + y₀^2 + z₀^2 = 1 ∧
    (z₀ + 1)^2 / (2 * x₀ * y₀ * z₀) < 3 + 2 * Real.sqrt 2 + ε ∧
    (z + 1)^2 / (2 * x * y * z) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2536_253667


namespace NUMINAMATH_CALUDE_election_win_percentage_l2536_253613

/-- The required percentage to win an election -/
def required_percentage_to_win (total_votes : ℕ) (candidate_percentage : ℚ) (additional_votes_needed : ℕ) : ℚ :=
  (candidate_percentage * total_votes + additional_votes_needed) / total_votes

/-- Theorem: The required percentage to win the election is 51% -/
theorem election_win_percentage :
  required_percentage_to_win 6000 (1 / 100) 3000 = 51 / 100 := by
  sorry

end NUMINAMATH_CALUDE_election_win_percentage_l2536_253613


namespace NUMINAMATH_CALUDE_investment_of_a_l2536_253670

/-- Represents a partnership investment. -/
structure Partnership where
  investmentA : ℝ
  investmentB : ℝ
  investmentC : ℝ
  profitB : ℝ
  profitA : ℝ
  months : ℕ

/-- Theorem stating the investment of partner A given the conditions. -/
theorem investment_of_a (p : Partnership) 
  (hB : p.investmentB = 21000)
  (hprofitB : p.profitB = 1540)
  (hprofitA : p.profitA = 1100)
  (hmonths : p.months = 8)
  (h_profit_prop : p.profitA / p.profitB = p.investmentA / p.investmentB) :
  p.investmentA = 15000 :=
sorry

end NUMINAMATH_CALUDE_investment_of_a_l2536_253670


namespace NUMINAMATH_CALUDE_truck_and_goods_problem_l2536_253621

theorem truck_and_goods_problem (x : ℕ) (total_goods : ℕ) :
  (3 * x + 5 = total_goods) →  -- Condition 1
  (4 * (x - 5) = total_goods) →  -- Condition 2
  (x = 25 ∧ total_goods = 80) :=  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_truck_and_goods_problem_l2536_253621


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_factorization_l2536_253699

theorem perfect_square_trinomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 = (x + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_factorization_l2536_253699


namespace NUMINAMATH_CALUDE_system_solution_l2536_253698

theorem system_solution (a b c : ℝ) :
  ∃! (x y z : ℝ),
    (x + a * y + a^2 * z + a^3 = 0) ∧
    (x + b * y + b^2 * z + b^3 = 0) ∧
    (x + c * y + c^2 * z + c^3 = 0) ∧
    (x = -a * b * c) ∧
    (y = a * b + b * c + c * a) ∧
    (z = -(a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2536_253698


namespace NUMINAMATH_CALUDE_hyperbola_focus_coincides_with_parabola_focus_l2536_253665

/-- The focus of a parabola y^2 = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (5 * x^2) / 3 - (5 * y^2) / 2 = 1

/-- The right focus of the hyperbola -/
def hyperbola_right_focus : ℝ × ℝ := (1, 0)

/-- Theorem stating that the right focus of the hyperbola coincides with the focus of the parabola -/
theorem hyperbola_focus_coincides_with_parabola_focus :
  hyperbola_right_focus = parabola_focus :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_coincides_with_parabola_focus_l2536_253665


namespace NUMINAMATH_CALUDE_unique_solution_is_twelve_l2536_253656

/-- Definition of the ♣ operation -/
def clubsuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating that 12 is the unique solution to A ♣ 7 = 76 -/
theorem unique_solution_is_twelve :
  ∃! A : ℝ, clubsuit A 7 = 76 ∧ A = 12 := by sorry

end NUMINAMATH_CALUDE_unique_solution_is_twelve_l2536_253656


namespace NUMINAMATH_CALUDE_one_twelfth_day_in_minutes_l2536_253620

/-- Proves that 1/12 of a day is equal to 120 minutes -/
theorem one_twelfth_day_in_minutes : 
  (∀ (hours_per_day minutes_per_hour : ℕ), 
    hours_per_day = 24 → 
    minutes_per_hour = 60 → 
    (1 / 12 : ℚ) * (hours_per_day * minutes_per_hour) = 120) := by
  sorry

#check one_twelfth_day_in_minutes

end NUMINAMATH_CALUDE_one_twelfth_day_in_minutes_l2536_253620


namespace NUMINAMATH_CALUDE_antifreeze_concentration_l2536_253600

/-- Proves that the concentration of the certain antifreeze is 100% -/
theorem antifreeze_concentration
  (total_volume : ℝ)
  (final_concentration : ℝ)
  (certain_volume : ℝ)
  (other_concentration : ℝ)
  (h1 : total_volume = 55)
  (h2 : final_concentration = 0.20)
  (h3 : certain_volume = 6.11)
  (h4 : other_concentration = 0.10)
  : ∃ (certain_concentration : ℝ),
    certain_concentration = 1 ∧
    certain_volume * certain_concentration +
    (total_volume - certain_volume) * other_concentration =
    total_volume * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_antifreeze_concentration_l2536_253600


namespace NUMINAMATH_CALUDE_next_interval_is_two_point_five_l2536_253668

/-- Represents a decreasing geometric sequence of comet appearance intervals. -/
structure CometIntervals where
  /-- The common ratio of the geometric sequence. -/
  ratio : ℝ
  /-- Constraint that the ratio is positive and less than 1. -/
  ratio_pos : 0 < ratio
  ratio_lt_one : ratio < 1
  /-- The first term of the sequence (latest interval). -/
  first_term : ℝ
  /-- Constraint that the first term is positive. -/
  first_term_pos : 0 < first_term

/-- The three latest intervals satisfy the cubic equation. -/
def satisfies_cubic_equation (intervals : CometIntervals) : Prop :=
  ∃ c : ℝ, 
    let t₁ := intervals.first_term
    let t₂ := t₁ * intervals.ratio
    let t₃ := t₂ * intervals.ratio
    t₁^3 - c * t₁^2 + 350 * t₁ - 1000 = 0 ∧
    t₂^3 - c * t₂^2 + 350 * t₂ - 1000 = 0 ∧
    t₃^3 - c * t₃^2 + 350 * t₃ - 1000 = 0

/-- The theorem stating that the next interval will be 2.5 years. -/
theorem next_interval_is_two_point_five (intervals : CometIntervals) 
  (h : satisfies_cubic_equation intervals) : 
  intervals.first_term * intervals.ratio^3 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_next_interval_is_two_point_five_l2536_253668


namespace NUMINAMATH_CALUDE_rectangle_y_value_l2536_253678

theorem rectangle_y_value (y : ℝ) : 
  y > 0 →  -- y is positive
  (5 - (-3)) * (y - 2) = 64 →  -- area of rectangle is 64
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l2536_253678


namespace NUMINAMATH_CALUDE_total_cost_is_2075_l2536_253625

def grapes_quantity : ℕ := 8
def grapes_price : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_price : ℕ := 55
def apples_quantity : ℕ := 4
def apples_price : ℕ := 40
def oranges_quantity : ℕ := 6
def oranges_price : ℕ := 30
def pineapples_quantity : ℕ := 2
def pineapples_price : ℕ := 90
def cherries_quantity : ℕ := 5
def cherries_price : ℕ := 100

def total_cost : ℕ := 
  grapes_quantity * grapes_price + 
  mangoes_quantity * mangoes_price + 
  apples_quantity * apples_price + 
  oranges_quantity * oranges_price + 
  pineapples_quantity * pineapples_price + 
  cherries_quantity * cherries_price

theorem total_cost_is_2075 : total_cost = 2075 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_2075_l2536_253625


namespace NUMINAMATH_CALUDE_tangent_points_tangent_parallel_points_coordinates_of_tangent_points_l2536_253615

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_points (x : ℝ) : 
  (3 * x^2 + 1 = 4) ↔ (x = 1 ∨ x = -1) :=
sorry

theorem tangent_parallel_points : 
  {x : ℝ | 3 * x^2 + 1 = 4} = {1, -1} :=
sorry

theorem coordinates_of_tangent_points : 
  {p : ℝ × ℝ | p.1 ∈ {x : ℝ | 3 * x^2 + 1 = 4} ∧ p.2 = f p.1} = {(1, 0), (-1, -4)} :=
sorry

end NUMINAMATH_CALUDE_tangent_points_tangent_parallel_points_coordinates_of_tangent_points_l2536_253615


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2536_253680

/-- The eccentricity of an ellipse with equation x²/16 + y²/25 = 1 is 3/5 -/
theorem ellipse_eccentricity : 
  let a : ℝ := 5
  let b : ℝ := 4
  let c : ℝ := Real.sqrt (a^2 - b^2)
  c / a = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2536_253680


namespace NUMINAMATH_CALUDE_nilpotent_in_finite_ring_l2536_253644

/-- A ring with exactly n elements -/
class FiniteRing (A : Type) extends Ring A where
  card : ℕ
  finite : Fintype A
  card_eq : Fintype.card A = card

theorem nilpotent_in_finite_ring {n m : ℕ} {A : Type} [FiniteRing A] (hn : n ≥ 2) (hm : m ≥ 0) 
  (h_card : (FiniteRing.card A) = n) (a : A) 
  (h_inv : ∀ k ∈ Finset.range (n - 1), k ≥ m + 1 → IsUnit (1 - a ^ (k + 1))) :
  ∃ k : ℕ, a ^ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_in_finite_ring_l2536_253644


namespace NUMINAMATH_CALUDE_road_repair_group_size_l2536_253632

/-- The number of persons in the first group -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 10

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

theorem road_repair_group_size :
  first_group_size * first_group_days * first_group_hours_per_day =
  second_group_size * second_group_days * second_group_hours_per_day :=
by sorry

end NUMINAMATH_CALUDE_road_repair_group_size_l2536_253632


namespace NUMINAMATH_CALUDE_solution_value_l2536_253672

/-- A system of two linear equations in two variables -/
structure LinearSystem :=
  (a b c : ℝ)
  (d e f : ℝ)

/-- The condition for a linear system to not have a unique solution -/
def noUniqueSolution (sys : LinearSystem) : Prop :=
  sys.a * sys.e = sys.b * sys.d ∧ sys.a * sys.f = sys.c * sys.d

/-- The theorem stating that if the given system doesn't have a unique solution, then d = 40 -/
theorem solution_value (k : ℝ) :
  let sys : LinearSystem := ⟨12, 16, d, k, 12, 30⟩
  noUniqueSolution sys → d = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_solution_value_l2536_253672


namespace NUMINAMATH_CALUDE_no_fixed_points_iff_negative_discriminant_l2536_253683

/-- A function f: ℝ → ℝ has a fixed point if there exists an x₀ ∈ ℝ such that f(x₀) = x₀ -/
def has_fixed_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = x₀

/-- The quadratic function f(x) = x² + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The discriminant of the equation x² + (a-1)x + 1 = 0 -/
def discriminant (a : ℝ) : ℝ := (a-1)^2 - 4

theorem no_fixed_points_iff_negative_discriminant (a : ℝ) :
  ¬(has_fixed_point (f a)) ↔ discriminant a < 0 :=
sorry

end NUMINAMATH_CALUDE_no_fixed_points_iff_negative_discriminant_l2536_253683


namespace NUMINAMATH_CALUDE_point_M_coordinates_l2536_253624

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ := |p.y|

/-- The distance of a point from the y-axis -/
def distanceFromYAxis (p : Point) : ℝ := |p.x|

theorem point_M_coordinates (a : ℝ) :
  let M : Point := ⟨2 - a, 3 * a + 6⟩
  distanceFromXAxis M = distanceFromYAxis M →
  M = ⟨3, 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l2536_253624


namespace NUMINAMATH_CALUDE_find_m_value_l2536_253643

def A : Set ℝ := {-1, 1, 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem find_m_value : ∀ m : ℝ, B m ⊆ A → m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l2536_253643


namespace NUMINAMATH_CALUDE_inversion_of_line_l2536_253610

/-- A circle in a plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- A line in a plane -/
structure Line :=
  (point : ℝ × ℝ)
  (direction : ℝ × ℝ)

/-- The result of inverting a line with respect to a circle -/
inductive InversionResult
  | SameLine : InversionResult
  | Circle : (ℝ × ℝ) → ℝ → InversionResult

/-- Inversion of a line with respect to a circle -/
def invert (l : Line) (c : Circle) : InversionResult :=
  sorry

/-- Theorem: The image of a line under inversion is either the line itself or a circle passing through the center of inversion -/
theorem inversion_of_line (l : Line) (c : Circle) :
  (invert l c = InversionResult.SameLine ∧ l.point = c.center) ∨
  (∃ center radius, invert l c = InversionResult.Circle center radius ∧ center = c.center) :=
sorry

end NUMINAMATH_CALUDE_inversion_of_line_l2536_253610


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l2536_253623

/-- Given two vectors a and b that are parallel, prove that sin²α + 2sinα*cosα = 3/2 -/
theorem parallel_vectors_trig_identity (α : ℝ) : 
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (Real.sin (α - π/3), Real.cos α + π/3)
  (∃ k : ℝ, b = k • a) →
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l2536_253623


namespace NUMINAMATH_CALUDE_increasing_function_t_bound_l2536_253642

def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 3

theorem increasing_function_t_bound (t : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f t x₁ < f t x₂) →
  t ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_t_bound_l2536_253642


namespace NUMINAMATH_CALUDE_simplify_expression_l2536_253659

theorem simplify_expression (p : ℝ) : 
  ((5 * p + 1) - 2 * p * 4) * 3 + (4 - 1 / 3) * (6 * p - 9) = 13 * p - 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2536_253659


namespace NUMINAMATH_CALUDE_average_price_decrease_l2536_253634

theorem average_price_decrease (original_price final_price : ℝ) 
  (h1 : original_price = 10000)
  (h2 : final_price = 6400)
  (h3 : final_price = original_price * (1 - x)^2)
  (h4 : x > 0 ∧ x < 1) : 
  x = 0.2 :=
sorry

end NUMINAMATH_CALUDE_average_price_decrease_l2536_253634


namespace NUMINAMATH_CALUDE_f_deriv_at_one_l2536_253631

-- Define a differentiable function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the condition f(x) = 2xf'(1) + ln x
axiom f_condition (x : ℝ) : f x = 2 * x * (deriv f 1) + Real.log x

-- Theorem statement
theorem f_deriv_at_one : deriv f 1 = -1 := by sorry

end NUMINAMATH_CALUDE_f_deriv_at_one_l2536_253631


namespace NUMINAMATH_CALUDE_marys_tuesday_payment_l2536_253639

theorem marys_tuesday_payment 
  (credit_limit : ℕ) 
  (thursday_payment : ℕ) 
  (remaining_payment : ℕ) : 
  credit_limit - (thursday_payment + remaining_payment) = 15 :=
by
  sorry

#check marys_tuesday_payment 100 23 62

end NUMINAMATH_CALUDE_marys_tuesday_payment_l2536_253639


namespace NUMINAMATH_CALUDE_four_digit_square_modification_l2536_253612

/-- A function that returns the first digit of a natural number -/
def firstDigit (n : ℕ) : ℕ :=
  if n < 10 then n else firstDigit (n / 10)

/-- A function that returns the last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- A function that modifies a number by decreasing its first digit by 3
    and increasing its last digit by 3 -/
def modifyNumber (n : ℕ) : ℕ :=
  n - 3000 + 3

theorem four_digit_square_modification :
  ∃ n : ℕ,
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    ∃ a : ℕ, n = a^2 ∧      -- perfect square
    ∃ b : ℕ, modifyNumber n = b^2  -- modified number is also a perfect square
  := by sorry

end NUMINAMATH_CALUDE_four_digit_square_modification_l2536_253612


namespace NUMINAMATH_CALUDE_gcd_problem_l2536_253663

def gcd_notation (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_problem (b : ℕ) : 
  (gcd_notation (gcd_notation 20 16) (18 * b) = 2) → b = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2536_253663


namespace NUMINAMATH_CALUDE_four_letter_sets_count_l2536_253629

/-- The number of letters available (A through H) -/
def num_letters : ℕ := 8

/-- The number of letters in each set of initials -/
def set_size : ℕ := 4

/-- The total number of possible four-letter sets of initials using letters A through H -/
def total_sets : ℕ := num_letters ^ set_size

theorem four_letter_sets_count : total_sets = 4096 := by
  sorry

end NUMINAMATH_CALUDE_four_letter_sets_count_l2536_253629


namespace NUMINAMATH_CALUDE_parabola_directrix_l2536_253696

/-- For a parabola with equation y = 2x^2, its directrix has the equation y = -1/8 -/
theorem parabola_directrix (x y : ℝ) :
  y = 2 * x^2 → (∃ (k : ℝ), y = k ∧ k = -1/8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2536_253696


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l2536_253637

theorem set_equality_implies_sum_of_powers (x y : ℝ) : 
  ({x, x * y, x + y} : Set ℝ) = ({0, |x|, y} : Set ℝ) → x^2018 + y^2018 = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l2536_253637
