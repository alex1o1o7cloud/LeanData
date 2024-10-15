import Mathlib

namespace NUMINAMATH_CALUDE_f_inequality_implies_b_geq_one_l3639_363963

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp (-x)

-- State the theorem
theorem f_inequality_implies_b_geq_one :
  ∀ b : ℝ,
  (∀ a : ℝ, a ≤ 0 → ∀ x : ℝ, x ≥ 0 → f a x ≤ b * Real.log (x + 1)) →
  b ≥ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_f_inequality_implies_b_geq_one_l3639_363963


namespace NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l3639_363947

theorem scientific_notation_of_56_99_million :
  (56.99 * 1000000 : ℝ) = 5.699 * (10 ^ 7) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l3639_363947


namespace NUMINAMATH_CALUDE_largest_difference_even_odd_three_digit_l3639_363986

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def all_even_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 0

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

theorem largest_difference_even_odd_three_digit : 
  ∃ (a b : ℕ), 
    is_three_digit_number a ∧
    is_three_digit_number b ∧
    has_distinct_digits a ∧
    has_distinct_digits b ∧
    all_even_digits a ∧
    all_odd_digits b ∧
    (∀ (x y : ℕ), 
      is_three_digit_number x ∧
      is_three_digit_number y ∧
      has_distinct_digits x ∧
      has_distinct_digits y ∧
      all_even_digits x ∧
      all_odd_digits y →
      x - y ≤ a - b) ∧
    a - b = 729 :=
sorry

end NUMINAMATH_CALUDE_largest_difference_even_odd_three_digit_l3639_363986


namespace NUMINAMATH_CALUDE_standard_equation_of_M_no_B_on_circle_and_M_l3639_363990

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the focus F₁ and vertex C
def F₁ : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (-2, 0)

-- Theorem for part I
theorem standard_equation_of_M : 
  ∀ x y : ℝ, ellipse_M x y ↔ x^2 / 4 + y^2 / 3 = 1 :=
sorry

-- Theorem for part II
theorem no_B_on_circle_and_M :
  ¬ ∃ x₀ y₀ : ℝ, 
    ellipse_M x₀ y₀ ∧ 
    -2 < x₀ ∧ x₀ < 2 ∧
    (x₀ + 1)^2 + y₀^2 = (x₀ + 2)^2 + y₀^2 :=
sorry

end NUMINAMATH_CALUDE_standard_equation_of_M_no_B_on_circle_and_M_l3639_363990


namespace NUMINAMATH_CALUDE_max_cities_is_four_l3639_363904

/-- Represents the modes of transportation --/
inductive TransportMode
| Bus
| Train
| Airplane

/-- Represents a city in the country --/
structure City where
  id : Nat

/-- Represents the transportation network of the country --/
structure TransportNetwork where
  cities : List City
  connections : List City → List City → TransportMode → Prop

/-- Checks if the network satisfies the condition that no city is serviced by all three types of transportation --/
def noTripleService (network : TransportNetwork) : Prop :=
  ∀ c : City, c ∈ network.cities →
    ¬(∃ (c1 c2 c3 : City), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
      network.connections [c, c1] [c, c1] TransportMode.Bus ∧
      network.connections [c, c2] [c, c2] TransportMode.Train ∧
      network.connections [c, c3] [c, c3] TransportMode.Airplane)

/-- Checks if the network satisfies the condition that no three cities are connected by the same mode of transportation --/
def noTripleConnection (network : TransportNetwork) : Prop :=
  ∀ mode : TransportMode, ¬(∃ (c1 c2 c3 : City), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    network.connections [c1, c2] [c1, c2] mode ∧
    network.connections [c2, c3] [c2, c3] mode ∧
    network.connections [c1, c3] [c1, c3] mode)

/-- The main theorem stating that the maximum number of cities is 4 --/
theorem max_cities_is_four :
  ∀ (network : TransportNetwork),
    (∀ (c1 c2 : City), c1 ≠ c2 → ∃ (mode : TransportMode), network.connections [c1, c2] [c1, c2] mode) →
    noTripleService network →
    noTripleConnection network →
    List.length network.cities ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_cities_is_four_l3639_363904


namespace NUMINAMATH_CALUDE_gcd_lcm_42_30_l3639_363916

theorem gcd_lcm_42_30 :
  (Nat.gcd 42 30 = 6) ∧ (Nat.lcm 42 30 = 210) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_42_30_l3639_363916


namespace NUMINAMATH_CALUDE_appliance_price_ratio_l3639_363970

theorem appliance_price_ratio : 
  ∀ (c p q : ℝ), 
  p = 0.8 * c →  -- 20% loss
  q = 1.25 * c → -- 25% profit
  q / p = 25 / 16 := by
sorry

end NUMINAMATH_CALUDE_appliance_price_ratio_l3639_363970


namespace NUMINAMATH_CALUDE_solve_for_d_l3639_363959

theorem solve_for_d (n c b d : ℝ) (h : n = (d * c * b) / (c - d)) :
  d = (n * c) / (c * b + n) :=
by sorry

end NUMINAMATH_CALUDE_solve_for_d_l3639_363959


namespace NUMINAMATH_CALUDE_candy_probability_l3639_363936

/-- Represents the number of red candies in the jar -/
def red_candies : ℕ := 15

/-- Represents the number of blue candies in the jar -/
def blue_candies : ℕ := 20

/-- Represents the total number of candies in the jar -/
def total_candies : ℕ := red_candies + blue_candies

/-- Represents the number of candies each person picks -/
def picks : ℕ := 3

/-- The probability of Terry and Mary getting the same color combination -/
def same_color_probability : ℚ := 243 / 6825

theorem candy_probability : 
  let terry_red_prob := (red_candies.choose picks : ℚ) / (total_candies.choose picks)
  let terry_blue_prob := (blue_candies.choose picks : ℚ) / (total_candies.choose picks)
  let mary_red_prob := ((red_candies - picks).choose picks : ℚ) / ((total_candies - picks).choose picks)
  let mary_blue_prob := ((blue_candies - picks).choose picks : ℚ) / ((total_candies - picks).choose picks)
  terry_red_prob * mary_red_prob + terry_blue_prob * mary_blue_prob = same_color_probability :=
sorry

end NUMINAMATH_CALUDE_candy_probability_l3639_363936


namespace NUMINAMATH_CALUDE_yellow_paint_calculation_l3639_363974

/-- Given a ratio of red:yellow:blue paint and the amount of blue paint,
    calculate the amount of yellow paint required. -/
def yellow_paint_amount (red yellow blue : ℚ) (blue_amount : ℚ) : ℚ :=
  (yellow / blue) * blue_amount

/-- Prove that for the given ratio and blue paint amount, 
    the required yellow paint amount is 9 quarts. -/
theorem yellow_paint_calculation :
  let red : ℚ := 5
  let yellow : ℚ := 3
  let blue : ℚ := 7
  let blue_amount : ℚ := 21
  yellow_paint_amount red yellow blue blue_amount = 9 := by
  sorry

#eval yellow_paint_amount 5 3 7 21

end NUMINAMATH_CALUDE_yellow_paint_calculation_l3639_363974


namespace NUMINAMATH_CALUDE_kaplan_bobby_slice_ratio_l3639_363905

/-- Represents the number of pizzas Bobby has -/
def bobby_pizzas : ℕ := 2

/-- Represents the number of slices per pizza -/
def slices_per_pizza : ℕ := 6

/-- Represents the number of slices Mrs. Kaplan has -/
def kaplan_slices : ℕ := 3

/-- Calculates the total number of slices Bobby has -/
def bobby_slices : ℕ := bobby_pizzas * slices_per_pizza

/-- Represents the ratio of Mrs. Kaplan's slices to Bobby's slices -/
def slice_ratio : Rat := kaplan_slices / bobby_slices

theorem kaplan_bobby_slice_ratio :
  slice_ratio = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_kaplan_bobby_slice_ratio_l3639_363905


namespace NUMINAMATH_CALUDE_investment_result_l3639_363937

/-- Given a total investment split between two interest rates, calculates the total investment with interest after one year. -/
def total_investment_with_interest (total_investment : ℝ) (amount_at_low_rate : ℝ) (low_rate : ℝ) (high_rate : ℝ) : ℝ :=
  let amount_at_high_rate := total_investment - amount_at_low_rate
  let interest_low := amount_at_low_rate * low_rate
  let interest_high := amount_at_high_rate * high_rate
  total_investment + interest_low + interest_high

/-- Theorem stating that given the specific investment conditions, the total investment with interest is $1,046.00 -/
theorem investment_result : 
  let total_investment := 1000
  let amount_at_low_rate := 699.99
  let low_rate := 0.04
  let high_rate := 0.06
  (total_investment_with_interest total_investment amount_at_low_rate low_rate high_rate) = 1046 := by
sorry

end NUMINAMATH_CALUDE_investment_result_l3639_363937


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3639_363969

theorem polynomial_multiplication (a b : ℝ) : (2*a + 3*b) * (2*a - b) = 4*a^2 + 4*a*b - 3*b^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3639_363969


namespace NUMINAMATH_CALUDE_platform_length_platform_length_is_150_l3639_363957

/-- Given a train passing a platform and a man, calculate the platform length -/
theorem platform_length (train_speed : Real) (platform_time : Real) (man_time : Real) : Real :=
  let train_speed_ms := train_speed * 1000 / 3600
  let train_length := train_speed_ms * man_time
  let platform_length := train_speed_ms * platform_time - train_length
  platform_length

/-- Prove that the platform length is 150 meters given the specified conditions -/
theorem platform_length_is_150 :
  platform_length 54 30 20 = 150 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_is_150_l3639_363957


namespace NUMINAMATH_CALUDE_pudding_distribution_l3639_363909

theorem pudding_distribution (pudding_cups : ℕ) (students : ℕ) 
  (h1 : pudding_cups = 4752) (h2 : students = 3019) : 
  let additional_cups := (students * ((pudding_cups + students - 1) / students)) - pudding_cups
  additional_cups = 1286 := by
sorry

end NUMINAMATH_CALUDE_pudding_distribution_l3639_363909


namespace NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l3639_363913

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    with eccentricity e = √7/2, and a point P on the right branch of the hyperbola
    such that PF₂ ⊥ F₁F₂ and PF₂ = 9/2, prove that the length of the conjugate axis
    is 6√3. -/
theorem hyperbola_conjugate_axis_length
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (he : Real.sqrt 7 / 2 = Real.sqrt (1 + b^2 / a^2))
  (hP : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ x > 0)
  (hPF2 : b^2 / a = 9 / 2) :
  2 * b = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l3639_363913


namespace NUMINAMATH_CALUDE_large_pots_delivered_l3639_363921

/-- The number of boxes delivered -/
def num_boxes : ℕ := 32

/-- The number of small pots in each box -/
def small_pots_per_box : ℕ := 36

/-- The number of large pots in each box -/
def large_pots_per_box : ℕ := 12

/-- The total number of large pots delivered -/
def total_large_pots : ℕ := num_boxes * large_pots_per_box

/-- The number of boxes used for comparison -/
def comparison_boxes : ℕ := 8

theorem large_pots_delivered :
  total_large_pots = 384 ∧
  total_large_pots = comparison_boxes * (small_pots_per_box + large_pots_per_box) :=
by sorry

end NUMINAMATH_CALUDE_large_pots_delivered_l3639_363921


namespace NUMINAMATH_CALUDE_potassium_bromate_weight_l3639_363941

/-- The atomic weight of Potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- The atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Potassium atoms in Potassium Bromate -/
def num_K : ℕ := 1

/-- The number of Bromine atoms in Potassium Bromate -/
def num_Br : ℕ := 1

/-- The number of Oxygen atoms in Potassium Bromate -/
def num_O : ℕ := 3

/-- The molecular weight of Potassium Bromate in g/mol -/
def molecular_weight_KBrO3 : ℝ :=
  num_K * atomic_weight_K + num_Br * atomic_weight_Br + num_O * atomic_weight_O

/-- Theorem stating that the molecular weight of Potassium Bromate is 167.00 g/mol -/
theorem potassium_bromate_weight : molecular_weight_KBrO3 = 167.00 := by
  sorry

end NUMINAMATH_CALUDE_potassium_bromate_weight_l3639_363941


namespace NUMINAMATH_CALUDE_lower_parallel_length_l3639_363997

/-- A triangle with a base of 20 inches and two parallel lines dividing it into four equal areas -/
structure EqualAreaTriangle where
  /-- The base of the triangle -/
  base : ℝ
  /-- The length of the parallel line closer to the base -/
  lower_parallel : ℝ
  /-- The base is 20 inches -/
  base_length : base = 20
  /-- The parallel lines divide the triangle into four equal areas -/
  equal_areas : lower_parallel^2 / base^2 = 1/4

/-- The length of the parallel line closer to the base is 10 inches -/
theorem lower_parallel_length (t : EqualAreaTriangle) : t.lower_parallel = 10 := by
  sorry

end NUMINAMATH_CALUDE_lower_parallel_length_l3639_363997


namespace NUMINAMATH_CALUDE_a_minus_c_value_l3639_363991

/-- Given that A = 742, B = A + 397, and B = C + 693, prove that A - C = 296 -/
theorem a_minus_c_value (A B C : ℤ) 
  (h1 : A = 742)
  (h2 : B = A + 397)
  (h3 : B = C + 693) : 
  A - C = 296 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_value_l3639_363991


namespace NUMINAMATH_CALUDE_waiter_net_earnings_waiter_earnings_result_l3639_363931

/-- Calculates the waiter's net earnings from tips after commission --/
theorem waiter_net_earnings (customers : Nat) 
  (tipping_customers : Nat)
  (bill1 bill2 bill3 bill4 : ℝ)
  (tip_percent1 tip_percent2 tip_percent3 tip_percent4 : ℝ)
  (commission_rate : ℝ) : ℝ :=
  let total_tips := 
    bill1 * tip_percent1 + 
    bill2 * tip_percent2 + 
    bill3 * tip_percent3 + 
    bill4 * tip_percent4
  let commission := total_tips * commission_rate
  let net_earnings := total_tips - commission
  net_earnings

/-- The waiter's net earnings are approximately $16.82 --/
theorem waiter_earnings_result : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |waiter_net_earnings 9 4 25 22 35 30 0.15 0.18 0.20 0.10 0.05 - 16.82| < ε :=
sorry

end NUMINAMATH_CALUDE_waiter_net_earnings_waiter_earnings_result_l3639_363931


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_45_l3639_363965

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the factorization of 45
axiom factorization_of_45 : 45 = 3 * 3 * 5

-- Theorem statement
theorem no_primes_divisible_by_45 :
  ∀ p : ℕ, is_prime p → ¬(45 ∣ p) :=
sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_45_l3639_363965


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3639_363934

def polynomial_remainder_problem (p : ℝ → ℝ) (r : ℝ → ℝ) : Prop :=
  (p (-1) = 2) ∧ 
  (p 3 = -2) ∧ 
  (p (-4) = 5) ∧ 
  (∃ q : ℝ → ℝ, ∀ x, p x = (x + 1) * (x - 3) * (x + 4) * q x + r x) ∧
  (r (-5) = 6)

theorem polynomial_remainder_theorem :
  ∃ p r : ℝ → ℝ, polynomial_remainder_problem p r :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3639_363934


namespace NUMINAMATH_CALUDE_loss_recording_l3639_363940

/-- Represents the recording of a financial transaction -/
def record (amount : Int) : Int := amount

/-- A profit of $300 is recorded as $+300 -/
axiom profit_recording : record 300 = 300

/-- Theorem: If a profit of $300 is recorded as $+300, then a loss of $300 should be recorded as $-300 -/
theorem loss_recording : record (-300) = -300 := by
  sorry

end NUMINAMATH_CALUDE_loss_recording_l3639_363940


namespace NUMINAMATH_CALUDE_expected_value_is_correct_l3639_363933

def number_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_colored (n : ℕ) : Bool := sorry

def is_red (n : ℕ) : Bool := sorry

def is_blue (n : ℕ) : Bool := sorry

def probability_red : ℚ := 1/2

def probability_blue : ℚ := 1/2

def is_sum_of_red_and_blue (n : ℕ) : Bool := sorry

def expected_value : ℚ := sorry

theorem expected_value_is_correct : expected_value = 423/32 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_correct_l3639_363933


namespace NUMINAMATH_CALUDE_expression_simplification_l3639_363924

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3639_363924


namespace NUMINAMATH_CALUDE_plotted_points_form_circle_l3639_363994

theorem plotted_points_form_circle :
  ∀ (x y : ℝ), (∃ t : ℝ, x = Real.cos t ∧ y = Real.sin t) →
  x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_plotted_points_form_circle_l3639_363994


namespace NUMINAMATH_CALUDE_count_numbers_mod_three_eq_one_l3639_363945

theorem count_numbers_mod_three_eq_one (n : ℕ) : 
  (Finset.filter (fun x => x % 3 = 1) (Finset.range 50)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_mod_three_eq_one_l3639_363945


namespace NUMINAMATH_CALUDE_point_q_location_l3639_363984

/-- Given four points O, A, B, C on a straight line and a point Q on AB, prove that Q's position relative to O is 2a + 2.5b -/
theorem point_q_location (a b c : ℝ) (O A B C Q : ℝ) : 
  O < A ∧ A < B ∧ B < C ∧  -- Points are in order
  A - O = 2 * a ∧  -- OA = 2a
  B - A = 3 * b ∧  -- AB = 3b
  C - B = 4 * c ∧  -- BC = 4c
  A ≤ Q ∧ Q ≤ B ∧  -- Q is on segment AB
  (Q - A) / (B - Q) = 3 / 1  -- AQ:QB = 3:1
  → Q - O = 2 * a + 2.5 * b :=
by sorry

end NUMINAMATH_CALUDE_point_q_location_l3639_363984


namespace NUMINAMATH_CALUDE_tire_promotion_price_l3639_363971

/-- The regular price of a tire under the given promotion -/
def regular_price : ℝ := 105

/-- The total cost of five tires under the promotion -/
def total_cost : ℝ := 421

/-- The promotion: buy four tires at regular price, get the fifth for $1 -/
theorem tire_promotion_price : 
  4 * regular_price + 1 = total_cost := by sorry

end NUMINAMATH_CALUDE_tire_promotion_price_l3639_363971


namespace NUMINAMATH_CALUDE_onions_count_prove_onions_count_l3639_363967

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onion_difference : ℕ := 5200

theorem onions_count : ℕ :=
  (tomatoes + corn) - onion_difference

theorem prove_onions_count : onions_count = 985 := by
  sorry

end NUMINAMATH_CALUDE_onions_count_prove_onions_count_l3639_363967


namespace NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l3639_363993

/-- A positive geometric progression -/
def is_positive_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- An arithmetic progression -/
def is_arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n, b (n + 1) = b n + d

/-- The main theorem -/
theorem geometric_arithmetic_inequality
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_geo : is_positive_geometric_progression a)
  (h_arith : is_arithmetic_progression b)
  (h_eq : a 6 = b 7) :
  a 3 + a 9 ≥ b 4 + b 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l3639_363993


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3639_363900

theorem solve_exponential_equation :
  ∃ x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x = (256 : ℝ)^3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3639_363900


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3639_363985

theorem quadratic_root_property : ∀ m n : ℝ,
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ x = m ∨ x = n) →
  m + n - m*n = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3639_363985


namespace NUMINAMATH_CALUDE_max_prime_factor_of_arithmetic_sequence_number_l3639_363946

/-- A 3-digit decimal number with digits forming an arithmetic sequence -/
def ArithmeticSequenceNumber (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = 100 * a + 10 * b + c ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    b = a + d ∧
    c = a + 2 * d

theorem max_prime_factor_of_arithmetic_sequence_number :
  ∀ n : ℕ, ArithmeticSequenceNumber n →
    (∀ p : ℕ, Nat.Prime p → p ∣ n → p ≤ 317) ∧
    (∃ m : ℕ, ArithmeticSequenceNumber m ∧ ∃ p : ℕ, Nat.Prime p ∧ p ∣ m ∧ p = 317) :=
by sorry

end NUMINAMATH_CALUDE_max_prime_factor_of_arithmetic_sequence_number_l3639_363946


namespace NUMINAMATH_CALUDE_solve_for_t_l3639_363920

theorem solve_for_t (s t : ℚ) 
  (eq1 : 15 * s + 7 * t = 236)
  (eq2 : t = 2 * s + 1) : 
  t = 487 / 29 := by
sorry

end NUMINAMATH_CALUDE_solve_for_t_l3639_363920


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l3639_363932

theorem smallest_of_five_consecutive_even_numbers (x : ℤ) : 
  (∀ i : ℕ, i < 5 → 2 ∣ (x + 2*i)) →  -- x and the next 4 numbers are even
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 200) →  -- sum is 200
  x = 36 :=  -- smallest number is 36
by
  sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l3639_363932


namespace NUMINAMATH_CALUDE_tenth_term_geometric_sequence_l3639_363944

theorem tenth_term_geometric_sequence :
  let a : ℚ := 5  -- First term
  let r : ℚ := 4/3  -- Common ratio
  let n : ℕ := 10  -- Term number we're looking for
  let a_n : ℚ := a * r^(n - 1)  -- Formula for nth term of geometric sequence
  a_n = 1310720/19683 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_geometric_sequence_l3639_363944


namespace NUMINAMATH_CALUDE_vanilla_jelly_beans_count_l3639_363999

theorem vanilla_jelly_beans_count :
  ∀ (vanilla grape : ℕ),
    grape = 5 * vanilla + 50 →
    vanilla + grape = 770 →
    vanilla = 120 := by
  sorry

end NUMINAMATH_CALUDE_vanilla_jelly_beans_count_l3639_363999


namespace NUMINAMATH_CALUDE_correlation_coefficient_properties_l3639_363951

/-- Linear correlation coefficient between two variables -/
def linear_correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- Positive correlation between two variables -/
def positively_correlated (x y : ℝ → ℝ) : Prop := sorry

/-- Perfect linear relationship between two variables -/
def perfect_linear_relationship (x y : ℝ → ℝ) : Prop := sorry

theorem correlation_coefficient_properties
  (x y : ℝ → ℝ) (r : ℝ) (h : r = linear_correlation_coefficient x y) :
  ((r > 0 → positively_correlated x y) ∧
   (r = 1 ∨ r = -1 → perfect_linear_relationship x y)) := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_properties_l3639_363951


namespace NUMINAMATH_CALUDE_f_composition_negative_three_eq_pi_l3639_363901

/-- Piecewise function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2
  else if x = 0 then Real.pi
  else 0

/-- Theorem stating that f(f(-3)) = π -/
theorem f_composition_negative_three_eq_pi : f (f (-3)) = Real.pi := by sorry

end NUMINAMATH_CALUDE_f_composition_negative_three_eq_pi_l3639_363901


namespace NUMINAMATH_CALUDE_stock_price_change_l3639_363987

theorem stock_price_change (total_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : ∃ (higher lower : ℕ), 
    higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5)) :
  ∃ (higher : ℕ), higher = 1080 ∧ 
    ∃ (lower : ℕ), higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5) := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l3639_363987


namespace NUMINAMATH_CALUDE_profit_share_difference_l3639_363928

/-- Given the investments of A, B, and C, and B's profit share,
    prove that the difference between A's and C's profit shares is 1600. -/
theorem profit_share_difference
  (investment_A investment_B investment_C : ℕ)
  (profit_share_B : ℕ)
  (h1 : investment_A = 8000)
  (h2 : investment_B = 10000)
  (h3 : investment_C = 12000)
  (h4 : profit_share_B = 4000) :
  let total_investment := investment_A + investment_B + investment_C
  let total_profit := (total_investment * profit_share_B) / investment_B
  let profit_share_A := (investment_A * total_profit) / total_investment
  let profit_share_C := (investment_C * total_profit) / total_investment
  profit_share_C - profit_share_A = 1600 := by
sorry


end NUMINAMATH_CALUDE_profit_share_difference_l3639_363928


namespace NUMINAMATH_CALUDE_complex_number_location_l3639_363958

theorem complex_number_location (z : ℂ) (h : z * Complex.I = 2 - Complex.I) :
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3639_363958


namespace NUMINAMATH_CALUDE_derivative_log2_l3639_363908

theorem derivative_log2 (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_log2_l3639_363908


namespace NUMINAMATH_CALUDE_butter_cheese_ratio_l3639_363976

/-- Represents the prices of items bought by Ursula -/
structure Prices where
  butter : ℝ
  bread : ℝ
  cheese : ℝ
  tea : ℝ

/-- The conditions of Ursula's shopping trip -/
def shopping_conditions (p : Prices) : Prop :=
  p.tea = 10 ∧
  p.tea = 2 * p.cheese ∧
  p.bread = p.butter / 2 ∧
  p.butter + p.bread + p.cheese + p.tea = 21

/-- The theorem stating that under the given conditions, 
    the price of butter is 80% of the price of cheese -/
theorem butter_cheese_ratio (p : Prices) 
  (h : shopping_conditions p) : p.butter / p.cheese = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_butter_cheese_ratio_l3639_363976


namespace NUMINAMATH_CALUDE_equal_cake_distribution_l3639_363962

theorem equal_cake_distribution (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) :
  total_cakes = 18 →
  num_children = 3 →
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
sorry

end NUMINAMATH_CALUDE_equal_cake_distribution_l3639_363962


namespace NUMINAMATH_CALUDE_largest_quantity_l3639_363915

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2010 / 2011 + 2012 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l3639_363915


namespace NUMINAMATH_CALUDE_consecutive_points_distance_l3639_363966

/-- Given 5 consecutive points on a straight line, if certain conditions are met, 
    then the distance between the first two points is 5. -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (c - b) = 2 * (d - c) →  -- bc = 2cd
  (e - d) = 4 →            -- de = 4
  (c - a) = 11 →           -- ac = 11
  (e - a) = 18 →           -- ae = 18
  (b - a) = 5 :=           -- ab = 5
by sorry

end NUMINAMATH_CALUDE_consecutive_points_distance_l3639_363966


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3639_363960

-- Define the equations
def equation1 (x : ℝ) : Prop := x - 2 * Real.sqrt x + 1 = 0
def equation2 (x : ℝ) : Prop := x + 2 + Real.sqrt (x + 2) = 0

-- Theorem for the first equation
theorem solution_equation1 : ∃ (x : ℝ), equation1 x ∧ x = 1 :=
  sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ (x : ℝ), equation2 x ∧ x = -2 :=
  sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3639_363960


namespace NUMINAMATH_CALUDE_arrasta_um_solvable_l3639_363927

/-- Represents a move in the Arrasta Um game -/
inductive Move
| Up : Move
| Down : Move
| Left : Move
| Right : Move

/-- Represents the state of the Arrasta Um game -/
structure ArrastaUmState where
  n : Nat  -- size of the board
  blackPos : Nat × Nat  -- position of the black piece
  emptyPos : Nat × Nat  -- position of the empty cell

/-- Checks if a position is valid on the board -/
def isValidPosition (n : Nat) (pos : Nat × Nat) : Prop :=
  pos.1 < n ∧ pos.2 < n

/-- Checks if two positions are adjacent -/
def isAdjacent (pos1 pos2 : Nat × Nat) : Prop :=
  (pos1.1 = pos2.1 ∧ (pos1.2 + 1 = pos2.2 ∨ pos2.2 + 1 = pos1.2)) ∨
  (pos1.2 = pos2.2 ∧ (pos1.1 + 1 = pos2.1 ∨ pos2.1 + 1 = pos1.1))

/-- Applies a move to the game state -/
def applyMove (state : ArrastaUmState) (move : Move) : ArrastaUmState :=
  match move with
  | Move.Up => { state with blackPos := (state.blackPos.1 - 1, state.blackPos.2), emptyPos := state.blackPos }
  | Move.Down => { state with blackPos := (state.blackPos.1 + 1, state.blackPos.2), emptyPos := state.blackPos }
  | Move.Left => { state with blackPos := (state.blackPos.1, state.blackPos.2 - 1), emptyPos := state.blackPos }
  | Move.Right => { state with blackPos := (state.blackPos.1, state.blackPos.2 + 1), emptyPos := state.blackPos }

/-- Checks if a move is valid -/
def isValidMove (state : ArrastaUmState) (move : Move) : Prop :=
  isAdjacent state.blackPos state.emptyPos ∧
  isValidPosition state.n (applyMove state move).blackPos

/-- Theorem: It's possible to finish Arrasta Um in 6n-8 moves on an n × n board -/
theorem arrasta_um_solvable (n : Nat) (h : n ≥ 2) :
  ∃ (moves : List Move), moves.length = 6 * n - 8 ∧
    (moves.foldl applyMove { n := n, blackPos := (n - 1, 0), emptyPos := (n - 1, 1) }).blackPos = (0, n - 1) :=
  sorry


end NUMINAMATH_CALUDE_arrasta_um_solvable_l3639_363927


namespace NUMINAMATH_CALUDE_valid_course_combinations_l3639_363938

def total_courses : ℕ := 7
def required_courses : ℕ := 4
def math_courses : ℕ := 3
def other_courses : ℕ := 4

def valid_combinations : ℕ := (total_courses - 1).choose (required_courses - 1) - other_courses.choose (required_courses - 1)

theorem valid_course_combinations :
  valid_combinations = 16 :=
sorry

end NUMINAMATH_CALUDE_valid_course_combinations_l3639_363938


namespace NUMINAMATH_CALUDE_power_equality_l3639_363978

theorem power_equality (q : ℕ) (h : (81 : ℕ)^6 = 3^q) : q = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3639_363978


namespace NUMINAMATH_CALUDE_andrews_age_l3639_363952

theorem andrews_age (carlos_age bella_age andrew_age : ℕ) : 
  carlos_age = 20 →
  bella_age = carlos_age + 4 →
  andrew_age = bella_age - 5 →
  andrew_age = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_andrews_age_l3639_363952


namespace NUMINAMATH_CALUDE_sequence_integer_condition_l3639_363949

def sequence_condition (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → x n = (x (n - 2) * x (n - 1)) / (2 * x (n - 2) - x (n - 1))

def infinitely_many_integers (x : ℕ → ℝ) : Prop :=
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ ∃ k : ℤ, x n = k

theorem sequence_integer_condition (x : ℕ → ℝ) :
  (∀ n : ℕ, x n ≠ 0) →
  sequence_condition x →
  (infinitely_many_integers x ↔ ∃ k : ℤ, k ≠ 0 ∧ x 1 = k ∧ x 2 = k) :=
sorry

end NUMINAMATH_CALUDE_sequence_integer_condition_l3639_363949


namespace NUMINAMATH_CALUDE_amanda_notebooks_l3639_363922

/-- Represents the number of notebooks Amanda ordered -/
def ordered_notebooks : ℕ := 6

/-- Amanda's initial number of notebooks -/
def initial_notebooks : ℕ := 10

/-- Number of notebooks Amanda lost -/
def lost_notebooks : ℕ := 2

/-- Amanda's final number of notebooks -/
def final_notebooks : ℕ := 14

theorem amanda_notebooks :
  initial_notebooks + ordered_notebooks - lost_notebooks = final_notebooks :=
by sorry

end NUMINAMATH_CALUDE_amanda_notebooks_l3639_363922


namespace NUMINAMATH_CALUDE_fixed_ray_exists_l3639_363925

/-- Represents a circle with a color -/
structure ColoredCircle where
  center : ℝ × ℝ
  radius : ℝ
  color : Bool

/-- Represents an angle with colored sides -/
structure ColoredAngle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop
  color1 : Bool
  color2 : Bool

/-- Represents a configuration of circles and an angle -/
structure Configuration where
  circle1 : ColoredCircle
  circle2 : ColoredCircle
  angle : ColoredAngle

/-- Predicate to check if circles are non-overlapping -/
def non_overlapping (c1 c2 : ColoredCircle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 > (c1.radius + c2.radius) ^ 2

/-- Predicate to check if a point is outside an angle -/
def outside_angle (p : ℝ × ℝ) (a : ColoredAngle) : Prop :=
  ¬a.side1 p ∧ ¬a.side2 p

/-- Predicate to check if a side touches a circle -/
def touches (side : ℝ × ℝ → Prop) (c : ColoredCircle) : Prop :=
  ∃ p : ℝ × ℝ, side p ∧ (p.1 - c.center.1) ^ 2 + (p.2 - c.center.2) ^ 2 = c.radius ^ 2

/-- Main theorem statement -/
theorem fixed_ray_exists (config : Configuration) 
  (h1 : non_overlapping config.circle1 config.circle2)
  (h2 : config.circle1.color ≠ config.circle2.color)
  (h3 : config.angle.color1 = config.circle1.color)
  (h4 : config.angle.color2 = config.circle2.color)
  (h5 : outside_angle config.circle1.center config.angle)
  (h6 : outside_angle config.circle2.center config.angle)
  (h7 : touches config.angle.side1 config.circle1)
  (h8 : touches config.angle.side2 config.circle2)
  (h9 : config.angle.vertex ≠ config.circle1.center)
  (h10 : config.angle.vertex ≠ config.circle2.center) :
  ∃ (ray : ℝ × ℝ → Prop), ∀ (config' : Configuration), 
    (config'.circle1 = config.circle1 ∧ 
     config'.circle2 = config.circle2 ∧
     config'.angle.vertex = config.angle.vertex ∧
     touches config'.angle.side1 config'.circle1 ∧
     touches config'.angle.side2 config'.circle2) →
    ∃ p : ℝ × ℝ, ray p ∧ 
      (∃ t : ℝ, t > 0 ∧ p = (config'.angle.vertex.1 + t * (p.1 - config'.angle.vertex.1),
                             config'.angle.vertex.2 + t * (p.2 - config'.angle.vertex.2))) :=
sorry

end NUMINAMATH_CALUDE_fixed_ray_exists_l3639_363925


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l3639_363926

/-- A system of equations with consecutive non-integer complex solutions -/
structure ConsecutiveComplexSystem where
  x : ℂ
  y : ℂ
  z : ℂ
  eq1 : (x + 5) * (y - 5) = 0
  eq2 : (y + 5) * (z - 5) = 0
  eq3 : (z + 5) * (x - 5) = 0
  consecutive : ∃ (a b : ℝ), x = a + b * Complex.I ∧ 
                              y = a + (b + 1) * Complex.I ∧ 
                              z = a + (b + 2) * Complex.I
  non_integer : x.im ≠ 0 ∧ y.im ≠ 0 ∧ z.im ≠ 0

/-- The smallest possible sum of absolute squares -/
theorem smallest_sum_of_squares (s : ConsecutiveComplexSystem) : 
  Complex.abs s.x ^ 2 + Complex.abs s.y ^ 2 + Complex.abs s.z ^ 2 ≥ 83.75 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l3639_363926


namespace NUMINAMATH_CALUDE_largest_solution_reciprocal_sixth_power_l3639_363961

noncomputable def largest_solution (x : ℝ) : Prop :=
  (Real.log 10 / Real.log (10 * x^3)) + (Real.log 10 / Real.log (100 * x^4)) = -1 ∧
  ∀ y, (Real.log 10 / Real.log (10 * y^3)) + (Real.log 10 / Real.log (100 * y^4)) = -1 → y ≤ x

theorem largest_solution_reciprocal_sixth_power (x : ℝ) :
  largest_solution x → 1 / x^6 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_reciprocal_sixth_power_l3639_363961


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l3639_363989

theorem base_10_to_base_7 : 
  (1 * 7^3 + 5 * 7^2 + 1 * 7^1 + 5 * 7^0 : ℕ) = 600 := by
  sorry

#eval 1 * 7^3 + 5 * 7^2 + 1 * 7^1 + 5 * 7^0

end NUMINAMATH_CALUDE_base_10_to_base_7_l3639_363989


namespace NUMINAMATH_CALUDE_triangle_side_length_l3639_363968

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  a : Real
  b : Real

-- Define the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : t.A = π / 3)  -- 60 degrees in radians
  (h2 : t.a = Real.sqrt 3)
  (h3 : t.B = π / 6)  -- 30 degrees in radians
  : t.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3639_363968


namespace NUMINAMATH_CALUDE_select_five_from_eight_l3639_363918

/-- The number of ways to select k items from n items without considering order -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: Selecting 5 books from 8 books without order consideration yields 56 ways -/
theorem select_five_from_eight : combination 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l3639_363918


namespace NUMINAMATH_CALUDE_housing_units_without_cable_or_vcr_l3639_363912

theorem housing_units_without_cable_or_vcr 
  (total : ℝ) 
  (cable : ℝ) 
  (vcr : ℝ) 
  (both : ℝ) 
  (h1 : cable = (1 / 5) * total) 
  (h2 : vcr = (1 / 10) * total) 
  (h3 : both = (1 / 3) * cable) :
  (total - (cable + vcr - both)) / total = 23 / 30 := by
sorry

end NUMINAMATH_CALUDE_housing_units_without_cable_or_vcr_l3639_363912


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3639_363983

/-- Given a line y = kx + 1 tangent to the curve y = 1/x at point (a, 1/a) and passing through (0, 1), k equals -1/4 -/
theorem tangent_line_slope (k a : ℝ) : 
  (∀ x, x ≠ 0 → (k * x + 1) = 1 / x ∨ (k * x + 1) > 1 / x) → -- tangent condition
  (k * 0 + 1 = 1) →                                         -- passes through (0, 1)
  (k * a + 1 = 1 / a) →                                     -- point of tangency
  (k = -1 / (a^2)) →                                        -- slope at point of tangency
  (k = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3639_363983


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_two_l3639_363906

theorem points_three_units_from_negative_two : 
  ∀ x : ℝ, (abs (x - (-2)) = 3) ↔ (x = -5 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_two_l3639_363906


namespace NUMINAMATH_CALUDE_problem_statement_l3639_363980

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : y > x) 
  (h : x / y + y / x = 3) : (x + y) / (x - y) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3639_363980


namespace NUMINAMATH_CALUDE_quadratic_from_means_l3639_363910

theorem quadratic_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 6)
  (h_geometric : Real.sqrt (a * b) = 10) :
  ∀ x : ℝ, x^2 - 12*x + 100 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_from_means_l3639_363910


namespace NUMINAMATH_CALUDE_sum_and_difference_l3639_363995

theorem sum_and_difference : 2345 + 3452 + 4523 + 5234 - 1234 = 14320 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_difference_l3639_363995


namespace NUMINAMATH_CALUDE_expression_evaluation_l3639_363911

theorem expression_evaluation (x : ℕ) (h : x = 3) :
  x + x * (x^x) + (x^(x^x)) = 7625597485071 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3639_363911


namespace NUMINAMATH_CALUDE_probability_at_most_one_first_class_l3639_363930

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of products -/
def total_products : ℕ := 5

/-- The number of first-class products -/
def first_class_products : ℕ := 3

/-- The number of second-class products -/
def second_class_products : ℕ := 2

/-- The number of products to be selected -/
def selected_products : ℕ := 2

theorem probability_at_most_one_first_class :
  (choose first_class_products 1 * choose second_class_products 1 + choose second_class_products 2) /
  choose total_products selected_products = 7 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_at_most_one_first_class_l3639_363930


namespace NUMINAMATH_CALUDE_extra_apples_l3639_363943

theorem extra_apples (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 43)
  (h2 : green_apples = 32)
  (h3 : students = 2) :
  red_apples + green_apples - students = 73 := by
  sorry

end NUMINAMATH_CALUDE_extra_apples_l3639_363943


namespace NUMINAMATH_CALUDE_parcera_triples_l3639_363917

def isParcera (p q r : Nat) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  p ∣ (q^2 - 4) ∧ q ∣ (r^2 - 4) ∧ r ∣ (p^2 - 4)

theorem parcera_triples :
  ∀ p q r : Nat, isParcera p q r ↔ 
    ((p, q, r) = (2, 2, 2) ∨ 
     (p, q, r) = (5, 3, 7) ∨ 
     (p, q, r) = (7, 5, 3) ∨ 
     (p, q, r) = (3, 7, 5)) :=
by sorry

end NUMINAMATH_CALUDE_parcera_triples_l3639_363917


namespace NUMINAMATH_CALUDE_pastry_production_theorem_l3639_363956

/-- Represents a baker's production --/
structure BakerProduction where
  mini_cupcakes : ℕ
  pop_tarts : ℕ
  blueberry_pies : ℕ
  chocolate_eclairs : ℕ
  macarons : ℕ

/-- Calculates the total number of pastries for a baker --/
def total_pastries (bp : BakerProduction) : ℕ :=
  bp.mini_cupcakes + bp.pop_tarts + bp.blueberry_pies + bp.chocolate_eclairs + bp.macarons

/-- Calculates the total cost of pastries for a baker --/
def total_cost (bp : BakerProduction) : ℚ :=
  bp.mini_cupcakes * (1/2) + bp.pop_tarts * 1 + bp.blueberry_pies * 3 + bp.chocolate_eclairs * 2 + bp.macarons * (3/2)

theorem pastry_production_theorem (lola lulu lila luka : BakerProduction) : 
  lola = { mini_cupcakes := 13, pop_tarts := 10, blueberry_pies := 8, chocolate_eclairs := 6, macarons := 0 } →
  lulu = { mini_cupcakes := 16, pop_tarts := 12, blueberry_pies := 14, chocolate_eclairs := 9, macarons := 0 } →
  lila = { mini_cupcakes := 22, pop_tarts := 15, blueberry_pies := 10, chocolate_eclairs := 12, macarons := 0 } →
  luka = { mini_cupcakes := 18, pop_tarts := 20, blueberry_pies := 7, chocolate_eclairs := 14, macarons := 25 } →
  (total_pastries lola + total_pastries lulu + total_pastries lila + total_pastries luka = 231) ∧
  (total_cost lola + total_cost lulu + total_cost lila + total_cost luka = 328) := by
  sorry

end NUMINAMATH_CALUDE_pastry_production_theorem_l3639_363956


namespace NUMINAMATH_CALUDE_class_mean_calculation_l3639_363942

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_mean : ℚ) 
  (group2_students : ℕ) (group2_mean : ℚ) : 
  total_students = 50 →
  group1_students = 45 →
  group2_students = 5 →
  group1_mean = 85 / 100 →
  group2_mean = 90 / 100 →
  let overall_mean := (group1_students * group1_mean + group2_students * group2_mean) / total_students
  overall_mean = 855 / 1000 := by
sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l3639_363942


namespace NUMINAMATH_CALUDE_calculator_square_presses_l3639_363929

def square (x : ℕ) : ℕ := x * x

def exceed_1000 (n : ℕ) : Prop := n > 1000

theorem calculator_square_presses :
  (∃ k : ℕ, exceed_1000 (square (square (square 3)))) ∧
  (∀ m : ℕ, m < 3 → ¬exceed_1000 (Nat.iterate square 3 m)) :=
by sorry

end NUMINAMATH_CALUDE_calculator_square_presses_l3639_363929


namespace NUMINAMATH_CALUDE_newspaper_photos_theorem_l3639_363953

/-- Represents the number of photos in a section of the newspaper --/
def photos_in_section (pages : ℕ) (photos_per_page : ℕ) : ℕ :=
  pages * photos_per_page

/-- Calculates the total number of photos in the newspaper for a given day --/
def total_photos_per_day (section_a : ℕ) (section_b : ℕ) (section_c : ℕ) : ℕ :=
  section_a + section_b + section_c

theorem newspaper_photos_theorem :
  let section_a := photos_in_section 25 4
  let section_b := photos_in_section 18 6
  let section_c_monday := photos_in_section 12 5
  let section_c_tuesday := photos_in_section 15 3
  let monday_total := total_photos_per_day section_a section_b section_c_monday
  let tuesday_total := total_photos_per_day section_a section_b section_c_tuesday
  monday_total + tuesday_total = 521 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_photos_theorem_l3639_363953


namespace NUMINAMATH_CALUDE_vector_distance_inequality_l3639_363903

noncomputable def max_T : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

theorem vector_distance_inequality (a b : ℝ × ℝ) :
  (∀ m n : ℝ, let c := (m, 1 - m)
               let d := (n, 1 - n)
               (a.1 - c.1)^2 + (a.2 - c.2)^2 + (b.1 - d.1)^2 + (b.2 - d.2)^2 ≥ max_T^2) →
  (norm a = 1 ∧ norm b = 1 ∧ a.1 * b.1 + a.2 * b.2 = 1/2) :=
sorry

end NUMINAMATH_CALUDE_vector_distance_inequality_l3639_363903


namespace NUMINAMATH_CALUDE_max_b_is_maximum_l3639_363979

def is_lattice_point (x y : ℤ) : Prop := true

def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 50 → is_lattice_point x y → line_equation m x ≠ y

def max_b : ℚ := 11/51

theorem max_b_is_maximum :
  (∀ m : ℚ, 2/5 < m → m < max_b → no_lattice_points m) ∧
  ∀ b : ℚ, b > max_b → ∃ m : ℚ, 2/5 < m ∧ m < b ∧ ¬(no_lattice_points m) :=
sorry

end NUMINAMATH_CALUDE_max_b_is_maximum_l3639_363979


namespace NUMINAMATH_CALUDE_polynomial_integer_root_theorem_l3639_363992

theorem polynomial_integer_root_theorem (n : ℕ+) :
  (∃ (k : Fin n → ℤ) (P : Polynomial ℤ),
    (∀ (i j : Fin n), i ≠ j → k i ≠ k j) ∧
    (Polynomial.degree P ≤ n) ∧
    (∀ (i : Fin n), P.eval (k i) = n) ∧
    (∃ (z : ℤ), P.eval z = 0)) ↔
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

#check polynomial_integer_root_theorem

end NUMINAMATH_CALUDE_polynomial_integer_root_theorem_l3639_363992


namespace NUMINAMATH_CALUDE_no_valid_right_triangle_with_prime_angles_l3639_363907

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_valid_right_triangle_with_prime_angles :
  ¬ ∃ (x : ℕ), 
    x > 0 ∧ 
    3 * x < 90 ∧ 
    x + 3 * x = 90 ∧ 
    is_prime x ∧ 
    is_prime (3 * x) :=
sorry

end NUMINAMATH_CALUDE_no_valid_right_triangle_with_prime_angles_l3639_363907


namespace NUMINAMATH_CALUDE_probability_ratio_l3639_363954

-- Define the total number of slips
def total_slips : ℕ := 30

-- Define the number of different numbers on the slips
def num_options : ℕ := 6

-- Define the number of slips for each number
def slips_per_number : ℕ := 5

-- Define the number of slips drawn
def drawn_slips : ℕ := 4

-- Define the probability of drawing four slips with the same number
def p : ℚ := (num_options * slips_per_number) / Nat.choose total_slips drawn_slips

-- Define the probability of drawing two pairs of slips with different numbers
def q : ℚ := (Nat.choose num_options 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

-- Theorem statement
theorem probability_ratio : q / p = 50 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l3639_363954


namespace NUMINAMATH_CALUDE_jakes_birdhouse_depth_l3639_363975

/-- Calculates the depth of Jake's birdhouse given the dimensions of both birdhouses and their volume difference --/
theorem jakes_birdhouse_depth
  (sara_width : ℝ) (sara_height : ℝ) (sara_depth : ℝ)
  (jake_width : ℝ) (jake_height : ℝ)
  (volume_difference : ℝ)
  (h1 : sara_width = 1) -- 1 foot
  (h2 : sara_height = 2) -- 2 feet
  (h3 : sara_depth = 2) -- 2 feet
  (h4 : jake_width = 16) -- 16 inches
  (h5 : jake_height = 20) -- 20 inches
  (h6 : volume_difference = 1152) -- 1,152 cubic inches
  : ∃ (jake_depth : ℝ),
    jake_depth = 25.2 ∧
    (jake_width * jake_height * jake_depth) - (sara_width * sara_height * sara_depth * 12^3) = volume_difference :=
by sorry

end NUMINAMATH_CALUDE_jakes_birdhouse_depth_l3639_363975


namespace NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l3639_363982

theorem rectangle_triangle_equal_area (perimeter : ℝ) (height : ℝ) (x : ℝ) : 
  perimeter = 60 →
  height = 30 →
  ∃ a b : ℝ, 
    a + b = 30 ∧
    a * b = (1/2) * height * x →
  x = 15 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l3639_363982


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3639_363948

theorem sum_of_x_and_y (x y : ℝ) (hx : 3 + x = 5) (hy : -3 + y = 5) : x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3639_363948


namespace NUMINAMATH_CALUDE_min_value_fractional_sum_l3639_363973

theorem min_value_fractional_sum (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^3 / (y - 2)) + (y^3 / (x - 2)) ≥ 96 ∧
  ((x^3 / (y - 2)) + (y^3 / (x - 2)) = 96 ↔ x = 4 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_fractional_sum_l3639_363973


namespace NUMINAMATH_CALUDE_star_three_four_l3639_363914

-- Define the star operation
def star (a b : ℝ) : ℝ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem star_three_four : star 3 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_three_four_l3639_363914


namespace NUMINAMATH_CALUDE_line_l_equation_l3639_363919

/-- A line l passes through point P(-1,2) and has equal distances from points A(2,3) and B(-4,6) -/
def line_l (x y : ℝ) : Prop :=
  (x = -1 ∧ y = 2) ∨ 
  (abs ((2 * x - y + 2) / Real.sqrt (x^2 + 1)) = abs ((-4 * x - y + 2) / Real.sqrt (x^2 + 1)))

/-- The equation of line l is either x+2y-3=0 or x=-1 -/
theorem line_l_equation : 
  ∀ x y : ℝ, line_l x y ↔ (x + 2*y - 3 = 0 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_l_equation_l3639_363919


namespace NUMINAMATH_CALUDE_sandy_total_marks_l3639_363923

/-- Sandy's marking system and attempt results -/
structure SandyAttempt where
  correct_marks : ℕ  -- Marks for each correct sum
  incorrect_penalty : ℕ  -- Marks lost for each incorrect sum
  total_attempts : ℕ  -- Total number of sums attempted
  correct_attempts : ℕ  -- Number of correct sums

/-- Calculate Sandy's total marks -/
def calculate_total_marks (s : SandyAttempt) : ℤ :=
  (s.correct_attempts * s.correct_marks : ℤ) -
  ((s.total_attempts - s.correct_attempts) * s.incorrect_penalty : ℤ)

/-- Theorem stating that Sandy's total marks is 65 -/
theorem sandy_total_marks :
  let s : SandyAttempt := {
    correct_marks := 3,
    incorrect_penalty := 2,
    total_attempts := 30,
    correct_attempts := 25
  }
  calculate_total_marks s = 65 := by
  sorry

end NUMINAMATH_CALUDE_sandy_total_marks_l3639_363923


namespace NUMINAMATH_CALUDE_hamburger_combinations_l3639_363955

theorem hamburger_combinations : 
  let num_condiments : ℕ := 10
  let num_bun_types : ℕ := 2
  let num_patty_choices : ℕ := 3
  (2 ^ num_condiments) * num_bun_types * num_patty_choices = 6144 :=
by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l3639_363955


namespace NUMINAMATH_CALUDE_no_consecutive_tails_probability_l3639_363972

/-- Represents the number of ways to toss n coins without getting two consecutive tails -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => a (n + 1) + a n

/-- The probability of not getting two consecutive tails when tossing five fair coins -/
theorem no_consecutive_tails_probability : 
  (a 5 : ℚ) / (2^5 : ℚ) = 13 / 32 := by sorry

end NUMINAMATH_CALUDE_no_consecutive_tails_probability_l3639_363972


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l3639_363902

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l3639_363902


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l3639_363988

/-- The set T of points (x,y) in the coordinate plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b : ℝ), 
    ((a = 7 ∧ b = p.1 - 3) ∨ 
     (a = 7 ∧ b = p.2 + 5) ∨ 
     (a = p.1 - 3 ∧ b = p.2 + 5)) ∧
    (a = b) ∧
    (7 ≥ a ∧ p.1 - 3 ≥ a ∧ p.2 + 5 ≥ a)}

/-- A ray in the plane, defined by its starting point and direction -/
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- The property that T consists of three rays with a common point -/
def isThreeRaysWithCommonPoint (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ) (r₁ r₂ r₃ : Ray),
    r₁.start = p ∧ r₂.start = p ∧ r₃.start = p ∧
    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    s = {q : ℝ × ℝ | ∃ (t : ℝ), t ≥ 0 ∧
      (q = r₁.start + t • r₁.direction ∨
       q = r₂.start + t • r₂.direction ∨
       q = r₃.start + t • r₃.direction)}

theorem T_is_three_rays_with_common_point : isThreeRaysWithCommonPoint T := by
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l3639_363988


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3639_363998

theorem sum_of_fifth_powers (n : ℕ) : 
  (∃ (A B C D E : ℤ), n = A^5 + B^5 + C^5 + D^5 + E^5) ∧ 
  (¬ ∃ (A B C D : ℤ), n = A^5 + B^5 + C^5 + D^5) := by
  sorry

#check sum_of_fifth_powers 2018

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3639_363998


namespace NUMINAMATH_CALUDE_reduced_oil_price_l3639_363939

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  original_quantity : ℝ
  additional_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem reduced_oil_price 
  (scenario : OilPriceReduction) 
  (h1 : scenario.reduced_price = 0.8 * scenario.original_price) 
  (h2 : scenario.original_quantity * scenario.original_price = scenario.total_cost) 
  (h3 : (scenario.original_quantity + scenario.additional_quantity) * scenario.reduced_price = scenario.total_cost) 
  (h4 : scenario.additional_quantity = 4) 
  (h5 : scenario.total_cost = 600) : 
  scenario.reduced_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_reduced_oil_price_l3639_363939


namespace NUMINAMATH_CALUDE_rent_and_earnings_increase_l3639_363996

theorem rent_and_earnings_increase (last_year_earnings : ℝ) (increase_percent : ℝ) : 
  (0.3 * (last_year_earnings * (1 + increase_percent / 100)) = 2.025 * (0.2 * last_year_earnings)) →
  increase_percent = 35 := by
  sorry

end NUMINAMATH_CALUDE_rent_and_earnings_increase_l3639_363996


namespace NUMINAMATH_CALUDE_f_6_equals_0_l3639_363981

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x in ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has the property f(x+2) = -f(x) for all x in ℝ -/
def HasPeriod2WithSignFlip (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

theorem f_6_equals_0 (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : HasPeriod2WithSignFlip f) : f 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_6_equals_0_l3639_363981


namespace NUMINAMATH_CALUDE_max_y_value_max_y_achieved_l3639_363950

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = 4) : y ≤ 7 := by
  sorry

theorem max_y_achieved : ∃ x y : ℤ, x * y + 3 * x + 2 * y = 4 ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_max_y_achieved_l3639_363950


namespace NUMINAMATH_CALUDE_gumball_distribution_l3639_363935

theorem gumball_distribution (joanna_initial : Nat) (jacques_initial : Nat) : 
  joanna_initial = 40 →
  jacques_initial = 60 →
  let joanna_final := joanna_initial + 5 * joanna_initial
  let jacques_final := jacques_initial + 3 * jacques_initial
  let total := joanna_final + jacques_final
  let shared := total / 2
  shared = 240 :=
by sorry

end NUMINAMATH_CALUDE_gumball_distribution_l3639_363935


namespace NUMINAMATH_CALUDE_no_geometric_progression_with_1_2_5_l3639_363964

theorem no_geometric_progression_with_1_2_5 :
  ¬ ∃ (a q : ℝ) (m n p : ℕ), 
    m ≠ n ∧ n ≠ p ∧ m ≠ p ∧
    a * q^m = 1 ∧ a * q^n = 2 ∧ a * q^p = 5 :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_progression_with_1_2_5_l3639_363964


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l3639_363977

theorem vector_subtraction_and_scalar_multiplication :
  (⟨3, -8⟩ : ℝ × ℝ) - 3 • (⟨-2, 6⟩ : ℝ × ℝ) = (⟨9, -26⟩ : ℝ × ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l3639_363977
