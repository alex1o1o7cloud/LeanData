import Mathlib

namespace NUMINAMATH_CALUDE_largest_base_for_12_4th_power_l26_2632

/-- Given a natural number n and a base b, returns the sum of digits of n in base b -/
def sumOfDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Returns true if b is a valid base (greater than 1) -/
def isValidBase (b : ℕ) : Prop := b > 1

theorem largest_base_for_12_4th_power :
  ∀ b : ℕ, isValidBase b →
    (b ≤ 7 ↔ sumOfDigits (12^4) b ≠ 2^5) ∧
    (b > 7 → sumOfDigits (12^4) b = 2^5) :=
sorry

end NUMINAMATH_CALUDE_largest_base_for_12_4th_power_l26_2632


namespace NUMINAMATH_CALUDE_arcsin_arccos_equation_solutions_l26_2698

theorem arcsin_arccos_equation_solutions :
  ∀ x : ℝ, (x = 0 ∨ x = 1/2 ∨ x = -1/2) →
  Real.arcsin (2*x) + Real.arcsin (1 - 2*x) = Real.arccos (2*x) := by
sorry

end NUMINAMATH_CALUDE_arcsin_arccos_equation_solutions_l26_2698


namespace NUMINAMATH_CALUDE_sum_first_last_is_14_l26_2650

/-- A sequence of seven terms satisfying specific conditions -/
structure SevenTermSequence where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  U : ℝ
  V : ℝ
  R_eq_7 : R = 7
  sum_consecutive_3 : ∀ (x y z : ℝ), (x = P ∧ y = Q ∧ z = R) ∨
                                     (x = Q ∧ y = R ∧ z = S) ∨
                                     (x = R ∧ y = S ∧ z = T) ∨
                                     (x = S ∧ y = T ∧ z = U) ∨
                                     (x = T ∧ y = U ∧ z = V) →
                                     x + y + z = 21

/-- The sum of the first and last terms in a seven-term sequence is 14 -/
theorem sum_first_last_is_14 (seq : SevenTermSequence) : seq.P + seq.V = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_last_is_14_l26_2650


namespace NUMINAMATH_CALUDE_ten_people_no_adjacent_standing_probability_l26_2605

/-- Represents the number of valid arrangements for n people where no two adjacent people stand --/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing in a circular arrangement of n people --/
def noAdjacentStandingProbability (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n : ℚ)

theorem ten_people_no_adjacent_standing_probability :
  noAdjacentStandingProbability 10 = 123 / 1024 := by
  sorry


end NUMINAMATH_CALUDE_ten_people_no_adjacent_standing_probability_l26_2605


namespace NUMINAMATH_CALUDE_crow_eating_quarter_l26_2689

/-- Represents the time it takes for a crow to eat a certain fraction of nuts -/
def crow_eating_time (fraction_eaten : ℚ) (time : ℚ) : Prop :=
  fraction_eaten * time = (1 : ℚ) / 5 * 6

/-- Proves that it takes 7.5 hours for a crow to eat 1/4 of the nuts, 
    given that it eats 1/5 of the nuts in 6 hours -/
theorem crow_eating_quarter : 
  crow_eating_time (1 / 4) (15 / 2) :=
sorry

end NUMINAMATH_CALUDE_crow_eating_quarter_l26_2689


namespace NUMINAMATH_CALUDE_reflection_x_axis_l26_2668

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem reflection_x_axis (P : ℝ × ℝ) (h : P = (-2, 1)) :
  reflect_x P = (-2, -1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_x_axis_l26_2668


namespace NUMINAMATH_CALUDE_fraction_evaluation_l26_2622

theorem fraction_evaluation : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l26_2622


namespace NUMINAMATH_CALUDE_hundred_with_six_digits_l26_2649

theorem hundred_with_six_digits (x : ℕ) (h : x ≠ 0) (h2 : x < 10) :
  (100 * x + 10 * x + x) - (10 * x + x) = 100 * x :=
by sorry

end NUMINAMATH_CALUDE_hundred_with_six_digits_l26_2649


namespace NUMINAMATH_CALUDE_price_increase_demand_decrease_l26_2628

theorem price_increase_demand_decrease (P Q : ℝ) (P_new Q_new : ℝ) : 
  P_new = 1.2 * P →  -- Price increases by 20%
  P_new * Q_new = 1.1 * (P * Q) →  -- Total income increases by 10%
  (Q - Q_new) / Q = 1 / 12 :=  -- Demand decreases by 1/12
by sorry

end NUMINAMATH_CALUDE_price_increase_demand_decrease_l26_2628


namespace NUMINAMATH_CALUDE_total_money_value_l26_2633

def us_100_bills : ℕ := 2
def us_50_bills : ℕ := 5
def us_10_bills : ℕ := 5
def canadian_20_bills : ℕ := 15
def euro_10_notes : ℕ := 20
def us_quarters : ℕ := 50
def us_dimes : ℕ := 120

def cad_to_usd_rate : ℚ := 0.80
def eur_to_usd_rate : ℚ := 1.10

def total_us_currency : ℚ := 
  us_100_bills * 100 + 
  us_50_bills * 50 + 
  us_10_bills * 10 + 
  us_quarters * 0.25 + 
  us_dimes * 0.10

def total_cad_in_usd : ℚ := canadian_20_bills * 20 * cad_to_usd_rate
def total_eur_in_usd : ℚ := euro_10_notes * 10 * eur_to_usd_rate

theorem total_money_value : 
  total_us_currency + total_cad_in_usd + total_eur_in_usd = 984.50 := by
  sorry

end NUMINAMATH_CALUDE_total_money_value_l26_2633


namespace NUMINAMATH_CALUDE_prob_different_suits_no_jokers_l26_2655

/-- Extended deck with 54 cards including 2 jokers -/
def extendedDeck : ℕ := 54

/-- Number of jokers in the extended deck -/
def jokers : ℕ := 2

/-- Number of suits in a standard deck -/
def numSuits : ℕ := 4

/-- Number of cards per suit in a standard deck -/
def cardsPerSuit : ℕ := 13

/-- Probability of picking two cards of different suits given no jokers are picked -/
theorem prob_different_suits_no_jokers :
  let nonJokerCards := extendedDeck - jokers
  let firstPickOptions := nonJokerCards
  let secondPickOptions := nonJokerCards - 1
  let differentSuitOptions := (numSuits - 1) * cardsPerSuit
  (differentSuitOptions : ℚ) / secondPickOptions = 13 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_no_jokers_l26_2655


namespace NUMINAMATH_CALUDE_coffee_shop_usage_l26_2601

/-- The number of bags of coffee beans used every morning -/
def morning_bags : ℕ := 3

/-- The number of bags of coffee beans used every afternoon -/
def afternoon_bags : ℕ := 3 * morning_bags

/-- The number of bags of coffee beans used every evening -/
def evening_bags : ℕ := 2 * morning_bags

/-- The total number of bags used in a week -/
def weekly_bags : ℕ := 126

theorem coffee_shop_usage :
  7 * (morning_bags + afternoon_bags + evening_bags) = weekly_bags :=
sorry

end NUMINAMATH_CALUDE_coffee_shop_usage_l26_2601


namespace NUMINAMATH_CALUDE_wooden_block_volume_l26_2634

/-- A rectangular wooden block -/
structure WoodenBlock where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The volume of a wooden block -/
def volume (block : WoodenBlock) : ℝ :=
  block.length * block.width * block.height

/-- The surface area of a wooden block -/
def surfaceArea (block : WoodenBlock) : ℝ :=
  2 * (block.length * block.width + block.length * block.height + block.width * block.height)

/-- The increase in surface area after sawing -/
def surfaceAreaIncrease (block : WoodenBlock) (sections : ℕ) : ℝ :=
  2 * (sections - 1) * block.width * block.height

theorem wooden_block_volume
  (block : WoodenBlock)
  (h_length : block.length = 10)
  (h_sections : ℕ)
  (h_sections_eq : h_sections = 6)
  (h_area_increase : surfaceAreaIncrease block h_sections = 1) :
  volume block = 10 := by
  sorry

end NUMINAMATH_CALUDE_wooden_block_volume_l26_2634


namespace NUMINAMATH_CALUDE_projection_of_congruent_vectors_l26_2680

/-- Definition of vector congruence -/
def is_congruent (a b : ℝ × ℝ) : Prop :=
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.sqrt (a.1^2 + a.2^2) / Real.sqrt (b.1^2 + b.2^2) = Real.cos θ ∧ 0 ≤ θ ∧ θ < Real.pi / 2

/-- Theorem: Projection of a-b on a when b is congruent to a -/
theorem projection_of_congruent_vectors (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0))
    (h_congruent : is_congruent b a) :
  let proj := ((a.1 - b.1) * a.1 + (a.2 - b.2) * a.2) / Real.sqrt (a.1^2 + a.2^2)
  proj = (a.1^2 + a.2^2 - (b.1^2 + b.2^2)) / Real.sqrt (a.1^2 + a.2^2) := by
  sorry

end NUMINAMATH_CALUDE_projection_of_congruent_vectors_l26_2680


namespace NUMINAMATH_CALUDE_point_with_specific_rate_of_change_l26_2640

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 5

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 2*x - 3

theorem point_with_specific_rate_of_change :
  ∃ (x y : ℝ), f x = y ∧ f' x = 5 ∧ x = 4 ∧ y = 9 := by sorry

end NUMINAMATH_CALUDE_point_with_specific_rate_of_change_l26_2640


namespace NUMINAMATH_CALUDE_max_value_complex_l26_2665

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2)^2 * (z + 2)) ≤ 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_l26_2665


namespace NUMINAMATH_CALUDE_problem_solution_l26_2644

theorem problem_solution (x y : ℤ) : 
  x > y ∧ y > 0 ∧ x + y + x * y = 152 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l26_2644


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l26_2636

/-- Proves that the number of adult tickets sold is 122, given the total number of tickets
    and the relationship between student and adult tickets. -/
theorem adult_tickets_sold (total_tickets : ℕ) (adult_tickets : ℕ) (student_tickets : ℕ)
    (h1 : total_tickets = 366)
    (h2 : student_tickets = 2 * adult_tickets)
    (h3 : total_tickets = adult_tickets + student_tickets) :
    adult_tickets = 122 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l26_2636


namespace NUMINAMATH_CALUDE_correct_equation_l26_2642

/-- Represents the problem of sending a letter over a certain distance with two horses of different speeds. -/
def letter_problem (distance : ℝ) (slow_delay : ℝ) (fast_early : ℝ) (speed_ratio : ℝ) :=
  ∀ x : ℝ, x > 3 → (distance / (x + slow_delay)) * speed_ratio = distance / (x - fast_early)

/-- The theorem states that the given equation correctly represents the problem for the specific values mentioned. -/
theorem correct_equation : letter_problem 900 1 3 2 := by sorry

end NUMINAMATH_CALUDE_correct_equation_l26_2642


namespace NUMINAMATH_CALUDE_complement_of_N_in_U_l26_2613

-- Define the universal set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set N
def N : Set ℝ := {x | 0 ≤ x ∧ x < 2}

-- Theorem statement
theorem complement_of_N_in_U :
  (U \ N) = {x | (-3 ≤ x ∧ x < 0) ∨ (2 ≤ x ∧ x ≤ 3)} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_N_in_U_l26_2613


namespace NUMINAMATH_CALUDE_coordinate_sum_theorem_l26_2679

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Theorem statement
theorem coordinate_sum_theorem (h : 3 * (f 2) = 5) :
  2 * (f_inv (5/3)) = 4 ∧ 5/3 + 4 = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_theorem_l26_2679


namespace NUMINAMATH_CALUDE_grade_distribution_l26_2677

theorem grade_distribution (total_students : ℕ) 
  (prob_A : ℝ) (prob_B : ℝ) (prob_C : ℝ) 
  (h1 : prob_A = 0.8 * prob_B) 
  (h2 : prob_C = 1.2 * prob_B) 
  (h3 : prob_A + prob_B + prob_C = 1) 
  (h4 : total_students = 40) :
  ∃ (A B C : ℕ), 
    A + B + C = total_students ∧ 
    A = 10 ∧ 
    B = 14 ∧ 
    C = 16 := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l26_2677


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l26_2631

theorem isosceles_triangle_base_length 
  (perimeter : ℝ) 
  (one_side : ℝ) 
  (h_perimeter : perimeter = 15) 
  (h_one_side : one_side = 3) 
  (h_isosceles : ∃ (leg : ℝ), 2 * leg + one_side = perimeter) :
  one_side = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l26_2631


namespace NUMINAMATH_CALUDE_binomial_12_11_l26_2683

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l26_2683


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_successors_l26_2652

theorem unique_prime_with_prime_successors :
  ∃! p : ℕ, Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_successors_l26_2652


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l26_2611

theorem sarahs_bowling_score (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 60 →
  (sarah_score + greg_score) / 2 = 110 →
  sarah_score + greg_score < 450 →
  sarah_score = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l26_2611


namespace NUMINAMATH_CALUDE_jeans_price_increase_l26_2616

/-- Given a manufacturing cost C, calculate the percentage increase from the retailer's price to the customer's price -/
theorem jeans_price_increase (C : ℝ) : 
  let retailer_price := C * 1.4
  let customer_price := C * 1.82
  (customer_price - retailer_price) / retailer_price * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_jeans_price_increase_l26_2616


namespace NUMINAMATH_CALUDE_nested_fraction_square_l26_2646

theorem nested_fraction_square (x : ℚ) (h : x = 1/3) :
  let f := (x + 2) / (x - 2)
  ((f + 2) / (f - 2))^2 = 961/1369 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_square_l26_2646


namespace NUMINAMATH_CALUDE_boys_in_class_l26_2675

theorem boys_in_class (initial_girls : ℕ) (initial_boys : ℕ) (final_girls : ℕ) :
  (initial_girls : ℚ) / initial_boys = 5 / 6 →
  (final_girls : ℚ) / initial_boys = 2 / 3 →
  initial_girls - final_girls = 20 →
  initial_boys = 120 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l26_2675


namespace NUMINAMATH_CALUDE_modular_inverse_of_35_mod_37_l26_2645

theorem modular_inverse_of_35_mod_37 :
  ∃ x : ℤ, (35 * x) % 37 = 1 ∧ x % 37 = 18 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_35_mod_37_l26_2645


namespace NUMINAMATH_CALUDE_smallest_tablecloth_diameter_l26_2667

/-- The smallest diameter of a circular tablecloth that can completely cover a square table with sides of 1 meter is √2 meters. -/
theorem smallest_tablecloth_diameter (table_side : ℝ) (h : table_side = 1) :
  let diagonal := Real.sqrt (2 * table_side ^ 2)
  diagonal = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_tablecloth_diameter_l26_2667


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_six_l26_2695

/-- Represents a pair of integers in the sequence -/
structure Pair :=
  (first : ℕ)
  (second : ℕ)

/-- Returns the sum of elements in a pair -/
def pairSum (p : Pair) : ℕ := p.first + p.second

/-- Returns the number of pairs in the first n levels -/
def pairsInFirstNLevels (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Returns the nth pair in the sequence -/
def nthPair (n : ℕ) : Pair :=
  sorry -- Implementation details omitted

/-- The main theorem to prove -/
theorem sixtieth_pair_is_five_six :
  nthPair 60 = Pair.mk 5 6 := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_six_l26_2695


namespace NUMINAMATH_CALUDE_base_five_representation_l26_2687

theorem base_five_representation (b : ℕ) : 
  (b^3 ≤ 329 ∧ 329 < b^4 ∧ 329 % b % 2 = 0) ↔ b = 5 :=
by sorry

end NUMINAMATH_CALUDE_base_five_representation_l26_2687


namespace NUMINAMATH_CALUDE_car_tire_rotation_theorem_l26_2664

/-- Calculates the number of miles each tire is used given the total number of tires, 
    tires used simultaneously, and total miles traveled. -/
def miles_per_tire (total_tires : ℕ) (tires_used : ℕ) (total_miles : ℕ) : ℕ :=
  (total_miles * tires_used) / total_tires

/-- Proves that for a car with 5 tires, where 4 are used simultaneously over 30,000 miles,
    each tire is used for 24,000 miles. -/
theorem car_tire_rotation_theorem :
  miles_per_tire 5 4 30000 = 24000 := by
  sorry

#eval miles_per_tire 5 4 30000

end NUMINAMATH_CALUDE_car_tire_rotation_theorem_l26_2664


namespace NUMINAMATH_CALUDE_triangle_properties_l26_2647

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 6)
  (h2 : Real.cos t.A = 1/8)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = 15 * Real.sqrt 7 / 4) :
  Real.sin t.C = Real.sqrt 7 / 4 ∧ t.b + t.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l26_2647


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l26_2619

theorem divisibility_by_eleven (m : Nat) : 
  m < 10 → -- m is a single digit
  (742 * 100000 + m * 10000 + 834) % 11 = 0 → -- 742m834 is divisible by 11
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l26_2619


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l26_2602

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l26_2602


namespace NUMINAMATH_CALUDE_max_value_expression_l26_2692

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (⨆ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + b^2 + c))) = 3/2 * (b^2 + c) + 3 * a^2 := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l26_2692


namespace NUMINAMATH_CALUDE_point_on_h_graph_l26_2658

theorem point_on_h_graph (g : ℝ → ℝ) (h : ℝ → ℝ) : 
  g 4 = 7 → 
  (∀ x, h x = (g x + 1)^2) → 
  ∃ x y, h x = y ∧ x + y = 68 := by
sorry

end NUMINAMATH_CALUDE_point_on_h_graph_l26_2658


namespace NUMINAMATH_CALUDE_triangle_properties_l26_2617

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with angles A, B, C and opposite sides a, b, c
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- B is an obtuse angle
  π/2 < B ∧ B < π ∧
  -- √3a = 2b sin A
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  -- 1. B = 2π/3
  B = 2 * π / 3 ∧
  -- 2. If the area is 15√3/4 and b = 7, then a + c = 8
  (1/2 * a * c * Real.sin B = 15 * Real.sqrt 3 / 4 ∧ b = 7 → a + c = 8) ∧
  -- 3. If b = 6, the maximum area is 3√3
  (b = 6 → ∀ (a' c' : ℝ), 1/2 * a' * c' * Real.sin B ≤ 3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l26_2617


namespace NUMINAMATH_CALUDE_sequence_convergence_l26_2606

def S (seq : List Int) : List Int :=
  let n := seq.length
  List.zipWith (· * ·) seq ((seq.drop 1).append [seq.head!])

def all_ones (seq : List Int) : Prop :=
  seq.all (· = 1)

theorem sequence_convergence (n : Nat) (seq : List Int) :
  seq.length = 2 * n →
  (∀ i, i ∈ seq → i = 1 ∨ i = -1) →
  ∃ k : Nat, all_ones (Nat.iterate S k seq) := by
  sorry

end NUMINAMATH_CALUDE_sequence_convergence_l26_2606


namespace NUMINAMATH_CALUDE_tank_capacity_l26_2662

theorem tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (used_gallons : ℕ) : 
  initial_fraction = 3/4 → 
  final_fraction = 1/4 → 
  used_gallons = 24 → 
  ∃ (total_capacity : ℕ), 
    total_capacity = 48 ∧ 
    (initial_fraction - final_fraction) * total_capacity = used_gallons :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l26_2662


namespace NUMINAMATH_CALUDE_intersection_sum_l26_2670

/-- Given two lines y = mx + 5 and y = 2x + b intersecting at (7, 10),
    prove that the sum of constants b and m is equal to -23/7 -/
theorem intersection_sum (m b : ℚ) : 
  (7 * m + 5 = 10) → (2 * 7 + b = 10) → b + m = -23/7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l26_2670


namespace NUMINAMATH_CALUDE_land_area_needed_l26_2661

def land_cost_per_sqm : ℝ := 50
def brick_cost_per_1000 : ℝ := 100
def roof_tile_cost_per_tile : ℝ := 10
def num_bricks_needed : ℝ := 10000
def num_roof_tiles_needed : ℝ := 500
def total_construction_cost : ℝ := 106000

theorem land_area_needed :
  ∃ (x : ℝ),
    x * land_cost_per_sqm +
    (num_bricks_needed / 1000) * brick_cost_per_1000 +
    num_roof_tiles_needed * roof_tile_cost_per_tile =
    total_construction_cost ∧
    x = 2000 :=
by sorry

end NUMINAMATH_CALUDE_land_area_needed_l26_2661


namespace NUMINAMATH_CALUDE_correct_calculation_l26_2678

theorem correct_calculation (a : ℝ) : a^5 + a^5 = 2*a^5 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l26_2678


namespace NUMINAMATH_CALUDE_no_three_way_partition_of_positive_integers_l26_2604

theorem no_three_way_partition_of_positive_integers :
  ¬ ∃ (A B C : Set ℕ+),
    (A ∪ B ∪ C = Set.univ) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
    (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
    (∀ x ∈ A, ∀ y ∈ B, (x^2 - x*y + y^2) ∈ C) ∧
    (∀ x ∈ B, ∀ y ∈ C, (x^2 - x*y + y^2) ∈ A) ∧
    (∀ x ∈ C, ∀ y ∈ A, (x^2 - x*y + y^2) ∈ B) :=
by sorry

end NUMINAMATH_CALUDE_no_three_way_partition_of_positive_integers_l26_2604


namespace NUMINAMATH_CALUDE_equation_solution_l26_2627

theorem equation_solution : 
  {x : ℝ | (x ≠ 0 ∧ x + 2 ≠ 0 ∧ x + 4 ≠ 0 ∧ x + 6 ≠ 0 ∧ x + 8 ≠ 0) ∧ 
           (1/x + 1/(x+2) - 1/(x+4) - 1/(x+6) + 1/(x+8) = 0)} = 
  {-4 - 2 * Real.sqrt 3, 2 - 2 * Real.sqrt 3} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l26_2627


namespace NUMINAMATH_CALUDE_kylie_daisies_l26_2638

theorem kylie_daisies (initial : ℕ) (final : ℕ) (sister_gave : ℕ) : 
  initial = 5 →
  final = 7 →
  (initial + sister_gave) / 2 = final →
  sister_gave = 9 :=
by sorry

end NUMINAMATH_CALUDE_kylie_daisies_l26_2638


namespace NUMINAMATH_CALUDE_prime_square_diff_divisible_by_24_l26_2659

theorem prime_square_diff_divisible_by_24 (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) : 
  24 ∣ (p^2 - q^2) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_diff_divisible_by_24_l26_2659


namespace NUMINAMATH_CALUDE_max_current_speed_is_26_l26_2637

/-- The speed of Mumbo running -/
def mumbo_speed : ℝ := 11

/-- The speed of Yumbo walking -/
def yumbo_speed : ℝ := 6

/-- Predicate to check if a given speed is a valid river current speed -/
def is_valid_current_speed (v : ℝ) : Prop :=
  v ≥ 6 ∧ ∃ (n : ℕ), v = n

/-- Predicate to check if Yumbo arrives before Mumbo given distances and current speed -/
def yumbo_arrives_first (x y v : ℝ) : Prop :=
  y / yumbo_speed < x / mumbo_speed + (x + y) / v

/-- The maximum possible river current speed -/
def max_current_speed : ℕ := 26

/-- The main theorem stating that 26 km/h is the maximum possible river current speed -/
theorem max_current_speed_is_26 :
  ∀ (v : ℝ), is_valid_current_speed v →
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x < y ∧ yumbo_arrives_first x y v) →
  v ≤ max_current_speed :=
sorry

end NUMINAMATH_CALUDE_max_current_speed_is_26_l26_2637


namespace NUMINAMATH_CALUDE_berry_package_cost_l26_2672

/-- The cost of one package of berries given Martin's consumption habits and spending --/
theorem berry_package_cost (daily_consumption : ℚ) (package_size : ℚ) (days : ℕ) (total_spent : ℚ) : 
  daily_consumption = 1/2 →
  package_size = 1 →
  days = 30 →
  total_spent = 30 →
  (total_spent / (days * daily_consumption / package_size) = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_berry_package_cost_l26_2672


namespace NUMINAMATH_CALUDE_white_patterns_count_l26_2657

/-- The number of different white figures on an n × n board created by k rectangles -/
def whitePatterns (n k : ℕ) : ℕ :=
  (Nat.choose n k) ^ 2

/-- Theorem stating the number of different white figures -/
theorem white_patterns_count (n k : ℕ) (h1 : n > 0) (h2 : k > 0) (h3 : k ≤ n) :
  whitePatterns n k = (Nat.choose n k) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_white_patterns_count_l26_2657


namespace NUMINAMATH_CALUDE_G_minimized_at_three_l26_2669

/-- The number of devices required for a base-n system -/
noncomputable def G (n : ℕ) (M : ℕ) : ℝ :=
  (n : ℝ) / Real.log n * Real.log (M + 1)

/-- The theorem stating that G is minimized when n = 3 -/
theorem G_minimized_at_three (M : ℕ) :
  ∀ n : ℕ, n ≥ 2 → G 3 M ≤ G n M :=
sorry

end NUMINAMATH_CALUDE_G_minimized_at_three_l26_2669


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l26_2609

theorem meaningful_expression_range (x : ℝ) : 
  (∃ (y : ℝ), y = (Real.sqrt (x + 4)) / (x - 2)) ↔ (x ≥ -4 ∧ x ≠ 2) := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l26_2609


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l26_2697

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 4 = 4 → a 2 + a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l26_2697


namespace NUMINAMATH_CALUDE_discount_difference_l26_2639

theorem discount_difference : 
  let initial_amount : ℝ := 12000
  let single_discount : ℝ := 0.3
  let first_successive_discount : ℝ := 0.2
  let second_successive_discount : ℝ := 0.1
  let single_discounted_amount : ℝ := initial_amount * (1 - single_discount)
  let successive_discounted_amount : ℝ := initial_amount * (1 - first_successive_discount) * (1 - second_successive_discount)
  successive_discounted_amount - single_discounted_amount = 240 :=
by sorry

end NUMINAMATH_CALUDE_discount_difference_l26_2639


namespace NUMINAMATH_CALUDE_games_in_specific_league_l26_2608

/-- The number of games played in a season for a league with a given number of teams and repetitions -/
def games_in_season (num_teams : ℕ) (repetitions : ℕ) : ℕ :=
  (num_teams * (num_teams - 1) / 2) * repetitions

/-- Theorem stating the number of games in a season for a specific league setup -/
theorem games_in_specific_league : games_in_season 14 5 = 455 := by
  sorry

end NUMINAMATH_CALUDE_games_in_specific_league_l26_2608


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l26_2656

theorem arithmetic_expression_equality : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l26_2656


namespace NUMINAMATH_CALUDE_s_scale_indeterminate_l26_2653

/-- Represents a linear relationship between two measurement scales -/
structure ScaleRelation where
  /-- Slope of the linear relationship -/
  a : ℝ
  /-- Y-intercept of the linear relationship -/
  b : ℝ

/-- Converts a p-scale measurement to an s-scale measurement -/
def toSScale (relation : ScaleRelation) (p : ℝ) : ℝ :=
  relation.a * p + relation.b

/-- Theorem stating that the s-scale measurement for p=24 cannot be uniquely determined -/
theorem s_scale_indeterminate (known_p : ℝ) (known_s : ℝ) (target_p : ℝ) 
    (h1 : known_p = 6) (h2 : known_s = 30) (h3 : target_p = 24) :
    ∃ (r1 r2 : ScaleRelation), r1 ≠ r2 ∧ 
    toSScale r1 known_p = known_s ∧
    toSScale r2 known_p = known_s ∧
    toSScale r1 target_p ≠ toSScale r2 target_p :=
  sorry

end NUMINAMATH_CALUDE_s_scale_indeterminate_l26_2653


namespace NUMINAMATH_CALUDE_cubic_equation_result_l26_2671

theorem cubic_equation_result (x : ℝ) (h : x^2 + 3*x - 1 = 0) : 
  x^3 + 5*x^2 + 5*x + 18 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_result_l26_2671


namespace NUMINAMATH_CALUDE_faye_crayons_count_l26_2696

/-- Given that Faye arranges her crayons in 16 rows with 6 crayons per row,
    prove that she has 96 crayons in total. -/
theorem faye_crayons_count : 
  let rows : ℕ := 16
  let crayons_per_row : ℕ := 6
  rows * crayons_per_row = 96 := by
sorry

end NUMINAMATH_CALUDE_faye_crayons_count_l26_2696


namespace NUMINAMATH_CALUDE_problem_solution_l26_2660

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 10) :
  (x + y) / (x - y) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l26_2660


namespace NUMINAMATH_CALUDE_spinning_class_duration_l26_2691

/-- Calculates the number of hours worked out in each spinning class. -/
def hours_per_class (classes_per_week : ℕ) (calories_per_minute : ℕ) (total_calories_per_week : ℕ) : ℚ :=
  (total_calories_per_week / classes_per_week) / (calories_per_minute * 60)

/-- Proves that given the specified conditions, James works out for 1.5 hours in each spinning class. -/
theorem spinning_class_duration :
  let classes_per_week : ℕ := 3
  let calories_per_minute : ℕ := 7
  let total_calories_per_week : ℕ := 1890
  hours_per_class classes_per_week calories_per_minute total_calories_per_week = 3/2 := by
  sorry

#eval hours_per_class 3 7 1890

end NUMINAMATH_CALUDE_spinning_class_duration_l26_2691


namespace NUMINAMATH_CALUDE_local_taxes_in_cents_l26_2618

/-- The hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- The local tax rate as a decimal -/
def tax_rate : ℝ := 0.024

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The amount of local taxes paid in cents per hour is 60 -/
theorem local_taxes_in_cents : 
  (hourly_wage * tax_rate * cents_per_dollar : ℝ) = 60 := by sorry

end NUMINAMATH_CALUDE_local_taxes_in_cents_l26_2618


namespace NUMINAMATH_CALUDE_parabola_locus_l26_2686

/-- The locus of points from which a parabola is seen at a 45° angle -/
theorem parabola_locus (p : ℝ) (u v : ℝ) : 
  (∃ (m₁ m₂ : ℝ), 
    -- Two distinct tangent lines exist
    m₁ ≠ m₂ ∧
    -- The tangent lines touch the parabola
    (∀ (x y : ℝ), y^2 = 2*p*x → (y - v = m₁*(x - u) ∨ y - v = m₂*(x - u))) ∧
    -- The angle between the tangent lines is 45°
    (m₁ - m₂) / (1 + m₁*m₂) = 1) →
  -- The point (u, v) lies on the hyperbola
  (u + 3*p/2)^2 - v^2 = 2*p^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_locus_l26_2686


namespace NUMINAMATH_CALUDE_cyclic_trapezoid_area_l26_2643

/-- Represents a cyclic trapezoid with parallel sides a and b, where a < b -/
structure CyclicTrapezoid where
  a : ℝ
  b : ℝ
  h : a < b

/-- The area of a cyclic trapezoid given the conditions -/
def area (t : CyclicTrapezoid) : Set ℝ :=
  let t₁ := (t.a + t.b) * (t.a - Real.sqrt (2 * t.a^2 - t.b^2)) / 4
  let t₂ := (t.a + t.b) * (t.a + Real.sqrt (2 * t.a^2 - t.b^2)) / 4
  {t₁, t₂}

/-- Theorem stating that the area of the cyclic trapezoid is either t₁ or t₂ -/
theorem cyclic_trapezoid_area (t : CyclicTrapezoid) :
  ∃ A ∈ area t, A = (t.a + t.b) * (t.a - Real.sqrt (2 * t.a^2 - t.b^2)) / 4 ∨
                 A = (t.a + t.b) * (t.a + Real.sqrt (2 * t.a^2 - t.b^2)) / 4 := by
  sorry


end NUMINAMATH_CALUDE_cyclic_trapezoid_area_l26_2643


namespace NUMINAMATH_CALUDE_raffle_ticket_average_l26_2614

/-- Represents a charitable association with male and female members selling raffle tickets -/
structure CharitableAssociation where
  male_members : ℕ
  female_members : ℕ
  male_avg_tickets : ℕ
  female_avg_tickets : ℕ

/-- The overall average number of raffle tickets sold per member -/
def overall_average (ca : CharitableAssociation) : ℚ :=
  (ca.male_members * ca.male_avg_tickets + ca.female_members * ca.female_avg_tickets : ℚ) /
  (ca.male_members + ca.female_members : ℚ)

/-- Theorem stating the overall average of raffle tickets sold per member -/
theorem raffle_ticket_average (ca : CharitableAssociation) 
  (h1 : ca.female_members = 2 * ca.male_members)
  (h2 : ca.female_avg_tickets = 70)
  (h3 : ca.male_avg_tickets = 58) :
  overall_average ca = 66 := by
  sorry

end NUMINAMATH_CALUDE_raffle_ticket_average_l26_2614


namespace NUMINAMATH_CALUDE_a_sequence_square_values_l26_2621

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | (n + 3) => a (n + 2) + a (n + 1) + a n

theorem a_sequence_square_values (n : ℕ) : 
  (n > 0 ∧ a (n - 1) = n^2) ↔ (n = 1 ∨ n = 9) := by
  sorry

#check a_sequence_square_values

end NUMINAMATH_CALUDE_a_sequence_square_values_l26_2621


namespace NUMINAMATH_CALUDE_robert_second_trade_l26_2620

def cards_traded_problem (padma_initial : ℕ) (robert_initial : ℕ) (padma_traded_first : ℕ) 
  (robert_traded_first : ℕ) (padma_traded_second : ℕ) (total_traded : ℕ) : Prop :=
  padma_initial = 75 ∧
  robert_initial = 88 ∧
  padma_traded_first = 2 ∧
  robert_traded_first = 10 ∧
  padma_traded_second = 15 ∧
  total_traded = 35 ∧
  total_traded = padma_traded_first + robert_traded_first + padma_traded_second + 
    (total_traded - padma_traded_first - robert_traded_first - padma_traded_second)

theorem robert_second_trade (padma_initial robert_initial padma_traded_first robert_traded_first 
  padma_traded_second total_traded : ℕ) :
  cards_traded_problem padma_initial robert_initial padma_traded_first robert_traded_first 
    padma_traded_second total_traded →
  total_traded - padma_traded_first - robert_traded_first - padma_traded_second = 25 :=
by sorry

end NUMINAMATH_CALUDE_robert_second_trade_l26_2620


namespace NUMINAMATH_CALUDE_total_cost_is_1975_l26_2623

def first_laptop_cost : ℝ := 500
def second_laptop_multiplier : ℝ := 3
def discount_rate : ℝ := 0.15
def external_hard_drive_cost : ℝ := 80
def mouse_cost : ℝ := 20

def total_cost : ℝ :=
  let second_laptop_cost := first_laptop_cost * second_laptop_multiplier
  let discounted_second_laptop_cost := second_laptop_cost * (1 - discount_rate)
  let accessories_cost := external_hard_drive_cost + mouse_cost
  first_laptop_cost + discounted_second_laptop_cost + 2 * accessories_cost

theorem total_cost_is_1975 : total_cost = 1975 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_1975_l26_2623


namespace NUMINAMATH_CALUDE_expand_product_l26_2682

theorem expand_product (x : ℝ) : (x + 3) * (x + 8) = x^2 + 11*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l26_2682


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_quadratic_equation_with_factoring_solutions_l26_2610

theorem quadratic_equation_solutions :
  (∃ x : ℝ, x^2 - 5*x + 6 = 0) ↔ (∃ x : ℝ, x = 2 ∨ x = 3) :=
by sorry

theorem quadratic_equation_with_factoring_solutions :
  (∃ x : ℝ, (x - 2)^2 = 2*(x - 3)*(x - 2)) ↔ (∃ x : ℝ, x = 2 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_quadratic_equation_with_factoring_solutions_l26_2610


namespace NUMINAMATH_CALUDE_central_number_is_ten_l26_2674

/-- A triangular grid with 10 integers -/
structure TriangularGrid :=
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 x : ℤ)

/-- The sum of all ten numbers is 43 -/
def total_sum (g : TriangularGrid) : Prop :=
  g.a1 + g.a2 + g.a3 + g.b1 + g.b2 + g.b3 + g.c1 + g.c2 + g.c3 + g.x = 43

/-- The sum of any three numbers such that any two of them are close is 11 -/
def close_sum (g : TriangularGrid) : Prop :=
  g.a1 + g.a2 + g.a3 = 11 ∧
  g.b1 + g.b2 + g.b3 = 11 ∧
  g.c1 + g.c2 + g.c3 = 11

/-- Theorem: The central number is 10 -/
theorem central_number_is_ten (g : TriangularGrid) 
  (h1 : total_sum g) (h2 : close_sum g) : g.x = 10 := by
  sorry

end NUMINAMATH_CALUDE_central_number_is_ten_l26_2674


namespace NUMINAMATH_CALUDE_walkway_and_border_area_is_912_l26_2648

/-- Represents the dimensions and layout of a garden -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed_width : ℕ
  bed_height : ℕ
  walkway_width : ℕ
  border_width : ℕ

/-- Calculates the total area of walkways and decorative border in the garden -/
def walkway_and_border_area (g : Garden) : ℕ :=
  let total_width := g.columns * g.bed_width + (g.columns + 1) * g.walkway_width + 2 * g.border_width
  let total_height := g.rows * g.bed_height + (g.rows + 1) * g.walkway_width + 2 * g.border_width
  let total_area := total_width * total_height
  let beds_area := g.rows * g.columns * g.bed_width * g.bed_height
  total_area - beds_area

/-- Theorem stating that the walkway and border area for the given garden specifications is 912 square feet -/
theorem walkway_and_border_area_is_912 :
  walkway_and_border_area ⟨4, 3, 8, 3, 2, 4⟩ = 912 := by
  sorry

end NUMINAMATH_CALUDE_walkway_and_border_area_is_912_l26_2648


namespace NUMINAMATH_CALUDE_base9_calculation_l26_2685

/-- Converts a base 10 number to its base 9 representation -/
def toBase9 (n : ℕ) : ℕ := sorry

/-- Converts a base 9 number to its base 10 representation -/
def fromBase9 (n : ℕ) : ℕ := sorry

/-- Addition in base 9 -/
def addBase9 (a b : ℕ) : ℕ := toBase9 (fromBase9 a + fromBase9 b)

/-- Subtraction in base 9 -/
def subBase9 (a b : ℕ) : ℕ := toBase9 (fromBase9 a - fromBase9 b)

theorem base9_calculation :
  subBase9 (addBase9 (addBase9 2365 1484) 782) 671 = 4170 := by sorry

end NUMINAMATH_CALUDE_base9_calculation_l26_2685


namespace NUMINAMATH_CALUDE_quadratic_integer_intersections_l26_2635

def f (m : ℕ+) (x : ℝ) : ℝ := m * x^2 + (-m - 2) * x + 2

theorem quadratic_integer_intersections (m : ℕ+) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0) →
  (f m 1 = 0 ∧ f m 2 = 0 ∧ f m 0 = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_intersections_l26_2635


namespace NUMINAMATH_CALUDE_remainder_problem_l26_2651

theorem remainder_problem (N : ℕ) : 
  (∃ R, N = 5 * 2 + R ∧ R < 5) → 
  (∃ Q, N = 4 * Q + 2) → 
  (∃ R, N = 5 * 2 + R ∧ R = 4) := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l26_2651


namespace NUMINAMATH_CALUDE_total_birds_count_l26_2600

def birds_monday : ℕ := 70

def birds_tuesday : ℕ := birds_monday / 2

def birds_wednesday : ℕ := birds_tuesday + 8

theorem total_birds_count : birds_monday + birds_tuesday + birds_wednesday = 148 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_count_l26_2600


namespace NUMINAMATH_CALUDE_max_min_sum_l26_2663

noncomputable def f (x : ℝ) : ℝ := (2 * (x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_min_sum (M m : ℝ) (hM : ∀ x, f x ≤ M) (hm : ∀ x, m ≤ f x) : M + m = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_l26_2663


namespace NUMINAMATH_CALUDE_product_selection_theorem_product_display_theorem_l26_2603

def total_products : ℕ := 10
def ineligible_products : ℕ := 2
def products_to_select : ℕ := 4
def display_positions : ℕ := 6
def gold_medal_products : ℕ := 2

def arrangement (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem product_selection_theorem :
  arrangement (total_products - ineligible_products) products_to_select = 1680 :=
sorry

theorem product_display_theorem :
  arrangement display_positions gold_medal_products *
  arrangement (total_products - gold_medal_products) (display_positions - gold_medal_products) = 50400 :=
sorry

end NUMINAMATH_CALUDE_product_selection_theorem_product_display_theorem_l26_2603


namespace NUMINAMATH_CALUDE_set_equality_implies_coefficients_l26_2693

def A : Set ℝ := {-1, 3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem set_equality_implies_coefficients (a b : ℝ) : 
  A = B a b → a = -2 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_coefficients_l26_2693


namespace NUMINAMATH_CALUDE_periodic_sine_function_l26_2690

theorem periodic_sine_function (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = Real.sin (2 * x - π / 4)) →
  a ∈ Set.Ioo 0 π →
  (∀ x, f (x + a) = f (x + 3 * a)) →
  a = π / 2 := by
sorry

end NUMINAMATH_CALUDE_periodic_sine_function_l26_2690


namespace NUMINAMATH_CALUDE_complex_modulus_l26_2624

theorem complex_modulus (z : ℂ) : z = (1 + 3*I) / (3 - I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l26_2624


namespace NUMINAMATH_CALUDE_fraction_value_l26_2625

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l26_2625


namespace NUMINAMATH_CALUDE_cubic_roots_from_quadratic_roots_l26_2629

theorem cubic_roots_from_quadratic_roots (a b c d x₁ x₂ : ℝ) :
  (x₁^2 - (a + d)*x₁ + ad - bc = 0) →
  (x₂^2 - (a + d)*x₂ + ad - bc = 0) →
  ((x₁^3)^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*(x₁^3) + (a*d - b*c)^3 = 0) ∧
  ((x₂^3)^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*(x₂^3) + (a*d - b*c)^3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_from_quadratic_roots_l26_2629


namespace NUMINAMATH_CALUDE_f_2019_equals_2_l26_2681

def f_property (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) + f (x - 2) = 2 * f 2) ∧
  (∀ x, f (x + 1) = -f (-x - 1)) ∧
  (f 1 = 2)

theorem f_2019_equals_2 (f : ℝ → ℝ) (h : f_property f) : f 2019 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_equals_2_l26_2681


namespace NUMINAMATH_CALUDE_not_divisible_by_power_of_two_l26_2626

theorem not_divisible_by_power_of_two (n : ℕ) (h : n > 1) :
  ¬(2^n ∣ 3^n + 1) :=
by sorry

end NUMINAMATH_CALUDE_not_divisible_by_power_of_two_l26_2626


namespace NUMINAMATH_CALUDE_beths_marbles_l26_2699

/-- Proves that given the conditions of Beth's marble problem, she initially had 72 marbles. -/
theorem beths_marbles (initial_per_color : ℕ) : 
  (3 * initial_per_color) - (5 + 10 + 15) = 42 → 
  3 * initial_per_color = 72 := by
  sorry

end NUMINAMATH_CALUDE_beths_marbles_l26_2699


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l26_2684

theorem arithmetic_expression_equality : 2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l26_2684


namespace NUMINAMATH_CALUDE_solutions_to_z_sixth_eq_neg_eight_l26_2666

theorem solutions_to_z_sixth_eq_neg_eight :
  {z : ℂ | z^6 = -8} = {1 + I, 1 - I, -1 + I, -1 - I} := by
  sorry

end NUMINAMATH_CALUDE_solutions_to_z_sixth_eq_neg_eight_l26_2666


namespace NUMINAMATH_CALUDE_interest_calculation_l26_2654

/-- Calculates the total interest earned on an investment --/
def total_interest_earned (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Proves that the total interest earned is approximately $563.16 --/
theorem interest_calculation : 
  let principal := 1200
  let rate := 0.08
  let time := 5
  abs (total_interest_earned principal rate time - 563.16) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l26_2654


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l26_2612

theorem repeating_decimal_sum (c d : ℕ) (h : (4 : ℚ) / 13 = 0.1 * c + 0.01 * d + 0.001 * (c + d / 10)) : c + d = 10 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l26_2612


namespace NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l26_2607

/-- Represents a convex quadrilateral ABCD with given side lengths and angle properties -/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angle_CBA_is_right : Bool
  tan_angle_ACD : ℝ

/-- Calculates the area of the quadrilateral ABCD -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific quadrilateral is 122/3 -/
theorem area_of_specific_quadrilateral :
  let q : Quadrilateral := {
    AB := 6,
    BC := 8,
    CD := 5,
    DA := 10,
    angle_CBA_is_right := true,
    tan_angle_ACD := 4/3
  }
  area q = 122/3 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l26_2607


namespace NUMINAMATH_CALUDE_sequence_bound_l26_2688

theorem sequence_bound (k : ℕ) (h_k : k > 0) : 
  (∃ (a : ℕ → ℚ), a 0 = 1 / k ∧ 
    (∀ n : ℕ, n > 0 → a n = a (n - 1) + (1 : ℚ) / (n ^ 2 : ℚ) * (a (n - 1)) ^ 2) ∧
    (∀ n : ℕ, n > 0 → a n < 1)) ↔ 
  k ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_sequence_bound_l26_2688


namespace NUMINAMATH_CALUDE_volume_of_bounded_figure_l26_2630

-- Define a cube with edge length 1
def cube : Set (Fin 3 → ℝ) := {v | ∀ i, 0 ≤ v i ∧ v i ≤ 1}

-- Define the planes through centers of adjacent sides
def planes : Set (Set (Fin 3 → ℝ)) :=
  {p | ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    p = {v | v i + v j + v k = 3/2}}

-- Define the bounded figure
def bounded_figure : Set (Fin 3 → ℝ) :=
  {v ∈ cube | ∀ p ∈ planes, v ∈ p}

-- Theorem statement
theorem volume_of_bounded_figure :
  MeasureTheory.volume bounded_figure = 1/2 := by sorry

end NUMINAMATH_CALUDE_volume_of_bounded_figure_l26_2630


namespace NUMINAMATH_CALUDE_stream_rate_calculation_l26_2694

/-- Proves that given a boat with speed 16 km/hr in still water, traveling 126 km downstream in 6 hours, the rate of the stream is 5 km/hr. -/
theorem stream_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_rate : ℝ) :
  boat_speed = 16 →
  distance = 126 →
  time = 6 →
  distance = (boat_speed + stream_rate) * time →
  stream_rate = 5 := by
sorry

end NUMINAMATH_CALUDE_stream_rate_calculation_l26_2694


namespace NUMINAMATH_CALUDE_solve_equation_l26_2673

theorem solve_equation (x y : ℝ) (h1 : (12 : ℝ)^2 * x^4 / 432 = y) (h2 : y = 432) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l26_2673


namespace NUMINAMATH_CALUDE_circle_radius_l26_2615

/-- The radius of the circle with equation x^2 - 10x + y^2 - 4y + 24 = 0 is √5 -/
theorem circle_radius (x y : ℝ) : 
  (x^2 - 10*x + y^2 - 4*y + 24 = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l26_2615


namespace NUMINAMATH_CALUDE_regular_octagon_area_1_5_sqrt_2_l26_2641

/-- The area of a regular octagon with side length s -/
noncomputable def regularOctagonArea (s : ℝ) : ℝ := 2 * s^2 * (1 + Real.sqrt 2)

theorem regular_octagon_area_1_5_sqrt_2 :
  regularOctagonArea (1.5 * Real.sqrt 2) = 9 + 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_area_1_5_sqrt_2_l26_2641


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l26_2676

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l26_2676
