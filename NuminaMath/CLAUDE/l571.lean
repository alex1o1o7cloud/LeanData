import Mathlib

namespace NUMINAMATH_CALUDE_compound_weight_l571_57155

theorem compound_weight (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 352 → moles = 8 → moles * molecular_weight = 2816 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l571_57155


namespace NUMINAMATH_CALUDE_correct_statements_count_l571_57177

/-- A circle in a plane. -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in a plane. -/
structure Line where
  point1 : Point
  point2 : Point

/-- A statement about circle geometry. -/
inductive CircleStatement
  | perpRadiusTangent : CircleStatement
  | centerPerpTangentThruPoint : CircleStatement
  | tangentPerpThruCenterPoint : CircleStatement
  | radiusEndPerpTangent : CircleStatement
  | chordTangentMidpoint : CircleStatement

/-- Determines if a circle statement is correct. -/
def isCorrectStatement (s : CircleStatement) : Bool :=
  match s with
  | CircleStatement.perpRadiusTangent => false
  | CircleStatement.centerPerpTangentThruPoint => true
  | CircleStatement.tangentPerpThruCenterPoint => true
  | CircleStatement.radiusEndPerpTangent => false
  | CircleStatement.chordTangentMidpoint => true

/-- The list of all circle statements. -/
def allStatements : List CircleStatement :=
  [CircleStatement.perpRadiusTangent,
   CircleStatement.centerPerpTangentThruPoint,
   CircleStatement.tangentPerpThruCenterPoint,
   CircleStatement.radiusEndPerpTangent,
   CircleStatement.chordTangentMidpoint]

/-- Counts the number of correct statements. -/
def countCorrectStatements (statements : List CircleStatement) : Nat :=
  statements.filter isCorrectStatement |>.length

theorem correct_statements_count :
  countCorrectStatements allStatements = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l571_57177


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l571_57192

theorem triangle_angle_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles is 180°
  b = 4 * a →              -- ratio condition
  c = 7 * a →              -- ratio condition
  a = 15 ∧ b = 60 ∧ c = 105 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l571_57192


namespace NUMINAMATH_CALUDE_two_number_difference_l571_57166

theorem two_number_difference (a b : ℕ) (h1 : a = 10 * b) (h2 : a + b = 17402) : 
  a - b = 14238 := by
sorry

end NUMINAMATH_CALUDE_two_number_difference_l571_57166


namespace NUMINAMATH_CALUDE_x_inequality_l571_57116

theorem x_inequality (x : ℝ) : (x < 0 ∧ x < 1 / (4 * x)) ↔ (-1/2 < x ∧ x < 0) :=
sorry

end NUMINAMATH_CALUDE_x_inequality_l571_57116


namespace NUMINAMATH_CALUDE_coconut_trees_per_square_meter_l571_57171

/-- Represents the coconut farm scenario -/
structure CoconutFarm where
  size : ℝ
  treesPerSquareMeter : ℝ
  coconutsPerTree : ℕ
  harvestFrequency : ℕ
  pricePerCoconut : ℝ
  earningsAfterSixMonths : ℝ

/-- Theorem stating the number of coconut trees per square meter -/
theorem coconut_trees_per_square_meter (farm : CoconutFarm)
  (h1 : farm.size = 20)
  (h2 : farm.coconutsPerTree = 6)
  (h3 : farm.harvestFrequency = 3)
  (h4 : farm.pricePerCoconut = 0.5)
  (h5 : farm.earningsAfterSixMonths = 240) :
  farm.treesPerSquareMeter = 2 := by
  sorry


end NUMINAMATH_CALUDE_coconut_trees_per_square_meter_l571_57171


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l571_57126

theorem gcd_from_lcm_and_ratio (A B : ℕ+) : 
  Nat.lcm A B = 180 → 
  (A : ℚ) / B = 2 / 5 → 
  Nat.gcd A B = 18 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l571_57126


namespace NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l571_57157

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x₀ : ℝ, f x₀ ≤ 0) ↔ (∀ x : ℝ, f x > 0) :=
by sorry

theorem cubic_inequality_negation :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l571_57157


namespace NUMINAMATH_CALUDE_halfway_point_fractions_l571_57185

theorem halfway_point_fractions : 
  (1 / 6 + 1 / 7 + 1 / 8) / 3 = 73 / 504 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_fractions_l571_57185


namespace NUMINAMATH_CALUDE_jack_jill_water_fetching_l571_57162

/-- A problem about Jack and Jill fetching water --/
theorem jack_jill_water_fetching :
  -- Tank capacity
  ∀ (tank_capacity : ℕ),
  -- Bucket capacity
  ∀ (bucket_capacity : ℕ),
  -- Jack's bucket carrying capacity
  ∀ (jack_buckets : ℕ),
  -- Jill's bucket carrying capacity
  ∀ (jill_buckets : ℕ),
  -- Number of trips Jill made
  ∀ (jill_trips : ℕ),
  -- Conditions
  tank_capacity = 600 →
  bucket_capacity = 5 →
  jack_buckets = 2 →
  jill_buckets = 1 →
  jill_trips = 30 →
  -- Conclusion: Jack's trips in the time Jill makes two trips
  ∃ (jack_trips : ℕ), jack_trips = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_jill_water_fetching_l571_57162


namespace NUMINAMATH_CALUDE_division_remainder_problem_l571_57119

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 1565)
  (h2 : divisor = 24)
  (h3 : quotient = 65) :
  dividend = divisor * quotient + 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l571_57119


namespace NUMINAMATH_CALUDE_not_divisible_by_two_l571_57138

theorem not_divisible_by_two (n : ℕ) (h_pos : n > 0) 
  (h_sum : ∃ k : ℤ, (1 : ℚ) / 2 + 1 / 3 + 1 / 5 + 1 / n = k) : 
  ¬(2 ∣ n) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_two_l571_57138


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l571_57187

/-- The area of wrapping paper required to wrap a rectangular box -/
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  l * w + 2 * l * h + 2 * w * h + 4 * h^2

/-- Theorem stating the area of wrapping paper required for a rectangular box -/
theorem wrapping_paper_area_theorem (l w h : ℝ) (h1 : l > w) (h2 : l > 0) (h3 : w > 0) (h4 : h > 0) :
  let box_base_area := l * w
  let box_side_area := 2 * (l * h + w * h)
  let corner_area := 4 * h^2
  box_base_area + box_side_area + corner_area = wrapping_paper_area l w h :=
by
  sorry

#check wrapping_paper_area_theorem

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l571_57187


namespace NUMINAMATH_CALUDE_total_markers_l571_57175

theorem total_markers (red_markers blue_markers : ℕ) : 
  red_markers = 2315 → blue_markers = 1028 → red_markers + blue_markers = 3343 := by
  sorry

end NUMINAMATH_CALUDE_total_markers_l571_57175


namespace NUMINAMATH_CALUDE_max_m_is_maximum_l571_57131

/-- The maximum value of m for which the given conditions hold --/
def max_m : ℝ := 9

/-- Condition that abc ≤ 1/4 --/
def condition_product (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c ≤ 1/4

/-- Condition that 1/a² + 1/b² + 1/c² < m --/
def condition_sum (a b c m : ℝ) : Prop :=
  1/a^2 + 1/b^2 + 1/c^2 < m

/-- Condition that a, b, c can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem stating that max_m is the maximum value satisfying all conditions --/
theorem max_m_is_maximum :
  ∀ m : ℝ, m > 0 →
  (∀ a b c : ℝ, condition_product a b c → condition_sum a b c m → can_form_triangle a b c) →
  m ≤ max_m :=
sorry

end NUMINAMATH_CALUDE_max_m_is_maximum_l571_57131


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l571_57156

theorem prime_power_divisibility (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → p ∣ a^n → p^n ∣ a^n := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l571_57156


namespace NUMINAMATH_CALUDE_glucose_solution_volume_l571_57170

/-- Given a glucose solution with a concentration of 15 grams per 100 cubic centimeters,
    prove that a volume containing 9.75 grams of glucose is 65 cubic centimeters. -/
theorem glucose_solution_volume 
  (concentration : ℝ) 
  (volume : ℝ) 
  (glucose_mass : ℝ) 
  (h1 : concentration = 15 / 100) 
  (h2 : glucose_mass = 9.75) 
  (h3 : concentration * volume = glucose_mass) : 
  volume = 65 := by
sorry

end NUMINAMATH_CALUDE_glucose_solution_volume_l571_57170


namespace NUMINAMATH_CALUDE_not_always_congruent_l571_57137

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)

-- Define the property of having two equal sides and three equal angles
def hasTwoEqualSidesThreeEqualAngles (t1 t2 : Triangle) : Prop :=
  ((t1.a = t2.a ∧ t1.b = t2.b) ∨ (t1.a = t2.a ∧ t1.c = t2.c) ∨ (t1.b = t2.b ∧ t1.c = t2.c)) ∧
  (t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ)

-- Define triangle congruence
def isCongruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem statement
theorem not_always_congruent :
  ∃ (t1 t2 : Triangle), hasTwoEqualSidesThreeEqualAngles t1 t2 ∧ ¬isCongruent t1 t2 :=
sorry

end NUMINAMATH_CALUDE_not_always_congruent_l571_57137


namespace NUMINAMATH_CALUDE_fraction_simplification_l571_57196

theorem fraction_simplification :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l571_57196


namespace NUMINAMATH_CALUDE_books_sold_at_same_price_l571_57181

/-- Proves that two books bought for a total of 360 Rs, where one book costs 210 Rs 
    and is sold at a 15% loss, while the other is sold at a 19% gain, 
    are both sold at the same price of 178.5 Rs. -/
theorem books_sold_at_same_price (total_cost book1_cost : ℚ) 
  (loss_percent gain_percent : ℚ) : 
  total_cost = 360 ∧ book1_cost = 210 ∧ loss_percent = 15 ∧ gain_percent = 19 →
  ∃ (selling_price : ℚ), 
    selling_price = book1_cost * (1 - loss_percent / 100) ∧
    selling_price = (total_cost - book1_cost) * (1 + gain_percent / 100) ∧
    selling_price = 178.5 :=
by sorry


end NUMINAMATH_CALUDE_books_sold_at_same_price_l571_57181


namespace NUMINAMATH_CALUDE_power_sum_equality_l571_57147

theorem power_sum_equality : 2^345 + 9^8 / 9^5 = 2^345 + 729 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l571_57147


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l571_57143

theorem triangle_side_lengths (a b c : ℝ) (C : ℝ) (area : ℝ) :
  a = 3 →
  C = 2 * Real.pi / 3 →
  area = 3 * Real.sqrt 3 / 4 →
  area = 1 / 2 * a * b * Real.sin C →
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
  b = 1 ∧ c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l571_57143


namespace NUMINAMATH_CALUDE_smallest_number_l571_57199

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def number_85_9 : Nat := to_decimal [5, 8] 9
def number_210_6 : Nat := to_decimal [0, 1, 2] 6
def number_1000_4 : Nat := to_decimal [0, 0, 0, 1] 4
def number_111111_2 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

theorem smallest_number :
  number_111111_2 = min number_85_9 (min number_210_6 (min number_1000_4 number_111111_2)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l571_57199


namespace NUMINAMATH_CALUDE_function_nonnegative_m_range_l571_57160

theorem function_nonnegative_m_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 ≥ 0) → -2 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_nonnegative_m_range_l571_57160


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l571_57121

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₃ = 3 and a₅ = -3, prove a₇ = -9 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h3 : a 3 = 3) 
  (h5 : a 5 = -3) : 
  a 7 = -9 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l571_57121


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l571_57194

/-- The number of candies Bobby eats per day during weekdays -/
def weekday_candies : ℕ := 2

/-- The number of candies Bobby eats per day during weekends -/
def weekend_candies : ℕ := 1

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The number of weeks it takes Bobby to finish the packets -/
def weeks : ℕ := 3

/-- The number of packets Bobby buys -/
def packets : ℕ := 2

/-- The number of candies in a packet -/
def candies_per_packet : ℕ := 18

theorem bobby_candy_consumption :
  weekday_candies * weekdays * weeks +
  weekend_candies * weekend_days * weeks =
  candies_per_packet * packets := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l571_57194


namespace NUMINAMATH_CALUDE_only_2015_could_be_base6_l571_57186

/-- Checks if a number could be represented in base 6 --/
def couldBeBase6 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (n.digits 10) → d < 6

/-- The theorem stating that among 66, 108, 732, and 2015, only 2015 could be a base-6 number --/
theorem only_2015_could_be_base6 :
  ¬(couldBeBase6 66) ∧
  ¬(couldBeBase6 108) ∧
  ¬(couldBeBase6 732) ∧
  couldBeBase6 2015 :=
by sorry

end NUMINAMATH_CALUDE_only_2015_could_be_base6_l571_57186


namespace NUMINAMATH_CALUDE_jacks_speed_l571_57189

/-- Prove Jack's speed given the conditions of the problem -/
theorem jacks_speed (initial_distance : ℝ) (christina_speed : ℝ) (lindy_speed : ℝ) (lindy_distance : ℝ) :
  initial_distance = 360 →
  christina_speed = 7 →
  lindy_speed = 12 →
  lindy_distance = 360 →
  ∃ (jack_speed : ℝ), jack_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_jacks_speed_l571_57189


namespace NUMINAMATH_CALUDE_fred_has_nine_dimes_l571_57139

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The amount of money Fred has in cents -/
def fred_money : ℕ := 90

/-- The number of dimes Fred has -/
def fred_dimes : ℕ := fred_money / dime_value

theorem fred_has_nine_dimes : fred_dimes = 9 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_nine_dimes_l571_57139


namespace NUMINAMATH_CALUDE_arithmetic_sequence_part_1_arithmetic_sequence_part_2_l571_57142

/-- An arithmetic sequence with its sum of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms

/-- Theorem for part I -/
theorem arithmetic_sequence_part_1 (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 1) (h2 : seq.S 10 = 100) :
  ∀ n : ℕ, seq.a n = 2 * n - 1 := by sorry

/-- Theorem for part II -/
theorem arithmetic_sequence_part_2 (seq : ArithmeticSequence) 
  (h : ∀ n : ℕ, seq.S n = n^2 - 6*n) :
  ∀ n : ℕ, (seq.S n + seq.a n > 2*n) ↔ (n > 7) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_part_1_arithmetic_sequence_part_2_l571_57142


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_over_four_l571_57176

theorem tan_alpha_plus_pi_over_four (α : Real) (h : Real.tan α = 2) :
  Real.tan (α + π / 4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_over_four_l571_57176


namespace NUMINAMATH_CALUDE_bottom_right_is_one_l571_57190

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two positions are adjacent --/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ q.2.val + 1 = p.2.val)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ q.1.val + 1 = p.1.val))

/-- Check if two numbers are consecutive --/
def consecutive (m n : Fin 9) : Prop :=
  m.val + 1 = n.val ∨ n.val + 1 = m.val

/-- The theorem to prove --/
theorem bottom_right_is_one (g : Grid) :
  (∀ i j k l : Fin 3, g i j ≠ g k l → (i, j) ≠ (k, l)) →
  (∀ i j k l : Fin 3, consecutive (g i j) (g k l) → adjacent (i, j) (k, l)) →
  (g 0 0).val + (g 0 2).val + (g 2 0).val + (g 2 2).val = 24 →
  (g 1 1).val + (g 0 1).val + (g 1 0).val + (g 1 2).val + (g 2 1).val = 25 →
  (g 2 2).val = 1 := by
  sorry

end NUMINAMATH_CALUDE_bottom_right_is_one_l571_57190


namespace NUMINAMATH_CALUDE_inequality_proof_l571_57174

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b^2 / a + a^2 / b ≥ a + b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l571_57174


namespace NUMINAMATH_CALUDE_arithmetic_sequence_21st_term_l571_57127

/-- Given an arithmetic sequence with first term 11 and common difference -3,
    prove that the 21st term is -49. -/
theorem arithmetic_sequence_21st_term :
  let a : ℕ → ℤ := λ n => 11 + (n - 1) * (-3)
  a 21 = -49 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_21st_term_l571_57127


namespace NUMINAMATH_CALUDE_givenPointInFirstQuadrant_l571_57141

/-- A point in the Cartesian coordinate system. -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant in the Cartesian coordinate system. -/
def isInFirstQuadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point (3,2) in the Cartesian coordinate system. -/
def givenPoint : CartesianPoint :=
  { x := 3, y := 2 }

/-- Theorem stating that the given point (3,2) lies in the first quadrant. -/
theorem givenPointInFirstQuadrant : isInFirstQuadrant givenPoint := by
  sorry

end NUMINAMATH_CALUDE_givenPointInFirstQuadrant_l571_57141


namespace NUMINAMATH_CALUDE_bird_feeding_problem_l571_57169

/-- Given the following conditions:
    - There are 6 baby birds
    - Papa bird caught 9 worms
    - Mama bird caught 13 worms
    - 2 worms were stolen from Mama bird
    - Mama bird needs to catch 34 more worms
    - The worms are needed for 3 days
    Prove that each baby bird needs 3 worms per day. -/
theorem bird_feeding_problem (
  num_babies : ℕ)
  (papa_worms : ℕ)
  (mama_worms : ℕ)
  (stolen_worms : ℕ)
  (additional_worms : ℕ)
  (num_days : ℕ)
  (h1 : num_babies = 6)
  (h2 : papa_worms = 9)
  (h3 : mama_worms = 13)
  (h4 : stolen_worms = 2)
  (h5 : additional_worms = 34)
  (h6 : num_days = 3) :
  (papa_worms + mama_worms - stolen_worms + additional_worms) / (num_babies * num_days) = 3 := by
  sorry

#eval (9 + 13 - 2 + 34) / (6 * 3)  -- This should output 3

end NUMINAMATH_CALUDE_bird_feeding_problem_l571_57169


namespace NUMINAMATH_CALUDE_three_factor_numbers_product_l571_57129

theorem three_factor_numbers_product (x y z : ℕ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  (∃ p₁ : ℕ, Prime p₁ ∧ x = p₁^2) →
  (∃ p₂ : ℕ, Prime p₂ ∧ y = p₂^2) →
  (∃ p₃ : ℕ, Prime p₃ ∧ z = p₃^2) →
  (Nat.card {d : ℕ | d ∣ x} = 3) →
  (Nat.card {d : ℕ | d ∣ y} = 3) →
  (Nat.card {d : ℕ | d ∣ z} = 3) →
  Nat.card {d : ℕ | d ∣ (x^2 * y^3 * z^4)} = 315 := by
sorry

end NUMINAMATH_CALUDE_three_factor_numbers_product_l571_57129


namespace NUMINAMATH_CALUDE_complex_equation_solution_l571_57111

theorem complex_equation_solution (m n : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : m / (1 + i) = 1 - n * i) : 
  m = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l571_57111


namespace NUMINAMATH_CALUDE_min_sum_distances_l571_57188

open Real

/-- The minimum sum of distances between four points in a Cartesian plane -/
theorem min_sum_distances :
  let A : ℝ × ℝ := (-2, -3)
  let B : ℝ × ℝ := (4, -1)
  let C : ℝ → ℝ × ℝ := λ m ↦ (m, 0)
  let D : ℝ → ℝ × ℝ := λ n ↦ (n, n)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (m n : ℝ), ∀ (m' n' : ℝ),
    distance A B + distance B (C m) + distance (C m) (D n) + distance (D n) A ≤
    distance A B + distance B (C m') + distance (C m') (D n') + distance (D n') A ∧
    distance A B + distance B (C m) + distance (C m) (D n) + distance (D n) A = 58 + 2 * Real.sqrt 10 :=
by sorry


end NUMINAMATH_CALUDE_min_sum_distances_l571_57188


namespace NUMINAMATH_CALUDE_water_needed_for_recipe_l571_57149

/-- Represents the ratio of ingredients in the fruit punch recipe -/
structure PunchRatio where
  water : ℕ
  orange : ℕ
  lemon : ℕ

/-- Calculates the amount of water needed for a given punch recipe and total volume -/
def water_needed (ratio : PunchRatio) (total_gallons : ℚ) (quarts_per_gallon : ℕ) : ℚ :=
  let total_parts := ratio.water + ratio.orange + ratio.lemon
  let water_fraction := ratio.water / total_parts
  water_fraction * total_gallons * quarts_per_gallon

/-- Proves that the amount of water needed for the given recipe and volume is 15/2 quarts -/
theorem water_needed_for_recipe : 
  let recipe := PunchRatio.mk 5 2 1
  let total_gallons := 3
  let quarts_per_gallon := 4
  water_needed recipe total_gallons quarts_per_gallon = 15/2 := by
  sorry


end NUMINAMATH_CALUDE_water_needed_for_recipe_l571_57149


namespace NUMINAMATH_CALUDE_labourer_income_l571_57100

/-- Proves that the monthly income of a labourer is 69 given the described conditions -/
theorem labourer_income (
  avg_expenditure_6months : ℝ)
  (reduced_monthly_expense : ℝ)
  (savings : ℝ)
  (h1 : avg_expenditure_6months = 70)
  (h2 : reduced_monthly_expense = 60)
  (h3 : savings = 30)
  : ∃ (monthly_income : ℝ),
    monthly_income = 69 ∧
    6 * monthly_income < 6 * avg_expenditure_6months ∧
    4 * monthly_income = 4 * reduced_monthly_expense + (6 * avg_expenditure_6months - 6 * monthly_income) + savings :=
by
  sorry

end NUMINAMATH_CALUDE_labourer_income_l571_57100


namespace NUMINAMATH_CALUDE_tangent_length_to_circle_l571_57132

/-- The length of the tangent segment from the origin to the circle passing through 
    the points (2,3), (4,6), and (3,9) is 3√5. -/
theorem tangent_length_to_circle : 
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (4, 6)
  let C : ℝ × ℝ := (3, 9)
  let O : ℝ × ℝ := (0, 0)
  ∃ (circle : Set (ℝ × ℝ)) (T : ℝ × ℝ),
    A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧
    T ∈ circle ∧
    (∀ P ∈ circle, dist O P ≥ dist O T) ∧
    dist O T = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_to_circle_l571_57132


namespace NUMINAMATH_CALUDE_john_annual_profit_l571_57158

def annual_profit (num_subletters : ℕ) (subletter_rent : ℕ) (monthly_expense : ℕ) (months_per_year : ℕ) : ℕ :=
  (num_subletters * subletter_rent - monthly_expense) * months_per_year

theorem john_annual_profit :
  annual_profit 3 400 900 12 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_profit_l571_57158


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l571_57151

theorem smallest_solution_quadratic (x : ℝ) : 
  (2 * x^2 + 30 * x - 84 = x * (x + 15)) → x ≥ -28 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l571_57151


namespace NUMINAMATH_CALUDE_ordering_abc_l571_57152

theorem ordering_abc :
  let a : ℝ := 31/32
  let b : ℝ := Real.cos (1/4)
  let c : ℝ := 4 * Real.sin (1/4)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l571_57152


namespace NUMINAMATH_CALUDE_gate_ticket_price_l571_57182

/-- The price of plane tickets bought at the gate -/
def gate_price : ℝ := 200

/-- The number of people who pre-bought tickets -/
def pre_bought_count : ℕ := 20

/-- The price of pre-bought tickets -/
def pre_bought_price : ℝ := 155

/-- The number of people who bought tickets at the gate -/
def gate_count : ℕ := 30

/-- The additional amount paid in total by those who bought at the gate -/
def additional_gate_cost : ℝ := 2900

theorem gate_ticket_price :
  gate_price * gate_count = pre_bought_price * pre_bought_count + additional_gate_cost :=
by sorry

end NUMINAMATH_CALUDE_gate_ticket_price_l571_57182


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l571_57112

theorem gold_coin_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 13 * k + 3) → 
  n < 150 → 
  (∀ m : ℕ, (∃ j : ℕ, m = 13 * j + 3) → m < 150 → m ≤ n) → 
  n = 146 := by
sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l571_57112


namespace NUMINAMATH_CALUDE_same_university_probability_l571_57195

theorem same_university_probability (n : ℕ) (h : n = 5) :
  let total_outcomes := n * n
  let favorable_outcomes := n
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_university_probability_l571_57195


namespace NUMINAMATH_CALUDE_max_value_theorem_l571_57178

theorem max_value_theorem (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a / b + b / c + c / d + d / a = 4) (h_prod : a * c = b * d) :
  ∃ (max : ℝ), max = -12 ∧ ∀ (x : ℝ), x ≤ max ∧ (∃ (a' b' c' d' : ℝ), 
    a' / b' + b' / c' + c' / d' + d' / a' = 4 ∧ 
    a' * c' = b' * d' ∧
    x = a' / c' + b' / d' + c' / a' + d' / b') :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l571_57178


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l571_57153

theorem similar_triangle_perimeter (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a = 24) (h4 : b = 12) 
  (h5 : a = b * 2) (c : ℝ) (h6 : c = 30) :
  let scale := c / b
  let new_a := a * scale
  let new_b := b * scale
  2 * new_a + new_b = 150 := by sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l571_57153


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l571_57167

theorem geometric_sequence_problem (a : ℝ) : 
  a > 0 ∧ 
  (∃ r : ℝ, 140 * r = a ∧ a * r = 45 / 28) →
  a = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l571_57167


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l571_57115

-- Define a function to determine if an angle is in the first quadrant
def is_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2

-- Define a function to determine if an angle is in the first or third quadrant
def is_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi < α ∧ α < k * Real.pi + Real.pi / 2

-- Theorem statement
theorem half_angle_quadrant (α : ℝ) :
  is_first_quadrant α → is_first_or_third_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l571_57115


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l571_57179

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 8*a*x + 21 < 0 ↔ -7 < x ∧ x < -1) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l571_57179


namespace NUMINAMATH_CALUDE_problem_solution_l571_57124

theorem problem_solution (a b c : ℚ) 
  (sum_condition : a + b + c = 200)
  (equal_condition : a + 10 = b - 10 ∧ b - 10 = 10 * c) : 
  b = 2210 / 21 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l571_57124


namespace NUMINAMATH_CALUDE_triangle_properties_l571_57140

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.a^2 - t.c^2 - 1/2 * t.b * t.c = t.a * t.b * Real.cos t.C ∧
  t.a = 2 * Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.A = 2 * Real.pi / 3 ∧
  4 * Real.sqrt 3 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 4 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l571_57140


namespace NUMINAMATH_CALUDE_policeman_can_reach_gangster_side_l571_57107

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length s -/
structure Square (s : ℝ) where
  center : Point
  vertex : Point

/-- Represents the maximum speeds of the policeman and gangster -/
structure Speeds where
  policeman : ℝ
  gangster : ℝ

/-- Theorem stating that the policeman can always reach the same side as the gangster -/
theorem policeman_can_reach_gangster_side (s : ℝ) (square : Square s) (speeds : Speeds) :
  s > 0 ∧
  square.center = Point.mk (s/2) (s/2) ∧
  (square.vertex = Point.mk 0 0 ∨ square.vertex = Point.mk s 0 ∨
   square.vertex = Point.mk 0 s ∨ square.vertex = Point.mk s s) ∧
  speeds.gangster = 2.9 * speeds.policeman →
  ∃ (t : ℝ), t > 0 ∧ 
    ∃ (p : Point), (p.x = 0 ∨ p.x = s ∨ p.y = 0 ∨ p.y = s) ∧
      (p.x - square.center.x)^2 + (p.y - square.center.y)^2 ≤ (speeds.policeman * t)^2 ∧
      ((p.x - square.vertex.x)^2 + (p.y - square.vertex.y)^2 ≤ (speeds.gangster * t)^2 ∨
       (p.x - square.vertex.x)^2 + (p.y - square.vertex.y)^2 = (s * speeds.gangster * t)^2) :=
by sorry

end NUMINAMATH_CALUDE_policeman_can_reach_gangster_side_l571_57107


namespace NUMINAMATH_CALUDE_teachers_present_l571_57163

/-- The number of teachers present in a program --/
def num_teachers (parents pupils total : ℕ) : ℕ :=
  total - (parents + pupils)

/-- Theorem: Given 73 parents, 724 pupils, and 1541 total people,
    there were 744 teachers present in the program --/
theorem teachers_present :
  num_teachers 73 724 1541 = 744 := by
  sorry

end NUMINAMATH_CALUDE_teachers_present_l571_57163


namespace NUMINAMATH_CALUDE_exist_similar_triangles_same_color_l571_57148

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorFunction : Point → Color := sorry

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define similarity between triangles
def areSimilar (t1 t2 : Triangle) (ratio : ℝ) : Prop := sorry

-- Define the main theorem
theorem exist_similar_triangles_same_color :
  ∃ (t1 t2 : Triangle) (color : Color),
    areSimilar t1 t2 1995 ∧
    colorFunction t1.a = color ∧
    colorFunction t1.b = color ∧
    colorFunction t1.c = color ∧
    colorFunction t2.a = color ∧
    colorFunction t2.b = color ∧
    colorFunction t2.c = color := by
  sorry

end NUMINAMATH_CALUDE_exist_similar_triangles_same_color_l571_57148


namespace NUMINAMATH_CALUDE_painting_club_teams_l571_57144

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem painting_club_teams (n : ℕ) (h : n = 7) : 
  choose n 4 * choose (n - 4) 2 = 105 :=
by sorry

end NUMINAMATH_CALUDE_painting_club_teams_l571_57144


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l571_57191

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l571_57191


namespace NUMINAMATH_CALUDE_largest_divisor_of_cube_divisible_by_127_l571_57180

theorem largest_divisor_of_cube_divisible_by_127 (n : ℕ+) 
  (h : 127 ∣ n^3) : 
  ∀ m : ℕ+, m ∣ n → m ≤ 127 := by
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_cube_divisible_by_127_l571_57180


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l571_57184

/-- Proves that for a right circular cone with base radius r and altitude h, 
    and a sphere with radius r, if the volume of the cone is one-third that 
    of the sphere, then the ratio h/r = 4/3 -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * h) = (1 / 3 * (4 / 3 * π * r^3)) → h / r = 4 / 3 := by
  sorry

#check cone_sphere_ratio

end NUMINAMATH_CALUDE_cone_sphere_ratio_l571_57184


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l571_57123

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Valid angle measures
  A + B + C = π →  -- Angle sum in a triangle
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A →  -- Given condition
  b^2 = 3 * a * c →  -- Given condition
  A = π/12 ∨ A = 7*π/12 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l571_57123


namespace NUMINAMATH_CALUDE_cubic_root_sum_l571_57122

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 6*p - 3 = 0 →
  q^3 - 8*q^2 + 6*q - 3 = 0 →
  r^3 - 8*r^2 + 6*r - 3 = 0 →
  p/(q*r-1) + q/(p*r-1) + r/(p*q-1) = -14 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l571_57122


namespace NUMINAMATH_CALUDE_min_floor_sum_l571_57108

theorem min_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + b^2 + c^2 = a*b*c) : 
  ⌊(a^2 + b^2) / c⌋ + ⌊(b^2 + c^2) / a⌋ + ⌊(c^2 + a^2) / b⌋ ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_floor_sum_l571_57108


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_3x_l571_57102

theorem factorization_x_squared_minus_3x (x : ℝ) : x^2 - 3*x = x*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_3x_l571_57102


namespace NUMINAMATH_CALUDE_score_difference_is_3_4_l571_57130

-- Define the score distribution
def score_distribution : List (ℝ × ℝ) := [
  (60, 0.15),
  (75, 0.20),
  (88, 0.25),
  (92, 0.10),
  (98, 0.30)
]

-- Define the mean score
def mean_score : ℝ := (score_distribution.map (λ (score, freq) => score * freq)).sum

-- Define the median score
def median_score : ℝ := 88

-- Theorem statement
theorem score_difference_is_3_4 :
  |median_score - mean_score| = 3.4 := by sorry

end NUMINAMATH_CALUDE_score_difference_is_3_4_l571_57130


namespace NUMINAMATH_CALUDE_jamie_remaining_capacity_l571_57145

/-- Jamie's bathroom limit in ounces -/
def bathroom_limit : ℕ := 32

/-- Amount of milk Jamie consumed in ounces -/
def milk_consumed : ℕ := 8

/-- Amount of grape juice Jamie consumed in ounces -/
def grape_juice_consumed : ℕ := 16

/-- Total amount of liquid Jamie consumed before the test -/
def total_consumed : ℕ := milk_consumed + grape_juice_consumed

/-- Theorem: Jamie can drink 8 ounces during the test before needing the bathroom -/
theorem jamie_remaining_capacity : bathroom_limit - total_consumed = 8 := by
  sorry

end NUMINAMATH_CALUDE_jamie_remaining_capacity_l571_57145


namespace NUMINAMATH_CALUDE_impossible_arrangement_l571_57173

/-- Represents a 3x3 grid of digits -/
def Grid := Fin 3 → Fin 3 → Fin 4

/-- The set of digits used in the grid -/
def Digits : Finset (Fin 4) := {0, 1, 2, 3}

/-- Checks if a row contains three different digits -/
def row_valid (g : Grid) (i : Fin 3) : Prop :=
  (Finset.card {g i 0, g i 1, g i 2}) = 3

/-- Checks if a column contains three different digits -/
def col_valid (g : Grid) (j : Fin 3) : Prop :=
  (Finset.card {g 0 j, g 1 j, g 2 j}) = 3

/-- Checks if the main diagonal contains three different digits -/
def main_diag_valid (g : Grid) : Prop :=
  (Finset.card {g 0 0, g 1 1, g 2 2}) = 3

/-- Checks if the anti-diagonal contains three different digits -/
def anti_diag_valid (g : Grid) : Prop :=
  (Finset.card {g 0 2, g 1 1, g 2 0}) = 3

/-- Checks if the grid is valid according to all conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ i : Fin 3, row_valid g i) ∧
  (∀ j : Fin 3, col_valid g j) ∧
  main_diag_valid g ∧
  anti_diag_valid g

theorem impossible_arrangement : ¬∃ (g : Grid), valid_grid g := by
  sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l571_57173


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l571_57118

theorem average_of_a_and_b (a b c : ℝ) : 
  (4 + 6 + 8 + 12 + a + b + c) / 7 = 20 →
  a + b + c = 3 * ((4 + 6 + 8) / 3) →
  (a + b) / 2 = (18 - c) / 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l571_57118


namespace NUMINAMATH_CALUDE_sequence_a_properties_l571_57120

def sequence_a (n : ℕ) : ℕ := sorry

theorem sequence_a_properties :
  (∀ n : ℕ, ∃ s t : ℕ, s < t ∧ sequence_a n = 2^s + 2^t) ∧
  (∀ n m : ℕ, n < m → sequence_a n < sequence_a m) ∧
  sequence_a 5 = 10 ∧
  (∃ n : ℕ, sequence_a n = 16640 ∧ n = 100) :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_properties_l571_57120


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l571_57133

/-- The slope of a chord on an ellipse, given its midpoint -/
theorem ellipse_chord_slope (x₁ x₂ y₁ y₂ : ℝ) :
  (x₁^2 / 16 + y₁^2 / 9 = 1) →
  (x₂^2 / 16 + y₂^2 / 9 = 1) →
  ((x₁ + x₂) / 2 = 1) →
  ((y₁ + y₂) / 2 = 2) →
  (y₁ - y₂) / (x₁ - x₂) = -9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l571_57133


namespace NUMINAMATH_CALUDE_correct_answer_after_resolving_errors_l571_57117

theorem correct_answer_after_resolving_errors 
  (incorrect_divisor : ℝ)
  (correct_divisor : ℝ)
  (incorrect_answer : ℝ)
  (subtracted_value : ℝ)
  (should_add_value : ℝ)
  (h1 : incorrect_divisor = 63.5)
  (h2 : correct_divisor = 36.2)
  (h3 : incorrect_answer = 24)
  (h4 : subtracted_value = 12)
  (h5 : should_add_value = 8) :
  ∃ (correct_answer : ℝ), abs (correct_answer - 42.98) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_correct_answer_after_resolving_errors_l571_57117


namespace NUMINAMATH_CALUDE_initial_nails_l571_57165

theorem initial_nails (found_nails : ℕ) (nails_to_buy : ℕ) (total_nails : ℕ) 
  (h1 : found_nails = 144)
  (h2 : nails_to_buy = 109)
  (h3 : total_nails = 500)
  : total_nails = found_nails + nails_to_buy + 247 := by
  sorry

end NUMINAMATH_CALUDE_initial_nails_l571_57165


namespace NUMINAMATH_CALUDE_expression_simplification_l571_57104

theorem expression_simplification (b : ℝ) : ((3 * b + 6) - 5 * b) / 3 = -2/3 * b + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l571_57104


namespace NUMINAMATH_CALUDE_min_sum_squares_distances_l571_57135

/-- An isosceles right triangle with leg length a -/
structure IsoscelesRightTriangle (a : ℝ) :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (legs_length : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = a ∧ B.2 = 0 ∧ C.1 = 0 ∧ C.2 = a)
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)

/-- The sum of squares of distances from a point to the vertices of the triangle -/
def sum_of_squares_distances (a : ℝ) (triangle : IsoscelesRightTriangle a) (point : ℝ × ℝ) : ℝ :=
  (point.1 - triangle.A.1)^2 + (point.2 - triangle.A.2)^2 +
  (point.1 - triangle.B.1)^2 + (point.2 - triangle.B.2)^2 +
  (point.1 - triangle.C.1)^2 + (point.2 - triangle.C.2)^2

/-- The theorem stating the minimum point and value -/
theorem min_sum_squares_distances (a : ℝ) (triangle : IsoscelesRightTriangle a) :
  ∃ (min_point : ℝ × ℝ),
    (∀ (point : ℝ × ℝ), sum_of_squares_distances a triangle min_point ≤ sum_of_squares_distances a triangle point) ∧
    min_point = (a/3, a/3) ∧
    sum_of_squares_distances a triangle min_point = (4*a^2)/3 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_distances_l571_57135


namespace NUMINAMATH_CALUDE_four_times_hash_58_l571_57110

-- Define the function #
def hash (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem four_times_hash_58 : hash (hash (hash (hash 58))) = 11.8688 := by
  sorry

end NUMINAMATH_CALUDE_four_times_hash_58_l571_57110


namespace NUMINAMATH_CALUDE_tea_set_problem_l571_57128

/-- Tea Set Problem -/
theorem tea_set_problem (cost_A cost_B : ℕ) 
  (h1 : cost_A + 2 * cost_B = 250)
  (h2 : 3 * cost_A + 4 * cost_B = 600)
  (h3 : ∀ a b : ℕ, a + b = 80 → 108 * a + 60 * b ≤ 6240)
  (h4 : ∀ a b : ℕ, a + b = 80 → 30 * a + 20 * b ≤ 1900)
  : ∃ a b : ℕ, a + b = 80 ∧ 30 * a + 20 * b = 1900 :=
sorry

end NUMINAMATH_CALUDE_tea_set_problem_l571_57128


namespace NUMINAMATH_CALUDE_f_one_root_iff_a_in_set_l571_57197

/-- A quadratic function f(x) = ax^2 + (3-a)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (3 - a) * x + 1

/-- The condition for a quadratic function to have exactly one root -/
def has_one_root (a : ℝ) : Prop :=
  (a = 0 ∧ ∃! x, f a x = 0) ∨
  (a ≠ 0 ∧ (3 - a)^2 - 4*a = 0)

/-- The theorem stating that f has only one common point with the x-axis iff a ∈ {0, 1, 9} -/
theorem f_one_root_iff_a_in_set :
  ∀ a : ℝ, has_one_root a ↔ a ∈ ({0, 1, 9} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_f_one_root_iff_a_in_set_l571_57197


namespace NUMINAMATH_CALUDE_RS_length_l571_57161

-- Define the triangle RFS
structure Triangle :=
  (R F S : ℝ × ℝ)

-- Define the given lengths
def FD : ℝ := 5
def DR : ℝ := 8
def FR : ℝ := 6
def FS : ℝ := 9

-- Define the angles
def angle_RFS (t : Triangle) : ℝ := sorry
def angle_FDR : ℝ := sorry

-- State the theorem
theorem RS_length (t : Triangle) :
  angle_RFS t = angle_FDR →
  FR = 6 →
  FS = 9 →
  ∃ (RS : ℝ), abs (RS - 10.25) < 0.01 := by sorry

end NUMINAMATH_CALUDE_RS_length_l571_57161


namespace NUMINAMATH_CALUDE_least_faces_combined_l571_57101

/-- Represents a fair die with faces numbered from 1 to n -/
structure Die (n : ℕ) where
  faces : Fin n → ℕ
  is_fair : ∀ i : Fin n, faces i = i.val + 1

/-- Represents a pair of dice -/
structure DicePair (a b : ℕ) where
  die1 : Die a
  die2 : Die b
  die2_numbering : ∀ i : Fin b, die2.faces i = 2 * i.val + 2

/-- Probability of rolling a specific sum with a pair of dice -/
def prob_sum (d : DicePair a b) (sum : ℕ) : ℚ :=
  (Fintype.card {(i, j) : Fin a × Fin b | d.die1.faces i + d.die2.faces j = sum} : ℚ) / (a * b)

/-- The main theorem stating the least possible number of faces on two dice combined -/
theorem least_faces_combined (a b : ℕ) (d : DicePair a b) :
  (prob_sum d 8 = 2 * prob_sum d 12) →
  (prob_sum d 13 = 2 * prob_sum d 8) →
  a + b ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_least_faces_combined_l571_57101


namespace NUMINAMATH_CALUDE_infinitely_many_integers_l571_57136

theorem infinitely_many_integers (k : ℕ) (hk : k > 1) :
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a - 1) / b + (b - 1) / c + (c - 1) / a = k + 1 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_integers_l571_57136


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l571_57103

/-- Given the initial conditions of a pricing problem, prove that the profit-maximizing price is 95 yuan. -/
theorem profit_maximizing_price 
  (initial_cost : ℝ)
  (initial_price : ℝ)
  (initial_units : ℝ)
  (price_increase : ℝ)
  (units_decrease : ℝ)
  (h1 : initial_cost = 80)
  (h2 : initial_price = 90)
  (h3 : initial_units = 400)
  (h4 : price_increase = 1)
  (h5 : units_decrease = 20)
  : ∃ (max_price : ℝ), max_price = 95 ∧ 
    ∀ (x : ℝ), 
      (initial_price + x) * (initial_units - units_decrease * x) - initial_cost * (initial_units - units_decrease * x) ≤ 
      (initial_price + (max_price - initial_price)) * (initial_units - units_decrease * (max_price - initial_price)) - 
      initial_cost * (initial_units - units_decrease * (max_price - initial_price)) :=
by sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l571_57103


namespace NUMINAMATH_CALUDE_savings_calculation_l571_57125

/-- Calculates the amount left in savings after distributing funds to family members --/
def savings_amount (initial : ℚ) (wife_fraction : ℚ) (son1_fraction : ℚ) (son2_fraction : ℚ) : ℚ :=
  let wife_share := wife_fraction * initial
  let after_wife := initial - wife_share
  let son1_share := son1_fraction * after_wife
  let after_son1 := after_wife - son1_share
  let son2_share := son2_fraction * after_son1
  after_son1 - son2_share

/-- Theorem stating the amount left in savings after distribution --/
theorem savings_calculation :
  savings_amount 2000 (2/5) (2/5) (40/100) = 432 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l571_57125


namespace NUMINAMATH_CALUDE_circle_area_l571_57164

theorem circle_area (c : ℝ) (h : c = 18 * Real.pi) :
  ∃ r : ℝ, c = 2 * Real.pi * r ∧ Real.pi * r^2 = 81 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l571_57164


namespace NUMINAMATH_CALUDE_remaining_surface_area_l571_57198

/-- The surface area of the remaining part of a cube after cutting a smaller cube from its vertex -/
theorem remaining_surface_area (original_edge : ℝ) (small_edge : ℝ) 
  (h1 : original_edge = 9) 
  (h2 : small_edge = 2) : 
  6 * original_edge^2 - 3 * small_edge^2 + 3 * small_edge^2 = 486 :=
by sorry

end NUMINAMATH_CALUDE_remaining_surface_area_l571_57198


namespace NUMINAMATH_CALUDE_quadratic_function_example_l571_57150

theorem quadratic_function_example : ∃ (a b c : ℝ),
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (f 1 = 0) ∧ (f 5 = 0) ∧ (f 3 = 10) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_example_l571_57150


namespace NUMINAMATH_CALUDE_percent_increase_proof_l571_57114

def initial_cost : ℝ := 120000
def final_cost : ℝ := 192000

theorem percent_increase_proof :
  (final_cost - initial_cost) / initial_cost * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_proof_l571_57114


namespace NUMINAMATH_CALUDE_sphere_cube_paint_equivalence_l571_57113

theorem sphere_cube_paint_equivalence (M : ℝ) : 
  let cube_side : ℝ := 3
  let cube_surface_area : ℝ := 6 * cube_side^2
  let sphere_surface_area : ℝ := cube_surface_area
  let sphere_volume : ℝ := (M * Real.sqrt 3) / Real.sqrt Real.pi
  (∃ (r : ℝ), 
    sphere_surface_area = 4 * Real.pi * r^2 ∧ 
    sphere_volume = (4 / 3) * Real.pi * r^3) →
  M = 36 := by
sorry

end NUMINAMATH_CALUDE_sphere_cube_paint_equivalence_l571_57113


namespace NUMINAMATH_CALUDE_real_roots_quadratic_l571_57134

theorem real_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 16*k = 0) ↔ k ≤ 0 ∨ k ≥ 64 := by
sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_l571_57134


namespace NUMINAMATH_CALUDE_chair_cost_l571_57106

/-- Proves that the cost of one chair is $11 given the conditions of Nadine's garage sale purchase. -/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  ∃ (chair_cost : ℕ), 
    chair_cost * num_chairs = total_spent - table_cost ∧
    chair_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_chair_cost_l571_57106


namespace NUMINAMATH_CALUDE_power_fraction_equality_l571_57154

theorem power_fraction_equality : (2^4 * 3^2 * 5^3 * 7^2) / 11 = 80182 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l571_57154


namespace NUMINAMATH_CALUDE_exactlyOneAndTwoBlackMutuallyExclusiveNotContradictory_l571_57193

/-- Represents the outcome of drawing two balls from a bag -/
inductive DrawOutcome
| OneBOne  -- One black, one red
| TwoB     -- Two black
| TwoR     -- Two red

/-- The probability space for drawing two balls from a bag with 2 red and 3 black balls -/
def drawProbSpace : Type := DrawOutcome

/-- The event "Exactly one black ball is drawn" -/
def exactlyOneBlack (outcome : drawProbSpace) : Prop :=
  outcome = DrawOutcome.OneBOne

/-- The event "Exactly two black balls are drawn" -/
def exactlyTwoBlack (outcome : drawProbSpace) : Prop :=
  outcome = DrawOutcome.TwoB

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (e1 e2 : drawProbSpace → Prop) : Prop :=
  ∀ (outcome : drawProbSpace), ¬(e1 outcome ∧ e2 outcome)

/-- Two events are contradictory if exactly one of them must occur -/
def contradictory (e1 e2 : drawProbSpace → Prop) : Prop :=
  ∀ (outcome : drawProbSpace), e1 outcome ↔ ¬(e2 outcome)

theorem exactlyOneAndTwoBlackMutuallyExclusiveNotContradictory :
  mutuallyExclusive exactlyOneBlack exactlyTwoBlack ∧
  ¬(contradictory exactlyOneBlack exactlyTwoBlack) := by
  sorry

end NUMINAMATH_CALUDE_exactlyOneAndTwoBlackMutuallyExclusiveNotContradictory_l571_57193


namespace NUMINAMATH_CALUDE_center_of_mass_distance_three_points_l571_57159

/-- Given three material points with masses and distances from a line,
    prove the formula for the distance of their center of mass from the line. -/
theorem center_of_mass_distance_three_points
  (m₁ m₂ m₃ y₁ y₂ y₃ : ℝ)
  (hm : m₁ > 0 ∧ m₂ > 0 ∧ m₃ > 0) :
  let z := (m₁ * y₁ + m₂ * y₂ + m₃ * y₃) / (m₁ + m₂ + m₃)
  ∃ (com : ℝ), com = z ∧ 
    com * (m₁ + m₂ + m₃) = m₁ * y₁ + m₂ * y₂ + m₃ * y₃ :=
by sorry

end NUMINAMATH_CALUDE_center_of_mass_distance_three_points_l571_57159


namespace NUMINAMATH_CALUDE_repeat_three_times_divisible_l571_57172

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : Nat
  is_three_digit : value ≥ 100 ∧ value ≤ 999

/-- Represents a nine-digit number formed by repeating a three-digit number three times -/
def repeat_three_times (n : ThreeDigitNumber) : Nat :=
  1000000 * n.value + 1000 * n.value + n.value

/-- Theorem: Any nine-digit number formed by repeating a three-digit number three times is divisible by 1001001 -/
theorem repeat_three_times_divisible (n : ThreeDigitNumber) :
  (repeat_three_times n) % 1001001 = 0 := by
  sorry

end NUMINAMATH_CALUDE_repeat_three_times_divisible_l571_57172


namespace NUMINAMATH_CALUDE_pencil_count_l571_57168

theorem pencil_count (pens pencils : ℕ) : 
  (pens : ℚ) / pencils = 5 / 6 →
  pencils = pens + 8 →
  pencils = 48 :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l571_57168


namespace NUMINAMATH_CALUDE_set_forms_triangle_l571_57109

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three given lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set (6, 8, 13) can form a triangle -/
theorem set_forms_triangle : can_form_triangle 6 8 13 := by
  sorry


end NUMINAMATH_CALUDE_set_forms_triangle_l571_57109


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_l571_57105

/-- The probability of selecting at least one female student when randomly choosing 2 students
    from a group of 3 males and 1 female is equal to 1/2. -/
theorem prob_at_least_one_female (total_students : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (team_size : ℕ) (h1 : total_students = male_students + female_students) 
  (h2 : male_students = 3) (h3 : female_students = 1) (h4 : team_size = 2) :
  1 - (Nat.choose male_students team_size : ℚ) / (Nat.choose total_students team_size : ℚ) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_female_l571_57105


namespace NUMINAMATH_CALUDE_pencil_distribution_l571_57146

theorem pencil_distribution (total_pencils : Nat) (pencils_per_box : Nat) : 
  total_pencils = 48297858 → pencils_per_box = 6 → total_pencils % pencils_per_box = 0 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l571_57146


namespace NUMINAMATH_CALUDE_ian_money_left_l571_57183

/-- Calculates the amount of money Ian has left after expenses and taxes --/
def money_left (hours_worked : ℕ) (hourly_rate : ℚ) (monthly_expense : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_earnings := hours_worked * hourly_rate
  let tax := tax_rate * total_earnings
  let net_earnings := total_earnings - tax
  let amount_spent := (1/2) * net_earnings
  let remaining_after_spending := net_earnings - amount_spent
  remaining_after_spending - monthly_expense

/-- Theorem stating that Ian has $14.80 left after expenses and taxes --/
theorem ian_money_left :
  money_left 8 18 50 (1/10) = 148/10 :=
by sorry

end NUMINAMATH_CALUDE_ian_money_left_l571_57183
