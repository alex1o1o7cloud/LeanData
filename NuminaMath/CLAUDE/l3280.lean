import Mathlib

namespace remaining_balloons_l3280_328002

def initial_balloons : ℕ := 30
def balloons_given : ℕ := 16

theorem remaining_balloons : initial_balloons - balloons_given = 14 := by
  sorry

end remaining_balloons_l3280_328002


namespace polynomial_factorization_and_sum_power_l3280_328024

theorem polynomial_factorization_and_sum_power (a b : ℤ) : 
  (∀ x : ℝ, x^2 + x - 6 = (x + a) * (x + b)) → (a + b)^2023 = 1 :=
by sorry

end polynomial_factorization_and_sum_power_l3280_328024


namespace final_sum_after_transformation_l3280_328022

theorem final_sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by sorry

end final_sum_after_transformation_l3280_328022


namespace opposite_of_eight_l3280_328041

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- Theorem: The opposite of 8 is -8. -/
theorem opposite_of_eight : opposite 8 = -8 := by
  sorry

end opposite_of_eight_l3280_328041


namespace monomial_count_l3280_328017

def is_monomial (expr : String) : Bool :=
  match expr with
  | "a" => true
  | "-2ab" => true
  | "x+y" => false
  | "x^2+y^2" => false
  | "-1" => true
  | "1/2ab^2c^3" => true
  | _ => false

def expressions : List String := ["a", "-2ab", "x+y", "x^2+y^2", "-1", "1/2ab^2c^3"]

theorem monomial_count :
  (expressions.filter is_monomial).length = 4 := by sorry

end monomial_count_l3280_328017


namespace min_sum_of_cubes_for_sum_eight_l3280_328092

theorem min_sum_of_cubes_for_sum_eight :
  ∀ x y : ℝ, x + y = 8 →
  x^3 + y^3 ≥ 2 * 4^3 ∧
  (x^3 + y^3 = 2 * 4^3 ↔ x = 4 ∧ y = 4) :=
by sorry

end min_sum_of_cubes_for_sum_eight_l3280_328092


namespace correct_article_usage_l3280_328009

/-- Represents the possible articles that can be used. -/
inductive Article
  | The
  | A
  | Blank

/-- Represents a pair of articles used in the sentence. -/
structure ArticlePair where
  first : Article
  second : Article

/-- Defines the correct article usage for the given sentence. -/
def correct_usage : ArticlePair :=
  { first := Article.The, second := Article.The }

/-- Determines if a noun is specific and known. -/
def is_specific_known (noun : String) : Bool :=
  match noun with
  | "bed" => true
  | _ => false

/-- Determines if a noun is made specific by additional information. -/
def is_specific_by_info (noun : String) (info : String) : Bool :=
  match noun, info with
  | "book", "I lost last week" => true
  | _, _ => false

/-- Theorem stating that the correct article usage is "the; the" given the conditions. -/
theorem correct_article_usage
  (bed : String)
  (book : String)
  (info : String)
  (h1 : is_specific_known bed = true)
  (h2 : is_specific_by_info book info = true) :
  correct_usage = { first := Article.The, second := Article.The } :=
by sorry

end correct_article_usage_l3280_328009


namespace age_height_not_function_l3280_328006

-- Define the types for our variables
def Age := ℕ
def Height := ℝ
def Radius := ℝ
def Circumference := ℝ
def Angle := ℝ
def SineValue := ℝ
def NumSides := ℕ
def SumInteriorAngles := ℝ

-- Define the relationships as functions
def radiusToCircumference : Radius → Circumference := sorry
def angleToSine : Angle → SineValue := sorry
def sidesToInteriorAnglesSum : NumSides → SumInteriorAngles := sorry

-- Define the relationship between age and height
def ageHeightRelation : Age → Set Height := sorry

-- Theorem to prove
theorem age_height_not_function :
  ¬(∃ (f : Age → Height), ∀ a : Age, ∃! h : Height, h ∈ ageHeightRelation a) :=
sorry

end age_height_not_function_l3280_328006


namespace john_candy_count_l3280_328020

/-- Represents the number of candies each friend has -/
structure CandyCounts where
  bob : ℕ
  mary : ℕ
  john : ℕ
  sue : ℕ
  sam : ℕ

/-- The total number of candies all friends have together -/
def totalCandies : ℕ := 50

/-- The given candy counts for Bob, Mary, Sue, and Sam -/
def givenCounts : CandyCounts where
  bob := 10
  mary := 5
  john := 0  -- We don't know John's count yet
  sue := 20
  sam := 10

/-- Theorem stating that John's candy count is equal to the total minus the sum of others -/
theorem john_candy_count (c : CandyCounts) (h : c = givenCounts) :
  c.john = totalCandies - (c.bob + c.mary + c.sue + c.sam) :=
by sorry

end john_candy_count_l3280_328020


namespace factorization_equality_l3280_328015

theorem factorization_equality (a b : ℝ) : 12 * a^3 * b - 12 * a^2 * b + 3 * a * b = 3 * a * b * (2*a - 1)^2 := by
  sorry

end factorization_equality_l3280_328015


namespace second_pile_magazines_l3280_328004

/-- A sequence of 5 terms representing the number of magazines in each pile. -/
def MagazineSequence : Type := Fin 5 → ℕ

/-- The properties of the magazine sequence based on the given information. -/
def IsValidMagazineSequence (s : MagazineSequence) : Prop :=
  s 0 = 3 ∧ s 2 = 6 ∧ s 3 = 9 ∧ s 4 = 13 ∧
  ∀ i : Fin 4, s (i + 1) - s i = s 1 - s 0

/-- Theorem stating that for any valid magazine sequence, the second term (index 1) must be 3. -/
theorem second_pile_magazines (s : MagazineSequence) 
  (h : IsValidMagazineSequence s) : s 1 = 3 := by
  sorry

end second_pile_magazines_l3280_328004


namespace factorization_equality_l3280_328067

theorem factorization_equality (a : ℝ) : 
  (a^2 + a)^2 + 4*(a^2 + a) - 12 = (a - 1)*(a + 2)*(a^2 + a + 6) := by
  sorry

end factorization_equality_l3280_328067


namespace noodles_given_to_william_l3280_328039

/-- Given that Daniel initially had 54.0 noodles and was left with 42 noodles after giving some to William,
    prove that the number of noodles Daniel gave to William is 12. -/
theorem noodles_given_to_william (initial_noodles : ℝ) (remaining_noodles : ℝ) 
    (h1 : initial_noodles = 54.0) 
    (h2 : remaining_noodles = 42) : 
  initial_noodles - remaining_noodles = 12 := by
  sorry

end noodles_given_to_william_l3280_328039


namespace sum_of_coefficients_is_two_l3280_328070

theorem sum_of_coefficients_is_two 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + 
    a₉*(x - 1)^9 + a₁₀*(x - 1)^10 + a₁₁*(x - 1)^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 2 :=
by sorry

end sum_of_coefficients_is_two_l3280_328070


namespace two_amoebas_fill_time_l3280_328073

/-- The time (in minutes) it takes for amoebas to fill a bottle -/
def fill_time (initial_count : ℕ) : ℕ → ℕ
| 60 => 1  -- One amoeba fills the bottle in 60 minutes
| t => initial_count * 2^(t / 3)  -- Amoeba count at time t

/-- Theorem stating that two amoebas fill the bottle in 57 minutes -/
theorem two_amoebas_fill_time : fill_time 2 57 = fill_time 1 60 := by
  sorry

end two_amoebas_fill_time_l3280_328073


namespace intersection_distance_l3280_328077

theorem intersection_distance : ∃ (p q : ℕ+), 
  (∀ (d : ℕ+), d ∣ p ∧ d ∣ q → d = 1) ∧ 
  (∃ (x₁ x₂ : ℝ), 
    2 = x₁^2 + 2*x₁ - 2 ∧ 
    2 = x₂^2 + 2*x₂ - 2 ∧ 
    (x₂ - x₁)^2 = 20 ∧
    (x₂ - x₁)^2 * q^2 = p) ∧
  p - q = 19 :=
sorry

end intersection_distance_l3280_328077


namespace train_crossing_time_l3280_328061

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 450 ∧ train_speed_kmh = 180 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 9 := by
  sorry

end train_crossing_time_l3280_328061


namespace floor_of_pi_l3280_328080

theorem floor_of_pi : ⌊Real.pi⌋ = 3 := by sorry

end floor_of_pi_l3280_328080


namespace w_squared_value_l3280_328038

theorem w_squared_value (w : ℝ) (h : (w + 10)^2 = (4*w + 6)*(w + 5)) : w^2 = 70/3 := by
  sorry

end w_squared_value_l3280_328038


namespace point_on_y_axis_has_zero_x_coordinate_l3280_328090

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on the y-axis -/
def lies_on_y_axis (p : Point) : Prop := p.x = 0

/-- Theorem: If a point lies on the y-axis, its x-coordinate is zero -/
theorem point_on_y_axis_has_zero_x_coordinate (m n : ℝ) :
  lies_on_y_axis (Point.mk m n) → m = 0 := by
  sorry


end point_on_y_axis_has_zero_x_coordinate_l3280_328090


namespace geometric_sequence_product_l3280_328025

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 * a 6 = 6 →
  a 8 * a 10 * a 12 = 24 →
  a 5 * a 7 * a 9 = 12 := by
sorry

end geometric_sequence_product_l3280_328025


namespace largest_difference_l3280_328028

def A : ℕ := 3 * 1003^1004
def B : ℕ := 1003^1004
def C : ℕ := 1002 * 1003^1003
def D : ℕ := 3 * 1003^1003
def E : ℕ := 1003^1003
def F : ℕ := 1003^1002

theorem largest_difference :
  A - B > max (B - C) (max (C - D) (max (D - E) (E - F))) :=
by sorry

end largest_difference_l3280_328028


namespace result_units_digit_is_seven_l3280_328084

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10
  hundreds_gt_0 : hundreds > 0

/-- The original three-digit number satisfying the condition -/
def original : ThreeDigitNumber := sorry

/-- The condition that the hundreds digit is 3 more than the units digit -/
axiom hundreds_units_relation : original.hundreds = original.units + 3

/-- The reversed number -/
def reversed : ThreeDigitNumber := sorry

/-- The result of subtracting the reversed number from the original number -/
def result : Nat := 
  (100 * original.hundreds + 10 * original.tens + original.units) - 
  (100 * reversed.hundreds + 10 * reversed.tens + reversed.units)

/-- The theorem stating that the units digit of the result is 7 -/
theorem result_units_digit_is_seven : result % 10 = 7 := by sorry

end result_units_digit_is_seven_l3280_328084


namespace two_digit_multiplication_error_l3280_328081

theorem two_digit_multiplication_error (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →
  (10 ≤ b ∧ b < 100) →
  a * b = 936 →
  ((a + 40) * b = 2496 ∨ a * (b + 40) = 2496) →
  a + b = 63 :=
by sorry

end two_digit_multiplication_error_l3280_328081


namespace triangle_abc_properties_l3280_328086

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  a > c →
  (1/2) * a * c * Real.sin B = 3/2 →
  Real.cos B = 4/5 →
  b = 3 * Real.sqrt 2 →
  (a = 5 ∧ c = 1) ∧
  Real.cos (B - C) = (31 * Real.sqrt 2) / 50 := by
  sorry

end triangle_abc_properties_l3280_328086


namespace division_remainder_l3280_328095

theorem division_remainder : 
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  let dividend := 220040
  dividend % sum = 40 :=
by sorry

end division_remainder_l3280_328095


namespace twenty_nine_is_perfect_factorization_condition_equation_solution_perfect_number_condition_l3280_328050

-- Definition of perfect number
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

-- Statement 1
theorem twenty_nine_is_perfect : is_perfect_number 29 := by sorry

-- Statement 2
theorem factorization_condition (m n : ℝ) :
  (∀ x : ℝ, x^2 - 6*x + 5 = (x - m)^2 + n) → m*n = -12 := by sorry

-- Statement 3
theorem equation_solution :
  ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 5 = 0 → x + y = -1 := by sorry

-- Statement 4
theorem perfect_number_condition :
  ∃ k : ℤ, ∀ x y : ℤ, ∃ p q : ℤ, x^2 + 4*y^2 + 4*x - 12*y + k = p^2 + q^2 := by sorry

end twenty_nine_is_perfect_factorization_condition_equation_solution_perfect_number_condition_l3280_328050


namespace expression_factorization_l3280_328013

theorem expression_factorization (x : ℝ) :
  (16 * x^7 + 32 * x^5 - 9) - (4 * x^7 - 8 * x^5 + 9) = 2 * (6 * x^7 + 20 * x^5 - 9) := by
  sorry

end expression_factorization_l3280_328013


namespace convex_quadrilateral_probability_l3280_328085

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from the total chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def convex_quads : ℕ := n.choose k

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := convex_quads / total_selections

theorem convex_quadrilateral_probability :
  probability = 1 / 171 :=
sorry

end convex_quadrilateral_probability_l3280_328085


namespace quadratic_inequality_solution_l3280_328062

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 6

-- Define the solution set condition
def solution_set (a b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

-- Theorem statement
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, f a x > 4 ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = 2) ∧
  (∀ c, ∀ x, a * x^2 - (a * c + b) * x + b * c < 0 ↔ 1 < x ∧ x < 2 * c) :=
sorry

end quadratic_inequality_solution_l3280_328062


namespace modified_counting_game_l3280_328012

theorem modified_counting_game (n : ℕ) (a₁ : ℕ) (d : ℕ) (aₙ : ℕ → ℕ) :
  a₁ = 1 →
  d = 2 →
  (∀ k, aₙ k = a₁ + (k - 1) * d) →
  aₙ 53 = 105 :=
by sorry

end modified_counting_game_l3280_328012


namespace unit_vector_parallel_to_3_4_l3280_328058

def is_unit_vector (v : ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 = 1

def is_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem unit_vector_parallel_to_3_4 :
  ∃ (v : ℝ × ℝ), is_unit_vector v ∧ is_parallel v (3, 4) ∧
  (v = (3/5, 4/5) ∨ v = (-3/5, -4/5)) :=
sorry

end unit_vector_parallel_to_3_4_l3280_328058


namespace train_passing_time_symmetry_l3280_328087

theorem train_passing_time_symmetry 
  (fast_train_length slow_train_length : ℝ)
  (time_slow_passes_fast : ℝ)
  (fast_train_length_pos : 0 < fast_train_length)
  (slow_train_length_pos : 0 < slow_train_length)
  (time_slow_passes_fast_pos : 0 < time_slow_passes_fast) :
  let total_length := fast_train_length + slow_train_length
  let relative_speed := total_length / time_slow_passes_fast
  total_length / relative_speed = time_slow_passes_fast :=
by sorry

end train_passing_time_symmetry_l3280_328087


namespace triangle_inradius_l3280_328003

/-- Given a triangle with perimeter 32 cm and area 40 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : p = 32) 
  (h_area : A = 40) 
  (h_inradius : A = r * p / 2) : 
  r = 2.5 := by
sorry

end triangle_inradius_l3280_328003


namespace least_common_time_for_seven_horses_l3280_328023

def horse_times : Finset ℕ := Finset.range 12

theorem least_common_time_for_seven_horses :
  ∃ (S : Finset ℕ), S ⊆ horse_times ∧ S.card = 7 ∧
  (∀ n ∈ S, n > 0) ∧
  (∀ (T : ℕ), (∀ n ∈ S, T % n = 0) → T ≥ 420) ∧
  (∀ n ∈ S, 420 % n = 0) :=
sorry

end least_common_time_for_seven_horses_l3280_328023


namespace prime_sequence_l3280_328035

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
by sorry

end prime_sequence_l3280_328035


namespace souvenir_theorem_l3280_328042

/-- Represents the souvenirs sold at the Beijing Winter Olympics store -/
structure Souvenir where
  costA : ℕ  -- Cost price of souvenir A
  costB : ℕ  -- Cost price of souvenir B
  totalA : ℕ -- Total cost for souvenir A
  totalB : ℕ -- Total cost for souvenir B

/-- Represents the sales data for the souvenirs -/
structure SalesData where
  initPriceA : ℕ  -- Initial selling price of A
  initPriceB : ℕ  -- Initial selling price of B
  initSoldA : ℕ   -- Initial units of A sold per day
  initSoldB : ℕ   -- Initial units of B sold per day
  priceChangeA : ℤ -- Price change for A
  priceChangeB : ℤ -- Price change for B
  soldChangeA : ℕ  -- Change in units sold for A per 1 yuan price change
  soldChangeB : ℕ  -- Change in units sold for B per 1 yuan price change
  totalSold : ℕ   -- Total souvenirs sold on a certain day

/-- Theorem stating the cost prices and maximum profit -/
theorem souvenir_theorem (s : Souvenir) (d : SalesData) 
  (h1 : s.costB = s.costA + 9)
  (h2 : s.totalA = 10400)
  (h3 : s.totalB = 14000)
  (h4 : d.initPriceA = 46)
  (h5 : d.initPriceB = 45)
  (h6 : d.initSoldA = 40)
  (h7 : d.initSoldB = 80)
  (h8 : d.soldChangeA = 4)
  (h9 : d.soldChangeB = 2)
  (h10 : d.totalSold = 140) :
  s.costA = 26 ∧ s.costB = 35 ∧ 
  ∃ (profit : ℕ), profit = 2000 ∧ 
  ∀ (p : ℕ), p ≤ profit := by
    sorry

end souvenir_theorem_l3280_328042


namespace max_value_product_l3280_328027

theorem max_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 6 * y < 90) :
  x * y * (90 - 5 * x - 6 * y) ≤ 900 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 5 * x₀ + 6 * y₀ < 90 ∧ x₀ * y₀ * (90 - 5 * x₀ - 6 * y₀) = 900 :=
by sorry

end max_value_product_l3280_328027


namespace quilt_shaded_fraction_l3280_328029

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : Nat
  total_squares : Nat
  shaded_area : Rat

/-- Creates a quilt block with the given specifications -/
def create_quilt_block : QuiltBlock :=
  { size := 4,
    total_squares := 16,
    shaded_area := 2 }

/-- Calculates the fraction of the quilt that is shaded -/
def shaded_fraction (quilt : QuiltBlock) : Rat :=
  quilt.shaded_area / quilt.total_squares

theorem quilt_shaded_fraction :
  let quilt := create_quilt_block
  shaded_fraction quilt = 1 / 8 := by sorry

end quilt_shaded_fraction_l3280_328029


namespace sqrt_3_squared_times_5_to_6_l3280_328000

theorem sqrt_3_squared_times_5_to_6 : Real.sqrt (3^2 * 5^6) = 375 := by
  sorry

end sqrt_3_squared_times_5_to_6_l3280_328000


namespace inequality_solution_set_l3280_328026

theorem inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | (x - 1) * (a * x - 4) < 0} = {x : ℝ | x > 1 ∨ x < 4 / a} := by
  sorry

end inequality_solution_set_l3280_328026


namespace base_b_difference_divisibility_l3280_328037

def base_conversion (b : ℕ) : ℤ := 2 * b^3 - 2 * b^2 + b - 1

theorem base_b_difference_divisibility (b : ℕ) (h : 4 ≤ b ∧ b ≤ 8) :
  ¬(5 ∣ base_conversion b) ↔ b = 6 :=
by sorry

end base_b_difference_divisibility_l3280_328037


namespace jacob_age_l3280_328010

theorem jacob_age (maya drew peter john jacob : ℕ) 
  (h1 : drew = maya + 5)
  (h2 : peter = drew + 4)
  (h3 : john = 30)
  (h4 : john = 2 * maya)
  (h5 : jacob + 2 = (peter + 2) / 2) :
  jacob = 11 := by
  sorry

end jacob_age_l3280_328010


namespace prob_green_ball_l3280_328071

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def probGreen (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The four containers described in the problem -/
def containerA : Container := ⟨5, 7⟩
def containerB : Container := ⟨7, 3⟩
def containerC : Container := ⟨8, 2⟩
def containerD : Container := ⟨4, 6⟩

/-- The probability of selecting each container -/
def probContainer : ℚ := 1 / 4

/-- Theorem stating the probability of selecting a green ball -/
theorem prob_green_ball : 
  probContainer * probGreen containerA +
  probContainer * probGreen containerB +
  probContainer * probGreen containerC +
  probContainer * probGreen containerD = 101 / 240 := by
  sorry


end prob_green_ball_l3280_328071


namespace quadratic_solution_property_l3280_328016

theorem quadratic_solution_property (a : ℝ) : 
  a^2 - 2*a - 1 = 0 → 2*a^2 - 4*a + 2023 = 2025 := by
  sorry

end quadratic_solution_property_l3280_328016


namespace p_squared_minus_q_squared_l3280_328075

theorem p_squared_minus_q_squared (p q : ℝ) 
  (h1 : p + q = 10) 
  (h2 : p - q = 4) : 
  p^2 - q^2 = 40 := by
sorry

end p_squared_minus_q_squared_l3280_328075


namespace tangent_line_to_circle_l3280_328093

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∃ (x y : ℝ), 2*x + 2*y = r ∧ x^2 + y^2 = 2*r) →
  (∀ (x y : ℝ), 2*x + 2*y = r → x^2 + y^2 ≥ 2*r) →
  r = 16 :=
by sorry

end tangent_line_to_circle_l3280_328093


namespace gear_teeth_problem_l3280_328064

theorem gear_teeth_problem :
  ∀ (initial_teeth_1 initial_teeth_2 final_teeth_1 final_teeth_2 : ℕ),
    (initial_teeth_1 : ℚ) / initial_teeth_2 = 7 / 9 →
    final_teeth_1 = initial_teeth_1 + 3 →
    final_teeth_2 = initial_teeth_2 - 3 →
    (final_teeth_1 : ℚ) / final_teeth_2 = 3 / 1 →
    initial_teeth_1 = 9 ∧ initial_teeth_2 = 7 ∧ final_teeth_1 = 12 ∧ final_teeth_2 = 4 :=
by sorry

end gear_teeth_problem_l3280_328064


namespace ellipse_b_squared_value_l3280_328049

/-- Given an ellipse and a hyperbola with coinciding foci, prove the value of b^2 for the ellipse -/
theorem ellipse_b_squared_value (b : ℝ) : 
  (∀ x y, x^2/25 + y^2/b^2 = 1 → x^2/169 - y^2/64 = 1/36) → 
  (∃ c : ℝ, c^2 = 25 - b^2 ∧ c^2 = 233/36) →
  b^2 = 667/36 := by
sorry

end ellipse_b_squared_value_l3280_328049


namespace hall_dimension_difference_l3280_328018

/-- For a rectangular hall with width equal to half its length and area 450 sq. m,
    the difference between length and width is 15 meters. -/
theorem hall_dimension_difference (length width : ℝ) : 
  width = length / 2 →
  length * width = 450 →
  length - width = 15 := by
  sorry

end hall_dimension_difference_l3280_328018


namespace stamp_collection_problem_l3280_328079

theorem stamp_collection_problem : ∃! x : ℕ, 
  x % 2 = 1 ∧ 
  x % 3 = 1 ∧ 
  x % 5 = 3 ∧ 
  x % 9 = 7 ∧ 
  150 < x ∧ 
  x ≤ 300 ∧ 
  x = 223 := by
sorry

end stamp_collection_problem_l3280_328079


namespace find_s_value_l3280_328099

/-- Given a relationship between R, S, and T, prove that S = 3/2 when R = 18 and T = 2 -/
theorem find_s_value (k : ℝ) : 
  (2 = k * 1^2 / 8) →  -- When R = 2, S = 1, and T = 8
  (18 = k * S^2 / 2) →  -- When R = 18 and T = 2
  S = 3/2 := by sorry

end find_s_value_l3280_328099


namespace bug_shortest_distance_l3280_328036

/-- The shortest distance between two bugs moving on an equilateral triangle --/
theorem bug_shortest_distance (side_length : ℝ) (speed1 speed2 : ℝ) :
  side_length = 60 ∧ speed1 = 4 ∧ speed2 = 3 →
  ∃ (t d : ℝ),
    t = 300 / 37 ∧
    d = Real.sqrt (43200 / 37) ∧
    ∀ (t' : ℝ), t' ≥ 0 →
      (speed1 * t')^2 + (side_length - speed2 * t')^2 -
      2 * (speed1 * t') * (side_length - speed2 * t') * (1/2) ≥ d^2 :=
by sorry

end bug_shortest_distance_l3280_328036


namespace cost_price_calculation_l3280_328053

theorem cost_price_calculation (C : ℝ) : C = 400 :=
  let SP := 0.8 * C
  have selling_price : SP = 0.8 * C := by sorry
  have increased_price : SP + 100 = 1.05 * C := by sorry
  sorry

end cost_price_calculation_l3280_328053


namespace triangle_covering_theorem_l3280_328059

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A convex polygon represented by its vertices -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)

/-- Predicate to check if a triangle covers a convex polygon -/
def covers (t : Triangle) (p : ConvexPolygon) : Prop := sorry

/-- Predicate to check if two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- Predicate to check if a line is parallel to or coincident with a side of a polygon -/
def parallel_or_coincident_with_side (line : ℝ × ℝ → ℝ × ℝ → Prop) (p : ConvexPolygon) : Prop := sorry

theorem triangle_covering_theorem (ABC : Triangle) (M : ConvexPolygon) :
  covers ABC M →
  ∃ T : Triangle, congruent T ABC ∧ covers T M ∧
    ∃ side : ℝ × ℝ → ℝ × ℝ → Prop, parallel_or_coincident_with_side side M :=
by sorry

end triangle_covering_theorem_l3280_328059


namespace complex_equation_solution_l3280_328091

theorem complex_equation_solution (z : ℂ) (h : (3 + z) * Complex.I = 1) : z = -3 - Complex.I := by
  sorry

end complex_equation_solution_l3280_328091


namespace function_property_l3280_328030

theorem function_property (f : ℝ → ℝ) :
  (∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y) →
  f 400 = 4 →
  f 800 = 2 := by
  sorry

end function_property_l3280_328030


namespace inequality_proof_l3280_328046

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
by sorry

end inequality_proof_l3280_328046


namespace twenty_five_percent_less_than_hundred_l3280_328065

theorem twenty_five_percent_less_than_hundred (x : ℝ) : x + (1/4) * x = 75 → x = 60 := by
  sorry

end twenty_five_percent_less_than_hundred_l3280_328065


namespace machines_needed_for_multiple_production_l3280_328005

/-- Given that 4 machines produce x units in 6 days, prove that 4m machines are needed to produce m*x units in 6 days, where all machines work at the same constant rate. -/
theorem machines_needed_for_multiple_production 
  (x : ℝ) (m : ℝ) (rate : ℝ) (h1 : x > 0) (h2 : m > 0) (h3 : rate > 0) :
  4 * rate * 6 = x → (4 * m) * rate * 6 = m * x :=
by
  sorry

#check machines_needed_for_multiple_production

end machines_needed_for_multiple_production_l3280_328005


namespace ed_marbles_l3280_328098

theorem ed_marbles (doug_initial : ℕ) (ed_more : ℕ) (doug_lost : ℕ) : 
  doug_initial = 22 → ed_more = 5 → doug_lost = 3 →
  doug_initial + ed_more = 27 :=
by sorry

end ed_marbles_l3280_328098


namespace sum_of_coefficients_zero_l3280_328001

theorem sum_of_coefficients_zero (x y z : ℝ) :
  (λ x y z => (2*x - 3*y + z)^20) 1 1 1 = 0 :=
by sorry

end sum_of_coefficients_zero_l3280_328001


namespace savings_account_calculation_final_amount_is_690_l3280_328082

/-- Calculates the final amount in a savings account after two years with given conditions --/
theorem savings_account_calculation (initial_deposit : ℝ) (first_year_rate : ℝ) 
  (withdrawal_percentage : ℝ) (second_year_rate : ℝ) : ℝ :=
  let first_year_balance := initial_deposit * (1 + first_year_rate)
  let remaining_after_withdrawal := first_year_balance * (1 - withdrawal_percentage)
  let final_balance := remaining_after_withdrawal * (1 + second_year_rate)
  final_balance

/-- Proves that the final amount in the account is $690 given the specified conditions --/
theorem final_amount_is_690 : 
  savings_account_calculation 1000 0.20 0.50 0.15 = 690 := by
sorry

end savings_account_calculation_final_amount_is_690_l3280_328082


namespace sum_upper_bound_l3280_328051

/-- Given positive real numbers x and y satisfying 2x + 8y - xy = 0,
    the sum x + y is always less than or equal to 18. -/
theorem sum_upper_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : 2 * x + 8 * y - x * y = 0) : 
  x + y ≤ 18 := by
sorry

end sum_upper_bound_l3280_328051


namespace max_unpainted_cubes_l3280_328008

/-- Represents a 3D coordinate in a 3x3x3 cube arrangement -/
structure Coordinate where
  x : Fin 3
  y : Fin 3
  z : Fin 3

/-- Represents a cube in the 3x3x3 arrangement -/
structure Cube where
  coord : Coordinate
  painted : Bool

/-- Represents the 3x3x3 cube arrangement -/
def CubeArrangement : Type := Array Cube

/-- Checks if a cube is on the surface of the 3x3x3 arrangement -/
def isOnSurface (c : Coordinate) : Bool :=
  c.x = 0 || c.x = 2 || c.y = 0 || c.y = 2 || c.z = 0 || c.z = 2

/-- Counts the number of unpainted cubes in the arrangement -/
def countUnpaintedCubes (arr : CubeArrangement) : Nat :=
  arr.foldl (fun count cube => if !cube.painted then count + 1 else count) 0

/-- The main theorem stating the maximum number of unpainted cubes -/
theorem max_unpainted_cubes (arr : CubeArrangement) :
  arr.size = 27 → countUnpaintedCubes arr ≤ 15 := by sorry

end max_unpainted_cubes_l3280_328008


namespace pup_difference_l3280_328068

/-- Represents the number of pups each type of dog has -/
structure PupCounts where
  husky : ℕ
  pitbull : ℕ
  golden : ℕ

/-- Represents the counts of each type of dog -/
structure DogCounts where
  husky : ℕ
  pitbull : ℕ
  golden : ℕ

/-- Calculates the total number of pups -/
def totalPups (counts : DogCounts) (pupCounts : PupCounts) : ℕ :=
  counts.husky * pupCounts.husky + counts.pitbull * pupCounts.pitbull + counts.golden * pupCounts.golden

/-- Calculates the total number of adult dogs -/
def totalAdultDogs (counts : DogCounts) : ℕ :=
  counts.husky + counts.pitbull + counts.golden

theorem pup_difference (counts : DogCounts) (pupCounts : PupCounts) :
  counts.husky = 5 →
  counts.pitbull = 2 →
  counts.golden = 4 →
  pupCounts.husky = 3 →
  pupCounts.pitbull = 3 →
  pupCounts.golden = pupCounts.husky + 2 →
  totalPups counts pupCounts - totalAdultDogs counts = 30 := by
  sorry

#check pup_difference

end pup_difference_l3280_328068


namespace percentage_relation_l3280_328066

theorem percentage_relation (A B x : ℝ) (hA : A > 0) (hB : B > 0) (h : A = (x / 100) * B) : 
  x = 100 * (A / B) := by
  sorry

end percentage_relation_l3280_328066


namespace remainder_problem_l3280_328007

theorem remainder_problem (x y : ℤ) 
  (hx : x % 126 = 11) 
  (hy : y % 126 = 25) : 
  (x + y + 23) % 63 = 59 := by
  sorry

end remainder_problem_l3280_328007


namespace cube_volume_fourth_power_l3280_328055

/-- The volume of a cube with surface area 864 square units, expressed as the fourth power of its side length -/
theorem cube_volume_fourth_power (s : ℝ) (h : 6 * s^2 = 864) : s^4 = 20736 := by
  sorry

end cube_volume_fourth_power_l3280_328055


namespace lcm_gcd_relation_l3280_328083

theorem lcm_gcd_relation (n : ℕ+) : 
  Nat.lcm n.val 180 = Nat.gcd n.val 180 + 360 → n.val = 450 := by
  sorry

end lcm_gcd_relation_l3280_328083


namespace bens_daily_start_amount_l3280_328054

/-- Proves that given the conditions of Ben's savings scenario, he must start with $50 each day -/
theorem bens_daily_start_amount :
  ∀ (X : ℚ),
  (∃ (D : ℕ),
    (2 * (D * (X - 15)) + 10 = 500) ∧
    (D = 7)) →
  X = 50 := by
  sorry

end bens_daily_start_amount_l3280_328054


namespace upstream_distance_l3280_328069

/-- Proves that a man swimming downstream 16 km in 2 hours and upstream for 2 hours,
    with a speed of 6.5 km/h in still water, swims 10 km upstream. -/
theorem upstream_distance
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (still_water_speed : ℝ)
  (h_downstream_distance : downstream_distance = 16)
  (h_downstream_time : downstream_time = 2)
  (h_upstream_time : upstream_time = 2)
  (h_still_water_speed : still_water_speed = 6.5)
  : ∃ upstream_distance : ℝ,
    upstream_distance = 10 ∧
    upstream_distance = still_water_speed * upstream_time - 
      (downstream_distance / downstream_time - still_water_speed) * upstream_time :=
by
  sorry

end upstream_distance_l3280_328069


namespace petes_number_l3280_328072

theorem petes_number : ∃ x : ℝ, 4 * (2 * x + 20) = 200 ∧ x = 15 := by
  sorry

end petes_number_l3280_328072


namespace train_crossing_time_l3280_328089

/-- Given a train crossing two platforms of different lengths, prove the time taken to cross the second platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform1_length platform2_length : ℝ)
  (time1 : ℝ) 
  (h1 : train_length = 30)
  (h2 : platform1_length = 180)
  (h3 : platform2_length = 250)
  (h4 : time1 = 15)
  (h5 : (train_length + platform1_length) / time1 = (train_length + platform2_length) / (20 : ℝ)) :
  (train_length + platform2_length) / ((train_length + platform1_length) / time1) = 20 := by
  sorry

end train_crossing_time_l3280_328089


namespace probability_face_card_is_three_thirteenths_l3280_328014

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of jacks, queens, and kings in a standard deck -/
def face_cards : ℕ := 12

/-- The probability of drawing a jack, queen, or king from a standard deck -/
def probability_face_card : ℚ := face_cards / deck_size

theorem probability_face_card_is_three_thirteenths :
  probability_face_card = 3 / 13 := by sorry

end probability_face_card_is_three_thirteenths_l3280_328014


namespace complex_number_properties_l3280_328057

open Complex

theorem complex_number_properties (z₁ z₂ z : ℂ) (b : ℝ) : 
  z₁ = 1 - I ∧ 
  z₂ = 4 + 6*I ∧ 
  z = 1 + b*I ∧ 
  (z + z₁).im = 0 →
  (abs z₁ + z₂ = Complex.mk (Real.sqrt 2 + 4) 6) ∧ 
  abs z = Real.sqrt 2 := by
sorry

end complex_number_properties_l3280_328057


namespace rod_friction_coefficient_l3280_328056

noncomputable def coefficient_of_friction (initial_normal_force_ratio : ℝ) (tilt_angle : ℝ) : ℝ :=
  (1 - initial_normal_force_ratio * Real.cos tilt_angle) / (initial_normal_force_ratio * Real.sin tilt_angle)

theorem rod_friction_coefficient (initial_normal_force_ratio : ℝ) (tilt_angle : ℝ) 
  (h1 : initial_normal_force_ratio = 11)
  (h2 : tilt_angle = 80 * π / 180) :
  ∃ ε > 0, |coefficient_of_friction initial_normal_force_ratio tilt_angle - 0.17| < ε :=
sorry

end rod_friction_coefficient_l3280_328056


namespace solution_set_part_i_min_pq_part_ii_l3280_328034

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x - 3|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := by sorry

-- Part II
theorem min_pq_part_ii (m p q : ℝ) (hm : m > 0) (hp : p > 0) (hq : q > 0) :
  (∀ x, f m x ≥ 3) ∧ (∃ x, f m x = 3) ∧ (1/p + 1/(2*q) = m) →
  ∀ r s, r > 0 ∧ s > 0 ∧ 1/r + 1/(2*s) = m → p*q ≤ r*s ∧ p*q = 1/18 := by sorry

end solution_set_part_i_min_pq_part_ii_l3280_328034


namespace f_properties_l3280_328021

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x + k * x^2 + (2 * k + 1) * x

theorem f_properties (k : ℝ) :
  (k ≥ 0 → ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f k x₁ < f k x₂) ∧
  (k < 0 → ∀ x : ℝ, 0 < x → f k x ≤ -3 / (4 * k) - 2) := by
  sorry

end f_properties_l3280_328021


namespace exam_scores_difference_l3280_328033

theorem exam_scores_difference (score1 score2 : ℕ) : 
  score1 = 42 →
  score2 = 33 →
  score1 = (56 * (score1 + score2)) / 100 →
  score1 - score2 = 9 :=
by
  sorry

end exam_scores_difference_l3280_328033


namespace perimeter_after_adding_tiles_l3280_328088

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  num_tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration -/
def add_tiles (initial : TileConfiguration) (added_tiles : ℕ) : TileConfiguration :=
  { num_tiles := initial.num_tiles + added_tiles,
    perimeter := initial.perimeter } -- Placeholder, actual calculation would depend on tile placement

/-- The theorem to be proved -/
theorem perimeter_after_adding_tiles :
  ∃ (final : TileConfiguration),
    let initial : TileConfiguration := { num_tiles := 9, perimeter := 16 }
    let with_added_tiles := add_tiles initial 3
    with_added_tiles.perimeter = 18 :=
sorry

end perimeter_after_adding_tiles_l3280_328088


namespace min_value_of_function_l3280_328045

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x + 1 / (x + 1) ≥ 1 ∧ ∃ y > -1, y + 1 / (y + 1) = 1 := by
  sorry

end min_value_of_function_l3280_328045


namespace remainder_3_pow_2040_mod_11_l3280_328031

theorem remainder_3_pow_2040_mod_11 : 3^2040 % 11 = 1 := by
  sorry

end remainder_3_pow_2040_mod_11_l3280_328031


namespace base_7_representation_of_500_l3280_328047

/-- Converts a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_7_representation_of_500 :
  toBase7 500 = [1, 3, 1, 3] ∧ fromBase7 [1, 3, 1, 3] = 500 := by
  sorry

end base_7_representation_of_500_l3280_328047


namespace probability_two_present_one_absent_l3280_328011

/-- The probability of a student being absent on a given day -/
def p_absent : ℚ := 2 / 50

/-- The probability of a student being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of students we are considering -/
def n_students : ℕ := 3

/-- The number of students that should be present -/
def n_present : ℕ := 2

theorem probability_two_present_one_absent :
  (n_students.choose n_present : ℚ) * p_present ^ n_present * p_absent ^ (n_students - n_present) = 1728 / 15625 := by
  sorry

end probability_two_present_one_absent_l3280_328011


namespace max_children_theorem_l3280_328052

/-- Represents the movie theater pricing and budget scenario -/
structure MovieTheater where
  budget : ℕ
  adultTicketCost : ℕ
  childTicketCost : ℕ
  childTicketGroupDiscount : ℕ
  groupDiscountThreshold : ℕ
  snackCost : ℕ

/-- Calculates the maximum number of children that can be taken to the movies -/
def maxChildren (mt : MovieTheater) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of children is 12 with group discount -/
theorem max_children_theorem (mt : MovieTheater) 
  (h1 : mt.budget = 100)
  (h2 : mt.adultTicketCost = 12)
  (h3 : mt.childTicketCost = 6)
  (h4 : mt.childTicketGroupDiscount = 4)
  (h5 : mt.groupDiscountThreshold = 5)
  (h6 : mt.snackCost = 3) :
  maxChildren mt = 12 ∧ 
  12 * mt.childTicketGroupDiscount + 12 * mt.snackCost + mt.adultTicketCost ≤ mt.budget :=
sorry

end max_children_theorem_l3280_328052


namespace inequality_equivalence_l3280_328097

theorem inequality_equivalence (x : ℝ) :
  (3 * x - 4 < 9 - 2 * x + |x - 1|) ↔ (x < 3 ∧ x ≥ 1) := by
  sorry

end inequality_equivalence_l3280_328097


namespace arithmetic_sequence_problem_l3280_328078

-- Define the arithmetic sequence
def arithmetic_sequence (x y z : ℤ) : Prop :=
  ∃ d : ℤ, y = x + d ∧ z = y + d

-- Theorem statement
theorem arithmetic_sequence_problem (x y z w u : ℤ) 
  (h1 : arithmetic_sequence x y z)
  (h2 : x = 1370)
  (h3 : z = 1070)
  (h4 : w = -180)
  (h5 : u = -6430) :
  w^3 - u^2 + y^2 = -44200100 := by
  sorry

end arithmetic_sequence_problem_l3280_328078


namespace lice_check_time_is_three_hours_l3280_328096

/-- The total time required to check all students for lice -/
def total_check_time (kindergarteners first_graders second_graders third_graders : ℕ) 
  (check_time_per_student : ℕ) : ℚ :=
  let total_students := kindergarteners + first_graders + second_graders + third_graders
  let total_minutes := total_students * check_time_per_student
  (total_minutes : ℚ) / 60

/-- Theorem stating that the total check time for the given number of students is 3 hours -/
theorem lice_check_time_is_three_hours :
  total_check_time 26 19 20 25 2 = 3 := by
  sorry

end lice_check_time_is_three_hours_l3280_328096


namespace man_work_time_l3280_328032

theorem man_work_time (total_work : ℝ) (man_rate : ℝ) (son_rate : ℝ) 
  (h1 : man_rate + son_rate = total_work / 3)
  (h2 : son_rate = total_work / 5.25) :
  man_rate = total_work / 7 := by
sorry

end man_work_time_l3280_328032


namespace min_value_3x_4y_l3280_328043

theorem min_value_3x_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = x * y) :
  3 * x + 4 * y ≥ 25 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 3 * y = x * y ∧ 3 * x + 4 * y = 25 :=
sorry

end min_value_3x_4y_l3280_328043


namespace sqrt_difference_approximation_l3280_328074

/-- Approximation of square root of 11 -/
def sqrt11_approx : ℝ := 3.31662

/-- Approximation of square root of 6 -/
def sqrt6_approx : ℝ := 2.44948

/-- The result we want to prove is close to the actual difference -/
def result : ℝ := 0.87

/-- Theorem stating that the difference between sqrt(11) and sqrt(6) is close to 0.87 -/
theorem sqrt_difference_approximation : |Real.sqrt 11 - Real.sqrt 6 - result| < 0.005 := by
  sorry

end sqrt_difference_approximation_l3280_328074


namespace quadratic_inequality_l3280_328094

def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality (b c : ℝ) (h : f b c (-1) = f b c 3) :
  f b c 1 < c ∧ c < f b c (-1) := by sorry

end quadratic_inequality_l3280_328094


namespace largest_reciprocal_l3280_328044

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/3 → b = 2/5 → c = 1 → d = 5 → e = 1986 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end largest_reciprocal_l3280_328044


namespace min_value_sum_reciprocals_l3280_328019

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  (1 / (a + 3*b) + 1 / (b + 3*c) + 1 / (c + 3*a)) ≥ 3/4 ∧ 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ 
    1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) = 3/4 :=
by sorry

end min_value_sum_reciprocals_l3280_328019


namespace problem_solution_l3280_328060

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : a^b = b^a) (h2 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end problem_solution_l3280_328060


namespace sabrina_pencils_l3280_328076

theorem sabrina_pencils (total : ℕ) (justin_extra : ℕ) : 
  total = 50 → justin_extra = 8 →
  ∃ (sabrina : ℕ), 
    sabrina + (2 * sabrina + justin_extra) = total ∧ 
    sabrina = 14 := by
  sorry

end sabrina_pencils_l3280_328076


namespace train_length_l3280_328048

theorem train_length (crossing_time : ℝ) (speed_kmh : ℝ) : 
  crossing_time = 100 → speed_kmh = 90 → 
  crossing_time * (speed_kmh * (1000 / 3600)) = 2500 := by
  sorry

end train_length_l3280_328048


namespace seven_balls_four_boxes_l3280_328040

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 104 := by
  sorry

end seven_balls_four_boxes_l3280_328040


namespace geometric_sequence_sum_l3280_328063

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a 2 → a 1 + a 3 = 5 → a 2 + a 4 = 10 := by
  sorry

end geometric_sequence_sum_l3280_328063
