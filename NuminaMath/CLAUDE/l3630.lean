import Mathlib

namespace NUMINAMATH_CALUDE_round_trip_speed_l3630_363003

theorem round_trip_speed (x : ℝ) : 
  x > 0 →
  (2 : ℝ) / ((1 / x) + (1 / 3)) = 5 →
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_round_trip_speed_l3630_363003


namespace NUMINAMATH_CALUDE_two_cars_total_distance_l3630_363027

/-- Proves that given two cars with specified fuel efficiencies and consumption,
    the total distance driven is 1750 miles. -/
theorem two_cars_total_distance
  (efficiency1 : ℝ) (efficiency2 : ℝ) (total_consumption : ℝ) (consumption1 : ℝ)
  (h1 : efficiency1 = 25)
  (h2 : efficiency2 = 40)
  (h3 : total_consumption = 55)
  (h4 : consumption1 = 30) :
  efficiency1 * consumption1 + efficiency2 * (total_consumption - consumption1) = 1750 :=
by sorry

end NUMINAMATH_CALUDE_two_cars_total_distance_l3630_363027


namespace NUMINAMATH_CALUDE_final_score_is_94_l3630_363059

/-- Represents the scoring system for a choir competition -/
structure ScoringSystem where
  songContentWeight : Real
  singingSkillsWeight : Real
  spiritWeight : Real
  weightSum : songContentWeight + singingSkillsWeight + spiritWeight = 1

/-- Represents the scores of a participating team -/
structure TeamScores where
  songContent : Real
  singingSkills : Real
  spirit : Real

/-- Calculates the final score given a scoring system and team scores -/
def calculateFinalScore (system : ScoringSystem) (scores : TeamScores) : Real :=
  system.songContentWeight * scores.songContent +
  system.singingSkillsWeight * scores.singingSkills +
  system.spiritWeight * scores.spirit

theorem final_score_is_94 (system : ScoringSystem) (scores : TeamScores)
    (h1 : system.songContentWeight = 0.3)
    (h2 : system.singingSkillsWeight = 0.4)
    (h3 : system.spiritWeight = 0.3)
    (h4 : scores.songContent = 90)
    (h5 : scores.singingSkills = 94)
    (h6 : scores.spirit = 98) :
    calculateFinalScore system scores = 94 := by
  sorry


end NUMINAMATH_CALUDE_final_score_is_94_l3630_363059


namespace NUMINAMATH_CALUDE_overlapping_sectors_area_l3630_363000

/-- The area of the overlapping region of two sectors in a circle -/
theorem overlapping_sectors_area (r : ℝ) (θ₁ θ₂ : ℝ) (h_r : r = 10) (h_θ₁ : θ₁ = 45) (h_θ₂ : θ₂ = 90) :
  let sector_area (θ : ℝ) := (θ / 360) * π * r^2
  min (sector_area θ₁) (sector_area θ₂) = 12.5 * π :=
by sorry

end NUMINAMATH_CALUDE_overlapping_sectors_area_l3630_363000


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l3630_363030

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) (h2 : x * y = 8) : x^2 + y^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l3630_363030


namespace NUMINAMATH_CALUDE_naturalNumberDecimal_irrational_l3630_363058

/-- Represents the infinite decimal 0.1234567891011121314... -/
def naturalNumberDecimal : ℝ :=
  sorry

/-- The digits of naturalNumberDecimal after the decimal point consist of all natural numbers in order -/
axiom naturalNumberDecimal_property : ∀ n : ℕ, ∃ k : ℕ, sorry

theorem naturalNumberDecimal_irrational : Irrational naturalNumberDecimal := by
  sorry

end NUMINAMATH_CALUDE_naturalNumberDecimal_irrational_l3630_363058


namespace NUMINAMATH_CALUDE_a_squared_gt_b_squared_neither_sufficient_nor_necessary_l3630_363011

theorem a_squared_gt_b_squared_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b) ∧ ¬(∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_a_squared_gt_b_squared_neither_sufficient_nor_necessary_l3630_363011


namespace NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l3630_363092

theorem complex_exp_13pi_over_2 : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l3630_363092


namespace NUMINAMATH_CALUDE_angle_triple_complement_l3630_363017

theorem angle_triple_complement : ∃ x : ℝ, x + (180 - x) = 180 ∧ x = 3 * (180 - x) ∧ x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l3630_363017


namespace NUMINAMATH_CALUDE_proposition_all_lines_proposition_line_planes_proposition_all_planes_l3630_363026

-- Define the basic types
inductive GeometricObject
| Line
| Plane

-- Define the relationships
def perpendicular (a b : GeometricObject) : Prop := sorry
def parallel (a b : GeometricObject) : Prop := sorry

-- Define the proposition
def proposition (x y z : GeometricObject) : Prop :=
  perpendicular x y ∧ parallel y z → perpendicular x z

-- Theorem for case 1: all lines
theorem proposition_all_lines :
  ∀ x y z : GeometricObject,
  x = GeometricObject.Line ∧ 
  y = GeometricObject.Line ∧ 
  z = GeometricObject.Line →
  proposition x y z :=
sorry

-- Theorem for case 2: x is line, y and z are planes
theorem proposition_line_planes :
  ∀ x y z : GeometricObject,
  x = GeometricObject.Line ∧ 
  y = GeometricObject.Plane ∧ 
  z = GeometricObject.Plane →
  proposition x y z :=
sorry

-- Theorem for case 3: all planes
theorem proposition_all_planes :
  ∀ x y z : GeometricObject,
  x = GeometricObject.Plane ∧ 
  y = GeometricObject.Plane ∧ 
  z = GeometricObject.Plane →
  proposition x y z :=
sorry

end NUMINAMATH_CALUDE_proposition_all_lines_proposition_line_planes_proposition_all_planes_l3630_363026


namespace NUMINAMATH_CALUDE_two_cars_gas_consumption_l3630_363038

/-- Represents the gas consumption and mileage of a car for a week -/
structure CarData where
  mpg : ℝ
  gallons_consumed : ℝ
  miles_driven : ℝ

/-- Calculates the total gas consumption for two cars in a week -/
def total_gas_consumption (car1 : CarData) (car2 : CarData) : ℝ :=
  car1.gallons_consumed + car2.gallons_consumed

/-- Theorem stating the total gas consumption of two cars given specific conditions -/
theorem two_cars_gas_consumption
  (car1 : CarData)
  (car2 : CarData)
  (h1 : car1.mpg = 25)
  (h2 : car2.mpg = 40)
  (h3 : car1.gallons_consumed = 30)
  (h4 : car1.miles_driven + car2.miles_driven = 1825)
  (h5 : car1.miles_driven = car1.mpg * car1.gallons_consumed)
  (h6 : car2.miles_driven = car2.mpg * car2.gallons_consumed) :
  total_gas_consumption car1 car2 = 56.875 := by
  sorry

#eval Float.round ((25 : Float) * 30 + (1825 - 25 * 30) / 40) * 1000 / 1000

end NUMINAMATH_CALUDE_two_cars_gas_consumption_l3630_363038


namespace NUMINAMATH_CALUDE_quadratic_always_greater_than_ten_l3630_363099

theorem quadratic_always_greater_than_ten (k : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + k > 10) ↔ k > 11 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_greater_than_ten_l3630_363099


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3630_363004

theorem greatest_divisor_with_remainders : Nat.gcd (1442 - 12) (1816 - 6) = 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3630_363004


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3630_363056

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3630_363056


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l3630_363005

theorem not_all_perfect_squares (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(∃ x y z : ℕ, (2 * a^2 + b^2 + 3 = x^2) ∧ (2 * b^2 + c^2 + 3 = y^2) ∧ (2 * c^2 + a^2 + 3 = z^2)) :=
by sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l3630_363005


namespace NUMINAMATH_CALUDE_triangle_side_length_l3630_363050

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Given conditions
  c = Real.sqrt 3 →
  A = π / 4 →  -- 45° in radians
  C = π / 3 →  -- 60° in radians
  -- Law of Sines
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  -- Conclusion
  a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3630_363050


namespace NUMINAMATH_CALUDE_system_of_equations_l3630_363043

theorem system_of_equations (a b c d k : ℝ) 
  (h1 : a + b = 11)
  (h2 : b^2 + c^2 = k)
  (h3 : b + c = 9)
  (h4 : c + d = 3)
  (h5 : k > 0) :
  a + d = 5 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l3630_363043


namespace NUMINAMATH_CALUDE_factor_implies_values_l3630_363036

theorem factor_implies_values (p q : ℝ) : 
  (∃ (a b c : ℝ), (X^4 + p*X^2 + q) = (X^2 + 2*X + 5) * (a*X^2 + b*X + c)) → 
  p = 6 ∧ q = 25 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_values_l3630_363036


namespace NUMINAMATH_CALUDE_bethany_saw_80_paintings_l3630_363089

/-- The number of paintings Bethany saw at the museum -/
structure MuseumVisit where
  portraits : ℕ
  stillLifes : ℕ

/-- Bethany's visit to the museum satisfies the given conditions -/
def bethanysVisit : MuseumVisit where
  portraits := 16
  stillLifes := 4 * 16

/-- The total number of paintings Bethany saw -/
def totalPaintings (visit : MuseumVisit) : ℕ :=
  visit.portraits + visit.stillLifes

/-- Theorem stating that Bethany saw 80 paintings in total -/
theorem bethany_saw_80_paintings :
  totalPaintings bethanysVisit = 80 := by
  sorry

end NUMINAMATH_CALUDE_bethany_saw_80_paintings_l3630_363089


namespace NUMINAMATH_CALUDE_thirty_percent_more_than_hundred_l3630_363095

theorem thirty_percent_more_than_hundred (x : ℝ) : x + (1/4) * x = 130 → x = 104 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_more_than_hundred_l3630_363095


namespace NUMINAMATH_CALUDE_mom_bought_packages_l3630_363019

def shirts_per_package : ℕ := 6
def total_shirts : ℕ := 426

theorem mom_bought_packages : 
  ∃ (packages : ℕ), packages * shirts_per_package = total_shirts ∧ packages = 71 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_packages_l3630_363019


namespace NUMINAMATH_CALUDE_no_perfect_squares_l3630_363082

def sequence_x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * sequence_x (n + 1) - sequence_x n

theorem no_perfect_squares (n : ℕ) (h : n ≥ 1) :
  ¬ ∃ m : ℤ, sequence_x n = m * m := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l3630_363082


namespace NUMINAMATH_CALUDE_quadratic_function_sum_of_coefficients_l3630_363025

theorem quadratic_function_sum_of_coefficients 
  (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : (1 : ℝ) = a * (1 : ℝ)^2 + b * (1 : ℝ) - 1) : 
  a + b = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_sum_of_coefficients_l3630_363025


namespace NUMINAMATH_CALUDE_smallest_m_divisible_by_15_l3630_363020

/-- The largest prime with 2015 digits -/
def q : ℕ := sorry

theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ 15 ∣ (q^2 - m) ∧ ∀ k : ℕ, 0 < k ∧ k < m → ¬(15 ∣ (q^2 - k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_divisible_by_15_l3630_363020


namespace NUMINAMATH_CALUDE_remainder_98_35_mod_100_l3630_363010

theorem remainder_98_35_mod_100 : 98^35 ≡ -24 [ZMOD 100] := by sorry

end NUMINAMATH_CALUDE_remainder_98_35_mod_100_l3630_363010


namespace NUMINAMATH_CALUDE_domain_transformation_l3630_363083

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Icc 0 4

-- Define the domain of f(x²)
def domain_f_squared : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem domain_transformation (hf : ∀ x ∈ domain_f, f x ≠ 0) :
  {x : ℝ | f (x^2) ≠ 0} = domain_f_squared := by sorry

end NUMINAMATH_CALUDE_domain_transformation_l3630_363083


namespace NUMINAMATH_CALUDE_missing_sale_is_7562_l3630_363013

/-- Calculates the missing sale amount given sales for 5 out of 6 months and the average sale -/
def calculate_missing_sale (sale1 sale2 sale3 sale4 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale4 + sale6)

theorem missing_sale_is_7562 (sale1 sale2 sale3 sale4 sale6 average_sale : ℕ) 
  (h1 : sale1 = 7435)
  (h2 : sale2 = 7927)
  (h3 : sale3 = 7855)
  (h4 : sale4 = 8230)
  (h5 : sale6 = 5991)
  (h6 : average_sale = 7500) :
  calculate_missing_sale sale1 sale2 sale3 sale4 sale6 average_sale = 7562 := by
  sorry

#eval calculate_missing_sale 7435 7927 7855 8230 5991 7500

end NUMINAMATH_CALUDE_missing_sale_is_7562_l3630_363013


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l3630_363035

theorem shaded_area_between_circles (R : ℝ) (r : ℝ) : 
  R = 10 → r = 4 → π * R^2 - 2 * π * r^2 = 68 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l3630_363035


namespace NUMINAMATH_CALUDE_cake_distribution_l3630_363031

theorem cake_distribution (total_cakes : ℕ) (num_children : ℕ) : 
  total_cakes = 18 → num_children = 3 → 
  ∃ (oldest middle youngest : ℕ),
    oldest = (2 * total_cakes / 5 : ℕ) ∧
    middle = total_cakes / 3 ∧
    youngest = total_cakes - (oldest + middle) ∧
    oldest = 7 ∧ middle = 6 ∧ youngest = 5 := by
  sorry

#check cake_distribution

end NUMINAMATH_CALUDE_cake_distribution_l3630_363031


namespace NUMINAMATH_CALUDE_locus_and_line_equations_l3630_363033

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def F₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 36

-- Define the locus C
def C (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1 ∧ x ≠ -4

-- Define the line l
def l (x y : ℝ) : Prop := 3 * x - 2 * y - 8 = 0

-- Define the point Q
def Q : ℝ × ℝ := (2, -1)

-- Theorem statement
theorem locus_and_line_equations :
  ∃ (M : ℝ × ℝ → Prop),
    (∀ x y, M (x, y) → F₁ x y) ∧
    (∀ x y, M (x, y) → F₂ x y) ∧
    (∀ x y, C x y ↔ ∃ r > 0, M (x, y) ∧ r = 2) ∧
    (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2 ∧ Q = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :=
sorry

end NUMINAMATH_CALUDE_locus_and_line_equations_l3630_363033


namespace NUMINAMATH_CALUDE_range_of_p_or_q_range_of_a_intersection_l3630_363077

-- Define sets A, B, and C
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 3}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) : Prop := x ∈ B

-- Theorem 1: The set of x satisfying p ∨ q is equal to [-2, 5)
theorem range_of_p_or_q : {x : ℝ | p x ∨ q x} = Set.Ico (-2) 5 := by sorry

-- Theorem 2: The set of a satisfying A ∩ C = C is equal to (-∞, -4] ∪ [-1, 1/2]
theorem range_of_a_intersection : 
  {a : ℝ | A ∩ C a = C a} = Set.Iic (-4) ∪ Set.Icc (-1) (1/2) := by sorry

end NUMINAMATH_CALUDE_range_of_p_or_q_range_of_a_intersection_l3630_363077


namespace NUMINAMATH_CALUDE_square_tiling_for_n_ge_5_l3630_363046

/-- A rectangle is dominant if it is similar to a 2 × 1 rectangle -/
def DominantRectangle (r : Rectangle) : Prop := sorry

/-- A tiling of a square with n dominant rectangles -/
def SquareTiling (n : ℕ) : Prop := sorry

/-- Theorem: For all integers n ≥ 5, it is possible to tile a square with n dominant rectangles -/
theorem square_tiling_for_n_ge_5 (n : ℕ) (h : n ≥ 5) : SquareTiling n := by
  sorry

end NUMINAMATH_CALUDE_square_tiling_for_n_ge_5_l3630_363046


namespace NUMINAMATH_CALUDE_exactly_one_girl_and_exactly_two_girls_mutually_exclusive_but_not_contrary_l3630_363009

/-- Represents the possible outcomes when selecting 2 students from a group of 2 boys and 2 girls -/
inductive Outcome
  | TwoBoys
  | OneGirlOneBoy
  | TwoGirls

/-- The sample space of all possible outcomes -/
def SampleSpace : Set Outcome := {Outcome.TwoBoys, Outcome.OneGirlOneBoy, Outcome.TwoGirls}

/-- The event "Exactly 1 girl" -/
def ExactlyOneGirl : Set Outcome := {Outcome.OneGirlOneBoy}

/-- The event "Exactly 2 girls" -/
def ExactlyTwoGirls : Set Outcome := {Outcome.TwoGirls}

/-- Theorem stating that "Exactly 1 girl" and "Exactly 2 girls" are mutually exclusive but not contrary -/
theorem exactly_one_girl_and_exactly_two_girls_mutually_exclusive_but_not_contrary :
  (ExactlyOneGirl ∩ ExactlyTwoGirls = ∅) ∧
  (ExactlyOneGirl ∪ ExactlyTwoGirls ≠ SampleSpace) := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_girl_and_exactly_two_girls_mutually_exclusive_but_not_contrary_l3630_363009


namespace NUMINAMATH_CALUDE_family_ages_solution_l3630_363057

/-- Represents the ages of a family at a given time --/
structure FamilyAges where
  man : ℕ
  father : ℕ
  sister : ℕ

/-- Checks if the given ages satisfy the problem conditions --/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.man = (2 * ages.father) / 5 ∧
  ages.man + 10 = (ages.father + 10) / 2 ∧
  ages.sister + 10 = (3 * (ages.father + 10)) / 4

/-- The theorem stating the solution to the problem --/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ 
    ages.man = 20 ∧ ages.father = 50 ∧ ages.sister = 35 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l3630_363057


namespace NUMINAMATH_CALUDE_correct_relative_pronoun_l3630_363067

/-- Represents a relative pronoun -/
inductive RelativePronoun
| When
| That
| Where
| Which

/-- Represents the context of an opportunity -/
structure OpportunityContext where
  universal : Bool
  independentOfAge : Bool
  independentOfProfession : Bool
  independentOfReligion : Bool
  independentOfBackground : Bool

/-- Represents the function of a relative pronoun in a sentence -/
structure PronounFunction where
  modifiesNoun : Bool
  describesCircumstances : Bool
  introducesAdjectiveClause : Bool

/-- Determines if a relative pronoun is correct for the given sentence -/
def isCorrectPronoun (pronoun : RelativePronoun) (context : OpportunityContext) (function : PronounFunction) : Prop :=
  context.universal ∧
  context.independentOfAge ∧
  context.independentOfProfession ∧
  context.independentOfReligion ∧
  context.independentOfBackground ∧
  function.modifiesNoun ∧
  function.describesCircumstances ∧
  function.introducesAdjectiveClause ∧
  pronoun = RelativePronoun.Where

theorem correct_relative_pronoun (context : OpportunityContext) (function : PronounFunction) :
  isCorrectPronoun RelativePronoun.Where context function :=
by sorry

end NUMINAMATH_CALUDE_correct_relative_pronoun_l3630_363067


namespace NUMINAMATH_CALUDE_power_of_four_exponent_l3630_363045

theorem power_of_four_exponent (n : ℕ) (x : ℕ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (h2 : n = 17) : 
  x = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_exponent_l3630_363045


namespace NUMINAMATH_CALUDE_leilas_savings_l3630_363088

theorem leilas_savings (savings : ℚ) : 
  (3 / 4 : ℚ) * savings + 20 = savings → savings = 80 :=
by sorry

end NUMINAMATH_CALUDE_leilas_savings_l3630_363088


namespace NUMINAMATH_CALUDE_yoojeong_drank_most_l3630_363008

def yoojeong_milk : ℚ := 7/10
def eunji_milk : ℚ := 1/2
def yuna_milk : ℚ := 6/10

theorem yoojeong_drank_most : 
  yoojeong_milk > eunji_milk ∧ yoojeong_milk > yuna_milk := by
  sorry

end NUMINAMATH_CALUDE_yoojeong_drank_most_l3630_363008


namespace NUMINAMATH_CALUDE_linear_program_coefficient_l3630_363034

/-- Given a set of linear constraints and a linear objective function,
    prove that the value of the coefficient m in the objective function
    is -2/3 when the minimum value of the function is -3. -/
theorem linear_program_coefficient (x y : ℝ) (m : ℝ) : 
  (x + y - 2 ≥ 0) →
  (x - y + 1 ≥ 0) →
  (x ≤ 3) →
  (∀ x y, x + y - 2 ≥ 0 → x - y + 1 ≥ 0 → x ≤ 3 → m * x + y ≥ -3) →
  (∃ x y, x + y - 2 ≥ 0 ∧ x - y + 1 ≥ 0 ∧ x ≤ 3 ∧ m * x + y = -3) →
  m = -2/3 := by
sorry

end NUMINAMATH_CALUDE_linear_program_coefficient_l3630_363034


namespace NUMINAMATH_CALUDE_chocolate_distribution_l3630_363037

theorem chocolate_distribution (total_chocolates : ℕ) (boys_chocolates : ℕ) (girls_chocolates : ℕ) 
  (num_boys : ℕ) (num_girls : ℕ) :
  total_chocolates = 3000 →
  boys_chocolates = 2 →
  girls_chocolates = 3 →
  num_boys = 60 →
  num_girls = 60 →
  num_boys * boys_chocolates + num_girls * girls_chocolates = total_chocolates →
  num_boys + num_girls = 120 := by
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l3630_363037


namespace NUMINAMATH_CALUDE_inscribe_square_in_circle_l3630_363015

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if four points form a square -/
def is_square (p1 p2 p3 p4 : Point) : Prop :=
  let d12 := (p1.x - p2.x)^2 + (p1.y - p2.y)^2
  let d23 := (p2.x - p3.x)^2 + (p2.y - p3.y)^2
  let d34 := (p3.x - p4.x)^2 + (p3.y - p4.y)^2
  let d41 := (p4.x - p1.x)^2 + (p4.y - p1.y)^2
  let d13 := (p1.x - p3.x)^2 + (p1.y - p3.y)^2
  let d24 := (p2.x - p4.x)^2 + (p2.y - p4.y)^2
  (d12 = d23) ∧ (d23 = d34) ∧ (d34 = d41) ∧ (d13 = d24)

/-- Check if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Construct a line through two points -/
def line_through_points (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y,
    b := p1.x - p2.x,
    c := p2.x * p1.y - p1.x * p2.y }

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  let l := line_through_points p1 p2
  l.a * p3.x + l.b * p3.y + l.c = 0

/-- Theorem: Given a circle with a marked center, it is possible to construct
    four points on the circle that form the vertices of a square using only
    straightedge constructions -/
theorem inscribe_square_in_circle (c : Circle) :
  ∃ (p1 p2 p3 p4 : Point),
    on_circle p1 c ∧ on_circle p2 c ∧ on_circle p3 c ∧ on_circle p4 c ∧
    is_square p1 p2 p3 p4 :=
sorry

end NUMINAMATH_CALUDE_inscribe_square_in_circle_l3630_363015


namespace NUMINAMATH_CALUDE_valid_gift_wrapping_combinations_l3630_363072

/-- The number of wrapping paper varieties -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 3

/-- The number of gift card types -/
def gift_card_types : ℕ := 5

/-- The number of invalid combinations (red ribbon with birthday card) -/
def invalid_combinations : ℕ := 1

/-- Theorem stating the number of valid gift wrapping combinations -/
theorem valid_gift_wrapping_combinations :
  wrapping_paper_varieties * ribbon_colors * gift_card_types - invalid_combinations = 149 := by
sorry

end NUMINAMATH_CALUDE_valid_gift_wrapping_combinations_l3630_363072


namespace NUMINAMATH_CALUDE_prob_b_is_point_four_l3630_363096

/-- Given two events a and b, prove that the probability of b is 0.4 -/
theorem prob_b_is_point_four (a b : Set α) (p : Set α → ℝ) 
  (h1 : p a = 2/5)
  (h2 : p (a ∩ b) = 0.16000000000000003)
  (h3 : p (a ∩ b) = p a * p b) : 
  p b = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_prob_b_is_point_four_l3630_363096


namespace NUMINAMATH_CALUDE_x_plus_y_values_l3630_363078

theorem x_plus_y_values (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 40) :
  x + y = 2 + 2 * Real.sqrt 3 ∨ x + y = 2 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l3630_363078


namespace NUMINAMATH_CALUDE_prime_sum_2019_power_l3630_363065

theorem prime_sum_2019_power (p q : ℕ) : 
  Prime p → Prime q → p + q = 2019 → (p - 1)^(q - 1) = 1 ∨ (p - 1)^(q - 1) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_2019_power_l3630_363065


namespace NUMINAMATH_CALUDE_spinner_direction_l3630_363007

-- Define the possible directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation function
def rotate (initial : Direction) (clockwise : Rat) (counterclockwise : Rat) : Direction :=
  sorry

-- Theorem statement
theorem spinner_direction :
  let initial_direction := Direction.North
  let clockwise_rotation : Rat := 7/4
  let counterclockwise_rotation : Rat := 5/2
  rotate initial_direction clockwise_rotation counterclockwise_rotation = Direction.East :=
by sorry

end NUMINAMATH_CALUDE_spinner_direction_l3630_363007


namespace NUMINAMATH_CALUDE_square_implies_composite_l3630_363018

theorem square_implies_composite (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h_square : ∃ n : ℕ, x^2 + x*y - y = n^2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ x + y + 1 = a * b :=
by sorry

end NUMINAMATH_CALUDE_square_implies_composite_l3630_363018


namespace NUMINAMATH_CALUDE_toothpick_grid_50_40_l3630_363029

/-- Calculates the total number of toothpicks in a rectangular grid -/
def toothpick_grid_count (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem stating that a 50x40 toothpick grid contains 4090 toothpicks -/
theorem toothpick_grid_50_40 :
  toothpick_grid_count 50 40 = 4090 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grid_50_40_l3630_363029


namespace NUMINAMATH_CALUDE_sams_remaining_seashells_l3630_363066

-- Define the initial number of seashells Sam found
def initial_seashells : ℕ := 35

-- Define the number of seashells Sam gave to Joan
def given_away : ℕ := 18

-- Theorem stating how many seashells Sam has now
theorem sams_remaining_seashells : 
  initial_seashells - given_away = 17 := by sorry

end NUMINAMATH_CALUDE_sams_remaining_seashells_l3630_363066


namespace NUMINAMATH_CALUDE_even_increasing_neg_sum_positive_l3630_363085

/-- An even function that is increasing on the negative real line -/
def EvenIncreasingNeg (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ (∀ x y, x < y ∧ y ≤ 0 → f x < f y)

/-- Theorem statement -/
theorem even_increasing_neg_sum_positive
  (f : ℝ → ℝ) (hf : EvenIncreasingNeg f) (x₁ x₂ : ℝ)
  (hx₁ : x₁ > 0) (hx₂ : x₂ < 0) (hf_x : f x₁ < f x₂) :
  x₁ + x₂ > 0 :=
sorry

end NUMINAMATH_CALUDE_even_increasing_neg_sum_positive_l3630_363085


namespace NUMINAMATH_CALUDE_total_books_is_54_l3630_363091

/-- The total number of books Darla, Katie, and Gary have is 54 -/
theorem total_books_is_54 (darla_books : ℕ) (katie_books : ℕ) (gary_books : ℕ)
  (h1 : darla_books = 6)
  (h2 : katie_books = darla_books / 2)
  (h3 : gary_books = 5 * (darla_books + katie_books)) :
  darla_books + katie_books + gary_books = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_54_l3630_363091


namespace NUMINAMATH_CALUDE_tissue_cost_theorem_l3630_363060

/-- Calculates the total cost of tissues given the number of boxes, packs per box,
    tissues per pack, and cost per tissue. -/
def totalCost (boxes : ℕ) (packsPerBox : ℕ) (tissuesPerPack : ℕ) (costPerTissue : ℚ) : ℚ :=
  (boxes * packsPerBox * tissuesPerPack : ℚ) * costPerTissue

/-- Proves that the total cost of 10 boxes of tissues, with 20 packs per box,
    100 tissues per pack, and 5 cents per tissue, is $1000. -/
theorem tissue_cost_theorem :
  totalCost 10 20 100 (5 / 100) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_tissue_cost_theorem_l3630_363060


namespace NUMINAMATH_CALUDE_line_slope_proof_l3630_363028

/-- Given two vectors in a Cartesian coordinate plane and a line with certain properties,
    prove that the slope of the line is 2/5. -/
theorem line_slope_proof (OA OB : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  OA = (1, 4) →
  OB = (-3, 1) →
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ y = k * x) →
  (∀ (v : ℝ × ℝ), v ∈ l → v.2 > 0) →
  (∀ (C : ℝ × ℝ), C ∈ l → OA.1 * C.1 + OA.2 * C.2 = OB.1 * C.1 + OB.2 * C.2) →
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ y = k * x ∧ k = 2/5) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_proof_l3630_363028


namespace NUMINAMATH_CALUDE_acute_angle_inequalities_l3630_363051

theorem acute_angle_inequalities (α β : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_α_lt_β : α < β) : 
  (α - Real.sin α < β - Real.sin β) ∧ 
  (Real.tan α - α < Real.tan β - β) := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_inequalities_l3630_363051


namespace NUMINAMATH_CALUDE_remainder_98_power_50_mod_50_l3630_363069

theorem remainder_98_power_50_mod_50 : 98^50 % 50 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_power_50_mod_50_l3630_363069


namespace NUMINAMATH_CALUDE_daps_equivalent_to_24_dips_l3630_363055

-- Define the units
variable (dap dop dip : ℝ)

-- Define the relationships between units
axiom dap_to_dop : 5 * dap = 4 * dop
axiom dop_to_dip : 3 * dop = 8 * dip

-- Theorem to prove
theorem daps_equivalent_to_24_dips : 
  24 * dip = (45/4) * dap := by sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_24_dips_l3630_363055


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3630_363098

/-- A random variable following a normal distribution -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability function for the normal distribution -/
noncomputable def prob (X : NormalDistribution) (a b : ℝ) : ℝ :=
  sorry

theorem normal_distribution_probability (X : NormalDistribution) 
  (h1 : X.μ = 4)
  (h2 : X.σ = 1)
  (h3 : prob X (X.μ - 2*X.σ) (X.μ + 2*X.σ) = 0.9544)
  (h4 : prob X (X.μ - X.σ) (X.μ + X.σ) = 0.6826) :
  prob X 5 6 = 0.1359 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3630_363098


namespace NUMINAMATH_CALUDE_hiring_range_l3630_363073

/-- The number of standard deviations that includes all accepted ages -/
def num_std_dev (avg : ℕ) (std_dev : ℕ) (num_ages : ℕ) : ℚ :=
  (num_ages - 1) / (2 * std_dev)

theorem hiring_range (avg : ℕ) (std_dev : ℕ) (num_ages : ℕ)
  (h_avg : avg = 20)
  (h_std_dev : std_dev = 8)
  (h_num_ages : num_ages = 17) :
  num_std_dev avg std_dev num_ages = 1 := by
sorry

end NUMINAMATH_CALUDE_hiring_range_l3630_363073


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l3630_363063

theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, x = 2 → (x - 2) * (x + 5) = 0) ∧
  (∃ x : ℝ, (x - 2) * (x + 5) = 0 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l3630_363063


namespace NUMINAMATH_CALUDE_ratio_to_eight_l3630_363070

theorem ratio_to_eight : ∃ x : ℚ, (5 : ℚ) / 1 = x / 8 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_eight_l3630_363070


namespace NUMINAMATH_CALUDE_four_correct_propositions_l3630_363093

theorem four_correct_propositions : 
  (2 ≤ 3) ∧ 
  (∀ m : ℝ, m ≥ 0 → ∃ x : ℝ, x^2 + x - m = 0) ∧ 
  (∀ x y : ℝ, x^2 = y^2 → |x| = |y|) ∧ 
  (∀ a b c : ℝ, a > b ↔ a + c > b + c) := by
  sorry

end NUMINAMATH_CALUDE_four_correct_propositions_l3630_363093


namespace NUMINAMATH_CALUDE_oreo_multiple_l3630_363068

def total_oreos : ℕ := 52
def james_oreos : ℕ := 43

theorem oreo_multiple :
  ∃ (multiple : ℕ) (jordan_oreos : ℕ),
    james_oreos = multiple * jordan_oreos + 7 ∧
    total_oreos = james_oreos + jordan_oreos ∧
    multiple = 4 := by
  sorry

end NUMINAMATH_CALUDE_oreo_multiple_l3630_363068


namespace NUMINAMATH_CALUDE_sqrt_a_plus_b_is_three_l3630_363079

theorem sqrt_a_plus_b_is_three (a b : ℝ) 
  (h1 : 2*a - 1 = 9) 
  (h2 : 3*a + 2*b + 4 = 27) : 
  Real.sqrt (a + b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_b_is_three_l3630_363079


namespace NUMINAMATH_CALUDE_simplify_expression_l3630_363071

theorem simplify_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  3 * x ^ Real.sqrt 2 * (2 * x ^ (-Real.sqrt 2) * y * z) = 6 * y * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3630_363071


namespace NUMINAMATH_CALUDE_stock_value_change_l3630_363061

theorem stock_value_change (initial_value : ℝ) (h : initial_value > 0) :
  let day1_value := initial_value * 0.85
  let day2_value := day1_value * 1.4
  (day2_value - initial_value) / initial_value = 0.19 := by
sorry

end NUMINAMATH_CALUDE_stock_value_change_l3630_363061


namespace NUMINAMATH_CALUDE_unique_sundaes_count_l3630_363040

/-- The number of flavors available -/
def n : ℕ := 8

/-- The number of flavors in each sundae -/
def k : ℕ := 2

/-- The number of unique two scoop sundaes -/
def unique_sundaes : ℕ := Nat.choose n k

theorem unique_sundaes_count : unique_sundaes = 28 := by
  sorry

end NUMINAMATH_CALUDE_unique_sundaes_count_l3630_363040


namespace NUMINAMATH_CALUDE_oranges_for_juice_l3630_363054

/-- Given that 18 oranges make 27 liters of orange juice, 
    prove that 6 oranges are needed to make 9 liters of orange juice. -/
theorem oranges_for_juice (oranges : ℕ) (juice : ℕ) 
  (h : 18 * juice = 27 * oranges) : 
  6 * juice = 9 * oranges :=
by sorry

end NUMINAMATH_CALUDE_oranges_for_juice_l3630_363054


namespace NUMINAMATH_CALUDE_cubic_factorization_l3630_363021

theorem cubic_factorization (a : ℝ) : a^3 - 9*a = a*(a+3)*(a-3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3630_363021


namespace NUMINAMATH_CALUDE_johns_money_to_mother_l3630_363012

theorem johns_money_to_mother (initial_amount : ℝ) (father_fraction : ℝ) (amount_left : ℝ) :
  initial_amount = 200 →
  father_fraction = 3 / 10 →
  amount_left = 65 →
  ∃ (mother_fraction : ℝ), 
    mother_fraction = 3 / 8 ∧
    amount_left = initial_amount * (1 - (mother_fraction + father_fraction)) :=
by sorry

end NUMINAMATH_CALUDE_johns_money_to_mother_l3630_363012


namespace NUMINAMATH_CALUDE_isosceles_base_length_l3630_363081

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  /-- Length of one of the equal sides -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : y + 2*x = 20
  /-- x is positive and less than 10 -/
  xBound : 0 < x ∧ x < 10
  /-- y is positive -/
  yPositive : y > 0

/-- The base length of an isosceles triangle with perimeter 20 is 20 - 2x, where 5 < x < 10 -/
theorem isosceles_base_length (t : IsoscelesTriangle) : 
  t.y = 20 - 2*t.x ∧ 5 < t.x ∧ t.x < 10 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_base_length_l3630_363081


namespace NUMINAMATH_CALUDE_green_chips_count_l3630_363048

/-- Given a jar of chips where:
  * 3 blue chips represent 10% of the total
  * 50% of the chips are white
  * The remaining chips are green
  Prove that there are 12 green chips -/
theorem green_chips_count (total : ℕ) (blue white green : ℕ) : 
  blue = 3 ∧ 
  blue * 10 = total ∧ 
  2 * white = total ∧ 
  blue + white + green = total → 
  green = 12 := by
sorry

end NUMINAMATH_CALUDE_green_chips_count_l3630_363048


namespace NUMINAMATH_CALUDE_cooking_shopping_combinations_l3630_363053

theorem cooking_shopping_combinations (n : ℕ) (k : ℕ) (h : n = 5 ∧ k = 3) : 
  Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_cooking_shopping_combinations_l3630_363053


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3630_363075

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3630_363075


namespace NUMINAMATH_CALUDE_asymptote_parabola_intersection_distance_l3630_363052

/-- The distance between the two points where the asymptotes of the hyperbola x^2 - y^2 = 1
    intersect with the parabola y^2 = 4x is 8, given that one intersection point is the origin. -/
theorem asymptote_parabola_intersection_distance : 
  let hyperbola := fun (x y : ℝ) => x^2 - y^2 = 1
  let parabola := fun (x y : ℝ) => y^2 = 4*x
  let asymptote1 := fun (x : ℝ) => x
  let asymptote2 := fun (x : ℝ) => -x
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (4, 4)
  let B : ℝ × ℝ := (4, -4)
  (hyperbola O.1 O.2) ∧ 
  (parabola O.1 O.2) ∧
  (parabola A.1 A.2) ∧ 
  (parabola B.1 B.2) ∧
  (A.2 = asymptote1 A.1) ∧
  (B.2 = asymptote2 B.1) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by
sorry

end NUMINAMATH_CALUDE_asymptote_parabola_intersection_distance_l3630_363052


namespace NUMINAMATH_CALUDE_unique_square_number_l3630_363076

theorem unique_square_number : ∃! x : ℕ, 
  x > 39 ∧ x < 80 ∧ 
  ∃ y : ℕ, x = y * y ∧ 
  ∃ z : ℕ, x = 4 * z :=
by
  sorry

end NUMINAMATH_CALUDE_unique_square_number_l3630_363076


namespace NUMINAMATH_CALUDE_stating_largest_valid_m_l3630_363080

/-- 
Given a positive integer m, checks if m! can be expressed as the product 
of m - 4 consecutive positive integers.
-/
def is_valid (m : ℕ+) : Prop :=
  ∃ a : ℕ, m.val.factorial = (Finset.range (m - 4)).prod (λ i => i + a + 1)

/-- 
Theorem stating that 1 is the largest positive integer m such that m! 
can be expressed as the product of m - 4 consecutive positive integers.
-/
theorem largest_valid_m : 
  is_valid 1 ∧ ∀ m : ℕ+, m > 1 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_stating_largest_valid_m_l3630_363080


namespace NUMINAMATH_CALUDE_books_from_first_shop_l3630_363064

theorem books_from_first_shop (total_spent : ℕ) (second_shop_books : ℕ) (avg_price : ℕ) :
  total_spent = 768 →
  second_shop_books = 22 →
  avg_price = 12 →
  ∃ first_shop_books : ℕ,
    first_shop_books = 42 ∧
    total_spent = avg_price * (first_shop_books + second_shop_books) :=
by
  sorry

end NUMINAMATH_CALUDE_books_from_first_shop_l3630_363064


namespace NUMINAMATH_CALUDE_student_arrangement_count_l3630_363006

/-- The number of ways to arrange students among attractions -/
def arrange_students (n_students : ℕ) (n_attractions : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange students among attractions when two specific students are at the same attraction -/
def arrange_students_with_pair (n_students : ℕ) (n_attractions : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of arrangements under given conditions -/
theorem student_arrangement_count :
  let n_students : ℕ := 4
  let n_attractions : ℕ := 3
  arrange_students n_students n_attractions - arrange_students_with_pair n_students n_attractions = 30 :=
sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l3630_363006


namespace NUMINAMATH_CALUDE_complex_equality_l3630_363087

theorem complex_equality (x y : ℝ) (i : ℂ) (h : i * i = -1) :
  (x + y * i : ℂ) = 1 / i → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3630_363087


namespace NUMINAMATH_CALUDE_friendly_group_has_complete_subgroup_l3630_363016

/-- Represents the property of two people knowing each other -/
def knows (people : Type) : people → people → Prop := sorry

/-- A group of people satisfying the condition that among any three, two know each other -/
structure FriendlyGroup (people : Type) where
  size : Nat
  members : Finset people
  size_eq : members.card = size
  friendly : ∀ (a b c : people), a ∈ members → b ∈ members → c ∈ members →
    a ≠ b → b ≠ c → a ≠ c → (knows people a b ∨ knows people b c ∨ knows people a c)

/-- A complete subgroup where every pair knows each other -/
def CompleteSubgroup {people : Type} (group : FriendlyGroup people) (subgroup : Finset people) : Prop :=
  subgroup ⊆ group.members ∧ ∀ (a b : people), a ∈ subgroup → b ∈ subgroup → a ≠ b → knows people a b

/-- The main theorem: In a group of 9 people satisfying the friendly condition,
    there exists a complete subgroup of 4 people -/
theorem friendly_group_has_complete_subgroup 
  {people : Type} (group : FriendlyGroup people) (h : group.size = 9) :
  ∃ (subgroup : Finset people), subgroup.card = 4 ∧ CompleteSubgroup group subgroup := by
  sorry

end NUMINAMATH_CALUDE_friendly_group_has_complete_subgroup_l3630_363016


namespace NUMINAMATH_CALUDE_unique_x_l3630_363074

theorem unique_x : ∃! x : ℕ, 
  (∃ k : ℕ, x = 12 * k) ∧ 
  x^2 > 200 ∧ 
  x < 30 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_x_l3630_363074


namespace NUMINAMATH_CALUDE_total_ear_muffs_bought_l3630_363049

theorem total_ear_muffs_bought (before_december : ℕ) (during_december : ℕ)
  (h1 : before_december = 1346)
  (h2 : during_december = 6444) :
  before_december + during_december = 7790 := by
  sorry

end NUMINAMATH_CALUDE_total_ear_muffs_bought_l3630_363049


namespace NUMINAMATH_CALUDE_function_identity_l3630_363044

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 1 > 0)
  (h2 : ∀ m n : ℕ, f (m^2 + n^2) = (f m)^2 + (f n)^2) :
  ∀ n : ℕ, f n = n :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l3630_363044


namespace NUMINAMATH_CALUDE_fraction_simplification_l3630_363041

theorem fraction_simplification :
  (154 : ℚ) / 10780 = 1 / 70 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3630_363041


namespace NUMINAMATH_CALUDE_derivative_of_f_l3630_363086

-- Define the function
def f (x : ℝ) : ℝ := (5 * x - 3) ^ 3

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x => 15 * (5 * x - 3) ^ 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3630_363086


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l3630_363097

/-- Represents the amount of pizza eaten by each sibling -/
structure PizzaConsumption where
  alex : Rat
  beth : Rat
  cyril : Rat
  dan : Rat
  eliza : Rat

/-- Checks if a list of rationals is in decreasing order -/
def isDecreasing (l : List Rat) : Prop :=
  ∀ i j, i < j → j < l.length → l[i]! ≥ l[j]!

/-- The main theorem stating the correct order of pizza consumption -/
theorem pizza_consumption_order (p : PizzaConsumption) 
  (h1 : p.alex = 1/6)
  (h2 : p.beth = 1/4)
  (h3 : p.cyril = 1/3)
  (h4 : p.dan = 0)
  (h5 : p.eliza = 1 - (p.alex + p.beth + p.cyril + p.dan)) :
  isDecreasing [p.cyril, p.beth, p.eliza, p.alex, p.dan] := by
  sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l3630_363097


namespace NUMINAMATH_CALUDE_rectangle_formation_ways_l3630_363084

/-- The number of ways to choose 2 items from a set of 5 -/
def choose_2_from_5 : ℕ := 10

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 5

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 5

/-- The number of lines needed to form a rectangle -/
def lines_for_rectangle : ℕ := 4

/-- Theorem: The number of ways to choose 4 lines (2 horizontal and 2 vertical) 
    from 5 horizontal and 5 vertical lines to form a rectangle is 100 -/
theorem rectangle_formation_ways : 
  (choose_2_from_5 * choose_2_from_5 = 100) ∧ 
  (num_horizontal_lines = 5) ∧ 
  (num_vertical_lines = 5) ∧ 
  (lines_for_rectangle = 4) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_ways_l3630_363084


namespace NUMINAMATH_CALUDE_trigonometric_calculation_quadratic_equation_solution_l3630_363024

-- Problem 1
theorem trigonometric_calculation :
  3 * Real.tan (45 * π / 180) - (1 / 3)⁻¹ + (Real.sin (30 * π / 180) - 2022)^0 + |Real.cos (30 * π / 180) - Real.sqrt 3 / 2| = 1 := by
  sorry

-- Problem 2
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x * (x + 3) - 5 * (x + 3)
  (f 5 = 0 ∧ f (-3) = 0) ∧ ∀ x, f x = 0 → x = 5 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_calculation_quadratic_equation_solution_l3630_363024


namespace NUMINAMATH_CALUDE_area_traced_on_concentric_spheres_l3630_363022

/-- Theorem: Area traced by a sphere on concentric spheres
  Given:
  - Two concentric spheres with radii R₁ and R₂
  - A smaller sphere that traces areas on both spheres
  - The area traced on the inner sphere is A₁
  Prove:
  The area A₂ traced on the outer sphere is equal to A₁ * (R₂/R₁)²
-/
theorem area_traced_on_concentric_spheres
  (R₁ R₂ A₁ : ℝ)
  (h₁ : 0 < R₁)
  (h₂ : 0 < R₂)
  (h₃ : 0 < A₁)
  (h₄ : R₁ < R₂) :
  ∃ A₂ : ℝ, A₂ = A₁ * (R₂/R₁)^2 := by
  sorry

end NUMINAMATH_CALUDE_area_traced_on_concentric_spheres_l3630_363022


namespace NUMINAMATH_CALUDE_savings_duration_l3630_363014

/-- Thomas and Joseph's savings problem -/
theorem savings_duration : 
  ∀ (thomas_monthly joseph_monthly total_savings : ℚ),
  thomas_monthly = 40 →
  joseph_monthly = (3/5) * thomas_monthly →
  total_savings = 4608 →
  ∃ (months : ℕ), 
    (thomas_monthly + joseph_monthly) * months = total_savings ∧ 
    months = 72 := by
  sorry

end NUMINAMATH_CALUDE_savings_duration_l3630_363014


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3630_363023

def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℤ)
  (h_arith : arithmeticSequence a)
  (h_a9 : a 9 = -2012)
  (h_a17 : a 17 = -2012) :
  a 1 + a 25 < 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3630_363023


namespace NUMINAMATH_CALUDE_relay_team_permutations_l3630_363032

def team_size : ℕ := 4
def fixed_positions : ℕ := 1
def remaining_positions : ℕ := team_size - fixed_positions

theorem relay_team_permutations :
  Nat.factorial remaining_positions = 6 :=
by sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l3630_363032


namespace NUMINAMATH_CALUDE_equation_solutions_l3630_363094

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, (1/4) * (2*x + 3)^3 = 16 ↔ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3630_363094


namespace NUMINAMATH_CALUDE_circular_track_length_l3630_363001

/-- The length of the circular track in meters -/
def track_length : ℝ := 480

/-- Alex's speed in meters per unit time -/
def alex_speed : ℝ := 4

/-- Jamie's speed in meters per unit time -/
def jamie_speed : ℝ := 3

/-- Distance Alex runs to first meeting point in meters -/
def alex_first_meeting : ℝ := 150

/-- Distance Jamie runs after first meeting to second meeting point in meters -/
def jamie_second_meeting : ℝ := 180

theorem circular_track_length :
  track_length = 480 ∧
  alex_speed / jamie_speed = 4 / 3 ∧
  alex_first_meeting = 150 ∧
  jamie_second_meeting = 180 ∧
  track_length / 2 + alex_first_meeting = 
    (track_length / 2 - alex_first_meeting) + jamie_second_meeting + track_length / 2 :=
by sorry

end NUMINAMATH_CALUDE_circular_track_length_l3630_363001


namespace NUMINAMATH_CALUDE_inequality_proof_l3630_363062

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) : 
  0 ≤ x*y + y*z + z*x - 2*x*y*z ∧ x*y + y*z + z*x - 2*x*y*z ≤ 7/27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3630_363062


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l3630_363090

theorem geometric_arithmetic_sequence_problem :
  ∃ (a b c : ℝ) (d : ℝ),
    a + b + c = 114 ∧
    b^2 = a * c ∧
    b ≠ a ∧
    b = a + 3 * d ∧
    c = a + 24 * d ∧
    a = 2 ∧
    b = 14 ∧
    c = 98 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l3630_363090


namespace NUMINAMATH_CALUDE_half_filled_cylindrical_tank_volume_l3630_363047

/-- The volume of water in a half-filled cylindrical tank lying on its side -/
theorem half_filled_cylindrical_tank_volume
  (r : ℝ) -- radius of the tank
  (h : ℝ) -- height (length) of the tank
  (hr : r = 5) -- given radius is 5 feet
  (hh : h = 10) -- given height is 10 feet
  : (1 / 2 * π * r^2 * h) = 125 * π := by
  sorry

end NUMINAMATH_CALUDE_half_filled_cylindrical_tank_volume_l3630_363047


namespace NUMINAMATH_CALUDE_water_trough_theorem_l3630_363039

/-- Calculates the final amount of water in a trough after a given number of days -/
def water_trough_calculation (initial_amount : ℝ) (evaporation_rate : ℝ) (refill_rate : ℝ) (days : ℕ) : ℝ :=
  initial_amount - (evaporation_rate - refill_rate) * days

/-- Theorem stating the final amount of water in the trough after 45 days -/
theorem water_trough_theorem :
  water_trough_calculation 350 1 0.4 45 = 323 := by
  sorry

#eval water_trough_calculation 350 1 0.4 45

end NUMINAMATH_CALUDE_water_trough_theorem_l3630_363039


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l3630_363002

/-- Represents a trapezoid with given diagonals and bases -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Calculates the area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  sorry

theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.diagonal1 = 7)
  (h2 : t.diagonal2 = 8)
  (h3 : t.base1 = 3)
  (h4 : t.base2 = 6) :
  trapezoidArea t = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l3630_363002


namespace NUMINAMATH_CALUDE_john_total_spent_l3630_363042

def tshirt_price : ℕ := 20
def num_tshirts : ℕ := 3
def pants_price : ℕ := 50

def total_spent : ℕ := tshirt_price * num_tshirts + pants_price

theorem john_total_spent : total_spent = 110 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spent_l3630_363042
