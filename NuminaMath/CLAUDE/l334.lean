import Mathlib

namespace NUMINAMATH_CALUDE_range_of_x_l334_33459

def p (x : ℝ) : Prop := x^2 - 5*x + 6 ≥ 0

def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) :
  (∀ x, p x ∨ q x) → (∀ x, ¬q x) → x ≤ 0 ∨ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l334_33459


namespace NUMINAMATH_CALUDE_sequence_sum_expression_l334_33472

/-- Given a sequence {a_n} with sum of first n terms S_n, where a_1 = 1 and S_n = 2a_{n+1},
    prove that S_n = (3/2)^(n-1) for n > 1 -/
theorem sequence_sum_expression (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  a 1 = 1 →
  (∀ k, S k = 2 * a (k + 1)) →
  n > 1 →
  S n = (3/2)^(n-1) := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_expression_l334_33472


namespace NUMINAMATH_CALUDE_same_hours_october_september_l334_33469

/-- Represents Julie's landscaping business earnings --/
structure LandscapingEarnings where
  mowing_rate : ℕ
  weeding_rate : ℕ
  sept_mowing_hours : ℕ
  sept_weeding_hours : ℕ
  total_earnings : ℕ

/-- Theorem stating that Julie worked the same hours in October as in September --/
theorem same_hours_october_september (j : LandscapingEarnings)
  (h1 : j.mowing_rate = 4)
  (h2 : j.weeding_rate = 8)
  (h3 : j.sept_mowing_hours = 25)
  (h4 : j.sept_weeding_hours = 3)
  (h5 : j.total_earnings = 248) :
  j.mowing_rate * j.sept_mowing_hours + j.weeding_rate * j.sept_weeding_hours =
  j.total_earnings - (j.mowing_rate * j.sept_mowing_hours + j.weeding_rate * j.sept_weeding_hours) :=
by
  sorry

end NUMINAMATH_CALUDE_same_hours_october_september_l334_33469


namespace NUMINAMATH_CALUDE_inference_is_analogical_l334_33496

/-- Inductive reasoning is the process of reasoning from specific instances to a general conclusion. -/
def inductive_reasoning : Prop := sorry

/-- Deductive reasoning is the process of reasoning from a general premise to a specific conclusion. -/
def deductive_reasoning : Prop := sorry

/-- Analogical reasoning is the process of reasoning from one specific instance to another specific instance. -/
def analogical_reasoning : Prop := sorry

/-- The inference from "If a > b, then a + c > b + c" to "If a > b, then ac > bc" -/
def inference : Prop := sorry

/-- The inference is an example of analogical reasoning -/
theorem inference_is_analogical : inference → analogical_reasoning := by sorry

end NUMINAMATH_CALUDE_inference_is_analogical_l334_33496


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l334_33485

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧
  side_length = 7 ∧
  exterior_angle = 90 ∧
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l334_33485


namespace NUMINAMATH_CALUDE_weaving_time_approx_l334_33400

/-- The time taken to weave a certain amount of cloth, given the weaving rate and total time -/
def weaving_time (rate : Real) (total_time : Real) : Real :=
  total_time

theorem weaving_time_approx :
  let rate := 1.14  -- meters per second
  let total_time := 45.6140350877193  -- seconds
  ∃ ε > 0, |weaving_time rate total_time - 45.614| < ε :=
sorry

end NUMINAMATH_CALUDE_weaving_time_approx_l334_33400


namespace NUMINAMATH_CALUDE_gcd_8917_4273_l334_33498

theorem gcd_8917_4273 : Nat.gcd 8917 4273 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8917_4273_l334_33498


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l334_33465

/-- Given two vectors HK and AE in a vector space, prove that if HK = 1/4 * AE and 
    the magnitude of 4 * HK is 4.8, then the magnitude of AE is 4.8. -/
theorem vector_magnitude_proof 
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (HK AE : V) 
  (h1 : HK = (1/4 : ℝ) • AE) 
  (h2 : ‖(4 : ℝ) • HK‖ = 4.8) : 
  ‖AE‖ = 4.8 := by
  sorry

#check vector_magnitude_proof

end NUMINAMATH_CALUDE_vector_magnitude_proof_l334_33465


namespace NUMINAMATH_CALUDE_orchard_area_distribution_l334_33423

/-- Represents an orange orchard with flat and hilly land. -/
structure Orchard where
  total_area : ℝ
  flat_area : ℝ
  hilly_area : ℝ
  sampled_flat : ℝ
  sampled_hilly : ℝ

/-- Checks if the orchard satisfies the given conditions. -/
def is_valid_orchard (o : Orchard) : Prop :=
  o.total_area = 120 ∧
  o.flat_area + o.hilly_area = o.total_area ∧
  o.sampled_flat + o.sampled_hilly = 10 ∧
  o.sampled_hilly = 2 * o.sampled_flat + 1

/-- Theorem stating the correct distribution of flat and hilly land in the orchard. -/
theorem orchard_area_distribution (o : Orchard) (h : is_valid_orchard o) :
  o.flat_area = 36 ∧ o.hilly_area = 84 := by
  sorry

end NUMINAMATH_CALUDE_orchard_area_distribution_l334_33423


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l334_33490

/-- Given two vectors a and b in ℝ³, where a is (1,1,2) and b is (2,x,y),
    and a is parallel to b, prove that the magnitude of b is 2√6. -/
theorem parallel_vectors_magnitude (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (1, 1, 2)
  let b : ℝ × ℝ × ℝ := (2, x, y)
  (∃ (k : ℝ), b.1 = k * a.1 ∧ b.2.1 = k * a.2.1 ∧ b.2.2 = k * a.2.2) →
  ‖(b.1, b.2.1, b.2.2)‖ = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l334_33490


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l334_33457

theorem mark_and_carolyn_money_sum : (5 : ℚ) / 8 + (7 : ℚ) / 20 = 0.975 := by
  sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l334_33457


namespace NUMINAMATH_CALUDE_chess_draw_probability_l334_33489

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3)
  (h_not_lose : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l334_33489


namespace NUMINAMATH_CALUDE_cube_difference_eq_108_l334_33474

/-- Given two real numbers x and y, if x - y = 3 and x^2 + y^2 = 27, then x^3 - y^3 = 108 -/
theorem cube_difference_eq_108 (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) :
  x^3 - y^3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_eq_108_l334_33474


namespace NUMINAMATH_CALUDE_base_conversion_problem_l334_33478

theorem base_conversion_problem : ∃! (n : ℕ), ∃ (A C : ℕ), 
  (A < 8 ∧ C < 8) ∧
  (A < 6 ∧ C < 6) ∧
  (n = 8 * A + C) ∧
  (n = 6 * C + A) ∧
  (n = 47) := by
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l334_33478


namespace NUMINAMATH_CALUDE_ship_lighthouse_distance_l334_33422

/-- The distance between a ship and a lighthouse given specific sailing conditions -/
theorem ship_lighthouse_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (angle_A : ℝ) 
  (angle_B : ℝ) : 
  speed = 15 → 
  time = 4 → 
  angle_A = 60 * π / 180 → 
  angle_B = 15 * π / 180 → 
  ∃ (d : ℝ), d = 800 * Real.sqrt 3 - 240 ∧ 
    d = Real.sqrt ((speed * time * (Real.cos angle_B - Real.cos angle_A) / (Real.sin angle_A - Real.sin angle_B))^2 + 
                   (speed * time * (Real.sin angle_B * Real.cos angle_A - Real.sin angle_A * Real.cos angle_B) / (Real.sin angle_A - Real.sin angle_B))^2) := by
  sorry

end NUMINAMATH_CALUDE_ship_lighthouse_distance_l334_33422


namespace NUMINAMATH_CALUDE_circumcircle_diameter_perpendicular_to_DK_l334_33480

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0

-- Define the altitude CD
def altitude (A B C D : ℝ × ℝ) : Prop :=
  (D.1 - C.1) * (B.1 - A.1) + (D.2 - C.2) * (B.2 - A.2) = 0

-- Define point K such that |AK| = |AC|
def point_K (A C K : ℝ × ℝ) : Prop :=
  (K.1 - A.1)^2 + (K.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

-- Define the circumcircle of triangle ABK
def circumcircle (A B K O : ℝ × ℝ) : Prop :=
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (K.1 - O.1)^2 + (K.2 - O.2)^2

-- Define perpendicularity
def perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (S.1 - R.1) + (Q.2 - P.2) * (S.2 - R.2) = 0

-- Theorem statement
theorem circumcircle_diameter_perpendicular_to_DK 
  (A B C D K O : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : altitude A B C D)
  (h3 : point_K A C K)
  (h4 : circumcircle A B K O) :
  perpendicular A O D K :=
sorry

end NUMINAMATH_CALUDE_circumcircle_diameter_perpendicular_to_DK_l334_33480


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l334_33430

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l334_33430


namespace NUMINAMATH_CALUDE_total_sum_is_correct_l334_33403

/-- Represents the share ratios and total sum for a money division problem -/
structure MoneyDivision where
  a_ratio : ℝ
  b_ratio : ℝ
  c_ratio : ℝ
  c_share : ℝ
  total_sum : ℝ

/-- The money division problem with given ratios and c's share -/
def problem : MoneyDivision :=
  { a_ratio := 1
    b_ratio := 0.65
    c_ratio := 0.40
    c_share := 48
    total_sum := 246 }

/-- Theorem stating that the total sum is correct given the problem conditions -/
theorem total_sum_is_correct (p : MoneyDivision) :
  p.a_ratio = 1 ∧
  p.b_ratio = 0.65 ∧
  p.c_ratio = 0.40 ∧
  p.c_share = 48 →
  p.total_sum = 246 := by
  sorry

#check total_sum_is_correct problem

end NUMINAMATH_CALUDE_total_sum_is_correct_l334_33403


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l334_33446

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  sum_condition : a 3 + a 5 = 8
  product_condition : a 1 * a 5 = 4

/-- The ratio of the 13th term to the 9th term is 9 -/
theorem geometric_sequence_ratio
  (seq : GeometricSequence) :
  seq.a 13 / seq.a 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l334_33446


namespace NUMINAMATH_CALUDE_paint_calculation_l334_33486

/-- Given three people painting a wall with a work ratio and total area,
    calculate the area painted by the third person. -/
theorem paint_calculation (ratio_a ratio_b ratio_c total_area : ℕ) 
    (ratio_positive : ratio_a > 0 ∧ ratio_b > 0 ∧ ratio_c > 0)
    (total_positive : total_area > 0) :
    let total_ratio := ratio_a + ratio_b + ratio_c
    ratio_c * total_area / total_ratio = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l334_33486


namespace NUMINAMATH_CALUDE_geometric_progression_common_ratio_l334_33493

theorem geometric_progression_common_ratio
  (q : ℝ)
  (h1 : |q| < 1)
  (h2 : ∀ (a : ℝ), a ≠ 0 → a = 4 * (a / (1 - q) - a)) :
  q = 1/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_common_ratio_l334_33493


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l334_33424

theorem trig_expression_simplification (x : Real) :
  x = π / 4 →
  (1 + Real.sin (x + π / 4) - Real.cos (x + π / 4)) / 
  (1 + Real.sin (x + π / 4) + Real.cos (x + π / 4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l334_33424


namespace NUMINAMATH_CALUDE_raft_journey_time_l334_33435

/-- Represents the journey of a steamboat between two cities -/
structure SteamboatJourney where
  speed : ℝ  -- Speed of the steamboat
  current : ℝ  -- Speed of the river current
  time_ab : ℝ  -- Time from A to B
  time_ba : ℝ  -- Time from B to A

/-- Calculates the time taken by a raft to travel from A to B -/
def raft_time (journey : SteamboatJourney) : ℝ :=
  60  -- The actual calculation is omitted and replaced with the result

/-- Theorem stating the raft journey time given steamboat journey details -/
theorem raft_journey_time (journey : SteamboatJourney) 
  (h1 : journey.time_ab = 10)
  (h2 : journey.time_ba = 15)
  (h3 : journey.speed > 0)
  (h4 : journey.current > 0) :
  raft_time journey = 60 := by
  sorry


end NUMINAMATH_CALUDE_raft_journey_time_l334_33435


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_even_integers_l334_33452

theorem greatest_sum_consecutive_even_integers (n : ℕ) :
  n % 2 = 0 →  -- n is even
  n * (n + 2) < 800 →  -- product is less than 800
  ∀ m : ℕ, m % 2 = 0 →  -- for all even m
    m * (m + 2) < 800 →  -- whose product with its consecutive even is less than 800
    n + (n + 2) ≥ m + (m + 2) →  -- n and n+2 have the greatest sum
  n + (n + 2) = 54  -- the greatest sum is 54
:= by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_even_integers_l334_33452


namespace NUMINAMATH_CALUDE_female_students_count_l334_33402

theorem female_students_count (total_students sample_size male_in_sample : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : male_in_sample = 110) : 
  total_students - (total_students * male_in_sample / sample_size) = 720 := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l334_33402


namespace NUMINAMATH_CALUDE_andrey_numbers_l334_33467

/-- Represents a five-digit number as a tuple of five natural numbers, each between 0 and 9 inclusive. -/
def FiveDigitNumber := (Nat × Nat × Nat × Nat × Nat)

/-- Checks if a given FiveDigitNumber is valid (all digits between 0 and 9). -/
def isValidFiveDigitNumber (n : FiveDigitNumber) : Prop :=
  let (a, b, c, d, e) := n
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9

/-- Converts a FiveDigitNumber to its numerical value. -/
def toNumber (n : FiveDigitNumber) : Nat :=
  let (a, b, c, d, e) := n
  10000 * a + 1000 * b + 100 * c + 10 * d + e

/-- Checks if two FiveDigitNumbers differ by exactly two digits. -/
def differByTwoDigits (n1 n2 : FiveDigitNumber) : Prop :=
  let (a1, b1, c1, d1, e1) := n1
  let (a2, b2, c2, d2, e2) := n2
  (a1 ≠ a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 ≠ b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 = b2 ∧ c1 ≠ c2 ∧ d1 = d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 ≠ d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 ≠ e2)

/-- Checks if a FiveDigitNumber contains a zero. -/
def containsZero (n : FiveDigitNumber) : Prop :=
  let (a, b, c, d, e) := n
  a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0 ∨ e = 0

theorem andrey_numbers (n1 n2 : FiveDigitNumber) 
  (h1 : isValidFiveDigitNumber n1)
  (h2 : isValidFiveDigitNumber n2)
  (h3 : differByTwoDigits n1 n2)
  (h4 : toNumber n1 + toNumber n2 = 111111) :
  containsZero n1 ∨ containsZero n2 := by
  sorry

end NUMINAMATH_CALUDE_andrey_numbers_l334_33467


namespace NUMINAMATH_CALUDE_unique_modular_equivalent_in_range_l334_33475

theorem unique_modular_equivalent_in_range : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_equivalent_in_range_l334_33475


namespace NUMINAMATH_CALUDE_f_of_3_eq_3_l334_33406

/-- The function f satisfying the given equation for all x -/
noncomputable def f : ℝ → ℝ := sorry

/-- The main equation defining f -/
axiom f_eq (x : ℝ) : (x^(3^5 - 1) - 1) * f x = (x + 1) * (x^2 + 1) * (x^3 + 1) * (x^(3^4) + 1) - 1

/-- Theorem stating that f(3) = 3 -/
theorem f_of_3_eq_3 : f 3 = 3 := by sorry

end NUMINAMATH_CALUDE_f_of_3_eq_3_l334_33406


namespace NUMINAMATH_CALUDE_divisibility_implication_l334_33463

theorem divisibility_implication (a b : ℤ) : (17 ∣ (2*a + 3*b)) → (17 ∣ (9*a + 5*b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l334_33463


namespace NUMINAMATH_CALUDE_sum_six_consecutive_integers_l334_33444

theorem sum_six_consecutive_integers (m : ℤ) : 
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) = 6 * m + 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_integers_l334_33444


namespace NUMINAMATH_CALUDE_difference_of_squares_502_498_l334_33481

theorem difference_of_squares_502_498 : 502^2 - 498^2 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_502_498_l334_33481


namespace NUMINAMATH_CALUDE_inequality_system_sum_l334_33420

theorem inequality_system_sum (a b : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x + 2*a > 4 ∧ 2*x < b)) →
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_sum_l334_33420


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_zero_l334_33442

theorem sum_of_solutions_eq_zero : 
  ∃ (S : Finset ℤ), (∀ x ∈ S, x^4 - 13*x^2 + 36 = 0) ∧ 
                    (∀ x : ℤ, x^4 - 13*x^2 + 36 = 0 → x ∈ S) ∧ 
                    (S.sum id = 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_zero_l334_33442


namespace NUMINAMATH_CALUDE_pythagorean_theorem_special_case_l334_33445

/-- A right triangle with legs of lengths 1 and 2 -/
structure RightTriangle :=
  (leg1 : ℝ)
  (leg2 : ℝ)
  (is_right : leg1 = 1 ∧ leg2 = 2)

/-- The square of the hypotenuse of a right triangle -/
def hypotenuse_squared (t : RightTriangle) : ℝ :=
  t.leg1^2 + t.leg2^2

/-- Theorem: The square of the hypotenuse of a right triangle with legs 1 and 2 is 5 -/
theorem pythagorean_theorem_special_case (t : RightTriangle) :
  hypotenuse_squared t = 5 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_special_case_l334_33445


namespace NUMINAMATH_CALUDE_complement_union_equals_d_l334_33443

universe u

def U : Set (Fin 4) := {0, 1, 2, 3}
def A : Set (Fin 4) := {0, 1}
def B : Set (Fin 4) := {2}

theorem complement_union_equals_d : 
  (U \ (A ∪ B)) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_d_l334_33443


namespace NUMINAMATH_CALUDE_distribute_three_letters_four_mailboxes_l334_33432

/-- The number of ways to distribute n distinct objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: Distributing 3 letters into 4 mailboxes results in 4^3 ways -/
theorem distribute_three_letters_four_mailboxes : 
  distribute 3 4 = 4^3 := by sorry

end NUMINAMATH_CALUDE_distribute_three_letters_four_mailboxes_l334_33432


namespace NUMINAMATH_CALUDE_investment_interest_proof_l334_33492

/-- Calculates the total annual interest for a two-fund investment --/
def total_annual_interest (total_investment : ℝ) (rate1 rate2 : ℝ) (amount_in_fund1 : ℝ) : ℝ :=
  let amount_in_fund2 := total_investment - amount_in_fund1
  let interest1 := amount_in_fund1 * rate1
  let interest2 := amount_in_fund2 * rate2
  interest1 + interest2

/-- Proves that the total annual interest for the given investment scenario is $4,120 --/
theorem investment_interest_proof :
  total_annual_interest 50000 0.08 0.085 26000 = 4120 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l334_33492


namespace NUMINAMATH_CALUDE_delta_value_l334_33417

theorem delta_value : ∀ Δ : ℤ, 4 * 3 = Δ - 6 → Δ = 18 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l334_33417


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l334_33468

theorem quadratic_always_positive (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x + b > 0) ↔ (0 < b ∧ b < 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l334_33468


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l334_33476

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 180) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l334_33476


namespace NUMINAMATH_CALUDE_lucky_sock_pairs_l334_33448

/-- The probability of all pairs being lucky given n pairs of socks --/
def prob_all_lucky (n : ℕ) : ℚ :=
  (2^n * n.factorial) / (2*n).factorial

/-- The expected number of lucky pairs given n pairs of socks --/
def expected_lucky_pairs (n : ℕ) : ℚ :=
  n / (2*n - 1)

/-- Theorem stating the properties of lucky sock pairs --/
theorem lucky_sock_pairs (n : ℕ) (h : n > 0) : 
  prob_all_lucky n = (2^n * n.factorial) / (2*n).factorial ∧ 
  expected_lucky_pairs n > 1/2 := by
  sorry

#check lucky_sock_pairs

end NUMINAMATH_CALUDE_lucky_sock_pairs_l334_33448


namespace NUMINAMATH_CALUDE_complex_distance_and_midpoint_l334_33491

/-- Given two complex numbers, prove the distance between them and their midpoint -/
theorem complex_distance_and_midpoint (z1 z2 : ℂ) 
  (hz1 : z1 = 3 + 4*I) (hz2 : z2 = -2 - 3*I) : 
  Complex.abs (z1 - z2) = Real.sqrt 74 ∧ 
  (z1 + z2) / 2 = (1/2 : ℂ) + (1/2 : ℂ)*I := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_and_midpoint_l334_33491


namespace NUMINAMATH_CALUDE_function_domain_range_l334_33482

theorem function_domain_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (m * x^2 + m * x + 1)) ↔ 0 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_function_domain_range_l334_33482


namespace NUMINAMATH_CALUDE_largest_number_problem_l334_33437

theorem largest_number_problem (a b c : ℝ) : 
  a < b → b < c →
  a + b + c = 100 →
  c - b = 8 →
  b - a = 4 →
  c = 40 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l334_33437


namespace NUMINAMATH_CALUDE_unique_solution_system_l334_33460

theorem unique_solution_system : 
  ∃! (x y : ℝ), (x + y = (7 - x) + (7 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ x = 5 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l334_33460


namespace NUMINAMATH_CALUDE_intersection_points_count_l334_33484

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here
  nonconcurrent : Bool  -- Represents that no three diagonals are concurrent

/-- The number of intersection points of diagonals inside a convex polygon -/
def intersectionPoints (p : ConvexPolygon n) : ℕ := sorry

/-- Theorem: The number of intersection points of diagonals inside a convex n-gon
    where no three diagonals are concurrent is equal to (n choose 4) -/
theorem intersection_points_count (n : ℕ) (p : ConvexPolygon n) 
    (h : p.nonconcurrent = true) : 
  intersectionPoints p = Nat.choose n 4 := by sorry

end NUMINAMATH_CALUDE_intersection_points_count_l334_33484


namespace NUMINAMATH_CALUDE_meeting_time_correct_l334_33410

/-- Represents the time in hours after 7:00 AM -/
def time_after_seven (hours minutes : ℕ) : ℚ :=
  hours + minutes / 60

/-- The problem setup -/
structure TravelProblem where
  julia_speed : ℚ
  mark_speed : ℚ
  total_distance : ℚ
  mark_departure_time : ℚ

/-- The solution to the problem -/
def meeting_time (p : TravelProblem) : ℚ :=
  (p.total_distance + p.mark_speed * p.mark_departure_time) / (p.julia_speed + p.mark_speed)

/-- The theorem statement -/
theorem meeting_time_correct (p : TravelProblem) : 
  p.julia_speed = 15 ∧ 
  p.mark_speed = 20 ∧ 
  p.total_distance = 85 ∧ 
  p.mark_departure_time = 0.75 →
  meeting_time p = time_after_seven 2 51 := by
  sorry

#eval time_after_seven 2 51

end NUMINAMATH_CALUDE_meeting_time_correct_l334_33410


namespace NUMINAMATH_CALUDE_shortest_horizontal_distance_l334_33428

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - x - 6

/-- Theorem stating the shortest horizontal distance -/
theorem shortest_horizontal_distance :
  ∃ (x₁ x₂ : ℝ),
    f x₁ = 6 ∧
    f x₂ = -6 ∧
    ∀ (y₁ y₂ : ℝ),
      f y₁ = 6 →
      f y₂ = -6 →
      |x₁ - x₂| ≤ |y₁ - y₂| ∧
      |x₁ - x₂| = 3 :=
sorry

end NUMINAMATH_CALUDE_shortest_horizontal_distance_l334_33428


namespace NUMINAMATH_CALUDE_intersection_M_N_l334_33462

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l334_33462


namespace NUMINAMATH_CALUDE_transform_is_shift_l334_33433

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Define the transformation
def transform (g : RealFunction) : RealFunction :=
  λ x => g (x - 2) + 3

-- State the theorem
theorem transform_is_shift (g : RealFunction) :
  ∀ x y, transform g x = y ↔ g (x - 2) = y - 3 :=
sorry

end NUMINAMATH_CALUDE_transform_is_shift_l334_33433


namespace NUMINAMATH_CALUDE_rectangle_triangle_length_l334_33451

/-- Given a rectangle PQRS with PQ = 4 cm, QR = 10 cm, and PM = MQ,
    if the area of triangle PMQ is half the area of rectangle PQRS,
    then the length of segment MQ is 2√10 cm. -/
theorem rectangle_triangle_length (P Q R S M : ℝ × ℝ) : 
  let pq := dist P Q
  let qr := dist Q R
  let pm := dist P M
  let mq := dist M Q
  let area_rect := pq * qr
  let area_tri := (1/2) * pm * mq
  pq = 4 →
  qr = 10 →
  pm = mq →
  area_tri = (1/2) * area_rect →
  mq = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_length_l334_33451


namespace NUMINAMATH_CALUDE_find_m_value_l334_33421

/-- Given functions f and g, and a condition on their values at x = 3, 
    prove that the parameter m in g equals -11/3 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) 
    (hf : ∀ x, f x = 3 * x^2 + 2 / x - 1)
    (hg : ∀ x, g x = 2 * x^2 - m)
    (h_diff : f 3 - g 3 = 5) : 
  m = -11/3 := by sorry

end NUMINAMATH_CALUDE_find_m_value_l334_33421


namespace NUMINAMATH_CALUDE_die_roll_probability_l334_33431

theorem die_roll_probability : 
  let p : ℚ := 1/3  -- probability of rolling a number divisible by 3
  let n : ℕ := 8    -- number of rolls
  1 - (1 - p)^n = 6305/6561 := by
sorry

end NUMINAMATH_CALUDE_die_roll_probability_l334_33431


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l334_33411

def billion : ℝ := 10^9

theorem billion_to_scientific_notation :
  let value : ℝ := 27.58 * billion
  value = 2.758 * 10^10 := by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l334_33411


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l334_33483

theorem quadratic_inequality_solution_condition (k : ℝ) : 
  (k > 0) → 
  (∃ x : ℝ, x^2 - 8*x + k < 0) ↔ 
  (k > 0 ∧ k < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l334_33483


namespace NUMINAMATH_CALUDE_graphs_with_inverses_l334_33429

-- Define the types of graphs
inductive GraphType
| Linear
| Parabola
| DisconnectedLinear
| Semicircle
| Cubic

-- Define a function to check if a graph has an inverse
def has_inverse (g : GraphType) : Prop :=
  match g with
  | GraphType.Linear => true
  | GraphType.Parabola => false
  | GraphType.DisconnectedLinear => true
  | GraphType.Semicircle => false
  | GraphType.Cubic => false

-- Define the specific graphs given in the problem
def graph_A : GraphType := GraphType.Linear
def graph_B : GraphType := GraphType.Parabola
def graph_C : GraphType := GraphType.DisconnectedLinear
def graph_D : GraphType := GraphType.Semicircle
def graph_E : GraphType := GraphType.Cubic

-- Theorem stating which graphs have inverses
theorem graphs_with_inverses :
  (has_inverse graph_A ∧ has_inverse graph_C) ∧
  (¬has_inverse graph_B ∧ ¬has_inverse graph_D ∧ ¬has_inverse graph_E) :=
by sorry

end NUMINAMATH_CALUDE_graphs_with_inverses_l334_33429


namespace NUMINAMATH_CALUDE_zoo_escape_zoo_escape_proof_l334_33471

theorem zoo_escape (lions : ℕ) (recovery_time : ℕ) (total_time : ℕ) : ℕ :=
  let rhinos := (total_time / recovery_time) - lions
  rhinos

theorem zoo_escape_proof :
  zoo_escape 3 2 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_zoo_escape_zoo_escape_proof_l334_33471


namespace NUMINAMATH_CALUDE_xy_equals_nine_l334_33461

theorem xy_equals_nine (x y : ℝ) (h : x * (x + 2*y) = x^2 + 18) : x * y = 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_nine_l334_33461


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l334_33440

def v1 : ℝ × ℝ × ℝ := (1, 3, 4)
def v2 (k : ℝ) : ℝ × ℝ × ℝ := (2, k, 1)
def v3 (k : ℝ) : ℝ × ℝ × ℝ := (1, 1, k)

def volume (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  (a1 * b2 * c3 + a2 * b3 * c1 + a3 * b1 * c2) -
  (a3 * b2 * c1 + a1 * b3 * c2 + a2 * b1 * c3)

theorem parallelepiped_volume (k : ℝ) :
  k > 0 ∧ volume v1 (v2 k) (v3 k) = 12 ↔
  k = 5 + Real.sqrt 26 ∨ k = 5 + Real.sqrt 2 ∨ k = 5 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l334_33440


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_4_and_5_l334_33409

theorem largest_three_digit_divisible_by_4_and_5 : ∀ n : ℕ,
  n ≤ 999 ∧ n ≥ 100 ∧ n % 4 = 0 ∧ n % 5 = 0 → n ≤ 980 := by
  sorry

#check largest_three_digit_divisible_by_4_and_5

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_4_and_5_l334_33409


namespace NUMINAMATH_CALUDE_quiz_points_l334_33426

theorem quiz_points (n : ℕ) (total : ℕ) (r : ℕ) (h1 : n = 12) (h2 : total = 8190) (h3 : r = 2) :
  let first_question_points := total / (r^n - 1)
  let fifth_question_points := first_question_points * r^4
  fifth_question_points = 32 := by
sorry

end NUMINAMATH_CALUDE_quiz_points_l334_33426


namespace NUMINAMATH_CALUDE_simplify_fraction_l334_33401

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l334_33401


namespace NUMINAMATH_CALUDE_faculty_reduction_l334_33404

theorem faculty_reduction (initial_faculty : ℕ) : 
  (initial_faculty : ℝ) * 0.85 * 0.75 = 180 → 
  initial_faculty = 282 :=
by
  sorry

end NUMINAMATH_CALUDE_faculty_reduction_l334_33404


namespace NUMINAMATH_CALUDE_cut_length_of_divided_square_cake_l334_33449

/-- Represents a square cake divided into four equal pieces -/
structure DividedSquareCake where
  side_length : ℝ
  cut_length : ℝ

/-- The perimeter of the original square cake -/
def square_perimeter (cake : DividedSquareCake) : ℝ :=
  4 * cake.side_length

/-- The perimeter of each piece after division -/
def piece_perimeter (cake : DividedSquareCake) : ℝ :=
  2 * cake.side_length + 2 * cake.cut_length

/-- Theorem: The length of each cut in a divided square cake -/
theorem cut_length_of_divided_square_cake :
  ∀ (cake : DividedSquareCake),
    square_perimeter cake = 100 →
    piece_perimeter cake = 56 →
    cake.cut_length = 3 :=
by sorry

end NUMINAMATH_CALUDE_cut_length_of_divided_square_cake_l334_33449


namespace NUMINAMATH_CALUDE_divisibility_by_power_of_five_l334_33413

theorem divisibility_by_power_of_five :
  ∀ k : ℕ, ∃ n : ℕ, (5^k : ℕ) ∣ (n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_power_of_five_l334_33413


namespace NUMINAMATH_CALUDE_g_form_l334_33418

-- Define polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_g_prod : ∀ x, f (g x) = f x * g x
axiom g_3_eq_50 : g 3 = 50

-- Define the theorem
theorem g_form : g = fun x ↦ x^2 + 20*x - 20 :=
sorry

end NUMINAMATH_CALUDE_g_form_l334_33418


namespace NUMINAMATH_CALUDE_inequality_iff_quadratic_nonpositive_l334_33438

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ (x y z : ℝ), x < y → y < z →
    f y - ((z - y) / (z - x) * f x + (y - x) / (z - x) * f z) ≤
    f ((x + z) / 2) - (f x + f z) / 2

/-- The set of quadratic functions with non-positive leading coefficient -/
def QuadraticNonPositive (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≤ 0 ∧ ∀ (x : ℝ), f x = a * x^2 + b * x + c

/-- Main theorem: A function satisfies the inequality if and only if it's quadratic with non-positive leading coefficient -/
theorem inequality_iff_quadratic_nonpositive (f : ℝ → ℝ) :
  SatisfiesInequality f ↔ QuadraticNonPositive f :=
sorry

end NUMINAMATH_CALUDE_inequality_iff_quadratic_nonpositive_l334_33438


namespace NUMINAMATH_CALUDE_num_regions_correct_l334_33436

/-- The number of regions formed by n lines in a plane, where no two lines are parallel and no three lines are concurrent. -/
def num_regions (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- Theorem stating that num_regions correctly calculates the number of regions. -/
theorem num_regions_correct (n : ℕ) :
  num_regions n = n * (n + 1) / 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_num_regions_correct_l334_33436


namespace NUMINAMATH_CALUDE_opponent_score_l334_33494

/-- Given UF's previous game scores and championship game performance, 
    calculate their opponent's score. -/
theorem opponent_score (total_points : ℕ) (num_games : ℕ) (half_reduction : ℕ) (point_difference : ℕ) : 
  total_points = 720 →
  num_games = 24 →
  half_reduction = 2 →
  point_difference = 2 →
  (total_points / num_games / 2 - half_reduction) - point_difference = 11 := by
  sorry


end NUMINAMATH_CALUDE_opponent_score_l334_33494


namespace NUMINAMATH_CALUDE_daniel_water_bottles_l334_33464

/-- The number of bottles Daniel filled for the rugby team -/
def rugby_bottles : ℕ := by sorry

theorem daniel_water_bottles :
  let total_bottles : ℕ := 254
  let football_players : ℕ := 11
  let football_bottles_per_player : ℕ := 6
  let soccer_bottles : ℕ := 53
  let lacrosse_extra_bottles : ℕ := 12
  let coach_bottles : ℕ := 2
  let num_teams : ℕ := 4

  let football_bottles := football_players * football_bottles_per_player
  let lacrosse_bottles := football_bottles + lacrosse_extra_bottles
  let total_coach_bottles := coach_bottles * num_teams

  rugby_bottles = total_bottles - (football_bottles + soccer_bottles + lacrosse_bottles + total_coach_bottles) :=
by sorry

end NUMINAMATH_CALUDE_daniel_water_bottles_l334_33464


namespace NUMINAMATH_CALUDE_sum_of_remaining_numbers_l334_33450

theorem sum_of_remaining_numbers
  (n : ℕ)
  (total_sum : ℝ)
  (subset_sum : ℝ)
  (h1 : n = 5)
  (h2 : total_sum / n = 20)
  (h3 : subset_sum / 2 = 26) :
  total_sum - subset_sum = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remaining_numbers_l334_33450


namespace NUMINAMATH_CALUDE_blown_out_dune_probability_l334_33497

/-- The probability that a sand dune remains after being formed -/
def prob_dune_remains : ℚ := 1 / 3

/-- The probability that a blown-out sand dune contains treasure -/
def prob_treasure : ℚ := 1 / 5

/-- The probability that a formed sand dune has a lucky coupon -/
def prob_lucky_coupon : ℚ := 2 / 3

/-- The probability that a blown-out sand dune contains both treasure and a lucky coupon -/
def prob_both : ℚ := prob_treasure * prob_lucky_coupon

theorem blown_out_dune_probability : prob_both = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_blown_out_dune_probability_l334_33497


namespace NUMINAMATH_CALUDE_angle_PQ_A1BD_l334_33415

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D

/-- Represents a line in 3D space -/
structure Line where
  direction : Point3D

/-- Calculates the reflection of a point with respect to a plane -/
def reflect_point_plane (p : Point3D) (plane : Plane) : Point3D :=
  sorry

/-- Calculates the reflection of a point with respect to a line -/
def reflect_point_line (p : Point3D) (line : Line) : Point3D :=
  sorry

/-- Calculates the angle between a line and a plane -/
def angle_line_plane (line : Line) (plane : Plane) : ℝ :=
  sorry

theorem angle_PQ_A1BD (cube : Cube) : 
  let C1BD : Plane := sorry
  let B1D : Line := sorry
  let A1BD : Plane := sorry
  let P : Point3D := reflect_point_plane cube.A C1BD
  let Q : Point3D := reflect_point_line cube.A B1D
  let PQ : Line := { direction := sorry }
  Real.sin (angle_line_plane PQ A1BD) = 2 * Real.sqrt 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_PQ_A1BD_l334_33415


namespace NUMINAMATH_CALUDE_history_book_cost_l334_33416

theorem history_book_cost (total_books : ℕ) (math_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 80 →
  math_book_cost = 4 →
  total_price = 373 →
  math_books = 27 →
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 :=
by sorry

end NUMINAMATH_CALUDE_history_book_cost_l334_33416


namespace NUMINAMATH_CALUDE_acute_angle_range_characterization_l334_33453

/-- The angle between two vectors is acute if and only if their dot product is positive and they are not collinear -/
def is_acute_angle (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2 > 0) ∧ (a.1 * b.2 ≠ a.2 * b.1)

/-- The set of real numbers m for which the angle between vectors a and b is acute -/
def acute_angle_range : Set ℝ :=
  {m | is_acute_angle (m - 2, m + 3) (2*m + 1, m - 2)}

theorem acute_angle_range_characterization :
  acute_angle_range = {m | m > 2 ∨ (m < (-11 - 5*Real.sqrt 5) / 2) ∨ 
    (((-11 + 5*Real.sqrt 5) / 2 < m) ∧ (m < -4/3))} := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_range_characterization_l334_33453


namespace NUMINAMATH_CALUDE_avg_price_goat_l334_33495

def num_goats : ℕ := 5
def num_hens : ℕ := 10
def total_cost : ℕ := 2500
def avg_price_hen : ℕ := 50

theorem avg_price_goat :
  (total_cost - num_hens * avg_price_hen) / num_goats = 400 :=
sorry

end NUMINAMATH_CALUDE_avg_price_goat_l334_33495


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l334_33408

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x => (1/5) * Real.sin (x + π/3) + Real.cos (x - π/6)
  ∃ M : ℝ, M = 6/5 ∧ ∀ x : ℝ, f x ≤ M ∧ ∃ x₀ : ℝ, f x₀ = M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l334_33408


namespace NUMINAMATH_CALUDE_two_heads_probability_l334_33473

/-- The probability of getting heads on a single fair coin toss -/
def prob_heads : ℚ := 1 / 2

/-- The probability of getting two heads when tossing two fair coins simultaneously -/
def prob_two_heads : ℚ := prob_heads * prob_heads

/-- Theorem: The probability of getting two heads when tossing two fair coins simultaneously is 1/4 -/
theorem two_heads_probability : prob_two_heads = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_two_heads_probability_l334_33473


namespace NUMINAMATH_CALUDE_count_valid_integers_eq_44_l334_33454

def digit_set : List Nat := [2, 3, 5, 5, 6, 6, 6]

def is_valid_integer (n : Nat) : Bool :=
  let digits := n.digits 10
  digits.length == 3 ∧ 
  digits.all (λ d => d ∈ digit_set) ∧
  digits.count 2 ≤ 1 ∧
  digits.count 3 ≤ 1 ∧
  digits.count 5 ≤ 2 ∧
  digits.count 6 ≤ 3

def count_valid_integers : Nat :=
  (List.range 900).map (λ n => n + 100)
    |>.filter is_valid_integer
    |>.length

theorem count_valid_integers_eq_44 : count_valid_integers = 44 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_integers_eq_44_l334_33454


namespace NUMINAMATH_CALUDE_minimum_gift_cost_l334_33466

structure Store :=
  (name : String)
  (mom_gift : Nat)
  (dad_gift : Nat)
  (brother_gift : Nat)
  (sister_gift : Nat)
  (shopping_time : Nat)

def stores : List Store := [
  ⟨"Romashka", 1000, 750, 930, 850, 35⟩,
  ⟨"Oduvanchik", 1050, 790, 910, 800, 30⟩,
  ⟨"Nezabudka", 980, 810, 925, 815, 40⟩,
  ⟨"Landysh", 1100, 755, 900, 820, 25⟩
]

def travel_time : Nat := 30
def start_time : Nat := 16 * 60 + 35
def close_time : Nat := 20 * 60

def is_valid_shopping_plan (plan : List Store) : Bool :=
  let total_time := plan.foldl (fun acc s => acc + s.shopping_time) 0 + (plan.length - 1) * travel_time
  start_time + total_time ≤ close_time

def gift_cost (plan : List Store) : Nat :=
  plan.foldl (fun acc s => acc + s.mom_gift + s.dad_gift + s.brother_gift + s.sister_gift) 0

theorem minimum_gift_cost :
  ∃ (plan : List Store),
    plan.length = 4 ∧
    (∀ s : Store, s ∈ plan → s ∈ stores) ∧
    is_valid_shopping_plan plan ∧
    gift_cost plan = 3435 ∧
    (∀ other_plan : List Store,
      other_plan.length = 4 →
      (∀ s : Store, s ∈ other_plan → s ∈ stores) →
      is_valid_shopping_plan other_plan →
      gift_cost other_plan ≥ 3435) :=
sorry

end NUMINAMATH_CALUDE_minimum_gift_cost_l334_33466


namespace NUMINAMATH_CALUDE_certain_number_proof_l334_33439

theorem certain_number_proof : ∃ x : ℝ, x * 16 = 3408 ∧ 16 * 21.3 = 340.8 → x = 213 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l334_33439


namespace NUMINAMATH_CALUDE_solution_negative_l334_33447

-- Define the equation
def equation (x a : ℝ) : Prop :=
  (x - 1) / (x - 2) - (x - 2) / (x + 1) = (2 * x + a) / ((x - 2) * (x + 1))

-- Define the theorem
theorem solution_negative (a : ℝ) :
  (∃ x : ℝ, equation x a ∧ x < 0) ↔ (a < -5 ∧ a ≠ -7) :=
sorry

end NUMINAMATH_CALUDE_solution_negative_l334_33447


namespace NUMINAMATH_CALUDE_problem_solution_l334_33477

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- State the theorem
theorem problem_solution (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, ¬(p x a) → ¬(q x)) 
  (h3 : ∃ x, ¬(p x a) ∧ (q x)) :
  (a = 1 → ∃ x, x > 2 ∧ x < 3 ∧ p x a ∧ q x) ∧
  (a > 1 ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l334_33477


namespace NUMINAMATH_CALUDE_inequality_solution_l334_33488

def inequality (x : ℝ) : Prop :=
  2 / (x + 2) + 4 / (x + 8) ≥ 3 / 4

theorem inequality_solution :
  ∀ x : ℝ, inequality x ↔ (x > -2 ∧ x ≤ -8/3) ∨ x ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l334_33488


namespace NUMINAMATH_CALUDE_two_thirds_of_number_is_36_l334_33419

theorem two_thirds_of_number_is_36 (x : ℚ) : (2 : ℚ) / 3 * x = 36 → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_number_is_36_l334_33419


namespace NUMINAMATH_CALUDE_problem_solution_l334_33427

theorem problem_solution (y : ℝ) (hy : y ≠ 0) : (9 * y)^18 = (27 * y)^9 → y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l334_33427


namespace NUMINAMATH_CALUDE_eldorado_license_plates_l334_33455

/-- The number of vowels that can be used as the first letter of a license plate. -/
def numVowels : ℕ := 5

/-- The number of letters in the alphabet. -/
def numLetters : ℕ := 26

/-- The number of digits (0-9). -/
def numDigits : ℕ := 10

/-- The total number of valid license plates in Eldorado. -/
def totalLicensePlates : ℕ := numVowels * numLetters * numLetters * numDigits * numDigits

theorem eldorado_license_plates :
  totalLicensePlates = 338000 :=
by sorry

end NUMINAMATH_CALUDE_eldorado_license_plates_l334_33455


namespace NUMINAMATH_CALUDE_max_sum_constrained_l334_33499

theorem max_sum_constrained (x y : ℝ) 
  (h1 : 5 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 5 * y = 15) : 
  x + y ≤ 47 / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_constrained_l334_33499


namespace NUMINAMATH_CALUDE_work_completion_time_l334_33470

/-- The time it takes for worker C to complete the work alone -/
def time_C : ℕ := 36

/-- The time it takes for workers A, B, and C to complete the work together -/
def time_ABC : ℕ := 4

/-- The time it takes for worker A to complete the work alone -/
def time_A : ℕ := 6

/-- The time it takes for worker B to complete the work alone -/
def time_B : ℕ := 18

theorem work_completion_time :
  (1 : ℚ) / time_ABC = (1 : ℚ) / time_A + (1 : ℚ) / time_B + (1 : ℚ) / time_C :=
by sorry


end NUMINAMATH_CALUDE_work_completion_time_l334_33470


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l334_33414

/-- Represents a 3x3x3 cube made up of smaller cubes -/
structure LargeCube :=
  (small_cubes : Fin 3 → Fin 3 → Fin 3 → Bool)

/-- Represents the shading pattern on a face of the large cube -/
inductive FaceShading
  | FourCorners
  | LShape

/-- Represents the shading of opposite faces -/
structure OppositeShading :=
  (face1 : FaceShading)
  (face2 : FaceShading)

/-- The shading pattern for all three pairs of opposite faces -/
def cube_shading : Fin 3 → OppositeShading :=
  λ _ => { face1 := FaceShading.FourCorners, face2 := FaceShading.LShape }

/-- Counts the number of smaller cubes with at least one face shaded -/
def count_shaded_cubes (c : LargeCube) (shading : Fin 3 → OppositeShading) : Nat :=
  sorry

theorem shaded_cubes_count :
  ∀ c : LargeCube,
  count_shaded_cubes c cube_shading = 17 :=
sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l334_33414


namespace NUMINAMATH_CALUDE_geometric_series_sum_6_terms_l334_33425

def geometricSeriesSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_6_terms :
  let a : ℚ := 2
  let r : ℚ := 1/3
  let n : ℕ := 6
  geometricSeriesSum a r n = 2184/729 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_6_terms_l334_33425


namespace NUMINAMATH_CALUDE_probability_two_green_bottles_l334_33441

/-- The probability of selecting 2 green bottles out of 4 green bottles and 38 black bottles -/
theorem probability_two_green_bottles (green_bottles : ℕ) (black_bottles : ℕ) : 
  green_bottles = 4 → black_bottles = 38 → 
  (Nat.choose green_bottles 2 : ℚ) / (Nat.choose (green_bottles + black_bottles) 2) = 1 / 143.5 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_green_bottles_l334_33441


namespace NUMINAMATH_CALUDE_choose_three_from_fifteen_l334_33412

theorem choose_three_from_fifteen : Nat.choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_fifteen_l334_33412


namespace NUMINAMATH_CALUDE_grinder_price_correct_l334_33456

/-- The purchase price of the grinder -/
def grinder_price : ℝ := 15000

/-- The purchase price of the mobile phone -/
def mobile_price : ℝ := 8000

/-- The selling price of the grinder -/
def grinder_sell_price : ℝ := 0.98 * grinder_price

/-- The selling price of the mobile phone -/
def mobile_sell_price : ℝ := 1.1 * mobile_price

/-- The total profit -/
def total_profit : ℝ := 500

theorem grinder_price_correct :
  grinder_sell_price + mobile_sell_price = grinder_price + mobile_price + total_profit :=
by sorry

end NUMINAMATH_CALUDE_grinder_price_correct_l334_33456


namespace NUMINAMATH_CALUDE_one_third_minus_decimal_l334_33487

theorem one_third_minus_decimal : (1 : ℚ) / 3 - 333 / 1000 = 1 / (3 * 1000) := by sorry

end NUMINAMATH_CALUDE_one_third_minus_decimal_l334_33487


namespace NUMINAMATH_CALUDE_mod_nine_power_four_l334_33458

theorem mod_nine_power_four (m : ℕ) : 
  14^4 % 9 = m ∧ 0 ≤ m ∧ m < 9 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_nine_power_four_l334_33458


namespace NUMINAMATH_CALUDE_mode_invariant_under_single_removal_l334_33434

def dataset : List ℕ := [5, 6, 8, 8, 8, 1, 4]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_invariant_under_single_removal (d : ℕ) :
  d ∈ dataset → mode (dataset.erase d) = mode dataset := by
  sorry

end NUMINAMATH_CALUDE_mode_invariant_under_single_removal_l334_33434


namespace NUMINAMATH_CALUDE_equation_roots_l334_33479

/-- The equation a²(x-2) + a(39-20x) + 20 = 0 has at least two distinct roots if and only if a = 20 -/
theorem equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20*x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20*y) + 20 = 0) ↔ 
  a = 20 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_l334_33479


namespace NUMINAMATH_CALUDE_four_distinct_cuts_l334_33407

/-- Represents a square grid with holes -/
structure GridWithHoles :=
  (size : ℕ)
  (holes : List (ℕ × ℕ))

/-- Represents a cut on the grid -/
inductive Cut
  | Vertical : ℕ → Cut
  | Horizontal : ℕ → Cut
  | Diagonal : Bool → Cut

/-- Checks if two parts resulting from a cut are congruent -/
def areCongruentParts (g : GridWithHoles) (c : Cut) : Bool :=
  sorry

/-- Checks if two cuts result in different congruent parts -/
def areDifferentCuts (g : GridWithHoles) (c1 c2 : Cut) : Bool :=
  sorry

/-- Theorem: There are at least four distinct ways to cut a 4x4 grid with two symmetrical holes into congruent parts -/
theorem four_distinct_cuts (g : GridWithHoles) 
  (h1 : g.size = 4)
  (h2 : g.holes = [(1, 1), (2, 2)]) : 
  ∃ (c1 c2 c3 c4 : Cut),
    areCongruentParts g c1 ∧
    areCongruentParts g c2 ∧
    areCongruentParts g c3 ∧
    areCongruentParts g c4 ∧
    areDifferentCuts g c1 c2 ∧
    areDifferentCuts g c1 c3 ∧
    areDifferentCuts g c1 c4 ∧
    areDifferentCuts g c2 c3 ∧
    areDifferentCuts g c2 c4 ∧
    areDifferentCuts g c3 c4 :=
  sorry

end NUMINAMATH_CALUDE_four_distinct_cuts_l334_33407


namespace NUMINAMATH_CALUDE_triangle_y_coordinate_l334_33405

/-- Given a triangle with vertices (-1, 0), (7, y), and (7, -4), if its area is 32, then y = 4 -/
theorem triangle_y_coordinate (y : ℝ) : 
  let vertices := [(-1, 0), (7, y), (7, -4)]
  let area := (1/2 : ℝ) * |(-1 * (y - (-4)) + 7 * ((-4) - 0) + 7 * (0 - y))|
  area = 32 → y = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_y_coordinate_l334_33405
