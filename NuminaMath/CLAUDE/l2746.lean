import Mathlib

namespace NUMINAMATH_CALUDE_mountaineer_arrangement_count_l2746_274672

/-- The number of ways to arrange mountaineers -/
def arrange_mountaineers (total : ℕ) (familiar : ℕ) (group_size : ℕ) : ℕ :=
  -- Number of ways to divide familiar mountaineers
  (familiar.choose (group_size / 2) * (familiar - group_size / 2).choose (group_size / 2) / 2) *
  -- Number of ways to divide unfamiliar mountaineers
  ((total - familiar).choose ((total - familiar) / 2) * ((total - familiar) / 2).choose ((total - familiar) / 2) / 2) *
  -- Number of ways to pair groups
  2 *
  -- Number of ways to order the groups
  2

/-- The theorem stating the number of arrangements for the given problem -/
theorem mountaineer_arrangement_count : 
  arrange_mountaineers 10 4 2 = 120 := by sorry

end NUMINAMATH_CALUDE_mountaineer_arrangement_count_l2746_274672


namespace NUMINAMATH_CALUDE_max_value_3sin2x_l2746_274689

theorem max_value_3sin2x :
  ∀ x : ℝ, 3 * Real.sin (2 * x) ≤ 3 ∧ ∃ x₀ : ℝ, 3 * Real.sin (2 * x₀) = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_3sin2x_l2746_274689


namespace NUMINAMATH_CALUDE_rice_bag_weight_qualification_l2746_274639

def is_qualified (weight : ℝ) : Prop :=
  9.9 ≤ weight ∧ weight ≤ 10.1

theorem rice_bag_weight_qualification :
  is_qualified 10 ∧
  ¬ is_qualified 9.2 ∧
  ¬ is_qualified 10.2 ∧
  ¬ is_qualified 9.8 :=
by sorry

end NUMINAMATH_CALUDE_rice_bag_weight_qualification_l2746_274639


namespace NUMINAMATH_CALUDE_tims_bill_denomination_l2746_274654

theorem tims_bill_denomination :
  let unknown_bills : ℕ := 13
  let five_dollar_bills : ℕ := 11
  let one_dollar_bills : ℕ := 17
  let total_amount : ℕ := 128
  let min_bills_used : ℕ := 16
  
  ∃ (x : ℕ),
    x * unknown_bills + 5 * five_dollar_bills + one_dollar_bills = total_amount ∧
    unknown_bills + five_dollar_bills + one_dollar_bills ≥ min_bills_used ∧
    x = 4 :=
by sorry

end NUMINAMATH_CALUDE_tims_bill_denomination_l2746_274654


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2746_274635

theorem quadratic_root_implies_m_value :
  ∀ m : ℝ, ((-1)^2 - 2*(-1) + m = 0) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2746_274635


namespace NUMINAMATH_CALUDE_total_cost_shirt_and_shoes_l2746_274673

/-- The total cost of a shirt and shoes, given the shirt cost and the relationship between shirt and shoe costs -/
theorem total_cost_shirt_and_shoes (shirt_cost : ℕ) (h1 : shirt_cost = 97) :
  let shoe_cost := 2 * shirt_cost + 9
  shirt_cost + shoe_cost = 300 := by
sorry


end NUMINAMATH_CALUDE_total_cost_shirt_and_shoes_l2746_274673


namespace NUMINAMATH_CALUDE_third_term_is_nine_l2746_274691

/-- A geometric sequence where the first term is 4, the second term is 6, and the third term is x -/
def geometric_sequence (x : ℝ) : ℕ → ℝ
| 0 => 4
| 1 => 6
| 2 => x
| (n + 3) => sorry

/-- Theorem: In the given geometric sequence, the third term x is equal to 9 -/
theorem third_term_is_nine :
  ∃ x : ℝ, (∀ n : ℕ, geometric_sequence x (n + 1) = (geometric_sequence x n) * (geometric_sequence x 1 / geometric_sequence x 0)) → x = 9 :=
sorry

end NUMINAMATH_CALUDE_third_term_is_nine_l2746_274691


namespace NUMINAMATH_CALUDE_problem_solution_l2746_274627

-- Define the ⊗ operation
def otimes (a b : ℕ) : ℕ := sorry

-- Define the main property of ⊗
axiom otimes_prop (a b c : ℕ) : otimes a b = c ↔ a^c = b

theorem problem_solution :
  (∀ x, otimes 3 81 = x → x = 4) ∧
  (∀ a b c, otimes 3 5 = a → otimes 3 6 = b → otimes 3 10 = c → a < b ∧ b < c) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2746_274627


namespace NUMINAMATH_CALUDE_solve_equation_l2746_274615

theorem solve_equation (x : ℝ) : 3 * x = (62 - x) + 26 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2746_274615


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2746_274641

theorem solve_linear_equation : ∃ x : ℝ, 4 * x - 5 = 3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2746_274641


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2746_274699

theorem reciprocal_problem (x : ℚ) : (10 : ℚ) / 3 = 1 / x + 1 → x = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2746_274699


namespace NUMINAMATH_CALUDE_polynomial_roots_product_l2746_274623

theorem polynomial_roots_product (b c : ℤ) : 
  (∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b*r - c = 0) → b*c = 110 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_product_l2746_274623


namespace NUMINAMATH_CALUDE_approximateValuesOfSqrt3_cannot_form_set_l2746_274628

-- Define a type for the concept of "group of objects"
structure GroupOfObjects where
  elements : Set ℝ
  description : String

-- Define the properties required for a set
def hasDeterminacy (g : GroupOfObjects) : Prop :=
  ∀ x, x ∈ g.elements → (∃ y, y = x)

def hasDistinctness (g : GroupOfObjects) : Prop :=
  ∀ x y, x ∈ g.elements → y ∈ g.elements → x = y → x = y

def hasUnorderedness (g : GroupOfObjects) : Prop :=
  ∀ x y, x ∈ g.elements → y ∈ g.elements → x ≠ y → y ∈ g.elements

-- Define what it means for a group of objects to be able to form a set
def canFormSet (g : GroupOfObjects) : Prop :=
  hasDeterminacy g ∧ hasDistinctness g ∧ hasUnorderedness g

-- Define the group of all approximate values of √3
def approximateValuesOfSqrt3 : GroupOfObjects :=
  { elements := {x : ℝ | ∃ ε > 0, |x^2 - 3| < ε},
    description := "All approximate values of √3" }

-- The theorem to prove
theorem approximateValuesOfSqrt3_cannot_form_set :
  ¬(canFormSet approximateValuesOfSqrt3) :=
sorry

end NUMINAMATH_CALUDE_approximateValuesOfSqrt3_cannot_form_set_l2746_274628


namespace NUMINAMATH_CALUDE_miniature_toy_height_difference_l2746_274664

/-- Heights of different poodle types -/
structure PoodleHeights where
  standard : ℕ
  miniature : ℕ
  toy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (h : PoodleHeights) : Prop :=
  h.standard = 28 ∧ h.toy = 14 ∧ h.standard = h.miniature + 8

/-- The theorem to be proved -/
theorem miniature_toy_height_difference (h : PoodleHeights) 
  (hc : problem_conditions h) : h.miniature - h.toy = 6 := by
  sorry

end NUMINAMATH_CALUDE_miniature_toy_height_difference_l2746_274664


namespace NUMINAMATH_CALUDE_legs_of_special_triangle_l2746_274625

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of one leg of the triangle -/
  leg1 : ℝ
  /-- The length of the other leg of the triangle -/
  leg2 : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The distance from the center of the inscribed circle to one end of the hypotenuse -/
  dist1 : ℝ
  /-- The distance from the center of the inscribed circle to the other end of the hypotenuse -/
  dist2 : ℝ
  /-- The leg1 is positive -/
  leg1_pos : 0 < leg1
  /-- The leg2 is positive -/
  leg2_pos : 0 < leg2
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The dist1 is positive -/
  dist1_pos : 0 < dist1
  /-- The dist2 is positive -/
  dist2_pos : 0 < dist2
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : leg1^2 + leg2^2 = (dist1 + dist2)^2
  /-- The radius is related to the legs and distances as per the properties of an inscribed circle -/
  inscribed_circle : radius = (leg1 + leg2 - dist1 - dist2) / 2

/-- 
If the center of the inscribed circle in a right triangle is at distances √5 and √10 
from the ends of the hypotenuse, then the legs of the triangle are 3 and 4.
-/
theorem legs_of_special_triangle (t : RightTriangleWithInscribedCircle) 
  (h1 : t.dist1 = Real.sqrt 5) (h2 : t.dist2 = Real.sqrt 10) : 
  (t.leg1 = 3 ∧ t.leg2 = 4) ∨ (t.leg1 = 4 ∧ t.leg2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_legs_of_special_triangle_l2746_274625


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l2746_274608

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (vote_difference_percentage : ℝ)
  (candidate_b_votes : ℕ)
  (h1 : total_votes = 6720)
  (h2 : vote_difference_percentage = 0.15)
  (h3 : candidate_b_votes = 2184) :
  (total_votes - (2 * candidate_b_votes + vote_difference_percentage * total_votes)) / total_votes = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l2746_274608


namespace NUMINAMATH_CALUDE_three_primes_sum_l2746_274666

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def smallest_square_greater_than_15 : ℕ := 16

theorem three_primes_sum (p q r : ℕ) : 
  is_prime p → is_prime q → is_prime r →
  p + q + r = smallest_square_greater_than_15 →
  1 < p → p < q → q < r →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_three_primes_sum_l2746_274666


namespace NUMINAMATH_CALUDE_equation_equivalence_l2746_274655

theorem equation_equivalence : ∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2746_274655


namespace NUMINAMATH_CALUDE_average_of_pqrs_l2746_274661

theorem average_of_pqrs (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_pqrs_l2746_274661


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l2746_274694

/-- Calculates the gain percentage for a dishonest shopkeeper using a false weight -/
theorem shopkeeper_gain_percentage (false_weight : ℝ) (true_weight : ℝ) : 
  false_weight = 960 →
  true_weight = 1000 →
  (true_weight - false_weight) / false_weight * 100 = (1000 - 960) / 960 * 100 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l2746_274694


namespace NUMINAMATH_CALUDE_negative_twenty_is_spend_l2746_274609

/-- Represents a monetary transaction -/
inductive Transaction
| receive (amount : ℕ)
| spend (amount : ℕ)

/-- Converts a transaction to its signed representation -/
def signedAmount (t : Transaction) : ℤ :=
  match t with
  | Transaction.receive n => n
  | Transaction.spend n => -n

/-- The convention of representing transactions -/
structure TransactionConvention where
  positiveIsReceive : ∀ (n : ℕ), signedAmount (Transaction.receive n) > 0
  negativeIsSpend : ∀ (n : ℕ), signedAmount (Transaction.spend n) < 0

/-- The main theorem -/
theorem negative_twenty_is_spend (conv : TransactionConvention) :
  signedAmount (Transaction.spend 20) = -20 :=
by sorry

end NUMINAMATH_CALUDE_negative_twenty_is_spend_l2746_274609


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_seven_l2746_274619

theorem sqrt_expression_equals_seven :
  (Real.sqrt 3 - 2)^2 + Real.sqrt 12 + 6 * Real.sqrt (1/3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_seven_l2746_274619


namespace NUMINAMATH_CALUDE_third_car_year_l2746_274687

def year_first_car : ℕ := 1970
def years_between_first_and_second : ℕ := 10
def years_between_second_and_third : ℕ := 20

theorem third_car_year :
  year_first_car + years_between_first_and_second + years_between_second_and_third = 2000 := by
  sorry

end NUMINAMATH_CALUDE_third_car_year_l2746_274687


namespace NUMINAMATH_CALUDE_cat_arrangements_eq_six_l2746_274651

/-- The number of distinct arrangements of the letters in the word "CAT" -/
def cat_arrangements : ℕ :=
  Nat.factorial 3

theorem cat_arrangements_eq_six :
  cat_arrangements = 6 := by
  sorry

end NUMINAMATH_CALUDE_cat_arrangements_eq_six_l2746_274651


namespace NUMINAMATH_CALUDE_smallest_prime_10_less_than_perfect_square_l2746_274682

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_10_less_than_perfect_square : 
  (∀ p : ℕ, p < 71 → ¬(isPrime p ∧ ∃ q : ℕ, isPerfectSquare q ∧ p = q - 10)) ∧ 
  (isPrime 71 ∧ ∃ q : ℕ, isPerfectSquare q ∧ 71 = q - 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_10_less_than_perfect_square_l2746_274682


namespace NUMINAMATH_CALUDE_open_box_volume_l2746_274667

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_square_side : ℝ)
  (h_sheet_length : sheet_length = 46)
  (h_sheet_width : sheet_width = 36)
  (h_cut_square_side : cut_square_side = 8) :
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 4800 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l2746_274667


namespace NUMINAMATH_CALUDE_y_min_max_sum_l2746_274612

theorem y_min_max_sum (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ y', (∃ x' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11) → m ≤ y' ∧ y' ≤ M) ∧
  m + M = 8/3 :=
sorry

end NUMINAMATH_CALUDE_y_min_max_sum_l2746_274612


namespace NUMINAMATH_CALUDE_cat_speed_l2746_274610

/-- Proves that a cat's speed is 90 km/h given specific conditions -/
theorem cat_speed (rat_speed : ℝ) (head_start : ℝ) (catch_time : ℝ) :
  rat_speed = 36 →
  head_start = 6 →
  catch_time = 4 →
  rat_speed * (head_start + catch_time) = 90 * catch_time :=
by
  sorry

#check cat_speed

end NUMINAMATH_CALUDE_cat_speed_l2746_274610


namespace NUMINAMATH_CALUDE_jerry_max_throws_l2746_274676

/-- Represents the point system in Mrs. Carlton's class -/
structure PointSystem where
  interrupt_points : ℕ
  insult_points : ℕ
  throw_points : ℕ
  office_threshold : ℕ

/-- Represents Jerry's current misbehavior record -/
structure MisbehaviorRecord where
  interrupts : ℕ
  insults : ℕ

/-- Calculates the maximum number of times Jerry can throw things before reaching the office threshold -/
def max_throws (ps : PointSystem) (record : MisbehaviorRecord) : ℕ :=
  let current_points := record.interrupts * ps.interrupt_points + record.insults * ps.insult_points
  let remaining_points := ps.office_threshold - current_points
  remaining_points / ps.throw_points

/-- Theorem stating that Jerry can throw things twice before being sent to the office -/
theorem jerry_max_throws :
  let ps : PointSystem := {
    interrupt_points := 5,
    insult_points := 10,
    throw_points := 25,
    office_threshold := 100
  }
  let record : MisbehaviorRecord := {
    interrupts := 2,
    insults := 4
  }
  max_throws ps record = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_max_throws_l2746_274676


namespace NUMINAMATH_CALUDE_square_root_81_l2746_274696

theorem square_root_81 : ∀ (x : ℝ), x^2 = 81 ↔ x = 9 ∨ x = -9 := by sorry

end NUMINAMATH_CALUDE_square_root_81_l2746_274696


namespace NUMINAMATH_CALUDE_sales_amount_is_194_l2746_274693

/-- Represents the total sales amount from pencils in a stationery store. -/
def total_sales (eraser_price regular_price short_price : ℚ) 
                (eraser_sold regular_sold short_sold : ℕ) : ℚ :=
  eraser_price * eraser_sold + regular_price * regular_sold + short_price * short_sold

/-- Theorem stating that the total sales amount is $194 given the specific conditions. -/
theorem sales_amount_is_194 :
  total_sales 0.8 0.5 0.4 200 40 35 = 194 := by
  sorry

#eval total_sales 0.8 0.5 0.4 200 40 35

end NUMINAMATH_CALUDE_sales_amount_is_194_l2746_274693


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2746_274686

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence {a_n} where a_2 + a_8 = 12, a_5 = 6 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 2 + a 8 = 12) : 
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2746_274686


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l2746_274663

theorem number_with_specific_remainders : ∃ (N : ℕ), N % 13 = 11 ∧ N % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l2746_274663


namespace NUMINAMATH_CALUDE_solve_proportion_l2746_274604

theorem solve_proportion (x : ℚ) (h : x / 6 = 15 / 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_proportion_l2746_274604


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l2746_274640

/-- Given a geometric sequence {a_n} where a_1+1, a_3+4, a_5+7 form an arithmetic sequence,
    the common difference of this arithmetic sequence is 3. -/
theorem geometric_arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d) :
  ∃ d : ℝ, d = 3 ∧ (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l2746_274640


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2746_274606

theorem imaginary_part_of_complex_number :
  let z : ℂ := 1 - 2 * Complex.I
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2746_274606


namespace NUMINAMATH_CALUDE_division_problem_l2746_274648

theorem division_problem (dividend divisor : ℕ) : 
  (dividend / divisor = 3) → 
  (dividend % divisor = 20) → 
  (dividend + divisor + 3 + 20 = 303) → 
  (divisor = 65 ∧ dividend = 215) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2746_274648


namespace NUMINAMATH_CALUDE_count_congruent_is_77_l2746_274669

/-- The number of positive integers less than 1000 that are congruent to 7 (mod 13) -/
def count_congruent : ℕ :=
  (Finset.filter (fun n => n > 0 ∧ n < 1000 ∧ n % 13 = 7) (Finset.range 1000)).card

/-- Theorem stating that the count of such integers is 77 -/
theorem count_congruent_is_77 : count_congruent = 77 := by
  sorry

end NUMINAMATH_CALUDE_count_congruent_is_77_l2746_274669


namespace NUMINAMATH_CALUDE_unique_paintable_number_l2746_274616

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def paints_every_nth (start : ℕ) (step : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = start + k * step

def is_paintable (h t u : ℕ) : Prop :=
  h = 4 ∧
  t % 2 ≠ 0 ∧
  is_prime u ∧
  ∀ n : ℕ, n > 0 →
    (paints_every_nth 1 4 n ∨ paints_every_nth 3 t n ∨ paints_every_nth 5 u n) ∧
    ¬(paints_every_nth 1 4 n ∧ paints_every_nth 3 t n) ∧
    ¬(paints_every_nth 1 4 n ∧ paints_every_nth 5 u n) ∧
    ¬(paints_every_nth 3 t n ∧ paints_every_nth 5 u n)

theorem unique_paintable_number :
  ∀ h t u : ℕ, is_paintable h t u → 100 * t + 10 * u + h = 354 :=
sorry

end NUMINAMATH_CALUDE_unique_paintable_number_l2746_274616


namespace NUMINAMATH_CALUDE_sword_length_proof_l2746_274680

/-- The length of Christopher's sword in inches -/
def christopher_sword_length : ℕ := 15

/-- The length of Jameson's sword in inches -/
def jameson_sword_length : ℕ := 2 * christopher_sword_length + 3

/-- The length of June's sword in inches -/
def june_sword_length : ℕ := jameson_sword_length + 5

theorem sword_length_proof :
  (jameson_sword_length = 2 * christopher_sword_length + 3) ∧
  (june_sword_length = jameson_sword_length + 5) ∧
  (june_sword_length = christopher_sword_length + 23) →
  christopher_sword_length = 15 := by
sorry

#eval christopher_sword_length

end NUMINAMATH_CALUDE_sword_length_proof_l2746_274680


namespace NUMINAMATH_CALUDE_card_difference_l2746_274652

/-- The number of cards each person has -/
structure CardCounts where
  ann : ℕ
  anton : ℕ
  heike : ℕ

/-- The conditions of the problem -/
def card_problem (c : CardCounts) : Prop :=
  c.ann = 60 ∧
  c.ann = 6 * c.heike ∧
  c.anton = c.heike

/-- The theorem to prove -/
theorem card_difference (c : CardCounts) (h : card_problem c) : 
  c.ann - c.anton = 50 := by
  sorry

end NUMINAMATH_CALUDE_card_difference_l2746_274652


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2746_274614

theorem line_intersects_circle (a : ℝ) (h : a ≠ 0) :
  let d := |2*a| / Real.sqrt (a^2 + 1)
  d < 3 := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2746_274614


namespace NUMINAMATH_CALUDE_f_at_3_l2746_274613

def f (x : ℝ) : ℝ := 9*x^3 - 5*x^2 - 3*x + 7

theorem f_at_3 : f 3 = 196 := by
  sorry

end NUMINAMATH_CALUDE_f_at_3_l2746_274613


namespace NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l2746_274683

theorem square_difference_divided (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by
  sorry

theorem problem_solution : (245^2 - 205^2) / 40 = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l2746_274683


namespace NUMINAMATH_CALUDE_range_of_m_l2746_274677

def A (m : ℝ) : Set ℝ := {x | x^2 + Real.sqrt m * x + 1 = 0}

theorem range_of_m (m : ℝ) : (A m ∩ Set.univ = ∅) → (0 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2746_274677


namespace NUMINAMATH_CALUDE_price_reduction_correct_l2746_274629

/-- The final price of a medication after two price reductions -/
def final_price (m : ℝ) (x : ℝ) : ℝ := m * (1 - x)^2

/-- Theorem stating that the final price after two reductions is correct -/
theorem price_reduction_correct (m : ℝ) (x : ℝ) (y : ℝ) 
  (hm : m > 0) (hx : 0 ≤ x ∧ x < 1) :
  y = final_price m x ↔ y = m * (1 - x)^2 := by sorry

end NUMINAMATH_CALUDE_price_reduction_correct_l2746_274629


namespace NUMINAMATH_CALUDE_line_bisecting_circle_min_value_l2746_274659

/-- Given a line that always bisects the circumference of a circle, 
    prove the minimum value of 1/a + 1/b -/
theorem line_bisecting_circle_min_value (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, 2*a*x - b*y + 2 = 0 → 
    x^2 + y^2 + 2*x - 4*y + 1 = 0 → 
    -- The line bisects the circle (implicit condition)
    True) → 
  (1/a + 1/b) ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_line_bisecting_circle_min_value_l2746_274659


namespace NUMINAMATH_CALUDE_largest_integral_solution_l2746_274638

theorem largest_integral_solution : ∃ x : ℤ, (1 : ℚ) / 4 < x / 6 ∧ x / 6 < 7 / 9 ∧ ∀ y : ℤ, (1 : ℚ) / 4 < y / 6 ∧ y / 6 < 7 / 9 → y ≤ x := by
  sorry

end NUMINAMATH_CALUDE_largest_integral_solution_l2746_274638


namespace NUMINAMATH_CALUDE_allowance_spending_l2746_274671

theorem allowance_spending (weekly_allowance : ℚ) 
  (arcade_fraction : ℚ) (candy_amount : ℚ) : 
  weekly_allowance = 3.75 →
  arcade_fraction = 3/5 →
  candy_amount = 1 →
  let remaining_after_arcade := weekly_allowance - arcade_fraction * weekly_allowance
  let toy_store_amount := remaining_after_arcade - candy_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_allowance_spending_l2746_274671


namespace NUMINAMATH_CALUDE_pascal_leibniz_relation_l2746_274690

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Element of the Leibniz triangle -/
def leibniz (n k : ℕ) : ℚ := 1 / ((n + 1 : ℚ) * (binomial n k))

/-- Theorem stating the relationship between Pascal's and Leibniz's triangles -/
theorem pascal_leibniz_relation (n k : ℕ) (h : k ≤ n) :
  leibniz n k = 1 / ((n + 1 : ℚ) * (binomial n k)) := by
  sorry

end NUMINAMATH_CALUDE_pascal_leibniz_relation_l2746_274690


namespace NUMINAMATH_CALUDE_sand_pile_volume_l2746_274626

/-- The volume of a cylindrical pile of sand -/
theorem sand_pile_volume :
  ∀ (r h d : ℝ),
  d = 8 →                -- diameter is 8 feet
  r = d / 2 →            -- radius is half the diameter
  h = 2 * r →            -- height is twice the radius
  π * r^2 * h = 128 * π  -- volume is 128π cubic feet
  := by sorry

end NUMINAMATH_CALUDE_sand_pile_volume_l2746_274626


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l2746_274688

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k > 105 → ¬(∀ (m : ℕ), Even m → m > 0 → 
    k ∣ (m+1)*(m+3)*(m+5)*(m+7)*(m+9)*(m+11)) ∧ 
  (∀ (m : ℕ), Even m → m > 0 → 
    105 ∣ (m+1)*(m+3)*(m+5)*(m+7)*(m+9)*(m+11)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l2746_274688


namespace NUMINAMATH_CALUDE_correct_system_l2746_274624

/-- Represents the length of rope needed to go around the tree once. -/
def y : ℝ := sorry

/-- Represents the total length of the rope. -/
def x : ℝ := sorry

/-- The condition that when the rope goes around the tree 3 times, there will be an extra 5 feet of rope left. -/
axiom three_wraps : 3 * y + 5 = x

/-- The condition that when the rope goes around the tree 4 times, there will be 2 feet less of rope left. -/
axiom four_wraps : 4 * y - 2 = x

/-- Theorem stating that the system of equations correctly represents the problem. -/
theorem correct_system : (3 * y + 5 = x) ∧ (4 * y - 2 = x) := by sorry

end NUMINAMATH_CALUDE_correct_system_l2746_274624


namespace NUMINAMATH_CALUDE_circuit_board_count_l2746_274660

/-- The number of circuit boards that fail verification -/
def failed_boards : ℕ := 64

/-- The fraction of boards that pass verification but are faulty -/
def faulty_fraction : ℚ := 1 / 8

/-- The total number of faulty boards -/
def total_faulty : ℕ := 456

/-- The total number of circuit boards in the group -/
def total_boards : ℕ := 3200

theorem circuit_board_count :
  (failed_boards : ℚ) + faulty_fraction * (total_boards - failed_boards : ℚ) = total_faulty ∧
  total_boards = failed_boards + (total_faulty - failed_boards) / faulty_fraction := by
  sorry

end NUMINAMATH_CALUDE_circuit_board_count_l2746_274660


namespace NUMINAMATH_CALUDE_perpendicular_planes_condition_l2746_274658

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset_line_plane : Line → Plane → Prop)

variable (α β : Plane)
variable (m : Line)

-- Theorem statement
theorem perpendicular_planes_condition 
  (h_distinct : α ≠ β)
  (h_subset : subset_line_plane m α) :
  (∀ m, subset_line_plane m α → 
    (perpendicular_line_plane m β → perpendicular_plane_plane α β) ∧
    ¬(perpendicular_plane_plane α β → perpendicular_line_plane m β)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_condition_l2746_274658


namespace NUMINAMATH_CALUDE_decimal_point_problem_l2746_274645

theorem decimal_point_problem :
  ∃ (x y : ℝ), y - x = 7.02 ∧ y = 10 * x ∧ x = 0.78 ∧ y = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l2746_274645


namespace NUMINAMATH_CALUDE_m_values_l2746_274692

-- Define the sets A and B
def A : Set ℝ := {x | x^2 ≠ 1}
def B (m : ℝ) : Set ℝ := {x | m * x = 1}

-- State the theorem
theorem m_values (h : ∀ m : ℝ, A ∪ B m = A) :
  {m : ℝ | ∃ x, x ∈ B m} = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_m_values_l2746_274692


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_minus_2_squared_l2746_274601

theorem imaginary_part_of_i_minus_2_squared (i : ℂ) : 
  (i * i = -1) → Complex.im ((i - 2) ^ 2) = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_minus_2_squared_l2746_274601


namespace NUMINAMATH_CALUDE_boys_height_correction_l2746_274668

theorem boys_height_correction (n : ℕ) (initial_avg wrong_height actual_avg : ℝ) : 
  n = 35 →
  initial_avg = 183 →
  wrong_height = 166 →
  actual_avg = 181 →
  ∃ (correct_height : ℝ), 
    correct_height = wrong_height + (n * initial_avg - n * actual_avg) ∧
    correct_height = 236 :=
by sorry

end NUMINAMATH_CALUDE_boys_height_correction_l2746_274668


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l2746_274621

theorem greatest_prime_factor_of_154 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 154 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 154 → q ≤ p ∧ p = 11 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l2746_274621


namespace NUMINAMATH_CALUDE_equation_one_solution_l2746_274695

theorem equation_one_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ - 2)^2 - 5 = 0 ∧ (x₂ - 2)^2 - 5 = 0 ∧ x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_equation_one_solution_l2746_274695


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2746_274670

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  -- Length of side EF
  ef : ℝ
  -- Length of side GH
  gh : ℝ
  -- Height of the trapezoid
  height : ℝ
  -- Area of the trapezoid
  area : ℝ
  -- EF is half the length of GH
  ef_half_gh : ef = gh / 2
  -- Height is 6 units
  height_is_6 : height = 6
  -- Area is 90 square units
  area_is_90 : area = 90

/-- Calculate the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that the perimeter of the trapezoid is 30 + 2√61 -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 30 + 2 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2746_274670


namespace NUMINAMATH_CALUDE_sum_pairwise_reciprocal_sums_geq_three_halves_l2746_274630

theorem sum_pairwise_reciprocal_sums_geq_three_halves 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_pairwise_reciprocal_sums_geq_three_halves_l2746_274630


namespace NUMINAMATH_CALUDE_one_line_passes_through_trisection_point_l2746_274618

-- Define the points
def A : ℝ × ℝ := (-3, 6)
def B : ℝ × ℝ := (6, -3)
def P : ℝ × ℝ := (2, 3)

-- Define the trisection points
def T₁ : ℝ × ℝ := (0, 3)
def T₂ : ℝ × ℝ := (3, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + y - 9 = 0

-- Theorem statement
theorem one_line_passes_through_trisection_point :
  ∃ T, (T = T₁ ∨ T = T₂) ∧ 
       line_equation P.1 P.2 ∧
       line_equation T.1 T.2 :=
sorry

end NUMINAMATH_CALUDE_one_line_passes_through_trisection_point_l2746_274618


namespace NUMINAMATH_CALUDE_missing_number_proof_l2746_274644

theorem missing_number_proof (x : ℝ) : (4 + x) + (8 - 3 - 1) = 11 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2746_274644


namespace NUMINAMATH_CALUDE_kgonal_number_formula_l2746_274600

/-- The nth k-gonal number -/
def N (n k : ℕ) : ℚ :=
  match k with
  | 3 => (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n
  | 4 => (n^2 : ℚ)
  | 5 => (3/2 : ℚ) * n^2 - (1/2 : ℚ) * n
  | 6 => (2 : ℚ) * n^2 - (n : ℚ)
  | _ => ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n

theorem kgonal_number_formula (n k : ℕ) (h : k ≥ 3) :
  N n k = ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n := by
  sorry

end NUMINAMATH_CALUDE_kgonal_number_formula_l2746_274600


namespace NUMINAMATH_CALUDE_prob_other_side_green_l2746_274620

/-- Represents a card with two sides --/
inductive Card
| BlueBoth
| BlueGreen
| GreenBoth

/-- The box of cards --/
def box : Finset Card := sorry

/-- The number of cards in the box --/
def num_cards : ℕ := 8

/-- The number of cards that are blue on both sides --/
def num_blue_both : ℕ := 4

/-- The number of cards that are blue on one side and green on the other --/
def num_blue_green : ℕ := 2

/-- The number of cards that are green on both sides --/
def num_green_both : ℕ := 2

/-- Function to check if a given side of a card is green --/
def is_green (c : Card) (side : Bool) : Bool := sorry

/-- The probability of picking a card and observing a green side --/
def prob_green_side : ℚ := sorry

/-- The probability of both sides being green given that one observed side is green --/
def prob_both_green_given_one_green : ℚ := sorry

theorem prob_other_side_green : 
  prob_both_green_given_one_green = 2/3 := sorry

end NUMINAMATH_CALUDE_prob_other_side_green_l2746_274620


namespace NUMINAMATH_CALUDE_juan_number_puzzle_l2746_274649

theorem juan_number_puzzle (n : ℝ) : 
  (2 * ((n + 3)^2) - 3) / 2 = 49 → n = Real.sqrt (101 / 2) - 3 :=
by sorry

end NUMINAMATH_CALUDE_juan_number_puzzle_l2746_274649


namespace NUMINAMATH_CALUDE_cost_of_lettuce_cost_of_lettuce_is_one_dollar_l2746_274632

/-- The cost of the head of lettuce in Lauren's grocery purchase --/
theorem cost_of_lettuce : ℝ := by
  -- Define the known costs
  let meat_cost : ℝ := 2 * 3.5
  let buns_cost : ℝ := 1.5
  let tomato_cost : ℝ := 1.5 * 2
  let pickles_cost : ℝ := 2.5 - 1

  -- Define the total bill and change
  let total_paid : ℝ := 20
  let change : ℝ := 6

  -- Define the actual spent amount
  let actual_spent : ℝ := total_paid - change

  -- Define the sum of known costs
  let known_costs : ℝ := meat_cost + buns_cost + tomato_cost + pickles_cost

  -- The cost of lettuce is the difference between actual spent and known costs
  have lettuce_cost : ℝ := actual_spent - known_costs

  -- Prove that the cost of lettuce is 1.00
  sorry

/-- The cost of the head of lettuce is $1.00 --/
theorem cost_of_lettuce_is_one_dollar : cost_of_lettuce = 1 := by sorry

end NUMINAMATH_CALUDE_cost_of_lettuce_cost_of_lettuce_is_one_dollar_l2746_274632


namespace NUMINAMATH_CALUDE_thomas_drawings_l2746_274675

theorem thomas_drawings (colored_pencil : ℕ) (blending_markers : ℕ) (charcoal : ℕ)
  (h1 : colored_pencil = 14)
  (h2 : blending_markers = 7)
  (h3 : charcoal = 4) :
  colored_pencil + blending_markers + charcoal = 25 :=
by sorry

end NUMINAMATH_CALUDE_thomas_drawings_l2746_274675


namespace NUMINAMATH_CALUDE_factorization_problem_1_l2746_274697

theorem factorization_problem_1 (x y : ℝ) :
  9 - x^2 + 12*x*y - 36*y^2 = (3 + x - 6*y) * (3 - x + 6*y) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l2746_274697


namespace NUMINAMATH_CALUDE_sector_arc_length_l2746_274657

/-- Given a sector with central angle π/3 and radius 3, its arc length is π. -/
theorem sector_arc_length (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 3) :
  θ * r = π := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2746_274657


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l2746_274662

/-- Given a parabola y = ax^2 + 4 that is tangent to the line y = 3x + 1, prove that a = 3/4 -/
theorem parabola_tangent_to_line (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 = 3 * x + 1 ∧ 
   ∀ y : ℝ, y ≠ x → a * y^2 + 4 ≠ 3 * y + 1) →
  a = 3/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l2746_274662


namespace NUMINAMATH_CALUDE_smallest_n_for_square_root_96n_l2746_274622

theorem smallest_n_for_square_root_96n (n : ℕ) : 
  (∃ k : ℕ, k * k = 96 * n) → n ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_square_root_96n_l2746_274622


namespace NUMINAMATH_CALUDE_tire_company_cost_per_batch_l2746_274602

/-- A tire company's production and sales model -/
structure TireCompany where
  cost_per_batch : ℝ
  cost_per_tire : ℝ
  selling_price : ℝ
  batch_size : ℕ
  profit_per_tire : ℝ

/-- The cost per batch for the tire company -/
def cost_per_batch (company : TireCompany) : ℝ :=
  company.cost_per_batch

/-- Theorem stating the cost per batch for the given scenario -/
theorem tire_company_cost_per_batch :
  ∀ (company : TireCompany),
    company.cost_per_tire = 8 →
    company.selling_price = 20 →
    company.batch_size = 15000 →
    company.profit_per_tire = 10.5 →
    cost_per_batch company = 22500 := by
  sorry

end NUMINAMATH_CALUDE_tire_company_cost_per_batch_l2746_274602


namespace NUMINAMATH_CALUDE_sphere_with_n_plus_one_points_l2746_274636

open Set

variable {α : Type*} [MetricSpace α]

theorem sphere_with_n_plus_one_points
  (m n : ℕ)
  (points : Finset α)
  (h_card : points.card = m * n + 1)
  (h_distance : ∀ (subset : Finset α), subset ⊆ points → subset.card = m + 1 →
    ∃ (x y : α), x ∈ subset ∧ y ∈ subset ∧ x ≠ y ∧ dist x y ≤ 1) :
  ∃ (center : α), ∃ (subset : Finset α),
    subset ⊆ points ∧
    subset.card = n + 1 ∧
    ∀ x ∈ subset, dist center x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_sphere_with_n_plus_one_points_l2746_274636


namespace NUMINAMATH_CALUDE_count_square_functions_l2746_274678

-- Define the type for our function
def SquareFunction := Set ℤ → Set ℤ

-- Define what it means for a function to be in our family
def is_in_family (f : SquareFunction) : Prop :=
  ∃ (domain : Set ℤ),
    (∀ x ∈ domain, f domain = {y | ∃ x ∈ domain, y = x^2}) ∧
    (f domain = {1, 4})

-- State the theorem
theorem count_square_functions : 
  ∃! (n : ℕ), ∃ (functions : Finset SquareFunction),
    functions.card = n ∧
    (∀ f ∈ functions, is_in_family f) ∧
    (∀ f, is_in_family f → f ∈ functions) ∧
    n = 8 := by sorry

end NUMINAMATH_CALUDE_count_square_functions_l2746_274678


namespace NUMINAMATH_CALUDE_third_number_is_58_l2746_274605

def number_list : List ℕ := [54, 55, 58, 59, 62, 62, 63, 65, 65]

theorem third_number_is_58 : 
  number_list[2] = 58 := by sorry

end NUMINAMATH_CALUDE_third_number_is_58_l2746_274605


namespace NUMINAMATH_CALUDE_lagrange_interpolation_identities_l2746_274642

theorem lagrange_interpolation_identities 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hca : c ≠ a) : 
  (1 / ((a - b) * (a - c)) + 1 / ((b - c) * (b - a)) + 1 / ((c - a) * (c - b)) = 0) ∧
  (a / ((a - b) * (a - c)) + b / ((b - c) * (b - a)) + c / ((c - a) * (c - b)) = 0) ∧
  (a^2 / ((a - b) * (a - c)) + b^2 / ((b - c) * (b - a)) + c^2 / ((c - a) * (c - b)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_lagrange_interpolation_identities_l2746_274642


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2746_274633

/-- Given a hyperbola and a circle with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 5 = 0 →
    ∃ t : ℝ, (b*x + a*y = 0 ∨ b*x - a*y = 0) →
      (x - 3)^2 + y^2 = 4) →
  3^2 = a^2 - b^2 →
  a^2 = 5 ∧ b^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2746_274633


namespace NUMINAMATH_CALUDE_root_quadratic_implies_value_l2746_274607

theorem root_quadratic_implies_value (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → 2*m^2 - 4*m = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_implies_value_l2746_274607


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2746_274643

theorem imaginary_part_of_z : 
  let z : ℂ := (1 - Complex.I) / Complex.I
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2746_274643


namespace NUMINAMATH_CALUDE_gadget_sales_sum_l2746_274685

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Gadget sales problem -/
theorem gadget_sales_sum :
  arithmetic_sum 2 3 25 = 950 := by
  sorry

end NUMINAMATH_CALUDE_gadget_sales_sum_l2746_274685


namespace NUMINAMATH_CALUDE_f_at_three_fifths_l2746_274631

def f (x : ℝ) : ℝ := 15 * x^5 + 6 * x^4 + x^3 - x^2 - 2*x - 1

theorem f_at_three_fifths :
  f (3/5) = -2/5 := by sorry

end NUMINAMATH_CALUDE_f_at_three_fifths_l2746_274631


namespace NUMINAMATH_CALUDE_tan_half_period_l2746_274684

/-- The period of tan(x/2) is 2π -/
theorem tan_half_period : 
  ∀ f : ℝ → ℝ, (∀ x, f x = Real.tan (x / 2)) → 
  ∃ p : ℝ, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ p = 2 * Real.pi := by
  sorry

/-- The period of tan(x) is π -/
axiom tan_period : 
  ∀ x : ℝ, Real.tan (x + Real.pi) = Real.tan x

end NUMINAMATH_CALUDE_tan_half_period_l2746_274684


namespace NUMINAMATH_CALUDE_round_to_nearest_whole_number_l2746_274646

theorem round_to_nearest_whole_number : 
  let x : ℝ := 6703.4999
  ‖x - 6703‖ < ‖x - 6704‖ :=
by sorry

end NUMINAMATH_CALUDE_round_to_nearest_whole_number_l2746_274646


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_range_l2746_274637

theorem quadratic_real_solutions_range (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_range_l2746_274637


namespace NUMINAMATH_CALUDE_min_distance_between_points_l2746_274665

noncomputable section

def f (x : ℝ) : ℝ := Real.sin x + (1/6) * x^3
def g (x : ℝ) : ℝ := x - 1

theorem min_distance_between_points (x₁ x₂ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ ≥ 0) (h₃ : f x₁ = g x₂) :
  ∃ (d : ℝ), d = |x₂ - x₁| ∧ d ≥ 1 ∧ 
  (∀ (y₁ y₂ : ℝ), y₁ ≥ 0 → y₂ ≥ 0 → f y₁ = g y₂ → |y₂ - y₁| ≥ d) :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_points_l2746_274665


namespace NUMINAMATH_CALUDE_vector_angle_problem_l2746_274650

theorem vector_angle_problem (α β : Real) (a b : Fin 2 → Real) :
  a 0 = Real.cos α ∧ a 1 = Real.sin α ∧
  b 0 = Real.cos β ∧ b 1 = Real.sin β ∧
  Real.sqrt ((a 0 - b 0)^2 + (a 1 - b 1)^2) = 2 * Real.sqrt 5 / 5 ∧
  0 < α ∧ α < π / 2 ∧
  -π / 2 < β ∧ β < 0 ∧
  Real.sin β = -5 / 13 →
  Real.cos (α - β) = 3 / 5 ∧ Real.sin α = 33 / 65 := by
sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l2746_274650


namespace NUMINAMATH_CALUDE_probability_on_2x_is_one_twelfth_l2746_274611

/-- A die is a finite set of numbers from 1 to 6 -/
def Die : Finset ℕ := Finset.range 6

/-- The probability space of rolling a die twice -/
def DieRollSpace : Finset (ℕ × ℕ) := Die.product Die

/-- The event where (x, y) falls on y = 2x -/
def EventOn2x : Finset (ℕ × ℕ) := DieRollSpace.filter (fun (x, y) => y = 2 * x)

/-- The probability of the event -/
def ProbabilityOn2x : ℚ := (EventOn2x.card : ℚ) / (DieRollSpace.card : ℚ)

theorem probability_on_2x_is_one_twelfth : ProbabilityOn2x = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_on_2x_is_one_twelfth_l2746_274611


namespace NUMINAMATH_CALUDE_max_similar_triangles_five_points_l2746_274674

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle formed by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- The set of all triangles formed by choosing 3 points from a set of 5 points -/
def allTriangles (points : Finset Point) : Finset Triangle := sorry

/-- The set of all similar triangles from a set of triangles -/
def similarTriangles (triangles : Finset Triangle) : Finset (Finset Triangle) := sorry

/-- The theorem stating that the maximum number of similar triangles from 5 points is 4 -/
theorem max_similar_triangles_five_points (points : Finset Point) :
  points.card = 5 →
  (similarTriangles (allTriangles points)).sup (λ s => s.card) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_similar_triangles_five_points_l2746_274674


namespace NUMINAMATH_CALUDE_sequence_properties_l2746_274617

/-- Sequence a_n with sum S_n satisfying S_n = 2a_n - 3 for n ∈ ℕ* -/
def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ := 2 * a n - 3

/-- Sequence b_n defined as b_n = (n-1)a_n -/
def b (a : ℕ+ → ℝ) (n : ℕ+) : ℝ := (n.val - 1) * a n

/-- Sum T_n of the first n terms of sequence b_n -/
def T (b : ℕ+ → ℝ) : ℕ+ → ℝ := fun n ↦ (Finset.range n.val).sum (fun i ↦ b ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_properties (a : ℕ+ → ℝ) (k : ℝ) :
  (∀ n : ℕ+, S a n = 2 * a n - 3) →
  (∀ n : ℕ+, a n = 3 * 2^(n.val - 1)) ∧
  (∀ n : ℕ+, T (b a) n = 3 * (n.val - 2) * 2^n.val + 6) ∧
  (∀ n : ℕ+, T (b a) n > k * a n + 16 * n.val - 26 → k < 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2746_274617


namespace NUMINAMATH_CALUDE_dixon_passed_count_l2746_274681

/-- The number of students who passed in Dr. Collins' lecture -/
def collins_passed : ℕ := 18

/-- The total number of students in Dr. Collins' lecture -/
def collins_total : ℕ := 30

/-- The total number of students in Professor Dixon's lecture -/
def dixon_total : ℕ := 45

/-- The number of students who passed in Professor Dixon's lecture -/
def dixon_passed : ℕ := 27

theorem dixon_passed_count : 
  collins_passed * dixon_total = dixon_passed * collins_total := by
  sorry

end NUMINAMATH_CALUDE_dixon_passed_count_l2746_274681


namespace NUMINAMATH_CALUDE_unique_plane_through_parallel_lines_l2746_274679

-- Define a type for points in 3D space
variable (Point : Type)

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define a type for planes in 3D space
variable (Plane : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Define a relation for a line being contained in a plane
variable (contains : Plane → Line → Prop)

-- Theorem: Through two parallel lines, there is exactly one plane
theorem unique_plane_through_parallel_lines 
  (l1 l2 : Line) 
  (h : parallel l1 l2) : 
  ∃! p : Plane, contains p l1 ∧ contains p l2 :=
sorry

end NUMINAMATH_CALUDE_unique_plane_through_parallel_lines_l2746_274679


namespace NUMINAMATH_CALUDE_circle_radius_condition_l2746_274647

theorem circle_radius_condition (x y c : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 2*y + c = 0 → (x + 4)^2 + (y + 1)^2 = 25) → c = -8 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_condition_l2746_274647


namespace NUMINAMATH_CALUDE_shoes_in_box_l2746_274653

/-- The number of pairs of shoes in the box -/
def num_pairs : ℕ := 5

/-- The probability of selecting two matching shoes at random -/
def prob_matching : ℚ := 1 / 9

/-- The total number of shoes in the box -/
def total_shoes : ℕ := 2 * num_pairs

/-- Theorem stating that given the conditions, the total number of shoes is 10 -/
theorem shoes_in_box :
  (num_pairs = 5) →
  (prob_matching = 1 / 9) →
  (total_shoes = 10) := by
  sorry


end NUMINAMATH_CALUDE_shoes_in_box_l2746_274653


namespace NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_length_l2746_274634

/-- The length of the repeating block in the decimal expansion of 7/13 -/
def repeating_block_length : ℕ := 6

/-- 7 is prime -/
axiom seven_prime : Nat.Prime 7

/-- 13 is prime -/
axiom thirteen_prime : Nat.Prime 13

/-- The theorem stating that the length of the repeating block in the decimal expansion of 7/13 is 6 -/
theorem seven_thirteenths_repeating_block_length :
  repeating_block_length = 6 := by sorry

end NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_length_l2746_274634


namespace NUMINAMATH_CALUDE_trail_mix_packs_needed_l2746_274603

def total_people : ℕ := 18
def pouches_per_pack : ℕ := 6

theorem trail_mix_packs_needed :
  ∃ (packs : ℕ), packs * pouches_per_pack ≥ total_people ∧
  ∀ (x : ℕ), x * pouches_per_pack ≥ total_people → x ≥ packs :=
by sorry

end NUMINAMATH_CALUDE_trail_mix_packs_needed_l2746_274603


namespace NUMINAMATH_CALUDE_b_power_a_equals_sixteen_l2746_274656

theorem b_power_a_equals_sixteen (a b : ℝ) : 
  b = Real.sqrt (2 - a) + Real.sqrt (a - 2) - 4 → b^a = 16 := by
sorry

end NUMINAMATH_CALUDE_b_power_a_equals_sixteen_l2746_274656


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2746_274698

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) 
  (h5 : a + d = b + c) : 
  Real.sqrt d + Real.sqrt a < Real.sqrt b + Real.sqrt c := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2746_274698
