import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_simplification_l2231_223144

theorem polynomial_simplification (s : ℝ) :
  (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 4) = s^2 - 4 * s + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2231_223144


namespace NUMINAMATH_CALUDE_unused_ribbon_length_l2231_223175

/-- Given a ribbon of length 30 meters cut into 6 equal parts, 
    if 4 parts are used, then 10 meters of ribbon are not used. -/
theorem unused_ribbon_length 
  (total_length : ℝ) 
  (num_parts : ℕ) 
  (used_parts : ℕ) 
  (h1 : total_length = 30) 
  (h2 : num_parts = 6) 
  (h3 : used_parts = 4) : 
  total_length - (total_length / num_parts) * used_parts = 10 := by
  sorry


end NUMINAMATH_CALUDE_unused_ribbon_length_l2231_223175


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l2231_223160

theorem parallel_vectors_angle (α : Real) :
  let a : Fin 2 → Real := ![3/2, Real.sin α]
  let b : Fin 2 → Real := ![1, 1/3]
  (∃ (k : Real), a = k • b) →
  α = π/6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l2231_223160


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l2231_223153

/-- 
Given a quadratic equation (k-1)x^2 + 4x + 1 = 0, this theorem states that
for the equation to have two distinct real roots, k must be less than 5 and not equal to 1.
-/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k - 1) * x₁^2 + 4 * x₁ + 1 = 0 ∧ 
    (k - 1) * x₂^2 + 4 * x₂ + 1 = 0) ↔ 
  (k < 5 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l2231_223153


namespace NUMINAMATH_CALUDE_ratio_problem_l2231_223102

theorem ratio_problem : ∃ x : ℚ, (150 : ℚ) / 1 = x / 2 ∧ x = 300 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l2231_223102


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l2231_223161

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = 8 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 16 → 
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l2231_223161


namespace NUMINAMATH_CALUDE_count_eight_digit_numbers_with_product_4900_l2231_223182

/-- The number of eight-digit numbers whose digits' product equals 4900 -/
def eight_digit_numbers_with_product_4900 : ℕ := 4200

/-- Theorem stating that the number of eight-digit numbers whose digits' product equals 4900 is 4200 -/
theorem count_eight_digit_numbers_with_product_4900 :
  eight_digit_numbers_with_product_4900 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_count_eight_digit_numbers_with_product_4900_l2231_223182


namespace NUMINAMATH_CALUDE_dianas_biking_distance_l2231_223154

/-- Diana's biking problem -/
theorem dianas_biking_distance
  (initial_speed : ℝ)
  (initial_time : ℝ)
  (tired_speed : ℝ)
  (total_time : ℝ)
  (h1 : initial_speed = 3)
  (h2 : initial_time = 2)
  (h3 : tired_speed = 1)
  (h4 : total_time = 6) :
  initial_speed * initial_time + tired_speed * (total_time - initial_time) = 10 :=
by sorry

end NUMINAMATH_CALUDE_dianas_biking_distance_l2231_223154


namespace NUMINAMATH_CALUDE_op_theorem_l2231_223170

/-- The type representing elements in our set -/
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

/-- The operation ⊕ -/
def op : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem op_theorem : 
  op (op Element.four Element.one) (op Element.two Element.three) = Element.three :=
by sorry

end NUMINAMATH_CALUDE_op_theorem_l2231_223170


namespace NUMINAMATH_CALUDE_equation_solution_l2231_223128

theorem equation_solution (x : ℝ) :
  (x^2 - 7*x + 6)/(x - 1) + (2*x^2 + 7*x - 6)/(2*x - 1) = 1 ∧ 
  x ≠ 1 ∧ 
  x ≠ 1/2 →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2231_223128


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2231_223135

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 9 * x - 5) - (2 * x^2 + 4 * x - 15) = x^2 + 5 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2231_223135


namespace NUMINAMATH_CALUDE_circles_intersection_common_chord_l2231_223185

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

theorem circles_intersection_common_chord :
  (∃ x y : ℝ, C₁ x y ∧ C₂ x y) →
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord x y) :=
by sorry

end NUMINAMATH_CALUDE_circles_intersection_common_chord_l2231_223185


namespace NUMINAMATH_CALUDE_room_width_calculation_l2231_223198

theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 5.5 →
  cost_per_sqm = 400 →
  total_cost = 8250 →
  width * length * cost_per_sqm = total_cost →
  width = 3.75 := by
sorry

end NUMINAMATH_CALUDE_room_width_calculation_l2231_223198


namespace NUMINAMATH_CALUDE_train_length_l2231_223101

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 6 → ∃ length : ℝ, abs (length - 100.02) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2231_223101


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2231_223139

-- Define set P
def P : Set ℝ := {x | x * (x - 3) < 0}

-- Define set Q
def Q : Set ℝ := {x | |x| < 2}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2231_223139


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_sum_l2231_223130

theorem smallest_n_for_integer_sum : 
  ∃ (n : ℕ), n > 0 ∧ 
  (1/3 + 1/4 + 1/8 + 1/n : ℚ).isInt ∧ 
  (∀ m : ℕ, m > 0 ∧ (1/3 + 1/4 + 1/8 + 1/m : ℚ).isInt → n ≤ m) ∧ 
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_sum_l2231_223130


namespace NUMINAMATH_CALUDE_adults_count_is_21_l2231_223100

/-- Represents the trekking group and meal information -/
structure TrekkingGroup where
  childrenCount : ℕ
  adultMealCapacity : ℕ
  childrenMealCapacity : ℕ
  remainingChildrenCapacity : ℕ
  adultsMealCount : ℕ

/-- Theorem stating that the number of adults in the trekking group is 21 -/
theorem adults_count_is_21 (group : TrekkingGroup)
  (h1 : group.childrenCount = 70)
  (h2 : group.adultMealCapacity = 70)
  (h3 : group.childrenMealCapacity = 90)
  (h4 : group.remainingChildrenCapacity = 63)
  (h5 : group.adultsMealCount = 21) :
  group.adultsMealCount = 21 := by
  sorry

#check adults_count_is_21

end NUMINAMATH_CALUDE_adults_count_is_21_l2231_223100


namespace NUMINAMATH_CALUDE_triangle_properties_l2231_223119

open Real

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (t.a * cos t.C + t.c * cos t.A = 2 * t.b * cos t.A) →
  (t.A = π / 3) ∧
  (t.a = Real.sqrt 7 ∧ t.b = 2 →
    (1/2 * t.b * t.c * sin t.A = (3 * Real.sqrt 3) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2231_223119


namespace NUMINAMATH_CALUDE_student_tickets_sold_l2231_223146

/-- Proves that the number of student tickets sold is 9 given the specified conditions -/
theorem student_tickets_sold (adult_price : ℝ) (student_price : ℝ) (total_tickets : ℕ) (total_revenue : ℝ)
  (h1 : adult_price = 4)
  (h2 : student_price = 2.5)
  (h3 : total_tickets = 59)
  (h4 : total_revenue = 222.5) :
  ∃ (student_tickets : ℕ), 
    student_tickets = 9 ∧
    (total_tickets - student_tickets : ℝ) * adult_price + (student_tickets : ℝ) * student_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l2231_223146


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l2231_223180

theorem quadratic_factorization_sum (h b c d : ℤ) : 
  (∀ x : ℝ, 6 * x^2 + x - 12 = (h * x + b) * (c * x + d)) → 
  |h| + |b| + |c| + |d| = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l2231_223180


namespace NUMINAMATH_CALUDE_strawberry_cost_l2231_223118

theorem strawberry_cost (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (3 * B + 7 * J) = 301 → 
  7 * J * N / 100 = 196 / 100 :=
by sorry

end NUMINAMATH_CALUDE_strawberry_cost_l2231_223118


namespace NUMINAMATH_CALUDE_expression_evaluation_l2231_223132

theorem expression_evaluation : 
  (120^2 - 13^2) / (80^2 - 17^2) * ((80 - 17)*(80 + 17)) / ((120 - 13)*(120 + 13)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2231_223132


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2231_223103

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 64 ∧ 
  (∀ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^3 ≤ n * (w^6 + x^6 + y^6 + z^6)) ∧
  (∀ (m : ℕ), m < n → ∃ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^3 > m * (w^6 + x^6 + y^6 + z^6)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2231_223103


namespace NUMINAMATH_CALUDE_triangle_ratios_l2231_223176

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B = 8 ∧ d t.A t.C = 6 ∧ d t.B t.C = 4

-- Define angle bisector
def isAngleBisector (t : Triangle) (D : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.B D / d t.C D = d t.A t.B / d t.A t.C

-- Define the intersection point P
def intersectionPoint (t : Triangle) (D E : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- The actual calculation of the intersection point

-- Define the circumcenter
def circumcenter (t : Triangle) : ℝ × ℝ :=
  sorry  -- The actual calculation of the circumcenter

-- Main theorem
theorem triangle_ratios (t : Triangle) (D E : ℝ × ℝ) :
  isValidTriangle t →
  isAngleBisector t D →
  isAngleBisector t E →
  let P := intersectionPoint t D E
  let O := circumcenter t
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.B P / d P E = 2 ∧
  d O D / d D t.A = 1/3 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_ratios_l2231_223176


namespace NUMINAMATH_CALUDE_digestion_period_correct_l2231_223179

/-- The period (in days) for a python to completely digest an alligator -/
def digestion_period : ℕ := 7

/-- The number of days observed -/
def observation_days : ℕ := 616

/-- The maximum number of alligators eaten in the observation period -/
def max_alligators_eaten : ℕ := 88

/-- Theorem stating that the digestion period is correct given the observed data -/
theorem digestion_period_correct : 
  digestion_period * max_alligators_eaten = observation_days :=
by sorry

end NUMINAMATH_CALUDE_digestion_period_correct_l2231_223179


namespace NUMINAMATH_CALUDE_base5_to_base7_conversion_l2231_223196

/-- Converts a number from base 5 to base 10 -/
def base5_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def decimal_to_base7 (n : ℕ) : ℕ := sorry

/-- Theorem: The base-7 representation of 412₅ is 212₇ -/
theorem base5_to_base7_conversion :
  decimal_to_base7 (base5_to_decimal 412) = 212 := by sorry

end NUMINAMATH_CALUDE_base5_to_base7_conversion_l2231_223196


namespace NUMINAMATH_CALUDE_count_goats_l2231_223183

/-- Given a field with animals, prove the number of goats -/
theorem count_goats (total : ℕ) (cows : ℕ) (sheep_and_goats : ℕ) 
  (h1 : total = 200)
  (h2 : cows = 40)
  (h3 : sheep_and_goats = 56)
  : total - cows - sheep_and_goats = 104 := by
  sorry

end NUMINAMATH_CALUDE_count_goats_l2231_223183


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l2231_223104

theorem bobby_candy_problem (initial_candy : ℕ) (eaten_later : ℕ) (remaining_candy : ℕ)
  (h1 : initial_candy = 36)
  (h2 : eaten_later = 15)
  (h3 : remaining_candy = 4) :
  initial_candy - remaining_candy - eaten_later = 17 :=
by sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l2231_223104


namespace NUMINAMATH_CALUDE_students_with_puppies_and_parrots_l2231_223148

theorem students_with_puppies_and_parrots 
  (total_students : ℕ) 
  (puppy_percentage : ℚ) 
  (parrot_percentage : ℚ) 
  (h1 : total_students = 40)
  (h2 : puppy_percentage = 80 / 100)
  (h3 : parrot_percentage = 25 / 100) :
  ⌊(total_students : ℚ) * puppy_percentage * parrot_percentage⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_with_puppies_and_parrots_l2231_223148


namespace NUMINAMATH_CALUDE_roses_count_l2231_223189

theorem roses_count (vase_capacity : ℕ) (carnations : ℕ) (vases : ℕ) : 
  vase_capacity = 9 → carnations = 4 → vases = 3 → 
  vases * vase_capacity - carnations = 23 := by
  sorry

end NUMINAMATH_CALUDE_roses_count_l2231_223189


namespace NUMINAMATH_CALUDE_percentage_difference_in_earnings_l2231_223178

/-- Given Mike's and Phil's hourly earnings, calculate the percentage difference -/
theorem percentage_difference_in_earnings (mike_earnings phil_earnings : ℝ) 
  (h1 : mike_earnings = 14)
  (h2 : phil_earnings = 7) :
  (mike_earnings - phil_earnings) / mike_earnings * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_in_earnings_l2231_223178


namespace NUMINAMATH_CALUDE_mentor_fraction_is_one_seventh_l2231_223164

/-- Represents the mentorship program in a school --/
structure MentorshipProgram where
  seventh_graders : ℕ
  tenth_graders : ℕ
  mentored_seventh : ℕ
  mentoring_tenth : ℕ

/-- Conditions of the mentorship program --/
def valid_program (p : MentorshipProgram) : Prop :=
  p.mentoring_tenth = p.mentored_seventh ∧
  4 * p.mentoring_tenth = p.tenth_graders ∧
  3 * p.mentored_seventh = p.seventh_graders

/-- The fraction of students with a mentor --/
def mentor_fraction (p : MentorshipProgram) : ℚ :=
  p.mentored_seventh / (p.seventh_graders + p.tenth_graders)

/-- Theorem stating that the fraction of students with a mentor is 1/7 --/
theorem mentor_fraction_is_one_seventh (p : MentorshipProgram) 
  (h : valid_program p) : mentor_fraction p = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_mentor_fraction_is_one_seventh_l2231_223164


namespace NUMINAMATH_CALUDE_brick_ratio_proof_l2231_223149

theorem brick_ratio_proof (total_bricks : ℕ) (discount_price full_price total_spent : ℚ)
  (h1 : total_bricks = 1000)
  (h2 : discount_price = 0.25)
  (h3 : full_price = 0.50)
  (h4 : total_spent = 375)
  : ∃ (discounted_bricks : ℕ),
    discounted_bricks * discount_price + (total_bricks - discounted_bricks) * full_price = total_spent ∧
    discounted_bricks * 2 = total_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_ratio_proof_l2231_223149


namespace NUMINAMATH_CALUDE_joe_egg_count_l2231_223158

/-- The number of eggs Joe found around the club house -/
def club_house_eggs : ℕ := 12

/-- The number of eggs Joe found around the park -/
def park_eggs : ℕ := 5

/-- The number of eggs Joe found in the town hall garden -/
def town_hall_eggs : ℕ := 3

/-- The total number of eggs Joe found -/
def total_eggs : ℕ := club_house_eggs + park_eggs + town_hall_eggs

theorem joe_egg_count : total_eggs = 20 := by
  sorry

end NUMINAMATH_CALUDE_joe_egg_count_l2231_223158


namespace NUMINAMATH_CALUDE_product_of_differences_l2231_223129

theorem product_of_differences (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) :
  (a - 1) * (b - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l2231_223129


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l2231_223125

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_sum_theorem (p q r : ℕ) (hp : IsPrime p) (hq : IsPrime q) (hr : IsPrime r)
  (sum_eq : p + q = r) (p_lt_q : p < q) (one_lt_p : 1 < p) : p = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l2231_223125


namespace NUMINAMATH_CALUDE_subset_condition_l2231_223134

-- Define the sets A and B
def A : Set ℝ := {x | (x + 1) / (x - 2) < 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 > 0}

-- Define the complement of A in ℝ
def A_complement : Set ℝ := {x | ¬ (x ∈ A)}

-- State the theorem
theorem subset_condition (a : ℝ) :
  0 < a ∧ a ≤ 1/2 → B a ⊆ A_complement :=
by sorry

end NUMINAMATH_CALUDE_subset_condition_l2231_223134


namespace NUMINAMATH_CALUDE_no_real_roots_iff_b_positive_l2231_223193

/-- The polynomial has no real roots if and only if b is positive -/
theorem no_real_roots_iff_b_positive (b : ℝ) : 
  (∀ x : ℝ, x^4 + b*x^3 - 2*x^2 + b*x + 2 ≠ 0) ↔ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_b_positive_l2231_223193


namespace NUMINAMATH_CALUDE_solution_difference_l2231_223147

theorem solution_difference (a b : ℝ) : 
  a ≠ b ∧ 
  (6 * a - 18) / (a^2 + 3 * a - 18) = a + 3 ∧
  (6 * b - 18) / (b^2 + 3 * b - 18) = b + 3 ∧
  a > b →
  a - b = 3 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2231_223147


namespace NUMINAMATH_CALUDE_delacroix_band_size_l2231_223166

theorem delacroix_band_size (n : ℕ) : 
  (∃ k : ℕ, 30 * n = 28 * k + 6) →
  30 * n < 1200 →
  (∀ m : ℕ, (∃ j : ℕ, 30 * m = 28 * j + 6) → 30 * m < 1200 → 30 * m ≤ 30 * n) →
  30 * n = 930 :=
by sorry

end NUMINAMATH_CALUDE_delacroix_band_size_l2231_223166


namespace NUMINAMATH_CALUDE_unique_solution_l2231_223116

def equation1 (x y : ℝ) : Prop := 3 * x + 4 * y = 26

def equation2 (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 1)^2) + Real.sqrt ((x - 10)^2 + (y - 5)^2) = 10

theorem unique_solution :
  ∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 p.1 p.2 ∧ p = (6, 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2231_223116


namespace NUMINAMATH_CALUDE_chicken_cost_per_person_l2231_223124

def grocery_cost : ℝ := 16
def beef_price_per_pound : ℝ := 4
def beef_pounds : ℝ := 3
def oil_price : ℝ := 1
def number_of_people : ℕ := 3

theorem chicken_cost_per_person (chicken_cost : ℝ) : 
  chicken_cost = grocery_cost - (beef_price_per_pound * beef_pounds + oil_price) →
  chicken_cost / number_of_people = 1 := by sorry

end NUMINAMATH_CALUDE_chicken_cost_per_person_l2231_223124


namespace NUMINAMATH_CALUDE_cd_player_only_percentage_l2231_223163

theorem cd_player_only_percentage
  (power_windows : ℝ)
  (anti_lock_brakes : ℝ)
  (cd_player : ℝ)
  (gps_system : ℝ)
  (pw_abs : ℝ)
  (abs_cd : ℝ)
  (pw_cd : ℝ)
  (gps_abs : ℝ)
  (gps_cd : ℝ)
  (pw_gps : ℝ)
  (h1 : power_windows = 60)
  (h2 : anti_lock_brakes = 40)
  (h3 : cd_player = 75)
  (h4 : gps_system = 50)
  (h5 : pw_abs = 10)
  (h6 : abs_cd = 15)
  (h7 : pw_cd = 20)
  (h8 : gps_abs = 12)
  (h9 : gps_cd = 18)
  (h10 : pw_gps = 25)
  (h11 : ∀ x, x ≤ 100) -- Assuming percentages are ≤ 100%
  : cd_player - (abs_cd + pw_cd + gps_cd) = 22 :=
by sorry


end NUMINAMATH_CALUDE_cd_player_only_percentage_l2231_223163


namespace NUMINAMATH_CALUDE_average_speed_problem_l2231_223111

theorem average_speed_problem (D : ℝ) (S : ℝ) (h1 : D > 0) :
  (0.4 * D / S + 0.6 * D / 60) / D = 1 / 50 →
  S = 40 := by
sorry

end NUMINAMATH_CALUDE_average_speed_problem_l2231_223111


namespace NUMINAMATH_CALUDE_pizza_pooling_benefit_l2231_223136

/-- Represents a square pizza with side length and price --/
structure Pizza where
  side : ℕ
  price : ℕ

/-- Calculates the area of a square pizza --/
def pizzaArea (p : Pizza) : ℕ := p.side * p.side

/-- Calculates the number of pizzas that can be bought with a given amount of money --/
def pizzaCount (p : Pizza) (money : ℕ) : ℕ := money / p.price

/-- The small pizza option --/
def smallPizza : Pizza := { side := 6, price := 10 }

/-- The large pizza option --/
def largePizza : Pizza := { side := 9, price := 20 }

/-- The amount of money each friend has --/
def individualMoney : ℕ := 30

/-- The total amount of money when pooled --/
def pooledMoney : ℕ := 2 * individualMoney

theorem pizza_pooling_benefit :
  pizzaArea largePizza * pizzaCount largePizza pooledMoney -
  2 * (pizzaArea smallPizza * pizzaCount smallPizza individualMoney) = 135 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pooling_benefit_l2231_223136


namespace NUMINAMATH_CALUDE_identify_radioactive_balls_l2231_223143

/-- A device that tests two balls for radioactivity -/
structure RadioactivityTester :=
  (test : Fin 100 → Fin 100 → Bool)
  (test_correct : ∀ a b, test a b = true ↔ (a.val < 51 ∧ b.val < 51))

/-- A strategy to identify radioactive balls -/
def IdentificationStrategy := RadioactivityTester → Fin 100 → Bool

/-- The number of tests performed by a strategy -/
def num_tests (strategy : IdentificationStrategy) (tester : RadioactivityTester) : ℕ :=
  sorry

theorem identify_radioactive_balls :
  ∃ (strategy : IdentificationStrategy),
    ∀ (tester : RadioactivityTester),
      (∀ i, strategy tester i = true ↔ i.val < 51) ∧
      num_tests strategy tester ≤ 145 :=
sorry

end NUMINAMATH_CALUDE_identify_radioactive_balls_l2231_223143


namespace NUMINAMATH_CALUDE_nancy_homework_problem_l2231_223195

theorem nancy_homework_problem (finished : ℕ) (pages_left : ℕ) (problems_per_page : ℕ) : 
  finished = 47 → pages_left = 6 → problems_per_page = 9 →
  finished + pages_left * problems_per_page = 101 := by
  sorry

end NUMINAMATH_CALUDE_nancy_homework_problem_l2231_223195


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2231_223115

theorem no_solution_for_equation : ¬∃ (a b : ℕ), 2 * a^2 + 1 = 4 * b^2 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2231_223115


namespace NUMINAMATH_CALUDE_mary_max_earnings_l2231_223157

/-- Calculates Mary's weekly earnings based on her work hours and pay rates. -/
def maryEarnings (maxHours : Nat) (regularRate : ℚ) (overtimeRate : ℚ) (additionalRate : ℚ) : ℚ :=
  let regularHours := min maxHours 40
  let overtimeHours := min (maxHours - regularHours) 20
  let additionalHours := maxHours - regularHours - overtimeHours
  regularHours * regularRate + overtimeHours * overtimeRate + additionalHours * additionalRate

/-- Theorem stating Mary's earnings for working the maximum hours in a week. -/
theorem mary_max_earnings :
  let maxHours : Nat := 70
  let regularRate : ℚ := 10
  let overtimeRate : ℚ := regularRate * (1 + 30/100)
  let additionalRate : ℚ := regularRate * (1 + 60/100)
  maryEarnings maxHours regularRate overtimeRate additionalRate = 820 := by
  sorry


end NUMINAMATH_CALUDE_mary_max_earnings_l2231_223157


namespace NUMINAMATH_CALUDE_identity_holds_iff_k_equals_negative_one_l2231_223181

theorem identity_holds_iff_k_equals_negative_one :
  ∀ k : ℝ, (∀ a b c : ℝ, (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) + k * a * b * c) ↔ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_identity_holds_iff_k_equals_negative_one_l2231_223181


namespace NUMINAMATH_CALUDE_probability_ratio_l2231_223138

/-- The number of balls -/
def num_balls : ℕ := 24

/-- The number of bins -/
def num_bins : ℕ := 6

/-- The probability of the first distribution (6-6-3-3-3-3) -/
noncomputable def p : ℝ := sorry

/-- The probability of the second distribution (4-4-4-4-4-4) -/
noncomputable def q : ℝ := sorry

/-- Theorem stating that the ratio of probabilities p and q is 12 -/
theorem probability_ratio : p / q = 12 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l2231_223138


namespace NUMINAMATH_CALUDE_intersection_point_sum_l2231_223112

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (4, 4)
def D : ℝ × ℝ := (5, 2)

-- Define the quadrilateral
def ABCD : Set (ℝ × ℝ) := {A, B, C, D}

-- Define a function to calculate the area of a quadrilateral
def quadrilateralArea (q : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a function to check if a point is on line CD
def onLineCD (p : ℝ × ℝ) : Prop := sorry

-- Define a function to check if a line through A and a point divides ABCD into equal areas
def dividesEqually (p : ℝ × ℝ) : Prop := sorry

-- Define a function to check if a fraction is in lowest terms
def lowestTerms (p q : ℤ) : Prop := sorry

theorem intersection_point_sum :
  ∀ p q r s : ℤ,
  onLineCD (p/q, r/s) →
  dividesEqually (p/q, r/s) →
  lowestTerms p q →
  lowestTerms r s →
  p + q + r + s = 60 := by sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l2231_223112


namespace NUMINAMATH_CALUDE_jurassic_zoo_bill_l2231_223169

/-- The Jurassic Zoo billing problem -/
theorem jurassic_zoo_bill :
  let adult_price : ℕ := 8
  let child_price : ℕ := 4
  let total_people : ℕ := 201
  let total_children : ℕ := 161
  let total_adults : ℕ := total_people - total_children
  let adults_bill : ℕ := total_adults * adult_price
  let children_bill : ℕ := total_children * child_price
  let total_bill : ℕ := adults_bill + children_bill
  total_bill = 964 := by
  sorry

end NUMINAMATH_CALUDE_jurassic_zoo_bill_l2231_223169


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l2231_223105

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  9*x + 40*y + 18 = 0 ∨ x = -2

-- Theorem statement
theorem circle_and_tangent_line :
  -- Given conditions
  (circle_C 0 0) ∧
  (circle_C 6 0) ∧
  (circle_C 0 8) ∧
  -- Line l passes through (-2, 0)
  (line_l (-2) 0) ∧
  -- Line l is tangent to circle C
  (∃ (x y : ℝ), circle_C x y ∧ line_l x y ∧
    (∀ (x' y' : ℝ), line_l x' y' → (x' - x)^2 + (y' - y)^2 > 0 ∨ (x' = x ∧ y' = y))) →
  -- Conclusion: The equations of C and l are correct
  (∀ (x y : ℝ), circle_C x y ↔ (x - 3)^2 + (y - 4)^2 = 25) ∧
  (∀ (x y : ℝ), line_l x y ↔ (9*x + 40*y + 18 = 0 ∨ x = -2)) :=
by sorry


end NUMINAMATH_CALUDE_circle_and_tangent_line_l2231_223105


namespace NUMINAMATH_CALUDE_sports_club_size_l2231_223199

/-- The number of members in a sports club -/
def sports_club_members (badminton tennis both neither : ℕ) : ℕ :=
  badminton + tennis - both + neither

/-- Theorem: The sports club has 40 members -/
theorem sports_club_size :
  sports_club_members 20 18 3 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_size_l2231_223199


namespace NUMINAMATH_CALUDE_floor_neg_seven_thirds_l2231_223190

theorem floor_neg_seven_thirds : ⌊(-7 : ℝ) / 3⌋ = -3 := by sorry

end NUMINAMATH_CALUDE_floor_neg_seven_thirds_l2231_223190


namespace NUMINAMATH_CALUDE_symmetric_complex_division_l2231_223162

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_to_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- Given two complex numbers z₁ and z₂ symmetric with respect to the imaginary axis,
    where z₁ = -1 + i, prove that z₁ / z₂ = i. -/
theorem symmetric_complex_division (z₁ z₂ : ℂ) 
    (h_sym : symmetric_to_imaginary_axis z₁ z₂) 
    (h_z₁ : z₁ = -1 + Complex.I) : 
  z₁ / z₂ = Complex.I := by
  sorry


end NUMINAMATH_CALUDE_symmetric_complex_division_l2231_223162


namespace NUMINAMATH_CALUDE_city_population_problem_l2231_223141

theorem city_population_problem :
  ∃ (N : ℕ),
    (∃ (x : ℕ), N = x^2) ∧
    (∃ (y : ℕ), N + 100 = y^2 + 1) ∧
    (∃ (z : ℕ), N + 200 = z^2) ∧
    (∃ (k : ℕ), N = 7 * k) :=
by
  sorry

end NUMINAMATH_CALUDE_city_population_problem_l2231_223141


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l2231_223140

/-- Represents a box with a square base -/
structure Box where
  base_length : ℝ
  height : ℝ

/-- Calculates the area of wrapping paper needed for a given box -/
def wrapping_paper_area (box : Box) : ℝ :=
  2 * (box.base_length + box.height)^2

/-- Theorem stating that the area of the wrapping paper is 2(w+h)^2 -/
theorem wrapping_paper_area_theorem (box : Box) :
  wrapping_paper_area box = 2 * (box.base_length + box.height)^2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l2231_223140


namespace NUMINAMATH_CALUDE_grant_total_earnings_l2231_223114

/-- Grant's earnings over four months as a freelance math worker -/
def grant_earnings (X Y Z W : ℕ) : ℕ :=
  let month1 := X
  let month2 := 3 * X + Y
  let month3 := 2 * month2 - Z
  let month4 := (month1 + month2 + month3) / 3 + W
  month1 + month2 + month3 + month4

/-- Theorem stating Grant's total earnings over four months -/
theorem grant_total_earnings :
  grant_earnings 350 30 20 50 = 5810 := by
  sorry

end NUMINAMATH_CALUDE_grant_total_earnings_l2231_223114


namespace NUMINAMATH_CALUDE_min_cost_for_20_oranges_l2231_223133

/-- Represents a discount scheme for oranges -/
structure DiscountScheme where
  quantity : ℕ
  price : ℕ

/-- Calculates the cost of oranges given a discount scheme and number of groups -/
def calculateCost (scheme : DiscountScheme) (groups : ℕ) : ℕ :=
  scheme.price * groups

/-- Finds the minimum cost for a given number of oranges using available discount schemes -/
def minCostForOranges (schemes : List DiscountScheme) (targetOranges : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem min_cost_for_20_oranges :
  let schemes := [
    DiscountScheme.mk 4 12,
    DiscountScheme.mk 7 21
  ]
  minCostForOranges schemes 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_20_oranges_l2231_223133


namespace NUMINAMATH_CALUDE_incenter_coords_l2231_223106

/-- Triangle DEF with side lengths d, e, f -/
structure Triangle where
  d : ℝ
  e : ℝ
  f : ℝ

/-- Barycentric coordinates -/
structure BarycentricCoords where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : BarycentricCoords :=
  sorry

/-- The theorem stating that the barycentric coordinates of the incenter
    of triangle DEF with side lengths 8, 15, 17 are (8/40, 15/40, 17/40) -/
theorem incenter_coords :
  let t : Triangle := { d := 8, e := 15, f := 17 }
  let i : BarycentricCoords := incenter t
  i.x = 8/40 ∧ i.y = 15/40 ∧ i.z = 17/40 ∧ i.x + i.y + i.z = 1 := by
  sorry

end NUMINAMATH_CALUDE_incenter_coords_l2231_223106


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l2231_223109

theorem volleyball_team_starters (n : ℕ) (q : ℕ) (s : ℕ) (h1 : n = 16) (h2 : q = 4) (h3 : s = 6) :
  (Nat.choose (n - q) s) + q * (Nat.choose (n - q) (s - 1)) = 4092 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l2231_223109


namespace NUMINAMATH_CALUDE_sin_sixteen_thirds_pi_l2231_223123

theorem sin_sixteen_thirds_pi : Real.sin (16 * Real.pi / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixteen_thirds_pi_l2231_223123


namespace NUMINAMATH_CALUDE_min_guesses_theorem_l2231_223165

/-- The minimum number of guesses required to determine the leader's binary string -/
def minGuesses (n k : ℕ+) : ℕ :=
  if n = 2 * k then 2 else 1

/-- Theorem stating the minimum number of guesses required -/
theorem min_guesses_theorem (n k : ℕ+) (h : n > k) :
  minGuesses n k = 2 ↔ n = 2 * k :=
sorry

end NUMINAMATH_CALUDE_min_guesses_theorem_l2231_223165


namespace NUMINAMATH_CALUDE_odd_number_as_difference_of_squares_l2231_223131

theorem odd_number_as_difference_of_squares (n : ℕ) (hn : n > 0) :
  ∃! (x y : ℕ), (2 * n + 1 : ℕ) = x^2 - y^2 ∧ x = n + 1 ∧ y = n :=
by sorry

end NUMINAMATH_CALUDE_odd_number_as_difference_of_squares_l2231_223131


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l2231_223127

/-- Calculates the simple interest rate given loan amounts and repayment details -/
theorem calculate_interest_rate 
  (initial_loan : ℝ) 
  (additional_loan : ℝ) 
  (total_repayment : ℝ) 
  (initial_period : ℝ) 
  (total_period : ℝ)
  (h1 : initial_loan = 10000)
  (h2 : additional_loan = 12000)
  (h3 : total_repayment = 27160)
  (h4 : initial_period = 2)
  (h5 : total_period = 5) :
  ∃ r : ℝ, r = 6 ∧ 
    initial_loan * (1 + r / 100 * initial_period) + 
    (initial_loan + additional_loan) * (1 + r / 100 * (total_period - initial_period)) = 
    total_repayment :=
by sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l2231_223127


namespace NUMINAMATH_CALUDE_insects_distribution_l2231_223108

/-- The number of insects collected by boys -/
def boys_insects : ℕ := 200

/-- The number of insects collected by girls -/
def girls_insects : ℕ := 300

/-- The number of groups the class is divided into -/
def num_groups : ℕ := 4

/-- The total number of insects collected -/
def total_insects : ℕ := boys_insects + girls_insects

/-- The number of insects per group -/
def insects_per_group : ℕ := total_insects / num_groups

theorem insects_distribution :
  insects_per_group = 125 := by sorry

end NUMINAMATH_CALUDE_insects_distribution_l2231_223108


namespace NUMINAMATH_CALUDE_real_y_condition_l2231_223187

theorem real_y_condition (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 2 = 0) ↔ 
  (x ≤ (1 - Real.sqrt 7) / 3 ∨ x ≥ (1 + Real.sqrt 7) / 3) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l2231_223187


namespace NUMINAMATH_CALUDE_unique_valid_assignment_l2231_223145

/-- Represents the letters used in the triangle puzzle -/
inductive Letter
| A | B | C | D | E | F

/-- Represents the possible values for each letter -/
def LetterValue := Fin 6

/-- A function that assigns values to letters -/
def Assignment := Letter → LetterValue

/-- Checks if an assignment is valid according to the puzzle rules -/
def is_valid_assignment (f : Assignment) : Prop :=
  -- All values are distinct
  (∀ x y : Letter, x ≠ y → f x ≠ f y) ∧
  -- The sum of D, E, and B equals 14
  (f Letter.D).val + (f Letter.E).val + (f Letter.B).val = 14 ∧
  -- F is the sum of D and E minus the sum of A, B, and C
  (f Letter.F).val = (f Letter.D).val + (f Letter.E).val - ((f Letter.A).val + (f Letter.B).val + (f Letter.C).val)

/-- The unique valid assignment for the puzzle -/
def unique_assignment : Assignment :=
  fun l => match l with
  | Letter.A => ⟨0, by simp⟩  -- 1
  | Letter.B => ⟨2, by simp⟩  -- 3
  | Letter.C => ⟨1, by simp⟩  -- 2
  | Letter.D => ⟨4, by simp⟩  -- 5
  | Letter.E => ⟨5, by simp⟩  -- 6
  | Letter.F => ⟨3, by simp⟩  -- 4

/-- Theorem stating that the unique_assignment is the only valid assignment -/
theorem unique_valid_assignment :
  is_valid_assignment unique_assignment ∧
  ∀ f : Assignment, is_valid_assignment f → f = unique_assignment :=
sorry

end NUMINAMATH_CALUDE_unique_valid_assignment_l2231_223145


namespace NUMINAMATH_CALUDE_f_of_f_of_2_equals_394_l2231_223122

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 - 6

-- State the theorem
theorem f_of_f_of_2_equals_394 : f (f 2) = 394 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_2_equals_394_l2231_223122


namespace NUMINAMATH_CALUDE_original_price_from_decreased_price_l2231_223197

/-- Proves that if an article's price after a 24% decrease is 684, then its original price was 900. -/
theorem original_price_from_decreased_price (decreased_price : ℝ) (decrease_percentage : ℝ) :
  decreased_price = 684 ∧ decrease_percentage = 24 →
  (1 - decrease_percentage / 100) * 900 = decreased_price := by
  sorry

#check original_price_from_decreased_price

end NUMINAMATH_CALUDE_original_price_from_decreased_price_l2231_223197


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2231_223192

/-- Given a quadratic equation ax^2 + bx + c = 0, this function returns the triple (a, b, c) -/
def quadraticCoefficients (f : ℝ → ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem quadratic_equation_coefficients :
  quadraticCoefficients (fun x => x^2 - x) = (1, -1, 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2231_223192


namespace NUMINAMATH_CALUDE_trajectory_of_shared_focus_l2231_223191

/-- Given a parabola and a hyperbola sharing a focus, prove the trajectory of (m,n) -/
theorem trajectory_of_shared_focus (n m : ℝ) : 
  n < 0 → 
  (∃ (x y : ℝ), y^2 = 2*n*x) → 
  (∃ (x y : ℝ), x^2/4 - y^2/m^2 = 1) → 
  (∃ (f : ℝ × ℝ), f ∈ {p : ℝ × ℝ | p.1^2/(2*n) = p.2^2/m^2}) →
  n^2/16 - m^2/4 = 1 ∧ n < 0 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_shared_focus_l2231_223191


namespace NUMINAMATH_CALUDE_only_one_and_two_satisfy_property_l2231_223186

/-- A function that checks if a number has n-1 digits of 1 and one digit of 7 -/
def has_n_minus_1_ones_and_one_seven (x : ℕ) (n : ℕ) : Prop := sorry

/-- A function that generates all numbers with n-1 digits of 1 and one digit of 7 -/
def numbers_with_n_minus_1_ones_and_one_seven (n : ℕ) : Set ℕ := sorry

theorem only_one_and_two_satisfy_property :
  ∀ n : ℕ, (∀ x ∈ numbers_with_n_minus_1_ones_and_one_seven n, Nat.Prime x) ↔ (n = 1 ∨ n = 2) :=
sorry

end NUMINAMATH_CALUDE_only_one_and_two_satisfy_property_l2231_223186


namespace NUMINAMATH_CALUDE_smallest_c_inequality_l2231_223137

theorem smallest_c_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (∀ c : ℝ, c > 0 → c * |x^(2/3) - y^(2/3)| + (x*y)^(1/3) ≥ (x^(2/3) + y^(2/3))/2) ∧
  (∀ ε > 0, ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ ((1/2 - ε) * |x^(2/3) - y^(2/3)| + (x*y)^(1/3) < (x^(2/3) + y^(2/3))/2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_inequality_l2231_223137


namespace NUMINAMATH_CALUDE_square_triangle_perimeter_ratio_l2231_223110

theorem square_triangle_perimeter_ratio (s_square s_triangle : ℝ) 
  (h_positive_square : s_square > 0)
  (h_positive_triangle : s_triangle > 0)
  (h_equal_perimeter : 4 * s_square = 3 * s_triangle) :
  s_triangle / s_square = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_perimeter_ratio_l2231_223110


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l2231_223126

theorem quadratic_root_implies_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - 2 = 0 ∧ x = 2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l2231_223126


namespace NUMINAMATH_CALUDE_max_value_rational_function_l2231_223152

theorem max_value_rational_function (x : ℝ) (h : x < -1) :
  (x^2 + 7*x + 10) / (x + 1) ≤ 1 ∧
  (x^2 + 7*x + 10) / (x + 1) = 1 ↔ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_rational_function_l2231_223152


namespace NUMINAMATH_CALUDE_triple_application_of_f_l2231_223155

def f (p : ℝ) : ℝ := 2 * p + 20

theorem triple_application_of_f :
  ∃ p : ℝ, f (f (f p)) = -4 ∧ p = -18 := by
  sorry

end NUMINAMATH_CALUDE_triple_application_of_f_l2231_223155


namespace NUMINAMATH_CALUDE_reflection_of_P_l2231_223159

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point P -/
def P : ℝ × ℝ := (3, -5)

theorem reflection_of_P :
  reflect_y P = (-3, -5) := by sorry

end NUMINAMATH_CALUDE_reflection_of_P_l2231_223159


namespace NUMINAMATH_CALUDE_determinant_trig_matrix_l2231_223194

theorem determinant_trig_matrix (α β : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.cos (α + β), Real.sin (α + β), -Real.sin α],
    ![-Real.sin β, Real.cos β, 0],
    ![Real.sin α * Real.cos β, Real.sin α * Real.sin β, Real.cos α]
  ]
  Matrix.det M = 1 := by sorry

end NUMINAMATH_CALUDE_determinant_trig_matrix_l2231_223194


namespace NUMINAMATH_CALUDE_system_solution_l2231_223168

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (2 * x - Real.sqrt (x * y) - 4 * Real.sqrt (x / y) + 2 = 0) ∧
  (2 * x^2 + x^2 * y^4 = 18 * y^2) →
  ((x = 2 ∧ y = 2) ∨ (x = Real.sqrt (Real.sqrt 286) / 4 ∧ y = Real.sqrt (Real.sqrt 286))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2231_223168


namespace NUMINAMATH_CALUDE_waiting_time_is_twenty_l2231_223167

/-- Represents the time components of Mary's trip to the airport -/
structure TripTime where
  uber_to_house : ℕ
  uber_to_airport_multiplier : ℕ
  bag_check : ℕ
  security_multiplier : ℕ
  total_trip_time : ℕ

/-- Calculates the waiting time for the flight to start boarding -/
def waiting_time (t : TripTime) : ℕ :=
  let uber_to_airport := t.uber_to_house * t.uber_to_airport_multiplier
  let security := t.bag_check * t.security_multiplier
  let total_pre_wait := t.uber_to_house + uber_to_airport + t.bag_check + security
  let remaining_time := t.total_trip_time - total_pre_wait
  remaining_time / 3

/-- Theorem stating that the waiting time for the flight to start boarding is 20 minutes -/
theorem waiting_time_is_twenty (t : TripTime) 
  (h1 : t.uber_to_house = 10)
  (h2 : t.uber_to_airport_multiplier = 5)
  (h3 : t.bag_check = 15)
  (h4 : t.security_multiplier = 3)
  (h5 : t.total_trip_time = 180) : 
  waiting_time t = 20 := by
  sorry

end NUMINAMATH_CALUDE_waiting_time_is_twenty_l2231_223167


namespace NUMINAMATH_CALUDE_max_value_of_f_l2231_223172

noncomputable def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem max_value_of_f :
  ∃ (M : ℝ), M = 23 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2231_223172


namespace NUMINAMATH_CALUDE_megawheel_capacity_l2231_223117

/-- The Megawheel problem -/
theorem megawheel_capacity (total_seats : ℕ) (total_people : ℕ) (people_per_seat : ℕ) 
  (h1 : total_seats = 15)
  (h2 : total_people = 75)
  (h3 : people_per_seat * total_seats = total_people) :
  people_per_seat = 5 := by
  sorry

end NUMINAMATH_CALUDE_megawheel_capacity_l2231_223117


namespace NUMINAMATH_CALUDE_chord_bisection_range_l2231_223150

theorem chord_bisection_range :
  ∀ (x₀ : ℝ),
  (∃ (A B : ℝ × ℝ),
    (A.1^2 + A.2^2 = 1) ∧
    (B.1^2 + B.2^2 = 1) ∧
    (∃ (t : ℝ), A.1 + t * (B.1 - A.1) = x₀ / 2 ∧ A.2 + t * (B.2 - A.2) = 1 - x₀) ∧
    ((B.2 - A.2) * (x₀ / 2) = (A.1 - B.1) * (1 - x₀))) →
  0 < x₀ ∧ x₀ < 8/5 :=
by sorry

end NUMINAMATH_CALUDE_chord_bisection_range_l2231_223150


namespace NUMINAMATH_CALUDE_mark_leftover_money_l2231_223188

def original_wage : ℝ := 40
def raise_percentage : ℝ := 0.05
def hours_per_day : ℝ := 8
def days_per_week : ℝ := 5
def old_bills : ℝ := 600
def personal_trainer_cost : ℝ := 100

def new_wage : ℝ := original_wage * (1 + raise_percentage)
def daily_earnings : ℝ := new_wage * hours_per_day
def weekly_earnings : ℝ := daily_earnings * days_per_week
def weekly_expenses : ℝ := old_bills + personal_trainer_cost

theorem mark_leftover_money : weekly_earnings - weekly_expenses = 980 := by
  sorry

end NUMINAMATH_CALUDE_mark_leftover_money_l2231_223188


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2231_223171

/-- A line passing through (0, 7) perpendicular to 2x - 6y - 14 = 0 has equation y + 3x - 7 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) : 
  (∃ (m b : ℝ), y = m * x + b ∧ (0, 7) ∈ {(x, y) | y = m * x + b}) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 2 * x₁ - 6 * y₁ - 14 = 0 ∧ 2 * x₂ - 6 * y₂ - 14 = 0 → 
    (y₂ - y₁) * (y - 7) = -(x₂ - x₁) * x) → 
  y + 3 * x - 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2231_223171


namespace NUMINAMATH_CALUDE_negative_integer_sum_square_twelve_l2231_223120

theorem negative_integer_sum_square_twelve (M : ℤ) : 
  M < 0 → M^2 + M = 12 → M = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_sum_square_twelve_l2231_223120


namespace NUMINAMATH_CALUDE_max_teams_is_eight_l2231_223173

/-- Represents the number of teams that can be formed given the number of climbers in each skill level and the required composition of each team. -/
def max_teams (advanced intermediate beginner : ℕ) 
              (adv_per_team int_per_team beg_per_team : ℕ) : ℕ :=
  min (advanced / adv_per_team)
      (min (intermediate / int_per_team)
           (beginner / beg_per_team))

/-- Theorem stating that the maximum number of teams that can be formed is 8. -/
theorem max_teams_is_eight : 
  max_teams 45 70 57 5 8 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_teams_is_eight_l2231_223173


namespace NUMINAMATH_CALUDE_amount_after_two_years_l2231_223142

/-- The amount after n years given an initial amount and annual increase rate -/
def amount_after_years (initial_amount : ℝ) (increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + increase_rate) ^ years

/-- Theorem stating that 62000 increasing by 1/8 annually becomes 78468.75 after 2 years -/
theorem amount_after_two_years :
  let initial_amount : ℝ := 62000
  let increase_rate : ℝ := 1/8
  let years : ℕ := 2
  amount_after_years initial_amount increase_rate years = 78468.75 := by
  sorry

#check amount_after_two_years

end NUMINAMATH_CALUDE_amount_after_two_years_l2231_223142


namespace NUMINAMATH_CALUDE_factor_expression_l2231_223156

theorem factor_expression (x : ℝ) : 92 * x^3 - 184 * x^6 = 92 * x^3 * (1 - 2 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2231_223156


namespace NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l2231_223121

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem P_intersect_Q_eq_P : P ∩ Q = P := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l2231_223121


namespace NUMINAMATH_CALUDE_pentagon_square_side_ratio_l2231_223151

theorem pentagon_square_side_ratio :
  let pentagon_perimeter : ℝ := 100
  let square_perimeter : ℝ := 100
  let pentagon_side : ℝ := pentagon_perimeter / 5
  let square_side : ℝ := square_perimeter / 4
  pentagon_side / square_side = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_square_side_ratio_l2231_223151


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2231_223177

/-- A hyperbola with center at the origin, a focus at (√2, 0), and the distance
    from this focus to an asymptote being 1 has the equation x^2 - y^2 = 1 -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 - y^2 = 1) ↔
  (0, 0) ∈ C ∧ 
  F = (Real.sqrt 2, 0) ∧ 
  F ∈ C ∧
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ C → a * y = b * x ∨ a * y = -b * x) ∧
    (abs (b * Real.sqrt 2) / Real.sqrt (a^2 + b^2) = 1)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2231_223177


namespace NUMINAMATH_CALUDE_roy_school_days_l2231_223107

/-- Represents the number of hours Roy spends on sports activities in school each day -/
def daily_sports_hours : ℕ := 2

/-- Represents the number of days Roy missed within a week -/
def missed_days : ℕ := 2

/-- Represents the total hours Roy spent on sports in school for the week -/
def weekly_sports_hours : ℕ := 6

/-- Represents the number of days Roy goes to school in a week -/
def school_days : ℕ := 5

theorem roy_school_days : 
  daily_sports_hours * (school_days - missed_days) = weekly_sports_hours ∧ 
  school_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_roy_school_days_l2231_223107


namespace NUMINAMATH_CALUDE_curve_satisfies_conditions_l2231_223184

/-- The curve that satisfies the given conditions -/
def curve (x y : ℝ) : Prop := x * y = 4

/-- The tangent line to the curve at point (x,y) -/
def tangent_line (x y : ℝ) : Set (ℝ × ℝ) :=
  {(t, s) | s - y = -(y / x) * (t - x)}

theorem curve_satisfies_conditions :
  -- The curve passes through (1,4)
  curve 1 4 ∧
  -- For any point (x,y) on the curve, the tangent line intersects
  -- the x-axis at (2x,0) and the y-axis at (0,2y)
  ∀ x y : ℝ, x > 0 → y > 0 → curve x y →
    (2*x, 0) ∈ tangent_line x y ∧ (0, 2*y) ∈ tangent_line x y :=
by sorry


end NUMINAMATH_CALUDE_curve_satisfies_conditions_l2231_223184


namespace NUMINAMATH_CALUDE_douglas_fir_count_l2231_223113

/-- The number of Douglas fir trees in a forest -/
def douglas_fir : ℕ := sorry

/-- The number of ponderosa pine trees in a forest -/
def ponderosa_pine : ℕ := sorry

/-- The total number of trees in the forest -/
def total_trees : ℕ := 850

/-- The cost of a single Douglas fir tree -/
def douglas_fir_cost : ℕ := 300

/-- The cost of a single ponderosa pine tree -/
def ponderosa_pine_cost : ℕ := 225

/-- The total amount paid for all trees -/
def total_cost : ℕ := 217500

theorem douglas_fir_count : 
  douglas_fir = 350 ∧
  douglas_fir + ponderosa_pine = total_trees ∧
  douglas_fir * douglas_fir_cost + ponderosa_pine * ponderosa_pine_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_douglas_fir_count_l2231_223113


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_l2231_223174

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + y = 2 * x * y) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + y' = 2 * x' * y' → x' + 4 * y' ≥ 9/2) ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 2 * x₀ * y₀ ∧ x₀ + 4 * y₀ = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_l2231_223174
