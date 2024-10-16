import Mathlib

namespace NUMINAMATH_CALUDE_sphere_segment_heights_l3494_349407

/-- Given a sphere of radius r intersected by three parallel planes, if the heights
    of the resulting segments form a geometric progression with common ratio q,
    then the height of the first segment (m₁) is equal to 2r(q-1)/(q⁴-1). -/
theorem sphere_segment_heights (r q : ℝ) (h_r : r > 0) (h_q : q > 1) :
  let m₁ := 2 * r * (q - 1) / (q^4 - 1)
  let m₂ := m₁ * q
  let m₃ := m₁ * q^2
  let m₄ := m₁ * q^3
  (m₁ + m₂ + m₃ + m₄ = 2 * r) ∧
  (m₂ / m₁ = q) ∧ (m₃ / m₂ = q) ∧ (m₄ / m₃ = q) :=
by sorry

end NUMINAMATH_CALUDE_sphere_segment_heights_l3494_349407


namespace NUMINAMATH_CALUDE_binomial_150_1_l3494_349446

theorem binomial_150_1 : Nat.choose 150 1 = 150 := by sorry

end NUMINAMATH_CALUDE_binomial_150_1_l3494_349446


namespace NUMINAMATH_CALUDE_line_y_intercept_l3494_349461

/-- A line in the xy-plane is defined by its slope and a point it passes through. 
    This theorem proves that for a line with slope 2 passing through (498, 998), 
    the y-intercept is 2. -/
theorem line_y_intercept (m : ℝ) (x y b : ℝ) :
  m = 2 ∧ x = 498 ∧ y = 998 ∧ y = m * x + b → b = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l3494_349461


namespace NUMINAMATH_CALUDE_max_sum_of_digits_24hour_watch_l3494_349455

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours ≤ 23
  minutes_valid : minutes ≤ 59
  seconds_valid : seconds ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of all digits in a Time24 -/
def totalSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes + sumOfDigits t.seconds

/-- The theorem to be proved -/
theorem max_sum_of_digits_24hour_watch :
  ∃ (t : Time24), ∀ (t' : Time24), totalSumOfDigits t' ≤ totalSumOfDigits t ∧ totalSumOfDigits t = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_24hour_watch_l3494_349455


namespace NUMINAMATH_CALUDE_complex_z_in_first_quadrant_l3494_349471

theorem complex_z_in_first_quadrant (z : ℂ) (h : (1 : ℂ) + Complex.I = Complex.I / z) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_z_in_first_quadrant_l3494_349471


namespace NUMINAMATH_CALUDE_horner_v4_equals_80_l3494_349430

/-- Horner's Rule for polynomial evaluation --/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 12x^5 + 60x^4 - 160x^3 + 240x^2 - 192x + 64 --/
def f (x : ℝ) : ℝ :=
  x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

/-- The coefficients of the polynomial in reverse order --/
def coeffs : List ℝ := [64, -192, 240, -160, 60, -12, 1]

/-- The value of x for which we're evaluating the polynomial --/
def x : ℝ := 2

/-- The intermediate value v_4 in Horner's Rule calculation --/
def v_4 : ℝ := ((-80 * x) + 240)

theorem horner_v4_equals_80 : v_4 = 80 := by
  sorry

#eval v_4

end NUMINAMATH_CALUDE_horner_v4_equals_80_l3494_349430


namespace NUMINAMATH_CALUDE_prob_odd_sum_given_even_product_l3494_349460

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 3 / 8

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 5 / 8

/-- The number of ways to get an odd sum with one odd die -/
def odd_sum_one_odd : ℕ := num_dice * 3 * 5^4

/-- The number of ways to get an odd sum with three odd dice -/
def odd_sum_three_odd : ℕ := (num_dice.choose 3) * 3^3 * 5^2

/-- The number of ways to get an odd sum with all odd dice -/
def odd_sum_all_odd : ℕ := 3^5

/-- The total number of favorable outcomes (odd sum) -/
def total_favorable : ℕ := odd_sum_one_odd + odd_sum_three_odd + odd_sum_all_odd

/-- The total number of possible outcomes where the product is even -/
def total_possible : ℕ := 8^5 - (3/8)^5 * 8^5

/-- The probability of getting an odd sum given that the product is even -/
theorem prob_odd_sum_given_even_product :
  (total_favorable : ℚ) / total_possible =
  (5 * 3 * 5^4 + 10 * 27 * 25 + 243) / (8^5 - (3/8)^5 * 8^5) :=
by sorry

end NUMINAMATH_CALUDE_prob_odd_sum_given_even_product_l3494_349460


namespace NUMINAMATH_CALUDE_root_product_theorem_l3494_349403

theorem root_product_theorem (a b : ℂ) : 
  (a^4 + a^3 - 1 = 0) → 
  (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3494_349403


namespace NUMINAMATH_CALUDE_toothpaste_amount_l3494_349493

/-- The amount of toothpaste used by Anne's dad per brushing -/
def dadUsage : ℕ := 3

/-- The amount of toothpaste used by Anne's mom per brushing -/
def momUsage : ℕ := 2

/-- The amount of toothpaste used by Anne or her brother per brushing -/
def childUsage : ℕ := 1

/-- The number of times each family member brushes their teeth per day -/
def brushingsPerDay : ℕ := 3

/-- The number of days it takes for the toothpaste to run out -/
def daysUntilEmpty : ℕ := 5

/-- The number of children (Anne and her brother) -/
def numberOfChildren : ℕ := 2

/-- Theorem stating that the amount of toothpaste in the tube is 105 grams -/
theorem toothpaste_amount : 
  dadUsage * brushingsPerDay * daysUntilEmpty + 
  momUsage * brushingsPerDay * daysUntilEmpty + 
  childUsage * brushingsPerDay * daysUntilEmpty * numberOfChildren = 105 := by
  sorry

end NUMINAMATH_CALUDE_toothpaste_amount_l3494_349493


namespace NUMINAMATH_CALUDE_target_digit_is_five_l3494_349415

/-- The decimal expansion of 47/777 -/
def decimal_expansion : ℚ := 47 / 777

/-- The length of the repeating block in the decimal expansion -/
def repeating_block_length : ℕ := 6

/-- The position of the digit we're interested in -/
def target_position : ℕ := 156

/-- The function that returns the nth digit after the decimal point in the decimal expansion -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem target_digit_is_five :
  nth_digit (target_position - 1) = 5 := by sorry

end NUMINAMATH_CALUDE_target_digit_is_five_l3494_349415


namespace NUMINAMATH_CALUDE_pension_calculation_l3494_349418

/-- Given a pension system where:
  * The annual pension is proportional to the square root of years served
  * Serving 'a' additional years increases the pension by 'p' dollars
  * Serving 'b' additional years (b ≠ a) increases the pension by 'q' dollars
This theorem proves that the annual pension can be expressed in terms of a, b, p, and q. -/
theorem pension_calculation (a b p q : ℝ) (h_ab : a ≠ b) :
  ∃ (x y k : ℝ),
    x = k * Real.sqrt y ∧
    x + p = k * Real.sqrt (y + a) ∧
    x + q = k * Real.sqrt (y + b) →
    x = (a * q^2 - b * p^2) / (2 * (b * p - a * q)) :=
sorry

end NUMINAMATH_CALUDE_pension_calculation_l3494_349418


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3494_349436

theorem condition_necessary_not_sufficient : 
  (∀ x : ℝ, x = 0 → (2*x - 1)*x = 0) ∧ 
  ¬(∀ x : ℝ, (2*x - 1)*x = 0 → x = 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3494_349436


namespace NUMINAMATH_CALUDE_point_transformation_l3494_349410

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the initial point A
def A : Point2D := ⟨5, 4⟩

-- Define the transformation function
def transform (p : Point2D) : Point2D :=
  ⟨p.x - 4, p.y - 3⟩

-- State the theorem
theorem point_transformation :
  transform A = Point2D.mk 1 1 := by sorry

end NUMINAMATH_CALUDE_point_transformation_l3494_349410


namespace NUMINAMATH_CALUDE_executive_board_selection_l3494_349481

theorem executive_board_selection (n m : ℕ) (h1 : n = 12) (h2 : m = 5) :
  Nat.choose n m = 792 := by
  sorry

end NUMINAMATH_CALUDE_executive_board_selection_l3494_349481


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3494_349472

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3494_349472


namespace NUMINAMATH_CALUDE_area_between_parallel_chords_l3494_349437

theorem area_between_parallel_chords (r : ℝ) (d : ℝ) (h1 : r = 8) (h2 : d = 8) :
  let chord_length := 2 * Real.sqrt (r ^ 2 - (d / 2) ^ 2)
  let segment_area := (1 / 3) * π * r ^ 2 - (1 / 2) * chord_length * (d / 2)
  2 * segment_area = 32 * Real.sqrt 3 + 64 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_area_between_parallel_chords_l3494_349437


namespace NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l3494_349467

theorem sum_of_two_squares_equivalence (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2*x = u^2 + v^2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l3494_349467


namespace NUMINAMATH_CALUDE_prime_sum_square_cube_l3494_349464

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def validSolution (p q r : ℕ) : Prop :=
  isPrime p ∧ isPrime q ∧ isPrime r ∧ p + q^2 + r^3 = 200

theorem prime_sum_square_cube :
  {(p, q, r) : ℕ × ℕ × ℕ | validSolution p q r} =
  {(167, 5, 2), (71, 11, 2), (23, 13, 2), (71, 2, 5)} :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_square_cube_l3494_349464


namespace NUMINAMATH_CALUDE_recurring_decimal_subtraction_l3494_349479

theorem recurring_decimal_subtraction : 
  (246 : ℚ) / 999 - 135 / 999 - 579 / 999 = -52 / 111 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_subtraction_l3494_349479


namespace NUMINAMATH_CALUDE_quadratic_translation_l3494_349496

-- Define the original quadratic function
def f (x : ℝ) : ℝ := (x - 2009) * (x - 2008) + 4

-- Define the translated function
def g (x : ℝ) : ℝ := f x - 4

-- Theorem statement
theorem quadratic_translation :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ |x₁ - x₂| = 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_translation_l3494_349496


namespace NUMINAMATH_CALUDE_sum_of_altitudes_l3494_349456

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 10 * x + 8 * y = 80

-- Define the triangle formed by the line and coordinate axes
def triangle_vertices : Set (ℝ × ℝ) :=
  {(0, 0), (8, 0), (0, 10)}

-- State the theorem
theorem sum_of_altitudes :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ triangle_vertices → line_equation x y) ∧
    a + b + c = 18 + (80 * Real.sqrt 164) / 164 :=
sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_l3494_349456


namespace NUMINAMATH_CALUDE_average_buying_cost_theorem_ritas_average_buying_cost_l3494_349477

/-- Represents the cost and quantity of an item --/
structure Item where
  quantity : ℕ
  totalCost : ℚ

/-- Calculates the average buying cost per unit across all items --/
def averageBuyingCost (items : List Item) : ℚ :=
  let totalCost := items.map (λ i => i.totalCost) |>.sum
  let totalQuantity := items.map (λ i => i.quantity) |>.sum
  totalCost / totalQuantity

/-- The main theorem stating that the average buying cost is equal to the total cost divided by total quantity --/
theorem average_buying_cost_theorem (itemA itemB itemC : Item) :
  let items := [itemA, itemB, itemC]
  averageBuyingCost items = (itemA.totalCost + itemB.totalCost + itemC.totalCost) / (itemA.quantity + itemB.quantity + itemC.quantity) := by
  sorry

/-- Application of the theorem to Rita's specific case --/
theorem ritas_average_buying_cost :
  let itemA : Item := { quantity := 20, totalCost := 500 }
  let itemB : Item := { quantity := 15, totalCost := 700 }
  let itemC : Item := { quantity := 10, totalCost := 400 }
  let items := [itemA, itemB, itemC]
  averageBuyingCost items = 1600 / 45 := by
  sorry

end NUMINAMATH_CALUDE_average_buying_cost_theorem_ritas_average_buying_cost_l3494_349477


namespace NUMINAMATH_CALUDE_second_tract_length_l3494_349498

/-- Given two rectangular tracts of land with specified dimensions and combined area,
    prove that the length of the second tract is 250 meters. -/
theorem second_tract_length
  (tract1_length : ℝ)
  (tract1_width : ℝ)
  (tract2_width : ℝ)
  (combined_area : ℝ)
  (h1 : tract1_length = 300)
  (h2 : tract1_width = 500)
  (h3 : tract2_width = 630)
  (h4 : combined_area = 307500)
  : ∃ tract2_length : ℝ,
    tract2_length = 250 ∧
    tract1_length * tract1_width + tract2_length * tract2_width = combined_area :=
by
  sorry

end NUMINAMATH_CALUDE_second_tract_length_l3494_349498


namespace NUMINAMATH_CALUDE_largest_inexpressible_is_19_l3494_349423

/-- Represents the value of a coin in soldi -/
inductive Coin : Type
| five : Coin
| six : Coin

/-- Checks if a natural number can be expressed as a sum of multiples of 5 and 6 -/
def canExpress (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

/-- The largest value that cannot be expressed as a sum of multiples of 5 and 6 -/
def largestInexpressible : ℕ := 19

theorem largest_inexpressible_is_19 :
  largestInexpressible = 19 ∧
  ¬(canExpress largestInexpressible) ∧
  ∀ n : ℕ, n > largestInexpressible → n ≤ 50 → canExpress n :=
by sorry

end NUMINAMATH_CALUDE_largest_inexpressible_is_19_l3494_349423


namespace NUMINAMATH_CALUDE_inequality_solution_upper_bound_l3494_349480

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem inequality_solution (x : ℝ) : f x < |x| + 1 ↔ 0 < x ∧ x < 2 := by sorry

-- Part II
theorem upper_bound (x y : ℝ) (h1 : |x - y - 1| ≤ 1/3) (h2 : |2*y + 1| ≤ 1/6) : f x ≤ 5/6 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_upper_bound_l3494_349480


namespace NUMINAMATH_CALUDE_basketball_score_increase_l3494_349490

theorem basketball_score_increase (junior_score : ℕ) (total_score : ℕ) 
  (h1 : junior_score = 260) 
  (h2 : total_score = 572) : 
  (((total_score - junior_score) : ℚ) / junior_score) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_increase_l3494_349490


namespace NUMINAMATH_CALUDE_club_leadership_combinations_l3494_349413

/-- Represents the total number of members in the club -/
def total_members : ℕ := 24

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 12

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 12

/-- Represents the number of age groups -/
def num_age_groups : ℕ := 2

/-- Represents the number of members in each gender and age group combination -/
def members_per_group : ℕ := 6

/-- Theorem stating the number of ways to choose a president and vice-president -/
theorem club_leadership_combinations : 
  (num_boys * members_per_group + num_girls * members_per_group) = 144 := by
  sorry

end NUMINAMATH_CALUDE_club_leadership_combinations_l3494_349413


namespace NUMINAMATH_CALUDE_inequality_implies_a_range_l3494_349465

theorem inequality_implies_a_range (a : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 + 2*a*x + 1 ≥ 0) → a ≥ -5/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_range_l3494_349465


namespace NUMINAMATH_CALUDE_problem_statement_l3494_349441

theorem problem_statement (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3494_349441


namespace NUMINAMATH_CALUDE_inequality_proof_l3494_349421

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x^2 / (y + z) + y^2 / (z + x) + z^2 / (x + y)) ≥ 
  (1/2) * ((x^2 + y^2) / (x + y) + (y^2 + z^2) / (y + z) + (z^2 + x^2) / (z + x)) ∧
  (1/2) * ((x^2 + y^2) / (x + y) + (y^2 + z^2) / (y + z) + (z^2 + x^2) / (z + x)) ≥ (x + y + z) / 2 ∧
  (x + y + z) / 2 ≥ 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3494_349421


namespace NUMINAMATH_CALUDE_boats_by_april_l3494_349434

def boats_in_month (n : Nat) : Nat :=
  match n with
  | 0 => 4  -- January
  | 1 => 2  -- February
  | m + 2 => 3 * boats_in_month (m + 1)  -- March onwards

def total_boats (n : Nat) : Nat :=
  match n with
  | 0 => boats_in_month 0
  | m + 1 => boats_in_month (m + 1) + total_boats m

theorem boats_by_april : total_boats 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_boats_by_april_l3494_349434


namespace NUMINAMATH_CALUDE_count_tree_frogs_l3494_349419

theorem count_tree_frogs (total_frogs poison_frogs wood_frogs : ℕ) 
  (h1 : total_frogs = 78)
  (h2 : poison_frogs = 10)
  (h3 : wood_frogs = 13)
  (h4 : ∃ tree_frogs : ℕ, total_frogs = tree_frogs + poison_frogs + wood_frogs) :
  ∃ tree_frogs : ℕ, tree_frogs = 55 ∧ total_frogs = tree_frogs + poison_frogs + wood_frogs :=
by
  sorry

end NUMINAMATH_CALUDE_count_tree_frogs_l3494_349419


namespace NUMINAMATH_CALUDE_angle_half_in_third_quadrant_l3494_349433

/-- Given an angle α in the second quadrant with |cos(α/2)| = -cos(α/2),
    prove that α/2 is in the third quadrant. -/
theorem angle_half_in_third_quadrant (α : Real) :
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (|Real.cos (α/2)| = -Real.cos (α/2)) →  -- |cos(α/2)| = -cos(α/2)
  (π < α/2 ∧ α/2 < 3*π/2) :=  -- α/2 is in the third quadrant
by sorry

end NUMINAMATH_CALUDE_angle_half_in_third_quadrant_l3494_349433


namespace NUMINAMATH_CALUDE_part1_part2_l3494_349494

-- Define the inequality function
def inequality (a x : ℝ) : Prop := (a * x - 1) * (x + 1) > 0

-- Part 1: If the solution set is {x | -1 < x < -1/2}, then a = -2
theorem part1 (a : ℝ) : 
  (∀ x, inequality a x ↔ (-1 < x ∧ x < -1/2)) → a = -2 := 
sorry

-- Part 2: Solution sets for a ≤ 0
theorem part2 (a : ℝ) (h : a ≤ 0) : 
  (∀ x, inequality a x ↔ 
    (a < -1 ∧ -1 < x ∧ x < 1/a) ∨
    (a = -1 ∧ False) ∨
    (-1 < a ∧ a < 0 ∧ 1/a < x ∧ x < -1) ∨
    (a = 0 ∧ x < -1)) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3494_349494


namespace NUMINAMATH_CALUDE_equation_solution_l3494_349485

theorem equation_solution :
  ∃! (x : ℝ), x ≠ 0 ∧ (7*x)^5 = (14*x)^4 ∧ x = 16/7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3494_349485


namespace NUMINAMATH_CALUDE_squirrel_acorns_l3494_349417

/- Define the initial number of acorns -/
def initial_acorns : ℕ := 210

/- Define the number of parts the pile was divided into -/
def num_parts : ℕ := 3

/- Define the number of acorns left in each part after removal -/
def acorns_per_part : ℕ := 60

/- Define the total number of acorns removed -/
def total_removed : ℕ := 30

/- Theorem statement -/
theorem squirrel_acorns : 
  (initial_acorns / num_parts - acorns_per_part) * num_parts = total_removed ∧
  initial_acorns % num_parts = 0 := by
  sorry

#check squirrel_acorns

end NUMINAMATH_CALUDE_squirrel_acorns_l3494_349417


namespace NUMINAMATH_CALUDE_value_added_to_half_l3494_349411

theorem value_added_to_half : ∃ v : ℝ, (1/2 : ℝ) * 16 + v = 13 ∧ v = 5 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_half_l3494_349411


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_l3494_349476

theorem min_sum_of_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → (1 / (x + 2) + 1 / (y + 2) = 1 / 6) → 
  ∀ a b : ℝ, a > 0 → b > 0 → (1 / (a + 2) + 1 / (b + 2) = 1 / 6) → 
  x + y ≤ a + b ∧ x + y ≥ 20 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_l3494_349476


namespace NUMINAMATH_CALUDE_farm_animals_l3494_349406

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) (ducks : ℕ) (horses : ℕ) : 
  total_animals = 11 →
  total_legs = 30 →
  ducks + horses = total_animals →
  2 * ducks + 4 * horses = total_legs →
  ducks = 7 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l3494_349406


namespace NUMINAMATH_CALUDE_period_3_odd_function_inequality_l3494_349470

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem period_3_odd_function_inequality (f : ℝ → ℝ) (a : ℝ) 
    (h_periodic : is_periodic f 3)
    (h_odd : is_odd f)
    (h_f1 : f 1 < 1)
    (h_f2 : f 2 = (2*a - 1)/(a + 1)) :
    a < -1 ∨ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_period_3_odd_function_inequality_l3494_349470


namespace NUMINAMATH_CALUDE_power_product_cube_l3494_349454

theorem power_product_cube (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_cube_l3494_349454


namespace NUMINAMATH_CALUDE_claire_apple_pies_l3494_349484

theorem claire_apple_pies :
  ∃! n : ℕ, n < 30 ∧ n % 6 = 4 ∧ n % 8 = 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_claire_apple_pies_l3494_349484


namespace NUMINAMATH_CALUDE_total_paths_A_to_D_l3494_349458

/-- Number of paths between two points -/
def num_paths (start finish : ℕ) : ℕ := sorry

/-- The problem setup -/
axiom paths_A_to_B : num_paths 0 1 = 2
axiom paths_B_to_C : num_paths 1 2 = 2
axiom paths_A_to_C_direct : num_paths 0 2 = 1
axiom paths_C_to_D : num_paths 2 3 = 2

/-- The theorem to prove -/
theorem total_paths_A_to_D : num_paths 0 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_paths_A_to_D_l3494_349458


namespace NUMINAMATH_CALUDE_nickels_left_l3494_349425

def initial_cents : ℕ := 475
def exchanged_cents : ℕ := 75
def cents_per_nickel : ℕ := 5
def cents_per_dime : ℕ := 10

def peter_proportion : ℚ := 2/5
def randi_proportion : ℚ := 3/5
def paula_proportion : ℚ := 1/10

theorem nickels_left : ℕ := by
  -- Prove that Ray is left with 82 nickels
  sorry

end NUMINAMATH_CALUDE_nickels_left_l3494_349425


namespace NUMINAMATH_CALUDE_problem_solution_l3494_349475

theorem problem_solution (x y : ℝ) (h1 : x^(2*y) = 8) (h2 : x = 2) : y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3494_349475


namespace NUMINAMATH_CALUDE_complex_properties_l3494_349450

def Z (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 + 3*m + 2)

theorem complex_properties (m : ℝ) :
  (Z m = Complex.I * Complex.im (Z m) ∧ Complex.im (Z m) ≠ 0 ↔ m = 3) ∧
  (Complex.im (Z m) = 0 ↔ m = -1 ∨ m = -2) ∧
  (Complex.re (Z m) < 0 ∧ Complex.im (Z m) > 0 ↔ -1 < m ∧ m < 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_properties_l3494_349450


namespace NUMINAMATH_CALUDE_multiply_93_107_l3494_349427

theorem multiply_93_107 : 93 * 107 = 9951 := by
  sorry

end NUMINAMATH_CALUDE_multiply_93_107_l3494_349427


namespace NUMINAMATH_CALUDE_traditionalist_ratio_in_specific_country_l3494_349466

/-- Represents a country with provinces, progressives, and traditionalists -/
structure Country where
  num_provinces : ℕ
  num_progressives : ℕ
  num_traditionalists_per_province : ℕ

/-- The fraction of the country that is traditionalist -/
def traditionalist_fraction (c : Country) : ℚ :=
  (c.num_traditionalists_per_province * c.num_provinces : ℚ) / 
  (c.num_progressives + c.num_traditionalists_per_province * c.num_provinces : ℚ)

/-- The ratio of traditionalists in one province to total progressives -/
def traditionalist_to_progressive_ratio (c : Country) : ℚ :=
  (c.num_traditionalists_per_province : ℚ) / c.num_progressives

theorem traditionalist_ratio_in_specific_country :
  ∀ c : Country,
    c.num_provinces = 5 →
    traditionalist_fraction c = 3/4 →
    traditionalist_to_progressive_ratio c = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_traditionalist_ratio_in_specific_country_l3494_349466


namespace NUMINAMATH_CALUDE_multiply_a_equals_four_l3494_349473

theorem multiply_a_equals_four (a b x : ℝ) 
  (h1 : x * a = 5 * b) 
  (h2 : a * b ≠ 0) 
  (h3 : a / 5 = b / 4) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_a_equals_four_l3494_349473


namespace NUMINAMATH_CALUDE_range_of_b_l3494_349459

theorem range_of_b (y b : ℝ) (h1 : b > 1) (h2 : |y - 2| + |y - 5| < b) : b > 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l3494_349459


namespace NUMINAMATH_CALUDE_benzene_homolog_bonds_l3494_349429

/-- Represents the number of bonds in a molecule -/
def num_bonds (n : ℕ) : ℕ := 3 * n - 3

/-- Represents the number of valence electrons in a molecule -/
def num_valence_electrons (n : ℕ) : ℕ := 4 * n + (2 * n - 6)

/-- Theorem stating the relationship between carbon atoms and bonds in benzene homologs -/
theorem benzene_homolog_bonds (n : ℕ) : 
  num_bonds n = (num_valence_electrons n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_benzene_homolog_bonds_l3494_349429


namespace NUMINAMATH_CALUDE_existence_of_m_n_l3494_349453

theorem existence_of_m_n (p : ℕ) (hp : Prime p) (hp_gt_10 : p > 10) :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n < p ∧ (p ∣ (5^m * 7^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l3494_349453


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_l3494_349492

/-- The number of red balls in the bag -/
def num_red : ℕ := 3

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_white

/-- The number of balls drawn -/
def num_drawn : ℕ := 2

/-- The probability of drawing at least one white ball when selecting two balls -/
theorem prob_at_least_one_white :
  (1 : ℚ) - (num_red.choose num_drawn : ℚ) / (total_balls.choose num_drawn : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_l3494_349492


namespace NUMINAMATH_CALUDE_product_equality_l3494_349486

theorem product_equality (square : ℕ) : 
  10 * 20 * 30 * 40 * 50 = 100 * 2 * 300 * 4 * square → square = 50 := by
sorry

end NUMINAMATH_CALUDE_product_equality_l3494_349486


namespace NUMINAMATH_CALUDE_triangle_acute_from_inequalities_l3494_349402

theorem triangle_acute_from_inequalities (α β γ : Real) 
  (sum_angles : α + β + γ = Real.pi)
  (ineq1 : Real.sin α > Real.cos β)
  (ineq2 : Real.sin β > Real.cos γ)
  (ineq3 : Real.sin γ > Real.cos α) :
  α < Real.pi / 2 ∧ β < Real.pi / 2 ∧ γ < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_acute_from_inequalities_l3494_349402


namespace NUMINAMATH_CALUDE_equal_dice_probability_l3494_349401

/-- The number of dice being rolled -/
def num_dice : ℕ := 5

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The probability of a single die showing a number less than or equal to 10 -/
def prob_le_10 : ℚ := 1/2

/-- The probability of a single die showing a number greater than 10 -/
def prob_gt_10 : ℚ := 1/2

/-- The number of ways to choose dice showing numbers less than or equal to 10 -/
def ways_to_choose : ℕ := Nat.choose num_dice (num_dice / 2)

/-- The theorem stating the probability of rolling an equal number of dice showing
    numbers less than or equal to 10 as showing numbers greater than 10 -/
theorem equal_dice_probability :
  (2 * ways_to_choose : ℚ) * (prob_le_10 ^ num_dice) = 5/8 := by sorry

end NUMINAMATH_CALUDE_equal_dice_probability_l3494_349401


namespace NUMINAMATH_CALUDE_fish_catching_average_l3494_349416

theorem fish_catching_average (aang_fish sokka_fish toph_fish : ℕ) 
  (h1 : aang_fish = 7)
  (h2 : sokka_fish = 5)
  (h3 : toph_fish = 12) :
  (aang_fish + sokka_fish + toph_fish) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fish_catching_average_l3494_349416


namespace NUMINAMATH_CALUDE_jane_sequins_count_l3494_349424

/-- The number of rows of blue sequins -/
def blue_rows : Nat := 6

/-- The number of blue sequins in each row -/
def blue_per_row : Nat := 8

/-- The number of rows of purple sequins -/
def purple_rows : Nat := 5

/-- The number of purple sequins in each row -/
def purple_per_row : Nat := 12

/-- The number of rows of green sequins -/
def green_rows : Nat := 9

/-- The number of green sequins in each row -/
def green_per_row : Nat := 6

/-- The total number of sequins Jane adds to her costume -/
def total_sequins : Nat := blue_rows * blue_per_row + purple_rows * purple_per_row + green_rows * green_per_row

theorem jane_sequins_count : total_sequins = 162 := by
  sorry

end NUMINAMATH_CALUDE_jane_sequins_count_l3494_349424


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3494_349412

/-- Given a sphere with surface area 4π, its volume is 4π/3 -/
theorem sphere_volume_from_surface_area :
  ∀ r : ℝ, 4 * Real.pi * r^2 = 4 * Real.pi → (4 / 3) * Real.pi * r^3 = (4 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3494_349412


namespace NUMINAMATH_CALUDE_asphalt_work_hours_l3494_349422

/-- The number of hours per day the first group worked -/
def hours_per_day : ℝ := 8

/-- The number of men in the first group -/
def men_group1 : ℕ := 30

/-- The number of days the first group worked -/
def days_group1 : ℕ := 12

/-- The length of road asphalted by the first group in km -/
def road_length_group1 : ℝ := 1

/-- The number of men in the second group -/
def men_group2 : ℕ := 20

/-- The number of hours per day the second group worked -/
def hours_per_day_group2 : ℕ := 15

/-- The number of days the second group worked -/
def days_group2 : ℝ := 19.2

/-- The length of road asphalted by the second group in km -/
def road_length_group2 : ℝ := 2

theorem asphalt_work_hours :
  hours_per_day * men_group1 * days_group1 * road_length_group2 =
  hours_per_day_group2 * men_group2 * days_group2 * road_length_group1 :=
by sorry

end NUMINAMATH_CALUDE_asphalt_work_hours_l3494_349422


namespace NUMINAMATH_CALUDE_inequality_proof_l3494_349405

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) := x^2 + 2/x + a * Real.log x

theorem inequality_proof (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) (h₄ : a ≤ 0) :
  (f a x₁ + f a x₂) / 2 > f a ((x₁ + x₂) / 2) :=
sorry

end

end NUMINAMATH_CALUDE_inequality_proof_l3494_349405


namespace NUMINAMATH_CALUDE_positive_rationals_characterization_l3494_349445

theorem positive_rationals_characterization (M : Set ℚ) (h_nonempty : Set.Nonempty M) :
  (∀ a b : ℚ, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a * b) ∈ M) →
  (∀ r : ℚ, (r ∈ M ∧ -r ∉ M ∧ r ≠ 0) ∨ (-r ∈ M ∧ r ∉ M ∧ r ≠ 0) ∨ (r ∉ M ∧ -r ∉ M ∧ r = 0)) →
  M = {x : ℚ | x > 0} :=
by sorry

end NUMINAMATH_CALUDE_positive_rationals_characterization_l3494_349445


namespace NUMINAMATH_CALUDE_kendall_correlation_coefficient_l3494_349408

def scores_A : List ℝ := [95, 90, 86, 84, 75, 70, 62, 60, 57, 50]
def scores_B : List ℝ := [92, 93, 83, 80, 55, 60, 45, 72, 62, 70]

def kendall_tau_b (x y : List ℝ) : ℝ := sorry

theorem kendall_correlation_coefficient :
  kendall_tau_b scores_A scores_B = 0.51 := by sorry

end NUMINAMATH_CALUDE_kendall_correlation_coefficient_l3494_349408


namespace NUMINAMATH_CALUDE_inequality_proof_l3494_349452

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 8) :
  (a - 2) / (a + 1) + (b - 2) / (b + 1) + (c - 2) / (c + 1) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3494_349452


namespace NUMINAMATH_CALUDE_special_pentagon_theorem_l3494_349439

/-- A pentagon with two right angles and three known angles -/
structure SpecialPentagon where
  -- The measures of the three known angles
  angle_P : ℝ
  angle_Q : ℝ
  angle_R : ℝ
  -- The measures of the two unknown angles
  angle_U : ℝ
  angle_V : ℝ
  -- Conditions
  angle_P_eq : angle_P = 42
  angle_Q_eq : angle_Q = 60
  angle_R_eq : angle_R = 38
  -- The pentagon has two right angles
  has_two_right_angles : True
  -- The sum of all interior angles of a pentagon is 540°
  sum_of_angles : angle_P + angle_Q + angle_R + angle_U + angle_V + 180 = 540

theorem special_pentagon_theorem (p : SpecialPentagon) : p.angle_U + p.angle_V = 40 := by
  sorry

end NUMINAMATH_CALUDE_special_pentagon_theorem_l3494_349439


namespace NUMINAMATH_CALUDE_book_cost_l3494_349462

/-- Given that three identical books cost $36, prove that seven of these books cost $84. -/
theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 36) : 
  (7 / 3) * cost_of_three = 84 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l3494_349462


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l3494_349438

theorem largest_four_digit_divisible_by_six :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 6 = 0 → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l3494_349438


namespace NUMINAMATH_CALUDE_no_valid_sequence_of_ten_l3494_349451

-- Define the square ABCD
def square_ABCD : Set (ℝ × ℝ) :=
  {(1, 1), (-1, 1), (-1, -1), (1, -1)}

-- Define the rotation transformations
def L : (ℝ × ℝ) → (ℝ × ℝ) :=
  λ (x, y) ↦ (-y, x)

def R : (ℝ × ℝ) → (ℝ × ℝ) :=
  λ (x, y) ↦ (y, -x)

-- Define a sequence of transformations
def Transformation := List (Fin 2)

-- Define the application of a sequence of transformations
def apply_transformations (t : Transformation) (p : ℝ × ℝ) : ℝ × ℝ :=
  t.foldl (λ p i ↦ if i = 0 then L p else R p) p

-- Theorem statement
theorem no_valid_sequence_of_ten :
  ∀ t : Transformation, t.length = 10 →
    (∀ p ∈ square_ABCD, apply_transformations t p = p) → False :=
sorry

end NUMINAMATH_CALUDE_no_valid_sequence_of_ten_l3494_349451


namespace NUMINAMATH_CALUDE_scientific_notation_of_234_1_million_l3494_349435

theorem scientific_notation_of_234_1_million :
  let million : ℝ := 10^6
  234.1 * million = 2.341 * 10^6 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_234_1_million_l3494_349435


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l3494_349440

/-- Calculates the cost price per metre of cloth given the total metres sold,
    total selling price, and loss per metre. -/
theorem cost_price_per_metre
  (total_metres : ℕ)
  (total_selling_price : ℕ)
  (loss_per_metre : ℕ)
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 36000)
  (h3 : loss_per_metre = 10) :
  (total_selling_price + total_metres * loss_per_metre) / total_metres = 70 := by
  sorry

#check cost_price_per_metre

end NUMINAMATH_CALUDE_cost_price_per_metre_l3494_349440


namespace NUMINAMATH_CALUDE_relationship_abcd_l3494_349432

theorem relationship_abcd (a b c d : ℝ) 
  (h : (a + 2*b) / (b + 2*c) = (c + 2*d) / (d + 2*a)) :
  b = 2*a ∨ a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_relationship_abcd_l3494_349432


namespace NUMINAMATH_CALUDE_investment_profit_ratio_l3494_349409

/-- Represents a partner's investment details -/
structure Partner where
  investment : ℕ
  time : ℕ

/-- Calculates the profit ratio of two partners given their investment details -/
def profitRatio (p q : Partner) : Rat :=
  (p.investment * p.time : ℚ) / (q.investment * q.time)

theorem investment_profit_ratio :
  let p : Partner := ⟨7, 5⟩
  let q : Partner := ⟨5, 13⟩
  profitRatio p q = 7 / 13 := by
  sorry

end NUMINAMATH_CALUDE_investment_profit_ratio_l3494_349409


namespace NUMINAMATH_CALUDE_number_times_24_equals_173_times_240_l3494_349414

theorem number_times_24_equals_173_times_240 : ∃ x : ℕ, x * 24 = 173 * 240 ∧ x = 1730 := by
  sorry

end NUMINAMATH_CALUDE_number_times_24_equals_173_times_240_l3494_349414


namespace NUMINAMATH_CALUDE_max_volume_box_l3494_349469

/-- The volume of a lidless box formed from a rectangular sheet with corners cut out -/
def boxVolume (sheetLength sheetWidth squareSide : ℝ) : ℝ :=
  (sheetLength - 2 * squareSide) * (sheetWidth - 2 * squareSide) * squareSide

/-- The theorem stating the maximum volume of the box and the optimal side length of cut squares -/
theorem max_volume_box (sheetLength sheetWidth : ℝ) 
  (hL : sheetLength = 8) (hW : sheetWidth = 5) :
  ∃ (optimalSide maxVolume : ℝ),
    optimalSide = 1 ∧ 
    maxVolume = 18 ∧
    ∀ x, 0 < x → x < sheetWidth / 2 → 
      boxVolume sheetLength sheetWidth x ≤ maxVolume := by
  sorry

end NUMINAMATH_CALUDE_max_volume_box_l3494_349469


namespace NUMINAMATH_CALUDE_bathroom_extension_l3494_349499

/-- Represents the dimensions and area of a rectangular bathroom -/
structure Bathroom where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Calculates the new area of a bathroom after extension -/
def extended_area (b : Bathroom) (extension : ℝ) : ℝ :=
  (b.width + 2 * extension) * (b.length + 2 * extension)

/-- Theorem: Given a bathroom with area 96 sq ft and width 8 ft, 
    extending it by 2 ft on each side results in an area of 140 sq ft -/
theorem bathroom_extension :
  ∀ (b : Bathroom),
    b.area = 96 ∧ b.width = 8 →
    extended_area b 2 = 140 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_extension_l3494_349499


namespace NUMINAMATH_CALUDE_bacon_suggestion_count_l3494_349431

theorem bacon_suggestion_count (mashed_and_bacon : ℕ) (only_bacon : ℕ) : 
  mashed_and_bacon = 218 → only_bacon = 351 → 
  mashed_and_bacon + only_bacon = 569 := by sorry

end NUMINAMATH_CALUDE_bacon_suggestion_count_l3494_349431


namespace NUMINAMATH_CALUDE_bowl_glass_pairings_l3494_349428

/-- The number of possible pairings when choosing one bowl from a set of distinct bowls
    and one glass from a set of distinct glasses -/
def num_pairings (num_bowls : ℕ) (num_glasses : ℕ) : ℕ :=
  num_bowls * num_glasses

/-- Theorem stating that with 5 distinct bowls and 6 distinct glasses,
    the number of possible pairings is 30 -/
theorem bowl_glass_pairings :
  num_pairings 5 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_bowl_glass_pairings_l3494_349428


namespace NUMINAMATH_CALUDE_distance_on_foot_for_given_journey_l3494_349444

/-- A journey with two modes of transportation -/
structure Journey where
  total_distance : ℝ
  total_time : ℝ
  foot_speed : ℝ
  bicycle_speed : ℝ

/-- The distance traveled on foot for a given journey -/
def distance_on_foot (j : Journey) : ℝ :=
  -- Define this as a real number, but don't provide the actual calculation
  sorry

/-- Theorem stating the distance traveled on foot for the specific journey -/
theorem distance_on_foot_for_given_journey :
  let j : Journey := {
    total_distance := 80,
    total_time := 7,
    foot_speed := 8,
    bicycle_speed := 16
  }
  distance_on_foot j = 32 := by
  sorry

end NUMINAMATH_CALUDE_distance_on_foot_for_given_journey_l3494_349444


namespace NUMINAMATH_CALUDE_first_term_exceeding_2020_l3494_349488

theorem first_term_exceeding_2020 : 
  (∃ n : ℕ, 2 * n^2 ≥ 2020 ∧ ∀ m : ℕ, m < n → 2 * m^2 < 2020) → 
  (∃ n : ℕ, 2 * n^2 ≥ 2020 ∧ ∀ m : ℕ, m < n → 2 * m^2 < 2020) ∧ 
  (∀ n : ℕ, (2 * n^2 ≥ 2020 ∧ ∀ m : ℕ, m < n → 2 * m^2 < 2020) → n = 32) :=
by sorry

end NUMINAMATH_CALUDE_first_term_exceeding_2020_l3494_349488


namespace NUMINAMATH_CALUDE_inequality_solution_l3494_349442

theorem inequality_solution (x : ℝ) : 
  (-1 < (x^2 - 10*x + 9) / (x^2 - 4*x + 8) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 8) < 1) ↔ x > 1/6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3494_349442


namespace NUMINAMATH_CALUDE_john_spends_more_l3494_349447

/-- Calculates the difference in annual cost between John's new and former living arrangements -/
def annual_cost_difference (former_rent_per_sqft : ℚ) (former_size : ℚ) 
  (new_rent_first_half : ℚ) (new_rent_increase_percent : ℚ) 
  (winter_utilities : ℚ) (other_utilities : ℚ) : ℚ :=
  let former_annual_cost := former_rent_per_sqft * former_size * 12
  let new_rent_second_half := new_rent_first_half * (1 + new_rent_increase_percent)
  let new_annual_rent := new_rent_first_half * 6 + new_rent_second_half * 6
  let new_annual_utilities := winter_utilities * 3 + other_utilities * 9
  let new_total_cost := new_annual_rent + new_annual_utilities
  let john_new_cost := new_total_cost / 2
  john_new_cost - former_annual_cost

/-- Theorem stating that John spends $195 more annually in the new arrangement -/
theorem john_spends_more : 
  annual_cost_difference 2 750 2800 (5/100) 200 150 = 195 := by
  sorry

end NUMINAMATH_CALUDE_john_spends_more_l3494_349447


namespace NUMINAMATH_CALUDE_quadratic_residue_minus_one_l3494_349491

theorem quadratic_residue_minus_one (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  (∃ x : Nat, x^2 ≡ -1 [ZMOD p]) ↔ p ≡ 1 [ZMOD 4] := by
  sorry

end NUMINAMATH_CALUDE_quadratic_residue_minus_one_l3494_349491


namespace NUMINAMATH_CALUDE_zoe_money_made_l3494_349426

/-- Calculates the money made from selling chocolate bars -/
def money_made (cost_per_bar : ℕ) (total_bars : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * cost_per_bar

/-- Theorem: Zoe made $42 from selling chocolate bars -/
theorem zoe_money_made :
  let cost_per_bar : ℕ := 6
  let total_bars : ℕ := 13
  let unsold_bars : ℕ := 6
  money_made cost_per_bar total_bars unsold_bars = 42 := by
sorry

end NUMINAMATH_CALUDE_zoe_money_made_l3494_349426


namespace NUMINAMATH_CALUDE_mitch_savings_amount_l3494_349404

/-- Represents the amount of money Mitch has saved for his boating hobby. -/
def mitchSavings : ℕ := sorry

/-- The cost of a boat per foot of length. -/
def boatCostPerFoot : ℕ := 1500

/-- The cost of license and registration. -/
def licenseAndRegistrationCost : ℕ := 500

/-- The maximum length of boat Mitch can buy. -/
def maxBoatLength : ℕ := 12

/-- The docking fees are three times the license and registration cost. -/
def dockingFees : ℕ := 3 * licenseAndRegistrationCost

/-- The total cost of additional fees (license, registration, and docking). -/
def additionalFees : ℕ := licenseAndRegistrationCost + dockingFees

/-- The cost of the longest boat Mitch can buy. -/
def maxBoatCost : ℕ := boatCostPerFoot * maxBoatLength

/-- Theorem stating that Mitch has saved $20,000 for his boating hobby. -/
theorem mitch_savings_amount : mitchSavings = 20000 := by sorry

end NUMINAMATH_CALUDE_mitch_savings_amount_l3494_349404


namespace NUMINAMATH_CALUDE_distinct_naturals_reciprocal_sum_l3494_349487

theorem distinct_naturals_reciprocal_sum (x y z : ℕ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →  -- distinct
  0 < x ∧ 0 < y ∧ 0 < z →  -- natural numbers
  x < y ∧ y < z →  -- ascending order
  (∃ (n : ℕ), (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = n) →  -- sum is a natural number
  x = 2 ∧ y = 3 ∧ z = 6 :=
by sorry

end NUMINAMATH_CALUDE_distinct_naturals_reciprocal_sum_l3494_349487


namespace NUMINAMATH_CALUDE_election_votes_proof_l3494_349483

theorem election_votes_proof (total_votes : ℕ) : 
  (∃ (valid_votes_A valid_votes_B : ℕ),
    -- 20% of votes are invalid
    (total_votes : ℚ) * (4/5) = valid_votes_A + valid_votes_B ∧
    -- A's valid votes exceed B's by 15% of total votes
    valid_votes_A = valid_votes_B + (total_votes : ℚ) * (3/20) ∧
    -- B received 2834 valid votes
    valid_votes_B = 2834) →
  total_votes = 8720 := by
sorry

end NUMINAMATH_CALUDE_election_votes_proof_l3494_349483


namespace NUMINAMATH_CALUDE_no_intersection_l3494_349449

/-- The line equation 3x + 4y = 12 -/
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The circle equation x^2 + y^2 = 4 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The number of intersection points between the line and the circle -/
def intersection_count : ℕ := 0

theorem no_intersection :
  ∀ x y : ℝ, ¬(line_eq x y ∧ circle_eq x y) :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_l3494_349449


namespace NUMINAMATH_CALUDE_bottom_row_bricks_count_l3494_349474

/-- Represents a brick wall with a triangular pattern -/
structure BrickWall where
  rows : ℕ
  total_bricks : ℕ
  bottom_row_bricks : ℕ
  h_rows : rows > 0
  h_pattern : total_bricks = (2 * bottom_row_bricks - rows + 1) * rows / 2

/-- The specific brick wall in the problem -/
def problem_wall : BrickWall where
  rows := 5
  total_bricks := 200
  bottom_row_bricks := 42
  h_rows := by norm_num
  h_pattern := by norm_num

theorem bottom_row_bricks_count (wall : BrickWall) 
  (h_rows : wall.rows = 5) 
  (h_total : wall.total_bricks = 200) : 
  wall.bottom_row_bricks = 42 := by
  sorry

#check bottom_row_bricks_count

end NUMINAMATH_CALUDE_bottom_row_bricks_count_l3494_349474


namespace NUMINAMATH_CALUDE_max_value_abc_l3494_349489

theorem max_value_abc (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 5)
  (hb : 0 ≤ b ∧ b ≤ 5)
  (hc : 0 ≤ c ∧ c ≤ 5)
  (h_sum : 2 * a + b + c = 10) :
  a + 2 * b + 3 * c ≤ 25 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l3494_349489


namespace NUMINAMATH_CALUDE_complex_number_properties_l3494_349463

theorem complex_number_properties (z₁ z₂ : ℂ) (h : Complex.abs z₁ * Complex.abs z₂ ≠ 0) :
  (Complex.abs (z₁ + z₂) ≤ Complex.abs z₁ + Complex.abs z₂) ∧
  (Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3494_349463


namespace NUMINAMATH_CALUDE_divisors_of_900_l3494_349468

theorem divisors_of_900 : Finset.card (Nat.divisors 900) = 27 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_900_l3494_349468


namespace NUMINAMATH_CALUDE_meal_cost_l3494_349400

-- Define variables for the cost of each item
variable (s : ℝ) -- cost of one sandwich
variable (c : ℝ) -- cost of one cup of coffee
variable (p : ℝ) -- cost of one piece of pie

-- Define the given equations
def equation1 : Prop := 5 * s + 8 * c + p = 5
def equation2 : Prop := 7 * s + 12 * c + p = 7.2
def equation3 : Prop := 4 * s + 6 * c + 2 * p = 6

-- Theorem to prove
theorem meal_cost (h1 : equation1 s c p) (h2 : equation2 s c p) (h3 : equation3 s c p) :
  s + c + p = 1.9 := by sorry

end NUMINAMATH_CALUDE_meal_cost_l3494_349400


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3494_349420

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1 = -1 + Real.sqrt 5 ∧ x2 = -1 - Real.sqrt 5 ∧
    x1^2 + 2*x1 - 4 = 0 ∧ x2^2 + 2*x2 - 4 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = -2 ∧
    2*x1 - 6 = x1*(3-x1) ∧ 2*x2 - 6 = x2*(3-x2)) :=
by
  sorry

#check quadratic_equations_solutions

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3494_349420


namespace NUMINAMATH_CALUDE_exponent_calculations_l3494_349497

theorem exponent_calculations (a : ℝ) (h : a ≠ 0) : 
  (a^3 + a^3 ≠ a^6) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (a^2 * a^4 ≠ a^8) ∧ 
  (a^4 / a^3 = a) := by sorry

end NUMINAMATH_CALUDE_exponent_calculations_l3494_349497


namespace NUMINAMATH_CALUDE_problem_statement_l3494_349448

theorem problem_statement (a b c : ℝ) (h : a + 10 = b + 12 ∧ b + 12 = c + 15) :
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 38 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3494_349448


namespace NUMINAMATH_CALUDE_max_value_interval_max_value_at_one_l3494_349495

/-- The function f(x) = x^2 - 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

/-- f(x) is monotonically decreasing in (-∞, 2] -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ 2 → f a x ≥ f a y

theorem max_value_interval (a : ℝ) (h : is_monotone_decreasing a) :
  (∀ x ∈ Set.Icc 3 5, f a x ≤ 8) ∧ (∃ x ∈ Set.Icc 3 5, f a x = 8) :=
sorry

theorem max_value_at_one (a : ℝ) (h : is_monotone_decreasing a) :
  f a 1 ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_interval_max_value_at_one_l3494_349495


namespace NUMINAMATH_CALUDE_workshop_workers_l3494_349482

theorem workshop_workers (total_average : ℝ) (technician_count : ℕ) (technician_average : ℝ) (non_technician_average : ℝ) 
  (h1 : total_average = 8000)
  (h2 : technician_count = 7)
  (h3 : technician_average = 12000)
  (h4 : non_technician_average = 6000) :
  ∃ (total_workers : ℕ), 
    total_workers * total_average = 
      technician_count * technician_average + (total_workers - technician_count) * non_technician_average ∧
    total_workers = 21 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l3494_349482


namespace NUMINAMATH_CALUDE_specific_cistern_wet_surface_area_l3494_349443

/-- Calculates the total wet surface area of a rectangular cistern. -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the total wet surface area of a specific cistern. -/
theorem specific_cistern_wet_surface_area :
  cistern_wet_surface_area 10 6 1.35 = 103.2 := by
  sorry

end NUMINAMATH_CALUDE_specific_cistern_wet_surface_area_l3494_349443


namespace NUMINAMATH_CALUDE_gravitational_force_at_satellite_orbit_l3494_349457

/-- Gravitational force calculation -/
theorem gravitational_force_at_satellite_orbit 
  (surface_distance : ℝ) 
  (surface_force : ℝ) 
  (satellite_distance : ℝ) 
  (h1 : surface_distance = 6400)
  (h2 : surface_force = 800)
  (h3 : satellite_distance = 384000)
  (h4 : ∀ (d f : ℝ), f * d^2 = surface_force * surface_distance^2) :
  ∃ (satellite_force : ℝ), 
    satellite_force * satellite_distance^2 = surface_force * surface_distance^2 ∧ 
    satellite_force = 2/9 := by
  sorry


end NUMINAMATH_CALUDE_gravitational_force_at_satellite_orbit_l3494_349457


namespace NUMINAMATH_CALUDE_probability_between_lines_l3494_349478

/-- Line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The first quadrant -/
def firstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- Region below a line in the first quadrant -/
def regionBelowLine (l : Line) : Set (ℝ × ℝ) :=
  {p ∈ firstQuadrant | p.2 ≤ l.slope * p.1 + l.yIntercept}

/-- Region between two lines in the first quadrant -/
def regionBetweenLines (l1 l2 : Line) : Set (ℝ × ℝ) :=
  {p ∈ firstQuadrant | l2.slope * p.1 + l2.yIntercept ≤ p.2 ∧ p.2 ≤ l1.slope * p.1 + l1.yIntercept}

/-- Area of a region in the first quadrant -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Probability of selecting a point in a subregion of a given region -/
noncomputable def probability (subregion region : Set (ℝ × ℝ)) : ℝ :=
  area subregion / area region

theorem probability_between_lines :
  let k : Line := ⟨-3, 9⟩
  let n : Line := ⟨-6, 9⟩
  probability (regionBetweenLines k n) (regionBelowLine k) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_between_lines_l3494_349478
