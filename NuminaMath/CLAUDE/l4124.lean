import Mathlib

namespace NUMINAMATH_CALUDE_rhombus60_min_rotation_l4124_412415

/-- A rhombus with a 60° angle -/
structure Rhombus60 where
  /-- The rhombus has a 60° angle -/
  angle : ℝ
  angle_eq : angle = 60

/-- The minimum rotation angle for a Rhombus60 to coincide with its original position -/
def min_rotation_angle (r : Rhombus60) : ℝ := 180

/-- Theorem stating that the minimum rotation angle for a Rhombus60 is 180° -/
theorem rhombus60_min_rotation (r : Rhombus60) :
  min_rotation_angle r = 180 := by sorry

end NUMINAMATH_CALUDE_rhombus60_min_rotation_l4124_412415


namespace NUMINAMATH_CALUDE_yoongis_subtraction_mistake_l4124_412442

theorem yoongis_subtraction_mistake (A B : ℕ) : 
  A ≥ 1 ∧ A ≤ 9 ∧ B = 9 ∧ 
  (10 * A + 6) - 57 = 39 →
  10 * A + B = 99 := by
sorry

end NUMINAMATH_CALUDE_yoongis_subtraction_mistake_l4124_412442


namespace NUMINAMATH_CALUDE_jessica_bank_balance_l4124_412448

/-- Calculates the final balance in Jessica's bank account after withdrawing $400 and depositing 1/4 of the remaining balance. -/
theorem jessica_bank_balance (B : ℝ) (h : 2 / 5 * B = 400) : 
  (B - 400) + (1 / 4 * (B - 400)) = 750 := by
  sorry

#check jessica_bank_balance

end NUMINAMATH_CALUDE_jessica_bank_balance_l4124_412448


namespace NUMINAMATH_CALUDE_geese_survival_theorem_l4124_412426

/-- Represents the number of geese that survived the first year given the total number of eggs laid -/
def geese_survived_first_year (total_eggs : ℕ) : ℕ :=
  let hatched_eggs := (2 * total_eggs) / 3
  let survived_first_month := (3 * hatched_eggs) / 4
  let not_survived_first_year := (3 * survived_first_month) / 5
  survived_first_month - not_survived_first_year

/-- Theorem stating that the number of geese surviving the first year is 1/5 of the total eggs laid -/
theorem geese_survival_theorem (total_eggs : ℕ) :
  geese_survived_first_year total_eggs = total_eggs / 5 := by
  sorry

#eval geese_survived_first_year 60  -- Should output 12

end NUMINAMATH_CALUDE_geese_survival_theorem_l4124_412426


namespace NUMINAMATH_CALUDE_lisa_minimum_score_l4124_412446

def minimum_score_for_geometry (term1 term2 term3 term4 : ℝ) (required_average : ℝ) : ℝ :=
  5 * required_average - (term1 + term2 + term3 + term4)

theorem lisa_minimum_score :
  let term1 := 84
  let term2 := 80
  let term3 := 82
  let term4 := 87
  let required_average := 85
  minimum_score_for_geometry term1 term2 term3 term4 required_average = 92 := by
sorry

end NUMINAMATH_CALUDE_lisa_minimum_score_l4124_412446


namespace NUMINAMATH_CALUDE_lcm_of_8_24_36_54_l4124_412428

theorem lcm_of_8_24_36_54 : Nat.lcm 8 (Nat.lcm 24 (Nat.lcm 36 54)) = 216 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_8_24_36_54_l4124_412428


namespace NUMINAMATH_CALUDE_jerky_order_fulfillment_l4124_412459

/-- The number of days needed to fulfill a jerky order -/
def days_to_fulfill_order (bags_per_batch : ℕ) (order_size : ℕ) (bags_in_stock : ℕ) : ℕ :=
  let bags_to_make := order_size - bags_in_stock
  (bags_to_make + bags_per_batch - 1) / bags_per_batch

theorem jerky_order_fulfillment :
  days_to_fulfill_order 10 60 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jerky_order_fulfillment_l4124_412459


namespace NUMINAMATH_CALUDE_problem_statement_l4124_412447

theorem problem_statement (a b q r : ℕ) (ha : a > 0) (hb : b > 0) 
  (h_division : a^2 + b^2 = q * (a + b) + r) (h_constraint : q^2 + r = 2010) :
  a * b = 1643 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4124_412447


namespace NUMINAMATH_CALUDE_line_length_after_erasing_l4124_412488

/-- Calculates the remaining length of a line after erasing a portion. -/
def remaining_length (initial_length : ℝ) (erased_length : ℝ) : ℝ :=
  initial_length - erased_length

/-- Proves that erasing 24 cm from a 1 m line results in a 76 cm line. -/
theorem line_length_after_erasing :
  remaining_length 100 24 = 76 := by
  sorry

#check line_length_after_erasing

end NUMINAMATH_CALUDE_line_length_after_erasing_l4124_412488


namespace NUMINAMATH_CALUDE_unknown_number_problem_l4124_412491

theorem unknown_number_problem (x : ℚ) : (2 / 3) * x + 6 = 10 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l4124_412491


namespace NUMINAMATH_CALUDE_weight_of_b_l4124_412404

/-- Given three weights a, b, and c, prove that b = 70 under the given conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 60 →
  (a + b) / 2 = 70 →
  (b + c) / 2 = 50 →
  b = 70 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l4124_412404


namespace NUMINAMATH_CALUDE_buffy_whiskers_l4124_412431

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  juniper : ℕ

/-- Theorem stating the number of whiskers Buffy has -/
theorem buffy_whiskers (c : CatWhiskers) : 
  c.juniper = 12 →
  c.puffy = 3 * c.juniper →
  c.scruffy = 2 * c.puffy →
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3 →
  c.buffy = 40 := by
  sorry

#check buffy_whiskers

end NUMINAMATH_CALUDE_buffy_whiskers_l4124_412431


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l4124_412409

theorem compound_interest_calculation (principal : ℝ) (rate : ℝ) (time : ℕ) (final_amount : ℝ) : 
  principal = 8000 →
  rate = 0.05 →
  time = 2 →
  final_amount = 8820 →
  final_amount = principal * (1 + rate) ^ time :=
sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l4124_412409


namespace NUMINAMATH_CALUDE_complement_of_M_l4124_412473

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x ∈ U | (x - 1) * (x - 4) = 0}

theorem complement_of_M (x : ℕ) : x ∈ (U \ M) ↔ x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l4124_412473


namespace NUMINAMATH_CALUDE_digit_doubling_theorem_l4124_412410

def sumOfDigits (n : ℕ) : ℕ := sorry

def doubleDigitSum (n : ℕ) : ℕ := 2 * (sumOfDigits n)

def eventuallyOneDigit (n : ℕ) : Prop :=
  ∃ k, ∃ m : ℕ, (m < 10) ∧ (Nat.iterate doubleDigitSum k n = m)

theorem digit_doubling_theorem :
  (∀ n : ℕ, n ≠ 18 → doubleDigitSum n ≠ n) ∧
  (doubleDigitSum 18 = 18) ∧
  (∀ n : ℕ, n ≠ 18 → eventuallyOneDigit n) := by sorry

end NUMINAMATH_CALUDE_digit_doubling_theorem_l4124_412410


namespace NUMINAMATH_CALUDE_soda_survey_result_l4124_412460

/-- The number of people who chose "Soda" in a survey of 520 people,
    where the central angle of the "Soda" sector is 270° (to the nearest whole degree). -/
def soda_count : ℕ := 390

/-- The total number of people surveyed. -/
def total_surveyed : ℕ := 520

/-- The central angle of the "Soda" sector in degrees. -/
def soda_angle : ℕ := 270

theorem soda_survey_result :
  (soda_count : ℚ) / total_surveyed * 360 ≥ soda_angle - (1/2 : ℚ) ∧
  (soda_count : ℚ) / total_surveyed * 360 < soda_angle + (1/2 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_soda_survey_result_l4124_412460


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l4124_412465

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k*(x + 1) + 1

-- Theorem statement
theorem circle_and_line_properties :
  -- 1. The center of circle C is (1, 0)
  (∃ r : ℝ, ∀ x y : ℝ, circle_C x y ↔ (x - 1)^2 + y^2 = r^2) ∧
  -- 2. The point (-1, 1) lies on line l for any real k
  (∀ k : ℝ, line_l k (-1) 1) ∧
  -- 3. Line l intersects circle C for any real k
  (∀ k : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_l k x y) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l4124_412465


namespace NUMINAMATH_CALUDE_quadratic_roots_equivalence_l4124_412477

theorem quadratic_roots_equivalence (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 - b * x + c = 0 → x > 0) → a * c > 0 ↔
  a * c ≤ 0 → ¬(∀ x, a * x^2 - b * x + c = 0 → x > 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_equivalence_l4124_412477


namespace NUMINAMATH_CALUDE_prob_red_third_eq_147_1000_l4124_412434

/-- A fair 10-sided die with exactly 3 red sides -/
structure RedDie :=
  (sides : Nat)
  (red_sides : Nat)
  (h_sides : sides = 10)
  (h_red : red_sides = 3)

/-- The probability of rolling a non-red side -/
def prob_non_red (d : RedDie) : ℚ :=
  (d.sides - d.red_sides : ℚ) / d.sides

/-- The probability of rolling a red side -/
def prob_red (d : RedDie) : ℚ :=
  d.red_sides / d.sides

/-- The probability of rolling a red side for the first time on the third roll -/
def prob_red_third (d : RedDie) : ℚ :=
  (prob_non_red d) * (prob_non_red d) * (prob_red d)

theorem prob_red_third_eq_147_1000 (d : RedDie) : 
  prob_red_third d = 147 / 1000 := by sorry

end NUMINAMATH_CALUDE_prob_red_third_eq_147_1000_l4124_412434


namespace NUMINAMATH_CALUDE_parallel_implies_parallel_to_intersection_l4124_412425

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Checks if a line lies on a plane -/
def lies_on (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Returns the intersection line of two planes -/
def intersection (p1 p2 : Plane3D) : Line3D :=
  sorry

theorem parallel_implies_parallel_to_intersection
  (a b c : Line3D) (M N : Plane3D)
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h2 : lies_on a M)
  (h3 : lies_on b N)
  (h4 : c = intersection M N)
  (h5 : parallel a b) :
  parallel a c :=
sorry

end NUMINAMATH_CALUDE_parallel_implies_parallel_to_intersection_l4124_412425


namespace NUMINAMATH_CALUDE_lexie_paintings_count_l4124_412437

/-- The number of rooms where paintings are placed -/
def num_rooms : ℕ := 4

/-- The number of paintings placed in each room -/
def paintings_per_room : ℕ := 8

/-- The total number of Lexie's watercolor paintings -/
def total_paintings : ℕ := num_rooms * paintings_per_room

theorem lexie_paintings_count : total_paintings = 32 := by
  sorry

end NUMINAMATH_CALUDE_lexie_paintings_count_l4124_412437


namespace NUMINAMATH_CALUDE_monic_quartic_problem_l4124_412498

-- Define a monic quartic polynomial
def monicQuartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_problem (p : ℝ → ℝ) 
  (h_monic : monicQuartic p)
  (h1 : p 1 = 3)
  (h2 : p 2 = 7)
  (h3 : p 3 = 13)
  (h4 : p 4 = 21) :
  p 5 = 51 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_problem_l4124_412498


namespace NUMINAMATH_CALUDE_same_terminal_side_l4124_412470

theorem same_terminal_side (θ : ℝ) : ∃ k : ℤ, θ = (23 * π / 3 : ℝ) + 2 * π * k ↔ θ = (5 * π / 3 : ℝ) + 2 * π * k := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l4124_412470


namespace NUMINAMATH_CALUDE_apple_cost_price_l4124_412416

/-- Proves that the cost price of an apple is 24 rupees, given the selling price and loss ratio. -/
theorem apple_cost_price (selling_price : ℚ) (loss_ratio : ℚ) : 
  selling_price = 20 → loss_ratio = 1/6 → 
  ∃ cost_price : ℚ, cost_price = 24 ∧ selling_price = cost_price - loss_ratio * cost_price :=
by sorry

end NUMINAMATH_CALUDE_apple_cost_price_l4124_412416


namespace NUMINAMATH_CALUDE_sales_price_calculation_l4124_412476

theorem sales_price_calculation (C G : ℝ) (h1 : G = 1.6 * C) (h2 : G = 56) :
  C + G = 91 := by
  sorry

end NUMINAMATH_CALUDE_sales_price_calculation_l4124_412476


namespace NUMINAMATH_CALUDE_prob_first_red_given_second_black_l4124_412419

/-- Represents the contents of an urn -/
structure Urn :=
  (white : ℕ)
  (red : ℕ)
  (black : ℕ)

/-- The probability of drawing a specific color from an urn -/
def prob_draw (u : Urn) (color : String) : ℚ :=
  match color with
  | "white" => u.white / (u.white + u.red + u.black)
  | "red" => u.red / (u.white + u.red + u.black)
  | "black" => u.black / (u.white + u.red + u.black)
  | _ => 0

/-- The contents of Urn A -/
def urn_A : Urn := ⟨4, 2, 0⟩

/-- The contents of Urn B -/
def urn_B : Urn := ⟨0, 3, 3⟩

/-- The probability of selecting an urn -/
def prob_select_urn : ℚ := 1/2

theorem prob_first_red_given_second_black :
  let p_red_and_black := 
    (prob_select_urn * prob_draw urn_A "red" * prob_select_urn * prob_draw urn_B "black") +
    (prob_select_urn * prob_draw urn_B "red" * prob_select_urn * (prob_draw urn_B "black" * (urn_B.black - 1) / (urn_B.red + urn_B.black - 1)))
  let p_second_black :=
    (prob_select_urn * prob_select_urn * prob_draw urn_B "black") +
    (prob_select_urn * prob_draw urn_B "red" * prob_select_urn * (prob_draw urn_B "black" * (urn_B.black) / (urn_B.red + urn_B.black - 1))) +
    (prob_select_urn * prob_draw urn_B "black" * prob_select_urn * (prob_draw urn_B "black" * (urn_B.black - 1) / (urn_B.red + urn_B.black - 1)))
  p_red_and_black / p_second_black = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_first_red_given_second_black_l4124_412419


namespace NUMINAMATH_CALUDE_circle_center_correct_l4124_412468

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 5 = 0

/-- The center of the circle -/
def CircleCenter : ℝ × ℝ := (2, 1)

/-- Theorem: The center of the circle defined by CircleEquation is CircleCenter -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l4124_412468


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4124_412481

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = -1 + Real.sqrt 6 ∧ x₂ = -1 - Real.sqrt 6) ∧ 
  (x₁^2 + 2*x₁ - 5 = 0 ∧ x₂^2 + 2*x₂ - 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4124_412481


namespace NUMINAMATH_CALUDE_jelly_bean_distribution_l4124_412429

theorem jelly_bean_distribution (n : ℕ) (h1 : 10 ≤ n) (h2 : n ≤ 20) : 
  (∃ (total : ℕ), total = n^2 ∧ total % 5 = 0) → 
  (∃ (per_bag : ℕ), per_bag = 45 ∧ 5 * per_bag = n^2) := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_distribution_l4124_412429


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l4124_412493

/-- The diamond operation defined as a ◇ b = a * sqrt(b + sqrt(b + sqrt(b + ...))) -/
noncomputable def diamond (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

/-- Theorem stating that if 2 ◇ h = 8, then h = 12 -/
theorem diamond_equation_solution (h : ℝ) (eq : diamond 2 h = 8) : h = 12 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l4124_412493


namespace NUMINAMATH_CALUDE_dans_initial_money_l4124_412499

/-- Given that Dan bought a candy bar for $7 and a chocolate for $6,
    and spent $13 in total, prove that his initial amount was $13. -/
theorem dans_initial_money :
  ∀ (candy_price chocolate_price total_spent initial_amount : ℕ),
    candy_price = 7 →
    chocolate_price = 6 →
    total_spent = 13 →
    total_spent = candy_price + chocolate_price →
    initial_amount = total_spent →
    initial_amount = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_dans_initial_money_l4124_412499


namespace NUMINAMATH_CALUDE_blue_then_red_probability_l4124_412486

/-- The probability of drawing a blue ball first and a red ball second from a box 
    containing 15 balls (5 blue and 10 red) without replacement is 5/21. -/
theorem blue_then_red_probability (total : ℕ) (blue : ℕ) (red : ℕ) :
  total = 15 → blue = 5 → red = 10 →
  (blue : ℚ) / total * red / (total - 1) = 5 / 21 := by
  sorry

end NUMINAMATH_CALUDE_blue_then_red_probability_l4124_412486


namespace NUMINAMATH_CALUDE_sum_of_selected_numbers_l4124_412443

def set1 := Finset.Icc 10 19
def set2 := Finset.Icc 90 99

def is_valid_selection (s1 s2 : Finset ℕ) : Prop :=
  s1.card = 5 ∧ s2.card = 5 ∧ 
  s1 ⊆ set1 ∧ s2 ⊆ set2 ∧
  ∀ x ∈ s1, ∀ y ∈ s1, x ≠ y → (x - y) % 10 ≠ 0 ∧
  ∀ x ∈ s2, ∀ y ∈ s2, x ≠ y → (x - y) % 10 ≠ 0 ∧
  ∀ x ∈ s1, ∀ y ∈ s2, (x - y) % 10 ≠ 0

theorem sum_of_selected_numbers (s1 s2 : Finset ℕ) 
  (h : is_valid_selection s1 s2) : 
  (s1.sum id + s2.sum id) = 545 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_selected_numbers_l4124_412443


namespace NUMINAMATH_CALUDE_max_sum_is_3972_l4124_412469

/-- A function that generates all possible permutations of 9 digits -/
def generatePermutations : List (List Nat) := sorry

/-- A function that splits a list of 9 digits into three numbers -/
def splitIntoThreeNumbers (perm : List Nat) : (Nat × Nat × Nat) := sorry

/-- A function that calculates the sum of three numbers -/
def sumThreeNumbers (nums : Nat × Nat × Nat) : Nat := sorry

/-- The maximum sum achievable using digits 1 to 9 -/
def maxSum : Nat := 3972

theorem max_sum_is_3972 :
  ∀ perm ∈ generatePermutations,
    let (n1, n2, n3) := splitIntoThreeNumbers perm
    sumThreeNumbers (n1, n2, n3) ≤ maxSum :=
by sorry

end NUMINAMATH_CALUDE_max_sum_is_3972_l4124_412469


namespace NUMINAMATH_CALUDE_isabel_candy_theorem_l4124_412453

/-- The number of candy pieces Isabel has left after distribution -/
def remaining_candy (initial : ℕ) (friend : ℕ) (cousin : ℕ) (sister : ℕ) (distributed : ℕ) : ℤ :=
  (initial + friend + cousin + sister : ℤ) - distributed

/-- Theorem stating the number of candy pieces Isabel has left -/
theorem isabel_candy_theorem (x y z : ℕ) :
  remaining_candy 325 145 x y z = 470 + x + y - z := by
  sorry

end NUMINAMATH_CALUDE_isabel_candy_theorem_l4124_412453


namespace NUMINAMATH_CALUDE_marble_selection_problem_l4124_412444

theorem marble_selection_problem (n : ℕ) (k : ℕ) (total : ℕ) (red : ℕ) :
  n = 10 →
  k = 4 →
  total = Nat.choose n k →
  red = Nat.choose (n - 1) k →
  total - red = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_selection_problem_l4124_412444


namespace NUMINAMATH_CALUDE_alice_prob_three_turns_l4124_412435

/-- Represents the person holding the ball -/
inductive Person : Type
| Alice : Person
| Bob : Person

/-- The probability of tossing the ball for each person -/
def toss_prob (p : Person) : ℚ :=
  match p with
  | Person.Alice => 1/3
  | Person.Bob => 1/4

/-- The probability of keeping the ball for each person -/
def keep_prob (p : Person) : ℚ :=
  1 - toss_prob p

/-- The probability of Alice having the ball after n turns, given she starts with it -/
def alice_prob (n : ℕ) : ℚ :=
  sorry

theorem alice_prob_three_turns :
  alice_prob 3 = 227/432 :=
sorry

end NUMINAMATH_CALUDE_alice_prob_three_turns_l4124_412435


namespace NUMINAMATH_CALUDE_investment_rate_problem_l4124_412494

/-- Calculates simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_problem (principal : ℝ) (time : ℝ) (standardRate : ℝ) (additionalInterest : ℝ) :
  principal = 2500 →
  time = 2 →
  standardRate = 0.12 →
  additionalInterest = 300 →
  ∃ (rate : ℝ),
    simpleInterest principal rate time = simpleInterest principal standardRate time + additionalInterest ∧
    rate = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l4124_412494


namespace NUMINAMATH_CALUDE_smallest_c_value_l4124_412461

theorem smallest_c_value (c d : ℤ) : 
  (∃ (r₁ r₂ r₃ : ℤ), 
    r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
    ∀ (x : ℤ), x^3 - c*x^2 + d*x - 3990 = (x - r₁) * (x - r₂) * (x - r₃)) →
  c ≥ 56 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l4124_412461


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l4124_412478

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ (23 * n) % 11 = 5678 % 11 ∧
  ∀ (m : ℕ), m > 0 ∧ (23 * m) % 11 = 5678 % 11 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l4124_412478


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l4124_412472

/-- Given a geometric series with positive terms {a_n}, if a_1, 1/2 * a_3, and 2 * a_2 form an arithmetic sequence, then a_5 / a_3 = 3 + 2√2 -/
theorem geometric_series_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : (a 1 + 2 * a 2) / 2 = a 3 / 2) :
  a 5 / a 3 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l4124_412472


namespace NUMINAMATH_CALUDE_triangle_properties_l4124_412463

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  c = Real.sqrt 3 →
  c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A) →
  C = π/3 ∧ 0 < a - b/2 ∧ a - b/2 < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l4124_412463


namespace NUMINAMATH_CALUDE_correct_production_matching_equation_l4124_412457

/-- Represents a workshop producing bolts and nuts -/
structure Workshop where
  total_workers : ℕ
  bolt_production_rate : ℕ
  nut_production_rate : ℕ
  nuts_per_bolt : ℕ

/-- The equation for matching bolt and nut production in the workshop -/
def production_matching_equation (w : Workshop) (x : ℕ) : Prop :=
  2 * w.bolt_production_rate * x = w.nut_production_rate * (w.total_workers - x)

/-- Theorem stating the correct equation for matching bolt and nut production -/
theorem correct_production_matching_equation (w : Workshop) 
  (h1 : w.total_workers = 28)
  (h2 : w.bolt_production_rate = 12)
  (h3 : w.nut_production_rate = 18)
  (h4 : w.nuts_per_bolt = 2) :
  ∀ x, production_matching_equation w x ↔ 2 * 12 * x = 18 * (28 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_correct_production_matching_equation_l4124_412457


namespace NUMINAMATH_CALUDE_strawberry_cakes_ordered_l4124_412402

/-- The number of strawberry cakes Leila ordered -/
def strawberry_cakes : ℕ := 
  let chocolate_cake_price : ℕ := 12
  let strawberry_cake_price : ℕ := 22
  let chocolate_cakes_ordered : ℕ := 3
  let total_payment : ℕ := 168
  (total_payment - chocolate_cake_price * chocolate_cakes_ordered) / strawberry_cake_price

theorem strawberry_cakes_ordered : strawberry_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_cakes_ordered_l4124_412402


namespace NUMINAMATH_CALUDE_cos_2alpha_from_tan_alpha_plus_pi_4_l4124_412489

theorem cos_2alpha_from_tan_alpha_plus_pi_4 (α : Real) 
  (h : Real.tan (α + π/4) = 2) : 
  Real.cos (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_from_tan_alpha_plus_pi_4_l4124_412489


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l4124_412483

/-- The trajectory of point P -/
def trajectory (x y : ℝ) : Prop :=
  y^2 = 4*x

/-- The condition for point P -/
def point_condition (x y : ℝ) : Prop :=
  2 * Real.sqrt ((x - 1)^2 + y^2) = 2*(x + 1)

/-- The perpendicularity condition for OM and ON -/
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem trajectory_and_intersection :
  -- Point P satisfies the condition
  ∀ x y : ℝ, point_condition x y →
  -- The trajectory is y² = 4x
  (trajectory x y) ∧
  -- For any non-zero m where y = x + m intersects the trajectory at M and N
  ∀ m : ℝ, m ≠ 0 →
    ∃ x₁ y₁ x₂ y₂ : ℝ,
      -- M and N are on the trajectory
      trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
      -- M and N are on the line y = x + m
      y₁ = x₁ + m ∧ y₂ = x₂ + m ∧
      -- OM is perpendicular to ON
      perpendicular_condition x₁ y₁ x₂ y₂ →
      -- Then m = -4
      m = -4 :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l4124_412483


namespace NUMINAMATH_CALUDE_binomial_sum_cubes_l4124_412412

theorem binomial_sum_cubes (x y : ℤ) :
  (x^4 + 9*x*y^3)^3 + (-3*x^3*y - 9*y^4)^3 = x^12 - 729*y^12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_cubes_l4124_412412


namespace NUMINAMATH_CALUDE_set_inequality_l4124_412496

def S : Set ℤ := {x | ∃ k : ℕ+, x = 2 * k + 1}
def A : Set ℤ := {x | ∃ k : ℕ+, x = 2 * k - 1}

theorem set_inequality : A ≠ S := by
  sorry

end NUMINAMATH_CALUDE_set_inequality_l4124_412496


namespace NUMINAMATH_CALUDE_holly_chocolate_milk_container_size_l4124_412417

/-- Represents the amount of chocolate milk Holly has throughout the day -/
structure ChocolateMilk where
  initial : ℕ  -- Initial amount of chocolate milk
  breakfast : ℕ  -- Amount drunk at breakfast
  lunch : ℕ  -- Amount drunk at lunch
  dinner : ℕ  -- Amount drunk at dinner
  final : ℕ  -- Final amount of chocolate milk
  new_container : ℕ  -- Size of the new container bought at lunch

/-- Theorem stating the size of the new container Holly bought -/
theorem holly_chocolate_milk_container_size 
  (h : ChocolateMilk) 
  (h_initial : h.initial = 16)
  (h_breakfast : h.breakfast = 8)
  (h_lunch : h.lunch = 8)
  (h_dinner : h.dinner = 8)
  (h_final : h.final = 56)
  : h.new_container = 64 := by
  sorry

end NUMINAMATH_CALUDE_holly_chocolate_milk_container_size_l4124_412417


namespace NUMINAMATH_CALUDE_consecutive_integers_reciprocal_sum_l4124_412454

/-- The sum of reciprocals of all pairs of three consecutive integers is an integer -/
def is_sum_reciprocals_integer (x : ℤ) : Prop :=
  ∃ (n : ℤ), (x / (x + 1) : ℚ) + (x / (x + 2) : ℚ) + ((x + 1) / x : ℚ) + 
             ((x + 1) / (x + 2) : ℚ) + ((x + 2) / x : ℚ) + ((x + 2) / (x + 1) : ℚ) = n

/-- The only sets of three consecutive integers satisfying the condition are {1, 2, 3} and {-3, -2, -1} -/
theorem consecutive_integers_reciprocal_sum :
  ∀ x : ℤ, is_sum_reciprocals_integer x ↔ (x = 1 ∨ x = -3) :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_reciprocal_sum_l4124_412454


namespace NUMINAMATH_CALUDE_smallest_ellipse_area_l4124_412467

/-- The smallest area of an ellipse containing two specific circles -/
theorem smallest_ellipse_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) :
  ∃ k : ℝ, k = 1/2 ∧ ∀ a' b' : ℝ, (∀ x y : ℝ, x^2/a'^2 + y^2/b'^2 = 1 → 
    ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) → π * a' * b' ≥ k * π := by
  sorry


end NUMINAMATH_CALUDE_smallest_ellipse_area_l4124_412467


namespace NUMINAMATH_CALUDE_function_strictly_increasing_iff_a_in_range_l4124_412451

/-- The function f(x) = (a-2)a^x is strictly increasing if and only if a is in the set (0,1) ∪ (2,+∞) -/
theorem function_strictly_increasing_iff_a_in_range (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (((a - 2) * a^x₁ - (a - 2) * a^x₂) / (x₁ - x₂)) > 0) ↔
  (a ∈ Set.Ioo 0 1 ∪ Set.Ioi 2) :=
sorry

end NUMINAMATH_CALUDE_function_strictly_increasing_iff_a_in_range_l4124_412451


namespace NUMINAMATH_CALUDE_quadratic_function_property_l4124_412466

/-- A quadratic function with vertex (m, k) and point (k, m) on its graph -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  k : ℝ
  a_nonzero : a ≠ 0
  vertex_condition : k = a * m^2 + b * m + c
  point_condition : m = a * k^2 + b * k + c

/-- Theorem stating that a(m - k) > 0 for a quadratic function with the given conditions -/
theorem quadratic_function_property (f : QuadraticFunction) : f.a * (f.m - f.k) > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l4124_412466


namespace NUMINAMATH_CALUDE_set_equality_l4124_412445

def positive_naturals : Set ℕ := {n : ℕ | n > 0}

def set_A : Set ℕ := {x ∈ positive_naturals | x - 3 < 2}
def set_B : Set ℕ := {1, 2, 3, 4}

theorem set_equality : set_A = set_B := by sorry

end NUMINAMATH_CALUDE_set_equality_l4124_412445


namespace NUMINAMATH_CALUDE_smallest_m_for_multiple_factorizations_l4124_412414

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def has_multiple_factorizations (n : ℕ) : Prop :=
  ∃ (f1 f2 : List ℕ), 
    f1 ≠ f2 ∧ 
    f1.length = 16 ∧ 
    f2.length = 16 ∧ 
    f1.Nodup ∧ 
    f2.Nodup ∧ 
    f1.prod = n ∧ 
    f2.prod = n

theorem smallest_m_for_multiple_factorizations :
  (∀ m : ℕ, m > 0 ∧ m < 24 → ¬has_multiple_factorizations (factorial 15 * m)) ∧
  has_multiple_factorizations (factorial 15 * 24) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_multiple_factorizations_l4124_412414


namespace NUMINAMATH_CALUDE_b_investment_is_13650_l4124_412438

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Calculates B's investment given the partnership details. -/
def calculate_b_investment (p : Partnership) : ℕ :=
  p.total_profit * p.a_investment / p.a_profit_share - p.a_investment - p.c_investment

/-- Theorem stating that B's investment is 13650 given the specific partnership details. -/
theorem b_investment_is_13650 (p : Partnership) 
  (h1 : p.a_investment = 6300)
  (h2 : p.c_investment = 10500)
  (h3 : p.total_profit = 12100)
  (h4 : p.a_profit_share = 3630) :
  calculate_b_investment p = 13650 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_is_13650_l4124_412438


namespace NUMINAMATH_CALUDE_min_value_of_a_plus_2b_l4124_412487

theorem min_value_of_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 / a + 1 / b = 1) → (∀ a' b' : ℝ, a' > 0 → b' > 0 → 2 / a' + 1 / b' = 1 → a + 2*b ≤ a' + 2*b') → a + 2*b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_plus_2b_l4124_412487


namespace NUMINAMATH_CALUDE_store_revenue_l4124_412482

theorem store_revenue (december : ℝ) (h1 : december > 0) : 
  let november := (2 / 5 : ℝ) * december
  let january := (1 / 3 : ℝ) * november
  let average := (november + january) / 2
  december / average = 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_store_revenue_l4124_412482


namespace NUMINAMATH_CALUDE_semicircle_chord_length_l4124_412423

theorem semicircle_chord_length (d : ℝ) (h : d > 0) :
  let r := d / 2
  let remaining_area := π * r^2 / 2 - π * (d/4)^2
  remaining_area = 16 * π^3 →
  2 * Real.sqrt (r^2 - (d/4)^2) = 32 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_chord_length_l4124_412423


namespace NUMINAMATH_CALUDE_rectangle_area_after_length_decrease_l4124_412413

theorem rectangle_area_after_length_decrease (square_area : ℝ) 
  (rectangle_length_decrease_percent : ℝ) : 
  square_area = 49 →
  rectangle_length_decrease_percent = 20 →
  let square_side := Real.sqrt square_area
  let initial_rectangle_length := square_side
  let initial_rectangle_width := 2 * square_side
  let new_rectangle_length := initial_rectangle_length * (1 - rectangle_length_decrease_percent / 100)
  let new_rectangle_width := initial_rectangle_width
  new_rectangle_length * new_rectangle_width = 78.4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_after_length_decrease_l4124_412413


namespace NUMINAMATH_CALUDE_jeff_vehicle_collection_l4124_412439

theorem jeff_vehicle_collection (trucks : ℕ) : 
  let cars := 2 * trucks
  trucks + cars = 3 * trucks := by
sorry

end NUMINAMATH_CALUDE_jeff_vehicle_collection_l4124_412439


namespace NUMINAMATH_CALUDE_linear_equation_solution_l4124_412471

/-- Given a linear equation y = kx + b, prove the values of k and b,
    and find x for a specific y value. -/
theorem linear_equation_solution (k b : ℝ) :
  (4 * k + b = -20 ∧ -2 * k + b = 16) →
  (k = -6 ∧ b = 4) ∧
  (∀ x : ℝ, -6 * x + 4 = -8 → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l4124_412471


namespace NUMINAMATH_CALUDE_minimum_economic_loss_l4124_412430

def repair_times : List Nat := [12, 17, 8, 18, 23, 30, 14]
def num_workers : Nat := 3
def loss_per_minute : Nat := 2

def optimal_allocation (times : List Nat) (workers : Nat) : List (List Nat) :=
  sorry

def total_waiting_time (allocation : List (List Nat)) : Nat :=
  sorry

theorem minimum_economic_loss :
  let allocation := optimal_allocation repair_times num_workers
  let total_wait := total_waiting_time allocation
  total_wait * loss_per_minute = 358 := by
  sorry

end NUMINAMATH_CALUDE_minimum_economic_loss_l4124_412430


namespace NUMINAMATH_CALUDE_parabola_intersection_angle_l4124_412495

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line type -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Intersection point type -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem parabola_intersection_angle (C : Parabola) (F M : ℝ × ℝ) (l : Line) 
  (A B : IntersectionPoint) :
  C.equation = (fun x y => y^2 = 8*x) →
  F = (2, 0) →
  M = (-2, 2) →
  l.point = F →
  (C.equation A.x A.y ∧ C.equation B.x B.y) →
  (A.y - M.2) * (B.y - M.2) = -(A.x - M.1) * (B.x - M.1) →
  l.slope = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_angle_l4124_412495


namespace NUMINAMATH_CALUDE_floor_sum_sqrt_equals_floor_sqrt_9n_plus_8_l4124_412458

theorem floor_sum_sqrt_equals_floor_sqrt_9n_plus_8 (n : ℕ) :
  ⌊Real.sqrt n + Real.sqrt (n + 1) + Real.sqrt (n + 2)⌋ = ⌊Real.sqrt (9 * n + 8)⌋ :=
sorry

end NUMINAMATH_CALUDE_floor_sum_sqrt_equals_floor_sqrt_9n_plus_8_l4124_412458


namespace NUMINAMATH_CALUDE_ship_meetings_count_l4124_412462

/-- Represents the number of ships sailing in each direction -/
def num_ships_per_direction : ℕ := 5

/-- Represents the total number of ships -/
def total_ships : ℕ := 2 * num_ships_per_direction

/-- Calculates the total number of meetings between ships -/
def total_meetings : ℕ := num_ships_per_direction * num_ships_per_direction

/-- Theorem stating that the total number of meetings is 25 -/
theorem ship_meetings_count :
  total_meetings = 25 :=
by sorry

end NUMINAMATH_CALUDE_ship_meetings_count_l4124_412462


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l4124_412401

/-- A cylinder with base area S whose lateral surface unfolds into a square has lateral surface area 4πS -/
theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let h := 2 * Real.pi * r
  h = 2 * Real.pi * r →
  2 * Real.pi * r * h = 4 * Real.pi * S :=
by sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l4124_412401


namespace NUMINAMATH_CALUDE_soccer_stars_points_l4124_412484

/-- Calculates the total points for a soccer team given their game results -/
def calculate_total_points (total_games wins losses : ℕ) : ℕ :=
  let draws := total_games - wins - losses
  let points_per_win := 3
  let points_per_draw := 1
  let points_per_loss := 0
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss

/-- Theorem stating that the Soccer Stars team's total points is 46 -/
theorem soccer_stars_points :
  calculate_total_points 20 14 2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_soccer_stars_points_l4124_412484


namespace NUMINAMATH_CALUDE_candy_bar_sales_proof_l4124_412433

/-- The number of candy bars sold on the first day -/
def first_day_sales : ℕ := 190

/-- The number of days Sol sells candy bars in a week -/
def days_per_week : ℕ := 6

/-- The cost of each candy bar in cents -/
def candy_bar_cost : ℕ := 10

/-- The increase in candy bar sales each day after the first day -/
def daily_increase : ℕ := 4

/-- The total earnings in cents for the week -/
def total_earnings : ℕ := 1200

theorem candy_bar_sales_proof :
  (first_day_sales * days_per_week + 
   (daily_increase * (days_per_week - 1) * days_per_week) / 2) * 
  candy_bar_cost = total_earnings :=
sorry

end NUMINAMATH_CALUDE_candy_bar_sales_proof_l4124_412433


namespace NUMINAMATH_CALUDE_inequality_solution_l4124_412485

theorem inequality_solution (x : ℝ) : (x^2 - 1) / ((x + 2)^2) ≥ 0 ↔ 
  x < -2 ∨ (-2 < x ∧ x ≤ -1) ∨ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4124_412485


namespace NUMINAMATH_CALUDE_problem_statement_l4124_412436

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 3

theorem problem_statement :
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → f x ≥ (1/2) * g a x) → a ≤ 4) ∧
  (∀ x : ℝ, x > 0 → log x > 1/exp x - 2/(exp 1 * x)) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l4124_412436


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_l4124_412490

theorem sum_of_x_solutions (x y : ℝ) : 
  y = 8 → x^2 + y^2 = 144 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0 ∧ (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_l4124_412490


namespace NUMINAMATH_CALUDE_triangle_area_l4124_412456

open Real

/-- Given a triangle ABC where angle A is π/6 and the dot product of vectors AB and AC
    equals the tangent of angle A, prove that the area of the triangle is 1/6. -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let angle_A : ℝ := π / 6
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2 = tan angle_A →
  abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) / 2 = 1/6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l4124_412456


namespace NUMINAMATH_CALUDE_star_example_l4124_412408

/-- Custom binary operation ※ -/
def star (a b : ℕ) : ℕ := a * b + a + b

/-- Theorem: (3※4)※1 = 39 -/
theorem star_example : star (star 3 4) 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l4124_412408


namespace NUMINAMATH_CALUDE_share_purchase_price_l4124_412440

/-- Calculates the purchase price of shares given dividend rate, par value, and ROI -/
theorem share_purchase_price 
  (dividend_rate : ℝ) 
  (par_value : ℝ) 
  (roi : ℝ) 
  (h1 : dividend_rate = 0.125)
  (h2 : par_value = 40)
  (h3 : roi = 0.25) : 
  (dividend_rate * par_value) / roi = 20 := by
  sorry

#check share_purchase_price

end NUMINAMATH_CALUDE_share_purchase_price_l4124_412440


namespace NUMINAMATH_CALUDE_sum_coefficients_without_x_cubed_l4124_412474

theorem sum_coefficients_without_x_cubed : 
  let n : ℕ := 5
  let all_coeff_sum : ℕ := 2^n
  let x_cubed_coeff : ℕ := n.choose 3
  all_coeff_sum - x_cubed_coeff = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_coefficients_without_x_cubed_l4124_412474


namespace NUMINAMATH_CALUDE_cubic_feet_to_cubic_inches_l4124_412424

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Define the volume in cubic feet
def cubic_feet : ℕ := 4

-- Theorem statement
theorem cubic_feet_to_cubic_inches :
  cubic_feet * (inches_per_foot ^ 3) = 6912 := by
  sorry


end NUMINAMATH_CALUDE_cubic_feet_to_cubic_inches_l4124_412424


namespace NUMINAMATH_CALUDE_exponential_linear_independence_l4124_412407

theorem exponential_linear_independence 
  (k₁ k₂ k₃ : ℝ) 
  (h₁ : k₁ ≠ k₂) 
  (h₂ : k₁ ≠ k₃) 
  (h₃ : k₂ ≠ k₃) :
  ∀ (α₁ α₂ α₃ : ℝ), 
  (∀ x : ℝ, α₁ * Real.exp (k₁ * x) + α₂ * Real.exp (k₂ * x) + α₃ * Real.exp (k₃ * x) = 0) → 
  α₁ = 0 ∧ α₂ = 0 ∧ α₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_exponential_linear_independence_l4124_412407


namespace NUMINAMATH_CALUDE_molecular_weight_difference_l4124_412441

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def atomic_weight_C : ℝ := 12.01

-- Define molecular weights of compounds
def molecular_weight_A : ℝ := atomic_weight_N + 4 * atomic_weight_H + atomic_weight_Br
def molecular_weight_B : ℝ := 2 * atomic_weight_O + atomic_weight_C + 3 * atomic_weight_H

-- Theorem statement
theorem molecular_weight_difference :
  molecular_weight_A - molecular_weight_B = 50.91 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_difference_l4124_412441


namespace NUMINAMATH_CALUDE_circular_coin_flip_probability_l4124_412479

def num_people : ℕ := 10

-- Function to calculate the number of valid arrangements
def valid_arrangements : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| n + 3 => valid_arrangements (n + 1) + valid_arrangements (n + 2)

theorem circular_coin_flip_probability :
  (valid_arrangements num_people : ℚ) / 2^num_people = 123 / 1024 := by sorry

end NUMINAMATH_CALUDE_circular_coin_flip_probability_l4124_412479


namespace NUMINAMATH_CALUDE_expression_evaluation_l4124_412411

theorem expression_evaluation (x y : ℚ) (hx : x = -2) (hy : y = 1) :
  (-2 * x + x + 3 * y) - 2 * (-x^2 - 2 * x + 1/2 * y) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4124_412411


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4124_412464

-- Define an arithmetic sequence with first term a₁ and common difference d
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_common_difference :
  ∃ d : ℝ, ∀ n : ℕ, arithmeticSequence 2 d n = 2 + (n - 1) * d ∧ arithmeticSequence 2 d 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4124_412464


namespace NUMINAMATH_CALUDE_range_of_a_l4124_412405

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4124_412405


namespace NUMINAMATH_CALUDE_license_plate_count_l4124_412421

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 5

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of positions where the letter block can be placed -/
def letter_block_positions : ℕ := digits_count + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count : total_license_plates = 10584576000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l4124_412421


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l4124_412449

theorem binomial_expansion_example : 7^4 + 4*(7^3) + 6*(7^2) + 4*7 + 1 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l4124_412449


namespace NUMINAMATH_CALUDE_quadratic_roots_interlace_l4124_412422

theorem quadratic_roots_interlace (p1 p2 q1 q2 : ℝ) 
  (h : (q1 - q2)^2 + (p1 - p2)*(p1*q2 - p2*q1) < 0) :
  ∃ (α1 β1 α2 β2 : ℝ),
    (∀ x, x^2 + p1*x + q1 = (x - α1) * (x - β1)) ∧
    (∀ x, x^2 + p2*x + q2 = (x - α2) * (x - β2)) ∧
    ((α1 < α2 ∧ α2 < β1 ∧ β1 < β2) ∨ (α2 < α1 ∧ α1 < β2 ∧ β2 < β1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_interlace_l4124_412422


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l4124_412432

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a configuration -/
def add_tiles (config : TileConfiguration) (new_tiles : ℕ) : TileConfiguration :=
  { tiles := config.tiles + new_tiles, perimeter := config.perimeter }

theorem perimeter_after_adding_tiles 
  (initial : TileConfiguration) 
  (added_tiles : ℕ) 
  (final : TileConfiguration) :
  initial.tiles = 10 →
  initial.perimeter = 16 →
  added_tiles = 4 →
  final = add_tiles initial added_tiles →
  final.perimeter = 18 :=
by
  sorry

#check perimeter_after_adding_tiles

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l4124_412432


namespace NUMINAMATH_CALUDE_devin_age_l4124_412420

theorem devin_age (devin_age eden_age mom_age : ℕ) : 
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  (devin_age + eden_age + mom_age) / 3 = 28 →
  devin_age = 12 := by
sorry

end NUMINAMATH_CALUDE_devin_age_l4124_412420


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4124_412492

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) :
  (1 / (a + 3 * b)) + (1 / (b + 3 * c)) + (1 / (c + 3 * a)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4124_412492


namespace NUMINAMATH_CALUDE_min_value_sum_equality_condition_l4124_412452

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (5 * c)) + (c / (6 * a)) ≥ 3 / Real.rpow 90 (1/3) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (5 * c)) + (c / (6 * a)) = 3 / Real.rpow 90 (1/3) ↔
  (a / (3 * b)) = (b / (5 * c)) ∧ (b / (5 * c)) = (c / (6 * a)) ∧ 
  (c / (6 * a)) = Real.rpow (1/90) (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_equality_condition_l4124_412452


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4124_412427

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x + 1) * (x - 2) > 0} = {x : ℝ | x < -1 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4124_412427


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l4124_412406

-- Define the position function
def s (t : ℝ) : ℝ := t^2 - 2*t + 5

-- Define the velocity function as the derivative of the position function
def v (t : ℝ) : ℝ := 2*t - 2

-- Theorem statement
theorem instantaneous_velocity_at_4_seconds :
  v 4 = 6 :=
sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l4124_412406


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_99_l4124_412418

theorem six_digit_multiple_of_99 : ∃ n : ℕ, 
  (n ≥ 978600 ∧ n < 978700) ∧  -- Six-digit number starting with 9786
  (n % 99 = 0) ∧               -- Divisible by 99
  (n / 99 = 6039) :=           -- Quotient is 6039
by sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_99_l4124_412418


namespace NUMINAMATH_CALUDE_hot_dog_buns_packages_l4124_412480

/-- Calculates the number of packages of hot dog buns needed for a school picnic --/
theorem hot_dog_buns_packages (buns_per_package : ℕ) (num_classes : ℕ) (students_per_class : ℕ) (buns_per_student : ℕ) : 
  buns_per_package = 8 →
  num_classes = 4 →
  students_per_class = 30 →
  buns_per_student = 2 →
  (num_classes * students_per_class * buns_per_student + buns_per_package - 1) / buns_per_package = 30 := by
  sorry

#eval (4 * 30 * 2 + 8 - 1) / 8  -- Should output 30

end NUMINAMATH_CALUDE_hot_dog_buns_packages_l4124_412480


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l4124_412400

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a - 3)*x

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a - 3)

/-- Theorem stating the equation of the tangent line at the origin -/
theorem tangent_line_at_origin (a : ℝ) (h : ∀ x, f' a x = f' a (-x)) :
  ∃ m : ℝ, m = -3 ∧ ∀ x, f a x = m * x + f a 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l4124_412400


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l4124_412497

theorem simplify_sqrt_expression :
  Real.sqrt 5 - Real.sqrt 20 + Real.sqrt 45 - 2 * Real.sqrt 80 = -6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l4124_412497


namespace NUMINAMATH_CALUDE_distribute_10_4_l4124_412475

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem distribute_10_4 : distribute 10 4 = 286 := by
  sorry

end NUMINAMATH_CALUDE_distribute_10_4_l4124_412475


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l4124_412450

-- Problem 1
theorem problem_1 : Real.sqrt 9 + |3 - Real.pi| - Real.sqrt ((-3)^2) = Real.pi - 3 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x : ℝ, 3 * (x - 1)^3 = 81 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l4124_412450


namespace NUMINAMATH_CALUDE_frame_uncovered_area_l4124_412403

/-- The area of a rectangular frame not covered by a photo -/
theorem frame_uncovered_area (frame_length frame_width photo_length photo_width : ℝ)
  (h1 : frame_length = 40)
  (h2 : frame_width = 32)
  (h3 : photo_length = 32)
  (h4 : photo_width = 28) :
  frame_length * frame_width - photo_length * photo_width = 384 := by
  sorry

end NUMINAMATH_CALUDE_frame_uncovered_area_l4124_412403


namespace NUMINAMATH_CALUDE_boards_nailed_proof_l4124_412455

/-- Represents the number of boards nailed by each person -/
def num_boards : ℕ := 30

/-- Represents the total number of nails used by Petrov -/
def petrov_nails : ℕ := 87

/-- Represents the total number of nails used by Vasechkin -/
def vasechkin_nails : ℕ := 94

/-- Theorem stating that the number of boards nailed by each person is 30 -/
theorem boards_nailed_proof :
  ∃ (p2 p3 v3 v5 : ℕ),
    p2 + p3 = num_boards ∧
    v3 + v5 = num_boards ∧
    2 * p2 + 3 * p3 = petrov_nails ∧
    3 * v3 + 5 * v5 = vasechkin_nails :=
by sorry


end NUMINAMATH_CALUDE_boards_nailed_proof_l4124_412455
