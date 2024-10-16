import Mathlib

namespace NUMINAMATH_CALUDE_opposite_numbers_x_same_type_radicals_m_l1303_130332

-- Part 1
theorem opposite_numbers_x (x : ℝ) : 
  2 * x^2 + 3 * x - 5 = -((-2 * x + 2)) → x = -3/2 ∨ x = 1 := by sorry

-- Part 2
theorem same_type_radicals_m (m : ℝ) :
  m^2 - 6 ≥ 0 ∧ 6 * m + 1 ≥ 0 ∧ m^2 - 6 = 6 * m + 1 → m = 7 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_x_same_type_radicals_m_l1303_130332


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1303_130377

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (2 + a * Complex.I) / (1 - Complex.I)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1303_130377


namespace NUMINAMATH_CALUDE_price_decrease_l1303_130379

/-- The price of 6 packets last month in dollars -/
def last_month_price : ℚ := 7.5

/-- The number of packets in last month's offer -/
def last_month_packets : ℕ := 6

/-- The price of 10 packets this month in dollars -/
def this_month_price : ℚ := 11

/-- The number of packets in this month's offer -/
def this_month_packets : ℕ := 10

/-- The percent decrease in price per packet -/
def percent_decrease : ℚ := 12

theorem price_decrease :
  (last_month_price / last_month_packets - this_month_price / this_month_packets) /
  (last_month_price / last_month_packets) * 100 = percent_decrease := by
  sorry


end NUMINAMATH_CALUDE_price_decrease_l1303_130379


namespace NUMINAMATH_CALUDE_probability_brothers_names_l1303_130329

/-- The probability of selecting one letter from each brother's name -/
def probability_one_from_each (total_cards : ℕ) (allen_cards : ℕ) (james_cards : ℕ) : ℚ :=
  if total_cards = allen_cards + james_cards ∧ allen_cards = james_cards then
    (allen_cards : ℚ) * james_cards / (total_cards * (total_cards - 1))
  else
    0

theorem probability_brothers_names :
  probability_one_from_each 12 6 6 = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_probability_brothers_names_l1303_130329


namespace NUMINAMATH_CALUDE_implication_equiv_contrapositive_l1303_130376

-- Define the propositions
variable (P Q : Prop)

-- Define the original implication
def original : Prop := P → Q

-- Define the contrapositive
def contrapositive : Prop := ¬Q → ¬P

-- Theorem stating the equivalence of the original implication and its contrapositive
theorem implication_equiv_contrapositive :
  original P Q ↔ contrapositive P Q :=
sorry

end NUMINAMATH_CALUDE_implication_equiv_contrapositive_l1303_130376


namespace NUMINAMATH_CALUDE_brady_earnings_brady_earnings_200_l1303_130318

/-- Brady's earnings for transcribing recipe cards -/
theorem brady_earnings : ℕ → ℚ
  | cards => 
    let base_pay := (70 : ℚ) / 100 * cards
    let bonus := 10 * (cards / 100 : ℕ)
    base_pay + bonus

/-- Proof of Brady's earnings for 200 cards -/
theorem brady_earnings_200 : brady_earnings 200 = 160 := by
  sorry

end NUMINAMATH_CALUDE_brady_earnings_brady_earnings_200_l1303_130318


namespace NUMINAMATH_CALUDE_parallel_line_plane_conditions_l1303_130358

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the parallel relation
def parallel (x y : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the subset relation for a line in a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem parallel_line_plane_conditions
  (a b : Line) (α : Plane) (h : line_in_plane a α) :
  ¬(∀ (h1 : parallel a b), parallel_line_plane b α) ∧
  ¬(∀ (h2 : parallel_line_plane b α), parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_conditions_l1303_130358


namespace NUMINAMATH_CALUDE_bacteria_division_theorem_l1303_130378

/-- Represents a binary tree of bacteria -/
inductive BacteriaTree
  | Leaf : BacteriaTree
  | Node : BacteriaTree → BacteriaTree → BacteriaTree

/-- Counts the number of nodes in a BacteriaTree -/
def count_nodes : BacteriaTree → Nat
  | BacteriaTree.Leaf => 1
  | BacteriaTree.Node left right => count_nodes left + count_nodes right

/-- Checks if a subtree with the desired properties exists -/
def exists_balanced_subtree (tree : BacteriaTree) : Prop :=
  ∃ (subtree : BacteriaTree), 
    (count_nodes subtree ≥ 334 ∧ count_nodes subtree ≤ 667)

theorem bacteria_division_theorem (tree : BacteriaTree) 
  (h : count_nodes tree = 1000) : 
  exists_balanced_subtree tree :=
sorry

end NUMINAMATH_CALUDE_bacteria_division_theorem_l1303_130378


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1303_130360

-- System 1
theorem system_one_solution (x y : ℝ) : 
  x - y = 1 ∧ 2*x + y = 5 → x = 2 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  x/2 - (y+1)/3 = 1 ∧ x + y = 1 → x = 2 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1303_130360


namespace NUMINAMATH_CALUDE_naomi_saw_58_wheels_l1303_130322

/-- The number of regular bikes at the park -/
def regular_bikes : ℕ := 7

/-- The number of children's bikes at the park -/
def children_bikes : ℕ := 11

/-- The number of wheels on a regular bike -/
def regular_bike_wheels : ℕ := 2

/-- The number of wheels on a children's bike -/
def children_bike_wheels : ℕ := 4

/-- The total number of wheels Naomi saw at the park -/
def total_wheels : ℕ := regular_bikes * regular_bike_wheels + children_bikes * children_bike_wheels

theorem naomi_saw_58_wheels : total_wheels = 58 := by
  sorry

end NUMINAMATH_CALUDE_naomi_saw_58_wheels_l1303_130322


namespace NUMINAMATH_CALUDE_fifth_quiz_score_l1303_130330

def quiz_scores : List ℕ := [90, 98, 92, 94]
def desired_average : ℕ := 94
def total_quizzes : ℕ := 5

theorem fifth_quiz_score (scores : List ℕ) (avg : ℕ) (total : ℕ) :
  scores = quiz_scores ∧ avg = desired_average ∧ total = total_quizzes →
  (scores.sum + (avg * total - scores.sum)) / total = avg ∧
  avg * total - scores.sum = 96 := by
  sorry

end NUMINAMATH_CALUDE_fifth_quiz_score_l1303_130330


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l1303_130354

theorem perfect_square_binomial (a b : ℝ) : ∃ (x : ℝ), a^2 + 2*a*b + b^2 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l1303_130354


namespace NUMINAMATH_CALUDE_fraction_equality_l1303_130302

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 25)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 8) :
  a / d = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1303_130302


namespace NUMINAMATH_CALUDE_digital_display_overlap_l1303_130307

/-- Represents a digital number display in a rectangle -/
structure DigitalDisplay where
  width : Nat
  height : Nat
  numbers : List Nat

/-- Represents the overlap of two digital displays -/
def overlap (d1 d2 : DigitalDisplay) : Nat :=
  sorry

/-- The main theorem about overlapping digital displays -/
theorem digital_display_overlap :
  ∀ (d : DigitalDisplay),
    d.width = 8 ∧ 
    d.height = 5 ∧ 
    d.numbers = [1, 2, 1, 9] ∧
    (overlap d (DigitalDisplay.mk 8 5 [6, 1, 2, 1])) = 30 := by
  sorry

end NUMINAMATH_CALUDE_digital_display_overlap_l1303_130307


namespace NUMINAMATH_CALUDE_two_digit_reverse_diff_64_l1303_130342

/-- Given a two-digit number, return the number formed by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A two-digit number is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_reverse_diff_64 (N : ℕ) :
  is_two_digit N →
  N - reverse_digits N = 64 →
  N = 90 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_reverse_diff_64_l1303_130342


namespace NUMINAMATH_CALUDE_smallest_number_theorem_l1303_130316

def is_multiple_of_36 (n : ℕ) : Prop := ∃ k : ℕ, n = 36 * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of_36 n ∧ (digit_product n % 9 = 0)

theorem smallest_number_theorem :
  satisfies_conditions 936 ∧ ∀ m : ℕ, m < 936 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_theorem_l1303_130316


namespace NUMINAMATH_CALUDE_right_triangle_exists_l1303_130334

theorem right_triangle_exists (a : ℤ) (h : a ≥ 5) :
  ∃ b c : ℤ, c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_exists_l1303_130334


namespace NUMINAMATH_CALUDE_males_not_listening_l1303_130352

theorem males_not_listening (males_listening : ℕ) (females_not_listening : ℕ) 
  (total_listening : ℕ) (total_not_listening : ℕ) 
  (h1 : males_listening = 45)
  (h2 : females_not_listening = 87)
  (h3 : total_listening = 115)
  (h4 : total_not_listening = 160) : 
  total_listening + total_not_listening - (males_listening + (total_listening - males_listening + females_not_listening)) = 73 :=
by sorry

end NUMINAMATH_CALUDE_males_not_listening_l1303_130352


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1303_130351

theorem inequality_and_equality_condition (x y : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  8 * x * y ≤ 5 * x * (1 - x) + 5 * y * (1 - y) ∧
  (8 * x * y = 5 * x * (1 - x) + 5 * y * (1 - y) ↔ x = 1/2 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1303_130351


namespace NUMINAMATH_CALUDE_two_rotational_homotheties_l1303_130314

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rotational homothety -/
structure RotationalHomothety where
  center : ℝ × ℝ
  angle : ℝ
  scale : ℝ

/-- Applies a rotational homothety to a circle -/
def applyRotationalHomothety (h : RotationalHomothety) (c : Circle) : Circle :=
  sorry

/-- Checks if two circles are equal -/
def circlesEqual (c1 c2 : Circle) : Prop :=
  sorry

/-- Main theorem -/
theorem two_rotational_homotheties 
  (S₁ S₂ : Circle) 
  (h : S₁.center ≠ S₂.center) : 
  ∃! (pair : (RotationalHomothety × RotationalHomothety)),
    (circlesEqual (applyRotationalHomothety pair.1 S₁) S₂) ∧
    (circlesEqual (applyRotationalHomothety pair.2 S₁) S₂) ∧
    (pair.1.angle = π/2) ∧ (pair.2.angle = π/2) ∧
    (pair.1.center ≠ pair.2.center) :=
  sorry

end NUMINAMATH_CALUDE_two_rotational_homotheties_l1303_130314


namespace NUMINAMATH_CALUDE_second_company_base_rate_l1303_130303

/-- The base rate of United Telephone in dollars -/
def united_base_rate : ℝ := 7

/-- The per-minute rate of United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The per-minute rate of the second telephone company in dollars -/
def second_per_minute : ℝ := 0.20

/-- The number of minutes for which the bills are equal -/
def equal_minutes : ℝ := 100

/-- The base rate of the second telephone company in dollars -/
def second_base_rate : ℝ := 12

theorem second_company_base_rate :
  united_base_rate + united_per_minute * equal_minutes =
  second_base_rate + second_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_second_company_base_rate_l1303_130303


namespace NUMINAMATH_CALUDE_propositions_proof_l1303_130308

theorem propositions_proof :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ a + 1/a ≤ b + 1/b) ∧
  (∀ m n : ℝ, m > n ∧ n > 0 → (m + 1) / (n + 1) < m / n) ∧
  (∀ c a b : ℝ, c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) ∧
  (∀ a b : ℝ, a ≥ b ∧ b > -1 → a / (a + 1) ≥ b / (b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_propositions_proof_l1303_130308


namespace NUMINAMATH_CALUDE_correct_nail_count_l1303_130396

/-- Calculates the total number of nails used to fix a square plate -/
def total_nails (nails_per_side : ℕ) : ℕ :=
  4 * nails_per_side - 4

/-- Theorem stating the correct number of nails used -/
theorem correct_nail_count :
  let plate_side_length : ℕ := 24
  let nails_per_side : ℕ := 25
  total_nails nails_per_side = 96 := by
  sorry

end NUMINAMATH_CALUDE_correct_nail_count_l1303_130396


namespace NUMINAMATH_CALUDE_elements_less_than_k_bound_l1303_130333

/-- A sequence of positive integers satisfying the given conditions -/
def SpecialSequence : Type := ℕ → ℕ

/-- The first two elements of the sequence are 1 and 2 -/
axiom first_two_elements (s : SpecialSequence) : s 0 = 1 ∧ s 1 = 2

/-- The sequence contains only positive integers -/
axiom positive_elements (s : SpecialSequence) (n : ℕ) : s n > 0

/-- No sum of two different elements is in the sequence -/
axiom no_sum_in_sequence (s : SpecialSequence) (i j n : ℕ) :
  i ≠ j → s i + s j ≠ s n

/-- The number of elements less than k is at most k/3 + 2 -/
theorem elements_less_than_k_bound (s : SpecialSequence) (k : ℕ) :
  (Finset.filter (fun n => s n < k) (Finset.range k)).card ≤ k / 3 + 2 :=
sorry

end NUMINAMATH_CALUDE_elements_less_than_k_bound_l1303_130333


namespace NUMINAMATH_CALUDE_total_money_l1303_130365

/-- Given that A and C together have 200, B and C together have 350, and C has 200,
    prove that the total amount of money A, B, and C have between them is 350. -/
theorem total_money (A B C : ℕ) 
  (hAC : A + C = 200)
  (hBC : B + C = 350)
  (hC : C = 200) : 
  A + B + C = 350 := by
sorry

end NUMINAMATH_CALUDE_total_money_l1303_130365


namespace NUMINAMATH_CALUDE_at_least_three_pass_six_students_l1303_130375

def exam_pass_probability : ℚ := 1/3

def at_least_three_pass (n : ℕ) (p : ℚ) : ℚ :=
  1 - (Nat.choose n 0 * (1 - p)^n +
       Nat.choose n 1 * p * (1 - p)^(n-1) +
       Nat.choose n 2 * p^2 * (1 - p)^(n-2))

theorem at_least_three_pass_six_students :
  at_least_three_pass 6 exam_pass_probability = 353/729 := by
  sorry

end NUMINAMATH_CALUDE_at_least_three_pass_six_students_l1303_130375


namespace NUMINAMATH_CALUDE_brick_count_for_wall_l1303_130362

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

/-- Converts meters to centimeters -/
def meters_to_cm (m : ℝ) : ℝ :=
  m * 100

theorem brick_count_for_wall :
  let brick_length : ℝ := 20
  let brick_width : ℝ := 10
  let brick_height : ℝ := 7.5
  let wall_length : ℝ := 27
  let wall_width : ℝ := 2
  let wall_height : ℝ := 0.75
  let brick_volume : ℝ := volume brick_length brick_width brick_height
  let wall_volume : ℝ := volume (meters_to_cm wall_length) (meters_to_cm wall_width) (meters_to_cm wall_height)
  (wall_volume / brick_volume : ℝ) = 27000 :=
by sorry

end NUMINAMATH_CALUDE_brick_count_for_wall_l1303_130362


namespace NUMINAMATH_CALUDE_m_range_l1303_130324

def p (x : ℝ) : Prop := |1 - (x - 2)/3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem m_range (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → ¬(p x)) →
  (∃ x, q x m ∧ ¬(p x)) →
  m ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1303_130324


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1303_130356

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x - 2 > 0 ↔ -2 < x ∧ x < -1/4) →
  a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1303_130356


namespace NUMINAMATH_CALUDE_train_length_approximation_l1303_130340

/-- The length of a train given its speed and time to cross a fixed point -/
def trainLength (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train crossing a telegraph post in 13 seconds at 58.15384615384615 m/s has a length of approximately 756 meters -/
theorem train_length_approximation :
  let speed : ℝ := 58.15384615384615
  let time : ℝ := 13
  let length := trainLength speed time
  ∃ ε > 0, |length - 756| < ε :=
by sorry

end NUMINAMATH_CALUDE_train_length_approximation_l1303_130340


namespace NUMINAMATH_CALUDE_value_of_expression_l1303_130310

theorem value_of_expression (x y : ℝ) 
  (h1 : x^2 - x*y = 12) 
  (h2 : y^2 - x*y = 15) : 
  2*(x-y)^2 - 3 = 51 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l1303_130310


namespace NUMINAMATH_CALUDE_sqrt_product_property_l1303_130357

theorem sqrt_product_property : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_property_l1303_130357


namespace NUMINAMATH_CALUDE_electronics_store_purchase_l1303_130385

theorem electronics_store_purchase (total people_tv people_computer people_both : ℕ) 
  (h1 : total = 15)
  (h2 : people_tv = 9)
  (h3 : people_computer = 7)
  (h4 : people_both = 3)
  : total - (people_tv + people_computer - people_both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_electronics_store_purchase_l1303_130385


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l1303_130309

theorem complex_equation_real_solution (a : ℝ) : 
  (((a : ℂ) / (1 + Complex.I) + (1 + Complex.I) / 2).im = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l1303_130309


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_is_two_l1303_130380

/-- The ratio of students preferring spaghetti to those preferring manicotti -/
def pasta_preference_ratio (spaghetti_count : ℕ) (manicotti_count : ℕ) : ℚ :=
  spaghetti_count / manicotti_count

/-- The total number of students surveyed -/
def total_students : ℕ := 800

/-- The number of students who preferred spaghetti -/
def spaghetti_preference : ℕ := 320

/-- The number of students who preferred manicotti -/
def manicotti_preference : ℕ := 160

theorem pasta_preference_ratio_is_two :
  pasta_preference_ratio spaghetti_preference manicotti_preference = 2 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_is_two_l1303_130380


namespace NUMINAMATH_CALUDE_sum_of_angles_l1303_130315

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z^5 = -32 * i

-- Define the form of solutions
def solution_form (z : ℂ) (s : ℝ) (α : ℝ) : Prop :=
  z = s * (Complex.cos α + i * Complex.sin α)

-- Define the conditions on s and α
def valid_solution (s : ℝ) (α : ℝ) : Prop :=
  s > 0 ∧ 0 ≤ α ∧ α < 2 * Real.pi

-- Theorem statement
theorem sum_of_angles (z₁ z₂ z₃ z₄ z₅ : ℂ) (s₁ s₂ s₃ s₄ s₅ α₁ α₂ α₃ α₄ α₅ : ℝ) :
  equation z₁ ∧ equation z₂ ∧ equation z₃ ∧ equation z₄ ∧ equation z₅ ∧
  solution_form z₁ s₁ α₁ ∧ solution_form z₂ s₂ α₂ ∧ solution_form z₃ s₃ α₃ ∧
  solution_form z₄ s₄ α₄ ∧ solution_form z₅ s₅ α₅ ∧
  valid_solution s₁ α₁ ∧ valid_solution s₂ α₂ ∧ valid_solution s₃ α₃ ∧
  valid_solution s₄ α₄ ∧ valid_solution s₅ α₅ →
  α₁ + α₂ + α₃ + α₄ + α₅ = 5.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_l1303_130315


namespace NUMINAMATH_CALUDE_min_box_value_l1303_130341

theorem min_box_value (a b Box : ℤ) :
  (∀ x, (a * x + b) * (b * x + a) = 26 * x^2 + Box * x + 26) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  (∃ a' b' Box' : ℤ, 
    (∀ x, (a' * x + b') * (b' * x + a') = 26 * x^2 + Box' * x + 26) ∧
    a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' ∧
    Box' < Box) →
  Box ≥ 173 :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l1303_130341


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1303_130337

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  ArithmeticSequence a → a 2 = 5 → a 6 = 33 → a 3 + a 5 = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1303_130337


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l1303_130355

theorem consecutive_integers_cube_sum (n : ℤ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 2106 →
  (n - 1)^3 + n^3 + (n + 1)^3 + (n + 2)^3 = 45900 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l1303_130355


namespace NUMINAMATH_CALUDE_cat_rat_kill_time_l1303_130359

/-- The time it takes for cats to kill rats -/
def kill_time (n : ℕ) : ℝ :=
  3 -- 3 minutes

/-- Proposition: For any number of cats n ≥ 3, it takes 3 minutes for n cats to kill n rats -/
theorem cat_rat_kill_time (n : ℕ) (h : n ≥ 3) : kill_time n = 3 := by
  sorry

/-- Given: 3 cats can kill 3 rats in a certain amount of time -/
axiom three_cats_kill_three_rats : ∃ t : ℝ, t > 0 ∧ kill_time 3 = t

/-- Given: It takes 3 minutes for 100 cats to kill 100 rats -/
axiom hundred_cats_kill_hundred_rats : kill_time 100 = 3

end NUMINAMATH_CALUDE_cat_rat_kill_time_l1303_130359


namespace NUMINAMATH_CALUDE_sine_central_angle_is_zero_l1303_130344

/-- Represents a circle with intersecting chords -/
structure IntersectingChords where
  radius : ℝ
  pq_length : ℝ
  rt_length : ℝ

/-- The sine of the central angle subtending arc PR in the given circle configuration -/
def sine_central_angle (c : IntersectingChords) : ℝ :=
  sorry

/-- Theorem stating that the sine of the central angle is 0 for the given configuration -/
theorem sine_central_angle_is_zero (c : IntersectingChords) 
  (h1 : c.radius = 7)
  (h2 : c.pq_length = 14)
  (h3 : c.rt_length = 5) : 
  sine_central_angle c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_central_angle_is_zero_l1303_130344


namespace NUMINAMATH_CALUDE_probability_x_less_than_2y_is_five_sixths_l1303_130349

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The specific rectangle in the problem --/
def problemRectangle : Rectangle where
  x_min := 0
  x_max := 6
  y_min := 0
  y_max := 2
  h_x := by norm_num
  h_y := by norm_num

/-- The probability of selecting a point (x,y) from the rectangle such that x < 2y --/
def probabilityXLessThan2Y (r : Rectangle) : ℝ :=
  sorry

theorem probability_x_less_than_2y_is_five_sixths :
  probabilityXLessThan2Y problemRectangle = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_less_than_2y_is_five_sixths_l1303_130349


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1303_130345

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4) / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1303_130345


namespace NUMINAMATH_CALUDE_exists_parallel_planes_nonparallel_lines_perpendicular_line_implies_perpendicular_planes_l1303_130366

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Statement 1
theorem exists_parallel_planes_nonparallel_lines :
  ∃ (α β : Plane) (l m : Line),
    subset l α ∧ subset m β ∧ parallel_planes α β ∧ ¬parallel_lines l m :=
sorry

-- Statement 2
theorem perpendicular_line_implies_perpendicular_planes
  (α β : Plane) (l : Line) :
  subset l α → perpendicular_line_plane l β → perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_exists_parallel_planes_nonparallel_lines_perpendicular_line_implies_perpendicular_planes_l1303_130366


namespace NUMINAMATH_CALUDE_ways_to_write_1800_as_sum_of_twos_and_threes_l1303_130389

/-- The number of ways to write a positive integer as a sum of 2s and 3s -/
def num_ways_as_sum_of_twos_and_threes (n : ℕ) : ℕ :=
  (n / 6 + 1) - (n % 6 / 2)

/-- Theorem stating that there are 301 ways to write 1800 as a sum of 2s and 3s -/
theorem ways_to_write_1800_as_sum_of_twos_and_threes :
  num_ways_as_sum_of_twos_and_threes 1800 = 301 := by
  sorry

#eval num_ways_as_sum_of_twos_and_threes 1800

end NUMINAMATH_CALUDE_ways_to_write_1800_as_sum_of_twos_and_threes_l1303_130389


namespace NUMINAMATH_CALUDE_water_heater_capacity_l1303_130371

/-- Represents a water heater with given parameters -/
structure WaterHeater where
  initialCapacity : ℝ
  addRate : ℝ → ℝ
  dischargeRate : ℝ → ℝ
  maxPersonUsage : ℝ

/-- Calculates the water volume as a function of time -/
def waterVolume (heater : WaterHeater) (t : ℝ) : ℝ :=
  heater.initialCapacity + heater.addRate t - heater.dischargeRate t

/-- Theorem: The given water heater can supply at least 4 people for continuous showers -/
theorem water_heater_capacity (heater : WaterHeater) 
  (h1 : heater.initialCapacity = 200)
  (h2 : ∀ t, heater.addRate t = 2 * t^2)
  (h3 : ∀ t, heater.dischargeRate t = 34 * t)
  (h4 : heater.maxPersonUsage = 60) :
  ∃ n : ℕ, n ≥ 4 ∧ 
    (∃ t : ℝ, t > 0 ∧ 
      heater.dischargeRate t / heater.maxPersonUsage ≥ n ∧
      ∀ s, 0 ≤ s ∧ s ≤ t → waterVolume heater s ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_water_heater_capacity_l1303_130371


namespace NUMINAMATH_CALUDE_range_of_expression_l1303_130387

-- Define the conditions
def condition1 (x y : ℝ) : Prop := -1 < x + y ∧ x + y < 4
def condition2 (x y : ℝ) : Prop := 2 < x - y ∧ x - y < 3

-- Define the expression we're interested in
def expression (x y : ℝ) : ℝ := 3*x + 2*y

-- State the theorem
theorem range_of_expression (x y : ℝ) 
  (h1 : condition1 x y) (h2 : condition2 x y) :
  -3/2 < expression x y ∧ expression x y < 23/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l1303_130387


namespace NUMINAMATH_CALUDE_average_weight_increase_l1303_130364

/-- 
Proves that replacing a person weighing 65 kg with a person weighing 97 kg 
in a group of 10 people increases the average weight by 3.2 kg
-/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 10 * initial_average
  let new_total := initial_total - 65 + 97
  let new_average := new_total / 10
  new_average - initial_average = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l1303_130364


namespace NUMINAMATH_CALUDE_jewelry_sweater_difference_l1303_130319

theorem jewelry_sweater_difference (sweater_cost initial_fraction remaining : ℚ) :
  sweater_cost = 40 →
  initial_fraction = 1/4 →
  remaining = 20 →
  let initial_money := sweater_cost / initial_fraction
  let jewelry_cost := initial_money - sweater_cost - remaining
  jewelry_cost - sweater_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_sweater_difference_l1303_130319


namespace NUMINAMATH_CALUDE_eldoria_license_plates_l1303_130321

/-- The number of possible uppercase letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letter positions in a license plate. -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate. -/
def digit_positions : ℕ := 4

/-- The total number of possible license plates in Eldoria. -/
def total_license_plates : ℕ := num_letters ^ letter_positions * num_digits ^ digit_positions

theorem eldoria_license_plates :
  total_license_plates = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_eldoria_license_plates_l1303_130321


namespace NUMINAMATH_CALUDE_sum_of_cubes_roots_l1303_130374

/-- For a quadratic equation x^2 + ax + a + 1 = 0, the sum of cubes of its roots equals 1 iff a = -1 -/
theorem sum_of_cubes_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + a*x₁ + a + 1 = 0 ∧ x₂^2 + a*x₂ + a + 1 = 0 ∧ x₁^3 + x₂^3 = 1) ↔ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_roots_l1303_130374


namespace NUMINAMATH_CALUDE_fraction_sum_equals_negative_one_l1303_130328

theorem fraction_sum_equals_negative_one (a : ℝ) (h : 1 - 2*a ≠ 0) :
  a / (1 - 2*a) + (a - 1) / (1 - 2*a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_negative_one_l1303_130328


namespace NUMINAMATH_CALUDE_constant_in_toll_formula_l1303_130368

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ :=
  1.50 + constant * (x - 2)

/-- The number of axles on an 18-wheel truck -/
def axles_18_wheel_truck : ℕ := 9

/-- The toll for an 18-wheel truck -/
def toll_18_wheel_truck : ℝ := 5

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll_formula constant axles_18_wheel_truck = toll_18_wheel_truck ∧ 
    constant = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_constant_in_toll_formula_l1303_130368


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l1303_130367

theorem polygon_angle_sum (n : ℕ) : 
  (n ≥ 3) →
  (180 * (n - 2) = 3 * 360) →
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l1303_130367


namespace NUMINAMATH_CALUDE_cupcake_cost_proof_l1303_130361

theorem cupcake_cost_proof (total_cupcakes : ℕ) (people : ℕ) (cost_per_person : ℚ) :
  total_cupcakes = 12 →
  people = 2 →
  cost_per_person = 9 →
  (people * cost_per_person) / total_cupcakes = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_cost_proof_l1303_130361


namespace NUMINAMATH_CALUDE_parabola_focus_and_directrix_l1303_130383

/-- Given a parabola with equation y² = 8x, prove its focus coordinates and directrix equation -/
theorem parabola_focus_and_directrix :
  ∀ (x y : ℝ), y^2 = 8*x →
  (∃ (focus_x focus_y : ℝ), focus_x = 2 ∧ focus_y = 0) ∧
  (∃ (k : ℝ), k = -2 ∧ ∀ (x : ℝ), x = k → x ∈ {x | x = -2}) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_and_directrix_l1303_130383


namespace NUMINAMATH_CALUDE_square_root_of_one_incorrect_l1303_130317

theorem square_root_of_one_incorrect : ¬(∀ x : ℝ, x^2 = 1 → x = 1) := by
  sorry

#check square_root_of_one_incorrect

end NUMINAMATH_CALUDE_square_root_of_one_incorrect_l1303_130317


namespace NUMINAMATH_CALUDE_num_boys_in_first_group_is_16_l1303_130398

/-- The number of boys in the first group that, together with 12 men,
    can complete a piece of work in 5 days, given that:
    - 13 men and 24 boys can do the same work in 4 days
    - The ratio of daily work done by a man to that of a boy is 2 -/
def num_boys_in_first_group : ℕ := by
  sorry

theorem num_boys_in_first_group_is_16 :
  num_boys_in_first_group = 16 := by
  sorry

end NUMINAMATH_CALUDE_num_boys_in_first_group_is_16_l1303_130398


namespace NUMINAMATH_CALUDE_annie_travel_distance_l1303_130336

/-- The number of blocks Annie walked from her house to the bus stop -/
def blocks_to_bus_stop : ℕ := 5

/-- The number of blocks Annie rode the bus to the coffee shop -/
def blocks_on_bus : ℕ := 7

/-- The total number of blocks Annie traveled in her round trip -/
def total_blocks : ℕ := 2 * (blocks_to_bus_stop + blocks_on_bus)

theorem annie_travel_distance : total_blocks = 24 := by sorry

end NUMINAMATH_CALUDE_annie_travel_distance_l1303_130336


namespace NUMINAMATH_CALUDE_remainder_1997_pow_2000_mod_7_l1303_130353

theorem remainder_1997_pow_2000_mod_7 : 1997^2000 % 7 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_1997_pow_2000_mod_7_l1303_130353


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1303_130394

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 233 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1303_130394


namespace NUMINAMATH_CALUDE_hex_to_decimal_equality_l1303_130312

/-- Represents a hexadecimal digit as a natural number -/
def HexDigit := Fin 16

/-- Converts a base-6 number to decimal -/
def toDecimal (a b c d e : Fin 6) : ℕ :=
  e + 6 * d + 6^2 * c + 6^3 * b + 6^4 * a

/-- The theorem stating that if 3m502₍₆₎ = 4934 in decimal, then m = 4 -/
theorem hex_to_decimal_equality (m : Fin 6) :
  toDecimal 3 m 5 0 2 = 4934 → m = 4 := by
  sorry


end NUMINAMATH_CALUDE_hex_to_decimal_equality_l1303_130312


namespace NUMINAMATH_CALUDE_sabrina_leaves_l1303_130370

/-- The number of basil leaves Sabrina needs -/
def basil : ℕ := 12

/-- The number of sage leaves Sabrina needs -/
def sage : ℕ := basil / 2

/-- The number of verbena leaves Sabrina needs -/
def verbena : ℕ := sage + 5

/-- The total number of leaves Sabrina needs -/
def total : ℕ := basil + sage + verbena

theorem sabrina_leaves : total = 29 := by
  sorry

end NUMINAMATH_CALUDE_sabrina_leaves_l1303_130370


namespace NUMINAMATH_CALUDE_function_increasing_implies_a_leq_neg_two_l1303_130339

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

-- State the theorem
theorem function_increasing_implies_a_leq_neg_two :
  ∀ a : ℝ, (∀ x y : ℝ, -2 < x ∧ x < y ∧ y < 2 → f a x < f a y) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_function_increasing_implies_a_leq_neg_two_l1303_130339


namespace NUMINAMATH_CALUDE_range_of_a_l1303_130323

/-- The set of x satisfying p -/
def set_p (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- The set of x satisfying q -/
def set_q : Set ℝ := {x | x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0}

/-- The theorem stating the range of values for a -/
theorem range_of_a (a : ℝ) (h1 : a < 0) 
  (h2 : set_p a ⊆ set_q)
  (h3 : (Set.univ \ set_p a) ⊂ (Set.univ \ set_q)) :
  -4 ≤ a ∧ a < 0 ∨ a ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1303_130323


namespace NUMINAMATH_CALUDE_peaches_at_stand_l1303_130331

/-- The total number of peaches at Sally's stand after picking more -/
def total_peaches (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem stating that the total number of peaches is 55 given the initial and picked amounts -/
theorem peaches_at_stand (initial : ℕ) (picked : ℕ) 
  (h1 : initial = 13) (h2 : picked = 42) : 
  total_peaches initial picked = 55 := by
  sorry

end NUMINAMATH_CALUDE_peaches_at_stand_l1303_130331


namespace NUMINAMATH_CALUDE_other_man_age_is_ten_l1303_130335

/-- The age of the other replaced man given the conditions of the problem -/
def other_man_age (initial_men : ℕ) (replaced_men : ℕ) (age_increase : ℕ) 
  (known_man_age : ℕ) (women_avg_age : ℕ) : ℕ :=
  26 - age_increase

/-- Theorem stating the age of the other replaced man -/
theorem other_man_age_is_ten 
  (initial_men : ℕ) 
  (replaced_men : ℕ) 
  (age_increase : ℕ) 
  (known_man_age : ℕ) 
  (women_avg_age : ℕ) 
  (h1 : initial_men = 8)
  (h2 : replaced_men = 2)
  (h3 : age_increase = 2)
  (h4 : known_man_age = 20)
  (h5 : women_avg_age = 23) :
  other_man_age initial_men replaced_men age_increase known_man_age women_avg_age = 10 := by
  sorry


end NUMINAMATH_CALUDE_other_man_age_is_ten_l1303_130335


namespace NUMINAMATH_CALUDE_john_initial_diamonds_l1303_130372

/-- Represents the number of diamonds each pirate has -/
structure DiamondCount where
  bill : ℕ
  sam : ℕ
  john : ℕ

/-- Represents the average mass of diamonds for each pirate -/
structure AverageMass where
  bill : ℝ
  sam : ℝ
  john : ℝ

/-- The theft operation -/
def theft (d : DiamondCount) : DiamondCount :=
  { bill := d.bill,
    sam := d.sam,
    john := d.john }

/-- The change in average mass after the theft -/
def averageMassChange (initial : AverageMass) (final : AverageMass) : AverageMass :=
  { bill := initial.bill - final.bill,
    sam := initial.sam - final.sam,
    john := final.john - initial.john }

theorem john_initial_diamonds : 
  ∀ (initial : DiamondCount) (initial_avg : AverageMass) (final_avg : AverageMass),
    initial.bill = 12 →
    initial.sam = 12 →
    (averageMassChange initial_avg final_avg).bill = 1 →
    (averageMassChange initial_avg final_avg).sam = 2 →
    (averageMassChange initial_avg final_avg).john = 4 →
    initial.john = 9 := by
  sorry

end NUMINAMATH_CALUDE_john_initial_diamonds_l1303_130372


namespace NUMINAMATH_CALUDE_engineering_exam_pass_percentage_l1303_130343

theorem engineering_exam_pass_percentage
  (total_male : ℕ)
  (total_female : ℕ)
  (male_eng_percent : ℚ)
  (female_eng_percent : ℚ)
  (male_pass_percent : ℚ)
  (female_pass_percent : ℚ)
  (h1 : total_male = 120)
  (h2 : total_female = 100)
  (h3 : male_eng_percent = 25 / 100)
  (h4 : female_eng_percent = 20 / 100)
  (h5 : male_pass_percent = 20 / 100)
  (h6 : female_pass_percent = 25 / 100)
  : (↑(Nat.floor ((male_eng_percent * male_pass_percent * total_male + female_eng_percent * female_pass_percent * total_female) / (male_eng_percent * total_male + female_eng_percent * total_female) * 100)) : ℚ) = 22 := by
  sorry

end NUMINAMATH_CALUDE_engineering_exam_pass_percentage_l1303_130343


namespace NUMINAMATH_CALUDE_complex_modulus_sum_l1303_130300

theorem complex_modulus_sum : Complex.abs (3 - 5*Complex.I) + Complex.abs (3 + 7*Complex.I) = Real.sqrt 34 + Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sum_l1303_130300


namespace NUMINAMATH_CALUDE_inequality_proof_l1303_130326

theorem inequality_proof (t : ℝ) (h : t > 0) : (1 + 2/t) * Real.log (1 + t) > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1303_130326


namespace NUMINAMATH_CALUDE_min_disks_needed_prove_min_disks_l1303_130301

/-- Represents the storage capacity of a disk in MB -/
def disk_capacity : ℚ := 2

/-- Represents the total number of files -/
def total_files : ℕ := 36

/-- Represents the number of 1.2 MB files -/
def large_files : ℕ := 5

/-- Represents the number of 0.6 MB files -/
def medium_files : ℕ := 16

/-- Represents the size of large files in MB -/
def large_file_size : ℚ := 1.2

/-- Represents the size of medium files in MB -/
def medium_file_size : ℚ := 0.6

/-- Represents the size of small files in MB -/
def small_file_size : ℚ := 0.2

/-- Calculates the number of small files -/
def small_files : ℕ := total_files - large_files - medium_files

/-- Theorem stating the minimum number of disks needed -/
theorem min_disks_needed : ℕ := 14

/-- Proof of the minimum number of disks needed -/
theorem prove_min_disks : min_disks_needed = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_disks_needed_prove_min_disks_l1303_130301


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1303_130304

theorem sin_cos_identity (x y : ℝ) :
  Real.sin (x - y) * Real.cos y + Real.cos (x - y) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1303_130304


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l1303_130306

/-- The center of a circle given by the equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + y + a = 0

/-- The circle equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- Theorem: If the line 3x + y + a = 0 passes through the center of the circle
    x^2 + y^2 + 2x - 4y = 0, then a = 1 -/
theorem line_passes_through_circle_center (a : ℝ) :
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l1303_130306


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_all_zero_l1303_130320

theorem square_sum_zero_implies_all_zero (a b c : ℝ) : 
  a^2 + b^2 + c^2 = 0 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_all_zero_l1303_130320


namespace NUMINAMATH_CALUDE_weight_ratio_proof_l1303_130384

/-- Prove the ratio of weight added back to initial weight lost --/
theorem weight_ratio_proof (initial_weight final_weight : ℕ) 
  (first_loss third_loss final_gain : ℕ) (weight_added : ℕ) : 
  initial_weight = 99 →
  final_weight = 81 →
  first_loss = 12 →
  third_loss = 3 * first_loss →
  final_gain = 6 →
  initial_weight - first_loss + weight_added - third_loss + final_gain = final_weight →
  weight_added / first_loss = 2 := by
  sorry

end NUMINAMATH_CALUDE_weight_ratio_proof_l1303_130384


namespace NUMINAMATH_CALUDE_product_repeating_third_and_nine_l1303_130350

/-- The repeating decimal 0.3̄ -/
def repeating_third : ℚ := 1/3

/-- Theorem stating that the product of 0.3̄ and 9 is 3 -/
theorem product_repeating_third_and_nine :
  repeating_third * 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_third_and_nine_l1303_130350


namespace NUMINAMATH_CALUDE_post_height_l1303_130327

/-- The height of a cylindrical post given a squirrel's spiral path. -/
theorem post_height (circuit_rise : ℝ) (post_circumference : ℝ) (total_distance : ℝ) : 
  circuit_rise = 4 →
  post_circumference = 3 →
  total_distance = 9 →
  (total_distance / post_circumference) * circuit_rise = 12 :=
by sorry

end NUMINAMATH_CALUDE_post_height_l1303_130327


namespace NUMINAMATH_CALUDE_range_of_f_inequality_l1303_130382

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem range_of_f_inequality 
  (hdom : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → f x ≠ 0 → True)
  (hderiv : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → HasDerivAt f (x^2 + 2 * Real.cos x) x)
  (hf0 : f 0 = 0) :
  {x : ℝ | f (1 + x) + f (x - x^2) > 0} = Set.Ioo (1 - Real.sqrt 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_inequality_l1303_130382


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1303_130311

theorem quadratic_roots_property (α β : ℝ) : 
  (α^2 + α - 1 = 0) → 
  (β^2 + β - 1 = 0) → 
  (α ≠ β) →
  α^4 - 3*β = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1303_130311


namespace NUMINAMATH_CALUDE_orchard_sections_l1303_130369

/-- The number of sections in an apple orchard --/
def number_of_sections (daily_harvest_per_section : ℕ) (total_daily_harvest : ℕ) : ℕ :=
  total_daily_harvest / daily_harvest_per_section

/-- Theorem stating that the number of sections in the orchard is 8 --/
theorem orchard_sections :
  let daily_harvest_per_section := 45
  let total_daily_harvest := 360
  number_of_sections daily_harvest_per_section total_daily_harvest = 8 := by
  sorry

end NUMINAMATH_CALUDE_orchard_sections_l1303_130369


namespace NUMINAMATH_CALUDE_absolute_value_of_z_l1303_130390

theorem absolute_value_of_z (z z₀ : ℂ) : 
  z₀ = 3 + Complex.I ∧ z * z₀ = 3 * z + z₀ → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_l1303_130390


namespace NUMINAMATH_CALUDE_max_value_of_a_l1303_130386

theorem max_value_of_a : 
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → 
  ∃ a_max : ℝ, a_max = 1 ∧ ∀ a : ℝ, (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ a_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1303_130386


namespace NUMINAMATH_CALUDE_max_product_of_three_l1303_130346

def S : Finset Int := {-9, -7, -3, 1, 4, 6}

theorem max_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  a * b * c ≤ 378 ∧ ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 378 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_three_l1303_130346


namespace NUMINAMATH_CALUDE_third_square_is_G_l1303_130397

/-- Represents a 2x2 square on the table -/
structure Square :=
  (name : Char)
  (position : Nat)

/-- Represents the state of the table -/
structure Table :=
  (squares : List Square)
  (size : Nat)

/-- Checks if a square is fully visible -/
def isFullyVisible (s : Square) (t : Table) : Prop :=
  s.position = t.squares.length

/-- Checks if a square is partially visible -/
def isPartiallyVisible (s : Square) (t : Table) : Prop :=
  s.position < t.squares.length ∧ s.position ≠ 0

/-- The theorem to be proved -/
theorem third_square_is_G 
  (t : Table)
  (h1 : t.size = 4)
  (h2 : t.squares.length = 8)
  (h3 : ∃ e : Square, e.name = 'E' ∧ isFullyVisible e t)
  (h4 : ∀ s : Square, s ∈ t.squares → s.name ≠ 'E' → isPartiallyVisible s t) :
  ∃ g : Square, g.name = 'G' ∧ g.position = 3 :=
sorry

end NUMINAMATH_CALUDE_third_square_is_G_l1303_130397


namespace NUMINAMATH_CALUDE_polygon_distance_inequality_l1303_130391

-- Define a polygon type
structure Polygon :=
  (vertices : List (ℝ × ℝ))

-- Define the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := sorry

-- Define the sum of distances from a point to vertices
def sum_distances_to_vertices (o : ℝ × ℝ) (p : Polygon) : ℝ := sorry

-- Define the sum of distances from a point to sides
def sum_distances_to_sides (o : ℝ × ℝ) (p : Polygon) : ℝ := sorry

-- State the theorem
theorem polygon_distance_inequality (o : ℝ × ℝ) (m : Polygon) :
  let ρ := perimeter m
  let d := sum_distances_to_vertices o m
  let h := sum_distances_to_sides o m
  d^2 - h^2 ≥ ρ^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_distance_inequality_l1303_130391


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1303_130325

def has_integral_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℤ, a * x^2 + b * x + c = 0

theorem smallest_m_for_integral_solutions :
  (∀ m : ℤ, m > 0 ∧ m < 170 → ¬ has_integral_solutions 10 (-m) 720) ∧
  has_integral_solutions 10 (-170) 720 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1303_130325


namespace NUMINAMATH_CALUDE_pen_profit_percentage_l1303_130381

/-- Calculates the profit percentage for a retailer selling pens -/
theorem pen_profit_percentage 
  (num_pens : ℕ) 
  (cost_pens : ℕ) 
  (discount_percent : ℚ) : 
  num_pens = 60 → 
  cost_pens = 36 → 
  discount_percent = 1/100 →
  (((num_pens : ℚ) * (1 - discount_percent) - cost_pens) / cost_pens) * 100 = 65 := by
  sorry

#check pen_profit_percentage

end NUMINAMATH_CALUDE_pen_profit_percentage_l1303_130381


namespace NUMINAMATH_CALUDE_triangle_max_area_l1303_130392

/-- Given a triangle ABC inscribed in a circle of radius R, with sides a, b, c opposite to angles A, B, C respectively, 
    satisfying the condition 2R(sin²A - sin²C) = (√2*a - b)*sin B, 
    the maximum area of the triangle is (√2 + 1)/2 * R², achieved when a = b. -/
theorem triangle_max_area (R : ℝ) (a b c : ℝ) (A B C : ℝ) :
  (0 < R) →
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a = 2 * R * Real.sin (A / 2)) →
  (b = 2 * R * Real.sin (B / 2)) →
  (c = 2 * R * Real.sin (C / 2)) →
  (2 * R * (Real.sin A)^2 - 2 * R * (Real.sin C)^2 = (Real.sqrt 2 * a - b) * Real.sin B) →
  (∀ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' →
    a' = 2 * R * Real.sin (A / 2) →
    b' = 2 * R * Real.sin (B / 2) →
    c' = 2 * R * Real.sin (C / 2) →
    a' * b' * Real.sin C / 4 ≤ (Real.sqrt 2 + 1) / 2 * R^2) ∧
  (a * b * Real.sin C / 4 = (Real.sqrt 2 + 1) / 2 * R^2) ∧
  (a = b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1303_130392


namespace NUMINAMATH_CALUDE_fraction_simplification_l1303_130305

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : y - 1/x ≠ 0) : 
  (x - 1/y) / (y - 1/x) = x / y := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1303_130305


namespace NUMINAMATH_CALUDE_modulo_problem_l1303_130348

theorem modulo_problem (n : ℕ) : 
  (215 * 789) % 75 = n ∧ 0 ≤ n ∧ n < 75 → n = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_modulo_problem_l1303_130348


namespace NUMINAMATH_CALUDE_power_mod_prime_remainder_2_100_mod_101_l1303_130338

theorem power_mod_prime (p : Nat) (a : Nat) (h_prime : Nat.Prime p) (h_not_div : ¬(p ∣ a)) :
  a^(p - 1) ≡ 1 [MOD p] := by sorry

theorem remainder_2_100_mod_101 :
  2^100 ≡ 1 [MOD 101] := by
  have h_prime : Nat.Prime 101 := sorry
  have h_not_div : ¬(101 ∣ 2) := sorry
  have h_fermat := power_mod_prime 101 2 h_prime h_not_div
  sorry

end NUMINAMATH_CALUDE_power_mod_prime_remainder_2_100_mod_101_l1303_130338


namespace NUMINAMATH_CALUDE_vector_triangle_completion_l1303_130373

/-- Given vectors a and b, prove that vector c completes the triangle formed by 4a and 3b - 2a -/
theorem vector_triangle_completion (a b : ℝ × ℝ) (h : a = (1, -3) ∧ b = (-2, 4)) :
  let c : ℝ × ℝ := (4, -6)
  (4 • a) + (3 • b - 2 • a) + c = (0, 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_triangle_completion_l1303_130373


namespace NUMINAMATH_CALUDE_geometry_class_eligibility_l1303_130395

def minimum_score (s1 s2 s3 s4 : ℝ) : ℝ :=
  let required_average := 85
  let total_required := 5 * required_average
  let current_sum := s1 + s2 + s3 + s4
  total_required - current_sum

theorem geometry_class_eligibility 
  (s1 s2 s3 s4 : ℝ) 
  (h1 : s1 = 86) 
  (h2 : s2 = 82) 
  (h3 : s3 = 80) 
  (h4 : s4 = 84) : 
  minimum_score s1 s2 s3 s4 = 93 := by
  sorry

#eval minimum_score 86 82 80 84

end NUMINAMATH_CALUDE_geometry_class_eligibility_l1303_130395


namespace NUMINAMATH_CALUDE_math_class_size_l1303_130388

/-- Proves that the number of students in the mathematics class is 170/3 given the conditions of the problem. -/
theorem math_class_size (total : ℕ) (both : ℕ) (math_twice_physics : Prop) :
  total = 75 →
  both = 10 →
  math_twice_physics →
  (∃ (math physics : ℕ),
    math = (170 : ℚ) / 3 ∧
    physics = (total - both) - (math - both) ∧
    math = 2 * physics) :=
by sorry

end NUMINAMATH_CALUDE_math_class_size_l1303_130388


namespace NUMINAMATH_CALUDE_correct_balanced_redox_reaction_l1303_130363

/-- Represents a chemical species in a redox reaction -/
structure ChemicalSpecies where
  formula : String
  charge : Int

/-- Represents a half-reaction in a redox reaction -/
structure HalfReaction where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies
  electrons : Int

/-- Represents a complete redox reaction -/
structure RedoxReaction where
  oxidation : HalfReaction
  reduction : HalfReaction

/-- Standard conditions in an acidic solution -/
def standardAcidicConditions : Prop := sorry

/-- Salicylic acid -/
def salicylicAcid : ChemicalSpecies := ⟨"C7H6O2", 0⟩

/-- Iron (III) ion -/
def ironIII : ChemicalSpecies := ⟨"Fe", 3⟩

/-- 2,3-dihydroxybenzoic acid -/
def dihydroxybenzoicAcid : ChemicalSpecies := ⟨"C7H6O4", 0⟩

/-- Hydrogen ion -/
def hydrogenIon : ChemicalSpecies := ⟨"H", 1⟩

/-- Iron (II) ion -/
def ironII : ChemicalSpecies := ⟨"Fe", 2⟩

/-- The balanced redox reaction between iron (III) nitrate and salicylic acid under standard acidic conditions -/
def balancedRedoxReaction (conditions : Prop) : RedoxReaction := sorry

/-- Theorem stating that the given redox reaction is the correct balanced reaction under standard acidic conditions -/
theorem correct_balanced_redox_reaction :
  standardAcidicConditions →
  balancedRedoxReaction standardAcidicConditions =
    RedoxReaction.mk
      (HalfReaction.mk [salicylicAcid] [dihydroxybenzoicAcid, hydrogenIon, hydrogenIon] 2)
      (HalfReaction.mk [ironIII, ironIII] [ironII, ironII] (-2)) :=
sorry

end NUMINAMATH_CALUDE_correct_balanced_redox_reaction_l1303_130363


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_8_l1303_130313

theorem circle_area_with_diameter_8 (π : ℝ) :
  let diameter : ℝ := 8
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 16 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_8_l1303_130313


namespace NUMINAMATH_CALUDE_bill_proof_l1303_130393

/-- The number of friends who can pay -/
def paying_friends : ℕ := 9

/-- The number of friends including the one who can't pay -/
def total_friends : ℕ := 10

/-- The additional amount each paying friend contributes -/
def additional_amount : ℕ := 3

/-- The total bill amount -/
def total_bill : ℕ := 270

theorem bill_proof :
  (paying_friends : ℚ) * (total_bill / total_friends + additional_amount) = total_bill := by
  sorry

end NUMINAMATH_CALUDE_bill_proof_l1303_130393


namespace NUMINAMATH_CALUDE_gcd_n_power_7_minus_n_l1303_130399

theorem gcd_n_power_7_minus_n (n : ℤ) : 42 ∣ (n^7 - n) := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_power_7_minus_n_l1303_130399


namespace NUMINAMATH_CALUDE_count_zeros_100_to_50_l1303_130347

/-- The number of zeros following the numeral one in the expanded form of 100^50 -/
def zeros_after_one_in_100_to_50 : ℕ := 100

/-- Theorem stating that the number of zeros following the numeral one
    in the expanded form of 100^50 is equal to 100 -/
theorem count_zeros_100_to_50 :
  zeros_after_one_in_100_to_50 = 100 := by sorry

end NUMINAMATH_CALUDE_count_zeros_100_to_50_l1303_130347
