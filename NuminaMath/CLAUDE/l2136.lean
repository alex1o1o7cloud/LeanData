import Mathlib

namespace NUMINAMATH_CALUDE_exponent_equality_l2136_213688

theorem exponent_equality (x : ℕ) : 
  2010^2011 - 2010^2009 = 2010^x * 2009 * 2011 → x = 2009 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2136_213688


namespace NUMINAMATH_CALUDE_inequality_proof_l2136_213639

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (x^2 + y^2)) + (1 / x^2) + (1 / y^2) ≥ 10 / ((x + y)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2136_213639


namespace NUMINAMATH_CALUDE_present_age_of_b_l2136_213656

theorem present_age_of_b (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10))  -- In 10 years, A will be twice as old as B was 10 years ago
  (h2 : a = b + 9)              -- A is now 9 years older than B
  : b = 39 := by               -- The present age of B is 39 years
  sorry

end NUMINAMATH_CALUDE_present_age_of_b_l2136_213656


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_equals_five_l2136_213658

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_equals_five (k : ℕ) :
  k > 0 ∧ sum_arithmetic_sequence (-3) 2 k = 5 → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_equals_five_l2136_213658


namespace NUMINAMATH_CALUDE_equivalence_condition_l2136_213697

theorem equivalence_condition (x y : ℝ) (h : x * y ≠ 0) :
  (x + y = 0) ↔ (y / x + x / y = -2) := by
  sorry

end NUMINAMATH_CALUDE_equivalence_condition_l2136_213697


namespace NUMINAMATH_CALUDE_jim_journey_l2136_213637

theorem jim_journey (total_journey : ℕ) (remaining_miles : ℕ) 
  (h1 : total_journey = 1200)
  (h2 : remaining_miles = 558) :
  total_journey - remaining_miles = 642 := by
sorry

end NUMINAMATH_CALUDE_jim_journey_l2136_213637


namespace NUMINAMATH_CALUDE_least_number_of_marbles_eight_forty_satisfies_least_number_is_eight_forty_l2136_213602

theorem least_number_of_marbles (n : ℕ) : n > 0 ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n → n ≥ 840 := by
  sorry

theorem eight_forty_satisfies (n : ℕ) : 
  3 ∣ 840 ∧ 4 ∣ 840 ∧ 5 ∣ 840 ∧ 7 ∣ 840 ∧ 8 ∣ 840 := by
  sorry

theorem least_number_is_eight_forty : 
  ∃ (n : ℕ), n > 0 ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n ∧
  ∀ (m : ℕ), (m > 0 ∧ 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 8 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_marbles_eight_forty_satisfies_least_number_is_eight_forty_l2136_213602


namespace NUMINAMATH_CALUDE_total_subjects_l2136_213616

/-- Given the number of subjects taken by Monica, prove the total number of subjects taken by all four students. -/
theorem total_subjects (monica : ℕ) (h1 : monica = 10) : ∃ (marius millie michael : ℕ),
  marius = monica + 4 ∧
  millie = marius + 3 ∧
  michael = 2 * millie ∧
  monica + marius + millie + michael = 75 :=
by sorry

end NUMINAMATH_CALUDE_total_subjects_l2136_213616


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2136_213609

theorem complete_square_quadratic (x : ℝ) : 
  (∃ c d : ℝ, x^2 + 6*x - 5 = 0 ↔ (x + c)^2 = d) → 
  (∃ c : ℝ, (x + c)^2 = 14) := by
sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2136_213609


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2136_213683

-- Problem 1
theorem problem_1 (x : ℝ) : 
  (4 / (x^2 - 1) - 1 = (1 - x) / (x + 1)) ↔ x = 5/2 :=
sorry

-- Problem 2
theorem problem_2 : 
  ¬∃ (x : ℝ), (2 / (x - 3) + 2 = (1 - x) / (3 - x)) :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2136_213683


namespace NUMINAMATH_CALUDE_simplify_expression_l2136_213621

theorem simplify_expression (x : ℝ) (h : -1 < x ∧ x < 3) :
  Real.sqrt ((x - 3)^2) + |x + 1| = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2136_213621


namespace NUMINAMATH_CALUDE_quadratic_increasing_l2136_213668

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_increasing (a b c : ℝ) 
  (h1 : f a b c 0 = f a b c 6) 
  (h2 : f a b c 6 < f a b c 7) :
  ∀ x y, 3 < x → x < y → f a b c x < f a b c y := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_l2136_213668


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2136_213614

/-- Given a complex number z defined in terms of a real number m, 
    prove that when z is a pure imaginary number, m = -3 -/
theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 + m - 6)/m + (m^2 - 2*m)*I
  (z.re = 0 ∧ z.im ≠ 0) → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2136_213614


namespace NUMINAMATH_CALUDE_domino_tiling_theorem_l2136_213671

/-- Represents a rectangle tiled with dominoes -/
structure DominoRectangle where
  width : ℕ
  height : ℕ
  dominoes : ℕ

/-- Condition that any grid line intersects a multiple of four dominoes -/
def grid_line_condition (r : DominoRectangle) : Prop :=
  ∀ (line : ℕ), line ≤ r.width ∨ line ≤ r.height → 
    (if line ≤ r.width then r.height else r.width) % 4 = 0

/-- Main theorem: If the grid line condition holds, then one side is divisible by 4 -/
theorem domino_tiling_theorem (r : DominoRectangle) 
  (h : grid_line_condition r) : 
  r.width % 4 = 0 ∨ r.height % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_domino_tiling_theorem_l2136_213671


namespace NUMINAMATH_CALUDE_video_library_disk_space_l2136_213607

/-- Calculates the average disk space per hour of video in a library, rounded to the nearest integer -/
def averageDiskSpacePerHour (totalDays : ℕ) (totalSpace : ℕ) : ℕ :=
  let totalHours : ℕ := totalDays * 24
  let exactAverage : ℚ := totalSpace / totalHours
  (exactAverage + 1/2).floor.toNat

/-- Theorem stating that for a 15-day video library occupying 24000 MB, 
    the average disk space per hour rounded to the nearest integer is 67 MB -/
theorem video_library_disk_space :
  averageDiskSpacePerHour 15 24000 = 67 := by
  sorry

end NUMINAMATH_CALUDE_video_library_disk_space_l2136_213607


namespace NUMINAMATH_CALUDE_mn_inequality_characterization_l2136_213686

theorem mn_inequality_characterization :
  ∀ m n : ℕ+, 
    (1 ≤ m^n.val - n^m.val ∧ m^n.val - n^m.val ≤ m.val * n.val) ↔ 
    ((m ≥ 2 ∧ n = 1) ∨ (m = 2 ∧ n = 5) ∨ (m = 3 ∧ n = 2)) := by
  sorry

end NUMINAMATH_CALUDE_mn_inequality_characterization_l2136_213686


namespace NUMINAMATH_CALUDE_weight_loss_difference_l2136_213689

/-- Given the weight loss of three people, prove how much more Veronica lost compared to Seth. -/
theorem weight_loss_difference (seth_loss jerome_loss veronica_loss total_loss : ℝ) : 
  seth_loss = 17.5 →
  jerome_loss = 3 * seth_loss →
  total_loss = 89 →
  total_loss = seth_loss + jerome_loss + veronica_loss →
  veronica_loss > seth_loss →
  veronica_loss - seth_loss = 1.5 := by
  sorry

#check weight_loss_difference

end NUMINAMATH_CALUDE_weight_loss_difference_l2136_213689


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l2136_213623

-- Define the conditions p and q
def p (a : ℝ) : Prop := a < 0
def q (a : ℝ) : Prop := a^2 > a

-- Theorem statement
theorem not_p_necessary_not_sufficient_for_not_q :
  (∀ a, ¬(q a) → ¬(p a)) ∧ 
  (∃ a, ¬(p a) ∧ q a) :=
by sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l2136_213623


namespace NUMINAMATH_CALUDE_point_in_intersection_l2136_213698

def U : Set (ℝ × ℝ) := Set.univ

def A (m : ℝ) : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + m > 0}

def B (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n ≤ 0}

theorem point_in_intersection (m n : ℝ) :
  (2, 3) ∈ A m ∩ (U \ B n) ↔ m > -1 ∧ n < 5 := by
  sorry

end NUMINAMATH_CALUDE_point_in_intersection_l2136_213698


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l2136_213665

/-- The sequence G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of G_1000 is 2 -/
theorem units_digit_G_1000 : units_digit (G 1000) = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l2136_213665


namespace NUMINAMATH_CALUDE_unique_number_with_appended_digits_sum_l2136_213601

theorem unique_number_with_appended_digits_sum (A : ℕ) : 
  (∃ B : ℕ, B ≤ 999 ∧ 1000 * A + B = A * (A + 1) / 2) ↔ A = 1999 :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_appended_digits_sum_l2136_213601


namespace NUMINAMATH_CALUDE_negation_of_existence_l2136_213669

theorem negation_of_existence (p : Prop) :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2136_213669


namespace NUMINAMATH_CALUDE_board_numbers_theorem_l2136_213650

def pairwise_sums : List ℕ := [5, 8, 9, 13, 14, 14, 15, 17, 18, 23]

def is_valid_set (s : List ℕ) : Prop :=
  s.length = 5 ∧
  (List.map (λ (x, y) => x + y) (s.product s)).filter (λ x => x ∉ s) = pairwise_sums

theorem board_numbers_theorem :
  ∃ (s : List ℕ), is_valid_set s ∧ s.prod = 4752 := by sorry

end NUMINAMATH_CALUDE_board_numbers_theorem_l2136_213650


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l2136_213600

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 → b > 0 → -- Angles are positive
  a + b + 90 = 180 → -- Sum of angles in a triangle
  a / b = 7 / 2 → -- Ratio of acute angles
  (a = 70 ∧ b = 20) ∨ (a = 20 ∧ b = 70) := by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l2136_213600


namespace NUMINAMATH_CALUDE_right_pyramid_base_side_length_l2136_213603

/-- Given a right pyramid with a square base, if the area of one lateral face is 120 square meters
    and the slant height is 40 meters, then the length of the side of its base is 6 meters. -/
theorem right_pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  (2 * area_lateral_face) / slant_height = 6 :=
by sorry

end NUMINAMATH_CALUDE_right_pyramid_base_side_length_l2136_213603


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2136_213662

/-- Given a geometric sequence {a_n} where a_1 = 2 and a_1 + a_3 + a_5 = 14,
    prove that 1/a_1 + 1/a_3 + 1/a_5 = 7/8 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : a 1 = 2) 
    (h2 : a 1 + a 3 + a 5 = 14) 
    (h3 : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q) :
    1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2136_213662


namespace NUMINAMATH_CALUDE_kellys_snacks_l2136_213641

theorem kellys_snacks (peanuts raisins : ℝ) (h1 : peanuts = 0.1) (h2 : raisins = 0.4) :
  peanuts + raisins = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_kellys_snacks_l2136_213641


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2136_213661

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the concept of symmetry with respect to the origin
def symmetricToOrigin (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

-- Define the fourth quadrant
def inFourthQuadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

-- Theorem statement
theorem point_in_fourth_quadrant (a : ℝ) (P P_1 : Point2D) :
  a < 0 →
  P = Point2D.mk (-a^2 - 1) (-a + 3) →
  symmetricToOrigin P P_1 →
  inFourthQuadrant P_1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2136_213661


namespace NUMINAMATH_CALUDE_integral_proof_l2136_213675

open Real

noncomputable def f (x : ℝ) : ℝ := 3*x + log (abs x) + 2*log (abs (x+1)) - log (abs (x-2))

theorem integral_proof (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) (h3 : x ≠ 2) : 
  deriv f x = (3*x^3 - x^2 - 12*x - 2) / (x*(x+1)*(x-2)) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l2136_213675


namespace NUMINAMATH_CALUDE_order_of_expressions_l2136_213693

theorem order_of_expressions (x : ℝ) :
  let a : ℝ := -x^2 - 2*x
  let b : ℝ := -2*x^2 - 2
  let c : ℝ := Real.sqrt 5 - 1
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l2136_213693


namespace NUMINAMATH_CALUDE_jacket_sale_profit_l2136_213635

/-- Calculates the merchant's gross profit for a jacket sale --/
theorem jacket_sale_profit (purchase_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  purchase_price = 42 ∧ 
  markup_percentage = 0.3 ∧ 
  discount_percentage = 0.2 → 
  let selling_price := purchase_price / (1 - markup_percentage)
  let discounted_price := selling_price * (1 - discount_percentage)
  discounted_price - purchase_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_jacket_sale_profit_l2136_213635


namespace NUMINAMATH_CALUDE_gcf_of_36_and_54_l2136_213672

theorem gcf_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_36_and_54_l2136_213672


namespace NUMINAMATH_CALUDE_original_mean_calculation_l2136_213673

theorem original_mean_calculation (n : ℕ) (decrement : ℝ) (new_mean : ℝ) (h1 : n = 50) (h2 : decrement = 47) (h3 : new_mean = 153) :
  ∃ (original_mean : ℝ), original_mean * n = new_mean * n + decrement * n ∧ original_mean = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_mean_calculation_l2136_213673


namespace NUMINAMATH_CALUDE_lending_period_is_one_year_l2136_213611

/-- Proves that the lending period is 1 year given the problem conditions --/
theorem lending_period_is_one_year 
  (principal : ℝ)
  (borrowing_rate : ℝ)
  (lending_rate : ℝ)
  (annual_gain : ℝ)
  (h1 : principal = 5000)
  (h2 : borrowing_rate = 0.04)
  (h3 : lending_rate = 0.05)
  (h4 : annual_gain = 50)
  : ∃ t : ℝ, t = 1 ∧ principal * lending_rate * t - principal * borrowing_rate * t = annual_gain :=
sorry

end NUMINAMATH_CALUDE_lending_period_is_one_year_l2136_213611


namespace NUMINAMATH_CALUDE_cake_muffin_buyers_l2136_213613

theorem cake_muffin_buyers (cake_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ) 
  (prob_neither : ℚ) (h1 : cake_buyers = 50) (h2 : muffin_buyers = 40) 
  (h3 : both_buyers = 19) (h4 : prob_neither = 29/100) : 
  ∃ total_buyers : ℕ, 
    total_buyers = 100 ∧ 
    (cake_buyers + muffin_buyers - both_buyers : ℚ) + prob_neither * total_buyers = total_buyers :=
by sorry

end NUMINAMATH_CALUDE_cake_muffin_buyers_l2136_213613


namespace NUMINAMATH_CALUDE_unique_digit_B_l2136_213664

-- Define the number as a function of B
def number (B : Nat) : Nat := 58709310 + B

-- Theorem statement
theorem unique_digit_B :
  ∀ B : Nat,
  B < 10 →
  (number B) % 2 = 0 →
  (number B) % 3 = 0 →
  (number B) % 4 = 0 →
  (number B) % 5 = 0 →
  (number B) % 6 = 0 →
  (number B) % 10 = 0 →
  B = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_B_l2136_213664


namespace NUMINAMATH_CALUDE_largest_reciprocal_l2136_213606

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/8 → b = 3/4 → c = 1/2 → d = 10 → e = -2 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l2136_213606


namespace NUMINAMATH_CALUDE_number_equality_l2136_213604

theorem number_equality : ∃ x : ℝ, x / 0.144 = 14.4 / 0.0144 ∧ x = 144 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2136_213604


namespace NUMINAMATH_CALUDE_vertical_shift_theorem_l2136_213622

/-- The original line function -/
def original_line (x : ℝ) : ℝ := 2 * x

/-- The vertical shift amount -/
def shift : ℝ := 5

/-- The resulting line after vertical shift -/
def shifted_line (x : ℝ) : ℝ := original_line x + shift

theorem vertical_shift_theorem :
  ∀ x : ℝ, shifted_line x = 2 * x + 5 := by sorry

end NUMINAMATH_CALUDE_vertical_shift_theorem_l2136_213622


namespace NUMINAMATH_CALUDE_complex_square_simplification_l2136_213608

theorem complex_square_simplification :
  (4 - 3 * Complex.I) ^ 2 = 7 - 24 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l2136_213608


namespace NUMINAMATH_CALUDE_distinct_solutions_condition_l2136_213653

theorem distinct_solutions_condition (a x y : ℝ) : 
  x ≠ y → x = a - y^2 → y = a - x^2 → a > 3/4 := by
  sorry

end NUMINAMATH_CALUDE_distinct_solutions_condition_l2136_213653


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l2136_213682

theorem smallest_whole_number_above_sum : ∃ n : ℕ, 
  (n : ℝ) > 3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/8 - 2 ∧ 
  ∀ m : ℕ, (m : ℝ) > 3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/8 - 2 → m ≥ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l2136_213682


namespace NUMINAMATH_CALUDE_driving_time_proof_l2136_213667

/-- Proves that given the conditions of the driving problem, the driving times for route one and route two are 2 hours and 2.5 hours respectively. -/
theorem driving_time_proof (distance_one : ℝ) (distance_two : ℝ) (time_diff : ℝ) (speed_ratio : ℝ) :
  distance_one = 180 →
  distance_two = 150 →
  time_diff = 0.5 →
  speed_ratio = 1.5 →
  ∃ (time_one time_two : ℝ),
    time_one = 2 ∧
    time_two = 2.5 ∧
    time_two = time_one + time_diff ∧
    distance_one / time_one = speed_ratio * (distance_two / time_two) :=
by sorry


end NUMINAMATH_CALUDE_driving_time_proof_l2136_213667


namespace NUMINAMATH_CALUDE_water_percentage_in_dried_grapes_l2136_213605

/-- 
Given:
- Fresh grapes contain 90% water by weight
- 25 kg of fresh grapes yield 3.125 kg of dried grapes

Prove that the percentage of water in dried grapes is 20%
-/
theorem water_percentage_in_dried_grapes :
  let fresh_grape_weight : ℝ := 25
  let dried_grape_weight : ℝ := 3.125
  let fresh_water_percentage : ℝ := 90
  let dried_water_percentage : ℝ := (dried_grape_weight - (fresh_grape_weight * (1 - fresh_water_percentage / 100))) / dried_grape_weight * 100
  dried_water_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_water_percentage_in_dried_grapes_l2136_213605


namespace NUMINAMATH_CALUDE_log_equation_solution_l2136_213643

theorem log_equation_solution (x : ℝ) :
  Real.log (x + 8) / Real.log 8 = 3/2 → x = 8 * (2 * Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2136_213643


namespace NUMINAMATH_CALUDE_milk_liters_bought_l2136_213618

/-- Given the costs of ingredients and the total cost, prove the number of liters of milk bought. -/
theorem milk_liters_bought (flour_boxes : ℕ) (flour_cost : ℕ) (egg_trays : ℕ) (egg_cost : ℕ)
  (milk_cost : ℕ) (soda_boxes : ℕ) (soda_cost : ℕ) (total_cost : ℕ)
  (h1 : flour_boxes = 3) (h2 : flour_cost = 3) (h3 : egg_trays = 3) (h4 : egg_cost = 10)
  (h5 : milk_cost = 5) (h6 : soda_boxes = 2) (h7 : soda_cost = 3) (h8 : total_cost = 80) :
  (total_cost - (flour_boxes * flour_cost + egg_trays * egg_cost + soda_boxes * soda_cost)) / milk_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_liters_bought_l2136_213618


namespace NUMINAMATH_CALUDE_eustace_age_in_three_years_l2136_213636

/-- Proves that Eustace will be 39 years old in 3 years, given the conditions -/
theorem eustace_age_in_three_years
  (eustace_age : ℕ)
  (milford_age : ℕ)
  (h1 : eustace_age = 2 * milford_age)
  (h2 : milford_age + 3 = 21) :
  eustace_age + 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_eustace_age_in_three_years_l2136_213636


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2136_213620

theorem least_addition_for_divisibility (n m k : ℕ) (h : n + k = m * 29) : 
  ∀ j : ℕ, j < k → ¬(∃ l : ℕ, n + j = l * 29) :=
by
  sorry

#check least_addition_for_divisibility 1056 37 17

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2136_213620


namespace NUMINAMATH_CALUDE_derivative_at_zero_l2136_213659

theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*(deriv f 1)) :
  deriv f 0 = -4 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l2136_213659


namespace NUMINAMATH_CALUDE_possible_values_of_5x_plus_2_l2136_213679

theorem possible_values_of_5x_plus_2 (x : ℝ) : 
  (x - 4) * (5 * x + 2) = 0 → (5 * x + 2 = 0 ∨ 5 * x + 2 = 22) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_5x_plus_2_l2136_213679


namespace NUMINAMATH_CALUDE_no_solution_condition_l2136_213699

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (a * x) / (x - 1) ≠ 1 / (x - 1) + 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l2136_213699


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2136_213629

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define g(x) = f(x) - x^2
def g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x - x^2

-- Theorem statement
theorem quadratic_function_properties
  (a b c : ℝ)
  (origin : f a b c 0 = 0)
  (symmetry : ∀ x, f a b c (1 - x) = f a b c (1 + x))
  (odd_g : ∀ x, g a b c x = -g a b c (-x))
  : f a b c = fun x ↦ x^2 - 2*x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2136_213629


namespace NUMINAMATH_CALUDE_initial_cars_correct_l2136_213631

/-- Represents the car dealership scenario -/
structure CarDealership where
  initialCars : ℕ
  initialSilverPercent : ℚ
  newShipment : ℕ
  newNonSilverPercent : ℚ
  finalSilverPercent : ℚ

/-- The car dealership scenario with given conditions -/
def scenario : CarDealership :=
  { initialCars := 360,  -- This is what we want to prove
    initialSilverPercent := 15 / 100,
    newShipment := 80,
    newNonSilverPercent := 30 / 100,
    finalSilverPercent := 25 / 100 }

/-- Theorem stating that the initial number of cars is correct given the conditions -/
theorem initial_cars_correct (d : CarDealership) : 
  d.initialCars = scenario.initialCars →
  d.initialSilverPercent = scenario.initialSilverPercent →
  d.newShipment = scenario.newShipment →
  d.newNonSilverPercent = scenario.newNonSilverPercent →
  d.finalSilverPercent = scenario.finalSilverPercent →
  d.finalSilverPercent * (d.initialCars + d.newShipment) = 
    d.initialSilverPercent * d.initialCars + (1 - d.newNonSilverPercent) * d.newShipment :=
by sorry

#check initial_cars_correct

end NUMINAMATH_CALUDE_initial_cars_correct_l2136_213631


namespace NUMINAMATH_CALUDE_product_divisibility_probability_l2136_213628

/-- The number of dice rolled -/
def n : ℕ := 8

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The probability that a single die roll is even -/
def p_even : ℚ := 1/2

/-- The probability that a single die roll is divisible by 3 -/
def p_div3 : ℚ := 1/3

/-- The probability that the product of n dice rolls is divisible by both 4 and 3 -/
def prob_div_4_and_3 : ℚ := 1554975/1679616

theorem product_divisibility_probability :
  (1 - (1 - (1 - (1 - p_even)^n - n * (1 - p_even)^(n-1) * p_even))) *
  (1 - (1 - p_div3)^n) = prob_div_4_and_3 := by
  sorry

end NUMINAMATH_CALUDE_product_divisibility_probability_l2136_213628


namespace NUMINAMATH_CALUDE_pyramid_volume_from_rectangle_l2136_213619

/-- The volume of a pyramid formed from a rectangle with specific dimensions -/
theorem pyramid_volume_from_rectangle (AB BC : ℝ) (h : AB = 15 * Real.sqrt 2 ∧ BC = 17 * Real.sqrt 2) :
  let P : ℝ × ℝ × ℝ := (15 * Real.sqrt 2 / 2, 17 * Real.sqrt 2 / 2, Real.sqrt 257)
  let base_area : ℝ := (1 / 2) * AB * BC
  let volume : ℝ := (1 / 3) * base_area * P.2.2
  volume = 85 * Real.sqrt 257 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_from_rectangle_l2136_213619


namespace NUMINAMATH_CALUDE_beneficial_average_recording_l2136_213691

/-- Proves that recording the average of two new test grades is beneficial
    if the average of previous grades is higher than the average of the new grades -/
theorem beneficial_average_recording (n : ℕ) (x y : ℝ) (h : x / n > y / 2) :
  (x + y) / (n + 2) > (x + y / 2) / (n + 1) := by
  sorry

#check beneficial_average_recording

end NUMINAMATH_CALUDE_beneficial_average_recording_l2136_213691


namespace NUMINAMATH_CALUDE_sequence_difference_theorem_l2136_213633

theorem sequence_difference_theorem (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, a (n + 1) ≤ 2 * n) 
  (h3 : ∀ n : ℕ, a n < a (n + 1)) :
  ∀ n : ℕ, ∃ p q : ℕ, a p - a q = n :=
by sorry

end NUMINAMATH_CALUDE_sequence_difference_theorem_l2136_213633


namespace NUMINAMATH_CALUDE_brick_length_calculation_l2136_213692

/-- Calculates the length of a brick given wall dimensions and brick count --/
theorem brick_length_calculation (wall_length wall_height wall_thickness : ℝ)
                                 (brick_width brick_height : ℝ) (brick_count : ℕ) :
  wall_length = 750 ∧ wall_height = 600 ∧ wall_thickness = 22.5 ∧
  brick_width = 11.25 ∧ brick_height = 6 ∧ brick_count = 6000 →
  ∃ (brick_length : ℝ),
    brick_length = 25 ∧
    wall_length * wall_height * wall_thickness =
    brick_length * brick_width * brick_height * brick_count :=
by sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l2136_213692


namespace NUMINAMATH_CALUDE_solve_equation_l2136_213627

theorem solve_equation (r : ℚ) : 3 * (r - 7) = 4 * (2 - 2 * r) + 4 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2136_213627


namespace NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l2136_213649

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l2136_213649


namespace NUMINAMATH_CALUDE_x_intercept_of_parallel_lines_l2136_213612

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 m2 : ℚ) : Prop := m1 = m2

/-- Line l1 with slope m1 and y-intercept b1 -/
def line1 (x y : ℚ) (m1 b1 : ℚ) : Prop := y = m1 * x + b1

/-- Line l2 with slope m2 and y-intercept b2 -/
def line2 (x y : ℚ) (m2 b2 : ℚ) : Prop := y = m2 * x + b2

/-- The x-intercept of a line with slope m and y-intercept b -/
def x_intercept (m b : ℚ) : ℚ := -b / m

theorem x_intercept_of_parallel_lines 
  (a : ℚ) 
  (h_parallel : parallel (-(a+2)/3) (-(a-1)/2)) : 
  x_intercept (-(a+2)/3) (5/3) = 5/9 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_parallel_lines_l2136_213612


namespace NUMINAMATH_CALUDE_dog_food_cost_l2136_213651

-- Define the given constants
def puppy_cost : ℚ := 10
def days : ℕ := 21
def food_per_day : ℚ := 1/3
def food_per_bag : ℚ := 7/2
def total_cost : ℚ := 14

-- Define the theorem
theorem dog_food_cost :
  let total_food := days * food_per_day
  let bags_needed := total_food / food_per_bag
  let food_cost := total_cost - puppy_cost
  food_cost / bags_needed = 2 := by
sorry

end NUMINAMATH_CALUDE_dog_food_cost_l2136_213651


namespace NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_2y2_l2136_213624

theorem min_value_xy_over_x2_plus_2y2 (x y : ℝ) 
  (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) :
  (∃ m : ℝ, m = (x * y) / (x^2 + 2 * y^2) ∧ 
    (∀ x' y' : ℝ, 0.4 ≤ x' ∧ x' ≤ 0.6 → 0.3 ≤ y' ∧ y' ≤ 0.5 → 
      m ≤ (x' * y') / (x'^2 + 2 * y'^2)) ∧
    m = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_2y2_l2136_213624


namespace NUMINAMATH_CALUDE_problem_statement_l2136_213626

theorem problem_statement : 
  (∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≥ 0) ∧ ¬(∀ a b : ℝ, a < b → 1/a > 1/b) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2136_213626


namespace NUMINAMATH_CALUDE_lucille_earnings_l2136_213695

/-- Calculates the earnings from weeding a specific area -/
def calculate_earnings (small medium large : ℕ) : ℕ :=
  4 * small + 8 * medium + 12 * large

/-- Calculates the total cost of items after discount and tax -/
def calculate_total_cost (price : ℕ) (discount_rate tax_rate : ℚ) : ℕ :=
  let discounted_price := price - (price * discount_rate).floor
  (discounted_price + (discounted_price * tax_rate).ceil).toNat

theorem lucille_earnings : 
  let flower_bed := calculate_earnings 6 3 2
  let vegetable_patch := calculate_earnings 10 2 2
  let half_grass := calculate_earnings 10 5 1
  let new_area := calculate_earnings 7 4 1
  let total_earnings := flower_bed + vegetable_patch + half_grass + new_area
  let soda_snack_cost := calculate_total_cost 149 (1/10) (12/100)
  total_earnings - soda_snack_cost = 166 := by sorry

end NUMINAMATH_CALUDE_lucille_earnings_l2136_213695


namespace NUMINAMATH_CALUDE_sin_three_zeros_l2136_213687

/-- Given a function f(x) = sin(ωx + π/3) with ω > 0, if f has exactly 3 zeros
    in the interval [0, 2π/3], then 4 ≤ ω < 11/2 -/
theorem sin_three_zeros (ω : ℝ) (h₁ : ω > 0) :
  (∃! (zeros : Finset ℝ), zeros.card = 3 ∧
    (∀ x ∈ zeros, x ∈ Set.Icc 0 (2 * Real.pi / 3) ∧
      Real.sin (ω * x + Real.pi / 3) = 0)) →
  4 ≤ ω ∧ ω < 11 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_three_zeros_l2136_213687


namespace NUMINAMATH_CALUDE_jack_final_plate_count_l2136_213657

/-- Represents the number of plates Jack has of each type and the total number of plates --/
structure PlateCount where
  flower : ℕ
  checked : ℕ
  polkaDot : ℕ
  total : ℕ

/-- Calculates the final number of plates Jack has --/
def finalPlateCount (initial : PlateCount) : PlateCount :=
  let newPolkaDot := 2 * initial.checked
  let newFlower := initial.flower - 1
  { flower := newFlower
  , checked := initial.checked
  , polkaDot := newPolkaDot
  , total := newFlower + initial.checked + newPolkaDot
  }

/-- Theorem stating that Jack ends up with 27 plates --/
theorem jack_final_plate_count :
  let initial := { flower := 4, checked := 8, polkaDot := 0, total := 12 : PlateCount }
  (finalPlateCount initial).total = 27 := by
  sorry

end NUMINAMATH_CALUDE_jack_final_plate_count_l2136_213657


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l2136_213654

theorem sequence_gcd_property (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ i : ℕ, a i = i :=
sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l2136_213654


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_9_l2136_213646

/-- The number formed by inserting a digit d between 586 and 17 -/
def number (d : Nat) : Nat := 586000 + d * 1000 + 17

/-- Predicate to check if a number is divisible by 9 -/
def divisible_by_9 (n : Nat) : Prop := n % 9 = 0

theorem smallest_digit_divisible_by_9 :
  ∃ (d : Nat), d < 10 ∧ divisible_by_9 (number d) ∧
  ∀ (d' : Nat), d' < d → ¬(divisible_by_9 (number d')) :=
sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_9_l2136_213646


namespace NUMINAMATH_CALUDE_cookie_comparison_l2136_213625

theorem cookie_comparison (a b c : ℕ) (ha : a = 7) (hb : b = 8) (hc : c = 5) :
  (1 : ℚ) / c > (1 : ℚ) / a ∧ (1 : ℚ) / c > (1 : ℚ) / b :=
sorry

end NUMINAMATH_CALUDE_cookie_comparison_l2136_213625


namespace NUMINAMATH_CALUDE_units_digit_difference_l2136_213610

/-- Returns the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Predicate for a natural number being even -/
def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem units_digit_difference (p : ℕ) 
  (h1 : p > 0)
  (h2 : isEven p)
  (h3 : unitsDigit p > 0)
  (h4 : unitsDigit (p + 1) = 7) :
  unitsDigit (p^3) - unitsDigit (p^2) = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_difference_l2136_213610


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2136_213644

theorem pure_imaginary_complex_number (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x^2 + 3*x + 2)*I = (0 : ℂ) + y*I ∧ y ≠ 0) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2136_213644


namespace NUMINAMATH_CALUDE_min_value_theorem_l2136_213615

/-- A circle C with equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def CircleC (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- A line l with equation ax + by - 2 = 0 -/
def LineL (a b x y : ℝ) : Prop :=
  a*x + b*y - 2 = 0

/-- The theorem stating the minimum value of 1/a + 2/b -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_symmetry : ∃ (x y : ℝ), CircleC x y ∧ LineL a b x y) :
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 →
    (∃ (x y : ℝ), CircleC x y ∧ LineL a' b' x y) →
    1/a + 2/b ≤ 1/a' + 2/b') →
  1/a + 2/b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2136_213615


namespace NUMINAMATH_CALUDE_two_tangents_from_origin_l2136_213680

/-- The function f(x) = -x^3 + 3x^2 + 1 -/
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

/-- A point (t, f(t)) on the curve y = f(x) -/
def point_on_curve (t : ℝ) : ℝ × ℝ := (t, f t)

/-- The slope of the tangent line at point (t, f(t)) -/
def tangent_slope (t : ℝ) : ℝ := f' t

/-- The equation for finding points of tangency -/
def tangency_equation (t : ℝ) : Prop := 2*t^3 - 3*t^2 + 1 = 0

theorem two_tangents_from_origin :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    tangency_equation t₁ ∧ 
    tangency_equation t₂ ∧ 
    (∀ t, tangency_equation t → t = t₁ ∨ t = t₂) :=
sorry

end NUMINAMATH_CALUDE_two_tangents_from_origin_l2136_213680


namespace NUMINAMATH_CALUDE_difference_30th_28th_triangular_l2136_213670

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_30th_28th_triangular : 
  triangular_number 30 - triangular_number 28 = 59 := by
  sorry

end NUMINAMATH_CALUDE_difference_30th_28th_triangular_l2136_213670


namespace NUMINAMATH_CALUDE_sum_and_diff_expectations_l2136_213648

variable (X Y : ℝ → ℝ)

-- Define the expectation operator
def expectation (Z : ℝ → ℝ) : ℝ := sorry

-- Given conditions
axiom X_expectation : expectation X = 3
axiom Y_expectation : expectation Y = 2

-- Linearity of expectation
axiom expectation_sum (Z W : ℝ → ℝ) : expectation (Z + W) = expectation Z + expectation W
axiom expectation_diff (Z W : ℝ → ℝ) : expectation (Z - W) = expectation Z - expectation W

-- Theorem to prove
theorem sum_and_diff_expectations :
  expectation (X + Y) = 5 ∧ expectation (X - Y) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_and_diff_expectations_l2136_213648


namespace NUMINAMATH_CALUDE_pond_length_l2136_213677

/-- Given a rectangular field and a square pond, prove the length of the pond --/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_area : ℝ) : 
  field_length = 32 →
  field_width = field_length / 2 →
  pond_area = (field_length * field_width) / 8 →
  ∃ (pond_length : ℝ), pond_length^2 = pond_area ∧ pond_length = 8 :=
by sorry

end NUMINAMATH_CALUDE_pond_length_l2136_213677


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2136_213685

/-- Given a sum P put at simple interest rate R for 4 years, 
    if increasing the rate by 3% results in Rs. 120 more interest, 
    then P = 1000. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  (P * (R + 3) * 4) / 100 - (P * R * 4) / 100 = 120 →
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2136_213685


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2136_213678

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem set_intersection_theorem :
  (Set.univ \ A) ∩ B = {x | 2 < x ∧ x ≤ 3} :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2136_213678


namespace NUMINAMATH_CALUDE_expected_small_supermarkets_l2136_213642

/-- Represents the types of supermarkets --/
inductive SupermarketType
| Small
| Medium
| Large

/-- Represents the count of each type of supermarket --/
structure SupermarketCounts where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the sample size for each type of supermarket --/
structure SampleSizes where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the expected number of small supermarkets in a subsample --/
def expectedSmallInSubsample (counts : SupermarketCounts) (sample : SampleSizes) (subsampleSize : ℕ) : ℚ :=
  (sample.small : ℚ) / ((sample.small + sample.medium + sample.large) : ℚ) * subsampleSize

/-- Theorem stating the expected number of small supermarkets in the subsample --/
theorem expected_small_supermarkets 
  (counts : SupermarketCounts)
  (sample : SampleSizes)
  (h1 : counts.small = 72 ∧ counts.medium = 24 ∧ counts.large = 12)
  (h2 : sample.small + sample.medium + sample.large = 9)
  (h3 : sample.small = 6 ∧ sample.medium = 2 ∧ sample.large = 1)
  : expectedSmallInSubsample counts sample 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_small_supermarkets_l2136_213642


namespace NUMINAMATH_CALUDE_sum_of_primes_with_square_property_l2136_213666

theorem sum_of_primes_with_square_property : ∃ (S : Finset Nat),
  (∀ p ∈ S, Nat.Prime p ∧ ∃ q, Nat.Prime q ∧ ∃ k, p^2 + p*q + q^2 = k^2) ∧
  (∀ p, Nat.Prime p → (∃ q, Nat.Prime q ∧ ∃ k, p^2 + p*q + q^2 = k^2) → p ∈ S) ∧
  S.sum id = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_primes_with_square_property_l2136_213666


namespace NUMINAMATH_CALUDE_geometric_sequence_term_count_l2136_213634

theorem geometric_sequence_term_count (a₁ q aₙ : ℚ) (n : ℕ) :
  a₁ = 1/2 →
  q = 1/2 →
  aₙ = 1/32 →
  aₙ = a₁ * q^(n-1) →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_term_count_l2136_213634


namespace NUMINAMATH_CALUDE_car_production_proof_l2136_213638

/-- The total number of cars produced over two days, given production on each day --/
def total_cars (day1 : ℕ) (day2 : ℕ) : ℕ := day1 + day2

/-- The number of cars produced on the second day is twice that of the first day --/
def double_production (day1 : ℕ) : ℕ := 2 * day1

theorem car_production_proof (day1 : ℕ) (h1 : day1 = 60) :
  total_cars day1 (double_production day1) = 180 := by
  sorry


end NUMINAMATH_CALUDE_car_production_proof_l2136_213638


namespace NUMINAMATH_CALUDE_inequality_problem_l2136_213652

/-- Given an inequality with parameter a, prove that a = 8 and find the solution set -/
theorem inequality_problem (a : ℝ) : 
  (∀ x : ℝ, |x^2 - 4*x + a| + |x - 3| ≤ 5 → x ≤ 3) ∧ 
  (∃ x : ℝ, x = 3 ∧ |x^2 - 4*x + a| + |x - 3| = 5) →
  a = 8 ∧ ∀ x : ℝ, (|x^2 - 4*x + a| + |x - 3| ≤ 5 ↔ 2 ≤ x ∧ x ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_inequality_problem_l2136_213652


namespace NUMINAMATH_CALUDE_fruit_condition_percentage_l2136_213630

theorem fruit_condition_percentage 
  (total_oranges : ℕ) 
  (total_bananas : ℕ) 
  (rotten_oranges_percent : ℚ) 
  (rotten_bananas_percent : ℚ) 
  (h1 : total_oranges = 600) 
  (h2 : total_bananas = 400) 
  (h3 : rotten_oranges_percent = 15 / 100) 
  (h4 : rotten_bananas_percent = 8 / 100) : 
  (1 - (rotten_oranges_percent * total_oranges + rotten_bananas_percent * total_bananas) / 
   (total_oranges + total_bananas)) * 100 = 878 / 10 :=
by sorry

end NUMINAMATH_CALUDE_fruit_condition_percentage_l2136_213630


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l2136_213684

/-- Given a mixture of milk and water with an initial ratio of 3:2, prove that
    after adding 48 liters of water to make the new ratio 3:4, the initial
    volume of the mixture was 120 liters. -/
theorem initial_mixture_volume
  (initial_milk : ℚ) (initial_water : ℚ)
  (initial_ratio : initial_milk / initial_water = 3 / 2)
  (new_ratio : initial_milk / (initial_water + 48) = 3 / 4) :
  initial_milk + initial_water = 120 := by
sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l2136_213684


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l2136_213694

theorem geometric_sequence_inequality (a : Fin 8 → ℝ) (q : ℝ) :
  (∀ i : Fin 8, a i > 0) →
  (∀ i : Fin 7, a (i + 1) = a i * q) →
  q ≠ 1 →
  a 1 + a 8 > a 4 + a 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l2136_213694


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2136_213696

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 > Real.log x)) ↔ (∃ x : ℝ, x^2 ≤ Real.log x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2136_213696


namespace NUMINAMATH_CALUDE_calculate_expression_l2136_213674

theorem calculate_expression : (-Real.sqrt 6)^2 - 3 * Real.sqrt 2 * Real.sqrt 18 = -12 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2136_213674


namespace NUMINAMATH_CALUDE_leak_empty_time_proof_l2136_213640

/-- The time (in hours) it takes to fill the tank without a leak -/
def fill_time_without_leak : ℝ := 3

/-- The time (in hours) it takes to fill the tank with a leak -/
def fill_time_with_leak : ℝ := 4

/-- The capacity of the tank -/
def tank_capacity : ℝ := 1

/-- The time (in hours) it takes for the leak to empty the tank -/
def leak_empty_time : ℝ := 12

theorem leak_empty_time_proof :
  let fill_rate := tank_capacity / fill_time_without_leak
  let combined_rate := tank_capacity / fill_time_with_leak
  let leak_rate := fill_rate - combined_rate
  leak_empty_time = tank_capacity / leak_rate :=
by sorry

end NUMINAMATH_CALUDE_leak_empty_time_proof_l2136_213640


namespace NUMINAMATH_CALUDE_parabola_reflection_translation_sum_l2136_213632

/-- Original parabola function -/
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Reflected parabola function -/
def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := -(a * x^2 + b * x + c)

/-- Translated original parabola (3 units right) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := original_parabola a b c (x - 3)

/-- Translated reflected parabola (4 units left) -/
def g (a b c : ℝ) (x : ℝ) : ℝ := reflected_parabola a b c (x + 4)

/-- Sum of translated original and reflected parabolas -/
def f_plus_g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_reflection_translation_sum (a b c : ℝ) :
  ∀ x, f_plus_g a b c x = -14 * a * x - 19 * a - 7 * b :=
by sorry

end NUMINAMATH_CALUDE_parabola_reflection_translation_sum_l2136_213632


namespace NUMINAMATH_CALUDE_apartment_groceries_cost_l2136_213690

/-- Proves the cost of groceries for three roommates given their expenses -/
theorem apartment_groceries_cost 
  (rent : ℕ) 
  (utilities : ℕ) 
  (internet : ℕ) 
  (cleaning_supplies : ℕ) 
  (one_roommate_total : ℕ) 
  (h1 : rent = 1100)
  (h2 : utilities = 114)
  (h3 : internet = 60)
  (h4 : cleaning_supplies = 40)
  (h5 : one_roommate_total = 924) :
  (one_roommate_total - (rent + utilities + internet + cleaning_supplies) / 3) * 3 = 1458 :=
by sorry

end NUMINAMATH_CALUDE_apartment_groceries_cost_l2136_213690


namespace NUMINAMATH_CALUDE_min_distance_four_points_l2136_213681

/-- Given four points P, Q, R, and S on a line with specified distances between them,
    prove that the minimum possible distance between P and S is 0. -/
theorem min_distance_four_points (P Q R S : ℝ) 
  (h1 : |Q - P| = 12) 
  (h2 : |R - Q| = 7) 
  (h3 : |S - R| = 5) : 
  ∃ (P' Q' R' S' : ℝ), 
    |Q' - P'| = 12 ∧ 
    |R' - Q'| = 7 ∧ 
    |S' - R'| = 5 ∧ 
    |S' - P'| = 0 :=
sorry

end NUMINAMATH_CALUDE_min_distance_four_points_l2136_213681


namespace NUMINAMATH_CALUDE_remainder_problem_l2136_213663

theorem remainder_problem (x : ℤ) (h1 : x % 62 = 7) (h2 : ∃ n : ℤ, (x + n) % 31 = 18) : 
  ∃ n : ℕ, n > 0 ∧ (x + n) % 31 = 18 ∧ ∀ m : ℕ, m > 0 ∧ (x + m) % 31 = 18 → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2136_213663


namespace NUMINAMATH_CALUDE_hotdog_eating_competition_l2136_213676

theorem hotdog_eating_competition (x y z : ℕ+) :
  y = 1 ∧
  x = z - 2 ∧
  6 * ((2*x - 3) + (3*x - y) + (4*x + z) + (x^2 - 5) + (3*y + 5*z) + (x*(y+z)) + ((x^2)+(y*z) - 2) + (x^3*y^2*z-15)) = 10000 →
  ∃ (hotdogs : ℕ), hotdogs = 6 * (x^3 * y^2 * z - 15) :=
by sorry

end NUMINAMATH_CALUDE_hotdog_eating_competition_l2136_213676


namespace NUMINAMATH_CALUDE_smallest_n_is_101_l2136_213645

/-- Represents a square in the n × n table -/
structure Square where
  row : Nat
  col : Nat

/-- Represents a rectangle in the table -/
structure Rectangle where
  topLeft : Square
  bottomRight : Square

/-- Represents the n × n table -/
structure Table (n : Nat) where
  blueSquares : Finset Square
  rectangles : Finset Rectangle
  uniquePartition : Prop
  oneBluePerRectangle : Prop

/-- The main theorem -/
theorem smallest_n_is_101 :
  ∀ n : Nat,
  ∃ (t : Table n),
  (t.blueSquares.card = 101) →
  t.uniquePartition →
  t.oneBluePerRectangle →
  n ≥ 101 ∧ ∃ (t' : Table 101), 
    t'.blueSquares.card = 101 ∧
    t'.uniquePartition ∧
    t'.oneBluePerRectangle :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_101_l2136_213645


namespace NUMINAMATH_CALUDE_season_games_count_l2136_213660

/-- The number of games in a football season -/
def season_games : ℕ := 16

/-- Archie's record for touchdown passes in a season -/
def archie_record : ℕ := 89

/-- Richard's average touchdowns per game -/
def richard_avg : ℕ := 6

/-- Required average touchdowns in final two games to beat the record -/
def final_avg : ℕ := 3

/-- Number of final games -/
def final_games : ℕ := 2

theorem season_games_count :
  ∃ (x : ℕ), 
    x + final_games = season_games ∧
    richard_avg * x + final_avg * final_games > archie_record :=
by sorry

end NUMINAMATH_CALUDE_season_games_count_l2136_213660


namespace NUMINAMATH_CALUDE_complex_equation_product_l2136_213647

theorem complex_equation_product (a b : ℝ) : 
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_product_l2136_213647


namespace NUMINAMATH_CALUDE_journey_time_increase_l2136_213655

theorem journey_time_increase (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 ∧ 
  first_half_speed = 80 ∧ 
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time = 2 := by
sorry

end NUMINAMATH_CALUDE_journey_time_increase_l2136_213655


namespace NUMINAMATH_CALUDE_license_plate_difference_l2136_213617

def florida_combinations : ℕ := 26^4 * 10^3
def georgia_combinations : ℕ := 26^3 * 10^3

theorem license_plate_difference : 
  florida_combinations - georgia_combinations = 439400000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l2136_213617
