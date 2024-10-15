import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_condition_l2215_221538

-- Define propositions P and Q
def P (a b c d : ℝ) : Prop := a ≥ b → c > d
def Q (a b e f : ℝ) : Prop := e ≤ f → a < b

-- Main theorem
theorem sufficient_condition (a b c d e f : ℝ) 
  (hP : P a b c d) 
  (hnotQ : ¬(Q a b e f)) : 
  c ≤ d → e ≤ f := by
sorry

end NUMINAMATH_CALUDE_sufficient_condition_l2215_221538


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l2215_221543

def total_students : ℕ := 300
def cat_owners : ℕ := 45

theorem percentage_of_cat_owners : 
  (cat_owners : ℚ) / (total_students : ℚ) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l2215_221543


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l2215_221597

theorem butterflies_in_garden (initial : ℕ) (flew_away_fraction : ℚ) (remaining : ℕ) : 
  initial = 9 → 
  flew_away_fraction = 1/3 → 
  remaining = initial - (initial * flew_away_fraction).num →
  remaining = 6 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l2215_221597


namespace NUMINAMATH_CALUDE_expression_value_l2215_221588

theorem expression_value (a b c : ℤ) (ha : a = 10) (hb : b = 15) (hc : c = 3) :
  (a - (b - c)) - ((a - b) + c) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2215_221588


namespace NUMINAMATH_CALUDE_side_to_perimeter_ratio_l2215_221520

/-- Represents a square garden -/
structure SquareGarden where
  side_length : ℝ

/-- Calculate the perimeter of a square garden -/
def perimeter (g : SquareGarden) : ℝ := 4 * g.side_length

/-- Theorem stating the ratio of side length to perimeter for a 15-foot square garden -/
theorem side_to_perimeter_ratio (g : SquareGarden) (h : g.side_length = 15) :
  g.side_length / perimeter g = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_side_to_perimeter_ratio_l2215_221520


namespace NUMINAMATH_CALUDE_mika_stickers_l2215_221585

/-- The number of stickers Mika has after all events -/
def final_stickers (initial bought birthday given_away used : ℕ) : ℕ :=
  initial + bought + birthday - given_away - used

/-- Theorem stating that Mika is left with 28 stickers -/
theorem mika_stickers : final_stickers 45 53 35 19 86 = 28 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_l2215_221585


namespace NUMINAMATH_CALUDE_triangle_side_angle_inequality_l2215_221547

/-- Triangle inequality for side lengths and angles -/
theorem triangle_side_angle_inequality 
  (a b c : ℝ) (α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angle_sum : α + β + γ = Real.pi) : 
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_angle_inequality_l2215_221547


namespace NUMINAMATH_CALUDE_sum_removal_equals_half_l2215_221564

theorem sum_removal_equals_half :
  let original_sum := (1 : ℚ) / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18
  let removed_terms := 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18
  original_sum - removed_terms = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_removal_equals_half_l2215_221564


namespace NUMINAMATH_CALUDE_largest_solution_and_ratio_l2215_221553

theorem largest_solution_and_ratio (x a b c d : ℤ) : 
  (7 * x / 5 + 3 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (a = -15 ∧ b = 1 ∧ c = 785 ∧ d = 14) →
  (x = (-15 + Real.sqrt 785) / 14 ∧ a * c * d / b = -164850) := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_and_ratio_l2215_221553


namespace NUMINAMATH_CALUDE_days_worked_l2215_221541

/-- Proves that given the conditions of the problem, the number of days worked is 23 -/
theorem days_worked (total_days : ℕ) (daily_wage : ℕ) (daily_forfeit : ℕ) (net_earnings : ℕ) 
  (h1 : total_days = 25)
  (h2 : daily_wage = 20)
  (h3 : daily_forfeit = 5)
  (h4 : net_earnings = 450) :
  ∃ (worked_days : ℕ), 
    worked_days * daily_wage - (total_days - worked_days) * daily_forfeit = net_earnings ∧ 
    worked_days = 23 := by
  sorry

#check days_worked

end NUMINAMATH_CALUDE_days_worked_l2215_221541


namespace NUMINAMATH_CALUDE_equation_holds_l2215_221574

theorem equation_holds (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c ↔ b + c = 12 :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_l2215_221574


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l2215_221559

/-- Represents the dimensions of a rectangular shape -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular shape given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Calculates the number of smaller rectangles that can fit into a larger rectangle -/
def number_of_pieces (tray : Dimensions) (piece : Dimensions) : ℕ :=
  (area tray) / (area piece)

theorem brownie_pieces_count :
  let tray := Dimensions.mk 24 16
  let piece := Dimensions.mk 2 2
  number_of_pieces tray piece = 96 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l2215_221559


namespace NUMINAMATH_CALUDE_complex_division_l2215_221510

theorem complex_division (i : ℂ) (h : i * i = -1) : (3 - 4*i) / i = -4 - 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l2215_221510


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2215_221587

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, b > a ∧ a > 0 → 1/a > 1/b) ∧
  ¬(∀ a b : ℝ, 1/a > 1/b → b > a ∧ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2215_221587


namespace NUMINAMATH_CALUDE_initial_blue_balls_l2215_221575

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 15 → removed = 3 → prob = 1/3 → 
  ∃ (initial_blue : ℕ), 
    initial_blue = 7 ∧ 
    (initial_blue - removed : ℚ) / (total - removed) = prob :=
by sorry

end NUMINAMATH_CALUDE_initial_blue_balls_l2215_221575


namespace NUMINAMATH_CALUDE_original_number_proof_l2215_221567

theorem original_number_proof (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2215_221567


namespace NUMINAMATH_CALUDE_number_solution_l2215_221589

theorem number_solution : ∃ x : ℚ, x + (3/5) * x = 240 ∧ x = 150 := by sorry

end NUMINAMATH_CALUDE_number_solution_l2215_221589


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2215_221554

theorem sqrt_inequality (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) < Real.sqrt (3 * a^2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2215_221554


namespace NUMINAMATH_CALUDE_temperature_difference_l2215_221573

theorem temperature_difference (M L N : ℝ) : 
  (M = L + N) →  -- Minneapolis is N degrees warmer than St. Louis at noon
  (|((L + N) - 6) - (L + 4)| = 3) →  -- Temperature difference at 5:00 PM
  (N = 13 ∨ N = 7) ∧ (13 * 7 = 91) :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_l2215_221573


namespace NUMINAMATH_CALUDE_aprons_to_sew_is_49_l2215_221521

def total_aprons : ℕ := 150
def aprons_sewn_initially : ℕ := 13

def aprons_sewn_today (initial : ℕ) : ℕ := 3 * initial

def remaining_aprons (total sewn : ℕ) : ℕ := total - sewn

def aprons_to_sew_tomorrow (remaining : ℕ) : ℕ := remaining / 2

theorem aprons_to_sew_is_49 : 
  aprons_to_sew_tomorrow (remaining_aprons total_aprons (aprons_sewn_initially + aprons_sewn_today aprons_sewn_initially)) = 49 := by
  sorry

end NUMINAMATH_CALUDE_aprons_to_sew_is_49_l2215_221521


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2215_221583

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 20*r^2 + 18*r - 7 = 0 →
  s^3 - 20*s^2 + 18*s - 7 = 0 →
  t^3 - 20*t^2 + 18*t - 7 = 0 →
  (r / ((1/r) + s*t)) + (s / ((1/s) + t*r)) + (t / ((1/t) + r*s)) = 91/2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2215_221583


namespace NUMINAMATH_CALUDE_minimum_value_of_expression_l2215_221524

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

theorem minimum_value_of_expression (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (∀ k : ℕ, a k > 0) →
  a m * a n = 4 * (a 2)^2 →
  (2 : ℝ) / m + 1 / (2 * n) ≥ 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_expression_l2215_221524


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2215_221501

theorem inequality_equivalence (x : ℝ) : 
  (5 ≤ x / (2 * x - 6) ∧ x / (2 * x - 6) < 10) ↔ (3 < x ∧ x < 60 / 19) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2215_221501


namespace NUMINAMATH_CALUDE_evaluate_expression_l2215_221569

theorem evaluate_expression : (2^3)^4 * 3^2 = 36864 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2215_221569


namespace NUMINAMATH_CALUDE_square_sum_of_solution_l2215_221542

theorem square_sum_of_solution (x y : ℝ) : 
  x * y = 8 → 
  x^2 * y + x * y^2 + x + y = 80 → 
  x^2 + y^2 = 5104 / 81 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_solution_l2215_221542


namespace NUMINAMATH_CALUDE_line_through_point_l2215_221596

/-- Given a line 2kx - my = 4 passing through the point (3, -2), prove that k = 2/5 and m = 4/5 -/
theorem line_through_point (k m : ℚ) : 
  (2 * k * 3 - m * (-2) = 4) → 
  k = 2/5 ∧ m = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2215_221596


namespace NUMINAMATH_CALUDE_little_john_money_distribution_l2215_221517

theorem little_john_money_distribution 
  (initial_amount : ℚ) 
  (sweets_cost : ℚ) 
  (amount_left : ℚ) 
  (num_friends : ℕ) 
  (h1 : initial_amount = 5.1)
  (h2 : sweets_cost = 1.05)
  (h3 : amount_left = 2.05)
  (h4 : num_friends = 2) :
  let total_spent := initial_amount - amount_left
  let friends_money := total_spent - sweets_cost
  friends_money / num_friends = 1 := by
sorry

end NUMINAMATH_CALUDE_little_john_money_distribution_l2215_221517


namespace NUMINAMATH_CALUDE_coin_flip_difference_l2215_221552

theorem coin_flip_difference (total_flips : ℕ) (heads : ℕ) (h1 : total_flips = 211) (h2 : heads = 65) :
  total_flips - heads - heads = 81 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_difference_l2215_221552


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l2215_221505

/-- Given two rectangles of equal area, where one rectangle has dimensions 9 inches by 20 inches,
    and the other has a width of 15 inches, prove that the length of the second rectangle is 12 inches. -/
theorem equal_area_rectangles (carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_width = 15)
    (h2 : jordan_length = 9)
    (h3 : jordan_width = 20)
    (h4 : carol_width * carol_length = jordan_length * jordan_width)
    : carol_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l2215_221505


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2215_221523

theorem product_of_three_numbers (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x + y + z = 30)
  (h_first : x = 3 * (y + z))
  (h_second : y = 8 * z) : 
  x * y * z = 125 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2215_221523


namespace NUMINAMATH_CALUDE_min_sum_of_product_1806_l2215_221560

theorem min_sum_of_product_1806 (a b c : ℕ+) : 
  a * b * c = 1806 → 
  (Even a ∨ Even b ∨ Even c) → 
  (∀ x y z : ℕ+, x * y * z = 1806 → (Even x ∨ Even y ∨ Even z) → a + b + c ≤ x + y + z) →
  a + b + c = 112 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1806_l2215_221560


namespace NUMINAMATH_CALUDE_fruit_consumption_l2215_221534

theorem fruit_consumption (total_fruits initial_kept friday_fruits : ℕ) 
  (h_total : total_fruits = 10)
  (h_kept : initial_kept = 2)
  (h_friday : friday_fruits = 3) :
  ∃ (a b o : ℕ),
    a = b ∧ 
    o = 2 * a ∧
    a + b + o = total_fruits - (initial_kept + friday_fruits) ∧
    a = 1 ∧ 
    b = 1 ∧ 
    o = 2 ∧
    a + b + o = 4 := by
  sorry

end NUMINAMATH_CALUDE_fruit_consumption_l2215_221534


namespace NUMINAMATH_CALUDE_all_inequalities_true_l2215_221518

theorem all_inequalities_true (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z > 0) :
  (x + z > y + z) ∧
  (x - 2*z > y - 2*z) ∧
  (x*z^2 > y*z^2) ∧
  (x/z > y/z) ∧
  (x - z^2 > y - z^2) := by
  sorry

end NUMINAMATH_CALUDE_all_inequalities_true_l2215_221518


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l2215_221540

/-- Given that a is inversely proportional to b, prove that b₁/b₂ = 5/4 when a₁/a₂ = 4/5 -/
theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ : ℝ) (ha₁ : a₁ ≠ 0) (ha₂ : a₂ ≠ 0) (hb₁ : b₁ ≠ 0) (hb₂ : b₂ ≠ 0)
    (h_inverse : ∃ k : ℝ, a₁ * b₁ = k ∧ a₂ * b₂ = k) (h_ratio : a₁ / a₂ = 4 / 5) :
  b₁ / b₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l2215_221540


namespace NUMINAMATH_CALUDE_card_area_reduction_l2215_221544

/-- Given a 5x7 inch card, if reducing one side by 2 inches results in an area of 21 square inches,
    then reducing the other side by 2 inches instead will result in an area of 25 square inches. -/
theorem card_area_reduction (length width : ℝ) : 
  length = 5 ∧ width = 7 ∧ 
  ((length - 2) * width = 21 ∨ length * (width - 2) = 21) →
  (length * (width - 2) = 25 ∨ (length - 2) * width = 25) := by
sorry

end NUMINAMATH_CALUDE_card_area_reduction_l2215_221544


namespace NUMINAMATH_CALUDE_library_book_distribution_l2215_221593

/-- The number of ways to distribute books between the library and checked out -/
def distributeBooks (total : ℕ) (minInLibrary : ℕ) (minCheckedOut : ℕ) : ℕ :=
  (total - minInLibrary - minCheckedOut + 1)

/-- Theorem: There are 6 ways to distribute 10 books with at least 2 in the library and 3 checked out -/
theorem library_book_distribution :
  distributeBooks 10 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_library_book_distribution_l2215_221593


namespace NUMINAMATH_CALUDE_condition_relation_l2215_221509

theorem condition_relation (A B C : Prop) 
  (h1 : B → A)  -- A is a necessary condition for B
  (h2 : C → B)  -- C is a sufficient condition for B
  (h3 : ¬(B → C))  -- C is not a necessary condition for B
  : (C → A) ∧ ¬(A → C) := by
  sorry

end NUMINAMATH_CALUDE_condition_relation_l2215_221509


namespace NUMINAMATH_CALUDE_product_increase_value_l2215_221525

theorem product_increase_value (x : ℝ) (v : ℝ) : 
  x = 3 → 5 * x + v = 19 → v = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_increase_value_l2215_221525


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l2215_221527

theorem negation_of_universal_quantifier :
  (¬ ∀ x : ℝ, x ≥ Real.sqrt 2 → x^2 ≥ 2) ↔ (∃ x : ℝ, x ≥ Real.sqrt 2 ∧ x^2 < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l2215_221527


namespace NUMINAMATH_CALUDE_sum_of_factors_l2215_221598

theorem sum_of_factors (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 20 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 30 = (x + b) * (x - c)) →
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_l2215_221598


namespace NUMINAMATH_CALUDE_range_of_m_l2215_221519

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + y = 1)
  (h_ineq : ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 4 * x^2 + y^2 + Real.sqrt (x * y) - m < 0) :
  m > 17/16 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2215_221519


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l2215_221535

theorem complex_magnitude_proof : 
  Complex.abs ((11/13 : ℂ) + (12/13 : ℂ) * Complex.I)^12 = (Real.sqrt 265 / 13)^12 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l2215_221535


namespace NUMINAMATH_CALUDE_raindrop_probability_l2215_221500

/-- The probability of a raindrop landing on the third slope of a triangular pyramid roof -/
theorem raindrop_probability (α β : Real) : 
  -- The roof is a triangular pyramid with all plane angles at the vertex being right angles
  -- The red slope is inclined at an angle α to the horizontal
  -- The blue slope is inclined at an angle β to the horizontal
  -- We assume 0 ≤ α ≤ π/2 and 0 ≤ β ≤ π/2 to ensure valid angles
  0 ≤ α ∧ α ≤ π/2 ∧ 0 ≤ β ∧ β ≤ π/2 →
  -- The probability of a raindrop landing on the green slope
  ∃ (p : Real), p = 1 - (Real.cos α)^2 - (Real.cos β)^2 ∧ 0 ≤ p ∧ p ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_raindrop_probability_l2215_221500


namespace NUMINAMATH_CALUDE_cousins_in_rooms_l2215_221511

/-- The number of ways to distribute n indistinguishable objects into k indistinguishable containers -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- There are 4 cousins and 4 identical rooms -/
theorem cousins_in_rooms : distribute 4 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cousins_in_rooms_l2215_221511


namespace NUMINAMATH_CALUDE_largest_even_not_sum_of_odd_composites_l2215_221594

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m

/-- A number is odd if it leaves a remainder of 1 when divided by 2 -/
def IsOdd (n : ℕ) : Prop :=
  n % 2 = 1

/-- The property of being expressible as the sum of two odd composite numbers -/
def IsSumOfTwoOddComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsOdd a ∧ IsOdd b ∧ IsComposite a ∧ IsComposite b ∧ n = a + b

/-- 38 is the largest even integer that cannot be written as the sum of two odd composite numbers -/
theorem largest_even_not_sum_of_odd_composites :
  (∀ n : ℕ, n % 2 = 0 → n > 38 → IsSumOfTwoOddComposites n) ∧
  ¬IsSumOfTwoOddComposites 38 :=
sorry

end NUMINAMATH_CALUDE_largest_even_not_sum_of_odd_composites_l2215_221594


namespace NUMINAMATH_CALUDE_number_greater_than_one_eighth_l2215_221546

theorem number_greater_than_one_eighth : ∃ x : ℝ, x = 1/8 + 0.0020000000000000018 ∧ x = 0.1270000000000000018 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_one_eighth_l2215_221546


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2215_221508

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 6) * (x + 2) = k + 3 * x) ↔ k = 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2215_221508


namespace NUMINAMATH_CALUDE_inequality_proof_l2215_221566

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (a b c : ℝ)
  (ha : Real.sqrt a = x * (y - z)^2)
  (hb : Real.sqrt b = y * (z - x)^2)
  (hc : Real.sqrt c = z * (x - y)^2) :
  a^2 + b^2 + c^2 ≥ 2 * (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2215_221566


namespace NUMINAMATH_CALUDE_penny_collection_difference_l2215_221529

theorem penny_collection_difference (cassandra_pennies james_pennies : ℕ) : 
  cassandra_pennies = 5000 →
  james_pennies < cassandra_pennies →
  cassandra_pennies + james_pennies = 9724 →
  cassandra_pennies - james_pennies = 276 := by
sorry

end NUMINAMATH_CALUDE_penny_collection_difference_l2215_221529


namespace NUMINAMATH_CALUDE_inclination_angle_range_l2215_221551

open Set

-- Define the line equation
def line_equation (x y : ℝ) (α : ℝ) : Prop :=
  x * Real.sin α + y + 2 = 0

-- Define the range of the inclination angle
def inclination_range : Set ℝ :=
  Icc 0 (Real.pi / 4) ∪ Ico (3 * Real.pi / 4) Real.pi

-- Theorem statement
theorem inclination_angle_range :
  ∀ α, (∃ x y, line_equation x y α) → α ∈ inclination_range :=
sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l2215_221551


namespace NUMINAMATH_CALUDE_fraction_simplification_l2215_221507

theorem fraction_simplification :
  (1722^2 - 1715^2) / (1731^2 - 1706^2) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2215_221507


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2215_221502

theorem complex_modulus_problem (z : ℂ) : 
  z = ((1 - I) * (2 - I)) / (1 + 2*I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2215_221502


namespace NUMINAMATH_CALUDE_worker_pay_calculation_l2215_221522

/-- Calculate the worker's pay given the following conditions:
  * The total period is 60 days
  * The pay rate for working is Rs. 20 per day
  * The deduction rate for idle days is Rs. 3 per day
  * The number of idle days is 40 days
-/
def worker_pay (total_days : ℕ) (work_rate : ℕ) (idle_rate : ℕ) (idle_days : ℕ) : ℕ :=
  let work_days := total_days - idle_days
  let earnings := work_days * work_rate
  let deductions := idle_days * idle_rate
  earnings - deductions

theorem worker_pay_calculation :
  worker_pay 60 20 3 40 = 280 := by
  sorry

end NUMINAMATH_CALUDE_worker_pay_calculation_l2215_221522


namespace NUMINAMATH_CALUDE_new_average_weight_l2215_221568

def initial_average_weight : ℝ := 48
def initial_members : ℕ := 23
def new_person1_weight : ℝ := 78
def new_person2_weight : ℝ := 93

theorem new_average_weight :
  let total_initial_weight := initial_average_weight * initial_members
  let total_new_weight := new_person1_weight + new_person2_weight
  let total_weight := total_initial_weight + total_new_weight
  let new_members := initial_members + 2
  total_weight / new_members = 51 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l2215_221568


namespace NUMINAMATH_CALUDE_smallest_among_given_numbers_l2215_221578

theorem smallest_among_given_numbers :
  ∀ (a b c d : ℝ), a = -1 ∧ b = 0 ∧ c = -Real.sqrt 2 ∧ d = 2 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_given_numbers_l2215_221578


namespace NUMINAMATH_CALUDE_cut_cube_theorem_l2215_221528

/-- Represents a cube that has been cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes with exactly 2 painted faces -/
  two_face_cubes : ℕ
  /-- The total number of smaller cubes created -/
  total_cubes : ℕ

/-- Theorem stating that if a cube is cut such that there are 12 smaller cubes
    with 2 painted faces, then the total number of smaller cubes is 8 -/
theorem cut_cube_theorem (c : CutCube) :
  c.two_face_cubes = 12 → c.total_cubes = 8 := by
  sorry

end NUMINAMATH_CALUDE_cut_cube_theorem_l2215_221528


namespace NUMINAMATH_CALUDE_volcano_eruption_percentage_l2215_221572

theorem volcano_eruption_percentage (total_volcanoes : ℕ) 
  (intact_volcanoes : ℕ) (mid_year_percentage : ℝ) 
  (end_year_percentage : ℝ) :
  total_volcanoes = 200 →
  intact_volcanoes = 48 →
  mid_year_percentage = 0.4 →
  end_year_percentage = 0.5 →
  ∃ (x : ℝ),
    x ≥ 0 ∧ x ≤ 100 ∧
    (total_volcanoes : ℝ) * (1 - x / 100) * (1 - mid_year_percentage) * (1 - end_year_percentage) = intact_volcanoes ∧
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_volcano_eruption_percentage_l2215_221572


namespace NUMINAMATH_CALUDE_f_at_six_l2215_221557

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 2 * x^4 + 5 * x^3 - x^2 + 3 * x + 4

-- Theorem stating that f(6) = 3658
theorem f_at_six : f 6 = 3658 := by sorry

end NUMINAMATH_CALUDE_f_at_six_l2215_221557


namespace NUMINAMATH_CALUDE_sheridan_cats_l2215_221571

def current_cats : ℕ := sorry
def needed_cats : ℕ := 32
def total_cats : ℕ := 43

theorem sheridan_cats : current_cats = 11 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_l2215_221571


namespace NUMINAMATH_CALUDE_line_equivalence_l2215_221576

theorem line_equivalence :
  ∀ (x y : ℝ),
  (3 : ℝ) * (x - 2) + (-4 : ℝ) * (y - (-1)) = 0 ↔
  y = (3/4 : ℝ) * x - (5/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_line_equivalence_l2215_221576


namespace NUMINAMATH_CALUDE_inequality_solution_l2215_221531

-- Define the inequality function
def f (a x : ℝ) : ℝ := x^2 - a*x + a - 1

-- Define the solution set for a > 2
def solution_set_gt2 (a : ℝ) : Set ℝ := 
  {x | x < 1 ∨ x > a - 1}

-- Define the solution set for a = 2
def solution_set_eq2 : Set ℝ := 
  {x | x < 1 ∨ x > 1}

-- Define the solution set for a < 2
def solution_set_lt2 (a : ℝ) : Set ℝ := 
  {x | x < a - 1 ∨ x > 1}

-- Theorem statement
theorem inequality_solution (a : ℝ) :
  (∀ x, f a x > 0 ↔ 
    (a > 2 ∧ x ∈ solution_set_gt2 a) ∨
    (a = 2 ∧ x ∈ solution_set_eq2) ∨
    (a < 2 ∧ x ∈ solution_set_lt2 a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2215_221531


namespace NUMINAMATH_CALUDE_triangle_side_length_expression_l2215_221515

/-- Given a triangle with side lengths a, b, and c, 
    the expression |a-b+c| - |a-b-c| simplifies to 2a - 2b -/
theorem triangle_side_length_expression (a b c : ℝ) 
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  |a - b + c| - |a - b - c| = 2*a - 2*b := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_expression_l2215_221515


namespace NUMINAMATH_CALUDE_units_digit_17_2005_l2215_221536

theorem units_digit_17_2005 (h : 17 % 10 = 7) : (17^2005) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_2005_l2215_221536


namespace NUMINAMATH_CALUDE_nearest_gardeners_to_flower_l2215_221555

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Represents a gardener -/
structure Gardener where
  position : Point

/-- Represents a flower -/
structure Flower where
  position : Point

/-- Theorem: The three nearest gardeners to a flower in the top-left quarter
    of a 2x2 grid are those at the top-left, top-right, and bottom-left corners -/
theorem nearest_gardeners_to_flower 
  (gardenerA : Gardener) 
  (gardenerB : Gardener)
  (gardenerC : Gardener)
  (gardenerD : Gardener)
  (flower : Flower)
  (h1 : gardenerA.position = ⟨0, 2⟩)
  (h2 : gardenerB.position = ⟨2, 2⟩)
  (h3 : gardenerC.position = ⟨0, 0⟩)
  (h4 : gardenerD.position = ⟨2, 0⟩)
  (h5 : 0 < flower.position.x ∧ flower.position.x < 1)
  (h6 : 1 < flower.position.y ∧ flower.position.y < 2) :
  squaredDistance flower.position gardenerA.position < squaredDistance flower.position gardenerD.position ∧
  squaredDistance flower.position gardenerB.position < squaredDistance flower.position gardenerD.position ∧
  squaredDistance flower.position gardenerC.position < squaredDistance flower.position gardenerD.position :=
by sorry

end NUMINAMATH_CALUDE_nearest_gardeners_to_flower_l2215_221555


namespace NUMINAMATH_CALUDE_nail_trimming_sounds_l2215_221586

/-- Represents the number of customers --/
def num_customers : Nat := 3

/-- Represents the number of appendages per customer --/
def appendages_per_customer : Nat := 4

/-- Represents the number of nails per appendage --/
def nails_per_appendage : Nat := 4

/-- Calculates the total number of nail trimming sounds --/
def total_nail_sounds : Nat :=
  num_customers * appendages_per_customer * nails_per_appendage

/-- Theorem stating that the total number of nail trimming sounds is 48 --/
theorem nail_trimming_sounds :
  total_nail_sounds = 48 := by
  sorry

end NUMINAMATH_CALUDE_nail_trimming_sounds_l2215_221586


namespace NUMINAMATH_CALUDE_edward_candy_purchase_l2215_221504

/-- The number of candy pieces Edward can buy given his tickets and the candy cost --/
theorem edward_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : 
  whack_a_mole_tickets = 3 →
  skee_ball_tickets = 5 →
  candy_cost = 4 →
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_candy_purchase_l2215_221504


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l2215_221570

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem first_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_prod : a 2 * a 3 * a 4 = 27) 
  (h_seventh : a 7 = 27) : 
  a 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l2215_221570


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l2215_221532

-- Define the circle C
def circle_C (x y : ℝ) (a : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + a = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - y - 3 = 0

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define perpendicularity of vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem circle_and_line_intersection (a : ℝ) :
  (∃ (x y : ℝ), circle_C x y a ∧ line_l x y) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ a ∧ line_l x₁ y₁ ∧
    circle_C x₂ y₂ a ∧ line_l x₂ y₂ ∧
    perpendicular (x₁, y₁) (x₂, y₂)) →
  (∀ (x y : ℝ), line_m x y ↔ (x = -2 ∧ y = 1) ∨ (x + y + 1 = 0)) ∧
  a = -18 := by sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l2215_221532


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l2215_221550

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 ∨ Real.sqrt (Real.sqrt 81) = -3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l2215_221550


namespace NUMINAMATH_CALUDE_opposite_of_negative_hundred_l2215_221539

theorem opposite_of_negative_hundred : -((-100 : ℤ)) = (100 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_hundred_l2215_221539


namespace NUMINAMATH_CALUDE_caterpillars_on_tree_l2215_221592

theorem caterpillars_on_tree (initial : ℕ) (hatched : ℕ) (left : ℕ) : 
  initial = 14 → hatched = 4 → left = 8 → 
  initial + hatched - left = 10 := by sorry

end NUMINAMATH_CALUDE_caterpillars_on_tree_l2215_221592


namespace NUMINAMATH_CALUDE_orange_juice_serving_size_l2215_221514

/-- Represents the ratio of concentrate to water in the orange juice mixture -/
def concentrateToWaterRatio : ℚ := 1 / 3

/-- The number of cans of concentrate required -/
def concentrateCans : ℕ := 35

/-- The volume of each can of concentrate in ounces -/
def canSize : ℕ := 12

/-- The number of servings to be prepared -/
def numberOfServings : ℕ := 280

/-- The size of each serving in ounces -/
def servingSize : ℚ := 6

theorem orange_juice_serving_size :
  (concentrateCans * canSize * (1 + concentrateToWaterRatio)) / numberOfServings = servingSize :=
sorry

end NUMINAMATH_CALUDE_orange_juice_serving_size_l2215_221514


namespace NUMINAMATH_CALUDE_fraction_irreducible_l2215_221565

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l2215_221565


namespace NUMINAMATH_CALUDE_angle_bisector_m_abs_z_over_one_plus_i_l2215_221595

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m - 6) (m^2 - 2*m - 15)

-- Theorem 1: When z is on the angle bisector of the first and third quadrants, m = -3
theorem angle_bisector_m (m : ℝ) : z m = Complex.mk (z m).re (z m).re → m = -3 := by
  sorry

-- Theorem 2: When m = -1, |z/(1+i)| = √74
theorem abs_z_over_one_plus_i : Complex.abs (z (-1) / (1 + Complex.I)) = Real.sqrt 74 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_m_abs_z_over_one_plus_i_l2215_221595


namespace NUMINAMATH_CALUDE_natural_subset_rational_l2215_221561

theorem natural_subset_rational :
  (∀ x : ℕ, ∃ y : ℚ, (x : ℚ) = y) ∧
  (∃ z : ℚ, ∀ w : ℕ, (w : ℚ) ≠ z) :=
by sorry

end NUMINAMATH_CALUDE_natural_subset_rational_l2215_221561


namespace NUMINAMATH_CALUDE_min_value_quadratic_roots_l2215_221516

theorem min_value_quadratic_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*m*x₁ + m^2 + 3*m - 2 = 0) →
  (x₂^2 + 2*m*x₂ + m^2 + 3*m - 2 = 0) →
  (∃ (min : ℝ), ∀ (m : ℝ), x₁*(x₂ + x₁) + x₂^2 ≥ min ∧ 
  ∃ (m₀ : ℝ), x₁*(x₂ + x₁) + x₂^2 = min) →
  (∃ (min : ℝ), min = 5/4 ∧ 
  ∀ (m : ℝ), x₁*(x₂ + x₁) + x₂^2 ≥ min ∧ 
  ∃ (m₀ : ℝ), x₁*(x₂ + x₁) + x₂^2 = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_roots_l2215_221516


namespace NUMINAMATH_CALUDE_sons_age_few_years_back_l2215_221562

/-- Proves that the son's age a few years back is 22, given the conditions of the problem -/
theorem sons_age_few_years_back (father_current_age : ℕ) (son_current_age : ℕ) : 
  father_current_age = 44 →
  father_current_age - son_current_age = son_current_age →
  son_current_age = 22 :=
by
  sorry

#check sons_age_few_years_back

end NUMINAMATH_CALUDE_sons_age_few_years_back_l2215_221562


namespace NUMINAMATH_CALUDE_min_value_theorem_l2215_221599

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 3| - t

-- State the theorem
theorem min_value_theorem (t : ℝ) (a b : ℝ) :
  (∀ x, f t (x + 2) ≤ 0 ↔ x ∈ Set.Icc (-1) 3) →
  (a > 0 ∧ b > 0) →
  (a * b - 2 * a - 8 * b = 2 * t - 2) →
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ - 2 * a₀ - 8 * b₀ = 2 * t - 2 ∧
    ∀ a' b', a' > 0 → b' > 0 → a' * b' - 2 * a' - 8 * b' = 2 * t - 2 → a₀ + 2 * b₀ ≤ a' + 2 * b') →
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ - 2 * a₀ - 8 * b₀ = 2 * t - 2 ∧ a₀ + 2 * b₀ = 36) :=
by sorry


end NUMINAMATH_CALUDE_min_value_theorem_l2215_221599


namespace NUMINAMATH_CALUDE_magnitude_relationship_l2215_221506

theorem magnitude_relationship (x : ℝ) (h : 0 < x ∧ x < π/4) :
  let A := Real.cos (x^(Real.sin (x^(Real.sin x))))
  let B := Real.sin (x^(Real.cos (x^(Real.sin x))))
  let C := Real.cos (x^(Real.sin (x * x^(Real.cos x))))
  B < A ∧ A < C := by
  sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l2215_221506


namespace NUMINAMATH_CALUDE_line_properties_l2215_221580

/-- A line passing through point A(4, -1) with equal intercepts on x and y axes --/
def line_with_equal_intercepts : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + p.2 = 3) ∨ (p.1 + 4 * p.2 = 0)}

/-- The point A(4, -1) --/
def point_A : ℝ × ℝ := (4, -1)

/-- Theorem stating that the line passes through point A and has equal intercepts --/
theorem line_properties :
  point_A ∈ line_with_equal_intercepts ∧
  ∃ a : ℝ, (a, 0) ∈ line_with_equal_intercepts ∧ (0, a) ∈ line_with_equal_intercepts :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l2215_221580


namespace NUMINAMATH_CALUDE_average_salary_example_l2215_221513

/-- The average salary of 5 people given their individual salaries -/
def average_salary (a b c d e : ℕ) : ℚ :=
  (a + b + c + d + e : ℚ) / 5

/-- Theorem: The average salary of 5 people with salaries 8000, 5000, 14000, 7000, and 9000 is 8200 -/
theorem average_salary_example : average_salary 8000 5000 14000 7000 9000 = 8200 := by
  sorry

#eval average_salary 8000 5000 14000 7000 9000

end NUMINAMATH_CALUDE_average_salary_example_l2215_221513


namespace NUMINAMATH_CALUDE_equation_solution_l2215_221584

theorem equation_solution (x : ℝ) : 
  12 * Real.sin x - 5 * Real.cos x = 13 ↔ 
  ∃ k : ℤ, x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2215_221584


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2215_221582

/-- Given an arithmetic sequence with the first four terms x^2 + 2y, x^2 - 2y, x+y, and x-y,
    the fifth term of the sequence is x - 5y. -/
theorem arithmetic_sequence_fifth_term 
  (x y : ℝ) 
  (seq : ℕ → ℝ)
  (h1 : seq 0 = x^2 + 2*y)
  (h2 : seq 1 = x^2 - 2*y)
  (h3 : seq 2 = x + y)
  (h4 : seq 3 = x - y)
  (h_arithmetic : ∀ n, seq (n + 1) - seq n = seq 1 - seq 0) :
  seq 4 = x - 5*y :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2215_221582


namespace NUMINAMATH_CALUDE_sin_two_phi_l2215_221577

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_l2215_221577


namespace NUMINAMATH_CALUDE_satellite_selection_probabilities_l2215_221530

/-- The number of geostationary Earth orbit (GEO) satellites -/
def num_geo : ℕ := 3

/-- The number of inclined geosynchronous orbit (IGSO) satellites -/
def num_igso : ℕ := 3

/-- The total number of satellites to select -/
def num_select : ℕ := 2

/-- The probability of selecting exactly one GEO satellite and one IGSO satellite -/
def prob_one_geo_one_igso : ℚ := 3/5

/-- The probability of selecting at least one IGSO satellite -/
def prob_at_least_one_igso : ℚ := 4/5

theorem satellite_selection_probabilities :
  (num_geo = 3 ∧ num_igso = 3 ∧ num_select = 2) →
  (prob_one_geo_one_igso = 3/5 ∧ prob_at_least_one_igso = 4/5) :=
by sorry

end NUMINAMATH_CALUDE_satellite_selection_probabilities_l2215_221530


namespace NUMINAMATH_CALUDE_jerry_shelves_theorem_l2215_221579

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  ((total_books - books_taken) + books_per_shelf - 1) / books_per_shelf

theorem jerry_shelves_theorem :
  shelves_needed 34 7 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelves_theorem_l2215_221579


namespace NUMINAMATH_CALUDE_used_car_seller_problem_l2215_221533

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) :
  num_clients = 9 →
  cars_per_client = 4 →
  selections_per_car = 3 →
  num_clients * cars_per_client = selections_per_car * (num_clients * cars_per_client / selections_per_car) :=
by sorry

end NUMINAMATH_CALUDE_used_car_seller_problem_l2215_221533


namespace NUMINAMATH_CALUDE_units_digit_of_5_pow_150_plus_7_l2215_221549

theorem units_digit_of_5_pow_150_plus_7 : 
  (5^150 + 7) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_5_pow_150_plus_7_l2215_221549


namespace NUMINAMATH_CALUDE_fraction_equality_y_value_l2215_221556

theorem fraction_equality_y_value (a b c d y : ℚ) 
  (h1 : a ≠ b) 
  (h2 : a ≠ 0) 
  (h3 : c ≠ d) 
  (h4 : (b + y) / (a + y) = d / c) : 
  y = (a * d - b * c) / (c - d) := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_y_value_l2215_221556


namespace NUMINAMATH_CALUDE_equation_equivalence_l2215_221537

theorem equation_equivalence (x : ℝ) : x^2 - 10*x - 1 = 0 ↔ (x-5)^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2215_221537


namespace NUMINAMATH_CALUDE_books_from_second_shop_l2215_221548

/- Define the problem parameters -/
def books_shop1 : ℕ := 50
def cost_shop1 : ℕ := 1000
def cost_shop2 : ℕ := 800
def avg_price : ℕ := 20

/- Define the function to calculate the number of books from the second shop -/
def books_shop2 : ℕ :=
  (cost_shop1 + cost_shop2) / avg_price - books_shop1

/- Theorem statement -/
theorem books_from_second_shop :
  books_shop2 = 40 :=
sorry

end NUMINAMATH_CALUDE_books_from_second_shop_l2215_221548


namespace NUMINAMATH_CALUDE_ladder_construction_theorem_l2215_221563

/-- Represents the ladder construction problem --/
def LadderProblem (totalWood rungeLength rungSpacing heightNeeded : ℝ) : Prop :=
  let inchesToFeet : ℝ → ℝ := (· / 12)
  let rungLengthFeet := inchesToFeet rungeLength
  let rungSpacingFeet := inchesToFeet rungSpacing
  let verticalDistanceBetweenRungs := rungLengthFeet + rungSpacingFeet
  let numRungs := heightNeeded / verticalDistanceBetweenRungs
  let woodForRungs := numRungs * rungLengthFeet
  let woodForSides := heightNeeded * 2
  let totalWoodNeeded := woodForRungs + woodForSides
  let remainingWood := totalWood - totalWoodNeeded
  remainingWood = 162.5 ∧ totalWoodNeeded ≤ totalWood

theorem ladder_construction_theorem :
  LadderProblem 300 18 6 50 :=
sorry

end NUMINAMATH_CALUDE_ladder_construction_theorem_l2215_221563


namespace NUMINAMATH_CALUDE_jose_profit_share_l2215_221558

structure Partner where
  investment : ℕ
  duration : ℕ

def totalInvestmentTime (partners : List Partner) : ℕ :=
  partners.foldl (fun acc p => acc + p.investment * p.duration) 0

def profitShare (partner : Partner) (partners : List Partner) (totalProfit : ℕ) : ℚ :=
  (partner.investment * partner.duration : ℚ) / (totalInvestmentTime partners : ℚ) * totalProfit

theorem jose_profit_share :
  let tom : Partner := { investment := 30000, duration := 12 }
  let jose : Partner := { investment := 45000, duration := 10 }
  let angela : Partner := { investment := 60000, duration := 8 }
  let rebecca : Partner := { investment := 75000, duration := 6 }
  let partners : List Partner := [tom, jose, angela, rebecca]
  let totalProfit : ℕ := 72000
  abs (profitShare jose partners totalProfit - 18620.69) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_jose_profit_share_l2215_221558


namespace NUMINAMATH_CALUDE_mollys_age_l2215_221512

theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / (molly_age : ℚ) = 4 / 3 →
  sandy_age + 6 = 30 →
  molly_age = 18 :=
by sorry

end NUMINAMATH_CALUDE_mollys_age_l2215_221512


namespace NUMINAMATH_CALUDE_range_m_f_less_than_one_solution_sets_f_geq_mx_range_m_f_nonnegative_in_interval_l2215_221545

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + m - 1

/-- Theorem for the range of m when f(x) < 1 for all x in ℝ -/
theorem range_m_f_less_than_one :
  ∀ m : ℝ, (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

/-- Theorem for the solution sets of f(x) ≥ (m+1)x -/
theorem solution_sets_f_geq_mx (m : ℝ) :
  (m = -1 ∧ {x : ℝ | x ≥ 1} = {x : ℝ | f m x ≥ (m + 1) * x}) ∨
  (m > -1 ∧ {x : ℝ | x ≤ (m - 1) / (m + 1) ∨ x ≥ 1} = {x : ℝ | f m x ≥ (m + 1) * x}) ∨
  (m < -1 ∧ {x : ℝ | 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)} = {x : ℝ | f m x ≥ (m + 1) * x}) :=
sorry

/-- Theorem for the range of m when f(x) ≥ 0 for all x in [-1/2, 1/2] -/
theorem range_m_f_nonnegative_in_interval :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → f m x ≥ 0) ↔ m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_m_f_less_than_one_solution_sets_f_geq_mx_range_m_f_nonnegative_in_interval_l2215_221545


namespace NUMINAMATH_CALUDE_product_of_sums_l2215_221526

theorem product_of_sums (x : ℝ) (h : (x - 2) * (x + 2) = 2021) : (x - 1) * (x + 1) = 2024 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_l2215_221526


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2215_221590

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : a < 0) 
  (h2 : Set.Ioo (-2 : ℝ) 3 = {x | a * x^2 + b * x + c > 0}) : 
  Set.Ioo (-(1/2) : ℝ) (1/3) = {x | c * x^2 + b * x + a < 0} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2215_221590


namespace NUMINAMATH_CALUDE_scientific_notation_of_41800000000_l2215_221591

theorem scientific_notation_of_41800000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 41800000000 = a * (10 : ℝ) ^ n ∧ a = 4.18 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_41800000000_l2215_221591


namespace NUMINAMATH_CALUDE_eight_N_plus_nine_is_perfect_square_l2215_221503

theorem eight_N_plus_nine_is_perfect_square (n : ℕ) : 
  let N := 2^(4*n + 1) - 4^n - 1
  (∃ k : ℤ, N = 9 * k) → 
  ∃ m : ℕ, 8 * N + 9 = m^2 := by
sorry

end NUMINAMATH_CALUDE_eight_N_plus_nine_is_perfect_square_l2215_221503


namespace NUMINAMATH_CALUDE_fraction_inequality_l2215_221581

theorem fraction_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  1 / a + 4 / (1 - a) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2215_221581
