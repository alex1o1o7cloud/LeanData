import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1382_138222

-- Define a triangle with integer side lengths
structure Triangle :=
  (a b c : ℕ)

-- Define the properties of the triangle
def ValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

def Perimeter (t : Triangle) : ℕ :=
  t.a + t.b + t.c

-- Define the property of the incenter
def IncenterProperty (t : Triangle) : Prop :=
  -- The area of quadrilateral ABCS is 4/5 of the area of triangle ABC
  -- This is equivalent to one side being 1/5 of the perimeter
  t.a = Perimeter t / 5 ∨ t.b = Perimeter t / 5 ∨ t.c = Perimeter t / 5

-- The main theorem
theorem triangle_side_lengths :
  ∀ t : Triangle,
    ValidTriangle t →
    Perimeter t = 15 →
    IncenterProperty t →
    (t = ⟨3, 5, 7⟩ ∨ t = ⟨3, 6, 6⟩ ∨ t = ⟨5, 3, 7⟩ ∨ t = ⟨6, 3, 6⟩ ∨ t = ⟨7, 3, 5⟩ ∨ t = ⟨6, 6, 3⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l1382_138222


namespace NUMINAMATH_CALUDE_system_of_equations_l1382_138289

theorem system_of_equations (a : ℝ) :
  let x := 2 * a + 3
  let y := -a - 2
  (x > 0 ∧ y ≥ 0) →
  ((-3 < a ∧ a ≤ -2) ∧
   (a = -5/3 → x = y) ∧
   (a = -2 → x + y = 5 + a)) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_l1382_138289


namespace NUMINAMATH_CALUDE_remainder_of_2007_pow_2008_mod_10_l1382_138246

theorem remainder_of_2007_pow_2008_mod_10 : 2007^2008 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2007_pow_2008_mod_10_l1382_138246


namespace NUMINAMATH_CALUDE_employees_without_increase_l1382_138286

theorem employees_without_increase (total : ℕ) (salary_percent : ℚ) (travel_percent : ℚ) (both_percent : ℚ) :
  total = 480 →
  salary_percent = 1/10 →
  travel_percent = 1/5 →
  both_percent = 1/20 →
  (total : ℚ) - (salary_percent + travel_percent - both_percent) * total = 360 := by
  sorry

end NUMINAMATH_CALUDE_employees_without_increase_l1382_138286


namespace NUMINAMATH_CALUDE_circumcircle_equation_l1382_138204

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define point P
def P : ℝ × ℝ := (4, 2)

-- Define that P is outside C
axiom P_outside_C : P ∉ C

-- Define that there are two tangent points A and B
axiom tangent_points_exist : ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ A ≠ B

-- Define the circumcircle of triangle ABP
def circumcircle (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 5}

-- Theorem statement
theorem circumcircle_equation (A B : ℝ × ℝ) 
  (h1 : A ∈ C) (h2 : B ∈ C) (h3 : A ≠ B) :
  circumcircle A B = {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 5} :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l1382_138204


namespace NUMINAMATH_CALUDE_maria_candy_l1382_138225

/-- Calculates the remaining candy pieces after eating some. -/
def remaining_candy (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: Maria has 3 pieces of candy left -/
theorem maria_candy : remaining_candy 67 64 = 3 := by
  sorry

end NUMINAMATH_CALUDE_maria_candy_l1382_138225


namespace NUMINAMATH_CALUDE_rational_absolute_value_equality_l1382_138212

theorem rational_absolute_value_equality (a : ℚ) : 
  |(-3 - a)| = 3 + |a| → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_absolute_value_equality_l1382_138212


namespace NUMINAMATH_CALUDE_last_remaining_card_l1382_138248

/-- Represents a playing card --/
inductive Card
  | Joker : Bool → Card  -- True for Big Joker, False for Small Joker
  | Regular : Suit → Rank → Card

/-- Represents the suit of a card --/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- Represents the rank of a card --/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents a deck of cards --/
def Deck := List Card

/-- Creates a standard deck of cards in the specified order --/
def createDeck : Deck := sorry

/-- Combines two decks of cards --/
def combinedDecks : Deck := sorry

/-- Simulates the process of discarding and moving cards --/
def discardAndMove (deck : Deck) : Card := sorry

/-- Theorem stating that the last remaining card is the 6 of Diamonds from the second deck --/
theorem last_remaining_card :
  discardAndMove combinedDecks = Card.Regular Suit.Diamonds Rank.Six := by sorry

end NUMINAMATH_CALUDE_last_remaining_card_l1382_138248


namespace NUMINAMATH_CALUDE_rationalize_sqrt3_minus1_l1382_138208

theorem rationalize_sqrt3_minus1 : 1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt3_minus1_l1382_138208


namespace NUMINAMATH_CALUDE_solution_set_f_plus_x_squared_range_of_m_l1382_138267

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (m : ℝ) (x : ℝ) : ℝ := -|x + 3| + m

-- Theorem 1: Solution set of |x-1| + x^2 - 1 > 0
theorem solution_set_f_plus_x_squared (x : ℝ) : 
  (|x - 1| + x^2 - 1 > 0) ↔ (x > 1 ∨ x < 0) := by sorry

-- Theorem 2: If f(x) < g(x) has a non-empty solution set, then m > 4
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < g m x) → m > 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_plus_x_squared_range_of_m_l1382_138267


namespace NUMINAMATH_CALUDE_total_cookies_l1382_138236

theorem total_cookies (num_people : ℕ) (cookies_per_person : ℕ) 
  (h1 : num_people = 6)
  (h2 : cookies_per_person = 4) :
  num_people * cookies_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l1382_138236


namespace NUMINAMATH_CALUDE_equation_solution_l1382_138244

/-- Given an equation x^4 - 10x^3 - 2(a-11)x^2 + 2(5a+6)x + 2a + a^2 = 0,
    where a is a constant and a ≥ -6, prove the solutions for a and x. -/
theorem equation_solution (a x : ℝ) (h : a ≥ -6) :
  x^4 - 10*x^3 - 2*(a-11)*x^2 + 2*(5*a+6)*x + 2*a + a^2 = 0 →
  ((a = x^2 - 4*x - 2) ∨ (a = x^2 - 6*x)) ∧
  ((∃ (i : Fin 2), x = 2 + (-1)^(i : ℕ) * Real.sqrt (a + 6)) ∨
   (∃ (i : Fin 2), x = 3 + (-1)^(i : ℕ) * Real.sqrt (a + 9))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1382_138244


namespace NUMINAMATH_CALUDE_g_value_at_4_l1382_138255

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧  -- g is a cubic polynomial
  (g 0 = -2) ∧  -- g(0) = -2
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2)  -- roots of g are squares of roots of f

-- Theorem statement
theorem g_value_at_4 (g : ℝ → ℝ) (h : g_properties g) : g 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_4_l1382_138255


namespace NUMINAMATH_CALUDE_apple_packing_difference_is_500_l1382_138218

/-- Represents the apple packing scenario over two weeks -/
structure ApplePacking where
  apples_per_box : ℕ
  boxes_per_day : ℕ
  days_per_week : ℕ
  total_apples_two_weeks : ℕ

/-- Calculates the difference in daily apple packing between the first and second week -/
def daily_packing_difference (ap : ApplePacking) : ℕ :=
  let normal_daily_packing := ap.apples_per_box * ap.boxes_per_day
  let first_week_total := normal_daily_packing * ap.days_per_week
  let second_week_total := ap.total_apples_two_weeks - first_week_total
  let second_week_daily_average := second_week_total / ap.days_per_week
  normal_daily_packing - second_week_daily_average

/-- Theorem stating the difference in daily apple packing is 500 -/
theorem apple_packing_difference_is_500 :
  ∀ (ap : ApplePacking),
    ap.apples_per_box = 40 ∧
    ap.boxes_per_day = 50 ∧
    ap.days_per_week = 7 ∧
    ap.total_apples_two_weeks = 24500 →
    daily_packing_difference ap = 500 := by
  sorry

end NUMINAMATH_CALUDE_apple_packing_difference_is_500_l1382_138218


namespace NUMINAMATH_CALUDE_circle_diameter_l1382_138241

theorem circle_diameter (r : ℝ) (h : r = 4) : 2 * r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l1382_138241


namespace NUMINAMATH_CALUDE_smallest_assembly_size_l1382_138200

theorem smallest_assembly_size : ∃ n : ℕ, n > 50 ∧ 
  (∃ x : ℕ, n = 4 * x + (x + 2)) ∧ 
  (∀ m : ℕ, m > 50 → (∃ y : ℕ, m = 4 * y + (y + 2)) → m ≥ n) ∧
  n = 52 :=
by sorry

end NUMINAMATH_CALUDE_smallest_assembly_size_l1382_138200


namespace NUMINAMATH_CALUDE_john_payment_l1382_138258

def hearing_aid_cost : ℝ := 2500
def insurance_coverage_percent : ℝ := 80
def number_of_hearing_aids : ℕ := 2

theorem john_payment (total_cost : ℝ) (insurance_payment : ℝ) (john_payment : ℝ) :
  total_cost = hearing_aid_cost * number_of_hearing_aids →
  insurance_payment = (insurance_coverage_percent / 100) * total_cost →
  john_payment = total_cost - insurance_payment →
  john_payment = 1000 := by sorry

end NUMINAMATH_CALUDE_john_payment_l1382_138258


namespace NUMINAMATH_CALUDE_sqrt_abs_equation_l1382_138239

theorem sqrt_abs_equation (a b : ℤ) :
  (Real.sqrt (a - 2023 : ℝ) + |b + 2023| - 1 = 0) → (a + b = 1 ∨ a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_equation_l1382_138239


namespace NUMINAMATH_CALUDE_probability_one_good_one_inferior_l1382_138252

/-- The probability of drawing one good quality bulb and one inferior quality bulb -/
theorem probability_one_good_one_inferior (total : ℕ) (good : ℕ) (inferior : ℕ) :
  total = 6 →
  good = 4 →
  inferior = 2 →
  (good + inferior : ℚ) / total * inferior / total + inferior / total * good / total = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_good_one_inferior_l1382_138252


namespace NUMINAMATH_CALUDE_lcm_three_integers_l1382_138242

theorem lcm_three_integers (A₁ A₂ A₃ : ℤ) :
  let D := Int.gcd (A₁ * A₂) (Int.gcd (A₂ * A₃) (A₃ * A₁))
  Int.lcm A₁ (Int.lcm A₂ A₃) = (A₁ * A₂ * A₃) / D :=
by sorry

end NUMINAMATH_CALUDE_lcm_three_integers_l1382_138242


namespace NUMINAMATH_CALUDE_generalized_inequality_l1382_138260

theorem generalized_inequality (x : ℝ) (n : ℕ) (h : x > 0) :
  x^n + n/x > n + 1 := by sorry

end NUMINAMATH_CALUDE_generalized_inequality_l1382_138260


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l1382_138214

theorem largest_prime_factor_of_1729 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l1382_138214


namespace NUMINAMATH_CALUDE_tray_height_is_seven_l1382_138272

/-- Represents the dimensions of the rectangular paper --/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Represents the parameters of the cuts made on the paper --/
structure CutParameters where
  distance_from_corner : ℝ
  angle : ℝ

/-- Calculates the height of the tray formed from the paper --/
def tray_height (paper : PaperDimensions) (cut : CutParameters) : ℝ :=
  sorry

/-- Theorem stating that the height of the tray is 7 for the given parameters --/
theorem tray_height_is_seven :
  let paper := PaperDimensions.mk 150 100
  let cut := CutParameters.mk 7 (π / 4)
  tray_height paper cut = 7 := by
  sorry

end NUMINAMATH_CALUDE_tray_height_is_seven_l1382_138272


namespace NUMINAMATH_CALUDE_total_cost_proof_l1382_138280

def flower_cost : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

theorem total_cost_proof :
  (roses_bought + daisies_bought) * flower_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l1382_138280


namespace NUMINAMATH_CALUDE_least_product_of_two_primes_above_50_l1382_138270

theorem least_product_of_two_primes_above_50 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧ 
    p ≠ q ∧
    p > 50 ∧ q > 50 ∧
    p * q = 3127 ∧
    ∀ r s : ℕ, r.Prime → s.Prime → r ≠ s → r > 50 → s > 50 → r * s ≥ 3127 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_two_primes_above_50_l1382_138270


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1382_138291

theorem rationalize_denominator : 35 / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1382_138291


namespace NUMINAMATH_CALUDE_rational_roots_quadratic_l1382_138217

theorem rational_roots_quadratic (r : ℚ) : 
  (∃ (n : ℤ), 2 * r = n) →
  (∃ (x : ℚ), (r^2 + r) * x^2 + 4 - r^2 = 0) →
  r = 2 ∨ r = -2 ∨ r = -4 := by
sorry

end NUMINAMATH_CALUDE_rational_roots_quadratic_l1382_138217


namespace NUMINAMATH_CALUDE_max_mn_for_exponential_intersection_max_mn_achieved_l1382_138288

/-- The maximum value of mn for a line mx + ny = 1 that intersects
    the graph of y = a^(x-1) at a fixed point, where a > 0 and a ≠ 1 -/
theorem max_mn_for_exponential_intersection (a : ℝ) (m n : ℝ) 
  (ha : a > 0) (ha_ne_one : a ≠ 1) : 
  (∃ (x y : ℝ), y = a^(x-1) ∧ m*x + n*y = 1) → m*n ≤ 1/4 := by
  sorry

/-- The maximum value of mn is achieved when m = n = 1/2 -/
theorem max_mn_achieved (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∃ (m n : ℝ), m*n = 1/4 ∧ 
  (∃ (x y : ℝ), y = a^(x-1) ∧ m*x + n*y = 1) := by
  sorry

end NUMINAMATH_CALUDE_max_mn_for_exponential_intersection_max_mn_achieved_l1382_138288


namespace NUMINAMATH_CALUDE_outlets_per_room_l1382_138202

theorem outlets_per_room (total_rooms : ℕ) (total_outlets : ℕ) (h1 : total_rooms = 7) (h2 : total_outlets = 42) :
  total_outlets / total_rooms = 6 := by
  sorry

end NUMINAMATH_CALUDE_outlets_per_room_l1382_138202


namespace NUMINAMATH_CALUDE_equation_solution_l1382_138296

theorem equation_solution : ∃! x : ℝ, (x^2 + x)^2 + Real.sqrt (x^2 - 1) = 0 ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1382_138296


namespace NUMINAMATH_CALUDE_triangle_inequality_l1382_138232

/-- Theorem: For any triangle ABC, the sum of square roots of specific ratios involving side lengths, altitude, and inradius is less than or equal to 3/4. -/
theorem triangle_inequality (a b c h_a r : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_pos_h_a : 0 < h_a) (h_pos_r : 0 < r) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let f (x y z w v) := Real.sqrt (x * (w - 2 * v) / ((3 * x + y + z) * (w + 2 * v)))
  (f a b c h_a r) + (f b c a h_a r) + (f c a b h_a r) ≤ 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1382_138232


namespace NUMINAMATH_CALUDE_sum_of_digits_base7_squared_expectation_l1382_138205

/-- Sum of digits in base 7 -/
def sum_of_digits_base7 (n : ℕ) : ℕ :=
  sorry

/-- Expected value of a function over a finite range -/
def expected_value {α : Type*} (f : α → ℝ) (range : Finset α) : ℝ :=
  sorry

theorem sum_of_digits_base7_squared_expectation :
  expected_value (λ n => (sum_of_digits_base7 n)^2) (Finset.range (7^20)) = 3680 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_base7_squared_expectation_l1382_138205


namespace NUMINAMATH_CALUDE_alternate_shading_six_by_six_l1382_138257

theorem alternate_shading_six_by_six (grid_size : Nat) (shaded_squares : Nat) :
  grid_size = 6 → shaded_squares = 18 → (shaded_squares : ℚ) / (grid_size * grid_size) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_alternate_shading_six_by_six_l1382_138257


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1382_138285

/-- A quadratic equation with coefficients m and n has exactly one real root if and only if m > 0 and n = 9m^2 -/
theorem quadratic_one_root (m n : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + n = 0) ∧ (m > 0) ∧ (n > 0) ↔ (m > 0) ∧ (n = 9*m^2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1382_138285


namespace NUMINAMATH_CALUDE_triangle_area_l1382_138295

/-- The area of a triangle with base 2t and height 3t + 2, where t = 6 -/
theorem triangle_area (t : ℝ) (h : t = 6) : (1/2 : ℝ) * (2*t) * (3*t + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1382_138295


namespace NUMINAMATH_CALUDE_total_spent_is_2100_l1382_138290

/-- Calculates the total amount spent on a computer setup -/
def total_spent (computer_cost monitor_peripheral_ratio original_video_card_cost new_video_card_ratio : ℚ) : ℚ :=
  computer_cost + 
  (monitor_peripheral_ratio * computer_cost) + 
  (new_video_card_ratio * original_video_card_cost - original_video_card_cost)

/-- Proves that the total amount spent is $2100 given the specified costs and ratios -/
theorem total_spent_is_2100 : 
  total_spent 1500 (1/5) 300 2 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_2100_l1382_138290


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l1382_138224

/-- The total number of technical personnel --/
def total_personnel : ℕ := 37

/-- The number of attendees --/
def n : ℕ := 18

/-- Proves that n satisfies the conditions of the systematic sampling problem --/
theorem systematic_sampling_proof :
  (total_personnel - 1) % n = 0 ∧ 
  (total_personnel - 3) % (n - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l1382_138224


namespace NUMINAMATH_CALUDE_negation_of_all_nonnegative_squares_l1382_138293

theorem negation_of_all_nonnegative_squares (p : Prop) : 
  (p ↔ ∀ x : ℝ, x^2 ≥ 0) → (¬p ↔ ∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_nonnegative_squares_l1382_138293


namespace NUMINAMATH_CALUDE_water_bottles_count_l1382_138207

theorem water_bottles_count (water_bottles : ℕ) (apple_bottles : ℕ) : 
  apple_bottles = water_bottles + 6 →
  water_bottles + apple_bottles = 54 →
  water_bottles = 24 := by
sorry

end NUMINAMATH_CALUDE_water_bottles_count_l1382_138207


namespace NUMINAMATH_CALUDE_parallelogram_angles_l1382_138219

/-- A parallelogram with angles measured in degrees -/
structure Parallelogram where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_360 : A + B + C + D = 360
  opposite_equal_AC : A = C
  opposite_equal_BD : B = D

/-- Theorem: In a parallelogram ABCD where angle A measures 125°, 
    the measures of angles B, C, and D are 55°, 125°, and 55° respectively. -/
theorem parallelogram_angles (p : Parallelogram) (h : p.A = 125) : 
  p.B = 55 ∧ p.C = 125 ∧ p.D = 55 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_angles_l1382_138219


namespace NUMINAMATH_CALUDE_contractor_absent_days_l1382_138210

/-- Proves that given the specified contract conditions, the number of days absent is 10 -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (payment_per_day : ℚ) 
  (fine_per_day : ℚ) 
  (total_amount : ℚ) : 
  total_days = 30 ∧ 
  payment_per_day = 25 ∧ 
  fine_per_day = 7.5 ∧ 
  total_amount = 425 → 
  ∃ (days_worked : ℕ) (days_absent : ℕ), 
    days_worked + days_absent = total_days ∧ 
    days_absent = 10 ∧
    (payment_per_day * days_worked : ℚ) - (fine_per_day * days_absent : ℚ) = total_amount :=
by sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l1382_138210


namespace NUMINAMATH_CALUDE_max_expectation_exp_l1382_138274

-- Define a random variable X
variable (X : ℝ → ℝ)

-- Define probability measure P
variable (P : Set ℝ → ℝ)

-- Define expectation E
variable (E : (ℝ → ℝ) → ℝ)

-- Define variance D
variable (D : (ℝ → ℝ) → ℝ)

-- Constants σ and b
variable (σ b : ℝ)

-- Conditions
variable (h1 : P {x | |X x| ≤ b} = 1)
variable (h2 : E X = 0)
variable (h3 : D X = σ^2)
variable (h4 : σ > 0)
variable (h5 : b > 0)

-- Theorem statement
theorem max_expectation_exp :
  (∀ Y : ℝ → ℝ, P {x | |Y x| ≤ b} = 1 → E Y = 0 → D Y = σ^2 →
    E (fun x => Real.exp (Y x)) ≤ (Real.exp b * σ^2 + Real.exp (-σ^2 / b) * b^2) / (σ^2 + b^2)) ∧
  (E (fun x => Real.exp (X x)) = (Real.exp b * σ^2 + Real.exp (-σ^2 / b) * b^2) / (σ^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_max_expectation_exp_l1382_138274


namespace NUMINAMATH_CALUDE_nails_per_plank_l1382_138211

theorem nails_per_plank (total_planks : ℕ) (total_nails : ℕ) 
  (h1 : total_planks = 16) (h2 : total_nails = 32) : 
  total_nails / total_planks = 2 := by
  sorry

end NUMINAMATH_CALUDE_nails_per_plank_l1382_138211


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_twice_C_x_plus_y_equals_76_l1382_138228

-- Define the angles
def angle_A : ℝ := 34
def angle_B : ℝ := 80
def angle_C : ℝ := 38

-- Define x and y as real numbers (representing angle measures)
variable (x y : ℝ)

-- State the theorem
theorem sum_of_x_and_y_equals_twice_C :
  x + y = 2 * angle_C := by sorry

-- Prove that x + y equals 76
theorem x_plus_y_equals_76 :
  x + y = 76 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_twice_C_x_plus_y_equals_76_l1382_138228


namespace NUMINAMATH_CALUDE_yellow_to_red_ratio_l1382_138277

/-- Represents the number of marbles Beth has initially -/
def total_marbles : ℕ := 72

/-- Represents the number of colors of marbles -/
def num_colors : ℕ := 3

/-- Represents the number of red marbles Beth loses -/
def red_marbles_lost : ℕ := 5

/-- Represents the number of marbles Beth has left after losing some of each color -/
def marbles_left : ℕ := 42

/-- Theorem stating the ratio of yellow marbles lost to red marbles lost -/
theorem yellow_to_red_ratio :
  let initial_per_color := total_marbles / num_colors
  let blue_marbles_lost := 2 * red_marbles_lost
  let yellow_marbles_lost := initial_per_color - (marbles_left - (2 * initial_per_color - red_marbles_lost - blue_marbles_lost))
  yellow_marbles_lost / red_marbles_lost = 3 := by sorry

end NUMINAMATH_CALUDE_yellow_to_red_ratio_l1382_138277


namespace NUMINAMATH_CALUDE_shift_increasing_interval_l1382_138297

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem shift_increasing_interval :
  IncreasingOn f (-2) 3 → IncreasingOn (fun x ↦ f (x + 4)) (-6) (-1) := by
  sorry

end NUMINAMATH_CALUDE_shift_increasing_interval_l1382_138297


namespace NUMINAMATH_CALUDE_hexagon_vertex_recovery_erased_vertex_recoverable_l1382_138282

/-- Represents a hexagon with numbers on its vertices -/
structure Hexagon where
  -- Vertex numbers
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Theorem: Any vertex number in a hexagon can be determined from the other five -/
theorem hexagon_vertex_recovery (h : Hexagon) :
  h.a = h.b + h.d + h.f - h.c - h.e :=
by sorry

/-- Corollary: It's possible to recover an erased vertex number in the hexagon -/
theorem erased_vertex_recoverable (h : Hexagon) :
  ∃ (x : ℝ), x = h.b + h.d + h.f - h.c - h.e :=
by sorry

end NUMINAMATH_CALUDE_hexagon_vertex_recovery_erased_vertex_recoverable_l1382_138282


namespace NUMINAMATH_CALUDE_condition_equivalent_to_inequality_l1382_138230

-- Define an increasing function on ℝ
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem condition_equivalent_to_inequality
  (f : ℝ → ℝ) (h : IncreasingFunction f) :
  (∀ a b : ℝ, a + b > 0 ↔ f a + f b > f (-a) + f (-b)) :=
by sorry

end NUMINAMATH_CALUDE_condition_equivalent_to_inequality_l1382_138230


namespace NUMINAMATH_CALUDE_bus_row_capacity_l1382_138299

/-- Represents a school bus with a given number of rows and total capacity. -/
structure SchoolBus where
  rows : ℕ
  totalCapacity : ℕ

/-- Calculates the capacity of each row in the school bus. -/
def rowCapacity (bus : SchoolBus) : ℕ :=
  bus.totalCapacity / bus.rows

/-- Theorem stating that for a bus with 20 rows and a total capacity of 80,
    the capacity of each row is 4. -/
theorem bus_row_capacity :
  let bus : SchoolBus := { rows := 20, totalCapacity := 80 }
  rowCapacity bus = 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_row_capacity_l1382_138299


namespace NUMINAMATH_CALUDE_largest_square_area_l1382_138259

-- Define the right triangle XYZ
structure RightTriangle where
  xy : ℝ  -- length of side XY
  xz : ℝ  -- length of side XZ
  yz : ℝ  -- length of hypotenuse YZ
  right_angle : xy^2 + xz^2 = yz^2  -- Pythagorean theorem

-- Define the theorem
theorem largest_square_area (t : RightTriangle) 
  (sum_of_squares : t.xy^2 + t.xz^2 + t.yz^2 = 450) :
  t.yz^2 = 225 := by
  sorry


end NUMINAMATH_CALUDE_largest_square_area_l1382_138259


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l1382_138279

theorem arithmetic_series_sum : 
  ∀ (a l d n : ℤ),
  a = -41 →
  l = 1 →
  d = 2 →
  n = (l - a) / d + 1 →
  (n : ℚ) / 2 * (a + l) = -440 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l1382_138279


namespace NUMINAMATH_CALUDE_cat_count_l1382_138243

/-- The number of cats that can jump -/
def jump : ℕ := 45

/-- The number of cats that can fetch -/
def fetch : ℕ := 25

/-- The number of cats that can meow -/
def meow : ℕ := 40

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 15

/-- The number of cats that can fetch and meow -/
def fetch_meow : ℕ := 20

/-- The number of cats that can jump and meow -/
def jump_meow : ℕ := 23

/-- The number of cats that can do all three tricks -/
def all_three : ℕ := 10

/-- The number of cats that can do no tricks -/
def no_tricks : ℕ := 5

/-- The total number of cats in the training center -/
def total_cats : ℕ := 67

theorem cat_count : 
  jump + fetch + meow - jump_fetch - fetch_meow - jump_meow + all_three + no_tricks = total_cats :=
by sorry

end NUMINAMATH_CALUDE_cat_count_l1382_138243


namespace NUMINAMATH_CALUDE_union_equal_iff_x_zero_l1382_138226

def A (x : ℝ) : Set ℝ := {0, Real.exp x}
def B : Set ℝ := {-1, 0, 1}

theorem union_equal_iff_x_zero (x : ℝ) : A x ∪ B = B ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_union_equal_iff_x_zero_l1382_138226


namespace NUMINAMATH_CALUDE_tetris_arrangement_exists_l1382_138294

/-- Represents a Tetris piece type -/
inductive TetrisPiece
  | O | I | T | S | Z | L | J

/-- Represents a position on the 6x6 grid -/
structure Position where
  x : Fin 6
  y : Fin 6

/-- Represents a placed Tetris piece on the grid -/
structure PlacedPiece where
  piece : TetrisPiece
  positions : List Position

/-- Checks if a list of placed pieces forms a valid arrangement -/
def isValidArrangement (pieces : List PlacedPiece) : Prop :=
  -- Each position on the 6x6 grid is covered exactly once
  ∀ (x y : Fin 6), ∃! p : PlacedPiece, p ∈ pieces ∧ Position.mk x y ∈ p.positions

/-- Checks if all piece types are used at least once -/
def allPiecesUsed (pieces : List PlacedPiece) : Prop :=
  ∀ t : TetrisPiece, ∃ p : PlacedPiece, p ∈ pieces ∧ p.piece = t

/-- Main theorem: There exists a valid arrangement of Tetris pieces -/
theorem tetris_arrangement_exists : 
  ∃ (pieces : List PlacedPiece), isValidArrangement pieces ∧ allPiecesUsed pieces :=
sorry

end NUMINAMATH_CALUDE_tetris_arrangement_exists_l1382_138294


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l1382_138220

/-- Definition of equivalent rational number pair -/
def is_equivalent_pair (m n : ℚ) : Prop := m + n = m * n

/-- Part 1: Prove that (3, 3/2) is an equivalent rational number pair -/
theorem part_one : is_equivalent_pair 3 (3/2) := by sorry

/-- Part 2: If (x+1, 4) is an equivalent rational number pair, then x = 1/3 -/
theorem part_two (x : ℚ) : is_equivalent_pair (x + 1) 4 → x = 1/3 := by sorry

/-- Part 3: If (m, n) is an equivalent rational number pair, 
    then 12 - 6mn + 6m + 6n = 12 -/
theorem part_three (m n : ℚ) : 
  is_equivalent_pair m n → 12 - 6*m*n + 6*m + 6*n = 12 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l1382_138220


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1382_138240

/-- Given a point P in polar coordinates, find its symmetric point with respect to the pole -/
theorem symmetric_point_coordinates (r : ℝ) (θ : ℝ) :
  let P : ℝ × ℝ := (r, θ)
  let symmetric_polar : ℝ × ℝ := (r, θ + π)
  let symmetric_cartesian : ℝ × ℝ := (r * Real.cos (θ + π), r * Real.sin (θ + π))
  P = (2, -5 * π / 3) →
  symmetric_polar = (2, -2 * π / 3) ∧
  symmetric_cartesian = (-1, -Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1382_138240


namespace NUMINAMATH_CALUDE_smallest_sum_of_five_relatively_prime_numbers_l1382_138227

/-- A function that checks if two natural numbers are relatively prime -/
def isRelativelyPrime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- A function that checks if a list of natural numbers are pairwise relatively prime -/
def arePairwiseRelativelyPrime (list : List ℕ) : Prop :=
  ∀ (i j : Fin list.length), i.val < j.val → isRelativelyPrime (list.get i) (list.get j)

/-- The main theorem statement -/
theorem smallest_sum_of_five_relatively_prime_numbers :
  ∃ (list : List ℕ),
    list.length = 5 ∧
    arePairwiseRelativelyPrime list ∧
    (∀ (sum : ℕ),
      (∃ (other_list : List ℕ),
        other_list.length = 5 ∧
        arePairwiseRelativelyPrime other_list ∧
        sum = other_list.sum) →
      list.sum ≤ sum) ∧
    list.sum = 4 :=
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_five_relatively_prime_numbers_l1382_138227


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1382_138233

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 25| + |x - 15| = |2*x - 40| :=
by
  -- The unique solution is x = 20
  use 20
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1382_138233


namespace NUMINAMATH_CALUDE_power_division_equality_l1382_138215

theorem power_division_equality : 8^15 / 64^6 = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l1382_138215


namespace NUMINAMATH_CALUDE_correct_amount_paid_l1382_138223

/-- The amount paid by Mr. Doré given the costs of items and change received -/
def amount_paid (pants_cost shirt_cost tie_cost change : ℕ) : ℕ :=
  pants_cost + shirt_cost + tie_cost + change

/-- Theorem stating that the amount paid is correct given the problem conditions -/
theorem correct_amount_paid :
  amount_paid 140 43 15 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_correct_amount_paid_l1382_138223


namespace NUMINAMATH_CALUDE_hyperbola_parameter_l1382_138271

/-- Proves that given a hyperbola with specific properties, the parameter a equals √10 -/
theorem hyperbola_parameter (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ k : ℝ, ∀ x : ℝ, k * x^2 + 1 = (a/b * x)^2) →  -- Asymptotes tangent to parabola
  (∀ x y : ℝ, x^2 + (y - a)^2 = 1 → 
    ∃ x₁ y₁ x₂ y₂ : ℝ, 
      x₁^2 + (y₁ - a)^2 = 1 ∧ 
      x₂^2 + (y₂ - a)^2 = 1 ∧
      y₁^2 / a^2 - x₁^2 / b^2 = 1 ∧
      y₂^2 / a^2 - x₂^2 / b^2 = 1 ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2) →  -- Chord length condition
  a = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_l1382_138271


namespace NUMINAMATH_CALUDE_sticker_enlargement_l1382_138235

/-- Given a rectangle with original width and height, and a new width,
    calculate the new height when enlarged proportionately -/
def new_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Theorem stating that a 3x2 inch rectangle enlarged to 12 inches wide
    will be 8 inches tall -/
theorem sticker_enlargement :
  new_height 3 2 12 = 8 := by sorry

end NUMINAMATH_CALUDE_sticker_enlargement_l1382_138235


namespace NUMINAMATH_CALUDE_negation_equivalence_l1382_138266

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1382_138266


namespace NUMINAMATH_CALUDE_hyperbola_area_ratio_l1382_138292

noncomputable def hyperbola_ratio (a b : ℝ) (F₁ F₂ A B : ℝ × ℝ) : Prop :=
  let x := λ p : ℝ × ℝ => p.1
  let y := λ p : ℝ × ℝ => p.2
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((x p - x q)^2 + (y p - y q)^2)
  let area := λ p q r : ℝ × ℝ => abs ((x q - x p) * (y r - y p) - (x r - x p) * (y q - y p)) / 2
  (∀ p : ℝ × ℝ, (x p)^2 / a^2 - (y p)^2 / b^2 = 1 → 
    (x p - x F₁) * (x p - x F₂) + (y p - y F₁) * (y p - y F₂) = a^2 - b^2) ∧
  (x F₁ = -Real.sqrt (a^2 + b^2) ∧ y F₁ = 0) ∧
  (x F₂ = Real.sqrt (a^2 + b^2) ∧ y F₂ = 0) ∧
  ((x A)^2 / a^2 - (y A)^2 / b^2 = 1) ∧
  ((x B)^2 / a^2 - (y B)^2 / b^2 = 1) ∧
  (y B - y A) * (x A - x F₁) = (x B - x A) * (y A - y F₁) ∧
  dist A F₁ / dist A F₂ = 1/2 →
  area A F₁ F₂ / area A B F₂ = 4/9

theorem hyperbola_area_ratio : 
  hyperbola_ratio 3 4 (-5, 0) (5, 0) (-27/5, 8*Real.sqrt 14/5) (0, 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_area_ratio_l1382_138292


namespace NUMINAMATH_CALUDE_parabola_no_real_roots_l1382_138298

def parabola (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem parabola_no_real_roots :
  ∀ x : ℝ, parabola x ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_no_real_roots_l1382_138298


namespace NUMINAMATH_CALUDE_inequality_proof_l1382_138221

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq : a + b = c + d) (prod_gt : a * b > c * d) : 
  (Real.sqrt a + Real.sqrt b > Real.sqrt c + Real.sqrt d) ∧ 
  (|a - b| < |c - d|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1382_138221


namespace NUMINAMATH_CALUDE_real_number_inequalities_l1382_138264

theorem real_number_inequalities (a b c : ℝ) : 
  (a > b → a > (a + b) / 2 ∧ (a + b) / 2 > b) ∧
  (a > b ∧ b > 0 → a > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > b) ∧
  (a > b ∧ b > 0 ∧ c > 0 → (b + c) / (a + c) > b / a) :=
by sorry

end NUMINAMATH_CALUDE_real_number_inequalities_l1382_138264


namespace NUMINAMATH_CALUDE_train_crossing_time_l1382_138287

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 133.33333333333334 →
  train_speed_kmh = 60 →
  crossing_time = 8 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1382_138287


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l1382_138237

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  elderly_population : ℕ
  young_population : ℕ
  young_sample : ℕ
  (elderly_population_le_total : elderly_population ≤ total_population)
  (young_population_le_total : young_population ≤ total_population)
  (young_sample_le_young_population : young_sample ≤ young_population)

/-- Calculates the number of elderly in the sample based on stratified sampling -/
def elderly_in_sample (s : StratifiedSample) : ℚ :=
  s.elderly_population * (s.young_sample : ℚ) / s.young_population

/-- Theorem stating the result of the stratified sampling problem -/
theorem stratified_sampling_result (s : StratifiedSample) 
  (h_total : s.total_population = 430)
  (h_elderly : s.elderly_population = 90)
  (h_young : s.young_population = 160)
  (h_young_sample : s.young_sample = 32) :
  elderly_in_sample s = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l1382_138237


namespace NUMINAMATH_CALUDE_cookie_bags_l1382_138269

theorem cookie_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) :
  total_cookies / cookies_per_bag = 37 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_l1382_138269


namespace NUMINAMATH_CALUDE_x_over_y_equals_four_l1382_138275

theorem x_over_y_equals_four (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 2 * Real.log (x - 2*y) = Real.log x + Real.log y) : x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_equals_four_l1382_138275


namespace NUMINAMATH_CALUDE_men_per_table_l1382_138238

theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 9 →
  women_per_table = 7 →
  total_customers = 90 →
  ∃ (men_per_table : ℕ), 
    men_per_table * num_tables + women_per_table * num_tables = total_customers ∧
    men_per_table = 3 :=
by sorry

end NUMINAMATH_CALUDE_men_per_table_l1382_138238


namespace NUMINAMATH_CALUDE_prime_iff_divides_factorial_plus_one_l1382_138249

theorem prime_iff_divides_factorial_plus_one (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ n ∣ (Nat.factorial (n - 1) + 1) :=
sorry

end NUMINAMATH_CALUDE_prime_iff_divides_factorial_plus_one_l1382_138249


namespace NUMINAMATH_CALUDE_inequality_proof_l1382_138278

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 1) (h3 : -1 < c) (h4 : c < 0) :
  b * Real.log |c| / Real.log a > a * Real.log |c| / Real.log b :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1382_138278


namespace NUMINAMATH_CALUDE_max_students_above_median_l1382_138229

theorem max_students_above_median (n : ℕ) (h : n = 81) :
  (n + 1) / 2 = (n + 1) / 2 ∧ (n - (n + 1) / 2) = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_students_above_median_l1382_138229


namespace NUMINAMATH_CALUDE_k_range_theorem_l1382_138209

/-- The range of k given the conditions in the problem -/
def k_range : Set ℝ := Set.Iic 0 ∪ Set.Ioo (1/2) (5/2)

/-- p: the function y=kx+1 is increasing on ℝ -/
def p (k : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ + 1 < k * x₂ + 1

/-- q: the equation x^2+(2k-3)x+1=0 has real solutions -/
def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + (2*k - 3)*x + 1 = 0

/-- Main theorem stating the range of k -/
theorem k_range_theorem (h1 : ∀ k : ℝ, ¬(p k ∧ q k)) (h2 : ∀ k : ℝ, p k ∨ q k) : 
  ∀ k : ℝ, k ∈ k_range ↔ (p k ∨ q k) :=
sorry

end NUMINAMATH_CALUDE_k_range_theorem_l1382_138209


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l1382_138203

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 13*n + 36 ≤ 0 ∧
  n = 9 ∧
  ∀ (m : ℤ), m^2 - 13*m + 36 ≤ 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l1382_138203


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1382_138253

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the set M
def M : Set Nat := {1, 3, 5}

-- Theorem stating that the complement of M in U is {2, 4, 6}
theorem complement_of_M_in_U :
  (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1382_138253


namespace NUMINAMATH_CALUDE_max_value_p_l1382_138256

theorem max_value_p (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b * c + a + c = b) : 
  let p := 2 / (1 + a^2) - 2 / (1 + b^2) + 3 / (1 + c^2)
  ∃ (max_p : ℝ), max_p = 10/3 ∧ p ≤ max_p := by
  sorry

end NUMINAMATH_CALUDE_max_value_p_l1382_138256


namespace NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_angle_proof_by_contradiction_uses_correct_assumption_l1382_138261

/-- A triangle has at most one obtuse angle -/
theorem triangle_at_most_one_obtuse_angle : 
  ∀ (T : Type) (is_triangle : T → Prop) (is_obtuse_angle : T → T → Prop),
  (∀ t : T, is_triangle t → 
    ∃! a : T, is_obtuse_angle t a) :=
by
  sorry

/-- The correct assumption for proof by contradiction of the above theorem -/
def contradiction_assumption (T : Type) (is_triangle : T → Prop) (is_obtuse_angle : T → T → Prop) : Prop :=
  ∃ t : T, is_triangle t ∧ ∃ a b : T, a ≠ b ∧ is_obtuse_angle t a ∧ is_obtuse_angle t b

/-- The proof by contradiction uses the correct assumption -/
theorem proof_by_contradiction_uses_correct_assumption :
  ∀ (T : Type) (is_triangle : T → Prop) (is_obtuse_angle : T → T → Prop),
  ¬(contradiction_assumption T is_triangle is_obtuse_angle) →
  (∀ t : T, is_triangle t → 
    ∃! a : T, is_obtuse_angle t a) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_angle_proof_by_contradiction_uses_correct_assumption_l1382_138261


namespace NUMINAMATH_CALUDE_ordered_pairs_satisfying_equation_l1382_138284

theorem ordered_pairs_satisfying_equation : 
  ∃! (n : ℕ), n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      let a := p.1
      let b := p.2
      a > 0 ∧ b > 0 ∧ 
      a * b + 80 = 15 * Nat.lcm a b + 10 * Nat.gcd a b)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ordered_pairs_satisfying_equation_l1382_138284


namespace NUMINAMATH_CALUDE_isosceles_triangle_count_l1382_138213

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point

/-- Represents the set of all colored points in the triangle -/
def ColoredPoints (t : EquilateralTriangle) : Set Point :=
  sorry

/-- Checks if a triangle formed by three points is isosceles -/
def IsIsosceles (p1 p2 p3 : Point) : Prop :=
  sorry

/-- Counts the number of isosceles triangles formed by the colored points -/
def CountIsoscelesTriangles (t : EquilateralTriangle) : ℕ :=
  sorry

/-- Main theorem: There are exactly 18 isosceles triangles with vertices at the colored points -/
theorem isosceles_triangle_count (t : EquilateralTriangle) :
  CountIsoscelesTriangles t = 18 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_count_l1382_138213


namespace NUMINAMATH_CALUDE_specific_prism_surface_area_l1382_138276

/-- A right prism with an isosceles trapezoid base -/
structure RightPrism where
  AB : ℝ
  BC : ℝ
  AD : ℝ
  diagonal_cross_section_area : ℝ

/-- The total surface area of the right prism -/
def total_surface_area (p : RightPrism) : ℝ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the total surface area of the specific prism -/
theorem specific_prism_surface_area :
  ∃ (p : RightPrism),
    p.AB = 13 ∧
    p.BC = 11 ∧
    p.AD = 21 ∧
    p.diagonal_cross_section_area = 180 ∧
    total_surface_area p = 906 := by
  sorry

end NUMINAMATH_CALUDE_specific_prism_surface_area_l1382_138276


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_integers_with_five_l1382_138245

def is_divisible (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

theorem largest_divisor_of_consecutive_integers_with_five (n : ℤ) :
  (is_divisible n 5 ∨ is_divisible (n + 1) 5 ∨ is_divisible (n + 2) 5) →
  is_divisible (n * (n + 1) * (n + 2)) 15 ∧
  ∀ m : ℤ, m > 15 → ¬(∀ k : ℤ, (is_divisible k 5 ∨ is_divisible (k + 1) 5 ∨ is_divisible (k + 2) 5) →
                              is_divisible (k * (k + 1) * (k + 2)) m) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_integers_with_five_l1382_138245


namespace NUMINAMATH_CALUDE_operation_2012_equals_55_l1382_138251

def operation_sequence (n : ℕ) : ℕ :=
  match n % 3 with
  | 1 => 133
  | 2 => 55
  | 0 => 250
  | _ => 0  -- This case is unreachable, but needed for completeness

theorem operation_2012_equals_55 : operation_sequence 2012 = 55 := by
  sorry

end NUMINAMATH_CALUDE_operation_2012_equals_55_l1382_138251


namespace NUMINAMATH_CALUDE_proportional_function_ratio_l1382_138254

/-- Proves that for a proportional function y = kx passing through the points (1, 3) and (a, b) where b ≠ 0, a/b = 1/3 -/
theorem proportional_function_ratio (k a b : ℝ) (h1 : b ≠ 0) (h2 : 3 = k * 1) (h3 : b = k * a) : a / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_ratio_l1382_138254


namespace NUMINAMATH_CALUDE_unique_valid_f_l1382_138273

def is_valid_f (f : ℕ → ℕ) : Prop :=
  (∀ m, f m = 1 ↔ m = 1) ∧
  (∀ m n, f (m * n) = f m * f n / f (Nat.gcd m n)) ∧
  (∀ m, (f^[2000]) m = f m)

theorem unique_valid_f :
  ∃! f : ℕ → ℕ, is_valid_f f ∧ ∀ n, f n = n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_f_l1382_138273


namespace NUMINAMATH_CALUDE_classroom_gpa_l1382_138250

theorem classroom_gpa (n : ℝ) (h : n > 0) : 
  (1/3 * n * 45 + 2/3 * n * 60) / n = 55 := by
  sorry

end NUMINAMATH_CALUDE_classroom_gpa_l1382_138250


namespace NUMINAMATH_CALUDE_triangle_area_l1382_138263

theorem triangle_area (a c : ℝ) (B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = Real.pi / 3) :
  (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1382_138263


namespace NUMINAMATH_CALUDE_dot_product_equals_three_l1382_138281

def vector_a : ℝ × ℝ := (2, -1)
def vector_b (x : ℝ) : ℝ × ℝ := (3, x)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_equals_three (x : ℝ) :
  dot_product vector_a (vector_b x) = 3 → x = 3 := by
sorry

end NUMINAMATH_CALUDE_dot_product_equals_three_l1382_138281


namespace NUMINAMATH_CALUDE_inequality_theorem_equality_theorem_l1382_138247

-- Define the condition
def condition (x y : ℝ) : Prop := (x + 1) * (y + 2) = 8

-- Define the main theorem
theorem inequality_theorem (x y : ℝ) (h : condition x y) :
  (x * y - 10)^2 ≥ 64 ∧
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) :=
by sorry

-- Define the equality cases
def equality_cases (x y : ℝ) : Prop :=
  (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)

-- Theorem for the equality cases
theorem equality_theorem (x y : ℝ) (h : condition x y) :
  (x * y - 10)^2 = 64 ↔ equality_cases x y :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_equality_theorem_l1382_138247


namespace NUMINAMATH_CALUDE_evaluate_expression_l1382_138234

theorem evaluate_expression (a b : ℤ) (ha : a = 3) (hb : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1382_138234


namespace NUMINAMATH_CALUDE_square_formation_total_l1382_138265

/-- Given a square formation of people where one person is the 5th from each side,
    prove that the total number of people is 81. -/
theorem square_formation_total (n : ℕ) (h : n = 5) :
  (2 * n - 1) * (2 * n - 1) = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_formation_total_l1382_138265


namespace NUMINAMATH_CALUDE_ends_with_two_zeros_l1382_138262

theorem ends_with_two_zeros (x y : ℕ) :
  (x^2 + x*y + y^2) % 10 = 0 → (x^2 + x*y + y^2) % 100 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ends_with_two_zeros_l1382_138262


namespace NUMINAMATH_CALUDE_perpendicular_median_triangle_sides_l1382_138268

/-- A triangle with sides x, y, and z, where two medians are mutually perpendicular -/
structure PerpendicularMedianTriangle where
  x : ℕ
  y : ℕ
  z : ℕ
  perp_medians : x^2 + y^2 = 5 * z^2

/-- The theorem stating that the triangle with perpendicular medians and integer sides has sides 22, 19, and 13 -/
theorem perpendicular_median_triangle_sides :
  ∀ t : PerpendicularMedianTriangle, t.x = 22 ∧ t.y = 19 ∧ t.z = 13 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_median_triangle_sides_l1382_138268


namespace NUMINAMATH_CALUDE_bucket_fill_lcm_l1382_138283

/-- Time to fill bucket A completely -/
def time_A : ℕ := 135

/-- Time to fill bucket B completely -/
def time_B : ℕ := 240

/-- Time to fill bucket C completely -/
def time_C : ℕ := 200

/-- Function to calculate the least common multiple of three natural numbers -/
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem bucket_fill_lcm :
  (2 * time_A = 3 * 90) ∧
  (time_B = 2 * 120) ∧
  (3 * time_C = 4 * 150) →
  lcm_three time_A time_B time_C = 1200 := by
  sorry

end NUMINAMATH_CALUDE_bucket_fill_lcm_l1382_138283


namespace NUMINAMATH_CALUDE_jay_scored_six_more_l1382_138206

/-- Represents the scores of players in a basketball game. -/
structure BasketballScores where
  tobee : ℕ
  jay : ℕ
  sean : ℕ

/-- Conditions of the basketball game scores. -/
def validScores (scores : BasketballScores) : Prop :=
  scores.tobee = 4 ∧
  scores.jay > scores.tobee ∧
  scores.sean = scores.tobee + scores.jay - 2 ∧
  scores.tobee + scores.jay + scores.sean = 26

/-- Theorem stating that Jay scored 6 more points than Tobee. -/
theorem jay_scored_six_more (scores : BasketballScores) 
  (h : validScores scores) : scores.jay = scores.tobee + 6 := by
  sorry

#check jay_scored_six_more

end NUMINAMATH_CALUDE_jay_scored_six_more_l1382_138206


namespace NUMINAMATH_CALUDE_puppies_weight_difference_l1382_138216

/-- The weight difference between two dogs after a year, given their initial weights and weight gain percentage -/
def weight_difference (labrador_initial : ℝ) (dachshund_initial : ℝ) (weight_gain_percentage : ℝ) : ℝ :=
  (labrador_initial * (1 + weight_gain_percentage)) - (dachshund_initial * (1 + weight_gain_percentage))

/-- Theorem stating that the weight difference between the labrador and dachshund puppies after a year is 35 pounds -/
theorem puppies_weight_difference :
  weight_difference 40 12 0.25 = 35 := by
  sorry

end NUMINAMATH_CALUDE_puppies_weight_difference_l1382_138216


namespace NUMINAMATH_CALUDE_store_profit_calculation_l1382_138201

theorem store_profit_calculation (C : ℝ) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_discount := 0.18
  let initial_price := C * (1 + initial_markup)
  let new_year_price := initial_price * (1 + new_year_markup)
  let final_price := new_year_price * (1 - february_discount)
  let profit := final_price - C
  profit = 0.23 * C := by sorry

end NUMINAMATH_CALUDE_store_profit_calculation_l1382_138201


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l1382_138231

theorem sum_of_numbers_in_ratio (x : ℝ) :
  x > 0 →
  x^2 + (2*x)^2 + (5*x)^2 = 4320 →
  x + 2*x + 5*x = 96 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l1382_138231
