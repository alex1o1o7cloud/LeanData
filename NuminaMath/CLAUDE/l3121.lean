import Mathlib

namespace NUMINAMATH_CALUDE_election_winner_votes_l3121_312180

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h1 : winner_percentage = 56 / 100) 
  (h2 : vote_difference = 288) 
  (h3 : ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference) :
  ↑total_votes * winner_percentage = 1344 :=
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3121_312180


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3121_312116

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem stating that if M(3, a-2) and N(b, a) are symmetric with respect to the origin, then a + b = -2 -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_wrt_origin 3 (a - 2) b a → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3121_312116


namespace NUMINAMATH_CALUDE_photo_comparison_l3121_312173

theorem photo_comparison (claire lisa robert : ℕ) 
  (h1 : lisa = 3 * claire)
  (h2 : robert = lisa)
  : robert = 2 * claire + claire := by
  sorry

end NUMINAMATH_CALUDE_photo_comparison_l3121_312173


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3121_312186

theorem sum_of_fifth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 14) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 44 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3121_312186


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l3121_312113

theorem line_slope_intercept_product (m b : ℚ) : 
  m = 3/4 → b = 5/2 → m * b > 1 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l3121_312113


namespace NUMINAMATH_CALUDE_real_part_of_z_squared_neg_four_l3121_312198

theorem real_part_of_z_squared_neg_four (z : ℂ) : z^2 = -4 → Complex.re z = 0 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_squared_neg_four_l3121_312198


namespace NUMINAMATH_CALUDE_right_triangle_angle_bisectors_l3121_312128

/-- Given a right triangle with legs 3 and 4, prove the lengths of its angle bisectors. -/
theorem right_triangle_angle_bisectors :
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := (a^2 + b^2).sqrt
  let d : ℝ := (a * b * ((a + b + c) / (a + b - c))).sqrt
  let l_b : ℝ := (b * a * (1 - ((b - a)^2 / b^2))).sqrt
  let l_a : ℝ := (a * b * (1 - ((a - b)^2 / a^2))).sqrt
  (d = 12 * Real.sqrt 2 / 7) ∧
  (l_b = 3 * Real.sqrt 5 / 2) ∧
  (l_a = 4 * Real.sqrt 10 / 3) := by
sorry


end NUMINAMATH_CALUDE_right_triangle_angle_bisectors_l3121_312128


namespace NUMINAMATH_CALUDE_symmetry_axes_intersection_l3121_312188

-- Define a polygon as a set of points in 2D space
def Polygon := Set (ℝ × ℝ)

-- Define an axis of symmetry for a polygon
def IsAxisOfSymmetry (p : Polygon) (axis : Set (ℝ × ℝ)) : Prop := sorry

-- Define the center of mass for a set of points
def CenterOfMass (points : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define a property that a point lies on a line
def PointOnLine (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem symmetry_axes_intersection (p : Polygon) 
  (h_multiple_axes : ∃ (axis1 axis2 : Set (ℝ × ℝ)), axis1 ≠ axis2 ∧ IsAxisOfSymmetry p axis1 ∧ IsAxisOfSymmetry p axis2) :
  ∀ (axis : Set (ℝ × ℝ)), IsAxisOfSymmetry p axis → 
    PointOnLine (CenterOfMass p) axis :=
sorry

end NUMINAMATH_CALUDE_symmetry_axes_intersection_l3121_312188


namespace NUMINAMATH_CALUDE_alex_is_26_l3121_312191

-- Define the ages as natural numbers
def inez_age : ℕ := 18
def zack_age : ℕ := inez_age + 5
def jose_age : ℕ := zack_age - 3
def alex_age : ℕ := jose_age + 6

-- Theorem to prove
theorem alex_is_26 : alex_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_alex_is_26_l3121_312191


namespace NUMINAMATH_CALUDE_problem_solution_l3121_312139

theorem problem_solution (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : 3 * m + 2 * n = 225) (h4 : Nat.gcd m n = 15) : m + n = 105 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3121_312139


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3121_312109

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Represents a line in 2D space of the form y = mx + b -/
structure Line where
  m : ℚ
  b : ℚ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : IntersectionPoint) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- The theorem stating the intersection point of two specific lines -/
theorem intersection_of_lines :
  let line1 : Line := { m := 3, b := -1 }
  let line2 : Line := { m := -6, b := -4 }
  let point : IntersectionPoint := { x := -1/3, y := -2 }
  (pointOnLine point line1) ∧ (pointOnLine point line2) ∧
  (∀ p : IntersectionPoint, (pointOnLine p line1) ∧ (pointOnLine p line2) → p = point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3121_312109


namespace NUMINAMATH_CALUDE_solution_system_l3121_312189

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 110) : 
  x^2 + y^2 = 8044 / 169 := by
  sorry

end NUMINAMATH_CALUDE_solution_system_l3121_312189


namespace NUMINAMATH_CALUDE_complex_magnitude_l3121_312125

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I)^2 = 1 + Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3121_312125


namespace NUMINAMATH_CALUDE_shoe_difference_l3121_312159

/-- Scott's number of shoe pairs -/
def scott_shoes : ℕ := 7

/-- Anthony's number of shoe pairs -/
def anthony_shoes : ℕ := 3 * scott_shoes

/-- Jim's number of shoe pairs -/
def jim_shoes : ℕ := anthony_shoes - 2

/-- The difference between Anthony's and Jim's shoe pairs -/
theorem shoe_difference : anthony_shoes - jim_shoes = 2 := by
  sorry

end NUMINAMATH_CALUDE_shoe_difference_l3121_312159


namespace NUMINAMATH_CALUDE_inequality_solution_l3121_312107

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Iio (-2) ∪ Set.Ioo (-1) 1 ∪ Set.Ioo 2 3 ∪ Set.Ioo 4 6 ∪ Set.Ioi 7) ↔ 
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ 
   (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 24)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3121_312107


namespace NUMINAMATH_CALUDE_hat_cost_theorem_l3121_312104

def josh_shopping (initial_money : ℝ) (pencil_cost : ℝ) (cookie_cost : ℝ) (cookie_count : ℕ) (money_left : ℝ) : ℝ :=
  initial_money - (pencil_cost + cookie_cost * cookie_count) - money_left

theorem hat_cost_theorem (initial_money : ℝ) (pencil_cost : ℝ) (cookie_cost : ℝ) (cookie_count : ℕ) (money_left : ℝ) 
  (h1 : initial_money = 20)
  (h2 : pencil_cost = 2)
  (h3 : cookie_cost = 1.25)
  (h4 : cookie_count = 4)
  (h5 : money_left = 3) :
  josh_shopping initial_money pencil_cost cookie_cost cookie_count money_left = 10 := by
  sorry

#eval josh_shopping 20 2 1.25 4 3

end NUMINAMATH_CALUDE_hat_cost_theorem_l3121_312104


namespace NUMINAMATH_CALUDE_prob_two_good_in_four_draws_l3121_312142

/-- Represents the number of light bulbs in the box -/
def total_bulbs : ℕ := 10

/-- Represents the number of good quality bulbs -/
def good_bulbs : ℕ := 8

/-- Represents the number of defective bulbs -/
def defective_bulbs : ℕ := 2

/-- Represents the number of draws -/
def num_draws : ℕ := 4

/-- Represents the number of good quality bulbs to be drawn -/
def target_good_bulbs : ℕ := 2

/-- Calculates the probability of drawing exactly 2 good quality bulbs in 4 draws -/
theorem prob_two_good_in_four_draws :
  (defective_bulbs * (defective_bulbs - 1) * good_bulbs * (good_bulbs - 1)) / 
  (total_bulbs * (total_bulbs - 1) * (total_bulbs - 2) * (total_bulbs - 3)) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_good_in_four_draws_l3121_312142


namespace NUMINAMATH_CALUDE_expansion_properties_l3121_312108

theorem expansion_properties (x : ℝ) (n : ℕ) :
  (∃ k : ℝ, 2 * (n.choose 2) = (n.choose 1) + (n.choose 3) ∧ k ≠ 0) →
  (n = 7 ∧ ∀ r : ℕ, r ≤ n → (7 - 2*r ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l3121_312108


namespace NUMINAMATH_CALUDE_ellipse_equation_l3121_312161

/-- Given an ellipse C₁ and a circle C₂, prove that C₁ has the equation x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (P : ℝ × ℝ) :
  a > b ∧ b > 0 ∧  -- a > b > 0
  P = (0, -1) ∧  -- P(0,-1) is a vertex of C₁
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →  -- Equation of C₁
  2 * a = 4 ∧  -- Major axis of C₁ is diameter of C₂
  ∀ x y : ℝ, x^2 + y^2 = 4 →  -- Equation of C₂
  ∀ x y : ℝ, x^2 / 4 + y^2 = 1  -- Equation of C₁ we want to prove
  := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3121_312161


namespace NUMINAMATH_CALUDE_eight_digit_number_theorem_l3121_312117

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def last_digit (n : ℕ) : ℕ := n % 10

def move_last_to_first (n : ℕ) : ℕ :=
  (last_digit n) * 10^7 + n / 10

theorem eight_digit_number_theorem (B : ℕ) 
  (h1 : B > 7777777)
  (h2 : is_coprime B 36)
  (h3 : ∃ A : ℕ, A = move_last_to_first B) :
  ∃ A_min A_max : ℕ, 
    (A_min = move_last_to_first B ∧ A_min ≥ 17777779) ∧
    (A_max = move_last_to_first B ∧ A_max ≤ 99999998) ∧
    (∀ A : ℕ, A = move_last_to_first B → A_min ≤ A ∧ A ≤ A_max) :=
by sorry

end NUMINAMATH_CALUDE_eight_digit_number_theorem_l3121_312117


namespace NUMINAMATH_CALUDE_exact_one_solver_probability_l3121_312101

/-- The probability that exactly one person solves a problem, given the probabilities
    for two independent solvers. -/
theorem exact_one_solver_probability (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  (p₁ * (1 - p₂) + p₂ * (1 - p₁)) = 
  (p₁ + p₂ - 2 * p₁ * p₂) := by
  sorry

#check exact_one_solver_probability

end NUMINAMATH_CALUDE_exact_one_solver_probability_l3121_312101


namespace NUMINAMATH_CALUDE_expression_value_l3121_312114

/-- Proves that the expression (3a+b)^2 - (3a-b)(3a+b) - 5b(a-b) equals 26 when a=1 and b=-2 -/
theorem expression_value (a b : ℤ) (h1 : a = 1) (h2 : b = -2) :
  (3*a + b)^2 - (3*a - b)*(3*a + b) - 5*b*(a - b) = 26 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3121_312114


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3121_312121

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 4 * x + 2 = 0}

theorem unique_solution_condition (a : ℝ) : 
  (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3121_312121


namespace NUMINAMATH_CALUDE_childless_count_bertha_l3121_312124

structure Family :=
  (daughters : ℕ)
  (total_descendants : ℕ)
  (grandchildren_per_daughter : ℕ)

def childless_count (f : Family) : ℕ :=
  f.total_descendants - f.daughters

theorem childless_count_bertha (f : Family) 
  (h1 : f.daughters = 8)
  (h2 : f.total_descendants = 40)
  (h3 : f.grandchildren_per_daughter = 4)
  (h4 : f.total_descendants = f.daughters + f.daughters * f.grandchildren_per_daughter) :
  childless_count f = 32 := by
  sorry


end NUMINAMATH_CALUDE_childless_count_bertha_l3121_312124


namespace NUMINAMATH_CALUDE_solution_composition_l3121_312195

theorem solution_composition (x_a : Real) (y_a y_b : Real) (mix_a : Real) (mix_x : Real) :
  x_a = 0.4 →
  y_a = 0.5 →
  y_b = 0.5 →
  mix_a = 0.47 →
  mix_x = 0.3 →
  mix_x * x_a + (1 - mix_x) * y_a = mix_a →
  1 - x_a = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_solution_composition_l3121_312195


namespace NUMINAMATH_CALUDE_unique_solution_l3121_312133

theorem unique_solution : 
  ∃! (x y z t : ℕ), 31 * (x * y * z * t + x * y + x * t + z * t + 1) = 40 * (y * z * t + y + t) ∧
  x = 1 ∧ y = 3 ∧ z = 2 ∧ t = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3121_312133


namespace NUMINAMATH_CALUDE_towel_length_decrease_l3121_312151

/-- Theorem: Percentage decrease in towel length
Given a towel that lost some percentage of its length and 20% of its breadth,
resulting in a 36% decrease in area, prove that the percentage decrease in length is 20%.
-/
theorem towel_length_decrease (L B : ℝ) (L' B' : ℝ) (h_positive : L > 0 ∧ B > 0) :
  B' = 0.8 * B →                         -- Breadth decreased by 20%
  L' * B' = 0.64 * (L * B) →             -- Area decreased by 36%
  L' = 0.8 * L                           -- Length decreased by 20%
  := by sorry

end NUMINAMATH_CALUDE_towel_length_decrease_l3121_312151


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_sqrt_1968_l3121_312106

theorem sqrt_sum_eq_sqrt_1968 : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ s ↔ Real.sqrt x + Real.sqrt y = Real.sqrt 1968) ∧ 
    s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_sqrt_1968_l3121_312106


namespace NUMINAMATH_CALUDE_camp_wonka_marshmallows_l3121_312122

theorem camp_wonka_marshmallows (total_campers : ℕ) 
  (boys_fraction : ℚ) (girls_fraction : ℚ) 
  (boys_toasting_percentage : ℚ) (girls_toasting_percentage : ℚ) :
  total_campers = 96 →
  boys_fraction = 2/3 →
  girls_fraction = 1/3 →
  boys_toasting_percentage = 1/2 →
  girls_toasting_percentage = 3/4 →
  (boys_fraction * total_campers * boys_toasting_percentage + 
   girls_fraction * total_campers * girls_toasting_percentage : ℚ) = 56 := by
  sorry

end NUMINAMATH_CALUDE_camp_wonka_marshmallows_l3121_312122


namespace NUMINAMATH_CALUDE_angle_bisector_length_right_triangle_l3121_312158

theorem angle_bisector_length_right_triangle (a b c : ℝ) (h1 : a = 15) (h2 : b = 20) (h3 : c = 25) 
  (h4 : a^2 + b^2 = c^2) : ∃ (AA₁ : ℝ), AA₁ = (20 * Real.sqrt 10) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_length_right_triangle_l3121_312158


namespace NUMINAMATH_CALUDE_coin_toss_outcomes_l3121_312155

/-- The number of possible outcomes when throwing three coins -/
def coin_outcomes : ℕ := 8

/-- The number of coins being thrown -/
def num_coins : ℕ := 3

/-- The number of possible states for each coin (heads or tails) -/
def states_per_coin : ℕ := 2

/-- Theorem stating that the number of possible outcomes when throwing three coins,
    each with two possible states, is equal to 8 -/
theorem coin_toss_outcomes :
  coin_outcomes = states_per_coin ^ num_coins :=
by sorry

end NUMINAMATH_CALUDE_coin_toss_outcomes_l3121_312155


namespace NUMINAMATH_CALUDE_rope_length_problem_l3121_312112

theorem rope_length_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece / longer_piece = 3 / 4 →
  longer_piece = 20 →
  total_length = shorter_piece + longer_piece →
  total_length = 35 := by
sorry

end NUMINAMATH_CALUDE_rope_length_problem_l3121_312112


namespace NUMINAMATH_CALUDE_kolya_mistake_l3121_312166

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens < 10
  h_ones : ones < 10

/-- Represents a four-digit number of the form effe -/
structure FourDigitNumberEFFE where
  e : Nat
  f : Nat
  h_e : e < 10
  h_f : f < 10

/-- Function to check if a two-digit number is divisible by 11 -/
def isDivisibleBy11 (n : TwoDigitNumber) : Prop :=
  (n.tens - n.ones) % 11 = 0

/-- The main theorem -/
theorem kolya_mistake
  (ab cd : TwoDigitNumber)
  (effe : FourDigitNumberEFFE)
  (h_mult : ab.tens * 10 + ab.ones * cd.tens * 10 + cd.ones = effe.e * 1000 + effe.f * 100 + effe.f * 10 + effe.e)
  (h_distinct : ab.tens ≠ ab.ones ∧ cd.tens ≠ cd.ones ∧ ab.tens ≠ cd.tens ∧ ab.tens ≠ cd.ones ∧ ab.ones ≠ cd.tens ∧ ab.ones ≠ cd.ones)
  : isDivisibleBy11 ab ∨ isDivisibleBy11 cd :=
sorry

end NUMINAMATH_CALUDE_kolya_mistake_l3121_312166


namespace NUMINAMATH_CALUDE_sequence_conjecture_l3121_312144

theorem sequence_conjecture (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, a (n + 1) - a n > 0) ∧
  (∀ n : ℕ, (a (n + 1) - a n)^2 - 2 * (a (n + 1) + a n) + 1 = 0) →
  ∀ n : ℕ, a n = n^2 := by
sorry

end NUMINAMATH_CALUDE_sequence_conjecture_l3121_312144


namespace NUMINAMATH_CALUDE_function_properties_l3121_312168

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1))) 
  (h2 : is_odd (λ x => f (x - 1))) : 
  (∀ x, f (x + 4) = f x) ∧ 
  (is_odd (λ x => f (x + 3))) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l3121_312168


namespace NUMINAMATH_CALUDE_cubic_double_root_abs_ab_l3121_312183

/-- Given a cubic polynomial with a double root and an integer third root, prove |ab| = 3360 -/
theorem cubic_double_root_abs_ab (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℝ, (x - r)^2 * (x - s) = x^3 + a*x^2 + b*x + 16*a)) →
  |a * b| = 3360 :=
by sorry

end NUMINAMATH_CALUDE_cubic_double_root_abs_ab_l3121_312183


namespace NUMINAMATH_CALUDE_distance_at_4_seconds_l3121_312115

/-- The distance traveled by an object given time t -/
def distance (t : ℝ) : ℝ := 5 * t^2 + 2 * t

theorem distance_at_4_seconds :
  distance 4 = 88 := by
  sorry

end NUMINAMATH_CALUDE_distance_at_4_seconds_l3121_312115


namespace NUMINAMATH_CALUDE_prob_sum_14_four_dice_l3121_312131

/-- The number of faces on a standard die -/
def faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The target sum we're aiming for -/
def target_sum : ℕ := 14

/-- The total number of possible outcomes when rolling four dice -/
def total_outcomes : ℕ := faces ^ num_dice

/-- The number of favorable outcomes (sum of 14) -/
def favorable_outcomes : ℕ := 54

/-- The probability of rolling a sum of 14 with four standard six-faced dice -/
theorem prob_sum_14_four_dice : 
  (favorable_outcomes : ℚ) / total_outcomes = 54 / 1296 := by sorry

end NUMINAMATH_CALUDE_prob_sum_14_four_dice_l3121_312131


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3121_312153

/-- Calculates the combined tax rate for Mork and Mindy -/
theorem combined_tax_rate (mork_rate mindy_rate : ℚ) (income_ratio : ℚ) :
  mork_rate = 2/5 →
  mindy_rate = 1/4 →
  income_ratio = 4 →
  (mork_rate + income_ratio * mindy_rate) / (1 + income_ratio) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l3121_312153


namespace NUMINAMATH_CALUDE_subscription_difference_is_4000_l3121_312145

/-- Represents the subscription amounts and profit distribution in a business venture. -/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  c_profit : ℕ
  b_extra : ℕ

/-- Calculates the difference between A's and B's subscriptions. -/
def subscription_difference (bv : BusinessVenture) : ℕ :=
  let c_subscription := bv.c_profit * bv.total_subscription / bv.total_profit
  let b_subscription := c_subscription + bv.b_extra
  let a_subscription := bv.total_subscription - b_subscription - c_subscription
  a_subscription - b_subscription

/-- Theorem stating that the difference between A's and B's subscriptions is 4000. -/
theorem subscription_difference_is_4000 :
  subscription_difference ⟨50000, 35000, 8400, 5000⟩ = 4000 := by
  sorry


end NUMINAMATH_CALUDE_subscription_difference_is_4000_l3121_312145


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3121_312160

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 6*x + 8) + (x^2 + 5*x - 7) = (x^2 + 5*x + 2) * (x^2 + 5*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3121_312160


namespace NUMINAMATH_CALUDE_problem_solution_l3121_312164

theorem problem_solution : 
  ((2023 - Real.sqrt 5) ^ 0 - 2 + abs (Real.sqrt 3 - 1) = Real.sqrt 3 - 2) ∧
  ((Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) + (Real.sqrt 15 * Real.sqrt 3) / Real.sqrt 5 = 2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3121_312164


namespace NUMINAMATH_CALUDE_student_council_max_profit_l3121_312172

/-- Calculate the maximum amount of money the student council can make from selling erasers --/
theorem student_council_max_profit (
  boxes : ℕ)
  (erasers_per_box : ℕ)
  (price_per_eraser : ℚ)
  (bulk_discount_rate : ℚ)
  (bulk_purchase_threshold : ℕ)
  (sales_tax_rate : ℚ)
  (h1 : boxes = 48)
  (h2 : erasers_per_box = 24)
  (h3 : price_per_eraser = 3/4)
  (h4 : bulk_discount_rate = 1/10)
  (h5 : bulk_purchase_threshold = 10)
  (h6 : sales_tax_rate = 3/50)
  : ∃ (max_profit : ℚ), max_profit = 82426/100 :=
by
  sorry

end NUMINAMATH_CALUDE_student_council_max_profit_l3121_312172


namespace NUMINAMATH_CALUDE_smallest_N_value_l3121_312126

/-- Represents a point in the rectangular array. -/
structure Point where
  row : Fin 5
  col : ℕ

/-- The first numbering system (left to right, top to bottom). -/
def firstNumber (N : ℕ) (p : Point) : ℕ :=
  N * p.row.val + p.col

/-- The second numbering system (top to bottom, left to right). -/
def secondNumber (p : Point) : ℕ :=
  5 * (p.col - 1) + p.row.val + 1

/-- The main theorem stating the smallest possible value of N. -/
theorem smallest_N_value :
  ∃ (N : ℕ) (p₁ p₂ p₃ p₄ p₅ : Point),
    p₁.row = 0 ∧ p₂.row = 1 ∧ p₃.row = 2 ∧ p₄.row = 3 ∧ p₅.row = 4 ∧
    (∀ p : Point, p.col < N) ∧
    firstNumber N p₁ = secondNumber p₂ ∧
    firstNumber N p₂ = secondNumber p₁ ∧
    firstNumber N p₃ = secondNumber p₄ ∧
    firstNumber N p₄ = secondNumber p₅ ∧
    firstNumber N p₅ = secondNumber p₃ ∧
    (∀ N' : ℕ, N' < N →
      ¬∃ (q₁ q₂ q₃ q₄ q₅ : Point),
        q₁.row = 0 ∧ q₂.row = 1 ∧ q₃.row = 2 ∧ q₄.row = 3 ∧ q₅.row = 4 ∧
        (∀ q : Point, q.col < N') ∧
        firstNumber N' q₁ = secondNumber q₂ ∧
        firstNumber N' q₂ = secondNumber q₁ ∧
        firstNumber N' q₃ = secondNumber q₄ ∧
        firstNumber N' q₄ = secondNumber q₅ ∧
        firstNumber N' q₅ = secondNumber q₃) ∧
    N = 149 := by
  sorry

end NUMINAMATH_CALUDE_smallest_N_value_l3121_312126


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3121_312127

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    if S_n = 2 and S_{3n} = 18, then S_{4n} = 26 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, S k = (k * (2 * (a 1) + (k - 1) * (a 2 - a 1))) / 2) →
  S n = 2 →
  S (3 * n) = 18 →
  S (4 * n) = 26 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3121_312127


namespace NUMINAMATH_CALUDE_unique_brigade_solution_l3121_312174

/-- Represents a brigade with newspapers and members -/
structure Brigade where
  newspapers : ℕ
  members : ℕ

/-- Properties of a valid brigade -/
def is_valid_brigade (b : Brigade) : Prop :=
  ∀ (m : ℕ) (n : ℕ), m ≤ b.members → n ≤ b.newspapers →
    (∃! (c : ℕ), c = 2) ∧  -- Each member reads exactly 2 newspapers
    (∃! (d : ℕ), d = 5) ∧  -- Each newspaper is read by exactly 5 members
    (∃! (e : ℕ), e = 1)    -- Each combination of 2 newspapers is read by exactly 1 member

/-- Theorem stating the unique solution for a valid brigade -/
theorem unique_brigade_solution (b : Brigade) (h : is_valid_brigade b) :
  b.newspapers = 6 ∧ b.members = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_brigade_solution_l3121_312174


namespace NUMINAMATH_CALUDE_fourth_sample_is_20_l3121_312175

def random_numbers : List ℕ := [71, 11, 5, 65, 9, 95, 86, 68, 76, 83, 20, 37, 90, 57, 16, 3, 11, 63, 14, 90]

def is_valid_sample (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 50

def get_fourth_sample (numbers : List ℕ) : ℕ :=
  (numbers.filter is_valid_sample).nthLe 3 sorry

theorem fourth_sample_is_20 :
  get_fourth_sample random_numbers = 20 := by sorry

end NUMINAMATH_CALUDE_fourth_sample_is_20_l3121_312175


namespace NUMINAMATH_CALUDE_max_value_problem_l3121_312137

theorem max_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2 * a + b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧
    2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ 2 * Real.sqrt (x * y) - 4 * x^2 - y^2) ∧
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ (Real.sqrt 2 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l3121_312137


namespace NUMINAMATH_CALUDE_min_draws_for_18_l3121_312140

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee at least n of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The actual box contents -/
def boxContents : BallCounts :=
  { red := 30, green := 25, yellow := 22, blue := 15, white := 12, black := 6 }

/-- The main theorem -/
theorem min_draws_for_18 :
  minDraws boxContents 18 = 85 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_18_l3121_312140


namespace NUMINAMATH_CALUDE_sum_of_n_values_l3121_312120

theorem sum_of_n_values (m n : ℕ+) : 
  (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 5 →
  ∃ (n₁ n₂ n₃ : ℕ+), 
    (∀ k : ℕ+, ((1 : ℚ) / m + (1 : ℚ) / k = (1 : ℚ) / 5) → (k = n₁ ∨ k = n₂ ∨ k = n₃)) ∧
    n₁.val + n₂.val + n₃.val = 46 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_n_values_l3121_312120


namespace NUMINAMATH_CALUDE_orange_apple_cost_difference_l3121_312156

/-- The cost difference between an orange and an apple -/
def cost_difference (apple_cost orange_cost : ℚ) : ℚ := orange_cost - apple_cost

theorem orange_apple_cost_difference 
  (apple_cost orange_cost : ℚ) 
  (total_cost : ℚ)
  (h1 : apple_cost > 0)
  (h2 : orange_cost > apple_cost)
  (h3 : 3 * apple_cost + 7 * orange_cost = total_cost)
  (h4 : total_cost = 456/100) : 
  ∃ (diff : ℚ), cost_difference apple_cost orange_cost = diff ∧ diff > 0 := by
  sorry

#eval cost_difference (26/100) (36/100)

end NUMINAMATH_CALUDE_orange_apple_cost_difference_l3121_312156


namespace NUMINAMATH_CALUDE_goldfish_cost_graph_piecewise_linear_l3121_312149

/-- The cost function for goldfish purchases -/
def cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 20 * n else 20 * n - 5

/-- The graph of the cost function is piecewise linear -/
theorem goldfish_cost_graph_piecewise_linear :
  ∃ (f g : ℚ → ℚ),
    (∀ x, f x = 20 * x) ∧
    (∀ x, g x = 20 * x - 5) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 15 →
      (n ≤ 10 ∧ cost n = f n) ∨
      (10 < n ∧ cost n = g n)) :=
by sorry

end NUMINAMATH_CALUDE_goldfish_cost_graph_piecewise_linear_l3121_312149


namespace NUMINAMATH_CALUDE_operations_to_equality_l3121_312111

def num_operations (a b : ℕ) (subtract_a add_b : ℕ) : ℕ :=
  (a - b) / (subtract_a + add_b)

theorem operations_to_equality : num_operations 365 24 19 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_operations_to_equality_l3121_312111


namespace NUMINAMATH_CALUDE_intersection_equals_B_l3121_312192

def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {0, 1}

theorem intersection_equals_B : A ∩ B = B := by sorry

end NUMINAMATH_CALUDE_intersection_equals_B_l3121_312192


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3121_312130

/-- The perimeter of a rhombus with diagonals 18 and 12 is 12√13 -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 12) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 12 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3121_312130


namespace NUMINAMATH_CALUDE_snooker_tournament_revenue_l3121_312147

theorem snooker_tournament_revenue
  (total_tickets : ℕ)
  (vip_price general_price : ℚ)
  (fewer_vip : ℕ) :
  total_tickets = 320 →
  vip_price = 40 →
  general_price = 15 →
  fewer_vip = 212 →
  ∃ (vip_tickets general_tickets : ℕ),
    vip_tickets + general_tickets = total_tickets ∧
    vip_tickets = general_tickets - fewer_vip ∧
    vip_price * vip_tickets + general_price * general_tickets = 6150 :=
by sorry

end NUMINAMATH_CALUDE_snooker_tournament_revenue_l3121_312147


namespace NUMINAMATH_CALUDE_radius_is_seven_l3121_312123

/-- Represents a circle with a point P outside it and a secant PQR -/
structure CircleWithSecant where
  /-- Distance from P to the center of the circle -/
  s : ℝ
  /-- Length of external segment PQ -/
  pq : ℝ
  /-- Length of chord QR -/
  qr : ℝ

/-- The radius of the circle given the secant configuration -/
def radius (c : CircleWithSecant) : ℝ :=
  sorry

/-- Theorem stating that the radius is 7 given the specific measurements -/
theorem radius_is_seven (c : CircleWithSecant) 
  (h1 : c.s = 17) 
  (h2 : c.pq = 12) 
  (h3 : c.qr = 8) : 
  radius c = 7 := by
  sorry

end NUMINAMATH_CALUDE_radius_is_seven_l3121_312123


namespace NUMINAMATH_CALUDE_school_visit_arrangements_l3121_312170

/-- Represents the number of days in a week -/
def week_days : Nat := 7

/-- Represents the number of consecutive days School A visits -/
def school_a_days : Nat := 2

/-- Represents the number of days School B visits -/
def school_b_days : Nat := 1

/-- Represents the number of days School C visits -/
def school_c_days : Nat := 1

/-- Calculates the number of arrangements for the school visits -/
def calculate_arrangements : Nat :=
  (week_days - school_a_days - school_b_days - school_c_days + 1) *
  (week_days - school_a_days - school_b_days - school_c_days)

/-- Theorem stating that the number of arrangements is 40 -/
theorem school_visit_arrangements :
  calculate_arrangements = 40 := by
  sorry

end NUMINAMATH_CALUDE_school_visit_arrangements_l3121_312170


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l3121_312169

theorem power_mod_seventeen : 7^2048 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l3121_312169


namespace NUMINAMATH_CALUDE_sixth_operation_result_l3121_312176

def operation (a b : ℕ) : ℕ := (a + b) * a - a

theorem sixth_operation_result : operation 7 8 = 98 := by
  sorry

end NUMINAMATH_CALUDE_sixth_operation_result_l3121_312176


namespace NUMINAMATH_CALUDE_complement_of_A_in_B_l3121_312154

def A : Set ℕ := {2, 3}
def B : Set ℕ := {0, 1, 2, 3, 4}

theorem complement_of_A_in_B :
  (B \ A) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_B_l3121_312154


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l3121_312102

theorem part_to_whole_ratio (N : ℚ) (P : ℚ) : 
  (1 / 4 : ℚ) * P = 25 →
  (40 / 100 : ℚ) * N = 300 →
  P / ((2 / 5 : ℚ) * N) = (1 / 3 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l3121_312102


namespace NUMINAMATH_CALUDE_no_distinct_naturals_satisfying_equation_l3121_312163

theorem no_distinct_naturals_satisfying_equation :
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + 1 : ℚ) / a = ((b + 1 : ℚ) / b + (c + 1 : ℚ) / c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_distinct_naturals_satisfying_equation_l3121_312163


namespace NUMINAMATH_CALUDE_live_flowers_l3121_312132

theorem live_flowers (total : ℕ) (withered : ℕ) (h1 : total = 13) (h2 : withered = 7) :
  total - withered = 6 := by
  sorry

end NUMINAMATH_CALUDE_live_flowers_l3121_312132


namespace NUMINAMATH_CALUDE_expression_nonnegative_l3121_312100

theorem expression_nonnegative (x : ℝ) : 
  (2*x - 6*x^2 + 9*x^3) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 3 :=
sorry

end NUMINAMATH_CALUDE_expression_nonnegative_l3121_312100


namespace NUMINAMATH_CALUDE_smallest_x_for_1260x_perfect_square_l3121_312179

theorem smallest_x_for_1260x_perfect_square : 
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), ∃ (N : ℤ), 1260 * y = N^2 → x ≤ y) ∧
    (∃ (N : ℤ), 1260 * x = N^2) ∧
    x = 35 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_1260x_perfect_square_l3121_312179


namespace NUMINAMATH_CALUDE_cherry_pies_count_l3121_312118

/-- Given a total number of pies and a ratio of three types of pies,
    calculate the number of pies of the third type. -/
def calculate_cherry_pies (total_pies : ℕ) (ratio_apple : ℕ) (ratio_blueberry : ℕ) (ratio_cherry : ℕ) : ℕ :=
  let total_ratio := ratio_apple + ratio_blueberry + ratio_cherry
  let pies_per_part := total_pies / total_ratio
  ratio_cherry * pies_per_part

/-- Theorem stating that given 36 total pies and a ratio of 2:3:4 for apple, blueberry, and cherry pies,
    the number of cherry pies is 16. -/
theorem cherry_pies_count :
  calculate_cherry_pies 36 2 3 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pies_count_l3121_312118


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3121_312136

open Set

theorem solution_set_of_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, HasDerivAt f (f' x) x) →  -- f' is the derivative of f
  (∀ x > 0, x * f' x + 3 * f x > 0) →  -- given condition
  {x : ℝ | x^3 * f x + (2*x - 1)^3 * f (1 - 2*x) < 0} = Iic (1/3) ∪ Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3121_312136


namespace NUMINAMATH_CALUDE_cottage_rent_division_l3121_312181

/-- The total rent for the cottage -/
def total_rent : ℤ := 300

/-- The amount paid by the first friend -/
def first_friend_payment (f2 f3 f4 : ℤ) : ℤ := (f2 + f3 + f4) / 2

/-- The amount paid by the second friend -/
def second_friend_payment (f1 f3 f4 : ℤ) : ℤ := (f1 + f3 + f4) / 3

/-- The amount paid by the third friend -/
def third_friend_payment (f1 f2 f4 : ℤ) : ℤ := (f1 + f2 + f4) / 4

/-- The amount paid by the fourth friend -/
def fourth_friend_payment (f1 f2 f3 : ℤ) : ℤ := total_rent - (f1 + f2 + f3)

theorem cottage_rent_division :
  ∃ (f1 f2 f3 f4 : ℤ),
    f1 = first_friend_payment f2 f3 f4 ∧
    f2 = second_friend_payment f1 f3 f4 ∧
    f3 = third_friend_payment f1 f2 f4 ∧
    f4 = fourth_friend_payment f1 f2 f3 ∧
    f1 + f2 + f3 + f4 = total_rent ∧
    f4 = 65 :=
by sorry

end NUMINAMATH_CALUDE_cottage_rent_division_l3121_312181


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l3121_312190

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 13 ∣ n → 104 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l3121_312190


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l3121_312110

/-- Given a function f(x) = ax^2 + 3x - 2, prove that if the slope of the tangent line
    at the point (2, f(2)) is 7, then a = 1. -/
theorem tangent_slope_implies_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 + 3 * x - 2
  (deriv f 2 = 7) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l3121_312110


namespace NUMINAMATH_CALUDE_tensor_equation_solution_l3121_312184

/-- Custom binary operation ⊗ -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem tensor_equation_solution :
  ∀ m : ℝ, m > 0 → tensor 1 m = 3 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_tensor_equation_solution_l3121_312184


namespace NUMINAMATH_CALUDE_stadium_length_l3121_312119

/-- Given a rectangular stadium with perimeter 800 meters and breadth 300 meters, its length is 100 meters. -/
theorem stadium_length (perimeter breadth : ℝ) (h1 : perimeter = 800) (h2 : breadth = 300) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter → perimeter / 2 - breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_l3121_312119


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l3121_312129

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  ∃ m : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l3121_312129


namespace NUMINAMATH_CALUDE_cube_sum_odd_implies_product_odd_l3121_312194

theorem cube_sum_odd_implies_product_odd (n m : ℤ) : 
  Odd (n^3 + m^3) → Odd (n * m) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_odd_implies_product_odd_l3121_312194


namespace NUMINAMATH_CALUDE_range_of_absolute_linear_function_l3121_312182

theorem range_of_absolute_linear_function 
  (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  let f : ℝ → ℝ := fun x ↦ |a * x + b|
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ max (|b|) (|a + b|)) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f x = 0) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f x = max (|b|) (|a + b|)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_absolute_linear_function_l3121_312182


namespace NUMINAMATH_CALUDE_complex_square_sum_of_squares_l3121_312148

theorem complex_square_sum_of_squares (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (↑a + Complex.I * ↑b)^2 = (3 : ℂ) + Complex.I * 4 →
  a^2 + b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_complex_square_sum_of_squares_l3121_312148


namespace NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l3121_312185

theorem greatest_perimeter_of_special_triangle : 
  let is_valid_triangle (x : ℕ) := x + 3*x > 15 ∧ x + 15 > 3*x ∧ 3*x + 15 > x
  let perimeter (x : ℕ) := x + 3*x + 15
  ∀ x : ℕ, is_valid_triangle x → perimeter x ≤ 43 ∧ ∃ y : ℕ, is_valid_triangle y ∧ perimeter y = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l3121_312185


namespace NUMINAMATH_CALUDE_inequality_preservation_l3121_312135

theorem inequality_preservation (a b : ℝ) : a < b → 1 - a > 1 - b := by sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3121_312135


namespace NUMINAMATH_CALUDE_inequality_solution_l3121_312197

theorem inequality_solution (x : ℝ) :
  x ≥ -14 → (x + 2 < Real.sqrt (x + 14) ↔ -14 ≤ x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3121_312197


namespace NUMINAMATH_CALUDE_painting_time_with_break_l3121_312146

/-- The time it takes to paint a room together, including a break -/
theorem painting_time_with_break (doug_rate dave_rate ella_rate : ℝ) 
  (break_time : ℝ) (h1 : doug_rate = 1 / 5) (h2 : dave_rate = 1 / 7) 
  (h3 : ella_rate = 1 / 10) (h4 : break_time = 2) : 
  ∃ t : ℝ, (doug_rate + dave_rate + ella_rate) * (t - break_time) = 1 ∧ t = 132 / 31 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_with_break_l3121_312146


namespace NUMINAMATH_CALUDE_rug_coverage_area_l3121_312199

/-- Given three rugs with specified overlap areas, calculate the total floor area covered. -/
theorem rug_coverage_area (total_rug_area : ℝ) (double_layer_area : ℝ) (triple_layer_area : ℝ)
  (h1 : total_rug_area = 200)
  (h2 : double_layer_area = 24)
  (h3 : triple_layer_area = 19) :
  total_rug_area - double_layer_area - 2 * triple_layer_area = 138 := by
  sorry

#check rug_coverage_area

end NUMINAMATH_CALUDE_rug_coverage_area_l3121_312199


namespace NUMINAMATH_CALUDE_extreme_values_range_c_l3121_312138

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 2*c*x^2 + x

-- Define the property of having extreme values
def has_extreme_values (c : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (3*x₁^2 - 4*c*x₁ + 1 = 0) ∧ 
    (3*x₂^2 - 4*c*x₂ + 1 = 0)

-- State the theorem
theorem extreme_values_range_c :
  ∀ c : ℝ, has_extreme_values c ↔ (c < -Real.sqrt 3 / 2 ∨ c > Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_range_c_l3121_312138


namespace NUMINAMATH_CALUDE_square_side_length_average_l3121_312141

theorem square_side_length_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l3121_312141


namespace NUMINAMATH_CALUDE_square_areas_and_perimeters_l3121_312150

theorem square_areas_and_perimeters (x : ℝ) : 
  (∃ s₁ s₂ : ℝ, 
    s₁^2 = x^2 + 4*x + 4 ∧ 
    s₂^2 = 4*x^2 - 12*x + 9 ∧ 
    4*s₁ + 4*s₂ = 32) → 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_square_areas_and_perimeters_l3121_312150


namespace NUMINAMATH_CALUDE_max_leftover_candy_exists_max_leftover_candy_l3121_312157

theorem max_leftover_candy (x : ℕ) : x % 11 ≤ 10 := by sorry

theorem exists_max_leftover_candy : ∃ x : ℕ, x % 11 = 10 := by sorry

end NUMINAMATH_CALUDE_max_leftover_candy_exists_max_leftover_candy_l3121_312157


namespace NUMINAMATH_CALUDE_work_completion_days_l3121_312103

-- Define the daily work done by a man and a boy
variable (M B : ℝ)

-- Define the total work to be done
variable (W : ℝ)

-- Define the number of days for the first group
variable (D : ℝ)

-- Theorem statement
theorem work_completion_days 
  (h1 : M = 2 * B) -- A man's daily work is twice that of a boy
  (h2 : (13 * M + 24 * B) * 4 = W) -- 13 men and 24 boys complete the work in 4 days
  (h3 : (12 * M + 16 * B) * D = W) -- 12 men and 16 boys complete the work in D days
  : D = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l3121_312103


namespace NUMINAMATH_CALUDE_not_always_prime_l3121_312193

theorem not_always_prime : ∃ n : ℤ, ¬(Nat.Prime (Int.natAbs (n^2 + n + 41))) := by sorry

end NUMINAMATH_CALUDE_not_always_prime_l3121_312193


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3121_312178

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7 : ℝ)^3 + (9 : ℝ)^3 - 100 → 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |x^(1/3) - n| ≤ |x^(1/3) - m| :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3121_312178


namespace NUMINAMATH_CALUDE_juice_boxes_calculation_l3121_312165

/-- Calculates the total number of juice boxes needed for a school year. -/
def total_juice_boxes (num_children : ℕ) (days_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  num_children * days_per_week * weeks_per_year

/-- Proves that the total number of juice boxes needed for the given conditions is 375. -/
theorem juice_boxes_calculation :
  let num_children : ℕ := 3
  let days_per_week : ℕ := 5
  let weeks_per_year : ℕ := 25
  total_juice_boxes num_children days_per_week weeks_per_year = 375 := by
  sorry


end NUMINAMATH_CALUDE_juice_boxes_calculation_l3121_312165


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l3121_312143

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the interval (1, 2]
def interval : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Statement to prove
theorem intersection_equals_interval : M ∩ N = interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l3121_312143


namespace NUMINAMATH_CALUDE_zoo_line_theorem_l3121_312171

/-- The number of ways to arrange 6 people in a line with specific conditions -/
def zoo_line_arrangements : ℕ := 24

/-- Two fathers in a group of 6 people -/
def fathers : ℕ := 2

/-- Two mothers in a group of 6 people -/
def mothers : ℕ := 2

/-- Two children in a group of 6 people -/
def children : ℕ := 2

/-- Total number of people in the group -/
def total_people : ℕ := fathers + mothers + children

theorem zoo_line_theorem :
  zoo_line_arrangements = 24 :=
sorry

end NUMINAMATH_CALUDE_zoo_line_theorem_l3121_312171


namespace NUMINAMATH_CALUDE_flowers_planted_per_day_l3121_312167

theorem flowers_planted_per_day (total_people : ℕ) (total_days : ℕ) (total_flowers : ℕ) 
  (h1 : total_people = 5)
  (h2 : total_days = 2)
  (h3 : total_flowers = 200)
  (h4 : total_people > 0)
  (h5 : total_days > 0) :
  total_flowers / (total_people * total_days) = 20 := by
sorry

end NUMINAMATH_CALUDE_flowers_planted_per_day_l3121_312167


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l3121_312187

theorem imaginary_part_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i)^2 * (2 + i)
  Complex.im z = 4 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l3121_312187


namespace NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l3121_312152

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : 3150 ∣ b) :
  Nat.gcd (Int.natAbs (b^2 + 9*b + 54)) (Int.natAbs (b + 4)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l3121_312152


namespace NUMINAMATH_CALUDE_garden_breadth_l3121_312134

/-- The perimeter of a rectangle given its length and breadth -/
def perimeter (length breadth : ℝ) : ℝ := 2 * (length + breadth)

/-- Theorem: For a rectangular garden with perimeter 500 m and length 150 m, the breadth is 100 m -/
theorem garden_breadth :
  ∃ (breadth : ℝ), perimeter 150 breadth = 500 ∧ breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l3121_312134


namespace NUMINAMATH_CALUDE_provisions_after_reinforcement_l3121_312196

def initial_garrison : ℕ := 2000
def initial_provisions_days : ℕ := 54
def days_before_reinforcement : ℕ := 15
def reinforcement : ℕ := 1900

theorem provisions_after_reinforcement :
  let remaining_days : ℕ := initial_provisions_days - days_before_reinforcement
  let total_men : ℕ := initial_garrison + reinforcement
  let days_after_reinforcement : ℕ := (initial_garrison * remaining_days) / total_men
  days_after_reinforcement = 20 := by sorry

end NUMINAMATH_CALUDE_provisions_after_reinforcement_l3121_312196


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3121_312177

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3121_312177


namespace NUMINAMATH_CALUDE_fibonacci_seventh_term_l3121_312162

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_seventh_term : fibonacci 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_seventh_term_l3121_312162


namespace NUMINAMATH_CALUDE_meeting_arrangement_count_l3121_312105

/-- Represents the number of schools -/
def num_schools : ℕ := 3

/-- Represents the number of members per school -/
def members_per_school : ℕ := 6

/-- Represents the total number of members -/
def total_members : ℕ := num_schools * members_per_school

/-- Represents the number of representatives sent by the host school -/
def host_representatives : ℕ := 2

/-- Represents the number of representatives sent by each non-host school -/
def non_host_representatives : ℕ := 1

/-- The number of ways to arrange the meeting -/
def meeting_arrangements : ℕ := 1620

/-- Theorem stating the number of ways to arrange the meeting -/
theorem meeting_arrangement_count :
  num_schools * (Nat.choose members_per_school host_representatives *
    (Nat.choose members_per_school non_host_representatives)^(num_schools - 1)) = meeting_arrangements :=
by sorry

end NUMINAMATH_CALUDE_meeting_arrangement_count_l3121_312105
