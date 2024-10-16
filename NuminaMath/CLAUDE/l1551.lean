import Mathlib

namespace NUMINAMATH_CALUDE_cos_difference_special_case_l1551_155130

theorem cos_difference_special_case (x₁ x₂ : Real) 
  (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 2 * Real.pi)
  (h4 : Real.sin x₁ = 1/3) (h5 : Real.sin x₂ = 1/3) : 
  Real.cos (x₁ - x₂) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_special_case_l1551_155130


namespace NUMINAMATH_CALUDE_surface_area_comparison_l1551_155114

/-- Given a cube, cylinder, and sphere with equal volumes, their surface areas satisfy S₃ < S₂ < S₁ -/
theorem surface_area_comparison 
  (V : ℝ) 
  (h_V_pos : V > 0) 
  (S₁ : ℝ) 
  (S₂ : ℝ) 
  (S₃ : ℝ) 
  (h_S₁ : S₁ = Real.rpow (216 * V^2) (1/3))
  (h_S₂ : S₂ = Real.rpow (54 * π * V^2) (1/3))
  (h_S₃ : S₃ = Real.rpow (36 * π * V^2) (1/3)) :
  S₃ < S₂ ∧ S₂ < S₁ :=
by sorry

end NUMINAMATH_CALUDE_surface_area_comparison_l1551_155114


namespace NUMINAMATH_CALUDE_inequality_proof_l1551_155160

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_sum : x + y + z = 3 * Real.sqrt 3) :
  (x^2 / (x + 2*y + 3*z)) + (y^2 / (y + 2*z + 3*x)) + (z^2 / (z + 2*x + 3*y)) ≥ Real.sqrt 3 / 2 := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l1551_155160


namespace NUMINAMATH_CALUDE_frank_skee_ball_tickets_proof_l1551_155163

def frank_skee_ball_tickets (whack_a_mole_tickets : ℕ) (candy_cost : ℕ) (candies_bought : ℕ) : ℕ :=
  candies_bought * candy_cost - whack_a_mole_tickets

theorem frank_skee_ball_tickets_proof :
  frank_skee_ball_tickets 33 6 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_frank_skee_ball_tickets_proof_l1551_155163


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l1551_155164

theorem min_value_abs_sum (x : ℚ) : 
  |x - 1| + |x + 3| ≥ 4 ∧ ∃ y : ℚ, |y - 1| + |y + 3| = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l1551_155164


namespace NUMINAMATH_CALUDE_age_ratio_l1551_155147

/-- The ages of John, Mary, and Tonya satisfy certain conditions. -/
def AgeRelations (john mary tonya : ℕ) : Prop :=
  john = tonya / 2 ∧ tonya = 60 ∧ (john + mary + tonya) / 3 = 35

/-- The ratio of John's age to Mary's age is 2:1. -/
theorem age_ratio (john mary tonya : ℕ) 
  (h : AgeRelations john mary tonya) : john = 2 * mary := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l1551_155147


namespace NUMINAMATH_CALUDE_special_triangle_area_l1551_155168

/-- Triangle with specific properties -/
structure SpecialTriangle where
  -- Three sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- One angle is 120°
  angle_120 : ∃ θ, θ = 2 * π / 3
  -- Sides form arithmetic sequence with difference 4
  arithmetic_seq : ∃ x : ℝ, a = x - 4 ∧ b = x ∧ c = x + 4

/-- The area of the special triangle is 15√3 -/
theorem special_triangle_area (t : SpecialTriangle) : 
  (1/2) * t.a * t.b * Real.sqrt 3 = 15 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_area_l1551_155168


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l1551_155159

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l1551_155159


namespace NUMINAMATH_CALUDE_sum_1_to_1000_equals_500500_sum_forward_equals_sum_backward_l1551_155170

def sum_1_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_1_to_1000_equals_500500 :
  sum_1_to_n 1000 = 500500 :=
by sorry

theorem sum_forward_equals_sum_backward (n : ℕ) :
  (List.range n).sum = (List.range n).reverse.sum :=
by sorry

#check sum_1_to_1000_equals_500500
#check sum_forward_equals_sum_backward

end NUMINAMATH_CALUDE_sum_1_to_1000_equals_500500_sum_forward_equals_sum_backward_l1551_155170


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1551_155187

/-- Given a quadratic function y = x^2 - 2px - p with two distinct roots, 
    prove properties about p and the roots. -/
theorem quadratic_function_properties (p : ℝ) 
  (h_distinct : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 2*p*x₁ - p = 0 ∧ x₂^2 - 2*p*x₂ - p = 0) :
  (∃ (x₁ x₂ : ℝ), 2*p*x₁ + x₂^2 + 3*p > 0) ∧
  (∃ (max_p : ℝ), max_p = 9/16 ∧ 
    ∀ (q : ℝ), (∃ (x₁ x₂ : ℝ), x₁^2 - 2*q*x₁ - q = 0 ∧ x₂^2 - 2*q*x₂ - q = 0 ∧ |x₁ - x₂| ≤ |2*q - 3|) 
    → q ≤ max_p) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1551_155187


namespace NUMINAMATH_CALUDE_hundredth_term_of_sequence_l1551_155173

def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem hundredth_term_of_sequence (a₁ d : ℕ) (h₁ : a₁ = 5) (h₂ : d = 4) :
  arithmeticSequence a₁ d 100 = 401 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_of_sequence_l1551_155173


namespace NUMINAMATH_CALUDE_prob_at_least_one_target_l1551_155196

/-- The number of cards in the modified deck -/
def deck_size : ℕ := 54

/-- The number of cards that are diamonds, aces, or jokers -/
def target_cards : ℕ := 18

/-- The probability of drawing a card that is not a diamond, ace, or joker -/
def prob_not_target : ℚ := (deck_size - target_cards) / deck_size

/-- The probability of drawing two cards with replacement, where at least one is a diamond, ace, or joker -/
theorem prob_at_least_one_target : 
  1 - prob_not_target ^ 2 = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_target_l1551_155196


namespace NUMINAMATH_CALUDE_total_frogs_in_pond_l1551_155118

def frogs_on_lilypads : ℕ := 5
def frogs_on_logs : ℕ := 3
def dozen : ℕ := 12
def baby_frogs_dozens : ℕ := 2

theorem total_frogs_in_pond : 
  frogs_on_lilypads + frogs_on_logs + baby_frogs_dozens * dozen = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_frogs_in_pond_l1551_155118


namespace NUMINAMATH_CALUDE_arrange_balls_and_boxes_eq_20_l1551_155184

/-- The number of ways to arrange 5 balls in 5 boxes with exactly two matches -/
def arrange_balls_and_boxes : ℕ :=
  let n : ℕ := 5  -- Total number of balls and boxes
  let k : ℕ := 2  -- Number of matches required
  let derangement_3 : ℕ := 2  -- Number of derangements for 3 elements
  (n.choose k) * derangement_3

/-- Theorem stating that the number of arrangements is 20 -/
theorem arrange_balls_and_boxes_eq_20 : arrange_balls_and_boxes = 20 := by
  sorry

end NUMINAMATH_CALUDE_arrange_balls_and_boxes_eq_20_l1551_155184


namespace NUMINAMATH_CALUDE_expenditure_representation_l1551_155126

/-- Represents a monetary transaction in yuan -/
structure Transaction where
  amount : Int
  deriving Repr

/-- Defines an income transaction -/
def is_income (t : Transaction) : Prop := t.amount > 0

/-- Defines an expenditure transaction -/
def is_expenditure (t : Transaction) : Prop := t.amount < 0

/-- Theorem stating that an expenditure of 50 yuan should be represented as -50 yuan -/
theorem expenditure_representation :
  ∀ (t : Transaction),
    is_expenditure t → t.amount = 50 → t.amount = -50 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_representation_l1551_155126


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_bound_l1551_155180

-- Define the function f(x) = -x^2 + 2ax + 3
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 3

-- Define what it means for a function to be decreasing on an interval
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem decreasing_quadratic_implies_a_bound :
  ∀ a : ℝ, decreasing_on (f a) 2 6 → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_bound_l1551_155180


namespace NUMINAMATH_CALUDE_passing_methods_after_six_passes_l1551_155174

/-- The number of ways the ball can be passed back to player A after n passes -/
def passing_methods (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if n = 2 then 2
  else 2^(n-1) - passing_methods (n-1)

/-- The theorem stating that there are 22 different passing methods after 6 passes -/
theorem passing_methods_after_six_passes :
  passing_methods 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_passing_methods_after_six_passes_l1551_155174


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l1551_155132

theorem christmas_tree_lights (T : ℝ) : ∃ (R Y G B : ℝ),
  R = 0.30 * T ∧
  Y = 0.45 * T ∧
  G = 110 ∧
  T = R + Y + G + B ∧
  B = 0.25 * T - 110 :=
by sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l1551_155132


namespace NUMINAMATH_CALUDE_triangle_angle_and_parameter_l1551_155175

/-- Given a triangle ABC where tan A and tan B are real roots of a quadratic equation,
    prove that angle C is 60° and find the value of p. -/
theorem triangle_angle_and_parameter
  (A B C : ℝ) (p : ℝ) (AB AC : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (h_roots : ∃ (x y : ℝ), x^2 + Real.sqrt 3 * p * x - p + 1 = 0 ∧
                          y^2 + Real.sqrt 3 * p * y - p + 1 = 0 ∧
                          x = Real.tan A ∧ y = Real.tan B)
  (h_AB : AB = 3)
  (h_AC : AC = Real.sqrt 6) :
  C = Real.pi / 3 ∧ p = -1 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_and_parameter_l1551_155175


namespace NUMINAMATH_CALUDE_opposite_reciprocal_expression_zero_l1551_155103

theorem opposite_reciprocal_expression_zero
  (a b c d : ℝ)
  (h1 : a = -b)
  (h2 : c = 1 / d)
  : 2 * c - a - 2 / d - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_expression_zero_l1551_155103


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l1551_155166

def number := 102 * 104 * 107 * 108

theorem distinct_prime_factors_count :
  Nat.card (Nat.factors number).toFinset = 5 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l1551_155166


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1551_155125

theorem expand_and_simplify (y : ℝ) : 5 * (4 * y^2 - 3 * y + 2) = 20 * y^2 - 15 * y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1551_155125


namespace NUMINAMATH_CALUDE_tagged_fish_count_l1551_155177

/-- The number of tagged fish found in the second catch -/
def tagged_fish_in_second_catch (total_fish : ℕ) (initially_tagged : ℕ) (second_catch : ℕ) : ℕ :=
  (initially_tagged * second_catch) / total_fish

/-- Proof that the number of tagged fish in the second catch is 2 -/
theorem tagged_fish_count :
  let total_fish : ℕ := 1800
  let initially_tagged : ℕ := 60
  let second_catch : ℕ := 60
  tagged_fish_in_second_catch total_fish initially_tagged second_catch = 2 := by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_count_l1551_155177


namespace NUMINAMATH_CALUDE_black_and_white_cartridge_cost_l1551_155108

/-- The cost of a black-and-white printer cartridge -/
def black_and_white_cost : ℕ := sorry

/-- The cost of a color printer cartridge -/
def color_cost : ℕ := 32

/-- The total cost of printer cartridges -/
def total_cost : ℕ := 123

/-- The number of color cartridges needed -/
def num_color_cartridges : ℕ := 3

/-- The number of black-and-white cartridges needed -/
def num_black_and_white_cartridges : ℕ := 1

theorem black_and_white_cartridge_cost :
  black_and_white_cost = 27 :=
by sorry

end NUMINAMATH_CALUDE_black_and_white_cartridge_cost_l1551_155108


namespace NUMINAMATH_CALUDE_leak_empty_time_l1551_155191

def tank_capacity : ℝ := 1
def fill_time_no_leak : ℝ := 3
def empty_time_leak : ℝ := 12

theorem leak_empty_time :
  let fill_rate : ℝ := tank_capacity / fill_time_no_leak
  let leak_rate : ℝ := tank_capacity / empty_time_leak
  tank_capacity / leak_rate = empty_time_leak := by
sorry

end NUMINAMATH_CALUDE_leak_empty_time_l1551_155191


namespace NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l1551_155136

/-- The floor on which Vasya lives -/
def vasya_floor (petya_steps : ℕ) (vasya_steps : ℕ) : ℕ :=
  let steps_per_floor := petya_steps / 2
  1 + vasya_steps / steps_per_floor

/-- Theorem stating that Vasya lives on the 5th floor -/
theorem vasya_lives_on_fifth_floor :
  vasya_floor 36 72 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l1551_155136


namespace NUMINAMATH_CALUDE_polynomial_sum_l1551_155189

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) : 
  (∃ (x : ℝ), f a b x = g c d x ∧ f a b x = -25 ∧ x = 50) →  -- f and g intersect at (50, -25)
  (∀ (x : ℝ), f a b x ≥ -25) →  -- minimum value of f is -25
  (∀ (x : ℝ), g c d x ≥ -25) →  -- minimum value of g is -25
  g c d (-a/2) = 0 →  -- vertex of f is root of g
  f a b (-c/2) = 0 →  -- vertex of g is root of f
  a ≠ c →  -- f and g are distinct
  a + c = -101 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1551_155189


namespace NUMINAMATH_CALUDE_function_values_unbounded_l1551_155145

/-- A function satisfying the given identity -/
def SatisfiesIdentity (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f (n, m) = (f (n - 1, m) + f (n + 1, m) + f (n, m - 1) + f (n, m + 1)) / 4

/-- The main theorem -/
theorem function_values_unbounded
  (f : ℤ × ℤ → ℤ)
  (h_satisfies : SatisfiesIdentity f)
  (h_nonconstant : ∃ (a b c d : ℤ), f (a, b) ≠ f (c, d)) :
  ∀ k : ℤ, (∃ n m : ℤ, f (n, m) > k) ∧ (∃ n m : ℤ, f (n, m) < k) :=
sorry

end NUMINAMATH_CALUDE_function_values_unbounded_l1551_155145


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1551_155121

/-- The sum of 0.222... and 0.0202... equals 8/33 -/
theorem repeating_decimal_sum : 
  let a : ℚ := 2/9  -- represents 0.222...
  let b : ℚ := 2/99 -- represents 0.0202...
  a + b = 8/33 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1551_155121


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l1551_155140

theorem smaller_root_of_equation : 
  let f (x : ℝ) := (x - 1/3)^2 + (x - 1/3)*(x - 2/3)
  ∃ y, f y = 0 ∧ y ≤ 1/3 ∧ ∀ z, f z = 0 → z ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l1551_155140


namespace NUMINAMATH_CALUDE_f_inequality_implies_m_bound_l1551_155101

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Theorem statement
theorem f_inequality_implies_m_bound (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < -m + 4) →
  m < 1/7 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_implies_m_bound_l1551_155101


namespace NUMINAMATH_CALUDE_ram_gopal_ratio_l1551_155113

theorem ram_gopal_ratio (ram_money : ℕ) (krishan_money : ℕ) (gopal_krishan_ratio : Rat) :
  ram_money = 735 →
  krishan_money = 4335 →
  gopal_krishan_ratio = 7 / 17 →
  (ram_money : Rat) / ((gopal_krishan_ratio * krishan_money) : Rat) = 7 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ram_gopal_ratio_l1551_155113


namespace NUMINAMATH_CALUDE_cherry_pie_problem_l1551_155169

/-- The number of single cherries in one pound of cherries -/
def cherries_per_pound (total_cherries : ℕ) (total_pounds : ℕ) : ℕ :=
  total_cherries / total_pounds

/-- The number of cherries that can be pitted in a given time -/
def cherries_pitted (time_minutes : ℕ) (cherries_per_10_min : ℕ) : ℕ :=
  (time_minutes / 10) * cherries_per_10_min

theorem cherry_pie_problem (pounds_needed : ℕ) (pitting_time_hours : ℕ) (cherries_per_10_min : ℕ) :
  pounds_needed = 3 →
  pitting_time_hours = 2 →
  cherries_per_10_min = 20 →
  cherries_per_pound (cherries_pitted (pitting_time_hours * 60) cherries_per_10_min) pounds_needed = 80 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_problem_l1551_155169


namespace NUMINAMATH_CALUDE_no_three_digit_perfect_square_difference_l1551_155141

theorem no_three_digit_perfect_square_difference :
  ¬ ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    ∃ (k : ℕ), (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = k^2 :=
by sorry

end NUMINAMATH_CALUDE_no_three_digit_perfect_square_difference_l1551_155141


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_5_l1551_155176

theorem smallest_positive_integer_ending_in_6_divisible_by_5 :
  ∃ (n : ℕ), n > 0 ∧ n % 10 = 6 ∧ n % 5 = 0 ∧
  ∀ (m : ℕ), m > 0 → m % 10 = 6 → m % 5 = 0 → m ≥ n :=
by
  use 46
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_5_l1551_155176


namespace NUMINAMATH_CALUDE_max_y_coordinate_cos_2theta_l1551_155134

/-- The maximum y-coordinate of a point on the curve r = cos 2θ in polar coordinates -/
theorem max_y_coordinate_cos_2theta : 
  let r : ℝ → ℝ := λ θ ↦ Real.cos (2 * θ)
  let x : ℝ → ℝ := λ θ ↦ r θ * Real.cos θ
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), y θ ≤ y θ_max ∧ y θ_max = 3 * Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_max_y_coordinate_cos_2theta_l1551_155134


namespace NUMINAMATH_CALUDE_angle_hda_measure_l1551_155129

-- Define the points
variable (A B C D E F G H I : Point)

-- Define the shapes
def is_square (A B C D : Point) : Prop := sorry
def is_equilateral_triangle (C D E : Point) : Prop := sorry
def is_regular_hexagon (D E F G H I : Point) : Prop := sorry
def is_isosceles_triangle (G H I : Point) : Prop := sorry

-- Define the angle measure function
def angle_measure (A B C : Point) : ℝ := sorry

-- Theorem statement
theorem angle_hda_measure 
  (h1 : is_square A B C D)
  (h2 : is_equilateral_triangle C D E)
  (h3 : is_regular_hexagon D E F G H I)
  (h4 : is_isosceles_triangle G H I) :
  angle_measure H D A = 270 := by sorry

end NUMINAMATH_CALUDE_angle_hda_measure_l1551_155129


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_l1551_155138

theorem cubic_expansion_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x, (2*x + 1)^3 = a₀*x^3 + a₁*x^2 + a₂*x + a₃) →
  a₁ + a₃ = 13 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_l1551_155138


namespace NUMINAMATH_CALUDE_decimal_17_to_binary_l1551_155185

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else go (m / 2) ((m % 2 = 1) :: acc)
    go n []

/-- Converts a list of bits to a string representation of a binary number -/
def binaryToString (bits : List Bool) : String :=
  bits.map (fun b => if b then '1' else '0') |> String.mk

theorem decimal_17_to_binary :
  binaryToString (toBinary 17) = "10001" := by
  sorry

end NUMINAMATH_CALUDE_decimal_17_to_binary_l1551_155185


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l1551_155188

theorem no_prime_roots_for_quadratic : 
  ¬ ∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    p ≠ q ∧
    (p : ℤ) + q = 107 ∧ 
    (p : ℤ) * q = k ∧
    p^2 - 107*p + k = 0 ∧ 
    q^2 - 107*q + k = 0 :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l1551_155188


namespace NUMINAMATH_CALUDE_tan_half_product_l1551_155152

theorem tan_half_product (a b : Real) :
  7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 2) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = 4 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_l1551_155152


namespace NUMINAMATH_CALUDE_sum_of_roots_l1551_155137

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 2026*x = 2023)
  (hy : y^3 + 6*y^2 + 2035*y = -4053) : 
  x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1551_155137


namespace NUMINAMATH_CALUDE_factor_to_increase_average_l1551_155156

theorem factor_to_increase_average (numbers : Finset ℝ) (factor : ℝ) : 
  Finset.card numbers = 5 →
  6 ∈ numbers →
  (Finset.sum numbers id) / 5 = 6.8 →
  ((Finset.sum numbers id) - 6 + 6 * factor) / 5 = 9.2 →
  factor = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_to_increase_average_l1551_155156


namespace NUMINAMATH_CALUDE_cubic_polynomial_existence_l1551_155194

theorem cubic_polynomial_existence (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ f : ℤ → ℤ, 
    (∃ p q r : ℤ, ∀ x, f x = p * x^3 + q * x^2 + r * x + (a * b * c)) ∧ 
    (p > 0) ∧
    (f a = a^3) ∧ (f b = b^3) ∧ (f c = c^3) :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_existence_l1551_155194


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1551_155179

theorem stratified_sampling_theorem (total_population : ℕ) (sample_size : ℕ) (stratum_size : ℕ) 
  (h1 : total_population = 500) 
  (h2 : sample_size = 100) 
  (h3 : stratum_size = 95) :
  (stratum_size : ℚ) / total_population * sample_size = 19 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1551_155179


namespace NUMINAMATH_CALUDE_remainder_theorem_l1551_155197

theorem remainder_theorem (n : ℤ) (h : n % 13 = 3) : (5 * n - 11) % 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1551_155197


namespace NUMINAMATH_CALUDE_expression_factorization_l1551_155100

theorem expression_factorization (x : ℝ) :
  (16 * x^7 + 49 * x^5 - 9) - (4 * x^7 - 7 * x^5 - 9) = 4 * x^5 * (3 * x^2 + 14) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1551_155100


namespace NUMINAMATH_CALUDE_simplify_sqrt_plus_x_l1551_155122

theorem simplify_sqrt_plus_x (x : ℝ) (h : 1 < x ∧ x < 2) : 
  Real.sqrt ((x - 2)^2) + x = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_plus_x_l1551_155122


namespace NUMINAMATH_CALUDE_space_creature_perimeter_calc_l1551_155144

/-- The perimeter of a space creature, which is a sector of a circle --/
def space_creature_perimeter (r : ℝ) (central_angle : ℝ) : ℝ :=
  r * central_angle + 2 * r

/-- Theorem: The perimeter of the space creature with radius 2 cm and central angle 270° is 3π + 4 cm --/
theorem space_creature_perimeter_calc :
  space_creature_perimeter 2 (3 * π / 2) = 3 * π + 4 := by
  sorry

#check space_creature_perimeter_calc

end NUMINAMATH_CALUDE_space_creature_perimeter_calc_l1551_155144


namespace NUMINAMATH_CALUDE_james_bought_ten_shirts_l1551_155128

/-- Represents the number of shirts James bought -/
def num_shirts : ℕ := 10

/-- Represents the number of pants James bought -/
def num_pants : ℕ := num_shirts / 2

/-- Represents the cost of a single shirt in dollars -/
def shirt_cost : ℕ := 6

/-- Represents the cost of a single pair of pants in dollars -/
def pants_cost : ℕ := 8

/-- Represents the total cost of the purchase in dollars -/
def total_cost : ℕ := 100

/-- Theorem stating that given the conditions, James bought 10 shirts -/
theorem james_bought_ten_shirts : 
  num_shirts * shirt_cost + num_pants * pants_cost = total_cost ∧ 
  num_shirts = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_bought_ten_shirts_l1551_155128


namespace NUMINAMATH_CALUDE_collinear_points_l1551_155150

/-- Three points are collinear if the slopes between any two pairs of points are equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem statement -/
theorem collinear_points : 
  ∃ b : ℝ, collinear 4 (-6) (2*b + 1) 4 (-3*b + 2) 1 ∧ b = -1/44 :=
by sorry

end NUMINAMATH_CALUDE_collinear_points_l1551_155150


namespace NUMINAMATH_CALUDE_sum_of_digits_is_three_l1551_155162

/-- Represents a 100-digit number with repeating pattern 5050 --/
def a : ℕ := 5050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050

/-- Represents a 100-digit number with repeating pattern 7070 --/
def b : ℕ := 7070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070

/-- The product of a and b --/
def product : ℕ := a * b

/-- Extracts the thousands digit from a number --/
def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10

/-- Extracts the units digit from a number --/
def units_digit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the sum of the thousands digit and units digit of the product is 3 --/
theorem sum_of_digits_is_three : 
  thousands_digit product + units_digit product = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_is_three_l1551_155162


namespace NUMINAMATH_CALUDE_partnership_capital_share_l1551_155139

theorem partnership_capital_share (Y : ℚ) : 
  (1 / 3 : ℚ) + (1 / 4 : ℚ) + Y + (1 - ((1 / 3 : ℚ) + (1 / 4 : ℚ) + Y)) = 1 →
  (1 / 3 : ℚ) + (1 / 4 : ℚ) + Y = 1 →
  Y = 5 / 12 := by
sorry

end NUMINAMATH_CALUDE_partnership_capital_share_l1551_155139


namespace NUMINAMATH_CALUDE_conference_handshakes_l1551_155190

/-- The number of handshakes in a conference with multiple companies --/
def num_handshakes (num_companies : ℕ) (representatives_per_company : ℕ) : ℕ :=
  let total_people := num_companies * representatives_per_company
  let handshakes_per_person := total_people - representatives_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a conference with 5 companies, each having 5 representatives,
    where every person shakes hands once with every person except those from
    their own company, the total number of handshakes is 250. --/
theorem conference_handshakes :
  num_handshakes 5 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1551_155190


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1551_155161

theorem shaded_area_calculation (carpet_side : ℝ) (large_square_side : ℝ) (small_square_side : ℝ) :
  carpet_side = 12 →
  carpet_side / large_square_side = 2 →
  large_square_side / small_square_side = 2 →
  12 * (small_square_side ^ 2) + large_square_side ^ 2 = 144 := by
  sorry

#check shaded_area_calculation

end NUMINAMATH_CALUDE_shaded_area_calculation_l1551_155161


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l1551_155198

theorem modular_arithmetic_problem :
  ∃ (x y : ℤ), 
    (7 * x ≡ 1 [ZMOD 56]) ∧ 
    (13 * y ≡ 1 [ZMOD 56]) ∧ 
    (3 * x + 9 * y ≡ 39 [ZMOD 56]) ∧ 
    (0 ≤ (3 * x + 9 * y) % 56) ∧ 
    ((3 * x + 9 * y) % 56 < 56) := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l1551_155198


namespace NUMINAMATH_CALUDE_q_div_p_eq_225_l1551_155183

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The number of cards with each number -/
def cards_per_number : ℕ := 5

/-- The probability of drawing five cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (total_cards.choose cards_drawn : ℚ)

/-- The probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (distinct_numbers * (distinct_numbers - 1) * cards_per_number * cards_per_number : ℚ) / (total_cards.choose cards_drawn : ℚ)

/-- The ratio of q to p is 225 -/
theorem q_div_p_eq_225 : q / p = 225 := by sorry

end NUMINAMATH_CALUDE_q_div_p_eq_225_l1551_155183


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l1551_155142

theorem smallest_divisible_by_15_16_18 : 
  ∃ n : ℕ, (n > 0) ∧ 
           (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) ∧ 
           (∀ m : ℕ, (m > 0) ∧ (15 ∣ m) ∧ (16 ∣ m) ∧ (18 ∣ m) → n ≤ m) ∧
           n = 720 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l1551_155142


namespace NUMINAMATH_CALUDE_brent_candy_count_l1551_155153

/-- The number of pieces of candy Brent has left after trick-or-treating and giving some away. -/
def candy_left : ℕ :=
  let kit_kat := 5
  let hershey := 3 * kit_kat
  let nerds := 8
  let lollipops := 11
  let baby_ruth := 10
  let reeses := baby_ruth / 2
  let total := kit_kat + hershey + nerds + lollipops + baby_ruth + reeses
  let given_away := 5
  total - given_away

/-- Theorem stating that Brent has 49 pieces of candy left. -/
theorem brent_candy_count : candy_left = 49 := by
  sorry

end NUMINAMATH_CALUDE_brent_candy_count_l1551_155153


namespace NUMINAMATH_CALUDE_game_score_problem_l1551_155109

theorem game_score_problem (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 30 →
  correct_points = 7 →
  incorrect_points = -12 →
  total_score = 77 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_answers * correct_points + (total_questions - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 23 := by
  sorry

end NUMINAMATH_CALUDE_game_score_problem_l1551_155109


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1551_155158

theorem fraction_to_decimal : (7 : ℚ) / 16 = (4375 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1551_155158


namespace NUMINAMATH_CALUDE_abs_neg_a_eq_three_l1551_155146

theorem abs_neg_a_eq_three (a : ℝ) : |(-a)| = 3 → a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_a_eq_three_l1551_155146


namespace NUMINAMATH_CALUDE_fraction_simplification_l1551_155165

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 4/7) 
  (hy : y = 5/8) : 
  (6*x - 4*y) / (36*x*y) = 13/180 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1551_155165


namespace NUMINAMATH_CALUDE_nancy_tortilla_chips_l1551_155102

/-- Nancy's tortilla chip distribution problem -/
theorem nancy_tortilla_chips : ∀ (initial brother sister : ℕ),
  initial = 22 →
  brother = 7 →
  sister = 5 →
  initial - (brother + sister) = 10 := by
  sorry

end NUMINAMATH_CALUDE_nancy_tortilla_chips_l1551_155102


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1551_155171

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ |x| ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1551_155171


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1551_155112

def M : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | x > 2}

theorem intersection_complement_theorem :
  M ∩ (Nᶜ) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1551_155112


namespace NUMINAMATH_CALUDE_A_intersect_B_l1551_155181

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem A_intersect_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1551_155181


namespace NUMINAMATH_CALUDE_difference_between_numbers_difference_is_1356_l1551_155106

theorem difference_between_numbers : ℝ → Prop :=
  fun diff : ℝ =>
    let smaller : ℝ := 268.2
    let larger : ℝ := 6 * smaller + 15
    diff = larger - smaller

theorem difference_is_1356 : difference_between_numbers 1356 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_numbers_difference_is_1356_l1551_155106


namespace NUMINAMATH_CALUDE_exists_1990_edge_no_triangle_l1551_155105

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  -- We don't need to define the internal structure of the polyhedron
  -- for this theorem statement.

/-- The number of edges in a polyhedron. -/
def num_edges (p : ConvexPolyhedron) : ℕ := sorry

/-- Predicate to check if a polyhedron has any triangular faces. -/
def has_triangular_face (p : ConvexPolyhedron) : Prop := sorry

/-- Theorem stating the existence of a convex polyhedron with 1990 edges and no triangular faces. -/
theorem exists_1990_edge_no_triangle : ∃ (p : ConvexPolyhedron), num_edges p = 1990 ∧ ¬has_triangular_face p :=
sorry

end NUMINAMATH_CALUDE_exists_1990_edge_no_triangle_l1551_155105


namespace NUMINAMATH_CALUDE_nicki_total_distance_l1551_155199

/-- Represents Nicki's exercise regime for a year --/
structure ExerciseRegime where
  running_miles_first_3_months : ℕ
  running_miles_next_3_months : ℕ
  running_miles_last_6_months : ℕ
  swimming_miles_first_6_months : ℕ
  hiking_miles_per_rest_week : ℕ
  weeks_in_year : ℕ
  weeks_per_month : ℕ

/-- Calculates the total distance covered in all exercises during the year --/
def totalDistance (regime : ExerciseRegime) : ℕ :=
  let running_weeks_per_month := regime.weeks_per_month - 1
  let running_miles := 
    (running_weeks_per_month * 3 * regime.running_miles_first_3_months) +
    (running_weeks_per_month * 3 * regime.running_miles_next_3_months) +
    (running_weeks_per_month * 6 * regime.running_miles_last_6_months)
  let swimming_miles := running_weeks_per_month * 6 * regime.swimming_miles_first_6_months
  let rest_weeks := regime.weeks_in_year / 4
  let hiking_miles := rest_weeks * regime.hiking_miles_per_rest_week
  running_miles + swimming_miles + hiking_miles

/-- Theorem stating that Nicki's total distance is 1095 miles --/
theorem nicki_total_distance :
  ∃ (regime : ExerciseRegime),
    regime.running_miles_first_3_months = 10 ∧
    regime.running_miles_next_3_months = 20 ∧
    regime.running_miles_last_6_months = 30 ∧
    regime.swimming_miles_first_6_months = 5 ∧
    regime.hiking_miles_per_rest_week = 15 ∧
    regime.weeks_in_year = 52 ∧
    regime.weeks_per_month = 4 ∧
    totalDistance regime = 1095 := by
  sorry

end NUMINAMATH_CALUDE_nicki_total_distance_l1551_155199


namespace NUMINAMATH_CALUDE_largest_number_l1551_155123

-- Define the numbers as real numbers
def A : ℝ := 8.03456
def B : ℝ := 8.034666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666
def C : ℝ := 8.034545454545454545454545454545454545454545454545454545454545454545454545454545454545454545454545454545
def D : ℝ := 8.034563456345634563456345634563456345634563456345634563456345634563456345634563456345634563456345634563456
def E : ℝ := 8.034560345603456034560345603456034560345603456034560345603456034560345603456034560345603456034560345603456

-- Theorem statement
theorem largest_number : B > A ∧ B > C ∧ B > D ∧ B > E := by sorry

end NUMINAMATH_CALUDE_largest_number_l1551_155123


namespace NUMINAMATH_CALUDE_substitution_method_simplification_l1551_155195

theorem substitution_method_simplification (x y : ℝ) :
  (4 * x - 3 * y = -1) ∧ (5 * x + y = 13) →
  y = 13 - 5 * x := by
sorry

end NUMINAMATH_CALUDE_substitution_method_simplification_l1551_155195


namespace NUMINAMATH_CALUDE_dividend_calculation_l1551_155111

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 10 * quotient)
  (h2 : divisor = 5 * remainder)
  (h3 : remainder = 46) :
  divisor * quotient + remainder = 5336 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1551_155111


namespace NUMINAMATH_CALUDE_smallest_land_fraction_for_120_members_l1551_155120

/-- Represents a noble family with land inheritance rules -/
structure NobleFamily :=
  (total_members : ℕ)
  (has_original_plot : Bool)

/-- The smallest fraction of land a family member can receive -/
def smallest_land_fraction (family : NobleFamily) : ℚ :=
  1 / (2 * 3^39)

/-- Theorem stating the smallest possible land fraction for a family of 120 members -/
theorem smallest_land_fraction_for_120_members 
  (family : NobleFamily) 
  (h1 : family.total_members = 120) 
  (h2 : family.has_original_plot = true) : 
  smallest_land_fraction family = 1 / (2 * 3^39) := by
  sorry

end NUMINAMATH_CALUDE_smallest_land_fraction_for_120_members_l1551_155120


namespace NUMINAMATH_CALUDE_centers_form_square_l1551_155133

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents a square -/
structure Square :=
  (center : Point)
  (side_length : ℝ)

/-- Function to construct squares on the sides of a parallelogram -/
def construct_squares (p : Parallelogram) : Square × Square × Square × Square :=
  sorry

/-- Function to check if four points form a square -/
def is_square (p q r s : Point) : Prop :=
  sorry

/-- Theorem: The centers of squares constructed on the sides of a parallelogram form a square -/
theorem centers_form_square (p : Parallelogram) :
  let (sq1, sq2, sq3, sq4) := construct_squares p
  is_square sq1.center sq2.center sq3.center sq4.center :=
sorry

end NUMINAMATH_CALUDE_centers_form_square_l1551_155133


namespace NUMINAMATH_CALUDE_ben_money_after_seven_days_l1551_155157

/-- Ben's daily allowance -/
def daily_allowance : ℕ := 50

/-- Ben's daily spending -/
def daily_spending : ℕ := 15

/-- Number of days -/
def num_days : ℕ := 7

/-- Ben's daily savings -/
def daily_savings : ℕ := daily_allowance - daily_spending

/-- Ben's total savings before mom's contribution -/
def initial_savings : ℕ := daily_savings * num_days

/-- Ben's savings after mom's contribution -/
def savings_after_mom : ℕ := 2 * initial_savings

/-- Dad's contribution -/
def dad_contribution : ℕ := 10

/-- Ben's final amount -/
def ben_final_amount : ℕ := savings_after_mom + dad_contribution

theorem ben_money_after_seven_days : ben_final_amount = 500 := by
  sorry

end NUMINAMATH_CALUDE_ben_money_after_seven_days_l1551_155157


namespace NUMINAMATH_CALUDE_points_earned_proof_l1551_155117

def video_game_points (total_enemies : ℕ) (points_per_enemy : ℕ) (enemies_not_destroyed : ℕ) : ℕ :=
  (total_enemies - enemies_not_destroyed) * points_per_enemy

theorem points_earned_proof :
  video_game_points 8 5 6 = 10 := by
sorry

end NUMINAMATH_CALUDE_points_earned_proof_l1551_155117


namespace NUMINAMATH_CALUDE_square_roots_to_N_l1551_155135

theorem square_roots_to_N (m : ℝ) (N : ℝ) : 
  (3 * m - 4) ^ 2 = N ∧ (7 - 4 * m) ^ 2 = N → N = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_to_N_l1551_155135


namespace NUMINAMATH_CALUDE_least_subtrahend_proof_l1551_155119

/-- The product of the first four prime numbers -/
def product_of_first_four_primes : ℕ := 2 * 3 * 5 * 7

/-- The original number from which we subtract -/
def original_number : ℕ := 427751

/-- The least number to be subtracted -/
def least_subtrahend : ℕ := 91

theorem least_subtrahend_proof :
  (∀ k : ℕ, k < least_subtrahend → ¬((original_number - k) % product_of_first_four_primes = 0)) ∧
  ((original_number - least_subtrahend) % product_of_first_four_primes = 0) :=
sorry

end NUMINAMATH_CALUDE_least_subtrahend_proof_l1551_155119


namespace NUMINAMATH_CALUDE_ten_gentlemen_hat_probability_l1551_155192

/-- The harmonic number H_n is defined as the sum of reciprocals of the first n positive integers. -/
def harmonicNumber (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

/-- The probability that n gentlemen each receive their own hat when distributed randomly. -/
def hatProbability (n : ℕ) : ℚ :=
  if n = 0 then 1 else
  (Finset.range (n - 1)).prod (fun i => harmonicNumber (i + 2) / (i + 2 : ℚ))

/-- Theorem stating the probability that 10 gentlemen each receive their own hat. -/
theorem ten_gentlemen_hat_probability :
  ∃ (p : ℚ), hatProbability 10 = p ∧ 0.000515 < p ∧ p < 0.000517 := by
  sorry


end NUMINAMATH_CALUDE_ten_gentlemen_hat_probability_l1551_155192


namespace NUMINAMATH_CALUDE_final_position_of_A_l1551_155116

-- Define the initial position of point A
def initial_position : ℝ := -3

-- Define the movement in the positive direction
def movement : ℝ := 4.5

-- Theorem to prove the final position of point A
theorem final_position_of_A : initial_position + movement = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_final_position_of_A_l1551_155116


namespace NUMINAMATH_CALUDE_area_of_region_is_24_l1551_155110

/-- The region in the plane defined by the given inequality -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (|p.1| + |3 * p.2| - 6) * (|3 * p.1| + |p.2| - 6) ≤ 0}

/-- The area of the region -/
def AreaOfRegion : ℝ := sorry

/-- Theorem stating that the area of the region is 24 -/
theorem area_of_region_is_24 : AreaOfRegion = 24 := by sorry

end NUMINAMATH_CALUDE_area_of_region_is_24_l1551_155110


namespace NUMINAMATH_CALUDE_original_line_equation_l1551_155186

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shift a line horizontally -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - l.slope * shift }

theorem original_line_equation (l : Line) :
  (shift_line l 2).slope = 2 ∧ (shift_line l 2).intercept = 3 →
  l.slope = 2 ∧ l.intercept = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_line_equation_l1551_155186


namespace NUMINAMATH_CALUDE_cube_volume_after_cylinder_removal_l1551_155131

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_volume_after_cylinder_removal (cube_side : ℝ) (cylinder_radius : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  let cube_volume := cube_side ^ 3
  let cylinder_volume := π * cylinder_radius ^ 2 * cube_side
  cube_volume - cylinder_volume = 216 - 54 * π := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_after_cylinder_removal_l1551_155131


namespace NUMINAMATH_CALUDE_large_cube_surface_area_l1551_155124

theorem large_cube_surface_area 
  (num_small_cubes : ℕ) 
  (small_cube_edge : ℝ) 
  (large_cube_edge : ℝ) :
  num_small_cubes = 27 →
  small_cube_edge = 4 →
  large_cube_edge = small_cube_edge * (num_small_cubes ^ (1/3 : ℝ)) →
  6 * large_cube_edge^2 = 864 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_surface_area_l1551_155124


namespace NUMINAMATH_CALUDE_mean_home_runs_l1551_155127

def num_players : List ℕ := [7, 5, 4, 2, 1]
def home_runs : List ℕ := [5, 6, 8, 9, 11]

theorem mean_home_runs : 
  (List.sum (List.zipWith (· * ·) num_players home_runs)) / (List.sum num_players) = 126 / 19 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l1551_155127


namespace NUMINAMATH_CALUDE_library_visitors_l1551_155167

theorem library_visitors (visitors_non_sunday : ℕ) (avg_visitors_per_day : ℕ) :
  visitors_non_sunday = 140 →
  avg_visitors_per_day = 200 →
  ∃ (visitors_sunday : ℕ),
    5 * visitors_sunday + 25 * visitors_non_sunday = 30 * avg_visitors_per_day ∧
    visitors_sunday = 500 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_l1551_155167


namespace NUMINAMATH_CALUDE_tall_trees_indeterminate_l1551_155182

/-- Represents the number of trees in the park -/
structure ParkTrees where
  short_current : ℕ
  short_planted : ℕ
  short_after : ℕ
  tall : ℕ

/-- The given information about the trees in the park -/
def park_info : ParkTrees where
  short_current := 41
  short_planted := 57
  short_after := 98
  tall := 0  -- We use 0 as a placeholder since the number is unknown

/-- Theorem stating that the number of tall trees cannot be determined -/
theorem tall_trees_indeterminate (park : ParkTrees) 
    (h1 : park.short_current = park_info.short_current)
    (h2 : park.short_planted = park_info.short_planted)
    (h3 : park.short_after = park_info.short_after)
    (h4 : park.short_after = park.short_current + park.short_planted) :
    ∀ n : ℕ, ∃ p : ParkTrees, p.short_current = park.short_current ∧ 
                               p.short_planted = park.short_planted ∧ 
                               p.short_after = park.short_after ∧ 
                               p.tall = n :=
by sorry

end NUMINAMATH_CALUDE_tall_trees_indeterminate_l1551_155182


namespace NUMINAMATH_CALUDE_right_triangle_set_l1551_155115

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def set_A : Fin 3 → ℕ := ![4, 5, 6]
def set_B : Fin 3 → ℕ := ![12, 16, 20]
def set_C : Fin 3 → ℕ := ![5, 10, 13]
def set_D : Fin 3 → ℕ := ![8, 40, 41]

/-- The main theorem --/
theorem right_triangle_set :
  (¬ is_right_triangle (set_A 0) (set_A 1) (set_A 2)) ∧
  (is_right_triangle (set_B 0) (set_B 1) (set_B 2)) ∧
  (¬ is_right_triangle (set_C 0) (set_C 1) (set_C 2)) ∧
  (¬ is_right_triangle (set_D 0) (set_D 1) (set_D 2)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_set_l1551_155115


namespace NUMINAMATH_CALUDE_prime_representation_l1551_155172

theorem prime_representation (k : ℕ) (h : k ∈ Finset.range 7 \ {0}) :
  (∀ p : ℕ, Prime p → 
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p^2 = a^2 + k*b^2) → 
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ p = x^2 + k*y^2)) ↔ 
  k ∈ ({1, 2, 4} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_prime_representation_l1551_155172


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l1551_155151

/-- The remaining volume of a cube after removing a cylindrical section --/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_angle : ℝ) : 
  cube_side = 5 → 
  cylinder_radius = 1 → 
  cylinder_angle = Real.pi / 4 → 
  (cube_side ^ 3 - π * cylinder_radius ^ 2 * cube_side * Real.sqrt 2) = 125 - 5 * Real.sqrt 2 * π :=
by sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l1551_155151


namespace NUMINAMATH_CALUDE_cafeteria_apples_l1551_155143

theorem cafeteria_apples (initial_apples : ℕ) : 
  (initial_apples - 2 + 23 = 38) → initial_apples = 17 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l1551_155143


namespace NUMINAMATH_CALUDE_division_reduction_l1551_155104

theorem division_reduction (x : ℝ) (h : x > 0) : 45 / x = 45 - 30 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_reduction_l1551_155104


namespace NUMINAMATH_CALUDE_f_min_value_l1551_155178

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x + y

/-- The minimum value of the function f -/
def min_value : ℝ := 3.7391

theorem f_min_value :
  ∀ x y : ℝ, f x y ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l1551_155178


namespace NUMINAMATH_CALUDE_coin_toss_probability_l1551_155148

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem coin_toss_probability :
  let n : ℕ := 10
  let k : ℕ := 3
  let p : ℚ := 1/2
  binomial_probability n k p = 15/128 := by
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l1551_155148


namespace NUMINAMATH_CALUDE_order_of_trig_expressions_l1551_155155

theorem order_of_trig_expressions :
  Real.arcsin (3/4) < Real.arccos (1/5) ∧ Real.arccos (1/5) < 1 + Real.arctan (2/3) := by
  sorry

end NUMINAMATH_CALUDE_order_of_trig_expressions_l1551_155155


namespace NUMINAMATH_CALUDE_quadratic_polynomial_equality_l1551_155193

theorem quadratic_polynomial_equality 
  (f : ℝ → ℝ) 
  (h_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c) 
  (h_equality : ∀ x, f (2 * x + 1) = 4 * x^2 + 14 * x + 7) :
  ∃ (a b c : ℝ), (a = 1 ∧ b = 5 ∧ c = 1) ∧ (∀ x, f x = x^2 + 5 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_equality_l1551_155193


namespace NUMINAMATH_CALUDE_fraction_of_fraction_tripled_l1551_155107

theorem fraction_of_fraction_tripled (a b c d : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 3 ∧ d = 8 → 
  3 * ((c / d) / (a / b)) = 27 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_tripled_l1551_155107


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1551_155154

/-- The repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 0.363636

theorem repeating_decimal_as_fraction :
  repeating_decimal = 4 / 11 ∧ 
  (4 : ℕ) + 11 = 15 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1551_155154


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1551_155149

theorem equal_roots_quadratic (c : ℝ) :
  (∃ x : ℝ, x^2 + 6*x + c = 0 ∧ (∀ y : ℝ, y^2 + 6*y + c = 0 → y = x)) →
  c = 9 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1551_155149
