import Mathlib

namespace NUMINAMATH_CALUDE_tangent_circle_radius_l4030_403071

/-- A circle tangent to coordinate axes and hypotenuse of a 45-45-90 triangle --/
structure TangentCircle where
  O : ℝ × ℝ  -- Center of the circle
  r : ℝ      -- Radius of the circle
  h : ℝ      -- hypotenuse length of the 45-45-90 triangle

/-- The circle is tangent to both axes and the hypotenuse --/
def is_tangent (c : TangentCircle) : Prop :=
  c.O.1 = c.r ∧ c.O.2 = c.r ∧ c.O.1 + c.O.2 + c.r = c.h

theorem tangent_circle_radius (c : TangentCircle) 
  (h_hypotenuse : c.h = 2 * Real.sqrt 2)
  (h_tangent : is_tangent c) : 
  c.r = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l4030_403071


namespace NUMINAMATH_CALUDE_inscribed_square_area_equals_rectangle_area_l4030_403011

/-- Given a right triangle with legs a and b, and a square with side length x
    inscribed such that one angle coincides with the right angle of the triangle
    and one vertex lies on the hypotenuse, the area of the square is equal to
    the area of the rectangle formed by the remaining segments of the legs. -/
theorem inscribed_square_area_equals_rectangle_area 
  (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < min a b) : 
  x^2 = (a - x) * (b - x) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_equals_rectangle_area_l4030_403011


namespace NUMINAMATH_CALUDE_bicycle_spokes_front_wheel_l4030_403013

/-- Proves that a bicycle with 60 total spokes and twice as many spokes on the back wheel as on the front wheel has 20 spokes on the front wheel. -/
theorem bicycle_spokes_front_wheel : 
  ∀ (front back : ℕ), 
  front + back = 60 → 
  back = 2 * front → 
  front = 20 := by
sorry

end NUMINAMATH_CALUDE_bicycle_spokes_front_wheel_l4030_403013


namespace NUMINAMATH_CALUDE_common_tangent_range_a_l4030_403000

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

def has_common_tangent (f g : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (deriv f x₁ = deriv g x₂) ∧ 
    (f x₁ - g x₂ = deriv f x₁ * (x₁ - x₂))

theorem common_tangent_range_a :
  ∀ a : ℝ, (∃ x < 0, has_common_tangent f (g a)) → 
    a ∈ Set.Ioi (Real.log (1/(2*Real.exp 1))) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_range_a_l4030_403000


namespace NUMINAMATH_CALUDE_game_installation_time_ratio_l4030_403023

theorem game_installation_time_ratio :
  ∀ (install_time : ℝ),
    install_time > 0 →
    10 + install_time + 3 * (10 + install_time) = 60 →
    install_time / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_game_installation_time_ratio_l4030_403023


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l4030_403055

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 39) 
  (h_sum : a 1 + a 3 = 74) : 
  ∀ n : ℕ, a n = -2 * n + 41 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l4030_403055


namespace NUMINAMATH_CALUDE_eliza_dress_ironing_time_l4030_403097

/-- Represents the time in minutes it takes Eliza to iron a dress -/
def dress_ironing_time : ℕ := sorry

/-- Represents the time in minutes it takes Eliza to iron a blouse -/
def blouse_ironing_time : ℕ := 15

/-- Represents the total time in minutes Eliza spends ironing blouses -/
def total_blouse_ironing_time : ℕ := 2 * 60

/-- Represents the total time in minutes Eliza spends ironing dresses -/
def total_dress_ironing_time : ℕ := 3 * 60

/-- Represents the total number of clothes Eliza ironed -/
def total_clothes : ℕ := 17

theorem eliza_dress_ironing_time :
  (total_blouse_ironing_time / blouse_ironing_time) +
  (total_dress_ironing_time / dress_ironing_time) = total_clothes →
  dress_ironing_time = 20 := by sorry

end NUMINAMATH_CALUDE_eliza_dress_ironing_time_l4030_403097


namespace NUMINAMATH_CALUDE_power_of_product_l4030_403044

theorem power_of_product (x : ℝ) : (2 * x)^3 = 8 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l4030_403044


namespace NUMINAMATH_CALUDE_tan_alpha_value_l4030_403095

theorem tan_alpha_value (α : ℝ) (h : Real.tan (α - π/4) = 1/5) : 
  Real.tan α = 3/2 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l4030_403095


namespace NUMINAMATH_CALUDE_sum_remainder_l4030_403057

theorem sum_remainder (m : ℤ) : (10 - 3*m + (5*m + 6)) % 8 = (2*m) % 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l4030_403057


namespace NUMINAMATH_CALUDE_lattice_point_theorem_l4030_403099

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The set of all lattice points -/
def L : Set LatticePoint := Set.univ

/-- Check if a line segment between two lattice points contains no other lattice points -/
def noInteriorLatticePoints (a b : LatticePoint) : Prop :=
  ∀ p : LatticePoint, p ∈ L → p ≠ a → p ≠ b → ¬(∃ t : ℚ, 0 < t ∧ t < 1 ∧ 
    p.x = a.x + t * (b.x - a.x) ∧ p.y = a.y + t * (b.y - a.y))

theorem lattice_point_theorem :
  (∀ a b c : LatticePoint, a ∈ L → b ∈ L → c ∈ L → a ≠ b → b ≠ c → a ≠ c →
    ∃ d : LatticePoint, d ∈ L ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
      noInteriorLatticePoints a d ∧ noInteriorLatticePoints b d ∧ noInteriorLatticePoints c d) ∧
  (∃ a b c d : LatticePoint, a ∈ L ∧ b ∈ L ∧ c ∈ L ∧ d ∈ L ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    (¬noInteriorLatticePoints a b ∨ ¬noInteriorLatticePoints b c ∨
     ¬noInteriorLatticePoints c d ∨ ¬noInteriorLatticePoints d a)) :=
by sorry

end NUMINAMATH_CALUDE_lattice_point_theorem_l4030_403099


namespace NUMINAMATH_CALUDE_noah_total_capacity_l4030_403005

-- Define Ali's closet capacity
def ali_closet_capacity : ℕ := 200

-- Define the ratio of Noah's closet capacity to Ali's
def noah_closet_ratio : ℚ := 1 / 4

-- Define the number of Noah's closets
def noah_closet_count : ℕ := 2

-- Theorem statement
theorem noah_total_capacity :
  noah_closet_count * (noah_closet_ratio * ali_closet_capacity) = 100 := by
  sorry


end NUMINAMATH_CALUDE_noah_total_capacity_l4030_403005


namespace NUMINAMATH_CALUDE_sector_central_angle_l4030_403058

/-- Given a circular sector with circumference 6 and area 2, prove that its central angle in radians is either 1 or 4. -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 6) (h2 : (1 / 2) * l * r = 2) :
  let α := l / r
  α = 1 ∨ α = 4 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l4030_403058


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l4030_403036

theorem complex_sum_theorem : 
  let Z₁ : ℂ := (1 - Complex.I) / (1 + Complex.I)
  let Z₂ : ℂ := (3 - Complex.I) * Complex.I
  Z₁ + Z₂ = 1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l4030_403036


namespace NUMINAMATH_CALUDE_second_month_sale_l4030_403051

/-- Given the sales of a grocer for four months, prove that the sale in the second month is 4000 --/
theorem second_month_sale
  (sale1 : ℕ)
  (sale3 : ℕ)
  (sale4 : ℕ)
  (average : ℕ)
  (h1 : sale1 = 2500)
  (h2 : sale3 = 3540)
  (h3 : sale4 = 1520)
  (h4 : average = 2890)
  (h5 : (sale1 + sale3 + sale4 + (4 * average - sale1 - sale3 - sale4)) / 4 = average) :
  4 * average - sale1 - sale3 - sale4 = 4000 := by
sorry

end NUMINAMATH_CALUDE_second_month_sale_l4030_403051


namespace NUMINAMATH_CALUDE_f_properties_l4030_403007

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a^x - a^(-x)) / (a - 1)

-- Theorem statement
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  StrictMono (f a) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l4030_403007


namespace NUMINAMATH_CALUDE_x_equals_2_valid_l4030_403050

/-- Represents an assignment statement -/
inductive AssignmentStatement
| constant : ℕ → AssignmentStatement
| variable : String → ℕ → AssignmentStatement
| consecutive : String → String → ℕ → AssignmentStatement
| expression : String → String → ℕ → AssignmentStatement

/-- Checks if an assignment statement is valid -/
def isValidAssignment (stmt : AssignmentStatement) : Prop :=
  match stmt with
  | AssignmentStatement.variable _ _ => True
  | _ => False

theorem x_equals_2_valid :
  isValidAssignment (AssignmentStatement.variable "x" 2) = True :=
by sorry

end NUMINAMATH_CALUDE_x_equals_2_valid_l4030_403050


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l4030_403019

theorem fourth_degree_polynomial_roots :
  let p : ℝ → ℝ := λ x => 3*x^4 - 19*x^3 + 34*x^2 - 19*x + 3
  (∀ x : ℝ, p x = 0 ↔ x = 2 + Real.sqrt 3 ∨ 
                      x = 2 - Real.sqrt 3 ∨ 
                      x = (7 + Real.sqrt 13) / 6 ∨ 
                      x = (7 - Real.sqrt 13) / 6) :=
by sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l4030_403019


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l4030_403093

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∀ n : ℕ, n > 0 → n^2 % 2 = 0 → n^2 % 3 = 0 → n^2 % 5 = 0 → n^2 ≥ 900 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l4030_403093


namespace NUMINAMATH_CALUDE_boys_in_class_l4030_403085

theorem boys_in_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) (boys : ℕ) : 
  total = 56 → 
  girls_ratio = 4 →
  boys_ratio = 3 →
  girls_ratio + boys_ratio = 7 →
  (girls_ratio : ℚ) / (boys_ratio : ℚ) = 4 / 3 →
  boys = (boys_ratio : ℚ) / (girls_ratio + boys_ratio : ℚ) * total →
  boys = 24 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l4030_403085


namespace NUMINAMATH_CALUDE_most_likely_outcome_l4030_403077

/-- The probability of a child being a girl -/
def p_girl : ℚ := 3/5

/-- The probability of a child being a boy -/
def p_boy : ℚ := 2/5

/-- The number of children born -/
def n : ℕ := 3

/-- The probability of having 2 girls and 1 boy out of 3 children -/
def p_two_girls_one_boy : ℚ := 54/125

theorem most_likely_outcome :
  p_two_girls_one_boy = Nat.choose n 2 * p_girl^2 * p_boy ∧
  p_two_girls_one_boy > p_boy^n ∧
  p_two_girls_one_boy > p_girl^n ∧
  p_two_girls_one_boy > Nat.choose n 1 * p_girl * p_boy^2 :=
by sorry

end NUMINAMATH_CALUDE_most_likely_outcome_l4030_403077


namespace NUMINAMATH_CALUDE_special_inequality_l4030_403039

/-- The equation x^2 - 4x + |a-3| = 0 has real roots with respect to x -/
def has_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4*x + |a-3| = 0

/-- The inequality t^2 - 2at + 12 < 0 holds for all a in [-1, 7] -/
def inequality_holds (t : ℝ) : Prop :=
  ∀ a : ℝ, -1 ≤ a ∧ a ≤ 7 → t^2 - 2*a*t + 12 < 0

theorem special_inequality (t : ℝ) :
  (∃ a : ℝ, has_real_roots a) →
  inequality_holds t →
  3 < t ∧ t < 4 := by
  sorry

end NUMINAMATH_CALUDE_special_inequality_l4030_403039


namespace NUMINAMATH_CALUDE_hdtv_horizontal_length_l4030_403018

theorem hdtv_horizontal_length :
  ∀ (diagonal : ℝ) (aspect_width aspect_height : ℕ),
    diagonal = 42 →
    aspect_width = 16 →
    aspect_height = 9 →
    ∃ (horizontal : ℝ),
      horizontal = (aspect_width : ℝ) * diagonal / Real.sqrt ((aspect_width ^ 2 : ℝ) + (aspect_height ^ 2 : ℝ)) ∧
      horizontal = 672 / Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_hdtv_horizontal_length_l4030_403018


namespace NUMINAMATH_CALUDE_playground_area_l4030_403081

theorem playground_area (perimeter width length : ℝ) (h1 : perimeter = 80) 
  (h2 : length = 3 * width) (h3 : perimeter = 2 * (length + width)) : 
  length * width = 300 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_l4030_403081


namespace NUMINAMATH_CALUDE_min_students_with_blue_shirt_and_red_shoes_l4030_403066

theorem min_students_with_blue_shirt_and_red_shoes
  (n : ℕ)  -- Total number of students
  (blue_shirt : ℕ)  -- Number of students wearing blue shirts
  (red_shoes : ℕ)  -- Number of students wearing red shoes
  (h1 : blue_shirt = n * 3 / 7)  -- 3/7 of students wear blue shirts
  (h2 : red_shoes = n * 4 / 9)  -- 4/9 of students wear red shoes
  : ∃ (both : ℕ), both ≥ 8 ∧ blue_shirt + red_shoes - both = n :=
sorry

end NUMINAMATH_CALUDE_min_students_with_blue_shirt_and_red_shoes_l4030_403066


namespace NUMINAMATH_CALUDE_absent_workers_fraction_l4030_403006

theorem absent_workers_fraction (p : ℕ) (W : ℝ) (h : p > 0) :
  let work_per_person := W / p
  let absent_fraction : ℝ → ℝ := λ x => x
  let present_workers : ℝ → ℝ := λ x => p * (1 - x)
  let increased_work_per_person := work_per_person * 1.2
  increased_work_per_person = W / (present_workers (absent_fraction (1/6))) :=
by sorry

end NUMINAMATH_CALUDE_absent_workers_fraction_l4030_403006


namespace NUMINAMATH_CALUDE_wood_gathering_proof_l4030_403092

/-- The number of pieces of wood that can be contained in one sack -/
def pieces_per_sack : ℕ := 20

/-- The number of sacks filled -/
def num_sacks : ℕ := 4

/-- The total number of pieces of wood gathered -/
def total_pieces : ℕ := pieces_per_sack * num_sacks

theorem wood_gathering_proof :
  total_pieces = 80 :=
by sorry

end NUMINAMATH_CALUDE_wood_gathering_proof_l4030_403092


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l4030_403082

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 ∧ y < 10 ∧ y = 2 * x ∧ (10 * x + y) - (x + y) = 8 → 
  (10 * y + x) - (10 * x + y) = 9 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l4030_403082


namespace NUMINAMATH_CALUDE_expression_not_prime_l4030_403033

def expression (x y : ℕ) : ℕ :=
  x^8 - x^7*y + x^6*y^2 - x^5*y^3 + x^4*y^4 - x^3*y^5 + x^2*y^6 - x*y^7 + y^8

theorem expression_not_prime (x y : ℕ) :
  ¬(Nat.Prime (expression x y)) :=
sorry

end NUMINAMATH_CALUDE_expression_not_prime_l4030_403033


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l4030_403020

theorem greatest_integer_for_all_real_domain : 
  ∃ (b : ℤ), (∀ (x : ℝ), (x^2 + b*x + 10 ≠ 0)) ∧ 
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + c*x + 10 = 0) ∧ 
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l4030_403020


namespace NUMINAMATH_CALUDE_angle_difference_range_l4030_403012

theorem angle_difference_range (α β : ℝ) 
  (h1 : -π < α ∧ α < β ∧ β < π) : 
  -2*π < α - β ∧ α - β < 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_difference_range_l4030_403012


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l4030_403062

/-- Given two positive integers with LCM 560 and product 42000, their HCF is 75 -/
theorem hcf_from_lcm_and_product (A B : ℕ+) 
  (h_lcm : Nat.lcm A B = 560)
  (h_product : A * B = 42000) :
  Nat.gcd A B = 75 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l4030_403062


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4030_403014

/-- Given that i² = -1, prove that (1 - i) / (2 + 3i) = -1/13 - 5/13 * i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (1 - i) / (2 + 3*i) = -1/13 - 5/13 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4030_403014


namespace NUMINAMATH_CALUDE_pairball_playing_time_l4030_403043

theorem pairball_playing_time (num_children : ℕ) (total_time : ℕ) (pair_size : ℕ) : 
  num_children = 6 →
  pair_size = 2 →
  total_time = 120 →
  (total_time * pair_size) / num_children = 40 := by
sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l4030_403043


namespace NUMINAMATH_CALUDE_range_of_a_l4030_403061

def P (a : ℝ) := {x : ℝ | a - 4 < x ∧ x < a + 4}

def Q := {x : ℝ | x^2 - 4*x + 3 < 0}

theorem range_of_a :
  (∀ x, x ∈ Q → x ∈ P a) ↔ -1 ≤ a ∧ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l4030_403061


namespace NUMINAMATH_CALUDE_solution_set_inequality_l4030_403008

theorem solution_set_inequality (x : ℝ) : (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l4030_403008


namespace NUMINAMATH_CALUDE_system_solution_l4030_403052

theorem system_solution (x y : ℝ) : 
  (x / y + y / x = 173 / 26 ∧ 1 / x + 1 / y = 15 / 26) → 
  ((x = 13 ∧ y = 2) ∨ (x = 2 ∧ y = 13)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l4030_403052


namespace NUMINAMATH_CALUDE_three_times_a_plus_square_of_b_l4030_403049

/-- The algebraic expression "three times a plus the square of b" is equivalent to 3a + b² -/
theorem three_times_a_plus_square_of_b (a b : ℝ) : 3 * a + b^2 = 3 * a + b^2 := by
  sorry

end NUMINAMATH_CALUDE_three_times_a_plus_square_of_b_l4030_403049


namespace NUMINAMATH_CALUDE_complex_absolute_value_l4030_403015

theorem complex_absolute_value : 
  Complex.abs (7/4 - 3*Complex.I + Real.sqrt 3) = 
  (Real.sqrt (241 + 56*Real.sqrt 3))/4 := by
sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l4030_403015


namespace NUMINAMATH_CALUDE_prob_same_suit_is_one_seventeenth_l4030_403087

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents the suit of a card -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- Represents the rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function that returns the suit of a card given its index in the deck -/
def cardSuit (card : Fin 52) : Suit :=
  sorry

/-- The probability of drawing two cards of the same suit from a standard deck -/
def probabilitySameSuit : ℚ :=
  1 / 17

/-- Theorem stating that the probability of drawing two cards of the same suit is 1/17 -/
theorem prob_same_suit_is_one_seventeenth :
  probabilitySameSuit = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_suit_is_one_seventeenth_l4030_403087


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l4030_403009

theorem rectangle_longest_side (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧                 -- positive dimensions
  2 * (l + w) = 180 ∧             -- perimeter is 180 feet
  l * w = 8 * 180 →               -- area is 8 times perimeter
  max l w = 72 := by sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l4030_403009


namespace NUMINAMATH_CALUDE_sum_of_divisors_1184_l4030_403042

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_1184 : sum_of_divisors 1184 = 2394 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_1184_l4030_403042


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l4030_403010

theorem max_value_expression (x y : ℝ) :
  (x + 2*y + 3) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 14 :=
sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (x + 2*y + 3) / Real.sqrt (x^2 + y^2 + 1) = Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l4030_403010


namespace NUMINAMATH_CALUDE_incenter_distance_l4030_403030

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let qr := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  let rp := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  pq = 17 ∧ qr = 15 ∧ rp = 8

-- Define the incenter
def Incenter (P Q R J : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (J.1 - P.1)^2 + (J.2 - P.2)^2 = r^2 ∧
  (J.1 - Q.1)^2 + (J.2 - Q.2)^2 = r^2 ∧
  (J.1 - R.1)^2 + (J.2 - R.2)^2 = r^2

-- Theorem statement
theorem incenter_distance (P Q R J : ℝ × ℝ) :
  Triangle P Q R → Incenter P Q R J →
  (J.1 - P.1)^2 + (J.2 - P.2)^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_incenter_distance_l4030_403030


namespace NUMINAMATH_CALUDE_divisibility_of_fourth_power_sum_l4030_403063

theorem divisibility_of_fourth_power_sum (a b c n : ℤ) 
  (h1 : n ∣ (a + b + c)) 
  (h2 : n ∣ (a^2 + b^2 + c^2)) : 
  n ∣ (a^4 + b^4 + c^4) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_fourth_power_sum_l4030_403063


namespace NUMINAMATH_CALUDE_segment_length_l4030_403022

theorem segment_length : Real.sqrt 157 = Real.sqrt ((8 - 2)^2 + (18 - 7)^2) := by sorry

end NUMINAMATH_CALUDE_segment_length_l4030_403022


namespace NUMINAMATH_CALUDE_length_lost_per_knot_l4030_403096

/-- Given a set of ropes and the total length after tying, calculate the length lost per knot -/
theorem length_lost_per_knot (rope_lengths : List ℝ) (total_length_after_tying : ℝ) : 
  rope_lengths = [8, 20, 2, 2, 2, 7] ∧ 
  total_length_after_tying = 35 → 
  (rope_lengths.sum - total_length_after_tying) / (rope_lengths.length - 1) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_length_lost_per_knot_l4030_403096


namespace NUMINAMATH_CALUDE_contractor_payment_proof_l4030_403037

/-- Calculates the total amount received by a contractor given the contract details and attendance. -/
def contractorPayment (totalDays duration : ℕ) (dailyWage dailyFine : ℚ) (absentDays : ℕ) : ℚ :=
  let workingDays := duration - absentDays
  let earnings := workingDays * dailyWage
  let fines := absentDays * dailyFine
  earnings - fines

/-- Proves that the contractor receives Rs. 490 under the given conditions. -/
theorem contractor_payment_proof :
  contractorPayment 30 30 25 (7.5 : ℚ) 8 = 490 := by
  sorry

#eval contractorPayment 30 30 25 (7.5 : ℚ) 8

end NUMINAMATH_CALUDE_contractor_payment_proof_l4030_403037


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l4030_403025

def f (x : ℝ) := x^3 + 3*x^2 - 9*x + 1

theorem tangent_line_and_extrema :
  ∃ (y : ℝ → ℝ),
    (∀ x, y x = -9*x + 1) ∧
    (∀ x, x ∈ [-1, 2] → f x ≤ 12) ∧
    (∀ x, x ∈ [-1, 2] → f x ≥ -4) ∧
    (∃ x₁ ∈ [-1, 2], f x₁ = 12) ∧
    (∃ x₂ ∈ [-1, 2], f x₂ = -4) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l4030_403025


namespace NUMINAMATH_CALUDE_calculation_proof_l4030_403069

theorem calculation_proof : ((5 + 7 + 3) * 2 - 4) / 2 - 5 / 2 = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4030_403069


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2019_l4030_403045

theorem last_four_digits_of_5_pow_2019 (h5 : 5^5 % 10000 = 3125)
                                       (h6 : 5^6 % 10000 = 5625)
                                       (h7 : 5^7 % 10000 = 8125)
                                       (h8 : 5^8 % 10000 = 0625) :
  5^2019 % 10000 = 8125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2019_l4030_403045


namespace NUMINAMATH_CALUDE_length_of_AB_is_10_l4030_403001

-- Define the triangle structures
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the isosceles property
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define the perimeter
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Theorem statement
theorem length_of_AB_is_10 
  (ABC : Triangle) 
  (CBD : Triangle) 
  (isIsoscelesABC : isIsosceles ABC)
  (isIsoscelesCBD : isIsosceles CBD)
  (angle_BAC_twice_ABC : True)  -- We can't directly represent angle relationships, so we use a placeholder
  (perim_CBD : perimeter CBD = 21)
  (perim_ABC : perimeter ABC = 26)
  (length_BD : CBD.c = 9)
  : ABC.a = 10 := by
  sorry


end NUMINAMATH_CALUDE_length_of_AB_is_10_l4030_403001


namespace NUMINAMATH_CALUDE_sammys_homework_l4030_403016

theorem sammys_homework (total : ℕ) (completed : ℕ) (h1 : total = 9) (h2 : completed = 2) :
  total - completed = 7 := by sorry

end NUMINAMATH_CALUDE_sammys_homework_l4030_403016


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l4030_403088

theorem tan_sum_simplification : 
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l4030_403088


namespace NUMINAMATH_CALUDE_sum_of_evens_1_to_101_l4030_403091

theorem sum_of_evens_1_to_101 : 
  (Finset.range 51).sum (fun i => 2 * i) = 2550 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_evens_1_to_101_l4030_403091


namespace NUMINAMATH_CALUDE_tan_alpha_value_l4030_403017

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 2) :
  Real.tan α = -8/3 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l4030_403017


namespace NUMINAMATH_CALUDE_halfway_fraction_l4030_403072

theorem halfway_fraction : (3 / 4 + 5 / 6) / 2 = 19 / 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l4030_403072


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_l4030_403079

theorem greatest_integer_fraction (x : ℤ) : 
  (∃ k : ℤ, (x^2 + 4*x + 9) / (x - 4) = k) → x ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_l4030_403079


namespace NUMINAMATH_CALUDE_tan_and_trig_identity_l4030_403078

open Real

theorem tan_and_trig_identity (α : ℝ) (h : tan (α + π/4) = 1/3) : 
  tan α = -1/2 ∧ 
  2 * sin α ^ 2 - sin (π - α) * sin (π/2 - α) + sin (3*π/2 + α) ^ 2 = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_and_trig_identity_l4030_403078


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l4030_403056

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l4030_403056


namespace NUMINAMATH_CALUDE_abs_sum_zero_implies_diff_l4030_403059

theorem abs_sum_zero_implies_diff (a b : ℝ) : 
  |a - 2| + |b + 3| = 0 → a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_zero_implies_diff_l4030_403059


namespace NUMINAMATH_CALUDE_x_is_irrational_l4030_403067

/-- Representation of the digits of 1987^k -/
def digits (k : ℕ) : List ℕ :=
  sorry

/-- Construct the number x as described in the problem -/
def x : ℝ :=
  sorry

/-- Theorem stating that x is irrational -/
theorem x_is_irrational : Irrational x := by
  sorry

end NUMINAMATH_CALUDE_x_is_irrational_l4030_403067


namespace NUMINAMATH_CALUDE_acute_triangle_count_l4030_403098

/-- Count of integers satisfying acute triangle conditions --/
theorem acute_triangle_count : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ 
    18 + 36 > x ∧ 
    18 + x > 36 ∧ 
    36 + x > 18 ∧ 
    (x > 36 → x^2 < 18^2 + 36^2) ∧ 
    (x ≤ 36 → 36^2 < 18^2 + x^2))
    (Finset.range 55)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_count_l4030_403098


namespace NUMINAMATH_CALUDE_amanda_loan_l4030_403076

/-- Calculates the earnings for a given number of hours based on a cyclic payment structure -/
def calculateEarnings (hours : ℕ) : ℕ :=
  let fullCycles := hours / 4
  let remainingHours := hours % 4
  let earningsPerCycle := 10
  let earningsFromFullCycles := fullCycles * earningsPerCycle
  let earningsFromRemainingHours := 
    if remainingHours = 1 then 2
    else if remainingHours = 2 then 5
    else if remainingHours = 3 then 7
    else 0
  earningsFromFullCycles + earningsFromRemainingHours

theorem amanda_loan (x : ℕ) : 
  (x = calculateEarnings 50) → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_amanda_loan_l4030_403076


namespace NUMINAMATH_CALUDE_chocolates_not_in_box_l4030_403065

theorem chocolates_not_in_box (initial_chocolates : ℕ) (initial_boxes : ℕ) 
  (additional_chocolates : ℕ) (additional_boxes : ℕ) :
  initial_chocolates = 50 →
  initial_boxes = 3 →
  additional_chocolates = 25 →
  additional_boxes = 2 →
  ∃ (chocolates_per_box : ℕ),
    chocolates_per_box * (initial_boxes + additional_boxes) = initial_chocolates + additional_chocolates →
    initial_chocolates - (chocolates_per_box * initial_boxes) = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_not_in_box_l4030_403065


namespace NUMINAMATH_CALUDE_max_value_of_s_l4030_403075

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 8)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 12) :
  s ≤ 2 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l4030_403075


namespace NUMINAMATH_CALUDE_exists_21_win_stretch_l4030_403038

/-- Represents the cumulative wins of a chess player over 77 days -/
def CumulativeWins := Fin 78 → ℕ

/-- The conditions for the chess player's winning record -/
def ValidWinningRecord (x : CumulativeWins) : Prop :=
  (∀ i : Fin 77, x (i + 1) > x i) ∧ 
  (∀ i : Fin 71, x (i + 7) - x i ≤ 12) ∧
  x 0 = 0 ∧ x 77 ≤ 132

/-- The theorem stating that there exists a stretch of consecutive days with exactly 21 wins -/
theorem exists_21_win_stretch (x : CumulativeWins) (h : ValidWinningRecord x) : 
  ∃ i j : Fin 78, i < j ∧ x j - x i = 21 := by
  sorry


end NUMINAMATH_CALUDE_exists_21_win_stretch_l4030_403038


namespace NUMINAMATH_CALUDE_quadratic_root_form_l4030_403094

/-- The quadratic equation 2x^2 - 5x - 4 = 0 -/
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 5 * x - 4 = 0

/-- The roots of the equation in the form (m ± √n) / p -/
def root_form (m n p : ℕ) (x : ℝ) : Prop :=
  ∃ (sign : Bool), x = (m + if sign then 1 else -1 * Real.sqrt n) / p

/-- m, n, and p are coprime -/
def coprime (m n p : ℕ) : Prop := Nat.gcd m (Nat.gcd n p) = 1

theorem quadratic_root_form :
  ∃ (m n p : ℕ), 
    (∀ x : ℝ, quadratic_equation x → root_form m n p x) ∧
    coprime m n p ∧
    n = 57 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l4030_403094


namespace NUMINAMATH_CALUDE_inequality_solution_l4030_403004

theorem inequality_solution (a : ℝ) :
  (a > 0 → (∀ x : ℝ, 6 * x^2 + a * x - a^2 < 0 ↔ -a/2 < x ∧ x < a/3)) ∧
  (a = 0 → ¬ ∃ x : ℝ, 6 * x^2 + a * x - a^2 < 0) ∧
  (a < 0 → (∀ x : ℝ, 6 * x^2 + a * x - a^2 < 0 ↔ a/3 < x ∧ x < -a/2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4030_403004


namespace NUMINAMATH_CALUDE_distance_school_to_david_value_total_distance_sum_l4030_403047

/-- The distance Craig walked from school to David's house -/
def distance_school_to_david : ℝ := sorry

/-- The distance Craig walked from David's house to his own house -/
def distance_david_to_craig : ℝ := 0.7

/-- The total distance Craig walked -/
def total_distance : ℝ := 0.9

/-- Theorem stating that the distance from school to David's house is 0.2 miles -/
theorem distance_school_to_david_value :
  distance_school_to_david = 0.2 :=
by
  sorry

/-- Theorem stating that the total distance is the sum of the two parts -/
theorem total_distance_sum :
  total_distance = distance_school_to_david + distance_david_to_craig :=
by
  sorry

end NUMINAMATH_CALUDE_distance_school_to_david_value_total_distance_sum_l4030_403047


namespace NUMINAMATH_CALUDE_triangle_perimeter_increase_l4030_403002

/-- The growth factor between consecutive triangles -/
def growthFactor : ℝ := 1.2

/-- The number of triangles -/
def numTriangles : ℕ := 5

/-- Calculates the percent increase between the first and last triangle -/
def percentIncrease : ℝ := (growthFactor ^ (numTriangles - 1) - 1) * 100

theorem triangle_perimeter_increase :
  ∃ ε > 0, ε < 0.1 ∧ |percentIncrease - 107.4| < ε :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_increase_l4030_403002


namespace NUMINAMATH_CALUDE_pass_through_walls_technique_l4030_403084

theorem pass_through_walls_technique (n : ℕ) :
  10 * Real.sqrt (10 / n) = Real.sqrt (10 * (10 / n)) ↔ n = 99 :=
sorry

end NUMINAMATH_CALUDE_pass_through_walls_technique_l4030_403084


namespace NUMINAMATH_CALUDE_function_identity_l4030_403046

-- Define the property that the function f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

-- State the theorem
theorem function_identity {f : ℝ → ℝ} (h : SatisfiesProperty f) : 
  ∀ x : ℝ, f x = x := by sorry

end NUMINAMATH_CALUDE_function_identity_l4030_403046


namespace NUMINAMATH_CALUDE_transaction_result_l4030_403026

theorem transaction_result (car_sale_price motorcycle_sale_price : ℝ)
  (car_loss_percent motorcycle_gain_percent : ℝ)
  (h1 : car_sale_price = 18000)
  (h2 : motorcycle_sale_price = 10000)
  (h3 : car_loss_percent = 10)
  (h4 : motorcycle_gain_percent = 25) :
  car_sale_price + motorcycle_sale_price =
  (car_sale_price / (100 - car_loss_percent) * 100) +
  (motorcycle_sale_price / (100 + motorcycle_gain_percent) * 100) :=
by sorry

end NUMINAMATH_CALUDE_transaction_result_l4030_403026


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l4030_403086

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 2 ∨ x = 4 ∨ x = 5) :
  c / d = 19 / 20 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l4030_403086


namespace NUMINAMATH_CALUDE_sufficient_material_l4030_403053

-- Define the surface area of a rectangular box
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

-- Define the volume of a rectangular box
def volume (l w h : ℝ) : ℝ := l * w * h

-- Theorem statement
theorem sufficient_material :
  ∃ (l w h : ℝ), l > 0 ∧ w > 0 ∧ h > 0 ∧ 
  surface_area l w h = 958 ∧ 
  volume l w h ≥ 1995 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_material_l4030_403053


namespace NUMINAMATH_CALUDE_apollonius_circle_l4030_403041

/-- The locus of points with a fixed distance ratio from two given points is a circle -/
theorem apollonius_circle (A B : ℝ × ℝ) (m n : ℝ) (h_pos : m > 0 ∧ n > 0) :
  ∃ (C : ℝ × ℝ) (r : ℝ), ∀ M : ℝ × ℝ,
    (dist M A) / (dist M B) = m / n ↔ dist M C = r :=
  sorry

end NUMINAMATH_CALUDE_apollonius_circle_l4030_403041


namespace NUMINAMATH_CALUDE_positive_integer_divisibility_l4030_403040

theorem positive_integer_divisibility (n : ℕ) :
  (n + 2009 ∣ n^2 + 2009) ∧ (n + 2010 ∣ n^2 + 2010) → n = 0 ∨ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_divisibility_l4030_403040


namespace NUMINAMATH_CALUDE_expression_evaluation_l4030_403073

theorem expression_evaluation :
  (3^2 - 3*2) - (4^2 - 4*2) + (5^2 - 5*2) - (6^2 - 6*2) = -14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4030_403073


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4030_403031

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 4 + a 6 = 3 →               -- given condition
  a 4^2 + 2*a 4*a 6 + a 5*a 7 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4030_403031


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l4030_403021

theorem profit_percentage_previous_year
  (revenue_prev : ℝ)
  (revenue_2009 : ℝ)
  (profit_prev : ℝ)
  (profit_2009 : ℝ)
  (h1 : revenue_2009 = 0.8 * revenue_prev)
  (h2 : profit_2009 = 0.11 * revenue_2009)
  (h3 : profit_2009 = 0.8800000000000001 * profit_prev) :
  profit_prev = 0.1 * revenue_prev :=
sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l4030_403021


namespace NUMINAMATH_CALUDE_no_savings_l4030_403083

-- Define the prices and fees
def in_store_price : ℚ := 129.99
def online_payment : ℚ := 29.99
def shipping_fee : ℚ := 11.99

-- Define the number of online payments
def num_payments : ℕ := 4

-- Define the function to calculate savings in cents
def savings_in_cents : ℚ :=
  (in_store_price - (num_payments * online_payment + shipping_fee)) * 100

-- Theorem statement
theorem no_savings : savings_in_cents = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_savings_l4030_403083


namespace NUMINAMATH_CALUDE_cube_split_l4030_403029

theorem cube_split (m : ℕ) (h1 : m > 1) : 
  (∃ k : ℕ, k ≥ 0 ∧ k < m ∧ m^2 - m + 1 + 2*k = 73) → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_l4030_403029


namespace NUMINAMATH_CALUDE_tens_digit_of_nine_power_2023_l4030_403090

theorem tens_digit_of_nine_power_2023 : 9^2023 % 100 = 29 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_nine_power_2023_l4030_403090


namespace NUMINAMATH_CALUDE_max_value_constraint_l4030_403060

theorem max_value_constraint (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 + Real.sqrt 2 / 3 ∧ 
    ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → x^2 + y^2 + z^2 = 1 → 
      x * y * Real.sqrt 3 + y * z * Real.sqrt 3 + z * x * Real.sqrt 2 ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l4030_403060


namespace NUMINAMATH_CALUDE_square_with_tens_digit_7_l4030_403024

/-- A square number with tens digit 7 has units digit 6 -/
theorem square_with_tens_digit_7 (n : ℕ) :
  (n^2 / 10) % 10 = 7 → n^2 % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_with_tens_digit_7_l4030_403024


namespace NUMINAMATH_CALUDE_tara_road_trip_cost_l4030_403003

/-- Represents a gas station with a price per gallon -/
structure GasStation :=
  (price : ℚ)

/-- Calculates the total cost of gas for a road trip -/
def total_gas_cost (tank_capacity : ℚ) (stations : List GasStation) : ℚ :=
  stations.map (λ station => station.price * tank_capacity) |>.sum

/-- Theorem: The total cost of gas for Tara's road trip is $180 -/
theorem tara_road_trip_cost :
  let tank_capacity : ℚ := 12
  let stations : List GasStation := [
    { price := 3 },
    { price := 7/2 },
    { price := 4 },
    { price := 9/2 }
  ]
  total_gas_cost tank_capacity stations = 180 := by
  sorry

end NUMINAMATH_CALUDE_tara_road_trip_cost_l4030_403003


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_integers_from_neg3_to_6_l4030_403080

def integers_range : List ℤ := List.range 10 |>.map (λ i => i - 3)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  (integers_range.sum : ℚ) / integers_range.length = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_integers_from_neg3_to_6_l4030_403080


namespace NUMINAMATH_CALUDE_ivanov_petrov_probability_l4030_403068

/-- The number of people in the group -/
def n : ℕ := 11

/-- The number of people that should be between Ivanov and Petrov -/
def k : ℕ := 3

/-- The probability of exactly k people sitting between two specific people
    in a random circular arrangement of n people -/
def probability (n k : ℕ) : ℚ :=
  if n > k + 1 then 1 / (n - 1) else 0

theorem ivanov_petrov_probability :
  probability n k = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_ivanov_petrov_probability_l4030_403068


namespace NUMINAMATH_CALUDE_sequence_sum_l4030_403027

theorem sequence_sum (n : ℕ) (s : ℕ → ℕ) : n = 2010 →
  (∀ i, i ∈ Finset.range (n - 1) → s (i + 1) = s i + 1) →
  (Finset.sum (Finset.range n) s = 5307) →
  (Finset.sum (Finset.range 1005) (fun i => s (2 * i))) = 2151 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l4030_403027


namespace NUMINAMATH_CALUDE_carlotta_time_theorem_l4030_403032

theorem carlotta_time_theorem (n : ℝ) :
  let s : ℝ := 6
  let p : ℝ := 2 * n * s
  let t : ℝ := 3 * n * s + s
  let C : ℝ := p + t + s
  C = 30 * n + 12 := by sorry

end NUMINAMATH_CALUDE_carlotta_time_theorem_l4030_403032


namespace NUMINAMATH_CALUDE_max_x_minus_y_l4030_403028

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l4030_403028


namespace NUMINAMATH_CALUDE_journey_time_proof_l4030_403089

/-- Proves that a journey of 336 km, with the first half traveled at 21 km/hr
    and the second half at 24 km/hr, takes 15 hours to complete. -/
theorem journey_time_proof (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 336 ∧ speed1 = 21 ∧ speed2 = 24 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_proof_l4030_403089


namespace NUMINAMATH_CALUDE_inequality_proof_l4030_403035

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 
  (a + b + c)^2 / (a * b * (a + b) + b * c * (b + c) + c * a * (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4030_403035


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l4030_403034

theorem sum_of_reciprocal_roots (a b : ℝ) : 
  a^2 - 6*a + 4 = 0 → b^2 - 6*b + 4 = 0 → a ≠ b → (1/a + 1/b) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l4030_403034


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4030_403064

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  x^4 + 4*x^2 + 20*x + 1 = (x^2 - 2*x + 7) * q + r ∧
  r.degree < (x^2 - 2*x + 7).degree ∧
  r = 8*x - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4030_403064


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l4030_403070

-- Define the conditions p and q
def p (x : ℝ) : Prop := (1 - x) * (x + 3) < 0
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

-- Define not_p
def not_p (x : ℝ) : Prop := ¬(p x)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_q :
  (∀ x, not_p x → q x) ∧ 
  ¬(∀ x, q x → not_p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l4030_403070


namespace NUMINAMATH_CALUDE_quadratic_root_value_l4030_403054

theorem quadratic_root_value (m : ℝ) : 
  m^2 + m - 1 = 0 → 2*m^2 + 2*m + 2025 = 2027 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l4030_403054


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l4030_403048

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- The theorem stating that if three of the areas are 24, 15, and 9, then the fourth is 15 -/
theorem fourth_rectangle_area (rect : DividedRectangle) 
  (h1 : rect.area1 = 24)
  (h2 : rect.area2 = 15)
  (h3 : rect.area3 = 9) :
  rect.area4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l4030_403048


namespace NUMINAMATH_CALUDE_f_value_at_2_l4030_403074

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_neg (x : ℝ) : ℝ := -x^2 + x

theorem f_value_at_2 (f : ℝ → ℝ) 
    (h_odd : is_odd_function f)
    (h_neg : ∀ x < 0, f x = f_neg x) : 
  f 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_2_l4030_403074
