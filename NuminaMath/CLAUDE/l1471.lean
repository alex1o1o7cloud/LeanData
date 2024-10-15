import Mathlib

namespace NUMINAMATH_CALUDE_largest_triangular_cross_section_area_l1471_147197

/-- The largest possible area of a triangular cross-section in a right circular cone -/
theorem largest_triangular_cross_section_area
  (slant_height : ℝ)
  (base_diameter : ℝ)
  (h_slant : slant_height = 5)
  (h_diameter : base_diameter = 8) :
  ∃ (area : ℝ), area = 12.5 ∧
  ∀ (other_area : ℝ),
    (∃ (a b c : ℝ),
      a ≤ slant_height ∧
      b ≤ slant_height ∧
      c ≤ base_diameter ∧
      other_area = (a * b) / 2) →
    other_area ≤ area :=
by sorry

end NUMINAMATH_CALUDE_largest_triangular_cross_section_area_l1471_147197


namespace NUMINAMATH_CALUDE_conservation_center_turtles_l1471_147119

/-- The number of green turtles -/
def green_turtles : ℕ := 800

/-- The number of hawksbill turtles -/
def hawksbill_turtles : ℕ := 2 * green_turtles + green_turtles

/-- The total number of turtles in the conservation center -/
def total_turtles : ℕ := green_turtles + hawksbill_turtles

theorem conservation_center_turtles : total_turtles = 3200 := by
  sorry

end NUMINAMATH_CALUDE_conservation_center_turtles_l1471_147119


namespace NUMINAMATH_CALUDE_michaels_investment_l1471_147117

theorem michaels_investment (total_investment : ℝ) (thrifty_rate : ℝ) (rich_rate : ℝ) 
  (years : ℕ) (final_amount : ℝ) (thrifty_investment : ℝ) :
  total_investment = 1500 →
  thrifty_rate = 0.04 →
  rich_rate = 0.06 →
  years = 3 →
  final_amount = 1738.84 →
  thrifty_investment * (1 + thrifty_rate) ^ years + 
    (total_investment - thrifty_investment) * (1 + rich_rate) ^ years = final_amount →
  thrifty_investment = 720.84 := by
sorry

end NUMINAMATH_CALUDE_michaels_investment_l1471_147117


namespace NUMINAMATH_CALUDE_eight_prof_sequences_l1471_147157

/-- The number of professors --/
def n : ℕ := 8

/-- The number of distinct sequences for scheduling n professors,
    where one specific professor must present before another specific professor --/
def num_sequences (n : ℕ) : ℕ := n.factorial / 2

/-- Theorem stating that the number of distinct sequences for scheduling 8 professors,
    where one specific professor must present before another specific professor,
    is equal to 8! / 2 --/
theorem eight_prof_sequences :
  num_sequences n = 20160 := by sorry

end NUMINAMATH_CALUDE_eight_prof_sequences_l1471_147157


namespace NUMINAMATH_CALUDE_side_ratio_l1471_147133

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def special_triangle (t : Triangle) : Prop :=
  t.A > t.B ∧ t.B > t.C ∧  -- A is largest, C is smallest
  t.A = 2 * t.C ∧          -- A = 2C
  t.a + t.c = 2 * t.b      -- a + c = 2b

-- Theorem statement
theorem side_ratio (t : Triangle) (h : special_triangle t) :
  ∃ (k : ℝ), k > 0 ∧ t.a = 6*k ∧ t.b = 5*k ∧ t.c = 3*k :=
sorry

end NUMINAMATH_CALUDE_side_ratio_l1471_147133


namespace NUMINAMATH_CALUDE_fraction_zeros_count_l1471_147172

/-- The number of zeros immediately following the decimal point in 1/((6 * 10)^10) -/
def zeros_after_decimal : ℕ := 17

/-- The fraction we're analyzing -/
def fraction : ℚ := 1 / ((6 * 10)^10)

/-- Theorem stating that the number of zeros after the decimal point in the 
    decimal representation of the fraction is equal to zeros_after_decimal -/
theorem fraction_zeros_count : 
  (∃ (n : ℕ) (r : ℚ), fraction * 10^zeros_after_decimal = n + r ∧ 0 < r ∧ r < 1) ∧ 
  (∀ (m : ℕ), m > zeros_after_decimal → ∃ (n : ℕ) (r : ℚ), fraction * 10^m = n + r ∧ r = 0) :=
sorry

end NUMINAMATH_CALUDE_fraction_zeros_count_l1471_147172


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l1471_147188

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ a.val + b.val = 64 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 15 → c.val + d.val ≥ 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l1471_147188


namespace NUMINAMATH_CALUDE_digit_150_of_17_70_l1471_147199

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The repeating sequence in the decimal representation of a rational number -/
def repeating_sequence (q : ℚ) : List ℕ := sorry

theorem digit_150_of_17_70 : 
  decimal_representation (17 / 70) 150 = 7 := by sorry

end NUMINAMATH_CALUDE_digit_150_of_17_70_l1471_147199


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1471_147115

-- Define the function f
def f (x : ℝ) : ℝ := |x| - x + 1

-- State the theorem
theorem inequality_solution_set (x : ℝ) : 
  f (1 - x^2) > f (1 - 2*x) ↔ x > 2 ∨ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1471_147115


namespace NUMINAMATH_CALUDE_linear_function_point_sum_l1471_147138

/-- If the point A(m, n) lies on the line y = -2x + 1, then 4m + 2n + 2022 = 2024 -/
theorem linear_function_point_sum (m n : ℝ) : n = -2 * m + 1 → 4 * m + 2 * n + 2022 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_point_sum_l1471_147138


namespace NUMINAMATH_CALUDE_interval_eq_set_representation_l1471_147137

-- Define the interval (-3, 2]
def interval : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 2}

-- Define the set representation
def set_representation : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 2}

-- Theorem stating that the interval and set representation are equal
theorem interval_eq_set_representation : interval = set_representation := by
  sorry

end NUMINAMATH_CALUDE_interval_eq_set_representation_l1471_147137


namespace NUMINAMATH_CALUDE_product_of_positive_real_solutions_l1471_147135

theorem product_of_positive_real_solutions (x : ℂ) : 
  (x^6 = -64) → 
  (∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^6 = -64 ∧ z.re > 0) ∧ 
    (∀ z, z^6 = -64 ∧ z.re > 0 → z ∈ S) ∧
    (S.prod id = 4)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_positive_real_solutions_l1471_147135


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1471_147163

theorem max_value_of_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) :
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 4 ∧ (2*a - b)^2 + (2*b - c)^2 + (2*c - a)^2 > (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2) →
  (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 ≤ 28 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1471_147163


namespace NUMINAMATH_CALUDE_investment_ratio_l1471_147111

/-- Given two investors P and Q, with P investing 50000 and profits divided in ratio 3:4, 
    prove that Q's investment is 66666.67 -/
theorem investment_ratio (p q : ℝ) (h1 : p = 50000) (h2 : p / q = 3 / 4) : 
  q = 66666.67 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l1471_147111


namespace NUMINAMATH_CALUDE_unchanged_flipped_nine_digit_numbers_l1471_147143

/-- 
Given that:
- A 9-digit number is considered unchanged when flipped if it reads the same upside down.
- Digits 0, 1, and 8 remain unchanged when flipped.
- Digits 6 and 9 become each other when flipped.
- Other digits have no meaning when flipped.

This theorem states that the number of 9-digit numbers that remain unchanged when flipped is 1500.
-/
theorem unchanged_flipped_nine_digit_numbers : ℕ := by
  -- Define the set of digits that remain unchanged when flipped
  let unchanged_digits : Finset ℕ := {0, 1, 8}
  
  -- Define the set of digit pairs that become each other when flipped
  let swapped_digits : Finset (ℕ × ℕ) := {(6, 9), (9, 6)}
  
  -- Define the number of valid options for the first and last digit
  let first_last_options : ℕ := 4
  
  -- Define the number of valid options for the second, third, fourth, and eighth digit
  let middle_pair_options : ℕ := 5
  
  -- Define the number of valid options for the center digit
  let center_options : ℕ := 3
  
  -- Calculate the total number of valid 9-digit numbers
  let total : ℕ := first_last_options * middle_pair_options^3 * center_options
  
  -- Assert that the total is equal to 1500
  have h : total = 1500 := by sorry
  
  -- Return the result
  exact 1500


end NUMINAMATH_CALUDE_unchanged_flipped_nine_digit_numbers_l1471_147143


namespace NUMINAMATH_CALUDE_money_difference_l1471_147148

/-- Given Eliza has 7q + 3 quarters and Tom has 2q + 8 quarters, where every 5 quarters
    over the count of the other person are converted into nickels, the difference in
    their money is 5(q - 1) cents. -/
theorem money_difference (q : ℤ) : 
  let eliza_quarters := 7 * q + 3
  let tom_quarters := 2 * q + 8
  let quarter_difference := eliza_quarters - tom_quarters
  let nickel_groups := quarter_difference / 5
  nickel_groups * 5 = 5 * (q - 1) := by sorry

end NUMINAMATH_CALUDE_money_difference_l1471_147148


namespace NUMINAMATH_CALUDE_system_ratio_value_l1471_147130

/-- Given a system of linear equations with a nontrivial solution,
    prove that the ratio xy/z^2 has a specific value. -/
theorem system_ratio_value (x y z k : ℝ) : 
  x ≠ 0 →
  y ≠ 0 →
  z ≠ 0 →
  x + k*y + 4*z = 0 →
  3*x + k*y - 3*z = 0 →
  2*x + 5*y - 3*z = 0 →
  -- The condition for nontrivial solution is implicitly included in the equations
  ∃ (c : ℝ), x*y / (z^2) = c :=
by
  sorry


end NUMINAMATH_CALUDE_system_ratio_value_l1471_147130


namespace NUMINAMATH_CALUDE_expected_practice_problems_l1471_147177

/-- Represents the number of pairs of shoes -/
def num_pairs : ℕ := 5

/-- Represents the number of days -/
def num_days : ℕ := 5

/-- Represents the probability of selecting two shoes of the same color on a given day -/
def prob_same_color : ℚ := 1 / 9

/-- Represents the expected number of practice problems done in one day -/
def expected_problems_per_day : ℚ := prob_same_color

/-- Theorem stating the expected value of practice problems over 5 days -/
theorem expected_practice_problems :
  (num_days : ℚ) * expected_problems_per_day = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expected_practice_problems_l1471_147177


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_three_l1471_147124

theorem least_positive_integer_to_multiple_of_three : 
  ∃ (n : ℕ), n > 0 ∧ (575 + n) % 3 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (575 + m) % 3 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_three_l1471_147124


namespace NUMINAMATH_CALUDE_max_triangles_for_three_families_of_ten_l1471_147153

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : Nat)

/-- Represents the configuration of three families of parallel lines -/
structure LineConfiguration :=
  (family1 : LineFamily)
  (family2 : LineFamily)
  (family3 : LineFamily)

/-- Calculates the maximum number of triangles formed by the given line configuration -/
def maxTriangles (config : LineConfiguration) : Nat :=
  sorry

/-- Theorem stating the maximum number of triangles formed by three families of 10 parallel lines each -/
theorem max_triangles_for_three_families_of_ten :
  ∃ (config : LineConfiguration),
    config.family1.count = 10 ∧
    config.family2.count = 10 ∧
    config.family3.count = 10 ∧
    maxTriangles config = 150 :=
  sorry

end NUMINAMATH_CALUDE_max_triangles_for_three_families_of_ten_l1471_147153


namespace NUMINAMATH_CALUDE_first_five_pages_drawings_l1471_147144

def drawings_on_page (page : Nat) : Nat :=
  5 * 2^(page - 1)

def total_drawings (n : Nat) : Nat :=
  (List.range n).map drawings_on_page |>.sum

theorem first_five_pages_drawings : total_drawings 5 = 155 := by
  sorry

end NUMINAMATH_CALUDE_first_five_pages_drawings_l1471_147144


namespace NUMINAMATH_CALUDE_bread_cost_l1471_147176

def total_money : ℝ := 60
def celery_cost : ℝ := 5
def cereal_original_cost : ℝ := 12
def cereal_discount : ℝ := 0.5
def milk_original_cost : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_cost : ℝ := 1
def potato_quantity : ℕ := 6
def money_left_for_coffee : ℝ := 26

theorem bread_cost : 
  total_money - 
  (celery_cost + 
   cereal_original_cost * (1 - cereal_discount) + 
   milk_original_cost * (1 - milk_discount) + 
   potato_cost * potato_quantity + 
   money_left_for_coffee) = 8 := by sorry

end NUMINAMATH_CALUDE_bread_cost_l1471_147176


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l1471_147150

theorem exactly_one_greater_than_one (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (prod_one : a * b * c = 1)
  (sum_greater : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l1471_147150


namespace NUMINAMATH_CALUDE_divisibility_properties_l1471_147107

theorem divisibility_properties (n : ℕ) :
  (∃ k : ℤ, 2^n - 1 = 7 * k) ↔ (∃ m : ℕ, n = 3 * m) ∧
  ¬(∃ k : ℤ, 2^n + 1 = 7 * k) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l1471_147107


namespace NUMINAMATH_CALUDE_square_root_existence_l1471_147128

theorem square_root_existence : 
  (∃ x : ℝ, x^2 = (-3)^2) ∧ 
  (∃ x : ℝ, x^2 = 0) ∧ 
  (∃ x : ℝ, x^2 = 1/8) ∧ 
  (¬∃ x : ℝ, x^2 = -6^3) := by
  sorry

end NUMINAMATH_CALUDE_square_root_existence_l1471_147128


namespace NUMINAMATH_CALUDE_probability_sum_five_l1471_147166

def Card : Type := Fin 4

def card_value (c : Card) : ℕ := c.val + 1

def sum_equals_five (c1 c2 : Card) : Prop :=
  card_value c1 + card_value c2 = 5

def total_outcomes : ℕ := 16

def favorable_outcomes : ℕ := 4

theorem probability_sum_five :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_five_l1471_147166


namespace NUMINAMATH_CALUDE_debby_bottles_left_l1471_147160

/-- Calculates the number of water bottles left after drinking a certain amount per day for a number of days. -/
def bottles_left (total : ℕ) (per_day : ℕ) (days : ℕ) : ℕ :=
  total - (per_day * days)

/-- Theorem stating that given 264 initial bottles, drinking 15 per day for 11 days leaves 99 bottles. -/
theorem debby_bottles_left : bottles_left 264 15 11 = 99 := by
  sorry

end NUMINAMATH_CALUDE_debby_bottles_left_l1471_147160


namespace NUMINAMATH_CALUDE_room_population_problem_l1471_147195

theorem room_population_problem (initial_men initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →  -- Initial ratio of men to women is 4:5
  initial_men + 2 = 14 →  -- Final number of men is 14
  (2 * (initial_women - 3) = 24) →  -- Final number of women is 24
  True :=
by sorry

end NUMINAMATH_CALUDE_room_population_problem_l1471_147195


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scaling_l1471_147179

/-- Given two 2D vectors a and b, prove that a - 2b results in the specified coordinates. -/
theorem vector_subtraction_and_scaling (a b : Fin 2 → ℝ) (h1 : a = ![1, 2]) (h2 : b = ![-3, 2]) :
  a - 2 • b = ![7, -2] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scaling_l1471_147179


namespace NUMINAMATH_CALUDE_polynomial_coefficient_bound_l1471_147147

theorem polynomial_coefficient_bound (a b c d : ℝ) : 
  (∀ x : ℝ, |x| < 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) →
  |a| + |b| + |c| + |d| ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_bound_l1471_147147


namespace NUMINAMATH_CALUDE_gcd_power_remainder_l1471_147142

theorem gcd_power_remainder (a b : Nat) : 
  (Nat.gcd (2^(30^10) - 2) (2^(30^45) - 2)) % 2013 = 2012 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_remainder_l1471_147142


namespace NUMINAMATH_CALUDE_light_reflection_theorem_l1471_147173

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Reflects a point across a line -/
def reflectPoint (p : Point) (l : Line) : Point :=
  sorry

/-- Constructs a line passing through two points -/
def lineThrough (p1 p2 : Point) : Line :=
  sorry

theorem light_reflection_theorem (P A : Point) (mirror : Line) :
  P = Point.mk 2 3 →
  A = Point.mk 1 1 →
  mirror = Line.mk 1 1 1 →
  let Q := reflectPoint P mirror
  let incidentRay := lineThrough P (Point.mk mirror.a mirror.b)
  let reflectedRay := lineThrough Q A
  incidentRay = Line.mk 2 (-1) (-1) ∧
  reflectedRay = Line.mk 4 (-5) 1 :=
sorry

end NUMINAMATH_CALUDE_light_reflection_theorem_l1471_147173


namespace NUMINAMATH_CALUDE_pipe_B_fill_time_l1471_147134

/-- The time it takes for pipe A to fill the cistern (in minutes) -/
def time_A : ℝ := 45

/-- The time it takes for the third pipe to empty the cistern (in minutes) -/
def time_empty : ℝ := 72

/-- The time it takes to fill the cistern when all three pipes are open (in minutes) -/
def time_all : ℝ := 40

/-- The time it takes for pipe B to fill the cistern (in minutes) -/
def time_B : ℝ := 60

theorem pipe_B_fill_time :
  ∃ (t : ℝ), t > 0 ∧ 1 / time_A + 1 / t - 1 / time_empty = 1 / time_all ∧ t = time_B := by
  sorry

end NUMINAMATH_CALUDE_pipe_B_fill_time_l1471_147134


namespace NUMINAMATH_CALUDE_company_female_employees_l1471_147187

theorem company_female_employees 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (total_managers : ℕ) 
  (male_managers : ℕ) 
  (h1 : total_managers = (2 : ℕ) * total_employees / (5 : ℕ)) 
  (h2 : male_managers = (2 : ℕ) * male_employees / (5 : ℕ)) 
  (h3 : total_managers = male_managers + 200) :
  total_employees - male_employees = 500 := by
sorry

end NUMINAMATH_CALUDE_company_female_employees_l1471_147187


namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1471_147198

/-- The minimum distance from a point on y = x^2 to y = 2x - 2 is √5/5 -/
theorem min_distance_parabola_to_line :
  let f : ℝ → ℝ := λ x => x^2  -- The curve y = x^2
  let g : ℝ → ℝ := λ x => 2*x - 2  -- The line y = 2x - 2
  ∃ (P : ℝ × ℝ), P.2 = f P.1 ∧  -- Point P on the curve
  (∀ (Q : ℝ × ℝ), Q.2 = f Q.1 →  -- For all points Q on the curve
    Real.sqrt 5 / 5 ≤ Real.sqrt ((Q.1 - (Q.2 + 2) / 2)^2 + (Q.2 - g ((Q.2 + 2) / 2))^2)) ∧
  (∃ (P : ℝ × ℝ), P.2 = f P.1 ∧
    Real.sqrt 5 / 5 = Real.sqrt ((P.1 - (P.2 + 2) / 2)^2 + (P.2 - g ((P.2 + 2) / 2))^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1471_147198


namespace NUMINAMATH_CALUDE_dragon_lion_equivalence_l1471_147182

-- Define the propositions
variable (P Q : Prop)

-- State the theorem
theorem dragon_lion_equivalence :
  (P → Q) ↔ (¬Q → ¬P) ∧ (¬P ∨ Q) :=
sorry

end NUMINAMATH_CALUDE_dragon_lion_equivalence_l1471_147182


namespace NUMINAMATH_CALUDE_cubic_sum_and_product_l1471_147104

theorem cubic_sum_and_product (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 ∧ a*b + b*c + c*a = -(a^3 + 12) / a := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_and_product_l1471_147104


namespace NUMINAMATH_CALUDE_Q_subset_P_l1471_147149

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by sorry

end NUMINAMATH_CALUDE_Q_subset_P_l1471_147149


namespace NUMINAMATH_CALUDE_box_volume_cubic_feet_l1471_147170

/-- Conversion factor from cubic inches to cubic feet -/
def cubic_inches_per_cubic_foot : ℕ := 1728

/-- Volume of the box in cubic inches -/
def box_volume_cubic_inches : ℕ := 1728

/-- Theorem stating that the volume of the box in cubic feet is 1 -/
theorem box_volume_cubic_feet : 
  (box_volume_cubic_inches : ℚ) / cubic_inches_per_cubic_foot = 1 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_cubic_feet_l1471_147170


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l1471_147140

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l1471_147140


namespace NUMINAMATH_CALUDE_opposite_signs_and_sum_negative_l1471_147121

theorem opposite_signs_and_sum_negative (a b : ℚ) 
  (h1 : a * b < 0) 
  (h2 : a + b < 0) : 
  a > 0 ∧ b < 0 ∧ |b| > a := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_and_sum_negative_l1471_147121


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l1471_147120

def matrix1 (a b c d e : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![a, 1, b, 2],
    ![2, 3, 4, 3],
    ![c, 5, d, 3],
    ![2, 4, 1, e]]

def matrix2 (f g h i j k : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![-7, f, -13, 3],
    ![g, -15, h, 2],
    ![3, i, 5, 1],
    ![2, j, 4, k]]

theorem inverse_matrices_sum (a b c d e f g h i j k : ℝ) :
  (matrix1 a b c d e) * (matrix2 f g h i j k) = 1 →
  a + b + c + d + e + f + g + h + i + j + k = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l1471_147120


namespace NUMINAMATH_CALUDE_cricketer_average_score_l1471_147118

/-- 
Given a cricketer whose average score increases by 4 after scoring 95 runs in the 19th inning,
this theorem proves that the cricketer's average score after 19 innings is 23 runs per inning.
-/
theorem cricketer_average_score 
  (initial_average : ℝ) 
  (score_increase : ℝ) 
  (runs_19th_inning : ℕ) :
  score_increase = 4 →
  runs_19th_inning = 95 →
  (18 * initial_average + runs_19th_inning) / 19 = initial_average + score_increase →
  initial_average + score_increase = 23 :=
by
  sorry

#check cricketer_average_score

end NUMINAMATH_CALUDE_cricketer_average_score_l1471_147118


namespace NUMINAMATH_CALUDE_common_ratio_is_negative_two_l1471_147189

def geometric_sequence : ℕ → ℚ
  | 0 => 10
  | 1 => -20
  | 2 => 40
  | 3 => -80
  | _ => 0  -- We only define the first 4 terms as given in the problem

theorem common_ratio_is_negative_two :
  ∀ n : ℕ, n < 3 → geometric_sequence (n + 1) / geometric_sequence n = -2 :=
by
  sorry

#eval geometric_sequence 0
#eval geometric_sequence 1
#eval geometric_sequence 2
#eval geometric_sequence 3

end NUMINAMATH_CALUDE_common_ratio_is_negative_two_l1471_147189


namespace NUMINAMATH_CALUDE_parking_lot_problem_l1471_147155

theorem parking_lot_problem (initial cars_left cars_entered final : ℕ) : 
  cars_left = 13 →
  cars_entered = cars_left + 5 →
  final = 85 →
  final = initial - cars_left + cars_entered →
  initial = 80 := by sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l1471_147155


namespace NUMINAMATH_CALUDE_cos_120_degrees_l1471_147110

theorem cos_120_degrees :
  Real.cos (120 * π / 180) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l1471_147110


namespace NUMINAMATH_CALUDE_book_pricing_problem_l1471_147178

-- Define the variables
variable (price_A : ℝ) (price_B : ℝ) (num_A : ℕ) (num_B : ℕ)

-- Define the conditions
def condition1 : Prop := price_A * num_A = 3000
def condition2 : Prop := price_B * num_B = 1600
def condition3 : Prop := price_A = 1.5 * price_B
def condition4 : Prop := num_A = num_B + 20

-- Define the World Book Day purchase
def world_book_day_expenditure : ℝ := 0.8 * (20 * price_A + 25 * price_B)

-- State the theorem
theorem book_pricing_problem 
  (h1 : condition1 price_A num_A)
  (h2 : condition2 price_B num_B)
  (h3 : condition3 price_A price_B)
  (h4 : condition4 num_A num_B) :
  price_A = 30 ∧ price_B = 20 ∧ world_book_day_expenditure price_A price_B = 880 := by
  sorry


end NUMINAMATH_CALUDE_book_pricing_problem_l1471_147178


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l1471_147181

theorem five_digit_divisible_by_nine :
  ∃ (n : ℕ), 
    n < 10 ∧ 
    (35000 + n * 100 + 72) % 9 = 0 ∧
    (3 + 5 + n + 7 + 2) % 9 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l1471_147181


namespace NUMINAMATH_CALUDE_range_of_a_l1471_147167

theorem range_of_a (A B : Set ℝ) (a : ℝ) :
  A = {x : ℝ | x ≤ 1} →
  B = {x : ℝ | x ≥ a} →
  A ∪ B = Set.univ →
  Set.Iic 1 = {a | ∀ x, (x ∈ A ∪ B ↔ x ∈ Set.univ)} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1471_147167


namespace NUMINAMATH_CALUDE_equation_solution_l1471_147109

theorem equation_solution (x : ℤ) (m : ℕ+) : 
  ((3 * x - 1) / 2 + m = 3) →
  ((m = 5 → x = 1) ∧ 
   (x > 0 → m = 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1471_147109


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l1471_147158

theorem isosceles_right_triangle
  (a b c : ℝ)
  (h1 : ∀ x, (b + c) * x^2 - 2 * a * x + (c - b) = 0 → (∃! y, x = y))
  (h2 : Real.sin b * Real.cos a - Real.cos b * Real.sin a = 0) :
  a = b ∧ a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l1471_147158


namespace NUMINAMATH_CALUDE_min_omega_for_shifted_periodic_function_l1471_147165

/-- The minimum value of ω for a periodic function with a specific shift -/
theorem min_omega_for_shifted_periodic_function (ω : ℝ) (h1 : ω > 0) : 
  (∀ x : ℝ, 3 * Real.sin (ω * x + π / 6) - 2 = 
            3 * Real.sin (ω * (x - 2 * π / 3) + π / 6) - 2) →
  ω ≥ 3 ∧ ∃ n : ℕ, ω = 3 * n :=
by sorry

end NUMINAMATH_CALUDE_min_omega_for_shifted_periodic_function_l1471_147165


namespace NUMINAMATH_CALUDE_ellipse_equation_and_intersection_l1471_147168

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ := (0, 0)
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- Theorem about the ellipse equation and intersection with a line -/
theorem ellipse_equation_and_intersection
  (e : Ellipse)
  (h1 : e.center = (0, 0))
  (h2 : e.foci_on_x_axis = true)
  (h3 : e.eccentricity = Real.sqrt 3 / 2)
  (h4 : e.passes_through = (4, 1)) :
  (∃ (a b : ℝ), ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 20 + y^2 / 5 = 1)) ∧
  (∀ m : ℝ, (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁^2 / 20 + y₁^2 / 5 = 1) ∧ (y₁ = x₁ + m) ∧
    (x₂^2 / 20 + y₂^2 / 5 = 1) ∧ (y₂ = x₂ + m)) ↔
   (-5 < m ∧ m < 5)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_intersection_l1471_147168


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l1471_147125

theorem quadratic_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ x^2 + m*x - 4 = 0) ↔ m > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l1471_147125


namespace NUMINAMATH_CALUDE_alice_has_winning_strategy_l1471_147156

/-- Represents the state of the game with three piles of coins. -/
structure GameState :=
  (pile1 : Nat) (pile2 : Nat) (pile3 : Nat)

/-- Represents a player in the game. -/
inductive Player
  | Alice | Bob | Charlie

/-- Represents a move in the game. -/
structure Move :=
  (pile : Fin 3) (coins : Fin 3)

/-- Defines if a game state is terminal (no coins left). -/
def isTerminal (state : GameState) : Prop :=
  state.pile1 = 0 ∧ state.pile2 = 0 ∧ state.pile3 = 0

/-- Defines a valid move in the game. -/
def validMove (state : GameState) (move : Move) : Prop :=
  match move.pile with
  | 0 => state.pile1 ≥ move.coins
  | 1 => state.pile2 ≥ move.coins
  | 2 => state.pile3 ≥ move.coins

/-- Applies a move to a game state. -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move.pile with
  | 0 => { state with pile1 := state.pile1 - move.coins }
  | 1 => { state with pile2 := state.pile2 - move.coins }
  | 2 => { state with pile3 := state.pile3 - move.coins }

/-- Defines the next player in turn. -/
def nextPlayer : Player → Player
  | Player.Alice => Player.Bob
  | Player.Bob => Player.Charlie
  | Player.Charlie => Player.Alice

/-- Theorem: Alice has a winning strategy in the game starting with piles of 5, 7, and 8 coins. -/
theorem alice_has_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (game : GameState → Player → Prop),
      (∀ s p, game s p → ¬isTerminal s → ∃ m, validMove s m ∧ game (applyMove s m) (nextPlayer p)) →
      (∀ s, isTerminal s → game s Player.Charlie) →
      game { pile1 := 5, pile2 := 7, pile3 := 8 } Player.Alice :=
by sorry

end NUMINAMATH_CALUDE_alice_has_winning_strategy_l1471_147156


namespace NUMINAMATH_CALUDE_average_of_w_and_x_l1471_147196

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 3 / w + 3 / x = 3 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_w_and_x_l1471_147196


namespace NUMINAMATH_CALUDE_units_digit_of_nine_to_eight_to_seven_l1471_147184

theorem units_digit_of_nine_to_eight_to_seven (n : Nat) : n = 9^(8^7) → n % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_nine_to_eight_to_seven_l1471_147184


namespace NUMINAMATH_CALUDE_impossible_assembly_l1471_147146

theorem impossible_assembly (p q r : ℕ) : ¬∃ (x y z : ℕ),
  (2 * p + 2 * r + 2 = 2 * x) ∧
  (2 * p + q + 1 = 2 * x + y) ∧
  (q + r = y + z) :=
by sorry

end NUMINAMATH_CALUDE_impossible_assembly_l1471_147146


namespace NUMINAMATH_CALUDE_intersection_point_l1471_147103

/-- A parametric curve in 2D space -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given parametric curve -/
def givenCurve : ParametricCurve where
  x := fun t => -2 + 5 * t
  y := fun t => 1 - 2 * t

/-- Theorem: The point (1/2, 0) is the intersection of the given curve with the x-axis -/
theorem intersection_point : 
  ∃ t : ℝ, givenCurve.x t = 1/2 ∧ givenCurve.y t = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l1471_147103


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l1471_147141

/-- Given 6 people with an average weight of 156 lbs, if a 7th person enters and
    the new average weight becomes 151 lbs, then the weight of the 7th person is 121 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (new_avg_weight : ℝ) (seventh_person_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 156 →
  new_avg_weight = 151 →
  (initial_people * initial_avg_weight + seventh_person_weight) / (initial_people + 1) = new_avg_weight →
  seventh_person_weight = 121 := by
  sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l1471_147141


namespace NUMINAMATH_CALUDE_min_blocks_for_cube_l1471_147123

/-- The length of the rectangular block -/
def block_length : ℕ := 5

/-- The width of the rectangular block -/
def block_width : ℕ := 4

/-- The height of the rectangular block -/
def block_height : ℕ := 3

/-- The side length of the cube formed by the blocks -/
def cube_side : ℕ := Nat.lcm (Nat.lcm block_length block_width) block_height

/-- The volume of the cube -/
def cube_volume : ℕ := cube_side ^ 3

/-- The volume of a single block -/
def block_volume : ℕ := block_length * block_width * block_height

/-- The number of blocks needed to form the cube -/
def blocks_needed : ℕ := cube_volume / block_volume

theorem min_blocks_for_cube : blocks_needed = 3600 := by
  sorry

end NUMINAMATH_CALUDE_min_blocks_for_cube_l1471_147123


namespace NUMINAMATH_CALUDE_concentration_reduction_proof_l1471_147154

def initial_concentration : ℝ := 0.9
def target_concentration : ℝ := 0.1
def concentration_reduction_factor : ℝ := 0.9

def minimum_operations : ℕ := 21

theorem concentration_reduction_proof :
  (∀ n : ℕ, n < minimum_operations → initial_concentration * concentration_reduction_factor ^ n ≥ target_concentration) ∧
  initial_concentration * concentration_reduction_factor ^ minimum_operations < target_concentration :=
by sorry

end NUMINAMATH_CALUDE_concentration_reduction_proof_l1471_147154


namespace NUMINAMATH_CALUDE_picture_area_l1471_147192

/-- The area of a picture on a sheet of paper with given dimensions and margins. -/
theorem picture_area (paper_width paper_length margin : ℝ) 
  (hw : paper_width = 8.5)
  (hl : paper_length = 10)
  (hm : margin = 1.5) : 
  (paper_width - 2 * margin) * (paper_length - 2 * margin) = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l1471_147192


namespace NUMINAMATH_CALUDE_ned_remaining_lives_l1471_147145

/-- Given that Ned started with 83 lives and lost 13 lives, prove that he now has 70 lives. -/
theorem ned_remaining_lives (initial_lives : ℕ) (lost_lives : ℕ) (remaining_lives : ℕ) : 
  initial_lives = 83 → lost_lives = 13 → remaining_lives = initial_lives - lost_lives → remaining_lives = 70 := by
  sorry

end NUMINAMATH_CALUDE_ned_remaining_lives_l1471_147145


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1471_147127

/-- The distance between the vertices of the hyperbola x²/121 - y²/36 = 1 is 22 -/
theorem hyperbola_vertices_distance : 
  let a : ℝ := Real.sqrt 121
  let b : ℝ := Real.sqrt 36
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 121 - y^2 / 36 = 1
  2 * a = 22 := by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1471_147127


namespace NUMINAMATH_CALUDE_log_inequality_l1471_147183

theorem log_inequality (a b : ℝ) 
  (ha : a = Real.log 0.4 / Real.log 0.2) 
  (hb : b = 1 - 1 / Real.log 4) : 
  a * b < a + b ∧ a + b < 0 := by sorry

end NUMINAMATH_CALUDE_log_inequality_l1471_147183


namespace NUMINAMATH_CALUDE_product_of_fractions_l1471_147171

theorem product_of_fractions (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1471_147171


namespace NUMINAMATH_CALUDE_value_set_of_t_l1471_147161

/-- The value set of t given the conditions -/
theorem value_set_of_t (t : ℝ) : 
  (∀ y, y > 2 * 1 - t + 1 → (1, y) = (1, t)) → 
  (∀ x, x^2 + (2*t - 4)*x + 4 > 0) → 
  3 < t ∧ t < 4 := by
  sorry

end NUMINAMATH_CALUDE_value_set_of_t_l1471_147161


namespace NUMINAMATH_CALUDE_same_floor_prob_is_one_fifth_l1471_147164

/-- A hotel with 6 rooms distributed across 3 floors -/
structure Hotel :=
  (total_rooms : ℕ)
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (h1 : total_rooms = 6)
  (h2 : floors = 3)
  (h3 : rooms_per_floor = 2)
  (h4 : total_rooms = floors * rooms_per_floor)

/-- The probability of two people choosing rooms on the same floor -/
def same_floor_probability (h : Hotel) : ℚ :=
  (h.floors * (h.rooms_per_floor * (h.rooms_per_floor - 1))) / (h.total_rooms * (h.total_rooms - 1))

theorem same_floor_prob_is_one_fifth (h : Hotel) : same_floor_probability h = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_floor_prob_is_one_fifth_l1471_147164


namespace NUMINAMATH_CALUDE_sugar_for_muffins_l1471_147105

/-- Given a recipe for muffins, calculate the required sugar for a larger batch -/
theorem sugar_for_muffins (original_muffins original_sugar target_muffins : ℕ) :
  original_muffins > 0 →
  original_sugar > 0 →
  target_muffins > 0 →
  (original_sugar * target_muffins) / original_muffins = 
    (3 * 72) / 24 :=
by
  sorry

#eval (3 * 72) / 24  -- This should output 9

end NUMINAMATH_CALUDE_sugar_for_muffins_l1471_147105


namespace NUMINAMATH_CALUDE_quadratic_shift_l1471_147191

/-- Represents a quadratic function of the form y = (x + a)^2 + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Shifts a quadratic function horizontally -/
def horizontalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a - shift, b := f.b }

/-- Shifts a quadratic function vertically -/
def verticalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b + shift }

/-- The main theorem stating that shifting y = (x+2)^2 - 3 by 1 unit left and 2 units up
    results in y = (x+3)^2 - 1 -/
theorem quadratic_shift :
  let f := QuadraticFunction.mk 2 (-3)
  let g := verticalShift (horizontalShift f 1) 2
  g = QuadraticFunction.mk 3 (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_l1471_147191


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1471_147100

theorem sufficient_not_necessary (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∃ b : ℝ, b ≠ 2 ∧ (b - 1) * (b - 2) = 0) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1471_147100


namespace NUMINAMATH_CALUDE_equation_solution_l1471_147129

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 90 + 5 * 12 / (180 / x) = 91 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1471_147129


namespace NUMINAMATH_CALUDE_right_triangle_median_on_hypotenuse_l1471_147101

theorem right_triangle_median_on_hypotenuse (a b : ℝ) (h : a = 5 ∧ b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  (c / 2) = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_median_on_hypotenuse_l1471_147101


namespace NUMINAMATH_CALUDE_total_points_is_1320_l1471_147186

def freshman_points : ℕ := 260

def sophomore_points : ℕ := freshman_points + (freshman_points * 15 / 100)

def junior_points : ℕ := sophomore_points + (sophomore_points * 20 / 100)

def senior_points : ℕ := junior_points + (junior_points * 12 / 100)

def total_points : ℕ := freshman_points + sophomore_points + junior_points + senior_points

theorem total_points_is_1320 : total_points = 1320 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_1320_l1471_147186


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l1471_147194

/-- Given a class of students and their dessert preferences, calculate the number of students who like both desserts. -/
theorem students_liking_both_desserts
  (total : ℕ)
  (like_apple : ℕ)
  (like_chocolate : ℕ)
  (like_neither : ℕ)
  (h1 : total = 35)
  (h2 : like_apple = 20)
  (h3 : like_chocolate = 17)
  (h4 : like_neither = 8) :
  like_apple + like_chocolate - (total - like_neither) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l1471_147194


namespace NUMINAMATH_CALUDE_point_distance_inequality_l1471_147151

theorem point_distance_inequality (x : ℝ) : 
  (|x - 0| > |x - (-1)|) → x < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_inequality_l1471_147151


namespace NUMINAMATH_CALUDE_gray_trees_sum_l1471_147169

/-- Represents the number of trees in a photograph -/
structure PhotoTrees where
  total : ℕ
  white : ℕ
  gray : ℕ

/-- The problem statement -/
theorem gray_trees_sum (photo1 photo2 photo3 : PhotoTrees) :
  photo1.total = 100 →
  photo2.total = 90 →
  photo3.total = photo3.white →
  photo1.white = photo2.white →
  photo2.white = photo3.white →
  photo3.white = 82 →
  photo1.gray + photo2.gray = 26 :=
by sorry

end NUMINAMATH_CALUDE_gray_trees_sum_l1471_147169


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1471_147126

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of sum for arithmetic sequence
  (∀ n, ∃ d, a (n + 1) - a n = d) →     -- Definition of arithmetic sequence
  a 1 = -2016 →                         -- Given condition
  (S 2015) / 2015 - (S 2012) / 2012 = 3 →  -- Given condition
  S 2016 = -2016 :=                     -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1471_147126


namespace NUMINAMATH_CALUDE_remainder_double_n_l1471_147174

theorem remainder_double_n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_double_n_l1471_147174


namespace NUMINAMATH_CALUDE_sum_five_consecutive_squares_not_perfect_square_l1471_147185

theorem sum_five_consecutive_squares_not_perfect_square (n : ℤ) :
  ¬∃ (m : ℤ), 5 * n^2 + 10 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_squares_not_perfect_square_l1471_147185


namespace NUMINAMATH_CALUDE_rebeccas_marbles_l1471_147175

/-- Rebecca's egg and marble problem -/
theorem rebeccas_marbles :
  ∀ (num_eggs num_marbles : ℕ),
  num_eggs = 20 →
  num_eggs = num_marbles + 14 →
  num_marbles = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_rebeccas_marbles_l1471_147175


namespace NUMINAMATH_CALUDE_sequence_sum_l1471_147114

theorem sequence_sum : 
  let a₁ : ℚ := 4/3
  let a₂ : ℚ := 7/5
  let a₃ : ℚ := 11/8
  let a₄ : ℚ := 19/15
  let a₅ : ℚ := 35/27
  let a₆ : ℚ := 67/52
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ - 9 = -17312.5 / 7020 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_l1471_147114


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l1471_147102

/-- The rate of mangoes per kg given the purchase details --/
theorem mango_rate_calculation (grape_quantity : ℕ) (grape_rate : ℕ) 
  (mango_quantity : ℕ) (total_paid : ℕ) : 
  grape_quantity = 8 →
  grape_rate = 70 →
  mango_quantity = 9 →
  total_paid = 965 →
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 45 :=
by sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l1471_147102


namespace NUMINAMATH_CALUDE_net_population_increase_l1471_147190

/-- The net population increase in one day given specific birth and death rates -/
theorem net_population_increase (birth_rate : ℕ) (death_rate : ℕ) (seconds_per_interval : ℕ) (seconds_per_day : ℕ) :
  birth_rate = 4 →
  death_rate = 2 →
  seconds_per_interval = 2 →
  seconds_per_day = 86400 →
  (birth_rate - death_rate) * (seconds_per_day / seconds_per_interval) = 86400 := by
  sorry

#check net_population_increase

end NUMINAMATH_CALUDE_net_population_increase_l1471_147190


namespace NUMINAMATH_CALUDE_complex_symmetry_quotient_l1471_147122

theorem complex_symmetry_quotient : 
  ∀ (z₁ z₂ : ℂ), 
  (z₁.im = -z₂.im) → 
  (z₁.re = z₂.re) → 
  (z₁ = 2 + I) → 
  (z₁ / z₂ = (3/5 : ℂ) + (4/5 : ℂ) * I) := by
sorry

end NUMINAMATH_CALUDE_complex_symmetry_quotient_l1471_147122


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1471_147132

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1471_147132


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l1471_147180

theorem fourth_term_of_sequence (x : ℤ) : 
  x^2 - 2*x - 3 < 0 → 
  ∃ (a : ℕ → ℤ), (∀ n, a (n+1) - a n = a 1 - a 0) ∧ 
                 (∀ n, a n = x → x^2 - 2*x - 3 < 0) ∧
                 (a 3 = 3 ∨ a 3 = -1) :=
sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l1471_147180


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1471_147162

/-- Calculates the time required to travel between two cities given a map scale, distance on the map, and car speed. -/
theorem travel_time_calculation (scale : ℚ) (map_distance : ℚ) (car_speed : ℚ) :
  scale = 1 / 3000000 →
  map_distance = 6 →
  car_speed = 30 →
  (map_distance * scale * 100000) / car_speed = 6000 := by
  sorry

#check travel_time_calculation

end NUMINAMATH_CALUDE_travel_time_calculation_l1471_147162


namespace NUMINAMATH_CALUDE_cannon_hit_probability_l1471_147112

theorem cannon_hit_probability (P1 P2 P3 : ℝ) : 
  P1 = 0.2 →
  P3 = 0.3 →
  (1 - P1) * (1 - P2) * (1 - P3) = 0.27999999999999997 →
  P2 = 0.5 := by
sorry

end NUMINAMATH_CALUDE_cannon_hit_probability_l1471_147112


namespace NUMINAMATH_CALUDE_keno_probability_value_l1471_147193

/-- The set of integers from 1 to 80 -/
def keno_numbers : Finset Nat := Finset.range 80

/-- The set of numbers from 1 to 80 that contain the digit 8 -/
def numbers_with_eight : Finset Nat := {8, 18, 28, 38, 48, 58, 68, 78}

/-- The set of numbers from 1 to 80 that do not contain the digit 8 -/
def numbers_without_eight : Finset Nat := keno_numbers \ numbers_with_eight

/-- The number of numbers to be drawn in a KENO game -/
def draw_count : Nat := 20

/-- The probability of drawing 20 numbers from 1 to 80 such that none contain the digit 8 -/
def keno_probability : ℚ := (Nat.choose numbers_without_eight.card draw_count : ℚ) / (Nat.choose keno_numbers.card draw_count)

theorem keno_probability_value : keno_probability = 27249 / 4267580 := by
  sorry

end NUMINAMATH_CALUDE_keno_probability_value_l1471_147193


namespace NUMINAMATH_CALUDE_product_sum_relation_l1471_147108

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 14 → b = 8 → b - a = 3 := by sorry

end NUMINAMATH_CALUDE_product_sum_relation_l1471_147108


namespace NUMINAMATH_CALUDE_reps_before_high_elevation_pushups_l1471_147139

/-- Calculates the number of reps reached before moving to the next push-up type -/
def repsBeforeNextType (totalWeeks : ℕ) (typesOfPushups : ℕ) (daysPerWeek : ℕ) (repsAddedPerDay : ℕ) (initialReps : ℕ) : ℕ :=
  let weeksPerType : ℕ := totalWeeks / typesOfPushups
  let totalDays : ℕ := weeksPerType * daysPerWeek
  initialReps + (totalDays * repsAddedPerDay)

theorem reps_before_high_elevation_pushups :
  repsBeforeNextType 9 4 5 1 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_reps_before_high_elevation_pushups_l1471_147139


namespace NUMINAMATH_CALUDE_car_journey_distance_l1471_147136

/-- Proves that given a car that travels a certain distance in 9 hours for the forward journey,
    and returns with a speed increased by 20 km/hr in 6 hours, the distance traveled is 360 km. -/
theorem car_journey_distance : ∀ (v : ℝ),
  v * 9 = (v + 20) * 6 →
  v * 9 = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_car_journey_distance_l1471_147136


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1471_147131

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 2*y = 1) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 2 ∧ ∀ z, z = 2^x + 4^y → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1471_147131


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1471_147116

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 6 →
  downstream_distance = 35.2 →
  downstream_time = 44 / 60 →
  ∃ (boat_speed : ℝ), boat_speed = 42 ∧ 
    downstream_distance = (boat_speed + current_speed) * downstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1471_147116


namespace NUMINAMATH_CALUDE_grain_transfer_theorem_transfer_valid_l1471_147152

/-- The amount of grain to be transferred from Warehouse B to Warehouse A -/
def transfer : ℕ := 15

/-- The initial amount of grain in Warehouse A -/
def initial_A : ℕ := 540

/-- The initial amount of grain in Warehouse B -/
def initial_B : ℕ := 200

/-- Theorem stating that transferring the specified amount will result in
    Warehouse A having three times the grain of Warehouse B -/
theorem grain_transfer_theorem :
  (initial_A + transfer) = 3 * (initial_B - transfer) := by
  sorry

/-- Proof that the transfer amount is non-negative and not greater than
    the initial amount in Warehouse B -/
theorem transfer_valid :
  0 ≤ transfer ∧ transfer ≤ initial_B := by
  sorry

end NUMINAMATH_CALUDE_grain_transfer_theorem_transfer_valid_l1471_147152


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_plus_constant_l1471_147106

/-- For any positive real number a, the function f(x) = a^x + 4 passes through the point (0, 5) -/
theorem fixed_point_of_exponential_plus_constant (a : ℝ) (ha : a > 0) :
  let f := fun (x : ℝ) => a^x + 4
  f 0 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_plus_constant_l1471_147106


namespace NUMINAMATH_CALUDE_remainder_problem_l1471_147159

theorem remainder_problem : (11^7 + 9^8 + 7^9) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1471_147159


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1471_147113

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Check if a ScientificNotation represents a given real number -/
def represents (sn : ScientificNotation) (x : ℝ) : Prop :=
  x = sn.coefficient * (10 : ℝ) ^ sn.exponent

/-- The number we want to represent in scientific notation -/
def target_number : ℝ := 37000000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 3.7
    exponent := 7
    h_coeff_range := by sorry }

theorem scientific_notation_correct :
  represents proposed_notation target_number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1471_147113
