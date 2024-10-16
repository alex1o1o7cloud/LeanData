import Mathlib

namespace NUMINAMATH_CALUDE_may_scarves_total_l1981_198188

theorem may_scarves_total (scarves_per_yarn : ℕ) (red_yarns blue_yarns yellow_yarns : ℕ)
  (h1 : scarves_per_yarn = 3)
  (h2 : red_yarns = 2)
  (h3 : blue_yarns = 6)
  (h4 : yellow_yarns = 4) :
  scarves_per_yarn * (red_yarns + blue_yarns + yellow_yarns) = 36 :=
by sorry

end NUMINAMATH_CALUDE_may_scarves_total_l1981_198188


namespace NUMINAMATH_CALUDE_nancy_crayon_packs_l1981_198110

theorem nancy_crayon_packs : ∀ (total_crayons pack_size num_packs : ℕ),
  total_crayons = 615 →
  pack_size = 15 →
  total_crayons = pack_size * num_packs →
  num_packs = 41 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayon_packs_l1981_198110


namespace NUMINAMATH_CALUDE_wedding_guests_l1981_198115

/-- The number of guests at Jenny's wedding --/
def total_guests : ℕ := 80

/-- The number of guests who want chicken --/
def chicken_guests : ℕ := 20

/-- The number of guests who want steak --/
def steak_guests : ℕ := 60

/-- The cost of a chicken entree in dollars --/
def chicken_cost : ℕ := 18

/-- The cost of a steak entree in dollars --/
def steak_cost : ℕ := 25

/-- The total catering budget in dollars --/
def total_budget : ℕ := 1860

theorem wedding_guests :
  (chicken_guests + steak_guests = total_guests) ∧
  (steak_guests = 3 * chicken_guests) ∧
  (chicken_cost * chicken_guests + steak_cost * steak_guests = total_budget) := by
  sorry

end NUMINAMATH_CALUDE_wedding_guests_l1981_198115


namespace NUMINAMATH_CALUDE_min_omega_for_sine_symmetry_l1981_198128

theorem min_omega_for_sine_symmetry :
  ∀ ω : ℕ+,
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sin (ω * x + π / 6)) →
  (∀ x : ℝ, Real.sin (ω * (π / 3 - x) + π / 6) = Real.sin (ω * x + π / 6)) →
  2 ≤ ω :=
by
  sorry

end NUMINAMATH_CALUDE_min_omega_for_sine_symmetry_l1981_198128


namespace NUMINAMATH_CALUDE_binomial_60_3_l1981_198105

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l1981_198105


namespace NUMINAMATH_CALUDE_prob_two_heads_in_three_fair_coin_l1981_198181

/-- A fair coin is a coin with probability 1/2 of landing heads -/
def fairCoin (p : ℝ) : Prop := p = (1 : ℝ) / 2

/-- The probability of getting exactly two heads in three independent coin flips -/
def probTwoHeadsInThree (p : ℝ) : ℝ := 3 * p^2 * (1 - p)

/-- Theorem: The probability of getting exactly two heads in three flips of a fair coin is 3/8 -/
theorem prob_two_heads_in_three_fair_coin :
  ∀ p : ℝ, fairCoin p → probTwoHeadsInThree p = (3 : ℝ) / 8 :=
by sorry

end NUMINAMATH_CALUDE_prob_two_heads_in_three_fair_coin_l1981_198181


namespace NUMINAMATH_CALUDE_equation_representation_l1981_198184

theorem equation_representation (x : ℝ) : 
  (2 * x + 4 = 8) → (∃ y : ℝ, y = 2 * x + 4 ∧ y = 8) := by
  sorry

end NUMINAMATH_CALUDE_equation_representation_l1981_198184


namespace NUMINAMATH_CALUDE_max_knights_is_eight_l1981_198119

/-- Represents a person who can be either a knight or a liar -/
inductive Person
| knight
| liar

/-- The type of statements a person can make about their number -/
inductive Statement
| greater_than (n : ℕ)
| less_than (n : ℕ)

/-- A function that determines if a statement is true for a given number -/
def is_true_statement (s : Statement) (num : ℕ) : Prop :=
  match s with
  | Statement.greater_than n => num > n
  | Statement.less_than n => num < n

/-- A function that determines if a person's statements are consistent with their type -/
def consistent_statements (p : Person) (num : ℕ) (s1 s2 : Statement) : Prop :=
  match p with
  | Person.knight => is_true_statement s1 num ∧ is_true_statement s2 num
  | Person.liar => ¬(is_true_statement s1 num) ∧ ¬(is_true_statement s2 num)

theorem max_knights_is_eight :
  ∃ (people : Fin 10 → Person) (numbers : Fin 10 → ℕ) 
    (statements1 statements2 : Fin 10 → Statement),
    (∀ i : Fin 10, ∃ n : ℕ, statements1 i = Statement.greater_than n ∧ n = i.val + 1) ∧
    (∀ i : Fin 10, ∃ n : ℕ, statements2 i = Statement.less_than n ∧ n ≤ 10) ∧
    (∀ i : Fin 10, consistent_statements (people i) (numbers i) (statements1 i) (statements2 i)) ∧
    (∀ n : ℕ, n > 8 → ¬∃ (people : Fin n → Person) (numbers : Fin n → ℕ) 
      (statements1 statements2 : Fin n → Statement),
      (∀ i : Fin n, ∃ m : ℕ, statements1 i = Statement.greater_than m ∧ m = i.val + 1) ∧
      (∀ i : Fin n, ∃ m : ℕ, statements2 i = Statement.less_than m ∧ m ≤ n) ∧
      (∀ i : Fin n, consistent_statements (people i) (numbers i) (statements1 i) (statements2 i)) ∧
      (∀ i : Fin n, people i = Person.knight)) :=
by sorry

end NUMINAMATH_CALUDE_max_knights_is_eight_l1981_198119


namespace NUMINAMATH_CALUDE_dessert_cost_calculation_dessert_cost_is_eleven_l1981_198198

/-- Calculates the cost of a dessert given the costs of other meal components and the total price --/
theorem dessert_cost_calculation 
  (appetizer_cost : ℝ) 
  (entree_cost : ℝ) 
  (tip_percentage : ℝ) 
  (total_price : ℝ) : ℝ :=
  let base_cost := appetizer_cost + 2 * entree_cost
  let dessert_cost := (total_price - base_cost) / (1 + tip_percentage)
  dessert_cost

/-- Proves that the dessert cost is $11.00 given the specific meal costs --/
theorem dessert_cost_is_eleven :
  dessert_cost_calculation 9 20 0.3 78 = 11 := by
  sorry

end NUMINAMATH_CALUDE_dessert_cost_calculation_dessert_cost_is_eleven_l1981_198198


namespace NUMINAMATH_CALUDE_propositions_p_and_q_true_l1981_198124

theorem propositions_p_and_q_true : (∃ x₀ : ℝ, x₀^2 < x₀) ∧ (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_true_l1981_198124


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1981_198109

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (1 + 2*i) / (i - 1)
  Complex.im z = -3/2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1981_198109


namespace NUMINAMATH_CALUDE_product_of_squares_l1981_198150

theorem product_of_squares (N : ℕ+) 
  (h : ∃! (a₁ b₁ a₂ b₂ a₃ b₃ : ℕ+), 
    a₁^2 * b₁^2 = N ∧ 
    a₂^2 * b₂^2 = N ∧ 
    a₃^2 * b₃^2 = N ∧
    (a₁, b₁) ≠ (a₂, b₂) ∧ 
    (a₁, b₁) ≠ (a₃, b₃) ∧ 
    (a₂, b₂) ≠ (a₃, b₃)) :
  ∃ (a₁ b₁ a₂ b₂ a₃ b₃ : ℕ+), 
    a₁^2 * b₁^2 * a₂^2 * b₂^2 * a₃^2 * b₃^2 = N^3 :=
sorry

end NUMINAMATH_CALUDE_product_of_squares_l1981_198150


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_theorem_l1981_198113

/-- The area of an isosceles triangle with two sides of length 5 units and a base of 6 units -/
def isosceles_triangle_area : ℝ := 12

/-- The length of the two equal sides of the isosceles triangle -/
def side_length : ℝ := 5

/-- The length of the base of the isosceles triangle -/
def base_length : ℝ := 6

theorem isosceles_triangle_area_theorem :
  let a := side_length
  let b := base_length
  let height := Real.sqrt (a^2 - (b/2)^2)
  (1/2) * b * height = isosceles_triangle_area :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_theorem_l1981_198113


namespace NUMINAMATH_CALUDE_f_max_is_k_max_b_ac_l1981_198162

/-- The function f(x) = |x-1| - 2|x+1| --/
def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

/-- The maximum value of f(x) --/
def k : ℝ := 2

/-- Theorem stating that k is the maximum value of f(x) --/
theorem f_max_is_k : ∀ x : ℝ, f x ≤ k :=
sorry

/-- Theorem for the maximum value of b(a+c) given the conditions --/
theorem max_b_ac (a b c : ℝ) (h : (a^2 + c^2) / 2 + b^2 = k) :
  b * (a + c) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_f_max_is_k_max_b_ac_l1981_198162


namespace NUMINAMATH_CALUDE_sin_160_eq_sin_20_l1981_198185

theorem sin_160_eq_sin_20 : Real.sin (160 * π / 180) = Real.sin (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_160_eq_sin_20_l1981_198185


namespace NUMINAMATH_CALUDE_common_measure_of_angles_l1981_198168

-- Define the angles and natural numbers
variable (α β : ℝ)
variable (m n : ℕ)

-- State the theorem
theorem common_measure_of_angles (h : α = β * (m / n)) :
  α / m = β / n ∧ 
  ∃ (k₁ k₂ : ℕ), α = k₁ * (α / m) ∧ β = k₂ * (β / n) :=
sorry

end NUMINAMATH_CALUDE_common_measure_of_angles_l1981_198168


namespace NUMINAMATH_CALUDE_no_nonzero_real_solution_l1981_198169

theorem no_nonzero_real_solution :
  ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 1/a + 1/b = 2/(a+b) := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_real_solution_l1981_198169


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1981_198152

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 3| < 1} = {x : ℝ | -4 < x ∧ x < -2} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1981_198152


namespace NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l1981_198164

theorem right_triangles_shared_hypotenuse (a : ℝ) (h : a ≥ Real.sqrt 7) :
  let BC : ℝ := 3
  let AC : ℝ := a
  let AD : ℝ := 4
  let AB : ℝ := Real.sqrt (AC^2 + BC^2)
  let BD : ℝ := Real.sqrt (AB^2 - AD^2)
  BD = Real.sqrt (a^2 - 7) :=
by sorry

end NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l1981_198164


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1981_198137

theorem polynomial_divisibility : ∃ (q : ℝ → ℝ), ∀ x : ℝ, 
  4 * x^2 - 6 * x - 18 = (x - 3) * q x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1981_198137


namespace NUMINAMATH_CALUDE_circle_radius_tripled_area_l1981_198187

theorem circle_radius_tripled_area (n : ℝ) :
  ∃ r : ℝ, r > 0 ∧ π * (r + n)^2 = 3 * π * r^2 → r = n * (Real.sqrt 3 - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_tripled_area_l1981_198187


namespace NUMINAMATH_CALUDE_unique_top_coloring_l1981_198146

/-- Represents the colors used for the cube corners -/
inductive Color
  | Red
  | Green
  | Blue
  | Purple

/-- Represents a corner of the cube -/
structure Corner where
  position : Fin 8
  color : Color

/-- Represents a cube with colored corners -/
structure ColoredCube where
  corners : Fin 8 → Corner

/-- Checks if all corners on a face have different colors -/
def faceHasDifferentColors (cube : ColoredCube) (face : Fin 6) : Prop := sorry

/-- Checks if the bottom four corners of the cube are colored with four different colors -/
def bottomCornersAreDifferent (cube : ColoredCube) : Prop := sorry

/-- The main theorem stating that there is only one way to color the top corners -/
theorem unique_top_coloring (cube : ColoredCube) : 
  bottomCornersAreDifferent cube →
  (∀ face, faceHasDifferentColors cube face) →
  ∃! topColoring : Fin 4 → Color, 
    ∀ i : Fin 4, (cube.corners (i + 4)).color = topColoring i :=
sorry

end NUMINAMATH_CALUDE_unique_top_coloring_l1981_198146


namespace NUMINAMATH_CALUDE_football_field_length_prove_football_field_length_l1981_198116

theorem football_field_length : ℝ → Prop :=
  fun length =>
    (4 * length + 500 = 1172) →
    length = 168

-- The proof is omitted
theorem prove_football_field_length : football_field_length 168 := by
  sorry

end NUMINAMATH_CALUDE_football_field_length_prove_football_field_length_l1981_198116


namespace NUMINAMATH_CALUDE_problem_solution_l1981_198129

noncomputable def x : ℝ := Real.sqrt (19 - 8 * Real.sqrt 3)

theorem problem_solution : 
  (x^4 - 6*x^3 - 2*x^2 + 18*x + 23) / (x^2 - 8*x + 15) = 5 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1981_198129


namespace NUMINAMATH_CALUDE_guitar_picks_problem_l1981_198156

theorem guitar_picks_problem (total : ℕ) (red blue yellow : ℕ) : 
  total > 0 ∧ 
  red = total / 2 ∧ 
  blue = total / 3 ∧ 
  yellow = 6 ∧ 
  red + blue + yellow = total → 
  blue = 12 := by
sorry

end NUMINAMATH_CALUDE_guitar_picks_problem_l1981_198156


namespace NUMINAMATH_CALUDE_f_is_directly_proportional_l1981_198147

/-- A function f is directly proportional if there exists a constant k such that f(x) = k * x for all x. -/
def IsDirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function f(x) = -x -/
def f (x : ℝ) : ℝ := -x

/-- Theorem: The function f(x) = -x is directly proportional -/
theorem f_is_directly_proportional : IsDirectlyProportional f := by
  sorry

end NUMINAMATH_CALUDE_f_is_directly_proportional_l1981_198147


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l1981_198153

theorem product_from_lcm_gcd (a b : ℕ+) (h1 : Nat.lcm a b = 120) (h2 : Nat.gcd a b = 8) :
  a * b = 960 := by sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l1981_198153


namespace NUMINAMATH_CALUDE_exterior_angle_theorem_l1981_198142

/-- The measure of the exterior angle BAC formed by a square and a regular octagon sharing a common side --/
def exterior_angle_measure : ℝ := 135

/-- A square and a regular octagon are coplanar and share a common side AD --/
axiom share_common_side : True

/-- Theorem: The measure of the exterior angle BAC is 135 degrees --/
theorem exterior_angle_theorem : exterior_angle_measure = 135 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_theorem_l1981_198142


namespace NUMINAMATH_CALUDE_rectangle_area_l1981_198182

/-- A rectangle with diagonal length x and length twice its width has area (2/5)x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ (w l : ℝ),
  w > 0 ∧ l > 0 ∧ l = 2 * w ∧ w^2 + l^2 = x^2 ∧ w * l = (2/5) * x^2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l1981_198182


namespace NUMINAMATH_CALUDE_columbus_discovery_year_l1981_198174

def is_15th_century (year : ℕ) : Prop := 1400 ≤ year ∧ year ≤ 1499

def sum_of_digits (year : ℕ) : ℕ :=
  (year / 1000) + ((year / 100) % 10) + ((year / 10) % 10) + (year % 10)

def tens_digit (year : ℕ) : ℕ := (year / 10) % 10

def units_digit (year : ℕ) : ℕ := year % 10

theorem columbus_discovery_year :
  ∃! year : ℕ,
    is_15th_century year ∧
    sum_of_digits year = 16 ∧
    tens_digit year / units_digit year = 4 ∧
    tens_digit year % units_digit year = 1 ∧
    year = 1492 :=
by
  sorry

end NUMINAMATH_CALUDE_columbus_discovery_year_l1981_198174


namespace NUMINAMATH_CALUDE_salt_solution_volume_l1981_198141

/-- Proves that the initial volume of a solution is 80 gallons, given the conditions of the problem -/
theorem salt_solution_volume : 
  ∀ (V : ℝ), 
  (0.1 * V = 0.08 * (V + 20)) → 
  V = 80 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l1981_198141


namespace NUMINAMATH_CALUDE_opposite_of_one_over_twentythree_l1981_198127

theorem opposite_of_one_over_twentythree :
  ∀ x : ℚ, x = 1 / 23 → -x = -(1 / 23) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_over_twentythree_l1981_198127


namespace NUMINAMATH_CALUDE_pink_balls_count_l1981_198195

theorem pink_balls_count (initial_green : ℕ) (added_green : ℕ) (initial_pink : ℕ) : 
  initial_green = 9 →
  added_green = 14 →
  initial_green + added_green = initial_pink →
  initial_pink = 23 := by
sorry

end NUMINAMATH_CALUDE_pink_balls_count_l1981_198195


namespace NUMINAMATH_CALUDE_pastry_distribution_combinations_l1981_198160

/-- The number of ways to distribute additional items among a subset of groups,
    given that each group already has one item. -/
def distribute_additional_items (total_items : ℕ) (total_groups : ℕ) (subset_groups : ℕ) : ℕ :=
  Nat.choose (subset_groups + (total_items - total_groups) - 1) (subset_groups - 1)

/-- Theorem stating that distributing 3 additional items among 4 groups,
    given that 5 items have already been distributed among 5 groups, results in 20 combinations. -/
theorem pastry_distribution_combinations :
  distribute_additional_items 8 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pastry_distribution_combinations_l1981_198160


namespace NUMINAMATH_CALUDE_digit_59_is_4_l1981_198166

/-- The decimal representation of 1/17 as a list of digits -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating sequence in the decimal representation of 1/17 -/
def cycle_length : Nat := 16

/-- The 59th digit after the decimal point in the decimal representation of 1/17 -/
def digit_59 : Nat := decimal_rep_1_17[(59 - 1) % cycle_length]

theorem digit_59_is_4 : digit_59 = 4 := by sorry

end NUMINAMATH_CALUDE_digit_59_is_4_l1981_198166


namespace NUMINAMATH_CALUDE_edward_spent_thirteen_l1981_198183

/-- The amount of money Edward spent -/
def amount_spent (initial_amount current_amount : ℕ) : ℕ :=
  initial_amount - current_amount

/-- Theorem: Edward spent $13 -/
theorem edward_spent_thirteen : amount_spent 19 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_edward_spent_thirteen_l1981_198183


namespace NUMINAMATH_CALUDE_triangle_third_side_validity_l1981_198194

theorem triangle_third_side_validity (a b c : ℝ) : 
  a = 4 → b = 10 → c = 11 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
  (c < a + b ∧ a < b + c ∧ b < c + a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_validity_l1981_198194


namespace NUMINAMATH_CALUDE_min_distance_from_point_on_unit_circle_l1981_198167

theorem min_distance_from_point_on_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z - (3 + 4 * Complex.I)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_from_point_on_unit_circle_l1981_198167


namespace NUMINAMATH_CALUDE_i_cubed_plus_i_squared_in_third_quadrant_l1981_198197

theorem i_cubed_plus_i_squared_in_third_quadrant :
  let z : ℂ := Complex.I^3 + Complex.I^2
  (z.re < 0) ∧ (z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_i_cubed_plus_i_squared_in_third_quadrant_l1981_198197


namespace NUMINAMATH_CALUDE_lcm_gcd_product_36_60_l1981_198134

theorem lcm_gcd_product_36_60 : Nat.lcm 36 60 * Nat.gcd 36 60 = 2160 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_36_60_l1981_198134


namespace NUMINAMATH_CALUDE_tangent_pentagon_division_l1981_198196

/-- A pentagon with sides tangent to a circle --/
structure TangentPentagon where
  /-- Lengths of the sides of the pentagon --/
  sides : Fin 5 → ℝ
  /-- All sides are positive --/
  sides_pos : ∀ i, sides i > 0
  /-- The sides are in the specified order --/
  sides_order : sides 0 = 5 ∧ sides 1 = 6 ∧ sides 2 = 7 ∧ sides 3 = 8 ∧ sides 4 = 9
  /-- The sides are tangent to a circle --/
  tangent_to_circle : ∃ (r : ℝ), r > 0 ∧ ∀ i, ∃ (x : ℝ), 0 < x ∧ x < sides i ∧
    x + (sides ((i + 1) % 5) - x) = sides i

/-- The theorem to be proved --/
theorem tangent_pentagon_division (p : TangentPentagon) :
  ∃ (x : ℝ), x = 3/2 ∧ p.sides 0 - x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_pentagon_division_l1981_198196


namespace NUMINAMATH_CALUDE_domain_of_y_l1981_198186

-- Define the function f with domain [0,4]
def f : Set ℝ := Set.Icc 0 4

-- Define the function y
def y (f : Set ℝ) (x : ℝ) : Prop :=
  (x + 3 ∈ f) ∧ (x^2 ∈ f)

-- Theorem statement
theorem domain_of_y (f : Set ℝ) :
  f = Set.Icc 0 4 →
  {x : ℝ | y f x} = Set.Icc (-2) 1 := by
sorry

end NUMINAMATH_CALUDE_domain_of_y_l1981_198186


namespace NUMINAMATH_CALUDE_cos_leq_half_range_l1981_198176

theorem cos_leq_half_range (x : Real) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.cos x ≤ 1/2 ↔ x ∈ Set.Icc (Real.pi/3) (5*Real.pi/3)) :=
by sorry

end NUMINAMATH_CALUDE_cos_leq_half_range_l1981_198176


namespace NUMINAMATH_CALUDE_batch_size_calculation_l1981_198131

theorem batch_size_calculation (N : ℕ) (sample_size : ℕ) (prob : ℚ) 
  (h1 : sample_size = 30)
  (h2 : prob = 1/4)
  (h3 : (sample_size : ℚ) / N = prob) : 
  N = 120 := by
  sorry

end NUMINAMATH_CALUDE_batch_size_calculation_l1981_198131


namespace NUMINAMATH_CALUDE_multiplication_equation_l1981_198108

theorem multiplication_equation (m : ℕ) : 72519 * m = 724827405 → m = 9999 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_l1981_198108


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1981_198102

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 - 8*m + 5 = 0) → 
  (n^2 - 8*n + 5 = 0) → 
  (1 / (m - 1) + 1 / (n - 1) = -3) := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1981_198102


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_eq_1_l1981_198165

theorem sin_50_plus_sqrt3_tan_10_eq_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_eq_1_l1981_198165


namespace NUMINAMATH_CALUDE_line_l_equation_l1981_198163

-- Define the intersection point of the two given lines
def intersection_point : ℝ × ℝ := (2, 1)

-- Define point A
def point_A : ℝ × ℝ := (5, 0)

-- Define the distance from point A to line l
def distance_to_l : ℝ := 3

-- Define the two possible equations for line l
def line_eq1 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line_eq2 (x : ℝ) : Prop := x = 2

-- Theorem statement
theorem line_l_equation : 
  ∃ (l : ℝ → ℝ → Prop), 
    (∀ x y, l x y ↔ (line_eq1 x y ∨ line_eq2 x)) ∧
    (l (intersection_point.1) (intersection_point.2)) ∧
    (∀ x y, l x y → 
      (|4 * point_A.1 - 3 * point_A.2 - 5| / Real.sqrt (4^2 + 3^2) = distance_to_l ∨
       |point_A.1 - 2| = distance_to_l)) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_l1981_198163


namespace NUMINAMATH_CALUDE_intersection_empty_iff_t_leq_neg_one_l1981_198106

-- Define sets A and B
def A : Set ℝ := {x | |x - 2| ≤ 3}
def B (t : ℝ) : Set ℝ := {x | x < t}

-- State the theorem
theorem intersection_empty_iff_t_leq_neg_one (t : ℝ) :
  A ∩ B t = ∅ ↔ t ≤ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_t_leq_neg_one_l1981_198106


namespace NUMINAMATH_CALUDE_min_value_theorem_l1981_198193

theorem min_value_theorem (a b : ℝ) (hb : b > 0) (h : a + 2*b = 1) :
  (3/b) + (1/a) ≥ 7 + 2*Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1981_198193


namespace NUMINAMATH_CALUDE_waiter_tips_l1981_198159

/-- Calculates the total tips earned by a waiter --/
def total_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Proves that the waiter earned $15 in tips --/
theorem waiter_tips : total_tips 10 5 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_l1981_198159


namespace NUMINAMATH_CALUDE_division_problem_l1981_198192

theorem division_problem (L S Q : ℕ) : 
  L - S = 2500 → 
  L = 2982 → 
  L = Q * S + 15 → 
  Q = 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1981_198192


namespace NUMINAMATH_CALUDE_smallest_m_for_tax_price_l1981_198103

theorem smallest_m_for_tax_price : ∃ (x : ℕ), x > 0 ∧ x + (6 * x) / 100 = 2 * 53 * 100 ∧
  ∀ (m : ℕ) (y : ℕ), m > 0 ∧ m < 53 → y > 0 → y + (6 * y) / 100 ≠ 2 * m * 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_for_tax_price_l1981_198103


namespace NUMINAMATH_CALUDE_prime_extension_l1981_198122

theorem prime_extension (n : ℕ+) (h : ∀ k : ℕ, 0 ≤ k ∧ k < Real.sqrt ((n + 2) / 3) → Nat.Prime (k^2 + k + n + 2)) :
  ∀ k : ℕ, Real.sqrt ((n + 2) / 3) ≤ k ∧ k ≤ n → Nat.Prime (k^2 + k + n + 2) := by
  sorry

end NUMINAMATH_CALUDE_prime_extension_l1981_198122


namespace NUMINAMATH_CALUDE_expression_value_l1981_198140

theorem expression_value (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : c - d = -3) : 
  (b - c) - (-d - a) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1981_198140


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_n_l1981_198112

/-- Represents a number in base 5 -/
def BaseNumber (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The number 201032021 in base 5 -/
def n : Nat := BaseNumber [1, 2, 0, 2, 3, 0, 1, 0, 2]

/-- 31 is a prime number -/
axiom thirty_one_prime : Prime 31

/-- 31 divides n -/
axiom thirty_one_divides_n : 31 ∣ n

theorem largest_prime_divisor_of_n :
  ∀ p : Nat, Prime p → p ∣ n → p ≤ 31 := by
  sorry

#check largest_prime_divisor_of_n

end NUMINAMATH_CALUDE_largest_prime_divisor_of_n_l1981_198112


namespace NUMINAMATH_CALUDE_sequence_convergence_l1981_198191

def sequence_property (r : ℝ) (a : ℕ → ℤ) : Prop :=
  r ≥ 0 ∧ ∀ n, a n ≤ a (n + 2) ∧ (a (n + 2) : ℝ)^2 ≤ (a n : ℝ)^2 + r * (a (n + 1) : ℝ)

theorem sequence_convergence (r : ℝ) (a : ℕ → ℤ) (h : sequence_property r a) :
  (r ≤ 2 → ∃ N, ∀ n ≥ N, a (n + 2) = a n) ∧
  (r > 2 → ∃ a : ℕ → ℤ, sequence_property r a ∧ ∀ N, ∃ n ≥ N, a (n + 2) ≠ a n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_convergence_l1981_198191


namespace NUMINAMATH_CALUDE_mark_balloon_cost_l1981_198125

/-- Represents a bag of water balloons -/
structure BalloonBag where
  price : ℕ
  quantity : ℕ

/-- The available bag sizes -/
def availableBags : List BalloonBag := [
  { price := 4, quantity := 50 },
  { price := 6, quantity := 75 },
  { price := 12, quantity := 200 }
]

/-- The total number of balloons Mark wants to buy -/
def targetBalloons : ℕ := 400

/-- Calculates the minimum cost to buy the target number of balloons -/
def minCost (bags : List BalloonBag) (target : ℕ) : ℕ :=
  sorry

theorem mark_balloon_cost :
  minCost availableBags targetBalloons = 24 :=
sorry

end NUMINAMATH_CALUDE_mark_balloon_cost_l1981_198125


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1981_198173

theorem min_value_quadratic_form (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) :
  |5 * x^2 + 11 * x * y - 5 * y^2| ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1981_198173


namespace NUMINAMATH_CALUDE_albert_number_puzzle_l1981_198100

theorem albert_number_puzzle (n : ℕ) : 
  (1 : ℚ) / n + (1 : ℚ) / 2 = (1 : ℚ) / 3 + (2 : ℚ) / (n + 1) ↔ n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_albert_number_puzzle_l1981_198100


namespace NUMINAMATH_CALUDE_set_equality_invariant_under_variable_renaming_l1981_198179

theorem set_equality_invariant_under_variable_renaming :
  {x : ℝ | x ≤ 1} = {t : ℝ | t ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_invariant_under_variable_renaming_l1981_198179


namespace NUMINAMATH_CALUDE_spare_time_is_five_hours_l1981_198123

/-- Calculates the spare time for painting a room given the following conditions:
  * The room has 5 walls
  * Each wall is 2 meters by 3 meters
  * The painter can paint 1 square meter every 10 minutes
  * The painter has 10 hours to paint everything
-/
def spare_time_for_painting : ℕ :=
  let num_walls : ℕ := 5
  let wall_width : ℕ := 2
  let wall_height : ℕ := 3
  let painting_rate : ℕ := 10  -- minutes per square meter
  let total_time : ℕ := 10 * 60  -- total time in minutes

  let wall_area : ℕ := wall_width * wall_height
  let total_area : ℕ := num_walls * wall_area
  let painting_time : ℕ := total_area * painting_rate
  let spare_time_minutes : ℕ := total_time - painting_time
  spare_time_minutes / 60

theorem spare_time_is_five_hours : spare_time_for_painting = 5 := by
  sorry

end NUMINAMATH_CALUDE_spare_time_is_five_hours_l1981_198123


namespace NUMINAMATH_CALUDE_missing_number_proof_l1981_198149

def set1_sum (x y : ℝ) : ℝ := x + 50 + 78 + 104 + y
def set2_sum (x : ℝ) : ℝ := 48 + 62 + 98 + 124 + x

theorem missing_number_proof (x y : ℝ) :
  set1_sum x y / 5 = 62 ∧ set2_sum x / 5 = 76.4 → y = 28 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1981_198149


namespace NUMINAMATH_CALUDE_max_value_product_ratios_l1981_198143

/-- Line l in Cartesian coordinates -/
def line_l (y : ℝ) : Prop := y = 8

/-- Circle C in parametric form -/
def circle_C (x y φ : ℝ) : Prop := x = 2 + 2 * Real.cos φ ∧ y = 2 * Real.sin φ

/-- Ray OM in polar coordinates -/
def ray_OM (θ α : ℝ) : Prop := θ = α ∧ 0 < α ∧ α < Real.pi / 2

/-- Ray ON in polar coordinates -/
def ray_ON (θ α : ℝ) : Prop := θ = α - Real.pi / 2

/-- Theorem stating the maximum value of the product of ratios -/
theorem max_value_product_ratios (α : ℝ) 
  (h_ray_OM : ray_OM α α) 
  (h_ray_ON : ray_ON (α - Real.pi / 2) α) : 
  ∃ (OP OM OQ ON : ℝ), 
    (OP / OM) * (OQ / ON) ≤ 1 / 16 ∧ 
    ∃ (α_max : ℝ), (OP / OM) * (OQ / ON) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_product_ratios_l1981_198143


namespace NUMINAMATH_CALUDE_solve_system_l1981_198154

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20)
  (eq2 : 6 * p + 5 * q = 29) :
  q = -25 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1981_198154


namespace NUMINAMATH_CALUDE_translate_linear_function_l1981_198114

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Translates a linear function vertically by a given amount -/
def translateVertically (f : LinearFunction) (dy : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + dy }

/-- The theorem stating that translating y = -2x + 1 up 4 units results in y = -2x + 5 -/
theorem translate_linear_function :
  let f : LinearFunction := { m := -2, b := 1 }
  let g : LinearFunction := translateVertically f 4
  g.m = -2 ∧ g.b = 5 := by sorry

end NUMINAMATH_CALUDE_translate_linear_function_l1981_198114


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1981_198126

def A : Set ℝ := {x | Real.tan x > Real.sqrt 3}
def B : Set ℝ := {x | x^2 - 4 < 0}

theorem intersection_of_A_and_B : 
  A ∩ B = Set.Ioo (-2) (-Real.pi/2) ∪ Set.Ioo (Real.pi/3) (Real.pi/2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1981_198126


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l1981_198172

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + 2*a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + 2*a = 0) :
  a + b ≥ 2 * Real.sqrt 2 + 4/3 * Real.sqrt (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l1981_198172


namespace NUMINAMATH_CALUDE_apple_division_l1981_198155

theorem apple_division (total_apples : ℕ) (total_weight : ℚ) (portions : ℕ) 
  (h1 : total_apples = 28)
  (h2 : total_weight = 3)
  (h3 : portions = 7) :
  (1 : ℚ) / portions = 1 / 7 ∧ total_weight / portions = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_apple_division_l1981_198155


namespace NUMINAMATH_CALUDE_circle_c_and_line_theorem_l1981_198145

/-- Circle C with given properties -/
structure CircleC where
  radius : ℝ
  center : ℝ × ℝ
  chord_length : ℝ
  center_below_x_axis : center.2 < 0
  center_on_y_eq_x : center.1 = center.2
  radius_eq_3 : radius = 3
  chord_eq_2root5 : chord_length = 2 * Real.sqrt 5

/-- Line with slope 1 -/
structure Line where
  b : ℝ
  equation : ℝ → ℝ
  slope_eq_1 : ∀ x, equation x = x + b

/-- Theorem about CircleC and related Line -/
theorem circle_c_and_line_theorem (c : CircleC) :
  (∃ x y, (x + 2)^2 + (y + 2)^2 = 9 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
  (∃ l : Line, (l.b = 1 ∨ l.b = -1) ∧
    ∃ x₁ y₁ x₂ y₂, ((x₁ - c.center.1)^2 + (y₁ - c.center.2)^2 = c.radius^2 ∧
                    (x₂ - c.center.1)^2 + (y₂ - c.center.2)^2 = c.radius^2 ∧
                    y₁ = l.equation x₁ ∧ y₂ = l.equation x₂ ∧
                    x₁ * x₂ + y₁ * y₂ = 0)) :=
sorry

end NUMINAMATH_CALUDE_circle_c_and_line_theorem_l1981_198145


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1981_198148

/-- A hyperbola with foci on the x-axis, real axis length 4√2, and eccentricity √5/2 -/
structure Hyperbola where
  /-- Real axis length -/
  real_axis_length : ℝ
  real_axis_length_eq : real_axis_length = 4 * Real.sqrt 2
  /-- Eccentricity -/
  e : ℝ
  e_eq : e = Real.sqrt 5 / 2

/-- Standard form of the hyperbola equation -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 2 = 1

/-- Equation of the trajectory of point Q -/
def trajectory_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 4 = 1 ∧ x ≠ 2 * Real.sqrt 2 ∧ x ≠ -2 * Real.sqrt 2

theorem hyperbola_properties (h : Hyperbola) :
  (∀ x y, standard_equation h x y ↔ 
    x^2 / (2 * Real.sqrt 2)^2 - y^2 / ((Real.sqrt 5 / 2) * 2 * Real.sqrt 2)^2 = 1) ∧
  (∀ x y, trajectory_equation h x y ↔
    x^2 / 8 - y^2 / 4 = 1 ∧ x ≠ 2 * Real.sqrt 2 ∧ x ≠ -2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1981_198148


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1981_198111

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Represents a line -/
structure Line where
  k : ℝ

/-- The area of a triangle formed by two points on an ellipse and a fixed point -/
def triangleArea (e : Ellipse) (l : Line) (A : Point) : ℝ := sorry

/-- The main theorem -/
theorem ellipse_intersection_theorem (e : Ellipse) (l : Line) (A : Point) :
  e.a = 2 ∧ e.b = Real.sqrt 2 ∧ 
  A.x = 2 ∧ A.y = 0 ∧
  triangleArea e l A = Real.sqrt 10 / 3 →
  l.k = 1 ∨ l.k = -1 := by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1981_198111


namespace NUMINAMATH_CALUDE_bubble_sort_probability_correct_l1981_198139

/-- The probability of the 25th element in a random permutation of 50 distinct real numbers
    ending up in the 40th position after one bubble pass -/
def bubble_sort_probability : ℚ :=
  1 / 1640

/-- The sequence length -/
def n : ℕ := 50

/-- The initial position of the element we're tracking -/
def initial_position : ℕ := 25

/-- The final position of the element we're tracking -/
def final_position : ℕ := 40

theorem bubble_sort_probability_correct :
  bubble_sort_probability = 1 / 1640 ∧ n = 50 ∧ initial_position = 25 ∧ final_position = 40 := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_probability_correct_l1981_198139


namespace NUMINAMATH_CALUDE_y₁_gt_y₂_l1981_198138

/-- A linear function y = -2x + 3 --/
def f (x : ℝ) : ℝ := -2 * x + 3

/-- Point P₁ on the graph of f --/
def P₁ : ℝ × ℝ := (-2, f (-2))

/-- Point P₂ on the graph of f --/
def P₂ : ℝ × ℝ := (3, f 3)

/-- The y-coordinate of P₁ --/
def y₁ : ℝ := P₁.2

/-- The y-coordinate of P₂ --/
def y₂ : ℝ := P₂.2

theorem y₁_gt_y₂ : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_gt_y₂_l1981_198138


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l1981_198177

/-- Calculates the remaining volume of a cube after removing a cylindrical section. -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  cylinder_height = 6 →
  cube_side^3 - π * cylinder_radius^2 * cylinder_height = 216 - 54 * π :=
by
  sorry

#check remaining_cube_volume

end NUMINAMATH_CALUDE_remaining_cube_volume_l1981_198177


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1981_198189

/-- Calculate the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  let bottom_area := length * width
  let long_sides_area := 2 * length * depth
  let short_sides_area := 2 * width * depth
  bottom_area + long_sides_area + short_sides_area

/-- Theorem stating that the total wet surface area of the given cistern is 83 square meters -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 7 4 1.25 = 83 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1981_198189


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l1981_198104

theorem cube_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 8)
  (h2 : a * b + a * c + b * c = 9)
  (h3 : a * b * c = -18) :
  a^3 + b^3 + c^3 = 242 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l1981_198104


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1981_198107

/-- The eccentricity of the ellipse x²/25 + y²/16 = 1 is 3/5 -/
theorem ellipse_eccentricity :
  let e : ℝ := Real.sqrt (1 - 16 / 25)
  e = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1981_198107


namespace NUMINAMATH_CALUDE_subset_implies_m_value_l1981_198118

def A (m : ℝ) : Set ℝ := {1, 3, 2*m+3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_value (m : ℝ) : B m ⊆ A m → m = 1 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_value_l1981_198118


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1981_198171

/-- A rectangle with a circle inscribed such that the circle is tangent to three sides of the rectangle and its center lies on a diagonal of the rectangle. -/
structure InscribedCircleRectangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The width of the rectangle -/
  w : ℝ
  /-- The height of the rectangle -/
  h : ℝ
  /-- The circle is tangent to three sides of the rectangle -/
  tangent_to_sides : w = 2 * r ∧ h = r
  /-- The center of the circle lies on a diagonal of the rectangle -/
  center_on_diagonal : True

/-- The area of a rectangle with an inscribed circle as described is equal to 2r^2 -/
theorem inscribed_circle_rectangle_area (rect : InscribedCircleRectangle) :
  rect.w * rect.h = 2 * rect.r^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1981_198171


namespace NUMINAMATH_CALUDE_incorrect_transformation_l1981_198175

theorem incorrect_transformation (a b : ℝ) :
  ¬(∀ a b : ℝ, a = b → a / b = 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_transformation_l1981_198175


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_numbers_l1981_198101

theorem sum_of_five_consecutive_even_numbers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_numbers_l1981_198101


namespace NUMINAMATH_CALUDE_twenty_four_is_seventy_five_percent_of_thirty_two_l1981_198132

theorem twenty_four_is_seventy_five_percent_of_thirty_two (x : ℝ) :
  24 / x = 75 / 100 → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_is_seventy_five_percent_of_thirty_two_l1981_198132


namespace NUMINAMATH_CALUDE_percentage_of_returned_books_l1981_198151

/-- Given a library's special collection with initial and final book counts,
    and the number of books loaned out, prove the percentage of returned books. -/
theorem percentage_of_returned_books
  (initial_books : ℕ)
  (final_books : ℕ)
  (loaned_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : final_books = 69)
  (h3 : loaned_books = 30) :
  (initial_books - final_books : ℚ) / loaned_books * 100 = 20 := by
  sorry

#check percentage_of_returned_books

end NUMINAMATH_CALUDE_percentage_of_returned_books_l1981_198151


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1981_198157

theorem triangle_angle_proof (a b c A B C : Real) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧
  b = a * (Real.sin C + Real.cos C) →
  A = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1981_198157


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1981_198120

/-- Given a geometric sequence where:
    - The first term is 4
    - The second term is 12y
    - The third term is 36y^3
    Prove that the fourth term is 108y^5 -/
theorem fourth_term_of_geometric_sequence (y : ℝ) :
  let a₁ : ℝ := 4
  let a₂ : ℝ := 12 * y
  let a₃ : ℝ := 36 * y^3
  let a₄ : ℝ := 108 * y^5
  (∃ (r : ℝ), a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1981_198120


namespace NUMINAMATH_CALUDE_optimal_strategy_l1981_198158

-- Define the set of available numbers
def availableNumbers : Finset Nat := Finset.range 17

-- Define the rules of the game
def isValidChoice (chosen : Finset Nat) (n : Nat) : Bool :=
  n ∈ availableNumbers ∧
  n ∉ chosen ∧
  ¬(∃m ∈ chosen, n = 2 * m ∨ 2 * n = m)

-- Define the state after Player A's move
def initialState : Finset Nat := {8}

-- Define Player B's optimal move
def optimalMove : Nat := 6

-- Theorem to prove
theorem optimal_strategy :
  isValidChoice initialState optimalMove ∧
  ∀ n : Nat, n ≠ optimalMove → 
    (isValidChoice initialState n → 
      ∃ m : Nat, isValidChoice (insert n initialState) m) →
    ¬(∃ m : Nat, isValidChoice (insert optimalMove initialState) m) :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_l1981_198158


namespace NUMINAMATH_CALUDE_current_speed_calculation_l1981_198121

theorem current_speed_calculation (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ)
  (h1 : boat_speed = 16)
  (h2 : upstream_time = 20 / 60)
  (h3 : downstream_time = 15 / 60) :
  ∃ (current_speed : ℝ),
    (boat_speed - current_speed) * upstream_time = (boat_speed + current_speed) * downstream_time ∧
    current_speed = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_calculation_l1981_198121


namespace NUMINAMATH_CALUDE_chest_contents_l1981_198130

/-- Represents the types of coins that can be in a chest -/
inductive CoinType
  | Gold
  | Silver
  | Copper

/-- Represents a chest with its inscription and actual content -/
structure Chest where
  inscription : CoinType → Prop
  content : CoinType

/-- The problem setup -/
def chestProblem (c1 c2 c3 : Chest) : Prop :=
  -- Inscriptions
  c1.inscription = fun t => t = CoinType.Gold ∧
  c2.inscription = fun t => t = CoinType.Silver ∧
  c3.inscription = fun t => t = CoinType.Gold ∨ t = CoinType.Silver ∧
  -- All inscriptions are incorrect
  ¬c1.inscription c1.content ∧
  ¬c2.inscription c2.content ∧
  ¬c3.inscription c3.content ∧
  -- One of each type of coin
  c1.content ≠ c2.content ∧
  c2.content ≠ c3.content ∧
  c3.content ≠ c1.content

/-- The theorem to prove -/
theorem chest_contents (c1 c2 c3 : Chest) :
  chestProblem c1 c2 c3 →
  c1.content = CoinType.Silver ∧
  c2.content = CoinType.Gold ∧
  c3.content = CoinType.Copper :=
by
  sorry

end NUMINAMATH_CALUDE_chest_contents_l1981_198130


namespace NUMINAMATH_CALUDE_unique_solution_mn_l1981_198135

theorem unique_solution_mn : 
  ∃! (m n : ℕ+), 18 * (m : ℝ) * (n : ℝ) = 73 - 9 * (m : ℝ) - 3 * (n : ℝ) ∧ m = 4 ∧ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l1981_198135


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1981_198190

theorem complex_equation_solution (x : ℝ) : 
  (Complex.I * (x + Complex.I) : ℂ) = -1 + 2 * Complex.I → x = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1981_198190


namespace NUMINAMATH_CALUDE_tenth_order_magic_constant_l1981_198199

/-- The magic constant of an nth-order magic square -/
def magic_constant (n : ℕ) : ℕ :=
  (n * (n^2 + 1)) / 2

/-- Theorem: The magic constant of a 10th-order magic square is 505 -/
theorem tenth_order_magic_constant :
  magic_constant 10 = 505 := by
  sorry

#eval magic_constant 10  -- This will evaluate to 505

end NUMINAMATH_CALUDE_tenth_order_magic_constant_l1981_198199


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1981_198136

/-- Given a sphere with surface area 400π cm², prove its volume is (4000/3)π cm³ -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
  (4 * π * r^2 = 400 * π) →  -- Surface area formula
  ((4 / 3) * π * r^3 = (4000 / 3) * π) -- Volume formula
  := by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1981_198136


namespace NUMINAMATH_CALUDE_existsShapeWithCircularTopView_multipleShapesWithCircularTopView_l1981_198161

-- Define a type for 3D shapes
inductive Shape3D
  | Sphere
  | Cylinder
  | Cone
  | Frustum
  | Other

-- Define a property for having a circular top view
def hasCircularTopView (s : Shape3D) : Prop :=
  match s with
  | Shape3D.Sphere => True
  | Shape3D.Cylinder => True
  | Shape3D.Cone => True
  | Shape3D.Frustum => True
  | Shape3D.Other => False

-- Theorem stating that there exist shapes with circular top views
theorem existsShapeWithCircularTopView : ∃ (s : Shape3D), hasCircularTopView s :=
  sorry

-- Theorem stating that multiple shapes can have circular top views
theorem multipleShapesWithCircularTopView :
  ∃ (s1 s2 : Shape3D), s1 ≠ s2 ∧ hasCircularTopView s1 ∧ hasCircularTopView s2 :=
  sorry

end NUMINAMATH_CALUDE_existsShapeWithCircularTopView_multipleShapesWithCircularTopView_l1981_198161


namespace NUMINAMATH_CALUDE_air_quality_probability_l1981_198133

theorem air_quality_probability (p_good : ℝ) (p_consecutive : ℝ) :
  p_good = 0.8 →
  p_consecutive = 0.6 →
  p_good * (p_consecutive / p_good) = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_air_quality_probability_l1981_198133


namespace NUMINAMATH_CALUDE_equation_roots_l1981_198144

-- Define the equation
def equation (x : ℝ) : Prop :=
  (3*x^2 + 1)/(x-2) - (3*x+8)/4 + (5-9*x)/(x-2) + 2 = 0

-- Define the roots
def root1 : ℝ := 3.29
def root2 : ℝ := -0.40

-- Theorem statement
theorem equation_roots :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (∀ (x : ℝ), equation x → (|x - root1| < ε ∨ |x - root2| < ε)) :=
sorry

end NUMINAMATH_CALUDE_equation_roots_l1981_198144


namespace NUMINAMATH_CALUDE_sum_always_positive_l1981_198178

-- Define a monotonically increasing odd function on ℝ
def MonoIncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Theorem statement
theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (hf : MonoIncreasingOddFunction f)
  (ha : ArithmeticSequence a)
  (ha3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l1981_198178


namespace NUMINAMATH_CALUDE_bisecting_line_sum_l1981_198180

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(9, 0) -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The triangle PQR with given coordinates -/
def trianglePQR : Triangle :=
  { P := (0, 10)
    Q := (3, 0)
    R := (9, 0) }

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The line that bisects the area of triangle PQR and passes through Q -/
def bisectingLine (t : Triangle) : Line :=
  sorry

/-- Theorem: The sum of the slope and y-intercept of the bisecting line is -20/3 -/
theorem bisecting_line_sum (t : Triangle) (h : t = trianglePQR) :
    (bisectingLine t).slope + (bisectingLine t).yIntercept = -20/3 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_l1981_198180


namespace NUMINAMATH_CALUDE_driver_weekly_distance_l1981_198117

/-- Represents the driving schedule for a city bus driver --/
structure DrivingSchedule where
  mwf_hours : ℝ
  mwf_speed : ℝ
  tue_hours : ℝ
  tue_speed : ℝ
  thu_hours : ℝ
  thu_speed : ℝ

/-- Calculates the total distance traveled by the driver in a week --/
def totalDistanceTraveled (schedule : DrivingSchedule) : ℝ :=
  3 * (schedule.mwf_hours * schedule.mwf_speed) +
  schedule.tue_hours * schedule.tue_speed +
  schedule.thu_hours * schedule.thu_speed

/-- Theorem stating that the driver travels 148 kilometers in a week --/
theorem driver_weekly_distance (schedule : DrivingSchedule)
  (h1 : schedule.mwf_hours = 3)
  (h2 : schedule.mwf_speed = 12)
  (h3 : schedule.tue_hours = 2.5)
  (h4 : schedule.tue_speed = 9)
  (h5 : schedule.thu_hours = 2.5)
  (h6 : schedule.thu_speed = 7) :
  totalDistanceTraveled schedule = 148 := by
  sorry

#eval totalDistanceTraveled {
  mwf_hours := 3,
  mwf_speed := 12,
  tue_hours := 2.5,
  tue_speed := 9,
  thu_hours := 2.5,
  thu_speed := 7
}

end NUMINAMATH_CALUDE_driver_weekly_distance_l1981_198117


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1981_198170

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 3a cos C = 2c cos A and tan A = 1/3, then angle B measures 135°. -/
theorem triangle_angle_measure (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- angles are positive
  A + B + C = π → -- sum of angles in a triangle
  3 * a * Real.cos C = 2 * c * Real.cos A →
  Real.tan A = 1 / 3 →
  B = π / 4 * 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1981_198170
