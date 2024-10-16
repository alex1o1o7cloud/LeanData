import Mathlib

namespace NUMINAMATH_CALUDE_nursery_paintable_area_l3461_346107

/-- Calculates the total paintable wall area for three identical rooms -/
def totalPaintableArea (length width height : ℝ) (unpaintableArea : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let paintableAreaPerRoom := wallArea - unpaintableArea
  3 * paintableAreaPerRoom

/-- Theorem stating that the total paintable area for three rooms with given dimensions is 1200 sq ft -/
theorem nursery_paintable_area :
  totalPaintableArea 14 11 9 50 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_nursery_paintable_area_l3461_346107


namespace NUMINAMATH_CALUDE_restaurant_bill_with_discounts_l3461_346127

theorem restaurant_bill_with_discounts
  (bob_bill : ℝ) (kate_bill : ℝ) (bob_discount_rate : ℝ) (kate_discount_rate : ℝ)
  (h_bob_bill : bob_bill = 30)
  (h_kate_bill : kate_bill = 25)
  (h_bob_discount : bob_discount_rate = 0.05)
  (h_kate_discount : kate_discount_rate = 0.02) :
  bob_bill * (1 - bob_discount_rate) + kate_bill * (1 - kate_discount_rate) = 53 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_with_discounts_l3461_346127


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l3461_346104

theorem multiple_with_binary_digits (n : ℤ) : ∃ k : ℤ,
  (∃ m : ℤ, k = n * m) ∧ 
  (∃ d : ℕ, d ≤ n ∧ k < 10^d) ∧
  (∀ i : ℕ, i < n → (k / 10^i) % 10 = 0 ∨ (k / 10^i) % 10 = 1) :=
sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l3461_346104


namespace NUMINAMATH_CALUDE_average_mark_first_class_l3461_346146

theorem average_mark_first_class 
  (n1 : ℕ) (n2 : ℕ) (avg2 : ℝ) (avg_total : ℝ)
  (h1 : n1 = 30)
  (h2 : n2 = 50)
  (h3 : avg2 = 80)
  (h4 : avg_total = 65) :
  (n1 + n2) * avg_total = n1 * ((n1 + n2) * avg_total - n2 * avg2) / n1 + n2 * avg2 :=
by sorry

end NUMINAMATH_CALUDE_average_mark_first_class_l3461_346146


namespace NUMINAMATH_CALUDE_data_average_is_four_l3461_346100

def data : List ℝ := [6, 3, 3, 5, 1]

def isMode (x : ℝ) (l : List ℝ) : Prop :=
  ∀ y ∈ l, (l.count x ≥ l.count y)

theorem data_average_is_four (x : ℝ) (h1 : isMode 3 (x::data)) (h2 : isMode 6 (x::data)) :
  (x::data).sum / (x::data).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_data_average_is_four_l3461_346100


namespace NUMINAMATH_CALUDE_tournament_probability_l3461_346149

/-- The number of teams in the tournament -/
def num_teams : ℕ := 30

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams.choose 2

/-- The probability of a team winning any given game -/
def win_probability : ℚ := 1/2

/-- The probability that no two teams win the same number of games -/
noncomputable def unique_wins_probability : ℚ := (num_teams.factorial : ℚ) / 2^total_games

theorem tournament_probability :
  ∃ (m : ℕ), Odd m ∧ unique_wins_probability = 1 / (2^409 * m) :=
sorry

end NUMINAMATH_CALUDE_tournament_probability_l3461_346149


namespace NUMINAMATH_CALUDE_expansion_temperature_difference_l3461_346171

-- Define the initial conditions and coefficients
def initial_length : ℝ := 2
def initial_temp : ℝ := 80
def alpha_iron : ℝ := 0.0000118
def alpha_zinc : ℝ := 0.000031
def length_difference : ℝ := 0.0015

-- Define the function for the length of a rod at temperature x
def rod_length (alpha : ℝ) (x : ℝ) : ℝ :=
  initial_length * (1 + alpha * (x - initial_temp))

-- Define the theorem to prove
theorem expansion_temperature_difference :
  ∃ (x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧
    (rod_length alpha_zinc x₁ - rod_length alpha_iron x₁ = length_difference ∨
     rod_length alpha_iron x₁ - rod_length alpha_zinc x₁ = length_difference) ∧
    (rod_length alpha_zinc x₂ - rod_length alpha_iron x₂ = length_difference ∨
     rod_length alpha_iron x₂ - rod_length alpha_zinc x₂ = length_difference) ∧
    ((x₁ = 119 ∧ x₂ = 41) ∨ (x₁ = 41 ∧ x₂ = 119)) :=
sorry

end NUMINAMATH_CALUDE_expansion_temperature_difference_l3461_346171


namespace NUMINAMATH_CALUDE_triangle_count_in_specific_rectangle_l3461_346102

/-- Represents a rectangle divided by vertical and horizontal lines -/
structure DividedRectangle where
  vertical_divisions : ℕ
  horizontal_divisions : ℕ

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (r : DividedRectangle) : ℕ :=
  let small_rectangles := r.vertical_divisions * r.horizontal_divisions
  let smallest_triangles := small_rectangles * 4
  let isosceles_by_width := small_rectangles
  let large_right_triangles := small_rectangles * 2
  let largest_isosceles := r.horizontal_divisions
  smallest_triangles + isosceles_by_width + large_right_triangles + largest_isosceles

/-- Theorem stating that a rectangle divided by 3 vertical and 2 horizontal lines contains 50 triangles -/
theorem triangle_count_in_specific_rectangle :
  let r : DividedRectangle := ⟨3, 2⟩
  count_triangles r = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_in_specific_rectangle_l3461_346102


namespace NUMINAMATH_CALUDE_tan_one_implies_sin_2a_minus_cos_sq_a_eq_half_l3461_346108

theorem tan_one_implies_sin_2a_minus_cos_sq_a_eq_half (α : Real) 
  (h : Real.tan α = 1) : Real.sin (2 * α) - Real.cos α ^ 2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_one_implies_sin_2a_minus_cos_sq_a_eq_half_l3461_346108


namespace NUMINAMATH_CALUDE_daves_video_games_l3461_346138

theorem daves_video_games (total_games : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) :
  total_games = 10 →
  price_per_game = 4 →
  total_earnings = 32 →
  total_games - (total_earnings / price_per_game) = 2 :=
by sorry

end NUMINAMATH_CALUDE_daves_video_games_l3461_346138


namespace NUMINAMATH_CALUDE_sqrt_square_of_negative_l3461_346129

theorem sqrt_square_of_negative : Real.sqrt ((-2023)^2) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_of_negative_l3461_346129


namespace NUMINAMATH_CALUDE_checkers_rectangle_exists_l3461_346194

/-- Represents the color of a checker -/
inductive Color
| White
| Black

/-- Represents a 3x7 grid of checkers -/
def CheckerGrid := Fin 3 → Fin 7 → Color

/-- Checks if four positions form a rectangle in the grid -/
def IsRectangle (a b c d : Fin 3 × Fin 7) : Prop :=
  (a.1 = b.1 ∧ c.1 = d.1 ∧ a.2 ≠ c.2) ∨
  (a.1 = c.1 ∧ b.1 = d.1 ∧ a.2 ≠ b.2)

/-- The main theorem -/
theorem checkers_rectangle_exists (grid : CheckerGrid) :
  ∃ (color : Color) (a b c d : Fin 3 × Fin 7),
    IsRectangle a b c d ∧
    grid a.1 a.2 = color ∧
    grid b.1 b.2 = color ∧
    grid c.1 c.2 = color ∧
    grid d.1 d.2 = color :=
sorry

end NUMINAMATH_CALUDE_checkers_rectangle_exists_l3461_346194


namespace NUMINAMATH_CALUDE_egyptian_fraction_proof_l3461_346197

theorem egyptian_fraction_proof :
  ∃! (b₂ b₃ b₅ b₆ b₇ b₈ : ℕ),
    (3 : ℚ) / 8 = b₂ / 2 + b₃ / 6 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 ∧
    b₂ < 2 ∧ b₃ < 3 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧ b₈ < 8 ∧
    b₂ + b₃ + b₅ + b₆ + b₇ + b₈ = 12 :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fraction_proof_l3461_346197


namespace NUMINAMATH_CALUDE_max_a_value_l3461_346191

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, x * Real.log x - (1 + a) * x + 1 ≥ 0) →
  a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3461_346191


namespace NUMINAMATH_CALUDE_class_composition_l3461_346196

theorem class_composition (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h1 : total_students = 56)
  (h2 : boy_ratio = 3)
  (h3 : girl_ratio = 4) :
  (boy_ratio : ℚ) / (boy_ratio + girl_ratio) * 100 = 42.86 ∧ 
  (girl_ratio * total_students) / (boy_ratio + girl_ratio) = 32 := by
sorry

end NUMINAMATH_CALUDE_class_composition_l3461_346196


namespace NUMINAMATH_CALUDE_no_integer_roots_l3461_346140

theorem no_integer_roots : ¬ ∃ (x : ℤ), x^3 - 4*x^2 - 14*x + 28 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3461_346140


namespace NUMINAMATH_CALUDE_trisha_remaining_money_l3461_346132

/-- Calculates the remaining money after shopping given the initial amount and expenses. -/
def remaining_money (initial : ℕ) (meat chicken veggies eggs dog_food : ℕ) : ℕ :=
  initial - (meat + chicken + veggies + eggs + dog_food)

/-- Proves that Trisha's remaining money after shopping is $35. -/
theorem trisha_remaining_money :
  remaining_money 167 17 22 43 5 45 = 35 := by
  sorry

end NUMINAMATH_CALUDE_trisha_remaining_money_l3461_346132


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3461_346198

theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  (∀ k, k ≥ 1 → S k = 3^k + r) →
  (∀ k, k ≥ 2 → a k = S k - S (k-1)) →
  (∀ k, k ≥ 2 → a k = 2 * 3^(k-1)) →
  a 1 = S 1 →
  r = -1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3461_346198


namespace NUMINAMATH_CALUDE_lemonade_amount_l3461_346124

/-- Represents the components of lemonade -/
structure LemonadeComponents where
  water : ℝ
  syrup : ℝ
  lemon_juice : ℝ

/-- Calculates the total amount of lemonade -/
def total_lemonade (c : LemonadeComponents) : ℝ :=
  c.water + c.syrup + c.lemon_juice

/-- Theorem stating the amount of lemonade made given the conditions -/
theorem lemonade_amount (c : LemonadeComponents) 
  (h1 : c.water = 4 * c.syrup) 
  (h2 : c.syrup = 2 * c.lemon_juice)
  (h3 : c.lemon_juice = 3) : 
  total_lemonade c = 24 := by
  sorry

#check lemonade_amount

end NUMINAMATH_CALUDE_lemonade_amount_l3461_346124


namespace NUMINAMATH_CALUDE_simplify_expression_l3461_346142

theorem simplify_expression : 110^2 - 109 * 111 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3461_346142


namespace NUMINAMATH_CALUDE_number_of_pencils_l3461_346128

theorem number_of_pencils (pens pencils : ℕ) : 
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 6 →
  pencils = 36 := by
sorry

end NUMINAMATH_CALUDE_number_of_pencils_l3461_346128


namespace NUMINAMATH_CALUDE_bipin_twice_chandan_age_l3461_346143

/-- Represents the ages and relationships of Bipin, Alok, and Chandan -/
structure AgeRelationship where
  alok_age : ℕ
  bipin_age : ℕ
  chandan_age : ℕ
  years_passed : ℕ

/-- The conditions given in the problem -/
def initial_conditions : AgeRelationship where
  alok_age := 5
  bipin_age := 6 * 5
  chandan_age := 7 + 3
  years_passed := 0

/-- The relationship between Bipin and Chandan's ages after some years -/
def age_relationship (ar : AgeRelationship) : Prop :=
  ar.bipin_age + ar.years_passed = 2 * (ar.chandan_age + ar.years_passed)

/-- The theorem to be proved -/
theorem bipin_twice_chandan_age :
  ∃ (final : AgeRelationship),
    final.alok_age = initial_conditions.alok_age + final.years_passed ∧
    final.bipin_age = initial_conditions.bipin_age + final.years_passed ∧
    final.chandan_age = initial_conditions.chandan_age + final.years_passed ∧
    age_relationship final ∧
    final.years_passed = 10 :=
  sorry

end NUMINAMATH_CALUDE_bipin_twice_chandan_age_l3461_346143


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_twenty_l3461_346122

theorem log_expression_equals_negative_twenty :
  (Real.log (1/4) - Real.log 25) / (100 ^ (-1/2 : ℝ)) = -20 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_twenty_l3461_346122


namespace NUMINAMATH_CALUDE_correct_assignment_count_l3461_346116

def num_rooms : ℕ := 6
def num_friends : ℕ := 6
def max_occupancy : ℕ := 3
def min_occupancy : ℕ := 1
def num_inseparable_friends : ℕ := 2

-- Function to calculate the number of ways to assign friends to rooms
def assignment_ways : ℕ := sorry

-- Theorem statement
theorem correct_assignment_count :
  assignment_ways = 3600 :=
sorry

end NUMINAMATH_CALUDE_correct_assignment_count_l3461_346116


namespace NUMINAMATH_CALUDE_triangle_equilateral_condition_l3461_346131

/-- If in a triangle ABC, a/cos(A) = b/cos(B) = c/cos(C), then the triangle is equilateral -/
theorem triangle_equilateral_condition (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C) :
  A = B ∧ B = C := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_condition_l3461_346131


namespace NUMINAMATH_CALUDE_total_salaries_proof_l3461_346179

/-- Proves that given the conditions of A and B's salaries and spending,
    their total salaries amount to $5000 -/
theorem total_salaries_proof (A_salary B_salary : ℝ) : 
  A_salary = 3750 →
  A_salary * 0.05 = B_salary * 0.15 →
  A_salary + B_salary = 5000 := by
  sorry

end NUMINAMATH_CALUDE_total_salaries_proof_l3461_346179


namespace NUMINAMATH_CALUDE_jump_frequency_proof_l3461_346117

def jump_data : List Nat := [50, 63, 77, 83, 87, 88, 89, 91, 93, 100, 102, 111, 117, 121, 130, 133, 146, 158, 177, 188]

def in_range (n : Nat) : Bool := 90 ≤ n ∧ n ≤ 110

def count_in_range (data : List Nat) : Nat :=
  data.filter in_range |>.length

theorem jump_frequency_proof :
  (count_in_range jump_data : Rat) / jump_data.length = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_jump_frequency_proof_l3461_346117


namespace NUMINAMATH_CALUDE_village_population_l3461_346119

theorem village_population (final_population : ℕ) : 
  final_population = 4860 → 
  ∃ (original_population : ℕ), 
    (original_population : ℝ) * 0.9 * 0.75 = final_population ∧ 
    original_population = 7200 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3461_346119


namespace NUMINAMATH_CALUDE_soda_discount_percentage_l3461_346181

/-- The discount percentage for soda cans purchased in 24-can cases -/
def discount_percentage (regular_price : ℚ) (discounted_price : ℚ) : ℚ :=
  (1 - discounted_price / (100 * regular_price)) * 100

/-- Theorem stating that the discount percentage is 15% -/
theorem soda_discount_percentage :
  let regular_price : ℚ := 40 / 100  -- $0.40 per can
  let discounted_price : ℚ := 34     -- $34 for 100 cans
  discount_percentage regular_price discounted_price = 15 := by
  sorry

#eval discount_percentage (40/100) 34

end NUMINAMATH_CALUDE_soda_discount_percentage_l3461_346181


namespace NUMINAMATH_CALUDE_march_first_is_wednesday_l3461_346145

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the day of the week for a given number of days before a reference day -/
def daysBefore (referenceDay : DayOfWeek) (days : Nat) : DayOfWeek :=
  sorry

theorem march_first_is_wednesday (march13 : MarchDate) 
  (h : march13.day = 13 ∧ march13.dayOfWeek = DayOfWeek.Monday) :
  ∃ (march1 : MarchDate), march1.day = 1 ∧ march1.dayOfWeek = DayOfWeek.Wednesday :=
  sorry

end NUMINAMATH_CALUDE_march_first_is_wednesday_l3461_346145


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3461_346170

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x > 0}

-- Define set B
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3461_346170


namespace NUMINAMATH_CALUDE_expression_value_l3461_346180

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 5) :
  3 * x - 4 * y + 2 * z = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3461_346180


namespace NUMINAMATH_CALUDE_product_357_sum_28_l3461_346165

theorem product_357_sum_28 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 357 → 
  a + b + c + d = 28 := by
sorry

end NUMINAMATH_CALUDE_product_357_sum_28_l3461_346165


namespace NUMINAMATH_CALUDE_complex_modulus_l3461_346133

theorem complex_modulus (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z = (i + 1) / (1 - i)^2 → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3461_346133


namespace NUMINAMATH_CALUDE_tangent_line_and_inequalities_l3461_346153

noncomputable def f (x : ℝ) := x - x^2 + 3 * Real.log x

theorem tangent_line_and_inequalities :
  (∃ x₀ : ℝ, x₀ > 0 ∧ (∀ x > 0, f x ≤ 2 * x - 2) ∧
   (∀ k < 2, ∃ x₁ > 1, ∀ x ∈ Set.Ioo 1 x₁, f x ≥ k * (x - 1))) ∧
  (∃ a b : ℝ, ∀ x > 0, f x = 2 * x - 2 → x = a ∧ f x = b) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequalities_l3461_346153


namespace NUMINAMATH_CALUDE_correct_number_probability_l3461_346177

def first_four_options : List ℕ := [2960, 2961, 2990, 2991]
def last_three_digits : List ℕ := [6, 7, 8]

def total_possible_numbers : ℕ := (List.length first_four_options) * (Nat.factorial (List.length last_three_digits))

theorem correct_number_probability :
  (1 : ℚ) / total_possible_numbers = 1 / 24 :=
sorry

end NUMINAMATH_CALUDE_correct_number_probability_l3461_346177


namespace NUMINAMATH_CALUDE_equation_solution_l3461_346174

theorem equation_solution :
  ∃! x : ℝ, (1 : ℝ) / (x - 2) = (3 : ℝ) / (x - 5) ∧ x = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3461_346174


namespace NUMINAMATH_CALUDE_inheritance_calculation_l3461_346109

/-- The original inheritance amount -/
def inheritance : ℝ := sorry

/-- The state tax rate -/
def state_tax_rate : ℝ := 0.15

/-- The federal tax rate -/
def federal_tax_rate : ℝ := 0.25

/-- The total tax paid -/
def total_tax_paid : ℝ := 18000

theorem inheritance_calculation : 
  state_tax_rate * inheritance + 
  federal_tax_rate * (1 - state_tax_rate) * inheritance = 
  total_tax_paid := by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l3461_346109


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3461_346172

theorem cubic_equation_solution (x : ℝ) : 
  x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3461_346172


namespace NUMINAMATH_CALUDE_fraction_equivalence_prime_l3461_346139

theorem fraction_equivalence_prime (n : ℕ) : 
  Prime n ∧ (4 + n : ℚ) / (7 + n) = 7 / 8 ↔ n = 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_prime_l3461_346139


namespace NUMINAMATH_CALUDE_touching_balls_in_cylinder_l3461_346193

theorem touching_balls_in_cylinder (a b d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hd : d > 0)
  (h_touch : a + b = d)
  (h_larger_bottom : a ≥ b) : 
  Real.sqrt d = Real.sqrt a + Real.sqrt b := by
sorry

end NUMINAMATH_CALUDE_touching_balls_in_cylinder_l3461_346193


namespace NUMINAMATH_CALUDE_right_pentagonal_pyramid_base_side_length_l3461_346110

/-- Represents a right pyramid with a regular pentagonal base -/
structure RightPentagonalPyramid where
  base_side_length : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- 
Theorem: For a right pyramid with a regular pentagonal base, 
if the area of one lateral face is 120 square meters and the slant height is 40 meters, 
then the length of the side of its base is 6 meters.
-/
theorem right_pentagonal_pyramid_base_side_length 
  (pyramid : RightPentagonalPyramid) 
  (h1 : pyramid.lateral_face_area = 120) 
  (h2 : pyramid.slant_height = 40) : 
  pyramid.base_side_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_pentagonal_pyramid_base_side_length_l3461_346110


namespace NUMINAMATH_CALUDE_stamp_collection_total_l3461_346161

/-- Represents a stamp collection with various categories of stamps. -/
structure StampCollection where
  foreign : ℕ
  old : ℕ
  both_foreign_and_old : ℕ
  neither_foreign_nor_old : ℕ

/-- Calculates the total number of stamps in the collection. -/
def total_stamps (collection : StampCollection) : ℕ :=
  collection.foreign + collection.old - collection.both_foreign_and_old + collection.neither_foreign_nor_old

/-- Theorem stating that the total number of stamps in the given collection is 220. -/
theorem stamp_collection_total :
  ∃ (collection : StampCollection),
    collection.foreign = 90 ∧
    collection.old = 70 ∧
    collection.both_foreign_and_old = 20 ∧
    collection.neither_foreign_nor_old = 60 ∧
    total_stamps collection = 220 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_total_l3461_346161


namespace NUMINAMATH_CALUDE_sum_b_plus_d_l3461_346189

theorem sum_b_plus_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 42) 
  (h2 : a + c = 7) : 
  b + d = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_b_plus_d_l3461_346189


namespace NUMINAMATH_CALUDE_class_size_l3461_346176

theorem class_size (average_weight : ℝ) (teacher_weight : ℝ) (new_average : ℝ) :
  average_weight = 35 →
  teacher_weight = 45 →
  new_average = 35.4 →
  ∃ n : ℕ, (n : ℝ) * average_weight + teacher_weight = new_average * ((n : ℝ) + 1) ∧ n = 24 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l3461_346176


namespace NUMINAMATH_CALUDE_apartment_tax_calculation_l3461_346187

/-- Calculates the tax amount for an apartment --/
def calculate_tax (cadastral_value : ℝ) (tax_rate : ℝ) : ℝ :=
  cadastral_value * tax_rate

/-- Theorem: The tax amount for an apartment with a cadastral value of 3 million rubles
    and a tax rate of 0.1% is equal to 3000 rubles --/
theorem apartment_tax_calculation :
  let cadastral_value : ℝ := 3000000
  let tax_rate : ℝ := 0.001
  calculate_tax cadastral_value tax_rate = 3000 := by
sorry

/-- Additional information about the apartment (not used in the main calculation) --/
def apartment_area : ℝ := 70
def is_only_property : Prop := true

end NUMINAMATH_CALUDE_apartment_tax_calculation_l3461_346187


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3461_346147

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 17 →
  a * b + c + d = 94 →
  a * d + b * c = 195 →
  c * d = 120 →
  a^2 + b^2 + c^2 + d^2 ≤ 918 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3461_346147


namespace NUMINAMATH_CALUDE_f_has_minimum_at_6_l3461_346121

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 32

/-- Theorem stating that f has a minimum at x = 6 -/
theorem f_has_minimum_at_6 : 
  ∀ x : ℝ, f x ≥ f 6 := by sorry

end NUMINAMATH_CALUDE_f_has_minimum_at_6_l3461_346121


namespace NUMINAMATH_CALUDE_cube_volume_doubled_edges_l3461_346155

/-- Given a cube, doubling each edge results in a volume 8 times larger than the original. -/
theorem cube_volume_doubled_edges (a : ℝ) (ha : a > 0) :
  (2 * a)^3 = 8 * a^3 := by sorry

end NUMINAMATH_CALUDE_cube_volume_doubled_edges_l3461_346155


namespace NUMINAMATH_CALUDE_rope_cutting_probability_l3461_346123

theorem rope_cutting_probability (rope_length : ℝ) (min_segment_length : ℝ) : 
  rope_length = 4 →
  min_segment_length = 1.5 →
  (rope_length - 2 * min_segment_length) / rope_length = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_rope_cutting_probability_l3461_346123


namespace NUMINAMATH_CALUDE_photo_archive_album_size_l3461_346115

/-- Represents an album in the photo archive -/
structure Album where
  pages : ℕ
  photos_per_page : ℕ

/-- The photo archive system -/
structure PhotoArchive where
  album : Album
  /-- Ensures all albums are identical -/
  albums_identical : ∀ a b : Album, a = b

theorem photo_archive_album_size 
  (archive : PhotoArchive)
  (h1 : archive.album.photos_per_page = 4)
  (h2 : ∃ x : ℕ, 81 = (x - 1) * (archive.album.pages * archive.album.photos_per_page) + 5 * archive.album.photos_per_page)
  (h3 : ∃ y : ℕ, 171 = (y - 1) * (archive.album.pages * archive.album.photos_per_page) + 3 * archive.album.photos_per_page)
  : archive.album.pages * archive.album.photos_per_page = 32 := by
  sorry

end NUMINAMATH_CALUDE_photo_archive_album_size_l3461_346115


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l3461_346195

theorem division_multiplication_equality : (0.24 / 0.006) * 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l3461_346195


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3461_346137

theorem geometric_sequence_common_ratio 
  (a b : ℝ) 
  (h1 : 2 * a = 1 + b) 
  (h2 : (a + 2)^2 = 3 * (b + 5)) 
  (h3 : a + 2 ≠ 0) 
  (h4 : b + 5 ≠ 0) : 
  (a + 2) / 3 = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3461_346137


namespace NUMINAMATH_CALUDE_least_bench_sections_l3461_346120

theorem least_bench_sections (M : ℕ) : M > 0 ∧ 
  (∀ k : ℕ, k > 0 ∧ k < M → ¬(120 ∣ 8*k ∧ 120 ∣ 12*k ∧ 120 ∣ 10*k)) ∧ 
  (120 ∣ 8*M ∧ 120 ∣ 12*M ∧ 120 ∣ 10*M) → M = 15 := by
  sorry

end NUMINAMATH_CALUDE_least_bench_sections_l3461_346120


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3461_346105

theorem candy_bar_cost (selling_price : ℕ) (bought : ℕ) (sold : ℕ) (profit : ℕ) : 
  selling_price = 100 ∧ bought = 50 ∧ sold = 48 ∧ profit = 800 →
  (selling_price * sold - profit) / bought = 80 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l3461_346105


namespace NUMINAMATH_CALUDE_sequence_property_l3461_346151

/-- An integer sequence satisfying the given conditions -/
def IntegerSequence (a : ℕ → ℤ) : Prop :=
  a 3 = -1 ∧ 
  a 7 = 4 ∧ 
  (∀ n ≤ 6, ∃ d : ℤ, a (n + 1) - a n = d) ∧ 
  (∀ n ≥ 5, ∃ q : ℚ, a (n + 1) = a n * q)

/-- The property that needs to be satisfied for a given m -/
def SatisfiesProperty (a : ℕ → ℤ) (m : ℕ) : Prop :=
  a m + a (m + 1) + a (m + 2) = a m * a (m + 1) * a (m + 2)

/-- The main theorem statement -/
theorem sequence_property (a : ℕ → ℤ) (h : IntegerSequence a) :
  ∀ m : ℕ, m > 0 → (SatisfiesProperty a m ↔ m = 1 ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l3461_346151


namespace NUMINAMATH_CALUDE_new_students_weight_l3461_346160

/-- Given a class of 8 students, prove that when two students weighing 70kg and 80kg 
    are replaced and the average weight decreases by 2 kg, 
    the combined weight of the two new students is 134 kg. -/
theorem new_students_weight (total_weight : ℝ) : 
  (total_weight - 150 + 134) / 8 = total_weight / 8 - 2 := by
  sorry

end NUMINAMATH_CALUDE_new_students_weight_l3461_346160


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3461_346130

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3461_346130


namespace NUMINAMATH_CALUDE_all_functions_are_zero_l3461_346111

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ x y : ℕ, f (x * y) = f x + f y) ∧
  (f 30 = 0) ∧
  (∀ x : ℕ, x % 10 = 7 → f x = 0)

theorem all_functions_are_zero (f : ℕ → ℕ) (h : is_valid_function f) :
  ∀ n : ℕ, f n = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_functions_are_zero_l3461_346111


namespace NUMINAMATH_CALUDE_cube_edge_length_l3461_346166

/-- Given a cube with surface area 18 dm², prove that the length of its edge is √3 dm. -/
theorem cube_edge_length (S : ℝ) (edge : ℝ) (h1 : S = 18) (h2 : S = 6 * edge ^ 2) : 
  edge = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3461_346166


namespace NUMINAMATH_CALUDE_smallest_term_divisible_by_billion_l3461_346157

def geometric_sequence (a₁ : ℚ) (a₂ : ℚ) (n : ℕ) : ℚ :=
  a₁ * (a₂ / a₁) ^ (n - 1)

def is_divisible_by_billion (q : ℚ) : Prop :=
  ∃ k : ℤ, q = k * 10^9

theorem smallest_term_divisible_by_billion :
  let a₁ := 5 / 8
  let a₂ := 50
  (∀ n < 9, ¬ is_divisible_by_billion (geometric_sequence a₁ a₂ n)) ∧
  is_divisible_by_billion (geometric_sequence a₁ a₂ 9) :=
by sorry

end NUMINAMATH_CALUDE_smallest_term_divisible_by_billion_l3461_346157


namespace NUMINAMATH_CALUDE_boat_trip_distance_l3461_346159

/-- Proves that given a boat with speed 8 kmph in standing water, a stream with speed 6 kmph,
    and a round trip time of 120 hours, the distance to the destination is 210 km. -/
theorem boat_trip_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : stream_speed = 6) 
  (h3 : total_time = 120) : 
  (boat_speed + stream_speed) * (boat_speed - stream_speed) * total_time / 
  (2 * (boat_speed + stream_speed) * (boat_speed - stream_speed)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_boat_trip_distance_l3461_346159


namespace NUMINAMATH_CALUDE_frank_money_problem_l3461_346118

theorem frank_money_problem (initial_money : ℝ) : 
  initial_money > 0 →
  let remaining_after_groceries := initial_money - (1/5 * initial_money)
  let remaining_after_magazine := remaining_after_groceries - (1/4 * remaining_after_groceries)
  remaining_after_magazine = 360 →
  initial_money = 600 := by
sorry


end NUMINAMATH_CALUDE_frank_money_problem_l3461_346118


namespace NUMINAMATH_CALUDE_simplify_expression_l3461_346199

theorem simplify_expression (a : ℝ) : 
  3 * a * (a + 1) - (3 + a) * (3 - a) - (2 * a - 1)^2 = 7 * a - 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3461_346199


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_4_range_of_a_for_solutions_f_min_value_f_min_condition_l3461_346136

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for part I
theorem solution_set_f_less_than_4 :
  {x : ℝ | f x < 4} = Set.Ioo (-2) 2 := by sorry

-- Theorem for part II
theorem range_of_a_for_solutions :
  {a : ℝ | ∃ x, f x - |a - 1| < 0} = Set.Iio (-1) ∪ Set.Ioi 3 := by sorry

-- Helper theorem: Minimum value of f is 2
theorem f_min_value :
  ∀ x : ℝ, f x ≥ 2 := by sorry

-- Helper theorem: Condition for f to achieve its minimum value
theorem f_min_condition (x : ℝ) :
  f x = 2 ↔ (x + 1) * (x - 1) ≤ 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_4_range_of_a_for_solutions_f_min_value_f_min_condition_l3461_346136


namespace NUMINAMATH_CALUDE_final_savings_calculation_l3461_346144

def initial_savings : ℕ := 849400
def monthly_income : ℕ := 45000 + 35000 + 7000 + 10000 + 13000
def monthly_expenses : ℕ := 30000 + 10000 + 5000 + 4500 + 9000
def period : ℕ := 5

def total_income : ℕ := period * monthly_income
def total_expenses : ℕ := period * monthly_expenses

theorem final_savings_calculation :
  initial_savings + total_income - total_expenses = 1106900 :=
by sorry

end NUMINAMATH_CALUDE_final_savings_calculation_l3461_346144


namespace NUMINAMATH_CALUDE_min_value_expression_l3461_346182

theorem min_value_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 2)^2 + (z/y - 2)^2 + (5/z - 2)^2 ≥ 4 * (Real.rpow 5 (1/4) - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3461_346182


namespace NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l3461_346164

theorem disjunction_false_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → ¬p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l3461_346164


namespace NUMINAMATH_CALUDE_max_ships_on_battleship_board_l3461_346135

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a ship -/
structure Ship :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a placement of ships on a board -/
def Placement := List (ℕ × ℕ)

/-- Checks if two ships are adjacent or overlapping -/
def are_adjacent_or_overlapping (p1 p2 : ℕ × ℕ) (s : Ship) : Prop := sorry

/-- Checks if a placement is valid (no adjacent or overlapping ships) -/
def is_valid_placement (b : Board) (s : Ship) (p : Placement) : Prop := sorry

/-- The maximum number of ships that can be placed on the board -/
def max_ships (b : Board) (s : Ship) : ℕ := sorry

/-- The main theorem stating the maximum number of 1x4 ships on a 10x10 board -/
theorem max_ships_on_battleship_board :
  let b : Board := ⟨10, 10⟩
  let s : Ship := ⟨4, 1⟩
  max_ships b s = 24 := by sorry

end NUMINAMATH_CALUDE_max_ships_on_battleship_board_l3461_346135


namespace NUMINAMATH_CALUDE_unity_community_club_ratio_l3461_346186

theorem unity_community_club_ratio :
  ∀ (f m c : ℕ),
  f > 0 → m > 0 → c > 0 →
  (35 * f + 30 * m + 10 * c) / (f + m + c) = 25 →
  ∃ (k : ℕ), k > 0 ∧ f = k ∧ m = k ∧ c = k :=
by sorry

end NUMINAMATH_CALUDE_unity_community_club_ratio_l3461_346186


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3461_346141

-- Define the variables and conditions
theorem express_y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) :
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3461_346141


namespace NUMINAMATH_CALUDE_dave_won_three_more_than_jerry_l3461_346156

/-- Shuffleboard game results -/
structure ShuffleboardResults where
  dave_wins : ℕ
  ken_wins : ℕ
  jerry_wins : ℕ
  total_games : ℕ

/-- Conditions for the shuffleboard game results -/
def valid_results (r : ShuffleboardResults) : Prop :=
  r.ken_wins = r.dave_wins + 5 ∧
  r.dave_wins > r.jerry_wins ∧
  r.jerry_wins = 7 ∧
  r.total_games = r.dave_wins + r.ken_wins + r.jerry_wins ∧
  r.total_games = 32

/-- Theorem: Dave won 3 more games than Jerry -/
theorem dave_won_three_more_than_jerry (r : ShuffleboardResults) 
  (h : valid_results r) : r.dave_wins = r.jerry_wins + 3 := by
  sorry


end NUMINAMATH_CALUDE_dave_won_three_more_than_jerry_l3461_346156


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3461_346188

theorem quadratic_equation_roots (a b c : ℤ) : 
  a ≠ 0 → 
  (∃ x : ℚ, a * x^2 + b * x + c = 0) → 
  ¬(Odd a ∧ Odd b ∧ Odd c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3461_346188


namespace NUMINAMATH_CALUDE_frank_and_friends_count_l3461_346150

/-- The number of people, including Frank, who can eat brownies -/
def num_people (columns rows brownies_per_person : ℕ) : ℕ :=
  (columns * rows) / brownies_per_person

theorem frank_and_friends_count :
  num_people 6 3 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_frank_and_friends_count_l3461_346150


namespace NUMINAMATH_CALUDE_train_circuit_time_l3461_346106

/-- Represents the time in seconds -/
def seconds_per_circuit : ℕ := 71

/-- Represents the number of circuits -/
def num_circuits : ℕ := 6

/-- Converts seconds to minutes and remaining seconds -/
def seconds_to_minutes_and_seconds (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / 60, total_seconds % 60)

theorem train_circuit_time : 
  seconds_to_minutes_and_seconds (num_circuits * seconds_per_circuit) = (7, 6) := by
  sorry

end NUMINAMATH_CALUDE_train_circuit_time_l3461_346106


namespace NUMINAMATH_CALUDE_symmetry_implies_periodicity_l3461_346113

/-- A function f: ℝ → ℝ is symmetric with respect to the point (a, y₀) -/
def SymmetricPoint (f : ℝ → ℝ) (a y₀ : ℝ) : Prop :=
  ∀ x, f (a + x) - y₀ = y₀ - f (a - x)

/-- A function f: ℝ → ℝ is symmetric with respect to the line x = b -/
def SymmetricLine (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f (b + x) = f (b - x)

/-- A function f: ℝ → ℝ is periodic with period p -/
def Periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem symmetry_implies_periodicity (f : ℝ → ℝ) (a b y₀ : ℝ) (hb : b > a) 
    (h1 : SymmetricPoint f a y₀) (h2 : SymmetricLine f b) :
    Periodic f (4 * (b - a)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_periodicity_l3461_346113


namespace NUMINAMATH_CALUDE_ln_f_greater_than_one_max_a_value_l3461_346190

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- Part I
theorem ln_f_greater_than_one : ∀ x : ℝ, Real.log (f (-1) x) > 1 := by sorry

-- Part II
theorem max_a_value : 
  (∃ a_max : ℝ, 
    (∀ a : ℝ, (∀ x : ℝ, f a x ≥ a) → a ≤ a_max) ∧
    (∀ x : ℝ, f a_max x ≥ a_max) ∧
    a_max = 1) := by sorry

end NUMINAMATH_CALUDE_ln_f_greater_than_one_max_a_value_l3461_346190


namespace NUMINAMATH_CALUDE_points_collinear_l3461_346148

/-- Given four points P, A, B, C in space, if PC = 1/4 PA + 3/4 PB, then A, B, C are collinear -/
theorem points_collinear (P A B C : EuclideanSpace ℝ (Fin 3)) 
  (h : C - P = (1/4 : ℝ) • (A - P) + (3/4 : ℝ) • (B - P)) : 
  ∃ t : ℝ, C - A = t • (B - C) := by
  sorry

end NUMINAMATH_CALUDE_points_collinear_l3461_346148


namespace NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l3461_346184

theorem geometric_sequence_eighth_term
  (a₁ a₅ : ℚ)
  (h₁ : a₁ = 2187)
  (h₅ : a₅ = 960)
  (h_geom : ∃ r : ℚ, r ≠ 0 ∧ a₅ = a₁ * r^4) :
  ∃ a₈ : ℚ, a₈ = 35651584 / 4782969 ∧ (∃ r : ℚ, r ≠ 0 ∧ a₈ = a₁ * r^7) :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l3461_346184


namespace NUMINAMATH_CALUDE_find_a_value_l3461_346178

theorem find_a_value (x a : ℝ) (h : x = -1 ∧ -2 * (x - a) = 4) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l3461_346178


namespace NUMINAMATH_CALUDE_parallelogram_diagonals_l3461_346167

/-- Represents a parallelogram with given side length and diagonal lengths -/
structure Parallelogram where
  side : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Check if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem to prove -/
theorem parallelogram_diagonals (p : Parallelogram) : 
  p.side = 10 →
  (p.diagonal1 = 20 ∧ p.diagonal2 = 30) ↔
  (canFormTriangle (p.side) (p.diagonal1 / 2) (p.diagonal2 / 2) ∧
   ¬(canFormTriangle p.side 2 3) ∧
   ¬(canFormTriangle p.side 3 4) ∧
   ¬(canFormTriangle p.side 4 6)) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonals_l3461_346167


namespace NUMINAMATH_CALUDE_altitude_sum_less_than_side_sum_l3461_346183

/-- Triangle structure with sides and altitudes -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  altitude_positive : 0 < h_a ∧ 0 < h_b ∧ 0 < h_c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The sum of altitudes is less than the sum of sides in any triangle -/
theorem altitude_sum_less_than_side_sum (t : Triangle) :
  t.h_a + t.h_b + t.h_c < t.a + t.b + t.c := by
  sorry

end NUMINAMATH_CALUDE_altitude_sum_less_than_side_sum_l3461_346183


namespace NUMINAMATH_CALUDE_birds_remaining_after_week_l3461_346103

/-- Calculates the number of birds remaining after a week given initial counts and daily losses. -/
def birdsRemaining (initialChickens initialTurkeys initialGuineaFowls : ℕ)
  (oddDayLossChickens oddDayLossTurkeys oddDayLossGuineaFowls : ℕ)
  (evenDayLossChickens evenDayLossTurkeys evenDayLossGuineaFowls : ℕ) : ℕ :=
  let oddDays := 4
  let evenDays := 3
  let remainingChickens := initialChickens - (oddDays * oddDayLossChickens + evenDays * evenDayLossChickens)
  let remainingTurkeys := initialTurkeys - (oddDays * oddDayLossTurkeys + evenDays * evenDayLossTurkeys)
  let remainingGuineaFowls := initialGuineaFowls - (oddDays * oddDayLossGuineaFowls + evenDays * evenDayLossGuineaFowls)
  remainingChickens + remainingTurkeys + remainingGuineaFowls

/-- Theorem stating that given the initial bird counts and daily losses, 379 birds remain after a week. -/
theorem birds_remaining_after_week :
  birdsRemaining 300 200 80 20 8 5 15 5 3 = 379 := by sorry

end NUMINAMATH_CALUDE_birds_remaining_after_week_l3461_346103


namespace NUMINAMATH_CALUDE_singer_hire_duration_l3461_346126

theorem singer_hire_duration (hourly_rate : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h_rate : hourly_rate = 15)
  (h_tip : tip_percentage = 0.20)
  (h_total : total_paid = 54) :
  ∃ (hours : ℝ), hours = 3 ∧ 
    hourly_rate * hours * (1 + tip_percentage) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_singer_hire_duration_l3461_346126


namespace NUMINAMATH_CALUDE_floor_sum_example_l3461_346163

theorem floor_sum_example : ⌊(18.7 : ℝ)⌋ + ⌊(-18.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3461_346163


namespace NUMINAMATH_CALUDE_kittens_remaining_l3461_346152

def initial_kittens : ℕ := 8
def kittens_given_away : ℕ := 2

theorem kittens_remaining : initial_kittens - kittens_given_away = 6 := by
  sorry

end NUMINAMATH_CALUDE_kittens_remaining_l3461_346152


namespace NUMINAMATH_CALUDE_existence_of_prime_1021_n_l3461_346175

theorem existence_of_prime_1021_n : ∃ n : ℕ, n ≥ 3 ∧ Nat.Prime (n^3 + 2*n + 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_1021_n_l3461_346175


namespace NUMINAMATH_CALUDE_evaluate_expression_l3461_346192

theorem evaluate_expression : 
  2100^3 - 2 * 2099 * 2100^2 - 2099^2 * 2100 + 2099^3 = 4404902 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3461_346192


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3461_346173

theorem complex_fraction_equality : (2 - I) / (2 + I) = 3/5 - 4/5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3461_346173


namespace NUMINAMATH_CALUDE_fraction_meaningful_implies_a_not_negative_one_l3461_346162

theorem fraction_meaningful_implies_a_not_negative_one (a : ℝ) :
  (∃ x : ℝ, x = 2 / (a + 1)) → a ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_implies_a_not_negative_one_l3461_346162


namespace NUMINAMATH_CALUDE_expression_decrease_l3461_346169

theorem expression_decrease (x y : ℝ) (h : x > 0 ∧ y > 0) : 
  let original := 125 * x * y^2
  let new_x := 0.75 * x
  let new_y := 0.75 * y
  let new_value := 125 * new_x * new_y^2
  new_value = (27/64) * original := by
sorry

end NUMINAMATH_CALUDE_expression_decrease_l3461_346169


namespace NUMINAMATH_CALUDE_multiple_in_denominator_l3461_346112

theorem multiple_in_denominator (a b k : ℚ) : 
  a / b = 4 / 1 →
  (a - 3 * b) / (k * (a - b)) = 1 / 7 →
  k = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_multiple_in_denominator_l3461_346112


namespace NUMINAMATH_CALUDE_geometric_arithmetic_relation_l3461_346168

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_relation (a b : ℕ → ℝ) :
  geometric_sequence a →
  arithmetic_sequence b →
  a 3 * a 11 = 4 * a 7 →
  a 7 = b 7 →
  b 5 + b 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_relation_l3461_346168


namespace NUMINAMATH_CALUDE_smallest_nine_digit_multiple_of_seven_digit_smallest_nine_digit_is_100_times_smallest_seven_digit_l3461_346154

theorem smallest_nine_digit_multiple_of_seven_digit : ℕ → ℕ → Prop :=
  fun smallest_nine_digit smallest_seven_digit =>
    smallest_nine_digit / smallest_seven_digit = 100

/-- The smallest nine-digit number is 100 times the smallest seven-digit number -/
theorem smallest_nine_digit_is_100_times_smallest_seven_digit 
  (h1 : smallest_nine_digit = 100000000)
  (h2 : smallest_seven_digit = 1000000) :
  smallest_nine_digit_multiple_of_seven_digit smallest_nine_digit smallest_seven_digit :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_nine_digit_multiple_of_seven_digit_smallest_nine_digit_is_100_times_smallest_seven_digit_l3461_346154


namespace NUMINAMATH_CALUDE_average_daily_high_temperature_l3461_346114

def daily_highs : List ℝ := [49, 62, 58, 57, 46]

theorem average_daily_high_temperature :
  (daily_highs.sum / daily_highs.length : ℝ) = 54.4 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_high_temperature_l3461_346114


namespace NUMINAMATH_CALUDE_calculation_proof_l3461_346185

theorem calculation_proof : (π - 3.15) ^ 0 * (-1) ^ 2023 - (-1/3) ^ (-2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3461_346185


namespace NUMINAMATH_CALUDE_chocolate_cost_450_l3461_346125

/-- The cost of buying a specific number of chocolate candies, given the cost and quantity per box. -/
def chocolate_cost (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem: The cost of 450 chocolate candies is $120, given that a box of 30 candies costs $8. -/
theorem chocolate_cost_450 : chocolate_cost 30 8 450 = 120 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_450_l3461_346125


namespace NUMINAMATH_CALUDE_parabola_vertex_l3461_346134

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 3(x-1)^2 + 2 is (1, 2) -/
theorem parabola_vertex : 
  ∀ x : ℝ, parabola x ≥ parabola (vertex.1) ∧ parabola (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3461_346134


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l3461_346158

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 0, 4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), c = -1/12 ∧ d = 7/12 ∧ N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l3461_346158


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l3461_346101

-- Define the cylinder
def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 10

-- Theorem statement
theorem longest_segment_in_cylinder :
  let diagonal := Real.sqrt (cylinder_height ^ 2 + (2 * cylinder_radius) ^ 2)
  diagonal = 10 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l3461_346101
