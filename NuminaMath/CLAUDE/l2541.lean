import Mathlib

namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_l2541_254105

theorem least_positive_integer_multiple (x : ℕ) : x = 42 ↔ 
  (x > 0 ∧ ∀ y : ℕ, y > 0 → y < x → ¬(∃ k : ℤ, (2 * y + 45)^2 = 43 * k)) ∧
  (∃ k : ℤ, (2 * x + 45)^2 = 43 * k) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_l2541_254105


namespace NUMINAMATH_CALUDE_A_initial_investment_l2541_254180

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := sorry

/-- Represents B's contribution to the capital in rupees -/
def B_contribution : ℝ := 15750

/-- Represents the number of months A was in the business -/
def A_months : ℝ := 12

/-- Represents the number of months B was in the business -/
def B_months : ℝ := 4

/-- Represents the ratio of profit division for A -/
def A_profit_ratio : ℝ := 2

/-- Represents the ratio of profit division for B -/
def B_profit_ratio : ℝ := 3

/-- Theorem stating that A's initial investment is 1750 rupees -/
theorem A_initial_investment : 
  A_investment = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_A_initial_investment_l2541_254180


namespace NUMINAMATH_CALUDE_base_seven_54321_equals_13539_l2541_254183

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_54321_equals_13539 :
  base_seven_to_ten [1, 2, 3, 4, 5] = 13539 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_54321_equals_13539_l2541_254183


namespace NUMINAMATH_CALUDE_pop_survey_result_l2541_254162

/-- Given a survey of 600 people where the central angle for "Pop" is 270°
    (to the nearest whole degree), prove that 450 people chose "Pop". -/
theorem pop_survey_result (total : ℕ) (angle : ℕ) (h_total : total = 600) (h_angle : angle = 270) :
  ∃ (pop : ℕ), pop = 450 ∧ 
  (pop : ℝ) / total * 360 ≥ angle - 0.5 ∧
  (pop : ℝ) / total * 360 < angle + 0.5 :=
by sorry

end NUMINAMATH_CALUDE_pop_survey_result_l2541_254162


namespace NUMINAMATH_CALUDE_new_persons_weight_l2541_254126

/-- The combined weight of two new persons replacing two existing persons in a group,
    given the average weight increase of the group. -/
theorem new_persons_weight
  (n : ℕ) -- number of persons in the group
  (avg_increase : ℝ) -- average weight increase per person
  (weight1 : ℝ) -- weight of first replaced person
  (weight2 : ℝ) -- weight of second replaced person
  (h1 : n = 15)
  (h2 : avg_increase = 5.2)
  (h3 : weight1 = 68)
  (h4 : weight2 = 70) :
  n * avg_increase + weight1 + weight2 = 216 :=
sorry

end NUMINAMATH_CALUDE_new_persons_weight_l2541_254126


namespace NUMINAMATH_CALUDE_tangent_line_of_g_l2541_254144

/-- Given a function f with a tangent line y = 2x - 1 at (2, f(2)),
    the tangent line to g(x) = x^2 + f(x) at (2, g(2)) is 6x - y - 5 = 0 -/
theorem tangent_line_of_g (f : ℝ → ℝ) (h : HasDerivAt f 2 2) :
  let g := λ x => x^2 + f x
  ∃ L : ℝ → ℝ, HasDerivAt g 6 2 ∧ L x = 6*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_of_g_l2541_254144


namespace NUMINAMATH_CALUDE_sum_even_divisors_1000_l2541_254124

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem sum_even_divisors_1000 : sum_even_divisors 1000 = 2184 := by sorry

end NUMINAMATH_CALUDE_sum_even_divisors_1000_l2541_254124


namespace NUMINAMATH_CALUDE_camel_cost_l2541_254150

/-- Represents the cost of different animals in Rupees -/
structure AnimalCosts where
  camel : ℝ
  horse : ℝ
  ox : ℝ
  elephant : ℝ
  giraffe : ℝ
  zebra : ℝ
  llama : ℝ

/-- The conditions given in the problem -/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  3 * costs.elephant = 5 * costs.giraffe ∧
  8 * costs.giraffe = 12 * costs.zebra ∧
  20 * costs.zebra = 7 * costs.llama ∧
  10 * costs.elephant = 120000

theorem camel_cost (costs : AnimalCosts) :
  problem_conditions costs → costs.camel = 4800 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l2541_254150


namespace NUMINAMATH_CALUDE_nine_digit_number_bounds_l2541_254107

theorem nine_digit_number_bounds (A B : ℕ) : 
  (∃ C b : ℕ, B = 10 * C + b ∧ b < 10 ∧ A = 10^8 * b + C) →
  B > 22222222 →
  Nat.gcd B 18 = 1 →
  A ≥ 122222224 ∧ A ≤ 999999998 :=
by sorry

end NUMINAMATH_CALUDE_nine_digit_number_bounds_l2541_254107


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2541_254103

theorem max_sum_of_squares (a b c : ℤ) : 
  a + b + c = 3 → a^3 + b^3 + c^3 = 3 → a^2 + b^2 + c^2 ≤ 57 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2541_254103


namespace NUMINAMATH_CALUDE_lakeview_academy_teachers_l2541_254153

/-- Represents the number of teachers at Lakeview Academy -/
def num_teachers (total_students : ℕ) (classes_per_student : ℕ) (class_size : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  (total_students * classes_per_student * 2) / (class_size * classes_per_teacher)

/-- Theorem stating the number of teachers at Lakeview Academy -/
theorem lakeview_academy_teachers :
  num_teachers 1500 6 25 5 = 144 := by
  sorry

#eval num_teachers 1500 6 25 5

end NUMINAMATH_CALUDE_lakeview_academy_teachers_l2541_254153


namespace NUMINAMATH_CALUDE_x_minus_2y_bounds_l2541_254190

theorem x_minus_2y_bounds (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) :
  0 ≤ x - 2*y ∧ x - 2*y ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_2y_bounds_l2541_254190


namespace NUMINAMATH_CALUDE_triangle_properties_l2541_254164

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a = 2)
  (h2 : abc.c = 1)
  (h3 : Real.tan abc.A + Real.tan abc.B = -(Real.tan abc.A * Real.tan abc.B)) :
  (Real.tan (abc.A + abc.B) = 1) ∧ 
  (((2 : Real) - Real.sqrt 2) / 2 = 1/2 * abc.a * abc.b * Real.sin abc.C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2541_254164


namespace NUMINAMATH_CALUDE_lcm_48_140_l2541_254129

theorem lcm_48_140 : Nat.lcm 48 140 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_140_l2541_254129


namespace NUMINAMATH_CALUDE_expression_evaluation_l2541_254145

theorem expression_evaluation (a b c : ℝ) 
  (h1 : a = 2)
  (h2 : b = a + 4)
  (h3 : c = b - 20)
  (h4 : a^2 + a ≠ 0)
  (h5 : b^2 - 6*b + 8 ≠ 0)
  (h6 : c^2 + 12*c + 36 ≠ 0) :
  (a^2 + 2*a) / (a^2 + a) * (b^2 - 4) / (b^2 - 6*b + 8) * (c^2 + 16*c + 64) / (c^2 + 12*c + 36) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2541_254145


namespace NUMINAMATH_CALUDE_problem_solution_l2541_254181

theorem problem_solution : (π - 3.14) ^ 0 + (-1/2) ^ (-1 : ℤ) + |3 - Real.sqrt 8| - 4 * Real.cos (π/4) = 2 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2541_254181


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_ratio_l2541_254170

theorem sphere_volume_surface_area_ratio 
  (r₁ r₂ : ℝ) 
  (h_positive₁ : r₁ > 0) 
  (h_positive₂ : r₂ > 0) 
  (h_volume_ratio : (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27) : 
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 := by
  sorry

#check sphere_volume_surface_area_ratio

end NUMINAMATH_CALUDE_sphere_volume_surface_area_ratio_l2541_254170


namespace NUMINAMATH_CALUDE_max_xyz_value_l2541_254160

theorem max_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y + 2 * z = (x + z) * (y + z)) :
  x * y * z ≤ 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_max_xyz_value_l2541_254160


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2541_254142

theorem multiplication_puzzle (c d : ℕ) : 
  c < 10 → d < 10 →
  (∃ n : ℕ, n < 1000 ∧ n % 100 = 8 ∧ 3 * c * 10 + c = n / d / 10) →
  c * 4 % 10 = 2 →
  (∃ x : ℕ, x < 10 ∧ 34 * d + 12 ≥ 10 * x * 10 + 60 ∧ 34 * d + 12 < 10 * x * 10 + 70) →
  c + d = 5 := by
sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2541_254142


namespace NUMINAMATH_CALUDE_marching_band_formation_l2541_254195

/-- A function that returns the list of divisors of a natural number -/
def divisors (n : ℕ) : List ℕ := sorry

/-- A function that returns the number of divisors of a natural number within a given range -/
def countDivisorsInRange (n m l : ℕ) : ℕ := sorry

theorem marching_band_formation (total_musicians : ℕ) (min_per_row : ℕ) (num_formations : ℕ) 
  (h1 : total_musicians = 240)
  (h2 : min_per_row = 8)
  (h3 : num_formations = 8) :
  ∃ (max_per_row : ℕ), 
    countDivisorsInRange total_musicians min_per_row max_per_row = num_formations ∧ 
    max_per_row = 80 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_formation_l2541_254195


namespace NUMINAMATH_CALUDE_nonnegative_integer_solution_l2541_254179

theorem nonnegative_integer_solution (x y z : ℕ) :
  (16 / 3 : ℝ)^x * (27 / 25 : ℝ)^y * (5 / 4 : ℝ)^z = 256 →
  x + y + z = 6 := by
sorry

end NUMINAMATH_CALUDE_nonnegative_integer_solution_l2541_254179


namespace NUMINAMATH_CALUDE_test_questions_count_l2541_254157

theorem test_questions_count : ∀ (total : ℕ), 
  (total % 5 = 0) →  -- The test has 5 equal sections
  (32 : ℚ) / total > (70 : ℚ) / 100 →  -- Percentage of correct answers > 70%
  (32 : ℚ) / total < (77 : ℚ) / 100 →  -- Percentage of correct answers < 77%
  total = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l2541_254157


namespace NUMINAMATH_CALUDE_min_value_expression_l2541_254116

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 16) :
  x^2 + 4*x*y + 4*y^2 + z^3 ≥ 73 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 16 ∧ x^2 + 4*x*y + 4*y^2 + z^3 = 73 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2541_254116


namespace NUMINAMATH_CALUDE_major_axis_length_for_given_cylinder_l2541_254120

/-- The length of the major axis of an ellipse formed by cutting a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The length of the major axis of an ellipse formed by cutting a right circular cylinder
    with radius 2 is 5.6, given that the major axis is 40% longer than the minor axis --/
theorem major_axis_length_for_given_cylinder :
  major_axis_length 2 1.4 = 5.6 := by sorry

end NUMINAMATH_CALUDE_major_axis_length_for_given_cylinder_l2541_254120


namespace NUMINAMATH_CALUDE_roll_five_dice_probability_l2541_254199

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_sides ^ num_dice

/-- The number of ways to roll an equal number of 1's and 6's -/
def equal_ones_and_sixes : ℕ := 2424

/-- The probability of rolling more 1's than 6's -/
def prob_more_ones_than_sixes : ℚ := 167 / 486

theorem roll_five_dice_probability :
  prob_more_ones_than_sixes = 1 / 2 * (1 - equal_ones_and_sixes / total_outcomes) :=
sorry

end NUMINAMATH_CALUDE_roll_five_dice_probability_l2541_254199


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l2541_254109

theorem sine_cosine_relation (α : ℝ) (h : Real.cos (α + π / 12) = 1 / 5) :
  Real.sin (α + 7 * π / 12) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l2541_254109


namespace NUMINAMATH_CALUDE_cylinder_painted_area_l2541_254152

/-- The total painted area of a cylinder with given dimensions and painting conditions -/
theorem cylinder_painted_area (h r : ℝ) (paint_percent : ℝ) : 
  h = 15 → r = 5 → paint_percent = 0.75 → 
  (2 * π * r^2) + (paint_percent * 2 * π * r * h) = 162.5 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_painted_area_l2541_254152


namespace NUMINAMATH_CALUDE_eight_by_eight_diagonal_shaded_count_l2541_254187

/-- Represents a square grid with a diagonal shading pattern -/
structure DiagonalGrid where
  size : Nat
  shaded_rows : Nat
  shaded_per_row : Nat

/-- Calculates the total number of shaded squares in a DiagonalGrid -/
def total_shaded (grid : DiagonalGrid) : Nat :=
  grid.shaded_rows * grid.shaded_per_row

/-- Theorem stating that an 8×8 grid with 7 shaded rows and 7 shaded squares per row has 49 total shaded squares -/
theorem eight_by_eight_diagonal_shaded_count :
  ∀ (grid : DiagonalGrid),
    grid.size = 8 →
    grid.shaded_rows = 7 →
    grid.shaded_per_row = 7 →
    total_shaded grid = 49 := by
  sorry

end NUMINAMATH_CALUDE_eight_by_eight_diagonal_shaded_count_l2541_254187


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2541_254198

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) : 
  -- Triangle conditions
  α + β + γ = 180 ∧
  -- Isosceles triangle condition (two angles are equal)
  (α = β ∨ β = γ ∨ α = γ) ∧
  -- One angle is 80°
  (α = 80 ∨ β = 80 ∨ γ = 80) →
  -- The vertex angle (the one that's not equal to the other two) is either 20° or 80°
  (α ≠ β → γ = 20 ∨ γ = 80) ∧
  (β ≠ γ → α = 20 ∨ α = 80) ∧
  (α ≠ γ → β = 20 ∨ β = 80) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2541_254198


namespace NUMINAMATH_CALUDE_piggy_bank_savings_l2541_254122

/-- Calculates the remaining amount in a piggy bank after a year of regular spending -/
theorem piggy_bank_savings (initial_amount : ℕ) (spending_per_trip : ℕ) (trips_per_month : ℕ) (months_per_year : ℕ) :
  initial_amount = 200 →
  spending_per_trip = 2 →
  trips_per_month = 4 →
  months_per_year = 12 →
  initial_amount - (spending_per_trip * trips_per_month * months_per_year) = 104 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_savings_l2541_254122


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l2541_254172

theorem sqrt_two_irrational : Irrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l2541_254172


namespace NUMINAMATH_CALUDE_max_red_dragons_l2541_254178

-- Define the dragon colors
inductive DragonColor
| Red
| Green
| Blue

-- Define the structure of a dragon
structure Dragon where
  color : DragonColor
  heads : Fin 3 → Bool  -- Each head is either truthful (true) or lying (false)

-- Define the statements made by each head
def headStatements (d : Dragon) (left right : DragonColor) : Prop :=
  (d.heads 0 = (left = DragonColor.Green)) ∧
  (d.heads 1 = (right = DragonColor.Blue)) ∧
  (d.heads 2 = (left ≠ DragonColor.Red ∧ right ≠ DragonColor.Red))

-- Define the condition that at least one head tells the truth
def atLeastOneTruthful (d : Dragon) : Prop :=
  ∃ i : Fin 3, d.heads i = true

-- Define the arrangement of dragons around the table
def validArrangement (arrangement : Fin 530 → Dragon) : Prop :=
  ∀ i : Fin 530,
    let left := arrangement ((i.val - 1 + 530) % 530)
    let right := arrangement ((i.val + 1) % 530)
    headStatements (arrangement i) left.color right.color ∧
    atLeastOneTruthful (arrangement i)

-- The main theorem
theorem max_red_dragons :
  ∀ arrangement : Fin 530 → Dragon,
    validArrangement arrangement →
    (∃ n : Nat, n ≤ 176 ∧ (∀ i : Fin 530, (arrangement i).color = DragonColor.Red → i.val < n)) :=
sorry

end NUMINAMATH_CALUDE_max_red_dragons_l2541_254178


namespace NUMINAMATH_CALUDE_negation_equivalence_l2541_254130

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ 
  (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2541_254130


namespace NUMINAMATH_CALUDE_horner_v3_value_l2541_254138

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 2x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- The coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-0.8, 1.7, -2.6, 3.5, 2, 5]

/-- Theorem: The third intermediate value (v_3) in Horner's method for f(x) at x=1 is 7.9 -/
theorem horner_v3_value : 
  (horner (coeffs.take 4) 1) = 7.9 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l2541_254138


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2541_254188

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- The sum of specific repeating decimals -/
theorem repeating_decimal_sum :
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2541_254188


namespace NUMINAMATH_CALUDE_triangle_area_cosine_sum_maximum_l2541_254163

theorem triangle_area_cosine_sum_maximum (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a = Real.sqrt 3 →
  a^2 = b^2 + c^2 + b*c →
  S = (1/2) * a * b * Real.sin C →
  (∃ (k : ℝ), S + Real.sqrt 3 * Real.cos B * Real.cos C ≤ k) →
  (∀ (k : ℝ), S + Real.sqrt 3 * Real.cos B * Real.cos C ≤ k → k ≥ Real.sqrt 3) :=
by sorry

#check triangle_area_cosine_sum_maximum

end NUMINAMATH_CALUDE_triangle_area_cosine_sum_maximum_l2541_254163


namespace NUMINAMATH_CALUDE_unique_integer_congruence_l2541_254197

theorem unique_integer_congruence : ∃! n : ℤ, 4 ≤ n ∧ n ≤ 8 ∧ n ≡ 7882 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_congruence_l2541_254197


namespace NUMINAMATH_CALUDE_max_distance_between_functions_l2541_254185

open Real

theorem max_distance_between_functions :
  let f (x : ℝ) := 2 * (sin (π / 4 + x))^2
  let g (x : ℝ) := Real.sqrt 3 * cos (2 * x)
  ∀ a : ℝ, |f a - g a| ≤ 3 ∧ ∃ b : ℝ, |f b - g b| = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_functions_l2541_254185


namespace NUMINAMATH_CALUDE_employee_b_pay_is_220_l2541_254139

/-- Given two employees A and B with a total weekly pay and A's pay as a percentage of B's, 
    calculate B's weekly pay. -/
def calculate_employee_b_pay (total_pay : ℚ) (a_percentage : ℚ) : ℚ :=
  total_pay / (1 + a_percentage)

/-- Theorem stating that given the problem conditions, employee B's pay is 220. -/
theorem employee_b_pay_is_220 :
  calculate_employee_b_pay 550 (3/2) = 220 := by sorry

end NUMINAMATH_CALUDE_employee_b_pay_is_220_l2541_254139


namespace NUMINAMATH_CALUDE_stating_count_paths_correct_l2541_254137

/-- 
Counts the number of paths from (0,0) to (m,n) where m < n and 
at every intermediate point (a,b), a < b.
-/
def count_paths (m n : ℕ) : ℕ :=
  if m < n then
    (Nat.factorial (m + n - 1) * (n - m)) / (Nat.factorial m * Nat.factorial n)
  else 0

/-- 
Theorem stating that count_paths gives the correct number of paths
from (0,0) to (m,n) satisfying the given conditions.
-/
theorem count_paths_correct (m n : ℕ) (h : m < n) :
  count_paths m n = ((Nat.factorial (m + n - 1) * (n - m)) / (Nat.factorial m * Nat.factorial n)) :=
by sorry

end NUMINAMATH_CALUDE_stating_count_paths_correct_l2541_254137


namespace NUMINAMATH_CALUDE_square_equality_implies_m_equals_four_l2541_254167

theorem square_equality_implies_m_equals_four (n m : ℝ) :
  (∀ x : ℝ, (x + n)^2 = x^2 + 4*x + m) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_implies_m_equals_four_l2541_254167


namespace NUMINAMATH_CALUDE_bus_journey_max_time_l2541_254148

/-- Represents the transportation options available to Jenny --/
inductive TransportOption
  | Bus
  | Walk
  | Bike
  | Carpool
  | Train

/-- Calculates the total time for a given transportation option --/
def total_time (option : TransportOption) : ℝ :=
  match option with
  | .Bus => 30 + 15  -- Bus time + walking time
  | .Walk => 30
  | .Bike => 20
  | .Carpool => 25
  | .Train => 45

/-- Jenny's walking speed in miles per minute --/
def walking_speed : ℝ := 0.05

/-- The maximum time allowed for any transportation option --/
def max_allowed_time : ℝ := 45

theorem bus_journey_max_time :
  ∀ (option : TransportOption),
    total_time TransportOption.Bus ≤ max_allowed_time ∧
    total_time TransportOption.Bus = total_time option →
    30 = max_allowed_time - (0.75 / walking_speed) := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_max_time_l2541_254148


namespace NUMINAMATH_CALUDE_trees_chopped_second_half_proof_l2541_254191

def trees_chopped_first_half : ℕ := 200
def trees_planted_per_chopped : ℕ := 3
def total_trees_to_plant : ℕ := 1500

def trees_chopped_second_half : ℕ := 300

theorem trees_chopped_second_half_proof :
  trees_chopped_second_half = 
    (total_trees_to_plant - trees_planted_per_chopped * trees_chopped_first_half) / 
    trees_planted_per_chopped := by
  sorry

end NUMINAMATH_CALUDE_trees_chopped_second_half_proof_l2541_254191


namespace NUMINAMATH_CALUDE_inequality_solution_l2541_254131

theorem inequality_solution (x : ℝ) : (x - 1) / (x - 3) ≥ 3 ↔ x ∈ Set.Ioo 3 4 ∪ {4} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2541_254131


namespace NUMINAMATH_CALUDE_difference_of_squares_l2541_254196

theorem difference_of_squares : 49^2 - 25^2 = 1776 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2541_254196


namespace NUMINAMATH_CALUDE_color_assignment_theorem_l2541_254151

def numbers : List ℕ := List.range 13 |>.map (· + 13)

structure ColorAssignment where
  black : ℕ
  red : List ℕ
  blue : List ℕ
  yellow : List ℕ
  green : List ℕ

def isValidAssignment (ca : ColorAssignment) : Prop :=
  ca.black ∈ numbers ∧
  ca.red.length = 3 ∧ ca.red.all (· ∈ numbers) ∧
  ca.blue.length = 3 ∧ ca.blue.all (· ∈ numbers) ∧
  ca.yellow.length = 3 ∧ ca.yellow.all (· ∈ numbers) ∧
  ca.green.length = 3 ∧ ca.green.all (· ∈ numbers) ∧
  ca.red.sum = ca.blue.sum ∧ ca.blue.sum = ca.yellow.sum ∧ ca.yellow.sum = ca.green.sum ∧
  13 ∈ ca.red ∧ 15 ∈ ca.yellow ∧ 23 ∈ ca.blue ∧
  (ca.black :: ca.red ++ ca.blue ++ ca.yellow ++ ca.green).toFinset = numbers.toFinset

theorem color_assignment_theorem (ca : ColorAssignment) 
  (h : isValidAssignment ca) : 
  ca.black = 19 ∧ ca.green = [14, 21, 22] := by
  sorry

#check color_assignment_theorem

end NUMINAMATH_CALUDE_color_assignment_theorem_l2541_254151


namespace NUMINAMATH_CALUDE_heavy_equipment_operator_pay_is_140_l2541_254100

/-- Calculates the daily pay for heavy equipment operators given the total number of people hired,
    total payroll, number of laborers, and daily pay for laborers. -/
def heavy_equipment_operator_pay (total_hired : ℕ) (total_payroll : ℕ) (laborers : ℕ) (laborer_pay : ℕ) : ℕ :=
  (total_payroll - laborers * laborer_pay) / (total_hired - laborers)

/-- Proves that given the specified conditions, the daily pay for heavy equipment operators is $140. -/
theorem heavy_equipment_operator_pay_is_140 :
  heavy_equipment_operator_pay 35 3950 19 90 = 140 := by
  sorry

end NUMINAMATH_CALUDE_heavy_equipment_operator_pay_is_140_l2541_254100


namespace NUMINAMATH_CALUDE_quadratic_negative_value_l2541_254127

theorem quadratic_negative_value (a b c : ℝ) :
  (∃ x : ℝ, x^2 + b*x + c = 0) →
  (∃ x : ℝ, a*x^2 + x + c = 0) →
  (∃ x : ℝ, a*x^2 + b*x + 1 = 0) →
  (∃ x : ℝ, a*x^2 + b*x + c < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_negative_value_l2541_254127


namespace NUMINAMATH_CALUDE_system_solution_l2541_254102

theorem system_solution :
  ∃ (x y z : ℝ),
    (1 / x + 2 / y - 3 / z = 3) ∧
    (4 / x - 1 / y - 2 / z = 5) ∧
    (3 / x + 4 / y + 1 / z = 23) ∧
    (x = 1 / 3) ∧ (y = 1 / 3) ∧ (z = 1 / 2) :=
by
  use 1/3, 1/3, 1/2
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l2541_254102


namespace NUMINAMATH_CALUDE_only_thirteen_fourths_between_three_and_four_l2541_254125

theorem only_thirteen_fourths_between_three_and_four :
  let numbers : List ℚ := [5/2, 11/4, 11/5, 13/4, 13/5]
  ∀ x ∈ numbers, (3 < x ∧ x < 4) ↔ x = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_only_thirteen_fourths_between_three_and_four_l2541_254125


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2541_254168

/-- Given a and b are real numbers satisfying the equation a - bi = (1 + i)i^3,
    prove that a = 1 and b = -1. -/
theorem complex_equation_solution (a b : ℝ) :
  (a : ℂ) - b * Complex.I = (1 + Complex.I) * Complex.I^3 →
  a = 1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2541_254168


namespace NUMINAMATH_CALUDE_max_distance_line_equation_l2541_254111

/-- A line in 2D space --/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Returns true if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Returns true if a point (x, y) is on the given line --/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- Returns the distance between two parallel lines --/
noncomputable def distance_between_parallel_lines (l1 l2 : Line) : ℝ :=
  sorry

/-- The theorem to be proved --/
theorem max_distance_line_equation (l1 l2 : Line) (A B : ℝ × ℝ) :
  are_parallel l1 l2 →
  point_on_line l1 A.1 A.2 →
  point_on_line l2 B.1 B.2 →
  A = (1, 3) →
  B = (2, 4) →
  (∀ l1' l2' : Line, are_parallel l1' l2' →
    point_on_line l1' A.1 A.2 →
    point_on_line l2' B.1 B.2 →
    distance_between_parallel_lines l1' l2' ≤ distance_between_parallel_lines l1 l2) →
  l1 = { slope := -1, y_intercept := 4 } :=
sorry

end NUMINAMATH_CALUDE_max_distance_line_equation_l2541_254111


namespace NUMINAMATH_CALUDE_problem_statement_l2541_254113

theorem problem_statement (a b c : ℝ) 
  (h1 : |a - b| = 1)
  (h2 : |b - c| = 1)
  (h3 : |c - a| = 2)
  (h4 : a * b * c = 60) :
  a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2541_254113


namespace NUMINAMATH_CALUDE_inscribed_rectangle_semicircle_radius_l2541_254165

/-- Given a rectangle inscribed in a semi-circle with specific properties,
    prove that the radius of the semi-circle is 23.625 cm. -/
theorem inscribed_rectangle_semicircle_radius 
  (perimeter : ℝ) 
  (width : ℝ) 
  (length : ℝ) 
  (h1 : perimeter = 126)
  (h2 : length = 3 * width)
  (h3 : perimeter = 2 * length + 2 * width) : 
  (length / 2 : ℝ) = 23.625 := by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_semicircle_radius_l2541_254165


namespace NUMINAMATH_CALUDE_correct_weighted_mean_l2541_254119

def total_values : ℕ := 30
def incorrect_mean : ℝ := 150
def first_error : ℝ := 135 - 165
def second_error : ℝ := 170 - 200
def weight_first_half : ℝ := 2
def weight_second_half : ℝ := 3

theorem correct_weighted_mean :
  let original_sum := incorrect_mean * total_values
  let total_error := first_error + second_error
  let corrected_sum := original_sum - total_error
  let total_weight := weight_first_half * (total_values / 2) + weight_second_half * (total_values / 2)
  corrected_sum / total_weight = 59.2 := by sorry

end NUMINAMATH_CALUDE_correct_weighted_mean_l2541_254119


namespace NUMINAMATH_CALUDE_rabbit_jumps_l2541_254155

def N (a : ℤ) : ℕ :=
  sorry

theorem rabbit_jumps (a : ℤ) : Odd (N a) ↔ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_jumps_l2541_254155


namespace NUMINAMATH_CALUDE_salary_percent_increase_l2541_254175

theorem salary_percent_increase 
  (x y : ℝ) (z : ℝ) 
  (hx : x > 0) 
  (hy : y ≥ 0) 
  (hz : z = (y / x) * 100) : 
  z = (y / x) * 100 := by
sorry

end NUMINAMATH_CALUDE_salary_percent_increase_l2541_254175


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2541_254177

/-- The probability of getting exactly k successes in n trials --/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 4 heads in 10 flips of a coin with 3/7 probability of heads --/
theorem coin_flip_probability : 
  binomial_probability 10 4 (3/7) = 69874560 / 282576201 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2541_254177


namespace NUMINAMATH_CALUDE_petes_walking_distance_closest_to_2800_l2541_254101

/-- Represents a pedometer with a maximum step count --/
structure Pedometer :=
  (max_count : ℕ)

/-- Represents Pete's walking data for a year --/
structure YearlyWalkingData :=
  (pedometer : Pedometer)
  (flips : ℕ)
  (final_reading : ℕ)
  (steps_per_mile : ℕ)

/-- Calculates the total steps walked in a year --/
def total_steps (data : YearlyWalkingData) : ℕ :=
  data.flips * (data.pedometer.max_count + 1) + data.final_reading

/-- Calculates the total miles walked in a year --/
def total_miles (data : YearlyWalkingData) : ℚ :=
  (total_steps data : ℚ) / data.steps_per_mile

/-- Theorem stating that Pete's walking distance is closest to 2800 miles --/
theorem petes_walking_distance_closest_to_2800 (data : YearlyWalkingData) 
  (h1 : data.pedometer.max_count = 89999)
  (h2 : data.flips = 55)
  (h3 : data.final_reading = 30000)
  (h4 : data.steps_per_mile = 1800) :
  ∃ (n : ℕ), n ≤ 50 ∧ |total_miles data - 2800| < |total_miles data - (2800 - n)| ∧ 
             |total_miles data - 2800| < |total_miles data - (2800 + n)| :=
  sorry

#eval total_miles { pedometer := { max_count := 89999 }, flips := 55, final_reading := 30000, steps_per_mile := 1800 }

end NUMINAMATH_CALUDE_petes_walking_distance_closest_to_2800_l2541_254101


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2541_254118

theorem min_value_of_expression (a b c d : ℝ) 
  (hb : b > 0) (hc : c > 0) (ha : a ≥ 0) (hd : d ≥ 0) 
  (h_sum : b + c ≥ a + d) : 
  (b / (c + d) + c / (a + b)) ≥ Real.sqrt 2 - 1/2 := 
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2541_254118


namespace NUMINAMATH_CALUDE_sandy_age_multiple_is_ten_l2541_254194

/-- The multiple of Sandy's age that equals her monthly phone bill expense -/
def sandy_age_multiple : ℕ → ℕ → ℕ → ℕ
| kim_age, sandy_future_age, sandy_expense =>
  let sandy_current_age := sandy_future_age - 2
  sandy_expense / sandy_current_age

/-- Theorem stating the multiple of Sandy's age that equals her monthly phone bill expense -/
theorem sandy_age_multiple_is_ten :
  sandy_age_multiple 10 36 340 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sandy_age_multiple_is_ten_l2541_254194


namespace NUMINAMATH_CALUDE_orangeade_price_day1_l2541_254173

/-- Represents the price and volume data for orangeade sales over two days -/
structure OrangeadeSales where
  orange_juice : ℝ
  water_day1 : ℝ
  water_day2 : ℝ
  price_day2 : ℝ
  revenue : ℝ

/-- Calculates the price per glass on the first day given orangeade sales data -/
def price_day1 (sales : OrangeadeSales) : ℝ :=
  1.5 * sales.price_day2

/-- Theorem stating that under the given conditions, the price on the first day is $0.30 -/
theorem orangeade_price_day1 (sales : OrangeadeSales) 
  (h1 : sales.water_day1 = sales.orange_juice)
  (h2 : sales.water_day2 = 2 * sales.water_day1)
  (h3 : sales.price_day2 = 0.2)
  (h4 : sales.revenue = (sales.orange_juice + sales.water_day1) * (price_day1 sales))
  (h5 : sales.revenue = (sales.orange_juice + sales.water_day2) * sales.price_day2) :
  price_day1 sales = 0.3 := by
  sorry

#eval price_day1 { orange_juice := 1, water_day1 := 1, water_day2 := 2, price_day2 := 0.2, revenue := 0.6 }

end NUMINAMATH_CALUDE_orangeade_price_day1_l2541_254173


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2541_254176

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := n + 1

-- Define the sum of the first n terms of 2a_n
def S (n : ℕ) : ℝ := 2^(n+2) - 4

-- Theorem statement
theorem arithmetic_sequence_properties :
  (a 1 = 2) ∧ 
  (a 1 + a 2 + a 3 = 9) ∧
  (∀ n : ℕ, a n = n + 1) ∧
  (∀ n : ℕ, S n = 2^(n+2) - 4) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2541_254176


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l2541_254156

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (blueberry_fraction raspberry_fraction blackberry_fraction walnut_fraction : ℚ)
  (h_total : total_pies = 30)
  (h_blueberry : blueberry_fraction = 1/3)
  (h_raspberry : raspberry_fraction = 3/5)
  (h_blackberry : blackberry_fraction = 5/6)
  (h_walnut : walnut_fraction = 1/10) :
  ∃ (max_without_ingredients : ℕ), 
    max_without_ingredients ≤ total_pies ∧
    max_without_ingredients = total_pies - (total_pies * blackberry_fraction).floor ∧
    max_without_ingredients = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l2541_254156


namespace NUMINAMATH_CALUDE_set_relations_l2541_254186

open Set

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

theorem set_relations (m : ℝ) :
  (A m ⊆ B ↔ m < 2 ∨ m > 4) ∧
  (A m ∩ B = ∅ ↔ m ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_set_relations_l2541_254186


namespace NUMINAMATH_CALUDE_egg_weight_calculation_l2541_254154

/-- Given the total weight of eggs and the number of dozens, 
    calculate the weight of a single egg. -/
theorem egg_weight_calculation 
  (total_weight : ℝ) 
  (dozens : ℕ) 
  (h1 : total_weight = 6) 
  (h2 : dozens = 8) : 
  total_weight / (dozens * 12) = 0.0625 := by
  sorry

end NUMINAMATH_CALUDE_egg_weight_calculation_l2541_254154


namespace NUMINAMATH_CALUDE_jacob_and_nathan_letters_l2541_254174

/-- The number of letters Nathan can write in one hour -/
def nathan_letters_per_hour : ℕ := 25

/-- Jacob's writing speed relative to Nathan's -/
def jacob_speed_multiplier : ℕ := 2

/-- The number of hours Jacob and Nathan work together -/
def total_hours : ℕ := 10

/-- Theorem: Jacob and Nathan can write 750 letters in 10 hours together -/
theorem jacob_and_nathan_letters : 
  (nathan_letters_per_hour + jacob_speed_multiplier * nathan_letters_per_hour) * total_hours = 750 := by
  sorry

end NUMINAMATH_CALUDE_jacob_and_nathan_letters_l2541_254174


namespace NUMINAMATH_CALUDE_wayne_blocks_l2541_254193

/-- The number of blocks Wayne's father gave him -/
def blocks_given (initial final : ℕ) : ℕ := final - initial

/-- Proof that Wayne's father gave him 6 blocks -/
theorem wayne_blocks : blocks_given 9 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_wayne_blocks_l2541_254193


namespace NUMINAMATH_CALUDE_shortest_midpoint_to_midpoint_path_length_l2541_254132

-- Define a regular cube
structure RegularCube where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

-- Define a path on the surface of the cube
def SurfacePath (cube : RegularCube) := ℝ

-- Define the property of being a valid path from midpoint to midpoint of opposite edges
def IsValidMidpointToMidpointPath (cube : RegularCube) (path : SurfacePath cube) : Prop :=
  sorry

-- Define the length of a path
def PathLength (cube : RegularCube) (path : SurfacePath cube) : ℝ :=
  sorry

-- Theorem statement
theorem shortest_midpoint_to_midpoint_path_length 
  (cube : RegularCube) 
  (h : cube.edgeLength = 2) :
  ∃ (path : SurfacePath cube), 
    IsValidMidpointToMidpointPath cube path ∧ 
    PathLength cube path = 4 ∧
    ∀ (other_path : SurfacePath cube), 
      IsValidMidpointToMidpointPath cube other_path → 
      PathLength cube other_path ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_shortest_midpoint_to_midpoint_path_length_l2541_254132


namespace NUMINAMATH_CALUDE_regular_polygons_ratio_l2541_254140

/-- The interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 - 360 / n

/-- The theorem statement -/
theorem regular_polygons_ratio (r k : ℕ) : 
  (r > 2 ∧ k > 2) →  -- Ensure polygons have at least 3 sides
  (interior_angle r / interior_angle k = 5 / 3) →
  (r = 2 * k) →
  (r = 8 ∧ k = 4) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygons_ratio_l2541_254140


namespace NUMINAMATH_CALUDE_A_3_2_equals_13_l2541_254166

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_13 : A 3 2 = 13 := by sorry

end NUMINAMATH_CALUDE_A_3_2_equals_13_l2541_254166


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l2541_254158

theorem smallest_value_complex_sum (a b c d : ℕ) (ω : ℂ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ω^4 = 1 →
  ω ≠ 1 →
  ∃ (min : ℝ), min = Real.sqrt 14 ∧
    ∀ (x y z w : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
      Complex.abs (x + y*ω + z*ω^2 + w*ω^3) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l2541_254158


namespace NUMINAMATH_CALUDE_competition_winner_and_probability_l2541_254169

def prob_A_win_round1 : ℚ := 3/5
def prob_B_win_round1 : ℚ := 3/4
def prob_A_win_round2 : ℚ := 3/5
def prob_B_win_round2 : ℚ := 1/2

def prob_A_win_competition : ℚ := prob_A_win_round1 * prob_A_win_round2
def prob_B_win_competition : ℚ := prob_B_win_round1 * prob_B_win_round2

theorem competition_winner_and_probability :
  (prob_B_win_competition > prob_A_win_competition) ∧
  (1 - (1 - prob_A_win_competition) * (1 - prob_B_win_competition) = 3/5) := by
  sorry

end NUMINAMATH_CALUDE_competition_winner_and_probability_l2541_254169


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_75_6300_l2541_254149

theorem gcd_lcm_sum_75_6300 : Nat.gcd 75 6300 + Nat.lcm 75 6300 = 6375 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_75_6300_l2541_254149


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l2541_254112

/-- The second term of an infinite geometric series with common ratio 1/4 and sum 16 is 3 -/
theorem second_term_of_geometric_series (a : ℝ) : 
  (∑' n, a * (1/4)^n = 16) → a * (1/4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l2541_254112


namespace NUMINAMATH_CALUDE_set_equality_l2541_254161

theorem set_equality : {x : ℕ | x > 1 ∧ x ≤ 3} = {x : ℕ | x = 2 ∨ x = 3} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2541_254161


namespace NUMINAMATH_CALUDE_percent_relation_l2541_254159

theorem percent_relation (j k l m : ℝ) (x : ℝ) 
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (200 / 100) * (350 / 100))
  (hj : j ≠ 0) :
  x = 500 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2541_254159


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2541_254117

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ + m = 0 ∧ x₂^2 + 4*x₂ + m = 0) → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2541_254117


namespace NUMINAMATH_CALUDE_photographers_selection_l2541_254110

theorem photographers_selection (n m : ℕ) (h1 : n = 10) (h2 : m = 3) :
  Nat.choose n m = 120 := by
  sorry

end NUMINAMATH_CALUDE_photographers_selection_l2541_254110


namespace NUMINAMATH_CALUDE_problem_3_l2541_254135

theorem problem_3 (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a^2 + b^2 = 6*a*b) :
  (a + b) / (a - b) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_3_l2541_254135


namespace NUMINAMATH_CALUDE_negation_equivalence_l2541_254136

theorem negation_equivalence : 
  (¬(x = 3 → x^2 - 2*x - 3 = 0)) ↔ (x ≠ 3 → x^2 - 2*x - 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2541_254136


namespace NUMINAMATH_CALUDE_max_cos_value_l2541_254123

theorem max_cos_value (a b : ℝ) (h : Real.cos (a - b) = Real.cos a - Real.cos b) :
  ∀ x : ℝ, Real.cos a ≤ 1 ∧ (Real.cos x ≤ Real.cos a → x = a) :=
by sorry

end NUMINAMATH_CALUDE_max_cos_value_l2541_254123


namespace NUMINAMATH_CALUDE_kiddie_scoop_cost_l2541_254104

/-- The cost of ice cream scoops for the Martin family --/
def ice_cream_cost (kiddie_scoop : ℕ) : Prop :=
  let regular_scoop : ℕ := 4
  let double_scoop : ℕ := 6
  let total_cost : ℕ := 32
  let num_regular : ℕ := 2  -- Mr. and Mrs. Martin
  let num_kiddie : ℕ := 2   -- Their two children
  let num_double : ℕ := 3   -- Their three teenage children
  
  total_cost = num_regular * regular_scoop + num_kiddie * kiddie_scoop + num_double * double_scoop

theorem kiddie_scoop_cost : ice_cream_cost 3 := by
  sorry

end NUMINAMATH_CALUDE_kiddie_scoop_cost_l2541_254104


namespace NUMINAMATH_CALUDE_solution_set_abs_fraction_l2541_254128

theorem solution_set_abs_fraction (x : ℝ) : 
  (|x / (x - 1)| = x / (x - 1)) ↔ (x ≤ 0 ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_abs_fraction_l2541_254128


namespace NUMINAMATH_CALUDE_margie_change_is_six_l2541_254106

/-- The change Margie received after buying apples -/
def margieChange (numApples : ℕ) (costPerApple : ℚ) (amountPaid : ℚ) : ℚ :=
  amountPaid - (numApples : ℚ) * costPerApple

/-- Theorem: Margie's change is $6.00 -/
theorem margie_change_is_six :
  margieChange 5 (80 / 100) 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_margie_change_is_six_l2541_254106


namespace NUMINAMATH_CALUDE_apple_selling_price_l2541_254114

theorem apple_selling_price (cost_price : ℝ) (loss_fraction : ℝ) (selling_price : ℝ) :
  cost_price = 21 →
  loss_fraction = 1 / 6 →
  selling_price = cost_price * (1 - loss_fraction) →
  selling_price = 17.5 := by
sorry

end NUMINAMATH_CALUDE_apple_selling_price_l2541_254114


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l2541_254133

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) :
  (x ^ (1/4)) / (x ^ (1/7)) = x ^ (3/28) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l2541_254133


namespace NUMINAMATH_CALUDE_prime_sum_problem_l2541_254182

theorem prime_sum_problem (m n : ℕ) (hm : Nat.Prime m) (hn : Nat.Prime n) 
  (h : 5 * m + 7 * n = 129) : m + n = 19 ∨ m + n = 25 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l2541_254182


namespace NUMINAMATH_CALUDE_function_growth_l2541_254171

open Real

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem function_growth (hf : ∀ x, f x < f' x) :
  (f 1 > Real.exp 1 * f 0) ∧ (f 2023 > Real.exp 2023 * f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_growth_l2541_254171


namespace NUMINAMATH_CALUDE_reduced_price_is_30_l2541_254121

/-- Represents the price reduction of oil as a percentage -/
def price_reduction : ℝ := 0.20

/-- Represents the additional amount of oil that can be purchased after the price reduction -/
def additional_oil : ℝ := 10

/-- Represents the total cost in Rupees -/
def total_cost : ℝ := 1500

/-- Calculates the reduced price per kg of oil -/
def reduced_price_per_kg (original_price : ℝ) : ℝ :=
  original_price * (1 - price_reduction)

/-- Theorem stating that the reduced price per kg of oil is 30 Rupees -/
theorem reduced_price_is_30 :
  ∃ (original_price : ℝ) (original_quantity : ℝ),
    original_quantity > 0 ∧
    original_price > 0 ∧
    original_quantity * original_price = total_cost ∧
    (original_quantity + additional_oil) * reduced_price_per_kg original_price = total_cost ∧
    reduced_price_per_kg original_price = 30 :=
  sorry

end NUMINAMATH_CALUDE_reduced_price_is_30_l2541_254121


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_20_l2541_254141

theorem x_plus_2y_equals_20 (x y : ℝ) (hx : x = 10) (hy : y = 5) : x + 2 * y = 20 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_20_l2541_254141


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2541_254189

theorem quadratic_equation_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h1 : p^2 + p*p + q = 0) (h2 : q^2 + p*q + q = 0) : p = 1 ∧ q = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2541_254189


namespace NUMINAMATH_CALUDE_initial_capacity_correct_l2541_254108

/-- The capacity of each bucket in the set of 20 buckets -/
def initial_bucket_capacity : ℝ := 13.5

/-- The number of buckets in the initial set -/
def initial_bucket_count : ℕ := 20

/-- The capacity of each bucket in the set of 30 buckets -/
def new_bucket_capacity : ℝ := 9

/-- The number of buckets in the new set -/
def new_bucket_count : ℕ := 30

/-- The theorem states that the initial bucket capacity is correct -/
theorem initial_capacity_correct : 
  initial_bucket_capacity * initial_bucket_count = new_bucket_capacity * new_bucket_count := by
  sorry

end NUMINAMATH_CALUDE_initial_capacity_correct_l2541_254108


namespace NUMINAMATH_CALUDE_fathers_full_time_jobs_l2541_254115

theorem fathers_full_time_jobs (total_parents : ℝ) 
  (h1 : total_parents > 0) -- Ensure total_parents is positive
  (mothers_ratio : ℝ) 
  (h2 : mothers_ratio = 0.4) -- 40% of parents are mothers
  (mothers_full_time_ratio : ℝ) 
  (h3 : mothers_full_time_ratio = 3/4) -- 3/4 of mothers have full-time jobs
  (not_full_time_ratio : ℝ) 
  (h4 : not_full_time_ratio = 0.16) -- 16% of parents do not have full-time jobs
  : (total_parents * (1 - mothers_ratio) - 
     total_parents * (1 - not_full_time_ratio - mothers_ratio * mothers_full_time_ratio)) / 
    (total_parents * (1 - mothers_ratio)) = 9/10 := by
  sorry


end NUMINAMATH_CALUDE_fathers_full_time_jobs_l2541_254115


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_4_6_8_10_l2541_254147

def is_divisible_by_all (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (n % 6 = 0) ∧ (n % 8 = 0) ∧ (n % 10 = 0)

theorem smallest_number_divisible_by_4_6_8_10 :
  ∀ n : ℕ, n ≥ 136 → (is_divisible_by_all (n - 16) → n ≥ 136) ∧
  is_divisible_by_all (136 - 16) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_4_6_8_10_l2541_254147


namespace NUMINAMATH_CALUDE_trailing_zeros_25_factorial_l2541_254143

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 25! is 6 -/
theorem trailing_zeros_25_factorial :
  trailingZeros 25 = 6 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_25_factorial_l2541_254143


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2541_254192

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2541_254192


namespace NUMINAMATH_CALUDE_kabadi_players_count_l2541_254134

/-- Represents the number of players in different categories -/
structure PlayerCounts where
  total : ℕ
  khoKhoOnly : ℕ
  bothGames : ℕ

/-- Calculates the number of players who play kabadi -/
def kabadiPlayers (counts : PlayerCounts) : ℕ :=
  counts.total - counts.khoKhoOnly + counts.bothGames

/-- Theorem stating the number of kabadi players given the conditions -/
theorem kabadi_players_count (counts : PlayerCounts) 
  (h1 : counts.total = 30)
  (h2 : counts.khoKhoOnly = 20)
  (h3 : counts.bothGames = 5) :
  kabadiPlayers counts = 15 := by
  sorry


end NUMINAMATH_CALUDE_kabadi_players_count_l2541_254134


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2541_254184

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2541_254184


namespace NUMINAMATH_CALUDE_park_creatures_l2541_254146

/-- The number of dogs at the park -/
def num_dogs : ℕ := 60

/-- The number of people at the park -/
def num_people : ℕ := num_dogs / 2

/-- The number of snakes at the park -/
def num_snakes : ℕ := num_people / 2

/-- The total number of eyes and legs of all creatures at the park -/
def total_eyes_and_legs : ℕ := 510

theorem park_creatures :
  (num_dogs = 2 * num_people) ∧
  (num_people = 2 * num_snakes) ∧
  (4 * num_dogs + 4 * num_people + 2 * num_snakes = total_eyes_and_legs) :=
by sorry

#check park_creatures

end NUMINAMATH_CALUDE_park_creatures_l2541_254146
