import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l4015_401535

theorem complex_equation_solution (z : ℂ) :
  z * (1 - Complex.I) = 2 + Complex.I → z = (1 / 2 : ℂ) + (3 / 2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4015_401535


namespace NUMINAMATH_CALUDE_plain_lemonade_price_calculation_l4015_401525

/-- The price of a glass of plain lemonade -/
def plain_lemonade_price : ℚ := 3 / 4

/-- The number of glasses of plain lemonade sold -/
def plain_lemonade_sold : ℕ := 36

/-- The amount made from strawberry lemonade -/
def strawberry_lemonade_sales : ℕ := 16

/-- The difference between plain and strawberry lemonade sales -/
def sales_difference : ℕ := 11

theorem plain_lemonade_price_calculation :
  plain_lemonade_price * plain_lemonade_sold = 
  (strawberry_lemonade_sales + sales_difference : ℚ) := by sorry

end NUMINAMATH_CALUDE_plain_lemonade_price_calculation_l4015_401525


namespace NUMINAMATH_CALUDE_fixed_point_of_parabola_l4015_401524

/-- Theorem: All parabolas of the form y = 4x^2 + 2tx - 3t pass through the point (3, 36) for any real t. -/
theorem fixed_point_of_parabola (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + 2 * t * x - 3 * t
  f 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabola_l4015_401524


namespace NUMINAMATH_CALUDE_unique_solution_l4015_401536

/-- The solution set of the inequality |ax - 2| < 3 with respect to x -/
def SolutionSet (a : ℝ) : Set ℝ :=
  {x : ℝ | |a * x - 2| < 3}

/-- The given solution set -/
def GivenSet : Set ℝ :=
  {x : ℝ | -5/3 < x ∧ x < 1/3}

/-- The theorem stating that a = -3 is the unique value satisfying the conditions -/
theorem unique_solution : ∃! a : ℝ, SolutionSet a = GivenSet :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l4015_401536


namespace NUMINAMATH_CALUDE_distribute_four_items_three_bags_l4015_401513

/-- The number of ways to distribute n distinct items into k identical bags, allowing empty bags. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 14 ways to distribute 4 distinct items into 3 identical bags, allowing empty bags. -/
theorem distribute_four_items_three_bags : distribute 4 3 = 14 := by sorry

end NUMINAMATH_CALUDE_distribute_four_items_three_bags_l4015_401513


namespace NUMINAMATH_CALUDE_english_teachers_count_l4015_401558

def committee_size (E : ℕ) : ℕ := E + 4 + 2

def probability_two_math_teachers (E : ℕ) : ℚ :=
  6 / (committee_size E * (committee_size E - 1) / 2)

theorem english_teachers_count :
  ∃ E : ℕ, probability_two_math_teachers E = 1/12 ∧ E = 3 :=
sorry

end NUMINAMATH_CALUDE_english_teachers_count_l4015_401558


namespace NUMINAMATH_CALUDE_special_collection_books_l4015_401583

/-- The number of books in a special collection at the beginning of a month,
    given the number of books loaned, returned, and remaining at the end of the month. -/
theorem special_collection_books
  (loaned : ℕ)
  (return_rate : ℚ)
  (end_count : ℕ)
  (h1 : loaned = 30)
  (h2 : return_rate = 7/10)
  (h3 : end_count = 66)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_special_collection_books_l4015_401583


namespace NUMINAMATH_CALUDE_triangle_point_distance_height_inequality_l4015_401565

/-- Given a triangle and a point M inside it, this theorem states that the sum of the α-th powers
    of the ratios of distances from M to the sides to the corresponding heights of the triangle
    is always greater than or equal to 1/3ᵅ⁻¹, for α ≥ 1. -/
theorem triangle_point_distance_height_inequality
  (α : ℝ) (h_α : α ≥ 1)
  (k₁ k₂ k₃ h₁ h₂ h₃ : ℝ)
  (h_positive : k₁ > 0 ∧ k₂ > 0 ∧ k₃ > 0 ∧ h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0)
  (h_sum : k₁/h₁ + k₂/h₂ + k₃/h₃ = 1) :
  (k₁/h₁)^α + (k₂/h₂)^α + (k₃/h₃)^α ≥ 1/(3^(α-1)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_point_distance_height_inequality_l4015_401565


namespace NUMINAMATH_CALUDE_two_numbers_problem_l4015_401504

theorem two_numbers_problem (x y : ℚ) : 
  (4 * y = 9 * x) → 
  (y - x = 12) → 
  y = 108 / 5 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l4015_401504


namespace NUMINAMATH_CALUDE_smallest_b_value_l4015_401537

theorem smallest_b_value (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1/a + 1/b ≤ 1/2) →
  b ≥ (7 + Real.sqrt 17) / 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l4015_401537


namespace NUMINAMATH_CALUDE_vector_subtraction_l4015_401543

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 4) → b = (1, 2) → a - 2 • b = (1, 0) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l4015_401543


namespace NUMINAMATH_CALUDE_discount_calculation_l4015_401508

-- Define the initial discount
def initial_discount : ℝ := 0.40

-- Define the additional discount
def additional_discount : ℝ := 0.10

-- Define the claimed total discount
def claimed_discount : ℝ := 0.55

-- Theorem to prove the actual discount and the difference
theorem discount_calculation :
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_additional := remaining_after_initial * (1 - additional_discount)
  let actual_discount := 1 - remaining_after_additional
  let discount_difference := claimed_discount - actual_discount
  actual_discount = 0.46 ∧ discount_difference = 0.09 := by sorry

end NUMINAMATH_CALUDE_discount_calculation_l4015_401508


namespace NUMINAMATH_CALUDE_question_one_l4015_401534

theorem question_one (a b : ℚ) : |a| = 3 ∧ |b| = 1 ∧ a < b → a + b = -2 ∨ a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_question_one_l4015_401534


namespace NUMINAMATH_CALUDE_minimal_force_to_submerge_cube_l4015_401599

-- Define constants
def cube_volume : Real := 10e-6  -- 10 cm³ in m³
def cube_density : Real := 500   -- kg/m³
def water_density : Real := 1000 -- kg/m³
def gravity : Real := 10         -- m/s²

-- Define the minimal force function
def minimal_force (v : Real) (ρ_cube : Real) (ρ_water : Real) (g : Real) : Real :=
  (ρ_water - ρ_cube) * v * g

-- Theorem statement
theorem minimal_force_to_submerge_cube :
  minimal_force cube_volume cube_density water_density gravity = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_minimal_force_to_submerge_cube_l4015_401599


namespace NUMINAMATH_CALUDE_college_student_count_l4015_401548

/-- Proves that in a college with a given ratio of boys to girls and a known number of girls,
    the total number of students is as expected. -/
theorem college_student_count
  (boys_girls_ratio : Rat)
  (num_girls : ℕ)
  (h_ratio : boys_girls_ratio = 8 / 5)
  (h_girls : num_girls = 135) :
  (boys_girls_ratio * num_girls + num_girls : ℚ) = 351 := by
  sorry

#check college_student_count

end NUMINAMATH_CALUDE_college_student_count_l4015_401548


namespace NUMINAMATH_CALUDE_frankie_candy_count_l4015_401527

theorem frankie_candy_count (max_candy : ℕ) (extra_candy : ℕ) (frankie_candy : ℕ) : 
  max_candy = 92 → 
  extra_candy = 18 → 
  max_candy = frankie_candy + extra_candy → 
  frankie_candy = 74 := by
sorry

end NUMINAMATH_CALUDE_frankie_candy_count_l4015_401527


namespace NUMINAMATH_CALUDE_percentage_difference_l4015_401505

theorem percentage_difference : (56 * 0.50) - (50 * 0.30) = 13 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l4015_401505


namespace NUMINAMATH_CALUDE_fathers_age_l4015_401550

theorem fathers_age (son_age : ℕ) (father_age : ℕ) : 
  father_age = 4 * son_age →
  (son_age - 10) + (father_age - 10) = 60 →
  father_age = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l4015_401550


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l4015_401512

/-- The sum of interior numbers in a row of Pascal's Triangle -/
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_triangle_interior_sum :
  sumInteriorNumbers 6 = 30 →
  sumInteriorNumbers 8 = 126 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l4015_401512


namespace NUMINAMATH_CALUDE_exists_initial_points_for_82_final_l4015_401557

/-- The number of points after applying the procedure once -/
def points_after_first_procedure (n : ℕ) : ℕ := 3 * n - 2

/-- The number of points after applying the procedure twice -/
def points_after_second_procedure (n : ℕ) : ℕ := 9 * n - 8

/-- Theorem stating that it's possible to have 82 points after the two procedures -/
theorem exists_initial_points_for_82_final : ∃ n : ℕ, points_after_second_procedure n = 82 := by
  sorry

#eval points_after_second_procedure 10

end NUMINAMATH_CALUDE_exists_initial_points_for_82_final_l4015_401557


namespace NUMINAMATH_CALUDE_total_crayons_l4015_401503

/-- Represents the number of crayons in a box of type 1 -/
def box_type1 : ℕ := 8 + 4 + 5

/-- Represents the number of crayons in a box of type 2 -/
def box_type2 : ℕ := 7 + 6 + 3

/-- Represents the number of crayons in a box of type 3 -/
def box_type3 : ℕ := 11 + 5 + 2

/-- Represents the number of crayons in the unique box -/
def unique_box : ℕ := 9 + 2 + 7

/-- Represents the total number of boxes -/
def total_boxes : ℕ := 3 + 4 + 2 + 1

theorem total_crayons : 
  3 * box_type1 + 4 * box_type2 + 2 * box_type3 + unique_box = 169 :=
sorry

end NUMINAMATH_CALUDE_total_crayons_l4015_401503


namespace NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l4015_401538

theorem power_of_eight_sum_equals_power_of_two : ∃ x : ℕ, 8^3 + 8^3 + 8^3 + 8^3 = 2^x ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l4015_401538


namespace NUMINAMATH_CALUDE_letters_per_large_envelope_l4015_401507

theorem letters_per_large_envelope 
  (total_letters : ℕ) 
  (small_envelope_letters : ℕ) 
  (large_envelopes : ℕ) 
  (h1 : total_letters = 80) 
  (h2 : small_envelope_letters = 20) 
  (h3 : large_envelopes = 30) : 
  (total_letters - small_envelope_letters) / large_envelopes = 2 := by
  sorry

end NUMINAMATH_CALUDE_letters_per_large_envelope_l4015_401507


namespace NUMINAMATH_CALUDE_park_outer_diameter_l4015_401596

/-- Given a circular park with a central fountain, surrounded by a garden ring and a walking path,
    this theorem proves the diameter of the outer boundary of the walking path. -/
theorem park_outer_diameter
  (fountain_diameter : ℝ)
  (garden_width : ℝ)
  (path_width : ℝ)
  (h1 : fountain_diameter = 20)
  (h2 : garden_width = 10)
  (h3 : path_width = 6) :
  2 * (fountain_diameter / 2 + garden_width + path_width) = 52 :=
by sorry

end NUMINAMATH_CALUDE_park_outer_diameter_l4015_401596


namespace NUMINAMATH_CALUDE_twenty_eight_is_seventy_percent_of_forty_l4015_401539

theorem twenty_eight_is_seventy_percent_of_forty :
  (28 : ℝ) / 40 = 70 / 100 := by sorry

end NUMINAMATH_CALUDE_twenty_eight_is_seventy_percent_of_forty_l4015_401539


namespace NUMINAMATH_CALUDE_total_wood_needed_l4015_401579

def bench1_wood (length1 length2 : ℝ) (count1 count2 : ℕ) : ℝ :=
  length1 * count1 + length2 * count2

def bench2_wood (length1 length2 : ℝ) (count1 count2 : ℕ) : ℝ :=
  length1 * count1 + length2 * count2

def bench3_wood (length1 length2 : ℝ) (count1 count2 : ℕ) : ℝ :=
  length1 * count1 + length2 * count2

theorem total_wood_needed :
  let bench1 := bench1_wood 4 2 6 2
  let bench2 := bench2_wood 3 1.5 8 5
  let bench3 := bench3_wood 5 2.5 4 3
  bench1 + bench2 + bench3 = 87 := by sorry

end NUMINAMATH_CALUDE_total_wood_needed_l4015_401579


namespace NUMINAMATH_CALUDE_bank_line_theorem_l4015_401568

/-- Represents a bank line with fast and slow customers. -/
structure BankLine where
  total_customers : Nat
  fast_customers : Nat
  slow_customers : Nat
  fast_operation_time : Nat
  slow_operation_time : Nat

/-- Calculates the minimum total wasted person-minutes. -/
def minimum_wasted_time (line : BankLine) : Nat :=
  sorry

/-- Calculates the maximum total wasted person-minutes. -/
def maximum_wasted_time (line : BankLine) : Nat :=
  sorry

/-- Calculates the expected number of wasted person-minutes. -/
def expected_wasted_time (line : BankLine) : Nat :=
  sorry

/-- Theorem stating the results for the specific bank line scenario. -/
theorem bank_line_theorem (line : BankLine) 
    (h1 : line.total_customers = 8)
    (h2 : line.fast_customers = 5)
    (h3 : line.slow_customers = 3)
    (h4 : line.fast_operation_time = 1)
    (h5 : line.slow_operation_time = 5) :
  minimum_wasted_time line = 40 ∧
  maximum_wasted_time line = 100 ∧
  expected_wasted_time line = 70 :=
  sorry

end NUMINAMATH_CALUDE_bank_line_theorem_l4015_401568


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_fraction_l4015_401552

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio_equals_fraction : 
  (factorial 6)^2 / (factorial 5 * factorial 7) = 100 / 101 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_fraction_l4015_401552


namespace NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l4015_401516

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 68 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 68 := by sorry

end NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l4015_401516


namespace NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l4015_401576

/-- The equation holds for all real x if and only if a, b, p, and q have specific values -/
theorem equation_holds_iff_specific_values :
  ∀ (a b p q : ℝ),
    (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
    (a = (2^20 - 1)^(1/20) ∧
     b = -(2^20 - 1)^(1/20) / 2 ∧
     p = -1 ∧
     q = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_iff_specific_values_l4015_401576


namespace NUMINAMATH_CALUDE_cosine_sine_equation_l4015_401523

theorem cosine_sine_equation (n : ℕ) :
  (∀ k : ℤ, (Real.cos (2 * k * Real.pi)) ^ n - (Real.sin (2 * k * Real.pi)) ^ n = 1) ∧
  (Even n → ∀ k : ℤ, (Real.cos ((2 * k + 1) * Real.pi)) ^ n - (Real.sin ((2 * k + 1) * Real.pi)) ^ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_cosine_sine_equation_l4015_401523


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4015_401519

theorem min_value_quadratic (x : ℝ) :
  (4 * x^2 + 6 * x + 3 = 5) → x ≥ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4015_401519


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l4015_401518

theorem quadratic_equation_properties (m : ℝ) :
  (∀ m, ∃ x : ℝ, x^2 - (m-1)*x + (m-2) = 0) ∧
  (∃ x : ℝ, x^2 - (m-1)*x + (m-2) = 0 ∧ x > 6 → m > 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l4015_401518


namespace NUMINAMATH_CALUDE_math_problem_proof_l4015_401549

-- Define the line l: 2x + y + 3 = 0
def line_l (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define a directional vector
def is_directional_vector (u : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (t : ℝ), line (t * u.1) (t * u.2)

-- Define a line passing through a point with equal intercepts
def line_equal_intercepts (x y : ℝ) : Prop := x + y - 6 = 0

-- Define y-intercept
def y_intercept (m b : ℝ) : ℝ := b

theorem math_problem_proof :
  (is_directional_vector (1, -2) line_l) ∧
  (line_equal_intercepts 2 4) ∧
  (y_intercept 3 (-2) = -2) := by
  sorry

end NUMINAMATH_CALUDE_math_problem_proof_l4015_401549


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l4015_401584

/-- Calculates the sum of digits of a number in base 10 -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Calculates the sum of binary digits of a number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is less than 500 -/
def lessThan500 (n : ℕ) : Prop := n < 500

/-- Checks if the cost is the same for both options -/
def equalCost (n : ℕ) : Prop := sumDigits n = sumBinaryDigits n

theorem largest_equal_cost_number :
  ∀ n : ℕ, lessThan500 n → equalCost n → n ≤ 247 :=
sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_l4015_401584


namespace NUMINAMATH_CALUDE_sin_equation_implies_sin_formula_l4015_401511

theorem sin_equation_implies_sin_formula (α : ℝ) 
  (h : Real.sin α - Real.sqrt 3 * Real.cos α = 1) : 
  Real.sin ((7 * Real.pi / 6) - 2 * α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_equation_implies_sin_formula_l4015_401511


namespace NUMINAMATH_CALUDE_replaced_person_weight_is_65_l4015_401594

/-- The weight of the replaced person when the average weight of 6 persons
    increases by 2.5 kg after replacing one person with a new 80 kg person -/
def replacedPersonWeight (initialCount : ℕ) (averageIncrease : ℝ) (newPersonWeight : ℝ) : ℝ :=
  newPersonWeight - (initialCount : ℝ) * averageIncrease

/-- Theorem stating that under the given conditions, the weight of the replaced person is 65 kg -/
theorem replaced_person_weight_is_65 :
  replacedPersonWeight 6 2.5 80 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_is_65_l4015_401594


namespace NUMINAMATH_CALUDE_thirteen_fourth_mod_eight_l4015_401554

theorem thirteen_fourth_mod_eight : 13^4 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_fourth_mod_eight_l4015_401554


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l4015_401562

theorem roots_of_quadratic_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁^2 - 4 = 0 ∧ x₂^2 - 4 = 0) ∧ x₁ = 2 ∧ x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l4015_401562


namespace NUMINAMATH_CALUDE_line_parabola_tangency_false_l4015_401522

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define a line
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the concept of a common point
def common_point (p : ℝ) (a b c : ℝ) (x y : ℝ) : Prop :=
  parabola p x y ∧ line a b c x y

-- Define the concept of tangency
def is_tangent (p : ℝ) (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, common_point p a b c x y ∧
  ∀ x' y' : ℝ, common_point p a b c x' y' → x' = x ∧ y' = y

-- The theorem to be proved
theorem line_parabola_tangency_false :
  ¬(∀ p a b c : ℝ, (∃! x y : ℝ, common_point p a b c x y) → is_tangent p a b c) :=
sorry

end NUMINAMATH_CALUDE_line_parabola_tangency_false_l4015_401522


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l4015_401533

/-- The price of an imported car after 5 years of annual reduction -/
def price_after_five_years (initial_price : ℝ) (annual_reduction_rate : ℝ) : ℝ :=
  initial_price * (1 - annual_reduction_rate)^5

/-- Theorem stating the relationship between the initial price, 
    annual reduction rate, and final price after 5 years -/
theorem price_reduction_theorem (x : ℝ) :
  price_after_five_years 300000 (x / 100) = 30000 * (1 - x / 100)^5 := by
  sorry

#check price_reduction_theorem

end NUMINAMATH_CALUDE_price_reduction_theorem_l4015_401533


namespace NUMINAMATH_CALUDE_tangent_condition_orthogonal_intersection_condition_l4015_401521

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

-- Define the line equation
def line_eq (m x y : ℝ) : Prop := x + m*y = 3

-- Define the tangency condition
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y ∧
  ∀ (x' y' : ℝ), circle_eq x' y' ∧ line_eq m x' y' → (x', y') = (x, y)

-- Define the intersection condition
def intersects_at_orthogonal_points (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    line_eq m x₁ y₁ ∧ line_eq m x₂ y₂ ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧
    x₁ * x₂ + y₁ * y₂ = 0

-- Theorem statements
theorem tangent_condition :
  ∀ m : ℝ, is_tangent m ↔ m = 7/24 :=
sorry

theorem orthogonal_intersection_condition :
  ∀ m : ℝ, intersects_at_orthogonal_points m ↔ (m = 9 + 2*Real.sqrt 14 ∨ m = 9 - 2*Real.sqrt 14) :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_orthogonal_intersection_condition_l4015_401521


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l4015_401509

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (positive : a > 0 ∧ b > 0 ∧ c > 0)

-- Define the angle bisector property
def has_equal_angle_bisector_segments (t : Triangle) : Prop :=
  ∃ (d e f : ℝ), d > 0 ∧ e > 0 ∧ f > 0 ∧ d = e

-- Main theorem
theorem angle_bisector_theorem (t : Triangle) 
  (h : has_equal_angle_bisector_segments t) : 
  (t.a / (t.b + t.c) = t.b / (t.c + t.a) + t.c / (t.a + t.b)) ∧ 
  (Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c)) > Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l4015_401509


namespace NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l4015_401541

theorem three_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l4015_401541


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l4015_401532

-- Define the value of a banana in terms of oranges
def banana_value (banana_count : ℚ) (orange_count : ℕ) : Prop :=
  banana_count * (15 / 12) = orange_count

-- Theorem statement
theorem banana_orange_equivalence :
  banana_value (4 / 5 * 15) 12 →
  banana_value (3 / 4 * 8) 6 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l4015_401532


namespace NUMINAMATH_CALUDE_ellipse_isosceles_triangle_existence_l4015_401528

/-- Ellipse C with equation x²/9 + y²/8 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

/-- Line l passing through point P(0, 2) with slope k -/
def line_l (k x y : ℝ) : Prop := y = k*x + 2

/-- Point D on the x-axis -/
def point_D (m : ℝ) : Prop := ∃ y, y = 0

/-- Isosceles triangle condition -/
def isosceles_triangle (xA yA xB yB xD : ℝ) : Prop :=
  (xA - xD)^2 + yA^2 = (xB - xD)^2 + yB^2

theorem ellipse_isosceles_triangle_existence :
  ∀ k > 0,
  ∃ xA yA xB yB m,
    ellipse_C xA yA ∧
    ellipse_C xB yB ∧
    line_l k xA yA ∧
    line_l k xB yB ∧
    point_D m ∧
    isosceles_triangle xA yA xB yB m ∧
    -Real.sqrt 2 / 12 ≤ m ∧
    m < 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_isosceles_triangle_existence_l4015_401528


namespace NUMINAMATH_CALUDE_perimeter_triangle_pst_l4015_401593

/-- Given a triangle PQR with points S on PQ, T on PR, and U on ST, 
    prove that the perimeter of triangle PST is 36 under specific conditions. -/
theorem perimeter_triangle_pst (P Q R S T U : ℝ × ℝ) : 
  dist P Q = 19 →
  dist Q R = 18 →
  dist P R = 17 →
  ∃ t₁ : ℝ, S = (1 - t₁) • P + t₁ • Q →
  ∃ t₂ : ℝ, T = (1 - t₂) • P + t₂ • R →
  ∃ t₃ : ℝ, U = (1 - t₃) • S + t₃ • T →
  dist Q S = dist S U →
  dist U T = dist T R →
  dist P S + dist S T + dist P T = 36 :=
sorry

end NUMINAMATH_CALUDE_perimeter_triangle_pst_l4015_401593


namespace NUMINAMATH_CALUDE_twenty_point_circle_special_chords_l4015_401530

/-- A circle with equally spaced points on its circumference -/
structure PointedCircle where
  n : ℕ  -- number of points
  (n_pos : n > 0)

/-- Counts chords in a PointedCircle satisfying certain length conditions -/
def count_special_chords (c : PointedCircle) : ℕ :=
  sorry

/-- Theorem statement -/
theorem twenty_point_circle_special_chords :
  ∃ (c : PointedCircle), c.n = 20 ∧ count_special_chords c = 120 :=
sorry

end NUMINAMATH_CALUDE_twenty_point_circle_special_chords_l4015_401530


namespace NUMINAMATH_CALUDE_product_equals_fraction_l4015_401531

theorem product_equals_fraction : 12 * 0.5 * 3 * 0.2 = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l4015_401531


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4015_401570

/-- Given a > 0 and a ≠ 1, prove that the function f(x) = 2 - a^(x+1) always passes through the point (-1, 1) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ 2 - a^(x + 1)
  f (-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4015_401570


namespace NUMINAMATH_CALUDE_sum_product_bounds_l4015_401591

theorem sum_product_bounds (a b c : ℝ) (h : a + b + c = 3) :
  ∃ (lower_bound upper_bound : ℝ),
    lower_bound = -9/2 ∧
    upper_bound = 3 ∧
    (∀ ε > 0, ∃ (x y z : ℝ), x + y + z = 3 ∧ x*y + x*z + y*z < lower_bound + ε) ∧
    (∀ (x y z : ℝ), x + y + z = 3 → x*y + x*z + y*z ≤ upper_bound) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l4015_401591


namespace NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l4015_401506

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point is the incenter of a triangle -/
def is_incenter (I : Point) (t : Triangle) : Prop := sorry

/-- Checks if a point is the foot of the altitude from C to AB -/
def is_altitude_foot (H : Point) (t : Triangle) : Prop := sorry

/-- Checks if a point is the excenter opposite to C -/
def is_excenter_C (I_C : Point) (t : Triangle) : Prop := sorry

/-- Checks if the excenter touches side AB and extensions of AC and BC -/
def excenter_touches_sides (I_C : Point) (t : Triangle) : Prop := sorry

theorem triangle_reconstruction_uniqueness 
  (I H I_C : Point) : 
  ∃! t : Triangle, 
    is_incenter I t ∧ 
    is_altitude_foot H t ∧ 
    is_excenter_C I_C t ∧ 
    excenter_touches_sides I_C t :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l4015_401506


namespace NUMINAMATH_CALUDE_zero_multiple_of_all_integers_l4015_401515

theorem zero_multiple_of_all_integers : ∀ (n : ℤ), ∃ (k : ℤ), 0 = k * n := by
  sorry

end NUMINAMATH_CALUDE_zero_multiple_of_all_integers_l4015_401515


namespace NUMINAMATH_CALUDE_gcd_product_is_square_l4015_401553

theorem gcd_product_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, (Nat.gcd x y).gcd z * x * y * z = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_product_is_square_l4015_401553


namespace NUMINAMATH_CALUDE_joan_seashells_l4015_401529

/-- Calculates the number of seashells Joan has after giving some away -/
def remaining_seashells (found : ℕ) (given_away : ℕ) : ℕ :=
  found - given_away

/-- Proves that Joan has 16 seashells after finding 79 and giving away 63 -/
theorem joan_seashells : remaining_seashells 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l4015_401529


namespace NUMINAMATH_CALUDE_tricycle_count_l4015_401585

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) (walking_children : ℕ) :
  total_children = 10 →
  total_wheels = 24 →
  walking_children = 2 →
  ∃ (bicycles tricycles : ℕ),
    bicycles + tricycles + walking_children = total_children ∧
    2 * bicycles + 3 * tricycles = total_wheels ∧
    tricycles = 8 :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l4015_401585


namespace NUMINAMATH_CALUDE_expression_simplification_l4015_401520

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1/m) / ((m^2 - 2*m + 1) / m) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4015_401520


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l4015_401597

theorem solution_to_linear_equation :
  ∃ x : ℝ, 3 * x - 6 = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l4015_401597


namespace NUMINAMATH_CALUDE_abs_d_equals_three_l4015_401501

/-- A polynomial with integer coefficients that has 3+i as a root -/
def f (a b c d : ℤ) : ℂ → ℂ := λ x => a*x^5 + b*x^4 + c*x^3 + d*x^2 + b*x + a

/-- The theorem stating that under given conditions, |d| = 3 -/
theorem abs_d_equals_three (a b c d : ℤ) : 
  f a b c d (3 + I) = 0 → 
  Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs = 1 → 
  d.natAbs = 3 := by sorry

end NUMINAMATH_CALUDE_abs_d_equals_three_l4015_401501


namespace NUMINAMATH_CALUDE_tea_cost_price_l4015_401561

/-- The cost price per kg of the 80 kg of tea -/
def C : ℝ := sorry

/-- The total weight of the tea mixture in kg -/
def total_weight : ℝ := 100

/-- The weight of the tea with unknown cost price in kg -/
def unknown_weight : ℝ := 80

/-- The weight of the tea with known cost price in kg -/
def known_weight : ℝ := 20

/-- The cost price per kg of the known tea in dollars -/
def known_cost : ℝ := 20

/-- The sale price per kg of the mixed tea in dollars -/
def sale_price : ℝ := 20.8

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.30

theorem tea_cost_price : C = 15 := by
  sorry

end NUMINAMATH_CALUDE_tea_cost_price_l4015_401561


namespace NUMINAMATH_CALUDE_sin_plus_tan_10_deg_l4015_401580

theorem sin_plus_tan_10_deg : 
  Real.sin (10 * π / 180) + (Real.sqrt 3 / 4) * Real.tan (10 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_tan_10_deg_l4015_401580


namespace NUMINAMATH_CALUDE_walter_sticker_distribution_l4015_401572

/-- Miss Walter's sticker distribution problem -/
theorem walter_sticker_distribution 
  (gold : ℕ) 
  (silver : ℕ) 
  (bronze : ℕ) 
  (students : ℕ) :
  gold = 50 →
  silver = 2 * gold →
  bronze = silver - 20 →
  students = 5 →
  (gold + silver + bronze) / students = 46 :=
by sorry

end NUMINAMATH_CALUDE_walter_sticker_distribution_l4015_401572


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4015_401581

/-- 
Given a hyperbola with equation (x²/a² - y²/b² = 1) that passes through the point (√2, √3) 
and has eccentricity 2, prove that its equation is x² - y²/3 = 1.
-/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (2 / a^2 - 3 / b^2 = 1) →  -- The hyperbola passes through (√2, √3)
  ((a^2 + b^2) / a^2 = 4) →  -- The eccentricity is 2
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4015_401581


namespace NUMINAMATH_CALUDE_min_value_sum_l4015_401564

theorem min_value_sum (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) :
  (1/a) + (9/b) + (16/c) + (25/d) + (36/e) + (49/f) ≥ 676/9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l4015_401564


namespace NUMINAMATH_CALUDE_droid_weekly_usage_l4015_401574

/-- Represents the daily coffee bean usage in Droid's coffee shop -/
structure DailyUsage where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- Calculates the total daily usage -/
def totalDailyUsage (usage : DailyUsage) : ℕ :=
  usage.morning + usage.afternoon + usage.evening

/-- Represents the weekly coffee bean usage in Droid's coffee shop -/
structure WeeklyUsage where
  weekday : DailyUsage
  saturday : DailyUsage
  sunday : DailyUsage

/-- Calculates the total weekly usage -/
def totalWeeklyUsage (usage : WeeklyUsage) : ℕ :=
  5 * totalDailyUsage usage.weekday + totalDailyUsage usage.saturday + totalDailyUsage usage.sunday

/-- The coffee bean usage pattern for Droid's coffee shop -/
def droidUsage : WeeklyUsage where
  weekday := { morning := 3, afternoon := 9, evening := 6 }
  saturday := { morning := 4, afternoon := 8, evening := 6 }
  sunday := { morning := 2, afternoon := 2, evening := 2 }

theorem droid_weekly_usage : totalWeeklyUsage droidUsage = 114 := by
  sorry

end NUMINAMATH_CALUDE_droid_weekly_usage_l4015_401574


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l4015_401517

/-- The first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

/-- The last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem stating that the sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l4015_401517


namespace NUMINAMATH_CALUDE_vacation_cost_theorem_l4015_401545

/-- Calculates the total cost of a vacation in USD given specific expenses and exchange rates -/
def vacation_cost (num_people : ℕ) 
                  (rent_per_person : ℝ) 
                  (transport_per_person : ℝ) 
                  (food_per_person : ℝ) 
                  (activities_per_person : ℝ) 
                  (euro_to_usd : ℝ) 
                  (pound_to_usd : ℝ) 
                  (yen_to_usd : ℝ) : ℝ :=
  let total_rent := num_people * rent_per_person * euro_to_usd
  let total_transport := num_people * transport_per_person
  let total_food := num_people * food_per_person * pound_to_usd
  let total_activities := num_people * activities_per_person * yen_to_usd
  total_rent + total_transport + total_food + total_activities

/-- The total cost of the vacation is $1384.25 -/
theorem vacation_cost_theorem : 
  vacation_cost 7 65 25 50 2750 1.2 1.4 0.009 = 1384.25 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_theorem_l4015_401545


namespace NUMINAMATH_CALUDE_inequalities_proof_l4015_401500

theorem inequalities_proof (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 4) : 
  (1 / a + 1 / (b + 1) ≥ 4 / 5) ∧ 
  (4 / (a * b) + a / b ≥ (Real.sqrt 5 + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l4015_401500


namespace NUMINAMATH_CALUDE_smallest_number_l4015_401566

theorem smallest_number : 
  let numbers : List ℚ := [0, (-3)^2, |-9|, -1^4]
  (∀ x ∈ numbers, -1^4 ≤ x) ∧ (-1^4 ∈ numbers) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l4015_401566


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l4015_401560

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 5)^2 + (y - 3)^2 + (z - 1)^2 = 0 → x + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l4015_401560


namespace NUMINAMATH_CALUDE_work_completion_time_l4015_401578

theorem work_completion_time (total_work : ℝ) (raja_rate : ℝ) (ram_rate : ℝ) :
  raja_rate + ram_rate = total_work / 4 →
  raja_rate = total_work / 12 →
  ram_rate = total_work / 6 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l4015_401578


namespace NUMINAMATH_CALUDE_tanks_needed_l4015_401569

def existing_tanks : ℕ := 3
def existing_capacity : ℕ := 15
def new_capacity : ℕ := 10
def total_fish : ℕ := 75

theorem tanks_needed : 
  (total_fish - existing_tanks * existing_capacity) / new_capacity = 3 := by
  sorry

end NUMINAMATH_CALUDE_tanks_needed_l4015_401569


namespace NUMINAMATH_CALUDE_midSectionAreaProperty_l4015_401559

-- Define a right triangular pyramid
structure RightTriangularPyramid where
  -- We don't need to define all properties, just the essential ones for our theorem
  obliqueFace : Set (ℝ × ℝ)  -- Representing the base as a set of points in 2D
  midSection : Set (ℝ × ℝ)   -- Representing a mid-section as a set of points in 2D

-- Define the area function
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem midSectionAreaProperty (p : RightTriangularPyramid) :
  area p.midSection = (1/4) * area p.obliqueFace := by sorry

end NUMINAMATH_CALUDE_midSectionAreaProperty_l4015_401559


namespace NUMINAMATH_CALUDE_edith_total_books_first_shelf_novels_relation_writing_books_relation_l4015_401510

/-- The number of novels on the second shelf -/
def novels_second_shelf : ℕ := 56

/-- The number of novels on the first shelf -/
def novels_first_shelf : ℕ := 67

/-- The total number of novels on both shelves -/
def total_novels : ℕ := novels_first_shelf + novels_second_shelf

/-- The number of writing books -/
def writing_books : ℕ := 62

/-- The total number of books Edith has -/
def total_books : ℕ := total_novels + writing_books

/-- Theorem stating the total number of books Edith has -/
theorem edith_total_books : total_books = 185 := by
  sorry

/-- Theorem stating the relationship between novels on first and second shelves -/
theorem first_shelf_novels_relation : 
  novels_first_shelf ≥ novels_second_shelf ∧ 
  novels_first_shelf ≤ (novels_second_shelf * 6 + 2) / 5 := by
  sorry

/-- Theorem stating the relationship between writing books and total novels -/
theorem writing_books_relation :
  writing_books ≥ total_novels / 2 ∧ 
  writing_books ≤ (total_novels + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_edith_total_books_first_shelf_novels_relation_writing_books_relation_l4015_401510


namespace NUMINAMATH_CALUDE_fraction_product_l4015_401514

theorem fraction_product : (2 : ℚ) / 9 * 5 / 11 = 10 / 99 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l4015_401514


namespace NUMINAMATH_CALUDE_mans_upstream_speed_l4015_401586

/-- Given a man's downstream speed and still water speed, calculate his upstream speed -/
theorem mans_upstream_speed (downstream_speed still_water_speed : ℝ) 
  (h1 : downstream_speed = 20)
  (h2 : still_water_speed = 15) :
  still_water_speed - (downstream_speed - still_water_speed) = 10 := by
  sorry

#check mans_upstream_speed

end NUMINAMATH_CALUDE_mans_upstream_speed_l4015_401586


namespace NUMINAMATH_CALUDE_cubic_minus_xy_squared_factorization_l4015_401502

theorem cubic_minus_xy_squared_factorization (x y : ℝ) :
  x^3 - x*y^2 = x*(x+y)*(x-y) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_xy_squared_factorization_l4015_401502


namespace NUMINAMATH_CALUDE_common_factor_proof_l4015_401595

theorem common_factor_proof (n : ℤ) : ∃ (k₁ k₂ : ℤ), 
  n^2 - 1 = (n + 1) * k₁ ∧ n^2 + n = (n + 1) * k₂ := by
  sorry

end NUMINAMATH_CALUDE_common_factor_proof_l4015_401595


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l4015_401567

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 10)^2 - 11*(a 10) + 16 = 0 →
  (a 30)^2 - 11*(a 30) + 16 = 0 →
  a 20 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l4015_401567


namespace NUMINAMATH_CALUDE_joe_remaining_money_l4015_401575

theorem joe_remaining_money (pocket_money : ℚ) (chocolate_fraction : ℚ) (fruit_fraction : ℚ) :
  pocket_money = 450 ∧
  chocolate_fraction = 1/9 ∧
  fruit_fraction = 2/5 →
  pocket_money - (chocolate_fraction * pocket_money + fruit_fraction * pocket_money) = 220 :=
by sorry

end NUMINAMATH_CALUDE_joe_remaining_money_l4015_401575


namespace NUMINAMATH_CALUDE_no_negative_roots_l4015_401540

theorem no_negative_roots : 
  ∀ x : ℝ, x^4 - 4*x^3 - 6*x^2 - 3*x + 9 = 0 → x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l4015_401540


namespace NUMINAMATH_CALUDE_B_power_100_is_identity_l4015_401589

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]]

theorem B_power_100_is_identity :
  B ^ 100 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_100_is_identity_l4015_401589


namespace NUMINAMATH_CALUDE_winning_condition_l4015_401573

/-- Represents a chessboard of size n × n -/
structure Chessboard (n : ℕ) where
  size : n > 0

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- Represents the result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- The game played on a chessboard -/
def game (n : ℕ) (board : Chessboard n) : GameResult := sorry

/-- Theorem stating the winning condition based on the parity of n -/
theorem winning_condition (n : ℕ) (board : Chessboard n) :
  game n board = GameResult.FirstPlayerWins ↔ Even n := by sorry

end NUMINAMATH_CALUDE_winning_condition_l4015_401573


namespace NUMINAMATH_CALUDE_jennys_change_l4015_401592

/-- The problem of calculating Jenny's change --/
theorem jennys_change 
  (cost_per_page : ℚ)
  (num_copies : ℕ)
  (pages_per_essay : ℕ)
  (num_pens : ℕ)
  (cost_per_pen : ℚ)
  (payment : ℚ)
  (h1 : cost_per_page = 1/10)
  (h2 : num_copies = 7)
  (h3 : pages_per_essay = 25)
  (h4 : num_pens = 7)
  (h5 : cost_per_pen = 3/2)
  (h6 : payment = 40) :
  payment - (cost_per_page * num_copies * pages_per_essay + cost_per_pen * num_pens) = 12 := by
  sorry


end NUMINAMATH_CALUDE_jennys_change_l4015_401592


namespace NUMINAMATH_CALUDE_function_properties_l4015_401526

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem function_properties (f : ℝ → ℝ)
  (h1 : is_odd (λ x ↦ f (x + 2)))
  (h2 : ∀ x₁ x₂, x₁ ∈ Set.Ici 2 → x₂ ∈ Set.Ici 2 → x₁ < x₂ → (f x₂ - f x₁) / (x₂ - x₁) > 0) :
  (∀ x y, x < y → f x < f y) ∧
  {x : ℝ | f x < 0} = Set.Iio 2 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4015_401526


namespace NUMINAMATH_CALUDE_only_1680_is_product_of_four_consecutive_l4015_401571

/-- Given a natural number n, returns true if n can be written as a product of four consecutive natural numbers -/
def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) * (k + 2) * (k + 3)

/-- The theorem stating that among 712, 1262, and 1680, only 1680 can be written as a product of four consecutive natural numbers -/
theorem only_1680_is_product_of_four_consecutive :
  is_product_of_four_consecutive 1680 ∧
  ¬is_product_of_four_consecutive 712 ∧
  ¬is_product_of_four_consecutive 1262 :=
by sorry

end NUMINAMATH_CALUDE_only_1680_is_product_of_four_consecutive_l4015_401571


namespace NUMINAMATH_CALUDE_keith_turnips_l4015_401551

theorem keith_turnips (total : ℕ) (alyssa : ℕ) (keith : ℕ)
  (h1 : total = 15)
  (h2 : alyssa = 9)
  (h3 : keith + alyssa = total) :
  keith = 15 - 9 :=
by sorry

end NUMINAMATH_CALUDE_keith_turnips_l4015_401551


namespace NUMINAMATH_CALUDE_circle_C_tangent_line_l_line_AB_passes_through_intersection_l4015_401563

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = 3

-- Define circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define line AB
def line_AB (x y : ℝ) : Prop := 2*x + y - 4 = 0

-- Theorem 1: Circle C is tangent to line l
theorem circle_C_tangent_line_l : ∃ (x y : ℝ), circle_C x y ∧ line_l x := by sorry

-- Theorem 2: Line AB passes through the intersection of circles C and O
theorem line_AB_passes_through_intersection :
  ∀ (x y : ℝ), (circle_C x y ∧ circle_O x y) → line_AB x y := by sorry

end NUMINAMATH_CALUDE_circle_C_tangent_line_l_line_AB_passes_through_intersection_l4015_401563


namespace NUMINAMATH_CALUDE_problem_solution_l4015_401590

theorem problem_solution (m n : ℝ) 
  (hm : m^2 - 2*m - 1 = 0) 
  (hn : n^2 + 2*n - 1 = 0) 
  (hmn : m*n ≠ 1) : 
  (m*n + n + 1) / n = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4015_401590


namespace NUMINAMATH_CALUDE_complex_absolute_value_l4015_401544

theorem complex_absolute_value : Complex.abs ((2 : ℂ) + (2 * Complex.I * Real.sqrt 2)) ^ 6 = 576 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l4015_401544


namespace NUMINAMATH_CALUDE_mississippi_permutations_count_l4015_401582

/-- The number of unique permutations of MISSISSIPPI -/
def mississippi_permutations : ℕ :=
  Nat.factorial 11 / (Nat.factorial 1 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 2)

/-- Theorem stating that the number of unique permutations of MISSISSIPPI is 34650 -/
theorem mississippi_permutations_count : mississippi_permutations = 34650 := by
  sorry

end NUMINAMATH_CALUDE_mississippi_permutations_count_l4015_401582


namespace NUMINAMATH_CALUDE_balloon_difference_l4015_401542

theorem balloon_difference (x y : ℚ) 
  (eq1 : x = 2 * y - 3)
  (eq2 : y = x / 4 + 1) : 
  x - y = -5/2 := by sorry

end NUMINAMATH_CALUDE_balloon_difference_l4015_401542


namespace NUMINAMATH_CALUDE_range_of_a_l4015_401546

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, Real.exp x - a ≥ 0) → a ≤ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4015_401546


namespace NUMINAMATH_CALUDE_eight_custom_op_eight_eq_four_l4015_401577

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a * b) / (a + b)

/-- Theorem stating that 8 @ 8 = 4 -/
theorem eight_custom_op_eight_eq_four :
  custom_op 8 8 = 4 := by sorry

end NUMINAMATH_CALUDE_eight_custom_op_eight_eq_four_l4015_401577


namespace NUMINAMATH_CALUDE_existence_of_coprime_sum_l4015_401556

theorem existence_of_coprime_sum (n k : ℕ+) 
  (h : n.val % 2 = 1 ∨ (n.val % 2 = 0 ∧ k.val % 2 = 0)) :
  ∃ a b : ℤ, Nat.gcd a.natAbs n.val = 1 ∧ 
             Nat.gcd b.natAbs n.val = 1 ∧ 
             k.val = a + b := by
  sorry

end NUMINAMATH_CALUDE_existence_of_coprime_sum_l4015_401556


namespace NUMINAMATH_CALUDE_time_to_clean_wall_l4015_401547

/-- Represents the dimensions of the wall in large squares -/
structure WallDimensions where
  height : ℕ
  width : ℕ

/-- Represents the cleaning progress and rate -/
structure CleaningProgress where
  totalArea : ℕ
  cleanedArea : ℕ
  timeSpent : ℕ

/-- Calculates the time needed to clean the remaining area -/
def timeToCleanRemaining (wall : WallDimensions) (progress : CleaningProgress) : ℕ :=
  let remainingArea := wall.height * wall.width - progress.cleanedArea
  (remainingArea * progress.timeSpent) / progress.cleanedArea

/-- Theorem: Given the wall dimensions and cleaning progress, 
    the time to clean the remaining area is 161 minutes -/
theorem time_to_clean_wall 
  (wall : WallDimensions) 
  (progress : CleaningProgress) 
  (h1 : wall.height = 6) 
  (h2 : wall.width = 12) 
  (h3 : progress.totalArea = wall.height * wall.width)
  (h4 : progress.cleanedArea = 9)
  (h5 : progress.timeSpent = 23) :
  timeToCleanRemaining wall progress = 161 := by
  sorry

end NUMINAMATH_CALUDE_time_to_clean_wall_l4015_401547


namespace NUMINAMATH_CALUDE_voldemort_shopping_l4015_401555

theorem voldemort_shopping (book_price : ℝ) (journal_price : ℝ) : 
  book_price = 8 ∧ 
  book_price = (1/8) * (book_price * 8) ∧ 
  journal_price = 2 * book_price →
  (book_price * 8 = 64) ∧ 
  (book_price + journal_price = 24) := by
sorry

end NUMINAMATH_CALUDE_voldemort_shopping_l4015_401555


namespace NUMINAMATH_CALUDE_three_positions_from_eight_people_l4015_401598

def number_of_people : ℕ := 8
def number_of_positions : ℕ := 3

theorem three_positions_from_eight_people :
  (number_of_people.factorial) / ((number_of_people - number_of_positions).factorial) = 336 :=
sorry

end NUMINAMATH_CALUDE_three_positions_from_eight_people_l4015_401598


namespace NUMINAMATH_CALUDE_age_difference_l4015_401587

/-- Represents the ages of Linda and Jane -/
structure Ages where
  linda : ℕ
  jane : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.linda = 13 ∧
  ages.linda + ages.jane + 10 = 28 ∧
  ages.linda > 2 * ages.jane

/-- The theorem to prove -/
theorem age_difference (ages : Ages) :
  problem_conditions ages →
  ages.linda - 2 * ages.jane = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l4015_401587


namespace NUMINAMATH_CALUDE_f_range_l4015_401588

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sin x - 1

theorem f_range : Set.range f = Set.Icc (-5/4 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l4015_401588
