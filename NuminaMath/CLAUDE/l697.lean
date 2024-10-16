import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l697_69764

theorem hyperbola_asymptotes (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) →
  (2 * Real.sqrt (-1/m) = 4) →
  (∀ x y : ℝ, y = 2*x ∨ y = -2*x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l697_69764


namespace NUMINAMATH_CALUDE_merged_class_size_is_41_l697_69774

/-- Represents a group of students with a specific student's position --/
structure StudentGroup where
  right_rank : Nat
  left_rank : Nat

/-- Calculates the total number of students in a group --/
def group_size (g : StudentGroup) : Nat :=
  g.right_rank - 1 + g.left_rank

/-- Calculates the total number of students in the merged class --/
def merged_class_size (group_a group_b : StudentGroup) : Nat :=
  group_size group_a + group_size group_b

/-- Theorem stating the total number of students in the merged class --/
theorem merged_class_size_is_41 :
  let group_a : StudentGroup := ⟨13, 8⟩
  let group_b : StudentGroup := ⟨10, 12⟩
  merged_class_size group_a group_b = 41 := by
  sorry

#eval merged_class_size ⟨13, 8⟩ ⟨10, 12⟩

end NUMINAMATH_CALUDE_merged_class_size_is_41_l697_69774


namespace NUMINAMATH_CALUDE_digit_1997_of_1_22_digit_1997_of_1_27_l697_69717

/-- The nth decimal digit of a rational number -/
def nthDecimalDigit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- The 1997th decimal digit of 1/22 is 0 -/
theorem digit_1997_of_1_22 : nthDecimalDigit (1/22) 1997 = 0 := by sorry

/-- The 1997th decimal digit of 1/27 is 3 -/
theorem digit_1997_of_1_27 : nthDecimalDigit (1/27) 1997 = 3 := by sorry

end NUMINAMATH_CALUDE_digit_1997_of_1_22_digit_1997_of_1_27_l697_69717


namespace NUMINAMATH_CALUDE_four_digit_number_with_zero_removal_l697_69787

/-- Represents a four-digit number with one digit being zero -/
structure FourDigitNumberWithZero where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_less_than_10 : a < 10
  b_less_than_10 : b < 10
  c_less_than_10 : c < 10
  d_less_than_10 : d < 10
  has_one_zero : (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∨
                 (a ≠ 0 ∧ b = 0 ∧ c ≠ 0 ∧ d ≠ 0) ∨
                 (a ≠ 0 ∧ b ≠ 0 ∧ c = 0 ∧ d ≠ 0) ∨
                 (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d = 0)

/-- The value of the four-digit number -/
def value (n : FourDigitNumberWithZero) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The value of the number after removing the zero -/
def valueWithoutZero (n : FourDigitNumberWithZero) : Nat :=
  if n.a = 0 then 100 * n.b + 10 * n.c + n.d
  else if n.b = 0 then 100 * n.a + 10 * n.c + n.d
  else if n.c = 0 then 100 * n.a + 10 * n.b + n.d
  else 100 * n.a + 10 * n.b + n.c

theorem four_digit_number_with_zero_removal (n : FourDigitNumberWithZero) :
  (value n = 9 * valueWithoutZero n) → (value n = 2025 ∨ value n = 6075) := by
  sorry


end NUMINAMATH_CALUDE_four_digit_number_with_zero_removal_l697_69787


namespace NUMINAMATH_CALUDE_right_triangle_third_vertex_l697_69779

theorem right_triangle_third_vertex 
  (v1 : ℝ × ℝ) 
  (v2 : ℝ × ℝ) 
  (x : ℝ) :
  v1 = (4, 3) →
  v2 = (0, 0) →
  x > 0 →
  (1/2 : ℝ) * x * 3 = 24 →
  x = 16 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_vertex_l697_69779


namespace NUMINAMATH_CALUDE_building_height_l697_69727

/-- Given a flagpole and a building casting shadows under similar conditions,
    this theorem proves that the height of the building is 22 meters. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 55)
  : (flagpole_height / flagpole_shadow) * building_shadow = 22 :=
by sorry

end NUMINAMATH_CALUDE_building_height_l697_69727


namespace NUMINAMATH_CALUDE_bert_spent_nine_at_dry_cleaners_l697_69797

/-- Represents Bert's spending problem --/
def BertSpending (initial_amount : ℚ) (dry_cleaner_amount : ℚ) : Prop :=
  let hardware_store := initial_amount / 4
  let after_hardware := initial_amount - hardware_store
  let after_dry_cleaner := after_hardware - dry_cleaner_amount
  let grocery_store := after_dry_cleaner / 2
  let final_amount := after_dry_cleaner - grocery_store
  (initial_amount = 44) ∧ (final_amount = 12)

/-- Theorem stating that Bert spent $9 at the dry cleaners --/
theorem bert_spent_nine_at_dry_cleaners :
  ∃ (dry_cleaner_amount : ℚ), BertSpending 44 dry_cleaner_amount ∧ dry_cleaner_amount = 9 := by
  sorry

end NUMINAMATH_CALUDE_bert_spent_nine_at_dry_cleaners_l697_69797


namespace NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l697_69715

theorem polygon_sides_and_diagonals (n : ℕ) : 
  n + (n * (n - 3)) / 2 = 77 → n = 14 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l697_69715


namespace NUMINAMATH_CALUDE_ellipse_max_sum_l697_69795

/-- Given an ellipse defined by x^2/4 + y^2/2 = 1, 
    the maximum value of |x| + |y| is 2√3. -/
theorem ellipse_max_sum (x y : ℝ) : 
  x^2/4 + y^2/2 = 1 → |x| + |y| ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_max_sum_l697_69795


namespace NUMINAMATH_CALUDE_inequality_solution_set_l697_69703

theorem inequality_solution_set (x : ℝ) : 
  (x ≠ 0 ∧ (x - 1) / x ≤ 0) ↔ (0 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l697_69703


namespace NUMINAMATH_CALUDE_molecular_weight_BaCl2_correct_l697_69768

/-- The molecular weight of BaCl2 in g/mol -/
def molecular_weight_BaCl2 : ℝ := 207

/-- The number of moles given in the problem -/
def given_moles : ℝ := 8

/-- The total weight of the given moles of BaCl2 in grams -/
def total_weight : ℝ := 1656

/-- Theorem stating that the molecular weight of BaCl2 is correct -/
theorem molecular_weight_BaCl2_correct :
  molecular_weight_BaCl2 = total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_BaCl2_correct_l697_69768


namespace NUMINAMATH_CALUDE_exist_three_quadratic_polynomials_l697_69780

theorem exist_three_quadratic_polynomials :
  ∃ (f g h : ℝ → ℝ),
    (∀ x, f x = (x - 3)^2 - 1) ∧
    (∀ x, g x = x^2 - 1) ∧
    (∀ x, h x = (x + 3)^2 - 1) ∧
    (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
    (∃ y₁ y₂, y₁ ≠ y₂ ∧ g y₁ = 0 ∧ g y₂ = 0) ∧
    (∃ z₁ z₂, z₁ ≠ z₂ ∧ h z₁ = 0 ∧ h z₂ = 0) ∧
    (∀ x, (f x + g x) ≠ 0) ∧
    (∀ x, (f x + h x) ≠ 0) ∧
    (∀ x, (g x + h x) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_exist_three_quadratic_polynomials_l697_69780


namespace NUMINAMATH_CALUDE_range_of_a_l697_69793

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (1 - a)^x < (1 - a)^y

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → (0 ≤ a ∧ a < 2) ∨ (a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l697_69793


namespace NUMINAMATH_CALUDE_basket_apples_theorem_l697_69749

/-- The total number of apples in the basket -/
def total_apples : ℕ := 5

/-- The probability of selecting at least one spoiled apple when picking 2 apples randomly -/
def prob_spoiled : ℚ := 2/5

/-- The number of spoiled apples in the basket -/
def spoiled_apples : ℕ := 1

/-- The number of good apples in the basket -/
def good_apples : ℕ := total_apples - spoiled_apples

theorem basket_apples_theorem :
  (total_apples = spoiled_apples + good_apples) ∧
  (prob_spoiled = 1 - (good_apples / total_apples) * ((good_apples - 1) / (total_apples - 1))) :=
by sorry

end NUMINAMATH_CALUDE_basket_apples_theorem_l697_69749


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l697_69789

/-- The perimeter of a regular hexagon with side length 5 cm is 30 cm. -/
theorem regular_hexagon_perimeter :
  ∀ (side_length : ℝ),
  side_length = 5 →
  (6 : ℝ) * side_length = 30 :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l697_69789


namespace NUMINAMATH_CALUDE_inequality_solution_set_l697_69738

theorem inequality_solution_set :
  {x : ℝ | (3 / 8 : ℝ) + |x - (1 / 4 : ℝ)| < (7 / 8 : ℝ)} = Set.Ioo (-(1 / 4 : ℝ)) ((3 / 4 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l697_69738


namespace NUMINAMATH_CALUDE_license_plate_count_l697_69765

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letter positions in the license plate -/
def num_letter_positions : ℕ := 3

/-- The number of digit positions in the license plate -/
def num_digit_positions : ℕ := 3

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_letters ^ num_letter_positions * num_digits ^ num_digit_positions

theorem license_plate_count : total_license_plates = 17576000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l697_69765


namespace NUMINAMATH_CALUDE_doris_babysitting_earnings_l697_69712

/-- Represents the problem of calculating how many weeks Doris needs to earn enough for her monthly expenses --/
theorem doris_babysitting_earnings :
  let hourly_rate : ℚ := 20
  let weekday_hours : ℚ := 3
  let saturday_hours : ℚ := 5
  let monthly_expense : ℚ := 1200
  let weekly_hours := weekday_hours * 5 + saturday_hours
  let weekly_earnings := hourly_rate * weekly_hours
  let weeks_needed := monthly_expense / weekly_earnings
  weeks_needed = 3 := by sorry

end NUMINAMATH_CALUDE_doris_babysitting_earnings_l697_69712


namespace NUMINAMATH_CALUDE_solution_sum_l697_69798

theorem solution_sum (p q : ℝ) : 
  (2^2 - 2*p + 6 = 0) → 
  (2^2 + 6*2 - q = 0) → 
  p + q = 21 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l697_69798


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l697_69750

theorem rope_cutting_problem :
  Nat.gcd 48 (Nat.gcd 72 (Nat.gcd 96 120)) = 24 := by sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l697_69750


namespace NUMINAMATH_CALUDE_tangent_line_condition_no_positive_max_for_negative_integer_a_l697_69732

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * Real.exp x) / x + x

theorem tangent_line_condition (a : ℝ) :
  (∃ (m b : ℝ), m * 1 + b = f a 1 ∧ m * (1 - 0) = f a 1 - (-1) ∧ 0 * m + b = -1) →
  a = -1 / Real.exp 1 := by
  sorry

theorem no_positive_max_for_negative_integer_a :
  ∀ a : ℤ, a < 0 →
  ¬∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a x ≥ f a y ∧ f a x > 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_condition_no_positive_max_for_negative_integer_a_l697_69732


namespace NUMINAMATH_CALUDE_absolute_value_equality_l697_69781

theorem absolute_value_equality (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) ↔ 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 0 ∧ c = 1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = -1)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l697_69781


namespace NUMINAMATH_CALUDE_weight_replacement_l697_69707

theorem weight_replacement (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 6 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l697_69707


namespace NUMINAMATH_CALUDE_specific_ellipse_equation_l697_69718

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  /-- The focal length of the ellipse -/
  focal_length : ℝ
  /-- The x-coordinate of one directrix -/
  directrix_x : ℝ

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 4 = 1

/-- Theorem stating the equation of the specific ellipse -/
theorem specific_ellipse_equation (e : Ellipse) 
  (h1 : e.focal_length = 4)
  (h2 : e.directrix_x = -4) :
  ∀ x y : ℝ, ellipse_equation e x y := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_equation_l697_69718


namespace NUMINAMATH_CALUDE_divisibility_constraint_l697_69746

theorem divisibility_constraint (m n : ℕ) : 
  m ≥ 1 → n ≥ 1 → 
  (m * n ∣ 3^m + 1) → 
  (m * n ∣ 3^n + 1) → 
  ((m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_constraint_l697_69746


namespace NUMINAMATH_CALUDE_shifted_function_l697_69759

def g (x : ℝ) : ℝ := 5 * x^2

def f (x : ℝ) : ℝ := 5 * (x - 3)^2 - 2

theorem shifted_function (x : ℝ) : 
  f x = g (x - 3) - 2 := by sorry

end NUMINAMATH_CALUDE_shifted_function_l697_69759


namespace NUMINAMATH_CALUDE_plan_y_more_cost_effective_l697_69773

/-- Represents the cost in cents for Plan X given m megabytes -/
def cost_plan_x (m : ℕ) : ℕ := 15 * m

/-- Represents the cost in cents for Plan Y given m megabytes -/
def cost_plan_y (m : ℕ) : ℕ := 3000 + 7 * m

/-- The minimum whole number of megabytes for Plan Y to be more cost-effective -/
def min_megabytes : ℕ := 376

theorem plan_y_more_cost_effective :
  (∀ m : ℕ, m ≥ min_megabytes → cost_plan_y m < cost_plan_x m) ∧
  (∀ m : ℕ, m < min_megabytes → cost_plan_y m ≥ cost_plan_x m) :=
sorry

end NUMINAMATH_CALUDE_plan_y_more_cost_effective_l697_69773


namespace NUMINAMATH_CALUDE_pebble_ratio_l697_69705

def total_pebbles : ℕ := 30
def white_pebbles : ℕ := 20

def red_pebbles : ℕ := total_pebbles - white_pebbles

theorem pebble_ratio : 
  (red_pebbles : ℚ) / white_pebbles = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_pebble_ratio_l697_69705


namespace NUMINAMATH_CALUDE_logical_equivalence_l697_69724

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ ¬R) → ¬Q) ↔ (Q → (¬P ∨ R)) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l697_69724


namespace NUMINAMATH_CALUDE_luke_stickers_to_sister_l697_69744

/-- Calculates the number of stickers Luke gave to his sister -/
def stickers_given_to_sister (initial : ℕ) (bought : ℕ) (birthday : ℕ) (used : ℕ) (final : ℕ) : ℕ :=
  initial + bought + birthday - used - final

/-- Theorem stating the number of stickers Luke gave to his sister -/
theorem luke_stickers_to_sister :
  stickers_given_to_sister 20 12 20 8 39 = 5 := by
  sorry

end NUMINAMATH_CALUDE_luke_stickers_to_sister_l697_69744


namespace NUMINAMATH_CALUDE_point_transformation_l697_69755

def rotate_180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (c d : ℝ) :
  let (x₁, y₁) := rotate_180 c d 2 3
  let (x₂, y₂) := reflect_y_eq_x x₁ y₁
  (x₂ = 2 ∧ y₂ = -1) → d - c = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l697_69755


namespace NUMINAMATH_CALUDE_linear_function_point_relation_l697_69790

/-- Given a linear function f(x) = -x + b, prove that if P₁(-1, y₁) and P₂(2, y₂) 
    are points on the graph of f, then y₁ > y₂ -/
theorem linear_function_point_relation (b : ℝ) (y₁ y₂ : ℝ) 
    (h₁ : y₁ = -(-1) + b) 
    (h₂ : y₂ = -(2) + b) : 
  y₁ > y₂ := by
  sorry

#check linear_function_point_relation

end NUMINAMATH_CALUDE_linear_function_point_relation_l697_69790


namespace NUMINAMATH_CALUDE_opposite_sides_difference_equal_l697_69706

/-- An equiangular hexagon with sides a, b, c, d, e, f in order -/
structure EquiangularHexagon where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  equiangular : True  -- This represents the equiangular property

/-- The differences between opposite sides in an equiangular hexagon are equal -/
theorem opposite_sides_difference_equal (h : EquiangularHexagon) :
  h.a - h.d = h.e - h.b ∧ h.e - h.b = h.c - h.f :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_difference_equal_l697_69706


namespace NUMINAMATH_CALUDE_largest_rational_satisfying_equation_l697_69716

theorem largest_rational_satisfying_equation :
  ∀ x : ℚ, |x - 7/2| = 25/2 → x ≤ 16 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_rational_satisfying_equation_l697_69716


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_1_quadratic_inequality_theorem_2_quadratic_inequality_theorem_3_l697_69747

-- Define the quadratic inequality
def quadratic_inequality (k x : ℝ) : Prop := k * x^2 - 2 * x + 6 * k < 0

-- Define the solution sets
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2
def solution_set_2 (k x : ℝ) : Prop := x ≠ 1 / k
def solution_set_3 : Set ℝ := ∅

-- Theorem statements
theorem quadratic_inequality_theorem_1 (k : ℝ) :
  k ≠ 0 →
  (∀ x, quadratic_inequality k x ↔ solution_set_1 x) →
  k = -2/5 :=
sorry

theorem quadratic_inequality_theorem_2 (k : ℝ) :
  k ≠ 0 →
  (∀ x, quadratic_inequality k x ↔ solution_set_2 k x) →
  k = -Real.sqrt 6 / 6 :=
sorry

theorem quadratic_inequality_theorem_3 (k : ℝ) :
  k ≠ 0 →
  (∀ x, ¬quadratic_inequality k x) →
  k ≥ Real.sqrt 6 / 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_1_quadratic_inequality_theorem_2_quadratic_inequality_theorem_3_l697_69747


namespace NUMINAMATH_CALUDE_cubic_inequality_l697_69745

theorem cubic_inequality (a b c : ℝ) :
  a^6 + b^6 + c^6 - 3*a^2*b^2*c^2 ≥ (1/2) * (a-b)^2 * (b-c)^2 * (c-a)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l697_69745


namespace NUMINAMATH_CALUDE_smallest_configuration_l697_69748

/-- A configuration of points on a plane where each point is 1 unit away from exactly four others -/
structure PointConfiguration where
  n : ℕ
  points : Fin n → ℝ × ℝ
  distinct : ∀ i j, i ≠ j → points i ≠ points j
  distance_condition : ∀ i, ∃ s : Finset (Fin n), s.card = 4 ∧ 
    ∀ j ∈ s, (i ≠ j) ∧ Real.sqrt (((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2) = 1

/-- The smallest possible number of points in a valid configuration is 9 -/
theorem smallest_configuration : 
  (∃ c : PointConfiguration, c.n = 9) ∧ 
  (∀ c : PointConfiguration, c.n ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_smallest_configuration_l697_69748


namespace NUMINAMATH_CALUDE_max_value_of_quadratic_l697_69776

/-- The quadratic function we're considering -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The range of x values we're considering -/
def range : Set ℝ := { x | -5 ≤ x ∧ x ≤ 3 }

theorem max_value_of_quadratic :
  ∃ (m : ℝ), m = 36 ∧ ∀ x ∈ range, f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_quadratic_l697_69776


namespace NUMINAMATH_CALUDE_raccoon_stall_time_l697_69736

/-- Proves that the time both locks together stall the raccoons is 60 minutes -/
theorem raccoon_stall_time : ∀ (t1 t2 t_both : ℕ),
  t1 = 5 →
  t2 = 3 * t1 - 3 →
  t_both = 5 * t2 →
  t_both = 60 := by
  sorry

end NUMINAMATH_CALUDE_raccoon_stall_time_l697_69736


namespace NUMINAMATH_CALUDE_nabla_squared_l697_69734

theorem nabla_squared (odot nabla : ℕ) : 
  odot < 20 ∧ nabla < 20 ∧ 
  odot ≠ nabla ∧ 
  odot > 0 ∧ nabla > 0 ∧
  nabla * nabla * odot = nabla → 
  nabla * nabla = 64 := by
  sorry

end NUMINAMATH_CALUDE_nabla_squared_l697_69734


namespace NUMINAMATH_CALUDE_cylinder_height_equals_half_cm_l697_69740

/-- Given a cylinder and a sphere with specific dimensions, proves that the height of the cylinder is 0.5 cm when their volumes are equal. -/
theorem cylinder_height_equals_half_cm 
  (d_cylinder : ℝ) 
  (d_sphere : ℝ) 
  (h_cylinder : ℝ) :
  d_cylinder = 6 →
  d_sphere = 3 →
  π * (d_cylinder / 2)^2 * h_cylinder = (4/3) * π * (d_sphere / 2)^3 →
  h_cylinder = 0.5 := by
  sorry

#check cylinder_height_equals_half_cm

end NUMINAMATH_CALUDE_cylinder_height_equals_half_cm_l697_69740


namespace NUMINAMATH_CALUDE_two_numbers_product_l697_69785

theorem two_numbers_product (ε : ℝ) (h : ε > 0) : 
  ∃ x y : ℝ, x + y = 21 ∧ x^2 + y^2 = 527 ∧ |x * y + 43.05| < ε :=
sorry

end NUMINAMATH_CALUDE_two_numbers_product_l697_69785


namespace NUMINAMATH_CALUDE_biography_increase_l697_69701

theorem biography_increase (T : ℝ) (h1 : T > 0) : 
  let initial_bio := 0.20 * T
  let final_bio := 0.32 * T
  (final_bio - initial_bio) / initial_bio = 0.60
  := by sorry

end NUMINAMATH_CALUDE_biography_increase_l697_69701


namespace NUMINAMATH_CALUDE_expression_value_at_three_l697_69742

theorem expression_value_at_three : 
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 5)
  f 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l697_69742


namespace NUMINAMATH_CALUDE_digit_sum_in_t_shape_l697_69756

theorem digit_sum_in_t_shape : 
  ∀ (a b c d e f g : ℕ),
  a ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  b ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  c ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  d ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  e ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  f ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g →
  a + b + c = 23 →
  d + e + f + g = 12 →
  b = e →
  a + b + c + d + f + g = 29 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_in_t_shape_l697_69756


namespace NUMINAMATH_CALUDE_club_enrollment_l697_69731

/-- Given a club with the following properties:
  * Total members: 85
  * Members enrolled in coding course: 45
  * Members enrolled in design course: 32
  * Members enrolled in both courses: 18
  Prove that the number of members not enrolled in either course is 26. -/
theorem club_enrollment (total : ℕ) (coding : ℕ) (design : ℕ) (both : ℕ)
  (h_total : total = 85)
  (h_coding : coding = 45)
  (h_design : design = 32)
  (h_both : both = 18) :
  total - (coding + design - both) = 26 := by
  sorry

end NUMINAMATH_CALUDE_club_enrollment_l697_69731


namespace NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l697_69733

theorem cos_36_minus_cos_72_eq_half : Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l697_69733


namespace NUMINAMATH_CALUDE_g_inverse_composition_l697_69784

def g : Fin 5 → Fin 5
| 1 => 4
| 2 => 3
| 3 => 1
| 4 => 5
| 5 => 2

theorem g_inverse_composition (h : Function.Bijective g) :
  (Function.invFun g (Function.invFun g (Function.invFun g 3))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_inverse_composition_l697_69784


namespace NUMINAMATH_CALUDE_socks_cost_l697_69726

theorem socks_cost (num_players : ℕ) (jersey_cost shorts_cost total_cost : ℚ) : 
  num_players = 16 →
  jersey_cost = 25 →
  shorts_cost = 15.20 →
  total_cost = 752 →
  ∃ (socks_cost : ℚ), 
    num_players * (jersey_cost + shorts_cost + socks_cost) = total_cost ∧ 
    socks_cost = 6.80 := by
  sorry

end NUMINAMATH_CALUDE_socks_cost_l697_69726


namespace NUMINAMATH_CALUDE_teacher_distribution_count_l697_69775

/-- The number of ways to distribute teachers to classes --/
def distribute_teachers (n_teachers : ℕ) (n_classes : ℕ) : ℕ :=
  n_classes ^ n_teachers

/-- The number of ways to distribute teachers to classes with at least one empty class --/
def distribute_with_empty (n_teachers : ℕ) (n_classes : ℕ) : ℕ :=
  n_classes * (n_classes - 1) ^ n_teachers

/-- The number of valid distributions of teachers to classes --/
def valid_distributions (n_teachers : ℕ) (n_classes : ℕ) : ℕ :=
  distribute_teachers n_teachers n_classes - distribute_with_empty n_teachers n_classes

theorem teacher_distribution_count :
  valid_distributions 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_teacher_distribution_count_l697_69775


namespace NUMINAMATH_CALUDE_additive_increasing_nonneg_implies_odd_increasing_l697_69737

def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂

def is_increasing_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ x₂ → x₂ ≥ 0 → f x₁ ≥ f x₂

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ x₂ → f x₁ ≥ f x₂

theorem additive_increasing_nonneg_implies_odd_increasing
  (f : ℝ → ℝ) (h1 : is_additive f) (h2 : is_increasing_nonneg f) :
  is_odd f ∧ is_increasing f :=
sorry

end NUMINAMATH_CALUDE_additive_increasing_nonneg_implies_odd_increasing_l697_69737


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l697_69771

/-- Given a circular sector with radius 5 cm and area 11.25 cm², 
    prove that the length of the arc is 4.5 cm. -/
theorem arc_length_of_sector (r : ℝ) (area : ℝ) (arc_length : ℝ) : 
  r = 5 → 
  area = 11.25 → 
  arc_length = r * (2 * area / (r * r)) → 
  arc_length = 4.5 := by
sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l697_69771


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l697_69778

theorem gcd_lcm_sum : Nat.gcd 25 64 + Nat.lcm 15 20 = 61 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l697_69778


namespace NUMINAMATH_CALUDE_difference_of_squares_l697_69788

theorem difference_of_squares (x y : ℝ) 
  (sum_eq : x + y = 20) 
  (diff_eq : x - y = 8) : 
  x^2 - y^2 = 160 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l697_69788


namespace NUMINAMATH_CALUDE_lines_parallel_when_perpendicular_to_parallel_planes_l697_69722

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem lines_parallel_when_perpendicular_to_parallel_planes
  (a b : Line) (α β : Plane)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : perpendicular a α)
  (h4 : perpendicular b β)
  (h5 : planeParallel α β) :
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_when_perpendicular_to_parallel_planes_l697_69722


namespace NUMINAMATH_CALUDE_vasya_wins_in_four_moves_l697_69754

-- Define a polynomial with integer coefficients
def IntPolynomial := ℤ → ℤ

-- Define a function that counts the number of integer solutions for P(x) = a
def countIntegerSolutions (P : IntPolynomial) (a : ℤ) : ℕ :=
  sorry

-- Theorem statement
theorem vasya_wins_in_four_moves :
  ∀ (P : IntPolynomial),
  ∃ (S : Finset ℤ),
  (Finset.card S ≤ 4) ∧
  ∃ (a b : ℤ),
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧
  countIntegerSolutions P a = countIntegerSolutions P b :=
sorry

end NUMINAMATH_CALUDE_vasya_wins_in_four_moves_l697_69754


namespace NUMINAMATH_CALUDE_rationalize_denominator_l697_69752

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℚ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧
    B = 7 ∧
    C = 9 ∧
    D = 13 ∧
    E = 5 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l697_69752


namespace NUMINAMATH_CALUDE_negative_abs_equal_l697_69763

theorem negative_abs_equal : -|5| = -|-5| := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_equal_l697_69763


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_difference_l697_69794

theorem cubic_expansion_sum_difference (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5*x + 4)^3 = a + a₁*x + a₂*x^2 + a₃*x^3) →
  (a + a₂) - (a₁ + a₃) = -1 :=
by sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_difference_l697_69794


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l697_69782

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let new_area := (1.1 * L) * (1.1 * W)
  let original_area := L * W
  (new_area - original_area) / original_area * 100 = 21 := by
sorry


end NUMINAMATH_CALUDE_rectangle_area_increase_l697_69782


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l697_69757

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| ≥ 5} = {x : ℝ | x ≥ 6 ∨ x ≤ -4} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l697_69757


namespace NUMINAMATH_CALUDE_hyperbola_properties_l697_69758

/-- Given a hyperbola with specific properties, prove its equation and a property of its intersection with a line --/
theorem hyperbola_properties (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let e : ℝ := Real.sqrt 3
  let vertex : ℝ × ℝ := (Real.sqrt 3, 0)
  ∀ x y, C x y →
    (∃ c, c > 0 ∧ c^2 = a^2 + b^2 ∧ c / a = e) →
    (C (Real.sqrt 3) 0) →
    (∃ F₂ : ℝ × ℝ, F₂.1 > 0 ∧
      (∀ x y, (y - F₂.2) = Real.sqrt 3 / 3 * (x - F₂.1) →
        C x y →
        ∃ A B : ℝ × ℝ, A ≠ B ∧ C A.1 A.2 ∧ C B.1 B.2 ∧
          Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 * Real.sqrt 3 / 5)) →
  C x y ↔ x^2 / 3 - y^2 / 6 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l697_69758


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_geq_sqrt2_sum_l697_69719

theorem sqrt_sum_squares_geq_sqrt2_sum (a b : ℝ) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_geq_sqrt2_sum_l697_69719


namespace NUMINAMATH_CALUDE_clock_rings_count_l697_69713

def clock_rings (hour : ℕ) : Bool :=
  if hour ≤ 12 then
    hour % 2 = 1
  else
    hour % 4 = 1

def total_rings : ℕ :=
  (List.range 24).filter (λ h => clock_rings (h + 1)) |>.length

theorem clock_rings_count : total_rings = 10 := by
  sorry

end NUMINAMATH_CALUDE_clock_rings_count_l697_69713


namespace NUMINAMATH_CALUDE_kaleb_remaining_chocolates_l697_69729

def boxes_bought : ℕ := 14
def pieces_per_box : ℕ := 6
def boxes_given_away : ℕ := 5 + 2 + 3

def remaining_boxes : ℕ := boxes_bought - boxes_given_away
def remaining_pieces : ℕ := remaining_boxes * pieces_per_box

def eaten_pieces : ℕ := (remaining_pieces * 10) / 100

theorem kaleb_remaining_chocolates :
  remaining_pieces - eaten_pieces = 22 := by sorry

end NUMINAMATH_CALUDE_kaleb_remaining_chocolates_l697_69729


namespace NUMINAMATH_CALUDE_total_amount_shared_l697_69761

/-- The total amount shared by A, B, and C given specific conditions -/
theorem total_amount_shared (a b c : ℝ) : 
  a = (1/3) * (b + c) →  -- A gets one-third of what B and C together get
  b = (2/7) * (a + c) →  -- B gets two-sevenths of what A and C together get
  a = b + 35 →           -- A's amount is $35 more than B's amount
  a + b + c = 1260 :=    -- The total amount shared
by sorry

end NUMINAMATH_CALUDE_total_amount_shared_l697_69761


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l697_69766

theorem largest_divisor_of_expression (p q : ℤ) 
  (hp : Odd p) (hq : Odd q) (hlt : q < p) : 
  ∃ k : ℤ, p^2 - q^2 + 2*p - 2*q = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l697_69766


namespace NUMINAMATH_CALUDE_infinitely_many_primes_dividing_n_squared_plus_n_plus_one_l697_69723

theorem infinitely_many_primes_dividing_n_squared_plus_n_plus_one :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ n^2 + n + 1} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_dividing_n_squared_plus_n_plus_one_l697_69723


namespace NUMINAMATH_CALUDE_car_speed_problem_l697_69721

/-- Proves that the speed in the first hour is 90 km/h given the conditions -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 50 →
  average_speed = 70 →
  (speed_first_hour : ℝ) →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_first_hour = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l697_69721


namespace NUMINAMATH_CALUDE_chess_tournament_games_l697_69714

theorem chess_tournament_games (P : ℕ) (total_games : ℕ) (h1 : P = 21) (h2 : total_games = 210) :
  (P * (P - 1)) / 2 = total_games ∧ P - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l697_69714


namespace NUMINAMATH_CALUDE_phone_storage_theorem_l697_69769

/-- Calculates the maximum number of songs that can be stored on a phone given the total storage, used storage, and size of each song. -/
def max_songs (total_storage : ℕ) (used_storage : ℕ) (song_size : ℕ) : ℕ :=
  ((total_storage - used_storage) * 1000) / song_size

/-- Theorem stating that given a phone with 16 GB total storage, 4 GB already used, and songs of 30 MB each, the maximum number of additional songs that can be stored is 400. -/
theorem phone_storage_theorem :
  max_songs 16 4 30 = 400 := by
  sorry

end NUMINAMATH_CALUDE_phone_storage_theorem_l697_69769


namespace NUMINAMATH_CALUDE_frequency_converges_to_half_l697_69783

/-- A fair coin toss experiment -/
structure CoinTossExperiment where
  n : ℕ  -- number of tosses
  m : ℕ  -- number of heads
  h_m_le_n : m ≤ n  -- m cannot exceed n

/-- The frequency of heads in a coin toss experiment -/
def frequency (e : CoinTossExperiment) : ℚ :=
  e.m / e.n

/-- The theoretical probability of heads for a fair coin -/
def fairCoinProbability : ℚ := 1 / 2

/-- The main theorem: as n approaches infinity, the frequency converges to 1/2 -/
theorem frequency_converges_to_half :
  ∀ ε > 0, ∃ N, ∀ e : CoinTossExperiment, e.n ≥ N →
    |frequency e - fairCoinProbability| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_converges_to_half_l697_69783


namespace NUMINAMATH_CALUDE_base_conversion_1729_l697_69720

theorem base_conversion_1729 :
  (2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0) = 1729 := by
  sorry

#eval 2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0

end NUMINAMATH_CALUDE_base_conversion_1729_l697_69720


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_difference_two_l697_69735

def a (n : ℕ) : ℝ := 2 * (n + 1) + 3

theorem arithmetic_sequence_with_difference_two :
  ∀ n : ℕ, a (n + 1) - a n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_difference_two_l697_69735


namespace NUMINAMATH_CALUDE_chord_equation_l697_69799

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a circle with center (0,0) and radius 3 -/
def isOnCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 = 9

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m p q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

/-- Checks if a line passes through a point -/
def linePassesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem chord_equation (p q : Point) (m : Point) :
  isOnCircle p ∧ isOnCircle q ∧ isMidpoint m p q ∧ m.x = 1 ∧ m.y = 2 →
  ∃ l : Line, l.a = 1 ∧ l.b = 2 ∧ l.c = -5 ∧ linePassesThrough l p ∧ linePassesThrough l q :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l697_69799


namespace NUMINAMATH_CALUDE_treasure_sum_l697_69772

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- The four treasure values in base 7 --/
def treasure1 : List Nat := [4, 1, 2, 3]
def treasure2 : List Nat := [2, 5, 6, 1]
def treasure3 : List Nat := [1, 3, 4, 2]
def treasure4 : List Nat := [4, 5, 6]

theorem treasure_sum :
  base7ToBase10 treasure1 +
  base7ToBase10 treasure2 +
  base7ToBase10 treasure3 +
  base7ToBase10 treasure4 = 3049 := by
  sorry

end NUMINAMATH_CALUDE_treasure_sum_l697_69772


namespace NUMINAMATH_CALUDE_triangle_sides_l697_69796

/-- A right-angled triangle with sides in arithmetic progression and area 216 cm² --/
structure RightTriangle where
  a : ℝ
  d : ℝ
  area_eq : (1/2) * a * (a - d) = 216
  progression : a > 0 ∧ d > 0
  right_angle : (a - d)^2 + a^2 = (a + d)^2

/-- The sides of the triangle are 18, 24, and 30 --/
theorem triangle_sides (t : RightTriangle) : t.a = 24 ∧ t.d = 6 := by
  sorry

#check triangle_sides

end NUMINAMATH_CALUDE_triangle_sides_l697_69796


namespace NUMINAMATH_CALUDE_unique_alpha_l697_69760

def A (α : ℝ) : Set ℕ := {n : ℕ | ∃ k : ℕ, n = ⌊k * α⌋}

theorem unique_alpha : ∃! α : ℝ, 
  α ≥ 1 ∧ 
  (∃ r : ℕ, r < 2021 ∧ 
    (∀ n : ℕ, n > 0 → (n ∉ A α ↔ n % 2021 = r))) ∧
  α = 2021 / 2020 := by
sorry

end NUMINAMATH_CALUDE_unique_alpha_l697_69760


namespace NUMINAMATH_CALUDE_min_yacht_capacity_l697_69700

/-- Represents the number of sheikhs --/
def num_sheikhs : ℕ := 10

/-- Represents the number of wives per sheikh --/
def wives_per_sheikh : ℕ := 100

/-- Represents the total number of wives --/
def total_wives : ℕ := num_sheikhs * wives_per_sheikh

/-- Represents the law: a woman must not be with a man other than her husband unless her husband is present --/
def law_compliant (n : ℕ) : Prop :=
  ∀ (women_on_bank : ℕ) (men_on_bank : ℕ),
    women_on_bank ≤ total_wives ∧ men_on_bank ≤ num_sheikhs →
    (women_on_bank ≤ n ∨ men_on_bank = num_sheikhs ∨ women_on_bank = 0)

/-- Theorem stating that the smallest yacht capacity that allows all sheikhs and wives to cross the river while complying with the law is 10 --/
theorem min_yacht_capacity :
  ∃ (n : ℕ), n = 10 ∧ law_compliant n ∧ ∀ (m : ℕ), m < n → ¬law_compliant m :=
sorry

end NUMINAMATH_CALUDE_min_yacht_capacity_l697_69700


namespace NUMINAMATH_CALUDE_unique_cube_difference_nineteen_l697_69711

theorem unique_cube_difference_nineteen :
  ∀ x y : ℕ, x^3 - y^3 = 19 → x = 3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_difference_nineteen_l697_69711


namespace NUMINAMATH_CALUDE_four_digit_number_puzzle_l697_69767

theorem four_digit_number_puzzle :
  ∀ (A B x y : ℕ),
    1000 ≤ A ∧ A < 10000 →
    0 ≤ x ∧ x < 10 →
    0 ≤ y ∧ y < 10 →
    B = 100000 * x + 10 * A + y →
    B = 21 * A →
    A = 9091 ∧ B = 190911 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_puzzle_l697_69767


namespace NUMINAMATH_CALUDE_range_of_f_l697_69728

-- Define the function f
def f (x : ℝ) : ℝ := 3 - x

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ≤ 1, f x = y} = {y : ℝ | y ≥ 2} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l697_69728


namespace NUMINAMATH_CALUDE_surviving_trees_count_l697_69708

/-- Calculates the number of surviving trees after two months given initial conditions --/
theorem surviving_trees_count
  (tree_A_plants tree_B_plants tree_C_plants : ℕ)
  (tree_A_seeds_per_plant tree_B_seeds_per_plant tree_C_seeds_per_plant : ℕ)
  (tree_A_plant_rate tree_B_plant_rate tree_C_plant_rate : ℚ)
  (tree_A_first_month_survival_rate tree_B_first_month_survival_rate tree_C_first_month_survival_rate : ℚ)
  (second_month_survival_rate : ℚ)
  (h1 : tree_A_plants = 25)
  (h2 : tree_B_plants = 20)
  (h3 : tree_C_plants = 10)
  (h4 : tree_A_seeds_per_plant = 1)
  (h5 : tree_B_seeds_per_plant = 2)
  (h6 : tree_C_seeds_per_plant = 3)
  (h7 : tree_A_plant_rate = 3/5)
  (h8 : tree_B_plant_rate = 4/5)
  (h9 : tree_C_plant_rate = 1/2)
  (h10 : tree_A_first_month_survival_rate = 3/4)
  (h11 : tree_B_first_month_survival_rate = 9/10)
  (h12 : tree_C_first_month_survival_rate = 7/10)
  (h13 : second_month_survival_rate = 9/10) :
  ⌊(⌊tree_A_plants * tree_A_seeds_per_plant * tree_A_plant_rate * tree_A_first_month_survival_rate⌋ : ℚ) * second_month_survival_rate⌋ +
  ⌊(⌊tree_B_plants * tree_B_seeds_per_plant * tree_B_plant_rate * tree_B_first_month_survival_rate⌋ : ℚ) * second_month_survival_rate⌋ +
  ⌊(⌊tree_C_plants * tree_C_seeds_per_plant * tree_C_plant_rate * tree_C_first_month_survival_rate⌋ : ℚ) * second_month_survival_rate⌋ = 43 := by
  sorry


end NUMINAMATH_CALUDE_surviving_trees_count_l697_69708


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l697_69762

/-- Proves that given a train of length 75 meters, which crosses a bridge in 7.5 seconds
    and a lamp post on the bridge in 2.5 seconds, the length of the bridge is 150 meters. -/
theorem bridge_length_calculation (train_length : ℝ) (bridge_crossing_time : ℝ) (lamppost_crossing_time : ℝ)
  (h1 : train_length = 75)
  (h2 : bridge_crossing_time = 7.5)
  (h3 : lamppost_crossing_time = 2.5) :
  let bridge_length := (train_length * bridge_crossing_time / lamppost_crossing_time) - train_length
  bridge_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l697_69762


namespace NUMINAMATH_CALUDE_horse_journey_l697_69709

theorem horse_journey (a₁ : ℚ) : 
  (a₁ * (1 - (1/2)^7) / (1 - 1/2) = 700) → 
  (a₁ * (1/2)^6 = 700/127) := by
sorry

end NUMINAMATH_CALUDE_horse_journey_l697_69709


namespace NUMINAMATH_CALUDE_lauri_eating_days_l697_69753

/-- The number of days Lauri ate apples -/
def lauriDays : ℕ := 15

/-- The fraction of an apple Simone ate per day -/
def simonePerDay : ℚ := 1/2

/-- The number of days Simone ate apples -/
def simoneDays : ℕ := 16

/-- The fraction of an apple Lauri ate per day -/
def lauriPerDay : ℚ := 1/3

/-- The total number of apples both girls ate -/
def totalApples : ℕ := 13

theorem lauri_eating_days : 
  simonePerDay * simoneDays + lauriPerDay * lauriDays = totalApples := by
  sorry

end NUMINAMATH_CALUDE_lauri_eating_days_l697_69753


namespace NUMINAMATH_CALUDE_parkway_elementary_girls_not_playing_soccer_l697_69725

theorem parkway_elementary_girls_not_playing_soccer 
  (total_students : ℕ) 
  (total_boys : ℕ) 
  (total_soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 450 →
  total_boys = 320 →
  total_soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  ∃ (girls_not_playing_soccer : ℕ), 
    girls_not_playing_soccer = 
      total_students - total_boys - 
      (total_soccer_players - (boys_soccer_percentage * total_soccer_players).floor) :=
by
  sorry

end NUMINAMATH_CALUDE_parkway_elementary_girls_not_playing_soccer_l697_69725


namespace NUMINAMATH_CALUDE_polynomial_identity_l697_69743

theorem polynomial_identity (P : ℝ → ℝ) : 
  (∀ x : ℝ, P (x^3 - 2) = P x^3 - 2) ↔ (∀ x : ℝ, P x = x) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l697_69743


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l697_69770

def x : ℕ := 7 * 24 * 48

theorem smallest_y_for_perfect_fourth_power (y : ℕ) :
  y = 6174 ↔ (
    y > 0 ∧
    ∃ (n : ℕ), x * y = n^4 ∧
    ∀ (z : ℕ), 0 < z ∧ z < y → ¬∃ (m : ℕ), x * z = m^4
  ) := by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l697_69770


namespace NUMINAMATH_CALUDE_miles_driven_with_budget_l697_69704

-- Define the given conditions
def miles_per_gallon : ℝ := 32
def cost_per_gallon : ℝ := 4
def budget : ℝ := 20

-- Define the theorem
theorem miles_driven_with_budget :
  (budget / cost_per_gallon) * miles_per_gallon = 160 := by
  sorry

end NUMINAMATH_CALUDE_miles_driven_with_budget_l697_69704


namespace NUMINAMATH_CALUDE_assignment_schemes_l697_69741

def total_students : ℕ := 6
def selected_students : ℕ := 4
def restricted_students : ℕ := 2
def restricted_tasks : ℕ := 1

theorem assignment_schemes :
  (total_students.factorial / (total_students - selected_students).factorial) -
  (restricted_students * (total_students - 1).factorial / (total_students - selected_students).factorial) = 240 :=
sorry

end NUMINAMATH_CALUDE_assignment_schemes_l697_69741


namespace NUMINAMATH_CALUDE_complex_product_negative_l697_69710

theorem complex_product_negative (a : ℝ) :
  let z : ℂ := (a + Complex.I) * (-3 + a * Complex.I)
  (z.re < 0) → a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_product_negative_l697_69710


namespace NUMINAMATH_CALUDE_olivia_checking_time_l697_69730

def time_spent_checking (num_problems : ℕ) (time_per_problem : ℕ) (total_time : ℕ) : ℕ :=
  total_time - (num_problems * time_per_problem)

theorem olivia_checking_time :
  time_spent_checking 7 4 31 = 3 := by sorry

end NUMINAMATH_CALUDE_olivia_checking_time_l697_69730


namespace NUMINAMATH_CALUDE_equivalent_operations_l697_69739

theorem equivalent_operations (x : ℚ) : 
  (x * (4/5)) / (4/7) = x * (7/5) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operations_l697_69739


namespace NUMINAMATH_CALUDE_nine_times_nines_digit_sum_l697_69751

/-- Represents a number consisting of n nines -/
def nines (n : ℕ) : ℕ := (10^n - 1)

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Theorem stating that the product of 9 and a number with 120 nines has a digit sum of 1080 -/
theorem nine_times_nines_digit_sum :
  sumOfDigits (9 * nines 120) = 1080 := by sorry

end NUMINAMATH_CALUDE_nine_times_nines_digit_sum_l697_69751


namespace NUMINAMATH_CALUDE_darrel_coin_counting_machine_result_l697_69777

/-- Calculates the amount received after fees for a given coin type -/
def amountAfterFee (coinValue : ℚ) (count : ℕ) (feePercentage : ℚ) : ℚ :=
  let totalValue := coinValue * count
  totalValue - (totalValue * feePercentage / 100)

/-- Theorem stating the total amount Darrel receives after fees -/
theorem darrel_coin_counting_machine_result : 
  let quarterCount : ℕ := 127
  let dimeCount : ℕ := 183
  let nickelCount : ℕ := 47
  let pennyCount : ℕ := 237
  let halfDollarCount : ℕ := 64
  
  let quarterValue : ℚ := 25 / 100
  let dimeValue : ℚ := 10 / 100
  let nickelValue : ℚ := 5 / 100
  let pennyValue : ℚ := 1 / 100
  let halfDollarValue : ℚ := 50 / 100
  
  let quarterFee : ℚ := 12
  let dimeFee : ℚ := 7
  let nickelFee : ℚ := 15
  let pennyFee : ℚ := 10
  let halfDollarFee : ℚ := 5
  
  let totalAfterFees := 
    amountAfterFee quarterValue quarterCount quarterFee +
    amountAfterFee dimeValue dimeCount dimeFee +
    amountAfterFee nickelValue nickelCount nickelFee +
    amountAfterFee pennyValue pennyCount pennyFee +
    amountAfterFee halfDollarValue halfDollarCount halfDollarFee
  
  totalAfterFees = 7949 / 100 := by
  sorry


end NUMINAMATH_CALUDE_darrel_coin_counting_machine_result_l697_69777


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_l697_69786

theorem cubic_expansion_sum (x a₀ a₁ a₂ a₃ : ℝ) 
  (h : x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) : 
  a₁ + a₂ + a₃ = 19 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_l697_69786


namespace NUMINAMATH_CALUDE_divisible_by_77_l697_69702

theorem divisible_by_77 (n : ℕ) (h : ∀ k : ℕ, 2 ≤ k → k ≤ 76 → k ∣ n) : 77 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_77_l697_69702


namespace NUMINAMATH_CALUDE_restaurant_problem_l697_69791

theorem restaurant_problem (initial_wings : ℕ) (additional_wings : ℕ) (wings_per_friend : ℕ) 
  (h1 : initial_wings = 9)
  (h2 : additional_wings = 7)
  (h3 : wings_per_friend = 4) :
  (initial_wings + additional_wings) / wings_per_friend = 4 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_problem_l697_69791


namespace NUMINAMATH_CALUDE_subset_implies_a_bound_l697_69792

theorem subset_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 2 ≤ 0 → 1/(x-3) < a) → a > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_bound_l697_69792
