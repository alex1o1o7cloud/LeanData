import Mathlib

namespace NUMINAMATH_CALUDE_jaysons_mom_age_at_birth_l673_67316

theorem jaysons_mom_age_at_birth (jayson_age : ℕ) (dad_age : ℕ) (mom_age : ℕ) : 
  jayson_age = 10 →
  dad_age = 4 * jayson_age →
  mom_age = dad_age - 2 →
  mom_age - jayson_age = 28 := by
  sorry

end NUMINAMATH_CALUDE_jaysons_mom_age_at_birth_l673_67316


namespace NUMINAMATH_CALUDE_even_five_digit_numbers_l673_67399

def set1 : Finset ℕ := {1, 3, 5}
def set2 : Finset ℕ := {2, 4, 6, 8}

def is_valid_selection (s : Finset ℕ) : Prop :=
  s.card = 5 ∧ (s ∩ set1).card = 2 ∧ (s ∩ set2).card = 3

def is_even (n : ℕ) : Prop := n % 2 = 0

def count_even_numbers : ℕ := sorry

theorem even_five_digit_numbers :
  count_even_numbers = 864 :=
sorry

end NUMINAMATH_CALUDE_even_five_digit_numbers_l673_67399


namespace NUMINAMATH_CALUDE_window_side_length_is_five_l673_67341

/-- Represents the dimensions of a glass pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the window configuration -/
structure Window where
  pane : Pane
  rows : ℕ
  columns : ℕ
  border_width : ℝ

/-- Calculates the side length of a square window -/
def window_side_length (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- Theorem stating that the window's side length is 5 inches -/
theorem window_side_length_is_five (w : Window) 
  (h_square : window_side_length w = w.rows * w.pane.height + (w.rows + 1) * w.border_width)
  (h_rows : w.rows = 2)
  (h_columns : w.columns = 3)
  (h_border : w.border_width = 1) :
  window_side_length w = 5 := by
  sorry

#check window_side_length_is_five

end NUMINAMATH_CALUDE_window_side_length_is_five_l673_67341


namespace NUMINAMATH_CALUDE_set_classification_l673_67373

/-- The set of numbers we're working with -/
def S : Set ℝ := {-2, -3.14, 0.3, 0, Real.pi/3, 22/7, -0.1212212221}

/-- The set of positive numbers in S -/
def positiveS : Set ℝ := {x ∈ S | x > 0}

/-- The set of negative numbers in S -/
def negativeS : Set ℝ := {x ∈ S | x < 0}

/-- The set of integers in S -/
def integerS : Set ℝ := {x ∈ S | ∃ n : ℤ, x = n}

/-- The set of rational numbers in S -/
def rationalS : Set ℝ := {x ∈ S | ∃ p q : ℤ, q ≠ 0 ∧ x = p / q}

theorem set_classification :
  positiveS = {0.3, Real.pi/3, 22/7} ∧
  negativeS = {-2, -3.14, -0.1212212221} ∧
  integerS = {-2, 0} ∧
  rationalS = {-2, 0, 0.3, 22/7} := by
  sorry

end NUMINAMATH_CALUDE_set_classification_l673_67373


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l673_67390

theorem sin_product_equals_one_sixteenth :
  Real.sin (18 * π / 180) * Real.sin (42 * π / 180) *
  Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l673_67390


namespace NUMINAMATH_CALUDE_smallest_base_for_145_l673_67344

theorem smallest_base_for_145 :
  ∀ b : ℕ, b ≥ 2 →
    (∀ n : ℕ, n ≥ 2 ∧ n < b → n^2 ≤ 145 ∧ 145 < n^3) →
    b = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_145_l673_67344


namespace NUMINAMATH_CALUDE_grass_seed_problem_l673_67389

/-- Represents the cost and weight of a bag of grass seed -/
structure SeedBag where
  weight : Nat
  cost : Rat

/-- Represents a purchase of grass seed -/
structure Purchase where
  bags : List SeedBag
  totalWeight : Nat
  totalCost : Rat

def validPurchase (p : Purchase) : Prop :=
  p.totalWeight ≥ 65 ∧ p.totalWeight ≤ 80

def optimalPurchase (p : Purchase) : Prop :=
  validPurchase p ∧ p.totalCost = 98.75

/-- The theorem to be proved -/
theorem grass_seed_problem :
  ∃ (cost_5lb : Rat),
    let bag_5lb : SeedBag := ⟨5, cost_5lb⟩
    let bag_10lb : SeedBag := ⟨10, 20.40⟩
    let bag_25lb : SeedBag := ⟨25, 32.25⟩
    ∃ (p : Purchase),
      optimalPurchase p ∧
      bag_5lb ∈ p.bags ∧
      cost_5lb = 2.00 :=
sorry

end NUMINAMATH_CALUDE_grass_seed_problem_l673_67389


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l673_67304

/-- Given that i is the imaginary unit and (2+i)/(1+i) = a + bi where a and b are real numbers,
    prove that a + b = 1 -/
theorem complex_fraction_sum (i : ℂ) (a b : ℝ) 
    (h1 : i * i = -1) 
    (h2 : (2 + i) / (1 + i) = a + b * i) : 
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l673_67304


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_achievable_l673_67345

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  1 / x + 1 / y ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_reciprocal_sum_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_achievable_l673_67345


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l673_67329

theorem greatest_integer_fraction_inequality : 
  ∀ x : ℤ, (8 : ℚ) / 11 > (x : ℚ) / 15 ↔ x ≤ 10 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l673_67329


namespace NUMINAMATH_CALUDE_vacation_cost_split_vacation_cost_equalization_l673_67327

theorem vacation_cost_split (X Y : ℝ) (h : X > Y) : 
  (X - Y) / 2 = (X + Y) / 2 - Y := by sorry

theorem vacation_cost_equalization (X Y : ℝ) (h : X > Y) : 
  (X - Y) / 2 > 0 := by sorry

end NUMINAMATH_CALUDE_vacation_cost_split_vacation_cost_equalization_l673_67327


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l673_67317

theorem at_least_one_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l673_67317


namespace NUMINAMATH_CALUDE_bruce_mango_purchase_l673_67377

def bruce_purchase (grape_quantity : ℕ) (grape_price : ℕ) (mango_price : ℕ) (total_paid : ℕ) : ℕ :=
  let grape_cost := grape_quantity * grape_price
  let mango_cost := total_paid - grape_cost
  mango_cost / mango_price

theorem bruce_mango_purchase :
  bruce_purchase 7 70 55 985 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bruce_mango_purchase_l673_67377


namespace NUMINAMATH_CALUDE_sum_of_squared_ratios_bound_l673_67338

/-- Given positive real numbers a, b, and c, 
    the sum of three terms in the form (2x+y+z)²/(2x²+(y+z)²) 
    where x, y, z are cyclic permutations of a, b, c, is less than or equal to 8 -/
theorem sum_of_squared_ratios_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) + 
  (2*b + a + c)^2 / (2*b^2 + (c + a)^2) + 
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_ratios_bound_l673_67338


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_three_l673_67381

theorem sum_of_a_and_b_is_three (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a + 2 * i) / i = b - i * a) : a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_three_l673_67381


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l673_67302

/-- Given a triangle with sides 6, 10, and 11, prove that an equilateral triangle
    with the same perimeter has side length 9. -/
theorem equilateral_triangle_side_length : 
  ∀ (a b c s : ℝ), 
    a = 6 → b = 10 → c = 11 →  -- Given triangle side lengths
    3 * s = a + b + c →        -- Equilateral triangle has same perimeter
    s = 9 :=                   -- Side length of equilateral triangle is 9
by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l673_67302


namespace NUMINAMATH_CALUDE_stamp_collection_ratio_l673_67358

theorem stamp_collection_ratio : 
  ∀ (tom_original mike_gift harry_gift tom_final : ℕ),
    tom_original = 3000 →
    mike_gift = 17 →
    ∃ k : ℕ, harry_gift = k * mike_gift + 10 →
    tom_final = tom_original + mike_gift + harry_gift →
    tom_final = 3061 →
    harry_gift / mike_gift = 44 / 17 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_ratio_l673_67358


namespace NUMINAMATH_CALUDE_angle_in_square_l673_67388

/-- In a square ABCD with a segment CE, if CE forms angles of 7α and 8α with the sides of the square, then α = 9°. -/
theorem angle_in_square (α : ℝ) : 
  (7 * α + 8 * α + 45 = 180) → α = 9 := by sorry

end NUMINAMATH_CALUDE_angle_in_square_l673_67388


namespace NUMINAMATH_CALUDE_red_light_estimation_l673_67367

theorem red_light_estimation (total_students : ℕ) (total_yes : ℕ) (known_yes_rate : ℚ) :
  total_students = 600 →
  total_yes = 180 →
  known_yes_rate = 1/2 →
  ∃ (estimated_red_light : ℕ), estimated_red_light = 60 :=
by sorry

end NUMINAMATH_CALUDE_red_light_estimation_l673_67367


namespace NUMINAMATH_CALUDE_function_identity_l673_67342

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : 
  ∀ n : ℕ+, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l673_67342


namespace NUMINAMATH_CALUDE_unique_phone_number_l673_67351

-- Define the set of available digits
def available_digits : Finset Nat := {2, 3, 4, 5, 6, 7, 8}

-- Define a function to check if a list of digits is valid
def valid_phone_number (digits : List Nat) : Prop :=
  digits.length = 7 ∧
  digits.toFinset = available_digits ∧
  digits.Sorted (·<·)

-- Theorem statement
theorem unique_phone_number :
  ∃! digits : List Nat, valid_phone_number digits :=
sorry

end NUMINAMATH_CALUDE_unique_phone_number_l673_67351


namespace NUMINAMATH_CALUDE_a_squared_gt_one_neither_sufficient_nor_necessary_for_one_over_a_gt_zero_l673_67360

theorem a_squared_gt_one_neither_sufficient_nor_necessary_for_one_over_a_gt_zero :
  ¬(∀ a : ℝ, a^2 > 1 → 1/a > 0) ∧ ¬(∀ a : ℝ, 1/a > 0 → a^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_a_squared_gt_one_neither_sufficient_nor_necessary_for_one_over_a_gt_zero_l673_67360


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l673_67368

/-- The perimeter of a rhombus with diagonals of 18 inches and 32 inches is 4√337 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 32) :
  4 * (((d1 / 2) ^ 2 + (d2 / 2) ^ 2).sqrt) = 4 * Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l673_67368


namespace NUMINAMATH_CALUDE_kendra_spelling_goals_l673_67362

-- Define constants
def words_per_week : ℕ := 12
def first_goal : ℕ := 60
def second_goal : ℕ := 100
def reward_threshold : ℕ := 20
def words_learned : ℕ := 36
def weeks_to_birthday : ℕ := 3
def weeks_to_competition : ℕ := 6

-- Define the theorem
theorem kendra_spelling_goals (target : ℕ) :
  (target ≥ reward_threshold) ∧
  (target * weeks_to_birthday + words_learned ≥ first_goal) ∧
  (target * weeks_to_competition + words_learned ≥ second_goal) ↔
  target = reward_threshold :=
by sorry

end NUMINAMATH_CALUDE_kendra_spelling_goals_l673_67362


namespace NUMINAMATH_CALUDE_no_integer_solution_for_z_l673_67306

theorem no_integer_solution_for_z :
  ¬ ∃ (z : ℤ), (2 : ℚ) / z = 2 / (z + 1) + 2 / (z + 25) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_z_l673_67306


namespace NUMINAMATH_CALUDE_a_greater_than_b_l673_67363

theorem a_greater_than_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (eq1 : a^3 = a + 1) (eq2 : b^6 = b + 3*a) : a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l673_67363


namespace NUMINAMATH_CALUDE_square_sum_identity_l673_67380

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l673_67380


namespace NUMINAMATH_CALUDE_no_roots_equation_l673_67339

theorem no_roots_equation : ¬∃ (x : ℝ), x - 8 / (x - 4) = 4 - 8 / (x - 4) := by sorry

end NUMINAMATH_CALUDE_no_roots_equation_l673_67339


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l673_67391

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 2) : 
  (a^2 - 4*a + 4) / a / (a - 4/a) = 1 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l673_67391


namespace NUMINAMATH_CALUDE_intersection_parallel_perpendicular_line_l673_67396

/-- The equation of a line passing through the intersection of two lines,
    parallel to one line, and perpendicular to another line. -/
theorem intersection_parallel_perpendicular_line 
  (l1 l2 l_parallel l_perpendicular : ℝ → ℝ → Prop) 
  (h_l1 : ∀ x y, l1 x y ↔ 2*x - 3*y + 10 = 0)
  (h_l2 : ∀ x y, l2 x y ↔ 3*x + 4*y - 2 = 0)
  (h_parallel : ∀ x y, l_parallel x y ↔ x - y + 1 = 0)
  (h_perpendicular : ∀ x y, l_perpendicular x y ↔ 3*x - y - 2 = 0)
  : ∃ l : ℝ → ℝ → Prop, 
    (∃ x y, l1 x y ∧ l2 x y ∧ l x y) ∧ 
    (∀ x y, l x y ↔ x - y + 4 = 0) ∧
    (∀ a b c d, l a b ∧ l c d → (c - a) * (1) + (-1) * (d - b) = 0) ∧
    (∀ a b c d, l a b ∧ l c d → (c - a) * (3) + (-1) * (d - b) = 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_parallel_perpendicular_line_l673_67396


namespace NUMINAMATH_CALUDE_josh_cheese_purchase_cost_l673_67331

/-- Calculates the total cost of string cheese purchase including tax -/
def total_cost_with_tax (packs : ℕ) (pieces_per_pack : ℕ) (cost_per_piece : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost := packs * pieces_per_pack * cost_per_piece
  let tax := total_cost * tax_rate
  total_cost + tax

/-- The total cost of Josh's string cheese purchase including tax is $6.72 -/
theorem josh_cheese_purchase_cost :
  total_cost_with_tax 3 20 (10 / 100) (12 / 100) = 672 / 100 := by
  sorry

#eval total_cost_with_tax 3 20 (10 / 100) (12 / 100)

end NUMINAMATH_CALUDE_josh_cheese_purchase_cost_l673_67331


namespace NUMINAMATH_CALUDE_max_value_of_expression_l673_67353

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^2*(b+c) + b^2*(c+a) + c^2*(a+b)) / (a^3 + b^3 + c^3 - 2*a*b*c)
  A ≤ 6 ∧ (A = 6 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l673_67353


namespace NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l673_67349

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 150) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 152 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l673_67349


namespace NUMINAMATH_CALUDE_alto_saxophone_ratio_l673_67384

/-- The ratio of alto saxophone players to total saxophone players in a high school band -/
theorem alto_saxophone_ratio (total_students : ℕ) 
  (h1 : total_students = 600)
  (marching_band : ℕ) 
  (h2 : marching_band = total_students / 5)
  (brass_players : ℕ) 
  (h3 : brass_players = marching_band / 2)
  (saxophone_players : ℕ) 
  (h4 : saxophone_players = brass_players / 5)
  (alto_saxophone_players : ℕ) 
  (h5 : alto_saxophone_players = 4) :
  (alto_saxophone_players : ℚ) / saxophone_players = 1 / 3 := by
sorry


end NUMINAMATH_CALUDE_alto_saxophone_ratio_l673_67384


namespace NUMINAMATH_CALUDE_performance_stability_comparison_l673_67311

/-- Represents the variance of a student's scores -/
structure StudentVariance where
  value : ℝ
  positive : value > 0

/-- Defines when one performance is more stable than another based on variance -/
def more_stable (a b : StudentVariance) : Prop :=
  a.value > b.value

theorem performance_stability_comparison
  (S_A : StudentVariance)
  (S_B : StudentVariance)
  (h_A : S_A.value = 0.2)
  (h_B : S_B.value = 0.09) :
  more_stable S_A S_B = false :=
by sorry

end NUMINAMATH_CALUDE_performance_stability_comparison_l673_67311


namespace NUMINAMATH_CALUDE_common_chord_length_l673_67337

/-- The length of the common chord of two intersecting circles -/
theorem common_chord_length (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 10*y - 24 = 0) →
  (x^2 + y^2 + 2*x + 2*y - 8 = 0) →
  ∃ (l : ℝ), l = 2 * Real.sqrt 5 ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      (x1^2 + y1^2 - 2*x1 + 10*y1 - 24 = 0) ∧
      (x1^2 + y1^2 + 2*x1 + 2*y1 - 8 = 0) ∧
      (x2^2 + y2^2 - 2*x2 + 10*y2 - 24 = 0) ∧
      (x2^2 + y2^2 + 2*x2 + 2*y2 - 8 = 0) ∧
      l^2 = (x2 - x1)^2 + (y2 - y1)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_l673_67337


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l673_67335

/-- A cube with volume 5x and surface area x has x equal to 5400 -/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 5*x ∧ 6*s^2 = x) → x = 5400 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l673_67335


namespace NUMINAMATH_CALUDE_triangle_theorem_l673_67387

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : (2 * Real.sin t.C - Real.sin t.B) / Real.sin t.B = (t.a * Real.cos t.B) / (t.b * Real.cos t.A))
  (h2 : t.a = 3)
  (h3 : Real.sin t.C = 2 * Real.sin t.B) :
  t.A = π/3 ∧ t.b = Real.sqrt 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l673_67387


namespace NUMINAMATH_CALUDE_sandbox_sand_calculation_l673_67366

/-- Calculates the amount of sand needed to fill a square sandbox -/
theorem sandbox_sand_calculation (side_length : ℝ) (sand_weight_per_section : ℝ) (area_per_section : ℝ) :
  side_length = 40 →
  sand_weight_per_section = 30 →
  area_per_section = 80 →
  (side_length ^ 2 / area_per_section) * sand_weight_per_section = 600 :=
by sorry

end NUMINAMATH_CALUDE_sandbox_sand_calculation_l673_67366


namespace NUMINAMATH_CALUDE_play_role_assignments_l673_67310

def number_of_assignments (men women : ℕ) (specific_male_roles specific_female_roles either_gender_roles : ℕ) : ℕ :=
  men * women * (Nat.choose (men + women - 2) either_gender_roles)

theorem play_role_assignments :
  number_of_assignments 6 7 1 1 4 = 13860 := by sorry

end NUMINAMATH_CALUDE_play_role_assignments_l673_67310


namespace NUMINAMATH_CALUDE_horner_method_v2_l673_67385

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 - x + 5

def horner_v2 (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₅ * x + a₄) * x + a₃) * x + a₂

theorem horner_method_v2 :
  horner_v2 2 0 (-3) 2 (-1) 5 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v2_l673_67385


namespace NUMINAMATH_CALUDE_g_difference_l673_67379

/-- A linear function g satisfying g(d+2) - g(d) = 8 for all real numbers d -/
def g_property (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x + y) = g x + g y) ∧ 
  (∀ d : ℝ, g (d + 2) - g d = 8)

theorem g_difference (g : ℝ → ℝ) (h : g_property g) : g 1 - g 7 = -24 := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l673_67379


namespace NUMINAMATH_CALUDE_largest_value_in_interval_l673_67319

theorem largest_value_in_interval (x : ℝ) (h : 0 < x ∧ x < 1) :
  max (max (max (max x (x^3)) (3*x)) (x^(1/3))) (1/x) = 1/x := by sorry

end NUMINAMATH_CALUDE_largest_value_in_interval_l673_67319


namespace NUMINAMATH_CALUDE_fair_haired_women_percentage_l673_67393

/-- Given that 32% of employees are women with fair hair and 80% of employees have fair hair,
    prove that 40% of fair-haired employees are women. -/
theorem fair_haired_women_percentage 
  (total_employees : ℝ) 
  (h1 : total_employees > 0)
  (women_fair_hair : ℝ) 
  (h2 : women_fair_hair = 0.32 * total_employees)
  (fair_haired : ℝ) 
  (h3 : fair_haired = 0.80 * total_employees) :
  women_fair_hair / fair_haired = 0.40 := by
sorry

end NUMINAMATH_CALUDE_fair_haired_women_percentage_l673_67393


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l673_67352

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 2 * x + 15 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y - 2 * y + 15 = 0 → y = x) ↔ 
  (m = -2 + 6 * Real.sqrt 5 ∨ m = -2 - 6 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l673_67352


namespace NUMINAMATH_CALUDE_science_club_committee_selection_l673_67392

theorem science_club_committee_selection (total_candidates : Nat) 
  (previously_served : Nat) (committee_size : Nat) 
  (h1 : total_candidates = 20) (h2 : previously_served = 8) 
  (h3 : committee_size = 4) :
  Nat.choose total_candidates committee_size - 
  Nat.choose (total_candidates - previously_served) committee_size = 4350 :=
by
  sorry

end NUMINAMATH_CALUDE_science_club_committee_selection_l673_67392


namespace NUMINAMATH_CALUDE_resulting_figure_sides_l673_67312

/-- Represents a polygon in the construction --/
structure Polygon :=
  (sides : ℕ)
  (adjacentSides : ℕ)

/-- The construction of polygons --/
def construction : List Polygon :=
  [{ sides := 3, adjacentSides := 1 },  -- isosceles triangle
   { sides := 4, adjacentSides := 2 },  -- rectangle
   { sides := 6, adjacentSides := 2 },  -- first hexagon
   { sides := 7, adjacentSides := 2 },  -- heptagon
   { sides := 6, adjacentSides := 2 },  -- second hexagon
   { sides := 9, adjacentSides := 1 }]  -- nonagon

theorem resulting_figure_sides :
  (construction.map (λ p => p.sides - p.adjacentSides)).sum = 25 := by
  sorry

end NUMINAMATH_CALUDE_resulting_figure_sides_l673_67312


namespace NUMINAMATH_CALUDE_favorite_toy_change_probability_l673_67376

def toy_count : ℕ := 10
def min_price : ℚ := 1/2
def max_price : ℚ := 5
def price_increment : ℚ := 1/2
def initial_quarters : ℕ := 10
def favorite_toy_price : ℚ := 9/2

def toy_prices : List ℚ := 
  List.range toy_count |>.map (λ i => max_price - i * price_increment)

theorem favorite_toy_change_probability :
  let total_sequences := toy_count.factorial
  let favorable_sequences := (toy_count - 1).factorial + (toy_count - 2).factorial
  (1 : ℚ) - (favorable_sequences : ℚ) / total_sequences = 8/9 :=
sorry

end NUMINAMATH_CALUDE_favorite_toy_change_probability_l673_67376


namespace NUMINAMATH_CALUDE_room_area_in_square_yards_l673_67370

/-- Proves that the area of a 15 ft by 10 ft rectangular room is 16.67 square yards -/
theorem room_area_in_square_yards :
  let length : ℝ := 15
  let width : ℝ := 10
  let sq_feet_per_sq_yard : ℝ := 9
  let area_sq_feet : ℝ := length * width
  let area_sq_yards : ℝ := area_sq_feet / sq_feet_per_sq_yard
  area_sq_yards = 16.67 := by sorry

end NUMINAMATH_CALUDE_room_area_in_square_yards_l673_67370


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l673_67371

theorem same_terminal_side (θ₁ θ₂ : Real) : 
  ∃ k : Int, θ₂ = θ₁ + 2 * π * k → 
  θ₁.cos = θ₂.cos ∧ θ₁.sin = θ₂.sin :=
by sorry

theorem angle_with_same_terminal_side : 
  ∃ k : Int, (11 * π / 8 : Real) = (-5 * π / 8 : Real) + 2 * π * k :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l673_67371


namespace NUMINAMATH_CALUDE_sum_in_interval_l673_67309

theorem sum_in_interval : 
  let sum := 4 + 3/8 + 5 + 3/4 + 7 + 2/25
  17 < sum ∧ sum < 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_interval_l673_67309


namespace NUMINAMATH_CALUDE_sin_cos_sum_implies_tan_value_l673_67369

theorem sin_cos_sum_implies_tan_value (x : ℝ) (h1 : x ∈ Set.Ioo 0 π) 
  (h2 : Real.sin x + Real.cos x = 3 * Real.sqrt 2 / 5) : 
  (1 - Real.cos (2 * x)) / Real.sin (2 * x) = -7 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_implies_tan_value_l673_67369


namespace NUMINAMATH_CALUDE_tan_alpha_value_l673_67383

theorem tan_alpha_value (α : Real) 
  (h : (Real.cos (π/4 - α)) / (Real.cos (π/4 + α)) = 1/2) : 
  Real.tan α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l673_67383


namespace NUMINAMATH_CALUDE_fraction_evaluation_l673_67308

theorem fraction_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/5) = 5/8 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l673_67308


namespace NUMINAMATH_CALUDE_solve_rain_problem_l673_67357

def rain_problem (x : ℝ) : Prop :=
  let monday_total := x + 1
  let tuesday := 2 * monday_total
  let wednesday := 0
  let thursday := 1
  let friday := monday_total + tuesday + wednesday + thursday
  let total_rain := monday_total + tuesday + wednesday + thursday + friday
  let daily_average := 4
  total_rain = 7 * daily_average ∧ x > 0

theorem solve_rain_problem :
  ∃ x : ℝ, rain_problem x ∧ x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_rain_problem_l673_67357


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l673_67314

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  |x| + |y| ≤ 2 * Real.sqrt 2 ∧ ∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l673_67314


namespace NUMINAMATH_CALUDE_power_four_mod_nine_l673_67328

theorem power_four_mod_nine : 4^215 % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_four_mod_nine_l673_67328


namespace NUMINAMATH_CALUDE_residue_problem_l673_67326

theorem residue_problem : (198 * 6 - 16 * 8^2 + 5) % 16 = 9 := by
  sorry

end NUMINAMATH_CALUDE_residue_problem_l673_67326


namespace NUMINAMATH_CALUDE_parallelogram_height_l673_67375

theorem parallelogram_height (area base height : ℝ) : 
  area = 612 ∧ base = 34 ∧ area = base * height → height = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l673_67375


namespace NUMINAMATH_CALUDE_work_division_proof_l673_67332

/-- The number of days it takes x to finish the entire work -/
def x_total_days : ℝ := 18

/-- The number of days it takes y to finish the entire work -/
def y_total_days : ℝ := 15

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining_days : ℝ := 12

/-- The number of days y worked before leaving the job -/
def y_worked_days : ℝ := 5

theorem work_division_proof :
  let total_work : ℝ := 1
  let x_rate : ℝ := total_work / x_total_days
  let y_rate : ℝ := total_work / y_total_days
  y_worked_days * y_rate + x_remaining_days * x_rate = total_work :=
by sorry

end NUMINAMATH_CALUDE_work_division_proof_l673_67332


namespace NUMINAMATH_CALUDE_reflection_after_translation_l673_67301

/-- Given a point A with coordinates (-3, -2), prove that translating it 5 units
    to the right and then reflecting across the y-axis results in a point with
    coordinates (-2, -2). -/
theorem reflection_after_translation :
  let A : ℝ × ℝ := (-3, -2)
  let B : ℝ × ℝ := (A.1 + 5, A.2)
  let B' : ℝ × ℝ := (-B.1, B.2)
  B' = (-2, -2) := by
sorry

end NUMINAMATH_CALUDE_reflection_after_translation_l673_67301


namespace NUMINAMATH_CALUDE_simplify_expression_l673_67336

theorem simplify_expression (x y : ℝ) : 7*x + 9*y + 3 - x + 12*y + 15 = 6*x + 21*y + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l673_67336


namespace NUMINAMATH_CALUDE_domain_of_g_l673_67354

-- Define the function f with domain (-3, 6)
def f : {x : ℝ // -3 < x ∧ x < 6} → ℝ := sorry

-- Define the function g(x) = f(2x)
def g (x : ℝ) : ℝ := f ⟨2*x, sorry⟩

-- Theorem statement
theorem domain_of_g :
  ∀ x : ℝ, (∃ y : ℝ, g x = y) ↔ -3/2 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l673_67354


namespace NUMINAMATH_CALUDE_smallest_valid_m_l673_67305

def is_valid_partition (m : ℕ) (partition : Fin 14 → Set ℕ) : Prop :=
  (∀ i, partition i ⊆ Finset.range (m + 1)) ∧
  (∀ x, x ∈ Finset.range (m + 1) → ∃ i, x ∈ partition i) ∧
  (∀ i j, i ≠ j → partition i ∩ partition j = ∅)

def has_valid_subset (m : ℕ) (partition : Fin 14 → Set ℕ) : Prop :=
  ∃ i : Fin 14, 1 < i.val ∧ i.val < 14 ∧
    ∃ a b : ℕ, a ∈ partition i ∧ b ∈ partition i ∧
      b < a ∧ (a : ℚ) ≤ 4/3 * (b : ℚ)

theorem smallest_valid_m :
  (∀ m < 56, ∃ partition : Fin 14 → Set ℕ,
    is_valid_partition m partition ∧ ¬has_valid_subset m partition) ∧
  (∀ partition : Fin 14 → Set ℕ,
    is_valid_partition 56 partition → has_valid_subset 56 partition) := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_m_l673_67305


namespace NUMINAMATH_CALUDE_thirteenth_term_l673_67372

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (a 1 + a 9 = 16) ∧
  (a 4 = 1)

/-- The 13th term of the arithmetic sequence is 64 -/
theorem thirteenth_term (a : ℕ → ℚ) (h : arithmetic_sequence a) : a 13 = 64 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_term_l673_67372


namespace NUMINAMATH_CALUDE_money_and_costs_problem_l673_67325

/-- The problem of determining the original amounts of money and costs of wine and wheat. -/
theorem money_and_costs_problem 
  (x y z u : ℚ) -- x: A's money, y: B's money, z: cost of 1 hl wine, u: cost of 1 hl wheat
  (h1 : y / 4 = 6 * z)
  (h2 : x / 5 = 8 * z)
  (h3 : (x + 46) + (y - 46) / 3 = 30 * u)
  (h4 : (y - 46) + (x + 46) / 3 = 36 * u)
  : x = 520 ∧ y = 312 ∧ z = 13 ∧ u = 50/3 := by
  sorry

end NUMINAMATH_CALUDE_money_and_costs_problem_l673_67325


namespace NUMINAMATH_CALUDE_probability_even_product_l673_67300

def setA : Finset ℕ := {1, 2, 3, 4}
def setB : Finset ℕ := {5, 6, 7, 8}

def isEven (n : ℕ) : Bool := n % 2 = 0

def evenProductPairs : Finset (ℕ × ℕ) :=
  setA.product setB |>.filter (fun (a, b) => isEven (a * b))

theorem probability_even_product :
  (evenProductPairs.card : ℚ) / ((setA.card * setB.card) : ℚ) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_product_l673_67300


namespace NUMINAMATH_CALUDE_geometric_progression_squared_sum_l673_67374

theorem geometric_progression_squared_sum 
  (q : ℝ) 
  (S : ℝ) 
  (h1 : abs q < 1) 
  (h2 : S = 1 / (1 - q)) : 
  1 / (1 - q^2) = S^2 / (2*S - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_squared_sum_l673_67374


namespace NUMINAMATH_CALUDE_vacation_class_ratio_l673_67348

theorem vacation_class_ratio :
  ∀ (grant_vacations : ℕ) (kelvin_classes : ℕ),
    kelvin_classes = 90 →
    grant_vacations + kelvin_classes = 450 →
    (grant_vacations : ℚ) / kelvin_classes = 4 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_class_ratio_l673_67348


namespace NUMINAMATH_CALUDE_one_third_1206_percent_of_200_l673_67386

theorem one_third_1206_percent_of_200 : (1206 / 3) / 200 * 100 = 201 := by
  sorry

end NUMINAMATH_CALUDE_one_third_1206_percent_of_200_l673_67386


namespace NUMINAMATH_CALUDE_perimeter_of_quarter_circle_bounded_square_l673_67398

/-- The perimeter of a region bounded by quarter-circle arcs constructed on each side of a square --/
theorem perimeter_of_quarter_circle_bounded_square (s : ℝ) (h : s = 4 / Real.pi) :
  4 * (Real.pi * s / 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_quarter_circle_bounded_square_l673_67398


namespace NUMINAMATH_CALUDE_triangle_area_l673_67334

/-- Given vectors m and n, and function f, prove the area of triangle ABC -/
theorem triangle_area (x : ℝ) :
  let m : ℝ × ℝ := (Real.sqrt 3 * Real.sin x - Real.cos x, 1)
  let n : ℝ × ℝ := (Real.cos x, 1/2)
  let f : ℝ → ℝ := λ x => m.1 * n.1 + m.2 * n.2
  let a : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 4
  ∀ A : ℝ, f A = 1 →
    ∃ b : ℝ, 
      let s := (a + b + c) / 2
      2 * Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l673_67334


namespace NUMINAMATH_CALUDE_hotel_room_charge_comparison_l673_67350

theorem hotel_room_charge_comparison (G P R : ℝ) 
  (hP_G : P = G * 0.8)
  (hR_G : R = G * 1.6) :
  (R - P) / R * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charge_comparison_l673_67350


namespace NUMINAMATH_CALUDE_smallest_multiple_of_4_to_8_exists_840_multiple_l673_67382

theorem smallest_multiple_of_4_to_8 : ∀ n : ℕ, n > 0 → (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (7 ∣ n) ∧ (8 ∣ n) → n ≥ 840 :=
by
  sorry

theorem exists_840_multiple : (4 ∣ 840) ∧ (5 ∣ 840) ∧ (6 ∣ 840) ∧ (7 ∣ 840) ∧ (8 ∣ 840) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_4_to_8_exists_840_multiple_l673_67382


namespace NUMINAMATH_CALUDE_barbed_wire_cost_l673_67330

theorem barbed_wire_cost (area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (cost_per_meter : ℝ) : area = 3136 ∧ gate_width = 1 ∧ num_gates = 2 ∧ cost_per_meter = 1 → (4 * Real.sqrt area - num_gates * gate_width) * cost_per_meter = 222 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_cost_l673_67330


namespace NUMINAMATH_CALUDE_family_ages_correct_l673_67323

-- Define the family members' ages as natural numbers
def son_age : Nat := 7
def daughter_age : Nat := 12
def man_age : Nat := 27
def wife_age : Nat := 22
def father_age : Nat := 59

-- State the theorem
theorem family_ages_correct :
  -- Man is 20 years older than son
  man_age = son_age + 20 ∧
  -- Man is 15 years older than daughter
  man_age = daughter_age + 15 ∧
  -- In two years, man's age will be twice son's age
  man_age + 2 = 2 * (son_age + 2) ∧
  -- In two years, man's age will be three times daughter's age
  man_age + 2 = 3 * (daughter_age + 2) ∧
  -- Wife is 5 years younger than man
  wife_age = man_age - 5 ∧
  -- In 6 years, wife will be twice as old as daughter
  wife_age + 6 = 2 * (daughter_age + 6) ∧
  -- Father is 32 years older than man
  father_age = man_age + 32 := by
  sorry


end NUMINAMATH_CALUDE_family_ages_correct_l673_67323


namespace NUMINAMATH_CALUDE_total_weight_compounds_l673_67395

/-- The atomic mass of Nitrogen in g/mol -/
def mass_N : ℝ := 14.01

/-- The atomic mass of Hydrogen in g/mol -/
def mass_H : ℝ := 1.01

/-- The atomic mass of Bromine in g/mol -/
def mass_Br : ℝ := 79.90

/-- The atomic mass of Magnesium in g/mol -/
def mass_Mg : ℝ := 24.31

/-- The atomic mass of Chlorine in g/mol -/
def mass_Cl : ℝ := 35.45

/-- The molar mass of Ammonium Bromide (NH4Br) in g/mol -/
def molar_mass_NH4Br : ℝ := mass_N + 4 * mass_H + mass_Br

/-- The molar mass of Magnesium Chloride (MgCl2) in g/mol -/
def molar_mass_MgCl2 : ℝ := mass_Mg + 2 * mass_Cl

/-- The number of moles of Ammonium Bromide -/
def moles_NH4Br : ℝ := 3.72

/-- The number of moles of Magnesium Chloride -/
def moles_MgCl2 : ℝ := 2.45

theorem total_weight_compounds : 
  moles_NH4Br * molar_mass_NH4Br + moles_MgCl2 * molar_mass_MgCl2 = 597.64 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_compounds_l673_67395


namespace NUMINAMATH_CALUDE_bijection_probability_l673_67355

-- Define sets A and B
def A : Set (Fin 2) := Set.univ
def B : Set (Fin 3) := Set.univ

-- Define the total number of mappings from A to B
def total_mappings : ℕ := 3^2

-- Define the number of bijective mappings from A to B
def bijective_mappings : ℕ := 3 * 2

-- Define the probability of a random mapping being bijective
def prob_bijective : ℚ := bijective_mappings / total_mappings

-- Theorem statement
theorem bijection_probability :
  prob_bijective = 2/3 := by sorry

end NUMINAMATH_CALUDE_bijection_probability_l673_67355


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l673_67320

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : x^2 + y^2 = 4) 
  (h2 : (x - y)^2 = 5) : 
  (x + y)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l673_67320


namespace NUMINAMATH_CALUDE_andrew_payment_l673_67324

/-- The total amount Andrew paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_weight : ℕ) (grape_rate : ℕ) (mango_weight : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_weight * grape_rate + mango_weight * mango_rate

/-- Theorem stating that Andrew paid 908 to the shopkeeper -/
theorem andrew_payment : total_amount 7 68 9 48 = 908 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l673_67324


namespace NUMINAMATH_CALUDE_complex_equation_solution_l673_67364

theorem complex_equation_solution :
  ∃ z : ℂ, 4 + 2 * Complex.I * z = 3 - 5 * Complex.I * z ∧ z = Complex.I / 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l673_67364


namespace NUMINAMATH_CALUDE_june_sales_increase_l673_67321

def normal_monthly_sales : ℕ := 21122
def june_july_combined_sales : ℕ := 46166

theorem june_sales_increase : 
  (june_july_combined_sales - 2 * normal_monthly_sales) = 3922 := by
  sorry

end NUMINAMATH_CALUDE_june_sales_increase_l673_67321


namespace NUMINAMATH_CALUDE_population_reproduction_after_development_l673_67346

/-- Represents the types of population reproduction --/
inductive PopulationReproductionType
  | Primitive
  | Traditional
  | TransitionToModern
  | Modern

/-- Represents the state of society after a major development of productive forces --/
structure SocietyState where
  productiveForcesDeveloped : Bool
  materialWealthIncreased : Bool
  populationGrowthRapid : Bool
  healthCareImproved : Bool
  mortalityRatesDecreased : Bool

/-- Determines the type of population reproduction based on the society state --/
def determinePopulationReproductionType (state : SocietyState) : PopulationReproductionType :=
  if state.productiveForcesDeveloped ∧
     state.materialWealthIncreased ∧
     state.populationGrowthRapid ∧
     state.healthCareImproved ∧
     state.mortalityRatesDecreased
  then PopulationReproductionType.Traditional
  else PopulationReproductionType.Primitive

/-- Theorem stating that after the first major development of productive forces, 
    the population reproduction type was Traditional --/
theorem population_reproduction_after_development 
  (state : SocietyState) 
  (h1 : state.productiveForcesDeveloped)
  (h2 : state.materialWealthIncreased)
  (h3 : state.populationGrowthRapid)
  (h4 : state.healthCareImproved)
  (h5 : state.mortalityRatesDecreased) :
  determinePopulationReproductionType state = PopulationReproductionType.Traditional := by
  sorry

end NUMINAMATH_CALUDE_population_reproduction_after_development_l673_67346


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l673_67397

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Define the point (4, -1) on circle C
def point_on_C : Prop := circle_C 4 (-1)

-- Define the new circle with center (a, b) and radius 1
def new_circle (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1

-- Define the tangency condition
def is_tangent (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), circle_C x y ∧ new_circle a b x y

-- The theorem to prove
theorem tangent_circle_equation :
  point_on_C →
  is_tangent 5 (-1) ∨ is_tangent 3 (-1) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l673_67397


namespace NUMINAMATH_CALUDE_equation_solutions_equation_solutions_unique_l673_67313

theorem equation_solutions :
  (∃ x : ℝ, (2*x + 3)^2 = 4*(2*x + 3)) ∧
  (∃ x : ℝ, x^2 - 4*x + 2 = 0) :=
by
  constructor
  · use -3/2
    sorry
  · use 2 + Real.sqrt 2
    sorry

theorem equation_solutions_unique :
  (∀ x : ℝ, (2*x + 3)^2 = 4*(2*x + 3) ↔ (x = -3/2 ∨ x = 1/2)) ∧
  (∀ x : ℝ, x^2 - 4*x + 2 = 0 ↔ (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2)) :=
by
  constructor
  · intro x
    sorry
  · intro x
    sorry

end NUMINAMATH_CALUDE_equation_solutions_equation_solutions_unique_l673_67313


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l673_67333

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, StrictMonoOn f (Set.Ioo 0 2) := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l673_67333


namespace NUMINAMATH_CALUDE_expression_value_l673_67356

theorem expression_value (x y : ℝ) (h : x - 2*y = 3) : 5 - 2*x + 4*y = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l673_67356


namespace NUMINAMATH_CALUDE_f_at_one_l673_67347

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem f_at_one : f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_f_at_one_l673_67347


namespace NUMINAMATH_CALUDE_total_workers_count_l673_67307

def num_other_workers : ℕ := 5

def probability_jack_and_jill : ℚ := 1 / 21

theorem total_workers_count (num_selected : ℕ) (h1 : num_selected = 2) :
  ∃ (total_workers : ℕ),
    total_workers = num_other_workers + 2 ∧
    probability_jack_and_jill = 1 / (total_workers.choose num_selected) :=
by sorry

end NUMINAMATH_CALUDE_total_workers_count_l673_67307


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l673_67394

theorem cubic_root_sum_product (a b : ℝ) : 
  (a^3 - 4*a^2 - a + 4 = 0) → 
  (b^3 - 4*b^2 - b + 4 = 0) → 
  (a ≠ b) →
  (a + b + a*b = -1) := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l673_67394


namespace NUMINAMATH_CALUDE_quadratic_symmetry_point_l673_67365

/-- A quadratic function f(x) with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*(a-1)

theorem quadratic_symmetry_point (a : ℝ) :
  (∀ x ≤ 4, (f_derivative a x ≤ 0)) ∧
  (∀ x ≥ 4, (f_derivative a x ≥ 0)) →
  a = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_point_l673_67365


namespace NUMINAMATH_CALUDE_money_sharing_l673_67322

theorem money_sharing (jane_share : ℕ) (total : ℕ) : 
  jane_share = 30 →
  (2 : ℕ) * total = jane_share * (2 + 3 + 8) →
  total = 195 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l673_67322


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l673_67378

/-- The distance between the foci of an ellipse with equation 
    √((x-4)² + (y-5)²) + √((x+6)² + (y-9)²) = 24 is equal to 2√29 -/
theorem ellipse_foci_distance : 
  let ellipse_eq (x y : ℝ) := Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24
  ∃ f₁ f₂ : ℝ × ℝ, (∀ x y : ℝ, ellipse_eq x y → Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) + Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2) = 24) ∧
             Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l673_67378


namespace NUMINAMATH_CALUDE_scooter_cost_l673_67318

/-- The cost of a scooter given the amount saved and the additional amount needed. -/
theorem scooter_cost (saved : ℕ) (needed : ℕ) (cost : ℕ) 
  (h1 : saved = 57) 
  (h2 : needed = 33) : 
  cost = saved + needed := by
  sorry

end NUMINAMATH_CALUDE_scooter_cost_l673_67318


namespace NUMINAMATH_CALUDE_series_sum_equals_three_halves_l673_67340

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-3)/(3^n) is equal to 3/2 -/
theorem series_sum_equals_three_halves :
  ∑' n, (4 * n - 3 : ℝ) / (3 : ℝ) ^ n = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_halves_l673_67340


namespace NUMINAMATH_CALUDE_train_length_l673_67315

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 16 → speed_kmh * (5/18) * time_s = 240 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l673_67315


namespace NUMINAMATH_CALUDE_range_of_a_when_A_union_B_equals_A_l673_67359

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < 3-a}

-- State the theorem
theorem range_of_a_when_A_union_B_equals_A :
  ∀ a : ℝ, (A ∪ B a = A) → a ≥ (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_a_when_A_union_B_equals_A_l673_67359


namespace NUMINAMATH_CALUDE_area_ratio_is_one_l673_67343

/-- Theorem: The ratio of the areas of rectangles M and N is 1 -/
theorem area_ratio_is_one (a b x y : ℝ) : a > 0 → b > 0 → x > 0 → y > 0 → 
  b * x + a * y = a * b → (x * y) / ((a - x) * (b - y)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_one_l673_67343


namespace NUMINAMATH_CALUDE_height_edge_relationship_l673_67303

/-- A triangular pyramid with mutually perpendicular edges -/
structure TriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  h_pos : 0 < h
  perpendicular : True  -- Represents that SA, SB, and SC are mutually perpendicular

/-- The theorem about the relationship between height and edge lengths in a triangular pyramid -/
theorem height_edge_relationship (p : TriangularPyramid) : 
  1 / p.h^2 = 1 / p.a^2 + 1 / p.b^2 + 1 / p.c^2 := by
  sorry

end NUMINAMATH_CALUDE_height_edge_relationship_l673_67303


namespace NUMINAMATH_CALUDE_triangle_identities_l673_67361

theorem triangle_identities (a b c α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum_angles : α + β + γ = π)
  (h_law_of_sines : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ) :
  (a + b) / c = Real.cos ((α - β) / 2) / Real.sin (γ / 2) ∧
  (a - b) / c = Real.sin ((α - β) / 2) / Real.cos (γ / 2) := by
sorry

end NUMINAMATH_CALUDE_triangle_identities_l673_67361
