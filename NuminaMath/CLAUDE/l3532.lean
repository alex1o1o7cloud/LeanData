import Mathlib

namespace NUMINAMATH_CALUDE_pencil_ratio_l3532_353243

/-- Given the number of pencils for Tyrah, Tim, and Sarah, prove the ratio of Tim's to Sarah's pencils -/
theorem pencil_ratio (sarah_pencils tyrah_pencils tim_pencils : ℕ) 
  (h1 : tyrah_pencils = 6 * sarah_pencils)
  (h2 : tyrah_pencils = 12)
  (h3 : tim_pencils = 16) :
  tim_pencils / sarah_pencils = 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_ratio_l3532_353243


namespace NUMINAMATH_CALUDE_time_to_weave_cloth_l3532_353283

/-- Represents the industrial loom's weaving rate and characteristics -/
structure Loom where
  rate : Real
  sample_time : Real
  sample_cloth : Real

/-- Theorem: Time to weave cloth -/
theorem time_to_weave_cloth (loom : Loom) (x : Real) :
  loom.rate = 0.128 ∧ 
  loom.sample_time = 195.3125 ∧ 
  loom.sample_cloth = 25 →
  x / loom.rate = x / 0.128 := by
  sorry

#check time_to_weave_cloth

end NUMINAMATH_CALUDE_time_to_weave_cloth_l3532_353283


namespace NUMINAMATH_CALUDE_eight_four_two_power_l3532_353277

theorem eight_four_two_power : 8^8 * 4^4 / 2^28 = 16 := by sorry

end NUMINAMATH_CALUDE_eight_four_two_power_l3532_353277


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3532_353290

theorem sin_cos_identity :
  Real.sin (20 * π / 180) ^ 2 + Real.cos (50 * π / 180) ^ 2 + 
  Real.sin (20 * π / 180) * Real.cos (50 * π / 180) = 1 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3532_353290


namespace NUMINAMATH_CALUDE_triangle_problem_l3532_353206

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  2 * c * Real.sin B = (2 * a - c) * Real.tan C →
  c = 3 * a →
  D = ((0 : ℝ), (c / 2 : ℝ)) →  -- Assuming A is at (0,0) and C is at (0,c)
  Real.sqrt ((D.1 - b)^2 + D.2^2) = Real.sqrt 13 →
  B = π / 3 ∧ a + b + c = 8 + 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3532_353206


namespace NUMINAMATH_CALUDE_dividend_calculation_l3532_353248

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (h1 : quotient = 36)
  (h2 : divisor = 85)
  (h3 : remainder = 26) :
  divisor * quotient + remainder = 3086 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3532_353248


namespace NUMINAMATH_CALUDE_no_intersection_empty_union_equality_iff_l3532_353292

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Theorem 1: There is no value of a such that A ∩ B = ∅
theorem no_intersection_empty (a : ℝ) : (A a) ∩ B ≠ ∅ := by
  sorry

-- Theorem 2: A ∪ B = B if and only if a ∈ (-∞, -4) ∪ (5, ∞)
theorem union_equality_iff (a : ℝ) : (A a) ∪ B = B ↔ a < -4 ∨ a > 5 := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_empty_union_equality_iff_l3532_353292


namespace NUMINAMATH_CALUDE_blackboard_sum_l3532_353204

def Operation : Type := List ℕ → List ℕ → (List ℕ × List ℕ)

def performOperations (initialBoard : List ℕ) (n : ℕ) (op : Operation) : (List ℕ × List ℕ) :=
  sorry

theorem blackboard_sum (initialBoard : List ℕ) (finalBoard : List ℕ) (paperNumbers : List ℕ) :
  initialBoard = [1, 3, 5, 7, 9] →
  (∃ op : Operation, performOperations initialBoard 4 op = (finalBoard, paperNumbers)) →
  finalBoard.length = 1 →
  paperNumbers.length = 4 →
  paperNumbers.sum = 230 :=
  sorry

end NUMINAMATH_CALUDE_blackboard_sum_l3532_353204


namespace NUMINAMATH_CALUDE_cost_per_shot_l3532_353295

def number_of_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def shots_per_puppy : ℕ := 2
def total_cost : ℕ := 120

theorem cost_per_shot :
  (total_cost : ℚ) / (number_of_dogs * puppies_per_dog * shots_per_puppy) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_shot_l3532_353295


namespace NUMINAMATH_CALUDE_defective_probability_l3532_353281

/-- Represents a box of components -/
structure Box where
  total : ℕ
  defective : ℕ

/-- The probability of selecting a box -/
def boxProb : ℚ := 1 / 2

/-- The probability of selecting a defective component from a given box -/
def defectiveProb (box : Box) : ℚ := box.defective / box.total

/-- The two boxes of components -/
def box1 : Box := ⟨10, 2⟩
def box2 : Box := ⟨20, 3⟩

/-- The main theorem stating the probability of selecting a defective component -/
theorem defective_probability : 
  boxProb * defectiveProb box1 + boxProb * defectiveProb box2 = 7 / 40 := by
  sorry

end NUMINAMATH_CALUDE_defective_probability_l3532_353281


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3532_353245

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3532_353245


namespace NUMINAMATH_CALUDE_snow_probability_value_l3532_353284

/-- The probability of snow occurring at least once in a week, where the first 4 days 
    have a 1/4 chance of snow each day and the next 3 days have a 1/3 chance of snow each day. -/
def snow_probability : ℚ := 1 - (3/4)^4 * (2/3)^3

/-- Theorem stating that the probability of snow occurring at least once in the described week
    is equal to 125/128. -/
theorem snow_probability_value : snow_probability = 125/128 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_value_l3532_353284


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3532_353299

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (x + 1)⁻¹ + y⁻¹ = (1 : ℝ) / 2) :
  ∀ a b : ℝ, a > 0 → b > 0 → (a + 1)⁻¹ + b⁻¹ = (1 : ℝ) / 2 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + 1)⁻¹ + y⁻¹ = (1 : ℝ) / 2 ∧ x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3532_353299


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l3532_353228

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes
  (α β : Plane) (m : Line)
  (h1 : perpendicular m α)
  (h2 : parallel α β) :
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l3532_353228


namespace NUMINAMATH_CALUDE_fold_length_is_ten_rectangle_fold_length_l3532_353278

/-- Represents a folded rectangle with specific properties -/
structure FoldedRectangle where
  short_side : ℝ
  long_side : ℝ
  fold_length : ℝ
  congruent_triangles : Prop

/-- The folded rectangle satisfies the problem conditions -/
def satisfies_conditions (r : FoldedRectangle) : Prop :=
  r.short_side = 12 ∧
  r.long_side = r.short_side * 3/2 ∧
  r.congruent_triangles

/-- The theorem to be proved -/
theorem fold_length_is_ten 
  (r : FoldedRectangle) 
  (h : satisfies_conditions r) : 
  r.fold_length = 10 := by
  sorry

/-- The main theorem restated in terms of the problem -/
theorem rectangle_fold_length :
  ∃ (r : FoldedRectangle), 
    satisfies_conditions r ∧ 
    r.fold_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_fold_length_is_ten_rectangle_fold_length_l3532_353278


namespace NUMINAMATH_CALUDE_units_digit_of_33_power_l3532_353262

theorem units_digit_of_33_power (n : ℕ) : 
  (33^(33 * (7^7))) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_33_power_l3532_353262


namespace NUMINAMATH_CALUDE_min_n_for_sqrt_27n_integer_l3532_353273

theorem min_n_for_sqrt_27n_integer (n : ℕ+) (h : ∃ k : ℕ, k^2 = 27 * n) :
  ∀ m : ℕ+, (∃ j : ℕ, j^2 = 27 * m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_sqrt_27n_integer_l3532_353273


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3532_353257

theorem diophantine_equation_solution :
  ∀ x y z : ℕ, 3^x + 4^y = 5^z →
    ((x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = 0 ∧ y = 1 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3532_353257


namespace NUMINAMATH_CALUDE_cycle_original_price_l3532_353235

/-- Proves that given a cycle sold at a loss of 18% with a selling price of 1148, the original price of the cycle was 1400. -/
theorem cycle_original_price (loss_percentage : ℝ) (selling_price : ℝ) (original_price : ℝ) : 
  loss_percentage = 18 →
  selling_price = 1148 →
  selling_price = (1 - loss_percentage / 100) * original_price →
  original_price = 1400 := by
sorry

end NUMINAMATH_CALUDE_cycle_original_price_l3532_353235


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3532_353218

theorem quadratic_factorization (x : ℝ) : x^2 - 5*x + 6 = (x - 2) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3532_353218


namespace NUMINAMATH_CALUDE_triangle_abc_obtuse_l3532_353202

theorem triangle_abc_obtuse (A B C : ℝ) (a b c : ℝ) : 
  B = 2 * A → a = 1 → b = 4/3 → 0 < A → A < π → B > π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_obtuse_l3532_353202


namespace NUMINAMATH_CALUDE_line_position_l3532_353231

structure Line3D where
  -- Assume we have a suitable representation for 3D lines
  -- This is just a placeholder

def skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

def intersects (l1 l2 : Line3D) : Prop :=
  -- Definition of intersecting lines
  sorry

theorem line_position (L1 L2 m1 m2 : Line3D) 
  (h1 : skew L1 L2)
  (h2 : intersects m1 L1)
  (h3 : intersects m1 L2)
  (h4 : intersects m2 L1)
  (h5 : intersects m2 L2) :
  intersects m1 m2 ∨ skew m1 m2 :=
by
  sorry

end NUMINAMATH_CALUDE_line_position_l3532_353231


namespace NUMINAMATH_CALUDE_zack_traveled_18_countries_l3532_353294

-- Define the number of countries each person traveled to
def george_countries : ℕ := 6
def joseph_countries : ℕ := george_countries / 2
def patrick_countries : ℕ := joseph_countries * 3
def zack_countries : ℕ := patrick_countries * 2

-- Theorem to prove
theorem zack_traveled_18_countries : zack_countries = 18 := by
  sorry

end NUMINAMATH_CALUDE_zack_traveled_18_countries_l3532_353294


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3532_353234

def A (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A a ⊆ B) ↔ (a = -2 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3532_353234


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3532_353254

theorem complex_modulus_problem (z : ℂ) : (1 - Complex.I) * z = Complex.I → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3532_353254


namespace NUMINAMATH_CALUDE_uma_income_is_20000_l3532_353217

-- Define the income ratio
def income_ratio : ℚ := 4 / 3

-- Define the expenditure ratio
def expenditure_ratio : ℚ := 3 / 2

-- Define the savings amount
def savings : ℕ := 5000

-- Define Uma's income as a function of x
def uma_income (x : ℚ) : ℚ := 4 * x

-- Define Bala's income as a function of x
def bala_income (x : ℚ) : ℚ := 3 * x

-- Define Uma's expenditure as a function of y
def uma_expenditure (y : ℚ) : ℚ := 3 * y

-- Define Bala's expenditure as a function of y
def bala_expenditure (y : ℚ) : ℚ := 2 * y

-- Theorem stating Uma's income is $20000
theorem uma_income_is_20000 :
  ∃ (x y : ℚ),
    uma_income x - uma_expenditure y = savings ∧
    bala_income x - bala_expenditure y = savings ∧
    uma_income x = 20000 :=
  sorry

end NUMINAMATH_CALUDE_uma_income_is_20000_l3532_353217


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3532_353263

theorem imaginary_power_sum : Complex.I ^ 22 + Complex.I ^ 222 = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3532_353263


namespace NUMINAMATH_CALUDE_masked_digits_unique_solution_l3532_353233

def is_valid_pair (d : Nat) : Bool :=
  let product := d * d
  product ≥ 10 ∧ product < 100 ∧ product % 10 ≠ d

def get_last_digit (n : Nat) : Nat :=
  n % 10

theorem masked_digits_unique_solution :
  ∃! (elephant mouse pig panda : Nat),
    elephant ≠ mouse ∧ elephant ≠ pig ∧ elephant ≠ panda ∧
    mouse ≠ pig ∧ mouse ≠ panda ∧
    pig ≠ panda ∧
    is_valid_pair mouse ∧
    get_last_digit (mouse * mouse) = elephant ∧
    elephant = 6 ∧ mouse = 4 ∧ pig = 8 ∧ panda = 1 :=
by sorry

end NUMINAMATH_CALUDE_masked_digits_unique_solution_l3532_353233


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_attainable_l3532_353209

theorem max_value_inequality (x y : ℝ) :
  (2 * x + 3 * y + 5) / Real.sqrt (2 * x^2 + 3 * y^2 + 7) ≤ Real.sqrt 38 :=
by sorry

theorem max_value_attainable :
  ∃ x y : ℝ, (2 * x + 3 * y + 5) / Real.sqrt (2 * x^2 + 3 * y^2 + 7) = Real.sqrt 38 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_attainable_l3532_353209


namespace NUMINAMATH_CALUDE_proposition_truth_l3532_353293

theorem proposition_truth (x y : ℝ) : x + y ≠ 3 → x ≠ 2 ∨ y ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l3532_353293


namespace NUMINAMATH_CALUDE_sum_18_47_in_base5_l3532_353258

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_18_47_in_base5 :
  toBase5 (18 + 47) = [2, 3, 0] :=
sorry

end NUMINAMATH_CALUDE_sum_18_47_in_base5_l3532_353258


namespace NUMINAMATH_CALUDE_problem_solution_l3532_353272

theorem problem_solution : 
  (∃ x : ℝ, x^2 = 6) ∧ (∃ y : ℝ, y^2 = 2) ∧ (∃ z : ℝ, z^2 = 27) ∧ (∃ w : ℝ, w^2 = 9) ∧ (∃ v : ℝ, v^2 = 1/3) →
  (∃ a b : ℝ, 
    (a^2 = 6 ∧ b^2 = 2 ∧ a * b + Real.sqrt 27 / Real.sqrt 9 - Real.sqrt (1/3) = 8 * Real.sqrt 3 / 3) ∧
    ((Real.sqrt 5 - 1)^2 - (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 5 - 2 * Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3532_353272


namespace NUMINAMATH_CALUDE_smallest_divisor_and_quadratic_form_l3532_353219

theorem smallest_divisor_and_quadratic_form : ∃ k : ℕ,
  (∃ n : ℕ, (2^n + 15) % k = 0) ∧
  (∃ x y : ℤ, k = 3*x^2 - 4*x*y + 3*y^2) ∧
  (∀ m : ℕ, m < k →
    (∃ n : ℕ, (2^n + 15) % m = 0) ∧
    (∃ x y : ℤ, m = 3*x^2 - 4*x*y + 3*y^2) →
    False) ∧
  k = 23 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_and_quadratic_form_l3532_353219


namespace NUMINAMATH_CALUDE_pages_read_second_day_l3532_353251

theorem pages_read_second_day 
  (total_pages : ℕ) 
  (pages_first_day : ℕ) 
  (pages_left : ℕ) 
  (h1 : total_pages = 95) 
  (h2 : pages_first_day = 18) 
  (h3 : pages_left = 19) : 
  total_pages - pages_left - pages_first_day = 58 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_second_day_l3532_353251


namespace NUMINAMATH_CALUDE_function_characterization_l3532_353216

/-- Given a positive real number α, prove that any function f from positive integers to reals
    satisfying f(k + m) = f(k) + f(m) for any positive integers k and m where αm ≤ k < (α + 1)m,
    must be of the form f(n) = bn for some real number b and all positive integers n. -/
theorem function_characterization (α : ℝ) (hα : α > 0) :
  ∀ f : ℕ+ → ℝ,
  (∀ (k m : ℕ+), α * m.val ≤ k.val ∧ k.val < (α + 1) * m.val → f (k + m) = f k + f m) →
  ∃ b : ℝ, ∀ n : ℕ+, f n = b * n.val :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3532_353216


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_irrational_l3532_353214

theorem inscribed_circle_radius_irrational (b c : ℕ) : 
  b ≥ 1 → c ≥ 1 → 1 + b > c → 1 + c > b → b + c > 1 → 
  ¬ ∃ (r : ℚ), r = (Real.sqrt ((b : ℝ)^2 - 1/4)) / (1 + 2*(b : ℝ)) := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_irrational_l3532_353214


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l3532_353250

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l3532_353250


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l3532_353274

/-- Calculates the number of employees to be drawn from a department in a stratified sampling method. -/
def stratified_sample_size (total_employees : ℕ) (sample_size : ℕ) (department_size : ℕ) : ℕ :=
  (department_size * sample_size) / total_employees

/-- Theorem stating that for a company with 240 employees and a sample size of 20,
    the number of employees to be drawn from a department with 60 employees is 5. -/
theorem stratified_sample_theorem :
  stratified_sample_size 240 20 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l3532_353274


namespace NUMINAMATH_CALUDE_sequence_equality_l3532_353242

theorem sequence_equality (n : ℕ+) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l3532_353242


namespace NUMINAMATH_CALUDE_vector_addition_l3532_353244

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![1, -1]

theorem vector_addition :
  (vector_a + vector_b) = ![2, 1] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l3532_353244


namespace NUMINAMATH_CALUDE_camdens_dogs_legs_l3532_353253

def number_of_dogs (name : String) : ℕ :=
  match name with
  | "Justin" => 14
  | "Rico" => 24
  | "Camden" => 18
  | _ => 0

theorem camdens_dogs_legs : 
  (∀ (name : String), number_of_dogs name ≥ 0) →
  number_of_dogs "Rico" = number_of_dogs "Justin" + 10 →
  number_of_dogs "Camden" = (3 * number_of_dogs "Rico") / 4 →
  number_of_dogs "Camden" * 4 = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_camdens_dogs_legs_l3532_353253


namespace NUMINAMATH_CALUDE_compound_bar_chart_must_have_legend_l3532_353268

/-- Represents a compound bar chart -/
structure CompoundBarChart where
  distinguishes_two_quantities : Bool
  uses_different_colors_or_patterns : Bool

/-- Theorem: A compound bar chart must have a clearly indicated legend -/
theorem compound_bar_chart_must_have_legend (chart : CompoundBarChart) 
  (h1 : chart.distinguishes_two_quantities = true)
  (h2 : chart.uses_different_colors_or_patterns = true) : 
  ∃ legend : Bool, legend = true :=
sorry

end NUMINAMATH_CALUDE_compound_bar_chart_must_have_legend_l3532_353268


namespace NUMINAMATH_CALUDE_arithmetic_mean_product_l3532_353203

theorem arithmetic_mean_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 ∧ 
  a = 14 ∧ 
  b = 25 ∧ 
  c + 3 = d → 
  c * d = 418 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_product_l3532_353203


namespace NUMINAMATH_CALUDE_count_numerators_T_l3532_353260

/-- The set of rational numbers with repeating decimal expansion 0.overline(ab) -/
def T : Set ℚ :=
  {r | 0 < r ∧ r < 1 ∧ ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ r = (10 * a + b : ℚ) / 99}

/-- The number of different numerators required to express all elements of T in lowest terms -/
def num_different_numerators : ℕ := 53

/-- Theorem stating that the number of different numerators for T is 53 -/
theorem count_numerators_T : num_different_numerators = 53 := by
  sorry

end NUMINAMATH_CALUDE_count_numerators_T_l3532_353260


namespace NUMINAMATH_CALUDE_siblings_selection_probability_l3532_353232

theorem siblings_selection_probability 
  (p_ram : ℚ) (p_ravi : ℚ) (p_rina : ℚ)
  (h_ram : p_ram = 4/7)
  (h_ravi : p_ravi = 1/5)
  (h_rina : p_rina = 3/8) :
  p_ram * p_ravi * p_rina = 3/70 := by
sorry

end NUMINAMATH_CALUDE_siblings_selection_probability_l3532_353232


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l3532_353221

theorem min_value_of_sum_of_ratios (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  ∀ x y, x > 0 ∧ y > 0 → (b / (c + d)) + (c / (a + b)) ≥ Real.sqrt 2 - 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l3532_353221


namespace NUMINAMATH_CALUDE_min_dot_product_l3532_353298

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

-- Define the fixed point E
def E : ℝ × ℝ := (3, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define perpendicularity condition
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define dot product of vectors
def dot_product (A B C D : ℝ × ℝ) : ℝ :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2)

-- Theorem statement
theorem min_dot_product :
  ∀ P Q : ℝ × ℝ, 
  point_on_ellipse P → 
  point_on_ellipse Q → 
  perpendicular E P Q → 
  ∃ m : ℝ, 
  (∀ P' Q' : ℝ × ℝ, 
    point_on_ellipse P' → 
    point_on_ellipse Q' → 
    perpendicular E P' Q' → 
    m ≤ dot_product E P P Q) ∧ 
  m = 6 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_l3532_353298


namespace NUMINAMATH_CALUDE_intersection_condition_l3532_353215

/-- The set of possible values for a real number a, given the conditions. -/
def PossibleValues : Set ℝ := {-1, 0, 1}

/-- The set A defined by the equation ax + 1 = 0. -/
def A (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

/-- The set B containing -1 and 1. -/
def B : Set ℝ := {-1, 1}

/-- Theorem stating that if A ∩ B = A, then a must be in the set of possible values. -/
theorem intersection_condition (a : ℝ) : A a ∩ B = A a → a ∈ PossibleValues := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3532_353215


namespace NUMINAMATH_CALUDE_evaluate_expression_l3532_353282

theorem evaluate_expression : -(16 / 4 * 8 - 70 + 4 * 7) = 10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3532_353282


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3532_353225

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  arithmetic_property : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_of_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  sum_of_terms seq 9 = 54 → 2 + seq.a 4 + 9 = 307 / 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3532_353225


namespace NUMINAMATH_CALUDE_three_roots_iff_b_in_range_l3532_353297

/-- The function f(x) = 2x³ - 3x² + 1 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

/-- The statement that f(x) + b = 0 has three distinct real roots iff -1 < b < 0 -/
theorem three_roots_iff_b_in_range (b : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x + b = 0 ∧ f y + b = 0 ∧ f z + b = 0) ↔ 
  -1 < b ∧ b < 0 :=
sorry

end NUMINAMATH_CALUDE_three_roots_iff_b_in_range_l3532_353297


namespace NUMINAMATH_CALUDE_sheilas_extra_flour_l3532_353265

/-- Given that Katie needs 3 pounds of flour and the total flour needed is 8 pounds,
    prove that Sheila needs 2 pounds more flour than Katie. -/
theorem sheilas_extra_flour (katie_flour sheila_flour total_flour : ℕ) : 
  katie_flour = 3 → 
  total_flour = 8 → 
  sheila_flour = total_flour - katie_flour →
  sheila_flour - katie_flour = 2 := by
  sorry

end NUMINAMATH_CALUDE_sheilas_extra_flour_l3532_353265


namespace NUMINAMATH_CALUDE_equation_solutions_l3532_353264

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 36 = 0 ↔ x = 6 ∨ x = -6) ∧
  (∀ x : ℝ, (x+1)^3 + 27 = 0 ↔ x = -4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3532_353264


namespace NUMINAMATH_CALUDE_similar_triangles_exist_l3532_353210

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorFunction : Point → Color := sorry

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define similarity ratio between two triangles
def similarityRatio (T1 T2 : Triangle) : ℝ := sorry

-- Define a predicate to check if all vertices of a triangle have the same color
def sameColor (T : Triangle) : Prop :=
  colorFunction T.A = colorFunction T.B ∧ colorFunction T.B = colorFunction T.C

-- The main theorem
theorem similar_triangles_exist :
  ∃ (T1 T2 : Triangle), similarityRatio T1 T2 = 1995 ∧ sameColor T1 ∧ sameColor T2 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_exist_l3532_353210


namespace NUMINAMATH_CALUDE_f_continuous_at_x₀_delta_epsilon_relation_l3532_353222

def f (x : ℝ) : ℝ := 5 * x^2 + 1

def x₀ : ℝ := 7

theorem f_continuous_at_x₀ :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀| < ε :=
sorry

theorem delta_epsilon_relation :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 70 ∧
    ∀ x, |x - x₀| < δ → |f x - f x₀| < ε :=
sorry

end NUMINAMATH_CALUDE_f_continuous_at_x₀_delta_epsilon_relation_l3532_353222


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3532_353285

theorem perfect_square_condition (x y k : ℝ) : 
  (∃ a : ℝ, x^2 + k*x*y + 81*y^2 = a^2) ↔ k = 18 ∨ k = -18 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3532_353285


namespace NUMINAMATH_CALUDE_max_height_of_smaller_box_l3532_353236

/-- The maximum height of a smaller box that can fit in a larger box --/
theorem max_height_of_smaller_box 
  (large_length : ℝ) (large_width : ℝ) (large_height : ℝ)
  (small_length : ℝ) (small_width : ℝ)
  (max_boxes : ℕ) :
  large_length = 6 →
  large_width = 5 →
  large_height = 4 →
  small_length = 0.6 →
  small_width = 0.5 →
  max_boxes = 1000 →
  ∃ (h : ℝ), h ≤ 0.4 ∧ 
    (max_boxes : ℝ) * small_length * small_width * h ≤ 
    large_length * large_width * large_height :=
by sorry

end NUMINAMATH_CALUDE_max_height_of_smaller_box_l3532_353236


namespace NUMINAMATH_CALUDE_christopher_karen_money_difference_l3532_353240

theorem christopher_karen_money_difference : 
  let karen_quarters : ℕ := 32
  let christopher_quarters : ℕ := 64
  let quarter_value : ℚ := 1/4
  (christopher_quarters - karen_quarters) * quarter_value = 8 := by sorry

end NUMINAMATH_CALUDE_christopher_karen_money_difference_l3532_353240


namespace NUMINAMATH_CALUDE_half_times_two_thirds_times_three_fourths_l3532_353255

theorem half_times_two_thirds_times_three_fourths :
  (1 / 2 : ℚ) * (2 / 3 : ℚ) * (3 / 4 : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_half_times_two_thirds_times_three_fourths_l3532_353255


namespace NUMINAMATH_CALUDE_triangle_ratio_l3532_353237

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  b * (Real.cos C) + c * (Real.cos B) = 2 * b →
  a / b = 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_ratio_l3532_353237


namespace NUMINAMATH_CALUDE_scooter_rental_proof_l3532_353230

/-- Represents the rental cost structure for an electric scooter service -/
structure RentalCost where
  fixed : ℝ
  per_minute : ℝ

/-- Calculates the total cost for a given duration -/
def total_cost (rc : RentalCost) (duration : ℝ) : ℝ :=
  rc.fixed + rc.per_minute * duration

theorem scooter_rental_proof (rc : RentalCost) 
  (h1 : total_cost rc 3 = 78)
  (h2 : total_cost rc 8 = 108) :
  total_cost rc 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_scooter_rental_proof_l3532_353230


namespace NUMINAMATH_CALUDE_A_infinite_B_infinite_unique_representation_l3532_353220

/-- Two infinite sets of non-negative integers -/
def A : Set ℕ := sorry

/-- Two infinite sets of non-negative integers -/
def B : Set ℕ := sorry

/-- A is infinite -/
theorem A_infinite : Set.Infinite A := by sorry

/-- B is infinite -/
theorem B_infinite : Set.Infinite B := by sorry

/-- Every non-negative integer can be uniquely represented as a sum of elements from A and B -/
theorem unique_representation :
  ∀ n : ℕ, ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b := by sorry

end NUMINAMATH_CALUDE_A_infinite_B_infinite_unique_representation_l3532_353220


namespace NUMINAMATH_CALUDE_problem_statement_l3532_353212

theorem problem_statement (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 - x^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3532_353212


namespace NUMINAMATH_CALUDE_group_size_calculation_l3532_353226

theorem group_size_calculation (average_increase : ℝ) (weight_difference : ℝ) : 
  average_increase = 3 ∧ weight_difference = 30 → 
  (weight_difference / average_increase : ℝ) = 10 := by
  sorry

#check group_size_calculation

end NUMINAMATH_CALUDE_group_size_calculation_l3532_353226


namespace NUMINAMATH_CALUDE_sqrt_2_times_sqrt_24_between_6_and_7_l3532_353224

theorem sqrt_2_times_sqrt_24_between_6_and_7 : 6 < Real.sqrt 2 * Real.sqrt 24 ∧ Real.sqrt 2 * Real.sqrt 24 < 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_times_sqrt_24_between_6_and_7_l3532_353224


namespace NUMINAMATH_CALUDE_constant_function_proof_l3532_353256

theorem constant_function_proof (f g h : ℕ → ℕ)
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l3532_353256


namespace NUMINAMATH_CALUDE_symmetric_point_on_number_line_l3532_353201

/-- Given points A, B, and C on a number line, where A represents √7, B represents 1,
    and C is symmetric to A with respect to B, prove that C represents 2 - √7. -/
theorem symmetric_point_on_number_line (A B C : ℝ) : 
  A = Real.sqrt 7 → B = 1 → (A + C) / 2 = B → C = 2 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_on_number_line_l3532_353201


namespace NUMINAMATH_CALUDE_minus_510_in_third_quadrant_l3532_353267

-- Define a function to normalize an angle to the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Define a function to determine the quadrant of an angle
def getQuadrant (angle : Int) : Nat :=
  let normalizedAngle := normalizeAngle angle
  if 0 < normalizedAngle && normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle && normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle && normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem minus_510_in_third_quadrant :
  getQuadrant (-510) = 3 :=
sorry

end NUMINAMATH_CALUDE_minus_510_in_third_quadrant_l3532_353267


namespace NUMINAMATH_CALUDE_complex_modulus_range_l3532_353227

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z = Complex.mk a 1) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l3532_353227


namespace NUMINAMATH_CALUDE_equation_solutions_l3532_353280

-- Define the equation
def equation (x y : ℝ) : Prop :=
  (36 / Real.sqrt (abs x)) + (9 / Real.sqrt (abs y)) = 
  42 - 9 * (if x < 0 then Complex.I * Real.sqrt (abs x) else Real.sqrt x) - 
  (if y < 0 then Complex.I * Real.sqrt (abs y) else Real.sqrt y)

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) :=
  {(4, 9), (-4, 873 + 504 * Real.sqrt 3), (-4, 873 - 504 * Real.sqrt 3), 
   ((62 + 14 * Real.sqrt 13) / 9, -9), ((62 - 14 * Real.sqrt 13) / 9, -9)}

-- Theorem statement
theorem equation_solutions :
  ∀ x y : ℝ, equation x y ↔ (x, y) ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3532_353280


namespace NUMINAMATH_CALUDE_marathon_heart_beats_l3532_353213

/-- Calculates the number of heart beats during a marathon --/
def marathonHeartBeats (totalDistance : ℕ) (heartRate : ℕ) (firstHalfDistance : ℕ) (firstHalfPace : ℕ) (secondHalfPace : ℕ) : ℕ :=
  let firstHalfTime := firstHalfDistance * firstHalfPace
  let secondHalfTime := (totalDistance - firstHalfDistance) * secondHalfPace
  let totalTime := firstHalfTime + secondHalfTime
  totalTime * heartRate

/-- Theorem: The athlete's heart beats 23100 times during the marathon --/
theorem marathon_heart_beats :
  marathonHeartBeats 30 140 15 6 5 = 23100 := by
  sorry

#eval marathonHeartBeats 30 140 15 6 5

end NUMINAMATH_CALUDE_marathon_heart_beats_l3532_353213


namespace NUMINAMATH_CALUDE_arrangement_problem_l3532_353269

theorem arrangement_problem (n : ℕ) (h1 : n ≥ 2) : 
  ((n - 1) * (n - 1) = 25) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_problem_l3532_353269


namespace NUMINAMATH_CALUDE_room_width_is_twelve_l3532_353261

/-- Represents a rectangular room with a veranda -/
structure RoomWithVeranda where
  length : ℝ
  width : ℝ
  verandaWidth : ℝ

/-- Calculates the area of the veranda -/
def verandaArea (room : RoomWithVeranda) : ℝ :=
  (room.length + 2 * room.verandaWidth) * (room.width + 2 * room.verandaWidth) - room.length * room.width

/-- Theorem: If a rectangular room has length 19 m, is surrounded by a 2 m wide veranda,
    and the veranda area is 140 m², then the room's width is 12 m -/
theorem room_width_is_twelve
  (room : RoomWithVeranda)
  (h1 : room.length = 19)
  (h2 : room.verandaWidth = 2)
  (h3 : verandaArea room = 140) :
  room.width = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_width_is_twelve_l3532_353261


namespace NUMINAMATH_CALUDE_sum_product_inequality_l3532_353270

theorem sum_product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x*y + y*z + z*x) * (1/(x+y)^2 + 1/(y+z)^2 + 1/(z+x)^2) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l3532_353270


namespace NUMINAMATH_CALUDE_largest_k_for_tree_graph_condition_l3532_353211

/-- A tree graph with k vertices -/
structure TreeGraph (k : ℕ) where
  (vertices : Finset (Fin k))
  (edges : Finset (Fin k × Fin k))
  -- Add properties to ensure it's a tree

/-- Path between two vertices in a graph -/
def path (G : TreeGraph k) (u v : Fin k) : Finset (Fin k) := sorry

/-- Length of a path -/
def pathLength (p : Finset (Fin k)) : ℕ := sorry

/-- The condition for the existence of vertices u and v -/
def satisfiesCondition (G : TreeGraph k) (m n : ℕ) : Prop :=
  ∃ u v : Fin k, ∀ w : Fin k, 
    (∃ p : Finset (Fin k), p = path G u w ∧ pathLength p ≤ m) ∨
    (∃ p : Finset (Fin k), p = path G v w ∧ pathLength p ≤ n)

theorem largest_k_for_tree_graph_condition (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∀ k : ℕ, k ≤ min (2*n + 2*m + 2) (3*n + 2) → 
    ∀ G : TreeGraph k, satisfiesCondition G m n) ∧
  (∀ k : ℕ, k > min (2*n + 2*m + 2) (3*n + 2) → 
    ∃ G : TreeGraph k, ¬satisfiesCondition G m n) :=
sorry

end NUMINAMATH_CALUDE_largest_k_for_tree_graph_condition_l3532_353211


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3532_353252

theorem complex_fraction_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  (5 + 7*i) / (3 - 4*i) = (43 : ℚ)/25 + (41 : ℚ)/25 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3532_353252


namespace NUMINAMATH_CALUDE_max_sum_of_three_primes_l3532_353200

theorem max_sum_of_three_primes (a b c : ℕ) : 
  Prime a → Prime b → Prime c →
  a < b → b < c → c < 100 →
  (b - a) * (c - b) * (c - a) = 240 →
  a + b + c ≤ 111 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_three_primes_l3532_353200


namespace NUMINAMATH_CALUDE_jason_total_games_l3532_353238

/-- The number of football games Jason attended or plans to attend each month from January to July -/
def games_per_month : List Nat := [11, 17, 16, 20, 14, 14, 14]

/-- The total number of games Jason will have attended by the end of July -/
def total_games : Nat := games_per_month.sum

theorem jason_total_games : total_games = 106 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_games_l3532_353238


namespace NUMINAMATH_CALUDE_average_first_twelve_even_numbers_l3532_353239

-- Define the first 12 even numbers
def firstTwelveEvenNumbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- Theorem to prove
theorem average_first_twelve_even_numbers :
  (List.sum firstTwelveEvenNumbers) / (List.length firstTwelveEvenNumbers) = 13 := by
  sorry


end NUMINAMATH_CALUDE_average_first_twelve_even_numbers_l3532_353239


namespace NUMINAMATH_CALUDE_sector_area_with_diameter_4_and_angle_90_l3532_353286

theorem sector_area_with_diameter_4_and_angle_90 (π : Real) :
  let diameter : Real := 4
  let centralAngle : Real := 90
  let radius : Real := diameter / 2
  let sectorArea : Real := (centralAngle / 360) * π * radius^2
  sectorArea = π := by sorry

end NUMINAMATH_CALUDE_sector_area_with_diameter_4_and_angle_90_l3532_353286


namespace NUMINAMATH_CALUDE_population_growth_proof_l3532_353246

/-- The annual growth rate of the population -/
def annual_growth_rate : ℝ := 0.1

/-- The population after 2 years -/
def population_after_2_years : ℕ := 15730

/-- The initial population of the town -/
def initial_population : ℕ := 13000

theorem population_growth_proof :
  (1 + annual_growth_rate) * (1 + annual_growth_rate) * initial_population = population_after_2_years := by
  sorry

end NUMINAMATH_CALUDE_population_growth_proof_l3532_353246


namespace NUMINAMATH_CALUDE_lily_catches_mary_l3532_353271

/-- Mary's walking speed in miles per hour -/
def mary_speed : ℝ := 4

/-- Lily's walking speed in miles per hour -/
def lily_speed : ℝ := 6

/-- Initial distance between Mary and Lily in miles -/
def initial_distance : ℝ := 2

/-- Time in minutes for Lily to catch up to Mary -/
def catch_up_time : ℝ := 60

theorem lily_catches_mary : 
  (lily_speed - mary_speed) * catch_up_time / 60 = initial_distance := by
  sorry

end NUMINAMATH_CALUDE_lily_catches_mary_l3532_353271


namespace NUMINAMATH_CALUDE_triangle_expression_simplification_l3532_353229

/-- Given a triangle with side lengths x, y, and z, prove that |x+y-z|-2|y-x-z| = -x + 3y - 3z -/
theorem triangle_expression_simplification
  (x y z : ℝ)
  (hxy : x + y > z)
  (hyz : y + z > x)
  (hxz : x + z > y) :
  |x + y - z| - 2 * |y - x - z| = -x + 3*y - 3*z := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_simplification_l3532_353229


namespace NUMINAMATH_CALUDE_division_problem_l3532_353296

theorem division_problem (L S Q : ℕ) : 
  L - S = 1500 → 
  L = 1782 → 
  L = S * Q + 15 → 
  Q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3532_353296


namespace NUMINAMATH_CALUDE_min_weighings_required_l3532_353275

/-- Represents a 4x4 grid of coins -/
def CoinGrid := Fin 4 → Fin 4 → ℕ

/-- Predicate to check if two positions are adjacent in the grid -/
def adjacent (p q : Fin 4 × Fin 4) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ q.2.val + 1 = p.2.val)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ q.1.val + 1 = p.1.val))

/-- A valid coin grid satisfying the problem conditions -/
def valid_coin_grid (g : CoinGrid) : Prop :=
  ∃ p q : Fin 4 × Fin 4,
    adjacent p q ∧
    g p.1 p.2 = 9 ∧ g q.1 q.2 = 9 ∧
    ∀ r : Fin 4 × Fin 4, (r ≠ p ∧ r ≠ q) → g r.1 r.2 = 10

/-- A weighing selects a subset of coins and returns their total weight -/
def Weighing := Set (Fin 4 × Fin 4) → ℕ

/-- The theorem stating the minimum number of weighings required -/
theorem min_weighings_required (g : CoinGrid) (h : valid_coin_grid g) :
  ∃ (w₁ w₂ w₃ : Weighing),
    (∀ g₁ g₂ : CoinGrid, valid_coin_grid g₁ → valid_coin_grid g₂ →
      (∀ S : Set (Fin 4 × Fin 4), w₁ S = w₁ S → w₂ S = w₂ S → w₃ S = w₃ S) →
      g₁ = g₂) ∧
    (∀ w₁' w₂' : Weighing,
      ¬∀ g₁ g₂ : CoinGrid, valid_coin_grid g₁ → valid_coin_grid g₂ →
        (∀ S : Set (Fin 4 × Fin 4), w₁' S = w₁' S → w₂' S = w₂' S) →
        g₁ = g₂) :=
by
  sorry

end NUMINAMATH_CALUDE_min_weighings_required_l3532_353275


namespace NUMINAMATH_CALUDE_cloth_sale_proof_l3532_353291

/-- Given a trader selling cloth with a profit of 55 per meter and a total profit of 2200,
    prove that the number of meters sold is 40. -/
theorem cloth_sale_proof (profit_per_meter : ℕ) (total_profit : ℕ) 
    (h1 : profit_per_meter = 55) (h2 : total_profit = 2200) : 
    total_profit / profit_per_meter = 40 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_proof_l3532_353291


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l3532_353207

/-- A circle passing through two points and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_origin : center.1^2 + center.2^2 = radius^2
  passes_point : (center.1 - 4)^2 + center.2^2 = radius^2
  tangent_to_line : |center.2 - 1| = radius

/-- The equation of the circle -/
def circle_equation (c : TangentCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem tangent_circle_equation :
  ∀ (c : TangentCircle),
    ∀ (x y : ℝ),
      circle_equation c x y ↔ (x - 2)^2 + (y + 3/2)^2 = 25/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l3532_353207


namespace NUMINAMATH_CALUDE_complex_magnitude_l3532_353288

theorem complex_magnitude (z : ℂ) : z = (2 - Complex.I) / Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3532_353288


namespace NUMINAMATH_CALUDE_green_balls_count_l3532_353276

def bag_problem (blue_balls : ℕ) (prob_blue : ℚ) (red_balls : ℕ) (green_balls : ℕ) : Prop :=
  blue_balls = 10 ∧ 
  prob_blue = 2/7 ∧ 
  red_balls = 2 * blue_balls ∧
  prob_blue = blue_balls / (blue_balls + red_balls + green_balls)

theorem green_balls_count : 
  ∃ (green_balls : ℕ), bag_problem 10 (2/7) 20 green_balls → green_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l3532_353276


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3532_353249

theorem algebraic_expression_equality (a b : ℝ) (h : 5 * a + 3 * b = -4) :
  2 * (a + b) + 4 * (2 * a + b + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3532_353249


namespace NUMINAMATH_CALUDE_sequences_sum_product_l3532_353241

/-- Two sequences satisfying the given conditions -/
def Sequences (α β γ : ℕ) (a b : ℕ → ℕ) : Prop :=
  (α < γ) ∧ 
  (α * γ = β^2 + 1) ∧
  (a 0 = 1) ∧ 
  (b 0 = 1) ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n)

/-- The main theorem to be proved -/
theorem sequences_sum_product (α β γ : ℕ) (a b : ℕ → ℕ) 
  (h : Sequences α β γ a b) :
  ∀ m n : ℕ, a (m + n) + b (m + n) = a m * a n + b m * b n :=
sorry

end NUMINAMATH_CALUDE_sequences_sum_product_l3532_353241


namespace NUMINAMATH_CALUDE_midpoint_triangle_area_l3532_353208

/-- The area of the nth triangle formed by repeatedly connecting midpoints -/
def triangleArea (n : ℕ) : ℚ :=
  (1 / 4 : ℚ) ^ (n - 1) * (3 / 2 : ℚ)

/-- The original right triangle ABC with sides 3, 4, and 5 -/
structure OriginalTriangle where
  sideA : ℕ := 3
  sideB : ℕ := 4
  sideC : ℕ := 5

theorem midpoint_triangle_area (t : OriginalTriangle) (n : ℕ) (h : n ≥ 1) :
  triangleArea n = (1 / 4 : ℚ) ^ (n - 1) * (3 / 2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_triangle_area_l3532_353208


namespace NUMINAMATH_CALUDE_root_in_interval_l3532_353259

-- Define the function f(x) = x³ - 4
def f (x : ℝ) : ℝ := x^3 - 4

-- State the theorem
theorem root_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, x ∈ (Set.Ioo 1 2) ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l3532_353259


namespace NUMINAMATH_CALUDE_project_completion_equation_l3532_353205

/-- Represents the number of days required for a person to complete the project alone -/
structure ProjectTime where
  person_a : ℝ
  person_b : ℝ

/-- Represents the work schedule for the project -/
structure WorkSchedule where
  solo_days : ℝ
  total_days : ℝ

/-- Theorem stating the equation for the total number of days required to complete the project -/
theorem project_completion_equation (pt : ProjectTime) (ws : WorkSchedule) :
  pt.person_a = 12 →
  pt.person_b = 8 →
  ws.solo_days = 3 →
  ws.total_days / pt.person_a + (ws.total_days - ws.solo_days) / pt.person_b = 1 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_equation_l3532_353205


namespace NUMINAMATH_CALUDE_line_contains_point_l3532_353223

theorem line_contains_point (m : ℚ) : 
  (2 * m - 3 * (-1) = 5 * 3 + 1) ↔ (m = 13 / 2) := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l3532_353223


namespace NUMINAMATH_CALUDE_odd_power_eight_minus_one_mod_nine_l3532_353247

theorem odd_power_eight_minus_one_mod_nine (n : ℕ) (h : Odd n) : (8^n - 1) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_odd_power_eight_minus_one_mod_nine_l3532_353247


namespace NUMINAMATH_CALUDE_double_burger_cost_l3532_353266

/-- The cost of a double burger given the total spent, number of burgers, single burger cost, and number of double burgers. -/
theorem double_burger_cost 
  (total_spent : ℚ) 
  (total_burgers : ℕ) 
  (single_burger_cost : ℚ) 
  (double_burger_count : ℕ) 
  (h1 : total_spent = 66.5)
  (h2 : total_burgers = 50)
  (h3 : single_burger_cost = 1)
  (h4 : double_burger_count = 33) :
  (total_spent - single_burger_cost * (total_burgers - double_burger_count)) / double_burger_count = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_double_burger_cost_l3532_353266


namespace NUMINAMATH_CALUDE_al_sandwich_options_l3532_353289

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents whether turkey is available. -/
def turkey_available : Prop := True

/-- Represents whether roast beef is available. -/
def roast_beef_available : Prop := True

/-- Represents whether Swiss cheese is available. -/
def swiss_cheese_available : Prop := True

/-- Represents whether rye bread is available. -/
def rye_bread_available : Prop := True

/-- Represents the restriction that Al never orders a sandwich with a turkey/Swiss cheese combination. -/
def no_turkey_swiss : Prop := True

/-- Represents the restriction that Al never orders a sandwich with a rye bread/roast beef combination. -/
def no_rye_roast_beef : Prop := True

/-- The number of different sandwiches Al could order. -/
def num_al_sandwiches : ℕ := num_breads * num_meats * num_cheeses - 5 - 6

theorem al_sandwich_options :
  num_breads = 5 →
  num_meats = 7 →
  num_cheeses = 6 →
  turkey_available →
  roast_beef_available →
  swiss_cheese_available →
  rye_bread_available →
  no_turkey_swiss →
  no_rye_roast_beef →
  num_al_sandwiches = 199 := by
  sorry

#eval num_al_sandwiches -- This should output 199

end NUMINAMATH_CALUDE_al_sandwich_options_l3532_353289


namespace NUMINAMATH_CALUDE_student_event_arrangements_l3532_353279

theorem student_event_arrangements (n m : ℕ) (h1 : n = 7) (h2 : m = 5) : 
  (n.choose m * m.factorial) - ((n - 1).choose m * m.factorial) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_student_event_arrangements_l3532_353279


namespace NUMINAMATH_CALUDE_min_disks_is_twelve_l3532_353287

/-- Represents the number of files of each size --/
structure FileCount where
  large : Nat  -- 0.85 MB files
  medium : Nat -- 0.65 MB files
  small : Nat  -- 0.5 MB files

/-- Represents the constraints of the problem --/
structure DiskProblem where
  totalFiles : Nat
  diskCapacity : Float
  maxFilesPerDisk : Nat
  fileSizes : FileCount
  largeSizeMB : Float
  mediumSizeMB : Float
  smallSizeMB : Float

def problem : DiskProblem := {
  totalFiles := 35,
  diskCapacity := 1.44,
  maxFilesPerDisk := 4,
  fileSizes := { large := 5, medium := 15, small := 15 },
  largeSizeMB := 0.85,
  mediumSizeMB := 0.65,
  smallSizeMB := 0.5
}

/-- Calculates the minimum number of disks required --/
def minDisksRequired (p : DiskProblem) : Nat :=
  sorry -- Proof goes here

theorem min_disks_is_twelve : minDisksRequired problem = 12 := by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_min_disks_is_twelve_l3532_353287
