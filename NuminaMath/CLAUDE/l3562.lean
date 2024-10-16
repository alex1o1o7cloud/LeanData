import Mathlib

namespace NUMINAMATH_CALUDE_daily_pay_rate_is_twenty_l3562_356280

/-- Calculates the daily pay rate given the total days, worked days, forfeit amount, and net earnings -/
def calculate_daily_pay_rate (total_days : ℕ) (worked_days : ℕ) (forfeit_amount : ℚ) (net_earnings : ℚ) : ℚ :=
  let idle_days := total_days - worked_days
  let total_forfeit := idle_days * forfeit_amount
  (net_earnings + total_forfeit) / worked_days

/-- Theorem stating that given the specified conditions, the daily pay rate is $20 -/
theorem daily_pay_rate_is_twenty :
  calculate_daily_pay_rate 25 23 5 450 = 20 := by
  sorry

end NUMINAMATH_CALUDE_daily_pay_rate_is_twenty_l3562_356280


namespace NUMINAMATH_CALUDE_sqrt_3_squared_times_5_to_6_l3562_356210

theorem sqrt_3_squared_times_5_to_6 : Real.sqrt (3^2 * 5^6) = 375 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_squared_times_5_to_6_l3562_356210


namespace NUMINAMATH_CALUDE_function_properties_l3562_356299

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x ≤ f a b (-2)) ∧
    (f a b (-2) = 4) ∧
    (a = 3 ∧ b = 0) ∧
    (∀ x ∈ Set.Icc (-3) 1, f a b x ≤ 4) ∧
    (∃ x ∈ Set.Icc (-3) 1, f a b x = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_function_properties_l3562_356299


namespace NUMINAMATH_CALUDE_distinct_paintings_l3562_356269

/-- The number of disks in the circle -/
def n : ℕ := 12

/-- The number of disks to be painted blue -/
def blue : ℕ := 4

/-- The number of disks to be painted red -/
def red : ℕ := 3

/-- The number of disks to be painted green -/
def green : ℕ := 2

/-- The total number of disks to be painted -/
def painted : ℕ := blue + red + green

/-- The number of rotational symmetries of the circle -/
def symmetries : ℕ := n

/-- The number of ways to color the disks without considering symmetry -/
def total_colorings : ℕ := Nat.choose n blue * Nat.choose (n - blue) red * Nat.choose (n - blue - red) green

/-- The number of distinct paintings considering rotational symmetry -/
theorem distinct_paintings : (total_colorings / symmetries : ℚ) = 23100 := by
  sorry

end NUMINAMATH_CALUDE_distinct_paintings_l3562_356269


namespace NUMINAMATH_CALUDE_city_inhabitants_problem_l3562_356289

theorem city_inhabitants_problem :
  ∃ n : ℕ,
    n > 150 ∧
    (∃ x : ℕ, n = x^2) ∧
    (∃ y : ℕ, n + 1000 = y^2 + 1) ∧
    (∃ z : ℕ, n + 2000 = z^2) ∧
    n = 249001 := by
  sorry

end NUMINAMATH_CALUDE_city_inhabitants_problem_l3562_356289


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l3562_356211

theorem sum_of_coefficients_zero (x y z : ℝ) :
  (λ x y z => (2*x - 3*y + z)^20) 1 1 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l3562_356211


namespace NUMINAMATH_CALUDE_impossible_partition_l3562_356237

theorem impossible_partition : ¬ ∃ (A B C : Finset ℕ),
  (A ∪ B ∪ C = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (Finset.card A = 3) ∧ (Finset.card B = 3) ∧ (Finset.card C = 3) ∧
  (∃ (a₁ a₂ a₃ : ℕ), A = {a₁, a₂, a₃} ∧ max a₁ (max a₂ a₃) = a₁ + a₂ + a₃ - max a₁ (max a₂ a₃)) ∧
  (∃ (b₁ b₂ b₃ : ℕ), B = {b₁, b₂, b₃} ∧ max b₁ (max b₂ b₃) = b₁ + b₂ + b₃ - max b₁ (max b₂ b₃)) ∧
  (∃ (c₁ c₂ c₃ : ℕ), C = {c₁, c₂, c₃} ∧ max c₁ (max c₂ c₃) = c₁ + c₂ + c₃ - max c₁ (max c₂ c₃)) :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_partition_l3562_356237


namespace NUMINAMATH_CALUDE_problem_solution_l3562_356296

theorem problem_solution (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3562_356296


namespace NUMINAMATH_CALUDE_exp_T_equals_eleven_fourths_l3562_356255

/-- The integral T is defined as the definite integral of (2e^(3x) + e^(2x) - 1) / (e^(3x) + e^(2x) - e^x + 1) from 0 to ln(2) -/
noncomputable def T : ℝ := ∫ x in (0)..(Real.log 2), (2 * Real.exp (3 * x) + Real.exp (2 * x) - 1) / (Real.exp (3 * x) + Real.exp (2 * x) - Real.exp x + 1)

/-- The theorem states that e^T equals 11/4 -/
theorem exp_T_equals_eleven_fourths : Real.exp T = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_exp_T_equals_eleven_fourths_l3562_356255


namespace NUMINAMATH_CALUDE_stating_probability_calculation_l3562_356214

/-- The number of people participating in the block selection --/
def num_people : ℕ := 3

/-- The number of colors available for the blocks --/
def num_colors : ℕ := 5

/-- The number of sizes available for the blocks --/
def num_sizes : ℕ := 2

/-- The total number of distinct block types (color-size combinations) --/
def num_block_types : ℕ := num_colors * num_sizes

/-- The number of boxes available for placing the blocks --/
def num_boxes : ℕ := 5

/-- The probability of at least one box receiving 3 blocks of the same color and size --/
def probability_same_color_size : ℚ := 99 / 500

/-- 
Theorem stating that the probability of at least one box receiving 3 blocks 
of the same color and size is equal to 99/500
--/
theorem probability_calculation :
  (probability_same_color_size : ℚ) =
  (1 : ℚ) - (1 - (1 / num_boxes) ^ num_people) ^ (num_colors * num_sizes) :=
sorry

end NUMINAMATH_CALUDE_stating_probability_calculation_l3562_356214


namespace NUMINAMATH_CALUDE_grocers_sales_problem_l3562_356281

/-- Proof of the grocer's sales problem -/
theorem grocers_sales_problem
  (sales : Fin 6 → ℕ)
  (h1 : sales 0 = 6435)
  (h2 : sales 1 = 6927)
  (h3 : sales 2 = 6855)
  (h5 : sales 4 = 6562)
  (h6 : sales 5 = 7991)
  (avg : (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 7000) :
  sales 3 = 7230 := by
  sorry


end NUMINAMATH_CALUDE_grocers_sales_problem_l3562_356281


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_numbers_mod_9_l3562_356256

theorem sum_of_five_consecutive_numbers_mod_9 (n : ℕ) (h : n = 9154) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_numbers_mod_9_l3562_356256


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l3562_356238

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) :
  ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 8 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l3562_356238


namespace NUMINAMATH_CALUDE_initial_mixture_amount_l3562_356204

/-- Represents the problem of finding the initial amount of mixture -/
theorem initial_mixture_amount (initial_mixture : ℝ) : 
  (0.1 * initial_mixture / initial_mixture = 0.1) →  -- Initial mixture is 10% grape juice
  (0.25 * (initial_mixture + 10) = 0.1 * initial_mixture + 10) →  -- Resulting mixture is 25% grape juice
  initial_mixture = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_mixture_amount_l3562_356204


namespace NUMINAMATH_CALUDE_complex_quadrant_l3562_356246

theorem complex_quadrant (a : ℝ) (z : ℂ) : 
  z = a^2 - 3*a - 4 + (a - 4)*Complex.I →
  z.re = 0 →
  (a - a*Complex.I).re < 0 ∧ (a - a*Complex.I).im > 0 :=
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3562_356246


namespace NUMINAMATH_CALUDE_unique_rectangle_existence_l3562_356206

theorem unique_rectangle_existence (a b : ℝ) (h : 0 < a ∧ a < b) :
  ∃! (x y : ℝ), 0 < x ∧ x < y ∧ 2 * (x + y) = a + b ∧ x * y = (a * b) / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_existence_l3562_356206


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l3562_356217

theorem proportion_fourth_term (x y : ℝ) : 
  (0.75 : ℝ) / 1.2 = 5 / y → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l3562_356217


namespace NUMINAMATH_CALUDE_empty_subset_of_intersection_l3562_356283

theorem empty_subset_of_intersection (A B : Set α) 
  (hA : A ≠ ∅) (hB : B ≠ ∅) (hAB : A ≠ B) : 
  ∅ ⊆ A ∩ B :=
sorry

end NUMINAMATH_CALUDE_empty_subset_of_intersection_l3562_356283


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3562_356248

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : x + y = 30) (h3 : x - y = 10) : 
  x = 8 → y = 25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3562_356248


namespace NUMINAMATH_CALUDE_lightbulb_combinations_eq_seven_l3562_356239

/-- The number of ways to turn on at least one out of three lightbulbs -/
def lightbulb_combinations : ℕ :=
  -- Number of ways with one bulb on
  (3 : ℕ).choose 1 +
  -- Number of ways with two bulbs on
  (3 : ℕ).choose 2 +
  -- Number of ways with three bulbs on
  (3 : ℕ).choose 3

/-- Theorem stating that the number of ways to turn on at least one out of three lightbulbs is 7 -/
theorem lightbulb_combinations_eq_seven : lightbulb_combinations = 7 := by
  sorry

end NUMINAMATH_CALUDE_lightbulb_combinations_eq_seven_l3562_356239


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3562_356220

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 3 + a 5 = 21 →
  a 2 * a 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3562_356220


namespace NUMINAMATH_CALUDE_product_of_solutions_l3562_356203

theorem product_of_solutions (x : ℝ) : 
  (x^2 + 6*x - 21 = 0) → 
  (∃ α β : ℝ, (α + β = -6) ∧ (α * β = -21)) := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3562_356203


namespace NUMINAMATH_CALUDE_digit_150_of_22_70_l3562_356293

theorem digit_150_of_22_70 : ∃ (d : ℕ), d = 5 ∧ 
  (∃ (a b : ℕ) (l : List ℕ), 
    (22 : ℚ) / 70 = (a : ℚ) + (b : ℚ) / (10^150) + 
    (l.foldr (λ x acc => acc / 10 + x / (10^150)) 0 : ℚ) ∧
    d = b) := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_22_70_l3562_356293


namespace NUMINAMATH_CALUDE_pirates_escape_strategy_l3562_356223

-- Define the type for colors (0 to 9)
def Color := Fin 10

-- Define the type for the sequence of hat colors
def HatSequence := ℕ → Color

-- Define the type for a pirate's strategy
def Strategy := (ℕ → Color) → Color

-- Define the property of a valid strategy
def ValidStrategy (s : Strategy) (h : HatSequence) : Prop :=
  ∃ n : ℕ, ∀ m : ℕ, m ≥ n → s (fun i => h (i + m + 1)) = h m

-- Theorem statement
theorem pirates_escape_strategy :
  ∃ (s : Strategy), ∀ (h : HatSequence), ValidStrategy s h :=
sorry

end NUMINAMATH_CALUDE_pirates_escape_strategy_l3562_356223


namespace NUMINAMATH_CALUDE_special_sequence_2000th_term_l3562_356275

/-- A sequence where the sum of any three consecutive terms is 20 -/
def SpecialSequence (x : ℕ → ℝ) : Prop :=
  ∀ n, x n + x (n + 1) + x (n + 2) = 20

theorem special_sequence_2000th_term
  (x : ℕ → ℝ)
  (h_special : SpecialSequence x)
  (h_x1 : x 1 = 9)
  (h_x12 : x 12 = 7) :
  x 2000 = 4 := by
sorry

end NUMINAMATH_CALUDE_special_sequence_2000th_term_l3562_356275


namespace NUMINAMATH_CALUDE_base7_digit_sum_l3562_356218

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a base-7 number --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem base7_digit_sum :
  let a := toBase10 45
  let b := toBase10 16
  let c := toBase10 12
  let result := toBase7 ((a * b) + c)
  sumOfDigitsBase7 result = 17 := by sorry

end NUMINAMATH_CALUDE_base7_digit_sum_l3562_356218


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l3562_356252

theorem sum_of_reciprocal_equations (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : 1/x - 1/y = 1) : 
  x + y = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l3562_356252


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3562_356277

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x^2 + 5*x - 24 < 0 ↔ -8 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3562_356277


namespace NUMINAMATH_CALUDE_tan_sum_thirteen_thirtytwo_l3562_356266

theorem tan_sum_thirteen_thirtytwo : 
  let tan13 := Real.tan (13 * π / 180)
  let tan32 := Real.tan (32 * π / 180)
  tan13 + tan32 + tan13 * tan32 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_thirteen_thirtytwo_l3562_356266


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l3562_356262

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where a line passing through the left focus perpendicular to the x-axis 
    intersects the ellipse at points A and B, and the triangle ABF_2 
    (where F_2 is the right focus) is acute, 
    then the eccentricity e of the ellipse is between sqrt(2)-1 and 1. -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt (1 - b^2 / a^2)
  let c := a * e
  let x_A := -c
  let y_A := b^2 / a
  let x_B := -c
  let y_B := -b^2 / a
  let x_F2 := c
  let y_F2 := 0
  (∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → 
    ((x = x_A ∧ y = y_A) ∨ (x = x_B ∧ y = y_B))) →
  (x_A - x_F2)^2 + (y_A - y_F2)^2 < (x_A - x_B)^2 + (y_A - y_B)^2 →
  Real.sqrt 2 - 1 < e ∧ e < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l3562_356262


namespace NUMINAMATH_CALUDE_linear_relationship_values_l3562_356215

/-- Given a linear relationship between x and y, prove the values of y for specific x values -/
theorem linear_relationship_values (x y : ℝ) :
  (y = 3 * x - 1) →
  (x = 1 → y = 2) ∧ (x = 5 → y = 14) := by
  sorry

end NUMINAMATH_CALUDE_linear_relationship_values_l3562_356215


namespace NUMINAMATH_CALUDE_two_faces_same_edges_l3562_356290

/-- A face of a polyhedron -/
structure Face where
  edges : ℕ
  edges_ge_3 : edges ≥ 3

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Set Face
  nonempty : faces.Nonempty

theorem two_faces_same_edges (P : ConvexPolyhedron) : 
  ∃ f₁ f₂ : Face, f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.edges = f₂.edges :=
sorry

end NUMINAMATH_CALUDE_two_faces_same_edges_l3562_356290


namespace NUMINAMATH_CALUDE_beth_crayons_l3562_356259

/-- The number of crayons Beth has altogether -/
def total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) : ℕ :=
  packs * crayons_per_pack + extra_crayons

/-- Theorem stating that Beth has 175 crayons in total -/
theorem beth_crayons : total_crayons 8 20 15 = 175 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l3562_356259


namespace NUMINAMATH_CALUDE_problem_solution_l3562_356244

theorem problem_solution (x y : ℝ) : 
  x = 0.7 * y →
  x = 210 →
  y = 300 ∧ ¬(∃ k : ℤ, y = 7 * k) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3562_356244


namespace NUMINAMATH_CALUDE_ellipse_condition_l3562_356297

theorem ellipse_condition (k a : ℝ) : 
  (∀ x y : ℝ, 3*x^2 + 9*y^2 - 12*x + 27*y = k → 
    ∃ h₁ h₂ c : ℝ, h₁ > 0 ∧ h₂ > 0 ∧ 
    (x - c)^2 / h₁^2 + (y - c)^2 / h₂^2 = 1) ↔ 
  k > a := by
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3562_356297


namespace NUMINAMATH_CALUDE_chord_length_implies_a_value_l3562_356253

theorem chord_length_implies_a_value (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*a*x + a = 0 ∧ a*x + y + 1 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 - 2*a*x₁ + a = 0 ∧ 
    x₂^2 + y₂^2 - 2*a*x₂ + a = 0 ∧
    a*x₁ + y₁ + 1 = 0 ∧ 
    a*x₂ + y₂ + 1 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) →
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_chord_length_implies_a_value_l3562_356253


namespace NUMINAMATH_CALUDE_brownies_remaining_l3562_356245

/-- The number of brownies left after Tina, her husband, and guests eat some. -/
def brownies_left (total : ℕ) (tina_daily : ℕ) (tina_days : ℕ) (husband_daily : ℕ) (husband_days : ℕ) (shared : ℕ) : ℕ :=
  total - (tina_daily * tina_days + husband_daily * husband_days + shared)

/-- Theorem stating that 5 brownies are left under the given conditions. -/
theorem brownies_remaining : brownies_left 24 2 5 1 5 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_brownies_remaining_l3562_356245


namespace NUMINAMATH_CALUDE_sector_area_l3562_356268

/-- Given a circular sector with central angle π/3 and chord length 3 cm,
    the area of the sector is 3π/2 cm². -/
theorem sector_area (θ : Real) (chord_length : Real) (area : Real) :
  θ = π / 3 →
  chord_length = 3 →
  area = 3 * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l3562_356268


namespace NUMINAMATH_CALUDE_solution_set_f_leq_3_min_m_for_inequality_l3562_356207

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for part I
theorem solution_set_f_leq_3 :
  {x : ℝ | f x ≤ 3} = {x : ℝ | 0 ≤ x ∧ x ≤ 3} :=
by sorry

-- Theorem for part II
theorem min_m_for_inequality (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f x ≤ m - x - 4/x) ↔ m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_3_min_m_for_inequality_l3562_356207


namespace NUMINAMATH_CALUDE_total_crayons_is_116_l3562_356288

/-- The total number of crayons Wanda, Dina, and Jacob have -/
def total_crayons (wanda_crayons dina_crayons : ℕ) : ℕ :=
  wanda_crayons + dina_crayons + (dina_crayons - 2)

/-- Theorem stating that the total number of crayons is 116 -/
theorem total_crayons_is_116 :
  total_crayons 62 28 = 116 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_is_116_l3562_356288


namespace NUMINAMATH_CALUDE_factorization_equality_l3562_356242

theorem factorization_equality (x y : ℝ) : x * y^2 + 6 * x * y + 9 * x = x * (y + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3562_356242


namespace NUMINAMATH_CALUDE_defective_switch_probability_l3562_356270

/-- The probability of drawing a defective switch from a population,
    given the total number of switches, sample size, and number of defective switches in the sample. -/
def defective_probability (total : ℕ) (sample_size : ℕ) (defective_in_sample : ℕ) : ℚ :=
  defective_in_sample / sample_size

theorem defective_switch_probability :
  let total := 2000
  let sample_size := 100
  let defective_in_sample := 10
  defective_probability total sample_size defective_in_sample = 1/10 := by
sorry

end NUMINAMATH_CALUDE_defective_switch_probability_l3562_356270


namespace NUMINAMATH_CALUDE_frisbee_sales_receipts_l3562_356208

theorem frisbee_sales_receipts :
  ∀ (x y : ℕ),
  x + y = 64 →
  y ≥ 4 →
  3 * x + 4 * y = 196 :=
by sorry

end NUMINAMATH_CALUDE_frisbee_sales_receipts_l3562_356208


namespace NUMINAMATH_CALUDE_water_amount_in_new_recipe_l3562_356265

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  flour : ℚ
  water : ℚ
  sugar : ℚ

/-- The original recipe ratio -/
def original_ratio : RecipeRatio := ⟨7, 2, 1⟩

/-- The new recipe ratio -/
def new_ratio : RecipeRatio :=
  let flour_water_ratio := original_ratio.flour / original_ratio.water
  let flour_sugar_ratio := original_ratio.flour / original_ratio.sugar
  ⟨original_ratio.flour,
   original_ratio.flour / (2 * flour_water_ratio),
   original_ratio.flour / (flour_sugar_ratio / 2)⟩

/-- The amount of sugar in the new recipe -/
def sugar_amount : ℚ := 4

theorem water_amount_in_new_recipe :
  (sugar_amount * new_ratio.water / new_ratio.sugar) = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_in_new_recipe_l3562_356265


namespace NUMINAMATH_CALUDE_percent_of_y_l3562_356247

theorem percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 5 + (3 * y) / 10) / y * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l3562_356247


namespace NUMINAMATH_CALUDE_polygon_exterior_angle_l3562_356232

theorem polygon_exterior_angle (n : ℕ) (h : n > 2) : 
  (360 : ℝ) / (n : ℝ) = 24 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angle_l3562_356232


namespace NUMINAMATH_CALUDE_solution_greater_than_two_l3562_356216

theorem solution_greater_than_two (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = a ∧ x > 2) ↔ a > 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_greater_than_two_l3562_356216


namespace NUMINAMATH_CALUDE_lucy_has_19_snowballs_l3562_356222

-- Define the number of snowballs Charlie and Lucy have
def charlie_snowballs : ℕ := 50
def lucy_snowballs : ℕ := charlie_snowballs - 31

-- Theorem statement
theorem lucy_has_19_snowballs : lucy_snowballs = 19 := by
  sorry

end NUMINAMATH_CALUDE_lucy_has_19_snowballs_l3562_356222


namespace NUMINAMATH_CALUDE_invalid_votes_count_l3562_356201

/-- Proves that the number of invalid votes is 100 in an election with given conditions -/
theorem invalid_votes_count (total_votes : ℕ) (valid_votes : ℕ) (loser_percentage : ℚ) (vote_difference : ℕ) : 
  total_votes = 12600 →
  loser_percentage = 30/100 →
  vote_difference = 5000 →
  valid_votes = vote_difference / (1/2 - loser_percentage) →
  total_votes - valid_votes = 100 := by
  sorry

#check invalid_votes_count

end NUMINAMATH_CALUDE_invalid_votes_count_l3562_356201


namespace NUMINAMATH_CALUDE_family_siblings_product_l3562_356260

theorem family_siblings_product (total_sisters total_brothers : ℕ) 
  (h1 : total_sisters = 3) 
  (h2 : total_brothers = 5) : 
  ∃ (S B : ℕ), S * B = 10 ∧ S = total_sisters - 1 ∧ B = total_brothers :=
by sorry

end NUMINAMATH_CALUDE_family_siblings_product_l3562_356260


namespace NUMINAMATH_CALUDE_normal_distribution_probabilities_l3562_356273

-- Define a random variable following a normal distribution
def normal_distribution (μ : ℝ) (σ : ℝ) : Type := ℝ

-- Define the cumulative distribution function (CDF) for a normal distribution
noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_probabilities 
  (ξ : normal_distribution 1.5 σ) 
  (h : normal_cdf 1.5 σ 2.5 = 0.78) : 
  normal_cdf 1.5 σ 0.5 = 0.22 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probabilities_l3562_356273


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3562_356228

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ → ℝ × ℝ := λ x ↦ (-2, x)
  ∀ x : ℝ, are_parallel a (b x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3562_356228


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3562_356257

-- Define the function f(x) = x³ - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def interval : Set ℝ := Set.Icc (-3) 0

-- Theorem statement
theorem max_min_f_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ interval, f x ≤ max) ∧ 
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧ 
    (∃ x ∈ interval, f x = min) ∧
    max = 3 ∧ min = -17 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3562_356257


namespace NUMINAMATH_CALUDE_remaining_numbers_are_even_l3562_356221

def last_digit (n : ℕ) : ℕ := n % 10
def second_last_digit (n : ℕ) : ℕ := (n / 10) % 10

def is_removed (n : ℕ) : Prop :=
  (last_digit n % 2 = 1 ∧ second_last_digit n % 2 = 0) ∨
  (last_digit n % 2 = 1 ∧ last_digit n % 3 ≠ 0) ∨
  (second_last_digit n % 2 = 1 ∧ n % 3 = 0)

theorem remaining_numbers_are_even (n : ℕ) :
  ¬(is_removed n) → Even n :=
by sorry

end NUMINAMATH_CALUDE_remaining_numbers_are_even_l3562_356221


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3562_356292

theorem arithmetic_calculation : 8 / 2 - 3 + 2 * (4 - 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3562_356292


namespace NUMINAMATH_CALUDE_car_value_decrease_l3562_356250

/-- Given a car with an initial value and a value after a certain number of years,
    calculate the annual decrease in value. -/
def annual_decrease (initial_value : ℕ) (final_value : ℕ) (years : ℕ) : ℕ :=
  (initial_value - final_value) / years

theorem car_value_decrease :
  let initial_value : ℕ := 20000
  let final_value : ℕ := 14000
  let years : ℕ := 6
  annual_decrease initial_value final_value years = 1000 := by
  sorry

end NUMINAMATH_CALUDE_car_value_decrease_l3562_356250


namespace NUMINAMATH_CALUDE_largest_solution_is_four_l3562_356243

theorem largest_solution_is_four :
  let f : ℝ → ℝ := λ a => (3*a + 4)*(a - 2) - 9*a
  ∃ (a : ℝ), f a = 0 ∧ ∀ (b : ℝ), f b = 0 → b ≤ a ∧ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_is_four_l3562_356243


namespace NUMINAMATH_CALUDE_parabola_intersection_sum_l3562_356235

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

/-- The point R -/
def R : ℝ × ℝ := (10, -6)

/-- The line through R with slope n -/
def line_through_R (n : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 + 6 = n * (p.1 - 10)}

/-- The condition for non-intersection -/
def no_intersection (n : ℝ) : Prop :=
  line_through_R n ∩ P = ∅

theorem parabola_intersection_sum (a b : ℝ) :
  (∀ n, no_intersection n ↔ a < n ∧ n < b) →
  a + b = 40 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_sum_l3562_356235


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_13_l3562_356263

theorem smallest_four_digit_multiple_of_13 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 13 ∣ n → 1001 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_13_l3562_356263


namespace NUMINAMATH_CALUDE_two_same_color_points_at_unit_distance_l3562_356205

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to color points
def colorPoint : Point → Color := sorry

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem two_same_color_points_at_unit_distance :
  ∃ (p1 p2 : Point), colorPoint p1 = colorPoint p2 ∧ distance p1 p2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_two_same_color_points_at_unit_distance_l3562_356205


namespace NUMINAMATH_CALUDE_min_value_expression_l3562_356286

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (6 * a) / (b + 2 * c) + (6 * b) / (c + 2 * a) + (2 * c) / (a + 2 * b) + (6 * c) / (2 * a + b) ≥ 12 ∧
  ((6 * a) / (b + 2 * c) + (6 * b) / (c + 2 * a) + (2 * c) / (a + 2 * b) + (6 * c) / (2 * a + b) = 12 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3562_356286


namespace NUMINAMATH_CALUDE_unique_magnitude_of_complex_roots_l3562_356278

theorem unique_magnitude_of_complex_roots : 
  ∃! r : ℝ, ∃ z : ℂ, z^2 - 4*z + 29 = 0 ∧ Complex.abs z = r :=
sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_complex_roots_l3562_356278


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l3562_356271

/-- Proves that the initial average weight of 6 people in an elevator was 156 lbs,
    given that a 7th person weighing 121 lbs entered and increased the average to 151 lbs. -/
theorem elevator_weight_problem (initial_count : Nat) (new_person_weight : Nat) (new_average : Nat) :
  initial_count = 6 →
  new_person_weight = 121 →
  new_average = 151 →
  ∃ (initial_average : Nat),
    initial_average = 156 ∧
    (initial_count * initial_average + new_person_weight) / (initial_count + 1) = new_average :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l3562_356271


namespace NUMINAMATH_CALUDE_problem_statement_l3562_356279

theorem problem_statement (f : ℝ → ℝ) : 
  (∀ x, f x = (x^4 + 2*x^3 + 4*x - 5)^2004 + 2004) →
  f (Real.sqrt 3 - 1) = 2005 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3562_356279


namespace NUMINAMATH_CALUDE_power_mod_500_l3562_356267

theorem power_mod_500 : 7^(7^(5^2)) ≡ 43 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_mod_500_l3562_356267


namespace NUMINAMATH_CALUDE_octal_567_equals_decimal_375_l3562_356219

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let hundreds := octal / 100
  let tens := (octal / 10) % 10
  let ones := octal % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The octal number 567 is equal to the decimal number 375 --/
theorem octal_567_equals_decimal_375 : octal_to_decimal 567 = 375 := by
  sorry

end NUMINAMATH_CALUDE_octal_567_equals_decimal_375_l3562_356219


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l3562_356230

/-- The equation of a line tangent to an ellipse -/
theorem tangent_line_to_ellipse (x y : ℝ) :
  let P : ℝ × ℝ := (1, Real.sqrt 3 / 2)
  let ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
  let line (x y : ℝ) : Prop := x + 2 * Real.sqrt 3 * y - 4 = 0
  (ellipse P.1 P.2) →  -- Point P is on the ellipse
  (∀ x y, line x y → (x - P.1) * P.1 / 4 + (y - P.2) * P.2 = 0) →  -- Line passes through P
  (∀ x y, ellipse x y → line x y → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y', 
      (x' - x)^2 + (y' - y)^2 < δ^2 → ¬(ellipse x' y' ∧ line x' y')) -- Line is tangent to ellipse
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l3562_356230


namespace NUMINAMATH_CALUDE_exists_valid_grid_l3562_356282

/-- Represents a 6x6 grid of natural numbers -/
def Grid := Fin 6 → Fin 6 → Nat

/-- Checks if a given row contains numbers 1 to 6 without repetition -/
def valid_row (g : Grid) (row : Fin 6) : Prop :=
  ∀ n : Fin 6, ∃! col : Fin 6, g row col = n.val.succ

/-- Checks if a given column contains numbers 1 to 6 without repetition -/
def valid_column (g : Grid) (col : Fin 6) : Prop :=
  ∀ n : Fin 6, ∃! row : Fin 6, g row col = n.val.succ

/-- Checks if a given 2x3 block contains numbers 1 to 6 without repetition -/
def valid_block (g : Grid) (start_row start_col : Fin 3) : Prop :=
  ∀ n : Fin 6, ∃! (row : Fin 2) (col : Fin 3), 
    g (start_row * 2 + row) (start_col * 3 + col) = n.val.succ

/-- Checks if the number between two adjacent cells is their sum or product -/
def valid_between (g : Grid) : Prop :=
  ∀ (row col : Fin 6) (n : Nat),
    (row.val < 5 → n = g row col + g (row.succ) col ∨ n = g row col * g (row.succ) col) ∧
    (col.val < 5 → n = g row col + g row (col.succ) ∨ n = g row col * g row (col.succ))

/-- Main theorem: There exists a valid grid satisfying all conditions -/
theorem exists_valid_grid : ∃ (g : Grid),
  (∀ row : Fin 6, valid_row g row) ∧
  (∀ col : Fin 6, valid_column g col) ∧
  (∀ start_row start_col : Fin 3, valid_block g start_row start_col) ∧
  valid_between g :=
sorry

end NUMINAMATH_CALUDE_exists_valid_grid_l3562_356282


namespace NUMINAMATH_CALUDE_equal_std_dev_and_range_l3562_356272

variable (n : ℕ) (c : ℝ)
variable (x y : Fin n → ℝ)

-- Define the relationship between x and y
def y_def : Prop := ∀ i : Fin n, y i = x i + c

-- Define sample standard deviation
def sample_std_dev (z : Fin n → ℝ) : ℝ := sorry

-- Define sample range
def sample_range (z : Fin n → ℝ) : ℝ := sorry

-- Theorem statement
theorem equal_std_dev_and_range (hc : c ≠ 0) (h_y_def : y_def n c x y) :
  (sample_std_dev n x = sample_std_dev n y) ∧
  (sample_range n x = sample_range n y) := by sorry

end NUMINAMATH_CALUDE_equal_std_dev_and_range_l3562_356272


namespace NUMINAMATH_CALUDE_resettlement_threshold_year_consecutive_equal_proportion_l3562_356236

/-- The area of new housing constructed in the first year (2015) in millions of square meters. -/
def initial_new_housing : ℝ := 5

/-- The area of resettlement housing in the first year (2015) in millions of square meters. -/
def initial_resettlement : ℝ := 2

/-- The annual growth rate of new housing area. -/
def new_housing_growth_rate : ℝ := 0.1

/-- The annual increase in resettlement housing area in millions of square meters. -/
def resettlement_increase : ℝ := 0.5

/-- The cumulative area of resettlement housing after n years. -/
def cumulative_resettlement (n : ℕ) : ℝ :=
  25 * n^2 + 175 * n

/-- The area of new housing in the nth year. -/
def new_housing (n : ℕ) : ℝ :=
  initial_new_housing * (1 + new_housing_growth_rate)^(n - 1)

/-- The area of resettlement housing in the nth year. -/
def resettlement (n : ℕ) : ℝ :=
  initial_resettlement + resettlement_increase * (n - 1)

theorem resettlement_threshold_year :
  ∃ n : ℕ, cumulative_resettlement n ≥ 30 ∧ ∀ m < n, cumulative_resettlement m < 30 :=
sorry

theorem consecutive_equal_proportion :
  ∃ n : ℕ, resettlement n / new_housing n = resettlement (n + 1) / new_housing (n + 1) :=
sorry

end NUMINAMATH_CALUDE_resettlement_threshold_year_consecutive_equal_proportion_l3562_356236


namespace NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l3562_356274

/-- The sum of the present ages of a father and son is 36 years, given that 6 years ago
    the father was 3 times as old as his son, and now the father is only twice as old as his son. -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun son_age father_age =>
    (father_age - 6 = 3 * (son_age - 6)) ∧  -- 6 years ago condition
    (father_age = 2 * son_age) ∧             -- current age condition
    (son_age + father_age = 36)              -- sum of ages

/-- Proof of the theorem -/
theorem father_son_age_sum_proof : ∃ (son_age father_age : ℕ), father_son_age_sum son_age father_age :=
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l3562_356274


namespace NUMINAMATH_CALUDE_M_subset_P_l3562_356251

def M : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4)}
def P : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 2)}

theorem M_subset_P : M ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_M_subset_P_l3562_356251


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l3562_356264

theorem fraction_sum_simplification :
  3 / 840 + 37 / 120 = 131 / 420 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l3562_356264


namespace NUMINAMATH_CALUDE_calculate_savings_person_savings_l3562_356285

/-- Calculates a person's savings given their income sources and expenses --/
theorem calculate_savings (total_income : ℝ) 
  (source_a_percent source_b_percent source_c_percent : ℝ)
  (expense_a_percent expense_b_percent expense_c_percent : ℝ) : ℝ :=
  let source_a := source_a_percent * total_income
  let source_b := source_b_percent * total_income
  let source_c := source_c_percent * total_income
  let expense_a := expense_a_percent * source_a
  let expense_b := expense_b_percent * source_b
  let expense_c := expense_c_percent * source_c
  let total_expenses := expense_a + expense_b + expense_c
  total_income - total_expenses

/-- Proves that the person's savings is Rs. 19,005 given the specified conditions --/
theorem person_savings : 
  calculate_savings 21000 0.5 0.3 0.2 0.1 0.05 0.15 = 19005 := by
  sorry

end NUMINAMATH_CALUDE_calculate_savings_person_savings_l3562_356285


namespace NUMINAMATH_CALUDE_sophia_rental_cost_l3562_356240

/-- Calculates the total cost of car rental given daily rate, per-mile rate, days rented, and miles driven -/
def total_rental_cost (daily_rate : ℚ) (per_mile_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + per_mile_rate * miles

/-- Proves that the total cost for Sophia's car rental is $275 -/
theorem sophia_rental_cost :
  total_rental_cost 30 0.25 5 500 = 275 := by
  sorry

end NUMINAMATH_CALUDE_sophia_rental_cost_l3562_356240


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3562_356202

theorem solve_exponential_equation (y : ℝ) :
  (5 : ℝ) ^ (3 * y) = Real.sqrt 125 → y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3562_356202


namespace NUMINAMATH_CALUDE_equation_solutions_l3562_356231

def equation (x : ℝ) : Prop :=
  x ≠ 4 ∧ x ≠ 6 ∧
  (x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 5) * (x - 4) * (x - 3) /
  ((x - 4) * (x - 6) * (x - 4)) = 1

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3562_356231


namespace NUMINAMATH_CALUDE_tree_branches_after_eight_weeks_l3562_356276

def branch_growth (g : ℕ → ℕ) : Prop :=
  g 2 = 1 ∧
  g 3 = 2 ∧
  (∀ n ≥ 3, g (n + 1) = g n + g (n - 1)) ∧
  g 5 = 5

theorem tree_branches_after_eight_weeks (g : ℕ → ℕ) 
  (h : branch_growth g) : g 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_tree_branches_after_eight_weeks_l3562_356276


namespace NUMINAMATH_CALUDE_train_length_l3562_356294

theorem train_length (pole_time : ℝ) (platform_time : ℝ) (platform_length : ℝ) :
  pole_time = 11 →
  platform_time = 22 →
  platform_length = 120 →
  ∃ (train_length : ℝ) (train_speed : ℝ),
    train_speed = train_length / pole_time ∧
    train_speed = (train_length + platform_length) / platform_time ∧
    train_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3562_356294


namespace NUMINAMATH_CALUDE_absolute_value_of_3_plus_i_l3562_356224

theorem absolute_value_of_3_plus_i :
  let z : ℂ := 3 + Complex.I
  Complex.abs z = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_3_plus_i_l3562_356224


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3562_356229

def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem binomial_expansion_coefficient (k : ℚ) : 
  (binomial_coefficient 5 1) * (-k) = -10 → k = 2 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3562_356229


namespace NUMINAMATH_CALUDE_power_four_squared_cubed_minus_four_l3562_356225

theorem power_four_squared_cubed_minus_four : (4^2)^3 - 4 = 4092 := by
  sorry

end NUMINAMATH_CALUDE_power_four_squared_cubed_minus_four_l3562_356225


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l3562_356261

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpoint_octagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of the midpoint octagon is 1/4 of the original octagon's area -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpoint_octagon o) = (1/4) * area o := by
  sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l3562_356261


namespace NUMINAMATH_CALUDE_quadratic_equation_negative_roots_l3562_356284

theorem quadratic_equation_negative_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
   ∀ x : ℝ, 3 * x^2 + 6 * x + m = 0 ↔ (x = x₁ ∨ x = x₂)) ↔ 
  (m = 1 ∨ m = 2 ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_negative_roots_l3562_356284


namespace NUMINAMATH_CALUDE_perpendicular_probability_l3562_356254

/-- The set of positive integers less than 6 -/
def A : Set ℕ := {n | n < 6 ∧ n > 0}

/-- The line l: x + 2y + 1 = 0 -/
def l (x y : ℝ) : Prop := x + 2*y + 1 = 0

/-- The condition for the line from (a,b) to (0,0) being perpendicular to l -/
def perpendicular (a b : ℕ) : Prop := (b : ℝ) / (a : ℝ) = 2

/-- The number of ways to select 3 different elements from A -/
def total_outcomes : ℕ := Nat.choose 5 3

/-- The number of favorable outcomes -/
def favorable_outcomes : ℕ := 6

/-- The main theorem -/
theorem perpendicular_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_perpendicular_probability_l3562_356254


namespace NUMINAMATH_CALUDE_c2h6_moles_used_l3562_356298

-- Define the chemical species
structure ChemicalSpecies where
  formula : String
  moles : ℝ

-- Define the balanced chemical equation
def balancedEquation (reactant1 reactant2 product1 product2 : ChemicalSpecies) : Prop :=
  reactant1.formula = "C2H6" ∧
  reactant2.formula = "Cl2" ∧
  product1.formula = "C2Cl6" ∧
  product2.formula = "HCl" ∧
  reactant1.moles = 1 ∧
  reactant2.moles = 6 ∧
  product1.moles = 1 ∧
  product2.moles = 6

-- Define the reaction conditions
def reactionConditions (cl2 c2cl6 : ChemicalSpecies) : Prop :=
  cl2.formula = "Cl2" ∧
  cl2.moles = 6 ∧
  c2cl6.formula = "C2Cl6" ∧
  c2cl6.moles = 1

-- Theorem: The number of moles of C2H6 used in the reaction is 1
theorem c2h6_moles_used
  (reactant1 reactant2 product1 product2 cl2 c2cl6 : ChemicalSpecies)
  (h1 : balancedEquation reactant1 reactant2 product1 product2)
  (h2 : reactionConditions cl2 c2cl6) :
  ∃ c2h6 : ChemicalSpecies, c2h6.formula = "C2H6" ∧ c2h6.moles = 1 :=
sorry

end NUMINAMATH_CALUDE_c2h6_moles_used_l3562_356298


namespace NUMINAMATH_CALUDE_cyclist_distance_l3562_356226

/-- Represents a cyclist's journey -/
structure CyclistJourney where
  v : ℝ  -- speed in mph
  t : ℝ  -- time in hours
  d : ℝ  -- distance in miles

/-- Conditions for the cyclist's journey -/
def journeyConditions (j : CyclistJourney) : Prop :=
  j.d = j.v * j.t ∧
  j.d = (j.v + 1) * (3/4 * j.t) ∧
  j.d = (j.v - 1) * (j.t + 3)

theorem cyclist_distance (j : CyclistJourney) 
  (h : journeyConditions j) : j.d = 36 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_distance_l3562_356226


namespace NUMINAMATH_CALUDE_inscribed_polygon_area_l3562_356287

/-- A polygon inscribed around a circle -/
structure InscribedPolygon where
  /-- The radius of the circle -/
  r : ℝ
  /-- The semiperimeter of the polygon -/
  p : ℝ
  /-- The area of the polygon -/
  area : ℝ

/-- Theorem: The area of a polygon inscribed around a circle is equal to the product of its semiperimeter and the radius of the circle -/
theorem inscribed_polygon_area (poly : InscribedPolygon) : poly.area = poly.p * poly.r := by
  sorry

end NUMINAMATH_CALUDE_inscribed_polygon_area_l3562_356287


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3562_356227

/-- The cubic function f(x) = x³ + ax² + bx + c -/
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties :
  ∃ (a b c : ℝ),
    (∀ x, f' a b x = 0 → x = -1 ∨ x = 3) ∧
    (f a b c (-1) = 7) ∧
    (∀ x, f a b c x ≤ 7) ∧
    (∀ x, f a b c x ≥ f a b c 3) ∧
    (a = -3) ∧
    (b = -9) ∧
    (c = 2) ∧
    (f a b c 3 = -25) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3562_356227


namespace NUMINAMATH_CALUDE_cone_section_max_area_l3562_356234

/-- Given a cone whose lateral surface unfolds into a sector with radius 2 and central angle 5π/3,
    the maximum area of any section determined by two generatrices is 2. -/
theorem cone_section_max_area :
  ∀ (r : ℝ) (l : ℝ) (a : ℝ),
  r > 0 →
  l = 2 →
  2 * π * r = 10 * π / 3 →
  0 < a →
  a ≤ 10 / 3 →
  (∃ (h : ℝ), h > 0 ∧ h^2 + (a/2)^2 = l^2 ∧ 
    ∀ (S : ℝ), S = a * h / 2 → S ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_cone_section_max_area_l3562_356234


namespace NUMINAMATH_CALUDE_first_player_wins_l3562_356209

/-- Represents a point on the circle -/
structure Point where
  index : Nat

/-- Represents a triangle drawn on the circle -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- The game state -/
structure GameState where
  n : Nat
  unusedPoints : Finset Point
  drawnTriangles : List Triangle

/-- Predicate to check if a triangle is valid (no crossing edges) -/
def isValidTriangle (t : Triangle) (gs : GameState) : Prop := sorry

/-- Predicate to check if a move is possible given the current game state -/
def canMove (gs : GameState) : Prop := sorry

/-- The winning strategy for the first player -/
def firstPlayerStrategy (gs : GameState) : Option Triangle := sorry

/-- Theorem stating that the first player (Elmo's clone) has a winning strategy -/
theorem first_player_wins (n : Nat) (h : n ≥ 3) :
  ∃ (strategy : GameState → Option Triangle),
    ∀ (gs : GameState),
      gs.n = n →
      (∀ t ∈ gs.drawnTriangles, isValidTriangle t gs) →
      canMove gs →
      ∃ (t : Triangle), 
        strategy gs = some t ∧ 
        isValidTriangle t gs :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l3562_356209


namespace NUMINAMATH_CALUDE_largest_common_term_l3562_356241

def is_common_term (a : ℕ) : Prop :=
  ∃ n m : ℕ, a = 4 + 5 * n ∧ a = 3 + 9 * m

def is_largest_common_term_under_1000 (a : ℕ) : Prop :=
  is_common_term a ∧ 
  a < 1000 ∧
  ∀ b : ℕ, is_common_term b → b < 1000 → b ≤ a

theorem largest_common_term :
  ∃ a : ℕ, is_largest_common_term_under_1000 a ∧ a = 984 :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l3562_356241


namespace NUMINAMATH_CALUDE_pizza_counting_theorem_l3562_356200

/-- The number of available pizza toppings -/
def num_toppings : ℕ := 6

/-- Calculates the number of pizzas with exactly k toppings -/
def pizzas_with_k_toppings (k : ℕ) : ℕ := Nat.choose num_toppings k

/-- The total number of pizzas with one, two, or three toppings -/
def total_pizzas : ℕ := 
  pizzas_with_k_toppings 1 + pizzas_with_k_toppings 2 + pizzas_with_k_toppings 3

theorem pizza_counting_theorem : total_pizzas = 41 := by
  sorry

end NUMINAMATH_CALUDE_pizza_counting_theorem_l3562_356200


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l3562_356213

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + a - 1

-- State the theorem
theorem max_value_implies_a_equals_one :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a x ≤ 1) ∧ (∃ x ∈ Set.Icc 0 1, f a x = 1) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l3562_356213


namespace NUMINAMATH_CALUDE_taehyung_ran_160_meters_l3562_356258

/-- The perimeter of a square -/
def squarePerimeter (side : ℝ) : ℝ := 4 * side

/-- The distance Taehyung ran around the square park -/
def taehyungDistance : ℝ := squarePerimeter 40

theorem taehyung_ran_160_meters : taehyungDistance = 160 := by
  sorry

end NUMINAMATH_CALUDE_taehyung_ran_160_meters_l3562_356258


namespace NUMINAMATH_CALUDE_cubic_factorization_l3562_356295

theorem cubic_factorization (a : ℝ) : a^3 - 2*a^2 + a = a*(a-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3562_356295


namespace NUMINAMATH_CALUDE_product_mod_seven_l3562_356291

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l3562_356291


namespace NUMINAMATH_CALUDE_sprite_volume_calculation_l3562_356233

def maaza_volume : ℕ := 50
def pepsi_volume : ℕ := 144
def total_cans : ℕ := 281

def can_volume : ℕ := Nat.gcd maaza_volume pepsi_volume

def maaza_cans : ℕ := maaza_volume / can_volume
def pepsi_cans : ℕ := pepsi_volume / can_volume

def sprite_cans : ℕ := total_cans - (maaza_cans + pepsi_cans)

def sprite_volume : ℕ := sprite_cans * can_volume

theorem sprite_volume_calculation :
  sprite_volume = 368 :=
by sorry

end NUMINAMATH_CALUDE_sprite_volume_calculation_l3562_356233


namespace NUMINAMATH_CALUDE_cubic_roots_roots_product_l3562_356249

/-- Given a cubic equation x^3 - 7x^2 + 36 = 0 where the product of two of its roots is 18,
    prove that the roots are -2, 3, and 6. -/
theorem cubic_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 7*x^2 + 36 = 0 ∧ 
   r₁ * r₂ = 18 ∧
   (x = r₁ ∨ x = r₂ ∨ x = r₃)) →
  (x = -2 ∨ x = 3 ∨ x = 6) :=
by sorry

/-- The product of all three roots of the cubic equation x^3 - 7x^2 + 36 = 0 is -36. -/
theorem roots_product (r₁ r₂ r₃ : ℝ) :
  r₁^3 - 7*r₁^2 + 36 = 0 ∧ 
  r₂^3 - 7*r₂^2 + 36 = 0 ∧ 
  r₃^3 - 7*r₃^2 + 36 = 0 →
  r₁ * r₂ * r₃ = -36 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_roots_product_l3562_356249


namespace NUMINAMATH_CALUDE_remaining_balloons_l3562_356212

def initial_balloons : ℕ := 30
def balloons_given : ℕ := 16

theorem remaining_balloons : initial_balloons - balloons_given = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balloons_l3562_356212
