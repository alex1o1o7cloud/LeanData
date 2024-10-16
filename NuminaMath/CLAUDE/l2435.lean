import Mathlib

namespace NUMINAMATH_CALUDE_max_value_and_min_side_l2435_243508

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sin x - m * Real.cos x

theorem max_value_and_min_side (m : ℝ) (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f m x ≤ f m (π/3)) →  -- f achieves maximum at π/3
  f m (A - π/2) = 0 →         -- condition on angle A
  2 * b + c = 3 →             -- condition on sides b and c
  0 < A ∧ A < π →             -- A is a valid angle
  0 < B ∧ B < π →             -- B is a valid angle
  0 < C ∧ C < π →             -- C is a valid angle
  a > 0 ∧ b > 0 ∧ c > 0 →     -- sides are positive
  A + B + C = π →             -- sum of angles in a triangle
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- cosine rule
  m = -Real.sqrt 3 / 3 ∧ a ≥ 3 * Real.sqrt 21 / 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_min_side_l2435_243508


namespace NUMINAMATH_CALUDE_A_minus_B_equals_1790_l2435_243586

/-- Calculates the value of A based on the given groups -/
def calculate_A : ℕ := 1 * 1000 + 16 * 100 + 28 * 10

/-- Calculates the value of B based on the given jumps and interval -/
def calculate_B : ℕ := 355 + 3 * 245

/-- Proves that A - B equals 1790 -/
theorem A_minus_B_equals_1790 : calculate_A - calculate_B = 1790 := by
  sorry

end NUMINAMATH_CALUDE_A_minus_B_equals_1790_l2435_243586


namespace NUMINAMATH_CALUDE_vydmans_formula_l2435_243551

theorem vydmans_formula (h b x r : ℝ) (h_pos : h > 0) (b_pos : b > 0) (x_pos : x > 0) :
  r = Real.sqrt ((b / 2)^2 + ((h^2 + (b / 2 - x)^2 - b^2 / 4) / (2 * h))^2) →
  r^2 = (b / 2)^2 + ((h^2 + (b / 2 - x)^2 - b^2 / 4) / (2 * h))^2 :=
by sorry

end NUMINAMATH_CALUDE_vydmans_formula_l2435_243551


namespace NUMINAMATH_CALUDE_magazines_to_boxes_l2435_243596

theorem magazines_to_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_magazines_to_boxes_l2435_243596


namespace NUMINAMATH_CALUDE_max_n_minus_sum_digits_l2435_243553

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the maximum value of n satisfying n - S_n = 2007 -/
theorem max_n_minus_sum_digits (n : ℕ) :
  n - sum_of_digits n = 2007 → n ≤ 2019 :=
by sorry

end NUMINAMATH_CALUDE_max_n_minus_sum_digits_l2435_243553


namespace NUMINAMATH_CALUDE_bobs_small_gate_width_l2435_243593

/-- Represents a rectangular garden with gates and fencing -/
structure Garden where
  length : ℝ
  width : ℝ
  large_gate_width : ℝ
  total_fencing : ℝ

/-- Calculates the perimeter of a rectangle -/
def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

/-- Calculates the width of the small gate -/
def small_gate_width (g : Garden) : ℝ :=
  g.total_fencing - rectangle_perimeter g.length g.width + g.large_gate_width

/-- Theorem stating the width of the small gate in Bob's garden -/
theorem bobs_small_gate_width :
  let g : Garden := {
    length := 225,
    width := 125,
    large_gate_width := 10,
    total_fencing := 687
  }
  small_gate_width g = 3 := by
  sorry


end NUMINAMATH_CALUDE_bobs_small_gate_width_l2435_243593


namespace NUMINAMATH_CALUDE_square_product_exists_l2435_243528

theorem square_product_exists (A : Finset ℕ+) (h1 : A.card = 2016) 
  (h2 : ∀ x ∈ A, ∀ p : ℕ, Nat.Prime p → p ∣ x.val → p < 30) : 
  ∃ a b c d : ℕ+, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  ∃ m : ℕ, (a.val * b.val * c.val * d.val : ℕ) = m ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_square_product_exists_l2435_243528


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_of_numbers_l2435_243588

def numbers : List Nat := [16, 32, 48]

theorem gcf_lcm_sum_of_numbers (A B : Nat) 
  (h1 : A = Nat.gcd 16 (Nat.gcd 32 48))
  (h2 : B = Nat.lcm 16 (Nat.lcm 32 48)) : 
  A + B = 112 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_of_numbers_l2435_243588


namespace NUMINAMATH_CALUDE_range_of_a_l2435_243536

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ (3/2)^x = (2+3*a)/(5-a)) → 
  a ∈ Set.Ioo (-2/3 : ℝ) (3/4 : ℝ) := by
sorry


end NUMINAMATH_CALUDE_range_of_a_l2435_243536


namespace NUMINAMATH_CALUDE_dance_team_recruitment_l2435_243507

theorem dance_team_recruitment (total : ℕ) (track : ℕ) (choir : ℕ) (dance : ℕ) : 
  total = 100 ∧ 
  choir = 2 * track ∧ 
  dance = choir + 10 ∧ 
  total = track + choir + dance → 
  dance = 46 := by
sorry

end NUMINAMATH_CALUDE_dance_team_recruitment_l2435_243507


namespace NUMINAMATH_CALUDE_positive_difference_of_solutions_l2435_243590

theorem positive_difference_of_solutions : ∃ (x₁ x₂ : ℝ), 
  (|2 * x₁ - 3| = 15 ∧ |2 * x₂ - 3| = 15) ∧ |x₁ - x₂| = 15 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_of_solutions_l2435_243590


namespace NUMINAMATH_CALUDE_g_range_values_l2435_243563

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - 3*x) / (2 + 3*x))

theorem g_range_values :
  {y | ∃ x, g x = y} = {-π/2, π/4} := by sorry

end NUMINAMATH_CALUDE_g_range_values_l2435_243563


namespace NUMINAMATH_CALUDE_transformed_line_equation_l2435_243539

-- Define the original line
def original_line (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the stretch transformation
def stretch_transform (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 2 * y

-- Theorem: The equation of line l after transformation is x - y - 2 = 0
theorem transformed_line_equation (x' y' : ℝ) :
  (∃ x y, original_line x y ∧ stretch_transform x y x' y') →
  x' - y' - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_transformed_line_equation_l2435_243539


namespace NUMINAMATH_CALUDE_tile_difference_l2435_243583

theorem tile_difference (initial_blue : ℕ) (initial_green : ℕ) (border_tiles : ℕ) :
  initial_blue = 15 →
  initial_green = 8 →
  border_tiles = 12 →
  (initial_blue + border_tiles / 2) - (initial_green + border_tiles / 2) = 7 :=
by sorry

end NUMINAMATH_CALUDE_tile_difference_l2435_243583


namespace NUMINAMATH_CALUDE_sin_cos_equation_solutions_l2435_243516

theorem sin_cos_equation_solutions (x : Real) :
  (Real.sin x + Real.cos x + Real.sin x * Real.cos x = 1) ↔
  (∃ k : ℤ, x = k * (2 * Real.pi) ∨ x = Real.pi / 2 + k * (2 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solutions_l2435_243516


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2435_243549

theorem fraction_sum_equality (a b c : ℝ) : 
  (a / (30 - a) + b / (70 - b) + c / (55 - c) = 9) →
  (6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 5.08) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2435_243549


namespace NUMINAMATH_CALUDE_max_term_at_k_max_l2435_243568

/-- The value of k that maximizes the term C_{209}^k (√5)^k in the expansion of (1+√5)^209 -/
def k_max : ℕ := 145

/-- The binomial coefficient C(n,k) -/
def binomial_coeff (n k : ℕ) : ℕ := sorry

/-- The term C_{209}^k (√5)^k in the expansion of (1+√5)^209 -/
noncomputable def term (k : ℕ) : ℝ := (binomial_coeff 209 k) * (Real.sqrt 5) ^ k

theorem max_term_at_k_max :
  ∀ k : ℕ, k ≠ k_max → term k ≤ term k_max :=
sorry

end NUMINAMATH_CALUDE_max_term_at_k_max_l2435_243568


namespace NUMINAMATH_CALUDE_only_optionA_is_valid_l2435_243562

-- Define a type for programming statements
inductive ProgramStatement
  | Print (expr : String)
  | Input
  | InputAssign (var : String) (value : Nat)
  | PrintAssign (var : String) (expr : String)

-- Define a function to check if a statement is valid
def isValidStatement (stmt : ProgramStatement) : Prop :=
  match stmt with
  | ProgramStatement.Print expr => True
  | ProgramStatement.Input => False
  | ProgramStatement.InputAssign _ _ => False
  | ProgramStatement.PrintAssign _ _ => False

-- Define the given options
def optionA := ProgramStatement.Print "4*x"
def optionB := ProgramStatement.Input
def optionC := ProgramStatement.InputAssign "B" 3
def optionD := ProgramStatement.PrintAssign "y" "2*x+1"

-- Theorem to prove
theorem only_optionA_is_valid :
  isValidStatement optionA ∧
  ¬isValidStatement optionB ∧
  ¬isValidStatement optionC ∧
  ¬isValidStatement optionD :=
sorry

end NUMINAMATH_CALUDE_only_optionA_is_valid_l2435_243562


namespace NUMINAMATH_CALUDE_no_solution_when_m_is_seven_l2435_243582

theorem no_solution_when_m_is_seven :
  ∀ x : ℝ, x ≠ 4 → x ≠ 8 → (x - 3) / (x - 4) ≠ (x - 7) / (x - 8) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_when_m_is_seven_l2435_243582


namespace NUMINAMATH_CALUDE_smallest_nonprime_no_small_factors_is_529_l2435_243574

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p, is_prime p → p < k → ¬(n % p = 0)

def smallest_nonprime_no_small_factors : ℕ → Prop
| n => ¬(is_prime n) ∧ 
       n > 1 ∧ 
       has_no_prime_factors_less_than n 20 ∧
       ∀ m, 1 < m → m < n → ¬(¬(is_prime m) ∧ has_no_prime_factors_less_than m 20)

theorem smallest_nonprime_no_small_factors_is_529 :
  ∃ n, smallest_nonprime_no_small_factors n ∧ n = 529 :=
sorry

end NUMINAMATH_CALUDE_smallest_nonprime_no_small_factors_is_529_l2435_243574


namespace NUMINAMATH_CALUDE_base8_plus_15_l2435_243560

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (x : ℕ) : ℕ :=
  (x / 100) * 64 + ((x / 10) % 10) * 8 + (x % 10)

/-- The problem statement --/
theorem base8_plus_15 : base8_to_base10 123 + 15 = 98 := by
  sorry

end NUMINAMATH_CALUDE_base8_plus_15_l2435_243560


namespace NUMINAMATH_CALUDE_prob_six_largest_is_two_sevenths_l2435_243518

/-- A function that calculates the probability of selecting 6 as the largest value
    when drawing 4 cards from a set of 7 cards numbered 1 to 7 without replacement -/
def prob_six_largest (n : ℕ) (k : ℕ) : ℚ :=
  if n = 7 ∧ k = 4 then 2/7 else 0

/-- Theorem stating that the probability of selecting 6 as the largest value
    when drawing 4 cards from a set of 7 cards numbered 1 to 7 without replacement is 2/7 -/
theorem prob_six_largest_is_two_sevenths :
  prob_six_largest 7 4 = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_six_largest_is_two_sevenths_l2435_243518


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2435_243594

theorem polynomial_division_theorem :
  let f (x : ℝ) := x^4 - 8*x^3 + 18*x^2 - 22*x + 8
  let g (x : ℝ) := x^2 - 3*x + k
  let r (x : ℝ) := x + a
  ∀ (k a : ℝ),
  (∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x) →
  (k = 8/3 ∧ a = 64/9) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2435_243594


namespace NUMINAMATH_CALUDE_threeDigitNumberFormula_l2435_243550

/-- Given a natural number m, this function represents a three-digit number
    where the hundreds digit is 3m, the tens digit is m, and the units digit is m-1 -/
def threeDigitNumber (m : ℕ) : ℕ := 300 * m + 10 * m + (m - 1)

/-- Theorem stating that the three-digit number can be expressed as 311m - 1 -/
theorem threeDigitNumberFormula (m : ℕ) : 
  threeDigitNumber m = 311 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_threeDigitNumberFormula_l2435_243550


namespace NUMINAMATH_CALUDE_men_per_table_l2435_243556

theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ)
  (h1 : num_tables = 7)
  (h2 : women_per_table = 7)
  (h3 : total_customers = 63) :
  (total_customers - num_tables * women_per_table) / num_tables = 2 :=
by sorry

end NUMINAMATH_CALUDE_men_per_table_l2435_243556


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l2435_243500

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (c d : ℤ) 
  (hc : ∃ m : ℤ, c = 6 * m) (hd : ∃ n : ℤ, d = 9 * n) : 
  ∃ k : ℤ, c + d = 3 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l2435_243500


namespace NUMINAMATH_CALUDE_base6_265_equals_base10_113_l2435_243584

/-- Converts a base-6 number to base 10 --/
def base6ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 6^2 + tens * 6^1 + ones * 6^0

/-- Theorem: The base-6 number 265₆ is equal to 113 in base 10 --/
theorem base6_265_equals_base10_113 : base6ToBase10 2 6 5 = 113 := by
  sorry

end NUMINAMATH_CALUDE_base6_265_equals_base10_113_l2435_243584


namespace NUMINAMATH_CALUDE_factors_of_48_l2435_243517

def number_of_factors (n : Nat) : Nat :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_48 : number_of_factors 48 = 10 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_48_l2435_243517


namespace NUMINAMATH_CALUDE_church_attendance_l2435_243524

theorem church_attendance (male_adults female_adults total_people : ℕ) 
  (h1 : male_adults = 60)
  (h2 : female_adults = 60)
  (h3 : total_people = 200) :
  total_people - (male_adults + female_adults) = 80 := by
sorry

end NUMINAMATH_CALUDE_church_attendance_l2435_243524


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2435_243501

def i : ℂ := Complex.I

theorem imaginary_part_of_complex_fraction :
  ((-1 + i) / (2 - i)).im = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2435_243501


namespace NUMINAMATH_CALUDE_quadratic_properties_l2435_243542

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 2)^2 + 6

-- State the theorem
theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x > f y → f y > f ((y - x) + y)) ∧ 
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧
  (f 0 = 10) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2435_243542


namespace NUMINAMATH_CALUDE_division_calculation_l2435_243544

theorem division_calculation : (-1/30) / (2/3 - 1/10 + 1/6 - 2/5) = -1/10 := by
  sorry

end NUMINAMATH_CALUDE_division_calculation_l2435_243544


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l2435_243511

/-- Given two integers u and v, where 0 < v < u, and points A, B, C, D, E defined as:
    A = (u, v)
    B is the reflection of A across y = x
    C is the reflection of B across the y-axis
    D is the reflection of C across the x-axis
    E is the reflection of D across the y-axis
    If the area of pentagon ABCDE is 500, then u + v = 19. -/
theorem pentagon_area_sum (u v : ℤ) (hu : 0 < v) (hv : v < u) : 
  let A := (u, v)
  let B := (v, u)
  let C := (-v, u)
  let D := (-v, -u)
  let E := (v, -u)
  let area := (2 * u * 2 * v) + (2 * u * v)
  area = 500 → u + v = 19 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l2435_243511


namespace NUMINAMATH_CALUDE_sugar_recipes_l2435_243581

/-- The number of full recipes that can be made with a given amount of sugar -/
def full_recipes (total_sugar : ℚ) (sugar_per_recipe : ℚ) : ℚ :=
  total_sugar / sugar_per_recipe

/-- Theorem: Given 47 2/3 cups of sugar and a recipe requiring 1 1/2 cups of sugar,
    the number of full recipes that can be made is 31 7/9 -/
theorem sugar_recipes :
  let total_sugar : ℚ := 47 + 2/3
  let sugar_per_recipe : ℚ := 1 + 1/2
  full_recipes total_sugar sugar_per_recipe = 31 + 7/9 := by
sorry

end NUMINAMATH_CALUDE_sugar_recipes_l2435_243581


namespace NUMINAMATH_CALUDE_root_product_theorem_l2435_243537

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 5 = 0) → 
  (b^2 - m*b + 5 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 36/5 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l2435_243537


namespace NUMINAMATH_CALUDE_k_range_theorem_l2435_243585

/-- Proposition p: The equation represents an ellipse with foci on the y-axis -/
def p (k : ℝ) : Prop := 3 < k ∧ k < 9/2

/-- Proposition q: The equation represents a hyperbola with eccentricity e in (√3, 2) -/
def q (k : ℝ) : Prop := 4 < k ∧ k < 6

/-- The range of real values for k -/
def k_range (k : ℝ) : Prop := (3 < k ∧ k ≤ 4) ∨ (9/2 ≤ k ∧ k < 6)

theorem k_range_theorem (k : ℝ) : 
  (¬(p k ∧ q k) ∧ (p k ∨ q k)) → k_range k := by
  sorry

end NUMINAMATH_CALUDE_k_range_theorem_l2435_243585


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2435_243530

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 ∧ 
  k = 0 ∧ 
  c = Real.sqrt 50 ∧ 
  a = 5 ∧ 
  c^2 = a^2 + b^2 →
  h + k + a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2435_243530


namespace NUMINAMATH_CALUDE_jack_socks_problem_l2435_243564

theorem jack_socks_problem :
  ∀ (x y z : ℕ),
    x + y + z = 15 →
    2 * x + 4 * y + 5 * z = 36 →
    x ≥ 1 →
    y ≥ 1 →
    z ≥ 1 →
    x = 4 :=
by sorry

end NUMINAMATH_CALUDE_jack_socks_problem_l2435_243564


namespace NUMINAMATH_CALUDE_intersection_segment_length_l2435_243526

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (3, 0)

-- Define the line perpendicular to x-axis passing through the right focus
def perpendicular_line (x y : ℝ) : Prop := x = 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ perpendicular_line p.1 p.2}

-- Statement to prove
theorem intersection_segment_length :
  let A := (3, 16/5)
  let B := (3, -16/5)
  (A ∈ intersection_points) ∧ 
  (B ∈ intersection_points) ∧
  (∀ p ∈ intersection_points, p = A ∨ p = B) ∧
  (dist A B = 32/5) := by sorry


end NUMINAMATH_CALUDE_intersection_segment_length_l2435_243526


namespace NUMINAMATH_CALUDE_brooke_homework_time_l2435_243567

/-- Calculates the total time Brooke spends on homework, including breaks -/
def total_homework_time (math_problems : ℕ) (social_studies_problems : ℕ) (science_problems : ℕ)
  (math_time_per_problem : ℚ) (social_studies_time_per_problem : ℚ) (science_time_per_problem : ℚ)
  (math_break : ℕ) (social_studies_break : ℕ) (science_break : ℕ) : ℚ :=
  let math_time := math_problems * math_time_per_problem
  let social_studies_time := social_studies_problems * social_studies_time_per_problem / 60
  let science_time := science_problems * science_time_per_problem
  let total_problem_time := math_time + social_studies_time + science_time
  let total_break_time := math_break + social_studies_break + science_break
  total_problem_time + total_break_time

theorem brooke_homework_time :
  total_homework_time 15 6 10 2 (1/2) (3/2) 5 10 15 = 78 := by
  sorry

end NUMINAMATH_CALUDE_brooke_homework_time_l2435_243567


namespace NUMINAMATH_CALUDE_set_intersection_example_l2435_243579

theorem set_intersection_example : 
  ({3, 5, 6, 8} : Set ℕ) ∩ ({4, 5, 8} : Set ℕ) = {5, 8} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l2435_243579


namespace NUMINAMATH_CALUDE_prime_remainder_mod_30_l2435_243531

theorem prime_remainder_mod_30 (p : ℕ) (hp : Prime p) : 
  ∃ (r : ℕ), p % 30 = r ∧ (r = 1 ∨ (Prime r ∧ r < 30)) := by
  sorry

end NUMINAMATH_CALUDE_prime_remainder_mod_30_l2435_243531


namespace NUMINAMATH_CALUDE_stuffed_dogs_count_l2435_243532

/-- The number of boxes of stuffed toy dogs -/
def num_boxes : ℕ := 7

/-- The number of dogs in each box -/
def dogs_per_box : ℕ := 4

/-- The total number of dogs -/
def total_dogs : ℕ := num_boxes * dogs_per_box

theorem stuffed_dogs_count : total_dogs = 28 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_dogs_count_l2435_243532


namespace NUMINAMATH_CALUDE_equal_sum_product_quotient_l2435_243548

theorem equal_sum_product_quotient :
  ∃! (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a + b = a * b ∧ a + b = a / b ∧ a = 1/2 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_product_quotient_l2435_243548


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_m_geq_four_l2435_243555

theorem sufficient_condition_implies_m_geq_four (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x < 4 → x < m) → m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_m_geq_four_l2435_243555


namespace NUMINAMATH_CALUDE_inequality_proof_l2435_243515

theorem inequality_proof (a b c r : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 1) : 
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≤ 
  (a^r / (b^r + c^r)) + (b^r / (c^r + a^r)) + (c^r / (a^r + b^r)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2435_243515


namespace NUMINAMATH_CALUDE_work_completion_time_l2435_243512

-- Define the work rates for p, q, and r
def P : ℚ := 9 / 140
def Q : ℚ := 1 / 28
def R : ℚ := 1 / 35

-- Theorem statement
theorem work_completion_time :
  -- Condition 1: p can do the work in the same time as q and r together
  P = Q + R →
  -- Condition 2: p and q together can complete the work in 10 days
  P + Q = 1 / 10 →
  -- Condition 3: q alone can do the work in approximately 28 days
  Q = 1 / 28 →
  -- Conclusion: r needs 35 days to complete the work alone
  1 / R = 35 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2435_243512


namespace NUMINAMATH_CALUDE_light_bulb_state_l2435_243578

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def toggle_light (n : ℕ) (i : ℕ) : Bool := i % n = 0

def final_state (n : ℕ) : Bool :=
  (List.range n).foldl (fun acc i => acc ≠ toggle_light (i + 1) n) false

theorem light_bulb_state (n : ℕ) (hn : n ≤ 100) :
  final_state n = true ↔ is_perfect_square n :=
sorry

end NUMINAMATH_CALUDE_light_bulb_state_l2435_243578


namespace NUMINAMATH_CALUDE_breads_after_five_thieves_l2435_243576

/-- The number of breads remaining after a thief takes their share. -/
def remaining_breads (initial : ℕ) (thief : ℕ) : ℚ :=
  if thief = 0 then initial
  else (remaining_breads initial (thief - 1) / 2) - 1/2

/-- The theorem stating that after 5 thieves, 3 breads remain from an initial count of 127. -/
theorem breads_after_five_thieves :
  ⌊remaining_breads 127 5⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_breads_after_five_thieves_l2435_243576


namespace NUMINAMATH_CALUDE_quadratic_roots_sums_l2435_243513

theorem quadratic_roots_sums (p q x₁ x₂ : ℝ) 
  (hq : q ≠ 0)
  (hroots : x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) :
  (1/x₁ + 1/x₂ = -p/q) ∧
  (1/x₁^2 + 1/x₂^2 = (p^2 - 2*q)/q^2) ∧
  (1/x₁^3 + 1/x₂^3 = (p/q^3)*(3*q - p^2)) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_sums_l2435_243513


namespace NUMINAMATH_CALUDE_solution_replacement_fraction_l2435_243514

theorem solution_replacement_fraction (V : ℝ) (x : ℝ) 
  (h1 : V > 0)
  (h2 : 0 ≤ x ∧ x ≤ 1)
  (h3 : (0.80 * V - 0.80 * x * V) + 0.25 * x * V = 0.35 * V) :
  x = 9 / 11 := by
sorry

end NUMINAMATH_CALUDE_solution_replacement_fraction_l2435_243514


namespace NUMINAMATH_CALUDE_correct_age_ranking_l2435_243599

-- Define the set of friends
inductive Friend : Type
| David : Friend
| Emma : Friend
| Fiona : Friend
| George : Friend

-- Define the age relation
def OlderThan : Friend → Friend → Prop := sorry

-- Define the statements
def Statement1 : Prop := ∀ f : Friend, f ≠ Friend.Emma → OlderThan Friend.Emma f
def Statement2 : Prop := ∃ f : Friend, OlderThan f Friend.Fiona
def Statement3 : Prop := ∃ f : Friend, OlderThan Friend.David f
def Statement4 : Prop := ∃ f : Friend, OlderThan f Friend.George

-- Define the theorem
theorem correct_age_ranking :
  (∀ f1 f2 : Friend, f1 ≠ f2 → (OlderThan f1 f2 ∨ OlderThan f2 f1)) →
  (Statement1 ∨ Statement2 ∨ Statement3 ∨ Statement4) →
  (¬Statement1 ∨ ¬Statement2 ∨ ¬Statement3 ∨ ¬Statement4) →
  (OlderThan Friend.Fiona Friend.Emma ∧
   OlderThan Friend.Emma Friend.George ∧
   OlderThan Friend.George Friend.David) :=
by sorry

end NUMINAMATH_CALUDE_correct_age_ranking_l2435_243599


namespace NUMINAMATH_CALUDE_stating_investment_plans_count_l2435_243521

/-- Represents the number of cities available for investment --/
def num_cities : ℕ := 4

/-- Represents the number of projects to be distributed --/
def num_projects : ℕ := 3

/-- Represents the maximum number of projects allowed in a single city --/
def max_projects_per_city : ℕ := 2

/-- 
Calculates the number of ways to distribute distinct projects among cities,
with a limit on the number of projects per city.
--/
def investment_plans (cities : ℕ) (projects : ℕ) (max_per_city : ℕ) : ℕ := 
  sorry

/-- 
Theorem stating that the number of investment plans 
for the given conditions is 60.
--/
theorem investment_plans_count : 
  investment_plans num_cities num_projects max_projects_per_city = 60 := by
  sorry

end NUMINAMATH_CALUDE_stating_investment_plans_count_l2435_243521


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l2435_243558

def arrange_books (geometry_copies : ℕ) (algebra_copies : ℕ) : ℕ :=
  Nat.choose (geometry_copies + algebra_copies - 2) (algebra_copies - 2)

theorem book_arrangement_proof : 
  arrange_books 4 5 = 35 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l2435_243558


namespace NUMINAMATH_CALUDE_matches_played_calculation_l2435_243510

/-- A football competition with a specific scoring system and number of matches --/
structure FootballCompetition where
  totalMatches : ℕ
  pointsForWin : ℕ
  pointsForDraw : ℕ
  pointsForLoss : ℕ

/-- A team's current state in the competition --/
structure TeamState where
  pointsScored : ℕ
  matchesPlayed : ℕ

/-- Theorem stating the number of matches played by the team --/
theorem matches_played_calculation (comp : FootballCompetition)
    (state : TeamState) (minWinsNeeded : ℕ) (targetPoints : ℕ) :
    comp.totalMatches = 20 ∧
    comp.pointsForWin = 3 ∧
    comp.pointsForDraw = 1 ∧
    comp.pointsForLoss = 0 ∧
    state.pointsScored = 14 ∧
    minWinsNeeded = 6 ∧
    targetPoints = 40 →
    state.matchesPlayed = 14 := by
  sorry

end NUMINAMATH_CALUDE_matches_played_calculation_l2435_243510


namespace NUMINAMATH_CALUDE_overlapping_sectors_area_l2435_243570

/-- The area of the shaded region formed by the overlap of two 30° sectors in a circle with radius 10 is equal to the area of a single 30° sector. -/
theorem overlapping_sectors_area (r : ℝ) (angle : ℝ) : 
  r = 10 → angle = 30 * (π / 180) → 
  let sector_area := (angle / (2 * π)) * π * r^2
  let shaded_area := sector_area
  ∀ ε > 0, |shaded_area - sector_area| < ε :=
sorry

end NUMINAMATH_CALUDE_overlapping_sectors_area_l2435_243570


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l2435_243533

theorem stratified_sampling_size (total_employees : ℕ) (male_employees : ℕ) (female_sample : ℕ) (sample_size : ℕ) : 
  total_employees = 120 →
  male_employees = 90 →
  female_sample = 9 →
  (total_employees - male_employees) / total_employees = female_sample / sample_size →
  sample_size = 36 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l2435_243533


namespace NUMINAMATH_CALUDE_first_valid_year_is_1980_l2435_243589

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 1950 ∧ sum_of_digits year = 18

theorem first_valid_year_is_1980 :
  (∀ y : ℕ, y < 1980 → ¬(is_valid_year y)) ∧ is_valid_year 1980 := by
  sorry

end NUMINAMATH_CALUDE_first_valid_year_is_1980_l2435_243589


namespace NUMINAMATH_CALUDE_horner_v₂_for_specific_polynomial_v₂_value_at_10_l2435_243573

/-- Horner's Rule for a polynomial of degree 4 -/
def horner_rule (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₄ * x + a₃) * x + a₂) * x + a₁ * x + a₀

/-- The second intermediate value in Horner's Rule calculation -/
def v₂ (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  (a₄ * x + a₃) * x + a₂

theorem horner_v₂_for_specific_polynomial (x : ℝ) :
  v₂ 3 1 0 2 4 x = 3 * x * x + x := by sorry

theorem v₂_value_at_10 :
  v₂ 3 1 0 2 4 10 = 310 := by sorry

end NUMINAMATH_CALUDE_horner_v₂_for_specific_polynomial_v₂_value_at_10_l2435_243573


namespace NUMINAMATH_CALUDE_expanded_expression_equals_804095_l2435_243520

theorem expanded_expression_equals_804095 :
  8 * 10^5 + 4 * 10^3 + 9 * 10 + 5 = 804095 := by
  sorry

end NUMINAMATH_CALUDE_expanded_expression_equals_804095_l2435_243520


namespace NUMINAMATH_CALUDE_min_value_expression_l2435_243540

theorem min_value_expression (x y : ℝ) : x^2 + x*y + y^2 - 3*y ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2435_243540


namespace NUMINAMATH_CALUDE_tree_planting_solution_l2435_243561

/-- Represents the configuration of trees in a circle. -/
structure TreeCircle where
  total : ℕ
  birches : ℕ
  lindens : ℕ
  all_lindens_between_birches : Bool
  one_birch_same_neighbors : Bool

/-- The theorem stating the unique solution for the tree planting problem. -/
theorem tree_planting_solution (circle : TreeCircle) : 
  circle.total = 130 ∧ 
  circle.total = circle.birches + circle.lindens ∧ 
  circle.birches > 0 ∧ 
  circle.lindens > 0 ∧
  circle.all_lindens_between_birches = true ∧
  circle.one_birch_same_neighbors = true →
  circle.birches = 87 := by
  sorry

#check tree_planting_solution

end NUMINAMATH_CALUDE_tree_planting_solution_l2435_243561


namespace NUMINAMATH_CALUDE_josh_lost_marbles_l2435_243552

/-- Represents the number of marbles Josh lost -/
def marbles_lost (initial current : ℕ) : ℕ := initial - current

/-- Theorem stating that Josh lost 5 marbles -/
theorem josh_lost_marbles : marbles_lost 9 4 = 5 := by sorry

end NUMINAMATH_CALUDE_josh_lost_marbles_l2435_243552


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt5_l2435_243506

theorem rationalize_denominator_sqrt5 :
  ∃ (A B C : ℤ),
    (((2 : ℝ) + Real.sqrt 5) / ((3 : ℝ) - Real.sqrt 5) = (A : ℝ) + (B : ℝ) * Real.sqrt (C : ℝ)) ∧
    A * B * C = 275 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt5_l2435_243506


namespace NUMINAMATH_CALUDE_intersection_point_polar_curves_l2435_243571

theorem intersection_point_polar_curves (θ : Real) (ρ : Real) 
  (h1 : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h2 : ρ = 2 * Real.sin θ)
  (h3 : ρ * Real.cos θ = -1) :
  ∃ (ρ_intersect θ_intersect : Real),
    ρ_intersect = Real.sqrt (8 + 4 * Real.sqrt 3) ∧
    θ_intersect = 3 * Real.pi / 4 ∧
    ρ_intersect = 2 * Real.sin θ_intersect ∧
    ρ_intersect * Real.cos θ_intersect = -1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_polar_curves_l2435_243571


namespace NUMINAMATH_CALUDE_Z_is_real_Z_is_pure_imaginary_Z_in_fourth_quadrant_l2435_243565

-- Define the complex number Z as a function of real number m
def Z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

-- Theorem 1: Z is real iff m = -3 or m = 5
theorem Z_is_real (m : ℝ) : (Z m).im = 0 ↔ m = -3 ∨ m = 5 := by sorry

-- Theorem 2: Z is pure imaginary iff m = -2
theorem Z_is_pure_imaginary (m : ℝ) : (Z m).re = 0 ↔ m = -2 := by sorry

-- Theorem 3: Z is in the fourth quadrant iff -2 < m < 5
theorem Z_in_fourth_quadrant (m : ℝ) : 
  ((Z m).re > 0 ∧ (Z m).im < 0) ↔ -2 < m ∧ m < 5 := by sorry

end NUMINAMATH_CALUDE_Z_is_real_Z_is_pure_imaginary_Z_in_fourth_quadrant_l2435_243565


namespace NUMINAMATH_CALUDE_piece_in_313th_row_l2435_243509

/-- Represents a chessboard with pieces -/
structure Chessboard :=
  (size : ℕ)
  (pieces : ℕ)
  (symmetrical : Bool)

/-- Checks if a row contains a piece -/
def has_piece_in_row (board : Chessboard) (row : ℕ) : Prop :=
  sorry

theorem piece_in_313th_row (board : Chessboard) 
  (h1 : board.size = 625)
  (h2 : board.pieces = 1977)
  (h3 : board.symmetrical = true) :
  has_piece_in_row board 313 :=
sorry

end NUMINAMATH_CALUDE_piece_in_313th_row_l2435_243509


namespace NUMINAMATH_CALUDE_joan_marbles_l2435_243529

theorem joan_marbles (mary_marbles : ℕ) (total_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : total_marbles = 12) :
  total_marbles - mary_marbles = 3 :=
by sorry

end NUMINAMATH_CALUDE_joan_marbles_l2435_243529


namespace NUMINAMATH_CALUDE_max_value_of_sum_and_powers_l2435_243504

theorem max_value_of_sum_and_powers (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 → 
  ∃ (max : ℝ), max = 2 ∧ ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 2 → 
  a + b^3 + c^4 ≤ max := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_and_powers_l2435_243504


namespace NUMINAMATH_CALUDE_shortest_side_length_l2435_243587

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, the shortest side has length 1. -/
theorem shortest_side_length (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  a * c + c^2 = b^2 - a^2 →  -- given condition
  b = Real.sqrt 7 →  -- longest side is √7
  Real.sin C = 2 * Real.sin A →  -- given condition
  b ≥ a ∧ b ≥ c →  -- b is the longest side
  min a c = 1 :=  -- the shortest side has length 1
by sorry

end NUMINAMATH_CALUDE_shortest_side_length_l2435_243587


namespace NUMINAMATH_CALUDE_vector_subtraction_l2435_243591

/-- Given two plane vectors a and b, prove that a - 2b equals the expected result -/
theorem vector_subtraction (a b : ℝ × ℝ) : 
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2435_243591


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l2435_243543

theorem cylinder_radius_problem (r : ℝ) : 
  let h : ℝ := 3
  let volume_decrease_radius : ℝ := 3 * Real.pi * ((r - 4)^2 - r^2)
  let volume_decrease_height : ℝ := Real.pi * r^2 * (h - (h - 4))
  volume_decrease_radius = volume_decrease_height →
  (r = 6 + 2 * Real.sqrt 3 ∨ r = 6 - 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l2435_243543


namespace NUMINAMATH_CALUDE_rental_cost_difference_l2435_243575

/-- Calculates the total cost of renting a boat for two days with a discount on the second day -/
def total_cost (daily_rental : ℝ) (hourly_rental : ℝ) (hourly_fuel : ℝ) (hours_per_day : ℝ) (discount_rate : ℝ) : ℝ :=
  let first_day := daily_rental + hourly_fuel * hours_per_day
  let second_day := (daily_rental + hourly_fuel * hours_per_day) * (1 - discount_rate)
  first_day + second_day + hourly_rental * hours_per_day * 2

/-- The difference in cost between renting a ski boat and a sailboat -/
theorem rental_cost_difference : 
  let sailboat_cost := total_cost 60 0 10 3 0.1
  let ski_boat_cost := total_cost 0 80 20 3 0.1
  ski_boat_cost - sailboat_cost = 402 := by sorry

end NUMINAMATH_CALUDE_rental_cost_difference_l2435_243575


namespace NUMINAMATH_CALUDE_geometric_series_sum_2_to_2048_l2435_243557

def geometric_series_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

def last_term (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * r^(n - 1)

theorem geometric_series_sum_2_to_2048 :
  ∃ n : ℕ, 
    last_term 2 2 n = 2048 ∧ 
    geometric_series_sum 2 2 n = 4094 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_2_to_2048_l2435_243557


namespace NUMINAMATH_CALUDE_mobius_trip_time_l2435_243502

-- Define the constants from the problem
def distance : ℝ := 143
def speed_with_load : ℝ := 11
def speed_without_load : ℝ := 13
def rest_time_per_stop : ℝ := 0.5
def num_rest_stops : ℕ := 4

-- Define the theorem
theorem mobius_trip_time :
  let time_with_load := distance / speed_with_load
  let time_without_load := distance / speed_without_load
  let total_rest_time := rest_time_per_stop * num_rest_stops
  time_with_load + time_without_load + total_rest_time = 26 := by
sorry


end NUMINAMATH_CALUDE_mobius_trip_time_l2435_243502


namespace NUMINAMATH_CALUDE_hexagon_and_circle_construction_l2435_243597

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Constructs a hexagon from three non-adjacent vertex projections -/
def construct_hexagon (p1 p2 p3 : Point) : Hexagon :=
  sorry

/-- Constructs an inscribed circle for a given hexagon -/
def construct_inscribed_circle (h : Hexagon) : Circle :=
  sorry

theorem hexagon_and_circle_construction 
  (p1 p2 p3 : Point) 
  (h_not_collinear : ¬ are_collinear p1 p2 p3) :
  ∃ (hex : Hexagon) (circ : Circle), 
    hex = construct_hexagon p1 p2 p3 ∧ 
    circ = construct_inscribed_circle hex :=
  by sorry

end NUMINAMATH_CALUDE_hexagon_and_circle_construction_l2435_243597


namespace NUMINAMATH_CALUDE_seventh_day_cans_l2435_243522

/-- A sequence where the first term is 4 and each subsequent term increases by 5 -/
def canSequence : ℕ → ℕ
  | 0 => 4
  | n + 1 => canSequence n + 5

/-- The 7th term of the sequence is 34 -/
theorem seventh_day_cans : canSequence 6 = 34 := by
  sorry

end NUMINAMATH_CALUDE_seventh_day_cans_l2435_243522


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l2435_243545

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -6) : 
  x^2 + y^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l2435_243545


namespace NUMINAMATH_CALUDE_marbles_given_to_sam_l2435_243572

def initial_marbles : ℕ := 8
def remaining_marbles : ℕ := 4

theorem marbles_given_to_sam :
  initial_marbles - remaining_marbles = 4 :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_to_sam_l2435_243572


namespace NUMINAMATH_CALUDE_second_reader_average_pages_per_day_l2435_243580

/-- Calculates the average pages read per day by the second-place reader -/
def average_pages_per_day (break_days : ℕ) (deshaun_books : ℕ) (avg_pages_per_book : ℕ) (second_reader_percentage : ℚ) : ℚ :=
  let deshaun_total_pages := deshaun_books * avg_pages_per_book
  let second_reader_pages := second_reader_percentage * deshaun_total_pages
  second_reader_pages / break_days

/-- Theorem stating that the second-place reader averaged 180 pages per day -/
theorem second_reader_average_pages_per_day :
  average_pages_per_day 80 60 320 (3/4) = 180 := by
  sorry

end NUMINAMATH_CALUDE_second_reader_average_pages_per_day_l2435_243580


namespace NUMINAMATH_CALUDE_program_outputs_divisors_l2435_243525

/-- The set of numbers output by the program for a given input n -/
def program_output (n : ℕ) : Set ℕ :=
  {i : ℕ | i ≤ n ∧ n % i = 0}

/-- The set of all divisors of n -/
def divisors (n : ℕ) : Set ℕ :=
  {i : ℕ | i ∣ n}

/-- Theorem stating that the program output is equal to the set of all divisors -/
theorem program_outputs_divisors (n : ℕ) : program_output n = divisors n := by
  sorry

end NUMINAMATH_CALUDE_program_outputs_divisors_l2435_243525


namespace NUMINAMATH_CALUDE_salary_comparison_l2435_243547

/-- Represents the salary ratios of employees A, B, C, D, and E -/
def salary_ratios : Fin 5 → ℚ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5

/-- The combined salary of employees B, C, and D in rupees -/
def combined_salary_bcd : ℚ := 15000

/-- Calculates the base salary unit given the combined salary of B, C, and D -/
def base_salary : ℚ := combined_salary_bcd / (salary_ratios 1 + salary_ratios 2 + salary_ratios 3)

theorem salary_comparison :
  /- The salary of C is 200% more than that of A -/
  (salary_ratios 2 * base_salary - salary_ratios 0 * base_salary) / (salary_ratios 0 * base_salary) * 100 = 200 ∧
  /- The ratio of the salary of E to the combined salary of A and B is 5:3 -/
  (salary_ratios 4 * base_salary) / ((salary_ratios 0 + salary_ratios 1) * base_salary) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_salary_comparison_l2435_243547


namespace NUMINAMATH_CALUDE_no_real_solutions_l2435_243554

theorem no_real_solutions : ∀ x : ℝ, (2*x - 4*x + 7)^2 + 1 ≠ -|x^2 - 1| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2435_243554


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l2435_243559

theorem algebraic_expression_equality (a b : ℝ) (h1 : a = 3) (h2 : a - b = 1) :
  a^2 - a*b = a*(a - b) := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l2435_243559


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2435_243595

/-- Given that x and y are inversely proportional, x + y = 30, and x - y = 10,
    prove that y = 200/7 when x = 7 -/
theorem inverse_proportion_problem (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k)
    (h2 : x + y = 30) (h3 : x - y = 10) : 
    x = 7 → y = 200 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2435_243595


namespace NUMINAMATH_CALUDE_athlete_distance_difference_l2435_243503

theorem athlete_distance_difference : 
  let field_length : ℚ := 24
  let mary_fraction : ℚ := 3/8
  let edna_fraction : ℚ := 2/3
  let lucy_fraction : ℚ := 5/6
  let mary_distance : ℚ := field_length * mary_fraction
  let edna_distance : ℚ := mary_distance * edna_fraction
  let lucy_distance : ℚ := edna_distance * lucy_fraction
  mary_distance - lucy_distance = 4 := by
sorry

end NUMINAMATH_CALUDE_athlete_distance_difference_l2435_243503


namespace NUMINAMATH_CALUDE_equiangular_rational_sides_prime_is_regular_l2435_243546

/-- An equiangular polygon with p sides -/
structure EquiangularPolygon (p : ℕ) where
  sides : Fin p → ℚ
  is_equiangular : True  -- We assume this property is satisfied

/-- A regular polygon is an equiangular polygon with all sides equal -/
def is_regular (poly : EquiangularPolygon p) : Prop :=
  ∀ i j : Fin p, poly.sides i = poly.sides j

theorem equiangular_rational_sides_prime_is_regular
  (p : ℕ) (hp : p.Prime) (hp2 : p > 2) (poly : EquiangularPolygon p) :
  is_regular poly :=
sorry

end NUMINAMATH_CALUDE_equiangular_rational_sides_prime_is_regular_l2435_243546


namespace NUMINAMATH_CALUDE_basketball_practice_average_l2435_243534

/-- The average practice time per day for a basketball player over a week -/
theorem basketball_practice_average (weekday_hours : ℝ) (weekend_hours : ℝ) (weekdays : ℕ) (weekend_days : ℕ) :
  weekday_hours = 2 →
  weekend_hours = 11 →
  weekdays = 5 →
  weekend_days = 2 →
  (weekday_hours * weekdays + weekend_hours) / (weekdays + weekend_days) = 3 := by
  sorry

end NUMINAMATH_CALUDE_basketball_practice_average_l2435_243534


namespace NUMINAMATH_CALUDE_books_on_third_shelf_l2435_243592

/-- Represents the number of books on each shelf of a bookcase -/
structure Bookcase where
  shelf1 : ℕ
  shelf2 : ℕ
  shelf3 : ℕ

/-- Defines the properties of the bookcase in the problem -/
def ProblemBookcase (b : Bookcase) : Prop :=
  b.shelf1 + b.shelf2 + b.shelf3 = 275 ∧
  b.shelf3 = 3 * b.shelf2 + 8 ∧
  b.shelf1 = 2 * b.shelf2 - 3

theorem books_on_third_shelf :
  ∀ b : Bookcase, ProblemBookcase b → b.shelf3 = 188 :=
by
  sorry


end NUMINAMATH_CALUDE_books_on_third_shelf_l2435_243592


namespace NUMINAMATH_CALUDE_percentage_problem_l2435_243519

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 690 →
  (0.5 * x) = (P / 100 * 1500 - 30) →
  P = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2435_243519


namespace NUMINAMATH_CALUDE_mask_production_optimization_l2435_243541

/-- Represents the production plan for masks -/
structure MaskProduction where
  typeA : ℕ  -- Number of type A masks produced
  typeB : ℕ  -- Number of type B masks produced
  days : ℕ   -- Number of days used for production

/-- Checks if a production plan is valid according to the given conditions -/
def isValidProduction (p : MaskProduction) : Prop :=
  p.typeA + p.typeB = 50000 ∧
  p.typeA ≥ 18000 ∧
  p.days ≤ 8 ∧
  p.typeA ≤ 6000 * p.days ∧
  p.typeB ≤ 8000 * (p.days - (p.typeA / 6000))

/-- Calculates the profit for a given production plan -/
def profit (p : MaskProduction) : ℕ :=
  (p.typeA * 5 + p.typeB * 3) / 10

/-- Theorem stating the maximum profit and minimum production time -/
theorem mask_production_optimization :
  (∃ p : MaskProduction, isValidProduction p ∧
    (∀ q : MaskProduction, isValidProduction q → profit q ≤ profit p) ∧
    profit p = 23400) ∧
  (∃ p : MaskProduction, isValidProduction p ∧
    (∀ q : MaskProduction, isValidProduction q → p.days ≤ q.days) ∧
    p.days = 7) := by
  sorry


end NUMINAMATH_CALUDE_mask_production_optimization_l2435_243541


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2435_243535

/-- A line passing through point (1, 5) with equal intercepts on both coordinate axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point (1, 5) -/
  passes_through_point : slope * 1 + y_intercept = 5
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : y_intercept = -slope * y_intercept

/-- The equation of the line is either x + y - 6 = 0 or 5x - y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = -1 ∧ l.y_intercept = 6) ∨ (l.slope = 5 ∧ l.y_intercept = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2435_243535


namespace NUMINAMATH_CALUDE_max_determinable_elements_l2435_243538

open Finset

theorem max_determinable_elements : ∀ (a : Fin 11 → ℕ) (b : Fin 9 → ℕ),
  (∀ i : Fin 11, a i ∈ range 12 \ {0}) →
  (∀ i j : Fin 11, i ≠ j → a i ≠ a j) →
  (∀ i : Fin 9, b i = a i + a (i + 2)) →
  (∃ (S : Finset (Fin 11)), S.card = 5 ∧ 
    (∀ (a' : Fin 11 → ℕ),
      (∀ i : Fin 11, a' i ∈ range 12 \ {0}) →
      (∀ i j : Fin 11, i ≠ j → a' i ≠ a' j) →
      (∀ i : Fin 9, b i = a' i + a' (i + 2)) →
      (∀ i ∈ S, a i = a' i))) ∧
  ¬(∃ (S : Finset (Fin 11)), S.card > 5 ∧ 
    (∀ (a' : Fin 11 → ℕ),
      (∀ i : Fin 11, a' i ∈ range 12 \ {0}) →
      (∀ i j : Fin 11, i ≠ j → a' i ≠ a' j) →
      (∀ i : Fin 9, b i = a' i + a' (i + 2)) →
      (∀ i ∈ S, a i = a' i))) := by
  sorry

end NUMINAMATH_CALUDE_max_determinable_elements_l2435_243538


namespace NUMINAMATH_CALUDE_lines_concurrent_at_S_l2435_243523

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Tetrahedron SABC with points A', B', C' on edges SA, SB, SC respectively -/
structure Tetrahedron where
  S : Point3D
  A : Point3D
  B : Point3D
  C : Point3D
  A' : Point3D
  B' : Point3D
  C' : Point3D

/-- The intersection line d of planes ABC and A'B'C' -/
def intersection_line (t : Tetrahedron) : Line3D :=
  sorry

/-- Theorem: Lines AA', BB', CC' are concurrent at S for any rotation of A'B'C' around d -/
theorem lines_concurrent_at_S (t : Tetrahedron) (θ : ℝ) : 
  ∃ (S : Point3D), 
    (Line3D.mk t.A t.A').point = S ∧ 
    (Line3D.mk t.B t.B').point = S ∧ 
    (Line3D.mk t.C t.C').point = S := by
  sorry

end NUMINAMATH_CALUDE_lines_concurrent_at_S_l2435_243523


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_l2435_243577

/-- The area of a cross-section created by cutting a right circular cylinder -/
theorem cylinder_cross_section_area
  (r : ℝ) -- radius of the cylinder
  (h : ℝ) -- height of the cylinder
  (θ : ℝ) -- angle of the arc in radians
  (hr : r = 8) -- given radius
  (hh : h = 10) -- given height
  (hθ : θ = π / 2) -- 90° in radians
  : r^2 * θ * h = 320 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_l2435_243577


namespace NUMINAMATH_CALUDE_most_likely_parent_genotypes_l2435_243569

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Dominant hairy
| h  -- Recessive hairy
| S  -- Dominant smooth
| s  -- Recessive smooth

/-- Represents the genotype of a rabbit -/
structure Genotype :=
(allele1 : Allele)
(allele2 : Allele)

/-- Determines if a rabbit has hairy fur based on its genotype -/
def isHairy (g : Genotype) : Bool :=
  match g.allele1, g.allele2 with
  | Allele.H, _ => true
  | _, Allele.H => true
  | Allele.h, Allele.h => true
  | _, _ => false

/-- The probability of the hairy allele in the population -/
def p : ℝ := 0.1

/-- Theorem: The most likely genotype combination for parents resulting in all hairy offspring -/
theorem most_likely_parent_genotypes :
  ∃ (parent1 parent2 : Genotype),
    isHairy parent1 ∧
    ¬isHairy parent2 ∧
    (∀ (offspring : Genotype),
      (offspring.allele1 = parent1.allele1 ∨ offspring.allele1 = parent1.allele2) ∧
      (offspring.allele2 = parent2.allele1 ∨ offspring.allele2 = parent2.allele2) →
      isHairy offspring) ∧
    parent1 = ⟨Allele.H, Allele.H⟩ ∧
    parent2 = ⟨Allele.S, Allele.h⟩ :=
by sorry


end NUMINAMATH_CALUDE_most_likely_parent_genotypes_l2435_243569


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l2435_243505

theorem ancient_chinese_math_problem (people : ℕ) (price : ℕ) : 
  (8 * people - price = 3) →
  (price - 7 * people = 4) →
  people = 7 := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l2435_243505


namespace NUMINAMATH_CALUDE_return_speed_calculation_l2435_243566

/-- Given a round trip where the return speed is twice the outbound speed,
    prove that the return speed is 15 km/h when the total distance is 60 km
    and the total travel time is 6 hours. -/
theorem return_speed_calculation (distance : ℝ) (total_time : ℝ) (outbound_speed : ℝ) :
  distance = 60 →
  total_time = 6 →
  outbound_speed > 0 →
  distance / (2 * outbound_speed) + distance / (2 * (2 * outbound_speed)) = total_time →
  2 * outbound_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_return_speed_calculation_l2435_243566


namespace NUMINAMATH_CALUDE_cynthia_gallons_proof_l2435_243598

def pool_capacity : ℕ := 105
def num_trips : ℕ := 7
def caleb_gallons : ℕ := 7

theorem cynthia_gallons_proof :
  ∃ (cynthia_gallons : ℕ),
    cynthia_gallons * num_trips + caleb_gallons * num_trips = pool_capacity ∧
    cynthia_gallons = 8 := by
  sorry

end NUMINAMATH_CALUDE_cynthia_gallons_proof_l2435_243598


namespace NUMINAMATH_CALUDE_product_of_roots_l2435_243527

/-- The polynomial coefficients -/
def a : ℝ := 2
def b : ℝ := -5
def c : ℝ := -10
def d : ℝ := 22

/-- The polynomial equation -/
def polynomial (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem product_of_roots :
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, polynomial x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ * r₂ * r₃ = -11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2435_243527
