import Mathlib

namespace NUMINAMATH_CALUDE_max_sum_on_parabola_l2561_256182

theorem max_sum_on_parabola :
  ∃ (max : ℝ), max = 13/4 ∧ 
  ∀ (m n : ℝ), n = -m^2 + 3 → m + n ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_sum_on_parabola_l2561_256182


namespace NUMINAMATH_CALUDE_translate_quadratic_function_l2561_256120

/-- Represents a function f(x) = (x-a)^2 + b --/
def quadratic_function (a b : ℝ) : ℝ → ℝ := λ x ↦ (x - a)^2 + b

/-- Translates a function horizontally by h units and vertically by k units --/
def translate (f : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ := λ x ↦ f (x - h) + k

theorem translate_quadratic_function :
  let f := quadratic_function 2 1
  let g := translate f 1 1
  g = quadratic_function 1 2 := by sorry

end NUMINAMATH_CALUDE_translate_quadratic_function_l2561_256120


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l2561_256107

noncomputable section

open Real

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians),
    prove that if b = √3, c = 1, and B = π/3, then a = 2, A = π/2, and C = π/6 -/
theorem triangle_abc_problem (a b c A B C : ℝ) : 
  b = sqrt 3 → c = 1 → B = π/3 →
  (sin A) / a = (sin B) / b → (sin B) / b = (sin C) / c →  -- Law of sines
  A + B + C = π →  -- Angle sum in a triangle
  a = 2 ∧ A = π/2 ∧ C = π/6 := by
sorry

end

end NUMINAMATH_CALUDE_triangle_abc_problem_l2561_256107


namespace NUMINAMATH_CALUDE_some_number_value_l2561_256101

theorem some_number_value (some_number : ℝ) 
  (h1 : ∃ n : ℝ, (n / 18) * (n / some_number) = 1)
  (h2 : (54 / 18) * (54 / some_number) = 1) : 
  some_number = 162 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l2561_256101


namespace NUMINAMATH_CALUDE_inequality_proof_l2561_256148

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (a + c) ≥ 2 * Real.sqrt (a * b * c * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2561_256148


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_over_b_l2561_256196

-- Define the function f(x) = ax^2 + b
def f (a b x : ℝ) : ℝ := a * x^2 + b

-- Define the derivative of f
def f_derivative (a : ℝ) : ℝ → ℝ := λ x ↦ 2 * a * x

theorem tangent_slope_implies_a_over_b (a b : ℝ) : 
  f a b 1 = 3 ∧ f_derivative a 1 = 2 → a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_over_b_l2561_256196


namespace NUMINAMATH_CALUDE_test_configuration_theorem_l2561_256112

/-- Represents the fraction of problems that are difficult and the fraction of students who perform well -/
structure TestConfiguration (α : ℚ) :=
  (difficult_problems : ℚ)
  (well_performing_students : ℚ)
  (difficult_problems_ge : difficult_problems ≥ α)
  (well_performing_students_ge : well_performing_students ≥ α)

/-- Theorem stating the existence and non-existence of certain test configurations -/
theorem test_configuration_theorem :
  (∃ (config : TestConfiguration (2/3)), True) ∧
  (¬ ∃ (config : TestConfiguration (3/4)), True) ∧
  (¬ ∃ (config : TestConfiguration (7/10^7)), True) := by
  sorry

end NUMINAMATH_CALUDE_test_configuration_theorem_l2561_256112


namespace NUMINAMATH_CALUDE_negative_390_same_terminal_side_as_330_l2561_256118

-- Define a function to check if two angles have the same terminal side
def same_terminal_side (a b : Int) : Prop :=
  ∃ k : Int, a ≡ b + 360 * k [ZMOD 360]

-- State the theorem
theorem negative_390_same_terminal_side_as_330 :
  same_terminal_side (-390) 330 := by
  sorry

end NUMINAMATH_CALUDE_negative_390_same_terminal_side_as_330_l2561_256118


namespace NUMINAMATH_CALUDE_lucky_sum_equality_l2561_256170

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to form a sum of s using k distinct natural numbers less than n -/
def sumCombinations (s k n : ℕ) : ℕ := sorry

/-- The probability of event A: "the lucky sum in the main draw is 63" -/
def probA (n : ℕ) : ℚ :=
  (sumCombinations 63 10 n : ℚ) / choose n 10

/-- The probability of event B: "the lucky sum in the additional draw is 44" -/
def probB (n : ℕ) : ℚ :=
  (sumCombinations 44 8 n : ℚ) / choose n 8

theorem lucky_sum_equality :
  ∀ n : ℕ, (n ≥ 10 ∧ probA n = probB n) ↔ n = 18 := by sorry

end NUMINAMATH_CALUDE_lucky_sum_equality_l2561_256170


namespace NUMINAMATH_CALUDE_exam_correct_answers_l2561_256155

/-- Given an exam with the following conditions:
  * Total number of questions is 150
  * Correct answers score 4 marks
  * Wrong answers score -2 marks
  * Total score is 420 marks
  Prove that the number of correct answers is 120 -/
theorem exam_correct_answers 
  (total_questions : ℕ) 
  (correct_score wrong_score total_score : ℤ) 
  (h1 : total_questions = 150)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -2)
  (h4 : total_score = 420) : 
  ∃ (correct_answers : ℕ), 
    correct_answers = 120 ∧ 
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score :=
by
  sorry

end NUMINAMATH_CALUDE_exam_correct_answers_l2561_256155


namespace NUMINAMATH_CALUDE_harmonic_sum_terms_added_l2561_256106

theorem harmonic_sum_terms_added (k : ℕ) (h : k > 1) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_terms_added_l2561_256106


namespace NUMINAMATH_CALUDE_sector_arc_length_l2561_256161

/-- Given a circular sector with central angle 2π/3 and area 25π/3, 
    its arc length is 10π/3 -/
theorem sector_arc_length (α : Real) (S : Real) (l : Real) :
  α = 2 * π / 3 →
  S = 25 * π / 3 →
  l = 10 * π / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_sector_arc_length_l2561_256161


namespace NUMINAMATH_CALUDE_milk_cost_l2561_256145

/-- If 4 boxes of milk cost 26 yuan, then 6 boxes of the same milk will cost 39 yuan. -/
theorem milk_cost (cost : ℕ) (boxes : ℕ) (h1 : cost = 26) (h2 : boxes = 4) :
  (cost / boxes) * 6 = 39 :=
by sorry

end NUMINAMATH_CALUDE_milk_cost_l2561_256145


namespace NUMINAMATH_CALUDE_equation_solutions_l2561_256119

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = (1 + Real.sqrt 5) / 4 ∧ x2 = (1 - Real.sqrt 5) / 4 ∧
    4 * x1^2 - 2 * x1 - 1 = 0 ∧ 4 * x2^2 - 2 * x2 - 1 = 0) ∧
  (∃ y1 y2 : ℝ, y1 = 1 ∧ y2 = 0 ∧
    (y1 + 1)^2 = (3 * y1 - 1)^2 ∧ (y2 + 1)^2 = (3 * y2 - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2561_256119


namespace NUMINAMATH_CALUDE_function_inequality_l2561_256151

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) (h2 : ∀ x, deriv f x > 1) :
  f 3 > f 1 + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2561_256151


namespace NUMINAMATH_CALUDE_jerry_shelves_problem_l2561_256124

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (total_books - books_taken + books_per_shelf - 1) / books_per_shelf

theorem jerry_shelves_problem :
  shelves_needed 34 7 3 = 9 :=
by sorry

end NUMINAMATH_CALUDE_jerry_shelves_problem_l2561_256124


namespace NUMINAMATH_CALUDE_betty_balance_l2561_256122

/-- Betty's account balance given Gina's account information -/
theorem betty_balance (gina_account1 gina_account2 betty_balance : ℚ) : 
  gina_account1 = (1 / 4 : ℚ) * betty_balance →
  gina_account2 = (1 / 4 : ℚ) * betty_balance →
  gina_account1 + gina_account2 = 1728 →
  betty_balance = 3456 := by
  sorry

end NUMINAMATH_CALUDE_betty_balance_l2561_256122


namespace NUMINAMATH_CALUDE_perimeter_of_figure_C_l2561_256152

/-- Represents the dimensions of a rectangle in terms of small rectangles -/
structure RectDimension where
  width : ℕ
  height : ℕ

/-- Calculates the perimeter of a figure given its dimensions and the size of small rectangles -/
def perimeter (dim : RectDimension) (x y : ℝ) : ℝ :=
  2 * (dim.width * x + dim.height * y)

theorem perimeter_of_figure_C (x y : ℝ) : 
  perimeter ⟨6, 1⟩ x y = 56 →
  perimeter ⟨2, 3⟩ x y = 56 →
  perimeter ⟨1, 3⟩ x y = 40 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_C_l2561_256152


namespace NUMINAMATH_CALUDE_triangle_angle_zero_l2561_256162

theorem triangle_angle_zero (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_zero_l2561_256162


namespace NUMINAMATH_CALUDE_max_a_for_integer_solution_l2561_256128

theorem max_a_for_integer_solution : 
  (∃ (a : ℕ+), ∀ (b : ℕ+), 
    (∃ (x : ℤ), x^2 + (b : ℤ) * x = -30) → 
    (b : ℤ) ≤ (a : ℤ)) ∧ 
  (∃ (x : ℤ), x^2 + 31 * x = -30) := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_integer_solution_l2561_256128


namespace NUMINAMATH_CALUDE_value_of_d_l2561_256169

theorem value_of_d (c d : ℚ) (h1 : c / d = 4) (h2 : c = 15 - 4 * d) : d = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_d_l2561_256169


namespace NUMINAMATH_CALUDE_negation_equivalence_l2561_256132

theorem negation_equivalence :
  (¬ ∃ a ∈ Set.Icc (-1 : ℝ) 2, ∃ x : ℝ, a * x^2 + 1 < 0) ↔
  (∀ a ∈ Set.Icc (-1 : ℝ) 2, ∀ x : ℝ, a * x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2561_256132


namespace NUMINAMATH_CALUDE_regular_21gon_symmetry_sum_l2561_256173

/-- The number of sides in the regular polygon -/
def n : ℕ := 21

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle (in degrees) for which a regular n-gon has rotational symmetry -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 21-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is equal to 38 -/
theorem regular_21gon_symmetry_sum :
  (L n : ℚ) + R n = 38 := by sorry

end NUMINAMATH_CALUDE_regular_21gon_symmetry_sum_l2561_256173


namespace NUMINAMATH_CALUDE_divisibility_by_101_l2561_256197

theorem divisibility_by_101 (n : ℕ) : 
  (101 ∣ (10^n - 1)) ↔ (4 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_101_l2561_256197


namespace NUMINAMATH_CALUDE_tan_sum_diff_implies_sin_2alpha_cos_2beta_l2561_256179

theorem tan_sum_diff_implies_sin_2alpha_cos_2beta
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 1)
  (h2 : Real.tan (α - β) = 2) :
  (Real.sin (2 * α)) / (Real.cos (2 * β)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_diff_implies_sin_2alpha_cos_2beta_l2561_256179


namespace NUMINAMATH_CALUDE_example_polygon_area_l2561_256188

/-- A polygon on a unit grid with specified vertices -/
structure GridPolygon where
  vertices : List (Int × Int)

/-- Calculate the area of a GridPolygon -/
def area (p : GridPolygon) : ℕ :=
  sorry

/-- The specific polygon from the problem -/
def examplePolygon : GridPolygon :=
  { vertices := [(0,0), (20,0), (20,20), (10,20), (10,10), (0,10)] }

/-- Theorem stating that the area of the example polygon is 250 square units -/
theorem example_polygon_area : area examplePolygon = 250 :=
  sorry

end NUMINAMATH_CALUDE_example_polygon_area_l2561_256188


namespace NUMINAMATH_CALUDE_correlation_coefficient_properties_l2561_256178

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

-- Define positive correlation
def positively_correlated (x y : ℝ → ℝ) : Prop := 
  ∀ t₁ t₂, t₁ < t₂ → x t₁ < x t₂ → y t₁ < y t₂

-- Define the strength of linear correlation
def linear_correlation_strength (x y : ℝ → ℝ) : ℝ := sorry

-- Define perfect linear relationship
def perfect_linear_relationship (x y : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ t, y t = a * x t + b

-- Theorem statement
theorem correlation_coefficient_properties 
  (x y : ℝ → ℝ) (r : ℝ) (h : r = correlation_coefficient x y) :
  (r > 0 → positively_correlated x y) ∧
  (∀ ε > 0, |r| > 1 - ε → linear_correlation_strength x y > 1 - ε) ∧
  (r = 1 ∨ r = -1 → perfect_linear_relationship x y) := by
  sorry

end NUMINAMATH_CALUDE_correlation_coefficient_properties_l2561_256178


namespace NUMINAMATH_CALUDE_chord_division_ratio_l2561_256146

/-- Given a circle with radius 11 and a chord of length 18 intersected by a diameter at a point 7 units from the center, 
    the point of intersection divides the chord in a ratio of either 2:1 or 1:2. -/
theorem chord_division_ratio (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ) 
  (h_r : r = 11) (h_chord : chord_length = 18) (h_dist : intersection_distance = 7) :
  ∃ (x y : ℝ), (x + y = chord_length ∧ 
    ((x / y = 2 ∧ y / x = 1/2) ∨ (x / y = 1/2 ∧ y / x = 2)) ∧
    x * y = (r - intersection_distance) * (r + intersection_distance)) :=
sorry

end NUMINAMATH_CALUDE_chord_division_ratio_l2561_256146


namespace NUMINAMATH_CALUDE_g_domain_is_correct_l2561_256183

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def f_domain : Set ℝ := Set.Icc (-6) 9

-- Define the function g in terms of f
def g (x : ℝ) : ℝ := f (-3 * x)

-- Define the domain of g
def g_domain : Set ℝ := Set.Icc (-3) 2

-- Theorem statement
theorem g_domain_is_correct : 
  {x : ℝ | g x ∈ f_domain} = g_domain := by sorry

end NUMINAMATH_CALUDE_g_domain_is_correct_l2561_256183


namespace NUMINAMATH_CALUDE_fraction_equality_l2561_256180

theorem fraction_equality (a : ℕ+) : (a : ℚ) / (a + 35 : ℚ) = 875 / 1000 → a = 245 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2561_256180


namespace NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l2561_256126

theorem unique_solution_diophantine_equation :
  ∃! (x y z t : ℕ+), 1 + 5^x.val = 2^y.val + 2^z.val * 5^t.val :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l2561_256126


namespace NUMINAMATH_CALUDE_nickel_count_l2561_256108

structure CoinPurse where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

def totalValue (p : CoinPurse) : ℕ :=
  25 * p.quarters + 10 * p.dimes + 5 * p.nickels + p.pennies

def totalCoins (p : CoinPurse) : ℕ :=
  p.quarters + p.dimes + p.nickels + p.pennies

theorem nickel_count (p : CoinPurse) 
  (h1 : totalValue p = 17 * totalCoins p)
  (h2 : totalValue p - 1 = 18 * (totalCoins p - 1)) :
  p.nickels = 2 := by
sorry

end NUMINAMATH_CALUDE_nickel_count_l2561_256108


namespace NUMINAMATH_CALUDE_sue_necklace_purple_beads_l2561_256193

theorem sue_necklace_purple_beads :
  ∀ (purple blue green : ℕ),
    purple + blue + green = 46 →
    blue = 2 * purple →
    green = blue + 11 →
    purple = 7 := by
  sorry

end NUMINAMATH_CALUDE_sue_necklace_purple_beads_l2561_256193


namespace NUMINAMATH_CALUDE_cards_in_unfilled_box_l2561_256186

theorem cards_in_unfilled_box (total_cards : ℕ) (cards_per_box : ℕ) 
  (h1 : total_cards = 94) (h2 : cards_per_box = 8) : 
  total_cards % cards_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_cards_in_unfilled_box_l2561_256186


namespace NUMINAMATH_CALUDE_canada_population_density_l2561_256195

def population : ℕ := 38005238
def area_sq_miles : ℕ := 3855103
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

def total_sq_feet : ℕ := area_sq_miles * sq_feet_per_sq_mile
def avg_sq_feet_per_person : ℚ := total_sq_feet / population

theorem canada_population_density :
  (2700000 : ℚ) < avg_sq_feet_per_person ∧ avg_sq_feet_per_person < (2900000 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_canada_population_density_l2561_256195


namespace NUMINAMATH_CALUDE_tangent_line_to_exp_curve_l2561_256185

theorem tangent_line_to_exp_curve (x y : ℝ) :
  (∃ (m b : ℝ), y = m * x + b ∧ 
    (∀ (x₀ : ℝ), Real.exp x₀ = m * x₀ + b → x₀ = 1 ∨ x₀ = x) ∧
    0 = m * 1 + b) →
  Real.exp 2 * x - y - Real.exp 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_exp_curve_l2561_256185


namespace NUMINAMATH_CALUDE_is_arithmetic_sequence_pn_plus_q_l2561_256105

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of the sequence. -/
def a (n : ℕ) (p q : ℝ) : ℝ := p * n + q

/-- Theorem: A sequence with general term a_n = pn + q is an arithmetic sequence. -/
theorem is_arithmetic_sequence_pn_plus_q (p q : ℝ) :
  IsArithmeticSequence (a · p q) := by
  sorry

end NUMINAMATH_CALUDE_is_arithmetic_sequence_pn_plus_q_l2561_256105


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2561_256136

theorem necessary_but_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x ≤ 1/2 ∧ y ≤ 1/2 → x + y ≤ 1) ∧
  (∃ x y : ℝ, x + y ≤ 1 ∧ ¬(x ≤ 1/2 ∧ y ≤ 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2561_256136


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2561_256164

theorem floor_equation_solution (n : ℤ) : 
  (⌊n^2 / 4⌋ - ⌊n / 2⌋^2 = 3) ↔ (n = 7) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2561_256164


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2561_256138

theorem arithmetic_mean_problem (x y : ℝ) : 
  (8 + x + 21 + y + 14 + 11) / 6 = 15 → x + y = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2561_256138


namespace NUMINAMATH_CALUDE_expand_expression_l2561_256104

theorem expand_expression (x y : ℝ) : (x + 12) * (3 * y + 8) = 3 * x * y + 8 * x + 36 * y + 96 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2561_256104


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l2561_256147

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α : Plane) 
  (h_diff : m ≠ n) 
  (h_perp : perpendicular m α) 
  (h_contained : contained_in n α) : 
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l2561_256147


namespace NUMINAMATH_CALUDE_rational_roots_of_polynomial_l2561_256137

theorem rational_roots_of_polynomial (x : ℚ) :
  (4 * x^4 - 3 * x^3 - 13 * x^2 + 5 * x + 2 = 0) ↔ (x = 2 ∨ x = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_of_polynomial_l2561_256137


namespace NUMINAMATH_CALUDE_no_perfect_square_3000_001_l2561_256121

theorem no_perfect_square_3000_001 (n : ℕ) : ¬ ∃ k : ℤ, (3 * 10^n + 1 : ℤ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_3000_001_l2561_256121


namespace NUMINAMATH_CALUDE_expansive_sequence_existence_l2561_256167

def expansive (a : ℕ → ℝ) : Prop :=
  ∀ i j : ℕ, i < j → |a i - a j| ≥ 1 / j

theorem expansive_sequence_existence (C : ℝ) :
  (C > 0 ∧ ∃ a : ℕ → ℝ, expansive a ∧ ∀ n, 0 ≤ a n ∧ a n ≤ C) ↔ C ≥ 2 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_expansive_sequence_existence_l2561_256167


namespace NUMINAMATH_CALUDE_ethanol_in_full_tank_l2561_256135

def tank_capacity : ℝ := 212
def fuel_A_volume : ℝ := 98
def fuel_A_ethanol_percentage : ℝ := 0.12
def fuel_B_ethanol_percentage : ℝ := 0.16

theorem ethanol_in_full_tank : 
  let fuel_B_volume := tank_capacity - fuel_A_volume
  let ethanol_in_A := fuel_A_volume * fuel_A_ethanol_percentage
  let ethanol_in_B := fuel_B_volume * fuel_B_ethanol_percentage
  ethanol_in_A + ethanol_in_B = 30 := by
sorry

end NUMINAMATH_CALUDE_ethanol_in_full_tank_l2561_256135


namespace NUMINAMATH_CALUDE_largest_common_term_l2561_256154

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

def is_in_first_sequence (x : ℤ) : Prop :=
  ∃ n : ℕ, x = arithmetic_sequence 3 8 n

def is_in_second_sequence (x : ℤ) : Prop :=
  ∃ n : ℕ, x = arithmetic_sequence 5 9 n

theorem largest_common_term :
  ∀ x : ℤ, 1 ≤ x ∧ x ≤ 200 ∧ is_in_first_sequence x ∧ is_in_second_sequence x →
  x ≤ 131 ∧ is_in_first_sequence 131 ∧ is_in_second_sequence 131 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l2561_256154


namespace NUMINAMATH_CALUDE_series_sum_eight_l2561_256141

def series_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2^(n + 1) + series_sum n

theorem series_sum_eight : series_sum 8 = 510 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_eight_l2561_256141


namespace NUMINAMATH_CALUDE_prob_red_or_blue_l2561_256158

-- Define the total number of marbles
def total_marbles : ℕ := 120

-- Define the probabilities of each color
def prob_white : ℚ := 1/5
def prob_green : ℚ := 1/10
def prob_orange : ℚ := 1/6
def prob_violet : ℚ := 1/8

-- Theorem statement
theorem prob_red_or_blue :
  let prob_others := prob_white + prob_green + prob_orange + prob_violet
  (1 - prob_others) = 49/120 := by sorry

end NUMINAMATH_CALUDE_prob_red_or_blue_l2561_256158


namespace NUMINAMATH_CALUDE_jake_first_test_score_l2561_256166

/-- Represents the marks Jake scored in his tests -/
structure JakeScores where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Theorem stating that Jake's first test score was 80 given the conditions -/
theorem jake_first_test_score (scores : JakeScores) :
  (scores.first + scores.second + scores.third + scores.fourth) / 4 = 75 →
  scores.second = scores.first + 10 →
  scores.third = scores.fourth →
  scores.third = 65 →
  scores.first = 80 := by
  sorry

#check jake_first_test_score

end NUMINAMATH_CALUDE_jake_first_test_score_l2561_256166


namespace NUMINAMATH_CALUDE_difference_not_1998_l2561_256171

theorem difference_not_1998 (n m : ℕ) : (n^2 + 4*n) - (m^2 + 4*m) ≠ 1998 := by
  sorry

end NUMINAMATH_CALUDE_difference_not_1998_l2561_256171


namespace NUMINAMATH_CALUDE_g_max_value_l2561_256175

/-- The function g(x) = 4x - x^3 -/
def g (x : ℝ) : ℝ := 4 * x - x^3

/-- The maximum value of g(x) on [0, 2] is 8√3/9 -/
theorem g_max_value : 
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = (8 * Real.sqrt 3) / 9 := by
sorry

end NUMINAMATH_CALUDE_g_max_value_l2561_256175


namespace NUMINAMATH_CALUDE_only_positive_number_l2561_256176

theorem only_positive_number (numbers : Set ℝ) : 
  numbers = {0, 5, -1/2, -Real.sqrt 2} → 
  (∃ x ∈ numbers, x > 0) ∧ (∀ y ∈ numbers, y > 0 → y = 5) := by
sorry

end NUMINAMATH_CALUDE_only_positive_number_l2561_256176


namespace NUMINAMATH_CALUDE_unique_n_with_divisor_property_l2561_256184

def has_ten_divisors (n : ℕ) : Prop :=
  ∃ (d : Fin 10 → ℕ), d 0 = 1 ∧ d 9 = n ∧
    (∀ i : Fin 9, d i < d (i + 1)) ∧
    (∀ m : ℕ, m ∣ n ↔ ∃ i : Fin 10, d i = m)

theorem unique_n_with_divisor_property :
  ∀ n : ℕ, n > 0 →
    has_ten_divisors n →
    (∃ (d : Fin 10 → ℕ), 2 * n = (d 4)^2 + (d 5)^2 - 1) →
    n = 272 :=
sorry

end NUMINAMATH_CALUDE_unique_n_with_divisor_property_l2561_256184


namespace NUMINAMATH_CALUDE_crayons_given_to_friends_l2561_256160

theorem crayons_given_to_friends 
  (initial_crayons : ℕ)
  (lost_crayons : ℕ)
  (extra_crayons_given : ℕ)
  (h1 : initial_crayons = 589)
  (h2 : lost_crayons = 161)
  (h3 : extra_crayons_given = 410) :
  lost_crayons + extra_crayons_given = 571 :=
by
  sorry

end NUMINAMATH_CALUDE_crayons_given_to_friends_l2561_256160


namespace NUMINAMATH_CALUDE_opposite_sides_line_parameter_range_l2561_256139

/-- Given two points on opposite sides of a line, determine the range of the line's parameter --/
theorem opposite_sides_line_parameter_range :
  ∀ (m : ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ = 3 ∧ y₁ = 1 ∧ x₂ = -4 ∧ y₂ = 6 ∧
    (3 * x₁ - 2 * y₁ + m) * (3 * x₂ - 2 * y₂ + m) < 0) →
  7 < m ∧ m < 24 :=
by sorry


end NUMINAMATH_CALUDE_opposite_sides_line_parameter_range_l2561_256139


namespace NUMINAMATH_CALUDE_not_all_zero_iff_one_nonzero_l2561_256189

theorem not_all_zero_iff_one_nonzero (a b c : ℝ) :
  ¬(a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_not_all_zero_iff_one_nonzero_l2561_256189


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2561_256157

/-- Given two vectors a and b in ℝ², prove that if a = (1,2) and b = (-1,m) are perpendicular, then m = 1/2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) : 
  a = (1, 2) → b = (-1, m) → a.1 * b.1 + a.2 * b.2 = 0 → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2561_256157


namespace NUMINAMATH_CALUDE_juice_cost_proof_l2561_256117

/-- The cost of 5 cans of juice during a store's anniversary sale -/
def cost_of_five_juice_cans : ℝ := by sorry

theorem juice_cost_proof (original_ice_cream_price : ℝ) 
                         (ice_cream_discount : ℝ) 
                         (total_cost : ℝ) : 
  original_ice_cream_price = 12 →
  ice_cream_discount = 2 →
  total_cost = 24 →
  2 * (original_ice_cream_price - ice_cream_discount) + 2 * cost_of_five_juice_cans = total_cost →
  cost_of_five_juice_cans = 2 := by sorry

end NUMINAMATH_CALUDE_juice_cost_proof_l2561_256117


namespace NUMINAMATH_CALUDE_ellipse_intersection_ratio_l2561_256130

/-- Given an ellipse mx^2 + ny^2 = 1 intersecting with a line y = -x + 1,
    if a line through the origin and the midpoint of the intersection points
    has slope √2/2, then n/m = √2 -/
theorem ellipse_intersection_ratio (m n : ℝ) (h_pos : m > 0 ∧ n > 0) :
  (∃ A B : ℝ × ℝ,
    (m * A.1^2 + n * A.2^2 = 1) ∧
    (m * B.1^2 + n * B.2^2 = 1) ∧
    (A.2 = -A.1 + 1) ∧
    (B.2 = -B.1 + 1) ∧
    ((A.2 + B.2) / (A.1 + B.1) = Real.sqrt 2 / 2)) →
  n / m = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_ratio_l2561_256130


namespace NUMINAMATH_CALUDE_remaining_average_l2561_256168

theorem remaining_average (total : ℕ) (subset : ℕ) (total_avg : ℚ) (subset_avg : ℚ) :
  total = 6 →
  subset = 4 →
  total_avg = 8 →
  subset_avg = 5 →
  (total_avg * total - subset_avg * subset) / (total - subset) = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_average_l2561_256168


namespace NUMINAMATH_CALUDE_flower_cost_is_nine_l2561_256144

/-- The cost of planting flowers -/
def flower_planting (flower_cost : ℚ) : Prop :=
  let pot_cost : ℚ := flower_cost + 20
  let soil_cost : ℚ := flower_cost - 2
  flower_cost + pot_cost + soil_cost = 45

/-- Theorem: The cost of the flower is $9 -/
theorem flower_cost_is_nine : ∃ (flower_cost : ℚ), flower_cost = 9 ∧ flower_planting flower_cost := by
  sorry

end NUMINAMATH_CALUDE_flower_cost_is_nine_l2561_256144


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l2561_256109

theorem max_value_trig_expression (a b c : ℝ) :
  (∀ θ : ℝ, c * (Real.cos θ)^2 ≠ -a) →
  (∃ M : ℝ, M = Real.sqrt (a^2 + b^2 + c^2) ∧
    ∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.tan θ ≤ M) :=
sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l2561_256109


namespace NUMINAMATH_CALUDE_probability_continuous_stripe_is_two_over_81_l2561_256111

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron :=
  (faces : Fin 4 → Face)

/-- Represents a face of the tetrahedron -/
structure Face :=
  (vertices : Fin 3 → Vertex)
  (stripe_start : Vertex)

/-- Represents a vertex of a face -/
inductive Vertex
| A | B | C

/-- Represents a stripe configuration on the tetrahedron -/
def StripeConfiguration := RegularTetrahedron

/-- Checks if a stripe configuration forms a continuous stripe around the tetrahedron -/
def is_continuous_stripe (config : StripeConfiguration) : Prop :=
  sorry

/-- The total number of possible stripe configurations -/
def total_configurations : ℕ := 81

/-- The number of stripe configurations that form a continuous stripe -/
def continuous_stripe_configurations : ℕ := 2

/-- The probability of a continuous stripe encircling the tetrahedron -/
def probability_continuous_stripe : ℚ :=
  continuous_stripe_configurations / total_configurations

theorem probability_continuous_stripe_is_two_over_81 :
  probability_continuous_stripe = 2 / 81 :=
sorry

end NUMINAMATH_CALUDE_probability_continuous_stripe_is_two_over_81_l2561_256111


namespace NUMINAMATH_CALUDE_mass_percentage_iodine_value_of_x_l2561_256153

-- Define constants for molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_I : ℝ := 126.90
def molar_mass_H2O : ℝ := 18.015

-- Define the sample mass
def sample_mass : ℝ := 50

-- Define variables for masses of AlI₃ and H₂O in the sample
variable (mass_AlI3 : ℝ)
variable (mass_H2O : ℝ)

-- Calculate molar mass of AlI₃
def molar_mass_AlI3 : ℝ := molar_mass_Al + 3 * molar_mass_I

-- Define the theorem for mass percentage of iodine
theorem mass_percentage_iodine :
  let mass_iodine := mass_AlI3 * (3 * molar_mass_I / molar_mass_AlI3)
  (mass_iodine / sample_mass) * 100 = 
  (mass_AlI3 * (3 * molar_mass_I / molar_mass_AlI3) / sample_mass) * 100 :=
by sorry

-- Define the theorem for the value of x
theorem value_of_x :
  let moles_water := mass_H2O / molar_mass_H2O
  let moles_AlI3 := mass_AlI3 / molar_mass_AlI3
  (moles_water / moles_AlI3) = 
  (mass_H2O / molar_mass_H2O) / (mass_AlI3 / molar_mass_AlI3) :=
by sorry

end NUMINAMATH_CALUDE_mass_percentage_iodine_value_of_x_l2561_256153


namespace NUMINAMATH_CALUDE_probability_green_or_white_specific_l2561_256100

/-- The probability of drawing either a green or white marble from a bag -/
def probability_green_or_white (green white black : ℕ) : ℚ :=
  (green + white) / (green + white + black)

/-- Theorem stating the probability of drawing a green or white marble -/
theorem probability_green_or_white_specific :
  probability_green_or_white 4 3 8 = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_or_white_specific_l2561_256100


namespace NUMINAMATH_CALUDE_brookes_added_balloons_l2561_256102

/-- Prove that Brooke added 8 balloons to his collection -/
theorem brookes_added_balloons :
  ∀ (brooke_initial tracy_initial tracy_added total_after : ℕ) 
    (brooke_added : ℕ),
  brooke_initial = 12 →
  tracy_initial = 6 →
  tracy_added = 24 →
  total_after = 35 →
  brooke_initial + brooke_added + (tracy_initial + tracy_added) / 2 = total_after →
  brooke_added = 8 := by
sorry

end NUMINAMATH_CALUDE_brookes_added_balloons_l2561_256102


namespace NUMINAMATH_CALUDE_minjin_apples_l2561_256123

theorem minjin_apples : ∃ (initial : ℕ), 
  (initial % 8 = 0) ∧ 
  (6 * ((initial / 8) + 8 - 30) = 12) ∧ 
  (initial = 192) := by
  sorry

end NUMINAMATH_CALUDE_minjin_apples_l2561_256123


namespace NUMINAMATH_CALUDE_equation_linearity_implies_m_n_values_l2561_256127

/-- A linear equation in two variables has the form ax + by = c, where a, b, and c are constants -/
def is_linear_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

/-- The equation 3x^(2m+1) - 2y^(n-1) = 7 -/
def equation (m n : ℕ) (x y : ℝ) : ℝ :=
  3 * x^(2*m+1) - 2 * y^(n-1) - 7

theorem equation_linearity_implies_m_n_values (m n : ℕ) :
  is_linear_in_two_variables (equation m n) → m = 0 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_linearity_implies_m_n_values_l2561_256127


namespace NUMINAMATH_CALUDE_sin_cos_inequality_l2561_256165

theorem sin_cos_inequality (x : ℝ) : (Real.sin x + 2 * Real.cos (2 * x)) * (2 * Real.sin (2 * x) - Real.cos x) < 4.5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l2561_256165


namespace NUMINAMATH_CALUDE_rd_investment_exceeds_200_million_in_2019_l2561_256191

/-- Proves that 2019 is the first year when the annual R&D bonus investment exceeds $200 million -/
theorem rd_investment_exceeds_200_million_in_2019 
  (initial_investment : ℝ) 
  (annual_increase_rate : ℝ) 
  (h1 : initial_investment = 130) 
  (h2 : annual_increase_rate = 0.12) : 
  ∃ (n : ℕ), 
    (n = 2019) ∧ 
    (initial_investment * (1 + annual_increase_rate) ^ (n - 2015) > 200) ∧ 
    (∀ m : ℕ, m < n → initial_investment * (1 + annual_increase_rate) ^ (m - 2015) ≤ 200) := by
  sorry

end NUMINAMATH_CALUDE_rd_investment_exceeds_200_million_in_2019_l2561_256191


namespace NUMINAMATH_CALUDE_derivative_of_log2_l2561_256116

-- Define the base-2 logarithm
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem derivative_of_log2 (x : ℝ) (h : x > 0) :
  deriv log2 x = 1 / (x * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_log2_l2561_256116


namespace NUMINAMATH_CALUDE_total_spent_is_83_50_l2561_256103

-- Define the ticket prices
def adult_ticket_price : ℚ := 5.5
def child_ticket_price : ℚ := 3.5

-- Define the total number of tickets and number of adult tickets
def total_tickets : ℕ := 21
def adult_tickets : ℕ := 5

-- Define the function to calculate total spent
def total_spent : ℚ :=
  (adult_tickets : ℚ) * adult_ticket_price + 
  ((total_tickets - adult_tickets) : ℚ) * child_ticket_price

-- Theorem statement
theorem total_spent_is_83_50 : total_spent = 83.5 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_83_50_l2561_256103


namespace NUMINAMATH_CALUDE_min_value_2x_3y_l2561_256198

theorem min_value_2x_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y + 3*x*y = 6) :
  ∀ z : ℝ, (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 3*b + 3*a*b = 6 ∧ 2*a + 3*b = z) → z ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_3y_l2561_256198


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l2561_256172

theorem simple_interest_principal_calculation
  (rate : ℝ) (interest : ℝ) (time : ℝ) (principal : ℝ)
  (h_rate : rate = 4.166666666666667)
  (h_interest : interest = 130)
  (h_time : time = 4)
  (h_formula : interest = principal * rate * time / 100) :
  principal = 780 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l2561_256172


namespace NUMINAMATH_CALUDE_fifth_root_of_161051_l2561_256150

theorem fifth_root_of_161051 : ∃ n : ℕ, n^5 = 161051 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_161051_l2561_256150


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2561_256142

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 7003000

/-- The scientific notation representation of the number -/
def scientific_form : ScientificNotation :=
  { coefficient := 7.003
    exponent := 6
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2561_256142


namespace NUMINAMATH_CALUDE_gmat_exam_problem_l2561_256174

theorem gmat_exam_problem (total : ℕ) (h_total : total > 0) :
  let first_correct := (80 : ℚ) / 100 * total
  let second_correct := (75 : ℚ) / 100 * total
  let neither_correct := (5 : ℚ) / 100 * total
  let both_correct := first_correct + second_correct - total + neither_correct
  (both_correct / total) = (60 : ℚ) / 100 := by
sorry

end NUMINAMATH_CALUDE_gmat_exam_problem_l2561_256174


namespace NUMINAMATH_CALUDE_apples_for_juice_l2561_256159

/-- Given that 36 apples make 27 liters of apple juice, prove that 12 apples make 9 liters of apple juice -/
theorem apples_for_juice (apples : ℕ) (juice : ℕ) (h : 36 * juice = 27 * apples) : 
  12 * juice = 9 * apples :=
by sorry

end NUMINAMATH_CALUDE_apples_for_juice_l2561_256159


namespace NUMINAMATH_CALUDE_only_first_equation_has_nonzero_solution_l2561_256190

theorem only_first_equation_has_nonzero_solution :
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ Real.sqrt (a^2 + b^2) = a ∧
  (∀ (a b : ℝ), Real.sqrt (a^2 + b^2) = Real.sqrt a * Real.sqrt b → a = 0 ∧ b = 0) ∧
  (∀ (a b : ℝ), Real.sqrt (a^2 + b^2) = a * b → a = 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_only_first_equation_has_nonzero_solution_l2561_256190


namespace NUMINAMATH_CALUDE_min_max_perimeter_12_pieces_l2561_256113

/-- Represents a rectangular piece with length and width in centimeters -/
structure Piece where
  length : ℝ
  width : ℝ

/-- Represents a collection of identical rectangular pieces -/
structure PieceCollection where
  piece : Piece
  count : ℕ

/-- Calculates the area of a rectangular piece -/
def pieceArea (p : Piece) : ℝ := p.length * p.width

/-- Calculates the total area of a collection of pieces -/
def totalArea (pc : PieceCollection) : ℝ := (pieceArea pc.piece) * pc.count

/-- Calculates the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: Minimum and maximum perimeter of rectangle formed by 12 pieces of 4x3 cm -/
theorem min_max_perimeter_12_pieces :
  let pieces : PieceCollection := ⟨⟨4, 3⟩, 12⟩
  let area : ℝ := totalArea pieces
  ∃ (min_perim max_perim : ℝ),
    min_perim = 48 ∧
    max_perim = 102 ∧
    (∀ (l w : ℝ), l * w = area → rectanglePerimeter l w ≥ min_perim) ∧
    (∃ (l w : ℝ), l * w = area ∧ rectanglePerimeter l w = max_perim) :=
by sorry

end NUMINAMATH_CALUDE_min_max_perimeter_12_pieces_l2561_256113


namespace NUMINAMATH_CALUDE_all_heads_possible_l2561_256194

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents the state of all coins in a row -/
def CoinRow := Vector CoinState 100

/-- An operation that flips 7 equally spaced coins -/
def FlipOperation := Fin 100 → Fin 7 → Bool

/-- Applies a flip operation to a coin row -/
def applyFlip (row : CoinRow) (op : FlipOperation) : CoinRow :=
  sorry

/-- The theorem stating that any initial coin configuration can be transformed to all heads -/
theorem all_heads_possible (initial : CoinRow) : 
  ∃ (ops : List FlipOperation), 
    let final := ops.foldl applyFlip initial
    ∀ i, final.get i = CoinState.Heads :=
  sorry

end NUMINAMATH_CALUDE_all_heads_possible_l2561_256194


namespace NUMINAMATH_CALUDE_simplified_expression_ratio_l2561_256163

theorem simplified_expression_ratio (m : ℝ) :
  let original := (6 * m + 18) / 6
  ∃ (c d : ℤ), (∃ (x : ℝ), original = c * x + d) ∧ (c : ℚ) / d = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_ratio_l2561_256163


namespace NUMINAMATH_CALUDE_circle_equation_l2561_256125

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

def center_on_line (c : Circle) : Prop :=
  c.center.2 = 2 * c.center.1

def tangent_to_line (c : Circle) : Prop :=
  c.radius = |2 * c.center.1 - c.center.2 + 5| / Real.sqrt 5

-- Theorem statement
theorem circle_equation (c : Circle) :
  passes_through c (3, 2) ∧
  center_on_line c ∧
  tangent_to_line c →
  ((λ (x y : ℝ) => (x - 2)^2 + (y - 4)^2 = 5) c.center.1 c.center.2) ∨
  ((λ (x y : ℝ) => (x - 4/5)^2 + (y - 8/5)^2 = 5) c.center.1 c.center.2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2561_256125


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l2561_256114

/-- Given a quadratic function f(x) = 3x^2 - 2x + 8, when shifted 6 units to the left,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 141 -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3 * x^2 - 2 * x + 8) →
  (∀ x, g x = f (x + 6)) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 141 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l2561_256114


namespace NUMINAMATH_CALUDE_monomial_sum_implies_a_power_l2561_256115

/-- Given two monomials in x and y whose sum is a monomial, prove a^2004 - 1 = 0 --/
theorem monomial_sum_implies_a_power (m n : ℤ) (a : ℕ) :
  (∃ (x y : ℝ), (3 * m * x^a * y) + (-2 * n * x^(4*a - 3) * y) = x^k * y) →
  a^2004 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_a_power_l2561_256115


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2561_256143

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 180 * r = b ∧ b * r = 36 / 25) → 
  b = Real.sqrt (6480 / 25) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2561_256143


namespace NUMINAMATH_CALUDE_randy_blocks_proof_l2561_256149

theorem randy_blocks_proof (house_blocks tower_blocks : ℕ) 
  (h1 : house_blocks = 89)
  (h2 : tower_blocks = 63)
  (h3 : house_blocks - tower_blocks = 26) :
  house_blocks + tower_blocks = 152 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_proof_l2561_256149


namespace NUMINAMATH_CALUDE_unique_triple_solution_l2561_256181

theorem unique_triple_solution : ∃! (a b c : ℕ+), 5^(a.val) + 3^(b.val) - 2^(c.val) = 32 ∧ a = 2 ∧ b = 2 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l2561_256181


namespace NUMINAMATH_CALUDE_correct_subtraction_l2561_256177

theorem correct_subtraction (x : ℤ) : x - 32 = 25 → x - 23 = 34 := by sorry

end NUMINAMATH_CALUDE_correct_subtraction_l2561_256177


namespace NUMINAMATH_CALUDE_three_by_four_grid_squares_l2561_256131

/-- A structure representing a grid of squares -/
structure SquareGrid where
  rows : Nat
  cols : Nat
  total_small_squares : Nat

/-- Function to count the total number of squares in a grid -/
def count_total_squares (grid : SquareGrid) : Nat :=
  sorry

/-- Theorem stating that a 3x4 grid of 12 small squares contains 17 total squares -/
theorem three_by_four_grid_squares :
  let grid := SquareGrid.mk 3 4 12
  count_total_squares grid = 17 :=
by sorry

end NUMINAMATH_CALUDE_three_by_four_grid_squares_l2561_256131


namespace NUMINAMATH_CALUDE_half_of_three_fifths_of_120_l2561_256133

theorem half_of_three_fifths_of_120 : (1/2 : ℚ) * ((3/5 : ℚ) * 120) = 36 := by
  sorry

end NUMINAMATH_CALUDE_half_of_three_fifths_of_120_l2561_256133


namespace NUMINAMATH_CALUDE_no_valid_coloring_l2561_256199

/-- A coloring of a 5x5 board using 4 colors -/
def Coloring := Fin 5 → Fin 5 → Fin 4

/-- Predicate to check if a coloring satisfies the constraint -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ (r1 r2 c1 c2 : Fin 5), r1 ≠ r2 → c1 ≠ c2 →
    (Finset.card {c r1 c1, c r1 c2, c r2 c1, c r2 c2} ≥ 3)

/-- Theorem stating that no valid coloring exists -/
theorem no_valid_coloring : ¬ ∃ (c : Coloring), ValidColoring c := by
  sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l2561_256199


namespace NUMINAMATH_CALUDE_square_area_proof_l2561_256187

theorem square_area_proof (x : ℚ) :
  (5 * x - 20 : ℚ) = (25 - 2 * x : ℚ) →
  ((5 * x - 20)^2 : ℚ) = 7225 / 49 := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l2561_256187


namespace NUMINAMATH_CALUDE_four_part_cut_possible_five_triangular_part_cut_possible_l2561_256129

-- Define the original figure
def original_figure : Set (ℝ × ℝ) :=
  sorry

-- Define the area of the original figure
def original_area : ℝ := 64

-- Define a square with area 64
def target_square : Set (ℝ × ℝ) :=
  sorry

-- Define a function that represents cutting the figure into parts
def cut (figure : Set (ℝ × ℝ)) (n : ℕ) : List (Set (ℝ × ℝ)) :=
  sorry

-- Define a function that represents assembling parts into a new figure
def assemble (parts : List (Set (ℝ × ℝ))) : Set (ℝ × ℝ) :=
  sorry

-- Define a predicate to check if a set is triangular
def is_triangular (s : Set (ℝ × ℝ)) : Prop :=
  sorry

-- Theorem for part a
theorem four_part_cut_possible :
  ∃ (parts : List (Set (ℝ × ℝ))),
    parts.length ≤ 4 ∧
    (∀ p ∈ parts, p ⊆ original_figure) ∧
    assemble parts = target_square :=
  sorry

-- Theorem for part b
theorem five_triangular_part_cut_possible :
  ∃ (parts : List (Set (ℝ × ℝ))),
    parts.length ≤ 5 ∧
    (∀ p ∈ parts, p ⊆ original_figure ∧ is_triangular p) ∧
    assemble parts = target_square :=
  sorry

end NUMINAMATH_CALUDE_four_part_cut_possible_five_triangular_part_cut_possible_l2561_256129


namespace NUMINAMATH_CALUDE_temp_rise_negative_equals_decrease_l2561_256110

/-- Represents a temperature change in degrees Celsius -/
structure TemperatureChange where
  value : ℝ
  unit : String

/-- Defines a temperature rise -/
def temperature_rise (t : ℝ) : TemperatureChange :=
  { value := t, unit := "°C" }

/-- Defines a temperature decrease -/
def temperature_decrease (t : ℝ) : TemperatureChange :=
  { value := t, unit := "°C" }

/-- Theorem stating that a temperature rise of -2°C is equivalent to a temperature decrease of 2°C -/
theorem temp_rise_negative_equals_decrease :
  temperature_rise (-2) = temperature_decrease 2 := by
  sorry

end NUMINAMATH_CALUDE_temp_rise_negative_equals_decrease_l2561_256110


namespace NUMINAMATH_CALUDE_area_relationship_l2561_256140

/-- A right triangle with sides 18, 24, and 30 -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right : side1^2 + side2^2 = hypotenuse^2
  side1_eq : side1 = 18
  side2_eq : side2 = 24
  hypotenuse_eq : hypotenuse = 30

/-- Areas of non-triangular regions in a circumscribed circle -/
structure CircleAreas where
  D : ℝ
  E : ℝ
  F : ℝ
  F_largest : F ≥ D ∧ F ≥ E

/-- Theorem stating the relationship between areas D, E, F, and the triangle area -/
theorem area_relationship (t : RightTriangle) (areas : CircleAreas) :
  areas.D + areas.E + 216 = areas.F := by
  sorry

end NUMINAMATH_CALUDE_area_relationship_l2561_256140


namespace NUMINAMATH_CALUDE_solution_set_and_range_l2561_256156

def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≤ 6 ↔ x ∈ Set.Icc (-1) 2) ∧
  (∀ a : ℝ, a > 0 → (∃ x : ℝ, f x < |a - 2|) ↔ a > 6) := by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l2561_256156


namespace NUMINAMATH_CALUDE_f_g_3_eq_6_l2561_256192

def f (x : ℝ) : ℝ := 2 * x + 4

def g (x : ℝ) : ℝ := x^2 - 8

theorem f_g_3_eq_6 : f (g 3) = 6 := by sorry

end NUMINAMATH_CALUDE_f_g_3_eq_6_l2561_256192


namespace NUMINAMATH_CALUDE_largest_gold_coins_l2561_256134

theorem largest_gold_coins (n : ℕ) : 
  (∃ k : ℕ, n = 13 * k + 3) ∧ 
  n < 150 → 
  n ≤ 146 ∧ 
  (∃ m : ℕ, m > n ∧ (∃ j : ℕ, m = 13 * j + 3) → m ≥ 150) := by
sorry

end NUMINAMATH_CALUDE_largest_gold_coins_l2561_256134
