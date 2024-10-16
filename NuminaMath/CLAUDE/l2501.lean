import Mathlib

namespace NUMINAMATH_CALUDE_special_function_inequality_l2501_250199

/-- A function that is increasing on (1,+∞) and has F(x) = f(x+1) symmetrical about the y-axis -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x, f (-x + 1) = f (x + 1))

/-- Theorem: For a special function f, f(-1) > f(2) -/
theorem special_function_inequality (f : ℝ → ℝ) (h : SpecialFunction f) : f (-1) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l2501_250199


namespace NUMINAMATH_CALUDE_students_in_grade_l2501_250112

theorem students_in_grade (n : ℕ) (misha : ℕ) : 
  (misha = n - 59) ∧ (misha = 60) → n = 119 :=
by sorry

end NUMINAMATH_CALUDE_students_in_grade_l2501_250112


namespace NUMINAMATH_CALUDE_typing_sequences_count_l2501_250114

/-- Represents the state of letters in the secretary's inbox -/
structure LetterState where
  letters : List Nat
  typed : List Nat

/-- Calculates the number of possible typing sequences -/
def countSequences (state : LetterState) : Nat :=
  sorry

/-- The initial state of letters -/
def initialState : LetterState :=
  { letters := [1, 2, 3, 4, 5, 6, 7, 8, 9], typed := [] }

/-- The state after typing letters 8 and 5 -/
def stateAfterTyping : LetterState :=
  { letters := [6, 7, 9], typed := [8, 5] }

theorem typing_sequences_count :
  countSequences stateAfterTyping = 32 :=
sorry

end NUMINAMATH_CALUDE_typing_sequences_count_l2501_250114


namespace NUMINAMATH_CALUDE_zeros_in_square_of_999999999_l2501_250108

/-- The number of zeros in the decimal expansion of (999,999,999)^2 -/
def zeros_in_square_of_nines : ℕ := 8

/-- The observed pattern: squaring a number with n nines results in n-1 zeros -/
axiom pattern_holds (n : ℕ) : 
  ∀ x : ℕ, x = 10^n - 1 → (∃ k : ℕ, x^2 = k * 10^(n-1) ∧ k % 10 ≠ 0)

/-- Theorem: The number of zeros in the decimal expansion of (999,999,999)^2 is 8 -/
theorem zeros_in_square_of_999999999 : 
  ∃ k : ℕ, (999999999 : ℕ)^2 = k * 10^zeros_in_square_of_nines ∧ k % 10 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zeros_in_square_of_999999999_l2501_250108


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l2501_250127

/-- A hexagon in 2D space -/
structure Hexagon where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  v5 : ℝ × ℝ
  v6 : ℝ × ℝ

/-- The area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- The specific hexagon from the problem -/
def specificHexagon : Hexagon :=
  { v1 := (0, 0)
    v2 := (1, 4)
    v3 := (3, 4)
    v4 := (4, 0)
    v5 := (3, -4)
    v6 := (1, -4) }

/-- Theorem stating that the area of the specific hexagon is 24 square units -/
theorem specific_hexagon_area :
  hexagonArea specificHexagon = 24 := by sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l2501_250127


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l2501_250105

/-- Represents a normal distribution with mean μ and standard deviation σ -/
noncomputable def NormalDistribution (μ σ : ℝ) : Type :=
  ℝ → ℝ

/-- The probability that a random variable X from a normal distribution
    falls within the interval [a, b] -/
noncomputable def prob_between (X : NormalDistribution μ σ) (a b : ℝ) : ℝ :=
  sorry

/-- The probability that a random variable X from a normal distribution
    is greater than or equal to a given value -/
noncomputable def prob_ge (X : NormalDistribution μ σ) (a : ℝ) : ℝ :=
  sorry

theorem normal_distribution_symmetry 
  (X : NormalDistribution 100 σ) 
  (h : prob_between X 80 120 = 3/4) : 
  prob_ge X 120 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l2501_250105


namespace NUMINAMATH_CALUDE_square_root_equation_l2501_250147

theorem square_root_equation (x : ℝ) : Real.sqrt (5 * x - 1) = 3 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l2501_250147


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2501_250148

theorem modulus_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (1 - i^3) * (1 + 2*i)
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2501_250148


namespace NUMINAMATH_CALUDE_square_difference_l2501_250139

theorem square_difference : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2501_250139


namespace NUMINAMATH_CALUDE_sin_105_cos_105_l2501_250132

theorem sin_105_cos_105 : Real.sin (105 * π / 180) * Real.cos (105 * π / 180) = -(1/4) := by
  sorry

end NUMINAMATH_CALUDE_sin_105_cos_105_l2501_250132


namespace NUMINAMATH_CALUDE_total_distance_is_250_l2501_250165

/-- Represents a cyclist's journey with specific conditions -/
structure CyclistJourney where
  speed : ℝ
  time_store_to_friend : ℝ
  distance_store_to_friend : ℝ
  h_speed_positive : speed > 0
  h_time_positive : time_store_to_friend > 0
  h_distance_positive : distance_store_to_friend > 0
  h_distance_store_to_friend : distance_store_to_friend = 50
  h_time_relation : 2 * time_store_to_friend = speed * distance_store_to_friend

/-- The total distance cycled in the journey -/
def total_distance (j : CyclistJourney) : ℝ :=
  3 * j.distance_store_to_friend + j.distance_store_to_friend

/-- Theorem stating that the total distance cycled is 250 miles -/
theorem total_distance_is_250 (j : CyclistJourney) : total_distance j = 250 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_250_l2501_250165


namespace NUMINAMATH_CALUDE_journey_time_proof_l2501_250126

theorem journey_time_proof (s : ℝ) (h1 : s > 0) (h2 : s - 1/2 > 0) : 
  (45 / (s - 1/2) - 45 / s = 3/4) → (45 / s = 45 / s) :=
by
  sorry

end NUMINAMATH_CALUDE_journey_time_proof_l2501_250126


namespace NUMINAMATH_CALUDE_disjoint_subsets_sum_theorem_l2501_250131

theorem disjoint_subsets_sum_theorem (S : Set ℕ) (M₁ M₂ M₃ : Set ℕ) 
  (h1 : M₁ ⊆ S) (h2 : M₂ ⊆ S) (h3 : M₃ ⊆ S)
  (h4 : M₁ ∩ M₂ = ∅) (h5 : M₁ ∩ M₃ = ∅) (h6 : M₂ ∩ M₃ = ∅) :
  ∃ (X Y : ℕ), (X ∈ M₁ ∧ Y ∈ M₂) ∨ (X ∈ M₁ ∧ Y ∈ M₃) ∨ (X ∈ M₂ ∧ Y ∈ M₃) ∧ 
    (X + Y ∉ M₁ ∨ X + Y ∉ M₂ ∨ X + Y ∉ M₃) :=
by sorry

end NUMINAMATH_CALUDE_disjoint_subsets_sum_theorem_l2501_250131


namespace NUMINAMATH_CALUDE_largest_decimal_l2501_250160

theorem largest_decimal (a b c d e : ℚ) 
  (ha : a = 989/1000) 
  (hb : b = 9879/10000) 
  (hc : c = 98809/100000) 
  (hd : d = 9807/10000) 
  (he : e = 9819/10000) : 
  a = max a (max b (max c (max d e))) := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l2501_250160


namespace NUMINAMATH_CALUDE_exists_non_zero_sign_function_l2501_250133

/-- Given functions on a blackboard -/
def f₁ (x : ℝ) : ℝ := x + 1
def f₂ (x : ℝ) : ℝ := x^2 + 1
def f₃ (x : ℝ) : ℝ := x^3 + 1
def f₄ (x : ℝ) : ℝ := x^4 + 1

/-- The set of functions that can be constructed from the given functions -/
inductive ConstructibleFunction : (ℝ → ℝ) → Prop
  | base₁ : ConstructibleFunction f₁
  | base₂ : ConstructibleFunction f₂
  | base₃ : ConstructibleFunction f₃
  | base₄ : ConstructibleFunction f₄
  | sub (f g : ℝ → ℝ) : ConstructibleFunction f → ConstructibleFunction g → ConstructibleFunction (λ x => f x - g x)
  | mul (f g : ℝ → ℝ) : ConstructibleFunction f → ConstructibleFunction g → ConstructibleFunction (λ x => f x * g x)

/-- The theorem to be proved -/
theorem exists_non_zero_sign_function :
  ∃ (f : ℝ → ℝ), ConstructibleFunction f ∧ f ≠ 0 ∧
  (∀ x > 0, f x ≥ 0) ∧ (∀ x < 0, f x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_zero_sign_function_l2501_250133


namespace NUMINAMATH_CALUDE_prob_other_side_red_given_red_l2501_250190

/-- Represents the types of cards in the box -/
inductive Card
  | BlackBlack
  | BlackRed
  | RedRed

/-- The total number of cards in the box -/
def total_cards : Nat := 7

/-- The number of black-black cards -/
def black_black_cards : Nat := 2

/-- The number of black-red cards -/
def black_red_cards : Nat := 3

/-- The number of red-red cards -/
def red_red_cards : Nat := 2

/-- The total number of red faces -/
def total_red_faces : Nat := black_red_cards + 2 * red_red_cards

/-- The number of red faces on completely red cards -/
def red_faces_on_red_cards : Nat := 2 * red_red_cards

/-- The probability of seeing a red face and the other side being red -/
theorem prob_other_side_red_given_red (h1 : total_cards = black_black_cards + black_red_cards + red_red_cards)
  (h2 : total_red_faces = black_red_cards + 2 * red_red_cards)
  (h3 : red_faces_on_red_cards = 2 * red_red_cards) :
  (red_faces_on_red_cards : ℚ) / total_red_faces = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_other_side_red_given_red_l2501_250190


namespace NUMINAMATH_CALUDE_system_solution_unique_l2501_250193

theorem system_solution_unique : 
  ∃! (x y z : ℝ), 3*x + 2*y - z = 4 ∧ 2*x - y + 3*z = 9 ∧ x - 2*y + 2*z = 3 ∧ x = 1 ∧ y = 2 ∧ z = 3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2501_250193


namespace NUMINAMATH_CALUDE_figure_y_value_l2501_250166

/-- Given a figure with a right triangle and two squares, prove the value of y -/
theorem figure_y_value (y : ℝ) (total_area : ℝ) : 
  total_area = 980 →
  (3 * y)^2 + (6 * y)^2 + (1/2 * 3 * y * 6 * y) = total_area →
  y = 70/9 := by
sorry

end NUMINAMATH_CALUDE_figure_y_value_l2501_250166


namespace NUMINAMATH_CALUDE_rocky_fights_l2501_250151

/-- Represents the number of fights Rocky boxed in his career. -/
def total_fights : ℕ := sorry

/-- The fraction of fights that were knockouts. -/
def knockout_fraction : ℚ := 1/2

/-- The fraction of knockouts that were in the first round. -/
def first_round_knockout_fraction : ℚ := 1/5

/-- The number of knockouts in the first round. -/
def first_round_knockouts : ℕ := 19

theorem rocky_fights : 
  total_fights = 190 ∧ 
  (knockout_fraction * first_round_knockout_fraction * total_fights : ℚ) = first_round_knockouts := by
  sorry

end NUMINAMATH_CALUDE_rocky_fights_l2501_250151


namespace NUMINAMATH_CALUDE_john_annual_cost_l2501_250136

def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def replacements_per_year : ℕ := 2

def annual_cost : ℝ :=
  replacements_per_year * (epipen_cost * (1 - insurance_coverage))

theorem john_annual_cost : annual_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_cost_l2501_250136


namespace NUMINAMATH_CALUDE_line_properties_l2501_250180

/-- Given a line passing through two points and a direction vector format, prove the value of 'a' and the x-intercept. -/
theorem line_properties (p1 p2 : ℝ × ℝ) (a : ℝ) :
  p1 = (-3, 7) →
  p2 = (2, -2) →
  (∃ k : ℝ, k • (p2.1 - p1.1, p2.2 - p1.2) = (a, -1)) →
  a = 5/9 ∧ 
  (∃ x : ℝ, x = 4 ∧ 0 = -x + 4) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l2501_250180


namespace NUMINAMATH_CALUDE_system_solution_l2501_250152

def solution_set := {x : ℝ | 0 < x ∧ x < 1}

theorem system_solution : 
  {x : ℝ | x * (x + 2) > 0 ∧ |x| < 1} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2501_250152


namespace NUMINAMATH_CALUDE_square_sum_inequality_l2501_250111

theorem square_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  a^2 + b^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l2501_250111


namespace NUMINAMATH_CALUDE_max_cookies_without_ingredients_l2501_250125

theorem max_cookies_without_ingredients (total_cookies : ℕ) 
  (peanut_cookies : ℕ) (choc_cookies : ℕ) (almond_cookies : ℕ) (raisin_cookies : ℕ) : 
  total_cookies = 60 →
  peanut_cookies ≥ 20 →
  choc_cookies ≥ 15 →
  almond_cookies ≥ 12 →
  raisin_cookies ≥ 7 →
  ∃ (plain_cookies : ℕ), plain_cookies ≤ 6 ∧ 
    plain_cookies + peanut_cookies + choc_cookies + almond_cookies + raisin_cookies ≥ total_cookies := by
  sorry

end NUMINAMATH_CALUDE_max_cookies_without_ingredients_l2501_250125


namespace NUMINAMATH_CALUDE_perfect_apples_count_l2501_250121

/-- Represents the number of perfect apples in a batch with given conditions -/
def number_of_perfect_apples (total_apples : ℕ) 
  (small_ratio medium_ratio large_ratio : ℚ)
  (unripe_ratio partly_ripe_ratio fully_ripe_ratio : ℚ) : ℕ :=
  22

/-- Theorem stating the number of perfect apples under given conditions -/
theorem perfect_apples_count : 
  number_of_perfect_apples 60 (1/4) (1/2) (1/4) (1/3) (1/6) (1/2) = 22 := by
  sorry

end NUMINAMATH_CALUDE_perfect_apples_count_l2501_250121


namespace NUMINAMATH_CALUDE_inequality_proof_l2501_250189

theorem inequality_proof (a b c d e : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) :
  (a/b)^4 + (b/c)^4 + (c/d)^4 + (d/e)^4 + (e/a)^4 ≥ a/b + b/c + c/d + d/e + e/a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2501_250189


namespace NUMINAMATH_CALUDE_box_volume_formula_l2501_250173

/-- The volume of a box formed by cutting squares from corners of a metal sheet -/
def boxVolume (x : ℝ) : ℝ :=
  (16 - 2*x) * (12 - 2*x) * x

theorem box_volume_formula (x : ℝ) :
  boxVolume x = 192*x - 56*x^2 + 4*x^3 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_formula_l2501_250173


namespace NUMINAMATH_CALUDE_inequality_proof_l2501_250171

theorem inequality_proof (a b c : ℝ) 
  (ha : a = 1 / 10)
  (hb : b = Real.sin 1 / (9 + Real.cos 1))
  (hc : c = (Real.exp (1 / 10)) - 1) :
  b < a ∧ a < c :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2501_250171


namespace NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l2501_250134

-- Define the matrix A
def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 3; 5, d]

-- State the theorem
theorem matrix_inverse_scalar_multiple
  (d k : ℝ) :
  (A d)⁻¹ = k • (A d) →
  d = -2 ∧ k = 1/19 :=
sorry

end NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l2501_250134


namespace NUMINAMATH_CALUDE_sum_of_207_instances_of_33_difference_25_instances_of_112_from_3000_difference_product_and_sum_of_12_and_13_l2501_250146

-- Question 1
theorem sum_of_207_instances_of_33 : (Finset.range 207).sum (λ _ => 33) = 6831 := by sorry

-- Question 2
theorem difference_25_instances_of_112_from_3000 : 3000 - 25 * 112 = 200 := by sorry

-- Question 3
theorem difference_product_and_sum_of_12_and_13 : 12 * 13 - (12 + 13) = 131 := by sorry

end NUMINAMATH_CALUDE_sum_of_207_instances_of_33_difference_25_instances_of_112_from_3000_difference_product_and_sum_of_12_and_13_l2501_250146


namespace NUMINAMATH_CALUDE_theta_values_l2501_250150

theorem theta_values (a b : ℝ) (θ : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.cos (x + 2 * θ) + b * x + 3
  (f 1 = 5 ∧ f (-1) = 1) → (θ = π / 4 ∨ θ = -π / 4) :=
by sorry

end NUMINAMATH_CALUDE_theta_values_l2501_250150


namespace NUMINAMATH_CALUDE_missing_number_is_sixty_l2501_250142

/-- Given that the average of 20, 40, and 60 is 5 more than the average of 10, x, and 35,
    prove that x = 60. -/
theorem missing_number_is_sixty :
  ∃ x : ℝ, (20 + 40 + 60) / 3 = (10 + x + 35) / 3 + 5 → x = 60 := by
sorry

end NUMINAMATH_CALUDE_missing_number_is_sixty_l2501_250142


namespace NUMINAMATH_CALUDE_discount_calculation_l2501_250107

/-- Calculates the discount given the cost price, markup percentage, and loss percentage -/
def calculate_discount (cost_price : ℝ) (markup_percentage : ℝ) (loss_percentage : ℝ) : ℝ :=
  let marked_price := cost_price * (1 + markup_percentage)
  let selling_price := cost_price * (1 - loss_percentage)
  marked_price - selling_price

/-- Theorem stating that for a cost price of 100, a markup of 40%, and a loss of 1%, the discount is 41 -/
theorem discount_calculation :
  calculate_discount 100 0.4 0.01 = 41 := by
  sorry


end NUMINAMATH_CALUDE_discount_calculation_l2501_250107


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2501_250106

theorem quadratic_roots_property (m : ℝ) (hm : m ≠ -1) :
  let f : ℝ → ℝ := λ x => (m + 1) * x^2 + 4 * m * x + m - 3
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ (x₁ < -1 ∨ x₂ < -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2501_250106


namespace NUMINAMATH_CALUDE_vector_decomposition_l2501_250156

theorem vector_decomposition (x p q r : ℝ × ℝ × ℝ) 
  (hx : x = (-9, -8, -3))
  (hp : p = (1, 4, 1))
  (hq : q = (-3, 2, 0))
  (hr : r = (1, -1, 2)) :
  ∃ (α β γ : ℝ), x = α • p + β • q + γ • r ∧ α = -3 ∧ β = 2 ∧ γ = 0 :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l2501_250156


namespace NUMINAMATH_CALUDE_smallest_m_chess_tournament_l2501_250196

theorem smallest_m_chess_tournament : ∃ (m : ℕ), m > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (
    (∃ (x : ℕ), x > 0 ∧
      (4 * k * (4 * k - 1)) / 2 = 11 * x ∧
      8 * x + 3 * x = (4 * k * (4 * k - 1)) / 2
    ) → k ≥ m
  )) ∧ 
  (∃ (x : ℕ), x > 0 ∧
    (4 * m * (4 * m - 1)) / 2 = 11 * x ∧
    8 * x + 3 * x = (4 * m * (4 * m - 1)) / 2
  ) ∧
  m = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_chess_tournament_l2501_250196


namespace NUMINAMATH_CALUDE_triangle_side_length_l2501_250101

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Conditions
  b = 3 * Real.sqrt 3 →
  B = Real.pi / 3 →
  Real.sin A = 1 / 3 →
  -- Law of Sines (given as an additional condition since it's a fundamental property)
  a / Real.sin B = b / Real.sin A →
  -- Conclusion
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2501_250101


namespace NUMINAMATH_CALUDE_shopkeeper_milk_ounces_l2501_250192

/-- Calculates the total amount of milk in ounces bought by a shopkeeper -/
theorem shopkeeper_milk_ounces 
  (packets : ℕ) 
  (ml_per_packet : ℕ) 
  (ml_per_ounce : ℕ) 
  (h1 : packets = 150)
  (h2 : ml_per_packet = 250)
  (h3 : ml_per_ounce = 30) : 
  (packets * ml_per_packet) / ml_per_ounce = 1250 := by
  sorry

#check shopkeeper_milk_ounces

end NUMINAMATH_CALUDE_shopkeeper_milk_ounces_l2501_250192


namespace NUMINAMATH_CALUDE_apples_bought_by_junhyeok_and_jihyun_l2501_250103

/-- The number of apple boxes Junhyeok bought -/
def junhyeok_boxes : ℕ := 7

/-- The number of apples in each of Junhyeok's boxes -/
def junhyeok_apples_per_box : ℕ := 16

/-- The number of apple boxes Jihyun bought -/
def jihyun_boxes : ℕ := 6

/-- The number of apples in each of Jihyun's boxes -/
def jihyun_apples_per_box : ℕ := 25

/-- The total number of apples bought by Junhyeok and Jihyun -/
def total_apples : ℕ := junhyeok_boxes * junhyeok_apples_per_box + jihyun_boxes * jihyun_apples_per_box

theorem apples_bought_by_junhyeok_and_jihyun : total_apples = 262 := by
  sorry

end NUMINAMATH_CALUDE_apples_bought_by_junhyeok_and_jihyun_l2501_250103


namespace NUMINAMATH_CALUDE_correlation_coefficient_properties_l2501_250163

/-- Linear correlation coefficient between two variables -/
def linear_correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- Positive correlation between two variables -/
def positively_correlated (x y : ℝ → ℝ) : Prop := sorry

/-- Perfect linear relationship between two variables -/
def perfect_linear_relationship (x y : ℝ → ℝ) : Prop := sorry

theorem correlation_coefficient_properties
  (x y : ℝ → ℝ) (r : ℝ) (h : r = linear_correlation_coefficient x y) :
  ((r > 0 → positively_correlated x y) ∧
   (r = 1 ∨ r = -1 → perfect_linear_relationship x y)) := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_properties_l2501_250163


namespace NUMINAMATH_CALUDE_greatest_divisor_l2501_250186

def problem (n : ℕ) : Prop :=
  n > 0 ∧
  ∃ q1 q2 : ℕ, 1255 = n * q1 + 8 ∧ 1490 = n * q2 + 11 ∧
  ∀ m : ℕ, m > 0 → (∃ r1 r2 : ℕ, 1255 = m * r1 + 8 ∧ 1490 = m * r2 + 11) → m ≤ n

theorem greatest_divisor : problem 29 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_l2501_250186


namespace NUMINAMATH_CALUDE_three_points_on_circle_at_distance_from_line_l2501_250169

theorem three_points_on_circle_at_distance_from_line :
  ∃! (points : Finset (ℝ × ℝ)), points.card = 3 ∧
  (∀ p ∈ points, p.1^2 + p.2^2 = 4 ∧
    (|p.1 - p.2 + Real.sqrt 2|) / Real.sqrt 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_three_points_on_circle_at_distance_from_line_l2501_250169


namespace NUMINAMATH_CALUDE_arrangements_A_not_head_B_not_tail_arrangements_at_least_one_between_A_B_arrangements_A_B_together_C_D_not_together_l2501_250176

-- Define the number of people
def n : ℕ := 5

-- Define the factorial function
def factorial (m : ℕ) : ℕ := (List.range m).foldl (· * ·) 1

-- Define the permutation function
def permutation (m k : ℕ) : ℕ := 
  if k > m then 0
  else factorial m / factorial (m - k)

-- Theorem 1
theorem arrangements_A_not_head_B_not_tail : 
  permutation n n - 2 * permutation (n - 1) (n - 1) + permutation (n - 2) (n - 2) = 78 := by sorry

-- Theorem 2
theorem arrangements_at_least_one_between_A_B :
  permutation n n - permutation (n - 1) (n - 1) * permutation 2 2 = 72 := by sorry

-- Theorem 3
theorem arrangements_A_B_together_C_D_not_together :
  permutation 2 2 * permutation 2 2 * permutation 3 2 = 24 := by sorry

end NUMINAMATH_CALUDE_arrangements_A_not_head_B_not_tail_arrangements_at_least_one_between_A_B_arrangements_A_B_together_C_D_not_together_l2501_250176


namespace NUMINAMATH_CALUDE_cubic_equation_has_real_root_l2501_250117

theorem cubic_equation_has_real_root (a b : ℝ) : 
  ∃ x : ℝ, x^3 + a*x + b = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_has_real_root_l2501_250117


namespace NUMINAMATH_CALUDE_shuttle_speed_km_per_hour_l2501_250116

/-- The speed of a space shuttle orbiting the Earth -/
def shuttle_speed_km_per_sec : ℝ := 2

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: The speed of the space shuttle in kilometers per hour -/
theorem shuttle_speed_km_per_hour :
  shuttle_speed_km_per_sec * (seconds_per_hour : ℝ) = 7200 := by
  sorry

end NUMINAMATH_CALUDE_shuttle_speed_km_per_hour_l2501_250116


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2501_250172

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ → ℝ × ℝ := λ m ↦ (2 + m, 3 - m)
  let c : ℝ → ℝ × ℝ := λ m ↦ (3 * m, 1)
  ∀ m : ℝ, (∃ k : ℝ, a = k • (c m - b m)) → m = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2501_250172


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l2501_250170

/-- Calculates the rate of paving per square meter given room dimensions and total cost -/
theorem paving_rate_calculation (length width total_cost : ℝ) :
  length = 5.5 ∧ width = 3.75 ∧ total_cost = 12375 →
  total_cost / (length * width) = 600 := by
  sorry

#check paving_rate_calculation

end NUMINAMATH_CALUDE_paving_rate_calculation_l2501_250170


namespace NUMINAMATH_CALUDE_max_perfect_matchings_20gon_l2501_250178

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Maximum number of perfect matchings for a 2n-gon -/
def max_perfect_matchings (n : ℕ) : ℕ := fib n

/-- A convex polygon -/
structure ConvexPolygon where
  sides : ℕ
  convex : sides > 2

/-- A triangulation of a convex polygon -/
structure Triangulation (p : ConvexPolygon) where
  diagonals : ℕ
  triangles : ℕ
  valid : diagonals = p.sides - 3 ∧ triangles = p.sides - 2

/-- A perfect matching in a triangulation -/
structure PerfectMatching (t : Triangulation p) where
  edges : ℕ
  valid : edges = p.sides / 2

/-- Theorem: Maximum number of perfect matchings for a 20-gon -/
theorem max_perfect_matchings_20gon (p : ConvexPolygon) 
    (h : p.sides = 20) : 
    (∀ t : Triangulation p, ∀ m : PerfectMatching t, 
      ∃ n : ℕ, n ≤ max_perfect_matchings 10) ∧ 
    (∃ t : Triangulation p, ∃ m : PerfectMatching t, 
      max_perfect_matchings 10 = 55) :=
  sorry

end NUMINAMATH_CALUDE_max_perfect_matchings_20gon_l2501_250178


namespace NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l2501_250188

/-- Given that 4 pages cost 8 cents, prove that $15 allows copying 750 pages -/
theorem pages_copied_for_fifteen_dollars (cost_per_four_pages : ℚ) 
  (h1 : cost_per_four_pages = 8/100) : 
  (15 : ℚ) / (cost_per_four_pages / 4) = 750 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l2501_250188


namespace NUMINAMATH_CALUDE_kanul_spending_l2501_250195

/-- The total amount Kanul had initially -/
def T : ℝ := 5714.29

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 3000

/-- The amount spent on machinery -/
def machinery : ℝ := 1000

/-- The fraction of the total amount spent as cash -/
def cash_fraction : ℝ := 0.30

theorem kanul_spending :
  raw_materials + machinery + cash_fraction * T = T := by sorry

end NUMINAMATH_CALUDE_kanul_spending_l2501_250195


namespace NUMINAMATH_CALUDE_range_of_a_l2501_250182

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then a^x else (4 - a/2)*x + 2

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Icc 4 8 ∧ a ≠ 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2501_250182


namespace NUMINAMATH_CALUDE_light_year_scientific_notation_l2501_250110

def light_year : ℝ := 9500000000000

theorem light_year_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), light_year = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = 12 ∧ a = 9.5 :=
by sorry

end NUMINAMATH_CALUDE_light_year_scientific_notation_l2501_250110


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2501_250113

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) :
  z.im = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2501_250113


namespace NUMINAMATH_CALUDE_exists_even_b_for_odd_n_l2501_250124

def operation (p : ℕ × ℕ) : ℕ × ℕ :=
  if p.1 % 2 = 0 then (p.1 / 2, p.2 + p.1 / 2)
  else (p.1 + p.2 / 2, p.2 / 2)

def applyOperationNTimes (p : ℕ × ℕ) (n : ℕ) : ℕ × ℕ :=
  match n with
  | 0 => p
  | m + 1 => operation (applyOperationNTimes p m)

theorem exists_even_b_for_odd_n (n : ℕ) (h_odd : n % 2 = 1) (h_gt_1 : n > 1) :
  ∃ b : ℕ, b % 2 = 0 ∧ b < n ∧ ∃ k : ℕ, applyOperationNTimes (n, b) k = (b, n) := by
  sorry

end NUMINAMATH_CALUDE_exists_even_b_for_odd_n_l2501_250124


namespace NUMINAMATH_CALUDE_expression_value_l2501_250129

theorem expression_value (x y z : ℝ) : 
  (abs (x - 2) + (y + 3)^2 = 0) → 
  (z = -1) → 
  (2 * (x^2 * y + x * y * z) - 3 * (x^2 * y - x * y * z) - 4 * x^2 * y = 90) :=
by sorry


end NUMINAMATH_CALUDE_expression_value_l2501_250129


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l2501_250155

/-- Represents the revenue from ticket sales -/
def TicketRevenue (f h d : ℕ) (p : ℚ) : ℚ :=
  f * p + h * (p / 2) + d * (2 * p)

theorem charity_ticket_revenue :
  ∃ (f h d : ℕ) (p : ℚ),
    f + h + d = 200 ∧
    TicketRevenue f h d p = 5000 ∧
    f * p = 4500 :=
by sorry

end NUMINAMATH_CALUDE_charity_ticket_revenue_l2501_250155


namespace NUMINAMATH_CALUDE_washers_remaining_l2501_250197

/-- Calculates the number of washers remaining after a plumbing job. -/
theorem washers_remaining (pipe_length : ℕ) (feet_per_bolt : ℕ) (washers_per_bolt : ℕ) (initial_washers : ℕ) : 
  pipe_length = 40 ∧ 
  feet_per_bolt = 5 ∧ 
  washers_per_bolt = 2 ∧ 
  initial_washers = 20 → 
  initial_washers - (pipe_length / feet_per_bolt * washers_per_bolt) = 4 := by
sorry

end NUMINAMATH_CALUDE_washers_remaining_l2501_250197


namespace NUMINAMATH_CALUDE_plane_equation_proof_l2501_250138

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  t : ℝ → Point3D

/-- A plane in 3D space defined by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def lineInPlane (l : Line3D) (plane : Plane) : Prop :=
  ∀ t, pointOnPlane (l.t t) plane

/-- The specific point given in the problem -/
def givenPoint : Point3D :=
  { x := 1, y := -3, z := 6 }

/-- The specific line given in the problem -/
def givenLine : Line3D :=
  { t := λ t => { x := 4*t + 2, y := -t - 1, z := 2*t + 3 } }

/-- The plane we need to prove -/
def resultPlane : Plane :=
  { A := 1, B := -18, C := -7, D := -13 }

theorem plane_equation_proof :
  (pointOnPlane givenPoint resultPlane) ∧
  (lineInPlane givenLine resultPlane) ∧
  (resultPlane.A > 0) ∧
  (Nat.gcd (Nat.gcd (Int.natAbs resultPlane.A) (Int.natAbs resultPlane.B))
           (Nat.gcd (Int.natAbs resultPlane.C) (Int.natAbs resultPlane.D)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l2501_250138


namespace NUMINAMATH_CALUDE_simplify_expression_l2501_250100

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^4 - x^2) / (2*x^2))^2) = (x^4 - x^2 * Real.sqrt 2 * Real.sqrt (x^2) + Real.sqrt 5 * x^2) / (2*x^2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2501_250100


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2501_250153

theorem opposite_of_negative_2023 : -((-2023) : ℤ) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2501_250153


namespace NUMINAMATH_CALUDE_ab_and_a_reciprocal_b_relationship_l2501_250167

theorem ab_and_a_reciprocal_b_relationship (a b : ℝ) (h : a * b ≠ 0) :
  ¬(∀ a b, a * b > 1 → a > 1 / b) ∧ 
  ¬(∀ a b, a > 1 / b → a * b > 1) ∧
  ¬(∀ a b, a * b > 1 ↔ a > 1 / b) :=
by sorry

end NUMINAMATH_CALUDE_ab_and_a_reciprocal_b_relationship_l2501_250167


namespace NUMINAMATH_CALUDE_terrell_hike_distance_l2501_250177

theorem terrell_hike_distance (saturday_distance sunday_distance : ℝ) 
  (h1 : saturday_distance = 8.2)
  (h2 : sunday_distance = 1.6) :
  saturday_distance + sunday_distance = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_terrell_hike_distance_l2501_250177


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2501_250183

theorem quadratic_roots_sum_of_squares (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 8 = 0) → 
  (x₂^2 + 2*x₂ - 8 = 0) → 
  (x₁^2 + x₂^2 = 20) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2501_250183


namespace NUMINAMATH_CALUDE_students_playing_neither_l2501_250130

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 39 →
  football = 26 →
  tennis = 20 →
  both = 17 →
  total - (football + tennis - both) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l2501_250130


namespace NUMINAMATH_CALUDE_exponent_of_5_in_30_factorial_is_7_l2501_250104

/-- The exponent of 5 in the prime factorization of 30! -/
def exponent_of_5_in_30_factorial : ℕ :=
  (30 / 5) + (30 / 25)

/-- Theorem stating that the exponent of 5 in the prime factorization of 30! is 7 -/
theorem exponent_of_5_in_30_factorial_is_7 :
  exponent_of_5_in_30_factorial = 7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_of_5_in_30_factorial_is_7_l2501_250104


namespace NUMINAMATH_CALUDE_cubic_root_sum_power_l2501_250159

theorem cubic_root_sum_power (p q r t : ℝ) : 
  (p + q + r = 7) → 
  (p * q + q * r + r * p = 8) → 
  (p * q * r = 1) → 
  (t = Real.sqrt p + Real.sqrt q + Real.sqrt r) → 
  t^4 - 14 * t^2 - 8 * t = -18 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_power_l2501_250159


namespace NUMINAMATH_CALUDE_rectangle_area_l2501_250179

/-- 
Given a rectangle with length l and width w, where:
1. The length is four times the width (l = 4w)
2. The perimeter is 200 cm (2l + 2w = 200)
Prove that the area of the rectangle is 1600 square centimeters.
-/
theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2501_250179


namespace NUMINAMATH_CALUDE_james_sold_five_last_week_l2501_250115

/-- The number of chocolate bars sold last week -/
def chocolate_bars_sold_last_week (total : ℕ) (sold_this_week : ℕ) (need_to_sell : ℕ) : ℕ :=
  total - (sold_this_week + need_to_sell)

/-- Theorem stating that James sold 5 chocolate bars last week -/
theorem james_sold_five_last_week :
  chocolate_bars_sold_last_week 18 7 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_sold_five_last_week_l2501_250115


namespace NUMINAMATH_CALUDE_intersection_when_a_is_2_subset_range_l2501_250122

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 3*(a+1)*x + 2*(3*a+1) < 0}
def B (a : ℝ) : Set ℝ := {x | (x-2*a) / (x-(a^2+1)) < 0}

-- Theorem for part (1)
theorem intersection_when_a_is_2 : A 2 ∩ B 2 = Set.Ioo 4 5 := by sorry

-- Theorem for part (2)
theorem subset_range : {a : ℝ | B a ⊆ A a} = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_2_subset_range_l2501_250122


namespace NUMINAMATH_CALUDE_andrews_age_l2501_250164

theorem andrews_age (carlos_age bella_age andrew_age : ℕ) : 
  carlos_age = 20 →
  bella_age = carlos_age + 4 →
  andrew_age = bella_age - 5 →
  andrew_age = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_andrews_age_l2501_250164


namespace NUMINAMATH_CALUDE_integer_product_condition_l2501_250194

theorem integer_product_condition (a : ℚ) : 
  (∀ n : ℕ, ∃ k : ℤ, a * n * (n + 2) * (n + 3) * (n + 4) = k) ↔ 
  (∃ k : ℤ, a = k / 6) :=
sorry

end NUMINAMATH_CALUDE_integer_product_condition_l2501_250194


namespace NUMINAMATH_CALUDE_cake_change_calculation_l2501_250158

/-- Calculates the change received when buying cake slices -/
theorem cake_change_calculation (single_price double_price single_quantity double_quantity payment : ℕ) :
  single_price = 4 →
  double_price = 7 →
  single_quantity = 7 →
  double_quantity = 5 →
  payment = 100 →
  payment - (single_price * single_quantity + double_price * double_quantity) = 37 := by
  sorry

#check cake_change_calculation

end NUMINAMATH_CALUDE_cake_change_calculation_l2501_250158


namespace NUMINAMATH_CALUDE_tank_weight_l2501_250162

/-- Proves that the weight of a water tank filled to 80% capacity is 1360 pounds. -/
theorem tank_weight (tank_capacity : ℝ) (empty_tank_weight : ℝ) (water_weight_per_gallon : ℝ) :
  tank_capacity = 200 →
  empty_tank_weight = 80 →
  water_weight_per_gallon = 8 →
  empty_tank_weight + 0.8 * tank_capacity * water_weight_per_gallon = 1360 := by
  sorry

end NUMINAMATH_CALUDE_tank_weight_l2501_250162


namespace NUMINAMATH_CALUDE_museum_paintings_l2501_250168

theorem museum_paintings (initial : ℕ) (removed : ℕ) (remaining : ℕ) : 
  initial = 98 → removed = 3 → remaining = initial - removed → remaining = 95 := by
sorry

end NUMINAMATH_CALUDE_museum_paintings_l2501_250168


namespace NUMINAMATH_CALUDE_root_in_interval_l2501_250174

def f (x : ℝ) := x^5 + x - 3

theorem root_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2501_250174


namespace NUMINAMATH_CALUDE_remainder_problem_l2501_250185

theorem remainder_problem (L S R : ℕ) : 
  L - S = 2395 → 
  S = 476 → 
  L = 6 * S + R → 
  R < S → 
  R = 15 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2501_250185


namespace NUMINAMATH_CALUDE_sphere_with_cylindrical_hole_volume_l2501_250123

theorem sphere_with_cylindrical_hole_volume :
  let R : ℝ := Real.sqrt 3
  let sphere_volume := (4 / 3) * Real.pi * R^3
  let cylinder_radius := R / 2
  let cylinder_height := R * Real.sqrt 3
  let cylinder_volume := Real.pi * cylinder_radius^2 * cylinder_height
  let spherical_cap_height := R * (2 - Real.sqrt 3) / 2
  let spherical_cap_volume := (Real.pi * spherical_cap_height^2 * (3 * R - spherical_cap_height)) / 3
  let remaining_volume := sphere_volume - cylinder_volume - 2 * spherical_cap_volume
  remaining_volume = (9 * Real.pi) / 2 := by
sorry


end NUMINAMATH_CALUDE_sphere_with_cylindrical_hole_volume_l2501_250123


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l2501_250119

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l2501_250119


namespace NUMINAMATH_CALUDE_hotel_operations_cost_l2501_250184

/-- Proves that the total cost of operations is $100 given the specified conditions --/
theorem hotel_operations_cost (cost : ℝ) (payments : ℝ) (loss : ℝ) : 
  payments = (3/4) * cost → 
  loss = 25 → 
  payments + loss = cost → 
  cost = 100 := by
  sorry

end NUMINAMATH_CALUDE_hotel_operations_cost_l2501_250184


namespace NUMINAMATH_CALUDE_intersection_theorem_l2501_250128

def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | |x| > 2}

theorem intersection_theorem : A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2501_250128


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2501_250135

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2501_250135


namespace NUMINAMATH_CALUDE_unique_values_l2501_250187

/-- The polynomial we're working with -/
def f (p q : ℤ) (x : ℝ) : ℝ := x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 8

/-- The condition that the polynomial is divisible by (x + 2)(x - 1) -/
def is_divisible (p q : ℤ) : Prop :=
  ∀ x : ℝ, (x + 2 = 0 ∨ x - 1 = 0) → f p q x = 0

/-- The theorem stating that p = -54 and q = -48 are the unique values satisfying the condition -/
theorem unique_values :
  ∃! (p q : ℤ), is_divisible p q ∧ p = -54 ∧ q = -48 := by sorry

end NUMINAMATH_CALUDE_unique_values_l2501_250187


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2501_250145

theorem min_value_quadratic (x : ℝ) : 
  (∀ x, x^2 + 6*x ≥ -9) ∧ (∃ x, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2501_250145


namespace NUMINAMATH_CALUDE_expression_evaluation_l2501_250102

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2501_250102


namespace NUMINAMATH_CALUDE_combine_expression_l2501_250157

theorem combine_expression (a b : ℝ) : 3 * (2 * a - 3 * b) - 6 * (a - b) = -3 * b := by
  sorry

end NUMINAMATH_CALUDE_combine_expression_l2501_250157


namespace NUMINAMATH_CALUDE_range_of_m_plus_n_l2501_250175

/-- The function f(x) = x^2 + nx + m -/
def f (n m x : ℝ) : ℝ := x^2 + n*x + m

/-- The set of roots of f -/
def roots (n m : ℝ) : Set ℝ := {x | f n m x = 0}

/-- The set of roots of f(f(x)) -/
def roots_of_f_of_f (n m : ℝ) : Set ℝ := {x | f n m (f n m x) = 0}

theorem range_of_m_plus_n (n m : ℝ) :
  roots n m = roots_of_f_of_f n m ∧ roots n m ≠ ∅ → 0 < m + n ∧ m + n < 4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_plus_n_l2501_250175


namespace NUMINAMATH_CALUDE_sunny_cakes_l2501_250120

/-- Given that Sunny gives away 2 cakes, puts 6 candles on each remaining cake,
    and uses a total of 36 candles, prove that she initially baked 8 cakes. -/
theorem sunny_cakes (cakes_given_away : ℕ) (candles_per_cake : ℕ) (total_candles : ℕ) :
  cakes_given_away = 2 →
  candles_per_cake = 6 →
  total_candles = 36 →
  cakes_given_away + (total_candles / candles_per_cake) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sunny_cakes_l2501_250120


namespace NUMINAMATH_CALUDE_physics_marks_l2501_250140

theorem physics_marks (P C M : ℝ) 
  (avg_all : (P + C + M) / 3 = 65)
  (avg_PM : (P + M) / 2 = 90)
  (avg_PC : (P + C) / 2 = 70) :
  P = 125 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l2501_250140


namespace NUMINAMATH_CALUDE_alyssa_total_spending_l2501_250118

/-- Calculates the total cost of Alyssa's toy shopping, including discount and tax --/
def total_cost (football_price teddy_bear_price crayons_price puzzle_price doll_price : ℚ)
  (teddy_bear_discount : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let discounted_teddy_bear := teddy_bear_price * (1 - teddy_bear_discount)
  let subtotal := football_price + discounted_teddy_bear + crayons_price + puzzle_price + doll_price
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax

/-- Theorem stating that Alyssa's total spending matches the calculated amount --/
theorem alyssa_total_spending :
  total_cost 12.99 15.35 4.65 7.85 14.50 0.15 0.08 = 57.23 :=
by sorry

end NUMINAMATH_CALUDE_alyssa_total_spending_l2501_250118


namespace NUMINAMATH_CALUDE_positive_real_solution_l2501_250109

theorem positive_real_solution (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 - b*d)/(b + 2*c + d) + (b^2 - c*a)/(c + 2*d + a) + 
  (c^2 - d*b)/(d + 2*a + b) + (d^2 - a*c)/(a + 2*b + c) = 0 →
  a = c ∧ b = d := by
sorry

end NUMINAMATH_CALUDE_positive_real_solution_l2501_250109


namespace NUMINAMATH_CALUDE_angle_B_60_iff_arithmetic_progression_l2501_250141

theorem angle_B_60_iff_arithmetic_progression (A B C : ℝ) : 
  (A + B + C = 180) →  -- Sum of angles in a triangle is 180°
  (B = 60 ↔ ∃ d : ℝ, A = B - d ∧ C = B + d) :=
sorry

end NUMINAMATH_CALUDE_angle_B_60_iff_arithmetic_progression_l2501_250141


namespace NUMINAMATH_CALUDE_age_sum_product_total_l2501_250149

theorem age_sum_product_total (elvie_age arielle_age : ℕ) : 
  elvie_age = 10 → arielle_age = 11 → 
  (elvie_age + arielle_age) + (elvie_age * arielle_age) = 131 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_product_total_l2501_250149


namespace NUMINAMATH_CALUDE_typing_difference_is_1200_l2501_250143

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Micah's typing speed in words per minute -/
def micah_speed : ℕ := 20

/-- Isaiah's typing speed in words per minute -/
def isaiah_speed : ℕ := 40

/-- The difference in words typed per hour between Isaiah and Micah -/
def typing_difference : ℕ := isaiah_speed * minutes_per_hour - micah_speed * minutes_per_hour

theorem typing_difference_is_1200 : typing_difference = 1200 := by
  sorry

end NUMINAMATH_CALUDE_typing_difference_is_1200_l2501_250143


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l2501_250191

theorem sum_of_special_primes_is_prime (A B : ℕ+) : 
  Nat.Prime A ∧ 
  Nat.Prime B ∧ 
  Nat.Prime (A - B) ∧ 
  Nat.Prime (A + B) → 
  Nat.Prime (A + B + (A - B) + A + B) :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l2501_250191


namespace NUMINAMATH_CALUDE_fib_75_mod_9_l2501_250144

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_75_mod_9 : fib 74 % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fib_75_mod_9_l2501_250144


namespace NUMINAMATH_CALUDE_cubic_function_min_value_l2501_250198

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + c

-- State the theorem
theorem cubic_function_min_value 
  (c : ℝ) 
  (h_max : ∃ x, f c x ≤ 28 ∧ ∀ y, f c y ≤ f c x) : 
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f c x ≤ f c y) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f c x = -4) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_min_value_l2501_250198


namespace NUMINAMATH_CALUDE_expression_evaluation_l2501_250137

theorem expression_evaluation (a b : ℤ) (h1 : a = 2) (h2 : b = -1) :
  ((2*a + 3*b) * (2*a - 3*b) - (2*a - b)^2 - 3*a*b) / (-b) = -12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2501_250137


namespace NUMINAMATH_CALUDE_rectangle_b_product_l2501_250161

theorem rectangle_b_product : 
  ∀ (b₁ b₂ : ℝ),
  (∃ (x y : ℝ → ℝ),
    (∀ t, y t = 3 ∨ y t = 7 ∨ x t = -1 ∨ x t = b₁ ∨ x t = b₂) ∧
    (∃ (x₁ x₂ y₁ y₂ : ℝ), 
      y₁ = 3 ∧ y₂ = 7 ∧ x₁ = -1 ∧ (x₂ = b₁ ∨ x₂ = b₂) ∧
      x₂ - x₁ = y₂ - y₁)) →
  b₁ * b₂ = -15 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_b_product_l2501_250161


namespace NUMINAMATH_CALUDE_overlap_rectangle_area_l2501_250181

theorem overlap_rectangle_area : 
  let rect1_width : ℝ := 8
  let rect1_height : ℝ := 10
  let rect2_width : ℝ := 9
  let rect2_height : ℝ := 12
  let overlap_area : ℝ := 37
  let rect1_area : ℝ := rect1_width * rect1_height
  let rect2_area : ℝ := rect2_width * rect2_height
  let grey_area : ℝ := rect2_area - (rect1_area - overlap_area)
  grey_area = 65 := by
sorry

end NUMINAMATH_CALUDE_overlap_rectangle_area_l2501_250181


namespace NUMINAMATH_CALUDE_shoe_selection_theorem_l2501_250154

theorem shoe_selection_theorem (n : ℕ) (m : ℕ) (h : n = 5 ∧ m = 4) :
  (Nat.choose n 1) * (Nat.choose (n - 1) (m - 2)) * (Nat.choose 2 1) * (Nat.choose 2 1) = 120 :=
sorry

end NUMINAMATH_CALUDE_shoe_selection_theorem_l2501_250154
