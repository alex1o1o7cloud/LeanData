import Mathlib

namespace f_max_value_l3592_359239

/-- A function f(x) that is symmetric about x = -1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := -x^2 * (x^2 + a*x + b)

/-- Symmetry condition: f(x) = f(-2-x) for all x -/
def is_symmetric (a b : ℝ) : Prop := ∀ x, f a b x = f a b (-2-x)

/-- The maximum value of f(x) is 0 -/
theorem f_max_value (a b : ℝ) (h : is_symmetric a b) : 
  ∃ x₀, ∀ x, f a b x ≤ f a b x₀ ∧ f a b x₀ = 0 := by
  sorry

#check f_max_value

end f_max_value_l3592_359239


namespace min_value_ab_l3592_359203

theorem min_value_ab (a b : ℝ) (h : (4 / a) + (1 / b) = Real.sqrt (a * b)) : a * b ≥ 4 := by
  sorry

end min_value_ab_l3592_359203


namespace john_weight_is_250_l3592_359275

/-- The weight bench capacity in pounds -/
def bench_capacity : ℝ := 1000

/-- The safety margin percentage -/
def safety_margin : ℝ := 0.20

/-- The weight John puts on the bar in pounds -/
def bar_weight : ℝ := 550

/-- John's weight in pounds -/
def john_weight : ℝ := bench_capacity * (1 - safety_margin) - bar_weight

theorem john_weight_is_250 : john_weight = 250 := by
  sorry

end john_weight_is_250_l3592_359275


namespace rachel_essay_time_l3592_359285

/-- Calculates the total time spent on an essay given the writing speed, number of pages, research time, and editing time. -/
def total_essay_time (writing_speed : ℝ) (pages : ℕ) (research_time : ℝ) (editing_time : ℝ) : ℝ :=
  (pages : ℝ) * writing_speed + research_time + editing_time

/-- Proves that Rachel spent 5 hours on her essay given the conditions. -/
theorem rachel_essay_time : 
  let writing_speed : ℝ := 30  -- minutes per page
  let pages : ℕ := 6
  let research_time : ℝ := 45  -- minutes
  let editing_time : ℝ := 75   -- minutes
  total_essay_time writing_speed pages research_time editing_time / 60 = 5 := by
sorry


end rachel_essay_time_l3592_359285


namespace intersection_of_A_and_B_l3592_359271

def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l3592_359271


namespace gcd_of_powers_minus_one_l3592_359249

theorem gcd_of_powers_minus_one :
  Nat.gcd (2^2100 - 1) (2^1950 - 1) = 2^150 - 1 := by
  sorry

end gcd_of_powers_minus_one_l3592_359249


namespace polynomial_bound_l3592_359295

theorem polynomial_bound (a b c : ℝ) :
  (∀ x : ℝ, abs x ≤ 1 → abs (a * x^2 + b * x + c) ≤ 1) →
  (∀ x : ℝ, abs x ≤ 1 → abs (c * x^2 + b * x + a) ≤ 2) :=
by sorry

end polynomial_bound_l3592_359295


namespace andrew_bought_65_planks_l3592_359230

/-- The number of wooden planks Andrew bought initially -/
def total_planks : ℕ :=
  let andrew_bedroom := 8
  let living_room := 20
  let kitchen := 11
  let guest_bedroom := andrew_bedroom - 2
  let hallway := 4
  let num_hallways := 2
  let ruined_per_bedroom := 3
  let num_bedrooms := 2
  let leftover := 6
  andrew_bedroom + living_room + kitchen + guest_bedroom + 
  (hallway * num_hallways) + (ruined_per_bedroom * num_bedrooms) + leftover

/-- Theorem stating that Andrew bought 65 wooden planks initially -/
theorem andrew_bought_65_planks : total_planks = 65 := by
  sorry

end andrew_bought_65_planks_l3592_359230


namespace system_solution_l3592_359225

theorem system_solution :
  ∀ x y z : ℝ,
  (x^2 + y^2 + 25*z^2 = 6*x*z + 8*y*z) ∧
  (3*x^2 + 2*y^2 + z^2 = 240) →
  ((x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2)) :=
by sorry

end system_solution_l3592_359225


namespace A_equals_B_l3592_359297

/-- The number of ways to pair r girls with r boys in town A -/
def A (n r : ℕ) : ℕ := (n.choose r)^2 * r.factorial

/-- The number of ways to pair r girls with r boys in town B -/
def B : ℕ → ℕ → ℕ
| 0, _ => 0
| _, 0 => 1
| n+1, r+1 => (2*n+1 - r) * B n r + B n (r+1)

/-- The theorem stating that A(n,r) equals B(n,r) for all valid n and r -/
theorem A_equals_B (n r : ℕ) (h : r ≤ n) : A n r = B n r := by
  sorry

end A_equals_B_l3592_359297


namespace ratio_a_to_c_l3592_359288

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 7 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 28 := by
sorry

end ratio_a_to_c_l3592_359288


namespace strawberry_division_l3592_359241

def strawberry_problem (brother_baskets : ℕ) (strawberries_per_basket : ℕ) 
  (kimberly_multiplier : ℕ) (parents_difference : ℕ) (family_size : ℕ) : Prop :=
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let kimberly_strawberries := kimberly_multiplier * brother_strawberries
  let parents_strawberries := kimberly_strawberries - parents_difference
  let total_strawberries := kimberly_strawberries + brother_strawberries + parents_strawberries
  (total_strawberries / family_size = 168)

theorem strawberry_division :
  strawberry_problem 3 15 8 93 4 :=
by
  sorry

#check strawberry_division

end strawberry_division_l3592_359241


namespace frank_reading_speed_l3592_359276

/-- Given a book with a certain number of pages and the number of days to read it,
    calculate the number of pages read per day. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

/-- Theorem stating that for a book with 249 pages read in 3 days,
    the number of pages read per day is 83. -/
theorem frank_reading_speed :
  pages_per_day 249 3 = 83 := by
  sorry

end frank_reading_speed_l3592_359276


namespace A_power_2023_l3592_359272

def A : Matrix (Fin 3) (Fin 3) ℚ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_2023 : A^2023 = A := by sorry

end A_power_2023_l3592_359272


namespace disjoint_subsets_sum_theorem_l3592_359258

theorem disjoint_subsets_sum_theorem (S : Set ℕ) (M₁ M₂ M₃ : Set ℕ) 
  (h1 : M₁ ⊆ S) (h2 : M₂ ⊆ S) (h3 : M₃ ⊆ S)
  (h4 : M₁ ∩ M₂ = ∅) (h5 : M₁ ∩ M₃ = ∅) (h6 : M₂ ∩ M₃ = ∅) :
  ∃ (X Y : ℕ), (X ∈ M₁ ∧ Y ∈ M₂) ∨ (X ∈ M₁ ∧ Y ∈ M₃) ∨ (X ∈ M₂ ∧ Y ∈ M₃) ∧ 
    (X + Y ∉ M₁ ∨ X + Y ∉ M₂ ∨ X + Y ∉ M₃) :=
by sorry

end disjoint_subsets_sum_theorem_l3592_359258


namespace regression_satisfies_negative_correlation_l3592_359204

/-- Represents the regression equation for sales volume based on selling price -/
def regression_equation (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the correlation between sales volume and selling price -/
def negative_correlation (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂

theorem regression_satisfies_negative_correlation :
  negative_correlation regression_equation :=
sorry

end regression_satisfies_negative_correlation_l3592_359204


namespace quadratic_roots_condition_l3592_359209

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if both roots of a quadratic equation are greater than 1 -/
def bothRootsGreaterThanOne (eq : QuadraticEquation) : Prop :=
  ∀ x, eq.a * x^2 + eq.b * x + eq.c = 0 → x > 1

/-- The main theorem stating the condition on m -/
theorem quadratic_roots_condition (m : ℝ) :
  let eq : QuadraticEquation := ⟨8, 1 - m, m - 7⟩
  bothRootsGreaterThanOne eq → m ≥ 25 := by
  sorry

#check quadratic_roots_condition

end quadratic_roots_condition_l3592_359209


namespace function_equality_implies_m_equals_one_l3592_359289

/-- Given functions f and g, and a condition on their values at x = -1, 
    prove that the parameter m in g equals 1. -/
theorem function_equality_implies_m_equals_one :
  let f : ℝ → ℝ := λ x ↦ 3 * x^3 - 1/x + 5
  let g : ℝ → ℝ := λ x ↦ 3 * x^2 - m
  f (-1) - g (-1) = 1 →
  m = 1 := by
sorry

end function_equality_implies_m_equals_one_l3592_359289


namespace composite_expression_l3592_359220

theorem composite_expression (x y : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 2022*x^2 + 349*x + 72*x*y + 12*y + 2 = a * b := by
  sorry

end composite_expression_l3592_359220


namespace ab_plus_cd_eq_zero_l3592_359214

/-- Given real numbers a, b, c, d satisfying certain conditions, prove that ab + cd = 0 -/
theorem ab_plus_cd_eq_zero 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*c + b*d = 0) : 
  a*b + c*d = 0 := by
  sorry

end ab_plus_cd_eq_zero_l3592_359214


namespace zeros_in_repeated_nines_square_l3592_359229

/-- The number of zeros in the decimal representation of n^2 -/
def zeros_in_square (n : ℕ) : ℕ := sorry

/-- The number of nines in the decimal representation of n -/
def count_nines (n : ℕ) : ℕ := sorry

theorem zeros_in_repeated_nines_square (n : ℕ) :
  (∀ k ≤ 3, zeros_in_square (10^k - 1) = k - 1) →
  count_nines 999999 = 6 →
  zeros_in_square 999999 = 5 := by sorry

end zeros_in_repeated_nines_square_l3592_359229


namespace hyperbola_equation_theorem_l3592_359200

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  /-- The focal distance of the hyperbola -/
  focal_distance : ℝ
  /-- The distance from a focus to an asymptote -/
  focus_to_asymptote : ℝ
  /-- Assumption that the foci are on the x-axis -/
  foci_on_x_axis : Bool

/-- The equation of a hyperbola given its properties -/
def hyperbola_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 / 3 = 1 ↔ 
    h.focal_distance = 4 ∧ 
    h.focus_to_asymptote = Real.sqrt 3 ∧ 
    h.foci_on_x_axis = true

/-- Theorem stating that a hyperbola with the given properties has the specified equation -/
theorem hyperbola_equation_theorem (h : Hyperbola) :
  h.focal_distance = 4 →
  h.focus_to_asymptote = Real.sqrt 3 →
  h.foci_on_x_axis = true →
  hyperbola_equation h :=
by sorry

end hyperbola_equation_theorem_l3592_359200


namespace probability_of_selecting_A_and_B_l3592_359224

def total_students : ℕ := 5
def students_to_select : ℕ := 3

theorem probability_of_selecting_A_and_B :
  (Nat.choose (total_students - 2) (students_to_select - 2)) / 
  (Nat.choose total_students students_to_select) = 3 / 10 := by
  sorry

end probability_of_selecting_A_and_B_l3592_359224


namespace system_solution_l3592_359292

theorem system_solution (a : ℂ) (x y z : ℝ) (k l : ℤ) :
  Complex.abs (a + 1 / a) = 2 →
  Real.tan x = 1 ∨ Real.tan x = -1 →
  Real.sin y = 1 ∨ Real.sin y = -1 →
  Real.cos z = 0 →
  x = Real.pi / 2 + k * Real.pi ∧
  y = Real.pi / 2 + k * Real.pi ∧
  z = Real.pi / 2 + l * Real.pi :=
by sorry

end system_solution_l3592_359292


namespace cubic_equation_has_real_root_l3592_359274

theorem cubic_equation_has_real_root (a b : ℝ) : 
  ∃ x : ℝ, x^3 + a*x + b = 0 := by sorry

end cubic_equation_has_real_root_l3592_359274


namespace triangle_side_length_l3592_359265

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

end triangle_side_length_l3592_359265


namespace complex_not_purely_imaginary_range_l3592_359279

theorem complex_not_purely_imaginary_range (a : ℝ) : 
  ¬(∃ (y : ℝ), (a^2 - a - 2) + (|a-1| - 1)*I = y*I) → a ≠ -1 :=
by
  sorry

end complex_not_purely_imaginary_range_l3592_359279


namespace arithmetic_sequence_2005_l3592_359246

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_2005 :
  ∃ n : ℕ, arithmetic_sequence 1 3 n = 2005 ∧ n = 669 := by
  sorry

end arithmetic_sequence_2005_l3592_359246


namespace range_of_m_plus_n_l3592_359255

/-- The function f(x) = x^2 + nx + m -/
def f (n m x : ℝ) : ℝ := x^2 + n*x + m

/-- The set of roots of f -/
def roots (n m : ℝ) : Set ℝ := {x | f n m x = 0}

/-- The set of roots of f(f(x)) -/
def roots_of_f_of_f (n m : ℝ) : Set ℝ := {x | f n m (f n m x) = 0}

theorem range_of_m_plus_n (n m : ℝ) :
  roots n m = roots_of_f_of_f n m ∧ roots n m ≠ ∅ → 0 < m + n ∧ m + n < 4 := by sorry

end range_of_m_plus_n_l3592_359255


namespace zeros_in_square_of_999999999_l3592_359219

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

end zeros_in_square_of_999999999_l3592_359219


namespace james_sold_five_last_week_l3592_359233

/-- The number of chocolate bars sold last week -/
def chocolate_bars_sold_last_week (total : ℕ) (sold_this_week : ℕ) (need_to_sell : ℕ) : ℕ :=
  total - (sold_this_week + need_to_sell)

/-- Theorem stating that James sold 5 chocolate bars last week -/
theorem james_sold_five_last_week :
  chocolate_bars_sold_last_week 18 7 6 = 5 := by
  sorry

end james_sold_five_last_week_l3592_359233


namespace problem_solution_l3592_359242

theorem problem_solution (x y : ℝ) 
  (h1 : 2*x + 3*y = 9) 
  (h2 : x*y = -12) : 
  4*x^2 + 9*y^2 = 225 := by
sorry

end problem_solution_l3592_359242


namespace inequality_solution_and_function_property_l3592_359211

def f (x : ℝ) : ℝ := |x + 1|

theorem inequality_solution_and_function_property :
  (∀ x : ℝ, f (x + 8) ≥ 10 - f x ↔ x ≤ -10 ∨ x ≥ 0) ∧
  (∀ x y : ℝ, |x| > 1 → |y| < 1 → f y < |x| * f (y / x^2)) :=
by sorry

end inequality_solution_and_function_property_l3592_359211


namespace point_b_representation_l3592_359280

theorem point_b_representation (a b : ℝ) : 
  a = -2 → (b - a = 3 ∨ a - b = 3) → (b = 1 ∨ b = -5) := by sorry

end point_b_representation_l3592_359280


namespace expression_evaluation_l3592_359277

theorem expression_evaluation : (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 := by
  sorry

end expression_evaluation_l3592_359277


namespace square_root_equation_l3592_359207

theorem square_root_equation (x : ℝ) : Real.sqrt (x + 4) = 12 → x = 140 := by
  sorry

end square_root_equation_l3592_359207


namespace valid_perm_count_l3592_359228

/-- 
Given a permutation π of n distinct elements, we define:
inv_count(π, i) = number of elements to the left of π(i) that are greater than π(i) +
                  number of elements to the right of π(i) that are less than π(i)
-/
def inv_count (π : Fin n → Fin n) (i : Fin n) : ℕ := sorry

/-- A permutation is valid if inv_count is even for all elements -/
def is_valid_perm (π : Fin n → Fin n) : Prop :=
  ∀ i, Even (inv_count π i)

/-- The number of valid permutations -/
def count_valid_perms (n : ℕ) : ℕ := sorry

theorem valid_perm_count (n : ℕ) : 
  count_valid_perms n = (Nat.factorial (n / 2)) * (Nat.factorial ((n + 1) / 2)) := by
  sorry

end valid_perm_count_l3592_359228


namespace sin_pi_12_plus_theta_l3592_359251

theorem sin_pi_12_plus_theta (θ : ℝ) (h : Real.cos ((5 * Real.pi) / 12 - θ) = 1 / 3) :
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end sin_pi_12_plus_theta_l3592_359251


namespace geometric_sequence_seventh_term_l3592_359263

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geometric : IsGeometricSequence a)
  (h_fourth : a 4 = 16)
  (h_ninth : a 9 = 2) :
  a 7 = 8 := by
sorry

end geometric_sequence_seventh_term_l3592_359263


namespace minimum_point_of_translated_graph_l3592_359210

-- Define the original function
def f (x : ℝ) : ℝ := |x + 1| - 2

-- Define the translation
def translate_x : ℝ := 3
def translate_y : ℝ := 4

-- Define the translated function
def g (x : ℝ) : ℝ := f (x - translate_x) + translate_y

-- State the theorem
theorem minimum_point_of_translated_graph :
  ∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = 2 ∧
  ∀ (x : ℝ), g x ≥ g x₀ :=
sorry

end minimum_point_of_translated_graph_l3592_359210


namespace hyperbola_eccentricity_l3592_359291

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if a line through one vertex on the imaginary axis and perpendicular to the y-axis
    forms an equilateral triangle with the other vertex on the imaginary axis and
    the two points where it intersects the hyperbola, then the eccentricity is √10/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let B₁ := (0, b)
  let B₂ := (0, -b)
  let line := fun (x : ℝ) ↦ b
  let P := (-Real.sqrt 2 * a, b)
  let Q := (Real.sqrt 2 * a, b)
  hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2 ∧
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1 - B₂.1)^2 + (P.2 - B₂.2)^2 ∧
  (P.1 - B₂.1)^2 + (P.2 - B₂.2)^2 = (Q.1 - B₂.1)^2 + (Q.2 - B₂.2)^2 →
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 10 / 2 := by
sorry

end hyperbola_eccentricity_l3592_359291


namespace operation_proof_l3592_359293

theorem operation_proof (v : ℝ) : (v - v / 3) - (v - v / 3) / 3 = 12 → v = 27 := by
  sorry

end operation_proof_l3592_359293


namespace rectangle_area_l3592_359283

/-- 
Given a rectangle with length l and width w, where:
1. The length is four times the width (l = 4w)
2. The perimeter is 200 cm (2l + 2w = 200)
Prove that the area of the rectangle is 1600 square centimeters.
-/
theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 := by
sorry

end rectangle_area_l3592_359283


namespace pizza_fraction_eaten_l3592_359290

/-- The fraction of pizza eaten after n trips, where each trip consumes one-third of the remaining pizza -/
def fractionEaten (n : ℕ) : ℚ :=
  1 - (2/3)^n

/-- The number of trips to the refrigerator -/
def numTrips : ℕ := 6

theorem pizza_fraction_eaten :
  fractionEaten numTrips = 364 / 729 := by
  sorry

end pizza_fraction_eaten_l3592_359290


namespace unique_prime_with_prime_quadratics_l3592_359248

theorem unique_prime_with_prime_quadratics :
  ∃! p : ℕ, Nat.Prime p ∧ Nat.Prime (4 * p^2 + 1) ∧ Nat.Prime (6 * p^2 + 1) :=
by
  -- The proof would go here
  sorry

end unique_prime_with_prime_quadratics_l3592_359248


namespace discount_calculation_l3592_359218

/-- Calculates the discount given the cost price, markup percentage, and loss percentage -/
def calculate_discount (cost_price : ℝ) (markup_percentage : ℝ) (loss_percentage : ℝ) : ℝ :=
  let marked_price := cost_price * (1 + markup_percentage)
  let selling_price := cost_price * (1 - loss_percentage)
  marked_price - selling_price

/-- Theorem stating that for a cost price of 100, a markup of 40%, and a loss of 1%, the discount is 41 -/
theorem discount_calculation :
  calculate_discount 100 0.4 0.01 = 41 := by
  sorry


end discount_calculation_l3592_359218


namespace initial_oranges_count_l3592_359294

/-- The number of oranges initially in the basket -/
def initial_oranges : ℕ := sorry

/-- The number of oranges taken from the basket -/
def oranges_taken : ℕ := 5

/-- The number of oranges remaining in the basket -/
def oranges_remaining : ℕ := 3

/-- Theorem stating that the initial number of oranges is 8 -/
theorem initial_oranges_count : initial_oranges = 8 := by sorry

end initial_oranges_count_l3592_359294


namespace profit_ratio_equals_investment_ratio_l3592_359244

/-- The ratio of profits is proportional to the ratio of investments -/
theorem profit_ratio_equals_investment_ratio (p_investment q_investment : ℚ) 
  (hp : p_investment = 50000)
  (hq : q_investment = 66666.67)
  : ∃ (k : ℚ), k * p_investment = 3 ∧ k * q_investment = 4 := by
  sorry

end profit_ratio_equals_investment_ratio_l3592_359244


namespace distance_ratio_theorem_l3592_359247

theorem distance_ratio_theorem (x : ℝ) (h1 : x^2 + (-4)^2 = 8^2) :
  |(-4)| / 8 = 1/2 := by sorry

end distance_ratio_theorem_l3592_359247


namespace lara_flowers_in_vase_l3592_359201

/-- The number of flowers Lara put in the vase -/
def flowers_in_vase (total : ℕ) (to_mom : ℕ) (extra_to_grandma : ℕ) : ℕ :=
  total - (to_mom + (to_mom + extra_to_grandma))

/-- Theorem stating the number of flowers Lara put in the vase -/
theorem lara_flowers_in_vase :
  flowers_in_vase 52 15 6 = 16 := by sorry

end lara_flowers_in_vase_l3592_359201


namespace f_x_plus_5_l3592_359250

def f (x : ℝ) := 3 * x + 1

theorem f_x_plus_5 (x : ℝ) : f (x + 5) = 3 * x + 16 := by
  sorry

end f_x_plus_5_l3592_359250


namespace grassy_plot_width_l3592_359299

/-- Proves that the width of a rectangular grassy plot is 55 meters, given specific conditions -/
theorem grassy_plot_width : 
  ∀ (length width path_width : ℝ) (cost_per_sq_meter cost_total : ℝ),
  length = 110 →
  path_width = 2.5 →
  cost_per_sq_meter = 0.5 →
  cost_total = 425 →
  ((length + 2 * path_width) * (width + 2 * path_width) - length * width) * cost_per_sq_meter = cost_total →
  width = 55 := by
sorry

end grassy_plot_width_l3592_359299


namespace stratified_sampling_l3592_359216

/-- Represents the number of questionnaires for each unit -/
structure Questionnaires :=
  (a b c d : ℕ)

/-- Represents the sample sizes for each unit -/
structure SampleSizes :=
  (a b c d : ℕ)

/-- Checks if the given questionnaire counts form an arithmetic sequence -/
def is_arithmetic_sequence (q : Questionnaires) : Prop :=
  q.b - q.a = q.c - q.b ∧ q.c - q.b = q.d - q.c

/-- The main theorem statement -/
theorem stratified_sampling
  (q : Questionnaires)
  (s : SampleSizes)
  (h1 : q.a + q.b + q.c + q.d = 1000)
  (h2 : is_arithmetic_sequence q)
  (h3 : s.a + s.b + s.c + s.d = 150)
  (h4 : s.b = 30)
  (h5 : s.b * 1000 = q.b * 150) :
  s.d = 60 := by
  sorry

end stratified_sampling_l3592_359216


namespace hyperbola_auxiliary_lines_l3592_359222

/-- Represents a hyperbola with given equation and asymptotes -/
structure Hyperbola where
  a : ℝ
  eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / 16 = 1
  asymptote : ∀ x : ℝ, ∃ y : ℝ, y = 4/3 * x ∨ y = -4/3 * x
  a_pos : a > 0

/-- The auxiliary lines of a hyperbola -/
def auxiliary_lines (h : Hyperbola) : Set ℝ :=
  {x : ℝ | x = 9/5 ∨ x = -9/5}

/-- Theorem stating that the auxiliary lines of the given hyperbola are x = ±9/5 -/
theorem hyperbola_auxiliary_lines (h : Hyperbola) :
  auxiliary_lines h = {x : ℝ | x = 9/5 ∨ x = -9/5} := by
  sorry

end hyperbola_auxiliary_lines_l3592_359222


namespace bens_previous_salary_l3592_359238

/-- Prove that Ben's previous job's annual salary was $75,000 given the conditions of his new job --/
theorem bens_previous_salary (new_base_salary : ℝ) (commission_rate : ℝ) (sale_price : ℝ) (min_sales : ℝ) :
  new_base_salary = 45000 →
  commission_rate = 0.15 →
  sale_price = 750 →
  min_sales = 266.67 →
  ∃ (previous_salary : ℝ), 
    previous_salary ≥ new_base_salary + commission_rate * sale_price * min_sales ∧
    previous_salary < new_base_salary + commission_rate * sale_price * min_sales + 1 :=
by
  sorry

#eval (45000 : ℝ) + 0.15 * 750 * 266.67

end bens_previous_salary_l3592_359238


namespace quadratic_roots_range_l3592_359243

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 4

-- State the theorem
theorem quadratic_roots_range (a b : ℝ) : 
  a > 0 → 
  (∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  (∃ r : ℝ, 1 < r ∧ r < 2 ∧ f a b r = 0) →
  ∀ s : ℝ, s < 4 ↔ ∃ t : ℝ, a + b = t ∧ t < 4 :=
by sorry

end quadratic_roots_range_l3592_359243


namespace shuttle_speed_km_per_hour_l3592_359273

/-- The speed of a space shuttle orbiting the Earth -/
def shuttle_speed_km_per_sec : ℝ := 2

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: The speed of the space shuttle in kilometers per hour -/
theorem shuttle_speed_km_per_hour :
  shuttle_speed_km_per_sec * (seconds_per_hour : ℝ) = 7200 := by
  sorry

end shuttle_speed_km_per_hour_l3592_359273


namespace meaningful_fraction_l3592_359286

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end meaningful_fraction_l3592_359286


namespace range_of_fraction_l3592_359264

def f (a b c x : ℝ) : ℝ := 3 * a * x^2 - 2 * b * x + c

theorem range_of_fraction (a b c : ℝ) :
  (a - b + c = 0) →
  (f a b c 0 > 0) →
  (f a b c 1 > 0) →
  ∃ (y : ℝ), (4/3 < y ∧ y < 7/2 ∧ y = (a + 3*b + 7*c) / (2*a + b)) ∧
  ∀ (z : ℝ), (z = (a + 3*b + 7*c) / (2*a + b)) → (4/3 < z ∧ z < 7/2) :=
sorry

end range_of_fraction_l3592_359264


namespace martha_children_count_l3592_359245

theorem martha_children_count (total_cakes num_cakes_per_child : ℕ) 
  (h1 : total_cakes = 18)
  (h2 : num_cakes_per_child = 6)
  (h3 : total_cakes % num_cakes_per_child = 0) :
  total_cakes / num_cakes_per_child = 3 := by
  sorry

end martha_children_count_l3592_359245


namespace intersection_in_first_quadrant_l3592_359205

/-- The intersection point of two lines is in the first quadrant iff a is in the range (-1, 2) -/
theorem intersection_in_first_quadrant (a : ℝ) :
  (∃ x y : ℝ, ax + y - 4 = 0 ∧ x - y - 2 = 0 ∧ x > 0 ∧ y > 0) ↔ -1 < a ∧ a < 2 := by
  sorry

end intersection_in_first_quadrant_l3592_359205


namespace sin_105_cos_105_l3592_359259

theorem sin_105_cos_105 : Real.sin (105 * π / 180) * Real.cos (105 * π / 180) = -(1/4) := by
  sorry

end sin_105_cos_105_l3592_359259


namespace flood_monitoring_technologies_l3592_359269

-- Define the set of available technologies
inductive GeoTechnology
| RemoteSensing
| GPS
| GIS
| DigitalEarth

-- Define the capabilities of technologies
def canMonitorDisaster (tech : GeoTechnology) : Prop :=
  match tech with
  | GeoTechnology.RemoteSensing => true
  | _ => false

def canManageInfo (tech : GeoTechnology) : Prop :=
  match tech with
  | GeoTechnology.GIS => true
  | _ => false

def isEffectiveForFloodMonitoring (tech : GeoTechnology) : Prop :=
  canMonitorDisaster tech ∨ canManageInfo tech

-- Define the set of effective technologies
def effectiveTechnologies : Set GeoTechnology :=
  {tech | isEffectiveForFloodMonitoring tech}

-- Theorem statement
theorem flood_monitoring_technologies :
  effectiveTechnologies = {GeoTechnology.RemoteSensing, GeoTechnology.GIS} :=
sorry

end flood_monitoring_technologies_l3592_359269


namespace parallel_lines_m_value_l3592_359208

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  ((-1 : ℝ) / (1 + m) = -m / 2) ∧ (m ≠ -2)

/-- The value of m for which the lines x + (1+m)y = 2-m and mx + 2y + 8 = 0 are parallel -/
theorem parallel_lines_m_value : ∃! m : ℝ, parallel_lines m :=
  sorry

end parallel_lines_m_value_l3592_359208


namespace root_in_interval_l3592_359254

def f (x : ℝ) := x^5 + x - 3

theorem root_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f x = 0 := by
  sorry

end root_in_interval_l3592_359254


namespace expression_evaluation_l3592_359266

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) := by
  sorry

end expression_evaluation_l3592_359266


namespace variable_value_l3592_359260

theorem variable_value (w x v : ℝ) 
  (h1 : 5 / w + 5 / x = 5 / v) 
  (h2 : w * x = v)
  (h3 : (w + x) / 2 = 0.5) : 
  v = 0.25 := by
sorry

end variable_value_l3592_359260


namespace polynomial_value_equals_one_l3592_359298

theorem polynomial_value_equals_one (x y : ℝ) (h : x + y = -1) :
  x^4 + 5*x^3*y + x^2*y + 8*x^2*y^2 + x*y^2 + 5*x*y^3 + y^4 = 1 := by
  sorry

end polynomial_value_equals_one_l3592_359298


namespace simplify_square_roots_l3592_359256

theorem simplify_square_roots : 
  (Real.sqrt 648 / Real.sqrt 81) - (Real.sqrt 245 / Real.sqrt 49) = 2 * Real.sqrt 2 - Real.sqrt 5 := by
  sorry

end simplify_square_roots_l3592_359256


namespace vertical_angles_are_congruent_l3592_359257

-- Define what it means for two angles to be vertical
def are_vertical_angles (α β : Angle) : Prop := sorry

-- Define what it means for two angles to be congruent
def are_congruent (α β : Angle) : Prop := sorry

-- Theorem statement
theorem vertical_angles_are_congruent (α β : Angle) : 
  are_vertical_angles α β → are_congruent α β := by
  sorry

end vertical_angles_are_congruent_l3592_359257


namespace don_profit_l3592_359227

/-- Represents a person's rose bundle transaction -/
structure Transaction where
  bought : ℕ
  sold : ℕ
  profit : ℚ

/-- Represents the prices of rose bundles -/
structure Prices where
  buy : ℚ
  sell : ℚ

/-- The main theorem -/
theorem don_profit 
  (jamie : Transaction)
  (linda : Transaction)
  (don : Transaction)
  (prices : Prices)
  (h1 : jamie.bought = 20)
  (h2 : jamie.sold = 15)
  (h3 : jamie.profit = 60)
  (h4 : linda.bought = 34)
  (h5 : linda.sold = 24)
  (h6 : linda.profit = 69)
  (h7 : don.bought = 40)
  (h8 : don.sold = 36)
  (h9 : prices.sell > prices.buy)
  (h10 : jamie.profit = jamie.sold * prices.sell - jamie.bought * prices.buy)
  (h11 : linda.profit = linda.sold * prices.sell - linda.bought * prices.buy)
  (h12 : don.profit = don.sold * prices.sell - don.bought * prices.buy) :
  don.profit = 252 := by sorry

end don_profit_l3592_359227


namespace no_covalent_bond_IA_VIIA_l3592_359202

/-- Represents an element in the periodic table -/
structure Element where
  group : ℕ
  isHydrogen : Bool

/-- Represents the bonding behavior of an element -/
inductive BondingBehavior
  | LoseElectrons
  | GainElectrons

/-- Determines the bonding behavior of an element based on its group -/
def bondingBehavior (e : Element) : BondingBehavior :=
  if e.group = 1 ∧ ¬e.isHydrogen then BondingBehavior.LoseElectrons
  else if e.group = 17 then BondingBehavior.GainElectrons
  else BondingBehavior.LoseElectrons  -- Default case, not relevant for this problem

/-- Determines if two elements can form a covalent bond -/
def canFormCovalentBond (e1 e2 : Element) : Prop :=
  bondingBehavior e1 = bondingBehavior e2

/-- Theorem stating that elements in Group IA (except H) and Group VIIA cannot form covalent bonds -/
theorem no_covalent_bond_IA_VIIA :
  ∀ (e1 e2 : Element),
    ((e1.group = 1 ∧ ¬e1.isHydrogen) ∨ e1.group = 17) →
    ((e2.group = 1 ∧ ¬e2.isHydrogen) ∨ e2.group = 17) →
    ¬(canFormCovalentBond e1 e2) :=
by
  sorry

end no_covalent_bond_IA_VIIA_l3592_359202


namespace animus_tower_beavers_l3592_359270

/-- The number of beavers hired for the Animus Tower project -/
def num_beavers : ℕ := 862 - 544

/-- The total number of workers hired for the Animus Tower project -/
def total_workers : ℕ := 862

/-- The number of spiders hired for the Animus Tower project -/
def num_spiders : ℕ := 544

theorem animus_tower_beavers : num_beavers = 318 := by
  sorry

end animus_tower_beavers_l3592_359270


namespace xyz_sum_value_l3592_359281

theorem xyz_sum_value (x y z : ℝ) 
  (h1 : x^2 - y*z = 2) 
  (h2 : y^2 - z*x = 2) 
  (h3 : z^2 - x*y = 2) : 
  x*y + y*z + z*x = -2 := by
sorry

end xyz_sum_value_l3592_359281


namespace four_digit_numbers_with_specific_remainders_l3592_359284

theorem four_digit_numbers_with_specific_remainders :
  ∀ N : ℕ,
  (1000 ≤ N ∧ N ≤ 9999) →
  (N % 2 = 0 ∧ N % 3 = 1 ∧ N % 5 = 3 ∧ N % 7 = 5 ∧ N % 11 = 9) →
  (N = 2308 ∨ N = 4618 ∨ N = 6928 ∨ N = 9238) :=
by sorry

end four_digit_numbers_with_specific_remainders_l3592_359284


namespace sum_floor_equals_n_l3592_359237

/-- For any natural number n, the sum of floor((n + 2^k) / 2^(k+1)) from k = 0 to infinity equals n -/
theorem sum_floor_equals_n (n : ℕ) :
  (∑' k : ℕ, ⌊(n + 2^k : ℝ) / (2^(k+1) : ℝ)⌋) = n :=
sorry

end sum_floor_equals_n_l3592_359237


namespace square_perimeter_problem_l3592_359240

theorem square_perimeter_problem (area_A : ℝ) (prob_not_in_B : ℝ) : 
  area_A = 30 →
  prob_not_in_B = 0.4666666666666667 →
  let area_B := area_A * (1 - prob_not_in_B)
  let side_B := Real.sqrt area_B
  let perimeter_B := 4 * side_B
  perimeter_B = 16 := by
sorry

end square_perimeter_problem_l3592_359240


namespace some_number_divisibility_l3592_359221

theorem some_number_divisibility (x : ℕ) : (1425 * x * 1429) % 12 = 3 ↔ x % 4 = 3 := by
  sorry

end some_number_divisibility_l3592_359221


namespace typing_sequences_count_l3592_359232

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

end typing_sequences_count_l3592_359232


namespace arrangements_A_not_head_B_not_tail_arrangements_at_least_one_between_A_B_arrangements_A_B_together_C_D_not_together_l3592_359217

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

end arrangements_A_not_head_B_not_tail_arrangements_at_least_one_between_A_B_arrangements_A_B_together_C_D_not_together_l3592_359217


namespace right_triangle_ratio_l3592_359278

theorem right_triangle_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (right_triangle : a^2 + b^2 = c^2) (leg : b = 5) (hyp : c = 13) :
  a / c = 12 / 13 := by
sorry

end right_triangle_ratio_l3592_359278


namespace tim_apartment_complexes_l3592_359252

/-- The number of keys Tim needs to make -/
def total_keys : ℕ := 72

/-- The number of keys needed per lock -/
def keys_per_lock : ℕ := 3

/-- The number of apartments in each complex -/
def apartments_per_complex : ℕ := 12

/-- The number of apartment complexes Tim owns -/
def num_complexes : ℕ := total_keys / keys_per_lock / apartments_per_complex

theorem tim_apartment_complexes : num_complexes = 2 := by
  sorry

end tim_apartment_complexes_l3592_359252


namespace geometric_progression_ratio_l3592_359235

theorem geometric_progression_ratio (a : ℝ) (r : ℝ) :
  a > 0 ∧ r > 0 ∧ 
  (∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) →
  r = 1/2 := by
sorry

end geometric_progression_ratio_l3592_359235


namespace parade_function_correct_l3592_359231

/-- Represents the function relationship between row number and number of people in a trapezoidal parade. -/
def parade_function (x : ℤ) : ℤ := x + 39

/-- Theorem stating the correctness of the parade function for a specific trapezoidal parade configuration. -/
theorem parade_function_correct :
  ∀ x : ℤ, 1 ≤ x → x ≤ 60 →
  (parade_function x = 40 + (x - 1)) ∧
  (parade_function 1 = 40) ∧
  (∀ i : ℤ, 1 ≤ i → i < 60 → parade_function (i + 1) = parade_function i + 1) :=
by sorry

end parade_function_correct_l3592_359231


namespace maria_paint_cans_l3592_359223

/-- Represents the paint situation for Maria's room painting problem -/
structure PaintSituation where
  initialRooms : ℕ
  finalRooms : ℕ
  lostCans : ℕ

/-- Calculates the number of cans used for the final number of rooms -/
def cansUsed (s : PaintSituation) : ℕ :=
  s.finalRooms / ((s.initialRooms - s.finalRooms) / s.lostCans)

/-- Theorem stating that for Maria's specific situation, 16 cans were used -/
theorem maria_paint_cans :
  let s : PaintSituation := { initialRooms := 40, finalRooms := 32, lostCans := 4 }
  cansUsed s = 16 := by sorry

end maria_paint_cans_l3592_359223


namespace max_perfect_matchings_20gon_l3592_359282

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

end max_perfect_matchings_20gon_l3592_359282


namespace cos_pi_half_minus_A_l3592_359215

theorem cos_pi_half_minus_A (A : ℝ) (h : Real.sin (π - A) = 1/2) : 
  Real.cos (π/2 - A) = 1/2 := by
  sorry

end cos_pi_half_minus_A_l3592_359215


namespace grey_eyed_black_hair_count_l3592_359253

/-- Represents the number of students with a specific hair color and eye color combination -/
structure StudentCount where
  redHairGreenEyes : ℕ
  redHairGreyEyes : ℕ
  blackHairGreenEyes : ℕ
  blackHairGreyEyes : ℕ

/-- Theorem stating the number of grey-eyed students with black hair -/
theorem grey_eyed_black_hair_count (s : StudentCount) : s.blackHairGreyEyes = 20 :=
  by
  have total_students : s.redHairGreenEyes + s.redHairGreyEyes + s.blackHairGreenEyes + s.blackHairGreyEyes = 60 := by sorry
  have green_eyed_red_hair : s.redHairGreenEyes = 20 := by sorry
  have black_hair_total : s.blackHairGreenEyes + s.blackHairGreyEyes = 36 := by sorry
  have grey_eyes_total : s.redHairGreyEyes + s.blackHairGreyEyes = 24 := by sorry
  sorry

#check grey_eyed_black_hair_count

end grey_eyed_black_hair_count_l3592_359253


namespace find_second_number_l3592_359262

/-- Given two positive integers with known HCF and LCM, find the second number -/
theorem find_second_number (A B : ℕ+) (h1 : A = 24) 
  (h2 : Nat.gcd A B = 13) (h3 : Nat.lcm A B = 312) : B = 169 := by
  sorry

end find_second_number_l3592_359262


namespace train_platform_crossing_time_l3592_359267

/-- Given a train and platform with specific lengths, prove the time taken to cross the platform -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (signal_pole_crossing_time : ℝ) 
  (h1 : train_length = 600)
  (h2 : platform_length = 700)
  (h3 : signal_pole_crossing_time = 18) :
  (train_length + platform_length) / (train_length / signal_pole_crossing_time) = 39 := by
  sorry

#check train_platform_crossing_time

end train_platform_crossing_time_l3592_359267


namespace cost_for_36_people_l3592_359296

/-- The cost to feed a group of people with chicken combos -/
def cost_to_feed (people : ℕ) (combo_cost : ℚ) (people_per_combo : ℕ) : ℚ :=
  (people / people_per_combo : ℚ) * combo_cost

/-- Theorem: The cost to feed 36 people is $72.00 -/
theorem cost_for_36_people :
  cost_to_feed 36 12 6 = 72 := by
  sorry

end cost_for_36_people_l3592_359296


namespace abc_zero_l3592_359206

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 := by
sorry

end abc_zero_l3592_359206


namespace walking_cycling_speeds_l3592_359234

/-- Proves that given conditions result in specific walking and cycling speeds -/
theorem walking_cycling_speeds (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 2 →
  speed_ratio = 4 →
  time_difference = 1/3 →
  ∃ (walking_speed cycling_speed : ℝ),
    walking_speed = 4.5 ∧
    cycling_speed = 18 ∧
    cycling_speed = speed_ratio * walking_speed ∧
    distance / walking_speed - distance / cycling_speed = time_difference :=
by sorry

end walking_cycling_speeds_l3592_359234


namespace max_value_implies_a_l3592_359287

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -4 * x^2 + 4 * a * x - 4 * a - a^2

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ -5) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = -5) →
  a = 5/4 ∨ a = -5 :=
by sorry

end max_value_implies_a_l3592_359287


namespace remainder_divisibility_l3592_359268

theorem remainder_divisibility (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 30 = 18 := by
  sorry

end remainder_divisibility_l3592_359268


namespace average_shift_l3592_359261

theorem average_shift (x₁ x₂ x₃ x₄ : ℝ) :
  (x₁ + x₂ + x₃ + x₄) / 4 = 2 →
  ((x₁ + 3) + (x₂ + 3) + (x₃ + 3) + (x₄ + 3)) / 4 = 5 := by
sorry

end average_shift_l3592_359261


namespace polygon_angles_l3592_359226

theorem polygon_angles (n : ℕ) : 
  (n - 2) * 180 = 5 * 360 → n = 12 := by
sorry

end polygon_angles_l3592_359226


namespace waiter_customer_count_l3592_359236

theorem waiter_customer_count (initial : Float) (lunch_rush : Float) (later : Float) :
  initial = 29.0 →
  lunch_rush = 20.0 →
  later = 34.0 →
  initial + lunch_rush + later = 83.0 := by
  sorry

end waiter_customer_count_l3592_359236


namespace angle_two_pi_third_in_second_quadrant_l3592_359213

/-- The angle 2π/3 is in the second quadrant -/
theorem angle_two_pi_third_in_second_quadrant :
  let θ : Real := 2 * Real.pi / 3
  0 < θ ∧ θ < Real.pi / 2 → False
  ∧ Real.pi / 2 < θ ∧ θ ≤ Real.pi → True
  ∧ Real.pi < θ ∧ θ ≤ 3 * Real.pi / 2 → False
  ∧ 3 * Real.pi / 2 < θ ∧ θ < 2 * Real.pi → False :=
by sorry

end angle_two_pi_third_in_second_quadrant_l3592_359213


namespace senior_junior_ratio_l3592_359212

theorem senior_junior_ratio (j s : ℕ) (hj : j > 0) (hs : s > 0) : 
  (3 * j : ℚ) / 4 = (1 * s : ℚ) / 2 → (s : ℚ) / j = 3 / 2 :=
by sorry

end senior_junior_ratio_l3592_359212
