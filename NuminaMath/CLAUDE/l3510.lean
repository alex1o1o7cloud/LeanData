import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l3510_351017

theorem problem_statement (x : ℝ) : 
  (0.4 * 60 = (4/5) * x + 4) → x = 25 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3510_351017


namespace NUMINAMATH_CALUDE_equation_solution_l3510_351093

theorem equation_solution : 
  ∀ x : ℂ, (13*x - x^2)/(x + 1) * (x + (13 - x)/(x + 1)) = 54 ↔ 
  x = 3 ∨ x = 6 ∨ x = (5 + Complex.I * Real.sqrt 11)/2 ∨ x = (5 - Complex.I * Real.sqrt 11)/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3510_351093


namespace NUMINAMATH_CALUDE_hyperbola_foci_l3510_351073

/-- Given a hyperbola with equation x²/4 - y² = 1, prove that its foci are at (±√5, 0) -/
theorem hyperbola_foci (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → (∃ (s : ℝ), s^2 = 5 ∧ ((x = s ∨ x = -s) ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l3510_351073


namespace NUMINAMATH_CALUDE_triangle_area_solution_l3510_351065

/-- Given a triangle with vertices (0, 0), (x, 3x), and (x, 0), where x > 0,
    if the area of this triangle is 100 square units, then x = 10√6/3 -/
theorem triangle_area_solution (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 100 → x = (10 * Real.sqrt 6) / 3 := by
  sorry

#check triangle_area_solution

end NUMINAMATH_CALUDE_triangle_area_solution_l3510_351065


namespace NUMINAMATH_CALUDE_roses_per_bush_calculation_l3510_351070

/-- The number of rose petals needed to make one ounce of perfume -/
def petals_per_ounce : ℕ := 320

/-- The number of petals produced by each rose -/
def petals_per_rose : ℕ := 8

/-- The number of bushes harvested -/
def bushes_harvested : ℕ := 800

/-- The number of bottles of perfume to be made -/
def bottles_to_make : ℕ := 20

/-- The number of ounces in each bottle of perfume -/
def ounces_per_bottle : ℕ := 12

/-- The number of roses per bush -/
def roses_per_bush : ℕ := 12

theorem roses_per_bush_calculation :
  roses_per_bush * bushes_harvested * petals_per_rose =
  bottles_to_make * ounces_per_bottle * petals_per_ounce :=
by sorry

end NUMINAMATH_CALUDE_roses_per_bush_calculation_l3510_351070


namespace NUMINAMATH_CALUDE_consecutive_odd_product_equality_l3510_351050

/-- The product of consecutive integers from (n+1) to (n+n) -/
def consecutiveProduct (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => n + i + 1)

/-- The product of odd numbers from 1 to (2n-1) -/
def oddProduct (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => 2 * i + 1)

/-- The main theorem stating the equality -/
theorem consecutive_odd_product_equality (n : ℕ) :
  n > 0 → consecutiveProduct n = 2^n * oddProduct n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_equality_l3510_351050


namespace NUMINAMATH_CALUDE_quadratic_degree_reduction_l3510_351063

theorem quadratic_degree_reduction (x : ℝ) (h1 : x^2 - x - 1 = 0) (h2 : x > 0) :
  x^4 - 2*x^3 + 3*x = 1 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_degree_reduction_l3510_351063


namespace NUMINAMATH_CALUDE_sum_integers_neg25_to_55_l3510_351006

/-- The sum of integers from a to b, inclusive -/
def sum_integers (a b : ℤ) : ℤ := (b - a + 1) * (a + b) / 2

/-- Theorem: The sum of integers from -25 to 55 is 1215 -/
theorem sum_integers_neg25_to_55 : sum_integers (-25) 55 = 1215 := by
  sorry

end NUMINAMATH_CALUDE_sum_integers_neg25_to_55_l3510_351006


namespace NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_necessary_not_sufficient_l3510_351054

theorem abs_sum_eq_sum_abs_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a * b > 0 → |a + b| = |a| + |b|) ∧
  (∃ a b : ℝ, |a + b| = |a| + |b| ∧ a * b ≤ 0) := by sorry

end NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_necessary_not_sufficient_l3510_351054


namespace NUMINAMATH_CALUDE_smallest_sixth_power_sum_equality_holds_l3510_351058

theorem smallest_sixth_power_sum (n : ℕ) : n > 150 ∧ 135^6 + 115^6 + 85^6 + 30^6 = n^6 → n ≥ 165 := by
  sorry

theorem equality_holds : 135^6 + 115^6 + 85^6 + 30^6 = 165^6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sixth_power_sum_equality_holds_l3510_351058


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3510_351030

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 15) = 12 → x = 129 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3510_351030


namespace NUMINAMATH_CALUDE_complex_calculations_l3510_351033

theorem complex_calculations :
  let z₁ : ℂ := 1 - 2*I
  let z₂ : ℂ := 3 + 4*I
  let z₃ : ℂ := -2 + I
  let w₁ : ℂ := 1 + 2*I
  let w₂ : ℂ := 3 - 4*I
  (z₁ * z₂ * z₃ = 12 + 9*I) ∧
  (w₁ / w₂ = -1/5 + 2/5*I) := by
sorry

end NUMINAMATH_CALUDE_complex_calculations_l3510_351033


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3510_351060

theorem quadratic_equation_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (3 * x₁^2 - 2 * x₁ - 1 = 0) ∧ 
  (3 * x₂^2 - 2 * x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3510_351060


namespace NUMINAMATH_CALUDE_jennifer_theorem_l3510_351077

def jennifer_problem (initial_amount : ℚ) (sandwich_fraction : ℚ) (ticket_fraction : ℚ) (book_fraction : ℚ) : Prop :=
  let sandwich_cost := initial_amount * sandwich_fraction
  let ticket_cost := initial_amount * ticket_fraction
  let book_cost := initial_amount * book_fraction
  let total_spent := sandwich_cost + ticket_cost + book_cost
  let remaining := initial_amount - total_spent
  initial_amount = 90 ∧ 
  sandwich_fraction = 1/5 ∧ 
  ticket_fraction = 1/6 ∧ 
  book_fraction = 1/2 ∧ 
  remaining = 12

theorem jennifer_theorem : 
  ∃ (initial_amount sandwich_fraction ticket_fraction book_fraction : ℚ),
    jennifer_problem initial_amount sandwich_fraction ticket_fraction book_fraction :=
by
  sorry

end NUMINAMATH_CALUDE_jennifer_theorem_l3510_351077


namespace NUMINAMATH_CALUDE_coprime_divides_l3510_351014

theorem coprime_divides (a b n : ℕ) : 
  Nat.Coprime a b → a ∣ n → b ∣ n → (a * b) ∣ n := by
  sorry

end NUMINAMATH_CALUDE_coprime_divides_l3510_351014


namespace NUMINAMATH_CALUDE_family_size_l3510_351026

/-- Represents the number of slices per tomato -/
def slices_per_tomato : ℕ := 8

/-- Represents the number of slices needed for one person's meal -/
def slices_per_meal : ℕ := 20

/-- Represents the number of tomatoes Thelma needs -/
def total_tomatoes : ℕ := 20

/-- Theorem: Given the conditions, the family has 8 people -/
theorem family_size :
  (total_tomatoes * slices_per_tomato) / slices_per_meal = 8 := by
  sorry

end NUMINAMATH_CALUDE_family_size_l3510_351026


namespace NUMINAMATH_CALUDE_smallest_number_in_S_l3510_351089

def S : Set ℝ := {3.2, 2.3, 3, 2.23, 3.22}

theorem smallest_number_in_S : 
  ∃ (x : ℝ), x ∈ S ∧ ∀ y ∈ S, x ≤ y ∧ x = 2.23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_in_S_l3510_351089


namespace NUMINAMATH_CALUDE_rods_in_one_mile_l3510_351088

/-- Conversion factor from miles to chains -/
def mile_to_chain : ℚ := 10

/-- Conversion factor from chains to rods -/
def chain_to_rod : ℚ := 22

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_chain * chain_to_rod

theorem rods_in_one_mile :
  rods_in_mile = 220 :=
by sorry

end NUMINAMATH_CALUDE_rods_in_one_mile_l3510_351088


namespace NUMINAMATH_CALUDE_palindrome_product_sum_theorem_l3510_351029

/-- A positive three-digit palindrome is a natural number between 100 and 999 (inclusive) 
    that reads the same backwards as forwards. -/
def IsPositiveThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ ((n / 10) % 10 = (n % 100) / 10)

/-- The main theorem stating the existence of two positive three-digit palindromes 
    with the given product and sum. -/
theorem palindrome_product_sum_theorem : 
  ∃ (a b : ℕ), IsPositiveThreeDigitPalindrome a ∧ 
                IsPositiveThreeDigitPalindrome b ∧ 
                a * b = 436995 ∧ 
                a + b = 1332 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_product_sum_theorem_l3510_351029


namespace NUMINAMATH_CALUDE_min_value_theorem_l3510_351095

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (b^2 + b*c) + 1 / (c^2 + c*a) + 1 / (a^2 + a*b) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3510_351095


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3510_351071

/-- The sum of the geometric series 15 + 15r + 15r^2 + 15r^3 + ... for -1 < r < 1 -/
noncomputable def S (r : ℝ) : ℝ := 15 / (1 - r)

/-- For -1 < a < 1, if S(a)S(-a) = 2025, then S(a) + S(-a) = 270 -/
theorem geometric_series_sum (a : ℝ) (h1 : -1 < a) (h2 : a < 1) 
  (h3 : S a * S (-a) = 2025) : S a + S (-a) = 270 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3510_351071


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l3510_351040

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (a : ℝ) 
  (hx : x ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (hy : y ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l3510_351040


namespace NUMINAMATH_CALUDE_average_book_price_l3510_351079

/-- The average price of books bought by Rahim -/
theorem average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) 
  (h1 : books1 = 42)
  (h2 : books2 = 22)
  (h3 : price1 = 520)
  (h4 : price2 = 248) :
  (price1 + price2) / (books1 + books2 : ℚ) = 12 := by
  sorry

#check average_book_price

end NUMINAMATH_CALUDE_average_book_price_l3510_351079


namespace NUMINAMATH_CALUDE_area_traced_on_concentric_spheres_l3510_351035

/-- The area traced by a smaller sphere moving between two concentric spheres -/
theorem area_traced_on_concentric_spheres 
  (R1 R2 A1 : ℝ) 
  (h1 : 0 < R1) 
  (h2 : R1 < R2) 
  (h3 : 0 < A1) : 
  ∃ A2 : ℝ, A2 = A1 * (R2/R1)^2 := by
sorry

end NUMINAMATH_CALUDE_area_traced_on_concentric_spheres_l3510_351035


namespace NUMINAMATH_CALUDE_even_function_derivative_is_odd_l3510_351025

-- Define a function f on the real numbers
variable (f : ℝ → ℝ)

-- Define the derivative of f as g
variable (g : ℝ → ℝ)

-- State the theorem
theorem even_function_derivative_is_odd 
  (h1 : ∀ x, f (-x) = f x)  -- f is an even function
  (h2 : ∀ x, HasDerivAt f (g x) x) -- g is the derivative of f
  : ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_even_function_derivative_is_odd_l3510_351025


namespace NUMINAMATH_CALUDE_tim_prank_combinations_l3510_351055

/-- Represents the number of choices Tim has for each day of the week. -/
structure PrankChoices where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total number of prank combinations given the choices for each day. -/
def totalCombinations (choices : PrankChoices) : Nat :=
  choices.monday * choices.tuesday * choices.wednesday * choices.thursday * choices.friday

/-- Tim's prank choices for the week -/
def timChoices : PrankChoices :=
  { monday := 1
    tuesday := 3
    wednesday := 4
    thursday := 3
    friday := 1 }

/-- Theorem stating that the total number of combinations for Tim's prank is 36 -/
theorem tim_prank_combinations :
    totalCombinations timChoices = 36 := by
  sorry


end NUMINAMATH_CALUDE_tim_prank_combinations_l3510_351055


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l3510_351018

/-- A sequence of natural numbers from 1 to 10 -/
def Sequence := Fin 10 → ℕ

/-- Predicate to check if a sequence satisfies the integer percentage difference property -/
def IntegerPercentageDifference (s : Sequence) : Prop :=
  ∀ i : Fin 9, ∃ k : ℤ,
    s (i.succ) = s i + (s i * k) / 100 ∨
    s (i.succ) = s i - (s i * k) / 100

/-- Predicate to check if a sequence contains all numbers from 1 to 10 -/
def ContainsAllNumbers (s : Sequence) : Prop :=
  ∀ n : Fin 10, ∃ i : Fin 10, s i = n.val + 1

/-- Theorem stating that it's impossible to arrange numbers 1 to 10 with the given property -/
theorem no_valid_arrangement :
  ¬ ∃ s : Sequence, IntegerPercentageDifference s ∧ ContainsAllNumbers s := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l3510_351018


namespace NUMINAMATH_CALUDE_flour_for_cookies_l3510_351057

/-- Given a recipe where 20 cookies require 3 cups of flour,
    calculate the number of cups of flour needed for 100 cookies. -/
theorem flour_for_cookies (original_cookies : ℕ) (original_flour : ℕ) (target_cookies : ℕ) :
  original_cookies = 20 →
  original_flour = 3 →
  target_cookies = 100 →
  (target_cookies * original_flour) / original_cookies = 15 :=
by sorry

end NUMINAMATH_CALUDE_flour_for_cookies_l3510_351057


namespace NUMINAMATH_CALUDE_blue_face_prob_is_half_l3510_351085

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  blue_faces : ℕ
  red_faces : ℕ
  green_faces : ℕ
  face_sum : blue_faces + red_faces + green_faces = total_faces

/-- The probability of rolling a blue face on a colored cube -/
def blue_face_probability (cube : ColoredCube) : ℚ :=
  cube.blue_faces / cube.total_faces

/-- Theorem: The probability of rolling a blue face on a cube with 3 blue faces out of 6 total faces is 1/2 -/
theorem blue_face_prob_is_half (cube : ColoredCube) 
    (h1 : cube.total_faces = 6)
    (h2 : cube.blue_faces = 3) : 
    blue_face_probability cube = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_prob_is_half_l3510_351085


namespace NUMINAMATH_CALUDE_abrahams_budget_l3510_351013

/-- Abraham's shopping budget problem -/
theorem abrahams_budget :
  let shower_gel_count : ℕ := 4
  let shower_gel_price : ℕ := 4
  let toothpaste_price : ℕ := 3
  let laundry_detergent_price : ℕ := 11
  let remaining_budget : ℕ := 30
  shower_gel_count * shower_gel_price + toothpaste_price + laundry_detergent_price + remaining_budget = 60
  := by sorry

end NUMINAMATH_CALUDE_abrahams_budget_l3510_351013


namespace NUMINAMATH_CALUDE_video_game_cost_is_87_l3510_351067

/-- The cost of Lindsey's video game -/
def video_game_cost (sept_savings oct_savings nov_savings mom_gift remaining : ℕ) : ℕ :=
  sept_savings + oct_savings + nov_savings + mom_gift - remaining

/-- Theorem stating the cost of the video game -/
theorem video_game_cost_is_87 :
  video_game_cost 50 37 11 25 36 = 87 := by
  sorry

#eval video_game_cost 50 37 11 25 36

end NUMINAMATH_CALUDE_video_game_cost_is_87_l3510_351067


namespace NUMINAMATH_CALUDE_quadratic_shift_l3510_351068

/-- Represents a quadratic function of the form y = (x + a)^2 + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Shifts a quadratic function horizontally and vertically -/
def shift (f : QuadraticFunction) (right : ℝ) (down : ℝ) : QuadraticFunction :=
  { a := f.a - right,
    b := f.b - down }

theorem quadratic_shift :
  let f : QuadraticFunction := { a := 1, b := 3 }
  let g : QuadraticFunction := shift f 2 1
  g = { a := -1, b := 2 } := by sorry

end NUMINAMATH_CALUDE_quadratic_shift_l3510_351068


namespace NUMINAMATH_CALUDE_swap_three_of_eight_eq_112_l3510_351096

/-- The number of ways to select and swap 3 people out of 8 in a row --/
def swap_three_of_eight : ℕ :=
  Nat.choose 8 3 * 2

/-- Theorem stating that swapping 3 out of 8 people results in 112 different arrangements --/
theorem swap_three_of_eight_eq_112 : swap_three_of_eight = 112 := by
  sorry

end NUMINAMATH_CALUDE_swap_three_of_eight_eq_112_l3510_351096


namespace NUMINAMATH_CALUDE_largest_multiple_eleven_l3510_351020

theorem largest_multiple_eleven (n : ℤ) : 
  (n * 11 = -209) → 
  (-n * 11 > -210) ∧ 
  ∀ m : ℤ, (m > n) → (-m * 11 ≤ -210) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_eleven_l3510_351020


namespace NUMINAMATH_CALUDE_tan_315_degrees_l3510_351062

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l3510_351062


namespace NUMINAMATH_CALUDE_sum_interior_angles_formula_l3510_351097

/-- The sum of interior angles of an n-sided polygon -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- Theorem: The sum of the interior angles of an n-sided polygon is (n-2) × 180° -/
theorem sum_interior_angles_formula (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2) * 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_formula_l3510_351097


namespace NUMINAMATH_CALUDE_marys_next_birthday_age_l3510_351048

theorem marys_next_birthday_age 
  (mary sally danielle : ℝ) 
  (h1 : mary = 1.3 * sally) 
  (h2 : sally = 0.5 * danielle) 
  (h3 : mary + sally + danielle = 45) : 
  ⌊mary⌋ + 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_marys_next_birthday_age_l3510_351048


namespace NUMINAMATH_CALUDE_box_filled_by_small_cubes_l3510_351074

/-- Proves that a 1m³ box can be filled by 15625 cubes of 4cm edge length -/
theorem box_filled_by_small_cubes :
  let box_edge : ℝ := 1  -- 1 meter
  let small_cube_edge : ℝ := 0.04  -- 4 cm in meters
  let num_small_cubes : ℕ := 15625
  (box_edge ^ 3) = (small_cube_edge ^ 3) * num_small_cubes := by
  sorry

#check box_filled_by_small_cubes

end NUMINAMATH_CALUDE_box_filled_by_small_cubes_l3510_351074


namespace NUMINAMATH_CALUDE_f_properties_l3510_351086

/-- The function f(x) = tan(3x + φ) + 1 -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.tan (3 * x + φ) + 1

/-- Theorem stating the properties of the function f -/
theorem f_properties (φ : ℝ) (h1 : |φ| < π / 2) (h2 : f φ (π / 9) = 1) :
  (∃ (T : ℝ), T > 0 ∧ T = π / 3 ∧ ∀ (x : ℝ), f φ (x + T) = f φ x) ∧
  (∀ (x : ℝ), f φ x < 2 ↔ ∃ (k : ℤ), -π / 18 + k * π / 3 < x ∧ x < 7 * π / 36 + k * π / 3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3510_351086


namespace NUMINAMATH_CALUDE_sarah_interview_combinations_l3510_351047

/-- Represents the number of interview choices for each day of the week -/
structure WeekChoices where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total number of interview combinations for the week -/
def totalCombinations (choices : WeekChoices) : Nat :=
  choices.monday * choices.tuesday * choices.wednesday * choices.thursday * choices.friday

/-- Represents Sarah's interview choices for the week -/
def sarahChoices : WeekChoices :=
  { monday := 1
  , tuesday := 2
  , wednesday := 5  -- 2 + 3, accounting for both Tuesday possibilities
  , thursday := 5
  , friday := 1 }  -- No interviews, but included for completeness

/-- Theorem stating that Sarah's total interview combinations is 50 -/
theorem sarah_interview_combinations :
  totalCombinations sarahChoices = 50 := by
  sorry

#eval totalCombinations sarahChoices  -- Should output 50

end NUMINAMATH_CALUDE_sarah_interview_combinations_l3510_351047


namespace NUMINAMATH_CALUDE_beach_creatures_ratio_l3510_351031

theorem beach_creatures_ratio :
  ∀ (oysters_day1 crabs_day1 total_both_days : ℕ),
    oysters_day1 = 50 →
    crabs_day1 = 72 →
    total_both_days = 195 →
    ∃ (crabs_day2 : ℕ),
      oysters_day1 + crabs_day1 + (oysters_day1 / 2 + crabs_day2) = total_both_days ∧
      crabs_day2 * 3 = crabs_day1 * 2 :=
by sorry

end NUMINAMATH_CALUDE_beach_creatures_ratio_l3510_351031


namespace NUMINAMATH_CALUDE_largest_fraction_l3510_351042

theorem largest_fraction (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b = c) (h4 : c < d) :
  let f1 := (a + b) / (c + d)
  let f2 := (a + d) / (b + c)
  let f3 := (b + c) / (a + d)
  let f4 := (b + d) / (a + c)
  let f5 := (c + d) / (a + b)
  (f4 = f5) ∧ (f4 ≥ f1) ∧ (f4 ≥ f2) ∧ (f4 ≥ f3) :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l3510_351042


namespace NUMINAMATH_CALUDE_inequality_theorems_l3510_351002

theorem inequality_theorems :
  (∀ a b : ℝ, a > b → (1 / a < 1 / b → a * b > 0)) ∧
  (∀ a b : ℝ, a > b → (1 / a > 1 / b → a > 0 ∧ 0 > b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorems_l3510_351002


namespace NUMINAMATH_CALUDE_policemen_cover_all_streets_l3510_351012

-- Define the set of intersections
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

-- Define the streets as sets of intersections
def horizontal1 : Set Intersection := {Intersection.A, Intersection.B, Intersection.C, Intersection.D}
def horizontal2 : Set Intersection := {Intersection.E, Intersection.F, Intersection.G}
def horizontal3 : Set Intersection := {Intersection.H, Intersection.I, Intersection.J, Intersection.K}
def vertical1 : Set Intersection := {Intersection.A, Intersection.E, Intersection.H}
def vertical2 : Set Intersection := {Intersection.B, Intersection.F, Intersection.I}
def vertical3 : Set Intersection := {Intersection.D, Intersection.G, Intersection.J}
def diagonal1 : Set Intersection := {Intersection.H, Intersection.F, Intersection.C}
def diagonal2 : Set Intersection := {Intersection.C, Intersection.G, Intersection.K}

-- Define the set of all streets
def allStreets : Set (Set Intersection) := 
  {horizontal1, horizontal2, horizontal3, vertical1, vertical2, vertical3, diagonal1, diagonal2}

-- Define the chosen intersections for policemen
def chosenIntersections : Set Intersection := {Intersection.B, Intersection.G, Intersection.H}

-- Theorem: The chosen intersections cover all streets
theorem policemen_cover_all_streets : 
  ∀ street ∈ allStreets, ∃ intersection ∈ chosenIntersections, intersection ∈ street :=
sorry

end NUMINAMATH_CALUDE_policemen_cover_all_streets_l3510_351012


namespace NUMINAMATH_CALUDE_tax_reduction_scientific_notation_l3510_351032

theorem tax_reduction_scientific_notation :
  (15.75 * 10^9 : ℝ) = 1.575 * 10^10 := by sorry

end NUMINAMATH_CALUDE_tax_reduction_scientific_notation_l3510_351032


namespace NUMINAMATH_CALUDE_club_assignment_count_l3510_351003

/-- Represents the four clubs --/
inductive Club
| Literature
| Drama
| Anime
| Love

/-- Represents the five students --/
inductive Student
| A
| B
| C
| D
| E

/-- A valid club assignment is a function from Student to Club --/
def ClubAssignment := Student → Club

/-- Checks if a club assignment is valid according to the problem conditions --/
def is_valid_assignment (assignment : ClubAssignment) : Prop :=
  (∀ c : Club, ∃ s : Student, assignment s = c) ∧ 
  (assignment Student.A ≠ Club.Anime)

/-- The number of valid club assignments --/
def num_valid_assignments : ℕ := sorry

theorem club_assignment_count : num_valid_assignments = 180 := by sorry

end NUMINAMATH_CALUDE_club_assignment_count_l3510_351003


namespace NUMINAMATH_CALUDE_relay_race_distance_ratio_l3510_351019

theorem relay_race_distance_ratio :
  ∀ (last_year_distance : ℕ) (table_count : ℕ) (distance_1_to_3 : ℕ),
    last_year_distance = 300 →
    table_count = 6 →
    distance_1_to_3 = 400 →
    ∃ (this_year_distance : ℕ),
      this_year_distance % last_year_distance = 0 ∧
      (this_year_distance : ℚ) / last_year_distance = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_distance_ratio_l3510_351019


namespace NUMINAMATH_CALUDE_impossibility_of_transformation_l3510_351007

def operation (a b : ℤ) : ℤ × ℤ := (5*a - 2*b, 3*a - 4*b)

def initial_set : Set ℤ := {n | 1 ≤ n ∧ n ≤ 2018}

def target_sequence : Set ℤ := {n | ∃ k, 1 ≤ k ∧ k ≤ 2018 ∧ n = 3*k}

theorem impossibility_of_transformation :
  ∀ (S : Set ℤ), S = initial_set →
  ¬∃ (n : ℕ), ∃ (S' : Set ℤ),
    (∀ k ≤ n, ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧
      S' = (S \ {a, b}) ∪ {(operation a b).1, (operation a b).2}) →
    target_sequence ⊆ S' :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_transformation_l3510_351007


namespace NUMINAMATH_CALUDE_diagonal_sum_inequality_l3510_351092

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry

-- Define the sum of diagonal lengths for a quadrilateral
def sum_of_diagonals (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the "inside" relation for quadrilaterals
def inside (inner outer : ConvexQuadrilateral) : Prop := sorry

-- Theorem statement
theorem diagonal_sum_inequality {P P' : ConvexQuadrilateral} 
  (h_inside : inside P' P) : 
  sum_of_diagonals P' < 2 * sum_of_diagonals P := by
  sorry

end NUMINAMATH_CALUDE_diagonal_sum_inequality_l3510_351092


namespace NUMINAMATH_CALUDE_least_n_for_inequality_l3510_351021

theorem least_n_for_inequality : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, k > 0 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) < (1 : ℚ) / 8 → k ≥ n) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 8) ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_least_n_for_inequality_l3510_351021


namespace NUMINAMATH_CALUDE_inequalities_theorem_l3510_351059

theorem inequalities_theorem (a b m : ℝ) :
  (b < a ∧ a < 0 → 1 / a < 1 / b) ∧
  (b > a ∧ a > 0 ∧ m > 0 → (a + m) / (b + m) > a / b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3510_351059


namespace NUMINAMATH_CALUDE_inequality_solution_count_l3510_351066

theorem inequality_solution_count : 
  (∃ (S : Finset Int), 
    (∀ n : Int, n ∈ S ↔ Real.sqrt (2 * n) ≤ Real.sqrt (5 * n - 8) ∧ 
                        Real.sqrt (5 * n - 8) < Real.sqrt (3 * n + 7)) ∧
    S.card = 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l3510_351066


namespace NUMINAMATH_CALUDE_sum_of_a_roots_l3510_351069

theorem sum_of_a_roots (a b c : ℂ) : 
  a + b + a * c = 5 →
  b + c + a * b = 10 →
  c + a + b * c = 15 →
  ∃ (s : ℂ), s = -7 ∧ (∀ (a' : ℂ), (∃ (b' c' : ℂ), 
    a' + b' + a' * c' = 5 ∧
    b' + c' + a' * b' = 10 ∧
    c' + a' + b' * c' = 15) → 
    (a' - s) * (a' ^ 2 + 7 * a' + 11 - 5 / a') = 0) :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_roots_l3510_351069


namespace NUMINAMATH_CALUDE_shaded_area_outside_overlap_l3510_351016

/-- Given two rectangles with specific dimensions and overlap, calculate the shaded area outside the overlap -/
theorem shaded_area_outside_overlap (rect1_width rect1_height rect2_width rect2_height overlap_width overlap_height : ℕ) 
  (h1 : rect1_width = 4 ∧ rect1_height = 12)
  (h2 : rect2_width = 5 ∧ rect2_height = 9)
  (h3 : overlap_width = 4 ∧ overlap_height = 5) :
  rect1_width * rect1_height + rect2_width * rect2_height - overlap_width * overlap_height = 73 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_outside_overlap_l3510_351016


namespace NUMINAMATH_CALUDE_cubic_roots_of_27_l3510_351094

theorem cubic_roots_of_27 :
  let z₁ : ℂ := 3
  let z₂ : ℂ := -3/2 + (3*Complex.I*Real.sqrt 3)/2
  let z₃ : ℂ := -3/2 - (3*Complex.I*Real.sqrt 3)/2
  (z₁^3 = 27 ∧ z₂^3 = 27 ∧ z₃^3 = 27) ∧
  ∀ z : ℂ, z^3 = 27 → (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_of_27_l3510_351094


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3510_351080

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 6*x + 3 = 0) ↔ ((x + 3)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3510_351080


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l3510_351000

/-- An ellipse with a vertex at (0,1) and focal length 2√3 -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h0 : a > b
  h1 : b > 0
  h2 : b = 1
  h3 : a^2 = b^2 + 3

/-- The intersection points of a line with the ellipse -/
def LineEllipseIntersection (E : SpecialEllipse) (k : ℝ) :=
  {x : ℝ × ℝ | ∃ t, x.1 = -2 + t ∧ x.2 = 1 + k*t ∧ (x.1^2 / E.a^2 + x.2^2 / E.b^2 = 1)}

/-- The x-intercepts of lines connecting (0,1) to the intersection points -/
def XIntercepts (E : SpecialEllipse) (k : ℝ) (B C : ℝ × ℝ) :=
  {x : ℝ | ∃ t, (t*B.1 = x ∧ t*B.2 = 1) ∨ (t*C.1 = x ∧ t*C.2 = 1)}

theorem special_ellipse_properties (E : SpecialEllipse) :
  (∀ x y, x^2/4 + y^2 = 1 ↔ x^2/E.a^2 + y^2/E.b^2 = 1) ∧
  (∀ k : ℝ, k ≠ 0 →
    ∀ B C : ℝ × ℝ, B ∈ LineEllipseIntersection E k → C ∈ LineEllipseIntersection E k → B ≠ C →
    ∀ M N : ℝ, M ∈ XIntercepts E k B C → N ∈ XIntercepts E k B C → M ≠ N →
    (M - N)^2 * |k| = 16) :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l3510_351000


namespace NUMINAMATH_CALUDE_greatest_product_sum_300_l3510_351009

theorem greatest_product_sum_300 : 
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_greatest_product_sum_300_l3510_351009


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3510_351022

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction)
  (h : ∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3510_351022


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3510_351051

def A : Set ℤ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}

def B : Set ℤ := {y | ∃ x ∈ A, y = 2*x - 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3510_351051


namespace NUMINAMATH_CALUDE_max_visible_cubes_10x10x10_l3510_351072

/-- Represents a cube made of unit cubes -/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  let face_cubes := cube.size * cube.size
  let edge_cubes := cube.size - 1
  3 * face_cubes - 3 * edge_cubes + 1

/-- Theorem stating that for a 10x10x10 cube, the maximum number of visible unit cubes is 274 -/
theorem max_visible_cubes_10x10x10 :
  max_visible_cubes (UnitCube.mk 10) = 274 := by sorry

end NUMINAMATH_CALUDE_max_visible_cubes_10x10x10_l3510_351072


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l3510_351082

/-- A quadratic equation with parameter k -/
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x^2 + 6 * x + k^2 - k

theorem quadratic_root_zero (k : ℝ) :
  (quadratic_equation k 0 = 0) ∧ (k - 1 ≠ 0) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l3510_351082


namespace NUMINAMATH_CALUDE_difference_solution_eq_one_difference_solution_eq_two_difference_solution_eq_three_l3510_351091

/-- Definition of a difference solution equation -/
def is_difference_solution_equation (a b : ℝ) : Prop :=
  b / a = b - a

/-- Theorem for 4x = m -/
theorem difference_solution_eq_one (m : ℝ) :
  is_difference_solution_equation 4 m ↔ m = 16 / 3 := by sorry

/-- Theorem for 4x = ab + a -/
theorem difference_solution_eq_two (a b : ℝ) :
  is_difference_solution_equation 4 (a * b + a) → 3 * (a * b + a) = 16 := by sorry

/-- Theorem for 4x = mn + m and -2x = mn + n -/
theorem difference_solution_eq_three (m n : ℝ) :
  is_difference_solution_equation 4 (m * n + m) →
  is_difference_solution_equation (-2) (m * n + n) →
  3 * (m * n + m) - 9 * (m * n + n)^2 = 0 := by sorry

end NUMINAMATH_CALUDE_difference_solution_eq_one_difference_solution_eq_two_difference_solution_eq_three_l3510_351091


namespace NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l3510_351037

/-- Represents the ratios in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio := ⟨1, 12, 30⟩

/-- The sport formulation of the drink -/
def sport_ratio : DrinkRatio :=
  ⟨1, 
   3 * standard_ratio.corn_syrup / standard_ratio.flavoring,
   standard_ratio.water / (2 * standard_ratio.flavoring)⟩

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def corn_syrup_amount : ℚ := 8

/-- Theorem stating the amount of water in the sport formulation -/
theorem water_amount_in_sport_formulation :
  (corn_syrup_amount * sport_ratio.water) / sport_ratio.corn_syrup = 30 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l3510_351037


namespace NUMINAMATH_CALUDE_water_tank_capacity_l3510_351024

theorem water_tank_capacity (c : ℝ) : 
  (c / 3 + 10) / c = 2 / 5 → c = 150 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l3510_351024


namespace NUMINAMATH_CALUDE_coin_value_increase_l3510_351011

def coins_bought : ℕ := 20
def initial_price : ℚ := 15
def coins_sold : ℕ := 12

def original_investment : ℚ := coins_bought * initial_price
def selling_price : ℚ := original_investment

theorem coin_value_increase :
  (selling_price / coins_sold - initial_price) / initial_price = 2/3 :=
sorry

end NUMINAMATH_CALUDE_coin_value_increase_l3510_351011


namespace NUMINAMATH_CALUDE_adjacent_vertices_probability_l3510_351046

/-- A decagon is a polygon with 10 sides and vertices -/
def Decagon := Nat

/-- The number of vertices in a decagon -/
def num_vertices : Decagon → Nat := fun _ => 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def num_adjacent_vertices : Decagon → Nat := fun _ => 2

/-- The total number of ways to choose the second vertex -/
def total_second_vertex_choices : Decagon → Nat := fun d => num_vertices d - 1

theorem adjacent_vertices_probability (d : Decagon) :
  (num_adjacent_vertices d : ℚ) / (total_second_vertex_choices d) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_vertices_probability_l3510_351046


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_system_two_equations_solution_system_three_equations_solution_l3510_351064

-- Problem 1
theorem equation_one_solution (x : ℝ) : 3 * x - 2 = 10 - 2 * (x + 1) → x = 2 := by
  sorry

-- Problem 2
theorem equation_two_solution (x : ℝ) : (2 * x + 1) / 3 - (5 * x - 1) / 6 = 1 → x = -3 := by
  sorry

-- Problem 3
theorem system_two_equations_solution (x y : ℝ) : 
  x + 2 * y = 5 ∧ 3 * x - 2 * y = -1 → x = 1 ∧ y = 2 := by
  sorry

-- Problem 4
theorem system_three_equations_solution (x y z : ℝ) :
  2 * x + y + z = 15 ∧ x + 2 * y + z = 16 ∧ x + y + 2 * z = 17 → 
  x = 3 ∧ y = 4 ∧ z = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_system_two_equations_solution_system_three_equations_solution_l3510_351064


namespace NUMINAMATH_CALUDE_mental_health_survey_is_comprehensive_l3510_351045

/-- Represents a survey --/
structure Survey where
  description : String
  population : Set String
  environment : String

/-- Conditions for a comprehensive survey --/
def is_comprehensive (s : Survey) : Prop :=
  s.population.Finite ∧
  s.population.Nonempty ∧
  (∀ x ∈ s.population, ∃ y, y = x) ∧
  s.environment = "Contained"

/-- The survey on students' mental health --/
def mental_health_survey : Survey :=
  { description := "Survey on the current status of students' mental health in a school in Huicheng District"
  , population := {"Students in a school in Huicheng District"}
  , environment := "Contained" }

/-- Theorem stating that the mental health survey is comprehensive --/
theorem mental_health_survey_is_comprehensive :
  is_comprehensive mental_health_survey :=
sorry

end NUMINAMATH_CALUDE_mental_health_survey_is_comprehensive_l3510_351045


namespace NUMINAMATH_CALUDE_min_distance_parabola_circle_l3510_351049

/-- The minimum distance between a point on the parabola y^2 = x and a point on the circle (x-3)^2 + y^2 = 1 is 1/2 (√11 - 2). -/
theorem min_distance_parabola_circle :
  let parabola := {p : ℝ × ℝ | p.2^2 = p.1}
  let circle := {q : ℝ × ℝ | (q.1 - 3)^2 + q.2^2 = 1}
  ∃ (d : ℝ), d = (Real.sqrt 11 - 2) / 2 ∧
    ∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ parabola → q ∈ circle →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parabola_circle_l3510_351049


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l3510_351015

/-- A quadratic function passing through the origin -/
def passes_through_origin (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c = 0

theorem quadratic_through_origin (a b c : ℝ) (h : a ≠ 0) :
  passes_through_origin a b c ↔ c = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l3510_351015


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l3510_351041

/-- UFO Convention Attendees Problem -/
theorem ufo_convention_attendees :
  ∀ (male_attendees female_attendees : ℕ),
  male_attendees = 62 →
  male_attendees = female_attendees + 4 →
  male_attendees + female_attendees = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l3510_351041


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l3510_351027

def f (x : ℝ) : ℝ := -x^3

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l3510_351027


namespace NUMINAMATH_CALUDE_binomial_variance_four_third_l3510_351084

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  (p_nonneg : 0 ≤ p)
  (p_le_one : p ≤ 1)

/-- The variance of a binomial random variable -/
def variance (n : ℕ) (p : ℝ) (ξ : BinomialRV n p) : ℝ :=
  n * p * (1 - p)

theorem binomial_variance_four_third (ξ : BinomialRV 4 (1/3)) :
  variance 4 (1/3) ξ = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_four_third_l3510_351084


namespace NUMINAMATH_CALUDE_max_profit_is_45_6_l3510_351075

/-- Profit function for location A -/
def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def profit_B (x : ℝ) : ℝ := 2 * x

/-- Total number of cars sold -/
def total_cars : ℕ := 15

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (total_cars - x)

theorem max_profit_is_45_6 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_cars ∧
  ∀ y : ℝ, y ≥ 0 → y ≤ total_cars → total_profit y ≤ total_profit x ∧
  total_profit x = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_45_6_l3510_351075


namespace NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l3510_351023

theorem sum_of_four_primes_divisible_by_60 
  (p q r s : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hr : Nat.Prime r) 
  (hs : Nat.Prime s) 
  (h_order : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) : 
  60 ∣ (p + q + r + s) := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l3510_351023


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l3510_351010

theorem rectangular_solid_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 20)
  (h_front : front_area = 15)
  (h_bottom : bottom_area = 12) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l3510_351010


namespace NUMINAMATH_CALUDE_first_quarter_time_proportion_l3510_351028

/-- Represents the proportion of time spent traveling the first quarter of a distance
    when the speed for that quarter is 4 times the speed for the remaining distance -/
def time_proportion_first_quarter : ℚ := 1 / 13

/-- Proves that the proportion of time spent traveling the first quarter of the distance
    is 1/13 of the total time, given the specified speed conditions -/
theorem first_quarter_time_proportion 
  (D : ℝ) -- Total distance
  (V : ℝ) -- Speed for the remaining three-quarters of the distance
  (h1 : D > 0) -- Distance is positive
  (h2 : V > 0) -- Speed is positive
  : (D / (16 * V)) / ((D / (16 * V)) + (3 * D / (4 * V))) = time_proportion_first_quarter :=
sorry

end NUMINAMATH_CALUDE_first_quarter_time_proportion_l3510_351028


namespace NUMINAMATH_CALUDE_three_digit_subtraction_result_l3510_351039

theorem three_digit_subtraction_result :
  ∃ (a b c d e f : ℕ),
    100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c ≤ 999 ∧
    100 ≤ d * 100 + e * 10 + f ∧ d * 100 + e * 10 + f ≤ 999 ∧
    (∃ (g : ℕ), 0 ≤ g ∧ g ≤ 9 ∧ (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = g) ∧
    (∃ (h i : ℕ), 10 ≤ h * 10 + i ∧ h * 10 + i ≤ 99 ∧ (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = h * 10 + i) ∧
    (∃ (j k l : ℕ), 100 ≤ j * 100 + k * 10 + l ∧ j * 100 + k * 10 + l ≤ 999 ∧ (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = j * 100 + k * 10 + l) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_result_l3510_351039


namespace NUMINAMATH_CALUDE_prob_difference_increases_l3510_351061

/-- The probability of getting exactly 5 heads in 10 coin flips -/
def prob_five_heads : ℚ := 252 / 1024

/-- The probability of the absolute difference increasing given equal heads and tails -/
def prob_increase_equal : ℚ := 1

/-- The probability of the absolute difference increasing given unequal heads and tails -/
def prob_increase_unequal : ℚ := 1 / 2

/-- The probability of the absolute difference between heads and tails increasing after an 11th coin flip, given 10 initial flips -/
theorem prob_difference_increases : 
  prob_five_heads * prob_increase_equal + 
  (1 - prob_five_heads) * prob_increase_unequal = 319 / 512 := by
  sorry

end NUMINAMATH_CALUDE_prob_difference_increases_l3510_351061


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3510_351005

theorem trigonometric_identity (α β γ n : ℝ) 
  (h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)) :
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = (n + 1) / (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3510_351005


namespace NUMINAMATH_CALUDE_limes_remaining_l3510_351053

/-- The number of limes Mike picked -/
def limes_picked : Real := 32.0

/-- The number of limes Alyssa ate -/
def limes_eaten : Real := 25.0

/-- The number of limes left -/
def limes_left : Real := limes_picked - limes_eaten

theorem limes_remaining : limes_left = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_limes_remaining_l3510_351053


namespace NUMINAMATH_CALUDE_volleyball_teams_l3510_351098

theorem volleyball_teams (managers : ℕ) (employees : ℕ) (team_size : ℕ) : 
  managers = 23 → employees = 7 → team_size = 5 → 
  (managers + employees) / team_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_teams_l3510_351098


namespace NUMINAMATH_CALUDE_scalene_triangle_c_equals_four_l3510_351090

/-- A scalene triangle with integer side lengths satisfying a specific equation -/
structure ScaleneTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  equation : a^2 + b^2 - 6*a - 4*b + 13 = 0

/-- Theorem: If a scalene triangle satisfies the given equation, then c = 4 -/
theorem scalene_triangle_c_equals_four (t : ScaleneTriangle) : t.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_scalene_triangle_c_equals_four_l3510_351090


namespace NUMINAMATH_CALUDE_intersection_of_parallel_lines_l3510_351043

/-- The number of parallelograms formed by the intersection of two sets of parallel lines -/
def parallelograms (n m : ℕ) : ℕ := n.choose 2 * m

/-- Given two sets of parallel lines intersecting in a plane, 
    where one set has 8 lines and they form 280 parallelograms, 
    prove that the other set must have 10 lines -/
theorem intersection_of_parallel_lines : 
  ∃ (n : ℕ), n > 0 ∧ parallelograms n 8 = 280 ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_parallel_lines_l3510_351043


namespace NUMINAMATH_CALUDE_tripled_room_painting_cost_l3510_351034

/-- Represents the cost of painting a room -/
structure PaintingCost where
  original : ℝ
  scaled : ℝ

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculates the wall area of a room given its dimensions -/
def wallArea (d : RoomDimensions) : ℝ :=
  2 * (d.length + d.breadth) * d.height

/-- Scales the dimensions of a room by a factor -/
def scaleDimensions (d : RoomDimensions) (factor : ℝ) : RoomDimensions :=
  { length := d.length * factor
  , breadth := d.breadth * factor
  , height := d.height * factor }

/-- Theorem: The cost of painting a room with tripled dimensions is Rs. 3150 
    given that the original cost is Rs. 350 -/
theorem tripled_room_painting_cost 
  (d : RoomDimensions) 
  (c : PaintingCost) 
  (h1 : c.original = 350) 
  (h2 : c.original / wallArea d = c.scaled / wallArea (scaleDimensions d 3)) : 
  c.scaled = 3150 := by
  sorry

end NUMINAMATH_CALUDE_tripled_room_painting_cost_l3510_351034


namespace NUMINAMATH_CALUDE_card_sum_theorem_l3510_351044

theorem card_sum_theorem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end NUMINAMATH_CALUDE_card_sum_theorem_l3510_351044


namespace NUMINAMATH_CALUDE_odd_cube_minus_n_div_24_l3510_351083

theorem odd_cube_minus_n_div_24 (n : ℤ) (h : Odd n) : ∃ k : ℤ, n^3 - n = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_odd_cube_minus_n_div_24_l3510_351083


namespace NUMINAMATH_CALUDE_six_digit_square_from_three_squares_l3510_351004

/-- A function that concatenates three two-digit numbers into a six-digit number -/
def concatenate (a b c : Nat) : Nat :=
  10000 * a + 100 * b + c

/-- A predicate that checks if a number is a two-digit perfect square -/
def is_two_digit_perfect_square (n : Nat) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ k, k * k = n

/-- The main theorem statement -/
theorem six_digit_square_from_three_squares :
  ∀ a b c : Nat,
    is_two_digit_perfect_square a →
    is_two_digit_perfect_square b →
    is_two_digit_perfect_square c →
    (∃ t : Nat, t * t = concatenate a b c) →
    concatenate a b c = 166464 ∨ concatenate a b c = 646416 := by
  sorry


end NUMINAMATH_CALUDE_six_digit_square_from_three_squares_l3510_351004


namespace NUMINAMATH_CALUDE_hotel_room_charges_l3510_351038

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R - 0.5 * R) 
  (h2 : P = G - 0.1 * G) : 
  R = G + 0.8 * G := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l3510_351038


namespace NUMINAMATH_CALUDE_rose_price_theorem_l3510_351001

/-- The price of an individual rose -/
def individual_rose_price : ℝ := 7.5

/-- The cost of one dozen roses -/
def dozen_price : ℝ := 36

/-- The cost of two dozen roses -/
def two_dozen_price : ℝ := 50

/-- The maximum number of roses that can be purchased for $680 -/
def max_roses : ℕ := 316

/-- The total budget available -/
def total_budget : ℝ := 680

theorem rose_price_theorem :
  (dozen_price = 12 * individual_rose_price) ∧
  (two_dozen_price = 24 * individual_rose_price) ∧
  (∀ n : ℕ, n * individual_rose_price ≤ total_budget → n ≤ max_roses) ∧
  (max_roses * individual_rose_price ≤ total_budget) :=
by sorry

end NUMINAMATH_CALUDE_rose_price_theorem_l3510_351001


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l3510_351087

theorem ice_cream_combinations (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l3510_351087


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l3510_351052

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 6

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 5

/-- The minimum area requirement for the rectangle -/
def min_area : ℝ := 1

/-- The function to calculate the number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating the number of ways to choose four lines to form a rectangle with area ≥ 1 -/
theorem rectangle_formation_count :
  choose_two num_horizontal_lines * choose_two num_vertical_lines = 150 :=
sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l3510_351052


namespace NUMINAMATH_CALUDE_isabela_cucumber_purchase_l3510_351056

/-- The number of cucumbers Isabela bought -/
def cucumbers : ℕ := 100

/-- The number of pencils Isabela bought -/
def pencils : ℕ := cucumbers / 2

/-- The original price of each item in dollars -/
def original_price : ℕ := 20

/-- The discount percentage on pencils -/
def discount_percentage : ℚ := 20 / 100

/-- The discounted price of pencils in dollars -/
def discounted_pencil_price : ℚ := original_price * (1 - discount_percentage)

/-- The total amount spent in dollars -/
def total_spent : ℕ := 2800

theorem isabela_cucumber_purchase :
  cucumbers = 100 ∧
  cucumbers = 2 * pencils ∧
  (pencils : ℚ) * discounted_pencil_price + (cucumbers : ℚ) * original_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_isabela_cucumber_purchase_l3510_351056


namespace NUMINAMATH_CALUDE_triangle_properties_l3510_351076

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

def satisfies_equation (x : ℝ) : Prop :=
  x^2 - 2 * Real.sqrt 3 * x + 2 = 0

theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : satisfies_equation t.a)
  (h3 : satisfies_equation t.b)
  (h4 : Real.cos (t.A + t.B) = 1/2) :
  t.C = Real.pi/3 ∧ 
  t.c = Real.sqrt 6 ∧
  (1/2 * t.a * t.b * Real.sin t.C) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3510_351076


namespace NUMINAMATH_CALUDE_expected_heads_value_l3510_351081

/-- The probability of a coin landing heads -/
def p_heads : ℚ := 1/3

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The maximum number of flips allowed for each coin -/
def max_flips : ℕ := 4

/-- The probability of a coin showing heads after up to four flips -/
def p_heads_after_four_flips : ℚ :=
  p_heads + (1 - p_heads) * p_heads + (1 - p_heads)^2 * p_heads + (1 - p_heads)^3 * p_heads

/-- The expected number of coins showing heads after all flips -/
def expected_heads : ℚ := num_coins * p_heads_after_four_flips

theorem expected_heads_value : expected_heads = 6500/81 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_value_l3510_351081


namespace NUMINAMATH_CALUDE_roberto_outfits_l3510_351099

/-- The number of different outfits Roberto can assemble -/
def number_of_outfits : ℕ :=
  let trousers : ℕ := 4
  let shirts : ℕ := 8
  let jackets : ℕ := 3
  let belts : ℕ := 2
  trousers * shirts * jackets * belts

theorem roberto_outfits :
  number_of_outfits = 192 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l3510_351099


namespace NUMINAMATH_CALUDE_remainder_sum_l3510_351078

theorem remainder_sum (a b : ℤ) (ha : a % 60 = 41) (hb : b % 45 = 14) : (a + b) % 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3510_351078


namespace NUMINAMATH_CALUDE_tuesday_rainfall_correct_l3510_351036

/-- Represents the rainfall data for three days -/
structure RainfallData where
  total : Float
  monday : Float
  wednesday : Float

/-- Calculates the rainfall on Tuesday given the rainfall data for three days -/
def tuesdayRainfall (data : RainfallData) : Float :=
  data.total - (data.monday + data.wednesday)

/-- Theorem stating that the rainfall on Tuesday is correctly calculated -/
theorem tuesday_rainfall_correct (data : RainfallData) 
  (h1 : data.total = 0.6666666666666666)
  (h2 : data.monday = 0.16666666666666666)
  (h3 : data.wednesday = 0.08333333333333333) :
  tuesdayRainfall data = 0.41666666666666663 := by
  sorry

#eval tuesdayRainfall { 
  total := 0.6666666666666666, 
  monday := 0.16666666666666666, 
  wednesday := 0.08333333333333333 
}

end NUMINAMATH_CALUDE_tuesday_rainfall_correct_l3510_351036


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3510_351008

theorem min_value_of_sum_of_squares (x y : ℝ) : 
  (x + 5)^2 + (y - 12)^2 = 14^2 → ∃ (min : ℝ), min = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2 := by
  sorry

#check min_value_of_sum_of_squares

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3510_351008
