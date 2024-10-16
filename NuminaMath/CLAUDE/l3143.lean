import Mathlib

namespace NUMINAMATH_CALUDE_reading_time_difference_example_l3143_314350

/-- The difference in reading time (in minutes) between two readers for a given book -/
def reading_time_difference (xavier_speed maya_speed : ℕ) (book_pages : ℕ) : ℕ :=
  ((book_pages / maya_speed - book_pages / xavier_speed) * 60)

/-- Theorem: Given Xavier's and Maya's reading speeds and the book length, 
    the difference in reading time is 180 minutes -/
theorem reading_time_difference_example : 
  reading_time_difference 120 60 360 = 180 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_example_l3143_314350


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3143_314341

/-- A triangle with sides a, b, and c is isosceles if at least two sides are equal -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

/-- The triangle inequality theorem -/
def SatisfiesTriangleInequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- The perimeter of a triangle -/
def Perimeter (a b c : ℝ) : ℝ :=
  a + b + c

/-- The unique isosceles triangle with sides 10 and 22 has perimeter 54 -/
theorem isosceles_triangle_perimeter :
  ∃! (a b c : ℝ),
    a = 10 ∧ (b = 22 ∨ c = 22) ∧
    IsIsosceles a b c ∧
    SatisfiesTriangleInequality a b c ∧
    Perimeter a b c = 54 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3143_314341


namespace NUMINAMATH_CALUDE_max_abs_sum_quadratic_coeff_l3143_314363

theorem max_abs_sum_quadratic_coeff (a b c : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) → 
  |a| + |b| + |c| ≤ 3 ∧ ∃ a' b' c' : ℝ, 
    (∀ x : ℝ, |x| ≤ 1 → |a'*x^2 + b'*x + c'| ≤ 1) ∧ 
    |a'| + |b'| + |c'| = 3 :=
sorry

end NUMINAMATH_CALUDE_max_abs_sum_quadratic_coeff_l3143_314363


namespace NUMINAMATH_CALUDE_largest_fraction_l3143_314315

theorem largest_fraction : 
  let a := (5 : ℚ) / 12
  let b := (7 : ℚ) / 16
  let c := (23 : ℚ) / 48
  let d := (99 : ℚ) / 200
  let e := (201 : ℚ) / 400
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l3143_314315


namespace NUMINAMATH_CALUDE_f_sum_equals_negative_two_l3143_314392

def f (x : ℝ) : ℝ := x^3 - x - 1

theorem f_sum_equals_negative_two : 
  f 2023 + (deriv f) 2023 + f (-2023) - (deriv f) (-2023) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_negative_two_l3143_314392


namespace NUMINAMATH_CALUDE_area_original_figure_l3143_314391

/-- Given an isosceles trapezoid representing the isometric drawing of a horizontally placed figure,
    with a bottom angle of 60°, legs and top base of length 1,
    the area of the original plane figure is 3√6/2. -/
theorem area_original_figure (bottom_angle : ℝ) (leg_length : ℝ) (top_base : ℝ) : 
  bottom_angle = π / 3 →
  leg_length = 1 →
  top_base = 1 →
  ∃ (area : ℝ), area = (3 * Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_original_figure_l3143_314391


namespace NUMINAMATH_CALUDE_loop_n1_theorem_l3143_314325

theorem loop_n1_theorem (n : ℕ) (α : ℕ) (β : ℚ) :
  let ℓ : ℕ → ℕ := λ k => k
  let m : ℕ → ℚ := λ k => (n - 1 : ℚ) / (2^k : ℚ)
  (β = (n - 1 : ℚ) / (2^α : ℚ)) →
  (ℓ α = α ∧ m α = β) :=
by sorry

end NUMINAMATH_CALUDE_loop_n1_theorem_l3143_314325


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3143_314313

/-- Represents the repeating decimal 7.036036036... -/
def repeating_decimal : ℚ := 7 + 36 / 999

/-- The repeating decimal 7.036036036... is equal to the fraction 781/111 -/
theorem repeating_decimal_as_fraction : repeating_decimal = 781 / 111 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3143_314313


namespace NUMINAMATH_CALUDE_visitor_decrease_l3143_314388

theorem visitor_decrease (P V : ℝ) (h1 : P > 0) (h2 : V > 0) : 
  let R := P * V
  let P' := 1.5 * P
  let R' := 1.2 * R
  ∃ V', R' = P' * V' ∧ V' = 0.8 * V :=
by sorry

end NUMINAMATH_CALUDE_visitor_decrease_l3143_314388


namespace NUMINAMATH_CALUDE_women_per_table_l3143_314311

theorem women_per_table (tables : Nat) (men_per_table : Nat) (total_customers : Nat) :
  tables = 9 →
  men_per_table = 3 →
  total_customers = 90 →
  (total_customers - tables * men_per_table) / tables = 7 := by
  sorry

end NUMINAMATH_CALUDE_women_per_table_l3143_314311


namespace NUMINAMATH_CALUDE_expression_simplification_l3143_314372

theorem expression_simplification (x y : ℝ) (h : 2 * x + y - 3 = 0) :
  ((3 * x) / (x - y) + x / (x + y)) / (x / (x^2 - y^2)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3143_314372


namespace NUMINAMATH_CALUDE_max_questions_is_13_l3143_314336

/-- Represents a quiz with questions and student solutions -/
structure Quiz where
  questions : Nat
  students : Nat
  solvedBy : Nat → Finset Nat  -- For each question, the set of students who solved it
  solvedQuestions : Nat → Finset Nat  -- For each student, the set of questions they solved

/-- Properties that must hold for a valid quiz configuration -/
def ValidQuiz (q : Quiz) : Prop :=
  (∀ i : Nat, i < q.questions → (q.solvedBy i).card = 4) ∧
  (∀ i j : Nat, i < q.questions → j < q.questions → i ≠ j →
    (q.solvedBy i ∩ q.solvedBy j).card = 1) ∧
  (∀ s : Nat, s < q.students → (q.solvedQuestions s).card < q.questions)

/-- The maximum number of questions possible in a valid quiz configuration -/
def MaxQuestions : Nat := 13

/-- Theorem stating that 13 is the maximum number of questions in a valid quiz -/
theorem max_questions_is_13 :
  ∀ q : Quiz, ValidQuiz q → q.questions ≤ MaxQuestions :=
sorry

end NUMINAMATH_CALUDE_max_questions_is_13_l3143_314336


namespace NUMINAMATH_CALUDE_not_divisible_by_n_plus_4_l3143_314337

theorem not_divisible_by_n_plus_4 (n : ℕ) : ¬(∃ k : ℤ, n^2 + 8*n + 15 = (n + 4) * k) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_n_plus_4_l3143_314337


namespace NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l3143_314312

theorem geese_percentage_among_non_swans :
  ∀ (total : ℝ) (geese swan heron duck : ℝ),
    geese / total = 0.35 →
    swan / total = 0.20 →
    heron / total = 0.15 →
    duck / total = 0.30 →
    total > 0 →
    geese / (total - swan) = 0.4375 :=
by
  sorry

end NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l3143_314312


namespace NUMINAMATH_CALUDE_trajectory_of_T_l3143_314351

-- Define the curve C
def C (x y : ℝ) : Prop := 4 * x^2 - y + 1 = 0

-- Define the fixed point M
def M : ℝ × ℝ := (-2, 0)

-- Define the relationship between A, T, and M
def AT_TM_relation (A T : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xt, yt) := T
  (xa - xt, ya - yt) = (2 * (-2 - xt), 2 * (-yt))

-- Theorem statement
theorem trajectory_of_T (A T : ℝ × ℝ) :
  (∃ x y, A = (x, y) ∧ C x y) →  -- A is on curve C
  AT_TM_relation A T →           -- Relationship between A, T, and M holds
  4 * (3 * T.1 + 4)^2 - 3 * T.2 + 1 = 0 :=  -- Trajectory equation for T
by sorry

end NUMINAMATH_CALUDE_trajectory_of_T_l3143_314351


namespace NUMINAMATH_CALUDE_a₉₉_eq_182_l3143_314352

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  -- First term
  a₁ : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first 17 terms is 34
  sum_17 : 17 * a₁ + (17 * 16 / 2) * d = 34
  -- Third term is -10
  a₃ : a₁ + 2 * d = -10

/-- The 99th term of the arithmetic sequence -/
def a₉₉ (seq : ArithmeticSequence) : ℝ := seq.a₁ + 98 * seq.d

/-- Theorem stating that a₉₉ = 182 for the given arithmetic sequence -/
theorem a₉₉_eq_182 (seq : ArithmeticSequence) : a₉₉ seq = 182 := by
  sorry

end NUMINAMATH_CALUDE_a₉₉_eq_182_l3143_314352


namespace NUMINAMATH_CALUDE_negation_of_existence_l3143_314356

theorem negation_of_existence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l3143_314356


namespace NUMINAMATH_CALUDE_gcd_sum_ten_l3143_314320

theorem gcd_sum_ten (n : ℕ) : 
  (Nat.gcd 6 n + Nat.gcd 8 (2 * n) = 10) ↔ 
  (∃ t : ℕ, n = 12 * t + 4 ∨ n = 12 * t + 6 ∨ n = 12 * t + 8) :=
sorry

end NUMINAMATH_CALUDE_gcd_sum_ten_l3143_314320


namespace NUMINAMATH_CALUDE_spring_decrease_percentage_l3143_314327

theorem spring_decrease_percentage (initial_increase : ℝ) (total_change : ℝ) : 
  initial_increase = 0.05 →
  total_change = -0.1495 →
  ∃ spring_decrease : ℝ, 
    (1 + initial_increase) * (1 - spring_decrease) = 1 + total_change ∧
    spring_decrease = 0.19 :=
by sorry

end NUMINAMATH_CALUDE_spring_decrease_percentage_l3143_314327


namespace NUMINAMATH_CALUDE_min_distance_for_three_coloring_l3143_314397

-- Define the set of points in and on the regular hexagon
def hexagon_points : Set (ℝ × ℝ) := sorry

-- Define the distance function between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define a valid three-coloring scheme
def valid_three_coloring (r : ℝ) : Prop := 
  ∃ (coloring : (ℝ × ℝ) → Fin 3),
    ∀ (p q : hexagon_points), 
      coloring p = coloring q → distance p q < r

-- The main theorem
theorem min_distance_for_three_coloring : 
  (∀ r < 3/2, ¬ valid_three_coloring r) ∧ 
  valid_three_coloring (3/2) := by sorry

end NUMINAMATH_CALUDE_min_distance_for_three_coloring_l3143_314397


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l3143_314340

/-- The rate per kg of mangoes given the purchase details --/
def mango_rate (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (total_payment : ℕ) : ℕ :=
  (total_payment - grape_quantity * grape_rate) / mango_quantity

theorem mango_rate_calculation :
  mango_rate 9 70 9 1125 = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l3143_314340


namespace NUMINAMATH_CALUDE_fish_in_tank_l3143_314324

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  3 * blue = total →
  2 * spotted = blue →
  spotted = 5 →
  total = 30 := by
sorry

end NUMINAMATH_CALUDE_fish_in_tank_l3143_314324


namespace NUMINAMATH_CALUDE_candidate_a_republican_voters_l3143_314376

theorem candidate_a_republican_voters (total : ℝ) (h_total_pos : total > 0) : 
  let dem_percent : ℝ := 0.7
  let rep_percent : ℝ := 1 - dem_percent
  let dem_for_a_percent : ℝ := 0.8
  let total_for_a_percent : ℝ := 0.65
  let rep_for_a_percent : ℝ := 
    (total_for_a_percent - dem_percent * dem_for_a_percent) / rep_percent
  rep_for_a_percent = 0.3 := by
sorry

end NUMINAMATH_CALUDE_candidate_a_republican_voters_l3143_314376


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3143_314346

/-- The y-coordinate of the vertex of the parabola y = -3x^2 - 30x - 81 is -6 -/
theorem parabola_vertex_y_coordinate :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 - 30 * x - 81
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≤ f x₀ ∧ f x₀ = -6 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3143_314346


namespace NUMINAMATH_CALUDE_max_handshakes_l3143_314375

theorem max_handshakes (N : ℕ) (h1 : N > 4) : ∃ (max_shaken : ℕ),
  (∃ (not_shaken : Fin N → Prop),
    (∃ (a b : Fin N), a ≠ b ∧ not_shaken a ∧ not_shaken b ∧
      ∀ (x : Fin N), not_shaken x → (x = a ∨ x = b)) ∧
    (∀ (x : Fin N), ¬(not_shaken x) →
      ∀ (y : Fin N), y ≠ x → ∃ (shaken : Prop), shaken)) ∧
  max_shaken = N - 2 ∧
  ∀ (k : ℕ), k > max_shaken →
    ¬(∃ (not_shaken : Fin N → Prop),
      (∃ (a b : Fin N), a ≠ b ∧ not_shaken a ∧ not_shaken b ∧
        ∀ (x : Fin N), not_shaken x → (x = a ∨ x = b)) ∧
      (∀ (x : Fin N), ¬(not_shaken x) →
        ∀ (y : Fin N), y ≠ x → ∃ (shaken : Prop), shaken))
  := by sorry

end NUMINAMATH_CALUDE_max_handshakes_l3143_314375


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3143_314362

theorem polynomial_simplification (x : ℝ) : (x^2 - 4) * (x - 2) * (x + 2) = x^4 - 8*x^2 + 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3143_314362


namespace NUMINAMATH_CALUDE_scenic_spot_probabilities_l3143_314353

def total_spots : ℕ := 10
def five_a_spots : ℕ := 4
def four_a_spots : ℕ := 6

def spots_after_yuntai : ℕ := 4

theorem scenic_spot_probabilities :
  (five_a_spots : ℚ) / total_spots = 2 / 5 ∧
  (2 : ℚ) / (spots_after_yuntai * (spots_after_yuntai - 1)) = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_scenic_spot_probabilities_l3143_314353


namespace NUMINAMATH_CALUDE_linear_function_value_l3143_314367

/-- A linear function f(x) = px + q -/
def f (p q : ℝ) (x : ℝ) : ℝ := p * x + q

theorem linear_function_value (p q : ℝ) :
  f p q 3 = 5 → f p q 5 = 9 → f p q 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_value_l3143_314367


namespace NUMINAMATH_CALUDE_investment_interest_rate_proof_l3143_314323

/-- Proves that for an investment of 7000 over 2 years, if the interest earned is 840 more than
    what would be earned at 12% p.a., then the interest rate is 18% p.a. -/
theorem investment_interest_rate_proof 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_diff : ℝ) 
  (base_rate : ℝ) 
  (h1 : principal = 7000)
  (h2 : time = 2)
  (h3 : interest_diff = 840)
  (h4 : base_rate = 12)
  (h5 : principal * (rate / 100) * time - principal * (base_rate / 100) * time = interest_diff) :
  rate = 18 := by
  sorry

#check investment_interest_rate_proof

end NUMINAMATH_CALUDE_investment_interest_rate_proof_l3143_314323


namespace NUMINAMATH_CALUDE_pythagorean_triple_identity_l3143_314394

theorem pythagorean_triple_identity (n : ℕ+) :
  (2 * n + 1) ^ 2 + (2 * n ^ 2 + 2 * n) ^ 2 = (2 * n ^ 2 + 2 * n + 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identity_l3143_314394


namespace NUMINAMATH_CALUDE_company_average_salary_l3143_314309

theorem company_average_salary
  (num_managers : ℕ)
  (num_associates : ℕ)
  (avg_salary_managers : ℚ)
  (avg_salary_associates : ℚ)
  (h1 : num_managers = 15)
  (h2 : num_associates = 75)
  (h3 : avg_salary_managers = 90000)
  (h4 : avg_salary_associates = 30000) :
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates) = 40000 := by
sorry

end NUMINAMATH_CALUDE_company_average_salary_l3143_314309


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3143_314331

/-- Given a hyperbola with equation x^2 - y^2/m^2 = 1 where m > 0,
    if one of its asymptotes is x + √3 * y = 0, then m = √3/3 -/
theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) 
  (h2 : ∃ (x y : ℝ), x^2 - y^2/m^2 = 1 ∧ x + Real.sqrt 3 * y = 0) : 
  m = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3143_314331


namespace NUMINAMATH_CALUDE_parrot_female_fraction_l3143_314383

theorem parrot_female_fraction (total_birds : ℝ) (female_parrot_fraction : ℝ) : 
  (3 / 5 : ℝ) * total_birds +                   -- number of parrots
  (2 / 5 : ℝ) * total_birds =                   -- number of toucans
  total_birds ∧                                 -- total number of birds
  (3 / 4 : ℝ) * ((2 / 5 : ℝ) * total_birds) +   -- number of female toucans
  female_parrot_fraction * ((3 / 5 : ℝ) * total_birds) = -- number of female parrots
  (1 / 2 : ℝ) * total_birds →                   -- total number of female birds
  female_parrot_fraction = (1 / 3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_parrot_female_fraction_l3143_314383


namespace NUMINAMATH_CALUDE_max_savings_is_90_l3143_314310

structure Airline where
  name : String
  originalPrice : ℕ
  discountPercentage : ℕ

def calculateDiscountedPrice (airline : Airline) : ℕ :=
  airline.originalPrice - (airline.originalPrice * airline.discountPercentage / 100)

def airlines : List Airline := [
  { name := "Delta", originalPrice := 850, discountPercentage := 20 },
  { name := "United", originalPrice := 1100, discountPercentage := 30 },
  { name := "American", originalPrice := 950, discountPercentage := 25 },
  { name := "Southwest", originalPrice := 900, discountPercentage := 15 },
  { name := "JetBlue", originalPrice := 1200, discountPercentage := 40 }
]

theorem max_savings_is_90 :
  let discountedPrices := airlines.map calculateDiscountedPrice
  let cheapestPrice := discountedPrices.minimum?
  let maxSavings := discountedPrices.map (fun price => price - cheapestPrice.getD 0)
  maxSavings.maximum? = some 90 := by
  sorry

end NUMINAMATH_CALUDE_max_savings_is_90_l3143_314310


namespace NUMINAMATH_CALUDE_g_value_at_negative_1001_l3143_314304

/-- A function g satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x

theorem g_value_at_negative_1001 (g : ℝ → ℝ) 
    (h1 : FunctionalEquation g) (h2 : g 1 = 3) : g (-1001) = 1005 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_negative_1001_l3143_314304


namespace NUMINAMATH_CALUDE_existence_of_non_coprime_pair_l3143_314339

theorem existence_of_non_coprime_pair :
  ∃ m : ℤ, (Nat.gcd (100 + 101 * m).natAbs (101 - 100 * m).natAbs) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_coprime_pair_l3143_314339


namespace NUMINAMATH_CALUDE_stairs_calculation_l3143_314349

/-- The number of stairs run up and down one way during a football team's exercise routine. -/
def stairs_one_way : ℕ := 32

/-- The number of times players run up and down the bleachers. -/
def num_runs : ℕ := 40

/-- The number of calories burned per stair. -/
def calories_per_stair : ℕ := 2

/-- The total number of calories burned during the exercise. -/
def total_calories_burned : ℕ := 5120

/-- Theorem stating that the number of stairs run up and down one way is 32,
    given the conditions of the exercise routine. -/
theorem stairs_calculation :
  stairs_one_way = 32 ∧
  num_runs * (2 * stairs_one_way) * calories_per_stair = total_calories_burned :=
by sorry

end NUMINAMATH_CALUDE_stairs_calculation_l3143_314349


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l3143_314316

theorem choose_four_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 4 → Nat.choose n k = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l3143_314316


namespace NUMINAMATH_CALUDE_heart_ratio_theorem_l3143_314368

def heart (n m : ℕ) : ℕ := n^3 + m^2

theorem heart_ratio_theorem : (heart 3 5 : ℚ) / (heart 5 3 : ℚ) = 26 / 67 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_theorem_l3143_314368


namespace NUMINAMATH_CALUDE_max_area_quadrilateral_in_circle_l3143_314348

theorem max_area_quadrilateral_in_circle (d : Real) 
  (h1 : 0 ≤ d) (h2 : d < 1) : 
  ∃ (max_area : Real),
    (d < Real.sqrt 2 / 2 → max_area = 2 * Real.sqrt (1 - d^2)) ∧
    (Real.sqrt 2 / 2 ≤ d → max_area = 1 / d) ∧
    ∀ (area : Real), area ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_max_area_quadrilateral_in_circle_l3143_314348


namespace NUMINAMATH_CALUDE_custom_mult_theorem_l3143_314300

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := (a - b) ^ 2

/-- Theorem stating that ((x-y)^2 + 1) * ((y-x)^2 + 1) = 0 for the custom multiplication -/
theorem custom_mult_theorem (x y : ℝ) : 
  custom_mult ((x - y) ^ 2 + 1) ((y - x) ^ 2 + 1) = 0 := by
  sorry


end NUMINAMATH_CALUDE_custom_mult_theorem_l3143_314300


namespace NUMINAMATH_CALUDE_soccer_league_games_l3143_314393

/-- Calculates the total number of games in a soccer league --/
def total_games (n : ℕ) (k : ℕ) : ℕ :=
  -- Regular season games
  n * (n - 1) +
  -- Playoff games (single elimination format)
  (k - 1)

/-- Theorem stating the total number of games in the soccer league --/
theorem soccer_league_games :
  total_games 20 8 = 767 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l3143_314393


namespace NUMINAMATH_CALUDE_given_expression_is_proper_algebraic_notation_l3143_314378

/-- A predicate that determines if an expression meets the requirements for algebraic notation -/
def is_proper_algebraic_notation (expression : String) : Prop := 
  expression = "(3πm)/4"

/-- The given expression -/
def given_expression : String := "(3πm)/4"

/-- Theorem stating that the given expression meets the requirements for algebraic notation -/
theorem given_expression_is_proper_algebraic_notation : 
  is_proper_algebraic_notation given_expression := by
  sorry

end NUMINAMATH_CALUDE_given_expression_is_proper_algebraic_notation_l3143_314378


namespace NUMINAMATH_CALUDE_descending_order_exists_l3143_314301

theorem descending_order_exists (x y z : ℤ) : ∃ (a b c : ℤ), 
  ({a, b, c} : Finset ℤ) = {x, y, z} ∧ a ≥ b ∧ b ≥ c := by sorry

end NUMINAMATH_CALUDE_descending_order_exists_l3143_314301


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l3143_314359

theorem percentage_failed_hindi (failed_english : Real) (failed_both : Real) (passed_both : Real)
  (h1 : failed_english = 48)
  (h2 : failed_both = 27)
  (h3 : passed_both = 54) :
  failed_english + (100 - passed_both) - failed_both = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l3143_314359


namespace NUMINAMATH_CALUDE_tree_planting_theorem_l3143_314389

/-- Represents the tree planting activity -/
structure TreePlanting where
  totalVolunteers : ℕ
  poplars : ℕ
  seaBuckthorns : ℕ
  poplarTime : ℚ
  seaBuckthornsTime1 : ℚ
  seaBuckthornsTime2 : ℚ
  transferredVolunteers : ℕ

/-- Calculates the optimal allocation and durations for the tree planting activity -/
def optimalAllocation (tp : TreePlanting) : 
  (ℕ × ℕ) × ℚ × ℚ :=
  sorry

/-- The theorem stating the correctness of the optimal allocation and durations -/
theorem tree_planting_theorem (tp : TreePlanting) 
  (h1 : tp.totalVolunteers = 52)
  (h2 : tp.poplars = 150)
  (h3 : tp.seaBuckthorns = 200)
  (h4 : tp.poplarTime = 2/5)
  (h5 : tp.seaBuckthornsTime1 = 1/2)
  (h6 : tp.seaBuckthornsTime2 = 2/3)
  (h7 : tp.transferredVolunteers = 6) :
  let (allocation, initialDuration, finalDuration) := optimalAllocation tp
  allocation = (20, 32) ∧ 
  initialDuration = 25/8 ∧
  finalDuration = 27/7 :=
sorry

end NUMINAMATH_CALUDE_tree_planting_theorem_l3143_314389


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l3143_314354

/-- Represents the revenue from full-price tickets in a charity event -/
def revenue_full_price (full_price : ℝ) (num_full_price : ℝ) : ℝ :=
  full_price * num_full_price

/-- Represents the revenue from discounted tickets in a charity event -/
def revenue_discounted (full_price : ℝ) (num_discounted : ℝ) : ℝ :=
  0.75 * full_price * num_discounted

/-- Theorem stating that the revenue from full-price tickets can be determined -/
theorem charity_ticket_revenue 
  (full_price : ℝ) 
  (num_full_price num_discounted : ℝ) 
  (h1 : num_full_price + num_discounted = 150)
  (h2 : revenue_full_price full_price num_full_price + 
        revenue_discounted full_price num_discounted = 2250)
  : ∃ (r : ℝ), revenue_full_price full_price num_full_price = r :=
by
  sorry


end NUMINAMATH_CALUDE_charity_ticket_revenue_l3143_314354


namespace NUMINAMATH_CALUDE_specific_device_works_prob_l3143_314334

/-- A device with two components, each having a probability of failure --/
structure Device where
  component_failure_prob : ℝ
  num_components : ℕ

/-- The probability that the device works --/
def device_works_prob (d : Device) : ℝ :=
  (1 - d.component_failure_prob) ^ d.num_components

/-- Theorem: The probability that a specific device works is 0.81 --/
theorem specific_device_works_prob :
  ∃ (d : Device), device_works_prob d = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_specific_device_works_prob_l3143_314334


namespace NUMINAMATH_CALUDE_distance_to_charlie_l3143_314330

/-- The vertical distance Annie and Barbara walk together to reach Charlie -/
theorem distance_to_charlie 
  (annie_x annie_y barbara_x barbara_y charlie_x charlie_y : ℚ) : 
  annie_x = 6 → 
  annie_y = -20 → 
  barbara_x = 1 → 
  barbara_y = 14 → 
  charlie_x = 7/2 → 
  charlie_y = 2 → 
  charlie_y - (annie_y + barbara_y) / 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_distance_to_charlie_l3143_314330


namespace NUMINAMATH_CALUDE_min_blue_eyes_and_backpack_proof_l3143_314380

def min_blue_eyes_and_backpack (total_students blue_eyes backpacks glasses : ℕ) : ℕ :=
  blue_eyes - (total_students - backpacks)

theorem min_blue_eyes_and_backpack_proof 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (backpacks : ℕ) 
  (glasses : ℕ) 
  (h1 : total_students = 35)
  (h2 : blue_eyes = 18)
  (h3 : backpacks = 25)
  (h4 : glasses = 10)
  (h5 : ∃ (x : ℕ), x ≥ 2 ∧ x ≤ glasses ∧ x ≤ blue_eyes) :
  min_blue_eyes_and_backpack total_students blue_eyes backpacks glasses = 10 := by
  sorry

#eval min_blue_eyes_and_backpack 35 18 25 10

end NUMINAMATH_CALUDE_min_blue_eyes_and_backpack_proof_l3143_314380


namespace NUMINAMATH_CALUDE_roots_sum_product_l3143_314373

theorem roots_sum_product (a b : ℝ) : 
  (∀ x : ℝ, x^4 - 6*x - 1 = 0 ↔ x = a ∨ x = b) →
  a*b + 2*a + 2*b = 1.5 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_product_l3143_314373


namespace NUMINAMATH_CALUDE_minimum_mass_for_upward_roll_l3143_314305

/-- Given a cylinder of mass M on rails inclined at angle α = 45°, 
    the minimum mass m of a weight attached to a string wound around the cylinder 
    for it to roll upward without slipping is M(√2 + 1) -/
theorem minimum_mass_for_upward_roll (M : ℝ) (α : ℝ) 
    (h_α : α = π / 4) : 
    ∃ m : ℝ, m = M * (Real.sqrt 2 + 1) ∧ 
    m * (1 - Real.sin α) = M * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_minimum_mass_for_upward_roll_l3143_314305


namespace NUMINAMATH_CALUDE_max_product_sum_2004_l3143_314307

theorem max_product_sum_2004 :
  (∃ (a b : ℤ), a + b = 2004 ∧ a * b = 1004004) ∧
  (∀ (x y : ℤ), x + y = 2004 → x * y ≤ 1004004) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2004_l3143_314307


namespace NUMINAMATH_CALUDE_equal_number_of_boys_and_girls_l3143_314371

/-- Represents a school with boys and girls -/
structure School where
  boys : ℕ
  girls : ℕ
  boys_age_sum : ℕ
  girls_age_sum : ℕ

/-- The average age of boys -/
def boys_avg (s : School) : ℚ := s.boys_age_sum / s.boys

/-- The average age of girls -/
def girls_avg (s : School) : ℚ := s.girls_age_sum / s.girls

/-- The average age of all students -/
def total_avg (s : School) : ℚ := (s.boys_age_sum + s.girls_age_sum) / (s.boys + s.girls)

/-- The theorem stating that the number of boys equals the number of girls -/
theorem equal_number_of_boys_and_girls (s : School) 
  (h1 : boys_avg s ≠ girls_avg s) 
  (h2 : (boys_avg s + girls_avg s) / 2 = total_avg s) : 
  s.boys = s.girls := by sorry

end NUMINAMATH_CALUDE_equal_number_of_boys_and_girls_l3143_314371


namespace NUMINAMATH_CALUDE_parking_spaces_remaining_l3143_314328

theorem parking_spaces_remaining (total_spaces : ℕ) (caravan_spaces : ℕ) (parked_caravans : ℕ) : 
  total_spaces = 30 → caravan_spaces = 2 → parked_caravans = 3 → 
  total_spaces - (caravan_spaces * parked_caravans) = 24 := by
  sorry

end NUMINAMATH_CALUDE_parking_spaces_remaining_l3143_314328


namespace NUMINAMATH_CALUDE_f_max_min_range_l3143_314308

/-- A function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*((a+2)*x+1)

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- Theorem stating the range of a for which f has both a maximum and minimum -/
theorem f_max_min_range (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), ∀ (x : ℝ), f a x₁ ≤ f a x ∧ f a x ≤ f a x₂) →
  a < -1 ∨ a > 2 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_range_l3143_314308


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3143_314344

theorem polynomial_divisibility (a b c d : ℤ) :
  (∃ n : ℤ, 5 ∣ (a * n^3 + b * n^2 + c * n + d)) →
  ¬(5 ∣ d) →
  ∃ m : ℤ, 5 ∣ (a + b * m + c * m^2 + d * m^3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3143_314344


namespace NUMINAMATH_CALUDE_prob_from_third_farm_given_over_300kg_l3143_314335

/-- Represents the three farms supplying calves -/
inductive Farm : Type
  | first : Farm
  | second : Farm
  | third : Farm

/-- The proportion of calves from each farm -/
def farm_proportion : Farm → ℝ
  | Farm.first => 0.6
  | Farm.second => 0.3
  | Farm.third => 0.1

/-- The probability that a calf from a given farm weighs over 300 kg -/
def prob_over_300kg : Farm → ℝ
  | Farm.first => 0.15
  | Farm.second => 0.25
  | Farm.third => 0.35

/-- The probability that a randomly selected calf weighing over 300 kg came from the third farm -/
theorem prob_from_third_farm_given_over_300kg : 
  (farm_proportion Farm.third * prob_over_300kg Farm.third) / 
  (farm_proportion Farm.first * prob_over_300kg Farm.first + 
   farm_proportion Farm.second * prob_over_300kg Farm.second + 
   farm_proportion Farm.third * prob_over_300kg Farm.third) = 0.175 := by
  sorry

end NUMINAMATH_CALUDE_prob_from_third_farm_given_over_300kg_l3143_314335


namespace NUMINAMATH_CALUDE_part_one_part_two_l3143_314333

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := {x | a - b < x ∧ x < a + b}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Part 1
theorem part_one (a : ℝ) : 
  (A a 1 ∩ B = A a 1) → (a ≤ -2 ∨ a ≥ 6) := by sorry

-- Part 2
theorem part_two (b : ℝ) :
  (A 1 b ∩ B = ∅) → (b ≤ 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3143_314333


namespace NUMINAMATH_CALUDE_arrangement_proof_l3143_314345

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

def num_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  factorial n / factorial k

theorem arrangement_proof :
  let n : ℕ := 6  -- Total number of objects
  let k : ℕ := 2  -- Number of identical objects
  num_arrangements n k = 360 := by
sorry

end NUMINAMATH_CALUDE_arrangement_proof_l3143_314345


namespace NUMINAMATH_CALUDE_canteen_banana_requirement_l3143_314321

/-- The number of bananas required for the given period -/
def total_bananas : ℕ := 9828

/-- The number of weeks in the given period -/
def num_weeks : ℕ := 9

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of bananas in a dozen -/
def bananas_per_dozen : ℕ := 12

/-- Theorem: The canteen needs 13 dozens of bananas per day -/
theorem canteen_banana_requirement :
  (total_bananas / (num_weeks * days_per_week)) / bananas_per_dozen = 13 := by
  sorry

end NUMINAMATH_CALUDE_canteen_banana_requirement_l3143_314321


namespace NUMINAMATH_CALUDE_bernoulli_zero_success_l3143_314398

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The probability of failure in each trial -/
def q : ℚ := 1 - p

/-- The number of successes we're interested in -/
def k : ℕ := 0

/-- 
Theorem: In a series of 7 Bernoulli trials with a success probability of 2/7, 
the probability of 0 successes is equal to (5/7)^7.
-/
theorem bernoulli_zero_success : 
  (n.choose k) * p^k * q^(n-k) = (5/7)^7 := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_zero_success_l3143_314398


namespace NUMINAMATH_CALUDE_josh_siblings_count_josh_has_three_siblings_l3143_314374

/-- Calculates the number of Josh's siblings given the candy distribution scenario. -/
theorem josh_siblings_count : ℕ → Prop := fun s =>
  let initial_candies : ℕ := 100
  let candies_per_sibling : ℕ := 10
  let candies_for_self : ℕ := 16
  let candies_left : ℕ := 19
  
  let remaining_after_siblings : ℕ := initial_candies - s * candies_per_sibling
  let remaining_after_best_friend : ℕ := remaining_after_siblings / 2
  let final_remaining : ℕ := remaining_after_best_friend - candies_for_self
  
  final_remaining = candies_left → s = 3

/-- Proves that Josh has exactly 3 siblings. -/
theorem josh_has_three_siblings : ∃ s, josh_siblings_count s :=
  sorry

end NUMINAMATH_CALUDE_josh_siblings_count_josh_has_three_siblings_l3143_314374


namespace NUMINAMATH_CALUDE_money_difference_equals_5p_minus_20_l3143_314322

/-- The number of pennies in a nickel -/
def nickel_value : ℕ := 5

/-- The number of nickels Jessica has -/
def jessica_nickels (p : ℕ) : ℕ := 3 * p + 2

/-- The number of nickels Samantha has -/
def samantha_nickels (p : ℕ) : ℕ := 2 * p + 6

/-- The difference in money (in pennies) between Jessica and Samantha -/
def money_difference (p : ℕ) : ℤ :=
  nickel_value * (jessica_nickels p - samantha_nickels p)

theorem money_difference_equals_5p_minus_20 (p : ℕ) :
  money_difference p = 5 * p - 20 := by sorry

end NUMINAMATH_CALUDE_money_difference_equals_5p_minus_20_l3143_314322


namespace NUMINAMATH_CALUDE_consecutive_non_sum_of_squares_l3143_314326

theorem consecutive_non_sum_of_squares :
  ∃ m : ℕ+, ∀ k : Fin 2017, ¬∃ a b : ℤ, (m + k : ℤ) = a^2 + b^2 :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_sum_of_squares_l3143_314326


namespace NUMINAMATH_CALUDE_colonization_combinations_l3143_314381

def total_planets : ℕ := 15
def earth_like_planets : ℕ := 8
def mars_like_planets : ℕ := 7
def earth_like_cost : ℕ := 3
def mars_like_cost : ℕ := 1
def total_colonization_units : ℕ := 18

def valid_combination (earth_colonies mars_colonies : ℕ) : Prop :=
  earth_colonies ≤ earth_like_planets ∧
  mars_colonies ≤ mars_like_planets ∧
  earth_colonies * earth_like_cost + mars_colonies * mars_like_cost = total_colonization_units

def count_combinations : ℕ := sorry

theorem colonization_combinations :
  count_combinations = 2478 :=
sorry

end NUMINAMATH_CALUDE_colonization_combinations_l3143_314381


namespace NUMINAMATH_CALUDE_complex_division_l3143_314314

theorem complex_division (i : ℂ) (h : i * i = -1) : 
  (2 - i) / (1 + i) = 1/2 - 3/2 * i := by sorry

end NUMINAMATH_CALUDE_complex_division_l3143_314314


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3143_314358

theorem sum_of_cubes_of_roots (x₁ x₂ x₃ : ℂ) : 
  x₁^3 + x₂^3 + x₃^3 = 0 → x₁ + x₂ + x₃ = -2 → x₁*x₂ + x₂*x₃ + x₃*x₁ = 1 → x₁*x₂*x₃ = 3 → 
  x₁^3 + x₂^3 + x₃^3 = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3143_314358


namespace NUMINAMATH_CALUDE_find_number_l3143_314355

theorem find_number : ∃ x : ℝ, x / 2 = 9 ∧ x = 18 := by sorry

end NUMINAMATH_CALUDE_find_number_l3143_314355


namespace NUMINAMATH_CALUDE_ratio_of_system_l3143_314377

theorem ratio_of_system (x y a b : ℝ) (h1 : 4 * x - 2 * y = a) (h2 : 6 * y - 12 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_system_l3143_314377


namespace NUMINAMATH_CALUDE_candies_in_packet_l3143_314370

/-- The number of candies in a packet of candy -/
def candies_per_packet : ℕ := 18

/-- The number of packets Bobby buys -/
def num_packets : ℕ := 2

/-- The number of days per week Bobby eats 2 candies -/
def days_eating_two : ℕ := 5

/-- The number of days per week Bobby eats 1 candy -/
def days_eating_one : ℕ := 2

/-- The number of weeks it takes to finish the packets -/
def weeks_to_finish : ℕ := 3

/-- Theorem stating the number of candies in a packet -/
theorem candies_in_packet :
  candies_per_packet * num_packets = 
  (days_eating_two * 2 + days_eating_one * 1) * weeks_to_finish :=
by sorry

end NUMINAMATH_CALUDE_candies_in_packet_l3143_314370


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3143_314386

theorem complex_number_quadrant : ∃ (z : ℂ), z = (I : ℂ) / (Real.sqrt 3 - 3 * I) ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3143_314386


namespace NUMINAMATH_CALUDE_min_value_sequence_l3143_314384

theorem min_value_sequence (a : ℕ → ℝ) (h1 : a 1 = 25) 
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, n ≥ 1 → a n / n ≥ 9 ∧ ∃ m : ℕ, m ≥ 1 ∧ a m / m = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_sequence_l3143_314384


namespace NUMINAMATH_CALUDE_koi_fish_count_l3143_314390

/-- Calculates the number of koi fish after 3 weeks given the initial conditions --/
def koi_fish_after_three_weeks (initial_total : ℕ) (koi_added_per_day : ℕ) (goldfish_added_per_day : ℕ) (days : ℕ) (final_goldfish : ℕ) : ℕ :=
  let total_added := (koi_added_per_day + goldfish_added_per_day) * days
  let final_total := initial_total + total_added
  final_total - final_goldfish

/-- Theorem stating that the number of koi fish after 3 weeks is 227 --/
theorem koi_fish_count : koi_fish_after_three_weeks 280 2 5 21 200 = 227 := by
  sorry

end NUMINAMATH_CALUDE_koi_fish_count_l3143_314390


namespace NUMINAMATH_CALUDE_prism_volume_sum_l3143_314302

theorem prism_volume_sum (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  Nat.lcm a b = 72 →
  Nat.lcm a c = 24 →
  Nat.lcm b c = 18 →
  (∃ (a_min b_min c_min a_max b_max c_max : ℕ),
    (∀ a' b' c' : ℕ, 
      Nat.lcm a' b' = 72 → Nat.lcm a' c' = 24 → Nat.lcm b' c' = 18 →
      a' * b' * c' ≥ a_min * b_min * c_min) ∧
    (∀ a' b' c' : ℕ, 
      Nat.lcm a' b' = 72 → Nat.lcm a' c' = 24 → Nat.lcm b' c' = 18 →
      a' * b' * c' ≤ a_max * b_max * c_max) ∧
    a_min * b_min * c_min + a_max * b_max * c_max = 3024) := by
  sorry

#check prism_volume_sum

end NUMINAMATH_CALUDE_prism_volume_sum_l3143_314302


namespace NUMINAMATH_CALUDE_set_subset_relations_l3143_314366

theorem set_subset_relations : 
  ({1,2,3} : Set ℕ) ⊆ {1,2,3} ∧ (∅ : Set ℕ) ⊆ {1} := by sorry

end NUMINAMATH_CALUDE_set_subset_relations_l3143_314366


namespace NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l3143_314369

theorem roots_of_cubic_polynomial :
  let p : ℝ → ℝ := λ x => x^3 + x^2 - 6*x - 6
  (∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 3 ∨ x = -2) ∧
  (p (-1) = 0) ∧ (p 3 = 0) ∧ (p (-2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l3143_314369


namespace NUMINAMATH_CALUDE_f_minus_g_greater_than_two_l3143_314361

noncomputable def f (x : ℝ) : ℝ := (2 - x^3) * Real.exp x

noncomputable def g (x : ℝ) : ℝ := Real.log x / x

theorem f_minus_g_greater_than_two (x : ℝ) (h : x ∈ Set.Ioo 0 1) : f x - g x > 2 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_g_greater_than_two_l3143_314361


namespace NUMINAMATH_CALUDE_valentino_farm_birds_l3143_314303

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens ducks turkeys : ℕ) : ℕ :=
  chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  ∀ (chickens ducks turkeys : ℕ),
    chickens = 200 →
    ducks = 2 * chickens →
    turkeys = 3 * ducks →
    total_birds chickens ducks turkeys = 1800 := by
  sorry

end NUMINAMATH_CALUDE_valentino_farm_birds_l3143_314303


namespace NUMINAMATH_CALUDE_square_equation_solution_l3143_314395

theorem square_equation_solution (x : ℝ) : (x - 1)^2 = 4 → x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3143_314395


namespace NUMINAMATH_CALUDE_laundry_charge_calculation_l3143_314347

/-- The amount charged per kilo of laundry -/
def charge_per_kilo : ℝ := sorry

/-- The number of kilos washed two days ago -/
def kilos_two_days_ago : ℝ := 5

/-- The number of kilos washed yesterday -/
def kilos_yesterday : ℝ := kilos_two_days_ago + 5

/-- The number of kilos washed today -/
def kilos_today : ℝ := 2 * kilos_yesterday

/-- The total earnings for three days -/
def total_earnings : ℝ := 70

theorem laundry_charge_calculation :
  charge_per_kilo * (kilos_two_days_ago + kilos_yesterday + kilos_today) = total_earnings ∧
  charge_per_kilo = 2 := by sorry

end NUMINAMATH_CALUDE_laundry_charge_calculation_l3143_314347


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l3143_314387

/-- The focus of the parabola y = 2x^2 has coordinates (0, 1/8) -/
theorem parabola_focus_coordinates :
  let f : ℝ → ℝ := λ x => 2 * x^2
  ∃ (focus : ℝ × ℝ), focus = (0, 1/8) ∧
    ∀ (x y : ℝ), y = f x → 
      (x - focus.1)^2 + (y - focus.2)^2 = (y - focus.2 + 1/4)^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l3143_314387


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l3143_314399

theorem dormitory_to_city_distance :
  ∀ (D : ℝ),
  (1/3 : ℝ) * D + (3/5 : ℝ) * D + 2 = D →
  D = 30 := by
sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l3143_314399


namespace NUMINAMATH_CALUDE_village_population_equality_l3143_314329

/-- The initial population of Village X -/
def Px : ℕ := sorry

/-- The yearly decrease in population of Village X -/
def decrease_x : ℕ := 1200

/-- The initial population of Village Y -/
def Py : ℕ := 42000

/-- The yearly increase in population of Village Y -/
def increase_y : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years : ℕ := 17

theorem village_population_equality :
  Px - years * decrease_x = Py + years * increase_y ∧ Px = 76000 := by sorry

end NUMINAMATH_CALUDE_village_population_equality_l3143_314329


namespace NUMINAMATH_CALUDE_circle_diameter_l3143_314319

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 64 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l3143_314319


namespace NUMINAMATH_CALUDE_sahara_temperature_difference_l3143_314357

/-- The maximum temperature difference in the Sahara Desert --/
theorem sahara_temperature_difference (highest_temp lowest_temp : ℤ) 
  (h_highest : highest_temp = 58)
  (h_lowest : lowest_temp = -34) :
  highest_temp - lowest_temp = 92 := by
  sorry

end NUMINAMATH_CALUDE_sahara_temperature_difference_l3143_314357


namespace NUMINAMATH_CALUDE_smallest_n_guarantee_same_length_l3143_314343

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2017

/-- The number of distinct diagonal lengths from a single vertex -/
def distinct_lengths : ℕ := (n - 3) / 2

/-- The smallest number of diagonals to guarantee two of the same length -/
def smallest_n : ℕ := distinct_lengths + 1

theorem smallest_n_guarantee_same_length :
  smallest_n = 1008 := by sorry

end NUMINAMATH_CALUDE_smallest_n_guarantee_same_length_l3143_314343


namespace NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l3143_314365

def first_four_composite_numbers : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_first_four_composites :
  units_digit (product_of_list first_four_composite_numbers) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l3143_314365


namespace NUMINAMATH_CALUDE_min_rubles_to_win_l3143_314385

/-- Represents the state of the game machine -/
structure GameState :=
  (score : ℕ)
  (rubles_spent : ℕ)

/-- Defines the possible moves in the game -/
inductive Move
| insert_one : Move
| insert_two : Move

/-- Applies a move to the current game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.insert_one => ⟨state.score + 1, state.rubles_spent + 1⟩
  | Move.insert_two => ⟨state.score * 2, state.rubles_spent + 2⟩

/-- Checks if the game state is valid (score ≤ 50) -/
def is_valid_state (state : GameState) : Prop :=
  state.score ≤ 50

/-- Checks if the game is won (score = 50) -/
def is_winning_state (state : GameState) : Prop :=
  state.score = 50

/-- The main theorem to prove -/
theorem min_rubles_to_win :
  ∃ (moves : List Move),
    let final_state := moves.foldl apply_move ⟨0, 0⟩
    is_valid_state final_state ∧
    is_winning_state final_state ∧
    final_state.rubles_spent = 11 ∧
    (∀ (other_moves : List Move),
      let other_final_state := other_moves.foldl apply_move ⟨0, 0⟩
      is_valid_state other_final_state →
      is_winning_state other_final_state →
      other_final_state.rubles_spent ≥ 11) :=
sorry

end NUMINAMATH_CALUDE_min_rubles_to_win_l3143_314385


namespace NUMINAMATH_CALUDE_unique_four_digit_square_with_repeated_digits_l3143_314332

/-- A four-digit number with repeated first two digits and last two digits -/
def FourDigitRepeated (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = 1100 * a + 11 * b

/-- The property of being a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem unique_four_digit_square_with_repeated_digits : 
  ∃! n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ FourDigitRepeated n ∧ IsPerfectSquare n ∧ n = 7744 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_with_repeated_digits_l3143_314332


namespace NUMINAMATH_CALUDE_cubic_identity_l3143_314318

theorem cubic_identity (a b c : ℝ) : 
  a^3*(b^3 - c^3) + b^3*(c^3 - a^3) + c^3*(a^3 - b^3) = 
  (a - b)*(b - c)*(c - a) * ((a^2 + a*b + b^2)*(b^2 + b*c + c^2)*(c^2 + c*a + a^2)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_identity_l3143_314318


namespace NUMINAMATH_CALUDE_expression_value_l3143_314382

theorem expression_value : 
  (121^2 - 19^2) / (91^2 - 13^2) * ((91 - 13)*(91 + 13)) / ((121 - 19)*(121 + 19)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3143_314382


namespace NUMINAMATH_CALUDE_triangle_inequality_l3143_314317

/-- Theorem: For any triangle with side lengths a, b, c and perimeter 2, 
    the inequality a^2 + b^2 + c^2 < 2(1 - abc) holds. -/
theorem triangle_inequality (a b c : ℝ) 
  (triangle_cond : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (perimeter_cond : a + b + c = 2) : 
  a^2 + b^2 + c^2 < 2*(1 - a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3143_314317


namespace NUMINAMATH_CALUDE_prob_five_eight_sided_dice_l3143_314306

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The number of dice rolled -/
def k : ℕ := 5

/-- The probability of at least two dice showing the same number when rolling k fair n-sided dice -/
def prob_at_least_two_same (n k : ℕ) : ℚ :=
  1 - (n.factorial / (n - k).factorial : ℚ) / n^k

theorem prob_five_eight_sided_dice :
  prob_at_least_two_same n k = 3256 / 4096 :=
sorry

end NUMINAMATH_CALUDE_prob_five_eight_sided_dice_l3143_314306


namespace NUMINAMATH_CALUDE_joint_probability_female_literate_l3143_314338

/-- Represents the total number of employees -/
def total_employees : ℕ := 1400

/-- Represents the proportion of female employees -/
def female_ratio : ℚ := 3/5

/-- Represents the proportion of male employees -/
def male_ratio : ℚ := 2/5

/-- Represents the proportion of engineers in the workforce -/
def engineer_ratio : ℚ := 7/20

/-- Represents the proportion of managers in the workforce -/
def manager_ratio : ℚ := 1/4

/-- Represents the proportion of support staff in the workforce -/
def support_ratio : ℚ := 2/5

/-- Represents the overall computer literacy rate -/
def overall_literacy_rate : ℚ := 31/50

/-- Represents the computer literacy rate for male engineers -/
def male_engineer_literacy : ℚ := 4/5

/-- Represents the computer literacy rate for female engineers -/
def female_engineer_literacy : ℚ := 3/4

/-- Represents the computer literacy rate for male managers -/
def male_manager_literacy : ℚ := 11/20

/-- Represents the computer literacy rate for female managers -/
def female_manager_literacy : ℚ := 3/5

/-- Represents the computer literacy rate for male support staff -/
def male_support_literacy : ℚ := 2/5

/-- Represents the computer literacy rate for female support staff -/
def female_support_literacy : ℚ := 1/2

/-- Theorem stating that the joint probability of a randomly selected employee being both female and computer literate is equal to 36.75% -/
theorem joint_probability_female_literate : 
  (female_ratio * engineer_ratio * female_engineer_literacy + 
   female_ratio * manager_ratio * female_manager_literacy + 
   female_ratio * support_ratio * female_support_literacy) = 147/400 := by
  sorry

end NUMINAMATH_CALUDE_joint_probability_female_literate_l3143_314338


namespace NUMINAMATH_CALUDE_samuel_journey_length_l3143_314396

/-- Represents a journey divided into three parts -/
structure Journey where
  first_part : ℚ  -- Fraction of the total journey
  middle_part : ℚ  -- Length in miles
  last_part : ℚ  -- Fraction of the total journey

/-- Calculates the total length of a journey -/
def journey_length (j : Journey) : ℚ :=
  j.middle_part / (1 - j.first_part - j.last_part)

theorem samuel_journey_length :
  let j : Journey := {
    first_part := 1/4,
    middle_part := 30,
    last_part := 1/6
  }
  journey_length j = 360/7 := by
  sorry

end NUMINAMATH_CALUDE_samuel_journey_length_l3143_314396


namespace NUMINAMATH_CALUDE_van_helsing_earnings_l3143_314379

/-- Van Helsing's vampire and werewolf removal earnings problem -/
theorem van_helsing_earnings : ∀ (v w : ℕ),
  w = 4 * v →  -- There were 4 times as many werewolves as vampires
  w = 8 →      -- 8 werewolves were removed
  5 * (v / 2) + 10 * 8 = 85  -- Total earnings calculation
  := by sorry

end NUMINAMATH_CALUDE_van_helsing_earnings_l3143_314379


namespace NUMINAMATH_CALUDE_new_ratio_after_addition_l3143_314360

theorem new_ratio_after_addition (x : ℤ) : 
  (x : ℚ) / (4 * x : ℚ) = 1 / 4 →
  4 * x = 24 →
  (x + 6 : ℚ) / (4 * x : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_new_ratio_after_addition_l3143_314360


namespace NUMINAMATH_CALUDE_book_difference_l3143_314342

def initial_books : ℕ := 28
def jungkook_bought : ℕ := 18
def seokjin_bought : ℕ := 11

theorem book_difference : 
  (initial_books + jungkook_bought) - (initial_books + seokjin_bought) = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_difference_l3143_314342


namespace NUMINAMATH_CALUDE_daniel_correct_answers_l3143_314364

/-- Represents a mathematics competition --/
structure MathCompetition where
  total_problems : ℕ
  points_correct : ℕ
  points_incorrect : ℤ

/-- Represents a contestant's performance --/
structure ContestantPerformance where
  correct_answers : ℕ
  incorrect_answers : ℕ
  total_score : ℤ

/-- The specific competition Daniel participated in --/
def danielCompetition : MathCompetition :=
  { total_problems := 12
  , points_correct := 4
  , points_incorrect := -3 }

/-- Calculates the score based on correct and incorrect answers --/
def calculateScore (comp : MathCompetition) (perf : ContestantPerformance) : ℤ :=
  (comp.points_correct : ℤ) * perf.correct_answers + comp.points_incorrect * perf.incorrect_answers

/-- Theorem stating that Daniel must have answered 9 questions correctly --/
theorem daniel_correct_answers (comp : MathCompetition) (perf : ContestantPerformance) :
    comp = danielCompetition →
    perf.correct_answers + perf.incorrect_answers = comp.total_problems →
    calculateScore comp perf = 21 →
    perf.correct_answers = 9 := by
  sorry

end NUMINAMATH_CALUDE_daniel_correct_answers_l3143_314364
