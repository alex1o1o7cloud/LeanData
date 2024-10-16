import Mathlib

namespace NUMINAMATH_CALUDE_complex_fraction_equality_l664_66417

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ (1 + 2*i) / ((1 - i)^2) = 1 - (1/2)*i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l664_66417


namespace NUMINAMATH_CALUDE_gcd_of_squares_l664_66499

theorem gcd_of_squares : Nat.gcd (130^2 + 251^2 + 372^2) (129^2 + 250^2 + 373^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l664_66499


namespace NUMINAMATH_CALUDE_initial_sales_tax_percentage_l664_66407

/-- Proves that the initial sales tax percentage is 3.5% given the conditions -/
theorem initial_sales_tax_percentage 
  (market_price : ℝ) 
  (new_tax_rate : ℝ) 
  (tax_difference : ℝ) 
  (h1 : market_price = 7800)
  (h2 : new_tax_rate = 10 / 3)
  (h3 : tax_difference = 13) :
  ∃ (x : ℝ), x = 3.5 ∧ market_price * (x / 100 - new_tax_rate / 100) = tax_difference :=
sorry

end NUMINAMATH_CALUDE_initial_sales_tax_percentage_l664_66407


namespace NUMINAMATH_CALUDE_inequality_solution_length_l664_66481

theorem inequality_solution_length (k : ℝ) : 
  (∃ a b : ℝ, a < b ∧ 
    (∀ x : ℝ, a ≤ x ∧ x ≤ b ↔ 1 ≤ x^2 - 3*x + k ∧ x^2 - 3*x + k ≤ 5) ∧
    b - a = 8) →
  k = 9/4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_length_l664_66481


namespace NUMINAMATH_CALUDE_system_solution_unique_l664_66490

theorem system_solution_unique (x y : ℝ) : 
  x + y = 5 ∧ 3 * x + y = 7 ↔ x = 1 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l664_66490


namespace NUMINAMATH_CALUDE_simeon_water_consumption_l664_66457

/-- Simeon's daily water consumption in fluid ounces -/
def daily_water : ℕ := 64

/-- Size of old serving in fluid ounces -/
def old_serving : ℕ := 8

/-- Size of new serving in fluid ounces -/
def new_serving : ℕ := 16

/-- Difference in number of servings -/
def serving_difference : ℕ := 4

theorem simeon_water_consumption :
  ∃ (old_servings new_servings : ℕ),
    old_servings * old_serving = daily_water ∧
    new_servings * new_serving = daily_water ∧
    old_servings = new_servings + serving_difference :=
by sorry

end NUMINAMATH_CALUDE_simeon_water_consumption_l664_66457


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l664_66464

theorem polynomial_division_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, x^5 + x^2 + 3 = (x - 3)^2 * q x + 219 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l664_66464


namespace NUMINAMATH_CALUDE_unsold_books_l664_66488

theorem unsold_books (total_amount : ℝ) (price_per_book : ℝ) (fraction_sold : ℝ) :
  total_amount = 500 ∧
  price_per_book = 5 ∧
  fraction_sold = 2/3 →
  (1 - fraction_sold) * (total_amount / (price_per_book * fraction_sold)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_l664_66488


namespace NUMINAMATH_CALUDE_fifth_pile_magazines_l664_66469

/-- Represents the number of magazines in each pile -/
def magazine_sequence : ℕ → ℕ
| 0 => 3  -- First pile (health)
| 1 => 4  -- Second pile (technology)
| 2 => 6  -- Third pile (fashion)
| 3 => 9  -- Fourth pile (travel)
| n + 4 => magazine_sequence (n + 3) + (n + 4)  -- Subsequent piles

/-- The theorem stating that the fifth pile will contain 13 magazines -/
theorem fifth_pile_magazines : magazine_sequence 4 = 13 := by
  sorry


end NUMINAMATH_CALUDE_fifth_pile_magazines_l664_66469


namespace NUMINAMATH_CALUDE_final_i_is_16_l664_66421

def update_i (i : ℕ) : ℕ :=
  let new_i := 2 * i
  if new_i > 20 then new_i - 20 else new_i

def final_i : ℕ :=
  (List.range 5).foldl (fun acc _ => update_i acc) 2

theorem final_i_is_16 : final_i = 16 := by
  sorry

end NUMINAMATH_CALUDE_final_i_is_16_l664_66421


namespace NUMINAMATH_CALUDE_A_initial_investment_l664_66475

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := sorry

/-- Represents B's investment in rupees -/
def B_investment : ℝ := 21000

/-- Represents the number of months A invested -/
def A_months : ℝ := 12

/-- Represents the number of months B invested -/
def B_months : ℝ := 3

/-- Represents A's share in the profit ratio -/
def A_share : ℝ := 2

/-- Represents B's share in the profit ratio -/
def B_share : ℝ := 3

/-- Theorem stating that A's initial investment is 3500 rupees -/
theorem A_initial_investment : 
  (A_investment * A_months) / (B_investment * B_months) = A_share / B_share → 
  A_investment = 3500 := by sorry

end NUMINAMATH_CALUDE_A_initial_investment_l664_66475


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l664_66432

/-- Two hyperbolas have the same asymptotes if M = 18 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/32 - x^2/M = 1) →
  M = 18 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l664_66432


namespace NUMINAMATH_CALUDE_max_value_z_l664_66401

/-- The maximum value of z = 2x + y given the specified constraints -/
theorem max_value_z (x y : ℝ) (h1 : y ≤ 2 * x) (h2 : x - 2 * y - 4 ≤ 0) (h3 : y ≤ 4 - x) :
  (∀ x' y' : ℝ, y' ≤ 2 * x' → x' - 2 * y' - 4 ≤ 0 → y' ≤ 4 - x' → 2 * x' + y' ≤ 2 * x + y) ∧
  2 * x + y = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l664_66401


namespace NUMINAMATH_CALUDE_max_value_of_symmetric_f_l664_66443

/-- A function f(x) that is symmetric about the line x = -2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- The symmetry condition for f(x) about x = -2 -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x, f a b x = f a b (-4 - x)

/-- The maximum value of f(x) is 16 when it's symmetric about x = -2 -/
theorem max_value_of_symmetric_f (a b : ℝ) (h : is_symmetric a b) :
  ∃ x₀, ∀ x, f a b x ≤ f a b x₀ ∧ f a b x₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_symmetric_f_l664_66443


namespace NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l664_66456

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- The fixed point K -/
def K : ℝ × ℝ := (2, 0)

/-- Dot product of two 2D vectors -/
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

/-- Vector from K to a point -/
def vector_from_K (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - K.1, p.2 - K.2)

theorem min_dot_product_on_ellipse :
  ∀ (M N : ℝ × ℝ),
  is_on_ellipse M.1 M.2 →
  is_on_ellipse N.1 N.2 →
  dot_product (vector_from_K M) (vector_from_K N) = 0 →
  ∃ (min_value : ℝ),
    min_value = 23/3 ∧
    ∀ (P Q : ℝ × ℝ),
    is_on_ellipse P.1 P.2 →
    is_on_ellipse Q.1 Q.2 →
    dot_product (vector_from_K P) (vector_from_K Q) = 0 →
    dot_product (vector_from_K P) (vector_from_K Q - vector_from_K P) ≥ min_value :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l664_66456


namespace NUMINAMATH_CALUDE_complex_expression_equality_l664_66472

theorem complex_expression_equality (c d : ℂ) (h1 : c = 3 - 2*I) (h2 : d = 2 + 3*I) :
  3*c + 4*d + 2 = 19 + 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l664_66472


namespace NUMINAMATH_CALUDE_vector_expression_equality_l664_66474

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_expression_equality (a b : V) :
  (1 / 3 : ℝ) • (a - 2 • b) + b = (1 / 3 : ℝ) • a + (1 / 3 : ℝ) • b :=
by sorry

end NUMINAMATH_CALUDE_vector_expression_equality_l664_66474


namespace NUMINAMATH_CALUDE_sqrt_three_between_l664_66430

theorem sqrt_three_between (n : ℕ+) : 
  (1 + 3 / (n + 1 : ℝ) < Real.sqrt 3 ∧ Real.sqrt 3 < 1 + 3 / (n : ℝ)) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_between_l664_66430


namespace NUMINAMATH_CALUDE_equation_solution_l664_66471

theorem equation_solution (y : ℚ) : (1 / 3 : ℚ) + 1 / y = 7 / 9 → y = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l664_66471


namespace NUMINAMATH_CALUDE_sqrt_calculation_l664_66447

theorem sqrt_calculation : 
  Real.sqrt 48 / Real.sqrt 3 + Real.sqrt (1/2) * Real.sqrt 12 - Real.sqrt 24 = 4 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l664_66447


namespace NUMINAMATH_CALUDE_spatial_relationships_l664_66467

/-- Two lines are non-coincident -/
def non_coincident_lines (l m : Line) : Prop := l ≠ m

/-- Two planes are non-coincident -/
def non_coincident_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (α : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (α : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perp (α β : Plane) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perp (l m : Line) : Prop := sorry

theorem spatial_relationships (l m : Line) (α β : Plane) 
  (h1 : non_coincident_lines l m) (h2 : non_coincident_planes α β) :
  (lines_perp l m ∧ line_perp_plane l α ∧ line_perp_plane m β → planes_perp α β) ∧
  (line_perp_plane l β ∧ planes_perp α β → line_parallel_plane l α ∨ line_in_plane l α) :=
sorry

end NUMINAMATH_CALUDE_spatial_relationships_l664_66467


namespace NUMINAMATH_CALUDE_function_property_l664_66411

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x + f (1 - x) = 10)
  (h2 : ∃ a : ℝ, ∀ x : ℝ, f (1 + x) = a + f x)
  (h3 : ∀ x : ℝ, f x + f (-x) = 7) :
  ∃ a : ℝ, (∀ x : ℝ, f (1 + x) = a + f x) ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l664_66411


namespace NUMINAMATH_CALUDE_min_games_for_2015_scores_l664_66452

/-- Represents the scoring system for a football league -/
structure ScoringSystem where
  a : ℝ  -- Points for a win
  b : ℝ  -- Points for a draw
  h : a > b ∧ b > 0

/-- Calculates the number of possible scores after n games -/
def possibleScores (s : ScoringSystem) (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of games for 2015 possible scores -/
theorem min_games_for_2015_scores (s : ScoringSystem) :
  (∀ m : ℕ, m < 62 → possibleScores s m < 2015) ∧
  possibleScores s 62 = 2015 :=
sorry

end NUMINAMATH_CALUDE_min_games_for_2015_scores_l664_66452


namespace NUMINAMATH_CALUDE_uninterrupted_viewing_time_movie_problem_solution_l664_66486

/-- Calculates the uninterrupted viewing time at the end of a movie given the total viewing time,
    initial viewing periods, and rewind times. -/
theorem uninterrupted_viewing_time 
  (total_time : ℕ) 
  (first_viewing : ℕ) 
  (first_rewind : ℕ) 
  (second_viewing : ℕ) 
  (second_rewind : ℕ) : 
  total_time - (first_viewing + second_viewing + first_rewind + second_rewind) = 
  total_time - ((first_viewing + second_viewing) + (first_rewind + second_rewind)) :=
by sorry

/-- Proves that the uninterrupted viewing time at the end of the movie is 20 minutes
    given the specific conditions from the problem. -/
theorem movie_problem_solution 
  (total_time : ℕ) 
  (first_viewing : ℕ) 
  (first_rewind : ℕ) 
  (second_viewing : ℕ) 
  (second_rewind : ℕ) 
  (h1 : total_time = 120) 
  (h2 : first_viewing = 35) 
  (h3 : first_rewind = 5) 
  (h4 : second_viewing = 45) 
  (h5 : second_rewind = 15) : 
  total_time - (first_viewing + second_viewing + first_rewind + second_rewind) = 20 :=
by sorry

end NUMINAMATH_CALUDE_uninterrupted_viewing_time_movie_problem_solution_l664_66486


namespace NUMINAMATH_CALUDE_marble_223_is_white_l664_66413

def marble_color (n : ℕ) : String :=
  let cycle := n % 15
  if cycle < 6 then "gray"
  else if cycle < 11 then "white"
  else "black"

theorem marble_223_is_white :
  marble_color 223 = "white" := by
  sorry

end NUMINAMATH_CALUDE_marble_223_is_white_l664_66413


namespace NUMINAMATH_CALUDE_multiply_powers_of_x_l664_66465

theorem multiply_powers_of_x (x : ℝ) : 2 * (x^3) * (x^3) = 2 * (x^6) := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_x_l664_66465


namespace NUMINAMATH_CALUDE_system_solution_l664_66404

theorem system_solution (x y : ℝ) : 
  (1 / x + 1 / y = 2.25 ∧ x^2 / y + y^2 / x = 32.0625) ↔ 
  ((x = 4 ∧ y = 1/2) ∨ 
   (x = 1/12 * (-19 + Real.sqrt (1691/3)) ∧ 
    y = 1/12 * (-19 - Real.sqrt (1691/3)))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l664_66404


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l664_66414

theorem imaginary_part_of_complex_division :
  let i : ℂ := Complex.I
  (3 + 2*i) / i = Complex.mk 2 (-3) :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l664_66414


namespace NUMINAMATH_CALUDE_greatest_number_in_set_l664_66424

/-- Given a set of 45 consecutive multiples of 5 starting from 55, 
    the greatest number in the set is 275. -/
theorem greatest_number_in_set (s : Set ℕ) 
  (h1 : ∀ n ∈ s, ∃ k, n = 5 * k) 
  (h2 : ∀ n ∈ s, 55 ≤ n ∧ n ≤ 275)
  (h3 : ∀ n, 55 ≤ n ∧ n ≤ 275 ∧ 5 ∣ n → n ∈ s)
  (h4 : 55 ∈ s)
  (h5 : Fintype s)
  (h6 : Fintype.card s = 45) : 
  275 ∈ s ∧ ∀ n ∈ s, n ≤ 275 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_in_set_l664_66424


namespace NUMINAMATH_CALUDE_candy_bar_multiple_l664_66415

def fred_candy_bars : ℕ := 12
def uncle_bob_extra_candy_bars : ℕ := 6
def jacqueline_percentage : ℚ := 40 / 100
def jacqueline_percentage_amount : ℕ := 120

theorem candy_bar_multiple :
  let uncle_bob_candy_bars := fred_candy_bars + uncle_bob_extra_candy_bars
  let total_fred_uncle_bob := fred_candy_bars + uncle_bob_candy_bars
  let jacqueline_candy_bars := jacqueline_percentage_amount / jacqueline_percentage
  jacqueline_candy_bars / total_fred_uncle_bob = 10 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_multiple_l664_66415


namespace NUMINAMATH_CALUDE_factorial_fraction_l664_66476

theorem factorial_fraction (N : ℕ) : 
  (Nat.factorial (N - 1) * (N^2 + N)) / Nat.factorial (N + 2) = 1 / (N + 2) :=
sorry

end NUMINAMATH_CALUDE_factorial_fraction_l664_66476


namespace NUMINAMATH_CALUDE_crayon_count_prove_crayon_count_l664_66454

theorem crayon_count : ℕ → Prop :=
  fun red_count =>
    let blue_count := red_count + 5
    let yellow_count := 2 * blue_count - 6
    yellow_count = 32 → red_count = 14

/-- Proof of the crayon count theorem -/
theorem prove_crayon_count : ∃ (red_count : ℕ), crayon_count red_count :=
  sorry

end NUMINAMATH_CALUDE_crayon_count_prove_crayon_count_l664_66454


namespace NUMINAMATH_CALUDE_y_axis_inclination_l664_66426

-- Define the concept of an axis
def Axis : Type := ℝ → ℝ

-- Define the x-axis and y-axis
def x_axis : Axis := λ x => 0
def y_axis : Axis := λ y => y

-- Define the concept of perpendicular axes
def perpendicular (a b : Axis) : Prop := sorry

-- Define the concept of inclination angle
def inclination_angle (a : Axis) : ℝ := sorry

-- Theorem statement
theorem y_axis_inclination :
  perpendicular x_axis y_axis →
  inclination_angle y_axis = 90 :=
sorry

end NUMINAMATH_CALUDE_y_axis_inclination_l664_66426


namespace NUMINAMATH_CALUDE_michael_crayons_worth_l664_66495

/-- Calculates the total worth of crayons after a purchase --/
def total_worth_after_purchase (initial_packs : ℕ) (additional_packs : ℕ) (cost_per_pack : ℚ) : ℚ :=
  (initial_packs + additional_packs) * cost_per_pack

/-- Proves that the total worth of crayons after Michael's purchase is $15 --/
theorem michael_crayons_worth :
  let initial_packs := 4
  let additional_packs := 2
  let cost_per_pack := 5/2
  total_worth_after_purchase initial_packs additional_packs cost_per_pack = 15 := by
  sorry

end NUMINAMATH_CALUDE_michael_crayons_worth_l664_66495


namespace NUMINAMATH_CALUDE_factor_expression_l664_66444

theorem factor_expression (x y : ℝ) : 
  5 * x * (x + 1) + 7 * (x + 1) - 2 * y * (x + 1) = (x + 1) * (5 * x + 7 - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l664_66444


namespace NUMINAMATH_CALUDE_probability_two_defective_in_four_tests_l664_66459

def total_components : ℕ := 6
def defective_components : ℕ := 2
def good_components : ℕ := 4
def tests : ℕ := 4

theorem probability_two_defective_in_four_tests :
  (
    -- Probability of finding one defective in first three tests and second on fourth test
    (defective_components / total_components *
     good_components / (total_components - 1) *
     (good_components - 1) / (total_components - 2) *
     (defective_components - 1) / (total_components - 3)) +
    (good_components / total_components *
     defective_components / (total_components - 1) *
     (good_components - 1) / (total_components - 2) *
     (defective_components - 1) / (total_components - 3)) +
    (good_components / total_components *
     (good_components - 1) / (total_components - 1) *
     defective_components / (total_components - 2) *
     (defective_components - 1) / (total_components - 3)) +
    -- Probability of finding all good components in four tests
    (good_components / total_components *
     (good_components - 1) / (total_components - 1) *
     (good_components - 2) / (total_components - 2) *
     (good_components - 3) / (total_components - 3))
  ) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_defective_in_four_tests_l664_66459


namespace NUMINAMATH_CALUDE_sum_of_squares_greater_than_ten_l664_66483

theorem sum_of_squares_greater_than_ten (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁₂ : |x₁ - x₂| > 1) (h₁₃ : |x₁ - x₃| > 1) (h₁₄ : |x₁ - x₄| > 1) (h₁₅ : |x₁ - x₅| > 1)
  (h₂₃ : |x₂ - x₃| > 1) (h₂₄ : |x₂ - x₄| > 1) (h₂₅ : |x₂ - x₅| > 1)
  (h₃₄ : |x₃ - x₄| > 1) (h₃₅ : |x₃ - x₅| > 1)
  (h₄₅ : |x₄ - x₅| > 1) :
  x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 > 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_greater_than_ten_l664_66483


namespace NUMINAMATH_CALUDE_root_implies_k_value_l664_66455

theorem root_implies_k_value (k : ℝ) : ((-3 : ℝ)^2 + (-3) - k = 0) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l664_66455


namespace NUMINAMATH_CALUDE_find_number_l664_66478

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 7) = 99 :=
  sorry

end NUMINAMATH_CALUDE_find_number_l664_66478


namespace NUMINAMATH_CALUDE_weekend_grass_cutting_time_l664_66496

/-- Calculates the total time Jason spends cutting grass over the weekend -/
def total_weekend_time (small_time medium_time large_time break_time weather_delay : ℕ)
  (saturday_small saturday_medium saturday_large : ℕ)
  (sunday_medium sunday_large : ℕ) : ℕ :=
  let saturday_time := 
    saturday_small * small_time + 
    saturday_medium * medium_time + 
    saturday_large * large_time + 
    (saturday_small + saturday_medium + saturday_large - 1) * break_time
  let sunday_time := 
    sunday_medium * (medium_time + weather_delay) + 
    sunday_large * (large_time + weather_delay) + 
    (sunday_medium + sunday_large - 1) * break_time
  saturday_time + sunday_time

theorem weekend_grass_cutting_time :
  total_weekend_time 25 30 40 5 10 2 4 2 6 2 = 11 * 60 := by
  sorry

end NUMINAMATH_CALUDE_weekend_grass_cutting_time_l664_66496


namespace NUMINAMATH_CALUDE_equal_natural_numbers_l664_66436

theorem equal_natural_numbers (a b : ℕ) (h : a^3 + a + 4*b^2 = 4*a*b + b + b*a^2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_equal_natural_numbers_l664_66436


namespace NUMINAMATH_CALUDE_second_grade_sample_l664_66418

/-- Represents the number of students to be sampled from a stratum in stratified sampling -/
def stratifiedSample (totalSample : ℕ) (stratumWeight : ℕ) (totalWeight : ℕ) : ℕ :=
  (stratumWeight * totalSample) / totalWeight

/-- Theorem: In a school with grades in 3:3:4 ratio, stratified sampling of 50 students
    results in 15 students from the second grade -/
theorem second_grade_sample :
  let totalSample : ℕ := 50
  let firstGradeWeight : ℕ := 3
  let secondGradeWeight : ℕ := 3
  let thirdGradeWeight : ℕ := 4
  let totalWeight : ℕ := firstGradeWeight + secondGradeWeight + thirdGradeWeight
  stratifiedSample totalSample secondGradeWeight totalWeight = 15 := by
  sorry

#eval stratifiedSample 50 3 10  -- Expected output: 15

end NUMINAMATH_CALUDE_second_grade_sample_l664_66418


namespace NUMINAMATH_CALUDE_final_week_study_hours_l664_66412

def study_hours : List ℕ := [8, 10, 9, 11, 10, 7]
def total_weeks : ℕ := 7
def required_average : ℕ := 9

theorem final_week_study_hours :
  ∃ (x : ℕ), 
    (List.sum study_hours + x) / total_weeks = required_average ∧
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_week_study_hours_l664_66412


namespace NUMINAMATH_CALUDE_dihedral_angle_range_l664_66400

/-- The dihedral angle between two adjacent faces in a regular n-sided prism -/
def dihedral_angle (n : ℕ) (θ : ℝ) : Prop :=
  n ≥ 3 ∧ ((n - 2 : ℝ) / n * Real.pi < θ ∧ θ < Real.pi)

/-- Theorem: The dihedral angle in a regular n-sided prism is within the specified range -/
theorem dihedral_angle_range (n : ℕ) :
  ∃ θ : ℝ, dihedral_angle n θ :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_range_l664_66400


namespace NUMINAMATH_CALUDE_angle_C_value_triangle_area_l664_66498

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.a^2 - t.a * t.b - 2 * t.b^2 = 0

-- Theorem 1
theorem angle_C_value (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.B = π / 6) : 
  t.C = π / 3 := by sorry

-- Theorem 2
theorem triangle_area (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.C = 2 * π / 3) 
  (h3 : t.c = 14) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 14 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_C_value_triangle_area_l664_66498


namespace NUMINAMATH_CALUDE_locus_is_circle_l664_66473

/-- An equilateral triangle in a 2D plane -/
structure EquilateralTriangle where
  s : ℝ  -- side length
  A : ℝ × ℝ  -- coordinates of vertex A
  B : ℝ × ℝ  -- coordinates of vertex B
  C : ℝ × ℝ  -- coordinates of vertex C
  is_equilateral : 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = s^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = s^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = s^2

/-- The locus of points with constant sum of squared distances to triangle vertices -/
def ConstantSumLocus (tri : EquilateralTriangle) (a : ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | 
    (P.1 - tri.A.1)^2 + (P.2 - tri.A.2)^2 + 
    (P.1 - tri.B.1)^2 + (P.2 - tri.B.2)^2 + 
    (P.1 - tri.C.1)^2 + (P.2 - tri.C.2)^2 = a}

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

theorem locus_is_circle (tri : EquilateralTriangle) (a : ℝ) (h : a > tri.s^2) :
  ∃ (c : Circle), ConstantSumLocus tri a = {P : ℝ × ℝ | (P.1 - c.center.1)^2 + (P.2 - c.center.2)^2 = c.radius^2} :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l664_66473


namespace NUMINAMATH_CALUDE_polynomial_arrangement_l664_66484

-- Define the polynomial as a function
def polynomial (x y : ℝ) : ℝ := 2 * x^3 * y - 4 * y^2 + 5 * x^2

-- Define the arranged polynomial as a function
def arranged_polynomial (x y : ℝ) : ℝ := 5 * x^2 + 2 * x^3 * y - 4 * y^2

-- Theorem stating that the arranged polynomial is equal to the original polynomial
theorem polynomial_arrangement (x y : ℝ) : 
  polynomial x y = arranged_polynomial x y := by
  sorry

end NUMINAMATH_CALUDE_polynomial_arrangement_l664_66484


namespace NUMINAMATH_CALUDE_linda_cookies_theorem_l664_66491

/-- The number of batches Linda needs to bake to have enough cookies for her classmates -/
def batches_needed (num_classmates : ℕ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) 
  (choc_chip_batches : ℕ) (oatmeal_raisin_batches : ℕ) : ℕ :=
  let total_cookies_needed := num_classmates * cookies_per_student
  let cookies_made := (choc_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let cookies_left_to_make := total_cookies_needed - cookies_made
  (cookies_left_to_make + cookies_per_batch - 1) / cookies_per_batch

theorem linda_cookies_theorem : 
  batches_needed 24 10 48 2 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_linda_cookies_theorem_l664_66491


namespace NUMINAMATH_CALUDE_cube_sphere_volume_l664_66463

theorem cube_sphere_volume (cube_volume : Real) (h : cube_volume = 8) :
  ∃ (sphere_volume : Real), sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_l664_66463


namespace NUMINAMATH_CALUDE_system_solution_l664_66449

theorem system_solution : 
  ∀ x : ℝ, (3 * x^2 = Real.sqrt (36 * x^2) ∧ 3 * x^2 + 21 = 24 * x) ↔ (x = 7 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l664_66449


namespace NUMINAMATH_CALUDE_circle_set_equivalence_l664_66410

-- Define the circle C
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point D
def D : ℝ × ℝ := sorry

-- Define the circle C
def C : Circle := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the set of points A
def A : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem circle_set_equivalence :
  (∀ (p : ℝ × ℝ), p ∈ A ↔ 
    distance p D < C.radius ∧ 
    (∀ (q : ℝ × ℝ), distance q C.center = C.radius → distance p D ≤ distance p q)) ↔
  (∀ (p : ℝ × ℝ), p ∈ A ↔ 
    distance p D < C.radius) :=
sorry

end NUMINAMATH_CALUDE_circle_set_equivalence_l664_66410


namespace NUMINAMATH_CALUDE_arithmetic_sequence_iff_t_eq_zero_l664_66462

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) (t : ℝ) : ℝ := n^2 + 5*n + t

/-- The nth term of the sequence -/
def a (n : ℕ) (t : ℝ) : ℝ :=
  if n = 1 then S 1 t
  else S n t - S (n-1) t

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic_sequence (f : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, f (n+1) - f n = d

theorem arithmetic_sequence_iff_t_eq_zero (t : ℝ) :
  is_arithmetic_sequence (λ n => a n t) ↔ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_iff_t_eq_zero_l664_66462


namespace NUMINAMATH_CALUDE_sum_primes_square_bound_l664_66434

/-- S_n is the sum of the first n prime numbers -/
def S (n : ℕ) : ℕ := sorry

/-- The n-th prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

theorem sum_primes_square_bound :
  ∀ n : ℕ, n > 0 → ∃ m : ℕ, S n ≤ m^2 ∧ m^2 ≤ S (n + 1) :=
sorry

end NUMINAMATH_CALUDE_sum_primes_square_bound_l664_66434


namespace NUMINAMATH_CALUDE_book_arrangement_count_l664_66448

/-- The number of ways to arrange two types of indistinguishable objects in a row -/
def arrangement_count (n m : ℕ) : ℕ :=
  Nat.choose (n + m) n

/-- Theorem: Arranging 4 copies of one book and 5 copies of another book yields 126 possibilities -/
theorem book_arrangement_count :
  arrangement_count 4 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l664_66448


namespace NUMINAMATH_CALUDE_unique_solution_system_l664_66479

theorem unique_solution_system (x y : ℝ) : 
  (x^3 + y^3 + 3*x*y = 1 ∧ x^2 - y^2 = 1) →
  ((x ≥ 0 ∧ y ≥ 0) ∨ (x + y > 0)) →
  x = 1 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l664_66479


namespace NUMINAMATH_CALUDE_jana_travel_distance_l664_66403

/-- Calculates the total distance traveled by Jana given her walking and cycling rates and times. -/
theorem jana_travel_distance (walking_rate : ℝ) (walking_time : ℝ) (cycling_rate : ℝ) (cycling_time : ℝ) :
  walking_rate = 1 / 30 →
  walking_time = 45 →
  cycling_rate = 2 / 15 →
  cycling_time = 30 →
  walking_rate * walking_time + cycling_rate * cycling_time = 5.5 :=
by
  sorry

end NUMINAMATH_CALUDE_jana_travel_distance_l664_66403


namespace NUMINAMATH_CALUDE_journey_time_ratio_l664_66440

/-- Proves the ratio of new time to original time for a given journey -/
theorem journey_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 288 ∧ original_time = 6 ∧ new_speed = 32 →
  (distance / new_speed) / original_time = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_ratio_l664_66440


namespace NUMINAMATH_CALUDE_largest_modulus_of_cubic_root_l664_66450

theorem largest_modulus_of_cubic_root (a b c d z : ℂ) :
  (Complex.abs a = Complex.abs b) →
  (Complex.abs b = Complex.abs c) →
  (Complex.abs c = Complex.abs d) →
  (Complex.abs a > 0) →
  (a * z^3 + b * z^2 + c * z + d = 0) →
  ∃ t : ℝ, t^3 - t^2 - t - 1 = 0 ∧ Complex.abs z ≤ t :=
by sorry

end NUMINAMATH_CALUDE_largest_modulus_of_cubic_root_l664_66450


namespace NUMINAMATH_CALUDE_percentage_relations_l664_66433

theorem percentage_relations (x y z w : ℝ) 
  (h1 : x = 1.3 * y) 
  (h2 : y = 0.5 * z) 
  (h3 : w = 2 * x) : 
  x = 0.65 * z ∧ y = 0.5 * z ∧ w = 1.3 * z := by
  sorry

end NUMINAMATH_CALUDE_percentage_relations_l664_66433


namespace NUMINAMATH_CALUDE_sin_cos_equation_solutions_l664_66468

theorem sin_cos_equation_solutions (π : Real) (sin cos : Real → Real) :
  (∃ (x₁ x₂ : Real), x₁ ≠ x₂ ∧ 
   0 ≤ x₁ ∧ x₁ ≤ π ∧ 
   0 ≤ x₂ ∧ x₂ ≤ π ∧
   sin (π / 2 * cos x₁) = cos (π / 2 * sin x₁) ∧
   sin (π / 2 * cos x₂) = cos (π / 2 * sin x₂)) ∧
  (∀ (x y z : Real), 
   0 ≤ x ∧ x ≤ π ∧ 
   0 ≤ y ∧ y ≤ π ∧ 
   0 ≤ z ∧ z ≤ π ∧
   x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
   sin (π / 2 * cos x) = cos (π / 2 * sin x) ∧
   sin (π / 2 * cos y) = cos (π / 2 * sin y) ∧
   sin (π / 2 * cos z) = cos (π / 2 * sin z) →
   False) :=
by sorry


end NUMINAMATH_CALUDE_sin_cos_equation_solutions_l664_66468


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_five_l664_66497

theorem complex_magnitude_equals_five (m : ℝ) (hm : m > 0) :
  Complex.abs (Complex.mk (-1) (2 * m)) = 5 ↔ m = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_five_l664_66497


namespace NUMINAMATH_CALUDE_division_remainder_proof_l664_66451

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 162 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 9 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l664_66451


namespace NUMINAMATH_CALUDE_exactly_one_truck_congestion_at_least_two_trucks_congestion_l664_66419

-- Define the probabilities for highways I and II
def prob_congestion_I : ℚ := 1/10
def prob_no_congestion_I : ℚ := 9/10
def prob_congestion_II : ℚ := 3/5
def prob_no_congestion_II : ℚ := 2/5

-- Define the events
def event_A : ℚ := prob_congestion_I
def event_B : ℚ := prob_congestion_I
def event_C : ℚ := prob_congestion_II

-- Theorem for the first question
theorem exactly_one_truck_congestion :
  prob_congestion_I * prob_no_congestion_I + prob_no_congestion_I * prob_congestion_I = 9/50 := by sorry

-- Theorem for the second question
theorem at_least_two_trucks_congestion :
  event_A * event_B * (1 - event_C) + 
  event_A * (1 - event_B) * event_C + 
  (1 - event_A) * event_B * event_C + 
  event_A * event_B * event_C = 59/500 := by sorry

end NUMINAMATH_CALUDE_exactly_one_truck_congestion_at_least_two_trucks_congestion_l664_66419


namespace NUMINAMATH_CALUDE_coin_division_problem_l664_66453

theorem coin_division_problem (n : ℕ) : 
  (n > 0 ∧ 
   n % 8 = 7 ∧ 
   n % 7 = 5 ∧ 
   ∀ m : ℕ, (m > 0 ∧ m % 8 = 7 ∧ m % 7 = 5) → n ≤ m) →
  (n = 47 ∧ n % 9 = 2) :=
by sorry

end NUMINAMATH_CALUDE_coin_division_problem_l664_66453


namespace NUMINAMATH_CALUDE_emails_remaining_proof_l664_66420

/-- Given an initial number of emails, calculates the number of emails remaining in the inbox
    after moving half to trash and 40% of the remainder to a work folder. -/
def remaining_emails (initial : ℕ) : ℕ :=
  let after_trash := initial / 2
  let to_work_folder := (40 * after_trash) / 100
  after_trash - to_work_folder

/-- Proves that given 400 initial emails, 120 emails remain in the inbox after the operations. -/
theorem emails_remaining_proof :
  remaining_emails 400 = 120 := by
  sorry

#eval remaining_emails 400  -- Should output 120

end NUMINAMATH_CALUDE_emails_remaining_proof_l664_66420


namespace NUMINAMATH_CALUDE_lost_episodes_proof_l664_66461

/-- Represents the number of episodes lost per season after a computer failure --/
def episodes_lost_per_season (series1_seasons series2_seasons episodes_per_season remaining_episodes : ℕ) : ℕ :=
  let total_episodes := (series1_seasons + series2_seasons) * episodes_per_season
  let lost_episodes := total_episodes - remaining_episodes
  lost_episodes / (series1_seasons + series2_seasons)

/-- Theorem stating that given the problem conditions, 2 episodes were lost per season --/
theorem lost_episodes_proof :
  episodes_lost_per_season 12 14 16 364 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lost_episodes_proof_l664_66461


namespace NUMINAMATH_CALUDE_bouncy_balls_shipment_l664_66425

theorem bouncy_balls_shipment (displayed_percentage : ℚ) (warehouse_count : ℕ) : 
  displayed_percentage = 1/4 →
  warehouse_count = 90 →
  ∃ total : ℕ, total = 120 ∧ (1 - displayed_percentage) * total = warehouse_count :=
by sorry

end NUMINAMATH_CALUDE_bouncy_balls_shipment_l664_66425


namespace NUMINAMATH_CALUDE_wall_length_proof_l664_66428

/-- Proves that the length of a wall is 29 meters given specific brick and wall dimensions and the number of bricks required. -/
theorem wall_length_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_width : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_width = 2 →
  wall_height = 0.75 →
  num_bricks = 29000 →
  ∃ (wall_length : ℝ), wall_length = 29 :=
by
  sorry

#check wall_length_proof

end NUMINAMATH_CALUDE_wall_length_proof_l664_66428


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l664_66489

theorem same_solution_implies_c_value (y : ℝ) (c : ℝ) : 
  (3 * y - 9 = 0) ∧ (c * y + 15 = 3) → c = -4 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l664_66489


namespace NUMINAMATH_CALUDE_count_positive_area_triangles_l664_66438

/-- The total number of points in the grid -/
def total_points : ℕ := 7

/-- The number of sets of three collinear points -/
def collinear_sets : ℕ := 5

/-- The number of triangles with positive area -/
def positive_area_triangles : ℕ := 30

/-- Theorem stating the number of triangles with positive area -/
theorem count_positive_area_triangles :
  (Nat.choose total_points 3) - collinear_sets = positive_area_triangles :=
by sorry

end NUMINAMATH_CALUDE_count_positive_area_triangles_l664_66438


namespace NUMINAMATH_CALUDE_age_problem_l664_66406

theorem age_problem (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 40) : x = 10 ∧ y = 30 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l664_66406


namespace NUMINAMATH_CALUDE_min_paper_length_l664_66485

/-- Represents a binary message of length 2016 -/
def Message := Fin 2016 → Bool

/-- Represents a paper of length n with 10 pre-colored consecutive squares -/
structure Paper (n : ℕ) where
  squares : Fin n → Bool
  precolored_start : Fin (n - 9)
  precolored : Fin 10 → Bool

/-- A strategy for encoding and decoding messages -/
structure Strategy (n : ℕ) where
  encode : Message → Paper n → Paper n
  decode : Paper n → Message

/-- The strategy works with perfect accuracy -/
def perfect_accuracy (s : Strategy n) : Prop :=
  ∀ (m : Message) (p : Paper n), s.decode (s.encode m p) = m

/-- The main theorem: The minimum value of n for which a perfect strategy exists is 2026 -/
theorem min_paper_length :
  (∃ (s : Strategy 2026), perfect_accuracy s) ∧
  (∀ (n : ℕ), n < 2026 → ¬∃ (s : Strategy n), perfect_accuracy s) :=
sorry

end NUMINAMATH_CALUDE_min_paper_length_l664_66485


namespace NUMINAMATH_CALUDE_remainder_seven_n_l664_66480

theorem remainder_seven_n (n : ℤ) (h : n ≡ 3 [ZMOD 4]) : 7*n ≡ 1 [ZMOD 4] := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_n_l664_66480


namespace NUMINAMATH_CALUDE_sin_30_plus_cos_60_l664_66408

theorem sin_30_plus_cos_60 : Real.sin (30 * π / 180) + Real.cos (60 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_plus_cos_60_l664_66408


namespace NUMINAMATH_CALUDE_triangle_operation_result_l664_66427

-- Define the triangle operation
def triangle (a b : ℝ) : ℝ := a^2 - 2*b

-- Theorem statement
theorem triangle_operation_result : triangle (-2) (triangle 3 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_operation_result_l664_66427


namespace NUMINAMATH_CALUDE_green_balls_count_l664_66446

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) 
  (h_total : total = 60)
  (h_white : white = 22)
  (h_yellow : yellow = 8)
  (h_red : red = 5)
  (h_purple : purple = 7)
  (h_prob : prob_not_red_purple = 4/5) :
  ∃ green : ℕ, 
    green = total - (white + yellow + red + purple) ∧ 
    (white + green + yellow : ℚ) / total = prob_not_red_purple :=
by sorry

end NUMINAMATH_CALUDE_green_balls_count_l664_66446


namespace NUMINAMATH_CALUDE_three_dice_probability_l664_66431

/-- A fair 6-sided die -/
def Die : Type := Fin 6

/-- A roll of three dice -/
def ThreeDiceRoll : Type := Die × Die × Die

/-- The total number of possible outcomes when rolling three 6-sided dice -/
def totalOutcomes : ℕ := 216

/-- The number of permutations of three distinct numbers -/
def permutations : ℕ := 6

/-- The probability of rolling a 2, 3, and 4 in any order with three fair 6-sided dice -/
def winProbability : ℚ := 1 / 36

/-- Theorem: The probability of rolling a 2, 3, and 4 in any order with three fair 6-sided dice is 1/36 -/
theorem three_dice_probability :
  (permutations : ℚ) / totalOutcomes = winProbability :=
sorry

end NUMINAMATH_CALUDE_three_dice_probability_l664_66431


namespace NUMINAMATH_CALUDE_emilys_average_speed_l664_66466

-- Define the parameters of Emily's trip
def distance1 : ℝ := 450  -- miles
def time1 : ℝ := 7.5      -- hours (7 hours 30 minutes)
def break_time : ℝ := 1   -- hour
def distance2 : ℝ := 540  -- miles
def time2 : ℝ := 8        -- hours

-- Define the total distance and time
def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := time1 + break_time + time2

-- Theorem to prove
theorem emilys_average_speed :
  total_distance / total_time = 60 := by sorry

end NUMINAMATH_CALUDE_emilys_average_speed_l664_66466


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l664_66422

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x * y) / (x^5 + x * y + y^5) + (y * z) / (y^5 + y * z + z^5) + (z * x) / (z^5 + z * x + x^5) ≤ 1 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x * y) / (x^5 + x * y + y^5) + (y * z) / (y^5 + y * z + z^5) + (z * x) / (z^5 + z * x + x^5) = 1 ↔
  x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l664_66422


namespace NUMINAMATH_CALUDE_smallest_n_for_negative_sum_l664_66477

-- Define the arithmetic sequence and its sum
def a (n : ℕ) : ℤ := 7 - 2 * (n - 1)
def S (n : ℕ) : ℤ := n * (2 * 7 + (n - 1) * (-2)) / 2

-- State the theorem
theorem smallest_n_for_negative_sum :
  (∀ k < 9, S k ≥ 0) ∧ (S 9 < 0) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_negative_sum_l664_66477


namespace NUMINAMATH_CALUDE_binomial_29_5_l664_66493

theorem binomial_29_5 (h1 : Nat.choose 27 3 = 2925)
                      (h2 : Nat.choose 27 4 = 17550)
                      (h3 : Nat.choose 27 5 = 80730) :
  Nat.choose 29 5 = 118755 := by
  sorry

end NUMINAMATH_CALUDE_binomial_29_5_l664_66493


namespace NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l664_66482

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l664_66482


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_always_positive_l664_66487

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem quadratic_always_positive : 
  (¬ ∃ x : ℝ, x^2 + x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_always_positive_l664_66487


namespace NUMINAMATH_CALUDE_absolute_value_expression_l664_66460

theorem absolute_value_expression (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c > 0) : 
  |a| - |a + b| + |c - a| + |b - c| = 2 * c - a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l664_66460


namespace NUMINAMATH_CALUDE_kernels_in_first_bag_l664_66439

/-- The number of kernels in the first bag -/
def first_bag : ℕ := 74

/-- The number of popped kernels in the first bag -/
def popped_first : ℕ := 60

/-- The number of kernels in the second bag -/
def second_bag : ℕ := 50

/-- The number of popped kernels in the second bag -/
def popped_second : ℕ := 42

/-- The number of kernels in the third bag -/
def third_bag : ℕ := 100

/-- The number of popped kernels in the third bag -/
def popped_third : ℕ := 82

/-- The average percentage of popped kernels -/
def avg_percentage : ℚ := 82/100

theorem kernels_in_first_bag :
  (popped_first + popped_second + popped_third : ℚ) / 
  (first_bag + second_bag + third_bag : ℚ) = avg_percentage :=
by sorry

end NUMINAMATH_CALUDE_kernels_in_first_bag_l664_66439


namespace NUMINAMATH_CALUDE_right_angled_triangle_exists_l664_66409

theorem right_angled_triangle_exists (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 3) (h3 : c = 2) :
  a ^ 2 + b ^ 2 = c ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_exists_l664_66409


namespace NUMINAMATH_CALUDE_reciprocal_F_location_l664_66492

/-- A complex number in the first quadrant outside the unit circle -/
def F : ℂ :=
  sorry

/-- Theorem: The reciprocal of F is in the fourth quadrant inside the unit circle -/
theorem reciprocal_F_location :
  let z := F⁻¹
  0 < z.re ∧ z.im < 0 ∧ Complex.abs z < 1 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_F_location_l664_66492


namespace NUMINAMATH_CALUDE_number_problem_l664_66429

theorem number_problem : 
  ∃ x : ℝ, 2 * x = (10 / 100) * 900 ∧ x = 45 := by sorry

end NUMINAMATH_CALUDE_number_problem_l664_66429


namespace NUMINAMATH_CALUDE_farm_animal_count_l664_66437

/-- Represents a farm with chickens and buffalos -/
structure Farm where
  total_legs : ℕ
  num_chickens : ℕ

/-- Theorem stating that a farm with 4 chickens and 44 legs has 13 animals in total -/
theorem farm_animal_count (farm : Farm) 
  (h1 : farm.total_legs = 44)
  (h2 : farm.num_chickens = 4) : 
  farm.num_chickens + (farm.total_legs - 2 * farm.num_chickens) / 4 = 13 := by
  sorry

#check farm_animal_count

end NUMINAMATH_CALUDE_farm_animal_count_l664_66437


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l664_66435

/-- Calculates the length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 205 := by sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l664_66435


namespace NUMINAMATH_CALUDE_current_velocity_velocity_of_current_l664_66441

/-- Calculates the velocity of the current given rowing conditions -/
theorem current_velocity (still_water_speed : ℝ) (total_time : ℝ) (distance : ℝ) : ℝ :=
  let v : ℝ := 2  -- The velocity of the current we want to prove
  have h1 : still_water_speed = 10 := by sorry
  have h2 : total_time = 30 := by sorry
  have h3 : distance = 144 := by sorry
  have h4 : (distance / (still_water_speed - v) + distance / (still_water_speed + v)) = total_time := by sorry
  v

/-- The main theorem stating the velocity of the current -/
theorem velocity_of_current : current_velocity 10 30 144 = 2 := by sorry

end NUMINAMATH_CALUDE_current_velocity_velocity_of_current_l664_66441


namespace NUMINAMATH_CALUDE_mod_thirteen_five_eight_l664_66470

theorem mod_thirteen_five_eight (m : ℕ) : 
  13^5 % 8 = m → 0 ≤ m → m < 8 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_thirteen_five_eight_l664_66470


namespace NUMINAMATH_CALUDE_smallest_value_of_complex_sum_l664_66402

theorem smallest_value_of_complex_sum (a b c d : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_omega_power : ω^4 = 1)
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt (9/2) ∧
    ∀ (x : ℂ), x = a + b*ω + c*ω^2 + d*ω^3 → Complex.abs x ≥ m :=
  sorry

end NUMINAMATH_CALUDE_smallest_value_of_complex_sum_l664_66402


namespace NUMINAMATH_CALUDE_water_bucket_problem_l664_66423

theorem water_bucket_problem (initial_amount : ℝ) (added_amount : ℝ) :
  initial_amount = 3 →
  added_amount = 6.8 →
  initial_amount + added_amount = 9.8 :=
by
  sorry

end NUMINAMATH_CALUDE_water_bucket_problem_l664_66423


namespace NUMINAMATH_CALUDE_justin_and_tim_games_l664_66458

/-- The number of players in the league -/
def total_players : ℕ := 10

/-- The number of players in each game -/
def players_per_game : ℕ := 5

/-- The number of games where two specific players play together -/
def games_together : ℕ := 56

/-- The total number of possible game combinations -/
def total_combinations : ℕ := Nat.choose total_players players_per_game

theorem justin_and_tim_games :
  games_together = (players_per_game - 1) * total_combinations / (total_players - 1) :=
sorry

end NUMINAMATH_CALUDE_justin_and_tim_games_l664_66458


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l664_66405

def A : Set ℕ := {0, 1}
def B : Set ℕ := {0, 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l664_66405


namespace NUMINAMATH_CALUDE_largest_square_from_rectangle_l664_66416

theorem largest_square_from_rectangle (width length : ℕ) 
  (h_width : width = 63) (h_length : length = 42) :
  Nat.gcd width length = 21 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_from_rectangle_l664_66416


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l664_66494

/-- Given a hyperbola with foci on the y-axis, real axis length of 6,
    and asymptotes y = ± 3/2 x, its standard equation is y²/9 - x²/4 = 1 -/
theorem hyperbola_standard_equation
  (foci_on_y_axis : Bool)
  (real_axis_length : ℝ)
  (asymptote_slope : ℝ)
  (h_real_axis : real_axis_length = 6)
  (h_asymptote : asymptote_slope = 3/2) :
  ∃ (a b : ℝ),
    a = real_axis_length / 2 ∧
    b = a / asymptote_slope ∧
    (λ (x y : ℝ) => y^2 / a^2 - x^2 / b^2 = 1) =
    (λ (x y : ℝ) => y^2 / 9 - x^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l664_66494


namespace NUMINAMATH_CALUDE_exactly_two_lines_l664_66445

/-- Two lines in 3D space -/
structure Line3D where
  -- We'll represent a line by a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ := sorry

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- A point in 3D space -/
def Point3D := ℝ × ℝ × ℝ

/-- Count lines through a point forming a specific angle with two given lines -/
def count_lines_with_angle (a b : Line3D) (P : Point3D) (θ : ℝ) : ℕ := sorry

theorem exactly_two_lines 
  (a b : Line3D) (P : Point3D) 
  (h_skew : are_skew a b) 
  (h_angle : angle_between_lines a b = 40 * π / 180) :
  count_lines_with_angle a b P (30 * π / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_lines_l664_66445


namespace NUMINAMATH_CALUDE_rooks_attack_after_knight_moves_l664_66442

/-- Represents a position on the chess board -/
structure Position :=
  (row : Fin 15)
  (col : Fin 15)

/-- Represents a knight's move -/
inductive KnightMove
  | move1 : KnightMove  -- represents +2,+1 or -2,-1
  | move2 : KnightMove  -- represents +2,-1 or -2,+1
  | move3 : KnightMove  -- represents +1,+2 or -1,-2
  | move4 : KnightMove  -- represents +1,-2 or -1,+2

/-- Applies a knight's move to a position -/
def applyKnightMove (p : Position) (m : KnightMove) : Position :=
  sorry

/-- Checks if two positions are in the same row or column -/
def sameRowOrColumn (p1 p2 : Position) : Prop :=
  p1.row = p2.row ∨ p1.col = p2.col

theorem rooks_attack_after_knight_moves 
  (initial_positions : Fin 15 → Position)
  (h_no_initial_attack : ∀ i j, i ≠ j → ¬(sameRowOrColumn (initial_positions i) (initial_positions j)))
  (moves : Fin 15 → KnightMove) :
  ∃ i j, i ≠ j ∧ sameRowOrColumn (applyKnightMove (initial_positions i) (moves i)) (applyKnightMove (initial_positions j) (moves j)) :=
sorry

end NUMINAMATH_CALUDE_rooks_attack_after_knight_moves_l664_66442
