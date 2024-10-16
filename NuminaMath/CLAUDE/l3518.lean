import Mathlib

namespace NUMINAMATH_CALUDE_equation_A_is_quadratic_l3518_351874

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = -1 -/
def equation_A (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: The equation x² = -1 is a quadratic equation -/
theorem equation_A_is_quadratic : is_quadratic_equation equation_A := by
  sorry


end NUMINAMATH_CALUDE_equation_A_is_quadratic_l3518_351874


namespace NUMINAMATH_CALUDE_factor_decomposition_96_l3518_351864

theorem factor_decomposition_96 : 
  ∃ (x y : ℤ), x * y = 96 ∧ x^2 + y^2 = 208 := by
  sorry

end NUMINAMATH_CALUDE_factor_decomposition_96_l3518_351864


namespace NUMINAMATH_CALUDE_problem_solution_l3518_351871

def star (a b : ℕ) : ℕ := a^b + a*b

theorem problem_solution (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (h : star a b = 40) : a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3518_351871


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3518_351828

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : Real.sin (α - π / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3518_351828


namespace NUMINAMATH_CALUDE_not_right_triangle_l3518_351836

theorem not_right_triangle (A B C : ℝ) (h : A + B + C = 180) 
  (h_ratio : A / 3 = B / 4 ∧ B / 4 = C / 5) : 
  A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l3518_351836


namespace NUMINAMATH_CALUDE_only_D_is_simple_random_sample_l3518_351887

/-- Represents a sampling method --/
inductive SamplingMethod
| A  -- Every 1 million postcards form a lottery group
| B  -- Sample a package every 30 minutes
| C  -- Draw from different staff categories
| D  -- Select 3 out of 10 products randomly

/-- Defines what constitutes a simple random sample --/
def isSimpleRandomSample (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.A => false  -- Fixed interval
  | SamplingMethod.B => false  -- Fixed interval
  | SamplingMethod.C => false  -- Stratified sampling
  | SamplingMethod.D => true   -- Equal probability for each item

/-- Theorem stating that only method D is a simple random sample --/
theorem only_D_is_simple_random_sample :
  ∀ (method : SamplingMethod), isSimpleRandomSample method ↔ method = SamplingMethod.D :=
by sorry

end NUMINAMATH_CALUDE_only_D_is_simple_random_sample_l3518_351887


namespace NUMINAMATH_CALUDE_function_inequality_l3518_351898

-- Define the condition (1-x)/f'(x) ≥ 0
def condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (1 - x) / (deriv f x) ≥ 0

-- State the theorem
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) (h : condition f) :
  f 0 + f 2 < 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3518_351898


namespace NUMINAMATH_CALUDE_integral_minimization_l3518_351865

open Real

/-- Minimization of integral subject to constraints -/
theorem integral_minimization (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  let f (p q : ℝ) := ∫ x in a..b, (p * x + q - log x)
  let constraint (p q : ℝ) := ∀ x ∈ Set.Icc a b, p * x + q ≥ log x
  let p_min := 2 / (a + b)
  let q_min := log ((a + b) / 2) - 1
  let min_value := (b - a) * log ((a + b) / 2) + b - a - b * log b + a * log a
  (∀ p q, constraint p q → f p q ≥ f p_min q_min) ∧
  f p_min q_min = min_value :=
sorry

end NUMINAMATH_CALUDE_integral_minimization_l3518_351865


namespace NUMINAMATH_CALUDE_problem_solution_l3518_351878

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 4| - t

-- State the theorem
theorem problem_solution :
  ∀ t : ℝ,
  (∀ x : ℝ, f t x ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) →
  (t = 1 ∧
   ∀ a b c : ℝ,
   a > 0 → b > 0 → c > 0 →
   a + b + c = t →
   a^2 / b + b^2 / c + c^2 / a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3518_351878


namespace NUMINAMATH_CALUDE_triangle_properties_l3518_351806

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 1 ∧ Real.cos t.C + (2 * t.a + t.c) * Real.cos t.B = 0

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (∀ (s : ℝ), s = 1/2 * t.a * t.c * Real.sin t.B → s ≤ Real.sqrt 3 / 12) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3518_351806


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l3518_351844

theorem parametric_to_cartesian (θ : ℝ) :
  let x := Real.cos θ / (1 + Real.cos θ)
  let y := Real.sin θ / (1 + Real.cos θ)
  y^2 = -2 * (x - 1/2) :=
by sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l3518_351844


namespace NUMINAMATH_CALUDE_min_sum_of_roots_l3518_351855

theorem min_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0) 
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) : 
  a + b ≥ 48/27 + 9/4 * (9216/6561)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_roots_l3518_351855


namespace NUMINAMATH_CALUDE_soccer_games_per_month_l3518_351801

theorem soccer_games_per_month 
  (total_games : ℕ) 
  (season_months : ℕ) 
  (h1 : total_games = 27) 
  (h2 : season_months = 3) 
  (h3 : total_games % season_months = 0) : 
  total_games / season_months = 9 := by
sorry

end NUMINAMATH_CALUDE_soccer_games_per_month_l3518_351801


namespace NUMINAMATH_CALUDE_odd_number_characterization_l3518_351835

theorem odd_number_characterization (n : ℤ) : 
  Odd n ↔ ∃ k : ℤ, n = 2 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_odd_number_characterization_l3518_351835


namespace NUMINAMATH_CALUDE_chocolate_leftover_l3518_351882

/-- Calculates the amount of chocolate left over when making cookies -/
theorem chocolate_leftover (dough : ℝ) (total_chocolate : ℝ) (chocolate_percentage : ℝ) : 
  dough = 36 → 
  total_chocolate = 13 → 
  chocolate_percentage = 0.20 → 
  (total_chocolate - (chocolate_percentage * (dough + (chocolate_percentage * (dough + total_chocolate) / (1 - chocolate_percentage))))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_leftover_l3518_351882


namespace NUMINAMATH_CALUDE_vasyas_birthday_l3518_351805

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

theorem vasyas_birthday (today : DayOfWeek) 
  (h1 : next_day (next_day today) = DayOfWeek.Sunday) -- Sunday is the day after tomorrow
  (h2 : ∃ birthday : DayOfWeek, next_day birthday = today) -- Today is the day after Vasya's birthday
  : ∃ birthday : DayOfWeek, birthday = DayOfWeek.Thursday := by
  sorry

end NUMINAMATH_CALUDE_vasyas_birthday_l3518_351805


namespace NUMINAMATH_CALUDE_cookies_per_bag_l3518_351875

/-- Given 26 bags with an equal number of cookies and 52 cookies in total,
    prove that each bag contains 2 cookies. -/
theorem cookies_per_bag :
  ∀ (bags : ℕ) (total_cookies : ℕ) (cookies_per_bag : ℕ),
    bags = 26 →
    total_cookies = 52 →
    total_cookies = bags * cookies_per_bag →
    cookies_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l3518_351875


namespace NUMINAMATH_CALUDE_smallest_n_terminating_with_3_l3518_351811

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

def contains_digit_3 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 10 ∧ (n / 10^d) % 10 = 3

theorem smallest_n_terminating_with_3 :
  ∀ n : ℕ, n > 0 →
    (is_terminating_decimal n ∧ contains_digit_3 n) →
    n ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_terminating_with_3_l3518_351811


namespace NUMINAMATH_CALUDE_x_power_2187_minus_reciprocal_l3518_351800

theorem x_power_2187_minus_reciprocal (x : ℂ) :
  x - (1 / x) = Complex.I * Real.sqrt 2 →
  x^2187 - (1 / x^2187) = Complex.I * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_x_power_2187_minus_reciprocal_l3518_351800


namespace NUMINAMATH_CALUDE_slope_of_line_from_equation_l3518_351818

theorem slope_of_line_from_equation (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : x₁ ≠ x₂) 
  (h₂ : 4 / x₁ + 5 / y₁ = 0) 
  (h₃ : 4 / x₂ + 5 / y₂ = 0) : 
  (y₂ - y₁) / (x₂ - x₁) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_from_equation_l3518_351818


namespace NUMINAMATH_CALUDE_total_blankets_collected_l3518_351859

/-- Represents the blanket collection problem over three days --/
def blanket_collection (original_members : ℕ) (new_members : ℕ) 
  (blankets_per_original : ℕ) (blankets_per_new : ℕ) 
  (school_blankets : ℕ) (online_blankets : ℕ) : ℕ :=
  let day1 := original_members * blankets_per_original
  let day2_team := original_members * blankets_per_original + new_members * blankets_per_new
  let day2 := day2_team + 3 * day1
  let day3 := school_blankets + online_blankets
  day1 + day2 + day3

/-- The main theorem stating the total number of blankets collected --/
theorem total_blankets_collected : 
  blanket_collection 15 5 2 4 22 30 = 222 := by
  sorry

end NUMINAMATH_CALUDE_total_blankets_collected_l3518_351859


namespace NUMINAMATH_CALUDE_three_digit_numbers_19_times_sum_of_digits_l3518_351890

def isValidNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n = 19 * (n / 100 + (n / 10 % 10) + (n % 10))

theorem three_digit_numbers_19_times_sum_of_digits :
  {n : ℕ | isValidNumber n} = {114, 133, 152, 171, 190, 209, 228, 247, 266, 285, 399} :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_19_times_sum_of_digits_l3518_351890


namespace NUMINAMATH_CALUDE_pitcher_problem_l3518_351831

theorem pitcher_problem (C : ℝ) (h : C > 0) :
  let juice_volume := (2 / 3) * C
  let num_cups := 6
  let cup_volume := juice_volume / num_cups
  cup_volume / C = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pitcher_problem_l3518_351831


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3518_351813

theorem least_positive_integer_with_remainders : ∃ (M : ℕ), 
  (M > 0) ∧ 
  (M % 6 = 5) ∧ 
  (M % 7 = 6) ∧ 
  (M % 9 = 8) ∧ 
  (M % 10 = 9) ∧ 
  (M % 11 = 10) ∧ 
  (∀ (N : ℕ), 
    (N > 0) ∧ 
    (N % 6 = 5) ∧ 
    (N % 7 = 6) ∧ 
    (N % 9 = 8) ∧ 
    (N % 10 = 9) ∧ 
    (N % 11 = 10) → 
    M ≤ N) ∧
  M = 6929 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3518_351813


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l3518_351857

def total_knights : ℕ := 30
def chosen_knights : ℕ := 4

def prob_adjacent_knights : ℚ :=
  1 - (26 * 24 * 22 * 20 : ℚ) / (26 * 27 * 28 * 29 : ℚ)

theorem adjacent_knights_probability :
  prob_adjacent_knights = 553 / 1079 := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l3518_351857


namespace NUMINAMATH_CALUDE_even_odd_product_sum_zero_l3518_351868

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x ∈ ℝ -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

/-- For even function f and odd function g, f(-x)g(-x) + f(x)g(x) = 0 for all x ∈ ℝ -/
theorem even_odd_product_sum_zero (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
    ∀ x : ℝ, f (-x) * g (-x) + f x * g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_product_sum_zero_l3518_351868


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3518_351870

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.sin (α + π/2) = 3/5) : Real.sin α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3518_351870


namespace NUMINAMATH_CALUDE_phone_reselling_profit_l3518_351841

theorem phone_reselling_profit (initial_investment : ℝ) (profit_ratio : ℝ) (selling_price : ℝ) :
  initial_investment = 3000 →
  profit_ratio = 1 / 3 →
  selling_price = 20 →
  (initial_investment * (1 + profit_ratio)) / selling_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_phone_reselling_profit_l3518_351841


namespace NUMINAMATH_CALUDE_f_min_at_neg_three_l3518_351893

/-- The function f(x) = x^2 + 6x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 1

/-- Theorem stating that f(x) is minimized when x = -3 -/
theorem f_min_at_neg_three :
  ∀ x : ℝ, f (-3) ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_f_min_at_neg_three_l3518_351893


namespace NUMINAMATH_CALUDE_chessboard_invariant_l3518_351895

/-- Represents a chessboard configuration -/
def Chessboard := Matrix (Fin 8) (Fin 8) Int

/-- Initial chessboard configuration -/
def initialBoard : Chessboard :=
  fun i j => if i = 1 ∧ j = 7 then -1 else 1

/-- Represents a move (changing signs in a row or column) -/
inductive Move
  | row (i : Fin 8)
  | col (j : Fin 8)

/-- Apply a move to a chessboard -/
def applyMove (b : Chessboard) (m : Move) : Chessboard :=
  match m with
  | Move.row i => fun r c => if r = i then -b r c else b r c
  | Move.col j => fun r c => if c = j then -b r c else b r c

/-- Apply a sequence of moves to a chessboard -/
def applyMoves (b : Chessboard) : List Move → Chessboard
  | [] => b
  | m :: ms => applyMoves (applyMove b m) ms

/-- Product of all numbers on the board -/
def boardProduct (b : Chessboard) : Int :=
  (Finset.univ.prod fun i => Finset.univ.prod fun j => b i j)

/-- Main theorem -/
theorem chessboard_invariant (moves : List Move) :
    boardProduct (applyMoves initialBoard moves) = -1 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_invariant_l3518_351895


namespace NUMINAMATH_CALUDE_sin_870_degrees_l3518_351860

theorem sin_870_degrees : Real.sin (870 * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_870_degrees_l3518_351860


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3518_351869

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -5
  let y : ℝ := 4
  second_quadrant x y :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3518_351869


namespace NUMINAMATH_CALUDE_circumcircle_of_triangle_ABP_l3518_351850

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the external point P
def point_P : ℝ × ℝ := (4, 2)

-- Define the property of A and B being points of tangency
def tangent_points (A B : ℝ × ℝ) : Prop :=
  given_circle A.1 A.2 ∧ given_circle B.1 B.2 ∧
  (∀ t : ℝ, t ≠ 0 → ¬(given_circle (A.1 + t * (point_P.1 - A.1)) (A.2 + t * (point_P.2 - A.2)))) ∧
  (∀ t : ℝ, t ≠ 0 → ¬(given_circle (B.1 + t * (point_P.1 - B.1)) (B.2 + t * (point_P.2 - B.2))))

-- Define the equation of the circumcircle
def circumcircle_equation (x y : ℝ) : Prop := (x - 4)^2 + (y - 2)^2 = 16

-- Theorem statement
theorem circumcircle_of_triangle_ABP :
  ∀ A B : ℝ × ℝ, tangent_points A B →
  ∀ x y : ℝ, (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 →
  (x - point_P.1)^2 + (y - point_P.2)^2 = (x - A.1)^2 + (y - A.2)^2 →
  circumcircle_equation x y :=
sorry

end NUMINAMATH_CALUDE_circumcircle_of_triangle_ABP_l3518_351850


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l3518_351884

/-- Given that (1 - 2/x)³ = a₀ + a₁·(1/x) + a₂·(1/x)² + a₃·(1/x)³, prove that a₁ + a₂ = 6 -/
theorem binomial_expansion_sum (x : ℝ) (a₀ a₁ a₂ a₃ : ℝ) 
  (h : (1 - 2/x)^3 = a₀ + a₁ * (1/x) + a₂ * (1/x)^2 + a₃ * (1/x)^3) :
  a₁ + a₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l3518_351884


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3518_351802

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  (∀ x : ℝ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3518_351802


namespace NUMINAMATH_CALUDE_sum_of_cubes_and_reciprocals_l3518_351891

/-- Given real numbers x and y satisfying x + y = 6 and x * y = 5,
    prove that x + (x^3 / y^2) + (y^3 / x^2) + y = 137.04 -/
theorem sum_of_cubes_and_reciprocals (x y : ℝ) 
  (h1 : x + y = 6) (h2 : x * y = 5) : 
  x + (x^3 / y^2) + (y^3 / x^2) + y = 137.04 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_and_reciprocals_l3518_351891


namespace NUMINAMATH_CALUDE_parabola_vertex_distance_l3518_351808

/-- The parabola equation -/
def parabola (x c : ℝ) : ℝ := x^2 - 6*x + c - 2

/-- The vertex of the parabola -/
def vertex (c : ℝ) : ℝ × ℝ := (3, c - 11)

/-- The distance from the vertex to the x-axis -/
def distance_to_x_axis (c : ℝ) : ℝ := |c - 11|

theorem parabola_vertex_distance (c : ℝ) :
  distance_to_x_axis c = 3 → c = 8 ∨ c = 14 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_distance_l3518_351808


namespace NUMINAMATH_CALUDE_sandbag_weight_l3518_351832

/-- Calculates the weight of a partially filled sandbag with a heavier filling material -/
theorem sandbag_weight (bag_capacity : ℝ) (fill_percentage : ℝ) (material_weight_increase : ℝ) : 
  bag_capacity > 0 → 
  fill_percentage > 0 → 
  fill_percentage ≤ 1 → 
  material_weight_increase ≥ 0 →
  let sand_weight := bag_capacity * fill_percentage
  let material_weight := sand_weight * (1 + material_weight_increase)
  bag_capacity + material_weight = 530 :=
by
  sorry

#check sandbag_weight 250 0.8 0.4

end NUMINAMATH_CALUDE_sandbag_weight_l3518_351832


namespace NUMINAMATH_CALUDE_prob_ace_ten_queen_correct_l3518_351849

/-- The probability of drawing an Ace, then a 10, then a Queen from a standard 52-card deck without replacement -/
def prob_ace_ten_queen : ℚ := 8 / 16575

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of 10s in a standard deck -/
def num_tens : ℕ := 4

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

theorem prob_ace_ten_queen_correct (d : Deck) : 
  (num_aces : ℚ) / 52 * (num_tens : ℚ) / 51 * (num_queens : ℚ) / 50 = prob_ace_ten_queen :=
sorry

end NUMINAMATH_CALUDE_prob_ace_ten_queen_correct_l3518_351849


namespace NUMINAMATH_CALUDE_train_speed_l3518_351821

/-- Calculates the speed of a train given its length and time to cross an electric pole. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 800) (h2 : time = 20) :
  length / time = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3518_351821


namespace NUMINAMATH_CALUDE_david_pushups_l3518_351804

theorem david_pushups (zachary_pushups : ℕ) : 
  zachary_pushups + (zachary_pushups + 49) = 53 →
  zachary_pushups + 49 = 51 := by
  sorry

end NUMINAMATH_CALUDE_david_pushups_l3518_351804


namespace NUMINAMATH_CALUDE_ellipse_equal_angles_l3518_351803

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def onEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Represents a chord of the ellipse -/
structure Chord where
  A : Point
  B : Point

/-- Checks if a chord passes through a given point -/
def chordThroughPoint (c : Chord) (p : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    p.x = c.A.x + t * (c.B.x - c.A.x) ∧
    p.y = c.A.y + t * (c.B.y - c.A.y)

/-- Calculates the angle between three points -/
def angle (A B C : Point) : ℝ :=
  sorry -- Angle calculation implementation

/-- The main theorem to prove -/
theorem ellipse_equal_angles (e : Ellipse) (F P : Point) :
  e.a = 2 ∧ e.b = 1 ∧
  F.x = Real.sqrt 3 ∧ F.y = 0 ∧
  P.x = 2 ∧ P.y = 0 →
  ∀ (c : Chord), onEllipse e c.A ∧ onEllipse e c.B ∧ chordThroughPoint c F →
    angle c.A P F = angle c.B P F :=
  sorry

end NUMINAMATH_CALUDE_ellipse_equal_angles_l3518_351803


namespace NUMINAMATH_CALUDE_roots_and_minimum_value_l3518_351824

def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - x| - a*x

theorem roots_and_minimum_value :
  (∀ x, f (1/3) x = 0 ↔ x = 0 ∨ x = 2/3 ∨ x = 4/3) ∧
  (∀ a, a ≤ -1 →
    (∀ x ∈ Set.Icc (-2) 3, f a x ≥ 
      (if a ≤ -5 then 2*a + 6 else -(a+1)^2/4)) ∧
    (∃ x ∈ Set.Icc (-2) 3, f a x = 
      (if a ≤ -5 then 2*a + 6 else -(a+1)^2/4))) := by
  sorry

end NUMINAMATH_CALUDE_roots_and_minimum_value_l3518_351824


namespace NUMINAMATH_CALUDE_raindrop_probability_l3518_351819

/-- The probability of a raindrop landing on the third slope of a triangular pyramid roof -/
theorem raindrop_probability (α β : Real) : 
  -- The roof is a triangular pyramid with all plane angles at the vertex being right angles
  -- The red slope is inclined at an angle α to the horizontal
  -- The blue slope is inclined at an angle β to the horizontal
  -- We assume 0 ≤ α ≤ π/2 and 0 ≤ β ≤ π/2 to ensure valid angles
  0 ≤ α ∧ α ≤ π/2 ∧ 0 ≤ β ∧ β ≤ π/2 →
  -- The probability of a raindrop landing on the green slope
  ∃ (p : Real), p = 1 - (Real.cos α)^2 - (Real.cos β)^2 ∧ 0 ≤ p ∧ p ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_raindrop_probability_l3518_351819


namespace NUMINAMATH_CALUDE_john_school_year_hours_l3518_351825

/-- Calculates the number of hours John needs to work per week during the school year -/
def school_year_hours (summer_hours : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℕ :=
  let summer_hourly_rate := summer_earnings / (summer_hours * summer_weeks)
  let school_year_weekly_earnings := school_year_earnings / school_year_weeks
  school_year_weekly_earnings / summer_hourly_rate

/-- Theorem stating that John needs to work 8 hours per week during the school year -/
theorem john_school_year_hours : 
  school_year_hours 40 10 4000 50 4000 = 8 := by
  sorry

end NUMINAMATH_CALUDE_john_school_year_hours_l3518_351825


namespace NUMINAMATH_CALUDE_last_three_digits_factorial_sum_15_l3518_351899

def last_three_digits (n : ℕ) : ℕ := n % 1000

def factorial_sum (n : ℕ) : ℕ :=
  (List.range n).map Nat.factorial |> List.sum

theorem last_three_digits_factorial_sum_15 :
  last_three_digits (factorial_sum 15) = 193 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_factorial_sum_15_l3518_351899


namespace NUMINAMATH_CALUDE_power_equation_solution_l3518_351840

theorem power_equation_solution : ∃ x : ℕ, (5 ^ 5) * (9 ^ 3) = 3 * (15 ^ x) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3518_351840


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_II_l3518_351867

/-- Represents a linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Checks if a point (x, y) is in Quadrant II -/
def isInQuadrantII (x : ℝ) (y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- Theorem: The linear function y = 3x - 2 does not pass through Quadrant II -/
theorem linear_function_not_in_quadrant_II :
  let f : LinearFunction := { m := 3, b := -2 }
  ∀ x y : ℝ, y = f.m * x + f.b → ¬(isInQuadrantII x y) :=
by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_II_l3518_351867


namespace NUMINAMATH_CALUDE_parabola_theorem_l3518_351843

/-- Represents a parabola y² = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  pos : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about a parabola with specific properties -/
theorem parabola_theorem (par : Parabola) (directrix_point : Point)
  (h_directrix : directrix_point.x = -1 ∧ directrix_point.y = 1) :
  -- Part I: The equation of the parabola is y² = 4x
  par.p = 2 ∧
  -- Part II: If a line passing through the focus intersects the parabola at points A and B
  --          with |AB| = 5, then the line's equation is either 2x - y - 2 = 0 or 2x + y - 2 = 0
  ∀ (A B : Point),
    (A.y ^ 2 = 4 * A.x ∧ B.y ^ 2 = 4 * B.x) →  -- A and B are on the parabola
    (∃ (k : ℝ), k ≠ 0 ∧                       -- Line equation: y = k(x-1)
      A.y = k * (A.x - 1) ∧ B.y = k * (B.x - 1)) →
    (A.x - B.x) ^ 2 + (A.y - B.y) ^ 2 = 25 →  -- |AB| = 5
    (2 * A.x - A.y - 2 = 0 ∧ 2 * B.x - B.y - 2 = 0) ∨
    (2 * A.x + A.y - 2 = 0 ∧ 2 * B.x + B.y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_theorem_l3518_351843


namespace NUMINAMATH_CALUDE_bankers_discount_calculation_l3518_351822

/-- Banker's discount calculation -/
theorem bankers_discount_calculation
  (true_discount : ℝ)
  (sum_due : ℝ)
  (h1 : true_discount = 60)
  (h2 : sum_due = 360) :
  true_discount + (true_discount^2 / sum_due) = 70 :=
by sorry

end NUMINAMATH_CALUDE_bankers_discount_calculation_l3518_351822


namespace NUMINAMATH_CALUDE_unique_grid_solution_l3518_351823

-- Define the grid
def Grid := Fin 3 → Fin 3 → Option Char

-- Define adjacency
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨
  (i = k ∧ j.val = l.val + 1) ∨
  (i.val + 1 = k.val ∧ j = l) ∨
  (i.val = k.val + 1 ∧ j = l) ∨
  (i.val + 1 = k.val ∧ j.val + 1 = l.val) ∨
  (i.val + 1 = k.val ∧ j.val = l.val + 1) ∨
  (i.val = k.val + 1 ∧ j.val + 1 = l.val) ∨
  (i.val = k.val + 1 ∧ j.val = l.val + 1)

-- Define the constraints
def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ [none, some 'A', some 'B', some 'C']) ∧
  (∀ i, ∃! j, g i j = some 'A') ∧
  (∀ i, ∃! j, g i j = some 'B') ∧
  (∀ i, ∃! j, g i j = some 'C') ∧
  (∀ j, ∃! i, g i j = some 'A') ∧
  (∀ j, ∃! i, g i j = some 'B') ∧
  (∀ j, ∃! i, g i j = some 'C') ∧
  (∀ i j k l, adjacent i j k l → g i j ≠ g k l) ∧
  (g 0 1 = none ∧ g 1 0 = none)

-- Define the diagonal string
def diagonal_string (g : Grid) : String :=
  String.mk [
    (g 0 0).getD 'X',
    (g 1 1).getD 'X',
    (g 2 2).getD 'X'
  ]

-- The theorem to prove
theorem unique_grid_solution :
  ∀ g : Grid, valid_grid g → diagonal_string g = "XXC" := by
  sorry

end NUMINAMATH_CALUDE_unique_grid_solution_l3518_351823


namespace NUMINAMATH_CALUDE_debt_average_payment_l3518_351886

theorem debt_average_payment 
  (total_installments : ℕ) 
  (first_payment_count : ℕ) 
  (first_payment_amount : ℚ) 
  (payment_increase : ℚ) :
  total_installments = 65 →
  first_payment_count = 20 →
  first_payment_amount = 410 →
  payment_increase = 65 →
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + payment_increase
  let total_amount := first_payment_count * first_payment_amount + 
                      remaining_payment_count * remaining_payment_amount
  total_amount / total_installments = 455 := by
sorry

end NUMINAMATH_CALUDE_debt_average_payment_l3518_351886


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3518_351809

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_two : x + y + z = 2) : 
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3518_351809


namespace NUMINAMATH_CALUDE_max_scheduling_ways_after_15_games_l3518_351814

/-- Represents a chess tournament between schoolchildren and students. -/
structure ChessTournament where
  schoolchildren : Nat
  students : Nat
  total_games : Nat
  scheduled_games : Nat

/-- The maximum number of ways to schedule one game in the next round. -/
def max_scheduling_ways (tournament : ChessTournament) : Nat :=
  tournament.total_games - tournament.scheduled_games

/-- The theorem stating the maximum number of ways to schedule one game
    after uniquely scheduling 15 games in a tournament with 15 schoolchildren
    and 15 students. -/
theorem max_scheduling_ways_after_15_games
  (tournament : ChessTournament)
  (h1 : tournament.schoolchildren = 15)
  (h2 : tournament.students = 15)
  (h3 : tournament.total_games = tournament.schoolchildren * tournament.students)
  (h4 : tournament.scheduled_games = 15) :
  max_scheduling_ways tournament = 120 := by
  sorry


end NUMINAMATH_CALUDE_max_scheduling_ways_after_15_games_l3518_351814


namespace NUMINAMATH_CALUDE_inequality_proof_l3518_351863

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : Real.sqrt a + Real.sqrt b = 2) :
  (a * Real.sqrt b + b * Real.sqrt a ≤ 2) ∧ (2 ≤ a^2 + b^2) ∧ (a^2 + b^2 < 16) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3518_351863


namespace NUMINAMATH_CALUDE_november_rainfall_l3518_351866

/-- The total rainfall in November for a northwestern town -/
def total_rainfall (first_half_daily_rainfall : ℝ) (days_in_november : ℕ) : ℝ :=
  let first_half := 15
  let second_half := days_in_november - first_half
  let first_half_total := first_half * first_half_daily_rainfall
  let second_half_total := second_half * (2 * first_half_daily_rainfall)
  first_half_total + second_half_total

/-- Theorem stating the total rainfall in November is 180 inches -/
theorem november_rainfall : total_rainfall 4 30 = 180 := by
  sorry

end NUMINAMATH_CALUDE_november_rainfall_l3518_351866


namespace NUMINAMATH_CALUDE_probability_of_six_on_fifth_roll_l3518_351896

def fair_die_prob : ℚ := 1 / 6
def biased_die_6_prob : ℚ := 2 / 3
def biased_die_6_other_prob : ℚ := 1 / 15
def biased_die_3_prob : ℚ := 1 / 2
def biased_die_3_other_prob : ℚ := 1 / 10

def initial_pick_prob : ℚ := 1 / 3

def observed_rolls : ℕ := 4
def observed_sixes : ℕ := 3
def observed_threes : ℕ := 1

theorem probability_of_six_on_fifth_roll :
  let fair_prob := initial_pick_prob * (fair_die_prob ^ observed_sixes * fair_die_prob ^ observed_threes)
  let biased_6_prob := initial_pick_prob * (biased_die_6_prob ^ observed_sixes * biased_die_6_other_prob ^ observed_threes)
  let biased_3_prob := initial_pick_prob * (biased_die_3_other_prob ^ observed_sixes * biased_die_3_prob ^ observed_threes)
  let total_prob := fair_prob + biased_6_prob + biased_3_prob
  (biased_6_prob / total_prob) * biased_die_6_prob = 8 / 135 / (3457.65 / 3888) * (2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_probability_of_six_on_fifth_roll_l3518_351896


namespace NUMINAMATH_CALUDE_sum_of_ratios_zero_l3518_351888

theorem sum_of_ratios_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_sum : a + b + c = 0) :
  a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_zero_l3518_351888


namespace NUMINAMATH_CALUDE_total_seashells_is_fifty_l3518_351827

/-- The number of seashells Tim found -/
def tim_seashells : ℕ := 37

/-- The number of seashells Sally found -/
def sally_seashells : ℕ := 13

/-- The total number of seashells found by Tim and Sally -/
def total_seashells : ℕ := tim_seashells + sally_seashells

/-- Theorem stating that the total number of seashells found is 50 -/
theorem total_seashells_is_fifty : total_seashells = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_is_fifty_l3518_351827


namespace NUMINAMATH_CALUDE_max_value_of_a_l3518_351816

theorem max_value_of_a (a b c d : ℤ) 
  (h1 : a < 2*b) 
  (h2 : b < 3*c) 
  (h3 : c < 4*d) 
  (h4 : d < 100) : 
  a ≤ 2367 ∧ ∃ (a₀ b₀ c₀ d₀ : ℤ), a₀ = 2367 ∧ a₀ < 2*b₀ ∧ b₀ < 3*c₀ ∧ c₀ < 4*d₀ ∧ d₀ < 100 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3518_351816


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l3518_351894

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (|x| - 3) / (x + 3) = 0 → x = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l3518_351894


namespace NUMINAMATH_CALUDE_friend_reading_time_l3518_351838

theorem friend_reading_time (my_time : ℝ) (friend_speed_multiplier : ℝ) (distraction_time : ℝ) :
  my_time = 1.5 →
  friend_speed_multiplier = 5 →
  distraction_time = 0.25 →
  (my_time * 60) / friend_speed_multiplier + distraction_time = 33 :=
by sorry

end NUMINAMATH_CALUDE_friend_reading_time_l3518_351838


namespace NUMINAMATH_CALUDE_union_of_sets_l3518_351847

def set_A : Set ℝ := {x | |x - 1| < 3}
def set_B : Set ℝ := {x | x^2 - 4*x < 0}

theorem union_of_sets : set_A ∪ set_B = Set.Ioo (-2) 4 := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l3518_351847


namespace NUMINAMATH_CALUDE_multiplier_proof_l3518_351834

theorem multiplier_proof (number : ℝ) (difference : ℝ) (subtractor : ℝ) :
  number = 15.0 →
  difference = 40 →
  subtractor = 5 →
  ∃ (multiplier : ℝ), multiplier * number - subtractor = difference ∧ multiplier = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_proof_l3518_351834


namespace NUMINAMATH_CALUDE_difference_of_greatest_values_l3518_351872

def is_valid_three_digit_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000

def hundreds_digit (x : ℕ) : ℕ := (x / 100) % 10
def tens_digit (x : ℕ) : ℕ := (x / 10) % 10
def units_digit (x : ℕ) : ℕ := x % 10

def satisfies_conditions (x : ℕ) : Prop :=
  let a := hundreds_digit x
  let b := tens_digit x
  let c := units_digit x
  is_valid_three_digit_number x ∧ 2 * a = b ∧ b = 4 * c ∧ a > 0

theorem difference_of_greatest_values : 
  ∃ x₁ x₂ : ℕ, satisfies_conditions x₁ ∧ satisfies_conditions x₂ ∧
  (∀ x : ℕ, satisfies_conditions x → x ≤ x₁) ∧
  (∀ x : ℕ, satisfies_conditions x → x ≠ x₁ → x ≤ x₂) ∧
  x₁ - x₂ = 241 :=
sorry

end NUMINAMATH_CALUDE_difference_of_greatest_values_l3518_351872


namespace NUMINAMATH_CALUDE_integral_sum_equals_pi_over_four_plus_ln_two_l3518_351812

theorem integral_sum_equals_pi_over_four_plus_ln_two :
  (∫ (x : ℝ) in (0)..(1), Real.sqrt (1 - x^2)) + (∫ (x : ℝ) in (1)..(2), 1/x) = π/4 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sum_equals_pi_over_four_plus_ln_two_l3518_351812


namespace NUMINAMATH_CALUDE_belle_treats_cost_l3518_351839

/-- The cost of feeding Belle her treats for a week -/
def cost_per_week : ℚ :=
  let biscuits_per_day : ℕ := 4
  let bones_per_day : ℕ := 2
  let biscuit_cost : ℚ := 1/4
  let bone_cost : ℚ := 1
  let days_per_week : ℕ := 7
  (biscuits_per_day * biscuit_cost + bones_per_day * bone_cost) * days_per_week

/-- Theorem stating that the cost of feeding Belle her treats for a week is $21 -/
theorem belle_treats_cost : cost_per_week = 21 := by
  sorry

end NUMINAMATH_CALUDE_belle_treats_cost_l3518_351839


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3518_351854

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), C = (2 + 2 * Real.sqrt 7) / 3 ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z - 1)) ∧
  (∀ (C' : ℝ), C' > C → ∃ (x y z : ℝ), x^2 + y^2 + z^2 + 2 < C' * (x + y + z - 1)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3518_351854


namespace NUMINAMATH_CALUDE_anna_apple_count_l3518_351826

/-- The number of apples Anna ate on Tuesday -/
def tuesday_apples : ℕ := 4

/-- The number of apples Anna ate on Wednesday -/
def wednesday_apples : ℕ := 2 * tuesday_apples

/-- The number of apples Anna ate on Thursday -/
def thursday_apples : ℕ := tuesday_apples / 2

/-- The total number of apples Anna ate over the three days -/
def total_apples : ℕ := tuesday_apples + wednesday_apples + thursday_apples

theorem anna_apple_count : total_apples = 14 := by
  sorry

end NUMINAMATH_CALUDE_anna_apple_count_l3518_351826


namespace NUMINAMATH_CALUDE_smallest_square_with_40_and_49_existence_of_2000_square_smallest_2000_square_l3518_351883

theorem smallest_square_with_40_and_49 :
  ∀ n : ℕ, 
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n * n = 40 * 40 * a + 49 * 49 * b) →
    n ≥ 2000 :=
by sorry

theorem existence_of_2000_square :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2000 * 2000 = 40 * 40 * a + 49 * 49 * b :=
by sorry

theorem smallest_2000_square :
  (∀ n : ℕ, 
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n * n = 40 * 40 * a + 49 * 49 * b) →
    n ≥ 2000) ∧
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2000 * 2000 = 40 * 40 * a + 49 * 49 * b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_with_40_and_49_existence_of_2000_square_smallest_2000_square_l3518_351883


namespace NUMINAMATH_CALUDE_m_range_l3518_351837

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x, -m * x^2 + 2*x - m > 0

def q (m : ℝ) : Prop := ∀ x > 0, (4/x + x - (m - 1)) > 2

-- Define the theorem
theorem m_range :
  (∃ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (∃ m : ℝ, m ≥ -1 ∧ m < 3) ∧ (∀ m : ℝ, m < -1 ∨ m ≥ 3 → ¬(p m ∨ q m) ∨ (p m ∧ q m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3518_351837


namespace NUMINAMATH_CALUDE_brothers_age_difference_l3518_351830

/-- Bush and Matt are brothers with an age difference --/
def age_difference (bush_age : ℕ) (matt_future_age : ℕ) (years_to_future : ℕ) : ℕ :=
  (matt_future_age - years_to_future) - bush_age

/-- Theorem stating the age difference between Matt and Bush --/
theorem brothers_age_difference :
  age_difference 12 25 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_difference_l3518_351830


namespace NUMINAMATH_CALUDE_root_in_interval_l3518_351817

def f (x : ℝ) := x^3 - x - 3

theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_l3518_351817


namespace NUMINAMATH_CALUDE_sqrt_054_in_terms_of_sqrt_2_and_sqrt_3_l3518_351876

theorem sqrt_054_in_terms_of_sqrt_2_and_sqrt_3 (a b : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) : 
  Real.sqrt 0.54 = 0.3 * a * b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_054_in_terms_of_sqrt_2_and_sqrt_3_l3518_351876


namespace NUMINAMATH_CALUDE_tiling_impossibility_l3518_351852

/-- Represents a rectangular area that can be tiled. -/
structure TileableArea where
  width : ℕ
  height : ℕ

/-- Represents the count of each type of tile. -/
structure TileCount where
  two_by_two : ℕ
  one_by_four : ℕ

/-- Checks if a given area can be tiled with the given tile counts. -/
def can_tile (area : TileableArea) (tiles : TileCount) : Prop :=
  2 * tiles.two_by_two + 4 * tiles.one_by_four = area.width * area.height

/-- Theorem stating that if an area can be tiled, it becomes impossible
    to tile after replacing one 2x2 tile with a 1x4 tile. -/
theorem tiling_impossibility (area : TileableArea) (initial_tiles : TileCount) :
  can_tile area initial_tiles →
  ¬can_tile area { two_by_two := initial_tiles.two_by_two - 1,
                   one_by_four := initial_tiles.one_by_four + 1 } :=
by sorry

end NUMINAMATH_CALUDE_tiling_impossibility_l3518_351852


namespace NUMINAMATH_CALUDE_lcm_of_32_and_12_l3518_351845

theorem lcm_of_32_and_12 (n m : ℕ+) (h1 : n = 32) (h2 : m = 12) (h3 : Nat.gcd n m = 8) :
  Nat.lcm n m = 48 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_32_and_12_l3518_351845


namespace NUMINAMATH_CALUDE_devonshire_cows_cost_l3518_351892

/-- The number of hearts in a standard deck of 52 playing cards -/
def hearts_in_deck : ℕ := 13

/-- The number of cows in Devonshire -/
def cows_in_devonshire : ℕ := 2 * hearts_in_deck

/-- The price of each cow in dollars -/
def price_per_cow : ℕ := 200

/-- The total cost of all cows in Devonshire when sold -/
def total_cost : ℕ := cows_in_devonshire * price_per_cow

theorem devonshire_cows_cost : total_cost = 5200 := by
  sorry

end NUMINAMATH_CALUDE_devonshire_cows_cost_l3518_351892


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3518_351897

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 4 :=
by sorry

theorem equality_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) = 4 ↔ a^2 = 2 ∧ b = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3518_351897


namespace NUMINAMATH_CALUDE_curve_self_intersection_l3518_351810

/-- A curve in the xy-plane defined by x = t^2 - 4 and y = t^3 - 6t + 4 for all real t. -/
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 - 4, t^3 - 6*t + 4)

/-- The point where the curve crosses itself. -/
def self_intersection_point : ℝ × ℝ := (2, 4)

/-- Theorem stating that the curve crosses itself at the point (2, 4). -/
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = self_intersection_point :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l3518_351810


namespace NUMINAMATH_CALUDE_even_function_implies_m_eq_two_l3518_351820

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The given function f(x) -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem even_function_implies_m_eq_two :
  ∀ m : ℝ, IsEven (f m) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_eq_two_l3518_351820


namespace NUMINAMATH_CALUDE_planes_parallel_if_perp_to_same_line_l3518_351815

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- The main theorem
theorem planes_parallel_if_perp_to_same_line 
  (a : Line) (α β : Plane) 
  (h_diff : α ≠ β) 
  (h_perp_α : perp a α) 
  (h_perp_β : perp a β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perp_to_same_line_l3518_351815


namespace NUMINAMATH_CALUDE_four_numbers_perfect_square_product_l3518_351829

/-- A set of positive integers where all prime divisors are smaller than 30 -/
def SmallPrimeDivisorSet : Type := {s : Finset ℕ+ // ∀ n ∈ s, ∀ p : ℕ, Prime p → p ∣ n → p < 30}

theorem four_numbers_perfect_square_product (A : SmallPrimeDivisorSet) (h : A.val.card = 2016) :
  ∃ a b c d : ℕ+, a ∈ A.val ∧ b ∈ A.val ∧ c ∈ A.val ∧ d ∈ A.val ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ k : ℕ+, (a * b * c * d : ℕ) = k ^ 2 :=
sorry

end NUMINAMATH_CALUDE_four_numbers_perfect_square_product_l3518_351829


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l3518_351856

theorem inequality_not_always_true 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hc : c ≠ 0) : 
  ∃ c, ¬(a * c > b * c) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l3518_351856


namespace NUMINAMATH_CALUDE_system_inequalities_solution_set_l3518_351861

theorem system_inequalities_solution_set (m : ℝ) :
  (∀ x : ℝ, x > 4 ∧ x > m ↔ x > 4) ↔ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_set_l3518_351861


namespace NUMINAMATH_CALUDE_smallest_n_perfect_powers_l3518_351881

theorem smallest_n_perfect_powers : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℕ), 3 * n = x^4) ∧ 
  (∃ (y : ℕ), 2 * n = y^5) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 3 * m = x^4) → 
    (∃ (y : ℕ), 2 * m = y^5) → 
    n ≤ m) ∧
  n = 432 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_powers_l3518_351881


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l3518_351873

-- Define the types for lines and planes
variable (L : Type*) [LinearOrderedField L]
variable (P : Type*) [LinearOrderedField P]

-- Define the parallel and perpendicular relations
variable (parallel : L → L → Prop)
variable (perp : L → P → Prop)

-- State the theorem
theorem line_parallel_perpendicular
  (m n : L) (α : P)
  (h1 : m ≠ n)
  (h2 : parallel m n)
  (h3 : perp m α) :
  perp n α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l3518_351873


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l3518_351853

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l3518_351853


namespace NUMINAMATH_CALUDE_card_area_reduction_l3518_351862

/-- Given a 5x7 inch card, if reducing one side by 2 inches results in an area of 21 square inches,
    then reducing the other side by 2 inches instead will result in an area of 25 square inches. -/
theorem card_area_reduction (length width : ℝ) : 
  length = 5 ∧ width = 7 ∧ 
  ((length - 2) * width = 21 ∨ length * (width - 2) = 21) →
  (length * (width - 2) = 25 ∨ (length - 2) * width = 25) := by
sorry

end NUMINAMATH_CALUDE_card_area_reduction_l3518_351862


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3518_351877

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 + a 5 = Real.pi →
  a 4 * (a 2 + 2 * a 4 + a 6) = Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3518_351877


namespace NUMINAMATH_CALUDE_area_of_triangle_PAB_l3518_351880

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

noncomputable def line_equation (x y : ℝ) : Prop := y = x

noncomputable def tangent_line_equation (x y m : ℝ) : Prop := y = Real.sqrt 3 * x + m

theorem area_of_triangle_PAB 
  (O : ℝ × ℝ)  -- Center of the circle
  (A B P : ℝ × ℝ)  -- Points A, B, and P
  (m : ℝ)  -- Parameter for tangent line
  (h1 : ∀ (x y : ℝ), circle_equation x y ↔ (x - O.1)^2 + (y - O.2)^2 = 4)  -- Circle equation
  (h2 : line_equation A.1 A.2 ∧ line_equation B.1 B.2)  -- A and B on line y = x
  (h3 : circle_equation A.1 A.2 ∧ circle_equation B.1 B.2)  -- A and B on circle
  (h4 : tangent_line_equation P.1 P.2 m)  -- P on tangent line
  (h5 : circle_equation P.1 P.2)  -- P on circle
  (h6 : m > 0)  -- m is positive
  : Real.sqrt 6 + Real.sqrt 2 = 
    (1/2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * 
    (|P.2 - P.1 - (A.2 - A.1)| / Real.sqrt ((A.2 - A.1)^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_PAB_l3518_351880


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3518_351851

theorem arithmetic_sequence_sum (a₁ d n : ℝ) (S_n : ℕ → ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    y₁ = a₁ * x₁ ∧ 
    y₂ = a₁ * x₂ ∧ 
    (x₁ - 2)^2 + y₁^2 = 1 ∧ 
    (x₂ - 2)^2 + y₂^2 = 1 ∧
    (x₁ + y₁ + d) = -(x₂ + y₂ + d)) →
  (∀ k : ℕ, S_n k = k * a₁ + k * (k - 1) / 2 * d) →
  ∀ k : ℕ, S_n k = 2*k - k^2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3518_351851


namespace NUMINAMATH_CALUDE_M_subset_P_l3518_351848

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = -x^2 + 1}
def P : Set ℝ := Set.univ

-- State the theorem
theorem M_subset_P : M ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_M_subset_P_l3518_351848


namespace NUMINAMATH_CALUDE_shirt_pricing_solution_l3518_351879

/-- Represents the shirt pricing problem with given conditions --/
structure ShirtPricingProblem where
  cost_price : ℝ
  initial_sales : ℝ
  initial_profit_per_shirt : ℝ
  price_reduction_effect : ℝ
  target_daily_profit : ℝ

/-- Calculates the daily sales based on the price reduction --/
def daily_sales (p : ShirtPricingProblem) (selling_price : ℝ) : ℝ :=
  p.initial_sales + p.price_reduction_effect * (p.cost_price + p.initial_profit_per_shirt - selling_price)

/-- Calculates the daily profit based on the selling price --/
def daily_profit (p : ShirtPricingProblem) (selling_price : ℝ) : ℝ :=
  (selling_price - p.cost_price) * (daily_sales p selling_price)

/-- Theorem stating that the selling price should be either $105 or $120 --/
theorem shirt_pricing_solution (p : ShirtPricingProblem)
  (h1 : p.cost_price = 80)
  (h2 : p.initial_sales = 30)
  (h3 : p.initial_profit_per_shirt = 50)
  (h4 : p.price_reduction_effect = 2)
  (h5 : p.target_daily_profit = 2000) :
  ∃ (x : ℝ), (x = 105 ∨ x = 120) ∧ daily_profit p x = p.target_daily_profit :=
sorry

end NUMINAMATH_CALUDE_shirt_pricing_solution_l3518_351879


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l3518_351885

/-- Proves that 66 gallons of fuel A were added to a 204-gallon tank -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 204 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (fuel_a : ℝ), 
    fuel_a = 66 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l3518_351885


namespace NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l3518_351833

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of sides in the polygon -/
def sides : ℕ := 150

/-- Theorem: The number of diagonals in a polygon with 150 sides is 11025 -/
theorem diagonals_150_sided_polygon :
  num_diagonals sides = 11025 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l3518_351833


namespace NUMINAMATH_CALUDE_square_area_tripled_l3518_351807

theorem square_area_tripled (a : ℝ) (h : a > 0) :
  (a * Real.sqrt 3) ^ 2 = 3 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_tripled_l3518_351807


namespace NUMINAMATH_CALUDE_smallest_surface_area_l3518_351846

/-- Represents the dimensions of a cigarette box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  thickness : ℝ
  length_gt_width : length > width
  width_gt_thickness : width > thickness

/-- Calculates the surface area of a rectangular package -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + l * h)

/-- Represents different packaging methods for 10 boxes -/
inductive PackagingMethod
  | method1
  | method2
  | method3
  | method4

/-- Calculates the surface area for a given packaging method -/
def packaging_surface_area (method : PackagingMethod) (box : BoxDimensions) : ℝ :=
  match method with
  | .method1 => surface_area (10 * box.length) box.width box.thickness
  | .method2 => surface_area box.length (10 * box.width) box.thickness
  | .method3 => surface_area box.length box.width (10 * box.thickness)
  | .method4 => surface_area box.length (2 * box.width) (5 * box.thickness)

theorem smallest_surface_area (box : BoxDimensions) 
  (h1 : box.length = 88)
  (h2 : box.width = 58)
  (h3 : box.thickness = 22) :
  (∀ m : PackagingMethod, packaging_surface_area .method4 box ≤ packaging_surface_area m box) ∧ 
  packaging_surface_area .method4 box = 65296 := by
  sorry

#eval surface_area 88 (2 * 58) (5 * 22)

end NUMINAMATH_CALUDE_smallest_surface_area_l3518_351846


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3518_351858

theorem unknown_number_proof (x : ℝ) : 
  (14 + 32 + 53) / 3 = (21 + 47 + x) / 3 + 3 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3518_351858


namespace NUMINAMATH_CALUDE_min_committee_size_l3518_351889

/-- Represents a committee with the given properties -/
structure Committee where
  meetings : Nat
  attendees_per_meeting : Nat
  total_members : Nat
  attendance : Fin meetings → Finset (Fin total_members)
  ten_per_meeting : ∀ m, (attendance m).card = attendees_per_meeting
  at_most_once : ∀ i j m₁ m₂, i ≠ j → m₁ ≠ m₂ → 
    (i ∈ attendance m₁ ∧ i ∈ attendance m₂) → 
    (j ∉ attendance m₁ ∨ j ∉ attendance m₂)

/-- The main theorem stating the minimum number of members -/
theorem min_committee_size :
  ∀ c : Committee, c.meetings = 12 → c.attendees_per_meeting = 10 → c.total_members ≥ 58 :=
by sorry

end NUMINAMATH_CALUDE_min_committee_size_l3518_351889


namespace NUMINAMATH_CALUDE_bakery_payment_l3518_351842

theorem bakery_payment (bun_price croissant_price : ℕ) 
  (h1 : bun_price = 15) (h2 : croissant_price = 12) : 
  (¬ ∃ x y : ℕ, croissant_price * x + bun_price * y = 500) ∧
  (∃ x y : ℕ, croissant_price * x + bun_price * y = 600) := by
sorry

end NUMINAMATH_CALUDE_bakery_payment_l3518_351842
