import Mathlib

namespace daughter_age_is_40_l2522_252214

/-- Represents the family members' weights and ages -/
structure Family where
  mother_weight : ℝ
  daughter_weight : ℝ
  grandchild_weight : ℝ
  son_in_law_weight : ℝ
  mother_age : ℝ
  daughter_age : ℝ
  son_in_law_age : ℝ

/-- The family satisfies the given conditions -/
def satisfies_conditions (f : Family) : Prop :=
  f.mother_weight + f.daughter_weight + f.grandchild_weight + f.son_in_law_weight = 200 ∧
  f.daughter_weight + f.grandchild_weight = 60 ∧
  f.grandchild_weight = (1/5) * f.mother_weight ∧
  f.son_in_law_weight = 2 * f.daughter_weight ∧
  f.mother_age / f.daughter_age = 2 ∧
  f.daughter_age / f.son_in_law_age = 3/2 ∧
  f.mother_age = 80

/-- The theorem stating that if a family satisfies the given conditions, the daughter's age is 40 -/
theorem daughter_age_is_40 (f : Family) (h : satisfies_conditions f) : f.daughter_age = 40 := by
  sorry

end daughter_age_is_40_l2522_252214


namespace intersection_A_B_l2522_252210

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x^2 + 2*x < 0}

theorem intersection_A_B : A ∩ B = Set.Ioo (-1) 0 := by sorry

end intersection_A_B_l2522_252210


namespace inverse_variation_problem_l2522_252208

/-- Given that x² and y vary inversely, prove that when y = 20 for x = 3,
    then x = 3√10/50 when y = 5000 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) : 
  (∀ x y, x^2 * y = k) →  -- x² and y vary inversely
  (3^2 * 20 = k) →        -- y = 20 when x = 3
  (x^2 * 5000 = k) →      -- y = 5000 for the x we're looking for
  x = 3 * Real.sqrt 10 / 50 := by
sorry

end inverse_variation_problem_l2522_252208


namespace N_bounds_l2522_252253

/-- The number of divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The number of ordered pairs (x,y) satisfying the given conditions -/
def N (p : ℕ) : ℕ := (Finset.filter (fun pair : ℕ × ℕ =>
  1 ≤ pair.1 ∧ pair.1 ≤ p * (p - 1) ∧
  1 ≤ pair.2 ∧ pair.2 ≤ p * (p - 1) ∧
  (pair.1 ^ pair.2) % p = 1 ∧
  (pair.2 ^ pair.1) % p = 1
) (Finset.product (Finset.range (p * (p - 1) + 1)) (Finset.range (p * (p - 1) + 1)))).card

theorem N_bounds (p : ℕ) (h : Nat.Prime p) :
  (Nat.totient (p - 1) * d (p - 1))^2 ≤ N p ∧ N p ≤ ((p - 1) * d (p - 1))^2 := by
  sorry

end N_bounds_l2522_252253


namespace min_questions_to_determine_product_l2522_252242

theorem min_questions_to_determine_product (n : ℕ) (h : n > 3) :
  let min_questions_any_three := Int.ceil (n / 3 : ℚ)
  let min_questions_consecutive_three := if n % 3 = 0 then n / 3 else n
  true := by
  sorry

#check min_questions_to_determine_product

end min_questions_to_determine_product_l2522_252242


namespace rotate_A_180_l2522_252235

def rotate_180_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotate_A_180 :
  let A : ℝ × ℝ := (-3, 2)
  rotate_180_origin A = (3, -2) := by
  sorry

end rotate_A_180_l2522_252235


namespace line_plane_parallelism_l2522_252272

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (containedIn : Line → Plane → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem line_plane_parallelism 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_parallel_α : parallelLinePlane m α)
  (h_m_in_β : containedIn m β)
  (h_intersection : intersect α β = n) :
  parallelLine m n :=
sorry

end line_plane_parallelism_l2522_252272


namespace exponent_division_l2522_252223

theorem exponent_division (a : ℝ) (m n : ℕ) (h : a ≠ 0) : a ^ m / a ^ n = a ^ (m - n) := by
  sorry

end exponent_division_l2522_252223


namespace intersection_of_A_and_B_l2522_252268

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l2522_252268


namespace no_rational_solution_l2522_252293

theorem no_rational_solution (n : ℕ+) : ¬ ∃ (x y : ℚ), 0 < x ∧ 0 < y ∧ x + y + 1/x + 1/y = 3*n := by
  sorry

end no_rational_solution_l2522_252293


namespace unique_two_digit_number_l2522_252269

/-- A two-digit number satisfying specific conditions -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n % 10 = n / 10 + 2) ∧
  (n * (n / 10 + n % 10) = 144)

/-- Theorem stating that 24 is the only two-digit number satisfying the given conditions -/
theorem unique_two_digit_number : ∃! n : ℕ, TwoDigitNumber n ∧ n = 24 := by
  sorry

end unique_two_digit_number_l2522_252269


namespace chandler_savings_weeks_l2522_252265

def bike_cost : ℕ := 650
def birthday_money : ℕ := 50 + 35 + 15 + 20
def weekly_earnings : ℕ := 18

def weeks_to_save (cost birthday_money weekly_earnings : ℕ) : ℕ :=
  ((cost - birthday_money + weekly_earnings - 1) / weekly_earnings)

theorem chandler_savings_weeks :
  weeks_to_save bike_cost birthday_money weekly_earnings = 30 := by
  sorry

end chandler_savings_weeks_l2522_252265


namespace course_size_l2522_252276

theorem course_size (total : ℕ) 
  (grade_A : ℕ → Prop) (grade_B : ℕ → Prop) (grade_C : ℕ → Prop) (grade_D : ℕ → Prop)
  (h1 : ∀ n, grade_A n ↔ n = total / 5)
  (h2 : ∀ n, grade_B n ↔ n = total / 4)
  (h3 : ∀ n, grade_C n ↔ n = total / 2)
  (h4 : ∀ n, grade_D n ↔ n = 25)
  (h5 : ∀ n, n ≤ total → (grade_A n ∨ grade_B n ∨ grade_C n ∨ grade_D n))
  (h6 : ∀ n, (grade_A n → ¬grade_B n ∧ ¬grade_C n ∧ ¬grade_D n) ∧
             (grade_B n → ¬grade_A n ∧ ¬grade_C n ∧ ¬grade_D n) ∧
             (grade_C n → ¬grade_A n ∧ ¬grade_B n ∧ ¬grade_D n) ∧
             (grade_D n → ¬grade_A n ∧ ¬grade_B n ∧ ¬grade_C n)) :
  total = 500 := by
sorry

end course_size_l2522_252276


namespace contrapositive_even_sum_l2522_252284

theorem contrapositive_even_sum (a b : ℤ) : 
  (¬(Even (a + b)) → ¬(Even a ∧ Even b)) ↔ 
  (∀ (a b : ℤ), (Even a ∧ Even b) → Even (a + b))ᶜ :=
by sorry

end contrapositive_even_sum_l2522_252284


namespace min_cows_for_safe_ducks_l2522_252215

/-- Represents the arrangement of animals in Farmer Bill's circle -/
structure AnimalArrangement where
  total : Nat
  ducks : Nat
  cows : Nat
  rabbits : Nat

/-- Checks if the arrangement satisfies the safety condition for ducks -/
def isSafeArrangement (arr : AnimalArrangement) : Prop :=
  arr.ducks ≤ (arr.rabbits - 1) + 2 * arr.cows

/-- The main theorem stating the minimum number of cows required -/
theorem min_cows_for_safe_ducks (arr : AnimalArrangement) 
  (h1 : arr.total = 1000)
  (h2 : arr.ducks = 600)
  (h3 : arr.total = arr.ducks + arr.cows + arr.rabbits)
  (h4 : isSafeArrangement arr) :
  arr.cows ≥ 201 ∧ ∃ (safeArr : AnimalArrangement), 
    safeArr.total = 1000 ∧ 
    safeArr.ducks = 600 ∧ 
    safeArr.cows = 201 ∧
    isSafeArrangement safeArr :=
sorry

end min_cows_for_safe_ducks_l2522_252215


namespace regression_unit_increase_survey_regression_unit_increase_l2522_252256

/-- Linear regression equation parameters -/
structure RegressionParams where
  slope : ℝ
  intercept : ℝ

/-- Calculates the predicted y value for a given x -/
def predict (params : RegressionParams) (x : ℝ) : ℝ :=
  params.slope * x + params.intercept

/-- Theorem: The difference in predicted y when x increases by 1 is equal to the slope -/
theorem regression_unit_increase (params : RegressionParams) :
  ∀ x : ℝ, predict params (x + 1) - predict params x = params.slope := by
  sorry

/-- The specific regression equation from the problem -/
def survey_regression : RegressionParams :=
  { slope := 0.254, intercept := 0.321 }

/-- Theorem: For the given survey regression, the difference in predicted y
    when x increases by 1 is equal to 0.254 -/
theorem survey_regression_unit_increase :
  ∀ x : ℝ, predict survey_regression (x + 1) - predict survey_regression x = 0.254 := by
  sorry

end regression_unit_increase_survey_regression_unit_increase_l2522_252256


namespace complex_equation_solution_l2522_252288

theorem complex_equation_solution (m : ℝ) :
  (2 : ℂ) / (1 - Complex.I) = 1 + m * Complex.I → m = 1 := by
  sorry

end complex_equation_solution_l2522_252288


namespace special_isosceles_triangle_sides_l2522_252230

/-- An isosceles triangle with specific incenter properties -/
structure SpecialIsoscelesTriangle where
  -- The length of the two equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- The distance from the vertex to the incenter along the altitude
  vertexToIncenter : ℝ
  -- The distance from the incenter to the base along the altitude
  incenterToBase : ℝ
  -- Ensure the triangle is isosceles
  isIsosceles : side > 0
  -- Ensure the incenter divides the altitude as specified
  incenterDivision : vertexToIncenter = 5 ∧ incenterToBase = 3

/-- The theorem stating the side lengths of the special isosceles triangle -/
theorem special_isosceles_triangle_sides 
  (t : SpecialIsoscelesTriangle) : t.side = 10 ∧ t.base = 12 := by
  sorry

#check special_isosceles_triangle_sides

end special_isosceles_triangle_sides_l2522_252230


namespace functional_equation_solution_l2522_252209

theorem functional_equation_solution (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (2*m + 2*n) = f m * f n) : 
  ∀ x : ℕ, f x = 1 := by
sorry

end functional_equation_solution_l2522_252209


namespace abc_inequality_l2522_252287

theorem abc_inequality (a b c : ℝ) (h : a^2*b*c + a*b^2*c + a*b*c^2 = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by sorry

end abc_inequality_l2522_252287


namespace decoration_time_proof_l2522_252233

/-- The time needed for Mia and Billy to decorate Easter eggs -/
def decoration_time (mia_rate : ℕ) (billy_rate : ℕ) (total_eggs : ℕ) : ℚ :=
  total_eggs / (mia_rate + billy_rate)

/-- Theorem stating that Mia and Billy will take 5 hours to decorate 170 eggs -/
theorem decoration_time_proof :
  decoration_time 24 10 170 = 5 := by
  sorry

end decoration_time_proof_l2522_252233


namespace sum_of_naturals_equals_406_l2522_252219

theorem sum_of_naturals_equals_406 (n : ℕ) : (n * (n + 1)) / 2 = 406 → n = 28 := by sorry

end sum_of_naturals_equals_406_l2522_252219


namespace regular_soda_count_l2522_252280

/-- The number of bottles of regular soda in a grocery store -/
def regular_soda : ℕ := sorry

/-- The number of bottles of diet soda in a grocery store -/
def diet_soda : ℕ := 26

/-- The number of bottles of lite soda in a grocery store -/
def lite_soda : ℕ := 27

/-- The total number of soda bottles in a grocery store -/
def total_bottles : ℕ := 110

/-- Theorem stating that the number of bottles of regular soda is 57 -/
theorem regular_soda_count : regular_soda = 57 := by
  sorry

end regular_soda_count_l2522_252280


namespace survey_respondents_l2522_252228

theorem survey_respondents (prefer_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  prefer_x = 150 → ratio_x = 5 → ratio_y = 1 →
  ∃ (total : ℕ), total = prefer_x + (prefer_x / ratio_x * ratio_y) ∧ total = 180 :=
by
  sorry

end survey_respondents_l2522_252228


namespace max_y_rectangular_prism_l2522_252294

/-- The maximum value of y for a rectangular prism with volume 360 and integer dimensions x, y, z satisfying 1 < z < y < x -/
theorem max_y_rectangular_prism : 
  ∀ x y z : ℕ, 
  x * y * z = 360 → 
  1 < z → z < y → y < x → 
  y ≤ 9 :=
by sorry

end max_y_rectangular_prism_l2522_252294


namespace sum_minimized_at_24_l2522_252259

/-- The sum of the first n terms of an arithmetic sequence with general term a_n = 2n - 49 -/
def S (n : ℕ) : ℝ := n^2 - 48*n

/-- The value of n that minimizes S_n -/
def n_min : ℕ := 24

theorem sum_minimized_at_24 :
  ∀ n : ℕ, n ≠ 0 → S n ≥ S n_min := by sorry

end sum_minimized_at_24_l2522_252259


namespace house_development_problem_l2522_252238

theorem house_development_problem (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) 
  (h1 : total = 85)
  (h2 : garage = 50)
  (h3 : pool = 40)
  (h4 : both = 35) :
  total - (garage + pool - both) = 30 :=
by sorry

end house_development_problem_l2522_252238


namespace soccer_survey_l2522_252241

/-- Represents the fraction of students who enjoy playing soccer -/
def enjoy_soccer : ℚ := 1/2

/-- Represents the fraction of students who honestly say they enjoy soccer among those who enjoy it -/
def honest_enjoy : ℚ := 7/10

/-- Represents the fraction of students who honestly say they do not enjoy soccer among those who do not enjoy it -/
def honest_not_enjoy : ℚ := 8/10

/-- The fraction of students who claim they do not enjoy playing soccer but actually enjoy it -/
def fraction_false_claim : ℚ := 3/11

theorem soccer_survey :
  (enjoy_soccer * (1 - honest_enjoy)) / 
  ((enjoy_soccer * (1 - honest_enjoy)) + ((1 - enjoy_soccer) * honest_not_enjoy)) = fraction_false_claim := by
  sorry

end soccer_survey_l2522_252241


namespace negate_negate_eq_self_l2522_252205

theorem negate_negate_eq_self (n : ℤ) : -(-n) = n := by sorry

end negate_negate_eq_self_l2522_252205


namespace solution_exists_for_quadratic_cubic_congruence_l2522_252292

theorem solution_exists_for_quadratic_cubic_congruence (p : ℕ) (hp : Prime p) (a : ℤ) :
  ∃ (x y : ℤ), (x^2 + y^3) % p = a % p := by
  sorry

end solution_exists_for_quadratic_cubic_congruence_l2522_252292


namespace rhea_children_eggs_l2522_252297

/-- The number of eggs eaten by Rhea's son and daughter every morning -/
def eggs_eaten_by_children (
  trays_per_week : ℕ)  -- Number of trays bought per week
  (eggs_per_tray : ℕ)  -- Number of eggs per tray
  (eggs_eaten_by_parents : ℕ)  -- Number of eggs eaten by parents per night
  (eggs_not_eaten : ℕ)  -- Number of eggs not eaten per week
  : ℕ :=
  trays_per_week * eggs_per_tray - 7 * eggs_eaten_by_parents - eggs_not_eaten

/-- Theorem stating that Rhea's son and daughter eat 14 eggs every morning -/
theorem rhea_children_eggs : 
  eggs_eaten_by_children 2 24 4 6 = 14 := by
  sorry

end rhea_children_eggs_l2522_252297


namespace quadratic_a_value_l2522_252229

/-- A quadratic function with vertex (h, k) passing through point (x₀, y₀) -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ
  vertex_form : ∀ x, a * (x - h)^2 + k = a * x^2 + (a * h * (-2)) * x + (a * h^2 + k)
  passes_through : a * (x₀ - h)^2 + k = y₀

/-- The theorem stating that for a quadratic function with vertex (3, 5) passing through (0, -20), a = -25/9 -/
theorem quadratic_a_value (f : QuadraticFunction) 
    (h_vertex : f.h = 3 ∧ f.k = 5) 
    (h_point : f.x₀ = 0 ∧ f.y₀ = -20) : 
    f.a = -25/9 := by
  sorry

end quadratic_a_value_l2522_252229


namespace hilltop_volleyball_club_members_l2522_252296

/-- Represents the Hilltop Volleyball Club inventory problem -/
theorem hilltop_volleyball_club_members :
  let sock_cost : ℕ := 6
  let tshirt_cost : ℕ := sock_cost + 7
  let items_per_member : ℕ := 3
  let cost_per_member : ℕ := items_per_member * (sock_cost + tshirt_cost)
  let total_cost : ℕ := 4026
  total_cost / cost_per_member = 71 :=
by sorry

end hilltop_volleyball_club_members_l2522_252296


namespace system_solution_l2522_252218

theorem system_solution (x y b : ℝ) : 
  (4 * x + 2 * y = b) → 
  (3 * x + 4 * y = 3 * b) → 
  (x = 3) → 
  (b = -1) := by
sorry

end system_solution_l2522_252218


namespace cubic_discriminant_l2522_252221

theorem cubic_discriminant (p q : ℝ) (x₁ x₂ x₃ : ℝ) : 
  x₁^3 + p*x₁ + q = 0 → 
  x₂^3 + p*x₂ + q = 0 → 
  x₃^3 + p*x₃ + q = 0 → 
  (x₁ - x₂)^2 * (x₂ - x₃)^2 * (x₃ - x₁)^2 = -4*p^3 - 27*q^2 := by
sorry

end cubic_discriminant_l2522_252221


namespace first_robber_guarantee_l2522_252237

/-- Represents the coin division game between two robbers --/
structure CoinGame where
  totalCoins : Nat
  maxBags : Nat

/-- Calculates the guaranteed minimum coins for the first robber --/
def guaranteedCoins (game : CoinGame) : Nat :=
  game.totalCoins - (game.maxBags - 1) * (game.totalCoins / (2 * game.maxBags - 1))

/-- Theorem: In a game with 300 coins and 11 max bags, the first robber can guarantee at least 146 coins --/
theorem first_robber_guarantee (game : CoinGame) 
  (h1 : game.totalCoins = 300) 
  (h2 : game.maxBags = 11) : 
  guaranteedCoins game ≥ 146 := by
  sorry

#eval guaranteedCoins { totalCoins := 300, maxBags := 11 }

end first_robber_guarantee_l2522_252237


namespace drew_marbles_difference_l2522_252290

theorem drew_marbles_difference (drew_initial : ℕ) (marcus_initial : ℕ) (john_initial : ℕ) 
  (h1 : marcus_initial = 45)
  (h2 : john_initial = 70)
  (h3 : ∃ x : ℕ, drew_initial / 4 + marcus_initial = x ∧ drew_initial / 8 + john_initial = x) :
  drew_initial - marcus_initial = 155 :=
by sorry

end drew_marbles_difference_l2522_252290


namespace money_redistribution_theorem_l2522_252283

/-- Represents the money redistribution problem with Ben, Tom, and Max -/
theorem money_redistribution_theorem 
  (ben_start : ℕ) 
  (max_start_end : ℕ) 
  (ben_end : ℕ) 
  (tom_end : ℕ) 
  (max_end : ℕ) 
  (h1 : ben_start = 48)
  (h2 : max_start_end = 48)
  (h3 : max_end = max_start_end)
  (h4 : ben_end = ben_start)
  : ben_end + tom_end + max_end = 144 := by
  sorry

#check money_redistribution_theorem

end money_redistribution_theorem_l2522_252283


namespace tangent_line_and_bounds_l2522_252247

/-- The function f(x) = (ax+b)e^(-2x) -/
noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) * Real.exp (-2 * x)

/-- The function g(x) = f(x) + x * ln(x) -/
noncomputable def g (a b x : ℝ) : ℝ := f a b x + x * Real.log x

theorem tangent_line_and_bounds
  (a b : ℝ)
  (h1 : f a b 0 = 1)  -- f(0) = 1 from the tangent line equation
  (h2 : (deriv (f a b)) 0 = -1)  -- f'(0) = -1 from the tangent line equation
  : a = 1 ∧ b = 1 ∧ ∀ x, 0 < x → x < 1 → 2 * Real.exp (-2) - Real.exp (-1) < g a b x ∧ g a b x < 1 :=
sorry

end tangent_line_and_bounds_l2522_252247


namespace M_equals_N_l2522_252285

/-- Definition of set M -/
def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

/-- Definition of set N -/
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

/-- Theorem stating that M equals N -/
theorem M_equals_N : M = N := by
  sorry

end M_equals_N_l2522_252285


namespace inequality_solution_set_l2522_252273

theorem inequality_solution_set (x : ℝ) : 
  (5 - x^2 > 4*x) ↔ (x > -5 ∧ x < 1) :=
sorry

end inequality_solution_set_l2522_252273


namespace checkerboard_domino_cover_l2522_252224

/-- A checkerboard is a rectangular grid of squares. -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- A domino covers exactly two squares. -/
def domino_cover := 2

/-- The total number of squares on a checkerboard. -/
def total_squares (board : Checkerboard) : ℕ :=
  board.rows * board.cols

/-- A checkerboard can be covered by dominoes if its total number of squares is even. -/
theorem checkerboard_domino_cover (board : Checkerboard) :
  ∃ (n : ℕ), total_squares board = n * domino_cover ↔ Even (total_squares board) :=
sorry

end checkerboard_domino_cover_l2522_252224


namespace inverse_square_relation_l2522_252264

/-- Given that x varies inversely as the square of y, prove that x = 1 when y = 2,
    given that x = 0.1111111111111111 when y = 6. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) (h1 : x = k / y^2) 
    (h2 : 0.1111111111111111 = k / 6^2) : 
  1 = k / 2^2 := by
  sorry

end inverse_square_relation_l2522_252264


namespace equation_solution_l2522_252295

theorem equation_solution : ∃ x : ℚ, (x - 7) / 2 - (1 + x) / 3 = 1 ∧ x = 29 := by
  sorry

end equation_solution_l2522_252295


namespace log_decreasing_implies_a_range_l2522_252249

/-- A function f is decreasing on an interval [a, b] if for any x, y in [a, b] with x < y, f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x > f y

/-- The logarithm function with base a -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_decreasing_implies_a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  DecreasingOn (fun x => log a (5 - a * x)) 1 3 → 1 < a ∧ a < 5/3 := by
  sorry

end log_decreasing_implies_a_range_l2522_252249


namespace valid_integers_count_l2522_252291

/-- The number of digits in the integers we're counting -/
def num_digits : ℕ := 8

/-- The number of choices for the first digit (2-9) -/
def first_digit_choices : ℕ := 8

/-- The number of choices for each subsequent digit (0-9) -/
def other_digit_choices : ℕ := 10

/-- The number of different 8-digit positive integers where the first digit cannot be 0 or 1 -/
def count_valid_integers : ℕ := first_digit_choices * (other_digit_choices ^ (num_digits - 1))

theorem valid_integers_count :
  count_valid_integers = 80000000 := by sorry

end valid_integers_count_l2522_252291


namespace cone_generatrix_length_l2522_252212

/-- The length of the generatrix of a cone with base radius √2 and lateral surface forming a semicircle when unfolded is 2√2. -/
theorem cone_generatrix_length :
  ∀ (base_radius : ℝ) (generatrix_length : ℝ),
  base_radius = Real.sqrt 2 →
  2 * Real.pi * base_radius = Real.pi * generatrix_length →
  generatrix_length = 2 * Real.sqrt 2 := by
sorry

end cone_generatrix_length_l2522_252212


namespace lending_duration_l2522_252258

/-- Proves that the number of years the first part is lent is 8, given the problem conditions -/
theorem lending_duration (total_sum : ℚ) (second_part : ℚ) 
  (first_rate : ℚ) (second_rate : ℚ) (second_duration : ℚ) :
  total_sum = 2678 →
  second_part = 1648 →
  first_rate = 3/100 →
  second_rate = 5/100 →
  second_duration = 3 →
  ∃ (first_duration : ℚ),
    (total_sum - second_part) * first_rate * first_duration = 
    second_part * second_rate * second_duration ∧
    first_duration = 8 := by
  sorry

end lending_duration_l2522_252258


namespace sum_of_distances_constant_l2522_252200

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_pos : side > 0

/-- A point inside an equilateral triangle -/
structure InternalPoint (t : EquilateralTriangle) where
  x : ℝ
  y : ℝ
  inside : x > 0 ∧ y > 0 ∧ x + y < t.side

/-- The sum of perpendicular distances from an internal point to the three sides of an equilateral triangle -/
def sumOfDistances (t : EquilateralTriangle) (p : InternalPoint t) : ℝ :=
  p.x + p.y + (t.side - p.x - p.y)

/-- Theorem: The sum of perpendicular distances from any internal point to the three sides of an equilateral triangle is constant and equal to (√3/2) * side length -/
theorem sum_of_distances_constant (t : EquilateralTriangle) (p : InternalPoint t) :
  sumOfDistances t p = (Real.sqrt 3 / 2) * t.side := by
  sorry


end sum_of_distances_constant_l2522_252200


namespace line_increase_l2522_252248

/-- Given a line where an increase of 5 units in x corresponds to an increase of 11 units in y,
    prove that an increase of 15 units in x corresponds to an increase of 33 units in y. -/
theorem line_increase (m : ℝ) (h : m = 11 / 5) : m * 15 = 33 := by
  sorry

end line_increase_l2522_252248


namespace temperature_difference_l2522_252263

theorem temperature_difference (low high : ℤ) (h1 : low = -2) (h2 : high = 5) :
  high - low = 7 := by
  sorry

end temperature_difference_l2522_252263


namespace death_rate_is_eleven_l2522_252203

/-- Given a birth rate, net growth rate, and initial population, calculates the death rate. -/
def calculate_death_rate (birth_rate : ℝ) (net_growth_rate : ℝ) (initial_population : ℝ) : ℝ :=
  birth_rate - net_growth_rate * initial_population

/-- Proves that given the specified conditions, the death rate is 11. -/
theorem death_rate_is_eleven :
  let birth_rate : ℝ := 32
  let net_growth_rate : ℝ := 0.021
  let initial_population : ℝ := 1000
  calculate_death_rate birth_rate net_growth_rate initial_population = 11 := by
  sorry

#eval calculate_death_rate 32 0.021 1000

end death_rate_is_eleven_l2522_252203


namespace factor_expression_l2522_252257

theorem factor_expression (x : ℝ) : 63 * x^2 + 28 * x = 7 * x * (9 * x + 4) := by
  sorry

end factor_expression_l2522_252257


namespace rupert_ronald_jumps_l2522_252255

theorem rupert_ronald_jumps 
  (ronald_jumps : ℕ) 
  (total_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : total_jumps = 243)
  (h3 : ronald_jumps < total_jumps - ronald_jumps) :
  total_jumps - ronald_jumps - ronald_jumps = 86 := by
  sorry

end rupert_ronald_jumps_l2522_252255


namespace equation_one_solutions_equation_two_solutions_l2522_252239

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  3 * x^2 - 11 * x + 9 = 0 ↔ x = (11 + Real.sqrt 13) / 6 ∨ x = (11 - Real.sqrt 13) / 6 :=
sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  5 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 / 2 :=
sorry

end equation_one_solutions_equation_two_solutions_l2522_252239


namespace geometric_sequence_first_term_l2522_252286

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_a3 : a 3 = 2)
  (h_a4 : a 4 = 4) :
  a 1 = 1/2 := by
sorry

end geometric_sequence_first_term_l2522_252286


namespace functional_equation_solution_l2522_252213

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = ax² for some a ∈ ℚ -/
theorem functional_equation_solution :
  ∀ f : ℚ → ℚ, SatisfiesFunctionalEquation f →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 :=
by
  sorry


end functional_equation_solution_l2522_252213


namespace john_weight_loss_days_l2522_252262

/-- Calculates the number of days needed to lose a certain amount of weight given daily calorie intake, daily calorie burn, calories needed to lose one pound, and desired weight loss. -/
def days_to_lose_weight (calories_eaten : ℕ) (calories_burned : ℕ) (calories_per_pound : ℕ) (pounds_to_lose : ℕ) : ℕ :=
  let net_calories_burned := calories_burned - calories_eaten
  let total_calories_to_burn := calories_per_pound * pounds_to_lose
  total_calories_to_burn / net_calories_burned

/-- Theorem stating that it takes 80 days for John to lose 10 pounds given the specified conditions. -/
theorem john_weight_loss_days : 
  days_to_lose_weight 1800 2300 4000 10 = 80 := by
  sorry

end john_weight_loss_days_l2522_252262


namespace square_root_approximation_l2522_252207

theorem square_root_approximation : ∃ ε > 0, ε < 0.0001 ∧ 
  |Real.sqrt ((16^10 + 32^10) / (16^6 + 32^11)) - 0.1768| < ε :=
by sorry

end square_root_approximation_l2522_252207


namespace sales_solution_l2522_252202

def sales_problem (month1 month2 month4 month5 month6 average : ℕ) : Prop :=
  let total := average * 6
  let known_sum := month1 + month2 + month4 + month5 + month6
  let month3 := total - known_sum
  month3 = 7855

theorem sales_solution :
  sales_problem 7435 7920 8230 7560 6000 7500 := by
  sorry

end sales_solution_l2522_252202


namespace prob_at_least_one_heart_or_king_l2522_252232

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either a heart or a king -/
def target_cards : ℕ := 16

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The probability of drawing at least one heart or king in three draws with replacement -/
theorem prob_at_least_one_heart_or_king :
  1 - (((deck_size - target_cards : ℚ) / deck_size) ^ num_draws) = 1468 / 2197 := by
  sorry

end prob_at_least_one_heart_or_king_l2522_252232


namespace octadecagon_relation_l2522_252277

/-- Given a regular octadecagon inscribed in a circle with side length a and radius r,
    prove that a³ + r³ = 3r²a. -/
theorem octadecagon_relation (a r : ℝ) (h : a > 0) (k : r > 0) :
  a^3 + r^3 = 3 * r^2 * a := by
  sorry

end octadecagon_relation_l2522_252277


namespace fourth_roll_five_probability_l2522_252222

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 6
def biased_die_five_prob : ℚ := 1 / 2
def biased_die_other_prob : ℚ := 1 / 10

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the number of rolls
def num_rolls : ℕ := 4

-- Define the probability of choosing each die
def choose_prob : ℚ := 1 / 2

-- Theorem statement
theorem fourth_roll_five_probability :
  let p_fair := fair_die_prob ^ 3 * choose_prob
  let p_biased := biased_die_five_prob ^ 3 * choose_prob
  let p_fair_given_three_fives := p_fair / (p_fair + p_biased)
  let p_biased_given_three_fives := p_biased / (p_fair + p_biased)
  p_fair_given_three_fives * fair_die_prob + p_biased_given_three_fives * biased_die_five_prob = 41 / 84 :=
by sorry

end fourth_roll_five_probability_l2522_252222


namespace inequality_proof_l2522_252217

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end inequality_proof_l2522_252217


namespace x_squared_minus_x_plus_one_equals_seven_l2522_252252

theorem x_squared_minus_x_plus_one_equals_seven (x : ℝ) 
  (h : (x^2 - x)^2 - 4*(x^2 - x) - 12 = 0) : 
  x^2 - x + 1 = 7 := by
sorry

end x_squared_minus_x_plus_one_equals_seven_l2522_252252


namespace base_conversion_3012_to_octal_l2522_252243

theorem base_conversion_3012_to_octal :
  (3012 : ℕ) = 5 * (8 : ℕ)^3 + 7 * (8 : ℕ)^2 + 0 * (8 : ℕ)^1 + 4 * (8 : ℕ)^0 :=
by sorry

end base_conversion_3012_to_octal_l2522_252243


namespace three_digit_divisible_by_11_l2522_252225

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Predicate to check if the middle digit is the sum of outer digits -/
def middleDigitIsSumOfOuter (n : ThreeDigitNumber) : Prop :=
  n.tens = n.hundreds + n.units

theorem three_digit_divisible_by_11 (n : ThreeDigitNumber) 
  (h : middleDigitIsSumOfOuter n) : 
  (n.toNat % 11 = 0) := by
  sorry

#check three_digit_divisible_by_11

end three_digit_divisible_by_11_l2522_252225


namespace product_xyz_equals_42_l2522_252236

theorem product_xyz_equals_42 (x y z : ℝ) 
  (h1 : y = x + 1) 
  (h2 : x + y = 2 * z) 
  (h3 : x = 3) : 
  x * y * z = 42 := by
  sorry

end product_xyz_equals_42_l2522_252236


namespace chef_initial_apples_chef_had_46_apples_l2522_252245

/-- The number of apples a chef had initially, given the number of apples left after making pies
    and the difference between the initial and final number of apples. -/
theorem chef_initial_apples (apples_left : ℕ) (difference : ℕ) : ℕ :=
  apples_left + difference

/-- Proof that the chef initially had 46 apples -/
theorem chef_had_46_apples : chef_initial_apples 14 32 = 46 := by
  sorry

end chef_initial_apples_chef_had_46_apples_l2522_252245


namespace fraction_of_number_minus_constant_l2522_252260

theorem fraction_of_number_minus_constant (a b c d : ℕ) (h : a ≤ b) : 
  (a : ℚ) / b * c - d = 39 → a = 7 ∧ b = 8 ∧ c = 48 ∧ d = 3 := by
sorry

end fraction_of_number_minus_constant_l2522_252260


namespace new_person_weight_l2522_252281

theorem new_person_weight (n : ℕ) (old_weight average_increase : ℝ) :
  n = 10 →
  old_weight = 65 →
  average_increase = 3.2 →
  ∃ (new_weight : ℝ),
    new_weight = old_weight + n * average_increase ∧
    new_weight = 97 :=
by sorry

end new_person_weight_l2522_252281


namespace inequality_for_positive_reals_l2522_252289

theorem inequality_for_positive_reals : ∀ x : ℝ, x > 0 → x + 4 / x ≥ 4 := by sorry

end inequality_for_positive_reals_l2522_252289


namespace homework_problem_count_l2522_252226

/-- The number of sub tasks per homework problem -/
def sub_tasks_per_problem : ℕ := 5

/-- The total number of sub tasks to solve -/
def total_sub_tasks : ℕ := 200

/-- The total number of homework problems -/
def total_problems : ℕ := total_sub_tasks / sub_tasks_per_problem

theorem homework_problem_count :
  total_problems = 40 :=
by sorry

end homework_problem_count_l2522_252226


namespace movie_ticket_cost_l2522_252254

/-- The cost of a movie ticket in dollars -/
def ticket_cost : ℝ := 5

/-- The cost of popcorn in dollars -/
def popcorn_cost : ℝ := 0.8 * ticket_cost

/-- The cost of soda in dollars -/
def soda_cost : ℝ := 0.5 * popcorn_cost

/-- Theorem stating that the given conditions result in a ticket cost of $5 -/
theorem movie_ticket_cost : 
  4 * ticket_cost + 2 * popcorn_cost + 4 * soda_cost = 36 := by
  sorry


end movie_ticket_cost_l2522_252254


namespace min_value_of_z_l2522_252261

theorem min_value_of_z (x y z : ℝ) (h1 : 2 * x + y = 1) (h2 : z = 4^x + 2^y) : 
  z ≥ 2 * Real.sqrt 2 ∧ ∃ (x₀ y₀ : ℝ), 2 * x₀ + y₀ = 1 ∧ 4^x₀ + 2^y₀ = 2 * Real.sqrt 2 :=
sorry

end min_value_of_z_l2522_252261


namespace arrangements_eq_18_l2522_252275

/-- Represents the number of people in the lineup --/
def n : ℕ := 5

/-- Represents the possible positions for Person A --/
def A_positions : Set ℕ := {1, 2}

/-- Represents the possible positions for Person B --/
def B_positions : Set ℕ := {2, 3}

/-- The number of ways to arrange n people with the given constraints --/
def num_arrangements (n : ℕ) (A_pos : Set ℕ) (B_pos : Set ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of arrangements is 18 --/
theorem arrangements_eq_18 :
  num_arrangements n A_positions B_positions = 18 := by
  sorry

end arrangements_eq_18_l2522_252275


namespace rubble_short_by_8_75_l2522_252206

def initial_amount : ℚ := 45
def notebook_cost : ℚ := 4
def pen_cost : ℚ := 1.5
def eraser_cost : ℚ := 2.25
def pencil_case_cost : ℚ := 7.5
def notebook_count : ℕ := 5
def pen_count : ℕ := 8
def eraser_count : ℕ := 3
def pencil_case_count : ℕ := 2

def total_cost : ℚ :=
  notebook_cost * notebook_count +
  pen_cost * pen_count +
  eraser_cost * eraser_count +
  pencil_case_cost * pencil_case_count

theorem rubble_short_by_8_75 :
  initial_amount - total_cost = -8.75 := by
  sorry

end rubble_short_by_8_75_l2522_252206


namespace set_operations_and_intersection_l2522_252266

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x : ℝ | -a < x ∧ x ≤ a + 3}

-- Theorem statement
theorem set_operations_and_intersection (a : ℝ) : 
  (A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 8}) ∧ 
  ((Aᶜ : Set ℝ) ∩ B = {x : ℝ | 5 ≤ x ∧ x < 8}) ∧ 
  (C a ∩ A = C a ↔ a ≤ -1) := by
  sorry

end set_operations_and_intersection_l2522_252266


namespace max_value_trigonometric_expression_l2522_252267

theorem max_value_trigonometric_expression (θ : Real) 
  (h : 0 < θ ∧ θ < π / 2) : 
  ∃ (max : Real), max = 4 * Real.sqrt 2 ∧ 
    ∀ φ, 0 < φ ∧ φ < π / 2 → 
      3 * Real.sin φ + 2 * Real.cos φ + 1 / Real.cos φ ≤ max :=
by sorry

end max_value_trigonometric_expression_l2522_252267


namespace middle_position_theorem_l2522_252270

/-- Represents the color of a stone -/
inductive Color
  | Black
  | White

/-- Represents the state of the stone line -/
def StoneLine := Fin 2021 → Color

/-- Checks if a position is valid for the operation -/
def validPosition (n : Fin 2021) : Prop :=
  1 < n.val ∧ n.val < 2021

/-- Represents a single operation on the stone line -/
def operation (line : StoneLine) (n : Fin 2021) : StoneLine :=
  fun i => if i = n - 1 ∨ i = n + 1 then
    match line i with
    | Color.Black => Color.White
    | Color.White => Color.Black
    else line i

/-- Checks if all stones in the line are black -/
def allBlack (line : StoneLine) : Prop :=
  ∀ i, line i = Color.Black

/-- Initial configuration with one black stone at position n -/
def initialConfig (n : Fin 2021) : StoneLine :=
  fun i => if i = n then Color.Black else Color.White

/-- Represents the ability to make all stones black through operations -/
def canMakeAllBlack (line : StoneLine) : Prop :=
  ∃ (seq : List (Fin 2021)), 
    (∀ n ∈ seq, validPosition n) ∧
    allBlack (seq.foldl operation line)

/-- The main theorem to be proved -/
theorem middle_position_theorem :
  ∀ n : Fin 2021, canMakeAllBlack (initialConfig n) ↔ n = ⟨1011, sorry⟩ :=
sorry

end middle_position_theorem_l2522_252270


namespace min_distance_squared_l2522_252246

theorem min_distance_squared (a b c d : ℝ) 
  (h1 : Real.log a - Real.log 3 = Real.log c) 
  (h2 : b * d = -3) : 
  ∃ (min_val : ℝ), min_val = 18/5 ∧ 
    ∀ (x y : ℝ), (x - b)^2 + (y - c)^2 ≥ min_val :=
sorry

end min_distance_squared_l2522_252246


namespace simplify_expression_l2522_252299

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (343 : ℝ) ^ (1/3) = 28 := by
  sorry

end simplify_expression_l2522_252299


namespace divisibility_by_101_l2522_252271

theorem divisibility_by_101 (n : ℕ+) :
  (∃ k : ℕ+, n = k * 101 - 1) ↔
  (101 ∣ n^3 + 1) ∧ (101 ∣ n^2 - 1) :=
sorry

end divisibility_by_101_l2522_252271


namespace inverse_f_93_l2522_252278

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_f_93 : f⁻¹ 93 = (28 : ℝ)^(1/3) := by sorry

end inverse_f_93_l2522_252278


namespace existence_of_special_polynomial_l2522_252231

theorem existence_of_special_polynomial :
  ∃ (f : Polynomial ℤ), 
    (∀ (i : ℕ), (f.coeff i = 1 ∨ f.coeff i = -1)) ∧ 
    (∃ (g : Polynomial ℤ), f = g * (X - 1) ^ 2013) :=
by sorry

end existence_of_special_polynomial_l2522_252231


namespace inequality_proof_l2522_252240

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0) 
  (h5 : x + y + z + t = 2) : 
  Real.sqrt (x^2 + z^2) + Real.sqrt (x^2 + 1) + Real.sqrt (z^2 + y^2) + 
  Real.sqrt (y^2 + t^2) + Real.sqrt (t^2 + 4) ≥ 5 := by
  sorry

end inequality_proof_l2522_252240


namespace typing_area_percentage_l2522_252274

/-- Calculates the percentage of a rectangular sheet used for typing, given the sheet dimensions and margins. -/
theorem typing_area_percentage (sheet_length sheet_width side_margin top_bottom_margin : ℝ) 
  (sheet_length_pos : 0 < sheet_length)
  (sheet_width_pos : 0 < sheet_width)
  (side_margin_pos : 0 < side_margin)
  (top_bottom_margin_pos : 0 < top_bottom_margin)
  (side_margin_fit : 2 * side_margin < sheet_length)
  (top_bottom_margin_fit : 2 * top_bottom_margin < sheet_width) :
  let total_area := sheet_length * sheet_width
  let typing_length := sheet_length - 2 * side_margin
  let typing_width := sheet_width - 2 * top_bottom_margin
  let typing_area := typing_length * typing_width
  (typing_area / total_area) * 100 = 64 :=
sorry

end typing_area_percentage_l2522_252274


namespace bike_cost_calculation_l2522_252211

/-- The cost of Carrie's bike --/
def bike_cost (hourly_wage : ℕ) (weekly_hours : ℕ) (weeks_per_month : ℕ) (remaining_money : ℕ) : ℕ :=
  hourly_wage * weekly_hours * weeks_per_month - remaining_money

/-- Theorem stating the cost of the bike --/
theorem bike_cost_calculation :
  bike_cost 8 35 4 720 = 400 := by
  sorry

end bike_cost_calculation_l2522_252211


namespace zoo_visitors_l2522_252227

theorem zoo_visitors (total_people : ℕ) (adult_price kid_price : ℕ) (total_sales : ℕ) 
  (h1 : total_people = 254)
  (h2 : adult_price = 28)
  (h3 : kid_price = 12)
  (h4 : total_sales = 3864) :
  ∃ (adults kids : ℕ), 
    adults + kids = total_people ∧
    adults * adult_price + kids * kid_price = total_sales ∧
    kids = 202 := by
  sorry

end zoo_visitors_l2522_252227


namespace age_sum_is_75_l2522_252234

/-- Given the ages of Alice, Bob, and Carol satisfying certain conditions, prove that the sum of their current ages is 75 years. -/
theorem age_sum_is_75 (alice bob carol : ℕ) : 
  (alice - 10 = (bob - 10) / 2) →  -- 10 years ago, Alice was half of Bob's age
  (4 * alice = 3 * bob) →          -- The ratio of their present ages is 3:4
  (carol = alice + bob + 5) →      -- Carol is 5 years older than the sum of Alice and Bob's current ages
  alice + bob + carol = 75 :=
by sorry

end age_sum_is_75_l2522_252234


namespace product_of_coefficients_l2522_252201

theorem product_of_coefficients (b c : ℤ) : 
  (∀ r : ℝ, r^2 - 2*r - 1 = 0 → r^5 - b*r - c = 0) → 
  b * c = 348 := by
sorry

end product_of_coefficients_l2522_252201


namespace exists_common_divisor_l2522_252216

/-- A function from positive integers to integers greater than 1 -/
def PositiveFunction := ℕ+ → ℕ+

/-- The property that f(m+n) divides f(m) + f(n) for all positive integers m and n -/
def HasDivisibilityProperty (f : PositiveFunction) : Prop :=
  ∀ m n : ℕ+, (f (m + n)) ∣ (f m + f n)

/-- The main theorem: if f has the divisibility property, then there exists c > 1 that divides all values of f -/
theorem exists_common_divisor (f : PositiveFunction) (h : HasDivisibilityProperty f) :
  ∃ c : ℕ+, c > 1 ∧ ∀ n : ℕ+, c ∣ f n :=
sorry

end exists_common_divisor_l2522_252216


namespace kenya_peanuts_l2522_252250

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_more : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_more = 48) : 
  jose_peanuts + kenya_more = 133 := by
sorry

end kenya_peanuts_l2522_252250


namespace different_elements_same_image_single_element_unique_image_elements_without_preimage_l2522_252220

-- Define the mapping f from A to B
variable {A B : Type}
variable (f : A → B)

-- Statement 1: Different elements in A can have the same image in B
theorem different_elements_same_image :
  ∃ (x y : A), x ≠ y ∧ f x = f y :=
sorry

-- Statement 2: A single element in A cannot have different images in B
theorem single_element_unique_image :
  ∀ (x : A) (y z : B), f x = y ∧ f x = z → y = z :=
sorry

-- Statement 3: There can be elements in B that do not have a pre-image in A
theorem elements_without_preimage :
  ∃ (y : B), ∀ (x : A), f x ≠ y :=
sorry

end different_elements_same_image_single_element_unique_image_elements_without_preimage_l2522_252220


namespace sum_of_legs_is_48_l2522_252251

/-- A right triangle with consecutive even whole number legs and hypotenuse 34 -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ
  hypotenuse : ℕ
  is_right : leg1^2 + leg2^2 = hypotenuse^2
  consecutive_even : leg2 = leg1 + 2
  hypotenuse_34 : hypotenuse = 34

/-- The sum of the legs of the special right triangle is 48 -/
theorem sum_of_legs_is_48 (t : RightTriangle) : t.leg1 + t.leg2 = 48 := by
  sorry

#check sum_of_legs_is_48

end sum_of_legs_is_48_l2522_252251


namespace cereal_consumption_time_l2522_252282

/-- The time taken for two people to consume a given amount of cereal together,
    given their individual consumption rates. -/
def time_to_consume (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Mr. Fat's cereal consumption rate in pounds per minute -/
def fat_rate : ℚ := 1 / 25

/-- Mr. Thin's cereal consumption rate in pounds per minute -/
def thin_rate : ℚ := 1 / 35

/-- The amount of cereal to be consumed in pounds -/
def cereal_amount : ℚ := 5

theorem cereal_consumption_time :
  ∃ (t : ℚ), abs (t - time_to_consume fat_rate thin_rate cereal_amount) < 1 ∧
             t = 73 := by sorry

end cereal_consumption_time_l2522_252282


namespace binomial_sum_l2522_252279

theorem binomial_sum (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + x)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 63 :=
by sorry

end binomial_sum_l2522_252279


namespace superadditive_continuous_function_is_linear_l2522_252298

/-- A function satisfying the given conditions -/
def SuperadditiveContinuousFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ f 0 = 0 ∧ ∀ x y : ℝ, f (x + y) ≥ f x + f y

/-- The main theorem -/
theorem superadditive_continuous_function_is_linear
    (f : ℝ → ℝ) (hf : SuperadditiveContinuousFunction f) :
    ∃ a : ℝ, ∀ x : ℝ, f x = a * x := by
  sorry

end superadditive_continuous_function_is_linear_l2522_252298


namespace min_value_exponential_function_l2522_252244

theorem min_value_exponential_function :
  (∀ x : ℝ, Real.exp x + 4 * Real.exp (-x) ≥ 4) ∧
  (∃ x : ℝ, Real.exp x + 4 * Real.exp (-x) = 4) :=
by sorry

end min_value_exponential_function_l2522_252244


namespace sum_of_squares_zero_l2522_252204

theorem sum_of_squares_zero (a b c : ℝ) : 
  (a - 2)^2 + (b + 3)^2 + (c - 7)^2 = 0 → a + b + c = 6 := by
sorry

end sum_of_squares_zero_l2522_252204
