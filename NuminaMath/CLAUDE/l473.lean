import Mathlib

namespace expression_equality_l473_47322

theorem expression_equality (x : ℝ) : 
  (Real.sqrt x / Real.sqrt 0.81 + Real.sqrt 0.81 / Real.sqrt 0.49 = 2.507936507936508) → 
  x = 1.21 := by
sorry

end expression_equality_l473_47322


namespace triangle_property_l473_47379

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, cos A = 1/2 and the area is (3√3)/4 -/
theorem triangle_property (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = π →  -- Angle sum in a triangle
  c * Real.cos A + a * Real.cos C = 2 * b * Real.cos A →  -- Given condition
  a = Real.sqrt 7 →  -- Given condition
  b + c = 4 →  -- Given condition
  Real.cos A = 1 / 2 ∧  -- Conclusion 1
  (1 / 2) * b * c * Real.sqrt (1 - (Real.cos A)^2) = (3 * Real.sqrt 3) / 4  -- Conclusion 2 (area)
  := by sorry

end triangle_property_l473_47379


namespace ted_stick_count_l473_47345

/-- Represents the number of objects thrown by a person -/
structure ThrowCount where
  sticks : ℕ
  rocks : ℕ

/-- The scenario of Bill and Ted throwing objects into the river -/
def river_throw_scenario (ted : ThrowCount) (bill : ThrowCount) : Prop :=
  bill.sticks = ted.sticks + 6 ∧
  ted.rocks = 2 * bill.rocks ∧
  bill.sticks + bill.rocks = 21

theorem ted_stick_count (ted : ThrowCount) (bill : ThrowCount) 
  (h : river_throw_scenario ted bill) : ted.sticks = 15 := by
  sorry

#check ted_stick_count

end ted_stick_count_l473_47345


namespace height_difference_after_three_years_l473_47320

/-- Represents the seasons of the year -/
inductive Season
  | spring
  | summer
  | fall
  | winter

/-- Calculates the growth of an object over a season given its monthly growth rate -/
def seasonalGrowth (monthlyRate : ℕ) : ℕ := 3 * monthlyRate

/-- Calculates the total growth over a year given seasonal growth rates -/
def yearlyGrowth (spring summer fall winter : ℕ) : ℕ :=
  seasonalGrowth spring + seasonalGrowth summer + seasonalGrowth fall + seasonalGrowth winter

/-- Theorem: The height difference between the tree and the boy after 3 years is 73 inches -/
theorem height_difference_after_three_years :
  let initialTreeHeight : ℕ := 16
  let initialBoyHeight : ℕ := 24
  let treeGrowth : Season → ℕ
    | Season.spring => 4
    | Season.summer => 6
    | Season.fall => 2
    | Season.winter => 1
  let boyGrowth : Season → ℕ
    | Season.spring => 2
    | Season.summer => 2
    | Season.fall => 0
    | Season.winter => 0
  let treeYearlyGrowth := yearlyGrowth (treeGrowth Season.spring) (treeGrowth Season.summer) (treeGrowth Season.fall) (treeGrowth Season.winter)
  let boyYearlyGrowth := yearlyGrowth (boyGrowth Season.spring) (boyGrowth Season.summer) (boyGrowth Season.fall) (boyGrowth Season.winter)
  let finalTreeHeight := initialTreeHeight + 3 * treeYearlyGrowth
  let finalBoyHeight := initialBoyHeight + 3 * boyYearlyGrowth
  finalTreeHeight - finalBoyHeight = 73 := by
  sorry


end height_difference_after_three_years_l473_47320


namespace orange_count_l473_47386

theorem orange_count (initial : ℕ) (thrown_away : ℕ) (added : ℕ) : 
  initial = 5 → thrown_away = 2 → added = 28 → 
  initial - thrown_away + added = 31 :=
by sorry

end orange_count_l473_47386


namespace oranges_packed_l473_47319

/-- Given boxes that hold 10 oranges each and 265 boxes used, prove that the total number of oranges packed is 2650. -/
theorem oranges_packed (oranges_per_box : ℕ) (boxes_used : ℕ) (h1 : oranges_per_box = 10) (h2 : boxes_used = 265) :
  oranges_per_box * boxes_used = 2650 := by
  sorry

end oranges_packed_l473_47319


namespace third_term_geometric_series_l473_47383

/-- Theorem: Third term of a specific geometric series -/
theorem third_term_geometric_series
  (q : ℝ) 
  (h₁ : |q| < 1)
  (h₂ : 2 / (1 - q) = 8 / 5)
  (h₃ : 2 * q = -1 / 2) :
  2 * q^2 = 1 / 8 := by
  sorry

end third_term_geometric_series_l473_47383


namespace cafeteria_vertical_stripes_l473_47308

def cafeteria_problem (total : ℕ) (checkered : ℕ) (horizontal_multiplier : ℕ) : Prop :=
  let stripes : ℕ := total - checkered
  let horizontal : ℕ := horizontal_multiplier * checkered
  let vertical : ℕ := stripes - horizontal
  vertical = 5

theorem cafeteria_vertical_stripes :
  cafeteria_problem 40 7 4 := by
  sorry

end cafeteria_vertical_stripes_l473_47308


namespace min_max_values_on_interval_monotone_increasing_condition_l473_47333

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs (x - 1)

-- Part I
theorem min_max_values_on_interval (a : ℝ) (h : a = 2) :
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f a y ≤ f a x) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = 1) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = 6) := by
  sorry

-- Part II
theorem monotone_increasing_condition :
  (∀ a : ℝ, a ∈ Set.Icc (-2) 0 ↔ Monotone (f a)) := by
  sorry

end min_max_values_on_interval_monotone_increasing_condition_l473_47333


namespace negation_equivalence_l473_47339

theorem negation_equivalence (a b x : ℝ) : 
  ¬(x ≠ a ∧ x ≠ b → x^2 - (a+b)*x + a*b ≠ 0) ↔ 
  (x = a ∨ x = b → x^2 - (a+b)*x + a*b = 0) :=
by sorry

end negation_equivalence_l473_47339


namespace ten_steps_climb_l473_47393

def climb_stairs (n : ℕ) : ℕ :=
  if n ≤ 1 then 1
  else if n = 2 then 2
  else climb_stairs (n - 1) + climb_stairs (n - 2)

theorem ten_steps_climb : climb_stairs 10 = 89 := by
  sorry

end ten_steps_climb_l473_47393


namespace marble_count_exceeds_200_l473_47334

def marbles (n : ℕ) : ℕ := 3 * 2^n

theorem marble_count_exceeds_200 :
  (∃ k : ℕ, marbles k > 200) ∧ 
  (∀ j : ℕ, j < 8 → marbles j ≤ 200) ∧
  (marbles 8 > 200) := by
sorry

end marble_count_exceeds_200_l473_47334


namespace equation_solution_l473_47301

theorem equation_solution (x y : ℝ) :
  (2 * x) / (1 + x^2) = (1 + y^2) / (2 * y) →
  ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) := by
sorry

end equation_solution_l473_47301


namespace product_sum_6936_l473_47390

theorem product_sum_6936 : ∃ a b : ℕ, 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 6936 ∧ 
  a + b = 168 := by
sorry

end product_sum_6936_l473_47390


namespace regular_octahedron_has_six_vertices_l473_47315

/-- A regular octahedron is a Platonic solid with equilateral triangular faces. -/
structure RegularOctahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of vertices in a regular octahedron -/
def num_vertices (o : RegularOctahedron) : ℕ := 6

/-- Theorem: A regular octahedron has 6 vertices -/
theorem regular_octahedron_has_six_vertices (o : RegularOctahedron) : 
  num_vertices o = 6 := by
  sorry

end regular_octahedron_has_six_vertices_l473_47315


namespace sum_of_factors_72_l473_47382

/-- Sum of positive factors of a natural number -/
def sumOfFactors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

/-- Theorem: The sum of all positive factors of 72 is 195 -/
theorem sum_of_factors_72 : sumOfFactors 72 = 195 := by
  sorry

end sum_of_factors_72_l473_47382


namespace square_roots_problem_l473_47329

theorem square_roots_problem (a : ℝ) (x : ℝ) 
  (h1 : a > 0)
  (h2 : (2*x + 6)^2 = a)
  (h3 : (x - 18)^2 = a) :
  a = 196 := by
sorry

end square_roots_problem_l473_47329


namespace symmetric_point_of_P_l473_47369

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The symmetric point of a given point with respect to the x-axis. -/
def symmetricPointXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The symmetric point of P(2, -5) with respect to the x-axis is (2, 5). -/
theorem symmetric_point_of_P : 
  let P : Point := { x := 2, y := -5 }
  symmetricPointXAxis P = { x := 2, y := 5 } := by
  sorry

end symmetric_point_of_P_l473_47369


namespace average_salary_proof_l473_47371

theorem average_salary_proof (n : ℕ) (total_salary : ℕ → ℕ) : 
  (∃ (m : ℕ), m > 0 ∧ total_salary m / m = 8000) →
  total_salary 4 / 4 = 8450 →
  total_salary 1 = 6500 →
  total_salary 5 = 4700 →
  (total_salary 5 + (total_salary 4 - total_salary 1)) / 4 = 8000 :=
by
  sorry

end average_salary_proof_l473_47371


namespace rhombus_longest_diagonal_l473_47384

/-- Given a rhombus with area 144 and diagonal ratio 4:3, prove its longest diagonal is 8√6 -/
theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℝ) (long_diag : ℝ) : 
  area = 144 → ratio = 4/3 → long_diag = 8 * Real.sqrt 6 := by
  sorry

end rhombus_longest_diagonal_l473_47384


namespace marble_probability_l473_47368

theorem marble_probability (total : ℕ) (blue : ℕ) (red : ℕ) 
  (h_total : total = 50)
  (h_blue : blue = 12)
  (h_red : red = 18)
  (h_white : total - blue - red = 20) :
  (red + (total - blue - red)) / total = 19 / 25 := by
  sorry

end marble_probability_l473_47368


namespace element_in_M_l473_47389

def M : Set (ℕ × ℕ) := {(2, 3)}

theorem element_in_M : (2, 3) ∈ M := by
  sorry

end element_in_M_l473_47389


namespace complement_of_A_in_U_l473_47330

-- Define the universal set U
def U : Set ℝ := {x | -Real.sqrt 3 < x}

-- Define set A
def A : Set ℝ := {x | 2^x > Real.sqrt 2}

-- Statement to prove
theorem complement_of_A_in_U :
  Set.compl A ∩ U = Set.Icc (-Real.sqrt 3) (1/2) := by sorry

end complement_of_A_in_U_l473_47330


namespace card_probability_l473_47331

def standard_deck : ℕ := 52
def num_jacks : ℕ := 4
def num_queens : ℕ := 4

theorem card_probability : 
  let p_two_queens := (num_queens / standard_deck) * ((num_queens - 1) / (standard_deck - 1))
  let p_one_jack := 2 * (num_jacks / standard_deck) * ((standard_deck - num_jacks) / (standard_deck - 1))
  let p_two_jacks := (num_jacks / standard_deck) * ((num_jacks - 1) / (standard_deck - 1))
  p_two_queens + p_one_jack + p_two_jacks = 2 / 13 := by
  sorry

end card_probability_l473_47331


namespace sum_of_distinct_remainders_div_13_l473_47325

def remainders : List Nat :=
  (List.range 10).map (fun n => (n + 1)^2 % 13)

def distinct_remainders : List Nat :=
  remainders.eraseDups

theorem sum_of_distinct_remainders_div_13 :
  (distinct_remainders.sum) / 13 = 3 := by
  sorry

end sum_of_distinct_remainders_div_13_l473_47325


namespace fixed_point_of_exponential_function_l473_47394

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 1
  f 2 = 2 := by sorry

end fixed_point_of_exponential_function_l473_47394


namespace limit_one_minus_cos_over_exp_squared_l473_47336

theorem limit_one_minus_cos_over_exp_squared :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |((1 - Real.cos x) / (Real.exp (3 * x) - 1)^2) - (1/18)| < ε := by
  sorry

end limit_one_minus_cos_over_exp_squared_l473_47336


namespace vinegar_mixture_l473_47340

/-- Given a mixture of water and vinegar, prove the amount of vinegar used. -/
theorem vinegar_mixture (total_mixture water_fraction vinegar_fraction : ℚ) 
  (h_total : total_mixture = 27)
  (h_water : water_fraction = 3/5)
  (h_vinegar : vinegar_fraction = 5/6)
  (h_water_amount : water_fraction * 20 + vinegar_fraction * vinegar_amount = total_mixture) :
  vinegar_amount = 15 := by
  sorry

#check vinegar_mixture

end vinegar_mixture_l473_47340


namespace count_non_negative_l473_47353

def number_set : List ℚ := [-15, 16/3, -23/100, 0, 76/10, 2, -3/5, 314/100]

theorem count_non_negative : (number_set.filter (λ x => x ≥ 0)).length = 5 := by
  sorry

end count_non_negative_l473_47353


namespace price_crossover_year_l473_47360

def price_X (year : ℕ) : ℚ :=
  5.20 + 0.45 * (year - 2001 : ℚ)

def price_Y (year : ℕ) : ℚ :=
  7.30 + 0.20 * (year - 2001 : ℚ)

theorem price_crossover_year :
  (∀ y : ℕ, y < 2010 → price_X y ≤ price_Y y) ∧
  price_X 2010 > price_Y 2010 := by
  sorry

end price_crossover_year_l473_47360


namespace sum_areas_externally_tangent_circles_l473_47338

/-- Given a 5-12-13 right triangle with vertices as centers of three mutually externally tangent circles,
    the sum of the areas of these circles is 113π. -/
theorem sum_areas_externally_tangent_circles (r s t : ℝ) : 
  r + s = 5 →
  s + t = 12 →
  r + t = 13 →
  π * (r^2 + s^2 + t^2) = 113 * π := by
  sorry

end sum_areas_externally_tangent_circles_l473_47338


namespace no_solution_quadratic_with_constraint_l473_47376

theorem no_solution_quadratic_with_constraint : 
  ¬ ∃ (x : ℝ), x^2 - 4*x + 4 = 0 ∧ x ≠ 2 := by
sorry

end no_solution_quadratic_with_constraint_l473_47376


namespace haley_cider_pints_l473_47378

/-- Represents the number of pints of cider Haley can make --/
def cider_pints (golden_apples_per_pint : ℕ) (pink_apples_per_pint : ℕ) 
  (apples_per_hour : ℕ) (farmhands : ℕ) (hours_worked : ℕ) 
  (golden_to_pink_ratio : ℚ) : ℕ :=
  let total_apples := apples_per_hour * farmhands * hours_worked
  let apples_per_pint := golden_apples_per_pint + pink_apples_per_pint
  total_apples / apples_per_pint

/-- Theorem stating the number of pints of cider Haley can make --/
theorem haley_cider_pints : 
  cider_pints 20 40 240 6 5 (1/3) = 120 := by
  sorry

end haley_cider_pints_l473_47378


namespace div_chain_equals_fraction_l473_47300

theorem div_chain_equals_fraction : (132 / 6) / 3 = 22 / 3 := by
  sorry

end div_chain_equals_fraction_l473_47300


namespace richard_david_age_difference_l473_47399

/-- The ages of Richard, David, and Scott in a family -/
structure FamilyAges where
  R : ℕ  -- Richard's age
  D : ℕ  -- David's age
  S : ℕ  -- Scott's age

/-- The conditions of the family ages problem -/
def FamilyAgesProblem (ages : FamilyAges) : Prop :=
  ages.R > ages.D ∧                 -- Richard is older than David
  ages.D = ages.S + 8 ∧             -- David is 8 years older than Scott
  ages.R + 8 = 2 * (ages.S + 8) ∧   -- In 8 years, Richard will be twice as old as Scott
  ages.D = 14                       -- David was 9 years old 5 years ago

/-- The theorem stating that Richard is 6 years older than David -/
theorem richard_david_age_difference (ages : FamilyAges) 
  (h : FamilyAgesProblem ages) : ages.R = ages.D + 6 := by
  sorry


end richard_david_age_difference_l473_47399


namespace card_combinations_l473_47318

theorem card_combinations : Nat.choose 40 7 = 1860480 := by sorry

end card_combinations_l473_47318


namespace cos_18_degrees_l473_47350

theorem cos_18_degrees : Real.cos (18 * π / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end cos_18_degrees_l473_47350


namespace f_satisfies_properties_l473_47327

-- Define the function f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_satisfies_properties :
  -- Property 1: f(x₁x₂) = f(x₁)f(x₂)
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧
  -- Property 2: When x ∈ (0, +∞), f'(x) > 0
  (∀ x : ℝ, x > 0 → deriv f x > 0) ∧
  -- Property 3: f'(x) is an odd function
  (∀ x : ℝ, deriv f (-x) = -(deriv f x)) :=
by
  sorry


end f_satisfies_properties_l473_47327


namespace new_unsigned_books_l473_47307

def adventure_books : ℕ := 13
def mystery_books : ℕ := 17
def scifi_books : ℕ := 25
def nonfiction_books : ℕ := 10
def used_books : ℕ := 42
def signed_books : ℕ := 10

theorem new_unsigned_books : 
  adventure_books + mystery_books + scifi_books + nonfiction_books - used_books - signed_books = 13 := by
  sorry

end new_unsigned_books_l473_47307


namespace ravens_age_l473_47317

theorem ravens_age (phoebe_age : ℕ) (raven_age : ℕ) : 
  phoebe_age = 10 →
  raven_age + 5 = 4 * (phoebe_age + 5) →
  raven_age = 55 := by
sorry

end ravens_age_l473_47317


namespace parabola_shift_right_parabola_shift_result_l473_47347

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c }

theorem parabola_shift_right (x : ℝ) :
  let original := Parabola.mk 1 6 0
  let shifted := shift_parabola original 4
  (x^2 + 6*x) = ((x-4)^2 + 6*(x-4)) :=
by sorry

theorem parabola_shift_result :
  let original := Parabola.mk 1 6 0
  let shifted := shift_parabola original 4
  shifted.a * x^2 + shifted.b * x + shifted.c = (x - 1)^2 - 9 :=
by sorry

end parabola_shift_right_parabola_shift_result_l473_47347


namespace max_pawns_2019_l473_47388

/-- Represents a chessboard with dimensions n x n -/
structure Chessboard (n : ℕ) where
  size : ℕ := n

/-- Represents the placement of pieces on the chessboard -/
structure Placement (n : ℕ) where
  board : Chessboard n
  pawns : ℕ
  rooks : ℕ
  no_rooks_see_each_other : Bool

/-- The maximum number of pawns that can be placed -/
def max_pawns (n : ℕ) : ℕ := (n / 2) ^ 2

/-- Theorem stating the maximum number of pawns for a 2019 x 2019 chessboard -/
theorem max_pawns_2019 :
  ∃ (p : Placement 2019),
    p.pawns = max_pawns 2019 ∧
    p.rooks = p.pawns + 2019 ∧
    p.no_rooks_see_each_other = true ∧
    ∀ (q : Placement 2019),
      q.no_rooks_see_each_other = true →
      q.rooks = q.pawns + 2019 →
      q.pawns ≤ p.pawns :=
by sorry

end max_pawns_2019_l473_47388


namespace leo_money_after_settling_debts_l473_47332

/-- The total amount of money Leo and Ryan have together -/
def total_amount : ℚ := 48

/-- The fraction of the total amount that Ryan owns -/
def ryan_fraction : ℚ := 2/3

/-- The amount Ryan owes Leo -/
def ryan_owes_leo : ℚ := 10

/-- The amount Leo owes Ryan -/
def leo_owes_ryan : ℚ := 7

/-- Leo's final amount after settling debts -/
def leo_final_amount : ℚ := 19

theorem leo_money_after_settling_debts :
  let ryan_initial := ryan_fraction * total_amount
  let leo_initial := total_amount - ryan_initial
  let net_debt := ryan_owes_leo - leo_owes_ryan
  leo_initial + net_debt = leo_final_amount := by
sorry

end leo_money_after_settling_debts_l473_47332


namespace find_A_in_rounding_l473_47323

theorem find_A_in_rounding : ∃ A : ℕ, 
  (A < 10) ∧ 
  (6000 + A * 100 + 35 ≥ 6100) ∧ 
  (6000 + (A + 1) * 100 + 35 > 6100) → 
  A = 1 := by
sorry

end find_A_in_rounding_l473_47323


namespace max_non_intersecting_diagonals_correct_l473_47342

/-- The maximum number of non-intersecting diagonals in a convex n-gon --/
def max_non_intersecting_diagonals (n : ℕ) : ℕ := n - 3

/-- Theorem stating that the maximum number of non-intersecting diagonals in a convex n-gon is n-3 --/
theorem max_non_intersecting_diagonals_correct (n : ℕ) (h : n ≥ 3) :
  max_non_intersecting_diagonals n = n - 3 :=
by sorry

end max_non_intersecting_diagonals_correct_l473_47342


namespace parking_lot_increase_l473_47328

def initial_cars : ℕ := 24
def final_cars : ℕ := 48

def percentage_increase (initial final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

theorem parking_lot_increase :
  percentage_increase initial_cars final_cars = 100 := by
  sorry

end parking_lot_increase_l473_47328


namespace arithmetic_sequence_common_difference_l473_47397

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 7 = -8)
  (h_second : a 2 = 2) :
  ∃ d : ℤ, d = -3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l473_47397


namespace right_triangle_hypotenuse_l473_47380

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 36 →
  b = 48 →
  c^2 = a^2 + b^2 →
  c = 60 :=
by sorry

end right_triangle_hypotenuse_l473_47380


namespace quadratic_no_real_roots_l473_47367

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := by
sorry

end quadratic_no_real_roots_l473_47367


namespace vacation_cost_division_l473_47354

theorem vacation_cost_division (total_cost : ℝ) (initial_people : ℕ) (cost_reduction : ℝ) (n : ℕ) : 
  total_cost = 375 ∧ 
  initial_people = 3 ∧ 
  cost_reduction = 50 ∧ 
  (total_cost / initial_people) - (total_cost / n) = cost_reduction →
  n = 5 := by
sorry

end vacation_cost_division_l473_47354


namespace max_triangle_area_l473_47346

/-- Given a triangle ABC with side lengths a, b, c and internal angles A, B, C,
    this theorem states that the maximum area of the triangle is √2 when
    a = √2, b² - c² = 6, and angle A is at its maximum. -/
theorem max_triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 2 →
  b^2 - c^2 = 6 →
  (∀ (a' b' c' : ℝ) (A' B' C' : ℝ),
    a' = Real.sqrt 2 →
    b'^2 - c'^2 = 6 →
    A' ≤ A) →
  (1/2 * b * c * Real.sin A) = Real.sqrt 2 :=
sorry

end max_triangle_area_l473_47346


namespace two_out_of_five_permutation_l473_47396

theorem two_out_of_five_permutation : 
  (Finset.range 5).card * (Finset.range 4).card = 20 := by
  sorry

end two_out_of_five_permutation_l473_47396


namespace smallest_satisfying_number_l473_47361

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

def satisfies_condition (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ all_odd_digits (n + reverse_digits n)

theorem smallest_satisfying_number :
  satisfies_condition 209 ∧
  ∀ m : ℕ, satisfies_condition m → m ≥ 209 :=
sorry

end smallest_satisfying_number_l473_47361


namespace house_rent_calculation_l473_47387

/-- Given a person's expenditure pattern and petrol cost, calculate house rent -/
theorem house_rent_calculation (income : ℝ) (petrol_percentage : ℝ) (rent_percentage : ℝ) (petrol_cost : ℝ) : 
  petrol_percentage = 0.3 →
  rent_percentage = 0.2 →
  petrol_cost = 300 →
  petrol_cost = petrol_percentage * income →
  rent_percentage * (income - petrol_cost) = 140 :=
by sorry

end house_rent_calculation_l473_47387


namespace sams_french_bulldogs_count_l473_47377

/-- The number of French Bulldogs Sam has -/
def sams_french_bulldogs : ℕ := 4

/-- The number of German Shepherds Sam has -/
def sams_german_shepherds : ℕ := 3

/-- The total number of dogs Peter wants -/
def peters_total_dogs : ℕ := 17

theorem sams_french_bulldogs_count :
  sams_french_bulldogs = 4 :=
by
  have h1 : peters_total_dogs = 3 * sams_german_shepherds + 2 * sams_french_bulldogs :=
    by sorry
  sorry

end sams_french_bulldogs_count_l473_47377


namespace strawberry_jam_money_l473_47365

/-- Calculates the total money made from selling strawberry jam given the number of strawberries picked by Betty, Matthew, and Natalie, and the jam-making and selling conditions. -/
theorem strawberry_jam_money (betty_strawberries : ℕ) (matthew_extra : ℕ) (jam_strawberries : ℕ) (jar_price : ℕ) : 
  betty_strawberries = 25 →
  matthew_extra = 30 →
  jam_strawberries = 12 →
  jar_price = 6 →
  let matthew_strawberries := betty_strawberries + matthew_extra
  let natalie_strawberries := matthew_strawberries / 3
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let jars := total_strawberries / jam_strawberries
  let total_money := jars * jar_price
  total_money = 48 := by
sorry

end strawberry_jam_money_l473_47365


namespace initial_cells_count_l473_47358

/-- Calculates the number of cells after one hour given the initial number -/
def cellsAfterOneHour (initialCells : ℕ) : ℕ :=
  2 * (initialCells - 2)

/-- Calculates the number of cells after n hours given the initial number -/
def cellsAfterNHours (initialCells n : ℕ) : ℕ :=
  match n with
  | 0 => initialCells
  | m + 1 => cellsAfterOneHour (cellsAfterNHours initialCells m)

/-- Theorem stating that if there are 164 cells after 5 hours, the initial number of cells was 9 -/
theorem initial_cells_count (initialCells : ℕ) :
  cellsAfterNHours initialCells 5 = 164 → initialCells = 9 :=
by
  sorry

#check initial_cells_count

end initial_cells_count_l473_47358


namespace parallel_lines_distance_l473_47385

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0
def line2 (x y : ℝ) (b c : ℝ) : Prop := 6 * x + b * y + c = 0

-- Define the distance between lines
def distance_between_lines (b c : ℝ) : ℝ := 3

-- Define the parallelism condition
def parallel_lines (b : ℝ) : Prop := b = 8

theorem parallel_lines_distance (b c : ℝ) :
  parallel_lines b → distance_between_lines b c = 3 →
  (b + c = -12 ∨ b + c = 48) :=
by sorry

end parallel_lines_distance_l473_47385


namespace solution_set_when_a_is_3_range_of_a_l473_47343

-- Define the function f
def f (a x : ℝ) : ℝ := |a - x| + |x + 2|

-- Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x < 7} = {x : ℝ | -3 < x ∧ x < 4} := by sorry

-- Part II
theorem range_of_a :
  {a : ℝ | ∀ x ∈ Set.Icc 1 2, |f a x| ≤ |x + 4|} = Set.Icc 0 3 := by sorry

end solution_set_when_a_is_3_range_of_a_l473_47343


namespace square_difference_pattern_l473_47348

theorem square_difference_pattern (n : ℕ) :
  (2*n + 2)^2 - (2*n)^2 = 4*(2*n + 1) := by
  sorry

end square_difference_pattern_l473_47348


namespace stratified_sampling_suitable_l473_47305

/-- Represents a sampling method -/
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a school population -/
structure SchoolPopulation where
  total_students : Nat
  boys : Nat
  girls : Nat
  sample_size : Nat

/-- Determines if a sampling method is suitable for a given school population -/
def is_suitable_sampling_method (population : SchoolPopulation) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Stratified ∧
  population.total_students = population.boys + population.girls ∧
  population.sample_size < population.total_students

/-- Theorem stating that stratified sampling is suitable for the given school population -/
theorem stratified_sampling_suitable (population : SchoolPopulation) 
  (h1 : population.total_students = 1000)
  (h2 : population.boys = 520)
  (h3 : population.girls = 480)
  (h4 : population.sample_size = 100) :
  is_suitable_sampling_method population SamplingMethod.Stratified :=
by
  sorry


end stratified_sampling_suitable_l473_47305


namespace crayons_difference_proof_l473_47344

-- Define the given conditions
def initial_crayons : ℕ := 4 * 8
def crayons_to_mae : ℕ := 5
def crayons_left : ℕ := 15

-- Define the theorem
theorem crayons_difference_proof : 
  (initial_crayons - crayons_to_mae - crayons_left) - crayons_to_mae = 7 := by
  sorry

end crayons_difference_proof_l473_47344


namespace horner_method_for_f_l473_47309

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_method_for_f :
  f 2 = horner [2, 3, 0, 5, -4] 2 ∧ horner [2, 3, 0, 5, -4] 2 = 62 := by
  sorry

end horner_method_for_f_l473_47309


namespace even_quadratic_implies_k_equals_one_l473_47373

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The quadratic function f(x) = kx^2 + (k-1)x + 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 3

theorem even_quadratic_implies_k_equals_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end even_quadratic_implies_k_equals_one_l473_47373


namespace matrix_power_vector_product_l473_47312

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; -1, 4]
def a : Matrix (Fin 2) (Fin 1) ℝ := !![7; 4]

theorem matrix_power_vector_product :
  A^6 * a = !![435; 339] := by sorry

end matrix_power_vector_product_l473_47312


namespace polygon_sides_l473_47362

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →                           -- n is at least 3 (for a valid polygon)
  ((n - 2) * 180 = 3 * 360) →         -- sum of interior angles = 3 * sum of exterior angles
  n = 8                               -- the polygon has 8 sides
:= by sorry

end polygon_sides_l473_47362


namespace B_power_98_l473_47395

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]]

theorem B_power_98 : B^98 = ![![-1, 0, 0],
                              ![0, -1, 0],
                              ![0, 0, 1]] := by sorry

end B_power_98_l473_47395


namespace set_intersection_example_l473_47335

theorem set_intersection_example :
  let A : Set ℕ := {0, 1, 2}
  let B : Set ℕ := {1, 2, 3, 4}
  A ∩ B = {1, 2} := by
sorry

end set_intersection_example_l473_47335


namespace solve_for_Y_l473_47359

theorem solve_for_Y : ∃ Y : ℤ, 80 - (Y - (6 + 2 * (7 - 8 - 5))) = 89 ∧ Y = -15 := by
  sorry

end solve_for_Y_l473_47359


namespace solve_board_game_problem_l473_47374

def board_game_problem (cost_per_game : ℕ) (payment : ℕ) (change_bills : ℕ) (change_denomination : ℕ) : Prop :=
  let total_change : ℕ := change_bills * change_denomination
  let spent : ℕ := payment - total_change
  spent / cost_per_game = 6 ∧ spent % cost_per_game = 0

theorem solve_board_game_problem :
  board_game_problem 15 100 2 5 := by
  sorry

end solve_board_game_problem_l473_47374


namespace sum_of_24_numbers_l473_47364

theorem sum_of_24_numbers (numbers : List ℤ) : 
  numbers.length = 24 → numbers.sum = 576 → 
  (∀ n ∈ numbers, Even n) ∨ 
  (∃ (evens odds : List ℤ), 
    numbers = evens ++ odds ∧ 
    (∀ n ∈ evens, Even n) ∧ 
    (∀ n ∈ odds, Odd n) ∧ 
    Even (odds.length)) :=
by sorry

end sum_of_24_numbers_l473_47364


namespace find_different_coins_possible_l473_47341

/-- Represents the result of a weighing --/
inductive WeighResult
  | Equal : WeighResult
  | Left : WeighResult
  | Right : WeighResult

/-- Represents a set of coins --/
structure CoinSet where
  total : Nat
  heavy : Nat
  light : Nat
  h_equal_light : heavy = light
  h_total : total = heavy + light

/-- Represents a weighing operation --/
def weigh (left right : CoinSet) : WeighResult :=
  sorry

/-- Represents the process of finding two coins of different weights --/
def findDifferentCoins (coins : CoinSet) (maxWeighings : Nat) : Bool :=
  sorry

/-- The main theorem to be proved --/
theorem find_different_coins_possible :
  ∃ (strategy : CoinSet → Nat → Bool),
    let initialCoins : CoinSet := {
      total := 128,
      heavy := 64,
      light := 64,
      h_equal_light := rfl,
      h_total := rfl
    }
    strategy initialCoins 7 = true :=
  sorry

end find_different_coins_possible_l473_47341


namespace sum_of_fractions_l473_47349

theorem sum_of_fractions : (1 : ℚ) / 6 + 5 / 12 = 7 / 12 := by
  sorry

end sum_of_fractions_l473_47349


namespace poster_area_is_28_l473_47351

/-- The area of a rectangular poster -/
def poster_area (width height : ℝ) : ℝ := width * height

/-- Theorem: The area of a rectangular poster with width 4 inches and height 7 inches is 28 square inches -/
theorem poster_area_is_28 : poster_area 4 7 = 28 := by
  sorry

end poster_area_is_28_l473_47351


namespace max_intersections_proof_l473_47356

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 12

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 6

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := Nat.choose num_x_points 2 * Nat.choose num_y_points 2

theorem max_intersections_proof :
  max_intersections = 990 :=
sorry

end max_intersections_proof_l473_47356


namespace math_club_members_l473_47370

theorem math_club_members (total_books : ℕ) (books_per_member : ℕ) (members_per_book : ℕ) :
  total_books = 12 →
  books_per_member = 2 →
  members_per_book = 3 →
  total_books * members_per_book = books_per_member * (total_books * members_per_book / books_per_member) :=
by sorry

end math_club_members_l473_47370


namespace triangle_side_and_area_l473_47324

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  cosC : ℝ

/-- Theorem about the side length and area of a specific triangle -/
theorem triangle_side_and_area (t : Triangle) 
  (h1 : t.a = 1)
  (h2 : t.b = 2)
  (h3 : t.cosC = 1/4) :
  t.c = 2 ∧ (1/2 * t.a * t.b * Real.sqrt (1 - t.cosC^2)) = Real.sqrt 15 / 4 := by
  sorry


end triangle_side_and_area_l473_47324


namespace jam_eaten_for_lunch_l473_47357

theorem jam_eaten_for_lunch (x : ℚ) : 
  (1 - x) * (1 - 1/7) = 4/7 → x = 1/21 := by
  sorry

end jam_eaten_for_lunch_l473_47357


namespace car_distance_is_360_l473_47372

/-- The distance a car needs to cover, given initial time, time factor, and new speed. -/
def car_distance (initial_time : ℝ) (time_factor : ℝ) (new_speed : ℝ) : ℝ :=
  initial_time * time_factor * new_speed

/-- Theorem stating that the car distance is 360 kilometers under given conditions. -/
theorem car_distance_is_360 :
  car_distance 6 (3/2) 40 = 360 := by
  sorry

end car_distance_is_360_l473_47372


namespace curve_intersection_and_tangent_l473_47355

noncomputable section

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := Real.exp x * (c*x + d)

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 2*x + a

-- Define the derivative of g
def g' (c d x : ℝ) : ℝ := Real.exp x * (c*x + d + c)

-- State the theorem
theorem curve_intersection_and_tangent (a b c d : ℝ) :
  (f a b 0 = 2) →
  (g c d 0 = 2) →
  (f' a 0 = 4) →
  (g' c d 0 = 4) →
  (a = 4 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∧
  (∀ k, (∀ x, x ≥ -2 → f 4 2 x ≤ k * g 2 2 x) ↔ (1 ≤ k ∧ k ≤ Real.exp 2)) :=
by sorry

end

end curve_intersection_and_tangent_l473_47355


namespace cubic_function_coefficient_l473_47337

/-- Given a cubic function f(x) = ax^3 + bx^2 + cx + d, 
    if f(-1) = 0, f(1) = 0, and f(0) = 2, then b = -2 -/
theorem cubic_function_coefficient (a b c d : ℝ) : 
  let f := λ x : ℝ => a * x^3 + b * x^2 + c * x + d
  (f (-1) = 0) → (f 1 = 0) → (f 0 = 2) → b = -2 := by
  sorry

end cubic_function_coefficient_l473_47337


namespace batsman_average_problem_l473_47326

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem statement for the batsman's average problem -/
theorem batsman_average_problem (stats : BatsmanStats) 
  (h1 : stats.innings = 11)
  (h2 : newAverage stats 55 = stats.average + 1) :
  newAverage stats 55 = 44 := by
  sorry

#check batsman_average_problem

end batsman_average_problem_l473_47326


namespace prism_36_edges_14_faces_l473_47314

/-- A prism is a polyhedron with two congruent parallel faces (bases) and lateral faces that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  2 + p.edges / 3

/-- Theorem: A prism with 36 edges has 14 faces. -/
theorem prism_36_edges_14_faces (p : Prism) (h : p.edges = 36) : num_faces p = 14 := by
  sorry


end prism_36_edges_14_faces_l473_47314


namespace inequality_addition_l473_47321

theorem inequality_addition (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end inequality_addition_l473_47321


namespace sqrt_cube_root_problem_l473_47381

theorem sqrt_cube_root_problem (x y : ℝ) : 
  y = Real.sqrt (x - 24) + Real.sqrt (24 - x) - 8 → 
  (x - 5 * y)^(1/3 : ℝ) = 4 := by
  sorry

end sqrt_cube_root_problem_l473_47381


namespace inequality_proof_l473_47352

theorem inequality_proof (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hu₁ : x₁ * y₁ - z₁^2 > 0) (hu₂ : x₂ * y₂ - z₂^2 > 0) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) := by
  sorry

end inequality_proof_l473_47352


namespace perfect_square_condition_l473_47313

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, n^4 - 4*n^3 + 22*n^2 - 36*n + 18 = m^2) ↔ (n = 1 ∨ n = 3) :=
sorry

end perfect_square_condition_l473_47313


namespace polynomial_divisibility_sum_l473_47316

/-- A complex number ω such that ω^2 + ω + 1 = 0 -/
def ω : ℂ := sorry

/-- The polynomial x^103 + Cx^2 + Dx + E -/
def f (C D E : ℝ) (x : ℂ) : ℂ := x^103 + C*x^2 + D*x + E

/-- The polynomial x^2 + x + 1 -/
def g (x : ℂ) : ℂ := x^2 + x + 1

theorem polynomial_divisibility_sum (C D E : ℝ) :
  (∀ x, g x = 0 → f C D E x = 0) → C + D + E = 2 := by sorry

end polynomial_divisibility_sum_l473_47316


namespace tuna_price_is_two_l473_47304

/-- Represents the daily catch and earnings of a fisherman -/
structure FishermanData where
  red_snappers : ℕ
  tunas : ℕ
  red_snapper_price : ℚ
  daily_earnings : ℚ

/-- Calculates the price of a Tuna given the fisherman's data -/
def tuna_price (data : FishermanData) : ℚ :=
  (data.daily_earnings - data.red_snappers * data.red_snapper_price) / data.tunas

/-- Theorem stating that the price of a Tuna is $2 given the fisherman's data -/
theorem tuna_price_is_two (data : FishermanData)
  (h1 : data.red_snappers = 8)
  (h2 : data.tunas = 14)
  (h3 : data.red_snapper_price = 3)
  (h4 : data.daily_earnings = 52) :
  tuna_price data = 2 := by
  sorry

end tuna_price_is_two_l473_47304


namespace max_popsicles_l473_47366

def lucy_budget : ℚ := 15
def popsicle_cost : ℚ := 2.4

theorem max_popsicles : 
  ∀ n : ℕ, (n : ℚ) * popsicle_cost ≤ lucy_budget → n ≤ 6 :=
by sorry

end max_popsicles_l473_47366


namespace red_balls_count_l473_47311

theorem red_balls_count (total : ℕ) (white : ℕ) (red : ℕ) (prob_white : ℚ) : 
  white = 5 →
  total = white + red →
  prob_white = 1/4 →
  (white : ℚ) / total = prob_white →
  red = 15 := by
sorry

end red_balls_count_l473_47311


namespace solution_value_l473_47391

theorem solution_value (x y a : ℝ) : 
  x = 1 → y = -3 → a * x - y = 1 → a = -2 := by
  sorry

end solution_value_l473_47391


namespace problem_solution_l473_47375

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6 * a * x - 11
def g (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 12

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 6 * a

theorem problem_solution :
  ∀ a k : ℝ,
  (f' a (-1) = 0 → a = -2) ∧
  (∃ x y : ℝ, f a x = k * x + 9 ∧ f' a x = k ∧ g x = k * x + 9 ∧ (3 * 2 * x + 6) = k → k = 0) :=
by sorry

end problem_solution_l473_47375


namespace supplementary_angle_of_39_23_l473_47392

-- Define the angle type with degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the supplementary angle function
def supplementaryAngle (a : Angle) : Angle :=
  let totalMinutes := 180 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem supplementary_angle_of_39_23 :
  let a : Angle := ⟨39, 23⟩
  supplementaryAngle a = ⟨140, 37⟩ := by
  sorry

end supplementary_angle_of_39_23_l473_47392


namespace fixed_point_of_exponential_translation_l473_47306

theorem fixed_point_of_exponential_translation (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x-3) + 3
  f 3 = 4 := by
  sorry

end fixed_point_of_exponential_translation_l473_47306


namespace optimal_cylinder_ratio_l473_47302

/-- Theorem: Optimal ratio of height to radius for a cylinder with minimal surface area -/
theorem optimal_cylinder_ratio (V : ℝ) (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  V = π * r^2 * h ∧ V = 1000 → 
  (∀ h' r', h' > 0 → r' > 0 → V = π * r'^2 * h' → 
    2 * π * r^2 + 2 * π * r * h ≤ 2 * π * r'^2 + 2 * π * r' * h') →
  h / r = 1 := by
sorry

end optimal_cylinder_ratio_l473_47302


namespace regular_18gon_symmetry_sum_l473_47398

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  n_pos : n > 0

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_18gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by sorry

end regular_18gon_symmetry_sum_l473_47398


namespace constant_sequence_conditions_l473_47363

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is constant if all its terms are equal -/
def is_constant (a : Sequence) : Prop :=
  ∀ n m : ℕ, a n = a m

/-- A sequence is geometric if the ratio of consecutive terms is constant -/
def is_geometric (a : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is arithmetic if the difference of consecutive terms is constant -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem constant_sequence_conditions (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) (a : Sequence) :
  (is_geometric a ∧ is_geometric (fun n ↦ k * a n + b)) ∨
  (is_arithmetic a ∧ is_geometric (fun n ↦ k * a n + b)) ∨
  (is_geometric a ∧ is_arithmetic (fun n ↦ k * a n + b))
  → is_constant a := by
  sorry

end constant_sequence_conditions_l473_47363


namespace quadratic_from_means_l473_47310

theorem quadratic_from_means (α β : ℝ) : 
  (α + β) / 2 = 8 → 
  (α * β) = 15^2 → 
  ∀ x, x^2 - 16*x + 225 = 0 ↔ (x = α ∨ x = β) := by
sorry

end quadratic_from_means_l473_47310


namespace antons_offer_is_cheapest_l473_47303

/-- Represents a shareholder in the company -/
structure Shareholder where
  name : String
  shares : Nat
  yield : Rat

/-- Represents the company and its shareholders -/
structure Company where
  totalShares : Nat
  sharePrice : Nat
  shareholders : List Shareholder

def Company.largestShareholderShares (c : Company) : Nat :=
  c.shareholders.map (·.shares) |>.maximum?.getD 0

def buySharesCost (sharePrice : Nat) (shares : Nat) (yield : Rat) : Nat :=
  Nat.ceil (sharePrice * shares * (1 + yield))

theorem antons_offer_is_cheapest (c : Company) (arina : Shareholder) : 
  c.totalShares = 300000 ∧
  c.sharePrice = 10 ∧
  arina.shares = 90001 ∧
  c.shareholders = [
    ⟨"Maxim", 104999, 1/10⟩,
    ⟨"Inga", 30000, 1/4⟩,
    ⟨"Yuri", 30000, 3/20⟩,
    ⟨"Yulia", 30000, 3/10⟩,
    ⟨"Anton", 15000, 2/5⟩
  ] →
  let requiredShares := c.largestShareholderShares - arina.shares + 1
  let antonsCost := buySharesCost c.sharePrice (c.shareholders.find? (·.name = "Anton") |>.map (·.shares) |>.getD 0) (2/5)
  ∀ s ∈ c.shareholders, s.name ≠ "Anton" →
    buySharesCost c.sharePrice s.shares s.yield ≥ antonsCost ∧
    s.shares ≥ requiredShares →
    antonsCost ≤ buySharesCost c.sharePrice s.shares s.yield :=
by sorry


end antons_offer_is_cheapest_l473_47303
