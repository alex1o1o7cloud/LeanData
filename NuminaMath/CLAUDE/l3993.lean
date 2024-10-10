import Mathlib

namespace jorge_goals_this_season_l3993_399314

/-- Given that Jorge scored 156 goals last season and the total number of goals he scored is 343,
    prove that the number of goals he scored this season is 187. -/
theorem jorge_goals_this_season (goals_last_season goals_total : ℕ)
    (h1 : goals_last_season = 156)
    (h2 : goals_total = 343) :
    goals_total - goals_last_season = 187 := by
  sorry

end jorge_goals_this_season_l3993_399314


namespace parallel_line_necessary_not_sufficient_l3993_399385

-- Define the type for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLineToPlane : Line → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_line_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (distinct : α ≠ β)
  (m_in_α : lineInPlane m α) :
  (parallelPlanes α β → parallelLineToPlane m β) ∧
  ¬(parallelLineToPlane m β → parallelPlanes α β) :=
sorry

end parallel_line_necessary_not_sufficient_l3993_399385


namespace fraction_problem_l3993_399339

theorem fraction_problem (f : ℚ) : 3 + (1/2) * f * (1/5) * 90 = (1/15) * 90 ↔ f = 1/3 := by
  sorry

end fraction_problem_l3993_399339


namespace functional_equation_solution_l3993_399377

/-- A monotonic continuous function on the real numbers satisfying f(x)·f(y) = f(x+y) -/
def FunctionalEquationSolution (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ Continuous f ∧ ∀ x y : ℝ, f x * f y = f (x + y)

/-- The solution to the functional equation is of the form f(x) = a^x for some a > 0 and a ≠ 1 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquationSolution f) :
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f x = a^x :=
sorry

end functional_equation_solution_l3993_399377


namespace distance_B_C_is_250_l3993_399303

/-- Represents a city in a triangle of cities -/
structure City :=
  (name : String)

/-- Represents the distance between two cities -/
def distance (a b : City) : ℝ := sorry

/-- The theorem stating the distance between cities B and C -/
theorem distance_B_C_is_250 (A B C : City) 
  (h1 : distance A B = distance A C + distance B C - 200)
  (h2 : distance A C = distance A B + distance B C - 300) :
  distance B C = 250 := by sorry

end distance_B_C_is_250_l3993_399303


namespace carly_payment_l3993_399329

/-- The final amount Carly needs to pay after discount -/
def final_amount (wallet_cost purse_cost shoes_cost discount_rate : ℝ) : ℝ :=
  let total_cost := wallet_cost + purse_cost + shoes_cost
  total_cost * (1 - discount_rate)

/-- Theorem: Given the conditions, Carly needs to pay $198.90 after discount -/
theorem carly_payment : 
  ∀ (wallet_cost purse_cost shoes_cost : ℝ),
    wallet_cost = 22 →
    purse_cost = 4 * wallet_cost - 3 →
    shoes_cost = wallet_cost + purse_cost + 7 →
    final_amount wallet_cost purse_cost shoes_cost 0.1 = 198.90 :=
by
  sorry

#eval final_amount 22 85 114 0.1

end carly_payment_l3993_399329


namespace equation_solutions_l3993_399324

def solution_set : Set (ℤ × ℤ) :=
  {(0, -4), (0, 8), (-2, 0), (-4, 8), (-6, 6), (0, 0), (-10, 4)}

def satisfies_equation (x y : ℤ) : Prop :=
  x + y ≠ 0 ∧ (x - y)^2 / (x + y) = x - y + 6

theorem equation_solutions :
  {p : ℤ × ℤ | satisfies_equation p.1 p.2} = solution_set := by sorry

end equation_solutions_l3993_399324


namespace max_value_inequality_l3993_399356

theorem max_value_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 2*a*b*Real.sqrt 2 + 2*b*c ≤ Real.sqrt 3 := by
  sorry

end max_value_inequality_l3993_399356


namespace complement_of_N_in_U_l3993_399315

def U : Set ℕ := {x | x > 0 ∧ x ≤ 5}
def N : Set ℕ := {2, 4}

theorem complement_of_N_in_U :
  U \ N = {1, 3, 5} := by
  sorry

end complement_of_N_in_U_l3993_399315


namespace inequality_not_always_preserved_l3993_399382

theorem inequality_not_always_preserved (a b : ℝ) (h : a < b) :
  ∃ m : ℝ, ¬(m * a > m * b) :=
sorry

end inequality_not_always_preserved_l3993_399382


namespace floor_sqrt_20_squared_l3993_399386

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end floor_sqrt_20_squared_l3993_399386


namespace tile_coverage_l3993_399374

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

theorem tile_coverage (tile : Dimensions) (region : Dimensions) : 
  tile.length = 2 ∧ tile.width = 6 ∧ 
  region.length = feetToInches 3 ∧ region.width = feetToInches 4 → 
  (area region / area tile : ℕ) = 144 := by
  sorry

#check tile_coverage

end tile_coverage_l3993_399374


namespace eq2_most_suitable_for_factorization_l3993_399355

/-- Represents a quadratic equation --/
inductive QuadraticEquation
  | Eq1 : QuadraticEquation  -- (x+1)(x-3)=2
  | Eq2 : QuadraticEquation  -- 2(x-2)^2=x^2-4
  | Eq3 : QuadraticEquation  -- x^2+3x-1=0
  | Eq4 : QuadraticEquation  -- 5(2-x)^2=3

/-- Predicate to determine if an equation is suitable for factorization --/
def isSuitableForFactorization : QuadraticEquation → Prop :=
  fun eq => match eq with
    | QuadraticEquation.Eq1 => False
    | QuadraticEquation.Eq2 => True
    | QuadraticEquation.Eq3 => False
    | QuadraticEquation.Eq4 => False

/-- Theorem stating that Eq2 is the most suitable for factorization --/
theorem eq2_most_suitable_for_factorization :
  ∀ eq : QuadraticEquation, 
    isSuitableForFactorization eq → eq = QuadraticEquation.Eq2 :=
by
  sorry

end eq2_most_suitable_for_factorization_l3993_399355


namespace first_class_size_l3993_399301

/-- The number of students in the second class -/
def second_class_students : ℕ := 48

/-- The average marks of the first class -/
def first_class_average : ℚ := 60

/-- The average marks of the second class -/
def second_class_average : ℚ := 58

/-- The average marks of all students -/
def total_average : ℚ := 59067961165048544 / 1000000000000000

/-- The number of students in the first class -/
def first_class_students : ℕ := 55

theorem first_class_size :
  (first_class_students * first_class_average + second_class_students * second_class_average) / 
  (first_class_students + second_class_students) = total_average :=
sorry

end first_class_size_l3993_399301


namespace chess_tournament_participants_l3993_399348

/-- Represents a chess tournament -/
structure ChessTournament where
  /-- The number of participants in the tournament -/
  participants : ℕ
  /-- The total number of games played in the tournament -/
  total_games : ℕ
  /-- Each participant plays exactly one game with each other participant -/
  one_game_each : total_games = participants * (participants - 1) / 2

/-- Theorem: A chess tournament with 190 games has 20 participants -/
theorem chess_tournament_participants (t : ChessTournament) 
    (h : t.total_games = 190) : t.participants = 20 := by
  sorry

#check chess_tournament_participants

end chess_tournament_participants_l3993_399348


namespace scientific_notation_of_1010659_l3993_399316

/-- The original number to be expressed in scientific notation -/
def original_number : ℕ := 1010659

/-- The number of significant figures to keep -/
def significant_figures : ℕ := 3

/-- Function to convert a natural number to scientific notation with given significant figures -/
noncomputable def to_scientific_notation (n : ℕ) (sig_figs : ℕ) : ℝ × ℤ := sorry

/-- Theorem stating that the scientific notation of 1,010,659 with three significant figures is 1.01 × 10^6 -/
theorem scientific_notation_of_1010659 :
  to_scientific_notation original_number significant_figures = (1.01, 6) := by sorry

end scientific_notation_of_1010659_l3993_399316


namespace equation_solution_l3993_399327

theorem equation_solution (x : ℝ) :
  (x / 3) / 3 = 9 / (x / 3) → x = 3^(5/2) ∨ x = -(3^(5/2)) :=
by sorry

end equation_solution_l3993_399327


namespace partnership_profit_l3993_399345

/-- Represents a partnership between two individuals -/
structure Partnership where
  investment_ratio : ℕ  -- Ratio of investments (larger : smaller)
  time_ratio : ℕ        -- Ratio of investment periods (longer : shorter)
  smaller_profit : ℕ    -- Profit of the partner with smaller investment

/-- Calculates the total profit of a partnership -/
def total_profit (p : Partnership) : ℕ :=
  let profit_ratio := p.investment_ratio * p.time_ratio + 1
  profit_ratio * p.smaller_profit

/-- Theorem: For a partnership where one partner's investment is triple and 
    investment period is double that of the other, if the partner with 
    smaller investment receives 7000, the total profit is 49000 -/
theorem partnership_profit : 
  ∀ (p : Partnership), 
    p.investment_ratio = 3 → 
    p.time_ratio = 2 → 
    p.smaller_profit = 7000 → 
    total_profit p = 49000 := by
  sorry

end partnership_profit_l3993_399345


namespace hyperbola_eccentricity_l3993_399397

/-- The eccentricity of the hyperbola x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2 = 1
  ∃ e : ℝ, e = Real.sqrt 2 ∧ ∀ x y : ℝ, h x y → e = (Real.sqrt (x^2 + y^2)) / (Real.sqrt (x^2 - 1)) :=
by sorry

end hyperbola_eccentricity_l3993_399397


namespace max_value_of_k_l3993_399310

theorem max_value_of_k : ∃ (k : ℝ), k = Real.sqrt 10 ∧ 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 7 → Real.sqrt (x - 2) + Real.sqrt (7 - x) ≤ k) ∧
  (∀ ε > 0, ∃ x : ℝ, 2 ≤ x ∧ x ≤ 7 ∧ Real.sqrt (x - 2) + Real.sqrt (7 - x) > k - ε) := by
  sorry

end max_value_of_k_l3993_399310


namespace locus_of_point_P_l3993_399354

/-- The locus of point P where a moving circle M with diameter PF₁ is tangent internally 
    to a fixed circle C -/
theorem locus_of_point_P (n m : ℝ) (h_positive : 0 < n ∧ n < m) :
  ∃ (locus : ℝ × ℝ → Prop),
    (∀ (P : ℝ × ℝ), locus P ↔ 
      (P.1^2 / m^2 + P.2^2 / (m^2 - n^2) = 1)) ∧
    (∀ (P : ℝ × ℝ), locus P → 
      ∃ (M : ℝ × ℝ),
        -- M is the center of the moving circle
        M = ((P.1 - (-n)) / 2, P.2 / 2) ∧
        -- M is internally tangent to the fixed circle C
        ((M.1^2 + M.2^2)^(1/2) + ((M.1 - (-n))^2 + M.2^2)^(1/2) = m) ∧
        -- PF₁ is a diameter of the moving circle
        (P.1 - (-n))^2 + P.2^2 = (2 * ((M.1 - (-n))^2 + M.2^2)^(1/2))^2) :=
by sorry

end locus_of_point_P_l3993_399354


namespace smallest_multiple_45_60_not_25_l3993_399399

theorem smallest_multiple_45_60_not_25 : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (45 ∣ n) ∧ 
  (60 ∣ n) ∧ 
  ¬(25 ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (45 ∣ m) ∧ (60 ∣ m) ∧ ¬(25 ∣ m) → n ≤ m) ∧
  n = 180 := by
  sorry

end smallest_multiple_45_60_not_25_l3993_399399


namespace complement_union_equals_ge_one_l3993_399331

open Set

def M : Set ℝ := {x | (x + 3) / (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

theorem complement_union_equals_ge_one : 
  (M ∪ N)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end complement_union_equals_ge_one_l3993_399331


namespace order_of_cube_roots_l3993_399393

theorem order_of_cube_roots (a : ℝ) (x y z : ℝ) 
  (hx : x = (1 + 991 * a) ^ (1/3))
  (hy : y = (1 + 992 * a) ^ (1/3))
  (hz : z = (1 + 993 * a) ^ (1/3))
  (ha : a ≤ 0) : 
  z ≤ y ∧ y ≤ x := by
sorry

end order_of_cube_roots_l3993_399393


namespace smallest_divisible_by_6_and_35_after_2015_l3993_399388

theorem smallest_divisible_by_6_and_35_after_2015 :
  ∀ n : ℕ, n > 2015 ∧ 6 ∣ n ∧ 35 ∣ n → n ≥ 2100 :=
by
  sorry

end smallest_divisible_by_6_and_35_after_2015_l3993_399388


namespace cube_sum_eq_triple_product_l3993_399370

theorem cube_sum_eq_triple_product (a b c : ℝ) (h : a + b + c = 0) :
  a^3 + b^3 + c^3 = 3*a*b*c := by sorry

end cube_sum_eq_triple_product_l3993_399370


namespace divisibility_by_24_l3993_399337

theorem divisibility_by_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := by sorry

end divisibility_by_24_l3993_399337


namespace max_player_salary_l3993_399302

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (max_team_salary : ℕ) :
  num_players = 23 →
  min_salary = 17000 →
  max_team_salary = 800000 →
  ∃ (max_single_salary : ℕ),
    max_single_salary = 426000 ∧
    (num_players - 1) * min_salary + max_single_salary = max_team_salary ∧
    ∀ (alternative_salary : ℕ),
      (num_players - 1) * min_salary + alternative_salary ≤ max_team_salary →
      alternative_salary ≤ max_single_salary :=
by
  sorry

end max_player_salary_l3993_399302


namespace second_term_of_geometric_sequence_l3993_399350

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = (a n : ℚ) * r

theorem second_term_of_geometric_sequence
    (a : ℕ → ℕ)
    (is_geometric : IsGeometricSequence a)
    (first_term : a 1 = 5)
    (fifth_term : a 5 = 320) :
  a 2 = 10 := by
  sorry

end second_term_of_geometric_sequence_l3993_399350


namespace det_A_equals_two_l3993_399347

theorem det_A_equals_two (a d : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = !![a, -2; 1, d] →
  A + 2 * A⁻¹ = 0 →
  Matrix.det A = 2 := by
sorry

end det_A_equals_two_l3993_399347


namespace real_part_of_z_l3993_399375

theorem real_part_of_z (z : ℂ) (h1 : Complex.abs (z - 1) = 2) (h2 : Complex.abs (z^2 - 1) = 6) :
  z.re = 5/4 := by sorry

end real_part_of_z_l3993_399375


namespace quadratic_inequality_solution_l3993_399389

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 + a*x + b < 0 ↔ 2 < x ∧ x < 4) → b - a = 14 := by
  sorry

end quadratic_inequality_solution_l3993_399389


namespace equation_solution_l3993_399366

theorem equation_solution :
  ∃ x : ℚ, (2 * x + 5 * x = 500 - (4 * x + 6 * x + 10)) ∧ x = 490 / 17 := by
  sorry

end equation_solution_l3993_399366


namespace complex_fraction_evaluation_l3993_399383

theorem complex_fraction_evaluation :
  2 + (3 / (2 + (1 / (2 + (1/2))))) = 13/4 := by
  sorry

end complex_fraction_evaluation_l3993_399383


namespace bob_orders_12_muffins_l3993_399309

/-- The number of muffins Bob orders per day -/
def muffins_per_day : ℕ := sorry

/-- The cost price of each muffin in cents -/
def cost_price : ℕ := 75

/-- The selling price of each muffin in cents -/
def selling_price : ℕ := 150

/-- The profit Bob makes per week in cents -/
def weekly_profit : ℕ := 6300

/-- Theorem stating that Bob orders 12 muffins per day -/
theorem bob_orders_12_muffins : muffins_per_day = 12 := by
  sorry

end bob_orders_12_muffins_l3993_399309


namespace binomial_coefficient_two_l3993_399396

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l3993_399396


namespace quartic_sum_at_3_and_neg_3_l3993_399318

def quartic_polynomial (d a b c m : ℝ) (x : ℝ) : ℝ :=
  d * x^4 + a * x^3 + b * x^2 + c * x + m

theorem quartic_sum_at_3_and_neg_3 
  (d a b c m : ℝ) 
  (h1 : quartic_polynomial d a b c m 0 = m)
  (h2 : quartic_polynomial d a b c m 1 = 3 * m)
  (h3 : quartic_polynomial d a b c m (-1) = 4 * m) :
  quartic_polynomial d a b c m 3 + quartic_polynomial d a b c m (-3) = 144 * d + 47 * m := by
  sorry

end quartic_sum_at_3_and_neg_3_l3993_399318


namespace expression_undefined_iff_x_eq_11_l3993_399341

theorem expression_undefined_iff_x_eq_11 (x : ℝ) :
  ¬ (∃ y : ℝ, y = (3 * x^3 + 4) / (x^2 - 22*x + 121)) ↔ x = 11 := by
  sorry

end expression_undefined_iff_x_eq_11_l3993_399341


namespace fidos_yard_exploration_l3993_399361

theorem fidos_yard_exploration (s : ℝ) (s_pos : s > 0) :
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let circle_area := π * s^2
  let ratio := circle_area / hexagon_area
  ∃ (a b : ℕ), (a > 0 ∧ b > 0) ∧ 
    ratio = Real.sqrt a / b * π ∧
    a * b = 27 := by
  sorry

end fidos_yard_exploration_l3993_399361


namespace polynomial_roots_l3993_399343

theorem polynomial_roots : 
  let p (x : ℝ) := x^4 - 3*x^3 + 3*x^2 - x - 6
  ∀ x : ℝ, p x = 0 ↔ x = 3 ∨ x = 2 ∨ x = -1 := by
sorry

end polynomial_roots_l3993_399343


namespace pencil_distribution_problem_l3993_399379

theorem pencil_distribution_problem :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 6 ∣ n ∧ 9 ∣ n ∧ n % 7 = 1 ∧ n = 36 :=
by sorry

end pencil_distribution_problem_l3993_399379


namespace area_ratio_for_specific_trapezoid_l3993_399338

/-- Represents a trapezoid with extended legs -/
structure ExtendedTrapezoid where
  PQ : ℝ  -- Length of base PQ
  RS : ℝ  -- Length of base RS
  -- Assume other necessary properties of a trapezoid

/-- The ratio of the area of triangle TPQ to the area of trapezoid PQRS -/
def area_ratio (t : ExtendedTrapezoid) : ℚ :=
  100 / 341

/-- Theorem stating the area ratio for the given trapezoid -/
theorem area_ratio_for_specific_trapezoid :
  ∃ t : ExtendedTrapezoid, t.PQ = 10 ∧ t.RS = 21 ∧ area_ratio t = 100 / 341 := by
  sorry

end area_ratio_for_specific_trapezoid_l3993_399338


namespace max_profit_theorem_l3993_399394

/-- Represents the daily production and profit of an eco-friendly bag factory --/
structure BagFactory where
  totalBags : ℕ
  costA : ℚ
  sellA : ℚ
  costB : ℚ
  sellB : ℚ
  maxInvestment : ℚ

/-- Calculates the profit function for the bag factory --/
def profitFunction (factory : BagFactory) (x : ℚ) : ℚ :=
  (factory.sellA - factory.costA) * x + (factory.sellB - factory.costB) * (factory.totalBags - x)

/-- Theorem stating the maximum profit of the bag factory --/
theorem max_profit_theorem (factory : BagFactory) 
    (h1 : factory.totalBags = 4500)
    (h2 : factory.costA = 2)
    (h3 : factory.sellA = 2.3)
    (h4 : factory.costB = 3)
    (h5 : factory.sellB = 3.5)
    (h6 : factory.maxInvestment = 10000) :
    ∃ x : ℚ, x ≥ 0 ∧ x ≤ factory.totalBags ∧
    factory.costA * x + factory.costB * (factory.totalBags - x) ≤ factory.maxInvestment ∧
    ∀ y : ℚ, y ≥ 0 → y ≤ factory.totalBags →
    factory.costA * y + factory.costB * (factory.totalBags - y) ≤ factory.maxInvestment →
    profitFunction factory x ≥ profitFunction factory y ∧
    profitFunction factory x = 1550 := by
  sorry


end max_profit_theorem_l3993_399394


namespace last_item_to_second_recipient_l3993_399371

/-- Represents the cyclic distribution of items among recipients. -/
def cyclicDistribution (items : ℕ) (recipients : ℕ) : ℕ :=
  (items - 1) % recipients + 1

/-- Theorem stating that in a cyclic distribution of 278 items among 6 recipients,
    the 2nd recipient in the initial order receives the last item. -/
theorem last_item_to_second_recipient :
  cyclicDistribution 278 6 = 2 := by
  sorry

end last_item_to_second_recipient_l3993_399371


namespace ratio_of_areas_ratio_of_perimeters_l3993_399336

-- Define the side lengths of squares A and B
def side_length_A : ℝ := 48
def side_length_B : ℝ := 60

-- Define the areas of squares A and B
def area_A : ℝ := side_length_A ^ 2
def area_B : ℝ := side_length_B ^ 2

-- Define the perimeters of squares A and B
def perimeter_A : ℝ := 4 * side_length_A
def perimeter_B : ℝ := 4 * side_length_B

-- Theorem stating the ratio of areas
theorem ratio_of_areas :
  area_A / area_B = 16 / 25 := by sorry

-- Theorem stating the ratio of perimeters
theorem ratio_of_perimeters :
  perimeter_A / perimeter_B = 4 / 5 := by sorry

end ratio_of_areas_ratio_of_perimeters_l3993_399336


namespace range_of_f_l3993_399342

-- Define the function f
def f (x : ℝ) : ℝ := 2*x - x^2

-- Define the domain
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem range_of_f :
  ∃ (y : ℝ), (∃ (x : ℝ), domain x ∧ f x = y) ↔ -3 ≤ y ∧ y ≤ 1 :=
sorry

end range_of_f_l3993_399342


namespace trig_identity_l3993_399363

theorem trig_identity : 
  (Real.tan (7.5 * π / 180) * Real.tan (15 * π / 180)) / 
    (Real.tan (15 * π / 180) - Real.tan (7.5 * π / 180)) + 
  Real.sqrt 3 * (Real.sin (7.5 * π / 180)^2 - Real.cos (7.5 * π / 180)^2) = 
  -Real.sqrt 2 := by sorry

end trig_identity_l3993_399363


namespace jake_peaches_l3993_399340

/-- The number of peaches Jill has -/
def jill_peaches : ℕ := 5

/-- The number of peaches Steven has more than Jill -/
def steven_more_than_jill : ℕ := 18

/-- The number of peaches Jake has fewer than Steven -/
def jake_fewer_than_steven : ℕ := 6

/-- Theorem: Jake has 17 peaches -/
theorem jake_peaches : 
  jill_peaches + steven_more_than_jill - jake_fewer_than_steven = 17 := by
  sorry

end jake_peaches_l3993_399340


namespace tommy_initial_balloons_l3993_399358

/-- The number of balloons Tommy's mom gave him -/
def balloons_given : ℝ := 34.5

/-- The total number of balloons Tommy had after receiving more -/
def total_balloons : ℝ := 60.75

/-- The number of balloons Tommy had initially -/
def initial_balloons : ℝ := total_balloons - balloons_given

theorem tommy_initial_balloons :
  initial_balloons = 26.25 := by sorry

end tommy_initial_balloons_l3993_399358


namespace tv_price_change_l3993_399334

theorem tv_price_change (P : ℝ) : 
  P > 0 → (P * 0.8 * 1.4) = P * 1.12 := by
  sorry

end tv_price_change_l3993_399334


namespace rectangular_prism_parallel_edges_l3993_399344

/-- A rectangular prism with specific proportions -/
structure RectangularPrism where
  width : ℝ
  length : ℝ
  height : ℝ
  length_eq : length = 2 * width
  height_eq : height = 3 * width

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 8

/-- Theorem stating that a rectangular prism with the given proportions has 8 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) : 
  parallel_edge_pairs prism = 8 := by
  sorry

end rectangular_prism_parallel_edges_l3993_399344


namespace wire_problem_l3993_399365

theorem wire_problem (total_length : ℝ) (num_parts : ℕ) (used_parts : ℕ) : 
  total_length = 50 ∧ 
  num_parts = 5 ∧ 
  used_parts = 3 → 
  total_length - (total_length / num_parts) * used_parts = 20 := by
sorry

end wire_problem_l3993_399365


namespace set_a_range_l3993_399320

theorem set_a_range (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 2*x + a > 0}
  1 ∉ A → a ≥ 1 := by
sorry

end set_a_range_l3993_399320


namespace function_range_l3993_399367

def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem function_range :
  {y | ∃ x ∈ Set.Ioo (-1) 2, f x = y} = Set.Icc (-4) 0 := by sorry

end function_range_l3993_399367


namespace oil_quantity_function_correct_l3993_399330

/-- Represents the remaining oil quantity in liters at time t in minutes -/
def Q (t : ℝ) : ℝ := 40 - 0.2 * t

/-- The initial oil quantity in liters -/
def initial_quantity : ℝ := 40

/-- The outflow rate in liters per minute -/
def outflow_rate : ℝ := 0.2

theorem oil_quantity_function_correct :
  ∀ t : ℝ, t ≥ 0 →
    Q t = initial_quantity - outflow_rate * t ∧
    Q 0 = initial_quantity ∧
    ∀ t₁ t₂ : ℝ, t₁ < t₂ → Q t₂ < Q t₁ := by
  sorry

end oil_quantity_function_correct_l3993_399330


namespace max_whole_nine_one_number_l3993_399328

def is_nine_one_number (t : ℕ) : Prop :=
  t ≥ 1000 ∧ t ≤ 9999 ∧
  (t / 1000 + (t / 10) % 10 = 9) ∧
  ((t / 100) % 10 - t % 10 = 1)

def P (t : ℕ) : ℕ := 2 * (t / 1000) + (t % 10)

def Q (t : ℕ) : ℕ := 2 * ((t / 100) % 10) + ((t / 10) % 10)

def G (t : ℕ) : ℚ := 2 * (P t : ℚ) / (Q t : ℚ)

def is_whole_nine_one_number (t : ℕ) : Prop :=
  is_nine_one_number t ∧ (G t).isInt

theorem max_whole_nine_one_number :
  ∃ M : ℕ,
    is_whole_nine_one_number M ∧
    ∀ t : ℕ, is_whole_nine_one_number t → t ≤ M ∧
    M = 7524 :=
sorry

end max_whole_nine_one_number_l3993_399328


namespace ladder_height_correct_l3993_399306

/-- The height of the ceiling in centimeters -/
def ceiling_height : ℝ := 300

/-- The distance of the light fixture below the ceiling in centimeters -/
def fixture_below_ceiling : ℝ := 15

/-- Bob's height in centimeters -/
def bob_height : ℝ := 170

/-- The distance Bob can reach above his head in centimeters -/
def bob_reach : ℝ := 52

/-- The height of the ladder in centimeters -/
def ladder_height : ℝ := 63

theorem ladder_height_correct :
  ceiling_height - fixture_below_ceiling = bob_height + bob_reach + ladder_height := by
  sorry

end ladder_height_correct_l3993_399306


namespace parallel_line_through_point_A_l3993_399325

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 1)

-- Define the parallel line passing through point A
def parallel_line (x y : ℝ) : Prop := 2 * x - y - 3 = 0

theorem parallel_line_through_point_A :
  (parallel_line point_A.1 point_A.2) ∧
  (∀ (x y : ℝ), parallel_line x y → given_line x y → x = y) ∧
  (∃ (m b : ℝ), ∀ (x y : ℝ), parallel_line x y ↔ y = m * x + b) :=
sorry

end parallel_line_through_point_A_l3993_399325


namespace potato_bag_weights_l3993_399326

/-- Represents the weights of three bags of potatoes -/
structure BagWeights where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Calculates the new weights after adjustments -/
def adjustedWeights (w : BagWeights) : BagWeights :=
  { A := w.A - 0.1 * w.C
  , B := w.B + 0.15 * w.A
  , C := w.C }

/-- Theorem stating the result of the potato bag weight problem -/
theorem potato_bag_weights :
  ∀ w : BagWeights,
    w.A = 12 + 1/2 * w.B →
    w.B = 8 + 1/3 * w.C →
    w.C = 20 + 2 * w.A →
    let new_w := adjustedWeights w
    (new_w.A + new_w.B + new_w.C) = 139.55 := by
  sorry


end potato_bag_weights_l3993_399326


namespace max_area_rectangle_in_ellipse_l3993_399384

/-- Given an ellipse b² x² + a² y² = a² b², prove that the rectangle with the largest possible area
    inscribed in the ellipse has vertices at (±(a/2)√2, ±(b/2)√2) -/
theorem max_area_rectangle_in_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let ellipse := {p : ℝ × ℝ | b^2 * p.1^2 + a^2 * p.2^2 = a^2 * b^2}
  let inscribed_rectangle (p : ℝ × ℝ) := 
    {q : ℝ × ℝ | q ∈ ellipse ∧ |q.1| ≤ |p.1| ∧ |q.2| ≤ |p.2|}
  let area (p : ℝ × ℝ) := 4 * |p.1 * p.2|
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧
    (∀ q : ℝ × ℝ, q ∈ ellipse → area q ≤ area p) ∧
    p = (a/2 * Real.sqrt 2, b/2 * Real.sqrt 2) :=
by sorry

end max_area_rectangle_in_ellipse_l3993_399384


namespace heptagon_angle_sum_l3993_399300

/-- A polygon with vertices A, B, C, D, E, F, G -/
structure Heptagon :=
  (A B C D E F G : ℝ × ℝ)

/-- The angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The sum of angles FAD, GBC, BCE, ADG, CEF, AFE, DGB -/
def angle_sum (h : Heptagon) : ℝ :=
  angle h.F h.A h.D +
  angle h.G h.B h.C +
  angle h.B h.C h.E +
  angle h.A h.D h.G +
  angle h.C h.E h.F +
  angle h.A h.F h.E +
  angle h.D h.G h.B

theorem heptagon_angle_sum (h : Heptagon) : angle_sum h = 540 := by sorry

end heptagon_angle_sum_l3993_399300


namespace union_of_sets_l3993_399323

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4, 5}
  A ∪ B = {1, 2, 4, 5} := by sorry

end union_of_sets_l3993_399323


namespace trigonometric_simplification_l3993_399378

theorem trigonometric_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.cos y * Real.sin (x + y) = Real.cos y ^ 2 := by
  sorry

end trigonometric_simplification_l3993_399378


namespace repeating_decimal_denominator_l3993_399368

theorem repeating_decimal_denominator : ∃ (n d : ℕ), d > 0 ∧ (n / d : ℚ) = 2 / 3 ∧ 
  (∀ (n' d' : ℕ), d' > 0 → (n' / d' : ℚ) = 2 / 3 → d ≤ d') := by
  sorry

end repeating_decimal_denominator_l3993_399368


namespace race_length_proof_l3993_399311

/-- The length of the race in metres -/
def race_length : ℝ := 200

/-- The fraction of the race completed -/
def fraction_completed : ℝ := 0.25

/-- The distance run so far in metres -/
def distance_run : ℝ := 50

theorem race_length_proof : 
  fraction_completed * race_length = distance_run :=
by sorry

end race_length_proof_l3993_399311


namespace remainder_theorem_l3993_399381

theorem remainder_theorem (x : ℕ+) (h : (7 * x.val) % 29 = 1) :
  (13 + x.val) % 29 = 9 := by
  sorry

end remainder_theorem_l3993_399381


namespace irreducible_fractions_exist_l3993_399398

theorem irreducible_fractions_exist : ∃ (a b : ℕ), 
  Nat.gcd a b = 1 ∧ Nat.gcd (a + 1) b = 1 ∧ Nat.gcd (a + 1) (b + 1) = 1 := by
  sorry

end irreducible_fractions_exist_l3993_399398


namespace complex_arithmetic_equality_l3993_399357

theorem complex_arithmetic_equality : (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) + 2 * Complex.I * (3 - 5 * Complex.I) = 11 - 9 * Complex.I :=
by sorry

end complex_arithmetic_equality_l3993_399357


namespace complex_modulus_range_l3993_399391

theorem complex_modulus_range (a : ℝ) : 
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) ↔ 
  a ∈ Set.Icc (-(Real.sqrt 5 / 5)) (Real.sqrt 5 / 5) := by
sorry

end complex_modulus_range_l3993_399391


namespace emiliano_consumption_theorem_l3993_399369

/-- Represents the number of fruits in a basket -/
structure FruitBasket where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Calculates the number of fruits Emiliano consumes -/
def emilianoConsumption (basket : FruitBasket) : ℕ :=
  (3 * basket.apples / 5) + (2 * basket.oranges / 3) + (4 * basket.bananas / 7)

/-- Theorem: Given the conditions, Emiliano consumes 16 fruits -/
theorem emiliano_consumption_theorem (basket : FruitBasket) 
  (h1 : basket.apples = 15)
  (h2 : basket.apples = 4 * basket.oranges)
  (h3 : basket.bananas = 3 * basket.oranges) :
  emilianoConsumption basket = 16 := by
  sorry


end emiliano_consumption_theorem_l3993_399369


namespace janets_class_size_l3993_399390

/-- The number of children in Janet's class -/
def num_children : ℕ := 35

/-- The number of chaperones -/
def num_chaperones : ℕ := 5

/-- The number of additional lunches -/
def additional_lunches : ℕ := 3

/-- The cost of each lunch in dollars -/
def lunch_cost : ℕ := 7

/-- The total cost of all lunches in dollars -/
def total_cost : ℕ := 308

theorem janets_class_size :
  num_children + num_chaperones + 1 + additional_lunches = total_cost / lunch_cost :=
sorry

end janets_class_size_l3993_399390


namespace vacation_cost_l3993_399317

theorem vacation_cost (C : ℝ) : C / 3 - C / 4 = 40 → C = 480 := by
  sorry

end vacation_cost_l3993_399317


namespace tangent_line_at_2_range_of_m_l3993_399333

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2 :
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
  (∀ x y, y = f x → (x = 2 → A * x + B * y + C = 0)) ∧
  A = 12 ∧ B = -1 ∧ C = -17 :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
sorry

end tangent_line_at_2_range_of_m_l3993_399333


namespace houses_in_block_l3993_399351

/-- Given a block where each house receives 32 pieces of junk mail and
    the entire block receives 640 pieces of junk mail, prove that
    there are 20 houses in the block. -/
theorem houses_in_block (mail_per_house : ℕ) (mail_per_block : ℕ)
    (h1 : mail_per_house = 32)
    (h2 : mail_per_block = 640) :
    mail_per_block / mail_per_house = 20 := by
  sorry

end houses_in_block_l3993_399351


namespace line_points_relation_l3993_399376

/-- Given a line in the xy-coordinate system with equation x = 2y + 5,
    if (m, n) and (m + 1, n + k) are two points on this line,
    then k = 1/2 -/
theorem line_points_relation (m n k : ℝ) : 
  (m = 2 * n + 5) →  -- (m, n) is on the line
  (m + 1 = 2 * (n + k) + 5) →  -- (m + 1, n + k) is on the line
  k = 1 / 2 := by
  sorry

end line_points_relation_l3993_399376


namespace simplify_expression_l3993_399321

variable (a b : ℝ)

theorem simplify_expression (hb : b ≠ 0) :
  6 * a^5 * b^2 / (3 * a^3 * b^2) + (2 * a * b^3)^2 / (-b^2)^3 = -2 * a^2 := by
  sorry

end simplify_expression_l3993_399321


namespace boat_journey_l3993_399307

-- Define the given constants
def total_time : ℝ := 19
def stream_velocity : ℝ := 4
def boat_speed : ℝ := 14

-- Define the distance between A and B
def distance_AB : ℝ := 122.14

-- Theorem statement
theorem boat_journey :
  let downstream_speed := boat_speed + stream_velocity
  let upstream_speed := boat_speed - stream_velocity
  total_time = distance_AB / downstream_speed + (distance_AB / 2) / upstream_speed :=
by
  sorry

#check boat_journey

end boat_journey_l3993_399307


namespace line_slope_and_angle_l3993_399360

/-- Theorem: For a line passing through points (-2,3) and (-1,2), its slope is -1
    and the angle it makes with the positive x-axis is 3π/4 -/
theorem line_slope_and_angle :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (-1, 2)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let angle : ℝ := Real.arctan slope
  slope = -1 ∧ angle = 3 * π / 4 := by
  sorry

end line_slope_and_angle_l3993_399360


namespace ball_max_height_l3993_399335

/-- The height function of the ball -/
def f (t : ℝ) : ℝ := -16 * t^2 + 96 * t + 15

/-- Theorem stating that the maximum height of the ball is 159 feet -/
theorem ball_max_height :
  ∃ t_max : ℝ, ∀ t : ℝ, f t ≤ f t_max ∧ f t_max = 159 := by
  sorry

end ball_max_height_l3993_399335


namespace suzy_age_l3993_399387

theorem suzy_age (mary_age : ℕ) (suzy_age : ℕ) : 
  mary_age = 8 → 
  suzy_age + 4 = 2 * (mary_age + 4) → 
  suzy_age = 20 := by
sorry

end suzy_age_l3993_399387


namespace f_properties_l3993_399308

noncomputable def f (x : ℝ) : ℝ := (2^x) / (4^x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f x ≤ (1/2 : ℝ)) ∧
  (∃ x : ℝ, f x = (1/2 : ℝ)) :=
by sorry

end f_properties_l3993_399308


namespace lines_perpendicular_to_plane_are_parallel_l3993_399312

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop := sorry

theorem lines_perpendicular_to_plane_are_parallel 
  (l1 l2 : Line3D) (p : Plane3D) :
  perpendicular l1 p → perpendicular l2 p → parallel l1 l2 := by
  sorry

end lines_perpendicular_to_plane_are_parallel_l3993_399312


namespace sin_90_degrees_l3993_399392

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end sin_90_degrees_l3993_399392


namespace sum_of_x_and_y_l3993_399380

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 10*x - 6*y - 34) : x + y = 2 := by
  sorry

end sum_of_x_and_y_l3993_399380


namespace extended_volume_of_specific_box_l3993_399362

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def extendedVolume (b : Box) : ℝ :=
  sorry

/-- The main theorem -/
theorem extended_volume_of_specific_box :
  let box : Box := { length := 2, width := 3, height := 6 }
  extendedVolume box = (324 + 37 * Real.pi) / 3 := by
  sorry

end extended_volume_of_specific_box_l3993_399362


namespace quadratic_roots_integer_parts_l3993_399332

theorem quadratic_roots_integer_parts (n : ℕ) (h : n ≥ 1) :
  let original_eq := fun x : ℝ => x^2 + (2*n + 1)*x + 6*n - 5
  let result_eq := fun x : ℝ => x^2 + 2*(n + 1)*x + 8*(n - 1)
  ∃ (r₁ r₂ : ℝ), original_eq r₁ = 0 ∧ original_eq r₂ = 0 ∧
    result_eq (⌊r₁⌋) = 0 ∧ result_eq (⌊r₂⌋) = 0 :=
by sorry

end quadratic_roots_integer_parts_l3993_399332


namespace BC_equals_2AB_l3993_399353

def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (4, 3)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

theorem BC_equals_2AB : vector_BC = (2 * vector_AB.1, 2 * vector_AB.2) := by
  sorry

end BC_equals_2AB_l3993_399353


namespace rhombicosidodecahedron_symmetries_l3993_399322

/-- Represents a rhombicosidodecahedron -/
structure Rhombicosidodecahedron where
  triangular_faces : ℕ
  square_faces : ℕ
  pentagonal_faces : ℕ
  is_archimedean : Prop
  is_convex : Prop
  is_isogonal : Prop
  is_nonprismatic : Prop

/-- The number of rotational symmetries of a rhombicosidodecahedron -/
def rotational_symmetries (r : Rhombicosidodecahedron) : ℕ := 60

/-- Theorem stating that a rhombicosidodecahedron has 60 rotational symmetries -/
theorem rhombicosidodecahedron_symmetries (r : Rhombicosidodecahedron) 
  (h1 : r.triangular_faces = 20)
  (h2 : r.square_faces = 30)
  (h3 : r.pentagonal_faces = 12)
  (h4 : r.is_archimedean)
  (h5 : r.is_convex)
  (h6 : r.is_isogonal)
  (h7 : r.is_nonprismatic) :
  rotational_symmetries r = 60 := by
  sorry

end rhombicosidodecahedron_symmetries_l3993_399322


namespace ethanol_percentage_in_fuel_B_l3993_399359

/-- Proves that the percentage of ethanol in fuel B is 16% given the problem conditions -/
theorem ethanol_percentage_in_fuel_B (tank_capacity : ℝ) (ethanol_A : ℝ) (total_ethanol : ℝ) (fuel_A_volume : ℝ) : 
  tank_capacity = 200 →
  ethanol_A = 0.12 →
  total_ethanol = 28 →
  fuel_A_volume = 99.99999999999999 →
  (total_ethanol - ethanol_A * fuel_A_volume) / (tank_capacity - fuel_A_volume) = 0.16 := by
sorry

#eval (28 - 0.12 * 99.99999999999999) / (200 - 99.99999999999999)

end ethanol_percentage_in_fuel_B_l3993_399359


namespace product_is_even_l3993_399395

def pi_digits : Finset ℕ := {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4}

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem product_is_even (a : Fin 24 → ℕ) (h : ∀ i, a i ∈ pi_digits) :
  is_even ((a 0 - a 1) * (a 2 - a 3) * (a 4 - a 5) * (a 6 - a 7) * (a 8 - a 9) * (a 10 - a 11) *
           (a 12 - a 13) * (a 14 - a 15) * (a 16 - a 17) * (a 18 - a 19) * (a 20 - a 21) * (a 22 - a 23)) :=
by sorry

end product_is_even_l3993_399395


namespace exp_greater_than_power_e_l3993_399304

theorem exp_greater_than_power_e (x : ℝ) (h1 : x > 0) (h2 : x ≠ ℯ) : ℯ^x > x^ℯ := by
  sorry

end exp_greater_than_power_e_l3993_399304


namespace inscribed_square_side_length_l3993_399349

/-- A right triangle with an inscribed square -/
structure RightTriangleWithSquare where
  /-- Length of the first leg -/
  leg1 : ℝ
  /-- Length of the second leg -/
  leg2 : ℝ
  /-- Length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The triangle is right-angled -/
  right_angle : leg1 ^ 2 + leg2 ^ 2 = hypotenuse ^ 2
  /-- All sides are positive -/
  leg1_pos : leg1 > 0
  leg2_pos : leg2 > 0
  hypotenuse_pos : hypotenuse > 0
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed in the triangle -/
  inscribed : square_side > 0 ∧ square_side < leg1 ∧ square_side < leg2

/-- The side length of the inscribed square in the given right triangle is 12/5 -/
theorem inscribed_square_side_length (t : RightTriangleWithSquare) 
    (h1 : t.leg1 = 5) (h2 : t.leg2 = 12) (h3 : t.hypotenuse = 13) : 
    t.square_side = 12/5 := by
  sorry

end inscribed_square_side_length_l3993_399349


namespace laura_debt_l3993_399346

/-- Calculates the total amount owed after one year given a principal amount,
    an annual interest rate, and assuming simple interest. -/
def totalAmountOwed (principal : ℝ) (interestRate : ℝ) : ℝ :=
  principal * (1 + interestRate)

/-- Proves that given a principal of $35 and an interest rate of 9%,
    the total amount owed after one year is $38.15. -/
theorem laura_debt : totalAmountOwed 35 0.09 = 38.15 := by
  sorry

end laura_debt_l3993_399346


namespace kolya_is_wrong_l3993_399352

/-- Represents a statement about the number of pencils. -/
structure PencilStatement where
  blue : ℕ
  green : ℕ

/-- The box of colored pencils. -/
def pencil_box : PencilStatement := sorry

/-- Vasya's statement -/
def vasya_statement (box : PencilStatement) : Prop :=
  box.blue ≥ 4

/-- Kolya's statement -/
def kolya_statement (box : PencilStatement) : Prop :=
  box.green ≥ 5

/-- Petya's statement -/
def petya_statement (box : PencilStatement) : Prop :=
  box.blue ≥ 3 ∧ box.green ≥ 4

/-- Misha's statement -/
def misha_statement (box : PencilStatement) : Prop :=
  box.blue ≥ 4 ∧ box.green ≥ 4

/-- Three statements are true and one is false -/
axiom three_true_one_false :
  (vasya_statement pencil_box ∧ petya_statement pencil_box ∧ misha_statement pencil_box ∧ ¬kolya_statement pencil_box) ∨
  (vasya_statement pencil_box ∧ petya_statement pencil_box ∧ ¬misha_statement pencil_box ∧ kolya_statement pencil_box) ∨
  (vasya_statement pencil_box ∧ ¬petya_statement pencil_box ∧ misha_statement pencil_box ∧ kolya_statement pencil_box) ∨
  (¬vasya_statement pencil_box ∧ petya_statement pencil_box ∧ misha_statement pencil_box ∧ kolya_statement pencil_box)

theorem kolya_is_wrong :
  ¬kolya_statement pencil_box ∧
  vasya_statement pencil_box ∧
  petya_statement pencil_box ∧
  misha_statement pencil_box :=
by sorry

end kolya_is_wrong_l3993_399352


namespace balls_to_one_pile_l3993_399373

/-- Represents a configuration of piles of balls -/
structure BallConfiguration (n : ℕ) where
  piles : List ℕ
  sum_balls : List.sum piles = 2^n

/-- Represents a move between two piles -/
inductive Move (n : ℕ)
| move : (a b : ℕ) → a ≥ b → a + b ≤ 2^n → Move n

/-- Represents a sequence of moves -/
def MoveSequence (n : ℕ) := List (Move n)

/-- Applies a move to a configuration -/
def applyMove (config : BallConfiguration n) (m : Move n) : BallConfiguration n :=
  sorry

/-- Applies a sequence of moves to a configuration -/
def applyMoveSequence (config : BallConfiguration n) (seq : MoveSequence n) : BallConfiguration n :=
  sorry

/-- Checks if all balls are in one pile -/
def isOnePile (config : BallConfiguration n) : Prop :=
  ∃ p, config.piles = [p]

/-- The main theorem to prove -/
theorem balls_to_one_pile (n : ℕ) (initial : BallConfiguration n) :
  ∃ (seq : MoveSequence n), isOnePile (applyMoveSequence initial seq) :=
sorry

end balls_to_one_pile_l3993_399373


namespace interest_rate_calculation_l3993_399313

/-- The interest rate at which B lent to C, given the conditions of the problem -/
def interest_rate_B_to_C (principal : ℚ) (rate_A_to_B : ℚ) (years : ℚ) (gain_B : ℚ) : ℚ :=
  let interest_A_to_B := principal * rate_A_to_B * years
  let total_interest_B_from_C := interest_A_to_B + gain_B
  (total_interest_B_from_C * 100) / (principal * years)

theorem interest_rate_calculation (principal : ℚ) (rate_A_to_B : ℚ) (years : ℚ) (gain_B : ℚ) :
  principal = 2000 →
  rate_A_to_B = 15 / 100 →
  years = 4 →
  gain_B = 160 →
  interest_rate_B_to_C principal rate_A_to_B years gain_B = 17 / 100 := by
  sorry

#eval interest_rate_B_to_C 2000 (15/100) 4 160

end interest_rate_calculation_l3993_399313


namespace unique_solution_l3993_399372

theorem unique_solution : ∃! (x y z : ℤ),
  (y^4 + 2*z^2) % 3 = 2 ∧
  (3*x^4 + z^2) % 5 = 1 ∧
  y^4 + 2*z^2 = 3*x^4 + z^2 - 6 ∧
  x = 5 ∧ y = 3 ∧ z = 19 := by
sorry

end unique_solution_l3993_399372


namespace system_solution_l3993_399305

/-- The system of equations:
    y^2 = (x+8)(x^2 + 2)
    y^2 - (8+4x)y + (16+16x-5x^2) = 0
    has solutions (0, ±4), (-2, ±6), (-5, ±9), and (19, ±99) -/
theorem system_solution :
  ∀ (x y : ℝ),
    (y^2 = (x+8)*(x^2 + 2) ∧
     y^2 - (8+4*x)*y + (16+16*x-5*x^2) = 0) ↔
    ((x = 0 ∧ (y = 4 ∨ y = -4)) ∨
     (x = -2 ∧ (y = 6 ∨ y = -6)) ∨
     (x = -5 ∧ (y = 9 ∨ y = -9)) ∨
     (x = 19 ∧ (y = 99 ∨ y = -99))) :=
by sorry

end system_solution_l3993_399305


namespace square_ratio_problem_l3993_399364

theorem square_ratio_problem :
  ∀ (s1 s2 : ℝ),
  (s1^2 / s2^2 = 75 / 128) →
  ∃ (a b c : ℕ),
  (s1 / s2 = (a : ℝ) * Real.sqrt b / c) ∧
  a = 5 ∧ b = 6 ∧ c = 16 ∧
  a + b + c = 27 := by
sorry

end square_ratio_problem_l3993_399364


namespace product_sum_equality_l3993_399319

/-- Given a base b, this function converts a number from base b to base 10 -/
def baseToDecimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, this function converts a number from base 10 to base b -/
def decimalToBase (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Theorem: If (13)(15)(17) = 4652 in base b, then 13 + 15 + 17 = 51 in base b -/
theorem product_sum_equality (b : ℕ) (h : b > 1) :
  (baseToDecimal 13 b * baseToDecimal 15 b * baseToDecimal 17 b = baseToDecimal 4652 b) →
  (decimalToBase (baseToDecimal 13 b + baseToDecimal 15 b + baseToDecimal 17 b) b = 51) :=
by sorry

end product_sum_equality_l3993_399319
