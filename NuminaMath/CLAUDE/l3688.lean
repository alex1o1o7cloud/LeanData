import Mathlib

namespace number_exceeding_percentage_l3688_368807

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.16 * x + 105 ∧ x = 125 := by
  sorry

end number_exceeding_percentage_l3688_368807


namespace simplify_and_evaluate_1_l3688_368854

theorem simplify_and_evaluate_1 (y : ℝ) (h : y = 2) :
  -3 * y^2 - 6 * y + 2 * y^2 + 5 * y = -6 := by
  sorry

end simplify_and_evaluate_1_l3688_368854


namespace dorothy_and_sister_ages_l3688_368849

/-- Proves the ages of Dorothy and her sister given the conditions -/
theorem dorothy_and_sister_ages :
  ∀ (d s : ℕ),
  d = 3 * s →
  d + 5 = 2 * (s + 5) →
  d = 15 ∧ s = 5 := by
  sorry

end dorothy_and_sister_ages_l3688_368849


namespace digital_earth_has_info_at_fingertips_l3688_368882

-- Define the set of technologies
inductive Technology
| Internet
| VirtualWorld
| DigitalEarth
| InformationSuperhighway

-- Define the property of "information at your fingertips"
def hasInfoAtFingertips (t : Technology) : Prop :=
  match t with
  | Technology.DigitalEarth => true
  | _ => false

-- Theorem statement
theorem digital_earth_has_info_at_fingertips :
  hasInfoAtFingertips Technology.DigitalEarth :=
by
  sorry

#check digital_earth_has_info_at_fingertips

end digital_earth_has_info_at_fingertips_l3688_368882


namespace linear_system_solution_quadratic_system_result_l3688_368895

-- Define the system of linear equations
def linear_system (x y : ℝ) : Prop :=
  3 * x - 2 * y = 5 ∧ 9 * x - 4 * y = 19

-- Define the system of quadratic equations
def quadratic_system (x y : ℝ) : Prop :=
  3 * x^2 - 2 * x * y + 12 * y^2 = 47 ∧ 2 * x^2 + x * y + 8 * y^2 = 36

-- Theorem for the linear system
theorem linear_system_solution :
  ∃ x y : ℝ, linear_system x y ∧ x = 3 ∧ y = 2 :=
sorry

-- Theorem for the quadratic system
theorem quadratic_system_result :
  ∀ x y : ℝ, quadratic_system x y → x^2 + 4 * y^2 = 17 :=
sorry

end linear_system_solution_quadratic_system_result_l3688_368895


namespace fifth_largest_divisor_l3688_368892

def n : ℕ := 1020000000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a > b ∧ b > c ∧ c > d ∧ d > e ∧ 
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    ∀ (x : ℕ), x ∣ n → x ≤ e ∨ x = d ∨ x = c ∨ x = b ∨ x = a ∨ x = n)

theorem fifth_largest_divisor : is_fifth_largest_divisor 63750000 := by
  sorry

end fifth_largest_divisor_l3688_368892


namespace cosine_product_fifteen_l3688_368839

theorem cosine_product_fifteen : 
  (Real.cos (π/15)) * (Real.cos (2*π/15)) * (Real.cos (3*π/15)) * 
  (Real.cos (4*π/15)) * (Real.cos (5*π/15)) * (Real.cos (6*π/15)) * 
  (Real.cos (7*π/15)) = -1/128 := by
sorry

end cosine_product_fifteen_l3688_368839


namespace arithmetic_sequence_sum_l3688_368897

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h5 : a 5 = 15) :
  a 3 + a 4 + a 7 + a 6 = 60 := by
  sorry

end arithmetic_sequence_sum_l3688_368897


namespace expression_value_l3688_368874

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 4 * y + 5 * z = 21 := by
  sorry

end expression_value_l3688_368874


namespace ceiling_sqrt_count_l3688_368837

theorem ceiling_sqrt_count (x : ℤ) : (∃ (count : ℕ), count = 39 ∧ 
  (∀ y : ℤ, ⌈Real.sqrt (y : ℝ)⌉ = 20 ↔ 362 ≤ y ∧ y ≤ 400) ∧
  count = (Finset.range 39).card) :=
sorry

end ceiling_sqrt_count_l3688_368837


namespace inequality_and_equality_condition_l3688_368818

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_not_all_equal : ¬(a = b ∧ b = c)) : 
  (((a - b*c)^2 + (b - a*c)^2 + (c - a*b)^2) ≥ 
   (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2)) ∧
  ((a - b*c)^2 + (b - a*c)^2 + (c - a*b)^2 = 
   (1/2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) ↔ 
   ((a > 0 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b > 0 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c > 0))) :=
by sorry

end inequality_and_equality_condition_l3688_368818


namespace science_fiction_readers_l3688_368803

theorem science_fiction_readers
  (total : ℕ)
  (literary : ℕ)
  (both : ℕ)
  (h1 : total = 150)
  (h2 : literary = 90)
  (h3 : both = 60) :
  total = literary + (total - literary - both) - both :=
by sorry

end science_fiction_readers_l3688_368803


namespace scientific_notation_of_29150000_l3688_368848

theorem scientific_notation_of_29150000 : 
  29150000 = 2.915 * (10 : ℝ)^7 := by sorry

end scientific_notation_of_29150000_l3688_368848


namespace quadratic_real_roots_l3688_368819

theorem quadratic_real_roots (k : ℝ) : 
  k > 0 → ∃ x : ℝ, x^2 - x - k = 0 := by
  sorry

end quadratic_real_roots_l3688_368819


namespace random_opening_page_8_is_random_event_l3688_368884

/-- Represents a math book with a specified number of pages. -/
structure MathBook where
  pages : ℕ
  pages_positive : pages > 0

/-- Represents the act of opening a book randomly. -/
def RandomOpening (book : MathBook) := Unit

/-- Represents the result of opening a book to a specific page. -/
structure OpeningResult (book : MathBook) where
  page : ℕ
  page_valid : page > 0 ∧ page ≤ book.pages

/-- Defines what it means for an event to be random. -/
def IsRandomEvent (P : Prop) : Prop :=
  ¬(P ↔ True) ∧ ¬(P ↔ False)

/-- Theorem stating that opening a 200-page math book randomly and landing on page 8 is a random event. -/
theorem random_opening_page_8_is_random_event (book : MathBook) 
  (h_pages : book.pages = 200) :
  IsRandomEvent (∃ (opening : RandomOpening book) (result : OpeningResult book), result.page = 8) :=
sorry

end random_opening_page_8_is_random_event_l3688_368884


namespace least_sum_with_equation_l3688_368880

theorem least_sum_with_equation (x y z : ℕ+) 
  (eq : 4 * x.val = 5 * y.val) 
  (least_sum : ∀ (a b c : ℕ+), 4 * a.val = 5 * b.val → a.val + b.val + c.val ≥ x.val + y.val + z.val) 
  (sum_37 : x.val + y.val + z.val = 37) : 
  z.val = 28 := by
sorry

end least_sum_with_equation_l3688_368880


namespace M_equals_N_l3688_368855

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by
  sorry

end M_equals_N_l3688_368855


namespace lcm_problem_l3688_368843

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 11) (h2 : a * b = 1991) :
  Nat.lcm a b = 181 := by
  sorry

end lcm_problem_l3688_368843


namespace min_cuts_for_quadrilaterals_l3688_368856

/-- Represents the number of cuts made on the paper -/
def num_cuts : ℕ := 1699

/-- Represents the number of quadrilaterals to be obtained -/
def target_quadrilaterals : ℕ := 100

/-- Represents the initial number of vertices in a square -/
def initial_vertices : ℕ := 4

/-- Represents the maximum number of new vertices added per cut -/
def max_new_vertices_per_cut : ℕ := 4

/-- Represents the number of vertices in a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

theorem min_cuts_for_quadrilaterals :
  (num_cuts + 1 = target_quadrilaterals) ∧
  (initial_vertices + num_cuts * max_new_vertices_per_cut ≥ target_quadrilaterals * vertices_per_quadrilateral) :=
sorry

end min_cuts_for_quadrilaterals_l3688_368856


namespace curve_slope_range_l3688_368889

/-- The curve y = ln x + ax² - 2x has no tangent lines with negative slope -/
def no_negative_slope (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (1 / x + 2 * a * x - 2) ≥ 0

/-- The range of a for which the curve has no negative slope tangents -/
theorem curve_slope_range (a : ℝ) : no_negative_slope a → a ≥ (1 / 2) := by
  sorry

end curve_slope_range_l3688_368889


namespace picture_area_l3688_368809

theorem picture_area (x y : ℤ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : (3*x + 2)*(y + 4) - x*y = 62) : 
  x * y = 10 := by
sorry

end picture_area_l3688_368809


namespace dorothy_doughnuts_l3688_368868

/-- Represents the problem of calculating the number of doughnuts Dorothy made. -/
theorem dorothy_doughnuts (ingredient_cost : ℕ) (selling_price : ℕ) (profit : ℕ) 
  (h1 : ingredient_cost = 53)
  (h2 : selling_price = 3)
  (h3 : profit = 22) :
  ∃ (num_doughnuts : ℕ), 
    selling_price * num_doughnuts = ingredient_cost + profit ∧ 
    num_doughnuts = 25 := by
  sorry


end dorothy_doughnuts_l3688_368868


namespace six_times_r_of_30_l3688_368866

def r (θ : ℚ) : ℚ := 1 / (2 - θ)

theorem six_times_r_of_30 : r (r (r (r (r (r 30))))) = 144 / 173 := by
  sorry

end six_times_r_of_30_l3688_368866


namespace complex_fraction_simplification_l3688_368858

theorem complex_fraction_simplification :
  (2 : ℂ) / (Complex.I * (3 - Complex.I)) = (1 - 3 * Complex.I) / 5 := by
  sorry

end complex_fraction_simplification_l3688_368858


namespace function_properties_and_range_l3688_368842

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_properties_and_range 
  (A ω φ : ℝ) 
  (h_A : A > 0) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < π)
  (h_max : f A ω φ (π/6) = 2)
  (h_roots : ∃ x₁ x₂, f A ω φ x₁ = 0 ∧ f A ω φ x₂ = 0 ∧ 
    ∀ y₁ y₂, f A ω φ y₁ = 0 → f A ω φ y₂ = 0 → |y₁ - y₂| ≥ π) :
  (∀ x, f A ω φ x = 2 * Real.sin (x + π/3)) ∧
  (∀ x ∈ Set.Icc (-π/4) (π/4), 
    2 * Real.sin (2*x + π/3) ∈ Set.Icc (-1) 2) := by
  sorry

end function_properties_and_range_l3688_368842


namespace problem_statement_l3688_368820

-- Define the quadratic equation
def quadratic (x : ℝ) : Prop := x^2 - 4*x - 5 = 0

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem problem_statement :
  -- Part 1: x = 5 is sufficient but not necessary for quadratic
  (∃ x ≠ 5, quadratic x) ∧ (quadratic 5) ∧
  -- Part 2: (∃x, tan x = 1) ∧ (¬(∀x, x^2 - x + 1 > 0)) is false
  ¬((∃ x : ℝ, Real.tan x = 1) ∧ ¬(∀ x : ℝ, x^2 - x + 1 > 0)) ∧
  -- Part 3: Tangent line equation at (2, f(2)) is y = -3
  (∃ m : ℝ, f 2 = -3 ∧ (deriv f) 2 = m ∧ m = 0) :=
sorry

end problem_statement_l3688_368820


namespace sin_alpha_cos_beta_value_l3688_368828

theorem sin_alpha_cos_beta_value (α β : Real) 
  (h1 : Real.sin (α + β) = 1/2) 
  (h2 : Real.sin (α - β) = 1/4) : 
  Real.sin α * Real.cos β = 3/8 := by
  sorry

end sin_alpha_cos_beta_value_l3688_368828


namespace rotated_square_height_l3688_368875

theorem rotated_square_height :
  let square_side : ℝ := 1
  let rotation_angle : ℝ := 30 * π / 180
  let initial_center_height : ℝ := square_side / 2
  let diagonal : ℝ := square_side * Real.sqrt 2
  let rotated_height : ℝ := diagonal * Real.sin rotation_angle
  initial_center_height + rotated_height = (1 + Real.sqrt 2) / 2 := by
sorry

end rotated_square_height_l3688_368875


namespace auto_finance_fraction_l3688_368812

theorem auto_finance_fraction (total_credit auto_credit finance_credit : ℝ) 
  (h1 : total_credit = 291.6666666666667)
  (h2 : auto_credit = 0.36 * total_credit)
  (h3 : finance_credit = 35) :
  finance_credit / auto_credit = 1/3 := by
sorry

end auto_finance_fraction_l3688_368812


namespace count_nines_to_800_l3688_368817

/-- Count of digit 9 occurrences in integers from 1 to n -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Theorem: The count of digit 9 occurrences in integers from 1 to 800 is 160 -/
theorem count_nines_to_800 : count_nines 800 = 160 := by sorry

end count_nines_to_800_l3688_368817


namespace expression_evaluation_l3688_368824

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b + 1) 
  (h2 : b = a + 5) 
  (h3 : a = 3) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  ((a + 3) / (a + 2)) * ((b + 1) / (b - 3)) * ((c + 9) / (c + 7)) = 243 / 100 := by
  sorry

end expression_evaluation_l3688_368824


namespace great_eighteen_games_l3688_368832

/-- The Great Eighteen Hockey League -/
structure HockeyLeague where
  total_teams : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculate the total number of scheduled games in the league -/
def total_scheduled_games (league : HockeyLeague) : Nat :=
  let total_intra_division_games := league.total_teams * (league.teams_per_division - 1) * league.intra_division_games
  let total_inter_division_games := league.total_teams * league.teams_per_division * league.inter_division_games
  (total_intra_division_games + total_inter_division_games) / 2

/-- The Great Eighteen Hockey League satisfies the given conditions -/
def great_eighteen : HockeyLeague :=
  { total_teams := 18
  , teams_per_division := 9
  , intra_division_games := 3
  , inter_division_games := 2
  }

/-- Theorem: The total number of scheduled games in the Great Eighteen Hockey League is 378 -/
theorem great_eighteen_games :
  total_scheduled_games great_eighteen = 378 := by
  sorry

end great_eighteen_games_l3688_368832


namespace milburg_children_count_l3688_368814

/-- The number of children in Milburg -/
def children_count (total_population : ℕ) (adult_count : ℕ) : ℕ :=
  total_population - adult_count

/-- Theorem: The number of children in Milburg is 2987 -/
theorem milburg_children_count :
  children_count 5256 2269 = 2987 := by
  sorry

end milburg_children_count_l3688_368814


namespace project_rotation_lcm_l3688_368808

theorem project_rotation_lcm : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 12)) = 120 := by
  sorry

end project_rotation_lcm_l3688_368808


namespace empty_lorry_weight_l3688_368867

/-- The weight of an empty lorry given the following conditions:
  * The lorry is loaded with 20 bags of apples.
  * Each bag of apples weighs 60 pounds.
  * The weight of the loaded lorry is 1700 pounds.
-/
theorem empty_lorry_weight : ℕ := by
  sorry

#check empty_lorry_weight

end empty_lorry_weight_l3688_368867


namespace sum_excluding_20_formula_l3688_368885

/-- The sum of natural numbers from 1 to n, excluding 20 -/
def sum_excluding_20 (n : ℕ) : ℕ := 
  (Finset.range n).sum id - if n ≥ 20 then 20 else 0

/-- Theorem: For any natural number n > 20, the sum of all natural numbers 
    from 1 to n, excluding 20, is equal to n(n+1)/2 - 20 -/
theorem sum_excluding_20_formula {n : ℕ} (h : n > 20) : 
  sum_excluding_20 n = n * (n + 1) / 2 - 20 := by
  sorry


end sum_excluding_20_formula_l3688_368885


namespace combined_rate_is_90_l3688_368894

/-- Represents the fish fillet production scenario -/
structure FishFilletProduction where
  totalRequired : ℕ
  deadline : ℕ
  firstTeamProduction : ℕ
  secondTeamProduction : ℕ
  thirdTeamRate : ℕ

/-- Calculates the combined production rate of the third and fourth teams -/
def combinedRate (p : FishFilletProduction) : ℕ :=
  let remainingPieces := p.totalRequired - (p.firstTeamProduction + p.secondTeamProduction)
  let thirdTeamProduction := p.thirdTeamRate * p.deadline
  let fourthTeamProduction := remainingPieces - thirdTeamProduction
  p.thirdTeamRate + (fourthTeamProduction / p.deadline)

/-- Theorem stating that the combined production rate is 90 pieces per hour -/
theorem combined_rate_is_90 (p : FishFilletProduction)
    (h1 : p.totalRequired = 500)
    (h2 : p.deadline = 2)
    (h3 : p.firstTeamProduction = 189)
    (h4 : p.secondTeamProduction = 131)
    (h5 : p.thirdTeamRate = 45) :
    combinedRate p = 90 := by
  sorry

#eval combinedRate {
  totalRequired := 500,
  deadline := 2,
  firstTeamProduction := 189,
  secondTeamProduction := 131,
  thirdTeamRate := 45
}

end combined_rate_is_90_l3688_368894


namespace min_c_for_unique_solution_l3688_368821

/-- Given positive integers a, b, c with a < b < c, the minimum value of c for which the system
    of equations 2x + y = 2022 and y = |x-a| + |x-b| + |x-c| has exactly one solution is 1012. -/
theorem min_c_for_unique_solution (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b) (hbc : b < c) :
  (∃! x y : ℝ, 2 * x + y = 2022 ∧ y = |x - a| + |x - b| + |x - c|) →
  c ≥ 1012 ∧ 
  (c = 1012 → ∃! x y : ℝ, 2 * x + y = 2022 ∧ y = |x - a| + |x - b| + |x - c|) :=
by sorry

end min_c_for_unique_solution_l3688_368821


namespace regular_polygon_sides_l3688_368844

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 15 → n = 24 := by
  sorry

end regular_polygon_sides_l3688_368844


namespace trapezoid_existence_l3688_368816

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of marked vertices in a polygon -/
def MarkedVertices (n : ℕ) (m : ℕ) := Fin m → Fin n

/-- Four points form a trapezoid if two sides are parallel and not all four points are collinear -/
def IsTrapezoid (a b c d : ℝ × ℝ) : Prop := sorry

theorem trapezoid_existence (polygon : RegularPolygon 2015) (marked : MarkedVertices 2015 64) :
  ∃ (a b c d : Fin 64), IsTrapezoid (polygon.vertices (marked a)) (polygon.vertices (marked b)) 
                                    (polygon.vertices (marked c)) (polygon.vertices (marked d)) := by
  sorry

end trapezoid_existence_l3688_368816


namespace max_k_value_l3688_368883

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the condition for intersection
def has_intersection (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line k x y ∧ 
  (∃ (x' y' : ℝ), circle_C x' y' ∧ (x - x')^2 + (y - y')^2 ≤ 1)

-- Theorem statement
theorem max_k_value :
  (∀ k : ℝ, k ≤ 4/3 → has_intersection k) ∧
  (∀ k : ℝ, k > 4/3 → ¬has_intersection k) :=
sorry

end max_k_value_l3688_368883


namespace car_speed_problem_l3688_368827

/-- Proves that car R's speed is 75 mph given the problem conditions -/
theorem car_speed_problem (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  distance = 1200 →
  time_difference = 4 →
  speed_difference = 20 →
  ∃ (speed_R : ℝ),
    distance / speed_R - time_difference = distance / (speed_R + speed_difference) ∧
    speed_R = 75 := by
  sorry


end car_speed_problem_l3688_368827


namespace train_speed_calculation_l3688_368896

/-- Proves that given a jogger running at 9 kmph, 230 meters ahead of a 120-meter long train,
    if the train passes the jogger in 35 seconds, then the speed of the train is 19 kmph. -/
theorem train_speed_calculation (jogger_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  initial_distance = 230 →
  train_length = 120 →
  passing_time = 35 / 3600 →
  ∃ (train_speed : ℝ), train_speed = 19 ∧
    (initial_distance + train_length) / passing_time = train_speed - jogger_speed :=
by sorry

end train_speed_calculation_l3688_368896


namespace sum_of_squares_of_roots_l3688_368862

theorem sum_of_squares_of_roots : ∃ (a b c d : ℝ),
  (∀ x : ℝ, x^4 - 15*x^2 + 56 = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) →
  a^2 + b^2 + c^2 + d^2 = 30 := by
  sorry

end sum_of_squares_of_roots_l3688_368862


namespace unique_zero_implies_a_gt_one_main_theorem_l3688_368834

/-- A function f(x) = 2ax^2 - x - 1 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x^2 - x - 1

/-- The property that f has exactly one zero in the interval (0,1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 1 ∧ f a x = 0

/-- Theorem stating that if f has exactly one zero in (0,1), then a > 1 -/
theorem unique_zero_implies_a_gt_one :
  ∀ a : ℝ, has_unique_zero_in_interval a → a > 1 :=
sorry

/-- The main theorem: if f has exactly one zero in (0,1), then a ∈ (1, +∞) -/
theorem main_theorem :
  ∀ a : ℝ, has_unique_zero_in_interval a → a ∈ Set.Ioi 1 :=
sorry

end unique_zero_implies_a_gt_one_main_theorem_l3688_368834


namespace annes_wandering_time_l3688_368865

/-- Proves that Anne's wandering time is 1.5 hours given her distance and speed -/
theorem annes_wandering_time
  (distance : ℝ) (speed : ℝ)
  (h1 : distance = 3.0)
  (h2 : speed = 2.0)
  : distance / speed = 1.5 := by
  sorry

end annes_wandering_time_l3688_368865


namespace license_plate_count_l3688_368871

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 4

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 3

/-- The number of license plate combinations with the given conditions -/
def license_plate_combinations : ℕ :=
  alphabet_size *  -- Choose the repeated letter
  (alphabet_size - 1).choose 2 *  -- Choose the other two distinct letters
  letter_positions.choose 2 *  -- Arrange the repeated letters
  2 *  -- Arrange the remaining two letters
  digit_count *  -- Choose the digit to repeat
  digit_positions.choose 2 *  -- Choose positions for the repeated digit
  (digit_count - 1)  -- Choose the second, different digit

/-- Theorem stating the number of possible license plate combinations -/
theorem license_plate_count : license_plate_combinations = 4212000 := by
  sorry

end license_plate_count_l3688_368871


namespace second_smallest_box_count_l3688_368850

theorem second_smallest_box_count : 
  (∃ n : ℕ, n > 0 ∧ n < 8 ∧ 12 * n % 10 = 6) ∧
  (∀ n : ℕ, n > 0 ∧ n < 8 → 12 * n % 10 ≠ 6) ∧
  12 * 8 % 10 = 6 := by
sorry

end second_smallest_box_count_l3688_368850


namespace third_grade_trees_l3688_368879

theorem third_grade_trees (total_students : ℕ) (total_trees : ℕ) 
  (trees_per_third : ℕ) (trees_per_fourth : ℕ) (trees_per_fifth : ℚ) :
  total_students = 100 →
  total_trees = 566 →
  trees_per_third = 4 →
  trees_per_fourth = 5 →
  trees_per_fifth = 13/2 →
  ∃ (third_students fourth_students fifth_students : ℕ),
    third_students = fourth_students ∧
    third_students + fourth_students + fifth_students = total_students ∧
    third_students * trees_per_third + fourth_students * trees_per_fourth + 
      (fifth_students : ℚ) * trees_per_fifth = total_trees ∧
    third_students * trees_per_third = 84 :=
by
  sorry

#check third_grade_trees

end third_grade_trees_l3688_368879


namespace solve_exponential_equation_l3688_368870

theorem solve_exponential_equation (y : ℝ) :
  (5 : ℝ) ^ (3 * y) = Real.sqrt 125 → y = 1 / 2 := by
  sorry

end solve_exponential_equation_l3688_368870


namespace least_positive_integer_multiple_l3688_368864

theorem least_positive_integer_multiple (x : ℕ) : x = 16 ↔ 
  (x > 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → ¬(∃ k : ℤ, (3*y)^2 + 2*58*3*y + 58^2 = 53*k) ∧
   ∃ k : ℤ, (3*x)^2 + 2*58*3*x + 58^2 = 53*k) := by
sorry

end least_positive_integer_multiple_l3688_368864


namespace car_journey_initial_speed_l3688_368886

/-- Represents the speed and position of a car on a journey --/
structure CarJourney where
  initial_speed : ℝ
  total_distance : ℝ
  distance_to_b : ℝ
  distance_to_c : ℝ
  time_remaining_at_b : ℝ
  speed_reduction : ℝ

/-- Theorem stating the conditions of the car journey and the initial speed to be proved --/
theorem car_journey_initial_speed (j : CarJourney) 
  (h1 : j.total_distance = 100)
  (h2 : j.time_remaining_at_b = 0.5)
  (h3 : j.speed_reduction = 10)
  (h4 : j.distance_to_c = 80)
  (h5 : (j.distance_to_b / (j.initial_speed - j.speed_reduction) - 
         (j.distance_to_c - j.distance_to_b) / (j.initial_speed - 2 * j.speed_reduction)) = 1/12)
  : j.initial_speed = 100 := by
  sorry

#check car_journey_initial_speed

end car_journey_initial_speed_l3688_368886


namespace two_pairs_satisfying_equation_l3688_368810

theorem two_pairs_satisfying_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℕ), 
    (2 * x₁^3 = y₁^4) ∧ 
    (2 * x₂^3 = y₂^4) ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by sorry

end two_pairs_satisfying_equation_l3688_368810


namespace inequality_solution_set_l3688_368829

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < a) → a > 2 := by
  sorry

end inequality_solution_set_l3688_368829


namespace arithmetic_sequence_problem_l3688_368831

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 7th term of the arithmetic sequence is 1 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) 
    (h_a4 : a 4 = 4)
    (h_sum : a 3 + a 8 = 5) : 
  a 7 = 1 := by
  sorry

end arithmetic_sequence_problem_l3688_368831


namespace volume_Q₃_l3688_368853

/-- Represents a polyhedron in the sequence -/
structure Polyhedron where
  volume : ℚ

/-- Generates the next polyhedron in the sequence -/
def next_polyhedron (Q : Polyhedron) : Polyhedron :=
  { volume := Q.volume + 4 * (27/64) * Q.volume }

/-- The initial tetrahedron Q₀ -/
def Q₀ : Polyhedron :=
  { volume := 2 }

/-- The sequence of polyhedra -/
def Q : ℕ → Polyhedron
  | 0 => Q₀
  | n + 1 => next_polyhedron (Q n)

theorem volume_Q₃ : (Q 3).volume = 156035 / 65536 := by sorry

end volume_Q₃_l3688_368853


namespace quadratic_distinct_roots_l3688_368859

/-- The quadratic equation x^2 - 2x + k - 1 = 0 has two distinct real roots if and only if k < 2 -/
theorem quadratic_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k - 1 = 0 ∧ y^2 - 2*y + k - 1 = 0) ↔ k < 2 := by
  sorry

end quadratic_distinct_roots_l3688_368859


namespace cannot_determine_unique_order_l3688_368873

/-- Represents a query about the relative ordering of 3 weights -/
structure Query where
  a : Fin 5
  b : Fin 5
  c : Fin 5
  h₁ : a ≠ b
  h₂ : b ≠ c
  h₃ : a ≠ c

/-- Represents a permutation of 5 weights -/
def Permutation := Fin 5 → Fin 5

/-- Checks if a permutation is consistent with a query -/
def consistentWithQuery (p : Permutation) (q : Query) : Prop :=
  p q.a < p q.b ∧ p q.b < p q.c

/-- Checks if a permutation is consistent with all queries in a list -/
def consistentWithAllQueries (p : Permutation) (qs : List Query) : Prop :=
  ∀ q ∈ qs, consistentWithQuery p q

theorem cannot_determine_unique_order :
  ∀ (qs : List Query),
    qs.length = 9 →
    ∃ (p₁ p₂ : Permutation),
      p₁ ≠ p₂ ∧
      consistentWithAllQueries p₁ qs ∧
      consistentWithAllQueries p₂ qs :=
sorry

end cannot_determine_unique_order_l3688_368873


namespace insurance_coverage_percentage_l3688_368800

theorem insurance_coverage_percentage
  (frames_cost : ℝ)
  (lenses_cost : ℝ)
  (coupon_value : ℝ)
  (final_cost : ℝ)
  (h1 : frames_cost = 200)
  (h2 : lenses_cost = 500)
  (h3 : coupon_value = 50)
  (h4 : final_cost = 250) :
  (((frames_cost + lenses_cost - coupon_value) - final_cost) / lenses_cost) * 100 = 80 :=
by sorry

end insurance_coverage_percentage_l3688_368800


namespace candy_mixture_weight_l3688_368825

/-- Proves that a candy mixture weighs 80 pounds given specific conditions -/
theorem candy_mixture_weight :
  ∀ (x : ℝ),
  x ≥ 0 →
  2 * x + 3 * 16 = 2.20 * (x + 16) →
  x + 16 = 80 :=
by
  sorry

end candy_mixture_weight_l3688_368825


namespace probability_less_than_20_l3688_368813

theorem probability_less_than_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 120) (h2 : over_30 = 90) :
  let under_20 := total - over_30
  (under_20 : ℚ) / total = 1 / 4 := by
  sorry

end probability_less_than_20_l3688_368813


namespace complement_A_inter_B_l3688_368872

def U : Set Int := {-1, 0, 1, 2, 3, 4}
def A : Set Int := {1, 2, 3, 4}
def B : Set Int := {0, 2}

theorem complement_A_inter_B : (U \ A) ∩ B = {0} := by sorry

end complement_A_inter_B_l3688_368872


namespace min_coefficient_value_l3688_368888

theorem min_coefficient_value (c d : ℤ) (box : ℤ) : 
  (c * d = 42) →
  (c ≠ d) → (c ≠ box) → (d ≠ box) →
  (∀ x, (c * x + d) * (d * x + c) = 42 * x^2 + box * x + 42) →
  (∀ c' d' box' : ℤ, 
    (c' * d' = 42) → 
    (c' ≠ d') → (c' ≠ box') → (d' ≠ box') →
    (∀ x, (c' * x + d') * (d' * x + c') = 42 * x^2 + box' * x + 42) →
    box ≤ box') →
  box = 85 := by
sorry

end min_coefficient_value_l3688_368888


namespace triangle_problem_l3688_368852

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * Real.sin t.B = Real.sin t.A + Real.cos t.A * Real.tan t.C)
  (h2 : t.b = 4)
  (h3 : (t.a + t.b + t.c) / 2 * (Real.sqrt 3 / 2) = t.a * t.b * Real.sin t.C / 2) :
  t.C = Real.pi / 3 ∧ t.a - t.c = -1 := by
  sorry

end triangle_problem_l3688_368852


namespace pipe_fill_time_l3688_368860

theorem pipe_fill_time (fill_time_A fill_time_all empty_time : ℝ) 
  (h1 : fill_time_A = 60)
  (h2 : fill_time_all = 50)
  (h3 : empty_time = 100.00000000000001) :
  ∃ fill_time_B : ℝ, fill_time_B = 75 ∧ 
  (1 / fill_time_A + 1 / fill_time_B - 1 / empty_time = 1 / fill_time_all) := by
  sorry

#check pipe_fill_time

end pipe_fill_time_l3688_368860


namespace unfactorable_quartic_l3688_368891

theorem unfactorable_quartic :
  ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
    x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) :=
by sorry

end unfactorable_quartic_l3688_368891


namespace used_car_selection_l3688_368836

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 12 →
  num_clients = 9 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 4 := by
sorry

end used_car_selection_l3688_368836


namespace agency_a_cheaper_l3688_368863

/-- Represents a travel agency with a pricing function -/
structure TravelAgency where
  price : ℕ → ℝ

/-- The initial price per person -/
def initialPrice : ℝ := 200

/-- Travel Agency A with 25% discount for all -/
def agencyA : TravelAgency :=
  { price := λ x => initialPrice * 0.75 * x }

/-- Travel Agency B with one free and 20% discount for the rest -/
def agencyB : TravelAgency :=
  { price := λ x => initialPrice * 0.8 * (x - 1) }

/-- Theorem stating when Agency A is cheaper than Agency B -/
theorem agency_a_cheaper (x : ℕ) :
  x > 16 → agencyA.price x < agencyB.price x :=
sorry

end agency_a_cheaper_l3688_368863


namespace product_in_base5_l3688_368841

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base-10 number to base-5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec convert (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else convert (m / 5) ((m % 5) :: acc)
    convert n []

theorem product_in_base5 :
  let a := [4, 1, 3, 2]  -- 2314₅ in reverse order
  let b := [3, 2]        -- 23₅ in reverse order
  base10ToBase5 (base5ToBase10 a * base5ToBase10 b) = [2, 3, 3, 8, 6] :=
by sorry

end product_in_base5_l3688_368841


namespace b_2016_equals_zero_l3688_368847

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the sequence b_n
def b (n : ℕ) : ℕ := fib n % 4

-- Theorem statement
theorem b_2016_equals_zero : b 2016 = 0 := by
  sorry

end b_2016_equals_zero_l3688_368847


namespace greatest_three_digit_multiple_of_17_l3688_368822

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l3688_368822


namespace bathroom_extension_l3688_368846

/-- Represents the dimensions and area of a rectangular bathroom -/
structure Bathroom where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Calculates the new area of a bathroom after extension -/
def extended_area (b : Bathroom) (extension : ℝ) : ℝ :=
  (b.width + 2 * extension) * (b.length + 2 * extension)

/-- Theorem: Given a bathroom with area 96 sq ft and width 8 ft, 
    extending it by 2 ft on each side results in an area of 140 sq ft -/
theorem bathroom_extension :
  ∀ (b : Bathroom),
    b.area = 96 ∧ b.width = 8 →
    extended_area b 2 = 140 := by
  sorry

end bathroom_extension_l3688_368846


namespace triangle_side_ratio_sum_equals_one_l3688_368840

theorem triangle_side_ratio_sum_equals_one (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let angle_A : ℝ := 60 * π / 180
  (a^2 = b^2 + c^2 - 2*b*c*(angle_A.cos)) →
  (c / (a + b) + b / (a + c) = 1) := by
  sorry

end triangle_side_ratio_sum_equals_one_l3688_368840


namespace fixed_point_on_line_l3688_368893

theorem fixed_point_on_line (a b : ℝ) (h : a + 2 * b = 1) :
  a * (1/2) + 3 * (-1/6) + b = 0 := by
sorry

end fixed_point_on_line_l3688_368893


namespace students_playing_neither_l3688_368805

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 35 →
  football = 26 →
  tennis = 20 →
  both = 17 →
  total - (football + tennis - both) = 6 :=
by sorry

end students_playing_neither_l3688_368805


namespace compute_expression_l3688_368890

theorem compute_expression : 10 + 4 * (5 - 10)^3 = -490 := by
  sorry

end compute_expression_l3688_368890


namespace washers_remaining_l3688_368801

/-- Calculates the number of washers remaining after a plumbing job. -/
theorem washers_remaining
  (total_pipe_length : ℕ)
  (pipe_per_bolt : ℕ)
  (washers_per_bolt : ℕ)
  (initial_washers : ℕ)
  (h1 : total_pipe_length = 40)
  (h2 : pipe_per_bolt = 5)
  (h3 : washers_per_bolt = 2)
  (h4 : initial_washers = 20) :
  initial_washers - (total_pipe_length / pipe_per_bolt * washers_per_bolt) = 4 :=
by
  sorry


end washers_remaining_l3688_368801


namespace invalid_votes_count_l3688_368869

/-- Proves that the number of invalid votes is 100 in an election with given conditions -/
theorem invalid_votes_count (total_votes : ℕ) (valid_votes : ℕ) (loser_percentage : ℚ) (vote_difference : ℕ) : 
  total_votes = 12600 →
  loser_percentage = 30/100 →
  vote_difference = 5000 →
  valid_votes = vote_difference / (1/2 - loser_percentage) →
  total_votes - valid_votes = 100 := by
  sorry

#check invalid_votes_count

end invalid_votes_count_l3688_368869


namespace min_value_f_min_value_f_attained_max_value_g_max_value_g_attained_l3688_368877

-- Part Ⅰ
theorem min_value_f (x : ℝ) (hx : x > 0) : 12/x + 3*x ≥ 12 := by
  sorry

theorem min_value_f_attained : ∃ x : ℝ, x > 0 ∧ 12/x + 3*x = 12 := by
  sorry

-- Part Ⅱ
theorem max_value_g (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) : x*(1 - 3*x) ≤ 1/12 := by
  sorry

theorem max_value_g_attained : ∃ x : ℝ, x > 0 ∧ x < 1/3 ∧ x*(1 - 3*x) = 1/12 := by
  sorry

end min_value_f_min_value_f_attained_max_value_g_max_value_g_attained_l3688_368877


namespace a_spending_percentage_l3688_368861

def total_salary : ℝ := 4000
def a_salary : ℝ := 3000
def b_spending_percentage : ℝ := 0.85

theorem a_spending_percentage :
  ∃ (a_spending : ℝ),
    a_spending = 0.95 ∧
    a_salary * (1 - a_spending) = (total_salary - a_salary) * (1 - b_spending_percentage) :=
by sorry

end a_spending_percentage_l3688_368861


namespace pizza_counting_theorem_l3688_368878

/-- The number of available pizza toppings -/
def num_toppings : ℕ := 6

/-- Calculates the number of pizzas with exactly k toppings -/
def pizzas_with_k_toppings (k : ℕ) : ℕ := Nat.choose num_toppings k

/-- The total number of pizzas with one, two, or three toppings -/
def total_pizzas : ℕ := 
  pizzas_with_k_toppings 1 + pizzas_with_k_toppings 2 + pizzas_with_k_toppings 3

theorem pizza_counting_theorem : total_pizzas = 41 := by
  sorry

end pizza_counting_theorem_l3688_368878


namespace intersection_point_theorem_l3688_368835

theorem intersection_point_theorem (n : ℕ) (hn : n > 0) :
  let x : ℝ := n
  let y : ℝ := n^2
  (y = n * x) ∧ (y = n^3 / x) :=
by sorry

end intersection_point_theorem_l3688_368835


namespace average_speed_uphill_downhill_l3688_368899

/-- Theorem: Average speed of a car traveling uphill and downhill -/
theorem average_speed_uphill_downhill 
  (uphill_speed : ℝ) 
  (downhill_speed : ℝ) 
  (uphill_distance : ℝ) 
  (downhill_distance : ℝ) 
  (h1 : uphill_speed = 30) 
  (h2 : downhill_speed = 40) 
  (h3 : uphill_distance = 100) 
  (h4 : downhill_distance = 50) : 
  (uphill_distance + downhill_distance) / 
  (uphill_distance / uphill_speed + downhill_distance / downhill_speed) = 1800 / 55 := by
  sorry

end average_speed_uphill_downhill_l3688_368899


namespace unique_n_exists_l3688_368826

theorem unique_n_exists : ∃! n : ℤ,
  50 < n ∧ n < 150 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n = 63 := by
sorry

end unique_n_exists_l3688_368826


namespace second_tract_length_l3688_368845

/-- Given two rectangular tracts of land with specified dimensions and combined area,
    prove that the length of the second tract is 250 meters. -/
theorem second_tract_length
  (tract1_length : ℝ)
  (tract1_width : ℝ)
  (tract2_width : ℝ)
  (combined_area : ℝ)
  (h1 : tract1_length = 300)
  (h2 : tract1_width = 500)
  (h3 : tract2_width = 630)
  (h4 : combined_area = 307500)
  : ∃ tract2_length : ℝ,
    tract2_length = 250 ∧
    tract1_length * tract1_width + tract2_length * tract2_width = combined_area :=
by
  sorry

end second_tract_length_l3688_368845


namespace power_mod_eleven_l3688_368830

theorem power_mod_eleven : 5^303 % 11 = 4 := by
  sorry

end power_mod_eleven_l3688_368830


namespace min_prime_sum_l3688_368804

theorem min_prime_sum (m n p : ℕ) : 
  m.Prime ∧ n.Prime ∧ p.Prime →
  ∃ k : ℕ, k = 47 + m ∧ k = 53 + n ∧ k = 71 + p →
  m + n + p ≥ 57 :=
by sorry

end min_prime_sum_l3688_368804


namespace nursery_seedling_price_l3688_368811

theorem nursery_seedling_price :
  ∀ (price_day2 : ℝ),
    (price_day2 > 0) →
    (2 * (8000 / (price_day2 - 5)) = 17000 / price_day2) →
    price_day2 = 85 := by
  sorry

end nursery_seedling_price_l3688_368811


namespace circle_reflection_translation_l3688_368823

/-- Given a point (3, -4), prove that after reflecting it across the x-axis
    and translating it 3 units to the right, the resulting coordinates are (6, 4). -/
theorem circle_reflection_translation :
  let initial_point : ℝ × ℝ := (3, -4)
  let reflected_point := (initial_point.1, -initial_point.2)
  let translated_point := (reflected_point.1 + 3, reflected_point.2)
  translated_point = (6, 4) := by sorry

end circle_reflection_translation_l3688_368823


namespace inner_perimeter_le_outer_perimeter_l3688_368876

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : sorry -- Axiom stating that the polygon is convex

/-- Defines when one polygon is inside another -/
def is_inside (inner outer : ConvexPolygon) : Prop := sorry

/-- Calculates the perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- Theorem: If one convex polygon is inside another, then the perimeter of the inner polygon
    does not exceed the perimeter of the outer polygon -/
theorem inner_perimeter_le_outer_perimeter (inner outer : ConvexPolygon) 
  (h : is_inside inner outer) : perimeter inner ≤ perimeter outer := by
  sorry

end inner_perimeter_le_outer_perimeter_l3688_368876


namespace geometric_sequence_sum_l3688_368806

/-- Given that a, b, c, and d form a geometric sequence,
    prove that a+b, b+c, c+d form a geometric sequence -/
theorem geometric_sequence_sum (a b c d : ℝ) 
  (h : ∃ (r : ℝ), b = a * r ∧ c = b * r ∧ d = c * r) : 
  ∃ (q : ℝ), (b + c) = (a + b) * q ∧ (c + d) = (b + c) * q := by
  sorry

end geometric_sequence_sum_l3688_368806


namespace problem_solution_l3688_368833

theorem problem_solution (a b : ℝ) (m n : ℕ) 
  (h : (2 * a^m * b^(m+n))^3 = 8 * a^9 * b^15) : 
  m = 3 ∧ n = 2 := by
sorry

end problem_solution_l3688_368833


namespace lcm_factor_proof_l3688_368851

theorem lcm_factor_proof (A B : ℕ+) (Y : ℕ+) : 
  Nat.gcd A B = 63 →
  Nat.lcm A B = 63 * 11 * Y →
  A = 1071 →
  Y = 17 := by
sorry

end lcm_factor_proof_l3688_368851


namespace lcm_1584_1188_l3688_368857

theorem lcm_1584_1188 : Nat.lcm 1584 1188 = 4752 := by
  sorry

end lcm_1584_1188_l3688_368857


namespace pyramid_height_equals_cube_volume_l3688_368887

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 5 →
  pyramid_base = 6 →
  (1 / 3) * pyramid_base^2 * pyramid_height = cube_edge^3 →
  pyramid_height = 125 / 12 :=
by sorry

end pyramid_height_equals_cube_volume_l3688_368887


namespace cm_per_inch_l3688_368815

/-- Theorem: Given the map scale and measured distance, prove the number of centimeters in one inch -/
theorem cm_per_inch (map_scale_inches : Real) (map_scale_miles : Real) 
  (measured_cm : Real) (measured_miles : Real) :
  map_scale_inches = 1.5 →
  map_scale_miles = 24 →
  measured_cm = 47 →
  measured_miles = 296.06299212598424 →
  (measured_cm / (measured_miles / (map_scale_miles / map_scale_inches))) = 2.54 :=
by sorry

end cm_per_inch_l3688_368815


namespace set_A_is_correct_l3688_368881

-- Define the set A
def A : Set ℝ := {x : ℝ | x = -3 ∨ x = -1/2 ∨ x = 1/3 ∨ x = 2}

-- Define the property that if a ∈ A, then (1+a)/(1-a) ∈ A
def closure_property (S : Set ℝ) : Prop :=
  ∀ a ∈ S, (1 + a) / (1 - a) ∈ S

-- Theorem statement
theorem set_A_is_correct :
  -3 ∈ A ∧ closure_property A → A = {-3, -1/2, 1/3, 2} := by sorry

end set_A_is_correct_l3688_368881


namespace linear_function_passes_through_points_l3688_368802

/-- A linear function passing through (-1, 4) also passes through (1, 0) -/
theorem linear_function_passes_through_points :
  ∀ k : ℝ, (4 = k * (-1) - k) → (0 = k * 1 - k) := by
  sorry

end linear_function_passes_through_points_l3688_368802


namespace trigonometric_identities_l3688_368898

theorem trigonometric_identities :
  (Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4) ∧
  (Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4) ∧
  (Real.sin (18 * π / 180) = (-1 + Real.sqrt 5) / 4) ∧
  (Real.cos (18 * π / 180) = Real.sqrt (10 + 2 * Real.sqrt 5) / 4) := by
  sorry

end trigonometric_identities_l3688_368898


namespace derivative_f_at_1_l3688_368838

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 24 := by sorry

end derivative_f_at_1_l3688_368838
