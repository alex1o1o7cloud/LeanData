import Mathlib

namespace max_sum_of_squares_l3271_327170

theorem max_sum_of_squares (m n : ℕ) : 
  1 ≤ m ∧ m ≤ 1981 ∧ 1 ≤ n ∧ n ≤ 1981 →
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end max_sum_of_squares_l3271_327170


namespace salary_spending_l3271_327165

theorem salary_spending (S : ℝ) (h1 : S > 0) : 
  let first_week := S / 4
  let unspent := S * 0.15
  let total_spent := S - unspent
  let last_three_weeks := total_spent - first_week
  last_three_weeks / (3 * S) = 0.2 := by sorry

end salary_spending_l3271_327165


namespace max_value_of_sum_of_roots_max_value_at_zero_l3271_327127

theorem max_value_of_sum_of_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (x + 15) + Real.sqrt (17 - x) + Real.sqrt x ≤ Real.sqrt 15 + Real.sqrt 17 :=
by sorry

theorem max_value_at_zero :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 17 ∧
  Real.sqrt (x + 15) + Real.sqrt (17 - x) + Real.sqrt x = Real.sqrt 15 + Real.sqrt 17 :=
by sorry

end max_value_of_sum_of_roots_max_value_at_zero_l3271_327127


namespace x_value_l3271_327167

theorem x_value (h1 : 25 * x^2 - 9 = 7) (h2 : 8 * (x - 2)^3 = 27) : x = 7/2 := by
  sorry

end x_value_l3271_327167


namespace odd_function_negative_x_l3271_327190

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_x 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = -x + 1) :
  ∀ x < 0, f x = -x - 1 := by
sorry

end odd_function_negative_x_l3271_327190


namespace division_problem_l3271_327168

theorem division_problem (x : ℕ+) (y : ℚ) (m : ℤ) 
  (h1 : (x : ℚ) = 11 * y + 4)
  (h2 : (2 * x : ℚ) = 8 * m * y + 3)
  (h3 : 13 * y - x = 1) :
  m = 3 := by sorry

end division_problem_l3271_327168


namespace coin_distribution_impossibility_l3271_327123

theorem coin_distribution_impossibility : ∀ n : ℕ,
  n = 44 →
  n < (10 * 9) / 2 :=
by
  sorry

#check coin_distribution_impossibility

end coin_distribution_impossibility_l3271_327123


namespace participating_countries_form_set_l3271_327198

/-- A type representing countries --/
structure Country where
  name : String

/-- A type representing a specific event --/
structure Event where
  name : String
  year : Nat

/-- A predicate that determines if a country participated in an event --/
def participated (country : Country) (event : Event) : Prop := sorry

/-- Definition of a set with definite elements --/
def isDefiniteSet (S : Set α) : Prop :=
  ∀ x, (x ∈ S) ∨ (x ∉ S)

/-- Theorem stating that countries participating in a specific event form a definite set --/
theorem participating_countries_form_set (event : Event) :
  isDefiniteSet {country : Country | participated country event} := by
  sorry

end participating_countries_form_set_l3271_327198


namespace square_side_length_l3271_327153

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * Real.sqrt 2 = diagonal ∧ side = 2 := by
  sorry

end square_side_length_l3271_327153


namespace x_less_than_y_l3271_327196

theorem x_less_than_y (n : ℕ) (x y : ℝ) 
  (hn : n > 2) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxn : x^n = x + 1) 
  (hyn : y^(n+1) = y^3 + 1) : 
  x < y :=
by sorry

end x_less_than_y_l3271_327196


namespace systematic_sampling_distance_l3271_327135

/-- Calculates the sampling distance for systematic sampling -/
def sampling_distance (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

theorem systematic_sampling_distance :
  let population : ℕ := 1200
  let sample_size : ℕ := 30
  sampling_distance population sample_size = 40 := by
  sorry

end systematic_sampling_distance_l3271_327135


namespace log_relation_l3271_327161

theorem log_relation (a b : ℝ) (h1 : a = Real.log 625 / Real.log 4) (h2 : b = Real.log 25 / Real.log 5) :
  a = 4 / b := by
  sorry

end log_relation_l3271_327161


namespace no_two_digit_integer_satisfies_conditions_l3271_327164

theorem no_two_digit_integer_satisfies_conditions : 
  ∀ n : ℕ, 10 ≤ n → n < 100 → 
  ¬(∃ (a b : ℕ), n = 10 * a + b ∧ a < 10 ∧ b < 10 ∧ 
    (n % (a + b) = 0) ∧ (n % (a^2 * b) = 0)) := by
  sorry

end no_two_digit_integer_satisfies_conditions_l3271_327164


namespace subset_increase_l3271_327185

theorem subset_increase (m k : ℕ) (hm : m > 0) (hk : k ≥ 2) :
  let original_subsets := 2^m
  let new_subsets_one := 2^(m+1)
  let new_subsets_k := 2^(m+k)
  (new_subsets_one - original_subsets = 2^m) ∧
  (new_subsets_k - original_subsets = (2^k - 1) * 2^m) := by
  sorry

end subset_increase_l3271_327185


namespace fraction_simplification_l3271_327117

theorem fraction_simplification (c : ℝ) : (6 + 2 * c) / 7 + 3 = (27 + 2 * c) / 7 := by
  sorry

end fraction_simplification_l3271_327117


namespace circle_area_comparison_l3271_327132

theorem circle_area_comparison (r s : ℝ) (h : 2 * r = (3 + Real.sqrt 2) * s) :
  π * r^2 = ((11 + 6 * Real.sqrt 2) / 4) * (π * s^2) := by
  sorry

end circle_area_comparison_l3271_327132


namespace rectangular_shape_x_value_l3271_327169

/-- A shape formed entirely of rectangles with all internal angles 90 degrees -/
structure RectangularShape where
  top_lengths : List ℝ
  bottom_lengths : List ℝ

/-- The sum of lengths in a list -/
def sum_lengths (lengths : List ℝ) : ℝ := lengths.sum

/-- The property that the sum of top lengths equals the sum of bottom lengths -/
def equal_total_length (shape : RectangularShape) : Prop :=
  sum_lengths shape.top_lengths = sum_lengths shape.bottom_lengths

theorem rectangular_shape_x_value (shape : RectangularShape) 
  (h1 : shape.top_lengths = [2, 3, 4, X])
  (h2 : shape.bottom_lengths = [1, 2, 4, 6])
  (h3 : equal_total_length shape) :
  X = 4 := by
  sorry

#check rectangular_shape_x_value

end rectangular_shape_x_value_l3271_327169


namespace fraction_multiplication_l3271_327103

theorem fraction_multiplication : (2 : ℚ) / 15 * 5 / 8 = 1 / 12 := by
  sorry

end fraction_multiplication_l3271_327103


namespace store_desktop_sales_l3271_327108

/-- Given a ratio of laptops to desktops and an expected number of laptop sales,
    calculate the expected number of desktop sales. -/
def expected_desktop_sales (laptop_ratio : ℕ) (desktop_ratio : ℕ) (expected_laptops : ℕ) : ℕ :=
  (expected_laptops * desktop_ratio) / laptop_ratio

/-- Proof that given the specific ratio and expected laptop sales,
    the expected desktop sales is 24. -/
theorem store_desktop_sales : expected_desktop_sales 5 3 40 = 24 := by
  sorry

#eval expected_desktop_sales 5 3 40

end store_desktop_sales_l3271_327108


namespace tangent_line_at_point_l3271_327133

/-- The equation of the tangent line to y = 2x² at (1, 2) is y = 4x - 2 -/
theorem tangent_line_at_point (x y : ℝ) :
  (y = 2 * x^2) →  -- Given curve
  (∃ P : ℝ × ℝ, P = (1, 2)) →  -- Given point
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ∧ y = 4 * x - 2) -- Tangent line equation
  := by sorry

end tangent_line_at_point_l3271_327133


namespace union_of_P_and_Q_l3271_327113

def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {1, 3, 9}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3, 9} := by
  sorry

end union_of_P_and_Q_l3271_327113


namespace calculation_result_l3271_327106

theorem calculation_result : 50 + 50 / 50 + 50 = 101 := by sorry

end calculation_result_l3271_327106


namespace ratio_simplification_l3271_327151

theorem ratio_simplification (A B C : ℚ) : 
  (A / B = 5 / 3 / (29 / 6)) → 
  (C / A = (11 / 5) / (11 / 3)) → 
  ∃ (k : ℚ), k * A = 10 ∧ k * B = 29 ∧ k * C = 6 :=
by sorry

end ratio_simplification_l3271_327151


namespace fraction_equality_l3271_327105

theorem fraction_equality (x y p q : ℚ) : 
  (7 * x + 6 * y) / (x - 2 * y) = 27 → x / (2 * y) = p / q → p / q = 3 / 2 := by
  sorry

end fraction_equality_l3271_327105


namespace frosting_theorem_l3271_327101

/-- Jon's frosting rate in cupcakes per second -/
def jon_rate : ℚ := 1 / 40

/-- Mary's frosting rate in cupcakes per second -/
def mary_rate : ℚ := 1 / 24

/-- Time frame in seconds -/
def time_frame : ℕ := 12 * 60

/-- The number of cupcakes Jon and Mary can frost together in the given time frame -/
def cupcakes_frosted : ℕ := 48

theorem frosting_theorem : 
  ⌊(jon_rate + mary_rate) * time_frame⌋ = cupcakes_frosted := by
  sorry

end frosting_theorem_l3271_327101


namespace sin_15_cos_15_half_l3271_327116

theorem sin_15_cos_15_half : 2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end sin_15_cos_15_half_l3271_327116


namespace quadratic_equations_solutions_l3271_327176

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x => 4 * x^2 - 9 = 0
  let eq2 : ℝ → Prop := λ x => 2 * x^2 - 3 * x - 5 = 0
  let solutions1 : Set ℝ := {3/2, -3/2}
  let solutions2 : Set ℝ := {1, 5/2}
  (∀ x : ℝ, eq1 x ↔ x ∈ solutions1) ∧
  (∀ x : ℝ, eq2 x ↔ x ∈ solutions2) :=
by
  sorry


end quadratic_equations_solutions_l3271_327176


namespace rotated_A_coordinates_l3271_327134

-- Define the triangle OAB
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)

-- Define the properties of the triangle
structure Triangle where
  A : ℝ × ℝ
  first_quadrant : A.1 > 0 ∧ A.2 > 0
  right_angle : (A.1 - B.1) * (A.1 - O.1) + (A.2 - B.2) * (A.2 - O.2) = 0
  angle_AOB : Real.arctan ((A.2 - O.2) / (A.1 - O.1)) = π / 4

-- Function to rotate a point 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

-- Theorem statement
theorem rotated_A_coordinates (t : Triangle) : 
  rotate90 t.A = (-8, 8) := by sorry

end rotated_A_coordinates_l3271_327134


namespace sticker_distribution_l3271_327125

theorem sticker_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (Nat.choose (n + k - 1) (k - 1)) = 1001 :=
by sorry

end sticker_distribution_l3271_327125


namespace product_xyz_equals_one_l3271_327146

theorem product_xyz_equals_one (x y z : ℝ) 
  (eq1 : x + 1/y = 2) 
  (eq2 : y + 1/z = 2) 
  (eq3 : z + 1/x = 2) : 
  x * y * z = 1 := by
sorry

end product_xyz_equals_one_l3271_327146


namespace orange_cost_calculation_l3271_327128

theorem orange_cost_calculation (cost_three_dozen : ℝ) (dozen_count : ℕ) :
  cost_three_dozen = 22.5 →
  dozen_count = 4 →
  (cost_three_dozen / 3) * dozen_count = 30 :=
by
  sorry


end orange_cost_calculation_l3271_327128


namespace distance_on_number_line_l3271_327118

theorem distance_on_number_line (a b : ℝ) (ha : a = 5) (hb : b = -3) :
  |a - b| = 8 := by sorry

end distance_on_number_line_l3271_327118


namespace unique_subset_with_nonempty_intersection_l3271_327120

def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def B : Set ℕ := {4, 5, 6, 7, 8}

theorem unique_subset_with_nonempty_intersection :
  ∃! S : Set ℕ, S ⊆ A ∧ S ∩ B ≠ ∅ ∧ S = {5, 6} := by sorry

end unique_subset_with_nonempty_intersection_l3271_327120


namespace solve_jogging_problem_l3271_327100

def jogging_problem (daily_time : ℕ) (first_week_days : ℕ) (total_time : ℕ) : Prop :=
  let total_minutes : ℕ := total_time * 60
  let first_week_minutes : ℕ := first_week_days * daily_time
  let second_week_minutes : ℕ := total_minutes - first_week_minutes
  let second_week_days : ℕ := second_week_minutes / daily_time
  second_week_days = 5

theorem solve_jogging_problem :
  jogging_problem 30 3 4 := by sorry

end solve_jogging_problem_l3271_327100


namespace derivative_zero_necessary_not_sufficient_l3271_327139

-- Define a real-valued function on the real line
variable (f : ℝ → ℝ)

-- Assume f is differentiable
variable (hf : Differentiable ℝ f)

-- Define what it means for f to have an extremum at a point
def has_extremum_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem derivative_zero_necessary_not_sufficient :
  (∀ x : ℝ, has_extremum_at f x → deriv f x = 0) ∧
  ¬(∀ x : ℝ, deriv f x = 0 → has_extremum_at f x) :=
sorry

end derivative_zero_necessary_not_sufficient_l3271_327139


namespace jonah_aquarium_fish_count_l3271_327192

/-- The number of fish in Jonah's aquarium after all the changes -/
def final_fish_count (initial_fish : ℕ) (added_fish : ℕ) (exchanged_fish : ℕ) (x : ℕ) : ℤ :=
  (initial_fish + added_fish : ℤ) - 2 * x + exchanged_fish

theorem jonah_aquarium_fish_count :
  final_fish_count 14 2 3 x = 19 - 2 * x :=
by sorry

end jonah_aquarium_fish_count_l3271_327192


namespace selling_price_theorem_l3271_327188

/-- Calculates the selling price per tire given production costs and profit -/
def selling_price_per_tire (cost_per_batch : ℝ) (cost_per_tire : ℝ) (batch_size : ℕ) (profit_per_tire : ℝ) : ℝ :=
  cost_per_tire + profit_per_tire

/-- Theorem: The selling price per tire is the sum of cost per tire and profit per tire -/
theorem selling_price_theorem (cost_per_batch : ℝ) (cost_per_tire : ℝ) (batch_size : ℕ) (profit_per_tire : ℝ) :
  selling_price_per_tire cost_per_batch cost_per_tire batch_size profit_per_tire = cost_per_tire + profit_per_tire :=
by sorry

#eval selling_price_per_tire 22500 8 15000 10.5

end selling_price_theorem_l3271_327188


namespace adam_book_spending_l3271_327144

theorem adam_book_spending :
  ∀ (initial_amount spent_amount : ℝ),
    initial_amount = 91 →
    (initial_amount - spent_amount) / spent_amount = 10 / 3 →
    spent_amount = 21 := by
  sorry

end adam_book_spending_l3271_327144


namespace archibald_apple_eating_l3271_327154

theorem archibald_apple_eating (apples_per_day_first_two_weeks : ℕ) 
  (apples_per_day_last_two_weeks : ℕ) (total_weeks : ℕ) (average_apples_per_week : ℕ) :
  apples_per_day_first_two_weeks = 1 →
  apples_per_day_last_two_weeks = 3 →
  total_weeks = 7 →
  average_apples_per_week = 10 →
  ∃ (weeks_same_as_first_two : ℕ),
    weeks_same_as_first_two = 2 ∧
    (2 * 7 * apples_per_day_first_two_weeks) + 
    (weeks_same_as_first_two * 7 * apples_per_day_first_two_weeks) + 
    (2 * 7 * apples_per_day_last_two_weeks) = 
    total_weeks * average_apples_per_week :=
by sorry

end archibald_apple_eating_l3271_327154


namespace problem_solution_l3271_327194

theorem problem_solution (a b : ℕ) (ha : a = 3) (hb : b = 2) : 
  (a^(b+1))^a + (b^(a+1))^b = 19939 := by
  sorry

end problem_solution_l3271_327194


namespace set_intersection_complement_l3271_327124

theorem set_intersection_complement (U A B : Set ℤ) : 
  U = Set.univ ∧ A = {-1, 1, 2} ∧ B = {-1, 1} → A ∩ (U \ B) = {2} :=
by
  sorry

end set_intersection_complement_l3271_327124


namespace existence_of_critical_point_and_positive_function_l3271_327171

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem existence_of_critical_point_and_positive_function :
  (∃ t : ℝ, t ∈ Set.Ioo (1/2) 1 ∧ ∀ y : ℝ, y ∈ Set.Ioo (1/2) 1 → (deriv (f 1)) t = 0 ∧ (deriv (f 1)) y = 0 → y = t) ∧
  (∃ m : ℝ, m ∈ Set.Ioo 0 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) :=
sorry

end existence_of_critical_point_and_positive_function_l3271_327171


namespace percent_relation_l3271_327158

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : c = 0.5 * b) :
  b = 0.5 * a := by
  sorry

end percent_relation_l3271_327158


namespace committee_formation_count_l3271_327183

theorem committee_formation_count :
  let dept_A : Finset ℕ := Finset.range 6
  let dept_B : Finset ℕ := Finset.range 7
  let dept_C : Finset ℕ := Finset.range 5
  (dept_A.card * dept_B.card * dept_C.card : ℕ) = 210 :=
by
  sorry

end committee_formation_count_l3271_327183


namespace petya_wins_l3271_327136

/-- Represents the state of the game -/
structure GameState :=
  (contacts : Nat)
  (wires : Nat)
  (player_turn : Bool)

/-- The initial game state -/
def initial_state : GameState :=
  { contacts := 2000
  , wires := 2000 * 1999 / 2
  , player_turn := true }

/-- Represents a move in the game -/
inductive Move
  | cut_one
  | cut_three

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.cut_one => 
      { state with 
        wires := state.wires - 1
        player_turn := ¬state.player_turn }
  | Move.cut_three => 
      { state with 
        wires := state.wires - 3
        player_turn := ¬state.player_turn }

/-- Checks if the game is over -/
def is_game_over (state : GameState) : Bool :=
  state.wires < state.contacts

/-- Theorem: Player 2 (Petya) has a winning strategy -/
theorem petya_wins : 
  ∃ (strategy : GameState → Move), 
    ∀ (game : List Move), 
      is_game_over (List.foldl apply_move initial_state game) → 
        (List.length game % 2 = 0) :=
sorry

end petya_wins_l3271_327136


namespace inequality_solution_set_l3271_327172

def inequality (a x : ℝ) : Prop := a * x^2 - (a + 1) * x + 1 > 0

def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 then Set.Ioo (1/a) 1
  else if 0 < a ∧ a < 1 then Set.Iio 1 ∪ Set.Ioi (1/a)
  else if a = 1 then Set.Iio 1 ∪ Set.Ioi 1
  else Set.Iio (1/a) ∪ Set.Ioi 1

theorem inequality_solution_set (a : ℝ) (h : a ≠ 0) :
  {x : ℝ | inequality a x} = solution_set a :=
sorry

end inequality_solution_set_l3271_327172


namespace whipped_cream_theorem_l3271_327152

/-- Represents the number of each type of baked good produced on odd and even days -/
structure BakingSchedule where
  odd_pumpkin : ℕ
  odd_apple : ℕ
  odd_chocolate : ℕ
  even_pumpkin : ℕ
  even_apple : ℕ
  even_chocolate : ℕ
  even_lemon : ℕ

/-- Represents the amount of whipped cream needed for each type of baked good -/
structure WhippedCreamRequirement where
  pumpkin : ℚ
  apple : ℚ
  chocolate : ℚ
  lemon : ℚ

/-- Represents the number of each type of baked good Tiffany eats -/
structure TiffanyEats where
  pumpkin : ℕ
  apple : ℕ
  chocolate : ℕ
  lemon : ℕ

/-- Calculates the number of cans of whipped cream needed given the baking schedule,
    whipped cream requirements, and what Tiffany eats -/
def whippedCreamNeeded (schedule : BakingSchedule) (requirement : WhippedCreamRequirement) 
                       (tiffanyEats : TiffanyEats) : ℕ :=
  sorry

theorem whipped_cream_theorem (schedule : BakingSchedule) (requirement : WhippedCreamRequirement) 
                               (tiffanyEats : TiffanyEats) : 
  schedule = {
    odd_pumpkin := 3, odd_apple := 2, odd_chocolate := 1,
    even_pumpkin := 2, even_apple := 4, even_chocolate := 2, even_lemon := 1
  } →
  requirement = {
    pumpkin := 2, apple := 1, chocolate := 3, lemon := 3/2
  } →
  tiffanyEats = {
    pumpkin := 2, apple := 5, chocolate := 1, lemon := 1
  } →
  whippedCreamNeeded schedule requirement tiffanyEats = 252 :=
by
  sorry


end whipped_cream_theorem_l3271_327152


namespace xy_value_l3271_327138

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x + y) = 32)
  (h2 : (16 : ℝ)^(x + y) / (4 : ℝ)^(3 * y) = 256) : 
  x * y = -2 := by
  sorry

end xy_value_l3271_327138


namespace julia_watch_collection_l3271_327143

theorem julia_watch_collection (silver : ℕ) (bronze : ℕ) (gold : ℕ) : 
  silver = 20 →
  bronze = 3 * silver →
  gold = (silver + bronze) / 10 →
  silver + bronze + gold = 88 :=
by
  sorry

end julia_watch_collection_l3271_327143


namespace investment_sum_l3271_327104

/-- Represents the investment scenario described in the problem -/
structure Investment where
  principal : ℝ  -- The initial sum invested
  rate : ℝ       -- The annual simple interest rate
  peter_years : ℕ := 3
  david_years : ℕ := 4
  peter_return : ℝ := 815
  david_return : ℝ := 854

/-- The amount returned after a given number of years with simple interest -/
def amount_after (i : Investment) (years : ℕ) : ℝ :=
  i.principal + (i.principal * i.rate * years)

/-- The theorem stating that the invested sum is 698 given the conditions -/
theorem investment_sum (i : Investment) : 
  (amount_after i i.peter_years = i.peter_return) → 
  (amount_after i i.david_years = i.david_return) → 
  i.principal = 698 := by
  sorry

end investment_sum_l3271_327104


namespace arithmetic_sequence_problem_l3271_327137

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Given a_13 = S_13 = 13, prove a_1 = -11 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 13 = 13) (h2 : seq.S 13 = 13) : seq.a 1 = -11 := by
  sorry

end arithmetic_sequence_problem_l3271_327137


namespace zero_sponsorship_prob_high_sponsorship_prob_l3271_327184

-- Define the number of students and experts
def num_students : ℕ := 3
def num_experts : ℕ := 2

-- Define the probability of a "support" review
def support_prob : ℚ := 1/2

-- Define the function to calculate the probability of k successes in n trials
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Theorem for the probability of zero total sponsorship
theorem zero_sponsorship_prob :
  binomial_prob (num_students * num_experts) 0 support_prob = 1/64 := by sorry

-- Theorem for the probability of sponsorship exceeding 150,000 yuan
theorem high_sponsorship_prob :
  (binomial_prob (num_students * num_experts) 4 support_prob +
   binomial_prob (num_students * num_experts) 5 support_prob +
   binomial_prob (num_students * num_experts) 6 support_prob) = 11/32 := by sorry

end zero_sponsorship_prob_high_sponsorship_prob_l3271_327184


namespace product_derivative_at_one_l3271_327109

theorem product_derivative_at_one
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (f1 : f 1 = -1)
  (f'1 : deriv f 1 = 2)
  (g1 : g 1 = -2)
  (g'1 : deriv g 1 = 1) :
  deriv (λ x => f x * g x) 1 = -5 := by
  sorry

end product_derivative_at_one_l3271_327109


namespace line_passes_through_point_l3271_327163

/-- Given that m + 2n - 1 = 0, prove that the line mx + 3y + n = 0 passes through the point (1/2, -1/6) -/
theorem line_passes_through_point (m n : ℝ) (h : m + 2 * n - 1 = 0) :
  m * (1/2 : ℝ) + 3 * (-1/6 : ℝ) + n = 0 := by
  sorry

end line_passes_through_point_l3271_327163


namespace broadway_ticket_price_l3271_327174

theorem broadway_ticket_price (num_adults num_children : ℕ) (total_amount : ℚ) :
  num_adults = 400 →
  num_children = 200 →
  total_amount = 16000 →
  ∃ (adult_price child_price : ℚ),
    adult_price = 2 * child_price ∧
    num_adults * adult_price + num_children * child_price = total_amount ∧
    adult_price = 32 := by
  sorry

end broadway_ticket_price_l3271_327174


namespace min_value_expression_l3271_327142

theorem min_value_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m : ℝ), (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → a^2 + b^2 + 4/a^2 + b/a ≥ m) ∧
             (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ c^2 + d^2 + 4/c^2 + d/c = m) ∧
             m = Real.sqrt 15 := by
  sorry

end min_value_expression_l3271_327142


namespace min_value_of_sum_l3271_327177

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_eq : a + b + c = 3) (prod_sum_eq : a * b + b * c + a * c = 2) :
  a + b ≥ (6 - 2 * Real.sqrt 3) / 3 := by
  sorry

end min_value_of_sum_l3271_327177


namespace train_speed_calculation_l3271_327155

/-- Given a train of length 240 m crossing a platform of equal length in 27 s,
    its speed is approximately 64 km/h. -/
theorem train_speed_calculation (train_length platform_length : ℝ)
  (crossing_time : ℝ) (h1 : train_length = 240)
  (h2 : platform_length = train_length) (h3 : crossing_time = 27) :
  ∃ (speed : ℝ), abs (speed - 64) < 0.5 ∧ speed = (train_length + platform_length) / crossing_time * 3.6 :=
sorry

end train_speed_calculation_l3271_327155


namespace sin_arccos_circle_l3271_327182

theorem sin_arccos_circle (x y : ℝ) :
  y = Real.sin (Real.arccos x) ↔ x^2 + y^2 = 1 ∧ x ∈ Set.Icc (-1) 1 ∧ y ≥ 0 := by
  sorry

end sin_arccos_circle_l3271_327182


namespace floor_of_negative_three_point_seven_l3271_327199

theorem floor_of_negative_three_point_seven :
  ⌊(-3.7 : ℝ)⌋ = -4 := by sorry

end floor_of_negative_three_point_seven_l3271_327199


namespace perfect_square_property_l3271_327162

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + a n

theorem perfect_square_property (n : ℕ) (h : n > 0) : 
  ∃ m : ℤ, 2 * ((a (2 * n))^2 - 1) = m^2 := by
sorry

end perfect_square_property_l3271_327162


namespace sum_of_squares_bound_l3271_327112

theorem sum_of_squares_bound {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^4 + y^4 + z^4 = 1) : x^2 + y^2 + z^2 < Real.sqrt 3 := by
  sorry

end sum_of_squares_bound_l3271_327112


namespace f_monotonicity_and_properties_l3271_327122

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / a + a / x

theorem f_monotonicity_and_properties :
  ∀ a : ℝ, ∀ x : ℝ, x > 0 →
  (a > 0 → (
    (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a x₁ > f a x₂)
  )) ∧
  (a < 0 → (
    (∀ x₁ x₂, -a < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -a → f a x₁ < f a x₂)
  )) ∧
  (a = 1/2 → (
    ∀ x₀, x₀ > 0 →
    (2 - 1 / (2 * x₀^2) = 3/2 →
      ∀ x y, 3*x - 2*y + 2 = 0 ↔ y - f (1/2) x₀ = 3/2 * (x - x₀))
  )) ∧
  (a = 1/2 → ∀ x, x > 0 → f (1/2) x > Real.log x + x/2) :=
sorry

end f_monotonicity_and_properties_l3271_327122


namespace divisibility_implies_lower_bound_l3271_327131

theorem divisibility_implies_lower_bound (n a : ℕ) 
  (h1 : n > 1) 
  (h2 : a > n^2) 
  (h3 : ∀ i ∈ Finset.range n, ∃ x ∈ Finset.range n, (n^2 + i + 1) ∣ (a + x + 1)) : 
  a > n^4 - n^3 := by
sorry

end divisibility_implies_lower_bound_l3271_327131


namespace quadrilateral_ae_length_l3271_327102

/-- Represents a convex quadrilateral ABCD with point E at the intersection of diagonals -/
structure ConvexQuadrilateral :=
  (A B C D E : ℝ × ℝ)

/-- Properties of the specific quadrilateral in the problem -/
def QuadrilateralProperties (quad : ConvexQuadrilateral) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist quad.A quad.B = 10 ∧
  dist quad.C quad.D = 15 ∧
  dist quad.A quad.C = 17 ∧
  (quad.E.1 - quad.A.1) * (quad.D.2 - quad.A.2) = (quad.E.2 - quad.A.2) * (quad.D.1 - quad.A.1) ∧
  (quad.E.1 - quad.B.1) * (quad.C.2 - quad.B.2) = (quad.E.2 - quad.B.2) * (quad.C.1 - quad.B.1)

theorem quadrilateral_ae_length 
  (quad : ConvexQuadrilateral) 
  (h : QuadrilateralProperties quad) : 
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist quad.A quad.E = 6.8 := by
  sorry

end quadrilateral_ae_length_l3271_327102


namespace vector_sum_magnitude_l3271_327189

/-- Given two vectors a and b in R², where a is parallel to (a - b), 
    prove that the magnitude of their sum is 3√5/2. -/
theorem vector_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), a = k • (a - b)) → 
  ‖a + b‖ = (3 * Real.sqrt 5) / 2 := by
sorry

end vector_sum_magnitude_l3271_327189


namespace g_neither_even_nor_odd_l3271_327178

noncomputable def g (x : ℝ) : ℝ := Real.log (x + 2 + Real.sqrt (1 + (x + 2)^2))

theorem g_neither_even_nor_odd : 
  (∀ x, g (-x) = g x) ∧ (∀ x, g (-x) = -g x) → False := by
  sorry

end g_neither_even_nor_odd_l3271_327178


namespace arithmetic_sequence_properties_l3271_327150

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_properties
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = -3)
  (h_S5 : S a 5 = 0) :
  (∀ n : ℕ, a n = (3 * (3 - n : ℚ)) / 2) ∧
  (∀ n : ℕ, a n * S a n < 0 ↔ n = 4) :=
sorry

end arithmetic_sequence_properties_l3271_327150


namespace convenience_store_analysis_l3271_327119

-- Define the data types
structure YearData :=
  (year : Nat)
  (profit : Real)

-- Define the dataset
def dataset : List YearData := [
  ⟨2014, 27.6⟩, ⟨2015, 42.0⟩, ⟨2016, 38.4⟩, ⟨2017, 48.0⟩, ⟨2018, 63.6⟩,
  ⟨2019, 63.7⟩, ⟨2020, 72.8⟩, ⟨2021, 80.1⟩, ⟨2022, 60.5⟩, ⟨2023, 99.3⟩
]

-- Define the contingency table
def contingencyTable : Matrix (Fin 2) (Fin 2) Nat :=
  ![![2, 5],
    ![3, 0]]

-- Define the chi-square critical value
def chiSquareCritical : Real := 3.841

-- Define the prediction year
def predictionYear : Nat := 2024

-- Define the theorem
theorem convenience_store_analysis :
  -- Chi-square value is greater than the critical value
  ∃ (chiSquareValue : Real),
    chiSquareValue > chiSquareCritical ∧
    -- Predictions from two models are different
    ∃ (prediction1 prediction2 : Real),
      prediction1 ≠ prediction2 ∧
      -- Model 1: Using data from 2014 to 2023 (excluding 2022)
      (∃ (a1 b1 : Real),
        prediction1 = a1 * predictionYear + b1 ∧
        -- Model 2: Using data from 2019 to 2023
        ∃ (a2 b2 : Real),
          prediction2 = a2 * predictionYear + b2) :=
sorry

end convenience_store_analysis_l3271_327119


namespace average_cookies_per_package_l3271_327173

def cookie_counts : List Nat := [9, 11, 13, 19, 23, 27]

theorem average_cookies_per_package : 
  (List.sum cookie_counts) / (List.length cookie_counts) = 17 := by
  sorry

end average_cookies_per_package_l3271_327173


namespace unique_cube_root_property_l3271_327159

theorem unique_cube_root_property : ∃! (n : ℕ), n > 0 ∧ (∃ (a b : ℕ), 
  n = 1000 * a + b ∧ 
  b < 1000 ∧ 
  a^3 = n ∧ 
  a = n / 1000) :=
by sorry

end unique_cube_root_property_l3271_327159


namespace initial_wallet_amount_l3271_327107

def initial_investment : ℝ := 2000
def stock_price_increase : ℝ := 0.3
def final_total : ℝ := 2900

theorem initial_wallet_amount :
  let investment_value := initial_investment * (1 + stock_price_increase)
  let initial_wallet := final_total - investment_value
  initial_wallet = 300 := by sorry

end initial_wallet_amount_l3271_327107


namespace houses_in_block_l3271_327149

theorem houses_in_block (junk_mail_per_house : ℕ) (total_junk_mail : ℕ) 
  (h1 : junk_mail_per_house = 2) 
  (h2 : total_junk_mail = 14) : 
  total_junk_mail / junk_mail_per_house = 7 := by
sorry

end houses_in_block_l3271_327149


namespace quadratic_equation_has_solution_l3271_327115

theorem quadratic_equation_has_solution (a b : ℝ) :
  ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 := by
  sorry

end quadratic_equation_has_solution_l3271_327115


namespace problem_statement_l3271_327195

theorem problem_statement (x y : ℝ) (h1 : x * y = 12) (h2 : x + y = -8) :
  y * Real.sqrt (x / y) + x * Real.sqrt (y / x) = -4 * Real.sqrt 3 := by
  sorry

end problem_statement_l3271_327195


namespace divisibility_by_133_l3271_327186

theorem divisibility_by_133 (n : ℕ) : ∃ k : ℤ, 11^(n+2) + 12^(2*n+1) = 133 * k := by
  sorry

end divisibility_by_133_l3271_327186


namespace swimming_speed_is_10_l3271_327130

/-- The swimming speed of a person in still water. -/
def swimming_speed : ℝ := 10

/-- The speed of the water current. -/
def water_speed : ℝ := 8

/-- The time taken to swim against the current. -/
def swim_time : ℝ := 8

/-- The distance swam against the current. -/
def swim_distance : ℝ := 16

/-- Theorem stating that the swimming speed in still water is 10 km/h given the conditions. -/
theorem swimming_speed_is_10 :
  swimming_speed = 10 ∧
  water_speed = 8 ∧
  swim_time = 8 ∧
  swim_distance = 16 ∧
  swim_distance = (swimming_speed - water_speed) * swim_time :=
by sorry

end swimming_speed_is_10_l3271_327130


namespace range_theorem_fixed_point_theorem_l3271_327114

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y = 0

-- Define the range function
def in_range (z : ℝ) : Prop := (6 - 2*Real.sqrt 3) / 3 ≤ z ∧ z ≤ (6 + 2*Real.sqrt 3) / 3

-- Theorem 1: Range of (y+3)/x for points on circle C
theorem range_theorem (x y : ℝ) : 
  circle_C x y → in_range ((y + 3) / x) :=
sorry

-- Define a point on line l
def point_on_line_l (t : ℝ) : ℝ × ℝ := (t, 2*t)

-- Define the circle passing through P, A, C, and B
def circle_PACB (t x y : ℝ) : Prop :=
  (x - (t + 2) / 2)^2 + (y - t)^2 = (5*t^2 - 4*t + 4) / 4

-- Theorem 2: Circle PACB passes through (2/5, 4/5)
theorem fixed_point_theorem (t : ℝ) :
  circle_PACB t (2/5) (4/5) :=
sorry

end range_theorem_fixed_point_theorem_l3271_327114


namespace least_five_digit_square_cube_l3271_327121

theorem least_five_digit_square_cube : 
  (∀ n : ℕ, n < 15625 → (n < 10000 ∨ ¬∃ a b : ℕ, n = a^2 ∧ n = b^3)) ∧ 
  15625 ≥ 10000 ∧ 
  ∃ a b : ℕ, 15625 = a^2 ∧ 15625 = b^3 :=
sorry

end least_five_digit_square_cube_l3271_327121


namespace log_inequality_l3271_327166

theorem log_inequality : (Real.log 2 / Real.log 3) < (Real.log 3 / Real.log 2) ∧ (Real.log 3 / Real.log 2) < (Real.log 5 / Real.log 2) := by
  sorry

end log_inequality_l3271_327166


namespace triangle_right_angle_l3271_327197

theorem triangle_right_angle (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- sides are positive
  0 < A ∧ 0 < B ∧ 0 < C →  -- angles are positive
  A + B + C = π →  -- sum of angles in a triangle
  a * Real.cos A + b * Real.cos B = c * Real.cos C →  -- given condition
  a^2 = b^2 + c^2  -- conclusion: right triangle with a as hypotenuse
  := by sorry

end triangle_right_angle_l3271_327197


namespace coin_toss_probability_l3271_327175

/-- The probability of getting a specific sequence of heads and tails in 10 coin tosses -/
theorem coin_toss_probability : 
  let n : ℕ := 10  -- number of tosses
  let p : ℚ := 1/2  -- probability of heads (or tails) in a single toss
  (p ^ n : ℚ) = 1/1024 := by sorry

end coin_toss_probability_l3271_327175


namespace shop_length_is_20_l3271_327110

/-- Calculates the length of a shop given its monthly rent, width, and annual rent per square foot. -/
def shop_length (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) : ℕ :=
  let annual_rent := monthly_rent * 12
  let total_sqft := annual_rent / annual_rent_per_sqft
  total_sqft / width

/-- Theorem stating that for a shop with given parameters, its length is 20 feet. -/
theorem shop_length_is_20 :
  shop_length 3600 15 144 = 20 := by
  sorry

end shop_length_is_20_l3271_327110


namespace mrs_hilt_total_chapters_l3271_327180

/-- The total number of chapters Mrs. Hilt has read -/
def total_chapters_read : ℕ :=
  let last_month_17ch := 4 * 17
  let last_month_25ch := 3 * 25
  let last_month_30ch := 2 * 30
  let this_month_book1 := 18
  let this_month_book2 := 24
  last_month_17ch + last_month_25ch + last_month_30ch + this_month_book1 + this_month_book2

/-- Theorem stating that Mrs. Hilt has read 245 chapters in total -/
theorem mrs_hilt_total_chapters : total_chapters_read = 245 := by
  sorry

end mrs_hilt_total_chapters_l3271_327180


namespace min_pencils_in_box_l3271_327157

theorem min_pencils_in_box (total_pencils : ℕ) (num_boxes : ℕ) (max_capacity : ℕ)
  (h1 : total_pencils = 74)
  (h2 : num_boxes = 13)
  (h3 : max_capacity = 6) :
  ∃ (min_pencils : ℕ), 
    (∀ (box : ℕ), box ≤ num_boxes → min_pencils ≤ (total_pencils / num_boxes)) ∧
    (∃ (box : ℕ), box ≤ num_boxes ∧ (total_pencils / num_boxes) - min_pencils < 1) ∧
    min_pencils = 2 := by
  sorry

end min_pencils_in_box_l3271_327157


namespace octagon_diagonals_l3271_327111

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diagonals :
  num_diagonals 8 = 20 := by sorry

end octagon_diagonals_l3271_327111


namespace complex_inequality_l3271_327141

theorem complex_inequality (x y a b : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end complex_inequality_l3271_327141


namespace infinitely_many_pairs_divisibility_l3271_327179

theorem infinitely_many_pairs_divisibility :
  ∀ k : ℕ, ∃ n m : ℕ, (n + m)^2 / (n + 7) = k :=
by sorry

end infinitely_many_pairs_divisibility_l3271_327179


namespace problem_statement_l3271_327145

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : Real.exp a + Real.log b = 1) :
  (a + Real.log b < 0) ∧ (Real.exp a + b > 2) ∧ (a + b > 1) := by
  sorry

end problem_statement_l3271_327145


namespace parallelogram_side_length_l3271_327193

theorem parallelogram_side_length (s : ℝ) : 
  s > 0 → -- Ensure s is positive
  let side1 := 3 * s
  let side2 := s
  let angle := 30 * π / 180 -- Convert 30 degrees to radians
  let area := side1 * side2 * Real.sin angle
  area = 9 * Real.sqrt 3 → s = 3 := by
  sorry

end parallelogram_side_length_l3271_327193


namespace range_of_inequality_l3271_327181

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def monotonic_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f x < f y

def even_function_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 2) = f (-x - 2)

-- State the theorem
theorem range_of_inequality (h1 : monotonic_increasing_on_interval f) 
                            (h2 : even_function_shifted f) :
  {x : ℝ | f (2 * x) < f (x + 2)} = Set.Ioo (-2) 2 := by
  sorry

end range_of_inequality_l3271_327181


namespace fraction_equality_l3271_327156

theorem fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let x := (1/2) * (Real.sqrt (a/b) - Real.sqrt (b/a))
  (2*a * Real.sqrt (1 + x^2)) / (x + Real.sqrt (1 + x^2)) = a + b := by
  sorry

end fraction_equality_l3271_327156


namespace biology_quiz_probability_l3271_327126

/-- The number of questions in the quiz --/
def total_questions : ℕ := 20

/-- The number of questions Jessica guesses randomly --/
def guessed_questions : ℕ := 5

/-- The number of answer choices for each question --/
def answer_choices : ℕ := 4

/-- The probability of getting a single question correct by random guessing --/
def prob_correct : ℚ := 1 / answer_choices

/-- The probability of getting at least two questions correct out of five randomly guessed questions --/
def prob_at_least_two_correct : ℚ := 47 / 128

theorem biology_quiz_probability :
  (1 : ℚ) - (Nat.choose guessed_questions 0 * (1 - prob_correct)^guessed_questions +
             Nat.choose guessed_questions 1 * (1 - prob_correct)^(guessed_questions - 1) * prob_correct) =
  prob_at_least_two_correct :=
sorry

end biology_quiz_probability_l3271_327126


namespace matrix_sum_equals_result_l3271_327148

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 0; 1, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-5, -7; 4, -9]

theorem matrix_sum_equals_result : A + B = !![(-2), (-7); 5, (-7)] := by sorry

end matrix_sum_equals_result_l3271_327148


namespace soda_price_increase_l3271_327129

theorem soda_price_increase (candy_new : ℝ) (soda_new : ℝ) (candy_increase : ℝ) (total_old : ℝ)
  (h1 : candy_new = 20)
  (h2 : soda_new = 6)
  (h3 : candy_increase = 0.25)
  (h4 : total_old = 20) :
  (soda_new - (total_old - candy_new / (1 + candy_increase))) / (total_old - candy_new / (1 + candy_increase)) = 0.5 := by
  sorry

end soda_price_increase_l3271_327129


namespace difference_of_squares_l3271_327187

theorem difference_of_squares (a : ℝ) : a * a - (a - 1) * (a + 1) = 1 := by
  sorry

end difference_of_squares_l3271_327187


namespace even_increasing_function_inequality_l3271_327160

/-- An even function that is monotonically increasing on the positive reals -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y)

/-- Theorem: For an even function that is monotonically increasing on the positive reals,
    f(-3) > f(2) > f(-1) -/
theorem even_increasing_function_inequality (f : ℝ → ℝ) 
  (hf : EvenIncreasingFunction f) : f (-3) > f 2 ∧ f 2 > f (-1) := by
  sorry

end even_increasing_function_inequality_l3271_327160


namespace circular_sequence_three_elements_l3271_327147

/-- A circular sequence of distinct elements -/
structure CircularSequence (α : Type*) where
  elements : List α
  distinct : elements.Nodup
  circular : elements ≠ []

/-- Predicate to check if a CircularSequence contains zero -/
def containsZero (s : CircularSequence ℤ) : Prop :=
  0 ∈ s.elements

/-- Predicate to check if a CircularSequence has an odd number of elements -/
def hasOddElements (s : CircularSequence ℤ) : Prop :=
  s.elements.length % 2 = 1

/-- The main theorem -/
theorem circular_sequence_three_elements
  (s : CircularSequence ℤ)
  (zero_in_s : containsZero s)
  (odd_elements : hasOddElements s) :
  s.elements.length = 3 :=
sorry

end circular_sequence_three_elements_l3271_327147


namespace new_average_income_after_death_l3271_327191

/-- Calculates the new average income after a member's death -/
def new_average_income (original_members : ℕ) (original_average : ℚ) (deceased_income : ℚ) : ℚ :=
  (original_members * original_average - deceased_income) / (original_members - 1)

/-- Theorem: The new average income after a member's death is 650 -/
theorem new_average_income_after_death :
  new_average_income 4 782 1178 = 650 := by
  sorry

end new_average_income_after_death_l3271_327191


namespace scientist_news_sharing_l3271_327140

/-- Represents the state of scientists' knowledge before and after pairing -/
structure ScientistState where
  total : Nat
  initial_knowledgeable : Nat
  final_knowledgeable : Nat

/-- Probability of a specific final state given initial conditions -/
def probability (s : ScientistState) : Rat :=
  sorry

/-- Expected number of scientists knowing the news after pairing -/
def expected_final_knowledgeable (total : Nat) (initial_knowledgeable : Nat) : Rat :=
  sorry

/-- Main theorem about scientists and news sharing -/
theorem scientist_news_sharing :
  let s₁ : ScientistState := ⟨18, 10, 13⟩
  let s₂ : ScientistState := ⟨18, 10, 14⟩
  probability s₁ = 0 ∧
  probability s₂ = 1120 / 2431 ∧
  expected_final_knowledgeable 18 10 = 14^12 / 17 :=
by sorry

end scientist_news_sharing_l3271_327140
