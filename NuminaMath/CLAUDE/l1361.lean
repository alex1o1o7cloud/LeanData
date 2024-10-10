import Mathlib

namespace circle_angle_change_l1361_136119

theorem circle_angle_change (R L α r l β : ℝ) : 
  r = R / 2 → 
  l = 3 * L / 2 → 
  L = R * α → 
  l = r * β → 
  β / α = 3 := by sorry

end circle_angle_change_l1361_136119


namespace diving_survey_contradiction_l1361_136134

structure Survey where
  population : ℕ
  sample : ℕ
  topic : String

def is_sampling_survey (s : Survey) : Prop :=
  s.sample < s.population

theorem diving_survey_contradiction (s : Survey) 
  (h1 : s.population = 2000)
  (h2 : s.sample = 150)
  (h3 : s.topic = "interest in diving")
  (h4 : is_sampling_survey s) : 
  s.sample ≠ 150 := by
  sorry

end diving_survey_contradiction_l1361_136134


namespace ant_position_after_2020_moves_l1361_136120

/-- Represents the direction the ant is facing -/
inductive Direction
| East
| North
| West
| South

/-- Represents the position and state of the ant -/
structure AntState :=
  (x : Int) (y : Int) (direction : Direction) (moveCount : Nat)

/-- Function to update the ant's state after one move -/
def move (state : AntState) : AntState :=
  match state.direction with
  | Direction.East => { x := state.x + state.moveCount + 1, y := state.y, direction := Direction.North, moveCount := state.moveCount + 1 }
  | Direction.North => { x := state.x, y := state.y + state.moveCount + 1, direction := Direction.West, moveCount := state.moveCount + 1 }
  | Direction.West => { x := state.x - state.moveCount - 1, y := state.y, direction := Direction.South, moveCount := state.moveCount + 1 }
  | Direction.South => { x := state.x, y := state.y - state.moveCount - 1, direction := Direction.East, moveCount := state.moveCount + 1 }

/-- Function to update the ant's state after n moves -/
def moveN (state : AntState) (n : Nat) : AntState :=
  match n with
  | 0 => state
  | Nat.succ m => move (moveN state m)

/-- The main theorem to prove -/
theorem ant_position_after_2020_moves :
  let initialState : AntState := { x := -20, y := 20, direction := Direction.East, moveCount := 0 }
  let finalState := moveN initialState 2020
  finalState.x = -1030 ∧ finalState.y = -990 := by sorry

end ant_position_after_2020_moves_l1361_136120


namespace fixed_point_of_exponential_function_l1361_136104

theorem fixed_point_of_exponential_function 
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) - 2
  f (-2) = -1 := by sorry

end fixed_point_of_exponential_function_l1361_136104


namespace pizza_problem_l1361_136197

/-- Represents a pizza with a given number of slices and topping distribution. -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  both_toppings_slices : ℕ

/-- The pizza satisfies the given conditions. -/
def valid_pizza (p : Pizza) : Prop :=
  p.total_slices = 15 ∧
  p.pepperoni_slices = 8 ∧
  p.mushroom_slices = 12 ∧
  p.both_toppings_slices ≤ p.pepperoni_slices ∧
  p.both_toppings_slices ≤ p.mushroom_slices ∧
  p.pepperoni_slices + p.mushroom_slices - p.both_toppings_slices = p.total_slices

theorem pizza_problem (p : Pizza) (h : valid_pizza p) : p.both_toppings_slices = 5 := by
  sorry

end pizza_problem_l1361_136197


namespace greatest_integer_fraction_inequality_l1361_136103

theorem greatest_integer_fraction_inequality :
  ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 :=
by sorry

end greatest_integer_fraction_inequality_l1361_136103


namespace intersection_theorem_l1361_136121

-- Define the hyperbola and line equations
def hyperbola (x y : ℝ) : Prop := y = 9 / (x^2 + 1)
def line (x y : ℝ) : Prop := x + y = 4

-- Define the intersection points
def intersection_points : Set ℝ := {1, (3 + Real.sqrt 29) / 2, (3 - Real.sqrt 29) / 2}

-- Theorem statement
theorem intersection_theorem :
  ∀ x ∈ intersection_points, ∃ y, hyperbola x y ∧ line x y :=
by sorry

end intersection_theorem_l1361_136121


namespace bert_profit_is_one_l1361_136168

/-- Calculates the profit from a sale given the sale price, markup, and tax rate. -/
def calculate_profit (sale_price markup tax_rate : ℚ) : ℚ :=
  let purchase_price := sale_price - markup
  let tax := sale_price * tax_rate
  sale_price - purchase_price - tax

/-- Proves that the profit is $1 given the specific conditions of Bert's sale. -/
theorem bert_profit_is_one :
  calculate_profit 90 10 (1/10) = 1 := by
  sorry

end bert_profit_is_one_l1361_136168


namespace soccer_match_draw_probability_l1361_136128

theorem soccer_match_draw_probability 
  (p_win : ℝ) 
  (p_not_lose : ℝ) 
  (h1 : p_win = 0.3) 
  (h2 : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end soccer_match_draw_probability_l1361_136128


namespace dmitry_black_socks_l1361_136179

/-- Proves that Dmitry bought 22 pairs of black socks -/
theorem dmitry_black_socks :
  let initial_blue : ℕ := 10
  let initial_black : ℕ := 22
  let initial_white : ℕ := 12
  let bought_black : ℕ := x
  let total_initial : ℕ := initial_blue + initial_black + initial_white
  let total_after : ℕ := total_initial + bought_black
  let black_after : ℕ := initial_black + bought_black
  (black_after : ℚ) / (total_after : ℚ) = 2 / 3 →
  x = 22 := by
sorry

end dmitry_black_socks_l1361_136179


namespace second_square_weight_l1361_136132

/-- Represents a square piece of metal -/
structure MetalSquare where
  side_length : ℝ
  weight : ℝ

/-- The density of the metal in ounces per square inch -/
def metal_density : ℝ := 0.5

theorem second_square_weight
  (first_square : MetalSquare)
  (h1 : first_square.side_length = 4)
  (h2 : first_square.weight = 8)
  (second_square : MetalSquare)
  (h3 : second_square.side_length = 7) :
  second_square.weight = 24.5 := by
  sorry

end second_square_weight_l1361_136132


namespace sum_of_four_consecutive_integers_l1361_136133

theorem sum_of_four_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) = 34 → n = 7 := by
  sorry

end sum_of_four_consecutive_integers_l1361_136133


namespace snowman_volume_l1361_136177

theorem snowman_volume (π : ℝ) (h : π > 0) : 
  let sphere_volume (r : ℝ) := (4 / 3) * π * r^3
  sphere_volume 4 + sphere_volume 6 + sphere_volume 8 + sphere_volume 10 = (7168 / 3) * π :=
by sorry

end snowman_volume_l1361_136177


namespace greatest_whole_number_inequality_l1361_136158

theorem greatest_whole_number_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 3 * x + 2 < 5 - 2 * x :=
by sorry

end greatest_whole_number_inequality_l1361_136158


namespace different_course_choices_l1361_136191

theorem different_course_choices (n : ℕ) (k : ℕ) : n = 4 → k = 2 →
  (Nat.choose n k)^2 - (Nat.choose n k) = 30 := by
  sorry

end different_course_choices_l1361_136191


namespace complex_product_sum_l1361_136152

theorem complex_product_sum (a b : ℝ) : (1 + Complex.I) * (1 - Complex.I) = a + b * Complex.I → a + b = 2 := by
  sorry

end complex_product_sum_l1361_136152


namespace fraction_zero_implies_x_one_third_l1361_136159

theorem fraction_zero_implies_x_one_third (x : ℝ) :
  (3*x - 1) / (x^2 + 1) = 0 → x = 1/3 :=
by
  sorry

end fraction_zero_implies_x_one_third_l1361_136159


namespace quadratic_trinomial_condition_l1361_136131

/-- Given a constant m, if x^|m| + (m-2)x - 10 is a quadratic trinomial, then m = -2 -/
theorem quadratic_trinomial_condition (m : ℝ) : 
  (∃ (a b c : ℝ), ∀ x, x^(|m|) + (m-2)*x - 10 = a*x^2 + b*x + c) → m = -2 := by
  sorry

end quadratic_trinomial_condition_l1361_136131


namespace ratio_problem_l1361_136154

theorem ratio_problem (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 1 / 3) :
  t / q = 2 := by sorry

end ratio_problem_l1361_136154


namespace complement_M_union_N_when_a_is_2_M_union_N_equals_M_iff_a_in_range_l1361_136164

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x ≤ 10}
def N (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for the first part of the problem
theorem complement_M_union_N_when_a_is_2 :
  (Set.univ \ M) ∪ N 2 = {x | x > 5 ∨ x < -2 ∨ (1 ≤ x ∧ x ≤ 5)} := by sorry

-- Theorem for the second part of the problem
theorem M_union_N_equals_M_iff_a_in_range (a : ℝ) :
  M ∪ N a = M ↔ a < -1 ∨ (-1 ≤ a ∧ a ≤ 2) := by sorry

end complement_M_union_N_when_a_is_2_M_union_N_equals_M_iff_a_in_range_l1361_136164


namespace slope_angle_range_l1361_136196

/-- The range of slope angles for a line passing through (2,1) and (1,m²) where m ∈ ℝ -/
theorem slope_angle_range :
  ∀ m : ℝ, ∃ θ : ℝ, 
    (θ ∈ Set.Icc 0 (π/2) ∪ Set.Ioo (π/2) π) ∧ 
    (θ = Real.arctan ((m^2 - 1) / (2 - 1)) ∨ θ = Real.arctan ((m^2 - 1) / (2 - 1)) + π) :=
sorry

end slope_angle_range_l1361_136196


namespace sequence_property_l1361_136180

/-- Represents the number of items in the nth row of the sequence -/
def num_items (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the sum of items in the nth row of the sequence -/
def sum_items (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The row number we're interested in -/
def target_row : ℕ := 1005

/-- The target value we're trying to match -/
def target_value : ℕ := 2009^2

theorem sequence_property :
  num_items target_row = 2009 ∧ sum_items target_row = target_value := by
  sorry

end sequence_property_l1361_136180


namespace work_completion_time_l1361_136124

/-- The efficiency ratio between p and q -/
def efficiency_ratio : ℝ := 1.6

/-- The time taken by p and q working together -/
def combined_time : ℝ := 16

/-- The time taken by p working alone -/
def p_time : ℝ := 26

theorem work_completion_time :
  (efficiency_ratio * combined_time) / (efficiency_ratio + 1) = p_time := by
  sorry

end work_completion_time_l1361_136124


namespace marys_income_percentage_l1361_136138

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.7)
  (h2 : mary = tim * 1.6) :
  mary = juan * 1.12 := by
sorry

end marys_income_percentage_l1361_136138


namespace spoiled_milk_percentage_l1361_136144

theorem spoiled_milk_percentage
  (egg_rotten_percentage : ℝ)
  (flour_weevil_percentage : ℝ)
  (all_good_probability : ℝ)
  (h1 : egg_rotten_percentage = 60)
  (h2 : flour_weevil_percentage = 25)
  (h3 : all_good_probability = 24) :
  ∃ (spoiled_milk_percentage : ℝ),
    spoiled_milk_percentage = 20 ∧
    (100 - spoiled_milk_percentage) / 100 * (100 - egg_rotten_percentage) / 100 * (100 - flour_weevil_percentage) / 100 = all_good_probability / 100 :=
by sorry

end spoiled_milk_percentage_l1361_136144


namespace prime_fraction_equation_l1361_136137

theorem prime_fraction_equation (p q r : ℕ+) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (1 : ℚ) / (p + 1) + (1 : ℚ) / (q + 1) - 1 / ((p + 1) * (q + 1)) = 1 / r →
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) ∧ r = 2 := by
  sorry

end prime_fraction_equation_l1361_136137


namespace arithmetic_sequence_sum_find_S_5_l1361_136156

/-- Given an arithmetic sequence {aₙ}, Sₙ represents the sum of its first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- aₙ represents the nth term of the arithmetic sequence -/
def a (n : ℕ) : ℝ := sorry

/-- d represents the common difference of the arithmetic sequence -/
def d : ℝ := sorry

theorem arithmetic_sequence_sum (n : ℕ) :
  S n = n * a 1 + (n * (n - 1) / 2) * d := sorry

axiom sum_condition : S 3 + S 6 = 18

theorem find_S_5 : S 5 = 10 := by sorry

end arithmetic_sequence_sum_find_S_5_l1361_136156


namespace fraction_decomposition_l1361_136171

theorem fraction_decomposition (x : ℚ) (A B : ℚ) 
  (h : x ≠ -5 ∧ x ≠ 2/3) : 
  (7 * x - 13) / (3 * x^2 + 13 * x - 10) = A / (x + 5) + B / (3 * x - 2) → 
  A = 48/17 ∧ B = -25/17 :=
by sorry

end fraction_decomposition_l1361_136171


namespace football_team_size_l1361_136147

theorem football_team_size :
  ∀ (P : ℕ),
  (49 : ℕ) ≤ P →  -- There are at least 49 throwers
  (63 : ℕ) ≤ P →  -- There are at least 63 right-handed players
  (P - 49) % 3 = 0 →  -- The non-throwers can be divided into thirds
  63 = 49 + (2 * (P - 49) / 3) →  -- Right-handed players equation
  P = 70 :=
by
  sorry

end football_team_size_l1361_136147


namespace complex_fraction_value_l1361_136143

theorem complex_fraction_value : Complex.I / (1 - Complex.I)^2 = -1/2 := by
  sorry

end complex_fraction_value_l1361_136143


namespace least_value_of_x_l1361_136199

theorem least_value_of_x (x p q : ℕ) : 
  x > 0 →
  Nat.Prime p →
  Nat.Prime q →
  p < q →
  x / (12 * p * q) = 2 →
  2 * p - q = 3 →
  ∀ y, y > 0 ∧ 
       ∃ p' q', Nat.Prime p' ∧ Nat.Prime q' ∧ p' < q' ∧
                y / (12 * p' * q') = 2 ∧
                2 * p' - q' = 3 →
       x ≤ y →
  x = 840 := by
sorry


end least_value_of_x_l1361_136199


namespace sum_of_powers_l1361_136169

theorem sum_of_powers (a b : ℝ) : 
  (a + b = 1) → 
  (a^2 + b^2 = 3) → 
  (a^3 + b^3 = 4) → 
  (a^4 + b^4 = 7) → 
  (a^5 + b^5 = 11) → 
  (∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) →
  a^6 + b^6 = 18 := by
sorry

end sum_of_powers_l1361_136169


namespace division_problem_l1361_136100

theorem division_problem (total : ℕ) (p q r : ℕ) : 
  total = 1210 →
  p * 4 = q * 5 →
  q * 10 = r * 9 →
  p + q + r = total →
  r = 400 := by
sorry

end division_problem_l1361_136100


namespace pancake_covers_center_l1361_136186

-- Define the pan
def Pan : Type := Unit

-- Define the area of the pan
def pan_area : ℝ := 1

-- Define the pancake
def Pancake : Type := Unit

-- Define the property of the pancake being convex
def is_convex (p : Pancake) : Prop := sorry

-- Define the area of the pancake
def pancake_area (p : Pancake) : ℝ := sorry

-- Define the center of the pan
def pan_center (pan : Pan) : Set ℝ := sorry

-- Define the region covered by the pancake
def pancake_region (p : Pancake) : Set ℝ := sorry

-- The theorem to be proved
theorem pancake_covers_center (pan : Pan) (p : Pancake) :
  is_convex p →
  pancake_area p > 1/2 →
  pan_center pan ⊆ pancake_region p :=
sorry

end pancake_covers_center_l1361_136186


namespace class_composition_l1361_136110

theorem class_composition (total : ℕ) (boys girls : ℕ) : 
  total = 20 →
  boys + girls = total →
  (boys : ℚ) / total = 3/4 * ((girls : ℚ) / total) →
  boys = 12 ∧ girls = 8 :=
by
  sorry

end class_composition_l1361_136110


namespace angle_between_vectors_l1361_136126

/-- Given two unit vectors a and b in a real inner product space,
    prove that the angle between them is 2π/3 if |a-2b| = √7 -/
theorem angle_between_vectors 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (h : ‖a - 2 • b‖ = Real.sqrt 7) :
  Real.arccos (inner a b) = 2 * Real.pi / 3 := by
  sorry

#check angle_between_vectors

end angle_between_vectors_l1361_136126


namespace second_graders_borrowed_books_l1361_136114

theorem second_graders_borrowed_books (initial_books : ℕ) (remaining_books : ℕ) 
  (h1 : initial_books = 75) 
  (h2 : remaining_books = 57) : 
  initial_books - remaining_books = 18 := by
  sorry

end second_graders_borrowed_books_l1361_136114


namespace production_line_b_l1361_136175

def total_production : ℕ := 5000

def sampling_ratio : List ℕ := [1, 2, 2]

theorem production_line_b (a b c : ℕ) : 
  a + b + c = total_production →
  [a, b, c] = sampling_ratio.map (λ x => x * (total_production / sampling_ratio.sum)) →
  b = 2000 := by
  sorry

end production_line_b_l1361_136175


namespace singing_competition_ratio_l1361_136185

/-- Proves that the ratio of female contestants to the total number of contestants is 1/3 -/
theorem singing_competition_ratio :
  let total_contestants : ℕ := 18
  let male_contestants : ℕ := 12
  let female_contestants : ℕ := total_contestants - male_contestants
  (female_contestants : ℚ) / total_contestants = 1 / 3 := by
  sorry

end singing_competition_ratio_l1361_136185


namespace sugar_calculation_l1361_136135

/-- The amount of sugar needed for one chocolate bar in grams -/
def sugar_per_bar : ℝ := 1.5

/-- The number of chocolate bars produced per minute -/
def bars_per_minute : ℕ := 36

/-- The number of minutes of production -/
def production_time : ℕ := 2

/-- Calculates the total amount of sugar used in grams -/
def total_sugar_used : ℝ := sugar_per_bar * bars_per_minute * production_time

theorem sugar_calculation :
  total_sugar_used = 108 := by sorry

end sugar_calculation_l1361_136135


namespace pages_per_day_l1361_136139

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 96) (h2 : days = 12) :
  total_pages / days = 8 := by
sorry

end pages_per_day_l1361_136139


namespace inequality_solution_set_l1361_136188

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end inequality_solution_set_l1361_136188


namespace sine_function_omega_values_l1361_136187

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is symmetric about a point (a, b) if f(a + x) = f(a - x) for all x -/
def IsSymmetricAbout (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

/-- A function f is monotonic on an interval [a, b] if it is either increasing or decreasing on that interval -/
def IsMonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem sine_function_omega_values 
  (ω φ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : f = fun x ↦ Real.sin (ω * x + φ))
  (h2 : ω > 0)
  (h3 : 0 ≤ φ ∧ φ ≤ π)
  (h4 : IsEven f)
  (h5 : IsSymmetricAbout f (3 * π / 4) 0)
  (h6 : IsMonotonicOn f 0 (π / 2)) :
  ω = 2/3 ∨ ω = 2 := by
  sorry

end sine_function_omega_values_l1361_136187


namespace max_badminton_rackets_l1361_136198

theorem max_badminton_rackets 
  (table_tennis_price badminton_price : ℕ)
  (total_rackets : ℕ)
  (max_expenditure : ℕ)
  (h1 : 2 * table_tennis_price + badminton_price = 220)
  (h2 : 3 * table_tennis_price + 2 * badminton_price = 380)
  (h3 : total_rackets = 30)
  (h4 : ∀ m : ℕ, m ≤ total_rackets → 
        (total_rackets - m) * table_tennis_price + m * badminton_price ≤ max_expenditure) :
  ∃ max_badminton : ℕ, 
    max_badminton ≤ total_rackets ∧
    (total_rackets - max_badminton) * table_tennis_price + max_badminton * badminton_price ≤ max_expenditure ∧
    ∀ n : ℕ, n > max_badminton → 
      (total_rackets - n) * table_tennis_price + n * badminton_price > max_expenditure :=
by
  sorry

end max_badminton_rackets_l1361_136198


namespace asymptote_sum_l1361_136117

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x^3 + A*x^2 + B*x + C = (x + 1)*(x - 3)*(x - 4)) →
  A + B + C = 11 := by
sorry

end asymptote_sum_l1361_136117


namespace geometric_sequence_log_sum_l1361_136136

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_log_sum 
  (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a) 
  (h_prod : a 2 * a 5 = 10) : 
  Real.log (a 3) + Real.log (a 4) = 1 :=
sorry

end geometric_sequence_log_sum_l1361_136136


namespace school_boys_count_l1361_136116

theorem school_boys_count (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) (other_count : ℕ) :
  muslim_percent = 44/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  other_count = 117 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℚ) / total = 1) ∧
    total = 650 := by
  sorry

end school_boys_count_l1361_136116


namespace unique_B_for_divisibility_l1361_136151

/-- Represents a four-digit number in the form 4BB2 -/
def fourDigitNumber (B : ℕ) : ℕ := 4000 + 100 * B + 10 * B + 2

/-- Checks if a number is divisible by 11 -/
def divisibleBy11 (n : ℕ) : Prop := n % 11 = 0

/-- B is a single digit -/
def isSingleDigit (B : ℕ) : Prop := B ≥ 0 ∧ B ≤ 9

theorem unique_B_for_divisibility : 
  ∃! B : ℕ, isSingleDigit B ∧ divisibleBy11 (fourDigitNumber B) ∧ B = 3 :=
sorry

end unique_B_for_divisibility_l1361_136151


namespace minimum_draws_for_divisible_by_3_or_5_l1361_136172

theorem minimum_draws_for_divisible_by_3_or_5 (n : ℕ) (hn : n = 90) :
  let divisible_by_3_or_5 (k : ℕ) := k % 3 = 0 ∨ k % 5 = 0
  let count_divisible := (Finset.range n).filter divisible_by_3_or_5 |>.card
  49 = n - count_divisible + 1 :=
by sorry

end minimum_draws_for_divisible_by_3_or_5_l1361_136172


namespace investment_partnership_profit_share_l1361_136106

/-- Investment partnership problem -/
theorem investment_partnership_profit_share
  (investment_B : ℝ)
  (investment_A : ℝ)
  (investment_C : ℝ)
  (investment_D : ℝ)
  (time_A : ℝ)
  (time_B : ℝ)
  (time_C : ℝ)
  (time_D : ℝ)
  (total_profit : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : investment_B = 2 / 3 * investment_C)
  (h3 : investment_D = 1 / 2 * (investment_A + investment_B + investment_C))
  (h4 : time_A = 6)
  (h5 : time_B = 9)
  (h6 : time_C = 12)
  (h7 : time_D = 4)
  (h8 : total_profit = 22000) :
  (investment_B * time_B) / (investment_A * time_A + investment_B * time_B + investment_C * time_C + investment_D * time_D) * total_profit = 3666.67 := by
  sorry

end investment_partnership_profit_share_l1361_136106


namespace cupcake_dozens_correct_l1361_136178

/-- The number of dozens of cupcakes Jose needs to make -/
def cupcake_dozens : ℕ := 3

/-- The number of tablespoons of lemon juice needed for one dozen cupcakes -/
def juice_per_dozen : ℕ := 12

/-- The number of tablespoons of lemon juice provided by one lemon -/
def juice_per_lemon : ℕ := 4

/-- The number of lemons Jose needs -/
def lemons_needed : ℕ := 9

/-- Theorem stating that the number of dozens of cupcakes Jose needs to make is correct -/
theorem cupcake_dozens_correct : 
  cupcake_dozens = (lemons_needed * juice_per_lemon) / juice_per_dozen :=
by sorry

end cupcake_dozens_correct_l1361_136178


namespace fifteenth_triangular_number_l1361_136174

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem fifteenth_triangular_number : triangular_number 15 = 120 := by
  sorry

end fifteenth_triangular_number_l1361_136174


namespace logo_shaded_area_l1361_136125

/-- Calculates the shaded area of a logo design with a rectangle and four tangent circles -/
theorem logo_shaded_area (length width : ℝ) (h1 : length = 30) (h2 : width = 15) : 
  let rectangle_area := length * width
  let circle_radius := width / 4
  let circle_area := π * circle_radius^2
  let total_circle_area := 4 * circle_area
  rectangle_area - total_circle_area = 450 - 56.25 * π := by
  sorry

end logo_shaded_area_l1361_136125


namespace complex_conjugate_roots_imply_zero_coefficients_l1361_136155

theorem complex_conjugate_roots_imply_zero_coefficients 
  (c d : ℝ) 
  (h : ∃ (u v : ℝ), (Complex.I * v + u)^2 + (15 + Complex.I * c) * (Complex.I * v + u) + (35 + Complex.I * d) = 0 ∧ 
                     (Complex.I * -v + u)^2 + (15 + Complex.I * c) * (Complex.I * -v + u) + (35 + Complex.I * d) = 0) : 
  c = 0 ∧ d = 0 := by
sorry

end complex_conjugate_roots_imply_zero_coefficients_l1361_136155


namespace equilateral_triangle_perimeter_area_ratio_l1361_136148

/-- The ratio of the perimeter to the area of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_perimeter_area_ratio :
  let side_length : ℝ := 6
  let perimeter : ℝ := 3 * side_length
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  perimeter / area = 2 * Real.sqrt 3 / 3 := by sorry

end equilateral_triangle_perimeter_area_ratio_l1361_136148


namespace min_value_weighted_sum_l1361_136183

theorem min_value_weighted_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 2*x + 3*y + 4*z = 1) :
  (4/x) + (9/y) + (8/z) ≥ 81 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    2*x₀ + 3*y₀ + 4*z₀ = 1 ∧ (4/x₀) + (9/y₀) + (8/z₀) = 81 := by
  sorry

end min_value_weighted_sum_l1361_136183


namespace quartic_trinomial_m_value_l1361_136127

theorem quartic_trinomial_m_value (m : ℤ) : 
  (abs (m - 3) = 4) → (m - 7 ≠ 0) → m = -1 := by
  sorry

end quartic_trinomial_m_value_l1361_136127


namespace power_of_power_of_three_l1361_136108

theorem power_of_power_of_three : (3 : ℕ) ^ ((3 : ℕ) ^ (3 : ℕ)) = (3 : ℕ) ^ (27 : ℕ) := by
  sorry

end power_of_power_of_three_l1361_136108


namespace arithmetic_sequence_sum_l1361_136173

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  (a 2 + a 11 = 3) → (a 5 + a 8 = 3) := by
  sorry

end arithmetic_sequence_sum_l1361_136173


namespace hyperbola_standard_equation_l1361_136105

/-- The standard equation of a hyperbola with foci on the y-axis, given a + c = 9 and b = 3 -/
theorem hyperbola_standard_equation (a c b : ℝ) (h1 : a + c = 9) (h2 : b = 3) :
  ∃ (x y : ℝ), y^2 / 16 - x^2 / 9 = 1 := by
  sorry

end hyperbola_standard_equation_l1361_136105


namespace corveus_weekly_lack_of_sleep_l1361_136149

/-- Calculates the weekly lack of sleep given actual and recommended daily sleep hours -/
def weeklyLackOfSleep (actualSleep recommendedSleep : ℕ) : ℕ :=
  (recommendedSleep - actualSleep) * 7

/-- Proves that Corveus lacks 14 hours of sleep in a week -/
theorem corveus_weekly_lack_of_sleep :
  weeklyLackOfSleep 4 6 = 14 := by
  sorry

end corveus_weekly_lack_of_sleep_l1361_136149


namespace surtido_criterion_l1361_136153

def sum_of_digits (A : ℕ) : ℕ := sorry

def is_sum_of_digits (A n : ℕ) : Prop := sorry

def is_surtido (A : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ sum_of_digits A → is_sum_of_digits A k

theorem surtido_criterion (A : ℕ) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 8 → is_sum_of_digits A k) → is_surtido A := by sorry

end surtido_criterion_l1361_136153


namespace printer_problem_l1361_136189

/-- Calculates the time needed to print a given number of pages with breaks -/
def print_time (pages_per_minute : ℕ) (total_pages : ℕ) (pages_before_break : ℕ) (break_duration : ℕ) : ℕ :=
  let full_segments := total_pages / pages_before_break
  let remaining_pages := total_pages % pages_before_break
  let printing_time := (full_segments * pages_before_break + remaining_pages) / pages_per_minute
  let break_time := full_segments * break_duration
  printing_time + break_time

theorem printer_problem :
  print_time 25 350 150 5 = 24 := by
sorry

end printer_problem_l1361_136189


namespace quadratic_positivity_l1361_136109

theorem quadratic_positivity (a : ℝ) : 
  (∀ x, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
by sorry

end quadratic_positivity_l1361_136109


namespace partnership_gain_l1361_136107

/-- Represents the investment and profit share of a partner in the partnership. -/
structure Partner where
  investment : ℕ  -- Amount invested
  duration : ℕ    -- Duration of investment in months
  share : ℕ       -- Share of profit

/-- Represents the partnership with three partners. -/
structure Partnership where
  a : Partner
  b : Partner
  c : Partner

/-- Calculates the total annual gain of the partnership. -/
def totalAnnualGain (p : Partnership) : ℕ :=
  p.a.share + p.b.share + p.c.share

/-- Theorem stating the total annual gain of the partnership. -/
theorem partnership_gain (p : Partnership) 
  (h1 : p.a.investment > 0)
  (h2 : p.b.investment = 2 * p.a.investment)
  (h3 : p.c.investment = 3 * p.a.investment)
  (h4 : p.a.duration = 12)
  (h5 : p.b.duration = 6)
  (h6 : p.c.duration = 4)
  (h7 : p.a.share = 6100)
  (h8 : p.a.share = p.b.share)
  (h9 : p.b.share = p.c.share) :
  totalAnnualGain p = 18300 := by
  sorry

end partnership_gain_l1361_136107


namespace flat_cost_calculation_l1361_136115

theorem flat_cost_calculation (x : ℝ) : 
  x > 0 →  -- Assuming the cost is positive
  0.11 * x - (-0.11 * x) = 1.21 →
  x = 5.50 := by
  sorry

end flat_cost_calculation_l1361_136115


namespace distance_between_points_l1361_136190

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-2, 4)
  let p2 : ℝ × ℝ := (3, -8)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 13 := by
  sorry

end distance_between_points_l1361_136190


namespace ramsey_theorem_for_interns_l1361_136101

/-- Represents the relationship between two interns -/
inductive Relationship
  | Knows
  | DoesNotKnow

/-- Defines a group of interns and their relationships -/
structure InternGroup :=
  (size : Nat)
  (relationships : Fin size → Fin size → Relationship)

/-- The main theorem -/
theorem ramsey_theorem_for_interns (group : InternGroup) (h : group.size = 6) :
  ∃ (a b c : Fin group.size),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    ((group.relationships a b = Relationship.Knows ∧
      group.relationships b c = Relationship.Knows ∧
      group.relationships a c = Relationship.Knows) ∨
     (group.relationships a b = Relationship.DoesNotKnow ∧
      group.relationships b c = Relationship.DoesNotKnow ∧
      group.relationships a c = Relationship.DoesNotKnow)) :=
sorry

end ramsey_theorem_for_interns_l1361_136101


namespace smallest_n_for_B_exceeds_A_l1361_136112

def A (n : ℕ) : ℚ := 490 * n - 10 * n^2

def B (n : ℕ) : ℚ := 500 * n + 400 - 500 / 2^(n-1)

theorem smallest_n_for_B_exceeds_A :
  ∀ k : ℕ, k < 4 → B k ≤ A k ∧ B 4 > A 4 := by sorry

end smallest_n_for_B_exceeds_A_l1361_136112


namespace equation_solution_l1361_136113

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (7 * x / (x - 2) - 5 / (x - 2) = 2 / (x - 2)) ↔ x = 1 := by
  sorry

end equation_solution_l1361_136113


namespace volume_ratio_cone_cylinder_l1361_136111

/-- Given a cylinder and a cone with the same radius, where the cone's height is 1/3 of the cylinder's height,
    the ratio of the volume of the cone to the volume of the cylinder is 1/9. -/
theorem volume_ratio_cone_cylinder (r h : ℝ) (h_pos : 0 < r) (h_height : 0 < h) :
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry


end volume_ratio_cone_cylinder_l1361_136111


namespace minimum_pages_per_day_l1361_136192

theorem minimum_pages_per_day (total_pages : ℕ) (days : ℕ) (pages_per_day : ℕ) : 
  total_pages = 220 → days = 7 → 
  (pages_per_day * days ≥ total_pages ∧ 
   ∀ n : ℕ, n * days ≥ total_pages → n ≥ pages_per_day) →
  pages_per_day = 32 := by
sorry

end minimum_pages_per_day_l1361_136192


namespace circle_center_and_radius_l1361_136176

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the circle: x^2 + y^2 - 4x = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x = 0

/-- The circle defined by the equation -/
def given_circle : Circle :=
  { center := (2, 0),
    radius := 2 }

/-- Theorem: The given equation defines a circle with center (2, 0) and radius 2 -/
theorem circle_center_and_radius :
  ∀ x y : ℝ, circle_equation x y ↔ 
    (x - given_circle.center.1)^2 + (y - given_circle.center.2)^2 = given_circle.radius^2 :=
by sorry

end circle_center_and_radius_l1361_136176


namespace quadratic_completion_l1361_136181

theorem quadratic_completion (y : ℝ) : ∃ (k : ℤ) (a : ℝ), y^2 + 12*y + 40 = (y + a)^2 + k := by
  sorry

end quadratic_completion_l1361_136181


namespace shelter_cat_count_l1361_136184

/-- Proves that the total number of cats and kittens in the shelter is 280 --/
theorem shelter_cat_count : ∀ (adult_cats female_cats litters kittens_per_litter : ℕ),
  adult_cats = 120 →
  female_cats = 2 * adult_cats / 3 →
  litters = 2 * female_cats / 5 →
  kittens_per_litter = 5 →
  adult_cats + litters * kittens_per_litter = 280 :=
by
  sorry

end shelter_cat_count_l1361_136184


namespace factorization_problem_1_factorization_problem_2_l1361_136150

-- Problem 1
theorem factorization_problem_1 (x : ℝ) : 2*x^2 - 4*x = 2*x*(x - 2) := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) : x*y^2 - 2*x*y + x = x*(y - 1)^2 := by sorry

end factorization_problem_1_factorization_problem_2_l1361_136150


namespace remainder_of_196c_pow_2008_mod_97_l1361_136122

theorem remainder_of_196c_pow_2008_mod_97 (c : ℤ) : (196 * c)^2008 % 97 = 44 := by
  sorry

end remainder_of_196c_pow_2008_mod_97_l1361_136122


namespace sum_of_median_scores_l1361_136129

-- Define the type for basketball scores
def Score := ℕ

-- Define a function to calculate the median of a list of scores
noncomputable def median (scores : List Score) : ℝ := sorry

-- Define the scores for player A
def scoresA : List Score := sorry

-- Define the scores for player B
def scoresB : List Score := sorry

-- Theorem to prove
theorem sum_of_median_scores : 
  median scoresA + median scoresB = 64 := by sorry

end sum_of_median_scores_l1361_136129


namespace arithmetic_expression_equality_l1361_136140

theorem arithmetic_expression_equality : (11 * 24 - 23 * 9) / 3 + 3 = 22 := by
  sorry

end arithmetic_expression_equality_l1361_136140


namespace line_passes_first_and_fourth_quadrants_l1361_136193

/-- A line passes through the first quadrant if there exists a point (x, y) on the line where both x and y are positive. -/
def passes_through_first_quadrant (k b : ℝ) : Prop :=
  ∃ x > 0, k * x + b > 0

/-- A line passes through the fourth quadrant if there exists a point (x, y) on the line where x is positive and y is negative. -/
def passes_through_fourth_quadrant (k b : ℝ) : Prop :=
  ∃ x > 0, k * x + b < 0

/-- If bk < 0, then the line y = kx + b passes through both the first and fourth quadrants. -/
theorem line_passes_first_and_fourth_quadrants (k b : ℝ) (h : b * k < 0) :
  passes_through_first_quadrant k b ∧ passes_through_fourth_quadrant k b :=
sorry

end line_passes_first_and_fourth_quadrants_l1361_136193


namespace simplify_expression_l1361_136118

theorem simplify_expression (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a ≥ b) :
  (a - b) / (Real.sqrt a + Real.sqrt b) + (a * Real.sqrt a + b * Real.sqrt b) / (a - Real.sqrt (a * b) + b) = 2 * Real.sqrt a :=
by sorry

end simplify_expression_l1361_136118


namespace midpoint_rectangle_area_l1361_136130

/-- Given a rectangle with area 48 and length-to-width ratio 3:2, 
    the area of the rectangle formed by connecting its side midpoints is 12. -/
theorem midpoint_rectangle_area (length width : ℝ) : 
  length * width = 48 →
  length / width = 3 / 2 →
  (length / 2) * (width / 2) = 12 := by
  sorry

end midpoint_rectangle_area_l1361_136130


namespace shirt_price_l1361_136160

/-- Given the sales of shoes and shirts, prove the price of each shirt -/
theorem shirt_price (num_shoes : ℕ) (shoe_price : ℚ) (num_shirts : ℕ) (total_earnings_per_person : ℚ) :
  num_shoes = 6 →
  shoe_price = 3 →
  num_shirts = 18 →
  total_earnings_per_person = 27 →
  ∃ (shirt_price : ℚ), 
    (↑num_shoes * shoe_price + ↑num_shirts * shirt_price) / 2 = total_earnings_per_person ∧
    shirt_price = 2 :=
by sorry

end shirt_price_l1361_136160


namespace toothpicks_in_45x25_grid_with_gaps_l1361_136123

/-- Calculates the number of effective lines in a grid with gaps every fifth line -/
def effectiveLines (total : ℕ) : ℕ :=
  total + 1 - (total + 1) / 5

/-- Calculates the total number of toothpicks in a rectangular grid with gaps -/
def toothpicksInGrid (length width : ℕ) : ℕ :=
  let verticalLines := effectiveLines length
  let horizontalLines := effectiveLines width
  verticalLines * width + horizontalLines * length

/-- Theorem: A 45x25 grid with every fifth row and column missing uses 1722 toothpicks -/
theorem toothpicks_in_45x25_grid_with_gaps :
  toothpicksInGrid 45 25 = 1722 := by
  sorry

#eval toothpicksInGrid 45 25

end toothpicks_in_45x25_grid_with_gaps_l1361_136123


namespace a_equals_one_sufficient_not_necessary_l1361_136165

/-- A complex number is pure imaginary if its real part is zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem a_equals_one_sufficient_not_necessary :
  ∃ (a : ℝ),
    (a = 1 → is_pure_imaginary ((a - 1) * (a + 2) + (a + 3) * Complex.I)) ∧
    (∃ (b : ℝ), b ≠ 1 ∧ is_pure_imaginary ((b - 1) * (b + 2) + (b + 3) * Complex.I)) :=
by sorry

end a_equals_one_sufficient_not_necessary_l1361_136165


namespace square_difference_of_integers_l1361_136170

theorem square_difference_of_integers (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 20) :
  a^2 - b^2 = 1200 := by
  sorry

end square_difference_of_integers_l1361_136170


namespace square_difference_l1361_136167

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) :
  (x - y)^2 = 9 := by
  sorry

end square_difference_l1361_136167


namespace basketball_team_selection_l1361_136182

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the number of quadruplets that must be in the starting lineup
def quadruplets_in_lineup : ℕ := 3

-- Theorem statement
theorem basketball_team_selection :
  (Nat.choose num_quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)) = 1980 := by
  sorry

end basketball_team_selection_l1361_136182


namespace length_AB_is_6_l1361_136142

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right angle at A
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- Angle B is 45°
  (C.2 - B.2) / (C.1 - B.1) = 1 ∧
  -- BC = 6√2
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 6 * Real.sqrt 2

-- Theorem statement
theorem length_AB_is_6 (A B C : ℝ × ℝ) (h : Triangle A B C) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 6 :=
sorry

end length_AB_is_6_l1361_136142


namespace complex_power_sum_l1361_136146

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (Real.pi/4)) :
  z^8 + 1/z^8 = 2 := by sorry

end complex_power_sum_l1361_136146


namespace quadratic_function_range_l1361_136194

/-- Given a quadratic function y = -x^2 + 2ax + a + 1, if y > a + 1 for all x in (-1, a),
    then -1 < a ≤ -1/2 -/
theorem quadratic_function_range (a : ℝ) :
  (∀ x, -1 < x ∧ x < a → -x^2 + 2*a*x + a + 1 > a + 1) →
  -1 < a ∧ a ≤ -1/2 := by
  sorry

end quadratic_function_range_l1361_136194


namespace second_to_last_digit_even_for_valid_numbers_l1361_136102

def ends_in_valid_digit (k : ℕ) : Prop :=
  k % 10 = 1 ∨ k % 10 = 3 ∨ k % 10 = 7 ∨ k % 10 = 9 ∨ k % 10 = 5 ∨ k % 10 = 0

def second_to_last_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem second_to_last_digit_even_for_valid_numbers (k n : ℕ) 
  (h : ends_in_valid_digit k) : 
  Even (second_to_last_digit (k^n)) :=
sorry

end second_to_last_digit_even_for_valid_numbers_l1361_136102


namespace investment_rate_proof_l1361_136163

-- Define the given values
def total_investment : ℚ := 12000
def first_investment : ℚ := 5000
def first_rate : ℚ := 3 / 100
def second_investment : ℚ := 3000
def second_return : ℚ := 90
def desired_income : ℚ := 480

-- Define the theorem
theorem investment_rate_proof :
  let remaining_investment := total_investment - first_investment - second_investment
  let known_income := first_investment * first_rate + second_return
  let required_income := desired_income - known_income
  let rate := required_income / remaining_investment
  rate = 6 / 100 := by sorry

end investment_rate_proof_l1361_136163


namespace complex_equation_solution_l1361_136157

/-- Given the complex equation (1+2i)a + b = 2i, where a and b are real numbers, prove that a = 1 and b = -1. -/
theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * 2 + 1 * (a : ℂ) + (b : ℂ) = (Complex.I : ℂ) * 2 → a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l1361_136157


namespace octagon_diagonals_l1361_136145

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by sorry

end octagon_diagonals_l1361_136145


namespace units_digit_of_first_four_composites_product_l1361_136195

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def first_four_composites : List ℕ := [4, 6, 8, 9]

theorem units_digit_of_first_four_composites_product :
  (first_four_composites.prod % 10 = 8) ∧
  (∀ n ∈ first_four_composites, is_composite n) ∧
  (∀ m, is_composite m → m ≥ 4 → ∃ n ∈ first_four_composites, n ≤ m) :=
by sorry

end units_digit_of_first_four_composites_product_l1361_136195


namespace inequality_solution_l1361_136141

theorem inequality_solution (m n : ℤ) : 
  (∀ x : ℝ, x > 0 → (m * x + 5) * (x^2 - n) ≤ 0) →
  (m + n ∈ ({-4, 24} : Set ℤ)) :=
by sorry

end inequality_solution_l1361_136141


namespace smallest_multiple_35_with_digit_product_35_l1361_136162

def is_multiple_of_35 (n : ℕ) : Prop := ∃ k : ℕ, n = 35 * k

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  let digits := n.digits 10
  digits.prod

theorem smallest_multiple_35_with_digit_product_35 :
  ∀ n : ℕ, n > 0 → is_multiple_of_35 n → is_multiple_of_35 (digit_product n) →
  n ≥ 735 := by sorry

end smallest_multiple_35_with_digit_product_35_l1361_136162


namespace smallest_positive_multiple_of_45_l1361_136166

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 * 1 ≤ 45 * n := by sorry

end smallest_positive_multiple_of_45_l1361_136166


namespace time_to_cook_one_potato_l1361_136161

theorem time_to_cook_one_potato 
  (total_potatoes : ℕ) 
  (cooked_potatoes : ℕ) 
  (time_for_remaining : ℕ) : 
  total_potatoes = 13 → 
  cooked_potatoes = 5 → 
  time_for_remaining = 48 → 
  (time_for_remaining / (total_potatoes - cooked_potatoes) : ℚ) = 6 := by
  sorry

end time_to_cook_one_potato_l1361_136161
