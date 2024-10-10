import Mathlib

namespace log_2_15_in_terms_of_a_b_l1091_109124

/-- Given a = log₃6 and b = log₅20, prove that log₂15 = (2a + b - 3) / ((a - 1)(b - 1)) -/
theorem log_2_15_in_terms_of_a_b (a b : ℝ) 
  (ha : a = Real.log 6 / Real.log 3) 
  (hb : b = Real.log 20 / Real.log 5) : 
  Real.log 15 / Real.log 2 = (2 * a + b - 3) / ((a - 1) * (b - 1)) := by
  sorry

end log_2_15_in_terms_of_a_b_l1091_109124


namespace geometric_sequence_fourth_term_l1091_109107

theorem geometric_sequence_fourth_term (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence condition
  q = 2 →                       -- Common ratio is 2
  a 1 * a 3 = 6 * a 2 →         -- Given condition
  a 4 = 24 :=                   -- Conclusion to prove
by
  sorry

end geometric_sequence_fourth_term_l1091_109107


namespace intersection_implies_m_equals_three_subset_complement_implies_m_range_l1091_109165

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x - 4 ≤ 0}

-- Theorem 1
theorem intersection_implies_m_equals_three (m : ℝ) :
  A ∩ B m = {x | 1 ≤ x ∧ x ≤ 3} → m = 3 := by sorry

-- Theorem 2
theorem subset_complement_implies_m_range (m : ℝ) :
  A ⊆ (B m)ᶜ → m < -3 ∨ m > 5 := by sorry

end intersection_implies_m_equals_three_subset_complement_implies_m_range_l1091_109165


namespace average_speed_calculation_l1091_109116

/-- Calculate the average speed of a round trip with different speeds and wind conditions -/
theorem average_speed_calculation (outward_speed return_speed : ℝ) 
  (tailwind headwind : ℝ) : 
  outward_speed = 110 →
  tailwind = 15 →
  return_speed = 72 →
  headwind = 10 →
  (2 * (outward_speed + tailwind) * (return_speed - headwind)) / 
  ((outward_speed + tailwind) + (return_speed - headwind)) = 250 * 62 / 187 :=
by sorry

end average_speed_calculation_l1091_109116


namespace job_productivity_solution_l1091_109101

/-- Represents the productivity of workers on a job -/
structure JobProductivity where
  workers : ℕ
  hours_per_day : ℕ
  days_to_complete : ℕ

/-- The job productivity satisfies the given conditions -/
def satisfies_conditions (jp : JobProductivity) : Prop :=
  ∃ (t : ℝ),
    -- Condition 1: Initial setup
    jp.workers * jp.hours_per_day * jp.days_to_complete * t = jp.workers * jp.hours_per_day * 14 * t ∧
    -- Condition 2: 4 more workers, 1 hour longer, 10 days
    jp.workers * jp.hours_per_day * 14 * t = (jp.workers + 4) * (jp.hours_per_day + 1) * 10 * t ∧
    -- Condition 3: 10 more workers, 2 hours longer, 7 days
    jp.workers * jp.hours_per_day * 14 * t = (jp.workers + 10) * (jp.hours_per_day + 2) * 7 * t

/-- The theorem to be proved -/
theorem job_productivity_solution :
  ∃ (jp : JobProductivity),
    satisfies_conditions jp ∧ jp.workers = 20 ∧ jp.hours_per_day = 6 :=
sorry

end job_productivity_solution_l1091_109101


namespace parabola_standard_equation_l1091_109156

/-- The standard equation of a parabola with focus (2,0) is y^2 = 8x -/
theorem parabola_standard_equation (F : ℝ × ℝ) (h : F = (2, 0)) :
  ∃ (f : ℝ → ℝ), (∀ x y : ℝ, f y = 8*x ↔ y^2 = 8*x) := by
  sorry

end parabola_standard_equation_l1091_109156


namespace min_f_at_75_l1091_109141

/-- The function representing the total time needed for production -/
def f (x : ℕ) : ℚ := 9000 / x + 1000 / (100 - x)

/-- The theorem stating that f(x) reaches its minimum when x = 75 -/
theorem min_f_at_75 :
  ∀ x : ℕ, 1 ≤ x → x ≤ 99 → f 75 ≤ f x :=
sorry

end min_f_at_75_l1091_109141


namespace n_squared_divisible_by_144_l1091_109115

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∀ w : ℕ+, w ∣ n → w ≤ 12) : 144 ∣ n^2 := by
  sorry

end n_squared_divisible_by_144_l1091_109115


namespace only_traffic_light_random_l1091_109181

-- Define the type for events
inductive Event
  | SunRise
  | TrafficLight
  | PeanutOil
  | NegativeSum

-- Define a predicate for random events
def isRandom (e : Event) : Prop :=
  match e with
  | Event.TrafficLight => True
  | _ => False

-- Theorem statement
theorem only_traffic_light_random :
  ∀ (e : Event), isRandom e ↔ e = Event.TrafficLight :=
by sorry

end only_traffic_light_random_l1091_109181


namespace no_positive_integer_solutions_l1091_109192

theorem no_positive_integer_solutions :
  ¬∃ (x y z : ℕ+), x^3 + 2*y^3 = 4*z^3 := by
  sorry

end no_positive_integer_solutions_l1091_109192


namespace certain_number_proof_l1091_109137

theorem certain_number_proof (y : ℝ) : 
  (0.25 * 660 = 0.12 * y - 15) → y = 1500 := by
  sorry

end certain_number_proof_l1091_109137


namespace new_girls_count_l1091_109125

theorem new_girls_count (initial_girls : ℕ) (initial_boys : ℕ) (total_after : ℕ) : 
  initial_girls = 706 →
  initial_boys = 222 →
  total_after = 1346 →
  total_after - (initial_girls + initial_boys) = 418 :=
by
  sorry

end new_girls_count_l1091_109125


namespace derivative_of_cubic_composition_l1091_109129

/-- The derivative of y = f(a - bx) where f(x) = x^3 and a, b are real numbers -/
theorem derivative_of_cubic_composition (a b : ℝ) :
  deriv (fun x => (a - b*x)^3) = fun x => -3*b*(a - b*x)^2 := by sorry

end derivative_of_cubic_composition_l1091_109129


namespace johns_age_l1091_109136

theorem johns_age (john grandmother : ℕ) 
  (age_difference : john = grandmother - 48)
  (sum_of_ages : john + grandmother = 100) :
  john = 26 := by
sorry

end johns_age_l1091_109136


namespace line_ellipse_intersection_l1091_109198

theorem line_ellipse_intersection (m : ℝ) : 
  (∃! x y : ℝ, y = m * x + 2 ∧ x^2 + 6 * y^2 = 4) → m^2 = 5/6 := by
  sorry

end line_ellipse_intersection_l1091_109198


namespace modular_inverse_of_5_mod_23_l1091_109179

theorem modular_inverse_of_5_mod_23 :
  ∃ x : ℕ, x ≤ 22 ∧ (5 * x) % 23 = 1 :=
by
  use 14
  sorry

end modular_inverse_of_5_mod_23_l1091_109179


namespace carnation_count_flower_vase_problem_l1091_109100

/-- Proves that the number of carnations is 7 given the problem conditions -/
theorem carnation_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (vase_capacity : ℕ) (num_roses : ℕ) (num_vases : ℕ) (num_carnations : ℕ) =>
    (vase_capacity = 6 ∧ num_roses = 47 ∧ num_vases = 9) →
    (num_vases * vase_capacity = num_roses + num_carnations) →
    num_carnations = 7

/-- The main theorem stating the solution to the flower vase problem -/
theorem flower_vase_problem : ∃ (c : ℕ), carnation_count 6 47 9 c := by
  sorry

end carnation_count_flower_vase_problem_l1091_109100


namespace colleen_pays_more_than_joy_l1091_109178

/-- Calculates the cost of a pencil purchase based on the quantity --/
def pencilCost (quantity : ℕ) : ℚ :=
  if quantity < 20 then 4
  else if quantity < 40 then 7/2
  else 3

/-- Calculates the total cost of multiple pencil purchases --/
def totalCost (purchases : List ℕ) : ℚ :=
  purchases.map (λ q => q * pencilCost q) |>.sum

theorem colleen_pays_more_than_joy : 
  totalCost [25, 25] - totalCost [10, 15, 5] = 55 := by
  sorry

end colleen_pays_more_than_joy_l1091_109178


namespace rectangle_area_l1091_109117

-- Define the shapes in the rectangle
structure Rectangle :=
  (squares : Fin 2 → ℝ)  -- Areas of the two squares
  (triangle : ℝ)         -- Area of the triangle

-- Define the properties of the rectangle
def valid_rectangle (r : Rectangle) : Prop :=
  r.squares 0 = 4 ∧                     -- Area of smaller square is 4
  r.squares 1 = r.squares 0 ∧           -- Both squares have the same area
  r.triangle = r.squares 0 / 2          -- Area of triangle is half of square area

-- Theorem: The area of the rectangle is 10 square inches
theorem rectangle_area (r : Rectangle) (h : valid_rectangle r) : 
  r.squares 0 + r.squares 1 + r.triangle = 10 := by
  sorry

#check rectangle_area

end rectangle_area_l1091_109117


namespace shaded_region_perimeter_l1091_109152

-- Define the circle and its properties
def circle_radius : ℝ := 10
def central_angle : ℝ := 270

-- Theorem statement
theorem shaded_region_perimeter :
  let perimeter := 2 * circle_radius + (central_angle / 360) * (2 * Real.pi * circle_radius)
  perimeter = 20 + 15 * Real.pi := by sorry

end shaded_region_perimeter_l1091_109152


namespace jose_profit_share_l1091_109111

/-- Calculates the share of profit for an investor in a business partnership --/
def calculate_profit_share (total_profit investment_amount investment_duration total_investment_duration : ℕ) : ℕ :=
  (investment_amount * investment_duration * total_profit) / (total_investment_duration)

theorem jose_profit_share :
  let tom_investment := 30000
  let jose_investment := 45000
  let total_profit := 27000
  let tom_duration := 12
  let jose_duration := 10
  let total_investment_duration := tom_investment * tom_duration + jose_investment * jose_duration
  
  calculate_profit_share total_profit jose_investment jose_duration total_investment_duration = 15000 := by
  sorry

end jose_profit_share_l1091_109111


namespace dunk_height_above_rim_l1091_109110

/-- Represents the height of a basketball player in inches -/
def player_height : ℝ := 6 * 12

/-- Represents the additional reach of the player above their head in inches -/
def player_reach : ℝ := 22

/-- Represents the height of the basketball rim in inches -/
def rim_height : ℝ := 10 * 12

/-- Theorem stating the height above the rim a player must reach to dunk -/
theorem dunk_height_above_rim : 
  rim_height - (player_height + player_reach) = 26 := by sorry

end dunk_height_above_rim_l1091_109110


namespace yogurt_shop_combinations_l1091_109155

/-- The number of combinations of one flavor from n flavors and two different toppings from m toppings -/
def yogurt_combinations (n m : ℕ) : ℕ :=
  n * (m.choose 2)

/-- Theorem: There are 105 combinations of one flavor from 5 flavors and two different toppings from 7 toppings -/
theorem yogurt_shop_combinations :
  yogurt_combinations 5 7 = 105 := by
  sorry

end yogurt_shop_combinations_l1091_109155


namespace guppies_count_l1091_109112

/-- The number of guppies Haylee has -/
def haylee_guppies : ℕ := 3 * 12

/-- The number of guppies Jose has -/
def jose_guppies : ℕ := haylee_guppies / 2

/-- The number of guppies Charliz has -/
def charliz_guppies : ℕ := jose_guppies / 3

/-- The number of guppies Nicolai has -/
def nicolai_guppies : ℕ := charliz_guppies * 4

/-- The total number of guppies owned by all four friends -/
def total_guppies : ℕ := haylee_guppies + jose_guppies + charliz_guppies + nicolai_guppies

theorem guppies_count : total_guppies = 84 := by
  sorry

end guppies_count_l1091_109112


namespace lollipop_calories_l1091_109154

/-- Calculates the calories in a giant lollipop based on given candy information --/
theorem lollipop_calories 
  (chocolate_bars : ℕ) 
  (sugar_per_bar : ℕ) 
  (total_sugar : ℕ) 
  (calories_per_gram : ℕ) 
  (h1 : chocolate_bars = 14)
  (h2 : sugar_per_bar = 10)
  (h3 : total_sugar = 177)
  (h4 : calories_per_gram = 4) :
  (total_sugar - chocolate_bars * sugar_per_bar) * calories_per_gram = 148 := by
  sorry

#check lollipop_calories

end lollipop_calories_l1091_109154


namespace comparison_of_A_and_B_l1091_109122

theorem comparison_of_A_and_B (a b c : ℝ) : a^2 + b^2 + c^2 + 14 ≥ 2*a + 4*b + 6*c := by
  sorry

end comparison_of_A_and_B_l1091_109122


namespace quadratic_value_bound_l1091_109186

theorem quadratic_value_bound (a b : ℝ) : ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ |x^2 + a*x + b| ≥ (1/2 : ℝ) := by
  sorry

end quadratic_value_bound_l1091_109186


namespace g_fixed_points_l1091_109144

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
  sorry

end g_fixed_points_l1091_109144


namespace second_rack_dvds_l1091_109169

def dvd_sequence (n : ℕ) : ℕ → ℕ 
  | 0 => 2
  | i + 1 => 2 * dvd_sequence n i

theorem second_rack_dvds (n : ℕ) (h : n ≥ 5) :
  dvd_sequence n 0 = 2 ∧
  dvd_sequence n 2 = 8 ∧
  dvd_sequence n 3 = 16 ∧
  dvd_sequence n 4 = 32 ∧
  dvd_sequence n 5 = 64 →
  dvd_sequence n 1 = 4 := by
sorry

end second_rack_dvds_l1091_109169


namespace certain_number_exists_l1091_109182

theorem certain_number_exists : ∃ x : ℝ, 
  (x * 0.0729 * 28.9) / (0.0017 * 0.025 * 8.1) = 382.5 ∧ 
  abs (x - 50.35) < 0.01 := by
  sorry

end certain_number_exists_l1091_109182


namespace quadratic_integer_roots_l1091_109135

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end quadratic_integer_roots_l1091_109135


namespace max_bing_games_and_wins_l1091_109160

/-- Represents a player in the table tennis game -/
inductive Player : Type
| jia : Player
| yi : Player
| bing : Player

/-- The game state, tracking the number of games played by each player -/
structure GameState :=
  (jia_games : ℕ)
  (yi_games : ℕ)
  (bing_games : ℕ)
  (bing_wins : ℕ)

/-- Checks if the game state is valid according to the rules -/
def is_valid_state (state : GameState) : Prop :=
  state.jia_games = 10 ∧ 
  state.yi_games = 7 ∧ 
  state.bing_games ≤ state.jia_games ∧ 
  state.bing_games ≤ state.yi_games + state.bing_wins ∧
  state.bing_wins ≤ state.bing_games

/-- The main theorem to prove -/
theorem max_bing_games_and_wins :
  ∃ (state : GameState), 
    is_valid_state state ∧ 
    ∀ (other_state : GameState), 
      is_valid_state other_state → 
      other_state.bing_games ≤ state.bing_games ∧
      other_state.bing_wins ≤ state.bing_wins ∧
      state.bing_games = 13 ∧
      state.bing_wins = 10 := by
  sorry

end max_bing_games_and_wins_l1091_109160


namespace remainder_sum_mod_11_l1091_109142

theorem remainder_sum_mod_11 (a b c : ℕ) : 
  1 ≤ a ∧ a ≤ 10 →
  1 ≤ b ∧ b ≤ 10 →
  1 ≤ c ∧ c ≤ 10 →
  (a * b * c) % 11 = 2 →
  (7 * c) % 11 = 3 →
  (8 * b) % 11 = (4 + b) % 11 →
  (a + b + c) % 11 = 0 := by
sorry

end remainder_sum_mod_11_l1091_109142


namespace chocolate_theorem_l1091_109127

/-- Represents a square chocolate bar -/
structure ChocolateBar where
  side_length : ℕ
  piece_size : ℕ

/-- Calculates the number of pieces eaten along the sides of a square chocolate bar -/
def pieces_eaten (bar : ChocolateBar) : ℕ :=
  4 * (bar.side_length * 2 - 4)

/-- Theorem stating that for a 100cm square chocolate bar with 1cm pieces,
    the number of pieces eaten along the sides is 784 -/
theorem chocolate_theorem :
  let bar : ChocolateBar := { side_length := 100, piece_size := 1 }
  pieces_eaten bar = 784 := by
  sorry

end chocolate_theorem_l1091_109127


namespace triangle_problem_l1091_109126

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (a - c) * (a + c) * Real.sin C = c * (b - c) * Real.sin B →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  Real.sin B * Real.sin C = 1/4 →
  A = π/3 ∧ a = 2 * Real.sqrt 3 := by
  sorry

end triangle_problem_l1091_109126


namespace jenna_profit_l1091_109120

/-- Calculates the total profit for Jenna's wholesale business --/
def calculate_profit (
  widget_cost : ℝ)
  (widget_price : ℝ)
  (rent : ℝ)
  (tax_rate : ℝ)
  (worker_salary : ℝ)
  (num_workers : ℕ)
  (widgets_sold : ℕ) : ℝ :=
  let total_sales := widget_price * widgets_sold
  let total_cost := widget_cost * widgets_sold
  let salaries := worker_salary * num_workers
  let profit_before_tax := total_sales - total_cost - rent - salaries
  let tax := tax_rate * profit_before_tax
  profit_before_tax - tax

/-- Theorem stating that Jenna's profit is $4000 given the problem conditions --/
theorem jenna_profit :
  calculate_profit 3 8 10000 0.2 2500 4 5000 = 4000 := by
  sorry

end jenna_profit_l1091_109120


namespace tan_315_degrees_l1091_109123

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end tan_315_degrees_l1091_109123


namespace product_67_63_l1091_109194

theorem product_67_63 : 67 * 63 = 4221 := by
  sorry

end product_67_63_l1091_109194


namespace parallelogram_side_length_l1091_109145

/-- Proves that for a parallelogram with adjacent sides of lengths s and 2s units forming a 60-degree angle, if the area of the parallelogram is 12√3 square units, then s = 2√3. -/
theorem parallelogram_side_length (s : ℝ) :
  s > 0 →
  let side1 : ℝ := s
  let side2 : ℝ := 2 * s
  let angle : ℝ := π / 3  -- 60 degrees in radians
  let area : ℝ := 12 * Real.sqrt 3
  side2 * (side1 * Real.sin angle) = area →
  s = 2 * Real.sqrt 3 := by
sorry

end parallelogram_side_length_l1091_109145


namespace village_assistants_selection_l1091_109158

theorem village_assistants_selection (n m k : ℕ) : 
  n = 10 → m = 3 → k = 2 →
  (Nat.choose (n - 1) m) - (Nat.choose (n - k - 1) m) = 49 := by
  sorry

end village_assistants_selection_l1091_109158


namespace x_eq_x_squared_is_quadratic_l1091_109162

/-- A quadratic equation in terms of x is an equation that can be written in the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 - x represents the equation x = x^2 -/
def f (x : ℝ) : ℝ := x^2 - x

/-- Theorem: The equation x = x^2 is a quadratic equation in terms of x -/
theorem x_eq_x_squared_is_quadratic : is_quadratic_equation f := by
  sorry


end x_eq_x_squared_is_quadratic_l1091_109162


namespace points_scored_third_game_l1091_109118

-- Define the average points per game after 2 games
def avg_points_2_games : ℝ := 61.5

-- Define the total points needed to exceed after 3 games
def total_points_threshold : ℕ := 500

-- Define the additional points needed after 3 games
def additional_points_needed : ℕ := 330

-- Theorem to prove
theorem points_scored_third_game :
  let total_points_2_games := 2 * avg_points_2_games
  let points_third_game := total_points_threshold - additional_points_needed - total_points_2_games
  points_third_game = 47 := by
  sorry

end points_scored_third_game_l1091_109118


namespace circumscribed_sphere_surface_area_l1091_109167

/-- The surface area of a sphere circumscribing a rectangular solid with edge lengths 3, 4, and 5 emanating from one vertex is equal to 50π. -/
theorem circumscribed_sphere_surface_area (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  4 * π * radius^2 = 50 * π := by sorry

end circumscribed_sphere_surface_area_l1091_109167


namespace consecutive_interior_angles_indeterminate_l1091_109180

-- Define consecutive interior angles
def consecutive_interior_angles (α β : ℝ) : Prop := sorry

-- Theorem statement
theorem consecutive_interior_angles_indeterminate (α β : ℝ) :
  consecutive_interior_angles α β → α = 55 → ¬∃!β, consecutive_interior_angles α β :=
sorry

end consecutive_interior_angles_indeterminate_l1091_109180


namespace infinitely_many_divisible_by_digit_sum_l1091_109168

/-- Define a function that creates a number with n ones -/
def ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Sum of digits of a number with all ones -/
def sum_of_digits (n : ℕ) : ℕ := n

/-- The main theorem -/
theorem infinitely_many_divisible_by_digit_sum :
  ∀ n : ℕ, (ones (3^n)) % (sum_of_digits (3^n)) = 0 :=
by sorry

end infinitely_many_divisible_by_digit_sum_l1091_109168


namespace inequality_proof_l1091_109174

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + a^2) 
  ≥ Real.sqrt 2 * (a + b + c + d) := by
  sorry

end inequality_proof_l1091_109174


namespace perpendicular_lines_and_intersection_l1091_109188

-- Define the four lines
def line1 (x y : ℚ) : Prop := 4 * y - 3 * x = 15
def line2 (x y : ℚ) : Prop := -3 * x - 4 * y = 15
def line3 (x y : ℚ) : Prop := 4 * y + 3 * x = 15
def line4 (x y : ℚ) : Prop := 3 * y + 4 * x = 15

-- Define perpendicularity
def perpendicular (f g : ℚ → ℚ → Prop) : Prop :=
  ∃ m1 m2 : ℚ, (∀ x y, f x y ↔ y = m1 * x + (15 / 4)) ∧
             (∀ x y, g x y ↔ y = m2 * x + 5) ∧
             m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_and_intersection :
  perpendicular line1 line4 ∧
  line1 (15/32) (35/8) ∧
  line4 (15/32) (35/8) := by sorry

end perpendicular_lines_and_intersection_l1091_109188


namespace square_sum_geq_negative_double_product_l1091_109190

theorem square_sum_geq_negative_double_product (a b : ℝ) : a^2 + b^2 ≥ -2*a*b := by
  sorry

end square_sum_geq_negative_double_product_l1091_109190


namespace domain_of_g_l1091_109149

-- Define the function f with domain [-3, 1]
def f : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}

-- Define the function g(x) = f(x) + f(-x)
def g (x : ℝ) : Prop := x ∈ f ∧ (-x) ∈ f

-- Theorem statement
theorem domain_of_g : 
  {x : ℝ | g x} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
sorry

end domain_of_g_l1091_109149


namespace square_sequence_theorem_l1091_109159

/-- The number of nonoverlapping unit squares in the nth figure -/
def f (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 3

/-- The theorem stating the properties of the sequence and the value for the 100th figure -/
theorem square_sequence_theorem :
  (f 0 = 3) ∧ (f 1 = 9) ∧ (f 2 = 19) ∧ (f 3 = 33) → f 100 = 20403 :=
by
  sorry

end square_sequence_theorem_l1091_109159


namespace angle_inequality_l1091_109140

theorem angle_inequality (α β : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : α / (2 * (1 + Real.cos (α / 2))) < Real.tan β ∧ Real.tan β < (1 - Real.cos α) / α) :
  α / 4 < β ∧ β < α / 2 := by sorry

end angle_inequality_l1091_109140


namespace present_age_of_B_l1091_109109

/-- 
Given two people A and B, where:
1. In 20 years, A will be twice as old as B was 20 years ago.
2. A is now 10 years older than B.
Prove that the present age of B is 70 years.
-/
theorem present_age_of_B (A B : ℕ) 
  (h1 : A + 20 = 2 * (B - 20))
  (h2 : A = B + 10) : 
  B = 70 := by
  sorry


end present_age_of_B_l1091_109109


namespace equal_roots_quadratic_l1091_109195

/-- 
Given a quadratic equation x^2 + 6x + m = 0 with two equal real roots,
prove that m = 9.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + 6*y + m = 0 → y = x) → 
  m = 9 := by
sorry

end equal_roots_quadratic_l1091_109195


namespace sin_150_degrees_l1091_109108

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l1091_109108


namespace arcsin_neg_half_equals_neg_pi_sixth_l1091_109193

theorem arcsin_neg_half_equals_neg_pi_sixth : 
  Real.arcsin (-1/2) = -π/6 := by
  sorry

end arcsin_neg_half_equals_neg_pi_sixth_l1091_109193


namespace geometric_series_problem_l1091_109196

theorem geometric_series_problem (a₁ : ℝ) (r₁ r₂ : ℝ) (m : ℝ) :
  a₁ = 15 →
  a₁ * r₁ = 9 →
  a₁ * r₂ = 9 + m →
  a₁ / (1 - r₂) = 3 * (a₁ / (1 - r₁)) →
  m = 4 := by
sorry

end geometric_series_problem_l1091_109196


namespace tangent_implies_m_six_or_twelve_l1091_109171

/-- An ellipse defined by x^2 + 9y^2 = 9 -/
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- A hyperbola defined by x^2 - m(y-1)^2 = 4 -/
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y-1)^2 = 4

/-- The condition for the ellipse and hyperbola to be tangent -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x y, ellipse x y ∧ hyperbola x y m ∧
  ∀ x' y', ellipse x' y' ∧ hyperbola x' y' m → (x', y') = (x, y)

/-- The theorem stating that if the ellipse and hyperbola are tangent, then m must be 6 or 12 -/
theorem tangent_implies_m_six_or_twelve :
  ∀ m, are_tangent m → m = 6 ∨ m = 12 :=
sorry

end tangent_implies_m_six_or_twelve_l1091_109171


namespace smallest_a_equals_36_l1091_109199

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_double (x : ℝ) (h : x > 0) : f (2 * x) = 2 * f x

axiom f_interval (x : ℝ) (h : 1 < x ∧ x ≤ 2) : f x = 2 - x

-- Define the theorem
theorem smallest_a_equals_36 :
  ∃ a : ℝ, a > 0 ∧ f a = f 2020 ∧ ∀ b : ℝ, b > 0 → f b = f 2020 → a ≤ b :=
sorry

end smallest_a_equals_36_l1091_109199


namespace average_income_A_and_C_l1091_109173

/-- Given the monthly incomes of individuals A, B, and C, prove that the average income of A and C is 4200. -/
theorem average_income_A_and_C (A B C : ℕ) : 
  (A + B) / 2 = 4050 →
  (B + C) / 2 = 5250 →
  A = 3000 →
  (A + C) / 2 = 4200 := by
sorry

end average_income_A_and_C_l1091_109173


namespace soccer_games_total_l1091_109143

theorem soccer_games_total (win_percentage : ℝ) (games_won : ℕ) (h1 : win_percentage = 0.65) (h2 : games_won = 182) :
  (games_won : ℝ) / win_percentage = 280 := by
  sorry

end soccer_games_total_l1091_109143


namespace ellipse_equation_not_standard_l1091_109114

theorem ellipse_equation_not_standard (a c : ℝ) (h1 : a = 6) (h2 : c = 1) :
  let b := Real.sqrt (a^2 - c^2)
  ¬ ((∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/36 + y^2/35 = 1) ∨
     (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ y^2/36 + x^2/35 = 1)) :=
by sorry

end ellipse_equation_not_standard_l1091_109114


namespace dividend_percentage_calculation_l1091_109189

/-- Calculates the dividend percentage given investment details and dividend amount -/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_percentage : ℝ)
  (dividend_received : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_percentage = 20)
  (h4 : dividend_received = 600) :
  let share_cost := share_face_value * (1 + premium_percentage / 100)
  let num_shares := investment / share_cost
  let dividend_per_share := dividend_received / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 5 := by sorry

end dividend_percentage_calculation_l1091_109189


namespace arithmetic_series_first_term_l1091_109184

-- Define an arithmetic series
def ArithmeticSeries (a₁ : ℚ) (d : ℚ) : ℕ → ℚ := fun n ↦ a₁ + (n - 1 : ℚ) * d

-- Sum of first n terms of an arithmetic series
def SumArithmeticSeries (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem arithmetic_series_first_term
  (a₁ : ℚ)
  (d : ℚ)
  (h1 : SumArithmeticSeries a₁ d 60 = 240)
  (h2 : SumArithmeticSeries (ArithmeticSeries a₁ d 61) d 60 = 3600) :
  a₁ = -353/15 := by
  sorry

end arithmetic_series_first_term_l1091_109184


namespace amanda_kitchen_upgrade_cost_l1091_109191

/-- The cost of Amanda's kitchen upgrade --/
def kitchen_upgrade_cost (cabinet_knobs_count : ℕ) (cabinet_knob_price : ℚ) 
                         (drawer_pulls_count : ℕ) (drawer_pull_price : ℚ) : ℚ :=
  (cabinet_knobs_count : ℚ) * cabinet_knob_price + (drawer_pulls_count : ℚ) * drawer_pull_price

/-- Theorem stating that the cost of Amanda's kitchen upgrade is $77.00 --/
theorem amanda_kitchen_upgrade_cost : 
  kitchen_upgrade_cost 18 (5/2) 8 4 = 77 :=
by sorry

end amanda_kitchen_upgrade_cost_l1091_109191


namespace log_equality_implies_ratio_one_l1091_109102

theorem log_equality_implies_ratio_one (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 10) ∧
  (Real.log p / Real.log 8 = Real.log (p^2 + q) / Real.log 20) →
  p^2 / q = 1 := by
sorry

end log_equality_implies_ratio_one_l1091_109102


namespace max_tulips_is_15_l1091_109197

/-- Represents the cost of a yellow tulip in rubles -/
def yellow_cost : ℕ := 50

/-- Represents the cost of a red tulip in rubles -/
def red_cost : ℕ := 31

/-- Represents the maximum budget in rubles -/
def max_budget : ℕ := 600

/-- Represents a valid bouquet of tulips -/
structure Bouquet where
  yellow : ℕ
  red : ℕ
  odd_total : (yellow + red) % 2 = 1
  color_diff : (yellow = red + 1) ∨ (red = yellow + 1)
  within_budget : yellow * yellow_cost + red * red_cost ≤ max_budget

/-- The maximum number of tulips in a bouquet -/
def max_tulips : ℕ := 15

/-- Theorem stating that the maximum number of tulips in a valid bouquet is 15 -/
theorem max_tulips_is_15 : 
  ∀ b : Bouquet, b.yellow + b.red ≤ max_tulips ∧ 
  ∃ b' : Bouquet, b'.yellow + b'.red = max_tulips :=
sorry

end max_tulips_is_15_l1091_109197


namespace crayon_cost_theorem_l1091_109128

/-- The number of crayons in a half dozen -/
def half_dozen : ℕ := 6

/-- The number of half dozens bought -/
def num_half_dozens : ℕ := 4

/-- The cost of one crayon in dollars -/
def cost_per_crayon : ℕ := 2

/-- The total number of crayons bought -/
def total_crayons : ℕ := num_half_dozens * half_dozen

/-- The total cost of the crayons in dollars -/
def total_cost : ℕ := total_crayons * cost_per_crayon

theorem crayon_cost_theorem : total_cost = 48 := by
  sorry

end crayon_cost_theorem_l1091_109128


namespace product_remainder_mod_17_l1091_109170

theorem product_remainder_mod_17 :
  (2021 * 2022 * 2023 * 2024 * 2025) % 17 = 0 := by
  sorry

end product_remainder_mod_17_l1091_109170


namespace interior_angle_of_regular_polygon_with_five_diagonals_l1091_109104

/-- Given a regular polygon where at most 5 diagonals can be drawn from a vertex,
    prove that one of its interior angles measures 135°. -/
theorem interior_angle_of_regular_polygon_with_five_diagonals :
  ∀ (n : ℕ), 
    n ≥ 3 →  -- Ensures it's a valid polygon
    n - 3 = 5 →  -- At most 5 diagonals can be drawn from a vertex
    (180 * (n - 2) : ℝ) / n = 135 :=  -- One interior angle measures 135°
by sorry

end interior_angle_of_regular_polygon_with_five_diagonals_l1091_109104


namespace line_through_point_l1091_109138

/-- Given a line equation 1 - 3kx = 7y and a point (-2/3, 3) on the line, prove that k = 10 -/
theorem line_through_point (k : ℝ) : 
  (1 - 3 * k * (-2/3) = 7 * 3) → k = 10 := by
  sorry

end line_through_point_l1091_109138


namespace modulus_of_z_l1091_109177

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3 * Complex.I) = 6 + 4 * Complex.I) : 
  Complex.abs z = 2 := by
sorry

end modulus_of_z_l1091_109177


namespace staircase_pencils_staircase_steps_l1091_109105

/-- Represents a staircase with n steps, where each step k has three segments of k pencils each. -/
def Staircase (n : ℕ) : ℕ := 3 * (n * (n + 1) / 2)

/-- Theorem stating that a staircase with 15 steps uses exactly 360 pencils. -/
theorem staircase_pencils : Staircase 15 = 360 := by
  sorry

/-- Theorem stating that if a staircase uses 360 pencils, it must have 15 steps. -/
theorem staircase_steps (n : ℕ) (h : Staircase n = 360) : n = 15 := by
  sorry

end staircase_pencils_staircase_steps_l1091_109105


namespace petyas_friends_l1091_109183

theorem petyas_friends (total_stickers : ℕ) : 
  (∃ (x : ℕ), 5 * x + 8 = total_stickers ∧ 6 * x = total_stickers + 11) → 
  (∃ (x : ℕ), x = 19 ∧ 5 * x + 8 = total_stickers ∧ 6 * x = total_stickers + 11) :=
by sorry

end petyas_friends_l1091_109183


namespace average_marks_chemistry_math_l1091_109134

theorem average_marks_chemistry_math (P C M : ℕ) : 
  P + C + M = P + 110 → (C + M) / 2 = 55 := by
  sorry

end average_marks_chemistry_math_l1091_109134


namespace operational_probability_independent_of_root_l1091_109187

/-- Represents a computer network -/
structure ComputerNetwork where
  servers : Type
  channels : servers → servers → Prop
  failure_prob : ℝ
  failure_prob_nonneg : 0 ≤ failure_prob
  failure_prob_le_one : failure_prob ≤ 1

/-- Predicate to check if a server can reach another server using operating channels -/
def can_reach (G : ComputerNetwork) (s t : G.servers) : Prop :=
  sorry

/-- Predicate to check if a network is operational with respect to a root server -/
def is_operational (G : ComputerNetwork) (r : G.servers) : Prop :=
  ∀ s : G.servers, can_reach G s r

/-- The probability that a network is operational -/
noncomputable def operational_probability (G : ComputerNetwork) (r : G.servers) : ℝ :=
  sorry

/-- Theorem stating that the operational probability is independent of the choice of root server -/
theorem operational_probability_independent_of_root (G : ComputerNetwork) 
  (r₁ r₂ : G.servers) (h : r₁ ≠ r₂) : 
  operational_probability G r₁ = operational_probability G r₂ :=
sorry

end operational_probability_independent_of_root_l1091_109187


namespace two_numbers_with_given_means_l1091_109153

theorem two_numbers_with_given_means (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0) :
  Real.sqrt (x * y) = Real.sqrt 5 ∧ (x + y) / 2 = 5 →
  (x = 5 + 2 * Real.sqrt 5 ∧ y = 5 - 2 * Real.sqrt 5) ∨
  (x = 5 - 2 * Real.sqrt 5 ∧ y = 5 + 2 * Real.sqrt 5) := by
  sorry

end two_numbers_with_given_means_l1091_109153


namespace binomial_coefficient_21_13_l1091_109121

theorem binomial_coefficient_21_13 :
  (Nat.choose 20 13 = 77520) →
  (Nat.choose 20 12 = 125970) →
  (Nat.choose 21 13 = 203490) := by
sorry

end binomial_coefficient_21_13_l1091_109121


namespace rectangles_count_l1091_109133

/-- The number of rectangles formed by p parallel lines and q perpendicular lines -/
def num_rectangles (p q : ℕ) : ℚ :=
  (p * (p - 1) * q * (q - 1)) / 4

/-- Theorem stating that num_rectangles gives the correct number of rectangles -/
theorem rectangles_count (p q : ℕ) :
  num_rectangles p q = (p * (p - 1) * q * (q - 1)) / 4 := by
  sorry

#check rectangles_count

end rectangles_count_l1091_109133


namespace square_of_difference_l1091_109119

theorem square_of_difference (x : ℝ) :
  (7 - (x^3 - 49)^(1/3))^2 = 49 - 14 * (x^3 - 49)^(1/3) + ((x^3 - 49)^(1/3))^2 := by
  sorry

end square_of_difference_l1091_109119


namespace no_integer_solution_l1091_109103

theorem no_integer_solution : ¬∃ (a b : ℤ), a * b * (a + b) = 20182017 := by
  sorry

end no_integer_solution_l1091_109103


namespace decagon_triangles_l1091_109151

/-- The number of marked points on the decagon -/
def n : ℕ := 20

/-- The number of sides of the decagon -/
def sides : ℕ := 10

/-- The number of points to choose for each triangle -/
def k : ℕ := 3

/-- The total number of ways to choose 3 points out of 20 -/
def total_combinations : ℕ := Nat.choose n k

/-- The number of non-triangle-forming sets (collinear points on each side) -/
def non_triangle_sets : ℕ := sides

/-- The number of valid triangles -/
def valid_triangles : ℕ := total_combinations - non_triangle_sets

theorem decagon_triangles :
  valid_triangles = 1130 :=
sorry

end decagon_triangles_l1091_109151


namespace aunt_marge_candy_count_l1091_109113

/-- The number of candy pieces each child receives -/
structure CandyDistribution where
  kate : ℕ
  bill : ℕ
  robert : ℕ
  mary : ℕ

/-- The conditions of Aunt Marge's candy distribution -/
def is_valid_distribution (d : CandyDistribution) : Prop :=
  d.robert = d.kate + 2 ∧
  d.bill = d.mary - 6 ∧
  d.mary = d.robert + 2 ∧
  d.kate = d.bill + 2 ∧
  d.kate = 4

/-- The theorem stating that Aunt Marge has 24 pieces of candy in total -/
theorem aunt_marge_candy_count (d : CandyDistribution) 
  (h : is_valid_distribution d) : 
  d.kate + d.bill + d.robert + d.mary = 24 := by
  sorry

#check aunt_marge_candy_count

end aunt_marge_candy_count_l1091_109113


namespace hans_age_l1091_109150

theorem hans_age (hans_age josiah_age : ℕ) : 
  josiah_age = 3 * hans_age →
  hans_age + 3 + josiah_age + 3 = 66 →
  hans_age = 15 := by
sorry

end hans_age_l1091_109150


namespace binomial_sum_simplification_l1091_109106

theorem binomial_sum_simplification (n : ℕ) (p : ℝ) :
  (Finset.range (n + 1)).sum (λ k => k * (n.choose k) * p^k * (1 - p)^(n - k)) = n * p := by
  sorry

end binomial_sum_simplification_l1091_109106


namespace davids_biology_marks_l1091_109157

def marks_english : ℕ := 72
def marks_mathematics : ℕ := 60
def marks_physics : ℕ := 35
def marks_chemistry : ℕ := 62
def num_subjects : ℕ := 5
def average_marks : ℚ := 62.6

theorem davids_biology_marks :
  ∃ (marks_biology : ℕ),
    (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology) / num_subjects = average_marks ∧
    marks_biology = 84 := by
  sorry

end davids_biology_marks_l1091_109157


namespace division_chain_l1091_109185

theorem division_chain : (180 / 6) / 3 = 10 := by
  sorry

end division_chain_l1091_109185


namespace a_fourth_plus_inverse_a_fourth_l1091_109139

theorem a_fourth_plus_inverse_a_fourth (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^4 + 1/a^4 = 7 := by
sorry

end a_fourth_plus_inverse_a_fourth_l1091_109139


namespace floor_x_floor_x_eq_48_l1091_109172

theorem floor_x_floor_x_eq_48 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 48 ↔ 48 / 7 ≤ x ∧ x < 49 / 7 := by
  sorry

end floor_x_floor_x_eq_48_l1091_109172


namespace isosceles_triangle_vertex_angle_l1091_109163

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define the exterior angle for this problem
  exteriorAngle : ℝ

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.exteriorAngle = 40) : 
  -- The vertex angle is 140°
  (180 - triangle.exteriorAngle) = 140 := by
  sorry

end isosceles_triangle_vertex_angle_l1091_109163


namespace polynomial_equation_solution_l1091_109175

theorem polynomial_equation_solution (p : Polynomial ℝ) :
  (∀ x : ℝ, x ≠ 0 → p.eval x ^ 2 + p.eval (1 / x) ^ 2 = p.eval (x ^ 2) * p.eval (1 / x ^ 2)) →
  p = 0 := by
sorry

end polynomial_equation_solution_l1091_109175


namespace lori_marbles_l1091_109130

/-- The number of friends Lori shares her marbles with -/
def num_friends : ℕ := 5

/-- The number of marbles each friend gets when Lori shares her marbles -/
def marbles_per_friend : ℕ := 6

/-- The total number of marbles Lori has -/
def total_marbles : ℕ := num_friends * marbles_per_friend

theorem lori_marbles : total_marbles = 30 := by
  sorry

end lori_marbles_l1091_109130


namespace tim_running_schedule_l1091_109146

/-- The number of days Tim used to run per week -/
def previous_running_days (hours_per_day : ℕ) (total_hours_per_week : ℕ) (extra_days : ℕ) : ℕ :=
  total_hours_per_week / hours_per_day - extra_days

theorem tim_running_schedule :
  previous_running_days 2 10 2 = 3 := by
  sorry

end tim_running_schedule_l1091_109146


namespace largest_n_for_square_sum_l1091_109161

theorem largest_n_for_square_sum : ∃ (m : ℕ), 
  (4^995 + 4^1500 + 4^2004 = m^2) ∧ 
  (∀ (k : ℕ), k > 2004 → ¬∃ (l : ℕ), 4^995 + 4^1500 + 4^k = l^2) := by
  sorry

end largest_n_for_square_sum_l1091_109161


namespace michelle_crayon_boxes_l1091_109132

theorem michelle_crayon_boxes (total_crayons : ℕ) (crayons_per_box : ℕ) (num_boxes : ℕ) :
  total_crayons = 35 →
  crayons_per_box = 5 →
  num_boxes * crayons_per_box = total_crayons →
  num_boxes = 7 := by
  sorry

end michelle_crayon_boxes_l1091_109132


namespace doctors_lawyers_ratio_l1091_109164

theorem doctors_lawyers_ratio (d l : ℕ) (h1 : d + l > 0) :
  (40 * d + 55 * l : ℚ) / (d + l : ℚ) = 45 →
  (d : ℚ) / (l : ℚ) = 2 := by
  sorry

end doctors_lawyers_ratio_l1091_109164


namespace inlet_fill_rate_l1091_109148

/-- The rate at which the inlet pipe fills water, given tank capacity and emptying times. -/
theorem inlet_fill_rate (tank_capacity : ℝ) (leak_empty_time : ℝ) (combined_empty_time : ℝ) :
  tank_capacity = 5760 →
  leak_empty_time = 6 →
  combined_empty_time = 8 →
  (tank_capacity / leak_empty_time - tank_capacity / combined_empty_time) / 60 = 12 :=
by sorry

end inlet_fill_rate_l1091_109148


namespace installment_plan_properties_l1091_109147

/-- Represents the installment plan for a household appliance purchase -/
structure InstallmentPlan where
  initialPrice : ℝ
  initialPayment : ℝ
  monthlyBasePayment : ℝ
  monthlyInterestRate : ℝ

/-- Calculates the payment for a given month in the installment plan -/
def monthlyPayment (plan : InstallmentPlan) (month : ℕ) : ℝ :=
  plan.monthlyBasePayment + (plan.initialPrice - plan.initialPayment - plan.monthlyBasePayment * (month - 1)) * plan.monthlyInterestRate

/-- Calculates the total amount paid over the course of the installment plan -/
def totalPayment (plan : InstallmentPlan) (totalMonths : ℕ) : ℝ :=
  plan.initialPayment + (Finset.range totalMonths).sum (fun i => monthlyPayment plan (i + 1))

/-- Theorem stating the properties of the specific installment plan -/
theorem installment_plan_properties :
  let plan : InstallmentPlan := {
    initialPrice := 1150,
    initialPayment := 150,
    monthlyBasePayment := 50,
    monthlyInterestRate := 0.01
  }
  (monthlyPayment plan 10 = 55.5) ∧
  (totalPayment plan 20 = 1255) := by
  sorry


end installment_plan_properties_l1091_109147


namespace freds_basketball_games_l1091_109176

theorem freds_basketball_games 
  (missed_this_year : ℕ) 
  (attended_last_year : ℕ) 
  (total_attended : ℕ) 
  (h1 : missed_this_year = 35)
  (h2 : attended_last_year = 11)
  (h3 : total_attended = 47) :
  total_attended - attended_last_year = 36 :=
by sorry

end freds_basketball_games_l1091_109176


namespace smallest_possible_median_l1091_109166

def number_set (x : ℤ) : Finset ℤ := {x, 3*x, 4, 3, 7}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem smallest_possible_median :
  ∀ x : ℤ, ∃ m : ℤ, is_median m (number_set x) ∧ m = 3 ∧ 
  ∀ m' : ℤ, is_median m' (number_set x) → m ≤ m' :=
by sorry

end smallest_possible_median_l1091_109166


namespace multiply_72514_99999_l1091_109131

theorem multiply_72514_99999 : 72514 * 99999 = 7250675486 := by
  sorry

end multiply_72514_99999_l1091_109131
