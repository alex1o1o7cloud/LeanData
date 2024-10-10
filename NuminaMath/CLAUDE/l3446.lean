import Mathlib

namespace monomial_sum_implies_mn_four_l3446_344639

/-- If the sum of two monomials -3a^m*b^2 and (1/2)a^2*b^n is still a monomial, then mn = 4 -/
theorem monomial_sum_implies_mn_four (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ) (p q : ℕ), -3 * a^m * b^2 + (1/2) * a^2 * b^n = k * a^p * b^q) →
  m * n = 4 :=
by sorry

end monomial_sum_implies_mn_four_l3446_344639


namespace modified_triangle_pieces_count_l3446_344643

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a1 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

/-- Represents the modified triangle construction -/
structure ModifiedTriangle where
  rows : ℕ
  rodStart : ℕ
  rodIncrease : ℕ
  connectorStart : ℕ
  connectorIncrease : ℕ
  supportStart : ℕ
  supportIncrease : ℕ
  supportStartRow : ℕ

/-- Calculates the total number of pieces in the modified triangle -/
def totalPieces (t : ModifiedTriangle) : ℕ :=
  let rods := arithmeticSum t.rodStart t.rodIncrease t.rows
  let connectors := arithmeticSum t.connectorStart t.connectorIncrease (t.rows + 1)
  let supports := arithmeticSum t.supportStart t.supportIncrease (t.rows - t.supportStartRow + 1)
  rods + connectors + supports

/-- The theorem to be proved -/
theorem modified_triangle_pieces_count :
  let t : ModifiedTriangle := {
    rows := 10,
    rodStart := 4,
    rodIncrease := 5,
    connectorStart := 1,
    connectorIncrease := 1,
    supportStart := 2,
    supportIncrease := 2,
    supportStartRow := 3
  }
  totalPieces t = 395 := by sorry

end modified_triangle_pieces_count_l3446_344643


namespace binomial_equation_solution_l3446_344695

theorem binomial_equation_solution :
  ∃! (A B C : ℝ), ∀ (n : ℕ), n > 0 →
    2 * n^3 + 3 * n^2 = A * (n.choose 3) + B * (n.choose 2) + C * (n.choose 1) ∧
    A = 12 ∧ B = 18 ∧ C = 5 := by
  sorry

end binomial_equation_solution_l3446_344695


namespace sum_and_count_integers_l3446_344652

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_integers (x y : ℕ) :
  x = sum_integers 40 60 ∧
  y = count_even_integers 40 60 ∧
  x + y = 1061 →
  x = 1050 := by sorry

end sum_and_count_integers_l3446_344652


namespace a_7_greater_than_3_l3446_344677

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sequence is monotonically increasing -/
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The theorem statement -/
theorem a_7_greater_than_3 (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : monotonically_increasing a) 
  (h3 : a 1 + a 10 = 6) : 
  a 7 > 3 := by
  sorry


end a_7_greater_than_3_l3446_344677


namespace original_price_calculation_l3446_344605

theorem original_price_calculation (original_price new_price : ℝ) : 
  new_price = 0.8 * original_price ∧ new_price = 80 → original_price = 100 := by
  sorry

end original_price_calculation_l3446_344605


namespace halloween_candy_theorem_l3446_344632

/-- Calculates the remaining candy after Debby and her sister combine their Halloween candy and eat some. -/
def remaining_candy (debby_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  debby_candy + sister_candy - eaten_candy

/-- Theorem stating that the remaining candy is correct given the initial conditions. -/
theorem halloween_candy_theorem (debby_candy sister_candy eaten_candy : ℕ) :
  remaining_candy debby_candy sister_candy eaten_candy = debby_candy + sister_candy - eaten_candy :=
by sorry

end halloween_candy_theorem_l3446_344632


namespace growth_rate_ratio_l3446_344678

/-- Given a linear regression equation y = ax + b where a = 4.4,
    prove that the ratio of the growth rate between x and y is 5/22 -/
theorem growth_rate_ratio (a b : ℝ) (h : a = 4.4) :
  (1 / a : ℝ) = 5 / 22 := by
  sorry

end growth_rate_ratio_l3446_344678


namespace optimal_mask_purchase_l3446_344633

/-- Represents the profit function for mask sales -/
def profit_function (x : ℝ) : ℝ := -0.05 * x + 400

/-- Represents the constraints on the number of masks -/
def mask_constraints (x : ℝ) : Prop := 500 ≤ x ∧ x ≤ 1000

/-- Theorem stating the optimal purchase for maximum profit -/
theorem optimal_mask_purchase :
  ∀ x : ℝ, mask_constraints x →
  profit_function 500 ≥ profit_function x :=
sorry

end optimal_mask_purchase_l3446_344633


namespace hyperbola_asymptote_intersection_l3446_344636

/-- Given a hyperbola, its asymptote, a parabola, and a circle, prove the value of a parameter. -/
theorem hyperbola_asymptote_intersection (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ x₀ : ℝ, y = 2*x₀*x - x₀^2 + 1) →      -- Asymptote equation (tangent to parabola)
  (∃ x y : ℝ, x^2 + (y - a)^2 = 1) →       -- Circle equation
  (∃ x₁ y₁ x₂ y₂ : ℝ,                      -- Chord endpoints
    x₁^2 + (y₁ - a)^2 = 1 ∧ 
    x₂^2 + (y₂ - a)^2 = 1 ∧ 
    y₁ = 2*x₀*x₁ - x₀^2 + 1 ∧ 
    y₂ = 2*x₀*x₂ - x₀^2 + 1 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2) →       -- Chord length
  a = Real.sqrt 10 / 2 :=
by sorry

end hyperbola_asymptote_intersection_l3446_344636


namespace final_sum_after_operations_l3446_344676

theorem final_sum_after_operations (S a b : ℝ) : 
  a + b = S → 3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end final_sum_after_operations_l3446_344676


namespace factorization_theorem_l3446_344607

variable (x : ℝ)

theorem factorization_theorem :
  (x^2 - 4*x + 3 = (x-1)*(x-3)) ∧
  (4*x^2 + 12*x - 7 = (2*x+7)*(2*x-1)) := by
  sorry

end factorization_theorem_l3446_344607


namespace racketCostProof_l3446_344606

/-- Calculates the total cost of two rackets under a specific promotion --/
def totalCostOfRackets (fullPrice : ℚ) : ℚ :=
  fullPrice + (fullPrice / 2)

/-- Proves that the total cost of two rackets is $90 under the given conditions --/
theorem racketCostProof : totalCostOfRackets 60 = 90 := by
  sorry

end racketCostProof_l3446_344606


namespace employee_selection_probability_l3446_344689

/-- Represents the survey results of employees -/
structure EmployeeSurvey where
  total : ℕ
  uninsured : ℕ
  partTime : ℕ
  uninsuredPartTime : ℕ
  multipleJobs : ℕ
  alternativeInsurance : ℕ

/-- Calculates the probability of selecting an employee with specific characteristics -/
def calculateProbability (survey : EmployeeSurvey) : ℚ :=
  let neitherUninsuredNorPartTime := survey.total - (survey.uninsured + survey.partTime - survey.uninsuredPartTime)
  let targetEmployees := neitherUninsuredNorPartTime - survey.multipleJobs - survey.alternativeInsurance
  targetEmployees / survey.total

/-- The main theorem stating the probability of selecting an employee with specific characteristics -/
theorem employee_selection_probability :
  let survey := EmployeeSurvey.mk 500 140 80 6 35 125
  calculateProbability survey = 63 / 250 := by sorry

end employee_selection_probability_l3446_344689


namespace value_of_expression_l3446_344614

theorem value_of_expression (x : ℝ) (h : x = 4) : (3*x + 7)^2 = 361 := by
  sorry

end value_of_expression_l3446_344614


namespace special_ellipse_major_axis_length_l3446_344693

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : Bool
  /-- The x-coordinate of both foci -/
  foci_x : ℝ
  /-- The y-coordinate of the first focus -/
  focus1_y : ℝ
  /-- The y-coordinate of the second focus -/
  focus2_y : ℝ

/-- The length of the major axis of the special ellipse -/
def majorAxisLength (e : SpecialEllipse) : ℝ := 2

/-- Theorem stating that the length of the major axis is 2 for the given ellipse -/
theorem special_ellipse_major_axis_length (e : SpecialEllipse) 
  (h1 : e.tangent_to_axes = true)
  (h2 : e.foci_x = 4)
  (h3 : e.focus1_y = 1 + 2 * Real.sqrt 2)
  (h4 : e.focus2_y = 1 - 2 * Real.sqrt 2) :
  majorAxisLength e = 2 := by sorry

end special_ellipse_major_axis_length_l3446_344693


namespace complement_of_M_in_U_l3446_344681

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}

theorem complement_of_M_in_U :
  (U \ M) = {3, 4, 6} := by sorry

end complement_of_M_in_U_l3446_344681


namespace at_least_one_multiple_of_11_l3446_344690

def base_n_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem at_least_one_multiple_of_11 :
  ∃ n : Nat, 2 ≤ n ∧ n ≤ 101 ∧ 
  (base_n_to_decimal [3, 4, 5, 7, 6, 2] n) % 11 = 0 :=
sorry

end at_least_one_multiple_of_11_l3446_344690


namespace cubic_root_squared_l3446_344616

theorem cubic_root_squared (r : ℝ) : 
  r^3 - r + 3 = 0 → (r^2)^3 - 2*(r^2)^2 + r^2 - 9 = 0 := by
  sorry

end cubic_root_squared_l3446_344616


namespace triangle_angle_B_l3446_344619

theorem triangle_angle_B (a b : ℝ) (A : Real) (h1 : a = 4) (h2 : b = 4 * Real.sqrt 3) (h3 : A = 30 * π / 180) :
  let B := Real.arcsin ((b * Real.sin A) / a)
  B = 60 * π / 180 ∨ B = 120 * π / 180 := by
  sorry

end triangle_angle_B_l3446_344619


namespace divisible_by_56_l3446_344628

theorem divisible_by_56 (n : ℕ) 
  (h1 : ∃ k : ℕ, 3 * n + 1 = k ^ 2) 
  (h2 : ∃ m : ℕ, 4 * n + 1 = m ^ 2) : 
  56 ∣ n := by
sorry

end divisible_by_56_l3446_344628


namespace sport_water_amount_l3446_344654

/-- Represents the ratio of flavoring, corn syrup, and water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_ratio (std : DrinkRatio) : DrinkRatio :=
  { flavoring := std.flavoring,
    corn_syrup := std.corn_syrup / 3,
    water := std.water * 2 }

/-- Calculates the amount of water given the amount of corn syrup and the drink ratio -/
def water_amount (corn_syrup_amount : ℚ) (ratio : DrinkRatio) : ℚ :=
  (corn_syrup_amount / ratio.corn_syrup) * ratio.water

theorem sport_water_amount :
  water_amount 4 (sport_ratio standard_ratio) = 60 := by
  sorry

end sport_water_amount_l3446_344654


namespace parallelogram_base_length_l3446_344683

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 448) 
  (h2 : height = 14) 
  (h3 : area = base * height) : 
  base = 32 := by
sorry

end parallelogram_base_length_l3446_344683


namespace noahs_lights_l3446_344610

theorem noahs_lights (W : ℝ) 
  (h1 : W > 0)  -- Assuming W is positive
  (h2 : 2 * W + 2 * (3 * W) + 2 * (4 * W) = 96) : W = 6 := by
  sorry

end noahs_lights_l3446_344610


namespace ned_games_before_l3446_344669

/-- The number of games Ned had before giving away some -/
def games_before : ℕ := sorry

/-- The number of games Ned gave away -/
def games_given_away : ℕ := 13

/-- The number of games Ned has now -/
def games_now : ℕ := 6

/-- Theorem stating the number of games Ned had before -/
theorem ned_games_before :
  games_before = games_given_away + games_now := by sorry

end ned_games_before_l3446_344669


namespace problem_solution_l3446_344658

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem problem_solution (m : ℤ) (h_odd : m % 2 = 1) (h_eq : g (g (g m)) = 39) : m = 63 := by
  sorry

end problem_solution_l3446_344658


namespace arithmetic_sequence_sum_l3446_344608

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 4 + a 9 + a 11 = 32) →
  (a 6 + a 7 = 16) :=
by
  sorry

end arithmetic_sequence_sum_l3446_344608


namespace smallest_n_for_candy_purchase_l3446_344697

theorem smallest_n_for_candy_purchase : 
  (∃ n : ℕ, n > 0 ∧ 
    24 * n % 10 = 0 ∧ 
    24 * n % 16 = 0 ∧ 
    24 * n % 18 = 0 ∧
    (∀ m : ℕ, m > 0 → 
      (24 * m % 10 = 0 ∧ 24 * m % 16 = 0 ∧ 24 * m % 18 = 0) → 
      m ≥ n)) → 
  (∃ n : ℕ, n = 30 ∧
    24 * n % 10 = 0 ∧ 
    24 * n % 16 = 0 ∧ 
    24 * n % 18 = 0 ∧
    (∀ m : ℕ, m > 0 → 
      (24 * m % 10 = 0 ∧ 24 * m % 16 = 0 ∧ 24 * m % 18 = 0) → 
      m ≥ n)) :=
by sorry

#check smallest_n_for_candy_purchase

end smallest_n_for_candy_purchase_l3446_344697


namespace coefficient_x_cubed_in_expansion_l3446_344604

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 30
  let k : ℕ := 3
  let a : ℕ := 2
  (Nat.choose n k) * a^(n - k) = 4060 * 2^27 := by
  sorry

end coefficient_x_cubed_in_expansion_l3446_344604


namespace line_parameterization_l3446_344663

/-- Given a line y = 2x - 40 parameterized by (x,y) = (g(t), 20t - 14), prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ x y t, y = 2*x - 40 ∧ x = g t ∧ y = 20*t - 14) → 
  (∀ t, g t = 10*t + 13) := by
sorry

end line_parameterization_l3446_344663


namespace water_remaining_l3446_344691

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 11/8 → remaining = initial - used → remaining = 13/8 := by
  sorry

end water_remaining_l3446_344691


namespace bart_mixtape_problem_l3446_344625

/-- A mixtape with two sides -/
structure Mixtape where
  first_side_songs : ℕ
  second_side_songs : ℕ
  song_length : ℕ
  total_length : ℕ

/-- The problem statement -/
theorem bart_mixtape_problem (m : Mixtape) 
  (h1 : m.second_side_songs = 4)
  (h2 : m.song_length = 4)
  (h3 : m.total_length = 40) :
  m.first_side_songs = 6 := by
  sorry


end bart_mixtape_problem_l3446_344625


namespace work_done_on_bullet_l3446_344600

theorem work_done_on_bullet (m : Real) (v1 v2 : Real) :
  m = 0.01 →
  v1 = 500 →
  v2 = 200 →
  let K1 := (1/2) * m * v1^2
  let K2 := (1/2) * m * v2^2
  K1 - K2 = 1050 := by sorry

end work_done_on_bullet_l3446_344600


namespace binary_101_equals_5_l3446_344648

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.foldr (fun bit acc => 2 * acc + bit) 0

theorem binary_101_equals_5 : binary_to_decimal [1, 0, 1] = 5 := by
  sorry

end binary_101_equals_5_l3446_344648


namespace starting_lineup_count_l3446_344664

def total_team_members : ℕ := 12
def offensive_linemen : ℕ := 4
def linemen_quarterbacks : ℕ := 2
def running_backs : ℕ := 3

def starting_lineup_combinations : ℕ := 
  offensive_linemen * linemen_quarterbacks * running_backs * (total_team_members - 3)

theorem starting_lineup_count : starting_lineup_combinations = 216 := by
  sorry

end starting_lineup_count_l3446_344664


namespace power_sum_difference_l3446_344696

theorem power_sum_difference (m k p q : ℕ) : 
  2^m + 2^k = p → 2^m - 2^k = q → 2^(m+k) = (p^2 - q^2) / 4 :=
by sorry

end power_sum_difference_l3446_344696


namespace no_integer_solution_l3446_344647

theorem no_integer_solution : ¬ ∃ (x : ℤ), 7 - 3 * (x^2 - 2) > 19 := by
  sorry

end no_integer_solution_l3446_344647


namespace min_value_theorem_l3446_344644

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : 2*m + 2*n = 2) : 
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧ 
    (∀ (x y : ℝ), x > 0 → y > 0 → 2*x + 2*y = 2 → 1/x + 2/y ≥ min_val) :=
by sorry

end min_value_theorem_l3446_344644


namespace square_in_base_seven_l3446_344679

theorem square_in_base_seven :
  ∃ (b : ℕ) (h : b > 6), 
    (1 * b^4 + 6 * b^3 + 3 * b^2 + 2 * b + 4) = (1 * b^2 + 2 * b + 5)^2 ∧ b = 7 := by
  sorry

end square_in_base_seven_l3446_344679


namespace domain_of_g_l3446_344630

-- Define the function f with domain [0,2]
def f : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define the function g(x) = f(x²)
def g (x : ℝ) : Prop := x^2 ∈ f

-- Theorem statement
theorem domain_of_g : 
  {x : ℝ | g x} = {x : ℝ | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2} :=
by sorry

end domain_of_g_l3446_344630


namespace ice_cream_flavors_l3446_344665

/-- The number of basic ice cream flavors -/
def num_flavors : ℕ := 4

/-- The number of scoops used to create a new flavor -/
def num_scoops : ℕ := 5

/-- The number of ways to distribute n identical objects into k distinct categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem ice_cream_flavors : 
  distribute num_scoops num_flavors = 56 := by
sorry

end ice_cream_flavors_l3446_344665


namespace hot_dog_truck_profit_l3446_344642

/-- Calculates the profit for a hot dog food truck over a three-day period --/
theorem hot_dog_truck_profit
  (friday_customers : ℕ)
  (friday_tip_average : ℝ)
  (saturday_customer_multiplier : ℕ)
  (saturday_tip_average : ℝ)
  (sunday_customers : ℕ)
  (sunday_tip_average : ℝ)
  (hot_dog_price : ℝ)
  (ingredient_cost : ℝ)
  (daily_maintenance : ℝ)
  (weekend_taxes : ℝ)
  (h1 : friday_customers = 28)
  (h2 : friday_tip_average = 2)
  (h3 : saturday_customer_multiplier = 3)
  (h4 : saturday_tip_average = 2.5)
  (h5 : sunday_customers = 36)
  (h6 : sunday_tip_average = 1.5)
  (h7 : hot_dog_price = 4)
  (h8 : ingredient_cost = 1.25)
  (h9 : daily_maintenance = 50)
  (h10 : weekend_taxes = 150) :
  (friday_customers * friday_tip_average + 
   friday_customers * saturday_customer_multiplier * saturday_tip_average + 
   sunday_customers * sunday_tip_average +
   (friday_customers + friday_customers * saturday_customer_multiplier + sunday_customers) * 
   (hot_dog_price - ingredient_cost) - 
   3 * daily_maintenance - weekend_taxes) = 427 := by
  sorry


end hot_dog_truck_profit_l3446_344642


namespace exists_k_in_interval_l3446_344685

theorem exists_k_in_interval (x : ℝ) (hx_pos : 0 < x) (hx_le_one : x ≤ 1) :
  ∃ k : ℕ+, (4/3 : ℝ) < (k : ℝ) * x ∧ (k : ℝ) * x ≤ 2 := by
  sorry

end exists_k_in_interval_l3446_344685


namespace distance_between_cities_l3446_344687

/-- The distance between City A and City B in miles -/
def distance : ℝ := 427.5

/-- The initial travel time from City A to City B in hours -/
def initial_time_AB : ℝ := 6

/-- The initial travel time from City B to City A in hours -/
def initial_time_BA : ℝ := 4.5

/-- The time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- The average speed for the round trip after saving time in miles per hour -/
def average_speed : ℝ := 90

theorem distance_between_cities :
  2 * distance / (initial_time_AB + initial_time_BA - 2 * time_saved) = average_speed :=
sorry

end distance_between_cities_l3446_344687


namespace k_range_l3446_344618

open Real

theorem k_range (k : ℝ) : 
  (∀ x > 1, k * (exp (k * x) + 1) - (1 / x + 1) * log x > 0) → 
  k > 1 / exp 1 := by
  sorry

end k_range_l3446_344618


namespace baseball_league_games_l3446_344659

theorem baseball_league_games (P Q : ℕ) : 
  P > 2 * Q →
  Q > 6 →
  4 * P + 5 * Q = 82 →
  4 * P = 52 := by
sorry

end baseball_league_games_l3446_344659


namespace g_fixed_points_l3446_344641

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = -2 ∨ x = 3 := by
sorry

end g_fixed_points_l3446_344641


namespace complex_angle_for_one_plus_i_sqrt_seven_l3446_344655

theorem complex_angle_for_one_plus_i_sqrt_seven :
  let z : ℂ := 1 + Complex.I * Real.sqrt 7
  let r : ℝ := Complex.abs z
  let θ : ℝ := Complex.arg z
  θ = π / 8 := by sorry

end complex_angle_for_one_plus_i_sqrt_seven_l3446_344655


namespace rhombus_area_l3446_344651

/-- The area of a rhombus with side length 4 cm and an angle of 30° between adjacent sides is 8√3 cm². -/
theorem rhombus_area (side_length : ℝ) (angle : ℝ) :
  side_length = 4 →
  angle = 30 * π / 180 →
  let area := side_length * side_length * Real.sin angle
  area = 8 * Real.sqrt 3 := by
  sorry

end rhombus_area_l3446_344651


namespace complement_of_M_in_U_l3446_344609

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def M : Set ℕ := {2, 4, 7}

theorem complement_of_M_in_U :
  U \ M = {1, 3, 5, 6} := by sorry

end complement_of_M_in_U_l3446_344609


namespace connor_date_cost_l3446_344698

def movie_date_cost (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) (cup_price : ℚ) : ℚ :=
  let discounted_ticket := ticket_price * (1/2)
  let tickets_total := ticket_price + discounted_ticket
  let candy_total := 2 * candy_price * (1 - 1/5)
  let cup_total := cup_price - 1
  tickets_total + combo_price + candy_total + cup_total

theorem connor_date_cost :
  movie_date_cost 14 11 2.5 5 = 40 := by
  sorry

end connor_date_cost_l3446_344698


namespace repeating_decimal_equiv_fraction_l3446_344637

/-- Represents a repeating decimal with an integer part, a non-repeating fractional part, and a repeating part -/
structure RepeatingDecimal where
  integerPart : ℤ
  nonRepeatingPart : ℚ
  repeatingPart : ℚ
  nonRepeatingPartLessThanOne : nonRepeatingPart < 1
  repeatingPartLessThanOne : repeatingPart < 1

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + d.nonRepeatingPart + d.repeatingPart / (1 - (1/10)^(d.repeatingPart.den))

/-- Checks if a fraction is in its lowest terms -/
def isLowestTerms (n d : ℤ) : Prop :=
  Nat.gcd n.natAbs d.natAbs = 1

theorem repeating_decimal_equiv_fraction :
  let d : RepeatingDecimal := ⟨0, 4/10, 37/100, by norm_num, by norm_num⟩
  d.toRational = 433 / 990 ∧ isLowestTerms 433 990 := by
  sorry

end repeating_decimal_equiv_fraction_l3446_344637


namespace cheesecake_problem_l3446_344671

/-- A problem about cheesecakes in a bakery -/
theorem cheesecake_problem
  (initial_display : ℕ)
  (sold : ℕ)
  (total_left : ℕ)
  (h1 : initial_display = 10)
  (h2 : sold = 7)
  (h3 : total_left = 18) :
  initial_display - sold + (total_left - (initial_display - sold)) = 15 :=
by sorry

end cheesecake_problem_l3446_344671


namespace factor_expression_l3446_344653

theorem factor_expression (x : ℝ) : 75 * x^13 + 200 * x^26 = 25 * x^13 * (3 + 8 * x^13) := by
  sorry

end factor_expression_l3446_344653


namespace mean_home_runs_l3446_344612

def total_players : ℕ := 6 + 4 + 3 + 1 + 1 + 1

def total_home_runs : ℕ := 6 * 6 + 7 * 4 + 8 * 3 + 10 * 1 + 11 * 1 + 12 * 1

theorem mean_home_runs : 
  (total_home_runs : ℚ) / (total_players : ℚ) = 7.5625 := by sorry

end mean_home_runs_l3446_344612


namespace edward_games_count_l3446_344657

theorem edward_games_count :
  let sold_games : ℕ := 19
  let boxes_used : ℕ := 2
  let games_per_box : ℕ := 8
  let packed_games : ℕ := boxes_used * games_per_box
  let total_games : ℕ := sold_games + packed_games
  total_games = 35 := by sorry

end edward_games_count_l3446_344657


namespace factorization_expr1_l3446_344624

theorem factorization_expr1 (a b : ℝ) :
  -3 * a^2 * b + 12 * a * b - 12 * b = -3 * b * (a - 2)^2 := by sorry

end factorization_expr1_l3446_344624


namespace symmetric_point_to_origin_l3446_344672

/-- If |a-3|+(b+4)^2=0, then the point (a,b) is (3,-4) and its symmetric point to the origin is (-3,4) -/
theorem symmetric_point_to_origin (a b : ℝ) : 
  (|a - 3| + (b + 4)^2 = 0) → 
  (a = 3 ∧ b = -4) ∧ 
  ((-a, -b) = (-3, 4)) := by
sorry

end symmetric_point_to_origin_l3446_344672


namespace swaps_theorem_l3446_344621

/-- Represents a mode of letter swapping -/
inductive SwapMode
| Adjacent : SwapMode
| Any : SwapMode

/-- Represents a string of letters -/
def Text : Type := List Char

/-- Calculate the minimum number of swaps required to transform one text into another -/
def minSwaps (original : Text) (target : Text) (mode : SwapMode) : Nat :=
  match mode with
  | SwapMode.Adjacent => sorry
  | SwapMode.Any => sorry

/-- The original text -/
def originalText : Text := ['M', 'E', 'G', 'Y', 'E', 'I', ' ', 'T', 'A', 'K', 'A', 'R', 'É', 'K', 'P', 'É', 'N', 'Z', 'T', 'Á', 'R', ' ', 'R', '.', ' ', 'T', '.']

/-- The target text -/
def targetText : Text := ['T', 'A', 'T', 'Á', 'R', ' ', 'G', 'Y', 'E', 'R', 'M', 'E', 'K', ' ', 'A', ' ', 'P', 'É', 'N', 'Z', 'T', ' ', 'K', 'É', 'R', 'I', '.']

theorem swaps_theorem :
  (minSwaps originalText targetText SwapMode.Adjacent = 85) ∧
  (minSwaps originalText targetText SwapMode.Any = 11) :=
sorry

end swaps_theorem_l3446_344621


namespace supplementary_angles_ratio_l3446_344611

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 5 / 4 →  -- The ratio of the angles is 5:4
  min a b = 80 :=  -- The smaller angle is 80°
by sorry

end supplementary_angles_ratio_l3446_344611


namespace parallel_vectors_component_l3446_344613

/-- Given two parallel vectors a and b, prove that the second component of b is 5/3. -/
theorem parallel_vectors_component (a b : ℝ × ℝ) : 
  a = (3, 5) → b.1 = 1 → (∃ (k : ℝ), a = k • b) → b.2 = 5/3 := by
  sorry

end parallel_vectors_component_l3446_344613


namespace bus_meeting_problem_l3446_344688

theorem bus_meeting_problem (n k : ℕ) (h1 : n > 3) 
  (h2 : n * (n - 1) * (2 * k - 1) = 600) : n * k = 52 ∨ n * k = 40 := by
  sorry

end bus_meeting_problem_l3446_344688


namespace man_rowing_speed_l3446_344686

/-- Proves that given a man's speed in still water and his speed rowing downstream,
    his speed rowing upstream can be calculated. -/
theorem man_rowing_speed
  (speed_still : ℝ)
  (speed_downstream : ℝ)
  (h_still : speed_still = 30)
  (h_downstream : speed_downstream = 35) :
  speed_still - (speed_downstream - speed_still) = 25 :=
by sorry

end man_rowing_speed_l3446_344686


namespace count_divisible_by_three_is_334_l3446_344675

/-- The number obtained by writing the integers 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The count of numbers b_k divisible by 3, where 1 ≤ k ≤ 500 -/
def count_divisible_by_three : ℕ := sorry

theorem count_divisible_by_three_is_334 : count_divisible_by_three = 334 := by sorry

end count_divisible_by_three_is_334_l3446_344675


namespace sine_identity_l3446_344601

theorem sine_identity (α : Real) (h : α = π / 7) :
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := by
  sorry

end sine_identity_l3446_344601


namespace rectangle_dimension_increase_l3446_344634

theorem rectangle_dimension_increase (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.1 * L) (h2 : L' * B' = 1.43 * (L * B)) : B' = 1.3 * B :=
sorry

end rectangle_dimension_increase_l3446_344634


namespace community_A_sample_l3446_344661

/-- Represents the number of low-income households in a community -/
structure Community where
  households : ℕ

/-- Represents the total number of affordable housing units -/
def housing_units : ℕ := 90

/-- Calculates the number of households to be sampled from a community using stratified sampling -/
def stratified_sample (community : Community) (total_households : ℕ) : ℕ :=
  (community.households * housing_units) / total_households

/-- Theorem: The number of low-income households to be sampled from community A is 40 -/
theorem community_A_sample :
  let community_A : Community := ⟨360⟩
  let community_B : Community := ⟨270⟩
  let community_C : Community := ⟨180⟩
  let total_households := community_A.households + community_B.households + community_C.households
  stratified_sample community_A total_households = 40 := by
  sorry

end community_A_sample_l3446_344661


namespace field_length_proof_l3446_344662

/-- Proves that for a rectangular field with given conditions, the length is 24 meters -/
theorem field_length_proof (width : ℝ) (length : ℝ) : 
  width = 13.5 → length = 2 * width - 3 → length = 24 := by
  sorry

end field_length_proof_l3446_344662


namespace stratified_sampling_selection_l3446_344649

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of liberal arts students -/
def liberal_arts_students : ℕ := 5

/-- The number of science students -/
def science_students : ℕ := 10

/-- The number of liberal arts students to be selected -/
def selected_liberal_arts : ℕ := 2

/-- The number of science students to be selected -/
def selected_science : ℕ := 4

theorem stratified_sampling_selection :
  (binomial liberal_arts_students selected_liberal_arts) * 
  (binomial science_students selected_science) = 2100 := by
sorry

end stratified_sampling_selection_l3446_344649


namespace min_value_of_f_l3446_344682

/-- Given positive real numbers a, b, c, x, y, z satisfying the given conditions,
    the minimum value of the function f is 1/2 -/
theorem min_value_of_f (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : c * y + b * z = a)
  (eq2 : a * z + c * x = b)
  (eq3 : b * x + a * y = c) :
  (∀ x' y' z' : ℝ, 0 < x' → 0 < y' → 0 < z' →
    c * y' + b * z' = a →
    a * z' + c * x' = b →
    b * x' + a * y' = c →
    x'^2 / (1 + x') + y'^2 / (1 + y') + z'^2 / (1 + z') ≥ 1/2) ∧
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1/2 :=
by sorry

end min_value_of_f_l3446_344682


namespace bridge_length_l3446_344646

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 265 := by
  sorry

end bridge_length_l3446_344646


namespace squirrel_acorns_l3446_344680

/-- Represents the number of acorns each animal hides per hole -/
structure AcornsPerHole where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- Represents the number of holes each animal dug -/
structure Holes where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- The forest scenario with animals hiding acorns -/
def ForestScenario (a : AcornsPerHole) (h : Holes) : Prop :=
  -- Chipmunk and squirrel stash the same number of acorns
  a.chipmunk * h.chipmunk = a.squirrel * h.squirrel ∧
  -- Rabbit stashes the same number of acorns as the chipmunk
  a.rabbit * h.rabbit = a.chipmunk * h.chipmunk ∧
  -- Rabbit needs 3 more holes than the squirrel
  h.rabbit = h.squirrel + 3

/-- The theorem stating that the squirrel stashed 40 acorns -/
theorem squirrel_acorns (a : AcornsPerHole) (h : Holes)
  (ha : a.chipmunk = 4 ∧ a.squirrel = 5 ∧ a.rabbit = 3)
  (hf : ForestScenario a h) : 
  a.squirrel * h.squirrel = 40 := by
  sorry

end squirrel_acorns_l3446_344680


namespace max_min_values_of_f_l3446_344627

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 + 9 * x^2 - 2

-- Define the interval
def interval : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem max_min_values_of_f :
  (∃ x ∈ interval, f x = 50) ∧
  (∀ y ∈ interval, f y ≤ 50) ∧
  (∃ x ∈ interval, f x = -2) ∧
  (∀ y ∈ interval, f y ≥ -2) := by
  sorry

end max_min_values_of_f_l3446_344627


namespace sum_of_integers_l3446_344656

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 4) 
  (h2 : x.val * y.val = 98) : 
  x.val + y.val = 18 := by
sorry

end sum_of_integers_l3446_344656


namespace trigonometric_expressions_l3446_344674

open Real

theorem trigonometric_expressions :
  (∀ α : ℝ, tan α = 2 →
    (sin (2 * π - α) + cos (π + α)) / (cos (α - π) - cos ((3 * π) / 2 - α)) = -3) ∧
  sin (50 * π / 180) * (1 + Real.sqrt 3 * tan (10 * π / 180)) = 1 :=
by sorry

end trigonometric_expressions_l3446_344674


namespace plates_theorem_l3446_344615

def plates_problem (flower_plates checked_plates : ℕ) : ℕ :=
  let initial_plates := flower_plates + checked_plates
  let polka_plates := 2 * checked_plates
  let total_before_smash := initial_plates + polka_plates
  total_before_smash - 1

theorem plates_theorem : 
  plates_problem 4 8 = 27 := by
  sorry

end plates_theorem_l3446_344615


namespace bijective_function_theorem_l3446_344673

theorem bijective_function_theorem (a : ℝ) :
  (∃ f : ℝ → ℝ, Function.Bijective f ∧
    (∀ x : ℝ, f (f x) = x^2 * f x + a * x^2)) →
  a = 0 := by
sorry

end bijective_function_theorem_l3446_344673


namespace sum_x_y_equals_three_l3446_344660

/-- Given a system of linear equations, prove that x + y = 3 -/
theorem sum_x_y_equals_three (x y : ℝ) 
  (eq1 : 2 * x + y = 5) 
  (eq2 : x + 2 * y = 4) : 
  x + y = 3 := by
  sorry

end sum_x_y_equals_three_l3446_344660


namespace jane_rejection_calculation_l3446_344629

/-- The percentage of products John rejected -/
def john_rejection_rate : ℝ := 0.007

/-- The total percentage of products rejected -/
def total_rejection_rate : ℝ := 0.0075

/-- The fraction of products Jane inspected -/
def jane_inspection_fraction : ℝ := 0.5

/-- The percentage of products Jane rejected -/
def jane_rejection_rate : ℝ := 0.001

theorem jane_rejection_calculation :
  john_rejection_rate + jane_rejection_rate * jane_inspection_fraction = total_rejection_rate :=
sorry

end jane_rejection_calculation_l3446_344629


namespace table_tennis_racket_sales_l3446_344631

/-- Profit function for table tennis racket sales -/
def profit_function (c : ℝ) (x : ℝ) : ℝ :=
  let y := -10 * x + 900
  y * (x - c)

/-- Problem statement for table tennis racket sales -/
theorem table_tennis_racket_sales 
  (c : ℝ) 
  (max_price : ℝ) 
  (min_profit : ℝ) 
  (h1 : c = 50) 
  (h2 : max_price = 75) 
  (h3 : min_profit = 3000) :
  ∃ (optimal_price : ℝ) (max_profit : ℝ) (price_range : Set ℝ),
    -- 1. The monthly profit function
    (∀ x, profit_function c x = -10 * x^2 + 1400 * x - 45000) ∧
    -- 2. The optimal price and maximum profit
    (optimal_price = 70 ∧ 
     max_profit = profit_function c optimal_price ∧
     max_profit = 4000 ∧
     ∀ x, profit_function c x ≤ max_profit) ∧
    -- 3. The range of acceptable selling prices
    (price_range = {x | 60 ≤ x ∧ x ≤ 75} ∧
     ∀ x ∈ price_range, 
       x ≤ max_price ∧ 
       profit_function c x ≥ min_profit) :=
by sorry

end table_tennis_racket_sales_l3446_344631


namespace parabola_and_point_theorem_l3446_344603

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

def on_parabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

def perpendicular (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

theorem parabola_and_point_theorem (C : Parabola) (A B O : Point) :
  on_parabola A C →
  on_parabola B C →
  A.x = 1 →
  A.y = 2 →
  O.x = 0 →
  O.y = 0 →
  B.x ≠ 0 →
  perpendicular A O B →
  (C.p = 2 ∧ B.x = 16 ∧ B.y = -8) := by sorry

end parabola_and_point_theorem_l3446_344603


namespace fifth_rack_dvds_sixth_rack_dvds_prove_fifth_rack_l3446_344638

def dvd_sequence : Nat → Nat
  | 0 => 2
  | n + 1 => 2 * dvd_sequence n

theorem fifth_rack_dvds : dvd_sequence 4 = 32 :=
by
  sorry

theorem sixth_rack_dvds : dvd_sequence 5 = 64 :=
by
  sorry

theorem prove_fifth_rack (h : dvd_sequence 5 = 64) : dvd_sequence 4 = 32 :=
by
  sorry

end fifth_rack_dvds_sixth_rack_dvds_prove_fifth_rack_l3446_344638


namespace catch_up_time_is_correct_l3446_344640

/-- The time (in minutes) for the minute hand to catch up with the hour hand after 8:00 --/
def catch_up_time : ℚ :=
  let minute_hand_speed : ℚ := 6
  let hour_hand_speed : ℚ := 1/2
  let initial_hour_hand_position : ℚ := 240
  (initial_hour_hand_position / (minute_hand_speed - hour_hand_speed))

theorem catch_up_time_is_correct : catch_up_time = 43 + 7/11 := by
  sorry

end catch_up_time_is_correct_l3446_344640


namespace quadratic_distinct_roots_l3446_344645

/-- The quadratic equation (k-1)x^2 + 4x + 1 = 0 has two distinct real roots
    if and only if k < 5 and k ≠ 1 -/
theorem quadratic_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    (k - 1) * x^2 + 4 * x + 1 = 0 ∧
    (k - 1) * y^2 + 4 * y + 1 = 0) ↔
  (k < 5 ∧ k ≠ 1) :=
sorry

end quadratic_distinct_roots_l3446_344645


namespace line_points_l3446_344635

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_points : 
  let p1 : Point := ⟨4, 10⟩
  let p2 : Point := ⟨-2, -8⟩
  let p3 : Point := ⟨1, 1⟩
  let p4 : Point := ⟨-1, -5⟩
  let p5 : Point := ⟨3, 7⟩
  let p6 : Point := ⟨0, -1⟩
  let p7 : Point := ⟨2, 3⟩
  collinear p1 p2 p3 ∧ 
  collinear p1 p2 p4 ∧ 
  collinear p1 p2 p5 ∧ 
  ¬collinear p1 p2 p6 ∧ 
  ¬collinear p1 p2 p7 := by sorry

end line_points_l3446_344635


namespace max_product_value_l3446_344692

/-- Given two real-valued functions f and g with specified ranges and a condition on their maxima,
    this theorem states that the maximum value of their product is 35. -/
theorem max_product_value (f g : ℝ → ℝ) (hf : ∀ x, 1 ≤ f x ∧ f x ≤ 7) 
    (hg : ∀ x, -3 ≤ g x ∧ g x ≤ 5) 
    (hmax : ∃ x, f x = 7 ∧ g x = 5) : 
    (∃ b, ∀ x, f x * g x ≤ b) ∧ (∀ b, (∀ x, f x * g x ≤ b) → b ≥ 35) :=
sorry

end max_product_value_l3446_344692


namespace decrease_by_percentage_eighty_decreased_by_eightyfive_percent_l3446_344684

theorem decrease_by_percentage (x : ℝ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 100) :
  x - (p / 100) * x = x * (1 - p / 100) :=
sorry

theorem eighty_decreased_by_eightyfive_percent :
  80 - (85 / 100) * 80 = 12 :=
sorry

end decrease_by_percentage_eighty_decreased_by_eightyfive_percent_l3446_344684


namespace quadrangular_pyramid_faces_l3446_344670

/-- A quadrangular pyramid is a geometric shape with triangular lateral faces and a quadrilateral base. -/
structure QuadrangularPyramid where
  lateral_faces : Nat
  base_face : Nat
  lateral_faces_are_triangles : lateral_faces = 4
  base_is_quadrilateral : base_face = 1

/-- The total number of faces in a quadrangular pyramid is 5. -/
theorem quadrangular_pyramid_faces (p : QuadrangularPyramid) : 
  p.lateral_faces + p.base_face = 5 := by
  sorry

end quadrangular_pyramid_faces_l3446_344670


namespace customized_bowling_ball_volume_l3446_344650

/-- The volume of a customized bowling ball -/
theorem customized_bowling_ball_volume :
  let ball_diameter : ℝ := 24
  let hole_depth : ℝ := 10
  let hole_diameters : List ℝ := [1.5, 1.5, 2, 2.5]
  let sphere_volume := (4 / 3) * π * (ball_diameter / 2) ^ 3
  let hole_volumes := hole_diameters.map (fun d => π * (d / 2) ^ 2 * hole_depth)
  sphere_volume - hole_volumes.sum = 2233.375 * π := by
  sorry

end customized_bowling_ball_volume_l3446_344650


namespace amusement_park_elementary_students_l3446_344622

theorem amusement_park_elementary_students 
  (total_women : ℕ) 
  (women_elementary : ℕ) 
  (more_men : ℕ) 
  (men_not_elementary : ℕ) 
  (h1 : total_women = 1518)
  (h2 : women_elementary = 536)
  (h3 : more_men = 525)
  (h4 : men_not_elementary = 1257) :
  women_elementary + (total_women + more_men - men_not_elementary) = 1322 :=
by
  sorry

end amusement_park_elementary_students_l3446_344622


namespace unchanged_fraction_l3446_344667

theorem unchanged_fraction (x y : ℝ) : 
  (2 * x) / (3 * x - y) = (2 * (3 * x)) / (3 * (3 * x) - (3 * y)) :=
by sorry

end unchanged_fraction_l3446_344667


namespace sandy_correct_sums_l3446_344694

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : total_marks = 60) 
  (h3 : correct_marks = 3) 
  (h4 : incorrect_marks = 2) : 
  ∃ (correct : ℕ), correct = 24 ∧ 
    correct + (total_sums - correct) = total_sums ∧ 
    (correct_marks : ℤ) * correct - incorrect_marks * (total_sums - correct) = total_marks :=
by sorry

end sandy_correct_sums_l3446_344694


namespace remainder_theorem_l3446_344617

theorem remainder_theorem (P D D' D'' Q Q' Q'' R R' R'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  (h3 : Q' = Q'' * D'' + R'')
  (h4 : R < D)
  (h5 : R' < D')
  (h6 : R'' < D'') :
  P % (D * D' * D'') = R'' * D * D' + R' * D + R := by
  sorry

end remainder_theorem_l3446_344617


namespace robert_ate_more_chocolates_l3446_344699

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 13

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 4

/-- The difference in chocolates eaten between Robert and Nickel -/
def chocolate_difference : ℕ := robert_chocolates - nickel_chocolates

theorem robert_ate_more_chocolates : chocolate_difference = 9 := by
  sorry

end robert_ate_more_chocolates_l3446_344699


namespace intersection_and_inequality_l3446_344666

-- Define the solution sets
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection
def intersection : Set ℝ := A ∩ B

-- Define the quadratic inequality with parameters a and b
def quadratic_inequality (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- Define the linear inequality with parameters a and b
def linear_inequality (a b : ℝ) : Set ℝ := {x | a*x^2 + x + b < 0}

theorem intersection_and_inequality :
  (intersection = Set.Ioo (-1) 2) ∧
  (∃ a b : ℝ, quadratic_inequality a b = Set.Ioo (-1) 2 → linear_inequality a b = Set.univ) :=
sorry


end intersection_and_inequality_l3446_344666


namespace binomial_product_l3446_344623

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 := by
  sorry

end binomial_product_l3446_344623


namespace red_grapes_count_l3446_344626

/-- Represents the number of fruits in a fruit salad. -/
structure FruitSalad where
  green_grapes : ℕ
  red_grapes : ℕ
  raspberries : ℕ

/-- Defines the conditions for a valid fruit salad. -/
def is_valid_fruit_salad (fs : FruitSalad) : Prop :=
  fs.red_grapes = 3 * fs.green_grapes + 7 ∧
  fs.raspberries = fs.green_grapes - 5 ∧
  fs.green_grapes + fs.red_grapes + fs.raspberries = 102

/-- Theorem stating that in a valid fruit salad, there are 67 red grapes. -/
theorem red_grapes_count (fs : FruitSalad) 
  (h : is_valid_fruit_salad fs) : fs.red_grapes = 67 := by
  sorry


end red_grapes_count_l3446_344626


namespace units_digit_G_500_l3446_344668

/-- The Modified Fermat number for a given n -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_500 : units_digit (G 500) = 2 := by sorry

end units_digit_G_500_l3446_344668


namespace binary_101_equals_5_l3446_344602

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 5 -/
def binary_five : List Bool := [true, false, true]

/-- Theorem stating that the binary representation [1,0,1] is equal to 5 in decimal -/
theorem binary_101_equals_5 : binary_to_decimal binary_five = 5 := by
  sorry

end binary_101_equals_5_l3446_344602


namespace farm_animal_count_l3446_344620

/-- Given a farm with cows and ducks, calculate the total number of animals --/
theorem farm_animal_count (total_legs : ℕ) (num_cows : ℕ) : total_legs = 42 → num_cows = 6 → ∃ (num_ducks : ℕ), num_cows + num_ducks = 15 := by
  sorry

end farm_animal_count_l3446_344620
