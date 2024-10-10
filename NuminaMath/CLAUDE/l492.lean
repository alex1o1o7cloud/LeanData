import Mathlib

namespace union_of_A_and_B_l492_49231

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B : Set ℝ := {x | x > 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > -2} := by
  sorry

end union_of_A_and_B_l492_49231


namespace cost_price_change_l492_49245

theorem cost_price_change (x : ℝ) : 
  (50 * (1 + x / 100) * (1 - x / 100) = 48) → x = 20 :=
by sorry

end cost_price_change_l492_49245


namespace algebraic_expression_value_l492_49249

theorem algebraic_expression_value 
  (a b m n x : ℝ) 
  (h1 : a = -b) 
  (h2 : m * n = 1) 
  (h3 : |x - 2| = 3) : 
  (a + b - m * n) * x + (a + b) ^ 2022 + (-m * n) ^ 2023 = -6 ∨ 
  (a + b - m * n) * x + (a + b) ^ 2022 + (-m * n) ^ 2023 = 0 :=
sorry

end algebraic_expression_value_l492_49249


namespace monotonic_increasing_interval_l492_49229

def f (x : ℝ) := x^2 - 2

theorem monotonic_increasing_interval :
  ∀ x y : ℝ, 0 ≤ x ∧ x < y → f x < f y :=
by sorry

end monotonic_increasing_interval_l492_49229


namespace f_period_and_g_max_l492_49203

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2) * Real.cos (x - Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem f_period_and_g_max :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 3) → g x ≤ M) :=
sorry

end f_period_and_g_max_l492_49203


namespace expression_equals_two_l492_49276

theorem expression_equals_two (α : Real) (h : Real.tan α = 3) :
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π/2 - α) + Real.cos (π/2 + α)) = 2 := by
  sorry

end expression_equals_two_l492_49276


namespace local_minimum_implies_a_eq_2_l492_49213

/-- The function f(x) = x(x-a)² has a local minimum at x = 2 -/
def has_local_minimum (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → f x ≥ f 2

/-- The function f(x) = x(x-a)² -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - a)^2

theorem local_minimum_implies_a_eq_2 :
  ∀ a : ℝ, has_local_minimum (f a) a → a = 2 := by
  sorry

end local_minimum_implies_a_eq_2_l492_49213


namespace identify_brother_l492_49261

-- Define the brothers
inductive Brother : Type
| Tweedledum : Brother
| Tweedledee : Brother

-- Define the card suits
inductive Suit : Type
| Red : Suit
| Black : Suit

-- Define the statement made by one of the brothers
def statement (b : Brother) (s : Suit) : Prop :=
  b = Brother.Tweedledum ∨ s = Suit.Black

-- Define the rule that someone with a black card cannot make a true statement
axiom black_card_rule : ∀ (b : Brother) (s : Suit), 
  s = Suit.Black → ¬(statement b s)

-- Theorem to prove
theorem identify_brother : 
  ∃ (b : Brother) (s : Suit), statement b s ∧ b = Brother.Tweedledum ∧ s = Suit.Red :=
sorry

end identify_brother_l492_49261


namespace happy_valley_farm_arrangement_l492_49243

/-- The number of ways to arrange animals in cages -/
def arrange_animals (chickens dogs cats : Nat) : Nat :=
  Nat.factorial 3 * Nat.factorial chickens * Nat.factorial dogs * Nat.factorial cats

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem happy_valley_farm_arrangement :
  arrange_animals 5 3 4 = 103680 := by
  sorry

end happy_valley_farm_arrangement_l492_49243


namespace test_question_count_l492_49219

theorem test_question_count (total_points : ℕ) (total_questions : ℕ) 
  (h1 : total_points = 200)
  (h2 : total_questions = 30)
  (h3 : ∃ (five_point_count ten_point_count : ℕ), 
    five_point_count + ten_point_count = total_questions ∧
    5 * five_point_count + 10 * ten_point_count = total_points) :
  ∃ (five_point_count : ℕ), five_point_count = 20 ∧
    ∃ (ten_point_count : ℕ), 
      five_point_count + ten_point_count = total_questions ∧
      5 * five_point_count + 10 * ten_point_count = total_points :=
by
  sorry

end test_question_count_l492_49219


namespace real_number_operations_closure_l492_49283

theorem real_number_operations_closure :
  (∀ (a b : ℝ), ∃ (c : ℝ), a + b = c) ∧
  (∀ (a b : ℝ), ∃ (c : ℝ), a - b = c) ∧
  (∀ (a b : ℝ), ∃ (c : ℝ), a * b = c) ∧
  (∀ (a b : ℝ), b ≠ 0 → ∃ (c : ℝ), a / b = c) :=
by sorry

end real_number_operations_closure_l492_49283


namespace ceiling_sqrt_196_l492_49240

theorem ceiling_sqrt_196 : ⌈Real.sqrt 196⌉ = 14 := by
  sorry

end ceiling_sqrt_196_l492_49240


namespace clothing_cost_problem_l492_49232

theorem clothing_cost_problem (total_spent : ℕ) (num_pieces : ℕ) (piece1_cost : ℕ) (piece2_cost : ℕ) (same_cost_piece : ℕ) :
  total_spent = 610 →
  num_pieces = 7 →
  piece1_cost = 49 →
  same_cost_piece = 96 →
  total_spent = piece1_cost + piece2_cost + 5 * same_cost_piece →
  piece2_cost = 81 :=
by
  sorry

end clothing_cost_problem_l492_49232


namespace square_diagonal_l492_49284

theorem square_diagonal (perimeter : ℝ) (h : perimeter = 40) :
  let side := perimeter / 4
  let diagonal := side * Real.sqrt 2
  diagonal = 10 * Real.sqrt 2 := by sorry

end square_diagonal_l492_49284


namespace cricket_players_l492_49239

/-- The number of students who like to play basketball -/
def B : ℕ := 7

/-- The number of students who like to play both basketball and cricket -/
def B_and_C : ℕ := 5

/-- The number of students who like to play basketball or cricket or both -/
def B_or_C : ℕ := 10

/-- The number of students who like to play cricket -/
def C : ℕ := B_or_C - B + B_and_C

theorem cricket_players : C = 8 := by
  sorry

end cricket_players_l492_49239


namespace smaller_number_proof_l492_49268

theorem smaller_number_proof (x y : ℝ) : 
  y = 3 * x + 11 → x + y = 55 → x = 11 := by
sorry

end smaller_number_proof_l492_49268


namespace vector_magnitude_AB_l492_49206

/-- The magnitude of the vector from point A(1, 0) to point B(0, -1) is √2 -/
theorem vector_magnitude_AB : Real.sqrt 2 = Real.sqrt ((0 - 1)^2 + (-1 - 0)^2) := by
  sorry

end vector_magnitude_AB_l492_49206


namespace shanna_garden_harvest_l492_49265

/-- Calculates the total number of vegetables harvested given the initial plant counts and deaths --/
def total_vegetables_harvested (tomato_plants eggplant_plants pepper_plants : ℕ) 
  (tomato_deaths pepper_deaths : ℕ) (vegetables_per_plant : ℕ) : ℕ :=
  let surviving_tomatoes := tomato_plants - tomato_deaths
  let surviving_peppers := pepper_plants - pepper_deaths
  let total_surviving_plants := surviving_tomatoes + surviving_peppers + eggplant_plants
  total_surviving_plants * vegetables_per_plant

/-- Proves that Shanna harvested 56 vegetables given the initial conditions --/
theorem shanna_garden_harvest : 
  total_vegetables_harvested 6 2 4 3 1 7 = 56 := by
  sorry

end shanna_garden_harvest_l492_49265


namespace circle_condition_intersection_condition_l492_49262

-- Define the equation C
def C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem 1: Range of m for C to represent a circle
theorem circle_condition (m : ℝ) :
  (∃ x y, C x y m) ∧ (∀ x y, C x y m → (x - 1)^2 + (y - 2)^2 = 5 - m) →
  m < 5 :=
sorry

-- Theorem 2: Value of m when C intersects l with |MN| = 4√5/5
theorem intersection_condition (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂, C x₁ y₁ m ∧ C x₂ y₂ m ∧ l x₁ y₁ ∧ l x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4*Real.sqrt 5 / 5)^2) →
  m = 4 :=
sorry

end circle_condition_intersection_condition_l492_49262


namespace translated_cosine_monotonicity_l492_49241

open Real

theorem translated_cosine_monotonicity (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 2 * cos (2 * x)) →
  (∀ x, g x = f (x - π / 6)) →
  (∀ x ∈ Set.Icc (2 * a) (7 * π / 6), StrictMono g) →
  a ∈ Set.Icc (π / 3) (7 * π / 12) :=
sorry

end translated_cosine_monotonicity_l492_49241


namespace equation_solution_l492_49247

theorem equation_solution :
  ∃ x : ℝ, (1 / x + (2 / x) / (4 / x) = 3 / 4) ∧ x = 4 := by
  sorry

end equation_solution_l492_49247


namespace parallelogram_area_l492_49288

/-- Proves that the area of a parallelogram with base 7 and altitude twice the base is 98 square units --/
theorem parallelogram_area : ∀ (base altitude area : ℝ),
  base = 7 →
  altitude = 2 * base →
  area = base * altitude →
  area = 98 := by
  sorry

end parallelogram_area_l492_49288


namespace expression_value_l492_49286

theorem expression_value : ((2525 - 2424)^2 + 100) / 225 = 46 := by
  sorry

end expression_value_l492_49286


namespace log_sum_power_twenty_l492_49222

theorem log_sum_power_twenty (log_2 log_5 : ℝ) (h : log_2 + log_5 = 1) :
  (log_2 + log_5)^20 = 1 := by
  sorry

end log_sum_power_twenty_l492_49222


namespace system_solution_ratio_l492_49236

theorem system_solution_ratio (x y c d : ℝ) (h1 : 4*x - 2*y = c) (h2 : 6*y - 12*x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end system_solution_ratio_l492_49236


namespace average_income_B_C_l492_49254

def average_income (x y : ℕ) : ℚ := (x + y) / 2

theorem average_income_B_C 
  (h1 : average_income A_income B_income = 4050)
  (h2 : average_income A_income C_income = 4200)
  (h3 : A_income = 3000) :
  average_income B_income C_income = 5250 :=
by
  sorry


end average_income_B_C_l492_49254


namespace factor_t_squared_minus_49_l492_49252

theorem factor_t_squared_minus_49 : ∀ t : ℝ, t^2 - 49 = (t - 7) * (t + 7) := by
  sorry

end factor_t_squared_minus_49_l492_49252


namespace quadratic_increasing_condition_l492_49293

/-- A quadratic function f(x) = 4x² - mx + 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f m x < f m y

/-- If f(x) = 4x² - mx + 5 is increasing on [-2, +∞), then m ≤ -16 -/
theorem quadratic_increasing_condition (m : ℝ) :
  is_increasing_on_interval m → m ≤ -16 := by
  sorry

end quadratic_increasing_condition_l492_49293


namespace geometric_sequence_sum_l492_49210

/-- Given a geometric sequence {aₙ} with a₁ = 1 and common ratio q ≠ 1,
    if -3a₁, -a₂, and a₃ form an arithmetic sequence,
    then the sum of the first 4 terms (S₄) equals -20. -/
theorem geometric_sequence_sum (q : ℝ) (h1 : q ≠ 1) : 
  let a : ℕ → ℝ := λ n => q^(n-1)
  ∀ n, a n = q^(n-1)
  → -3 * (a 1) + (a 3) = 2 * (-a 2)
  → (a 1) = 1
  → (a 1) + (a 2) + (a 3) + (a 4) = -20 := by
sorry

end geometric_sequence_sum_l492_49210


namespace cosine_sine_difference_equals_sine_double_angle_l492_49244

theorem cosine_sine_difference_equals_sine_double_angle (α : ℝ) :
  (Real.cos (π / 4 - α))^2 - (Real.sin (π / 4 - α))^2 = Real.sin (2 * α) := by
  sorry

end cosine_sine_difference_equals_sine_double_angle_l492_49244


namespace f_min_value_l492_49256

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 4 * y^2 - 12 * x - 8 * y

/-- The theorem stating the minimum value and where it occurs -/
theorem f_min_value :
  (∀ x y : ℝ, f x y ≥ -28) ∧
  f (8/3) (-1) = -28 :=
sorry

end f_min_value_l492_49256


namespace parabola_vertex_l492_49220

/-- The parabola is defined by the equation y = (x - 1)² - 3 -/
def parabola (x : ℝ) : ℝ := (x - 1)^2 - 3

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex -/
def vertex_y : ℝ := -3

/-- Theorem: The vertex of the parabola y = (x - 1)² - 3 has coordinates (1, -3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola vertex_x) ∧ 
  parabola vertex_x = vertex_y := by
  sorry

end parabola_vertex_l492_49220


namespace expand_polynomial_l492_49292

theorem expand_polynomial (x : ℝ) : (5*x^2 + 3*x - 7) * 4*x^3 = 20*x^5 + 12*x^4 - 28*x^3 := by
  sorry

end expand_polynomial_l492_49292


namespace magnitude_of_complex_square_l492_49289

theorem magnitude_of_complex_square : 
  let z : ℂ := (3 + Complex.I) ^ 2
  ‖z‖ = 10 := by sorry

end magnitude_of_complex_square_l492_49289


namespace largest_number_l492_49275

theorem largest_number (a b c : ℝ) : 
  a = (1 : ℝ) / 2 →
  b = Real.log 3 / Real.log 4 →
  c = Real.sin (π / 8) →
  b ≥ a ∧ b ≥ c :=
by sorry

end largest_number_l492_49275


namespace negative_a_range_l492_49290

theorem negative_a_range (a : ℝ) :
  a < 0 →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) →
  a ≤ -2 := by
  sorry

end negative_a_range_l492_49290


namespace martin_goldfish_l492_49209

/-- Calculates the number of goldfish after a given number of weeks -/
def goldfish_after_weeks (initial : ℕ) (die_per_week : ℕ) (buy_per_week : ℕ) (weeks : ℕ) : ℤ :=
  initial - (die_per_week * weeks : ℕ) + (buy_per_week * weeks : ℕ)

/-- Theorem stating the number of goldfish Martin will have after 7 weeks -/
theorem martin_goldfish : goldfish_after_weeks 18 5 3 7 = 4 := by
  sorry

end martin_goldfish_l492_49209


namespace A_inter_B_equals_open_interval_l492_49270

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x | |x - 1| < 2}

theorem A_inter_B_equals_open_interval : A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end A_inter_B_equals_open_interval_l492_49270


namespace largest_decimal_l492_49221

theorem largest_decimal : 
  let a := 0.938
  let b := 0.9389
  let c := 0.93809
  let d := 0.839
  let e := 0.893
  b = max a (max b (max c (max d e))) := by
  sorry

end largest_decimal_l492_49221


namespace problem_statement_l492_49298

theorem problem_statement : ((18^18 / 18^17)^3 * 9^3) / 3^6 = 5832 := by
  sorry

end problem_statement_l492_49298


namespace quadratic_root_k_value_l492_49253

theorem quadratic_root_k_value (k : ℝ) : 
  (2 : ℝ)^2 - k = 5 → k = -1 := by
  sorry

end quadratic_root_k_value_l492_49253


namespace distance_from_point_to_y_axis_l492_49272

-- Define a point in 2D Cartesian coordinate system
def point : ℝ × ℝ := (3, -5)

-- Define the distance from a point to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ := |p.1|

-- Theorem statement
theorem distance_from_point_to_y_axis :
  distance_to_y_axis point = 3 := by
  sorry

end distance_from_point_to_y_axis_l492_49272


namespace concatenated_digits_theorem_l492_49274

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Concatenation of two natural numbers -/
def concatenate (a b : ℕ) : ℕ := sorry

theorem concatenated_digits_theorem :
  num_digits (concatenate (5^1971) (2^1971)) = 1972 := by sorry

end concatenated_digits_theorem_l492_49274


namespace remaining_average_l492_49242

theorem remaining_average (total : ℝ) (avg1 avg2 : ℝ) :
  total = 6 * 5.40 ∧
  avg1 = 5.2 ∧
  avg2 = 5.80 →
  (total - 2 * avg1 - 2 * avg2) / 2 = 5.20 :=
by sorry

end remaining_average_l492_49242


namespace triangle_square_side_ratio_l492_49215

theorem triangle_square_side_ratio (t s : ℝ) : 
  (3 * t = 12) →  -- Perimeter of equilateral triangle
  (4 * s = 12) →  -- Perimeter of square
  (t / s = 4 / 3) :=  -- Ratio of side lengths
by
  sorry

end triangle_square_side_ratio_l492_49215


namespace sum_of_common_ratios_is_three_l492_49278

-- Define the geometric sequences and their properties
def geometric_sequence (k a₂ a₃ : ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2

-- Define the theorem
theorem sum_of_common_ratios_is_three
  (k a₂ a₃ b₂ b₃ : ℝ)
  (h₁ : geometric_sequence k a₂ a₃)
  (h₂ : geometric_sequence k b₂ b₃)
  (h₃ : ∃ p r : ℝ, p ≠ r ∧ a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2)
  (h₄ : a₃ - b₃ = 3 * (a₂ - b₂))
  (h₅ : k ≠ 0) :
  ∃ p r : ℝ, p + r = 3 ∧ 
    geometric_sequence k a₂ a₃ ∧
    geometric_sequence k b₂ b₃ ∧
    a₂ = k * p ∧ a₃ = k * p^2 ∧
    b₂ = k * r ∧ b₃ = k * r^2 :=
sorry

end sum_of_common_ratios_is_three_l492_49278


namespace solve_system_l492_49234

theorem solve_system (c d : ℤ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 6 + d = 9 + c) : 
  5 - c = 6 := by
sorry

end solve_system_l492_49234


namespace train_speed_l492_49238

/-- The speed of a train given its length and time to cross a point -/
theorem train_speed (length time : ℝ) (h1 : length = 400) (h2 : time = 20) :
  length / time = 20 := by
  sorry

end train_speed_l492_49238


namespace min_distance_to_hyperbola_l492_49212

/-- The minimum distance between A(4,4) and P(x, 1/x) where x > 0 is √14 -/
theorem min_distance_to_hyperbola :
  let A : ℝ × ℝ := (4, 4)
  let P : ℝ → ℝ × ℝ := fun x ↦ (x, 1/x)
  let distance (x : ℝ) : ℝ := Real.sqrt ((P x).1 - A.1)^2 + ((P x).2 - A.2)^2
  ∀ x > 0, distance x ≥ Real.sqrt 14 ∧ ∃ x₀ > 0, distance x₀ = Real.sqrt 14 :=
by sorry

end min_distance_to_hyperbola_l492_49212


namespace sector_central_angle_l492_49225

theorem sector_central_angle (r : ℝ) (α : ℝ) : 
  r > 0 → 
  r * α = 6 → 
  (1/2) * r * r * α = 6 → 
  α = 3 := by
sorry

end sector_central_angle_l492_49225


namespace arithmetic_mean_odd_primes_under_30_l492_49217

def odd_primes_under_30 : List Nat := [3, 5, 7, 11, 13, 17, 19, 23, 29]

def arithmetic_mean (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem arithmetic_mean_odd_primes_under_30 :
  arithmetic_mean odd_primes_under_30 = 14 := by sorry

end arithmetic_mean_odd_primes_under_30_l492_49217


namespace radhika_video_games_l492_49277

/-- The number of video games Radhika received on Christmas. -/
def christmas_games : ℕ := 12

/-- The number of video games Radhika received on her birthday. -/
def birthday_games : ℕ := 8

/-- The number of video games Radhika already owned. -/
def owned_games : ℕ := (christmas_games + birthday_games) / 2

/-- The total number of video games Radhika owns now. -/
def total_games : ℕ := christmas_games + birthday_games + owned_games

theorem radhika_video_games :
  total_games = 30 :=
by sorry

end radhika_video_games_l492_49277


namespace acute_angle_x_l492_49279

theorem acute_angle_x (x : Real) (h : 0 < x ∧ x < π / 2) 
  (eq : Real.sin (3 * π / 5) * Real.cos x + Real.cos (2 * π / 5) * Real.sin x = Real.sqrt 3 / 2) : 
  x = 4 * π / 15 := by
  sorry

end acute_angle_x_l492_49279


namespace train_length_proof_l492_49271

/-- Given a train that crosses an electric pole in 30 seconds at a speed of 43.2 km/h,
    prove that its length is 360 meters. -/
theorem train_length_proof (crossing_time : ℝ) (speed_kmh : ℝ) (length : ℝ) : 
  crossing_time = 30 →
  speed_kmh = 43.2 →
  length = speed_kmh * 1000 / 3600 * crossing_time →
  length = 360 := by
  sorry

#check train_length_proof

end train_length_proof_l492_49271


namespace billy_homework_problem_l492_49296

theorem billy_homework_problem (first_hour second_hour third_hour total : ℕ) : 
  first_hour > 0 →
  second_hour = 2 * first_hour →
  third_hour = 3 * first_hour →
  third_hour = 132 →
  total = first_hour + second_hour + third_hour →
  total = 264 := by
  sorry

end billy_homework_problem_l492_49296


namespace binomial_12_11_l492_49202

theorem binomial_12_11 : (12 : ℕ).choose 11 = 12 := by
  sorry

end binomial_12_11_l492_49202


namespace midpoint_barycentric_coords_l492_49260

/-- Barycentric coordinates of a point -/
structure BarycentricCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Given two points in barycentric coordinates, compute the midpoint -/
def midpoint_barycentric (M N : BarycentricCoord) : Prop :=
  let m := M.x + M.y + M.z
  let n := N.x + N.y + N.z
  ∃ (k : ℝ) (S : BarycentricCoord), k ≠ 0 ∧
    S.x = k * (M.x / (2 * m) + N.x / (2 * n)) ∧
    S.y = k * (M.y / (2 * m) + N.y / (2 * n)) ∧
    S.z = k * (M.z / (2 * m) + N.z / (2 * n))

theorem midpoint_barycentric_coords (M N : BarycentricCoord) : 
  midpoint_barycentric M N := by sorry

end midpoint_barycentric_coords_l492_49260


namespace unique_zamena_assignment_l492_49235

def digit := Fin 5

structure Assignment where
  Z : digit
  A : digit
  M : digit
  E : digit
  N : digit
  H : digit

def satisfies_inequalities (a : Assignment) : Prop :=
  (3 > a.A.val + 1) ∧ 
  (a.A.val > a.M.val) ∧ 
  (a.M.val < a.E.val) ∧ 
  (a.E.val < a.H.val) ∧ 
  (a.H.val < a.A.val)

def all_different (a : Assignment) : Prop :=
  a.Z ≠ a.A ∧ a.Z ≠ a.M ∧ a.Z ≠ a.E ∧ a.Z ≠ a.N ∧ a.Z ≠ a.H ∧
  a.A ≠ a.M ∧ a.A ≠ a.E ∧ a.A ≠ a.N ∧ a.A ≠ a.H ∧
  a.M ≠ a.E ∧ a.M ≠ a.N ∧ a.M ≠ a.H ∧
  a.E ≠ a.N ∧ a.E ≠ a.H ∧
  a.N ≠ a.H

def zamena_value (a : Assignment) : ℕ :=
  100000 * (a.Z.val + 1) + 10000 * (a.A.val + 1) + 1000 * (a.M.val + 1) +
  100 * (a.E.val + 1) + 10 * (a.N.val + 1) + (a.A.val + 1)

theorem unique_zamena_assignment :
  ∀ a : Assignment, 
    satisfies_inequalities a → all_different a → 
    zamena_value a = 541234 :=
sorry

end unique_zamena_assignment_l492_49235


namespace chicken_nuggets_cost_l492_49267

/-- Calculates the total cost of chicken nuggets including discount and tax -/
def total_cost (nuggets : ℕ) (box_size : ℕ) (box_price : ℚ) (discount_threshold : ℕ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let boxes := nuggets / box_size
  let initial_cost := boxes * box_price
  let discounted_cost := if nuggets ≥ discount_threshold then initial_cost * (1 - discount_rate) else initial_cost
  let total := discounted_cost * (1 + tax_rate)
  total

/-- The problem statement -/
theorem chicken_nuggets_cost :
  total_cost 100 20 4 80 (75/1000) (8/100) = 1998/100 :=
sorry

end chicken_nuggets_cost_l492_49267


namespace weeks_to_save_for_coat_l492_49269

/-- Calculates the number of weeks needed to save for a coat given specific conditions -/
theorem weeks_to_save_for_coat (weekly_savings : ℚ) (bill_fraction : ℚ) (gift : ℚ) (coat_cost : ℚ) :
  weekly_savings = 25 ∧ 
  bill_fraction = 1/3 ∧ 
  gift = 70 ∧ 
  coat_cost = 170 →
  ∃ w : ℕ, w * weekly_savings - (bill_fraction * 7 * weekly_savings) + gift = coat_cost ∧ w = 19 :=
by sorry

end weeks_to_save_for_coat_l492_49269


namespace election_votes_l492_49257

theorem election_votes (total_votes : ℕ) 
  (winning_percentage : ℚ) (vote_majority : ℕ) :
  winning_percentage = 70 / 100 →
  vote_majority = 192 →
  (winning_percentage * total_votes - (1 - winning_percentage) * total_votes : ℚ) = vote_majority →
  total_votes = 480 :=
by sorry

end election_votes_l492_49257


namespace line_equation_from_slope_and_intercept_l492_49204

/-- Given a line with slope 6 and y-intercept -4, its equation is 6x - y - 4 = 0 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (f : ℝ → ℝ),
  (∀ x y : ℝ, f y - f x = 6 * (y - x)) →  -- slope is 6
  (f 0 = -4) →                           -- y-intercept is -4
  ∀ x : ℝ, 6 * x - f x - 4 = 0 :=
by sorry


end line_equation_from_slope_and_intercept_l492_49204


namespace prob_not_overcome_is_half_l492_49273

-- Define the set of elements
inductive Element : Type
| Metal : Element
| Wood : Element
| Water : Element
| Fire : Element
| Earth : Element

-- Define the overcoming relation
def overcomes : Element → Element → Prop
| Element.Metal, Element.Wood => True
| Element.Wood, Element.Earth => True
| Element.Earth, Element.Water => True
| Element.Water, Element.Fire => True
| Element.Fire, Element.Metal => True
| _, _ => False

-- Define the probability of selecting two elements that do not overcome each other
def prob_not_overcome : ℚ :=
  let total_pairs := (5 * 4) / 2  -- C(5,2)
  let overcoming_pairs := 5       -- Number of overcoming relationships
  1 - (overcoming_pairs : ℚ) / total_pairs

-- State the theorem
theorem prob_not_overcome_is_half : prob_not_overcome = 1/2 := by
  sorry

end prob_not_overcome_is_half_l492_49273


namespace min_fraction_value_l492_49207

theorem min_fraction_value (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100)
  (h : y^2 - 1 = a^2 * (x^2 - 1)) :
  ∃ (k : ℚ), k = 2 ∧ (∀ (a' x' : ℕ), a' > 100 → x' > 100 → ∃ (y' : ℕ), y' > 100 ∧ 
    y'^2 - 1 = a'^2 * (x'^2 - 1) → (a' : ℚ) / x' ≥ k) ∧
  (∃ (a'' x'' y'' : ℕ), a'' > 100 ∧ x'' > 100 ∧ y'' > 100 ∧
    y''^2 - 1 = a''^2 * (x''^2 - 1) ∧ (a'' : ℚ) / x'' = k) :=
by sorry

end min_fraction_value_l492_49207


namespace kindergarten_tissues_l492_49285

/-- The number of tissues in a mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The number of students in the first kindergartner group -/
def group1_size : ℕ := 9

/-- The number of students in the second kindergartner group -/
def group2_size : ℕ := 10

/-- The number of students in the third kindergartner group -/
def group3_size : ℕ := 11

/-- The total number of tissues brought by all kindergartner groups -/
def total_tissues : ℕ := (group1_size + group2_size + group3_size) * tissues_per_box

theorem kindergarten_tissues : total_tissues = 1200 := by
  sorry

end kindergarten_tissues_l492_49285


namespace complex_modulus_of_fraction_l492_49291

theorem complex_modulus_of_fraction (z : ℂ) : z = (2 - I) / (2 + I) → Complex.abs z = 1 := by
  sorry

end complex_modulus_of_fraction_l492_49291


namespace convex_quadrilateral_probability_l492_49211

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from total_chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def favorable_outcomes : ℕ := n.choose k

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := favorable_outcomes / total_selections

theorem convex_quadrilateral_probability : probability = 2 / 585 := by
  sorry

end convex_quadrilateral_probability_l492_49211


namespace solution_count_is_correct_l492_49214

/-- The number of groups of integer solutions for the equation xyz = 2009 -/
def solution_count : ℕ := 72

/-- A function that counts the number of groups of integer solutions for xyz = 2009 -/
noncomputable def count_solutions : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of groups of integer solutions for xyz = 2009 is 72 -/
theorem solution_count_is_correct : count_solutions = solution_count := by
  sorry

end solution_count_is_correct_l492_49214


namespace circle_area_with_diameter_8_l492_49201

theorem circle_area_with_diameter_8 (π : ℝ) :
  let diameter : ℝ := 8
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 16 * π := by sorry

end circle_area_with_diameter_8_l492_49201


namespace dollar_op_six_three_l492_49246

def dollar_op (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem dollar_op_six_three : dollar_op 6 3 = 18 := by
  sorry

end dollar_op_six_three_l492_49246


namespace reciprocal_of_negative_five_squared_l492_49248

theorem reciprocal_of_negative_five_squared :
  ((-5 : ℝ)^2)⁻¹ = (1 / 25 : ℝ) := by
  sorry

end reciprocal_of_negative_five_squared_l492_49248


namespace distance_is_1000_l492_49281

/-- The distance between Liang Liang's home and school in meters. -/
def distance : ℝ := sorry

/-- The time taken (in minutes) when walking at 40 meters per minute. -/
def time_at_40 : ℝ := sorry

/-- Assertion that distance equals speed multiplied by time for 40 m/min speed. -/
axiom distance_eq_40_times_time : distance = 40 * time_at_40

/-- Assertion that distance equals speed multiplied by time for 50 m/min speed. -/
axiom distance_eq_50_times_time_minus_5 : distance = 50 * (time_at_40 - 5)

theorem distance_is_1000 : distance = 1000 := by sorry

end distance_is_1000_l492_49281


namespace negative_reciprocal_inequality_l492_49295

theorem negative_reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  -1/a < -1/b := by
  sorry

end negative_reciprocal_inequality_l492_49295


namespace number_difference_l492_49205

theorem number_difference (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : 0.075 * A = 0.125 * B) (h4 : A = 2430) : A - B = 972 := by
  sorry

end number_difference_l492_49205


namespace circle_packing_theorem_l492_49250

theorem circle_packing_theorem :
  ∃ (n : ℕ+), (n : ℝ) / 2 > 2008 ∧
  ∀ (i j : Fin (n^2)), i ≠ j →
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧
  (x - (i.val % n : ℕ) / n)^2 + (y - (i.val / n : ℕ) / n)^2 ≤ (1 / (2*n))^2 ∧
  (x - (j.val % n : ℕ) / n)^2 + (y - (j.val / n : ℕ) / n)^2 ≤ (1 / (2*n))^2 →
  (x - (i.val % n : ℕ) / n)^2 + (y - (i.val / n : ℕ) / n)^2 = (1 / (2*n))^2 ∨
  (x - (j.val % n : ℕ) / n)^2 + (y - (j.val / n : ℕ) / n)^2 = (1 / (2*n))^2 ∨
  ((x - (i.val % n : ℕ) / n) - (x - (j.val % n : ℕ) / n))^2 +
  ((y - (i.val / n : ℕ) / n) - (y - (j.val / n : ℕ) / n))^2 ≥ (1 / n)^2 :=
by sorry

end circle_packing_theorem_l492_49250


namespace nick_running_speed_l492_49251

/-- Represents the speed required for the fourth lap to achieve a target average speed -/
def fourth_lap_speed (first_three_speed : ℝ) (target_avg_speed : ℝ) : ℝ :=
  4 * target_avg_speed - 3 * first_three_speed

/-- Proves that if a runner completes three laps at 9 mph and needs to achieve an average 
    speed of 10 mph for four laps, then the speed required for the fourth lap is 15 mph -/
theorem nick_running_speed : fourth_lap_speed 9 10 = 15 := by
  sorry

end nick_running_speed_l492_49251


namespace benny_work_hours_l492_49282

theorem benny_work_hours (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) : 
  hours_per_day = 5 → days_worked = 12 → total_hours = hours_per_day * days_worked → total_hours = 60 := by
  sorry

end benny_work_hours_l492_49282


namespace first_cross_fraction_solution_second_cross_fraction_solution_third_cross_fraction_solution_l492_49280

/-- Definition of a cross fraction equation -/
def is_cross_fraction_equation (m n x : ℝ) : Prop :=
  m ≠ 0 ∧ n ≠ 0 ∧ x + m * n / x = m + n

/-- Theorem for the first cross fraction equation -/
theorem first_cross_fraction_solution :
  ∀ x₁ x₂ : ℝ, is_cross_fraction_equation (-3) (-4) x₁ ∧ is_cross_fraction_equation (-3) (-4) x₂ →
  (x₁ = -3 ∧ x₂ = -4) ∨ (x₁ = -4 ∧ x₂ = -3) :=
sorry

/-- Theorem for the second cross fraction equation -/
theorem second_cross_fraction_solution :
  ∀ a b : ℝ, is_cross_fraction_equation a b a ∧ is_cross_fraction_equation a b b →
  b / a + a / b + 1 = -31 / 6 :=
sorry

/-- Theorem for the third cross fraction equation -/
theorem third_cross_fraction_solution :
  ∀ k x₁ x₂ : ℝ, k > 2 → x₁ > x₂ →
  is_cross_fraction_equation (2023 * k - 2022) 1 x₁ ∧ is_cross_fraction_equation (2023 * k - 2022) 1 x₂ →
  (x₁ + 4044) / x₂ = 2022 :=
sorry

end first_cross_fraction_solution_second_cross_fraction_solution_third_cross_fraction_solution_l492_49280


namespace correct_reasoning_definitions_l492_49223

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
  | PartToWhole
  | GeneralToSpecific
  | SpecificToSpecific

-- Define a function that maps a reasoning type to its correct direction
def correct_reasoning_direction : ReasoningType → ReasoningDirection
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating that the correct reasoning directions are as defined
theorem correct_reasoning_definitions :
  (correct_reasoning_direction ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (correct_reasoning_direction ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (correct_reasoning_direction ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end correct_reasoning_definitions_l492_49223


namespace mary_remaining_money_l492_49237

-- Define the initial amount Mary received
def initial_amount : ℚ := 150

-- Define the original price of the video game
def game_price : ℚ := 60

-- Define the discount rate for the video game
def game_discount_rate : ℚ := 0.15

-- Define the percentage spent on goggles
def goggles_spend_rate : ℚ := 0.20

-- Define the sales tax rate for the goggles
def goggles_tax_rate : ℚ := 0.08

-- Function to calculate the discounted price of the video game
def discounted_game_price : ℚ :=
  game_price * (1 - game_discount_rate)

-- Function to calculate the amount left after buying the video game
def amount_after_game : ℚ :=
  initial_amount - discounted_game_price

-- Function to calculate the price of the goggles before tax
def goggles_price_before_tax : ℚ :=
  amount_after_game * goggles_spend_rate

-- Function to calculate the total price of the goggles including tax
def goggles_total_price : ℚ :=
  goggles_price_before_tax * (1 + goggles_tax_rate)

-- Theorem stating that Mary has $77.62 left after her shopping trip
theorem mary_remaining_money :
  initial_amount - discounted_game_price - goggles_total_price = 77.62 := by
  sorry

end mary_remaining_money_l492_49237


namespace resultant_of_quadratics_l492_49266

/-- The resultant of two quadratic polynomials -/
def resultant (a b p q : ℝ) : ℝ :=
  (p - a) * (p * b - a * q) + (q - b)^2

/-- Roots of a quadratic polynomial -/
structure QuadraticRoots (a b : ℝ) where
  x₁ : ℝ
  x₂ : ℝ
  sum : x₁ + x₂ = -a
  product : x₁ * x₂ = b

theorem resultant_of_quadratics (a b p q : ℝ) 
  (f_roots : QuadraticRoots a b) (g_roots : QuadraticRoots p q) :
  (f_roots.x₁ - g_roots.x₁) * (f_roots.x₁ - g_roots.x₂) * 
  (f_roots.x₂ - g_roots.x₁) * (f_roots.x₂ - g_roots.x₂) = 
  resultant a b p q := by
  sorry

end resultant_of_quadratics_l492_49266


namespace abes_age_l492_49287

theorem abes_age (present_age : ℕ) 
  (h : present_age + (present_age - 7) = 27) : 
  present_age = 17 := by
sorry

end abes_age_l492_49287


namespace a_bounds_l492_49230

theorem a_bounds (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 3)
  (square_sum_condition : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) :
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end a_bounds_l492_49230


namespace hex_to_decimal_equality_l492_49200

/-- Represents a hexadecimal digit as a natural number -/
def HexDigit := Fin 16

/-- Converts a base-6 number to decimal -/
def toDecimal (a b c d e : Fin 6) : ℕ :=
  e + 6 * d + 6^2 * c + 6^3 * b + 6^4 * a

/-- The theorem stating that if 3m502₍₆₎ = 4934 in decimal, then m = 4 -/
theorem hex_to_decimal_equality (m : Fin 6) :
  toDecimal 3 m 5 0 2 = 4934 → m = 4 := by
  sorry


end hex_to_decimal_equality_l492_49200


namespace ellipse_equation_l492_49294

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  major_axis : ℝ
  focal_distance : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / (e.major_axis/2)^2 + y^2 / ((e.major_axis/2)^2 - (e.focal_distance/2)^2) = 1

/-- Theorem stating the standard equation for a specific ellipse -/
theorem ellipse_equation (e : Ellipse) 
    (h1 : e.center = (0, 0))
    (h2 : e.major_axis = 18)
    (h3 : e.focal_distance = 12) :
    ∀ x y : ℝ, standard_equation e x y ↔ x^2/81 + y^2/45 = 1 := by
  sorry

end ellipse_equation_l492_49294


namespace parallelepiped_intersection_length_l492_49263

/-- A parallelepiped with points A, B, C, D, A₁, B₁, C₁, D₁ -/
structure Parallelepiped (V : Type*) [NormedAddCommGroup V] :=
  (A B C D A₁ B₁ C₁ D₁ : V)

/-- Point X on edge A₁D₁ -/
def X {V : Type*} [NormedAddCommGroup V] (p : Parallelepiped V) : V :=
  p.A₁ + 5 • (p.D₁ - p.A₁)

/-- Point Y on edge BC -/
def Y {V : Type*} [NormedAddCommGroup V] (p : Parallelepiped V) : V :=
  p.B + 3 • (p.C - p.B)

/-- Intersection point Z of plane C₁XY and ray DA -/
noncomputable def Z {V : Type*} [NormedAddCommGroup V] (p : Parallelepiped V) : V :=
  sorry

/-- Theorem stating that DZ = 20 -/
theorem parallelepiped_intersection_length
  {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (p : Parallelepiped V) :
  ‖p.D - Z p‖ = 20 ∧ ‖p.B₁ - p.C₁‖ = 14 :=
sorry

end parallelepiped_intersection_length_l492_49263


namespace number_puzzle_l492_49233

theorem number_puzzle : ∃ x : ℝ, 13 * x = x + 180 ∧ x = 15 := by sorry

end number_puzzle_l492_49233


namespace bug_visits_29_tiles_l492_49258

/-- Represents a rectangular floor --/
structure RectangularFloor where
  width : ℕ
  length : ℕ

/-- Calculates the number of tiles a bug visits when walking diagonally across a rectangular floor --/
def tilesVisited (floor : RectangularFloor) : ℕ :=
  floor.width + floor.length - Nat.gcd floor.width floor.length

/-- The specific floor in the problem --/
def problemFloor : RectangularFloor :=
  { width := 11, length := 19 }

/-- Theorem stating that a bug walking diagonally across the problem floor visits 29 tiles --/
theorem bug_visits_29_tiles : tilesVisited problemFloor = 29 := by
  sorry

end bug_visits_29_tiles_l492_49258


namespace train_speed_calculation_l492_49228

/-- Calculates the speed of a train given its length, time to cross a bridge, and total length of bridge and train. -/
theorem train_speed_calculation 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (total_length : ℝ) 
  (h1 : train_length = 130) 
  (h2 : crossing_time = 30) 
  (h3 : total_length = 245) : 
  (total_length - train_length) / crossing_time * 3.6 = 45 :=
by sorry

end train_speed_calculation_l492_49228


namespace complement_of_angle_l492_49299

theorem complement_of_angle (A : ℝ) : A = 35 → 180 - A = 145 := by
  sorry

end complement_of_angle_l492_49299


namespace tangent_length_is_6_l492_49224

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the line passing through the center
def line_equation (x y : ℝ) (a : ℝ) : Prop :=
  x + a*y - 1 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 1)

-- Define point A
def point_A (a : ℝ) : ℝ × ℝ := (-4, a)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem tangent_length_is_6 :
  ∃ (a : ℝ),
    line_equation (circle_center.1) (circle_center.2) a ∧
    (∃ (x y : ℝ), circle_equation x y ∧
      ∃ (B : ℝ × ℝ), B.1 = x ∧ B.2 = y ∧
        (point_A a).1 - B.1 = a * (B.2 - (point_A a).2) ∧
        Real.sqrt (((point_A a).1 - B.1)^2 + ((point_A a).2 - B.2)^2) = 6) :=
by sorry

end tangent_length_is_6_l492_49224


namespace Q_subset_P_l492_49218

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by
  sorry

end Q_subset_P_l492_49218


namespace parallel_implies_x_half_perpendicular_implies_x_two_or_neg_two_l492_49264

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define u and v
def u (x : ℝ) : ℝ × ℝ := a + b x
def v (x : ℝ) : ℝ × ℝ := a - b x

-- Helper function to check if two vectors are parallel
def isParallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v1.1 * k = v2.1 ∧ v1.2 * k = v2.2

-- Helper function to check if two vectors are perpendicular
def isPerpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Theorem for part I
theorem parallel_implies_x_half :
  ∀ x : ℝ, isParallel (u x) (v x) → x = 1/2 := by sorry

-- Theorem for part II
theorem perpendicular_implies_x_two_or_neg_two :
  ∀ x : ℝ, isPerpendicular (u x) (v x) → x = 2 ∨ x = -2 := by sorry

end parallel_implies_x_half_perpendicular_implies_x_two_or_neg_two_l492_49264


namespace solve_simple_interest_l492_49227

def simple_interest_problem (principal : ℝ) (interest_paid : ℝ) : Prop :=
  ∃ (rate : ℝ),
    principal = 900 ∧
    interest_paid = 729 ∧
    rate > 0 ∧
    rate < 100 ∧
    interest_paid = (principal * rate * rate) / 100 ∧
    rate = 9

theorem solve_simple_interest :
  ∀ (principal interest_paid : ℝ),
    simple_interest_problem principal interest_paid :=
  sorry

end solve_simple_interest_l492_49227


namespace allan_brought_six_balloons_l492_49226

/-- The number of balloons Jake initially brought to the park -/
def jake_initial_balloons : ℕ := 2

/-- The number of balloons Jake bought at the park -/
def jake_bought_balloons : ℕ := 3

/-- The difference between Allan's and Jake's balloon count -/
def allan_jake_difference : ℕ := 1

/-- The total number of balloons Jake had in the park -/
def jake_total_balloons : ℕ := jake_initial_balloons + jake_bought_balloons

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := jake_total_balloons + allan_jake_difference

theorem allan_brought_six_balloons : allan_balloons = 6 := by
  sorry

end allan_brought_six_balloons_l492_49226


namespace unique_products_count_l492_49259

def bag_A : Finset ℕ := {1, 3, 5, 7}
def bag_B : Finset ℕ := {2, 4, 6, 8}

theorem unique_products_count : 
  Finset.card ((bag_A.product bag_B).image (λ (p : ℕ × ℕ) => p.1 * p.2)) = 15 := by
  sorry

end unique_products_count_l492_49259


namespace symmetric_points_sum_l492_49297

/-- Two points are symmetric about the x-axis if their x-coordinates are equal and 
    their y-coordinates are equal in magnitude but opposite in sign -/
def symmetric_about_x_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = x₂ ∧ y₁ = -y₂

/-- Given two points A(2,m) and B(n,-3) that are symmetric about the x-axis,
    prove that m + n = 5 -/
theorem symmetric_points_sum (m n : ℝ) 
  (h : symmetric_about_x_axis 2 m n (-3)) : m + n = 5 := by
  sorry

end symmetric_points_sum_l492_49297


namespace water_hyacinth_demonstrates_interconnection_and_diversity_l492_49208

/-- Represents the introduction and effects of water hyacinth -/
structure WaterHyacinthIntroduction where
  introduced_as_fodder : Prop
  rapid_spread : Prop
  decrease_native_species : Prop
  water_pollution : Prop
  increase_mosquitoes : Prop

/-- Represents the philosophical conclusions drawn from the water hyacinth case -/
structure PhilosophicalConclusions where
  universal_interconnection : Prop
  diverse_connections : Prop

/-- Theorem stating that the introduction of water hyacinth demonstrates universal interconnection and diverse connections -/
theorem water_hyacinth_demonstrates_interconnection_and_diversity 
  (wh : WaterHyacinthIntroduction) : PhilosophicalConclusions :=
by sorry

end water_hyacinth_demonstrates_interconnection_and_diversity_l492_49208


namespace girls_combined_average_score_l492_49255

theorem girls_combined_average_score 
  (f1 l1 f2 l2 : ℕ) 
  (h1 : (71 * f1 + 76 * l1) / (f1 + l1) = 74)
  (h2 : (81 * f2 + 90 * l2) / (f2 + l2) = 84)
  (h3 : (71 * f1 + 81 * f2) / (f1 + f2) = 79)
  : (76 * l1 + 90 * l2) / (l1 + l2) = 84 := by
  sorry


end girls_combined_average_score_l492_49255


namespace tenth_term_of_arithmetic_progression_l492_49216

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

/-- Theorem: The 10th term of an arithmetic progression with first term 8 and common difference 2 is 26 -/
theorem tenth_term_of_arithmetic_progression :
  arithmeticProgressionTerm 8 2 10 = 26 := by
  sorry

end tenth_term_of_arithmetic_progression_l492_49216
