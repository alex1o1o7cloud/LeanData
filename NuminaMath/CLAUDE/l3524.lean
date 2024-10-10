import Mathlib

namespace angle_value_l3524_352492

theorem angle_value (PQR : ℝ) (x : ℝ) : 
  PQR = 90 → 2*x + x = PQR → x = 30 := by sorry

end angle_value_l3524_352492


namespace basketball_games_count_l3524_352468

/-- Proves that a basketball team played 94 games in a season given specific conditions -/
theorem basketball_games_count :
  ∀ (total_games : ℕ) 
    (first_40_wins : ℕ) 
    (remaining_wins : ℕ),
  first_40_wins = 14 →  -- 35% of 40 games
  remaining_wins ≥ (0.7 : ℝ) * (total_games - 40) →  -- At least 70% of remaining games
  first_40_wins + remaining_wins = (0.55 : ℝ) * total_games →  -- 55% total win rate
  total_games = 94 := by
sorry

end basketball_games_count_l3524_352468


namespace mean_home_runs_l3524_352498

def num_players : List ℕ := [7, 5, 4, 2, 1]
def home_runs : List ℕ := [5, 6, 8, 9, 11]

theorem mean_home_runs : 
  (List.sum (List.zipWith (· * ·) num_players home_runs)) / (List.sum num_players) = 126 / 19 := by
  sorry

end mean_home_runs_l3524_352498


namespace log_condition_equivalence_l3524_352428

theorem log_condition_equivalence (m n : ℝ) (hm : m > 0 ∧ m ≠ 1) (hn : n > 0) :
  Real.log n / Real.log m < 0 ↔ (m - 1) * (n - 1) < 0 := by
  sorry

end log_condition_equivalence_l3524_352428


namespace income_ratio_l3524_352401

/-- Represents a person's financial information -/
structure Person where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup -/
def problem_setup (p1 p2 : Person) : Prop :=
  p1.income = 5000 ∧
  p1.savings = 2000 ∧
  p2.savings = 2000 ∧
  3 * p2.expenditure = 2 * p1.expenditure ∧
  p1.income = p1.expenditure + p1.savings ∧
  p2.income = p2.expenditure + p2.savings

/-- The theorem to prove -/
theorem income_ratio (p1 p2 : Person) :
  problem_setup p1 p2 → 5 * p2.income = 4 * p1.income :=
by
  sorry


end income_ratio_l3524_352401


namespace matching_shoes_probability_l3524_352491

/-- The number of pairs of shoes in the box -/
def num_pairs : ℕ := 7

/-- The total number of shoes in the box -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of ways to select two shoes from the box -/
def total_combinations : ℕ := (total_shoes * (total_shoes - 1)) / 2

/-- The number of ways to select a matching pair of shoes -/
def matching_pairs : ℕ := num_pairs

/-- The probability of selecting a matching pair of shoes -/
def probability : ℚ := matching_pairs / total_combinations

theorem matching_shoes_probability :
  probability = 1 / 13 := by sorry

end matching_shoes_probability_l3524_352491


namespace cubic_difference_l3524_352443

theorem cubic_difference (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 2 * x + y = 16) : 
  x^3 - y^3 = -448 := by
sorry

end cubic_difference_l3524_352443


namespace replacement_solution_concentration_l3524_352487

/-- Given an 80% chemical solution, if 50% of it is replaced with a solution
    of unknown concentration P%, resulting in a 50% chemical solution,
    then P% must be 20%. -/
theorem replacement_solution_concentration
  (original_concentration : ℝ)
  (replaced_fraction : ℝ)
  (final_concentration : ℝ)
  (replacement_concentration : ℝ)
  (h1 : original_concentration = 0.8)
  (h2 : replaced_fraction = 0.5)
  (h3 : final_concentration = 0.5)
  (h4 : final_concentration = (1 - replaced_fraction) * original_concentration
                            + replaced_fraction * replacement_concentration) :
  replacement_concentration = 0.2 := by
sorry

end replacement_solution_concentration_l3524_352487


namespace visited_none_count_l3524_352450

/-- Represents the number of people who have visited a country or combination of countries. -/
structure VisitCount where
  total : Nat
  iceland : Nat
  norway : Nat
  sweden : Nat
  all_three : Nat
  iceland_norway : Nat
  iceland_sweden : Nat
  norway_sweden : Nat

/-- Calculates the number of people who have visited neither Iceland, Norway, nor Sweden. -/
def people_visited_none (vc : VisitCount) : Nat :=
  vc.total - (vc.iceland + vc.norway + vc.sweden - vc.iceland_norway - vc.iceland_sweden - vc.norway_sweden + vc.all_three)

/-- Theorem stating that given the conditions, 42 people have visited neither country. -/
theorem visited_none_count (vc : VisitCount) 
  (h_total : vc.total = 100)
  (h_iceland : vc.iceland = 45)
  (h_norway : vc.norway = 37)
  (h_sweden : vc.sweden = 21)
  (h_all_three : vc.all_three = 12)
  (h_iceland_norway : vc.iceland_norway = 20)
  (h_iceland_sweden : vc.iceland_sweden = 15)
  (h_norway_sweden : vc.norway_sweden = 10) :
  people_visited_none vc = 42 := by
  sorry

end visited_none_count_l3524_352450


namespace total_birds_on_fence_l3524_352406

def birds_on_fence (initial : ℕ) (additional : ℕ) : ℕ := initial + additional

theorem total_birds_on_fence :
  birds_on_fence 12 8 = 20 := by sorry

end total_birds_on_fence_l3524_352406


namespace sum_of_inverse_conjugates_l3524_352449

theorem sum_of_inverse_conjugates (m n : ℝ) : 
  m = (Real.sqrt 2 - 1)⁻¹ → n = (Real.sqrt 2 + 1)⁻¹ → m + n = 2 * Real.sqrt 2 := by
  sorry

end sum_of_inverse_conjugates_l3524_352449


namespace raghu_investment_l3524_352465

/-- Represents the investment amounts of Raghu, Trishul, and Vishal --/
structure Investments where
  raghu : ℝ
  trishul : ℝ
  vishal : ℝ

/-- Defines the conditions of the investment problem --/
def InvestmentConditions (i : Investments) : Prop :=
  i.trishul = 0.9 * i.raghu ∧
  i.vishal = 1.1 * i.trishul ∧
  i.raghu + i.trishul + i.vishal = 7225

/-- Theorem stating that under the given conditions, Raghu's investment is 2500 --/
theorem raghu_investment (i : Investments) (h : InvestmentConditions i) : i.raghu = 2500 := by
  sorry


end raghu_investment_l3524_352465


namespace distance_between_squares_l3524_352463

/-- Given two squares where the smaller square has a perimeter of 8 cm and the larger square has an area of 36 cm², 
    prove that the distance between opposite corners of the two squares is approximately 8.9 cm. -/
theorem distance_between_squares (small_square_perimeter : ℝ) (large_square_area : ℝ) 
  (h1 : small_square_perimeter = 8) 
  (h2 : large_square_area = 36) : 
  ∃ (distance : ℝ), abs (distance - Real.sqrt 80) < 0.1 := by
  sorry

end distance_between_squares_l3524_352463


namespace lcm_of_ratio_and_sum_l3524_352417

theorem lcm_of_ratio_and_sum (a b : ℕ+) : 
  (a : ℚ) / b = 2 / 3 → a + b = 40 → Nat.lcm a b = 24 := by
  sorry

end lcm_of_ratio_and_sum_l3524_352417


namespace multiplication_and_division_l3524_352404

theorem multiplication_and_division : 
  (8 * 4 = 32) ∧ (36 / 9 = 4) := by sorry

end multiplication_and_division_l3524_352404


namespace min_value_of_a_l3524_352435

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioc (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end min_value_of_a_l3524_352435


namespace bicycle_cost_price_l3524_352432

/-- Represents the selling and buying of a bicycle through two transactions -/
def bicycle_sales (initial_cost : ℝ) : Prop :=
  let first_sale := initial_cost * 1.5
  let final_sale := first_sale * 1.25
  final_sale = 225

theorem bicycle_cost_price : ∃ (initial_cost : ℝ), 
  bicycle_sales initial_cost ∧ initial_cost = 120 := by sorry

end bicycle_cost_price_l3524_352432


namespace no_intersection_at_roots_l3524_352474

theorem no_intersection_at_roots : ∀ x : ℝ, 
  (x^2 - 3*x + 2 = 0) → 
  ¬(x^2 - 1 = 3*x - 1) :=
by
  sorry

end no_intersection_at_roots_l3524_352474


namespace expression_simplification_and_evaluation_l3524_352408

theorem expression_simplification_and_evaluation (x : ℝ) (h : x ≠ 1) :
  let expr := ((2 * x + 1) / (x - 1) - 1) / ((x + 2) / ((x - 1)^2))
  expr = x - 1 ∧ (x = 5 → expr = 4) := by
  sorry

end expression_simplification_and_evaluation_l3524_352408


namespace total_problems_solved_l3524_352493

/-- The number of problems Seokjin initially solved -/
def initial_problems : ℕ := 12

/-- The number of additional problems Seokjin solved -/
def additional_problems : ℕ := 7

/-- Theorem: The total number of problems Seokjin solved is 19 -/
theorem total_problems_solved : initial_problems + additional_problems = 19 := by
  sorry

end total_problems_solved_l3524_352493


namespace inscribed_squares_ratio_l3524_352452

/-- Given a right triangle with sides 5, 12, and 13, x is the side length of a square
    inscribed with one vertex at the right angle, and y is the side length of a square
    inscribed with one side on the longest leg (12). -/
def triangle_with_squares (x y : ℝ) : Prop :=
  -- Right triangle condition
  5^2 + 12^2 = 13^2 ∧
  -- Condition for square with side x
  x / 5 = x / 12 ∧
  -- Condition for square with side y
  y + y = 12

/-- The ratio of the side lengths of the two inscribed squares is 10/17. -/
theorem inscribed_squares_ratio :
  ∀ x y : ℝ, triangle_with_squares x y → x / y = 10 / 17 :=
by sorry

end inscribed_squares_ratio_l3524_352452


namespace perpendicular_to_parallel_plane_l3524_352415

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_to_parallel_plane
  (α β : Plane) (l : Line)
  (h1 : perpendicular l α)
  (h2 : parallel α β) :
  perpendicular l β :=
sorry

end perpendicular_to_parallel_plane_l3524_352415


namespace medium_stores_selected_is_ten_l3524_352459

/-- Represents the number of stores to be selected in a stratified sampling -/
def total_sample : ℕ := 30

/-- Represents the total number of stores -/
def total_stores : ℕ := 1500

/-- Represents the ratio of large stores -/
def large_ratio : ℕ := 1

/-- Represents the ratio of medium stores -/
def medium_ratio : ℕ := 5

/-- Represents the ratio of small stores -/
def small_ratio : ℕ := 9

/-- Calculates the number of medium-sized stores to be selected in the stratified sampling -/
def medium_stores_selected : ℕ := 
  (total_sample * medium_ratio) / (large_ratio + medium_ratio + small_ratio)

/-- Theorem stating that the number of medium-sized stores to be selected is 10 -/
theorem medium_stores_selected_is_ten : medium_stores_selected = 10 := by
  sorry


end medium_stores_selected_is_ten_l3524_352459


namespace reciprocal_problem_l3524_352429

theorem reciprocal_problem (x : ℚ) : 7 * x = 3 → 70 * (1 / x) = 490 / 3 := by
  sorry

end reciprocal_problem_l3524_352429


namespace possible_signs_l3524_352455

theorem possible_signs (a b c : ℝ) : 
  a + b + c = 0 → 
  abs a > abs b → 
  abs b > abs c → 
  ∃ (a' b' c' : ℝ), a' + b' + c' = 0 ∧ 
                     abs a' > abs b' ∧ 
                     abs b' > abs c' ∧ 
                     c' > 0 ∧ 
                     a' < 0 :=
sorry

end possible_signs_l3524_352455


namespace collinear_points_condition_l3524_352476

/-- Given non-collinear plane vectors a and b, and points A, B, C such that
    AB = a - 2b and BC = 3a + kb, prove that A, B, and C are collinear iff k = -6 -/
theorem collinear_points_condition (a b : ℝ × ℝ) (k : ℝ) 
  (h_non_collinear : ¬ ∃ (r : ℝ), a = r • b) 
  (A B C : ℝ × ℝ) 
  (h_AB : B - A = a - 2 • b) 
  (h_BC : C - B = 3 • a + k • b) :
  (∃ (t : ℝ), C - A = t • (B - A)) ↔ k = -6 :=
sorry

end collinear_points_condition_l3524_352476


namespace linear_equation_solution_l3524_352495

theorem linear_equation_solution (a : ℝ) :
  (∀ x, ax^2 + 5*x + 14 = 2*x^2 - 2*x + 3*a → x = -8/7) ∧
  (∃ m b, ∀ x, ax^2 + 5*x + 14 = 2*x^2 - 2*x + 3*a ↔ m*x + b = 0) :=
by sorry

end linear_equation_solution_l3524_352495


namespace smallest_x_abs_equation_l3524_352471

theorem smallest_x_abs_equation : ∃ x : ℝ, (∀ y : ℝ, |y + 3| = 15 → x ≤ y) ∧ |x + 3| = 15 := by
  sorry

end smallest_x_abs_equation_l3524_352471


namespace complement_of_complement_l3524_352442

theorem complement_of_complement (α : ℝ) (h : α = 35) :
  90 - (90 - α) = α := by sorry

end complement_of_complement_l3524_352442


namespace max_value_h_two_roots_range_max_positive_integer_a_l3524_352451

noncomputable section

open Real

-- Define the functions
def f (x : ℝ) := exp x
def g (a b : ℝ) (x : ℝ) := (a / 2) * x + b
def h (a b : ℝ) (x : ℝ) := f x * g a b x

-- Statement 1
theorem max_value_h (a b : ℝ) :
  a = -4 → b = 1 - a / 2 →
  ∃ (M : ℝ), M = 2 * exp (1 / 2) ∧ ∀ x ∈ Set.Icc 0 1, h a b x ≤ M :=
sorry

-- Statement 2
theorem two_roots_range (b : ℝ) :
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ f x₁ = g 4 b x₁ ∧ f x₂ = g 4 b x₂) ↔
  b ∈ Set.Ioo (2 - 2 * log 2) 1 :=
sorry

-- Statement 3
theorem max_positive_integer_a :
  ∃ (a : ℕ), a = 14 ∧ ∀ x : ℝ, f x > g a (-15/2) x ∧
  ∀ n : ℕ, n > a → ∃ y : ℝ, f y ≤ g n (-15/2) y :=
sorry

end

end max_value_h_two_roots_range_max_positive_integer_a_l3524_352451


namespace regular_polygon_sides_l3524_352444

theorem regular_polygon_sides : ∀ n : ℕ, 
  n > 2 → (3 * (n * (n - 3) / 2) - n = 21 ↔ n = 6) := by
  sorry

end regular_polygon_sides_l3524_352444


namespace contractor_fine_calculation_l3524_352418

/-- Calculates the fine per day of absence for a contractor --/
def calculate_fine (contract_days : ℕ) (daily_pay : ℚ) (absent_days : ℕ) (total_payment : ℚ) : ℚ :=
  let worked_days := contract_days - absent_days
  let earned_amount := worked_days * daily_pay
  (earned_amount - total_payment) / absent_days

theorem contractor_fine_calculation :
  let contract_days : ℕ := 30
  let daily_pay : ℚ := 25
  let absent_days : ℕ := 6
  let total_payment : ℚ := 555
  calculate_fine contract_days daily_pay absent_days total_payment = 7.5 := by
  sorry

#eval calculate_fine 30 25 6 555

end contractor_fine_calculation_l3524_352418


namespace inequality_proof_l3524_352456

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 9)
  (h5 : a^2 + b^2 + c^2 + d^2 = 21) :
  a * b - c * d ≥ 2 := by
  sorry

end inequality_proof_l3524_352456


namespace closest_fraction_l3524_352478

def medals_won : ℚ := 23 / 150

def fractions : List ℚ := [1/6, 1/7, 1/8, 1/9, 1/10]

theorem closest_fraction :
  (fractions.argmin (λ x => |x - medals_won|)).get! = 1/7 := by sorry

end closest_fraction_l3524_352478


namespace parabola_line_intersection_l3524_352467

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def line (x y : ℝ) : Prop := y = 2*x - 4

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the length of chord AB
def chordLength : ℝ := sorry

-- Define point P on the parabola
def P : ℝ × ℝ := sorry

-- Define the area of triangle ABP
def triangleArea : ℝ := sorry

theorem parabola_line_intersection :
  (parabola A.1 A.2 ∧ line A.1 A.2) ∧
  (parabola B.1 B.2 ∧ line B.1 B.2) ∧
  chordLength = 3 * Real.sqrt 5 ∧
  parabola P.1 P.2 ∧
  triangleArea = 12 →
  (P = (9, 6) ∨ P = (4, -4)) :=
by sorry

end parabola_line_intersection_l3524_352467


namespace equation_solution_l3524_352447

theorem equation_solution : 
  ∃ x : ℝ, (5 + 3.2 * x = 2.4 * x - 15) ∧ (x = -25) := by
  sorry

end equation_solution_l3524_352447


namespace two_numbers_problem_l3524_352440

theorem two_numbers_problem :
  ∃ (x y : ℕ), 
    x = y + 75 ∧
    x * y = (227 * y + 113) + 1000 ∧
    x > y ∧ y > 0 := by
  sorry

end two_numbers_problem_l3524_352440


namespace number_multiplying_a_l3524_352411

theorem number_multiplying_a (a b : ℝ) (h1 : ∃ x, x * a = 8 * b) (h2 : a ≠ 0 ∧ b ≠ 0) (h3 : a / 8 = b / 7) :
  ∃ x, x * a = 8 * b ∧ x = 7 := by
  sorry

end number_multiplying_a_l3524_352411


namespace magic_8_ball_probability_l3524_352439

theorem magic_8_ball_probability : 
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 3  -- number of positive answers
  let p : ℚ := 1/3  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 := by
sorry

end magic_8_ball_probability_l3524_352439


namespace lcm_of_proportional_numbers_l3524_352400

def A : ℕ := 18
def B : ℕ := 24
def C : ℕ := 30

theorem lcm_of_proportional_numbers :
  (A : ℕ) / gcd A B = 3 ∧
  (B : ℕ) / gcd A B = 4 ∧
  (C : ℕ) / gcd A B = 5 ∧
  gcd A (gcd B C) = 6 ∧
  12 ∣ lcm A (lcm B C) →
  lcm A (lcm B C) = 360 := by
sorry

end lcm_of_proportional_numbers_l3524_352400


namespace ellipse_focal_property_l3524_352458

/-- A point on an ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / 16 + y^2 / 4 = 1

/-- The distance from a point to a focus -/
def distance_to_focus (P : PointOnEllipse) (F : ℝ × ℝ) : ℝ := sorry

/-- The foci of the ellipse -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) := sorry

theorem ellipse_focal_property (P : PointOnEllipse) :
  distance_to_focus P (foci.1) = 3 →
  distance_to_focus P (foci.2) = 5 := by sorry

end ellipse_focal_property_l3524_352458


namespace committee_formation_possibilities_l3524_352405

/-- The number of ways to choose k elements from a set of n elements --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of members in the club --/
def club_size : ℕ := 25

/-- The size of the executive committee --/
def committee_size : ℕ := 4

/-- Theorem stating that choosing 4 people from 25 results in 12650 possibilities --/
theorem committee_formation_possibilities :
  choose club_size committee_size = 12650 := by sorry

end committee_formation_possibilities_l3524_352405


namespace fixed_point_of_exponential_function_l3524_352486

/-- The function f(x) = α^(x-2) - 1 always passes through the point (2, 0) for any α > 0 and α ≠ 1 -/
theorem fixed_point_of_exponential_function (α : ℝ) (h1 : α > 0) (h2 : α ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ α^(x - 2) - 1
  f 2 = 0 := by sorry

end fixed_point_of_exponential_function_l3524_352486


namespace triangle_side_length_l3524_352407

theorem triangle_side_length (b c : ℝ) (cosA : ℝ) (h1 : b = 3) (h2 : c = 5) (h3 : cosA = -1/2) :
  ∃ a : ℝ, a^2 = b^2 + c^2 - 2*b*c*cosA ∧ a = 7 := by
  sorry

end triangle_side_length_l3524_352407


namespace ratio_of_P_and_Q_l3524_352413

theorem ratio_of_P_and_Q (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 4 →
    P / (x + 5) + Q / (x^2 - 4*x) = (x^2 + x + 15) / (x^3 + x^2 - 20*x)) →
  (Q : ℚ) / P = -45 / 2 := by
sorry

end ratio_of_P_and_Q_l3524_352413


namespace max_garden_area_l3524_352422

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Calculates the fencing required for three sides of a rectangular garden -/
def fencingRequired (d : GardenDimensions) : ℝ :=
  d.length + 2 * d.width

/-- Theorem: The maximum area of a rectangular garden with 400 feet of fencing
    for three sides is 20000 square feet -/
theorem max_garden_area :
  ∃ (d : GardenDimensions),
    fencingRequired d = 400 ∧
    ∀ (d' : GardenDimensions), fencingRequired d' = 400 →
      gardenArea d' ≤ gardenArea d ∧
      gardenArea d = 20000 := by
  sorry

end max_garden_area_l3524_352422


namespace min_value_ab_l3524_352414

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 5/a + 20/b = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → 5/x + 20/y = 4 → a * b ≤ x * y ∧ a * b = 25 := by
  sorry

end min_value_ab_l3524_352414


namespace sequence_properties_l3524_352469

/-- Proof of properties of sequences A, G, and H -/
theorem sequence_properties
  (x y k : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hk : k > 0)
  (hxy : x ≠ y)
  (hk1 : k ≠ 1)
  (A : ℕ → ℝ)
  (G : ℕ → ℝ)
  (H : ℕ → ℝ)
  (hA1 : A 1 = (k * x + y) / (k + 1))
  (hG1 : G 1 = (x^k * y)^(1 / (k + 1)))
  (hH1 : H 1 = ((k + 1) * x * y) / (k * x + y))
  (hAn : ∀ n ≥ 2, A n = (A (n-1) + H (n-1)) / 2)
  (hGn : ∀ n ≥ 2, G n = (A (n-1) * H (n-1))^(1/2))
  (hHn : ∀ n ≥ 2, H n = 2 / (1 / A (n-1) + 1 / H (n-1))) :
  (∀ n ≥ 1, A (n+1) < A n) ∧
  (∀ n ≥ 1, G (n+1) = G n) ∧
  (∀ n ≥ 1, H n < H (n+1)) :=
by sorry

end sequence_properties_l3524_352469


namespace homer_investment_interest_l3524_352453

/-- Calculates the interest earned on an investment with annual compounding -/
def interest_earned (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Proves that the interest earned on a $2000 investment at 2% for 3 years is $122.416 -/
theorem homer_investment_interest :
  let principal : ℝ := 2000
  let rate : ℝ := 0.02
  let time : ℕ := 3
  abs (interest_earned principal rate time - 122.416) < 0.001 := by
  sorry


end homer_investment_interest_l3524_352453


namespace sin_810_degrees_equals_one_l3524_352496

theorem sin_810_degrees_equals_one : Real.sin (810 * π / 180) = 1 := by
  sorry

end sin_810_degrees_equals_one_l3524_352496


namespace g_neg_two_l3524_352421

def g (x : ℝ) : ℝ := x^3 - x^2 + x

theorem g_neg_two : g (-2) = -14 := by sorry

end g_neg_two_l3524_352421


namespace potential_solution_check_l3524_352403

theorem potential_solution_check (x y : ℕ+) (h : 1 + 2^x.val + 2^(2*x.val+1) = y.val^2) : 
  x = 3 ∨ ∃ z : ℕ+, (1 + 2^z.val + 2^(2*z.val+1) = y.val^2 ∧ z ≠ 3) :=
sorry

end potential_solution_check_l3524_352403


namespace max_sum_scores_max_sum_scores_achievable_l3524_352472

/-- Represents the scoring system for an exam -/
structure ExamScoring where
  m : ℕ             -- number of questions
  n : ℕ             -- number of students
  x : Fin m → ℕ     -- number of students who answered each question incorrectly
  h_m : m ≥ 2
  h_n : n ≥ 2
  h_x : ∀ k, x k ≤ n

/-- The score of a student -/
def student_score (E : ExamScoring) : ℕ → ℕ := sorry

/-- The highest score in the exam -/
def max_score (E : ExamScoring) : ℕ := sorry

/-- The lowest score in the exam -/
def min_score (E : ExamScoring) : ℕ := sorry

/-- Theorem: The maximum possible sum of the highest and lowest scores is m(n-1) -/
theorem max_sum_scores (E : ExamScoring) : 
  max_score E + min_score E ≤ E.m * (E.n - 1) :=
sorry

/-- Theorem: The maximum sum of scores is achievable -/
theorem max_sum_scores_achievable (m n : ℕ) (h_m : m ≥ 2) (h_n : n ≥ 2) : 
  ∃ E : ExamScoring, E.m = m ∧ E.n = n ∧ max_score E + min_score E = m * (n - 1) :=
sorry

end max_sum_scores_max_sum_scores_achievable_l3524_352472


namespace abs_neg_five_equals_five_l3524_352482

theorem abs_neg_five_equals_five :
  abs (-5 : ℤ) = 5 := by sorry

end abs_neg_five_equals_five_l3524_352482


namespace rabbit_weeks_calculation_l3524_352454

/-- The number of weeks Julia has had the rabbit -/
def weeks_with_rabbit : ℕ := 2

/-- The total weekly cost for both animals' food -/
def total_weekly_cost : ℕ := 30

/-- The number of weeks Julia has had the parrot -/
def weeks_with_parrot : ℕ := 3

/-- The total spent on food so far -/
def total_spent : ℕ := 114

/-- The weekly cost of rabbit food -/
def weekly_rabbit_cost : ℕ := 12

theorem rabbit_weeks_calculation :
  weeks_with_rabbit * weekly_rabbit_cost + weeks_with_parrot * total_weekly_cost = total_spent :=
sorry

end rabbit_weeks_calculation_l3524_352454


namespace expansion_equality_constant_term_proof_l3524_352488

/-- The constant term in the expansion of (1/x^2 + 4x^2 + 4)^3 -/
def constantTerm : ℕ := 160

/-- The original expression (1/x^2 + 4x^2 + 4)^3 can be rewritten as (2x + 1/x)^6 -/
theorem expansion_equality (x : ℝ) (hx : x ≠ 0) :
  (1 / x^2 + 4 * x^2 + 4)^3 = (2 * x + 1 / x)^6 := by sorry

/-- The constant term in the expansion of (1/x^2 + 4x^2 + 4)^3 is equal to constantTerm -/
theorem constant_term_proof :
  constantTerm = 160 := by sorry

end expansion_equality_constant_term_proof_l3524_352488


namespace min_groups_for_30_students_max_6_l3524_352441

/-- Given a total number of students and a maximum group size, 
    calculate the smallest number of equal-sized groups. -/
def minGroups (totalStudents : ℕ) (maxGroupSize : ℕ) : ℕ :=
  (totalStudents + maxGroupSize - 1) / maxGroupSize

/-- Theorem: For 30 students and a maximum group size of 6, 
    the smallest number of equal-sized groups is 5. -/
theorem min_groups_for_30_students_max_6 :
  minGroups 30 6 = 5 := by sorry

end min_groups_for_30_students_max_6_l3524_352441


namespace product_evaluation_l3524_352437

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_evaluation_l3524_352437


namespace sum_of_squared_pairs_l3524_352462

theorem sum_of_squared_pairs (p q r : ℝ) : 
  (p^3 - 18*p^2 + 25*p - 6 = 0) →
  (q^3 - 18*q^2 + 25*q - 6 = 0) →
  (r^3 - 18*r^2 + 25*r - 6 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 598 := by
sorry

end sum_of_squared_pairs_l3524_352462


namespace expenditure_representation_l3524_352497

/-- Represents a monetary transaction in yuan -/
structure Transaction where
  amount : Int
  deriving Repr

/-- Defines an income transaction -/
def is_income (t : Transaction) : Prop := t.amount > 0

/-- Defines an expenditure transaction -/
def is_expenditure (t : Transaction) : Prop := t.amount < 0

/-- Theorem stating that an expenditure of 50 yuan should be represented as -50 yuan -/
theorem expenditure_representation :
  ∀ (t : Transaction),
    is_expenditure t → t.amount = 50 → t.amount = -50 := by
  sorry

end expenditure_representation_l3524_352497


namespace consecutive_squares_divisors_l3524_352420

theorem consecutive_squares_divisors :
  ∃ (n : ℕ), 
    (∃ (a : ℕ), a > 1 ∧ a * a ∣ n) ∧
    (∃ (b : ℕ), b > 1 ∧ b * b ∣ (n + 1)) ∧
    (∃ (c : ℕ), c > 1 ∧ c * c ∣ (n + 2)) ∧
    (∃ (d : ℕ), d > 1 ∧ d * d ∣ (n + 3)) :=
by sorry

end consecutive_squares_divisors_l3524_352420


namespace angle_triple_complement_l3524_352427

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end angle_triple_complement_l3524_352427


namespace min_value_theorem_l3524_352446

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 1) + 9 / y = 1) : 
  4 * x + y ≥ 21 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1 / (x₀ + 1) + 9 / y₀ = 1 ∧ 4 * x₀ + y₀ = 21 := by
  sorry

end min_value_theorem_l3524_352446


namespace not_prime_sum_product_l3524_352480

theorem not_prime_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a) 
  (h_eq : a * c + b * d = (b + d - a + c) * (b + d + a - c)) : 
  ¬ Nat.Prime (a * b + c * d) := by
sorry

end not_prime_sum_product_l3524_352480


namespace problem_statement_l3524_352494

theorem problem_statement (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a^b = c^d) (h2 : a / (2 * c) = b / d) (h3 : a / (2 * c) = 2) :
  1 / c = 16 := by
  sorry

end problem_statement_l3524_352494


namespace bracelet_price_is_4_l3524_352419

/-- The price of a bracelet in dollars -/
def bracelet_price : ℝ := sorry

/-- The price of a keychain in dollars -/
def keychain_price : ℝ := 5

/-- The price of a coloring book in dollars -/
def coloring_book_price : ℝ := 3

/-- The total cost of the purchases -/
def total_cost : ℝ := 20

theorem bracelet_price_is_4 :
  2 * bracelet_price + keychain_price + bracelet_price + coloring_book_price = total_cost →
  bracelet_price = 4 := by
  sorry

end bracelet_price_is_4_l3524_352419


namespace select_defective_products_l3524_352460

def total_products : ℕ := 100
def defective_products : ℕ := 6
def products_to_select : ℕ := 3

theorem select_defective_products :
  Nat.choose total_products products_to_select -
  Nat.choose (total_products - defective_products) products_to_select =
  Nat.choose total_products products_to_select -
  Nat.choose 94 products_to_select :=
by sorry

end select_defective_products_l3524_352460


namespace cindys_calculation_l3524_352423

theorem cindys_calculation (x : ℝ) : 
  (x - 4) / 7 = 43 → (x - 7) / 4 = 74.5 := by
  sorry

end cindys_calculation_l3524_352423


namespace min_value_of_expression_l3524_352464

theorem min_value_of_expression (a b : ℕ+) (h : a > b) :
  let E := |(a + 2*b : ℝ) / (a - b : ℝ) + (a - b : ℝ) / (a + 2*b : ℝ)|
  ∀ x : ℝ, E ≥ 2 :=
by sorry

end min_value_of_expression_l3524_352464


namespace little_john_money_distribution_l3524_352424

theorem little_john_money_distribution 
  (initial_amount : ℚ)
  (spent_on_sweets : ℚ)
  (num_friends : ℕ)
  (remaining_amount : ℚ)
  (h1 : initial_amount = 10.10)
  (h2 : spent_on_sweets = 3.25)
  (h3 : num_friends = 2)
  (h4 : remaining_amount = 2.45) :
  (initial_amount - spent_on_sweets - remaining_amount) / num_friends = 2.20 :=
by sorry

end little_john_money_distribution_l3524_352424


namespace gcd_repeated_digits_l3524_352489

def is_repeated_digit (n : ℕ) : Prop :=
  ∃ (m : ℕ), 100 ≤ m ∧ m < 1000 ∧ n = 1001 * m

theorem gcd_repeated_digits :
  ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℕ), is_repeated_digit n → d ∣ n) ∧
  (∀ (k : ℕ), k > 0 → (∀ (n : ℕ), is_repeated_digit n → k ∣ n) → k ∣ d) :=
sorry

end gcd_repeated_digits_l3524_352489


namespace polynomial_symmetry_l3524_352416

/-- Given a polynomial function g(x) = ax^5 + bx^3 + cx - 3 where g(-5) = 3, prove that g(5) = -9 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let g : ℝ → ℝ := λ x => a * x^5 + b * x^3 + c * x - 3
  g (-5) = 3 → g 5 = -9 := by
  sorry

end polynomial_symmetry_l3524_352416


namespace unique_two_digit_number_l3524_352436

theorem unique_two_digit_number : 
  ∃! (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (Nat.factorial a - a * b) % 10 = 2 := by
  sorry

end unique_two_digit_number_l3524_352436


namespace all_reachable_l3524_352402

def step (x : ℚ) : Set ℚ := {x + 1, -1 / x}

def reachable : Set ℚ → Prop :=
  λ S => ∀ y ∈ S, ∃ n : ℕ, ∃ f : ℕ → ℚ,
    f 0 = 1 ∧ (∀ i < n, f (i + 1) ∈ step (f i)) ∧ f n = y

theorem all_reachable : reachable {-2, 1/2, 5/3, 7} := by
  sorry

end all_reachable_l3524_352402


namespace last_two_digits_of_sum_of_factorials_15_l3524_352484

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_sum_of_factorials_15 :
  last_two_digits (sum_of_factorials 15) = 13 := by
  sorry

end last_two_digits_of_sum_of_factorials_15_l3524_352484


namespace arithmetic_sequence_max_sum_l3524_352434

/-- Given an arithmetic sequence with certain properties, prove its maximum sum -/
theorem arithmetic_sequence_max_sum (k : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ) :
  k ≥ 2 →
  S (k - 1) = 8 →
  S k = 0 →
  S (k + 1) = -10 →
  (∀ n, S (n + 1) - S n = a (n + 1)) →
  (∃ d : ℤ, ∀ n, a (n + 1) - a n = d) →
  (∃ n : ℕ, ∀ m : ℕ, S m ≤ S n) →
  (∃ n : ℕ, S n = 20) :=
by sorry

end arithmetic_sequence_max_sum_l3524_352434


namespace max_AB_is_five_l3524_352438

/-- Represents a convex quadrilateral ABCD inscribed in a circle -/
structure CyclicQuadrilateral where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  AB_shortest : AB ≤ BC ∧ AB ≤ CD ∧ AB ≤ DA
  distinct_sides : AB ≠ BC ∧ AB ≠ CD ∧ AB ≠ DA ∧ BC ≠ CD ∧ BC ≠ DA ∧ CD ≠ DA
  max_side_10 : AB ≤ 10 ∧ BC ≤ 10 ∧ CD ≤ 10 ∧ DA ≤ 10
  area_ratio_int : ∃ k : ℕ, BC * CD = k * DA * AB

/-- The maximum possible value of AB in a CyclicQuadrilateral is 5 -/
theorem max_AB_is_five (q : CyclicQuadrilateral) : q.AB ≤ 5 :=
  sorry

end max_AB_is_five_l3524_352438


namespace triangle_properties_l3524_352485

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : a = 5) 
  (h3 : c = 6) 
  (h4 : Real.sin B = 3/5) :
  b = Real.sqrt 13 ∧ 
  Real.sin A = (3 * Real.sqrt 13) / 13 ∧ 
  Real.sin (2 * A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end triangle_properties_l3524_352485


namespace ellipse_parabola_intersection_range_l3524_352499

/-- The range of 'a' for which the ellipse x^2 + 4(y-a)^2 = 4 intersects with the parabola x^2 = 2y -/
theorem ellipse_parabola_intersection_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x^2 + 4*(y-a)^2 = 4 ∧ x^2 = 2*y) →
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end ellipse_parabola_intersection_range_l3524_352499


namespace insect_leg_paradox_l3524_352431

theorem insect_leg_paradox (total_legs : ℕ) (six_leg_insects : ℕ) (eight_leg_insects : ℕ) 
  (h1 : total_legs = 190)
  (h2 : 6 * six_leg_insects = 78)
  (h3 : 8 * eight_leg_insects = 24) :
  ¬∃ (ten_leg_insects : ℕ), 
    6 * six_leg_insects + 8 * eight_leg_insects + 10 * ten_leg_insects = total_legs :=
by sorry

end insect_leg_paradox_l3524_352431


namespace function_properties_l3524_352425

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem function_properties (a : ℝ) (h_min : ∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f a x ≤ f a y) 
  (h_min_value : ∃ (x : ℝ), x ∈ interval ∧ f a x = -37) :
  a = 3 ∧ ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), x ∈ interval → f a x ≤ m := by
  sorry

end function_properties_l3524_352425


namespace milk_exchange_theorem_l3524_352430

/-- Represents the number of liters of milk obtainable from a given number of empty bottles -/
def milk_obtained (empty_bottles : ℕ) : ℕ :=
  let full_bottles := empty_bottles / 4
  let remaining_empty := empty_bottles % 4
  if full_bottles = 0 then
    0
  else
    full_bottles + milk_obtained (full_bottles + remaining_empty)

/-- Theorem stating that 43 empty bottles can be exchanged for 14 liters of milk -/
theorem milk_exchange_theorem :
  milk_obtained 43 = 14 := by
  sorry

end milk_exchange_theorem_l3524_352430


namespace derangement_even_index_odd_l3524_352490

/-- Definition of derangement numbers -/
def D : ℕ → ℕ
  | 0 => 0  -- D₀ is defined as 0 for completeness
  | 1 => 0
  | 2 => 1
  | 3 => 2
  | 4 => 9
  | (n + 5) => (n + 4) * (D (n + 4) + D (n + 3))

/-- Theorem: D₂ₙ is odd for all positive natural numbers n -/
theorem derangement_even_index_odd (n : ℕ+) : Odd (D (2 * n)) := by
  sorry

end derangement_even_index_odd_l3524_352490


namespace tan_600_degrees_l3524_352412

theorem tan_600_degrees : Real.tan (600 * π / 180) = -Real.sqrt 3 := by
  sorry

end tan_600_degrees_l3524_352412


namespace triangle_equal_area_l3524_352475

/-- Given two triangles FGH and IJK with the specified properties, prove that JK = 10 -/
theorem triangle_equal_area (FG FH IJ IK : ℝ) (angle_GFH angle_IJK : ℝ) :
  FG = 5 →
  FH = 4 →
  angle_GFH = 30 * π / 180 →
  IJ = 2 →
  IK = 6 →
  angle_IJK = 30 * π / 180 →
  angle_GFH = angle_IJK →
  (1/2 * FG * FH * Real.sin angle_GFH) = (1/2 * IJ * 10 * Real.sin angle_IJK) →
  ∃ (JK : ℝ), JK = 10 := by
  sorry

end triangle_equal_area_l3524_352475


namespace polynomial_factorization_l3524_352448

theorem polynomial_factorization (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x-1)^2 := by
  sorry

end polynomial_factorization_l3524_352448


namespace quadratic_root_range_l3524_352483

theorem quadratic_root_range (m : ℝ) : 
  (∃ x y : ℝ, x < 1 ∧ y > 1 ∧ 
   x^2 + (m-1)*x + m^2 - 2 = 0 ∧
   y^2 + (m-1)*y + m^2 - 2 = 0) →
  -2 < m ∧ m < 1 := by
sorry

end quadratic_root_range_l3524_352483


namespace papers_per_envelope_l3524_352445

theorem papers_per_envelope 
  (total_papers : ℕ) 
  (num_envelopes : ℕ) 
  (h1 : total_papers = 120) 
  (h2 : num_envelopes = 12) : 
  total_papers / num_envelopes = 10 := by
sorry

end papers_per_envelope_l3524_352445


namespace necessary_but_not_sufficient_condition_l3524_352477

theorem necessary_but_not_sufficient_condition 
  (A B C : Set α) 
  (h_nonempty_A : A.Nonempty) 
  (h_nonempty_B : B.Nonempty) 
  (h_nonempty_C : C.Nonempty)
  (h_union : A ∪ B = C) 
  (h_not_subset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ x, x ∈ C ∧ x ∉ A) := by
  sorry

end necessary_but_not_sufficient_condition_l3524_352477


namespace mary_next_birthday_age_l3524_352479

theorem mary_next_birthday_age (m s d t : ℝ) : 
  m = 1.25 * s →
  s = 0.7 * d →
  t = 2 * s →
  m + s + d + t = 38 →
  ⌊m⌋ + 1 = 9 :=
sorry

end mary_next_birthday_age_l3524_352479


namespace volleyball_team_selection_l3524_352461

/-- The number of ways to choose 7 starters from a volleyball team -/
def volleyball_starters_count : ℕ := 2376

/-- The total number of players in the team -/
def total_players : ℕ := 15

/-- The number of triplets in the team -/
def triplets_count : ℕ := 3

/-- The number of starters to be chosen -/
def starters_count : ℕ := 7

/-- The number of triplets that must be in the starting lineup -/
def required_triplets : ℕ := 2

theorem volleyball_team_selection :
  volleyball_starters_count = 
    (Nat.choose triplets_count required_triplets) * 
    (Nat.choose (total_players - triplets_count) (starters_count - required_triplets)) := by
  sorry

end volleyball_team_selection_l3524_352461


namespace sqrt_12_plus_3_minus_pi_pow_0_plus_abs_1_minus_sqrt_3_equals_3_sqrt_3_l3524_352433

theorem sqrt_12_plus_3_minus_pi_pow_0_plus_abs_1_minus_sqrt_3_equals_3_sqrt_3 :
  Real.sqrt 12 + (3 - Real.pi) ^ (0 : ℕ) + |1 - Real.sqrt 3| = 3 * Real.sqrt 3 := by
  sorry

end sqrt_12_plus_3_minus_pi_pow_0_plus_abs_1_minus_sqrt_3_equals_3_sqrt_3_l3524_352433


namespace monkey_bird_problem_l3524_352481

theorem monkey_bird_problem (initial_birds : ℕ) (eaten_birds : ℕ) (monkey_percentage : ℚ) : 
  initial_birds = 6 →
  eaten_birds = 2 →
  monkey_percentage = 6/10 →
  ∃ (initial_monkeys : ℕ), 
    initial_monkeys = 6 ∧
    (initial_monkeys : ℚ) / ((initial_monkeys : ℚ) + (initial_birds - eaten_birds : ℚ)) = monkey_percentage :=
by sorry

end monkey_bird_problem_l3524_352481


namespace average_price_is_16_l3524_352409

/-- The average price of books bought by Rahim -/
def average_price_per_book (books_shop1 books_shop2 : ℕ) (price_shop1 price_shop2 : ℕ) : ℚ :=
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2)

/-- Theorem stating that the average price per book is 16 given the problem conditions -/
theorem average_price_is_16 :
  average_price_per_book 55 60 1500 340 = 16 := by
  sorry

end average_price_is_16_l3524_352409


namespace completing_square_equiv_l3524_352426

/-- Proves that y = -x^2 + 2x + 3 can be rewritten as y = -(x-1)^2 + 4 -/
theorem completing_square_equiv :
  ∀ x y : ℝ, y = -x^2 + 2*x + 3 ↔ y = -(x-1)^2 + 4 :=
by sorry

end completing_square_equiv_l3524_352426


namespace impossible_table_l3524_352466

/-- Represents a 6x6 table of integers -/
def Table := Fin 6 → Fin 6 → ℤ

/-- Checks if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → t i j ≠ t k l

/-- Checks if the sum of a 1x5 rectangle is valid (2022 or 2023) -/
def valid_sum (s : ℤ) : Prop := s = 2022 ∨ s = 2023

/-- Checks if all 1x5 rectangles (horizontal and vertical) have valid sums -/
def all_rectangles_valid (t : Table) : Prop :=
  (∀ i j, valid_sum (t i j + t i (j+1) + t i (j+2) + t i (j+3) + t i (j+4))) ∧
  (∀ i j, valid_sum (t i j + t (i+1) j + t (i+2) j + t (i+3) j + t (i+4) j))

/-- The main theorem: it's impossible to fill the table satisfying all conditions -/
theorem impossible_table : ¬∃ (t : Table), all_distinct t ∧ all_rectangles_valid t := by
  sorry

end impossible_table_l3524_352466


namespace removed_triangles_area_l3524_352473

-- Define the square side length
def square_side : ℝ := 16

-- Define the ratio of r to s
def r_to_s_ratio : ℝ := 3

-- Theorem statement
theorem removed_triangles_area (r s : ℝ) : 
  r / s = r_to_s_ratio →
  (r + s)^2 + (r - s)^2 = square_side^2 →
  4 * (1/2 * r * s) = 76.8 := by
  sorry

end removed_triangles_area_l3524_352473


namespace train_crossing_time_l3524_352457

/-- Given a train and two platforms, calculate the time to cross the first platform -/
theorem train_crossing_time (Lt Lp1 Lp2 Tp2 : ℝ) (h1 : Lt = 30)
    (h2 : Lp1 = 180) (h3 : Lp2 = 250) (h4 : Tp2 = 20) :
  (Lt + Lp1) / ((Lt + Lp2) / Tp2) = 15 := by
  sorry

end train_crossing_time_l3524_352457


namespace more_books_than_maddie_l3524_352410

/-- Proves that Amy and Luisa have 9 more books than Maddie -/
theorem more_books_than_maddie 
  (maddie_books : ℕ) 
  (luisa_books : ℕ) 
  (amy_books : ℕ)
  (h1 : maddie_books = 15)
  (h2 : luisa_books = 18)
  (h3 : amy_books = 6) :
  luisa_books + amy_books - maddie_books = 9 := by
sorry

end more_books_than_maddie_l3524_352410


namespace train_passing_platform_l3524_352470

/-- Time taken for a train to pass a platform -/
theorem train_passing_platform (train_length platform_length : ℝ) (train_speed : ℝ) : 
  train_length = 360 →
  platform_length = 140 →
  train_speed = 45 →
  (train_length + platform_length) / (train_speed * 1000 / 3600) = 40 :=
by
  sorry

end train_passing_platform_l3524_352470
