import Mathlib

namespace students_just_passed_l713_71359

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 25 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_no_fail : first_div_percent + second_div_percent ≤ 1) :
  total - (total * (first_div_percent + second_div_percent)).floor = 63 := by
  sorry

end students_just_passed_l713_71359


namespace coinciding_rest_days_count_l713_71327

/-- Craig's work cycle in days -/
def craig_cycle : ℕ := 6

/-- Dana's work cycle in days -/
def dana_cycle : ℕ := 6

/-- Number of days Craig works in his cycle -/
def craig_work_days : ℕ := 4

/-- Number of days Dana works in her cycle -/
def dana_work_days : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 1000

/-- The number of days both Craig and Dana have rest-days on the same day -/
def coinciding_rest_days : ℕ := total_days / craig_cycle

theorem coinciding_rest_days_count :
  coinciding_rest_days = 166 := by sorry

end coinciding_rest_days_count_l713_71327


namespace intersection_when_a_2_b_subset_a_range_l713_71363

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Theorem 1: Intersection of A and B when a = 2
theorem intersection_when_a_2 :
  A 2 ∩ B 2 = {x | 4 < x ∧ x < 5} :=
sorry

-- Theorem 2: Range of a for which B is a subset of A
theorem b_subset_a_range :
  ∀ a : ℝ, B a ⊆ A a ↔ a = -1 ∨ (1 ≤ a ∧ a ≤ 3) :=
sorry

end intersection_when_a_2_b_subset_a_range_l713_71363


namespace pencil_grouping_l713_71319

theorem pencil_grouping (total_pencils : ℕ) (num_groups : ℕ) (pencils_per_group : ℕ) :
  total_pencils = 25 →
  num_groups = 5 →
  total_pencils = num_groups * pencils_per_group →
  pencils_per_group = 5 :=
by sorry

end pencil_grouping_l713_71319


namespace partnership_profit_calculation_l713_71334

/-- Represents the profit distribution in a partnership business -/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  c_profit : ℕ

/-- Calculates the total profit given a profit distribution -/
def total_profit (pd : ProfitDistribution) : ℕ :=
  let total_investment := pd.a_investment + pd.b_investment + pd.c_investment
  let c_ratio := pd.c_investment / (total_investment / 20)
  (pd.c_profit * 20) / c_ratio

/-- Theorem stating that given the specific investments and c's profit, 
    the total profit is $60,000 -/
theorem partnership_profit_calculation (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 45000)
  (h2 : pd.b_investment = 63000)
  (h3 : pd.c_investment = 72000)
  (h4 : pd.c_profit = 24000) :
  total_profit pd = 60000 := by
  sorry

end partnership_profit_calculation_l713_71334


namespace maria_roses_l713_71300

/-- The number of roses Maria bought -/
def roses : ℕ := sorry

/-- The price of each flower -/
def flower_price : ℕ := 6

/-- The number of daisies Maria bought -/
def daisies : ℕ := 3

/-- The total amount Maria spent -/
def total_spent : ℕ := 60

theorem maria_roses :
  roses * flower_price + daisies * flower_price = total_spent →
  roses = 7 := by sorry

end maria_roses_l713_71300


namespace miles_to_tie_l713_71332

/-- The number of miles Billy runs each day from Sunday to Friday -/
def billy_daily_miles : ℚ := 1

/-- The number of miles Tiffany runs each day from Sunday to Tuesday -/
def tiffany_daily_miles_sun_to_tue : ℚ := 2

/-- The number of miles Tiffany runs each day from Wednesday to Friday -/
def tiffany_daily_miles_wed_to_fri : ℚ := 1/3

/-- The number of days Billy and Tiffany run from Sunday to Tuesday -/
def days_sun_to_tue : ℕ := 3

/-- The number of days Billy and Tiffany run from Wednesday to Friday -/
def days_wed_to_fri : ℕ := 3

theorem miles_to_tie : 
  (tiffany_daily_miles_sun_to_tue * days_sun_to_tue + 
   tiffany_daily_miles_wed_to_fri * days_wed_to_fri) - 
  (billy_daily_miles * (days_sun_to_tue + days_wed_to_fri)) = 1 := by
  sorry

end miles_to_tie_l713_71332


namespace carlas_marbles_l713_71320

theorem carlas_marbles (x : ℕ) : 
  x + 134 - 68 + 56 = 244 → x = 122 := by
  sorry

end carlas_marbles_l713_71320


namespace circle_condition_l713_71311

/-- The equation x^2 + y^2 + ax - ay + 2 = 0 represents a circle if and only if a > 2 or a < -2 -/
theorem circle_condition (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + a*x - a*y + 2 = 0 ∧ 
   ∀ x' y' : ℝ, x'^2 + y'^2 + a*x' - a*y' + 2 = 0 → (x' - x)^2 + (y' - y)^2 = ((x' - x)^2 + (y' - y)^2)) 
  ↔ (a > 2 ∨ a < -2) :=
sorry

end circle_condition_l713_71311


namespace cubic_curve_rational_points_l713_71390

-- Define a cubic curve with rational coefficients
def CubicCurve (f : ℚ → ℚ → ℚ) : Prop :=
  ∀ x y, ∃ a b c d e g h k l : ℚ, 
    f x y = a*x^3 + b*x^2*y + c*x*y^2 + d*y^3 + e*x^2 + g*x*y + h*y^2 + k*x + l*y

-- Define a point on the curve
def PointOnCurve (f : ℚ → ℚ → ℚ) (x y : ℚ) : Prop :=
  f x y = 0

-- Theorem statement
theorem cubic_curve_rational_points 
  (f : ℚ → ℚ → ℚ) 
  (hf : CubicCurve f) 
  (x₀ y₀ : ℚ) 
  (h₀ : PointOnCurve f x₀ y₀) :
  ∃ x' y' : ℚ, x' ≠ x₀ ∧ y' ≠ y₀ ∧ PointOnCurve f x' y' :=
sorry

end cubic_curve_rational_points_l713_71390


namespace specific_figure_triangles_l713_71398

/-- Represents a triangular figure composed of smaller equilateral triangles. -/
structure TriangularFigure where
  row1 : Nat -- Number of triangles in the first row
  row2 : Nat -- Number of triangles in the second row
  row3 : Nat -- Number of triangles in the third row
  has_outer_triangle : Bool -- Whether there's a large triangle spanning all smaller triangles
  has_diagonal_cut : Bool -- Whether there's a diagonal cut over the bottom two rows

/-- Calculates the total number of triangles in the figure. -/
def total_triangles (figure : TriangularFigure) : Nat :=
  sorry

/-- Theorem stating that for the specific triangular figure described,
    the total number of triangles is 11. -/
theorem specific_figure_triangles :
  let figure : TriangularFigure := {
    row1 := 3,
    row2 := 2,
    row3 := 1,
    has_outer_triangle := true,
    has_diagonal_cut := true
  }
  total_triangles figure = 11 := by sorry

end specific_figure_triangles_l713_71398


namespace xy_value_l713_71374

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^2 + y^2 = 4) (h2 : x^4 + y^4 = 7) : 
  x * y = 3 * Real.sqrt 2 / 2 := by
  sorry

end xy_value_l713_71374


namespace min_value_theorem_l713_71364

def f (a x : ℝ) : ℝ := x^2 + (a+8)*x + a^2 + a - 12

theorem min_value_theorem (a : ℝ) (h1 : a < 0) 
  (h2 : f a (a^2 - 4) = f a (2*a - 8)) :
  (∀ n : ℕ+, (f a n - 4*a) / (n + 1) ≥ 37/4) ∧ 
  (∃ n : ℕ+, (f a n - 4*a) / (n + 1) = 37/4) := by
  sorry

end min_value_theorem_l713_71364


namespace zero_in_M_l713_71308

def M : Set ℝ := {x | x ≤ 2}

theorem zero_in_M : (0 : ℝ) ∈ M := by
  sorry

end zero_in_M_l713_71308


namespace book_pages_count_l713_71376

def days_in_week : ℕ := 7
def first_period : ℕ := 4
def second_period : ℕ := 2
def last_day : ℕ := 1
def pages_per_day_first_period : ℕ := 42
def pages_per_day_second_period : ℕ := 50
def pages_last_day : ℕ := 30

theorem book_pages_count :
  first_period * pages_per_day_first_period +
  second_period * pages_per_day_second_period +
  pages_last_day = 298 :=
by sorry

end book_pages_count_l713_71376


namespace figure_area_theorem_l713_71342

theorem figure_area_theorem (x : ℝ) :
  let small_square_area := (3 * x)^2
  let large_square_area := (7 * x)^2
  let triangle_area := (1/2) * (3 * x) * (7 * x)
  small_square_area + large_square_area + triangle_area = 2200 →
  x = Real.sqrt (4400 / 137) :=
by sorry

end figure_area_theorem_l713_71342


namespace unique_bounded_sequence_l713_71353

def sequence_relation (a : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 3 → a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

theorem unique_bounded_sequence (a : ℕ → ℕ) :
  sequence_relation a →
  (∃ M : ℕ, ∀ n : ℕ, a n ≤ M) →
  (∀ n : ℕ, a n = 2) :=
sorry

end unique_bounded_sequence_l713_71353


namespace find_C_l713_71396

theorem find_C (A B C : ℕ) : A = 680 → A = B + 157 → B = C + 185 → C = 338 := by
  sorry

end find_C_l713_71396


namespace petya_win_probability_l713_71343

/-- The "Pile of Stones" game --/
structure PileOfStones :=
  (initial_stones : ℕ)
  (min_take : ℕ)
  (max_take : ℕ)

/-- A player in the "Pile of Stones" game --/
inductive Player
| Petya
| Computer

/-- The strategy of a player --/
def Strategy := ℕ → ℕ

/-- The optimal strategy for the second player --/
def optimal_strategy : Strategy := sorry

/-- A random strategy that always takes between min_take and max_take stones --/
def random_strategy (game : PileOfStones) : Strategy := sorry

/-- The probability of winning for a player given their strategy and the opponent's strategy --/
def win_probability (game : PileOfStones) (player : Player) (player_strategy : Strategy) (opponent_strategy : Strategy) : ℚ := sorry

/-- The main theorem: Petya's probability of winning is 1/256 --/
theorem petya_win_probability :
  let game : PileOfStones := ⟨16, 1, 4⟩
  win_probability game Player.Petya (random_strategy game) optimal_strategy = 1 / 256 := by sorry

end petya_win_probability_l713_71343


namespace mary_berry_cost_l713_71392

/-- The amount Mary paid for berries, given her total payment, peach cost, and change received. -/
theorem mary_berry_cost (total_paid change peach_cost : ℚ) 
  (h1 : total_paid = 20)
  (h2 : change = 598/100)
  (h3 : peach_cost = 683/100) :
  total_paid - change - peach_cost = 719/100 := by
  sorry


end mary_berry_cost_l713_71392


namespace k_domain_l713_71335

-- Define the function h
def h : ℝ → ℝ := sorry

-- Define the domain of h
def h_domain : Set ℝ := Set.Icc (-8) 4

-- Define the function k in terms of h
def k (x : ℝ) : ℝ := h (3 * x + 1)

-- State the theorem
theorem k_domain :
  {x : ℝ | k x ∈ Set.range h} = Set.Icc (-3) 1 := by sorry

end k_domain_l713_71335


namespace gcd_three_digit_palindromes_l713_71350

def three_digit_palindrome (a b : ℕ) : ℕ := 101 * a + 10 * b

theorem gcd_three_digit_palindromes :
  ∃ (d : ℕ), d > 0 ∧
  (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
    d ∣ three_digit_palindrome a b) ∧
  (∀ (d' : ℕ), d' > d →
    ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
      ¬(d' ∣ three_digit_palindrome a b)) ∧
  d = 1 :=
by sorry

end gcd_three_digit_palindromes_l713_71350


namespace smallest_cut_for_non_triangle_l713_71316

theorem smallest_cut_for_non_triangle (a b c : ℝ) (ha : a = 10) (hb : b = 24) (hc : c = 26) :
  let f := fun x => (a - x) + (b - x) ≤ (c - x)
  ∃ x₀ : ℝ, x₀ = 8 ∧ (∀ x, 0 ≤ x ∧ x < a → (f x → x ≥ x₀) ∧ (x < x₀ → ¬f x)) := by
  sorry

end smallest_cut_for_non_triangle_l713_71316


namespace fourier_expansion_arccos_plus_one_l713_71331

-- Define the Chebyshev polynomials
noncomputable def T (n : ℕ) (x : ℝ) : ℝ := Real.cos (n * Real.arccos x)

-- Define the function to be expanded
noncomputable def f (x : ℝ) : ℝ := Real.arccos x + 1

-- Define the interval
def I : Set ℝ := Set.Ioo (-1) 1

-- Define the Fourier coefficient
noncomputable def a (n : ℕ) : ℝ :=
  if n = 0
  then (Real.pi + 2) / 2
  else 2 / Real.pi * ((-1)^n - 1) / (n^2 : ℝ)

-- State the theorem
theorem fourier_expansion_arccos_plus_one :
  ∀ x ∈ I, f x = (Real.pi + 2) / 2 + ∑' n, a n * T n x :=
sorry

end fourier_expansion_arccos_plus_one_l713_71331


namespace art_students_count_l713_71355

theorem art_students_count (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 40)
  (h3 : both = 10)
  (h4 : neither = 450) :
  ∃ art : ℕ, art = total - (music - both) - both - neither :=
by
  sorry

end art_students_count_l713_71355


namespace n_squared_divisible_by_144_l713_71382

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∀ d : ℕ+, d ∣ n → d ≤ 12) :
  144 ∣ n^2 := by
  sorry

end n_squared_divisible_by_144_l713_71382


namespace square_of_binomial_l713_71312

theorem square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 4*x^2 + 12*x + a = (2*x + b)^2) → a = 9 :=
by sorry

end square_of_binomial_l713_71312


namespace solve_boat_speed_l713_71325

def boat_speed_problem (stream_speed : ℝ) (distance : ℝ) (total_time : ℝ) : Prop :=
  ∃ (boat_speed : ℝ),
    boat_speed > stream_speed ∧
    (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = total_time ∧
    boat_speed = 9

theorem solve_boat_speed : boat_speed_problem 1.5 105 24 := by
  sorry

end solve_boat_speed_l713_71325


namespace min_value_theorem_l713_71301

theorem min_value_theorem (a : ℝ) (h : a > 3) :
  a + 1 / (a - 3) ≥ 5 ∧ (a + 1 / (a - 3) = 5 ↔ a = 4) := by
  sorry

end min_value_theorem_l713_71301


namespace paco_cookie_difference_l713_71348

/-- The number of more salty cookies than sweet cookies eaten by Paco -/
def cookies_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) : ℕ :=
  eaten_salty - eaten_sweet

/-- Theorem stating that Paco ate 13 more salty cookies than sweet cookies -/
theorem paco_cookie_difference :
  cookies_difference 40 25 15 28 = 13 := by
  sorry

end paco_cookie_difference_l713_71348


namespace unicorn_rope_problem_l713_71351

theorem unicorn_rope_problem (rope_length : ℝ) (tower_radius : ℝ) (rope_height : ℝ) (rope_distance : ℝ) 
  (h1 : rope_length = 30)
  (h2 : tower_radius = 10)
  (h3 : rope_height = 6)
  (h4 : rope_distance = 6) :
  ∃ (p q r : ℕ), 
    (p > 0 ∧ q > 0 ∧ r > 0) ∧ 
    Nat.Prime r ∧
    (p - Real.sqrt q) / r = 
      (rope_length * Real.sqrt ((tower_radius + rope_distance)^2 + rope_height^2)) / 
      (tower_radius + Real.sqrt ((tower_radius + rope_distance)^2 + rope_height^2)) ∧
    p + q + r = 1290 := by
  sorry

end unicorn_rope_problem_l713_71351


namespace circle_polar_equation_l713_71375

/-- A circle in the polar coordinate system with center at (1,0) and passing through the pole -/
structure PolarCircle where
  /-- The radius of the circle as a function of the angle θ -/
  ρ : ℝ → ℝ

/-- The polar coordinate equation of the circle -/
def polar_equation (c : PolarCircle) : Prop :=
  ∀ θ : ℝ, c.ρ θ = 2 * Real.cos θ

/-- Theorem stating that the polar coordinate equation of a circle with center at (1,0) 
    and passing through the pole is ρ = 2cos θ -/
theorem circle_polar_equation :
  ∀ c : PolarCircle, polar_equation c :=
sorry

end circle_polar_equation_l713_71375


namespace cube_sum_and_reciprocal_l713_71357

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cube_sum_and_reciprocal_l713_71357


namespace complement_of_B_in_U_l713_71329

open Set

theorem complement_of_B_in_U (U A B : Set ℕ) : 
  U = A ∪ B → 
  A = {1, 2, 3} → 
  A ∩ B = {1} → 
  U \ B = {2, 3} := by sorry

end complement_of_B_in_U_l713_71329


namespace fraction_multiplication_l713_71352

theorem fraction_multiplication : (1 : ℚ) / 3 * (1 : ℚ) / 2 * (3 : ℚ) / 4 * (5 : ℚ) / 6 = (5 : ℚ) / 48 := by
  sorry

end fraction_multiplication_l713_71352


namespace total_paid_is_230_l713_71339

/-- The cost of an item before tax -/
def cost : ℝ := 200

/-- The tax rate as a decimal -/
def tax_rate : ℝ := 0.15

/-- The total amount paid after tax -/
def total_paid : ℝ := cost + (cost * tax_rate)

/-- Theorem stating that the total amount paid after tax is $230 -/
theorem total_paid_is_230 : total_paid = 230 := by
  sorry

end total_paid_is_230_l713_71339


namespace max_groups_is_two_l713_71345

/-- Represents the number of boys in the class -/
def num_boys : ℕ := 20

/-- Represents the number of girls in the class -/
def num_girls : ℕ := 24

/-- Represents the total number of students in the class -/
def total_students : ℕ := num_boys + num_girls

/-- Represents the number of age groups -/
def num_age_groups : ℕ := 3

/-- Represents the number of skill levels -/
def num_skill_levels : ℕ := 3

/-- Represents the maximum number of groups that can be formed -/
def max_groups : ℕ := 2

/-- Theorem stating that the maximum number of groups is 2 -/
theorem max_groups_is_two :
  (num_boys % max_groups = 0) ∧
  (num_girls % max_groups = 0) ∧
  (total_students % max_groups = 0) ∧
  (max_groups % num_age_groups = 0) ∧
  (max_groups % num_skill_levels = 0) ∧
  (∀ n : ℕ, n > max_groups →
    (num_boys % n ≠ 0) ∨
    (num_girls % n ≠ 0) ∨
    (total_students % n ≠ 0) ∨
    (n % num_age_groups ≠ 0) ∨
    (n % num_skill_levels ≠ 0)) :=
by sorry

end max_groups_is_two_l713_71345


namespace ball_triangle_ratio_l713_71384

theorem ball_triangle_ratio (q : ℝ) (hq : q ≠ 1) :
  let r := 2012 / q
  let side1 := r * (1 + q)
  let side2 := 2012 * (1 + q)
  let side3 := 2012 * (1 + q^2) / q
  (side1^2 + side2^2 + side3^2) / (side1 + side2 + side3) = 4024 :=
sorry

end ball_triangle_ratio_l713_71384


namespace wire_cutting_problem_l713_71380

theorem wire_cutting_problem (initial_length second_length num_pieces : ℕ) 
  (h1 : initial_length = 1000)
  (h2 : second_length = 1050)
  (h3 : num_pieces = 14)
  (h4 : ∃ (piece_length : ℕ), 
    piece_length * num_pieces = initial_length ∧ 
    piece_length * num_pieces = second_length) :
  ∃ (piece_length : ℕ), piece_length = 71 ∧ 
    piece_length * num_pieces = initial_length ∧ 
    piece_length * num_pieces = second_length :=
by sorry

end wire_cutting_problem_l713_71380


namespace cubic_factorization_sum_of_squares_l713_71356

theorem cubic_factorization_sum_of_squares (a b c d e f : ℤ) : 
  (∀ x : ℚ, 729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 := by
sorry

end cubic_factorization_sum_of_squares_l713_71356


namespace necessary_but_not_sufficient_condition_l713_71386

/-- The equation x^2 + ax + b = 0 has two distinct positive roots less than 1 -/
def has_two_distinct_positive_roots_less_than_one (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
    x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0

/-- p is a necessary but not sufficient condition for q -/
theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (has_two_distinct_positive_roots_less_than_one a b → -2 < a ∧ a < 0 ∧ 0 < b ∧ b < 1) ∧
  ¬((-2 < a ∧ a < 0 ∧ 0 < b ∧ b < 1) → has_two_distinct_positive_roots_less_than_one a b) :=
by sorry

end necessary_but_not_sufficient_condition_l713_71386


namespace geometric_series_sum_l713_71371

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/3
  let S : ℝ := ∑' n, a * r^n
  S = 3/2 := by sorry

end geometric_series_sum_l713_71371


namespace valid_numbers_with_sum_444_l713_71354

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 10 ≠ 0 ∧ (n / 10) % 10 ≠ 0 ∧ n / 100 ≠ 0

def sum_of_permutations (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  if a = b ∧ b = c then
    n
  else if a = b ∨ b = c ∨ a = c then
    100 * (a + b + c) + 10 * (a + b + c) + (a + b + c)
  else
    222 * (a + b + c)

theorem valid_numbers_with_sum_444 (n : ℕ) :
  is_valid_number n ∧ sum_of_permutations n = 444 →
  n = 112 ∨ n = 121 ∨ n = 211 ∨ n = 444 :=
by
  sorry

end valid_numbers_with_sum_444_l713_71354


namespace dinosaur_weight_theorem_l713_71344

/-- The combined weight of Barney, five regular dinosaurs, and their food -/
def total_weight (regular_weight food_weight : ℕ) : ℕ :=
  let regular_combined := 5 * regular_weight
  let barney_weight := regular_combined + 1500
  barney_weight + regular_combined + food_weight

/-- Theorem stating the total weight of the dinosaurs and their food -/
theorem dinosaur_weight_theorem (X : ℕ) :
  total_weight 800 X = 9500 + X :=
by
  sorry

end dinosaur_weight_theorem_l713_71344


namespace max_pairs_after_loss_l713_71388

/-- Given a collection of shoes and a number of lost individual shoes,
    calculate the maximum number of matching pairs remaining. -/
def maxRemainingPairs (totalPairs : ℕ) (lostShoes : ℕ) : ℕ :=
  totalPairs - (lostShoes / 2) - (lostShoes % 2)

/-- Theorem: Given 150 pairs of shoes and a loss of 37 individual shoes,
    the maximum number of matching pairs remaining is 131. -/
theorem max_pairs_after_loss :
  maxRemainingPairs 150 37 = 131 := by
  sorry

#eval maxRemainingPairs 150 37

end max_pairs_after_loss_l713_71388


namespace leahs_coins_value_l713_71347

/-- Represents the number and value of coins Leah has -/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total number of coins -/
def CoinCollection.total (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes

/-- Calculates the total value of coins in cents -/
def CoinCollection.value (c : CoinCollection) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes

/-- Theorem stating that Leah's coins are worth 66 cents -/
theorem leahs_coins_value (c : CoinCollection) : c.value = 66 :=
  by
  have h1 : c.total = 15 := sorry
  have h2 : c.pennies = c.nickels + 3 := sorry
  sorry

end leahs_coins_value_l713_71347


namespace inclination_angle_range_l713_71372

/-- Given a line with slope k in [-1, √3] and inclination angle α in [0, π),
    prove that the range of α is [0, π/3] ∪ [3π/4, π) -/
theorem inclination_angle_range (k α : ℝ) :
  k ∈ Set.Icc (-1) (Real.sqrt 3) →
  α ∈ Set.Ico 0 π →
  k = Real.tan α →
  α ∈ Set.Icc 0 (π / 3) ∪ Set.Ico (3 * π / 4) π :=
sorry

end inclination_angle_range_l713_71372


namespace norma_bananas_l713_71337

theorem norma_bananas (initial : ℕ) (lost : ℕ) (final : ℕ) :
  initial = 47 →
  lost = 45 →
  final = initial - lost →
  final = 2 :=
by sorry

end norma_bananas_l713_71337


namespace max_xy_on_line_segment_l713_71303

/-- The maximum value of xy for a point P(x,y) on the line segment between A(3,0) and B(0,4) is 3 -/
theorem max_xy_on_line_segment : 
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 4)
  ∃ (M : ℝ), M = 3 ∧ 
    ∀ (P : ℝ × ℝ), (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) → 
      P.1 * P.2 ≤ M :=
by sorry

end max_xy_on_line_segment_l713_71303


namespace empty_union_l713_71379

theorem empty_union (A : Set α) : A ∪ ∅ = A := by sorry

end empty_union_l713_71379


namespace prime_natural_equation_solutions_l713_71323

theorem prime_natural_equation_solutions :
  ∀ p n : ℕ,
    Prime p →
    p^2 + n^2 = 3*p*n + 1 →
    ((p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8)) := by
  sorry

end prime_natural_equation_solutions_l713_71323


namespace breakfast_rearrangements_count_l713_71341

/-- The number of distinguishable rearrangements of "BREAKFAST" with vowels first -/
def breakfast_rearrangements : ℕ :=
  let vowels := 3  -- Number of vowels in "BREAKFAST"
  let repeated_vowels := 2  -- Number of times 'A' appears
  let consonants := 6  -- Number of consonants in "BREAKFAST"
  (vowels.factorial / repeated_vowels.factorial) * consonants.factorial

/-- Theorem stating that the number of rearrangements is 2160 -/
theorem breakfast_rearrangements_count :
  breakfast_rearrangements = 2160 := by
  sorry

end breakfast_rearrangements_count_l713_71341


namespace sum_of_number_and_its_square_l713_71310

theorem sum_of_number_and_its_square : 17 + 17^2 = 306 := by
  sorry

end sum_of_number_and_its_square_l713_71310


namespace sphere_radius_ratio_l713_71378

theorem sphere_radius_ratio (V₁ V₂ : ℝ) (r₁ r₂ : ℝ) 
  (h₁ : V₁ = (4 / 3) * π * r₁^3)
  (h₂ : V₂ = (4 / 3) * π * r₂^3)
  (h₃ : V₁ = 432 * π)
  (h₄ : V₂ = 0.25 * V₁) :
  r₂ / r₁ = 1 / Real.rpow 3 (1/3) := by
sorry

end sphere_radius_ratio_l713_71378


namespace calculate_expression_l713_71304

theorem calculate_expression : 15 * 25 + 35 * 15 + 16 * 28 + 32 * 16 = 1860 := by
  sorry

end calculate_expression_l713_71304


namespace power_of_two_sum_l713_71362

theorem power_of_two_sum : 2^3 * 2^4 + 2^5 = 160 := by
  sorry

end power_of_two_sum_l713_71362


namespace lcm_sum_implies_product_div_3_or_5_l713_71346

theorem lcm_sum_implies_product_div_3_or_5 (a b c d : ℕ) :
  Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = a + b + c + d →
  3 ∣ (a * b * c * d) ∨ 5 ∣ (a * b * c * d) := by
  sorry

end lcm_sum_implies_product_div_3_or_5_l713_71346


namespace binary_multiplication_division_l713_71333

/-- Converts a binary string to a natural number -/
def binary_to_nat (s : String) : ℕ := sorry

/-- Converts a natural number to a binary string -/
def nat_to_binary (n : ℕ) : String := sorry

theorem binary_multiplication_division :
  let a := binary_to_nat "1011010"
  let b := binary_to_nat "1010100"
  let c := binary_to_nat "100"
  let result := binary_to_nat "110001111100"
  (a / c) * b = result := by sorry

end binary_multiplication_division_l713_71333


namespace inheritance_division_l713_71321

/-- Proves that dividing $527,500 equally among 5 people results in each person receiving $105,500 -/
theorem inheritance_division (total_amount : ℕ) (num_people : ℕ) (individual_share : ℕ) : 
  total_amount = 527500 → num_people = 5 → individual_share = total_amount / num_people → 
  individual_share = 105500 := by
  sorry

end inheritance_division_l713_71321


namespace circle_construction_cases_l713_71393

/-- Two lines in a plane -/
structure Line where
  -- Add necessary fields for a line

/-- A point in a plane -/
structure Point where
  -- Add necessary fields for a point

/-- A circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry

/-- Predicate to check if two lines intersect -/
def lines_intersect (l1 l2 : Line) : Prop :=
  sorry

/-- Predicate to check if two lines are perpendicular -/
def lines_perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- Predicate to check if a circle is tangent to a line -/
def circle_tangent_to_line (c : Circle) (l : Line) : Prop :=
  sorry

/-- Predicate to check if a point is on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop :=
  sorry

/-- Main theorem -/
theorem circle_construction_cases
  (a b : Line) (P : Point)
  (h1 : lines_intersect a b)
  (h2 : point_on_line P b) :
  (∃ c1 c2 : Circle,
    c1 ≠ c2 ∧
    circle_tangent_to_line c1 a ∧
    circle_tangent_to_line c2 a ∧
    point_on_circle P c1 ∧
    point_on_circle P c2 ∧
    point_on_line c1.center b ∧
    point_on_line c2.center b) ∨
  (∃ Q : Point, point_on_line Q a ∧ point_on_line Q b ∧ P = Q) ∨
  (lines_perpendicular a b) :=
sorry

end circle_construction_cases_l713_71393


namespace company_merger_profit_distribution_l713_71373

theorem company_merger_profit_distribution (company_a_profit company_b_profit : ℝ) 
  (company_a_percentage : ℝ) :
  company_a_profit = 90000 ∧ 
  company_b_profit = 60000 ∧ 
  company_a_percentage = 60 →
  (company_b_profit / (company_a_profit + company_b_profit)) * 100 = 40 := by
  sorry

end company_merger_profit_distribution_l713_71373


namespace exactly_three_correct_delivery_l713_71377

/-- The number of houses and packages -/
def n : ℕ := 5

/-- The number of correctly delivered packages -/
def k : ℕ := 3

/-- The probability of exactly k out of n packages being delivered correctly -/
def prob_correct_delivery (n k : ℕ) : ℚ :=
  (n.choose k * (n - k).factorial) / n.factorial

theorem exactly_three_correct_delivery :
  prob_correct_delivery n k = 1 / 12 := by sorry

end exactly_three_correct_delivery_l713_71377


namespace square_diagonal_l713_71326

theorem square_diagonal (s : Real) (h : s > 0) (area_eq : s * s = 8) :
  Real.sqrt (2 * s * s) = 4 := by
  sorry

end square_diagonal_l713_71326


namespace vector_arrangements_l713_71358

-- Define a structure for a vector in 2D space
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define a function to check if two vectors are parallel
def areParallel (v1 v2 : Vector2D) : Prop :=
  ∃ (k : ℝ), v1.x = k * v2.x ∧ v1.y = k * v2.y

-- Define a function to check if a quadrilateral is non-convex
def isNonConvex (v1 v2 v3 v4 : Vector2D) : Prop :=
  sorry -- Definition of non-convex quadrilateral

-- Define a function to check if a four-segment broken line is self-intersecting
def isSelfIntersecting (v1 v2 v3 v4 : Vector2D) : Prop :=
  sorry -- Definition of self-intersecting broken line

theorem vector_arrangements (v1 v2 v3 v4 : Vector2D) :
  (¬ areParallel v1 v2 ∧ ¬ areParallel v1 v3 ∧ ¬ areParallel v1 v4 ∧
   ¬ areParallel v2 v3 ∧ ¬ areParallel v2 v4 ∧ ¬ areParallel v3 v4) →
  (v1.x + v2.x + v3.x + v4.x = 0 ∧ v1.y + v2.y + v3.y + v4.y = 0) →
  (∃ (a b c d : Vector2D), isNonConvex a b c d) ∧
  (∃ (a b c d : Vector2D), isSelfIntersecting a b c d) :=
by
  sorry


end vector_arrangements_l713_71358


namespace complex_fraction_equality_l713_71336

theorem complex_fraction_equality (z : ℂ) (h : z = 1 - I) : 
  (z^2 - 2*z) / (z - 1) = -1 - I := by sorry

end complex_fraction_equality_l713_71336


namespace log_of_geometric_is_arithmetic_l713_71361

theorem log_of_geometric_is_arithmetic (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_geom : b / a = c / b) : 
  Real.log b - Real.log a = Real.log c - Real.log b :=
sorry

end log_of_geometric_is_arithmetic_l713_71361


namespace no_solution_exists_l713_71314

theorem no_solution_exists : ¬∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧
  ∀ (n : ℕ), n > 0 → ((n - 2) / a ≤ ⌊b * n⌋ ∧ ⌊b * n⌋ < (n - 1) / a) :=
by sorry

end no_solution_exists_l713_71314


namespace root_in_interval_l713_71307

def f (x : ℝ) : ℝ := x^3 - x - 3

theorem root_in_interval :
  ∃ c ∈ Set.Icc 1 2, f c = 0 :=
sorry

end root_in_interval_l713_71307


namespace oldest_to_rick_age_ratio_l713_71318

/-- Proves that the ratio of the oldest brother's age to Rick's age is 2:1 --/
theorem oldest_to_rick_age_ratio :
  ∀ (rick_age oldest_age middle_age smallest_age youngest_age : ℕ),
    rick_age = 15 →
    ∃ (k : ℕ), oldest_age = k * rick_age →
    middle_age = oldest_age / 3 →
    smallest_age = middle_age / 2 →
    youngest_age = smallest_age - 2 →
    youngest_age = 3 →
    oldest_age / rick_age = 2 :=
by
  sorry

end oldest_to_rick_age_ratio_l713_71318


namespace inverse_variation_problem_l713_71302

-- Define the inverse relationship between a^3 and b^4
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^4 = k

-- Theorem statement
theorem inverse_variation_problem (a₀ b₀ a₁ b₁ : ℝ) 
  (h_inverse : inverse_relation a₀ b₀ ∧ inverse_relation a₁ b₁)
  (h_initial : a₀ = 2 ∧ b₀ = 4)
  (h_final_a : a₁ = 8) :
  b₁ = Real.sqrt 2 := by
  sorry

end inverse_variation_problem_l713_71302


namespace negative_fractions_comparison_l713_71360

theorem negative_fractions_comparison : -2/3 > -3/4 := by
  sorry

end negative_fractions_comparison_l713_71360


namespace shortest_distance_ln_to_line_l713_71317

/-- The shortest distance from a point on the curve y = ln x to the line 2x - y + 3 = 0 -/
theorem shortest_distance_ln_to_line : ∃ (d : ℝ), d = (4 + Real.log 2) / Real.sqrt 5 ∧
  ∀ (x y : ℝ), y = Real.log x →
    d ≤ (|2 * x - y + 3|) / Real.sqrt 5 := by
  sorry

end shortest_distance_ln_to_line_l713_71317


namespace select_blocks_count_l713_71397

/-- The number of ways to select 4 blocks from a 6x6 grid such that no two blocks are in the same row or column -/
def select_blocks : ℕ := (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

/-- Theorem stating that the number of ways to select 4 blocks from a 6x6 grid
    such that no two blocks are in the same row or column is 5400 -/
theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end select_blocks_count_l713_71397


namespace opposite_of_negative_half_l713_71365

theorem opposite_of_negative_half : -(-(1/2)) = 1/2 := by
  sorry

end opposite_of_negative_half_l713_71365


namespace parallel_lines_a_equals_two_l713_71305

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The first line equation: x + ay - 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y - 1 = 0

/-- The second line equation: ax + 4y + 2 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y + 2 = 0

theorem parallel_lines_a_equals_two :
  ∃ a : ℝ, (∀ x y, line1 a x y ↔ line2 a x y) → a = 2 :=
sorry

end parallel_lines_a_equals_two_l713_71305


namespace prob_more_surgeons_than_internists_mean_surgeons_selected_variance_surgeons_selected_l713_71328

/-- Represents the selection of doctors for a medical outreach program. -/
structure DoctorSelection where
  total : Nat
  surgeons : Nat
  internists : Nat
  ophthalmologists : Nat
  selected : Nat

/-- The specific scenario of selecting 3 out of 6 doctors. -/
def scenario : DoctorSelection :=
  { total := 6
  , surgeons := 2
  , internists := 2
  , ophthalmologists := 2
  , selected := 3 }

/-- The probability of selecting more surgeons than internists. -/
def probMoreSurgeonsThanInternists (s : DoctorSelection) : ℚ :=
  3 / 10

/-- The mean number of surgeons selected. -/
def meanSurgeonsSelected (s : DoctorSelection) : ℚ :=
  1

/-- The variance of the number of surgeons selected. -/
def varianceSurgeonsSelected (s : DoctorSelection) : ℚ :=
  2 / 5

/-- Theorem stating the probability of selecting more surgeons than internists. -/
theorem prob_more_surgeons_than_internists :
  probMoreSurgeonsThanInternists scenario = 3 / 10 := by
  sorry

/-- Theorem stating the mean number of surgeons selected. -/
theorem mean_surgeons_selected :
  meanSurgeonsSelected scenario = 1 := by
  sorry

/-- Theorem stating the variance of the number of surgeons selected. -/
theorem variance_surgeons_selected :
  varianceSurgeonsSelected scenario = 2 / 5 := by
  sorry

end prob_more_surgeons_than_internists_mean_surgeons_selected_variance_surgeons_selected_l713_71328


namespace ellipse_equation_l713_71340

/-- Prove that an ellipse passing through (2,0) with focal distance 2√2 has the equation x²/4 + y²/2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → x^2/4 + y^2/2 = 1) ↔
  (4/a^2 + 0^2/b^2 = 1 ∧ a^2 - b^2 = 2) := by
  sorry

end ellipse_equation_l713_71340


namespace box_counting_l713_71385

theorem box_counting (initial_boxes : Nat) (boxes_per_fill : Nat) (non_empty_boxes : Nat) :
  initial_boxes = 7 →
  boxes_per_fill = 7 →
  non_empty_boxes = 10 →
  initial_boxes + (non_empty_boxes - 1) * boxes_per_fill = 77 := by
  sorry

end box_counting_l713_71385


namespace geometric_sequence_general_term_l713_71383

/-- A geometric sequence with specific properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) ∧
  |a 1| = 1 ∧
  a 5 = -8 * a 2 ∧
  a 5 > a 2

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := (-2) ^ (n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  geometric_sequence a → (∀ n : ℕ, a n = general_term n) :=
by sorry

end geometric_sequence_general_term_l713_71383


namespace largest_c_for_negative_three_in_range_l713_71399

-- Define the function f
def f (x c : ℝ) : ℝ := x^2 + 5*x + c

-- State the theorem
theorem largest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), 
    (∃ (x : ℝ), f x c = -3) → 
    (∃ (x : ℝ), f x d = -3) → 
    d ≤ c) ∧
  (∃ (x : ℝ), f x (13/4) = -3) :=
sorry

end largest_c_for_negative_three_in_range_l713_71399


namespace f_properties_l713_71368

/-- The function f(x) = mx² + 1 + ln x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 1 + Real.log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := 2 * m * x + 1 / x

/-- Theorem stating the main properties to be proved -/
theorem f_properties (m : ℝ) (n : ℝ) (a b : ℝ) :
  (∃ (t : ℝ), t = f_deriv m 1 ∧ 2 = f m 1 + t * (-2)) →  -- Tangent line condition
  (f m a = n ∧ f m b = n ∧ a < b) →                      -- Roots condition
  (∀ x > 0, f m x ≤ 1 - x) ∧                             -- Property 1
  (b - a < 1 - 2 * n)                                    -- Property 2
  := by sorry

end f_properties_l713_71368


namespace candy_sampling_probability_l713_71324

theorem candy_sampling_probability :
  let p_choose_A : ℝ := 0.40
  let p_choose_B : ℝ := 0.35
  let p_choose_C : ℝ := 0.25
  let p_sample_A : ℝ := 0.16 + 0.07
  let p_sample_B : ℝ := 0.24 + 0.15
  let p_sample_C : ℝ := 0.31 + 0.22
  let p_sample : ℝ := p_choose_A * p_sample_A + p_choose_B * p_sample_B + p_choose_C * p_sample_C
  p_sample = 0.361 :=
by sorry

end candy_sampling_probability_l713_71324


namespace book_sale_profit_percentage_l713_71367

/-- Calculates the profit percentage after tax for a book sale -/
theorem book_sale_profit_percentage 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (tax_rate : ℝ) 
  (h1 : cost_price = 32) 
  (h2 : selling_price = 56) 
  (h3 : tax_rate = 0.07) : 
  (selling_price * (1 - tax_rate) - cost_price) / cost_price * 100 = 62.75 := by
sorry

end book_sale_profit_percentage_l713_71367


namespace geometric_series_problem_l713_71330

theorem geometric_series_problem (n : ℝ) : 
  let a₁ := 15
  let r₁ := 5 / 15
  let S₁ := a₁ / (1 - r₁)
  let a₂ := 15
  let r₂ := (5 + n) / 15
  let S₂ := a₂ / (1 - r₂)
  S₂ = 3 * S₁ → n = 20/3 := by
sorry

end geometric_series_problem_l713_71330


namespace pages_read_on_tuesday_l713_71338

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Berry's daily reading goal -/
def daily_goal : ℕ := 50

/-- Pages read on Sunday -/
def sunday_pages : ℕ := 43

/-- Pages read on Monday -/
def monday_pages : ℕ := 65

/-- Pages read on Wednesday -/
def wednesday_pages : ℕ := 0

/-- Pages read on Thursday -/
def thursday_pages : ℕ := 70

/-- Pages read on Friday -/
def friday_pages : ℕ := 56

/-- Pages to be read on Saturday -/
def saturday_pages : ℕ := 88

/-- Theorem stating that Berry must have read 28 pages on Tuesday to achieve his weekly goal -/
theorem pages_read_on_tuesday : 
  ∃ (tuesday_pages : ℕ), 
    (sunday_pages + monday_pages + tuesday_pages + wednesday_pages + 
     thursday_pages + friday_pages + saturday_pages) = 
    (daily_goal * days_in_week) ∧ tuesday_pages = 28 := by
  sorry

end pages_read_on_tuesday_l713_71338


namespace parking_lot_cars_l713_71389

theorem parking_lot_cars (red_cars : ℕ) (black_cars : ℕ) : 
  red_cars = 33 → 
  red_cars * 8 = black_cars * 3 → 
  black_cars = 88 := by
sorry

end parking_lot_cars_l713_71389


namespace min_cubes_is_60_l713_71366

/-- The dimensions of the box in centimeters -/
def box_dimensions : Fin 3 → ℕ
| 0 => 30
| 1 => 40
| 2 => 50
| _ => 0

/-- The function to calculate the minimum number of cubes -/
def min_cubes (dimensions : Fin 3 → ℕ) : ℕ :=
  let cube_side := Nat.gcd (dimensions 0) (Nat.gcd (dimensions 1) (dimensions 2))
  (dimensions 0 / cube_side) * (dimensions 1 / cube_side) * (dimensions 2 / cube_side)

/-- Theorem stating that the minimum number of cubes is 60 -/
theorem min_cubes_is_60 : min_cubes box_dimensions = 60 := by
  sorry

#eval min_cubes box_dimensions

end min_cubes_is_60_l713_71366


namespace right_triangle_inequality_l713_71395

theorem right_triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b < c) (hright : a^2 + b^2 = c^2) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3 * Real.sqrt 2) * a * b * c :=
sorry

end right_triangle_inequality_l713_71395


namespace box_storage_calculation_l713_71387

/-- Calculates the total number of boxes stored on a rectangular piece of land over two days -/
theorem box_storage_calculation (land_width land_length : ℕ) 
  (box_dimension : ℕ) (day1_layers day2_layers : ℕ) : 
  land_width = 44 → 
  land_length = 35 → 
  box_dimension = 1 → 
  day1_layers = 7 → 
  day2_layers = 3 → 
  (land_width / box_dimension) * (land_length / box_dimension) * (day1_layers + day2_layers) = 15400 :=
by
  sorry

#check box_storage_calculation

end box_storage_calculation_l713_71387


namespace simplify_expression_l713_71370

theorem simplify_expression (y : ℝ) : 3*y + 5*y + 6*y + 10 = 14*y + 10 := by
  sorry

end simplify_expression_l713_71370


namespace function_difference_bound_l713_71313

/-- Given a function f(x) = x^2 - x + c and a real number a such that |x - a| < 1,
    prove that |f(x) - f(a)| < 2(|a| + 1) -/
theorem function_difference_bound (c a x : ℝ) (h : |x - a| < 1) :
  let f := fun (t : ℝ) => t^2 - t + c
  |f x - f a| < 2 * (|a| + 1) := by
sorry


end function_difference_bound_l713_71313


namespace sum_range_l713_71381

theorem sum_range (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let S := a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)
  1 ≤ S ∧ S ≤ 4 / 3 :=
by sorry

end sum_range_l713_71381


namespace triangle_max_area_l713_71394

theorem triangle_max_area (A B C : ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) (h5 : Real.cos A / Real.sin B + Real.cos B / Real.sin A = 2) 
  (h6 : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 12 ∧ 
    a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  ∃ (S : ℝ), S ≤ 36 * (3 - 2 * Real.sqrt 2) ∧ 
    (∀ (S' : ℝ), (∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 12 ∧ 
      a' / Real.sin A = b' / Real.sin B ∧ b' / Real.sin B = c' / Real.sin C ∧ 
      S' = 1/2 * a' * b' * Real.sin C) → S' ≤ S) :=
sorry

end triangle_max_area_l713_71394


namespace cost_of_eggs_l713_71315

/-- The amount Samantha spent on the crate of eggs -/
def cost : ℝ := 5

/-- The number of eggs in the crate -/
def total_eggs : ℕ := 30

/-- The price of each egg in dollars -/
def price_per_egg : ℝ := 0.20

/-- The number of eggs left when Samantha recovers her capital -/
def eggs_left : ℕ := 5

/-- Theorem stating that the cost of the crate is $5 -/
theorem cost_of_eggs : cost = (total_eggs - eggs_left) * price_per_egg := by
  sorry

end cost_of_eggs_l713_71315


namespace group_collection_problem_l713_71306

theorem group_collection_problem (n : ℕ) (total_rupees : ℚ) : 
  (n : ℚ) * n = total_rupees * 100 →
  total_rupees = 19.36 →
  n = 44 := by
  sorry

end group_collection_problem_l713_71306


namespace all_propositions_true_l713_71309

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (lineparallel : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (equidistant : Plane → Plane → Point → Prop)
variable (noncollinear : Point → Point → Point → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Define the theorem
theorem all_propositions_true :
  (∀ (n : Line) (α β : Plane), perpendicular n α → perpendicular n β → parallel α β) ∧
  (∀ (α β : Plane) (p q r : Point), noncollinear p q r → equidistant α β p → equidistant α β q → equidistant α β r → parallel α β) ∧
  (∀ (m n : Line) (α β : Plane), skew m n → contains α n → lineparallel n β → contains β m → lineparallel m α → parallel α β) :=
sorry

end all_propositions_true_l713_71309


namespace mans_rate_l713_71349

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 6)
  (h2 : speed_against_stream = 3) :
  (speed_with_stream + speed_against_stream) / 2 = 4.5 := by
  sorry

#check mans_rate

end mans_rate_l713_71349


namespace divides_product_l713_71322

theorem divides_product (a b c d : ℤ) (h1 : a ∣ b) (h2 : c ∣ d) : a * c ∣ b * d := by
  sorry

end divides_product_l713_71322


namespace tick_to_burr_ratio_l713_71369

/-- Given a dog with burrs and ticks in its fur, prove the ratio of ticks to burrs. -/
theorem tick_to_burr_ratio (num_burrs num_total : ℕ) (h1 : num_burrs = 12) (h2 : num_total = 84) :
  (num_total - num_burrs) / num_burrs = 6 := by
  sorry

end tick_to_burr_ratio_l713_71369


namespace work_completion_time_l713_71391

theorem work_completion_time 
  (people : ℕ) 
  (original_time : ℕ) 
  (original_work : ℝ) 
  (h1 : original_time = 16) 
  (h2 : people * original_time = original_work) :
  (2 * people) * 8 = original_work / 2 :=
sorry

end work_completion_time_l713_71391
