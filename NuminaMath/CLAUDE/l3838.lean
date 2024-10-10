import Mathlib

namespace exam_max_marks_calculation_l3838_383804

/-- Represents the maximum marks and passing criteria for a subject -/
structure Subject where
  max_marks : ℕ
  passing_percentage : ℚ

/-- Represents a student's performance in a subject -/
structure Performance where
  score : ℕ
  failed_by : ℕ

/-- Calculates the maximum marks for a subject given the performance and passing criteria -/
def calculate_max_marks (perf : Performance) (pass_percentage : ℚ) : ℕ :=
  ((perf.score + perf.failed_by : ℚ) / pass_percentage).ceil.toNat

theorem exam_max_marks_calculation (math science english : Subject) 
    (math_perf science_perf english_perf : Performance) : 
    math.max_marks = 275 ∧ science.max_marks = 414 ∧ english.max_marks = 300 :=
  by
    have h_math : math.passing_percentage = 2/5 := by sorry
    have h_science : science.passing_percentage = 7/20 := by sorry
    have h_english : english.passing_percentage = 3/10 := by sorry
    
    have h_math_perf : math_perf = ⟨90, 20⟩ := by sorry
    have h_science_perf : science_perf = ⟨110, 35⟩ := by sorry
    have h_english_perf : english_perf = ⟨80, 10⟩ := by sorry
    
    have h_math_max : math.max_marks = calculate_max_marks math_perf math.passing_percentage := by sorry
    have h_science_max : science.max_marks = calculate_max_marks science_perf science.passing_percentage := by sorry
    have h_english_max : english.max_marks = calculate_max_marks english_perf english.passing_percentage := by sorry
    
    sorry

end exam_max_marks_calculation_l3838_383804


namespace triangle_properties_l3838_383821

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 + t.b^2 - t.c^2 = 4 * t.area) 
  (h2 : t.c = Real.sqrt 2) : 
  (t.C = Real.pi / 4) ∧ 
  (-1 < t.a - (Real.sqrt 2 / 2) * t.b) ∧ 
  (t.a - (Real.sqrt 2 / 2) * t.b < Real.sqrt 2) := by
  sorry


end triangle_properties_l3838_383821


namespace function_properties_l3838_383852

open Real

theorem function_properties (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < -1) :
  (f (-2) > f 2 + 4) ∧ 
  (∀ x : ℝ, f x > f (x + 1) + 1) ∧ 
  (∃ x : ℝ, x ≥ 0 ∧ f (sqrt x) + sqrt x < f 0) ∧
  (∀ a : ℝ, a ≠ 0 → f (|a| + 1 / |a|) + |a| + 1 / |a| < f 2 + 3) :=
by sorry

end function_properties_l3838_383852


namespace daltons_uncle_gift_l3838_383889

/-- The amount of money Dalton's uncle gave him -/
def uncles_gift (jump_rope_cost board_game_cost playground_ball_cost savings needed : ℕ) : ℕ :=
  jump_rope_cost + board_game_cost + playground_ball_cost - savings - needed

theorem daltons_uncle_gift :
  uncles_gift 7 12 4 6 4 = 13 := by
  sorry

end daltons_uncle_gift_l3838_383889


namespace three_circles_tangency_theorem_l3838_383878

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the structure for a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to check if two circles are tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define the function to get the tangency point of two circles
def tangency_point (c1 c2 : Circle) : Point :=
  sorry

-- Define the function to check if a point is on a circle
def point_on_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := c.center
  (p.x - x)^2 + (p.y - y)^2 = c.radius^2

-- Define the function to check if two points form a diameter of a circle
def is_diameter (p1 p2 : Point) (c : Circle) : Prop :=
  let (x, y) := c.center
  (p1.x + p2.x) / 2 = x ∧ (p1.y + p2.y) / 2 = y

-- Theorem statement
theorem three_circles_tangency_theorem (S1 S2 S3 : Circle) :
  are_tangent S1 S2 ∧ are_tangent S2 S3 ∧ are_tangent S3 S1 →
  let C := tangency_point S1 S2
  let A := tangency_point S2 S3
  let B := tangency_point S3 S1
  let A1 := sorry -- Intersection of line CA with S3
  let B1 := sorry -- Intersection of line CB with S3
  point_on_circle A1 S3 ∧ point_on_circle B1 S3 ∧ is_diameter A1 B1 S3 :=
sorry

end three_circles_tangency_theorem_l3838_383878


namespace quadratic_increasing_condition_l3838_383811

/-- A function f is increasing on an interval (a, +∞) if for all x₁, x₂ in the interval,
    x₁ < x₂ implies f x₁ < f x₂ -/
def IncreasingOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The quadratic function we're considering -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1 - a)*x + 2

theorem quadratic_increasing_condition (a : ℝ) :
  IncreasingOn (f a) 4 → a ≤ 9 := by
  sorry

end quadratic_increasing_condition_l3838_383811


namespace sophie_coin_distribution_l3838_383864

/-- The minimum number of additional coins needed for Sophie's distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins Sophie needs. -/
theorem sophie_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 10) (h2 : initial_coins = 40) : 
  min_additional_coins num_friends initial_coins = 15 := by
  sorry

#eval min_additional_coins 10 40

end sophie_coin_distribution_l3838_383864


namespace rational_equation_solution_l3838_383848

theorem rational_equation_solution :
  let x : ℚ := -26/9
  (2*x + 18) / (x - 6) = (2*x - 4) / (x + 10) := by
  sorry

end rational_equation_solution_l3838_383848


namespace next_common_term_l3838_383891

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem next_common_term
  (a₁ b₁ d₁ d₂ : ℤ)
  (h₁ : a₁ = 3)
  (h₂ : b₁ = 16)
  (h₃ : d₁ = 17)
  (h₄ : d₂ = 11)
  (h₅ : ∃ (n m : ℕ), arithmetic_sequence a₁ d₁ n = 71 ∧ arithmetic_sequence b₁ d₂ m = 71)
  : ∃ (k l : ℕ), 
    arithmetic_sequence a₁ d₁ k = arithmetic_sequence b₁ d₂ l ∧
    arithmetic_sequence a₁ d₁ k > 71 ∧
    arithmetic_sequence a₁ d₁ k = 258 :=
sorry

end next_common_term_l3838_383891


namespace square_field_area_l3838_383850

/-- Given a square field where the diagonal can be traversed at 8 km/hr in 0.5 hours,
    the area of the field is 8 square kilometers. -/
theorem square_field_area (speed : ℝ) (time : ℝ) (diagonal : ℝ) (side : ℝ) (area : ℝ) : 
  speed = 8 →
  time = 0.5 →
  diagonal = speed * time →
  diagonal^2 = 2 * side^2 →
  area = side^2 →
  area = 8 := by
sorry

end square_field_area_l3838_383850


namespace hyperbola_equation_l3838_383824

/-- The standard equation of a hyperbola with the same asymptotes as x²/9 - y²/16 = 1
    and passing through the point (-√3, 2√3) -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ m : ℝ, x^2 / 9 - y^2 / 16 = m) ∧
  ((-Real.sqrt 3)^2 / 9 - (2 * Real.sqrt 3)^2 / 16 = -5/12) →
  y^2 / 5 - x^2 / (15/4) = 1 :=
sorry

end hyperbola_equation_l3838_383824


namespace a_less_than_one_l3838_383828

theorem a_less_than_one : 
  (0.99999 : ℝ)^(1.00001 : ℝ) * (1.00001 : ℝ)^(0.99999 : ℝ) < 1 := by
  sorry

end a_less_than_one_l3838_383828


namespace train_distance_problem_l3838_383865

theorem train_distance_problem (speed1 speed2 distance_difference : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 25)
  (h3 : distance_difference = 65)
  : speed1 * (distance_difference / (speed2 - speed1)) + 
    speed2 * (distance_difference / (speed2 - speed1)) = 585 := by
  sorry

end train_distance_problem_l3838_383865


namespace division_remainder_proof_l3838_383870

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 166) (h2 : divisor = 20) (h3 : quotient = 8) :
  dividend % divisor = 6 := by
sorry

end division_remainder_proof_l3838_383870


namespace coastal_village_population_l3838_383805

theorem coastal_village_population (total_population : ℕ) 
  (h1 : total_population = 540) 
  (h2 : ∃ (part_size : ℕ), 4 * part_size = total_population) 
  (h3 : ∃ (male_population : ℕ), male_population = 2 * (total_population / 4)) :
  ∃ (male_population : ℕ), male_population = 270 := by
sorry

end coastal_village_population_l3838_383805


namespace coffee_stock_problem_l3838_383813

/-- Represents the coffee stock problem --/
theorem coffee_stock_problem 
  (initial_stock : ℝ) 
  (additional_purchase : ℝ) 
  (decaf_percent_additional : ℝ) 
  (total_decaf_percent : ℝ) : 
  initial_stock = 400 ∧ 
  additional_purchase = 100 ∧ 
  decaf_percent_additional = 60 ∧ 
  total_decaf_percent = 32 → 
  (initial_stock * (25 / 100) + additional_purchase * (decaf_percent_additional / 100)) / 
  (initial_stock + additional_purchase) = total_decaf_percent / 100 := by
  sorry

#check coffee_stock_problem

end coffee_stock_problem_l3838_383813


namespace extremum_implies_f_2_l3838_383803

/-- A function f(x) with an extremum at x = 1 and f(1) = 10 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2 (a b : ℝ) :
  f' a b 1 = 0 → f a b 1 = 10 → f a b 2 = 2 := by
  sorry

#check extremum_implies_f_2

end extremum_implies_f_2_l3838_383803


namespace x_intercept_of_line_l3838_383857

theorem x_intercept_of_line (x y : ℝ) : 
  (5 * x - 2 * y - 10 = 0) → (y = 0 → x = 2) :=
by sorry

end x_intercept_of_line_l3838_383857


namespace smallest_base_perfect_square_five_is_smallest_smallest_base_is_five_l3838_383819

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 4 → (∃ n : ℕ, 4 * b + 5 = n^2) → b ≥ 5 :=
by
  sorry

theorem five_is_smallest :
  ∃ n : ℕ, 4 * 5 + 5 = n^2 :=
by
  sorry

theorem smallest_base_is_five :
  ∀ b : ℕ, b > 4 ∧ (∃ n : ℕ, 4 * b + 5 = n^2) → b ≥ 5 :=
by
  sorry

end smallest_base_perfect_square_five_is_smallest_smallest_base_is_five_l3838_383819


namespace heart_diamond_spade_probability_l3838_383862

/-- Probability of drawing a heart, then a diamond, then a spade from a standard 52-card deck -/
theorem heart_diamond_spade_probability : 
  let total_cards : ℕ := 52
  let hearts : ℕ := 13
  let diamonds : ℕ := 13
  let spades : ℕ := 13
  (hearts : ℚ) / total_cards * 
  (diamonds : ℚ) / (total_cards - 1) * 
  (spades : ℚ) / (total_cards - 2) = 2197 / 132600 := by
sorry

end heart_diamond_spade_probability_l3838_383862


namespace factorial_ratio_simplification_l3838_383893

theorem factorial_ratio_simplification (N : ℕ) :
  (Nat.factorial (N + 1) * N) / Nat.factorial (N + 2) = N / (N + 2) := by
  sorry

end factorial_ratio_simplification_l3838_383893


namespace project_speedup_l3838_383860

/-- Calculates the number of days saved when additional workers join a project -/
def days_saved (original_workers : ℕ) (original_days : ℕ) (additional_workers : ℕ) : ℕ :=
  original_days - (original_workers * original_days) / (original_workers + additional_workers)

/-- Theorem stating that 10 additional workers save 6 days on a 12-day project with 10 original workers -/
theorem project_speedup :
  days_saved 10 12 10 = 6 := by sorry

end project_speedup_l3838_383860


namespace polynomial_subtraction_l3838_383881

theorem polynomial_subtraction :
  let p₁ : Polynomial ℝ := X^5 - 3*X^4 + X^2 + 15
  let p₂ : Polynomial ℝ := 2*X^5 - 3*X^3 + 2*X^2 + 18
  p₁ - p₂ = -X^5 - 3*X^4 + 3*X^3 - X^2 - 3 := by sorry

end polynomial_subtraction_l3838_383881


namespace f_properties_l3838_383823

noncomputable section

def f (x : ℝ) := Real.log x - (x - 1)^2 / 2

theorem f_properties :
  let φ := (1 + Real.sqrt 5) / 2
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < φ → f x₁ < f x₂) ∧
  (∀ x, x > 1 → f x < x - 1) ∧
  (∀ k, k < 1 → ∃ x₀ > 1, ∀ x, 1 < x ∧ x < x₀ → f x > k * (x - 1)) ∧
  (∀ k, k ≥ 1 → ¬∃ x₀ > 1, ∀ x, 1 < x ∧ x < x₀ → f x > k * (x - 1)) :=
by sorry

end

end f_properties_l3838_383823


namespace num_ways_to_sum_correct_l3838_383817

/-- The number of ways to choose k natural numbers that sum to n -/
def num_ways_to_sum (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem: The number of ways to choose k natural numbers that sum to n
    is equal to (n+k-1) choose (k-1) -/
theorem num_ways_to_sum_correct (n k : ℕ) :
  num_ways_to_sum n k = Nat.choose (n + k - 1) (k - 1) := by
  sorry

#check num_ways_to_sum_correct

end num_ways_to_sum_correct_l3838_383817


namespace distance_from_origin_l3838_383897

theorem distance_from_origin (x y : ℝ) (n : ℝ) : 
  y = 15 →
  (x - 2)^2 + (y - 8)^2 = 13^2 →
  x > 2 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (349 + 8 * Real.sqrt 30) := by
sorry

end distance_from_origin_l3838_383897


namespace pure_imaginary_complex_number_l3838_383899

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) * (2 - Complex.I)
  (z.re = 0) → m = -2 := by
sorry

end pure_imaginary_complex_number_l3838_383899


namespace salary_comparison_l3838_383800

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) :
  (b - a) / a * 100 = 25 := by
  sorry

end salary_comparison_l3838_383800


namespace solve_for_y_l3838_383890

theorem solve_for_y (x y : ℝ) (h1 : x^(y+1) = 16) (h2 : x = 8) : y = 0 := by
  sorry

end solve_for_y_l3838_383890


namespace johns_donation_size_l3838_383840

/-- Represents the donation problem with given conditions -/
structure DonationProblem where
  num_previous_donations : ℕ
  new_average : ℚ
  increase_percentage : ℚ

/-- Calculates John's donation size based on the given conditions -/
def calculate_donation_size (problem : DonationProblem) : ℚ :=
  let previous_average := problem.new_average / (1 + problem.increase_percentage)
  let total_before := previous_average * problem.num_previous_donations
  let total_after := problem.new_average * (problem.num_previous_donations + 1)
  total_after - total_before

/-- Theorem stating that John's donation size is $225 given the problem conditions -/
theorem johns_donation_size (problem : DonationProblem) 
  (h1 : problem.num_previous_donations = 6)
  (h2 : problem.new_average = 75)
  (h3 : problem.increase_percentage = 1/2) :
  calculate_donation_size problem = 225 := by
  sorry

#eval calculate_donation_size { num_previous_donations := 6, new_average := 75, increase_percentage := 1/2 }

end johns_donation_size_l3838_383840


namespace slope_angle_of_line_l3838_383868

theorem slope_angle_of_line (x y : ℝ) :
  x + Real.sqrt 3 * y - 3 = 0 →
  let m := -1 / Real.sqrt 3
  let α := Real.arctan m
  α = 150 * π / 180 := by
sorry

end slope_angle_of_line_l3838_383868


namespace quadratic_roots_problem_l3838_383876

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + k * x₁ - 2 = 0) → 
  (2 * x₂^2 + k * x₂ - 2 = 0) → 
  ((x₁ - 2) * (x₂ - 2) = 10) → 
  k = 7 :=
by sorry

end quadratic_roots_problem_l3838_383876


namespace problem_statement_l3838_383816

theorem problem_statement :
  (∀ x : ℝ, x < 0 → (2 : ℝ)^x > (3 : ℝ)^x) ∧
  (¬ ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ Real.sin x > x) :=
by sorry

end problem_statement_l3838_383816


namespace triangle_area_is_one_l3838_383801

-- Define the complex number z
def z : ℂ := sorry

-- State the given conditions
axiom z_magnitude : Complex.abs z = Real.sqrt 2
axiom z_squared_imag : Complex.im (z ^ 2) = 2

-- Define the points A, B, and C
def A : ℂ := z
def B : ℂ := z ^ 2
def C : ℂ := z - z ^ 2

-- Define the area of the triangle
def triangle_area : ℝ := sorry

-- State the theorem to be proved
theorem triangle_area_is_one : triangle_area = 1 := by sorry

end triangle_area_is_one_l3838_383801


namespace zoo_rabbits_l3838_383892

/-- Given a zoo with parrots and rabbits, where the ratio of parrots to rabbits
    is 3:4 and there are 21 parrots, prove that there are 28 rabbits. -/
theorem zoo_rabbits (parrots : ℕ) (rabbits : ℕ) : 
  parrots = 21 → 3 * rabbits = 4 * parrots → rabbits = 28 := by
  sorry

end zoo_rabbits_l3838_383892


namespace total_ways_eq_19_l3838_383853

/-- The number of direct bus services from place A to place B -/
def direct_services : ℕ := 4

/-- The number of bus services from place A to place C -/
def services_A_to_C : ℕ := 5

/-- The number of bus services from place C to place B -/
def services_C_to_B : ℕ := 3

/-- The total number of ways to travel from place A to place B -/
def total_ways : ℕ := direct_services + services_A_to_C * services_C_to_B

theorem total_ways_eq_19 : total_ways = 19 := by
  sorry

end total_ways_eq_19_l3838_383853


namespace consecutive_integers_average_l3838_383849

theorem consecutive_integers_average (c d : ℤ) : 
  (c > 0) →
  (d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7) →
  ((d + (d+1) + (d+2) + (d+3) + (d+4) + (d+5) + (d+6)) / 7 = c + 6) :=
by sorry

end consecutive_integers_average_l3838_383849


namespace largest_three_digit_integer_l3838_383833

theorem largest_three_digit_integer (n : ℕ) (a b c : ℕ) : 
  n = 100 * a + 10 * b + c →
  100 ≤ n → n < 1000 →
  2 ∣ a →
  3 ∣ (10 * a + b) →
  ¬(6 ∣ (10 * a + b)) →
  5 ∣ n →
  ¬(7 ∣ n) →
  n ≤ 870 :=
by sorry

end largest_three_digit_integer_l3838_383833


namespace expression_factorization_l3838_383875

theorem expression_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (-(x*y + x*z + y*z)) := by
  sorry

end expression_factorization_l3838_383875


namespace fixed_point_of_exponential_function_l3838_383871

/-- The function f(x) = a^(x+2) - 3 passes through the point (-2, -2) for a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) - 3
  f (-2) = -2 := by sorry

end fixed_point_of_exponential_function_l3838_383871


namespace sandys_money_l3838_383898

theorem sandys_money (pie_cost sandwich_cost book_cost remaining_money : ℕ) : 
  pie_cost = 6 →
  sandwich_cost = 3 →
  book_cost = 10 →
  remaining_money = 38 →
  pie_cost + sandwich_cost + book_cost + remaining_money = 57 := by
sorry

end sandys_money_l3838_383898


namespace zero_necessary_for_odd_zero_not_sufficient_for_odd_l3838_383883

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- f(0) = 0 is a necessary condition for f to be odd -/
theorem zero_necessary_for_odd (f : ℝ → ℝ) :
  IsOdd f → f 0 = 0 :=
sorry

/-- f(0) = 0 is not a sufficient condition for f to be odd -/
theorem zero_not_sufficient_for_odd :
  ∃ f : ℝ → ℝ, f 0 = 0 ∧ ¬IsOdd f :=
sorry

end zero_necessary_for_odd_zero_not_sufficient_for_odd_l3838_383883


namespace images_per_card_l3838_383867

/-- The number of pictures John takes per day -/
def pictures_per_day : ℕ := 10

/-- The number of years John has been taking pictures -/
def years : ℕ := 3

/-- The cost of each memory card in dollars -/
def cost_per_card : ℕ := 60

/-- The total amount John spent on memory cards in dollars -/
def total_spent : ℕ := 13140

/-- The number of days in a year (assuming no leap years) -/
def days_per_year : ℕ := 365

theorem images_per_card : 
  (years * days_per_year * pictures_per_day) / (total_spent / cost_per_card) = 50 := by
  sorry

end images_per_card_l3838_383867


namespace union_when_m_neg_one_subset_condition_l3838_383847

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem for part 1
theorem union_when_m_neg_one :
  A ∪ B (-1) = {x : ℝ | -2 < x ∧ x < 3} := by sorry

-- Theorem for part 2
theorem subset_condition (m : ℝ) :
  A ⊆ B m ↔ m ≤ -2 := by sorry

end union_when_m_neg_one_subset_condition_l3838_383847


namespace gcd_of_squared_sums_gcd_of_specific_squared_sums_l3838_383829

theorem gcd_of_squared_sums (a b c d e f : ℕ) : 
  Nat.gcd (a^2 + b^2 + c^2) (d^2 + e^2 + f^2) = 
  Nat.gcd ((a^2 + b^2 + c^2) - (d^2 + e^2 + f^2)) (d^2 + e^2 + f^2) :=
by sorry

theorem gcd_of_specific_squared_sums : 
  Nat.gcd (131^2 + 243^2 + 357^2) (130^2 + 242^2 + 358^2) = 1 :=
by sorry

end gcd_of_squared_sums_gcd_of_specific_squared_sums_l3838_383829


namespace complex_magnitude_problem_l3838_383825

theorem complex_magnitude_problem (z : ℂ) (h : (1 - I) * z = 1 + I) : Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l3838_383825


namespace sum_of_composite_functions_l3838_383809

def p (x : ℝ) : ℝ := |x + 1| - 3

def q (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composite_functions :
  (x_values.map (λ x => q (p x))).sum = -12 := by
  sorry

end sum_of_composite_functions_l3838_383809


namespace acme_cheaper_than_beta_l3838_383854

/-- Acme's pricing function -/
def acme_price (n : ℕ) : ℕ := 45 + 10 * n

/-- Beta's pricing function -/
def beta_price (n : ℕ) : ℕ := 15 * n

/-- Beta's minimum order quantity -/
def beta_min_order : ℕ := 5

/-- The minimum number of shirts above Beta's minimum order for which Acme is cheaper -/
def min_shirts_above_min : ℕ := 5

theorem acme_cheaper_than_beta :
  ∀ n : ℕ, n ≥ beta_min_order + min_shirts_above_min →
    acme_price (beta_min_order + min_shirts_above_min) < beta_price (beta_min_order + min_shirts_above_min) ∧
    ∀ m : ℕ, m < beta_min_order + min_shirts_above_min → acme_price m ≥ beta_price m :=
by sorry

end acme_cheaper_than_beta_l3838_383854


namespace meeting_percentage_is_37_5_l3838_383877

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 8 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 45

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Represents the percentage of work day spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100

theorem meeting_percentage_is_37_5 : meeting_percentage = 37.5 := by
  sorry

end meeting_percentage_is_37_5_l3838_383877


namespace unique_solution_f_f_eq_zero_l3838_383851

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x + 4 else 3*x - 6

-- Theorem statement
theorem unique_solution_f_f_eq_zero :
  ∃! x : ℝ, f (f x) = 0 :=
sorry

end unique_solution_f_f_eq_zero_l3838_383851


namespace polynomial_evaluation_l3838_383834

theorem polynomial_evaluation :
  let x : ℝ := -2
  x^4 + x^3 + x^2 + x + 2 = 12 := by sorry

end polynomial_evaluation_l3838_383834


namespace energy_bar_difference_l3838_383830

theorem energy_bar_difference (older younger : ℕ) 
  (h1 : older = younger + 17) : 
  (older - 3) = (younger + 3) + 11 := by
  sorry

end energy_bar_difference_l3838_383830


namespace nested_fraction_simplification_l3838_383810

theorem nested_fraction_simplification :
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end nested_fraction_simplification_l3838_383810


namespace no_two_digit_factorization_2109_l3838_383802

/-- A function that returns the number of ways to factor a positive integer
    as a product of two two-digit numbers -/
def count_two_digit_factorizations (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    10 ≤ p.1 ∧ p.1 < 100 ∧ 
    10 ≤ p.2 ∧ p.2 < 100 ∧ 
    p.1 * p.2 = n)
    (Finset.product (Finset.range 90) (Finset.range 90))).card / 2

/-- Theorem stating that 2109 cannot be factored as a product of two two-digit numbers -/
theorem no_two_digit_factorization_2109 : 
  count_two_digit_factorizations 2109 = 0 := by
  sorry

end no_two_digit_factorization_2109_l3838_383802


namespace total_students_l3838_383818

/-- The number of students in different study halls --/
structure StudyHalls where
  general : ℕ
  biology : ℕ
  chemistry : ℕ
  math : ℕ
  arts : ℕ

/-- Conditions for the study halls problem --/
def study_halls_conditions (halls : StudyHalls) : Prop :=
  halls.general = 30 ∧
  halls.biology = 2 * halls.general ∧
  halls.chemistry = halls.general + 10 ∧
  halls.math = (3 * (halls.general + halls.biology + halls.chemistry)) / 5 ∧
  halls.arts * 20 / 100 = halls.general

/-- Theorem stating that the total number of students is 358 --/
theorem total_students (halls : StudyHalls) 
  (h : study_halls_conditions halls) : 
  halls.general + halls.biology + halls.chemistry + halls.math + halls.arts = 358 := by
  sorry

end total_students_l3838_383818


namespace matrix_power_10_l3838_383895

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 1, 1]

theorem matrix_power_10 : A ^ 10 = !![512, 512; 512, 512] := by
  sorry

end matrix_power_10_l3838_383895


namespace trace_equation_equiv_equal_distance_l3838_383874

/-- 
For any point P(x, y) in a 2D coordinate system, the trace equation y = |x| 
is equivalent to the condition that the distance from P to the x-axis 
is equal to the distance from P to the y-axis.
-/
theorem trace_equation_equiv_equal_distance (x y : ℝ) : 
  y = |x| ↔ |y| = |x| :=
sorry

end trace_equation_equiv_equal_distance_l3838_383874


namespace polynomial_value_l3838_383869

theorem polynomial_value (a b : ℝ) (h1 : a * b = 7) (h2 : a + b = 2) :
  a^2 * b + a * b^2 - 20 = -6 := by sorry

end polynomial_value_l3838_383869


namespace walter_gets_49_bananas_l3838_383843

/-- Calculates the number of bananas Walter gets when sharing with Jefferson -/
def walters_bananas (jeffersons_bananas : ℕ) : ℕ :=
  let walters_fewer := jeffersons_bananas / 4
  let walters_original := jeffersons_bananas - walters_fewer
  let total_bananas := jeffersons_bananas + walters_original
  total_bananas / 2

/-- Proves that Walter gets 49 bananas when sharing with Jefferson -/
theorem walter_gets_49_bananas :
  walters_bananas 56 = 49 := by
  sorry

end walter_gets_49_bananas_l3838_383843


namespace roses_in_vase_l3838_383812

theorem roses_in_vase (total_flowers : ℕ) (carnations : ℕ) (roses : ℕ) : 
  total_flowers = 10 → carnations = 5 → total_flowers = roses + carnations → roses = 5 := by
  sorry

end roses_in_vase_l3838_383812


namespace A_equals_B_l3838_383837

-- Define set A
def A : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 2*y^2}

-- Define set B
def B : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 6*x*y + 11*y^2}

-- Theorem statement
theorem A_equals_B : A = B := by sorry

end A_equals_B_l3838_383837


namespace marked_circles_alignment_l3838_383879

/-- Two identical circles, each marked with k arcs -/
structure MarkedCircle where
  k : ℕ
  arcs : Fin k → ℝ
  arc_measure : ∀ i, arcs i < 180 / (k^2 - k + 1)
  alignment : ∃ r : ℝ, ∀ i, ∃ j, arcs i = (fun x => (x + r) % 360) (arcs j)

/-- The theorem statement -/
theorem marked_circles_alignment (c1 c2 : MarkedCircle) (h : c1 = c2) :
  ∃ r : ℝ, ∀ i, ∀ j, c1.arcs i ≠ (fun x => (x + r) % 360) (c2.arcs j) := by
  sorry

end marked_circles_alignment_l3838_383879


namespace removed_triangles_area_l3838_383845

theorem removed_triangles_area (r s : ℝ) : 
  (r + s)^2 + (r - s)^2 = 16^2 → 
  2 * (r^2 + s^2) = 256 := by
  sorry

end removed_triangles_area_l3838_383845


namespace set_operations_l3838_383888

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 5}
def B : Set Nat := {3, 5, 6}

theorem set_operations :
  (A ∩ B = {3, 5}) ∧ ((U \ A) ∪ B = {3, 4, 5, 6}) := by
  sorry

end set_operations_l3838_383888


namespace geometric_sequence_sum_l3838_383842

/-- A geometric sequence with first term a and common ratio r -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r^(n-1)

theorem geometric_sequence_sum (a r : ℝ) (h1 : a < 0) :
  let seq := geometric_sequence a r
  (seq 2 * seq 4 + 2 * seq 3 * seq 5 + seq 4 * seq 6 = 36) →
  (seq 3 + seq 5 = -6) :=
by sorry

end geometric_sequence_sum_l3838_383842


namespace number_problem_l3838_383886

theorem number_problem (x : ℝ) : 0.4 * x = 0.2 * 650 + 190 ↔ x = 800 := by
  sorry

end number_problem_l3838_383886


namespace both_languages_students_l3838_383806

/-- The number of students taking both French and Spanish classes -/
def students_taking_both (french_class : ℕ) (spanish_class : ℕ) (total_students : ℕ) (students_one_language : ℕ) : ℕ :=
  french_class + spanish_class - total_students

theorem both_languages_students :
  let french_class : ℕ := 21
  let spanish_class : ℕ := 21
  let students_one_language : ℕ := 30
  let total_students : ℕ := students_one_language + students_taking_both french_class spanish_class total_students students_one_language
  students_taking_both french_class spanish_class total_students students_one_language = 6 := by
  sorry

end both_languages_students_l3838_383806


namespace selection_theorem_l3838_383846

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of books on the shelf -/
def total_books : ℕ := 10

/-- The number of books to be selected -/
def books_to_select : ℕ := 5

/-- The number of specific books that must be included -/
def specific_books : ℕ := 2

/-- The number of ways to select 5 books from 10 books, given that 2 specific books must always be included -/
def selection_ways : ℕ := binomial (total_books - specific_books) (books_to_select - specific_books)

theorem selection_theorem : selection_ways = 56 := by sorry

end selection_theorem_l3838_383846


namespace largest_divisor_of_n4_minus_n2_l3838_383807

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : ∃ (k : ℤ), n^4 - n^2 = 12 * k ∧ ∀ (m : ℤ), (∀ (n : ℤ), ∃ (l : ℤ), n^4 - n^2 = m * l) → m ≤ 12 := by
  sorry

end largest_divisor_of_n4_minus_n2_l3838_383807


namespace easter_egg_distribution_l3838_383863

theorem easter_egg_distribution (total_people : ℕ) (eggs_per_person : ℕ) (num_baskets : ℕ) :
  total_people = 20 →
  eggs_per_person = 9 →
  num_baskets = 15 →
  (total_people * eggs_per_person) / num_baskets = 12 := by
sorry

end easter_egg_distribution_l3838_383863


namespace decagon_diagonals_l3838_383841

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A decagon (10-sided polygon) has 35 diagonals -/
theorem decagon_diagonals :
  num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l3838_383841


namespace absolute_value_sum_l3838_383831

theorem absolute_value_sum (m n p : ℤ) 
  (h : |m - n|^3 + |p - m|^5 = 1) : 
  |p - m| + |m - n| + 2 * |n - p| = 3 := by sorry

end absolute_value_sum_l3838_383831


namespace excluded_angle_measure_l3838_383855

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180° -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: In a polygon where the sum of all interior angles except one is 1680°,
    the measure of the excluded interior angle is 120°. -/
theorem excluded_angle_measure (n : ℕ) (h : sum_interior_angles n - 120 = 1680) :
  120 = sum_interior_angles n - 1680 := by
  sorry

end excluded_angle_measure_l3838_383855


namespace common_ratio_is_three_l3838_383826

/-- An arithmetic-geometric sequence with its properties -/
structure ArithGeomSeq where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  q : ℝ      -- Common ratio
  h1 : a 3 = 2 * S 2 + 1
  h2 : a 4 = 2 * S 3 + 1
  h3 : ∀ n : ℕ, n ≥ 2 → a (n+1) = q * a n

/-- The common ratio of the arithmetic-geometric sequence is 3 -/
theorem common_ratio_is_three (seq : ArithGeomSeq) : seq.q = 3 := by
  sorry

end common_ratio_is_three_l3838_383826


namespace range_of_m_for_true_proposition_l3838_383858

theorem range_of_m_for_true_proposition (m : ℝ) :
  (∀ x : ℝ, 4^x - 2^(x + 1) + m = 0) →
  m ≤ 1 ∧ ∀ y : ℝ, y < m → ∃ x : ℝ, 4^x - 2^(x + 1) + y ≠ 0 :=
by sorry

end range_of_m_for_true_proposition_l3838_383858


namespace equation_solution_l3838_383880

theorem equation_solution : 
  ∃ (x : ℝ), 
    x ≠ (3/2) ∧ 
    (5 - 3*x = 1) ∧
    ((1 + 1/(1 + 1/(1 + 1/(2*x - 3)))) = 1/(x - 1)) ∧
    x = 4/3 :=
by sorry

end equation_solution_l3838_383880


namespace house_of_cards_layers_l3838_383836

/-- Calculates the maximum number of layers in a house of cards --/
def maxLayers (decks : ℕ) (cardsPerDeck : ℕ) (cardsPerLayer : ℕ) : ℕ :=
  (decks * cardsPerDeck) / cardsPerLayer

/-- Theorem: Given 16 decks of 52 cards each, using 26 cards per layer,
    the maximum number of layers in a house of cards is 32 --/
theorem house_of_cards_layers :
  maxLayers 16 52 26 = 32 := by
  sorry

end house_of_cards_layers_l3838_383836


namespace basketball_points_distribution_l3838_383896

theorem basketball_points_distribution (x : ℝ) (y : ℕ) : 
  (1/3 : ℝ) * x + (3/8 : ℝ) * x + 18 + y = x →
  y ≤ 15 →
  (∀ i ∈ Finset.range 5, (y : ℝ) / 5 ≤ 3) →
  y = 14 :=
by sorry

end basketball_points_distribution_l3838_383896


namespace counterexample_exists_l3838_383815

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem counterexample_exists : ∃ n : ℕ, 
  ¬(is_prime n) ∧ ¬(is_prime (n - 5)) ∧ n = 20 := by
  sorry

end counterexample_exists_l3838_383815


namespace decimal_to_binary_111_octal_to_decimal_77_l3838_383839

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : ℕ) : List Bool := sorry

/-- Converts an octal number to its decimal representation -/
def octalToDecimal (n : ℕ) : ℕ := sorry

theorem decimal_to_binary_111 :
  decimalToBinary 111 = [true, true, false, true, true, true, true] := by sorry

theorem octal_to_decimal_77 :
  octalToDecimal 77 = 63 := by sorry

end decimal_to_binary_111_octal_to_decimal_77_l3838_383839


namespace house_sale_profit_l3838_383822

theorem house_sale_profit (initial_value : ℝ) (first_sale_profit_percent : ℝ) (second_sale_loss_percent : ℝ) : 
  initial_value = 200000 ∧ 
  first_sale_profit_percent = 15 ∧ 
  second_sale_loss_percent = 20 → 
  (initial_value * (1 + first_sale_profit_percent / 100)) * (1 - second_sale_loss_percent / 100) - initial_value = 46000 :=
by sorry

end house_sale_profit_l3838_383822


namespace maximize_expected_score_l3838_383887

structure QuestionType where
  correct_prob : ℝ
  points : ℕ

def expected_score (first second : QuestionType) : ℝ :=
  first.correct_prob * (first.points + second.correct_prob * second.points) +
  first.correct_prob * (1 - second.correct_prob) * first.points

theorem maximize_expected_score (type_a type_b : QuestionType)
  (ha : type_a.correct_prob = 0.8)
  (hb : type_b.correct_prob = 0.6)
  (pa : type_a.points = 20)
  (pb : type_b.points = 80) :
  expected_score type_b type_a > expected_score type_a type_b :=
sorry

end maximize_expected_score_l3838_383887


namespace complex_geometric_sequence_l3838_383873

theorem complex_geometric_sequence (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 2*a + 2*Complex.I
  let z₃ : ℂ := 3*a + 4*Complex.I
  (∃ r : ℝ, r > 0 ∧ Complex.abs z₂ = r * Complex.abs z₁ ∧ Complex.abs z₃ = r * Complex.abs z₂) →
  a = 0 := by
sorry

end complex_geometric_sequence_l3838_383873


namespace unique_solution_for_exponential_equation_l3838_383814

theorem unique_solution_for_exponential_equation :
  ∀ a b p : ℕ+,
  p.val.Prime →
  (2 : ℕ)^(a : ℕ) + (p : ℕ)^(b : ℕ) = 19^(a : ℕ) →
  a = 1 ∧ b = 1 ∧ p = 17 :=
by sorry

end unique_solution_for_exponential_equation_l3838_383814


namespace fraction_equality_l3838_383872

theorem fraction_equality : (1-2+4-8+16-32+64)/(2-4+8-16+32-64+128) = 1/2 := by
  sorry

end fraction_equality_l3838_383872


namespace min_value_problem_l3838_383866

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1/a + 2) * (1/b + 2) ≥ 16 ∧
  ((1/a + 2) * (1/b + 2) = 16 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end min_value_problem_l3838_383866


namespace percentage_problem_l3838_383884

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 50 → 
  (0.6 * N) = ((P / 100) * 10 + 27) → 
  P = 30 := by
sorry

end percentage_problem_l3838_383884


namespace paula_shopping_theorem_l3838_383827

/-- Calculates the remaining money after Paula's shopping trip -/
def remaining_money (initial_amount : ℕ) (num_shirts : ℕ) (shirt_price : ℕ) 
  (num_pants : ℕ) (pants_price : ℕ) : ℕ :=
  initial_amount - (num_shirts * shirt_price + num_pants * pants_price)

/-- Proves that Paula has $100 left after her shopping trip -/
theorem paula_shopping_theorem :
  remaining_money 250 5 15 3 25 = 100 := by
  sorry

end paula_shopping_theorem_l3838_383827


namespace acute_angles_equal_l3838_383808

/-- A circle with a rhombus and an isosceles trapezoid inscribed around it -/
structure InscribedFigures where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Acute angle of the rhombus -/
  α : ℝ
  /-- Acute angle of the isosceles trapezoid -/
  β : ℝ
  /-- The rhombus and trapezoid are inscribed around the same circle -/
  inscribed : r > 0
  /-- The areas of the rhombus and trapezoid are equal -/
  equal_areas : (4 * r^2) / Real.sin α = (4 * r^2) / Real.sin β

/-- 
Given a rhombus and an isosceles trapezoid inscribed around the same circle with equal areas,
their acute angles are equal.
-/
theorem acute_angles_equal (fig : InscribedFigures) : fig.α = fig.β :=
  sorry

end acute_angles_equal_l3838_383808


namespace bathroom_width_l3838_383885

/-- Proves that the width of Mrs. Garvey's bathroom is 6 feet -/
theorem bathroom_width : 
  ∀ (length width : ℝ) (tile_side : ℝ) (num_tiles : ℕ),
  length = 10 →
  tile_side = 0.5 →
  num_tiles = 240 →
  width * length = (tile_side^2 * num_tiles) →
  width = 6 :=
by
  sorry

end bathroom_width_l3838_383885


namespace simplify_trig_expression_l3838_383838

theorem simplify_trig_expression (α : Real) 
  (h : -3 * Real.pi < α ∧ α < -(5/2) * Real.pi) : 
  Real.sqrt ((1 + Real.cos (α - 2018 * Real.pi)) / 2) = -Real.cos (α / 2) := by
  sorry

end simplify_trig_expression_l3838_383838


namespace sufficient_not_necessary_l3838_383844

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x^2 - 2*x < 0 → 0 < x ∧ x < 4) ∧ 
  (∃ x : ℝ, 0 < x ∧ x < 4 ∧ x^2 - 2*x ≥ 0) := by
  sorry

end sufficient_not_necessary_l3838_383844


namespace rational_function_value_l3838_383832

-- Define f as a function from ℚ to ℚ (rational numbers)
variable (f : ℚ → ℚ)

-- State the main theorem
theorem rational_function_value : 
  (∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + 3 * f x / x = 2 * x^2) →
  f (-3) = 494 / 117 := by
  sorry

end rational_function_value_l3838_383832


namespace penguin_count_l3838_383861

theorem penguin_count (zebras tigers zookeepers : ℕ) 
  (h1 : zebras = 22)
  (h2 : tigers = 8)
  (h3 : zookeepers = 12)
  (h4 : ∀ (penguins : ℕ), 
    (penguins + zebras + tigers + zookeepers) + 132 = 
    4 * penguins + 4 * zebras + 4 * tigers + 2 * zookeepers) :
  ∃ (penguins : ℕ), penguins = 10 := by
sorry

end penguin_count_l3838_383861


namespace saree_discount_problem_l3838_383820

/-- Proves that given a saree with an original price of 600, after a 20% discount
    and a second discount resulting in a final price of 456, the second discount percentage is 5% -/
theorem saree_discount_problem (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ)
    (h1 : original_price = 600)
    (h2 : first_discount = 20)
    (h3 : final_price = 456) :
    let price_after_first_discount := original_price * (1 - first_discount / 100)
    let second_discount_amount := price_after_first_discount - final_price
    let second_discount_percentage := (second_discount_amount / price_after_first_discount) * 100
    second_discount_percentage = 5 := by
  sorry

end saree_discount_problem_l3838_383820


namespace tangent_iff_k_eq_zero_l3838_383859

/-- A line with equation x - ky - 1 = 0 -/
structure Line (k : ℝ) where
  equation : ∀ x y : ℝ, x - k * y - 1 = 0

/-- A circle with center (2,1) and radius 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}

/-- The line is tangent to the circle -/
def IsTangent (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, p ∈ Circle ∧ p.1 - k * p.2 - 1 = 0

/-- The main theorem: k = 0 is necessary and sufficient for the line to be tangent to the circle -/
theorem tangent_iff_k_eq_zero (k : ℝ) : IsTangent k ↔ k = 0 := by
  sorry

end tangent_iff_k_eq_zero_l3838_383859


namespace largest_square_from_rectangle_l3838_383894

theorem largest_square_from_rectangle (width length : ℕ) 
  (h_width : width = 63) (h_length : length = 42) :
  Nat.gcd width length = 21 := by
  sorry

end largest_square_from_rectangle_l3838_383894


namespace survey_order_correct_l3838_383856

-- Define the steps of the survey process
inductive SurveyStep
  | CollectData
  | OrganizeData
  | DrawPieChart
  | AnalyzeData

-- Define a function to represent the correct order of steps
def correctOrder : List SurveyStep :=
  [SurveyStep.CollectData, SurveyStep.OrganizeData, SurveyStep.DrawPieChart, SurveyStep.AnalyzeData]

-- Define a function to check if a given order is correct
def isCorrectOrder (order : List SurveyStep) : Prop :=
  order = correctOrder

-- Theorem stating that the given order is correct
theorem survey_order_correct :
  isCorrectOrder [SurveyStep.CollectData, SurveyStep.OrganizeData, SurveyStep.DrawPieChart, SurveyStep.AnalyzeData] :=
by sorry

end survey_order_correct_l3838_383856


namespace cos_alpha_minus_beta_l3838_383835

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : 2 * Real.cos α - Real.cos β = 3/2)
  (h2 : 2 * Real.sin α - Real.sin β = 2) :
  Real.cos (α - β) = -5/16 := by
  sorry

end cos_alpha_minus_beta_l3838_383835


namespace square_root_of_1708249_l3838_383882

theorem square_root_of_1708249 :
  Real.sqrt 1708249 = 1307 := by sorry

end square_root_of_1708249_l3838_383882
