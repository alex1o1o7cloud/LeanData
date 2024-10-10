import Mathlib

namespace count_integers_in_range_l2844_284415

theorem count_integers_in_range : ∃ (S : Finset Int), 
  (∀ n : Int, n ∈ S ↔ 15 < n^2 ∧ n^2 < 120) ∧ Finset.card S = 14 := by
  sorry

end count_integers_in_range_l2844_284415


namespace susie_earnings_l2844_284483

/-- Calculates the total earnings from selling pizza slices and whole pizzas --/
def calculate_earnings (price_per_slice : ℚ) (price_per_whole : ℚ) (slices_sold : ℕ) (whole_sold : ℕ) : ℚ :=
  price_per_slice * slices_sold + price_per_whole * whole_sold

/-- Proves that Susie's earnings are $117 given the specified prices and sales --/
theorem susie_earnings : 
  let price_per_slice : ℚ := 3
  let price_per_whole : ℚ := 15
  let slices_sold : ℕ := 24
  let whole_sold : ℕ := 3
  calculate_earnings price_per_slice price_per_whole slices_sold whole_sold = 117 := by
  sorry

end susie_earnings_l2844_284483


namespace complex_equation_solution_l2844_284441

theorem complex_equation_solution (a b : ℝ) (h : b ≠ 0) :
  (Complex.I : ℂ)^2 = -1 →
  (a + b * Complex.I)^2 = -b * Complex.I →
  (a = -1/2 ∧ (b = 1/2 ∨ b = -1/2)) := by
  sorry

end complex_equation_solution_l2844_284441


namespace least_positive_integer_with_remainders_l2844_284423

theorem least_positive_integer_with_remainders (N : ℕ) : 
  (N % 7 = 5) ∧ 
  (N % 8 = 6) ∧ 
  (N % 9 = 7) ∧ 
  (N % 10 = 8) ∧ 
  (∀ m : ℕ, m < N → 
    (m % 7 ≠ 5) ∨ 
    (m % 8 ≠ 6) ∨ 
    (m % 9 ≠ 7) ∨ 
    (m % 10 ≠ 8)) → 
  N = 2518 := by
sorry

end least_positive_integer_with_remainders_l2844_284423


namespace seed_germination_percentage_l2844_284414

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 25 / 100 →
  germination_rate2 = 30 / 100 →
  let total_seeds := seeds_plot1 + seeds_plot2
  let germinated_seeds1 := (seeds_plot1 : ℚ) * germination_rate1
  let germinated_seeds2 := (seeds_plot2 : ℚ) * germination_rate2
  let total_germinated := germinated_seeds1 + germinated_seeds2
  (total_germinated / total_seeds) * 100 = 27 := by
sorry

end seed_germination_percentage_l2844_284414


namespace largest_two_digit_prime_factor_l2844_284467

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), is_prime p ∧ 
             p ≥ 10 ∧ p < 100 ∧
             p ∣ binomial_coefficient 300 150 ∧
             ∀ (q : ℕ), is_prime q → q ≥ 10 → q < 100 → q ∣ binomial_coefficient 300 150 → q ≤ p :=
by sorry

end largest_two_digit_prime_factor_l2844_284467


namespace cookies_milk_proportion_l2844_284434

/-- Given that 24 cookies require 5 quarts of milk and 1 quart equals 4 cups,
    prove that 8 cookies require 20/3 cups of milk. -/
theorem cookies_milk_proportion :
  let cookies_24 : ℕ := 24
  let quarts_24 : ℕ := 5
  let cups_per_quart : ℕ := 4
  let cookies_8 : ℕ := 8
  cookies_24 * (cups_per_quart * quarts_24) / cookies_8 = 20 / 3 := by sorry

end cookies_milk_proportion_l2844_284434


namespace f_satisfies_conditions_l2844_284461

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions : 
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by
  sorry

#check f_satisfies_conditions

end f_satisfies_conditions_l2844_284461


namespace simultaneous_inequalities_condition_l2844_284468

theorem simultaneous_inequalities_condition (a b : ℝ) :
  (a > b ∧ 1 / a > 1 / b) ↔ (a > 0 ∧ 0 > b) :=
by sorry

end simultaneous_inequalities_condition_l2844_284468


namespace most_accurate_approximation_l2844_284490

def reading_lower_bound : ℝ := 10.65
def reading_upper_bound : ℝ := 10.85
def major_tick_interval : ℝ := 0.1

def options : List ℝ := [10.68, 10.72, 10.74, 10.75]

theorem most_accurate_approximation :
  ∃ (x : ℝ), 
    reading_lower_bound ≤ x ∧ 
    x ≤ reading_upper_bound ∧ 
    (∀ y ∈ options, |x - 10.75| ≤ |x - y|) :=
by sorry

end most_accurate_approximation_l2844_284490


namespace women_per_table_l2844_284458

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 5 →
  men_per_table = 3 →
  total_customers = 40 →
  ∃ (women_per_table : ℕ),
    women_per_table * num_tables + men_per_table * num_tables = total_customers ∧
    women_per_table = 5 :=
by sorry

end women_per_table_l2844_284458


namespace power_inequality_set_l2844_284473

theorem power_inequality_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | a^(x+3) > a^(2*x)} = {x : ℝ | x > 3} := by sorry

end power_inequality_set_l2844_284473


namespace range_of_a_l2844_284436

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 4

-- State the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x ∈ Set.Icc (a - 2) (a^2), f a x ∈ Set.Icc (-4) 0) →
  a ∈ Set.Icc 1 2 :=
by sorry

end range_of_a_l2844_284436


namespace garys_money_l2844_284427

/-- Gary's initial amount of money -/
def initial_amount : ℕ := sorry

/-- Amount Gary spent on the snake -/
def spent_amount : ℕ := 55

/-- Amount Gary had left after buying the snake -/
def remaining_amount : ℕ := 18

/-- Theorem stating that Gary's initial amount equals the sum of spent and remaining amounts -/
theorem garys_money : initial_amount = spent_amount + remaining_amount := by sorry

end garys_money_l2844_284427


namespace max_value_trig_function_l2844_284405

theorem max_value_trig_function :
  ∃ M : ℝ, M = -1/2 ∧
  (∀ x : ℝ, 2 * Real.sin x ^ 2 + 2 * Real.cos x - 3 ≤ M) ∧
  ∀ ε > 0, ∃ x : ℝ, 2 * Real.sin x ^ 2 + 2 * Real.cos x - 3 > M - ε :=
by sorry

end max_value_trig_function_l2844_284405


namespace model_b_piano_keys_l2844_284450

theorem model_b_piano_keys : ∃ (x : ℕ), 
  (104 : ℕ) = 2 * x - 72 → x = 88 := by
  sorry

end model_b_piano_keys_l2844_284450


namespace max_curved_sides_l2844_284407

/-- A figure formed by the intersection of circles -/
structure IntersectionFigure where
  n : ℕ
  n_ge_two : n ≥ 2

/-- The number of curved sides in an intersection figure -/
def curved_sides (F : IntersectionFigure) : ℕ := 2 * F.n - 2

/-- The theorem stating the maximum number of curved sides -/
theorem max_curved_sides (F : IntersectionFigure) :
  curved_sides F ≤ 2 * F.n - 2 :=
sorry

end max_curved_sides_l2844_284407


namespace abs_ratio_sqrt_five_halves_l2844_284454

theorem abs_ratio_sqrt_five_halves (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^2 = 18*a*b) : 
  |((a+b)/(a-b))| = Real.sqrt 5 / 2 := by
sorry

end abs_ratio_sqrt_five_halves_l2844_284454


namespace tangent_line_cubic_l2844_284433

/-- Given a cubic function f(x) = ax³ + x + 1, prove that if its tangent line at 
    (1, f(1)) passes through (2, 7), then a = 1. -/
theorem tangent_line_cubic (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + x + 1
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 1
  let tangent_slope : ℝ := f' 1
  let point_on_curve : ℝ := f 1
  (point_on_curve - 7) / (1 - 2) = tangent_slope → a = 1 := by
  sorry

end tangent_line_cubic_l2844_284433


namespace arccos_gt_arctan_iff_l2844_284422

-- Define the approximate value of the upper bound
def upperBound : ℝ := 0.54

-- State the theorem
theorem arccos_gt_arctan_iff (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 →
  Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1 : ℝ) upperBound :=
by sorry

end arccos_gt_arctan_iff_l2844_284422


namespace contrapositive_example_l2844_284439

theorem contrapositive_example : 
  (∀ x : ℝ, x > 2 → x^2 > 4) ↔ (∀ x : ℝ, x^2 ≤ 4 → x ≤ 2) := by
sorry

end contrapositive_example_l2844_284439


namespace complex_reciprocal_sum_magnitude_l2844_284456

theorem complex_reciprocal_sum_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
sorry

end complex_reciprocal_sum_magnitude_l2844_284456


namespace cafeteria_milk_cartons_l2844_284452

/-- Given a number of full stacks of milk cartons and the number of cartons per stack,
    calculate the total number of milk cartons. -/
def totalCartons (numStacks : ℕ) (cartonsPerStack : ℕ) : ℕ :=
  numStacks * cartonsPerStack

/-- Theorem stating that 133 full stacks of 6 milk cartons each result in 798 total cartons. -/
theorem cafeteria_milk_cartons :
  totalCartons 133 6 = 798 := by
  sorry

end cafeteria_milk_cartons_l2844_284452


namespace hill_climb_time_l2844_284416

/-- Proves that the time taken to reach the top of the hill is 4 hours -/
theorem hill_climb_time (descent_time : ℝ) (avg_speed_total : ℝ) (avg_speed_climb : ℝ) :
  descent_time = 2 →
  avg_speed_total = 2 →
  avg_speed_climb = 1.5 →
  let ascent_time := 4
  let total_time := ascent_time + descent_time
  let total_distance := avg_speed_total * total_time
  let climb_distance := avg_speed_climb * ascent_time
  climb_distance * 2 = total_distance →
  ascent_time = 4 := by
  sorry

end hill_climb_time_l2844_284416


namespace gcd_50404_40303_l2844_284492

theorem gcd_50404_40303 : Nat.gcd 50404 40303 = 3 := by
  sorry

end gcd_50404_40303_l2844_284492


namespace donut_hole_count_donut_hole_count_proof_l2844_284499

/-- The number of donut holes Nira will have coated when all three workers finish simultaneously -/
theorem donut_hole_count : ℕ :=
  let nira_radius : ℝ := 5
  let theo_radius : ℝ := 7
  let kaira_side : ℝ := 6
  let nira_surface_area : ℝ := 4 * Real.pi * nira_radius ^ 2
  let theo_surface_area : ℝ := 4 * Real.pi * theo_radius ^ 2
  let kaira_surface_area : ℝ := 6 * kaira_side ^ 2
  5292

/-- Proof that Nira will have coated 5292 donut holes when all three workers finish simultaneously -/
theorem donut_hole_count_proof : donut_hole_count = 5292 := by
  sorry

end donut_hole_count_donut_hole_count_proof_l2844_284499


namespace average_age_proof_l2844_284477

def luke_age : ℕ := 20
def years_future : ℕ := 8

theorem average_age_proof :
  let bernard_future_age := 3 * luke_age
  let bernard_current_age := bernard_future_age - years_future
  let average_age := (luke_age + bernard_current_age) / 2
  average_age = 36 := by sorry

end average_age_proof_l2844_284477


namespace complete_square_sum_l2844_284400

theorem complete_square_sum (x : ℝ) :
  ∃ (a b c : ℤ), 
    a > 0 ∧
    (25 * x^2 + 30 * x - 55 = 0 ↔ (a * x + b)^2 = c) ∧
    a + b + c = -38 := by
  sorry

end complete_square_sum_l2844_284400


namespace log_and_perpendicular_lines_l2844_284451

theorem log_and_perpendicular_lines (S T : ℝ) : 
  (Real.log S / Real.log 9 = 3/2) →
  ((1 : ℝ) * ((-S : ℝ)) + 5 * T = 0) →
  (S = 27 ∧ T = 135) := by sorry

end log_and_perpendicular_lines_l2844_284451


namespace square_side_length_l2844_284479

/-- Given a square ABCD with specific points and conditions, prove its side length is 10 -/
theorem square_side_length (A B C D P Q R S Z : ℝ × ℝ) : 
  (∃ s : ℝ, 
    -- Square ABCD
    A = (0, 0) ∧ B = (s, 0) ∧ C = (s, s) ∧ D = (0, s) ∧
    -- P on AB, Q on BC, R on CD, S on DA
    (∃ t₁ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ P = (t₁ * s, 0)) ∧
    (∃ t₂ : ℝ, 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ Q = (s, (1 - t₂) * s)) ∧
    (∃ t₃ : ℝ, 0 ≤ t₃ ∧ t₃ ≤ 1 ∧ R = ((1 - t₃) * s, s)) ∧
    (∃ t₄ : ℝ, 0 ≤ t₄ ∧ t₄ ≤ 1 ∧ S = (0, t₄ * s)) ∧
    -- PR parallel to BC, SQ parallel to AB
    (R.1 - P.1) * (C.2 - B.2) = (R.2 - P.2) * (C.1 - B.1) ∧
    (Q.1 - S.1) * (B.2 - A.2) = (Q.2 - S.2) * (B.1 - A.1) ∧
    -- Z is intersection of PR and SQ
    (Z.1 - P.1) * (R.2 - P.2) = (Z.2 - P.2) * (R.1 - P.1) ∧
    (Z.1 - S.1) * (Q.2 - S.2) = (Z.2 - S.2) * (Q.1 - S.1) ∧
    -- Given distances
    ‖B - P‖ = 7 ∧
    ‖B - Q‖ = 6 ∧
    ‖D - Z‖ = 5) →
  s = 10 := by
sorry

end square_side_length_l2844_284479


namespace f_value_at_2_l2844_284474

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 2) : f a b 2 = -10 := by
  sorry

end f_value_at_2_l2844_284474


namespace number_remainder_l2844_284401

theorem number_remainder (A : ℤ) (h : 9 * A + 1 = 10 * A - 100) : A % 7 = 3 := by
  sorry

end number_remainder_l2844_284401


namespace sum_of_digits_2010_5012_6_l2844_284424

def digit_sum (n : ℕ) : ℕ := sorry

theorem sum_of_digits_2010_5012_6 :
  digit_sum (2^2010 * 5^2012 * 6) = 6 := by sorry

end sum_of_digits_2010_5012_6_l2844_284424


namespace bird_migration_problem_l2844_284411

theorem bird_migration_problem (distance_jim_disney : ℕ) (distance_disney_london : ℕ) (total_distance : ℕ) :
  distance_jim_disney = 50 →
  distance_disney_london = 60 →
  total_distance = 2200 →
  ∃ (num_birds : ℕ), num_birds * (distance_jim_disney + distance_disney_london) = total_distance ∧ num_birds = 20 :=
by
  sorry

end bird_migration_problem_l2844_284411


namespace ball_bounce_distance_l2844_284475

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFraction : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The problem statement -/
theorem ball_bounce_distance :
  totalDistance 150 (3/4) 4 = 765.234375 :=
sorry

end ball_bounce_distance_l2844_284475


namespace axis_of_symmetry_sin_l2844_284476

theorem axis_of_symmetry_sin (x : ℝ) : 
  x = π / 12 → 
  ∃ k : ℤ, 2 * x + π / 3 = π / 2 + k * π :=
by sorry

end axis_of_symmetry_sin_l2844_284476


namespace equation_with_multiple_solutions_l2844_284469

theorem equation_with_multiple_solutions (a b : ℝ) :
  (∀ x y : ℝ, x ≠ y → a * x + (b - 3) = (5 * a - 1) * x + 3 * b ∧
                     a * y + (b - 3) = (5 * a - 1) * y + 3 * b) →
  100 * a + 4 * b = 31 := by
sorry

end equation_with_multiple_solutions_l2844_284469


namespace divisibility_property_l2844_284432

theorem divisibility_property (a b c : ℤ) (h : a + b + c = 0) :
  (∃ k : ℤ, a^4 + b^4 + c^4 = k * (a^2 + b^2 + c^2)) ∧
  (∃ m : ℤ, a^100 + b^100 + c^100 = m * (a^2 + b^2 + c^2)) := by
  sorry

end divisibility_property_l2844_284432


namespace quadratic_solution_sum_l2844_284462

theorem quadratic_solution_sum (m n : ℝ) (h1 : m ≠ 0) :
  m * 1^2 + n * 1 - 1 = 0 → m + n = 1 := by
  sorry

end quadratic_solution_sum_l2844_284462


namespace boys_girls_points_not_equal_l2844_284429

/-- Represents a round-robin chess tournament with boys and girls -/
structure ChessTournament where
  num_boys : Nat
  num_girls : Nat

/-- Calculate the total number of games in a round-robin tournament -/
def total_games (t : ChessTournament) : Nat :=
  (t.num_boys + t.num_girls) * (t.num_boys + t.num_girls - 1) / 2

/-- Calculate the number of games between boys -/
def boys_games (t : ChessTournament) : Nat :=
  t.num_boys * (t.num_boys - 1) / 2

/-- Calculate the number of games between girls -/
def girls_games (t : ChessTournament) : Nat :=
  t.num_girls * (t.num_girls - 1) / 2

/-- Calculate the number of games between boys and girls -/
def mixed_games (t : ChessTournament) : Nat :=
  t.num_boys * t.num_girls

/-- Theorem: In a round-robin chess tournament with 9 boys and 3 girls,
    the total points scored by all boys cannot equal the total points scored by all girls -/
theorem boys_girls_points_not_equal (t : ChessTournament) 
        (h1 : t.num_boys = 9) 
        (h2 : t.num_girls = 3) : 
        ¬ (boys_games t + mixed_games t / 2 = girls_games t + mixed_games t / 2) := by
  sorry

#eval boys_games ⟨9, 3⟩
#eval girls_games ⟨9, 3⟩
#eval mixed_games ⟨9, 3⟩

end boys_girls_points_not_equal_l2844_284429


namespace least_common_multiple_9_6_l2844_284487

theorem least_common_multiple_9_6 : Nat.lcm 9 6 = 18 := by
  sorry

end least_common_multiple_9_6_l2844_284487


namespace simultaneous_arrival_l2844_284485

theorem simultaneous_arrival (total_distance : ℝ) (alyosha_walk_speed alyosha_cycle_speed vitia_walk_speed vitia_cycle_speed : ℝ) 
  (h1 : total_distance = 20)
  (h2 : alyosha_walk_speed = 4)
  (h3 : alyosha_cycle_speed = 15)
  (h4 : vitia_walk_speed = 5)
  (h5 : vitia_cycle_speed = 20)
  : ∃ x : ℝ, 
    x / alyosha_cycle_speed + (total_distance - x) / alyosha_walk_speed = 
    x / vitia_walk_speed + (total_distance - x) / vitia_cycle_speed ∧ 
    x = 12 := by
  sorry

end simultaneous_arrival_l2844_284485


namespace cos_960_degrees_l2844_284457

theorem cos_960_degrees : Real.cos (960 * π / 180) = -1/2 := by
  sorry

end cos_960_degrees_l2844_284457


namespace sin_sum_identity_l2844_284420

theorem sin_sum_identity (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 3) :
  Real.sin (x - 5 * π / 6) + Real.sin (π / 3 - x) ^ 2 = 5 / 9 := by
  sorry

end sin_sum_identity_l2844_284420


namespace ellipse_properties_l2844_284412

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  f1 : ℝ × ℝ  -- Focus 1
  f2 : ℝ × ℝ  -- Focus 2
  p : ℝ × ℝ   -- Point on ellipse
  h1 : a > b
  h2 : b > 0
  h3 : (p.1^2 / a^2) + (p.2^2 / b^2) = 1  -- P is on the ellipse
  h4 : (p.1 - f1.1) * (p.1 - f2.1) + (p.2 - f1.2) * (p.2 - f2.2) = 0  -- PF₁ ⟂ PF₂
  h5 : (f1.1 - f2.1)^2 + (f1.2 - f2.2)^2 = 12  -- |F₁F₂| = 2√3
  h6 : abs ((p.1 - f1.1) * (p.2 - f2.2) - (p.2 - f1.2) * (p.1 - f2.1)) = 2  -- Area of triangle PF₁F₂ is 1

/-- The theorem to be proved -/
theorem ellipse_properties (e : Ellipse) :
  (e.a = 2 ∧ e.b = 1) ∧
  (∀ m : ℝ, ∃ A B : ℝ × ℝ,
    (A.1^2 / 4 + A.2^2 = 1) ∧
    (B.1^2 / 4 + B.2^2 = 1) ∧
    (A.2 + B.2 = A.1 + B.1 + 2*m) ↔
    -3 * Real.sqrt 5 / 5 < m ∧ m < 3 * Real.sqrt 5 / 5) :=
sorry

end ellipse_properties_l2844_284412


namespace clara_cookie_sales_l2844_284466

/-- Represents the number of cookies in each type of box --/
structure CookieBox where
  type1 : Nat
  type2 : Nat
  type3 : Nat

/-- Represents the number of boxes sold for each type --/
structure BoxesSold where
  type1 : Nat
  type2 : Nat
  type3 : Nat

/-- Calculates the total number of cookies sold --/
def totalCookiesSold (c : CookieBox) (b : BoxesSold) : Nat :=
  c.type1 * b.type1 + c.type2 * b.type2 + c.type3 * b.type3

theorem clara_cookie_sales (c : CookieBox) (b : BoxesSold) 
    (h1 : c.type1 = 12)
    (h2 : c.type2 = 20)
    (h3 : c.type3 = 16)
    (h4 : b.type2 = 80)
    (h5 : b.type3 = 70)
    (h6 : totalCookiesSold c b = 3320) :
    b.type1 = 50 := by
  sorry

end clara_cookie_sales_l2844_284466


namespace pinocchio_problem_l2844_284428

theorem pinocchio_problem (x : ℕ) : 
  x ≠ 0 ∧ x < 10 ∧ (x + x + 1) * x = 111 * x → x = 5 :=
by sorry

end pinocchio_problem_l2844_284428


namespace abs_not_positive_iff_eq_l2844_284470

theorem abs_not_positive_iff_eq (y : ℚ) : ¬(0 < |5*y - 8|) ↔ y = 8/5 := by sorry

end abs_not_positive_iff_eq_l2844_284470


namespace circle_center_coordinate_sum_l2844_284480

/-- Given a circle with diameter endpoints (10, -6) and (-6, 2), 
    the sum of the coordinates of its center is 0. -/
theorem circle_center_coordinate_sum : 
  let x1 : ℝ := 10
  let y1 : ℝ := -6
  let x2 : ℝ := -6
  let y2 : ℝ := 2
  let center_x : ℝ := (x1 + x2) / 2
  let center_y : ℝ := (y1 + y2) / 2
  center_x + center_y = 0 := by sorry

end circle_center_coordinate_sum_l2844_284480


namespace actual_distance_traveled_l2844_284438

/-- Given a person walking at two different speeds, prove the actual distance traveled -/
theorem actual_distance_traveled 
  (initial_speed : ℝ) 
  (increased_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : initial_speed = 10) 
  (h2 : increased_speed = 15) 
  (h3 : additional_distance = 15) 
  (h4 : ∃ t : ℝ, increased_speed * t = initial_speed * t + additional_distance) : 
  ∃ d : ℝ, d = 30 ∧ d = initial_speed * (additional_distance / (increased_speed - initial_speed)) :=
by sorry

end actual_distance_traveled_l2844_284438


namespace expand_product_l2844_284493

theorem expand_product (x a : ℝ) : 2 * (x + (a + 2)) * (x + (a - 3)) = 2 * x^2 + (4 * a - 2) * x + 2 * a^2 - 2 * a - 12 := by
  sorry

end expand_product_l2844_284493


namespace malcolm_followers_l2844_284406

def total_followers (instagram : ℕ) (facebook : ℕ) : ℕ :=
  let twitter := (instagram + facebook) / 2
  let tiktok := 3 * twitter
  let youtube := tiktok + 510
  instagram + facebook + twitter + tiktok + youtube

theorem malcolm_followers : total_followers 240 500 = 3840 := by
  sorry

end malcolm_followers_l2844_284406


namespace systematic_sample_theorem_l2844_284409

/-- Represents a systematic sample from a class -/
structure SystematicSample where
  totalStudents : ℕ
  sampleSize : ℕ
  firstSample : ℕ
  knownSamples : Finset ℕ

/-- Checks if a number is part of the systematic sample -/
def isInSample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.firstSample + k * (s.totalStudents / s.sampleSize)

theorem systematic_sample_theorem (s : SystematicSample)
  (h1 : s.totalStudents = 48)
  (h2 : s.sampleSize = 6)
  (h3 : s.firstSample = 5)
  (h4 : s.knownSamples = {5, 21, 29, 37, 45})
  (h5 : ∀ n ∈ s.knownSamples, isInSample s n) :
  isInSample s 13 ∧ (∀ n, isInSample s n → n = 13 ∨ n ∈ s.knownSamples) :=
sorry

end systematic_sample_theorem_l2844_284409


namespace book_purchase_savings_l2844_284481

theorem book_purchase_savings (full_price_book1 full_price_book2 : ℝ) : 
  full_price_book1 = 33 →
  full_price_book2 > 0 →
  let total_paid := full_price_book1 + (full_price_book2 / 2)
  let full_price := full_price_book1 + full_price_book2
  let savings_ratio := (full_price - total_paid) / full_price
  savings_ratio = 1/5 →
  full_price - total_paid = 11 :=
by sorry

end book_purchase_savings_l2844_284481


namespace unpainted_cubes_in_64_cube_l2844_284463

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  side_count : ℕ
  total_cubes : ℕ
  inner_side_count : ℕ

/-- The number of small cubes with no painted faces in a cut cube -/
def unpainted_cubes (c : CutCube) : ℕ :=
  c.inner_side_count ^ 3

/-- Theorem: In a cube cut into 64 equal smaller cubes, 
    the number of small cubes with no painted faces is 8 -/
theorem unpainted_cubes_in_64_cube :
  ∃ c : CutCube, c.side_count = 4 ∧ c.total_cubes = 64 ∧ c.inner_side_count = 2 ∧ 
  unpainted_cubes c = 8 :=
sorry

end unpainted_cubes_in_64_cube_l2844_284463


namespace no_linear_term_implies_equal_coefficients_l2844_284413

theorem no_linear_term_implies_equal_coefficients (x m n : ℝ) : 
  (x + m) * (x - n) = x^2 + (-m * n) → m = n :=
by sorry

end no_linear_term_implies_equal_coefficients_l2844_284413


namespace previous_day_visitors_l2844_284486

def total_visitors : ℕ := 406
def current_day_visitors : ℕ := 132

theorem previous_day_visitors : 
  total_visitors - current_day_visitors = 274 := by
  sorry

end previous_day_visitors_l2844_284486


namespace annes_distance_is_six_l2844_284482

/-- The distance traveled by Anne given her walking time and speed -/
def annes_distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Theorem stating that Anne's distance traveled is 6 miles -/
theorem annes_distance_is_six : annes_distance 3 2 = 6 := by
  sorry

end annes_distance_is_six_l2844_284482


namespace wire_length_proof_l2844_284431

theorem wire_length_proof (total_wires : ℕ) (overall_avg : ℝ) (long_wires : ℕ) (long_avg : ℝ) :
  total_wires = 6 →
  overall_avg = 80 →
  long_wires = 4 →
  long_avg = 85 →
  let short_wires := total_wires - long_wires
  let short_avg := (total_wires * overall_avg - long_wires * long_avg) / short_wires
  short_avg = 70 := by sorry

end wire_length_proof_l2844_284431


namespace triangle_area_l2844_284489

/-- The area of a triangle with base 12 and height 15 is 90 -/
theorem triangle_area (base height area : ℝ) : 
  base = 12 → height = 15 → area = (1/2) * base * height → area = 90 := by sorry

end triangle_area_l2844_284489


namespace carls_watermelon_profit_l2844_284425

/-- Calculates the profit of a watermelon seller -/
def watermelon_profit (initial_count : ℕ) (final_count : ℕ) (price_per_melon : ℕ) : ℕ :=
  (initial_count - final_count) * price_per_melon

/-- Theorem: Carl's watermelon profit -/
theorem carls_watermelon_profit :
  watermelon_profit 53 18 3 = 105 := by
  sorry

end carls_watermelon_profit_l2844_284425


namespace sin_product_equals_one_sixteenth_l2844_284471

theorem sin_product_equals_one_sixteenth :
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (54 * π / 180) * Real.sin (72 * π / 180) = 1 / 16 := by
  sorry

end sin_product_equals_one_sixteenth_l2844_284471


namespace expanded_figure_perimeter_l2844_284435

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the figure composed of squares -/
structure ExpandedFigure where
  squares : List Square
  bottomRowCount : ℕ
  topRowCount : ℕ

/-- Calculates the perimeter of the expanded figure -/
def perimeter (figure : ExpandedFigure) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem expanded_figure_perimeter :
  ∀ (figure : ExpandedFigure),
    (∀ s ∈ figure.squares, s.sideLength = 2) →
    figure.bottomRowCount = 3 →
    figure.topRowCount = 1 →
    figure.squares.length = 4 →
    perimeter figure = 20 :=
  sorry

end expanded_figure_perimeter_l2844_284435


namespace total_distance_12_hours_l2844_284440

def car_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  let speeds := List.range hours |>.map (fun h => initial_speed + h * speed_increase)
  speeds.sum

theorem total_distance_12_hours :
  car_distance 50 2 12 = 782 := by
  sorry

end total_distance_12_hours_l2844_284440


namespace tenth_graders_truth_count_l2844_284410

def is_valid_response_count (n : ℕ) (truth_tellers : ℕ) : Prop :=
  n > 0 ∧ truth_tellers ≤ n ∧ 
  (truth_tellers * (n - 1) + (n - truth_tellers) * truth_tellers = 44) ∧
  (truth_tellers * (n - truth_tellers) + (n - truth_tellers) * (n - 1 - truth_tellers) = 28)

theorem tenth_graders_truth_count :
  ∃ (n : ℕ) (t : ℕ), 
    is_valid_response_count n t ∧ 
    (t * (n - 1) = 16 ∨ t * (n - 1) = 56) := by
  sorry

end tenth_graders_truth_count_l2844_284410


namespace race_cars_alignment_l2844_284443

theorem race_cars_alignment (a b c : ℕ) (ha : a = 28) (hb : b = 24) (hc : c = 32) :
  Nat.lcm (Nat.lcm a b) c = 672 := by
  sorry

end race_cars_alignment_l2844_284443


namespace root_of_cubic_l2844_284403

theorem root_of_cubic (x₁ x₂ x₃ : ℝ) (p q r : ℝ) :
  (∀ x, x^3 + p*x^2 + q*x + r = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (Real.sqrt 2)^3 - 3*(Real.sqrt 2)^2*(Real.sqrt 2) + 7*(Real.sqrt 2) - 3*(Real.sqrt 2) = 0 :=
by sorry

end root_of_cubic_l2844_284403


namespace product_xy_equals_25_l2844_284447

theorem product_xy_equals_25 (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 32)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(5*y) = 1024) :
  x * y = 25 := by
  sorry

end product_xy_equals_25_l2844_284447


namespace line_relationships_l2844_284430

-- Define a type for lines in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- For simplicity, we'll just use an opaque type
  mk :: (dummy : Unit)

-- Define the relationships between lines
def parallel (l1 l2 : Line3D) : Prop := sorry

def intersects (l1 l2 : Line3D) : Prop := sorry

def skew (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem line_relationships (a b c : Line3D) 
  (h1 : parallel a b) 
  (h2 : intersects a c) :
  skew b c ∨ intersects b c := by sorry

end line_relationships_l2844_284430


namespace same_color_probability_l2844_284494

/-- Represents the number of sides on each die -/
def totalSides : ℕ := 20

/-- Represents the number of orange sides on each die -/
def orangeSides : ℕ := 3

/-- Represents the number of purple sides on each die -/
def purpleSides : ℕ := 5

/-- Represents the number of green sides on each die -/
def greenSides : ℕ := 6

/-- Represents the number of blue sides on each die -/
def blueSides : ℕ := 5

/-- Represents the number of sparkly sides on each die -/
def sparklySides : ℕ := 1

/-- Theorem stating the probability of rolling the same color or shade on both dice -/
theorem same_color_probability : 
  (orangeSides^2 + purpleSides^2 + greenSides^2 + blueSides^2 + sparklySides^2) / totalSides^2 = 24 / 100 := by
  sorry

end same_color_probability_l2844_284494


namespace mary_fruit_difference_l2844_284417

/-- Proves that Mary has 33 fewer peaches than apples given the conditions about Jake, Steven, and Mary's fruits. -/
theorem mary_fruit_difference :
  ∀ (steven_apples steven_peaches jake_apples jake_peaches mary_apples mary_peaches : ℕ),
  steven_apples = 11 →
  steven_peaches = 18 →
  jake_peaches + 8 = steven_peaches →
  jake_apples = steven_apples + 10 →
  mary_apples = 2 * jake_apples →
  mary_peaches * 2 = steven_peaches →
  (mary_peaches : ℤ) - (mary_apples : ℤ) = -33 := by
sorry

end mary_fruit_difference_l2844_284417


namespace quiz_competition_participants_l2844_284449

theorem quiz_competition_participants (initial_participants : ℕ) : 
  (initial_participants : ℝ) * 0.4 * 0.25 = 30 → initial_participants = 300 := by
  sorry

end quiz_competition_participants_l2844_284449


namespace sixteen_equal_parts_l2844_284453

/-- Represents a rectangular frame with a hollow space inside -/
structure RectangularFrame where
  height : ℝ
  width : ℝ
  hollow : Bool

/-- Represents a division of a rectangular frame -/
structure FrameDivision where
  horizontal_cuts : ℕ
  vertical_cuts : ℕ

/-- Calculates the number of parts resulting from a frame division -/
def number_of_parts (d : FrameDivision) : ℕ :=
  (d.horizontal_cuts + 1) * (d.vertical_cuts + 1)

/-- Theorem stating that one horizontal cut and seven vertical cuts result in 16 equal parts -/
theorem sixteen_equal_parts (f : RectangularFrame) :
  let d := FrameDivision.mk 1 7
  number_of_parts d = 16 := by
  sorry

end sixteen_equal_parts_l2844_284453


namespace cube_volume_surface_area_l2844_284419

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 3*x ∧ 6*s^2 = 6*x) → x = 3 := by
sorry

end cube_volume_surface_area_l2844_284419


namespace rectangular_solid_diagonal_l2844_284459

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 26) 
  (h2 : 4 * (a + b + c) = 28) : 
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 23 := by
  sorry

end rectangular_solid_diagonal_l2844_284459


namespace box_area_is_2144_l2844_284496

/-- The surface area of a box formed by removing square corners from a rectangular sheet. -/
def box_surface_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Theorem stating that the surface area of the box is 2144 square units. -/
theorem box_area_is_2144 :
  box_surface_area 60 40 8 = 2144 :=
by sorry

end box_area_is_2144_l2844_284496


namespace average_of_last_part_calculation_l2844_284495

def average_of_last_part (total_count : ℕ) (total_average : ℚ) (first_part_count : ℕ) (first_part_average : ℚ) (middle_result : ℚ) : ℚ :=
  let last_part_count := total_count - first_part_count - 1
  let total_sum := total_count * total_average
  let first_part_sum := first_part_count * first_part_average
  (total_sum - first_part_sum - middle_result) / last_part_count

theorem average_of_last_part_calculation :
  average_of_last_part 25 50 12 14 878 = 204 / 13 := by
  sorry

end average_of_last_part_calculation_l2844_284495


namespace sibling_pair_probability_l2844_284484

theorem sibling_pair_probability 
  (business_students : ℕ) 
  (law_students : ℕ) 
  (sibling_pairs : ℕ) 
  (h1 : business_students = 500) 
  (h2 : law_students = 800) 
  (h3 : sibling_pairs = 30) : 
  (sibling_pairs : ℚ) / (business_students * law_students) = 30 / 400000 := by
  sorry

end sibling_pair_probability_l2844_284484


namespace three_planes_division_l2844_284448

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this problem

/-- The number of regions that a set of planes divides 3D space into -/
def num_regions (planes : List Plane3D) : ℕ := sorry

theorem three_planes_division :
  ∀ (p1 p2 p3 : Plane3D),
  ∃ (min max : ℕ),
    (∀ (n : ℕ), n = num_regions [p1, p2, p3] → min ≤ n ∧ n ≤ max) ∧
    min = 4 ∧ max = 8 := by sorry

end three_planes_division_l2844_284448


namespace square_equals_multiplication_l2844_284404

theorem square_equals_multiplication (a : ℝ) : a * a = a ^ 2 := by
  sorry

end square_equals_multiplication_l2844_284404


namespace complex_sum_powers_of_i_l2844_284402

theorem complex_sum_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ i + i^2 + i^3 + i^4 = 0 := by
  sorry

end complex_sum_powers_of_i_l2844_284402


namespace counterfeit_coin_determination_l2844_284437

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a group of coins -/
structure CoinGroup where
  size : Nat
  hasFake : Bool

/-- Represents a weighing action -/
structure Weighing where
  left : CoinGroup
  right : CoinGroup

/-- The state of knowledge about the counterfeit coins -/
inductive FakeState
  | Unknown : FakeState
  | Heavier : FakeState
  | Lighter : FakeState

/-- A strategy for determining the state of the counterfeit coins -/
def Strategy := List Weighing

/-- The result of applying a strategy -/
def StrategyResult := FakeState

/-- Axiom: There are 239 coins in total -/
axiom total_coins : Nat
axiom total_coins_eq : total_coins = 239

/-- Axiom: There are exactly two counterfeit coins -/
axiom num_fake_coins : Nat
axiom num_fake_coins_eq : num_fake_coins = 2

/-- Theorem: It is possible to determine whether the counterfeit coins are heavier or lighter in exactly three weighings -/
theorem counterfeit_coin_determination :
  ∃ (s : Strategy),
    (s.length = 3) ∧
    (∀ (fake_heavier : Bool),
      ∃ (result : StrategyResult),
        (result = FakeState.Heavier ∧ fake_heavier = true) ∨
        (result = FakeState.Lighter ∧ fake_heavier = false)) :=
by sorry

end counterfeit_coin_determination_l2844_284437


namespace intersection_theorem_l2844_284446

-- Define the sets A and B
def A : Set ℝ := {x | x < -3 ∨ x > 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the expected result
def expected_result : Set ℝ := {x | 1 < x ∧ x ≤ 4}

-- State the theorem
theorem intersection_theorem : A_intersect_B = expected_result := by sorry

end intersection_theorem_l2844_284446


namespace parabola_sum_coefficients_l2844_284426

/-- A parabola with equation x = ay^2 + by + c, vertex at (-3, 2), and passing through (-1, 0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_condition : -3 = a * 2^2 + b * 2 + c
  point_condition : -1 = a * 0^2 + b * 0 + c

/-- The sum of coefficients a, b, and c for the given parabola is -7/2 -/
theorem parabola_sum_coefficients (p : Parabola) : p.a + p.b + p.c = -7/2 := by
  sorry

end parabola_sum_coefficients_l2844_284426


namespace consecutive_pages_sum_l2844_284455

theorem consecutive_pages_sum (x y : ℕ) : 
  x + y = 125 → y = x + 1 → y = 63 := by
  sorry

end consecutive_pages_sum_l2844_284455


namespace gcd_problem_l2844_284460

theorem gcd_problem (m n : ℕ+) (h : Nat.gcd m n = 10) : Nat.gcd (12 * m) (18 * n) = 60 := by
  sorry

end gcd_problem_l2844_284460


namespace single_elimination_512_players_games_l2844_284498

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_players : ℕ
  single_elimination : Bool

/-- Calculates the number of games required to determine a champion in a single-elimination tournament. -/
def games_required (t : Tournament) : ℕ :=
  if t.single_elimination then t.num_players - 1 else 0

/-- Theorem stating that a single-elimination tournament with 512 players requires 511 games. -/
theorem single_elimination_512_players_games (t : Tournament) 
  (h1 : t.num_players = 512) 
  (h2 : t.single_elimination = true) : 
  games_required t = 511 := by
  sorry

#eval games_required ⟨512, true⟩

end single_elimination_512_players_games_l2844_284498


namespace multiples_of_15_between_12_and_202_l2844_284497

theorem multiples_of_15_between_12_and_202 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ n > 12 ∧ n < 202) (Finset.range 202)).card = 13 := by
  sorry

end multiples_of_15_between_12_and_202_l2844_284497


namespace hexagon_pillar_height_l2844_284491

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a regular hexagon with pillars -/
structure HexagonWithPillars where
  sideLength : ℝ
  A : Point3D
  B : Point3D
  C : Point3D
  E : Point3D

/-- The theorem to be proved -/
theorem hexagon_pillar_height 
  (h : HexagonWithPillars) 
  (h_side : h.sideLength > 0)
  (h_A : h.A = ⟨0, 0, 12⟩)
  (h_B : h.B = ⟨h.sideLength, 0, 9⟩)
  (h_C : h.C = ⟨h.sideLength / 2, h.sideLength * Real.sqrt 3 / 2, 10⟩)
  (h_E : h.E = ⟨-h.sideLength, 0, h.E.z⟩) :
  h.E.z = 17 := by
  sorry


end hexagon_pillar_height_l2844_284491


namespace matrix_multiplication_result_l2844_284464

theorem matrix_multiplication_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  A * B = !![23, -7; 24, -16] := by
  sorry

end matrix_multiplication_result_l2844_284464


namespace intersection_of_A_and_B_l2844_284445

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x^2 - x < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l2844_284445


namespace symmetric_point_quadrant_l2844_284418

/-- Given that point P(m,m-n) is symmetric to point Q(2,3) with respect to the origin,
    prove that point M(m,n) is in the second quadrant. -/
theorem symmetric_point_quadrant (m n : ℝ) : 
  (m = -2 ∧ m - n = -3) → m < 0 ∧ n > 0 :=
by sorry

end symmetric_point_quadrant_l2844_284418


namespace function_value_at_negative_l2844_284444

theorem function_value_at_negative (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x + 1 / x - 1) (h2 : f a = 2) :
  f (-a) = -4 := by
  sorry

end function_value_at_negative_l2844_284444


namespace existence_of_multiple_2002_l2844_284478

theorem existence_of_multiple_2002 (a : Fin 41 → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ i j m n p q : Fin 41, i ≠ j ∧ m ≠ n ∧ p ≠ q ∧
    i ≠ m ∧ i ≠ n ∧ i ≠ p ∧ i ≠ q ∧
    j ≠ m ∧ j ≠ n ∧ j ≠ p ∧ j ≠ q ∧
    m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧
    (2002 ∣ (a i - a j) * (a m - a n) * (a p - a q)) :=
sorry

end existence_of_multiple_2002_l2844_284478


namespace covered_area_is_56_l2844_284442

/-- Represents a rectangular strip of paper -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℝ := s.length * s.width

/-- Calculates the area of intersection between two perpendicular strips -/
def intersectionArea (s1 s2 : Strip) : ℝ := s1.width * s2.width

/-- Represents the arrangement of strips on the table -/
structure StripArrangement where
  horizontalStrips : Fin 3 → Strip
  verticalStrips : Fin 2 → Strip
  all_strips_same : ∀ (i : Fin 3) (j : Fin 2), 
    (horizontalStrips i).length = 8 ∧ (horizontalStrips i).width = 2 ∧
    (verticalStrips j).length = 8 ∧ (verticalStrips j).width = 2

/-- Calculates the total area covered by the strips -/
def coveredArea (arr : StripArrangement) : ℝ :=
  let totalStripArea := (3 * stripArea (arr.horizontalStrips 0)) + (2 * stripArea (arr.verticalStrips 0))
  let totalOverlapArea := 6 * intersectionArea (arr.horizontalStrips 0) (arr.verticalStrips 0)
  totalStripArea - totalOverlapArea

/-- Theorem stating that the covered area is 56 square units -/
theorem covered_area_is_56 (arr : StripArrangement) : coveredArea arr = 56 := by
  sorry

end covered_area_is_56_l2844_284442


namespace surviving_cells_after_6_hours_l2844_284465

def cell_population (n : ℕ) : ℕ := 2^n + 1

theorem surviving_cells_after_6_hours :
  cell_population 6 = 65 :=
sorry

end surviving_cells_after_6_hours_l2844_284465


namespace sum_lower_bound_l2844_284472

theorem sum_lower_bound (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) (h4 : a = 1 / b) :
  a + 2014 * b > 2015 := by
  sorry

end sum_lower_bound_l2844_284472


namespace characterize_S_l2844_284421

/-- The function f(A, B, C) = A^3 + B^3 + C^3 - 3ABC -/
def f (A B C : ℕ) : ℤ := A^3 + B^3 + C^3 - 3 * A * B * C

/-- The set of all possible values of f(A, B, C) -/
def S : Set ℤ := {n | ∃ (A B C : ℕ), f A B C = n}

/-- The theorem stating the characterization of S -/
theorem characterize_S : S = {n : ℤ | n ≥ 0 ∧ n % 9 ≠ 3 ∧ n % 9 ≠ 6} := by sorry

end characterize_S_l2844_284421


namespace equilateral_triangle_side_length_l2844_284408

/-- The side length of an equilateral triangle with inscribed circle and smaller touching circles -/
theorem equilateral_triangle_side_length (r : ℝ) (h : r > 0) : ∃ a : ℝ, 
  a > 0 ∧ 
  (∃ R : ℝ, R > 0 ∧ 
    -- R is the radius of the inscribed circle
    R = (a * Real.sqrt 3) / 6 ∧
    -- Relationship between R, r, and the altitude of the triangle
    R / r = (a * Real.sqrt 3 / 3) / (a * Real.sqrt 3 / 3 - R - r)) ∧
  a = 6 * r * Real.sqrt 3 := by
  sorry

end equilateral_triangle_side_length_l2844_284408


namespace integer_fraction_solutions_l2844_284488

theorem integer_fraction_solutions (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℚ) / (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1) = k) ↔
  (∃ n : ℕ+, (a = n ∧ b = 2 * n) ∨ 
             (a = 8 * n ^ 4 - n ∧ b = 2 * n) ∨ 
             (a = 2 * n ∧ b = 1)) :=
sorry

end integer_fraction_solutions_l2844_284488
