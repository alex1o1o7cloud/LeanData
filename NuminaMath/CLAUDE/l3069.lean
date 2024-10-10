import Mathlib

namespace twentieth_term_is_59_l3069_306962

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 20th term of the arithmetic sequence with first term 2 and common difference 3 is 59 -/
theorem twentieth_term_is_59 :
  arithmeticSequenceTerm 2 3 20 = 59 := by
  sorry

end twentieth_term_is_59_l3069_306962


namespace max_value_of_f_l3069_306903

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ f x) ∧
  f x = Real.pi / 12 + Real.sqrt 3 / 2 := by
sorry

end max_value_of_f_l3069_306903


namespace power_division_equality_l3069_306995

theorem power_division_equality : (3 : ℕ)^12 / ((27 : ℕ)^2 * 3^3) = 27 := by
  sorry

end power_division_equality_l3069_306995


namespace probability_neither_red_nor_purple_l3069_306916

theorem probability_neither_red_nor_purple :
  let total_balls : ℕ := 120
  let red_balls : ℕ := 15
  let purple_balls : ℕ := 3
  let neither_red_nor_purple : ℕ := total_balls - (red_balls + purple_balls)
  (neither_red_nor_purple : ℚ) / total_balls = 17 / 20 := by
  sorry

end probability_neither_red_nor_purple_l3069_306916


namespace initial_mixture_volume_l3069_306901

theorem initial_mixture_volume 
  (initial_x_percentage : Real) 
  (initial_y_percentage : Real)
  (added_x_volume : Real)
  (final_x_percentage : Real) :
  initial_x_percentage = 0.20 →
  initial_y_percentage = 0.80 →
  added_x_volume = 20 →
  final_x_percentage = 0.36 →
  ∃ (initial_volume : Real),
    initial_volume = 80 ∧
    (initial_x_percentage * initial_volume + added_x_volume) / (initial_volume + added_x_volume) = final_x_percentage :=
by sorry

end initial_mixture_volume_l3069_306901


namespace modulus_of_z_l3069_306910

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_z_l3069_306910


namespace seven_keys_three_adjacent_l3069_306913

/-- The number of distinct arrangements of keys on a keychain. -/
def keychain_arrangements (total_keys : ℕ) (adjacent_keys : ℕ) : ℕ :=
  (adjacent_keys.factorial * ((total_keys - adjacent_keys + 1 - 1).factorial / 2))

/-- Theorem stating the number of distinct arrangements for 7 keys with 3 adjacent -/
theorem seven_keys_three_adjacent :
  keychain_arrangements 7 3 = 72 := by
  sorry

end seven_keys_three_adjacent_l3069_306913


namespace chess_tournament_score_change_l3069_306981

/-- Represents a chess tournament with 2n players -/
structure ChessTournament (n : ℕ) where
  players : Fin (2 * n)
  score : Fin (2 * n) → ℝ
  score_change : Fin (2 * n) → ℝ

/-- The theorem to be proved -/
theorem chess_tournament_score_change (n : ℕ) (tournament : ChessTournament n) :
  (∀ p, tournament.score_change p ≥ n) →
  (∀ p, tournament.score_change p = n) :=
by sorry

end chess_tournament_score_change_l3069_306981


namespace quadratic_properties_l3069_306972

def f (x : ℝ) := (x - 1)^2 + 3

theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x > f y → f ((x + y) / 2) < f x) ∧ 
  (∀ x : ℝ, f (x + 1) = f (1 - x)) ∧
  (f 1 = 3 ∧ ∀ x : ℝ, f x ≥ 3) ∧
  f 0 ≠ 3 :=
sorry

end quadratic_properties_l3069_306972


namespace rational_square_sum_l3069_306906

theorem rational_square_sum (a b c : ℚ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  ∃ r : ℚ, (1 / (a - b)^2 + 1 / (b - c)^2 + 1 / (c - a)^2) = r^2 := by
  sorry

end rational_square_sum_l3069_306906


namespace simple_interest_calculation_l3069_306955

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 780)
  (h2 : rate = 4.166666666666667 / 100)
  (h3 : time = 4) :
  principal * rate * time = 130 := by
  sorry

end simple_interest_calculation_l3069_306955


namespace cats_not_liking_tuna_or_chicken_l3069_306908

theorem cats_not_liking_tuna_or_chicken 
  (total : ℕ) (tuna : ℕ) (chicken : ℕ) (both : ℕ) :
  total = 80 → tuna = 15 → chicken = 60 → both = 10 →
  total - (tuna + chicken - both) = 15 := by
sorry

end cats_not_liking_tuna_or_chicken_l3069_306908


namespace f_not_monotonic_implies_k_range_l3069_306932

noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + 1

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y ∨ f x > f y

theorem f_not_monotonic_implies_k_range (k : ℝ) :
  (∀ x, x > 0 → f x = x^2 - (1/2) * Real.log x + 1) →
  (¬ is_monotonic f (k - 1) (k + 1)) →
  k ∈ Set.Icc 1 (3/2) :=
sorry

end f_not_monotonic_implies_k_range_l3069_306932


namespace correct_elderly_sample_size_l3069_306944

/-- Represents the number of people in each age group -/
structure Population where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Calculates the total population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Calculates the number of elderly people to be sampled -/
def elderlySampleSize (p : Population) (sampleSize : ℕ) : ℕ :=
  (p.elderly * sampleSize) / totalPopulation p

theorem correct_elderly_sample_size (p : Population) (sampleSize : ℕ) 
  (h1 : p.elderly = 30)
  (h2 : p.middleAged = 90)
  (h3 : p.young = 60)
  (h4 : sampleSize = 36) :
  elderlySampleSize p sampleSize = 6 := by
  sorry

end correct_elderly_sample_size_l3069_306944


namespace function_property_k_value_l3069_306949

theorem function_property_k_value (f : ℝ → ℝ) (k : ℝ) 
  (h1 : f 1 = 4)
  (h2 : ∀ x y, f (x + y) = f x + f y + k * x * y + 4)
  (h3 : f 2 + f 5 = 125) :
  k = 7 := by sorry

end function_property_k_value_l3069_306949


namespace bird_sanctuary_geese_percentage_l3069_306917

theorem bird_sanctuary_geese_percentage :
  let total_percentage : ℚ := 100
  let geese_percentage : ℚ := 40
  let swan_percentage : ℚ := 20
  let heron_percentage : ℚ := 15
  let duck_percentage : ℚ := 25
  let non_duck_percentage : ℚ := total_percentage - duck_percentage
  geese_percentage / non_duck_percentage * 100 = 53 + 1/3 :=
by sorry

end bird_sanctuary_geese_percentage_l3069_306917


namespace negation_of_existence_proposition_l3069_306948

theorem negation_of_existence_proposition :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end negation_of_existence_proposition_l3069_306948


namespace f_of_7_eq_17_l3069_306943

/-- The polynomial function f(x) = 2x^4 - 17x^3 + 26x^2 - 24x - 60 -/
def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 26*x^2 - 24*x - 60

/-- Theorem: The value of f(7) is 17 -/
theorem f_of_7_eq_17 : f 7 = 17 := by
  sorry

end f_of_7_eq_17_l3069_306943


namespace abc_inequality_l3069_306912

theorem abc_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1/3) ∧ 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ 1/2) := by
  sorry

end abc_inequality_l3069_306912


namespace quadrilateral_circumscribed_l3069_306975

-- Define the structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the property of being convex
def is_convex (q : Quadrilateral) : Prop := sorry

-- Define the property of a point being inside a quadrilateral
def is_interior_point (P : Point) (q : Quadrilateral) : Prop := sorry

-- Define angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- Define the property of being circumscribed
def is_circumscribed (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_circumscribed 
  (q : Quadrilateral) (P : Point) 
  (h_convex : is_convex q)
  (h_interior : is_interior_point P q)
  (h_angle1 : angle_measure q.A P q.B + angle_measure q.C P q.D = 
              angle_measure q.B P q.C + angle_measure q.D P q.A)
  (h_angle2 : angle_measure P q.A q.D + angle_measure P q.C q.D = 
              angle_measure P q.A q.B + angle_measure P q.C q.B)
  (h_angle3 : angle_measure P q.D q.C + angle_measure P q.B q.C = 
              angle_measure P q.D q.A + angle_measure P q.B q.A) :
  is_circumscribed q :=
sorry

end quadrilateral_circumscribed_l3069_306975


namespace smallest_n_divisible_by_2009_n_42_divisible_by_2009_exists_unique_smallest_n_l3069_306976

theorem smallest_n_divisible_by_2009 :
  ∀ n : ℕ, n > 1 → n^2 * (n - 1) % 2009 = 0 → n ≥ 42 :=
by
  sorry

theorem n_42_divisible_by_2009 : 42^2 * (42 - 1) % 2009 = 0 :=
by
  sorry

theorem exists_unique_smallest_n :
  ∃! n : ℕ, n > 1 ∧ n^2 * (n - 1) % 2009 = 0 ∧ ∀ m : ℕ, m > 1 → m^2 * (m - 1) % 2009 = 0 → n ≤ m :=
by
  sorry

end smallest_n_divisible_by_2009_n_42_divisible_by_2009_exists_unique_smallest_n_l3069_306976


namespace car_distribution_l3069_306996

theorem car_distribution (total_cars : ℕ) (first_supplier : ℕ) : 
  total_cars = 5650000 →
  first_supplier = 1000000 →
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let remaining_cars := total_cars - (first_supplier + second_supplier + third_supplier)
  remaining_cars / 2 = 325000 :=
by sorry

end car_distribution_l3069_306996


namespace cab_driver_first_day_income_l3069_306939

def cab_driver_income (day2 day3 day4 day5 : ℕ) (average : ℕ) : Prop :=
  ∃ day1 : ℕ,
    day2 = 250 ∧
    day3 = 450 ∧
    day4 = 400 ∧
    day5 = 800 ∧
    average = 500 ∧
    (day1 + day2 + day3 + day4 + day5) / 5 = average ∧
    day1 = 600

theorem cab_driver_first_day_income :
  ∀ day2 day3 day4 day5 average : ℕ,
    cab_driver_income day2 day3 day4 day5 average →
    ∃ day1 : ℕ, day1 = 600 :=
by
  sorry

end cab_driver_first_day_income_l3069_306939


namespace line_equation_correct_l3069_306947

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfies_equation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Check if a line equation represents a line with a given slope -/
def has_slope (eq : LineEquation) (m : ℝ) : Prop :=
  eq.a ≠ 0 ∧ eq.b ≠ 0 ∧ m = -eq.a / eq.b

theorem line_equation_correct (L : Line) (eq : LineEquation) : 
  L.point = (-2, 5) →
  L.slope = -3/4 →
  eq = ⟨3, 4, -14⟩ →
  satisfies_equation L.point eq ∧ has_slope eq L.slope :=
sorry

end line_equation_correct_l3069_306947


namespace inequality_solution_l3069_306923

theorem inequality_solution :
  {x : ℝ | |x - 2| + |x + 3| + |2*x - 1| < 7} = Set.Icc (-1.5) 2 := by
  sorry

end inequality_solution_l3069_306923


namespace seven_mile_taxi_cost_l3069_306968

/-- The cost of a taxi ride given the distance traveled -/
def taxi_cost (fixed_cost : ℚ) (per_mile_cost : ℚ) (miles : ℚ) : ℚ :=
  fixed_cost + per_mile_cost * miles

/-- Theorem: The cost of a 7-mile taxi ride is $4.10 -/
theorem seven_mile_taxi_cost :
  taxi_cost 2 0.3 7 = 4.1 := by
  sorry

end seven_mile_taxi_cost_l3069_306968


namespace smallest_addition_for_divisibility_smallest_addition_for_27452_div_9_smallest_addition_is_7_l3069_306946

theorem smallest_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem smallest_addition_for_27452_div_9 :
  ∃ (x : ℕ), x < 9 ∧ (27452 + x) % 9 = 0 ∧ ∀ (y : ℕ), y < x → (27452 + y) % 9 ≠ 0 :=
by
  apply smallest_addition_for_divisibility 27452 9
  norm_num

theorem smallest_addition_is_7 :
  7 < 9 ∧ (27452 + 7) % 9 = 0 ∧ ∀ (y : ℕ), y < 7 → (27452 + y) % 9 ≠ 0 :=
by sorry

end smallest_addition_for_divisibility_smallest_addition_for_27452_div_9_smallest_addition_is_7_l3069_306946


namespace three_squares_sum_l3069_306960

theorem three_squares_sum (n : ℤ) : 3*(n-1)^2 + 8 = (n-3)^2 + (n-1)^2 + (n+1)^2 := by
  sorry

end three_squares_sum_l3069_306960


namespace complex_equation_solution_l3069_306921

theorem complex_equation_solution (a : ℝ) : 
  (a^2 - a : ℂ) + (3*a - 1 : ℂ)*Complex.I = 2 + 5*Complex.I → a = 2 := by
  sorry

end complex_equation_solution_l3069_306921


namespace value_of_expression_l3069_306918

theorem value_of_expression (x : ℝ) (h : x^2 - x = 1) : 1 + 2*x - 2*x^2 = -1 := by
  sorry

end value_of_expression_l3069_306918


namespace ferris_wheel_capacity_is_120_l3069_306924

/-- Calculates the maximum number of people who can ride a Ferris wheel simultaneously -/
def max_ferris_wheel_capacity (total_seats : ℕ) (people_per_seat : ℕ) (broken_seats : ℕ) : ℕ :=
  (total_seats - broken_seats) * people_per_seat

/-- Proves that the maximum capacity of the Ferris wheel under given conditions is 120 people -/
theorem ferris_wheel_capacity_is_120 :
  max_ferris_wheel_capacity 18 15 10 = 120 := by
  sorry

end ferris_wheel_capacity_is_120_l3069_306924


namespace impossible_to_reach_target_l3069_306904

/-- Represents the configuration of matchsticks on a square's vertices -/
structure SquareConfig where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  a₄ : ℕ

/-- Calculates S for a given configuration -/
def S (config : SquareConfig) : ℤ :=
  config.a₁ - config.a₂ + config.a₃ - config.a₄

/-- Represents a valid move in the matchstick game -/
inductive Move
  | move_a₁ (k : ℕ)
  | move_a₂ (k : ℕ)
  | move_a₃ (k : ℕ)
  | move_a₄ (k : ℕ)

/-- Applies a move to a configuration -/
def apply_move (config : SquareConfig) (move : Move) : SquareConfig :=
  match move with
  | Move.move_a₁ k => ⟨config.a₁ - k, config.a₂ + k, config.a₃, config.a₄ + k⟩
  | Move.move_a₂ k => ⟨config.a₁ + k, config.a₂ - k, config.a₃ + k, config.a₄⟩
  | Move.move_a₃ k => ⟨config.a₁, config.a₂ + k, config.a₃ - k, config.a₄ + k⟩
  | Move.move_a₄ k => ⟨config.a₁ + k, config.a₂, config.a₃ + k, config.a₄ - k⟩

/-- The main theorem stating the impossibility of reaching the target configuration -/
theorem impossible_to_reach_target :
  ∀ (moves : List Move),
  let start_config := ⟨1, 0, 0, 0⟩
  let end_config := List.foldl apply_move start_config moves
  end_config ≠ ⟨1, 9, 8, 9⟩ := by
  sorry

/-- Lemma: S mod 3 is invariant under moves -/
lemma S_mod_3_invariant (config : SquareConfig) (move : Move) :
  (S config) % 3 = (S (apply_move config move)) % 3 := by
  sorry

end impossible_to_reach_target_l3069_306904


namespace valid_range_for_square_root_fraction_l3069_306980

theorem valid_range_for_square_root_fraction (x : ℝ) :
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 1)) ↔ x > 1 := by
  sorry

end valid_range_for_square_root_fraction_l3069_306980


namespace unique_positive_solution_l3069_306900

def f (x : ℝ) : ℝ := x^4 + 9*x^3 + 18*x^2 + 2023*x - 2021

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x^12 + 9*x^11 + 18*x^10 + 2023*x^9 - 2021*x^8 = 0 :=
by
  sorry


end unique_positive_solution_l3069_306900


namespace grandfathers_age_is_79_l3069_306966

/-- The age of Caleb's grandfather based on the number of candles on the cake -/
def grandfathers_age (yellow_candles red_candles blue_candles : ℕ) : ℕ :=
  yellow_candles + red_candles + blue_candles

/-- Theorem stating that Caleb's grandfather's age is 79 given the number of candles -/
theorem grandfathers_age_is_79 :
  grandfathers_age 27 14 38 = 79 := by
  sorry

end grandfathers_age_is_79_l3069_306966


namespace two_solutions_exist_sum_of_solutions_l3069_306952

/-- Sum of digits of a positive integer in base 10 -/
def sum_of_digits (n : ℕ+) : ℕ :=
  sorry

/-- The equation n - 3 * sum_of_digits n = 2022 has exactly two solutions -/
theorem two_solutions_exist :
  ∃ (n1 n2 : ℕ+),
    n1 - 3 * sum_of_digits n1 = 2022 ∧
    n2 - 3 * sum_of_digits n2 = 2022 ∧
    n1 ≠ n2 ∧
    ∀ (n : ℕ+), n - 3 * sum_of_digits n = 2022 → n = n1 ∨ n = n2 :=
  sorry

/-- The sum of the two solutions is 4107 -/
theorem sum_of_solutions :
  ∃ (n1 n2 : ℕ+),
    n1 - 3 * sum_of_digits n1 = 2022 ∧
    n2 - 3 * sum_of_digits n2 = 2022 ∧
    n1 ≠ n2 ∧
    n1 + n2 = 4107 :=
  sorry

end two_solutions_exist_sum_of_solutions_l3069_306952


namespace regular_polygon_162_degrees_l3069_306994

/-- A regular polygon with interior angles measuring 162 degrees has 20 sides -/
theorem regular_polygon_162_degrees : ∀ n : ℕ, 
  n > 2 → 
  (180 * (n - 2) : ℝ) / n = 162 → 
  n = 20 := by
  sorry

end regular_polygon_162_degrees_l3069_306994


namespace binomial_expansion_constant_term_l3069_306965

/-- Given a natural number n, prove that if the sum of all coefficients in the expansion of (√x + 3/x)^n
    plus the sum of binomial coefficients equals 72, then the constant term in the expansion is 9. -/
theorem binomial_expansion_constant_term (n : ℕ) : 
  (4^n + 2^n = 72) → 
  (∃ (r : ℕ), r < n ∧ (n.choose r) * 3^r = 9) :=
by sorry

end binomial_expansion_constant_term_l3069_306965


namespace jasons_initial_money_l3069_306973

theorem jasons_initial_money (initial_money : ℝ) : 
  let remaining_after_books := (3/4 : ℝ) * initial_money - 10
  let remaining_after_dvds := remaining_after_books - (2/5 : ℝ) * remaining_after_books - 8
  remaining_after_dvds = 130 → initial_money = 320 := by
sorry

end jasons_initial_money_l3069_306973


namespace binary_arithmetic_l3069_306991

/-- Addition and subtraction of binary numbers --/
theorem binary_arithmetic : 
  let a := 0b1101
  let b := 0b10
  let c := 0b101
  let d := 0b11
  let result := 0b1011
  (a + b + c) - d = result :=
by sorry

end binary_arithmetic_l3069_306991


namespace min_reciprocal_sum_l3069_306967

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 3) :
  (1/x + 1/y : ℝ) ≥ 1 + 2*Real.sqrt 2/3 :=
sorry

end min_reciprocal_sum_l3069_306967


namespace product_of_solutions_l3069_306982

-- Define the equation
def equation (x : ℝ) : Prop :=
  (2 * x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 5)

-- Theorem statement
theorem product_of_solutions : 
  ∃ (x₁ x₂ : ℝ), equation x₁ ∧ equation x₂ ∧ x₁ * x₂ = 3 :=
sorry

end product_of_solutions_l3069_306982


namespace problem_solved_probability_l3069_306929

theorem problem_solved_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h1 : prob_A = 2/3) 
  (h2 : prob_B = 3/4) 
  : prob_A + prob_B - prob_A * prob_B = 11/12 :=
by
  sorry

end problem_solved_probability_l3069_306929


namespace current_algae_count_l3069_306928

/-- The number of algae plants originally in Milford Lake -/
def original_algae : ℕ := 809

/-- The number of additional algae plants in Milford Lake -/
def additional_algae : ℕ := 2454

/-- Theorem stating the current total number of algae plants in Milford Lake -/
theorem current_algae_count : original_algae + additional_algae = 3263 := by
  sorry

end current_algae_count_l3069_306928


namespace at_least_seven_zeros_l3069_306983

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem at_least_seven_zeros (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 3) 
  (h_zero : f 2 = 0) : 
  ∃ (S : Finset ℝ), S.card ≥ 7 ∧ (∀ x ∈ S, 0 < x ∧ x < 6 ∧ f x = 0) :=
sorry

end at_least_seven_zeros_l3069_306983


namespace fraction_ordering_l3069_306984

theorem fraction_ordering : 8 / 31 < 11 / 33 ∧ 11 / 33 < 12 / 29 := by
  sorry

end fraction_ordering_l3069_306984


namespace quadratic_equation_translation_l3069_306988

/-- Quadratic form in two variables -/
def Q (a b c : ℝ) (x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

/-- Theorem: Transformation of quadratic equation using parallel translation -/
theorem quadratic_equation_translation
  (a b c d e f x₀ y₀ : ℝ)
  (h : a * c - b^2 ≠ 0) :
  ∃ f' : ℝ,
    (∀ x y, Q a b c x y + 2 * d * x + 2 * e * y = f) ↔
    (∀ x' y', Q a b c x' y' = f' ∧
      x' = x + x₀ ∧
      y' = y + y₀ ∧
      f' = f - Q a b c x₀ y₀ + 2 * (d * x₀ + e * y₀)) :=
sorry

end quadratic_equation_translation_l3069_306988


namespace complex_number_location_l3069_306940

theorem complex_number_location :
  let z : ℂ := 1 / ((1 + Complex.I)^2 + 1)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_number_location_l3069_306940


namespace min_keys_required_l3069_306989

/-- Represents the hotel key distribution problem -/
structure HotelKeyProblem where
  rooms : ℕ
  guests : ℕ
  returningGuests : ℕ
  keys : ℕ

/-- Predicate to check if a key distribution is valid -/
def isValidDistribution (p : HotelKeyProblem) : Prop :=
  p.rooms > 0 ∧ 
  p.guests > p.rooms ∧ 
  p.returningGuests = p.rooms ∧
  ∀ (subset : Finset ℕ), subset.card = p.returningGuests → 
    ∃ (f : subset → Fin p.rooms), Function.Injective f

/-- Theorem stating the minimum number of keys required -/
theorem min_keys_required (p : HotelKeyProblem) 
  (h : isValidDistribution p) : p.keys ≥ p.rooms * (p.guests - p.rooms + 1) :=
sorry

end min_keys_required_l3069_306989


namespace prize_pricing_and_quantity_l3069_306950

/-- Represents the price of a type A prize -/
def price_A : ℕ := sorry

/-- Represents the price of a type B prize -/
def price_B : ℕ := sorry

/-- The cost of one type A prize and two type B prizes -/
def cost_combination1 : ℕ := 220

/-- The cost of two type A prizes and three type B prizes -/
def cost_combination2 : ℕ := 360

/-- The total number of prizes to be purchased -/
def total_prizes : ℕ := 30

/-- The maximum total cost allowed -/
def max_total_cost : ℕ := 2300

/-- The minimum number of type A prizes that can be purchased -/
def min_type_A_prizes : ℕ := sorry

theorem prize_pricing_and_quantity :
  (price_A + 2 * price_B = cost_combination1) ∧
  (2 * price_A + 3 * price_B = cost_combination2) ∧
  (price_A = 60) ∧
  (price_B = 80) ∧
  (∀ m : ℕ, m ≥ min_type_A_prizes →
    price_A * m + price_B * (total_prizes - m) ≤ max_total_cost) ∧
  (min_type_A_prizes = 5) := by sorry

end prize_pricing_and_quantity_l3069_306950


namespace second_divisor_problem_l3069_306992

theorem second_divisor_problem :
  ∃! D : ℤ, 19 < D ∧ D < 242 ∧
  (∃ N : ℤ, N % 242 = 100 ∧ N % D = 19) ∧
  D = 27 := by
sorry

end second_divisor_problem_l3069_306992


namespace xiao_ming_brother_age_l3069_306997

def has_no_repeated_digits (year : ℕ) : Prop := sorry

def is_multiple_of_19 (year : ℕ) : Prop := year % 19 = 0

theorem xiao_ming_brother_age (birth_year : ℕ) 
  (h1 : is_multiple_of_19 birth_year)
  (h2 : ∀ y : ℕ, birth_year ≤ y → y < 2013 → ¬(has_no_repeated_digits y))
  (h3 : has_no_repeated_digits 2013) :
  2013 - birth_year = 18 := by sorry

end xiao_ming_brother_age_l3069_306997


namespace total_flowers_l3069_306926

def flower_collection (arwen_tulips arwen_roses : ℕ) : ℕ :=
  let elrond_tulips := 2 * arwen_tulips
  let elrond_roses := 3 * arwen_roses
  let galadriel_tulips := 3 * elrond_tulips
  let galadriel_roses := 2 * arwen_roses
  arwen_tulips + arwen_roses + elrond_tulips + elrond_roses + galadriel_tulips + galadriel_roses

theorem total_flowers : flower_collection 20 18 = 288 := by
  sorry

end total_flowers_l3069_306926


namespace mini_croissant_cost_gala_luncheon_cost_l3069_306964

/-- Calculates the cost of mini croissants for a committee luncheon --/
theorem mini_croissant_cost (people : ℕ) (sandwiches_per_person : ℕ) 
  (croissants_per_pack : ℕ) (pack_price : ℚ) : ℕ → ℚ :=
  λ total_croissants =>
    let packs_needed := (total_croissants + croissants_per_pack - 1) / croissants_per_pack
    packs_needed * pack_price

/-- Proves that the cost of mini croissants for the committee luncheon is $32.00 --/
theorem gala_luncheon_cost : 
  mini_croissant_cost 24 2 12 8 (24 * 2) = 32 := by
  sorry

end mini_croissant_cost_gala_luncheon_cost_l3069_306964


namespace cubic_polynomial_proof_l3069_306970

theorem cubic_polynomial_proof : 
  let p : ℝ → ℝ := λ x => -5/6 * x^3 + 5 * x^2 - 85/6 * x - 5
  (p 1 = -10) ∧ (p 2 = -20) ∧ (p 3 = -30) ∧ (p 5 = -70) := by
  sorry

end cubic_polynomial_proof_l3069_306970


namespace barn_paint_area_l3069_306954

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a barn with a dividing wall -/
def totalPaintArea (d : BarnDimensions) : ℝ :=
  let externalWallArea := 2 * (d.width * d.height + d.length * d.height)
  let dividingWallArea := 2 * (d.width * d.height)
  let ceilingArea := d.width * d.length
  2 * externalWallArea + dividingWallArea + ceilingArea

/-- The dimensions of the barn in the problem -/
def problemBarn : BarnDimensions :=
  { width := 12
  , length := 15
  , height := 5 }

theorem barn_paint_area :
  totalPaintArea problemBarn = 840 := by
  sorry


end barn_paint_area_l3069_306954


namespace election_vote_count_l3069_306985

theorem election_vote_count : ∃ (total_votes : ℕ), 
  (total_votes > 0) ∧ 
  (∃ (candidate_votes rival_votes : ℕ),
    (candidate_votes = (total_votes * 3) / 10) ∧
    (rival_votes = candidate_votes + 4000) ∧
    (candidate_votes + rival_votes = total_votes) ∧
    (total_votes = 10000)) := by
  sorry

end election_vote_count_l3069_306985


namespace even_sum_probability_l3069_306930

/-- Represents a die with faces numbered from 1 to n -/
structure Die (n : ℕ) where
  faces : Finset ℕ
  face_count : faces.card = n
  valid_faces : ∀ x ∈ faces, 1 ≤ x ∧ x ≤ n

/-- A regular die with faces numbered from 1 to 6 -/
def regular_die : Die 6 := {
  faces := Finset.range 6,
  face_count := sorry,
  valid_faces := sorry
}

/-- An odd-numbered die with faces 1, 3, 5, 7, 9, 11 -/
def odd_die : Die 6 := {
  faces := Finset.range 6,
  face_count := sorry,
  valid_faces := sorry
}

/-- The probability of an event occurring -/
def probability (event : Prop) : ℚ := sorry

/-- The sum of the top faces of three dice -/
def dice_sum (d1 d2 : Die 6) (d3 : Die 6) : ℕ := sorry

/-- The statement to be proved -/
theorem even_sum_probability :
  probability (∃ (r1 r2 : Die 6) (o : Die 6), 
    r1 = regular_die ∧ 
    r2 = regular_die ∧ 
    o = odd_die ∧ 
    Even (dice_sum r1 r2 o)) = 1/2 := sorry

end even_sum_probability_l3069_306930


namespace angle_CBO_is_20_l3069_306942

-- Define the triangle ABC and point O
variable (A B C O : Point)

-- Define the angles as real numbers (in degrees)
variable (angle_BAO angle_CAO angle_CBO angle_ABO angle_ACO angle_BCO angle_AOC : ℝ)

-- State the theorem
theorem angle_CBO_is_20 
  (h1 : angle_BAO = angle_CAO)
  (h2 : angle_CBO = angle_ABO)
  (h3 : angle_ACO = angle_BCO)
  (h4 : angle_AOC = 110) :
  angle_CBO = 20 := by
    sorry

end angle_CBO_is_20_l3069_306942


namespace original_profit_margin_l3069_306951

theorem original_profit_margin (original_price selling_price : ℝ) : 
  original_price > 0 →
  selling_price > original_price →
  let new_price := 0.9 * original_price
  let original_margin := (selling_price - original_price) / original_price
  let new_margin := (selling_price - new_price) / new_price
  new_margin - original_margin = 0.12 →
  original_margin = 0.08 := by
sorry

end original_profit_margin_l3069_306951


namespace sarah_brings_nine_photos_l3069_306931

/-- The number of photos Sarah brings to fill a photo album -/
def sarahs_photos (total_slots : ℕ) (cristina_photos : ℕ) (john_photos : ℕ) (clarissa_photos : ℕ) : ℕ :=
  total_slots - (cristina_photos + john_photos + clarissa_photos)

/-- Theorem stating that Sarah brings 9 photos given the conditions in the problem -/
theorem sarah_brings_nine_photos :
  sarahs_photos 40 7 10 14 = 9 := by
  sorry

#eval sarahs_photos 40 7 10 14

end sarah_brings_nine_photos_l3069_306931


namespace quadratic_equations_solutions_l3069_306945

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ x₁^2 - 4*x₁ + 3 = 0 ∧ x₂^2 - 4*x₂ + 3 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = (1 + Real.sqrt 3) / 2 ∧ y₂ = (1 - Real.sqrt 3) / 2 ∧
    2*y₁^2 - 2*y₁ - 1 = 0 ∧ 2*y₂^2 - 2*y₂ - 1 = 0) :=
by sorry

end quadratic_equations_solutions_l3069_306945


namespace choose_three_cooks_from_ten_l3069_306987

theorem choose_three_cooks_from_ten (n : ℕ) (k : ℕ) : n = 10 ∧ k = 3 → Nat.choose n k = 120 := by
  sorry

end choose_three_cooks_from_ten_l3069_306987


namespace second_year_sample_size_l3069_306990

/-- Represents the ratio of students in first, second, and third grades -/
structure GradeRatio where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the number of students in a specific grade for a stratified sample -/
def stratifiedSampleSize (ratio : GradeRatio) (totalSample : ℕ) (grade : ℕ) : ℕ :=
  (totalSample * grade) / (ratio.first + ratio.second + ratio.third)

/-- Theorem: In a stratified sample of 240 students with a grade ratio of 5:4:3,
    the number of second-year students in the sample is 80 -/
theorem second_year_sample_size :
  let ratio : GradeRatio := { first := 5, second := 4, third := 3 }
  stratifiedSampleSize ratio 240 ratio.second = 80 := by
  sorry

end second_year_sample_size_l3069_306990


namespace quadratic_real_roots_range_l3069_306922

theorem quadratic_real_roots_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a = 0) ↔ a ≤ 1 := by sorry

end quadratic_real_roots_range_l3069_306922


namespace base_4_9_digit_difference_l3069_306914

theorem base_4_9_digit_difference (n : ℕ) : n = 1296 →
  (Nat.log 4 n + 1) - (Nat.log 9 n + 1) = 2 := by
  sorry

end base_4_9_digit_difference_l3069_306914


namespace dog_grouping_combinations_l3069_306999

def number_of_dogs : ℕ := 12
def group_sizes : List ℕ := [4, 5, 3]

def rocky_group : ℕ := 2
def nipper_group : ℕ := 1
def scruffy_group : ℕ := 0

def remaining_dogs : ℕ := number_of_dogs - 3

theorem dog_grouping_combinations : 
  (remaining_dogs.choose (group_sizes[rocky_group] - 1)) * 
  ((remaining_dogs - (group_sizes[rocky_group] - 1)).choose (group_sizes[scruffy_group] - 1)) = 1260 := by
  sorry

end dog_grouping_combinations_l3069_306999


namespace stratified_sampling_appropriate_l3069_306909

/-- Represents a sampling method -/
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a student population -/
structure Population where
  male_count : Nat
  female_count : Nat

/-- Represents a survey -/
structure Survey where
  sample_size : Nat
  method : SamplingMethod

/-- Determines if a sampling method is appropriate for a given population and survey -/
def is_appropriate_method (pop : Population) (survey : Survey) : Prop :=
  pop.male_count = pop.female_count ∧ 
  pop.male_count + pop.female_count > survey.sample_size ∧
  survey.method = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the appropriate method for the given scenario -/
theorem stratified_sampling_appropriate (pop : Population) (survey : Survey) :
  pop.male_count = 500 ∧ pop.female_count = 500 ∧ survey.sample_size = 100 →
  is_appropriate_method pop survey :=
sorry

end stratified_sampling_appropriate_l3069_306909


namespace m_gt_n_gt_0_neither_sufficient_nor_necessary_l3069_306935

/-- Represents an ellipse defined by the equation mx² + ny² = 1 --/
structure Ellipse (m n : ℝ) :=
  (equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1)

/-- Predicate to check if an ellipse has foci on the x-axis --/
def has_foci_on_x_axis (e : Ellipse m n) : Prop :=
  n > m ∧ m > 0

/-- The main theorem stating that m > n > 0 is neither sufficient nor necessary
    for an ellipse to have foci on the x-axis --/
theorem m_gt_n_gt_0_neither_sufficient_nor_necessary :
  ¬(∀ m n : ℝ, m > n ∧ n > 0 → (∀ e : Ellipse m n, has_foci_on_x_axis e)) ∧
  ¬(∀ m n : ℝ, (∀ e : Ellipse m n, has_foci_on_x_axis e) → m > n ∧ n > 0) :=
sorry

end m_gt_n_gt_0_neither_sufficient_nor_necessary_l3069_306935


namespace arithmetic_sequence_difference_l3069_306993

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_difference (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  (3 * a 6 = a 3 + a 4 + a 5 + 6) →
  d = 1 := by
sorry

end arithmetic_sequence_difference_l3069_306993


namespace x_positive_necessary_not_sufficient_l3069_306919

theorem x_positive_necessary_not_sufficient :
  (∃ x : ℝ, x > 0 ∧ (x - 2) * (x - 4) ≥ 0) ∧
  (∀ x : ℝ, (x - 2) * (x - 4) < 0 → x > 0) :=
by sorry

end x_positive_necessary_not_sufficient_l3069_306919


namespace final_value_of_A_l3069_306978

theorem final_value_of_A : ∀ (A : ℤ), A = 15 → -A + 5 = -10 := by
  sorry

end final_value_of_A_l3069_306978


namespace fraction_simplification_l3069_306998

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^10 + 15*x^5 + 125) / (x^5 + 5) = 248 + 25/62 := by
  sorry

end fraction_simplification_l3069_306998


namespace factory_uses_systematic_sampling_factory_sampling_is_systematic_l3069_306969

-- Define the characteristics of the sampling method
structure SamplingMethod where
  regular_intervals : Bool
  fixed_position : Bool
  continuous_process : Bool

-- Define Systematic Sampling
def SystematicSampling : SamplingMethod :=
  { regular_intervals := true
  , fixed_position := true
  , continuous_process := true }

-- Define the factory's sampling method
def FactorySamplingMethod : SamplingMethod :=
  { regular_intervals := true  -- Every 5 minutes
  , fixed_position := true     -- Fixed position on conveyor belt
  , continuous_process := true -- Conveyor belt process
  }

-- Theorem to prove
theorem factory_uses_systematic_sampling :
  FactorySamplingMethod = SystematicSampling := by
  sorry

-- Additional theorem to show that the factory's method is indeed Systematic Sampling
theorem factory_sampling_is_systematic :
  FactorySamplingMethod.regular_intervals ∧
  FactorySamplingMethod.fixed_position ∧
  FactorySamplingMethod.continuous_process := by
  sorry

end factory_uses_systematic_sampling_factory_sampling_is_systematic_l3069_306969


namespace sequence_bound_l3069_306907

theorem sequence_bound (a : ℕ → ℝ) 
  (h_pos : ∀ n, n ≥ 1 → a n > 0)
  (h_ineq : ∀ n, n ≥ 1 → (a (n + 1))^2 + (a n) * (a (n + 2)) ≤ a n + a (n + 2)) :
  a 2023 ≤ 1 := by
  sorry

end sequence_bound_l3069_306907


namespace range_of_a_l3069_306971

theorem range_of_a (π : ℝ) (h : π > 0) : 
  ∀ a : ℝ, (∃ x : ℝ, x < 0 ∧ (1/π)^x = (1+a)/(1-a)) → 0 < a ∧ a < 1 :=
sorry

end range_of_a_l3069_306971


namespace gasoline_tank_capacity_l3069_306936

theorem gasoline_tank_capacity : ∃ x : ℚ, 
  (3/4 : ℚ) * x - (1/3 : ℚ) * x = 18 ∧ x = 43.2 := by
  sorry

end gasoline_tank_capacity_l3069_306936


namespace quadratic_equation_solution_l3069_306941

theorem quadratic_equation_solution : 
  {x : ℝ | 2 * x^2 + 5 * x = 0} = {0, -5/2} :=
by sorry

end quadratic_equation_solution_l3069_306941


namespace mike_fred_salary_ratio_l3069_306979

/-- Proves that Mike earned 11 times more money than Fred five months ago -/
theorem mike_fred_salary_ratio :
  ∀ (fred_salary mike_salary_now : ℕ),
    fred_salary = 1000 →
    mike_salary_now = 15400 →
    ∃ (mike_salary_before : ℕ),
      mike_salary_now = (140 * mike_salary_before) / 100 ∧
      mike_salary_before = 11 * fred_salary :=
by sorry

end mike_fred_salary_ratio_l3069_306979


namespace smallest_number_l3069_306937

theorem smallest_number (S : Set ℤ) (h : S = {0, -1, -4, 5}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -4 :=
by sorry

end smallest_number_l3069_306937


namespace sum_of_fifth_powers_l3069_306953

theorem sum_of_fifth_powers (α β γ : ℂ) 
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 8) :
  α^5 + β^5 + γ^5 = 46.5 := by
  sorry

end sum_of_fifth_powers_l3069_306953


namespace evaluate_expression_l3069_306925

theorem evaluate_expression : 
  (128 : ℝ)^(1/3) * (729 : ℝ)^(1/2) = 108 * 2^(1/3) :=
by
  -- Definitions based on given conditions
  have h1 : (128 : ℝ) = 2^7 := by sorry
  have h2 : (729 : ℝ) = 3^6 := by sorry
  
  -- Proof goes here
  sorry

end evaluate_expression_l3069_306925


namespace symmetric_circle_equation_l3069_306977

/-- The equation of a circle symmetric to another circle with respect to the line y = x -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∃ (a b r : ℝ), (x + a)^2 + (y - b)^2 = r^2) →  -- Original circle
  (∃ (c d : ℝ), (x - c)^2 + (y + d)^2 = 5) :=     -- Symmetric circle
by sorry

end symmetric_circle_equation_l3069_306977


namespace pump_out_time_l3069_306933

/-- Calculates the time needed to pump out water from a flooded basement -/
theorem pump_out_time (length width depth : ℝ) (num_pumps pump_rate : ℝ) (conversion_rate : ℝ) :
  length = 20 ∧ 
  width = 40 ∧ 
  depth = 2 ∧ 
  num_pumps = 5 ∧ 
  pump_rate = 10 ∧ 
  conversion_rate = 7.5 →
  (length * width * depth * conversion_rate) / (num_pumps * pump_rate) = 240 := by
  sorry

end pump_out_time_l3069_306933


namespace faye_coloring_books_l3069_306957

/-- Calculates the total number of coloring books Faye has after giving some away and buying more. -/
def total_coloring_books (initial : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - given_away + bought

/-- Proves that Faye ends up with 79 coloring books given the initial conditions. -/
theorem faye_coloring_books : total_coloring_books 34 3 48 = 79 := by
  sorry

end faye_coloring_books_l3069_306957


namespace max_absolute_value_on_circle_l3069_306956

theorem max_absolute_value_on_circle (z : ℂ) (h : Complex.abs (z - (1 - Complex.I)) = 1) :
  Complex.abs z ≤ Real.sqrt 2 + 1 ∧ ∃ w : ℂ, Complex.abs (w - (1 - Complex.I)) = 1 ∧ Complex.abs w = Real.sqrt 2 + 1 :=
by sorry

end max_absolute_value_on_circle_l3069_306956


namespace average_weight_increase_l3069_306920

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem average_weight_increase
  (n : ℕ)                           -- number of people in the group
  (initial_weight : ℝ)              -- weight of the person being replaced
  (new_weight : ℝ)                  -- weight of the new person
  (h1 : n = 8)                      -- there are 8 people in the group
  (h2 : initial_weight = 55)        -- the initial person weighs 55 kg
  (h3 : new_weight = 75)            -- the new person weighs 75 kg
  : (new_weight - initial_weight) / n = 2.5 := by
  sorry

end average_weight_increase_l3069_306920


namespace thirteenth_number_with_digit_sum_12_l3069_306961

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 13th number with digit sum 12 is 174 -/
theorem thirteenth_number_with_digit_sum_12 : 
  nth_number_with_digit_sum_12 13 = 174 := by sorry

end thirteenth_number_with_digit_sum_12_l3069_306961


namespace equation_solution_l3069_306934

theorem equation_solution : ∀ x : ℝ, 3 * x + 15 = (1/3) * (6 * x + 45) → x - 5 = -5 := by
  sorry

end equation_solution_l3069_306934


namespace cheese_balls_per_serving_l3069_306958

/-- Given the information about cheese balls in barrels, calculate the number of cheese balls per serving -/
theorem cheese_balls_per_serving 
  (barrel_24oz : ℕ) 
  (barrel_35oz : ℕ) 
  (servings_24oz : ℕ) 
  (cheese_balls_35oz : ℕ) 
  (h1 : barrel_24oz = 24) 
  (h2 : barrel_35oz = 35) 
  (h3 : servings_24oz = 60) 
  (h4 : cheese_balls_35oz = 1050) : 
  (cheese_balls_35oz / barrel_35oz * barrel_24oz) / servings_24oz = 12 := by
  sorry


end cheese_balls_per_serving_l3069_306958


namespace quadratic_equation_result_l3069_306915

theorem quadratic_equation_result (x : ℝ) (h : x^2 - x + 3 = 0) : 
  (x - 3) * (x + 2) = -9 := by
sorry

end quadratic_equation_result_l3069_306915


namespace fish_fillet_distribution_l3069_306963

theorem fish_fillet_distribution (total : ℕ) (second_team : ℕ) (third_team : ℕ) 
  (h1 : total = 500)
  (h2 : second_team = 131)
  (h3 : third_team = 180) :
  total - (second_team + third_team) = 189 := by
  sorry

end fish_fillet_distribution_l3069_306963


namespace circle_properties_l3069_306938

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem circle_properties :
  -- The circle passes through the origin
  circle_equation 0 0 ∧
  -- The circle contains the point (2,0)
  circle_equation 2 0 ∧
  -- The line contains the point (2,0)
  line_equation 2 0 ∧
  -- The circle is tangent to the line at (2,0)
  ∃ (t : ℝ), t ≠ 0 ∧
    ∀ (x y : ℝ),
      circle_equation x y ∧ line_equation x y →
      x = 2 ∧ y = 0 :=
sorry

end circle_properties_l3069_306938


namespace probability_less_than_10_l3069_306959

theorem probability_less_than_10 (p_10_ring : ℝ) (h1 : p_10_ring = 0.22) :
  1 - p_10_ring = 0.78 := by
  sorry

end probability_less_than_10_l3069_306959


namespace wooden_toy_price_is_20_l3069_306911

/-- The price of a wooden toy at the Craftee And Best store -/
def wooden_toy_price : ℕ := sorry

/-- The price of a hat at the Craftee And Best store -/
def hat_price : ℕ := 10

/-- The amount Kendra initially had -/
def initial_amount : ℕ := 100

/-- The number of wooden toys Kendra bought -/
def wooden_toys_bought : ℕ := 2

/-- The number of hats Kendra bought -/
def hats_bought : ℕ := 3

/-- The amount Kendra received in change -/
def change_received : ℕ := 30

theorem wooden_toy_price_is_20 :
  wooden_toy_price = 20 :=
by
  sorry

end wooden_toy_price_is_20_l3069_306911


namespace trigonometric_equality_quadratic_equation_l3069_306905

theorem trigonometric_equality (x : ℝ) : 
  (1 - 2 * Real.sin x * Real.cos x) / (Real.cos x^2 - Real.sin x^2) = 
  (1 - Real.tan x) / (1 + Real.tan x) := by sorry

theorem quadratic_equation (θ a b : ℝ) :
  Real.tan θ + Real.sin θ = a ∧ Real.tan θ - Real.sin θ = b →
  (a^2 - b^2)^2 = 16 * a * b := by sorry

end trigonometric_equality_quadratic_equation_l3069_306905


namespace quadratic_equation_solution_l3069_306902

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + 8*x₁ = 9 ∧ x₂^2 + 8*x₂ = 9) ∧ 
  (x₁ = -9 ∧ x₂ = 1) := by
sorry

end quadratic_equation_solution_l3069_306902


namespace exists_sequence_iff_N_ge_4_l3069_306974

/-- A sequence of positive integers -/
def PositiveIntegerSequence := ℕ+ → ℕ+

/-- Strictly increasing sequence -/
def StrictlyIncreasing (s : PositiveIntegerSequence) : Prop :=
  ∀ n m : ℕ+, n < m → s n < s m

/-- The property that the sequence satisfies for a given N -/
def SatisfiesProperty (s : PositiveIntegerSequence) (N : ℝ) : Prop :=
  ∀ n : ℕ+, (s (2 * n - 1) + s (2 * n)) / s n = N

/-- The main theorem -/
theorem exists_sequence_iff_N_ge_4 (N : ℝ) : 
  (∃ s : PositiveIntegerSequence, StrictlyIncreasing s ∧ SatisfiesProperty s N) ↔ N ≥ 4 := by
  sorry

end exists_sequence_iff_N_ge_4_l3069_306974


namespace minimum_choir_size_l3069_306986

def is_valid_choir_size (n : ℕ) : Prop :=
  n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0

theorem minimum_choir_size :
  ∃ (n : ℕ), is_valid_choir_size n ∧ ∀ (m : ℕ), m < n → ¬ is_valid_choir_size m :=
by
  use 360
  sorry

end minimum_choir_size_l3069_306986


namespace probability_green_is_9_31_l3069_306927

/-- Represents the number of jelly beans of each color in the bag -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the total number of jelly beans in the bag -/
def totalJellyBeans (bag : JellyBeanBag) : ℕ :=
  bag.red + bag.green + bag.yellow + bag.blue

/-- Calculates the probability of selecting a green jelly bean -/
def probabilityGreen (bag : JellyBeanBag) : ℚ :=
  bag.green / totalJellyBeans bag

/-- The specific bag of jelly beans described in the problem -/
def specificBag : JellyBeanBag :=
  { red := 10, green := 9, yellow := 5, blue := 7 }

/-- Theorem stating that the probability of selecting a green jelly bean
    from the specific bag is 9/31 -/
theorem probability_green_is_9_31 :
  probabilityGreen specificBag = 9 / 31 := by
  sorry

end probability_green_is_9_31_l3069_306927
