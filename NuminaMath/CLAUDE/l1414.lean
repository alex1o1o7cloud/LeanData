import Mathlib

namespace coronavirus_cases_day3_l1414_141471

/-- Represents the number of Coronavirus cases over three days -/
structure CoronavirusCases where
  initial_cases : ℕ
  day2_increase : ℕ
  day2_recoveries : ℕ
  day3_recoveries : ℕ
  final_total : ℕ

/-- Calculates the number of new cases on day 3 -/
def new_cases_day3 (c : CoronavirusCases) : ℕ :=
  c.final_total - (c.initial_cases + c.day2_increase - c.day2_recoveries - c.day3_recoveries)

/-- Theorem stating that given the conditions, the number of new cases on day 3 is 1500 -/
theorem coronavirus_cases_day3 (c : CoronavirusCases) 
  (h1 : c.initial_cases = 2000)
  (h2 : c.day2_increase = 500)
  (h3 : c.day2_recoveries = 50)
  (h4 : c.day3_recoveries = 200)
  (h5 : c.final_total = 3750) :
  new_cases_day3 c = 1500 := by
  sorry

end coronavirus_cases_day3_l1414_141471


namespace more_birds_than_nests_l1414_141444

theorem more_birds_than_nests :
  let birds : ℕ := 6
  let nests : ℕ := 3
  birds - nests = 3 :=
by sorry

end more_birds_than_nests_l1414_141444


namespace time_to_fill_tank_l1414_141479

/-- Represents the tank and pipe system -/
structure TankSystem where
  capacity : ℝ
  pipeA_rate : ℝ
  pipeB_rate : ℝ
  pipeC_rate : ℝ
  pipeA_time : ℝ
  pipeB_time : ℝ
  pipeC_time : ℝ

/-- Calculates the net volume filled in one cycle -/
def netVolumeFilled (system : TankSystem) : ℝ :=
  system.pipeA_rate * system.pipeA_time +
  system.pipeB_rate * system.pipeB_time -
  system.pipeC_rate * system.pipeC_time

/-- Calculates the time for one cycle -/
def cycleTime (system : TankSystem) : ℝ :=
  system.pipeA_time + system.pipeB_time + system.pipeC_time

/-- Theorem stating the time to fill the tank -/
theorem time_to_fill_tank (system : TankSystem)
  (h1 : system.capacity = 2000)
  (h2 : system.pipeA_rate = 200)
  (h3 : system.pipeB_rate = 50)
  (h4 : system.pipeC_rate = 25)
  (h5 : system.pipeA_time = 1)
  (h6 : system.pipeB_time = 2)
  (h7 : system.pipeC_time = 2) :
  (system.capacity / netVolumeFilled system) * cycleTime system = 40 := by
  sorry

end time_to_fill_tank_l1414_141479


namespace bahs_to_yahs_conversion_l1414_141443

/-- The number of bahs in one rah -/
def bahs_per_rah : ℚ := 18 / 30

/-- The number of yahs in one rah -/
def yahs_per_rah : ℚ := 10 / 6

/-- Proves that 432 bahs are equal to 1200 yahs -/
theorem bahs_to_yahs_conversion : 
  432 * bahs_per_rah = 1200 / yahs_per_rah := by sorry

end bahs_to_yahs_conversion_l1414_141443


namespace dot_product_range_l1414_141485

theorem dot_product_range (a b : EuclideanSpace ℝ (Fin n)) :
  norm a = 2 →
  norm b = 2 →
  (∀ x : ℝ, norm (a + x • b) ≥ 1) →
  -2 * Real.sqrt 3 ≤ inner a b ∧ inner a b ≤ 2 * Real.sqrt 3 := by
  sorry

end dot_product_range_l1414_141485


namespace number_approximation_l1414_141430

-- Define the function f
def f (x : ℝ) : ℝ := x

-- Define the approximation relation
def approx (x y : ℝ) : Prop := abs (x - y) < 0.000000000000001

-- State the theorem
theorem number_approximation (x : ℝ) :
  approx (f (69.28 * 0.004) / x) 9.237333333333334 →
  approx x 0.03 :=
by
  sorry

end number_approximation_l1414_141430


namespace quadratic_root_sum_squares_l1414_141462

theorem quadratic_root_sum_squares (h : ℝ) : 
  (∃ r s : ℝ, r^2 + 2*h*r + 2 = 0 ∧ s^2 + 2*h*s + 2 = 0 ∧ r^2 + s^2 = 8) → 
  |h| = Real.sqrt 3 := by
sorry

end quadratic_root_sum_squares_l1414_141462


namespace f_maximum_properties_l1414_141495

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem f_maximum_properties (x₀ : ℝ) 
  (h₁ : ∀ x > 0, f x ≤ f x₀) 
  (h₂ : x₀ > 0) : 
  f x₀ = x₀ ∧ f x₀ < (1/2) := by
  sorry

end f_maximum_properties_l1414_141495


namespace fraction_equality_l1414_141401

theorem fraction_equality (x y : ℚ) (h : x / y = 2 / 7) : (x + y) / y = 9 / 7 := by
  sorry

end fraction_equality_l1414_141401


namespace percentage_difference_l1414_141470

theorem percentage_difference (x y : ℝ) (h : y = 1.8 * x) : 
  (x - y) / y * 100 = -(4 / 9) * 100 := by sorry

end percentage_difference_l1414_141470


namespace sandy_kim_age_multiple_l1414_141442

/-- Proves that Sandy will be 3 times as old as Kim in two years -/
theorem sandy_kim_age_multiple :
  ∀ (sandy_age kim_age : ℕ) (sandy_bill : ℕ),
    sandy_bill = 10 * sandy_age →
    sandy_bill = 340 →
    kim_age = 10 →
    (sandy_age + 2) / (kim_age + 2) = 3 :=
by
  sorry

end sandy_kim_age_multiple_l1414_141442


namespace imaginary_unit_equation_l1414_141438

theorem imaginary_unit_equation (a : ℝ) (h1 : a > 0) :
  Complex.abs ((a + Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 := by
  sorry

end imaginary_unit_equation_l1414_141438


namespace emma_coins_l1414_141428

theorem emma_coins (x : ℚ) (hx : x > 0) : 
  let lost := x / 3
  let found := (3 / 4) * lost
  x - (x - lost + found) = x / 12 := by sorry

end emma_coins_l1414_141428


namespace ratio_problem_l1414_141463

theorem ratio_problem (first_part : ℝ) (ratio_percent : ℝ) (second_part : ℝ) :
  first_part = 25 →
  ratio_percent = 50 →
  first_part / (first_part + second_part) * 100 = ratio_percent →
  second_part = 25 := by
  sorry

end ratio_problem_l1414_141463


namespace triangle_inequality_l1414_141411

theorem triangle_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end triangle_inequality_l1414_141411


namespace smaller_root_of_quadratic_l1414_141460

theorem smaller_root_of_quadratic (x : ℝ) : 
  (x - 2/3)^2 + (x - 2/3)*(x - 1/3) = 0 → 
  (x = 1/2 ∨ x = 2/3) ∧ 1/2 < 2/3 := by
  sorry

end smaller_root_of_quadratic_l1414_141460


namespace triangle_side_length_l1414_141406

/-- Given a triangle ABC with sides a, b, c and angle C, 
    prove that c = √19 when a + b = 5, ab = 2, and C = 60° --/
theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a + b = 5 → ab = 2 → C = π / 3 → c = Real.sqrt 19 := by
  sorry

end triangle_side_length_l1414_141406


namespace odd_factors_of_360_l1414_141476

/-- The number of odd factors of 360 -/
def num_odd_factors_360 : ℕ := sorry

/-- 360 is the number we're considering -/
def n : ℕ := 360

theorem odd_factors_of_360 : num_odd_factors_360 = 6 := by sorry

end odd_factors_of_360_l1414_141476


namespace max_min_difference_z_l1414_141416

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (sum_squares_eq : x^2 + y^2 + z^2 = 18) : 
  ∃ (z_max z_min : ℝ), 
    (∀ w : ℝ, (∃ u v : ℝ, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 18) → w ≤ z_max) ∧
    (∀ w : ℝ, (∃ u v : ℝ, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 18) → w ≥ z_min) ∧
    z_max - z_min = 6 :=
by sorry

end max_min_difference_z_l1414_141416


namespace paige_folders_l1414_141475

def initial_files : ℕ := 135
def deleted_files : ℕ := 27
def files_per_folder : ℚ := 8.5

theorem paige_folders : 
  ∃ (folders : ℕ), 
    folders = (initial_files - deleted_files : ℚ) / files_per_folder
    ∧ folders = 13 := by sorry

end paige_folders_l1414_141475


namespace ratio_of_percentages_l1414_141461

theorem ratio_of_percentages (A B C D : ℝ) 
  (hA : A = 0.4 * B) 
  (hB : B = 0.25 * C) 
  (hD : D = 0.6 * C) 
  (hC : C ≠ 0) : A / D = 1 / 6 := by
  sorry

end ratio_of_percentages_l1414_141461


namespace base9_726_to_base3_l1414_141433

/-- Converts a digit from base 9 to two digits in base 3 -/
def base9ToBase3Digit (d : Nat) : Nat × Nat :=
  sorry

/-- Converts a number from base 9 to base 3 -/
def base9ToBase3 (n : Nat) : Nat :=
  sorry

theorem base9_726_to_base3 :
  base9ToBase3 726 = 210220 :=
sorry

end base9_726_to_base3_l1414_141433


namespace quadratic_function_range_l1414_141415

/-- A quadratic function with a positive leading coefficient -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_range
  (f : ℝ → ℝ)
  (h1 : QuadraticFunction f)
  (h2 : ∀ x : ℝ, f x = f (4 - x))
  (h3 : ∀ x : ℝ, f (1 - 2*x^2) < f (1 + 2*x - x^2)) :
  ∀ x : ℝ, f (1 - 2*x^2) < f (1 + 2*x - x^2) → -2 < x ∧ x < 0 :=
by sorry

end quadratic_function_range_l1414_141415


namespace wendy_shoes_theorem_l1414_141427

/-- The number of pairs of shoes Wendy gave away -/
def shoes_given_away (total : ℕ) (left : ℕ) : ℕ := total - left

/-- Theorem stating that Wendy gave away 14 pairs of shoes -/
theorem wendy_shoes_theorem (total : ℕ) (left : ℕ) 
  (h1 : total = 33) 
  (h2 : left = 19) : 
  shoes_given_away total left = 14 := by
  sorry

end wendy_shoes_theorem_l1414_141427


namespace math_majors_consecutive_probability_l1414_141472

/-- The number of people sitting at the table -/
def total_people : ℕ := 12

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 4

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 3

/-- The probability of all math majors sitting consecutively -/
def prob_consecutive_math : ℚ := 1 / 66

theorem math_majors_consecutive_probability :
  (total_people : ℚ) / (total_people.choose math_majors) = prob_consecutive_math := by
  sorry

end math_majors_consecutive_probability_l1414_141472


namespace find_k_l1414_141497

/-- The function f(x) -/
def f (k a x : ℝ) : ℝ := 2*k + (k^3)*a - x

/-- The function g(x) -/
def g (k a x : ℝ) : ℝ := x^2 + f k a x

theorem find_k (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ k : ℝ,
    (∀ x : ℝ, f k a x = -f k a (-x)) ∧  -- f is odd
    (∃ x : ℝ, f k a x = 3) ∧  -- f = 3 for some x
    (∀ x : ℝ, x ≥ 2 → g k a x ≥ -2) ∧  -- g has minimum -2 on [2, +∞)
    (∃ x : ℝ, x ≥ 2 ∧ g k a x = -2) ∧  -- g achieves minimum -2 on [2, +∞)
    k = 1 :=
  sorry

end find_k_l1414_141497


namespace average_of_25_results_l1414_141432

theorem average_of_25_results (first_12_avg : ℝ) (last_12_avg : ℝ) (result_13 : ℝ) 
  (h1 : first_12_avg = 14)
  (h2 : last_12_avg = 17)
  (h3 : result_13 = 228) :
  (12 * first_12_avg + result_13 + 12 * last_12_avg) / 25 = 24 := by
  sorry

end average_of_25_results_l1414_141432


namespace tournament_sequences_l1414_141447

/-- Represents a team in the tournament -/
structure Team :=
  (players : Finset ℕ)
  (size : players.card = 7)

/-- Represents a tournament between two teams -/
structure Tournament :=
  (teamA : Team)
  (teamB : Team)

/-- Represents a sequence of matches in the tournament -/
def MatchSequence (t : Tournament) := Finset ℕ

/-- The number of possible match sequences in a tournament -/
def numSequences (t : Tournament) : ℕ := Nat.choose 14 7

/-- Theorem: The number of possible match sequences in a tournament
    between two teams of 7 players each is equal to C(14,7) -/
theorem tournament_sequences (t : Tournament) :
  numSequences t = 3432 :=
by sorry

end tournament_sequences_l1414_141447


namespace cistern_wet_surface_area_l1414_141446

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width +  -- bottom area
  2 * length * depth +  -- long sides area
  2 * width * depth  -- short sides area

/-- Theorem: The total wet surface area of a cistern with given dimensions is 62 square meters -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 4 1.25 = 62 := by
  sorry


end cistern_wet_surface_area_l1414_141446


namespace initial_blue_balls_l1414_141481

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 18 → removed = 3 → prob = 1/5 → 
  ∃ (initial_blue : ℕ), 
    initial_blue = 6 ∧ 
    (initial_blue - removed : ℚ) / (total - removed) = prob :=
by sorry

end initial_blue_balls_l1414_141481


namespace rate_percent_is_twelve_l1414_141490

/-- Calculates the rate percent on simple interest given principal, amount, and time. -/
def calculate_rate_percent (principal amount : ℚ) (time : ℕ) : ℚ :=
  let simple_interest := amount - principal
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that the rate percent on simple interest is 12% for the given conditions. -/
theorem rate_percent_is_twelve :
  let principal : ℚ := 750
  let amount : ℚ := 1200
  let time : ℕ := 5
  calculate_rate_percent principal amount time = 12 := by
  sorry

#eval calculate_rate_percent 750 1200 5

end rate_percent_is_twelve_l1414_141490


namespace factorial_prime_factorization_l1414_141405

theorem factorial_prime_factorization (x i a m p : ℕ) : 
  x = (List.range 8).foldl (· * ·) 1 →
  x = 2^i * 3^a * 5^m * 7^p →
  i + a + m + p = 11 →
  a = 2 := by
  sorry

end factorial_prime_factorization_l1414_141405


namespace factor_x6_minus_64_l1414_141402

theorem factor_x6_minus_64 (x : ℝ) : 
  x^6 - 64 = (x - 2) * (x + 2) * (x^2 + 2*x + 4) * (x^2 - 2*x + 4) := by
  sorry

end factor_x6_minus_64_l1414_141402


namespace p_amount_l1414_141434

theorem p_amount : ∃ (p : ℚ), p = 49 ∧ p = (2 * (1/7) * p + 35) := by
  sorry

end p_amount_l1414_141434


namespace box_weight_is_42_l1414_141417

/-- The weight of a box of books -/
def box_weight (book_weight : ℕ) (num_books : ℕ) : ℕ :=
  book_weight * num_books

/-- Theorem: The weight of a box of books is 42 pounds -/
theorem box_weight_is_42 :
  box_weight 3 14 = 42 := by
  sorry

end box_weight_is_42_l1414_141417


namespace parallel_vectors_l1414_141429

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b (t : ℝ) : ℝ × ℝ := (3, t)

-- Define the parallel condition
def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem parallel_vectors (t : ℝ) :
  is_parallel (b t) (a.1 + (b t).1, a.2 + (b t).2) → t = -3 :=
by sorry

end parallel_vectors_l1414_141429


namespace largest_ball_radius_is_four_l1414_141424

/-- Represents a torus in 3D space -/
structure Torus where
  inner_radius : ℝ
  outer_radius : ℝ
  circle_center : ℝ × ℝ × ℝ
  circle_radius : ℝ

/-- Represents a spherical ball in 3D space -/
structure SphericalBall where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The largest spherical ball that can be placed on top of a torus -/
def largest_ball_on_torus (t : Torus) : SphericalBall :=
  { center := (0, 0, 4),
    radius := 4 }

/-- Theorem stating that the largest ball on the torus has radius 4 -/
theorem largest_ball_radius_is_four (t : Torus) 
  (h1 : t.inner_radius = 3)
  (h2 : t.outer_radius = 5)
  (h3 : t.circle_center = (4, 0, 1))
  (h4 : t.circle_radius = 1) :
  (largest_ball_on_torus t).radius = 4 := by
  sorry

#check largest_ball_radius_is_four

end largest_ball_radius_is_four_l1414_141424


namespace necessary_but_not_sufficient_l1414_141440

theorem necessary_but_not_sufficient :
  ∀ x : ℝ,
  (x + 2 = 0 → x^2 - 4 = 0) ∧
  ¬(x^2 - 4 = 0 → x + 2 = 0) :=
by sorry

end necessary_but_not_sufficient_l1414_141440


namespace parallelogram_area_l1414_141484

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 12 inches and 20 inches is equal to 120 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) : 
  a = 12 → b = 20 → θ = 150 * π / 180 → 
  a * b * Real.sin θ = 120 := by sorry

end parallelogram_area_l1414_141484


namespace kevin_cards_l1414_141419

theorem kevin_cards (initial_cards lost_cards : ℝ) 
  (h1 : initial_cards = 47.0)
  (h2 : lost_cards = 7.0) : 
  initial_cards - lost_cards = 40.0 := by
  sorry

end kevin_cards_l1414_141419


namespace inequality_proof_l1414_141425

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hne : ¬(a = b ∧ b = c)) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) := by
sorry

end inequality_proof_l1414_141425


namespace pole_length_after_cut_l1414_141456

theorem pole_length_after_cut (original_length : ℝ) (cut_percentage : ℝ) (new_length : ℝ) : 
  original_length = 20 →
  cut_percentage = 30 →
  new_length = original_length * (1 - cut_percentage / 100) →
  new_length = 14 :=
by sorry

end pole_length_after_cut_l1414_141456


namespace cyclic_inequality_l1414_141441

theorem cyclic_inequality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (y + z) / (2 * x) + (z + x) / (2 * y) + (x + y) / (2 * z) ≥ 
  2 * x / (y + z) + 2 * y / (z + x) + 2 * z / (x + y) := by
  sorry

end cyclic_inequality_l1414_141441


namespace smallest_possible_b_l1414_141474

theorem smallest_possible_b (a b : ℝ) 
  (h1 : 2 < a ∧ a < b) 
  (h2 : 2 + a ≤ b) 
  (h3 : 1/a + 1/b ≤ 1/2) : 
  b ≥ 3 + Real.sqrt 5 := by
  sorry

end smallest_possible_b_l1414_141474


namespace polygon_exterior_angles_l1414_141496

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 36) → (n * exterior_angle = 360) → n = 10 :=
by sorry

end polygon_exterior_angles_l1414_141496


namespace number_problem_l1414_141459

theorem number_problem (x : ℝ) : (0.2 * x = 0.2 * 650 + 190) → x = 1600 := by
  sorry

end number_problem_l1414_141459


namespace function_domain_range_sum_l1414_141452

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Define the domain and range
def is_valid_domain_and_range (m n : ℝ) : Prop :=
  (∀ x, m ≤ x ∧ x ≤ n → 3*m ≤ f x ∧ f x ≤ 3*n) ∧
  (∃ x, m ≤ x ∧ x ≤ n ∧ f x = 3*m) ∧
  (∃ x, m ≤ x ∧ x ≤ n ∧ f x = 3*n)

-- State the theorem
theorem function_domain_range_sum :
  ∃ m n : ℝ, is_valid_domain_and_range m n ∧ m = -1 ∧ n = 0 ∧ m + n = -1 :=
sorry

end function_domain_range_sum_l1414_141452


namespace exponential_inequality_range_l1414_141431

theorem exponential_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (2 : ℝ)^(x^2 - 4*x) > (2 : ℝ)^(2*a*x + a)) ↔ -4 < a ∧ a < -1 := by
  sorry

end exponential_inequality_range_l1414_141431


namespace routes_in_3x3_grid_l1414_141450

/-- The number of routes from top-left to bottom-right in a 3x3 grid -/
def number_of_routes : ℕ := 20

/-- The size of the grid -/
def grid_size : ℕ := 3

/-- The total number of moves required to reach the bottom-right corner -/
def total_moves : ℕ := 2 * grid_size

/-- The number of right moves (or down moves) required -/
def moves_in_one_direction : ℕ := grid_size

theorem routes_in_3x3_grid : 
  number_of_routes = Nat.choose total_moves moves_in_one_direction := by
  sorry

end routes_in_3x3_grid_l1414_141450


namespace unique_prime_divisibility_l1414_141455

theorem unique_prime_divisibility : 
  ∀ p : ℕ, Prime p → 
  (p = 3 ↔ 
    ∃! a : ℕ, a ∈ Finset.range p ∧ 
    p ∣ (a^3 - 3*a + 1)) := by sorry

end unique_prime_divisibility_l1414_141455


namespace range_of_expression_l1414_141483

theorem range_of_expression (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) :
  0 < x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by sorry

end range_of_expression_l1414_141483


namespace reroll_one_die_probability_l1414_141409

def dice_sum (d1 d2 d3 : Nat) : Nat := d1 + d2 + d3

def is_valid_die (d : Nat) : Prop := 1 ≤ d ∧ d ≤ 6

def reroll_one_probability : ℚ :=
  let total_outcomes : Nat := 6^3
  let favorable_outcomes : Nat := 19 * 6
  favorable_outcomes / total_outcomes

theorem reroll_one_die_probability :
  ∀ (d1 d2 d3 : Nat),
    is_valid_die d1 → is_valid_die d2 → is_valid_die d3 →
    (∃ (r : Nat), is_valid_die r ∧ dice_sum d1 d2 r = 9 ∨
                  dice_sum d1 r d3 = 9 ∨
                  dice_sum r d2 d3 = 9) →
    reroll_one_probability = 19/216 :=
by sorry

end reroll_one_die_probability_l1414_141409


namespace expression_evaluation_l1414_141454

theorem expression_evaluation : (5 + 2 + 6) * 2 / 3 - 4 / 3 = 22 / 3 := by
  sorry

end expression_evaluation_l1414_141454


namespace sequence_b_decreasing_l1414_141408

/-- Given a sequence {a_n} that satisfies the following conditions:
    1) a_1 = 2
    2) 2 * a_n * a_{n+1} = a_n^2 + 1
    Define b_n = (a_n - 1) / (a_n + 1)
    Then the sequence {b_n} is decreasing. -/
theorem sequence_b_decreasing (a : ℕ → ℝ) (b : ℕ → ℝ) :
  a 1 = 2 ∧
  (∀ n : ℕ, 2 * a n * a (n + 1) = a n ^ 2 + 1) ∧
  (∀ n : ℕ, b n = (a n - 1) / (a n + 1)) →
  ∀ n : ℕ, b (n + 1) < b n :=
by sorry

end sequence_b_decreasing_l1414_141408


namespace chess_probabilities_l1414_141468

theorem chess_probabilities (p_draw p_b_win : ℝ) 
  (h_draw : p_draw = 1/2)
  (h_b_win : p_b_win = 1/3) :
  let p_a_win := 1 - p_draw - p_b_win
  let p_a_not_lose := p_draw + p_a_win
  (p_a_win = 1/6) ∧ (p_a_not_lose = 2/3) := by
  sorry

end chess_probabilities_l1414_141468


namespace vector_at_t_4_l1414_141473

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  point : ℝ → (ℝ × ℝ × ℝ)

/-- The given line satisfying the conditions -/
def given_line : ParametricLine :=
  { point := sorry }

theorem vector_at_t_4 :
  given_line.point 1 = (4, 5, 9) →
  given_line.point 3 = (1, 0, -2) →
  given_line.point 4 = (-1, 0, -15) :=
by sorry

end vector_at_t_4_l1414_141473


namespace probability_three_defective_before_two_good_l1414_141413

/-- Represents the number of good products in the box -/
def goodProducts : ℕ := 9

/-- Represents the number of defective products in the box -/
def defectiveProducts : ℕ := 3

/-- Represents the total number of products in the box -/
def totalProducts : ℕ := goodProducts + defectiveProducts

/-- Calculates the probability of selecting 3 defective products before 2 good products -/
def probabilityThreeDefectiveBeforeTwoGood : ℚ :=
  (4 : ℚ) / 55

/-- Theorem stating that the probability of selecting 3 defective products
    before 2 good products is 4/55 -/
theorem probability_three_defective_before_two_good :
  probabilityThreeDefectiveBeforeTwoGood = (4 : ℚ) / 55 := by
  sorry

#eval probabilityThreeDefectiveBeforeTwoGood

end probability_three_defective_before_two_good_l1414_141413


namespace number_of_lineups_l1414_141418

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def cant_play_together : ℕ := 3
def injured : ℕ := 1

theorem number_of_lineups :
  (Nat.choose (team_size - cant_play_together - injured) lineup_size) +
  (cant_play_together * Nat.choose (team_size - cant_play_together - injured) (lineup_size - 1)) = 1452 :=
by sorry

end number_of_lineups_l1414_141418


namespace school_boys_count_l1414_141467

theorem school_boys_count (boys girls : ℕ) : 
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 128 →
  boys = 80 := by
sorry

end school_boys_count_l1414_141467


namespace tan_two_fifths_pi_plus_theta_l1414_141465

theorem tan_two_fifths_pi_plus_theta (θ : ℝ) 
  (h : Real.sin ((12 / 5) * Real.pi + θ) + 2 * Real.sin ((11 / 10) * Real.pi - θ) = 0) : 
  Real.tan ((2 / 5) * Real.pi + θ) = 2 := by
sorry

end tan_two_fifths_pi_plus_theta_l1414_141465


namespace line_through_vectors_l1414_141491

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem line_through_vectors (a b : V) (k : ℝ) (h : a ≠ b) :
  (∃ t : ℝ, k • a + (2/3 : ℝ) • b = a + t • (b - a)) →
  k = 1/3 := by
sorry

end line_through_vectors_l1414_141491


namespace pokemon_cards_cost_l1414_141487

/-- The cost of a pack of Pokemon cards -/
def pokemon_cost (football_pack_cost baseball_deck_cost total_cost : ℚ) : ℚ :=
  total_cost - (2 * football_pack_cost + baseball_deck_cost)

/-- Theorem: The cost of the Pokemon cards is $4.01 -/
theorem pokemon_cards_cost : 
  pokemon_cost 2.73 8.95 18.42 = 4.01 := by
  sorry

end pokemon_cards_cost_l1414_141487


namespace monotonic_increasing_interval_f_l1414_141498

noncomputable def f (x : ℝ) := (x - 2) * Real.exp x

theorem monotonic_increasing_interval_f :
  {x : ℝ | ∀ y, x < y → f x < f y} = {x : ℝ | x > 1} := by sorry

end monotonic_increasing_interval_f_l1414_141498


namespace balls_sold_l1414_141426

theorem balls_sold (selling_price : ℕ) (cost_price : ℕ) (loss : ℕ) : 
  selling_price = 720 →
  loss = 5 * cost_price →
  cost_price = 120 →
  selling_price + loss = 11 * cost_price :=
by
  sorry

end balls_sold_l1414_141426


namespace shop_width_calculation_l1414_141482

/-- Calculates the width of a shop given its monthly rent, length, and annual rent per square foot. -/
theorem shop_width_calculation (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_sqft : ℕ) : 
  monthly_rent = 1440 → length = 18 → annual_rent_per_sqft = 48 → 
  (monthly_rent * 12) / (annual_rent_per_sqft * length) = 20 := by
  sorry

end shop_width_calculation_l1414_141482


namespace perfect_cube_implies_one_l1414_141457

theorem perfect_cube_implies_one (a : ℕ) 
  (h : ∀ n : ℕ, ∃ k : ℕ, 4 * (a^n + 1) = k^3) : 
  a = 1 := by
  sorry

end perfect_cube_implies_one_l1414_141457


namespace ramesh_investment_l1414_141499

def suresh_investment : ℕ := 24000
def total_profit : ℕ := 19000
def ramesh_profit_share : ℕ := 11875

theorem ramesh_investment :
  ∃ (ramesh_investment : ℕ),
    (ramesh_investment * suresh_investment * ramesh_profit_share
     = (total_profit - ramesh_profit_share) * suresh_investment * total_profit)
    ∧ ramesh_investment = 42000 :=
by sorry

end ramesh_investment_l1414_141499


namespace condition_not_well_defined_l1414_141412

-- Define a type for students
structure Student :=
  (height : ℝ)
  (school : String)

-- Define a type for conditions
inductive Condition
  | TallStudents : Condition
  | PointsAwayFromOrigin : Condition
  | PrimesLessThan100 : Condition
  | QuadraticEquationSolutions : Condition

-- Define a predicate for well-defined sets
def IsWellDefinedSet (c : Condition) : Prop :=
  match c with
  | Condition.TallStudents => false
  | Condition.PointsAwayFromOrigin => true
  | Condition.PrimesLessThan100 => true
  | Condition.QuadraticEquationSolutions => true

-- Theorem statement
theorem condition_not_well_defined :
  ∃ c : Condition, ¬(IsWellDefinedSet c) ∧
  ∀ c' : Condition, c' ≠ c → IsWellDefinedSet c' :=
sorry

end condition_not_well_defined_l1414_141412


namespace min_max_sum_l1414_141421

theorem min_max_sum (a b c d e f : ℕ+) 
  (sum_eq : a + b + c + d + e + f = 4020) : 
  804 ≤ max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f)))) := by
  sorry

end min_max_sum_l1414_141421


namespace flour_cups_needed_l1414_141435

-- Define the total number of 1/4 cup scoops
def total_scoops : ℚ := 15

-- Define the amounts of other ingredients in cups
def white_sugar : ℚ := 1
def brown_sugar : ℚ := 1/4
def oil : ℚ := 1/2

-- Define the conversion factor from scoops to cups
def scoops_to_cups : ℚ := 1/4

-- Theorem to prove
theorem flour_cups_needed :
  let other_ingredients_scoops := white_sugar / scoops_to_cups + brown_sugar / scoops_to_cups + oil / scoops_to_cups
  let flour_scoops := total_scoops - other_ingredients_scoops
  let flour_cups := flour_scoops * scoops_to_cups
  flour_cups = 2 := by sorry

end flour_cups_needed_l1414_141435


namespace cookies_needed_to_fill_bags_l1414_141469

/-- Represents the number of cookies needed to fill a bag completely -/
def bagCapacity : ℕ := 16

/-- Represents the total number of cookies Edgar bought -/
def totalCookies : ℕ := 292

/-- Represents the number of chocolate chip cookies Edgar bought -/
def chocolateChipCookies : ℕ := 154

/-- Represents the number of oatmeal raisin cookies Edgar bought -/
def oatmealRaisinCookies : ℕ := 86

/-- Represents the number of sugar cookies Edgar bought -/
def sugarCookies : ℕ := 52

/-- Calculates the number of additional cookies needed to fill the last bag completely -/
def additionalCookiesNeeded (cookieCount : ℕ) : ℕ :=
  bagCapacity - (cookieCount % bagCapacity)

theorem cookies_needed_to_fill_bags :
  additionalCookiesNeeded chocolateChipCookies = 6 ∧
  additionalCookiesNeeded oatmealRaisinCookies = 10 ∧
  additionalCookiesNeeded sugarCookies = 12 :=
by
  sorry

#check cookies_needed_to_fill_bags

end cookies_needed_to_fill_bags_l1414_141469


namespace harry_friday_speed_l1414_141422

-- Define Harry's running speeds
def monday_speed : ℝ := 10
def tuesday_to_thursday_increase : ℝ := 0.5
def friday_increase : ℝ := 0.6

-- Define the function to calculate speed increase
def speed_increase (base_speed : ℝ) (increase_percentage : ℝ) : ℝ :=
  base_speed * (1 + increase_percentage)

-- Theorem statement
theorem harry_friday_speed :
  let tuesday_to_thursday_speed := speed_increase monday_speed tuesday_to_thursday_increase
  let friday_speed := speed_increase tuesday_to_thursday_speed friday_increase
  friday_speed = 24 := by sorry

end harry_friday_speed_l1414_141422


namespace problem_one_l1414_141410

theorem problem_one : (1 / 3)⁻¹ + Real.sqrt 12 - |Real.sqrt 3 - 2| - (π - 2023)^0 = 3 * Real.sqrt 3 := by
  sorry

end problem_one_l1414_141410


namespace inequality_system_solution_l1414_141480

theorem inequality_system_solution (x : ℝ) :
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4 → -2 < x ∧ x ≤ 1 := by
  sorry

end inequality_system_solution_l1414_141480


namespace braking_distance_at_120_less_than_33_braking_distance_at_40_equals_10_braking_distance_linear_and_nonnegative_l1414_141466

/-- Represents the braking distance function for a car -/
def brakingDistance (speed : ℝ) : ℝ :=
  0.25 * speed

/-- Theorem: The braking distance at 120 km/h is less than 33m -/
theorem braking_distance_at_120_less_than_33 :
  brakingDistance 120 < 33 := by
  sorry

/-- Theorem: The braking distance at 40 km/h is 10m -/
theorem braking_distance_at_40_equals_10 :
  brakingDistance 40 = 10 := by
  sorry

/-- Theorem: The braking distance function is linear and non-negative for non-negative speeds -/
theorem braking_distance_linear_and_nonnegative :
  ∀ (speed : ℝ), speed ≥ 0 → brakingDistance speed ≥ 0 ∧ 
  ∀ (speed1 speed2 : ℝ), brakingDistance (speed1 + speed2) = brakingDistance speed1 + brakingDistance speed2 := by
  sorry

end braking_distance_at_120_less_than_33_braking_distance_at_40_equals_10_braking_distance_linear_and_nonnegative_l1414_141466


namespace max_value_is_eight_l1414_141477

/-- The feasible region defined by the given constraints -/
def FeasibleRegion (x y : ℝ) : Prop :=
  x + y - 7 ≤ 0 ∧ x - 3*y + 1 ≤ 0 ∧ 3*x - y - 5 ≥ 0

/-- The objective function to be maximized -/
def ObjectiveFunction (x y : ℝ) : ℝ :=
  2*x - y

/-- Theorem stating that the maximum value of the objective function is 8 -/
theorem max_value_is_eight :
  ∃ (x y : ℝ), FeasibleRegion x y ∧
    ∀ (x' y' : ℝ), FeasibleRegion x' y' →
      ObjectiveFunction x y ≥ ObjectiveFunction x' y' ∧
      ObjectiveFunction x y = 8 :=
by sorry

end max_value_is_eight_l1414_141477


namespace infinitely_many_odd_floor_squares_l1414_141489

theorem infinitely_many_odd_floor_squares (α : ℝ) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, Odd ⌊n^2 * α⌋ :=
sorry

end infinitely_many_odd_floor_squares_l1414_141489


namespace average_weight_calculation_l1414_141453

theorem average_weight_calculation (num_men num_women : ℕ) (avg_weight_men avg_weight_women : ℚ) :
  num_men = 8 →
  num_women = 6 →
  avg_weight_men = 170 →
  avg_weight_women = 130 →
  let total_weight := num_men * avg_weight_men + num_women * avg_weight_women
  let total_people := num_men + num_women
  abs ((total_weight / total_people) - 153) < 1 := by
sorry

end average_weight_calculation_l1414_141453


namespace arithmetic_sequence_properties_l1414_141494

-- Define the arithmetic sequence a_n
def a (n : ℕ+) : ℚ :=
  sorry

-- Define the sum S_n of the first n terms of a_n
def S (n : ℕ+) : ℚ :=
  sorry

-- Define the sequence b_n
def b (n : ℕ+) : ℚ :=
  1 / (a n ^ 2 - 1)

-- Define the sum T_n of the first n terms of b_n
def T (n : ℕ+) : ℚ :=
  sorry

theorem arithmetic_sequence_properties :
  (a 3 = 6) ∧
  (a 5 + a 7 = 24) ∧
  (∀ n : ℕ+, a n = 2 * n) ∧
  (∀ n : ℕ+, S n = n^2 + n) ∧
  (∀ n : ℕ+, T n = n / (2 * n + 1)) :=
by sorry

end arithmetic_sequence_properties_l1414_141494


namespace hypotenuse_length_is_5_sqrt_211_l1414_141420

/-- Right triangle ABC with specific properties -/
structure RightTriangle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  -- AB and AC are legs of the right triangle
  ab_leg : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- X is on AB
  x_on_ab : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  -- Y is on AC
  y_on_ac : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Y = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- AX:XB = 2:3
  ax_xb_ratio : dist A X / dist X B = 2 / 3
  -- AY:YC = 2:3
  ay_yc_ratio : dist A Y / dist Y C = 2 / 3
  -- BY = 18 units
  by_length : dist B Y = 18
  -- CX = 15 units
  cx_length : dist C X = 15

/-- The length of hypotenuse BC in the right triangle -/
def hypotenuseLength (t : RightTriangle) : ℝ :=
  dist t.B t.C

/-- Theorem: The length of hypotenuse BC is 5√211 units -/
theorem hypotenuse_length_is_5_sqrt_211 (t : RightTriangle) :
  hypotenuseLength t = 5 * Real.sqrt 211 := by
  sorry


end hypotenuse_length_is_5_sqrt_211_l1414_141420


namespace venus_hall_rental_cost_prove_venus_hall_rental_cost_l1414_141492

/-- The rental cost of Venus Hall, given the conditions of the prom venue problem -/
theorem venus_hall_rental_cost : ℝ → Prop :=
  fun v =>
    let caesars_total : ℝ := 800 + 60 * 30
    let venus_total : ℝ := v + 60 * 35
    caesars_total = venus_total →
    v = 500

/-- Proof of the venus_hall_rental_cost theorem -/
theorem prove_venus_hall_rental_cost : ∃ v, venus_hall_rental_cost v :=
  sorry

end venus_hall_rental_cost_prove_venus_hall_rental_cost_l1414_141492


namespace salary_change_l1414_141464

/-- Proves that when a salary is increased by 10% and then reduced by 10%, 
    the net change is a decrease of 1% of the original salary. -/
theorem salary_change (S : ℝ) : 
  (S + S * (10 / 100)) * (1 - 10 / 100) = S * 0.99 := by
  sorry

end salary_change_l1414_141464


namespace sales_tax_percentage_l1414_141451

theorem sales_tax_percentage (total_before_tax : ℝ) (total_with_tax : ℝ) : 
  total_before_tax = 150 → 
  total_with_tax = 162 → 
  (total_with_tax - total_before_tax) / total_before_tax * 100 = 8 := by
  sorry

end sales_tax_percentage_l1414_141451


namespace min_value_of_x_plus_four_over_x_squared_min_value_achieved_l1414_141445

theorem min_value_of_x_plus_four_over_x_squared (x : ℝ) (h : x > 0) :
  x + 4 / x^2 ≥ 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 0) :
  x + 4 / x^2 = 3 ↔ x = 2 :=
sorry

end min_value_of_x_plus_four_over_x_squared_min_value_achieved_l1414_141445


namespace ellipse_eccentricity_l1414_141404

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    and a line y = kx intersecting the ellipse at points B and C,
    if the product of the slopes of AB and AC is -3/4,
    then the eccentricity of the ellipse is 1/2. -/
theorem ellipse_eccentricity (a b : ℝ) (k : ℝ) :
  a > b ∧ b > 0 →
  ∃ (B C : ℝ × ℝ),
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
    (C.1^2 / a^2 + C.2^2 / b^2 = 1) ∧
    (B.2 = k * B.1) ∧
    (C.2 = k * C.1) ∧
    ((B.2 - b) / B.1 * (C.2 + b) / C.1 = -3/4) →
  Real.sqrt (1 - b^2 / a^2) = 1/2 := by
sorry

end ellipse_eccentricity_l1414_141404


namespace calculation_proof_l1414_141488

theorem calculation_proof : (1 / 6 : ℚ) * (-6 : ℚ) / (-1/6 : ℚ) * (6 : ℚ) = 36 := by
  sorry

end calculation_proof_l1414_141488


namespace lateral_surface_area_of_specific_pyramid_l1414_141436

/-- Regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  baseEdgeLength : ℝ
  volume : ℝ

/-- Lateral surface area of a regular hexagonal pyramid -/
def lateralSurfaceArea (pyramid : RegularHexagonalPyramid) : ℝ :=
  sorry

theorem lateral_surface_area_of_specific_pyramid :
  let pyramid : RegularHexagonalPyramid :=
    { baseEdgeLength := 2
    , volume := 2 * Real.sqrt 3 }
  lateralSurfaceArea pyramid = 12 := by
  sorry

end lateral_surface_area_of_specific_pyramid_l1414_141436


namespace orange_basket_problem_l1414_141458

theorem orange_basket_problem (N : ℕ) : 
  N % 10 = 2 → N % 12 = 0 → N = 72 := by
  sorry

end orange_basket_problem_l1414_141458


namespace prime_factorization_l1414_141423

theorem prime_factorization (n : ℕ) (h : n ≥ 2) :
  ∃ (primes : List ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ (n = primes.prod) := by
  sorry

end prime_factorization_l1414_141423


namespace cyclist_pedestrian_meeting_point_l1414_141486

/-- The meeting point of a cyclist and a pedestrian on a straight path --/
theorem cyclist_pedestrian_meeting_point (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let total_distance := a + b
  let cyclist_speed := total_distance
  let pedestrian_speed := a
  let meeting_point := a * (a + b) / (2 * a + b)
  meeting_point / cyclist_speed = (a - meeting_point) / pedestrian_speed ∧
  meeting_point < a :=
by sorry

end cyclist_pedestrian_meeting_point_l1414_141486


namespace dice_sum_probability_l1414_141437

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the number of dice rolled
def num_dice : ℕ := 4

-- Define the total number of possible outcomes
def total_outcomes : ℕ := die_sides ^ num_dice

-- Define a function to calculate favorable outcomes
noncomputable def favorable_outcomes : ℕ := 
  -- This function would calculate the number of favorable outcomes
  -- Based on the problem, this should be 480, but we don't assume this knowledge
  sorry

-- Theorem statement
theorem dice_sum_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 10 / 27 := by
  sorry

end dice_sum_probability_l1414_141437


namespace willies_stickers_l1414_141414

/-- The final sticker count for Willie -/
def final_sticker_count (initial_count : ℝ) (received_count : ℝ) : ℝ :=
  initial_count + received_count

/-- Theorem stating that Willie's final sticker count is the sum of his initial count and received stickers -/
theorem willies_stickers (initial_count received_count : ℝ) :
  final_sticker_count initial_count received_count = initial_count + received_count :=
by sorry

end willies_stickers_l1414_141414


namespace distance_after_two_hours_l1414_141449

/-- The distance between two people walking in opposite directions for 2 hours -/
theorem distance_after_two_hours 
  (jay_speed : ℝ) 
  (paul_speed : ℝ) 
  (h1 : jay_speed = 0.8 / 15) 
  (h2 : paul_speed = 3 / 30) 
  : jay_speed * 120 + paul_speed * 120 = 18.4 := by
  sorry

end distance_after_two_hours_l1414_141449


namespace sum_of_fractions_equals_five_elevenths_l1414_141478

theorem sum_of_fractions_equals_five_elevenths :
  (1 / (2^2 - 1) + 1 / (4^2 - 1) + 1 / (6^2 - 1) + 1 / (8^2 - 1) + 1 / (10^2 - 1) : ℚ) = 5 / 11 := by
  sorry

end sum_of_fractions_equals_five_elevenths_l1414_141478


namespace susies_house_rooms_l1414_141439

/-- The number of rooms in Susie's house -/
def number_of_rooms : ℕ := 6

/-- The time it takes Susie to vacuum the whole house, in hours -/
def total_vacuum_time : ℝ := 2

/-- The time it takes Susie to vacuum one room, in minutes -/
def time_per_room : ℝ := 20

/-- Theorem stating that the number of rooms in Susie's house is 6 -/
theorem susies_house_rooms :
  number_of_rooms = (total_vacuum_time * 60) / time_per_room :=
by sorry

end susies_house_rooms_l1414_141439


namespace employee_pay_calculation_l1414_141493

/-- Given two employees with a total weekly pay of 560, where one employee's pay is 150% of the other's, prove that the employee with the lower pay receives 224 per week. -/
theorem employee_pay_calculation (total_pay : ℝ) (a_pay b_pay : ℝ) : 
  total_pay = 560 →
  a_pay = 1.5 * b_pay →
  a_pay + b_pay = total_pay →
  b_pay = 224 := by sorry

end employee_pay_calculation_l1414_141493


namespace tennis_racket_weight_tennis_racket_weight_proof_l1414_141448

theorem tennis_racket_weight : ℝ → ℝ → Prop :=
  fun (racket_weight bicycle_weight : ℝ) =>
    (10 * racket_weight = 8 * bicycle_weight) →
    (4 * bicycle_weight = 120) →
    racket_weight = 24

-- Proof
theorem tennis_racket_weight_proof :
  ∃ (racket_weight bicycle_weight : ℝ),
    tennis_racket_weight racket_weight bicycle_weight :=
by
  sorry

end tennis_racket_weight_tennis_racket_weight_proof_l1414_141448


namespace a_in_closed_unit_interval_l1414_141400

def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}

theorem a_in_closed_unit_interval (a : ℝ) :
  P ∪ M a = P → a ∈ Set.Icc (-1) 1 := by
  sorry

end a_in_closed_unit_interval_l1414_141400


namespace plain_calculations_l1414_141403

/-- Given information about two plains A and B, prove various calculations about their areas, populations, and elevation difference. -/
theorem plain_calculations (area_B area_A : ℝ) (pop_density_A pop_density_B : ℝ) 
  (distance_AB : ℝ) (elevation_gradient : ℝ) :
  area_B = 200 →
  area_A = area_B - 50 →
  pop_density_A = 50 →
  pop_density_B = 75 →
  distance_AB = 25 →
  elevation_gradient = 500 / 10 →
  (area_A = 150 ∧ 
   area_A * pop_density_A + area_B * pop_density_B = 22500 ∧
   elevation_gradient * distance_AB = 125) :=
by sorry

end plain_calculations_l1414_141403


namespace right_triangle_area_in_square_yards_l1414_141407

/-- The area of a right triangle with legs of 60 feet and 80 feet in square yards -/
theorem right_triangle_area_in_square_yards : 
  let leg1 : ℝ := 60
  let leg2 : ℝ := 80
  let triangle_area_sqft : ℝ := (1/2) * leg1 * leg2
  let sqft_per_sqyd : ℝ := 9
  triangle_area_sqft / sqft_per_sqyd = 800/3 := by
  sorry

end right_triangle_area_in_square_yards_l1414_141407
