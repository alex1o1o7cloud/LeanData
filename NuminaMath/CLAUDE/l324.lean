import Mathlib

namespace range_of_z_l324_32425

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  12 ≤ x^2 + 4*y^2 ∧ x^2 + 4*y^2 ≤ 20 := by
  sorry

end range_of_z_l324_32425


namespace factorization_equality_l324_32489

theorem factorization_equality (x : ℝ) : x^2 - x - 6 = (x - 3) * (x + 2) := by
  sorry

end factorization_equality_l324_32489


namespace exists_integer_sqrt_20n_is_integer_l324_32468

theorem exists_integer_sqrt_20n_is_integer : ∃ n : ℤ, ∃ m : ℤ, 20 * n = m^2 := by
  sorry

end exists_integer_sqrt_20n_is_integer_l324_32468


namespace eastward_fish_caught_fraction_l324_32400

/-- Given the following conditions:
  - 1800 fish swim westward
  - 3200 fish swim eastward
  - 500 fish swim north
  - Fishers catch 3/4 of the fish that swam westward
  - There are 2870 fish left in the sea
Prove that the fraction of eastward-swimming fish caught by fishers is 2/5 -/
theorem eastward_fish_caught_fraction :
  let total_fish : ℕ := 1800 + 3200 + 500
  let westward_fish : ℕ := 1800
  let eastward_fish : ℕ := 3200
  let northward_fish : ℕ := 500
  let westward_caught_fraction : ℚ := 3 / 4
  let remaining_fish : ℕ := 2870
  let eastward_caught_fraction : ℚ := 2 / 5
  (total_fish : ℚ) - (westward_caught_fraction * westward_fish + eastward_caught_fraction * eastward_fish) = remaining_fish :=
by
  sorry

#check eastward_fish_caught_fraction

end eastward_fish_caught_fraction_l324_32400


namespace irrational_approximation_l324_32466

theorem irrational_approximation (x : ℝ) (h_pos : x > 0) (h_irr : Irrational x) :
  ∀ N : ℕ, ∃ p q : ℤ, q > N ∧ q > 0 ∧ |x - (p : ℝ) / q| < 1 / q^2 := by
  sorry

end irrational_approximation_l324_32466


namespace line_intercepts_sum_l324_32487

/-- Given a line with equation y - 6 = -3(x - 5), 
    the sum of its x-intercept and y-intercept is 28 -/
theorem line_intercepts_sum : 
  ∀ (x y : ℝ), y - 6 = -3 * (x - 5) → 
  ∃ (x_int y_int : ℝ),
    (y_int - 6 = -3 * (x_int - 5) ∧ y_int = 0) ∧
    (0 - 6 = -3 * (0 - 5) ∧ y = y_int) ∧
    x_int + y_int = 28 :=
by sorry

end line_intercepts_sum_l324_32487


namespace gcd_12345_23456_34567_l324_32494

theorem gcd_12345_23456_34567 : Nat.gcd 12345 (Nat.gcd 23456 34567) = 1 := by
  sorry

end gcd_12345_23456_34567_l324_32494


namespace remaining_money_l324_32408

def initial_amount : ℚ := 50
def shirt_cost : ℚ := 7.85
def meal_cost : ℚ := 15.49
def magazine_cost : ℚ := 6.13
def debt_payment : ℚ := 3.27
def cd_cost : ℚ := 11.75

theorem remaining_money :
  initial_amount - (shirt_cost + meal_cost + magazine_cost + debt_payment + cd_cost) = 5.51 := by
  sorry

end remaining_money_l324_32408


namespace apple_basket_solution_l324_32414

def basket_problem (x : ℕ) : Prop :=
  let first_sale := x / 4 + 6
  let remaining_after_first := x - first_sale
  let second_sale := remaining_after_first / 3 + 4
  let remaining_after_second := remaining_after_first - second_sale
  let third_sale := remaining_after_second / 2 + 3
  let final_remaining := remaining_after_second - third_sale
  final_remaining = 4

theorem apple_basket_solution :
  ∃ x : ℕ, basket_problem x ∧ x = 28 :=
sorry

end apple_basket_solution_l324_32414


namespace intersection_A_complement_B_l324_32447

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x | 1 < x ∧ x ≤ 2} := by sorry

end intersection_A_complement_B_l324_32447


namespace isosceles_triangles_l324_32453

/-- A circle with two equal chords that extend to intersect -/
structure CircleWithIntersectingChords where
  /-- The circle -/
  circle : Set (ℝ × ℝ)
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- First chord endpoint -/
  A : ℝ × ℝ
  /-- Second chord endpoint -/
  B : ℝ × ℝ
  /-- Third chord endpoint -/
  C : ℝ × ℝ
  /-- Fourth chord endpoint -/
  D : ℝ × ℝ
  /-- Intersection point of extended chords -/
  P : ℝ × ℝ
  /-- A and B are on the circle -/
  hAB : A ∈ circle ∧ B ∈ circle
  /-- C and D are on the circle -/
  hCD : C ∈ circle ∧ D ∈ circle
  /-- AB and CD are equal chords -/
  hEqualChords : dist A B = dist C D
  /-- P is on the extension of AB beyond B -/
  hPAB : ∃ t > 1, P = A + t • (B - A)
  /-- P is on the extension of CD beyond C -/
  hPCD : ∃ t > 1, P = D + t • (C - D)

/-- The main theorem: triangles APD and BPC are isosceles -/
theorem isosceles_triangles (cfg : CircleWithIntersectingChords) :
  dist cfg.A cfg.P = dist cfg.D cfg.P ∧ dist cfg.B cfg.P = dist cfg.C cfg.P := by
  sorry

end isosceles_triangles_l324_32453


namespace horners_method_polynomial_transformation_l324_32438

theorem horners_method_polynomial_transformation (x : ℝ) :
  6 * x^3 + 5 * x^2 + 4 * x + 3 = x * (x * (6 * x + 5) + 4) + 3 := by
  sorry

end horners_method_polynomial_transformation_l324_32438


namespace even_function_property_l324_32445

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the main theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_positive : ∀ x > 0, f x = x) : 
  ∀ x < 0, f x = -x :=
by
  sorry


end even_function_property_l324_32445


namespace first_book_length_l324_32460

theorem first_book_length :
  ∀ (book1 book2 total_pages daily_pages days : ℕ),
    book2 = 100 →
    days = 14 →
    daily_pages = 20 →
    total_pages = daily_pages * days →
    book1 + book2 = total_pages →
    book1 = 180 := by
sorry

end first_book_length_l324_32460


namespace inequalities_proof_l324_32451

theorem inequalities_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 3) :
  (a^2 + b^2 ≥ 9/5) ∧ (a^3*b + 4*a*b^3 ≤ 81/16) := by
  sorry

end inequalities_proof_l324_32451


namespace least_possible_difference_l324_32488

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → Odd y → Odd z → 
  (∀ d : ℤ, z - x ≥ d → d ≥ 9) :=
by sorry

end least_possible_difference_l324_32488


namespace fourth_derivative_y_l324_32406

noncomputable def y (x : ℝ) : ℝ := (3 * x - 7) * (3 : ℝ)^(-x)

theorem fourth_derivative_y (x : ℝ) :
  (deriv^[4] y) x = (7 * Real.log 3 - 12 - 3 * Real.log 3 * x) * (Real.log 3)^3 * (3 : ℝ)^(-x) :=
by sorry

end fourth_derivative_y_l324_32406


namespace waiting_by_stump_is_random_waiting_by_stump_unique_random_l324_32450

-- Define the type for idioms
inductive Idiom
  | FishingForMoon
  | CastlesInAir
  | WaitingByStump
  | CatchingTurtle

-- Define a property for idioms
def describesRandomEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.FishingForMoon => False
  | Idiom.CastlesInAir => False
  | Idiom.WaitingByStump => True
  | Idiom.CatchingTurtle => False

-- Theorem stating that "Waiting by a stump for a hare" describes a random event
theorem waiting_by_stump_is_random :
  describesRandomEvent Idiom.WaitingByStump :=
by sorry

-- Theorem stating that "Waiting by a stump for a hare" is the only idiom
-- among the given options that describes a random event
theorem waiting_by_stump_unique_random :
  ∀ (i : Idiom), describesRandomEvent i ↔ i = Idiom.WaitingByStump :=
by sorry

end waiting_by_stump_is_random_waiting_by_stump_unique_random_l324_32450


namespace hyperbola_eccentricity_l324_32422

/-- The eccentricity of a hyperbola with given equation and asymptote -/
theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, Real.sqrt 5 * x - 2 * y = 0 → y = (Real.sqrt 5 / 2) * x) →
  b / a = Real.sqrt 5 / 2 →
  Real.sqrt ((a^2 + b^2) / a^2) = 3 / 2 :=
by sorry

end hyperbola_eccentricity_l324_32422


namespace triple_lcm_equation_solution_l324_32491

theorem triple_lcm_equation_solution (a b c n : ℕ+) 
  (h1 : a^2 + b^2 = n * Nat.lcm a b + n^2)
  (h2 : b^2 + c^2 = n * Nat.lcm b c + n^2)
  (h3 : c^2 + a^2 = n * Nat.lcm c a + n^2) :
  a = n ∧ b = n ∧ c = n := by
sorry

end triple_lcm_equation_solution_l324_32491


namespace car_price_before_discount_l324_32464

theorem car_price_before_discount 
  (discount_percentage : ℝ) 
  (price_after_discount : ℝ) 
  (h1 : discount_percentage = 55) 
  (h2 : price_after_discount = 450000) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_percentage / 100) = price_after_discount ∧ 
    original_price = 1000000 := by
  sorry

end car_price_before_discount_l324_32464


namespace batsman_average_runs_l324_32465

def average_runs (total_runs : ℕ) (num_matches : ℕ) : ℚ :=
  (total_runs : ℚ) / (num_matches : ℚ)

theorem batsman_average_runs :
  let first_20_matches := 20
  let next_10_matches := 10
  let total_matches := first_20_matches + next_10_matches
  let avg_first_20 := 40
  let avg_next_10 := 13
  let total_runs_first_20 := first_20_matches * avg_first_20
  let total_runs_next_10 := next_10_matches * avg_next_10
  let total_runs := total_runs_first_20 + total_runs_next_10
  average_runs total_runs total_matches = 31 := by
sorry

end batsman_average_runs_l324_32465


namespace base_b_121_is_perfect_square_l324_32409

/-- Represents a number in base b as a list of digits --/
def BaseRepresentation (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => acc * b + digit) 0

/-- Checks if a number is a perfect square --/
def IsPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem base_b_121_is_perfect_square (b : Nat) :
  (b > 2) ↔ IsPerfectSquare (BaseRepresentation [1, 2, 1] b) :=
by sorry

end base_b_121_is_perfect_square_l324_32409


namespace total_balloons_l324_32429

theorem total_balloons (tom_balloons sara_balloons alex_balloons : ℕ) 
  (h1 : tom_balloons = 18) 
  (h2 : sara_balloons = 12) 
  (h3 : alex_balloons = 7) : 
  tom_balloons + sara_balloons + alex_balloons = 37 := by
  sorry

end total_balloons_l324_32429


namespace bus_capacity_l324_32411

/-- The number of rows in a bus -/
def rows : ℕ := 9

/-- The number of children that can be accommodated in each row -/
def children_per_row : ℕ := 4

/-- The total number of children a bus can accommodate -/
def total_children : ℕ := rows * children_per_row

theorem bus_capacity : total_children = 36 := by
  sorry

end bus_capacity_l324_32411


namespace squirrel_rainy_days_l324_32463

theorem squirrel_rainy_days 
  (sunny_nuts : ℕ) 
  (rainy_nuts : ℕ) 
  (total_nuts : ℕ) 
  (average_nuts : ℕ) 
  (h1 : sunny_nuts = 20)
  (h2 : rainy_nuts = 12)
  (h3 : total_nuts = 112)
  (h4 : average_nuts = 14)
  : ∃ (rainy_days : ℕ), rainy_days = 6 ∧ 
    ∃ (total_days : ℕ), 
      total_days * average_nuts = total_nuts ∧
      rainy_days * rainy_nuts + (total_days - rainy_days) * sunny_nuts = total_nuts :=
by sorry

end squirrel_rainy_days_l324_32463


namespace equation_solution_l324_32495

theorem equation_solution (x : ℝ) :
  (2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x)) ↔
  (∃ k : ℤ, x = (π / 16) * (4 * ↑k + 1)) :=
by sorry

end equation_solution_l324_32495


namespace bisection_method_max_experiments_l324_32484

theorem bisection_method_max_experiments (n : ℕ) (h : n = 33) :
  ∃ k : ℕ, k = 6 ∧ ∀ m : ℕ, 2^m < n → m < k :=
sorry

end bisection_method_max_experiments_l324_32484


namespace max_two_scoop_sundaes_l324_32443

theorem max_two_scoop_sundaes (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by
  sorry

end max_two_scoop_sundaes_l324_32443


namespace total_fish_count_l324_32448

/-- The number of fish owned by each person -/
def lilly_fish : ℕ := 10
def rosy_fish : ℕ := 11
def alex_fish : ℕ := 15
def jamie_fish : ℕ := 8
def sam_fish : ℕ := 20

/-- Theorem stating that the total number of fish is 64 -/
theorem total_fish_count : 
  lilly_fish + rosy_fish + alex_fish + jamie_fish + sam_fish = 64 := by
  sorry

end total_fish_count_l324_32448


namespace cosine_triangle_condition_l324_32486

/-- A cubic equation with real coefficients -/
structure CubicEquation where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The roots of a cubic equation are all real and positive -/
def has_real_positive_roots (eq : CubicEquation) : Prop := sorry

/-- The roots of a cubic equation are the cosines of the angles of a triangle -/
def roots_are_triangle_cosines (eq : CubicEquation) : Prop := sorry

/-- The necessary and sufficient condition for the roots to be the cosines of the angles of a triangle -/
theorem cosine_triangle_condition (eq : CubicEquation) :
  roots_are_triangle_cosines eq ↔
    eq.p^2 - 2*eq.q - 2*eq.r - 1 = 0 ∧ eq.p < 0 ∧ eq.q > 0 ∧ eq.r < 0 :=
by sorry

end cosine_triangle_condition_l324_32486


namespace triangle_problem_l324_32449

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  -- Conclusions
  B = π / 3 ∧
  a = Real.sqrt 3 ∧
  c = 2 * Real.sqrt 3 := by
  sorry

end triangle_problem_l324_32449


namespace normal_distribution_std_dev_l324_32462

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  stdDev : ℝ

/-- The value that is exactly k standard deviations from the mean --/
def valueAtStdDev (d : NormalDistribution) (k : ℝ) : ℝ :=
  d.mean + k * d.stdDev

theorem normal_distribution_std_dev (d : NormalDistribution) :
  d.mean = 15 ∧ valueAtStdDev d (-2) = 12 → d.stdDev = 1.5 := by
  sorry

end normal_distribution_std_dev_l324_32462


namespace wall_length_theorem_l324_32416

/-- Calculates the length of a wall built by a different number of workers in a different time, 
    given the original wall length and worker-days. -/
def calculate_wall_length (original_workers : ℕ) (original_days : ℕ) (original_length : ℕ) 
                          (new_workers : ℕ) (new_days : ℕ) : ℚ :=
  (original_workers * original_days * original_length : ℚ) / (new_workers * new_days)

theorem wall_length_theorem (original_workers : ℕ) (original_days : ℕ) (original_length : ℕ) 
                            (new_workers : ℕ) (new_days : ℕ) :
  original_workers = 18 →
  original_days = 42 →
  original_length = 140 →
  new_workers = 30 →
  new_days = 18 →
  calculate_wall_length original_workers original_days original_length new_workers new_days = 196 := by
  sorry

#eval calculate_wall_length 18 42 140 30 18

end wall_length_theorem_l324_32416


namespace circle_M_equation_l324_32446

-- Define the circle M
structure CircleM where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
axiom center_on_line (M : CircleM) : M.center.2 = -2 * M.center.1

axiom passes_through_A (M : CircleM) :
  (2 - M.center.1)^2 + (-1 - M.center.2)^2 = M.radius^2

axiom tangent_to_line (M : CircleM) :
  |M.center.1 + M.center.2 - 1| / Real.sqrt 2 = M.radius

-- Define the theorem to be proved
theorem circle_M_equation (M : CircleM) :
  ∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 2 ↔
    (x - M.center.1)^2 + (y - M.center.2)^2 = M.radius^2 :=
sorry

end circle_M_equation_l324_32446


namespace vaccine_development_probabilities_l324_32442

/-- Success probability of Company A for developing vaccine A -/
def prob_A : ℚ := 2/3

/-- Success probability of Company B for developing vaccine A -/
def prob_B : ℚ := 1/2

/-- The theorem states that given the success probabilities of Company A and Company B,
    1) The probability that both succeed is 1/3
    2) The probability of vaccine A being successfully developed is 5/6 -/
theorem vaccine_development_probabilities :
  (prob_A * prob_B = 1/3) ∧
  (1 - (1 - prob_A) * (1 - prob_B) = 5/6) :=
sorry

end vaccine_development_probabilities_l324_32442


namespace distance_proof_l324_32439

/-- The distance between three equidistant points A, B, and C. -/
def distance_between_points : ℝ := 26

/-- The speed of the cyclist traveling from A to B in km/h. -/
def cyclist_speed : ℝ := 15

/-- The speed of the tourist traveling from B to C in km/h. -/
def tourist_speed : ℝ := 5

/-- The time at which the cyclist and tourist are at their shortest distance, in hours. -/
def time_shortest_distance : ℝ := 1.4

/-- The theorem stating that the distance between the points is 26 km under the given conditions. -/
theorem distance_proof :
  ∀ (S : ℝ),
  (S > 0) →
  (S = distance_between_points) →
  (∀ (t : ℝ), 
    (t > 0) →
    (cyclist_speed * t ≤ S) →
    (tourist_speed * t ≤ S) →
    (S^2 - 35*t*S + 325*t^2 ≥ S^2 - 35*time_shortest_distance*S + 325*time_shortest_distance^2)) →
  (S = 26) :=
sorry

end distance_proof_l324_32439


namespace jogging_distance_l324_32401

/-- Calculates the distance traveled given a constant rate and time. -/
def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Proves that jogging at 4 miles per hour for 2 hours results in a distance of 8 miles. -/
theorem jogging_distance : distance 4 2 = 8 := by
  sorry

end jogging_distance_l324_32401


namespace min_sum_absolute_values_l324_32444

theorem min_sum_absolute_values : ∀ x : ℝ, 
  |x + 3| + |x + 5| + |x + 6| ≥ 5 ∧ 
  ∃ y : ℝ, |y + 3| + |y + 5| + |y + 6| = 5 := by
  sorry

end min_sum_absolute_values_l324_32444


namespace binomial_expansion_theorem_l324_32455

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) : 
  n ≥ 3 →
  a * b ≠ 0 →
  a = k^2 * b →
  k > 0 →
  (n.choose 2) * (a + b)^(n - 2) * a * b + (n.choose 3) * (a + b)^(n - 3) * a^2 * b = 0 →
  n = 3 * k + 2 := by
sorry

end binomial_expansion_theorem_l324_32455


namespace whitney_cant_afford_l324_32467

def poster_price : ℚ := 7.5
def notebook_price : ℚ := 5.25
def bookmark_price : ℚ := 3.1
def pencil_price : ℚ := 1.15
def sales_tax_rate : ℚ := 0.08
def initial_money : ℚ := 40

def total_cost (poster_qty notebook_qty bookmark_qty pencil_qty : ℕ) : ℚ :=
  let subtotal := poster_price * poster_qty + notebook_price * notebook_qty + 
                  bookmark_price * bookmark_qty + pencil_price * pencil_qty
  subtotal * (1 + sales_tax_rate)

theorem whitney_cant_afford (poster_qty notebook_qty bookmark_qty pencil_qty : ℕ) 
  (h_poster : poster_qty = 3)
  (h_notebook : notebook_qty = 4)
  (h_bookmark : bookmark_qty = 5)
  (h_pencil : pencil_qty = 2) :
  total_cost poster_qty notebook_qty bookmark_qty pencil_qty > initial_money :=
by sorry

end whitney_cant_afford_l324_32467


namespace vector_at_zero_given_two_points_l324_32440

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  -- The vector on the line at parameter t
  vector : ℝ → ℝ × ℝ

theorem vector_at_zero_given_two_points (L : ParameterizedLine) :
  L.vector 1 = (2, 3) →
  L.vector 4 = (8, -5) →
  L.vector 0 = (0, 17/3) := by
  sorry

end vector_at_zero_given_two_points_l324_32440


namespace congruence_solution_l324_32437

theorem congruence_solution (n : ℤ) : 13 * n ≡ 19 [ZMOD 47] ↔ n ≡ 30 [ZMOD 47] := by sorry

end congruence_solution_l324_32437


namespace positive_real_inequality_l324_32436

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end positive_real_inequality_l324_32436


namespace percentage_problem_l324_32479

theorem percentage_problem (P : ℝ) : P = (354.2 * 6 * 100) / 1265 ↔ (P / 100) * 1265 / 6 = 354.2 := by
  sorry

end percentage_problem_l324_32479


namespace inverse_composition_equals_negative_one_l324_32458

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 5

-- Define the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℝ := (y - 5) / 4

-- Theorem statement
theorem inverse_composition_equals_negative_one :
  f_inv (f_inv 9) = -1 := by sorry

end inverse_composition_equals_negative_one_l324_32458


namespace systematic_sampling_elimination_l324_32412

theorem systematic_sampling_elimination (population : Nat) (sample_size : Nat) 
    (h1 : population = 1252) 
    (h2 : sample_size = 50) : 
  population % sample_size = 2 := by
  sorry

end systematic_sampling_elimination_l324_32412


namespace geese_survival_theorem_l324_32492

/-- Represents the fraction of geese that did not survive the first year out of those that survived the first month -/
def fraction_not_survived_first_year (
  total_eggs : ℕ
  ) (
  hatch_rate : ℚ
  ) (
  first_month_survival_rate : ℚ
  ) (
  first_year_survivors : ℕ
  ) : ℚ :=
  1 - (first_year_survivors : ℚ) / (total_eggs * hatch_rate * first_month_survival_rate)

/-- Theorem stating that the fraction of geese that did not survive the first year is 0 -/
theorem geese_survival_theorem (
  total_eggs : ℕ
  ) (
  hatch_rate : ℚ
  ) (
  first_month_survival_rate : ℚ
  ) (
  first_year_survivors : ℕ
  ) (
  h1 : hatch_rate = 1/3
  ) (
  h2 : first_month_survival_rate = 4/5
  ) (
  h3 : first_year_survivors = 120
  ) (
  h4 : total_eggs * hatch_rate * first_month_survival_rate = first_year_survivors
  ) : fraction_not_survived_first_year total_eggs hatch_rate first_month_survival_rate first_year_survivors = 0 :=
by
  sorry

#check geese_survival_theorem

end geese_survival_theorem_l324_32492


namespace andrews_balloons_l324_32477

/-- Given a number of blue and purple balloons, calculates how many balloons are left after sharing half of the total. -/
def balloons_left (blue : ℕ) (purple : ℕ) : ℕ :=
  (blue + purple) / 2

/-- Theorem stating that given 303 blue balloons and 453 purple balloons, 
    the number of balloons left after sharing half is 378. -/
theorem andrews_balloons : balloons_left 303 453 = 378 := by
  sorry

end andrews_balloons_l324_32477


namespace sqrt_seven_irrational_rational_numbers_sqrt_seven_is_irrational_l324_32419

theorem sqrt_seven_irrational :
  ∀ (a b : ℚ), a^2 ≠ 7 * b^2 :=
sorry

theorem rational_numbers :
  ∃ (q₁ q₂ q₃ : ℚ),
    (q₁ : ℝ) = 3.14159265 ∧
    (q₂ : ℝ) = Real.sqrt 36 ∧
    (q₃ : ℝ) = 4.1 :=
sorry

theorem sqrt_seven_is_irrational :
  Irrational (Real.sqrt 7) :=
sorry

end sqrt_seven_irrational_rational_numbers_sqrt_seven_is_irrational_l324_32419


namespace travel_time_calculation_l324_32457

/-- Given a speed of 65 km/hr and a distance of 195 km, the travel time is 3 hours -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (time : ℝ) :
  speed = 65 → distance = 195 → time = distance / speed → time = 3 := by
  sorry

end travel_time_calculation_l324_32457


namespace prob_different_classes_correct_expected_value_class1_correct_l324_32403

/-- Represents the number of classes in the first year -/
def num_classes : ℕ := 8

/-- Represents the total number of students selected for the community service group -/
def total_selected : ℕ := 10

/-- Represents the number of students selected from Class 1 -/
def class1_selected : ℕ := 3

/-- Represents the number of students selected from each of the other classes -/
def other_classes_selected : ℕ := 1

/-- Represents the number of students randomly selected for the activity -/
def activity_selected : ℕ := 3

/-- Probability of selecting 3 students from different classes -/
def prob_different_classes : ℚ := 49/60

/-- Expected value of the number of students selected from Class 1 -/
def expected_value_class1 : ℚ := 43/40

/-- Theorem stating the probability of selecting 3 students from different classes -/
theorem prob_different_classes_correct :
  let total_ways := Nat.choose total_selected activity_selected
  let ways_with_one_from_class1 := Nat.choose class1_selected 1 * Nat.choose (total_selected - class1_selected) 2
  let ways_with_none_from_class1 := Nat.choose class1_selected 0 * Nat.choose (total_selected - class1_selected) 3
  (ways_with_one_from_class1 + ways_with_none_from_class1) / total_ways = prob_different_classes :=
sorry

/-- Theorem stating the expected value of the number of students selected from Class 1 -/
theorem expected_value_class1_correct :
  let p0 := (7 : ℚ) / 24
  let p1 := (21 : ℚ) / 40
  let p2 := (7 : ℚ) / 40
  let p3 := (1 : ℚ) / 120
  0 * p0 + 1 * p1 + 2 * p2 + 3 * p3 = expected_value_class1 :=
sorry

end prob_different_classes_correct_expected_value_class1_correct_l324_32403


namespace childrens_ticket_cost_l324_32428

/-- Prove that the cost of a children's ticket is $4.50 -/
theorem childrens_ticket_cost
  (adult_ticket_cost : ℝ)
  (total_tickets : ℕ)
  (total_revenue : ℝ)
  (childrens_tickets : ℕ)
  (h1 : adult_ticket_cost = 6)
  (h2 : total_tickets = 400)
  (h3 : total_revenue = 2100)
  (h4 : childrens_tickets = 200) :
  ∃ (childrens_ticket_cost : ℝ),
    childrens_ticket_cost * childrens_tickets +
    adult_ticket_cost * (total_tickets - childrens_tickets) = total_revenue ∧
    childrens_ticket_cost = 4.5 :=
by
  sorry


end childrens_ticket_cost_l324_32428


namespace inscribed_circle_radius_in_quarter_sector_inscribed_circle_radius_3cm_l324_32470

/-- The radius of an inscribed circle in a quarter circular sector --/
theorem inscribed_circle_radius_in_quarter_sector (R : ℝ) (h : R > 0) :
  let r := R * (Real.sqrt 2 - 1)
  r > 0 ∧ r * (1 + Real.sqrt 2) = R :=
by
  sorry

/-- The specific case where the outer radius is 3 cm --/
theorem inscribed_circle_radius_3cm :
  let r := 3 * (Real.sqrt 2 - 1)
  r > 0 ∧ r * (1 + Real.sqrt 2) = 3 :=
by
  sorry

end inscribed_circle_radius_in_quarter_sector_inscribed_circle_radius_3cm_l324_32470


namespace lcm_gcf_problem_l324_32430

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 16 = 48 → Nat.gcd n 16 = 18 → n = 54 := by
  sorry

end lcm_gcf_problem_l324_32430


namespace triangle_equals_four_l324_32482

/-- Given that △ is a digit and △7₁₂ = △3₁₃, prove that △ = 4 -/
theorem triangle_equals_four (triangle : ℕ) 
  (h1 : triangle < 10) 
  (h2 : triangle * 12 + 7 = triangle * 13 + 3) : 
  triangle = 4 := by sorry

end triangle_equals_four_l324_32482


namespace diophantine_equation_solution_l324_32423

theorem diophantine_equation_solution :
  ∀ x y : ℕ+, x^4 = y^2 + 71 ↔ x = 6 ∧ y = 35 := by
  sorry

end diophantine_equation_solution_l324_32423


namespace smallest_palindrome_base3_l324_32404

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def convertBase (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Number of digits of a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_palindrome_base3 :
  ∀ n : ℕ,
  isPalindrome n 3 ∧ numDigits n 3 = 5 →
  (∃ b : ℕ, b ≠ 3 ∧ isPalindrome (convertBase n 3 b) b ∧ numDigits (convertBase n 3 b) b = 3) →
  n ≥ 81 := by
  sorry

end smallest_palindrome_base3_l324_32404


namespace existence_of_x0_iff_b_negative_l324_32490

open Real

theorem existence_of_x0_iff_b_negative (a b : ℝ) (ha : a > 0) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ log x₀ > a * sqrt x₀ + b / sqrt x₀) ↔ b < 0 :=
sorry

end existence_of_x0_iff_b_negative_l324_32490


namespace coefficient_x2y3_in_binomial_expansion_l324_32459

theorem coefficient_x2y3_in_binomial_expansion :
  (Finset.range 6).sum (fun k => Nat.choose 5 k * (if k = 3 then 1 else 0)) = 10 := by
  sorry

end coefficient_x2y3_in_binomial_expansion_l324_32459


namespace det_of_specific_matrix_l324_32498

theorem det_of_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 4, 7]
  Matrix.det A = -1 := by
sorry

end det_of_specific_matrix_l324_32498


namespace exists_six_digit_number_with_digit_sum_43_l324_32461

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem exists_six_digit_number_with_digit_sum_43 :
  ∃ n : ℕ, n < 500000 ∧ n ≥ 100000 ∧ sum_of_digits n = 43 :=
by
  sorry

end exists_six_digit_number_with_digit_sum_43_l324_32461


namespace simplify_expression_l324_32452

theorem simplify_expression (x y : ℝ) : 7*x + 3 - 2*x + 15 + y = 5*x + y + 18 := by
  sorry

end simplify_expression_l324_32452


namespace toms_lawn_mowing_l324_32480

/-- Proves the number of lawns Tom mowed given his earnings and expenses -/
theorem toms_lawn_mowing (charge_per_lawn : ℕ) (gas_expense : ℕ) (weed_income : ℕ) (total_profit : ℕ) 
  (h1 : charge_per_lawn = 12)
  (h2 : gas_expense = 17)
  (h3 : weed_income = 10)
  (h4 : total_profit = 29) :
  ∃ (lawns_mowed : ℕ), 
    lawns_mowed * charge_per_lawn + weed_income - gas_expense = total_profit ∧ 
    lawns_mowed = 3 := by
  sorry


end toms_lawn_mowing_l324_32480


namespace max_percent_x_correct_l324_32410

/-- The maximum percentage of liquid X in the resulting solution --/
def max_percent_x : ℝ := 1.71

/-- Percentage of liquid X in solution A --/
def percent_x_a : ℝ := 0.8

/-- Percentage of liquid X in solution B --/
def percent_x_b : ℝ := 1.8

/-- Percentage of liquid X in solution C --/
def percent_x_c : ℝ := 3

/-- Percentage of liquid Y in solution A --/
def percent_y_a : ℝ := 2

/-- Percentage of liquid Y in solution B --/
def percent_y_b : ℝ := 1

/-- Percentage of liquid Y in solution C --/
def percent_y_c : ℝ := 0.5

/-- Amount of solution A in grams --/
def amount_a : ℝ := 500

/-- Amount of solution B in grams --/
def amount_b : ℝ := 700

/-- Amount of solution C in grams --/
def amount_c : ℝ := 300

/-- Maximum combined percentage of liquids X and Y in the resulting solution --/
def max_combined_percent : ℝ := 2.5

/-- Theorem stating that the maximum percentage of liquid X in the resulting solution is correct --/
theorem max_percent_x_correct :
  let total_amount := amount_a + amount_b + amount_c
  let amount_x := percent_x_a / 100 * amount_a + percent_x_b / 100 * amount_b + percent_x_c / 100 * amount_c
  let amount_y := percent_y_a / 100 * amount_a + percent_y_b / 100 * amount_b + percent_y_c / 100 * amount_c
  (amount_x + amount_y) / total_amount * 100 ≤ max_combined_percent ∧
  amount_x / total_amount * 100 = max_percent_x :=
by sorry

end max_percent_x_correct_l324_32410


namespace prime_square_mod_30_l324_32427

theorem prime_square_mod_30 (p : ℕ) (hp : Prime p) (h2 : p ≠ 2) (h3 : p ≠ 3) (h5 : p ≠ 5) :
  p ^ 2 % 30 = 1 ∨ p ^ 2 % 30 = 19 := by
sorry

end prime_square_mod_30_l324_32427


namespace shirt_sales_price_solution_l324_32405

/-- Represents the selling price and profit calculation for shirts -/
def ShirtSales (x : ℝ) : Prop :=
  let cost_price : ℝ := 80
  let initial_daily_sales : ℝ := 30
  let price_reduction : ℝ := 130 - x
  let additional_sales : ℝ := 2 * price_reduction
  let total_daily_sales : ℝ := initial_daily_sales + additional_sales
  let profit_per_shirt : ℝ := x - cost_price
  let daily_profit : ℝ := profit_per_shirt * total_daily_sales
  daily_profit = 2000

theorem shirt_sales_price_solution :
  ∃ x : ℝ, ShirtSales x ∧ (x = 105 ∨ x = 120) :=
sorry

end shirt_sales_price_solution_l324_32405


namespace final_parity_after_odd_changes_not_even_after_33_changes_l324_32483

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- Function to change the parity -/
def changeParity (p : Parity) : Parity :=
  match p with
  | Parity.Even => Parity.Odd
  | Parity.Odd => Parity.Even

/-- Function to apply n changes to initial parity -/
def applyNChanges (initial : Parity) (n : Nat) : Parity :=
  match n with
  | 0 => initial
  | k + 1 => changeParity (applyNChanges initial k)

theorem final_parity_after_odd_changes 
  (initial : Parity) (n : Nat) (h : Odd n) :
  applyNChanges initial n ≠ initial := by
  sorry

/-- Main theorem: After 33 changes, an initially even number cannot be even -/
theorem not_even_after_33_changes :
  applyNChanges Parity.Even 33 ≠ Parity.Even := by
  sorry

end final_parity_after_odd_changes_not_even_after_33_changes_l324_32483


namespace perpendicular_line_through_point_l324_32485

-- Define the given line
def given_line (x y : ℝ) : Prop := 2*x + y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 3)

-- Define the new line
def new_line (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (∀ x y : ℝ, given_line x y → (new_line x y → ¬given_line x y)) ∧
  new_line point_A.1 point_A.2 :=
sorry

end perpendicular_line_through_point_l324_32485


namespace school_population_l324_32493

theorem school_population (total_students : ℕ) : 
  (128 : ℕ) = (total_students / 2) →
  total_students = 256 := by
sorry

end school_population_l324_32493


namespace harmonic_numbers_theorem_l324_32475

/-- Definition of harmonic numbers -/
def are_harmonic (a b c : ℝ) : Prop :=
  1/b - 1/a = 1/c - 1/b

/-- Theorem: For harmonic numbers x, 5, 3 where x > 5, x = 15 -/
theorem harmonic_numbers_theorem (x : ℝ) 
  (h1 : are_harmonic x 5 3)
  (h2 : x > 5) : 
  x = 15 := by
  sorry

end harmonic_numbers_theorem_l324_32475


namespace furniture_shop_pricing_l324_32433

theorem furniture_shop_pricing (cost_price : ℝ) (markup_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 6525 →
  markup_percentage = 24 →
  selling_price = cost_price * (1 + markup_percentage / 100) →
  selling_price = 8091 := by
sorry

end furniture_shop_pricing_l324_32433


namespace inequality_proof_l324_32431

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8*a) < (a + b) / 2 - Real.sqrt (a*b) ∧
  (a + b) / 2 - Real.sqrt (a*b) < (a - b)^2 / (8*b) := by
  sorry

end inequality_proof_l324_32431


namespace perpendicular_vectors_magnitude_l324_32426

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ m ↦ (2, m)

theorem perpendicular_vectors_magnitude (m : ℝ) :
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) →
  Real.sqrt ((b m).1^2 + (b m).2^2) = Real.sqrt 5 := by
  sorry

end perpendicular_vectors_magnitude_l324_32426


namespace janous_inequality_l324_32497

theorem janous_inequality (x y : ℝ) (hx : x > -1) (hy : y > -1) (hsum : x + y = 1) :
  x / (y + 1) + y / (x + 1) ≥ 2 / 3 ∧
  (x / (y + 1) + y / (x + 1) = 2 / 3 ↔ x = 1 / 2 ∧ y = 1 / 2) :=
by sorry

end janous_inequality_l324_32497


namespace regression_lines_intersection_l324_32496

/-- A linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point (x, y) lies on the regression line -/
def on_line (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : on_line l₁ s t)
  (h₂ : on_line l₂ s t) :
  ∃ (x y : ℝ), on_line l₁ x y ∧ on_line l₂ x y ∧ x = s ∧ y = t :=
sorry

end regression_lines_intersection_l324_32496


namespace hotel_discount_l324_32472

/-- Calculate the discount for a hotel stay given the number of nights, cost per night, and total amount paid. -/
theorem hotel_discount (nights : ℕ) (cost_per_night : ℕ) (total_paid : ℕ) : 
  nights = 3 → cost_per_night = 250 → total_paid = 650 → 
  nights * cost_per_night - total_paid = 100 := by
sorry

end hotel_discount_l324_32472


namespace prime_sum_to_square_l324_32420

theorem prime_sum_to_square (a b : ℕ) : 
  let P := (Nat.lcm a b / (a + 1)) + (Nat.lcm a b / (b + 1))
  Prime P → ∃ n : ℕ, 4 * P + 5 = n^2 := by
  sorry

end prime_sum_to_square_l324_32420


namespace special_polyhedron_body_diagonals_l324_32441

/-- A convex polyhedron with specific face composition -/
structure SpecialPolyhedron where
  /-- The polyhedron is convex -/
  is_convex : Bool
  /-- Number of square faces -/
  num_squares : Nat
  /-- Number of regular hexagon faces -/
  num_hexagons : Nat
  /-- Number of regular octagon faces -/
  num_octagons : Nat
  /-- At each vertex, a square, a hexagon, and an octagon meet -/
  vertex_composition : Bool
  /-- The surface is composed of exactly 12 squares, 8 hexagons, and 6 octagons -/
  face_composition : num_squares = 12 ∧ num_hexagons = 8 ∧ num_octagons = 6

/-- The number of body diagonals in the special polyhedron -/
def num_body_diagonals (p : SpecialPolyhedron) : Nat :=
  sorry

/-- Theorem: The number of body diagonals in the special polyhedron is 840 -/
theorem special_polyhedron_body_diagonals (p : SpecialPolyhedron) : 
  num_body_diagonals p = 840 := by
  sorry

end special_polyhedron_body_diagonals_l324_32441


namespace same_color_probability_value_l324_32435

/-- The number of green balls in the bag -/
def green_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 5

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := green_balls + red_balls + blue_balls

/-- The probability of drawing two balls of the same color with replacement -/
def same_color_probability : ℚ :=
  (green_balls / total_balls) ^ 2 +
  (red_balls / total_balls) ^ 2 +
  (blue_balls / total_balls) ^ 2

theorem same_color_probability_value :
  same_color_probability = 49 / 128 := by
  sorry

end same_color_probability_value_l324_32435


namespace solution_difference_l324_32421

theorem solution_difference (x y : ℝ) : 
  (Int.floor x + (y - Int.floor y) = 3.7) →
  ((x - Int.floor x) + Int.floor y = 4.2) →
  |x - y| = 1.5 := by sorry

end solution_difference_l324_32421


namespace earnings_left_over_l324_32424

/-- Calculates the percentage of earnings left over after spending on rent and dishwasher -/
theorem earnings_left_over (rent_percentage : ℝ) (dishwasher_discount : ℝ) : 
  rent_percentage = 25 →
  dishwasher_discount = 10 →
  100 - (rent_percentage + (rent_percentage - rent_percentage * dishwasher_discount / 100)) = 52.5 := by
  sorry


end earnings_left_over_l324_32424


namespace rectangle_area_perimeter_optimization_l324_32473

/-- Given a positive real number S, this theorem states that for any rectangle with area S and perimeter p,
    the expression S / (2S + p + 2) is maximized when the rectangle is a square, 
    and the maximum value is S / (2(√S + 1)²). -/
theorem rectangle_area_perimeter_optimization (S : ℝ) (hS : S > 0) :
  ∀ (a b : ℝ), a > 0 → b > 0 → a * b = S →
    S / (2 * S + 2 * (a + b) + 2) ≤ S / (2 * (Real.sqrt S + 1)^2) :=
by sorry

end rectangle_area_perimeter_optimization_l324_32473


namespace at_least_one_is_one_l324_32474

theorem at_least_one_is_one (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 1/a + 1/b + 1/c) : 
  a = 1 ∨ b = 1 ∨ c = 1 := by
sorry

end at_least_one_is_one_l324_32474


namespace zachary_bus_ride_length_l324_32469

theorem zachary_bus_ride_length : 
  let vince_ride : ℚ := 0.625
  let difference : ℚ := 0.125
  let zachary_ride : ℚ := vince_ride - difference
  zachary_ride = 0.500 := by sorry

end zachary_bus_ride_length_l324_32469


namespace quadratic_properties_l324_32417

def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem quadratic_properties :
  ∀ (a b m : ℝ),
  (quadratic_function a b 2 = 0) →
  (quadratic_function a b 1 = m) →
  (
    (m = 3 → a = -2 ∧ b = 3) ∧
    (m = 3 → ∀ x, -1 ≤ x ∧ x ≤ 2 → -3 ≤ quadratic_function a b x ∧ quadratic_function a b x ≤ 25/8) ∧
    (a > 0 → m < 1)
  ) := by sorry

end quadratic_properties_l324_32417


namespace marble_problem_l324_32454

theorem marble_problem (total initial_marbles : ℕ) 
  (white red blue : ℕ) 
  (h1 : total = 50)
  (h2 : red = blue)
  (h3 : white + red + blue = total)
  (h4 : total - (2 * (white - blue)) = 40) :
  white = 5 := by sorry

end marble_problem_l324_32454


namespace quadratic_inequality_minimum_l324_32471

theorem quadratic_inequality_minimum (a b c : ℝ) : 
  (∀ x, ax^2 + b*x + c < 0 ↔ -1 < x ∧ x < 3) →
  (∃ m, ∀ a b c, (∀ x, ax^2 + b*x + c < 0 ↔ -1 < x ∧ x < 3) → 
    b - 2*c + 1/a ≥ m ∧ 
    (∃ a₀ b₀ c₀, (∀ x, a₀*x^2 + b₀*x + c₀ < 0 ↔ -1 < x ∧ x < 3) ∧ 
      b₀ - 2*c₀ + 1/a₀ = m)) ∧
  m = 4 :=
sorry

end quadratic_inequality_minimum_l324_32471


namespace divisors_ending_in_2_mod_2010_l324_32434

-- Define the number 2010
def n : ℕ := 2010

-- Define the function that counts divisors ending in 2
def count_divisors_ending_in_2 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisors_ending_in_2_mod_2010 : 
  count_divisors_ending_in_2 (n^n) % n = 503 := by sorry

end divisors_ending_in_2_mod_2010_l324_32434


namespace eric_pencils_l324_32478

theorem eric_pencils (containers : ℕ) (additional_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : containers = 5)
  (h2 : additional_pencils = 30)
  (h3 : total_pencils = 36)
  (h4 : total_pencils % containers = 0) :
  total_pencils - additional_pencils = 6 := by
  sorry

end eric_pencils_l324_32478


namespace range_of_f_l324_32432

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (x - 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} :=
sorry

end range_of_f_l324_32432


namespace cosine_range_theorem_l324_32476

theorem cosine_range_theorem (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  x ∈ {x | Real.cos x ≤ 1/2} ↔ x ∈ Set.Icc (Real.pi/3) (5*Real.pi/3) := by
sorry

end cosine_range_theorem_l324_32476


namespace preferred_pets_combinations_l324_32402

/-- The number of puppies available in the pet store -/
def num_puppies : Nat := 20

/-- The number of kittens available in the pet store -/
def num_kittens : Nat := 10

/-- The number of hamsters available in the pet store -/
def num_hamsters : Nat := 12

/-- The number of ways Alice, Bob, and Charlie can buy their preferred pets -/
def num_ways : Nat := num_puppies * num_kittens * num_hamsters

/-- Theorem stating that the number of ways to buy preferred pets is 2400 -/
theorem preferred_pets_combinations : num_ways = 2400 := by
  sorry

end preferred_pets_combinations_l324_32402


namespace sum_of_squares_positive_l324_32415

theorem sum_of_squares_positive (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  x₁^2 + 2*x₁ - k = 0 → 
  x₂^2 + 2*x₂ - k = 0 → 
  x₁^2 + x₂^2 > 0 :=
by sorry

end sum_of_squares_positive_l324_32415


namespace gcd_2028_2100_l324_32413

theorem gcd_2028_2100 : Nat.gcd 2028 2100 = 36 := by
  sorry

end gcd_2028_2100_l324_32413


namespace problem_1_problem_2_l324_32418

-- Problem 1
theorem problem_1 : 
  (2 + 1/4)^(1/2) + (-3.8)^0 - Real.sqrt 3 * (3/2)^(1/3) * (12^(1/6)) = -1/2 := by sorry

-- Problem 2
theorem problem_2 : 
  2 * (Real.log 2 / Real.log 3) - Real.log (32/9) / Real.log 3 + Real.log 8 / Real.log 3 - 
  (Real.log 9 / Real.log 2) * (Real.log 2 / Real.log 3) = 2 := by sorry

end problem_1_problem_2_l324_32418


namespace rowing_speed_in_still_water_l324_32456

/-- Represents the rowing scenario with upstream and downstream times, current speed, and still water speed. -/
structure RowingScenario where
  upstream_time : ℝ
  downstream_time : ℝ
  current_speed : ℝ
  still_water_speed : ℝ

/-- Theorem stating that given the conditions, the man's rowing speed in still water is 3.6 km/hr. -/
theorem rowing_speed_in_still_water 
  (scenario : RowingScenario)
  (h1 : scenario.upstream_time = 2 * scenario.downstream_time)
  (h2 : scenario.current_speed = 1.2)
  : scenario.still_water_speed = 3.6 :=
by sorry

end rowing_speed_in_still_water_l324_32456


namespace trigonometric_identity_l324_32499

theorem trigonometric_identity : 
  Real.cos (12 * π / 180) * Real.sin (42 * π / 180) - 
  Real.sin (12 * π / 180) * Real.cos (42 * π / 180) = 1/2 := by
  sorry

end trigonometric_identity_l324_32499


namespace f_min_at_neg_two_l324_32407

/-- The polynomial f(x) = x^2 + 4x + 4 -/
def f (x : ℝ) : ℝ := x^2 + 4*x + 4

/-- The minimum value of f occurs at x = -2 -/
theorem f_min_at_neg_two :
  ∀ x : ℝ, f x ≥ f (-2) :=
by
  sorry

end f_min_at_neg_two_l324_32407


namespace even_function_inequality_l324_32481

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is monotonic on (-∞, 0] if it's either
    nondecreasing or nonincreasing on that interval -/
def IsMonotonicOnNegative (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y) ∨ (∀ x y, x ≤ y ∧ y ≤ 0 → f y ≤ f x)

theorem even_function_inequality (f : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_monotonic : IsMonotonicOnNegative f)
  (h_inequality : f (-2) < f 1) :
  f 5 < f (-3) ∧ f (-3) < f (-1) := by
  sorry

end even_function_inequality_l324_32481
