import Mathlib

namespace weight_loss_difference_l1204_120412

theorem weight_loss_difference (total_loss weight_first weight_third weight_fourth : ℕ) :
  total_loss = weight_first + weight_third + weight_fourth + (weight_first - 7) →
  weight_third = weight_fourth →
  total_loss = 103 →
  weight_first = 27 →
  weight_third = 28 →
  7 = weight_first - (total_loss - weight_first - weight_third - weight_fourth) :=
by sorry

end weight_loss_difference_l1204_120412


namespace remaining_balance_l1204_120430

def house_price : ℕ := 100000
def down_payment_percentage : ℚ := 20 / 100
def parents_payment_percentage : ℚ := 30 / 100

theorem remaining_balance (hp : ℕ) (dp : ℚ) (pp : ℚ) : 
  hp - hp * dp - (hp - hp * dp) * pp = 56000 :=
by sorry

end remaining_balance_l1204_120430


namespace count_primes_with_squares_between_5000_and_9000_l1204_120423

theorem count_primes_with_squares_between_5000_and_9000 :
  ∃ (S : Finset Nat),
    (∀ p ∈ S, Nat.Prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧
    (∀ p : Nat, Nat.Prime p → 5000 ≤ p^2 → p^2 ≤ 9000 → p ∈ S) ∧
    Finset.card S = 6 := by
  sorry

end count_primes_with_squares_between_5000_and_9000_l1204_120423


namespace indigo_restaurant_rating_l1204_120421

/-- The average star rating for a restaurant given the number of reviews for each star rating. -/
def averageStarRating (fiveStars fourStars threeStars twoStars : ℕ) : ℚ :=
  let totalStars := 5 * fiveStars + 4 * fourStars + 3 * threeStars + 2 * twoStars
  let totalReviews := fiveStars + fourStars + threeStars + twoStars
  (totalStars : ℚ) / totalReviews

/-- Theorem stating that the average star rating for Indigo Restaurant is 4 stars. -/
theorem indigo_restaurant_rating :
  averageStarRating 6 7 4 1 = 4 := by
  sorry

end indigo_restaurant_rating_l1204_120421


namespace proportion_equation_proof_l1204_120475

theorem proportion_equation_proof (x y : ℝ) (h1 : y ≠ 0) (h2 : 3 * x = 5 * y) :
  x / 5 = y / 3 := by sorry

end proportion_equation_proof_l1204_120475


namespace quadratic_inequality_range_l1204_120427

theorem quadratic_inequality_range :
  {a : ℝ | ∃ x : ℝ, a * x^2 + 2 * x + a < 0} = {a : ℝ | a < 1} := by sorry

end quadratic_inequality_range_l1204_120427


namespace triangle_side_length_l1204_120401

/-- Given a triangle ABC where:
    - a, b, c are sides opposite to angles A, B, C respectively
    - A = 60°
    - b = 4
    - Area of triangle ABC = 4√3
    Prove that a = 4 -/
theorem triangle_side_length (a b c : ℝ) (A : Real) (S : ℝ) : 
  A = π / 3 → 
  b = 4 → 
  S = 4 * Real.sqrt 3 → 
  S = (1 / 2) * b * c * Real.sin A → 
  a = 4 := by
  sorry

end triangle_side_length_l1204_120401


namespace abc_inequality_l1204_120483

theorem abc_inequality (a b c : ℝ) (sum_eq_one : a + b + c = 1) (prod_pos : a * b * c > 0) :
  a * b + b * c + c * a < Real.sqrt (a * b * c) / 2 + 1 / 4 := by
  sorry

end abc_inequality_l1204_120483


namespace expression_equivalence_l1204_120448

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * 
  (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end expression_equivalence_l1204_120448


namespace city_population_ratio_l1204_120494

theorem city_population_ratio :
  ∀ (pop_X pop_Y pop_Z : ℕ),
    pop_X = 3 * pop_Y →
    pop_Y = 2 * pop_Z →
    pop_X / pop_Z = 6 :=
by
  sorry

end city_population_ratio_l1204_120494


namespace no_five_cent_combination_l1204_120409

/-- Represents the types of coins available -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | HalfDollar

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.HalfDollar => 50

/-- A function that takes a list of 5 coins and returns their total value in cents -/
def totalValue (coins : List Coin) : ℕ :=
  coins.map coinValue |>.sum

/-- Theorem stating that it's impossible to select 5 coins with a total value of 5 cents -/
theorem no_five_cent_combination :
  ¬ ∃ (coins : List Coin), coins.length = 5 ∧ totalValue coins = 5 := by
  sorry


end no_five_cent_combination_l1204_120409


namespace root_implies_m_value_l1204_120496

theorem root_implies_m_value (m : ℝ) : 
  (2^2 - m*2 + 2 = 0) → m = 3 := by
sorry

end root_implies_m_value_l1204_120496


namespace shelby_driving_time_l1204_120422

/-- Represents the driving scenario for Shelby --/
structure DrivingScenario where
  sunnySpeed : ℝ  -- Speed in miles per hour when not raining
  rainySpeed : ℝ  -- Speed in miles per hour when raining
  totalDistance : ℝ  -- Total distance traveled in miles
  totalTime : ℝ  -- Total time traveled in hours

/-- Calculates the time spent driving in the rain --/
def timeInRain (scenario : DrivingScenario) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the time spent driving in the rain is 40 minutes --/
theorem shelby_driving_time (scenario : DrivingScenario) 
  (h1 : scenario.sunnySpeed = 40)
  (h2 : scenario.rainySpeed = 25)
  (h3 : scenario.totalDistance = 20)
  (h4 : scenario.totalTime = 0.75)  -- 45 minutes in hours
  : timeInRain scenario = 40 / 60 := by
  sorry


end shelby_driving_time_l1204_120422


namespace fourth_number_proof_l1204_120403

theorem fourth_number_proof (x : ℝ) : 
  3 + 33 + 333 + x = 369.63 → x = 0.63 := by sorry

end fourth_number_proof_l1204_120403


namespace average_temperature_l1204_120476

theorem average_temperature (temperatures : List ℝ) (h1 : temperatures = [18, 21, 19, 22, 20]) :
  temperatures.sum / temperatures.length = 20 := by
sorry

end average_temperature_l1204_120476


namespace cyclist_round_time_l1204_120419

/-- Given a rectangular park with length L and breadth B, prove that a cyclist
    traveling at 12 km/hr along the park's boundary will complete one round in 8 minutes
    when the length to breadth ratio is 1:3 and the area is 120,000 sq. m. -/
theorem cyclist_round_time (L B : ℝ) (h_ratio : B = 3 * L) (h_area : L * B = 120000) :
  (2 * L + 2 * B) / (12000 / 60) = 8 := by
  sorry

end cyclist_round_time_l1204_120419


namespace rosie_pies_l1204_120499

/-- Represents the number of pies that can be made given ingredients and their ratios -/
def pies_made (apples_per_pie oranges_per_pie available_apples available_oranges : ℚ) : ℚ :=
  min (available_apples / apples_per_pie) (available_oranges / oranges_per_pie)

/-- Theorem stating that Rosie can make 9 pies with the given ingredients -/
theorem rosie_pies :
  let apples_per_pie : ℚ := 12 / 3
  let oranges_per_pie : ℚ := 6 / 3
  let available_apples : ℚ := 36
  let available_oranges : ℚ := 18
  pies_made apples_per_pie oranges_per_pie available_apples available_oranges = 9 := by
  sorry

#eval pies_made (12 / 3) (6 / 3) 36 18

end rosie_pies_l1204_120499


namespace x_sixth_power_is_one_l1204_120463

theorem x_sixth_power_is_one (x : ℝ) (h : x + 1/x = 2) : x^6 = 1 := by
  sorry

end x_sixth_power_is_one_l1204_120463


namespace perpendicular_foot_is_circumcenter_l1204_120474

-- Define the plane
variable (π : Set (Fin 3 → ℝ))

-- Define points
variable (P A B C : Fin 3 → ℝ)

-- Define the foot of the perpendicular
variable (F : Fin 3 → ℝ)

-- P is outside the plane
axiom P_outside : P ∉ π

-- A, B, C are in the plane
axiom A_in_plane : A ∈ π
axiom B_in_plane : B ∈ π
axiom C_in_plane : C ∈ π

-- F is in the plane
axiom F_in_plane : F ∈ π

-- PA, PB, PC are equal
axiom equal_distances : norm (P - A) = norm (P - B) ∧ norm (P - B) = norm (P - C)

-- F is on the perpendicular from P to the plane
axiom F_on_perpendicular : ∀ X ∈ π, norm (P - F) ≤ norm (P - X)

-- Define what it means for F to be the circumcenter of triangle ABC
def is_circumcenter (F A B C : Fin 3 → ℝ) : Prop :=
  norm (F - A) = norm (F - B) ∧ norm (F - B) = norm (F - C)

-- The theorem to prove
theorem perpendicular_foot_is_circumcenter :
  is_circumcenter F A B C :=
sorry

end perpendicular_foot_is_circumcenter_l1204_120474


namespace complex_division_problem_l1204_120424

theorem complex_division_problem (a : ℝ) (h : (a^2 - 9 : ℂ) + (a + 3 : ℂ) * I = (0 : ℂ) + b * I) :
  (a + I^19) / (1 + I) = 1 - 2*I :=
sorry

end complex_division_problem_l1204_120424


namespace smallest_square_divisible_by_2016_l1204_120492

theorem smallest_square_divisible_by_2016 :
  ∀ n : ℕ, n > 0 → n^2 % 2016 = 0 → n ≥ 168 :=
by sorry

end smallest_square_divisible_by_2016_l1204_120492


namespace race_head_start_l1204_120450

theorem race_head_start (L : ℝ) (va vb : ℝ) (h : va = (15 / 13) * vb) :
  let H := (L - (13 * L / 15) + (1 / 4 * L))
  H = (23 / 60) * L := by sorry

end race_head_start_l1204_120450


namespace bottle_caps_count_l1204_120434

theorem bottle_caps_count (initial_caps : ℕ) (added_caps : ℕ) : 
  initial_caps = 7 → added_caps = 7 → initial_caps + added_caps = 14 := by
  sorry

end bottle_caps_count_l1204_120434


namespace joan_bought_72_eggs_l1204_120478

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Joan bought -/
def dozens_bought : ℕ := 6

/-- Theorem: Joan bought 72 eggs -/
theorem joan_bought_72_eggs : dozens_bought * eggs_per_dozen = 72 := by
  sorry

end joan_bought_72_eggs_l1204_120478


namespace cubic_root_sum_l1204_120400

/-- Given that p, q, and r are the roots of x³ - 3x - 2 = 0,
    prove that p(q - r)² + q(r - p)² + r(p - q)² = -18 -/
theorem cubic_root_sum (p q r : ℝ) : 
  (p^3 = 3*p + 2) → 
  (q^3 = 3*q + 2) → 
  (r^3 = 3*r + 2) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -18 := by
  sorry

end cubic_root_sum_l1204_120400


namespace cotangent_half_angle_identity_l1204_120435

theorem cotangent_half_angle_identity (α : Real) (m : Real) :
  (Real.tan (α / 2))⁻¹ = m → (1 - Real.sin α) / Real.cos α = (m - 1) / (m + 1) := by
  sorry

end cotangent_half_angle_identity_l1204_120435


namespace plane_division_l1204_120454

/-- The maximum number of regions that can be created by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := sorry

/-- The number of additional regions created by adding a line that intersects all existing lines -/
def additional_regions (n : ℕ) : ℕ := sorry

theorem plane_division (total_lines : ℕ) (parallel_lines : ℕ) 
  (h1 : total_lines = 10) 
  (h2 : parallel_lines = 4) 
  (h3 : parallel_lines ≤ total_lines) :
  max_regions (total_lines - parallel_lines) + 
  (parallel_lines * additional_regions (total_lines - parallel_lines)) = 50 := by
  sorry

end plane_division_l1204_120454


namespace normal_dist_probability_l1204_120437

-- Define the normal distribution
def normal_dist (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, X x = f ((x - μ) / σ)

-- Define the probability function
noncomputable def P (a b : ℝ) (X : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem normal_dist_probability 
  (X : ℝ → ℝ) (μ σ : ℝ) 
  (h1 : normal_dist μ σ X)
  (h2 : P (μ - 2*σ) (μ + 2*σ) X = 0.9544)
  (h3 : P (μ - σ) (μ + σ) X = 0.682)
  (h4 : μ = 4)
  (h5 : σ = 1) :
  P 5 6 X = 0.1359 := by sorry

end normal_dist_probability_l1204_120437


namespace intersection_k_value_l1204_120410

/-- Given two lines that intersect at a point, find the value of k -/
theorem intersection_k_value (m n : ℝ → ℝ) (k : ℝ) :
  (∀ x, m x = 4 * x + 2) →  -- Line m equation
  (∀ x, n x = k * x - 8) →  -- Line n equation
  m (-2) = -6 →             -- Lines intersect at (-2, -6)
  n (-2) = -6 →             -- Lines intersect at (-2, -6)
  k = -1 := by
sorry

end intersection_k_value_l1204_120410


namespace sufficient_condition_B_proper_subset_A_l1204_120413

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | x * m + 1 = 0}

theorem sufficient_condition_B_proper_subset_A :
  ∃ S : Set ℝ, (S = {0, 1/3}) ∧ 
  (∀ m : ℝ, m ∈ S → B m ⊂ A) ∧
  (∃ m : ℝ, m ∉ S ∧ B m ⊂ A) :=
sorry

end sufficient_condition_B_proper_subset_A_l1204_120413


namespace puzzle_solution_l1204_120426

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

theorem puzzle_solution (P Q R S : Digit) 
  (h1 : (P.val * 10 + Q.val) * R.val = S.val * 10 + P.val)
  (h2 : (P.val * 10 + Q.val) + (R.val * 10 + P.val) = S.val * 10 + Q.val) :
  S.val = 1 := by
  sorry

end puzzle_solution_l1204_120426


namespace perpendicular_vectors_l1204_120405

/-- Given vectors a and b in ℝ², if a is perpendicular to b, then the y-coordinate of b is -1 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) :
  a = (1, 3) → b.1 = 3 → b.2 = -1 := by sorry

end perpendicular_vectors_l1204_120405


namespace arrange_four_on_eight_l1204_120462

/-- The number of ways to arrange n people on m chairs in a row,
    such that no two people sit next to each other -/
def arrangePeople (n m : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 120 ways to arrange 4 people on 8 chairs in a row,
    such that no two people sit next to each other -/
theorem arrange_four_on_eight :
  arrangePeople 4 8 = 120 := by
  sorry

end arrange_four_on_eight_l1204_120462


namespace point_on_600_degree_angle_l1204_120411

theorem point_on_600_degree_angle (a : ℝ) : 
  (∃ θ : ℝ, θ = 600 * Real.pi / 180 ∧ 
   (-1 : ℝ) = Real.cos θ ∧ 
   a = Real.sin θ) → 
  a = -Real.sqrt 3 := by
sorry

end point_on_600_degree_angle_l1204_120411


namespace three_hundredth_term_of_specific_sequence_l1204_120402

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem three_hundredth_term_of_specific_sequence :
  let a₁ := 8
  let a₂ := -8
  let r := a₂ / a₁
  geometric_sequence a₁ r 300 = -8 := by
sorry

end three_hundredth_term_of_specific_sequence_l1204_120402


namespace tangent_line_intercept_l1204_120457

/-- Given a curve y = x³ + ax + 1 and a line y = kx + b tangent to the curve at (2, 3), prove b = -15 -/
theorem tangent_line_intercept (a k b : ℝ) : 
  (3 = 2^3 + a*2 + 1) →  -- The curve passes through (2, 3)
  (k = 3*2^2 + a) →      -- The slope of the tangent line equals the derivative at x = 2
  (3 = k*2 + b) →        -- The line passes through (2, 3)
  b = -15 := by
sorry

end tangent_line_intercept_l1204_120457


namespace circle_and_tangent_lines_l1204_120449

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points A, B, and M
def A : ℝ × ℝ := (0, -6)
def B : ℝ × ℝ := (1, -5)
def M : ℝ × ℝ := (2, 8)

-- Define the line l: x - y + 1 = 0
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

-- Theorem statement
theorem circle_and_tangent_lines 
  (C : ℝ × ℝ) -- Center of the circle
  (h1 : C ∈ l) -- Center lies on line l
  (h2 : A ∈ Circle C (|C.1 - A.1|)) -- A is on the circle
  (h3 : B ∈ Circle C (|C.1 - A.1|)) -- B is on the circle
  : 
  -- 1. Standard equation of the circle
  (∀ p : ℝ × ℝ, p ∈ Circle C (|C.1 - A.1|) ↔ (p.1 + 3)^2 + (p.2 + 2)^2 = 25) ∧
  -- 2. Equations of tangent lines
  (∀ p : ℝ × ℝ, (p.1 = 2 ∨ 3*p.1 - 4*p.2 + 26 = 0) ↔ 
    (p ∈ {q : ℝ × ℝ | (q.1 - M.1) * (C.1 - q.1) + (q.2 - M.2) * (C.2 - q.2) = 0} ∧
     p ∈ Circle C (|C.1 - A.1|))) :=
by sorry

end circle_and_tangent_lines_l1204_120449


namespace sample_size_comparison_l1204_120417

/-- Given two samples with different means, prove that the number of elements in the first sample
    is less than or equal to the number of elements in the second sample, based on the combined mean. -/
theorem sample_size_comparison (m n : ℕ) (x_bar y_bar z_bar : ℝ) (a : ℝ) :
  x_bar ≠ y_bar →
  z_bar = a * x_bar + (1 - a) * y_bar →
  0 < a →
  a ≤ 1/2 →
  z_bar = (m * x_bar + n * y_bar) / (m + n : ℝ) →
  m ≤ n := by
  sorry


end sample_size_comparison_l1204_120417


namespace unique_solution_implies_a_equals_two_l1204_120461

theorem unique_solution_implies_a_equals_two (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 := by
  sorry

end unique_solution_implies_a_equals_two_l1204_120461


namespace inscribed_circle_radius_l1204_120498

theorem inscribed_circle_radius (d : ℝ) (h : d = Real.sqrt 12) : 
  let R := d / 2
  let s := R * Real.sqrt 3
  let h := (Real.sqrt 3 / 2) * s
  let a := Real.sqrt (h^2 - (h/2)^2)
  let r := (a * Real.sqrt 3) / 6
  r = 9/8 := by sorry

end inscribed_circle_radius_l1204_120498


namespace intersection_nonempty_iff_a_in_range_l1204_120406

/-- Given sets A and B, prove that their intersection is non-empty if and only if -1 < a < 3 -/
theorem intersection_nonempty_iff_a_in_range (a : ℝ) :
  let A := {x : ℝ | x - 1 > a^2}
  let B := {x : ℝ | x - 4 < 2*a}
  (∃ x, x ∈ A ∩ B) ↔ -1 < a ∧ a < 3 :=
by sorry

end intersection_nonempty_iff_a_in_range_l1204_120406


namespace f_not_monotonic_iff_m_in_open_zero_one_l1204_120431

open Real

-- Define the function f(x) = |log₂(x)|
noncomputable def f (x : ℝ) : ℝ := abs (log x / log 2)

-- Define the property of not being monotonic in an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- State the theorem
theorem f_not_monotonic_iff_m_in_open_zero_one (m : ℝ) :
  m > 0 → (not_monotonic f m (2*m + 1) ↔ 0 < m ∧ m < 1) :=
sorry

end f_not_monotonic_iff_m_in_open_zero_one_l1204_120431


namespace alternating_series_sum_l1204_120458

def alternating_series (n : ℕ) : ℤ := 
  if n % 2 = 0 then (n + 1) else -(n + 1)

def series_sum (n : ℕ) : ℤ := 
  (Finset.range n).sum (λ i => alternating_series i)

theorem alternating_series_sum : series_sum 10001 = -5001 := by
  sorry

end alternating_series_sum_l1204_120458


namespace isosceles_triangle_perimeter_l1204_120440

/-- Represents an isosceles triangle with base 4 and leg length x -/
structure IsoscelesTriangle where
  x : ℝ
  is_root : x^2 - 5*x + 6 = 0
  is_valid : x + x > 4

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.x + 4

theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, perimeter t = 10 := by
  sorry

end isosceles_triangle_perimeter_l1204_120440


namespace factor_problem_l1204_120470

theorem factor_problem (n : ℤ) (f : ℚ) (h1 : n = 9) (h2 : (n + 2) * f = 24 + n) : f = 3 := by
  sorry

end factor_problem_l1204_120470


namespace monotonic_cubic_function_l1204_120486

/-- A function f(x) = x^3 - 3x^2 + ax - 5 is monotonically increasing on ℝ if and only if a ≥ 3 -/
theorem monotonic_cubic_function (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => x^3 - 3*x^2 + a*x - 5)) ↔ a ≥ 3 := by
sorry

end monotonic_cubic_function_l1204_120486


namespace vendor_apples_thrown_away_l1204_120460

/-- Represents the percentage of apples remaining after each operation -/
def apples_remaining (initial_percentage : ℚ) (sell_percentage : ℚ) : ℚ :=
  initial_percentage * (1 - sell_percentage)

/-- Represents the percentage of apples thrown away -/
def apples_thrown (initial_percentage : ℚ) (throw_percentage : ℚ) : ℚ :=
  initial_percentage * throw_percentage

theorem vendor_apples_thrown_away :
  let initial_stock := 1
  let first_day_remaining := apples_remaining initial_stock (60 / 100)
  let first_day_thrown := apples_thrown first_day_remaining (40 / 100)
  let second_day_remaining := apples_remaining (first_day_remaining - first_day_thrown) (50 / 100)
  let second_day_thrown := second_day_remaining
  first_day_thrown + second_day_thrown = 28 / 100 := by
  sorry

end vendor_apples_thrown_away_l1204_120460


namespace solution_set_inequality_l1204_120482

/-- Given that the solution set of ax^2 + bx + c < 0 is (-∞, -1) ∪ (1/2, +∞),
    prove that the solution set of cx^2 - bx + a < 0 is (-2, 1) -/
theorem solution_set_inequality (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + c < 0 ↔ x < -1 ∨ x > 1/2) →
  (∀ x : ℝ, c*x^2 - b*x + a < 0 ↔ -2 < x ∧ x < 1) :=
sorry

end solution_set_inequality_l1204_120482


namespace purely_imaginary_condition_l1204_120441

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_condition (a : ℝ) :
  is_purely_imaginary ((a^2 - 1 : ℝ) + (2 * (a + 1) : ℝ) * I) ↔ a = 1 := by
  sorry

end purely_imaginary_condition_l1204_120441


namespace expand_product_l1204_120404

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end expand_product_l1204_120404


namespace not_all_probabilities_equal_l1204_120443

/-- Represents a student in the sampling process -/
structure Student :=
  (id : Nat)

/-- Represents the sampling process -/
structure SamplingProcess :=
  (totalStudents : Nat)
  (selectedStudents : Nat)
  (excludedStudents : Nat)

/-- Represents the probability of a student being selected -/
def selectionProbability (student : Student) (process : SamplingProcess) : ℝ :=
  sorry

/-- The main theorem stating that not all probabilities are equal -/
theorem not_all_probabilities_equal
  (process : SamplingProcess)
  (h1 : process.totalStudents = 2010)
  (h2 : process.selectedStudents = 50)
  (h3 : process.excludedStudents = 10) :
  ∃ (s1 s2 : Student), selectionProbability s1 process ≠ selectionProbability s2 process :=
sorry

end not_all_probabilities_equal_l1204_120443


namespace two_vertices_same_degree_l1204_120465

-- Define a graph
def Graph (α : Type) := α → α → Prop

-- Define the degree of a vertex in a graph
def degree {α : Type} (G : Graph α) (v : α) : ℕ := sorry

theorem two_vertices_same_degree {α : Type} (G : Graph α) (n : ℕ) (h : Fintype α) :
  (Fintype.card α = n) →
  (∀ v : α, degree G v < n) →
  ∃ u v : α, u ≠ v ∧ degree G u = degree G v :=
sorry

end two_vertices_same_degree_l1204_120465


namespace like_terms_imply_n_eq_one_l1204_120451

/-- Two terms are considered like terms if they have the same variables with the same exponents -/
def like_terms (term1 term2 : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b : ℤ), ∀ (x y : ℤ), term1 x y = a * x^2 * y ∧ term2 x y = b * x^2 * y

/-- If -x^2y^n and 3yx^2 are like terms, then n = 1 -/
theorem like_terms_imply_n_eq_one :
  ∀ n : ℕ, like_terms (λ x y => -x^2 * y^n) (λ x y => 3 * y * x^2) → n = 1 :=
by
  sorry

end like_terms_imply_n_eq_one_l1204_120451


namespace fourth_animal_is_sheep_l1204_120485

def animals : List String := ["Horses", "Cows", "Pigs", "Sheep", "Rabbits", "Squirrels"]

theorem fourth_animal_is_sheep : animals[3] = "Sheep" := by
  sorry

end fourth_animal_is_sheep_l1204_120485


namespace bowling_team_weight_problem_l1204_120428

theorem bowling_team_weight_problem (initial_players : ℕ) (initial_avg_weight : ℝ)
  (new_player2_weight : ℝ) (new_avg_weight : ℝ) :
  initial_players = 7 →
  initial_avg_weight = 121 →
  new_player2_weight = 60 →
  new_avg_weight = 113 →
  ∃ new_player1_weight : ℝ,
    new_player1_weight = 110 ∧
    (initial_players : ℝ) * initial_avg_weight + new_player1_weight + new_player2_weight =
      ((initial_players : ℝ) + 2) * new_avg_weight :=
by sorry

end bowling_team_weight_problem_l1204_120428


namespace second_parentheses_zero_l1204_120416

-- Define the custom operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem second_parentheses_zero : 
  let x : ℝ := Real.sqrt 6
  (diamond x x = (x + x)^2 - (x - x)^2) ∧ (x - x = 0) := by sorry

end second_parentheses_zero_l1204_120416


namespace feathers_per_crown_l1204_120489

theorem feathers_per_crown (total_feathers : ℕ) (total_crowns : ℕ) 
  (h1 : total_feathers = 6538) 
  (h2 : total_crowns = 934) : 
  total_feathers / total_crowns = 7 := by
  sorry

end feathers_per_crown_l1204_120489


namespace probability_three_white_two_black_l1204_120495

/-- The probability of drawing exactly 3 white and 2 black balls from a box
    containing 8 white and 7 black balls, when 5 balls are drawn at random. -/
theorem probability_three_white_two_black : 
  let total_balls : ℕ := 8 + 7
  let white_balls : ℕ := 8
  let black_balls : ℕ := 7
  let drawn_balls : ℕ := 5
  let white_drawn : ℕ := 3
  let black_drawn : ℕ := 2
  let favorable_outcomes : ℕ := (Nat.choose white_balls white_drawn) * (Nat.choose black_balls black_drawn)
  let total_outcomes : ℕ := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes = 8 / 17 := by
sorry

end probability_three_white_two_black_l1204_120495


namespace ladybug_count_l1204_120468

theorem ladybug_count (num_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
  (h1 : num_leaves = 84) 
  (h2 : ladybugs_per_leaf = 139) : 
  num_leaves * ladybugs_per_leaf = 11676 := by
  sorry

end ladybug_count_l1204_120468


namespace min_moves_for_equal_stones_l1204_120473

/-- Operation that can be performed on boxes -/
structure Operation where
  stones_per_box : Nat
  num_boxes : Nat
  extra_stones : Nat

/-- Problem setup -/
def total_boxes : Nat := 2019
def operation : Operation := { stones_per_box := 100, num_boxes := 100, extra_stones := 0 }

/-- Function to calculate the minimum number of moves -/
def min_moves (n : Nat) (op : Operation) : Nat :=
  let d := Nat.gcd n op.num_boxes
  Nat.ceil ((n^2 : Rat) / (d * op.num_boxes))

/-- Theorem statement -/
theorem min_moves_for_equal_stones :
  min_moves total_boxes operation = 40762 := by
  sorry

end min_moves_for_equal_stones_l1204_120473


namespace terminal_side_quadrant_l1204_120414

-- Define the quadrants
inductive Quadrant
  | I
  | II
  | III
  | IV

-- Define a function to determine the quadrant of an angle
def angle_quadrant (α : Real) : Quadrant := sorry

-- Define the theorem
theorem terminal_side_quadrant (α : Real) 
  (h1 : Real.sin α * Real.cos α < 0) 
  (h2 : Real.sin α * Real.tan α > 0) : 
  (angle_quadrant (α / 2) = Quadrant.II) ∨ (angle_quadrant (α / 2) = Quadrant.IV) := by
  sorry

end terminal_side_quadrant_l1204_120414


namespace board_and_sum_properties_l1204_120481

/-- The number of squares in a square board -/
def boardSquares (n : ℕ) : ℕ := n * n

/-- The number of squares in each region separated by the diagonal -/
def regionSquares (n : ℕ) : ℕ := (n * n - n) / 2

/-- The sum of consecutive integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem board_and_sum_properties :
  (boardSquares 11 = 121) ∧
  (regionSquares 11 = 55) ∧
  (sumIntegers 10 = 55) ∧
  (sumIntegers 100 = 5050) :=
sorry

end board_and_sum_properties_l1204_120481


namespace kabadi_players_count_l1204_120445

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 15

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 15

/-- The number of people who play both games -/
def both_games : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 25

/-- Theorem stating that the number of kabadi players is correct given the conditions -/
theorem kabadi_players_count : 
  kabadi_players = total_players - kho_kho_only + both_games :=
by sorry

end kabadi_players_count_l1204_120445


namespace line_equation_through_midpoint_l1204_120456

/-- Given an ellipse and a point M, prove the equation of a line passing through M and intersecting the ellipse at two points where M is their midpoint. -/
theorem line_equation_through_midpoint (x y : ℝ) : 
  let M : ℝ × ℝ := (2, 1)
  let ellipse (x y : ℝ) := x^2 / 16 + y^2 / 4 = 1
  ∃ A B : ℝ × ℝ, 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
    ∃ l : ℝ → ℝ → Prop, 
      (∀ x y, l x y ↔ x + 2*y - 4 = 0) ∧
      l A.1 A.2 ∧ 
      l B.1 B.2 ∧ 
      l M.1 M.2 :=
by sorry

end line_equation_through_midpoint_l1204_120456


namespace four_team_hierarchy_exists_l1204_120491

/-- Represents a volleyball team -/
structure Team :=
  (id : Nat)

/-- Represents the result of a match between two teams -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a tournament with n teams -/
structure Tournament (n : Nat) :=
  (teams : Fin n → Team)
  (results : Fin n → Fin n → MatchResult)
  (results_valid : ∀ i j, i ≠ j → results i j ≠ results j i)

/-- Theorem stating the existence of four teams with the specified winning relationships -/
theorem four_team_hierarchy_exists (t : Tournament 8) :
  ∃ (a b c d : Fin 8),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    t.results a b = MatchResult.Win ∧
    t.results a c = MatchResult.Win ∧
    t.results a d = MatchResult.Win ∧
    t.results b c = MatchResult.Win ∧
    t.results b d = MatchResult.Win ∧
    t.results c d = MatchResult.Win :=
  sorry

end four_team_hierarchy_exists_l1204_120491


namespace real_solution_implies_a_eq_one_no_purely_imaginary_roots_l1204_120420

variable (a : ℝ)

/-- The complex polynomial z^2 - (a+i)z - (i+2) = 0 -/
def f (z : ℂ) : ℂ := z^2 - (a + Complex.I) * z - (Complex.I + 2)

theorem real_solution_implies_a_eq_one :
  (∃ x : ℝ, f a x = 0) → a = 1 := by sorry

theorem no_purely_imaginary_roots :
  ¬∃ y : ℝ, y ≠ 0 ∧ f a (Complex.I * y) = 0 := by sorry

end real_solution_implies_a_eq_one_no_purely_imaginary_roots_l1204_120420


namespace simplify_expression_l1204_120477

theorem simplify_expression :
  -2^2005 + (-2)^2006 + 3^2007 - 2^2008 = -7 * 2^2005 + 3^2007 := by
  sorry

end simplify_expression_l1204_120477


namespace surface_area_of_cuboid_from_cubes_l1204_120469

/-- The surface area of a cuboid formed by three cubes in a row -/
theorem surface_area_of_cuboid_from_cubes (cube_side_length : ℝ) (h : cube_side_length = 8) : 
  let cuboid_length : ℝ := 3 * cube_side_length
  let cuboid_width : ℝ := cube_side_length
  let cuboid_height : ℝ := cube_side_length
  2 * (cuboid_length * cuboid_width + cuboid_length * cuboid_height + cuboid_width * cuboid_height) = 896 := by
  sorry

end surface_area_of_cuboid_from_cubes_l1204_120469


namespace percentage_of_blue_shirts_l1204_120432

theorem percentage_of_blue_shirts (total_students : ℕ) 
  (red_percent green_percent : ℚ) (other_count : ℕ) : 
  total_students = 700 →
  red_percent = 23/100 →
  green_percent = 15/100 →
  other_count = 119 →
  (1 - (red_percent + green_percent + (other_count : ℚ) / total_students)) * 100 = 45 := by
sorry

end percentage_of_blue_shirts_l1204_120432


namespace probability_at_least_10_rubles_l1204_120415

-- Define the total number of tickets
def total_tickets : ℕ := 100

-- Define the number of tickets for each prize category
def tickets_20_rubles : ℕ := 5
def tickets_15_rubles : ℕ := 10
def tickets_10_rubles : ℕ := 15
def tickets_2_rubles : ℕ := 25

-- Define the probability of winning at least 10 rubles
def prob_at_least_10_rubles : ℚ :=
  (tickets_20_rubles + tickets_15_rubles + tickets_10_rubles) / total_tickets

-- Theorem statement
theorem probability_at_least_10_rubles :
  prob_at_least_10_rubles = 30 / 100 := by
  sorry

end probability_at_least_10_rubles_l1204_120415


namespace expected_ones_is_half_l1204_120493

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1/6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ := 
  0 * (prob_not_one ^ num_dice) +
  1 * (num_dice.choose 1 * prob_one * (prob_not_one ^ 2)) +
  2 * (num_dice.choose 2 * (prob_one ^ 2) * prob_not_one) +
  3 * (prob_one ^ num_dice)

theorem expected_ones_is_half : expected_ones = 1/2 := by sorry

end expected_ones_is_half_l1204_120493


namespace equation_solution_l1204_120464

theorem equation_solution : 
  ∃ x : ℝ, (2 * x / (x - 2) + 3 / (2 - x) = 1) ∧ (x = 1) :=
by
  sorry

end equation_solution_l1204_120464


namespace halfway_fraction_l1204_120497

theorem halfway_fraction (a b c d : ℕ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 7) :
  (a / b + c / d) / 2 = 41 / 56 :=
sorry

end halfway_fraction_l1204_120497


namespace new_average_after_grace_marks_l1204_120455

theorem new_average_after_grace_marks 
  (num_students : ℕ) 
  (original_average : ℚ) 
  (grace_marks : ℚ) 
  (h1 : num_students = 35) 
  (h2 : original_average = 37) 
  (h3 : grace_marks = 3) : 
  (num_students * original_average + num_students * grace_marks) / num_students = 40 := by
sorry

end new_average_after_grace_marks_l1204_120455


namespace right_triangle_area_l1204_120408

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 5) (h3 : a = 4) :
  (1/2) * a * b = 6 := by
  sorry

end right_triangle_area_l1204_120408


namespace swimmer_speed_l1204_120452

/-- The speed of a swimmer in still water, given his downstream and upstream speeds and distances. -/
theorem swimmer_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) :
  downstream_distance = 62 →
  downstream_time = 10 →
  upstream_distance = 84 →
  upstream_time = 14 →
  ∃ (v_m v_s : ℝ),
    v_m + v_s = downstream_distance / downstream_time ∧
    v_m - v_s = upstream_distance / upstream_time ∧
    v_m = 6.1 :=
by sorry

end swimmer_speed_l1204_120452


namespace diana_grace_age_ratio_l1204_120446

/-- The ratio of Diana's age to Grace's age -/
def age_ratio (diana_age : ℕ) (grace_age : ℕ) : ℚ :=
  diana_age / grace_age

/-- Grace's current age -/
def grace_current_age (grace_last_year : ℕ) : ℕ :=
  grace_last_year + 1

theorem diana_grace_age_ratio :
  let diana_age : ℕ := 8
  let grace_last_year : ℕ := 3
  age_ratio diana_age (grace_current_age grace_last_year) = 2 := by
sorry

end diana_grace_age_ratio_l1204_120446


namespace smallest_a_for_sum_of_squares_l1204_120487

theorem smallest_a_for_sum_of_squares (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*a*x + a^2 = 0 → 
   ∃ x1 x2 : ℝ, x1^2 + x2^2 = 0.28 ∧ x1 ≠ x2 ∧ 
   (∀ y : ℝ, y^2 - 3*a*y + a^2 = 0 → y = x1 ∨ y = x2)) →
  a = -0.2 ∧ 
  (∀ b : ℝ, b < -0.2 → 
   ¬(∀ x : ℝ, x^2 - 3*b*x + b^2 = 0 → 
     ∃ x1 x2 : ℝ, x1^2 + x2^2 = 0.28 ∧ x1 ≠ x2 ∧ 
     (∀ y : ℝ, y^2 - 3*b*y + b^2 = 0 → y = x1 ∨ y = x2))) :=
by sorry

end smallest_a_for_sum_of_squares_l1204_120487


namespace unique_solution_cubic_system_l1204_120418

theorem unique_solution_cubic_system (x y z : ℝ) 
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 3)
  (h3 : x^3 + y^3 + z^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end unique_solution_cubic_system_l1204_120418


namespace quadratic_square_plus_constant_l1204_120433

theorem quadratic_square_plus_constant :
  ∃ k : ℤ, ∀ z : ℂ, z^2 - 6*z + 17 = (z - 3)^2 + k := by
  sorry

end quadratic_square_plus_constant_l1204_120433


namespace no_quadratic_term_implies_k_equals_three_l1204_120447

/-- 
Given an algebraic expression in x and y: (-3kxy+3y)+(9xy-8x+1),
prove that if there is no quadratic term, then k = 3.
-/
theorem no_quadratic_term_implies_k_equals_three (k : ℚ) : 
  (∀ x y : ℚ, (-3*k*x*y + 3*y) + (9*x*y - 8*x + 1) = (-3*k + 9)*x*y + 3*y - 8*x + 1) →
  (∀ x y : ℚ, (-3*k + 9)*x*y + 3*y - 8*x + 1 = 3*y - 8*x + 1) →
  k = 3 := by
sorry

end no_quadratic_term_implies_k_equals_three_l1204_120447


namespace solve_equations_l1204_120436

theorem solve_equations :
  (∃ x1 x2 : ℝ, 2 * x1 * (x1 - 1) = 1 ∧ 2 * x2 * (x2 - 1) = 1 ∧
    x1 = (1 + Real.sqrt 3) / 2 ∧ x2 = (1 - Real.sqrt 3) / 2) ∧
  (∃ y1 y2 : ℝ, y1^2 + 8*y1 + 7 = 0 ∧ y2^2 + 8*y2 + 7 = 0 ∧
    y1 = -7 ∧ y2 = -1) :=
by sorry

end solve_equations_l1204_120436


namespace grants_yearly_expense_l1204_120425

/-- Grant's yearly newspaper delivery expense --/
def grants_expense : ℝ := 200

/-- Juanita's daily expense from Monday to Saturday --/
def juanita_weekday_expense : ℝ := 0.5

/-- Juanita's Sunday expense --/
def juanita_sunday_expense : ℝ := 2

/-- Number of weeks in a year --/
def weeks_per_year : ℕ := 52

/-- Difference between Juanita's and Grant's yearly expenses --/
def expense_difference : ℝ := 60

/-- Theorem stating Grant's yearly newspaper delivery expense --/
theorem grants_yearly_expense : 
  grants_expense = 
    weeks_per_year * (6 * juanita_weekday_expense + juanita_sunday_expense) - expense_difference :=
by sorry

end grants_yearly_expense_l1204_120425


namespace inequality_solution_l1204_120467

theorem inequality_solution (x : ℝ) : (x - 2) / (x - 4) ≥ 3 ↔ x < 4 := by
  sorry

end inequality_solution_l1204_120467


namespace wednesday_work_time_l1204_120466

/-- Represents the work time in minutes for each day of the week -/
structure WorkWeek where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ
  friday : ℚ

/-- Calculates the total work time for the week in minutes -/
def totalWorkTime (w : WorkWeek) : ℚ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- Converts hours to minutes -/
def hoursToMinutes (hours : ℚ) : ℚ :=
  hours * 60

theorem wednesday_work_time (w : WorkWeek) : 
  w.monday = hoursToMinutes (3/4) ∧ 
  w.tuesday = hoursToMinutes (1/2) ∧ 
  w.thursday = hoursToMinutes (5/6) ∧ 
  w.friday = 75 ∧ 
  totalWorkTime w = hoursToMinutes 4 → 
  w.wednesday = 40 := by
sorry

end wednesday_work_time_l1204_120466


namespace sum_of_segments_9_9_l1204_120484

/-- The sum of lengths of all line segments formed by dividing a line segment into equal parts -/
def sum_of_segments (total_length : ℕ) (num_divisions : ℕ) : ℕ :=
  let unit_length := total_length / num_divisions
  let sum_short_segments := (num_divisions - 1) * num_divisions * unit_length
  let sum_long_segments := (num_divisions * (num_divisions + 1) * unit_length) / 2
  sum_short_segments + sum_long_segments

/-- Theorem: The sum of lengths of all line segments formed by dividing a line segment of length 9 into 9 equal parts is equal to 165 -/
theorem sum_of_segments_9_9 :
  sum_of_segments 9 9 = 165 := by
  sorry

end sum_of_segments_9_9_l1204_120484


namespace min_value_geometric_sequence_l1204_120471

theorem min_value_geometric_sequence (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
    (h_condition : a 2 * a 3 * a 4 = a 2 + a 3 + a 4) : 
  a 3 ≥ Real.sqrt 3 ∧ ∃ a' : ℕ → ℝ, (∀ n, a' n > 0) ∧ 
    (∃ q' : ℝ, q' > 0 ∧ ∀ n, a' (n + 1) = a' n * q') ∧
    (a' 2 * a' 3 * a' 4 = a' 2 + a' 3 + a' 4) ∧
    (a' 3 = Real.sqrt 3) :=
by sorry

end min_value_geometric_sequence_l1204_120471


namespace angle_equality_l1204_120480

-- Define the problem statement
theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (15 * π / 180) = Real.cos θ + Real.sin θ) : 
  θ = 15 * π / 180 :=
sorry

end angle_equality_l1204_120480


namespace min_sum_of_six_l1204_120490

def consecutive_numbers (start : ℕ) : List ℕ :=
  List.range 11 |>.map (λ i => start + i)

theorem min_sum_of_six (start : ℕ) :
  (consecutive_numbers start).length = 11 →
  start + (start + 10) = 90 →
  ∃ (subset : List ℕ), subset.length = 6 ∧ 
    subset.all (λ x => x ∈ consecutive_numbers start) ∧
    subset.sum = 90 ∧
    ∀ (other_subset : List ℕ), other_subset.length = 6 →
      other_subset.all (λ x => x ∈ consecutive_numbers start) →
      other_subset.sum ≥ 90 :=
by
  sorry

#check min_sum_of_six

end min_sum_of_six_l1204_120490


namespace max_value_of_expression_l1204_120479

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^3*(b+c) + b^3*(c+a) + c^3*(a+b)) / ((a+b+c)^4 - 79*(a*b*c)^(4/3))
  A ≤ 3 ∧ (A = 3 ↔ a = b ∧ b = c) :=
by sorry

end max_value_of_expression_l1204_120479


namespace tree_heights_theorem_l1204_120442

/-- Represents the heights of 5 trees -/
structure TreeHeights where
  h1 : ℤ
  h2 : ℤ
  h3 : ℤ
  h4 : ℤ
  h5 : ℤ

/-- The condition that each tree is either twice as tall or half as tall as the one to its right -/
def validHeights (h : TreeHeights) : Prop :=
  (h.h1 = 2 * h.h2 ∨ 2 * h.h1 = h.h2) ∧
  (h.h2 = 2 * h.h3 ∨ 2 * h.h2 = h.h3) ∧
  (h.h3 = 2 * h.h4 ∨ 2 * h.h3 = h.h4) ∧
  (h.h4 = 2 * h.h5 ∨ 2 * h.h4 = h.h5)

/-- The average height of the trees -/
def averageHeight (h : TreeHeights) : ℚ :=
  (h.h1 + h.h2 + h.h3 + h.h4 + h.h5) / 5

/-- The main theorem -/
theorem tree_heights_theorem (h : TreeHeights) 
  (h_valid : validHeights h) 
  (h_second : h.h2 = 11) : 
  averageHeight h = 121 / 5 := by
  sorry

end tree_heights_theorem_l1204_120442


namespace hannah_spending_l1204_120429

-- Define the quantities and prices
def num_sweatshirts : ℕ := 3
def price_sweatshirt : ℚ := 15
def num_tshirts : ℕ := 2
def price_tshirt : ℚ := 10
def num_socks : ℕ := 4
def price_socks : ℚ := 5
def price_jacket : ℚ := 50
def discount_rate : ℚ := 0.1

-- Define the total cost before discount
def total_cost_before_discount : ℚ :=
  num_sweatshirts * price_sweatshirt +
  num_tshirts * price_tshirt +
  num_socks * price_socks +
  price_jacket

-- Define the discount amount
def discount_amount : ℚ := discount_rate * total_cost_before_discount

-- Define the final cost after discount
def final_cost : ℚ := total_cost_before_discount - discount_amount

-- Theorem statement
theorem hannah_spending :
  final_cost = 121.5 := by sorry

end hannah_spending_l1204_120429


namespace system_solutions_l1204_120472

def has_solution (a : ℝ) (x y : ℝ) : Prop :=
  x > 0 ∧ y ≥ 0 ∧ 2*y - 2 = a*(x - 2) ∧ 4*y / (|x| + x) = Real.sqrt y

theorem system_solutions :
  ∀ a : ℝ,
    (a < 0 ∨ a > 1 → 
      has_solution a (2 - 2/a) 0 ∧ has_solution a 2 1) ∧
    (0 ≤ a ∧ a ≤ 1 → 
      has_solution a 2 1) ∧
    ((1 < a ∧ a < 2) ∨ a > 2 → 
      has_solution a (2 - 2/a) 0 ∧ 
      has_solution a 2 1 ∧ 
      has_solution a (2*a - 2) ((a-1)^2)) :=
sorry

end system_solutions_l1204_120472


namespace power_of_product_l1204_120459

theorem power_of_product (a : ℝ) : (-2 * a^4)^3 = -8 * a^12 := by
  sorry

end power_of_product_l1204_120459


namespace marias_minimum_score_l1204_120439

/-- The minimum score needed in the fifth term to achieve a given average -/
def minimum_fifth_score (score1 score2 score3 score4 : ℝ) (required_average : ℝ) : ℝ :=
  5 * required_average - (score1 + score2 + score3 + score4)

/-- Theorem: Maria's minimum required score for the 5th term is 101% -/
theorem marias_minimum_score :
  minimum_fifth_score 84 80 82 78 85 = 101 := by
  sorry

end marias_minimum_score_l1204_120439


namespace conference_attendees_l1204_120488

theorem conference_attendees :
  ∃ n : ℕ,
    n < 50 ∧
    n % 8 = 5 ∧
    n % 6 = 3 ∧
    n = 45 := by
  sorry

end conference_attendees_l1204_120488


namespace factor_and_multiple_greatest_factor_smallest_multiple_smallest_multiple_one_prime_sum_10_product_21_prime_sum_20_product_91_l1204_120438

-- (1)
theorem factor_and_multiple (n : ℕ) : 
  n ∣ 42 ∧ 7 ∣ n ∧ 2 ∣ n ∧ 3 ∣ n → n = 42 := by sorry

-- (2)
theorem greatest_factor_smallest_multiple (n : ℕ) :
  (∀ m : ℕ, m ∣ n → m ≤ 18) ∧ (∀ k : ℕ, n ∣ k → k ≥ 18) → n = 18 := by sorry

-- (3)
theorem smallest_multiple_one (n : ℕ) :
  (∀ k : ℕ, n ∣ k → k ≥ 1) → n = 1 := by sorry

-- (4)
theorem prime_sum_10_product_21 (p q : ℕ) :
  Prime p ∧ Prime q ∧ p + q = 10 ∧ p * q = 21 → (p = 3 ∧ q = 7) ∨ (p = 7 ∧ q = 3) := by sorry

-- (5)
theorem prime_sum_20_product_91 (p q : ℕ) :
  Prime p ∧ Prime q ∧ p + q = 20 ∧ p * q = 91 → (p = 13 ∧ q = 7) ∨ (p = 7 ∧ q = 13) := by sorry

end factor_and_multiple_greatest_factor_smallest_multiple_smallest_multiple_one_prime_sum_10_product_21_prime_sum_20_product_91_l1204_120438


namespace elizabeth_study_time_l1204_120453

/-- The total study time for Elizabeth given her time spent on science and math tests -/
def total_study_time (science_time math_time : ℕ) : ℕ :=
  science_time + math_time

/-- Theorem stating that Elizabeth's total study time is 60 minutes -/
theorem elizabeth_study_time :
  total_study_time 25 35 = 60 := by
  sorry

end elizabeth_study_time_l1204_120453


namespace apple_heavier_than_kiwi_l1204_120444

-- Define a type for fruits
inductive Fruit
  | Apple
  | Banana
  | Kiwi

-- Define a weight relation between fruits
def heavier_than (a b : Fruit) : Prop := sorry

-- State the theorem
theorem apple_heavier_than_kiwi 
  (h1 : heavier_than Fruit.Apple Fruit.Banana) 
  (h2 : heavier_than Fruit.Banana Fruit.Kiwi) : 
  heavier_than Fruit.Apple Fruit.Kiwi := by
  sorry

end apple_heavier_than_kiwi_l1204_120444


namespace plane_contains_points_plane_uniqueness_l1204_120407

/-- A plane in 3D space defined by its equation coefficients -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- A point in 3D space with integer coordinates -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Check if a point lies on a plane -/
def Point3D.liesOn (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- The specific plane we're proving about -/
def targetPlane : Plane := {
  A := 1
  B := 3
  C := -2
  D := -11
  A_pos := by simp
  gcd_one := by sorry
}

/-- The three points given in the problem -/
def p1 : Point3D := ⟨0, 3, -1⟩
def p2 : Point3D := ⟨2, 3, 1⟩
def p3 : Point3D := ⟨4, 1, 0⟩

/-- The main theorem stating that the target plane contains the three given points -/
theorem plane_contains_points : 
  p1.liesOn targetPlane ∧ p2.liesOn targetPlane ∧ p3.liesOn targetPlane :=
by sorry

/-- The theorem stating that the target plane is unique -/
theorem plane_uniqueness (plane : Plane) :
  p1.liesOn plane ∧ p2.liesOn plane ∧ p3.liesOn plane → plane = targetPlane :=
by sorry

end plane_contains_points_plane_uniqueness_l1204_120407
