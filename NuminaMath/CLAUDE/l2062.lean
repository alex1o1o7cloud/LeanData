import Mathlib

namespace sum_of_exponents_l2062_206284

theorem sum_of_exponents (a b c : ℕ+) : 
  4^(a : ℕ) * 5^(b : ℕ) * 6^(c : ℕ) = 8^8 * 9^9 * 10^10 → 
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 36 := by
sorry

end sum_of_exponents_l2062_206284


namespace second_shift_participation_theorem_l2062_206296

/-- The percentage of second shift employees participating in the pension program -/
def second_shift_participation_rate : ℝ := 40

theorem second_shift_participation_theorem :
  let total_employees : ℕ := 60 + 50 + 40
  let first_shift : ℕ := 60
  let second_shift : ℕ := 50
  let third_shift : ℕ := 40
  let first_shift_rate : ℝ := 20
  let third_shift_rate : ℝ := 10
  let total_participation_rate : ℝ := 24
  let first_shift_participants : ℝ := first_shift_rate / 100 * first_shift
  let third_shift_participants : ℝ := third_shift_rate / 100 * third_shift
  let total_participants : ℝ := total_participation_rate / 100 * total_employees
  let second_shift_participants : ℝ := total_participants - first_shift_participants - third_shift_participants
  second_shift_participation_rate = second_shift_participants / second_shift * 100 :=
by
  sorry

end second_shift_participation_theorem_l2062_206296


namespace percent_of_percent_l2062_206249

theorem percent_of_percent (y : ℝ) : 0.21 * y = 0.3 * (0.7 * y) := by sorry

end percent_of_percent_l2062_206249


namespace unique_prime_value_l2062_206282

def f (n : ℕ) : ℤ := n^3 - 9*n^2 + 23*n - 15

theorem unique_prime_value : ∃! (n : ℕ), n > 0 ∧ Nat.Prime (Int.natAbs (f n)) :=
sorry

end unique_prime_value_l2062_206282


namespace factor_difference_of_squares_l2062_206285

theorem factor_difference_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4*y) * (5 + 4*y) := by
  sorry

end factor_difference_of_squares_l2062_206285


namespace first_term_of_geometric_sequence_l2062_206288

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem first_term_of_geometric_sequence (a : ℕ → ℝ) :
  IsGeometricSequence a →
  a 2 = 16 →
  a 4 = 128 →
  a 0 = 4 := by
  sorry

end first_term_of_geometric_sequence_l2062_206288


namespace product_of_extreme_roots_l2062_206237

-- Define the equation
def equation (x : ℝ) : Prop := x * |x| - 5 * |x| + 6 = 0

-- Define the set of roots
def roots : Set ℝ := {x : ℝ | equation x}

-- Statement to prove
theorem product_of_extreme_roots :
  ∃ (max_root min_root : ℝ),
    max_root ∈ roots ∧
    min_root ∈ roots ∧
    (∀ x ∈ roots, x ≤ max_root) ∧
    (∀ x ∈ roots, x ≥ min_root) ∧
    max_root * min_root = -3 :=
sorry

end product_of_extreme_roots_l2062_206237


namespace ten_mile_taxi_cost_l2062_206289

def taxi_cost (initial_cost : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  initial_cost + cost_per_mile * distance

theorem ten_mile_taxi_cost :
  taxi_cost 2 0.3 10 = 5 := by
  sorry

end ten_mile_taxi_cost_l2062_206289


namespace unique_solution_l2062_206272

def A (x : ℝ) : Set ℝ := {x^2, x+1, -3}
def B (x : ℝ) : Set ℝ := {x-5, 2*x-1, x^2+1}

theorem unique_solution : 
  ∃! x : ℝ, A x ∩ B x = {-3} ∧ x = -1 := by sorry

end unique_solution_l2062_206272


namespace concentric_circles_radii_difference_l2062_206244

theorem concentric_circles_radii_difference 
  (r R : ℝ) 
  (h_positive : r > 0) 
  (h_ratio : π * R^2 = 3 * π * r^2) : 
  ∃ ε > 0, |R - r - 0.73 * r| < ε := by
sorry

end concentric_circles_radii_difference_l2062_206244


namespace even_function_four_zeroes_range_l2062_206283

/-- An even function is a function that is symmetric about the y-axis -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- A function has four distinct zeroes if there exist four different real numbers that make the function equal to zero -/
def HasFourDistinctZeroes (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0

theorem even_function_four_zeroes_range (f : ℝ → ℝ) (h_even : EvenFunction f) :
  (∃ m : ℝ, HasFourDistinctZeroes (fun x => f x - m)) →
  (∀ m : ℝ, m ≠ 0 → ∃ x : ℝ, f x = m) ∧ (¬∃ x : ℝ, f x = 0) :=
sorry

end even_function_four_zeroes_range_l2062_206283


namespace greatest_divisor_four_consecutive_integers_l2062_206276

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 0 ∧ m ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ k : ℕ, k > m → ¬(∀ p : ℕ, p > 0 → k ∣ (p * (p + 1) * (p + 2) * (p + 3))) →
  m = 24 :=
by sorry

end greatest_divisor_four_consecutive_integers_l2062_206276


namespace first_group_size_correct_l2062_206206

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 33

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group works -/
def second_group_days : ℕ := 11

/-- The number of hours per day the second group works -/
def second_group_hours : ℕ := 6

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_correct :
  first_group_size * first_group_hours * first_group_days =
  second_group_size * second_group_hours * second_group_days :=
by sorry

end first_group_size_correct_l2062_206206


namespace quadratic_intersection_properties_l2062_206242

/-- A quadratic function f(x) = x^2 + 2x + b intersecting both coordinate axes at three points -/
structure QuadraticIntersection (b : ℝ) :=
  (intersects_axes : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + b = 0 ∧ x₂^2 + 2*x₂ + b = 0)
  (y_intercept : b ≠ 0)

/-- The circle passing through the three intersection points -/
def intersection_circle (b : ℝ) (h : QuadraticIntersection b) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - (b + 1)*p.2 + b = 0}

/-- Main theorem: properties of the quadratic function and its intersection circle -/
theorem quadratic_intersection_properties (b : ℝ) (h : QuadraticIntersection b) :
  b < 1 ∧ 
  ∀ (p : ℝ × ℝ), p ∈ intersection_circle b h ↔ p.1^2 + p.2^2 + 2*p.1 - (b + 1)*p.2 + b = 0 :=
sorry

end quadratic_intersection_properties_l2062_206242


namespace triangle_area_tripled_sides_l2062_206232

/-- Given a triangle, prove that tripling its sides multiplies its area by 9 -/
theorem triangle_area_tripled_sides (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let s' := ((3 * a) + (3 * b) + (3 * c)) / 2
  let area' := Real.sqrt (s' * (s' - 3 * a) * (s' - 3 * b) * (s' - 3 * c))
  area' = 9 * area := by
  sorry


end triangle_area_tripled_sides_l2062_206232


namespace sea_creatures_lost_l2062_206209

-- Define the initial number of items collected
def sea_stars : ℕ := 34
def seashells : ℕ := 21
def snails : ℕ := 29

-- Define the number of items left at the end
def items_left : ℕ := 59

-- Define the total number of items collected
def total_collected : ℕ := sea_stars + seashells + snails

-- Define the number of items lost
def items_lost : ℕ := total_collected - items_left

-- Theorem statement
theorem sea_creatures_lost : items_lost = 25 := by
  sorry

end sea_creatures_lost_l2062_206209


namespace lee_quiz_probability_l2062_206273

theorem lee_quiz_probability (p : ℚ) (h : p = 5/8) :
  1 - p = 3/8 := by
  sorry

end lee_quiz_probability_l2062_206273


namespace min_odd_in_A_P_l2062_206243

/-- The set A_P for a polynomial P -/
def A_P (P : ℝ → ℝ) : Set ℝ := {x : ℝ | ∃ c : ℝ, P x = c}

/-- A polynomial is of degree 8 -/
def is_degree_8 (P : ℝ → ℝ) : Prop :=
  ∃ a₈ a₇ a₆ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ, a₈ ≠ 0 ∧
    ∀ x, P x = a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + 
           a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

theorem min_odd_in_A_P (P : ℝ → ℝ) (h : is_degree_8 P) (h8 : 8 ∈ A_P P) :
  ∃ x ∈ A_P P, Odd x :=
sorry

end min_odd_in_A_P_l2062_206243


namespace divisibility_by_37_l2062_206278

theorem divisibility_by_37 (a b c : ℕ) :
  (37 ∣ (100 * a + 10 * b + c)) →
  (37 ∣ (100 * b + 10 * c + a)) ∧
  (37 ∣ (100 * c + 10 * a + b)) := by
  sorry

end divisibility_by_37_l2062_206278


namespace sin_five_pi_sixths_minus_two_alpha_l2062_206252

theorem sin_five_pi_sixths_minus_two_alpha 
  (h : Real.cos (π / 6 - α) = 1 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -7 / 9 := by
  sorry

end sin_five_pi_sixths_minus_two_alpha_l2062_206252


namespace intersection_A_B_l2062_206202

-- Define set A
def A : Set ℝ := {x : ℝ | |x| < 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end intersection_A_B_l2062_206202


namespace optimal_allocation_l2062_206297

/-- Represents the production capacity of workers in a workshop --/
structure Workshop where
  total_workers : ℕ
  bolts_per_worker : ℕ
  nuts_per_worker : ℕ
  nuts_per_bolt : ℕ

/-- Represents the allocation of workers to bolt and nut production --/
structure WorkerAllocation where
  bolt_workers : ℕ
  nut_workers : ℕ

/-- Checks if a given allocation is valid for the workshop --/
def is_valid_allocation (w : Workshop) (a : WorkerAllocation) : Prop :=
  a.bolt_workers + a.nut_workers = w.total_workers ∧
  a.bolt_workers * w.bolts_per_worker * w.nuts_per_bolt = a.nut_workers * w.nuts_per_worker

/-- The theorem stating the optimal allocation for the given workshop conditions --/
theorem optimal_allocation (w : Workshop) 
    (h1 : w.total_workers = 28)
    (h2 : w.bolts_per_worker = 12)
    (h3 : w.nuts_per_worker = 18)
    (h4 : w.nuts_per_bolt = 2) :
  ∃ (a : WorkerAllocation), 
    is_valid_allocation w a ∧ 
    a.bolt_workers = 12 ∧ 
    a.nut_workers = 16 := by
  sorry

end optimal_allocation_l2062_206297


namespace haley_final_lives_l2062_206250

/-- Calculate the final number of lives in a video game scenario -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Theorem stating that for the given scenario, the final number of lives is 46 -/
theorem haley_final_lives :
  final_lives 14 4 36 = 46 := by
  sorry

end haley_final_lives_l2062_206250


namespace transformed_area_theorem_l2062_206212

/-- A 2x2 matrix representing the transformation --/
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 5, -2]

/-- The area of the original region T --/
def original_area : ℝ := 12

/-- Theorem stating that applying the transformation matrix to a region with area 12 results in a new region with area 312 --/
theorem transformed_area_theorem :
  abs (Matrix.det transformation_matrix) * original_area = 312 := by
  sorry

#check transformed_area_theorem

end transformed_area_theorem_l2062_206212


namespace bills_omelet_time_l2062_206207

/-- The time it takes to prepare and cook omelets -/
def total_time (pepper_chop_time onion_chop_time cheese_grate_time assemble_cook_time : ℕ) 
               (num_peppers num_onions num_omelets : ℕ) : ℕ :=
  (pepper_chop_time * num_peppers) + 
  (onion_chop_time * num_onions) + 
  ((cheese_grate_time + assemble_cook_time) * num_omelets)

/-- Theorem stating that Bill's total preparation and cooking time for five omelets is 50 minutes -/
theorem bills_omelet_time : 
  total_time 3 4 1 5 4 2 5 = 50 := by
  sorry

end bills_omelet_time_l2062_206207


namespace g_value_at_half_l2062_206213

/-- Given a function g : ℝ → ℝ satisfying the equation
    g(x) - 3g(1/x) = 4^x + e^x for all x ≠ 0,
    prove that g(1/2) = (3e^2 - 13√e + 82) / 8 -/
theorem g_value_at_half (g : ℝ → ℝ) 
    (h : ∀ x : ℝ, x ≠ 0 → g x - 3 * g (1/x) = 4^x + Real.exp x) : 
  g (1/2) = (3 * Real.exp 2 - 13 * Real.sqrt (Real.exp 1) + 82) / 8 := by
sorry

end g_value_at_half_l2062_206213


namespace f_of_tan_squared_l2062_206214

noncomputable def f (x : ℝ) : ℝ := 1 / (((x / (x - 1)) - 1) / (x / (x - 1)))^2

theorem f_of_tan_squared (t : ℝ) (h : 0 ≤ t ∧ t ≤ π/4) :
  f (Real.tan t)^2 = (Real.cos (2*t) / Real.sin t^2)^2 := by
  sorry

end f_of_tan_squared_l2062_206214


namespace arithmetic_geometric_sequence_first_term_l2062_206291

/-- An arithmetic sequence with common difference 2 where a_1, a_2, and a_4 form a geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧ 
  (a 2)^2 = a 1 * a 4

theorem arithmetic_geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h : ArithmeticGeometricSequence a) : 
  a 1 = 2 := by
sorry

end arithmetic_geometric_sequence_first_term_l2062_206291


namespace milk_left_over_problem_l2062_206219

/-- Calculates the amount of milk left over given the total milk production,
    percentage consumed by kids, and percentage of remainder used for cooking. -/
def milk_left_over (total_milk : ℝ) (kids_consumption_percent : ℝ) (cooking_percent : ℝ) : ℝ :=
  let remaining_after_kids := total_milk * (1 - kids_consumption_percent)
  let used_for_cooking := remaining_after_kids * cooking_percent
  remaining_after_kids - used_for_cooking

/-- Proves that given 16 cups of milk, with 75% consumed by kids and 50% of the remainder
    used for cooking, the amount of milk left over is 2 cups. -/
theorem milk_left_over_problem :
  milk_left_over 16 0.75 0.50 = 2 := by
  sorry

end milk_left_over_problem_l2062_206219


namespace woojung_high_school_students_l2062_206290

theorem woojung_high_school_students (first_year : ℕ) (non_first_year : ℕ) : 
  non_first_year = 954 → 
  first_year = non_first_year - 468 → 
  first_year + non_first_year = 1440 := by
sorry

end woojung_high_school_students_l2062_206290


namespace positive_net_return_l2062_206275

/-- Represents the annual interest rate of a mortgage loan as a percentage -/
def mortgage_rate : ℝ := 12.5

/-- Represents the annual dividend rate of preferred shares as a percentage -/
def dividend_rate : ℝ := 17

/-- Calculates the net return from keeping shares and taking a mortgage loan -/
def net_return (dividend : ℝ) (mortgage : ℝ) : ℝ := dividend - mortgage

/-- Theorem stating that the net return is positive given the specified rates -/
theorem positive_net_return : net_return dividend_rate mortgage_rate > 0 := by
  sorry

end positive_net_return_l2062_206275


namespace youtube_video_dislikes_l2062_206251

theorem youtube_video_dislikes (initial_likes : ℕ) (initial_dislikes : ℕ) (additional_dislikes : ℕ) : 
  initial_likes = 3000 →
  initial_dislikes = initial_likes / 2 + 100 →
  additional_dislikes = 1000 →
  initial_dislikes + additional_dislikes = 2600 :=
by
  sorry

end youtube_video_dislikes_l2062_206251


namespace updated_mean_after_decrement_l2062_206270

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 15 →
  (n * original_mean - n * decrement) / n = 185 := by
  sorry

end updated_mean_after_decrement_l2062_206270


namespace min_value_a2_plus_b2_l2062_206238

theorem min_value_a2_plus_b2 (a b : ℝ) (h : a^2 + 2*a*b - 3*b^2 = 1) :
  a^2 + b^2 ≥ (Real.sqrt 5 + 1) / 4 :=
by sorry

end min_value_a2_plus_b2_l2062_206238


namespace circle_radius_from_area_l2062_206248

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 36 * Real.pi) :
  A = Real.pi * r^2 → r = 6 := by
  sorry

end circle_radius_from_area_l2062_206248


namespace inequalities_with_squares_and_roots_l2062_206229

theorem inequalities_with_squares_and_roots (a b : ℝ) : 
  (a > 0 ∧ b > 0 ∧ a^2 - b^2 = 1 → a - b ≤ 1) ∧
  (a > 0 ∧ b > 0 ∧ Real.sqrt a - Real.sqrt b = 1 → a - b ≥ 1) := by
  sorry

end inequalities_with_squares_and_roots_l2062_206229


namespace propositions_truth_values_l2062_206203

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the lines and planes
variable (m n : Line) (α β : Plane)

-- State the theorem
theorem propositions_truth_values :
  -- Proposition ① is false
  ¬(∀ m n α β, parallelLP m α → parallelLP n β → parallelPP α β → parallel m n) ∧
  -- Proposition ② is true
  (∀ m n α β, parallel m n → contains α m → perpendicularLP n β → perpendicularPP α β) ∧
  -- Proposition ③ is false
  ¬(∀ m n α β, intersect α β m → parallel m n → parallelLP n α ∧ parallelLP n β) ∧
  -- Proposition ④ is true
  (∀ m n α β, perpendicular m n → intersect α β m → perpendicularLP n α ∨ perpendicularLP n β) :=
by sorry

end propositions_truth_values_l2062_206203


namespace least_years_to_double_l2062_206231

theorem least_years_to_double (rate : ℝ) (h : rate = 0.5) : 
  (∃ t : ℕ, (1 + rate)^t > 2) ∧ 
  (∀ t : ℕ, (1 + rate)^t > 2 → t ≥ 2) :=
by
  sorry

#check least_years_to_double

end least_years_to_double_l2062_206231


namespace composition_theorem_l2062_206246

def f (x : ℝ) : ℝ := 1 - 2 * x
def g (x : ℝ) : ℝ := x^2 + 3

theorem composition_theorem :
  (∀ x : ℝ, f (g x) = -2 * x^2 - 5) ∧
  (∀ x : ℝ, g (f x) = 4 * x^2 - 4 * x + 4) := by
sorry

end composition_theorem_l2062_206246


namespace remainder_eight_power_2010_mod_100_l2062_206260

theorem remainder_eight_power_2010_mod_100 : 8^2010 % 100 = 24 := by
  sorry

end remainder_eight_power_2010_mod_100_l2062_206260


namespace larger_number_problem_l2062_206279

theorem larger_number_problem (x y : ℝ) (h_product : x * y = 30) (h_sum : x + y = 13) :
  max x y = 10 := by
  sorry

end larger_number_problem_l2062_206279


namespace total_stamps_l2062_206241

theorem total_stamps (a b : ℕ) (h1 : a * 4 = b * 5) (h2 : (a - 5) * 5 = (b + 5) * 4) : a + b = 45 := by
  sorry

end total_stamps_l2062_206241


namespace magnitude_of_complex_fraction_l2062_206216

/-- The magnitude of the vector corresponding to the complex number 2/(1+i) is √2 -/
theorem magnitude_of_complex_fraction : Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end magnitude_of_complex_fraction_l2062_206216


namespace quadratic_abs_inequality_l2062_206247

theorem quadratic_abs_inequality (x : ℝ) : 
  x^2 + 4*x - 96 > |x| ↔ x < -12 ∨ x > 8 := by
sorry

end quadratic_abs_inequality_l2062_206247


namespace correct_equation_by_moving_digit_l2062_206217

theorem correct_equation_by_moving_digit : ∃ (a b c : ℕ), 
  (a = 101 ∧ b = 10 ∧ c = 2) ∧ 
  (a - b^c = 1) ∧
  (∃ (x y : ℕ), x * 10 + y = 102 ∧ (x = 10 ∧ y = c)) :=
by
  sorry

end correct_equation_by_moving_digit_l2062_206217


namespace fraction_sum_simplification_l2062_206263

theorem fraction_sum_simplification : (1 : ℚ) / 210 + 17 / 30 = 4 / 7 := by
  sorry

end fraction_sum_simplification_l2062_206263


namespace product_trailing_zeros_l2062_206234

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros : trailing_zeros (45 * 160 * 7) = 2 := by sorry

end product_trailing_zeros_l2062_206234


namespace arithmetic_sequence_a3_l2062_206240

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h6 : a 6 = 6) (h9 : a 9 = 9) : a 3 = 3 := by
  sorry

end arithmetic_sequence_a3_l2062_206240


namespace infinitely_many_H_points_l2062_206266

/-- The curve C defined by x/4 + y^2 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 / 4 + p.2^2 = 1}

/-- The line l defined by x = 4 -/
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

/-- A point P is an H point if there exists a line through P intersecting C at A 
    and l at B, with either |PA| = |PB| or |PA| = |AB| -/
def is_H_point (P : ℝ × ℝ) : Prop :=
  P ∈ C ∧ ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ l ∧ A ≠ P ∧
    (∃ (k m : ℝ), ∀ x y, y = k * x + m → 
      ((x, y) = P ∨ (x, y) = A ∨ (x, y) = B)) ∧
    (dist P A = dist P B ∨ dist P A = dist A B)

/-- There are infinitely many H points on C, but not all points on C are H points -/
theorem infinitely_many_H_points : 
  (∃ (S : Set (ℝ × ℝ)), S ⊆ C ∧ Infinite S ∧ ∀ p ∈ S, is_H_point p) ∧
  (∃ p ∈ C, ¬is_H_point p) := by sorry


end infinitely_many_H_points_l2062_206266


namespace range_of_a_in_system_with_one_integer_solution_l2062_206265

/-- Given a system of inequalities with exactly one integer solution, prove the range of a -/
theorem range_of_a_in_system_with_one_integer_solution :
  ∀ a : ℝ,
  (∃! x : ℤ, (2 * (x : ℝ) + 3 > 5 ∧ (x : ℝ) - a ≤ 0)) →
  (2 ≤ a ∧ a < 3) :=
by sorry

end range_of_a_in_system_with_one_integer_solution_l2062_206265


namespace stating_chameleon_change_theorem_l2062_206286

/-- Represents the change in the number of chameleons of a specific color. -/
structure ChameleonChange where
  green : ℤ
  yellow : ℤ

/-- Represents the weather conditions for a month. -/
structure WeatherConditions where
  sunny_days : ℕ
  cloudy_days : ℕ

/-- 
Theorem stating that the increase in green chameleons is equal to 
the increase in yellow chameleons plus the difference between sunny and cloudy days.
-/
theorem chameleon_change_theorem (weather : WeatherConditions) (change : ChameleonChange) :
  change.yellow = 5 →
  weather.sunny_days = 18 →
  weather.cloudy_days = 12 →
  change.green = change.yellow + (weather.sunny_days - weather.cloudy_days) :=
by sorry

end stating_chameleon_change_theorem_l2062_206286


namespace lcm_ratio_sum_l2062_206230

theorem lcm_ratio_sum (a b c : ℕ+) : 
  a.val * 3 = b.val * 2 →
  b.val * 7 = c.val * 3 →
  Nat.lcm a.val (Nat.lcm b.val c.val) = 126 →
  a.val + b.val + c.val = 216 := by
sorry

end lcm_ratio_sum_l2062_206230


namespace unique_equilateral_hyperbola_l2062_206226

/-- An equilateral hyperbola passing through (3, -1) with axes of symmetry on coordinate axes -/
def equilateral_hyperbola (a : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 = a ∧ (x = 3 ∧ y = -1)

/-- The unique value of 'a' for which the hyperbola is equilateral and passes through (3, -1) -/
theorem unique_equilateral_hyperbola :
  ∃! a : ℝ, equilateral_hyperbola a ∧ a = 8 := by sorry

end unique_equilateral_hyperbola_l2062_206226


namespace binomial_150_1_l2062_206277

theorem binomial_150_1 : Nat.choose 150 1 = 150 := by
  sorry

end binomial_150_1_l2062_206277


namespace words_with_e_count_l2062_206220

/-- The number of letters in the alphabet we're using -/
def n : ℕ := 5

/-- The length of the words we're creating -/
def k : ℕ := 4

/-- The number of letters in the alphabet excluding E -/
def m : ℕ := 4

/-- The number of 4-letter words that can be made from the letters A, B, C, D, and E, 
    allowing repetition and using the letter E at least once -/
def words_with_e : ℕ := n^k - m^k

theorem words_with_e_count : words_with_e = 369 := by sorry

end words_with_e_count_l2062_206220


namespace restaurant_group_composition_l2062_206261

/-- Proves that in a group of 11 people, where adult meals cost $8 each and kids eat free,
    if the total cost is $72, then the number of kids in the group is 2. -/
theorem restaurant_group_composition (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_people = 11)
  (h2 : adult_meal_cost = 8)
  (h3 : total_cost = 72) :
  total_people - (total_cost / adult_meal_cost) = 2 :=
by sorry

end restaurant_group_composition_l2062_206261


namespace root_reciprocal_sum_l2062_206292

theorem root_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → x₂^2 - 3*x₂ - 1 = 0 → x₁ ≠ x₂ → 
  (1/x₁) + (1/x₂) = -3 := by
  sorry

end root_reciprocal_sum_l2062_206292


namespace real_part_of_fraction_l2062_206287

theorem real_part_of_fraction (z : ℂ) (x : ℝ) (h1 : z.im ≠ 0) (h2 : Complex.abs z = 2) (h3 : z.re = x) :
  (1 / (1 - z)).re = (1 - x) / (5 - 2 * x) := by
  sorry

end real_part_of_fraction_l2062_206287


namespace no_prime_solution_l2062_206269

theorem no_prime_solution : ¬∃ p : ℕ, Nat.Prime p ∧ 2 * p^3 - p^2 - 16 * p + 26 = 0 := by sorry

end no_prime_solution_l2062_206269


namespace sum_of_factors_144_l2062_206208

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_144 : sum_of_factors 144 = 403 := by
  sorry

end sum_of_factors_144_l2062_206208


namespace intersection_height_l2062_206294

/-- The height of the intersection point of lines drawn between two poles -/
theorem intersection_height (h1 h2 d : ℝ) (h1_pos : 0 < h1) (h2_pos : 0 < h2) (d_pos : 0 < d) :
  let x := (h1 * h2 * d) / (h1 * d + h2 * d)
  h1 = 20 → h2 = 80 → d = 100 → x = 16 := by
  sorry


end intersection_height_l2062_206294


namespace log_equation_solution_l2062_206205

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 2 = 9 → x = 2^(27/10) := by
  sorry

end log_equation_solution_l2062_206205


namespace expression_value_l2062_206245

theorem expression_value (a b c : ℝ) : 
  a * (-2)^5 + b * (-2)^3 + c * (-2) - 5 = 7 →
  a * 2^5 + b * 2^3 + c * 2 - 5 = -17 := by
sorry

end expression_value_l2062_206245


namespace compound_interest_proof_l2062_206236

/-- Calculates the final amount after compound interest --/
def final_amount (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that $30,000 increased by 55% annually for 2 years results in $72,075 --/
theorem compound_interest_proof :
  let principal : ℝ := 30000
  let rate : ℝ := 0.55
  let years : ℕ := 2
  final_amount principal rate years = 72075 := by sorry

end compound_interest_proof_l2062_206236


namespace complex_fraction_equals_neg_i_l2062_206259

theorem complex_fraction_equals_neg_i : (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by sorry

end complex_fraction_equals_neg_i_l2062_206259


namespace guide_is_native_l2062_206258

/-- Represents the two tribes on the island -/
inductive Tribe
  | Native
  | Alien

/-- Represents a person on the island -/
structure Person where
  tribe : Tribe

/-- Represents the claim a person makes about their tribe -/
def claim (p : Person) : Tribe :=
  match p.tribe with
  | Tribe.Native => Tribe.Native
  | Tribe.Alien => Tribe.Native

/-- Represents the report a guide makes about another person's claim -/
def report (guide : Person) (other : Person) : Tribe :=
  match guide.tribe with
  | Tribe.Native => claim other
  | Tribe.Alien => claim other

theorem guide_is_native (guide : Person) (other : Person) :
  report guide other = Tribe.Native → guide.tribe = Tribe.Native :=
by
  sorry

#check guide_is_native

end guide_is_native_l2062_206258


namespace text_files_deleted_l2062_206256

theorem text_files_deleted (pictures_deleted : ℕ) (songs_deleted : ℕ) (total_deleted : ℕ) :
  pictures_deleted = 2 →
  songs_deleted = 8 →
  total_deleted = 17 →
  total_deleted = pictures_deleted + songs_deleted + (total_deleted - pictures_deleted - songs_deleted) →
  total_deleted - pictures_deleted - songs_deleted = 7 :=
by sorry

end text_files_deleted_l2062_206256


namespace vector_problem_l2062_206233

def a : ℝ × ℝ := (1, 2)

theorem vector_problem (b : ℝ × ℝ) (θ : ℝ) :
  (b.1 ^ 2 + b.2 ^ 2 = 20) →
  (∃ (k : ℝ), b = k • a) →
  (b = (2, 4) ∨ b = (-2, -4)) ∧
  ((2 * a.1 - 3 * b.1) * (2 * a.1 + b.1) + (2 * a.2 - 3 * b.2) * (2 * a.2 + b.2) = -20) →
  θ = 2 * Real.pi / 3 :=
by sorry

end vector_problem_l2062_206233


namespace third_circle_radius_l2062_206267

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, prove the radius of the third circle -/
theorem third_circle_radius (center_P center_Q center_R : ℝ × ℝ) 
  (radius_P radius_Q radius_R : ℝ) : 
  radius_P = 2 →
  radius_Q = 6 →
  (center_P.1 - center_Q.1)^2 + (center_P.2 - center_Q.2)^2 = (radius_P + radius_Q)^2 →
  (center_P.1 - center_R.1)^2 + (center_P.2 - center_R.2)^2 = (radius_P + radius_R)^2 →
  (center_Q.1 - center_R.1)^2 + (center_Q.2 - center_R.2)^2 = (radius_Q + radius_R)^2 →
  (∃ (t : ℝ), center_R.2 = t * center_P.2 + (1 - t) * center_Q.2 ∧ 
              center_R.1 = t * center_P.1 + (1 - t) * center_Q.1 ∧ 
              0 ≤ t ∧ t ≤ 1) →
  radius_R = 3 := by
  sorry

end third_circle_radius_l2062_206267


namespace f_derivative_at_sqrt2_over_2_l2062_206257

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem f_derivative_at_sqrt2_over_2 :
  (deriv f) (Real.sqrt 2 / 2) = -3/2 := by sorry

end f_derivative_at_sqrt2_over_2_l2062_206257


namespace square_difference_formula_inapplicable_l2062_206280

theorem square_difference_formula_inapplicable :
  ¬∃ (a b : ℝ → ℝ), ∀ x, (x + 1) * (1 + x) = a x ^ 2 - b x ^ 2 :=
sorry

end square_difference_formula_inapplicable_l2062_206280


namespace total_nuts_equals_1_05_l2062_206222

/-- The amount of walnuts Karen added to the trail mix in cups -/
def w : ℝ := 0.25

/-- The amount of almonds Karen added to the trail mix in cups -/
def a : ℝ := 0.25

/-- The amount of peanuts Karen added to the trail mix in cups -/
def p : ℝ := 0.15

/-- The amount of cashews Karen added to the trail mix in cups -/
def c : ℝ := 0.40

/-- The total amount of nuts Karen added to the trail mix -/
def total_nuts : ℝ := w + a + p + c

theorem total_nuts_equals_1_05 : total_nuts = 1.05 := by sorry

end total_nuts_equals_1_05_l2062_206222


namespace womens_average_age_l2062_206201

theorem womens_average_age (n : ℕ) (A : ℝ) (age_increase : ℝ) (man1_age man2_age : ℕ) :
  n = 8 ∧ age_increase = 2 ∧ man1_age = 20 ∧ man2_age = 28 →
  ∃ W1 W2 : ℝ,
    W1 + W2 = n * (A + age_increase) - (n * A - man1_age - man2_age) ∧
    (W1 + W2) / 2 = 32 := by
  sorry

end womens_average_age_l2062_206201


namespace increase_by_percentage_seventy_five_increased_by_ninety_percent_l2062_206227

theorem increase_by_percentage (x : ℝ) (p : ℝ) : 
  x * (1 + p / 100) = x + x * (p / 100) := by sorry

theorem seventy_five_increased_by_ninety_percent : 
  75 * (1 + 90 / 100) = 142.5 := by sorry

end increase_by_percentage_seventy_five_increased_by_ninety_percent_l2062_206227


namespace sqrt_sum_equals_4_sqrt_5_l2062_206274

theorem sqrt_sum_equals_4_sqrt_5 : 
  Real.sqrt (24 - 8 * Real.sqrt 2) + Real.sqrt (24 + 8 * Real.sqrt 2) = 4 * Real.sqrt 5 := by
  sorry

end sqrt_sum_equals_4_sqrt_5_l2062_206274


namespace r_fourth_plus_inverse_fourth_l2062_206218

theorem r_fourth_plus_inverse_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end r_fourth_plus_inverse_fourth_l2062_206218


namespace triangle_side_length_l2062_206211

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  c = 2 → b = 6 → B = 2 * π / 3 → a = 2 := by
  sorry

end triangle_side_length_l2062_206211


namespace parabola_intersection_range_l2062_206225

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the focus
def line (x y : ℝ) : Prop := y = x - 1

-- Define the circle E with AB as diameter
def circle_E (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 16

-- Define the point D
def point_D (t : ℝ) : ℝ × ℝ := (-2, t)

theorem parabola_intersection_range (t : ℝ) :
  (∃ (A B P Q : ℝ × ℝ),
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_E P.1 P.2 ∧ circle_E Q.1 Q.2 ∧
    (∃ (r : ℝ), (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4*r^2 ∧
                ((point_D t).1 - P.1)^2 + ((point_D t).2 - P.2)^2 = r^2)) →
  2 - Real.sqrt 7 ≤ t ∧ t ≤ 2 + Real.sqrt 7 :=
by sorry

end parabola_intersection_range_l2062_206225


namespace larger_number_problem_l2062_206299

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 20775)
  (h2 : L = 23 * S + 143) :
  L = 21713 := by
sorry

end larger_number_problem_l2062_206299


namespace divide_by_three_l2062_206210

theorem divide_by_three (n : ℚ) (h : n / 4 = 12) : n / 3 = 16 := by
  sorry

end divide_by_three_l2062_206210


namespace complement_A_intersect_B_l2062_206228

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 5}
def B : Set Nat := {1, 3, 4}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {3, 4} :=
sorry

end complement_A_intersect_B_l2062_206228


namespace six_lines_regions_l2062_206253

/-- The number of regions created by n lines in a plane where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- The property that no two lines are parallel and no three are concurrent -/
def general_position (n : ℕ) : Prop := sorry

theorem six_lines_regions :
  general_position 6 → num_regions 6 = 22 := by sorry

end six_lines_regions_l2062_206253


namespace words_with_vowels_count_l2062_206293

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

theorem words_with_vowels_count :
  (alphabet.card ^ word_length) - (consonants.card ^ word_length) = 6752 := by
  sorry

end words_with_vowels_count_l2062_206293


namespace symmetric_points_sum_l2062_206215

/-- 
Given two points A and B in the Cartesian coordinate system,
where A has coordinates (2, m) and B has coordinates (n, -1),
if A and B are symmetric with respect to the x-axis,
then m + n = 3.
-/
theorem symmetric_points_sum (m n : ℝ) : 
  (2 : ℝ) = n ∧ m = -(-1 : ℝ) → m + n = 3 := by
  sorry

end symmetric_points_sum_l2062_206215


namespace tank_insulation_cost_l2062_206204

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular tank -/
def surfaceArea (d : TankDimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Calculates the cost of insulation for a given surface area and cost per square foot -/
def insulationCost (area : ℝ) (costPerSqFt : ℝ) : ℝ :=
  area * costPerSqFt

/-- Theorem: The cost to insulate a tank with given dimensions is $1640 -/
theorem tank_insulation_cost :
  let tankDim : TankDimensions := { length := 7, width := 3, height := 2 }
  let costPerSqFt : ℝ := 20
  insulationCost (surfaceArea tankDim) costPerSqFt = 1640 := by
  sorry


end tank_insulation_cost_l2062_206204


namespace perpendicular_and_minimum_points_l2062_206254

-- Define the vectors
def OA : Fin 2 → ℝ := ![1, 7]
def OB : Fin 2 → ℝ := ![5, 1]
def OP : Fin 2 → ℝ := ![2, 1]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := v 0 * w 0 + v 1 * w 1

-- Define the function for OQ based on parameter t
def OQ (t : ℝ) : Fin 2 → ℝ := ![2*t, t]

-- Define QA as a function of t
def QA (t : ℝ) : Fin 2 → ℝ := ![1 - 2*t, 7 - t]

-- Define QB as a function of t
def QB (t : ℝ) : Fin 2 → ℝ := ![5 - 2*t, 1 - t]

theorem perpendicular_and_minimum_points :
  (∃ t : ℝ, dot_product (QA t) OP = 0 ∧ OQ t = ![18/5, 9/5]) ∧
  (∃ t : ℝ, ∀ s : ℝ, dot_product OA (QB t) ≤ dot_product OA (QB s) ∧ OQ t = ![4, 2]) :=
sorry

end perpendicular_and_minimum_points_l2062_206254


namespace discount_percentage_proof_l2062_206262

theorem discount_percentage_proof (jacket_price shirt_price shoes_price : ℝ)
  (jacket_discount shirt_discount shoes_discount : ℝ) :
  jacket_price = 120 ∧ shirt_price = 60 ∧ shoes_price = 90 ∧
  jacket_discount = 0.30 ∧ shirt_discount = 0.50 ∧ shoes_discount = 0.25 →
  (jacket_price * jacket_discount + shirt_price * shirt_discount + shoes_price * shoes_discount) /
  (jacket_price + shirt_price + shoes_price) = 0.328 := by
  sorry

end discount_percentage_proof_l2062_206262


namespace intersection_point_of_f_and_inverse_l2062_206239

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

-- Theorem statement
theorem intersection_point_of_f_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-1, -1) := by
  sorry

end intersection_point_of_f_and_inverse_l2062_206239


namespace smallest_factorial_not_divisible_by_62_l2062_206255

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_factorial_not_divisible_by_62 :
  (∀ n : ℕ, n < 31 → is_factor 62 (factorial n)) ∧
  ¬ is_factor 62 (factorial 31) ∧
  (∀ k : ℕ, k < 62 → (∃ m : ℕ, is_factor k (factorial m)) ∨ is_prime k) ∧
  ¬ is_prime 62 := by
  sorry

end smallest_factorial_not_divisible_by_62_l2062_206255


namespace pony_price_is_18_l2062_206281

/-- The regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The discount rate for Pony jeans as a decimal -/
def pony_discount : ℝ := 0.14

/-- The sum of discount rates for Fox and Pony jeans as a decimal -/
def total_discount : ℝ := 0.22

/-- The total savings from purchasing 5 pairs of jeans -/
def total_savings : ℝ := 8.64

/-- The number of Fox jeans purchased -/
def fox_count : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_count : ℕ := 2

/-- The regular price of Pony jeans -/
def pony_price : ℝ := 18

theorem pony_price_is_18 :
  fox_count * fox_price * (total_discount - pony_discount) +
  pony_count * pony_price * pony_discount = total_savings :=
by sorry

end pony_price_is_18_l2062_206281


namespace communication_system_probabilities_l2062_206221

/-- Represents a communication system with two signals A and B --/
structure CommunicationSystem where
  pTransmitA : ℝ  -- Probability of transmitting signal A
  pTransmitB : ℝ  -- Probability of transmitting signal B
  pDistortAtoB : ℝ  -- Probability of A being distorted to B
  pDistortBtoA : ℝ  -- Probability of B being distorted to A

/-- Theorem about probabilities in the communication system --/
theorem communication_system_probabilities (sys : CommunicationSystem) 
  (h1 : sys.pTransmitA = 0.72)
  (h2 : sys.pTransmitB = 0.28)
  (h3 : sys.pDistortAtoB = 1/6)
  (h4 : sys.pDistortBtoA = 1/7) :
  let pReceiveA := sys.pTransmitA * (1 - sys.pDistortAtoB) + sys.pTransmitB * sys.pDistortBtoA
  let pTransmittedAGivenReceivedA := (sys.pTransmitA * (1 - sys.pDistortAtoB)) / pReceiveA
  pReceiveA = 0.64 ∧ pTransmittedAGivenReceivedA = 0.9375 := by
  sorry


end communication_system_probabilities_l2062_206221


namespace valid_trapezoid_iff_s_gt_8r_l2062_206298

/-- A right-angled tangential trapezoid with an inscribed circle -/
structure RightAngledTangentialTrapezoid where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Perimeter of the trapezoid -/
  s : ℝ
  /-- r is positive -/
  r_pos : r > 0
  /-- s is positive -/
  s_pos : s > 0

/-- Theorem: A valid right-angled tangential trapezoid exists iff s > 8r -/
theorem valid_trapezoid_iff_s_gt_8r (t : RightAngledTangentialTrapezoid) :
  ∃ (trapezoid : RightAngledTangentialTrapezoid), trapezoid.r = t.r ∧ trapezoid.s = t.s ↔ t.s > 8 * t.r :=
by sorry

end valid_trapezoid_iff_s_gt_8r_l2062_206298


namespace marks_speeding_ticket_cost_l2062_206224

/-- Calculates the total amount owed for a speeding ticket -/
def speeding_ticket_cost (base_fine speed_limit actual_speed additional_penalty_per_mph : ℕ)
  (school_zone : Bool) (court_costs lawyer_fee_per_hour lawyer_hours : ℕ) : ℕ :=
  let speed_difference := actual_speed - speed_limit
  let additional_penalty := speed_difference * additional_penalty_per_mph
  let total_fine := base_fine + additional_penalty
  let doubled_fine := if school_zone then 2 * total_fine else total_fine
  let fine_with_court_costs := doubled_fine + court_costs
  let lawyer_fees := lawyer_fee_per_hour * lawyer_hours
  fine_with_court_costs + lawyer_fees

/-- Theorem: Mark's speeding ticket cost is $820 -/
theorem marks_speeding_ticket_cost :
  speeding_ticket_cost 50 30 75 2 true 300 80 3 = 820 := by
  sorry

end marks_speeding_ticket_cost_l2062_206224


namespace domain_of_composed_function_l2062_206223

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem domain_of_composed_function :
  (∀ x ∈ domain_f, f x ≠ 0) →
  {x : ℝ | f (2*x + 1) ≠ 0} = Set.Icc (-3/2) (1/2) := by
  sorry

end domain_of_composed_function_l2062_206223


namespace quadratic_inequality_solution_l2062_206235

theorem quadratic_inequality_solution (x : ℝ) : 
  (3 * x^2 - 2 * x - 8 ≤ 0) ↔ (-4/3 ≤ x ∧ x ≤ 2) := by
  sorry

end quadratic_inequality_solution_l2062_206235


namespace area_under_curve_l2062_206295

open Real MeasureTheory

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2

-- State the theorem
theorem area_under_curve : 
  ∫ x in (1)..(2), f x = 7 := by
  sorry

end area_under_curve_l2062_206295


namespace second_number_in_first_set_l2062_206264

theorem second_number_in_first_set (X : ℝ) : 
  ((20 + X + 60) / 3 = (10 + 50 + 45) / 3 + 5) → X = 40 := by
  sorry

end second_number_in_first_set_l2062_206264


namespace spaghetti_tortellini_ratio_l2062_206200

theorem spaghetti_tortellini_ratio : 
  ∀ (total_students : ℕ) 
    (spaghetti_students tortellini_students : ℕ) 
    (grade_levels : ℕ),
  total_students = 800 →
  spaghetti_students = 300 →
  tortellini_students = 120 →
  grade_levels = 4 →
  (spaghetti_students / grade_levels) / (tortellini_students / grade_levels) = 5 / 2 := by
sorry

end spaghetti_tortellini_ratio_l2062_206200


namespace suzanna_distance_l2062_206268

/-- Represents the distance in miles Suzanna cycles in a given time -/
def distance_cycled (time_minutes : ℕ) : ℚ :=
  (time_minutes / 10 : ℚ) * 2

/-- Proves that Suzanna cycles 8 miles in 40 minutes given her steady speed -/
theorem suzanna_distance : distance_cycled 40 = 8 := by
  sorry

end suzanna_distance_l2062_206268


namespace line_plane_relationships_l2062_206271

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields

/-- Defines when a line is parallel to a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- Defines when two lines are parallel -/
def lines_parallel (l1 l2 : Line3D) : Prop := sorry

/-- Defines when a line intersects a plane -/
def line_intersects_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- Theorem representing the four statements -/
theorem line_plane_relationships :
  (∀ (a b : Line3D) (α : Plane),
    line_parallel_to_plane a α → line_in_plane b α → lines_parallel a b) = False
  ∧
  (∀ (a b : Line3D) (α : Plane) (P : Point),
    line_intersects_plane a α → line_in_plane b α → ¬lines_parallel a b) = True
  ∧
  (∀ (a : Line3D) (α : Plane),
    ¬line_in_plane a α → line_parallel_to_plane a α) = False
  ∧
  (∀ (a b : Line3D) (α : Plane),
    line_parallel_to_plane a α → line_parallel_to_plane b α → lines_parallel a b) = False :=
by sorry

end line_plane_relationships_l2062_206271
