import Mathlib

namespace NUMINAMATH_CALUDE_fraction_simplification_l2534_253494

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) :
  (a - b) / (2*a*b - b^2 - a^2) = 1 / (b - a) := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2534_253494


namespace NUMINAMATH_CALUDE_sum_of_eleven_terms_l2534_253459

def a (n : ℕ) : ℤ := 1 - 2 * n

def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

def sequence_sum (n : ℕ) : ℚ := 
  Finset.sum (Finset.range n) (λ i => S (i + 1) / (i + 1))

theorem sum_of_eleven_terms : sequence_sum 11 = -66 := by sorry

end NUMINAMATH_CALUDE_sum_of_eleven_terms_l2534_253459


namespace NUMINAMATH_CALUDE_bow_collection_problem_l2534_253484

theorem bow_collection_problem (total : ℕ) (yellow : ℕ) :
  yellow = 36 →
  (1 : ℚ) / 4 * total + (1 : ℚ) / 3 * total + (1 : ℚ) / 6 * total + yellow = total →
  (1 : ℚ) / 6 * total = 24 := by
  sorry

end NUMINAMATH_CALUDE_bow_collection_problem_l2534_253484


namespace NUMINAMATH_CALUDE_sqrt_24_times_sqrt_3_over_2_equals_6_l2534_253409

theorem sqrt_24_times_sqrt_3_over_2_equals_6 :
  Real.sqrt 24 * Real.sqrt (3/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_24_times_sqrt_3_over_2_equals_6_l2534_253409


namespace NUMINAMATH_CALUDE_dish_initial_temp_l2534_253486

/-- The initial temperature of a dish given its heating rate and time to reach a final temperature --/
def initial_temperature (final_temp : ℝ) (heating_rate : ℝ) (heating_time : ℝ) : ℝ :=
  final_temp - heating_rate * heating_time

/-- Theorem stating that the initial temperature of the dish is 20 degrees --/
theorem dish_initial_temp : initial_temperature 100 5 16 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dish_initial_temp_l2534_253486


namespace NUMINAMATH_CALUDE_agathas_bike_frame_cost_l2534_253491

/-- Agatha's bike purchase problem -/
theorem agathas_bike_frame_cost (total : ℕ) (wheel_cost : ℕ) (remaining : ℕ) (frame_cost : ℕ) :
  total = 60 →
  wheel_cost = 25 →
  remaining = 20 →
  frame_cost = total - wheel_cost - remaining →
  frame_cost = 15 := by
sorry

end NUMINAMATH_CALUDE_agathas_bike_frame_cost_l2534_253491


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_four_l2534_253424

theorem sum_of_solutions_eq_four :
  let f : ℝ → ℝ := λ N => N * (N - 4)
  let solutions := {N : ℝ | f N = -21}
  (∃ N₁ N₂, N₁ ∈ solutions ∧ N₂ ∈ solutions ∧ N₁ ≠ N₂) →
  (∀ N, N ∈ solutions → N₁ = N ∨ N₂ = N) →
  N₁ + N₂ = 4
  := by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_four_l2534_253424


namespace NUMINAMATH_CALUDE_common_area_of_30_60_90_triangles_l2534_253455

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shortLeg : ℝ
  longLeg : ℝ

/-- The area common to two congruent 30-60-90 triangles with coinciding shorter legs -/
def commonArea (t : Triangle30_60_90) : ℝ := t.shortLeg ^ 2

/-- Theorem: The area common to two congruent 30-60-90 triangles with hypotenuse 16 and coinciding shorter legs is 64 square units -/
theorem common_area_of_30_60_90_triangles :
  ∀ t : Triangle30_60_90,
  t.hypotenuse = 16 →
  t.shortLeg = t.hypotenuse / 2 →
  t.longLeg = t.shortLeg * Real.sqrt 3 →
  commonArea t = 64 := by
  sorry

end NUMINAMATH_CALUDE_common_area_of_30_60_90_triangles_l2534_253455


namespace NUMINAMATH_CALUDE_melon_count_l2534_253456

/-- Given the number of watermelons and apples, calculate the number of melons -/
theorem melon_count (watermelons apples : ℕ) (h1 : watermelons = 3) (h2 : apples = 7) :
  2 * (watermelons + apples) = 20 := by
  sorry

end NUMINAMATH_CALUDE_melon_count_l2534_253456


namespace NUMINAMATH_CALUDE_simplest_form_iff_odd_l2534_253410

theorem simplest_form_iff_odd (n : ℤ) : 
  (∀ d : ℤ, d ∣ (3*n + 10) ∧ d ∣ (5*n + 16) → d = 1 ∨ d = -1) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_simplest_form_iff_odd_l2534_253410


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2534_253427

/-- A point M with coordinates (m+3, m+1) lies on the y-axis if and only if its coordinates are (0, -2) -/
theorem point_on_y_axis (m : ℝ) : 
  (m + 3 = 0 ∧ m + 1 = -2) ↔ (m + 3 = 0 ∧ m + 1 = -2) :=
by sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l2534_253427


namespace NUMINAMATH_CALUDE_equation_system_solutions_l2534_253457

theorem equation_system_solutions (x y : ℝ) : 
  ((1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7 ∧
   (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7) →
  ((x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solutions_l2534_253457


namespace NUMINAMATH_CALUDE_sin_neg_135_degrees_l2534_253498

theorem sin_neg_135_degrees : Real.sin (-(135 * π / 180)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_135_degrees_l2534_253498


namespace NUMINAMATH_CALUDE_real_part_of_z_is_zero_l2534_253497

theorem real_part_of_z_is_zero :
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / (-2*i + 1)
  Complex.re z = 0 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_z_is_zero_l2534_253497


namespace NUMINAMATH_CALUDE_vp_factorial_and_binomial_l2534_253404

/-- The p-adic valuation of a natural number -/
noncomputable def v_p (p : ℕ) (n : ℕ) : ℕ := sorry

/-- The sum of floor of N divided by increasing powers of p -/
def sum_floor (N : ℕ) (p : ℕ) : ℕ := sorry

theorem vp_factorial_and_binomial 
  (N k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_pow : ∃ n, N = p ^ n) (h_ge : N ≥ k) :
  (v_p p (N.factorial) = sum_floor N p) ∧ 
  (v_p p (Nat.choose N k) = v_p p N - v_p p k) := by
  sorry

end NUMINAMATH_CALUDE_vp_factorial_and_binomial_l2534_253404


namespace NUMINAMATH_CALUDE_nigella_commission_rate_l2534_253412

/-- Nigella's commission rate problem -/
theorem nigella_commission_rate :
  let base_salary : ℚ := 3000
  let total_earnings : ℚ := 8000
  let house_a_cost : ℚ := 60000
  let house_b_cost : ℚ := 3 * house_a_cost
  let house_c_cost : ℚ := 2 * house_a_cost - 110000
  let total_houses_cost : ℚ := house_a_cost + house_b_cost + house_c_cost
  let commission : ℚ := total_earnings - base_salary
  let commission_rate : ℚ := commission / total_houses_cost
  commission_rate = 1/50 := by sorry

end NUMINAMATH_CALUDE_nigella_commission_rate_l2534_253412


namespace NUMINAMATH_CALUDE_wheel_probability_l2534_253495

theorem wheel_probability (pA pB pC pD pE : ℚ) : 
  pA = 1/5 →
  pB = 1/3 →
  pD = pE →
  pA + pB + pC + pD + pE = 1 →
  pC = 0 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l2534_253495


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l2534_253475

theorem root_difference_implies_k_value :
  ∀ (k : ℝ) (r s : ℝ),
  (r^2 + k*r + 12 = 0) ∧ 
  (s^2 + k*s + 12 = 0) ∧
  ((r-3)^2 - k*(r-3) + 12 = 0) ∧ 
  ((s-3)^2 - k*(s-3) + 12 = 0) →
  k = -3 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l2534_253475


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_l2534_253462

theorem fourth_term_coefficient : 
  let a := (1/2 : ℚ)
  let b := (2/3 : ℚ)
  let n := 6
  let k := 4
  (n.choose (k-1)) * a^(n-(k-1)) * b^(k-1) = 20 := by sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_l2534_253462


namespace NUMINAMATH_CALUDE_barbell_cost_is_270_l2534_253429

/-- The cost of each barbell given the total amount paid, change received, and number of barbells purchased. -/
def barbell_cost (total_paid : ℕ) (change : ℕ) (num_barbells : ℕ) : ℕ :=
  (total_paid - change) / num_barbells

/-- Theorem stating that the cost of each barbell is $270 under the given conditions. -/
theorem barbell_cost_is_270 :
  barbell_cost 850 40 3 = 270 := by
  sorry

end NUMINAMATH_CALUDE_barbell_cost_is_270_l2534_253429


namespace NUMINAMATH_CALUDE_initial_investment_rate_l2534_253480

/-- Proves that the initial investment rate is 5% given the problem conditions --/
theorem initial_investment_rate
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 8000)
  (h2 : additional_investment = 4000)
  (h3 : additional_rate = 8)
  (h4 : total_rate = 6)
  (h5 : initial_investment + additional_investment = 12000) :
  ∃ R : ℝ, R = 5 ∧
    (initial_investment * R / 100 + additional_investment * additional_rate / 100 =
     (initial_investment + additional_investment) * total_rate / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_initial_investment_rate_l2534_253480


namespace NUMINAMATH_CALUDE_lily_bouquet_cost_l2534_253487

/-- The cost of a bouquet is directly proportional to the number of lilies it contains. -/
def DirectlyProportional (cost : ℝ → ℝ) (lilies : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, cost x = k * lilies x

theorem lily_bouquet_cost 
  (cost : ℝ → ℝ) 
  (lilies : ℝ → ℝ) 
  (h_prop : DirectlyProportional cost lilies)
  (h_18 : cost 18 = 30)
  (h_pos : ∀ x, lilies x > 0) :
  cost 27 = 45 := by
sorry

end NUMINAMATH_CALUDE_lily_bouquet_cost_l2534_253487


namespace NUMINAMATH_CALUDE_orange_distribution_l2534_253413

theorem orange_distribution (num_oranges : ℕ) (pieces_per_orange : ℕ) (pieces_per_friend : ℕ) : 
  num_oranges = 80 → 
  pieces_per_orange = 10 → 
  pieces_per_friend = 4 → 
  (num_oranges * pieces_per_orange) / pieces_per_friend = 200 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l2534_253413


namespace NUMINAMATH_CALUDE_calculator_squared_key_l2534_253452

theorem calculator_squared_key (n : ℕ) : (5 ^ (2 ^ n) > 10000) ↔ n ≥ 3 :=
  sorry

end NUMINAMATH_CALUDE_calculator_squared_key_l2534_253452


namespace NUMINAMATH_CALUDE_sum_of_digits_n_n_is_greatest_divisor_l2534_253437

/-- The greatest number that divides 1305, 4665, and 6905 leaving the same remainder -/
def n : ℕ := 1120

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  if m < 10 then m else (m % 10) + sum_of_digits (m / 10)

/-- Theorem stating that the sum of digits of n is 4 -/
theorem sum_of_digits_n : sum_of_digits n = 4 := by
  sorry

/-- Theorem stating that n is the greatest number that divides 1305, 4665, and 6905 
    leaving the same remainder -/
theorem n_is_greatest_divisor : 
  ∀ m : ℕ, m > n → ¬(∃ r : ℕ, 1305 % m = r ∧ 4665 % m = r ∧ 6905 % m = r) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_n_n_is_greatest_divisor_l2534_253437


namespace NUMINAMATH_CALUDE_quartic_polynomial_unique_l2534_253438

/-- A monic quartic polynomial with real coefficients -/
def QuarticPolynomial (a b c d : ℝ) : ℝ → ℂ :=
  fun x ↦ (x^4 : ℂ) + a * (x^3 : ℂ) + b * (x^2 : ℂ) + c * (x : ℂ) + d

theorem quartic_polynomial_unique
  (q : ℝ → ℂ)
  (h_monic : ∀ x, q x = (x^4 : ℂ) + (a * x^3 : ℂ) + (b * x^2 : ℂ) + (c * x : ℂ) + d)
  (h_root : q (5 - 3*I) = 0)
  (h_constant : q 0 = -150) :
  q = QuarticPolynomial (-658/34) (19206/34) (-3822/17) (-150) :=
by sorry

end NUMINAMATH_CALUDE_quartic_polynomial_unique_l2534_253438


namespace NUMINAMATH_CALUDE_special_right_triangle_angles_l2534_253436

/-- A right triangle with the property that when rotated four times,
    each time aligning the shorter leg with the hypotenuse and
    matching the vertex of the acute angle with the vertex of the right angle,
    results in an isosceles fifth triangle. -/
structure SpecialRightTriangle where
  /-- The measure of one of the acute angles in the triangle -/
  α : Real
  /-- The triangle is a right triangle -/
  is_right_triangle : α + (90 - α) + 90 = 180
  /-- The fifth triangle is isosceles -/
  fifth_triangle_isosceles : 4 * α = 180 - 4 * (90 + α)

/-- Theorem stating that the acute angles in the special right triangle are both 90°/11 -/
theorem special_right_triangle_angles (t : SpecialRightTriangle) : t.α = 90 / 11 := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_angles_l2534_253436


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2534_253476

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2534_253476


namespace NUMINAMATH_CALUDE_packet_B_height_l2534_253463

/-- Growth rate of Packet A sunflowers -/
def R_A (x y : ℝ) : ℝ := 2 * x + y

/-- Growth rate of Packet B sunflowers -/
def R_B (x y : ℝ) : ℝ := 3 * x - y

/-- Theorem stating the height of Packet B sunflowers on day 10 -/
theorem packet_B_height (h_A : ℝ) (h_B : ℝ) :
  R_A 10 6 = 26 →
  R_B 10 6 = 24 →
  h_A = 192 →
  h_A = h_B + 0.2 * h_B →
  h_B = 160 := by
  sorry

#check packet_B_height

end NUMINAMATH_CALUDE_packet_B_height_l2534_253463


namespace NUMINAMATH_CALUDE_right_triangle_area_l2534_253464

/-- The area of a right triangle with hypotenuse 14 inches and one 45-degree angle is 49 square inches. -/
theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 14 →
  angle = 45 * (π / 180) →
  let leg := hypotenuse / Real.sqrt 2
  let area := (1 / 2) * leg * leg
  area = 49 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2534_253464


namespace NUMINAMATH_CALUDE_quadratic_sum_l2534_253419

-- Define the quadratic function
def f (x : ℝ) : ℝ := -8 * x^2 + 16 * x + 320

-- Define the completed square form
def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum : ∃ a b c : ℝ, 
  (∀ x, f x = g a b c x) ∧ 
  (a + b + c = 319) := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2534_253419


namespace NUMINAMATH_CALUDE_josh_marbles_l2534_253467

/-- The number of marbles Josh has after receiving marbles from Jack -/
def total_marbles (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Josh has 42 marbles after receiving marbles from Jack -/
theorem josh_marbles :
  let initial_marbles : ℕ := 22
  let marbles_from_jack : ℕ := 20
  total_marbles initial_marbles marbles_from_jack = 42 := by
sorry

end NUMINAMATH_CALUDE_josh_marbles_l2534_253467


namespace NUMINAMATH_CALUDE_min_nuts_is_480_l2534_253411

/-- Represents the nut-gathering process of three squirrels -/
structure NutGathering where
  n1 : ℕ  -- nuts picked by first squirrel
  n2 : ℕ  -- nuts picked by second squirrel
  n3 : ℕ  -- nuts picked by third squirrel

/-- Checks if the nut distribution satisfies the given conditions -/
def is_valid_distribution (ng : NutGathering) : Prop :=
  let total := ng.n1 + ng.n2 + ng.n3
  let s1_final := (5 * ng.n1) / 6 + ng.n2 / 12 + (3 * ng.n3) / 16
  let s2_final := ng.n1 / 12 + (3 * ng.n2) / 4 + (3 * ng.n3) / 16
  let s3_final := ng.n1 / 12 + ng.n2 / 4 + (5 * ng.n3) / 8
  (s1_final : ℚ) / 5 = (s2_final : ℚ) / 3 ∧
  (s2_final : ℚ) / 3 = (s3_final : ℚ) / 2 ∧
  s1_final * 3 = s2_final * 5 ∧
  s2_final * 2 = s3_final * 3 ∧
  (5 * ng.n1) % 6 = 0 ∧
  ng.n2 % 12 = 0 ∧
  (3 * ng.n3) % 16 = 0 ∧
  ng.n1 % 12 = 0 ∧
  (3 * ng.n2) % 4 = 0 ∧
  ng.n2 % 4 = 0 ∧
  (5 * ng.n3) % 8 = 0

/-- The least possible total number of nuts -/
def min_total_nuts : ℕ := 480

/-- Theorem stating that the minimum total number of nuts is 480 -/
theorem min_nuts_is_480 :
  ∀ ng : NutGathering, is_valid_distribution ng →
    ng.n1 + ng.n2 + ng.n3 ≥ min_total_nuts :=
by sorry

end NUMINAMATH_CALUDE_min_nuts_is_480_l2534_253411


namespace NUMINAMATH_CALUDE_basketball_win_rate_l2534_253402

theorem basketball_win_rate (games_won : ℕ) (first_games : ℕ) (remaining_games : ℕ) (target_win_rate : ℚ) : 
  games_won = 45 →
  first_games = 60 →
  remaining_games = 54 →
  target_win_rate = 3/4 →
  ∃ (additional_wins : ℕ), 
    (games_won + additional_wins : ℚ) / (first_games + remaining_games : ℚ) = target_win_rate ∧
    additional_wins = 41 :=
by sorry

end NUMINAMATH_CALUDE_basketball_win_rate_l2534_253402


namespace NUMINAMATH_CALUDE_spade_problem_l2534_253446

/-- Custom operation ⊙ for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

/-- Theorem stating that 2 ⊙ (3 ⊙ 4) = 384 -/
theorem spade_problem : spade 2 (spade 3 4) = 384 := by
  sorry

end NUMINAMATH_CALUDE_spade_problem_l2534_253446


namespace NUMINAMATH_CALUDE_gas_used_for_appointments_l2534_253449

def distance_to_dermatologist : ℝ := 30
def distance_to_gynecologist : ℝ := 50
def car_efficiency : ℝ := 20

theorem gas_used_for_appointments : 
  (2 * distance_to_dermatologist + 2 * distance_to_gynecologist) / car_efficiency = 8 := by
  sorry

end NUMINAMATH_CALUDE_gas_used_for_appointments_l2534_253449


namespace NUMINAMATH_CALUDE_tree_planting_theorem_l2534_253407

/-- The number of trees planted by Class 2-5 -/
def trees_2_5 : ℕ := 142

/-- The difference in trees planted between Class 2-5 and Class 2-3 -/
def difference : ℕ := 18

/-- The number of trees planted by Class 2-3 -/
def trees_2_3 : ℕ := trees_2_5 - difference

/-- The total number of trees planted by both classes -/
def total_trees : ℕ := trees_2_5 + trees_2_3

theorem tree_planting_theorem :
  trees_2_3 = 124 ∧ total_trees = 266 :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_theorem_l2534_253407


namespace NUMINAMATH_CALUDE_range_of_m_range_of_t_l2534_253492

noncomputable section

-- Define the functions f and g
def f (x t : ℝ) : ℝ := -x^2 + 2 * Real.exp 1 * x + t - 1
def g (x : ℝ) : ℝ := x + (Real.exp 1)^2 / x

-- State the theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x : ℝ, x > 0 ∧ g x = m) ↔ m ≥ 2 * Real.exp 1 :=
sorry

-- State the theorem for the range of t
theorem range_of_t :
  ∀ t : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ g x₁ - f x₁ t = 0 ∧ g x₂ - f x₂ t = 0)
  ↔ t > 2 * Real.exp 1 - (Real.exp 1)^2 + 1 :=
sorry

end

end NUMINAMATH_CALUDE_range_of_m_range_of_t_l2534_253492


namespace NUMINAMATH_CALUDE_range_of_a_l2534_253454

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ a ∧ x < 2 ↔ x < 2) ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2534_253454


namespace NUMINAMATH_CALUDE_tan_equation_solution_exists_l2534_253474

open Real

theorem tan_equation_solution_exists :
  ∃! θ : ℝ, 0 < θ ∧ θ < π/6 ∧
  tan θ + tan (θ + π/6) + tan (3*θ) = 0 ∧
  0 < tan θ ∧ tan θ < 1 := by
sorry

end NUMINAMATH_CALUDE_tan_equation_solution_exists_l2534_253474


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2534_253460

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2534_253460


namespace NUMINAMATH_CALUDE_divisible_by_thirty_l2534_253432

theorem divisible_by_thirty (n : ℕ) (h : n > 0) : ∃ k : ℤ, n^19 - n^7 = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_thirty_l2534_253432


namespace NUMINAMATH_CALUDE_negation_of_square_non_negative_l2534_253441

theorem negation_of_square_non_negative :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_square_non_negative_l2534_253441


namespace NUMINAMATH_CALUDE_count_prime_base_n_l2534_253408

/-- Represents the number 10001 in base n -/
def base_n (n : ℕ) : ℕ := n^4 + 1

/-- Counts the number of positive integers n ≥ 2 for which 10001_n is prime -/
theorem count_prime_base_n : ∃! (n : ℕ), n ≥ 2 ∧ Nat.Prime (base_n n) := by
  sorry

end NUMINAMATH_CALUDE_count_prime_base_n_l2534_253408


namespace NUMINAMATH_CALUDE_negation_of_forall_inequality_negation_of_inequality_negation_of_proposition_l2534_253447

theorem negation_of_forall_inequality (P : ℝ → Prop) :
  (¬ ∀ x < 0, P x) ↔ (∃ x < 0, ¬ P x) := by sorry

theorem negation_of_inequality (x : ℝ) :
  ¬(1 - x > Real.exp x) ↔ (1 - x ≤ Real.exp x) := by sorry

theorem negation_of_proposition :
  (¬ ∀ x < 0, 1 - x > Real.exp x) ↔ (∃ x < 0, 1 - x ≤ Real.exp x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_inequality_negation_of_inequality_negation_of_proposition_l2534_253447


namespace NUMINAMATH_CALUDE_b_work_time_l2534_253420

/-- The time it takes for worker a to complete the work alone -/
def a_time : ℝ := 14

/-- The time it takes for workers a and b to complete the work together -/
def ab_time : ℝ := 5.833333333333333

/-- The time it takes for worker b to complete the work alone -/
def b_time : ℝ := 10

/-- The total amount of work to be completed -/
def total_work : ℝ := 1

theorem b_work_time : 
  (1 / a_time + 1 / b_time = 1 / ab_time) ∧
  (1 / b_time = 1 / total_work) := by
  sorry

end NUMINAMATH_CALUDE_b_work_time_l2534_253420


namespace NUMINAMATH_CALUDE_number_is_composite_l2534_253440

theorem number_is_composite : ∃ (k : ℕ), k > 1 ∧ k ∣ (53 * 83 * 109 + 40 * 66 * 96) := by
  -- We claim that 149 divides the given number
  use 149
  constructor
  · -- 149 > 1
    norm_num
  · -- 149 divides the given number
    sorry


end NUMINAMATH_CALUDE_number_is_composite_l2534_253440


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l2534_253478

/-- Given a farm with sheep and horses, calculate the amount of food per horse -/
theorem stewart_farm_horse_food 
  (sheep_count : ℕ) 
  (sheep_horse_ratio : ℚ) 
  (total_horse_food : ℕ) 
  (h1 : sheep_count = 48)
  (h2 : sheep_horse_ratio = 6 / 7)
  (h3 : total_horse_food = 12880) : 
  (total_horse_food : ℚ) / ((sheep_count : ℚ) / sheep_horse_ratio) = 230 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l2534_253478


namespace NUMINAMATH_CALUDE_sugar_required_for_cake_l2534_253445

/-- Given a recipe for a cake, prove the amount of sugar required -/
theorem sugar_required_for_cake (total_flour : ℕ) (flour_added : ℕ) (extra_sugar : ℕ) : 
  total_flour = 9 → 
  flour_added = 4 → 
  extra_sugar = 6 → 
  (total_flour - flour_added) + extra_sugar = 11 := by
  sorry

end NUMINAMATH_CALUDE_sugar_required_for_cake_l2534_253445


namespace NUMINAMATH_CALUDE_jane_sarah_age_sum_l2534_253418

theorem jane_sarah_age_sum : 
  ∀ (jane sarah : ℝ),
  jane = sarah + 5 →
  jane + 9 = 3 * (sarah - 3) →
  jane + sarah = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_jane_sarah_age_sum_l2534_253418


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l2534_253425

/-- A regular polygon with exterior angle 90 degrees and side length 7 units has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  n * exterior_angle = 360 →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l2534_253425


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_specific_circle_equation_l2534_253485

/-- Given two points A and B as the endpoints of a circle's diameter, 
    prove that the equation of the circle is (x-h)^2 + (y-k)^2 = r^2,
    where (h,k) is the midpoint of AB and r is half the distance between A and B. -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let h := (x₁ + x₂) / 2
  let k := (y₁ + y₂) / 2
  let r := Real.sqrt (((x₁ - x₂)^2 + (y₁ - y₂)^2) / 4)
  ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ 
    ((x - x₁)^2 + (y - y₁)^2) * ((x - x₂)^2 + (y - y₂)^2) = 
    ((x - x₁)^2 + (y - y₁)^2 + (x - x₂)^2 + (y - y₂)^2)^2 / 4 :=
by sorry

/-- The equation of the circle with diameter endpoints A(4,9) and B(6,3) is (x-5)^2 + (y-6)^2 = 10 -/
theorem specific_circle_equation : 
  ∀ (x y : ℝ), (x - 5)^2 + (y - 6)^2 = 10 ↔ 
    ((x - 4)^2 + (y - 9)^2) * ((x - 6)^2 + (y - 3)^2) = 
    ((x - 4)^2 + (y - 9)^2 + (x - 6)^2 + (y - 3)^2)^2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_specific_circle_equation_l2534_253485


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l2534_253415

theorem unique_solution_quadratic_equation :
  ∃! x : ℝ, (3012 + x)^2 = x^2 ∧ x = -1506 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l2534_253415


namespace NUMINAMATH_CALUDE_line_above_function_l2534_253461

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1/a) - a*x

theorem line_above_function (a : ℝ) (h : a ≠ 0) :
  (∀ x, a*x > f a x) ↔ a > Real.exp 1 / 2 := by sorry

end NUMINAMATH_CALUDE_line_above_function_l2534_253461


namespace NUMINAMATH_CALUDE_complex_difference_modulus_l2534_253431

theorem complex_difference_modulus (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1)
  (h2 : Complex.abs z₂ = 1)
  (h3 : Complex.abs (z₁ + z₂) = Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_modulus_l2534_253431


namespace NUMINAMATH_CALUDE_impossible_three_similar_parts_l2534_253435

theorem impossible_three_similar_parts : 
  ∀ (x : ℝ), x > 0 → ¬∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = x ∧
    a ≤ b ∧ b ≤ c ∧
    c ≤ Real.sqrt 2 * b ∧
    b ≤ Real.sqrt 2 * a :=
by sorry

end NUMINAMATH_CALUDE_impossible_three_similar_parts_l2534_253435


namespace NUMINAMATH_CALUDE_max_substances_l2534_253496

/-- The number of substances generated when ethane is mixed with chlorine gas under lighting conditions -/
def num_substances : ℕ := sorry

/-- The number of isomers for monochloroethane -/
def mono_isomers : ℕ := 1

/-- The number of isomers for dichloroethane (including geometric isomers) -/
def di_isomers : ℕ := 3

/-- The number of isomers for trichloroethane -/
def tri_isomers : ℕ := 2

/-- The number of isomers for tetrachloroethane -/
def tetra_isomers : ℕ := 2

/-- The number of isomers for pentachloroethane -/
def penta_isomers : ℕ := 1

/-- The number of isomers for hexachloroethane -/
def hexa_isomers : ℕ := 1

/-- Hydrogen chloride is also formed -/
def hcl_formed : Prop := true

theorem max_substances :
  num_substances = mono_isomers + di_isomers + tri_isomers + tetra_isomers + penta_isomers + hexa_isomers + 1 ∧
  num_substances = 10 := by sorry

end NUMINAMATH_CALUDE_max_substances_l2534_253496


namespace NUMINAMATH_CALUDE_bus_trip_distance_l2534_253403

theorem bus_trip_distance (v : ℝ) (d : ℝ) : 
  v = 40 → 
  d / v - d / (v + 5) = 1 → 
  d = 360 := by sorry

end NUMINAMATH_CALUDE_bus_trip_distance_l2534_253403


namespace NUMINAMATH_CALUDE_g_of_3_eq_15_l2534_253453

/-- A function g satisfying the given conditions -/
def g (x : ℝ) : ℝ := sorry

/-- The theorem stating that g(3) = 15 -/
theorem g_of_3_eq_15 (h1 : g 1 = 7) (h2 : g 2 = 11) 
  (h3 : ∃ (c d : ℝ), ∀ x, g x = c * x + d * x + 3) : 
  g 3 = 15 := by sorry

end NUMINAMATH_CALUDE_g_of_3_eq_15_l2534_253453


namespace NUMINAMATH_CALUDE_logarithm_identity_l2534_253448

theorem logarithm_identity : Real.log 5 ^ 2 + Real.log 2 * Real.log 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_identity_l2534_253448


namespace NUMINAMATH_CALUDE_four_at_six_equals_twenty_l2534_253451

-- Define the @ operation
def at_operation (a b : ℤ) : ℤ := 4*a - 2*b + a^2

-- Theorem statement
theorem four_at_six_equals_twenty : at_operation 4 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_four_at_six_equals_twenty_l2534_253451


namespace NUMINAMATH_CALUDE_special_polyhedron_hexagon_count_l2534_253472

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  -- V: vertices, E: edges, F: faces, P: pentagonal faces, H: hexagonal faces
  V : ℕ
  E : ℕ
  F : ℕ
  P : ℕ
  H : ℕ
  vertex_degree : V * 3 = E * 2
  face_types : F = P + H
  euler : V - E + F = 2
  edge_count : E * 2 = P * 5 + H * 6
  both_face_types : P > 0 ∧ H > 0

/-- Theorem: In a SpecialPolyhedron, the number of hexagonal faces is at least 2 -/
theorem special_polyhedron_hexagon_count (poly : SpecialPolyhedron) : poly.H ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_hexagon_count_l2534_253472


namespace NUMINAMATH_CALUDE_distance_a_travels_is_60km_l2534_253426

/-- Represents the movement of two objects towards each other with doubling speed -/
structure DoubleSpeedMeeting where
  initial_distance : ℝ
  initial_speed_a : ℝ
  initial_speed_b : ℝ

/-- Calculates the distance traveled by object a until meeting object b -/
def distance_traveled_by_a (meeting : DoubleSpeedMeeting) : ℝ :=
  sorry

/-- Theorem stating that given the specific initial conditions, a travels 60 km until meeting b -/
theorem distance_a_travels_is_60km :
  let meeting := DoubleSpeedMeeting.mk 90 10 5
  distance_traveled_by_a meeting = 60 := by
  sorry

end NUMINAMATH_CALUDE_distance_a_travels_is_60km_l2534_253426


namespace NUMINAMATH_CALUDE_binomial_600_0_l2534_253434

theorem binomial_600_0 : (600 : ℕ).choose 0 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_600_0_l2534_253434


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2534_253400

theorem expand_and_simplify (x y : ℝ) : 12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2534_253400


namespace NUMINAMATH_CALUDE_exist_k_m_with_prime_divisor_diff_l2534_253477

/-- The number of prime divisors of a positive integer -/
def num_prime_divisors (n : ℕ+) : ℕ := sorry

/-- For any positive integer n, there exist positive integers k and m such that 
    k - m = n and the number of prime divisors of k is exactly one more than 
    the number of prime divisors of m -/
theorem exist_k_m_with_prime_divisor_diff (n : ℕ+) : 
  ∃ (k m : ℕ+), k - m = n ∧ num_prime_divisors k = num_prime_divisors m + 1 := by sorry

end NUMINAMATH_CALUDE_exist_k_m_with_prime_divisor_diff_l2534_253477


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2534_253450

theorem trig_identity_proof : 
  4 * Real.cos (10 * π / 180) - Real.tan (80 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2534_253450


namespace NUMINAMATH_CALUDE_rectangle_perimeter_width_ratio_l2534_253442

/-- Given a rectangle with area 150 square centimeters and length 15 centimeters,
    prove that the ratio of its perimeter to its width is 5:1 -/
theorem rectangle_perimeter_width_ratio 
  (area : ℝ) (length : ℝ) (width : ℝ) (perimeter : ℝ) :
  area = 150 →
  length = 15 →
  area = length * width →
  perimeter = 2 * (length + width) →
  perimeter / width = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_width_ratio_l2534_253442


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l2534_253405

theorem unique_quadratic_root (k : ℝ) : 
  (∃! x : ℝ, x^2 - 4*x + k = 0) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l2534_253405


namespace NUMINAMATH_CALUDE_new_lift_count_correct_l2534_253489

/-- The number of times Terrell must lift the new weight configuration to match the total weight of the original configuration -/
def new_lift_count : ℕ := 12

/-- The weight of each item in the original configuration -/
def original_weight : ℕ := 12

/-- The number of weights in the original configuration -/
def original_count : ℕ := 3

/-- The number of times Terrell lifts the original configuration -/
def original_lifts : ℕ := 20

/-- The weights in the new configuration -/
def new_weights : List ℕ := [18, 18, 24]

theorem new_lift_count_correct :
  new_lift_count * (new_weights.sum) = original_weight * original_count * original_lifts :=
by sorry

end NUMINAMATH_CALUDE_new_lift_count_correct_l2534_253489


namespace NUMINAMATH_CALUDE_unique_prime_pair_l2534_253414

theorem unique_prime_pair : ∃! p : ℕ, Prime p ∧ Prime (p + 15) := by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l2534_253414


namespace NUMINAMATH_CALUDE_x_value_proof_l2534_253473

theorem x_value_proof (x : ℝ) : (-1 : ℝ) * 2 * x * 4 = 24 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2534_253473


namespace NUMINAMATH_CALUDE_fifteenSidedFigureArea_l2534_253423

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices -/
structure Polygon where
  vertices : List Point

/-- The area of a polygon -/
noncomputable def area (p : Polygon) : ℝ := sorry

/-- The fifteen-sided figure defined in the problem -/
def fifteenSidedFigure : Polygon :=
  { vertices := [
      {x := 1, y := 1}, {x := 1, y := 3}, {x := 3, y := 5}, {x := 4, y := 5},
      {x := 5, y := 4}, {x := 5, y := 3}, {x := 6, y := 3}, {x := 6, y := 2},
      {x := 5, y := 1}, {x := 4, y := 1}, {x := 3, y := 2}, {x := 2, y := 2},
      {x := 1, y := 1}
    ]
  }

/-- Theorem stating that the area of the fifteen-sided figure is 11 cm² -/
theorem fifteenSidedFigureArea : area fifteenSidedFigure = 11 := by sorry

end NUMINAMATH_CALUDE_fifteenSidedFigureArea_l2534_253423


namespace NUMINAMATH_CALUDE_triangle_inequality_ratio_l2534_253416

theorem triangle_inequality_ratio (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) ≥ 1 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    (a'^2 + b'^2 + c'^2) / (a'*b' + b'*c' + c'*a') = 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_ratio_l2534_253416


namespace NUMINAMATH_CALUDE_survey_preference_theorem_l2534_253406

theorem survey_preference_theorem (total_students : ℕ) 
                                  (mac_preference : ℕ) 
                                  (no_preference : ℕ) 
                                  (h1 : total_students = 350)
                                  (h2 : mac_preference = 100)
                                  (h3 : no_preference = 140) : 
  total_students - mac_preference - (mac_preference / 5) - no_preference = 90 := by
  sorry

#check survey_preference_theorem

end NUMINAMATH_CALUDE_survey_preference_theorem_l2534_253406


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_theorem_l2534_253468

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 3 different mathematics books and 3 different Chinese books
    on a shelf, such that books of the same type are not adjacent. -/
theorem book_arrangement_count : ℕ := 
  2 * permutations 3 * permutations 3

/-- Prove that the number of ways to arrange 3 different mathematics books and 3 different Chinese books
    on a shelf, such that books of the same type are not adjacent, is equal to 72. -/
theorem book_arrangement_theorem : book_arrangement_count = 72 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_theorem_l2534_253468


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_y_equals_5_l2534_253401

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, prove that if they are parallel, then y = 5 -/
theorem parallel_vectors_imply_y_equals_5 :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, y + 1)
  parallel a b → y = 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_y_equals_5_l2534_253401


namespace NUMINAMATH_CALUDE_cylindrical_cans_radius_l2534_253499

/-- Proves that for two cylindrical cans with equal volumes, where one can is four times taller than the other,
    if the taller can has a radius of 5 units, then the shorter can has a radius of 10 units. -/
theorem cylindrical_cans_radius (volume : ℝ) (h : ℝ) (r : ℝ) :
  volume = 500 ∧
  volume = π * 5^2 * (4 * h) ∧
  volume = π * r^2 * h →
  r = 10 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_cans_radius_l2534_253499


namespace NUMINAMATH_CALUDE_train_speed_l2534_253422

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 240) (h2 : time = 6) :
  length / time = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2534_253422


namespace NUMINAMATH_CALUDE_correct_calculation_l2534_253482

theorem correct_calculation (x : ℤ) (h : x - 48 = 52) : x + 48 = 148 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2534_253482


namespace NUMINAMATH_CALUDE_parabola_shift_l2534_253421

/-- Given a parabola y = x^2 + 2, shifting it 3 units left and 4 units down results in y = (x + 3)^2 - 2 -/
theorem parabola_shift (x y : ℝ) : 
  (y = x^2 + 2) → 
  (y = (x + 3)^2 - 2) ↔ 
  (y + 4 = ((x + 3) + 3)^2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2534_253421


namespace NUMINAMATH_CALUDE_greatest_difference_is_nine_l2534_253443

/-- A three-digit integer in the form 84x that is a multiple of 3 -/
def ValidNumber (x : ℕ) : Prop :=
  x < 10 ∧ (840 + x) % 3 = 0

/-- The set of all valid x values -/
def ValidXSet : Set ℕ :=
  {x | ValidNumber x}

/-- The greatest possible difference between two valid x values -/
theorem greatest_difference_is_nine :
  ∃ (a b : ℕ), a ∈ ValidXSet ∧ b ∈ ValidXSet ∧
    ∀ (x y : ℕ), x ∈ ValidXSet → y ∈ ValidXSet →
      (a - b : ℤ).natAbs ≥ (x - y : ℤ).natAbs ∧
      (a - b : ℤ).natAbs = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_is_nine_l2534_253443


namespace NUMINAMATH_CALUDE_box_weight_example_l2534_253470

/-- Calculates the weight of an open box given its dimensions, thickness, and metal density. -/
def box_weight (length width height thickness : ℝ) (metal_density : ℝ) : ℝ :=
  let outer_volume := length * width * height
  let inner_length := length - 2 * thickness
  let inner_width := width - 2 * thickness
  let inner_height := height - thickness
  let inner_volume := inner_length * inner_width * inner_height
  let metal_volume := outer_volume - inner_volume
  metal_volume * metal_density

/-- Theorem stating that the weight of the specified box is 5504 grams. -/
theorem box_weight_example : 
  box_weight 50 40 23 2 0.5 = 5504 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_example_l2534_253470


namespace NUMINAMATH_CALUDE_empty_box_weight_l2534_253439

def box_weight_problem (initial_weight : ℝ) (half_removed_weight : ℝ) : Prop :=
  ∃ (apple_weight : ℝ) (num_apples : ℕ) (box_weight : ℝ),
    initial_weight = box_weight + apple_weight * num_apples ∧
    half_removed_weight = box_weight + apple_weight * (num_apples / 2) ∧
    box_weight = 1

theorem empty_box_weight :
  box_weight_problem 9 5 := by
  sorry

end NUMINAMATH_CALUDE_empty_box_weight_l2534_253439


namespace NUMINAMATH_CALUDE_xf_inequality_solution_l2534_253444

noncomputable section

variable (f : ℝ → ℝ)

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- When x < 0, f(x) + xf'(x) < 0
def condition_negative (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x + x * (deriv f x) < 0

-- f(3) = 0
def f_3_is_0 (f : ℝ → ℝ) : Prop := f 3 = 0

-- The solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | x < -3 ∨ (0 < x ∧ x < 3)}

theorem xf_inequality_solution
  (heven : even_function f)
  (hneg : condition_negative f)
  (hf3 : f_3_is_0 f) :
  {x : ℝ | x * f x > 0} = solution_set f :=
sorry

end

end NUMINAMATH_CALUDE_xf_inequality_solution_l2534_253444


namespace NUMINAMATH_CALUDE_base_conversion_2450_l2534_253433

/-- Converts a base-10 number to its base-8 representation -/
def toBase8 (n : ℕ) : ℕ := sorry

/-- Converts a base-8 number to its base-10 representation -/
def fromBase8 (n : ℕ) : ℕ := sorry

theorem base_conversion_2450 :
  toBase8 2450 = 4622 ∧ fromBase8 4622 = 2450 := by sorry

end NUMINAMATH_CALUDE_base_conversion_2450_l2534_253433


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2534_253481

open Set

theorem intersection_of_sets :
  let A : Set ℝ := {x | x > 2}
  let B : Set ℝ := {x | (x - 1) * (x - 3) < 0}
  A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2534_253481


namespace NUMINAMATH_CALUDE_order_of_numbers_l2534_253430

def w : ℕ := 2^129 * 3^81 * 5^128
def x : ℕ := 2^127 * 3^81 * 5^128
def y : ℕ := 2^126 * 3^82 * 5^128
def z : ℕ := 2^125 * 3^82 * 5^129

theorem order_of_numbers : x < y ∧ y < z ∧ z < w := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l2534_253430


namespace NUMINAMATH_CALUDE_eight_player_tournament_l2534_253458

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 8 players, the total number of matches is 28. -/
theorem eight_player_tournament : num_matches 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_eight_player_tournament_l2534_253458


namespace NUMINAMATH_CALUDE_divisibility_by_290304_l2534_253490

theorem divisibility_by_290304 (a b : Nat) (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (ga : a > 7) (gb : b > 7) : 
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_290304_l2534_253490


namespace NUMINAMATH_CALUDE_units_digit_problem_l2534_253488

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_problem : units_digit (8 * 19 * 1978 - 8^3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l2534_253488


namespace NUMINAMATH_CALUDE_initial_orchids_l2534_253469

/-- Proves that the initial number of orchids in the vase was 2, given that there are now 21 orchids
    in the vase after 19 orchids were added. -/
theorem initial_orchids (final_orchids : ℕ) (added_orchids : ℕ) 
  (h1 : final_orchids = 21) 
  (h2 : added_orchids = 19) : 
  final_orchids - added_orchids = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_orchids_l2534_253469


namespace NUMINAMATH_CALUDE_bethany_riding_time_l2534_253428

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of minutes in an hour -/
def minutes_in_hour : ℕ := 60

/-- Represents the number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- Represents Bethany's riding schedule -/
structure RidingSchedule where
  monday : ℕ     -- minutes ridden on Monday
  wednesday : ℕ  -- minutes ridden on Wednesday
  friday : ℕ     -- minutes ridden on Friday
  tuesday : ℕ    -- minutes ridden on Tuesday
  thursday : ℕ   -- minutes ridden on Thursday
  saturday : ℕ   -- minutes ridden on Saturday

/-- Calculates the total minutes ridden in two weeks -/
def total_minutes (schedule : RidingSchedule) : ℕ :=
  2 * (schedule.monday + schedule.wednesday + schedule.friday + 
       schedule.tuesday + schedule.thursday + schedule.saturday)

/-- Theorem stating Bethany's riding time on Monday, Wednesday, and Friday -/
theorem bethany_riding_time (schedule : RidingSchedule) 
  (h1 : schedule.tuesday = 30)
  (h2 : schedule.thursday = 30)
  (h3 : schedule.saturday = 2 * minutes_in_hour)
  (h4 : total_minutes schedule = 12 * minutes_in_hour) :
  2 * (schedule.monday + schedule.wednesday + schedule.friday) = 6 * minutes_in_hour := by
  sorry

#check bethany_riding_time

end NUMINAMATH_CALUDE_bethany_riding_time_l2534_253428


namespace NUMINAMATH_CALUDE_currency_notes_count_l2534_253417

/-- Given a total amount of currency notes and specific conditions, 
    prove the total number of notes. -/
theorem currency_notes_count 
  (total_amount : ℕ) 
  (denomination_70 : ℕ) 
  (denomination_50 : ℕ) 
  (amount_in_50 : ℕ) 
  (h1 : total_amount = 5000)
  (h2 : denomination_70 = 70)
  (h3 : denomination_50 = 50)
  (h4 : amount_in_50 = 100)
  (h5 : ∃ (x y : ℕ), denomination_70 * x + denomination_50 * y = total_amount ∧ 
                     denomination_50 * (amount_in_50 / denomination_50) = amount_in_50) :
  ∃ (x y : ℕ), denomination_70 * x + denomination_50 * y = total_amount ∧ x + y = 72 := by
  sorry

end NUMINAMATH_CALUDE_currency_notes_count_l2534_253417


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l2534_253493

/-- Two different lines in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem plane_perpendicularity (m n : Line3D) (α β : Plane3D) 
  (h1 : m ≠ n) (h2 : α ≠ β) (h3 : perpendicular m α) (h4 : parallel m β) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l2534_253493


namespace NUMINAMATH_CALUDE_cards_distribution_l2534_253483

theorem cards_distribution (total_cards : Nat) (num_people : Nat) (h1 : total_cards = 60) (h2 : num_people = 9) :
  let cards_per_person := total_cards / num_people
  let extra_cards := total_cards % num_people
  let people_with_extra := extra_cards
  num_people - people_with_extra = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2534_253483


namespace NUMINAMATH_CALUDE_periodic_odd_function_sum_l2534_253466

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_sum (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_odd : is_odd f)
  (h_def : ∀ x, 0 < x → x < 1 → f x = 4^x) :
  f (-5/2) + f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_sum_l2534_253466


namespace NUMINAMATH_CALUDE_factorial_gcd_l2534_253471

theorem factorial_gcd : Nat.gcd (Nat.factorial 6) (Nat.factorial 9) = Nat.factorial 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_gcd_l2534_253471


namespace NUMINAMATH_CALUDE_roof_shingle_length_l2534_253479

/-- A rectangular roof shingle with given width and area has a specific length -/
theorem roof_shingle_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 7 → area = 70 → area = width * length → length = 10 := by
  sorry

end NUMINAMATH_CALUDE_roof_shingle_length_l2534_253479


namespace NUMINAMATH_CALUDE_one_match_theorem_one_empty_theorem_l2534_253465

/-- The number of ways to arrange 4 balls in 4 boxes with exactly one match -/
def arrange_one_match : ℕ := 8

/-- The number of ways to arrange 4 balls in 4 boxes with exactly one empty box -/
def arrange_one_empty : ℕ := 144

/-- Theorem for the number of arrangements with exactly one match -/
theorem one_match_theorem : arrange_one_match = 8 := by sorry

/-- Theorem for the number of arrangements with exactly one empty box -/
theorem one_empty_theorem : arrange_one_empty = 144 := by sorry

end NUMINAMATH_CALUDE_one_match_theorem_one_empty_theorem_l2534_253465
