import Mathlib

namespace NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l27_2758

/-- The maximum number of intersection points between circles -/
def max_circle_intersections (n : ℕ) : ℕ := n.choose 2 * 2

/-- The maximum number of intersection points between circles and a line -/
def max_circle_line_intersections (n : ℕ) : ℕ := n * 2

/-- The maximum number of intersection points for n circles and one line -/
def max_total_intersections (n : ℕ) : ℕ :=
  max_circle_intersections n + max_circle_line_intersections n

theorem max_intersections_three_circles_one_line :
  max_total_intersections 3 = 12 := by sorry

end NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l27_2758


namespace NUMINAMATH_CALUDE_machine_a_time_proof_l27_2782

/-- The time it takes for Machine A to finish the job alone -/
def machine_a_time : ℝ := 4

/-- The time it takes for Machine B to finish the job alone -/
def machine_b_time : ℝ := 12

/-- The time it takes for Machine C to finish the job alone -/
def machine_c_time : ℝ := 6

/-- The time it takes for all machines to finish the job together -/
def combined_time : ℝ := 2

theorem machine_a_time_proof :
  (1 / machine_a_time + 1 / machine_b_time + 1 / machine_c_time) * combined_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_machine_a_time_proof_l27_2782


namespace NUMINAMATH_CALUDE_fraction_of_third_is_eighth_l27_2796

theorem fraction_of_third_is_eighth (x : ℚ) : x * (1/3 : ℚ) = 1/8 → x = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_third_is_eighth_l27_2796


namespace NUMINAMATH_CALUDE_test_scores_l27_2788

/-- Represents the score of a test taker -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  total_questions : Nat
  h_sum : correct + unanswered + incorrect = total_questions

/-- Calculates the score based on the test results -/
def calculate_score (ts : TestScore) : Nat :=
  4 * ts.correct + 2 * ts.unanswered

/-- Checks if a score is possible given the test parameters -/
def is_possible_score (score : Nat) : Prop :=
  ∃ (ts : TestScore), ts.total_questions = 30 ∧ calculate_score ts = score

theorem test_scores :
  is_possible_score 116 ∧
  ¬is_possible_score 117 ∧
  is_possible_score 118 ∧
  ¬is_possible_score 119 ∧
  is_possible_score 120 :=
sorry

end NUMINAMATH_CALUDE_test_scores_l27_2788


namespace NUMINAMATH_CALUDE_emily_savings_l27_2751

def shoe_price : ℕ := 50
def promotion_b_discount : ℕ := 20

def cost_promotion_a (price : ℕ) : ℕ := price + price / 2

def cost_promotion_b (price : ℕ) (discount : ℕ) : ℕ := price + (price - discount)

theorem emily_savings : 
  cost_promotion_b shoe_price promotion_b_discount - cost_promotion_a shoe_price = 5 := by
sorry

end NUMINAMATH_CALUDE_emily_savings_l27_2751


namespace NUMINAMATH_CALUDE_polynomial_degree_theorem_l27_2722

theorem polynomial_degree_theorem : 
  let p : Polynomial ℝ := (X^3 + 1)^5 * (X^4 + 1)^2
  Polynomial.degree p = 23 := by sorry

end NUMINAMATH_CALUDE_polynomial_degree_theorem_l27_2722


namespace NUMINAMATH_CALUDE_no_real_roots_of_quartic_equation_l27_2777

theorem no_real_roots_of_quartic_equation :
  ∀ x : ℝ, 5 * x^4 - 28 * x^3 + 57 * x^2 - 28 * x + 5 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_of_quartic_equation_l27_2777


namespace NUMINAMATH_CALUDE_hyperbola_equation_l27_2724

/-- A hyperbola is defined by its equation and properties --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eqn : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1
  a_pos : a > 0
  b_pos : b > 0
  imaginary_axis : b = 1
  asymptote : (x : ℝ) → x / 2 = a / b

/-- The theorem states that a hyperbola with given properties has a specific equation --/
theorem hyperbola_equation (h : Hyperbola) : 
  ∀ x y : ℝ, x^2 / 4 - y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l27_2724


namespace NUMINAMATH_CALUDE_symmetric_periodic_function_max_period_l27_2718

/-- A function with symmetry around x=1 and x=8, and a periodic property -/
def SymmetricPeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T > 0 ∧
  (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x : ℝ, f (1 + x) = f (1 - x)) ∧
  (∀ x : ℝ, f (8 + x) = f (8 - x)) ∧
  ∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T'

theorem symmetric_periodic_function_max_period :
  ∀ f : ℝ → ℝ, SymmetricPeriodicFunction f →
  ∃ T : ℝ, T > 0 ∧ SymmetricPeriodicFunction f ∧ T = 14 ∧
  ∀ T' : ℝ, T' > 0 → SymmetricPeriodicFunction f → T' ≤ T :=
sorry

end NUMINAMATH_CALUDE_symmetric_periodic_function_max_period_l27_2718


namespace NUMINAMATH_CALUDE_two_true_propositions_l27_2760

theorem two_true_propositions :
  let P : ℝ → Prop := λ x => x > -3
  let Q : ℝ → Prop := λ x => x > -6
  let original := ∀ x, P x → Q x
  let converse := ∀ x, Q x → P x
  let inverse := ∀ x, ¬(P x) → ¬(Q x)
  let contrapositive := ∀ x, ¬(Q x) → ¬(P x)
  (original ∧ contrapositive ∧ ¬converse ∧ ¬inverse) ∨
  (original ∧ contrapositive ∧ converse ∧ ¬inverse) ∨
  (original ∧ contrapositive ∧ ¬converse ∧ inverse) :=
by
  sorry


end NUMINAMATH_CALUDE_two_true_propositions_l27_2760


namespace NUMINAMATH_CALUDE_arnel_kept_pencils_l27_2787

/-- Calculates the number of pencils Arnel kept given the problem conditions -/
def pencils_kept (num_boxes : ℕ) (num_friends : ℕ) (pencils_per_friend : ℕ) (pencils_left_per_box : ℕ) : ℕ :=
  let total_pencils := num_boxes * (pencils_per_friend * num_friends / num_boxes + pencils_left_per_box)
  total_pencils - pencils_per_friend * num_friends

/-- Theorem stating that Arnel kept 50 pencils under the given conditions -/
theorem arnel_kept_pencils :
  pencils_kept 10 5 8 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_arnel_kept_pencils_l27_2787


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l27_2750

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (hr2 : r ≠ 0) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l27_2750


namespace NUMINAMATH_CALUDE_unique_solution_for_sum_and_product_l27_2709

theorem unique_solution_for_sum_and_product (x y z : ℝ) :
  x + y + z = 38 →
  x * y * z = 2002 →
  0 < x →
  x ≤ 11 →
  z ≥ 14 →
  x = 11 ∧ y = 13 ∧ z = 14 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_sum_and_product_l27_2709


namespace NUMINAMATH_CALUDE_lcm_gcd_fraction_lower_bound_lcm_gcd_fraction_bound_achievable_l27_2726

theorem lcm_gcd_fraction_lower_bound (a b c : ℕ+) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (Nat.lcm a b + Nat.lcm b c + Nat.lcm c a : ℚ) / (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) ≥ 5 / 2 :=
sorry

theorem lcm_gcd_fraction_bound_achievable :
  ∃ (a b c : ℕ+), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (Nat.lcm a b + Nat.lcm b c + Nat.lcm c a : ℚ) / (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) = 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_lcm_gcd_fraction_lower_bound_lcm_gcd_fraction_bound_achievable_l27_2726


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l27_2703

/-- The quadratic equation x^2 - 2x + k - 1 = 0 has two distinct real roots if and only if k < 2 -/
theorem quadratic_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k - 1 = 0 ∧ y^2 - 2*y + k - 1 = 0) ↔ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l27_2703


namespace NUMINAMATH_CALUDE_cary_calorie_deficit_l27_2714

/-- Calculates the net calorie deficit for a person walking a round trip and consuming a candy bar. -/
def net_calorie_deficit (round_trip_distance : ℕ) (calories_per_mile : ℕ) (candy_bar_calories : ℕ) : ℕ :=
  round_trip_distance * calories_per_mile - candy_bar_calories

/-- Proves that given specific conditions, the net calorie deficit is 250 calories. -/
theorem cary_calorie_deficit :
  net_calorie_deficit 3 150 200 = 250 := by
  sorry

#eval net_calorie_deficit 3 150 200

end NUMINAMATH_CALUDE_cary_calorie_deficit_l27_2714


namespace NUMINAMATH_CALUDE_range_of_m_l27_2754

/-- A function f is decreasing on (0, +∞) -/
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f y < f x

/-- The solution set of (x-1)² > m is ℝ -/
def SolutionSetIsReals (m : ℝ) : Prop :=
  ∀ x, (x - 1)^2 > m

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : IsDecreasingOn f)
  (h2 : SolutionSetIsReals m)
  (h3 : (IsDecreasingOn f) ∨ (SolutionSetIsReals m))
  (h4 : ¬((IsDecreasingOn f) ∧ (SolutionSetIsReals m))) :
  0 ≤ m ∧ m < (1/2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l27_2754


namespace NUMINAMATH_CALUDE_fraction_reciprocal_l27_2766

theorem fraction_reciprocal (a b : ℚ) (h : a ≠ b) :
  let c := -(a + b)
  (a + c) / (b + c) = b / a := by
sorry

end NUMINAMATH_CALUDE_fraction_reciprocal_l27_2766


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l27_2791

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z := (3 + i) / (1 - i)
  Complex.im z = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l27_2791


namespace NUMINAMATH_CALUDE_remainder_sum_l27_2768

theorem remainder_sum (n : ℤ) (h : n % 24 = 11) : (n % 4 + n % 6 = 8) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l27_2768


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l27_2773

theorem unique_solution_cube_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l27_2773


namespace NUMINAMATH_CALUDE_product_equals_zero_l27_2755

theorem product_equals_zero (b : ℤ) (h : b = 3) : 
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b * (b + 1) * (b + 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_product_equals_zero_l27_2755


namespace NUMINAMATH_CALUDE_wedding_couples_theorem_l27_2705

/-- The number of couples invited by the bride and groom to their wedding reception --/
def couples_invited (total_guests : ℕ) (friends : ℕ) : ℕ :=
  (total_guests - friends) / 2

theorem wedding_couples_theorem (total_guests : ℕ) (friends : ℕ) 
  (h1 : total_guests = 180) 
  (h2 : friends = 100) :
  couples_invited total_guests friends = 40 := by
  sorry

end NUMINAMATH_CALUDE_wedding_couples_theorem_l27_2705


namespace NUMINAMATH_CALUDE_intersection_and_union_union_equality_condition_l27_2767

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 6}
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m}

-- Theorem for part (I)
theorem intersection_and_union :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 6}) ∧
  ((Aᶜ) ∪ B = {x | -1 < x ∧ x ≤ 6}) :=
sorry

-- Theorem for part (II)
theorem union_equality_condition (m : ℝ) :
  (B ∪ C m = B) ↔ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_and_union_union_equality_condition_l27_2767


namespace NUMINAMATH_CALUDE_prob_B_wins_match_value_l27_2786

/-- The probability of player B winning a single game -/
def p_B : ℝ := 0.4

/-- The probability of player A winning a single game -/
def p_A : ℝ := 1 - p_B

/-- The probability of player B winning a best-of-three billiards match -/
def prob_B_wins_match : ℝ := p_B^2 + 2 * p_B^2 * p_A

theorem prob_B_wins_match_value :
  prob_B_wins_match = 0.352 := by sorry

end NUMINAMATH_CALUDE_prob_B_wins_match_value_l27_2786


namespace NUMINAMATH_CALUDE_same_color_probability_is_five_eighteenths_l27_2706

/-- Represents the number of jelly beans of each color that Abe has -/
structure AbeJellyBeans where
  green : Nat
  blue : Nat

/-- Represents the number of jelly beans of each color that Bob has -/
structure BobJellyBeans where
  green : Nat
  blue : Nat
  red : Nat

/-- Calculates the probability of both Abe and Bob showing the same color jelly bean -/
def probability_same_color (abe : AbeJellyBeans) (bob : BobJellyBeans) : Rat :=
  sorry

/-- The main theorem stating the probability of Abe and Bob showing the same color jelly bean -/
theorem same_color_probability_is_five_eighteenths 
  (abe : AbeJellyBeans) 
  (bob : BobJellyBeans) 
  (h1 : abe.green = 2)
  (h2 : abe.blue = 1)
  (h3 : bob.green = 2)
  (h4 : bob.blue = 1)
  (h5 : bob.red = 3) :
  probability_same_color abe bob = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_five_eighteenths_l27_2706


namespace NUMINAMATH_CALUDE_triangle_angle_equality_l27_2708

theorem triangle_angle_equality (A B C : ℝ) (a b c : ℝ) :
  0 < B ∧ B < π →
  0 < a ∧ 0 < b ∧ 0 < c →
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A →
  B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_equality_l27_2708


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_graph_not_in_third_quadrant_l27_2764

/-- A linear function f(x) = kx + b does not pass through the third quadrant
    if and only if k < 0 and b > 0 -/
theorem linear_function_not_in_third_quadrant (k b : ℝ) :
  k < 0 ∧ b > 0 → ∀ x y : ℝ, y = k * x + b → ¬(x < 0 ∧ y < 0) := by
  sorry

/-- The graph of y = -2x + 1 does not pass through the third quadrant -/
theorem graph_not_in_third_quadrant :
  ∀ x y : ℝ, y = -2 * x + 1 → ¬(x < 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_graph_not_in_third_quadrant_l27_2764


namespace NUMINAMATH_CALUDE_wage_problem_l27_2793

/-- Given a sum of money S that can pay q's wages for 40 days and both p and q's wages for 15 days,
    prove that S can pay p's wages for 24 days. -/
theorem wage_problem (S P Q : ℝ) (hS_positive : S > 0) (hP_positive : P > 0) (hQ_positive : Q > 0)
  (hS_q : S = 40 * Q) (hS_pq : S = 15 * (P + Q)) :
  S = 24 * P := by
  sorry

end NUMINAMATH_CALUDE_wage_problem_l27_2793


namespace NUMINAMATH_CALUDE_candy_distribution_l27_2744

theorem candy_distribution (x : ℚ) 
  (laura_candies : x > 0)
  (mark_candies : ℚ → ℚ)
  (nina_candies : ℚ → ℚ)
  (oliver_candies : ℚ → ℚ)
  (mark_def : mark_candies x = 4 * x)
  (nina_def : nina_candies x = 2 * mark_candies x)
  (oliver_def : oliver_candies x = 6 * nina_candies x)
  (total_candies : x + mark_candies x + nina_candies x + oliver_candies x = 360) :
  x = 360 / 61 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l27_2744


namespace NUMINAMATH_CALUDE_complex_sum_zero_l27_2713

theorem complex_sum_zero : 
  let z : ℂ := (1 / 2 : ℂ) + (Complex.I * Real.sqrt 3 / 2)
  z + z^2 + z^3 + z^4 + z^5 + z^6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l27_2713


namespace NUMINAMATH_CALUDE_log_2_base_10_bounds_l27_2778

-- Define the given conditions
axiom pow_10_4 : (10 : ℝ) ^ 4 = 10000
axiom pow_10_5 : (10 : ℝ) ^ 5 = 100000
axiom pow_2_12 : (2 : ℝ) ^ 12 = 4096
axiom pow_2_15 : (2 : ℝ) ^ 15 = 32768

-- State the theorem
theorem log_2_base_10_bounds :
  0.30 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_log_2_base_10_bounds_l27_2778


namespace NUMINAMATH_CALUDE_rabbit_toy_cost_l27_2734

theorem rabbit_toy_cost (total_cost pet_food_cost cage_cost found_money : ℚ)
  (h1 : total_cost = 24.81)
  (h2 : pet_food_cost = 5.79)
  (h3 : cage_cost = 12.51)
  (h4 : found_money = 1.00) :
  total_cost - (pet_food_cost + cage_cost) + found_money = 7.51 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_toy_cost_l27_2734


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l27_2732

/-- An isosceles trapezoid circumscribed about a circle -/
structure IsoscelesTrapezoid where
  /-- The longer base of the trapezoid -/
  longerBase : ℝ
  /-- One base angle of the trapezoid -/
  baseAngle : ℝ

/-- The area of the isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem isosceles_trapezoid_area
  (t : IsoscelesTrapezoid)
  (h1 : t.longerBase = 16)
  (h2 : t.baseAngle = Real.arcsin 0.8) :
  area t = 80 :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l27_2732


namespace NUMINAMATH_CALUDE_softball_team_composition_l27_2765

theorem softball_team_composition (total : ℕ) (ratio : ℚ) : 
  total = 20 → ratio = 2/3 → ∃ (men women : ℕ), 
    men + women = total ∧ 
    (men : ℚ) / (women : ℚ) = ratio ∧ 
    women = men + 4 := by
  sorry

end NUMINAMATH_CALUDE_softball_team_composition_l27_2765


namespace NUMINAMATH_CALUDE_team_formation_count_l27_2742

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of male doctors -/
def male_doctors : ℕ := 5

/-- The number of female doctors -/
def female_doctors : ℕ := 4

/-- The size of the team -/
def team_size : ℕ := 3

/-- The number of ways to form a team with both male and female doctors -/
def team_formations : ℕ := 
  choose male_doctors 2 * choose female_doctors 1 + 
  choose male_doctors 1 * choose female_doctors 2

theorem team_formation_count : team_formations = 70 := by sorry

end NUMINAMATH_CALUDE_team_formation_count_l27_2742


namespace NUMINAMATH_CALUDE_integer_multiplication_result_l27_2728

theorem integer_multiplication_result (x : ℤ) : 
  (10 * x = 64 ∨ 10 * x = 32 ∨ 10 * x = 12 ∨ 10 * x = 25 ∨ 10 * x = 30) → 10 * x = 30 := by
sorry

end NUMINAMATH_CALUDE_integer_multiplication_result_l27_2728


namespace NUMINAMATH_CALUDE_arc_length_for_given_circle_l27_2761

/-- Given a circle with radius 2 and a central angle of 2 radians, 
    the corresponding arc length is 4. -/
theorem arc_length_for_given_circle : 
  ∀ (r θ l : ℝ), r = 2 → θ = 2 → l = r * θ → l = 4 :=
by sorry

end NUMINAMATH_CALUDE_arc_length_for_given_circle_l27_2761


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l27_2737

def first_four_primes : List Nat := [2, 3, 5, 7]

theorem arithmetic_mean_reciprocals_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l27_2737


namespace NUMINAMATH_CALUDE_exists_perfect_square_2022_not_perfect_square_for_a_2_l27_2747

-- Part (a)
theorem exists_perfect_square_2022 : ∃ n : ℕ, ∃ k : ℕ, n * (n + 2022) + 2 = k^2 := by
  sorry

-- Part (b)
theorem not_perfect_square_for_a_2 : ∀ n : ℕ, ¬∃ k : ℕ, n * (n + 2) + 2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_perfect_square_2022_not_perfect_square_for_a_2_l27_2747


namespace NUMINAMATH_CALUDE_grass_field_width_l27_2753

theorem grass_field_width 
  (length : ℝ) 
  (path_width : ℝ) 
  (path_area : ℝ) 
  (h1 : length = 75) 
  (h2 : path_width = 2.8) 
  (h3 : path_area = 1518.72) : 
  ∃ width : ℝ, 
    (length + 2 * path_width) * (width + 2 * path_width) - length * width = path_area ∧ 
    width = 190.6 := by
  sorry

end NUMINAMATH_CALUDE_grass_field_width_l27_2753


namespace NUMINAMATH_CALUDE_find_other_number_l27_2785

theorem find_other_number (A B : ℕ+) : 
  Nat.gcd A B = 16 →
  Nat.lcm A B = 312 →
  A = 24 →
  B = 208 := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l27_2785


namespace NUMINAMATH_CALUDE_angle_bisector_sum_l27_2762

/-- Triangle with vertices P, Q, R -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- Angle bisector equation coefficients -/
structure AngleBisectorEq where
  a : ℝ
  c : ℝ

/-- Theorem: For the given triangle, the angle bisector equation of ∠P has a + c = 89 -/
theorem angle_bisector_sum (t : Triangle) (eq : AngleBisectorEq) : 
  t.P = (-8, 5) → t.Q = (-15, -19) → t.R = (1, -7) → 
  (∃ (x y : ℝ), eq.a * x + 2 * y + eq.c = 0) →
  eq.a + eq.c = 89 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_l27_2762


namespace NUMINAMATH_CALUDE_max_profit_is_21000_l27_2725

/-- Represents the production capabilities and constraints of a furniture factory -/
structure FurnitureFactory where
  carpenterHoursChair : ℕ
  carpenterHoursDesk : ℕ
  maxCarpenterHours : ℕ
  painterHoursChair : ℕ
  painterHoursDesk : ℕ
  maxPainterHours : ℕ
  profitChair : ℕ
  profitDesk : ℕ

/-- Calculates the profit for a given production plan -/
def calculateProfit (factory : FurnitureFactory) (chairs : ℕ) (desks : ℕ) : ℕ :=
  chairs * factory.profitChair + desks * factory.profitDesk

/-- Checks if a production plan is feasible given the factory's constraints -/
def isFeasible (factory : FurnitureFactory) (chairs : ℕ) (desks : ℕ) : Prop :=
  chairs * factory.carpenterHoursChair + desks * factory.carpenterHoursDesk ≤ factory.maxCarpenterHours ∧
  chairs * factory.painterHoursChair + desks * factory.painterHoursDesk ≤ factory.maxPainterHours

/-- Theorem stating that the maximum profit is 21000 yuan -/
theorem max_profit_is_21000 (factory : FurnitureFactory) 
  (h1 : factory.carpenterHoursChair = 4)
  (h2 : factory.carpenterHoursDesk = 8)
  (h3 : factory.maxCarpenterHours = 8000)
  (h4 : factory.painterHoursChair = 2)
  (h5 : factory.painterHoursDesk = 1)
  (h6 : factory.maxPainterHours = 1300)
  (h7 : factory.profitChair = 15)
  (h8 : factory.profitDesk = 20) :
  (∀ chairs desks, isFeasible factory chairs desks → calculateProfit factory chairs desks ≤ 21000) ∧
  (∃ chairs desks, isFeasible factory chairs desks ∧ calculateProfit factory chairs desks = 21000) :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_21000_l27_2725


namespace NUMINAMATH_CALUDE_monthly_average_production_l27_2745

/-- The daily average production for a month given production rates for different periods -/
theorem monthly_average_production 
  (days_first_period : ℕ) 
  (days_second_period : ℕ) 
  (avg_first_period : ℕ) 
  (avg_second_period : ℕ) 
  (h1 : days_first_period = 25)
  (h2 : days_second_period = 5)
  (h3 : avg_first_period = 50)
  (h4 : avg_second_period = 38) :
  (days_first_period * avg_first_period + days_second_period * avg_second_period) / 
  (days_first_period + days_second_period) = 48 := by
  sorry

#check monthly_average_production

end NUMINAMATH_CALUDE_monthly_average_production_l27_2745


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l27_2729

/-- The set of functions satisfying the given conditions -/
def S : Set (ℕ → ℝ) :=
  {f | f 1 = 2 ∧ ∀ n, f (n + 1) ≥ f n ∧ f n ≥ (n : ℝ) / (n + 1) * f (2 * n)}

/-- The smallest natural number M such that for any f ∈ S and any n ∈ ℕ, f(n) < M -/
def M : ℕ := 10

theorem smallest_upper_bound : 
  (∀ f ∈ S, ∀ n, f n < M) ∧ 
  (∀ m < M, ∃ f ∈ S, ∃ n, f n ≥ m) :=
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l27_2729


namespace NUMINAMATH_CALUDE_marble_capacity_l27_2702

/-- 
Given:
- A small bottle with volume 20 ml can hold 40 marbles
- A larger bottle has volume 60 ml
Prove that the larger bottle can hold 120 marbles
-/
theorem marble_capacity (small_volume small_capacity large_volume : ℕ) 
  (h1 : small_volume = 20)
  (h2 : small_capacity = 40)
  (h3 : large_volume = 60) :
  (large_volume * small_capacity) / small_volume = 120 := by
  sorry

end NUMINAMATH_CALUDE_marble_capacity_l27_2702


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l27_2735

theorem simplify_fraction_product : 
  (3 * 5 : ℚ) / (9 * 11) * (7 * 9 * 11) / (3 * 5 * 7) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l27_2735


namespace NUMINAMATH_CALUDE_gcd_g_102_103_l27_2770

def g (x : ℤ) : ℤ := x^2 - x + 2007

theorem gcd_g_102_103 : Int.gcd (g 102) (g 103) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_102_103_l27_2770


namespace NUMINAMATH_CALUDE_characterization_of_n_l27_2763

def has_finite_multiples_with_n_divisors (n : ℕ+) : Prop :=
  ∃ (S : Finset ℕ+), ∀ (k : ℕ+), (n ∣ k) → (Nat.card (Nat.divisors k) = n) → k ∈ S

def not_divisible_by_square_of_prime (n : ℕ+) : Prop :=
  ∀ (p : ℕ+), Nat.Prime p → (p * p ∣ n) → False

theorem characterization_of_n (n : ℕ+) :
  has_finite_multiples_with_n_divisors n ↔ not_divisible_by_square_of_prime n ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_n_l27_2763


namespace NUMINAMATH_CALUDE_triangle_area_l27_2756

theorem triangle_area (base height : ℝ) (h1 : base = 25) (h2 : height = 60) :
  (base * height) / 2 = 750 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l27_2756


namespace NUMINAMATH_CALUDE_rationalize_sum_l27_2710

theorem rationalize_sum (A B C D E F G H I : ℤ) : 
  (∃ (a b c d e f g h i : ℚ),
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) = 
    (a * Real.sqrt b + c * Real.sqrt d + e * Real.sqrt f + g * Real.sqrt h) / i ∧
    a = A ∧ b = B ∧ c = C ∧ d = D ∧ e = E ∧ f = F ∧ g = G ∧ h = H ∧ i = I ∧
    i > 0 ∧
    -- Simplest radical form and lowest terms conditions would be defined here
    True) →
  A + B + C + D + E + F + G + H + I = 112 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_sum_l27_2710


namespace NUMINAMATH_CALUDE_grid_adjacent_difference_l27_2752

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 18
  col : Fin 18

/-- The type of the grid -/
def Grid := Fin 18 → Fin 18 → ℕ+

/-- Two cells are adjacent if they share an edge -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- The main theorem -/
theorem grid_adjacent_difference (g : Grid) 
  (h : ∀ (c1 c2 : Cell), c1 ≠ c2 → g c1.row c1.col ≠ g c2.row c2.col) :
  ∃ (c1 c2 c3 c4 : Cell), 
    adjacent c1 c2 ∧ adjacent c3 c4 ∧ 
    (c1, c2) ≠ (c3, c4) ∧
    (g c1.row c1.col).val + 10 ≤ (g c2.row c2.col).val ∧
    (g c3.row c3.col).val + 10 ≤ (g c4.row c4.col).val :=
sorry

end NUMINAMATH_CALUDE_grid_adjacent_difference_l27_2752


namespace NUMINAMATH_CALUDE_flag_arrangement_count_l27_2736

def total_arrangements (n m : ℕ) : ℕ := (n + m).factorial / (n.factorial * m.factorial)

def consecutive_arrangements (n m : ℕ) : ℕ := (m + 1).factorial / m.factorial

theorem flag_arrangement_count : 
  let total := total_arrangements 3 4
  let red_consecutive := consecutive_arrangements 1 4
  let blue_consecutive := consecutive_arrangements 3 1
  let both_consecutive := 2
  total - red_consecutive - blue_consecutive + both_consecutive = 28 := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_count_l27_2736


namespace NUMINAMATH_CALUDE_pie_shop_pricing_l27_2749

/-- The number of slices per whole pie -/
def slices_per_pie : ℕ := 4

/-- The number of pies sold -/
def pies_sold : ℕ := 9

/-- The total revenue from selling all pies -/
def total_revenue : ℕ := 180

/-- The price per slice of pie -/
def price_per_slice : ℚ := 5

theorem pie_shop_pricing :
  price_per_slice = total_revenue / (pies_sold * slices_per_pie) := by
  sorry

end NUMINAMATH_CALUDE_pie_shop_pricing_l27_2749


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l27_2792

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y + 1) = x + y + 1

/-- The theorem stating that there exists exactly one function satisfying the equation -/
theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, SatisfiesEquation f :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l27_2792


namespace NUMINAMATH_CALUDE_pizza_area_increase_l27_2740

theorem pizza_area_increase : 
  let small_diameter : ℝ := 12
  let large_diameter : ℝ := 18
  let small_area := Real.pi * (small_diameter / 2)^2
  let large_area := Real.pi * (large_diameter / 2)^2
  let area_increase := large_area - small_area
  let percent_increase := (area_increase / small_area) * 100
  percent_increase = 125 :=
by sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l27_2740


namespace NUMINAMATH_CALUDE_angle_supplement_l27_2730

theorem angle_supplement (α : ℝ) : 
  (90 - α = 125) → (180 - α = 125) := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_l27_2730


namespace NUMINAMATH_CALUDE_calculate_food_price_l27_2769

/-- Given a total bill that includes tax and tip, calculate the original food price -/
theorem calculate_food_price (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (food_price : ℝ) : 
  total = 211.20 ∧ 
  tax_rate = 0.10 ∧ 
  tip_rate = 0.20 ∧ 
  total = food_price * (1 + tax_rate) * (1 + tip_rate) → 
  food_price = 160 := by
  sorry

end NUMINAMATH_CALUDE_calculate_food_price_l27_2769


namespace NUMINAMATH_CALUDE_certain_amount_calculation_l27_2704

theorem certain_amount_calculation (x : ℝ) (A : ℝ) (h1 : x = 190) (h2 : 0.65 * x = 0.20 * A) : A = 617.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_calculation_l27_2704


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_l27_2716

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (perpendicularLineToPlane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_to_plane 
  (α β : Plane) (m n : Line) :
  perpendicular α β →
  intersect α β m →
  contains α n →
  perpendicularLines n m →
  perpendicularLineToPlane n β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_l27_2716


namespace NUMINAMATH_CALUDE_chemistry_books_count_l27_2723

theorem chemistry_books_count (biology_books : ℕ) (total_ways : ℕ) : 
  biology_books = 14 →
  total_ways = 2548 →
  (∃ chemistry_books : ℕ, 
    total_ways = (biology_books.choose 2) * (chemistry_books.choose 2)) →
  ∃ chemistry_books : ℕ, chemistry_books = 8 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_books_count_l27_2723


namespace NUMINAMATH_CALUDE_trivia_team_score_l27_2794

/-- Represents a trivia team member's performance -/
structure MemberPerformance where
  two_point_questions : ℕ
  four_point_questions : ℕ
  six_point_questions : ℕ

/-- Calculates the total points for a member's performance -/
def calculate_member_points (performance : MemberPerformance) : ℕ :=
  2 * performance.two_point_questions +
  4 * performance.four_point_questions +
  6 * performance.six_point_questions

/-- The trivia team's performance -/
def team_performance : List MemberPerformance := [
  ⟨3, 0, 0⟩, -- Member A
  ⟨0, 5, 1⟩, -- Member B
  ⟨0, 0, 2⟩, -- Member C
  ⟨4, 2, 0⟩, -- Member D
  ⟨1, 3, 0⟩, -- Member E
  ⟨0, 0, 5⟩, -- Member F
  ⟨1, 2, 0⟩, -- Member G
  ⟨2, 0, 3⟩, -- Member H
  ⟨0, 1, 4⟩, -- Member I
  ⟨7, 1, 0⟩  -- Member J
]

theorem trivia_team_score :
  (team_performance.map calculate_member_points).sum = 182 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_score_l27_2794


namespace NUMINAMATH_CALUDE_hexagon_triangle_count_l27_2739

/-- Regular hexagon with area 6 -/
structure RegularHexagon :=
  (area : ℝ)
  (is_regular : area = 6)

/-- Equilateral triangle with area 4 -/
structure EquilateralTriangle :=
  (area : ℝ)
  (is_equilateral : area = 4)

/-- Configuration of four regular hexagons -/
def HexagonConfiguration := Fin 4 → RegularHexagon

/-- Count of equilateral triangles formed by vertices of hexagons -/
def count_triangles (config : HexagonConfiguration) : ℕ := sorry

/-- Main theorem: There are 12 equilateral triangles with area 4 -/
theorem hexagon_triangle_count (config : HexagonConfiguration) :
  count_triangles config = 12 := by sorry

end NUMINAMATH_CALUDE_hexagon_triangle_count_l27_2739


namespace NUMINAMATH_CALUDE_prime_power_fraction_l27_2700

theorem prime_power_fraction (u v : ℕ+) :
  (∃ (p : ℕ) (n : ℕ), Prime p ∧ (u.val * v.val^3 : ℚ) / (u.val^2 + v.val^2) = p^n) ↔
  (∃ (k : ℕ), k ≥ 1 ∧ u.val = 2^k ∧ v.val = 2^k) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_fraction_l27_2700


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_fractional_inequality_solution_l27_2715

-- Part 1
theorem quadratic_inequality_solution (x : ℝ) :
  x^2 + 3*x - 4 > 0 ↔ x > 1 ∨ x < -4 := by sorry

-- Part 2
theorem fractional_inequality_solution (x : ℝ) :
  x ≠ 5 → ((1 - x) / (x - 5) ≥ 1 ↔ 3 ≤ x ∧ x < 5) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_fractional_inequality_solution_l27_2715


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l27_2701

theorem min_perimeter_triangle (a b c : ℕ) : 
  a = 52 → b = 76 → c > 0 → 
  (a + b > c) → (a + c > b) → (b + c > a) →
  (∀ x : ℕ, x > 0 → (a + b > x) → (a + x > b) → (b + x > a) → c ≤ x) →
  a + b + c = 153 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l27_2701


namespace NUMINAMATH_CALUDE_sum_lent_l27_2776

/-- Given a sum of money divided into two parts where:
    1) The interest on the first part for 8 years at 3% per annum
       equals the interest on the second part for 3 years at 5% per annum
    2) The second part is Rs. 1680
    Prove that the total sum lent is Rs. 2730 -/
theorem sum_lent (first_part second_part : ℝ) : 
  second_part = 1680 →
  (first_part * 8 * 3) / 100 = (second_part * 3 * 5) / 100 →
  first_part + second_part = 2730 := by
  sorry

#check sum_lent

end NUMINAMATH_CALUDE_sum_lent_l27_2776


namespace NUMINAMATH_CALUDE_tallest_tree_height_l27_2741

theorem tallest_tree_height (h_shortest h_middle h_tallest : ℝ) : 
  h_middle = (2/3) * h_tallest →
  h_shortest = (1/2) * h_middle →
  h_shortest = 50 →
  h_tallest = 150 := by
sorry

end NUMINAMATH_CALUDE_tallest_tree_height_l27_2741


namespace NUMINAMATH_CALUDE_hyperbola_t_squared_l27_2759

/-- A hyperbola centered at the origin, opening horizontally -/
structure Hyperbola where
  /-- The equation of the hyperbola: x²/a² - y²/b² = 1 -/
  equation : ℝ → ℝ → Prop

/-- The hyperbola passes through the given points -/
def passes_through (h : Hyperbola) (x y : ℝ) : Prop :=
  h.equation x y

theorem hyperbola_t_squared (h : Hyperbola) :
  passes_through h 2 3 →
  passes_through h 3 0 →
  passes_through h t 5 →
  t^2 = 1854/81 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_t_squared_l27_2759


namespace NUMINAMATH_CALUDE_proposition_analysis_l27_2779

theorem proposition_analysis (P Q : Prop) 
  (h_P : P ↔ (2 + 2 = 5))
  (h_Q : Q ↔ (3 > 2)) : 
  (¬(P ∧ Q)) ∧ (¬P) := by
  sorry

end NUMINAMATH_CALUDE_proposition_analysis_l27_2779


namespace NUMINAMATH_CALUDE_original_recipe_flour_amount_l27_2781

/-- Given a recipe that uses 8 ounces of butter for some amount of flour,
    and knowing that 12 ounces of butter is used for 56 cups of flour
    when the recipe is quadrupled, prove that the original recipe
    requires 37 cups of flour. -/
theorem original_recipe_flour_amount :
  ∀ (x : ℚ),
  (8 : ℚ) / x = (12 : ℚ) / (4 * 56) →
  x = 37 := by
sorry

end NUMINAMATH_CALUDE_original_recipe_flour_amount_l27_2781


namespace NUMINAMATH_CALUDE_basket_weight_is_20_l27_2738

/-- The weight of the basket in kilograms -/
def basket_weight : ℝ := 20

/-- The lifting capacity of one standard balloon in kilograms -/
def balloon_capacity : ℝ := 60

/-- One standard balloon can lift a basket with contents weighing not more than 80 kg -/
axiom one_balloon_limit : basket_weight + balloon_capacity ≤ 80

/-- Two standard balloons can lift the same basket with contents weighing not more than 180 kg -/
axiom two_balloon_limit : basket_weight + 2 * balloon_capacity ≤ 180

theorem basket_weight_is_20 : basket_weight = 20 := by sorry

end NUMINAMATH_CALUDE_basket_weight_is_20_l27_2738


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l27_2731

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 5 * (b + c))
  (second_eq : b = 9 * c) :
  a * b * c = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l27_2731


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l27_2790

/-- The value of m^2 for which the line y = mx + 2 is tangent to the ellipse x^2 + 9y^2 = 9 -/
theorem line_tangent_to_ellipse :
  ∃ (m : ℝ),
    (∀ (x y : ℝ), y = m * x + 2 → x^2 + 9 * y^2 = 9) →
    (∃! (x y : ℝ), y = m * x + 2 ∧ x^2 + 9 * y^2 = 9) →
    m^2 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l27_2790


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l27_2717

theorem right_triangle_third_side_product (a b c d : ℝ) :
  a = 3 ∧ b = 6 ∧ 
  ((a^2 + b^2 = c^2) ∨ (a^2 + d^2 = b^2)) ∧
  (c > 0) ∧ (d > 0) →
  c * d = Real.sqrt 1215 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l27_2717


namespace NUMINAMATH_CALUDE_rectangle_color_theorem_l27_2783

/-- A cell in the rectangle can be either white or black -/
inductive CellColor
  | White
  | Black

/-- The rectangle is represented as a 3 × 7 matrix of cell colors -/
def Rectangle := Matrix (Fin 3) (Fin 7) CellColor

/-- A point in the rectangle, represented by its row and column -/
structure Point where
  row : Fin 3
  col : Fin 7

/-- Check if four points form a rectangle parallel to the sides of the original rectangle -/
def isParallelRectangle (p1 p2 p3 p4 : Point) : Prop :=
  (p1.row = p2.row ∧ p3.row = p4.row ∧ p1.col = p3.col ∧ p2.col = p4.col) ∨
  (p1.row = p3.row ∧ p2.row = p4.row ∧ p1.col = p2.col ∧ p3.col = p4.col)

/-- Check if all four points have the same color in the given rectangle -/
def sameColor (rect : Rectangle) (p1 p2 p3 p4 : Point) : Prop :=
  rect p1.row p1.col = rect p2.row p2.col ∧
  rect p2.row p2.col = rect p3.row p3.col ∧
  rect p3.row p3.col = rect p4.row p4.col

theorem rectangle_color_theorem (rect : Rectangle) :
  ∃ p1 p2 p3 p4 : Point,
    isParallelRectangle p1 p2 p3 p4 ∧
    sameColor rect p1 p2 p3 p4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_color_theorem_l27_2783


namespace NUMINAMATH_CALUDE_marble_problem_l27_2746

theorem marble_problem (atticus jensen cruz harper : ℕ) : 
  4 * (atticus + jensen + cruz + harper) = 120 →
  atticus = jensen / 2 →
  atticus = 4 →
  jensen = 2 * harper →
  cruz = 14 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l27_2746


namespace NUMINAMATH_CALUDE_fixed_point_of_power_function_l27_2775

/-- For any real α, the function f(x) = (x-1)^α passes through the point (2,1) -/
theorem fixed_point_of_power_function (α : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ (x - 1) ^ α
  f 2 = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_power_function_l27_2775


namespace NUMINAMATH_CALUDE_compound_weight_l27_2748

/-- The atomic weight of Aluminum-27 in atomic mass units -/
def aluminum_weight : ℕ := 27

/-- The atomic weight of Iodine-127 in atomic mass units -/
def iodine_weight : ℕ := 127

/-- The atomic weight of Oxygen-16 in atomic mass units -/
def oxygen_weight : ℕ := 16

/-- The number of Aluminum-27 atoms in the compound -/
def aluminum_count : ℕ := 1

/-- The number of Iodine-127 atoms in the compound -/
def iodine_count : ℕ := 3

/-- The number of Oxygen-16 atoms in the compound -/
def oxygen_count : ℕ := 2

/-- The molecular weight of the compound -/
def molecular_weight : ℕ := 
  aluminum_count * aluminum_weight + 
  iodine_count * iodine_weight + 
  oxygen_count * oxygen_weight

theorem compound_weight : molecular_weight = 440 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l27_2748


namespace NUMINAMATH_CALUDE_two_numbers_difference_l27_2743

theorem two_numbers_difference (a b : ℝ) : 
  a + b = 10 → a^2 - b^2 = 40 → |a - b| = 4 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l27_2743


namespace NUMINAMATH_CALUDE_dvds_bought_online_l27_2733

theorem dvds_bought_online (total : ℕ) (store : ℕ) (online : ℕ) : 
  total = 10 → store = 8 → online = total - store → online = 2 := by
  sorry

end NUMINAMATH_CALUDE_dvds_bought_online_l27_2733


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l27_2711

theorem trigonometric_expression_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l27_2711


namespace NUMINAMATH_CALUDE_percentage_problem_l27_2757

theorem percentage_problem : 
  ∃ (P : ℝ), (0.1 * 30 + P * 50 = 10.5) ∧ (P = 0.15) := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l27_2757


namespace NUMINAMATH_CALUDE_committee_formation_ways_l27_2727

theorem committee_formation_ways (n m : ℕ) (hn : n = 8) (hm : m = 4) :
  Nat.choose n m = 70 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_ways_l27_2727


namespace NUMINAMATH_CALUDE_smallest_x_value_l27_2780

theorem smallest_x_value (x : ℝ) : 
  (3 * x^2 + 36 * x - 90 = 2 * x * (x + 16)) → x ≥ -10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l27_2780


namespace NUMINAMATH_CALUDE_dice_game_winning_probability_l27_2789

/-- Represents the outcome of rolling three dice -/
inductive DiceOutcome
  | AllSame
  | TwoSame
  | AllDifferent

/-- The probability of winning the dice game -/
def winning_probability : ℚ := 2177 / 10000

/-- The strategy for rerolling dice based on the initial outcome -/
def reroll_strategy (outcome : DiceOutcome) : ℕ :=
  match outcome with
  | DiceOutcome.AllSame => 0
  | DiceOutcome.TwoSame => 1
  | DiceOutcome.AllDifferent => 3

theorem dice_game_winning_probability :
  ∀ (num_rolls : ℕ) (max_rerolls : ℕ),
    num_rolls = 3 ∧ max_rerolls = 2 →
    (∀ (outcome : DiceOutcome), reroll_strategy outcome ≤ num_rolls) →
    winning_probability = 2177 / 10000 := by
  sorry


end NUMINAMATH_CALUDE_dice_game_winning_probability_l27_2789


namespace NUMINAMATH_CALUDE_persimmon_count_l27_2719

theorem persimmon_count (total : ℕ) (difference : ℕ) (persimmons : ℕ) (tangerines : ℕ) : 
  total = 129 →
  difference = 43 →
  total = persimmons + tangerines →
  persimmons + difference = tangerines →
  persimmons = 43 := by
sorry

end NUMINAMATH_CALUDE_persimmon_count_l27_2719


namespace NUMINAMATH_CALUDE_smallest_integer_above_sqrt_difference_power_l27_2772

theorem smallest_integer_above_sqrt_difference_power :
  ∃ n : ℤ, (n = 9737 ∧ ∀ m : ℤ, (m > (Real.sqrt 5 - Real.sqrt 3)^8 → m ≥ n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_above_sqrt_difference_power_l27_2772


namespace NUMINAMATH_CALUDE_a_mod_4_is_2_or_3_a_not_perfect_square_l27_2712

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a n * a (n + 1) + 1

theorem a_mod_4_is_2_or_3 (n : ℕ) (h : n ≥ 2) : 
  (a n) % 4 = 2 ∨ (a n) % 4 = 3 :=
by sorry

theorem a_not_perfect_square (n : ℕ) (h : n ≥ 2) : 
  ¬ ∃ (k : ℕ), a n = k * k :=
by sorry

end NUMINAMATH_CALUDE_a_mod_4_is_2_or_3_a_not_perfect_square_l27_2712


namespace NUMINAMATH_CALUDE_count_valid_primes_l27_2707

/-- Convert a number from base p to base 10 --/
def to_base_10 (digits : List Nat) (p : Nat) : Nat :=
  digits.foldr (fun d acc => d + p * acc) 0

/-- Check if the equation holds for a given prime p --/
def equation_holds (p : Nat) : Prop :=
  to_base_10 [9, 7, 6] p + to_base_10 [5, 0, 7] p + to_base_10 [2, 3, 8] p =
  to_base_10 [4, 2, 9] p + to_base_10 [5, 9, 5] p + to_base_10 [6, 9, 7] p

/-- The main theorem --/
theorem count_valid_primes :
  ∃ (S : Finset Nat), S.card = 3 ∧ 
  (∀ p ∈ S, Nat.Prime p ∧ p < 10 ∧ equation_holds p) ∧
  (∀ p, Nat.Prime p → p < 10 → equation_holds p → p ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_valid_primes_l27_2707


namespace NUMINAMATH_CALUDE_domain_of_sqrt_plus_fraction_l27_2795

theorem domain_of_sqrt_plus_fraction (x : ℝ) :
  (x + 3 ≥ 0 ∧ x + 2 ≠ 0) ↔ (x ≥ -3 ∧ x ≠ -2) := by sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_plus_fraction_l27_2795


namespace NUMINAMATH_CALUDE_consumer_installment_credit_l27_2799

theorem consumer_installment_credit (C : ℝ) 
  (h1 : C * 0.2 = C * 0.15 + 100 + 57 * 3)  -- Auto credit = Student loans + 100 + Auto finance
  (h2 : C * 0.15 = 80)                      -- Student loans
  (h3 : C * 0.25 = C * 0.2 + 100)           -- Credit cards = Auto credit + 100
  (h4 : C * 0.4 + C * 0.25 + C * 0.2 + C * 0.15 = C)  -- Total percentages sum to 100%
  : C = 1084 := by
  sorry

end NUMINAMATH_CALUDE_consumer_installment_credit_l27_2799


namespace NUMINAMATH_CALUDE_long_jump_comparison_l27_2798

/-- Proves that Margarita ran and jumped 1 foot farther than Ricciana -/
theorem long_jump_comparison (ricciana_total : ℕ) (ricciana_run : ℕ) (ricciana_jump : ℕ)
  (margarita_run : ℕ) :
  ricciana_total = 24 →
  ricciana_run = 20 →
  ricciana_jump = 4 →
  margarita_run = 18 →
  (margarita_run + (2 * ricciana_jump - 1)) - ricciana_total = 1 :=
by sorry

end NUMINAMATH_CALUDE_long_jump_comparison_l27_2798


namespace NUMINAMATH_CALUDE_train_crossing_time_l27_2774

/-- Proves that a train 400 meters long, traveling at 144 km/hr, will take 10 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 400 ∧ train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 10 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l27_2774


namespace NUMINAMATH_CALUDE_triangle_sine_triple_angle_sum_bounds_l27_2784

theorem triangle_sine_triple_angle_sum_bounds 
  (A B C : Real) 
  (h : A + B + C = Real.pi) : 
  -2 ≤ Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ∧ 
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ≤ 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_triple_angle_sum_bounds_l27_2784


namespace NUMINAMATH_CALUDE_game_tie_fraction_l27_2720

theorem game_tie_fraction (mark_wins jane_wins : ℚ) 
  (h1 : mark_wins = 5 / 12)
  (h2 : jane_wins = 1 / 4) : 
  1 - (mark_wins + jane_wins) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_game_tie_fraction_l27_2720


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_675_l27_2771

theorem sin_n_equals_cos_675 (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) :
  Real.sin (n * π / 180) = Real.cos (675 * π / 180) → n = 45 := by
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_675_l27_2771


namespace NUMINAMATH_CALUDE_recruit_count_l27_2721

theorem recruit_count (peter nikolai denis total : ℕ) : 
  peter = 50 →
  nikolai = 100 →
  denis = 170 →
  (total - peter - 1 = 4 * (total - denis - 1) ∨
   total - nikolai - 1 = 4 * (total - denis - 1) ∨
   total - peter - 1 = 4 * (total - nikolai - 1)) →
  total = 213 :=
by sorry

end NUMINAMATH_CALUDE_recruit_count_l27_2721


namespace NUMINAMATH_CALUDE_scenario_equivalence_l27_2797

/-- Represents the cost of trees in yuan -/
structure TreeCost where
  pine : ℝ
  cypress : ℝ

/-- Represents the given scenario for tree costs -/
def scenario (cost : TreeCost) : Prop :=
  2 * cost.pine + 3 * cost.cypress = 120 ∧
  2 * cost.pine - cost.cypress = 20

/-- The correct system of equations for the scenario -/
def correct_system (x y : ℝ) : Prop :=
  2 * x + 3 * y = 120 ∧
  2 * x - y = 20

/-- Theorem stating that the correct system accurately represents the scenario -/
theorem scenario_equivalence :
  ∀ (cost : TreeCost), scenario cost ↔ correct_system cost.pine cost.cypress :=
by sorry

end NUMINAMATH_CALUDE_scenario_equivalence_l27_2797
