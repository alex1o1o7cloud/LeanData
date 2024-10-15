import Mathlib

namespace NUMINAMATH_CALUDE_x_squared_range_l1682_168296

theorem x_squared_range (x : ℝ) : 
  (Real.rpow (x + 9) (1/3) - Real.rpow (x - 9) (1/3) = 3) → 
  75 < x^2 ∧ x^2 < 85 := by
sorry

end NUMINAMATH_CALUDE_x_squared_range_l1682_168296


namespace NUMINAMATH_CALUDE_square_plus_fourth_power_equality_l1682_168249

theorem square_plus_fourth_power_equality (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : n > 3) 
  (h4 : m^2 + n^4 = 2*(m - 6)^2 + 2*(n + 1)^2) : 
  m^2 + n^4 = 1994 := by
sorry

end NUMINAMATH_CALUDE_square_plus_fourth_power_equality_l1682_168249


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1682_168252

-- Define the hyperbola and its properties
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptotes and intersection points
def asymptote_intersections (h : Hyperbola) (x : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x = h.a^2 / h.c ∧ (y = h.b * x / h.a ∨ y = -h.b * x / h.a)}

-- Define the angle AFB
def angle_AFB (h : Hyperbola) (A B : ℝ × ℝ) : ℝ := sorry

-- Define the eccentricity
def eccentricity (h : Hyperbola) : ℝ := sorry

-- State the theorem
theorem hyperbola_eccentricity_range (h : Hyperbola) 
  (A B : ℝ × ℝ) (hA : A ∈ asymptote_intersections h (h.a^2 / h.c)) 
  (hB : B ∈ asymptote_intersections h (h.a^2 / h.c))
  (hAngle : π/3 < angle_AFB h A B ∧ angle_AFB h A B < π/2) :
  Real.sqrt 2 < eccentricity h ∧ eccentricity h < 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1682_168252


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1682_168294

def U : Set ℕ := {1, 3, 5, 7}
def A : Set ℕ := {3, 5}
def B : Set ℕ := {1, 3, 7}

theorem intersection_with_complement : A ∩ (U \ B) = {5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1682_168294


namespace NUMINAMATH_CALUDE_smallest_m_fibonacci_l1682_168238

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def representable (F : ℕ → ℕ) (x : List ℕ) (n : ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.Sublist x ∧ subset.sum = F n

theorem smallest_m_fibonacci :
  ∃ (m : ℕ) (x : List ℕ),
    (∀ i, i ∈ x → i > 0) ∧
    (∀ n, n ≤ 2018 → representable fibonacci x n) ∧
    (∀ m' < m, ¬∃ x' : List ℕ,
      (∀ i, i ∈ x' → i > 0) ∧
      (∀ n, n ≤ 2018 → representable fibonacci x' n)) ∧
    m = 1009 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_fibonacci_l1682_168238


namespace NUMINAMATH_CALUDE_equation_solution_l1682_168206

theorem equation_solution : ∃! x : ℚ, 
  (1 : ℚ) / ((x + 12)^2) + (1 : ℚ) / ((x + 8)^2) = 
  (1 : ℚ) / ((x + 13)^2) + (1 : ℚ) / ((x + 7)^2) ∧ 
  x = -15/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1682_168206


namespace NUMINAMATH_CALUDE_fuel_consumption_rate_l1682_168225

/-- Given a plane with a certain amount of fuel and remaining flight time,
    calculate the rate of fuel consumption per hour. -/
theorem fuel_consumption_rate (fuel_left : ℝ) (time_left : ℝ) :
  fuel_left = 6.3333 →
  time_left = 0.6667 →
  ∃ (rate : ℝ), abs (rate - (fuel_left / time_left)) < 0.01 ∧ abs (rate - 9.5) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_fuel_consumption_rate_l1682_168225


namespace NUMINAMATH_CALUDE_carolyn_practice_time_l1682_168231

/-- Calculates the total practice time in minutes for a month given daily practice times and schedule -/
def monthly_practice_time (piano_time : ℕ) (violin_multiplier : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let violin_time := piano_time * violin_multiplier
  let daily_total := piano_time + violin_time
  let weekly_total := daily_total * days_per_week
  weekly_total * weeks_per_month

/-- Proves that Carolyn's monthly practice time is 1920 minutes -/
theorem carolyn_practice_time :
  monthly_practice_time 20 3 6 4 = 1920 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_practice_time_l1682_168231


namespace NUMINAMATH_CALUDE_sqrt_identity_l1682_168245

theorem sqrt_identity : (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_l1682_168245


namespace NUMINAMATH_CALUDE_law_firm_associates_tenure_l1682_168274

theorem law_firm_associates_tenure (total : ℝ) (first_year : ℝ) (second_year : ℝ) (more_than_two_years : ℝ)
  (h1 : second_year / total = 0.3)
  (h2 : (total - first_year) / total = 0.6) :
  more_than_two_years / total = 0.6 - 0.3 := by
sorry

end NUMINAMATH_CALUDE_law_firm_associates_tenure_l1682_168274


namespace NUMINAMATH_CALUDE_trig_identity_quadratic_equation_solution_l1682_168235

-- Problem 1
theorem trig_identity : 
  Real.sin (π / 4) - 3 * Real.tan (π / 6) + Real.sqrt 2 * Real.cos (π / 3) = Real.sqrt 2 - Real.sqrt 3 := by
  sorry

-- Problem 2
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x + 8
  ∀ x : ℝ, f x = 0 ↔ x = 4 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_quadratic_equation_solution_l1682_168235


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l1682_168219

theorem distance_from_origin_to_point :
  let x : ℝ := 8
  let y : ℝ := -15
  (x^2 + y^2).sqrt = 17 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l1682_168219


namespace NUMINAMATH_CALUDE_river_road_cars_l1682_168291

theorem river_road_cars (B C : ℕ) 
  (h1 : B * 13 = C)  -- ratio of buses to cars is 1:13
  (h2 : B = C - 60)  -- there are 60 fewer buses than cars
  : C = 65 := by sorry

end NUMINAMATH_CALUDE_river_road_cars_l1682_168291


namespace NUMINAMATH_CALUDE_unique_intersection_point_l1682_168268

/-- The function g(x) = x^3 + 5x^2 + 12x + 20 -/
def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 12*x + 20

/-- Theorem: The unique intersection point of g(x) and its inverse is (-4, -4) -/
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-4, -4) := by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l1682_168268


namespace NUMINAMATH_CALUDE_sqrt_product_difference_of_squares_l1682_168279

-- Problem 1
theorem sqrt_product : Real.sqrt 2 * Real.sqrt 5 = Real.sqrt 10 := by sorry

-- Problem 2
theorem difference_of_squares : (3 + Real.sqrt 6) * (3 - Real.sqrt 6) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_difference_of_squares_l1682_168279


namespace NUMINAMATH_CALUDE_triangle_radii_relations_l1682_168253

/-- Given a triangle ABC with sides a, b, c, inradius r, exradii r_a, r_b, r_c, semi-perimeter p, and area S -/
theorem triangle_radii_relations (a b c r r_a r_b r_c p S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ p > 0 ∧ S > 0)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_area_inradius : S = p * r)
  (h_area_exradius_a : S = (p - a) * r_a)
  (h_area_exradius_b : S = (p - b) * r_b)
  (h_area_exradius_c : S = (p - c) * r_c) :
  (1 / r = 1 / r_a + 1 / r_b + 1 / r_c) ∧ 
  (S = Real.sqrt (r * r_a * r_b * r_c)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_radii_relations_l1682_168253


namespace NUMINAMATH_CALUDE_b_finishes_in_two_days_l1682_168202

/-- The number of days A takes to finish the work alone -/
def a_days : ℚ := 4

/-- The number of days B takes to finish the work alone -/
def b_days : ℚ := 8

/-- The number of days A and B work together before A leaves -/
def days_together : ℚ := 2

/-- The fraction of work completed per day when A and B work together -/
def combined_work_rate : ℚ := 1 / a_days + 1 / b_days

/-- The fraction of work completed when A and B work together for 2 days -/
def work_completed_together : ℚ := days_together * combined_work_rate

/-- The fraction of work remaining after A leaves -/
def remaining_work : ℚ := 1 - work_completed_together

/-- The number of days B takes to finish the remaining work alone -/
def days_for_b_to_finish : ℚ := remaining_work / (1 / b_days)

theorem b_finishes_in_two_days : days_for_b_to_finish = 2 := by
  sorry

end NUMINAMATH_CALUDE_b_finishes_in_two_days_l1682_168202


namespace NUMINAMATH_CALUDE_no_primes_in_factorial_range_l1682_168248

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 2) :
  ∀ k : ℕ, n! + 2 < k ∧ k < n! + n + 1 → ¬ Nat.Prime k :=
by sorry

end NUMINAMATH_CALUDE_no_primes_in_factorial_range_l1682_168248


namespace NUMINAMATH_CALUDE_expansion_has_four_terms_l1682_168265

/-- The expression after substituting 2x for the asterisk and expanding -/
def expanded_expression (x : ℝ) : ℝ := x^6 + x^4 + 4*x^2 + 4

/-- The original expression with the asterisk replaced by 2x -/
def original_expression (x : ℝ) : ℝ := (x^3 - 2)^2 + (x^2 + 2*x)^2

theorem expansion_has_four_terms :
  ∀ x : ℝ, original_expression x = expanded_expression x ∧ 
  (∃ a b c d : ℝ, expanded_expression x = a*x^6 + b*x^4 + c*x^2 + d ∧ 
   a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_expansion_has_four_terms_l1682_168265


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1682_168281

/-- 
For a polynomial of the form x^2 - 18x + k to be a perfect square binomial,
k must equal 81.
-/
theorem perfect_square_condition (k : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - 18*x + k = (x + a)^2 + b) ↔ k = 81 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1682_168281


namespace NUMINAMATH_CALUDE_sin_240_degrees_l1682_168259

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l1682_168259


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l1682_168288

theorem simplify_sqrt_difference : 
  (Real.sqrt 704 / Real.sqrt 64) - (Real.sqrt 300 / Real.sqrt 75) = Real.sqrt 11 - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l1682_168288


namespace NUMINAMATH_CALUDE_lcm_144_132_l1682_168256

theorem lcm_144_132 : lcm 144 132 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_lcm_144_132_l1682_168256


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1682_168280

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {2, 3, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1682_168280


namespace NUMINAMATH_CALUDE_min_value_expression_l1682_168210

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1682_168210


namespace NUMINAMATH_CALUDE_sum_evaluation_l1682_168211

theorem sum_evaluation : 
  (4 : ℚ) / 3 + 7 / 6 + 13 / 12 + 25 / 24 + 49 / 48 + 97 / 96 - 8 = -43 / 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_evaluation_l1682_168211


namespace NUMINAMATH_CALUDE_nuts_mixed_with_raisins_l1682_168243

/-- The number of pounds of nuts mixed with raisins -/
def pounds_of_nuts : ℝ := 4

/-- The number of pounds of raisins -/
def pounds_of_raisins : ℝ := 5

/-- The cost ratio of nuts to raisins -/
def cost_ratio : ℝ := 3

/-- The fraction of the total cost that the raisins represent -/
def raisin_cost_fraction : ℝ := 0.29411764705882354

/-- Proves that the number of pounds of nuts mixed with 5 pounds of raisins is 4 -/
theorem nuts_mixed_with_raisins :
  let r := 1  -- Cost of 1 pound of raisins (arbitrary unit)
  let n := cost_ratio * r  -- Cost of 1 pound of nuts
  pounds_of_nuts * n / (pounds_of_nuts * n + pounds_of_raisins * r) = 1 - raisin_cost_fraction :=
by sorry

end NUMINAMATH_CALUDE_nuts_mixed_with_raisins_l1682_168243


namespace NUMINAMATH_CALUDE_factor_sum_l1682_168270

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 2*X + 5) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 31 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l1682_168270


namespace NUMINAMATH_CALUDE_backpack_profit_equation_l1682_168224

/-- Represents the profit calculation for a backpack sale -/
theorem backpack_profit_equation (x : ℝ) : 
  (1 + 0.5) * x * 0.8 - x = 8 ↔ 
  (x > 0 ∧ 
   (1 + 0.5) * x * 0.8 = x + 8) :=
by sorry

#check backpack_profit_equation

end NUMINAMATH_CALUDE_backpack_profit_equation_l1682_168224


namespace NUMINAMATH_CALUDE_club_leadership_selection_l1682_168216

def total_members : ℕ := 24
def num_boys : ℕ := 14
def num_girls : ℕ := 10

theorem club_leadership_selection :
  (num_boys * (num_boys - 1) + num_girls * (num_girls - 1) : ℕ) = 272 :=
by sorry

end NUMINAMATH_CALUDE_club_leadership_selection_l1682_168216


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l1682_168220

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b :=
by sorry

theorem sqrt_three_times_sqrt_two_equals_sqrt_six : 
  Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l1682_168220


namespace NUMINAMATH_CALUDE_borrowing_schemes_l1682_168285

theorem borrowing_schemes (n : ℕ) (m : ℕ) :
  n = 5 →  -- number of students
  m = 4 →  -- number of novels
  (∃ (schemes : ℕ), schemes = 60) :=
by
  intros hn hm
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_borrowing_schemes_l1682_168285


namespace NUMINAMATH_CALUDE_harriet_miles_run_l1682_168212

/-- Proves that given four runners who ran a combined total of 195 miles,
    with one runner running 51 miles and the other three runners running equal distances,
    each of the other three runners ran 48 miles. -/
theorem harriet_miles_run (total_miles : ℕ) (katarina_miles : ℕ) (other_runners : ℕ) :
  total_miles = 195 →
  katarina_miles = 51 →
  other_runners = 3 →
  ∃ (harriet_miles : ℕ),
    harriet_miles * other_runners = total_miles - katarina_miles ∧
    harriet_miles = 48 := by
  sorry

end NUMINAMATH_CALUDE_harriet_miles_run_l1682_168212


namespace NUMINAMATH_CALUDE_inequality_solution_l1682_168293

theorem inequality_solution (x : ℝ) : x / (x^2 + 3*x + 2) ≥ 0 ↔ x ∈ Set.Ioo (-2) (-1) ∪ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1682_168293


namespace NUMINAMATH_CALUDE_display_board_sides_l1682_168250

/-- A polygonal display board with given perimeter and side ribbon length has a specific number of sides. -/
theorem display_board_sides (perimeter : ℝ) (side_ribbon_length : ℝ) (num_sides : ℕ) : 
  perimeter = 42 → side_ribbon_length = 7 → num_sides * side_ribbon_length = perimeter → num_sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_display_board_sides_l1682_168250


namespace NUMINAMATH_CALUDE_candidate_B_votes_l1682_168298

/-- Represents a candidate in the election --/
inductive Candidate
  | A | B | C | D | E

/-- The total number of people in the class --/
def totalVotes : Nat := 46

/-- The number of votes received by candidate A --/
def votesForA : Nat := 25

/-- The number of votes received by candidate E --/
def votesForE : Nat := 4

/-- The voting results satisfy the given conditions --/
def validVotingResult (votes : Candidate → Nat) : Prop :=
  votes Candidate.A = votesForA ∧
  votes Candidate.E = votesForE ∧
  votes Candidate.B > votes Candidate.E ∧
  votes Candidate.B < votes Candidate.A ∧
  votes Candidate.C = votes Candidate.D ∧
  votes Candidate.A + votes Candidate.B + votes Candidate.C + votes Candidate.D + votes Candidate.E = totalVotes

theorem candidate_B_votes (votes : Candidate → Nat) 
  (h : validVotingResult votes) : votes Candidate.B = 7 := by
  sorry

end NUMINAMATH_CALUDE_candidate_B_votes_l1682_168298


namespace NUMINAMATH_CALUDE_arrangement_count_is_2880_l1682_168254

/-- The number of ways to arrange 4 boys and 3 girls in a row with constraints -/
def arrangementCount : ℕ :=
  let numBoys : ℕ := 4
  let numGirls : ℕ := 3
  let waysToChooseTwoGirls : ℕ := Nat.choose numGirls 2
  let waysToArrangeBoys : ℕ := Nat.factorial numBoys
  let spacesForGirlUnits : ℕ := numBoys + 1
  let waysToInsertGirlUnits : ℕ := Nat.descFactorial spacesForGirlUnits 2
  waysToChooseTwoGirls * waysToArrangeBoys * waysToInsertGirlUnits

theorem arrangement_count_is_2880 : arrangementCount = 2880 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_2880_l1682_168254


namespace NUMINAMATH_CALUDE_factory_production_time_l1682_168221

/-- The number of dolls produced by the factory -/
def num_dolls : ℕ := 12000

/-- The number of shoes per doll -/
def shoes_per_doll : ℕ := 2

/-- The number of bags per doll -/
def bags_per_doll : ℕ := 3

/-- The number of cosmetics sets per doll -/
def cosmetics_per_doll : ℕ := 1

/-- The number of hats per doll -/
def hats_per_doll : ℕ := 5

/-- The time in seconds to make one doll -/
def time_per_doll : ℕ := 45

/-- The time in seconds to make one accessory -/
def time_per_accessory : ℕ := 10

/-- The total combined machine operation time for manufacturing all dolls and accessories -/
def total_time : ℕ := 1860000

theorem factory_production_time : 
  num_dolls * time_per_doll + 
  num_dolls * (shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll) * time_per_accessory = 
  total_time := by sorry

end NUMINAMATH_CALUDE_factory_production_time_l1682_168221


namespace NUMINAMATH_CALUDE_binary_110101_is_53_l1682_168286

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101 -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

theorem binary_110101_is_53 : binary_to_decimal binary_110101 = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_is_53_l1682_168286


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_implies_a_range_l1682_168292

theorem tangent_line_y_intercept_implies_a_range (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.exp x + a * x^2
  ∀ m : ℝ, m > 1 →
    let f' : ℝ → ℝ := λ x ↦ Real.exp x + 2 * a * x
    let tangent_slope : ℝ := f' m
    let tangent_y_intercept : ℝ := f m - tangent_slope * m
    tangent_y_intercept < 1 →
    a ∈ Set.Ici (-1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_implies_a_range_l1682_168292


namespace NUMINAMATH_CALUDE_min_unboxed_balls_l1682_168207

/-- Represents the number of balls that can be stored in a big box -/
def big_box_capacity : ℕ := 25

/-- Represents the number of balls that can be stored in a small box -/
def small_box_capacity : ℕ := 20

/-- Represents the total number of balls to be stored -/
def total_balls : ℕ := 104

/-- 
Given:
- Big boxes can store 25 balls each
- Small boxes can store 20 balls each
- There are 104 balls to be stored

Prove that the minimum number of balls that cannot be completely boxed is 4.
-/
theorem min_unboxed_balls : 
  ∀ (big_boxes small_boxes : ℕ), 
    big_boxes * big_box_capacity + small_boxes * small_box_capacity ≤ total_balls →
    4 ≤ total_balls - (big_boxes * big_box_capacity + small_boxes * small_box_capacity) :=
by sorry

end NUMINAMATH_CALUDE_min_unboxed_balls_l1682_168207


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_exists_l1682_168244

theorem min_value_theorem (x : ℝ) (h : x > 0) : x^2 + 12*x + 108/x^4 ≥ 42 := by
  sorry

theorem equality_exists : ∃ x : ℝ, x > 0 ∧ x^2 + 12*x + 108/x^4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_exists_l1682_168244


namespace NUMINAMATH_CALUDE_jared_yearly_income_l1682_168215

/-- Calculates the yearly income of a degree holder after one year of employment --/
def yearly_income_after_one_year (diploma_salary : ℝ) : ℝ :=
  let degree_salary := 3 * diploma_salary
  let annual_salary := 12 * degree_salary
  let salary_after_raise := annual_salary * 1.05
  salary_after_raise * 1.05

/-- Theorem stating that Jared's yearly income after one year is $158760 --/
theorem jared_yearly_income :
  yearly_income_after_one_year 4000 = 158760 := by
  sorry

#eval yearly_income_after_one_year 4000

end NUMINAMATH_CALUDE_jared_yearly_income_l1682_168215


namespace NUMINAMATH_CALUDE_inequality_range_l1682_168228

theorem inequality_range (a : ℝ) : (∀ x : ℝ, a < |x - 4| + |x + 3|) → a < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1682_168228


namespace NUMINAMATH_CALUDE_component_is_unqualified_l1682_168276

-- Define the nominal diameter and tolerance
def nominal_diameter : ℝ := 20
def tolerance : ℝ := 0.02

-- Define the measured diameter
def measured_diameter : ℝ := 19.9

-- Define what it means for a component to be qualified
def is_qualified (d : ℝ) : Prop :=
  nominal_diameter - tolerance ≤ d ∧ d ≤ nominal_diameter + tolerance

-- Theorem stating that the component is unqualified
theorem component_is_unqualified : ¬ is_qualified measured_diameter := by
  sorry

end NUMINAMATH_CALUDE_component_is_unqualified_l1682_168276


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l1682_168284

/-- Proves that the cost of each adult ticket is $4.50 -/
theorem adult_ticket_cost :
  let student_ticket_price : ℚ := 2
  let total_tickets : ℕ := 20
  let total_income : ℚ := 60
  let student_tickets_sold : ℕ := 12
  let adult_tickets_sold : ℕ := total_tickets - student_tickets_sold
  let adult_ticket_price : ℚ := (total_income - (student_ticket_price * student_tickets_sold)) / adult_tickets_sold
  adult_ticket_price = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l1682_168284


namespace NUMINAMATH_CALUDE_uncovered_area_of_shoebox_l1682_168227

/-- Uncovered area of a shoebox with a square block inside -/
theorem uncovered_area_of_shoebox (shoebox_length shoebox_width block_side : ℕ) 
  (h1 : shoebox_length = 6)
  (h2 : shoebox_width = 4)
  (h3 : block_side = 4) :
  shoebox_length * shoebox_width - block_side * block_side = 8 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_of_shoebox_l1682_168227


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1682_168232

/-- Given two hyperbolas with equations x^2/16 - y^2/25 = 1 and y^2/50 - x^2/M = 1,
    if they have the same asymptotes, then M = 32 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/16 - y^2/25 = 1 ↔ y^2/50 - x^2/M = 1) →
  M = 32 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1682_168232


namespace NUMINAMATH_CALUDE_power_expression_evaluation_l1682_168208

theorem power_expression_evaluation (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_evaluation_l1682_168208


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1682_168236

theorem p_sufficient_not_necessary_for_q 
  (h1 : p → q) 
  (h2 : ¬(¬p → ¬q)) : 
  (∃ (x : Prop), x → q) ∧ (∃ (y : Prop), q ∧ ¬y) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1682_168236


namespace NUMINAMATH_CALUDE_monday_kids_count_l1682_168218

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 18

/-- The total number of kids Julia played with on Monday and Tuesday combined -/
def monday_tuesday_total : ℕ := 33

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := monday_tuesday_total - tuesday_kids

theorem monday_kids_count : monday_kids = 15 := by
  sorry

end NUMINAMATH_CALUDE_monday_kids_count_l1682_168218


namespace NUMINAMATH_CALUDE_area_ratio_is_three_fourths_l1682_168260

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Points on the sides of the octagon -/
structure OctagonPoints (oct : RegularOctagon) where
  I : ℝ × ℝ
  J : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  on_sides : sorry
  equally_spaced : sorry

/-- The ratio of areas of the inner octagon to the outer octagon -/
def area_ratio (oct : RegularOctagon) (pts : OctagonPoints oct) : ℝ := sorry

/-- The main theorem -/
theorem area_ratio_is_three_fourths (oct : RegularOctagon) (pts : OctagonPoints oct) :
  area_ratio oct pts = 3/4 := by sorry

end NUMINAMATH_CALUDE_area_ratio_is_three_fourths_l1682_168260


namespace NUMINAMATH_CALUDE_scientific_notation_4212000_l1682_168257

theorem scientific_notation_4212000 :
  4212000 = 4.212 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_4212000_l1682_168257


namespace NUMINAMATH_CALUDE_button_remainder_l1682_168272

theorem button_remainder (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 3)
  (h4 : n % 5 = 3) : 
  n % 12 = 7 := by sorry

end NUMINAMATH_CALUDE_button_remainder_l1682_168272


namespace NUMINAMATH_CALUDE_consecutive_sum_2016_l1682_168283

def is_valid_n (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a : ℕ, n * (2 * a + n - 1) = 4032

theorem consecutive_sum_2016 :
  {n : ℕ | is_valid_n n} = {3, 7, 9, 21, 63} :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_2016_l1682_168283


namespace NUMINAMATH_CALUDE_six_point_five_minutes_in_seconds_l1682_168213

/-- Converts minutes to seconds -/
def minutes_to_seconds (minutes : ℝ) : ℝ := minutes * 60

/-- Theorem stating that 6.5 minutes equals 390 seconds -/
theorem six_point_five_minutes_in_seconds : 
  minutes_to_seconds 6.5 = 390 := by sorry

end NUMINAMATH_CALUDE_six_point_five_minutes_in_seconds_l1682_168213


namespace NUMINAMATH_CALUDE_total_amount_spent_l1682_168271

/-- Calculates the total amount spent on a meal given the base food price, sales tax rate, and tip rate. -/
theorem total_amount_spent
  (food_price : ℝ)
  (sales_tax_rate : ℝ)
  (tip_rate : ℝ)
  (h1 : food_price = 150)
  (h2 : sales_tax_rate = 0.1)
  (h3 : tip_rate = 0.2) :
  food_price * (1 + sales_tax_rate) * (1 + tip_rate) = 198 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_spent_l1682_168271


namespace NUMINAMATH_CALUDE_route_count_is_70_l1682_168262

-- Define the grid structure
structure Grid :=
  (levels : Nat)
  (segments_between_levels : List Nat)

-- Define a route
def Route := List (Nat × Nat)

-- Function to check if a route is valid (doesn't intersect itself)
def is_valid_route (g : Grid) (r : Route) : Bool := sorry

-- Function to generate all possible routes
def all_routes (g : Grid) : List Route := sorry

-- Function to count valid routes
def count_valid_routes (g : Grid) : Nat :=
  (all_routes g).filter (is_valid_route g) |>.length

-- Define our specific grid
def our_grid : Grid :=
  { levels := 4,
    segments_between_levels := [3, 5, 3] }

-- Theorem statement
theorem route_count_is_70 :
  count_valid_routes our_grid = 70 := by sorry

end NUMINAMATH_CALUDE_route_count_is_70_l1682_168262


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1682_168223

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x - m > 0 ∧ 2*x + 1 > 3) ↔ x > 1) → m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1682_168223


namespace NUMINAMATH_CALUDE_cherry_pie_angle_l1682_168201

theorem cherry_pie_angle (total_students : ℕ) (chocolate_pref : ℕ) (apple_pref : ℕ) (blueberry_pref : ℕ) :
  total_students = 45 →
  chocolate_pref = 15 →
  apple_pref = 10 →
  blueberry_pref = 9 →
  let remaining := total_students - (chocolate_pref + apple_pref + blueberry_pref)
  let cherry_pref := remaining / 2
  let lemon_pref := remaining / 3
  let pecan_pref := remaining - cherry_pref - lemon_pref
  (cherry_pref : ℚ) / total_students * 360 = 40 :=
by sorry

end NUMINAMATH_CALUDE_cherry_pie_angle_l1682_168201


namespace NUMINAMATH_CALUDE_prob_A_win_match_is_correct_l1682_168275

/-- The probability of player A winning a single game -/
def prob_A_win : ℝ := 0.6

/-- The probability of player B winning a single game -/
def prob_B_win : ℝ := 0.4

/-- The probability of player A winning the match after winning the first game -/
def prob_A_win_match : ℝ := prob_A_win + prob_B_win * prob_A_win

/-- Theorem stating that the probability of A winning the match after winning the first game is 0.84 -/
theorem prob_A_win_match_is_correct : prob_A_win_match = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_win_match_is_correct_l1682_168275


namespace NUMINAMATH_CALUDE_range_of_a_l1682_168251

/-- The function f(x) = ax^2 - 2x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*x + 1

/-- f is decreasing on [1, +∞) -/
def f_decreasing (a : ℝ) : Prop := 
  ∀ x y, 1 ≤ x → x < y → f a y < f a x

/-- The range of a is (-∞, 0] -/
theorem range_of_a : 
  (∃ a, f_decreasing a) ↔ (∀ a, f_decreasing a → a ≤ 0) ∧ (∃ a ≤ 0, f_decreasing a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1682_168251


namespace NUMINAMATH_CALUDE_pencil_cost_l1682_168273

/-- Proves that the cost of each pencil Cindi bought is $0.50 --/
theorem pencil_cost (cindi_pencils : ℕ) (marcia_pencils : ℕ) (donna_pencils : ℕ) 
  (h1 : marcia_pencils = 2 * cindi_pencils)
  (h2 : donna_pencils = 3 * marcia_pencils)
  (h3 : donna_pencils + marcia_pencils = 480)
  (h4 : cindi_pencils * (cost_per_pencil : ℚ) = 30) : 
  cost_per_pencil = 1/2 := by
  sorry

#check pencil_cost

end NUMINAMATH_CALUDE_pencil_cost_l1682_168273


namespace NUMINAMATH_CALUDE_rhombus_rectangle_diagonals_bisect_l1682_168229

-- Define a quadrilateral
class Quadrilateral :=
  (diagonals_bisect : Bool)
  (diagonals_perpendicular : Bool)
  (diagonals_equal : Bool)

-- Define a rhombus
def Rhombus : Quadrilateral :=
{ diagonals_bisect := true,
  diagonals_perpendicular := true,
  diagonals_equal := false }

-- Define a rectangle
def Rectangle : Quadrilateral :=
{ diagonals_bisect := true,
  diagonals_perpendicular := false,
  diagonals_equal := true }

-- Theorem: Both rhombuses and rectangles have diagonals that bisect each other
theorem rhombus_rectangle_diagonals_bisect :
  Rhombus.diagonals_bisect ∧ Rectangle.diagonals_bisect :=
sorry

end NUMINAMATH_CALUDE_rhombus_rectangle_diagonals_bisect_l1682_168229


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l1682_168241

theorem cylinder_radius_problem (r : ℝ) :
  (r > 0) →
  (5 * π * (r + 4)^2 = 15 * π * r^2) →
  (r = 2 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l1682_168241


namespace NUMINAMATH_CALUDE_min_fruits_problem_l1682_168247

theorem min_fruits_problem : ∃ n : ℕ, n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 6 = 5 ∧ 
  (∀ m : ℕ, m > 0 → m % 3 = 2 → m % 4 = 3 → m % 5 = 4 → m % 6 = 5 → m ≥ n) ∧
  n = 59 := by
sorry

end NUMINAMATH_CALUDE_min_fruits_problem_l1682_168247


namespace NUMINAMATH_CALUDE_hannah_fair_money_l1682_168209

theorem hannah_fair_money (initial_money : ℝ) : 
  (initial_money / 2 + 5 + 10 = initial_money) → initial_money = 30 := by
  sorry

end NUMINAMATH_CALUDE_hannah_fair_money_l1682_168209


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1682_168290

theorem absolute_value_equation (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) ↔ 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 0 ∧ c = 1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = -1)) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1682_168290


namespace NUMINAMATH_CALUDE_spheres_radius_is_half_l1682_168222

/-- A cube with side length 2 containing eight congruent spheres --/
structure SpheresInCube where
  -- The side length of the cube
  cube_side : ℝ
  -- The radius of each sphere
  sphere_radius : ℝ
  -- The number of spheres
  num_spheres : ℕ
  -- Condition that the cube side length is 2
  cube_side_is_two : cube_side = 2
  -- Condition that there are 8 spheres
  eight_spheres : num_spheres = 8
  -- Condition that spheres are tangent to three faces and neighboring spheres
  spheres_tangent : True  -- This is a simplification, as we can't easily express this geometric condition

/-- Theorem stating that the radius of each sphere is 1/2 --/
theorem spheres_radius_is_half (s : SpheresInCube) : s.sphere_radius = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_spheres_radius_is_half_l1682_168222


namespace NUMINAMATH_CALUDE_intersection_M_N_l1682_168242

def M : Set ℝ := {x | x^2 + x - 2 = 0}
def N : Set ℝ := {x | x < 0}

theorem intersection_M_N : M ∩ N = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1682_168242


namespace NUMINAMATH_CALUDE_books_in_year_l1682_168264

/-- The number of books Jack can read in a day -/
def books_per_day : ℕ := 9

/-- The number of days in a year -/
def days_in_year : ℕ := 365

/-- Theorem: Jack can read 3285 books in a year -/
theorem books_in_year : books_per_day * days_in_year = 3285 := by
  sorry

end NUMINAMATH_CALUDE_books_in_year_l1682_168264


namespace NUMINAMATH_CALUDE_inequality_proof_l1682_168263

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : 1/b - 1/a > 1) :
  Real.sqrt (1 + a) > 1 / Real.sqrt (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1682_168263


namespace NUMINAMATH_CALUDE_concatenation_product_relation_l1682_168217

theorem concatenation_product_relation :
  ∃! (x y : ℕ), 10 ≤ x ∧ x ≤ 99 ∧ 100 ≤ y ∧ y ≤ 999 ∧ 
  1000 * x + y = 11 * x * y ∧ x + y = 110 := by
sorry

end NUMINAMATH_CALUDE_concatenation_product_relation_l1682_168217


namespace NUMINAMATH_CALUDE_max_grid_size_is_five_l1682_168237

/-- A coloring of an n × n grid using two colors. -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- Predicate to check if a rectangle in the grid has all corners of the same color. -/
def hasMonochromaticRectangle (c : Coloring n) : Prop :=
  ∃ (i j k l : Fin n), i < k ∧ j < l ∧
    c i j = c i l ∧ c i l = c k j ∧ c k j = c k l

/-- The maximum size of a grid that can be colored without monochromatic rectangles. -/
def maxGridSize : ℕ := 5

/-- Theorem stating that 5 is the maximum size of a grid that can be colored
    with two colors such that no rectangle has all four corners the same color. -/
theorem max_grid_size_is_five :
  (∀ n : ℕ, n ≤ maxGridSize →
    ∃ c : Coloring n, ¬hasMonochromaticRectangle c) ∧
  (∀ n : ℕ, n > maxGridSize →
    ∀ c : Coloring n, hasMonochromaticRectangle c) :=
sorry

end NUMINAMATH_CALUDE_max_grid_size_is_five_l1682_168237


namespace NUMINAMATH_CALUDE_even_odd_solution_l1682_168214

theorem even_odd_solution (m n p q : ℤ) 
  (h_m_odd : Odd m)
  (h_n_even : Even n)
  (h_eq1 : p - 1998*q = n)
  (h_eq2 : 1999*p + 3*q = m) :
  Even p ∧ Odd q := by
sorry

end NUMINAMATH_CALUDE_even_odd_solution_l1682_168214


namespace NUMINAMATH_CALUDE_profit_share_difference_is_1000_l1682_168240

/-- Represents the profit share calculation for a business partnership --/
structure BusinessPartnership where
  investment_a : ℕ
  investment_b : ℕ
  investment_c : ℕ
  profit_share_b : ℕ

/-- Calculates the difference between profit shares of partners C and A --/
def profit_share_difference (bp : BusinessPartnership) : ℕ :=
  let total_investment := bp.investment_a + bp.investment_b + bp.investment_c
  let total_profit := bp.profit_share_b * total_investment / bp.investment_b
  let share_a := total_profit * bp.investment_a / total_investment
  let share_c := total_profit * bp.investment_c / total_investment
  share_c - share_a

/-- Theorem stating that for the given investments and B's profit share, 
    the difference between C's and A's profit shares is 1000 --/
theorem profit_share_difference_is_1000 :
  profit_share_difference ⟨8000, 10000, 12000, 2500⟩ = 1000 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_is_1000_l1682_168240


namespace NUMINAMATH_CALUDE_quadratic_and_related_function_properties_l1682_168287

/-- Given a quadratic function f and its derivative, prove properties about its coefficients and a related function g --/
theorem quadratic_and_related_function_properties
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h₁ : ∀ x, f x = a * x^2 + b * x + 3)
  (h₂ : a ≠ 0)
  (h₃ : ∀ x, deriv f x = 2 * x - 8)
  (g : ℝ → ℝ)
  (h₄ : ∀ x, g x = Real.exp x * Real.sin x + f x) :
  a = 1 ∧ b = -8 ∧
  (∃ m c : ℝ, m = 7 ∧ c = -3 ∧ ∀ x y, y = deriv g 0 * (x - 0) + g 0 ↔ m * x + y + c = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_and_related_function_properties_l1682_168287


namespace NUMINAMATH_CALUDE_cubic_equation_solution_range_l1682_168200

theorem cubic_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^3 - 3*x + m = 0) → m ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_range_l1682_168200


namespace NUMINAMATH_CALUDE_no_digit_product_6552_l1682_168230

theorem no_digit_product_6552 : ¬ ∃ (s : List ℕ), (∀ n ∈ s, n ≤ 9) ∧ s.prod = 6552 := by
  sorry

end NUMINAMATH_CALUDE_no_digit_product_6552_l1682_168230


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l1682_168289

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m > 120 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ k : ℤ, k ≤ 120 → (k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l1682_168289


namespace NUMINAMATH_CALUDE_joel_laps_l1682_168261

/-- Given that Yvonne swims 10 laps in 5 minutes, her younger sister swims half as many laps,
    and Joel swims three times as many laps as the younger sister,
    prove that Joel swims 15 laps in 5 minutes. -/
theorem joel_laps (yvonne_laps : ℕ) (younger_sister_ratio : ℚ) (joel_ratio : ℕ) :
  yvonne_laps = 10 →
  younger_sister_ratio = 1 / 2 →
  joel_ratio = 3 →
  (yvonne_laps : ℚ) * younger_sister_ratio * joel_ratio = 15 := by
  sorry

end NUMINAMATH_CALUDE_joel_laps_l1682_168261


namespace NUMINAMATH_CALUDE_no_triple_exists_l1682_168282

theorem no_triple_exists : ¬∃ (a b c : ℕ+), 
  let p := (a.val - 2) * (b.val - 2) * (c.val - 2) + 12
  Nat.Prime p ∧ 
  ∃ (k : ℕ+), k * p = a.val^2 + b.val^2 + c.val^2 + a.val * b.val * c.val - 2017 := by
  sorry

end NUMINAMATH_CALUDE_no_triple_exists_l1682_168282


namespace NUMINAMATH_CALUDE_juice_reduction_fraction_l1682_168269

/-- Proves that the fraction of the original volume that the juice was reduced to is 1/12 --/
theorem juice_reduction_fraction (original_volume : ℚ) (quart_to_cup : ℚ) (sugar_added : ℚ) (final_volume : ℚ) :
  original_volume = 6 →
  quart_to_cup = 4 →
  sugar_added = 1 →
  final_volume = 3 →
  (final_volume - sugar_added) / (original_volume * quart_to_cup) = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_juice_reduction_fraction_l1682_168269


namespace NUMINAMATH_CALUDE_train_length_l1682_168267

theorem train_length (tree_time platform_time platform_length : ℝ) :
  tree_time = 120 →
  platform_time = 230 →
  platform_length = 1100 →
  ∃ (train_length : ℝ),
    train_length / tree_time = (train_length + platform_length) / platform_time ∧
    train_length = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1682_168267


namespace NUMINAMATH_CALUDE_objects_meet_time_l1682_168205

/-- Two objects moving towards each other meet after 10 seconds -/
theorem objects_meet_time : ∃ t : ℝ, t = 10 ∧ 
  390 = 3 * t^2 + 0.012 * (t - 5) := by sorry

end NUMINAMATH_CALUDE_objects_meet_time_l1682_168205


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l1682_168239

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 1986

/-- The total number of books in all Oak Grove libraries -/
def total_books : ℕ := 7092

/-- The number of books in Oak Grove's school libraries -/
def school_library_books : ℕ := total_books - public_library_books

theorem oak_grove_library_books : school_library_books = 5106 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l1682_168239


namespace NUMINAMATH_CALUDE_world_cup_souvenir_production_l1682_168295

def planned_daily_production : ℕ := 10000

def production_deviations : List ℤ := [41, -34, -52, 127, -72, 36, -29]

def production_cost : ℕ := 35

def selling_price : ℕ := 40

theorem world_cup_souvenir_production 
  (planned_daily_production : ℕ)
  (production_deviations : List ℤ)
  (production_cost selling_price : ℕ)
  (h1 : planned_daily_production = 10000)
  (h2 : production_deviations = [41, -34, -52, 127, -72, 36, -29])
  (h3 : production_cost = 35)
  (h4 : selling_price = 40) :
  (∃ (max min : ℤ), max ∈ production_deviations ∧ 
                    min ∈ production_deviations ∧ 
                    max - min = 199) ∧
  (production_deviations.sum = 17) ∧
  ((7 * planned_daily_production + production_deviations.sum) * 
   (selling_price - production_cost) = 350085) :=
by sorry

end NUMINAMATH_CALUDE_world_cup_souvenir_production_l1682_168295


namespace NUMINAMATH_CALUDE_intersection_properties_l1682_168278

-- Define the line l: ax - y + 2 - 2a = 0
def line_equation (a x y : ℝ) : Prop := a * x - y + 2 - 2 * a = 0

-- Define the circle C: (x - 4)² + (y - 1)² = r²
def circle_equation (x y r : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = r^2

-- Define the intersection condition
def intersects_at_two_points (a r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
    circle_equation x₁ y₁ r ∧ circle_equation x₂ y₂ r

theorem intersection_properties (a r : ℝ) (hr : r > 0) 
  (h_intersect : intersects_at_two_points a r) :
  -- 1. The line passes through (2, 2)
  (line_equation a 2 2) ∧
  -- 2. r > √5
  (r > Real.sqrt 5) ∧
  -- 3. When r = 3, the chord length is between 4 and 6
  (r = 3 → ∃ (l : ℝ), 4 ≤ l ∧ l ≤ 6 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
      circle_equation x₁ y₁ 3 ∧ circle_equation x₂ y₂ 3 →
      l = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ∧
  -- 4. When r = 5, the minimum dot product is -25
  (r = 5 → ∃ (min_dot_product : ℝ), min_dot_product = -25 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
      circle_equation x₁ y₁ 5 ∧ circle_equation x₂ y₂ 5 →
      ((x₁ - 4) * (x₂ - 4) + (y₁ - 1) * (y₂ - 1)) ≥ min_dot_product) :=
by sorry

end NUMINAMATH_CALUDE_intersection_properties_l1682_168278


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_2_pow_12_l1682_168204

-- Define the function to get the last digit of a rational number's decimal expansion
noncomputable def lastDigitOfDecimalExpansion (q : ℚ) : ℕ :=
  sorry

-- Theorem statement
theorem last_digit_of_one_over_2_pow_12 :
  lastDigitOfDecimalExpansion (1 / 2^12) = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_2_pow_12_l1682_168204


namespace NUMINAMATH_CALUDE_wand_original_price_l1682_168277

/-- If a price is one-eighth of the original price and equals $12, then the original price is $96. -/
theorem wand_original_price (price : ℝ) (original : ℝ) : 
  price = original * (1/8) → price = 12 → original = 96 := by sorry

end NUMINAMATH_CALUDE_wand_original_price_l1682_168277


namespace NUMINAMATH_CALUDE_desired_annual_profit_l1682_168234

def annual_fixed_costs : ℕ := 50200000
def average_cost_per_vehicle : ℕ := 5000
def forecasted_sales : ℕ := 20000
def selling_price_per_car : ℕ := 9035

theorem desired_annual_profit :
  (selling_price_per_car * forecasted_sales) - 
  (annual_fixed_costs + average_cost_per_vehicle * forecasted_sales) = 30500000 := by
  sorry

end NUMINAMATH_CALUDE_desired_annual_profit_l1682_168234


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1682_168226

-- Problem 1
theorem problem_1 : (Real.sqrt 48 - (1/4) * Real.sqrt 6) / (-(1/9) * Real.sqrt 27) = -12 + (3/4) * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = (1/2) * (Real.sqrt 3 + 1)) (hy : y = (1/2) * (1 - Real.sqrt 3)) :
  x^2 + y^2 - 2*x*y = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1682_168226


namespace NUMINAMATH_CALUDE_quadratic_root_implies_b_l1682_168255

theorem quadratic_root_implies_b (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x - 6 = 0) ∧ (2^2 + b*2 - 6 = 0) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_b_l1682_168255


namespace NUMINAMATH_CALUDE_exactly_two_cheaper_to_buy_more_l1682_168203

-- Define the cost function
def C (n : ℕ) : ℝ :=
  if n ≤ 30 then 15 * n - 20
  else if n ≤ 55 then 14 * n
  else 13 * n + 10

-- Define a function that checks if it's cheaper to buy n+1 books than n books
def cheaperToBuyMore (n : ℕ) : Prop := C (n + 1) < C n

-- Theorem statement
theorem exactly_two_cheaper_to_buy_more :
  ∃! (s : Finset ℕ), (∀ n ∈ s, cheaperToBuyMore n) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_cheaper_to_buy_more_l1682_168203


namespace NUMINAMATH_CALUDE_equation_solution_l1682_168233

theorem equation_solution (x : ℝ) : 
  1 - 6/x + 9/x^2 - 4/x^3 = 0 → (3/x = 3 ∨ 3/x = 3/4) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1682_168233


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1682_168246

/-- Definition of a quadratic equation in x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation x^2 + 3x - 5 = 0 -/
def f (x : ℝ) : ℝ := x^2 + 3*x - 5

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1682_168246


namespace NUMINAMATH_CALUDE_interest_rate_equation_l1682_168299

/-- Given the following conditions:
  - Manoj borrowed Rs. 3900 from Anwar
  - The loan is for 3 years
  - Manoj lent Rs. 5655 to Ramu for 3 years at 9% p.a. simple interest
  - Manoj gains Rs. 824.85 from the whole transaction
Prove that the interest rate r at which Manoj borrowed from Anwar satisfies the equation:
5655 * 0.09 * 3 - 3900 * (r / 100) * 3 = 824.85 -/
theorem interest_rate_equation (borrowed : ℝ) (lent : ℝ) (duration : ℝ) (ramu_rate : ℝ) (gain : ℝ) (r : ℝ) 
    (h1 : borrowed = 3900)
    (h2 : lent = 5655)
    (h3 : duration = 3)
    (h4 : ramu_rate = 0.09)
    (h5 : gain = 824.85) :
  lent * ramu_rate * duration - borrowed * (r / 100) * duration = gain := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equation_l1682_168299


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1682_168297

theorem quadratic_root_problem (m : ℝ) :
  (∃ x : ℝ, 3 * x^2 - m * x - 3 = 0 ∧ x = 1) →
  (∃ y : ℝ, 3 * y^2 - m * y - 3 = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1682_168297


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1682_168258

theorem max_value_of_expression (x y : ℝ) (h : x^2 + y^2 ≤ 1) :
  |x^2 + 2*x*y - y^2| ≤ Real.sqrt 2 ∧ ∃ x y, x^2 + y^2 ≤ 1 ∧ |x^2 + 2*x*y - y^2| = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1682_168258


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l1682_168266

/-- An isosceles right triangle with perimeter 14 + 14√2 has a hypotenuse of length 28 -/
theorem isosceles_right_triangle_hypotenuse : ∀ a c : ℝ,
  a > 0 → c > 0 →
  a = c / Real.sqrt 2 →  -- Condition for isosceles right triangle
  2 * a + c = 14 + 14 * Real.sqrt 2 →  -- Perimeter condition
  c = 28 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l1682_168266
