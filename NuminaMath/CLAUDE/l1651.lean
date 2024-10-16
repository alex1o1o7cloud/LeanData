import Mathlib

namespace NUMINAMATH_CALUDE_min_socks_for_15_pairs_l1651_165132

/-- Represents a box of socks with four different colors. -/
structure SockBox where
  purple : ℕ
  orange : ℕ
  yellow : ℕ
  green : ℕ

/-- The minimum number of socks needed to guarantee at least n pairs. -/
def min_socks_for_pairs (n : ℕ) : ℕ := 2 * n + 3

/-- Theorem stating the minimum number of socks needed for 15 pairs. -/
theorem min_socks_for_15_pairs (box : SockBox) :
  min_socks_for_pairs 15 = 33 :=
sorry

end NUMINAMATH_CALUDE_min_socks_for_15_pairs_l1651_165132


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l1651_165188

theorem cos_pi_third_minus_alpha (α : ℝ) (h : Real.sin (π / 6 + α) = 2 / 3) :
  Real.cos (π / 3 - α) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l1651_165188


namespace NUMINAMATH_CALUDE_raft_drift_theorem_l1651_165118

/-- The time for a raft to drift between two villages -/
def raft_drift_time (distance : ℝ) (steamboat_time : ℝ) (motorboat_time : ℝ) : ℝ :=
  90

/-- Theorem: The raft drift time is 90 minutes given the conditions -/
theorem raft_drift_theorem (distance : ℝ) (steamboat_time : ℝ) (motorboat_time : ℝ) 
  (h1 : distance = 1)
  (h2 : steamboat_time = 1)
  (h3 : motorboat_time = 45 / 60)
  (h4 : ∃ (steamboat_speed : ℝ), 
    motorboat_time = distance / (2 * steamboat_speed + (distance / steamboat_time - steamboat_speed))) :
  raft_drift_time distance steamboat_time motorboat_time = 90 := by
  sorry

#check raft_drift_theorem

end NUMINAMATH_CALUDE_raft_drift_theorem_l1651_165118


namespace NUMINAMATH_CALUDE_house_transaction_profit_l1651_165159

def initial_value : ℝ := 15000
def profit_percentage : ℝ := 0.20
def loss_percentage : ℝ := 0.15

theorem house_transaction_profit : 
  let first_sale := initial_value * (1 + profit_percentage)
  let second_sale := first_sale * (1 - loss_percentage)
  first_sale - second_sale = 2700 := by sorry

end NUMINAMATH_CALUDE_house_transaction_profit_l1651_165159


namespace NUMINAMATH_CALUDE_trig_identities_l1651_165117

theorem trig_identities (θ : Real) (h : Real.sin (θ - π/3) = 1/3) :
  (Real.sin (θ + 2*π/3) = -1/3) ∧ (Real.cos (θ - 5*π/6) = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l1651_165117


namespace NUMINAMATH_CALUDE_light_glow_theorem_l1651_165194

def seconds_since_midnight (hours minutes seconds : ℕ) : ℕ :=
  hours * 3600 + minutes * 60 + seconds

def light_glow_count (start_time end_time glow_interval : ℕ) : ℕ :=
  (end_time - start_time) / glow_interval

theorem light_glow_theorem (start_a start_b start_c end_time : ℕ) 
  (interval_a interval_b interval_c : ℕ) : 
  let count_a := light_glow_count start_a end_time interval_a
  let count_b := light_glow_count start_b end_time interval_b
  let count_c := light_glow_count start_c end_time interval_c
  ∃ (x y z : ℕ), x = count_a ∧ y = count_b ∧ z = count_c := by
  sorry

#eval light_glow_count (seconds_since_midnight 1 57 58) (seconds_since_midnight 3 20 47) 14
#eval light_glow_count (seconds_since_midnight 2 0 25) (seconds_since_midnight 3 20 47) 21
#eval light_glow_count (seconds_since_midnight 2 10 15) (seconds_since_midnight 3 20 47) 10

end NUMINAMATH_CALUDE_light_glow_theorem_l1651_165194


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l1651_165196

theorem trigonometric_system_solution :
  let eq1 (x y : Real) := 
    (Real.sin x + Real.cos x) / (Real.sin y + Real.cos y) + 
    (Real.sin y - Real.cos y) / (Real.sin x + Real.cos x) = 
    1 / (Real.sin (x + y) + Real.cos (x - y))
  let eq2 (x y : Real) := 
    2 * (Real.sin x + Real.cos x)^2 - (2 * Real.cos y^2 + 1) = Real.sqrt 3 / 2
  let solutions : List (Real × Real) := 
    [(π/6, π/12), (π/6, 13*π/12), (π/3, 11*π/12), (π/3, 23*π/12)]
  ∀ (x y : Real), (x, y) ∈ solutions → eq1 x y ∧ eq2 x y :=
by
  sorry


end NUMINAMATH_CALUDE_trigonometric_system_solution_l1651_165196


namespace NUMINAMATH_CALUDE_percentage_non_swimmers_basketball_l1651_165121

/-- Represents the percentage of students who play basketball -/
def basketball_players : ℝ := 0.7

/-- Represents the percentage of students who swim -/
def swimmers : ℝ := 0.5

/-- Represents the percentage of basketball players who also swim -/
def basketball_and_swim : ℝ := 0.3

/-- Theorem: The percentage of non-swimmers who play basketball is 98% -/
theorem percentage_non_swimmers_basketball : 
  (basketball_players - basketball_players * basketball_and_swim) / (1 - swimmers) = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_percentage_non_swimmers_basketball_l1651_165121


namespace NUMINAMATH_CALUDE_no_real_j_for_single_solution_l1651_165146

theorem no_real_j_for_single_solution :
  ¬ ∃ j : ℝ, ∃! x : ℝ, (2 * x + 7) * (x - 5) + 3 * x^2 = -20 + (j + 3) * x + 3 * x^2 :=
by sorry

end NUMINAMATH_CALUDE_no_real_j_for_single_solution_l1651_165146


namespace NUMINAMATH_CALUDE_pencil_average_price_l1651_165105

/-- Given the purchase of pens and pencils, prove the average price of a pencil -/
theorem pencil_average_price 
  (total_cost : ℝ) 
  (num_pens : ℕ) 
  (num_pencils : ℕ) 
  (pen_avg_price : ℝ) 
  (h1 : total_cost = 570)
  (h2 : num_pens = 30)
  (h3 : num_pencils = 75)
  (h4 : pen_avg_price = 14) :
  (total_cost - num_pens * pen_avg_price) / num_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_average_price_l1651_165105


namespace NUMINAMATH_CALUDE_equation_solution_l1651_165164

theorem equation_solution :
  ∀ (k m n : ℕ),
  (1/2 : ℝ)^16 * (1/81 : ℝ)^k = (1/18 : ℝ)^16 →
  (1/3 : ℝ)^n * (1/27 : ℝ)^m = (1/18 : ℝ)^k →
  k = 8 ∧ n + 3 * m = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1651_165164


namespace NUMINAMATH_CALUDE_N_smallest_with_digit_sum_2021_sum_of_digits_N_plus_2021_l1651_165173

/-- The smallest positive integer whose digits sum to 2021 -/
def N : ℕ := sorry

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the property of N -/
theorem N_smallest_with_digit_sum_2021 :
  (∀ m : ℕ, m < N → sum_of_digits m ≠ 2021) ∧
  sum_of_digits N = 2021 := by sorry

/-- Main theorem to prove -/
theorem sum_of_digits_N_plus_2021 :
  sum_of_digits (N + 2021) = 10 := by sorry

end NUMINAMATH_CALUDE_N_smallest_with_digit_sum_2021_sum_of_digits_N_plus_2021_l1651_165173


namespace NUMINAMATH_CALUDE_square_plus_n_equals_n_times_n_plus_one_l1651_165119

theorem square_plus_n_equals_n_times_n_plus_one (n : ℕ) : n^2 + n = n * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_plus_n_equals_n_times_n_plus_one_l1651_165119


namespace NUMINAMATH_CALUDE_average_assembly_rate_l1651_165174

/-- Represents the car assembly problem with the given conditions -/
def CarAssemblyProblem (x : ℝ) : Prop :=
  let original_plan := 21
  let assembled_before_order := 6
  let additional_order := 5
  let increased_rate := x + 2
  (original_plan / x) - (assembled_before_order / x) - 
    ((original_plan - assembled_before_order + additional_order) / increased_rate) = 1

/-- Theorem stating that the average daily assembly rate after the additional order is 5 cars per day -/
theorem average_assembly_rate : ∃ x : ℝ, CarAssemblyProblem x ∧ x + 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_assembly_rate_l1651_165174


namespace NUMINAMATH_CALUDE_dal_gain_is_104_l1651_165115

/-- Calculates the total gain from selling a mixture of dals -/
def calculate_dal_gain (dal_a_kg : ℝ) (dal_a_rate : ℝ) (dal_b_kg : ℝ) (dal_b_rate : ℝ)
                       (dal_c_kg : ℝ) (dal_c_rate : ℝ) (dal_d_kg : ℝ) (dal_d_rate : ℝ)
                       (mixture_rate : ℝ) : ℝ :=
  let total_cost := dal_a_kg * dal_a_rate + dal_b_kg * dal_b_rate +
                    dal_c_kg * dal_c_rate + dal_d_kg * dal_d_rate
  let total_weight := dal_a_kg + dal_b_kg + dal_c_kg + dal_d_kg
  let total_revenue := total_weight * mixture_rate
  total_revenue - total_cost

theorem dal_gain_is_104 :
  calculate_dal_gain 15 14.5 10 13 12 16 8 18 17.5 = 104 := by
  sorry

end NUMINAMATH_CALUDE_dal_gain_is_104_l1651_165115


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1651_165166

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  geometric_subsequence : (a 2) ^ 2 = a 1 * a 6
  sum_condition : 2 * a 1 + a 2 = 1

/-- The main theorem stating the explicit formula for the nth term -/
theorem arithmetic_sequence_formula (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 5/3 - n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1651_165166


namespace NUMINAMATH_CALUDE_minimum_in_set_l1651_165138

theorem minimum_in_set (a b c : ℕ+) (h : b ∣ a * c - 1) :
  ∃! r : ℚ, r < 1 ∧ 
  (∀ x ∈ {x : ℚ | ∃ m n : ℤ, x = m * (r - a * c) + n * a * b}, x > 0 → x ≥ a * b / (a + b)) ∧
  r = a / (a + b) := by sorry

end NUMINAMATH_CALUDE_minimum_in_set_l1651_165138


namespace NUMINAMATH_CALUDE_line_intersections_l1651_165110

/-- The line equation 4y - 5x = 20 -/
def line_equation (x y : ℝ) : Prop := 4 * y - 5 * x = 20

/-- The x-axis intercept of the line -/
def x_intercept : ℝ × ℝ := (-4, 0)

/-- The y-axis intercept of the line -/
def y_intercept : ℝ × ℝ := (0, 5)

/-- Theorem stating that the line intersects the x-axis and y-axis at the given points -/
theorem line_intersections :
  (line_equation x_intercept.1 x_intercept.2) ∧
  (line_equation y_intercept.1 y_intercept.2) :=
by sorry

end NUMINAMATH_CALUDE_line_intersections_l1651_165110


namespace NUMINAMATH_CALUDE_four_nested_s_of_6_l1651_165198

-- Define the function s
def s (x : ℚ) : ℚ := 1 / (2 - x)

-- State the theorem
theorem four_nested_s_of_6 : s (s (s (s 6))) = 14 / 19 := by sorry

end NUMINAMATH_CALUDE_four_nested_s_of_6_l1651_165198


namespace NUMINAMATH_CALUDE_unique_intersection_characterization_l1651_165142

/-- A line that has only one common point (-1, -1) with the parabola y = 8x^2 + 10x + 1 -/
def uniqueIntersectionLine (f : ℝ → ℝ) : Prop :=
  (∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -1 ∧ p.2 = f p.1 ∧ p.2 = 8 * p.1^2 + 10 * p.1 + 1) ∧
  (∀ x : ℝ, f x = -6 * x - 7 ∨ (∀ y : ℝ, f y = -1))

/-- The theorem stating that a line has a unique intersection with the parabola
    if and only if it's either y = -6x - 7 or x = -1 -/
theorem unique_intersection_characterization :
  ∀ f : ℝ → ℝ, uniqueIntersectionLine f ↔ 
    (∀ x : ℝ, f x = -6 * x - 7) ∨ (∀ x : ℝ, f x = -1) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_characterization_l1651_165142


namespace NUMINAMATH_CALUDE_total_food_items_is_149_l1651_165158

/-- Represents the eating habits of a person -/
structure EatingHabits where
  croissants : ℕ
  cakes : ℕ
  pizzas : ℕ

/-- Calculates the total food items consumed by a person -/
def totalFoodItems (habits : EatingHabits) : ℕ :=
  habits.croissants + habits.cakes + habits.pizzas

/-- The eating habits of Jorge -/
def jorge : EatingHabits :=
  { croissants := 7, cakes := 18, pizzas := 30 }

/-- The eating habits of Giuliana -/
def giuliana : EatingHabits :=
  { croissants := 5, cakes := 14, pizzas := 25 }

/-- The eating habits of Matteo -/
def matteo : EatingHabits :=
  { croissants := 6, cakes := 16, pizzas := 28 }

/-- Theorem stating that the total food items consumed by Jorge, Giuliana, and Matteo is 149 -/
theorem total_food_items_is_149 :
  totalFoodItems jorge + totalFoodItems giuliana + totalFoodItems matteo = 149 := by
  sorry

end NUMINAMATH_CALUDE_total_food_items_is_149_l1651_165158


namespace NUMINAMATH_CALUDE_unique_triangle_solution_l1651_165116

theorem unique_triangle_solution (a b : ℝ) (A : ℝ) (ha : a = 30) (hb : b = 25) (hA : A = 150 * π / 180) :
  ∃! (c : ℝ) (B C : ℝ), 
    0 < c ∧ 0 < B ∧ 0 < C ∧
    a / Real.sin A = b / Real.sin B ∧
    b / Real.sin B = c / Real.sin C ∧
    A + B + C = π := by
  sorry

end NUMINAMATH_CALUDE_unique_triangle_solution_l1651_165116


namespace NUMINAMATH_CALUDE_insurance_covers_80_percent_l1651_165114

def number_of_vaccines : ℕ := 10
def cost_per_vaccine : ℚ := 45
def cost_of_doctors_visit : ℚ := 250
def trip_cost : ℚ := 1200
def toms_payment : ℚ := 1340

def total_medical_cost : ℚ := number_of_vaccines * cost_per_vaccine + cost_of_doctors_visit
def total_trip_cost : ℚ := trip_cost + total_medical_cost
def insurance_coverage : ℚ := total_trip_cost - toms_payment
def insurance_coverage_percentage : ℚ := insurance_coverage / total_medical_cost * 100

theorem insurance_covers_80_percent :
  insurance_coverage_percentage = 80 := by sorry

end NUMINAMATH_CALUDE_insurance_covers_80_percent_l1651_165114


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l1651_165175

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 1
  f 0 = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l1651_165175


namespace NUMINAMATH_CALUDE_cookie_price_equality_l1651_165180

/-- The radius of Art's circular cookies -/
def art_radius : ℝ := 2

/-- The side length of Roger's square cookies -/
def roger_side : ℝ := 4

/-- The number of cookies Art makes from one batch -/
def art_cookie_count : ℕ := 9

/-- The price of one of Art's cookies in cents -/
def art_cookie_price : ℕ := 50

/-- The price of one of Roger's cookies in cents -/
def roger_cookie_price : ℕ := 64

theorem cookie_price_equality :
  let art_total_area := art_cookie_count * Real.pi * art_radius^2
  let roger_cookie_area := roger_side^2
  let roger_cookie_count := art_total_area / roger_cookie_area
  art_cookie_count * art_cookie_price = ⌊roger_cookie_count⌋ * roger_cookie_price :=
sorry

end NUMINAMATH_CALUDE_cookie_price_equality_l1651_165180


namespace NUMINAMATH_CALUDE_recommendation_plans_count_l1651_165135

/-- Represents the number of recommendation spots for each language --/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the gender distribution of candidates --/
structure CandidateGenders :=
  (males : Nat)
  (females : Nat)

/-- Calculates the number of different recommendation plans --/
def count_recommendation_plans (spots : RecommendationSpots) (genders : CandidateGenders) : Nat :=
  sorry

/-- The main theorem to prove --/
theorem recommendation_plans_count :
  let spots := RecommendationSpots.mk 2 2 1
  let genders := CandidateGenders.mk 3 2
  count_recommendation_plans spots genders = 24 := by
  sorry

end NUMINAMATH_CALUDE_recommendation_plans_count_l1651_165135


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l1651_165179

/-- The number of ways to arrange 4 students into 2 out of 6 classes, with 2 students in each chosen class -/
def arrangementCount : ℕ :=
  (Nat.choose 6 2 * Nat.factorial 2 * Nat.choose 4 2) / 2

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_correct :
  arrangementCount = (Nat.choose 6 2 * Nat.factorial 2 * Nat.choose 4 2) / 2 := by
  sorry

#eval arrangementCount

end NUMINAMATH_CALUDE_arrangement_count_correct_l1651_165179


namespace NUMINAMATH_CALUDE_solution_and_minimum_l1651_165148

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 2|

-- Define the solution set M
def M : Set ℝ := {x | 2/3 ≤ x ∧ x ≤ 6}

-- State the theorem
theorem solution_and_minimum :
  (∀ x ∈ M, f x ≥ -1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 4*a + b + c = 6 →
    1/(2*a + b) + 1/(2*a + c) ≥ 2/3) :=
by sorry

end NUMINAMATH_CALUDE_solution_and_minimum_l1651_165148


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1651_165199

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  (a 1)^2 - 10*(a 1) + 16 = 0 →
  (a 2015)^2 - 10*(a 2015) + 16 = 0 →
  a 2 + a 1008 + a 2014 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1651_165199


namespace NUMINAMATH_CALUDE_spherical_coordinates_conversion_l1651_165193

/-- Converts non-standard spherical coordinates to standard form -/
def standardize_spherical (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Checks if spherical coordinates are in standard form -/
def is_standard_form (ρ θ φ : ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

theorem spherical_coordinates_conversion :
  let original := (5, 5 * Real.pi / 6, 9 * Real.pi / 4)
  let standard := (5, 11 * Real.pi / 6, 3 * Real.pi / 4)
  standardize_spherical original.1 original.2.1 original.2.2 = standard ∧
  is_standard_form standard.1 standard.2.1 standard.2.2 :=
sorry

end NUMINAMATH_CALUDE_spherical_coordinates_conversion_l1651_165193


namespace NUMINAMATH_CALUDE_exactly_one_line_with_two_rational_points_l1651_165123

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A rational point is a point with rational coordinates -/
def RationalPoint (p : Point) : Prop :=
  ∃ (qx qy : ℚ), p.x = qx ∧ p.y = qy

/-- A line passes through a point if the point satisfies the line equation -/
def LinePassesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- A line contains at least two rational points -/
def LineContainsTwoRationalPoints (l : Line) : Prop :=
  ∃ (p1 p2 : Point), p1 ≠ p2 ∧ RationalPoint p1 ∧ RationalPoint p2 ∧
    LinePassesThrough l p1 ∧ LinePassesThrough l p2

/-- The main theorem -/
theorem exactly_one_line_with_two_rational_points
  (a : ℝ) (h_irrational : ¬ ∃ (q : ℚ), a = q) :
  ∃! (l : Line), LinePassesThrough l (Point.mk a 0) ∧ LineContainsTwoRationalPoints l :=
sorry

end NUMINAMATH_CALUDE_exactly_one_line_with_two_rational_points_l1651_165123


namespace NUMINAMATH_CALUDE_gcd_36745_59858_l1651_165107

theorem gcd_36745_59858 : Nat.gcd 36745 59858 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_36745_59858_l1651_165107


namespace NUMINAMATH_CALUDE_peru_tst_imo_2006_q1_l1651_165172

theorem peru_tst_imo_2006_q1 : 
  {(x, y, z) : ℕ × ℕ × ℕ | 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    ∃ k : ℕ, (Real.sqrt (2006 / (x + y : ℝ)) + 
               Real.sqrt (2006 / (y + z : ℝ)) + 
               Real.sqrt (2006 / (z + x : ℝ))) = k}
  = {(2006, 2006, 2006), (1003, 1003, 7021), (9027, 9027, 9027)} := by
  sorry


end NUMINAMATH_CALUDE_peru_tst_imo_2006_q1_l1651_165172


namespace NUMINAMATH_CALUDE_isolated_sets_intersection_empty_l1651_165161

def is_isolated_element (x : ℤ) (A : Set ℤ) : Prop :=
  x ∈ A ∧ (x - 1) ∉ A ∧ (x + 1) ∉ A

def isolated_set (A : Set ℤ) : Set ℤ :=
  {x | is_isolated_element x A}

def M : Set ℤ := {0, 1, 3}
def N : Set ℤ := {0, 3, 4}

theorem isolated_sets_intersection_empty :
  (isolated_set M) ∩ (isolated_set N) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_isolated_sets_intersection_empty_l1651_165161


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1651_165124

theorem lcm_hcf_problem (x : ℕ) : 
  Nat.lcm 4 x = 36 → Nat.gcd 4 x = 2 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1651_165124


namespace NUMINAMATH_CALUDE_cubic_root_sum_square_l1651_165181

theorem cubic_root_sum_square (a b c s : ℝ) : 
  a^3 - 15*a^2 + 20*a - 4 = 0 →
  b^3 - 15*b^2 + 20*b - 4 = 0 →
  c^3 - 15*c^2 + 20*c - 4 = 0 →
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  s^4 - 28*s^2 - 20*s = 305 + 2*s^2 - 20*s := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_square_l1651_165181


namespace NUMINAMATH_CALUDE_expression_equals_one_l1651_165122

theorem expression_equals_one : (-1)^2 - |(-3)| + (-5) / (-5/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1651_165122


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l1651_165153

/-- The fraction of Earth's surface that humans can inhabit -/
def inhabitable_fraction : ℚ := 1/4

theorem earth_inhabitable_fraction :
  (earth_land_fraction : ℚ) = 1/3 →
  (habitable_land_fraction : ℚ) = 3/4 →
  inhabitable_fraction = earth_land_fraction * habitable_land_fraction :=
by sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l1651_165153


namespace NUMINAMATH_CALUDE_new_average_age_l1651_165168

theorem new_average_age (n : ℕ) (original_avg : ℝ) (new_person_age : ℝ) :
  n = 9 ∧ original_avg = 15 ∧ new_person_age = 35 →
  (n * original_avg + new_person_age) / (n + 1) = 17 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l1651_165168


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l1651_165125

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l1651_165125


namespace NUMINAMATH_CALUDE_exam_score_problem_l1651_165185

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 150 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 42 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l1651_165185


namespace NUMINAMATH_CALUDE_tan_two_implies_fraction_four_fifths_l1651_165162

theorem tan_two_implies_fraction_four_fifths (θ : Real) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_fraction_four_fifths_l1651_165162


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1651_165184

/-- An arithmetic sequence with first term 18 and non-zero common difference d -/
def arithmeticSequence (d : ℝ) (n : ℕ) : ℝ := 18 + (n - 1 : ℝ) * d

theorem arithmetic_geometric_sequence (d : ℝ) (h1 : d ≠ 0) :
  (arithmeticSequence d 4) ^ 2 = (arithmeticSequence d 1) * (arithmeticSequence d 8) →
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1651_165184


namespace NUMINAMATH_CALUDE_mans_speed_with_stream_l1651_165106

/-- 
Given a man's rate (speed in still water) and his speed against the stream,
this theorem proves his speed with the stream.
-/
theorem mans_speed_with_stream 
  (rate : ℝ) 
  (speed_against : ℝ) 
  (h1 : rate = 2) 
  (h2 : speed_against = 6) : 
  rate + (speed_against - rate) = 6 := by
sorry

end NUMINAMATH_CALUDE_mans_speed_with_stream_l1651_165106


namespace NUMINAMATH_CALUDE_min_radius_circle_equation_l1651_165160

/-- The line on which points A and B move --/
def line (x y : ℝ) : Prop := 3 * x + y - 10 = 0

/-- The circle M with diameter AB --/
def circle_M (a b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - (a.1 + b.1) / 2)^2 + (p.2 - (a.2 + b.2) / 2)^2 = ((a.1 - b.1)^2 + (a.2 - b.2)^2) / 4}

/-- The origin point --/
def origin : ℝ × ℝ := (0, 0)

/-- Theorem stating the standard equation of circle M when its radius is minimum --/
theorem min_radius_circle_equation :
  ∀ a b : ℝ × ℝ,
  line a.1 a.2 → line b.1 b.2 →
  origin ∈ circle_M a b →
  (∀ c d : ℝ × ℝ, line c.1 c.2 → line d.1 d.2 → origin ∈ circle_M c d →
    (a.1 - b.1)^2 + (a.2 - b.2)^2 ≤ (c.1 - d.1)^2 + (c.2 - d.2)^2) →
  circle_M a b = {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 10} :=
sorry

end NUMINAMATH_CALUDE_min_radius_circle_equation_l1651_165160


namespace NUMINAMATH_CALUDE_milk_distribution_l1651_165120

def milk_problem (total_milk myeongseok_milk minjae_milk : Real) (mingu_extra : Real) : Prop :=
  let mingu_milk := myeongseok_milk + mingu_extra
  let friends_total := myeongseok_milk + mingu_milk + minjae_milk
  let remaining_milk := total_milk - friends_total
  (total_milk = 1) ∧ 
  (myeongseok_milk = 0.1) ∧ 
  (mingu_extra = 0.2) ∧ 
  (minjae_milk = 0.3) ∧ 
  (remaining_milk = 0.3)

theorem milk_distribution : 
  ∃ (total_milk myeongseok_milk minjae_milk mingu_extra : Real),
    milk_problem total_milk myeongseok_milk minjae_milk mingu_extra :=
by
  sorry

end NUMINAMATH_CALUDE_milk_distribution_l1651_165120


namespace NUMINAMATH_CALUDE_vegetarian_eaters_count_l1651_165143

/-- Represents the dietary preferences of a family -/
structure DietaryPreferences where
  total : Nat
  vegetarianOnly : Nat
  nonVegetarianOnly : Nat
  bothVegAndNonVeg : Nat
  veganOnly : Nat
  pescatarian : Nat
  specificVegetarian : Nat

/-- Calculates the number of people eating vegetarian food -/
def countVegetarianEaters (prefs : DietaryPreferences) : Nat :=
  prefs.vegetarianOnly + prefs.bothVegAndNonVeg + prefs.veganOnly + prefs.pescatarian + prefs.specificVegetarian

/-- Theorem stating that 29 people eat vegetarian food in the given family -/
theorem vegetarian_eaters_count (prefs : DietaryPreferences)
  (h1 : prefs.total = 35)
  (h2 : prefs.vegetarianOnly = 11)
  (h3 : prefs.nonVegetarianOnly = 6)
  (h4 : prefs.bothVegAndNonVeg = 9)
  (h5 : prefs.veganOnly = 3)
  (h6 : prefs.pescatarian = 4)
  (h7 : prefs.specificVegetarian = 2) :
  countVegetarianEaters prefs = 29 := by
  sorry


end NUMINAMATH_CALUDE_vegetarian_eaters_count_l1651_165143


namespace NUMINAMATH_CALUDE_brenda_sally_meeting_distance_l1651_165151

theorem brenda_sally_meeting_distance 
  (track_length : ℝ) 
  (sally_extra_distance : ℝ) 
  (h1 : track_length = 300)
  (h2 : sally_extra_distance = 100) :
  let first_meeting_distance := (track_length / 2 + sally_extra_distance) / 2
  first_meeting_distance = 150 := by
  sorry

end NUMINAMATH_CALUDE_brenda_sally_meeting_distance_l1651_165151


namespace NUMINAMATH_CALUDE_alpha_sufficient_not_necessary_for_beta_l1651_165182

def α (x y : ℝ) : Prop := x = 1 ∧ y = 2
def β (x y : ℝ) : Prop := x + y = 3

theorem alpha_sufficient_not_necessary_for_beta :
  (∀ x y : ℝ, α x y → β x y) ∧
  (∃ x y : ℝ, β x y ∧ ¬(α x y)) := by
  sorry

end NUMINAMATH_CALUDE_alpha_sufficient_not_necessary_for_beta_l1651_165182


namespace NUMINAMATH_CALUDE_person_a_number_l1651_165155

theorem person_a_number : ∀ (A B : ℕ), 
  A < 10 → B < 10 →
  A + B = 8 →
  (10 * B + A) - (10 * A + B) = 18 →
  10 * A + B = 35 := by
sorry

end NUMINAMATH_CALUDE_person_a_number_l1651_165155


namespace NUMINAMATH_CALUDE_marcos_trading_cards_l1651_165171

theorem marcos_trading_cards (x : ℝ) : 
  let duplicates := (1/3) * x
  let new_cards_from_josh := (1/5) * duplicates
  let new_cards_kept_from_josh := (2/3) * new_cards_from_josh
  let new_cards_kept_from_alex := (1/3) * new_cards_kept_from_josh
  let final_new_cards := new_cards_from_josh + new_cards_kept_from_josh + new_cards_kept_from_alex
  final_new_cards = 850 → x = 6375 :=
by
  sorry

end NUMINAMATH_CALUDE_marcos_trading_cards_l1651_165171


namespace NUMINAMATH_CALUDE_book_selection_theorem_l1651_165100

/-- Given the number of books in each language, calculates the number of ways to select two books. -/
def book_selection (japanese : ℕ) (english : ℕ) (chinese : ℕ) :
  (ℕ × ℕ × ℕ) :=
  let different_languages := japanese * english + japanese * chinese + english * chinese
  let same_language := japanese * (japanese - 1) / 2 + english * (english - 1) / 2 + chinese * (chinese - 1) / 2
  let total := (japanese + english + chinese) * (japanese + english + chinese - 1) / 2
  (different_languages, same_language, total)

/-- Theorem stating the correct number of ways to select books given the specified quantities. -/
theorem book_selection_theorem :
  book_selection 5 7 10 = (155, 76, 231) := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l1651_165100


namespace NUMINAMATH_CALUDE_smallest_number_of_editors_l1651_165111

/-- The total number of people at the conference -/
def total : ℕ := 90

/-- The number of writers at the conference -/
def writers : ℕ := 45

/-- The number of people who are both writers and editors -/
def both : ℕ := 6

/-- The number of people who are neither writers nor editors -/
def neither : ℕ := 2 * both

/-- The number of editors at the conference -/
def editors : ℕ := total - writers - neither + both

theorem smallest_number_of_editors : editors = 39 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_editors_l1651_165111


namespace NUMINAMATH_CALUDE_cos_phi_value_l1651_165102

-- Define the function f
variable (f : ℝ → ℝ)

-- Define φ
variable (φ : ℝ)

-- Define x₁
variable (x₁ : ℝ)

-- f(x) - sin(x + φ) is an even function
axiom even_func : ∀ x, f (-x) - Real.sin (-x + φ) = f x - Real.sin (x + φ)

-- f(x) - cos(x + φ) is an odd function
axiom odd_func : ∀ x, f (-x) - Real.cos (-x + φ) = -(f x - Real.cos (x + φ))

-- The slopes of the tangent lines at P and Q are reciprocals
axiom reciprocal_slopes : 
  (deriv f x₁) * (deriv f (x₁ + Real.pi / 2)) = 1

-- Theorem statement
theorem cos_phi_value : Real.cos φ = 1 ∨ Real.cos φ = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_phi_value_l1651_165102


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1651_165191

theorem smallest_number_with_remainders (n : ℕ) :
  (∃ m : ℕ, n = 5 * m + 4) ∧
  (∃ m : ℕ, n = 6 * m + 5) ∧
  (((∃ m : ℕ, n = 7 * m + 6) → n ≥ 209) ∧
   ((∃ m : ℕ, n = 8 * m + 7) → n ≥ 119)) ∧
  (n = 209 ∨ n = 119) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1651_165191


namespace NUMINAMATH_CALUDE_inequality_proof_l1651_165178

theorem inequality_proof (a b c d : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0)
  (h : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1651_165178


namespace NUMINAMATH_CALUDE_beats_played_example_l1651_165195

/-- Given a person who plays music at a certain rate for a specific duration each day over multiple days, calculate the total number of beats played. -/
def totalBeatsPlayed (beatsPerMinute : ℕ) (hoursPerDay : ℕ) (numberOfDays : ℕ) : ℕ :=
  beatsPerMinute * (hoursPerDay * 60) * numberOfDays

/-- Theorem stating that playing 200 beats per minute for 2 hours a day for 3 days results in 72,000 beats total. -/
theorem beats_played_example : totalBeatsPlayed 200 2 3 = 72000 := by
  sorry

end NUMINAMATH_CALUDE_beats_played_example_l1651_165195


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l1651_165149

-- Define the number of posts
def num_posts : ℕ := 24

-- Define the width of each post in feet
def post_width : ℚ := 1 / 3

-- Define the distance between adjacent posts in feet
def post_spacing : ℕ := 5

-- Define the number of posts per side (excluding corners)
def posts_per_side : ℕ := (num_posts - 4) / 4

-- Define the total number of posts per side (including corners)
def total_posts_per_side : ℕ := posts_per_side + 2

-- Define the number of gaps between posts on one side
def gaps_per_side : ℕ := total_posts_per_side - 1

-- Define the length of one side of the square
def side_length : ℚ := gaps_per_side * post_spacing + total_posts_per_side * post_width

-- Theorem statement
theorem square_fence_perimeter :
  4 * side_length = 129 + 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l1651_165149


namespace NUMINAMATH_CALUDE_flag_design_count_l1651_165101

/-- The number of colors available for the flag -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 4

/-- A function that calculates the number of possible flag designs -/
def flag_designs (n : ℕ) (k : ℕ) : ℕ :=
  if n = 1 then k
  else k * (k - 1)^(n - 1)

/-- Theorem stating that the number of possible flag designs is 24 -/
theorem flag_design_count :
  flag_designs num_stripes num_colors = 24 := by
  sorry

end NUMINAMATH_CALUDE_flag_design_count_l1651_165101


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l1651_165147

/-- Given the following conditions:
    - 3 blankets at Rs. 100 each
    - 6 blankets at Rs. 150 each
    - 2 blankets at an unknown rate
    - The average price of all blankets is Rs. 150
    Prove that the unknown rate must be Rs. 225 per blanket -/
theorem unknown_blanket_rate (price1 : ℕ) (price2 : ℕ) (unknown_price : ℕ) 
    (h1 : price1 = 100)
    (h2 : price2 = 150)
    (h3 : (3 * price1 + 6 * price2 + 2 * unknown_price) / 11 = 150) :
    unknown_price = 225 := by
  sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l1651_165147


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l1651_165167

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 15 → 
  ∀ x, x^2 - 16*x + 225 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l1651_165167


namespace NUMINAMATH_CALUDE_skater_speed_l1651_165176

/-- Given a skater who travels 80 kilometers in 8 hours, prove their speed is 10 kilometers per hour. -/
theorem skater_speed (distance : ℝ) (time : ℝ) (h1 : distance = 80) (h2 : time = 8) :
  distance / time = 10 := by sorry

end NUMINAMATH_CALUDE_skater_speed_l1651_165176


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l1651_165170

theorem average_of_five_numbers : 
  let numbers : List ℕ := [8, 9, 10, 11, 12]
  (numbers.sum / numbers.length : ℚ) = 10 := by
sorry

end NUMINAMATH_CALUDE_average_of_five_numbers_l1651_165170


namespace NUMINAMATH_CALUDE_m_range_theorem_l1651_165144

-- Define the quadratic function p
def p (x : ℝ) : Prop := x^2 - 8*x - 20 > 0

-- Define the function q
def q (x m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) > 0

-- State the theorem
theorem m_range_theorem (h_sufficient : ∀ x m : ℝ, p x → q x m) 
                        (h_not_necessary : ∃ x m : ℝ, q x m ∧ ¬(p x))
                        (h_m_positive : m > 0) :
  0 < m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1651_165144


namespace NUMINAMATH_CALUDE_two_hour_charge_is_161_l1651_165103

/-- Represents the pricing structure and total charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ  -- Price for the first hour
  additional_hour : ℕ  -- Price for each additional hour
  first_hour_premium : first_hour = additional_hour + 35  -- First hour costs $35 more
  five_hour_total : first_hour + 4 * additional_hour = 350  -- Total for 5 hours is $350

/-- Calculates the total charge for 2 hours of therapy given the pricing structure. -/
def two_hour_charge (pricing : TherapyPricing) : ℕ :=
  pricing.first_hour + pricing.additional_hour

/-- Theorem stating that the total charge for 2 hours of therapy is $161. -/
theorem two_hour_charge_is_161 (pricing : TherapyPricing) : 
  two_hour_charge pricing = 161 := by
  sorry

end NUMINAMATH_CALUDE_two_hour_charge_is_161_l1651_165103


namespace NUMINAMATH_CALUDE_ginger_mat_straw_ratio_l1651_165163

/-- Given the conditions for Ginger's mat weaving, prove the ratio of green to orange straws per mat -/
theorem ginger_mat_straw_ratio :
  let red_per_mat : ℕ := 20
  let orange_per_mat : ℕ := 30
  let total_mats : ℕ := 10
  let total_straws : ℕ := 650
  let green_per_mat : ℕ := (total_straws - red_per_mat * total_mats - orange_per_mat * total_mats) / total_mats
  green_per_mat * 2 = orange_per_mat := by
  sorry

#check ginger_mat_straw_ratio

end NUMINAMATH_CALUDE_ginger_mat_straw_ratio_l1651_165163


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1651_165112

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 1 = 0

def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 1 = 0

def center1 : ℝ × ℝ := (2, -1)

def center2 : ℝ × ℝ := (-2, 2)

def radius1 : ℝ := 2

def radius2 : ℝ := 3

theorem circles_externally_tangent :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = radius1 + radius2 := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1651_165112


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1651_165139

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ+, 45 ≤ n * 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1651_165139


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_unique_positive_integer_solution_l1651_165128

/-- The quadratic equation x^2 - 2x + 2m - 1 = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 2*x + 2*m - 1 = 0

/-- m is a positive integer -/
def is_positive_integer (m : ℝ) : Prop :=
  m > 0 ∧ ∃ n : ℕ, m = n

theorem quadratic_equation_roots (m : ℝ) :
  has_real_roots m ↔ m ≤ 1 :=
sorry

theorem unique_positive_integer_solution (m : ℝ) :
  is_positive_integer m ∧ has_real_roots m →
  m = 1 ∧ ∃ x : ℝ, x = 1 ∧ x^2 - 2*x + 2*m - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_unique_positive_integer_solution_l1651_165128


namespace NUMINAMATH_CALUDE_unique_solution_proof_l1651_165134

/-- The positive value of m for which the quadratic equation 4x^2 + mx + 4 = 0 has exactly one real solution -/
def unique_solution_m : ℝ := 8

/-- The quadratic equation 4x^2 + mx + 4 = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  4 * x^2 + m * x + 4 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  m^2 - 4 * 4 * 4

theorem unique_solution_proof :
  unique_solution_m > 0 ∧
  discriminant unique_solution_m = 0 ∧
  ∀ m : ℝ, m > 0 → discriminant m = 0 → m = unique_solution_m :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_proof_l1651_165134


namespace NUMINAMATH_CALUDE_graph_not_in_second_quadrant_l1651_165130

/-- A linear function y = 3x + k - 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := 3 * x + k - 2

/-- The second quadrant -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The graph does not pass through the second quadrant -/
def not_in_second_quadrant (k : ℝ) : Prop :=
  ∀ x, ¬(second_quadrant x (f k x))

/-- Theorem: The graph of y = 3x + k - 2 does not pass through the second quadrant
    if and only if k ≤ 2 -/
theorem graph_not_in_second_quadrant (k : ℝ) :
  not_in_second_quadrant k ↔ k ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_graph_not_in_second_quadrant_l1651_165130


namespace NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l1651_165192

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The binary representation of the number 11010 -/
def binary_11010 : List Bool := [false, true, false, true, true]

/-- The octal representation of the number 32 -/
def octal_32 : List ℕ := [3, 2]

theorem binary_11010_equals_octal_32 :
  decimal_to_octal (binary_to_decimal binary_11010) = octal_32 := by
  sorry

end NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l1651_165192


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_for_air_composition_l1651_165186

/-- Represents different types of graphs -/
inductive GraphType
  | BarGraph
  | LineGraph
  | PieChart
  | Histogram

/-- Represents a component of air -/
structure AirComponent where
  name : String
  percentage : Float

/-- Determines if a graph type is suitable for representing percentage composition -/
def isSuitableForPercentageComposition (graphType : GraphType) : Prop :=
  match graphType with
  | GraphType.PieChart => True
  | _ => False

/-- The air composition representation problem -/
theorem pie_chart_most_suitable_for_air_composition 
  (components : List AirComponent) 
  (hComponents : components.all (λ c => c.percentage ≥ 0 ∧ c.percentage ≤ 100)) 
  (hTotalPercentage : components.foldl (λ acc c => acc + c.percentage) 0 = 100) :
  isSuitableForPercentageComposition GraphType.PieChart ∧ 
  (∀ g : GraphType, isSuitableForPercentageComposition g → g = GraphType.PieChart) :=
sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_for_air_composition_l1651_165186


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l1651_165145

/-- The time when the ball hits the ground, given the initial conditions and equation of motion -/
theorem ball_hitting_ground_time : ∃ t : ℝ, t > 0 ∧ -16 * t^2 + 32 * t + 180 = 0 ∧ t = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l1651_165145


namespace NUMINAMATH_CALUDE_red_light_probability_l1651_165169

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of seeing a red light given the traffic light durations -/
def probability_red_light (d : TrafficLightDuration) : ℚ :=
  d.red / (d.red + d.yellow + d.green)

/-- Theorem: The probability of seeing a red light is 2/5 for the given durations -/
theorem red_light_probability (d : TrafficLightDuration) 
  (h_red : d.red = 30)
  (h_yellow : d.yellow = 5)
  (h_green : d.green = 40) : 
  probability_red_light d = 2/5 := by
  sorry

#eval probability_red_light ⟨30, 5, 40⟩

end NUMINAMATH_CALUDE_red_light_probability_l1651_165169


namespace NUMINAMATH_CALUDE_parabola_properties_l1651_165189

-- Define the parabola and its properties
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_a_neg : a < 0
  h_m_bounds : 1 < m ∧ m < 2
  h_passes_through : a * (-1)^2 + b * (-1) + c = 0 ∧ a * m^2 + b * m + c = 0

-- Theorem statements
theorem parabola_properties (p : Parabola) :
  (p.b > 0) ∧
  (∀ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ < x₂ → x₁ + x₂ > 1 → 
    p.a * x₁^2 + p.b * x₁ + p.c = y₁ → 
    p.a * x₂^2 + p.b * x₂ + p.c = y₂ → 
    y₁ > y₂) ∧
  (p.a ≤ -1 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ p.a * x₁^2 + p.b * x₁ + p.c = 1 ∧ p.a * x₂^2 + p.b * x₂ + p.c = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1651_165189


namespace NUMINAMATH_CALUDE_where_is_waldo_books_l1651_165127

/-- The number of "Where's Waldo?" books published -/
def num_books : ℕ := 15

/-- The number of puzzles in each "Where's Waldo?" book -/
def puzzles_per_book : ℕ := 30

/-- The time in minutes to solve one puzzle -/
def time_per_puzzle : ℕ := 3

/-- The total time in minutes to solve all puzzles -/
def total_time : ℕ := 1350

/-- Theorem stating that the number of "Where's Waldo?" books is correct -/
theorem where_is_waldo_books :
  num_books = total_time / (puzzles_per_book * time_per_puzzle) :=
by sorry

end NUMINAMATH_CALUDE_where_is_waldo_books_l1651_165127


namespace NUMINAMATH_CALUDE_average_time_is_five_l1651_165152

/-- Colin's running times for each mile -/
def mile_times : List ℕ := [6, 5, 5, 4]

/-- Total number of miles run -/
def total_miles : ℕ := mile_times.length

/-- Calculates the average time per mile -/
def average_time_per_mile : ℚ :=
  (mile_times.sum : ℚ) / total_miles

/-- Theorem: The average time per mile is 5 minutes -/
theorem average_time_is_five : average_time_per_mile = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_time_is_five_l1651_165152


namespace NUMINAMATH_CALUDE_davids_age_twice_daughters_l1651_165108

/-- 
Given:
- David is currently 40 years old
- David's daughter is currently 12 years old

Prove that 16 years will pass before David's age is twice his daughter's age
-/
theorem davids_age_twice_daughters (david_age : ℕ) (daughter_age : ℕ) :
  david_age = 40 →
  daughter_age = 12 →
  ∃ (years : ℕ), david_age + years = 2 * (daughter_age + years) ∧ years = 16 :=
by sorry

end NUMINAMATH_CALUDE_davids_age_twice_daughters_l1651_165108


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1651_165197

theorem complex_number_modulus : 
  let z : ℂ := (-1 - 2*I) / (1 - I)^2
  ‖z‖ = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1651_165197


namespace NUMINAMATH_CALUDE_min_value_theorem_l1651_165177

/-- The line equation ax - 2by = 2 passes through the center of the circle x² + y² - 4x + 2y + 1 = 0 -/
def line_passes_through_center (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 ∧ a*x - 2*b*y = 2

/-- The minimum value of 1/a + 1/b + 1/(ab) given the conditions -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_center : line_passes_through_center a b) : 
  (1/a + 1/b + 1/(a*b)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1651_165177


namespace NUMINAMATH_CALUDE_fraction_sum_bound_l1651_165129

theorem fraction_sum_bound (a b c : ℕ) (h : (1 : ℚ) / a + 1 / b + 1 / c < 1) :
  (1 : ℚ) / a + 1 / b + 1 / c < 41 / 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_bound_l1651_165129


namespace NUMINAMATH_CALUDE_cyclic_inequality_l1651_165156

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) * Real.sqrt ((y + z) * (z + x)) +
  (y + z) * Real.sqrt ((z + x) * (x + y)) +
  (z + x) * Real.sqrt ((x + y) * (y + z)) ≥
  4 * (x * y + y * z + z * x) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l1651_165156


namespace NUMINAMATH_CALUDE_smallest_perfect_square_factor_l1651_165140

def y : ℕ := 2^5 * 3^2 * 4^6 * 5^6 * 7^8 * 8^9 * 9^10

theorem smallest_perfect_square_factor (k : ℕ) : 
  (k > 0 ∧ ∃ m : ℕ, k * y = m^2) → k ≥ 100 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_factor_l1651_165140


namespace NUMINAMATH_CALUDE_inverse_functions_l1651_165157

-- Define the types of functions
def LinearDecreasing : Type := ℝ → ℝ
def PiecewiseConstant : Type := ℝ → ℝ
def VerticalLine : Type := ℝ → ℝ
def Semicircle : Type := ℝ → ℝ
def ModifiedPolynomial : Type := ℝ → ℝ

-- Define the property of having an inverse
def HasInverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x

-- State the theorem
theorem inverse_functions 
  (F : LinearDecreasing) 
  (G : PiecewiseConstant) 
  (H : VerticalLine) 
  (I : Semicircle) 
  (J : ModifiedPolynomial) : 
  HasInverse F ∧ HasInverse G ∧ ¬HasInverse H ∧ ¬HasInverse I ∧ ¬HasInverse J := by
  sorry

end NUMINAMATH_CALUDE_inverse_functions_l1651_165157


namespace NUMINAMATH_CALUDE_polygon_area_is_300_l1651_165109

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The polygon described in the problem -/
def polygon : List Point := [
  ⟨0, 0⟩, ⟨10, 0⟩, ⟨10, 10⟩, ⟨10, 20⟩, ⟨10, 30⟩, ⟨0, 30⟩, ⟨0, 20⟩, ⟨0, 10⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ :=
  sorry

/-- Theorem: The area of the given polygon is 300 square units -/
theorem polygon_area_is_300 : polygonArea polygon = 300 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_is_300_l1651_165109


namespace NUMINAMATH_CALUDE_triangle_side_cube_l1651_165190

/-- Given a triangle ABC with positive integer side lengths a, b, and c, 
    where gcd(a,b,c) = 1 and ∠A = 3∠B, at least one of a, b, and c is a cube. -/
theorem triangle_side_cube (a b c : ℕ+) (angleA angleB : ℝ) : 
  (a.val.gcd (b.val.gcd c.val) = 1) →
  (angleA = 3 * angleB) →
  (∃ (x : ℕ+), x^3 = a ∨ x^3 = b ∨ x^3 = c) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_cube_l1651_165190


namespace NUMINAMATH_CALUDE_willies_stickers_l1651_165136

/-- Willie's sticker problem -/
theorem willies_stickers (initial : ℕ) (remaining : ℕ) (given : ℕ) : 
  initial = 36 → remaining = 29 → given = initial - remaining :=
by sorry

end NUMINAMATH_CALUDE_willies_stickers_l1651_165136


namespace NUMINAMATH_CALUDE_min_colors_correct_key_coloring_distinguishes_min_colors_optimal_l1651_165154

/-- The smallest number of colors needed to distinguish n keys arranged in a circle -/
def min_colors (n : ℕ) : ℕ :=
  if n ≤ 2 then n
  else if n ≤ 5 then 3
  else 2

/-- Theorem stating the minimum number of colors needed to distinguish n keys -/
theorem min_colors_correct (n : ℕ) :
  min_colors n = 
    if n ≤ 2 then n
    else if n ≤ 5 then 3
    else 2 :=
by
  sorry

/-- The coloring function that assigns colors to keys -/
def key_coloring (n : ℕ) : ℕ → Fin (min_colors n) :=
  sorry

/-- Theorem stating that the key_coloring function distinguishes all keys -/
theorem key_coloring_distinguishes (n : ℕ) :
  ∀ i j : Fin n, i ≠ j → 
    ∃ k : ℕ, (key_coloring n ((i + k) % n) ≠ key_coloring n ((j + k) % n)) ∨
            (key_coloring n ((n - i - k - 1) % n) ≠ key_coloring n ((n - j - k - 1) % n)) :=
by
  sorry

/-- Theorem stating that min_colors n is the smallest number that allows a distinguishing coloring -/
theorem min_colors_optimal (n : ℕ) :
  ∀ m : ℕ, m < min_colors n → 
    ¬∃ f : ℕ → Fin m, ∀ i j : Fin n, i ≠ j → 
      ∃ k : ℕ, (f ((i + k) % n) ≠ f ((j + k) % n)) ∨
              (f ((n - i - k - 1) % n) ≠ f ((n - j - k - 1) % n)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_colors_correct_key_coloring_distinguishes_min_colors_optimal_l1651_165154


namespace NUMINAMATH_CALUDE_simplify_expression_l1651_165165

theorem simplify_expression (x : ℝ) : (x - 3)^2 - (x + 1)*(x - 1) = -6*x + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1651_165165


namespace NUMINAMATH_CALUDE_sum_of_data_l1651_165141

theorem sum_of_data (a b c : ℝ) : 
  a + b = c → 
  b = 3 * a → 
  a = 12 → 
  a + b + c = 96 := by
sorry

end NUMINAMATH_CALUDE_sum_of_data_l1651_165141


namespace NUMINAMATH_CALUDE_contains_2850_thousandths_l1651_165137

theorem contains_2850_thousandths : (2.85 : ℝ) / 0.001 = 2850 := by sorry

end NUMINAMATH_CALUDE_contains_2850_thousandths_l1651_165137


namespace NUMINAMATH_CALUDE_class_size_from_incorrect_mark_l1651_165131

theorem class_size_from_incorrect_mark (original_mark correct_mark : ℚ)
  (h1 : original_mark = 33)
  (h2 : correct_mark = 85)
  (h3 : ∀ (n : ℕ) (A : ℚ), n * (A + 1/2) = n * A + (correct_mark - original_mark)) :
  ∃ (n : ℕ), n = 104 := by
  sorry

end NUMINAMATH_CALUDE_class_size_from_incorrect_mark_l1651_165131


namespace NUMINAMATH_CALUDE_competition_scores_l1651_165150

theorem competition_scores (score24 score46 score12 : ℕ) : 
  score24 + score46 + score12 = 285 →
  ∃ (x : ℕ), score24 - 8 = x ∧ score46 - 12 = x ∧ score12 - 7 = x →
  score24 + score12 = 187 := by
sorry

end NUMINAMATH_CALUDE_competition_scores_l1651_165150


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1651_165126

theorem fraction_evaluation : (1/4 - 1/6) / (1/3 - 1/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1651_165126


namespace NUMINAMATH_CALUDE_solve_equation_l1651_165104

theorem solve_equation (x : ℚ) : 3 * x + 15 = (1 / 3) * (4 * x + 28) → x = -17 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1651_165104


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1651_165187

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) (n : ℕ) :
  (∀ k, a (k + 1) = a k * q) →  -- geometric sequence condition
  q > 0 →  -- positive common ratio
  a 1 * a 2 * a 3 = 4 →  -- first condition
  a 4 * a 5 * a 6 = 8 →  -- second condition
  a n * a (n + 1) * a (n + 2) = 128 →  -- third condition
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1651_165187


namespace NUMINAMATH_CALUDE_montoya_family_budget_l1651_165113

theorem montoya_family_budget (budget : ℝ) 
  (grocery_fraction : ℝ) (total_food_fraction : ℝ) :
  grocery_fraction = 0.6 →
  total_food_fraction = 0.8 →
  total_food_fraction = grocery_fraction + (budget - grocery_fraction * budget) / budget →
  (budget - grocery_fraction * budget) / budget = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_montoya_family_budget_l1651_165113


namespace NUMINAMATH_CALUDE_picnic_adult_child_difference_l1651_165183

/-- A picnic scenario with men, women, adults, and children -/
structure Picnic where
  total : ℕ
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Conditions for the picnic scenario -/
def PicnicConditions (p : Picnic) : Prop :=
  p.total = 240 ∧
  p.men = p.women + 40 ∧
  p.men = 90 ∧
  p.adults > p.children ∧
  p.total = p.men + p.women + p.children ∧
  p.adults = p.men + p.women

/-- Theorem stating the difference between adults and children -/
theorem picnic_adult_child_difference (p : Picnic) 
  (h : PicnicConditions p) : p.adults - p.children = 40 := by
  sorry

#check picnic_adult_child_difference

end NUMINAMATH_CALUDE_picnic_adult_child_difference_l1651_165183


namespace NUMINAMATH_CALUDE_charges_needed_equals_total_rooms_l1651_165133

def battery_duration : ℕ := 10
def vacuum_time_per_room : ℕ := 8
def num_bedrooms : ℕ := 3
def num_kitchen : ℕ := 1
def num_living_room : ℕ := 1
def num_dining_room : ℕ := 1
def num_office : ℕ := 1
def num_bathrooms : ℕ := 2

def total_rooms : ℕ := num_bedrooms + num_kitchen + num_living_room + num_dining_room + num_office + num_bathrooms

theorem charges_needed_equals_total_rooms :
  battery_duration > vacuum_time_per_room ∧
  battery_duration < 2 * vacuum_time_per_room →
  total_rooms = total_rooms :=
by sorry

end NUMINAMATH_CALUDE_charges_needed_equals_total_rooms_l1651_165133
