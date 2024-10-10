import Mathlib

namespace puzzle_solution_l1101_110119

def addition_puzzle (F I V N E : Nat) : Prop :=
  F ≠ I ∧ F ≠ V ∧ F ≠ N ∧ F ≠ E ∧
  I ≠ V ∧ I ≠ N ∧ I ≠ E ∧
  V ≠ N ∧ V ≠ E ∧
  N ≠ E ∧
  F = 8 ∧
  I % 2 = 0 ∧
  1000 * N + 100 * I + 10 * N + E = 100 * F + 10 * I + V + 100 * F + 10 * I + V

theorem puzzle_solution :
  ∀ F I V N E, addition_puzzle F I V N E → V = 5 :=
by sorry

end puzzle_solution_l1101_110119


namespace random_phenomena_l1101_110112

-- Define a type for phenomena
inductive Phenomenon
| TrafficCount
| IntegerSuccessor
| ShellFiring
| ProductInspection

-- Define a predicate for random phenomena
def isRandom (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.TrafficCount => true
  | Phenomenon.IntegerSuccessor => false
  | Phenomenon.ShellFiring => true
  | Phenomenon.ProductInspection => true

-- Theorem statement
theorem random_phenomena :
  (isRandom Phenomenon.TrafficCount) ∧
  (¬isRandom Phenomenon.IntegerSuccessor) ∧
  (isRandom Phenomenon.ShellFiring) ∧
  (isRandom Phenomenon.ProductInspection) :=
by sorry

end random_phenomena_l1101_110112


namespace exactly_two_valid_sets_l1101_110173

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)  -- The first integer in the set
  (length : ℕ) -- The number of integers in the set

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set according to our problem -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 3 ∧ sum_consecutive s = 21

theorem exactly_two_valid_sets :
  ∃! (s₁ s₂ : ConsecutiveSet), is_valid_set s₁ ∧ is_valid_set s₂ ∧ s₁ ≠ s₂ ∧
    ∀ (s : ConsecutiveSet), is_valid_set s → s = s₁ ∨ s = s₂ :=
sorry

end exactly_two_valid_sets_l1101_110173


namespace largest_divisor_n4_minus_n_l1101_110139

/-- A positive integer greater than 1 is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k, 1 < k ∧ k < n ∧ n % k = 0

theorem largest_divisor_n4_minus_n (n : ℕ) (h : IsComposite n) :
  (∀ d : ℕ, d > 6 → ¬(d ∣ (n^4 - n))) ∧
  (6 ∣ (n^4 - n)) := by
  sorry

end largest_divisor_n4_minus_n_l1101_110139


namespace jo_stair_climbing_l1101_110183

/-- The number of ways to climb n stairs, taking 1, 2, or 3 stairs at a time -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => climbStairs (n + 2) + climbStairs (n + 1) + climbStairs n

/-- The number of stairs Jo climbs -/
def totalStairs : ℕ := 8

theorem jo_stair_climbing :
  climbStairs totalStairs = 81 :=
by
  sorry

end jo_stair_climbing_l1101_110183


namespace fraction_division_specific_fraction_division_l1101_110181

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem specific_fraction_division :
  (3 : ℚ) / 7 / ((5 : ℚ) / 9) = 27 / 35 :=
by sorry

end fraction_division_specific_fraction_division_l1101_110181


namespace sum_of_squares_roots_l1101_110151

theorem sum_of_squares_roots (h : ℝ) : 
  (∃ r s : ℝ, r^2 - 4*h*r - 8 = 0 ∧ s^2 - 4*h*s - 8 = 0 ∧ r^2 + s^2 = 20) → 
  h = 1/2 ∨ h = -1/2 := by
sorry

end sum_of_squares_roots_l1101_110151


namespace lcm_12_18_l1101_110140

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l1101_110140


namespace special_function_zero_location_l1101_110134

/-- A function f satisfying the given conditions -/
structure SpecialFunction (f : ℝ → ℝ) : Prop :=
  (decreasing : ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) < 0)

/-- The theorem statement -/
theorem special_function_zero_location
  (f : ℝ → ℝ) (hf : SpecialFunction f) (a b c d : ℝ)
  (h_order : c < b ∧ b < a)
  (h_product : f a * f b * f c < 0)
  (h_zero : f d = 0) :
  (d < c) ∨ (b < d ∧ d < a) :=
sorry

end special_function_zero_location_l1101_110134


namespace sqrt_49_times_sqrt_25_l1101_110186

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_49_times_sqrt_25_l1101_110186


namespace company_daily_production_l1101_110123

/-- Proves that a company producing enough bottles to fill 2000 cases, 
    where each case holds 25 bottles, produces 50000 bottles daily. -/
theorem company_daily_production 
  (bottles_per_case : ℕ) 
  (cases_per_day : ℕ) 
  (h1 : bottles_per_case = 25)
  (h2 : cases_per_day = 2000) :
  bottles_per_case * cases_per_day = 50000 := by
  sorry

#check company_daily_production

end company_daily_production_l1101_110123


namespace ellipse_foci_distance_l1101_110172

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144

-- Define the distance between foci
def distance_between_foci (eq : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem ellipse_foci_distance :
  distance_between_foci ellipse_equation = 2 * Real.sqrt 7 := by sorry

end ellipse_foci_distance_l1101_110172


namespace parabola_vertex_sum_max_l1101_110115

/-- Given a parabola y = ax^2 + bx + c passing through (0,0), (2T,0), and (2T+1,15),
    where a and T are integers and T ≠ 0, prove that the largest possible value of N is -10,
    where N is the sum of the coordinates of the vertex point. -/
theorem parabola_vertex_sum_max (a T : ℤ) (hT : T ≠ 0) : 
  ∃ (b c : ℤ),
    (0 = c) ∧
    (0 = 4*a*T^2 + 2*b*T + c) ∧
    (15 = a*(2*T+1)^2 + b*(2*T+1) + c) →
    (∀ N : ℤ, N = T - a*T^2 → N ≤ -10) :=
sorry

end parabola_vertex_sum_max_l1101_110115


namespace inequality_sqrt_ratios_l1101_110130

theorem inequality_sqrt_ratios (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end inequality_sqrt_ratios_l1101_110130


namespace foreign_trade_analysis_l1101_110104

-- Define the data points
def x : List ℝ := [1.8, 2.2, 2.6, 3.0]
def y : List ℝ := [2.0, 2.8, 3.2, 4.0]

-- Define the linear correlation function
def linear_correlation (b : ℝ) (x : ℝ) : ℝ := b * x - 0.84

-- Theorem statement
theorem foreign_trade_analysis :
  let x_mean := (List.sum x) / (List.length x : ℝ)
  let y_mean := (List.sum y) / (List.length y : ℝ)
  let b_hat := (y_mean + 0.84) / x_mean
  ∀ (ε : ℝ), ε > 0 →
    (abs (b_hat - 1.6) < ε) ∧
    (abs ((linear_correlation b_hat⁻¹ 6 + 0.84) / b_hat - 4.275) < ε) :=
by sorry

end foreign_trade_analysis_l1101_110104


namespace max_product_sum_300_l1101_110145

theorem max_product_sum_300 : 
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧ 
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end max_product_sum_300_l1101_110145


namespace min_distance_between_points_l1101_110179

noncomputable section

def f (x : ℝ) : ℝ := Real.sin x + (1/6) * x^3
def g (x : ℝ) : ℝ := x - 1

theorem min_distance_between_points (x₁ x₂ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ ≥ 0) (h₃ : f x₁ = g x₂) :
  ∃ (d : ℝ), d = |x₂ - x₁| ∧ d ≥ 1 ∧ 
  (∀ (y₁ y₂ : ℝ), y₁ ≥ 0 → y₂ ≥ 0 → f y₁ = g y₂ → |y₂ - y₁| ≥ d) :=
sorry

end min_distance_between_points_l1101_110179


namespace total_rubber_bands_l1101_110171

theorem total_rubber_bands (harper_bands : ℕ) (difference : ℕ) : 
  harper_bands = 15 → difference = 6 → harper_bands + (harper_bands - difference) = 24 := by
  sorry

end total_rubber_bands_l1101_110171


namespace miniature_toy_height_difference_l1101_110178

/-- Heights of different poodle types -/
structure PoodleHeights where
  standard : ℕ
  miniature : ℕ
  toy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (h : PoodleHeights) : Prop :=
  h.standard = 28 ∧ h.toy = 14 ∧ h.standard = h.miniature + 8

/-- The theorem to be proved -/
theorem miniature_toy_height_difference (h : PoodleHeights) 
  (hc : problem_conditions h) : h.miniature - h.toy = 6 := by
  sorry

end miniature_toy_height_difference_l1101_110178


namespace discount_percentage_l1101_110175

def original_price : ℝ := 6
def num_bags : ℕ := 2
def total_spent : ℝ := 3

theorem discount_percentage : 
  (1 - total_spent / (original_price * num_bags)) * 100 = 75 := by
  sorry

end discount_percentage_l1101_110175


namespace equation_solution_l1101_110195

theorem equation_solution (x k : ℝ) : 
  (7 * x + 2 = 3 * x - 6) ∧ (x + 1 = k) → 3 * k^2 - 1 = 2 := by
  sorry

end equation_solution_l1101_110195


namespace g_behavior_at_infinity_l1101_110131

def g (x : ℝ) := -3 * x^3 + 5 * x^2 + 4

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) :=
by sorry

end g_behavior_at_infinity_l1101_110131


namespace inscribed_rhombus_square_area_l1101_110160

/-- Represents a rhombus inscribed in a square -/
structure InscribedRhombus where
  -- Square side length
  a : ℝ
  -- Distances from square vertices to rhombus vertices
  pb : ℝ
  bq : ℝ
  pr : ℝ
  qs : ℝ
  -- Conditions
  pb_positive : pb > 0
  bq_positive : bq > 0
  pr_positive : pr > 0
  qs_positive : qs > 0
  pb_plus_bq : pb + bq = a
  pr_plus_qs : pr + qs = a

/-- The area of the square given the inscribed rhombus properties -/
def square_area (r : InscribedRhombus) : ℝ :=
  r.a ^ 2

/-- Theorem: The area of the square with the given inscribed rhombus is 40000/58 -/
theorem inscribed_rhombus_square_area :
  ∀ r : InscribedRhombus,
  r.pb = 10 ∧ r.bq = 25 ∧ r.pr = 20 ∧ r.qs = 40 →
  square_area r = 40000 / 58 := by
  sorry

end inscribed_rhombus_square_area_l1101_110160


namespace twins_age_problem_l1101_110154

theorem twins_age_problem :
  ∀ (x y : ℕ),
  x * x = 8 →
  (x + y) * (x + y) = x * x + 17 →
  y = 3 :=
by sorry

end twins_age_problem_l1101_110154


namespace more_freshmen_than_sophomores_l1101_110199

theorem more_freshmen_than_sophomores 
  (total : ℕ) 
  (junior_percent : ℚ) 
  (not_sophomore_percent : ℚ) 
  (seniors : ℕ) 
  (h1 : total = 800)
  (h2 : junior_percent = 22/100)
  (h3 : not_sophomore_percent = 74/100)
  (h4 : seniors = 160)
  : ℕ := by
  sorry

end more_freshmen_than_sophomores_l1101_110199


namespace cubic_expression_evaluation_l1101_110106

theorem cubic_expression_evaluation : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by
  sorry

end cubic_expression_evaluation_l1101_110106


namespace circle_passes_through_points_l1101_110176

/-- The equation of a circle passing through (0, 0), (-2, 3), and (-4, 1) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + (19/5)*x - (9/5)*y = 0

/-- Theorem stating that the circle passes through the required points -/
theorem circle_passes_through_points :
  circle_equation 0 0 ∧ circle_equation (-2) 3 ∧ circle_equation (-4) 1 := by
  sorry

end circle_passes_through_points_l1101_110176


namespace inequality_system_integer_solutions_l1101_110184

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (4 * (x - 1) > 3 * x - 2) ∧ (2 * x - 3 ≤ 5)}
  S = {3, 4} := by
sorry

end inequality_system_integer_solutions_l1101_110184


namespace lexie_crayon_count_l1101_110194

/-- The number of crayons that can fit in each box -/
def crayons_per_box : ℕ := 8

/-- The number of crayon boxes Lexie needs -/
def number_of_boxes : ℕ := 10

/-- The total number of crayons Lexie has -/
def total_crayons : ℕ := crayons_per_box * number_of_boxes

theorem lexie_crayon_count : total_crayons = 80 := by
  sorry

end lexie_crayon_count_l1101_110194


namespace circle_center_and_radius_l1101_110193

/-- Given a circle with diameter endpoints (2, -3) and (8, 5), 
    prove that its center is at (5, 1) and its radius is 5. -/
theorem circle_center_and_radius : 
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (8, 5)
  let center : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  let radius : ℝ := Real.sqrt ((center.1 - a.1)^2 + (center.2 - a.2)^2)
  center = (5, 1) ∧ radius = 5 :=
by sorry

end circle_center_and_radius_l1101_110193


namespace converse_abs_inequality_l1101_110105

theorem converse_abs_inequality (x y : ℝ) : x > |y| → x > y := by sorry

end converse_abs_inequality_l1101_110105


namespace max_score_for_successful_teams_l1101_110135

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  num_teams : Nat
  num_successful_teams : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- The maximum score that can be achieved by the successful teams -/
def max_total_score (t : FootballTournament) : Nat :=
  let internal_matches := t.num_successful_teams * (t.num_successful_teams - 1) / 2
  let external_matches := t.num_successful_teams * (t.num_teams - t.num_successful_teams)
  (internal_matches + external_matches) * t.points_for_win

/-- The theorem stating the maximum integer N for which at least 6 teams can score N points -/
theorem max_score_for_successful_teams (t : FootballTournament) 
    (h1 : t.num_teams = 15)
    (h2 : t.num_successful_teams = 6)
    (h3 : t.points_for_win = 3)
    (h4 : t.points_for_draw = 1)
    (h5 : t.points_for_loss = 0) :
    ∃ (N : Nat), N = 34 ∧ 
    (∀ (M : Nat), (M > N → ¬(t.num_successful_teams * M ≤ max_total_score t))) ∧
    (t.num_successful_teams * N ≤ max_total_score t) := by
  sorry

end max_score_for_successful_teams_l1101_110135


namespace two_children_gender_combinations_l1101_110169

/-- Represents the gender of a child -/
inductive Gender
  | Male
  | Female

/-- Represents a pair of children's genders -/
def ChildPair := Gender × Gender

/-- The set of all possible gender combinations for two children -/
def allGenderCombinations : Set ChildPair :=
  {(Gender.Male, Gender.Male), (Gender.Male, Gender.Female),
   (Gender.Female, Gender.Male), (Gender.Female, Gender.Female)}

/-- Theorem stating that the set of all possible gender combinations
    for two children is equal to the expected set -/
theorem two_children_gender_combinations :
  {pair : ChildPair | True} = allGenderCombinations := by
  sorry

end two_children_gender_combinations_l1101_110169


namespace correct_graph_representation_l1101_110167

/-- Represents a segment of Mike's trip -/
inductive TripSegment
  | CityDriving
  | HighwayDriving
  | Shopping
  | Refueling

/-- Represents the slope of a graph segment -/
inductive Slope
  | Flat
  | Gradual
  | Steep

/-- Represents Mike's trip -/
structure MikeTrip where
  segments : List TripSegment
  shoppingDuration : ℝ
  refuelingDuration : ℝ

/-- Represents a graph of Mike's trip -/
structure TripGraph where
  flatSections : Nat
  slopes : List Slope

/-- The correct graph representation of Mike's trip -/
def correctGraph : TripGraph :=
  { flatSections := 2
  , slopes := [Slope.Gradual, Slope.Steep, Slope.Flat, Slope.Flat, Slope.Steep, Slope.Gradual] }

theorem correct_graph_representation (trip : MikeTrip)
  (h1 : trip.segments = [TripSegment.CityDriving, TripSegment.HighwayDriving, TripSegment.Shopping, TripSegment.Refueling, TripSegment.HighwayDriving, TripSegment.CityDriving])
  (h2 : trip.shoppingDuration = 2)
  (h3 : trip.refuelingDuration = 0.5)
  : TripGraph.flatSections correctGraph = 2 ∧ 
    TripGraph.slopes correctGraph = [Slope.Gradual, Slope.Steep, Slope.Flat, Slope.Flat, Slope.Steep, Slope.Gradual] := by
  sorry

end correct_graph_representation_l1101_110167


namespace x_squared_less_than_abs_x_l1101_110122

theorem x_squared_less_than_abs_x (x : ℝ) :
  x^2 < |x| ↔ (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) := by
  sorry

end x_squared_less_than_abs_x_l1101_110122


namespace polynomial_identity_l1101_110177

/-- 
Given a, b, and c, prove that 
a(b - c)³ + b(c - a)³ + c(a - b)³ + (a - b)²(b - c)²(c - a)² = (a - b)(b - c)(c - a)(a + b + c + abc)
-/
theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + (a - b)^2 * (b - c)^2 * (c - a)^2 = 
  (a - b) * (b - c) * (c - a) * (a + b + c + a * b * c) := by
  sorry

end polynomial_identity_l1101_110177


namespace max_integers_above_18_l1101_110156

/-- Given 5 integers that sum to 17, the maximum number of these integers
    that can be larger than 18 is 2. -/
theorem max_integers_above_18 (a b c d e : ℤ) : 
  a + b + c + d + e = 17 → 
  (∀ k : ℕ, k ≤ 5 → 
    (∃ (S : Finset ℤ), S.card = k ∧ S ⊆ {a, b, c, d, e} ∧ (∀ x ∈ S, x > 18)) →
    k ≤ 2) := by
  sorry

end max_integers_above_18_l1101_110156


namespace minks_set_free_ratio_l1101_110132

/-- Represents the mink coat problem -/
structure MinkCoatProblem where
  skins_per_coat : ℕ
  initial_minks : ℕ
  babies_per_mink : ℕ
  coats_made : ℕ

/-- Calculates the total number of minks -/
def total_minks (p : MinkCoatProblem) : ℕ :=
  p.initial_minks * (1 + p.babies_per_mink)

/-- Calculates the number of minks used for coats -/
def minks_used_for_coats (p : MinkCoatProblem) : ℕ :=
  p.skins_per_coat * p.coats_made

/-- Calculates the number of minks set free -/
def minks_set_free (p : MinkCoatProblem) : ℕ :=
  total_minks p - minks_used_for_coats p

/-- The main theorem stating the ratio of minks set free to total minks -/
theorem minks_set_free_ratio (p : MinkCoatProblem) 
  (h1 : p.skins_per_coat = 15)
  (h2 : p.initial_minks = 30)
  (h3 : p.babies_per_mink = 6)
  (h4 : p.coats_made = 7) :
  minks_set_free p * 2 = total_minks p :=
sorry

end minks_set_free_ratio_l1101_110132


namespace jane_bagels_l1101_110102

theorem jane_bagels (b m : ℕ) : 
  b + m = 5 →
  (75 * b + 50 * m) % 100 = 0 →
  b = 2 :=
by sorry

end jane_bagels_l1101_110102


namespace alice_winning_equivalence_l1101_110170

/-- The game constant k, which is greater than 2 -/
def k : ℕ := sorry

/-- Definition of Alice-winning number -/
def is_alice_winning (n : ℕ) : Prop := sorry

/-- The radical of a number n with respect to k -/
def radical (n : ℕ) : ℕ := sorry

theorem alice_winning_equivalence (l l' : ℕ) 
  (h : ∀ p : ℕ, p.Prime → p ≤ k → (p ∣ l ↔ p ∣ l')) : 
  is_alice_winning l ↔ is_alice_winning l' := by sorry

end alice_winning_equivalence_l1101_110170


namespace infinitely_many_primes_4k_plus_3_l1101_110190

theorem infinitely_many_primes_4k_plus_3 : 
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ Prime p ∧ ∃ k : ℕ, p = 4 * k + 3 :=
sorry

end infinitely_many_primes_4k_plus_3_l1101_110190


namespace one_negative_number_l1101_110101

theorem one_negative_number (numbers : List ℝ := [-2, 1/2, 0, 3]) : 
  (numbers.filter (λ x => x < 0)).length = 1 := by
  sorry

end one_negative_number_l1101_110101


namespace rational_pair_sum_reciprocal_natural_l1101_110161

theorem rational_pair_sum_reciprocal_natural (x y : ℚ) :
  (∃ (m n : ℕ), x + 1 / y = m ∧ y + 1 / x = n) →
  ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 2)) :=
by sorry

end rational_pair_sum_reciprocal_natural_l1101_110161


namespace salad_ratio_l1101_110107

/-- Given a salad with cucumbers and tomatoes, prove the ratio of tomatoes to cucumbers -/
theorem salad_ratio (total : ℕ) (cucumbers : ℕ) (h1 : total = 280) (h2 : cucumbers = 70) :
  (total - cucumbers) / cucumbers = 3 := by
  sorry

end salad_ratio_l1101_110107


namespace right_triangle_hypotenuse_length_l1101_110180

/-- Given a right triangle ABC with legs AB and AC, and points X on AB and Y on AC,
    prove that the hypotenuse BC has length 6√42 under specific conditions. -/
theorem right_triangle_hypotenuse_length 
  (A B C X Y : ℝ × ℝ) -- Points in 2D plane
  (h_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) -- Right angle at A
  (h_X_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2))
  (h_Y_on_AC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Y = (s * C.1 + (1 - s) * A.1, s * C.2 + (1 - s) * A.2))
  (h_AX_XB : dist A X = (1/4) * dist A B)
  (h_AY_YC : dist A Y = (2/3) * dist A C)
  (h_BY : dist B Y = 24)
  (h_CX : dist C X = 18) :
  dist B C = 6 * Real.sqrt 42 := by
  sorry

end right_triangle_hypotenuse_length_l1101_110180


namespace imohkprelim_combinations_l1101_110192

def letter_list : List Char := ['I', 'M', 'O', 'H', 'K', 'P', 'R', 'E', 'L', 'I', 'M']

def count_combinations (letters : List Char) : Nat :=
  let unique_letters := letters.eraseDups
  let combinations_distinct := Nat.choose unique_letters.length 3
  let combinations_with_repeat := 
    (letters.filter (λ c => letters.count c > 1)).eraseDups.length * (unique_letters.length - 1)
  combinations_distinct + combinations_with_repeat

theorem imohkprelim_combinations :
  count_combinations letter_list = 100 := by
  sorry

end imohkprelim_combinations_l1101_110192


namespace triangle_area_proof_l1101_110126

/-- The line equation defining the hypotenuse of the triangle -/
def line_equation (x y : ℝ) : Prop := 3 * x + y = 9

/-- The x-intercept of the line -/
def x_intercept : ℝ := 3

/-- The y-intercept of the line -/
def y_intercept : ℝ := 9

/-- The area of the triangle -/
def triangle_area : ℝ := 13.5

theorem triangle_area_proof :
  triangle_area = (1/2) * x_intercept * y_intercept :=
sorry

end triangle_area_proof_l1101_110126


namespace asymptotes_necessary_not_sufficient_l1101_110187

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x²/a² - y²/b² = 1) -/
  equation : ℝ → ℝ → Prop

/-- Represents the asymptotes of a hyperbola -/
structure Asymptotes where
  /-- The equation of the asymptotes in the form y = ±mx -/
  equation : ℝ → ℝ → Prop

/-- The specific hyperbola C with equation x²/9 - y²/16 = 1 -/
def hyperbola_C : Hyperbola :=
  { equation := fun x y => x^2 / 9 - y^2 / 16 = 1 }

/-- The asymptotes with equation y = ±(4/3)x -/
def asymptotes_C : Asymptotes :=
  { equation := fun x y => y = 4/3 * x ∨ y = -4/3 * x }

/-- Theorem stating that the given asymptote equation is a necessary but not sufficient condition for the hyperbola equation -/
theorem asymptotes_necessary_not_sufficient :
  (∀ x y, hyperbola_C.equation x y → asymptotes_C.equation x y) ∧
  ¬(∀ x y, asymptotes_C.equation x y → hyperbola_C.equation x y) := by
  sorry

end asymptotes_necessary_not_sufficient_l1101_110187


namespace polar_to_rectangular_conversion_l1101_110100

theorem polar_to_rectangular_conversion :
  let r : ℝ := 10
  let θ : ℝ := 5 * Real.pi / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 5 ∧ y = -5 * Real.sqrt 3) := by
  sorry

end polar_to_rectangular_conversion_l1101_110100


namespace toms_calculation_l1101_110166

theorem toms_calculation (y : ℝ) (h : 4 * y + 7 = 39) : (y + 7) * 4 = 60 := by
  sorry

end toms_calculation_l1101_110166


namespace ages_product_l1101_110174

/-- Represents the ages of the individuals in the problem -/
structure Ages where
  thomas : ℕ
  roy : ℕ
  kelly : ℕ
  julia : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.thomas = ages.roy - 6 ∧
  ages.thomas = ages.kelly + 4 ∧
  ages.roy = ages.julia + 8 ∧
  ages.roy = ages.kelly + 4 ∧
  ages.roy + 2 = 3 * (ages.julia + 2) ∧
  ages.thomas + 2 = 2 * (ages.kelly + 2)

/-- The theorem to be proved -/
theorem ages_product (ages : Ages) :
  satisfies_conditions ages →
  (ages.roy + 2) * (ages.kelly + 2) * (ages.thomas + 2) = 576 := by
  sorry

end ages_product_l1101_110174


namespace factorial_fraction_equality_l1101_110128

theorem factorial_fraction_equality : (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 := by
  sorry

end factorial_fraction_equality_l1101_110128


namespace rope_initial_length_l1101_110159

/-- Given a rope cut into pieces, calculate its initial length -/
theorem rope_initial_length
  (num_pieces : ℕ)
  (tied_pieces : ℕ)
  (knot_reduction : ℕ)
  (final_length : ℕ)
  (h1 : num_pieces = 12)
  (h2 : tied_pieces = 3)
  (h3 : knot_reduction = 1)
  (h4 : final_length = 15) :
  (final_length + knot_reduction) * num_pieces = 192 :=
by sorry

end rope_initial_length_l1101_110159


namespace inequality_solution_set_l1101_110188

theorem inequality_solution_set (x : ℝ) : 
  (1 + 2 * (x - 1) ≤ 3) ↔ (x ≤ 2) := by sorry

end inequality_solution_set_l1101_110188


namespace symmetry_coordinates_l1101_110146

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis. -/
def symmetricToYAxis (a b : Point) : Prop :=
  b.x = -a.x ∧ b.y = a.y

/-- The theorem stating that if A(2, -5) is symmetric to B with respect to the y-axis,
    then B has coordinates (-2, -5). -/
theorem symmetry_coordinates :
  let a : Point := ⟨2, -5⟩
  let b : Point := ⟨-2, -5⟩
  symmetricToYAxis a b → b = ⟨-2, -5⟩ := by
  sorry

end symmetry_coordinates_l1101_110146


namespace milford_lake_algae_increase_l1101_110147

theorem milford_lake_algae_increase (original_algae current_algae : ℕ) 
  (h1 : original_algae = 809)
  (h2 : current_algae = 3263) : 
  current_algae - original_algae = 2454 := by
  sorry

end milford_lake_algae_increase_l1101_110147


namespace line_tangent_to_parabola_l1101_110120

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 3 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x ∧ 
   ∀ x' y' : ℝ, y' = 3*x' + c → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  c = 3 := by
sorry

end line_tangent_to_parabola_l1101_110120


namespace jennifer_blue_sweets_l1101_110165

theorem jennifer_blue_sweets (green : ℕ) (yellow : ℕ) (people : ℕ) (sweets_per_person : ℕ) :
  green = 212 →
  yellow = 502 →
  people = 4 →
  sweets_per_person = 256 →
  people * sweets_per_person - (green + yellow) = 310 := by
  sorry

end jennifer_blue_sweets_l1101_110165


namespace ball_count_theorem_l1101_110109

/-- Represents the number of balls of each color in a jar -/
structure BallCount where
  white : ℕ
  red : ℕ
  blue : ℕ

/-- Checks if the given ball count satisfies the ratio 4:3:2 for white:red:blue -/
def satisfiesRatio (bc : BallCount) : Prop :=
  4 * bc.red = 3 * bc.white ∧ 4 * bc.blue = 2 * bc.white

theorem ball_count_theorem (bc : BallCount) 
  (ratio_satisfied : satisfiesRatio bc) 
  (white_count : bc.white = 16) : 
  bc.red = 12 ∧ bc.blue = 8 := by
  sorry

#check ball_count_theorem

end ball_count_theorem_l1101_110109


namespace remainder_13754_div_11_l1101_110163

theorem remainder_13754_div_11 : 13754 % 11 = 4 := by
  sorry

end remainder_13754_div_11_l1101_110163


namespace binary_10011_equals_19_l1101_110124

/-- Convert a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 19 -/
def binary_19 : List Bool := [true, true, false, false, true]

/-- Theorem stating that the binary number 10011 is equal to the decimal number 19 -/
theorem binary_10011_equals_19 : binary_to_decimal binary_19 = 19 := by
  sorry

end binary_10011_equals_19_l1101_110124


namespace congruence_and_divisibility_solutions_l1101_110137

theorem congruence_and_divisibility_solutions : 
  {x : ℤ | x^3 ≡ -1 [ZMOD 7] ∧ (7 : ℤ) ∣ (x^2 - x + 1)} = {3, 5} := by
  sorry

end congruence_and_divisibility_solutions_l1101_110137


namespace diagonal_of_square_l1101_110118

theorem diagonal_of_square (side_length : ℝ) (h : side_length = 10) :
  let diagonal := Real.sqrt (2 * side_length ^ 2)
  diagonal = 10 * Real.sqrt 2 :=
by sorry

end diagonal_of_square_l1101_110118


namespace complement_of_A_in_I_l1101_110196

def I : Set ℕ := {x | 0 < x ∧ x < 6}
def A : Set ℕ := {1, 2, 3}

theorem complement_of_A_in_I :
  (I \ A) = {4, 5} := by
  sorry

end complement_of_A_in_I_l1101_110196


namespace green_shirt_pairs_l1101_110189

theorem green_shirt_pairs (total_students : ℕ) (blue_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (blue_pairs : ℕ) :
  total_students = 132 →
  blue_students = 65 →
  green_students = 67 →
  total_pairs = 66 →
  blue_pairs = 29 →
  blue_students + green_students = total_students →
  ∃ (green_pairs : ℕ), green_pairs = 30 ∧ 
    blue_pairs + green_pairs + (total_students - 2 * (blue_pairs + green_pairs)) / 2 = total_pairs :=
by sorry

end green_shirt_pairs_l1101_110189


namespace max_value_of_f_l1101_110148

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem max_value_of_f :
  ∃ (m : ℝ), m = 9 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 → f x ≤ m :=
by sorry

end max_value_of_f_l1101_110148


namespace sum_abs_coeff_2x_minus_1_pow_5_l1101_110168

/-- The sum of absolute values of coefficients (excluding constant term) 
    in the expansion of (2x-1)^5 is 242 -/
theorem sum_abs_coeff_2x_minus_1_pow_5 :
  let f : ℝ → ℝ := fun x ↦ (2*x - 1)^5
  ∃ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
    (∀ x, f x = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) ∧
    |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 242 :=
by sorry

end sum_abs_coeff_2x_minus_1_pow_5_l1101_110168


namespace school_population_l1101_110142

theorem school_population (b g t : ℕ) : 
  b = 4 * g → g = 5 * t → b + g + t = 26 * t := by
  sorry

end school_population_l1101_110142


namespace james_sheets_used_l1101_110125

/-- The number of books James prints -/
def num_books : ℕ := 2

/-- The number of pages in each book -/
def pages_per_book : ℕ := 600

/-- The number of pages printed on one side of a sheet -/
def pages_per_side : ℕ := 4

/-- Whether the printing is double-sided -/
def is_double_sided : Bool := true

/-- Calculate the total number of pages to be printed -/
def total_pages : ℕ := num_books * pages_per_book

/-- Calculate the number of pages that can be printed on a single sheet -/
def pages_per_sheet : ℕ := if is_double_sided then 2 * pages_per_side else pages_per_side

/-- The number of sheets of paper James uses -/
def sheets_used : ℕ := total_pages / pages_per_sheet

theorem james_sheets_used : sheets_used = 150 := by
  sorry

end james_sheets_used_l1101_110125


namespace square_area_error_l1101_110121

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
  sorry

end square_area_error_l1101_110121


namespace base9_734_equals_base3_211110_l1101_110136

/-- Converts a digit from base 9 to base 3 --/
def base9_to_base3 (d : ℕ) : ℕ := sorry

/-- Converts a number from base 9 to base 3 --/
def convert_base9_to_base3 (n : ℕ) : ℕ := sorry

/-- The main theorem stating that 734 in base 9 is equal to 211110 in base 3 --/
theorem base9_734_equals_base3_211110 :
  convert_base9_to_base3 734 = 211110 := by sorry

end base9_734_equals_base3_211110_l1101_110136


namespace train_average_speed_l1101_110113

-- Define the points and distances
def x : ℝ := 0
def y : ℝ := sorry
def z : ℝ := sorry

-- Define the speeds
def speed_xy : ℝ := 300
def speed_yz : ℝ := 100

-- State the theorem
theorem train_average_speed :
  -- Conditions
  (y - x = 2 * (z - y)) →  -- Distance from x to y is twice the distance from y to z
  -- Conclusion
  (z - x) / ((y - x) / speed_xy + (z - y) / speed_yz) = 180 :=
by
  sorry

end train_average_speed_l1101_110113


namespace cos_difference_from_sum_l1101_110117

theorem cos_difference_from_sum (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
  sorry

end cos_difference_from_sum_l1101_110117


namespace sum_equals_five_l1101_110116

/-- The mapping f that transforms (x, y) to (x, x+y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 + p.2)

/-- Theorem stating that a + b = 5 given the conditions -/
theorem sum_equals_five (a b : ℝ) (h : f (a, b) = (1, 3)) : a + b = 5 := by
  sorry

end sum_equals_five_l1101_110116


namespace dilution_problem_dilution_solution_l1101_110153

theorem dilution_problem (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : Prop :=
  initial_volume = 12 ∧ 
  initial_concentration = 0.6 ∧
  target_concentration = 0.4 ∧
  water_added = 6 →
  initial_volume * initial_concentration = 
    (initial_volume + water_added) * target_concentration

theorem dilution_solution : 
  ∃ (water_added : ℝ), dilution_problem 12 0.6 0.4 water_added :=
by
  sorry

end dilution_problem_dilution_solution_l1101_110153


namespace fraction_simplification_l1101_110133

theorem fraction_simplification : 
  (4 : ℝ) / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end fraction_simplification_l1101_110133


namespace fraction_addition_and_multiplication_l1101_110155

theorem fraction_addition_and_multiplication :
  (7 / 12 + 3 / 8) * 2 / 3 = 23 / 36 := by
  sorry

end fraction_addition_and_multiplication_l1101_110155


namespace max_temp_difference_example_l1101_110158

/-- The maximum temperature difference given the highest and lowest temperatures -/
def max_temp_difference (highest lowest : ℤ) : ℤ :=
  highest - lowest

/-- Theorem: The maximum temperature difference is 20℃ given the highest temperature of 18℃ and lowest temperature of -2℃ -/
theorem max_temp_difference_example : max_temp_difference 18 (-2) = 20 := by
  sorry

end max_temp_difference_example_l1101_110158


namespace average_increase_is_four_l1101_110150

/-- Represents a cricket player's performance --/
structure CricketPerformance where
  innings : ℕ
  totalRuns : ℕ
  newInningsRuns : ℕ

/-- Calculates the average runs per innings --/
def average (cp : CricketPerformance) : ℚ :=
  cp.totalRuns / cp.innings

/-- Calculates the new average after playing an additional innings --/
def newAverage (cp : CricketPerformance) : ℚ :=
  (cp.totalRuns + cp.newInningsRuns) / (cp.innings + 1)

/-- Theorem: The increase in average is 4 runs --/
theorem average_increase_is_four (cp : CricketPerformance) 
  (h1 : cp.innings = 10)
  (h2 : average cp = 18)
  (h3 : cp.newInningsRuns = 62) : 
  newAverage cp - average cp = 4 := by
  sorry


end average_increase_is_four_l1101_110150


namespace quadratic_function_properties_l1101_110185

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem statement
theorem quadratic_function_properties :
  (f (-1) = 0 ∧ f 3 = 0 ∧ f 0 = -3) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 4 → f x ≤ 2*m) ↔ 5/2 ≤ m) :=
by sorry


end quadratic_function_properties_l1101_110185


namespace seashell_collection_l1101_110143

theorem seashell_collection (current : ℕ) (target : ℕ) (additional : ℕ) : 
  current = 19 → target = 25 → current + additional = target → additional = 6 := by
  sorry

end seashell_collection_l1101_110143


namespace possible_two_black_one_white_l1101_110197

/-- Represents the possible marble replacement operations -/
inductive Operation
  | replaceThreeBlackWithTwoBlack
  | replaceTwoBlackOneWhiteWithTwoWhite
  | replaceOneBlackTwoWhiteWithOneBlackOneWhite
  | replaceThreeWhiteWithTwoBlack

/-- Represents the state of the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Applies a single operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.replaceThreeBlackWithTwoBlack =>
      UrnState.mk state.white (state.black - 1)
  | Operation.replaceTwoBlackOneWhiteWithTwoWhite =>
      UrnState.mk (state.white + 1) (state.black - 2)
  | Operation.replaceOneBlackTwoWhiteWithOneBlackOneWhite =>
      UrnState.mk (state.white - 1) state.black
  | Operation.replaceThreeWhiteWithTwoBlack =>
      UrnState.mk (state.white - 3) (state.black + 2)

/-- Theorem: It is possible to reach a state of 2 black marbles and 1 white marble -/
theorem possible_two_black_one_white :
  ∃ (operations : List Operation),
    let initial_state := UrnState.mk 150 200
    let final_state := operations.foldl applyOperation initial_state
    final_state.white = 1 ∧ final_state.black = 2 :=
  sorry


end possible_two_black_one_white_l1101_110197


namespace rhombus_area_l1101_110198

/-- The area of a rhombus with side length 13 and one diagonal 24 is 120 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (area : ℝ) : 
  side = 13 → diagonal1 = 24 → area = (diagonal1 * (2 * Real.sqrt (side^2 - (diagonal1/2)^2))) / 2 → area = 120 := by
  sorry

end rhombus_area_l1101_110198


namespace min_value_of_z_l1101_110157

theorem min_value_of_z (x y : ℝ) (h1 : x - 1 ≥ 0) (h2 : 2 * x - y - 1 ≤ 0) (h3 : x + y - 3 ≤ 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≥ -1 ∧ ∀ (w : ℝ), w = x - y → w ≥ z :=
sorry

end min_value_of_z_l1101_110157


namespace largest_whole_number_nine_times_less_than_150_l1101_110110

theorem largest_whole_number_nine_times_less_than_150 : 
  ∃ (x : ℕ), x = 16 ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) := by
  sorry

end largest_whole_number_nine_times_less_than_150_l1101_110110


namespace finite_valid_combinations_l1101_110182

/-- Represents the number of banknotes of each denomination --/
structure Banknotes :=
  (hun : Nat)
  (fif : Nat)
  (twe : Nat)
  (ten : Nat)

/-- The total value of a set of banknotes in yuan --/
def totalValue (b : Banknotes) : Nat :=
  100 * b.hun + 50 * b.fif + 20 * b.twe + 10 * b.ten

/-- The available banknotes --/
def availableBanknotes : Banknotes :=
  ⟨1, 2, 5, 10⟩

/-- A valid combination of banknotes is one that sums to 200 yuan and doesn't exceed the available banknotes --/
def isValidCombination (b : Banknotes) : Prop :=
  totalValue b = 200 ∧
  b.hun ≤ availableBanknotes.hun ∧
  b.fif ≤ availableBanknotes.fif ∧
  b.twe ≤ availableBanknotes.twe ∧
  b.ten ≤ availableBanknotes.ten

theorem finite_valid_combinations :
  ∃ (n : Nat), ∃ (combinations : Finset Banknotes),
    combinations.card = n ∧
    (∀ b ∈ combinations, isValidCombination b) ∧
    (∀ b, isValidCombination b → b ∈ combinations) :=
by sorry

end finite_valid_combinations_l1101_110182


namespace circle_bisection_l1101_110103

/-- Circle represented by its equation -/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- A circle bisects another circle if the line through their intersection points passes through the center of the bisected circle -/
def bisects (c1 c2 : Circle) : Prop :=
  ∃ (l : ℝ → ℝ → Prop), (∀ x y, l x y ↔ c1.equation x y ∧ c2.equation x y) ∧
                         l c2.center.1 c2.center.2

theorem circle_bisection (a b : ℝ) :
  let c1 : Circle := ⟨(a, b), λ x y => (x - a)^2 + (y - b)^2 = b^2 + 1⟩
  let c2 : Circle := ⟨(-1, -1), λ x y => (x + 1)^2 + (y + 1)^2 = 4⟩
  bisects c1 c2 → a^2 + 2*a + 2*b + 5 = 0 :=
by sorry

end circle_bisection_l1101_110103


namespace no_prime_roots_for_quadratic_l1101_110111

theorem no_prime_roots_for_quadratic : ¬∃ k : ℤ, ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p : ℤ) + q = 57 ∧ 
  (p : ℤ) * q = k ∧
  (p : ℤ) * q = 57 * (p + q) - (p^2 + q^2) := by
  sorry

end no_prime_roots_for_quadratic_l1101_110111


namespace simplify_expression_l1101_110191

theorem simplify_expression : (2^10 + 7^5) * (2^3 - (-2)^3)^8 = 76600653103936 := by
  sorry

end simplify_expression_l1101_110191


namespace number_problem_l1101_110144

theorem number_problem (x : ℝ) : (4/5 * x) + 16 = 0.9 * 40 → x = 25 := by
  sorry

end number_problem_l1101_110144


namespace birthday_dinner_cost_l1101_110129

theorem birthday_dinner_cost (num_people : ℕ) (meal_cost drink_cost dessert_cost : ℚ) :
  num_people = 5 →
  meal_cost = 12 →
  drink_cost = 3 →
  dessert_cost = 5 →
  (num_people : ℚ) * (meal_cost + drink_cost + dessert_cost) = 100 := by
  sorry

end birthday_dinner_cost_l1101_110129


namespace necessary_implies_sufficient_l1101_110141

theorem necessary_implies_sufficient (A B : Prop) :
  (A → B) → (A → B) :=
by
  sorry

end necessary_implies_sufficient_l1101_110141


namespace XY₂_atomic_numbers_l1101_110114

/-- Represents an element in the periodic table -/
structure Element where
  atomic_number : ℕ
  charge : ℤ
  group : ℕ

/-- Represents an ionic compound -/
structure IonicCompound where
  metal : Element
  nonmetal : Element
  metal_count : ℕ
  nonmetal_count : ℕ

/-- The XY₂ compound -/
def XY₂ : IonicCompound :=
  { metal := { atomic_number := 12, charge := 2, group := 2 },
    nonmetal := { atomic_number := 9, charge := -1, group := 17 },
    metal_count := 1,
    nonmetal_count := 2 }

theorem XY₂_atomic_numbers :
  XY₂.metal.atomic_number = 12 ∧ XY₂.nonmetal.atomic_number = 9 :=
by sorry

end XY₂_atomic_numbers_l1101_110114


namespace fibFactLastTwoDigitsSum_l1101_110108

/-- The Fibonacci Factorial Series up to 144 -/
def fibFactSeries : List ℕ := [1, 1, 2, 3, 8, 13, 21, 34, 55, 89, 144]

/-- Function to calculate the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Function to get the last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ :=
  n % 100

/-- Theorem stating that the sum of the last two digits of the Fibonacci Factorial Series is 30 -/
theorem fibFactLastTwoDigitsSum :
  (fibFactSeries.map (λ n => lastTwoDigits (factorial n))).sum = 30 := by
  sorry

/-- Lemma stating that factorials of numbers greater than 10 end with 00 -/
lemma factorialEndsWith00 (n : ℕ) (h : n > 10) :
  lastTwoDigits (factorial n) = 0 := by
  sorry

end fibFactLastTwoDigitsSum_l1101_110108


namespace absolute_difference_m_n_l1101_110162

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem absolute_difference_m_n (m n : ℝ) 
  (h : (m + 2 * i) / i = n + i) : 
  |m - n| = 3 := by sorry

end absolute_difference_m_n_l1101_110162


namespace sum_of_squares_of_coefficients_is_79_l1101_110152

/-- The expression to be simplified -/
def expression (y : ℝ) : ℝ := 5 * (y^2 - 3*y + 3) - 6 * (y^3 - 2*y + 2)

/-- The sum of squares of coefficients of the simplified expression -/
def sum_of_squares_of_coefficients : ℕ := 79

/-- Theorem stating that the sum of squares of coefficients of the simplified expression is 79 -/
theorem sum_of_squares_of_coefficients_is_79 : 
  sum_of_squares_of_coefficients = 79 := by sorry

end sum_of_squares_of_coefficients_is_79_l1101_110152


namespace logan_gas_budget_l1101_110164

/-- Calculates the amount Logan can spend on gas annually --/
def gas_budget (current_income rent groceries desired_savings income_increase : ℕ) : ℕ :=
  (current_income + income_increase) - (rent + groceries + desired_savings)

/-- Proves that Logan's gas budget is $8,000 given his financial constraints --/
theorem logan_gas_budget :
  gas_budget 65000 20000 5000 42000 10000 = 8000 := by
  sorry

end logan_gas_budget_l1101_110164


namespace limit_two_x_sin_x_over_one_minus_cos_x_l1101_110149

/-- The limit of (2x sin x) / (1 - cos x) as x approaches 0 is equal to 4 -/
theorem limit_two_x_sin_x_over_one_minus_cos_x : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → |((2 * x * Real.sin x) / (1 - Real.cos x)) - 4| < ε :=
sorry

end limit_two_x_sin_x_over_one_minus_cos_x_l1101_110149


namespace root_product_theorem_l1101_110127

-- Define the polynomial h(y)
def h (y : ℝ) : ℝ := y^5 - y^3 + 2

-- Define the function k(y)
def k (y : ℝ) : ℝ := y^2 - 3

-- State the theorem
theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ : ℝ) :
  h y₁ = 0 → h y₂ = 0 → h y₃ = 0 → h y₄ = 0 → h y₅ = 0 →
  k y₁ * k y₂ * k y₃ * k y₄ * k y₅ = 104 := by
  sorry

end root_product_theorem_l1101_110127


namespace students_using_red_l1101_110138

/-- Given a group of students painting a picture, calculate the number using red color. -/
theorem students_using_red (total green both : ℕ) (h1 : total = 70) (h2 : green = 52) (h3 : both = 38) :
  total = green + (green + both - total) - both → green + both - total = 56 := by
  sorry

end students_using_red_l1101_110138
