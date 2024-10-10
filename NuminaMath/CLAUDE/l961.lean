import Mathlib

namespace bijective_if_injective_or_surjective_finite_sets_l961_96177

theorem bijective_if_injective_or_surjective_finite_sets
  {X Y : Type} [Fintype X] [Fintype Y]
  (h_card_eq : Fintype.card X = Fintype.card Y)
  (f : X → Y)
  (h_inj_or_surj : Function.Injective f ∨ Function.Surjective f) :
  Function.Bijective f :=
sorry

end bijective_if_injective_or_surjective_finite_sets_l961_96177


namespace matrix_not_invertible_iff_l961_96101

def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![1 + x, 7;
     3 - x, 8]

theorem matrix_not_invertible_iff (x : ℚ) :
  ¬(Matrix.det (matrix x) ≠ 0) ↔ x = 13/15 := by sorry

end matrix_not_invertible_iff_l961_96101


namespace sqrt_sum_squares_ge_sum_over_sqrt2_l961_96180

theorem sqrt_sum_squares_ge_sum_over_sqrt2 (a b : ℝ) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end sqrt_sum_squares_ge_sum_over_sqrt2_l961_96180


namespace correct_urea_decomposing_bacteria_culture_l961_96157

-- Define the types of culture media
inductive CultureMedium
| SelectiveNitrogen
| IdentificationPhenolRed

-- Define the process of bacterial culture
def BacterialCulture := List CultureMedium

-- Define the property of being a correct culture process
def IsCorrectCulture (process : BacterialCulture) : Prop :=
  process = [CultureMedium.SelectiveNitrogen, CultureMedium.IdentificationPhenolRed]

-- Theorem: The correct culture process for urea-decomposing bacteria
theorem correct_urea_decomposing_bacteria_culture :
  IsCorrectCulture [CultureMedium.SelectiveNitrogen, CultureMedium.IdentificationPhenolRed] :=
by sorry

end correct_urea_decomposing_bacteria_culture_l961_96157


namespace first_graders_count_l961_96121

/-- The number of Kindergartners -/
def kindergartners : ℕ := 101

/-- The cost of an orange shirt for Kindergartners -/
def orange_shirt_cost : ℚ := 29/5

/-- The cost of a yellow shirt for first graders -/
def yellow_shirt_cost : ℚ := 5

/-- The number of second graders -/
def second_graders : ℕ := 107

/-- The cost of a blue shirt for second graders -/
def blue_shirt_cost : ℚ := 28/5

/-- The number of third graders -/
def third_graders : ℕ := 108

/-- The cost of a green shirt for third graders -/
def green_shirt_cost : ℚ := 21/4

/-- The total amount spent by the P.T.O. -/
def total_spent : ℚ := 2317

/-- The number of first graders wearing yellow shirts -/
def first_graders : ℕ := 113

theorem first_graders_count : 
  first_graders * yellow_shirt_cost + 
  kindergartners * orange_shirt_cost + 
  second_graders * blue_shirt_cost + 
  third_graders * green_shirt_cost = total_spent :=
sorry

end first_graders_count_l961_96121


namespace factorization_equality_l961_96150

theorem factorization_equality (x : ℝ) : x * (x - 2) + 1 = (x - 1)^2 := by
  sorry

end factorization_equality_l961_96150


namespace simplify_and_evaluate_l961_96130

theorem simplify_and_evaluate : 
  let x : ℚ := -1
  (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - x)) = -1 :=
by sorry

end simplify_and_evaluate_l961_96130


namespace intersection_of_two_lines_l961_96189

/-- The intersection point of two lines in a 2D plane -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Checks if a point satisfies the equation of a line -/
def satisfiesLine (p : IntersectionPoint) (a b c : ℝ) : Prop :=
  a * p.x + b * p.y = c

/-- The unique intersection point of two lines -/
def uniqueIntersection (p : IntersectionPoint) (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  satisfiesLine p a1 b1 c1 ∧
  satisfiesLine p a2 b2 c2 ∧
  ∀ q : IntersectionPoint, satisfiesLine q a1 b1 c1 ∧ satisfiesLine q a2 b2 c2 → q = p

theorem intersection_of_two_lines :
  uniqueIntersection ⟨3, -1⟩ 2 (-1) 7 3 2 7 :=
sorry

end intersection_of_two_lines_l961_96189


namespace cumulonimbus_cloud_count_l961_96115

theorem cumulonimbus_cloud_count :
  ∀ (cirrus cumulus cumulonimbus : ℕ),
    cirrus = 4 * cumulus →
    cumulus = 12 * cumulonimbus →
    cumulonimbus > 0 →
    cirrus = 144 →
    cumulonimbus = 3 := by
  sorry

end cumulonimbus_cloud_count_l961_96115


namespace symmetric_sine_value_l961_96109

/-- Given a function f(x) = 2sin(wx + φ) that is symmetric about x = π/6,
    prove that f(π/6) is either -2 or 2. -/
theorem symmetric_sine_value (w φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (w * x + φ)
  (∀ x, f (π/6 + x) = f (π/6 - x)) →
  f (π/6) = -2 ∨ f (π/6) = 2 :=
by sorry

end symmetric_sine_value_l961_96109


namespace smallest_factorization_coefficient_factorization_exists_smallest_b_is_85_l961_96128

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ r s : ℤ, x^2 + b*x + 1800 = (x + r) * (x + s)) →
  b ≥ 85 :=
by sorry

theorem factorization_exists : 
  ∃ r s : ℤ, x^2 + 85*x + 1800 = (x + r) * (x + s) :=
by sorry

theorem smallest_b_is_85 : 
  (∃ r s : ℤ, x^2 + 85*x + 1800 = (x + r) * (x + s)) ∧
  (∀ b : ℕ, b < 85 → ¬(∃ r s : ℤ, x^2 + b*x + 1800 = (x + r) * (x + s))) :=
by sorry

end smallest_factorization_coefficient_factorization_exists_smallest_b_is_85_l961_96128


namespace initial_players_count_l961_96106

/-- Represents a round-robin chess tournament. -/
structure ChessTournament where
  initial_players : ℕ
  matches_played : ℕ
  dropped_players : ℕ
  matches_per_dropped : ℕ

/-- Calculates the total number of matches in a round-robin tournament. -/
def total_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the initial number of players in the tournament. -/
theorem initial_players_count (t : ChessTournament) 
  (h1 : t.matches_played = 84)
  (h2 : t.dropped_players = 2)
  (h3 : t.matches_per_dropped = 3) :
  t.initial_players = 15 := by
  sorry

/-- The specific tournament instance described in the problem. -/
def problem_tournament : ChessTournament := {
  initial_players := 15,  -- This is what we're proving
  matches_played := 84,
  dropped_players := 2,
  matches_per_dropped := 3
}

end initial_players_count_l961_96106


namespace max_k_inequality_l961_96119

theorem max_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ k : ℝ, k = 100 ∧ 
  (∀ k' : ℝ, (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → 
    k' * a' * b' * c' / (a' + b' + c') ≤ (a' + b')^2 + (a' + b' + 4*c')^2) → 
  k' ≤ k) :=
by sorry

end max_k_inequality_l961_96119


namespace power_of_three_mod_five_l961_96159

theorem power_of_three_mod_five : 3^2040 % 5 = 1 := by
  sorry

end power_of_three_mod_five_l961_96159


namespace smallest_n_divisible_by_seven_l961_96182

theorem smallest_n_divisible_by_seven (n : ℕ) : 
  (n > 50000 ∧ 
   (9 * (n - 2)^6 - n^3 + 20*n - 48) % 7 = 0 ∧
   ∀ m, 50000 < m ∧ m < n → (9 * (m - 2)^6 - m^3 + 20*m - 48) % 7 ≠ 0) →
  n = 50001 := by
sorry

end smallest_n_divisible_by_seven_l961_96182


namespace sticker_distribution_equivalence_l961_96107

/-- The number of ways to distribute n identical objects into k distinct containers --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute s identical stickers among p identical sheets,
    where each sheet must have at least 1 sticker --/
def distribute_stickers (s p : ℕ) : ℕ := stars_and_bars (s - p) p

theorem sticker_distribution_equivalence :
  distribute_stickers 10 5 = stars_and_bars 5 5 :=
by sorry

end sticker_distribution_equivalence_l961_96107


namespace union_A_B_disjoint_A_B_l961_96155

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {y | a < y ∧ y ≤ a + 1}

-- Theorem 1: Union of A and B when a = 3/2
theorem union_A_B : A ∪ B (3/2) = {x | 1 < x ∧ x ≤ 5/2} := by sorry

-- Theorem 2: Condition for A and B to be disjoint
theorem disjoint_A_B : ∀ a : ℝ, A ∩ B a = ∅ ↔ a ≥ 2 ∨ a ≤ 0 := by sorry

end union_A_B_disjoint_A_B_l961_96155


namespace triangle_area_l961_96179

/-- Given a triangle ABC with circumcircle diameter 4√3/3, angle C = 60°, and a + b = ab, 
    the area of the triangle is √3. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  (∃ (R : ℝ), 2 * R = 4 * Real.sqrt 3 / 3) →  -- Circumcircle diameter condition
  C = π / 3 →                                 -- Angle C = 60°
  a + b = a * b →                             -- Given condition
  (∃ (S : ℝ), S = a * b * Real.sin C / 2 ∧ S = Real.sqrt 3) := by
  sorry

end triangle_area_l961_96179


namespace spirits_bottle_cost_l961_96190

/-- Calculates the cost of a bottle of spirits given the number of servings,
    price per serving, and profit per bottle. -/
def bottle_cost (servings : ℕ) (price_per_serving : ℚ) (profit : ℚ) : ℚ :=
  servings * price_per_serving - profit

/-- Proves that the cost of a bottle of spirits is $30.00 under given conditions. -/
theorem spirits_bottle_cost :
  bottle_cost 16 8 98 = 30 := by
  sorry

end spirits_bottle_cost_l961_96190


namespace maritime_silk_road_analysis_l961_96105

/-- Represents the Maritime Silk Road -/
structure MaritimeSilkRoad where
  economic_exchange : Bool
  cultural_exchange : Bool

/-- Represents the discussion method used -/
structure DiscussionMethod where
  theory_of_two_points : Bool
  theory_of_emphasis : Bool

/-- Represents the viewpoints in the discussion -/
inductive Viewpoint
  | economy_first
  | culture_first

/-- Theorem stating the analysis of the Maritime Silk Road discussion -/
theorem maritime_silk_road_analysis 
  (msr : MaritimeSilkRoad) 
  (method : DiscussionMethod) 
  (viewpoints : List Viewpoint) :
  msr.economic_exchange = true →
  msr.cultural_exchange = true →
  method.theory_of_two_points = true →
  method.theory_of_emphasis = true →
  viewpoints.length > 1 →
  (∃ (analysis : Bool), 
    analysis = true ↔ 
      (∃ (social_existence_consciousness : Bool) (culture_economy : Bool),
        social_existence_consciousness = true ∧ 
        culture_economy = true)) :=
by sorry

end maritime_silk_road_analysis_l961_96105


namespace jack_hunting_problem_l961_96170

theorem jack_hunting_problem (hunts_per_month : ℕ) (season_length : ℚ) 
  (deer_weight : ℕ) (kept_weight_ratio : ℚ) (total_kept_weight : ℕ) :
  hunts_per_month = 6 →
  season_length = 1 / 4 →
  deer_weight = 600 →
  kept_weight_ratio = 1 / 2 →
  total_kept_weight = 10800 →
  (total_kept_weight / kept_weight_ratio / deer_weight) / (hunts_per_month * (season_length * 12)) = 2 := by
sorry

end jack_hunting_problem_l961_96170


namespace jack_marbles_l961_96110

theorem jack_marbles (initial : ℕ) (shared : ℕ) (final : ℕ) :
  initial = 62 →
  shared = 33 →
  final = initial - shared →
  final = 29 :=
by sorry

end jack_marbles_l961_96110


namespace chinese_remainder_theorem_application_l961_96140

theorem chinese_remainder_theorem_application : 
  ∀ x : ℕ, 1000 < x ∧ x < 4000 ∧ 
    x % 11 = 2 ∧ x % 13 = 12 ∧ x % 19 = 18 ↔ 
    x = 1234 ∨ x = 3951 := by sorry

end chinese_remainder_theorem_application_l961_96140


namespace arrangement_of_digits_and_blanks_l961_96174

theorem arrangement_of_digits_and_blanks : 
  let n : ℕ := 6  -- total number of boxes
  let k : ℕ := 4  -- number of distinct digits
  let b : ℕ := 2  -- number of blank spaces
  n! / b! = 360 := by
sorry

end arrangement_of_digits_and_blanks_l961_96174


namespace triangle_expression_value_l961_96103

theorem triangle_expression_value (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + x*y + y^2/3 = 25)
  (eq2 : y^2/3 + z^2 = 9)
  (eq3 : z^2 + z*x + x^2 = 16) :
  x*y + 2*y*z + 3*z*x = 24*Real.sqrt 3 := by
sorry

end triangle_expression_value_l961_96103


namespace factorization_validity_l961_96197

theorem factorization_validity (x y : ℝ) : 5 * x^2 * y - 10 * x * y^2 = 5 * x * y * (x - 2 * y) := by
  sorry

end factorization_validity_l961_96197


namespace prob_two_private_teams_prob_distribution_ξ_expectation_ξ_l961_96139

/-- The number of guided tour teams -/
def guided_teams : ℕ := 6

/-- The number of private tour teams -/
def private_teams : ℕ := 3

/-- The total number of teams -/
def total_teams : ℕ := guided_teams + private_teams

/-- The number of draws with replacement -/
def num_draws : ℕ := 4

/-- The random variable representing the number of private teams drawn -/
def ξ : ℕ → ℝ := sorry

/-- The probability of drawing two private tour teams when selecting two numbers at a time -/
theorem prob_two_private_teams : 
  (Nat.choose private_teams 2 : ℚ) / (Nat.choose total_teams 2) = 1 / 12 := by sorry

/-- The probability distribution of ξ -/
theorem prob_distribution_ξ : 
  (ξ 0 = 16 / 81) ∧ 
  (ξ 1 = 32 / 81) ∧ 
  (ξ 2 = 8 / 27) ∧ 
  (ξ 3 = 8 / 81) ∧ 
  (ξ 4 = 1 / 81) := by sorry

/-- The mathematical expectation of ξ -/
theorem expectation_ξ : 
  (0 * ξ 0 + 1 * ξ 1 + 2 * ξ 2 + 3 * ξ 3 + 4 * ξ 4 : ℝ) = 4 / 3 := by sorry

end prob_two_private_teams_prob_distribution_ξ_expectation_ξ_l961_96139


namespace solution_of_equation_l961_96181

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := 3 * a - 4 * b

-- State the theorem
theorem solution_of_equation :
  ∃ x : ℝ, customOp 2 (customOp 2 x) = customOp 1 x ∧ x = 21 / 20 := by
  sorry

end solution_of_equation_l961_96181


namespace probability_five_green_marbles_l961_96117

def total_marbles : ℕ := 12
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 4
def num_draws : ℕ := 8
def num_green_draws : ℕ := 5

def prob_green : ℚ := green_marbles / total_marbles
def prob_purple : ℚ := purple_marbles / total_marbles

theorem probability_five_green_marbles :
  (Nat.choose num_draws num_green_draws : ℚ) * 
  (prob_green ^ num_green_draws) * 
  (prob_purple ^ (num_draws - num_green_draws)) = 1792 / 6561 := by
  sorry

end probability_five_green_marbles_l961_96117


namespace smallest_integer_gcf_24_is_4_l961_96162

theorem smallest_integer_gcf_24_is_4 : 
  ∀ n : ℕ, n > 100 → Nat.gcd n 24 = 4 → n ≥ 104 :=
by sorry

end smallest_integer_gcf_24_is_4_l961_96162


namespace empty_vessel_mass_l961_96123

/-- The mass of an empty vessel given the masses when filled with kerosene and water, and the densities of kerosene and water. -/
theorem empty_vessel_mass
  (mass_with_kerosene : ℝ)
  (mass_with_water : ℝ)
  (density_water : ℝ)
  (density_kerosene : ℝ)
  (h1 : mass_with_kerosene = 31)
  (h2 : mass_with_water = 33)
  (h3 : density_water = 1000)
  (h4 : density_kerosene = 800) :
  ∃ (empty_mass : ℝ) (volume : ℝ),
    empty_mass = 23 ∧
    mass_with_kerosene = empty_mass + density_kerosene * volume ∧
    mass_with_water = empty_mass + density_water * volume :=
by sorry

end empty_vessel_mass_l961_96123


namespace inequality_system_solution_l961_96147

theorem inequality_system_solution (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1) ∧ (1/2) * x - 1 ≤ 7 - (3/2) * x) ↔ (2 < x ∧ x ≤ 4) :=
by sorry

end inequality_system_solution_l961_96147


namespace circle_equal_circumference_area_diameter_l961_96144

/-- A circle with numerically equal circumference and area has a diameter of 4 -/
theorem circle_equal_circumference_area_diameter (r : ℝ) (h : r > 0) :
  π * (2 * r) = π * r^2 → 2 * r = 4 := by
  sorry

end circle_equal_circumference_area_diameter_l961_96144


namespace lcm_problem_l961_96141

theorem lcm_problem (a b c : ℕ+) (ha : a = 72) (hb : b = 108) (hlcm : Nat.lcm (Nat.lcm a b) c = 37800) : c = 175 := by
  sorry

end lcm_problem_l961_96141


namespace bisected_right_triangle_angles_l961_96142

/-- A right triangle with a bisected right angle -/
structure BisectedRightTriangle where
  /-- The measure of the first acute angle -/
  α : Real
  /-- The measure of the second acute angle -/
  β : Real
  /-- The right angle is 90 degrees -/
  right_angle : α + β = 90
  /-- The angle bisector divides the right angle into two 45-degree angles -/
  bisector_angle : Real
  bisector_property : bisector_angle = 45
  /-- The ratio of angles formed by the angle bisector and the hypotenuse is 7:11 -/
  hypotenuse_angles : Real × Real
  hypotenuse_angles_ratio : hypotenuse_angles.1 / hypotenuse_angles.2 = 7 / 11
  hypotenuse_angles_sum : hypotenuse_angles.1 + hypotenuse_angles.2 = 180 - bisector_angle

/-- The theorem stating the angles of the triangle given the conditions -/
theorem bisected_right_triangle_angles (t : BisectedRightTriangle) : 
  t.α = 65 ∧ t.β = 25 ∧ t.α + t.β = 90 := by
  sorry

end bisected_right_triangle_angles_l961_96142


namespace intersection_of_line_and_curve_l961_96165

/-- Line l is defined by the equation 2x - y - 2 = 0 -/
def line_l (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- Curve C is defined by the equation y² = 2x -/
def curve_C (x y : ℝ) : Prop := y^2 = 2 * x

/-- The intersection points of line l and curve C -/
def intersection_points : Set (ℝ × ℝ) := {(2, 2), (1/2, -1)}

/-- Theorem stating that the intersection points of line l and curve C are (2, 2) and (1/2, -1) -/
theorem intersection_of_line_and_curve :
  ∀ p : ℝ × ℝ, (line_l p.1 p.2 ∧ curve_C p.1 p.2) ↔ p ∈ intersection_points :=
by sorry

end intersection_of_line_and_curve_l961_96165


namespace geometric_sequence_problem_l961_96161

theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (a 3)^2 + 4*(a 3) + 1 = 0 →  -- a_3 is a root of x^2 + 4x + 1 = 0
  (a 15)^2 + 4*(a 15) + 1 = 0 →  -- a_15 is a root of x^2 + 4x + 1 = 0
  a 9 = 1 := by
sorry

end geometric_sequence_problem_l961_96161


namespace pyramid_hemisphere_tangency_l961_96156

theorem pyramid_hemisphere_tangency (h : ℝ) (r : ℝ) (edge_length : ℝ) : 
  h = 8 → r = 3 → 
  (edge_length * edge_length = 2 * ((h * h - r * r) / h * r)^2) →
  edge_length = 24 * Real.sqrt 110 / 55 :=
by sorry

end pyramid_hemisphere_tangency_l961_96156


namespace wall_length_calculation_l961_96154

/-- Calculates the length of a wall given its height, width, brick dimensions, and total number of bricks. -/
theorem wall_length_calculation (wall_height wall_width brick_length brick_width brick_height total_bricks : ℝ) :
  wall_height = 200 ∧ 
  wall_width = 25 ∧ 
  brick_length = 25 ∧ 
  brick_width = 11.25 ∧ 
  brick_height = 6 ∧ 
  total_bricks = 1185.1851851851852 →
  ∃ wall_length : ℝ, 
    wall_length = 400 ∧ 
    total_bricks = (wall_length * wall_height * wall_width) / (brick_length * brick_width * brick_height) :=
by sorry

end wall_length_calculation_l961_96154


namespace average_marks_equals_85_l961_96145

def english_marks : ℕ := 86
def math_marks : ℕ := 89
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 87
def biology_marks : ℕ := 81

def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks_equals_85 : (total_marks : ℚ) / num_subjects = 85 := by
  sorry

end average_marks_equals_85_l961_96145


namespace line_slope_intercept_sum_l961_96120

/-- Given a line with slope -3 passing through the point (2, 4), prove that m + b = 7 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -3 ∧ 4 = m * 2 + b → m + b = 7 := by sorry

end line_slope_intercept_sum_l961_96120


namespace train_length_calculation_l961_96171

/-- Calculates the length of a train given its speed, the length of a bridge it passes, and the time it takes to pass the bridge. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) : 
  train_speed = 50 * (1000 / 3600) → 
  bridge_length = 140 →
  passing_time = 36 →
  (train_speed * passing_time) - bridge_length = 360 := by
  sorry

#check train_length_calculation

end train_length_calculation_l961_96171


namespace linear_function_problem_l961_96160

/-- A linear function satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem linear_function_problem :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) ∧ 
  (∀ x : ℝ, f x = 3 * (f⁻¹ x) + 9) ∧
  f 2 = 5 ∧
  f 3 = 9 →
  f 5 = 9 - 8 * Real.sqrt 3 :=
sorry

end linear_function_problem_l961_96160


namespace second_player_cannot_lose_l961_96192

/-- Represents a player in the game -/
inductive Player : Type
| First : Player
| Second : Player

/-- Represents a move in the game -/
structure Move where
  player : Player
  moveNumber : Nat

/-- Represents the state of the game -/
structure GameState where
  currentMove : Move
  isGameOver : Bool

/-- The game can only end on an even-numbered move -/
axiom game_ends_on_even_move : 
  ∀ (gs : GameState), gs.isGameOver → gs.currentMove.moveNumber % 2 = 0

/-- The first player makes even-numbered moves -/
axiom first_player_even_moves :
  ∀ (m : Move), m.player = Player.First → m.moveNumber % 2 = 0

/-- Theorem: The second player cannot lose -/
theorem second_player_cannot_lose :
  ∀ (gs : GameState), gs.isGameOver → gs.currentMove.player ≠ Player.Second :=
by sorry


end second_player_cannot_lose_l961_96192


namespace quadratic_equation_root_l961_96100

theorem quadratic_equation_root (m : ℝ) : 
  (3 : ℝ) ^ 2 - 3 - m = 0 → m = 6 := by
  sorry

end quadratic_equation_root_l961_96100


namespace savings_account_interest_rate_l961_96173

theorem savings_account_interest_rate (initial_deposit : ℝ) (first_year_balance : ℝ) (total_increase_percentage : ℝ) :
  initial_deposit = 1000 →
  first_year_balance = 1100 →
  total_increase_percentage = 32 →
  let total_amount := initial_deposit * (1 + total_increase_percentage / 100)
  let second_year_increase := total_amount - first_year_balance
  let second_year_increase_percentage := (second_year_increase / first_year_balance) * 100
  second_year_increase_percentage = 20 := by
sorry

end savings_account_interest_rate_l961_96173


namespace train_bridge_crossing_time_l961_96116

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) : 
  train_length = 145 ∧ 
  train_speed_kmh = 45 ∧ 
  bridge_length = 230 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end train_bridge_crossing_time_l961_96116


namespace tan_sum_ratio_l961_96164

theorem tan_sum_ratio : 
  (Real.tan (20 * π / 180) + Real.tan (40 * π / 180) + Real.tan (120 * π / 180)) / 
  (Real.tan (20 * π / 180) * Real.tan (40 * π / 180)) = -Real.sqrt 3 := by
  sorry

end tan_sum_ratio_l961_96164


namespace solution_x_volume_l961_96111

/-- Proves that the volume of solution x is 50 milliliters, given the conditions of the mixing problem. -/
theorem solution_x_volume (x_concentration : Real) (y_concentration : Real) (y_volume : Real) (final_concentration : Real) :
  x_concentration = 0.10 →
  y_concentration = 0.30 →
  y_volume = 150 →
  final_concentration = 0.25 →
  ∃ (x_volume : Real),
    x_volume = 50 ∧
    (x_concentration * x_volume + y_concentration * y_volume) / (x_volume + y_volume) = final_concentration :=
by sorry

end solution_x_volume_l961_96111


namespace min_value_when_a_is_1_range_of_a_for_bounded_f_l961_96175

/-- The function f(x) defined as |2x-a| - |x+3| --/
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| - |x + 3|

/-- Theorem stating the minimum value of f(x) when a = 1 --/
theorem min_value_when_a_is_1 :
  ∃ (m : ℝ), m = -7/2 ∧ ∀ (x : ℝ), f 1 x ≥ m := by sorry

/-- Theorem stating the range of a for which f(x) ≤ 4 when x ∈ [0,3] --/
theorem range_of_a_for_bounded_f :
  ∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 0 3 → f a x ≤ 4) ↔ a ∈ Set.Icc (-4) 7 := by sorry

end min_value_when_a_is_1_range_of_a_for_bounded_f_l961_96175


namespace largest_value_of_P_10_l961_96151

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial := ℝ → ℝ

/-- The largest possible value of P(10) for a quadratic polynomial P satisfying given conditions -/
theorem largest_value_of_P_10 (P : QuadraticPolynomial) 
  (h1 : P 1 = 20)
  (h2 : P (-1) = 22)
  (h3 : P (P 0) = 400) :
  ∃ (max : ℝ), P 10 ≤ max ∧ max = 2486 := by
  sorry

end largest_value_of_P_10_l961_96151


namespace no_real_roots_l961_96108

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 2) + 2 = 0 := by
  sorry

end no_real_roots_l961_96108


namespace gcd_459_357_l961_96163

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l961_96163


namespace radical_conjugate_sum_times_three_l961_96167

theorem radical_conjugate_sum_times_three : 
  let x := 15 - Real.sqrt 500
  let y := 15 + Real.sqrt 500
  3 * (x + y) = 90 := by
sorry

end radical_conjugate_sum_times_three_l961_96167


namespace binomial_minus_five_l961_96196

theorem binomial_minus_five : Nat.choose 10 3 - 5 = 115 := by sorry

end binomial_minus_five_l961_96196


namespace complement_of_A_l961_96195

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {1,2,4,5}

theorem complement_of_A :
  (U \ A) = {3,6,7} := by sorry

end complement_of_A_l961_96195


namespace selection_ways_l961_96183

def boys : ℕ := 5
def girls : ℕ := 3
def total_subjects : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem selection_ways : 
  choose (boys + girls - 2) (total_subjects - 2) * choose (total_subjects - 2) 1 * permute (total_subjects - 2) (total_subjects - 2) = 360 := by
  sorry

end selection_ways_l961_96183


namespace investment_plans_count_l961_96178

/-- The number of candidate cities -/
def num_cities : ℕ := 4

/-- The number of projects to invest -/
def num_projects : ℕ := 3

/-- The maximum number of projects allowed in a single city -/
def max_projects_per_city : ℕ := 2

/-- A function that calculates the number of investment plans -/
def investment_plans : ℕ := sorry

/-- Theorem stating that the number of investment plans is 60 -/
theorem investment_plans_count : investment_plans = 60 := by sorry

end investment_plans_count_l961_96178


namespace solution_set_equiv_l961_96153

theorem solution_set_equiv (x : ℝ) : 
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end solution_set_equiv_l961_96153


namespace arithmetic_calculations_l961_96143

theorem arithmetic_calculations :
  ((-20) + 3 + 5 + (-7) = -19) ∧
  (((-32) / 4) * (1 / 4) = -2) ∧
  ((2 / 7 - 1 / 4) * 28 = 1) ∧
  (-(2^4) * (((-3) * (-(2 + 1 + 1/3)) - (-5))) / ((-2/5)^2) = -1500) :=
by sorry

end arithmetic_calculations_l961_96143


namespace anitas_strawberries_l961_96134

theorem anitas_strawberries (total_cartons : ℕ) (blueberry_cartons : ℕ) (cartons_to_buy : ℕ) 
  (h1 : total_cartons = 26)
  (h2 : blueberry_cartons = 9)
  (h3 : cartons_to_buy = 7) :
  total_cartons - (blueberry_cartons + cartons_to_buy) = 10 := by
  sorry

end anitas_strawberries_l961_96134


namespace gwen_recycling_points_l961_96148

/-- Calculates the points earned by recycling bags of cans. -/
def points_earned (total_bags : ℕ) (unrecycled_bags : ℕ) (points_per_bag : ℕ) : ℕ :=
  (total_bags - unrecycled_bags) * points_per_bag

/-- Proves that Gwen earns 16 points given the problem conditions. -/
theorem gwen_recycling_points :
  points_earned 4 2 8 = 16 := by
  sorry

end gwen_recycling_points_l961_96148


namespace james_truck_mpg_james_truck_mpg_proof_l961_96193

/-- Proves that given the conditions of James's truck driving job, his truck's fuel efficiency is 20 miles per gallon. -/
theorem james_truck_mpg : ℝ → Prop :=
  λ mpg : ℝ =>
    let pay_per_mile : ℝ := 0.5
    let gas_cost_per_gallon : ℝ := 4
    let trip_distance : ℝ := 600
    let profit : ℝ := 180
    let earnings : ℝ := pay_per_mile * trip_distance
    let gas_cost : ℝ := (trip_distance / mpg) * gas_cost_per_gallon
    earnings - gas_cost = profit → mpg = 20

/-- The proof of james_truck_mpg. -/
theorem james_truck_mpg_proof : james_truck_mpg 20 := by
  sorry

end james_truck_mpg_james_truck_mpg_proof_l961_96193


namespace irrational_minus_two_implies_irrational_l961_96127

theorem irrational_minus_two_implies_irrational (a : ℝ) :
  Irrational (a - 2) → Irrational a := by
  sorry

end irrational_minus_two_implies_irrational_l961_96127


namespace elmo_sandwich_jam_cost_l961_96114

/-- The cost of jam used in Elmo's sandwiches --/
theorem elmo_sandwich_jam_cost :
  ∀ (N B J H : ℕ),
    N > 1 →
    B > 0 →
    J > 0 →
    H > 0 →
    N * (3 * B + 6 * J + 2 * H) = 342 →
    N * J * 6 = 270 := by
  sorry

end elmo_sandwich_jam_cost_l961_96114


namespace scooter_initial_value_l961_96185

/-- Proves that the initial value of a scooter is 40000 given the depreciation rate and final value after 3 years -/
theorem scooter_initial_value (depreciation_rate : ℚ) (final_value : ℚ) : 
  depreciation_rate = 3/4 →
  final_value = 16875 →
  (depreciation_rate^3 * 40000 : ℚ) = final_value := by
  sorry

end scooter_initial_value_l961_96185


namespace y_value_l961_96176

theorem y_value (x z y : ℝ) (h1 : x = 2 * z) (h2 : y = 3 * z - 1) (h3 : x = 40) : y = 59 := by
  sorry

end y_value_l961_96176


namespace sum_of_cubes_l961_96149

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : x^3 + y^3 = 640 := by
  sorry

end sum_of_cubes_l961_96149


namespace wages_calculation_l961_96135

def total_budget : ℚ := 3000

def food_fraction : ℚ := 1/3
def supplies_fraction : ℚ := 1/4

def food_expense : ℚ := food_fraction * total_budget
def supplies_expense : ℚ := supplies_fraction * total_budget

def wages_expense : ℚ := total_budget - (food_expense + supplies_expense)

theorem wages_calculation : wages_expense = 1250 := by
  sorry

end wages_calculation_l961_96135


namespace joes_remaining_money_l961_96125

/-- Represents the problem of calculating Joe's remaining money after shopping --/
theorem joes_remaining_money (initial_amount : ℕ) (notebooks : ℕ) (books : ℕ) 
  (notebook_cost : ℕ) (book_cost : ℕ) : 
  initial_amount = 56 →
  notebooks = 7 →
  books = 2 →
  notebook_cost = 4 →
  book_cost = 7 →
  initial_amount - (notebooks * notebook_cost + books * book_cost) = 14 :=
by sorry

end joes_remaining_money_l961_96125


namespace function_expression_l961_96198

theorem function_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x + 1) = x + 1) :
  ∀ x : ℝ, f x = (1/2) * (x + 1) := by
sorry

end function_expression_l961_96198


namespace tan_double_angle_second_quadrant_l961_96194

/-- Given an angle α in the second quadrant with sin(π + α) = -3/5, prove that tan(2α) = -24/7 -/
theorem tan_double_angle_second_quadrant (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin (π + α) = -3/5) : 
  Real.tan (2 * α) = -24/7 := by
  sorry

end tan_double_angle_second_quadrant_l961_96194


namespace complex_equation_solution_l961_96102

theorem complex_equation_solution (a : ℝ) (ha : a ≥ 0) :
  let S := {z : ℂ | z^2 + 2 * Complex.abs z = a}
  S = {z : ℂ | z = -(1 - Real.sqrt (1 + a)) ∨ z = (1 - Real.sqrt (1 + a))} ∪
      (if 0 ≤ a ∧ a ≤ 1 then
        {z : ℂ | z = Complex.I * (1 + Real.sqrt (1 - a)) ∨
                 z = Complex.I * (-(1 + Real.sqrt (1 - a))) ∨
                 z = Complex.I * (1 - Real.sqrt (1 - a)) ∨
                 z = Complex.I * (-(1 - Real.sqrt (1 - a)))}
      else ∅) :=
by sorry

end complex_equation_solution_l961_96102


namespace functional_equation_problem_l961_96122

/-- The functional equation problem -/
theorem functional_equation_problem :
  ∀ (f h : ℝ → ℝ),
  (∀ x y : ℝ, f (x^2 + y * h x) = x * h x + f (x * y)) →
  ((∃ a b : ℝ, (∀ x : ℝ, f x = a) ∧ 
                (∀ x : ℝ, x ≠ 0 → h x = 0) ∧ 
                (h 0 = b)) ∨
   (∃ a : ℝ, (∀ x : ℝ, f x = x + a) ∧ 
             (∀ x : ℝ, h x = x))) :=
sorry

end functional_equation_problem_l961_96122


namespace johns_trip_cost_l961_96112

/-- Calculates the total cost of a car rental trip -/
def total_trip_cost (rental_cost : ℚ) (gas_price : ℚ) (gas_needed : ℚ) (mileage_cost : ℚ) (distance : ℚ) : ℚ :=
  rental_cost + gas_price * gas_needed + mileage_cost * distance

/-- Theorem stating that the total cost of John's trip is $338 -/
theorem johns_trip_cost : 
  total_trip_cost 150 3.5 8 0.5 320 = 338 := by
  sorry

end johns_trip_cost_l961_96112


namespace equation_solution_l961_96137

theorem equation_solution :
  ∃ x : ℝ, x = 4/3 ∧ 
    (3*x^2)/(x-2) - (3*x + 8)/4 + (6 - 9*x)/(x-2) + 2 = 0 :=
by
  sorry

end equation_solution_l961_96137


namespace smallest_number_with_conditions_l961_96146

theorem smallest_number_with_conditions : 
  ∃ (n : ℕ), n = 2102 ∧ 
  (11 ∣ n) ∧ 
  (∀ i : ℕ, 3 ≤ i → i ≤ 7 → n % i = 2) ∧
  (∀ m : ℕ, m < n → ¬((11 ∣ m) ∧ (∀ i : ℕ, 3 ≤ i → i ≤ 7 → m % i = 2))) :=
by sorry

end smallest_number_with_conditions_l961_96146


namespace total_tosses_equals_sum_of_heads_and_tails_l961_96133

/-- Represents the number of times Head came up in the coin tosses -/
def head_count : ℕ := 9

/-- Represents the number of times Tail came up in the coin tosses -/
def tail_count : ℕ := 5

/-- Theorem stating that the total number of coin tosses is the sum of head_count and tail_count -/
theorem total_tosses_equals_sum_of_heads_and_tails :
  head_count + tail_count = 14 := by sorry

end total_tosses_equals_sum_of_heads_and_tails_l961_96133


namespace part_to_whole_ratio_l961_96158

theorem part_to_whole_ratio (N P : ℚ) 
  (h1 : (1/4) * P = 17)
  (h2 : (2/5) * N = P)
  (h3 : (40/100) * N = 204) :
  P / N = 2 / 5 := by
  sorry

end part_to_whole_ratio_l961_96158


namespace set_operations_proof_l961_96104

def U := Set ℝ

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3 ∨ 4 < x ∧ x < 6}

def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem set_operations_proof :
  (Set.compl B = {x : ℝ | x < 2 ∨ x ≥ 5}) ∧
  (A ∩ (Set.compl B) = {x : ℝ | (1 ≤ x ∧ x < 2) ∨ (5 ≤ x ∧ x < 6)}) := by
  sorry

end set_operations_proof_l961_96104


namespace cindy_calculation_l961_96191

theorem cindy_calculation (x : ℝ) : 
  (x - 12) / 4 = 28 → (x - 5) / 8 = 14.875 := by sorry

end cindy_calculation_l961_96191


namespace division_reciprocal_equivalence_l961_96169

theorem division_reciprocal_equivalence (x : ℝ) (hx : x ≠ 0) :
  1 / x = 1 * (1 / x) :=
by sorry

end division_reciprocal_equivalence_l961_96169


namespace train_length_l961_96126

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 4 → ∃ length : ℝ, abs (length - 66.68) < 0.01 :=
by
  sorry

end train_length_l961_96126


namespace apples_remaining_l961_96136

/-- Calculates the number of apples remaining on a tree after three days of picking -/
theorem apples_remaining (total : ℕ) (day1_fraction : ℚ) (day2_multiplier : ℕ) (day3_addition : ℕ) : 
  total = 200 →
  day1_fraction = 1 / 5 →
  day2_multiplier = 2 →
  day3_addition = 20 →
  total - (total * day1_fraction).floor - day2_multiplier * (total * day1_fraction).floor - ((total * day1_fraction).floor + day3_addition) = 20 := by
sorry

end apples_remaining_l961_96136


namespace mandy_sister_age_difference_l961_96199

/-- Represents the ages and relationships in Mandy's family -/
structure Family where
  mandy_age : ℕ
  brother_age : ℕ
  sister_age : ℕ
  brother_age_relation : brother_age = 4 * mandy_age
  sister_age_relation : sister_age = brother_age - 5

/-- Calculates the age difference between Mandy and her sister -/
def age_difference (f : Family) : ℕ :=
  f.sister_age - f.mandy_age

/-- Theorem stating the age difference between Mandy and her sister -/
theorem mandy_sister_age_difference (f : Family) (h : f.mandy_age = 3) :
  age_difference f = 4 := by
  sorry

#check mandy_sister_age_difference

end mandy_sister_age_difference_l961_96199


namespace equation_holds_when_b_plus_c_is_ten_l961_96138

theorem equation_holds_when_b_plus_c_is_ten (a b c : ℕ) : 
  a > 0 → a < 10 → b > 0 → b < 10 → c > 0 → c < 10 → b + c = 10 → 
  (10 * b + a) * (10 * c + a) = 100 * b * c + 100 * a + a^2 := by
sorry

end equation_holds_when_b_plus_c_is_ten_l961_96138


namespace zero_point_implies_m_range_l961_96152

theorem zero_point_implies_m_range (m : ℝ) :
  (∃ x₀ ∈ Set.Icc (-2 : ℝ) 0, 3 * m * x₀ - 4 = 0) →
  m ∈ Set.Iic (-2/3 : ℝ) :=
by
  sorry

end zero_point_implies_m_range_l961_96152


namespace area_closed_region_l961_96184

/-- The area of the closed region formed by f(x) and g(x) over one period -/
theorem area_closed_region (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (a * x) + Real.cos (a * x)
  let g : ℝ → ℝ := λ x ↦ Real.sqrt (a^2 + 1)
  let period : ℝ := 2 * Real.pi / a
  ∃ (area : ℝ), area = period * Real.sqrt (a^2 + 1) :=
by sorry

end area_closed_region_l961_96184


namespace value_of_a_satisfying_equation_l961_96168

-- Define the function F
def F (a b c : ℝ) : ℝ := a * b^3 + c

-- State the theorem
theorem value_of_a_satisfying_equation :
  ∃ a : ℝ, F a 3 8 = F a 5 12 ∧ a = -2/49 := by sorry

end value_of_a_satisfying_equation_l961_96168


namespace max_min_values_of_f_on_interval_l961_96124

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x + 5

-- Define the interval
def interval : Set ℝ := {x | -5/2 ≤ x ∧ x ≤ 3/2}

-- Theorem statement
theorem max_min_values_of_f_on_interval :
  (∃ x ∈ interval, f x = 9 ∧ ∀ y ∈ interval, f y ≤ 9) ∧
  (∃ x ∈ interval, f x = -11.25 ∧ ∀ y ∈ interval, f y ≥ -11.25) :=
sorry

end max_min_values_of_f_on_interval_l961_96124


namespace infinite_perfect_squares_in_sequence_l961_96129

theorem infinite_perfect_squares_in_sequence : 
  ∀ k : ℕ, ∃ n : ℕ, ∃ m : ℕ, 2^n + 4^k = m^2 := by
  sorry

end infinite_perfect_squares_in_sequence_l961_96129


namespace hayden_earnings_l961_96131

/-- Calculates the total earnings for a limo driver based on given parameters. -/
def limo_driver_earnings (hourly_wage : ℕ) (hours_worked : ℕ) (ride_bonus : ℕ) (rides_given : ℕ) 
  (review_bonus : ℕ) (positive_reviews : ℕ) (gas_price : ℕ) (gas_used : ℕ) : ℕ :=
  hourly_wage * hours_worked + ride_bonus * rides_given + review_bonus * positive_reviews + gas_price * gas_used

/-- Proves that Hayden's earnings for the day equal $226 given the specified conditions. -/
theorem hayden_earnings : 
  limo_driver_earnings 15 8 5 3 20 2 3 17 = 226 := by
  sorry

end hayden_earnings_l961_96131


namespace number_exchange_ratio_l961_96186

theorem number_exchange_ratio (a b p q : ℝ) (h : p * q ≠ 1) :
  ∃ z : ℝ, (z + a - a) + ((p * z - a) + a) = q * ((z + a + b) - ((p * z - a) - b)) →
  z = (a + b) * (q + 1) / (p * q - 1) :=
by sorry

end number_exchange_ratio_l961_96186


namespace smallest_k_for_power_inequality_l961_96188

theorem smallest_k_for_power_inequality : ∃ k : ℕ, k = 14 ∧ 
  (∀ n : ℕ, n < k → (7 : ℝ)^n ≤ 4^19) ∧ (7 : ℝ)^k > 4^19 := by
  sorry

end smallest_k_for_power_inequality_l961_96188


namespace last_two_digits_product_l961_96166

def last_two_digits (n : ℕ) : ℕ × ℕ :=
  ((n / 10) % 10, n % 10)

theorem last_two_digits_product (n : ℕ) 
  (h1 : n % 4 = 0) 
  (h2 : (last_two_digits n).1 + (last_two_digits n).2 = 14) : 
  (last_two_digits n).1 * (last_two_digits n).2 = 48 := by
sorry

end last_two_digits_product_l961_96166


namespace f_monotonicity_and_max_k_l961_96132

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 2

theorem f_monotonicity_and_max_k :
  (∀ a ≤ 0, ∀ x y, x < y → f a x < f a y) ∧
  (∀ a > 0, ∀ x y, x < y → 
    ((x < Real.log a ∧ y < Real.log a → f a x > f a y) ∧
     (x > Real.log a ∧ y > Real.log a → f a x < f a y))) ∧
  (∀ k : ℤ, (∀ x > 0, (x - ↑k) * (Real.exp x - 1) + x + 1 > 0) → k ≤ 2) ∧
  (∀ x > 0, (x - 2) * (Real.exp x - 1) + x + 1 > 0) :=
by sorry

end f_monotonicity_and_max_k_l961_96132


namespace sequence_sum_l961_96113

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 1  -- First term is 1
  | n + 1 => 
    let k := (n + 1).sqrt  -- k-th group
    if (k * k ≤ n + 1) ∧ (n + 1 < (k + 1) * (k + 1)) then
      if n + 1 = k * k then 1 else 2
    else a n  -- This case should never happen, but Lean needs it for totality

-- Define the sum S_n
def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

-- Theorem statement
theorem sequence_sum :
  S 20 = 36 ∧ S 2017 = 3989 := by sorry

end sequence_sum_l961_96113


namespace smallest_x_for_inequality_l961_96118

theorem smallest_x_for_inequality : ∃ x : ℕ, (∀ y : ℕ, 27^y > 3^24 → x ≤ y) ∧ 27^x > 3^24 :=
by
  -- The proof goes here
  sorry

end smallest_x_for_inequality_l961_96118


namespace eight_elevenths_rounded_l961_96172

/-- Rounds a rational number to the specified number of decimal places -/
def round_to_decimal_places (q : ℚ) (places : ℕ) : ℚ :=
  (⌊q * 10^places + 1/2⌋ : ℚ) / 10^places

/-- Proves that 8/11 rounded to 3 decimal places is equal to 0.727 -/
theorem eight_elevenths_rounded : round_to_decimal_places (8/11) 3 = 727/1000 := by
  sorry

end eight_elevenths_rounded_l961_96172


namespace arithmetic_simplification_l961_96187

theorem arithmetic_simplification : (-18) + (-12) - (-33) + 17 = 20 := by
  sorry

end arithmetic_simplification_l961_96187
