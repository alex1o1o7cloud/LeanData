import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_for_sock_problem_l30_3044

theorem no_solution_for_sock_problem : ¬∃ (n m : ℕ), 
  n + m = 2009 ∧ (n^2 - 2*n*m + m^2) = (n + m) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_sock_problem_l30_3044


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l30_3029

/-- The minimum distance between two points on different curves with the same y-coordinate --/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  min_dist = 3/2 ∧
  ∀ (a : ℝ) (x₁ x₂ : ℝ),
    a = 2 * (x₁ + 1) →
    a = x₂ + Real.log x₂ →
    |x₂ - x₁| ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l30_3029


namespace NUMINAMATH_CALUDE_text_pages_count_l30_3075

theorem text_pages_count (total_pages : ℕ) (image_pages : ℕ) (intro_pages : ℕ) : 
  total_pages = 98 →
  image_pages = total_pages / 2 →
  intro_pages = 11 →
  (total_pages - image_pages - intro_pages) % 2 = 0 →
  (total_pages - image_pages - intro_pages) / 2 = 19 :=
by sorry

end NUMINAMATH_CALUDE_text_pages_count_l30_3075


namespace NUMINAMATH_CALUDE_min_value_two_over_x_plus_x_over_two_min_value_achievable_l30_3035

theorem min_value_two_over_x_plus_x_over_two (x : ℝ) (hx : x > 0) :
  2/x + x/2 ≥ 2 :=
sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > 0 ∧ 2/x + x/2 = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_two_over_x_plus_x_over_two_min_value_achievable_l30_3035


namespace NUMINAMATH_CALUDE_number_equality_l30_3028

theorem number_equality (x : ℝ) : (2 * x + 20 = 8 * x - 4) ↔ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l30_3028


namespace NUMINAMATH_CALUDE_survey_respondents_l30_3007

/-- Represents the number of people preferring each brand in a survey. -/
structure BrandPreference where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of respondents in a survey. -/
def totalRespondents (bp : BrandPreference) : ℕ :=
  bp.x + bp.y + bp.z

/-- Theorem stating that given the conditions of the survey, 
    the total number of respondents is 350. -/
theorem survey_respondents : 
  ∀ (bp : BrandPreference), 
    bp.x = 200 → 
    4 * bp.z = bp.x → 
    2 * bp.z = bp.y → 
    totalRespondents bp = 350 := by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l30_3007


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l30_3005

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem twentieth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 5) :
  arithmeticSequence a₁ (a₂ - a₁) 20 = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l30_3005


namespace NUMINAMATH_CALUDE_hyperbola_range_l30_3025

-- Define the equation
def equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + (2 - m) * y^2 = 1

-- Define what it means for the equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  m ≠ 0 ∧ m ≠ 2 ∧ m * (2 - m) < 0

-- Theorem statement
theorem hyperbola_range (m : ℝ) :
  is_hyperbola m ↔ m < 0 ∨ m > 2 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_range_l30_3025


namespace NUMINAMATH_CALUDE_count_valid_domains_l30_3037

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the set of possible domain elements
def domain_elements : Set ℝ := {-Real.sqrt 2, -1, 1, Real.sqrt 2}

-- Define the range
def target_range : Set ℝ := {1, 2}

-- Define a valid domain
def is_valid_domain (S : Set ℝ) : Prop :=
  S ⊆ domain_elements ∧ f '' S = target_range

-- Theorem statement
theorem count_valid_domains :
  ∃ (valid_domains : Finset (Set ℝ)),
    (∀ S ∈ valid_domains, is_valid_domain S) ∧
    (∀ S, is_valid_domain S → S ∈ valid_domains) ∧
    valid_domains.card = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_domains_l30_3037


namespace NUMINAMATH_CALUDE_square_sum_equals_90_l30_3065

theorem square_sum_equals_90 (x y : ℝ) 
  (h1 : x * (2 * x + y) = 18) 
  (h2 : y * (2 * x + y) = 72) : 
  (2 * x + y)^2 = 90 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_90_l30_3065


namespace NUMINAMATH_CALUDE_plane_sphere_intersection_ratio_sum_l30_3098

/-- The theorem states that for a plane intersecting the coordinate axes and a sphere passing through these intersection points and the origin, the sum of the ratios of a point on the plane to the sphere's center coordinates is 2. -/
theorem plane_sphere_intersection_ratio_sum (k : ℝ) (a b c p q r : ℝ) : 
  k ≠ 0 → -- k is a non-zero constant
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 → -- p, q, r are non-zero (as they are denominators)
  (∃ (α β γ : ℝ), α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0 ∧ -- A, B, C exist and are distinct from O
    (k*a/α + k*b/β + k*c/γ = 1) ∧ -- plane equation
    (p^2 + q^2 + r^2 = (p - α)^2 + q^2 + r^2) ∧ -- sphere equation for A
    (p^2 + q^2 + r^2 = p^2 + (q - β)^2 + r^2) ∧ -- sphere equation for B
    (p^2 + q^2 + r^2 = p^2 + q^2 + (r - γ)^2)) → -- sphere equation for C
  k*a/p + k*b/q + k*c/r = 2 := by
sorry

end NUMINAMATH_CALUDE_plane_sphere_intersection_ratio_sum_l30_3098


namespace NUMINAMATH_CALUDE_new_person_weight_is_75_l30_3082

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (initial_count : ℕ) (average_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the new person is 75 kg -/
theorem new_person_weight_is_75 :
  new_person_weight 8 (5/2) 55 = 75 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_75_l30_3082


namespace NUMINAMATH_CALUDE_conditional_probability_l30_3059

theorem conditional_probability (P_AB P_A : ℝ) (h1 : P_AB = 3/10) (h2 : P_A = 3/5) :
  P_AB / P_A = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_l30_3059


namespace NUMINAMATH_CALUDE_work_completion_time_l30_3023

/-- The number of days y needs to finish the work alone -/
def y_days : ℕ := 24

/-- The number of days y worked before leaving -/
def y_worked : ℕ := 12

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℕ := 18

/-- The number of days x needs to finish the work alone -/
def x_days : ℕ := 36

theorem work_completion_time :
  x_days = 36 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l30_3023


namespace NUMINAMATH_CALUDE_triangle_circles_tangency_l30_3026

theorem triangle_circles_tangency (DE DF EF : ℝ) (R S : ℝ) :
  DE = 120 →
  DF = 120 →
  EF = 70 →
  R = 20 →
  S > 0 →
  S + R > EF / 2 →
  S < DE - R →
  (S + R)^2 + (S - R)^2 = ((130 - 4*S) / 3)^2 →
  S = 55 - 5 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circles_tangency_l30_3026


namespace NUMINAMATH_CALUDE_volume_ratio_of_cubes_l30_3061

/-- The perimeter of a square face of a cube -/
def face_perimeter (s : ℝ) : ℝ := 4 * s

/-- The volume of a cube -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- Theorem: Given two cubes A and B with face perimeters 40 cm and 64 cm respectively, 
    the ratio of their volumes is 125:512 -/
theorem volume_ratio_of_cubes (s_A s_B : ℝ) 
  (h_A : face_perimeter s_A = 40)
  (h_B : face_perimeter s_B = 64) : 
  (cube_volume s_A) / (cube_volume s_B) = 125 / 512 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_of_cubes_l30_3061


namespace NUMINAMATH_CALUDE_complex_equation_sum_l30_3097

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (a + i) * i = b + i) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l30_3097


namespace NUMINAMATH_CALUDE_smallest_value_l30_3006

theorem smallest_value (A B C D : ℝ) : 
  A = Real.sin (50 * π / 180) * Real.cos (39 * π / 180) - Real.sin (40 * π / 180) * Real.cos (51 * π / 180) →
  B = -2 * Real.sin (40 * π / 180)^2 + 1 →
  C = 2 * Real.sin (6 * π / 180) * Real.cos (6 * π / 180) →
  D = Real.sqrt 3 / 2 * Real.sin (43 * π / 180) - 1 / 2 * Real.cos (43 * π / 180) →
  B < A ∧ B < C ∧ B < D :=
by sorry


end NUMINAMATH_CALUDE_smallest_value_l30_3006


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l30_3009

theorem consecutive_odd_numbers (x : ℤ) : 
  (∃ y z : ℤ, y = x + 2 ∧ z = x + 4 ∧ 
   Odd x ∧ Odd y ∧ Odd z ∧
   11 * x = 3 * z + 4 * y + 16) → 
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_l30_3009


namespace NUMINAMATH_CALUDE_subsidy_scheme_2_maximizes_profit_l30_3030

-- Define the daily processing capacity
def x : ℝ := 100

-- Define the constraints on x
axiom x_lower_bound : 70 ≤ x
axiom x_upper_bound : x ≤ 100

-- Define the total daily processing cost function
def total_cost (x : ℝ) : ℝ := 0.5 * x^2 + 40 * x + 3200

-- Define the selling price per ton
def selling_price : ℝ := 110

-- Define the two subsidy schemes
def subsidy_scheme_1 : ℝ := 2300
def subsidy_scheme_2 (x : ℝ) : ℝ := 30 * x

-- Define the profit functions for each subsidy scheme
def profit_scheme_1 (x : ℝ) : ℝ := selling_price * x - total_cost x + subsidy_scheme_1
def profit_scheme_2 (x : ℝ) : ℝ := selling_price * x - total_cost x + subsidy_scheme_2 x

-- Theorem: Subsidy scheme 2 maximizes profit
theorem subsidy_scheme_2_maximizes_profit :
  profit_scheme_2 x > profit_scheme_1 x :=
sorry

end NUMINAMATH_CALUDE_subsidy_scheme_2_maximizes_profit_l30_3030


namespace NUMINAMATH_CALUDE_stating_calculate_downstream_speed_l30_3085

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- 
Theorem stating that given a man's upstream and still water speeds, 
his downstream speed can be calculated.
-/
theorem calculate_downstream_speed (speed : RowingSpeed) 
  (h1 : speed.upstream = 15)
  (h2 : speed.stillWater = 20) :
  speed.downstream = 25 := by
  sorry

#check calculate_downstream_speed

end NUMINAMATH_CALUDE_stating_calculate_downstream_speed_l30_3085


namespace NUMINAMATH_CALUDE_initial_ratio_men_to_women_l30_3000

/-- Proves that the initial ratio of men to women in a room was 4:5 --/
theorem initial_ratio_men_to_women :
  ∀ (initial_men initial_women : ℕ),
  (initial_women - 3) * 2 = 24 →
  initial_men + 2 = 14 →
  (initial_men : ℚ) / initial_women = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_initial_ratio_men_to_women_l30_3000


namespace NUMINAMATH_CALUDE_purple_socks_probability_l30_3032

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer where
  green : ℕ
  purple : ℕ
  orange : ℕ

/-- Calculates the total number of socks in the drawer -/
def SockDrawer.total (d : SockDrawer) : ℕ :=
  d.green + d.purple + d.orange

/-- Calculates the probability of selecting a purple sock -/
def purpleProbability (d : SockDrawer) : ℚ :=
  d.purple / d.total

/-- The initial state of the sock drawer -/
def initialDrawer : SockDrawer :=
  { green := 6, purple := 18, orange := 12 }

/-- The number of purple socks added -/
def addedPurpleSocks : ℕ := 9

/-- The final state of the sock drawer after adding purple socks -/
def finalDrawer : SockDrawer :=
  { green := initialDrawer.green,
    purple := initialDrawer.purple + addedPurpleSocks,
    orange := initialDrawer.orange }

theorem purple_socks_probability :
  purpleProbability finalDrawer = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_purple_socks_probability_l30_3032


namespace NUMINAMATH_CALUDE_discount_equivalence_l30_3084

/-- Proves that a 30% discount followed by a 15% discount is equivalent to a 40.5% single discount -/
theorem discount_equivalence (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.30
  let second_discount := 0.15
  let discounted_price := original_price * (1 - first_discount)
  let final_price := discounted_price * (1 - second_discount)
  let equivalent_discount := 1 - (final_price / original_price)
  equivalent_discount = 0.405 := by
sorry

end NUMINAMATH_CALUDE_discount_equivalence_l30_3084


namespace NUMINAMATH_CALUDE_last_digit_89_base_5_l30_3083

theorem last_digit_89_base_5 : 89 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_89_base_5_l30_3083


namespace NUMINAMATH_CALUDE_ice_cream_ratio_is_two_to_one_l30_3003

/-- The ratio of Victoria's ice cream scoops to Oli's ice cream scoops -/
def ice_cream_ratio : ℚ := by
  -- Define Oli's number of scoops
  let oli_scoops : ℕ := 4
  -- Define Victoria's number of scoops
  let victoria_scoops : ℕ := oli_scoops + 4
  -- Calculate the ratio
  exact (victoria_scoops : ℚ) / oli_scoops

/-- Theorem stating that the ice cream ratio is 2:1 -/
theorem ice_cream_ratio_is_two_to_one : ice_cream_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_ratio_is_two_to_one_l30_3003


namespace NUMINAMATH_CALUDE_product_remainder_mod_17_l30_3039

theorem product_remainder_mod_17 : (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_17_l30_3039


namespace NUMINAMATH_CALUDE_ralph_peanuts_l30_3036

/-- Represents the number of peanuts Ralph starts with -/
def initial_peanuts : ℕ := sorry

/-- Represents the number of peanuts Ralph loses -/
def lost_peanuts : ℕ := 59

/-- Represents the number of peanuts Ralph ends up with -/
def final_peanuts : ℕ := 15

/-- Theorem stating that Ralph started with 74 peanuts -/
theorem ralph_peanuts : initial_peanuts = 74 :=
by
  sorry

end NUMINAMATH_CALUDE_ralph_peanuts_l30_3036


namespace NUMINAMATH_CALUDE_arithmetic_progression_equality_l30_3091

theorem arithmetic_progression_equality (n : ℕ) 
  (hn : n ≥ 2018) 
  (a b : Fin n → ℕ) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → (a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j))
  (h_bound : ∀ i : Fin n, a i ≤ 5*n ∧ b i ≤ 5*n)
  (h_positive : ∀ i : Fin n, a i > 0 ∧ b i > 0)
  (h_arithmetic : ∃ d : ℚ, ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) - (a j : ℚ) / (b j : ℚ) = (i.val - j.val : ℚ) * d) :
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equality_l30_3091


namespace NUMINAMATH_CALUDE_min_games_to_satisfy_condition_l30_3057

/-- The number of teams in the tournament -/
def num_teams : ℕ := 20

/-- The total number of possible games between all teams -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The maximum number of unplayed games while satisfying the condition -/
def max_unplayed_games : ℕ := (num_teams / 2) ^ 2

/-- A function that checks if the number of played games satisfies the condition -/
def satisfies_condition (played_games : ℕ) : Prop :=
  played_games ≥ total_games - max_unplayed_games

/-- The theorem stating the minimum number of games that must be played -/
theorem min_games_to_satisfy_condition :
  ∃ (min_games : ℕ), satisfies_condition min_games ∧
  ∀ (n : ℕ), n < min_games → ¬satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_min_games_to_satisfy_condition_l30_3057


namespace NUMINAMATH_CALUDE_problem_statement_l30_3090

theorem problem_statement (x y z a : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : x^2 - 1/y = y^2 - 1/z ∧ y^2 - 1/z = z^2 - 1/x ∧ z^2 - 1/x = a) :
  (x + y + z) * x * y * z = -a^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l30_3090


namespace NUMINAMATH_CALUDE_sleeping_passenger_journey_l30_3041

theorem sleeping_passenger_journey (total_journey : ℝ) (sleeping_distance : ℝ) :
  (sleeping_distance = total_journey / 3) ∧
  (total_journey / 2 = sleeping_distance + sleeping_distance / 2) →
  sleeping_distance / total_journey = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sleeping_passenger_journey_l30_3041


namespace NUMINAMATH_CALUDE_sum_of_six_consecutive_odd_integers_l30_3089

theorem sum_of_six_consecutive_odd_integers (S : ℤ) :
  (∃ n : ℤ, S = 6*n + 30 ∧ Odd n) ↔ (∃ k : ℤ, S - 30 = 6*k ∧ Odd k) :=
sorry

end NUMINAMATH_CALUDE_sum_of_six_consecutive_odd_integers_l30_3089


namespace NUMINAMATH_CALUDE_systematic_sampling_first_number_l30_3070

/-- Systematic sampling function -/
def systematicSample (firstSelected : ℕ) (groupSize : ℕ) (groupNumber : ℕ) : ℕ :=
  firstSelected + groupSize * (groupNumber - 1)

theorem systematic_sampling_first_number 
  (totalStudents : ℕ) 
  (sampleSize : ℕ) 
  (selectedNumber : ℕ) 
  (selectedGroup : ℕ) 
  (h1 : totalStudents = 800) 
  (h2 : sampleSize = 50) 
  (h3 : selectedNumber = 503) 
  (h4 : selectedGroup = 32) :
  ∃ (firstSelected : ℕ), 
    firstSelected = 7 ∧ 
    systematicSample firstSelected (totalStudents / sampleSize) selectedGroup = selectedNumber :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_number_l30_3070


namespace NUMINAMATH_CALUDE_negative_one_times_negative_three_equals_three_l30_3022

theorem negative_one_times_negative_three_equals_three :
  (-1 : ℤ) * (-3 : ℤ) = (3 : ℤ) := by sorry

end NUMINAMATH_CALUDE_negative_one_times_negative_three_equals_three_l30_3022


namespace NUMINAMATH_CALUDE_comparison_of_powers_and_log_l30_3004

theorem comparison_of_powers_and_log : 7^(3/10) > (3/10)^7 ∧ (3/10)^7 > Real.log (3/10) := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_powers_and_log_l30_3004


namespace NUMINAMATH_CALUDE_grooming_time_calculation_l30_3077

/-- Proves that the time to clip each claw is 10 seconds given the grooming conditions --/
theorem grooming_time_calculation (total_time : ℕ) (num_claws : ℕ) (ear_cleaning_time : ℕ) (shampoo_time_minutes : ℕ) :
  total_time = 640 →
  num_claws = 16 →
  ear_cleaning_time = 90 →
  shampoo_time_minutes = 5 →
  ∃ (claw_clip_time : ℕ),
    claw_clip_time = 10 ∧
    total_time = num_claws * claw_clip_time + 2 * ear_cleaning_time + shampoo_time_minutes * 60 :=
by sorry

end NUMINAMATH_CALUDE_grooming_time_calculation_l30_3077


namespace NUMINAMATH_CALUDE_cube_intersection_length_l30_3063

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with edge length a -/
structure Cube (a : ℝ) where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D
  F : Point3D
  G : Point3D
  H : Point3D

/-- The theorem to be proved -/
theorem cube_intersection_length (a : ℝ) (cube : Cube a) 
  (M : Point3D) (N : Point3D) (P : Point3D) (T : Point3D)
  (h_a : a > 0)
  (h_M : M.x = a ∧ M.y = a ∧ M.z = a/2)
  (h_N : N.x = a ∧ N.y = a/3 ∧ N.z = a)
  (h_P : P.x = 0 ∧ P.y = 0 ∧ P.z = 3*a/4)
  (h_T : T.x = 0 ∧ T.y = a ∧ 0 ≤ T.z ∧ T.z ≤ a)
  (h_plane : ∃ (k : ℝ), k * (M.x - P.x) * (N.y - P.y) * (T.z - P.z) = 
                         k * (N.x - P.x) * (M.y - P.y) * (T.z - P.z) + 
                         k * (T.x - P.x) * (M.y - P.y) * (N.z - P.z)) :
  ∃ (DT : ℝ), DT = 5*a/6 ∧ DT = Real.sqrt ((T.x - cube.D.x)^2 + (T.y - cube.D.y)^2 + (T.z - cube.D.z)^2) :=
sorry

end NUMINAMATH_CALUDE_cube_intersection_length_l30_3063


namespace NUMINAMATH_CALUDE_circle_rectangle_area_difference_l30_3064

/-- Given a rectangle with diagonal 10 and length-to-width ratio 2:1, and a circle with radius 5,
    prove that the difference between the circle's area and the rectangle's area is 25π - 40. -/
theorem circle_rectangle_area_difference :
  let rectangle_diagonal : ℝ := 10
  let length_width_ratio : ℚ := 2 / 1
  let circle_radius : ℝ := 5
  let rectangle_width : ℝ := (rectangle_diagonal ^ 2 / (1 + length_width_ratio ^ 2)) ^ (1 / 2 : ℝ)
  let rectangle_length : ℝ := length_width_ratio * rectangle_width
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let circle_area : ℝ := π * circle_radius ^ 2
  circle_area - rectangle_area = 25 * π - 40 := by
  sorry

end NUMINAMATH_CALUDE_circle_rectangle_area_difference_l30_3064


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l30_3014

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

-- State the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A a) ∩ (B a) = {-3} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l30_3014


namespace NUMINAMATH_CALUDE_freds_remaining_cards_l30_3002

/-- Calculates the number of baseball cards Fred has after Melanie's purchase. -/
def remaining_cards (initial : ℕ) (bought : ℕ) : ℕ :=
  initial - bought

/-- Theorem stating that Fred's remaining cards is the difference between his initial cards and those bought by Melanie. -/
theorem freds_remaining_cards :
  remaining_cards 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_freds_remaining_cards_l30_3002


namespace NUMINAMATH_CALUDE_six_digit_square_reverse_square_exists_l30_3066

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem six_digit_square_reverse_square_exists : ∃ n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧
  is_perfect_square n ∧
  is_perfect_square (reverse_digits n) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_square_reverse_square_exists_l30_3066


namespace NUMINAMATH_CALUDE_matthews_water_glass_size_l30_3008

/-- Given Matthew's water drinking habits, prove the number of ounces in each glass. -/
theorem matthews_water_glass_size 
  (glasses_per_day : ℕ) 
  (bottle_size : ℕ) 
  (fills_per_week : ℕ) 
  (h1 : glasses_per_day = 4)
  (h2 : bottle_size = 35)
  (h3 : fills_per_week = 4) :
  (bottle_size * fills_per_week) / (glasses_per_day * 7) = 5 := by
  sorry

#check matthews_water_glass_size

end NUMINAMATH_CALUDE_matthews_water_glass_size_l30_3008


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l30_3069

/-- Represents a rectangular grid --/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a rectangular shaded region within the grid --/
structure ShadedRegion :=
  (start_x : ℕ)
  (start_y : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Calculates the area of a shaded region --/
def area_of_region (region : ShadedRegion) : ℕ :=
  region.width * region.height

/-- Calculates the total area of multiple shaded regions --/
def total_shaded_area (regions : List ShadedRegion) : ℕ :=
  regions.map area_of_region |>.sum

theorem shaded_area_theorem (grid : Grid) (regions : List ShadedRegion) : 
  grid.width = 15 → 
  grid.height = 5 → 
  regions = [
    { start_x := 0, start_y := 0, width := 6, height := 3 },
    { start_x := 6, start_y := 3, width := 9, height := 2 }
  ] → 
  total_shaded_area regions = 36 := by
  sorry

#check shaded_area_theorem

end NUMINAMATH_CALUDE_shaded_area_theorem_l30_3069


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l30_3081

theorem quadratic_inequality_solution_set : 
  {x : ℝ | 3 * x^2 - 5 * x - 2 < 0} = {x : ℝ | -1/3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l30_3081


namespace NUMINAMATH_CALUDE_pats_calculation_l30_3072

theorem pats_calculation (x : ℝ) : 
  (x / 8) - 20 = 12 → 
  1800 < (x * 8) + 20 ∧ (x * 8) + 20 < 2200 :=
by
  sorry

end NUMINAMATH_CALUDE_pats_calculation_l30_3072


namespace NUMINAMATH_CALUDE_cyclic_inequality_with_powers_l30_3074

theorem cyclic_inequality_with_powers (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) (h₆ : x₆ > 0) :
  (x₂/x₁)^5 + (x₄/x₂)^5 + (x₆/x₃)^5 + (x₁/x₄)^5 + (x₃/x₅)^5 + (x₅/x₆)^5 ≥ 
  x₁/x₂ + x₂/x₄ + x₃/x₆ + x₄/x₁ + x₅/x₃ + x₆/x₅ := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_with_powers_l30_3074


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l30_3045

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l30_3045


namespace NUMINAMATH_CALUDE_plumber_toilet_charge_l30_3024

def sink_charge : ℕ := 30
def shower_charge : ℕ := 40

def job1_earnings (toilet_charge : ℕ) : ℕ := 3 * toilet_charge + 3 * sink_charge
def job2_earnings (toilet_charge : ℕ) : ℕ := 2 * toilet_charge + 5 * sink_charge
def job3_earnings (toilet_charge : ℕ) : ℕ := toilet_charge + 2 * shower_charge + 3 * sink_charge

def max_earnings : ℕ := 250

theorem plumber_toilet_charge :
  ∃ (toilet_charge : ℕ),
    (job1_earnings toilet_charge ≤ max_earnings) ∧
    (job2_earnings toilet_charge ≤ max_earnings) ∧
    (job3_earnings toilet_charge ≤ max_earnings) ∧
    ((job1_earnings toilet_charge = max_earnings) ∨
     (job2_earnings toilet_charge = max_earnings) ∨
     (job3_earnings toilet_charge = max_earnings)) ∧
    toilet_charge = 50 :=
by sorry

end NUMINAMATH_CALUDE_plumber_toilet_charge_l30_3024


namespace NUMINAMATH_CALUDE_units_digit_of_57_to_57_l30_3011

theorem units_digit_of_57_to_57 : (57^57) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_57_to_57_l30_3011


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l30_3099

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2_a5 : a 2 + a 5 = 12)
  (h_an : ∃ n : ℕ, a n = 25) :
  ∃ n : ℕ, n = 13 ∧ a n = 25 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l30_3099


namespace NUMINAMATH_CALUDE_rook_placements_count_l30_3017

/-- Represents a special chessboard with a long horizontal row at the bottom -/
structure SpecialChessboard where
  rows : Nat
  columns : Nat
  longRowLength : Nat

/-- Represents a rook placement on the special chessboard -/
structure RookPlacement where
  row : Nat
  column : Nat

/-- Checks if two rook placements attack each other on the special chessboard -/
def attacks (board : SpecialChessboard) (r1 r2 : RookPlacement) : Prop :=
  r1.row = r2.row ∨ r1.column = r2.column

/-- Counts the number of valid ways to place 3 rooks on the special chessboard -/
def countValidPlacements (board : SpecialChessboard) : Nat :=
  sorry

/-- The main theorem stating that there are 168 ways to place 3 rooks on the special chessboard -/
theorem rook_placements_count (board : SpecialChessboard) 
  (h1 : board.rows = 4) 
  (h2 : board.columns = 8) 
  (h3 : board.longRowLength = 8) : 
  countValidPlacements board = 168 := by
  sorry

end NUMINAMATH_CALUDE_rook_placements_count_l30_3017


namespace NUMINAMATH_CALUDE_senate_subcommittee_count_l30_3076

/-- The number of ways to form a subcommittee from a Senate committee -/
def subcommittee_ways (total_republicans : ℕ) (total_democrats : ℕ) 
  (subcommittee_republicans : ℕ) (min_subcommittee_democrats : ℕ) 
  (max_subcommittee_size : ℕ) : ℕ :=
  sorry

theorem senate_subcommittee_count : 
  subcommittee_ways 10 8 3 2 5 = 10080 := by sorry

end NUMINAMATH_CALUDE_senate_subcommittee_count_l30_3076


namespace NUMINAMATH_CALUDE_factor_expression_1_factor_expression_2_l30_3010

-- For the first expression
theorem factor_expression_1 (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (5 * x - 4 * y) * (x + 2 * y) := by
  sorry

-- For the second expression
theorem factor_expression_2 (x a : ℝ) :
  x^2 + 2 * x - 15 - a * x - 5 * a = (x + 5) * (x - 3 - a) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_1_factor_expression_2_l30_3010


namespace NUMINAMATH_CALUDE_expression_evaluation_l30_3096

theorem expression_evaluation : (5 ^ 2 : ℤ) + 15 / 3 - (3 * 2) ^ 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l30_3096


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l30_3094

-- Problem 1
theorem solve_equation_one (x : ℝ) : 4 * x^2 = 25 ↔ x = 5/2 ∨ x = -5/2 := by sorry

-- Problem 2
theorem solve_equation_two (x : ℝ) : (x + 1)^3 - 8 = 56 ↔ x = 3 := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l30_3094


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l30_3086

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 8. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 2 ∧ b = 2 ∧ c = 4 →  -- Two sides are 2, one side is 4
  a + b > c →              -- Triangle inequality
  a = b →                  -- Isosceles condition
  a + b + c = 8 :=         -- Perimeter is 8
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l30_3086


namespace NUMINAMATH_CALUDE_cistern_leak_emptying_time_l30_3015

/-- Given a cistern that fills in 8 hours without a leak and 9 hours with a leak,
    the time it takes for the leak to empty a full cistern is 72 hours. -/
theorem cistern_leak_emptying_time :
  ∀ (fill_rate_no_leak : ℝ) (fill_rate_with_leak : ℝ) (leak_rate : ℝ),
    fill_rate_no_leak = 1 / 8 →
    fill_rate_with_leak = 1 / 9 →
    fill_rate_with_leak = fill_rate_no_leak - leak_rate →
    (1 / leak_rate : ℝ) = 72 := by
  sorry


end NUMINAMATH_CALUDE_cistern_leak_emptying_time_l30_3015


namespace NUMINAMATH_CALUDE_smallest_integer_in_sequence_l30_3068

theorem smallest_integer_in_sequence (a b c : ℤ) : 
  a < b ∧ b < c ∧ c < 90 →
  (a + b + c + 90) / 4 = 72 →
  a ≥ 21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_sequence_l30_3068


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l30_3051

theorem square_plus_inverse_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l30_3051


namespace NUMINAMATH_CALUDE_second_number_value_l30_3047

theorem second_number_value (x y z : ℝ) : 
  z = 4.5 * y →
  y = 2.5 * x →
  (x + y + z) / 3 = 165 →
  y = 82.5 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l30_3047


namespace NUMINAMATH_CALUDE_problem_solution_l30_3043

def circle_times (a b : ℚ) : ℚ := (a + b) / (a - b)

def circle_plus (a b : ℚ) : ℚ := 2 * (circle_times a b)

theorem problem_solution : circle_plus (circle_plus 8 6) 2 = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l30_3043


namespace NUMINAMATH_CALUDE_hockey_team_selection_l30_3080

def number_of_players : ℕ := 18
def players_to_select : ℕ := 8

theorem hockey_team_selection :
  Nat.choose number_of_players players_to_select = 43758 := by
  sorry

end NUMINAMATH_CALUDE_hockey_team_selection_l30_3080


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l30_3020

/-- A geometric sequence with first four terms 25, -50, 100, -200 has a common ratio of -2. -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℚ), 
    (a 0 = 25) → 
    (a 1 = -50) → 
    (a 2 = 100) → 
    (a 3 = -200) → 
    (∀ n : ℕ, a (n + 1) = a n * (-2)) → 
    (∀ n : ℕ, a (n + 1) / a n = -2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l30_3020


namespace NUMINAMATH_CALUDE_full_spots_is_186_l30_3050

/-- Represents a parking garage with open spots on each level -/
structure ParkingGarage where
  levels : Nat
  spotsPerLevel : Nat
  openSpotsFirstLevel : Nat
  openSpotsSecondLevel : Nat
  openSpotsThirdLevel : Nat
  openSpotsFourthLevel : Nat

/-- Calculates the number of full parking spots in the garage -/
def fullParkingSpots (garage : ParkingGarage) : Nat :=
  garage.levels * garage.spotsPerLevel -
  (garage.openSpotsFirstLevel + garage.openSpotsSecondLevel +
   garage.openSpotsThirdLevel + garage.openSpotsFourthLevel)

/-- Theorem stating that the number of full parking spots is 186 -/
theorem full_spots_is_186 (garage : ParkingGarage)
  (h1 : garage.levels = 4)
  (h2 : garage.spotsPerLevel = 100)
  (h3 : garage.openSpotsFirstLevel = 58)
  (h4 : garage.openSpotsSecondLevel = garage.openSpotsFirstLevel + 2)
  (h5 : garage.openSpotsThirdLevel = garage.openSpotsSecondLevel + 5)
  (h6 : garage.openSpotsFourthLevel = 31) :
  fullParkingSpots garage = 186 := by
  sorry

#eval fullParkingSpots {
  levels := 4,
  spotsPerLevel := 100,
  openSpotsFirstLevel := 58,
  openSpotsSecondLevel := 60,
  openSpotsThirdLevel := 65,
  openSpotsFourthLevel := 31
}

end NUMINAMATH_CALUDE_full_spots_is_186_l30_3050


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_nonagon_l30_3078

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular polygon with n sides -/
def num_shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal in a regular polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  (num_shortest_diagonals n : ℚ) / (num_diagonals n : ℚ)

theorem prob_shortest_diagonal_nonagon :
  prob_shortest_diagonal 9 = 1/3 := by sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_nonagon_l30_3078


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l30_3088

theorem quadratic_root_implies_a (a : ℝ) :
  (3^2 + a*3 + a - 1 = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l30_3088


namespace NUMINAMATH_CALUDE_afternoon_session_count_l30_3087

/-- Represents the number of kids in each session for a sport -/
structure SportSessions :=
  (morning : ℕ)
  (afternoon : ℕ)
  (evening : ℕ)
  (undecided : ℕ)

/-- Calculates the total number of kids in afternoon sessions across all sports -/
def total_afternoon_kids (soccer : SportSessions) (basketball : SportSessions) (swimming : SportSessions) : ℕ :=
  soccer.afternoon + basketball.afternoon + swimming.afternoon

theorem afternoon_session_count :
  ∀ (total_kids : ℕ) 
    (soccer basketball swimming : SportSessions),
  total_kids = 2000 →
  soccer.morning + soccer.afternoon + soccer.evening + soccer.undecided = 400 →
  basketball.morning + basketball.afternoon + basketball.evening = 300 →
  swimming.morning + swimming.afternoon + swimming.evening = 300 →
  soccer.morning = 100 →
  soccer.afternoon = 280 →
  soccer.undecided = 20 →
  basketball.evening = 180 →
  basketball.morning = basketball.afternoon →
  swimming.morning = swimming.afternoon →
  swimming.afternoon = swimming.evening →
  ∃ (soccer_new basketball_new swimming_new : SportSessions),
    soccer_new.morning = soccer.morning + 30 →
    soccer_new.afternoon = soccer.afternoon - 30 →
    soccer_new.evening = soccer.evening →
    soccer_new.undecided = soccer.undecided →
    basketball_new = basketball →
    swimming_new.morning = swimming.morning + 15 →
    swimming_new.afternoon = swimming.afternoon - 15 →
    swimming_new.evening = swimming.evening →
    total_afternoon_kids soccer_new basketball_new swimming_new = 395 :=
by sorry

end NUMINAMATH_CALUDE_afternoon_session_count_l30_3087


namespace NUMINAMATH_CALUDE_fraction_problem_l30_3067

theorem fraction_problem (n d : ℕ) (h1 : d = 2*n - 1) (h2 : (n + 1) * 5 = (d + 1) * 3) : n = 5 ∧ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l30_3067


namespace NUMINAMATH_CALUDE_general_term_formula_l30_3031

/-- The sequence term for a given positive integer n -/
def a (n : ℕ+) : ℚ :=
  (4 * n^2 + n - 1) / (2 * n + 1)

/-- The first part of each term in the sequence -/
def b (n : ℕ+) : ℕ :=
  2 * n - 1

/-- The second part of each term in the sequence -/
def c (n : ℕ+) : ℚ :=
  n / (2 * n + 1)

/-- Theorem stating that the general term formula is correct -/
theorem general_term_formula (n : ℕ+) : 
  a n = (b n : ℚ) + c n :=
sorry

end NUMINAMATH_CALUDE_general_term_formula_l30_3031


namespace NUMINAMATH_CALUDE_ted_fruit_purchase_l30_3055

/-- The number of bananas Ted needs to purchase -/
def num_bananas : ℕ := 5

/-- The cost of one banana in dollars -/
def banana_cost : ℚ := 2

/-- The cost of one orange in dollars -/
def orange_cost : ℚ := 3/2

/-- The total amount Ted needs to spend on fruits in dollars -/
def total_cost : ℚ := 25

/-- The number of oranges Ted needs to purchase -/
def num_oranges : ℕ := 10

theorem ted_fruit_purchase :
  (num_bananas : ℚ) * banana_cost + (num_oranges : ℚ) * orange_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_ted_fruit_purchase_l30_3055


namespace NUMINAMATH_CALUDE_original_denominator_proof_l30_3056

theorem original_denominator_proof (d : ℚ) : 
  (2 : ℚ) / d ≠ 0 →    -- Ensure the fraction is well-defined
  (6 : ℚ) / (3 * d) = (2 : ℚ) / 3 → 
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l30_3056


namespace NUMINAMATH_CALUDE_system_solution_unique_l30_3052

theorem system_solution_unique :
  ∃! (x y z : ℝ),
    x^2 - 2*y + 1 = 0 ∧
    y^2 - 4*z + 7 = 0 ∧
    z^2 + 2*x - 2 = 0 ∧
    x = -1 ∧ y = 1 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l30_3052


namespace NUMINAMATH_CALUDE_xyz_congruence_l30_3021

theorem xyz_congruence (x y z : ℕ) : 
  x < 10 → y < 10 → z < 10 → x > 0 → y > 0 → z > 0 →
  (x * y * z) % 9 = 1 →
  (7 * z) % 9 = 4 →
  (8 * y) % 9 = (5 + y) % 9 →
  (x + y + z) % 9 = 2 := by sorry

end NUMINAMATH_CALUDE_xyz_congruence_l30_3021


namespace NUMINAMATH_CALUDE_cylinder_volume_l30_3058

/-- The volume of a cylinder with specific geometric conditions -/
theorem cylinder_volume (l α β : ℝ) (hl : l > 0) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) :
  ∃ V : ℝ, V = (π * l^3 * Real.sin (2*α) * Real.cos α^3) / (8 * Real.cos (α + β) * Real.cos (α - β)) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l30_3058


namespace NUMINAMATH_CALUDE_toy_price_difference_is_250_l30_3060

def toy_price_difference : ℝ → Prop :=
  λ price_diff : ℝ =>
    ∃ (a b : ℝ),
      a > 150 ∧ b > 150 ∧
      (∀ p : ℝ, a ≤ p ∧ p ≤ b →
        (0.2 * p ≥ 40 ∧ 0.2 * p ≥ 0.3 * (p - 150))) ∧
      (∀ p : ℝ, p < a ∨ p > b →
        (0.2 * p < 40 ∨ 0.2 * p < 0.3 * (p - 150))) ∧
      price_diff = b - a

theorem toy_price_difference_is_250 :
  toy_price_difference 250 :=
sorry

end NUMINAMATH_CALUDE_toy_price_difference_is_250_l30_3060


namespace NUMINAMATH_CALUDE_triangle_relations_l30_3092

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  let ⟨A, B, C, D⟩ := t
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 -- B is right angle

def BC_equals_2AB (t : Triangle) : Prop :=
  let ⟨A, B, C, D⟩ := t
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 * ((B.1 - A.1)^2 + (B.2 - A.2)^2)

def D_on_angle_bisector (t : Triangle) : Prop :=
  let ⟨A, B, C, D⟩ := t
  ∃ k : ℝ, 0 < k ∧ k < 1 ∧
    D.1 = A.1 + k * (C.1 - A.1) ∧
    D.2 = A.2 + k * (C.2 - A.2)

-- Theorem statement
theorem triangle_relations (t : Triangle) 
  (h1 : is_right_triangle t) 
  (h2 : BC_equals_2AB t) 
  (h3 : D_on_angle_bisector t) :
  let ⟨A, B, C, D⟩ := t
  (D.1 - B.1)^2 + (D.2 - B.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) * (Real.sin (18 * π / 180))^2 ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) * (Real.sin (36 * π / 180))^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_relations_l30_3092


namespace NUMINAMATH_CALUDE_point_satisfies_constraint_local_maximum_at_point_main_theorem_l30_3093

/-- The constraint function g(x₁, x₂) = x₁ - 2x₂ + 3 -/
def g (x₁ x₂ : ℝ) : ℝ := x₁ - 2*x₂ + 3

/-- The objective function f(x₁, x₂) = x₂² - x₁² -/
def f (x₁ x₂ : ℝ) : ℝ := x₂^2 - x₁^2

/-- The point (1, 2) satisfies the constraint -/
theorem point_satisfies_constraint : g 1 2 = 0 := by sorry

/-- The function f has a local maximum at (1, 2) under the constraint g(x₁, x₂) = 0 -/
theorem local_maximum_at_point :
  ∃ ε > 0, ∀ x₁ x₂ : ℝ, 
    g x₁ x₂ = 0 → 
    (x₁ - 1)^2 + (x₂ - 2)^2 < ε^2 → 
    f x₁ x₂ ≤ f 1 2 := by sorry

/-- The main theorem: f has a local maximum at (1, 2) under the constraint g(x₁, x₂) = 0 -/
theorem main_theorem : 
  ∃ (x₁ x₂ : ℝ), g x₁ x₂ = 0 ∧ 
  ∃ ε > 0, ∀ y₁ y₂ : ℝ, 
    g y₁ y₂ = 0 → 
    (y₁ - x₁)^2 + (y₂ - x₂)^2 < ε^2 → 
    f y₁ y₂ ≤ f x₁ x₂ :=
by
  use 1, 2
  constructor
  · exact point_satisfies_constraint
  · exact local_maximum_at_point

end NUMINAMATH_CALUDE_point_satisfies_constraint_local_maximum_at_point_main_theorem_l30_3093


namespace NUMINAMATH_CALUDE_expression_approximation_l30_3062

theorem expression_approximation :
  let x := ((69.28 * 0.004)^3 * Real.sin (Real.pi/3)) / (0.03^2 * Real.log 0.58 * Real.cos (Real.pi/4))
  ∃ ε > 0, |x + 37.644| < ε ∧ ε < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_expression_approximation_l30_3062


namespace NUMINAMATH_CALUDE_cube_inequality_l30_3046

theorem cube_inequality (a b : ℝ) (h : a < 0 ∧ 0 < b) : a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l30_3046


namespace NUMINAMATH_CALUDE_min_value_problem1_l30_3073

theorem min_value_problem1 (x : ℝ) (h : x > 3) : 4 / (x - 3) + x ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem1_l30_3073


namespace NUMINAMATH_CALUDE_overestimation_correct_l30_3054

/-- The overestimation in cents when y quarters are miscounted as half-dollars and y pennies are miscounted as nickels -/
def overestimation (y : ℕ) : ℕ := 29 * y

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a half-dollar in cents -/
def half_dollar_value : ℕ := 50

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

theorem overestimation_correct (y : ℕ) : 
  overestimation y = 
    y * (half_dollar_value - quarter_value) + 
    y * (nickel_value - penny_value) := by
  sorry

end NUMINAMATH_CALUDE_overestimation_correct_l30_3054


namespace NUMINAMATH_CALUDE_square_sum_value_l30_3042

theorem square_sum_value (x y : ℝ) :
  (x^2 + y^2 + 1) * (x^2 + y^2 + 2) = 6 → x^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l30_3042


namespace NUMINAMATH_CALUDE_percent_of_x_l30_3053

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 5 + x / 25) / x * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_l30_3053


namespace NUMINAMATH_CALUDE_coffee_bread_combinations_l30_3013

theorem coffee_bread_combinations (coffee_types bread_types : ℕ) 
  (h1 : coffee_types = 2) (h2 : bread_types = 3) : 
  coffee_types * bread_types = 6 := by
  sorry

end NUMINAMATH_CALUDE_coffee_bread_combinations_l30_3013


namespace NUMINAMATH_CALUDE_train_passing_time_train_passing_time_specific_l30_3079

/-- Time for a train to pass a tree -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (wind_speed : ℝ) : ℝ :=
  let train_speed_ms := train_speed * 1000 / 3600
  let wind_speed_ms := wind_speed * 1000 / 3600
  let effective_speed := train_speed_ms - wind_speed_ms
  train_length / effective_speed

/-- Proof that the time for a train of length 850 m, traveling at 85 km/hr against a 5 km/hr wind, to pass a tree is approximately 38.25 seconds -/
theorem train_passing_time_specific : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_passing_time 850 85 5 - 38.25| < ε :=
sorry

end NUMINAMATH_CALUDE_train_passing_time_train_passing_time_specific_l30_3079


namespace NUMINAMATH_CALUDE_dogs_accessible_area_l30_3018

theorem dogs_accessible_area (s : ℝ) (s_pos : s > 0) :
  let square_area := (2 * s) ^ 2
  let circle_area := π * s ^ 2
  circle_area / square_area = π / 4 := by
  sorry

#check dogs_accessible_area

end NUMINAMATH_CALUDE_dogs_accessible_area_l30_3018


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l30_3027

theorem geometric_sequence_sum (a r : ℝ) : 
  a + a * r = 7 →
  a * (r^6 - 1) / (r - 1) = 91 →
  a + a * r + a * r^2 + a * r^3 = 28 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l30_3027


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l30_3040

-- Problem 1
theorem problem_1 : (-1)^4 - 2 * Real.tan (60 * π / 180) + (Real.sqrt 3 - Real.sqrt 2)^0 + Real.sqrt 12 = 2 := by
  sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, (x - 1) / 3 ≥ x / 2 - 2 ↔ x ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l30_3040


namespace NUMINAMATH_CALUDE_inequality_proof_l30_3033

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * Real.sqrt (a * b)) / (Real.sqrt a + Real.sqrt b) ≤ (a * b) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l30_3033


namespace NUMINAMATH_CALUDE_simplify_expression_l30_3095

theorem simplify_expression : 
  2 - (2 / (2 + Real.sqrt 5)) + (2 / (2 - Real.sqrt 5)) = 2 - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l30_3095


namespace NUMINAMATH_CALUDE_exam_score_exam_score_specific_case_l30_3038

theorem exam_score (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : ℕ :=
  let wrong_answers := total_questions - correct_answers
  let total_marks := correct_answers * marks_per_correct - wrong_answers * marks_lost_per_wrong
  total_marks

theorem exam_score_specific_case : exam_score 75 40 4 1 = 125 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_exam_score_specific_case_l30_3038


namespace NUMINAMATH_CALUDE_zero_subset_A_l30_3048

def A : Set ℕ := {x | x < 4}

theorem zero_subset_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_zero_subset_A_l30_3048


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l30_3012

theorem solve_cubic_equation (t p s : ℝ) : 
  t = 3 * s^3 + 2 * p → t = 29 → p = 3 → s = (23/3)^(1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l30_3012


namespace NUMINAMATH_CALUDE_article_cost_price_l30_3071

theorem article_cost_price (loss_percentage : Real) (gain_percentage : Real) (price_increase : Real) 
  (cost_price : Real) :
  loss_percentage = 0.15 →
  gain_percentage = 0.125 →
  price_increase = 72.50 →
  (1 - loss_percentage) * cost_price + price_increase = (1 + gain_percentage) * cost_price →
  cost_price = 263.64 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l30_3071


namespace NUMINAMATH_CALUDE_initial_ratio_problem_l30_3019

theorem initial_ratio_problem (a b : ℕ) : 
  b = 6 → 
  (a + 2 : ℚ) / (b + 2 : ℚ) = 3 / 2 → 
  (a : ℚ) / b = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_initial_ratio_problem_l30_3019


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_110_l30_3016

/-- Given an arithmetic progression with first term a and common difference d -/
def arithmetic_progression (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1) * d

/-- Sum of first n terms of an arithmetic progression -/
def sum_arithmetic_progression (a d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1) * d)

theorem arithmetic_progression_sum_110 
  (a d : ℚ) 
  (h1 : sum_arithmetic_progression a d 10 = 100)
  (h2 : sum_arithmetic_progression a d 100 = 10) :
  sum_arithmetic_progression a d 110 = -110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_110_l30_3016


namespace NUMINAMATH_CALUDE_factory_B_is_better_l30_3001

/-- Represents a chicken leg factory --/
structure ChickenFactory where
  name : String
  mean : ℝ
  median : ℝ
  mode : ℝ
  variance : ℝ

/-- Determines if a factory is better based on its statistics --/
def isBetterFactory (f1 f2 : ChickenFactory) : Prop :=
  f1.mean = f2.mean ∧
  f1.variance < f2.variance ∧
  f1.median = f1.mean ∧
  f1.mode = f1.mean ∧
  (f2.median ≠ f2.mean ∨ f2.mode ≠ f2.mean)

/-- Factory A data --/
def factoryA : ChickenFactory :=
  { name := "A"
    mean := 75
    median := 74.5
    mode := 74
    variance := 3.4 }

/-- Factory B data --/
def factoryB : ChickenFactory :=
  { name := "B"
    mean := 75
    median := 75
    mode := 75
    variance := 2 }

/-- Theorem stating that Factory B is better than Factory A --/
theorem factory_B_is_better : isBetterFactory factoryB factoryA := by
  sorry

#check factory_B_is_better

end NUMINAMATH_CALUDE_factory_B_is_better_l30_3001


namespace NUMINAMATH_CALUDE_cos_negative_300_degrees_l30_3034

theorem cos_negative_300_degrees :
  Real.cos ((-300 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_300_degrees_l30_3034


namespace NUMINAMATH_CALUDE_vectors_opposite_directions_l30_3049

def a (x : ℝ) : ℝ × ℝ := (1, -x)
def b (x : ℝ) : ℝ × ℝ := (x, -16)

theorem vectors_opposite_directions :
  ∃ (k : ℝ), k ≠ 0 ∧ a (-5) = k • b (-5) :=
by sorry

end NUMINAMATH_CALUDE_vectors_opposite_directions_l30_3049
