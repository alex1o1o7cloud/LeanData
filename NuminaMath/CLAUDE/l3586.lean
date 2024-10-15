import Mathlib

namespace NUMINAMATH_CALUDE_fence_painting_combinations_l3586_358628

def number_of_colors : ℕ := 5
def number_of_tools : ℕ := 4

theorem fence_painting_combinations :
  number_of_colors * number_of_tools = 20 := by
  sorry

end NUMINAMATH_CALUDE_fence_painting_combinations_l3586_358628


namespace NUMINAMATH_CALUDE_nicky_pace_is_3_l3586_358605

/-- Nicky's pace in meters per second -/
def nicky_pace : ℝ := 3

/-- Cristina's pace in meters per second -/
def cristina_pace : ℝ := 5

/-- Head start given to Nicky in meters -/
def head_start : ℝ := 48

/-- Time it takes Cristina to catch up to Nicky in seconds -/
def catch_up_time : ℝ := 24

/-- Theorem stating that Nicky's pace is 3 meters per second given the conditions -/
theorem nicky_pace_is_3 :
  cristina_pace > nicky_pace ∧
  cristina_pace * catch_up_time = nicky_pace * catch_up_time + head_start →
  nicky_pace = 3 := by
  sorry


end NUMINAMATH_CALUDE_nicky_pace_is_3_l3586_358605


namespace NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l3586_358624

/-- Given a circle and three points forming part of a rectangle, 
    this theorem proves the coordinates of the fourth vertex. -/
theorem fourth_vertex_of_rectangle 
  (O : ℝ × ℝ) (R : ℝ) 
  (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) 
  (h_circle : (x₁ - O.1)^2 + (y₁ - O.2)^2 = R^2 ∧ (x₂ - O.1)^2 + (y₂ - O.2)^2 = R^2)
  (h_inside : (x₀ - O.1)^2 + (y₀ - O.2)^2 < R^2) :
  ∃ (x₄ y₄ : ℝ), 
    (x₄ = x₁ + x₂ - x₀ ∧ y₄ = y₁ + y₂ - y₀) ∧
    ((x₄ - O.1)^2 + (y₄ - O.2)^2 = R^2) ∧
    ((x₄ - x₀)^2 + (y₄ - y₀)^2 = (x₁ - x₂)^2 + (y₁ - y₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l3586_358624


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3586_358644

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + 8*x + b = (x + k)^2) → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3586_358644


namespace NUMINAMATH_CALUDE_train_passing_platform_l3586_358630

/-- Given a train of length 240 meters passing a pole in 24 seconds,
    prove that it takes 89 seconds to pass a platform of length 650 meters. -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (pole_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 240)
  (h2 : pole_passing_time = 24)
  (h3 : platform_length = 650) :
  (train_length + platform_length) / (train_length / pole_passing_time) = 89 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_platform_l3586_358630


namespace NUMINAMATH_CALUDE_factorization_theorem1_factorization_theorem2_l3586_358674

-- For the first expression
theorem factorization_theorem1 (x y : ℝ) :
  3 * (x + y) * (x - y) - (x - y)^2 = 2 * (x - y) * (x + 2*y) := by sorry

-- For the second expression
theorem factorization_theorem2 (x y : ℝ) :
  x^2 * (y^2 - 1) + 2*x * (y^2 - 1) = x * (y + 1) * (y - 1) * (x + 2) := by sorry

end NUMINAMATH_CALUDE_factorization_theorem1_factorization_theorem2_l3586_358674


namespace NUMINAMATH_CALUDE_prob_odd_females_committee_l3586_358600

/-- The number of men in the pool of candidates -/
def num_men : ℕ := 5

/-- The number of women in the pool of candidates -/
def num_women : ℕ := 4

/-- The size of the committee to be formed -/
def committee_size : ℕ := 3

/-- The probability of selecting a committee with an odd number of female members -/
def prob_odd_females : ℚ := 11 / 21

/-- Theorem stating that the probability of selecting a committee of three members
    with an odd number of female members from a pool of five men and four women,
    where all candidates are equally likely to be chosen, is 11/21 -/
theorem prob_odd_females_committee :
  let total_candidates := num_men + num_women
  let total_committees := Nat.choose total_candidates committee_size
  let committees_one_female := Nat.choose num_women 1 * Nat.choose num_men 2
  let committees_three_females := Nat.choose num_women 3 * Nat.choose num_men 0
  let favorable_outcomes := committees_one_female + committees_three_females
  (favorable_outcomes : ℚ) / total_committees = prob_odd_females := by
  sorry


end NUMINAMATH_CALUDE_prob_odd_females_committee_l3586_358600


namespace NUMINAMATH_CALUDE_arrangement_count_l3586_358692

def number_of_people : Nat := 6
def number_of_special_people : Nat := 3

theorem arrangement_count : 
  (number_of_people : Nat) = 6 →
  (number_of_special_people : Nat) = 3 →
  (∃ (arrangement_count : Nat), arrangement_count = 144 ∧
    arrangement_count = (number_of_people - number_of_special_people).factorial * 
                        (number_of_people - number_of_special_people + 1).choose number_of_special_people) :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3586_358692


namespace NUMINAMATH_CALUDE_operations_for_106_triangles_l3586_358658

/-- The number of triangles after n operations -/
def num_triangles (n : ℕ) : ℕ := 4 + 3 * (n - 1)

theorem operations_for_106_triangles :
  ∃ n : ℕ, n > 0 ∧ num_triangles n = 106 ∧ n = 35 := by sorry

end NUMINAMATH_CALUDE_operations_for_106_triangles_l3586_358658


namespace NUMINAMATH_CALUDE_new_person_weight_l3586_358648

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (leaving_weight : ℝ) (average_increase : ℝ) : ℝ :=
  leaving_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 65 kg -/
theorem new_person_weight :
  weight_of_new_person 8 45 2.5 = 65 := by
  sorry

#eval weight_of_new_person 8 45 2.5

end NUMINAMATH_CALUDE_new_person_weight_l3586_358648


namespace NUMINAMATH_CALUDE_infinitely_many_coprimes_in_arithmetic_sequence_l3586_358665

theorem infinitely_many_coprimes_in_arithmetic_sequence 
  (a b m : ℕ+) (h : Nat.Coprime a b) :
  ∃ (s : Set ℕ), Set.Infinite s ∧ ∀ k ∈ s, Nat.Coprime (a + k * b) m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_coprimes_in_arithmetic_sequence_l3586_358665


namespace NUMINAMATH_CALUDE_parabola_sum_l3586_358662

/-- Represents a parabola of the form y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- The area of the kite formed by the intersections of the parabolas with the axes -/
def kite_area (p1 p2 : Parabola) : ℝ := 12

/-- The parabolas intersect the coordinate axes at exactly four points -/
def intersect_at_four_points (p1 p2 : Parabola) : Prop := sorry

theorem parabola_sum (p1 p2 : Parabola) 
  (h1 : p1.c = -2 ∧ p2.c = 4) 
  (h2 : intersect_at_four_points p1 p2) 
  (h3 : kite_area p1 p2 = 12) : 
  p1.a + p2.a = 1.5 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_l3586_358662


namespace NUMINAMATH_CALUDE_opposite_of_reciprocal_of_negative_five_l3586_358677

theorem opposite_of_reciprocal_of_negative_five :
  -(1 / -5) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_reciprocal_of_negative_five_l3586_358677


namespace NUMINAMATH_CALUDE_saree_original_price_l3586_358641

theorem saree_original_price (P : ℝ) : 
  (P * (1 - 0.2) * (1 - 0.3) = 313.6) → P = 560 := by
  sorry

end NUMINAMATH_CALUDE_saree_original_price_l3586_358641


namespace NUMINAMATH_CALUDE_min_value_ratio_l3586_358637

theorem min_value_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ (x y : ℝ), x > y ∧ y > 0 ∧ 
    2*x + y + 1/(x-y) + 4/(x+2*y) < 2*a + b + 1/(a-b) + 4/(a+2*b)) ∨
  a/b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_ratio_l3586_358637


namespace NUMINAMATH_CALUDE_kody_half_age_of_mohamed_l3586_358682

def years_ago (mohamed_current_age kody_current_age : ℕ) : ℕ :=
  let x : ℕ := 4
  x

theorem kody_half_age_of_mohamed (mohamed_current_age kody_current_age : ℕ)
  (h1 : mohamed_current_age = 2 * 30)
  (h2 : kody_current_age = 32)
  (h3 : ∃ x : ℕ, kody_current_age - x = (mohamed_current_age - x) / 2) :
  years_ago mohamed_current_age kody_current_age = 4 := by
sorry

end NUMINAMATH_CALUDE_kody_half_age_of_mohamed_l3586_358682


namespace NUMINAMATH_CALUDE_max_xy_value_fraction_inequality_l3586_358653

-- Part I
theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 5 * y = 20) :
  ∃ (max_val : ℝ), max_val = 10 ∧ ∀ (z : ℝ), x * y ≤ z → z ≤ max_val :=
sorry

-- Part II
theorem fraction_inequality (a b c d k : ℝ) (hab : a > b) (hb : b > 0) (hcd : c < d) (hd : d < 0) (hk : k < 0) :
  k / (a - c) > k / (b - d) :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_fraction_inequality_l3586_358653


namespace NUMINAMATH_CALUDE_min_k_value_l3586_358699

theorem min_k_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ k : ℝ, (1/a + 1/b + k/(a+b) ≥ 0)) : 
  ∃ k_min : ℝ, k_min = -4 ∧ ∀ k : ℝ, (1/a + 1/b + k/(a+b) ≥ 0) → k ≥ k_min :=
sorry

end NUMINAMATH_CALUDE_min_k_value_l3586_358699


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l3586_358647

theorem coefficient_x_cubed_in_binomial_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) * (1 ^ (5 - k)) * (1 ^ k) * (if k = 3 then 1 else 0)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l3586_358647


namespace NUMINAMATH_CALUDE_sum_mod_seven_l3586_358697

theorem sum_mod_seven : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l3586_358697


namespace NUMINAMATH_CALUDE_two_from_three_permutations_l3586_358631

/-- The number of permutations of k items chosen from n items. -/
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

/-- Theorem: There are 6 ways to choose and line up 2 people from a group of 3. -/
theorem two_from_three_permutations :
  permutations 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_from_three_permutations_l3586_358631


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3586_358607

/-- The function f(x) = ax^2 + 4x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x - 1

/-- Predicate indicating that the graph of f has only one common point with the x-axis -/
def has_one_common_point (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Statement: a = -4 is a sufficient but not necessary condition for
    the graph of f to have only one common point with the x-axis -/
theorem sufficient_not_necessary :
  (has_one_common_point (-4)) ∧ 
  (∃ a : ℝ, a ≠ -4 ∧ has_one_common_point a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3586_358607


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3586_358611

/-- The constant term in the expansion of (1+2x^2)(x-1/x)^8 is -42 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (1 + 2*x^2) * (x - 1/x)^8
  ∃ g : ℝ → ℝ, (∀ x ≠ 0, f x = g x) ∧ g 0 = -42 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3586_358611


namespace NUMINAMATH_CALUDE_pizza_toppings_theorem_l3586_358617

/-- Represents the number of distinct toppings on a pizza slice -/
def toppings_on_slice (n k : ℕ+) (t : Fin (2 * k)) : ℕ :=
  sorry

/-- The minimum number of distinct toppings on any slice -/
def min_toppings (n k : ℕ+) : ℕ :=
  sorry

/-- The maximum number of distinct toppings on any slice -/
def max_toppings (n k : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that the sum of minimum and maximum toppings equals the total number of toppings -/
theorem pizza_toppings_theorem (n k : ℕ+) :
    min_toppings n k + max_toppings n k = n :=
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_theorem_l3586_358617


namespace NUMINAMATH_CALUDE_min_points_to_guarantee_highest_score_eighteen_points_achievable_smallest_points_to_guarantee_highest_score_l3586_358696

/-- Represents the possible points for a single race -/
inductive RacePoints
  | first : RacePoints
  | second : RacePoints
  | third : RacePoints
  | fourth : RacePoints

/-- Converts RacePoints to their numerical value -/
def pointValue (p : RacePoints) : Nat :=
  match p with
  | .first => 7
  | .second => 4
  | .third => 2
  | .fourth => 1

/-- Calculates the total points for a sequence of three races -/
def totalPoints (r1 r2 r3 : RacePoints) : Nat :=
  pointValue r1 + pointValue r2 + pointValue r3

/-- Theorem stating that 18 points is the minimum to guarantee the highest score -/
theorem min_points_to_guarantee_highest_score :
  ∀ (r1 r2 r3 : RacePoints),
    totalPoints r1 r2 r3 ≥ 18 →
    ∀ (s1 s2 s3 : RacePoints),
      totalPoints r1 r2 r3 > totalPoints s1 s2 s3 ∨
      (r1 = s1 ∧ r2 = s2 ∧ r3 = s3) :=
by sorry

/-- Theorem stating that 18 points is achievable -/
theorem eighteen_points_achievable :
  ∃ (r1 r2 r3 : RacePoints), totalPoints r1 r2 r3 = 18 :=
by sorry

/-- Main theorem combining the above results -/
theorem smallest_points_to_guarantee_highest_score :
  (∃ (r1 r2 r3 : RacePoints), totalPoints r1 r2 r3 = 18) ∧
  (∀ (r1 r2 r3 : RacePoints),
    totalPoints r1 r2 r3 ≥ 18 →
    ∀ (s1 s2 s3 : RacePoints),
      totalPoints r1 r2 r3 > totalPoints s1 s2 s3 ∨
      (r1 = s1 ∧ r2 = s2 ∧ r3 = s3)) ∧
  (∀ n : Nat, n < 18 →
    ∃ (s1 s2 s3 r1 r2 r3 : RacePoints),
      totalPoints s1 s2 s3 = n ∧
      totalPoints r1 r2 r3 > n) :=
by sorry

end NUMINAMATH_CALUDE_min_points_to_guarantee_highest_score_eighteen_points_achievable_smallest_points_to_guarantee_highest_score_l3586_358696


namespace NUMINAMATH_CALUDE_count_scalene_triangles_l3586_358615

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ a + b + c < 16

theorem count_scalene_triangles :
  ∃! (triangles : Finset (ℕ × ℕ × ℕ)),
    triangles.card = 6 ∧
    ∀ (t : ℕ × ℕ × ℕ), t ∈ triangles ↔ is_valid_scalene_triangle t.1 t.2.1 t.2.2 :=
by sorry

end NUMINAMATH_CALUDE_count_scalene_triangles_l3586_358615


namespace NUMINAMATH_CALUDE_total_spent_on_games_l3586_358649

def batman_game_cost : ℚ := 13.60
def superman_game_cost : ℚ := 5.06

theorem total_spent_on_games : batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_games_l3586_358649


namespace NUMINAMATH_CALUDE_one_third_displayed_l3586_358672

/-- Represents an art gallery with paintings and sculptures -/
structure ArtGallery where
  total_pieces : ℕ
  displayed_pieces : ℕ
  displayed_sculptures : ℕ
  not_displayed_paintings : ℕ
  not_displayed_sculptures : ℕ

/-- Conditions for the art gallery problem -/
def gallery_conditions (g : ArtGallery) : Prop :=
  g.total_pieces = 900 ∧
  g.not_displayed_sculptures = 400 ∧
  g.displayed_sculptures = g.displayed_pieces / 6 ∧
  g.not_displayed_paintings = (g.total_pieces - g.displayed_pieces) / 3

/-- Theorem stating that 1/3 of the pieces are displayed -/
theorem one_third_displayed (g : ArtGallery) 
  (h : gallery_conditions g) : 
  g.displayed_pieces = g.total_pieces / 3 := by
  sorry

#check one_third_displayed

end NUMINAMATH_CALUDE_one_third_displayed_l3586_358672


namespace NUMINAMATH_CALUDE_jim_journey_remaining_distance_l3586_358622

/-- Calculates the remaining distance to drive given the total journey distance and the distance already driven. -/
def remaining_distance (total_distance driven_distance : ℕ) : ℕ :=
  total_distance - driven_distance

/-- Theorem stating that for a 1200-mile journey with 923 miles driven, the remaining distance is 277 miles. -/
theorem jim_journey_remaining_distance :
  remaining_distance 1200 923 = 277 := by
  sorry

end NUMINAMATH_CALUDE_jim_journey_remaining_distance_l3586_358622


namespace NUMINAMATH_CALUDE_opposite_unit_vector_l3586_358616

def a : ℝ × ℝ := (4, 3)

theorem opposite_unit_vector (a : ℝ × ℝ) :
  let magnitude := Real.sqrt (a.1^2 + a.2^2)
  let opposite_unit := (-a.1 / magnitude, -a.2 / magnitude)
  opposite_unit = (-4/5, -3/5) ∧
  opposite_unit.1^2 + opposite_unit.2^2 = 1 ∧
  a.1 * opposite_unit.1 + a.2 * opposite_unit.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_opposite_unit_vector_l3586_358616


namespace NUMINAMATH_CALUDE_deandre_jordan_free_throws_l3586_358646

/-- The probability of scoring at least one point in two free throw attempts -/
def prob_at_least_one_point (success_rate : ℝ) : ℝ :=
  1 - (1 - success_rate) ^ 2

theorem deandre_jordan_free_throws :
  let success_rate : ℝ := 0.4
  prob_at_least_one_point success_rate = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_deandre_jordan_free_throws_l3586_358646


namespace NUMINAMATH_CALUDE_complement_A_union_B_when_a_is_5_A_union_B_equals_A_iff_l3586_358679

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | a - 2 ≤ x ∧ x ≤ 2*a - 3}

-- Statement for part 1
theorem complement_A_union_B_when_a_is_5 :
  (Set.univ \ A) ∪ B 5 = {x : ℝ | x ≤ -3 ∨ x ≥ 3} := by sorry

-- Statement for part 2
theorem A_union_B_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ a < 9/2 := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_when_a_is_5_A_union_B_equals_A_iff_l3586_358679


namespace NUMINAMATH_CALUDE_player_percentage_of_team_points_l3586_358684

def three_point_goals : ℕ := 5
def two_point_goals : ℕ := 10
def team_total_points : ℕ := 70

def player_points : ℕ := three_point_goals * 3 + two_point_goals * 2

theorem player_percentage_of_team_points :
  (player_points : ℚ) / team_total_points * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_player_percentage_of_team_points_l3586_358684


namespace NUMINAMATH_CALUDE_distribute_four_to_three_l3586_358609

/-- The number of ways to distribute n students to k villages, where each village gets at least one student -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 students to 3 villages results in 36 different plans -/
theorem distribute_four_to_three : distribute_students 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_to_three_l3586_358609


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3586_358654

theorem quadratic_discriminant :
  let a : ℚ := 5
  let b : ℚ := 5 + 1/5
  let c : ℚ := 1/5
  let discriminant := b^2 - 4*a*c
  discriminant = 576/25 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3586_358654


namespace NUMINAMATH_CALUDE_negation_true_l3586_358685

theorem negation_true : 
  ¬(∀ a : ℝ, a ≤ 3 → a^2 < 9) ↔ True :=
by sorry

end NUMINAMATH_CALUDE_negation_true_l3586_358685


namespace NUMINAMATH_CALUDE_probability_factor_less_than_eight_l3586_358626

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem probability_factor_less_than_eight :
  let f := factors 120
  (f.filter (· < 8)).card / f.card = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_eight_l3586_358626


namespace NUMINAMATH_CALUDE_unique_solution_xyz_l3586_358691

theorem unique_solution_xyz : 
  ∀ x y z : ℕ+, 
    (x : ℤ) + (y : ℤ)^2 + (z : ℤ)^3 = (x : ℤ) * (y : ℤ) * (z : ℤ) → 
    z = Nat.gcd x y → 
    (x = 5 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_xyz_l3586_358691


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3586_358606

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ordering of numbers
  b = 10 ∧  -- Median is 10
  (a + b + c) / 3 = a + 20 ∧  -- Mean is 20 more than least
  (a + b + c) / 3 = c - 25  -- Mean is 25 less than greatest
  → a + b + c = 45 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3586_358606


namespace NUMINAMATH_CALUDE_piggy_bank_savings_l3586_358636

/-- Calculates the remaining amount in a piggy bank after a year of regular spending -/
theorem piggy_bank_savings (initial_amount : ℕ) (spending_per_trip : ℕ) (trips_per_month : ℕ) (months_per_year : ℕ) :
  initial_amount = 200 →
  spending_per_trip = 2 →
  trips_per_month = 4 →
  months_per_year = 12 →
  initial_amount - (spending_per_trip * trips_per_month * months_per_year) = 104 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_savings_l3586_358636


namespace NUMINAMATH_CALUDE_range_of_k_for_inequality_l3586_358613

/-- Given functions f and g, prove the range of k for which g(x) ≥ k(x) holds. -/
theorem range_of_k_for_inequality (f g : ℝ → ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f x = Real.log x) →
  (∀ x : ℝ, g x = x - 1) →
  (∀ x : ℝ, x ≥ 0 → g x ≥ k * x) ↔ k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_for_inequality_l3586_358613


namespace NUMINAMATH_CALUDE_parallelogram_area_l3586_358651

theorem parallelogram_area (base height : ℝ) (h1 : base = 36) (h2 : height = 18) :
  base * height = 648 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3586_358651


namespace NUMINAMATH_CALUDE_work_completion_time_l3586_358643

/-- The number of days it takes for person A to complete the work alone -/
def days_A : ℝ := 15

/-- The fraction of work completed by A and B together in 5 days -/
def work_completed : ℝ := 0.5

/-- The number of days A and B work together -/
def days_together : ℝ := 5

/-- The number of days it takes for person B to complete the work alone -/
def days_B : ℝ := 30

theorem work_completion_time :
  (1 / days_A + 1 / days_B) * days_together = work_completed := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3586_358643


namespace NUMINAMATH_CALUDE_line_slope_points_l3586_358604

/-- Given m > 0 and three points on a line with slope m^2, prove m = √3 --/
theorem line_slope_points (m : ℝ) 
  (h_pos : m > 0)
  (h_line : ∃ (k b : ℝ), k = m^2 ∧ 
    3 = k * m + b ∧ 
    m = k * 1 + b ∧ 
    m^2 = k * 2 + b) : 
  m = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_points_l3586_358604


namespace NUMINAMATH_CALUDE_tank_emptying_l3586_358639

theorem tank_emptying (tank_capacity : ℝ) : 
  (3/4 * tank_capacity - 1/3 * tank_capacity = 15) → 
  (1/3 * tank_capacity = 12) :=
by sorry

end NUMINAMATH_CALUDE_tank_emptying_l3586_358639


namespace NUMINAMATH_CALUDE_fifty_second_card_is_ten_l3586_358693

def card_sequence : Fin 14 → String
| 0 => "A"
| 1 => "2"
| 2 => "3"
| 3 => "4"
| 4 => "5"
| 5 => "6"
| 6 => "7"
| 7 => "8"
| 8 => "9"
| 9 => "10"
| 10 => "J"
| 11 => "Q"
| 12 => "K"
| 13 => "Joker"

def nth_card (n : Nat) : String :=
  card_sequence (n % 14)

theorem fifty_second_card_is_ten :
  nth_card 51 = "10" := by
  sorry

end NUMINAMATH_CALUDE_fifty_second_card_is_ten_l3586_358693


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_is_11_l3586_358652

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the y-intercept of the common external tangent to two circles -/
noncomputable def commonExternalTangentYIntercept (c1 c2 : Circle) : ℝ :=
  sorry

/-- Theorem stating that the y-intercept of the common external tangent is 11 -/
theorem common_external_tangent_y_intercept_is_11 :
  let c1 : Circle := { center := (1, 3), radius := 3 }
  let c2 : Circle := { center := (10, 6), radius := 5 }
  commonExternalTangentYIntercept c1 c2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_is_11_l3586_358652


namespace NUMINAMATH_CALUDE_percentage_runs_from_running_approx_l3586_358659

def total_runs : ℕ := 120
def num_boundaries : ℕ := 5
def num_sixes : ℕ := 5
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

def runs_from_boundaries : ℕ := num_boundaries * runs_per_boundary
def runs_from_sixes : ℕ := num_sixes * runs_per_six
def runs_without_running : ℕ := runs_from_boundaries + runs_from_sixes
def runs_from_running : ℕ := total_runs - runs_without_running

theorem percentage_runs_from_running_approx (ε : ℚ) (h : ε > 0) :
  ∃ (p : ℚ), abs (p - (runs_from_running : ℚ) / (total_runs : ℚ) * 100) < ε ∧ 
             abs (p - 58.33) < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_runs_from_running_approx_l3586_358659


namespace NUMINAMATH_CALUDE_total_amount_after_four_years_l3586_358619

/-- Jo's annual earnings in USD -/
def annual_earnings : ℕ := 3^5 - 3^4 + 3^3 - 3^2 + 3

/-- Annual investment return in USD -/
def investment_return : ℕ := 2^5 - 2^4 + 2^3 - 2^2 + 2

/-- Number of years -/
def years : ℕ := 4

/-- Theorem stating the total amount after four years -/
theorem total_amount_after_four_years : 
  (annual_earnings + investment_return) * years = 820 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_after_four_years_l3586_358619


namespace NUMINAMATH_CALUDE_inequality_proof_l3586_358668

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3586_358668


namespace NUMINAMATH_CALUDE_solution_values_l3586_358695

-- Define the solution sets
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the equation with parameters a and b
def equation (a b : ℝ) (x : ℝ) : Prop := x^2 + a*x + b = 0

-- Theorem statement
theorem solution_values :
  ∃ (a b : ℝ), A_intersect_B = {x | equation a b x ∧ x^2 + a*x + b < 0} ∧ a = -1 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_values_l3586_358695


namespace NUMINAMATH_CALUDE_triangle_tangent_half_angles_sum_l3586_358632

theorem triangle_tangent_half_angles_sum (A B C : ℝ) 
  (h : A + B + C = π) : 
  Real.tan (A/2) * Real.tan (B/2) + Real.tan (B/2) * Real.tan (C/2) + Real.tan (C/2) * Real.tan (A/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_half_angles_sum_l3586_358632


namespace NUMINAMATH_CALUDE_hat_count_l3586_358608

/-- The number of hats in the box -/
def num_hats : ℕ := 3

/-- The set of all hats in the box -/
def Hats : Type := Fin num_hats

/-- A hat is red -/
def is_red : Hats → Prop := sorry

/-- A hat is blue -/
def is_blue : Hats → Prop := sorry

/-- A hat is yellow -/
def is_yellow : Hats → Prop := sorry

/-- All but 2 hats are red -/
axiom red_condition : ∃ (a b : Hats), a ≠ b ∧ ∀ (h : Hats), h ≠ a ∧ h ≠ b → is_red h

/-- All but 2 hats are blue -/
axiom blue_condition : ∃ (a b : Hats), a ≠ b ∧ ∀ (h : Hats), h ≠ a ∧ h ≠ b → is_blue h

/-- All but 2 hats are yellow -/
axiom yellow_condition : ∃ (a b : Hats), a ≠ b ∧ ∀ (h : Hats), h ≠ a ∧ h ≠ b → is_yellow h

/-- The main theorem: There are exactly 3 hats in the box -/
theorem hat_count : num_hats = 3 := by sorry

end NUMINAMATH_CALUDE_hat_count_l3586_358608


namespace NUMINAMATH_CALUDE_dining_group_size_l3586_358614

theorem dining_group_size (total_bill : ℝ) (tip_percentage : ℝ) (individual_payment : ℝ) : 
  total_bill = 139 ∧ tip_percentage = 0.1 ∧ individual_payment = 50.97 →
  ∃ n : ℕ, n = 3 ∧ n * individual_payment = total_bill * (1 + tip_percentage) := by
sorry

end NUMINAMATH_CALUDE_dining_group_size_l3586_358614


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3586_358689

theorem solution_satisfies_system :
  let x : ℚ := 7/2
  let y : ℚ := 1/2
  (2 * x + 4 * y = 9) ∧ (3 * x - 5 * y = 8) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3586_358689


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3586_358690

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- The sum of specific repeating decimals -/
theorem repeating_decimal_sum :
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3586_358690


namespace NUMINAMATH_CALUDE_consumer_installment_credit_l3586_358629

theorem consumer_installment_credit (total_credit : ℝ) : 
  (0.36 * total_credit = 3 * 57) → total_credit = 475 := by
  sorry

end NUMINAMATH_CALUDE_consumer_installment_credit_l3586_358629


namespace NUMINAMATH_CALUDE_product_of_largest_primes_l3586_358675

def largest_one_digit_primes : List Nat := [7, 5]
def largest_two_digit_prime : Nat := 97
def largest_three_digit_prime : Nat := 997

theorem product_of_largest_primes : 
  (List.prod largest_one_digit_primes) * largest_two_digit_prime * largest_three_digit_prime = 3383815 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_primes_l3586_358675


namespace NUMINAMATH_CALUDE_first_player_always_wins_l3586_358623

/-- Represents a cubic polynomial of the form x^3 + ax^2 + bx + c -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determines if a cubic polynomial has exactly one real root -/
def has_exactly_one_real_root (p : CubicPolynomial) : Prop :=
  ∃! x : ℝ, x^3 + p.a * x^2 + p.b * x + p.c = 0

/-- Represents a strategy for the first player -/
def first_player_strategy : CubicPolynomial → CubicPolynomial → CubicPolynomial :=
  sorry

/-- Represents a strategy for the second player -/
def second_player_strategy : CubicPolynomial → CubicPolynomial :=
  sorry

/-- The main theorem stating that the first player can always win -/
theorem first_player_always_wins :
  ∀ (second_strategy : CubicPolynomial → CubicPolynomial),
    ∃ (first_strategy : CubicPolynomial → CubicPolynomial → CubicPolynomial),
      ∀ (initial : CubicPolynomial),
        has_exactly_one_real_root (first_strategy initial (second_strategy initial)) :=
sorry

end NUMINAMATH_CALUDE_first_player_always_wins_l3586_358623


namespace NUMINAMATH_CALUDE_onion_chopping_difference_l3586_358603

/-- Represents the rate of chopping onions in terms of number of onions and time in minutes -/
structure ChoppingRate where
  onions : ℕ
  minutes : ℕ

/-- Calculates the number of onions chopped in a given time based on a chopping rate -/
def chop_onions (rate : ChoppingRate) (time : ℕ) : ℕ :=
  (rate.onions * time) / rate.minutes

theorem onion_chopping_difference :
  let brittney_rate : ChoppingRate := ⟨15, 5⟩
  let carl_rate : ChoppingRate := ⟨20, 5⟩
  let time : ℕ := 30
  chop_onions carl_rate time - chop_onions brittney_rate time = 30 := by
  sorry

end NUMINAMATH_CALUDE_onion_chopping_difference_l3586_358603


namespace NUMINAMATH_CALUDE_actual_speed_is_30_l3586_358661

/-- Given:
  1. Increasing speed by 10 mph reduces time by 1/4
  2. Increasing speed by 20 mph reduces time by an additional 1/3
  Prove that the actual average speed is 30 mph
-/
theorem actual_speed_is_30 (v : ℝ) (t : ℝ) (d : ℝ) :
  (d = v * t) →
  (d / (v + 10) = 3 / 4 * t) →
  (d / (v + 20) = 1 / 2 * t) →
  v = 30 := by
  sorry

end NUMINAMATH_CALUDE_actual_speed_is_30_l3586_358661


namespace NUMINAMATH_CALUDE_inequality_proof_l3586_358612

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2*x + y + z)^2 * (2*x^2 + (y + z)^2) +
  (2*y + z + x)^2 * (2*y^2 + (z + x)^2) +
  (2*z + x + y)^2 * (2*z^2 + (x + y)^2) ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3586_358612


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3586_358673

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a → a 5 = 21 → a 4 + a 5 + a 6 = 63 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3586_358673


namespace NUMINAMATH_CALUDE_weekly_diaper_sales_revenue_l3586_358698

/-- Represents the weekly diaper sales revenue calculation --/
theorem weekly_diaper_sales_revenue :
  let boxes_per_week : ℕ := 30
  let packs_per_box : ℕ := 40
  let diapers_per_pack : ℕ := 160
  let price_per_diaper : ℚ := 4
  let bundle_discount : ℚ := 0.05
  let special_discount : ℚ := 0.05
  let tax_rate : ℚ := 0.10

  let total_diapers : ℕ := boxes_per_week * packs_per_box * diapers_per_pack
  let base_revenue : ℚ := total_diapers * price_per_diaper
  let after_bundle_discount : ℚ := base_revenue * (1 - bundle_discount)
  let after_special_discount : ℚ := after_bundle_discount * (1 - special_discount)
  let final_revenue : ℚ := after_special_discount * (1 + tax_rate)

  final_revenue = 762432 :=
by sorry


end NUMINAMATH_CALUDE_weekly_diaper_sales_revenue_l3586_358698


namespace NUMINAMATH_CALUDE_perimeter_difference_l3586_358687

/-- Calculates the perimeter of a rectangle given its width and height. -/
def rectangle_perimeter (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width + height)

/-- Calculates the perimeter of a cross-shaped figure composed of 5 unit squares. -/
def cross_perimeter : ℕ := 8

/-- Theorem stating the difference between the perimeters of a 4x3 rectangle and a cross-shaped figure. -/
theorem perimeter_difference : 
  (rectangle_perimeter 4 3) - cross_perimeter = 6 := by sorry

end NUMINAMATH_CALUDE_perimeter_difference_l3586_358687


namespace NUMINAMATH_CALUDE_sqrt_13_minus_3_bounds_l3586_358633

theorem sqrt_13_minus_3_bounds : 0 < Real.sqrt 13 - 3 ∧ Real.sqrt 13 - 3 < 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_13_minus_3_bounds_l3586_358633


namespace NUMINAMATH_CALUDE_max_lg_product_l3586_358669

open Real

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := log x / log 10

-- State the theorem
theorem max_lg_product (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) 
  (h : lg x ^ 2 + lg y ^ 2 = lg (10 * x ^ 2) + lg (10 * y ^ 2)) :
  ∃ (max : ℝ), max = 2 + 2 * sqrt 2 ∧ lg (x * y) ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_lg_product_l3586_358669


namespace NUMINAMATH_CALUDE_train_speed_with_36_coaches_l3586_358625

/-- Represents the speed of a train given the number of coaches attached. -/
noncomputable def train_speed (initial_speed : ℝ) (k : ℝ) (coaches : ℝ) : ℝ :=
  initial_speed - k * Real.sqrt coaches

/-- The theorem states that given the initial conditions, 
    the speed of the train with 36 coaches is 48 kmph. -/
theorem train_speed_with_36_coaches 
  (initial_speed : ℝ) 
  (k : ℝ) 
  (speed_reduction : ∀ (c : ℝ), train_speed initial_speed k c = initial_speed - k * Real.sqrt c) 
  (h1 : initial_speed = 60) 
  (h2 : train_speed initial_speed k 36 = 48) :
  train_speed initial_speed k 36 = 48 := by
  sorry

#check train_speed_with_36_coaches

end NUMINAMATH_CALUDE_train_speed_with_36_coaches_l3586_358625


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3586_358655

theorem quadratic_equation_solutions
  (a b c : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a - b + c = 0)
  (h3 : a ≠ 0) :
  ∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3586_358655


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_condition_l3586_358688

theorem quadratic_no_real_roots_condition (m x : ℝ) : 
  (∀ x, x^2 - 2*x + m ≠ 0) → m ≥ 0 ∧ 
  ∃ m₀ ≥ 0, ∃ x₀, x₀^2 - 2*x₀ + m₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_condition_l3586_358688


namespace NUMINAMATH_CALUDE_coin_flip_probability_is_two_elevenths_l3586_358657

/-- The probability of getting 4 consecutive heads before 3 consecutive tails
    when repeatedly flipping a fair coin -/
def coin_flip_probability : ℚ :=
  2/11

/-- Theorem stating that the probability of getting 4 consecutive heads
    before 3 consecutive tails when repeatedly flipping a fair coin is 2/11 -/
theorem coin_flip_probability_is_two_elevenths :
  coin_flip_probability = 2/11 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_is_two_elevenths_l3586_358657


namespace NUMINAMATH_CALUDE_function_value_at_a_plus_one_l3586_358671

theorem function_value_at_a_plus_one (a : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + 1
  f (a + 1) = a^2 + 2*a + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_a_plus_one_l3586_358671


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3586_358666

theorem rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 10) 
  (h2 : b * c = 15) 
  (h3 : c * a = 18) : 
  a * b * c = 30 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3586_358666


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l3586_358621

theorem unique_prime_with_prime_sums : ∃! p : ℕ, 
  Nat.Prime p ∧ Nat.Prime (p + 28) ∧ Nat.Prime (p + 56) ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l3586_358621


namespace NUMINAMATH_CALUDE_banana_bread_pieces_l3586_358680

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of banana bread -/
structure BananaBreadPan where
  dimensions : Dimensions

/-- Represents a piece of banana bread -/
structure BananaBreadPiece where
  dimensions : Dimensions

/-- Calculates the number of pieces that can be cut from a pan -/
def num_pieces (pan : BananaBreadPan) (piece : BananaBreadPiece) : ℕ :=
  (area pan.dimensions) / (area piece.dimensions)

theorem banana_bread_pieces : 
  let pan := BananaBreadPan.mk (Dimensions.mk 24 20)
  let piece := BananaBreadPiece.mk (Dimensions.mk 3 4)
  num_pieces pan piece = 40 := by
  sorry

end NUMINAMATH_CALUDE_banana_bread_pieces_l3586_358680


namespace NUMINAMATH_CALUDE_triangle_side_count_l3586_358638

theorem triangle_side_count : ∃! n : ℕ, 
  n = (Finset.filter (fun x : ℕ => 
    x > 0 ∧ x + 5 > 8 ∧ 8 + 5 > x
  ) (Finset.range 100)).card ∧ n = 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_count_l3586_358638


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3586_358601

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 14 = 0) → 
  (3 * q^2 - 5 * q - 14 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3586_358601


namespace NUMINAMATH_CALUDE_negative_correlation_from_negative_coefficient_given_equation_negative_correlation_l3586_358634

/-- Represents a linear regression equation -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- Defines negative correlation between x and y -/
def negatively_correlated (eq : LinearRegression) : Prop :=
  eq.b < 0

/-- Theorem: If the coefficient of x in a linear regression equation is negative,
    then x and y are negatively correlated -/
theorem negative_correlation_from_negative_coefficient (eq : LinearRegression) :
  eq.b < 0 → negatively_correlated eq :=
by
  sorry

/-- The given empirical regression equation -/
def given_equation : LinearRegression :=
  { a := 2, b := -1 }

/-- Theorem: The given equation represents a negative correlation between x and y -/
theorem given_equation_negative_correlation :
  negatively_correlated given_equation :=
by
  sorry

end NUMINAMATH_CALUDE_negative_correlation_from_negative_coefficient_given_equation_negative_correlation_l3586_358634


namespace NUMINAMATH_CALUDE_circle_radius_c_value_l3586_358676

theorem circle_radius_c_value :
  ∀ c : ℝ,
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ↔ (x + 4)^2 + (y + 1)^2 = 25) →
  c = -8 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_c_value_l3586_358676


namespace NUMINAMATH_CALUDE_golden_ratio_trigonometry_l3586_358683

theorem golden_ratio_trigonometry (m n : ℝ) : 
  m = 2 * Real.sin (18 * π / 180) →
  m^2 + n = 4 →
  (m * Real.sqrt n) / (2 * (Real.cos (27 * π / 180))^2 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_trigonometry_l3586_358683


namespace NUMINAMATH_CALUDE_equation_2x_minus_y_eq_2_is_linear_l3586_358664

/-- A linear equation in two variables is of the form ax + by + c = 0, where a and b are not both zero -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y = a * x + b * y + c

/-- The function representing the equation 2x - y = 2 -/
def f (x y : ℝ) : ℝ := 2 * x - y - 2

theorem equation_2x_minus_y_eq_2_is_linear : is_linear_equation f :=
sorry

end NUMINAMATH_CALUDE_equation_2x_minus_y_eq_2_is_linear_l3586_358664


namespace NUMINAMATH_CALUDE_stacy_bought_two_packs_l3586_358620

/-- The number of sheets per pack of printer paper -/
def sheets_per_pack : ℕ := 240

/-- The number of sheets used per day -/
def sheets_per_day : ℕ := 80

/-- The number of days the paper lasts -/
def days_lasted : ℕ := 6

/-- The number of packs of printer paper Stacy bought -/
def packs_bought : ℕ := (sheets_per_day * days_lasted) / sheets_per_pack

theorem stacy_bought_two_packs : packs_bought = 2 := by
  sorry

end NUMINAMATH_CALUDE_stacy_bought_two_packs_l3586_358620


namespace NUMINAMATH_CALUDE_group_b_inspected_products_group_b_inspectors_l3586_358610

-- Define the number of workshops
def num_workshops : ℕ := 9

-- Define the number of inspectors in Group A
def group_a_inspectors : ℕ := 8

-- Define the initial number of finished products per workshop
variable (a : ℕ)

-- Define the daily production of finished products per workshop
variable (b : ℕ)

-- Define the number of days Group A inspects workshops 1 and 2
def days_group_a_1_2 : ℕ := 2

-- Define the number of days Group A inspects workshops 3 and 4
def days_group_a_3_4 : ℕ := 3

-- Define the total number of days for inspection
def total_inspection_days : ℕ := 5

-- Define the number of workshops inspected by Group B
def workshops_group_b : ℕ := 5

-- Theorem for the total number of finished products inspected by Group B
theorem group_b_inspected_products (a b : ℕ) :
  workshops_group_b * a + workshops_group_b * total_inspection_days * b = 5 * a + 25 * b :=
sorry

-- Theorem for the number of inspectors in Group B
theorem group_b_inspectors (a b : ℕ) (h : a = 4 * b) :
  (workshops_group_b * a + workshops_group_b * total_inspection_days * b) /
  ((3 / 4 : ℚ) * b * total_inspection_days) = 12 :=
sorry

end NUMINAMATH_CALUDE_group_b_inspected_products_group_b_inspectors_l3586_358610


namespace NUMINAMATH_CALUDE_f_extrema_l3586_358663

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (cos (3*x/2), sin (3*x/2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (cos (x/2), -sin (x/2))

noncomputable def f (x : ℝ) : ℝ := 
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - 
  Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2

theorem f_extrema :
  ∀ x ∈ Set.Icc (-π/3) (π/4),
    (∀ y ∈ Set.Icc (-π/3) (π/4), f y ≤ -1) ∧
    (∃ y ∈ Set.Icc (-π/3) (π/4), f y = -1) ∧
    (∀ y ∈ Set.Icc (-π/3) (π/4), f y ≥ -Real.sqrt 2) ∧
    (∃ y ∈ Set.Icc (-π/3) (π/4), f y = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l3586_358663


namespace NUMINAMATH_CALUDE_new_person_weight_l3586_358660

theorem new_person_weight (n : ℕ) (avg_increase weight_replaced : ℝ) :
  n = 7 →
  avg_increase = 6.2 →
  weight_replaced = 76 →
  n * avg_increase + weight_replaced = 119.4 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3586_358660


namespace NUMINAMATH_CALUDE_total_coins_last_month_l3586_358678

/-- The number of coins Mathilde had at the start of this month -/
def mathilde_this_month : ℕ := 100

/-- The number of coins Salah had at the start of this month -/
def salah_this_month : ℕ := 100

/-- The percentage increase in Mathilde's coins from last month to this month -/
def mathilde_increase_percent : ℚ := 25/100

/-- The percentage decrease in Salah's coins from last month to this month -/
def salah_decrease_percent : ℚ := 20/100

/-- The number of coins Mathilde had at the start of last month -/
def mathilde_last_month : ℚ := mathilde_this_month / (1 + mathilde_increase_percent)

/-- The number of coins Salah had at the start of last month -/
def salah_last_month : ℚ := salah_this_month / (1 - salah_decrease_percent)

theorem total_coins_last_month :
  mathilde_last_month + salah_last_month = 205 := by sorry

end NUMINAMATH_CALUDE_total_coins_last_month_l3586_358678


namespace NUMINAMATH_CALUDE_total_cost_with_tax_l3586_358602

def nike_price : ℝ := 150
def boots_price : ℝ := 120
def tax_rate : ℝ := 0.1

theorem total_cost_with_tax :
  let pre_tax_total := nike_price + boots_price
  let tax_amount := pre_tax_total * tax_rate
  let total_with_tax := pre_tax_total + tax_amount
  total_with_tax = 297 := by sorry

end NUMINAMATH_CALUDE_total_cost_with_tax_l3586_358602


namespace NUMINAMATH_CALUDE_cos_30_minus_cos_60_l3586_358642

theorem cos_30_minus_cos_60 :
  Real.cos (30 * π / 180) - Real.cos (60 * π / 180) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_minus_cos_60_l3586_358642


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3586_358694

theorem min_value_expression (x y : ℝ) : x^2 + y^2 - 8*x - 6*y + 30 ≥ 5 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, x^2 + y^2 - 8*x - 6*y + 30 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3586_358694


namespace NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l3586_358640

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Creates a triangle from a line and coordinate axes -/
def triangleFromLine (l : Line) : Triangle :=
  sorry

/-- Calculates the sum of altitudes of a triangle -/
def sumOfAltitudes (t : Triangle) : ℝ :=
  sorry

/-- The main theorem -/
theorem sum_of_altitudes_for_specific_line :
  let l : Line := { a := 15, b := 8, c := 120 }
  let t : Triangle := triangleFromLine l
  sumOfAltitudes t = 391 / 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l3586_358640


namespace NUMINAMATH_CALUDE_min_value_trigonometric_function_solution_set_quadratic_inequality_l3586_358650

-- Problem 1
theorem min_value_trigonometric_function (x : ℝ) (hx : 0 < x ∧ x < π / 2) :
  (1 / Real.sin x ^ 2) + (4 / Real.cos x ^ 2) ≥ 9 :=
sorry

-- Problem 2
theorem solution_set_quadratic_inequality (a b c α β : ℝ) 
  (h_sol : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β)
  (h_pos : 0 < α ∧ α < β) :
  ∀ x, c * x^2 + b * x + a < 0 ↔ x < 1/β ∨ x > 1/α :=
sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_function_solution_set_quadratic_inequality_l3586_358650


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_l3586_358681

/-- A regular tetrahedron with one vertex on the axis of a cylinder and the other three vertices on the lateral surface of the cylinder. -/
structure TetrahedronInCylinder where
  R : ℝ  -- Radius of the cylinder's base
  edge_length : ℝ  -- Edge length of the tetrahedron

/-- The edge length of the tetrahedron is either R√3 or (R√11)/3. -/
theorem tetrahedron_edge_length (t : TetrahedronInCylinder) :
  t.edge_length = t.R * Real.sqrt 3 ∨ t.edge_length = t.R * Real.sqrt 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_l3586_358681


namespace NUMINAMATH_CALUDE_divisors_of_8820_multiple_of_3_and_5_l3586_358656

def number_of_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_8820_multiple_of_3_and_5 : 
  number_of_divisors 8820 = 18 := by sorry

end NUMINAMATH_CALUDE_divisors_of_8820_multiple_of_3_and_5_l3586_358656


namespace NUMINAMATH_CALUDE_min_omega_two_max_sine_l3586_358635

theorem min_omega_two_max_sine (ω : Real) : ω > 0 → (∃ x₁ x₂ : Real, 
  0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧ 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → Real.sin (ω * x) ≤ Real.sin (ω * x₁)) ∧
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → Real.sin (ω * x) ≤ Real.sin (ω * x₂))) → 
  ω ≥ 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_min_omega_two_max_sine_l3586_358635


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3586_358670

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), 
  s > 0 → 
  6 * s^2 = 150 → 
  s^3 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3586_358670


namespace NUMINAMATH_CALUDE_zeros_sum_greater_than_twice_intersection_l3586_358645

noncomputable section

variables (a : ℝ) (x₁ x₂ x₀ : ℝ)

def f (x : ℝ) : ℝ := a * Real.log x - (1/2) * x^2

theorem zeros_sum_greater_than_twice_intersection
  (h₁ : a > Real.exp 1)
  (h₂ : f a x₁ = 0)
  (h₃ : f a x₂ = 0)
  (h₄ : x₁ ≠ x₂)
  (h₅ : x₀ = (x₁ + x₂) / ((a / (x₁ * x₂)) + 1)) :
  x₁ + x₂ > 2 * x₀ := by
  sorry

end NUMINAMATH_CALUDE_zeros_sum_greater_than_twice_intersection_l3586_358645


namespace NUMINAMATH_CALUDE_fraction_simplification_l3586_358667

theorem fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (2 / y) / (3 / x^2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3586_358667


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l3586_358686

/-- Given a line containing points (2, 9), (10, m), and (25, 4), prove that m = 167/23 -/
theorem collinear_points_m_value : 
  ∀ m : ℚ, 
  (∃ (line : Set (ℚ × ℚ)), 
    (2, 9) ∈ line ∧ 
    (10, m) ∈ line ∧ 
    (25, 4) ∈ line ∧ 
    (∀ (x y z : ℚ × ℚ), x ∈ line → y ∈ line → z ∈ line → 
      (z.2 - y.2) * (y.1 - x.1) = (y.2 - x.2) * (z.1 - y.1))) →
  m = 167 / 23 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l3586_358686


namespace NUMINAMATH_CALUDE_remainder_theorem_l3586_358618

theorem remainder_theorem (x : ℤ) (h : (x + 2) % 45 = 7) : 
  ((x + 2) % 20 = 7) ∧ (x % 19 = 5) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3586_358618


namespace NUMINAMATH_CALUDE_grocer_sales_problem_l3586_358627

/-- Calculates the first month's sale given sales for the next 4 months and desired average -/
def first_month_sale (month2 month3 month4 month5 desired_average : ℕ) : ℕ :=
  5 * desired_average - (month2 + month3 + month4 + month5)

/-- Proves that the first month's sale is 6790 given the problem conditions -/
theorem grocer_sales_problem : 
  first_month_sale 5660 6200 6350 6500 6300 = 6790 := by
  sorry

#eval first_month_sale 5660 6200 6350 6500 6300

end NUMINAMATH_CALUDE_grocer_sales_problem_l3586_358627
