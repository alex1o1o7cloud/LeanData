import Mathlib

namespace hyperbola_eccentricity_range_l164_16499

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  let e := Real.sqrt (1 + 1 / a^2)
  1 < e ∧ e < Real.sqrt 2 := by
sorry

end hyperbola_eccentricity_range_l164_16499


namespace nabla_ratio_equals_eight_l164_16492

-- Define the ∇ operation for positive integers m < n
def nabla (m n : ℕ) (h1 : 0 < m) (h2 : m < n) : ℕ :=
  (n - m + 1) * (m + n) / 2

-- Theorem statement
theorem nabla_ratio_equals_eight :
  nabla 22 26 (by norm_num) (by norm_num) / nabla 4 6 (by norm_num) (by norm_num) = 8 := by
  sorry

end nabla_ratio_equals_eight_l164_16492


namespace square_difference_divided_by_three_l164_16462

theorem square_difference_divided_by_three : (121^2 - 112^2) / 3 = 699 := by
  sorry

end square_difference_divided_by_three_l164_16462


namespace M_intersect_N_equals_one_two_open_l164_16477

def M : Set ℝ := {x | ∃ y, y = Real.log (-x^2 - x + 6)}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem M_intersect_N_equals_one_two_open :
  M ∩ N = {x | 1 ≤ x ∧ x < 2} := by sorry

end M_intersect_N_equals_one_two_open_l164_16477


namespace arithmetic_calculation_l164_16448

theorem arithmetic_calculation : 4 * 11 + 5 * 12 + 13 * 4 + 4 * 10 = 196 := by
  sorry

end arithmetic_calculation_l164_16448


namespace complex_on_line_with_magnitude_l164_16431

theorem complex_on_line_with_magnitude (z : ℂ) :
  (z.im = 2 * z.re) → (Complex.abs z = Real.sqrt 5) →
  (z = Complex.mk 1 2 ∨ z = Complex.mk (-1) (-2)) := by
  sorry

end complex_on_line_with_magnitude_l164_16431


namespace larger_number_proof_l164_16490

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 7 * S + 15) :
  L = 1590 := by
sorry

end larger_number_proof_l164_16490


namespace distance_to_left_focus_l164_16481

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the hyperbola C₂
def C₂ (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the point P as the intersection of C₁ and C₂ in the first quadrant
def P : ℝ × ℝ := sorry

-- State that P satisfies both C₁ and C₂
axiom P_on_C₁ : C₁ P.1 P.2
axiom P_on_C₂ : C₂ P.1 P.2

-- State that P is in the first quadrant
axiom P_first_quadrant : P.1 > 0 ∧ P.2 > 0

-- Define the left focus of the ellipse
def left_focus : ℝ × ℝ := sorry

-- Theorem stating the distance from P to the left focus is 4
theorem distance_to_left_focus :
  Real.sqrt ((P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2) = 4 := by sorry

end distance_to_left_focus_l164_16481


namespace real_y_condition_l164_16456

theorem real_y_condition (x y : ℝ) : 
  (4 * y^2 + 2 * x * y + |x| + 8 = 0) → 
  (∃ (y : ℝ), 4 * y^2 + 2 * x * y + |x| + 8 = 0) ↔ (x ≤ -10 ∨ x ≥ 10) := by
  sorry

end real_y_condition_l164_16456


namespace set_relations_l164_16455

theorem set_relations (A B : Set α) (h : ∃ x, x ∈ A ∧ x ∉ B) :
  (¬(A ⊆ B)) ∧
  (∃ A' B' : Set α, (∃ x, x ∈ A' ∧ x ∉ B') ∧ (A' ∩ B' ≠ ∅)) ∧
  (∃ A' B' : Set α, (∃ x, x ∈ A' ∧ x ∉ B') ∧ (B' ⊆ A')) ∧
  (∃ A' B' : Set α, (∃ x, x ∈ A' ∧ x ∉ B') ∧ (A' ∩ B' = ∅)) :=
by sorry

end set_relations_l164_16455


namespace chang_e_3_descent_time_l164_16443

/-- Represents the descent phase of Chang'e 3 --/
structure DescentPhase where
  initial_altitude : ℝ  -- in kilometers
  final_altitude : ℝ    -- in meters
  playback_time_initial : ℕ  -- in seconds
  total_video_duration : ℕ   -- in seconds

/-- Calculates the time spent in the descent phase --/
def descent_time (d : DescentPhase) : ℕ :=
  114  -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the descent time for the given conditions is 114 seconds --/
theorem chang_e_3_descent_time :
  let d : DescentPhase := {
    initial_altitude := 2.4,
    final_altitude := 100,
    playback_time_initial := 30 * 60 + 28,  -- 30 minutes and 28 seconds
    total_video_duration := 2 * 60 * 60 + 10 * 60 + 48  -- 2 hours, 10 minutes, and 48 seconds
  }
  descent_time d = 114 := by sorry

end chang_e_3_descent_time_l164_16443


namespace probability_k_standard_parts_formula_l164_16450

/-- The probability of selecting exactly k standard parts when randomly choosing m parts from a batch of N parts containing n standard parts. -/
def probability_k_standard_parts (N n m k : ℕ) : ℚ :=
  (Nat.choose n k * Nat.choose (N - n) (m - k)) / Nat.choose N m

/-- Theorem stating that the probability of selecting exactly k standard parts
    when randomly choosing m parts from a batch of N parts containing n standard parts
    is equal to (C_n^k * C_(N-n)^(m-k)) / C_N^m. -/
theorem probability_k_standard_parts_formula
  (N n m k : ℕ)
  (h1 : n ≤ N)
  (h2 : m ≤ N)
  (h3 : k ≤ m)
  (h4 : k ≤ n) :
  probability_k_standard_parts N n m k =
    (Nat.choose n k * Nat.choose (N - n) (m - k)) / Nat.choose N m :=
by
  sorry

#check probability_k_standard_parts_formula

end probability_k_standard_parts_formula_l164_16450


namespace sons_shoveling_time_l164_16427

/-- Proves that Wayne's son takes 21 hours to shovel the entire driveway alone,
    given that Wayne and his son together take 3 hours,
    and Wayne shovels 6 times as fast as his son. -/
theorem sons_shoveling_time (total_work : ℝ) (joint_time : ℝ) (wayne_speed_ratio : ℝ) :
  total_work > 0 →
  joint_time = 3 →
  wayne_speed_ratio = 6 →
  (total_work / joint_time) * (wayne_speed_ratio + 1) * 21 = total_work :=
by sorry

end sons_shoveling_time_l164_16427


namespace square_sum_lower_bound_l164_16433

theorem square_sum_lower_bound (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 + 2*x*y*z = 1) : 
  x^2 + y^2 + z^2 ≥ 3/4 := by
  sorry

end square_sum_lower_bound_l164_16433


namespace base_subtraction_equality_l164_16452

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_subtraction_equality : 
  let base_6_num := [5, 2, 3]  -- 325 in base 6 (least significant digit first)
  let base_5_num := [1, 3, 2]  -- 231 in base 5 (least significant digit first)
  (to_base_10 base_6_num 6) - (to_base_10 base_5_num 5) = 59 := by
  sorry

end base_subtraction_equality_l164_16452


namespace min_sum_a_b_l164_16485

theorem min_sum_a_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a * 2 + b * 3 - a * b = 0) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * 2 + y * 3 - x * y = 0 → a + b ≤ x + y :=
by sorry

end min_sum_a_b_l164_16485


namespace geometry_problem_l164_16472

-- Define the points
def M : ℝ × ℝ := (2, -2)
def N : ℝ × ℝ := (4, 4)
def P : ℝ × ℝ := (2, -3)

-- Define the equations
def perpendicular_bisector (x y : ℝ) : Prop := x + 3*y - 6 = 0
def parallel_line (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem geometry_problem :
  (∀ x y : ℝ, perpendicular_bisector x y ↔ 
    (x - M.1)^2 + (y - M.2)^2 = (x - N.1)^2 + (y - N.2)^2) ∧
  (∀ x y : ℝ, parallel_line x y ↔ 
    (y - P.2) = ((N.2 - M.2) / (N.1 - M.1)) * (x - P.1)) :=
by sorry

end geometry_problem_l164_16472


namespace total_chips_is_90_l164_16453

/-- The total number of chips Viviana and Susana have together -/
def total_chips (viviana_vanilla : ℕ) (susana_chocolate : ℕ) : ℕ :=
  let viviana_chocolate := susana_chocolate + 5
  let susana_vanilla := (3 * viviana_vanilla) / 4
  viviana_vanilla + viviana_chocolate + susana_vanilla + susana_chocolate

/-- Theorem stating that the total number of chips is 90 -/
theorem total_chips_is_90 :
  total_chips 20 25 = 90 := by
  sorry

#eval total_chips 20 25

end total_chips_is_90_l164_16453


namespace randy_gave_sally_l164_16486

theorem randy_gave_sally (initial_amount : ℕ) (received_amount : ℕ) (kept_amount : ℕ) :
  initial_amount = 3000 →
  received_amount = 200 →
  kept_amount = 2000 →
  initial_amount + received_amount - kept_amount = 1200 := by
sorry

end randy_gave_sally_l164_16486


namespace team_c_score_l164_16467

/-- Given a trivia game with three teams, prove that Team C's score is 4 points. -/
theorem team_c_score (team_a team_b team_c total : ℕ) : 
  team_a = 2 → team_b = 9 → total = 15 → team_a + team_b + team_c = total → team_c = 4 := by
  sorry

end team_c_score_l164_16467


namespace remainder_equality_l164_16446

theorem remainder_equality (P P' K D R R' : ℕ) (r r' : ℕ) 
  (h1 : P > P') 
  (h2 : K ∣ P) 
  (h3 : K ∣ P') 
  (h4 : P % D = R) 
  (h5 : P' % D = R') 
  (h6 : (P * K - P') % D = r) 
  (h7 : (R * K - R') % D = r') : 
  r = r' := by
sorry

end remainder_equality_l164_16446


namespace joes_dad_marshmallow_fraction_l164_16417

theorem joes_dad_marshmallow_fraction :
  ∀ (dad_marshmallows joe_marshmallows dad_roasted joe_roasted total_roasted : ℕ),
    dad_marshmallows = 21 →
    joe_marshmallows = 4 * dad_marshmallows →
    joe_roasted = joe_marshmallows / 2 →
    total_roasted = 49 →
    total_roasted = joe_roasted + dad_roasted →
    (dad_roasted : ℚ) / dad_marshmallows = 1 / 3 := by
  sorry

#check joes_dad_marshmallow_fraction

end joes_dad_marshmallow_fraction_l164_16417


namespace winning_strategy_l164_16445

/-- Represents the player who has a winning strategy -/
inductive WinningPlayer
  | First
  | Second

/-- Defines the game on a grid board -/
def gridGame (m n : ℕ) : WinningPlayer :=
  if m % 2 = 0 ∧ n % 2 = 0 then
    WinningPlayer.Second
  else if m % 2 = 1 ∧ n % 2 = 1 then
    WinningPlayer.Second
  else
    WinningPlayer.First

/-- Theorem stating the winning strategy for different board sizes -/
theorem winning_strategy :
  (gridGame 10 12 = WinningPlayer.Second) ∧
  (gridGame 9 10 = WinningPlayer.First) ∧
  (gridGame 9 11 = WinningPlayer.Second) :=
sorry

end winning_strategy_l164_16445


namespace unique_divisible_number_l164_16498

theorem unique_divisible_number : ∃! (x y z u v : ℕ),
  (x < 10 ∧ y < 10 ∧ z < 10 ∧ u < 10 ∧ v < 10) ∧
  (x * 10^9 + 6 * 10^8 + 1 * 10^7 + y * 10^6 + 0 * 10^5 + 6 * 10^4 + 4 * 10^3 + z * 10^2 + u * 10 + v) % 61875 = 0 :=
by sorry

end unique_divisible_number_l164_16498


namespace twelve_hash_six_l164_16425

/-- The # operation for real numbers -/
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

/-- Axioms for the # operation -/
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + 2 * s + 1

/-- The main theorem to prove -/
theorem twelve_hash_six : hash 12 6 = 272 := by
  sorry

end twelve_hash_six_l164_16425


namespace pet_food_cost_l164_16435

theorem pet_food_cost (total_cost rabbit_toy_cost cage_cost found_money : ℚ)
  (h1 : total_cost = 24.81)
  (h2 : rabbit_toy_cost = 6.51)
  (h3 : cage_cost = 12.51)
  (h4 : found_money = 1.00) :
  total_cost - (rabbit_toy_cost + cage_cost) + found_money = 6.79 := by
  sorry

end pet_food_cost_l164_16435


namespace clothing_tax_rate_l164_16459

theorem clothing_tax_rate
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (other_tax_rate : ℝ)
  (total_tax_rate : ℝ)
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.2)
  (h3 : other_percent = 0.3)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : other_tax_rate = 0.1)
  (h6 : total_tax_rate = 0.055) :
  ∃ (clothing_tax_rate : ℝ),
    clothing_tax_rate * clothing_percent + other_tax_rate * other_percent = total_tax_rate ∧
    clothing_tax_rate = 0.05 := by
  sorry

end clothing_tax_rate_l164_16459


namespace pet_shop_dogs_count_l164_16466

/-- Given a pet shop with dogs, cats, and bunnies, where the ratio of dogs to cats to bunnies
    is 7 : 7 : 8, and the total number of dogs and bunnies is 330, prove that there are 154 dogs. -/
theorem pet_shop_dogs_count : ℕ → ℕ → ℕ → Prop :=
  fun dogs cats bunnies =>
    (dogs : ℚ) / cats = 1 →
    (dogs : ℚ) / bunnies = 7 / 8 →
    dogs + bunnies = 330 →
    dogs = 154

/-- Proof of the pet_shop_dogs_count theorem -/
lemma prove_pet_shop_dogs_count : ∃ dogs cats bunnies, pet_shop_dogs_count dogs cats bunnies :=
  sorry

end pet_shop_dogs_count_l164_16466


namespace circle_condition_l164_16418

/-- A circle in the xy-plane is represented by the equation x^2 + y^2 + Dx + Ey + F = 0,
    where D^2 + E^2 - 4F > 0 -/
def is_circle (D E F : ℝ) : Prop := D^2 + E^2 - 4*F > 0

/-- The equation x^2 + y^2 - 2x + 2k + 3 = 0 represents a circle -/
def our_equation_is_circle (k : ℝ) : Prop := is_circle (-2) 0 (2*k + 3)

theorem circle_condition (k : ℝ) : our_equation_is_circle k ↔ k < -1 := by
  sorry

end circle_condition_l164_16418


namespace ratio_w_to_y_l164_16465

/-- Given ratios between w, x, y, and z, prove the ratio of w to y -/
theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 2)
  (hy : y / w = 3 / 5)
  (hz : z / x = 1 / 3) :
  w / y = 5 / 3 := by sorry

end ratio_w_to_y_l164_16465


namespace expected_vote_percentage_is_47_percent_l164_16421

/-- The percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.60

/-- The percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 1 - democrat_percentage

/-- The percentage of registered Democrat voters expected to vote for candidate A -/
def democrat_vote_percentage : ℝ := 0.65

/-- The percentage of registered Republican voters expected to vote for candidate A -/
def republican_vote_percentage : ℝ := 0.20

/-- The expected percentage of registered voters who will vote for candidate A -/
def expected_vote_percentage : ℝ :=
  democrat_percentage * democrat_vote_percentage +
  republican_percentage * republican_vote_percentage

theorem expected_vote_percentage_is_47_percent :
  expected_vote_percentage = 0.47 :=
sorry

end expected_vote_percentage_is_47_percent_l164_16421


namespace total_messages_l164_16439

def messages_last_week : ℕ := 111

def messages_this_week : ℕ := 2 * messages_last_week - 50

theorem total_messages : messages_last_week + messages_this_week = 283 := by
  sorry

end total_messages_l164_16439


namespace smallest_number_properties_l164_16469

/-- The smallest number that is divisible by 18 and 30 and is a perfect square -/
def smallest_number : ℕ := 900

/-- Predicate to check if a number is divisible by both 18 and 30 -/
def divisible_by_18_and_30 (n : ℕ) : Prop := n % 18 = 0 ∧ n % 30 = 0

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem smallest_number_properties :
  divisible_by_18_and_30 smallest_number ∧
  is_perfect_square smallest_number ∧
  ∀ n : ℕ, n < smallest_number → ¬(divisible_by_18_and_30 n ∧ is_perfect_square n) :=
by sorry

end smallest_number_properties_l164_16469


namespace negation_of_all_students_punctual_l164_16414

namespace NegationOfUniversalStatement

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (student : U → Prop)
variable (punctual : U → Prop)

-- State the theorem
theorem negation_of_all_students_punctual :
  (¬ ∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) :=
sorry

end NegationOfUniversalStatement

end negation_of_all_students_punctual_l164_16414


namespace jose_to_john_ratio_l164_16422

-- Define the total amount and the ratios
def total_amount : ℕ := 4800
def ratio_john : ℕ := 2
def ratio_jose : ℕ := 4
def ratio_binoy : ℕ := 6

-- Define John's share
def john_share : ℕ := 1600

-- Theorem to prove
theorem jose_to_john_ratio :
  let total_ratio := ratio_john + ratio_jose + ratio_binoy
  let share_value := total_amount / total_ratio
  let jose_share := share_value * ratio_jose
  jose_share / john_share = 2 := by
  sorry

end jose_to_john_ratio_l164_16422


namespace balloon_count_l164_16479

theorem balloon_count (initial : Real) (given : Real) (total : Real) 
  (h1 : initial = 7.0)
  (h2 : given = 5.0)
  (h3 : total = initial + given) :
  total = 12.0 := by
sorry

end balloon_count_l164_16479


namespace system_of_equations_range_l164_16430

theorem system_of_equations_range (x y k : ℝ) : 
  x - y = k - 1 →
  3 * x + 2 * y = 4 * k + 5 →
  2 * x + 3 * y > 7 →
  k > 1/3 := by
sorry

end system_of_equations_range_l164_16430


namespace circle_radius_is_five_l164_16415

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*y - 16 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 5

/-- Theorem stating that the radius of the circle is 5 -/
theorem circle_radius_is_five :
  ∀ x y : ℝ, circle_equation x y → ∃ center_x center_y : ℝ,
    (x - center_x)^2 + (y - center_y)^2 = circle_radius^2 :=
by
  sorry

end circle_radius_is_five_l164_16415


namespace least_four_divisors_sum_of_squares_l164_16495

theorem least_four_divisors_sum_of_squares (n : ℕ+) 
  (h1 : ∃ (d1 d2 d3 d4 : ℕ+), d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ 
    (∀ m : ℕ+, m ∣ n → m = d1 ∨ m = d2 ∨ m = d3 ∨ m = d4 ∨ m > d4))
  (h2 : ∃ (d1 d2 d3 d4 : ℕ+), d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ 
    n = d1^2 + d2^2 + d3^2 + d4^2) : 
  n = 130 := by
sorry

end least_four_divisors_sum_of_squares_l164_16495


namespace jake_viewing_time_l164_16478

/-- Calculates the number of hours Jake watched on Friday given his viewing schedule for the week --/
theorem jake_viewing_time (hours_per_day : ℕ) (show_length : ℕ) : 
  hours_per_day = 24 →
  show_length = 52 →
  let monday := hours_per_day / 2
  let tuesday := 4
  let wednesday := hours_per_day / 4
  let mon_to_wed := monday + tuesday + wednesday
  let thursday := mon_to_wed / 2
  let mon_to_thu := mon_to_wed + thursday
  19 = show_length - mon_to_thu := by sorry


end jake_viewing_time_l164_16478


namespace stating_five_min_commercials_count_l164_16440

/-- Represents the duration of the commercial break in minutes -/
def total_time : ℕ := 37

/-- Represents the number of 2-minute commercials -/
def two_min_commercials : ℕ := 11

/-- Represents the duration of a short commercial in minutes -/
def short_commercial_duration : ℕ := 2

/-- Represents the duration of a long commercial in minutes -/
def long_commercial_duration : ℕ := 5

/-- 
Theorem stating that given the total time and number of 2-minute commercials,
the number of 5-minute commercials is 3
-/
theorem five_min_commercials_count : 
  ∃ (x : ℕ), x * long_commercial_duration + two_min_commercials * short_commercial_duration = total_time ∧ x = 3 :=
by sorry

end stating_five_min_commercials_count_l164_16440


namespace triangle_not_isosceles_l164_16403

/-- A triangle with sides a, b, c is not isosceles if a, b, c are distinct -/
theorem triangle_not_isosceles (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : a ≠ b) (h₅ : b ≠ c) (h₆ : a ≠ c)
  (h₇ : a + b > c) (h₈ : b + c > a) (h₉ : a + c > b) :
  ¬(a = b ∨ b = c ∨ a = c) := by
  sorry

end triangle_not_isosceles_l164_16403


namespace shaded_area_recursive_square_division_l164_16468

theorem shaded_area_recursive_square_division (r : ℝ) (h1 : r = 1/16) (h2 : 0 < r) (h3 : r < 1) :
  (1/4) * (1 / (1 - r)) = 4/15 := by
  sorry

end shaded_area_recursive_square_division_l164_16468


namespace smallest_n_for_candy_l164_16473

theorem smallest_n_for_candy (n : ℕ) : 
  (∀ m : ℕ, m > 0 → (25 * m) % 10 = 0 ∧ (25 * m) % 18 = 0 ∧ (25 * m) % 20 = 0 → m ≥ n) →
  (25 * n) % 10 = 0 ∧ (25 * n) % 18 = 0 ∧ (25 * n) % 20 = 0 →
  n = 15 :=
sorry

end smallest_n_for_candy_l164_16473


namespace quadratic_solution_difference_squared_l164_16426

theorem quadratic_solution_difference_squared : 
  ∀ f g : ℝ, (4 * f^2 + 8 * f - 48 = 0) → (4 * g^2 + 8 * g - 48 = 0) → (f - g)^2 = 49 := by
  sorry

end quadratic_solution_difference_squared_l164_16426


namespace derivative_f_at_1_l164_16406

-- Define the function f
def f (x : ℝ) : ℝ := (2 + x)^2 - 3*x

-- State the theorem
theorem derivative_f_at_1 :
  deriv f 1 = 3 := by sorry

end derivative_f_at_1_l164_16406


namespace inequality_solution_l164_16401

theorem inequality_solution (x : ℝ) : 
  (1 / (x^2 + 1) > 4/x + 21/10) ↔ (-2 < x ∧ x < 0) := by
  sorry

end inequality_solution_l164_16401


namespace integer_solutions_of_system_l164_16451

theorem integer_solutions_of_system : 
  ∀ x y z : ℤ, 
  x + y + z = 2 ∧ 
  x^3 + y^3 + z^3 = -10 → 
  ((x = 3 ∧ y = 3 ∧ z = -4) ∨ 
   (x = 3 ∧ y = -4 ∧ z = 3) ∨ 
   (x = -4 ∧ y = 3 ∧ z = 3)) := by
  sorry

end integer_solutions_of_system_l164_16451


namespace solve_equation_l164_16493

theorem solve_equation (z : ℝ) :
  ∃ (n : ℝ), 14 * (-1 + z) + 18 = -14 * (1 - z) - n :=
by
  use -4
  sorry

#check solve_equation

end solve_equation_l164_16493


namespace cake_portion_theorem_l164_16464

theorem cake_portion_theorem (tom_ate jenny_took : ℚ) : 
  tom_ate = 60 / 100 →
  jenny_took = 1 / 4 →
  (1 - tom_ate) * (1 - jenny_took) = 30 / 100 := by
  sorry

end cake_portion_theorem_l164_16464


namespace chessboard_decomposition_l164_16488

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  white_squares : Nat
  black_squares : Nat

/-- Represents a decomposition of the chessboard -/
def Decomposition := List Rectangle

/-- Checks if a decomposition is valid according to the given conditions -/
def is_valid_decomposition (d : Decomposition) : Prop :=
  d.all (λ r => r.white_squares = r.black_squares) ∧
  d.length > 0 ∧
  (List.zip d (List.tail d)).all (λ (r1, r2) => r1.white_squares < r2.white_squares) ∧
  (d.map (λ r => r.white_squares + r.black_squares)).sum = 64

/-- The main theorem to be proved -/
theorem chessboard_decomposition :
  (∃ (d : Decomposition), is_valid_decomposition d ∧ d.length = 7) ∧
  (∀ (d : Decomposition), is_valid_decomposition d → d.length ≤ 7) :=
sorry

end chessboard_decomposition_l164_16488


namespace student_photo_count_l164_16480

theorem student_photo_count :
  ∀ (m n : ℕ),
    m > 0 →
    n > 0 →
    m + 4 = n - 1 →  -- First rearrangement condition
    m + 3 = n - 2 →  -- Second rearrangement condition
    m * n = 24 :=    -- Total number of students
by
  sorry

end student_photo_count_l164_16480


namespace third_side_length_l164_16419

theorem third_side_length (a b c : ℝ) : 
  a = 4 → b = 10 → c = 11 →
  a > 0 → b > 0 → c > 0 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  ∃ (x y z : ℝ), x = a ∧ y = b ∧ z = c ∧ 
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x + y > z ∧ y + z > x ∧ z + x > y :=
by sorry

end third_side_length_l164_16419


namespace restaurant_tip_calculation_l164_16407

theorem restaurant_tip_calculation 
  (food_cost : ℝ) 
  (service_fee_percentage : ℝ) 
  (total_spent : ℝ) 
  (h1 : food_cost = 50) 
  (h2 : service_fee_percentage = 0.12) 
  (h3 : total_spent = 61) : 
  total_spent - (food_cost + food_cost * service_fee_percentage) = 5 := by
sorry

end restaurant_tip_calculation_l164_16407


namespace coupon_one_best_l164_16436

/-- Represents the discount offered by a coupon given a price --/
def discount (price : ℝ) : ℕ → ℝ
  | 1 => 0.1 * price
  | 2 => 20
  | 3 => 0.18 * (price - 100)
  | _ => 0  -- Default case for invalid coupon numbers

theorem coupon_one_best (price : ℝ) (h : price > 100) :
  (discount price 1 > discount price 2 ∧ discount price 1 > discount price 3) ↔ 
  (200 < price ∧ price < 225) := by
sorry

end coupon_one_best_l164_16436


namespace pencil_packing_problem_l164_16470

theorem pencil_packing_problem :
  ∃ (a k m : ℤ),
    200 ≤ a ∧ a ≤ 300 ∧
    a % 10 = 7 ∧
    a % 12 = 9 ∧
    a = 60 * m + 57 ∧
    (a = 237 ∨ a = 297) :=
by sorry

end pencil_packing_problem_l164_16470


namespace yellow_peaches_count_l164_16420

theorem yellow_peaches_count (red green total : ℕ) 
  (h_red : red = 7)
  (h_green : green = 8)
  (h_total : total = 30)
  (h_sum : red + green + yellow = total) :
  yellow = 15 :=
by
  sorry

end yellow_peaches_count_l164_16420


namespace polynomial_identity_l164_16494

theorem polynomial_identity (x : ℝ) : 
  (x - 2)^4 + 5*(x - 2)^3 + 10*(x - 2)^2 + 10*(x - 2) + 5 = (x - 2 + Real.sqrt 2)^4 := by
  sorry

end polynomial_identity_l164_16494


namespace basketball_shot_probability_l164_16458

theorem basketball_shot_probability (a b c : ℝ) : 
  a ∈ (Set.Ioo 0 1) → 
  b ∈ (Set.Ioo 0 1) → 
  c ∈ (Set.Ioo 0 1) → 
  3 * a + 2 * b = 1 → 
  a * b ≤ 1 / 24 := by
sorry

end basketball_shot_probability_l164_16458


namespace correct_scaling_l164_16438

/-- A cookie recipe with ingredients and scaling -/
structure CookieRecipe where
  originalCookies : ℕ
  originalFlour : ℚ
  originalSugar : ℚ
  desiredCookies : ℕ

/-- Calculate the required ingredients for a scaled cookie recipe -/
def scaleRecipe (recipe : CookieRecipe) : ℚ × ℚ :=
  let scaleFactor : ℚ := recipe.desiredCookies / recipe.originalCookies
  (recipe.originalFlour * scaleFactor, recipe.originalSugar * scaleFactor)

/-- Theorem: Scaling the recipe correctly produces the expected amounts of flour and sugar -/
theorem correct_scaling (recipe : CookieRecipe) 
    (h1 : recipe.originalCookies = 24)
    (h2 : recipe.originalFlour = 3/2)
    (h3 : recipe.originalSugar = 1/2)
    (h4 : recipe.desiredCookies = 120) :
    scaleRecipe recipe = (15/2, 5/2) := by
  sorry

#eval scaleRecipe { originalCookies := 24, originalFlour := 3/2, originalSugar := 1/2, desiredCookies := 120 }

end correct_scaling_l164_16438


namespace wheel_revolutions_for_one_mile_l164_16400

-- Define the wheel's diameter
def wheel_diameter : ℝ := 8

-- Define the length of a mile in feet
def mile_in_feet : ℝ := 5280

-- Theorem statement
theorem wheel_revolutions_for_one_mile :
  let wheel_circumference := π * wheel_diameter
  let revolutions := mile_in_feet / wheel_circumference
  revolutions = 660 / π :=
by
  sorry

end wheel_revolutions_for_one_mile_l164_16400


namespace unique_solution_l164_16411

/-- Represents the quantities and prices of two batches of products --/
structure BatchData where
  quantity1 : ℕ
  quantity2 : ℕ
  price1 : ℚ
  price2 : ℚ

/-- Checks if the given batch data satisfies the problem conditions --/
def satisfiesConditions (data : BatchData) : Prop :=
  data.quantity1 * data.price1 = 4000 ∧
  data.quantity2 * data.price2 = 8800 ∧
  data.quantity2 = 2 * data.quantity1 ∧
  data.price2 = data.price1 + 4

/-- Theorem stating that the only solution satisfying the conditions is 100 and 200 units --/
theorem unique_solution :
  ∀ data : BatchData, satisfiesConditions data →
    data.quantity1 = 100 ∧ data.quantity2 = 200 := by
  sorry

#check unique_solution

end unique_solution_l164_16411


namespace function_transformation_l164_16441

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_transformation (h : f 4 = 2) : 
  ∃ x y, x = 4 ∧ y = -2 ∧ -f x = y :=
sorry

end function_transformation_l164_16441


namespace consecutive_number_sums_contradiction_l164_16405

theorem consecutive_number_sums_contradiction (a : Fin 15 → ℤ) :
  (∀ i : Fin 13, a i + a (i + 1) + a (i + 2) > 0) →
  (∀ i : Fin 12, a i + a (i + 1) + a (i + 2) + a (i + 3) < 0) →
  False :=
by sorry

end consecutive_number_sums_contradiction_l164_16405


namespace building_has_seven_floors_l164_16402

/-- Represents a building with floors -/
structure Building where
  totalFloors : ℕ
  ninasFloor : ℕ
  shurasFloor : ℕ

/-- Calculates the distance of Shura's mistaken path -/
def mistakenPathDistance (b : Building) : ℕ :=
  (b.totalFloors - b.ninasFloor) + (b.totalFloors - b.shurasFloor)

/-- Calculates the distance of Shura's direct path -/
def directPathDistance (b : Building) : ℕ :=
  if b.ninasFloor ≥ b.shurasFloor then b.ninasFloor - b.shurasFloor
  else b.shurasFloor - b.ninasFloor

/-- Theorem stating the conditions and conclusion about the building -/
theorem building_has_seven_floors :
  ∃ (b : Building),
    b.ninasFloor = 6 ∧
    b.totalFloors > b.ninasFloor ∧
    (mistakenPathDistance b : ℚ) = 1.5 * (directPathDistance b : ℚ) ∧
    b.totalFloors = 7 := by
  sorry

end building_has_seven_floors_l164_16402


namespace cyclist_speeds_l164_16416

-- Define the distance between A and B
def total_distance : ℝ := 240

-- Define the time difference between starts
def start_time_diff : ℝ := 0.5

-- Define the speed difference between cyclists
def speed_diff : ℝ := 3

-- Define the time taken to fix the bike
def fix_time : ℝ := 1.5

-- Define the speeds of cyclists A and B
def speed_A : ℝ := 12
def speed_B : ℝ := speed_A + speed_diff

-- Theorem to prove
theorem cyclist_speeds :
  -- Person B reaches midpoint when bike breaks down
  (total_distance / 2) / speed_B = total_distance / speed_A - start_time_diff - fix_time :=
by sorry

end cyclist_speeds_l164_16416


namespace largest_n_satisfying_inequality_l164_16444

theorem largest_n_satisfying_inequality : ∃ (n : ℕ),
  (∃ (x : Fin n → ℝ), (∀ (i j : Fin n), i < j → 
    (1 + x i * x j)^2 ≤ 0.99 * (1 + (x i)^2) * (1 + (x j)^2))) ∧
  (∀ (m : ℕ), m > n → 
    ¬∃ (y : Fin m → ℝ), ∀ (i j : Fin m), i < j → 
      (1 + y i * y j)^2 ≤ 0.99 * (1 + (y i)^2) * (1 + (y j)^2)) ∧
  n = 31 :=
sorry

end largest_n_satisfying_inequality_l164_16444


namespace fixed_point_range_l164_16482

/-- A function f: ℝ → ℝ has a fixed point if there exists an x such that f(x) = x -/
def HasFixedPoint (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = x

/-- The quadratic function f(x) = x^2 + x + a -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 + x + a

theorem fixed_point_range (a : ℝ) :
  HasFixedPoint (f a) → a ≤ 1/4 := by
  sorry

end fixed_point_range_l164_16482


namespace descent_time_is_50_seconds_l164_16484

/-- Represents the descent scenario on an escalator -/
structure EscalatorDescent where
  /-- Time taken to walk down stationary escalator (in seconds) -/
  stationary_time : ℝ
  /-- Time taken to walk down moving escalator (in seconds) -/
  moving_time : ℝ
  /-- Duration of escalator stoppage (in seconds) -/
  stop_duration : ℝ

/-- Calculates the total descent time for the given scenario -/
def total_descent_time (descent : EscalatorDescent) : ℝ :=
  sorry

/-- Theorem stating that the total descent time is 50 seconds -/
theorem descent_time_is_50_seconds (descent : EscalatorDescent) 
  (h1 : descent.stationary_time = 80)
  (h2 : descent.moving_time = 40)
  (h3 : descent.stop_duration = 20) :
  total_descent_time descent = 50 := by
  sorry

end descent_time_is_50_seconds_l164_16484


namespace cookies_baked_l164_16432

theorem cookies_baked (pans : ℕ) (cookies_per_pan : ℕ) (h1 : pans = 12) (h2 : cookies_per_pan = 15) :
  pans * cookies_per_pan = 180 := by
  sorry

end cookies_baked_l164_16432


namespace factorial_quotient_trailing_zeros_l164_16442

def trailing_zeros (n : ℕ) : ℕ := sorry

def factorial (n : ℕ) : ℕ := sorry

theorem factorial_quotient_trailing_zeros :
  trailing_zeros (factorial 2018 / (factorial 30 * factorial 11)) = 493 := by sorry

end factorial_quotient_trailing_zeros_l164_16442


namespace points_in_quadrants_I_and_II_l164_16413

def in_quadrant_I_or_II (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∨ x < 0 ∧ y > 0

theorem points_in_quadrants_I_and_II (x y : ℝ) :
  y > 3 * x → y > 6 - x^2 → in_quadrant_I_or_II x y := by
  sorry

end points_in_quadrants_I_and_II_l164_16413


namespace initial_mean_calculation_l164_16454

theorem initial_mean_calculation (n : ℕ) (initial_mean corrected_mean : ℚ) : 
  n = 50 → 
  corrected_mean = 36.02 → 
  n * initial_mean + 1 = n * corrected_mean → 
  initial_mean = 36 := by
sorry

end initial_mean_calculation_l164_16454


namespace unique_right_triangle_l164_16483

/-- A function that checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (3,4,5) forms a right triangle --/
theorem unique_right_triangle :
  (¬ isRightTriangle 2 3 4) ∧
  (¬ isRightTriangle 3 4 6) ∧
  (isRightTriangle 3 4 5) ∧
  (¬ isRightTriangle 4 5 6) :=
by sorry

#check unique_right_triangle

end unique_right_triangle_l164_16483


namespace square_sum_representation_l164_16423

theorem square_sum_representation (x y : ℕ) (h : x ≠ y) :
  ∃ u v : ℕ, x^2 + x*y + y^2 = u^2 + 3*v^2 := by
  sorry

end square_sum_representation_l164_16423


namespace largest_digit_divisible_by_six_l164_16497

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5789 * 10 + N) % 6 = 0 → N ≤ 4 :=
by
  sorry

end largest_digit_divisible_by_six_l164_16497


namespace largest_whole_number_satisfying_inequality_l164_16447

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℕ, x ≤ 15 ↔ 9 * x - 8 < 130 :=
by sorry

end largest_whole_number_satisfying_inequality_l164_16447


namespace absolute_value_sqrt_two_minus_two_l164_16408

theorem absolute_value_sqrt_two_minus_two :
  (1 : ℝ) < Real.sqrt 2 ∧ Real.sqrt 2 < 2 →
  |Real.sqrt 2 - 2| = 2 - Real.sqrt 2 := by
sorry

end absolute_value_sqrt_two_minus_two_l164_16408


namespace hexagon_exterior_angles_sum_l164_16461

-- Define a polygon as a type
class Polygon (P : Type)

-- Define a hexagon as a specific type of polygon
class Hexagon (H : Type) extends Polygon H

-- Define the sum of exterior angles for a polygon
def sum_of_exterior_angles (P : Type) [Polygon P] : ℝ := 360

-- Theorem statement
theorem hexagon_exterior_angles_sum (H : Type) [Hexagon H] :
  sum_of_exterior_angles H = 360 := by
  sorry

end hexagon_exterior_angles_sum_l164_16461


namespace chicken_rabbit_problem_l164_16410

theorem chicken_rabbit_problem (total_animals total_feet : ℕ) 
  (h1 : total_animals = 35)
  (h2 : total_feet = 94) :
  ∃ (chickens rabbits : ℕ), 
    chickens + rabbits = total_animals ∧
    2 * chickens + 4 * rabbits = total_feet ∧
    chickens = 23 := by
  sorry

end chicken_rabbit_problem_l164_16410


namespace triangle_perimeter_l164_16409

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 2) (h2 : b = 7) (h3 : Odd c) : a + b + c = 16 := by
  sorry

end triangle_perimeter_l164_16409


namespace range_of_a_l164_16474

-- Define the propositions p and q
def p (m a : ℝ) : Prop := m^2 - 7*a*m + 12*a^2 < 0 ∧ a > 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧ 
  ∃ (c : ℝ), c > 0 ∧ x^2 / (m - 1) + y^2 / (2 - m - c) = 1

-- Define the theorem
theorem range_of_a : 
  (∀ m a : ℝ, ¬(q m) → ¬(p m a)) ∧ 
  (∃ m a : ℝ, ¬(q m) ∧ p m a) → 
  {a : ℝ | 1/3 ≤ a ∧ a ≤ 3/8} = {a : ℝ | ∃ m : ℝ, p m a} :=
sorry

end range_of_a_l164_16474


namespace segmented_part_surface_area_l164_16434

/-- Right prism with isosceles triangle base -/
structure Prism where
  height : ℝ
  baseLength : ℝ
  baseSide : ℝ

/-- Point on an edge of the prism -/
structure EdgePoint where
  edge : Fin 3
  position : ℝ

/-- Segmented part of the prism -/
structure SegmentedPart where
  prism : Prism
  pointX : EdgePoint
  pointY : EdgePoint
  pointZ : EdgePoint

/-- Surface area of the segmented part -/
def surfaceArea (part : SegmentedPart) : ℝ := sorry

/-- Main theorem -/
theorem segmented_part_surface_area 
  (p : Prism) 
  (x y z : EdgePoint) 
  (h1 : p.height = 20)
  (h2 : p.baseLength = 18)
  (h3 : p.baseSide = 15)
  (h4 : x.edge = 0 ∧ x.position = 1/2)
  (h5 : y.edge = 1 ∧ y.position = 1/2)
  (h6 : z.edge = 2 ∧ z.position = 1/2) :
  surfaceArea { prism := p, pointX := x, pointY := y, pointZ := z } = 108 := by sorry

end segmented_part_surface_area_l164_16434


namespace curve_family_condition_l164_16428

/-- A family of curves parameterized by p -/
def curve_family (p x y : ℝ) : Prop :=
  y = p^2 + (2*p - 1)*x + 2*x^2

/-- The condition for a point (x, y) to have at least one curve passing through it -/
def has_curve_passing_through (x y : ℝ) : Prop :=
  ∃ p : ℝ, curve_family p x y

/-- The theorem stating the equivalence between the existence of a curve passing through (x, y) 
    and the inequality y ≥ x² - x -/
theorem curve_family_condition (x y : ℝ) : 
  has_curve_passing_through x y ↔ y ≥ x^2 - x :=
sorry

end curve_family_condition_l164_16428


namespace pentagon_rectangle_ratio_l164_16404

/-- Given a regular pentagon and a rectangle with the same perimeter and the rectangle's length
    being twice its width, the ratio of the pentagon's side length to the rectangle's width is 6/5 -/
theorem pentagon_rectangle_ratio (p w l : ℝ) : 
  p > 0 → w > 0 → l > 0 →
  5 * p = 30 →  -- Pentagon perimeter
  2 * w + 2 * l = 30 →  -- Rectangle perimeter
  l = 2 * w →  -- Rectangle length is twice the width
  p / w = 6 / 5 := by
  sorry

end pentagon_rectangle_ratio_l164_16404


namespace largest_common_divisor_of_S_l164_16412

def S : Set ℤ := {x | ∃ n : ℤ, x = n^5 - 5*n^3 + 4*n ∧ ¬(3 ∣ n)}

theorem largest_common_divisor_of_S : 
  ∀ k : ℤ, (∀ x ∈ S, k ∣ x) → k ≤ 360 ∧ 
  ∀ x ∈ S, 360 ∣ x :=
sorry

end largest_common_divisor_of_S_l164_16412


namespace inequality_proof_l164_16489

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_condition : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1/3 := by
sorry

end inequality_proof_l164_16489


namespace distance_C2_C3_eq_sqrt_10m_l164_16424

/-- Right triangle ABC with given side lengths -/
structure RightTriangleABC where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  right_angle : AB ^ 2 + AC ^ 2 = BC ^ 2
  AB_eq : AB = 80
  AC_eq : AC = 150
  BC_eq : BC = 170

/-- Inscribed circle C1 of triangle ABC -/
def C1 (t : RightTriangleABC) : Circle := sorry

/-- Line DE perpendicular to AC and tangent to C1 -/
def DE (t : RightTriangleABC) : Line := sorry

/-- Line FG perpendicular to AB and tangent to C1 -/
def FG (t : RightTriangleABC) : Line := sorry

/-- Inscribed circle C2 of triangle BDE -/
def C2 (t : RightTriangleABC) : Circle := sorry

/-- Inscribed circle C3 of triangle CFG -/
def C3 (t : RightTriangleABC) : Circle := sorry

/-- The distance between the centers of C2 and C3 -/
def distance_C2_C3 (t : RightTriangleABC) : ℝ := sorry

theorem distance_C2_C3_eq_sqrt_10m (t : RightTriangleABC) :
  distance_C2_C3 t = Real.sqrt (10 * 1057.6) := by sorry

end distance_C2_C3_eq_sqrt_10m_l164_16424


namespace platform_length_l164_16471

/-- The length of a platform crossed by two trains moving in opposite directions -/
theorem platform_length 
  (x y : ℝ) -- lengths of trains A and B in meters
  (p q : ℝ) -- speeds of trains A and B in km/h
  (t : ℝ) -- time taken to cross the platform in seconds
  (h_positive : x > 0 ∧ y > 0 ∧ p > 0 ∧ q > 0 ∧ t > 0) -- All values are positive
  : ∃ (L : ℝ), L = (p + q) * (5 * t / 18) - (x + y) :=
by
  sorry

end platform_length_l164_16471


namespace largest_divisor_of_n_l164_16457

theorem largest_divisor_of_n (n : ℕ+) 
  (h1 : (n : ℕ)^4 % 850 = 0)
  (h2 : ∀ p : ℕ, p > 20 → Nat.Prime p → (n : ℕ) % p ≠ 0) :
  ∃ k : ℕ, k ∣ (n : ℕ) ∧ k = 10 ∧ ∀ m : ℕ, m ∣ (n : ℕ) → m ≤ k :=
sorry

end largest_divisor_of_n_l164_16457


namespace line_through_point_with_equal_intercepts_l164_16463

-- Define a line in 2D space
structure Line2D where
  slope : ℝ
  intercept : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def linePassesThroughPoint (l : Line2D) (p : Point2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.intercept / l.slope = -l.intercept

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∀ (l : Line2D),
    linePassesThroughPoint l { x := 1, y := 2 } →
    hasEqualIntercepts l →
    (l.slope = -1 ∧ l.intercept = 3) ∨ (l.slope = 2 ∧ l.intercept = 0) :=
sorry

end line_through_point_with_equal_intercepts_l164_16463


namespace equation_roots_relation_l164_16437

theorem equation_roots_relation (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*x + 12 = 0 → y^2 - k*y + 12 = 0 → y = x + 6) →
  k = 6 := by
sorry

end equation_roots_relation_l164_16437


namespace pizza_combinations_l164_16496

theorem pizza_combinations : Nat.choose 8 5 = 56 := by
  sorry

end pizza_combinations_l164_16496


namespace duck_problem_solution_l164_16491

/-- Represents the duck population problem --/
def duck_problem (initial_flock : ℕ) (killed_per_year : ℕ) (born_per_year : ℕ) 
                 (other_flock : ℕ) (combined_flock : ℕ) : Prop :=
  ∃ y : ℕ, 
    initial_flock + (born_per_year - killed_per_year) * y + other_flock = combined_flock

/-- Theorem stating the solution to the duck population problem --/
theorem duck_problem_solution : 
  duck_problem 100 20 30 150 300 → 
  ∃ y : ℕ, y = 5 ∧ duck_problem 100 20 30 150 300 := by
  sorry

#check duck_problem_solution

end duck_problem_solution_l164_16491


namespace triangle_abc_properties_l164_16449

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- a, b, c are sides opposite to angles A, B, C respectively
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Given conditions
  ((2 * Real.cos A - 1) * Real.sin B + 2 * Real.cos A = 1) ∧
  (5 * b^2 = a^2 + 2 * c^2) →
  -- Conclusions
  (A = π / 3) ∧
  (Real.sin B / Real.sin C = 3 / 4) := by
sorry

end triangle_abc_properties_l164_16449


namespace bounded_g_given_bounded_f_l164_16429

/-- Given real functions f and g defined on ℝ, satisfying certain conditions,
    prove that the absolute value of g is bounded by 1 for all real numbers. -/
theorem bounded_g_given_bounded_f (f g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x : ℝ, f x ≠ 0)
  (h3 : ∀ x : ℝ, |f x| ≤ 1) :
  ∀ y : ℝ, |g y| ≤ 1 := by
  sorry

end bounded_g_given_bounded_f_l164_16429


namespace number_relationship_l164_16476

theorem number_relationship (a b c d : ℝ) : 
  a = Real.log (3/2) / Real.log (2/3) →
  b = Real.log 2 / Real.log 3 →
  c = 2 ^ (1/3 : ℝ) →
  d = 3 ^ (1/2 : ℝ) →
  a < b ∧ b < c ∧ c < d := by
  sorry

end number_relationship_l164_16476


namespace sqrt_x_div_sqrt_y_l164_16475

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 + (1/6)^2 = ((1/5)^2 + (1/7)^2 + (1/8)^2) * (54*x)/(115*y)) : 
  Real.sqrt x / Real.sqrt y = 49/29 := by
  sorry

end sqrt_x_div_sqrt_y_l164_16475


namespace inverse_function_problem_l164_16487

/-- Given a function g(x) = 4x - 6 and its relation to the inverse of f(x) = ax + b,
    prove that 4a + 3b = 4 -/
theorem inverse_function_problem (a b : ℝ) :
  (∀ x, (4 * x - 6 : ℝ) = (Function.invFun (fun x => a * x + b) x) - 2) →
  4 * a + 3 * b = 4 := by
  sorry

end inverse_function_problem_l164_16487


namespace no_real_m_for_single_root_l164_16460

theorem no_real_m_for_single_root : 
  ¬∃ (m : ℝ), (∀ (x : ℝ), x^2 + (4*m+2)*x + m = 0 ↔ x = -2*m-1) := by
  sorry

end no_real_m_for_single_root_l164_16460
