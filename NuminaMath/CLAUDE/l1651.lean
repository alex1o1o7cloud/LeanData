import Mathlib

namespace inequality_proof_l1651_165195

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  (a - b) * c^2 ≤ 0 := by
  sorry

end inequality_proof_l1651_165195


namespace maggie_red_packs_l1651_165168

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := sorry

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := 8

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := 160

theorem maggie_red_packs : red_packs = 4 := by
  sorry

end maggie_red_packs_l1651_165168


namespace min_reciprocal_sum_l1651_165129

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a - b)^2 = 4*(a*b)^3) :
  (1/a + 1/b) ≥ 2 * Real.sqrt 2 := by
  sorry

end min_reciprocal_sum_l1651_165129


namespace modular_inverse_of_9_mod_23_l1651_165102

theorem modular_inverse_of_9_mod_23 : ∃ x : ℕ, x ∈ Finset.range 23 ∧ (9 * x) % 23 = 1 :=
by
  -- The proof goes here
  sorry

end modular_inverse_of_9_mod_23_l1651_165102


namespace range_of_f_l1651_165173

def f (x : ℝ) := x^4 - 4*x^2 + 4

theorem range_of_f :
  Set.range f = Set.Ici (0 : ℝ) := by sorry

end range_of_f_l1651_165173


namespace parabola_properties_l1651_165150

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem parabola_properties :
  (∃ (x y : ℝ), IsLocalMin f x ∧ f x = y ∧ x = -1 ∧ y = -4) ∧
  (∀ x : ℝ, x ≥ 2 → f x ≥ 5) ∧
  (∃ x : ℝ, x ≥ 2 ∧ f x = 5) := by
  sorry

end parabola_properties_l1651_165150


namespace no_valid_box_dimensions_l1651_165158

theorem no_valid_box_dimensions : 
  ¬∃ (a b c : ℕ), 
    (1 ≤ a) ∧ (a ≤ b) ∧ (b ≤ c) ∧ 
    (a * b * c = 3 * (2 * a * b + 2 * b * c + 2 * c * a)) :=
by sorry

end no_valid_box_dimensions_l1651_165158


namespace dads_real_age_l1651_165167

theorem dads_real_age (reported_age : ℕ) (h : reported_age = 35) : 
  ∃ (real_age : ℕ), (5 : ℚ) / 7 * real_age = reported_age ∧ real_age = 49 := by
  sorry

end dads_real_age_l1651_165167


namespace vehicles_meeting_time_l1651_165124

/-- The time taken for two vehicles to meet when traveling towards each other -/
theorem vehicles_meeting_time (distance : ℝ) (speed1 speed2 : ℝ) (h1 : distance = 480) 
  (h2 : speed1 = 65) (h3 : speed2 = 55) : 
  (distance / (speed1 + speed2)) = 4 := by
  sorry

end vehicles_meeting_time_l1651_165124


namespace square_area_problem_l1651_165180

theorem square_area_problem (a b c : ℕ) : 
  4 * a < b → c^2 = a^2 + b^2 + 10 → c^2 = 36 := by
  sorry

end square_area_problem_l1651_165180


namespace scramble_language_word_count_l1651_165120

/-- The number of letters in the Scramble alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum word length in the Scramble language -/
def max_word_length : ℕ := 5

/-- Calculates the number of words of a given length that contain at least one 'B' -/
def words_with_b (length : ℕ) : ℕ :=
  alphabet_size ^ length - (alphabet_size - 1) ^ length

/-- The total number of valid words in the Scramble language -/
def total_valid_words : ℕ :=
  words_with_b 1 + words_with_b 2 + words_with_b 3 + words_with_b 4 + words_with_b 5

theorem scramble_language_word_count :
  total_valid_words = 1863701 :=
by sorry

end scramble_language_word_count_l1651_165120


namespace purple_car_count_l1651_165117

theorem purple_car_count (total : ℕ) (purple blue red orange yellow green : ℕ)
  (h_total : total = 987)
  (h_blue : blue = 2 * red)
  (h_red : red = 3 * orange)
  (h_yellow1 : yellow = orange / 2)
  (h_yellow2 : yellow = 3 * purple)
  (h_green : green = 5 * purple)
  (h_sum : purple + yellow + orange + red + blue + green = total) :
  purple = 14 := by
  sorry

end purple_car_count_l1651_165117


namespace intersection_M_N_l1651_165131

def M : Set ℝ := {x | x^2 - 1 < 0}

def N : Set ℝ := {y | ∃ x ∈ M, y = Real.log (x + 2)}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end intersection_M_N_l1651_165131


namespace john_vowel_learning_days_l1651_165162

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of days John takes to learn one alphabet -/
def days_per_alphabet : ℕ := 3

/-- The total number of days John needs to finish learning all vowels -/
def total_days : ℕ := num_vowels * days_per_alphabet

theorem john_vowel_learning_days : total_days = 15 := by
  sorry

end john_vowel_learning_days_l1651_165162


namespace intersection_A_B_l1651_165145

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {x | x^2 - 4 ≤ 0}

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l1651_165145


namespace largest_four_digit_negative_congruent_to_2_mod_25_l1651_165148

theorem largest_four_digit_negative_congruent_to_2_mod_25 :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n % 25 = 2 → n ≤ -1023 :=
by sorry

end largest_four_digit_negative_congruent_to_2_mod_25_l1651_165148


namespace triangular_square_l1651_165133

/-- Triangular numbers -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Main theorem -/
theorem triangular_square (m n : ℕ) (h : 2 * triangular m = triangular n) :
  triangular (2 * m - n) = (m - n) ^ 2 := by
  sorry

end triangular_square_l1651_165133


namespace banana_distribution_l1651_165164

/-- The number of bananas each child would normally receive -/
def normal_bananas : ℕ := 2

/-- The number of absent children -/
def absent_children : ℕ := 330

/-- The number of extra bananas each child received due to absences -/
def extra_bananas : ℕ := 2

/-- The actual number of children in the school -/
def actual_children : ℕ := 660

theorem banana_distribution (total_bananas : ℕ) :
  (total_bananas = normal_bananas * actual_children) ∧
  (total_bananas = (normal_bananas + extra_bananas) * (actual_children - absent_children)) →
  actual_children = 660 := by
  sorry

end banana_distribution_l1651_165164


namespace jacket_cost_ratio_l1651_165144

theorem jacket_cost_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_ratio : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_ratio)
  let cost_ratio : ℝ := 2/3
  let cost : ℝ := selling_price * cost_ratio
  cost / marked_price = 1/2 := by
sorry

end jacket_cost_ratio_l1651_165144


namespace parabola_vertex_l1651_165101

/-- The parabola is defined by the equation y = (x + 3)^2 - 1 -/
def parabola (x : ℝ) : ℝ := (x + 3)^2 - 1

/-- The vertex of the parabola y = (x + 3)^2 - 1 is at the point (-3, -1) -/
theorem parabola_vertex : 
  (∃ (a : ℝ), ∀ (x : ℝ), parabola x = a * (x + 3)^2 - 1) → 
  (∀ (x : ℝ), parabola x ≥ parabola (-3)) ∧ parabola (-3) = -1 :=
sorry

end parabola_vertex_l1651_165101


namespace same_color_probability_l1651_165194

/-- The probability of drawing two balls of the same color from a bag containing
    8 blue balls and 7 yellow balls, with replacement. -/
theorem same_color_probability (blue_balls yellow_balls : ℕ) 
    (h_blue : blue_balls = 8) (h_yellow : yellow_balls = 7) :
    let total_balls := blue_balls + yellow_balls
    let p_blue := blue_balls / total_balls
    let p_yellow := yellow_balls / total_balls
    p_blue ^ 2 + p_yellow ^ 2 = 113 / 225 := by
  sorry

end same_color_probability_l1651_165194


namespace burger_cost_is_100_cents_l1651_165123

/-- The cost of items in cents -/
structure ItemCosts where
  burger : ℕ
  soda : ℕ
  fries : ℕ

/-- Alice's purchase -/
def alice_purchase (costs : ItemCosts) : ℕ :=
  4 * costs.burger + 3 * costs.soda + costs.fries

/-- Bob's purchase -/
def bob_purchase (costs : ItemCosts) : ℕ :=
  3 * costs.burger + 2 * costs.soda + 2 * costs.fries

/-- Theorem stating that the cost of a burger is 100 cents -/
theorem burger_cost_is_100_cents :
  ∃ (costs : ItemCosts),
    alice_purchase costs = 540 ∧
    bob_purchase costs = 580 ∧
    costs.burger = 100 := by
  sorry

end burger_cost_is_100_cents_l1651_165123


namespace homothety_composition_l1651_165175

open Complex

def H_i_squared (i z : ℂ) : ℂ := 2 * (z - i) + i

def T (i z : ℂ) : ℂ := z + i

def H_0_squared (z : ℂ) : ℂ := 2 * z

theorem homothety_composition (i z : ℂ) : H_i_squared i z = (T i ∘ H_0_squared) (z - i) := by sorry

end homothety_composition_l1651_165175


namespace mothers_full_time_proportion_l1651_165178

/-- The proportion of mothers holding full-time jobs -/
def proportion_mothers_full_time : ℝ := sorry

/-- The proportion of fathers holding full-time jobs -/
def proportion_fathers_full_time : ℝ := 0.75

/-- The proportion of parents who are women -/
def proportion_women : ℝ := 0.4

/-- The proportion of parents who do not hold full-time jobs -/
def proportion_not_full_time : ℝ := 0.19

theorem mothers_full_time_proportion :
  proportion_mothers_full_time = 0.9 :=
by sorry

end mothers_full_time_proportion_l1651_165178


namespace polynomial_remainder_l1651_165154

def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 12*x^2 + 18*x - 22

theorem polynomial_remainder (x : ℝ) : 
  ∃ q : ℝ → ℝ, f x = (x - 4) * q x + 114 := by
sorry

end polynomial_remainder_l1651_165154


namespace turquoise_more_green_count_l1651_165199

/-- Represents the survey results about the perception of turquoise color --/
structure TurquoiseSurvey where
  total : Nat
  more_blue : Nat
  both : Nat
  neither : Nat

/-- Calculates the number of people who believe turquoise is "more green" --/
def more_green (survey : TurquoiseSurvey) : Nat :=
  survey.total - (survey.more_blue - survey.both) - survey.neither

/-- Theorem stating that given the survey conditions, 80 people believe turquoise is "more green" --/
theorem turquoise_more_green_count :
  ∀ (survey : TurquoiseSurvey),
  survey.total = 150 →
  survey.more_blue = 90 →
  survey.both = 40 →
  survey.neither = 20 →
  more_green survey = 80 := by
  sorry


end turquoise_more_green_count_l1651_165199


namespace count_non_adjacent_placements_correct_l1651_165170

/-- Represents an n × n grid board. -/
structure GridBoard where
  n : ℕ

/-- Counts the number of ways to place X and O on the grid such that they are not adjacent. -/
def countNonAdjacentPlacements (board : GridBoard) : ℕ :=
  board.n^4 - 3 * board.n^2 + 2 * board.n

/-- Theorem stating that countNonAdjacentPlacements gives the correct count. -/
theorem count_non_adjacent_placements_correct (board : GridBoard) :
  countNonAdjacentPlacements board =
    board.n^4 - 3 * board.n^2 + 2 * board.n :=
by sorry

end count_non_adjacent_placements_correct_l1651_165170


namespace intersection_parallel_or_intersect_intersection_parallel_implies_parallel_to_plane_l1651_165104

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic operations and relations
variable (belongs_to : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (plane_intersect : Plane → Plane → Line → Prop)

-- Theorem 1
theorem intersection_parallel_or_intersect
  (α β : Plane) (m n : Line)
  (h1 : plane_intersect α β m)
  (h2 : belongs_to n α) :
  parallel m n ∨ intersect m n :=
sorry

-- Theorem 2
theorem intersection_parallel_implies_parallel_to_plane
  (α β : Plane) (m n : Line)
  (h1 : plane_intersect α β m)
  (h2 : parallel m n) :
  parallel_plane n α ∨ parallel_plane n β :=
sorry

end intersection_parallel_or_intersect_intersection_parallel_implies_parallel_to_plane_l1651_165104


namespace units_digit_of_p_plus_five_l1651_165156

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

def has_positive_units_digit (n : ℕ) : Prop := n % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_p_plus_five (p : ℕ) 
  (h_even : is_positive_even p)
  (h_pos_digit : has_positive_units_digit p)
  (h_cube_square : units_digit (p^3) = units_digit (p^2)) :
  units_digit (p + 5) = 1 := by
  sorry

end units_digit_of_p_plus_five_l1651_165156


namespace multiply_fractions_l1651_165182

theorem multiply_fractions : 12 * (1 / 15) * 30 = 24 := by sorry

end multiply_fractions_l1651_165182


namespace congruence_solution_l1651_165152

theorem congruence_solution (x : ℤ) 
  (h1 : (2 + x) % (5^3) = 2^2 % (5^3))
  (h2 : (3 + x) % (7^3) = 3^2 % (7^3))
  (h3 : (4 + x) % (11^3) = 5^2 % (11^3)) :
  x % 385 = 307 := by
  sorry

end congruence_solution_l1651_165152


namespace angle_triple_supplement_l1651_165138

theorem angle_triple_supplement (x : ℝ) : 
  x = 3 * (180 - x) → x = 135 := by
  sorry

end angle_triple_supplement_l1651_165138


namespace not_always_fifteen_different_l1651_165107

/-- Represents a student with a t-shirt color and a pants color -/
structure Student :=
  (tshirt : Fin 15)
  (pants : Fin 15)

/-- The theorem stating that it's not always possible to find 15 students
    with all different t-shirt and pants colors -/
theorem not_always_fifteen_different (n : Nat) (h : n = 30) :
  ∃ (students : Finset Student),
    students.card = n ∧
    ∀ (subset : Finset Student),
      subset ⊆ students →
      subset.card = 15 →
      ∃ (s1 s2 : Student),
        s1 ∈ subset ∧ s2 ∈ subset ∧ s1 ≠ s2 ∧
        (s1.tshirt = s2.tshirt ∨ s1.pants = s2.pants) :=
by sorry

end not_always_fifteen_different_l1651_165107


namespace collinear_vectors_x_value_l1651_165185

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  collinear a b → x = (1 : ℝ) / 2 :=
by
  sorry

end collinear_vectors_x_value_l1651_165185


namespace choose_3_from_10_l1651_165100

theorem choose_3_from_10 : Nat.choose 10 3 = 120 := by
  sorry

end choose_3_from_10_l1651_165100


namespace rehabilitation_centers_count_l1651_165181

/-- The number of rehabilitation centers visited by Lisa, Jude, Han, and Jane. -/
def total_rehabilitation_centers (lisa jude han jane : ℕ) : ℕ :=
  lisa + jude + han + jane

/-- Theorem stating the total number of rehabilitation centers visited. -/
theorem rehabilitation_centers_count :
  ∃ (lisa jude han jane : ℕ),
    lisa = 6 ∧
    jude = lisa / 2 ∧
    han = 2 * jude - 2 ∧
    jane = 2 * han + 6 ∧
    total_rehabilitation_centers lisa jude han jane = 27 := by
  sorry

end rehabilitation_centers_count_l1651_165181


namespace john_saturday_earnings_l1651_165147

/-- The amount of money John earned on Saturday -/
def saturday_earnings : ℝ := sorry

/-- The amount of money John earned on Sunday -/
def sunday_earnings : ℝ := sorry

/-- The amount of money John earned the previous weekend -/
def previous_weekend_earnings : ℝ := 20

/-- The cost of the pogo stick -/
def pogo_stick_cost : ℝ := 60

/-- The additional amount John needs to buy the pogo stick -/
def additional_needed : ℝ := 13

theorem john_saturday_earnings :
  saturday_earnings = 18 ∧
  sunday_earnings = saturday_earnings / 2 ∧
  previous_weekend_earnings + saturday_earnings + sunday_earnings = pogo_stick_cost - additional_needed :=
by sorry

end john_saturday_earnings_l1651_165147


namespace infinitely_many_solutions_l1651_165103

theorem infinitely_many_solutions : Set.Infinite {n : ℤ | (n - 3) * (n + 5) > 0} := by
  sorry

end infinitely_many_solutions_l1651_165103


namespace cubic_roots_sum_cubes_l1651_165160

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (3 * a^3 - 5 * a^2 + 170 * a - 7 = 0) →
  (3 * b^3 - 5 * b^2 + 170 * b - 7 = 0) →
  (3 * c^3 - 5 * c^2 + 170 * c - 7 = 0) →
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 
  (11/3 - c)^3 + (11/3 - a)^3 + (11/3 - b)^3 := by
sorry

end cubic_roots_sum_cubes_l1651_165160


namespace uncle_fyodor_sandwiches_l1651_165118

theorem uncle_fyodor_sandwiches (sharik matroskin fyodor : ℕ) : 
  matroskin = 3 * sharik →
  fyodor = sharik + 21 →
  fyodor = 2 * (sharik + matroskin) →
  fyodor = 24 := by
sorry

end uncle_fyodor_sandwiches_l1651_165118


namespace no_real_solutions_for_x_l1651_165146

theorem no_real_solutions_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x + 1/y = 8) (eq2 : y + 1/x = 7/20) : False :=
by sorry

end no_real_solutions_for_x_l1651_165146


namespace quadratic_inequality_solution_set_l1651_165189

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 - 2*x + 3 < 0} = {x : ℝ | x < -3 ∨ x > 1} := by
  sorry

end quadratic_inequality_solution_set_l1651_165189


namespace units_digit_of_sum_l1651_165196

theorem units_digit_of_sum (n : ℕ) : n = 33^43 + 43^32 → n % 10 = 8 := by
  sorry

end units_digit_of_sum_l1651_165196


namespace count_divisible_by_eight_l1651_165177

theorem count_divisible_by_eight : ∃ n : ℕ, n = (Finset.filter (fun x => x % 8 = 0) (Finset.Icc 200 400)).card ∧ n = 24 := by
  sorry

end count_divisible_by_eight_l1651_165177


namespace arithmetic_computation_l1651_165143

theorem arithmetic_computation : -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11 := by
  sorry

end arithmetic_computation_l1651_165143


namespace largest_n_inequality_l1651_165183

theorem largest_n_inequality : ∃ (n : ℕ), n = 14 ∧ 
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → 
    (a^2 / (b/29 + c/31) + b^2 / (c/29 + a/31) + c^2 / (a/29 + b/31) ≥ n * (a + b + c))) ∧
  (∀ (m : ℕ), m > 14 → 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      (a^2 / (b/29 + c/31) + b^2 / (c/29 + a/31) + c^2 / (a/29 + b/31) < m * (a + b + c))) :=
by sorry

end largest_n_inequality_l1651_165183


namespace richard_david_age_difference_l1651_165169

/-- Represents the ages of the three sons -/
structure Ages where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def family_conditions (ages : Ages) : Prop :=
  ages.richard > ages.david ∧
  ages.david = ages.scott + 8 ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- The theorem to prove -/
theorem richard_david_age_difference (ages : Ages) 
  (h : family_conditions ages) : ages.richard - ages.david = 6 := by
  sorry

end richard_david_age_difference_l1651_165169


namespace probability_not_red_special_cube_l1651_165114

structure Cube where
  total_faces : ℕ
  green_faces : ℕ
  blue_faces : ℕ
  red_faces : ℕ

def probability_not_red (c : Cube) : ℚ :=
  (c.green_faces + c.blue_faces : ℚ) / c.total_faces

theorem probability_not_red_special_cube :
  let c : Cube := {
    total_faces := 6,
    green_faces := 3,
    blue_faces := 2,
    red_faces := 1
  }
  probability_not_red c = 5 / 6 := by
  sorry

end probability_not_red_special_cube_l1651_165114


namespace prob_same_heads_sum_l1651_165186

/-- Represents a coin with a given probability of landing heads -/
structure Coin where
  prob_heads : ℚ
  prob_heads_nonneg : 0 ≤ prob_heads
  prob_heads_le_one : prob_heads ≤ 1

/-- The set of three coins: two fair and one biased -/
def coin_set : Finset Coin := sorry

/-- The probability of getting the same number of heads when flipping the coin set twice -/
noncomputable def prob_same_heads (coins : Finset Coin) : ℚ := sorry

/-- The sum of numerator and denominator of the reduced fraction of prob_same_heads -/
noncomputable def sum_num_denom (coins : Finset Coin) : ℕ := sorry

theorem prob_same_heads_sum (h1 : coin_set.card = 3)
  (h2 : ∃ (c : Coin), c ∈ coin_set ∧ c.prob_heads = 1/2)
  (h3 : ∃ (c : Coin), c ∈ coin_set ∧ c.prob_heads = 3/5)
  (h4 : (coin_set.filter (fun c => c.prob_heads = 1/2)).card = 2) :
  sum_num_denom coin_set = 263 := by sorry

end prob_same_heads_sum_l1651_165186


namespace hexagon_side_count_l1651_165105

-- Define a convex hexagon with two distinct side lengths
structure ConvexHexagon where
  side_length1 : ℕ
  side_length2 : ℕ
  side_count1 : ℕ
  side_count2 : ℕ
  distinct_lengths : side_length1 ≠ side_length2
  total_sides : side_count1 + side_count2 = 6

-- Theorem statement
theorem hexagon_side_count (h : ConvexHexagon) 
  (side_ab : h.side_length1 = 7)
  (side_bc : h.side_length2 = 8)
  (perimeter : h.side_length1 * h.side_count1 + h.side_length2 * h.side_count2 = 46) :
  h.side_count2 = 4 := by
  sorry

end hexagon_side_count_l1651_165105


namespace screen_time_calculation_l1651_165142

/-- Calculates the remaining screen time for the evening given the total recommended time and time already used. -/
def remaining_screen_time (total_recommended : ℕ) (time_used : ℕ) : ℕ :=
  total_recommended - time_used

/-- Converts hours to minutes. -/
def hours_to_minutes (hours : ℕ) : ℕ :=
  hours * 60

theorem screen_time_calculation :
  let total_recommended := hours_to_minutes 2
  let time_used := 45
  remaining_screen_time total_recommended time_used = 75 := by
  sorry

end screen_time_calculation_l1651_165142


namespace shaded_region_perimeter_l1651_165111

theorem shaded_region_perimeter (r : ℝ) (h : r = 7) :
  let circle_fraction : ℝ := 3 / 4
  let arc_length := circle_fraction * (2 * π * r)
  let radii_length := 2 * r
  radii_length + arc_length = 14 + (21 / 2) * π := by
  sorry

end shaded_region_perimeter_l1651_165111


namespace range_of_a_l1651_165106

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - a| ≤ 2) ↔ a ∈ Set.Icc (-3) 1 := by
  sorry

end range_of_a_l1651_165106


namespace product_of_differences_of_squares_l1651_165193

theorem product_of_differences_of_squares : 
  let P := Real.sqrt 2023 + Real.sqrt 2022
  let Q := Real.sqrt 2023 - Real.sqrt 2022
  let R := Real.sqrt 2023 + Real.sqrt 2024
  let S := Real.sqrt 2023 - Real.sqrt 2024
  (P * Q) * (R * S) = -1 := by sorry

end product_of_differences_of_squares_l1651_165193


namespace average_of_x_and_y_l1651_165136

theorem average_of_x_and_y (x y : ℝ) : 
  (4 + 6 + 9 + x + y) / 5 = 20 → (x + y) / 2 = 40.5 := by
  sorry

end average_of_x_and_y_l1651_165136


namespace remainder_divisibility_l1651_165176

theorem remainder_divisibility (n : ℤ) : 
  ∃ k : ℤ, n = 125 * k + 40 → ∃ m : ℤ, n = 15 * m + 10 := by
  sorry

end remainder_divisibility_l1651_165176


namespace walking_distance_calculation_l1651_165132

-- Define the speeds and additional distances
def speed1_original : ℝ := 5
def speed1_alternative : ℝ := 15
def speed2_original : ℝ := 10
def speed2_alternative : ℝ := 20
def additional_distance1 : ℝ := 45
def additional_distance2 : ℝ := 30

-- Define the theorem
theorem walking_distance_calculation :
  ∃ (t1 t2 : ℝ),
    t1 > 0 ∧ t2 > 0 ∧
    (speed1_alternative * t1 - speed1_original * t1 = additional_distance1) ∧
    (speed2_alternative * t2 - speed2_original * t2 = additional_distance2) ∧
    (speed1_original * t1 = 22.5) ∧
    (speed2_original * t2 = 30) :=
  sorry

end walking_distance_calculation_l1651_165132


namespace power_seven_mod_nine_l1651_165191

theorem power_seven_mod_nine : 7^138 % 9 = 1 := by
  sorry

end power_seven_mod_nine_l1651_165191


namespace trigonometric_identity_l1651_165190

theorem trigonometric_identity (x : ℝ) (h : Real.sin (x + π / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 := by
  sorry

end trigonometric_identity_l1651_165190


namespace system_is_linear_l1651_165115

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and x and y are variables. -/
def IsLinearEquation (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, eq x y ↔ a * x + b * y = c

/-- A system of two linear equations is a pair of linear equations in two variables. -/
def IsSystemOfTwoLinearEquations (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  IsLinearEquation eq1 ∧ IsLinearEquation eq2

/-- The given system of equations. -/
def System : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) :=
  (fun x y ↦ x + y = 2, fun _ y ↦ y = 3)

theorem system_is_linear : IsSystemOfTwoLinearEquations System.1 System.2 := by
  sorry

#check system_is_linear

end system_is_linear_l1651_165115


namespace arithmetic_sequence_average_l1651_165116

/-- Given an arithmetic sequence with 7 terms, first term 10, and common difference 12,
    prove that the average of all terms is 46. -/
theorem arithmetic_sequence_average : 
  let n : ℕ := 7
  let a : ℕ := 10
  let d : ℕ := 12
  let sequence := (fun i => a + d * (i - 1))
  let sum := (sequence 1 + sequence n) * n / 2
  (sum : ℚ) / n = 46 := by
  sorry

end arithmetic_sequence_average_l1651_165116


namespace friends_team_assignment_l1651_165126

theorem friends_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k^n = 65536 := by
  sorry

end friends_team_assignment_l1651_165126


namespace combined_tax_rate_l1651_165172

theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.3)
  (h2 : mindy_rate = 0.2)
  (h3 : income_ratio = 3) :
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.225 := by
  sorry

end combined_tax_rate_l1651_165172


namespace quadratic_real_roots_l1651_165113

/-- Given a quadratic equation (a-5)x^2 - 4x - 1 = 0 with real roots, prove that a ≥ 1 and a ≠ 5 -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) → 
  (a ≥ 1 ∧ a ≠ 5) := by
  sorry

end quadratic_real_roots_l1651_165113


namespace fifth_term_of_sequence_l1651_165187

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fifth_term_of_sequence (x y : ℝ) (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_first : a 0 = x + 2*y)
  (h_second : a 1 = x - 2*y)
  (h_third : a 2 = 2*x*y)
  (h_fourth : a 3 = 2*x/y)
  (h_y_neq_half : y ≠ 1/2) :
  a 4 = (-12 - 8*y^2 + 4*y) / (2*y - 1) :=
sorry

end fifth_term_of_sequence_l1651_165187


namespace circumcircle_fixed_point_l1651_165135

/-- A parabola that intersects the coordinate axes at three different points -/
structure AxisIntersectingParabola where
  a : ℝ
  b : ℝ
  x₁ : ℝ
  x₂ : ℝ
  intersectsAxes : x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ b ≠ 0
  onParabola₁ : 0 = x₁^2 + a * x₁ + b
  onParabola₂ : 0 = x₂^2 + a * x₂ + b
  onParabola₃ : b = 0^2 + a * 0 + b

/-- The circumcircle of a triangle formed by the intersection points of a parabola with the coordinate axes passes through (0, 1) -/
theorem circumcircle_fixed_point (p : AxisIntersectingParabola) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center.1 - 0)^2 + (center.2 - 1)^2 = radius^2 ∧
    (center.1 - p.x₁)^2 + center.2^2 = radius^2 ∧
    (center.1 - p.x₂)^2 + center.2^2 = radius^2 ∧
    center.1^2 + (center.2 - p.b)^2 = radius^2 :=
sorry

end circumcircle_fixed_point_l1651_165135


namespace function_above_identity_l1651_165197

theorem function_above_identity (f : ℝ → ℝ) (hf : Continuous f) :
  (∀ a₁ ∈ Set.Ioo 0 1, ∀ n : ℕ, f^[n+1] a₁ > f^[n] a₁) →
  ∀ x ∈ Set.Ioo 0 1, f x > x :=
sorry

end function_above_identity_l1651_165197


namespace tangent_line_to_x_ln_x_l1651_165165

theorem tangent_line_to_x_ln_x (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (2 * x₀ + m = x₀ * Real.log x₀) ∧ 
    (2 = Real.log x₀ + 1)) → 
  m = -Real.exp 1 :=
by sorry

end tangent_line_to_x_ln_x_l1651_165165


namespace redo_profit_is_5000_l1651_165112

/-- Calculates the profit for Redo's horseshoe manufacturing --/
def horseshoe_profit (initial_outlay : ℕ) (cost_per_set : ℕ) (price_per_set : ℕ) (num_sets : ℕ) : ℤ :=
  let manufacturing_cost : ℕ := initial_outlay + cost_per_set * num_sets
  let revenue : ℕ := price_per_set * num_sets
  (revenue : ℤ) - (manufacturing_cost : ℤ)

theorem redo_profit_is_5000 :
  horseshoe_profit 10000 20 50 500 = 5000 := by
  sorry

end redo_profit_is_5000_l1651_165112


namespace free_throw_probabilities_l1651_165128

/-- Free throw success rates for players A and B -/
structure FreeThrowRates where
  player_a : ℚ
  player_b : ℚ

/-- Calculates the probability of exactly one successful shot when each player takes one free throw -/
def prob_one_success (rates : FreeThrowRates) : ℚ :=
  rates.player_a * (1 - rates.player_b) + rates.player_b * (1 - rates.player_a)

/-- Calculates the probability of at least one successful shot when each player takes two free throws -/
def prob_at_least_one_success (rates : FreeThrowRates) : ℚ :=
  1 - (1 - rates.player_a)^2 * (1 - rates.player_b)^2

/-- Theorem stating the probabilities for the given free throw rates -/
theorem free_throw_probabilities (rates : FreeThrowRates) 
  (h1 : rates.player_a = 1/2) (h2 : rates.player_b = 2/5) : 
  prob_one_success rates = 1/2 ∧ prob_at_least_one_success rates = 91/100 := by
  sorry

#eval prob_one_success ⟨1/2, 2/5⟩
#eval prob_at_least_one_success ⟨1/2, 2/5⟩

end free_throw_probabilities_l1651_165128


namespace jinyoung_fewest_marbles_l1651_165151

-- Define the number of marbles for each person
def minjeong_marbles : ℕ := 6
def joohwan_marbles : ℕ := 7
def sunho_marbles : ℕ := minjeong_marbles - 1
def jinyoung_marbles : ℕ := joohwan_marbles - 3

-- Define a function to get the number of marbles for each person
def marbles (person : String) : ℕ :=
  match person with
  | "Minjeong" => minjeong_marbles
  | "Joohwan" => joohwan_marbles
  | "Sunho" => sunho_marbles
  | "Jinyoung" => jinyoung_marbles
  | _ => 0

-- Theorem: Jinyoung has the fewest marbles
theorem jinyoung_fewest_marbles :
  ∀ person, person ≠ "Jinyoung" → marbles "Jinyoung" ≤ marbles person :=
by sorry

end jinyoung_fewest_marbles_l1651_165151


namespace similar_triangles_side_length_l1651_165184

/-- Given two similar triangles ABC and DEF, prove that DF = 6 -/
theorem similar_triangles_side_length 
  (A B C D E F : ℝ × ℝ) -- Points in 2D space
  (AB BC AC : ℝ) -- Sides of triangle ABC
  (DE EF : ℝ) -- Known sides of triangle DEF
  (angle_BAC angle_EDF : ℝ) -- Angles in radians
  (h_AB : dist A B = 8)
  (h_BC : dist B C = 18)
  (h_AC : dist A C = 12)
  (h_DE : dist D E = 4)
  (h_EF : dist E F = 9)
  (h_angle_BAC : angle_BAC = 2 * π / 3) -- 120° in radians
  (h_angle_EDF : angle_EDF = 2 * π / 3) -- 120° in radians
  : dist D F = 6 := by
  sorry


end similar_triangles_side_length_l1651_165184


namespace odd_number_2009_group_l1651_165125

/-- The cumulative sum of odd numbers up to the n-th group -/
def cumulative_sum (n : ℕ) : ℕ := n^2

/-- The size of the n-th group -/
def group_size (n : ℕ) : ℕ := 2*n - 1

/-- The theorem stating that 2009 belongs to the 32nd group -/
theorem odd_number_2009_group : 
  (cumulative_sum 31 < 2009) ∧ (2009 ≤ cumulative_sum 32) := by sorry

end odd_number_2009_group_l1651_165125


namespace triangle_perimeter_l1651_165108

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 6*x + 8 = 0 →
  x + 3 > 6 ∧ x + 6 > 3 ∧ 3 + 6 > x →
  x + 3 + 6 = 13 := by
sorry

end triangle_perimeter_l1651_165108


namespace swimmer_speed_in_still_water_l1651_165163

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  swimmer : ℝ
  stream : ℝ

/-- Calculates the effective speed of the swimmer given the direction. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water
  (s : SwimmerSpeeds)
  (h_downstream : effectiveSpeed s true * 4 = 32)
  (h_upstream : effectiveSpeed s false * 4 = 24) :
  s.swimmer = 7 := by
  sorry

end swimmer_speed_in_still_water_l1651_165163


namespace unique_solution_l1651_165153

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_solution :
  ∃! x : ℕ, digit_product x = x^2 - 10*x - 22 :=
by
  -- Proof goes here
  sorry

end unique_solution_l1651_165153


namespace final_number_is_172_l1651_165127

/-- Represents the state of the board at any given time -/
structure BoardState where
  numbers : List Nat
  deriving Repr

/-- The operation of erasing two numbers and replacing them with their sum minus 1 -/
def boardOperation (state : BoardState) (i j : Nat) : BoardState :=
  { numbers := 
      (state.numbers.removeNth i).removeNth j ++ 
      [state.numbers[i]! + state.numbers[j]! - 1] }

/-- The invariant of the board state -/
def boardInvariant (state : BoardState) : Int :=
  state.numbers.sum - state.numbers.length

/-- Initial board state with numbers 1 to 20 -/
def initialBoard : BoardState :=
  { numbers := List.range 20 |>.map (· + 1) }

/-- Theorem stating that after 19 operations, the final number on the board is 172 -/
theorem final_number_is_172 : 
  ∃ (operations : List (Nat × Nat)),
    operations.length = 19 ∧
    (operations.foldl 
      (fun state (i, j) => boardOperation state i j) 
      initialBoard).numbers = [172] := by
  sorry

end final_number_is_172_l1651_165127


namespace right_triangle_area_l1651_165188

/-- Given a right triangle ABC with ∠C = 90°, a + b = 14 cm, and c = 10 cm, 
    the area of the triangle is 24 cm². -/
theorem right_triangle_area (a b c : ℝ) : 
  a + b = 14 → c = 10 → a^2 + b^2 = c^2 → (1/2) * a * b = 24 := by
  sorry

end right_triangle_area_l1651_165188


namespace intersection_of_A_and_B_l1651_165155

def A : Set ℝ := {x | |x - 1| ≤ 1}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end intersection_of_A_and_B_l1651_165155


namespace petri_dishes_count_l1651_165198

-- Define the total number of germs
def total_germs : ℝ := 0.037 * 10^5

-- Define the number of germs per dish
def germs_per_dish : ℕ := 25

-- Define the number of petri dishes
def num_petri_dishes : ℕ := 148

-- Theorem statement
theorem petri_dishes_count :
  (total_germs / germs_per_dish : ℝ) = num_petri_dishes := by
  sorry

end petri_dishes_count_l1651_165198


namespace colored_paper_purchase_l1651_165157

theorem colored_paper_purchase (total_money : ℝ) (pencil_cost : ℝ) (paper_cost : ℝ) (pencils_bought : ℕ) :
  total_money = 10 →
  pencil_cost = 1.2 →
  paper_cost = 0.2 →
  pencils_bought = 5 →
  (total_money - pencil_cost * (pencils_bought : ℝ)) / paper_cost = 20 :=
by sorry

end colored_paper_purchase_l1651_165157


namespace sequence_equality_l1651_165139

theorem sequence_equality (C : ℝ) (a : ℕ → ℝ) 
  (hC : C > 1)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2)
  (h3 : ∀ m n : ℕ, m > 0 ∧ n > 0 → a (m * n) = a m * a n)
  (h4 : ∀ m n : ℕ, m > 0 ∧ n > 0 → a (m + n) ≤ C * (a m + a n))
  : ∀ n : ℕ, n > 0 → a n = n := by
  sorry

end sequence_equality_l1651_165139


namespace arithmetic_sequence_sixth_term_l1651_165192

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a1 : a 1 = 5)
  (h_a5 : a 5 = 1) :
  a 6 = 0 := by
  sorry

end arithmetic_sequence_sixth_term_l1651_165192


namespace gain_percent_calculation_l1651_165134

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 28 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = (11 / 14) * 100 := by
  sorry

end gain_percent_calculation_l1651_165134


namespace equation_solutions_count_l1651_165166

theorem equation_solutions_count :
  ∃! (solutions : Finset ℝ),
    Finset.card solutions = 8 ∧
    ∀ θ ∈ solutions,
      0 < θ ∧ θ ≤ 2 * Real.pi ∧
      2 - 4 * Real.sin θ + 6 * Real.cos (2 * θ) + Real.sin (3 * θ) = 0 ∧
    ∀ θ : ℝ,
      0 < θ ∧ θ ≤ 2 * Real.pi ∧
      2 - 4 * Real.sin θ + 6 * Real.cos (2 * θ) + Real.sin (3 * θ) = 0 →
      θ ∈ solutions :=
by sorry

end equation_solutions_count_l1651_165166


namespace area_transformation_l1651_165122

-- Define a function representing the area between a curve and the x-axis
noncomputable def area_between_curve_and_xaxis (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem area_transformation (f : ℝ → ℝ) (h : area_between_curve_and_xaxis f = 12) :
  area_between_curve_and_xaxis (λ x => 4 * f (x + 3)) = 48 :=
by sorry

end area_transformation_l1651_165122


namespace unique_k_with_prime_roots_l1651_165161

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The quadratic equation x^2 - 75x + k = 0 has two prime roots -/
def hasPrimeRoots (k : ℤ) : Prop :=
  ∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 75 ∧ p * q = k

/-- There is exactly one integer k such that x^2 - 75x + k = 0 has two prime roots -/
theorem unique_k_with_prime_roots :
  ∃! (k : ℤ), hasPrimeRoots k :=
sorry

end unique_k_with_prime_roots_l1651_165161


namespace prob_same_color_l1651_165109

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of black balls in the bag -/
def black_balls : ℕ := 4

/-- The number of red balls in the bag -/
def red_balls : ℕ := 5

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls + red_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- The probability of drawing at least two balls of the same color -/
theorem prob_same_color : 
  (1 : ℚ) - (white_balls * black_balls * red_balls : ℚ) / (total_balls * (total_balls - 1) * (total_balls - 2) / 6) = 8/11 := by
  sorry

end prob_same_color_l1651_165109


namespace rectangle_perimeter_product_l1651_165159

theorem rectangle_perimeter_product (a b c d : ℝ) : 
  (a + b = 11 ∧ a + b + c = 19.5 ∧ c = d) ∨
  (a + c = 11 ∧ a + b + c = 19.5 ∧ b = d) ∨
  (b + c = 11 ∧ a + b + c = 19.5 ∧ a = d) →
  (2 * (a + b)) * (2 * (a + c)) * (2 * (b + c)) = 15400 := by
sorry

end rectangle_perimeter_product_l1651_165159


namespace shadow_height_calculation_l1651_165121

/-- Given a tree and a person casting shadows, calculate the person's height -/
theorem shadow_height_calculation (tree_height tree_shadow alex_shadow : ℚ) 
  (h1 : tree_height = 50)
  (h2 : tree_shadow = 25)
  (h3 : alex_shadow = 20 / 12) : -- Convert 20 inches to feet
  tree_height / tree_shadow * alex_shadow = 10 / 3 := by
  sorry

#check shadow_height_calculation

end shadow_height_calculation_l1651_165121


namespace fraction_zero_implies_x_one_l1651_165130

theorem fraction_zero_implies_x_one (x : ℝ) (h : (x - 1) / x = 0) : x = 1 := by
  sorry

end fraction_zero_implies_x_one_l1651_165130


namespace tenth_term_of_specific_sequence_l1651_165140

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℤ  -- First term
  d : ℤ   -- Common difference

/-- The nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.a₁ + (n - 1 : ℤ) * seq.d

theorem tenth_term_of_specific_sequence :
  ∃ (seq : ArithmeticSequence),
    seq.a₁ = 10 ∧
    nthTerm seq 2 = 7 ∧
    nthTerm seq 3 = 4 ∧
    nthTerm seq 10 = -17 := by
  sorry

end tenth_term_of_specific_sequence_l1651_165140


namespace perseverance_permutations_count_l1651_165179

/-- The number of letters in "PERSEVERANCE" -/
def word_length : ℕ := 11

/-- The number of occurrences of 'E' in "PERSEVERANCE" -/
def e_count : ℕ := 3

/-- The number of occurrences of 'R' in "PERSEVERANCE" -/
def r_count : ℕ := 2

/-- The number of unique permutations of the letters in "PERSEVERANCE" -/
def perseverance_permutations : ℕ := word_length.factorial / (e_count.factorial * r_count.factorial * r_count.factorial)

theorem perseverance_permutations_count :
  perseverance_permutations = 1663200 :=
by sorry

end perseverance_permutations_count_l1651_165179


namespace history_book_cost_l1651_165141

/-- Given the following conditions:
  - Total number of books is 90
  - Math books cost $4 each
  - Total price of all books is $396
  - Number of math books bought is 54
  Prove that the cost of a history book is $5 -/
theorem history_book_cost (total_books : ℕ) (math_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 90 →
  math_book_cost = 4 →
  total_price = 396 →
  math_books = 54 →
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 :=
by sorry

end history_book_cost_l1651_165141


namespace unique_recurrence_solution_l1651_165119

/-- A sequence of positive real numbers satisfying the given recurrence relation. -/
def RecurrenceSequence (X : ℕ → ℝ) : Prop :=
  (∀ n, X n > 0) ∧ 
  (∀ n, X (n + 2) = (1 / X (n + 1) + X n) / 2)

/-- The theorem stating that the only sequence satisfying the recurrence relation is the constant sequence of 1. -/
theorem unique_recurrence_solution (X : ℕ → ℝ) :
  RecurrenceSequence X → (∀ n, X n = 1) := by
  sorry

#check unique_recurrence_solution

end unique_recurrence_solution_l1651_165119


namespace expression_evaluation_l1651_165149

theorem expression_evaluation : -2^3 / (-2) + (-2)^2 * (-5) = -16 := by
  sorry

end expression_evaluation_l1651_165149


namespace other_solution_quadratic_l1651_165137

theorem other_solution_quadratic (h : 49 * (5/7)^2 - 88 * (5/7) + 40 = 0) :
  49 * (8/7)^2 - 88 * (8/7) + 40 = 0 := by
  sorry

end other_solution_quadratic_l1651_165137


namespace not_consecutive_odd_beautiful_l1651_165110

def IsBeautiful (g : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, g x = g (a - x)

theorem not_consecutive_odd_beautiful
  (g : ℤ → ℤ)
  (h1 : ∀ x : ℤ, g x ≠ x)
  : ¬∃ a : ℤ, IsBeautiful g a ∧ IsBeautiful g (a + 2) ∧ Odd a :=
by sorry

end not_consecutive_odd_beautiful_l1651_165110


namespace patrol_theorem_l1651_165171

/-- The number of streets patrolled by an officer in one hour -/
def streets_per_hour (streets : ℕ) (hours : ℕ) : ℚ := streets / hours

/-- The total number of streets patrolled by all officers in one hour -/
def total_streets_per_hour (rate_A rate_B rate_C : ℚ) : ℚ := rate_A + rate_B + rate_C

theorem patrol_theorem (a x b y c z : ℕ) 
  (h1 : streets_per_hour a x = 9/1)
  (h2 : streets_per_hour b y = 11/1)
  (h3 : streets_per_hour c z = 7/1) :
  total_streets_per_hour (streets_per_hour a x) (streets_per_hour b y) (streets_per_hour c z) = 27 := by
  sorry

end patrol_theorem_l1651_165171


namespace unique_solution_to_equation_l1651_165174

theorem unique_solution_to_equation :
  ∃! (x y z : ℝ), 2*x^4 + 2*y^4 - 4*x^3*y + 6*x^2*y^2 - 4*x*y^3 + 7*y^2 + 7*z^2 - 14*y*z - 70*y + 70*z + 175 = 0 ∧
                   x = 0 ∧ y = 0 ∧ z = -5 := by
  sorry

end unique_solution_to_equation_l1651_165174
