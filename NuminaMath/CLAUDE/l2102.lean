import Mathlib

namespace min_sum_m_n_min_sum_value_min_sum_achieved_l2102_210230

theorem min_sum_m_n (m n : ℕ+) (h : 98 * m = n ^ 3) : 
  ∀ (m' n' : ℕ+), 98 * m' = n' ^ 3 → m + n ≤ m' + n' :=
by
  sorry

theorem min_sum_value (m n : ℕ+) (h : 98 * m = n ^ 3) :
  m + n ≥ 42 :=
by
  sorry

theorem min_sum_achieved : 
  ∃ (m n : ℕ+), 98 * m = n ^ 3 ∧ m + n = 42 :=
by
  sorry

end min_sum_m_n_min_sum_value_min_sum_achieved_l2102_210230


namespace jeff_scores_mean_l2102_210250

def jeff_scores : List ℝ := [90, 93, 85, 97, 92, 88]

theorem jeff_scores_mean : 
  (jeff_scores.sum / jeff_scores.length : ℝ) = 90.8333 := by
  sorry

end jeff_scores_mean_l2102_210250


namespace subset_implies_a_value_l2102_210201

theorem subset_implies_a_value (A B : Set ℤ) (a : ℤ) :
  A = {0, 1} →
  B = {-1, 0, a + 3} →
  A ⊆ B →
  a = -2 := by sorry

end subset_implies_a_value_l2102_210201


namespace range_of_a_l2102_210218

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (h : DecreasingFunction f) :
  (∀ a : ℝ, f (3 * a) < f (-2 * a + 10)) →
  (∃ c : ℝ, c = 2 ∧ ∀ a : ℝ, a > c) :=
sorry

end range_of_a_l2102_210218


namespace pens_for_friends_l2102_210283

/-- The number of friends who will receive pens from Kendra and Tony -/
def friends_receiving_pens (kendra_packs tony_packs pens_per_pack pens_kept_each : ℕ) : ℕ :=
  (kendra_packs + tony_packs) * pens_per_pack - 2 * pens_kept_each

/-- Theorem stating that Kendra and Tony will give pens to 14 friends -/
theorem pens_for_friends : 
  friends_receiving_pens 4 2 3 2 = 14 := by
  sorry

#eval friends_receiving_pens 4 2 3 2

end pens_for_friends_l2102_210283


namespace fair_die_probabilities_l2102_210266

-- Define the sample space for a fair six-sided die
def Ω : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define event A: number of points ≥ 3
def A : Finset ℕ := {3, 4, 5, 6}

-- Define event B: number of points is odd
def B : Finset ℕ := {1, 3, 5}

-- Define the probability measure for a fair die
def P (S : Finset ℕ) : ℚ := (S ∩ Ω).card / Ω.card

-- Theorem statement
theorem fair_die_probabilities :
  P A = 2/3 ∧ P (A ∪ B) = 5/6 ∧ P (A ∩ B) = 1/3 := by
  sorry

end fair_die_probabilities_l2102_210266


namespace max_area_triangle_l2102_210204

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the properties
def isAcute (t : Triangle) : Prop := sorry

def isSimilar (t1 t2 : Triangle) : Prop := sorry

def circumscribes (t1 t2 : Triangle) : Prop := sorry

def area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem max_area_triangle 
  (A₀B₀C₀ : Triangle) 
  (A'B'C' : Triangle) 
  (h1 : isAcute A₀B₀C₀) 
  (h2 : isAcute A'B'C') :
  ∃ (A₁B₁C₁ : Triangle),
    isSimilar A₁B₁C₁ A'B'C' ∧ 
    circumscribes A₁B₁C₁ A₀B₀C₀ ∧
    ∀ (ABC : Triangle),
      isSimilar ABC A'B'C' → 
      circumscribes ABC A₀B₀C₀ → 
      area ABC ≤ area A₁B₁C₁ :=
sorry

end max_area_triangle_l2102_210204


namespace coconut_grove_problem_l2102_210270

/-- The number of trees in the coconut grove that yield 60 nuts per year -/
def trees_60 (x : ℝ) : ℝ := x + 3

/-- The number of trees in the coconut grove that yield 120 nuts per year -/
def trees_120 (x : ℝ) : ℝ := x

/-- The number of trees in the coconut grove that yield 180 nuts per year -/
def trees_180 (x : ℝ) : ℝ := x - 3

/-- The total number of trees in the coconut grove -/
def total_trees (x : ℝ) : ℝ := trees_60 x + trees_120 x + trees_180 x

/-- The total number of nuts produced by all trees in the coconut grove -/
def total_nuts (x : ℝ) : ℝ := 60 * trees_60 x + 120 * trees_120 x + 180 * trees_180 x

/-- The average yield per tree per year -/
def average_yield : ℝ := 100

theorem coconut_grove_problem :
  ∃ x : ℝ, total_nuts x = average_yield * total_trees x ∧ x = 6 :=
by sorry

end coconut_grove_problem_l2102_210270


namespace multiply_72514_9999_l2102_210259

theorem multiply_72514_9999 : 72514 * 9999 = 725067486 := by
  sorry

end multiply_72514_9999_l2102_210259


namespace miraflores_can_win_l2102_210246

/-- Represents a voting system with multiple tiers --/
structure VotingSystem :=
  (total_voters : ℕ)
  (supporter_percentage : ℚ)
  (min_group_size : ℕ)
  (max_group_size : ℕ)

/-- Checks if a candidate can win in the given voting system --/
def can_win (vs : VotingSystem) : Prop :=
  ∃ (grouping : ℕ → ℕ),
    (∀ n, vs.min_group_size ≤ grouping n ∧ grouping n ≤ vs.max_group_size) ∧
    (∃ (final_group : ℕ), 
      final_group > 1 ∧
      final_group ≤ vs.total_voters ∧
      (vs.total_voters * vs.supporter_percentage).num * 2 > 
        (vs.total_voters * vs.supporter_percentage).den * final_group)

/-- The main theorem --/
theorem miraflores_can_win :
  ∃ (vs : VotingSystem), 
    vs.total_voters = 20000000 ∧
    vs.supporter_percentage = 1/100 ∧
    vs.min_group_size = 2 ∧
    vs.max_group_size = 5 ∧
    can_win vs :=
  sorry

end miraflores_can_win_l2102_210246


namespace cos_18_deg_root_l2102_210217

theorem cos_18_deg_root : ∃ (p : ℝ → ℝ), (∀ x, p x = 16 * x^4 - 20 * x^2 + 5) ∧ p (Real.cos (18 * Real.pi / 180)) = 0 := by
  sorry

end cos_18_deg_root_l2102_210217


namespace watermelon_weights_sum_l2102_210205

/-- Watermelon weights problem -/
theorem watermelon_weights_sum : 
  -- Given conditions
  let michael_largest : ℝ := 12
  let clay_first : ℝ := 1.5 * michael_largest
  let john_first : ℝ := clay_first / 2
  let emily : ℝ := 0.75 * john_first
  let sophie_first : ℝ := emily + 3
  let michael_smallest : ℝ := michael_largest * 0.7
  let clay_second : ℝ := clay_first * 1.2
  let john_second : ℝ := (john_first + emily) / 2
  let sophie_second : ℝ := 3 * (clay_second - clay_first)
  -- Theorem statement
  michael_largest + michael_smallest + clay_first + clay_second + 
  john_first + john_second + emily + sophie_first + sophie_second = 104.175 := by
  sorry

end watermelon_weights_sum_l2102_210205


namespace mike_unbroken_seashells_l2102_210203

/-- The number of unbroken seashells Mike found -/
def unbroken_seashells (total : ℕ) (broken : ℕ) : ℕ :=
  total - broken

/-- Theorem stating that Mike found 2 unbroken seashells -/
theorem mike_unbroken_seashells :
  unbroken_seashells 6 4 = 2 := by
  sorry

end mike_unbroken_seashells_l2102_210203


namespace product_of_xy_l2102_210276

theorem product_of_xy (x y : ℝ) (h : 3 * (2 * x * y + 9) = 51) : x * y = 4 := by
  sorry

end product_of_xy_l2102_210276


namespace leaves_first_hour_is_seven_l2102_210231

/-- The number of leaves that fell in the first hour -/
def leaves_first_hour : ℕ := 7

/-- The total number of hours -/
def total_hours : ℕ := 3

/-- The rate of leaves falling per hour in the second and third hour -/
def rate_later_hours : ℕ := 4

/-- The average number of leaves that fell per hour over the entire period -/
def average_leaves_per_hour : ℕ := 5

/-- Theorem stating that the number of leaves that fell in the first hour is 7 -/
theorem leaves_first_hour_is_seven :
  leaves_first_hour = 
    total_hours * average_leaves_per_hour - rate_later_hours * (total_hours - 1) :=
by sorry

end leaves_first_hour_is_seven_l2102_210231


namespace shooting_probability_theorem_l2102_210260

def shooting_probability (accuracy_A accuracy_B : ℝ) : ℝ × ℝ :=
  let prob_both_two := accuracy_A * accuracy_A * accuracy_B * accuracy_B
  let prob_at_least_three := prob_both_two + 
    accuracy_A * accuracy_A * accuracy_B * (1 - accuracy_B) +
    accuracy_A * (1 - accuracy_A) * accuracy_B * accuracy_B
  (prob_both_two, prob_at_least_three)

theorem shooting_probability_theorem :
  shooting_probability 0.4 0.6 = (0.0576, 0.1824) := by
  sorry

end shooting_probability_theorem_l2102_210260


namespace domain_of_f_is_all_reals_l2102_210275

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (5 - 2 * x) ^ (1/3)

-- Theorem stating that the domain of f is all real numbers
theorem domain_of_f_is_all_reals :
  ∀ x : ℝ, ∃ y : ℝ, f x = y :=
by
  sorry

end domain_of_f_is_all_reals_l2102_210275


namespace yard_length_22_trees_l2102_210267

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℕ) : ℕ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 22 trees planted at equal distances,
    with one tree at each end and 21 metres between consecutive trees, is 441 metres. -/
theorem yard_length_22_trees : yard_length 22 21 = 441 := by
  sorry

end yard_length_22_trees_l2102_210267


namespace octahedron_ant_path_probability_l2102_210210

/-- Represents a vertex in the octahedron --/
inductive Vertex
| Top
| Bottom
| Middle1
| Middle2
| Middle3
| Middle4

/-- Represents an octahedron --/
structure Octahedron where
  vertices : List Vertex
  edges : List (Vertex × Vertex)
  is_regular : Bool

/-- Represents the ant's path --/
structure AntPath where
  start : Vertex
  a : Vertex
  b : Vertex
  c : Vertex

/-- Function to check if a vertex is in the middle ring --/
def is_middle_ring (v : Vertex) : Bool :=
  match v with
  | Vertex.Middle1 | Vertex.Middle2 | Vertex.Middle3 | Vertex.Middle4 => true
  | _ => false

/-- Function to get adjacent vertices --/
def get_adjacent_vertices (o : Octahedron) (v : Vertex) : List Vertex :=
  sorry

/-- Function to calculate the probability of returning to the start --/
def return_probability (o : Octahedron) (path : AntPath) : Rat :=
  sorry

theorem octahedron_ant_path_probability (o : Octahedron) (path : AntPath) :
  o.is_regular = true →
  is_middle_ring path.start = true →
  path.a ∈ get_adjacent_vertices o path.start →
  path.b ∈ get_adjacent_vertices o path.a →
  path.c ∈ get_adjacent_vertices o path.b →
  return_probability o path = 1 / 16 :=
sorry

end octahedron_ant_path_probability_l2102_210210


namespace correct_systematic_sample_l2102_210211

/-- Represents a systematic sample of students -/
structure SystematicSample where
  totalStudents : Nat
  sampleSize : Nat
  sampleNumbers : List Nat

/-- Checks if a given sample is a valid systematic sample -/
def isValidSystematicSample (sample : SystematicSample) : Prop :=
  sample.totalStudents = 20 ∧
  sample.sampleSize = 4 ∧
  sample.sampleNumbers = [5, 10, 15, 20]

/-- Theorem stating that the given sample is the correct systematic sample -/
theorem correct_systematic_sample :
  ∃ (sample : SystematicSample), isValidSystematicSample sample :=
sorry

end correct_systematic_sample_l2102_210211


namespace spring_equation_l2102_210213

theorem spring_equation (RI G SP T M N : ℤ) (L : ℚ) : 
  RI + G + SP = 50 ∧
  RI + T + M = 63 ∧
  G + T + SP = 25 ∧
  SP + M = 13 ∧
  M + RI = 48 ∧
  N = 1 →
  L * M * T + SP * RI * N * G = 2023 →
  L = 341 / 40 := by
sorry

end spring_equation_l2102_210213


namespace tan_ratio_equals_two_l2102_210288

theorem tan_ratio_equals_two (α β γ : ℝ) 
  (h : Real.sin (2 * (α + γ)) = 3 * Real.sin (2 * β)) : 
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = 2 := by
  sorry

end tan_ratio_equals_two_l2102_210288


namespace expression_evaluation_l2102_210281

theorem expression_evaluation : 3^2 / 3 - 4 * 2 + 2^3 = 3 := by
  sorry

end expression_evaluation_l2102_210281


namespace largest_n_for_unique_k_l2102_210223

theorem largest_n_for_unique_k : 
  ∀ n : ℕ+, n ≤ 72 ↔ 
    ∃! k : ℤ, (9:ℚ)/17 < (n:ℚ)/(n + k) ∧ (n:ℚ)/(n + k) < 8/15 :=
by sorry

end largest_n_for_unique_k_l2102_210223


namespace xy_length_l2102_210291

/-- Represents a trapezoid WXYZ with specific properties -/
structure Trapezoid where
  -- W, X, Y, Z are points in the plane
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  -- WX is parallel to ZY
  wx_parallel_zy : (X.1 - W.1) * (Y.2 - Z.2) = (X.2 - W.2) * (Y.1 - Z.1)
  -- WY is perpendicular to ZY
  wy_perp_zy : (Y.1 - W.1) * (Y.1 - Z.1) + (Y.2 - W.2) * (Y.2 - Z.2) = 0
  -- YZ = 15
  yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 15
  -- tan Z = 2
  tan_z : (W.2 - Z.2) / (Y.1 - Z.1) = 2
  -- tan X = 3/2
  tan_x : (W.2 - Y.2) / (X.1 - W.1) = 3/2

/-- The length of XY in the trapezoid is 10√13 -/
theorem xy_length (t : Trapezoid) : 
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 10 * Real.sqrt 13 := by
  sorry

end xy_length_l2102_210291


namespace negation_equivalence_l2102_210297

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end negation_equivalence_l2102_210297


namespace number_of_valid_passwords_l2102_210292

/-- The number of digits in the password -/
def password_length : ℕ := 5

/-- The range of possible digits -/
def digit_range : ℕ := 10

/-- The number of passwords starting with the forbidden sequence -/
def forbidden_passwords : ℕ := 10

/-- Calculates the number of valid passwords -/
def valid_passwords : ℕ := digit_range ^ password_length - forbidden_passwords

/-- Theorem stating the number of valid passwords -/
theorem number_of_valid_passwords : valid_passwords = 99990 := by
  sorry

end number_of_valid_passwords_l2102_210292


namespace compound_interest_rate_l2102_210229

theorem compound_interest_rate (principal : ℝ) (final_amount : ℝ) (time : ℝ) 
  (h1 : principal = 400)
  (h2 : final_amount = 441)
  (h3 : time = 2) :
  ∃ (rate : ℝ), 
    final_amount = principal * (1 + rate) ^ time ∧ 
    rate = 0.05 := by
  sorry

end compound_interest_rate_l2102_210229


namespace intersection_of_A_and_B_l2102_210269

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {3} := by
  sorry

end intersection_of_A_and_B_l2102_210269


namespace dog_group_arrangements_count_l2102_210200

/-- The number of ways to divide 12 dogs into three groups -/
def dog_group_arrangements : ℕ :=
  let total_dogs : ℕ := 12
  let group_1_size : ℕ := 4
  let group_2_size : ℕ := 6
  let group_3_size : ℕ := 2
  let dogs_to_distribute : ℕ := total_dogs - 2  -- Fluffy and Nipper are pre-assigned
  let remaining_group_1_size : ℕ := group_1_size - 1  -- Fluffy is already in group 1
  let remaining_group_2_size : ℕ := group_2_size - 1  -- Nipper is already in group 2
  (Nat.choose dogs_to_distribute remaining_group_1_size) * 
  (Nat.choose (dogs_to_distribute - remaining_group_1_size) remaining_group_2_size)

/-- Theorem stating the number of ways to arrange the dogs -/
theorem dog_group_arrangements_count : dog_group_arrangements = 2520 := by
  sorry

end dog_group_arrangements_count_l2102_210200


namespace kelly_initial_games_l2102_210202

/-- The number of games Kelly gave away -/
def games_given_away : ℕ := 91

/-- The number of games Kelly has left -/
def games_left : ℕ := 92

/-- The initial number of games Kelly had -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_initial_games : initial_games = 183 := by
  sorry

end kelly_initial_games_l2102_210202


namespace pyramid_cross_sections_l2102_210287

/-- Theorem about cross-sectional areas in a pyramid --/
theorem pyramid_cross_sections
  (S : ℝ) -- Base area of the pyramid
  (S₁ S₂ S₃ : ℝ) -- Cross-sectional areas
  (h₁ : S₁ = S / 4) -- S₁ bisects lateral edges
  (h₂ : S₂ = S / 2) -- S₂ bisects lateral surface area
  (h₃ : S₃ = S / (4 ^ (1/3))) -- S₃ bisects volume
  : S₁ < S₂ ∧ S₂ < S₃ := by
  sorry

end pyramid_cross_sections_l2102_210287


namespace combined_cost_price_theorem_l2102_210255

/-- Calculates the cost price of a stock given its face value, discount/premium rate, and brokerage rate -/
def stockCostPrice (faceValue : ℝ) (discountRate : ℝ) (brokerageRate : ℝ) : ℝ :=
  let purchasePrice := faceValue * (1 + discountRate)
  let brokerageFee := purchasePrice * brokerageRate
  purchasePrice + brokerageFee

/-- Theorem stating the combined cost price of two stocks -/
theorem combined_cost_price_theorem :
  let stockA := stockCostPrice 100 (-0.02) 0.002
  let stockB := stockCostPrice 100 0.015 0.002
  stockA + stockB = 199.899 := by
  sorry


end combined_cost_price_theorem_l2102_210255


namespace jogger_ahead_of_train_l2102_210294

/-- Calculates the distance a jogger is ahead of a train given their speeds and the time for the train to pass the jogger. -/
def jogger_distance_ahead (jogger_speed : Real) (train_speed : Real) (train_length : Real) (passing_time : Real) : Real :=
  (train_speed - jogger_speed) * passing_time - train_length

/-- Theorem stating the distance a jogger is ahead of a train under specific conditions. -/
theorem jogger_ahead_of_train (jogger_speed : Real) (train_speed : Real) (train_length : Real) (passing_time : Real)
  (h1 : jogger_speed = 9 * 1000 / 3600)
  (h2 : train_speed = 45 * 1000 / 3600)
  (h3 : train_length = 120)
  (h4 : passing_time = 40.00000000000001) :
  ∃ ε > 0, |jogger_distance_ahead jogger_speed train_speed train_length passing_time - 280| < ε :=
by sorry

end jogger_ahead_of_train_l2102_210294


namespace soda_cost_for_reunion_l2102_210243

/-- Calculates the cost per family member for soda at a family reunion --/
def soda_cost_per_family_member (attendees : ℕ) (cans_per_person : ℕ) (cans_per_box : ℕ) (cost_per_box : ℚ) (family_members : ℕ) : ℚ :=
  let total_cans := attendees * cans_per_person
  let boxes_needed := (total_cans + cans_per_box - 1) / cans_per_box  -- Ceiling division
  let total_cost := boxes_needed * cost_per_box
  total_cost / family_members

/-- Theorem stating the cost per family member for the given scenario --/
theorem soda_cost_for_reunion : 
  soda_cost_per_family_member (5 * 12) 2 10 2 6 = 4 := by
  sorry

end soda_cost_for_reunion_l2102_210243


namespace smallest_solution_abs_equation_l2102_210216

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 4 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 4 → x ≤ y :=
by sorry

end smallest_solution_abs_equation_l2102_210216


namespace largest_four_digit_mod_5_3_l2102_210261

theorem largest_four_digit_mod_5_3 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 5 = 3 → n ≤ 9998 :=
by sorry

end largest_four_digit_mod_5_3_l2102_210261


namespace unique_polynomial_with_integer_root_l2102_210214

theorem unique_polynomial_with_integer_root :
  ∃! (a : ℕ+), ∃ (x : ℤ), x^2 - (a : ℤ) * x + (a : ℤ) = 0 :=
sorry

end unique_polynomial_with_integer_root_l2102_210214


namespace yellow_chip_value_l2102_210295

theorem yellow_chip_value :
  ∀ (y : ℕ) (b : ℕ),
    y > 0 →
    b > 0 →
    y^4 * (4 * b)^b * (5 * b)^b = 16000 →
    y = 2 :=
by
  sorry

end yellow_chip_value_l2102_210295


namespace subtract_from_percentage_l2102_210298

theorem subtract_from_percentage (n : ℝ) : n = 300 → 0.3 * n - 70 = 20 := by
  sorry

end subtract_from_percentage_l2102_210298


namespace zilla_savings_theorem_l2102_210224

/-- Represents Zilla's monthly financial breakdown -/
structure ZillaFinances where
  earnings : ℝ
  rent_percent : ℝ
  groceries_percent : ℝ
  entertainment_percent : ℝ
  transportation_percent : ℝ
  rent_amount : ℝ

/-- Calculates Zilla's savings based on her financial breakdown -/
def calculate_savings (z : ZillaFinances) : ℝ :=
  z.earnings * (1 - z.rent_percent - z.groceries_percent - z.entertainment_percent - z.transportation_percent)

/-- Theorem stating that Zilla's savings are $589 given her financial breakdown -/
theorem zilla_savings_theorem (z : ZillaFinances) 
  (h1 : z.rent_percent = 0.07)
  (h2 : z.groceries_percent = 0.3)
  (h3 : z.entertainment_percent = 0.2)
  (h4 : z.transportation_percent = 0.12)
  (h5 : z.rent_amount = 133)
  (h6 : z.earnings * z.rent_percent = z.rent_amount) :
  calculate_savings z = 589 := by
  sorry


end zilla_savings_theorem_l2102_210224


namespace hike_attendance_l2102_210225

/-- The number of cars used for the hike -/
def num_cars : ℕ := 3

/-- The number of taxis used for the hike -/
def num_taxis : ℕ := 6

/-- The number of vans used for the hike -/
def num_vans : ℕ := 2

/-- The number of people in each car -/
def people_per_car : ℕ := 4

/-- The number of people in each taxi -/
def people_per_taxi : ℕ := 6

/-- The number of people in each van -/
def people_per_van : ℕ := 5

/-- The total number of people who went on the hike -/
def total_people : ℕ := num_cars * people_per_car + num_taxis * people_per_taxi + num_vans * people_per_van

theorem hike_attendance : total_people = 58 := by
  sorry

end hike_attendance_l2102_210225


namespace population_ratio_l2102_210251

-- Define populations as real numbers
variable (P_A P_B P_C P_D P_E P_F : ℝ)

-- Define the relationships between city populations
def population_relations : Prop :=
  (P_A = 8 * P_B) ∧
  (P_B = 5 * P_C) ∧
  (P_D = 3 * P_C) ∧
  (P_D = P_E / 2) ∧
  (P_F = P_A / 4)

-- Theorem to prove
theorem population_ratio (h : population_relations P_A P_B P_C P_D P_E P_F) :
  P_E / P_B = 6 / 5 := by
  sorry

end population_ratio_l2102_210251


namespace stating_selling_price_is_43_l2102_210258

/-- Represents the selling price of an article when the loss is equal to the profit. -/
def selling_price_equal_loss_profit (cost_price : ℕ) (profit_price : ℕ) : ℕ :=
  cost_price * 2 - profit_price

/-- 
Theorem stating that the selling price of an article is 43 when the loss is equal to the profit,
given that the cost price is 50 and the profit obtained by selling for 57 is the same as the loss
obtained by selling for the unknown price.
-/
theorem selling_price_is_43 :
  selling_price_equal_loss_profit 50 57 = 43 := by
  sorry

#eval selling_price_equal_loss_profit 50 57

end stating_selling_price_is_43_l2102_210258


namespace emily_candy_duration_l2102_210279

/-- The number of days Emily's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  (neighbors_candy + sister_candy) / daily_consumption

/-- Proof that Emily's candy will last for 2 days -/
theorem emily_candy_duration :
  candy_duration 5 13 9 = 2 := by
  sorry

end emily_candy_duration_l2102_210279


namespace cauliflower_increase_l2102_210248

/-- Represents a square garden for growing cauliflowers -/
structure CauliflowerGarden where
  side : ℕ

/-- Calculates the number of cauliflowers in a garden -/
def cauliflowers (garden : CauliflowerGarden) : ℕ := garden.side * garden.side

/-- Theorem: If a square garden's cauliflower output increases by 401 while
    maintaining a square shape, the new total is 40,401 cauliflowers -/
theorem cauliflower_increase (old_garden new_garden : CauliflowerGarden) :
  cauliflowers new_garden - cauliflowers old_garden = 401 →
  cauliflowers new_garden = 40401 := by
  sorry


end cauliflower_increase_l2102_210248


namespace greatest_multiple_of_30_under_1000_l2102_210241

theorem greatest_multiple_of_30_under_1000 : 
  ∀ n : ℕ, n * 30 < 1000 → n * 30 ≤ 990 :=
by
  sorry

end greatest_multiple_of_30_under_1000_l2102_210241


namespace average_age_combined_l2102_210254

theorem average_age_combined (num_students : ℕ) (num_guardians : ℕ) 
  (avg_age_students : ℚ) (avg_age_guardians : ℚ) :
  num_students = 40 →
  num_guardians = 60 →
  avg_age_students = 10 →
  avg_age_guardians = 35 →
  (num_students * avg_age_students + num_guardians * avg_age_guardians) / (num_students + num_guardians) = 25 :=
by
  sorry

end average_age_combined_l2102_210254


namespace tennis_tournament_theorem_l2102_210257

-- Define the number of women players
def n : ℕ := sorry

-- Define the total number of players
def total_players : ℕ := n + 3 * n

-- Define the total number of matches
def total_matches : ℕ := (total_players * (total_players - 1)) / 2

-- Define the number of matches won by women
def women_wins : ℕ := sorry

-- Define the number of matches won by men
def men_wins : ℕ := sorry

-- Theorem stating the conditions and the result to be proved
theorem tennis_tournament_theorem :
  -- Each player plays with every other player
  (∀ p : ℕ, p < total_players → (total_players - 1) * p = total_matches * 2) →
  -- No ties
  (women_wins + men_wins = total_matches) →
  -- Ratio of women's wins to men's wins is 3/2
  (3 * men_wins = 2 * women_wins) →
  -- n equals 4
  n = 4 := by sorry

end tennis_tournament_theorem_l2102_210257


namespace pumpkin_contest_theorem_l2102_210252

def pumpkin_contest (brad jessica betty carlos emily dave : ℝ) : Prop :=
  brad = 54 ∧
  jessica = brad / 2 ∧
  betty = 4 * jessica ∧
  carlos = 2.5 * (brad + jessica) ∧
  emily = 1.5 * (betty - brad) ∧
  dave = (jessica + betty) / 2 + 20 ∧
  max brad (max jessica (max betty (max carlos (max emily dave)))) -
  min brad (min jessica (min betty (min carlos (min emily dave)))) = 175.5

theorem pumpkin_contest_theorem :
  ∃ brad jessica betty carlos emily dave : ℝ,
    pumpkin_contest brad jessica betty carlos emily dave :=
by
  sorry

end pumpkin_contest_theorem_l2102_210252


namespace g_five_equals_one_l2102_210247

theorem g_five_equals_one (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0) :
  g 5 = 1 := by
sorry

end g_five_equals_one_l2102_210247


namespace triangle_area_change_l2102_210209

theorem triangle_area_change (base height : ℝ) (base_new height_new : ℝ) :
  base_new = base * 1.1 →
  height_new = height * 0.95 →
  (base_new * height_new) / (base * height) - 1 = 0.045 := by
sorry

end triangle_area_change_l2102_210209


namespace smaller_square_area_percentage_l2102_210239

/-- Given a circle with radius 2√2 and a square inscribed in it with side length 4,
    prove that a smaller square with one side coinciding with the larger square
    and two vertices on the circle has an area that is 4% of the larger square's area. -/
theorem smaller_square_area_percentage (r : ℝ) (s : ℝ) (x : ℝ) :
  r = 2 * Real.sqrt 2 →
  s = 4 →
  (2 + 2*x)^2 + x^2 = r^2 →
  (2*x)^2 / s^2 = 0.04 := by
  sorry

end smaller_square_area_percentage_l2102_210239


namespace max_value_sum_of_roots_l2102_210285

theorem max_value_sum_of_roots (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + 9 * c^2 = 1) :
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + 9 * z^2 = 1 ∧
    Real.sqrt x + Real.sqrt y + Real.sqrt 3 * z > Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c) ∨
  Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c = Real.sqrt (21 / 3) := by
  sorry

end max_value_sum_of_roots_l2102_210285


namespace equal_intercept_line_equation_l2102_210234

/-- A line passing through a point with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The point the line passes through
  point : ℝ × ℝ
  -- The equation of the line in the form ax + by = c
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through the given point
  point_on_line : a * point.1 + b * point.2 = c
  -- The line has equal intercepts on both axes
  equal_intercepts : c / a = c / b

/-- The theorem stating the equation of the line -/
theorem equal_intercept_line_equation :
  ∀ (l : EqualInterceptLine),
  l.point = (3, 2) →
  (l.a = 2 ∧ l.b = -3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = 5) :=
by sorry

end equal_intercept_line_equation_l2102_210234


namespace inheritance_tax_problem_l2102_210299

theorem inheritance_tax_problem (x : ℝ) : 
  let federal_tax := 0.25 * x
  let after_federal := x - federal_tax
  let state_tax := 0.15 * after_federal
  let after_state := after_federal - state_tax
  let luxury_tax := 0.05 * after_state
  let total_tax := federal_tax + state_tax + luxury_tax
  total_tax = 20000 → x = 50700 := by
sorry

end inheritance_tax_problem_l2102_210299


namespace monster_family_kids_l2102_210215

/-- The number of kids in the monster family -/
def num_kids : ℕ := 3

/-- The number of eyes the mom has -/
def mom_eyes : ℕ := 1

/-- The number of eyes the dad has -/
def dad_eyes : ℕ := 3

/-- The number of eyes each kid has -/
def kid_eyes : ℕ := 4

/-- The total number of eyes in the family -/
def total_eyes : ℕ := 16

theorem monster_family_kids :
  mom_eyes + dad_eyes + num_kids * kid_eyes = total_eyes :=
by sorry

end monster_family_kids_l2102_210215


namespace container_capacity_solution_l2102_210235

def container_capacity (replace_volume : ℝ) (num_replacements : ℕ) 
  (final_ratio_original : ℝ) (final_ratio_new : ℝ) : ℝ → Prop :=
  λ C => (C - replace_volume)^num_replacements / C^(num_replacements - 1) = 
    (final_ratio_original / (final_ratio_original + final_ratio_new)) * C

theorem container_capacity_solution :
  ∃ C : ℝ, C > 0 ∧ container_capacity 15 4 81 256 C ∧ 
    C = 15 / (1 - 3 / (337 : ℝ)^(1/4)) :=
by sorry

end container_capacity_solution_l2102_210235


namespace sum_is_composite_l2102_210228

theorem sum_is_composite (a b c d : ℕ+) 
  (h : a^2 - a*b + b^2 = c^2 - c*d + d^2) : 
  ∃ (k m : ℕ+), k > 1 ∧ m > 1 ∧ a + b + c + d = k * m :=
sorry

end sum_is_composite_l2102_210228


namespace star_composition_l2102_210286

-- Define the star operations
def star_right (y : ℝ) : ℝ := 9 - y
def star_left (y : ℝ) : ℝ := y - 9

-- State the theorem
theorem star_composition : star_left (star_right 15) = -15 := by
  sorry

end star_composition_l2102_210286


namespace construction_workers_l2102_210256

theorem construction_workers (initial_workers : ℕ) (initial_days : ℕ) (remaining_days : ℕ)
  (initial_work_fraction : ℚ) (h1 : initial_workers = 60)
  (h2 : initial_days = 18) (h3 : remaining_days = 12)
  (h4 : initial_work_fraction = 1/3) :
  ∃ (additional_workers : ℕ),
    additional_workers = 60 ∧
    (additional_workers + initial_workers : ℚ) * remaining_days * initial_work_fraction =
    (1 - initial_work_fraction) * initial_workers * initial_days :=
by sorry

end construction_workers_l2102_210256


namespace composite_function_equality_l2102_210249

/-- Given two functions f and g, prove that f(g(f(3))) = 332 -/
theorem composite_function_equality (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 4 * x + 4) 
  (hg : ∀ x, g x = 5 * x + 2) : 
  f (g (f 3)) = 332 := by
  sorry

end composite_function_equality_l2102_210249


namespace ones_digit_of_3_to_26_l2102_210240

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- The ones digit of 3^n for any natural number n -/
def onesDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case is unreachable, but Lean requires it for exhaustiveness

theorem ones_digit_of_3_to_26 : onesDigit (3^26) = 9 := by
  sorry

end ones_digit_of_3_to_26_l2102_210240


namespace total_nails_eq_252_l2102_210263

/-- The number of nails/claws/toes to be cut -/
def total_nails : ℕ :=
  let dogs := 4
  let parrots := 8
  let cats := 2
  let rabbits := 6
  let dog_nails := dogs * 4 * 4
  let parrot_claws := (parrots - 1) * 2 * 3 + 1 * 2 * 4
  let cat_toes := cats * (2 * 5 + 2 * 4)
  let rabbit_nails := rabbits * (2 * 5 + 3 + 4)
  dog_nails + parrot_claws + cat_toes + rabbit_nails

/-- Theorem stating that the total number of nails/claws/toes to be cut is 252 -/
theorem total_nails_eq_252 : total_nails = 252 := by
  sorry

end total_nails_eq_252_l2102_210263


namespace price_reduction_percentage_l2102_210271

theorem price_reduction_percentage (original_price current_price : ℝ) 
  (h1 : original_price = 3000)
  (h2 : current_price = 2400) :
  (original_price - current_price) / original_price = 0.2 := by
  sorry

end price_reduction_percentage_l2102_210271


namespace smallest_prime_with_digit_sum_22_l2102_210233

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

theorem smallest_prime_with_digit_sum_22 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 22 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 22 → p ≤ q :=
by sorry

end smallest_prime_with_digit_sum_22_l2102_210233


namespace prob_at_least_one_boy_one_girl_l2102_210245

/-- The probability of having at least one boy and one girl in a family of four children,
    given that the birth of a boy or a girl is equally likely. -/
theorem prob_at_least_one_boy_one_girl : 
  let p_boy : ℚ := 1/2  -- Probability of having a boy
  let p_girl : ℚ := 1/2  -- Probability of having a girl
  let n : ℕ := 4  -- Number of children
  1 - (p_boy ^ n + p_girl ^ n) = 7/8 := by
sorry

end prob_at_least_one_boy_one_girl_l2102_210245


namespace circle_area_vs_circumference_probability_l2102_210265

theorem circle_area_vs_circumference_probability : 
  let die_roll : Set ℕ := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes : Set ℕ := {n ∈ die_roll | n > 1}
  let probability := (Finset.card favorable_outcomes.toFinset) / (Finset.card die_roll.toFinset)
  let area (r : ℝ) := π * r^2
  let circumference (r : ℝ) := 2 * π * r
  (∀ r ∈ die_roll, area r > (1/2) * circumference r ↔ r > 1) →
  probability = 5/6 := by
sorry

end circle_area_vs_circumference_probability_l2102_210265


namespace perpendicular_skew_lines_iff_plane_exists_l2102_210272

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew_lines (a b : Line3D) : Prop := sorry

/-- A line is perpendicular to another line if their direction vectors are orthogonal. -/
def line_perpendicular (a b : Line3D) : Prop := sorry

/-- A plane passes through a line if the line is contained in the plane. -/
def plane_passes_through_line (p : Plane3D) (l : Line3D) : Prop := sorry

/-- A plane is perpendicular to a line if the normal vector of the plane is parallel to the direction vector of the line. -/
def plane_perpendicular_to_line (p : Plane3D) (l : Line3D) : Prop := sorry

/-- Main theorem: For two skew lines, one line is perpendicular to the other if and only if
    there exists a plane passing through the first line and perpendicular to the second line. -/
theorem perpendicular_skew_lines_iff_plane_exists (a b : Line3D) 
  (h : are_skew_lines a b) : 
  line_perpendicular a b ↔ 
  ∃ (p : Plane3D), plane_passes_through_line p a ∧ plane_perpendicular_to_line p b := by
  sorry

end perpendicular_skew_lines_iff_plane_exists_l2102_210272


namespace graces_age_l2102_210207

/-- Grace's age problem -/
theorem graces_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ) : 
  mother_age = 80 →
  grandmother_age = 2 * mother_age →
  grace_age = (3 * grandmother_age) / 8 →
  grace_age = 60 := by
  sorry

end graces_age_l2102_210207


namespace new_concentration_after_replacement_l2102_210219

/-- Calculates the new concentration of a solution after partial replacement --/
def new_concentration (initial_conc : ℝ) (replacement_conc : ℝ) (fraction_replaced : ℝ) : ℝ :=
  (initial_conc * (1 - fraction_replaced) + replacement_conc * fraction_replaced)

/-- Theorem: New concentration after partial replacement --/
theorem new_concentration_after_replacement :
  new_concentration 0.4 0.25 (1/3) = 0.35 := by
  sorry

end new_concentration_after_replacement_l2102_210219


namespace pens_kept_each_l2102_210236

/-- Calculates the number of pens Kendra and Tony keep each after giving some away to friends. -/
theorem pens_kept_each (kendra_packs tony_packs pens_per_pack friends_given : ℕ) :
  kendra_packs = 4 →
  tony_packs = 2 →
  pens_per_pack = 3 →
  friends_given = 14 →
  let total_pens := (kendra_packs + tony_packs) * pens_per_pack
  let remaining_pens := total_pens - friends_given
  remaining_pens / 2 = 2 := by
sorry

end pens_kept_each_l2102_210236


namespace house_wall_nails_l2102_210284

/-- The number of large planks used for the house wall. -/
def large_planks : ℕ := 13

/-- The number of nails needed for each large plank. -/
def nails_per_plank : ℕ := 17

/-- The number of additional nails needed for smaller planks. -/
def additional_nails : ℕ := 8

/-- The total number of nails needed for the house wall. -/
def total_nails : ℕ := large_planks * nails_per_plank + additional_nails

theorem house_wall_nails : total_nails = 229 := by
  sorry

end house_wall_nails_l2102_210284


namespace inequality_proof_l2102_210282

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) := by
  sorry

end inequality_proof_l2102_210282


namespace function_property_l2102_210206

theorem function_property (f : ℝ → ℝ) (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y) 
  (h2 : f 8 = -3) : ∃ a : ℝ, a > 0 ∧ f a = 1/2 ∧ a = Real.sqrt 2 / 2 := by
  sorry

end function_property_l2102_210206


namespace sum_m_n_equals_negative_one_l2102_210253

theorem sum_m_n_equals_negative_one (m n : ℝ) 
  (h : Real.sqrt (m - 2) + (n + 3)^2 = 0) : m + n = -1 := by
  sorry

end sum_m_n_equals_negative_one_l2102_210253


namespace isosceles_triangle_properties_l2102_210273

-- Define the triangle and its properties
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (D : ℝ × ℝ), 
    -- AB = AC (isosceles)
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
    -- Line AB: 2x + y - 4 = 0
    2 * A.1 + A.2 - 4 = 0 ∧ 2 * B.1 + B.2 - 4 = 0 ∧
    -- Median AD: x - y + 1 = 0
    D.1 - D.2 + 1 = 0 ∧
    -- D is midpoint of BC
    D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧
    -- Point D: (4, 5)
    D = (4, 5)

-- Theorem statement
theorem isosceles_triangle_properties (A B C : ℝ × ℝ) 
  (h : Triangle A B C) : 
  -- Line BC: x + y - 9 = 0
  B.1 + B.2 - 9 = 0 ∧ C.1 + C.2 - 9 = 0 ∧
  -- Point B: (-5, 14)
  B = (-5, 14) ∧
  -- Point C: (13, -4)
  C = (13, -4) ∧
  -- Line AC: x + 2y - 5 = 0
  A.1 + 2 * A.2 - 5 = 0 ∧ C.1 + 2 * C.2 - 5 = 0 := by
  sorry

end isosceles_triangle_properties_l2102_210273


namespace triangle_problem_l2102_210293

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  A = π / 6 →
  b = (4 + 2 * Real.sqrt 3) * a * Real.cos B →
  b = 1 →
  B = 5 * π / 12 ∧ 
  (1 / 2) * b * c * Real.sin A = 1 / 4 :=
sorry

end triangle_problem_l2102_210293


namespace cube_surface_area_equal_volume_cylinder_l2102_210237

/-- The surface area of a cube with volume equal to a cylinder of radius 4 and height 12 -/
theorem cube_surface_area_equal_volume_cylinder (π : ℝ) :
  let cylinder_volume := π * 4^2 * 12
  let cube_edge := (cylinder_volume)^(1/3)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 6 * (192 * π)^(2/3) := by
  sorry

end cube_surface_area_equal_volume_cylinder_l2102_210237


namespace additional_oil_needed_l2102_210244

/-- Calculates the additional oil needed for a car engine -/
theorem additional_oil_needed
  (oil_per_cylinder : ℕ)
  (num_cylinders : ℕ)
  (oil_already_added : ℕ)
  (h1 : oil_per_cylinder = 8)
  (h2 : num_cylinders = 6)
  (h3 : oil_already_added = 16) :
  oil_per_cylinder * num_cylinders - oil_already_added = 32 :=
by sorry

end additional_oil_needed_l2102_210244


namespace least_multiple_15_with_digit_product_multiple_15_l2102_210238

/-- Given a natural number, returns the product of its digits. -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a multiple of 15. -/
def isMultipleOf15 (n : ℕ) : Prop := ∃ k, n = 15 * k

theorem least_multiple_15_with_digit_product_multiple_15 :
  ∀ n : ℕ, n > 0 → isMultipleOf15 n → isMultipleOf15 (digitProduct n) →
  n ≥ 315 ∧ (n = 315 → isMultipleOf15 (digitProduct 315)) := by sorry

end least_multiple_15_with_digit_product_multiple_15_l2102_210238


namespace series_sum_l2102_210296

/-- The sum of a specific infinite series given positive real numbers a and b where a > 3b -/
theorem series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > 3 * b) : 
  let series_term (n : ℕ) := 1 / (((3 * n - 6) * a - (n^2 - 5*n + 6) * b) * ((3 * n - 3) * a - (n^2 - 4*n + 3) * b))
  ∑' n, series_term n = 1 / (b * (a - b)) := by
sorry

end series_sum_l2102_210296


namespace rain_problem_l2102_210274

theorem rain_problem (first_hour : ℝ) (second_hour : ℝ) : 
  (second_hour = 2 * first_hour + 7) → 
  (first_hour + second_hour = 22) → 
  (first_hour = 5) := by
sorry

end rain_problem_l2102_210274


namespace dannys_travel_time_l2102_210222

theorem dannys_travel_time (danny_time steve_time halfway_danny halfway_steve : ℝ) 
  (h1 : steve_time = 2 * danny_time)
  (h2 : halfway_danny = danny_time / 2)
  (h3 : halfway_steve = steve_time / 2)
  (h4 : halfway_steve - halfway_danny = 12.5) :
  danny_time = 25 := by sorry

end dannys_travel_time_l2102_210222


namespace spelling_bee_points_ratio_l2102_210264

/-- Represents the spelling bee problem and proves the ratio of Val's points to Max and Dulce's combined points --/
theorem spelling_bee_points_ratio :
  -- Define the given points
  let max_points : ℕ := 5
  let dulce_points : ℕ := 3
  let opponents_points : ℕ := 40
  let points_behind : ℕ := 16

  -- Define Val's points as a multiple of Max and Dulce's combined points
  let val_points : ℕ → ℕ := λ k ↦ k * (max_points + dulce_points)

  -- Define the total points of Max, Dulce, and Val's team
  let team_total_points : ℕ → ℕ := λ k ↦ max_points + dulce_points + val_points k

  -- State the condition that their team's total points plus the points they're behind equals the opponents' points
  ∀ k : ℕ, team_total_points k + points_behind = opponents_points →

  -- Prove that the ratio of Val's points to Max and Dulce's combined points is 2:1
  ∃ k : ℕ, val_points k = 2 * (max_points + dulce_points) := by
  sorry


end spelling_bee_points_ratio_l2102_210264


namespace max_a_value_l2102_210227

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, 1 + a * Real.cos x ≥ 2/3 * Real.sin (π/2 + 2*x)) → 
  a ≤ 1/3 :=
sorry

end max_a_value_l2102_210227


namespace line_tangent_to_parabola_l2102_210262

/-- A line y = 3x + d is tangent to the parabola y² = 12x if and only if d = 1 -/
theorem line_tangent_to_parabola (d : ℝ) : 
  (∃ x y : ℝ, y = 3*x + d ∧ y^2 = 12*x ∧ 
    ∀ x' y' : ℝ, y' = 3*x' + d → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  d = 1 :=
sorry

end line_tangent_to_parabola_l2102_210262


namespace cosine_of_angle_between_lines_l2102_210208

def vector1 : ℝ × ℝ := (4, -1)
def vector2 : ℝ × ℝ := (2, 5)

theorem cosine_of_angle_between_lines :
  let v1 := vector1
  let v2 := vector2
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2
  let magnitude1 := Real.sqrt (v1.1^2 + v1.2^2)
  let magnitude2 := Real.sqrt (v2.1^2 + v2.2^2)
  dot_product / (magnitude1 * magnitude2) = 3 / Real.sqrt 493 := by
  sorry

end cosine_of_angle_between_lines_l2102_210208


namespace isosceles_triangles_height_ratio_l2102_210221

/-- Two isosceles triangles with equal vertical angles and area ratio 16:49 have height ratio 4:7 -/
theorem isosceles_triangles_height_ratio (b₁ h₁ b₂ h₂ : ℝ) : 
  b₁ > 0 → h₁ > 0 → b₂ > 0 → h₂ > 0 →  -- Positive dimensions
  (1/2 * b₁ * h₁) / (1/2 * b₂ * h₂) = 16/49 →  -- Area ratio
  b₁ / b₂ = h₁ / h₂ →  -- Similar triangles condition
  h₁ / h₂ = 4/7 := by
sorry

end isosceles_triangles_height_ratio_l2102_210221


namespace arithmetic_sequence_2017_l2102_210226

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is increasing if f(x) < f(y) whenever x < y -/
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (x : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, x (n + 1) = x n + d

theorem arithmetic_sequence_2017 (f : ℝ → ℝ) (x : ℕ → ℝ) :
  IsOdd f →
  IsIncreasing f →
  ArithmeticSequence x 2 →
  f (x 7) + f (x 8) = 0 →
  x 2017 = 4019 := by
  sorry

end arithmetic_sequence_2017_l2102_210226


namespace number_minus_six_l2102_210242

theorem number_minus_six (x : ℚ) : x / 5 = 2 → x - 6 = 4 := by
  sorry

end number_minus_six_l2102_210242


namespace arithmetic_sequence_common_difference_l2102_210278

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum1 : a 3 + a 4 + a 5 + a 6 + a 7 = 15)
  (h_sum2 : a 9 + a 10 + a 11 = 39) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end arithmetic_sequence_common_difference_l2102_210278


namespace quadratic_sum_l2102_210220

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, 6 * x^2 + 72 * x + 500 = a * (x + b)^2 + c) → 
  a + b + c = 296 := by
sorry

end quadratic_sum_l2102_210220


namespace unknown_number_in_set_l2102_210290

theorem unknown_number_in_set (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 40 := by
  sorry

end unknown_number_in_set_l2102_210290


namespace negation_of_existence_proposition_l2102_210289

theorem negation_of_existence_proposition :
  ¬(∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ 
  (∀ a : ℝ, ∀ x : ℝ, a * x^2 + 1 ≠ 0) :=
by sorry

end negation_of_existence_proposition_l2102_210289


namespace min_value_sum_of_fractions_l2102_210212

theorem min_value_sum_of_fractions (x y a b : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : a > 0) (h4 : b > 0) (h5 : x + y = 1) :
  a / x + b / y ≥ (Real.sqrt a + Real.sqrt b)^2 := by
  sorry

end min_value_sum_of_fractions_l2102_210212


namespace equation_solution_l2102_210277

theorem equation_solution (x : ℝ) : 
  x ≠ 1 → x ≠ -6 → 
  ((3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1)) ↔ (x = -4 ∨ x = -2) :=
by sorry

end equation_solution_l2102_210277


namespace coefficient_x_cubed_expansion_l2102_210280

theorem coefficient_x_cubed_expansion : 
  let expansion := (fun x => (x^2 + 1)^2 * (x - 1)^6)
  ∃ (a b c d e f g h : ℤ), 
    (∀ x, expansion x = a*x^8 + b*x^7 + c*x^6 + d*x^5 + e*x^4 + (-32)*x^3 + f*x^2 + g*x + h) :=
by sorry

end coefficient_x_cubed_expansion_l2102_210280


namespace misread_weight_correction_l2102_210268

theorem misread_weight_correction (n : ℕ) (incorrect_avg correct_avg misread_weight : ℝ) :
  n = 20 ∧ 
  incorrect_avg = 58.4 ∧ 
  correct_avg = 58.7 ∧ 
  misread_weight = 56 →
  ∃ correct_weight : ℝ,
    correct_weight = 62 ∧
    n * correct_avg = (n - 1) * incorrect_avg + correct_weight ∧
    n * incorrect_avg = (n - 1) * incorrect_avg + misread_weight :=
by sorry

end misread_weight_correction_l2102_210268


namespace boys_to_girls_ratio_l2102_210232

def num_boys : ℕ := 1500
def num_girls : ℕ := 1200

theorem boys_to_girls_ratio :
  (num_boys : ℚ) / (num_girls : ℚ) = 5 / 4 := by
  sorry

end boys_to_girls_ratio_l2102_210232
