import Mathlib

namespace NUMINAMATH_CALUDE_initial_bacteria_count_l1176_117625

def tripling_time : ℕ := 30  -- seconds
def total_time : ℕ := 300    -- seconds (5 minutes)
def final_count : ℕ := 1239220
def halfway_time : ℕ := 150  -- seconds (2.5 minutes)

def tripling_events (t : ℕ) : ℕ := t / tripling_time

theorem initial_bacteria_count :
  ∃ (n : ℕ),
    n * (3 ^ (tripling_events total_time)) / 2 = final_count ∧
    (n * (3 ^ (tripling_events halfway_time))) / 2 * (3 ^ (tripling_events halfway_time)) = final_count ∧
    n = 42 :=
by sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l1176_117625


namespace NUMINAMATH_CALUDE_problem_statement_l1176_117655

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2015 + b^2014 = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1176_117655


namespace NUMINAMATH_CALUDE_max_curved_sides_is_2n_minus_2_l1176_117659

/-- A type representing a figure formed by the intersection of circles -/
structure IntersectionFigure where
  n : ℕ
  h_n : n ≥ 2

/-- The number of curved sides in an intersection figure -/
def curved_sides (F : IntersectionFigure) : ℕ := sorry

/-- The maximum number of curved sides for a given number of circles -/
def max_curved_sides (n : ℕ) : ℕ := 2 * n - 2

/-- Theorem stating that the maximum number of curved sides is 2n - 2 -/
theorem max_curved_sides_is_2n_minus_2 (F : IntersectionFigure) :
  curved_sides F ≤ max_curved_sides F.n := by
  sorry

end NUMINAMATH_CALUDE_max_curved_sides_is_2n_minus_2_l1176_117659


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1176_117664

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x - 1) - 3 / (x - 3) + 5 / (x - 5) - 2 / (x - 7) < 1 / 15) ↔ 
  (x < -8 ∨ (-7 < x ∧ x < -1) ∨ (1 < x ∧ x < 3) ∨ (5 < x ∧ x < 7) ∨ x > 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1176_117664


namespace NUMINAMATH_CALUDE_complex_equation_roots_l1176_117665

theorem complex_equation_roots : 
  let z₁ : ℂ := -1 + Real.sqrt 5 - (2 * Real.sqrt 5 / 5) * I
  let z₂ : ℂ := -1 - Real.sqrt 5 + (2 * Real.sqrt 5 / 5) * I
  (z₁^2 + 2*z₁ = 3 - 4*I) ∧ (z₂^2 + 2*z₂ = 3 - 4*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l1176_117665


namespace NUMINAMATH_CALUDE_expected_participants_2008_l1176_117688

/-- The number of participants in the school festival after n years, given an initial number of participants and an annual increase rate. -/
def participants_after_n_years (initial : ℝ) (rate : ℝ) (n : ℕ) : ℝ :=
  initial * (1 + rate) ^ n

/-- The expected number of participants in 2008, given the initial number in 2005 and the annual increase rate. -/
theorem expected_participants_2008 :
  participants_after_n_years 1000 0.25 3 = 1953.125 := by
  sorry

#eval participants_after_n_years 1000 0.25 3

end NUMINAMATH_CALUDE_expected_participants_2008_l1176_117688


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1176_117622

def inversely_proportional_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = k

theorem tenth_term_of_sequence (a : ℕ → ℝ) :
  inversely_proportional_sequence a →
  a 1 = 3 →
  a 2 = 4 →
  a 10 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1176_117622


namespace NUMINAMATH_CALUDE_xyz_equals_five_l1176_117696

theorem xyz_equals_five
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = (b + c) / (x - 2))
  (eq_b : b = (a + c) / (y - 2))
  (eq_c : c = (a + b) / (z - 2))
  (sum_xy_xz_yz : x * y + x * z + y * z = 5)
  (sum_x_y_z : x + y + z = 3) :
  x * y * z = 5 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_five_l1176_117696


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l1176_117605

theorem magic_8_ball_probability :
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 4  -- number of positive answers
  let p : ℚ := 3/7  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 181440/823543 :=
by sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l1176_117605


namespace NUMINAMATH_CALUDE_cosine_theorem_trirectangular_angle_l1176_117641

-- Define the trirectangular angle
structure TrirectangularAngle where
  α : Real  -- plane angle opposite to SA
  β : Real  -- plane angle opposite to SB
  γ : Real  -- plane angle opposite to SC
  A : Real  -- dihedral angle at SA
  B : Real  -- dihedral angle at SB
  C : Real  -- dihedral angle at SC

-- State the theorem
theorem cosine_theorem_trirectangular_angle (t : TrirectangularAngle) :
  Real.cos t.α = Real.cos t.A * Real.cos t.B + Real.cos t.B * Real.cos t.C + Real.cos t.C * Real.cos t.A := by
  sorry

end NUMINAMATH_CALUDE_cosine_theorem_trirectangular_angle_l1176_117641


namespace NUMINAMATH_CALUDE_flour_for_dozen_cookies_l1176_117638

/-- Given information about cookie production and consumption, calculate the amount of flour needed for a dozen cookies -/
theorem flour_for_dozen_cookies 
  (bags : ℕ) 
  (weight_per_bag : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_left : ℕ) 
  (h1 : bags = 4) 
  (h2 : weight_per_bag = 5) 
  (h3 : cookies_eaten = 15) 
  (h4 : cookies_left = 105) : 
  (12 : ℝ) * (bags * weight_per_bag : ℝ) / ((cookies_left + cookies_eaten) : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_flour_for_dozen_cookies_l1176_117638


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l1176_117630

theorem mean_proportional_problem (B : ℝ) :
  (56.5 : ℝ) = Real.sqrt (49 * B) → B = 64.9 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l1176_117630


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l1176_117627

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | x < -5 ∨ x > 5}
def T : Set ℝ := {x : ℝ | -7 < x ∧ x < 3}

-- State the theorem
theorem set_intersection_theorem : S ∩ T = {x : ℝ | -7 < x ∧ x < -5} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l1176_117627


namespace NUMINAMATH_CALUDE_shopkeeper_percentage_gain_l1176_117609

/-- The percentage gain of a shopkeeper using a false weight --/
theorem shopkeeper_percentage_gain :
  let actual_weight : ℝ := 970
  let claimed_weight : ℝ := 1000
  let gain : ℝ := claimed_weight - actual_weight
  let percentage_gain : ℝ := (gain / actual_weight) * 100
  ∃ ε > 0, abs (percentage_gain - 3.09) < ε :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_percentage_gain_l1176_117609


namespace NUMINAMATH_CALUDE_factor_condition_l1176_117677

theorem factor_condition (x t : ℝ) : 
  (∃ k : ℝ, 6 * x^2 + 13 * x - 5 = (x - t) * k) ↔ (t = -5/2 ∨ t = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_factor_condition_l1176_117677


namespace NUMINAMATH_CALUDE_blue_to_red_ratio_is_four_to_one_l1176_117695

/-- Represents the number of pencils of each color and the total number of pencils. -/
structure PencilCounts where
  total : ℕ
  red : ℕ
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- Theorem stating that under given conditions, the ratio of blue to red pencils is 4:1. -/
theorem blue_to_red_ratio_is_four_to_one (p : PencilCounts)
    (h_total : p.total = 160)
    (h_red : p.red = 20)
    (h_yellow : p.yellow = 40)
    (h_green : p.green = p.red + p.blue) :
    p.blue / p.red = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_to_red_ratio_is_four_to_one_l1176_117695


namespace NUMINAMATH_CALUDE_amin_iff_ali_can_color_all_red_l1176_117644

-- Define a type for cell colors
inductive CellColor
| Black
| White
| Red

-- Define the table as a function from coordinates to cell colors
def Table (n : ℕ) := Fin n → Fin n → CellColor

-- Define Amin's move
def AminMove (t : Table n) (row : Fin n) : Table n :=
  sorry

-- Define Ali's move
def AliMove (t : Table n) (col : Fin n) : Table n :=
  sorry

-- Define a predicate to check if all cells are red
def AllRed (t : Table n) : Prop :=
  ∀ i j, t i j = CellColor.Red

-- Define a predicate to check if Amin can color all cells red
def AminCanColorAllRed (t : Table n) : Prop :=
  sorry

-- Define a predicate to check if Ali can color all cells red
def AliCanColorAllRed (t : Table n) : Prop :=
  sorry

-- The main theorem
theorem amin_iff_ali_can_color_all_red (n : ℕ) (t : Table n) :
  AminCanColorAllRed t ↔ AliCanColorAllRed t :=
sorry

end NUMINAMATH_CALUDE_amin_iff_ali_can_color_all_red_l1176_117644


namespace NUMINAMATH_CALUDE_average_of_combined_results_l1176_117634

theorem average_of_combined_results :
  let n₁ : ℕ := 40
  let avg₁ : ℚ := 30
  let n₂ : ℕ := 30
  let avg₂ : ℚ := 40
  let total_sum := n₁ * avg₁ + n₂ * avg₂
  let total_count := n₁ + n₂
  (total_sum / total_count : ℚ) = 2400 / 70 := by
sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l1176_117634


namespace NUMINAMATH_CALUDE_penny_bakery_revenue_l1176_117689

/-- Calculates the total money made from selling cheesecakes -/
def total_money_made (price_per_slice : ℕ) (slices_per_cake : ℕ) (cakes_sold : ℕ) : ℕ :=
  price_per_slice * slices_per_cake * cakes_sold

/-- Theorem: Penny's bakery makes $294 from selling 7 cheesecakes -/
theorem penny_bakery_revenue : total_money_made 7 6 7 = 294 := by
  sorry

end NUMINAMATH_CALUDE_penny_bakery_revenue_l1176_117689


namespace NUMINAMATH_CALUDE_polygon_deformation_to_triangle_l1176_117606

/-- A planar polygon represented by its vertices -/
structure PlanarPolygon where
  vertices : List (ℝ × ℝ)
  is_planar : sorry
  is_closed : sorry

/-- A function that checks if a polygon can be deformed into a triangle -/
def can_deform_to_triangle (p : PlanarPolygon) : Prop :=
  sorry

/-- The main theorem stating that any planar polygon with more than 4 sides
    can be deformed into a triangle -/
theorem polygon_deformation_to_triangle 
  (p : PlanarPolygon) (h : p.vertices.length > 4) :
  can_deform_to_triangle p :=
sorry

end NUMINAMATH_CALUDE_polygon_deformation_to_triangle_l1176_117606


namespace NUMINAMATH_CALUDE_sequence_inequality_l1176_117680

def a : ℕ → ℚ
  | 0 => 1
  | (n + 1) => a n - (a n)^2 / 2019

theorem sequence_inequality : a 2019 < 1/2 ∧ 1/2 < a 2018 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1176_117680


namespace NUMINAMATH_CALUDE_combined_salaries_l1176_117632

/-- The combined salaries of four employees given the salary of the fifth and the average of all five -/
theorem combined_salaries 
  (c_salary : ℕ) 
  (average_salary : ℕ) 
  (h1 : c_salary = 15000)
  (h2 : average_salary = 8800) :
  c_salary + 4 * average_salary - 5 * average_salary = 29000 :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_l1176_117632


namespace NUMINAMATH_CALUDE_tree_height_difference_l1176_117671

/-- The height difference between two trees -/
theorem tree_height_difference (pine_height maple_height : ℚ) 
  (h_pine : pine_height = 49/4)
  (h_maple : maple_height = 37/2) :
  maple_height - pine_height = 25/4 := by
  sorry

#eval (37/2 : ℚ) - (49/4 : ℚ)  -- Should output 25/4

end NUMINAMATH_CALUDE_tree_height_difference_l1176_117671


namespace NUMINAMATH_CALUDE_magician_trick_exists_strategy_l1176_117620

/-- Represents a card placement strategy for the magician's trick -/
structure CardPlacementStrategy (n : ℕ) :=
  (place_cards : Fin n → Fin n)
  (deduce_card1 : Fin n → Fin n → Fin n)
  (deduce_card2 : Fin n → Fin n → Fin n)

/-- The main theorem stating that a successful strategy exists for all n ≥ 3 -/
theorem magician_trick_exists_strategy (n : ℕ) (h : n ≥ 3) :
  ∃ (strategy : CardPlacementStrategy n),
    ∀ (card1_pos card2_pos : Fin n),
      card1_pos ≠ card2_pos →
      ∀ (magician_reveal spectator_reveal : Fin n),
        magician_reveal ≠ spectator_reveal →
        strategy.deduce_card1 magician_reveal spectator_reveal = card1_pos ∧
        strategy.deduce_card2 magician_reveal spectator_reveal = card2_pos :=
sorry

end NUMINAMATH_CALUDE_magician_trick_exists_strategy_l1176_117620


namespace NUMINAMATH_CALUDE_election_total_votes_l1176_117692

/-- An election with two candidates -/
structure Election :=
  (totalValidVotes : ℕ)
  (losingCandidatePercentage : ℚ)
  (voteDifference : ℕ)
  (invalidVotes : ℕ)

/-- The total number of polled votes in an election -/
def totalPolledVotes (e : Election) : ℕ :=
  e.totalValidVotes + e.invalidVotes

/-- Theorem stating the total number of polled votes in the given election -/
theorem election_total_votes (e : Election) 
  (h1 : e.losingCandidatePercentage = 45/100)
  (h2 : e.voteDifference = 9000)
  (h3 : e.invalidVotes = 83)
  : totalPolledVotes e = 90083 := by
  sorry

#eval totalPolledVotes { totalValidVotes := 90000, losingCandidatePercentage := 45/100, voteDifference := 9000, invalidVotes := 83 }

end NUMINAMATH_CALUDE_election_total_votes_l1176_117692


namespace NUMINAMATH_CALUDE_find_b_l1176_117648

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then 3 * x - b else 2^x

theorem find_b : ∃ b : ℝ, f b (f b (5/6)) = 4 ∧ b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l1176_117648


namespace NUMINAMATH_CALUDE_curve_translation_l1176_117660

-- Define the original curve
def original_curve (x y : ℝ) : Prop :=
  y * Real.cos x + 2 * y - 1 = 0

-- Define the translated curve
def translated_curve (x y : ℝ) : Prop :=
  (y - 1) * Real.sin x + 2 * y - 3 = 0

-- State the theorem
theorem curve_translation :
  ∀ (x y : ℝ),
    original_curve (x - π/2) (y + 1) ↔ translated_curve x y :=
by sorry

end NUMINAMATH_CALUDE_curve_translation_l1176_117660


namespace NUMINAMATH_CALUDE_collinear_vectors_m_values_l1176_117639

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def are_collinear (u v : V) : Prop := ∃ (k : ℝ), u = k • v

theorem collinear_vectors_m_values
  (a b : V)
  (h1 : ¬ are_collinear a b)
  (h2 : ∃ (k : ℝ), (m : ℝ) • a - 3 • b = k • (a + (2 - m) • b)) :
  m = -1 ∨ m = 3 :=
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_values_l1176_117639


namespace NUMINAMATH_CALUDE_original_price_of_discounted_items_l1176_117604

theorem original_price_of_discounted_items 
  (num_items : ℕ) 
  (discount_rate : ℚ) 
  (total_paid : ℚ) 
  (h1 : num_items = 6)
  (h2 : discount_rate = 1/2)
  (h3 : total_paid = 60) :
  (total_paid / (1 - discount_rate)) / num_items = 20 := by
sorry

end NUMINAMATH_CALUDE_original_price_of_discounted_items_l1176_117604


namespace NUMINAMATH_CALUDE_vector_subtraction_l1176_117698

/-- Given two vectors a and b in ℝ², prove that a - 2*b equals (7, 3) -/
theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (3, 5)) (h2 : b = (-2, 1)) :
  a - 2 • b = (7, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1176_117698


namespace NUMINAMATH_CALUDE_macaroon_ratio_is_two_to_one_l1176_117661

/-- Represents the numbers of macaroons in different states --/
structure MacaroonCounts where
  initial_red : ℕ
  initial_green : ℕ
  green_eaten : ℕ
  total_remaining : ℕ

/-- Calculates the ratio of red macaroons eaten to green macaroons eaten --/
def macaroon_ratio (m : MacaroonCounts) : ℚ :=
  let red_eaten := m.initial_red - (m.total_remaining - (m.initial_green - m.green_eaten))
  red_eaten / m.green_eaten

/-- Theorem stating that given the specific conditions, the ratio is 2:1 --/
theorem macaroon_ratio_is_two_to_one (m : MacaroonCounts) 
  (h1 : m.initial_red = 50)
  (h2 : m.initial_green = 40)
  (h3 : m.green_eaten = 15)
  (h4 : m.total_remaining = 45) :
  macaroon_ratio m = 2 := by
  sorry

end NUMINAMATH_CALUDE_macaroon_ratio_is_two_to_one_l1176_117661


namespace NUMINAMATH_CALUDE_sum_of_roots_is_negative_one_l1176_117687

-- Define the ∇ operation
def nabla (a b : ℝ) : ℝ := a * b - b * a^2

-- Theorem statement
theorem sum_of_roots_is_negative_one :
  let f : ℝ → ℝ := λ x => (nabla 2 x) - 8 - (nabla x 6)
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0) ∧ x₁ + x₂ = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_negative_one_l1176_117687


namespace NUMINAMATH_CALUDE_second_file_size_is_90_l1176_117681

/-- Represents the download scenario with given conditions -/
structure DownloadScenario where
  internetSpeed : ℕ  -- in megabits per minute
  totalTime : ℕ      -- in minutes
  fileCount : ℕ
  firstFileSize : ℕ  -- in megabits
  thirdFileSize : ℕ  -- in megabits

/-- Calculates the size of the second file given a download scenario -/
def secondFileSize (scenario : DownloadScenario) : ℕ :=
  scenario.internetSpeed * scenario.totalTime - scenario.firstFileSize - scenario.thirdFileSize

/-- Theorem stating that the size of the second file is 90 megabits -/
theorem second_file_size_is_90 (scenario : DownloadScenario) 
  (h1 : scenario.internetSpeed = 2)
  (h2 : scenario.totalTime = 120)
  (h3 : scenario.fileCount = 3)
  (h4 : scenario.firstFileSize = 80)
  (h5 : scenario.thirdFileSize = 70) :
  secondFileSize scenario = 90 := by
  sorry

#eval secondFileSize { 
  internetSpeed := 2, 
  totalTime := 120, 
  fileCount := 3, 
  firstFileSize := 80, 
  thirdFileSize := 70 
}

end NUMINAMATH_CALUDE_second_file_size_is_90_l1176_117681


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1176_117610

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≥ 100 → n.mod 17 = 0 → n ≥ 102 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1176_117610


namespace NUMINAMATH_CALUDE_marble_weight_proof_l1176_117650

/-- The weight of one marble in pounds -/
def marble_weight : ℚ := 100 / 9

/-- The weight of one waffle iron in pounds -/
def waffle_iron_weight : ℚ := 25

theorem marble_weight_proof :
  (9 * marble_weight = 4 * waffle_iron_weight) ∧
  (3 * waffle_iron_weight = 75) →
  marble_weight = 100 / 9 := by
sorry

end NUMINAMATH_CALUDE_marble_weight_proof_l1176_117650


namespace NUMINAMATH_CALUDE_restaurant_menu_combinations_l1176_117636

theorem restaurant_menu_combinations (n : ℕ) (h : n = 12) :
  n * (n - 1) = 132 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_menu_combinations_l1176_117636


namespace NUMINAMATH_CALUDE_doraemon_toys_count_l1176_117666

theorem doraemon_toys_count : ∃! n : ℕ, 40 ≤ n ∧ n ≤ 55 ∧ (n - 3) % 5 = 0 ∧ (n + 2) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_doraemon_toys_count_l1176_117666


namespace NUMINAMATH_CALUDE_direct_proportion_point_value_l1176_117653

/-- A directly proportional function passing through points (-2, 3) and (a, -3) has a = 2 -/
theorem direct_proportion_point_value (k a : ℝ) : 
  (∃ k : ℝ, k * (-2) = 3 ∧ k * a = -3) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_point_value_l1176_117653


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1176_117691

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The number of coins that need to match -/
def num_matching : ℕ := 3

/-- The number of possible outcomes for each coin -/
def outcomes_per_coin : ℕ := 2

/-- The total number of possible outcomes when flipping the coins -/
def total_outcomes : ℕ := outcomes_per_coin ^ num_coins

/-- The number of successful outcomes where the specified coins match -/
def successful_outcomes : ℕ := outcomes_per_coin * outcomes_per_coin ^ (num_coins - num_matching)

/-- The probability of the specified coins matching -/
def probability : ℚ := successful_outcomes / total_outcomes

theorem coin_flip_probability : probability = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1176_117691


namespace NUMINAMATH_CALUDE_complex_sum_simplification_l1176_117646

theorem complex_sum_simplification : 
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  z₁^12 + z₂^12 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_simplification_l1176_117646


namespace NUMINAMATH_CALUDE_rectangle_to_square_cut_l1176_117619

theorem rectangle_to_square_cut (rectangle_length : ℝ) (rectangle_width : ℝ) (num_parts : ℕ) :
  rectangle_length = 2 ∧ rectangle_width = 1 ∧ num_parts = 3 →
  ∃ (square_side : ℝ), square_side = Real.sqrt 2 ∧
    rectangle_length * rectangle_width = square_side * square_side :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_cut_l1176_117619


namespace NUMINAMATH_CALUDE_roots_are_imaginary_l1176_117663

theorem roots_are_imaginary (m : ℝ) : 
  (∃ x y : ℂ, x^2 - 4*m*x + 5*m^2 + 2 = 0 ∧ y^2 - 4*m*y + 5*m^2 + 2 = 0 ∧ x*y = 9) →
  (∃ a b : ℝ, a ≠ 0 ∧ (∀ z : ℂ, z^2 - 4*m*z + 5*m^2 + 2 = 0 → ∃ r : ℝ, z = Complex.mk r (a*r + b) ∨ z = Complex.mk r (-a*r - b))) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_imaginary_l1176_117663


namespace NUMINAMATH_CALUDE_quadratic_root_property_l1176_117670

theorem quadratic_root_property (a b s t : ℝ) (h_neq : s ≠ t) 
  (h_ps : s^2 + a*s + b = t) (h_pt : t^2 + a*t + b = s) : 
  (b - s*t)^2 + a*(b - s*t) + b - s*t = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l1176_117670


namespace NUMINAMATH_CALUDE_day_50_of_previous_year_is_thursday_l1176_117684

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year -/
structure Year where
  number : ℕ

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek :=
  sorry

/-- Returns the number of days in a year -/
def daysInYear (y : Year) : ℕ :=
  sorry

theorem day_50_of_previous_year_is_thursday
  (N : Year)
  (h1 : dayOfWeek N 250 = DayOfWeek.Friday)
  (h2 : dayOfWeek (Year.mk (N.number + 1)) 150 = DayOfWeek.Friday) :
  dayOfWeek (Year.mk (N.number - 1)) 50 = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_day_50_of_previous_year_is_thursday_l1176_117684


namespace NUMINAMATH_CALUDE_circle_parabola_height_difference_l1176_117628

/-- The height difference between the center of a circle and its points of tangency with the parabola y = 2x^2 -/
theorem circle_parabola_height_difference (a : ℝ) : 
  ∃ (b r : ℝ), 
    (∀ x y : ℝ, y = 2 * x^2 → x^2 + (y - b)^2 = r^2 → x = a ∨ x = -a) →
    (b - 2 * a^2 = 1/4 - a^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_parabola_height_difference_l1176_117628


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1176_117685

theorem inverse_variation_problem (y x : ℝ) (k : ℝ) :
  (∀ x y, y * x^2 = k) →  -- y varies inversely as x^2
  (1 * 4^2 = k) →         -- when x = 4, y = 1
  (0.25 * x^2 = k) →      -- condition for y = 0.25
  x = 8 :=                -- prove x = 8
by
  sorry

#check inverse_variation_problem

end NUMINAMATH_CALUDE_inverse_variation_problem_l1176_117685


namespace NUMINAMATH_CALUDE_oliver_socks_l1176_117612

/-- The number of socks Oliver initially had -/
def initial_socks : ℕ := 11

/-- The number of socks Oliver threw away -/
def thrown_away_socks : ℕ := 4

/-- The number of new socks Oliver bought -/
def new_socks : ℕ := 26

/-- The number of socks Oliver has now -/
def current_socks : ℕ := 33

theorem oliver_socks : 
  initial_socks - thrown_away_socks + new_socks = current_socks := by
  sorry


end NUMINAMATH_CALUDE_oliver_socks_l1176_117612


namespace NUMINAMATH_CALUDE_rational_roots_of_p_l1176_117668

def p (x : ℚ) : ℚ := x^4 - 3*x^3 - 8*x^2 + 12*x + 16

theorem rational_roots_of_p :
  {x : ℚ | p x = 0} = {-1, -2, 2, 4} := by sorry

end NUMINAMATH_CALUDE_rational_roots_of_p_l1176_117668


namespace NUMINAMATH_CALUDE_page_number_digit_difference_l1176_117614

/-- Counts the occurrences of a digit in a range of numbers -/
def countDigit (d : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The difference between the number of 3's and 7's in page numbers of a book -/
def digitDifference (pages : Nat) : Nat :=
  (countDigit 3 1 pages) - (countDigit 7 1 pages)

theorem page_number_digit_difference :
  digitDifference 350 = 56 := by sorry

end NUMINAMATH_CALUDE_page_number_digit_difference_l1176_117614


namespace NUMINAMATH_CALUDE_cyclic_trio_exists_l1176_117637

/-- Represents the result of a match between two players -/
inductive MatchResult
| Win
| Loss

/-- A tournament with a fixed number of players -/
structure Tournament where
  numPlayers : Nat
  results : Fin numPlayers → Fin numPlayers → MatchResult

/-- Predicate to check if player i defeated player j -/
def defeated (t : Tournament) (i j : Fin t.numPlayers) : Prop :=
  t.results i j = MatchResult.Win

theorem cyclic_trio_exists (t : Tournament) 
  (h1 : t.numPlayers = 12)
  (h2 : ∀ i j : Fin t.numPlayers, i ≠ j → (defeated t i j ∨ defeated t j i))
  (h3 : ∀ i : Fin t.numPlayers, ∃ j : Fin t.numPlayers, defeated t i j) :
  ∃ a b c : Fin t.numPlayers, 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    defeated t a b ∧ defeated t b c ∧ defeated t c a :=
sorry

end NUMINAMATH_CALUDE_cyclic_trio_exists_l1176_117637


namespace NUMINAMATH_CALUDE_survey_result_l1176_117656

def teachers_survey (total : ℕ) (high_bp : ℕ) (heart : ℕ) (diabetes : ℕ) 
  (high_bp_heart : ℕ) (diabetes_heart : ℕ) (diabetes_high_bp : ℕ) (all_three : ℕ) : Prop :=
  let teachers_with_condition := 
    high_bp + heart + diabetes - high_bp_heart - diabetes_heart - diabetes_high_bp + all_three
  let teachers_without_condition := total - teachers_with_condition
  (teachers_without_condition : ℚ) / total * 100 = 28

theorem survey_result : 
  teachers_survey 150 90 60 10 30 5 8 3 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_result_l1176_117656


namespace NUMINAMATH_CALUDE_two_numbers_satisfy_property_l1176_117658

/-- Given a two-digit positive integer, return the integer obtained by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The property that we're checking for each two-digit number -/
def has_property (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_perfect_square (n + (reverse_digits n)^3)

/-- The main theorem stating that exactly two numbers satisfy the property -/
theorem two_numbers_satisfy_property :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n, n ∈ s ↔ has_property n :=
sorry

end NUMINAMATH_CALUDE_two_numbers_satisfy_property_l1176_117658


namespace NUMINAMATH_CALUDE_parallel_tangents_imply_m_values_l1176_117611

-- Define the line and curve
def line (x y : ℝ) : Prop := x - 9 * y - 8 = 0
def curve (x y m : ℝ) : Prop := y = x^3 - m * x^2 + 3 * x

-- Define the tangent slope at a point on the curve
def tangent_slope (x m : ℝ) : ℝ := 3 * x^2 - 2 * m * x + 3

-- State the theorem
theorem parallel_tangents_imply_m_values (m : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    line x₁ y₁ ∧ line x₂ y₂ ∧
    curve x₁ y₁ m ∧ curve x₂ y₂ m ∧
    x₁ ≠ x₂ ∧
    tangent_slope x₁ m = tangent_slope x₂ m) →
  m = 4 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_parallel_tangents_imply_m_values_l1176_117611


namespace NUMINAMATH_CALUDE_expected_heads_is_40_l1176_117683

/-- A coin toss simulation with specific rules --/
def CoinTossSimulation :=
  { n : ℕ  // n = 80 }

/-- The probability of a coin showing heads after all tosses --/
def prob_heads (c : CoinTossSimulation) : ℚ :=
  1 / 2

/-- The expected number of heads in the simulation --/
def expected_heads (c : CoinTossSimulation) : ℚ :=
  c.val * prob_heads c

/-- Theorem stating that the expected number of heads is 40 --/
theorem expected_heads_is_40 (c : CoinTossSimulation) :
  expected_heads c = 40 := by
  sorry

#check expected_heads_is_40

end NUMINAMATH_CALUDE_expected_heads_is_40_l1176_117683


namespace NUMINAMATH_CALUDE_fixed_points_of_f_composition_l1176_117675

def f (x : ℝ) : ℝ := x^2 - 4*x

theorem fixed_points_of_f_composition (x : ℝ) : 
  f (f x) = f x ↔ x ∈ ({-1, 0, 4, 5} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_composition_l1176_117675


namespace NUMINAMATH_CALUDE_integer_coloring_theorem_l1176_117613

/-- A color type with four colors -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A coloring function that assigns a color to each integer -/
def coloring : ℤ → Color := sorry

theorem integer_coloring_theorem 
  (m n : ℤ) 
  (h_odd_m : Odd m) 
  (h_odd_n : Odd n) 
  (h_distinct : m ≠ n) 
  (h_sum_nonzero : m + n ≠ 0) :
  ∃ (a b : ℤ), 
    coloring a = coloring b ∧ 
    (a - b = m ∨ a - b = n ∨ a - b = m + n ∨ a - b = m - n) := by
  sorry

end NUMINAMATH_CALUDE_integer_coloring_theorem_l1176_117613


namespace NUMINAMATH_CALUDE_g_of_5_l1176_117640

/-- The function g satisfies the given functional equation for all real x -/
axiom functional_equation (g : ℝ → ℝ) :
  ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2

/-- The value of g(5) is -20.01 -/
theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2) :
  g 5 = -20.01 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l1176_117640


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1176_117678

theorem absolute_value_equation (a : ℝ) : 
  |2*a + 1| = 3*|a| - 2 → a = -1 ∨ a = 3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1176_117678


namespace NUMINAMATH_CALUDE_min_value_of_f_l1176_117608

-- Define the function
def f (y : ℝ) : ℝ := 3 * y^2 - 18 * y + 11

-- State the theorem
theorem min_value_of_f :
  ∃ (y_min : ℝ), ∀ (y : ℝ), f y ≥ f y_min ∧ f y_min = -16 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1176_117608


namespace NUMINAMATH_CALUDE_total_students_is_1076_l1176_117669

/-- Represents the number of students in a school --/
structure School where
  girls : ℕ
  boys : ℕ

/-- The total number of students in the school --/
def School.total (s : School) : ℕ := s.girls + s.boys

/-- A school with 402 more girls than boys and 739 girls --/
def our_school : School := {
  girls := 739,
  boys := 739 - 402
}

/-- Theorem stating that the total number of students in our_school is 1076 --/
theorem total_students_is_1076 : our_school.total = 1076 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_1076_l1176_117669


namespace NUMINAMATH_CALUDE_time_to_plant_trees_l1176_117672

-- Define the rate of planting trees
def trees_per_minute : ℚ := 10 / 3

-- Define the total number of trees to be planted
def total_trees : ℕ := 2500

-- Define the time it takes to plant all trees in hours
def planting_time : ℚ := 12.5

-- Theorem to prove
theorem time_to_plant_trees :
  trees_per_minute * 60 * planting_time = total_trees :=
sorry

end NUMINAMATH_CALUDE_time_to_plant_trees_l1176_117672


namespace NUMINAMATH_CALUDE_polar_to_cartesian_parabola_l1176_117682

/-- The curve defined by the polar equation r = 1 / (1 - sin θ) is a parabola -/
theorem polar_to_cartesian_parabola :
  ∃ (x y : ℝ), (∃ (r θ : ℝ), r = 1 / (1 - Real.sin θ) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  x^2 = 2*y + 1 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_parabola_l1176_117682


namespace NUMINAMATH_CALUDE_geometric_sequence_roots_l1176_117699

theorem geometric_sequence_roots (m n : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^2 - m*a + 2) * (a^2 - n*a + 2) = 0 ∧
    (b^2 - m*b + 2) * (b^2 - n*b + 2) = 0 ∧
    (c^2 - m*c + 2) * (c^2 - n*c + 2) = 0 ∧
    (d^2 - m*d + 2) * (d^2 - n*d + 2) = 0 ∧
    a = (1/2) ∧
    (∃ r : ℝ, b = a*r ∧ c = b*r ∧ d = c*r)) →
  |m - n| = (3/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_roots_l1176_117699


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l1176_117643

/-- The number of sides in a regular nonagon -/
def n : ℕ := 9

/-- The total number of line segments (sides and diagonals) in a regular nonagon -/
def total_segments : ℕ := n.choose 2

/-- The number of diagonals in a regular nonagon -/
def num_diagonals : ℕ := total_segments - n

/-- The number of ways to choose two diagonals -/
def ways_to_choose_diagonals : ℕ := num_diagonals.choose 2

/-- The number of ways to choose four points that form intersecting diagonals -/
def intersecting_diagonals : ℕ := n.choose 4

/-- The probability that two randomly chosen diagonals intersect inside the nonagon -/
def probability_intersect : ℚ := intersecting_diagonals / ways_to_choose_diagonals

theorem nonagon_diagonal_intersection_probability :
  probability_intersect = 6 / 13 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l1176_117643


namespace NUMINAMATH_CALUDE_remainder_3_250_mod_11_l1176_117617

theorem remainder_3_250_mod_11 : 3^250 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_250_mod_11_l1176_117617


namespace NUMINAMATH_CALUDE_roller_derby_teams_l1176_117697

/-- The number of teams competing in a roller derby --/
def number_of_teams (members_per_team : ℕ) (skates_per_member : ℕ) (laces_per_skate : ℕ) (total_laces : ℕ) : ℕ :=
  total_laces / (members_per_team * skates_per_member * laces_per_skate)

/-- Theorem stating that the number of teams competing is 4 --/
theorem roller_derby_teams : number_of_teams 10 2 3 240 = 4 := by
  sorry

end NUMINAMATH_CALUDE_roller_derby_teams_l1176_117697


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l1176_117662

theorem largest_n_for_factorization : ∃ (n : ℤ),
  (∀ m : ℤ, (∃ (a b c d : ℤ), 7 * X^2 + m * X + 56 = (a * X + b) * (c * X + d)) → m ≤ n) ∧
  (∃ (a b c d : ℤ), 7 * X^2 + n * X + 56 = (a * X + b) * (c * X + d)) ∧
  n = 393 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l1176_117662


namespace NUMINAMATH_CALUDE_least_k_factorial_multiple_of_315_l1176_117633

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem least_k_factorial_multiple_of_315 (k : ℕ) (h1 : k > 1) (h2 : 315 ∣ factorial k) :
  k ≥ 7 ∧ 315 ∣ factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_least_k_factorial_multiple_of_315_l1176_117633


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l1176_117631

theorem r_value_when_n_is_3 (n : ℕ) (s : ℕ) (r : ℕ) 
  (h1 : s = 2^n - 1) 
  (h2 : r = 3^s - s) 
  (h3 : n = 3) : 
  r = 2180 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l1176_117631


namespace NUMINAMATH_CALUDE_special_pizza_all_toppings_l1176_117674

/-- Represents a pizza with various toppings -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  olive_slices : ℕ
  all_toppings_slices : ℕ

/-- Conditions for our specific pizza -/
def special_pizza : Pizza := {
  total_slices := 24,
  pepperoni_slices := 15,
  mushroom_slices := 16,
  olive_slices := 10,
  all_toppings_slices := 2
}

/-- Every slice has at least one topping -/
def has_at_least_one_topping (p : Pizza) : Prop :=
  p.pepperoni_slices + p.mushroom_slices + p.olive_slices - p.all_toppings_slices ≥ p.total_slices

/-- The theorem to prove -/
theorem special_pizza_all_toppings :
  has_at_least_one_topping special_pizza ∧
  special_pizza.all_toppings_slices = 2 :=
sorry


end NUMINAMATH_CALUDE_special_pizza_all_toppings_l1176_117674


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1176_117652

theorem diophantine_equation_solution : ∃ (a b c d : ℕ+), 
  (a^3 + b^4 + c^5 = d^11) ∧ (a * b * c < 10^5) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1176_117652


namespace NUMINAMATH_CALUDE_point_on_line_between_l1176_117686

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Check if a point is between two other points -/
def between (p q r : Point) : Prop :=
  collinear p q r ∧
  min p.x r.x ≤ q.x ∧ q.x ≤ max p.x r.x ∧
  min p.y r.y ≤ q.y ∧ q.y ≤ max p.y r.y

theorem point_on_line_between (p₁ p₂ q : Point) 
  (h₁ : p₁ = ⟨8, 16⟩) 
  (h₂ : p₂ = ⟨2, 6⟩)
  (h₃ : q = ⟨5, 11⟩) : 
  between p₁ q p₂ := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_between_l1176_117686


namespace NUMINAMATH_CALUDE_x_2023_minus_1_values_l1176_117626

theorem x_2023_minus_1_values (x : ℝ) : 
  (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 → 
  x^2023 - 1 = 0 ∨ x^2023 - 1 = -2 := by
sorry

end NUMINAMATH_CALUDE_x_2023_minus_1_values_l1176_117626


namespace NUMINAMATH_CALUDE_solution_verification_l1176_117615

theorem solution_verification (x y : ℝ) : x = 2 ∧ x + y = 3 → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_verification_l1176_117615


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1176_117649

-- Problem 1
theorem problem_1 : 211 * (-455) + 365 * 455 - 211 * 545 + 545 * 365 = 154000 := by
  sorry

-- Problem 2
theorem problem_2 : (-7/5 * (-5/2) - 1) / 9 / (1/(-3/4)^2) - |2 + (-1/2)^3 * 5^2| = -31/32 := by
  sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (3*x + 2)*(x + 1) + 2*(x - 3)*(x + 2) = 5*x^2 + 3*x - 10 := by
  sorry

-- Problem 4
theorem problem_4 : ∃ (x : ℚ), (2*x + 3)/6 - (2*x - 1)/4 = 1 ∧ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1176_117649


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1176_117690

theorem simplify_and_evaluate (a : ℤ) (h : a = 2023) :
  a * (1 - 2 * a) + 2 * (a + 1) * (a - 1) = 2021 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1176_117690


namespace NUMINAMATH_CALUDE_smallest_prime_above_50_l1176_117600

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_prime_above_50 :
  ∃ p : ℕ, is_prime p ∧ p > 50 ∧ ∀ q : ℕ, is_prime q ∧ q > 50 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_above_50_l1176_117600


namespace NUMINAMATH_CALUDE_wendy_sales_l1176_117607

/-- Represents the sales data for a fruit vendor --/
structure FruitSales where
  apple_price : ℝ
  orange_price : ℝ
  morning_apples : ℕ
  morning_oranges : ℕ
  afternoon_apples : ℕ
  afternoon_oranges : ℕ

/-- Calculates the total sales for a given FruitSales instance --/
def total_sales (sales : FruitSales) : ℝ :=
  let total_apples := sales.morning_apples + sales.afternoon_apples
  let total_oranges := sales.morning_oranges + sales.afternoon_oranges
  (total_apples : ℝ) * sales.apple_price + (total_oranges : ℝ) * sales.orange_price

/-- Theorem stating that the total sales for the given conditions equal $205 --/
theorem wendy_sales : 
  let sales := FruitSales.mk 1.5 1 40 30 50 40
  total_sales sales = 205 := by
  sorry


end NUMINAMATH_CALUDE_wendy_sales_l1176_117607


namespace NUMINAMATH_CALUDE_xy_value_l1176_117603

theorem xy_value (x y : ℝ) (h : Real.sqrt (2 * x - 4) + |y - 1| = 0) : x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1176_117603


namespace NUMINAMATH_CALUDE_inverse_composition_l1176_117618

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- Assumption: f and g are bijective (to ensure inverses exist)
variable (hf : Function.Bijective f)
variable (hg : Function.Bijective g)

-- Define the relationship between f_inv and g
axiom relation (x : ℝ) : f_inv (g x) = 4 * x - 2

-- Theorem to prove
theorem inverse_composition :
  g_inv (f 5) = 7/4 :=
sorry

end NUMINAMATH_CALUDE_inverse_composition_l1176_117618


namespace NUMINAMATH_CALUDE_triangle_theorem_l1176_117645

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.b - 2 * t.a) * Real.cos t.C + t.c * Real.cos t.B = 0)
  (h2 : t.c = Real.sqrt 7)
  (h3 : t.b = 3 * t.a) :
  t.C = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1176_117645


namespace NUMINAMATH_CALUDE_local_min_implies_a_half_subset_implies_a_range_l1176_117601

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.log x - a * (x^2 - 1)

-- Part 1: Local minimum at x = 1 implies a = 1/2
theorem local_min_implies_a_half (a : ℝ) :
  (∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → f a x ≥ f a 1) →
  a = 1/2 :=
sorry

-- Part 2: N ⊆ M implies a ∈ (-∞, 1/2]
theorem subset_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) →
  a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_local_min_implies_a_half_subset_implies_a_range_l1176_117601


namespace NUMINAMATH_CALUDE_circle_plus_solution_l1176_117676

def circle_plus (a b : ℝ) : ℝ := a * b - 2 * b + 3 * a

theorem circle_plus_solution :
  ∃ x : ℝ, circle_plus 7 x = 61 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_solution_l1176_117676


namespace NUMINAMATH_CALUDE_correct_num_spiders_l1176_117616

/-- The number of spiders introduced to control pests in a garden --/
def num_spiders : ℕ := 12

/-- The initial number of bugs in the garden --/
def initial_bugs : ℕ := 400

/-- The number of bugs each spider eats --/
def bugs_per_spider : ℕ := 7

/-- The fraction of bugs remaining after spraying --/
def spray_factor : ℚ := 4/5

/-- The number of bugs remaining after pest control measures --/
def remaining_bugs : ℕ := 236

/-- Theorem stating that the number of spiders introduced is correct --/
theorem correct_num_spiders :
  (initial_bugs : ℚ) * spray_factor - (num_spiders : ℚ) * bugs_per_spider = remaining_bugs := by
  sorry

end NUMINAMATH_CALUDE_correct_num_spiders_l1176_117616


namespace NUMINAMATH_CALUDE_same_type_as_3a2b_l1176_117679

/-- Two terms are of the same type if they have the same variables with the same exponents. -/
def same_type (t1 t2 : ℕ → ℕ → ℕ) : Prop :=
  ∀ a b, t1 a b ≠ 0 ∧ t2 a b ≠ 0 → 
    (t1 a b).factors.toFinset = (t2 a b).factors.toFinset

/-- The term $3a^2b$ -/
def term1 (a b : ℕ) : ℕ := 3 * a^2 * b

/-- The term $2ab^2$ -/
def term2 (a b : ℕ) : ℕ := 2 * a * b^2

/-- The term $-a^2b$ -/
def term3 (a b : ℕ) : ℕ := a^2 * b

/-- The term $-2ab$ -/
def term4 (a b : ℕ) : ℕ := 2 * a * b

/-- The term $5a^2$ -/
def term5 (a b : ℕ) : ℕ := 5 * a^2

theorem same_type_as_3a2b :
  same_type term1 term3 ∧
  ¬ same_type term1 term2 ∧
  ¬ same_type term1 term4 ∧
  ¬ same_type term1 term5 :=
sorry

end NUMINAMATH_CALUDE_same_type_as_3a2b_l1176_117679


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_square_l1176_117621

/-- A quadrilateral with equal length diagonals that are perpendicular to each other -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has diagonals of equal length -/
  equal_diagonals : Bool
  /-- The diagonals are perpendicular to each other -/
  perpendicular_diagonals : Bool

/-- Definition of a square -/
def is_square (q : SpecialQuadrilateral) : Prop :=
  q.equal_diagonals ∧ q.perpendicular_diagonals

/-- Theorem stating that a quadrilateral with equal length diagonals that are perpendicular to each other is a square -/
theorem special_quadrilateral_is_square (q : SpecialQuadrilateral) 
  (h1 : q.equal_diagonals = true) 
  (h2 : q.perpendicular_diagonals = true) : 
  is_square q := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_is_square_l1176_117621


namespace NUMINAMATH_CALUDE_min_value_fraction_l1176_117654

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 2*y^2 + z^2) / (x*y + 3*y*z) ≥ 2*Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1176_117654


namespace NUMINAMATH_CALUDE_all_pairs_product_48_l1176_117673

theorem all_pairs_product_48 : 
  ((-6) * (-8) = 48) ∧
  ((-4) * (-12) = 48) ∧
  ((3/2 : ℚ) * 32 = 48) ∧
  (2 * 24 = 48) ∧
  ((4/3 : ℚ) * 36 = 48) := by
  sorry

end NUMINAMATH_CALUDE_all_pairs_product_48_l1176_117673


namespace NUMINAMATH_CALUDE_divisibility_condition_l1176_117647

theorem divisibility_condition (n : ℕ) : 
  (2^n + n) ∣ (8^n + n) ↔ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1176_117647


namespace NUMINAMATH_CALUDE_total_amount_calculation_l1176_117667

-- Define the given parameters
def interest_rate : ℚ := 8 / 100
def time_period : ℕ := 2
def compound_interest : ℚ := 2828.80

-- Define the compound interest formula
def compound_interest_formula (P : ℚ) : ℚ :=
  P * (1 + interest_rate) ^ time_period - P

-- Define the total amount formula
def total_amount (P : ℚ) : ℚ :=
  P + compound_interest

-- Theorem statement
theorem total_amount_calculation :
  ∃ P : ℚ, compound_interest_formula P = compound_interest ∧
           total_amount P = 19828.80 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l1176_117667


namespace NUMINAMATH_CALUDE_smallest_n_is_three_l1176_117602

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define x and y
noncomputable def x : ℂ := (-1 + i * Real.sqrt 3) / 2
noncomputable def y : ℂ := (-1 - i * Real.sqrt 3) / 2

-- Define the property we want to prove
def is_smallest_n (n : ℕ) : Prop :=
  n > 0 ∧ x^n + y^n = 2 ∧ ∀ m : ℕ, 0 < m ∧ m < n → x^m + y^m ≠ 2

-- The theorem we want to prove
theorem smallest_n_is_three : is_smallest_n 3 := by sorry

end NUMINAMATH_CALUDE_smallest_n_is_three_l1176_117602


namespace NUMINAMATH_CALUDE_sixth_game_score_l1176_117693

theorem sixth_game_score (scores : List ℕ) (mean : ℚ) : 
  scores.length = 7 ∧
  scores = [69, 68, 70, 61, 74, 65, 74] ∧
  mean = 67.9 ∧
  (∃ x : ℕ, (scores.sum + x) / 8 = mean) →
  ∃ x : ℕ, x = 62 ∧ (scores.sum + x) / 8 = mean := by
sorry

end NUMINAMATH_CALUDE_sixth_game_score_l1176_117693


namespace NUMINAMATH_CALUDE_matrix_vector_product_l1176_117629

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; -6, 5]
def v : Matrix (Fin 2) (Fin 1) ℝ := !![2; -3]

theorem matrix_vector_product :
  A * v = !![14; -27] := by sorry

end NUMINAMATH_CALUDE_matrix_vector_product_l1176_117629


namespace NUMINAMATH_CALUDE_dad_steps_count_l1176_117624

theorem dad_steps_count (dad_masha_ratio : ℕ → ℕ → Prop)
                        (masha_yasha_ratio : ℕ → ℕ → Prop)
                        (masha_yasha_total : ℕ) :
  dad_masha_ratio 3 5 →
  masha_yasha_ratio 3 5 →
  masha_yasha_total = 400 →
  ∃ (dad_steps : ℕ), dad_steps = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_count_l1176_117624


namespace NUMINAMATH_CALUDE_red_ball_certain_l1176_117623

/-- Represents the number of balls of each color in the box -/
structure BallCount where
  red : Nat
  yellow : Nat

/-- Represents the number of balls drawn from the box -/
def BallsDrawn : Nat := 3

/-- The initial state of the box -/
def initialBox : BallCount where
  red := 3
  yellow := 2

/-- A function to check if drawing at least one red ball is certain -/
def isRedBallCertain (box : BallCount) : Prop :=
  box.yellow < BallsDrawn

/-- Theorem stating that drawing at least one red ball is certain -/
theorem red_ball_certain :
  isRedBallCertain initialBox := by
  sorry

end NUMINAMATH_CALUDE_red_ball_certain_l1176_117623


namespace NUMINAMATH_CALUDE_triangle_angle_F_l1176_117635

theorem triangle_angle_F (D E : Real) (h1 : 2 * Real.sin D + 5 * Real.cos E = 7)
                         (h2 : 5 * Real.sin E + 2 * Real.cos D = 4) :
  Real.sin (π - D - E) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_F_l1176_117635


namespace NUMINAMATH_CALUDE_intersection_point_l1176_117657

-- Define the line using a parameter t
def line (t : ℝ) : ℝ × ℝ × ℝ := (1 - 2*t, 2 + t, -1 - t)

-- Define the plane equation
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  x - 2*y + 5*z + 17 = 0

-- Theorem statement
theorem intersection_point :
  ∃! p : ℝ × ℝ × ℝ, (∃ t : ℝ, line t = p) ∧ plane p ∧ p = (-1, 3, -2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1176_117657


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_l1176_117642

theorem sum_of_roots_equals_fourteen : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 ∧ x₁ + x₂ = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_l1176_117642


namespace NUMINAMATH_CALUDE_g_of_4_l1176_117651

def g (x : ℝ) : ℝ := 5 * x - 2

theorem g_of_4 : g 4 = 18 := by sorry

end NUMINAMATH_CALUDE_g_of_4_l1176_117651


namespace NUMINAMATH_CALUDE_stability_comparison_l1176_117694

/-- Represents a set of data with its variance -/
structure DataSet where
  variance : ℝ

/-- Stability comparison between two data sets -/
def more_stable (a b : DataSet) : Prop := a.variance < b.variance

/-- Theorem: If two data sets have the same average and set A has lower variance,
    then set A is more stable than set B -/
theorem stability_comparison (A B : DataSet) 
  (h1 : A.variance = 2)
  (h2 : B.variance = 2.5)
  : more_stable A B := by
  sorry

end NUMINAMATH_CALUDE_stability_comparison_l1176_117694
