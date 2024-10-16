import Mathlib

namespace NUMINAMATH_CALUDE_common_tangent_length_l3900_390067

/-- The length of the common tangent of two externally tangent circles -/
theorem common_tangent_length (R r : ℝ) (hR : R > 0) (hr : r > 0) :
  let d := R + r  -- distance between centers
  2 * Real.sqrt (r * R) = Real.sqrt (d^2 - (R - r)^2) :=
by sorry

end NUMINAMATH_CALUDE_common_tangent_length_l3900_390067


namespace NUMINAMATH_CALUDE_team_A_builds_22_5_meters_per_day_l3900_390033

def team_A_build_rate : ℝ → Prop := λ x => 
  (150 / x = 100 / (2 * x - 30)) ∧ (x > 0)

theorem team_A_builds_22_5_meters_per_day :
  ∃ x : ℝ, team_A_build_rate x ∧ x = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_team_A_builds_22_5_meters_per_day_l3900_390033


namespace NUMINAMATH_CALUDE_paint_intensity_problem_l3900_390072

/-- Proves that given an original paint with 45% intensity, if 25% of it is replaced with a new solution
    resulting in a 40% intensity mixture, then the intensity of the added solution is 25%. -/
theorem paint_intensity_problem (original_intensity new_intensity replaced_fraction : ℝ)
  (h1 : original_intensity = 0.45)
  (h2 : new_intensity = 0.40)
  (h3 : replaced_fraction = 0.25) :
  let remaining_fraction := 1 - replaced_fraction
  let added_intensity := (new_intensity - remaining_fraction * original_intensity) / replaced_fraction
  added_intensity = 0.25 := by
sorry

end NUMINAMATH_CALUDE_paint_intensity_problem_l3900_390072


namespace NUMINAMATH_CALUDE_equation_solution_l3900_390055

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 →
  (-15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) + 1) ↔ (x = 5/4 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3900_390055


namespace NUMINAMATH_CALUDE_buratino_arrival_time_l3900_390007

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds hours and minutes to a given time -/
def addTime (t : Time) (h : Nat) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + h * 60 + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem buratino_arrival_time :
  let departureTime : Time := { hours := 13, minutes := 40 }
  let normalJourneyTime : Real := 7.5
  let fasterJourneyTime : Real := normalJourneyTime * 4 / 5
  let timeDifference : Real := normalJourneyTime - fasterJourneyTime
  timeDifference = 1.5 →
  addTime departureTime 7 30 = { hours := 21, minutes := 10 } :=
by sorry

end NUMINAMATH_CALUDE_buratino_arrival_time_l3900_390007


namespace NUMINAMATH_CALUDE_skittle_groups_l3900_390031

/-- The number of groups formed when dividing Skittles into equal-sized groups -/
def number_of_groups (total_skittles : ℕ) (skittles_per_group : ℕ) : ℕ :=
  total_skittles / skittles_per_group

/-- Theorem stating that dividing 5929 Skittles into groups of 77 results in 77 groups -/
theorem skittle_groups : number_of_groups 5929 77 = 77 := by
  sorry

end NUMINAMATH_CALUDE_skittle_groups_l3900_390031


namespace NUMINAMATH_CALUDE_money_transfer_game_probability_l3900_390027

/-- Represents the state of the game as a triple of integers -/
def GameState := ℕ × ℕ × ℕ

/-- The initial state of the game -/
def initialState : GameState := (3, 3, 3)

/-- Represents a single transfer in the game -/
def Transfer := GameState → GameState

/-- The set of all possible transfers in the game -/
def allTransfers : Set Transfer := sorry

/-- The transition matrix for the Markov chain representing the game -/
def transitionMatrix : GameState → GameState → ℝ := sorry

/-- The steady state distribution of the Markov chain -/
def steadyStateDistribution : GameState → ℝ := sorry

theorem money_transfer_game_probability :
  steadyStateDistribution initialState = 8 / 13 := by sorry

end NUMINAMATH_CALUDE_money_transfer_game_probability_l3900_390027


namespace NUMINAMATH_CALUDE_self_inverse_matrix_l3900_390030

def A (c d : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4, -2; c, d]

theorem self_inverse_matrix (c d : ℚ) :
  A c d * A c d = 1 → c = 15/2 ∧ d = -4 := by
  sorry

end NUMINAMATH_CALUDE_self_inverse_matrix_l3900_390030


namespace NUMINAMATH_CALUDE_m_range_l3900_390081

theorem m_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ 
  (∀ x : ℝ, x^2 + m * x + 1 > 0) → 
  -2 < m ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_m_range_l3900_390081


namespace NUMINAMATH_CALUDE_gum_cost_proof_l3900_390066

/-- The cost of gum in dollars -/
def cost_in_dollars (pieces : ℕ) (cents_per_piece : ℕ) : ℚ :=
  (pieces * cents_per_piece : ℚ) / 100

/-- Proof that 500 pieces of gum at 2 cents each costs 10 dollars -/
theorem gum_cost_proof : cost_in_dollars 500 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gum_cost_proof_l3900_390066


namespace NUMINAMATH_CALUDE_probability_identical_after_rotation_l3900_390088

/-- Represents the colors available for painting the cube faces -/
inductive Color
  | Black
  | White
  | Red

/-- Represents a cube with painted faces -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- Checks if a cube satisfies the adjacent face color constraint -/
def validCube (c : Cube) : Prop := sorry

/-- Counts the number of valid cube colorings -/
def validColoringsCount : Nat := sorry

/-- Counts the number of ways cubes can be identical after rotation -/
def identicalAfterRotationCount : Nat := sorry

/-- Theorem stating the probability of three cubes being identical after rotation -/
theorem probability_identical_after_rotation :
  (identicalAfterRotationCount : ℚ) / (validColoringsCount ^ 3 : ℚ) = 1 / 45 := by sorry

end NUMINAMATH_CALUDE_probability_identical_after_rotation_l3900_390088


namespace NUMINAMATH_CALUDE_equation_root_conditions_l3900_390093

theorem equation_root_conditions (a : ℝ) : 
  (∃ x > 0, |x| = a*x - a) ∧ 
  (∀ x < 0, |x| ≠ a*x - a) → 
  a > 1 ∨ a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_conditions_l3900_390093


namespace NUMINAMATH_CALUDE_minimum_advantageous_discount_l3900_390096

theorem minimum_advantageous_discount (n : ℕ) : n = 29 ↔ 
  (∀ m : ℕ, m < n → 
    ((1 - m / 100 : ℝ) ≥ (1 - 0.12)^2 ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.08)^2 * (1 - 0.09) ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.20) * (1 - 0.10))) ∧
  ((1 - n / 100 : ℝ) < (1 - 0.12)^2 ∧
   (1 - n / 100 : ℝ) < (1 - 0.08)^2 * (1 - 0.09) ∧
   (1 - n / 100 : ℝ) < (1 - 0.20) * (1 - 0.10)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_advantageous_discount_l3900_390096


namespace NUMINAMATH_CALUDE_building_floors_l3900_390079

theorem building_floors (total_height : ℝ) (regular_floor_height : ℝ) (extra_height : ℝ) :
  total_height = 61 ∧
  regular_floor_height = 3 ∧
  extra_height = 0.5 →
  ∃ (n : ℕ), n = 20 ∧
    total_height = regular_floor_height * (n - 2 : ℝ) + (regular_floor_height + extra_height) * 2 :=
by
  sorry


end NUMINAMATH_CALUDE_building_floors_l3900_390079


namespace NUMINAMATH_CALUDE_homeless_families_donation_l3900_390018

theorem homeless_families_donation (total spent first_set second_set : ℝ) 
  (h1 : total = 900)
  (h2 : first_set = 325)
  (h3 : second_set = 260) :
  total - (first_set + second_set) = 315 := by
sorry

end NUMINAMATH_CALUDE_homeless_families_donation_l3900_390018


namespace NUMINAMATH_CALUDE_cookie_sales_ratio_l3900_390047

theorem cookie_sales_ratio : 
  ∀ (goal : ℕ) (first third fourth fifth left : ℕ),
    goal ≥ 150 →
    first = 5 →
    third = 10 →
    fifth = 10 →
    left = 75 →
    goal - left = first + 4 * first + third + fourth + fifth →
    fourth / third = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_ratio_l3900_390047


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3900_390042

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmph = 72 →
  crossing_time = 13.998880089592832 →
  ∃ (bridge_length : ℝ), 
    (169.97 < bridge_length) ∧ 
    (bridge_length < 169.99) ∧
    (bridge_length = train_speed_kmph * (1000 / 3600) * crossing_time - train_length) :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3900_390042


namespace NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l3900_390036

theorem systematic_sampling_smallest_number 
  (total_classes : Nat) 
  (selected_classes : Nat) 
  (sum_of_selected : Nat) : 
  total_classes = 30 → 
  selected_classes = 6 → 
  sum_of_selected = 87 → 
  (total_classes / selected_classes : Nat) = 5 → 
  ∃ x : Nat, 
    x + (x + 5) + (x + 10) + (x + 15) + (x + 20) + (x + 25) = sum_of_selected ∧ 
    x = 2 ∧ 
    (∀ y : Nat, y + (y + 5) + (y + 10) + (y + 15) + (y + 20) + (y + 25) = sum_of_selected → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l3900_390036


namespace NUMINAMATH_CALUDE_parallelogram_properties_l3900_390080

-- Define the points
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define the vector operation
def vectorOp : ℝ × ℝ := (3 * AB.1 - 2 * AC.1 + BC.1, 3 * AB.2 - 2 * AC.2 + BC.2)

-- Define point D
def D : ℝ × ℝ := (A.1 + BC.1, A.2 + BC.2)

theorem parallelogram_properties :
  vectorOp = (0, 2) ∧ D = (2, -1) := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_properties_l3900_390080


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3900_390091

/-- In triangle ABC, prove that given specific conditions, angle A and the area of the triangle can be determined. -/
theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given condition
  2 * Real.cos A * (b * Real.cos C + c * Real.cos B) = a →
  -- Additional conditions
  a = Real.sqrt 7 →
  b + c = 5 →
  -- Conclusions
  A = π / 3 ∧ 
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3900_390091


namespace NUMINAMATH_CALUDE_apples_per_box_is_correct_l3900_390008

/-- The number of apples packed in a box -/
def apples_per_box : ℕ := 40

/-- The number of boxes packed per day in the first week -/
def boxes_per_day : ℕ := 50

/-- The number of fewer apples packed per day in the second week -/
def fewer_apples_per_day : ℕ := 500

/-- The total number of apples packed in two weeks -/
def total_apples : ℕ := 24500

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem apples_per_box_is_correct :
  (boxes_per_day * days_per_week * apples_per_box) +
  ((boxes_per_day * apples_per_box - fewer_apples_per_day) * days_per_week) = total_apples :=
by sorry

end NUMINAMATH_CALUDE_apples_per_box_is_correct_l3900_390008


namespace NUMINAMATH_CALUDE_arrangement_theorems_l3900_390087

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 4

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of arrangements with boys together -/
def arrangements_boys_together : ℕ := 720

/-- The number of arrangements with alternating genders -/
def arrangements_alternating : ℕ := 144

/-- The number of arrangements with person A left of person B -/
def arrangements_A_left_of_B : ℕ := 2520

theorem arrangement_theorems :
  (arrangements_boys_together = 720) ∧
  (arrangements_alternating = 144) ∧
  (arrangements_A_left_of_B = 2520) := by sorry

end NUMINAMATH_CALUDE_arrangement_theorems_l3900_390087


namespace NUMINAMATH_CALUDE_fruit_tree_problem_l3900_390053

theorem fruit_tree_problem (initial_apples : ℕ) (pick_ratio : ℚ) : 
  initial_apples = 180 →
  pick_ratio = 3 / 5 →
  ∃ (initial_plums : ℕ),
    initial_plums * 3 = initial_apples ∧
    (initial_apples + initial_plums) * (1 - pick_ratio) = 96 := by
  sorry

end NUMINAMATH_CALUDE_fruit_tree_problem_l3900_390053


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l3900_390043

theorem diet_soda_bottles (total : ℕ) (regular : ℕ) (diet : ℕ) : 
  total = 30 → regular = 28 → diet = total - regular → diet = 2 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l3900_390043


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l3900_390099

theorem largest_triangle_perimeter (a b x : ℕ) : 
  a = 8 → b = 11 → x ∈ Set.Icc 4 18 → 
  (∀ y : ℕ, y ∈ Set.Icc 4 18 → a + b + y ≤ a + b + x) →
  a + b + x = 37 := by
sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l3900_390099


namespace NUMINAMATH_CALUDE_two_white_balls_possible_l3900_390038

/-- Represents the contents of the box -/
structure BoxContents where
  black : ℕ
  white : ℕ

/-- Represents a single replacement rule -/
inductive ReplacementRule
  | ThreeBlack
  | TwoBlackOneWhite
  | OneBlackTwoWhite
  | ThreeWhite

/-- Applies a single replacement rule to the box contents -/
def applyRule (contents : BoxContents) (rule : ReplacementRule) : BoxContents :=
  match rule with
  | ReplacementRule.ThreeBlack => 
      ⟨contents.black - 2, contents.white⟩
  | ReplacementRule.TwoBlackOneWhite => 
      ⟨contents.black - 1, contents.white⟩
  | ReplacementRule.OneBlackTwoWhite => 
      ⟨contents.black - 1, contents.white⟩
  | ReplacementRule.ThreeWhite => 
      ⟨contents.black + 1, contents.white - 2⟩

/-- Applies a sequence of replacement rules to the box contents -/
def applyRules (initial : BoxContents) (rules : List ReplacementRule) : BoxContents :=
  rules.foldl applyRule initial

theorem two_white_balls_possible : 
  ∃ (rules : List ReplacementRule), 
    (applyRules ⟨100, 100⟩ rules).white = 2 :=
  sorry


end NUMINAMATH_CALUDE_two_white_balls_possible_l3900_390038


namespace NUMINAMATH_CALUDE_interval_equivalence_l3900_390032

theorem interval_equivalence (x : ℝ) : 
  (3/4 < x ∧ x < 4/5) ↔ (3 < 5*x + 1 ∧ 5*x + 1 < 5) ∧ (3 < 4*x ∧ 4*x < 5) :=
by sorry

end NUMINAMATH_CALUDE_interval_equivalence_l3900_390032


namespace NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3900_390025

theorem largest_gold_coins_distribution (n : ℕ) : 
  n < 100 ∧ 
  n % 13 = 3 ∧ 
  (∀ m : ℕ, m < 100 ∧ m % 13 = 3 → m ≤ n) → 
  n = 94 := by
sorry

end NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3900_390025


namespace NUMINAMATH_CALUDE_distance_apex_to_sphere_top_l3900_390071

-- Define the cone parameters
def base_radius : ℝ := 10
def cone_height : ℝ := 30

-- Define the theorem
theorem distance_apex_to_sphere_top :
  let slant_height : ℝ := Real.sqrt (base_radius^2 + cone_height^2)
  let sphere_radius : ℝ := (cone_height * base_radius) / (slant_height + base_radius)
  cone_height - sphere_radius = 15 * Real.sqrt 10 + 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_apex_to_sphere_top_l3900_390071


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3900_390037

theorem increase_by_percentage (initial : ℕ) (percentage : ℕ) (result : ℕ) : 
  initial = 150 → percentage = 40 → result = initial + (initial * percentage) / 100 → result = 210 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3900_390037


namespace NUMINAMATH_CALUDE_hoseok_social_studies_score_l3900_390011

/-- Represents Hoseok's test scores -/
structure HoseokScores where
  average_three : ℝ  -- Average score of Korean, English, and Science
  average_four : ℝ   -- Average score after including Social studies
  social_studies : ℝ -- Score of Social studies test

/-- Theorem stating that given Hoseok's average scores, his Social studies score must be 93 -/
theorem hoseok_social_studies_score (scores : HoseokScores)
  (h1 : scores.average_three = 89)
  (h2 : scores.average_four = 90) :
  scores.social_studies = 93 := by
  sorry

#check hoseok_social_studies_score

end NUMINAMATH_CALUDE_hoseok_social_studies_score_l3900_390011


namespace NUMINAMATH_CALUDE_betty_beads_l3900_390004

/-- Given that Betty has 3 red beads for every 2 blue beads and she has 20 blue beads,
    prove that Betty has 30 red beads. -/
theorem betty_beads (red_blue_ratio : ℚ) (blue_beads : ℕ) (red_beads : ℕ) : 
  red_blue_ratio = 3 / 2 →
  blue_beads = 20 →
  red_beads = red_blue_ratio * blue_beads →
  red_beads = 30 := by
sorry

end NUMINAMATH_CALUDE_betty_beads_l3900_390004


namespace NUMINAMATH_CALUDE_parking_problem_l3900_390089

/-- Calculates the number of vehicles that can still park in a lot -/
def vehiclesCanPark (totalSpaces : ℕ) (caravanSpaces : ℕ) (caravansParked : ℕ) : ℕ :=
  totalSpaces - (caravanSpaces * caravansParked)

/-- Theorem: Given the problem conditions, 24 vehicles can still park -/
theorem parking_problem :
  vehiclesCanPark 30 2 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_parking_problem_l3900_390089


namespace NUMINAMATH_CALUDE_f_sum_difference_equals_two_l3900_390034

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem f_sum_difference_equals_two :
  f 2016 + (deriv f) 2016 + f (-2016) - (deriv f) (-2016) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_difference_equals_two_l3900_390034


namespace NUMINAMATH_CALUDE_target_shopping_expense_l3900_390077

/-- The total amount spent by Christy and Tanya at Target -/
def total_spent (tanya_face_moisturizer_price : ℕ) 
                (tanya_face_moisturizer_count : ℕ)
                (tanya_body_lotion_price : ℕ)
                (tanya_body_lotion_count : ℕ) : ℕ :=
  let tanya_total := tanya_face_moisturizer_price * tanya_face_moisturizer_count + 
                     tanya_body_lotion_price * tanya_body_lotion_count
  tanya_total * 3

theorem target_shopping_expense :
  total_spent 50 2 60 4 = 1020 :=
sorry

end NUMINAMATH_CALUDE_target_shopping_expense_l3900_390077


namespace NUMINAMATH_CALUDE_max_player_score_l3900_390098

theorem max_player_score (total_players : ℕ) (total_points : ℕ) (min_points : ℕ) 
  (h1 : total_players = 12)
  (h2 : total_points = 100)
  (h3 : min_points = 7)
  (h4 : ∀ player, player ≥ min_points) :
  ∃ max_score : ℕ, max_score = 23 ∧ 
  (∀ player_score : ℕ, player_score ≤ max_score) ∧
  (∃ player : ℕ, player = max_score) ∧
  (total_points = (total_players - 1) * min_points + max_score) :=
by sorry

end NUMINAMATH_CALUDE_max_player_score_l3900_390098


namespace NUMINAMATH_CALUDE_property_sale_gain_l3900_390045

/-- Represents the sale of two properties with given selling prices and percentage changes --/
def PropertySale (house_price store_price : ℝ) (house_loss store_gain : ℝ) : Prop :=
  ∃ (house_cost store_cost : ℝ),
    house_price = house_cost * (1 - house_loss) ∧
    store_price = store_cost * (1 + store_gain) ∧
    house_price + store_price - (house_cost + store_cost) = 1000

/-- Theorem stating that the given property sale results in a $1000 gain --/
theorem property_sale_gain :
  PropertySale 15000 18000 0.25 0.50 := by
  sorry

#check property_sale_gain

end NUMINAMATH_CALUDE_property_sale_gain_l3900_390045


namespace NUMINAMATH_CALUDE_total_volume_of_prisms_l3900_390022

theorem total_volume_of_prisms (length width height : ℝ) (num_prisms : ℕ) 
  (h1 : length = 5)
  (h2 : width = 3)
  (h3 : height = 6)
  (h4 : num_prisms = 4) :
  length * width * height * num_prisms = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_prisms_l3900_390022


namespace NUMINAMATH_CALUDE_marble_probability_theorem_l3900_390094

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability of drawing 4 marbles of the same color -/
def probSameColor (counts : MarbleCounts) : ℚ :=
  let total := counts.red + counts.white + counts.blue + counts.green
  let probRed := Nat.choose counts.red 4 / Nat.choose total 4
  let probWhite := Nat.choose counts.white 4 / Nat.choose total 4
  let probBlue := Nat.choose counts.blue 4 / Nat.choose total 4
  let probGreen := Nat.choose counts.green 4 / Nat.choose total 4
  probRed + probWhite + probBlue + probGreen

theorem marble_probability_theorem (counts : MarbleCounts) 
    (h : counts = ⟨6, 7, 8, 9⟩) : 
    probSameColor counts = 82 / 9135 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_theorem_l3900_390094


namespace NUMINAMATH_CALUDE_sum_six_terms_eq_neg_24_l3900_390065

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  first_term : a 1 = 1
  common_diff : ∀ n : ℕ, a (n + 1) = a n + d
  d_nonzero : d ≠ 0
  geometric_subseq : (a 3 / a 2) = (a 6 / a 3)

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- The main theorem -/
theorem sum_six_terms_eq_neg_24 (seq : ArithmeticSequence) :
  sum_n_terms seq 6 = -24 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_terms_eq_neg_24_l3900_390065


namespace NUMINAMATH_CALUDE_definite_integral_semicircle_l3900_390050

theorem definite_integral_semicircle (f : ℝ → ℝ) (r : ℝ) :
  (∀ x, f x = Real.sqrt (r^2 - x^2)) →
  r > 0 →
  ∫ x in (0)..(r), f x = (π * r^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_semicircle_l3900_390050


namespace NUMINAMATH_CALUDE_max_a4_in_geometric_sequence_l3900_390003

theorem max_a4_in_geometric_sequence (a : ℕ → ℝ) :
  (∀ n : ℕ, a n > 0) →  -- positive sequence
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence
  a 3 + a 5 = 4 →  -- given condition
  ∀ x : ℝ, a 4 ≤ x → x ≤ 2  -- maximum value of a_4 is 2
:= by sorry

end NUMINAMATH_CALUDE_max_a4_in_geometric_sequence_l3900_390003


namespace NUMINAMATH_CALUDE_jackson_money_l3900_390016

theorem jackson_money (williams_money : ℝ) (h1 : williams_money > 0) 
  (h2 : williams_money + 5 * williams_money = 150) : 
  5 * williams_money = 125 := by
sorry

end NUMINAMATH_CALUDE_jackson_money_l3900_390016


namespace NUMINAMATH_CALUDE_sequence_sum_l3900_390024

theorem sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 9*x₂ + 25*x₃ + 49*x₄ + 81*x₅ + 121*x₆ + 169*x₇ = 2)
  (eq2 : 9*x₁ + 25*x₂ + 49*x₃ + 81*x₄ + 121*x₅ + 169*x₆ + 225*x₇ = 24)
  (eq3 : 25*x₁ + 49*x₂ + 81*x₃ + 121*x₄ + 169*x₅ + 225*x₆ + 289*x₇ = 246) :
  49*x₁ + 81*x₂ + 121*x₃ + 169*x₄ + 225*x₅ + 289*x₆ + 361*x₇ = 668 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3900_390024


namespace NUMINAMATH_CALUDE_geometric_sequence_product_constant_geometric_sequence_product_16_l3900_390062

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of two terms equidistant from the beginning and end of the sequence is constant -/
theorem geometric_sequence_product_constant {a : ℕ → ℝ} (h : GeometricSequence a) :
  ∀ m n k : ℕ, m < n → a m * a n = a (m + k) * a (n - k) := by sorry

theorem geometric_sequence_product_16 (a : ℕ → ℝ) (h : GeometricSequence a) 
  (h2 : a 4 * a 8 = 16) : a 2 * a 10 = 16 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_constant_geometric_sequence_product_16_l3900_390062


namespace NUMINAMATH_CALUDE_third_fraction_is_two_ninths_l3900_390058

-- Define a fraction type
structure Fraction where
  numerator : ℤ
  denominator : ℕ
  denominator_nonzero : denominator ≠ 0

-- Define the HCF function for fractions
def hcf_fractions (f1 f2 f3 : Fraction) : ℚ :=
  sorry

-- Theorem statement
theorem third_fraction_is_two_ninths
  (f1 : Fraction)
  (f2 : Fraction)
  (f3 : Fraction)
  (h1 : f1 = ⟨2, 3, sorry⟩)
  (h2 : f2 = ⟨4, 9, sorry⟩)
  (h3 : hcf_fractions f1 f2 f3 = 1 / 9) :
  f3 = ⟨2, 9, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_third_fraction_is_two_ninths_l3900_390058


namespace NUMINAMATH_CALUDE_cab_ride_cost_total_cost_is_6720_l3900_390083

/-- Calculate the total cost of cab rides for a one-week event with carpooling --/
theorem cab_ride_cost (off_peak_rate : ℚ) (peak_rate : ℚ) (distance : ℚ) 
  (days : ℕ) (participants : ℕ) (discount : ℚ) : ℚ :=
  let daily_cost := off_peak_rate * distance + peak_rate * distance
  let total_cost := daily_cost * days
  let discounted_cost := total_cost * (1 - discount)
  discounted_cost

/-- Prove that the total cost for all participants is $6720 --/
theorem total_cost_is_6720 : 
  cab_ride_cost (5/2) (7/2) 200 7 4 (1/5) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_cab_ride_cost_total_cost_is_6720_l3900_390083


namespace NUMINAMATH_CALUDE_pancake_fundraiser_l3900_390019

/-- The civic league's pancake breakfast fundraiser --/
theorem pancake_fundraiser 
  (pancake_price : ℝ) 
  (bacon_price : ℝ) 
  (pancake_stacks_sold : ℕ) 
  (bacon_slices_sold : ℕ) 
  (h1 : pancake_price = 4)
  (h2 : bacon_price = 2)
  (h3 : pancake_stacks_sold = 60)
  (h4 : bacon_slices_sold = 90) :
  pancake_price * pancake_stacks_sold + bacon_price * bacon_slices_sold = 420 :=
by sorry

end NUMINAMATH_CALUDE_pancake_fundraiser_l3900_390019


namespace NUMINAMATH_CALUDE_final_amount_is_301_l3900_390048

def initial_quarters : ℕ := 7
def initial_dimes : ℕ := 3
def initial_nickels : ℕ := 5
def initial_pennies : ℕ := 12
def initial_half_dollars : ℕ := 3

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.1
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01
def half_dollar_value : ℚ := 0.5

def lose_one_of_each (q d n p h : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  (q - 1, d - 1, n - 1, p - 1, h - 1)

def exchange_nickels_for_dimes (n d : ℕ) : ℕ × ℕ :=
  (n - 3, d + 2)

def exchange_half_dollar (h q d : ℕ) : ℕ × ℕ × ℕ :=
  (h - 1, q + 1, d + 2)

def calculate_total (q d n p h : ℕ) : ℚ :=
  q * quarter_value + d * dime_value + n * nickel_value + 
  p * penny_value + h * half_dollar_value

theorem final_amount_is_301 :
  let (q1, d1, n1, p1, h1) := lose_one_of_each initial_quarters initial_dimes initial_nickels initial_pennies initial_half_dollars
  let (n2, d2) := exchange_nickels_for_dimes n1 d1
  let (h2, q2, d3) := exchange_half_dollar h1 q1 d2
  calculate_total q2 d3 n2 p1 h2 = 3.01 := by sorry

end NUMINAMATH_CALUDE_final_amount_is_301_l3900_390048


namespace NUMINAMATH_CALUDE_problem_solution_l3900_390054

theorem problem_solution (x y : ℝ) (hx : x = 1 - Real.sqrt 2) (hy : y = 1 + Real.sqrt 2) : 
  x^2 + 3*x*y + y^2 = 3 ∧ y/x - x/y = -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3900_390054


namespace NUMINAMATH_CALUDE_twelve_person_tournament_matches_l3900_390086

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a 12-person round-robin tournament, the number of matches is 66 -/
theorem twelve_person_tournament_matches : num_matches 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_twelve_person_tournament_matches_l3900_390086


namespace NUMINAMATH_CALUDE_junghyeon_stickers_l3900_390085

/-- Given a total of 25 stickers shared between Junghyeon and Yejin, 
    where Junghyeon has 1 more sticker than twice Yejin's, 
    prove that Junghyeon will have 17 stickers. -/
theorem junghyeon_stickers : 
  ∀ (junghyeon_stickers yejin_stickers : ℕ),
  junghyeon_stickers + yejin_stickers = 25 →
  junghyeon_stickers = 2 * yejin_stickers + 1 →
  junghyeon_stickers = 17 := by
sorry

end NUMINAMATH_CALUDE_junghyeon_stickers_l3900_390085


namespace NUMINAMATH_CALUDE_divisibility_theorem_l3900_390061

def group_digits (n : ℕ) : List ℕ :=
  sorry

def alternating_sum (groups : List ℕ) : ℤ :=
  sorry

theorem divisibility_theorem (A : ℕ) :
  let groups := group_digits A
  let B := alternating_sum groups
  (7 ∣ (A - B) ∧ 11 ∣ (A - B) ∧ 13 ∣ (A - B)) ↔ (7 ∣ A ∧ 11 ∣ A ∧ 13 ∣ A) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l3900_390061


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3900_390005

theorem quadratic_inequality_solution (a c : ℝ) (h : ∀ x, (a * x^2 + 5 * x + c > 0) ↔ (1/3 < x ∧ x < 1/2)) :
  (a = -6 ∧ c = -1) ∧
  (∀ b : ℝ, 
    (∀ x, (a * x^2 + (a * c + b) * x + b * c ≥ 0) ↔ 
      ((b > 6 ∧ 1 ≤ x ∧ x ≤ b/6) ∨
       (b = 6 ∧ x = 1) ∨
       (b < 6 ∧ b/6 ≤ x ∧ x ≤ 1)))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3900_390005


namespace NUMINAMATH_CALUDE_cherry_revenue_is_180_l3900_390073

/-- Calculates the revenue from cherry pies given the total number of pies,
    the ratio of pie types, and the price of a cherry pie. -/
def cherry_pie_revenue (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) (cherry_price : ℕ) : ℕ :=
  let total_ratio := apple_ratio + blueberry_ratio + cherry_ratio
  let cherry_pies := (total_pies * cherry_ratio) / total_ratio
  cherry_pies * cherry_price

/-- Theorem stating that given 36 total pies with a ratio of 3:2:5 for apple:blueberry:cherry pies,
    and a price of $10 per cherry pie, the total revenue from cherry pies is $180. -/
theorem cherry_revenue_is_180 :
  cherry_pie_revenue 36 3 2 5 10 = 180 := by
  sorry

end NUMINAMATH_CALUDE_cherry_revenue_is_180_l3900_390073


namespace NUMINAMATH_CALUDE_q_minus_p_equals_zero_l3900_390051

def P : Set ℕ := {1, 2, 3, 4, 5}
def Q : Set ℕ := {0, 2, 3}

def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem q_minus_p_equals_zero : set_difference Q P = {0} := by sorry

end NUMINAMATH_CALUDE_q_minus_p_equals_zero_l3900_390051


namespace NUMINAMATH_CALUDE_percentage_not_covering_politics_l3900_390026

/-- Represents the percentage of reporters covering local politics in country X -/
def local_politics_coverage : ℝ := 28

/-- Represents the percentage of political reporters not covering local politics in country X -/
def non_local_politics_coverage : ℝ := 30

/-- Theorem stating that 60% of reporters do not cover politics given the conditions -/
theorem percentage_not_covering_politics :
  let total_political_coverage := local_politics_coverage / (1 - non_local_politics_coverage / 100)
  100 - total_political_coverage = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_covering_politics_l3900_390026


namespace NUMINAMATH_CALUDE_fraction_simplification_l3900_390068

theorem fraction_simplification (x : ℝ) (h : 2 * x - 3 ≠ 0) :
  (18 * x^4 - 9 * x^3 - 86 * x^2 + 16 * x + 96) / (18 * x^4 - 63 * x^3 + 22 * x^2 + 112 * x - 96) = (2 * x + 3) / (2 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3900_390068


namespace NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l3900_390090

theorem smallest_third_term_geometric_progression (a b c : ℝ) : 
  a = 5 ∧ 
  b - a = c - b ∧ 
  (5 * (c + 27) = (b + 9)^2) →
  c + 27 ≥ 16 - 4 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l3900_390090


namespace NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l3900_390000

theorem square_sum_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l3900_390000


namespace NUMINAMATH_CALUDE_tomato_theorem_l3900_390082

def tomato_problem (initial_tomatoes : ℕ) : ℕ :=
  let after_first_birds := initial_tomatoes - initial_tomatoes / 3
  let after_second_birds := after_first_birds - after_first_birds / 2
  let final_tomatoes := after_second_birds + (after_second_birds + 1) / 2
  final_tomatoes

theorem tomato_theorem : tomato_problem 21 = 11 := by
  sorry

end NUMINAMATH_CALUDE_tomato_theorem_l3900_390082


namespace NUMINAMATH_CALUDE_students_in_both_competitions_l3900_390049

theorem students_in_both_competitions 
  (total_students : ℕ) 
  (math_students : ℕ) 
  (physics_students : ℕ) 
  (no_competition_students : ℕ) 
  (h1 : total_students = 45) 
  (h2 : math_students = 32) 
  (h3 : physics_students = 28) 
  (h4 : no_competition_students = 5) :
  total_students - no_competition_students - 
  (math_students + physics_students - total_students + no_competition_students) = 20 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_competitions_l3900_390049


namespace NUMINAMATH_CALUDE_inequality_proof_l3900_390006

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 → n^n = n^n :=
by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l3900_390006


namespace NUMINAMATH_CALUDE_function_equation_implies_constant_l3900_390060

/-- A function satisfying the given functional equation is constant -/
theorem function_equation_implies_constant
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, 2 * f x = f (x + y) + f (x + 2 * y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_equation_implies_constant_l3900_390060


namespace NUMINAMATH_CALUDE_necessary_condition_propositions_l3900_390075

-- Definition for necessary condition
def is_necessary_condition (p q : Prop) : Prop :=
  q → p

-- Proposition A
def prop_a (x y : ℝ) : Prop :=
  is_necessary_condition (x^2 > y^2) (x > y)

-- Proposition B
def prop_b (x : ℝ) : Prop :=
  is_necessary_condition (x > 5) (x > 10)

-- Proposition C
def prop_c (a b c : ℝ) : Prop :=
  is_necessary_condition (a * c = b * c) (a = b)

-- Proposition D
def prop_d (x y : ℝ) : Prop :=
  is_necessary_condition (2 * x + 1 = 2 * y + 1) (x = y)

-- Theorem stating which propositions have p as a necessary condition for q
theorem necessary_condition_propositions :
  (∃ x y : ℝ, ¬(prop_a x y)) ∧
  (∀ x : ℝ, prop_b x) ∧
  (∀ a b c : ℝ, c ≠ 0 → prop_c a b c) ∧
  (∀ x y : ℝ, prop_d x y) :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_propositions_l3900_390075


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3900_390057

def a (n : ℕ) : ℤ := 3 * n - 5

theorem arithmetic_sequence_proof :
  (∀ n : ℕ, a (n + 1) - a n = 3) ∧
  (a 1 = -2) ∧
  (∀ n : ℕ, a (n + 1) - a n = 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3900_390057


namespace NUMINAMATH_CALUDE_cuboid_breadth_l3900_390063

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: The breadth of a cuboid with given length, height, and surface area -/
theorem cuboid_breadth (l h area : ℝ) (hl : l = 8) (hh : h = 9) (harea : area = 432) :
  ∃ b : ℝ, cuboidSurfaceArea l b h = area ∧ b = 144 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_breadth_l3900_390063


namespace NUMINAMATH_CALUDE_no_real_solutions_l3900_390052

theorem no_real_solutions : ¬ ∃ x : ℝ, (5*x)/(x^2 + 2*x + 4) + (6*x)/(x^2 - 4*x + 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3900_390052


namespace NUMINAMATH_CALUDE_payment_calculation_l3900_390070

/-- Represents a store's pricing and promotion options for suits and ties. -/
structure StorePricing where
  suit_price : ℕ
  tie_price : ℕ
  option1 : ℕ → ℕ  -- Function representing the cost for Option 1
  option2 : ℕ → ℕ  -- Function representing the cost for Option 2

/-- Calculates the payment for a customer buying suits and ties under different options. -/
def calculate_payment (pricing : StorePricing) (suits : ℕ) (ties : ℕ) : ℕ × ℕ :=
  (pricing.option1 ties, pricing.option2 ties)

/-- Theorem stating the correct calculation of payments for the given problem. -/
theorem payment_calculation (x : ℕ) (h : x > 20) :
  let pricing := StorePricing.mk 1000 200
    (fun ties => 20000 + 200 * (ties - 20))
    (fun ties => (20 * 1000 + ties * 200) * 9 / 10)
  (calculate_payment pricing 20 x).1 = 200 * x + 16000 ∧
  (calculate_payment pricing 20 x).2 = 180 * x + 18000 := by
  sorry

end NUMINAMATH_CALUDE_payment_calculation_l3900_390070


namespace NUMINAMATH_CALUDE_min_players_on_team_l3900_390064

theorem min_players_on_team (total_score : ℕ) (min_score max_score : ℕ) : 
  total_score = 100 →
  min_score = 7 →
  max_score = 23 →
  (∃ (num_players : ℕ), 
    num_players ≥ 1 ∧
    (∀ (player_scores : List ℕ), 
      player_scores.length = num_players →
      (∀ score ∈ player_scores, min_score ≤ score ∧ score ≤ max_score) →
      player_scores.sum = total_score) ∧
    (∀ (n : ℕ), n < num_players →
      ¬∃ (player_scores : List ℕ),
        player_scores.length = n ∧
        (∀ score ∈ player_scores, min_score ≤ score ∧ score ≤ max_score) ∧
        player_scores.sum = total_score)) →
  (∃ (num_players : ℕ), num_players = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_min_players_on_team_l3900_390064


namespace NUMINAMATH_CALUDE_typing_time_is_35_minutes_l3900_390020

/-- Represents the typing scenario with given conditions -/
structure TypingScenario where
  barbaraMaxSpeed : ℕ
  barbaraInjuryReduction : ℕ
  barbaraFatigueReduction : ℕ
  barbaraFatigueInterval : ℕ
  jimSpeed : ℕ
  jimTime : ℕ
  monicaSpeed : ℕ
  monicaTime : ℕ
  breakDuration : ℕ
  breakInterval : ℕ
  documentLength : ℕ

/-- Calculates the minimum time required to type the document -/
def minTypingTime (scenario : TypingScenario) : ℕ :=
  sorry

/-- Theorem stating that the minimum typing time for the given scenario is 35 minutes -/
theorem typing_time_is_35_minutes (scenario : TypingScenario) 
  (h1 : scenario.barbaraMaxSpeed = 212)
  (h2 : scenario.barbaraInjuryReduction = 40)
  (h3 : scenario.barbaraFatigueReduction = 5)
  (h4 : scenario.barbaraFatigueInterval = 15)
  (h5 : scenario.jimSpeed = 100)
  (h6 : scenario.jimTime = 20)
  (h7 : scenario.monicaSpeed = 150)
  (h8 : scenario.monicaTime = 10)
  (h9 : scenario.breakDuration = 5)
  (h10 : scenario.breakInterval = 25)
  (h11 : scenario.documentLength = 3440) :
  minTypingTime scenario = 35 :=
by sorry

end NUMINAMATH_CALUDE_typing_time_is_35_minutes_l3900_390020


namespace NUMINAMATH_CALUDE_smallest_n_for_divisibility_l3900_390084

theorem smallest_n_for_divisibility (x y z : ℕ+) 
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3) :
  (∀ n : ℕ, n < 13 → ¬(x * y * z ∣ (x + y + z)^n)) ∧ 
  (x * y * z ∣ (x + y + z)^13) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisibility_l3900_390084


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3900_390028

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i : ℂ) / (3 + i) = (1 : ℂ) / 10 + (3 : ℂ) / 10 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3900_390028


namespace NUMINAMATH_CALUDE_max_true_statements_l3900_390017

theorem max_true_statements (y : ℝ) : 
  let statements := [
    (0 < y^3 ∧ y^3 < 2),
    (y^3 > 2),
    (-2 < y ∧ y < 0),
    (0 < y ∧ y < 2),
    (0 < y - y^3 ∧ y - y^3 < 2)
  ]
  ∀ (s : Finset (Fin 5)), (∀ i ∈ s, statements[i]) → s.card ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l3900_390017


namespace NUMINAMATH_CALUDE_line_parameterization_l3900_390095

/-- Given a line y = 2x - 15 parameterized by (x, y) = (g(t), 10t + 5), prove that g(t) = 5t + 10 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t : ℝ, 10 * t + 5 = 2 * (g t) - 15) → 
  (∀ t : ℝ, g t = 5 * t + 10) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3900_390095


namespace NUMINAMATH_CALUDE_digit_permutation_theorem_l3900_390010

/-- A k-digit number -/
def kDigitNumber (k : ℕ) := { n : ℕ // n < 10^k ∧ n ≥ 10^(k-1) }

/-- Inserting a k-digit number between two adjacent digits of another number -/
def insertNumber (n : ℕ) (k : ℕ) (a : kDigitNumber k) : ℕ := sorry

/-- Permutation of digits -/
def isPermutationOf (a b : ℕ) : Prop := sorry

theorem digit_permutation_theorem (k : ℕ) (p : ℕ) (A B : kDigitNumber k) :
  Prime p →
  p > 10^k →
  (∀ m : ℕ, m % p = 0 → (insertNumber m k A) % p = 0) →
  (∃ n : ℕ, (insertNumber n k A) % p = 0 ∧ (insertNumber (insertNumber n k A) k B) % p = 0) →
  isPermutationOf A.val B.val := by sorry

end NUMINAMATH_CALUDE_digit_permutation_theorem_l3900_390010


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_f_values_l3900_390023

theorem infinite_primes_dividing_f_values
  (f : ℕ+ → ℕ+)
  (h_non_constant : ∃ a b : ℕ+, f a ≠ f b)
  (h_divides : ∀ a b : ℕ+, a ≠ b → (a - b) ∣ (f a - f b)) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ c : ℕ+, p ∣ f c} :=
sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_f_values_l3900_390023


namespace NUMINAMATH_CALUDE_inequality_proof_l3900_390046

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) > Real.sqrt 3 * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3900_390046


namespace NUMINAMATH_CALUDE_solution_inequality1_solution_system_inequalities_l3900_390044

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x + 1 > 2*x - 3

def inequality2 (x : ℝ) : Prop := 2*x - 1 > x

def inequality3 (x : ℝ) : Prop := (x + 5) / 2 - x ≥ 1

-- Theorem for the first inequality
theorem solution_inequality1 : 
  {x : ℝ | inequality1 x} = {x : ℝ | x < 4} :=
sorry

-- Theorem for the system of inequalities
theorem solution_system_inequalities :
  {x : ℝ | inequality2 x ∧ inequality3 x} = {x : ℝ | 1 < x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_inequality1_solution_system_inequalities_l3900_390044


namespace NUMINAMATH_CALUDE_integer_solutions_l3900_390059

theorem integer_solutions (a : ℤ) : 
  (∃ b c : ℤ, ∀ x : ℤ, (x - a) * (x - 12) + 1 = (x + b) * (x + c)) ↔ 
  (a = 10 ∨ a = 14) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_l3900_390059


namespace NUMINAMATH_CALUDE_fiftieth_term_is_five_sixths_l3900_390041

/-- Represents a term in the sequence as a pair of natural numbers (numerator, denominator) -/
def SequenceTerm := ℕ × ℕ

/-- Generates the nth term of the sequence -/
def nthTerm (n : ℕ) : SequenceTerm :=
  sorry

/-- Converts a SequenceTerm to a rational number -/
def toRational (term : SequenceTerm) : ℚ :=
  sorry

/-- The main theorem stating that the 50th term of the sequence is 5/6 -/
theorem fiftieth_term_is_five_sixths :
  toRational (nthTerm 50) = 5 / 6 :=
sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_five_sixths_l3900_390041


namespace NUMINAMATH_CALUDE_painter_problem_l3900_390039

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculates the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Proves that for the given scenario, the time to paint the remaining rooms is 32 hours. -/
theorem painter_problem :
  let total_rooms : ℕ := 9
  let time_per_room : ℕ := 8
  let painted_rooms : ℕ := 5
  time_to_paint_remaining total_rooms time_per_room painted_rooms = 32 :=
by
  sorry


end NUMINAMATH_CALUDE_painter_problem_l3900_390039


namespace NUMINAMATH_CALUDE_cubic_function_property_l3900_390078

/-- Given a cubic function f(x) = ax³ + bx² with a maximum at x = 1 and f(1) = 3, prove that a + b = 3 -/
theorem cubic_function_property (a b : ℝ) : 
  let f := fun (x : ℝ) => a * x^3 + b * x^2
  let f' := fun (x : ℝ) => 3 * a * x^2 + 2 * b * x
  (f 1 = 3) → (f' 1 = 0) → (a + b = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3900_390078


namespace NUMINAMATH_CALUDE_prime_divisor_form_l3900_390009

theorem prime_divisor_form (a p : ℕ) (ha : a > 0) (hp : Nat.Prime p) 
  (hdiv : p ∣ a^3 - 3*a + 1) (hp_neq_3 : p ≠ 3) :
  ∃ k : ℤ, p = 9*k + 1 ∨ p = 9*k - 1 := by
sorry

end NUMINAMATH_CALUDE_prime_divisor_form_l3900_390009


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3900_390035

/-- If the simplest quadratic radical √a is of the same type as √27, then a = 3 -/
theorem simplest_quadratic_radical (a : ℝ) : (∃ k : ℕ+, a = 27 * k^2) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3900_390035


namespace NUMINAMATH_CALUDE_augmented_matrix_solution_l3900_390021

/-- Given an augmented matrix representing a system of linear equations with a known solution,
    prove that the difference between certain elements of the augmented matrix is 16. -/
theorem augmented_matrix_solution (c₁ c₂ : ℝ) : 
  (∃ (x y : ℝ), x = 3 ∧ y = 5 ∧ 
   2 * x + 3 * y = c₁ ∧
   y = c₂) →
  c₁ - c₂ = 16 := by
sorry

end NUMINAMATH_CALUDE_augmented_matrix_solution_l3900_390021


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3900_390076

-- Problem 1
theorem problem_1 : 
  |(-3)| + (-1)^2021 * (Real.pi - 3.14)^0 - (-1/2)⁻¹ = 4 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : 
  (x + 3)^2 - (x + 2) * (x - 2) = 6 * x + 13 := by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) : 
  (2*x - y + 3) * (2*x + y - 3) = 4*x^2 - y^2 + 6*y - 9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3900_390076


namespace NUMINAMATH_CALUDE_max_value_inequality_l3900_390012

theorem max_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (6 * a * b) / (9 * b^2 + a^2) + (2 * a * b) / (b^2 + a^2) ≤ 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3900_390012


namespace NUMINAMATH_CALUDE_max_ash_win_probability_l3900_390069

/-- Represents the types of monsters -/
inductive MonsterType
  | Fire
  | Grass
  | Water

/-- A lineup of monsters -/
def Lineup := List MonsterType

/-- The number of monsters in each lineup -/
def lineupSize : Nat := 15

/-- Calculates the probability of Ash winning given his lineup strategy -/
noncomputable def ashWinProbability (ashStrategy : Lineup) : ℝ :=
  sorry

/-- Theorem stating the maximum probability of Ash winning -/
theorem max_ash_win_probability :
  ∃ (optimalStrategy : Lineup),
    ashWinProbability optimalStrategy = 1 - (2/3)^lineupSize ∧
    ∀ (strategy : Lineup),
      ashWinProbability strategy ≤ ashWinProbability optimalStrategy :=
  sorry

end NUMINAMATH_CALUDE_max_ash_win_probability_l3900_390069


namespace NUMINAMATH_CALUDE_target_scientific_notation_l3900_390014

/-- Represents one billion in decimal notation -/
def billion : ℕ := 100000000

/-- The number we want to express in scientific notation -/
def target : ℕ := 1360000000

/-- Scientific notation for the target number -/
def scientific_notation (n : ℕ) : ℚ := 1.36 * (10 : ℚ) ^ n

theorem target_scientific_notation :
  ∃ n : ℕ, scientific_notation n = target ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_target_scientific_notation_l3900_390014


namespace NUMINAMATH_CALUDE_newspapers_sold_l3900_390002

theorem newspapers_sold (total : ℝ) (magazines : ℕ) 
  (h1 : total = 425.0) (h2 : magazines = 150) : 
  total - magazines = 275 := by
  sorry

end NUMINAMATH_CALUDE_newspapers_sold_l3900_390002


namespace NUMINAMATH_CALUDE_white_then_red_probability_l3900_390001

/-- The probability of drawing a white marble first and a red marble second from a bag with 4 red and 6 white marbles -/
theorem white_then_red_probability : 
  let total_marbles : ℕ := 4 + 6
  let red_marbles : ℕ := 4
  let white_marbles : ℕ := 6
  let prob_white_first : ℚ := white_marbles / total_marbles
  let prob_red_second : ℚ := red_marbles / (total_marbles - 1)
  prob_white_first * prob_red_second = 4 / 15 :=
by sorry

end NUMINAMATH_CALUDE_white_then_red_probability_l3900_390001


namespace NUMINAMATH_CALUDE_circle_equation_l3900_390013

/-- The equation of a circle in its general form -/
def is_circle (h x y a : ℝ) : Prop :=
  ∃ (c_x c_y r : ℝ), (x - c_x)^2 + (y - c_y)^2 = r^2 ∧ r > 0

/-- The given equation -/
def given_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 5*a = 0

theorem circle_equation (x y : ℝ) :
  is_circle 0 x y 1 ↔ given_equation x y 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3900_390013


namespace NUMINAMATH_CALUDE_average_age_of_eight_students_l3900_390092

theorem average_age_of_eight_students 
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_group1 : Nat)
  (num_group2 : Nat)
  (average_age_group2 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 8)
  (h4 : num_group2 = 6)
  (h5 : average_age_group2 = 16)
  (h6 : age_last_student = 17)
  (h7 : total_students = num_group1 + num_group2 + 1) :
  (total_students : ℝ) * average_age_all - 
  (num_group2 : ℝ) * average_age_group2 - 
  age_last_student = (num_group1 : ℝ) * 14 := by
    sorry

#check average_age_of_eight_students

end NUMINAMATH_CALUDE_average_age_of_eight_students_l3900_390092


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3900_390029

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 - 2*I) : 
  (z^2 + 3) / (z - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3900_390029


namespace NUMINAMATH_CALUDE_min_value_expression_l3900_390015

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 17/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3900_390015


namespace NUMINAMATH_CALUDE_fourth_side_length_l3900_390056

/-- A quadrilateral inscribed in a circle with radius 200√2, where three sides have length 200 -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of three sides of the quadrilateral -/
  side_length : ℝ
  /-- The fourth side of the quadrilateral -/
  fourth_side : ℝ
  /-- Assertion that the radius is 200√2 -/
  radius_eq : radius = 200 * Real.sqrt 2
  /-- Assertion that three sides have length 200 -/
  three_sides_eq : side_length = 200

/-- Theorem stating that the fourth side of the quadrilateral has length 500 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.fourth_side = 500 := by
  sorry

#check fourth_side_length

end NUMINAMATH_CALUDE_fourth_side_length_l3900_390056


namespace NUMINAMATH_CALUDE_function_symmetry_about_origin_l3900_390040

/-- The function f(x) = x^5 + x^3 is odd, implying symmetry about the origin -/
theorem function_symmetry_about_origin (x : ℝ) : 
  ((-x)^5 + (-x)^3) = -(x^5 + x^3) := by sorry

end NUMINAMATH_CALUDE_function_symmetry_about_origin_l3900_390040


namespace NUMINAMATH_CALUDE_min_product_xyz_l3900_390074

theorem min_product_xyz (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 3/125 := by
  sorry

end NUMINAMATH_CALUDE_min_product_xyz_l3900_390074


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l3900_390097

theorem intersection_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  (A ∩ B = {3}) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l3900_390097
