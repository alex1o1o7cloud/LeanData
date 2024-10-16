import Mathlib

namespace NUMINAMATH_CALUDE_exactly_four_even_probability_l2816_281698

def num_dice : ℕ := 8
def num_even : ℕ := 4
def prob_even : ℚ := 2/3
def prob_odd : ℚ := 1/3

theorem exactly_four_even_probability :
  (Nat.choose num_dice num_even) * (prob_even ^ num_even) * (prob_odd ^ (num_dice - num_even)) = 1120/6561 := by
  sorry

end NUMINAMATH_CALUDE_exactly_four_even_probability_l2816_281698


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2816_281696

def M (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def P (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, M a ∩ P a = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2816_281696


namespace NUMINAMATH_CALUDE_odd_function_m_value_l2816_281685

/-- Given a > 0 and a ≠ 1, if f(x) = 1/(a^x + 1) - m is an odd function, then m = 1/2 -/
theorem odd_function_m_value (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => 1 / (a^x + 1) - m
  (∀ x, f x + f (-x) = 0) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_m_value_l2816_281685


namespace NUMINAMATH_CALUDE_set_equality_proof_all_sets_satisfying_condition_l2816_281647

def solution_set : Set (Set Nat) :=
  {{3}, {1, 3}, {2, 3}, {1, 2, 3}}

theorem set_equality_proof (B : Set Nat) :
  ({1, 2} ∪ B = {1, 2, 3}) ↔ (B ∈ solution_set) := by
  sorry

theorem all_sets_satisfying_condition :
  {B : Set Nat | {1, 2} ∪ B = {1, 2, 3}} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_set_equality_proof_all_sets_satisfying_condition_l2816_281647


namespace NUMINAMATH_CALUDE_shaded_area_outside_overlap_l2816_281659

/-- Given two rectangles with specific dimensions and overlap, calculate the shaded area outside the overlap -/
theorem shaded_area_outside_overlap (rect1_width rect1_height rect2_width rect2_height overlap_width overlap_height : ℕ) 
  (h1 : rect1_width = 4 ∧ rect1_height = 12)
  (h2 : rect2_width = 5 ∧ rect2_height = 9)
  (h3 : overlap_width = 4 ∧ overlap_height = 5) :
  rect1_width * rect1_height + rect2_width * rect2_height - overlap_width * overlap_height = 73 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_outside_overlap_l2816_281659


namespace NUMINAMATH_CALUDE_apple_production_theorem_l2816_281629

/-- Apple production over three years -/
def AppleProduction : Prop :=
  let first_year : ℕ := 40
  let second_year : ℕ := 8 + 2 * first_year
  let third_year : ℕ := second_year - (second_year / 4)
  let total : ℕ := first_year + second_year + third_year
  total = 194

theorem apple_production_theorem : AppleProduction := by
  sorry

end NUMINAMATH_CALUDE_apple_production_theorem_l2816_281629


namespace NUMINAMATH_CALUDE_songs_per_album_l2816_281689

/-- The number of country albums bought -/
def country_albums : ℕ := 4

/-- The number of pop albums bought -/
def pop_albums : ℕ := 5

/-- The total number of songs bought -/
def total_songs : ℕ := 72

/-- Proves that if all albums have the same number of songs, then each album contains 8 songs -/
theorem songs_per_album :
  ∀ (songs_per_album : ℕ),
  country_albums * songs_per_album + pop_albums * songs_per_album = total_songs →
  songs_per_album = 8 := by
sorry

end NUMINAMATH_CALUDE_songs_per_album_l2816_281689


namespace NUMINAMATH_CALUDE_modulus_of_z_l2816_281686

theorem modulus_of_z (z : ℂ) (h : z * (1 + Complex.I) = 1 + 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2816_281686


namespace NUMINAMATH_CALUDE_factor_tree_product_l2816_281619

theorem factor_tree_product : ∀ (X F G H : ℕ),
  X = F * G →
  F = 11 * 7 →
  G = 7 * H →
  H = 17 * 2 →
  X = 57556 := by
sorry

end NUMINAMATH_CALUDE_factor_tree_product_l2816_281619


namespace NUMINAMATH_CALUDE_toy_organizer_price_correct_l2816_281622

/-- The price of a gaming chair in dollars -/
def gaming_chair_price : ℝ := 83

/-- The number of toy organizer sets ordered -/
def toy_organizer_sets : ℕ := 3

/-- The number of gaming chairs ordered -/
def gaming_chairs : ℕ := 2

/-- The delivery fee percentage -/
def delivery_fee_percent : ℝ := 0.05

/-- The total amount paid by Leon in dollars -/
def total_paid : ℝ := 420

/-- The price per set of toy organizers in dollars -/
def toy_organizer_price : ℝ := 78

theorem toy_organizer_price_correct :
  toy_organizer_price * toy_organizer_sets +
  gaming_chair_price * gaming_chairs +
  delivery_fee_percent * (toy_organizer_price * toy_organizer_sets + gaming_chair_price * gaming_chairs) =
  total_paid := by
  sorry

#check toy_organizer_price_correct

end NUMINAMATH_CALUDE_toy_organizer_price_correct_l2816_281622


namespace NUMINAMATH_CALUDE_pages_left_to_read_l2816_281613

theorem pages_left_to_read (total_pages read_pages : ℕ) 
  (h1 : total_pages = 563)
  (h2 : read_pages = 147) :
  total_pages - read_pages = 416 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l2816_281613


namespace NUMINAMATH_CALUDE_game_outcome_theorem_l2816_281621

/-- Represents the outcome of the game -/
inductive GameOutcome
| Draw
| BWin

/-- Defines the game rules and determines the outcome for a given n -/
def gameOutcome (n : ℕ+) : GameOutcome :=
  if n ∈ ({1, 2, 4, 6} : Finset ℕ+) then
    GameOutcome.Draw
  else
    GameOutcome.BWin

/-- Theorem stating the game outcome for all positive integers n -/
theorem game_outcome_theorem (n : ℕ+) :
  (gameOutcome n = GameOutcome.Draw ↔ n ∈ ({1, 2, 4, 6} : Finset ℕ+)) ∧
  (gameOutcome n = GameOutcome.BWin ↔ n ∉ ({1, 2, 4, 6} : Finset ℕ+)) :=
by sorry

end NUMINAMATH_CALUDE_game_outcome_theorem_l2816_281621


namespace NUMINAMATH_CALUDE_sequence_inequality_l2816_281633

theorem sequence_inequality (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 1) ≥ a n ^ 2 + 1/5) : 
  ∀ n : ℕ, n ≥ 5 → Real.sqrt (a (n + 5)) ≥ a (n - 5) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2816_281633


namespace NUMINAMATH_CALUDE_percentage_increase_is_20_percent_l2816_281652

/-- Represents the number of units in each building --/
structure BuildingUnits where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the percentage increase from the second to the third building --/
def percentageIncrease (units : BuildingUnits) : ℚ :=
  (units.third - units.second : ℚ) / units.second * 100

/-- The main theorem stating the percentage increase is 20% --/
theorem percentage_increase_is_20_percent 
  (total : ℕ) 
  (h1 : total = 7520) 
  (units : BuildingUnits) 
  (h2 : units.first = 4000) 
  (h3 : units.second = 2 * units.first / 5) 
  (h4 : total = units.first + units.second + units.third) : 
  percentageIncrease units = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_is_20_percent_l2816_281652


namespace NUMINAMATH_CALUDE_max_value_of_z_l2816_281678

/-- Given a system of inequalities, prove that the maximum value of z = 2x + 3y is 8 -/
theorem max_value_of_z (x y : ℝ) 
  (h1 : x + y - 1 ≥ 0) 
  (h2 : y - x - 1 ≤ 0) 
  (h3 : x ≤ 1) : 
  (∀ x' y' : ℝ, x' + y' - 1 ≥ 0 → y' - x' - 1 ≤ 0 → x' ≤ 1 → 2*x' + 3*y' ≤ 2*x + 3*y) →
  2*x + 3*y = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2816_281678


namespace NUMINAMATH_CALUDE_a_neg_one_necessary_not_sufficient_l2816_281643

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + (a + 2) * y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 2 = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ (x y : ℝ), l₁ a x y ↔ l₂ a x y

-- State the theorem
theorem a_neg_one_necessary_not_sufficient :
  (∀ a : ℝ, parallel a → a = -1) ∧ 
  ¬(∀ a : ℝ, a = -1 → parallel a) :=
sorry

end NUMINAMATH_CALUDE_a_neg_one_necessary_not_sufficient_l2816_281643


namespace NUMINAMATH_CALUDE_max_coincident_area_folded_triangle_l2816_281663

theorem max_coincident_area_folded_triangle :
  let a := 3 / 2
  let b := Real.sqrt 5 / 2
  let c := Real.sqrt 2
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let height := 2 * area / a
  let max_coincident_area := area + (1 / (2 * height)) - (1 / (4 * height^2)) - (3 / (4 * height^2))
  max_coincident_area = 9 / 28 := by sorry

end NUMINAMATH_CALUDE_max_coincident_area_folded_triangle_l2816_281663


namespace NUMINAMATH_CALUDE_technician_round_trip_completion_l2816_281679

theorem technician_round_trip_completion (distance : ℝ) (h : distance > 0) :
  let one_way := distance
  let round_trip := 2 * distance
  let completed := distance + 0.2 * distance
  (completed / round_trip) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_completion_l2816_281679


namespace NUMINAMATH_CALUDE_vector_at_minus_2_l2816_281624

/-- A line in a plane parameterized by t -/
def line (t : ℝ) : ℝ × ℝ := sorry

/-- The vector at t = 5 is (0, 5) -/
axiom vector_at_5 : line 5 = (0, 5)

/-- The vector at t = 8 is (9, 1) -/
axiom vector_at_8 : line 8 = (9, 1)

/-- The theorem to prove -/
theorem vector_at_minus_2 : line (-2) = (21, -23/3) := by sorry

end NUMINAMATH_CALUDE_vector_at_minus_2_l2816_281624


namespace NUMINAMATH_CALUDE_total_fans_l2816_281638

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- Defines the conditions of the problem -/
def fan_conditions (f : FanCounts) : Prop :=
  f.yankees * 2 = f.mets * 3 ∧  -- Ratio of Yankees to Mets fans is 3:2
  f.mets * 5 = f.red_sox * 4 ∧  -- Ratio of Mets to Red Sox fans is 4:5
  f.mets = 88                   -- There are 88 Mets fans

/-- The theorem to be proved -/
theorem total_fans (f : FanCounts) (h : fan_conditions f) : 
  f.yankees + f.mets + f.red_sox = 330 := by
  sorry

#check total_fans

end NUMINAMATH_CALUDE_total_fans_l2816_281638


namespace NUMINAMATH_CALUDE_shoe_box_problem_l2816_281628

theorem shoe_box_problem (num_pairs : ℕ) (prob_match : ℚ) :
  num_pairs = 6 →
  prob_match = 1 / 11 →
  (num_pairs * 2 : ℕ) = 12 :=
by sorry

end NUMINAMATH_CALUDE_shoe_box_problem_l2816_281628


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2816_281691

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2816_281691


namespace NUMINAMATH_CALUDE_inverse_f_58_l2816_281653

def f (x : ℝ) : ℝ := 2 * x^3 + 4

theorem inverse_f_58 : f⁻¹ 58 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_f_58_l2816_281653


namespace NUMINAMATH_CALUDE_range_of_a_for_non_negative_x_l2816_281625

theorem range_of_a_for_non_negative_x (a x : ℝ) : 
  (x - a = 1 - 2*x ∧ x ≥ 0) → a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_non_negative_x_l2816_281625


namespace NUMINAMATH_CALUDE_beidou_satellite_altitude_scientific_notation_l2816_281620

theorem beidou_satellite_altitude_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 18500000 = a * (10 : ℝ) ^ n ∧ a = 1.85 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_beidou_satellite_altitude_scientific_notation_l2816_281620


namespace NUMINAMATH_CALUDE_loom_weaving_rate_l2816_281650

/-- The rate at which an industrial loom weaves cloth, given the amount of cloth woven and the time taken. -/
theorem loom_weaving_rate (cloth_woven : Real) (time_taken : Real) (h : cloth_woven = 25 ∧ time_taken = 195.3125) :
  cloth_woven / time_taken = 0.128 := by
  sorry

end NUMINAMATH_CALUDE_loom_weaving_rate_l2816_281650


namespace NUMINAMATH_CALUDE_expression_simplification_l2816_281604

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3) :
  (x^2 - 2*x + 1) / (x^2 - x) / (x - 1) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2816_281604


namespace NUMINAMATH_CALUDE_shopping_trip_expenses_l2816_281609

theorem shopping_trip_expenses (T : ℝ) (h_positive : T > 0) : 
  let clothing_percent : ℝ := 0.50
  let other_percent : ℝ := 0.30
  let clothing_tax : ℝ := 0.05
  let other_tax : ℝ := 0.10
  let total_tax_percent : ℝ := 0.055
  let food_percent : ℝ := 1 - clothing_percent - other_percent

  clothing_tax * clothing_percent * T + other_tax * other_percent * T = total_tax_percent * T →
  food_percent = 0.20 := by
sorry

end NUMINAMATH_CALUDE_shopping_trip_expenses_l2816_281609


namespace NUMINAMATH_CALUDE_postage_fee_420g_l2816_281673

/-- Calculates the postage fee for a given weight in grams -/
def postage_fee (weight : ℕ) : ℚ :=
  0.7 + 0.4 * ((weight - 1) / 100 : ℕ)

/-- The postage fee for a 420g book is 2.3 yuan -/
theorem postage_fee_420g : postage_fee 420 = 2.3 := by sorry

end NUMINAMATH_CALUDE_postage_fee_420g_l2816_281673


namespace NUMINAMATH_CALUDE_number_problem_l2816_281603

theorem number_problem (x : ℝ) : 
  (0.3 * x = 0.6 * 150 + 120) → x = 700 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l2816_281603


namespace NUMINAMATH_CALUDE_ab_length_in_specific_triangle_l2816_281627

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def isAcute (t : Triangle) : Prop := sorry

def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

def triangleArea (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem ab_length_in_specific_triangle :
  ∀ (t : Triangle),
    isAcute t →
    sideLength t.A t.C = 4 →
    sideLength t.B t.C = 3 →
    triangleArea t = 3 * Real.sqrt 3 →
    sideLength t.A t.B = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ab_length_in_specific_triangle_l2816_281627


namespace NUMINAMATH_CALUDE_modified_factor_tree_l2816_281699

theorem modified_factor_tree (P X Y G Z : ℕ) : 
  P = X * Y ∧
  X = 7 * G ∧
  Y = 11 * Z ∧
  G = 7 * 4 ∧
  Z = 11 * 4 →
  P = 94864 := by
sorry

end NUMINAMATH_CALUDE_modified_factor_tree_l2816_281699


namespace NUMINAMATH_CALUDE_gym_class_counts_l2816_281612

/-- Given five gym classes with student counts P1, P2, P3, P4, and P5, prove that
    P2 = 5, P3 = 12.5, P4 = 25/3, and P5 = 25/3 given the following conditions:
    - P1 = 15
    - P1 = P2 + 10
    - P2 = 2 * P3 - 20
    - P3 = (P4 + P5) - 5
    - P4 = (1 / 2) * P5 + 5 -/
theorem gym_class_counts (P1 P2 P3 P4 P5 : ℚ) 
  (h1 : P1 = 15)
  (h2 : P1 = P2 + 10)
  (h3 : P2 = 2 * P3 - 20)
  (h4 : P3 = (P4 + P5) - 5)
  (h5 : P4 = (1 / 2) * P5 + 5) :
  P2 = 5 ∧ P3 = 25/2 ∧ P4 = 25/3 ∧ P5 = 25/3 := by
  sorry


end NUMINAMATH_CALUDE_gym_class_counts_l2816_281612


namespace NUMINAMATH_CALUDE_share_ratio_l2816_281601

def total_amount : ℕ := 544

def shares (A B C : ℕ) : Prop :=
  A + B + C = total_amount ∧ 4 * B = C

theorem share_ratio (A B C : ℕ) (h : shares A B C) (hA : A = 64) (hB : B = 96) (hC : C = 384) :
  A * 3 = B * 2 := by sorry

end NUMINAMATH_CALUDE_share_ratio_l2816_281601


namespace NUMINAMATH_CALUDE_mam_mgm_difference_bound_l2816_281684

theorem mam_mgm_difference_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  let mam := (a^(1/3) + b^(1/3)) / 2
  let mgm := (a * b)^(1/6)
  mam - mgm < (b - a) / (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_mam_mgm_difference_bound_l2816_281684


namespace NUMINAMATH_CALUDE_tournament_committee_count_l2816_281635

/-- Represents a frisbee league -/
structure FrisbeeLeague where
  teams : Nat
  membersPerTeam : Nat
  committeeSize : Nat
  hostTeamMembers : Nat
  nonHostTeamMembers : Nat

/-- The specific frisbee league described in the problem -/
def regionalLeague : FrisbeeLeague :=
  { teams := 5
  , membersPerTeam := 8
  , committeeSize := 11
  , hostTeamMembers := 4
  , nonHostTeamMembers := 3 }

/-- The number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

/-- The number of possible tournament committees -/
def numberOfCommittees (league : FrisbeeLeague) : Nat :=
  league.teams *
  (choose (league.membersPerTeam - 1) (league.hostTeamMembers - 1)) *
  (choose league.membersPerTeam league.nonHostTeamMembers ^ (league.teams - 1))

/-- Theorem stating the number of possible tournament committees -/
theorem tournament_committee_count :
  numberOfCommittees regionalLeague = 1723286800 := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l2816_281635


namespace NUMINAMATH_CALUDE_cube_angle_range_l2816_281694

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point3D) : ℝ := sorry

/-- Theorem: The angle between A₁M and C₁N is in the range (π/3, π/2) -/
theorem cube_angle_range (cube : Cube) (M N : Point3D) 
  (h_M : M.x > cube.A.x ∧ M.x < cube.B.x ∧ M.y = cube.A.y ∧ M.z = cube.A.z)
  (h_N : N.x = cube.B.x ∧ N.y > cube.B.y ∧ N.y < cube.B₁.y ∧ N.z = cube.B.z)
  (h_AM_eq_B₁N : (M.x - cube.A.x)^2 = (cube.B₁.y - N.y)^2) :
  let θ := angle (Point3D.mk (cube.A₁.x - M.x) (cube.A₁.y - M.y) (cube.A₁.z - M.z))
              (Point3D.mk (cube.C₁.x - N.x) (cube.C₁.y - N.y) (cube.C₁.z - N.z))
  π/3 < θ ∧ θ < π/2 := by
  sorry

end NUMINAMATH_CALUDE_cube_angle_range_l2816_281694


namespace NUMINAMATH_CALUDE_line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l2816_281660

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)
variable (para : Line → Plane → Prop)

-- State the theorem
theorem line_perp_to_plane_and_line_para_to_plane_implies_lines_perp
  (a b : Line) (α : Plane) :
  a ≠ b →  -- a and b are non-coincident
  perpToPlane a α →  -- a is perpendicular to α
  para b α →  -- b is parallel to α
  perp a b :=  -- then a is perpendicular to b
by sorry

end NUMINAMATH_CALUDE_line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l2816_281660


namespace NUMINAMATH_CALUDE_jennifer_apples_l2816_281687

/-- Given that Jennifer starts with 7 apples and ends with 81 apples,
    prove that she found 74 apples. -/
theorem jennifer_apples :
  let start_apples : ℕ := 7
  let end_apples : ℕ := 81
  let found_apples : ℕ := end_apples - start_apples
  found_apples = 74 := by sorry

end NUMINAMATH_CALUDE_jennifer_apples_l2816_281687


namespace NUMINAMATH_CALUDE_wrong_value_correction_l2816_281640

theorem wrong_value_correction (n : ℕ) (initial_mean correct_mean wrong_value : ℚ) 
  (h1 : n = 25)
  (h2 : initial_mean = 190)
  (h3 : wrong_value = 130)
  (h4 : correct_mean = 191.4) :
  let initial_sum := n * initial_mean
  let sum_without_wrong := initial_sum - wrong_value
  let correct_sum := n * correct_mean
  correct_sum - sum_without_wrong + wrong_value = 295 := by
sorry

end NUMINAMATH_CALUDE_wrong_value_correction_l2816_281640


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_in_20_years_l2816_281661

/-- 
Given a sum of money that doubles itself in 20 years at simple interest,
this theorem proves that the rate percent per annum is 5%.
-/
theorem simple_interest_rate_for_doubling_in_20_years :
  ∀ (principal : ℝ) (rate : ℝ),
  principal > 0 →
  principal * (1 + rate * 20 / 100) = 2 * principal →
  rate = 5 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_in_20_years_l2816_281661


namespace NUMINAMATH_CALUDE_spring_scale_reading_comparison_l2816_281669

/-- Represents the angular velocity of Earth's rotation -/
def earth_angular_velocity : ℝ := sorry

/-- Represents the radius of the Earth at the equator -/
def earth_equator_radius : ℝ := sorry

/-- Represents the acceleration due to gravity at the equator -/
def gravity_equator : ℝ := sorry

/-- Represents the acceleration due to gravity at the poles -/
def gravity_pole : ℝ := sorry

/-- Calculates the centrifugal force at the equator for an object of mass m -/
def centrifugal_force (m : ℝ) : ℝ :=
  m * earth_angular_velocity^2 * earth_equator_radius

/-- Calculates the apparent weight of an object at the equator -/
def apparent_weight_equator (m : ℝ) : ℝ :=
  m * gravity_equator - centrifugal_force m

/-- Calculates the apparent weight of an object at the pole -/
def apparent_weight_pole (m : ℝ) : ℝ :=
  m * gravity_pole

theorem spring_scale_reading_comparison (m : ℝ) (m_pos : m > 0) :
  apparent_weight_pole m > apparent_weight_equator m :=
by
  sorry

end NUMINAMATH_CALUDE_spring_scale_reading_comparison_l2816_281669


namespace NUMINAMATH_CALUDE_smallest_prime_longest_sequence_l2816_281672

def A₁₁ : ℕ := 30

def is_prime_sequence (p : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → Nat.Prime (p + k * A₁₁)

theorem smallest_prime_longest_sequence :
  ∃ n : ℕ, 
    Nat.Prime 7 ∧ 
    is_prime_sequence 7 n ∧
    ∀ p < 7, Nat.Prime p → ∀ m : ℕ, is_prime_sequence p m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_longest_sequence_l2816_281672


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2816_281693

theorem constant_term_expansion (x : ℝ) : 
  (fun r : ℕ => (-1)^r * (Nat.choose 6 r) * x^(6 - 2*r)) 3 = -20 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2816_281693


namespace NUMINAMATH_CALUDE_expression_evaluation_l2816_281666

theorem expression_evaluation :
  let x := Real.sqrt 2 * Real.sin (π / 4) + Real.tan (π / 3)
  (x / (x^2 - 1)) / (1 - 1 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2816_281666


namespace NUMINAMATH_CALUDE_solutions_eq1_solutions_eq2_l2816_281602

-- Equation 1
theorem solutions_eq1 : 
  ∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ x = 7 ∨ x = -1 := by sorry

-- Equation 2
theorem solutions_eq2 : 
  ∀ x : ℝ, 3*x^2 - 1 = 2*x ↔ x = 1 ∨ x = -1/3 := by sorry

end NUMINAMATH_CALUDE_solutions_eq1_solutions_eq2_l2816_281602


namespace NUMINAMATH_CALUDE_q_polynomial_l2816_281681

theorem q_polynomial (x : ℝ) (q : ℝ → ℝ) 
  (h : ∀ x, q x + (2*x^6 + 4*x^4 - 5*x^3 + 2*x) = (3*x^4 + x^3 - 11*x^2 + 6*x + 3)) :
  q x = -2*x^6 - x^4 + 6*x^3 - 11*x^2 + 4*x + 3 := by
sorry

end NUMINAMATH_CALUDE_q_polynomial_l2816_281681


namespace NUMINAMATH_CALUDE_right_triangle_sin_value_l2816_281646

/-- Given a right triangle DEF with �angle E = 90° and 4 sin D = 5 cos D, prove that sin D = (5√41) / 41 -/
theorem right_triangle_sin_value (D E F : ℝ) (h_right_angle : E = 90) 
  (h_sin_cos_relation : 4 * Real.sin D = 5 * Real.cos D) : 
  Real.sin D = (5 * Real.sqrt 41) / 41 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_value_l2816_281646


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2816_281615

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 5*x + 6 ≤ 0 ↔ 2 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2816_281615


namespace NUMINAMATH_CALUDE_linda_age_l2816_281692

/-- Given that Linda's age is 3 more than 2 times Jane's age, and in 5 years
    the sum of their ages will be 28, prove that Linda's current age is 13. -/
theorem linda_age (j : ℕ) : 
  (j + 5) + ((2 * j + 3) + 5) = 28 → 2 * j + 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_linda_age_l2816_281692


namespace NUMINAMATH_CALUDE_race_speed_factor_l2816_281618

/-- Represents the race scenario described in the problem -/
structure RaceScenario where
  k : ℝ  -- Factor by which A is faster than B
  startAdvantage : ℝ  -- Head start given to B in meters
  totalDistance : ℝ  -- Total race distance in meters

/-- Theorem stating that under the given conditions, A must be 4 times faster than B -/
theorem race_speed_factor (race : RaceScenario) 
  (h1 : race.startAdvantage = 72)
  (h2 : race.totalDistance = 96)
  (h3 : race.totalDistance / race.k = (race.totalDistance - race.startAdvantage)) :
  race.k = 4 := by
  sorry


end NUMINAMATH_CALUDE_race_speed_factor_l2816_281618


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2816_281605

/-- An infinite geometric series with common ratio 1/4 and sum 40 has a first term of 30. -/
theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1/4)
  (h_S : S = 40)
  (h_sum : S = a / (1 - r))
  : a = 30 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2816_281605


namespace NUMINAMATH_CALUDE_marbles_per_pack_l2816_281606

theorem marbles_per_pack (total_marbles : ℕ) (total_packs : ℕ) 
  (leo_packs manny_packs neil_packs : ℕ) : 
  total_marbles = 400 →
  leo_packs = 25 →
  manny_packs = total_packs / 4 →
  neil_packs = total_packs / 8 →
  leo_packs + manny_packs + neil_packs = total_packs →
  total_marbles / total_packs = 10 := by
  sorry

end NUMINAMATH_CALUDE_marbles_per_pack_l2816_281606


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l2816_281654

/-- Given two monomials -xy^(b+1) and (1/2)x^(a+2)y^3, if their sum is still a monomial, then a + b = 1 -/
theorem monomial_sum_condition (a b : ℤ) : 
  (∃ (k : ℚ), k * x * y^(b + 1) + (1/2) * x^(a + 2) * y^3 = c * x^m * y^n) → 
  a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l2816_281654


namespace NUMINAMATH_CALUDE_average_weight_problem_l2816_281648

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions and the average of a and b is 40. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 41 →
  b = 27 →
  (a + b) / 2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l2816_281648


namespace NUMINAMATH_CALUDE_touch_point_theorem_l2816_281608

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  -- The length of the hypotenuse
  hypotenuse : ℝ
  -- The radius of the inscribed circle
  radius : ℝ
  -- Assumption that the hypotenuse is positive
  hypotenuse_pos : hypotenuse > 0
  -- Assumption that the radius is positive
  radius_pos : radius > 0

/-- The length from one vertex to where the circle touches the hypotenuse -/
def touchPoint (t : RightTriangleWithInscribedCircle) : Set ℝ :=
  {x : ℝ | x = t.hypotenuse / 2 - t.radius ∨ x = t.hypotenuse / 2 + t.radius}

theorem touch_point_theorem (t : RightTriangleWithInscribedCircle) 
    (h1 : t.hypotenuse = 10) (h2 : t.radius = 2) : 
    touchPoint t = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_touch_point_theorem_l2816_281608


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2816_281616

-- Problem 1
theorem problem_1 (x y : ℝ) : (-4 * x * y^3) * (-2 * x)^2 = -16 * x^3 * y^3 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (3*x - 2) * (2*x - 3) - (x - 1) * (6*x + 5) = -12*x + 11 := by sorry

-- Problem 3
theorem problem_3 : (3 * (10^2)) * (5 * (10^5)) = (1.5 : ℝ) * (10^8) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2816_281616


namespace NUMINAMATH_CALUDE_complex_number_simplification_l2816_281697

theorem complex_number_simplification :
  let i : ℂ := Complex.I
  (i^3) / (1 - i) = (1 / 2 : ℂ) - (1 / 2 : ℂ) * i := by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l2816_281697


namespace NUMINAMATH_CALUDE_laptop_price_l2816_281665

def original_price : ℝ → Prop :=
  fun x => (0.80 * x - 50 = 0.70 * x - 30) ∧ (x > 0)

theorem laptop_price : ∃ x, original_price x ∧ x = 200 := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_l2816_281665


namespace NUMINAMATH_CALUDE_crude_oil_temperature_l2816_281655

-- Define the function f(x) = x^2 - 7x + 15
def f (x : ℝ) : ℝ := x^2 - 7*x + 15

-- Define the domain of f
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 8 }

theorem crude_oil_temperature (x : ℝ) (h : x ∈ domain) :
  -- The derivative of f at x = 4 is 1
  deriv f 4 = 1 ∧
  -- The function is increasing at x = 4
  0 < deriv f 4 := by
  sorry

end NUMINAMATH_CALUDE_crude_oil_temperature_l2816_281655


namespace NUMINAMATH_CALUDE_max_area_quadrilateral_l2816_281671

/-- Given a point P in the first quadrant and points A on the x-axis and B on the y-axis 
    such that PA = PB = 2, the maximum area of quadrilateral PAOB is 2 + 2√2. -/
theorem max_area_quadrilateral (P A B : ℝ × ℝ) : 
  (0 < P.1 ∧ 0 < P.2) →  -- P is in the first quadrant
  A.2 = 0 →  -- A is on the x-axis
  B.1 = 0 →  -- B is on the y-axis
  Real.sqrt ((P.1 - A.1)^2 + P.2^2) = 2 →  -- PA = 2
  Real.sqrt (P.1^2 + (P.2 - B.2)^2) = 2 →  -- PB = 2
  (∃ (area : ℝ), ∀ (Q : ℝ × ℝ), 
    (0 < Q.1 ∧ 0 < Q.2) →
    Real.sqrt ((Q.1 - A.1)^2 + Q.2^2) = 2 →
    Real.sqrt (Q.1^2 + (Q.2 - B.2)^2) = 2 →
    (1/2 * |A.1 * Q.1 + B.2 * Q.2| ≤ area)) ∧
  (1/2 * |A.1 * P.1 + B.2 * P.2| = 2 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_area_quadrilateral_l2816_281671


namespace NUMINAMATH_CALUDE_fifth_group_students_l2816_281657

theorem fifth_group_students (total : Nat) (group1 group2 group3 group4 : Nat)
  (h1 : total = 40)
  (h2 : group1 = 6)
  (h3 : group2 = 9)
  (h4 : group3 = 8)
  (h5 : group4 = 7) :
  total - (group1 + group2 + group3 + group4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fifth_group_students_l2816_281657


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_l2816_281630

theorem geometric_arithmetic_progression (b : ℝ) (q : ℝ) :
  b > 0 ∧ q > 1 →
  (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (b * q ^ i.val + b * q ^ k.val) / 2 = b * q ^ j.val) →
  q = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_l2816_281630


namespace NUMINAMATH_CALUDE_logarithm_inequality_l2816_281637

theorem logarithm_inequality (m : ℝ) (a b c : ℝ) 
  (h1 : 1/10 < m ∧ m < 1) 
  (h2 : a = Real.log m) 
  (h3 : b = Real.log (m^2)) 
  (h4 : c = Real.log (m^3)) : 
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l2816_281637


namespace NUMINAMATH_CALUDE_prob_standard_bulb_l2816_281688

/-- Probability of selecting a light bulb from the first factory -/
def p_factory1 : ℝ := 0.2

/-- Probability of selecting a light bulb from the second factory -/
def p_factory2 : ℝ := 0.3

/-- Probability of selecting a light bulb from the third factory -/
def p_factory3 : ℝ := 0.5

/-- Probability of producing a defective light bulb in the first factory -/
def q1 : ℝ := 0.01

/-- Probability of producing a defective light bulb in the second factory -/
def q2 : ℝ := 0.005

/-- Probability of producing a defective light bulb in the third factory -/
def q3 : ℝ := 0.006

/-- Theorem: The probability of randomly selecting a standard (non-defective) light bulb -/
theorem prob_standard_bulb : 
  p_factory1 * (1 - q1) + p_factory2 * (1 - q2) + p_factory3 * (1 - q3) = 0.9935 := by
  sorry

end NUMINAMATH_CALUDE_prob_standard_bulb_l2816_281688


namespace NUMINAMATH_CALUDE_area_ratio_of_similar_triangles_l2816_281682

-- Define two triangles
variable (T1 T2 : Set (ℝ × ℝ))

-- Define similarity ratio
variable (k : ℝ)

-- Define the property of similarity
def are_similar (T1 T2 : Set (ℝ × ℝ)) (k : ℝ) : Prop := sorry

-- Define the area of a triangle
def area (T : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_ratio_of_similar_triangles 
  (h_similar : are_similar T1 T2 k) 
  (h_k_pos : k > 0) :
  area T2 / area T1 = k^2 := sorry

end NUMINAMATH_CALUDE_area_ratio_of_similar_triangles_l2816_281682


namespace NUMINAMATH_CALUDE_solution_range_l2816_281662

theorem solution_range (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 1/x + 4/y = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 4/y = 1 ∧ x + y/4 < m^2 - 3*m) ↔ m < -1 ∨ m > 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2816_281662


namespace NUMINAMATH_CALUDE_bobs_weekly_profit_l2816_281617

/-- Calculates the weekly profit for Bob's muffin business -/
theorem bobs_weekly_profit (muffins_per_day : ℕ) (buy_price : ℚ) (sell_price : ℚ) (days_per_week : ℕ) :
  muffins_per_day = 12 →
  buy_price = 3/4 →
  sell_price = 3/2 →
  days_per_week = 7 →
  (sell_price - buy_price) * muffins_per_day * days_per_week = 63 := by
sorry

#eval (3/2 : ℚ) - (3/4 : ℚ)
#eval ((3/2 : ℚ) - (3/4 : ℚ)) * 12
#eval (((3/2 : ℚ) - (3/4 : ℚ)) * 12) * 7

end NUMINAMATH_CALUDE_bobs_weekly_profit_l2816_281617


namespace NUMINAMATH_CALUDE_presentation_students_l2816_281668

/-- The number of students in a presentation, given Eunjeong's position -/
def total_students (students_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  students_in_front + 1 + (position_from_back - 1)

/-- Theorem stating the total number of students in the presentation -/
theorem presentation_students : total_students 7 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_presentation_students_l2816_281668


namespace NUMINAMATH_CALUDE_fraction_problem_l2816_281664

theorem fraction_problem (x : ℚ) : x * 8 + 2 = 8 ↔ x = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l2816_281664


namespace NUMINAMATH_CALUDE_min_students_in_class_l2816_281674

theorem min_students_in_class (n b g : ℕ) : 
  n ≡ 2 [MOD 5] →
  (3 * g : ℕ) = (2 * b : ℕ) →
  n = b + g →
  n ≥ 57 ∧ (∀ m : ℕ, m < 57 → ¬(m ≡ 2 [MOD 5] ∧ ∃ b' g' : ℕ, (3 * g' : ℕ) = (2 * b' : ℕ) ∧ m = b' + g')) :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_class_l2816_281674


namespace NUMINAMATH_CALUDE_speed_difference_l2816_281667

/-- The difference in average speeds between two travelers -/
theorem speed_difference (distance : ℝ) (time1 time2 : ℝ) :
  distance > 0 ∧ time1 > 0 ∧ time2 > 0 →
  distance = 15 ∧ time1 = 1/3 ∧ time2 = 1/4 →
  (distance / time2) - (distance / time1) = 15 := by
  sorry

#check speed_difference

end NUMINAMATH_CALUDE_speed_difference_l2816_281667


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2816_281642

theorem solution_set_inequality (x : ℝ) : 
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2816_281642


namespace NUMINAMATH_CALUDE_probability_is_one_over_930_l2816_281632

/-- Represents a sequence of 40 distinct real numbers -/
def Sequence := { s : Fin 40 → ℝ // Function.Injective s }

/-- The operation that compares and potentially swaps adjacent elements -/
def operation (s : Sequence) : Sequence := sorry

/-- The probability that the 20th element moves to the 30th position after one operation -/
def probability_20_to_30 (s : Sequence) : ℚ := sorry

/-- Theorem stating that the probability is 1/930 -/
theorem probability_is_one_over_930 (s : Sequence) : 
  probability_20_to_30 s = 1 / 930 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_over_930_l2816_281632


namespace NUMINAMATH_CALUDE_heather_blocks_l2816_281677

/-- The number of blocks Heather ends up with after sharing -/
def blocks_remaining (initial : ℕ) (shared : ℕ) : ℕ :=
  initial - shared

/-- Theorem stating that Heather ends up with 45 blocks -/
theorem heather_blocks : blocks_remaining 86 41 = 45 := by
  sorry

end NUMINAMATH_CALUDE_heather_blocks_l2816_281677


namespace NUMINAMATH_CALUDE_specific_prism_volume_l2816_281651

/-- Represents a triangular prism -/
structure TriangularPrism :=
  (lateral_face_area : ℝ)
  (distance_to_face : ℝ)

/-- The volume of a triangular prism -/
def volume (prism : TriangularPrism) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific triangular prism -/
theorem specific_prism_volume :
  ∀ (prism : TriangularPrism),
    prism.lateral_face_area = 4 →
    prism.distance_to_face = 2 →
    volume prism = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_specific_prism_volume_l2816_281651


namespace NUMINAMATH_CALUDE_part_one_part_two_l2816_281680

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_arithmetic_sequence (B A C : ℝ) : Prop :=
  ∃ d : ℝ, A - B = C - A ∧ A - B = d

-- Theorem 1
theorem part_one (t : Triangle) (m : ℝ) 
  (h1 : is_arithmetic_sequence t.B t.A t.C)
  (h2 : t.a^2 - t.c^2 = t.b^2 - m*t.b*t.c) : 
  m = 1 := by sorry

-- Theorem 2
theorem part_two (t : Triangle)
  (h1 : is_arithmetic_sequence t.B t.A t.C)
  (h2 : t.a = Real.sqrt 3)
  (h3 : t.b + t.c = 3) :
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2816_281680


namespace NUMINAMATH_CALUDE_box_dimensions_l2816_281645

theorem box_dimensions (x : ℕ+) : 
  (((x : ℝ) + 3) * ((x : ℝ) - 4) * ((x : ℝ)^2 + 16) < 800 ∧ 
   (x : ℝ)^2 + 16 > 30) ↔ 
  (x = 4 ∨ x = 5) :=
sorry

end NUMINAMATH_CALUDE_box_dimensions_l2816_281645


namespace NUMINAMATH_CALUDE_fish_population_estimate_l2816_281649

/-- The number of fish initially tagged and returned to the pond -/
def tagged_fish : ℕ := 50

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- The number of tagged fish found in the second catch -/
def tagged_in_second_catch : ℕ := 2

/-- The total number of fish in the pond -/
def total_fish : ℕ := 1250

/-- Theorem stating that the given conditions lead to the correct total number of fish -/
theorem fish_population_estimate :
  (tagged_in_second_catch : ℚ) / second_catch = tagged_fish / total_fish :=
by sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l2816_281649


namespace NUMINAMATH_CALUDE_total_chickens_on_farm_l2816_281656

/-- Proves that the total number of chickens on a farm is 120, given the number of hens and their relation to roosters. -/
theorem total_chickens_on_farm (num_hens : ℕ) (num_roosters : ℕ) : 
  num_hens = 52 → 
  num_hens + 16 = num_roosters → 
  num_hens + num_roosters = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_chickens_on_farm_l2816_281656


namespace NUMINAMATH_CALUDE_fraction_ordering_l2816_281690

theorem fraction_ordering : 
  (20 : ℚ) / 16 < (18 : ℚ) / 14 ∧ (18 : ℚ) / 14 < (16 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2816_281690


namespace NUMINAMATH_CALUDE_year_2049_is_jisi_l2816_281614

/-- Represents the Heavenly Stems -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Heavenly Stems and Earthly Branches system -/
structure StemBranchYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

def next_stem (s : HeavenlyStem) : HeavenlyStem := sorry
def next_branch (b : EarthlyBranch) : EarthlyBranch := sorry

def advance_year (y : StemBranchYear) (n : ℕ) : StemBranchYear := sorry

theorem year_2049_is_jisi (year_2017 : StemBranchYear) 
  (h2017 : year_2017 = ⟨HeavenlyStem.Ding, EarthlyBranch.You⟩) :
  advance_year year_2017 32 = ⟨HeavenlyStem.Ji, EarthlyBranch.Si⟩ := by
  sorry

end NUMINAMATH_CALUDE_year_2049_is_jisi_l2816_281614


namespace NUMINAMATH_CALUDE_bhavan_score_percentage_l2816_281626

theorem bhavan_score_percentage (max_score : ℝ) (amar_percent : ℝ) (chetan_percent : ℝ) (average_mark : ℝ) :
  max_score = 900 →
  amar_percent = 64 →
  chetan_percent = 44 →
  average_mark = 432 →
  ∃ bhavan_percent : ℝ,
    bhavan_percent = 36 ∧
    3 * average_mark = (amar_percent / 100 * max_score) + (bhavan_percent / 100 * max_score) + (chetan_percent / 100 * max_score) :=
by sorry

end NUMINAMATH_CALUDE_bhavan_score_percentage_l2816_281626


namespace NUMINAMATH_CALUDE_divisor_not_zero_l2816_281676

theorem divisor_not_zero (a b : ℝ) : b ≠ 0 → ∃ (c : ℝ), a / b = c := by
  sorry

end NUMINAMATH_CALUDE_divisor_not_zero_l2816_281676


namespace NUMINAMATH_CALUDE_school_distance_proof_l2816_281644

/-- The distance to school in miles -/
def distance_to_school : ℝ := 5

/-- The speed of walking in miles per hour for the first scenario -/
def speed1 : ℝ := 4

/-- The speed of walking in miles per hour for the second scenario -/
def speed2 : ℝ := 5

/-- The time difference in hours between arriving early and late -/
def time_difference : ℝ := 0.25

theorem school_distance_proof :
  (distance_to_school / speed1 - distance_to_school / speed2 = time_difference) ∧
  (distance_to_school = 5) := by
  sorry

end NUMINAMATH_CALUDE_school_distance_proof_l2816_281644


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_less_than_350_l2816_281611

theorem largest_multiple_of_12_less_than_350 : 
  ∀ n : ℕ, n * 12 < 350 → n * 12 ≤ 348 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_less_than_350_l2816_281611


namespace NUMINAMATH_CALUDE_billy_feeds_twice_daily_l2816_281600

/-- The number of times Billy feeds his horses per day -/
def feedings_per_day (num_horses : ℕ) (oats_per_meal : ℕ) (total_oats : ℕ) (days : ℕ) : ℕ :=
  (total_oats / days) / (num_horses * oats_per_meal)

/-- Theorem: Billy feeds his horses twice a day -/
theorem billy_feeds_twice_daily :
  feedings_per_day 4 4 96 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_feeds_twice_daily_l2816_281600


namespace NUMINAMATH_CALUDE_diagonals_parity_iff_n_parity_l2816_281641

/-- The number of diagonals in a regular polygon with 2n+1 sides. -/
def num_diagonals (n : ℕ) : ℕ := (2 * n + 1).choose 2 - (2 * n + 1)

/-- Theorem: The number of diagonals in a regular polygon with 2n+1 sides is odd if and only if n is even. -/
theorem diagonals_parity_iff_n_parity (n : ℕ) (h : n > 1) :
  Odd (num_diagonals n) ↔ Even n := by
  sorry

end NUMINAMATH_CALUDE_diagonals_parity_iff_n_parity_l2816_281641


namespace NUMINAMATH_CALUDE_book_borrowing_growth_l2816_281634

/-- The number of books borrowed in 2015 -/
def books_2015 : ℕ := 7500

/-- The number of books borrowed in 2017 -/
def books_2017 : ℕ := 10800

/-- The average annual growth rate from 2015 to 2017 -/
def growth_rate : ℝ := 0.2

/-- The expected number of books borrowed in 2018 -/
def books_2018 : ℕ := 12960

/-- Theorem stating the relationship between the given values and the calculated growth rate and expected books for 2018 -/
theorem book_borrowing_growth :
  (books_2017 : ℝ) = books_2015 * (1 + growth_rate)^2 ∧
  books_2018 = Int.floor (books_2017 * (1 + growth_rate)) :=
sorry

end NUMINAMATH_CALUDE_book_borrowing_growth_l2816_281634


namespace NUMINAMATH_CALUDE_script_year_proof_l2816_281610

theorem script_year_proof : ∃! (year : ℕ), 
  year < 200 ∧ year^13 = 258145266804692077858261512663 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_script_year_proof_l2816_281610


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l2816_281658

/-- A quadratic function passing through the origin -/
def passes_through_origin (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c = 0

theorem quadratic_through_origin (a b c : ℝ) (h : a ≠ 0) :
  passes_through_origin a b c ↔ c = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l2816_281658


namespace NUMINAMATH_CALUDE_smaller_bill_denomination_l2816_281670

def total_amount : ℕ := 1000
def fraction_smaller : ℚ := 3 / 10
def larger_denomination : ℕ := 100
def total_bills : ℕ := 13

theorem smaller_bill_denomination :
  ∃ (smaller_denomination : ℕ),
    (fraction_smaller * total_amount) / smaller_denomination +
    ((1 - fraction_smaller) * total_amount) / larger_denomination = total_bills ∧
    smaller_denomination = 50 := by
  sorry

end NUMINAMATH_CALUDE_smaller_bill_denomination_l2816_281670


namespace NUMINAMATH_CALUDE_max_plates_buyable_l2816_281623

/-- The cost of a pan -/
def pan_cost : ℕ := 3

/-- The cost of a pot -/
def pot_cost : ℕ := 5

/-- The cost of a plate -/
def plate_cost : ℕ := 10

/-- The total budget -/
def total_budget : ℕ := 100

/-- The minimum number of each item to buy -/
def min_items : ℕ := 2

/-- A function to calculate the total cost of the purchase -/
def total_cost (pans pots plates : ℕ) : ℕ :=
  pan_cost * pans + pot_cost * pots + plate_cost * plates

/-- The main theorem stating the maximum number of plates that can be bought -/
theorem max_plates_buyable :
  ∃ (pans pots plates : ℕ),
    pans ≥ min_items ∧
    pots ≥ min_items ∧
    plates ≥ min_items ∧
    total_cost pans pots plates = total_budget ∧
    plates = 8 ∧
    ∀ (p : ℕ), p > plates →
      ∀ (x y : ℕ), x ≥ min_items → y ≥ min_items →
        total_cost x y p ≠ total_budget :=
by sorry

end NUMINAMATH_CALUDE_max_plates_buyable_l2816_281623


namespace NUMINAMATH_CALUDE_flowers_remaining_after_picking_l2816_281639

/-- The number of flowers remaining after Neznaika's picking --/
def remaining_flowers (total_flowers total_tulips watered_tulips picked_tulips unwatered_flowers : ℕ) : ℕ :=
  total_flowers - unwatered_flowers - picked_tulips

/-- Theorem stating the number of remaining flowers --/
theorem flowers_remaining_after_picking 
  (total_flowers : ℕ) 
  (total_tulips : ℕ)
  (total_peonies : ℕ)
  (watered_tulips : ℕ)
  (picked_tulips : ℕ)
  (unwatered_flowers : ℕ)
  (h1 : total_flowers = 30)
  (h2 : total_tulips = 15)
  (h3 : total_peonies = 15)
  (h4 : total_flowers = total_tulips + total_peonies)
  (h5 : watered_tulips = 10)
  (h6 : unwatered_flowers = 10)
  (h7 : picked_tulips = 6)
  : remaining_flowers total_flowers total_tulips watered_tulips picked_tulips unwatered_flowers = 19 :=
by
  sorry


end NUMINAMATH_CALUDE_flowers_remaining_after_picking_l2816_281639


namespace NUMINAMATH_CALUDE_no_solution_lcm_gcd_equation_l2816_281631

theorem no_solution_lcm_gcd_equation : ¬ ∃ (n : ℕ+), Nat.lcm n 120 = Nat.gcd n 120 + 300 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_lcm_gcd_equation_l2816_281631


namespace NUMINAMATH_CALUDE_dropped_students_scores_sum_l2816_281636

theorem dropped_students_scores_sum 
  (initial_students : ℕ) 
  (initial_average : ℝ) 
  (remaining_students : ℕ) 
  (new_average : ℝ) 
  (h1 : initial_students = 25) 
  (h2 : initial_average = 60.5) 
  (h3 : remaining_students = 23) 
  (h4 : new_average = 64.0) : 
  (initial_students : ℝ) * initial_average - (remaining_students : ℝ) * new_average = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_dropped_students_scores_sum_l2816_281636


namespace NUMINAMATH_CALUDE_tony_weightlifting_ratio_l2816_281695

/-- Given Tony's weightlifting capabilities, prove the ratio of his military press to curl weight. -/
theorem tony_weightlifting_ratio :
  ∀ (curl_weight military_press_weight squat_weight : ℝ),
    curl_weight = 90 →
    squat_weight = 5 * military_press_weight →
    squat_weight = 900 →
    military_press_weight / curl_weight = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_tony_weightlifting_ratio_l2816_281695


namespace NUMINAMATH_CALUDE_midpoint_property_l2816_281607

/-- Given two points A and B in the plane, if C is their midpoint,
    then 3 times the x-coordinate of C minus 5 times the y-coordinate of C equals 6. -/
theorem midpoint_property (A B : ℝ × ℝ) (h : A = (20, 10) ∧ B = (4, 2)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  3 * C.1 - 5 * C.2 = 6 := by sorry

end NUMINAMATH_CALUDE_midpoint_property_l2816_281607


namespace NUMINAMATH_CALUDE_election_vote_count_l2816_281675

theorem election_vote_count 
  (candidate1_percentage : ℝ) 
  (candidate2_votes : ℕ) 
  (total_votes : ℕ) : 
  candidate1_percentage = 0.7 →
  candidate2_votes = 240 →
  (candidate2_votes : ℝ) / total_votes = 1 - candidate1_percentage →
  total_votes = 800 := by
sorry

end NUMINAMATH_CALUDE_election_vote_count_l2816_281675


namespace NUMINAMATH_CALUDE_exists_monochromatic_isosceles_right_triangle_l2816_281683

/-- A color type with three possible values -/
inductive Color
  | Red
  | Green
  | Blue

/-- A point in the infinite grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point in the grid -/
def Coloring := GridPoint → Color

/-- An isosceles right triangle in the grid -/
structure IsoscelesRightTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  is_isosceles : (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2
  is_right : (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- The main theorem: In any coloring of an infinite grid with three colors,
    there exists an isosceles right triangle with vertices of the same color -/
theorem exists_monochromatic_isosceles_right_triangle (c : Coloring) :
  ∃ (t : IsoscelesRightTriangle), c t.p1 = c t.p2 ∧ c t.p2 = c t.p3 := by
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_isosceles_right_triangle_l2816_281683
