import Mathlib

namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l521_52101

/-- The line equation passing through a fixed point for all values of a -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 2) * x + (a + 1) * y + 6 = 0

/-- The fixed point coordinates -/
def fixed_point : ℝ × ℝ := (2, -2)

/-- Theorem stating that the line passes through the fixed point for all a -/
theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), line_equation a (fixed_point.1) (fixed_point.2) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l521_52101


namespace NUMINAMATH_CALUDE_puzzle_cost_theorem_l521_52127

/-- The cost of a small and large puzzle together -/
def puzzle_cost (small_cost large_cost : ℕ) : ℕ :=
  small_cost + large_cost

/-- The cost of one large and three small puzzles -/
def one_large_three_small (small_cost large_cost : ℕ) : ℕ :=
  large_cost + 3 * small_cost

theorem puzzle_cost_theorem (small_cost : ℕ) :
  puzzle_cost small_cost 15 = 23 ∧
  one_large_three_small small_cost 15 = 39 :=
sorry

end NUMINAMATH_CALUDE_puzzle_cost_theorem_l521_52127


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l521_52152

theorem triangle_angle_proof (A B : ℝ) (a b : ℝ) : 
  0 < A ∧ 0 < B ∧ A + B < π →  -- Ensuring A and B are valid triangle angles
  B = 2 * A →                  -- Given condition
  a / b = 1 / Real.sqrt 3 →    -- Given ratio of sides
  A = π / 6                    -- Conclusion (30° in radians)
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l521_52152


namespace NUMINAMATH_CALUDE_median_in_65_interval_l521_52175

/-- Represents a score interval with its lower bound and frequency -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (frequency : ℕ)

/-- Finds the interval containing the median score -/
def find_median_interval (intervals : List ScoreInterval) : Option ℕ :=
  let total_students := intervals.foldl (fun acc i => acc + i.frequency) 0
  let median_position := (total_students + 1) / 2
  let rec find_interval (acc : ℕ) (remaining : List ScoreInterval) : Option ℕ :=
    match remaining with
    | [] => none
    | i :: is =>
        if acc + i.frequency ≥ median_position then
          some i.lower_bound
        else
          find_interval (acc + i.frequency) is
  find_interval 0 intervals

theorem median_in_65_interval (score_data : List ScoreInterval) :
  score_data = [
    ⟨80, 20⟩, ⟨75, 15⟩, ⟨70, 10⟩, ⟨65, 25⟩, ⟨60, 15⟩, ⟨55, 15⟩
  ] →
  find_median_interval score_data = some 65 :=
by sorry

end NUMINAMATH_CALUDE_median_in_65_interval_l521_52175


namespace NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l521_52192

theorem range_of_a_for_false_proposition :
  (¬ ∃ x : ℝ, x^2 - a*x + 1 ≤ 0) ↔ -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l521_52192


namespace NUMINAMATH_CALUDE_comic_collection_problem_l521_52158

/-- The number of comic books that are in either Andrew's or John's collection, but not both -/
def exclusive_comics (shared comics_andrew comics_john_exclusive : ℕ) : ℕ :=
  (comics_andrew - shared) + comics_john_exclusive

theorem comic_collection_problem (shared comics_andrew comics_john_exclusive : ℕ) 
  (h1 : shared = 15)
  (h2 : comics_andrew = 22)
  (h3 : comics_john_exclusive = 10) :
  exclusive_comics shared comics_andrew comics_john_exclusive = 17 := by
  sorry

end NUMINAMATH_CALUDE_comic_collection_problem_l521_52158


namespace NUMINAMATH_CALUDE_income_average_difference_l521_52163

theorem income_average_difference (n : ℕ) (min_income max_income error_income : ℝ) :
  n > 0 ∧
  min_income = 8200 ∧
  max_income = 98000 ∧
  error_income = 980000 ∧
  n = 28 * 201000 →
  (error_income - max_income) / n = 882 :=
by sorry

end NUMINAMATH_CALUDE_income_average_difference_l521_52163


namespace NUMINAMATH_CALUDE_paul_pencil_production_l521_52162

/-- Calculates the number of pencils made per day given the initial stock, 
    final stock, number of pencils sold, and number of working days. -/
def pencils_per_day (initial_stock final_stock pencils_sold working_days : ℕ) : ℕ :=
  ((final_stock + pencils_sold) - initial_stock) / working_days

/-- Proves that Paul makes 100 pencils per day given the problem conditions. -/
theorem paul_pencil_production : 
  pencils_per_day 80 230 350 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_paul_pencil_production_l521_52162


namespace NUMINAMATH_CALUDE_pad_usage_duration_l521_52144

/-- Represents the number of sheets in a pad of paper -/
def sheets_per_pad : ℕ := 60

/-- Represents the number of working days per week -/
def working_days_per_week : ℕ := 3

/-- Represents the number of sheets used per working day -/
def sheets_per_day : ℕ := 12

/-- Calculates the number of weeks it takes to use a full pad of paper -/
def weeks_per_pad : ℚ :=
  sheets_per_pad / (working_days_per_week * sheets_per_day)

/-- Theorem stating that the rounded-up number of weeks to use a pad is 2 -/
theorem pad_usage_duration :
  ⌈weeks_per_pad⌉ = 2 := by sorry

end NUMINAMATH_CALUDE_pad_usage_duration_l521_52144


namespace NUMINAMATH_CALUDE_ellipse_tangent_intersection_l521_52105

-- Define the ellipse C
structure Ellipse :=
  (center : ℝ × ℝ)
  (a b : ℝ)
  (eccentricity : ℝ)

-- Define the parabola
def Parabola := {(x, y) : ℝ × ℝ | y^2 = 4*x}

-- Define a point on the ellipse
structure PointOnEllipse (C : Ellipse) :=
  (point : ℝ × ℝ)
  (on_ellipse : (point.1 - C.center.1)^2 / C.a^2 + (point.2 - C.center.2)^2 / C.b^2 = 1)

-- Define a tangent line to the ellipse
structure TangentLine (C : Ellipse) :=
  (point : PointOnEllipse C)
  (slope : ℝ)

-- Theorem statement
theorem ellipse_tangent_intersection 
  (C : Ellipse)
  (h1 : C.center = (0, 0))
  (h2 : C.eccentricity = Real.sqrt 2 / 2)
  (h3 : ∃ (f : ℝ × ℝ), f ∈ Parabola ∧ f ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 / C.a^2 + (p.2 - C.center.2)^2 / C.b^2 = C.eccentricity^2})
  (A : PointOnEllipse C)
  (tAB tAC : TangentLine C)
  (h4 : tAB.point = A ∧ tAC.point = A)
  (h5 : tAB.slope * tAC.slope = 1/4) :
  ∃ (P : ℝ × ℝ), P = (0, 3) ∧ 
    ∀ (B C : ℝ × ℝ), 
      (B.2 - A.point.2 = tAB.slope * (B.1 - A.point.1)) → 
      (C.2 - A.point.2 = tAC.slope * (C.1 - A.point.1)) → 
      (P.2 - B.2) / (P.1 - B.1) = (C.2 - B.2) / (C.1 - B.1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_tangent_intersection_l521_52105


namespace NUMINAMATH_CALUDE_min_value_theorem_l521_52118

theorem min_value_theorem (a b : ℝ) (ha : a > 0) 
  (h : ∀ x > 0, (a * x - 2) * (x^2 + b * x - 5) ≥ 0) :
  (∃ (b₀ : ℝ), b₀ + 4 / a = 2 * Real.sqrt 5) ∧ 
  (∀ (b₁ : ℝ), b₁ + 4 / a ≥ 2 * Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l521_52118


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l521_52198

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (al_count : ℕ) (o_count : ℕ) (h_count : ℕ) 
  (al_weight : ℝ) (o_weight : ℝ) (h_weight : ℝ) : ℝ :=
  al_count * al_weight + o_count * o_weight + h_count * h_weight

/-- The molecular weight of the compound AlO₃H₃ is 78.01 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 1 3 3 26.98 16.00 1.01 = 78.01 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l521_52198


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_one_l521_52172

theorem integral_sqrt_minus_one (f : ℝ → ℝ) :
  (∀ x, f x = Real.sqrt (1 - x^2) - 1) →
  (∫ x in (-1)..1, f x) = π / 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_one_l521_52172


namespace NUMINAMATH_CALUDE_special_points_divide_plane_into_four_regions_l521_52141

-- Define the set of points where one coordinate is four times the other
def special_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 4 * p.2 ∨ p.2 = 4 * p.1}

-- Define a function that counts the number of regions
def count_regions (S : Set (ℝ × ℝ)) : ℕ := sorry

-- Theorem statement
theorem special_points_divide_plane_into_four_regions :
  count_regions special_points = 4 := by sorry

end NUMINAMATH_CALUDE_special_points_divide_plane_into_four_regions_l521_52141


namespace NUMINAMATH_CALUDE_oatmeal_cookies_count_l521_52146

/-- The number of batches of chocolate chip cookies -/
def chocolate_chip_batches : ℕ := 2

/-- The number of cookies in each batch of chocolate chip cookies -/
def cookies_per_batch : ℕ := 3

/-- The total number of cookies baked -/
def total_cookies : ℕ := 10

/-- The number of oatmeal cookies -/
def oatmeal_cookies : ℕ := total_cookies - (chocolate_chip_batches * cookies_per_batch)

theorem oatmeal_cookies_count : oatmeal_cookies = 4 := by
  sorry

end NUMINAMATH_CALUDE_oatmeal_cookies_count_l521_52146


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l521_52114

theorem fraction_product_theorem :
  (7 / 5 : ℚ) * (8 / 12 : ℚ) * (21 / 15 : ℚ) * (16 / 24 : ℚ) * 
  (35 / 25 : ℚ) * (20 / 30 : ℚ) * (49 / 35 : ℚ) * (32 / 48 : ℚ) = 38416 / 50625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l521_52114


namespace NUMINAMATH_CALUDE_egypt_tour_promotion_l521_52164

theorem egypt_tour_promotion (total_tourists : ℕ) (free_tourists : ℕ) : 
  (13 : ℕ) + 4 * free_tourists = total_tourists ∧ 
  free_tourists + (100 : ℕ) = total_tourists →
  free_tourists = 29 := by
sorry

end NUMINAMATH_CALUDE_egypt_tour_promotion_l521_52164


namespace NUMINAMATH_CALUDE_expression_simplification_l521_52132

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (3 * a + b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹) = (a * b)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l521_52132


namespace NUMINAMATH_CALUDE_petya_cannot_guarantee_win_l521_52102

/-- Represents a position on the 9x9 board -/
structure Position :=
  (x : Fin 9)
  (y : Fin 9)

/-- Represents a direction of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a player in the game -/
inductive Player
  | Petya
  | Vasya

/-- The game state -/
structure GameState :=
  (position : Position)
  (lastDirection : Direction)
  (currentPlayer : Player)

/-- Checks if a move is valid for a given player -/
def isValidMove (player : Player) (lastDir : Direction) (newDir : Direction) : Prop :=
  match player with
  | Player.Petya => newDir = lastDir ∨ newDir = Direction.Right
  | Player.Vasya => newDir = lastDir ∨ newDir = Direction.Left

/-- Checks if a position is on the board -/
def isOnBoard (pos : Position) : Prop :=
  0 ≤ pos.x ∧ pos.x < 9 ∧ 0 ≤ pos.y ∧ pos.y < 9

/-- Theorem stating that Petya cannot guarantee a win -/
theorem petya_cannot_guarantee_win :
  ∀ (strategy : GameState → Direction),
  ∃ (counterStrategy : GameState → Direction),
  ∃ (finalState : GameState),
  (finalState.currentPlayer = Player.Petya ∧ 
   ¬∃ (dir : Direction), isValidMove Player.Petya finalState.lastDirection dir ∧ 
                         isOnBoard (finalState.position)) :=
sorry

end NUMINAMATH_CALUDE_petya_cannot_guarantee_win_l521_52102


namespace NUMINAMATH_CALUDE_non_right_triangles_count_l521_52115

-- Define the points on the grid
def Point := Fin 6

-- Define the grid
def Grid := Point → ℝ × ℝ

-- Define the specific grid layout
def grid_layout : Grid := sorry

-- Define a function to check if a triangle is right-angled
def is_right_angled (p q r : Point) (g : Grid) : Prop := sorry

-- Define a function to count non-right-angled triangles
def count_non_right_triangles (g : Grid) : ℕ := sorry

-- Theorem statement
theorem non_right_triangles_count :
  count_non_right_triangles grid_layout = 4 := by sorry

end NUMINAMATH_CALUDE_non_right_triangles_count_l521_52115


namespace NUMINAMATH_CALUDE_complex_equation_solution_l521_52194

theorem complex_equation_solution :
  ∀ y : ℝ,
  let z₁ : ℂ := 3 + y * Complex.I
  let z₂ : ℂ := 2 - Complex.I
  z₁ / z₂ = 1 + Complex.I →
  y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l521_52194


namespace NUMINAMATH_CALUDE_ribbon_length_reduction_l521_52109

theorem ribbon_length_reduction (original_length : ℝ) (ratio_original : ℝ) (ratio_new : ℝ) (new_length : ℝ) : 
  original_length = 55 →
  ratio_original = 11 →
  ratio_new = 7 →
  new_length = (original_length * ratio_new) / ratio_original →
  new_length = 35 := by
sorry

end NUMINAMATH_CALUDE_ribbon_length_reduction_l521_52109


namespace NUMINAMATH_CALUDE_alan_roof_weight_l521_52180

/-- The number of pine trees in Alan's backyard -/
def num_trees : ℕ := 8

/-- The number of pine cones each tree drops -/
def cones_per_tree : ℕ := 200

/-- The percentage of pine cones that fall on the roof -/
def roof_percentage : ℚ := 30 / 100

/-- The weight of each pine cone in ounces -/
def cone_weight : ℚ := 4

/-- The total weight of pine cones on Alan's roof in ounces -/
def roof_weight : ℚ := num_trees * cones_per_tree * roof_percentage * cone_weight

theorem alan_roof_weight : roof_weight = 1920 := by sorry

end NUMINAMATH_CALUDE_alan_roof_weight_l521_52180


namespace NUMINAMATH_CALUDE_tan_75_degrees_l521_52123

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_75_degrees_l521_52123


namespace NUMINAMATH_CALUDE_house_wall_nails_l521_52136

/-- The number of nails needed per plank -/
def nails_per_plank : ℕ := 2

/-- The number of planks used for the house wall -/
def planks_used : ℕ := 16

/-- The total number of nails needed for the house wall -/
def total_nails : ℕ := nails_per_plank * planks_used

theorem house_wall_nails : total_nails = 32 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_nails_l521_52136


namespace NUMINAMATH_CALUDE_discount_from_profit_l521_52183

/-- Represents a car sale transaction -/
structure CarSale where
  originalPrice : ℝ
  discountRate : ℝ
  profitRate : ℝ
  sellIncrease : ℝ

/-- Theorem stating the relationship between discount and profit in a specific car sale scenario -/
theorem discount_from_profit (sale : CarSale) 
  (h1 : sale.profitRate = 0.28000000000000004)
  (h2 : sale.sellIncrease = 0.60) : 
  sale.discountRate = 0.5333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_discount_from_profit_l521_52183


namespace NUMINAMATH_CALUDE_total_amount_is_80000_l521_52161

/-- Represents the problem of dividing money between two investments with different interest rates -/
def MoneyDivisionProblem (total_profit interest_10_amount : ℕ) : Prop :=
  ∃ (total_amount interest_20_amount : ℕ),
    -- Total amount is the sum of both investments
    total_amount = interest_10_amount + interest_20_amount ∧
    -- Profit calculation
    total_profit = (interest_10_amount * 10 / 100) + (interest_20_amount * 20 / 100)

/-- Theorem stating that given the problem conditions, the total amount is 80000 -/
theorem total_amount_is_80000 :
  MoneyDivisionProblem 9000 70000 → ∃ total_amount : ℕ, total_amount = 80000 :=
sorry

end NUMINAMATH_CALUDE_total_amount_is_80000_l521_52161


namespace NUMINAMATH_CALUDE_negative_eight_million_scientific_notation_l521_52128

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- Conversion function from ℝ to ScientificNotation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem negative_eight_million_scientific_notation :
  toScientificNotation (-8206000) = ScientificNotation.mk (-8.206) 6 sorry sorry := by
  sorry

end NUMINAMATH_CALUDE_negative_eight_million_scientific_notation_l521_52128


namespace NUMINAMATH_CALUDE_angle_relationship_l521_52107

theorem angle_relationship (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = (1/2) * Real.sin (α + β)) : α < β := by
  sorry

end NUMINAMATH_CALUDE_angle_relationship_l521_52107


namespace NUMINAMATH_CALUDE_reinforcement_theorem_l521_52167

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days passed before reinforcement, and remaining provision duration after reinforcement. -/
def reinforcement_size (initial_garrison : ℕ) (initial_duration : ℕ) 
    (days_before_reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  (initial_garrison * initial_duration - initial_garrison * days_before_reinforcement) / remaining_duration - initial_garrison

/-- Theorem stating that given the problem conditions, the reinforcement size is 2000. -/
theorem reinforcement_theorem : 
  reinforcement_size 2000 40 20 10 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_theorem_l521_52167


namespace NUMINAMATH_CALUDE_ruth_gave_53_stickers_l521_52148

/-- The number of stickers Janet initially had -/
def initial_stickers : ℕ := 3

/-- The total number of stickers Janet has after receiving more from Ruth -/
def final_stickers : ℕ := 56

/-- The number of stickers Ruth gave to Janet -/
def stickers_from_ruth : ℕ := final_stickers - initial_stickers

theorem ruth_gave_53_stickers : stickers_from_ruth = 53 := by
  sorry

end NUMINAMATH_CALUDE_ruth_gave_53_stickers_l521_52148


namespace NUMINAMATH_CALUDE_subset_implies_membership_l521_52187

theorem subset_implies_membership (P Q : Set α) 
  (h_nonempty_P : P.Nonempty) (h_nonempty_Q : Q.Nonempty) (h_subset : P ⊆ Q) : 
  ∀ x ∈ P, x ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_membership_l521_52187


namespace NUMINAMATH_CALUDE_professional_ratio_l521_52117

/-- Represents a professional group with engineers, doctors, and lawyers. -/
structure ProfessionalGroup where
  numEngineers : ℕ
  numDoctors : ℕ
  numLawyers : ℕ

/-- The average age of the entire group -/
def groupAverageAge : ℝ := 45

/-- The average age of engineers -/
def engineerAverageAge : ℝ := 40

/-- The average age of doctors -/
def doctorAverageAge : ℝ := 50

/-- The average age of lawyers -/
def lawyerAverageAge : ℝ := 60

/-- Theorem stating the ratio of professionals in the group -/
theorem professional_ratio (group : ProfessionalGroup) :
  group.numEngineers * (doctorAverageAge - groupAverageAge) =
  group.numDoctors * (groupAverageAge - engineerAverageAge) ∧
  group.numEngineers * (lawyerAverageAge - groupAverageAge) =
  3 * group.numLawyers * (groupAverageAge - engineerAverageAge) :=
sorry

end NUMINAMATH_CALUDE_professional_ratio_l521_52117


namespace NUMINAMATH_CALUDE_ratio_lcm_problem_l521_52126

theorem ratio_lcm_problem (a b : ℕ+) (h1 : a.val * 4 = b.val * 3) 
  (h2 : Nat.lcm a.val b.val = 180) (h3 : a.val = 45 ∨ b.val = 45) :
  (if a.val = 45 then b.val else a.val) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ratio_lcm_problem_l521_52126


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l521_52113

theorem exactly_one_greater_than_one 
  (x y z : ℝ) 
  (pos_x : x > 0) 
  (pos_y : y > 0) 
  (pos_z : z > 0) 
  (product_one : x * y * z = 1) 
  (sum_inequality : x + y + z > 1/x + 1/y + 1/z) : 
  (x > 1 ∧ y ≤ 1 ∧ z ≤ 1) ∨ 
  (x ≤ 1 ∧ y > 1 ∧ z ≤ 1) ∨ 
  (x ≤ 1 ∧ y ≤ 1 ∧ z > 1) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l521_52113


namespace NUMINAMATH_CALUDE_problem_solution_l521_52142

theorem problem_solution :
  (∀ x : ℝ, (1 : ℝ) > 0 ∧ x^2 - 4*1*x + 3*1^2 < 0 ∧ (x-3)/(x-2) ≤ 0 → 2 < x ∧ x < 3) ∧
  (∀ a : ℝ, a > 0 ∧ 
    (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 ≥ 0 → (x-3)/(x-2) > 0) ∧
    (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 ∧ (x-3)/(x-2) > 0) →
    1 < a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l521_52142


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_union_of_A_and_B_range_of_p_l521_52159

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}
def C (p : ℝ) : Set ℝ := {x | 4*x + p < 0}

-- State the theorems
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} := by sorry

theorem union_of_A_and_B : A ∪ B = Set.univ := by sorry

theorem range_of_p (p : ℝ) : C p ⊆ A → p ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_union_of_A_and_B_range_of_p_l521_52159


namespace NUMINAMATH_CALUDE_min_value_theorem_l521_52129

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let f := fun x : ℝ => a * x^3 + b * x + 2^x
  (∀ x ∈ Set.Icc 0 1, f x ≤ 4) ∧ (∃ x ∈ Set.Icc 0 1, f x = 4) →
  (∀ x ∈ Set.Icc (-1) 0, f x ≥ -3/2) ∧ (∃ x ∈ Set.Icc (-1) 0, f x = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l521_52129


namespace NUMINAMATH_CALUDE_inequality_solution_set_l521_52116

theorem inequality_solution_set (d : ℝ) : 
  (d / 4 ≤ 3 - d ∧ 3 - d < 1 - 2*d) ↔ (-2 < d ∧ d ≤ 12/5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l521_52116


namespace NUMINAMATH_CALUDE_unique_solution_values_non_monotonic_range_l521_52135

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + (a + 2) * x + b

-- Part 1
theorem unique_solution_values (a b : ℝ) :
  f a b (-1) = -2 →
  (∃! x, f a b x = 2 * x) →
  a = 2 ∧ b = 1 := by sorry

-- Part 2
theorem non_monotonic_range (a b : ℝ) :
  (∃ x y, x ∈ Set.Icc (-2 : ℝ) 2 ∧ 
          y ∈ Set.Icc (-2 : ℝ) 2 ∧ 
          x < y ∧ 
          f a b x > f a b y) →
  -6 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_values_non_monotonic_range_l521_52135


namespace NUMINAMATH_CALUDE_x_thirteen_percent_greater_than_80_l521_52197

theorem x_thirteen_percent_greater_than_80 :
  let x := 80 * (1 + 13 / 100)
  x = 90.4 := by sorry

end NUMINAMATH_CALUDE_x_thirteen_percent_greater_than_80_l521_52197


namespace NUMINAMATH_CALUDE_prob_sum_5_is_one_ninth_l521_52120

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := sides * sides

/-- The number of favorable outcomes (sum of 5) -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two dice -/
def prob_sum_5 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_5_is_one_ninth :
  prob_sum_5 = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_sum_5_is_one_ninth_l521_52120


namespace NUMINAMATH_CALUDE_vector_addition_subtraction_l521_52140

theorem vector_addition_subtraction :
  let v1 : Fin 3 → ℝ := ![4, -3, 7]
  let v2 : Fin 3 → ℝ := ![-1, 5, 2]
  let v3 : Fin 3 → ℝ := ![2, -4, 9]
  v1 + v2 - v3 = ![1, 6, 0] := by sorry

end NUMINAMATH_CALUDE_vector_addition_subtraction_l521_52140


namespace NUMINAMATH_CALUDE_binomial_26_6_l521_52179

theorem binomial_26_6 (h1 : Nat.choose 24 5 = 42504) 
                      (h2 : Nat.choose 24 6 = 134596) 
                      (h3 : Nat.choose 23 5 = 33649) : 
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l521_52179


namespace NUMINAMATH_CALUDE_complex_subtraction_l521_52165

theorem complex_subtraction : (5 * Complex.I) - (2 + 2 * Complex.I) = -2 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l521_52165


namespace NUMINAMATH_CALUDE_base2_to_base4_conversion_l521_52193

def base2_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List (Fin 4) :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

theorem base2_to_base4_conversion :
  decimal_to_base4 (base2_to_decimal [true, false, true, true, false, true, true, true, false]) =
  [1, 1, 2, 3, 2] :=
by sorry

end NUMINAMATH_CALUDE_base2_to_base4_conversion_l521_52193


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l521_52195

theorem consecutive_integers_product_812_sum_57 (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l521_52195


namespace NUMINAMATH_CALUDE_sum_of_cubes_l521_52121

theorem sum_of_cubes (x y z : ℕ+) :
  (x + y + z : ℕ+)^3 - x^3 - y^3 - z^3 = 504 →
  (x : ℕ) + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l521_52121


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l521_52188

/-- Given two planar vectors a and b, where a is perpendicular to b,
    prove that the value of m in a = (m, m-1) and b = (1, 2) is 2/3. -/
theorem perpendicular_vectors_m_value :
  ∀ (m : ℝ),
  let a : ℝ × ℝ := (m, m - 1)
  let b : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- dot product = 0 for perpendicular vectors
  m = 2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l521_52188


namespace NUMINAMATH_CALUDE_average_correction_problem_l521_52160

theorem average_correction_problem (initial_avg : ℚ) (misread : ℚ) (correct : ℚ) (correct_avg : ℚ) :
  initial_avg = 14 →
  misread = 26 →
  correct = 36 →
  correct_avg = 15 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_avg - misread + correct = (n : ℚ) * correct_avg ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_correction_problem_l521_52160


namespace NUMINAMATH_CALUDE_equation_solution_iff_m_equals_p_l521_52177

theorem equation_solution_iff_m_equals_p (p m : ℕ) (hp : Prime p) (hm : m ≥ 2) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x, y) ≠ (1, 1) ∧
    (x^p + y^p) / 2 = ((x + y) / 2)^m) ↔ m = p :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_iff_m_equals_p_l521_52177


namespace NUMINAMATH_CALUDE_laundry_items_not_done_l521_52145

theorem laundry_items_not_done (short_sleeve : ℕ) (long_sleeve : ℕ) (socks : ℕ) (handkerchiefs : ℕ)
  (shirts_washed : ℕ) (socks_folded : ℕ) (handkerchiefs_sorted : ℕ)
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 27)
  (h3 : socks = 50)
  (h4 : handkerchiefs = 34)
  (h5 : shirts_washed = 20)
  (h6 : socks_folded = 30)
  (h7 : handkerchiefs_sorted = 16) :
  (short_sleeve + long_sleeve - shirts_washed) + (socks - socks_folded) + (handkerchiefs - handkerchiefs_sorted) = 54 :=
by sorry

end NUMINAMATH_CALUDE_laundry_items_not_done_l521_52145


namespace NUMINAMATH_CALUDE_total_donation_theorem_l521_52112

def initial_donation : ℝ := 1707

def percentage_increases : List ℝ := [0.03, 0.05, 0.08, 0.02, 0.10, 0.04, 0.06, 0.09, 0.07, 0.03, 0.05]

def calculate_monthly_donation (prev_donation : ℝ) (percentage_increase : ℝ) : ℝ :=
  prev_donation * (1 + percentage_increase)

def calculate_total_donation (initial : ℝ) (increases : List ℝ) : ℝ :=
  let monthly_donations := increases.scanl calculate_monthly_donation initial
  initial + monthly_donations.sum

theorem total_donation_theorem :
  calculate_total_donation initial_donation percentage_increases = 29906.10 := by
  sorry

end NUMINAMATH_CALUDE_total_donation_theorem_l521_52112


namespace NUMINAMATH_CALUDE_inequality_implies_product_l521_52130

theorem inequality_implies_product (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) 
  (h3 : 4 * Real.log x + 2 * Real.log y ≥ x^2 + 4*y - 4) : 
  x * y = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_product_l521_52130


namespace NUMINAMATH_CALUDE_billys_songs_l521_52138

theorem billys_songs (total_songs : ℕ) (can_play : ℕ) (to_learn : ℕ) :
  total_songs = 52 →
  can_play = 24 →
  to_learn = 28 →
  can_play = total_songs - to_learn :=
by sorry

end NUMINAMATH_CALUDE_billys_songs_l521_52138


namespace NUMINAMATH_CALUDE_sum_of_specific_common_multiples_l521_52119

theorem sum_of_specific_common_multiples (a b : ℕ) (h : Nat.lcm a b = 21) :
  (9 * 21) + (10 * 21) + (11 * 21) = 630 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_common_multiples_l521_52119


namespace NUMINAMATH_CALUDE_noon_temperature_l521_52124

/-- The noon temperature -/
def T : ℝ := sorry

/-- The temperature at 4:00 PM -/
def T_4pm : ℝ := T + 8

/-- The temperature at 8:00 PM -/
def T_8pm : ℝ := T_4pm - 11

theorem noon_temperature : T = 4 := by
  have h1 : T_8pm = T + 1 := sorry
  sorry

end NUMINAMATH_CALUDE_noon_temperature_l521_52124


namespace NUMINAMATH_CALUDE_factory_production_l521_52176

/-- Given a factory that produces a certain number of toys per week and workers
    that work a certain number of days per week, calculate the number of toys
    produced each day (rounded down). -/
def toysPerDay (toysPerWeek : ℕ) (daysWorked : ℕ) : ℕ :=
  toysPerWeek / daysWorked

/-- Theorem stating that for a factory producing 6400 toys per week with workers
    working 3 days a week, the number of toys produced each day is 2133. -/
theorem factory_production :
  toysPerDay 6400 3 = 2133 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l521_52176


namespace NUMINAMATH_CALUDE_roots_of_f_l521_52190

-- Define the polynomial function f
def f (x : ℝ) : ℝ := -3 * (x + 5)^2 + 45 * (x + 5) - 108

-- State the theorem
theorem roots_of_f :
  (f 7 = 0) ∧ (f (-2) = 0) ∧
  (∀ x : ℝ, f x = 0 → x = 7 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_f_l521_52190


namespace NUMINAMATH_CALUDE_locus_of_circle_centers_is_hyperbola_l521_52111

/-- The locus of points equidistant from two fixed points forms a hyperbola --/
theorem locus_of_circle_centers_is_hyperbola 
  (M : ℝ × ℝ) -- Point M(x, y)
  (C₁ : ℝ × ℝ := (0, -1)) -- Center of circle C₁
  (C₂ : ℝ × ℝ := (0, 4)) -- Center of circle C₂
  (h : Real.sqrt ((M.1 - C₂.1)^2 + (M.2 - C₂.2)^2) - 
       Real.sqrt ((M.1 - C₁.1)^2 + (M.2 - C₁.2)^2) = 1) :
  -- The statement that M lies on a hyperbola
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (M.1^2 / a^2) - (M.2^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_circle_centers_is_hyperbola_l521_52111


namespace NUMINAMATH_CALUDE_subtracted_value_proof_l521_52157

theorem subtracted_value_proof (n : ℝ) (x : ℝ) : 
  n = 15.0 → 3 * n - x = 40 → x = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_proof_l521_52157


namespace NUMINAMATH_CALUDE_total_weight_carrots_cucumbers_l521_52106

theorem total_weight_carrots_cucumbers : 
  ∀ (weight_carrots : ℝ) (weight_ratio : ℝ),
    weight_carrots = 250 →
    weight_ratio = 2.5 →
    weight_carrots + weight_ratio * weight_carrots = 875 :=
by
  sorry

end NUMINAMATH_CALUDE_total_weight_carrots_cucumbers_l521_52106


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l521_52133

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1

/-- Two vectors have the same direction if their corresponding components have the same sign -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 ≥ 0) ∧ (a.2 * b.2 ≥ 0)

theorem collinear_vectors_m_value :
  ∀ (m : ℝ),
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4, m)
  collinear a b → same_direction a b → m = 2 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l521_52133


namespace NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l521_52139

theorem ceiling_squared_negative_fraction : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l521_52139


namespace NUMINAMATH_CALUDE_cheaper_plan_threshold_min_gigabytes_for_cheaper_plan_y_l521_52181

/-- Represents the cost of an internet plan in cents -/
def PlanCost (initialFee : ℕ) (costPerGB : ℕ) (gigabytes : ℕ) : ℕ :=
  initialFee * 100 + costPerGB * gigabytes

theorem cheaper_plan_threshold :
  ∀ g : ℕ, PlanCost 0 20 g ≤ PlanCost 30 10 g ↔ g ≤ 300 :=
by sorry

theorem min_gigabytes_for_cheaper_plan_y :
  ∃ g : ℕ, g = 301 ∧
    (∀ h : ℕ, PlanCost 0 20 h > PlanCost 30 10 h → h ≥ g) ∧
    PlanCost 0 20 g > PlanCost 30 10 g :=
by sorry

end NUMINAMATH_CALUDE_cheaper_plan_threshold_min_gigabytes_for_cheaper_plan_y_l521_52181


namespace NUMINAMATH_CALUDE_circle_area_ratio_l521_52174

theorem circle_area_ratio (R_C R_D : ℝ) (h : R_C > 0 ∧ R_D > 0) :
  (60 / 360 * (2 * Real.pi * R_C) = 2 * (40 / 360 * (2 * Real.pi * R_D))) →
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l521_52174


namespace NUMINAMATH_CALUDE_invalid_transformation_l521_52108

theorem invalid_transformation (x y m : ℝ) : 
  ¬(∀ (x y m : ℝ), x = y → x / m = y / m) :=
sorry

end NUMINAMATH_CALUDE_invalid_transformation_l521_52108


namespace NUMINAMATH_CALUDE_infinitely_many_n_not_equal_l521_52125

/-- For any positive integers a and b greater than 1, there are infinitely many n
    such that φ(a^n - 1) ≠ b^m - b^t for any positive integers m and t. -/
theorem infinitely_many_n_not_equal (a b : ℕ) (ha : a > 1) (hb : b > 1) :
  Set.Infinite {n : ℕ | ∀ m t : ℕ, m > 0 → t > 0 → Nat.totient (a^n - 1) ≠ b^m - b^t} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_not_equal_l521_52125


namespace NUMINAMATH_CALUDE_negative_square_and_subtraction_l521_52155

theorem negative_square_and_subtraction :
  (-4^2 = -16) ∧ ((-3) - (-6) = 3) := by sorry

end NUMINAMATH_CALUDE_negative_square_and_subtraction_l521_52155


namespace NUMINAMATH_CALUDE_camryn_practice_schedule_l521_52110

/-- Represents the number of days between Camryn's trumpet practices -/
def trumpet_interval : ℕ := 11

/-- Represents the number of days until Camryn practices both instruments again -/
def next_joint_practice : ℕ := 33

/-- Represents the number of days between Camryn's flute practices -/
def flute_interval : ℕ := 3

theorem camryn_practice_schedule :
  (trumpet_interval > 1) ∧
  (flute_interval > 1) ∧
  (flute_interval < trumpet_interval) ∧
  (next_joint_practice % trumpet_interval = 0) ∧
  (next_joint_practice % flute_interval = 0) :=
by sorry

end NUMINAMATH_CALUDE_camryn_practice_schedule_l521_52110


namespace NUMINAMATH_CALUDE_min_value_hyperbola_ellipse_foci_l521_52191

/-- The minimum value of (4/m + 1/n) given the conditions of the problem -/
theorem min_value_hyperbola_ellipse_foci (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, x^2/m - y^2/n = 1) → 
  (∃ x y : ℝ, x^2/5 + y^2/2 = 1) → 
  (∀ x y : ℝ, x^2/m - y^2/n = 1 ↔ x^2/5 + y^2/2 = 1) → 
  (4/m + 1/n ≥ 3 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 4/m₀ + 1/n₀ = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_hyperbola_ellipse_foci_l521_52191


namespace NUMINAMATH_CALUDE_calculation_proof_l521_52137

theorem calculation_proof : (4 - Real.sqrt 3) ^ 0 - 3 * Real.tan (π / 3) - (-1/2)⁻¹ + Real.sqrt 12 = 3 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l521_52137


namespace NUMINAMATH_CALUDE_inequality_proof_l521_52150

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / Real.sqrt (b + 1 / a + 1 / 2) + 1 / Real.sqrt (c + 1 / b + 1 / 2) + 1 / Real.sqrt (a + 1 / c + 1 / 2) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l521_52150


namespace NUMINAMATH_CALUDE_fourth_root_of_four_powers_l521_52131

theorem fourth_root_of_four_powers : (4^7 + 4^7 + 4^7 + 4^7 : ℝ)^(1/4) = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_four_powers_l521_52131


namespace NUMINAMATH_CALUDE_binary_representation_theorem_l521_52104

def is_multiple_of_17 (n : ℕ) : Prop := ∃ k : ℕ, n = 17 * k

def binary_ones_count (n : ℕ) : ℕ := (n.digits 2).count 1

def binary_zeros_count (n : ℕ) : ℕ := (n.digits 2).length - binary_ones_count n

theorem binary_representation_theorem (n : ℕ) 
  (h1 : is_multiple_of_17 n) 
  (h2 : binary_ones_count n = 3) : 
  (binary_zeros_count n ≥ 6) ∧ 
  (binary_zeros_count n = 7 → Even n) := by
sorry

end NUMINAMATH_CALUDE_binary_representation_theorem_l521_52104


namespace NUMINAMATH_CALUDE_coord_sum_of_point_on_line_l521_52168

/-- Given two points A and B in a 2D plane, where A is at the origin and B is on the line y = 5,
    if the slope of segment AB is 3/4, then the sum of the x- and y-coordinates of B is 35/3. -/
theorem coord_sum_of_point_on_line (B : ℝ × ℝ) : 
  B.2 = 5 →  -- B is on the line y = 5
  (B.2 - 0) / (B.1 - 0) = 3/4 →  -- slope of AB is 3/4
  B.1 + B.2 = 35/3 := by
sorry

end NUMINAMATH_CALUDE_coord_sum_of_point_on_line_l521_52168


namespace NUMINAMATH_CALUDE_students_allowance_l521_52199

theorem students_allowance (allowance : ℚ) : 
  (2 / 3 : ℚ) * (2 / 5 : ℚ) * allowance = 6 / 10 → 
  allowance = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_students_allowance_l521_52199


namespace NUMINAMATH_CALUDE_log_equation_solution_l521_52100

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the logarithm function for base 9
noncomputable def log9 (x : ℝ) : ℝ := Real.log x / Real.log 9

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  log3 y + log9 y = 5 → y = 3^(10/3) := by
  sorry


end NUMINAMATH_CALUDE_log_equation_solution_l521_52100


namespace NUMINAMATH_CALUDE_katie_miles_ran_l521_52184

theorem katie_miles_ran (katie_miles : ℝ) (adam_miles : ℝ) : 
  adam_miles = 3 * katie_miles →
  katie_miles + adam_miles = 240 →
  katie_miles = 60 := by
sorry

end NUMINAMATH_CALUDE_katie_miles_ran_l521_52184


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l521_52147

theorem quadratic_equation_solution (p : ℝ) (α β : ℝ) : 
  (∀ x, x^2 + p*x + p = 0 ↔ x = α ∨ x = β) →
  (α^2 + β^2 = 3) →
  p = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l521_52147


namespace NUMINAMATH_CALUDE_rachel_toys_l521_52182

theorem rachel_toys (jason_toys : ℕ) (john_toys : ℕ) (rachel_toys : ℕ)
  (h1 : jason_toys = 21)
  (h2 : jason_toys = 3 * john_toys)
  (h3 : john_toys = rachel_toys + 6) :
  rachel_toys = 1 := by
  sorry

end NUMINAMATH_CALUDE_rachel_toys_l521_52182


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l521_52189

/-- The eccentricity of an ellipse with equation 16x²+4y²=1 is √3/2 -/
theorem ellipse_eccentricity : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ (x y : ℝ), 16 * x^2 + 4 * y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a^2 - b^2) / a^2 = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l521_52189


namespace NUMINAMATH_CALUDE_johns_outfit_cost_l521_52149

/-- The cost of John's outfit given the cost of his pants and the relative cost of his shirt. -/
theorem johns_outfit_cost (pants_cost : ℝ) (shirt_relative_cost : ℝ) : 
  pants_cost = 50 →
  shirt_relative_cost = 0.6 →
  pants_cost + (pants_cost + pants_cost * shirt_relative_cost) = 130 := by
  sorry

end NUMINAMATH_CALUDE_johns_outfit_cost_l521_52149


namespace NUMINAMATH_CALUDE_minimum_buses_l521_52154

theorem minimum_buses (max_capacity : ℕ) (total_students : ℕ) (h1 : max_capacity = 45) (h2 : total_students = 495) :
  ∃ n : ℕ, n * max_capacity ≥ total_students ∧ ∀ m : ℕ, m * max_capacity ≥ total_students → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_buses_l521_52154


namespace NUMINAMATH_CALUDE_surf_festival_attendance_l521_52134

/-- The number of additional surfers on the second day of the Rip Curl Myrtle Beach Surf Festival --/
def additional_surfers : ℕ := 600

theorem surf_festival_attendance :
  let first_day : ℕ := 1500
  let third_day : ℕ := (2 : ℕ) * first_day / (5 : ℕ)
  let total_surfers : ℕ := first_day + (first_day + additional_surfers) + third_day
  let average_surfers : ℕ := 1400
  total_surfers / 3 = average_surfers :=
by sorry

end NUMINAMATH_CALUDE_surf_festival_attendance_l521_52134


namespace NUMINAMATH_CALUDE_minimum_gloves_needed_l521_52122

theorem minimum_gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) : 
  participants = 82 → gloves_per_participant = 2 → participants * gloves_per_participant = 164 := by
sorry

end NUMINAMATH_CALUDE_minimum_gloves_needed_l521_52122


namespace NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l521_52151

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B : Set ℝ := {x | 4 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Theorem for the complement of A ∩ B in U
theorem complement_A_inter_B :
  (A ∩ B)ᶜ = {x | x ≤ 4 ∨ x > 5} :=
sorry

-- Theorem for the range of values for a
theorem range_of_a (a : ℝ) (h : A ∪ B ⊆ C a) :
  a ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l521_52151


namespace NUMINAMATH_CALUDE_max_area_PCD_l521_52169

/-- Definition of the ellipse Γ -/
def Γ (a b x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

/-- Definition of point A (left vertex) -/
def A (a : ℝ) : ℝ × ℝ := (-a, 0)

/-- Definition of point B (top vertex) -/
def B (b : ℝ) : ℝ × ℝ := (0, b)

/-- Definition of point P on the ellipse in the fourth quadrant -/
def P (a b : ℝ) : {p : ℝ × ℝ // Γ a b p.1 p.2 ∧ p.1 > 0 ∧ p.2 < 0} := sorry

/-- Definition of point C (intersection of PA with y-axis) -/
def C (a b : ℝ) : ℝ × ℝ := sorry

/-- Definition of point D (intersection of PB with x-axis) -/
def D (a b : ℝ) : ℝ × ℝ := sorry

/-- Area of triangle PCD -/
def area_PCD (a b : ℝ) : ℝ := sorry

/-- Theorem stating the maximum area of triangle PCD -/
theorem max_area_PCD (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ (max_area : ℝ), max_area = (Real.sqrt 2 - 1) / 2 * a * b ∧
    ∀ (p : ℝ × ℝ), Γ a b p.1 p.2 → p.1 > 0 → p.2 < 0 →
      area_PCD a b ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_area_PCD_l521_52169


namespace NUMINAMATH_CALUDE_triangles_in_circle_l521_52170

/-- Given n points on a circle's circumference (n ≥ 6), with each pair connected by a chord
    and no three chords intersecting at a common point inside the circle,
    this function calculates the number of different triangles formed by the intersecting chords. -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6

/-- Theorem stating that the number of triangles formed by intersecting chords
    in a circle with n points (n ≥ 6) on its circumference is given by num_triangles n. -/
theorem triangles_in_circle (n : ℕ) (h : n ≥ 6) :
  (num_triangles n) =
    Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_circle_l521_52170


namespace NUMINAMATH_CALUDE_three_digit_sum_condition_l521_52196

/-- Represents a three-digit number abc in decimal form -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_range : a ≤ 9
  b_range : b ≤ 9
  c_range : c ≤ 9
  a_nonzero : a ≠ 0

/-- The value of a three-digit number abc in decimal form -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of all two-digit numbers formed by the digits a, b, c -/
def ThreeDigitNumber.sumTwoDigit (n : ThreeDigitNumber) : Nat :=
  2 * (10 * n.a + 10 * n.b + 10 * n.c)

/-- A three-digit number satisfies the condition if its value equals the sum of all two-digit numbers formed by its digits -/
def ThreeDigitNumber.satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  n.value = n.sumTwoDigit

/-- The theorem stating that only 132, 264, and 396 satisfy the condition -/
theorem three_digit_sum_condition :
  ∀ n : ThreeDigitNumber, n.satisfiesCondition ↔ (n.value = 132 ∨ n.value = 264 ∨ n.value = 396) := by
  sorry


end NUMINAMATH_CALUDE_three_digit_sum_condition_l521_52196


namespace NUMINAMATH_CALUDE_alicia_scored_14_points_per_half_l521_52171

/-- Alicia's points per half of the game -/
def alicia_points_per_half (total_points : ℕ) (num_players : ℕ) (other_players_average : ℕ) : ℕ :=
  (total_points - (num_players - 1) * other_players_average) / 2

/-- Proof that Alicia scored 14 points in each half of the game -/
theorem alicia_scored_14_points_per_half :
  alicia_points_per_half 63 8 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_alicia_scored_14_points_per_half_l521_52171


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l521_52156

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n |>.reverse

theorem binary_multiplication_theorem :
  let a := [true, true, false, true, false, true]  -- 110101₂
  let b := [true, true, true, false, true]  -- 11101₂
  let c := [true, false, true, false, true, true, true, false, true, false, true]  -- 10101110101₂
  binary_to_nat a * binary_to_nat b = binary_to_nat c := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l521_52156


namespace NUMINAMATH_CALUDE_parabola_vertex_in_fourth_quadrant_l521_52103

/-- Given a parabola y = 2x^2 + ax - 5 where a < 0, its vertex is in the fourth quadrant -/
theorem parabola_vertex_in_fourth_quadrant (a : ℝ) (ha : a < 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + a * x - 5
  let vertex_x : ℝ := -a / 4
  let vertex_y : ℝ := f vertex_x
  vertex_x > 0 ∧ vertex_y < 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_in_fourth_quadrant_l521_52103


namespace NUMINAMATH_CALUDE_perspective_right_angle_l521_52143

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the perspective transformation
def perspective_transform : Triangle → Triangle := sorry

-- Define the property of being horizontally placed
def is_horizontal (t : Triangle) : Prop := sorry

-- Define the property of a line being parallel to y' axis
def parallel_to_y_axis (p q : Point) : Prop := sorry

-- Define the property of a line being on x' axis
def on_x_axis (p q : Point) : Prop := sorry

-- Define the property of the angle formed by x'o'y' being 45°
def x_o_y_angle_45 (t : Triangle) : Prop := sorry

-- Define a right-angled triangle
def is_right_angled (t : Triangle) : Prop := sorry

-- The main theorem
theorem perspective_right_angle 
  (abc : Triangle) 
  (a'b'c' : Triangle) 
  (h1 : is_horizontal abc)
  (h2 : a'b'c' = perspective_transform abc)
  (h3 : parallel_to_y_axis a'b'c'.1 a'b'c'.2.1)
  (h4 : on_x_axis a'b'c'.2.1 a'b'c'.2.2)
  (h5 : x_o_y_angle_45 a'b'c') :
  is_right_angled abc :=
sorry

end NUMINAMATH_CALUDE_perspective_right_angle_l521_52143


namespace NUMINAMATH_CALUDE_correct_calculation_l521_52173

theorem correct_calculation (x : ℤ) (h : x - 749 = 280) : x + 479 = 1508 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l521_52173


namespace NUMINAMATH_CALUDE_thirteen_fourth_mod_eight_l521_52153

theorem thirteen_fourth_mod_eight (m : ℕ) : 
  13^4 % 8 = m ∧ 0 ≤ m ∧ m < 8 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_fourth_mod_eight_l521_52153


namespace NUMINAMATH_CALUDE_magnitude_of_z_is_one_l521_52166

-- Define the complex number z
variable (z : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem magnitude_of_z_is_one (h : (1 - z) / (1 + z) = 2 * i) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_is_one_l521_52166


namespace NUMINAMATH_CALUDE_negative_sqrt_16_l521_52186

theorem negative_sqrt_16 : -Real.sqrt 16 = -4 := by sorry

end NUMINAMATH_CALUDE_negative_sqrt_16_l521_52186


namespace NUMINAMATH_CALUDE_cut_cube_edges_l521_52185

/-- Represents a cube with cut corners -/
structure CutCube where
  originalEdges : Nat
  vertices : Nat
  cutsPerVertex : Nat
  newFacesPerCut : Nat
  newEdgesPerFace : Nat

/-- The number of edges in a cube with cut corners -/
def edgesAfterCut (c : CutCube) : Nat :=
  c.originalEdges + c.vertices * c.cutsPerVertex * c.newEdgesPerFace / 2

/-- Theorem stating that a cube with cut corners has 36 edges -/
theorem cut_cube_edges :
  ∀ c : CutCube,
  c.originalEdges = 12 ∧
  c.vertices = 8 ∧
  c.cutsPerVertex = 1 ∧
  c.newFacesPerCut = 1 ∧
  c.newEdgesPerFace = 4 →
  edgesAfterCut c = 36 := by
  sorry

#check cut_cube_edges

end NUMINAMATH_CALUDE_cut_cube_edges_l521_52185


namespace NUMINAMATH_CALUDE_combined_degrees_sum_l521_52178

/-- The combined number of degrees for Summer and Jolly -/
def combined_degrees (summer_degrees : ℕ) (difference : ℕ) : ℕ :=
  summer_degrees + (summer_degrees - difference)

/-- Theorem stating that given Summer has 150 degrees and 5 more degrees than Jolly,
    the combined number of degrees for Summer and Jolly is 295 -/
theorem combined_degrees_sum (summer_degrees : ℕ) (difference : ℕ)
  (h1 : summer_degrees = 150)
  (h2 : difference = 5) :
  combined_degrees summer_degrees difference = 295 := by
sorry

end NUMINAMATH_CALUDE_combined_degrees_sum_l521_52178
