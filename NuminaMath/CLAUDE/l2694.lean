import Mathlib

namespace NUMINAMATH_CALUDE_ramanujan_hardy_complex_numbers_l2694_269492

theorem ramanujan_hardy_complex_numbers :
  ∀ (z w : ℂ),
  z * w = 40 - 24 * I →
  w = 4 + 4 * I →
  z = 2 - 8 * I :=
by sorry

end NUMINAMATH_CALUDE_ramanujan_hardy_complex_numbers_l2694_269492


namespace NUMINAMATH_CALUDE_permutations_of_polarized_l2694_269470

theorem permutations_of_polarized (n : ℕ) (h : n = 9) :
  Nat.factorial n = 362880 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_polarized_l2694_269470


namespace NUMINAMATH_CALUDE_jamal_grade_jamal_grade_is_108_l2694_269450

theorem jamal_grade (total_students : ℕ) (absent_students : ℕ) (first_day_average : ℕ) 
  (new_average : ℕ) (taqeesha_score : ℕ) : ℕ :=
  let students_first_day := total_students - absent_students
  let total_score_first_day := students_first_day * first_day_average
  let total_score_all := total_students * new_average
  let combined_score_absent := total_score_all - total_score_first_day
  combined_score_absent - taqeesha_score

theorem jamal_grade_is_108 :
  jamal_grade 30 2 85 86 92 = 108 := by
  sorry

end NUMINAMATH_CALUDE_jamal_grade_jamal_grade_is_108_l2694_269450


namespace NUMINAMATH_CALUDE_orange_juice_production_l2694_269485

/-- The amount of oranges (in million tons) used for juice production -/
def juice_production (total : ℝ) (export_percent : ℝ) (juice_percent : ℝ) : ℝ :=
  total * (1 - export_percent) * juice_percent

/-- Theorem stating the amount of oranges used for juice production -/
theorem orange_juice_production :
  let total := 8
  let export_percent := 0.25
  let juice_percent := 0.60
  juice_production total export_percent juice_percent = 3.6 := by
sorry

#eval juice_production 8 0.25 0.60

end NUMINAMATH_CALUDE_orange_juice_production_l2694_269485


namespace NUMINAMATH_CALUDE_monotonicity_range_and_minimum_value_l2694_269474

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := exp x + a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * exp (a * x - 1) - 2 * a * x + f a x

theorem monotonicity_range_and_minimum_value 
  (h1 : ∀ x, x > 0)
  (h2 : ∀ a, a < 0) :
  (∃ S : Set ℝ, S = { a | ∀ x ∈ (Set.Ioo 0 (log 3)), 
    (Monotone (f a) ↔ Monotone (F a)) ∧ S = Set.Iic (-3)}) ∧
  (∀ a ∈ Set.Iic (-1 / (exp 2)), 
    IsMinOn (g a) (Set.Ioi 0) 0) := by sorry

end NUMINAMATH_CALUDE_monotonicity_range_and_minimum_value_l2694_269474


namespace NUMINAMATH_CALUDE_exterior_angle_square_octagon_is_135_l2694_269478

/-- The exterior angle formed by a square and a regular octagon sharing a common side -/
def exterior_angle_square_octagon : ℝ := 135

/-- Theorem: The exterior angle formed by a square and a regular octagon sharing a common side is 135 degrees -/
theorem exterior_angle_square_octagon_is_135 :
  exterior_angle_square_octagon = 135 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_square_octagon_is_135_l2694_269478


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2694_269454

/-- The minimum distance between a point on y = x^2 + 2 and a point on y = √(x - 2) -/
theorem min_distance_between_curves : ∃ (d : ℝ),
  d = (7 * Real.sqrt 2) / 4 ∧
  ∀ (xP yP xQ yQ : ℝ),
    yP = xP^2 + 2 →
    yQ = Real.sqrt (xQ - 2) →
    d ≤ Real.sqrt ((xP - xQ)^2 + (yP - yQ)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2694_269454


namespace NUMINAMATH_CALUDE_sum_of_ages_l2694_269455

theorem sum_of_ages (tom_age antonette_age : ℝ) : 
  tom_age = 40.5 → 
  antonette_age = 13.5 → 
  tom_age = 3 * antonette_age → 
  tom_age + antonette_age = 54 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2694_269455


namespace NUMINAMATH_CALUDE_total_marbles_l2694_269451

/-- Represents the number of marbles each boy has -/
structure MarbleDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The ratio of marbles between the three boys -/
def marbleRatio : MarbleDistribution := ⟨5, 2, 3⟩

/-- The number of additional marbles the first boy has -/
def additionalMarbles : ℕ := 3

/-- The number of marbles the middle (second) boy has -/
def middleBoyMarbles : ℕ := 12

/-- The theorem stating the total number of marbles -/
theorem total_marbles :
  ∃ (x : ℕ),
    x * marbleRatio.second = middleBoyMarbles ∧
    (x * marbleRatio.first + additionalMarbles) +
    (x * marbleRatio.second) +
    (x * marbleRatio.third) = 63 := by
  sorry


end NUMINAMATH_CALUDE_total_marbles_l2694_269451


namespace NUMINAMATH_CALUDE_mary_balloon_count_l2694_269464

/-- The number of yellow balloons each person has -/
structure BalloonCount where
  fred : ℕ
  sam : ℕ
  mary : ℕ

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- The actual balloon count for Fred, Sam, and Mary -/
def actual_count : BalloonCount where
  fred := 5
  sam := 6
  mary := 7

/-- Theorem stating that Mary has 7 yellow balloons -/
theorem mary_balloon_count :
  ∀ (count : BalloonCount),
    count.fred = actual_count.fred →
    count.sam = actual_count.sam →
    count.fred + count.sam + count.mary = total_balloons →
    count.mary = actual_count.mary :=
by
  sorry

end NUMINAMATH_CALUDE_mary_balloon_count_l2694_269464


namespace NUMINAMATH_CALUDE_cubes_arrangement_theorem_l2694_269462

/-- Represents the colors used to paint the cubes -/
inductive Color
  | White
  | Black
  | Red

/-- Represents a cube with 6 faces, each painted with a color -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- Represents the set of 16 cubes -/
def CubeSet := Fin 16 → Cube

/-- Represents an arrangement of the 16 cubes -/
structure Arrangement :=
  (placement : Fin 16 → Fin 3 × Fin 3 × Fin 3)
  (orientation : Fin 16 → Fin 6)

/-- Predicate to check if an arrangement shows only one color -/
def ShowsOnlyOneColor (cs : CubeSet) (arr : Arrangement) (c : Color) : Prop :=
  ∀ i : Fin 16, (cs i).faces (arr.orientation i) = c

/-- Theorem stating that it's possible to arrange the cubes to show only one color -/
theorem cubes_arrangement_theorem (cs : CubeSet) :
  ∃ (arr : Arrangement) (c : Color), ShowsOnlyOneColor cs arr c :=
sorry

end NUMINAMATH_CALUDE_cubes_arrangement_theorem_l2694_269462


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2694_269460

theorem complex_expression_evaluation :
  (7 - 3*Complex.I) - 3*(2 + 4*Complex.I) + (1 + 2*Complex.I) = 2 - 13*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2694_269460


namespace NUMINAMATH_CALUDE_probability_at_least_one_head_l2694_269402

theorem probability_at_least_one_head (p : ℝ) : 
  p = 1 - (1/2)^4 → p = 15/16 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_head_l2694_269402


namespace NUMINAMATH_CALUDE_equal_area_triangles_l2694_269449

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equal_area_triangles :
  triangleArea 13 13 10 = triangleArea 13 13 24 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_triangles_l2694_269449


namespace NUMINAMATH_CALUDE_expression_evaluation_l2694_269427

theorem expression_evaluation : (2^(1^(0^2)))^3 + (3^(1^2))^0 + 4^(0^1) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2694_269427


namespace NUMINAMATH_CALUDE_problem_solution_l2694_269473

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem problem_solution :
  (∀ x : ℝ, 0 < x → x < Real.exp 1 → (deriv f) x > 0) ∧
  (∀ x : ℝ, x > Real.exp 1 → (deriv f) x < 0) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → Real.log x + 1 / x > a) ↔ a < 1) ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = (1/6) * x₀ - (5/6) / x₀ + 2/3 ∧
    (deriv f) x₀ = (1/3) * x₀ + 2/3) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2694_269473


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l2694_269499

theorem sqrt_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l2694_269499


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2694_269414

theorem sum_of_coefficients (a b c d e : ℤ) : 
  (∀ x : ℚ, 1000 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 92 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2694_269414


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l2694_269453

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l2694_269453


namespace NUMINAMATH_CALUDE_clearance_sale_gain_percentage_shopkeeper_clearance_sale_gain_l2694_269480

/-- Calculates the gain percentage during a clearance sale -/
theorem clearance_sale_gain_percentage 
  (original_price : ℝ) 
  (original_gain_percent : ℝ) 
  (discount_percent : ℝ) : ℝ :=
  let cost_price := original_price / (1 + original_gain_percent / 100)
  let discounted_price := original_price * (1 - discount_percent / 100)
  let new_gain := discounted_price - cost_price
  let new_gain_percent := (new_gain / cost_price) * 100
  new_gain_percent

/-- The gain percentage during the clearance sale is approximately 21.5% -/
theorem shopkeeper_clearance_sale_gain :
  abs (clearance_sale_gain_percentage 30 35 10 - 21.5) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_clearance_sale_gain_percentage_shopkeeper_clearance_sale_gain_l2694_269480


namespace NUMINAMATH_CALUDE_tan_value_for_special_condition_l2694_269432

theorem tan_value_for_special_condition (a : Real) 
  (h1 : 0 < a ∧ a < π / 2) 
  (h2 : Real.sin a ^ 2 + Real.cos (2 * a) = 1) : 
  Real.tan a = 0 := by
sorry

end NUMINAMATH_CALUDE_tan_value_for_special_condition_l2694_269432


namespace NUMINAMATH_CALUDE_smallest_s_plus_d_l2694_269476

theorem smallest_s_plus_d : ∀ s d : ℕ+,
  (1 : ℚ) / s + (1 : ℚ) / (2 * s) + (1 : ℚ) / (3 * s) = (1 : ℚ) / (d^2 - 2*d) →
  ∀ s' d' : ℕ+,
  (1 : ℚ) / s' + (1 : ℚ) / (2 * s') + (1 : ℚ) / (3 * s') = (1 : ℚ) / (d'^2 - 2*d') →
  (s + d : ℕ) ≤ (s' + d' : ℕ) →
  (s + d : ℕ) = 50 :=
sorry

end NUMINAMATH_CALUDE_smallest_s_plus_d_l2694_269476


namespace NUMINAMATH_CALUDE_enemy_plane_hit_probability_l2694_269472

/-- The probability that the enemy plane is hit given A's and B's hit probabilities -/
theorem enemy_plane_hit_probability (p_A p_B : ℝ) (h_A : p_A = 0.6) (h_B : p_B = 0.4) :
  1 - (1 - p_A) * (1 - p_B) = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_enemy_plane_hit_probability_l2694_269472


namespace NUMINAMATH_CALUDE_employee_pay_l2694_269435

/-- Given two employees X and Y, proves that Y's weekly pay is 150 units -/
theorem employee_pay (total_pay x y : ℝ) : 
  total_pay = x + y → 
  x = 1.2 * y → 
  total_pay = 330 → 
  y = 150 := by sorry

end NUMINAMATH_CALUDE_employee_pay_l2694_269435


namespace NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l2694_269406

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l2694_269406


namespace NUMINAMATH_CALUDE_ellipse_equation_l2694_269495

theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0)
  (h4 : c / a = Real.sqrt 3 / 2)
  (h5 : a - c = 2 - Real.sqrt 3)
  (h6 : b^2 = a^2 - c^2) :
  ∃ (x y : ℝ), y^2 / 4 + x^2 = 1 ∧ y^2 / a^2 + x^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2694_269495


namespace NUMINAMATH_CALUDE_only_drunk_drivers_traffic_accidents_correlated_l2694_269404

-- Define the types of quantities
inductive Quantity
  | Time
  | Displacement
  | StudentGrade
  | Weight
  | DrunkDrivers
  | TrafficAccidents
  | Volume

-- Define the relationship between quantities
inductive Relationship
  | NoRelation
  | Correlation
  | FunctionalRelation

-- Define a function to determine the relationship between two quantities
def relationshipBetween (q1 q2 : Quantity) : Relationship :=
  match q1, q2 with
  | Quantity.Time, Quantity.Displacement => Relationship.FunctionalRelation
  | Quantity.StudentGrade, Quantity.Weight => Relationship.NoRelation
  | Quantity.DrunkDrivers, Quantity.TrafficAccidents => Relationship.Correlation
  | Quantity.Volume, Quantity.Weight => Relationship.FunctionalRelation
  | _, _ => Relationship.NoRelation

-- Theorem to prove
theorem only_drunk_drivers_traffic_accidents_correlated :
  ∀ q1 q2 : Quantity,
    relationshipBetween q1 q2 = Relationship.Correlation →
    (q1 = Quantity.DrunkDrivers ∧ q2 = Quantity.TrafficAccidents) ∨
    (q1 = Quantity.TrafficAccidents ∧ q2 = Quantity.DrunkDrivers) :=
by
  sorry


end NUMINAMATH_CALUDE_only_drunk_drivers_traffic_accidents_correlated_l2694_269404


namespace NUMINAMATH_CALUDE_ellipse_dimensions_l2694_269498

/-- An ellipse with foci F₁ and F₂, and a point P on the ellipse. -/
structure Ellipse (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  on_ellipse : (P.1 / a) ^ 2 + (P.2 / b) ^ 2 = 1
  perpendicular : (P.1 - F₁.1) * (P.2 - F₂.2) + (P.2 - F₁.2) * (P.1 - F₂.1) = 0
  triangle_area : abs ((F₁.1 - P.1) * (F₂.2 - P.2) - (F₁.2 - P.2) * (F₂.1 - P.1)) / 2 = 9
  triangle_perimeter : dist P F₁ + dist P F₂ + dist F₁ F₂ = 18

/-- The theorem stating that under the given conditions, a = 5 and b = 3 -/
theorem ellipse_dimensions (E : Ellipse a b) : a = 5 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_dimensions_l2694_269498


namespace NUMINAMATH_CALUDE_distribute_5_to_3_l2694_269439

/-- The number of ways to distribute n volunteers to k venues --/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 volunteers to 3 venues --/
theorem distribute_5_to_3 : distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_5_to_3_l2694_269439


namespace NUMINAMATH_CALUDE_orthocenter_quadrilateral_congruence_l2694_269431

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
def CyclicQuadrilateral (A B C D : Point) : Prop := sorry

/-- The orthocenter of a triangle is the point where the three altitudes of the triangle intersect. -/
def Orthocenter (H A B C : Point) : Prop := sorry

/-- Two quadrilaterals are congruent if they have the same shape and size. -/
def CongruentQuadrilaterals (A B C D A' B' C' D' : Point) : Prop := sorry

theorem orthocenter_quadrilateral_congruence 
  (A B C D A' B' C' D' : Point) :
  CyclicQuadrilateral A B C D →
  Orthocenter A' B C D →
  Orthocenter B' A C D →
  Orthocenter C' A B D →
  Orthocenter D' A B C →
  CongruentQuadrilaterals A B C D A' B' C' D' :=
sorry

end NUMINAMATH_CALUDE_orthocenter_quadrilateral_congruence_l2694_269431


namespace NUMINAMATH_CALUDE_nelly_painting_bid_l2694_269457

/-- The amount Nelly paid for the painting -/
def nellys_bid (joes_bid : ℕ) : ℕ :=
  3 * joes_bid + 2000

/-- Theorem stating Nelly's final bid given Joe's bid -/
theorem nelly_painting_bid :
  let joes_bid : ℕ := 160000
  nellys_bid joes_bid = 482000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_painting_bid_l2694_269457


namespace NUMINAMATH_CALUDE_horner_method_eval_l2694_269489

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + 5 * x^3 - 2.5 * x^2 + 1.5 * x - 0.7

theorem horner_method_eval :
  f 4 = horner_eval [3, -2, 5, -2.5, 1.5, -0.7] 4 ∧
  horner_eval [3, -2, 5, -2.5, 1.5, -0.7] 4 = 2845.3 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_eval_l2694_269489


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l2694_269437

/-- The count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with two adjacent identical digits -/
def numbers_with_two_adjacent_identical_digits : ℕ := 153

/-- The count of valid three-digit numbers according to the problem conditions -/
def valid_three_digit_numbers : ℕ := total_three_digit_numbers - numbers_with_two_adjacent_identical_digits

theorem valid_three_digit_numbers_count :
  valid_three_digit_numbers = 747 := by
  sorry

end NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l2694_269437


namespace NUMINAMATH_CALUDE_box_volume_increase_l2694_269403

/-- Theorem about the volume of a rectangular box after increasing its dimensions --/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + l * h + w * h) = 1950)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7198 := by sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2694_269403


namespace NUMINAMATH_CALUDE_clown_balloons_l2694_269410

/-- The number of balloons in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of balloons the clown initially has -/
def initial_dozens : ℕ := 3

/-- The number of boys who buy a balloon -/
def boys : ℕ := 3

/-- The number of girls who buy a balloon -/
def girls : ℕ := 12

/-- The number of balloons the clown is left with after selling to boys and girls -/
def remaining_balloons : ℕ := initial_dozens * dozen - (boys + girls)

theorem clown_balloons : remaining_balloons = 21 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l2694_269410


namespace NUMINAMATH_CALUDE_segment_transformation_midpoint_l2694_269442

/-- Given a segment with endpoints (3, -2) and (9, 6), when translated 4 units left and 2 units down,
    then rotated 90° counterclockwise about its midpoint, the resulting segment has a midpoint at (2, 0) -/
theorem segment_transformation_midpoint : 
  let s₁_start : ℝ × ℝ := (3, -2)
  let s₁_end : ℝ × ℝ := (9, 6)
  let translate : ℝ × ℝ := (-4, -2)
  let s₁_midpoint := ((s₁_start.1 + s₁_end.1) / 2, (s₁_start.2 + s₁_end.2) / 2)
  let s₂_midpoint := (s₁_midpoint.1 + translate.1, s₁_midpoint.2 + translate.2)
  s₂_midpoint = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_segment_transformation_midpoint_l2694_269442


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2019_l2694_269436

-- Define the arithmetic sequence and its sum
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)
def S (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- State the theorem
theorem arithmetic_sequence_sum_2019 (a₁ d : ℤ) :
  a₁ = -2017 →
  (S a₁ d 2017 / 2017 - S a₁ d 2015 / 2015 = 2) →
  S a₁ d 2019 = 2019 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2019_l2694_269436


namespace NUMINAMATH_CALUDE_melanie_dimes_problem_l2694_269420

/-- Calculates the number of dimes Melanie's mother gave her -/
def dimes_from_mother (initial : ℕ) (given_to_dad : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_to_dad)

theorem melanie_dimes_problem :
  dimes_from_mother 8 7 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_problem_l2694_269420


namespace NUMINAMATH_CALUDE_construct_m_is_perfect_square_l2694_269491

/-- The number of 1's in the sequence -/
def num_ones : ℕ := 1997

/-- The number of 2's in the sequence -/
def num_twos : ℕ := 1998

/-- Constructs the number m as described in the problem -/
def construct_m : ℕ :=
  let n := (10^num_ones * (10^num_ones - 1) + 2 * (10^num_ones - 1)) / 9
  10 * n + 25

/-- Theorem stating that the constructed number m is a perfect square -/
theorem construct_m_is_perfect_square : ∃ k : ℕ, construct_m = k^2 := by
  sorry

end NUMINAMATH_CALUDE_construct_m_is_perfect_square_l2694_269491


namespace NUMINAMATH_CALUDE_dots_not_visible_is_81_l2694_269421

/-- The number of faces on each die -/
def faces_per_die : ℕ := 6

/-- The number of dice -/
def num_dice : ℕ := 5

/-- The list of visible numbers on the dice -/
def visible_numbers : List ℕ := [1, 2, 3, 1, 4, 5, 6, 2]

/-- The total number of dots on all dice -/
def total_dots : ℕ := num_dice * (faces_per_die * (faces_per_die + 1) / 2)

/-- The sum of visible numbers -/
def sum_visible : ℕ := visible_numbers.sum

/-- Theorem: The number of dots not visible is 81 -/
theorem dots_not_visible_is_81 : total_dots - sum_visible = 81 := by
  sorry

end NUMINAMATH_CALUDE_dots_not_visible_is_81_l2694_269421


namespace NUMINAMATH_CALUDE_second_car_speed_l2694_269416

/-- Given two cars on a circular track, prove the speed of the second car. -/
theorem second_car_speed
  (track_length : ℝ)
  (first_car_speed : ℝ)
  (total_time : ℝ)
  (h1 : track_length = 150)
  (h2 : first_car_speed = 60)
  (h3 : total_time = 2)
  (h4 : ∃ (second_car_speed : ℝ),
    (first_car_speed + second_car_speed) * total_time = 2 * track_length) :
  ∃ (second_car_speed : ℝ), second_car_speed = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_second_car_speed_l2694_269416


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_l2694_269467

/-- 
Given a parabola with equation y^2 = 2px and its latus rectum with equation x = -2,
prove that p = 4.
-/
theorem parabola_latus_rectum (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Equation of the parabola
  (∀ y : ℝ, y^2 = 2*p*(-2)) → -- Equation of the latus rectum
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_l2694_269467


namespace NUMINAMATH_CALUDE_dantes_age_l2694_269430

theorem dantes_age (cooper dante maria : ℕ) : 
  cooper + dante + maria = 31 →
  cooper = dante / 2 →
  maria = dante + 1 →
  dante = 12 := by
sorry

end NUMINAMATH_CALUDE_dantes_age_l2694_269430


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l2694_269477

theorem fraction_sum_equation (x y : ℝ) (h1 : x ≠ y) 
  (h2 : x / y + (x + 6 * y) / (y + 6 * x) = 3) : 
  x / y = (8 + Real.sqrt 46) / 6 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l2694_269477


namespace NUMINAMATH_CALUDE_c_payment_l2694_269490

def work_rate (days : ℕ) : ℚ := 1 / days

def total_payment : ℕ := 3200

def completion_days : ℕ := 3

theorem c_payment (a_days b_days : ℕ) (ha : a_days = 6) (hb : b_days = 8) : 
  let a_rate := work_rate a_days
  let b_rate := work_rate b_days
  let ab_rate := a_rate + b_rate
  let ab_work := ab_rate * completion_days
  let c_work := 1 - ab_work
  c_work * total_payment = 400 := by sorry

end NUMINAMATH_CALUDE_c_payment_l2694_269490


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2694_269445

theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let circumference := 2 * Real.pi * r
  let height := circumference
  let lateral_area := circumference * height
  lateral_area = 4 * Real.pi * S := by
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2694_269445


namespace NUMINAMATH_CALUDE_movie_ratio_proof_l2694_269458

theorem movie_ratio_proof (total : ℕ) (dvd : ℕ) (bluray : ℕ) :
  total = 378 →
  dvd + bluray = total →
  dvd / (bluray - 4) = 9 / 2 →
  (dvd : ℚ) / bluray = 51 / 12 :=
by
  sorry

end NUMINAMATH_CALUDE_movie_ratio_proof_l2694_269458


namespace NUMINAMATH_CALUDE_percentage_to_pass_l2694_269466

/-- Given a test with maximum marks, a student's score, and the margin by which they failed,
    calculate the percentage of marks needed to pass the test. -/
theorem percentage_to_pass (max_marks student_score fail_margin : ℕ) : 
  max_marks = 200 → 
  student_score = 80 → 
  fail_margin = 40 → 
  (student_score + fail_margin) / max_marks * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l2694_269466


namespace NUMINAMATH_CALUDE_power_sum_l2694_269469

theorem power_sum (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l2694_269469


namespace NUMINAMATH_CALUDE_m_range_theorem_l2694_269428

/-- Proposition P: The equation x²/(2m) + y²/(9-m) = 1 represents an ellipse with foci on the y-axis -/
def P (m : ℝ) : Prop :=
  0 < m ∧ m < 3

/-- Proposition Q: The eccentricity e of the hyperbola y²/5 - x²/m = 1 is within the range (√6/2, √2) -/
def Q (m : ℝ) : Prop :=
  m > 0 ∧ 5/2 < m ∧ m < 5

/-- The set of valid m values -/
def M : Set ℝ :=
  {m | (0 < m ∧ m ≤ 5/2) ∨ (3 ≤ m ∧ m < 5)}

theorem m_range_theorem :
  ∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → m ∈ M :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2694_269428


namespace NUMINAMATH_CALUDE_arithmetic_mean_lower_bound_l2694_269424

theorem arithmetic_mean_lower_bound (a₁ a₂ a₃ : ℝ) 
  (h_positive : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0) 
  (h_sum : 2*a₁ + 3*a₂ + a₃ = 1) : 
  (1/(a₁ + a₂) + 1/(a₂ + a₃)) / 2 ≥ (3 + 2*Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_lower_bound_l2694_269424


namespace NUMINAMATH_CALUDE_snatch_percentage_increase_l2694_269471

/-- Calculates the percentage increase in Snatch weight given initial weights and new total -/
theorem snatch_percentage_increase
  (initial_clean_jerk : ℝ)
  (initial_snatch : ℝ)
  (new_total : ℝ)
  (h1 : initial_clean_jerk = 80)
  (h2 : initial_snatch = 50)
  (h3 : new_total = 250)
  (h4 : 2 * initial_clean_jerk + new_snatch = new_total)
  : (new_snatch - initial_snatch) / initial_snatch * 100 = 80 :=
by
  sorry

#check snatch_percentage_increase

end NUMINAMATH_CALUDE_snatch_percentage_increase_l2694_269471


namespace NUMINAMATH_CALUDE_multiply_fractions_l2694_269441

theorem multiply_fractions : (2 * (1/3)) * (3 * (1/2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_l2694_269441


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l2694_269481

theorem point_on_unit_circle (x : ℝ) (θ : ℝ) :
  (∃ (M : ℝ × ℝ), M = (x, 1) ∧ M.1 = x * Real.cos θ ∧ M.2 = x * Real.sin θ) →
  Real.cos θ = (Real.sqrt 2 / 2) * x →
  x = -1 ∨ x = 0 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l2694_269481


namespace NUMINAMATH_CALUDE_average_temperature_l2694_269417

def temperature_data : List ℝ := [90, 90, 90, 79, 71]
def num_years : ℕ := 5

theorem average_temperature : 
  (List.sum temperature_data) / num_years = 84 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l2694_269417


namespace NUMINAMATH_CALUDE_last_positive_term_is_six_l2694_269461

/-- Represents an arithmetic sequence with a given start and common difference. -/
structure ArithmeticSequence where
  start : ℤ
  diff : ℤ

/-- Calculates the nth term of an arithmetic sequence. -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.start + (n - 1 : ℤ) * seq.diff

/-- Theorem: The last term greater than 0 in the sequence (72, 61, 50, ...) is 6. -/
theorem last_positive_term_is_six :
  let seq := ArithmeticSequence.mk 72 (-11)
  ∃ n : ℕ, 
    (nthTerm seq n = 6) ∧ 
    (nthTerm seq n > 0) ∧ 
    (nthTerm seq (n + 1) ≤ 0) :=
by sorry

#check last_positive_term_is_six

end NUMINAMATH_CALUDE_last_positive_term_is_six_l2694_269461


namespace NUMINAMATH_CALUDE_marbles_remainder_l2694_269433

theorem marbles_remainder (n m k : ℤ) : (8*n + 5 + 7*m + 2 + 7*k + 4) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remainder_l2694_269433


namespace NUMINAMATH_CALUDE_clock_synchronization_l2694_269456

theorem clock_synchronization (arthur_gain oleg_gain cycle : ℕ) 
  (h1 : arthur_gain = 15)
  (h2 : oleg_gain = 12)
  (h3 : cycle = 720) :
  let sync_days := Nat.lcm (cycle / arthur_gain) (cycle / oleg_gain)
  sync_days = 240 ∧ 
  ∀ k : ℕ, k < sync_days → ¬(arthur_gain * k % cycle = 0 ∧ oleg_gain * k % cycle = 0) := by
  sorry

end NUMINAMATH_CALUDE_clock_synchronization_l2694_269456


namespace NUMINAMATH_CALUDE_san_antonio_bound_bus_encounters_l2694_269486

-- Define the time type (in minutes since midnight)
def Time := ℕ

-- Define the bus schedules
def austin_to_san_antonio_schedule (t : Time) : Prop :=
  ∃ n : ℕ, t = 360 + 120 * n

def san_antonio_to_austin_schedule (t : Time) : Prop :=
  ∃ n : ℕ, t = 390 + 60 * n

-- Define the travel time
def travel_time : ℕ := 360  -- 6 hours in minutes

-- Define the function to count encounters
def count_encounters (start_time : Time) : ℕ :=
  sorry  -- Implementation details omitted

-- Theorem statement
theorem san_antonio_bound_bus_encounters :
  ∀ (start_time : Time),
    san_antonio_to_austin_schedule start_time →
    count_encounters start_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_san_antonio_bound_bus_encounters_l2694_269486


namespace NUMINAMATH_CALUDE_gcd_228_1995_decimal_to_ternary_l2694_269425

-- Problem 1: GCD of 228 and 1995
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by sorry

-- Problem 2: Convert 104 to base 3
theorem decimal_to_ternary :
  ∃ (a b c d e : Nat),
    104 = a * 3^4 + b * 3^3 + c * 3^2 + d * 3^1 + e * 3^0 ∧
    a = 1 ∧ b = 0 ∧ c = 2 ∧ d = 1 ∧ e = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_228_1995_decimal_to_ternary_l2694_269425


namespace NUMINAMATH_CALUDE_caterpillar_problem_solution_l2694_269423

/-- Represents the caterpillar problem --/
structure CaterpillarProblem where
  initial_caterpillars : ℕ
  fallen_caterpillars : ℕ
  hatched_eggs : ℕ
  leaves_per_day : ℕ
  observation_days : ℕ
  cocooned_caterpillars : ℕ

/-- Calculates the number of caterpillars left on the tree and leaves eaten --/
def solve_caterpillar_problem (problem : CaterpillarProblem) : ℕ × ℕ :=
  let remaining_after_storm := problem.initial_caterpillars - problem.fallen_caterpillars
  let total_after_hatching := remaining_after_storm + problem.hatched_eggs
  let remaining_after_cocooning := total_after_hatching - problem.cocooned_caterpillars
  let final_caterpillars := remaining_after_cocooning / 2
  let leaves_eaten := problem.hatched_eggs * problem.leaves_per_day * problem.observation_days
  (final_caterpillars, leaves_eaten)

/-- Theorem stating the solution to the caterpillar problem --/
theorem caterpillar_problem_solution :
  let problem : CaterpillarProblem := {
    initial_caterpillars := 14,
    fallen_caterpillars := 3,
    hatched_eggs := 6,
    leaves_per_day := 2,
    observation_days := 7,
    cocooned_caterpillars := 9
  }
  solve_caterpillar_problem problem = (4, 84) := by
  sorry


end NUMINAMATH_CALUDE_caterpillar_problem_solution_l2694_269423


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2694_269488

theorem polynomial_remainder (s : ℝ) : (s^10 + 1) % (s - 2) = 1025 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2694_269488


namespace NUMINAMATH_CALUDE_g_zero_l2694_269493

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_def : h = f * g

-- Define the constant terms of f and h
axiom f_const : f.coeff 0 = -6
axiom h_const : h.coeff 0 = 12

-- Theorem to prove
theorem g_zero : g.eval 0 = -2 := by sorry

end NUMINAMATH_CALUDE_g_zero_l2694_269493


namespace NUMINAMATH_CALUDE_pencil_count_l2694_269475

/-- The total number of pencils after multiplication and addition -/
def total_pencils (initial : ℕ) (factor : ℕ) (additional : ℕ) : ℕ :=
  initial * factor + additional

/-- Theorem stating that the total number of pencils is 153 -/
theorem pencil_count : total_pencils 27 4 45 = 153 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2694_269475


namespace NUMINAMATH_CALUDE_usual_time_calculation_l2694_269422

theorem usual_time_calculation (T : ℝ) 
  (h1 : T > 0) 
  (h2 : (1 : ℝ) / 0.25 = (T + 24) / T) : T = 8 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_calculation_l2694_269422


namespace NUMINAMATH_CALUDE_at_least_one_negative_l2694_269444

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l2694_269444


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l2694_269413

/-- A trinomial ax^2 + bx + c is a perfect square if and only if b^2 - 4ac = 0 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  b^2 = 4*a*c

theorem perfect_square_trinomial_m_values :
  ∀ m : ℝ, (is_perfect_square_trinomial 1 (-2*(m+3)) 9) → (m = 0 ∨ m = -6) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l2694_269413


namespace NUMINAMATH_CALUDE_alicia_local_tax_l2694_269440

/-- Calculates the local tax amount in cents given an hourly wage in dollars and a tax rate percentage. -/
def local_tax_cents (hourly_wage : ℚ) (tax_rate : ℚ) : ℚ :=
  hourly_wage * 100 * (tax_rate / 100)

/-- Theorem stating that for an hourly wage of $25 and a 2% tax rate, the local tax amount is 50 cents. -/
theorem alicia_local_tax : local_tax_cents 25 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_alicia_local_tax_l2694_269440


namespace NUMINAMATH_CALUDE_postcard_price_calculation_bernie_postcard_problem_l2694_269415

theorem postcard_price_calculation (initial_postcards : Nat) 
  (sold_postcards : Nat) (price_per_sold : Nat) (final_total : Nat) : Nat :=
  let total_earned := sold_postcards * price_per_sold
  let remaining_original := initial_postcards - sold_postcards
  let new_postcards := final_total - remaining_original
  total_earned / new_postcards

theorem bernie_postcard_problem : 
  postcard_price_calculation 18 9 15 36 = 5 := by
  sorry

end NUMINAMATH_CALUDE_postcard_price_calculation_bernie_postcard_problem_l2694_269415


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2694_269465

theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h_arithmetic_a : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_arithmetic_b : ∀ n, b (n + 1) - b n = b 2 - b 1)
  (h_sum_S : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_sum_T : ∀ n, T n = n * (b 1 + b n) / 2)
  (h_ratio : ∀ n, S n / T n = (n + 3) / (2 * n + 1)) :
  a 6 / b 6 = 14 / 23 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2694_269465


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2694_269497

/-- A geometric sequence with first four terms 25, -50, 100, -200 has a common ratio of -2 -/
theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℚ), 
    a 0 = 25 ∧ a 1 = -50 ∧ a 2 = 100 ∧ a 3 = -200 →
    ∃ (r : ℚ), r = -2 ∧ ∀ (n : ℕ), a (n + 1) = r * a n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2694_269497


namespace NUMINAMATH_CALUDE_ball_drawing_probability_l2694_269482

theorem ball_drawing_probability : 
  let total_balls : ℕ := 25
  let black_balls : ℕ := 10
  let white_balls : ℕ := 10
  let red_balls : ℕ := 5
  let drawn_balls : ℕ := 4

  let probability : ℚ := 
    (Nat.choose black_balls 2 * Nat.choose white_balls 2 + 
     Nat.choose black_balls 2 * Nat.choose red_balls 2 + 
     Nat.choose white_balls 2 * Nat.choose red_balls 2) / 
    Nat.choose total_balls drawn_balls

  probability = 195 / 841 := by
sorry

end NUMINAMATH_CALUDE_ball_drawing_probability_l2694_269482


namespace NUMINAMATH_CALUDE_students_with_one_problem_l2694_269405

/-- Represents the number of problems created by students from each course -/
def ProblemsCourses : Type := Fin 5 → ℕ

/-- Represents the number of students in each course -/
def StudentsCourses : Type := Fin 5 → ℕ

/-- The total number of students -/
def TotalStudents : ℕ := 30

/-- The total number of problems created -/
def TotalProblems : ℕ := 40

/-- The condition that students from different courses created different numbers of problems -/
def DifferentProblems (p : ProblemsCourses) : Prop :=
  ∀ i j, i ≠ j → p i ≠ p j

/-- The condition that the total number of problems created matches the given total -/
def MatchesTotalProblems (p : ProblemsCourses) (s : StudentsCourses) : Prop :=
  (Finset.sum Finset.univ (λ i => p i * s i)) = TotalProblems

/-- The condition that the total number of students matches the given total -/
def MatchesTotalStudents (s : StudentsCourses) : Prop :=
  (Finset.sum Finset.univ s) = TotalStudents

theorem students_with_one_problem
  (p : ProblemsCourses)
  (s : StudentsCourses)
  (h1 : DifferentProblems p)
  (h2 : MatchesTotalProblems p s)
  (h3 : MatchesTotalStudents s) :
  (Finset.filter (λ i => p i = 1) Finset.univ).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_students_with_one_problem_l2694_269405


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2694_269484

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = q * a n) 
  (S : ℕ → ℝ) 
  (h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) 
  (h3 : 2 * S 4 = S 5 + S 6) :
  q = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2694_269484


namespace NUMINAMATH_CALUDE_peters_glass_purchase_l2694_269408

/-- Peter's glass purchase problem -/
theorem peters_glass_purchase
  (small_price : ℕ)
  (large_price : ℕ)
  (total_money : ℕ)
  (change : ℕ)
  (large_count : ℕ)
  (h1 : small_price = 3)
  (h2 : large_price = 5)
  (h3 : total_money = 50)
  (h4 : change = 1)
  (h5 : large_count = 5)
  : (total_money - change - large_count * large_price) / small_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_peters_glass_purchase_l2694_269408


namespace NUMINAMATH_CALUDE_smallest_a_value_l2694_269438

theorem smallest_a_value (a b : ℕ) (h : b^3 = 1176*a) : 
  (∀ x : ℕ, x < a → ¬∃ y : ℕ, y^3 = 1176*x) → a = 63 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2694_269438


namespace NUMINAMATH_CALUDE_sum_of_m_values_l2694_269400

/-- A triangle with vertices at (0,0), (2,2), and (8m,0) is divided into two equal areas by a line y = mx. -/
def Triangle (m : ℝ) := {A : ℝ × ℝ | A = (0, 0) ∨ A = (2, 2) ∨ A = (8*m, 0)}

/-- The line that divides the triangle into two equal areas -/
def DividingLine (m : ℝ) := {(x, y) : ℝ × ℝ | y = m * x}

/-- The condition that the line divides the triangle into two equal areas -/
def EqualAreasCondition (m : ℝ) : Prop := 
  ∃ (x : ℝ), (x, m*x) ∈ DividingLine m ∧ 
  (x = 4*m + 1) ∧ (m*x = 1)

/-- The theorem stating that the sum of all possible values of m is -1/4 -/
theorem sum_of_m_values (m₁ m₂ : ℝ) : 
  (EqualAreasCondition m₁ ∧ EqualAreasCondition m₂ ∧ m₁ ≠ m₂) → 
  m₁ + m₂ = -1/4 := by sorry

end NUMINAMATH_CALUDE_sum_of_m_values_l2694_269400


namespace NUMINAMATH_CALUDE_point_inside_circle_l2694_269483

/-- Definition of a circle with center (a, b) and radius r -/
def Circle (a b r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = r^2}

/-- Definition of a point being inside a circle -/
def InsideCircle (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  (p.1 - 2)^2 + (p.2 - 3)^2 < 4

/-- The main theorem -/
theorem point_inside_circle :
  let c : Set (ℝ × ℝ) := Circle 2 3 2
  let p : ℝ × ℝ := (1, 2)
  InsideCircle p c := by sorry

end NUMINAMATH_CALUDE_point_inside_circle_l2694_269483


namespace NUMINAMATH_CALUDE_only_coordinates_specific_l2694_269434

/-- Represents a location description --/
inductive LocationDescription
  | CinemaRow (row : Nat)
  | StreetAddress (street : String) (city : String)
  | Direction (angle : Float) (direction : String)
  | Coordinates (longitude : Float) (latitude : Float)

/-- Determines if a location description provides a specific, unique location --/
def isSpecificLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.Coordinates _ _ => True
  | _ => False

/-- Theorem stating that only the coordinates option provides a specific location --/
theorem only_coordinates_specific (desc : LocationDescription) :
  isSpecificLocation desc ↔ ∃ (long lat : Float), desc = LocationDescription.Coordinates long lat :=
sorry

#check only_coordinates_specific

end NUMINAMATH_CALUDE_only_coordinates_specific_l2694_269434


namespace NUMINAMATH_CALUDE_train_length_l2694_269426

/-- Calculates the length of a train given the bridge length, train speed, and time to pass the bridge. -/
theorem train_length (bridge_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  bridge_length = 160 ∧ 
  train_speed_kmh = 40 ∧ 
  time_to_pass = 25.2 →
  (train_speed_kmh * 1000 / 3600) * time_to_pass - bridge_length = 120 :=
by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2694_269426


namespace NUMINAMATH_CALUDE_derivative_at_one_l2694_269479

/-- Given a function f(x) = x³ - 2f'(1)x, prove that f'(1) = 1 -/
theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 2 * (deriv f 1) * x) : 
  deriv f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2694_269479


namespace NUMINAMATH_CALUDE_laborer_income_l2694_269459

/-- The monthly income of a laborer given certain expenditure and savings conditions -/
theorem laborer_income (
  average_expenditure : ℝ)
  (reduced_expenditure : ℝ)
  (months_initial : ℕ)
  (months_reduced : ℕ)
  (savings : ℝ)
  (h1 : average_expenditure = 90)
  (h2 : reduced_expenditure = 60)
  (h3 : months_initial = 6)
  (h4 : months_reduced = 4)
  (h5 : savings = 30)
  : ∃ (income : ℝ) (debt : ℝ),
    income * months_initial = average_expenditure * months_initial - debt ∧
    income * months_reduced = reduced_expenditure * months_reduced + debt + savings ∧
    income = 81 := by
  sorry

end NUMINAMATH_CALUDE_laborer_income_l2694_269459


namespace NUMINAMATH_CALUDE_prism_volume_l2694_269409

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h1 : side_area = 18)
  (h2 : front_area = 12)
  (h3 : bottom_area = 8) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2694_269409


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2694_269429

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2694_269429


namespace NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l2694_269412

/-- Represents the width of each rectangle --/
def rectangle_width : ℝ := 4

/-- Represents the length of each rectangle --/
def rectangle_length : ℝ := 4 * rectangle_width

/-- Represents the side length of the original square --/
def square_side : ℝ := rectangle_width + rectangle_length

/-- Represents the perimeter of the "P" shape --/
def p_perimeter : ℝ := 56

theorem square_perimeter_from_p_shape :
  p_perimeter = 2 * (square_side) + rectangle_length →
  4 * square_side = 80 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l2694_269412


namespace NUMINAMATH_CALUDE_max_abs_z5_l2694_269463

theorem max_abs_z5 (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h1 : Complex.abs z₁ ≤ 1)
  (h2 : Complex.abs z₂ ≤ 1)
  (h3 : Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h4 : Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h5 : Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄)) :
  Complex.abs z₅ ≤ Real.sqrt 3 ∧ ∃ z₁ z₂ z₃ z₄ z₅, 
    Complex.abs z₁ ≤ 1 ∧ 
    Complex.abs z₂ ≤ 1 ∧
    Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄) ∧
    Complex.abs z₅ = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_abs_z5_l2694_269463


namespace NUMINAMATH_CALUDE_largest_fraction_l2694_269419

theorem largest_fraction :
  let a := (2 : ℚ) / 5
  let b := (4 : ℚ) / 9
  let c := (7 : ℚ) / 15
  let d := (11 : ℚ) / 18
  let e := (16 : ℚ) / 35
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l2694_269419


namespace NUMINAMATH_CALUDE_world_record_rates_l2694_269407

-- Define the world records
def hotdog_record : ℕ := 75
def hotdog_time : ℕ := 10
def hamburger_record : ℕ := 97
def hamburger_time : ℕ := 3
def cheesecake_record : ℚ := 11
def cheesecake_time : ℕ := 9

-- Define Lisa's progress
def lisa_hotdogs : ℕ := 20
def lisa_hotdog_time : ℕ := 5
def lisa_hamburgers : ℕ := 60
def lisa_hamburger_time : ℕ := 2
def lisa_cheesecake : ℚ := 5
def lisa_cheesecake_time : ℕ := 5

-- Define the theorem
theorem world_record_rates : 
  (((hotdog_record - lisa_hotdogs : ℚ) / (hotdog_time - lisa_hotdog_time)) = 11) ∧
  (((hamburger_record - lisa_hamburgers : ℚ) / (hamburger_time - lisa_hamburger_time)) = 37) ∧
  (((cheesecake_record - lisa_cheesecake) / (cheesecake_time - lisa_cheesecake_time)) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_world_record_rates_l2694_269407


namespace NUMINAMATH_CALUDE_dave_tickets_remaining_l2694_269447

theorem dave_tickets_remaining (initial_tickets used_tickets : ℕ) :
  initial_tickets = 127 →
  used_tickets = 84 →
  initial_tickets - used_tickets = 43 :=
by sorry

end NUMINAMATH_CALUDE_dave_tickets_remaining_l2694_269447


namespace NUMINAMATH_CALUDE_cos_squared_plus_sin_minus_one_range_l2694_269494

theorem cos_squared_plus_sin_minus_one_range :
  ∀ x : ℝ, -2 ≤ (Real.cos x)^2 + Real.sin x - 1 ∧ (Real.cos x)^2 + Real.sin x - 1 ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_plus_sin_minus_one_range_l2694_269494


namespace NUMINAMATH_CALUDE_race_total_length_l2694_269468

/-- The total length of a race with four parts -/
def race_length (part1 part2 part3 part4 : ℝ) : ℝ :=
  part1 + part2 + part3 + part4

theorem race_total_length :
  race_length 15.5 21.5 21.5 16 = 74.5 := by
  sorry

end NUMINAMATH_CALUDE_race_total_length_l2694_269468


namespace NUMINAMATH_CALUDE_f_properties_l2694_269418

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

theorem f_properties :
  (∀ x > 0, f (-1) x ≥ 1/2) ∧ 
  (f (-1) 1 = 1/2) ∧
  (∀ x ≥ 1, f 1 x < (2/3) * x^3) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2694_269418


namespace NUMINAMATH_CALUDE_additive_inverse_solution_equal_surds_solution_l2694_269446

-- Part 1
theorem additive_inverse_solution (x : ℝ) : 
  x^2 + 3*x - 6 = -((-x + 1)) → x = -1 + Real.sqrt 6 ∨ x = -1 - Real.sqrt 6 := by sorry

-- Part 2
theorem equal_surds_solution (m : ℝ) :
  Real.sqrt (m^2 - 6) = Real.sqrt (6*m + 1) → m = 7 := by sorry

end NUMINAMATH_CALUDE_additive_inverse_solution_equal_surds_solution_l2694_269446


namespace NUMINAMATH_CALUDE_subtracted_number_l2694_269487

theorem subtracted_number (x y : ℤ) : x = 48 → 5 * x - y = 102 → y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2694_269487


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l2694_269411

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 3 * x + a

-- Define the property of m and 1 being roots of the equation f a x = 0
def are_roots (a m : ℝ) : Prop := f a m = 0 ∧ f a 1 = 0

-- Define the property of (m, 1) being the solution set of the inequality
def is_solution_set (a m : ℝ) : Prop :=
  ∀ x, f a x < 0 ↔ m < x ∧ x < 1

-- State the theorem
theorem solution_set_implies_m_value (a m : ℝ) :
  is_solution_set a m → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l2694_269411


namespace NUMINAMATH_CALUDE_scatter_plot_correct_placement_l2694_269443

/-- Represents a variable in a scatter plot -/
inductive Variable
| Forecast
| Explanatory

/-- Represents an axis in a scatter plot -/
inductive Axis
| X
| Y

/-- Determines the correct axis placement for a given variable -/
def correct_axis_placement (v : Variable) : Axis :=
  match v with
  | Variable.Forecast => Axis.Y
  | Variable.Explanatory => Axis.X

/-- Theorem stating the correct placement of variables in a scatter plot -/
theorem scatter_plot_correct_placement :
  (correct_axis_placement Variable.Forecast = Axis.Y) ∧
  (correct_axis_placement Variable.Explanatory = Axis.X) := by
  sorry

end NUMINAMATH_CALUDE_scatter_plot_correct_placement_l2694_269443


namespace NUMINAMATH_CALUDE_mode_median_mean_relationship_l2694_269496

def dataset : List ℕ := [20, 30, 40, 50, 60, 60, 70]

def mode (data : List ℕ) : ℕ := sorry

def median (data : List ℕ) : ℚ := sorry

def mean (data : List ℕ) : ℚ := sorry

theorem mode_median_mean_relationship :
  let m := mode dataset
  let med := median dataset
  let μ := mean dataset
  (m : ℚ) > med ∧ med > μ := by sorry

end NUMINAMATH_CALUDE_mode_median_mean_relationship_l2694_269496


namespace NUMINAMATH_CALUDE_ancient_chinese_rope_problem_l2694_269401

theorem ancient_chinese_rope_problem (x y : ℝ) :
  (1/2 : ℝ) * x - y = 5 ∧ y - (1/3 : ℝ) * x = 2 → x = 42 ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_rope_problem_l2694_269401


namespace NUMINAMATH_CALUDE_correct_average_equals_initial_l2694_269452

theorem correct_average_equals_initial (n : ℕ) (initial_avg : ℚ) 
  (correct1 incorrect1 correct2 incorrect2 : ℚ) : 
  n = 15 → 
  initial_avg = 37 → 
  correct1 = 64 → 
  incorrect1 = 52 → 
  correct2 = 27 → 
  incorrect2 = 39 → 
  (n * initial_avg - incorrect1 - incorrect2 + correct1 + correct2) / n = initial_avg := by
  sorry

end NUMINAMATH_CALUDE_correct_average_equals_initial_l2694_269452


namespace NUMINAMATH_CALUDE_square_garden_side_length_l2694_269448

/-- Given a square garden with a perimeter of 112 meters, prove that the length of each side is 28 meters. -/
theorem square_garden_side_length :
  ∀ (side_length : ℝ),
  (4 * side_length = 112) →
  side_length = 28 :=
by sorry

end NUMINAMATH_CALUDE_square_garden_side_length_l2694_269448
