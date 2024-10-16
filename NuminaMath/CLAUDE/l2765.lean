import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2765_276523

theorem complex_number_in_second_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := 1 + 2 * i + 3 * i^2
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2765_276523


namespace NUMINAMATH_CALUDE_tan_alpha_implies_c_equals_five_l2765_276500

theorem tan_alpha_implies_c_equals_five (α : Real) (c : Real) 
  (h1 : Real.tan α = -1/2) 
  (h2 : c = (2 * Real.cos α - Real.sin α) / (Real.sin α + Real.cos α)) : 
  c = 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_implies_c_equals_five_l2765_276500


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l2765_276540

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x^2 - 2*x) * Real.log x + (a - 1/2) * x^2 + 2*(1 - a)*x + a

theorem f_has_three_zeros (a : ℝ) (h : a < -2) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    ∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ :=
by sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l2765_276540


namespace NUMINAMATH_CALUDE_x_minus_y_values_l2765_276524

theorem x_minus_y_values (x y : ℝ) (hx : |x| = 4) (hy : |y| = 7) (hsum : x + y > 0) :
  x - y = -3 ∨ x - y = -11 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l2765_276524


namespace NUMINAMATH_CALUDE_stating_transfer_equality_l2765_276507

/-- Represents a glass containing a mixture of wine and water -/
structure Glass where
  total_volume : ℝ
  wine_volume : ℝ
  water_volume : ℝ
  volume_constraint : total_volume = wine_volume + water_volume

/-- Represents the state of two glasses after the transfer process -/
structure TransferState where
  wine_glass : Glass
  water_glass : Glass
  volume_conserved : wine_glass.total_volume = water_glass.total_volume

/-- 
Theorem stating that after the transfer process, the volume of wine in the water glass 
is equal to the volume of water in the wine glass 
-/
theorem transfer_equality (state : TransferState) : 
  state.wine_glass.water_volume = state.water_glass.wine_volume := by
  sorry

#check transfer_equality

end NUMINAMATH_CALUDE_stating_transfer_equality_l2765_276507


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2765_276548

theorem root_sum_theorem (a b c : ℝ) : 
  (a * b * c = -22) → 
  (a + b + c = 20) → 
  (a * b + b * c + c * a = 0) → 
  (b * c / a^2 + a * c / b^2 + a * b / c^2 = 3) := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2765_276548


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2765_276527

theorem triangle_area : ℝ → Prop :=
  fun area =>
    ∃ (x y : ℝ),
      (x + y = 2005 ∧
       x / 2005 + y / 2006 = 1 ∧
       x / 2006 + y / 2005 = 1) →
      area = 2005^2 / (2 * 4011)

-- The proof is omitted
theorem triangle_area_proof : triangle_area (2005^2 / (2 * 4011)) :=
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2765_276527


namespace NUMINAMATH_CALUDE_midpoint_endpoint_product_l2765_276595

/-- Given a segment CD with midpoint M and endpoint C, proves that the product of D's coordinates is -63 -/
theorem midpoint_endpoint_product (M C D : ℝ × ℝ) : 
  M = (4, -2) → C = (-1, 3) → M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → D.1 * D.2 = -63 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_endpoint_product_l2765_276595


namespace NUMINAMATH_CALUDE_ratio_hcf_to_lcm_l2765_276596

/-- Given two positive integers with a ratio of 3:4 and HCF of 4, their LCM is 48 -/
theorem ratio_hcf_to_lcm (a b : ℕ+) (h_ratio : a.val * 4 = b.val * 3) (h_hcf : Nat.gcd a.val b.val = 4) :
  Nat.lcm a.val b.val = 48 := by
  sorry

end NUMINAMATH_CALUDE_ratio_hcf_to_lcm_l2765_276596


namespace NUMINAMATH_CALUDE_binomial_square_value_l2765_276568

theorem binomial_square_value (a : ℚ) : 
  (∃ p q : ℚ, ∀ x, 9*x^2 + 27*x + a = (p*x + q)^2) → a = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_value_l2765_276568


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2765_276562

theorem quadratic_inequality (x : ℝ) : x^2 < x + 6 ↔ -2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2765_276562


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l2765_276529

def isComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬isSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l2765_276529


namespace NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l2765_276503

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  base_angle : ℝ

/-- Predicate to check if two isosceles triangles are different -/
def are_different_triangles (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.base ≠ t2.base ∨ t1.leg ≠ t2.leg

/-- Theorem stating that a base angle and opposite side do not uniquely determine an isosceles triangle -/
theorem isosceles_triangle_not_unique (base_angle : ℝ) (opposite_side : ℝ) :
  ∃ (t1 t2 : IsoscelesTriangle), 
    t1.base_angle = base_angle ∧
    t1.base = opposite_side ∧
    t2.base_angle = base_angle ∧
    t2.base = opposite_side ∧
    are_different_triangles t1 t2 :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l2765_276503


namespace NUMINAMATH_CALUDE_custom_op_example_l2765_276522

/-- Custom operation ⊕ for rational numbers -/
def custom_op (a b : ℚ) : ℚ := a * b + (a - b)

/-- Theorem stating that (-5) ⊕ 4 = -29 -/
theorem custom_op_example : custom_op (-5) 4 = -29 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l2765_276522


namespace NUMINAMATH_CALUDE_bruce_payment_l2765_276531

/-- The total amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1165 to the shopkeeper -/
theorem bruce_payment : total_amount 8 70 11 55 = 1165 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l2765_276531


namespace NUMINAMATH_CALUDE_pentagon_area_half_decagon_area_l2765_276569

/-- The area of a pentagon formed by connecting every second vertex of a regular decagon
    is half the area of the decagon. -/
theorem pentagon_area_half_decagon_area (n : ℝ) (h : n > 0) :
  ∃ (m : ℝ), m > 0 ∧ m / n = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_half_decagon_area_l2765_276569


namespace NUMINAMATH_CALUDE_percentage_problem_l2765_276508

theorem percentage_problem : 
  ∃ x : ℝ, (x / 100 * 50 + 50 / 100 * 860 = 860) ∧ x = 860 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2765_276508


namespace NUMINAMATH_CALUDE_goldfish_remaining_l2765_276526

def initial_goldfish : ℕ := 15
def fewer_goldfish : ℕ := 11

theorem goldfish_remaining : initial_goldfish - fewer_goldfish = 4 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_remaining_l2765_276526


namespace NUMINAMATH_CALUDE_chicken_problem_l2765_276560

theorem chicken_problem (total chickens_colten : ℕ) 
  (h_total : total = 383)
  (h_colten : chickens_colten = 37) : 
  ∃ (chickens_skylar chickens_quentin : ℕ),
    chickens_quentin = 2 * chickens_skylar + 25 ∧
    chickens_quentin + chickens_skylar + chickens_colten = total ∧
    3 * chickens_colten - chickens_skylar = 4 :=
by sorry

end NUMINAMATH_CALUDE_chicken_problem_l2765_276560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2765_276514

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_2 + a_3 = 15 and a_3 + a_4 = 20,
    prove that a_4 + a_5 = 25. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum1 : a 2 + a 3 = 15)
    (h_sum2 : a 3 + a 4 = 20) :
  a 4 + a 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2765_276514


namespace NUMINAMATH_CALUDE_min_correct_answers_is_17_l2765_276515

/-- AMC 12 scoring system and Sarah's strategy -/
structure AMC12 where
  total_questions : Nat
  attempted_questions : Nat
  points_correct : Nat
  points_incorrect : Nat
  points_unanswered : Nat
  min_score : Nat

/-- Calculate the minimum number of correct answers needed -/
def min_correct_answers (amc : AMC12) : Nat :=
  let unanswered := amc.total_questions - amc.attempted_questions
  let points_from_unanswered := unanswered * amc.points_unanswered
  let required_points := amc.min_score - points_from_unanswered
  (required_points + amc.points_correct - 1) / amc.points_correct

/-- Theorem stating the minimum number of correct answers needed -/
theorem min_correct_answers_is_17 (amc : AMC12) 
  (h1 : amc.total_questions = 30)
  (h2 : amc.attempted_questions = 24)
  (h3 : amc.points_correct = 7)
  (h4 : amc.points_incorrect = 0)
  (h5 : amc.points_unanswered = 2)
  (h6 : amc.min_score = 130) : 
  min_correct_answers amc = 17 := by
  sorry

#eval min_correct_answers {
  total_questions := 30,
  attempted_questions := 24,
  points_correct := 7,
  points_incorrect := 0,
  points_unanswered := 2,
  min_score := 130
}

end NUMINAMATH_CALUDE_min_correct_answers_is_17_l2765_276515


namespace NUMINAMATH_CALUDE_mixture_alcohol_percentage_l2765_276567

/-- Represents the alcohol content and volume of a solution -/
structure Solution where
  volume : ℝ
  alcoholPercentage : ℝ

/-- Calculates the amount of pure alcohol in a solution -/
def alcoholContent (s : Solution) : ℝ :=
  s.volume * s.alcoholPercentage

theorem mixture_alcohol_percentage 
  (x : Solution) 
  (y : Solution) 
  (h1 : x.volume = 250)
  (h2 : x.alcoholPercentage = 0.1)
  (h3 : y.alcoholPercentage = 0.3)
  (h4 : y.volume = 750) :
  let mixedSolution : Solution := ⟨x.volume + y.volume, (alcoholContent x + alcoholContent y) / (x.volume + y.volume)⟩
  mixedSolution.alcoholPercentage = 0.25 := by
sorry

end NUMINAMATH_CALUDE_mixture_alcohol_percentage_l2765_276567


namespace NUMINAMATH_CALUDE_coin_stack_problem_l2765_276557

/-- Thickness of a nickel in millimeters -/
def nickel_thickness : ℚ := 39/20

/-- Thickness of a quarter in millimeters -/
def quarter_thickness : ℚ := 35/20

/-- Total height of the stack in millimeters -/
def stack_height : ℚ := 20

/-- The number of coins in the stack -/
def num_coins : ℕ := 10

theorem coin_stack_problem :
  ∃ (n q : ℕ), n * nickel_thickness + q * quarter_thickness = stack_height ∧ n + q = num_coins :=
sorry

end NUMINAMATH_CALUDE_coin_stack_problem_l2765_276557


namespace NUMINAMATH_CALUDE_systematic_sampling_l2765_276532

/-- Systematic sampling problem -/
theorem systematic_sampling
  (total_population : ℕ)
  (sample_size : ℕ)
  (first_drawn : ℕ)
  (interval_start : ℕ)
  (interval_end : ℕ)
  (h1 : total_population = 960)
  (h2 : sample_size = 32)
  (h3 : first_drawn = 29)
  (h4 : interval_start = 200)
  (h5 : interval_end = 480) :
  (Finset.filter (fun n => interval_start ≤ (first_drawn + (total_population / sample_size) * (n - 1)) ∧
                           (first_drawn + (total_population / sample_size) * (n - 1)) ≤ interval_end)
                 (Finset.range sample_size)).card = 10 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_l2765_276532


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l2765_276512

def team_size : ℕ := 15
def starting_lineup_size : ℕ := 5
def preselected_players : ℕ := 3

theorem starting_lineup_combinations :
  Nat.choose (team_size - preselected_players) (starting_lineup_size - preselected_players) = 66 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l2765_276512


namespace NUMINAMATH_CALUDE_spider_legs_proof_l2765_276551

/-- The number of legs a single spider has -/
def spider_legs : ℕ := 8

/-- The number of spiders in the group -/
def group_size (L : ℕ) : ℕ := L / 2 + 10

theorem spider_legs_proof :
  (∀ L : ℕ, group_size L * L = 112) → spider_legs = 8 := by
  sorry

end NUMINAMATH_CALUDE_spider_legs_proof_l2765_276551


namespace NUMINAMATH_CALUDE_solve_for_x_l2765_276566

/-- The operation defined for real numbers a, b, c, d -/
def operation (a b c d : ℝ) : ℝ := a * d - b * c

/-- The theorem stating that if the operation on the given matrix equals 2023, then x = 2018 -/
theorem solve_for_x (x : ℝ) : operation (x + 1) (x + 2) (x - 3) (x - 1) = 2023 → x = 2018 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2765_276566


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_div_by_5_l2765_276563

/-- Converts a base 8 number to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a 3-digit base 8 number -/
def is_3digit_base8 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_div_by_5 :
  ∀ n : ℕ, is_3digit_base8 n → (base8_to_base10 n) % 5 = 0 → n ≤ 776 :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_div_by_5_l2765_276563


namespace NUMINAMATH_CALUDE_officer_selection_count_l2765_276587

/-- Represents a club with boys and girls -/
structure Club where
  boys : ℕ
  girls : ℕ

/-- Calculates the number of ways to choose officers of the same gender -/
def chooseOfficers (c : Club) : ℕ :=
  2 * (c.boys * (c.boys - 1) * (c.boys - 2))

theorem officer_selection_count (c : Club) (h1 : c.boys = 15) (h2 : c.girls = 15) :
  chooseOfficers c = 5460 := by
  sorry

#eval chooseOfficers { boys := 15, girls := 15 }

end NUMINAMATH_CALUDE_officer_selection_count_l2765_276587


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l2765_276521

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) :
  side_length = 7 ∧ exterior_angle = 45 →
  (360 / exterior_angle) * side_length = 56 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l2765_276521


namespace NUMINAMATH_CALUDE_hexagon_sectors_perimeter_l2765_276541

/-- The perimeter of a shape formed by removing three equal sectors from a regular hexagon -/
def shaded_perimeter (sector_perimeter : ℝ) : ℝ :=
  3 * sector_perimeter

theorem hexagon_sectors_perimeter :
  ∀ (sector_perimeter : ℝ),
  sector_perimeter = 18 →
  shaded_perimeter sector_perimeter = 54 := by
sorry

end NUMINAMATH_CALUDE_hexagon_sectors_perimeter_l2765_276541


namespace NUMINAMATH_CALUDE_stating_probability_all_types_proof_l2765_276546

/-- Represents the probability of finding all three types of dolls in 4 blind boxes -/
def probability_all_types (ratio_A ratio_B ratio_C : ℕ) : ℝ :=
  let total := ratio_A + ratio_B + ratio_C
  let p_A := ratio_A / total
  let p_B := ratio_B / total
  let p_C := ratio_C / total
  4 * p_C * 3 * p_B * p_A^2 + 4 * p_C * 3 * p_B^2 * p_A + 6 * p_C^2 * 2 * p_B * p_A

/-- 
Theorem stating that given the production ratio of dolls A:B:C as 6:3:1, 
the probability of finding all three types of dolls when buying 4 blind boxes at once is 0.216
-/
theorem probability_all_types_proof :
  probability_all_types 6 3 1 = 0.216 := by
  sorry

end NUMINAMATH_CALUDE_stating_probability_all_types_proof_l2765_276546


namespace NUMINAMATH_CALUDE_rosa_pages_last_week_l2765_276536

-- Define the total number of pages called
def total_pages : ℝ := 18.8

-- Define the number of pages called this week
def pages_this_week : ℝ := 8.6

-- Define the number of pages called last week
def pages_last_week : ℝ := total_pages - pages_this_week

-- Theorem to prove
theorem rosa_pages_last_week : pages_last_week = 10.2 := by
  sorry

end NUMINAMATH_CALUDE_rosa_pages_last_week_l2765_276536


namespace NUMINAMATH_CALUDE_four_point_theorem_l2765_276513

/-- Given four points A, B, C, D in a plane, if for any point P the inequality 
    PA + PD ≥ PB + PC holds, then B and C lie on the segment AD and AB = CD. -/
theorem four_point_theorem (A B C D : EuclideanSpace ℝ (Fin 2)) :
  (∀ P : EuclideanSpace ℝ (Fin 2), dist P A + dist P D ≥ dist P B + dist P C) →
  (∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ 
    B = (1 - t₁) • A + t₁ • D ∧ 
    C = (1 - t₂) • A + t₂ • D) ∧
  dist A B = dist C D := by
  sorry

end NUMINAMATH_CALUDE_four_point_theorem_l2765_276513


namespace NUMINAMATH_CALUDE_symmetry_of_point_l2765_276558

def is_symmetrical_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetry_of_point : 
  is_symmetrical_wrt_origin (-1, 1) (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l2765_276558


namespace NUMINAMATH_CALUDE_floor_equality_iff_in_interval_l2765_276589

theorem floor_equality_iff_in_interval (x : ℝ) :
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋ ↔ 5/3 ≤ x ∧ x < 7/3 := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_iff_in_interval_l2765_276589


namespace NUMINAMATH_CALUDE_calculate_y_l2765_276547

-- Define the triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real
  x : Real
  y : Real
  z : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.Z = 4 * t.X ∧ t.x = 35 ∧ t.z = 60

-- Define the Law of Sines
def law_of_sines (t : Triangle) : Prop :=
  t.x / Real.sin t.X = t.y / Real.sin t.Y ∧
  t.y / Real.sin t.Y = t.z / Real.sin t.Z ∧
  t.z / Real.sin t.Z = t.x / Real.sin t.X

-- Theorem statement
theorem calculate_y (t : Triangle) 
  (h1 : triangle_conditions t) 
  (h2 : law_of_sines t) : 
  ∃ y : Real, t.y = y :=
sorry

end NUMINAMATH_CALUDE_calculate_y_l2765_276547


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2765_276517

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ), ∀ (x : ℚ), x ≠ 4 ∧ x ≠ 2 →
    (6 * x + 2) / ((x - 4) * (x - 2)^3) = 
    P / (x - 4) + Q / (x - 2) + R / (x - 2)^3 ∧
    P = 13 / 4 ∧ Q = -13 / 2 ∧ R = -7 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2765_276517


namespace NUMINAMATH_CALUDE_solution_range_l2765_276556

-- Define the system of linear equations
def system (x y a : ℝ) : Prop :=
  (3 * x + y = 1 + a) ∧ (x + 3 * y = 3)

-- Define the theorem
theorem solution_range (x y a : ℝ) :
  system x y a → x + y < 2 → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2765_276556


namespace NUMINAMATH_CALUDE_andrew_payment_l2765_276597

/-- The total amount Andrew paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_price grape_weight mango_price mango_weight : ℕ) : ℕ :=
  grape_price * grape_weight + mango_price * mango_weight

/-- Theorem stating that Andrew paid 1428 to the shopkeeper -/
theorem andrew_payment :
  total_amount 98 11 50 7 = 1428 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l2765_276597


namespace NUMINAMATH_CALUDE_show_revenue_l2765_276588

theorem show_revenue (first_show_attendance : ℕ) (ticket_price : ℕ) : 
  first_show_attendance = 200 →
  ticket_price = 25 →
  (first_show_attendance * ticket_price + 3 * first_show_attendance * ticket_price) = 20000 :=
by sorry

end NUMINAMATH_CALUDE_show_revenue_l2765_276588


namespace NUMINAMATH_CALUDE_abs_complex_fraction_equals_sqrt_two_l2765_276537

/-- The absolute value of the complex number (1-3i)/(1+2i) is equal to √2 -/
theorem abs_complex_fraction_equals_sqrt_two :
  let z : ℂ := (1 - 3*I) / (1 + 2*I)
  ‖z‖ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_complex_fraction_equals_sqrt_two_l2765_276537


namespace NUMINAMATH_CALUDE_circle_radius_with_max_inscribed_rectangle_l2765_276561

theorem circle_radius_with_max_inscribed_rectangle (r : ℝ) : 
  r > 0 → 
  (∃ (rect_area : ℝ), rect_area = 50 ∧ rect_area = 2 * r^2) → 
  r = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_with_max_inscribed_rectangle_l2765_276561


namespace NUMINAMATH_CALUDE_triangle_sine_cosine_sum_l2765_276564

theorem triangle_sine_cosine_sum (A B C x y z : ℝ) :
  A + B + C = π →
  x * Real.sin A + y * Real.sin B + z * Real.sin C = 0 →
  (y + z * Real.cos A) * (z + x * Real.cos B) * (x + y * Real.cos C) + 
  (y * Real.cos A + z) * (z * Real.cos B + x) * (x * Real.cos C + y) = 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_cosine_sum_l2765_276564


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_coefficient_sum_l2765_276571

/-- The area of a cross-section in a cylinder --/
theorem cylinder_cross_section_area :
  ∀ (r : ℝ) (θ : ℝ),
  r = 8 →
  θ = π / 2 →
  ∃ (A : ℝ),
  A = 16 * Real.sqrt 3 * π + 16 * Real.sqrt 6 ∧
  A = (r^2 * θ / 4 + r^2 * Real.sin (θ / 2) * Real.cos (θ / 2)) * Real.sqrt 3 :=
by sorry

/-- The sum of coefficients in the area expression --/
theorem coefficient_sum :
  ∃ (d e : ℝ) (f : ℕ),
  16 * Real.sqrt 3 * π + 16 * Real.sqrt 6 = d * π + e * Real.sqrt f ∧
  d + e + f = 38 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_coefficient_sum_l2765_276571


namespace NUMINAMATH_CALUDE_bakers_remaining_cakes_l2765_276592

theorem bakers_remaining_cakes 
  (initial_cakes : ℝ) 
  (bought_cakes : ℝ) 
  (h1 : initial_cakes = 397.5) 
  (h2 : bought_cakes = 289) : 
  initial_cakes - bought_cakes = 108.5 := by
sorry

end NUMINAMATH_CALUDE_bakers_remaining_cakes_l2765_276592


namespace NUMINAMATH_CALUDE_fuel_cost_per_refill_l2765_276578

/-- 
Given the total fuel cost and number of refills, 
calculate the cost of one refilling.
-/
theorem fuel_cost_per_refill 
  (total_cost : ℕ) 
  (num_refills : ℕ) 
  (h1 : total_cost = 63)
  (h2 : num_refills = 3)
  : total_cost / num_refills = 21 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_per_refill_l2765_276578


namespace NUMINAMATH_CALUDE_path_length_proof_l2765_276516

theorem path_length_proof :
  let rectangle_width : ℝ := 3
  let rectangle_height : ℝ := 4
  let diagonal_length : ℝ := (rectangle_width^2 + rectangle_height^2).sqrt
  let vertical_segments : ℝ := 2 * rectangle_height
  let horizontal_segments : ℝ := 3 * rectangle_width
  diagonal_length + vertical_segments + horizontal_segments = 22 := by
sorry

end NUMINAMATH_CALUDE_path_length_proof_l2765_276516


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2765_276504

theorem determinant_of_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 2, 5]
  Matrix.det A = 18 := by sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2765_276504


namespace NUMINAMATH_CALUDE_probability_problem_l2765_276559

structure JarContents where
  red : Nat
  white : Nat
  black : Nat

def jarA : JarContents := { red := 5, white := 2, black := 3 }
def jarB : JarContents := { red := 4, white := 3, black := 3 }

def totalBalls (jar : JarContents) : Nat :=
  jar.red + jar.white + jar.black

def P_A1 : Rat := jarA.red / totalBalls jarA
def P_A2 : Rat := jarA.white / totalBalls jarA
def P_A3 : Rat := jarA.black / totalBalls jarA

def P_B_given_A1 : Rat := (jarB.red + 1) / (totalBalls jarB + 1)
def P_B_given_A2 : Rat := jarB.red / (totalBalls jarB + 1)
def P_B_given_A3 : Rat := jarB.red / (totalBalls jarB + 1)

theorem probability_problem :
  (P_B_given_A1 = 5 / 11) ∧
  (P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3 = 9 / 22) ∧
  (P_A1 + P_A2 + P_A3 = 1) :=
by sorry

#check probability_problem

end NUMINAMATH_CALUDE_probability_problem_l2765_276559


namespace NUMINAMATH_CALUDE_oplus_neg_two_three_l2765_276575

def oplus (a b : ℝ) : ℝ := a * (a - b) + 1

theorem oplus_neg_two_three : oplus (-2) 3 = 11 := by sorry

end NUMINAMATH_CALUDE_oplus_neg_two_three_l2765_276575


namespace NUMINAMATH_CALUDE_sin_70_degrees_l2765_276594

theorem sin_70_degrees (k : ℝ) (h : Real.sin (10 * π / 180) = k) :
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_70_degrees_l2765_276594


namespace NUMINAMATH_CALUDE_tangency_points_form_circular_arc_l2765_276520

structure Segment where
  A : Point
  B : Point

structure Circle where
  center : Point
  radius : ℝ

def TangentCirclePair (s : Segment) (c1 c2 : Circle) : Prop :=
  -- Definition of tangent circle pair inscribed in segment
  sorry

def TangencyPoint (s : Segment) (c1 c2 : Circle) : Point :=
  -- Definition of tangency point between two circles
  sorry

def CircularArc (A B : Point) : Set Point :=
  -- Definition of a circular arc with endpoints A and B
  sorry

def AngleBisector (s : Segment) (arc : Set Point) : Set Point :=
  -- Definition of angle bisector between chord AB and segment arc
  sorry

theorem tangency_points_form_circular_arc (s : Segment) :
  ∃ (arc : Set Point), 
    (arc = CircularArc s.A s.B) ∧ 
    (arc = AngleBisector s arc) ∧
    (∀ (c1 c2 : Circle), TangentCirclePair s c1 c2 → 
      TangencyPoint s c1 c2 ∈ arc) := by
  sorry

end NUMINAMATH_CALUDE_tangency_points_form_circular_arc_l2765_276520


namespace NUMINAMATH_CALUDE_unique_quadratic_pair_l2765_276580

/-- A function that checks if a quadratic equation has exactly one real solution -/
def hasExactlyOneRealSolution (a b c : ℤ) : Prop :=
  b * b = 4 * a * c

/-- The theorem stating that there exists exactly one ordered pair (b,c) satisfying the conditions -/
theorem unique_quadratic_pair :
  ∃! (b c : ℕ), 
    0 < b ∧ b ≤ 6 ∧
    0 < c ∧ c ≤ 6 ∧
    hasExactlyOneRealSolution 1 b c ∧
    hasExactlyOneRealSolution 1 c b :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_pair_l2765_276580


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l2765_276538

/-- The parabolas y = (x - 2)^2 and x - 5 = (y + 1)^2 intersect at four points that lie on a circle with radius squared equal to 1.5 -/
theorem intersection_points_on_circle :
  ∃ (c : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ), 
      (p.2 = (p.1 - 2)^2 ∧ p.1 - 5 = (p.2 + 1)^2) →
      (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    r^2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l2765_276538


namespace NUMINAMATH_CALUDE_at_least_two_sums_divisible_by_p_l2765_276584

def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem at_least_two_sums_divisible_by_p (p a b c d : ℕ) (hp : p > 2) (hprime : Nat.Prime p)
  (ha : ¬ p ∣ a) (hb : ¬ p ∣ b) (hc : ¬ p ∣ c) (hd : ¬ p ∣ d)
  (h : ∀ r : ℕ, ¬ p ∣ r → 
    fractional_part (r * a / p) + fractional_part (r * b / p) + 
    fractional_part (r * c / p) + fractional_part (r * d / p) = 2) :
  (∃ (x y : ℕ × ℕ), x ≠ y ∧ 
    (x ∈ [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]) ∧
    (y ∈ [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]) ∧
    p ∣ (x.1 + x.2) ∧ p ∣ (y.1 + y.2)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_sums_divisible_by_p_l2765_276584


namespace NUMINAMATH_CALUDE_crayons_in_drawer_l2765_276550

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 9

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- The total number of crayons in the drawer after Benny's addition -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem crayons_in_drawer : total_crayons = 12 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_drawer_l2765_276550


namespace NUMINAMATH_CALUDE_exists_n_divisible_by_1987_l2765_276577

theorem exists_n_divisible_by_1987 : ∃ n : ℕ, (1987 : ℕ) ∣ (n^n + (n+1)^n) := by
  sorry

end NUMINAMATH_CALUDE_exists_n_divisible_by_1987_l2765_276577


namespace NUMINAMATH_CALUDE_circle_has_zero_radius_l2765_276576

/-- The equation of a circle with radius 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 10*x + y^2 - 4*y + 29 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-5, 2)

theorem circle_has_zero_radius :
  ∀ x y : ℝ, circle_equation x y ↔ (x, y) = circle_center :=
by sorry

end NUMINAMATH_CALUDE_circle_has_zero_radius_l2765_276576


namespace NUMINAMATH_CALUDE_min_value_tangent_sum_l2765_276533

theorem min_value_tangent_sum (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π / 2) :
  3 * Real.tan B * Real.tan C + 2 * Real.tan A * Real.tan C + Real.tan A * Real.tan B ≥ 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_tangent_sum_l2765_276533


namespace NUMINAMATH_CALUDE_stream_speed_prove_stream_speed_l2765_276583

/-- The speed of a stream given a swimmer's still water speed and relative upstream/downstream times -/
theorem stream_speed (still_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  let stream_speed := (still_speed * (time_ratio - 1)) / (time_ratio + 1)
  stream_speed

/-- Proves that the speed of the stream is 3 km/h given the conditions -/
theorem prove_stream_speed :
  stream_speed 9 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_prove_stream_speed_l2765_276583


namespace NUMINAMATH_CALUDE_spinner_probability_l2765_276501

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 → pB = 1/3 → pD = 1/6 → pA + pB + pC + pD = 1 → pC = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2765_276501


namespace NUMINAMATH_CALUDE_suitable_land_size_l2765_276574

def previous_property : ℝ := 2
def land_multiplier : ℝ := 10
def pond_size : ℝ := 1

theorem suitable_land_size :
  let new_property := previous_property * land_multiplier
  let suitable_land := new_property - pond_size
  suitable_land = 19 := by sorry

end NUMINAMATH_CALUDE_suitable_land_size_l2765_276574


namespace NUMINAMATH_CALUDE_value_of_expression_l2765_276579

theorem value_of_expression (x : ℤ) (h : x = -4) : 5 * x - 2 = -22 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2765_276579


namespace NUMINAMATH_CALUDE_total_rectangles_area_2_l2765_276573

-- Define the structure of the figure
structure Figure where
  patterns : List String
  small_square_side : ℕ

-- Define a rectangle in the figure
structure Rectangle where
  width : ℕ
  height : ℕ

-- Function to calculate the area of a rectangle
def rectangle_area (r : Rectangle) : ℕ :=
  r.width * r.height

-- Function to count rectangles with area 2 in a specific pattern
def count_rectangles_area_2 (pattern : String) : ℕ :=
  match pattern with
  | "2" => 10
  | "0" => 12
  | "1" => 4
  | "4" => 8
  | _ => 0

-- Theorem stating the total number of rectangles with area 2
theorem total_rectangles_area_2 (fig : Figure) 
  (h1 : fig.small_square_side = 1) 
  (h2 : fig.patterns = ["2", "0", "1", "4"]) : 
  (fig.patterns.map count_rectangles_area_2).sum = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_rectangles_area_2_l2765_276573


namespace NUMINAMATH_CALUDE_integral_problems_l2765_276535

theorem integral_problems :
  (∃ k : ℝ, (∫ x in (0:ℝ)..2, (3*x^2 + k)) = 10 ∧ k = 1) ∧
  (∫ x in (-1:ℝ)..8, x^(1/3)) = 45/4 :=
by sorry

end NUMINAMATH_CALUDE_integral_problems_l2765_276535


namespace NUMINAMATH_CALUDE_modular_inverse_of_7_mod_26_l2765_276585

theorem modular_inverse_of_7_mod_26 : ∃ x : ℕ, x ∈ Finset.range 26 ∧ (7 * x) % 26 = 1 := by
  use 15
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_7_mod_26_l2765_276585


namespace NUMINAMATH_CALUDE_circle_ratio_after_radius_increase_l2765_276525

/-- 
For any circle with radius r, if the radius is increased by 2 units, 
the ratio of the new circumference to the new diameter is equal to π.
-/
theorem circle_ratio_after_radius_increase (r : ℝ) : 
  let new_radius : ℝ := r + 2
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_radius_increase_l2765_276525


namespace NUMINAMATH_CALUDE_complex_magnitude_l2765_276511

theorem complex_magnitude (r s : ℝ) (z : ℂ) (hr : |r| < 4) (hs : s ≠ 0) 
  (heq : s * z + 1 / z = r) :
  ∃ (c : ℝ), c ≥ 0 ∧ c^2 = (2 * (r^2 - 2*s) + 2*r * Real.sqrt (r^2 - 4*s)) / (4 * s^2) ∧ 
  Complex.abs z = c := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2765_276511


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l2765_276502

theorem parabola_hyperbola_tangent (m : ℝ) : 
  (∀ x y : ℝ, y = x^2 + 4 ∧ y^2 - 4*m*x^2 = 4 → 
    (∃! u : ℝ, u^2 + (8 - 4*m)*u + 12 = 0)) → 
  (m = 2 + Real.sqrt 3 ∨ m = 2 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l2765_276502


namespace NUMINAMATH_CALUDE_arith_geom_seq_sum_30_l2765_276549

/-- An arithmetic-geometric sequence with its partial sums -/
structure ArithGeomSeq where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums of the sequence
  is_arith_geom : ∀ n, (S (n + 10) - S n) / (S (n + 20) - S (n + 10)) = (S (n + 20) - S (n + 10)) / (S (n + 30) - S (n + 20))

/-- Theorem: For an arithmetic-geometric sequence, if S_10 = 10 and S_20 = 30, then S_30 = 70 -/
theorem arith_geom_seq_sum_30 (seq : ArithGeomSeq) (h1 : seq.S 10 = 10) (h2 : seq.S 20 = 30) : 
  seq.S 30 = 70 := by
  sorry

end NUMINAMATH_CALUDE_arith_geom_seq_sum_30_l2765_276549


namespace NUMINAMATH_CALUDE_sum_of_roots_and_constant_l2765_276519

theorem sum_of_roots_and_constant (a b c : ℝ) : 
  (1^2 + a*1 + 2 = 0) → 
  (a^2 + 5*a + c = 0) → 
  (b^2 + 5*b + c = 0) → 
  a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_and_constant_l2765_276519


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l2765_276505

theorem quadratic_real_solutions (p : ℝ) :
  (∃ x : ℝ, x^2 + p = 0) ↔ p ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l2765_276505


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2765_276510

theorem partial_fraction_decomposition :
  ∃! (A B C : ℝ), ∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 →
    5 * x^2 / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2 ∧
    A = 20 ∧ B = -15 ∧ C = -10 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2765_276510


namespace NUMINAMATH_CALUDE_emmy_lost_ipods_l2765_276544

/-- The number of iPods Emmy lost -/
def ipods_lost : ℕ := sorry

/-- The number of iPods Rosa has -/
def rosa_ipods : ℕ := sorry

theorem emmy_lost_ipods : ipods_lost = 6 :=
  by
  have h1 : 14 - ipods_lost = 2 * rosa_ipods := sorry
  have h2 : (14 - ipods_lost) + rosa_ipods = 12 := sorry
  sorry

#check emmy_lost_ipods

end NUMINAMATH_CALUDE_emmy_lost_ipods_l2765_276544


namespace NUMINAMATH_CALUDE_digit_sum_property_l2765_276570

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem digit_sum_property (n : ℕ) 
  (h1 : sum_of_digits n = 100)
  (h2 : sum_of_digits (44 * n) = 800) :
  sum_of_digits (3 * n) = 300 := by sorry

end NUMINAMATH_CALUDE_digit_sum_property_l2765_276570


namespace NUMINAMATH_CALUDE_f_additive_l2765_276534

/-- A function that satisfies f(a+b) = f(a) + f(b) for all real a and b -/
def f (x : ℝ) : ℝ := 3 * x

/-- Theorem stating that f(a+b) = f(a) + f(b) for all real a and b -/
theorem f_additive (a b : ℝ) : f (a + b) = f a + f b := by
  sorry

end NUMINAMATH_CALUDE_f_additive_l2765_276534


namespace NUMINAMATH_CALUDE_diamond_digit_value_l2765_276528

/-- Given that ◇6₉ = ◇3₁₀, where ◇ represents a digit, prove that ◇ = 3 -/
theorem diamond_digit_value :
  ∀ (diamond : ℕ),
  diamond < 10 →
  diamond * 9 + 6 = diamond * 10 + 3 →
  diamond = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_diamond_digit_value_l2765_276528


namespace NUMINAMATH_CALUDE_hex_1F4B_equals_8011_l2765_276582

-- Define the hexadecimal digits and their decimal equivalents
def hex_to_dec (c : Char) : Nat :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3 | '4' => 4 | '5' => 5 | '6' => 6 | '7' => 7
  | '8' => 8 | '9' => 9 | 'A' => 10 | 'B' => 11 | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

-- Define the conversion function from hexadecimal to decimal
def hex_to_decimal (s : String) : Nat :=
  s.foldl (fun acc c => 16 * acc + hex_to_dec c) 0

-- Theorem statement
theorem hex_1F4B_equals_8011 :
  hex_to_decimal "1F4B" = 8011 := by
  sorry

end NUMINAMATH_CALUDE_hex_1F4B_equals_8011_l2765_276582


namespace NUMINAMATH_CALUDE_female_officers_count_l2765_276590

theorem female_officers_count (total_on_duty : ℕ) (female_percentage : ℚ) 
  (h1 : total_on_duty = 180)
  (h2 : female_percentage = 18 / 100)
  (h3 : (total_on_duty / 2 : ℚ) = female_percentage * (female_officers_total : ℚ)) :
  female_officers_total = 500 :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l2765_276590


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2765_276506

/-- The repeating decimal 0.363636... as a real number -/
def repeating_decimal : ℚ := 36 / 99

/-- Theorem stating that the repeating decimal 0.363636... is equal to 4/11 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2765_276506


namespace NUMINAMATH_CALUDE_KBrO3_molecular_weight_l2765_276554

/-- Atomic weight of potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- Atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Molecular weight of KBrO3 in g/mol -/
def molecular_weight_KBrO3 : ℝ :=
  atomic_weight_K + atomic_weight_Br + 3 * atomic_weight_O

/-- Theorem stating that the molecular weight of KBrO3 is 167.00 g/mol -/
theorem KBrO3_molecular_weight :
  molecular_weight_KBrO3 = 167.00 := by
  sorry

end NUMINAMATH_CALUDE_KBrO3_molecular_weight_l2765_276554


namespace NUMINAMATH_CALUDE_pirate_count_l2765_276598

theorem pirate_count : ∃ p : ℕ, 
  p > 0 ∧ 
  (∃ (participants : ℕ), participants = p - 10 ∧ 
    (54 : ℚ) / 100 * participants = (↑⌊(54 : ℚ) / 100 * participants⌋ : ℚ) ∧ 
    (34 : ℚ) / 100 * participants = (↑⌊(34 : ℚ) / 100 * participants⌋ : ℚ) ∧ 
    (2 : ℚ) / 3 * p = (↑⌊(2 : ℚ) / 3 * p⌋ : ℚ)) ∧ 
  p = 60 := by
  sorry

end NUMINAMATH_CALUDE_pirate_count_l2765_276598


namespace NUMINAMATH_CALUDE_remainder_theorem_l2765_276530

theorem remainder_theorem (m : ℤ) (h : m % 9 = 3) : (3 * m + 2436) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2765_276530


namespace NUMINAMATH_CALUDE_max_students_is_eight_l2765_276518

/-- Represents the relationship between students -/
def KnowsRelation (n : ℕ) := Fin n → Fin n → Prop

/-- Property: Among any 3 students, there are 2 who know each other -/
def ThreeKnowTwo (n : ℕ) (knows : KnowsRelation n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    knows a b ∨ knows b c ∨ knows a c

/-- Property: Among any 4 students, there are 2 who do not know each other -/
def FourDontKnowTwo (n : ℕ) (knows : KnowsRelation n) : Prop :=
  ∀ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(knows a b) ∨ ¬(knows a c) ∨ ¬(knows a d) ∨
    ¬(knows b c) ∨ ¬(knows b d) ∨ ¬(knows c d)

/-- The maximum number of students satisfying the conditions is 8 -/
theorem max_students_is_eight :
  ∃ (knows : KnowsRelation 8), ThreeKnowTwo 8 knows ∧ FourDontKnowTwo 8 knows ∧
  ∀ n > 8, ¬∃ (knows : KnowsRelation n), ThreeKnowTwo n knows ∧ FourDontKnowTwo n knows :=
sorry

end NUMINAMATH_CALUDE_max_students_is_eight_l2765_276518


namespace NUMINAMATH_CALUDE_apple_difference_theorem_l2765_276581

/-- The difference between the initial number of apples and the remaining number of apples -/
def appleDifference (initialApples remainingApples : ℕ) : ℕ :=
  initialApples - remainingApples

/-- Theorem: The difference between 46 apples and 14 apples is 32 -/
theorem apple_difference_theorem :
  appleDifference 46 14 = 32 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_theorem_l2765_276581


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l2765_276543

theorem sum_of_reciprocals_squared (a b c d : ℝ) :
  a = 2 * Real.sqrt 2 + 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  b = -2 * Real.sqrt 2 + 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  c = 2 * Real.sqrt 2 - 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  d = -2 * Real.sqrt 2 - 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  (1/a + 1/b + 1/c + 1/d)^2 = 4/45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l2765_276543


namespace NUMINAMATH_CALUDE_odd_square_plus_multiple_l2765_276552

theorem odd_square_plus_multiple (o n : ℤ) 
  (ho : ∃ k, o = 2 * k + 1) : 
  Odd (o^2 + n * o) ↔ Even n := by
sorry

end NUMINAMATH_CALUDE_odd_square_plus_multiple_l2765_276552


namespace NUMINAMATH_CALUDE_milk_production_days_l2765_276599

/-- Given that x cows produce x+1 cans of milk in x+2 days, 
    this theorem proves the number of days it takes x+3 cows to produce x+5 cans of milk. -/
theorem milk_production_days (x : ℝ) (h : x > 0) : 
  let initial_cows := x
  let initial_milk := x + 1
  let initial_days := x + 2
  let new_cows := x + 3
  let new_milk := x + 5
  let daily_production_per_cow := initial_milk / (initial_cows * initial_days)
  let days_for_new_production := new_milk / (new_cows * daily_production_per_cow)
  days_for_new_production = x * (x + 2) * (x + 5) / ((x + 1) * (x + 3)) :=
by sorry

end NUMINAMATH_CALUDE_milk_production_days_l2765_276599


namespace NUMINAMATH_CALUDE_equation_solutions_l2765_276509

theorem equation_solutions :
  (∀ x : ℝ, 3 * x^2 - 4 * x + 5 ≠ 0) ∧
  (∃! s : Set ℝ, s = {-2, 1} ∧ ∀ x ∈ s, (x + 1) * (x + 2) = 2 * x + 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2765_276509


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_for_reciprocal_fraction_l2765_276539

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the decimal 0.abc -/
def DecimalABC (a b c : Digit) : ℚ := (a.val * 100 + b.val * 10 + c.val : ℕ) / 1000

/-- The theorem statement -/
theorem largest_sum_of_digits_for_reciprocal_fraction :
  ∀ (a b c : Digit) (y : ℕ+),
    (0 < y.val) → (y.val ≤ 16) →
    (DecimalABC a b c = 1 / y) →
    (∀ (a' b' c' : Digit) (y' : ℕ+),
      (0 < y'.val) → (y'.val ≤ 16) →
      (DecimalABC a' b' c' = 1 / y') →
      (a.val + b.val + c.val ≥ a'.val + b'.val + c'.val)) →
    (a.val + b.val + c.val = 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_of_digits_for_reciprocal_fraction_l2765_276539


namespace NUMINAMATH_CALUDE_pie_remainder_l2765_276591

theorem pie_remainder (carlos_portion jessica_portion : Real) : 
  carlos_portion = 0.6 →
  jessica_portion = 0.25 * (1 - carlos_portion) →
  1 - carlos_portion - jessica_portion = 0.3 := by
sorry

end NUMINAMATH_CALUDE_pie_remainder_l2765_276591


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l2765_276565

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. --/
def isArithmeticSequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_n_value
  (a : ℕ → ℤ) (d : ℤ) (h : isArithmeticSequence a d)
  (h2 : a 2 = 12) (hn : a n = -20) (hd : d = -2) :
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l2765_276565


namespace NUMINAMATH_CALUDE_books_returned_on_wednesday_l2765_276542

theorem books_returned_on_wednesday (initial_books : ℕ) (tuesday_out : ℕ) (thursday_out : ℕ) (final_books : ℕ) : 
  initial_books = 250 → 
  tuesday_out = 120 → 
  thursday_out = 15 → 
  final_books = 150 → 
  initial_books - tuesday_out + (initial_books - tuesday_out - final_books + thursday_out) - thursday_out = final_books := by
  sorry

end NUMINAMATH_CALUDE_books_returned_on_wednesday_l2765_276542


namespace NUMINAMATH_CALUDE_midpoint_triangle_half_area_l2765_276593

/-- A rectangle with midpoints on longer sides -/
structure RectangleWithMidpoints where
  length : ℝ
  width : ℝ
  width_half_length : width = length / 2
  p : ℝ × ℝ
  q : ℝ × ℝ
  p_midpoint : p = (0, length / 2)
  q_midpoint : q = (length, length / 2)

/-- The area of the triangle formed by midpoints and corner is half the rectangle area -/
theorem midpoint_triangle_half_area (r : RectangleWithMidpoints) :
    let triangle_area := (r.length * r.length / 2) / 2
    let rectangle_area := r.length * r.width
    triangle_area = rectangle_area / 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_triangle_half_area_l2765_276593


namespace NUMINAMATH_CALUDE_cubic_inequality_l2765_276553

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3*a*b*c ≥ a*b*(a + b) + b*c*(b + c) + c*a*(c + a) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2765_276553


namespace NUMINAMATH_CALUDE_total_insect_legs_l2765_276586

/-- The number of insects in the laboratory -/
def num_insects : ℕ := 6

/-- The number of legs each insect has -/
def legs_per_insect : ℕ := 6

/-- The total number of legs for all insects in the laboratory -/
def total_legs : ℕ := num_insects * legs_per_insect

/-- Theorem stating that the total number of legs is 36 -/
theorem total_insect_legs : total_legs = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_insect_legs_l2765_276586


namespace NUMINAMATH_CALUDE_exp_ge_x_plus_one_l2765_276545

theorem exp_ge_x_plus_one : ∀ x : ℝ, Real.exp x ≥ x + 1 := by
  sorry

end NUMINAMATH_CALUDE_exp_ge_x_plus_one_l2765_276545


namespace NUMINAMATH_CALUDE_different_suit_combinations_l2765_276572

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := 13

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 4

/-- Theorem stating the number of ways to choose 4 cards of different suits from a standard deck -/
theorem different_suit_combinations : 
  (number_of_suits.choose cards_to_choose) * (cards_per_suit ^ cards_to_choose) = 28561 := by
  sorry

end NUMINAMATH_CALUDE_different_suit_combinations_l2765_276572


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l2765_276555

/-- Two lines are parallel if and only if their slopes are equal -/
theorem parallel_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, (a + 2) * x + (a + 3) * y - 5 = 0 ∧ 
               6 * x + (2 * a - 1) * y - 5 = 0) →
  (a + 2) / 6 = (a + 3) / (2 * a - 1) →
  a = -5/2 := by
sorry


end NUMINAMATH_CALUDE_parallel_lines_slope_l2765_276555
