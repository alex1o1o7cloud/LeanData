import Mathlib

namespace NUMINAMATH_CALUDE_recordedLineLengthApprox_l1246_124674

/-- Represents the parameters of a record turntable --/
structure TurntableParams where
  revPerMinute : ℝ
  playTime : ℝ
  initialDiameter : ℝ
  finalDiameter : ℝ

/-- Calculates the length of the recorded line on a turntable --/
def recordedLineLength (params : TurntableParams) : ℝ :=
  sorry

/-- The main theorem stating the length of the recorded line --/
theorem recordedLineLengthApprox (params : TurntableParams) 
  (h1 : params.revPerMinute = 100)
  (h2 : params.playTime = 24.5)
  (h3 : params.initialDiameter = 29)
  (h4 : params.finalDiameter = 11.5) :
  abs (recordedLineLength params - 155862.265789099) < 1e-6 := by
  sorry

end NUMINAMATH_CALUDE_recordedLineLengthApprox_l1246_124674


namespace NUMINAMATH_CALUDE_nut_distribution_properties_l1246_124690

/-- Represents the state of nut distribution among three people -/
structure NutState where
  anya : ℕ
  borya : ℕ
  vitya : ℕ

/-- The nut distribution process -/
def distributeNuts (state : NutState) : NutState :=
  sorry

/-- Predicate to check if at least one nut is eaten during the entire process -/
def atLeastOneNutEaten (initialState : NutState) : Prop :=
  sorry

/-- Predicate to check if not all nuts are eaten during the entire process -/
def notAllNutsEaten (initialState : NutState) : Prop :=
  sorry

/-- Main theorem stating the properties of the nut distribution process -/
theorem nut_distribution_properties {n : ℕ} (h : n > 3) :
  let initialState : NutState := ⟨n, 0, 0⟩
  atLeastOneNutEaten initialState ∧ notAllNutsEaten initialState :=
by
  sorry

end NUMINAMATH_CALUDE_nut_distribution_properties_l1246_124690


namespace NUMINAMATH_CALUDE_min_value_of_z_l1246_124618

theorem min_value_of_z (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) :
  ∃ (z_min : ℝ), z_min = -5 ∧ ∀ (z : ℝ), z = 2*x + Real.sqrt 3 * y → z ≥ z_min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1246_124618


namespace NUMINAMATH_CALUDE_emails_left_in_inbox_l1246_124608

theorem emails_left_in_inbox (initial_emails : ℕ) : 
  initial_emails = 400 → 
  (initial_emails / 2 - (initial_emails / 2 * 40 / 100) : ℕ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_emails_left_in_inbox_l1246_124608


namespace NUMINAMATH_CALUDE_supermarket_prices_theorem_l1246_124625

/-- Represents the prices and discounts at supermarkets -/
structure SupermarketPrices where
  english_machine : ℕ
  backpack : ℕ
  discount_a : ℚ
  voucher_b : ℕ
  voucher_threshold : ℕ

/-- Theorem stating the correct prices and most cost-effective supermarket -/
theorem supermarket_prices_theorem (prices : SupermarketPrices)
    (h1 : prices.english_machine + prices.backpack = 452)
    (h2 : prices.english_machine = 4 * prices.backpack - 8)
    (h3 : prices.discount_a = 75 / 100)
    (h4 : prices.voucher_b = 30)
    (h5 : prices.voucher_threshold = 100)
    (h6 : 400 ≥ prices.english_machine + prices.backpack) :
    prices.english_machine = 360 ∧ 
    prices.backpack = 92 ∧ 
    (prices.english_machine + prices.backpack) * prices.discount_a < 
      prices.english_machine + prices.backpack - prices.voucher_b := by
  sorry


end NUMINAMATH_CALUDE_supermarket_prices_theorem_l1246_124625


namespace NUMINAMATH_CALUDE_wolves_heads_count_l1246_124611

/-- Represents the count of normal wolves -/
def normal_wolves : ℕ := sorry

/-- Represents the count of mutant wolves -/
def mutant_wolves : ℕ := sorry

/-- The total number of heads for all creatures -/
def total_heads : ℕ := 21

/-- The total number of legs for all creatures -/
def total_legs : ℕ := 57

/-- The number of heads a person has -/
def person_heads : ℕ := 1

/-- The number of legs a person has -/
def person_legs : ℕ := 2

/-- The number of heads a normal wolf has -/
def normal_wolf_heads : ℕ := 1

/-- The number of legs a normal wolf has -/
def normal_wolf_legs : ℕ := 4

/-- The number of heads a mutant wolf has -/
def mutant_wolf_heads : ℕ := 2

/-- The number of legs a mutant wolf has -/
def mutant_wolf_legs : ℕ := 3

theorem wolves_heads_count :
  normal_wolves * normal_wolf_heads + mutant_wolves * mutant_wolf_heads = total_heads - person_heads ∧
  normal_wolves * normal_wolf_legs + mutant_wolves * mutant_wolf_legs = total_legs - person_legs := by
  sorry

end NUMINAMATH_CALUDE_wolves_heads_count_l1246_124611


namespace NUMINAMATH_CALUDE_special_rectangle_area_l1246_124606

/-- Represents a rectangle divided into 6 squares with specific properties -/
structure SpecialRectangle where
  /-- The side length of the smallest square -/
  smallest_side : ℝ
  /-- The side length of the D square -/
  d_side : ℝ
  /-- Condition: The smallest square has an area of 4 square centimeters -/
  smallest_area : smallest_side ^ 2 = 4
  /-- Condition: The side lengths increase incrementally by 2 centimeters -/
  incremental_increase : d_side = smallest_side + 6

/-- The theorem stating the area of the special rectangle -/
theorem special_rectangle_area (r : SpecialRectangle) : 
  (2 * r.d_side + (r.d_side + 2)) * (r.d_side + 2 + (r.d_side + 4)) = 572 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_l1246_124606


namespace NUMINAMATH_CALUDE_smallest_divisor_property_solution_set_l1246_124642

def smallest_divisor (n : ℕ) : ℕ :=
  (Nat.factors n).head!

theorem smallest_divisor_property (n : ℕ) : 
  n > 1 → smallest_divisor n > 1 ∧ n % smallest_divisor n = 0 := by sorry

theorem solution_set : 
  {n : ℕ | n + smallest_divisor n = 30} = {25, 27, 28} := by sorry

end NUMINAMATH_CALUDE_smallest_divisor_property_solution_set_l1246_124642


namespace NUMINAMATH_CALUDE_salary_growth_rate_l1246_124631

/-- Proves that the given annual compound interest rate satisfies the salary growth equation -/
theorem salary_growth_rate (initial_salary final_salary total_increase : ℝ) 
  (years : ℕ) (rate : ℝ) 
  (h1 : initial_salary = final_salary - total_increase)
  (h2 : final_salary = 90000)
  (h3 : total_increase = 25000)
  (h4 : years = 3) :
  final_salary = initial_salary * (1 + rate)^years := by
  sorry

end NUMINAMATH_CALUDE_salary_growth_rate_l1246_124631


namespace NUMINAMATH_CALUDE_ellipse_line_segment_no_intersection_l1246_124641

theorem ellipse_line_segment_no_intersection (a : ℝ) :
  a > 0 →
  (∀ x y : ℝ, x^2 + (1/2) * y^2 = a^2 →
    ((2 ≤ x ∧ x ≤ 4 ∧ y = (3-1)/(4-2) * (x-2) + 1) → False)) →
  (0 < a ∧ a < 3 * Real.sqrt 2 / 2) ∨ (a > Real.sqrt 82 / 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_segment_no_intersection_l1246_124641


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1246_124665

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the line l passing through the origin
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the trajectory Γ
def trajectory_Γ (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0 ∧ 3/2 < x ∧ x ≤ 2

-- Define the line m
def line_m (a x y : ℝ) : Prop := y = a * x + 4

theorem circle_line_intersection
  (k : ℝ) -- Slope of line l
  (a : ℝ) -- Parameter for line m
  : 
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l k x1 y1 ∧ line_l k x2 y2) →
  (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3) ∧
  (∀ x y, trajectory_Γ x y ↔ 
    ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ 
    x = (x1 + x2) / 2 * (1 - t) + 2 * t ∧
    y = (y1 + y2) / 2 * (1 - t)) ∧
  ((∃! x y, trajectory_Γ x y ∧ line_m a x y) →
    (a = -15/8 ∨ (-Real.sqrt 3 - 8)/3 < a ∧ a ≤ (Real.sqrt 3 - 8)/3)) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1246_124665


namespace NUMINAMATH_CALUDE_root_equality_implies_b_equals_four_l1246_124670

theorem root_equality_implies_b_equals_four
  (a b c : ℕ)
  (a_gt_one : a > 1)
  (b_gt_one : b > 1)
  (c_gt_one : c > 1)
  (h : ∀ N : ℝ, N ≠ 1 → N^(1/a + 1/(a*b) + 1/(a*b*c) + 1/(a*b*c^2)) = N^(49/60)) :
  b = 4 := by sorry

end NUMINAMATH_CALUDE_root_equality_implies_b_equals_four_l1246_124670


namespace NUMINAMATH_CALUDE_valid_window_exists_l1246_124648

/-- A region in the window --/
structure Region where
  area : ℝ
  sides_equal : Bool

/-- A window configuration --/
structure Window where
  side_length : ℝ
  regions : List Region

/-- Checks if a window configuration is valid --/
def is_valid_window (w : Window) : Prop :=
  w.side_length = 1 ∧
  w.regions.length = 8 ∧
  w.regions.all (fun r => r.area = 1 / 8) ∧
  w.regions.all (fun r => r.sides_equal)

/-- Theorem: There exists a valid window configuration --/
theorem valid_window_exists : ∃ w : Window, is_valid_window w := by
  sorry


end NUMINAMATH_CALUDE_valid_window_exists_l1246_124648


namespace NUMINAMATH_CALUDE_q_divided_by_p_equals_44_l1246_124637

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 12

/-- The number of cards with each number -/
def cards_per_number : ℕ := 4

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The probability that all drawn cards bear the same number -/
noncomputable def p : ℚ := 12 / Nat.choose total_cards cards_drawn

/-- The probability that four cards bear one number and the fifth bears a different number -/
noncomputable def q : ℚ := 528 / Nat.choose total_cards cards_drawn

/-- Theorem stating that the ratio of q to p is 44 -/
theorem q_divided_by_p_equals_44 : q / p = 44 := by sorry

end NUMINAMATH_CALUDE_q_divided_by_p_equals_44_l1246_124637


namespace NUMINAMATH_CALUDE_necklace_ratio_l1246_124686

/-- The number of necklaces Haley, Jason, and Josh have. -/
structure Necklaces where
  haley : ℕ
  jason : ℕ
  josh : ℕ

/-- The conditions given in the problem. -/
def problem_conditions (n : Necklaces) : Prop :=
  n.haley = n.jason + 5 ∧
  n.haley = 25 ∧
  n.haley = n.josh + 15

/-- The theorem stating that under the given conditions, 
    the ratio of Josh's necklaces to Jason's necklaces is 1:2. -/
theorem necklace_ratio (n : Necklaces) 
  (h : problem_conditions n) : n.josh * 2 = n.jason := by
  sorry


end NUMINAMATH_CALUDE_necklace_ratio_l1246_124686


namespace NUMINAMATH_CALUDE_soap_decrease_l1246_124640

theorem soap_decrease (x : ℝ) (h : x > 0) : x * (0.8 ^ 2) ≤ (2/3) * x :=
sorry

end NUMINAMATH_CALUDE_soap_decrease_l1246_124640


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l1246_124699

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | 4 - x^2 ≤ 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

-- Theorem for the union of complements of A and B
theorem union_complement_A_B :
  (Set.univ \ A) ∪ (Set.univ \ B) = {x : ℝ | x < -2 ∨ x > -1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l1246_124699


namespace NUMINAMATH_CALUDE_larger_number_problem_l1246_124671

theorem larger_number_problem (L S : ℕ) (hL : L > S) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1246_124671


namespace NUMINAMATH_CALUDE_julia_played_with_two_kids_on_monday_l1246_124663

/-- The number of kids Julia played with on Monday and Tuesday -/
def total_kids : ℕ := 16

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 14

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := total_kids - tuesday_kids

theorem julia_played_with_two_kids_on_monday :
  monday_kids = 2 := by sorry

end NUMINAMATH_CALUDE_julia_played_with_two_kids_on_monday_l1246_124663


namespace NUMINAMATH_CALUDE_inequality_proof_l1246_124692

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  (a * b / (a^5 + a * b + b^5)) + 
  (b * c / (b^5 + b * c + c^5)) + 
  (c * a / (c^5 + c * a + a^5)) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1246_124692


namespace NUMINAMATH_CALUDE_first_problem_number_l1246_124677

/-- Given a sequence of 48 consecutive integers ending with 125, 
    the first number in the sequence is 78. -/
theorem first_problem_number (last_number : ℕ) (total_problems : ℕ) :
  last_number = 125 → total_problems = 48 → 
  (last_number - total_problems + 1 : ℕ) = 78 := by
  sorry

end NUMINAMATH_CALUDE_first_problem_number_l1246_124677


namespace NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l1246_124609

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones --/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- Calculates the maximum squared radius of a sphere fitting inside the intersecting cones --/
def maxSquaredSphereRadius (ic : IntersectingCones) : ℝ := sorry

/-- Theorem statement --/
theorem max_sphere_in_intersecting_cones :
  let ic : IntersectingCones := {
    cone1 := { baseRadius := 4, height := 10 },
    cone2 := { baseRadius := 4, height := 10 },
    intersectionDistance := 4
  }
  maxSquaredSphereRadius ic = 144 := by sorry

end NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l1246_124609


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l1246_124607

/-- The perimeter of a semicircle with radius 6.3 cm is equal to π * 6.3 + 2 * 6.3 cm -/
theorem semicircle_perimeter :
  let r : ℝ := 6.3
  (π * r + 2 * r) = (π * 6.3 + 2 * 6.3) :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l1246_124607


namespace NUMINAMATH_CALUDE_max_value_fraction_l1246_124672

theorem max_value_fraction (x : ℝ) (h : x > 1) :
  (x^4 - x^2) / (x^6 + 2*x^3 - 1) ≤ 1/5 := by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1246_124672


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1246_124660

/-- Calculates the total wet surface area of a rectangular cistern -/
def cisternWetSurfaceArea (length width depth : Real) : Real :=
  let bottomArea := length * width
  let longerSidesArea := 2 * length * depth
  let shorterSidesArea := 2 * width * depth
  bottomArea + longerSidesArea + shorterSidesArea

/-- Theorem: The wet surface area of a cistern with given dimensions is 68.6 square meters -/
theorem cistern_wet_surface_area :
  cisternWetSurfaceArea 7 5 1.40 = 68.6 := by
  sorry

#eval cisternWetSurfaceArea 7 5 1.40

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1246_124660


namespace NUMINAMATH_CALUDE_evaluate_expression_l1246_124664

theorem evaluate_expression : (3^2)^3 + 2*(3^2 - 2^3) = 731 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1246_124664


namespace NUMINAMATH_CALUDE_abc_inequality_l1246_124616

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) : a + b + c ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1246_124616


namespace NUMINAMATH_CALUDE_line_equivalence_l1246_124619

/-- Given a line expressed in vector form, prove it's equivalent to a specific slope-intercept form -/
theorem line_equivalence (x y : ℝ) : 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-4)) = 0 ↔ y = 2 * x - 10 := by sorry

end NUMINAMATH_CALUDE_line_equivalence_l1246_124619


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1246_124684

/-- Given a cubic equation 5x^3 + 500x + 3005 = 0 with roots a, b, and c,
    prove that (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 -/
theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (5 * a^3 + 500 * a + 3005 = 0) →
  (5 * b^3 + 500 * b + 3005 = 0) →
  (5 * c^3 + 500 * c + 3005 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1246_124684


namespace NUMINAMATH_CALUDE_inverse_trig_sum_equals_pi_l1246_124659

theorem inverse_trig_sum_equals_pi : 
  let arctan_sqrt3 := π / 3
  let arcsin_neg_half := -π / 6
  let arccos_zero := π / 2
  arctan_sqrt3 - arcsin_neg_half + arccos_zero = π := by sorry

end NUMINAMATH_CALUDE_inverse_trig_sum_equals_pi_l1246_124659


namespace NUMINAMATH_CALUDE_quadratic_negative_on_unit_interval_l1246_124673

/-- Given a quadratic function f(x) = ax^2 + bx + c where a > b > c and a + b + c = 0,
    prove that f(x) is negative for all x in the open interval (0,1) -/
theorem quadratic_negative_on_unit_interval 
  (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x ∈ Set.Ioo 0 1, a * x^2 + b * x + c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_negative_on_unit_interval_l1246_124673


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1246_124603

theorem system_solution_ratio (k x y z : ℚ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k * y + 4 * z = 0 →
  2 * x + k * y - 3 * z = 0 →
  x + 2 * y - 4 * z = 0 →
  x * z / (y * y) = 59 / 1024 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1246_124603


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l1246_124653

/-- The distance from the focus of the parabola x = (1/4)y^2 to the asymptote of the hyperbola x^2 - (y^2/3) = 1 is √3/2 -/
theorem distance_focus_to_asymptote :
  let focus : ℝ × ℝ := (1, 0)
  let asymptote (x : ℝ) : ℝ := Real.sqrt 3 * x
  let distance_point_to_line (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ :=
    |f p.1 - p.2| / Real.sqrt (1 + (Real.sqrt 3)^2)
  distance_point_to_line focus asymptote = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l1246_124653


namespace NUMINAMATH_CALUDE_triangle_inequality_l1246_124667

/-- For any triangle ABC and real numbers x, y, and z, 
    x^2 + y^2 + z^2 ≥ 2xy cos C + 2yz cos A + 2zx cos B -/
theorem triangle_inequality (A B C : ℝ) (x y z : ℝ) : 
  x^2 + y^2 + z^2 ≥ 2*x*y*(Real.cos C) + 2*y*z*(Real.cos A) + 2*z*x*(Real.cos B) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1246_124667


namespace NUMINAMATH_CALUDE_intersection_properties_l1246_124610

/-- A parabola intersecting a line and the x-axis -/
structure Intersection where
  a : ℝ
  b : ℝ
  c : ℝ
  k : ℝ
  haNonZero : a ≠ 0
  hIntersectLine : ∀ x, a * x^2 + b * x + c = k * x + 4 → (x = 1 ∨ x = 4)
  hIntersectXAxis : ∀ x, a * x^2 + b * x + c = 0 → (x = 0 ∨ ∃ y, y ≠ 0 ∧ a * y^2 + b * y + c = 0)

/-- The main theorem about the intersection -/
theorem intersection_properties (i : Intersection) :
  (∀ x, k * x + 4 = x + 4) ∧
  (∀ x, a * x^2 + b * x + c = -x^2 + 6 * x) ∧
  (∃ x, x > 0 ∧ -x^2 + 6 * x = 4 ∧
    2 * (1/2 * 1 * 5 + 1/2 * 3 * 3) = 1/2 * 6 * 4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_properties_l1246_124610


namespace NUMINAMATH_CALUDE_sum_of_special_integers_l1246_124661

theorem sum_of_special_integers (a b c d e : ℤ) : 
  (a + 1 = b) ∧ (c + 1 = d) ∧ (d + 1 = e) ∧ (a * b = 272) ∧ (c * d * e = 336) →
  a + b + c + d + e = 54 := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_integers_l1246_124661


namespace NUMINAMATH_CALUDE_tan_theta_value_l1246_124621

theorem tan_theta_value (θ : Real) : 
  (Real.sin (π - θ) + Real.cos (θ - 2*π)) / (Real.sin θ + Real.cos (π + θ)) = 1/2 → 
  Real.tan θ = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1246_124621


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l1246_124678

/-- Given a geometric sequence {a_n} where the first three terms are a-1, a+1, and a+4 respectively,
    prove that the general formula for the nth term is a_n = 4 · (3/2)^(n-1) -/
theorem geometric_sequence_formula (a : ℝ) (a_n : ℕ → ℝ) :
  a_n 1 = a - 1 ∧ a_n 2 = a + 1 ∧ a_n 3 = a + 4 →
  ∀ n : ℕ, a_n n = 4 * (3/2) ^ (n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l1246_124678


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l1246_124617

/-- Given a hyperbola and a parabola with specific properties, prove that n = 12 -/
theorem hyperbola_parabola_intersection (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ (x y : ℝ), x^2/m - y^2/n = 1) →  -- hyperbola equation
  (∃ (e : ℝ), e = 2) →  -- eccentricity is 2
  (∃ (x y : ℝ), y^2 = 4*m*x) →  -- parabola equation
  (∃ (c : ℝ), c = m) →  -- focus of hyperbola coincides with focus of parabola
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l1246_124617


namespace NUMINAMATH_CALUDE_function_f_properties_l1246_124652

/-- A function satisfying the given properties -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y) ∧
  (∀ x, x > 0 → f x > 0)

theorem function_f_properties (f : ℝ → ℝ) (hf : FunctionF f) :
  (f 0 = 0) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) ∧
  (f 1 = 2 → ∃ a, f (2 - a) = 6 ∧ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_function_f_properties_l1246_124652


namespace NUMINAMATH_CALUDE_riverdale_school_theorem_l1246_124698

def riverdale_school (total students_in_band students_in_chorus students_in_band_or_chorus : ℕ) : Prop :=
  students_in_band + students_in_chorus - students_in_band_or_chorus = 30

theorem riverdale_school_theorem :
  riverdale_school 250 90 120 180 := by
  sorry

end NUMINAMATH_CALUDE_riverdale_school_theorem_l1246_124698


namespace NUMINAMATH_CALUDE_no_polynomial_satisfies_condition_l1246_124682

/-- A polynomial function of degree exactly 3 -/
def PolynomialDegree3 (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x + c

/-- The condition that f(x^2) = [f(x)]^2 = f(f(x)) -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x^2) = (f x)^2 ∧ f (x^2) = f (f x)

theorem no_polynomial_satisfies_condition :
  ¬∃ f : ℝ → ℝ, PolynomialDegree3 f ∧ SatisfiesCondition f :=
sorry

end NUMINAMATH_CALUDE_no_polynomial_satisfies_condition_l1246_124682


namespace NUMINAMATH_CALUDE_water_displaced_squared_value_l1246_124613

/-- The square of the volume of water displaced by a fully submerged cube in a cylindrical barrel -/
def water_displaced_squared (cube_side : ℝ) (barrel_radius : ℝ) (barrel_height : ℝ) : ℝ :=
  (cube_side ^ 3) ^ 2

/-- Theorem stating that the square of the volume of water displaced by a fully submerged cube
    with side length 7 feet in a cylindrical barrel with radius 5 feet and height 15 feet is 117649 cubic feet -/
theorem water_displaced_squared_value :
  water_displaced_squared 7 5 15 = 117649 := by
  sorry

end NUMINAMATH_CALUDE_water_displaced_squared_value_l1246_124613


namespace NUMINAMATH_CALUDE_circumcircle_passes_through_O_l1246_124635

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric operations
variable (parallelogram : Point → Point → Point → Point → Prop)
variable (intersectionPoint : Point → Point → Point → Point → Point)
variable (circumcircle : Point → Point → Point → Circle)
variable (onCircle : Point → Circle → Prop)
variable (intersectionCircles : Circle → Circle → Point)

-- State the theorem
theorem circumcircle_passes_through_O 
  (A B C D P Q R M N O : Point) :
  parallelogram A B C D →
  O = intersectionPoint A C B D →
  Q = intersectionCircles (circumcircle P O B) (circumcircle O A D) →
  R = intersectionCircles (circumcircle P O C) (circumcircle O A D) →
  Q ≠ O →
  R ≠ O →
  parallelogram P Q A M →
  parallelogram P R D N →
  onCircle O (circumcircle M N P) :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_passes_through_O_l1246_124635


namespace NUMINAMATH_CALUDE_point_on_curve_in_third_quadrant_l1246_124623

theorem point_on_curve_in_third_quadrant :
  ∀ a : ℝ, a < 0 → 3 * a^2 + (2 * a)^2 = 28 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_in_third_quadrant_l1246_124623


namespace NUMINAMATH_CALUDE_only_rectangle_area_certain_l1246_124644

-- Define the events
inductive Event
  | WaterFreeze : Event
  | ExamScore : Event
  | CoinToss : Event
  | RectangleArea : Event

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.WaterFreeze => False
  | Event.ExamScore => False
  | Event.CoinToss => False
  | Event.RectangleArea => True

-- Theorem stating that only RectangleArea is a certain event
theorem only_rectangle_area_certain :
  ∀ (e : Event), is_certain e ↔ e = Event.RectangleArea :=
by sorry

end NUMINAMATH_CALUDE_only_rectangle_area_certain_l1246_124644


namespace NUMINAMATH_CALUDE_negative_thirty_two_to_five_thirds_l1246_124655

theorem negative_thirty_two_to_five_thirds :
  (-32 : ℝ) ^ (5/3) = -256 * (2 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_negative_thirty_two_to_five_thirds_l1246_124655


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l1246_124612

/-- A quadratic function passing through the origin with a given derivative -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ,
    (∀ x, f x = a * x^2 + b * x) ∧
    (f 0 = 0) ∧
    (∀ x, deriv f x = 3 * x - 1/2)

/-- The theorem stating the unique form of the quadratic function -/
theorem quadratic_function_unique (f : ℝ → ℝ) (hf : quadratic_function f) :
  ∀ x, f x = 3/2 * x^2 - 1/2 * x :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l1246_124612


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l1246_124630

theorem cube_plus_reciprocal_cube (m : ℝ) (h : m + 1/m = 10) :
  m^3 + 1/m^3 + 6 = 976 := by sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l1246_124630


namespace NUMINAMATH_CALUDE_eighth_term_is_84_l1246_124662

/-- The n-th term of the sequence -/
def S (n : ℕ) : ℚ := (3 * n * (n - 1)) / 2

/-- Theorem: The 8th term of the sequence is 84 -/
theorem eighth_term_is_84 : S 8 = 84 := by sorry

end NUMINAMATH_CALUDE_eighth_term_is_84_l1246_124662


namespace NUMINAMATH_CALUDE_cookies_taken_in_seven_days_l1246_124668

/-- Represents the number of cookies Jessica takes each day -/
def jessica_daily_cookies : ℝ := 1.5

/-- Represents the number of cookies Sarah takes each day -/
def sarah_daily_cookies : ℝ := 3 * jessica_daily_cookies

/-- Represents the number of cookies Paul takes each day -/
def paul_daily_cookies : ℝ := 2 * sarah_daily_cookies

/-- Represents the total number of cookies in the jar initially -/
def initial_cookies : ℕ := 200

/-- Represents the number of cookies left after 10 days -/
def cookies_left : ℕ := 50

/-- Represents the number of days they took cookies -/
def total_days : ℕ := 10

/-- Represents the number of days we want to calculate for -/
def target_days : ℕ := 7

theorem cookies_taken_in_seven_days :
  (jessica_daily_cookies + sarah_daily_cookies + paul_daily_cookies) * target_days = 105 :=
by sorry

end NUMINAMATH_CALUDE_cookies_taken_in_seven_days_l1246_124668


namespace NUMINAMATH_CALUDE_zero_point_implies_a_range_l1246_124604

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 2

-- Define the theorem
theorem zero_point_implies_a_range (a : ℝ) : 
  (∃ x ∈ Set.Icc (-2) 1, g a x = 0) → 
  (a < 2) → 
  a ∈ Set.Icc (-3/2) 2 := by
sorry

end NUMINAMATH_CALUDE_zero_point_implies_a_range_l1246_124604


namespace NUMINAMATH_CALUDE_correlation_of_product_l1246_124658

-- Define a random function type
def RandomFunction := ℝ → ℝ

-- Define the expectation operator
noncomputable def expectation (X : RandomFunction) : ℝ := sorry

-- Define the correlation function
noncomputable def correlation (X Y : RandomFunction) : ℝ := sorry

-- Define what it means for a random function to be centered
def is_centered (X : RandomFunction) : Prop :=
  expectation X = 0

-- Define what it means for two random functions to be uncorrelated
def are_uncorrelated (X Y : RandomFunction) : Prop :=
  expectation (fun t => X t * Y t) = expectation X * expectation Y

-- State the theorem
theorem correlation_of_product (X Y : RandomFunction) 
  (h1 : is_centered X) (h2 : is_centered Y) (h3 : are_uncorrelated X Y) :
  correlation (fun t => X t * Y t) (fun t => X t * Y t) = 
  correlation X X * correlation Y Y := by sorry

end NUMINAMATH_CALUDE_correlation_of_product_l1246_124658


namespace NUMINAMATH_CALUDE_max_objective_value_l1246_124681

/-- The system of inequalities and objective function --/
def LinearProgram (x y : ℝ) : Prop :=
  x + 7 * y ≤ 32 ∧
  2 * x + 5 * y ≤ 42 ∧
  3 * x + 4 * y ≤ 62 ∧
  2 * x + y = 34 ∧
  x ≥ 0 ∧ y ≥ 0

/-- The objective function --/
def ObjectiveFunction (x y : ℝ) : ℝ :=
  3 * x + 8 * y

/-- The theorem stating the maximum value of the objective function --/
theorem max_objective_value :
  ∃ (x y : ℝ), LinearProgram x y ∧
  ∀ (x' y' : ℝ), LinearProgram x' y' →
  ObjectiveFunction x y ≥ ObjectiveFunction x' y' ∧
  ObjectiveFunction x y = 64 :=
sorry

end NUMINAMATH_CALUDE_max_objective_value_l1246_124681


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_2_range_of_a_l1246_124696

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3| - |x - a|

-- Theorem for the first part of the problem
theorem solution_set_for_a_equals_2 :
  {x : ℝ | f 2 x ≤ -1/2} = {x : ℝ | x ≥ 11/4} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a :
  {a : ℝ | ∀ x, f a x ≥ a} = Set.Iic (3/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_2_range_of_a_l1246_124696


namespace NUMINAMATH_CALUDE_power_calculation_l1246_124694

theorem power_calculation : (4^4 / 4^3) * 2^8 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1246_124694


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_50_l1246_124634

theorem largest_four_digit_divisible_by_50 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 50 = 0 → n ≤ 9950 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_50_l1246_124634


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l1246_124693

theorem lcm_factor_proof (A B : ℕ) : 
  A > 0 ∧ B > 0 ∧ A ≥ B ∧ Nat.gcd A B = 30 ∧ A = 450 → 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ Nat.lcm A B = 30 * x * y ∧ (x = 15 ∨ y = 15) :=
by
  sorry

#check lcm_factor_proof

end NUMINAMATH_CALUDE_lcm_factor_proof_l1246_124693


namespace NUMINAMATH_CALUDE_other_number_value_l1246_124614

theorem other_number_value (x y : ℝ) : 
  y = 125 * 1.1 →
  x = y * 0.9 →
  x = 123.75 →
  y = 137.5 := by
sorry

end NUMINAMATH_CALUDE_other_number_value_l1246_124614


namespace NUMINAMATH_CALUDE_twenty_percent_greater_than_twelve_l1246_124651

theorem twenty_percent_greater_than_twelve (x : ℝ) : 
  x = 12 * (1 + 0.2) → x = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_greater_than_twelve_l1246_124651


namespace NUMINAMATH_CALUDE_congruence_solution_l1246_124628

theorem congruence_solution : 
  {x : ℤ | 20 ≤ x ∧ x ≤ 50 ∧ (6 * x + 5) % 10 = (-19) % 10} = 
  {21, 26, 31, 36, 41, 46} := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1246_124628


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1246_124650

theorem min_value_of_expression (x y : ℝ) 
  (hx : |x| ≤ 1) (hy : |y| ≤ 1) : 
  ∃ (min_val : ℝ), min_val = 3 ∧ 
  ∀ (a b : ℝ), |a| ≤ 1 → |b| ≤ 1 → |b + 1| + |2*b - a - 4| ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1246_124650


namespace NUMINAMATH_CALUDE_other_candidate_votes_l1246_124676

-- Define the total number of votes
def total_votes : ℕ := 7500

-- Define the percentage of invalid votes
def invalid_vote_percentage : ℚ := 20 / 100

-- Define the percentage of votes for the winning candidate
def winning_candidate_percentage : ℚ := 55 / 100

-- Theorem to prove
theorem other_candidate_votes :
  (total_votes * (1 - invalid_vote_percentage) * (1 - winning_candidate_percentage)).floor = 2700 :=
by sorry

end NUMINAMATH_CALUDE_other_candidate_votes_l1246_124676


namespace NUMINAMATH_CALUDE_shop_owner_gain_l1246_124605

/-- Represents the problem of calculating the gain in terms of cloth meters for a shop owner. -/
theorem shop_owner_gain (total_meters : ℝ) (gain_percentage : ℝ) (gain_meters : ℝ) : 
  total_meters = 30 ∧ 
  gain_percentage = 50 / 100 → 
  gain_meters = 10 := by
  sorry


end NUMINAMATH_CALUDE_shop_owner_gain_l1246_124605


namespace NUMINAMATH_CALUDE_cube_red_faces_ratio_l1246_124656

/-- Represents a cube with side length n -/
structure Cube where
  n : ℕ

/-- Calculates the number of red faces on the original cube -/
def redFaces (c : Cube) : ℕ := 6 * c.n^2

/-- Calculates the total number of faces of all small cubes -/
def totalFaces (c : Cube) : ℕ := 6 * c.n^3

/-- Theorem: The side length of the cube is 3 if and only if
    exactly one-third of the faces of the small cubes are red -/
theorem cube_red_faces_ratio (c : Cube) : 
  c.n = 3 ↔ 3 * redFaces c = totalFaces c := by
  sorry


end NUMINAMATH_CALUDE_cube_red_faces_ratio_l1246_124656


namespace NUMINAMATH_CALUDE_karen_grooms_six_rottweilers_l1246_124687

/-- Represents the time taken to groom different dog breeds and the total grooming time -/
structure GroomingInfo where
  rottweilerTime : ℕ
  borderCollieTime : ℕ
  chihuahuaTime : ℕ
  totalTime : ℕ
  borderCollieCount : ℕ
  chihuahuaCount : ℕ

/-- Calculates the number of Rottweilers groomed given the grooming information -/
def calculateRottweilers (info : GroomingInfo) : ℕ :=
  (info.totalTime - info.borderCollieTime * info.borderCollieCount - info.chihuahuaTime * info.chihuahuaCount) / info.rottweilerTime

/-- Theorem stating that Karen grooms 6 Rottweilers given the problem conditions -/
theorem karen_grooms_six_rottweilers (info : GroomingInfo)
  (h1 : info.rottweilerTime = 20)
  (h2 : info.borderCollieTime = 10)
  (h3 : info.chihuahuaTime = 45)
  (h4 : info.totalTime = 255)
  (h5 : info.borderCollieCount = 9)
  (h6 : info.chihuahuaCount = 1) :
  calculateRottweilers info = 6 := by
  sorry


end NUMINAMATH_CALUDE_karen_grooms_six_rottweilers_l1246_124687


namespace NUMINAMATH_CALUDE_range_of_m_l1246_124627

theorem range_of_m (x m : ℝ) : 
  (∀ x, (x ≥ -2 ∧ x ≤ 10) → (x + m - 1) * (x - m - 1) ≤ 0) →
  m > 0 →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1246_124627


namespace NUMINAMATH_CALUDE_gretchen_objects_l1246_124600

/-- The number of objects Gretchen can carry per trip -/
def objects_per_trip : ℕ := 3

/-- The number of trips Gretchen took -/
def number_of_trips : ℕ := 6

/-- The total number of objects Gretchen found underwater -/
def total_objects : ℕ := objects_per_trip * number_of_trips

theorem gretchen_objects : total_objects = 18 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_objects_l1246_124600


namespace NUMINAMATH_CALUDE_complex_number_solution_l1246_124697

theorem complex_number_solution (z : ℂ) : (Complex.I * z = 1) → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_solution_l1246_124697


namespace NUMINAMATH_CALUDE_tarun_departure_time_l1246_124679

theorem tarun_departure_time 
  (total_work : ℝ) 
  (combined_rate : ℝ) 
  (arun_rate : ℝ) 
  (remaining_days : ℝ) :
  combined_rate = total_work / 10 →
  arun_rate = total_work / 30 →
  remaining_days = 18 →
  ∃ (x : ℝ), x * combined_rate + remaining_days * arun_rate = total_work ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_tarun_departure_time_l1246_124679


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1246_124683

/-- A hyperbola centered at the origin -/
structure Hyperbola where
  center : ℝ × ℝ
  asymptotes : Set (ℝ → ℝ)

/-- A circle with equation (x-h)^2 + (y-k)^2 = r^2 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of eccentricity for a hyperbola -/
def eccentricity (h : Hyperbola) : Set ℝ := sorry

/-- Definition of a line being tangent to a circle -/
def is_tangent (l : ℝ → ℝ) (c : Circle) : Prop := sorry

theorem hyperbola_eccentricity (h : Hyperbola) (c : Circle) :
  h.center = (0, 0) →
  c.center = (2, 0) →
  c.radius = Real.sqrt 3 →
  (∀ a ∈ h.asymptotes, is_tangent a c) →
  eccentricity h = {2, 2 * Real.sqrt 3 / 3} := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1246_124683


namespace NUMINAMATH_CALUDE_median_intersection_l1246_124657

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a median
def median (t : Triangle) (v : Point) (m : Point) : Prop :=
  (v = t.A ∨ v = t.B ∨ v = t.C) ∧
  (m = ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2) ∨
   m = ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2) ∨
   m = ((t.C.1 + t.A.1) / 2, (t.C.2 + t.A.2) / 2))

-- Define the ratio of division
def divides_in_ratio (p : Point) (v : Point) (m : Point) : Prop :=
  let d1 := ((p.1 - v.1)^2 + (p.2 - v.2)^2).sqrt
  let d2 := ((m.1 - p.1)^2 + (m.2 - p.2)^2).sqrt
  d1 / d2 = 2 / 1

-- Theorem statement
theorem median_intersection (t : Triangle) : 
  ∃ (O : Point), 
    (∀ (v m : Point), median t v m → divides_in_ratio O v m) ∧
    (∀ (v1 m1 v2 m2 : Point), 
      median t v1 m1 → median t v2 m2 → 
      ∃ (k : ℝ), O = (k * v1.1 + (1 - k) * m1.1, k * v1.2 + (1 - k) * m1.2) ∧
                 O = (k * v2.1 + (1 - k) * m2.1, k * v2.2 + (1 - k) * m2.2)) :=
by
  sorry

end NUMINAMATH_CALUDE_median_intersection_l1246_124657


namespace NUMINAMATH_CALUDE_even_function_max_symmetry_l1246_124647

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f has a maximum value on an interval [a, b] if there exists
    a point c in [a, b] such that f(c) ≥ f(x) for all x in [a, b] -/
def HasMaximumOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c

/-- If f is an even function and has a maximum value on [1, 7],
    then it also has a maximum value on [-7, -1] -/
theorem even_function_max_symmetry (f : ℝ → ℝ) :
  EvenFunction f → HasMaximumOn f 1 7 → HasMaximumOn f (-7) (-1) :=
by
  sorry

end NUMINAMATH_CALUDE_even_function_max_symmetry_l1246_124647


namespace NUMINAMATH_CALUDE_five_from_six_circular_seating_l1246_124695

/-- The number of ways to seat 5 people from a group of 6 around a circular table -/
def circular_seating_arrangements (total_people : ℕ) (seated_people : ℕ) : ℕ :=
  (total_people.choose seated_people) * (seated_people - 1).factorial

/-- Theorem stating that the number of ways to seat 5 people from a group of 6 around a circular table is 144 -/
theorem five_from_six_circular_seating :
  circular_seating_arrangements 6 5 = 144 := by
  sorry

end NUMINAMATH_CALUDE_five_from_six_circular_seating_l1246_124695


namespace NUMINAMATH_CALUDE_regular_polygon_area_l1246_124689

theorem regular_polygon_area (n : ℕ) (R : ℝ) (h : n > 0) :
  (n * R^2 / 2) * (Real.sin (2 * Real.pi / n) + Real.cos (Real.pi / n)) = 4 * R^2 →
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_area_l1246_124689


namespace NUMINAMATH_CALUDE_congruence_solution_l1246_124626

theorem congruence_solution (p q : Nat) (n : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Odd p ∧ Odd q ∧ n > 0 ∧
  (q^(n+2) : Nat) % (p^n) = (3^(n+2) : Nat) % (p^n) ∧
  (p^(n+2) : Nat) % (q^n) = (3^(n+2) : Nat) % (q^n) →
  p = 3 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_congruence_solution_l1246_124626


namespace NUMINAMATH_CALUDE_graph_reflection_l1246_124633

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the reflection across y-axis
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Statement: The graph of y = g(-x) is the reflection of y = g(x) across the y-axis
theorem graph_reflection (x : ℝ) : 
  reflect_y (x, g x) = (-x, g (-x)) := by sorry

end NUMINAMATH_CALUDE_graph_reflection_l1246_124633


namespace NUMINAMATH_CALUDE_interval_intersection_l1246_124629

theorem interval_intersection : ∀ x : ℝ, 
  (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) := by sorry

end NUMINAMATH_CALUDE_interval_intersection_l1246_124629


namespace NUMINAMATH_CALUDE_incorrect_inequality_l1246_124615

-- Define the conditions
variable (a b : ℝ)
variable (h1 : b < a)
variable (h2 : a < 0)

-- Define the theorem
theorem incorrect_inequality :
  ¬((1/2:ℝ)^b < (1/2:ℝ)^a) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l1246_124615


namespace NUMINAMATH_CALUDE_factorization_of_2a2_minus_8b2_l1246_124649

theorem factorization_of_2a2_minus_8b2 (a b : ℝ) :
  2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a2_minus_8b2_l1246_124649


namespace NUMINAMATH_CALUDE_triangle_properties_area_condition1_area_condition2_l1246_124632

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)

-- Define the main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.B ≠ π/2)
  (h2 : Real.cos (2 * t.B) = Real.sqrt 3 * Real.cos t.B - 1) :
  t.B = π/6 ∧ 
  ((Real.sin t.A = Real.sqrt 3 * Real.sin t.C ∧ t.b = 2 → t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3) ∨
   (2 * t.b = 3 * t.a ∧ t.b * Real.sin t.A = 1 → t.a * t.c * Real.sin t.B / 2 = (Real.sqrt 3 + 2 * Real.sqrt 2) / 2)) :=
by sorry

-- Define additional theorems for each condition
theorem area_condition1 (t : Triangle) 
  (h1 : t.B = π/6)
  (h2 : Real.sin t.A = Real.sqrt 3 * Real.sin t.C)
  (h3 : t.b = 2) :
  t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3 :=
by sorry

theorem area_condition2 (t : Triangle)
  (h1 : t.B = π/6)
  (h2 : 2 * t.b = 3 * t.a)
  (h3 : t.b * Real.sin t.A = 1) :
  t.a * t.c * Real.sin t.B / 2 = (Real.sqrt 3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_area_condition1_area_condition2_l1246_124632


namespace NUMINAMATH_CALUDE_student_line_count_l1246_124643

/-- Given a line of students, if a student is 7th from the left and 5th from the right,
    then the total number of students in the line is 11. -/
theorem student_line_count (n : ℕ) 
  (left_position : ℕ) 
  (right_position : ℕ) 
  (h1 : left_position = 7) 
  (h2 : right_position = 5) : 
  n = left_position + right_position - 1 := by
  sorry

end NUMINAMATH_CALUDE_student_line_count_l1246_124643


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_fifteen_zeros_l1246_124601

/-- The number of trailing zeros in n! when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

theorem fifteen_factorial_base_fifteen_zeros :
  trailingZeros 15 15 = 3 :=
sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_fifteen_zeros_l1246_124601


namespace NUMINAMATH_CALUDE_parabola_and_range_l1246_124645

-- Define the parabola G
def G (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 4)

-- Define point A
def point_A : ℝ × ℝ := (-4, 0)

-- Define the condition for points B and C
def intersect_points (p k x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  G p x₁ y₁ ∧ G p x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the condition AC = 1/4 * AB when k = 1/2
def vector_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = 4*y₂

-- Define the y-intercept of the perpendicular bisector
def perpendicular_bisector_y_intercept (k : ℝ) : ℝ :=
  2*(k + 1)^2

-- Theorem statement
theorem parabola_and_range :
  ∀ p k x₁ y₁ x₂ y₂,
    G p (-4) 0 →
    intersect_points p k x₁ y₁ x₂ y₂ →
    k = 1/2 →
    vector_condition x₁ y₁ x₂ y₂ →
    (p = 2 ∧ 
     ∀ b, b > 2 ↔ ∃ k', perpendicular_bisector_y_intercept k' = b) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_range_l1246_124645


namespace NUMINAMATH_CALUDE_gcd_2703_1113_l1246_124620

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2703_1113_l1246_124620


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_length_l1246_124680

/-- Given a hyperbola and a circle with specific properties, prove the length of the chord formed by their intersection. -/
theorem hyperbola_circle_intersection_length :
  ∀ (a b : ℝ) (A B : ℝ × ℝ),
  a > 0 →
  b > 0 →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (∃ (t : ℝ), y = 2 * x * t ∨ y = -2 * x * t)) →  -- Asymptotes condition
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → Real.sqrt (1 + b^2 / a^2) = Real.sqrt 5) →  -- Eccentricity condition
  (∃ (t : ℝ), (A.1 - 2)^2 + (A.2 - 3)^2 = 1 ∧ 
              (B.1 - 2)^2 + (B.2 - 3)^2 = 1 ∧ 
              (A.2 = 2 * A.1 * t ∨ A.2 = -2 * A.1 * t) ∧ 
              (B.2 = 2 * B.1 * t ∨ B.2 = -2 * B.1 * t)) →  -- Intersection condition
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 := by
sorry


end NUMINAMATH_CALUDE_hyperbola_circle_intersection_length_l1246_124680


namespace NUMINAMATH_CALUDE_factor_theorem_application_l1246_124669

theorem factor_theorem_application (c : ℚ) : 
  (∀ x : ℚ, (x + 5) ∣ (2*c*x^3 + 14*x^2 - 6*c*x + 25)) → c = 75/44 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l1246_124669


namespace NUMINAMATH_CALUDE_embankment_construction_time_l1246_124602

theorem embankment_construction_time 
  (workers_initial : ℕ) 
  (days_initial : ℕ) 
  (embankments : ℕ) 
  (workers_new : ℕ) 
  (h1 : workers_initial = 75)
  (h2 : days_initial = 4)
  (h3 : embankments = 2)
  (h4 : workers_new = 60) :
  ∃ (days_new : ℕ), 
    (workers_initial * days_initial = workers_new * days_new) ∧ 
    (days_new = 5) := by
  sorry

end NUMINAMATH_CALUDE_embankment_construction_time_l1246_124602


namespace NUMINAMATH_CALUDE_incorrect_permutations_of_error_l1246_124622

def word : String := "error"

theorem incorrect_permutations_of_error (n : ℕ) :
  (n = word.length) →
  (n.choose 2 * 1 - 1 = 19) :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_permutations_of_error_l1246_124622


namespace NUMINAMATH_CALUDE_annie_candy_cost_l1246_124654

/-- Calculates the total cost of candies Annie bought for her class -/
theorem annie_candy_cost (class_size : ℕ) (candies_per_classmate : ℕ) (leftover_candies : ℕ) (candy_cost : ℚ) : 
  class_size = 35 → 
  candies_per_classmate = 2 → 
  leftover_candies = 12 → 
  candy_cost = 1/10 →
  (class_size * candies_per_classmate + leftover_candies) * candy_cost = 82/10 := by
  sorry

end NUMINAMATH_CALUDE_annie_candy_cost_l1246_124654


namespace NUMINAMATH_CALUDE_instrument_probability_l1246_124638

theorem instrument_probability (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = 3/5 →
  two_or_more = 96 →
  (((at_least_one * total) - two_or_more) / total : ℚ) = 12/25 := by
sorry

end NUMINAMATH_CALUDE_instrument_probability_l1246_124638


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1246_124639

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 6*x^2 + 9 + 11*x - 12*x^2 - 15 + 17*x + 18*x^2 - 2*x^3 = -2*x^3 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1246_124639


namespace NUMINAMATH_CALUDE_pet_store_combinations_l1246_124646

/-- The number of puppies available -/
def num_puppies : ℕ := 10

/-- The number of kittens available -/
def num_kittens : ℕ := 7

/-- The number of hamsters available -/
def num_hamsters : ℕ := 9

/-- The number of birds available -/
def num_birds : ℕ := 5

/-- The number of people buying pets -/
def num_people : ℕ := 4

/-- The number of ways to select one pet of each type and assign them to four different people -/
def num_ways : ℕ := num_puppies * num_kittens * num_hamsters * num_birds * Nat.factorial num_people

theorem pet_store_combinations : num_ways = 75600 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l1246_124646


namespace NUMINAMATH_CALUDE_parallel_line_plane_l1246_124685

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Define the theorem
theorem parallel_line_plane 
  (m n : Line) (α : Plane)
  (distinct_lines : m ≠ n)
  (m_parallel_n : parallel m n)
  (m_parallel_α : parallel_plane m α)
  (n_not_in_α : ¬ contained_in n α) :
  parallel_plane n α :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_l1246_124685


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l1246_124624

theorem circle_tangent_to_x_axis (x y : ℝ) :
  let center : ℝ × ℝ := (-3, 4)
  let equation := (x + 3)^2 + (y - 4)^2 = 16
  let is_tangent_to_x_axis := ∃ (x₀ : ℝ), (x₀ + 3)^2 + 4^2 = 16 ∧ ∀ (y : ℝ), y ≠ 0 → (x₀ + 3)^2 + (y - 4)^2 > 16
  equation ∧ is_tangent_to_x_axis :=
by
  sorry


end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l1246_124624


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1246_124688

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality_solution 
  (b c : ℝ) 
  (h1 : quadratic_function b c (-1) = 0)
  (h2 : quadratic_function b c 2 = 0) :
  {x : ℝ | quadratic_function b c x < 4} = Set.Ioo (-2) 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1246_124688


namespace NUMINAMATH_CALUDE_total_time_is_80_minutes_l1246_124691

/-- The total time students spend outside of class -/
def total_time_outside_class (recess1 recess2 lunch recess3 : ℕ) : ℕ :=
  recess1 + recess2 + lunch + recess3

/-- Theorem stating that the total time outside class is 80 minutes -/
theorem total_time_is_80_minutes :
  total_time_outside_class 15 15 30 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_80_minutes_l1246_124691


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_a_value_l1246_124675

/-- Trapezoid inscribed in a parabola -/
structure InscribedTrapezoid where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_a_gt_b : a > b
  h_sides_equal : 2*a + 2*b = 3/4 + Real.sqrt ((a - b)^2 + (a^2 - b^2)^2)
  h_ab : Real.sqrt ((a - b)^2 + (a^2 - b^2)^2) = 3/4

theorem inscribed_trapezoid_a_value (t : InscribedTrapezoid) : t.a = 27/40 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_a_value_l1246_124675


namespace NUMINAMATH_CALUDE_apple_orange_probability_l1246_124636

theorem apple_orange_probability (n : ℕ) : 
  (n : ℚ) / (n + 3 : ℚ) = 2 / 3 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_orange_probability_l1246_124636


namespace NUMINAMATH_CALUDE_nabla_computation_l1246_124666

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_computation : (nabla (nabla 2 3) 4) = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_computation_l1246_124666
