import Mathlib

namespace NUMINAMATH_CALUDE_blue_pill_cost_proof_l1876_187652

/-- The cost of a blue pill in dollars -/
def blue_pill_cost : ℝ := 23.5

/-- The cost of a red pill in dollars -/
def red_pill_cost : ℝ := blue_pill_cost - 2

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 945

theorem blue_pill_cost_proof :
  blue_pill_cost = 23.5 ∧
  red_pill_cost = blue_pill_cost - 2 ∧
  treatment_days = 21 ∧
  total_cost = 945 ∧
  treatment_days * (blue_pill_cost + red_pill_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_blue_pill_cost_proof_l1876_187652


namespace NUMINAMATH_CALUDE_train_platform_time_l1876_187681

/-- Given a train of length 1200 meters that takes 120 seconds to pass a tree,
    this theorem proves that the time required for the train to pass a platform
    of length 800 meters is 200 seconds. -/
theorem train_platform_time (train_length : ℝ) (tree_pass_time : ℝ) (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : tree_pass_time = 120)
  (h3 : platform_length = 800) :
  (train_length + platform_length) / (train_length / tree_pass_time) = 200 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_time_l1876_187681


namespace NUMINAMATH_CALUDE_b_share_is_108_l1876_187694

/-- Represents the share ratio of partners A, B, and C -/
structure ShareRatio where
  a : Rat
  b : Rat
  c : Rat

/-- Represents the capital contribution of partners over time -/
structure CapitalContribution where
  a : Rat
  b : Rat
  c : Rat

def initial_ratio : ShareRatio :=
  { a := 1/2, b := 1/3, c := 1/4 }

def total_profit : ℚ := 378

def months_before_withdrawal : ℕ := 2
def total_months : ℕ := 12

def capital_contribution (r : ShareRatio) : CapitalContribution :=
  { a := r.a * months_before_withdrawal + (r.a / 2) * (total_months - months_before_withdrawal),
    b := r.b * total_months,
    c := r.c * total_months }

theorem b_share_is_108 (r : ShareRatio) (cc : CapitalContribution) :
  r = initial_ratio →
  cc = capital_contribution r →
  (cc.b / (cc.a + cc.b + cc.c)) * total_profit = 108 :=
by sorry

end NUMINAMATH_CALUDE_b_share_is_108_l1876_187694


namespace NUMINAMATH_CALUDE_johns_final_push_time_l1876_187607

/-- The time of John's final push in a race, given specific conditions. -/
theorem johns_final_push_time (john_speed steve_speed initial_distance final_distance : ℝ) 
  (h1 : john_speed = 4.2)
  (h2 : steve_speed = 3.7)
  (h3 : initial_distance = 14)
  (h4 : final_distance = 2)
  : (initial_distance + final_distance) / john_speed = 16 / 4.2 := by
  sorry

#eval (14 + 2) / 4.2

end NUMINAMATH_CALUDE_johns_final_push_time_l1876_187607


namespace NUMINAMATH_CALUDE_profit_percentage_specific_l1876_187617

/-- The profit percentage after markup and discount -/
def profit_percentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem: Given a 60% markup and a 25% discount, the profit percentage is 20% -/
theorem profit_percentage_specific : profit_percentage 0.6 0.25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_specific_l1876_187617


namespace NUMINAMATH_CALUDE_annie_cookies_total_l1876_187680

/-- The number of cookies Annie ate on Monday -/
def monday_cookies : ℕ := 5

/-- The number of cookies Annie ate on Tuesday -/
def tuesday_cookies : ℕ := 2 * monday_cookies

/-- The number of cookies Annie ate on Wednesday -/
def wednesday_cookies : ℕ := tuesday_cookies + (tuesday_cookies * 40 / 100)

/-- The total number of cookies Annie ate over three days -/
def total_cookies : ℕ := monday_cookies + tuesday_cookies + wednesday_cookies

/-- Theorem stating that Annie ate 29 cookies in total over three days -/
theorem annie_cookies_total : total_cookies = 29 := by
  sorry

end NUMINAMATH_CALUDE_annie_cookies_total_l1876_187680


namespace NUMINAMATH_CALUDE_set_a_contains_one_l1876_187690

theorem set_a_contains_one (a : ℝ) : 
  let A : Set ℝ := {a, a^2}
  1 ∈ A → a = -1 := by sorry

end NUMINAMATH_CALUDE_set_a_contains_one_l1876_187690


namespace NUMINAMATH_CALUDE_transform_invariant_l1876_187673

def initial_point : ℝ × ℝ × ℝ := (1, 1, 1)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p
  |> rotate_z_90
  |> reflect_yz
  |> reflect_xz
  |> rotate_z_90
  |> reflect_xy
  |> reflect_xz

theorem transform_invariant : transform initial_point = initial_point := by
  sorry

end NUMINAMATH_CALUDE_transform_invariant_l1876_187673


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1876_187636

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : 0 < d ∧ d < 1) 
  (h3 : ∀ k : ℤ, a 5 ≠ k * (π / 2)) 
  (h4 : Real.sin (a 3) ^ 2 + 2 * Real.sin (a 5) * Real.cos (a 5) = Real.sin (a 7) ^ 2) : 
  d = π / 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1876_187636


namespace NUMINAMATH_CALUDE_bottle_caps_per_visit_l1876_187679

def store_visits : ℕ := 5
def total_bottle_caps : ℕ := 25

theorem bottle_caps_per_visit :
  total_bottle_caps / store_visits = 5 :=
by sorry

end NUMINAMATH_CALUDE_bottle_caps_per_visit_l1876_187679


namespace NUMINAMATH_CALUDE_min_value_problem_l1876_187625

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  (y/x) + (1/y) ≥ 4 ∧ ((y/x) + (1/y) = 4 ↔ x = 1/3 ∧ y = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l1876_187625


namespace NUMINAMATH_CALUDE_square_area_with_side_5_l1876_187623

theorem square_area_with_side_5 :
  let side_length : ℝ := 5
  let area : ℝ := side_length * side_length
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_with_side_5_l1876_187623


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l1876_187624

def arrangement_count : ℕ := 
  (Finset.range 3).sum (λ i =>
    Nat.choose 4 i * Nat.choose 6 (i + 2) * Nat.choose 5 i)

theorem correct_arrangement_count : arrangement_count = 1315 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l1876_187624


namespace NUMINAMATH_CALUDE_base_difference_in_right_trapezoid_l1876_187699

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- The measure of the largest angle in degrees -/
  largest_angle : ℝ
  /-- The length of the shorter leg -/
  shorter_leg : ℝ
  /-- The length of the longer base -/
  longer_base : ℝ
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- Condition that the largest angle is 135° -/
  largest_angle_eq : largest_angle = 135
  /-- Condition that the shorter leg is 18 -/
  shorter_leg_eq : shorter_leg = 18
  /-- Condition that it's a right trapezoid (one angle is 90°) -/
  is_right : True

/-- Theorem stating the difference between bases in a right trapezoid with specific properties -/
theorem base_difference_in_right_trapezoid (t : RightTrapezoid) : 
  t.longer_base - t.shorter_base = 18 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_in_right_trapezoid_l1876_187699


namespace NUMINAMATH_CALUDE_digits_of_2_pow_120_l1876_187640

theorem digits_of_2_pow_120 (h : ∃ n : ℕ, 10^60 ≤ 2^200 ∧ 2^200 < 10^61) :
  ∃ m : ℕ, 10^36 ≤ 2^120 ∧ 2^120 < 10^37 :=
sorry

end NUMINAMATH_CALUDE_digits_of_2_pow_120_l1876_187640


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1876_187658

def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x + 3 * y + 2 = 0

def inclination_angle (eq : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem line_inclination_angle :
  inclination_angle line_equation = 150 * Real.pi / 180 :=
sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1876_187658


namespace NUMINAMATH_CALUDE_union_A_B_union_complement_A_B_l1876_187671

-- Define the universe set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 7}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x : ℝ | 2 ≤ x ∧ x ≤ 7} := by sorry

-- Theorem for (∁ₓA) ∪ (∁ₓB)
theorem union_complement_A_B : (Set.compl A) ∪ (Set.compl B) = {x : ℝ | x < 3 ∨ x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_union_A_B_union_complement_A_B_l1876_187671


namespace NUMINAMATH_CALUDE_parabola_intersection_l1876_187621

theorem parabola_intersection (x₁ x₂ : ℝ) (m : ℝ) : 
  (∀ x y, x^2 = 4*y → ∃ k, y = k*x + m) →  -- Line passes through (0, m) and intersects parabola
  x₁ * x₂ = -4 →                           -- Product of x-coordinates is -4
  m = 1 :=                                 -- Conclusion: m must be 1
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1876_187621


namespace NUMINAMATH_CALUDE_trig_identity_l1876_187614

theorem trig_identity (α : ℝ) : -Real.sin α + Real.sqrt 3 * Real.cos α = 2 * Real.sin (α + 2 * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1876_187614


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1876_187632

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def is_nonagon (n : ℕ) : Prop := n = 9

theorem nonagon_diagonals :
  ∀ n : ℕ, is_nonagon n → num_diagonals n = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1876_187632


namespace NUMINAMATH_CALUDE_board_numbers_proof_l1876_187666

def pairwise_sums (a b c d e : ℤ) : Finset ℤ :=
  {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e}

theorem board_numbers_proof :
  ∃ (a b c d e : ℤ),
    pairwise_sums a b c d e = {5, 9, 10, 11, 12, 16, 16, 17, 21, 23} ∧
    Finset.toList {a, b, c, d, e} = [2, 3, 7, 9, 14] ∧
    a * b * c * d * e = 5292 :=
by sorry

end NUMINAMATH_CALUDE_board_numbers_proof_l1876_187666


namespace NUMINAMATH_CALUDE_x1_value_l1876_187665

theorem x1_value (x1 x2 x3 x4 : ℝ) 
  (h_order : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h_sum : (1-x1)^2 + (x1-x2)^2 + (x2-x3)^2 + (x3-x4)^2 + x4^2 = 9/16) : 
  x1 = 1 - 15 / Real.sqrt 80 := by
  sorry

end NUMINAMATH_CALUDE_x1_value_l1876_187665


namespace NUMINAMATH_CALUDE_mixture_volume_proof_l1876_187697

theorem mixture_volume_proof (initial_water_percent : Real) 
                             (final_water_percent : Real)
                             (added_water : Real) :
  initial_water_percent = 0.20 →
  final_water_percent = 0.25 →
  added_water = 10 →
  ∃ (initial_volume : Real),
    initial_volume * initial_water_percent + added_water = 
    final_water_percent * (initial_volume + added_water) ∧
    initial_volume = 150 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_proof_l1876_187697


namespace NUMINAMATH_CALUDE_min_value_expression_l1876_187626

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x + 3 * y) * (y + 3 * z) * (x + 3 * z + 1) ≥ 24 * Real.sqrt 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 3 * y₀) * (y₀ + 3 * z₀) * (x₀ + 3 * z₀ + 1) = 24 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1876_187626


namespace NUMINAMATH_CALUDE_min_value_theorem_l1876_187689

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : f a b 1 = 2) :
  ∃ (min_val : ℝ), min_val = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → f a' b' 1 = 2 → 1/a' + 4/b' ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1876_187689


namespace NUMINAMATH_CALUDE_second_quadrant_points_characterization_l1876_187682

def second_quadrant_points : Set (ℤ × ℤ) :=
  {p | p.1 < 0 ∧ p.2 > 0 ∧ p.2 ≤ p.1 + 4}

theorem second_quadrant_points_characterization :
  second_quadrant_points = {(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-3, 1)} := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_points_characterization_l1876_187682


namespace NUMINAMATH_CALUDE_expression_evaluation_l1876_187684

theorem expression_evaluation :
  let x : ℝ := 2
  (2 * (x^2 - 1) - 7*x - (2*x^2 - x + 3)) = -17 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1876_187684


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l1876_187677

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (feet_per_step : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (length_steps : ℝ) * feet_per_step * (width_steps : ℝ) * feet_per_step * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  expected_potato_yield 18 25 3 (3/4) = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l1876_187677


namespace NUMINAMATH_CALUDE_union_cardinality_lower_bound_equality_holds_l1876_187695

theorem union_cardinality_lower_bound 
  (A B C : Finset ℕ) 
  (h : A ∩ B ∩ C = ∅) : 
  (A ∪ B ∪ C).card ≥ (A.card + B.card + C.card) / 2 := by
  sorry

def equality_example : Finset ℕ × Finset ℕ × Finset ℕ :=
  ({1, 2}, {2, 3}, {3, 1})

theorem equality_holds (A B C : Finset ℕ) 
  (h : (A, B, C) = equality_example) :
  (A ∪ B ∪ C).card = (A.card + B.card + C.card) / 2 := by
  sorry

end NUMINAMATH_CALUDE_union_cardinality_lower_bound_equality_holds_l1876_187695


namespace NUMINAMATH_CALUDE_bird_migration_l1876_187616

/-- Bird migration problem -/
theorem bird_migration 
  (total_families : ℕ)
  (africa_families : ℕ)
  (asia_families : ℕ)
  (south_america_families : ℕ)
  (africa_days : ℕ)
  (asia_days : ℕ)
  (south_america_days : ℕ)
  (h1 : total_families = 200)
  (h2 : africa_families = 60)
  (h3 : asia_families = 95)
  (h4 : south_america_families = 30)
  (h5 : africa_days = 7)
  (h6 : asia_days = 14)
  (h7 : south_america_days = 10) :
  (total_families - (africa_families + asia_families + south_america_families) = 15) ∧
  (africa_families * africa_days + asia_families * asia_days + south_america_families * south_america_days = 2050) := by
  sorry


end NUMINAMATH_CALUDE_bird_migration_l1876_187616


namespace NUMINAMATH_CALUDE_line_parametric_to_cartesian_l1876_187676

/-- Given a line with parametric equations x = 1 + t/2 and y = 2 + (√3/2)t,
    its Cartesian equation is √3x - y + 2 - √3 = 0 --/
theorem line_parametric_to_cartesian :
  ∀ (x y t : ℝ),
  (x = 1 + t / 2 ∧ y = 2 + (Real.sqrt 3 / 2) * t) ↔
  (Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_parametric_to_cartesian_l1876_187676


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l1876_187630

/-- Ellipse definition -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

/-- Line equation -/
def line_equation (x y x₀ y₀ : ℝ) : Prop :=
  x * x₀ / 16 + y * y₀ / 4 = 1

/-- Tangent line property -/
def is_tangent_line (x₀ y₀ : ℝ) : Prop :=
  is_on_ellipse x₀ y₀ →
  ∀ x y : ℝ, line_equation x y x₀ y₀ →
  (x = x₀ ∧ y = y₀) ∨ ¬(is_on_ellipse x y)

/-- Main theorem -/
theorem tangent_line_to_ellipse :
  ∀ x₀ y₀ : ℝ, is_tangent_line x₀ y₀ :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l1876_187630


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l1876_187619

def U : Set ℕ := {x | 0 ≤ x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem intersection_complement_equals : S ∩ (U \ T) = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l1876_187619


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1876_187686

theorem circle_area_ratio (C D : Real) (r_C r_D : ℝ) :
  (60 / 360) * (2 * Real.pi * r_C) = (40 / 360) * (2 * Real.pi * r_D) →
  (Real.pi * r_C^2) / (Real.pi * r_D^2) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1876_187686


namespace NUMINAMATH_CALUDE_B_equals_zero_one_two_l1876_187605

def A : Set ℤ := {1, 0, -1, 2}

def B : Set ℕ := {y | ∃ x ∈ A, y = |x|}

theorem B_equals_zero_one_two : B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_B_equals_zero_one_two_l1876_187605


namespace NUMINAMATH_CALUDE_and_or_relationship_l1876_187648

theorem and_or_relationship (p q : Prop) : 
  ((p ∧ q) → (p ∨ q)) ∧ 
  (∃ (p q : Prop), (p ∨ q) ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_and_or_relationship_l1876_187648


namespace NUMINAMATH_CALUDE_power_function_through_point_l1876_187698

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x ^ n

-- Theorem statement
theorem power_function_through_point :
  ∀ f : ℝ → ℝ, isPowerFunction f → f 3 = Real.sqrt 3 → f = fun x => Real.sqrt x :=
by sorry


end NUMINAMATH_CALUDE_power_function_through_point_l1876_187698


namespace NUMINAMATH_CALUDE_system_solution_l1876_187661

theorem system_solution :
  ∃ (x y : ℝ),
    (4 * x - 3 * y = -2.4) ∧
    (5 * x + 6 * y = 7.5) ∧
    (x = 2.7 / 13) ∧
    (y = 1.0769) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1876_187661


namespace NUMINAMATH_CALUDE_book_reorganization_l1876_187612

theorem book_reorganization (initial_boxes : Nat) (initial_books_per_box : Nat) (new_books_per_box : Nat) :
  initial_boxes = 1278 →
  initial_books_per_box = 45 →
  new_books_per_box = 46 →
  (initial_boxes * initial_books_per_box) % new_books_per_box = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_reorganization_l1876_187612


namespace NUMINAMATH_CALUDE_brothers_initial_money_l1876_187609

theorem brothers_initial_money (michael_initial : ℕ) (brother_final : ℕ) (candy_cost : ℕ) :
  michael_initial = 42 →
  brother_final = 35 →
  candy_cost = 3 →
  ∃ (brother_initial : ℕ),
    brother_initial + michael_initial / 2 = brother_final + candy_cost ∧
    brother_initial = 17 :=
by sorry

end NUMINAMATH_CALUDE_brothers_initial_money_l1876_187609


namespace NUMINAMATH_CALUDE_price_increase_percentage_l1876_187618

theorem price_increase_percentage (old_price new_price : ℝ) 
  (hold : old_price = 300 ∧ new_price = 330) : 
  (new_price - old_price) / old_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l1876_187618


namespace NUMINAMATH_CALUDE_sum_of_products_l1876_187696

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 281) 
  (h2 : a + b + c = 17) : 
  a*b + b*c + c*a = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l1876_187696


namespace NUMINAMATH_CALUDE_set_as_interval_l1876_187647

def S : Set ℝ := {x : ℝ | -12 ≤ x ∧ x < 10 ∨ x > 11}

theorem set_as_interval : S = Set.Icc (-12) 10 ∪ Set.Ioi 11 := by sorry

end NUMINAMATH_CALUDE_set_as_interval_l1876_187647


namespace NUMINAMATH_CALUDE_system_solution_l1876_187646

/-- Given a system of equations, prove the conditions for x and y -/
theorem system_solution (a x y : ℝ) : 
  (x / 2 - (2 * x - 3 * y) / 5 = a - 1) →
  (x + 3 = y / 3) →
  (x < 0 ∧ y > 0) ↔ (7/10 < a ∧ a < 64/10) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1876_187646


namespace NUMINAMATH_CALUDE_units_digit_sum_octal_l1876_187668

/-- Converts a number from base 8 to base 10 -/
def octalToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 -/
def decimalToOctal (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a number in base 8 -/
def unitsDigitOctal (n : ℕ) : ℕ := sorry

theorem units_digit_sum_octal :
  unitsDigitOctal (decimalToOctal (octalToDecimal 45 + octalToDecimal 67)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_octal_l1876_187668


namespace NUMINAMATH_CALUDE_fibonacci_polynomial_property_l1876_187642

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Define the theorem
theorem fibonacci_polynomial_property (p : ℝ → ℝ) :
  (∀ k ∈ Finset.range 991, p (k + 992) = fib (k + 992)) →
  p 1983 = fib 1083 - 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_polynomial_property_l1876_187642


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1876_187615

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0
  h_a1_nonzero : a 1 ≠ 0
  h_geometric : (a 2) ^ 2 = a 1 * a 4

/-- The main theorem -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) :
  (seq.a 1 + seq.a 14) / seq.a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1876_187615


namespace NUMINAMATH_CALUDE_function_inequality_l1876_187635

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_deriv : ∀ x, deriv f x < 1) (h_f3 : f 3 = 4) :
  ∀ x, f (x + 1) < x + 2 ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1876_187635


namespace NUMINAMATH_CALUDE_simon_makes_three_pies_l1876_187678

/-- The number of blueberry pies Simon can make -/
def blueberry_pies (own_berries nearby_berries berries_per_pie : ℕ) : ℕ :=
  (own_berries + nearby_berries) / berries_per_pie

/-- Proof that Simon can make 3 blueberry pies -/
theorem simon_makes_three_pies :
  blueberry_pies 100 200 100 = 3 := by
  sorry

end NUMINAMATH_CALUDE_simon_makes_three_pies_l1876_187678


namespace NUMINAMATH_CALUDE_dog_food_cans_per_package_l1876_187650

/-- Given the following conditions:
  * Chad bought 6 packages of cat food, each containing 9 cans
  * Chad bought 2 packages of dog food
  * The total number of cat food cans is 48 more than the total number of dog food cans
  Prove that each package of dog food contains 3 cans -/
theorem dog_food_cans_per_package :
  let cat_packages : ℕ := 6
  let cat_cans_per_package : ℕ := 9
  let dog_packages : ℕ := 2
  let total_cat_cans : ℕ := cat_packages * cat_cans_per_package
  let dog_cans_per_package : ℕ := total_cat_cans / dog_packages - 24
  dog_cans_per_package = 3 := by sorry

end NUMINAMATH_CALUDE_dog_food_cans_per_package_l1876_187650


namespace NUMINAMATH_CALUDE_card_pack_size_l1876_187685

theorem card_pack_size (prob_not_face : ℝ) (num_face_cards : ℕ) (h1 : prob_not_face = 0.6923076923076923) (h2 : num_face_cards = 12) : 
  ∃ n : ℕ, n = 39 ∧ (n - num_face_cards : ℝ) / n = prob_not_face := by
sorry

end NUMINAMATH_CALUDE_card_pack_size_l1876_187685


namespace NUMINAMATH_CALUDE_vector_operation_l1876_187606

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (2, -2)) :
  2 • a - b = (4, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l1876_187606


namespace NUMINAMATH_CALUDE_chinese_in_group_l1876_187613

theorem chinese_in_group (total : ℕ) (americans : ℕ) (australians : ℕ) 
  (h1 : total = 49)
  (h2 : americans = 16)
  (h3 : australians = 11) :
  total - americans - australians = 22 :=
by sorry

end NUMINAMATH_CALUDE_chinese_in_group_l1876_187613


namespace NUMINAMATH_CALUDE_properties_of_f_l1876_187631

def f (x : ℝ) := -5 * x

theorem properties_of_f :
  (∃ m b : ℝ, ∀ x, f x = m * x + b) ∧  -- f is linear
  (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂) ∧   -- f is decreasing
  (f 0 = 0) ∧                          -- f passes through (0,0)
  (∀ x ≠ 0, x * (f x) < 0) :=          -- f is in 2nd and 4th quadrants
by sorry

end NUMINAMATH_CALUDE_properties_of_f_l1876_187631


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l1876_187604

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_problem 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)
  (h_condition : a 6 - (a 7)^2 + a 8 = 0)
  (h_geometric : geometric_sequence b)
  (h_equal : b 7 = a 7) :
  b 3 * b 8 * b 10 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l1876_187604


namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_l1876_187638

theorem min_value_cyclic_fraction (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / b + b / c + c / d + d / a ≥ 4 ∧
  (a / b + b / c + c / d + d / a = 4 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_l1876_187638


namespace NUMINAMATH_CALUDE_range_of_t_below_line_l1876_187653

/-- A point (x, y) is below a line ax + by + c = 0 if ax + by + c > 0 -/
def IsBelowLine (x y a b c : ℝ) : Prop := a * x + b * y + c > 0

/-- The theorem stating the range of t given the conditions -/
theorem range_of_t_below_line :
  ∀ t : ℝ, IsBelowLine 2 (3 * t) 2 (-1) 6 → t < 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_below_line_l1876_187653


namespace NUMINAMATH_CALUDE_smallest_number_of_rectangles_l1876_187688

/-- The side length of the rectangle along one dimension -/
def rectangle_side1 : ℕ := 3

/-- The side length of the rectangle along the other dimension -/
def rectangle_side2 : ℕ := 4

/-- The area of a single rectangle -/
def rectangle_area : ℕ := rectangle_side1 * rectangle_side2

/-- The side length of the square that can be covered exactly by the rectangles -/
def square_side : ℕ := lcm rectangle_side1 rectangle_side2

/-- The area of the square -/
def square_area : ℕ := square_side ^ 2

/-- The number of rectangles needed to cover the square -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_number_of_rectangles :
  num_rectangles = 12 ∧
  ∀ n : ℕ, n < num_rectangles →
    ¬(∃ s : ℕ, s ^ 2 = n * rectangle_area ∧ s % rectangle_side1 = 0 ∧ s % rectangle_side2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_rectangles_l1876_187688


namespace NUMINAMATH_CALUDE_soccer_survey_l1876_187629

/-- Represents the fraction of students who enjoy playing soccer -/
def enjoy_soccer : ℚ := 1/2

/-- Represents the fraction of students who honestly say they enjoy soccer among those who enjoy it -/
def honest_enjoy : ℚ := 7/10

/-- Represents the fraction of students who honestly say they do not enjoy soccer among those who do not enjoy it -/
def honest_not_enjoy : ℚ := 8/10

/-- The fraction of students who claim they do not enjoy playing soccer but actually enjoy it -/
def fraction_false_claim : ℚ := 3/11

theorem soccer_survey :
  (enjoy_soccer * (1 - honest_enjoy)) / 
  ((enjoy_soccer * (1 - honest_enjoy)) + ((1 - enjoy_soccer) * honest_not_enjoy)) = fraction_false_claim := by
  sorry

end NUMINAMATH_CALUDE_soccer_survey_l1876_187629


namespace NUMINAMATH_CALUDE_factorization_equality_l1876_187622

theorem factorization_equality (x : ℝ) : 3 * x * (x - 5) + 4 * (x - 5) = (3 * x + 4) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1876_187622


namespace NUMINAMATH_CALUDE_odd_m_triple_g_35_l1876_187663

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 then n / 3
  else n  -- This case is not specified in the original problem, but needed for completeness

theorem odd_m_triple_g_35 (m : ℤ) (h_odd : m % 2 = 1) (h_triple_g : g (g (g m)) = 35) : m = 85 := by
  sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_35_l1876_187663


namespace NUMINAMATH_CALUDE_jim_ran_16_miles_l1876_187683

/-- The number of miles Jim ran in 2 hours -/
def jim_miles : ℝ := 16

/-- The number of hours Jim ran -/
def jim_hours : ℝ := 2

/-- The number of miles Frank ran in 2 hours -/
def frank_miles : ℝ := 20

/-- The difference in miles between Frank and Jim in one hour -/
def miles_difference : ℝ := 2

theorem jim_ran_16_miles :
  jim_miles = 16 ∧
  jim_hours = 2 ∧
  frank_miles = 20 ∧
  miles_difference = 2 →
  jim_miles = 16 :=
by sorry

end NUMINAMATH_CALUDE_jim_ran_16_miles_l1876_187683


namespace NUMINAMATH_CALUDE_hyperbola_center_l1876_187611

/-- The center of a hyperbola is the point (h, k) such that the equation of the hyperbola
    can be written in the form ((y-k)/a)² - ((x-h)/b)² = 1 for some non-zero real numbers a and b. -/
def is_center_of_hyperbola (h k : ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  ∀ (x y : ℝ), ((3 * y + 3)^2 / 7^2) - ((4 * x - 5)^2 / 3^2) = 1 ↔
                ((y - k) / a)^2 - ((x - h) / b)^2 = 1

/-- The center of the hyperbola (3y+3)²/7² - (4x-5)²/3² = 1 is (5/4, -1). -/
theorem hyperbola_center :
  is_center_of_hyperbola (5/4) (-1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_l1876_187611


namespace NUMINAMATH_CALUDE_five_divides_square_iff_five_divides_l1876_187603

theorem five_divides_square_iff_five_divides (a : ℤ) : 
  5 ∣ a^2 ↔ 5 ∣ a := by sorry

end NUMINAMATH_CALUDE_five_divides_square_iff_five_divides_l1876_187603


namespace NUMINAMATH_CALUDE_max_y_coordinate_of_ellipse_l1876_187660

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the equation of the ellipse -/
def isOnEllipse (p : Point) : Prop :=
  (p.x - 3)^2 / 49 + (p.y - 4)^2 / 25 = 1

/-- Theorem: The maximum y-coordinate of any point on the given ellipse is 9 -/
theorem max_y_coordinate_of_ellipse :
  ∀ p : Point, isOnEllipse p → p.y ≤ 9 ∧ ∃ q : Point, isOnEllipse q ∧ q.y = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_of_ellipse_l1876_187660


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l1876_187649

-- Define the ratio of angles
def angle_ratio : List Nat := [2, 4, 4, 5, 7]

-- Define the number of sides in a pentagon
def pentagon_sides : Nat := 5

-- Theorem statement
theorem largest_angle_in_pentagon (ε : Real) (h_ε : ε > 0) :
  ∃ (x : Real),
    (x > 0) ∧
    (List.sum (List.map (· * x) angle_ratio) = (pentagon_sides - 2) * 180) ∧
    (abs ((List.maximum angle_ratio).get! * x - 171.82) < ε) :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l1876_187649


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1876_187610

/-- The equation of the curve in Cartesian coordinates -/
def curve_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

/-- The condition for the curve to be an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), curve_equation m x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)

/-- The theorem stating the range of m for which the curve is an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ m > 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l1876_187610


namespace NUMINAMATH_CALUDE_points_below_line_l1876_187669

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d ∧ a₄ = a₃ + d

-- Define the geometric sequence
def geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ q : ℝ, a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q

theorem points_below_line (x₁ x₂ y₁ y₂ : ℝ) :
  arithmetic_sequence 1 x₁ x₂ 2 →
  geometric_sequence 1 y₁ y₂ 2 →
  x₁ > y₁ ∧ x₂ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_points_below_line_l1876_187669


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l1876_187602

theorem quadratic_root_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x + 2 = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₁ + x₂ = -4 → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l1876_187602


namespace NUMINAMATH_CALUDE_tax_to_savings_ratio_l1876_187656

/-- Esperanza's monthly finances -/
def monthly_finances (rent food mortgage savings tax gross_salary : ℚ) : Prop :=
  rent = 600 ∧
  food = (3/5) * rent ∧
  mortgage = 3 * food ∧
  savings = 2000 ∧
  gross_salary = 4840 ∧
  tax = gross_salary - (rent + food + mortgage + savings)

/-- The ratio of tax to savings is 2:5 -/
theorem tax_to_savings_ratio 
  (rent food mortgage savings tax gross_salary : ℚ) 
  (h : monthly_finances rent food mortgage savings tax gross_salary) : 
  tax / savings = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_tax_to_savings_ratio_l1876_187656


namespace NUMINAMATH_CALUDE_min_positive_temperatures_l1876_187637

theorem min_positive_temperatures (n : ℕ) (pos_products neg_products : ℕ) : 
  n * (n - 1) = pos_products + neg_products →
  pos_products = 68 →
  neg_products = 64 →
  ∃ k : ℕ, k ≤ n ∧ k * (k - 1) = pos_products ∧ k ≥ 4 ∧ 
  ∀ m : ℕ, m < k → m * (m - 1) ≠ pos_products :=
by sorry

end NUMINAMATH_CALUDE_min_positive_temperatures_l1876_187637


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l1876_187634

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → 
  a = 25 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l1876_187634


namespace NUMINAMATH_CALUDE_select_captains_l1876_187641

theorem select_captains (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_select_captains_l1876_187641


namespace NUMINAMATH_CALUDE_orchids_count_l1876_187643

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  initial_roses : Nat
  initial_orchids : Nat
  current_roses : Nat
  orchid_rose_difference : Nat

/-- Calculates the current number of orchids in the vase -/
def current_orchids (vase : FlowerVase) : Nat :=
  vase.current_roses + vase.orchid_rose_difference

theorem orchids_count (vase : FlowerVase) 
  (h1 : vase.initial_roses = 7)
  (h2 : vase.initial_orchids = 12)
  (h3 : vase.current_roses = 11)
  (h4 : vase.orchid_rose_difference = 9) :
  current_orchids vase = 20 := by
  sorry

end NUMINAMATH_CALUDE_orchids_count_l1876_187643


namespace NUMINAMATH_CALUDE_bill_face_value_is_12250_l1876_187608

/-- Calculates the face value of a bill given the true discount, time period, and annual discount rate. -/
def calculate_face_value (true_discount : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  (true_discount * (100 + rate * time)) / (rate * time)

/-- Theorem stating that given the specific conditions, the face value of the bill is 12250. -/
theorem bill_face_value_is_12250 :
  let true_discount : ℚ := 3500
  let time : ℚ := 2
  let rate : ℚ := 20
  calculate_face_value true_discount time rate = 12250 := by
  sorry

#eval calculate_face_value 3500 2 20

end NUMINAMATH_CALUDE_bill_face_value_is_12250_l1876_187608


namespace NUMINAMATH_CALUDE_part_one_part_two_l1876_187664

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ :=
  if a ≥ b then b^2 else 2*a - b

-- Theorem for part (1)
theorem part_one : triangle (-3) (-4) = 16 := by sorry

-- Theorem for part (2)
theorem part_two : triangle (triangle (-2) 3) (-8) = 64 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1876_187664


namespace NUMINAMATH_CALUDE_trig_identity_l1876_187654

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1876_187654


namespace NUMINAMATH_CALUDE_complex_sum_pure_imaginary_l1876_187657

theorem complex_sum_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (z₁ + z₂).re = 0 → a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_pure_imaginary_l1876_187657


namespace NUMINAMATH_CALUDE_least_common_period_is_36_l1876_187687

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 6) + f (x - 6) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q, 0 < q ∧ q < p → ¬IsPeriod f q

theorem least_common_period_is_36 :
  ∃ p, p = 36 ∧ 
    (∀ f : ℝ → ℝ, FunctionalEquation f → IsLeastPeriod f p) ∧
    (∀ q, q > 0 → 
      (∀ f : ℝ → ℝ, FunctionalEquation f → IsPeriod f q) → 
      q ≥ p) :=
sorry

end NUMINAMATH_CALUDE_least_common_period_is_36_l1876_187687


namespace NUMINAMATH_CALUDE_function_inequality_l1876_187645

theorem function_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  let f := fun x => |x + a| * |x + b|
  f 1 * f c ≥ 16 * a * b * c := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1876_187645


namespace NUMINAMATH_CALUDE_midpoint_distance_is_1300_l1876_187633

/-- The distance from the school to the midpoint of the total path -/
def midpoint_distance (school_to_kindergarten_km : ℕ) (school_to_kindergarten_m : ℕ) (kindergarten_to_house_m : ℕ) : ℕ :=
  ((school_to_kindergarten_km * 1000 + school_to_kindergarten_m + kindergarten_to_house_m) / 2)

/-- Theorem stating that the midpoint distance is 1300 meters -/
theorem midpoint_distance_is_1300 :
  midpoint_distance 1 700 900 = 1300 := by
  sorry

#eval midpoint_distance 1 700 900

end NUMINAMATH_CALUDE_midpoint_distance_is_1300_l1876_187633


namespace NUMINAMATH_CALUDE_max_inequality_sqrt_sum_l1876_187693

theorem max_inequality_sqrt_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) :
  ∃ (m : ℝ), m = 2 + Real.sqrt 5 ∧ 
  (∀ (x : ℝ), Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1) > x → x ≤ m) ∧
  Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1) > m :=
by sorry

end NUMINAMATH_CALUDE_max_inequality_sqrt_sum_l1876_187693


namespace NUMINAMATH_CALUDE_range_of_m_l1876_187620

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) ↔ m ∈ Set.Icc (-3) 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1876_187620


namespace NUMINAMATH_CALUDE_cut_tetrahedron_unfolds_to_given_config_l1876_187601

/-- Represents a polyhedron with vertices and edges -/
structure Polyhedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)

/-- Represents the unfolded configuration of a polyhedron -/
structure UnfoldedConfig where
  vertices : Finset (ℝ × ℝ)
  edges : Finset ((ℝ × ℝ) × (ℝ × ℝ))

/-- The given unfolded configuration from the problem -/
def given_config : UnfoldedConfig := sorry

/-- A tetrahedron with a smaller tetrahedron removed -/
def cut_tetrahedron : Polyhedron where
  vertices := {1, 2, 3, 4, 5, 6, 7, 8}
  edges := {(1,2), (1,3), (1,4), (2,3), (2,4), (3,4),
            (1,5), (2,6), (3,7), (4,8),
            (5,6), (6,7), (7,8)}

/-- Function to unfold a polyhedron onto a plane -/
def unfold (p : Polyhedron) : UnfoldedConfig := sorry

theorem cut_tetrahedron_unfolds_to_given_config :
  unfold cut_tetrahedron = given_config := by sorry

end NUMINAMATH_CALUDE_cut_tetrahedron_unfolds_to_given_config_l1876_187601


namespace NUMINAMATH_CALUDE_capri_sun_pouches_per_box_l1876_187659

theorem capri_sun_pouches_per_box 
  (total_boxes : ℕ) 
  (total_paid : ℚ) 
  (cost_per_pouch : ℚ) 
  (h1 : total_boxes = 10) 
  (h2 : total_paid = 12) 
  (h3 : cost_per_pouch = 1/5) : 
  (total_paid / cost_per_pouch) / total_boxes = 6 := by
sorry

end NUMINAMATH_CALUDE_capri_sun_pouches_per_box_l1876_187659


namespace NUMINAMATH_CALUDE_cost_per_load_is_25_cents_l1876_187600

/-- Represents the detergent scenario -/
structure DetergentScenario where
  loads_per_bottle : ℕ
  regular_price : ℚ
  sale_price : ℚ

/-- Calculates the cost per load in cents for a given detergent scenario -/
def cost_per_load_cents (scenario : DetergentScenario) : ℚ :=
  (2 * scenario.sale_price * 100) / (2 * scenario.loads_per_bottle)

/-- Theorem stating that the cost per load is 25 cents for the given scenario -/
theorem cost_per_load_is_25_cents (scenario : DetergentScenario)
  (h1 : scenario.loads_per_bottle = 80)
  (h2 : scenario.regular_price = 25)
  (h3 : scenario.sale_price = 20) :
  cost_per_load_cents scenario = 25 := by
  sorry

#eval cost_per_load_cents { loads_per_bottle := 80, regular_price := 25, sale_price := 20 }

end NUMINAMATH_CALUDE_cost_per_load_is_25_cents_l1876_187600


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1876_187672

/-- A geometric sequence of positive integers with first term 2 and fourth term 162 has third term 18 -/
theorem geometric_sequence_third_term : 
  ∀ (a : ℕ → ℕ) (r : ℕ),
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 1 = 2 →                     -- first term is 2
  a 4 = 162 →                   -- fourth term is 162
  a 3 = 18 :=                   -- third term is 18
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1876_187672


namespace NUMINAMATH_CALUDE_ellipse_condition_l1876_187662

-- Define the equation
def equation (m x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

-- Define what it means for the equation to represent an ellipse
def represents_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), equation m x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)

-- Theorem statement
theorem ellipse_condition (m : ℝ) :
  represents_ellipse m → m > 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1876_187662


namespace NUMINAMATH_CALUDE_trig_inequality_l1876_187674

theorem trig_inequality : Real.tan (55 * π / 180) > Real.cos (55 * π / 180) ∧ Real.cos (55 * π / 180) > Real.sin (33 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l1876_187674


namespace NUMINAMATH_CALUDE_x_squared_plus_3x_equals_one_l1876_187692

theorem x_squared_plus_3x_equals_one (x : ℝ) 
  (h : (x^2 + 3*x)^2 + 2*(x^2 + 3*x) - 3 = 0) : 
  x^2 + 3*x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_3x_equals_one_l1876_187692


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1876_187691

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 15 * y^2 - 82 * y + 56 = (C * y - 14) * (D * y - 4)) →
  C * D + C = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1876_187691


namespace NUMINAMATH_CALUDE_good_apples_count_l1876_187667

theorem good_apples_count (total : ℕ) (unripe : ℕ) (h1 : total = 14) (h2 : unripe = 6) :
  total - unripe = 8 := by
sorry

end NUMINAMATH_CALUDE_good_apples_count_l1876_187667


namespace NUMINAMATH_CALUDE_x_sixth_power_equals_one_l1876_187675

theorem x_sixth_power_equals_one (x : ℝ) (h : 1 + x + x^2 + x^3 + x^4 + x^5 = 0) : x^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_power_equals_one_l1876_187675


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_falling_object_time_l1876_187644

-- Part 1: Solving (x-1)^2 = 49
theorem solve_quadratic_equation :
  ∀ x : ℝ, (x - 1)^2 = 49 ↔ x = 8 ∨ x = -6 :=
by sorry

-- Part 2: Finding the time for an object to reach the ground
theorem falling_object_time (h t : ℝ) :
  h = 4.9 * t^2 →
  h = 10 →
  t = 10 / 7 :=
by sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_falling_object_time_l1876_187644


namespace NUMINAMATH_CALUDE_roots_relationship_l1876_187655

/-- The polynomial h(x) = x^3 - 2x^2 - x + 2 -/
def h (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

/-- The polynomial j(x) = x^3 + bx^2 + cx + d -/
def j (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- The theorem stating the relationship between h and j -/
theorem roots_relationship (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → j (x^3) b c d = 0) →
  b = 10 ∧ c = 31 ∧ d = 30 := by
sorry

end NUMINAMATH_CALUDE_roots_relationship_l1876_187655


namespace NUMINAMATH_CALUDE_smallest_tree_height_l1876_187639

/-- Proves that the height of the smallest tree is 12 feet given the conditions of the problem -/
theorem smallest_tree_height (tallest middle smallest : ℝ) : 
  tallest = 108 ∧ 
  middle = tallest / 2 - 6 ∧ 
  smallest = middle / 4 → 
  smallest = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_tree_height_l1876_187639


namespace NUMINAMATH_CALUDE_books_movies_difference_l1876_187651

theorem books_movies_difference : 
  ∀ (total_books total_movies : ℕ),
    total_books = 10 →
    total_movies = 6 →
    total_books - total_movies = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_books_movies_difference_l1876_187651


namespace NUMINAMATH_CALUDE_bacteria_count_scientific_notation_l1876_187670

/-- The number of bacteria on a pair of unwashed hands -/
def bacteria_count : ℕ := 750000

/-- Scientific notation representation of the bacteria count -/
def scientific_notation : ℝ := 7.5 * (10 ^ 5)

theorem bacteria_count_scientific_notation : 
  (bacteria_count : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_bacteria_count_scientific_notation_l1876_187670


namespace NUMINAMATH_CALUDE_inequality_proof_l1876_187628

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0) 
  (h5 : x + y + z + t = 2) : 
  Real.sqrt (x^2 + z^2) + Real.sqrt (x^2 + 1) + Real.sqrt (z^2 + y^2) + 
  Real.sqrt (y^2 + t^2) + Real.sqrt (t^2 + 4) ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1876_187628


namespace NUMINAMATH_CALUDE_toy_store_revenue_l1876_187627

theorem toy_store_revenue (D : ℚ) (D_pos : D > 0) : 
  let N := (2 : ℚ) / 5 * D
  let J := (1 : ℚ) / 5 * N
  let F := (3 : ℚ) / 4 * D
  let avg := (N + J + F) / 3
  D / avg = 100 / 41 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l1876_187627
