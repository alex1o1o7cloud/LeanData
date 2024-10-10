import Mathlib

namespace pill_supply_lasts_eight_months_l1870_187082

/-- Calculates the duration in months that a pill supply will last -/
def pill_supply_duration (total_pills : ℕ) (days_per_pill : ℕ) (days_per_month : ℕ) : ℕ :=
  (total_pills * days_per_pill) / days_per_month

/-- Proves that a supply of 120 pills, taken every two days, lasts 8 months -/
theorem pill_supply_lasts_eight_months :
  pill_supply_duration 120 2 30 = 8 := by
  sorry

#eval pill_supply_duration 120 2 30

end pill_supply_lasts_eight_months_l1870_187082


namespace donuts_per_box_l1870_187031

theorem donuts_per_box (total_boxes : Nat) (boxes_given : Nat) (extra_donuts_given : Nat) (donuts_left : Nat) :
  total_boxes = 4 →
  boxes_given = 1 →
  extra_donuts_given = 6 →
  donuts_left = 30 →
  ∃ (donuts_per_box : Nat), 
    donuts_per_box * total_boxes = 
      donuts_per_box * boxes_given + extra_donuts_given + donuts_left ∧
    donuts_per_box = 12 := by
  sorry

end donuts_per_box_l1870_187031


namespace rain_in_first_hour_l1870_187040

theorem rain_in_first_hour (first_hour : ℝ) (second_hour : ℝ) : 
  second_hour = 2 * first_hour + 7 →
  first_hour + second_hour = 22 →
  first_hour = 5 := by sorry

end rain_in_first_hour_l1870_187040


namespace infinitely_many_perfect_squares_l1870_187023

theorem infinitely_many_perfect_squares (k : ℕ+) :
  ∃ f : ℕ → ℕ+, Monotone f ∧ ∀ i : ℕ, ∃ m : ℕ+, (f i : ℕ) * 2^(k : ℕ) - 7 = m^2 :=
sorry

end infinitely_many_perfect_squares_l1870_187023


namespace range_of_m_l1870_187076

theorem range_of_m (P : ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x+1) + m = 0) :
  ∀ m : ℝ, (∃ x : ℝ, 4^x - 2^(x+1) + m = 0) → m ≤ 1 :=
by sorry

end range_of_m_l1870_187076


namespace units_digit_of_8_power_2022_l1870_187038

theorem units_digit_of_8_power_2022 : 8^2022 % 10 = 4 := by
  sorry

end units_digit_of_8_power_2022_l1870_187038


namespace allens_mother_age_l1870_187072

-- Define Allen's age as a function of his mother's age
def allen_age (mother_age : ℕ) : ℕ := mother_age - 25

-- Define the condition that in 3 years, the sum of their ages will be 41
def future_age_sum (mother_age : ℕ) : Prop :=
  (mother_age + 3) + (allen_age mother_age + 3) = 41

-- Theorem stating that Allen's mother's present age is 30
theorem allens_mother_age :
  ∃ (mother_age : ℕ), 
    (allen_age mother_age = mother_age - 25) ∧ 
    (future_age_sum mother_age) ∧ 
    (mother_age = 30) := by
  sorry

end allens_mother_age_l1870_187072


namespace charlie_max_success_ratio_l1870_187081

theorem charlie_max_success_ratio 
  (alpha_first_two : ℚ)
  (alpha_last_two : ℚ)
  (charlie_daily : ℕ → ℚ)
  (charlie_attempted : ℕ → ℕ)
  (h1 : alpha_first_two = 120 / 200)
  (h2 : alpha_last_two = 80 / 200)
  (h3 : ∀ i ∈ Finset.range 4, 0 < charlie_daily i)
  (h4 : ∀ i ∈ Finset.range 4, charlie_daily i < 1)
  (h5 : ∀ i ∈ Finset.range 2, charlie_daily i < alpha_first_two)
  (h6 : ∀ i ∈ Finset.range 2, charlie_daily (i + 2) < alpha_last_two)
  (h7 : ∀ i ∈ Finset.range 4, charlie_attempted i > 0)
  (h8 : charlie_attempted 0 + charlie_attempted 1 < 200)
  (h9 : (charlie_attempted 0 + charlie_attempted 1 + charlie_attempted 2 + charlie_attempted 3) = 400)
  : (charlie_daily 0 * charlie_attempted 0 + charlie_daily 1 * charlie_attempted 1 + 
     charlie_daily 2 * charlie_attempted 2 + charlie_daily 3 * charlie_attempted 3) / 400 ≤ 239 / 400 :=
sorry

end charlie_max_success_ratio_l1870_187081


namespace S_equals_zero_two_neg_two_l1870_187093

def imaginary_unit : ℂ := Complex.I

def S : Set ℂ := {z | ∃ n : ℤ, z = (imaginary_unit ^ n) + (imaginary_unit ^ (-n))}

theorem S_equals_zero_two_neg_two : S = {0, 2, -2} := by sorry

end S_equals_zero_two_neg_two_l1870_187093


namespace perpendicular_point_k_range_l1870_187041

/-- Given points A(1,0) and B(3,0), if there exists a point P on the line y = kx + 1
    such that PA ⊥ PB, then -4/3 ≤ k ≤ 0. -/
theorem perpendicular_point_k_range (k : ℝ) :
  (∃ P : ℝ × ℝ, P.2 = k * P.1 + 1 ∧
    ((P.1 - 1) * (P.1 - 3) + P.2^2 = 0)) →
  -4/3 ≤ k ∧ k ≤ 0 := by
  sorry

end perpendicular_point_k_range_l1870_187041


namespace complex_arithmetic_equality_l1870_187024

theorem complex_arithmetic_equality : 10 - 9 * 8 + 7^2 / 2 - 3 * 4 + 6 - 5 = -48.5 := by
  sorry

end complex_arithmetic_equality_l1870_187024


namespace k_range_for_positive_f_l1870_187085

/-- Given a function f(x) = 32x - (k + 1)3^x + 2 that is always positive for real x,
    prove that k is in the range (-∞, 2^(-1)). -/
theorem k_range_for_positive_f (k : ℝ) :
  (∀ x : ℝ, 32 * x - (k + 1) * 3^x + 2 > 0) →
  k < 1/2 :=
by sorry

end k_range_for_positive_f_l1870_187085


namespace symmetric_point_y_axis_symmetric_points_coordinates_l1870_187074

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The symmetric point about the y-axis -/
def symmetricAboutYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

theorem symmetric_point_y_axis (p : Point2D) :
  let p' := symmetricAboutYAxis p
  p'.x = -p.x ∧ p'.y = p.y := by sorry

/-- Given points A, B, and C -/
def A : Point2D := { x := -3, y := 2 }
def B : Point2D := { x := -4, y := -3 }
def C : Point2D := { x := -1, y := -1 }

/-- Symmetric points A', B', and C' -/
def A' : Point2D := symmetricAboutYAxis A
def B' : Point2D := symmetricAboutYAxis B
def C' : Point2D := symmetricAboutYAxis C

theorem symmetric_points_coordinates :
  A'.x = 3 ∧ A'.y = 2 ∧
  B'.x = 4 ∧ B'.y = -3 ∧
  C'.x = 1 ∧ C'.y = -1 := by sorry

end symmetric_point_y_axis_symmetric_points_coordinates_l1870_187074


namespace attic_junk_items_l1870_187016

theorem attic_junk_items (useful_percent : ℝ) (heirloom_percent : ℝ) (junk_percent : ℝ) (useful_count : ℕ) :
  useful_percent = 0.20 →
  heirloom_percent = 0.10 →
  junk_percent = 0.70 →
  useful_percent + heirloom_percent + junk_percent = 1 →
  useful_count = 8 →
  ⌊(useful_count / useful_percent) * junk_percent⌋ = 28 := by
sorry

end attic_junk_items_l1870_187016


namespace guards_in_team_l1870_187012

theorem guards_in_team (s b n : ℕ) : 
  s > 0 ∧ b > 0 ∧ n > 0 →  -- positive integers
  s * b * n = 1001 →  -- total person-nights
  s < n →  -- guards in team less than nights slept
  n < b →  -- nights slept less than number of teams
  s = 7 :=  -- prove number of guards in a team is 7
by sorry

end guards_in_team_l1870_187012


namespace coefficient_of_x_is_14_l1870_187017

def expression (x : ℝ) : ℝ := 2 * (x - 6) + 5 * (3 - 3 * x^2 + 6 * x) - 6 * (3 * x - 5)

theorem coefficient_of_x_is_14 : 
  ∃ a b c : ℝ, ∀ x : ℝ, expression x = a * x^2 + 14 * x + c :=
sorry

end coefficient_of_x_is_14_l1870_187017


namespace triangle_angle_inequality_l1870_187099

theorem triangle_angle_inequality (α β γ s R : Real) : 
  α > 0 → β > 0 → γ > 0 → 
  α + β + γ = π →
  s > 0 → R > 0 →
  (α + β) * (β + γ) * (γ + α) ≤ 4 * (π / Real.sqrt 3)^3 * R / s := by
  sorry

end triangle_angle_inequality_l1870_187099


namespace power_fraction_product_l1870_187068

theorem power_fraction_product : (-4/5)^2022 * (5/4)^2023 = 5/4 := by
  sorry

end power_fraction_product_l1870_187068


namespace opposite_of_negative_2022_opposite_of_negative_2022_is_2022_l1870_187033

theorem opposite_of_negative_2022 : ℤ → Prop :=
  fun x => ((-2022 : ℤ) + x = 0) → x = 2022

-- The proof is omitted
theorem opposite_of_negative_2022_is_2022 : opposite_of_negative_2022 2022 := by
  sorry

end opposite_of_negative_2022_opposite_of_negative_2022_is_2022_l1870_187033


namespace honey_eaten_by_bears_l1870_187005

theorem honey_eaten_by_bears (initial_honey : Real) (remaining_honey : Real)
  (h1 : initial_honey = 0.36)
  (h2 : remaining_honey = 0.31) :
  initial_honey - remaining_honey = 0.05 := by
  sorry

end honey_eaten_by_bears_l1870_187005


namespace line_parallel_to_y_axis_l1870_187092

/-- A line parallel to the y-axis passing through a point has a constant x-coordinate -/
theorem line_parallel_to_y_axis (x₀ y₀ : ℝ) :
  let L := {p : ℝ × ℝ | p.1 = x₀}
  ((-1, 3) ∈ L) → (∀ p ∈ L, ∀ q ∈ L, p.2 ≠ q.2 → p.1 = q.1) →
  (∀ p ∈ L, p.1 = -1) :=
by sorry

end line_parallel_to_y_axis_l1870_187092


namespace mean_scores_equal_7_l1870_187056

def class1_scores : List Nat := [10, 9, 8, 7, 7, 7, 7, 5, 5, 5]
def class2_scores : List Nat := [9, 8, 8, 7, 7, 7, 7, 7, 5, 5]

def mean (scores : List Nat) : Rat :=
  (scores.sum : Rat) / scores.length

theorem mean_scores_equal_7 :
  mean class1_scores = 7 ∧ mean class2_scores = 7 := by
  sorry

end mean_scores_equal_7_l1870_187056


namespace female_salmon_count_l1870_187055

theorem female_salmon_count (male_salmon : ℕ) (total_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : total_salmon = 971639) :
  total_salmon - male_salmon = 259378 := by
  sorry

end female_salmon_count_l1870_187055


namespace product_remainder_remainder_98_102_mod_11_l1870_187071

theorem product_remainder (a b n : ℕ) (h : n > 0) : (a * b) % n = ((a % n) * (b % n)) % n := by sorry

theorem remainder_98_102_mod_11 : (98 * 102) % 11 = 1 := by sorry

end product_remainder_remainder_98_102_mod_11_l1870_187071


namespace bakers_pastries_l1870_187054

/-- Baker's pastry problem -/
theorem bakers_pastries 
  (total_cakes : ℕ) 
  (sold_cakes : ℕ) 
  (sold_pastries : ℕ) 
  (remaining_pastries : ℕ) 
  (h1 : total_cakes = 124) 
  (h2 : sold_cakes = 104) 
  (h3 : sold_pastries = 29) 
  (h4 : remaining_pastries = 27) : 
  sold_pastries + remaining_pastries = 56 := by
  sorry

end bakers_pastries_l1870_187054


namespace symmetry_implies_m_equals_4_l1870_187025

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Checks if two points are symmetric with respect to the y-axis -/
def symmetric_y_axis (a b : Point2D) : Prop :=
  a.x = -b.x ∧ a.y = b.y

theorem symmetry_implies_m_equals_4 (m : ℝ) :
  let a := Point2D.mk (-3) m
  let b := Point2D.mk 3 4
  symmetric_y_axis a b → m = 4 := by
  sorry

end symmetry_implies_m_equals_4_l1870_187025


namespace selling_price_calculation_l1870_187064

/-- Calculates the selling price of an article given the gain and gain percentage -/
theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) : 
  gain = 30 ∧ gain_percentage = 20 → 
  (gain / (gain_percentage / 100)) + gain = 180 := by
  sorry

end selling_price_calculation_l1870_187064


namespace towel_length_decrease_l1870_187069

/-- Theorem: Percentage decrease in towel length
Given a towel that lost some percentage of its length and 20% of its breadth,
resulting in a 36% decrease in area, prove that the percentage decrease in length is 20%.
-/
theorem towel_length_decrease (L B : ℝ) (L' B' : ℝ) (h_positive : L > 0 ∧ B > 0) :
  B' = 0.8 * B →                         -- Breadth decreased by 20%
  L' * B' = 0.64 * (L * B) →             -- Area decreased by 36%
  L' = 0.8 * L                           -- Length decreased by 20%
  := by sorry

end towel_length_decrease_l1870_187069


namespace trig_expression_simplification_l1870_187042

theorem trig_expression_simplification (α : ℝ) :
  (Real.tan (2 * Real.pi + α)) / (Real.tan (α + Real.pi) - Real.cos (-α) + Real.sin (Real.pi / 2 - α)) = 1 := by
  sorry

end trig_expression_simplification_l1870_187042


namespace losing_position_characterization_l1870_187094

/-- Represents the state of the table-folding game -/
structure GameState where
  n : ℕ
  m : ℕ

/-- Predicate to determine if a game state is a losing position -/
def is_losing_position (state : GameState) : Prop :=
  ∃ k : ℕ, state.m = (state.n + 1) * 2^k - 1

/-- The main theorem stating the characterization of losing positions -/
theorem losing_position_characterization (state : GameState) :
  is_losing_position state ↔ 
  (∀ fold : ℕ, fold > 0 → fold ≤ state.m → 
    ¬is_losing_position ⟨state.n, state.m - fold⟩) ∧
  (∀ fold : ℕ, fold > 0 → fold ≤ state.n → 
    ¬is_losing_position ⟨state.n - fold, state.m⟩) :=
sorry

end losing_position_characterization_l1870_187094


namespace horse_grazing_area_l1870_187008

/-- The area over which a horse can graze when tethered to one corner of a rectangular field --/
theorem horse_grazing_area (field_length : ℝ) (field_width : ℝ) (rope_length : ℝ) 
    (h1 : field_length = 40)
    (h2 : field_width = 24)
    (h3 : rope_length = 14)
    (h4 : rope_length ≤ field_length / 2)
    (h5 : rope_length ≤ field_width / 2) :
  (1/4 : ℝ) * Real.pi * rope_length^2 = 49 * Real.pi := by
  sorry

#check horse_grazing_area

end horse_grazing_area_l1870_187008


namespace number_equation_l1870_187022

theorem number_equation (x : ℝ) : 38 + 2 * x = 124 ↔ x = 43 := by
  sorry

end number_equation_l1870_187022


namespace complex_power_four_l1870_187029

theorem complex_power_four (i : ℂ) : i^2 = -1 → 2 * i^4 = 2 := by sorry

end complex_power_four_l1870_187029


namespace circle_tangent_implies_m_equals_9_l1870_187039

/-- Circle C with equation x^2 + y^2 - 6x - 8y + m = 0 -/
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 8*y + m = 0

/-- Unit circle with equation x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : Prop :=
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2

/-- Main theorem: If circle C is externally tangent to the unit circle, then m = 9 -/
theorem circle_tangent_implies_m_equals_9 (m : ℝ) :
  (∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_C m x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    externally_tangent center (0, 0) radius 1) →
  m = 9 :=
sorry

end circle_tangent_implies_m_equals_9_l1870_187039


namespace partnership_capital_share_l1870_187083

theorem partnership_capital_share (T : ℝ) (x : ℝ) : 
  (x + 1/4 + 1/5 + (11/20 - x) = 1) →  -- Total shares add up to 1
  (810 / 2430 = x) →                   -- A's profit share equals capital share
  (x = 1/3) :=                         -- A's capital share is 1/3
by sorry

end partnership_capital_share_l1870_187083


namespace sqrt_sum_inequality_l1870_187036

theorem sqrt_sum_inequality (x y α : ℝ) (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) : 
  x + y ≥ 2 * α := by
  sorry

end sqrt_sum_inequality_l1870_187036


namespace square_difference_equality_l1870_187053

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end square_difference_equality_l1870_187053


namespace sum_of_abc_l1870_187013

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (eq1 : a^2 + b*c = 115)
  (eq2 : b^2 + a*c = 127)
  (eq3 : c^2 + a*b = 115) :
  a + b + c = 22 := by
sorry

end sum_of_abc_l1870_187013


namespace parallel_lines_ratio_l1870_187096

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  L1 : ℝ → ℝ → Prop
  L2 : ℝ → ℝ → Prop
  a : ℝ
  c : ℝ
  c_pos : c > 0
  is_parallel : ∀ x y, L1 x y ↔ x - y + 1 = 0
  L2_eq : ∀ x y, L2 x y ↔ 3*x + a*y - c = 0
  distance : ℝ

/-- The theorem stating the value of (a-3)/c for the given parallel lines -/
theorem parallel_lines_ratio (lines : ParallelLines) 
  (h_dist : lines.distance = Real.sqrt 2) : 
  (lines.a - 3) / lines.c = -2 := by sorry

end parallel_lines_ratio_l1870_187096


namespace point_not_on_transformed_plane_l1870_187018

/-- Similarity transformation of a plane with coefficient k and center at the origin -/
def transform_plane (a b c d k : ℝ) : ℝ → ℝ → ℝ → Prop :=
  fun x y z ↦ a * x + b * y + c * z + k * d = 0

/-- The point A -/
def A : ℝ × ℝ × ℝ := (-1, 2, 3)

/-- The original plane equation -/
def plane_a : ℝ → ℝ → ℝ → Prop :=
  fun x y z ↦ x - 3 * y + z + 2 = 0

/-- The similarity transformation coefficient -/
def k : ℝ := 2.5

theorem point_not_on_transformed_plane :
  ¬ transform_plane 1 (-3) 1 2 k A.1 A.2.1 A.2.2 :=
sorry

end point_not_on_transformed_plane_l1870_187018


namespace part_one_part_two_l1870_187057

-- Define the inequality function
def inequality (a x : ℝ) : Prop := (a * x - 1) * (x + 1) > 0

-- Part 1
theorem part_one : 
  (∀ x : ℝ, inequality (-2) x ↔ -1 < x ∧ x < -1/2) :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x : ℝ, inequality a x ↔ 
    (a < -1 ∧ -1 < x ∧ x < 1/a) ∨
    (a = -1 ∧ False) ∨
    (-1 < a ∧ a < 0 ∧ 1/a < x ∧ x < -1) ∨
    (a = 0 ∧ x < -1) ∨
    (a > 0 ∧ (x < -1 ∨ x > 1/a))) :=
sorry

end part_one_part_two_l1870_187057


namespace expand_a_expand_b_expand_c_expand_d_expand_e_l1870_187095

-- Define variables
variable (x y m n : ℝ)

-- Theorem for expression (a)
theorem expand_a : (x + 3*y)^2 = x^2 + 6*x*y + 9*y^2 := by sorry

-- Theorem for expression (b)
theorem expand_b : (2*x + 3*y)^2 = 4*x^2 + 12*x*y + 9*y^2 := by sorry

-- Theorem for expression (c)
theorem expand_c : (m^3 + n^5)^2 = m^6 + 2*m^3*n^5 + n^10 := by sorry

-- Theorem for expression (d)
theorem expand_d : (5*x - 3*y)^2 = 25*x^2 - 30*x*y + 9*y^2 := by sorry

-- Theorem for expression (e)
theorem expand_e : (3*m^5 - 4*n^2)^2 = 9*m^10 - 24*m^5*n^2 + 16*n^4 := by sorry

end expand_a_expand_b_expand_c_expand_d_expand_e_l1870_187095


namespace quadratic_monotone_increasing_condition_l1870_187061

/-- A quadratic function f(x) = ax^2 + bx + c is monotonically increasing on [1, +∞)
    if and only if b ≥ -2a, where a > 0. -/
theorem quadratic_monotone_increasing_condition (a b c : ℝ) (ha : a > 0) :
  (∀ x y, x ∈ Set.Ici (1 : ℝ) → y ∈ Set.Ici (1 : ℝ) → x ≤ y →
    a * x^2 + b * x + c ≤ a * y^2 + b * y + c) ↔
  b ≥ -2 * a :=
sorry

end quadratic_monotone_increasing_condition_l1870_187061


namespace lcm_gcd_product_l1870_187011

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  sorry

end lcm_gcd_product_l1870_187011


namespace complement_of_A_in_U_l1870_187090

def U : Finset Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Finset Nat := {1, 2, 5, 7}

theorem complement_of_A_in_U : 
  (U \ A : Finset Nat) = {3, 4, 6} := by sorry

end complement_of_A_in_U_l1870_187090


namespace typing_problem_solution_l1870_187091

/-- Represents the typing speed and time for two typists -/
structure TypistData where
  x : ℝ  -- Time taken by first typist to type entire manuscript
  y : ℝ  -- Time taken by second typist to type entire manuscript

/-- Checks if the given typing times satisfy the manuscript typing conditions -/
def satisfiesConditions (d : TypistData) : Prop :=
  let totalPages : ℝ := 80
  let pagesTypedIn5Hours : ℝ := 65
  let timeDiff : ℝ := 3
  (totalPages / d.y - totalPages / d.x = timeDiff) ∧
  (5 * (totalPages / d.x + totalPages / d.y) = pagesTypedIn5Hours)

/-- Theorem stating the solution to the typing problem -/
theorem typing_problem_solution :
  ∃ d : TypistData, satisfiesConditions d ∧ d.x = 10 ∧ d.y = 16 := by
  sorry


end typing_problem_solution_l1870_187091


namespace jade_cal_difference_l1870_187067

/-- The number of transactions handled by different people on Thursday -/
def thursday_transactions : ℕ → ℕ
| 0 => 90  -- Mabel's transactions
| 1 => (110 * thursday_transactions 0) / 100  -- Anthony's transactions
| 2 => (2 * thursday_transactions 1) / 3  -- Cal's transactions
| 3 => 84  -- Jade's transactions
| _ => 0

/-- The theorem stating the difference between Jade's and Cal's transactions -/
theorem jade_cal_difference : 
  thursday_transactions 3 - thursday_transactions 2 = 18 :=
sorry

end jade_cal_difference_l1870_187067


namespace nancy_quarters_l1870_187030

/-- The number of quarters Nancy has -/
def number_of_quarters (total_amount : ℚ) (quarter_value : ℚ) : ℚ :=
  total_amount / quarter_value

theorem nancy_quarters : 
  let total_amount : ℚ := 3
  let quarter_value : ℚ := 1/4
  number_of_quarters total_amount quarter_value = 12 := by
sorry

end nancy_quarters_l1870_187030


namespace age_difference_l1870_187037

theorem age_difference (A B C : ℤ) 
  (h1 : A + B = B + C + 15) 
  (h2 : C = A - 15) : 
  (A + B) - (B + C) = 15 := by
sorry

end age_difference_l1870_187037


namespace function_coefficient_l1870_187006

theorem function_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x^3 - 2*x) →
  f (-1) = 4 →
  a = -2 := by
sorry

end function_coefficient_l1870_187006


namespace power_function_inequality_l1870_187089

-- Define the power function
def f (x : ℝ) : ℝ := x^(4/5)

-- State the theorem
theorem power_function_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) :
  f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2 := by
  sorry

end power_function_inequality_l1870_187089


namespace minimum_area_is_14_l1870_187062

-- Define the variation ranges
def normal_variation : ℝ := 0.5
def approximate_variation : ℝ := 1.0

-- Define the reported dimensions
def reported_length : ℝ := 4.0
def reported_width : ℝ := 5.0

-- Define the actual minimum dimensions
def min_length : ℝ := reported_length - normal_variation
def min_width : ℝ := reported_width - approximate_variation

-- Define the minimum area
def min_area : ℝ := min_length * min_width

-- Theorem statement
theorem minimum_area_is_14 : min_area = 14 := by
  sorry

end minimum_area_is_14_l1870_187062


namespace shorter_lateral_side_length_l1870_187058

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- One angle of the trapezoid -/
  angle : ℝ
  /-- The midline (median) of the trapezoid -/
  midline : ℝ
  /-- One of the bases of the trapezoid -/
  base : ℝ
  /-- The angle is 30 degrees -/
  angle_is_30 : angle = 30
  /-- The lines containing the lateral sides intersect at a right angle -/
  lateral_sides_right_angle : True
  /-- The midline is 10 -/
  midline_is_10 : midline = 10
  /-- One base is 8 -/
  base_is_8 : base = 8

/-- The theorem stating the length of the shorter lateral side -/
theorem shorter_lateral_side_length (t : SpecialTrapezoid) : 
  ∃ (shorter_side : ℝ), shorter_side = 2 := by
  sorry

end shorter_lateral_side_length_l1870_187058


namespace cherry_price_theorem_l1870_187015

/-- The price of a bag of cherries satisfies the given conditions -/
theorem cherry_price_theorem (olive_price : ℝ) (bag_count : ℕ) (discount_rate : ℝ) (final_cost : ℝ) :
  olive_price = 7 →
  bag_count = 50 →
  discount_rate = 0.1 →
  final_cost = 540 →
  ∃ (cherry_price : ℝ),
    cherry_price = 5 ∧
    (1 - discount_rate) * (bag_count * cherry_price + bag_count * olive_price) = final_cost :=
by sorry

end cherry_price_theorem_l1870_187015


namespace circle_center_sum_l1870_187035

/-- Given a circle with equation x² + y² = 4x - 6y + 9, 
    prove that the sum of the coordinates of its center is -1 -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9) ∧ h + k = -1) :=
by sorry

end circle_center_sum_l1870_187035


namespace pizza_bill_friends_l1870_187049

theorem pizza_bill_friends (total_price : ℕ) (price_per_person : ℕ) (bob_included : Bool) : 
  total_price = 40 → price_per_person = 8 → bob_included = true → 
  (total_price / price_per_person) - 1 = 4 := by
  sorry

end pizza_bill_friends_l1870_187049


namespace constant_sum_sequence_2013_l1870_187028

/-- A sequence where the sum of any three consecutive terms is constant -/
def ConstantSumSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n + a (n + 1) + a (n + 2) = a (n + 1) + a (n + 2) + a (n + 3)

theorem constant_sum_sequence_2013 (a : ℕ → ℝ) (x : ℝ) 
    (h_constant_sum : ConstantSumSequence a)
    (h_a3 : a 3 = x)
    (h_a999 : a 999 = 3 - 2*x) :
    a 2013 = 1 := by
  sorry

end constant_sum_sequence_2013_l1870_187028


namespace bicycle_distance_theorem_l1870_187097

/-- Represents a bicycle wheel -/
structure Wheel where
  perimeter : ℝ

/-- Represents a bicycle with two wheels -/
structure Bicycle where
  backWheel : Wheel
  frontWheel : Wheel

/-- Calculates the distance traveled by a wheel given the number of revolutions -/
def distanceTraveled (wheel : Wheel) (revolutions : ℝ) : ℝ :=
  wheel.perimeter * revolutions

theorem bicycle_distance_theorem (bike : Bicycle) 
    (h1 : bike.backWheel.perimeter = 9)
    (h2 : bike.frontWheel.perimeter = 7)
    (h3 : ∃ (r : ℝ), distanceTraveled bike.backWheel r = distanceTraveled bike.frontWheel (r + 10)) :
  ∃ (d : ℝ), d = 315 ∧ ∃ (r : ℝ), d = distanceTraveled bike.backWheel r ∧ d = distanceTraveled bike.frontWheel (r + 10) := by
  sorry

end bicycle_distance_theorem_l1870_187097


namespace alice_profit_l1870_187045

def total_bracelets : ℕ := 52
def design_a_bracelets : ℕ := 30
def design_b_bracelets : ℕ := 22
def cost_a : ℚ := 2
def cost_b : ℚ := 4.5
def given_away_a : ℕ := 5
def given_away_b : ℕ := 3
def sell_price_a : ℚ := 0.25
def sell_price_b : ℚ := 0.5

def total_cost : ℚ := design_a_bracelets * cost_a + design_b_bracelets * cost_b
def remaining_a : ℕ := design_a_bracelets - given_away_a
def remaining_b : ℕ := design_b_bracelets - given_away_b
def total_revenue : ℚ := remaining_a * sell_price_a + remaining_b * sell_price_b
def profit : ℚ := total_revenue - total_cost

theorem alice_profit :
  profit = -143.25 :=
sorry

end alice_profit_l1870_187045


namespace remainder_98_power_50_mod_100_l1870_187079

theorem remainder_98_power_50_mod_100 : 98^50 % 100 = 24 := by
  sorry

end remainder_98_power_50_mod_100_l1870_187079


namespace parabola_transformation_l1870_187084

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { f := λ x => p.f (x - h) }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { f := λ x => p.f x + v }

/-- The original parabola y = 3x² -/
def original_parabola : Parabola :=
  { f := λ x => 3 * x^2 }

/-- The final parabola after transformations -/
def final_parabola : Parabola :=
  { f := λ x => 3 * (x + 1)^2 - 2 }

theorem parabola_transformation :
  (shift_vertical (shift_horizontal original_parabola 1) (-2)).f = final_parabola.f :=
by sorry

end parabola_transformation_l1870_187084


namespace no_one_left_behind_l1870_187075

/-- Represents the Ferris wheel problem -/
structure FerrisWheel where
  seats_per_rotation : ℕ
  total_rotations : ℕ
  initial_queue : ℕ
  impatience_rate : ℚ

/-- Calculates the number of people remaining in the queue after a given number of rotations -/
def people_remaining (fw : FerrisWheel) (rotations : ℕ) : ℕ :=
  sorry

/-- The main theorem: proves that no one is left in the queue after three rotations -/
theorem no_one_left_behind (fw : FerrisWheel) 
  (h1 : fw.seats_per_rotation = 56)
  (h2 : fw.total_rotations = 3)
  (h3 : fw.initial_queue = 92)
  (h4 : fw.impatience_rate = 1/10) :
  people_remaining fw 3 = 0 :=
sorry

end no_one_left_behind_l1870_187075


namespace ceiling_squared_negative_fraction_l1870_187000

theorem ceiling_squared_negative_fraction : ⌈((-7/4 : ℚ)^2)⌉ = 4 := by sorry

end ceiling_squared_negative_fraction_l1870_187000


namespace cone_sphere_volume_l1870_187020

/-- Given a cone with lateral surface forming a semicircle of radius 2√3 when unrolled,
    and its vertex and base circumference lying on a sphere O,
    prove that the volume of sphere O is 32π/3 -/
theorem cone_sphere_volume (l : ℝ) (r : ℝ) (h : ℝ) (R : ℝ) :
  l = 2 * Real.sqrt 3 →                  -- lateral surface radius
  r = l / 2 →                            -- base radius
  h^2 + r^2 = l^2 →                      -- Pythagorean theorem
  2 * R * h = l^2 →                      -- sphere diameter relation
  (4 / 3) * π * R^3 = (32 * π) / 3 :=    -- sphere volume
by sorry

end cone_sphere_volume_l1870_187020


namespace no_rain_no_snow_probability_l1870_187086

theorem no_rain_no_snow_probability
  (rain_prob : ℚ)
  (snow_prob : ℚ)
  (rain_prob_def : rain_prob = 4 / 10)
  (snow_prob_def : snow_prob = 1 / 5)
  (events_independent : True) :
  (1 - rain_prob) * (1 - snow_prob) = 12 / 25 := by
  sorry

end no_rain_no_snow_probability_l1870_187086


namespace visitor_difference_l1870_187066

def visitors_previous_day : ℕ := 100
def visitors_that_day : ℕ := 666

theorem visitor_difference : visitors_that_day - visitors_previous_day = 566 := by
  sorry

end visitor_difference_l1870_187066


namespace business_trip_distance_l1870_187034

theorem business_trip_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 8 →
  speed1 = 70 →
  speed2 = 85 →
  (total_time / 2 * speed1) + (total_time / 2 * speed2) = 620 := by
  sorry

end business_trip_distance_l1870_187034


namespace forest_area_relationship_l1870_187065

/-- 
Given forest areas a, b, c for three consecutive years,
with constant growth rate in the last two years,
prove that ac = b²
-/
theorem forest_area_relationship (a b c : ℝ) 
  (h : ∃ x : ℝ, b = a * (1 + x) ∧ c = b * (1 + x)) : 
  a * c = b ^ 2 := by
  sorry

end forest_area_relationship_l1870_187065


namespace cube_sum_equation_l1870_187027

theorem cube_sum_equation (y : ℝ) (h : y^3 + 4 / y^3 = 110) : y + 4 / y = 6 := by
  sorry

end cube_sum_equation_l1870_187027


namespace hall_tables_l1870_187004

theorem hall_tables (total_chairs : ℕ) (tables_with_three : ℕ) : 
  total_chairs = 91 → tables_with_three = 5 →
  ∃ (total_tables : ℕ), 
    (total_tables / 2 : ℚ) * 2 + 
    (tables_with_three : ℚ) * 3 + 
    ((total_tables : ℚ) - (total_tables / 2 : ℚ) - (tables_with_three : ℚ)) * 4 = 
    total_chairs ∧ 
    total_tables = 32 := by
  sorry

end hall_tables_l1870_187004


namespace tangent_ratio_equals_three_l1870_187098

theorem tangent_ratio_equals_three (α : Real) 
  (h : Real.tan α = 2 * Real.tan (π / 5)) : 
  Real.cos (α - 3 * π / 10) / Real.sin (α - π / 5) = 3 := by
  sorry

end tangent_ratio_equals_three_l1870_187098


namespace fraction_to_decimal_l1870_187088

theorem fraction_to_decimal : (3 : ℚ) / 80 = 0.0375 := by
  sorry

end fraction_to_decimal_l1870_187088


namespace gcf_of_90_and_105_l1870_187003

theorem gcf_of_90_and_105 : Nat.gcd 90 105 = 15 := by
  sorry

end gcf_of_90_and_105_l1870_187003


namespace nine_digit_prime_square_product_l1870_187078

/-- Represents a nine-digit number of the form a₁a₂a₃b₁b₂b₃a₁a₂a₃ --/
def NineDigitNumber (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : ℕ :=
  a₁ * 100000000 + a₂ * 10000000 + a₃ * 1000000 + 
  b₁ * 100000 + b₂ * 10000 + b₃ * 1000 + 
  a₁ * 100 + a₂ * 10 + a₃

/-- Condition: ⎯⎯⎯⎯⎯b₁b₂b₃ = 2 * ⎯⎯⎯⎯⎯(a₁a₂a₃) --/
def MiddleIsDoubleFirst (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : Prop :=
  b₁ * 100 + b₂ * 10 + b₃ = 2 * (a₁ * 100 + a₂ * 10 + a₃)

/-- The number is the product of the squares of four different prime numbers --/
def IsProductOfFourPrimeSquares (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁^2 * p₂^2 * p₃^2 * p₄^2

theorem nine_digit_prime_square_product :
  ∃ a₁ a₂ a₃ b₁ b₂ b₃ : ℕ,
    a₁ ≠ 0 ∧
    MiddleIsDoubleFirst a₁ a₂ a₃ b₁ b₂ b₃ ∧
    IsProductOfFourPrimeSquares (NineDigitNumber a₁ a₂ a₃ b₁ b₂ b₃) :=
by sorry

end nine_digit_prime_square_product_l1870_187078


namespace marks_money_theorem_l1870_187044

/-- The amount of money Mark's father gave him. -/
def fathers_money : ℕ := 85

/-- The number of books Mark bought. -/
def num_books : ℕ := 10

/-- The cost of each book in dollars. -/
def book_cost : ℕ := 5

/-- The amount of money Mark has left after buying the books. -/
def money_left : ℕ := 35

/-- Theorem stating that the amount of money Mark's father gave him is correct. -/
theorem marks_money_theorem :
  fathers_money = num_books * book_cost + money_left :=
by sorry

end marks_money_theorem_l1870_187044


namespace point_symmetry_l1870_187063

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the x-axis -/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- Symmetry about the y-axis -/
def symmetricAboutYAxis (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = q.y

/-- The main theorem -/
theorem point_symmetry (M N P : Point) :
  symmetricAboutXAxis M P →
  symmetricAboutYAxis N M →
  N = Point.mk 1 2 →
  P = Point.mk (-1) (-2) := by
  sorry

end point_symmetry_l1870_187063


namespace prob_white_given_popped_l1870_187077

/-- Represents the color of a kernel -/
inductive KernelColor
  | White
  | Yellow
  | Blue

/-- The probability of selecting a kernel of a given color -/
def selectProb (c : KernelColor) : ℚ :=
  match c with
  | KernelColor.White => 2/5
  | KernelColor.Yellow => 1/5
  | KernelColor.Blue => 2/5

/-- The probability of a kernel popping given its color -/
def popProb (c : KernelColor) : ℚ :=
  match c with
  | KernelColor.White => 1/4
  | KernelColor.Yellow => 3/4
  | KernelColor.Blue => 1/2

/-- The probability that a randomly selected kernel that popped was white -/
theorem prob_white_given_popped :
  (selectProb KernelColor.White * popProb KernelColor.White) /
  (selectProb KernelColor.White * popProb KernelColor.White +
   selectProb KernelColor.Yellow * popProb KernelColor.Yellow +
   selectProb KernelColor.Blue * popProb KernelColor.Blue) = 2/9 := by
  sorry


end prob_white_given_popped_l1870_187077


namespace train_length_calculation_l1870_187048

/-- Calculates the length of a train given its speed and time to cross a point -/
theorem train_length_calculation (speed_km_hr : ℝ) (time_seconds : ℝ) : 
  speed_km_hr = 144 →
  time_seconds = 0.9999200063994881 →
  ∃ (length_meters : ℝ), abs (length_meters - 39.997) < 0.001 := by
  sorry

end train_length_calculation_l1870_187048


namespace total_chairs_is_528_l1870_187019

/-- Calculates the total number of chairs carried to the hall by Kingsley and her friends -/
def total_chairs : ℕ :=
  let kingsley_chairs := 7
  let friend_chairs := [6, 8, 5, 9, 7]
  let trips := List.range 6 |>.map (λ i => 10 + i)
  (kingsley_chairs :: friend_chairs).zip trips
  |>.map (λ (chairs, trip) => chairs * trip)
  |>.sum

/-- Theorem stating that the total number of chairs carried is 528 -/
theorem total_chairs_is_528 : total_chairs = 528 := by
  sorry

end total_chairs_is_528_l1870_187019


namespace dividend_calculation_l1870_187080

theorem dividend_calculation (dividend divisor : ℕ) : 
  dividend / divisor = 15 →
  dividend % divisor = 5 →
  dividend + divisor + 15 + 5 = 2169 →
  dividend = 2015 := by
sorry

end dividend_calculation_l1870_187080


namespace polar_equations_and_intersection_range_l1870_187070

-- Define the line l
def line_l (x : ℝ) : Prop := x = 2

-- Define the curve C
def curve_C (x y α : ℝ) : Prop := x = Real.cos α ∧ y = 1 + Real.sin α

-- Define the polar coordinates
def polar_coords (x y ρ θ : ℝ) : Prop := x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- State the theorem
theorem polar_equations_and_intersection_range :
  ∀ (x y ρ θ α β : ℝ),
  (0 < β ∧ β < Real.pi / 2) →
  (line_l x →
    ∃ (ρ_l : ℝ), polar_coords x y ρ_l θ ∧ ρ_l * Real.cos θ = 2) ∧
  (curve_C x y α →
    ∃ (ρ_c : ℝ), polar_coords x y ρ_c θ ∧ ρ_c = 2 * Real.sin θ) ∧
  (∃ (ρ_p ρ_m : ℝ),
    polar_coords x y ρ_p β ∧
    curve_C x y α ∧
    polar_coords x y ρ_m β ∧
    line_l x ∧
    0 < ρ_p / ρ_m ∧ ρ_p / ρ_m ≤ 1 / 2) :=
by sorry

end polar_equations_and_intersection_range_l1870_187070


namespace largest_prime_factor_of_7_factorial_plus_8_factorial_l1870_187050

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_prime_factor (n : ℕ) : ℕ :=
  (Nat.factors n).foldl max 0

theorem largest_prime_factor_of_7_factorial_plus_8_factorial :
  largest_prime_factor (factorial 7 + factorial 8) = 7 := by
  sorry

end largest_prime_factor_of_7_factorial_plus_8_factorial_l1870_187050


namespace double_acute_angle_less_than_180_l1870_187059

theorem double_acute_angle_less_than_180 (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  2 * α < Real.pi := by
  sorry

end double_acute_angle_less_than_180_l1870_187059


namespace gift_card_balance_l1870_187021

/-- Calculates the remaining balance on a gift card after a coffee purchase -/
theorem gift_card_balance 
  (gift_card_amount : ℝ) 
  (coffee_price_per_pound : ℝ) 
  (pounds_purchased : ℝ) 
  (h1 : gift_card_amount = 70) 
  (h2 : coffee_price_per_pound = 8.58) 
  (h3 : pounds_purchased = 4) : 
  gift_card_amount - (coffee_price_per_pound * pounds_purchased) = 35.68 := by
sorry

end gift_card_balance_l1870_187021


namespace abs_neg_2023_l1870_187073

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_neg_2023_l1870_187073


namespace perimeter_bounds_l1870_187087

/-- A quadrilateral inscribed in a circle with specific properties -/
structure InscribedQuadrilateral where
  AB : ℕ+
  BC : ℕ+
  CD : ℕ+
  DA : ℕ+
  DA_eq_2005 : DA = 2005
  right_angles : True  -- Represents ∠ABC = ∠ADC = 90°
  max_side_lt_2005 : max AB BC < 2005 ∧ max (max AB BC) CD < 2005

/-- The perimeter of the quadrilateral -/
def perimeter (q : InscribedQuadrilateral) : ℕ :=
  q.AB.val + q.BC.val + q.CD.val + q.DA.val

/-- Theorem stating the bounds on the perimeter -/
theorem perimeter_bounds (q : InscribedQuadrilateral) :
  4160 ≤ perimeter q ∧ perimeter q ≤ 7772 := by
  sorry

end perimeter_bounds_l1870_187087


namespace geometric_sequence_product_l1870_187007

/-- Given a geometric sequence {a_n} where a_2 and a_3 are the roots of x^2 - x - 2013 = 0,
    prove that a_1 * a_4 = -2013 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n) →  -- geometric sequence condition
  a 2^2 - a 2 - 2013 = 0 →  -- a_2 is a root
  a 3^2 - a 3 - 2013 = 0 →  -- a_3 is a root
  a 1 * a 4 = -2013 := by
sorry

end geometric_sequence_product_l1870_187007


namespace arithmetic_sequence_cosines_l1870_187010

theorem arithmetic_sequence_cosines (a : ℝ) : 
  (0 < a) ∧ (a < 2 * Real.pi) ∧ 
  (∃ d : ℝ, (Real.cos (2 * a) = Real.cos a + d) ∧ 
            (Real.cos (3 * a) = Real.cos (2 * a) + d)) ↔ 
  (a = Real.pi / 4) ∨ (a = 3 * Real.pi / 4) ∨ 
  (a = 5 * Real.pi / 4) ∨ (a = 7 * Real.pi / 4) :=
by sorry

#check arithmetic_sequence_cosines

end arithmetic_sequence_cosines_l1870_187010


namespace integer_solution_l1870_187043

theorem integer_solution (n : ℤ) : 
  n + 15 > 16 ∧ 4 * n < 20 ∧ |n - 2| ≤ 2 → n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

end integer_solution_l1870_187043


namespace fraction_simplification_l1870_187046

theorem fraction_simplification (b y θ : ℝ) (h : b^2 + y^2 ≠ 0) :
  (Real.sqrt (b^2 + y^2) + (y^2 - b^2) / Real.sqrt (b^2 + y^2) * Real.cos θ) / (b^2 + y^2) =
  (b^2 * (Real.sqrt (b^2 + y^2) - Real.cos θ) + y^2 * (Real.sqrt (b^2 + y^2) + Real.cos θ)) /
  (b^2 + y^2)^(3/2) := by
  sorry

end fraction_simplification_l1870_187046


namespace angle_measure_l1870_187026

theorem angle_measure (A : ℝ) : 
  (90 - A = (180 - A) / 3 - 10) → A = 60 := by
  sorry

end angle_measure_l1870_187026


namespace hyperbola_equation_l1870_187001

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def asymptote (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

def parabola (x y : ℝ) : Prop :=
  y^2 = 24 * x

def directrix (x : ℝ) : Prop :=
  x = -6

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y : ℝ, hyperbola a b x y ∧ asymptote x y) →
  (∃ x : ℝ, directrix x ∧ ∃ y : ℝ, hyperbola a b x y) →
  (∀ x y : ℝ, hyperbola a b x y ↔ hyperbola 3 (Real.sqrt 27) x y) :=
sorry

end hyperbola_equation_l1870_187001


namespace michaels_boxes_l1870_187014

/-- Given that Michael has 16 blocks and each box must contain 2 blocks, 
    prove that the number of boxes Michael has is 8. -/
theorem michaels_boxes (total_blocks : ℕ) (blocks_per_box : ℕ) (h1 : total_blocks = 16) (h2 : blocks_per_box = 2) :
  total_blocks / blocks_per_box = 8 := by
  sorry


end michaels_boxes_l1870_187014


namespace exist_three_quadratics_with_specific_root_properties_l1870_187051

theorem exist_three_quadratics_with_specific_root_properties :
  ∃ (p₁ p₂ p₃ : ℝ → ℝ),
    (∃ x₁, p₁ x₁ = 0) ∧
    (∃ x₂, p₂ x₂ = 0) ∧
    (∃ x₃, p₃ x₃ = 0) ∧
    (∀ x, p₁ x + p₂ x ≠ 0) ∧
    (∀ x, p₂ x + p₃ x ≠ 0) ∧
    (∀ x, p₁ x + p₃ x ≠ 0) ∧
    (∀ x, p₁ x = (x - 1)^2) ∧
    (∀ x, p₂ x = x^2) ∧
    (∀ x, p₃ x = (x - 2)^2) :=
by
  sorry


end exist_three_quadratics_with_specific_root_properties_l1870_187051


namespace total_legs_sea_creatures_l1870_187009

/-- Calculate the total number of legs for sea creatures --/
theorem total_legs_sea_creatures :
  let num_octopuses : ℕ := 5
  let num_crabs : ℕ := 3
  let num_starfish : ℕ := 2
  let legs_per_octopus : ℕ := 8
  let legs_per_crab : ℕ := 10
  let legs_per_starfish : ℕ := 5
  num_octopuses * legs_per_octopus +
  num_crabs * legs_per_crab +
  num_starfish * legs_per_starfish = 80 :=
by sorry

end total_legs_sea_creatures_l1870_187009


namespace book_sale_revenue_l1870_187032

theorem book_sale_revenue (total_books : ℕ) (sold_price : ℕ) (remaining_books : ℕ) : 
  (2 : ℚ) / 3 * total_books = total_books - remaining_books ∧
  remaining_books = 50 ∧
  sold_price = 5 →
  (total_books - remaining_books) * sold_price = 500 := by
  sorry

end book_sale_revenue_l1870_187032


namespace john_incentive_calculation_l1870_187047

/-- The incentive calculation for John's agency fees --/
theorem john_incentive_calculation 
  (commission : ℕ) 
  (advance_fees : ℕ) 
  (amount_given : ℕ) 
  (h1 : commission = 25000)
  (h2 : advance_fees = 8280)
  (h3 : amount_given = 18500) :
  amount_given - (commission - advance_fees) = 1780 :=
by sorry

end john_incentive_calculation_l1870_187047


namespace inequality_proof_l1870_187060

theorem inequality_proof (a b c : ℝ) : 
  a = Real.sqrt ((1 - Real.cos (110 * π / 180)) / 2) →
  b = (Real.sqrt 2 / 2) * (Real.sin (20 * π / 180) + Real.cos (20 * π / 180)) →
  c = (1 + Real.tan (20 * π / 180)) / (1 - Real.tan (20 * π / 180)) →
  a < b ∧ b < c := by
  sorry

end inequality_proof_l1870_187060


namespace reciprocal_of_sum_l1870_187002

theorem reciprocal_of_sum : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by sorry

end reciprocal_of_sum_l1870_187002


namespace quadratic_inequality_coefficient_sum_l1870_187052

/-- Given a quadratic inequality x^2 - ax + b < 0 with solution set {x | 1 < x < 2},
    prove that the sum of coefficients a and b is equal to 5. -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2) → a + b = 5 := by
  sorry

end quadratic_inequality_coefficient_sum_l1870_187052
