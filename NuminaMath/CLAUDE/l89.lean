import Mathlib

namespace NUMINAMATH_CALUDE_square_field_area_l89_8985

/-- Prove that a square field with the given conditions has an area of 27889 square meters -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 1.4 →
  total_cost = 932.4 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    total_cost = wire_cost_per_meter * (4 * side_length - num_gates * gate_width) ∧
    side_length^2 = 27889 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l89_8985


namespace NUMINAMATH_CALUDE_peters_claim_impossible_l89_8997

/-- Represents the shooting scenario with initial bullets, shots made, and successful hits -/
structure ShootingScenario where
  initialBullets : ℕ
  shotsMade : ℕ
  successfulHits : ℕ

/-- Calculates the total number of bullets available after successful hits -/
def totalBullets (s : ShootingScenario) : ℕ :=
  s.initialBullets + s.successfulHits * 5

/-- Defines when a shooting scenario is possible -/
def isPossible (s : ShootingScenario) : Prop :=
  totalBullets s ≥ s.shotsMade

/-- Theorem stating that Peter's claim is impossible -/
theorem peters_claim_impossible :
  ¬ isPossible ⟨5, 50, 8⟩ := by
  sorry


end NUMINAMATH_CALUDE_peters_claim_impossible_l89_8997


namespace NUMINAMATH_CALUDE_range_of_a_l89_8998

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x > a - x^2) → a < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l89_8998


namespace NUMINAMATH_CALUDE_inverse_inequality_l89_8919

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l89_8919


namespace NUMINAMATH_CALUDE_volleyball_starters_count_l89_8936

def volleyball_team_size : ℕ := 14
def triplet_size : ℕ := 3
def starter_size : ℕ := 6

def choose_starters (team_size triplet_size starter_size : ℕ) : ℕ :=
  let non_triplet_size := team_size - triplet_size
  let remaining_spots := starter_size - 2
  triplet_size * Nat.choose non_triplet_size remaining_spots

theorem volleyball_starters_count :
  choose_starters volleyball_team_size triplet_size starter_size = 990 :=
sorry

end NUMINAMATH_CALUDE_volleyball_starters_count_l89_8936


namespace NUMINAMATH_CALUDE_circle_equation_circle_properties_l89_8949

theorem circle_equation (x y : ℝ) : 
  (x^2 + y^2 + 2*x - 4*y - 6 = 0) ↔ ((x + 1)^2 + (y - 2)^2 = 11) :=
by sorry

theorem circle_properties :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧ 
    radius = Real.sqrt 11 ∧
    ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 6 = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_circle_properties_l89_8949


namespace NUMINAMATH_CALUDE_school_transfer_percentage_l89_8901

/-- Proves the percentage of students from school A going to school C -/
theorem school_transfer_percentage
  (total_students : ℕ)
  (school_A_percentage : ℚ)
  (school_B_to_C_percentage : ℚ)
  (total_to_C_percentage : ℚ)
  (h1 : school_A_percentage = 60 / 100)
  (h2 : school_B_to_C_percentage = 40 / 100)
  (h3 : total_to_C_percentage = 34 / 100)
  : ∃ (school_A_to_C_percentage : ℚ),
    school_A_to_C_percentage = 30 / 100 ∧
    (school_A_percentage * total_students * school_A_to_C_percentage +
     (1 - school_A_percentage) * total_students * school_B_to_C_percentage =
     total_students * total_to_C_percentage) := by
  sorry


end NUMINAMATH_CALUDE_school_transfer_percentage_l89_8901


namespace NUMINAMATH_CALUDE_adjusted_smallest_part_is_correct_l89_8931

-- Define the total amount
def total : ℚ := 100

-- Define the proportions
def proportions : List ℚ := [1, 3, 4, 6]

-- Define the extra amount added to the smallest part
def extra : ℚ := 12

-- Define the function to calculate the adjusted smallest part
def adjusted_smallest_part (total : ℚ) (proportions : List ℚ) (extra : ℚ) : ℚ :=
  let sum_proportions := proportions.sum
  let smallest_part := total * (proportions.head! / sum_proportions)
  smallest_part + extra

-- Theorem statement
theorem adjusted_smallest_part_is_correct :
  adjusted_smallest_part total proportions extra = 19 + 1/7 := by
  sorry

end NUMINAMATH_CALUDE_adjusted_smallest_part_is_correct_l89_8931


namespace NUMINAMATH_CALUDE_sector_max_area_l89_8938

theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 10) (h_positive : r > 0 ∧ l > 0) :
  (1 / 2) * l * r ≤ 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l89_8938


namespace NUMINAMATH_CALUDE_expression_always_defined_l89_8929

theorem expression_always_defined (x : ℝ) : 
  ∃ y : ℝ, y = x^2 / (2*x^2 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_expression_always_defined_l89_8929


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l89_8950

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 2 < 0 ↔ 1 < x ∧ x < 2) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l89_8950


namespace NUMINAMATH_CALUDE_integer_fraction_pairs_l89_8904

theorem integer_fraction_pairs : 
  ∀ a b : ℕ+, 
    (((a : ℤ)^3 * (b : ℤ) - 1) % ((a : ℤ) + 1) = 0 ∧ 
     ((b : ℤ)^3 * (a : ℤ) + 1) % ((b : ℤ) - 1) = 0) ↔ 
    ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_pairs_l89_8904


namespace NUMINAMATH_CALUDE_class_average_height_l89_8960

/-- The average height of a class of girls -/
theorem class_average_height 
  (total_girls : ℕ) 
  (group1_girls : ℕ) 
  (group1_avg_height : ℝ) 
  (group2_avg_height : ℝ) 
  (h1 : total_girls = 40)
  (h2 : group1_girls = 30)
  (h3 : group1_avg_height = 160)
  (h4 : group2_avg_height = 156) :
  (group1_girls * group1_avg_height + (total_girls - group1_girls) * group2_avg_height) / total_girls = 159 := by
  sorry


end NUMINAMATH_CALUDE_class_average_height_l89_8960


namespace NUMINAMATH_CALUDE_clara_score_reversal_l89_8926

theorem clara_score_reversal (a b : ℕ) :
  (∃ (second_game third_game : ℕ),
    second_game = 45 ∧
    third_game = 54 ∧
    (10 * b + a) + second_game + third_game = (10 * a + b) + second_game + third_game + 132) →
  (10 * b + a) - (10 * a + b) = 126 :=
by sorry

end NUMINAMATH_CALUDE_clara_score_reversal_l89_8926


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l89_8934

theorem polynomial_root_sum (a b c d e : ℝ) : 
  a ≠ 0 → 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 5 ∨ x = -3 ∨ x = 2 ∨ x = (-(b + c + d) / a - 4)) →
  (b + c + d) / a = -6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l89_8934


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l89_8933

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l89_8933


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l89_8961

-- Define the quadratic function f
def f : ℝ → ℝ := fun x ↦ 2 * x^2 - 4 * x + 3

-- State the theorem
theorem quadratic_function_properties :
  (∀ x : ℝ, f x ≥ 1) ∧  -- minimum value is 1
  (f 0 = 3) ∧ (f 2 = 3) ∧  -- f(0) = f(2) = 3
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-3) (-1) → f x > 2 * x + 2 * m + 1) → m < 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l89_8961


namespace NUMINAMATH_CALUDE_article_supports_statements_l89_8912

/-- Represents the content of the given article about Chinese literature and Mo Yan's Nobel Prize -/
def ArticleContent : Type := sorry

/-- Represents the manifestations of the marginalization of literature since the 1990s -/
def LiteratureMarginalization (content : ArticleContent) : Prop := sorry

/-- Represents the effects of mentioning Mo Yan's award multiple times -/
def MoYanAwardEffects (content : ArticleContent) : Prop := sorry

/-- Represents ways to better develop pure literature -/
def PureLiteratureDevelopment (content : ArticleContent) : Prop := sorry

/-- The main theorem stating that the article supports the given statements -/
theorem article_supports_statements (content : ArticleContent) :
  LiteratureMarginalization content ∧
  MoYanAwardEffects content ∧
  PureLiteratureDevelopment content :=
sorry

end NUMINAMATH_CALUDE_article_supports_statements_l89_8912


namespace NUMINAMATH_CALUDE_ellipse_properties_l89_8900

/-- The ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_ecc : (a^2 - b^2) / a^2 = 5 / 9
  h_minor : b = 2

/-- The condition for the line x = m -/
def line_condition (C : Ellipse) (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / C.a^2 + y^2 / C.b^2 = 1 →
  x ≠ -C.a ∧ x ≠ C.a →
  (x - C.a) * (5 / 9 * m - 13 / 3) = 0

theorem ellipse_properties (C : Ellipse) :
  C.a^2 = 9 ∧ C.b^2 = 4 ∧ line_condition C (39 / 5) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l89_8900


namespace NUMINAMATH_CALUDE_bill_drew_four_pentagons_l89_8907

/-- The number of pentagons Bill drew -/
def num_pentagons : ℕ := sorry

/-- The total number of lines Bill drew -/
def total_lines : ℕ := 88

/-- The number of triangles Bill drew -/
def num_triangles : ℕ := 12

/-- The number of squares Bill drew -/
def num_squares : ℕ := 8

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

theorem bill_drew_four_pentagons :
  num_pentagons = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_bill_drew_four_pentagons_l89_8907


namespace NUMINAMATH_CALUDE_eccentricity_of_special_ellipse_l89_8918

/-- Theorem: Eccentricity of a special ellipse -/
theorem eccentricity_of_special_ellipse 
  (a b c : ℝ) 
  (h_pos : a > b ∧ b > 0) 
  (h_c : c = Real.sqrt (a^2 - b^2)) 
  (P : ℝ × ℝ) 
  (h_P_on_ellipse : (P.1^2 / a^2) + (P.2^2 / b^2) = 1) 
  (l : ℝ → ℝ) 
  (h_l : ∀ x, l x = -a^2 / c) 
  (Q : ℝ × ℝ) 
  (h_PQ_perp_l : (P.1 - Q.1) * 1 + (P.2 - Q.2) * 0 = 0) 
  (F : ℝ × ℝ) 
  (h_F : F = (-c, 0)) 
  (h_isosceles : (P.1 - F.1)^2 + (P.2 - F.2)^2 = (Q.1 - F.1)^2 + (Q.2 - F.2)^2) :
  c / a = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_eccentricity_of_special_ellipse_l89_8918


namespace NUMINAMATH_CALUDE_seating_theorem_l89_8975

/-- The number of seating arrangements for 3 people in 6 seats with exactly two adjacent empty seats -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) (adjacent_empty : ℕ) : ℕ :=
  2 * Nat.factorial (people + 1)

theorem seating_theorem :
  seating_arrangements 6 3 2 = 48 :=
by sorry

end NUMINAMATH_CALUDE_seating_theorem_l89_8975


namespace NUMINAMATH_CALUDE_cube_surface_area_l89_8973

theorem cube_surface_area (volume : ℝ) (h : volume = 1331) :
  let side := (volume ^ (1/3 : ℝ))
  6 * side^2 = 726 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l89_8973


namespace NUMINAMATH_CALUDE_gift_fund_equations_correct_l89_8945

/-- Represents the crowdfunding scenario for teachers' New Year gift package. -/
structure GiftFundScenario where
  x : ℕ  -- number of teachers
  y : ℕ  -- price of the gift package

/-- The correct system of equations for the gift fund scenario. -/
def correct_equations (s : GiftFundScenario) : Prop :=
  18 * s.x = s.y + 3 ∧ 17 * s.x = s.y - 4

/-- Theorem stating that the given system of equations correctly describes the gift fund scenario. -/
theorem gift_fund_equations_correct (s : GiftFundScenario) : correct_equations s :=
sorry

end NUMINAMATH_CALUDE_gift_fund_equations_correct_l89_8945


namespace NUMINAMATH_CALUDE_other_number_proof_l89_8955

theorem other_number_proof (A B : ℕ) (hA : A = 24) (hHCF : Nat.gcd A B = 17) (hLCM : Nat.lcm A B = 312) : B = 221 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l89_8955


namespace NUMINAMATH_CALUDE_sqrt_sum_theorem_l89_8969

theorem sqrt_sum_theorem (a : ℝ) (h : a + 1/a = 3) : 
  Real.sqrt a + 1 / Real.sqrt a = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_theorem_l89_8969


namespace NUMINAMATH_CALUDE_cubic_function_nonnegative_l89_8927

theorem cubic_function_nonnegative (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a * x^3 - 3 * x + 1 ≥ 0) ↔ a = 4 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_nonnegative_l89_8927


namespace NUMINAMATH_CALUDE_wheel_speed_proof_l89_8941

/-- Proves that the original speed of a wheel is 7.5 mph given specific conditions -/
theorem wheel_speed_proof (wheel_circumference : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  wheel_circumference = 15 →  -- circumference in feet
  speed_increase = 8 →        -- speed increase in mph
  time_decrease = 1/3 →       -- time decrease in seconds
  ∃ (original_speed : ℝ),
    original_speed = 7.5 ∧    -- original speed in mph
    (original_speed + speed_increase) * (3600 * (15 / (5280 * original_speed)) - time_decrease / 3600) =
    15 / 5280 * 3600 :=
by
  sorry


end NUMINAMATH_CALUDE_wheel_speed_proof_l89_8941


namespace NUMINAMATH_CALUDE_purchase_price_problem_l89_8979

/-- A linear function relating purchase quantity (y) to unit price (x) -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem purchase_price_problem (k b : ℝ) 
  (h1 : 1000 = linear_function k b 800)
  (h2 : 2000 = linear_function k b 700) :
  linear_function k b 5000 = 400 := by sorry

end NUMINAMATH_CALUDE_purchase_price_problem_l89_8979


namespace NUMINAMATH_CALUDE_max_xy_geometric_mean_l89_8953

theorem max_xy_geometric_mean (x y : ℝ) : 
  x^2 = (1 + 2*y) * (1 - 2*y) → 
  ∃ (k : ℝ), k = x*y ∧ k ≤ (1/4 : ℝ) ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 = (1 + 2*y₀) * (1 - 2*y₀) ∧ x₀ * y₀ = (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_xy_geometric_mean_l89_8953


namespace NUMINAMATH_CALUDE_expression_evaluation_l89_8956

theorem expression_evaluation :
  let x : ℚ := -3
  let expr := ((-2 * x^3 - 6*x) / (-2*x)) - 2*(3*x + 1)*(3*x - 1) + 7*x*(x - 1)
  expr = -64 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l89_8956


namespace NUMINAMATH_CALUDE_stating_unique_dissection_l89_8993

/-- A type representing a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Additional structure details would go here, but are omitted for simplicity

/-- A type representing a right triangle with integer-ratio sides -/
structure IntegerRatioRightTriangle where
  -- Additional structure details would go here, but are omitted for simplicity

/-- 
Predicate indicating whether a regular n-sided polygon can be 
completely dissected into integer-ratio right triangles 
-/
def canBeDissected (n : ℕ) : Prop :=
  ∃ (p : RegularPolygon n) (triangles : Set IntegerRatioRightTriangle), 
    -- The formal definition of complete dissection would go here
    True  -- Placeholder

/-- 
Theorem stating that 4 is the only integer n ≥ 3 for which 
a regular n-sided polygon can be completely dissected into 
integer-ratio right triangles 
-/
theorem unique_dissection : 
  ∀ n : ℕ, n ≥ 3 → (canBeDissected n ↔ n = 4) := by
  sorry


end NUMINAMATH_CALUDE_stating_unique_dissection_l89_8993


namespace NUMINAMATH_CALUDE_total_cards_l89_8970

theorem total_cards (initial_cards : ℕ) (added_cards : ℕ) : 
  initial_cards = 4 → added_cards = 3 → initial_cards + added_cards = 7 :=
by sorry

end NUMINAMATH_CALUDE_total_cards_l89_8970


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l89_8963

theorem smaller_solution_of_quadratic (x : ℝ) :
  x^2 - 13*x + 36 = 0 →
  (∃ y : ℝ, y ≠ x ∧ y^2 - 13*y + 36 = 0) →
  (∀ z : ℝ, z^2 - 13*z + 36 = 0 → z ≥ 4) ∧
  (∃ w : ℝ, w^2 - 13*w + 36 = 0 ∧ w = 4) :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l89_8963


namespace NUMINAMATH_CALUDE_remaining_fun_is_1050_l89_8916

/-- Calculates the remaining amount for fun after a series of financial actions --/
def remaining_for_fun (initial_winnings : ℝ) (tax_rate : ℝ) (mortgage_rate : ℝ) 
  (retirement_rate : ℝ) (college_rate : ℝ) (savings : ℝ) : ℝ :=
  let after_tax := initial_winnings * (1 - tax_rate)
  let after_mortgage := after_tax * (1 - mortgage_rate)
  let after_retirement := after_mortgage * (1 - retirement_rate)
  let after_college := after_retirement * (1 - college_rate)
  after_college - savings

/-- Theorem stating that given the specific financial actions, 
    the remaining amount for fun is $1050 --/
theorem remaining_fun_is_1050 : 
  remaining_for_fun 20000 0.55 0.5 (1/3) 0.25 1200 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_remaining_fun_is_1050_l89_8916


namespace NUMINAMATH_CALUDE_probability_three_tails_one_head_l89_8996

def coin_toss_probability : ℚ := 1/2

def number_of_coins : ℕ := 4

def number_of_tails : ℕ := 3

def number_of_heads : ℕ := 1

def number_of_favorable_outcomes : ℕ := 4

theorem probability_three_tails_one_head :
  (number_of_favorable_outcomes : ℚ) * coin_toss_probability ^ number_of_coins = 1/4 :=
sorry

end NUMINAMATH_CALUDE_probability_three_tails_one_head_l89_8996


namespace NUMINAMATH_CALUDE_intersection_segment_length_l89_8948

/-- The length of the line segment AB, where A and B are the intersection points
    of the line y = √3 x and the circle (x + √3)² + (y + 2)² = 1, is equal to √3. -/
theorem intersection_segment_length :
  let line := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1}
  let circle := {p : ℝ × ℝ | (p.1 + Real.sqrt 3)^2 + (p.2 + 2)^2 = 1}
  let intersection := {p : ℝ × ℝ | p ∈ line ∩ circle}
  ∃ A B : ℝ × ℝ, A ∈ intersection ∧ B ∈ intersection ∧ A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l89_8948


namespace NUMINAMATH_CALUDE_angle_on_straight_line_l89_8930

/-- Given a straight line ABC with two angles, one measuring 40° and the other measuring x°, 
    prove that the value of x is 140°. -/
theorem angle_on_straight_line (x : ℝ) : 
  x + 40 = 180 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_angle_on_straight_line_l89_8930


namespace NUMINAMATH_CALUDE_inequality_solution_set_l89_8915

def solution_set (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem inequality_solution_set :
  ∀ x : ℝ, (x - 3) * (x + 2) < 0 ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l89_8915


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l89_8937

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 6) : y = 2 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l89_8937


namespace NUMINAMATH_CALUDE_friend_meeting_point_l89_8911

theorem friend_meeting_point (trail_length : ℝ) (speed_difference : ℝ) 
  (h1 : trail_length = 60)
  (h2 : speed_difference = 0.4) : 
  let faster_friend_distance := trail_length * (1 + speed_difference) / (2 + speed_difference)
  faster_friend_distance = 35 := by
  sorry

end NUMINAMATH_CALUDE_friend_meeting_point_l89_8911


namespace NUMINAMATH_CALUDE_largest_table_sum_l89_8967

def numbers : List ℕ := [2, 3, 5, 7, 11, 17, 19]

def is_valid_arrangement (top : List ℕ) (left : List ℕ) : Prop :=
  top.length = 3 ∧ left.length = 3 ∧ (top ++ left).toFinset ⊆ numbers.toFinset

def table_sum (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum * left.sum)

theorem largest_table_sum :
  ∀ (top left : List ℕ), is_valid_arrangement top left →
  table_sum top left ≤ 1024 :=
sorry

end NUMINAMATH_CALUDE_largest_table_sum_l89_8967


namespace NUMINAMATH_CALUDE_frog_jumps_theorem_l89_8921

-- Define the jump sequences for each frog
def SmallFrogJumps : List Int := [2, 3]
def MediumFrogJumps : List Int := [2, 4]
def LargeFrogJumps : List Int := [6, 9]

-- Define the target rungs for each frog
def SmallFrogTarget : Int := 7
def MediumFrogTarget : Int := 1
def LargeFrogTarget : Int := 3

-- Function to check if a target can be reached using given jumps
def canReachTarget (jumps : List Int) (target : Int) : Prop :=
  ∃ (sequence : List Int), 
    (∀ x ∈ sequence, x ∈ jumps ∨ -x ∈ jumps) ∧ 
    sequence.sum = target

theorem frog_jumps_theorem :
  (canReachTarget SmallFrogJumps SmallFrogTarget) ∧
  ¬(canReachTarget MediumFrogJumps MediumFrogTarget) ∧
  (canReachTarget LargeFrogJumps LargeFrogTarget) := by
  sorry


end NUMINAMATH_CALUDE_frog_jumps_theorem_l89_8921


namespace NUMINAMATH_CALUDE_detergent_calculation_l89_8935

/-- The amount of detergent used per pound of clothes -/
def detergent_per_pound : ℝ := 2

/-- The number of pounds of clothes washed -/
def pounds_of_clothes : ℝ := 9

/-- The total amount of detergent used -/
def total_detergent : ℝ := detergent_per_pound * pounds_of_clothes

theorem detergent_calculation : total_detergent = 18 := by
  sorry

end NUMINAMATH_CALUDE_detergent_calculation_l89_8935


namespace NUMINAMATH_CALUDE_pqr_sum_fraction_prime_l89_8974

theorem pqr_sum_fraction_prime (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  (∃ k : ℕ, p * q * r = k * (p + q + r)) → 
  Nat.Prime (p * q * r / (p + q + r)) :=
by sorry

end NUMINAMATH_CALUDE_pqr_sum_fraction_prime_l89_8974


namespace NUMINAMATH_CALUDE_product_of_y_coordinates_l89_8922

/-- Given a point P on the line x = -3 that is 10 units from (5, 2),
    the product of all possible y-coordinates of P is -32. -/
theorem product_of_y_coordinates : ∀ y₁ y₂ : ℝ,
  ((-3 - 5)^2 + (y₁ - 2)^2 = 10^2) →
  ((-3 - 5)^2 + (y₂ - 2)^2 = 10^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -32 := by
  sorry

end NUMINAMATH_CALUDE_product_of_y_coordinates_l89_8922


namespace NUMINAMATH_CALUDE_inequality_problem_l89_8951

/-- Given positive real numbers a and b such that 1/a + 1/b = 2√2, 
    prove the minimum value of a² + b² and the value of ab under certain conditions. -/
theorem inequality_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h : 1/a + 1/b = 2 * Real.sqrt 2) : 
  (∃ (min : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 * Real.sqrt 2 → x^2 + y^2 ≥ min ∧ 
    a^2 + b^2 = min) ∧ 
  ((a - b)^2 ≥ 4 * (a*b)^3 → a * b = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l89_8951


namespace NUMINAMATH_CALUDE_min_grades_for_average_l89_8939

theorem min_grades_for_average (n : ℕ) (s : ℕ) : n ≥ 51 ↔ 
  (∃ s : ℕ, (4.5 : ℝ) < (s : ℝ) / (n : ℝ) ∧ (s : ℝ) / (n : ℝ) < 4.51) :=
sorry

end NUMINAMATH_CALUDE_min_grades_for_average_l89_8939


namespace NUMINAMATH_CALUDE_balance_weights_theorem_l89_8980

/-- The double factorial of an odd number -/
def oddDoubleFactorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => (k + 1) * oddDoubleFactorial k

/-- The number of ways to place weights on a balance -/
def balanceWeights (n : ℕ) : ℕ :=
  oddDoubleFactorial (2 * n - 1)

/-- Theorem: The number of ways to place n weights (2^0, 2^1, ..., 2^(n-1)) on a balance,
    such that the right pan is never heavier than the left pan, is equal to (2n-1)!! -/
theorem balance_weights_theorem (n : ℕ) (h : n > 0) :
  balanceWeights n = oddDoubleFactorial (2 * n - 1) :=
by
  sorry

#eval balanceWeights 3  -- Expected output: 15
#eval balanceWeights 4  -- Expected output: 105

end NUMINAMATH_CALUDE_balance_weights_theorem_l89_8980


namespace NUMINAMATH_CALUDE_fruit_mix_cherries_l89_8906

/-- Proves that in a fruit mix with the given conditions, the number of cherries is 167 -/
theorem fruit_mix_cherries (b r c : ℕ) : 
  b + r + c = 300 → 
  r = 3 * b → 
  c = 5 * b → 
  c = 167 := by
  sorry

end NUMINAMATH_CALUDE_fruit_mix_cherries_l89_8906


namespace NUMINAMATH_CALUDE_min_touches_equal_total_buttons_l89_8966

/-- Represents a button in the array -/
inductive ButtonState
| OFF
| ON

/-- Represents the array of buttons -/
def ButtonArray := Fin 40 → Fin 50 → ButtonState

/-- The initial state of the array where all buttons are OFF -/
def initialState : ButtonArray := λ _ _ => ButtonState.OFF

/-- The final state of the array where all buttons are ON -/
def finalState : ButtonArray := λ _ _ => ButtonState.ON

/-- Represents a touch operation on a button -/
def touch (array : ButtonArray) (row : Fin 40) (col : Fin 50) : ButtonArray :=
  λ r c => if r = row ∨ c = col then
    match array r c with
    | ButtonState.OFF => ButtonState.ON
    | ButtonState.ON => ButtonState.OFF
  else
    array r c

/-- The theorem stating that the minimum number of touches to switch all buttons from OFF to ON
    is equal to the total number of buttons in the array -/
theorem min_touches_equal_total_buttons :
  ∃ (touches : List (Fin 40 × Fin 50)),
    touches.length = 40 * 50 ∧
    touches.foldl (λ acc (r, c) => touch acc r c) initialState = finalState ∧
    ∀ (touches' : List (Fin 40 × Fin 50)),
      touches'.foldl (λ acc (r, c) => touch acc r c) initialState = finalState →
      touches'.length ≥ 40 * 50 := by
  sorry


end NUMINAMATH_CALUDE_min_touches_equal_total_buttons_l89_8966


namespace NUMINAMATH_CALUDE_A_divisible_by_1980_l89_8925

def A : ℕ := sorry

theorem A_divisible_by_1980 : 1980 ∣ A := by sorry

end NUMINAMATH_CALUDE_A_divisible_by_1980_l89_8925


namespace NUMINAMATH_CALUDE_cleaning_earnings_proof_l89_8972

/-- Calculates the total earnings for cleaning all rooms in a building -/
def total_earnings (floors : ℕ) (rooms_per_floor : ℕ) (hours_per_room : ℕ) (dollars_per_hour : ℕ) : ℕ :=
  floors * rooms_per_floor * hours_per_room * dollars_per_hour

/-- Proves that the total earnings for cleaning the given building is $32,000 -/
theorem cleaning_earnings_proof :
  total_earnings 10 20 8 20 = 32000 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_earnings_proof_l89_8972


namespace NUMINAMATH_CALUDE_auto_credit_percentage_l89_8923

/-- Given that automobile finance companies extended $57 billion of credit, which is 1/3 of the
    total automobile installment credit, and the total consumer installment credit outstanding
    is $855 billion, prove that automobile installment credit accounts for 20% of all outstanding
    consumer installment credit. -/
theorem auto_credit_percentage (finance_company_credit : ℝ) (total_consumer_credit : ℝ)
    (h1 : finance_company_credit = 57)
    (h2 : total_consumer_credit = 855)
    (h3 : finance_company_credit = (1/3) * (3 * finance_company_credit)) :
    (3 * finance_company_credit) / total_consumer_credit = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_auto_credit_percentage_l89_8923


namespace NUMINAMATH_CALUDE_pizza_delivery_theorem_l89_8982

/-- Represents a pizza delivery scenario -/
structure PizzaDelivery where
  total_pizzas : ℕ
  double_pizza_stops : ℕ
  single_pizza_stops : ℕ
  total_time : ℕ

/-- Calculates the average time per stop for a pizza delivery -/
def average_time_per_stop (pd : PizzaDelivery) : ℚ :=
  pd.total_time / (pd.double_pizza_stops + pd.single_pizza_stops)

/-- Theorem: Given the conditions, the average time per stop is 4 minutes -/
theorem pizza_delivery_theorem (pd : PizzaDelivery) 
  (h1 : pd.total_pizzas = 12)
  (h2 : pd.double_pizza_stops = 2)
  (h3 : pd.single_pizza_stops = pd.total_pizzas - 2 * pd.double_pizza_stops)
  (h4 : pd.total_time = 40) :
  average_time_per_stop pd = 4 := by
  sorry


end NUMINAMATH_CALUDE_pizza_delivery_theorem_l89_8982


namespace NUMINAMATH_CALUDE_larger_number_proof_l89_8971

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : max x y = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l89_8971


namespace NUMINAMATH_CALUDE_intersection_sum_reciprocal_constant_l89_8968

/-- The curve C representing the locus of centers of the moving circle M -/
def curve_C (x y : ℝ) : Prop :=
  x > 0 ∧ x^2 / 4 - y^2 / 12 = 1

/-- A point on the curve C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_curve : curve_C x y

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Distance squared between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem intersection_sum_reciprocal_constant
  (P Q : PointOnC)
  (h_perp : (P.x * Q.x + P.y * Q.y = 0)) : -- OP ⊥ OQ condition
  1 / dist_squared O (P.x, P.y) + 1 / dist_squared O (Q.x, Q.y) = 1/6 :=
sorry

end NUMINAMATH_CALUDE_intersection_sum_reciprocal_constant_l89_8968


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l89_8946

/-- Given a quadratic inequality x² + bx + c < 0 with solution set (-1, 2),
    prove that bx² + x + c < 0 has solution set ℝ -/
theorem quadratic_inequality_solution_set
  (b c : ℝ)
  (h : Set.Ioo (-1 : ℝ) 2 = {x | x^2 + b*x + c < 0}) :
  Set.univ = {x | b*x^2 + x + c < 0} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l89_8946


namespace NUMINAMATH_CALUDE_range_of_y_l89_8943

theorem range_of_y (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_y_l89_8943


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l89_8959

theorem polynomial_division_remainder : 
  let dividend := fun z : ℝ => 4 * z^3 + 5 * z^2 - 20 * z + 7
  let divisor := fun z : ℝ => 4 * z - 3
  let quotient := fun z : ℝ => z^2 + 2 * z + 1/4
  let remainder := fun z : ℝ => -15 * z + 31/4
  ∀ z : ℝ, dividend z = divisor z * quotient z + remainder z :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l89_8959


namespace NUMINAMATH_CALUDE_inequality_proof_l89_8908

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b > 2007 * a + 2008 * b) :
  a + b > (Real.sqrt 2007 + Real.sqrt 2008) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l89_8908


namespace NUMINAMATH_CALUDE_annual_interest_rate_is_33_point_33_percent_l89_8978

/-- Represents the banker's gain in rupees -/
def bankers_gain : ℝ := 360

/-- Represents the banker's discount in rupees -/
def bankers_discount : ℝ := 1360

/-- Represents the time period in years -/
def time : ℝ := 3

/-- Calculates the true discount based on banker's discount and banker's gain -/
def true_discount : ℝ := bankers_discount - bankers_gain

/-- Calculates the present value based on banker's discount and banker's gain -/
def present_value : ℝ := bankers_discount - bankers_gain

/-- Calculates the face value as the sum of present value and true discount -/
def face_value : ℝ := present_value + true_discount

/-- Theorem stating that the annual interest rate is 100/3 percent -/
theorem annual_interest_rate_is_33_point_33_percent :
  ∃ (r : ℝ), r = 100 / 3 ∧ true_discount = (present_value * r * time) / 100 :=
sorry

end NUMINAMATH_CALUDE_annual_interest_rate_is_33_point_33_percent_l89_8978


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l89_8944

/-- Given two points A and B that are symmetric about the y-axis, 
    prove that the sum of the offsets from A's coordinates equals 1. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (1 + m, 1 - n) ∧ 
    B = (-3, 2) ∧ 
    (A.1 = -B.1) ∧ 
    (A.2 = B.2)) →
  m + n = 1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l89_8944


namespace NUMINAMATH_CALUDE_boat_speed_proof_l89_8952

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 10

/-- The speed of the stream -/
def stream_speed : ℝ := 2

/-- The distance traveled -/
def distance : ℝ := 36

/-- The time difference between upstream and downstream travel -/
def time_difference : ℝ := 1.5

theorem boat_speed_proof :
  (distance / (boat_speed - stream_speed) - distance / (boat_speed + stream_speed) = time_difference) ∧
  (boat_speed > stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_proof_l89_8952


namespace NUMINAMATH_CALUDE_janets_dress_pockets_janets_dress_pockets_correct_l89_8957

theorem janets_dress_pockets : ℕ → ℕ
  | total_dresses =>
    let dresses_with_pockets := total_dresses / 2
    let dresses_with_two_pockets := dresses_with_pockets / 3
    let dresses_with_three_pockets := dresses_with_pockets - dresses_with_two_pockets
    let total_pockets := dresses_with_two_pockets * 2 + dresses_with_three_pockets * 3
    total_pockets

theorem janets_dress_pockets_correct : janets_dress_pockets 24 = 32 := by
  sorry

end NUMINAMATH_CALUDE_janets_dress_pockets_janets_dress_pockets_correct_l89_8957


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l89_8902

theorem cyclic_sum_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) :
  Real.sqrt (1 - a*b) + Real.sqrt (1 - b*c) + Real.sqrt (1 - c*d) + 
  Real.sqrt (1 - d*a) + Real.sqrt (1 - a*c) + Real.sqrt (1 - b*d) ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l89_8902


namespace NUMINAMATH_CALUDE_model1_is_best_fitting_l89_8983

-- Define the structure for a regression model
structure RegressionModel where
  name : String
  r_squared : Real

-- Define the four models
def model1 : RegressionModel := ⟨"Model 1", 0.98⟩
def model2 : RegressionModel := ⟨"Model 2", 0.80⟩
def model3 : RegressionModel := ⟨"Model 3", 0.50⟩
def model4 : RegressionModel := ⟨"Model 4", 0.25⟩

-- Define a list of all models
def allModels : List RegressionModel := [model1, model2, model3, model4]

-- Define a function to determine if a model is the best fitting
def isBestFitting (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

-- Theorem stating that Model 1 is the best fitting model
theorem model1_is_best_fitting :
  isBestFitting model1 allModels := by
  sorry

end NUMINAMATH_CALUDE_model1_is_best_fitting_l89_8983


namespace NUMINAMATH_CALUDE_sara_quarters_l89_8994

theorem sara_quarters (initial : ℕ) : 
  initial + 49 = 70 → initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l89_8994


namespace NUMINAMATH_CALUDE_min_cross_section_area_and_volume_ratio_l89_8986

/-- A regular triangular pyramid inscribed in a sphere -/
structure RegularTriangularPyramid (R : ℝ) where
  /-- The radius of the circumscribing sphere -/
  radius : ℝ
  /-- The height of the pyramid -/
  height : ℝ
  /-- The height is 4R/3 -/
  height_eq : height = 4 * R / 3

/-- A cross-section of the pyramid passing through a median of its base -/
structure CrossSection (R : ℝ) (pyramid : RegularTriangularPyramid R) where
  /-- The area of the cross-section -/
  area : ℝ
  /-- The ratio of the volumes of the two parts divided by the cross-section -/
  volume_ratio : ℚ × ℚ

/-- The theorem stating the minimum area of the cross-section and the volume ratio -/
theorem min_cross_section_area_and_volume_ratio (R : ℝ) (pyramid : RegularTriangularPyramid R) :
  ∃ (cs : CrossSection R pyramid),
    cs.area = 2 * Real.sqrt 2 / Real.sqrt 33 * R^2 ∧
    cs.volume_ratio = (3, 19) ∧
    ∀ (other_cs : CrossSection R pyramid), cs.area ≤ other_cs.area :=
sorry

end NUMINAMATH_CALUDE_min_cross_section_area_and_volume_ratio_l89_8986


namespace NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l89_8920

theorem ones_digit_of_8_to_47 : (8^47 : ℕ) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l89_8920


namespace NUMINAMATH_CALUDE_sara_pumpkins_left_l89_8991

def pumpkins_left (initial : ℕ) (eaten_by_rabbits : ℕ) (eaten_by_raccoons : ℕ) (given_away : ℕ) : ℕ :=
  initial - eaten_by_rabbits - eaten_by_raccoons - given_away

theorem sara_pumpkins_left : 
  pumpkins_left 43 23 5 7 = 8 := by sorry

end NUMINAMATH_CALUDE_sara_pumpkins_left_l89_8991


namespace NUMINAMATH_CALUDE_printer_time_ratio_l89_8910

/-- Given four printers with their individual completion times, prove the ratio of time taken by printer x alone to the time taken by printers y, z, and w together. -/
theorem printer_time_ratio (x y z w : ℝ) (hx : x = 12) (hy : y = 10) (hz : z = 20) (hw : w = 15) :
  x / (1 / (1/y + 1/z + 1/w)) = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_printer_time_ratio_l89_8910


namespace NUMINAMATH_CALUDE_a4_value_l89_8903

theorem a4_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5) →
  a₄ = -5 := by
sorry

end NUMINAMATH_CALUDE_a4_value_l89_8903


namespace NUMINAMATH_CALUDE_button_probability_l89_8989

/-- Represents a jar containing buttons of two colors -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- Represents the state of two jars after button transfer -/
structure JarState where
  jarA : Jar
  jarB : Jar

def initialJarA : Jar := { red := 7, blue := 9 }

def buttonTransfer (initial : Jar) : JarState :=
  { jarA := { red := initial.red - 3, blue := initial.blue - 2 },
    jarB := { red := 3, blue := 2 } }

def probability_red (jar : Jar) : ℚ :=
  jar.red / (jar.red + jar.blue)

theorem button_probability (initial : Jar := initialJarA) :
  let final := buttonTransfer initial
  let probA := probability_red final.jarA
  let probB := probability_red final.jarB
  probA * probB = 12 / 55 := by
  sorry

end NUMINAMATH_CALUDE_button_probability_l89_8989


namespace NUMINAMATH_CALUDE_factor_expression_l89_8905

theorem factor_expression (x y : ℝ) : 60 * x^2 + 40 * y = 20 * (3 * x^2 + 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l89_8905


namespace NUMINAMATH_CALUDE_residue_calculation_l89_8962

theorem residue_calculation : (204 * 15 - 16 * 8 + 5) % 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l89_8962


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l89_8914

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition from the problem
def condition (x y : ℝ) : Prop :=
  x / (1 + i) = 1 - y * i

-- State the theorem
theorem point_in_first_quadrant (x y : ℝ) (h : condition x y) :
  x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l89_8914


namespace NUMINAMATH_CALUDE_triangle_sin_A_l89_8976

theorem triangle_sin_A (a b : ℝ) (sinB : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sinB = 2/3) :
  let sinA := a * sinB / b
  sinA = 1/2 := by sorry

end NUMINAMATH_CALUDE_triangle_sin_A_l89_8976


namespace NUMINAMATH_CALUDE_land_value_calculation_l89_8992

/-- Proves that if Blake gives Connie $20,000, and the value of the land Connie buys triples in one year,
    then half of the land's value after one year is $30,000. -/
theorem land_value_calculation (initial_amount : ℕ) (value_multiplier : ℕ) : 
  initial_amount = 20000 → value_multiplier = 3 → 
  (initial_amount * value_multiplier) / 2 = 30000 := by
  sorry

end NUMINAMATH_CALUDE_land_value_calculation_l89_8992


namespace NUMINAMATH_CALUDE_function_inequality_solution_l89_8917

/-- Given a function f defined on positive integers and a constant a,
    prove that f(n) = a^(n*(n-1)/2) * f(1) satisfies f(n+1) ≥ a^n * f(n) for all positive integers n. -/
theorem function_inequality_solution (a : ℝ) (f : ℕ+ → ℝ) :
  (∀ n : ℕ+, f n = a^(n.val*(n.val-1)/2) * f 1) →
  (∀ n : ℕ+, f (n + 1) ≥ a^n.val * f n) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_solution_l89_8917


namespace NUMINAMATH_CALUDE_white_car_rental_cost_l89_8932

/-- Represents the cost of renting a white car per minute -/
def white_car_cost : ℝ := 2

/-- Represents the number of red cars -/
def red_cars : ℕ := 3

/-- Represents the number of white cars -/
def white_cars : ℕ := 2

/-- Represents the cost of renting a red car per minute -/
def red_car_cost : ℝ := 3

/-- Represents the total rental time in minutes -/
def rental_time : ℕ := 3 * 60

/-- Represents the total earnings -/
def total_earnings : ℝ := 2340

theorem white_car_rental_cost :
  red_cars * red_car_cost * rental_time + white_cars * white_car_cost * rental_time = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_white_car_rental_cost_l89_8932


namespace NUMINAMATH_CALUDE_second_field_rows_l89_8940

/-- Represents a corn field with a certain number of full rows -/
structure CornField where
  rows : ℕ

/-- Represents a farm with two corn fields -/
structure Farm where
  field1 : CornField
  field2 : CornField

def cobsPerRow : ℕ := 4

theorem second_field_rows (farm : Farm) 
  (h1 : farm.field1.rows = 13) 
  (h2 : farm.field1.rows * cobsPerRow + farm.field2.rows * cobsPerRow = 116) : 
  farm.field2.rows = 16 := by
  sorry

end NUMINAMATH_CALUDE_second_field_rows_l89_8940


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l89_8928

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2*x - 2 = 0) :
  x*(x+2) + (x+1)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l89_8928


namespace NUMINAMATH_CALUDE_expression_one_value_expression_two_value_l89_8995

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem for the first expression
theorem expression_one_value :
  (-0.1)^0 + 32 * 2^(2/3) + (1/4)^(-(1/2)) = 5 := by sorry

-- Theorem for the second expression
theorem expression_two_value :
  lg 500 + lg (8/5) - (1/2) * lg 64 + 50 * (lg 2 + lg 5)^2 = 52 := by sorry

end NUMINAMATH_CALUDE_expression_one_value_expression_two_value_l89_8995


namespace NUMINAMATH_CALUDE_system_solution_l89_8999

theorem system_solution : ∃ (x y : ℚ), 
  (x - 30) / 3 = (2 * y + 7) / 4 ∧ 
  x - y = 10 ∧ 
  x = -81/2 ∧ 
  y = -101/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l89_8999


namespace NUMINAMATH_CALUDE_min_value_theorem_l89_8964

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 2/b ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l89_8964


namespace NUMINAMATH_CALUDE_two_cones_intersection_angle_l89_8987

/-- Represents a cone with height and base radius -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the configuration of two cones -/
structure TwoCones where
  cone1 : Cone
  cone2 : Cone
  commonVertex : Bool
  touchingEachOther : Bool
  touchingPlane : Bool

/-- The angle between the line of intersection of the base planes and the touching plane -/
def intersectionAngle (tc : TwoCones) : ℝ := sorry

theorem two_cones_intersection_angle 
  (tc : TwoCones) 
  (h1 : tc.cone1 = tc.cone2) 
  (h2 : tc.cone1.height = 2) 
  (h3 : tc.cone1.baseRadius = 1) 
  (h4 : tc.commonVertex = true) 
  (h5 : tc.touchingEachOther = true) 
  (h6 : tc.touchingPlane = true) : 
  intersectionAngle tc = Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_two_cones_intersection_angle_l89_8987


namespace NUMINAMATH_CALUDE_number_difference_l89_8965

theorem number_difference (L S : ℤ) (h1 : ∃ X, L = 2*S + X) (h2 : L + S = 27) (h3 : L = 19) :
  L - 2*S = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l89_8965


namespace NUMINAMATH_CALUDE_jenna_bob_difference_l89_8942

/-- Prove that Jenna has $20 less than Bob in her account given the conditions. -/
theorem jenna_bob_difference (bob phil jenna : ℕ) : 
  bob = 60 → 
  phil = bob / 3 → 
  jenna = 2 * phil → 
  bob - jenna = 20 := by
sorry

end NUMINAMATH_CALUDE_jenna_bob_difference_l89_8942


namespace NUMINAMATH_CALUDE_cereal_eating_time_l89_8924

def fat_rate : ℚ := 1 / 20
def thin_rate : ℚ := 1 / 30
def average_rate : ℚ := 1 / 24
def total_cereal : ℚ := 5

theorem cereal_eating_time :
  let combined_rate := fat_rate + thin_rate + average_rate
  (total_cereal / combined_rate) = 40 := by sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l89_8924


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l89_8909

theorem sqrt_three_squared : Real.sqrt 3 ^ 2 = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l89_8909


namespace NUMINAMATH_CALUDE_renovation_calculation_l89_8990

/-- Represents the dimensions and characteristics of a bedroom --/
structure Bedroom where
  length : ℝ
  width : ℝ
  height : ℝ
  unpaintable_area : ℝ
  fixed_furniture_area : ℝ

/-- Calculates the total area to be painted in all bedrooms --/
def total_paintable_area (b : Bedroom) (num_bedrooms : ℕ) : ℝ :=
  num_bedrooms * (2 * (b.length * b.height + b.width * b.height) - b.unpaintable_area)

/-- Calculates the total carpet area for all bedrooms --/
def total_carpet_area (b : Bedroom) (num_bedrooms : ℕ) : ℝ :=
  num_bedrooms * (b.length * b.width - b.fixed_furniture_area)

/-- Theorem stating the correct paintable area and carpet area --/
theorem renovation_calculation (b : Bedroom) (h1 : b.length = 14)
    (h2 : b.width = 11) (h3 : b.height = 9) (h4 : b.unpaintable_area = 70)
    (h5 : b.fixed_furniture_area = 24) :
    total_paintable_area b 4 = 1520 ∧ total_carpet_area b 4 = 520 := by
  sorry


end NUMINAMATH_CALUDE_renovation_calculation_l89_8990


namespace NUMINAMATH_CALUDE_hawks_lost_percentage_l89_8977

/-- Represents a team's game statistics -/
structure TeamStats where
  total_games : ℕ
  win_ratio : ℚ
  loss_ratio : ℚ

/-- Calculates the percentage of games lost -/
def percent_lost (stats : TeamStats) : ℚ :=
  (stats.loss_ratio / (stats.win_ratio + stats.loss_ratio)) * 100

theorem hawks_lost_percentage :
  let hawks : TeamStats := {
    total_games := 64,
    win_ratio := 5,
    loss_ratio := 3
  }
  percent_lost hawks = 37.5 := by sorry

end NUMINAMATH_CALUDE_hawks_lost_percentage_l89_8977


namespace NUMINAMATH_CALUDE_chocolate_bars_distribution_l89_8954

theorem chocolate_bars_distribution (total_bars : ℕ) (num_people : ℕ) (bars_for_two : ℕ) : 
  total_bars = 12 → num_people = 3 → bars_for_two = (total_bars / num_people) * 2 → bars_for_two = 8 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_distribution_l89_8954


namespace NUMINAMATH_CALUDE_clarinet_rate_is_40_l89_8947

/-- The hourly rate for clarinet lessons --/
def clarinet_rate : ℝ := 40

/-- The number of hours of clarinet lessons per week --/
def clarinet_hours_per_week : ℝ := 3

/-- The number of hours of piano lessons per week --/
def piano_hours_per_week : ℝ := 5

/-- The hourly rate for piano lessons --/
def piano_rate : ℝ := 28

/-- The difference in annual cost between piano and clarinet lessons --/
def annual_cost_difference : ℝ := 1040

/-- The number of weeks in a year --/
def weeks_per_year : ℝ := 52

theorem clarinet_rate_is_40 : 
  piano_hours_per_week * piano_rate * weeks_per_year = 
  clarinet_hours_per_week * clarinet_rate * weeks_per_year + annual_cost_difference :=
by sorry

end NUMINAMATH_CALUDE_clarinet_rate_is_40_l89_8947


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l89_8958

theorem simplify_and_evaluate : 
  let x : ℤ := -2
  (2 * x + 1) * (x - 2) - (2 - x)^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l89_8958


namespace NUMINAMATH_CALUDE_shortest_path_ratio_bound_l89_8981

/-- Represents a city in the network -/
structure City where
  id : Nat

/-- Represents the road network -/
structure RoadNetwork where
  cities : Set City
  distance : City → City → ℝ
  shortest_path_length : City → ℝ

/-- The main theorem: the ratio of shortest path lengths between any two cities is at most 1.5 -/
theorem shortest_path_ratio_bound (network : RoadNetwork) :
  ∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities →
  network.shortest_path_length c1 ≤ 1.5 * network.shortest_path_length c2 :=
by sorry

end NUMINAMATH_CALUDE_shortest_path_ratio_bound_l89_8981


namespace NUMINAMATH_CALUDE_problem_solution_l89_8988

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

noncomputable def f_derivative (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f_derivative a b x * Real.exp x

theorem problem_solution (a b : ℝ) :
  f_derivative a b 1 = 2*a ∧ f_derivative a b 2 = -b →
  (a = -3/2 ∧ b = -3) ∧
  (∀ x, g a b x ≥ g a b 1) ∧
  (∀ x, g a b x ≤ g a b (-2)) ∧
  g a b 1 = -3 * Real.exp 1 ∧
  g a b (-2) = 15 * Real.exp (-2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l89_8988


namespace NUMINAMATH_CALUDE_circle_area_when_equal_to_circumference_l89_8984

/-- Given a circle where the circumference and area are numerically equal,
    and the diameter is 4, prove that the area is 4π. -/
theorem circle_area_when_equal_to_circumference (r : ℝ) : 
  2 * Real.pi * r = Real.pi * r^2 →   -- Circumference equals area
  4 = 2 * r →                         -- Diameter is 4
  Real.pi * r^2 = 4 * Real.pi :=      -- Area is 4π
by
  sorry

#check circle_area_when_equal_to_circumference

end NUMINAMATH_CALUDE_circle_area_when_equal_to_circumference_l89_8984


namespace NUMINAMATH_CALUDE_min_c_squared_l89_8913

theorem min_c_squared (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + 2*b = 4 →
  a * Real.sin A + 4*b * Real.sin B = 6*a * Real.sin B * Real.sin C →
  c^2 ≥ 5 - (4 * Real.sqrt 5) / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_c_squared_l89_8913
