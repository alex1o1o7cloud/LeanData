import Mathlib

namespace NUMINAMATH_CALUDE_car_A_original_speed_l675_67521

/-- Represents the speed and position of a car --/
structure Car where
  speed : ℝ
  position : ℝ

/-- Represents the scenario of two cars meeting --/
structure MeetingScenario where
  carA : Car
  carB : Car
  meetingTime : ℝ
  meetingPosition : ℝ

/-- The original scenario where cars meet at point C --/
def originalScenario : MeetingScenario := sorry

/-- Scenario where car B increases speed by 5 km/h --/
def scenarioBFaster : MeetingScenario := sorry

/-- Scenario where car A increases speed by 5 km/h --/
def scenarioAFaster : MeetingScenario := sorry

theorem car_A_original_speed :
  ∃ (s : ℝ),
    (originalScenario.carA.speed = s) ∧
    (originalScenario.meetingTime = 6) ∧
    (scenarioBFaster.carB.speed = originalScenario.carB.speed + 5) ∧
    (scenarioBFaster.meetingPosition = originalScenario.meetingPosition - 12) ∧
    (scenarioAFaster.carA.speed = originalScenario.carA.speed + 5) ∧
    (scenarioAFaster.meetingPosition = originalScenario.meetingPosition + 16) ∧
    (s = 30) := by
  sorry

end NUMINAMATH_CALUDE_car_A_original_speed_l675_67521


namespace NUMINAMATH_CALUDE_quadratic_solution_l675_67577

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9 : ℝ) - 36 = 0) → b = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l675_67577


namespace NUMINAMATH_CALUDE_polygon_sides_l675_67533

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 → n = 8 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l675_67533


namespace NUMINAMATH_CALUDE_original_number_proof_l675_67585

theorem original_number_proof : ∃ x : ℝ, x / 12.75 = 16 ∧ x = 204 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l675_67585


namespace NUMINAMATH_CALUDE_sports_club_members_l675_67580

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  badminton : ℕ  -- Number of members playing badminton
  tennis : ℕ     -- Number of members playing tennis
  neither : ℕ    -- Number of members playing neither badminton nor tennis
  both : ℕ       -- Number of members playing both badminton and tennis

/-- Calculates the total number of members in the sports club -/
def totalMembers (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 30 -/
theorem sports_club_members :
  ∃ (club : SportsClub), 
    club.badminton = 16 ∧ 
    club.tennis = 19 ∧ 
    club.neither = 2 ∧ 
    club.both = 7 ∧ 
    totalMembers club = 30 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l675_67580


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l675_67514

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l675_67514


namespace NUMINAMATH_CALUDE_spoiled_milk_percentage_l675_67568

theorem spoiled_milk_percentage
  (egg_rotten_percent : ℝ)
  (flour_weevil_percent : ℝ)
  (all_good_probability : ℝ)
  (h1 : egg_rotten_percent = 60)
  (h2 : flour_weevil_percent = 25)
  (h3 : all_good_probability = 24) :
  ∃ (spoiled_milk_percent : ℝ),
    spoiled_milk_percent = 20 ∧
    (1 - spoiled_milk_percent / 100) * (1 - egg_rotten_percent / 100) * (1 - flour_weevil_percent / 100) = all_good_probability / 100 :=
by sorry

end NUMINAMATH_CALUDE_spoiled_milk_percentage_l675_67568


namespace NUMINAMATH_CALUDE_abs_z_equals_sqrt_two_l675_67517

-- Define the complex number z
def z : ℂ := 1 + 2 * Complex.I + Complex.I ^ 3

-- Theorem statement
theorem abs_z_equals_sqrt_two : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_sqrt_two_l675_67517


namespace NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_for_skew_lines_l675_67522

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the intersection relation between two planes
variable (intersect : Plane → Plane → Prop)

-- Define the skew relation between two lines
variable (skew : Line → Line → Prop)

-- Define the theorem
theorem planes_intersect_necessary_not_sufficient_for_skew_lines
  (α β : Plane) (m n : Line)
  (distinct_planes : α ≠ β)
  (m_perp_α : perpendicular m α)
  (n_perp_β : perpendicular n β) :
  (∀ m n, skew m n → intersect α β) ∧
  ¬(∀ m n, intersect α β → skew m n) :=
sorry

end NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_for_skew_lines_l675_67522


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l675_67520

theorem fraction_sum_equality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_sum : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 
  1 / (b - c)^2 + 1 / (c - a)^2 + 1 / (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l675_67520


namespace NUMINAMATH_CALUDE_triangle_third_side_l675_67530

theorem triangle_third_side (a b c : ℕ) : 
  (a - b = 7 ∨ b - a = 7) →  -- difference between two sides is 7
  (a + b + c) % 2 = 1 →      -- perimeter is odd
  c = 8                      -- third side is 8
:= by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l675_67530


namespace NUMINAMATH_CALUDE_fourth_member_income_l675_67591

def family_size : ℕ := 4
def average_income : ℕ := 10000
def member1_income : ℕ := 8000
def member2_income : ℕ := 15000
def member3_income : ℕ := 6000

theorem fourth_member_income :
  let total_income := family_size * average_income
  let known_members_income := member1_income + member2_income + member3_income
  total_income - known_members_income = 11000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_member_income_l675_67591


namespace NUMINAMATH_CALUDE_certain_number_is_three_l675_67546

theorem certain_number_is_three (a b x : ℝ) 
  (h1 : 2 * a = x * b) 
  (h2 : a * b ≠ 0) 
  (h3 : (a / 3) / (b / 2) = 1) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_certain_number_is_three_l675_67546


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l675_67575

/-- A geometric sequence with a₂ = 8 and a₅ = 64 has a common ratio of 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence definition
  a 2 = 8 →                              -- Given condition
  a 5 = 64 →                             -- Given condition
  a 2 / a 1 = 2 :=                       -- Common ratio q = 2
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l675_67575


namespace NUMINAMATH_CALUDE_flea_treatment_result_l675_67548

/-- The number of fleas on a dog after a series of treatments -/
def fleas_after_treatments (initial_fleas : ℕ) (num_treatments : ℕ) : ℕ :=
  initial_fleas / (2^num_treatments)

/-- Theorem: If a dog undergoes four flea treatments, where each treatment halves the number of fleas,
    and the initial number of fleas is 210 more than the final number, then the final number of fleas is 14. -/
theorem flea_treatment_result :
  ∀ F : ℕ,
  (F + 210 = fleas_after_treatments (F + 210) 4) →
  F = 14 :=
by sorry

end NUMINAMATH_CALUDE_flea_treatment_result_l675_67548


namespace NUMINAMATH_CALUDE_sand_weight_l675_67554

/-- Given the total weight of materials and the weight of gravel, 
    calculate the weight of sand -/
theorem sand_weight (total_weight gravel_weight : ℝ) 
  (h1 : total_weight = 14.02)
  (h2 : gravel_weight = 5.91) : 
  total_weight - gravel_weight = 8.11 := by
  sorry

end NUMINAMATH_CALUDE_sand_weight_l675_67554


namespace NUMINAMATH_CALUDE_triangle_height_inequality_l675_67502

/-- For a triangle with side lengths a ≤ b ≤ c, heights h_a, h_b, h_c, 
    circumradius R, and semiperimeter p, the following inequality holds. -/
theorem triangle_height_inequality 
  (a b c : ℝ) (h_a h_b h_c R p : ℝ) 
  (h_order : a ≤ b ∧ b ≤ c) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ p > 0) 
  (h_heights : h_a = 2 * (p - a) * (p - b) * (p - c) / (a * b * c) ∧ 
               h_b = 2 * (p - a) * (p - b) * (p - c) / (a * b * c) ∧ 
               h_c = 2 * (p - a) * (p - b) * (p - c) / (a * b * c))
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_circumradius : R = a * b * c / (4 * (p - a) * (p - b) * (p - c))) :
  h_a + h_b + h_c ≤ 3 * b * (a^2 + a*c + c^2) / (4 * p * R) :=
sorry

end NUMINAMATH_CALUDE_triangle_height_inequality_l675_67502


namespace NUMINAMATH_CALUDE_vector_projection_and_magnitude_l675_67556

/-- Given vectors a and b in R², if the projection of a in its direction is -√2,
    then the second component of b is 4 and the magnitude of b is 2√5. -/
theorem vector_projection_and_magnitude (a b : ℝ × ℝ) :
  a = (1, -1) →
  b.1 = 2 →
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2) = -Real.sqrt 2 →
  b.2 = 4 ∧ Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_and_magnitude_l675_67556


namespace NUMINAMATH_CALUDE_range_of_b_l675_67583

/-- The curve representing a semi-circle -/
def curve (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

/-- The line intersecting the curve -/
def line (x y b : ℝ) : Prop := y = x + b

/-- The domain constraints for x and y -/
def domain_constraints (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3

/-- The theorem stating the range of b -/
theorem range_of_b :
  ∀ b : ℝ, (∃ x y : ℝ, curve x y ∧ line x y b ∧ domain_constraints x y) ↔ 
  (1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l675_67583


namespace NUMINAMATH_CALUDE_translated_line_equation_translation_result_l675_67534

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount -/
def translate_line_vertical (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

theorem translated_line_equation (original : Line) (translation : ℝ) :
  let translated := translate_line_vertical original (-translation)
  translated.slope = original.slope ∧
  translated.intercept = original.intercept - translation :=
by sorry

/-- The original line y = -2x + 1 -/
def original_line : Line :=
  { slope := -2, intercept := 1 }

/-- The amount of downward translation -/
def translation_amount : ℝ := 4

theorem translation_result :
  let translated := translate_line_vertical original_line (-translation_amount)
  translated.slope = -2 ∧ translated.intercept = -3 :=
by sorry

end NUMINAMATH_CALUDE_translated_line_equation_translation_result_l675_67534


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l675_67565

theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : a = 90) (h2 : b = 120) 
  (h3 : c^2 = a^2 + b^2) : c = 150 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l675_67565


namespace NUMINAMATH_CALUDE_root_sum_ratio_l675_67540

theorem root_sum_ratio (k₁ k₂ : ℝ) : 
  (∃ a b : ℝ, (k₁ * (a^2 - a) + a + 7 = 0 ∧ k₂ * (b^2 - b) + b + 7 = 0) ∧
              (a / b + b / a = 5 / 6)) →
  k₁ / k₂ + k₂ / k₁ = 433 / 36 := by
sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l675_67540


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_profit_function_concave_down_l675_67572

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 280 * x - 1600

/-- Represents the optimal selling price -/
def optimal_price : ℝ := 14

/-- Represents the maximum profit -/
def max_profit : ℝ := 360

/-- Theorem stating that the optimal price maximizes the profit function -/
theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_price :=
sorry

/-- Theorem stating that the maximum profit is achieved at the optimal price -/
theorem max_profit_at_optimal_price :
  profit_function optimal_price = max_profit :=
sorry

/-- Theorem stating that the profit function is concave down -/
theorem profit_function_concave_down :
  ∀ x y t : ℝ, 0 ≤ t ∧ t ≤ 1 →
  profit_function (t * x + (1 - t) * y) ≥ t * profit_function x + (1 - t) * profit_function y :=
sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_profit_function_concave_down_l675_67572


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_line_contained_line_l675_67527

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def contains (p : Plane) (l : Line) : Prop := sorry

-- State the theorems
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n := by sorry

theorem perpendicular_line_contained_line (m n : Line) (α : Plane) :
  perpendicular m α → contains α n → perpendicular m n := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_line_contained_line_l675_67527


namespace NUMINAMATH_CALUDE_expression_value_l675_67537

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 3)    -- absolute value of m is 3
  : (a + b) / m + c * d + m = 4 ∨ (a + b) / m + c * d + m = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l675_67537


namespace NUMINAMATH_CALUDE_angle_calculations_l675_67596

theorem angle_calculations (α : Real) (h : Real.tan α = -3/7) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/7 ∧
  2 + Real.sin α * Real.cos α - Real.cos α ^ 2 = 23/29 := by
sorry

end NUMINAMATH_CALUDE_angle_calculations_l675_67596


namespace NUMINAMATH_CALUDE_basketball_players_count_l675_67569

def students_jumping_rope : ℕ := 6

def students_playing_basketball : ℕ := 4 * students_jumping_rope

theorem basketball_players_count : students_playing_basketball = 24 := by
  sorry

end NUMINAMATH_CALUDE_basketball_players_count_l675_67569


namespace NUMINAMATH_CALUDE_sum_lower_bound_l675_67552

theorem sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b + 3 = a * b) :
  a + b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l675_67552


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l675_67501

/-- Two lines in the plane are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_a_value :
  (∀ x y, y = x ↔ 2 * x + a * y = 1) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l675_67501


namespace NUMINAMATH_CALUDE_daria_concert_money_l675_67589

theorem daria_concert_money (total_tickets : ℕ) (ticket_cost : ℕ) (current_savings : ℕ) :
  total_tickets = 4 →
  ticket_cost = 90 →
  current_savings = 189 →
  total_tickets * ticket_cost - current_savings = 171 :=
by sorry

end NUMINAMATH_CALUDE_daria_concert_money_l675_67589


namespace NUMINAMATH_CALUDE_john_vacation_money_l675_67584

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ :=
  sorry

/-- Calculates the remaining money after buying a ticket -/
def remainingMoney (savings : ℕ) (ticketCost : ℕ) : ℕ :=
  savings - ticketCost

theorem john_vacation_money :
  let savings := base8ToBase10 5555
  let ticketCost := 1200
  remainingMoney savings ticketCost = 1725 := by
  sorry

end NUMINAMATH_CALUDE_john_vacation_money_l675_67584


namespace NUMINAMATH_CALUDE_linear_function_property_l675_67563

/-- Given a linear function f(x) = ax + b, if f(1) = 2 and f'(1) = 2, then f(2) = 4 -/
theorem linear_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x + b)
    (h2 : f 1 = 2)
    (h3 : (deriv f) 1 = 2) : 
  f 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l675_67563


namespace NUMINAMATH_CALUDE_gift_cost_l675_67545

theorem gift_cost (dave_money : ℕ) (kyle_initial : ℕ) (kyle_after_snowboarding : ℕ) (lisa_money : ℕ) (gift_cost : ℕ) : 
  dave_money = 46 →
  kyle_initial = 3 * dave_money - 12 →
  kyle_after_snowboarding = kyle_initial - kyle_initial / 3 →
  lisa_money = kyle_after_snowboarding + 20 →
  gift_cost = (kyle_after_snowboarding + lisa_money) / 2 →
  gift_cost = 94 := by
sorry

end NUMINAMATH_CALUDE_gift_cost_l675_67545


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_area_l675_67531

theorem right_triangle_perimeter_area (a c : ℝ) :
  a > 0 ∧ c > 0 ∧  -- Positive sides
  c > a ∧  -- Hypotenuse is longest side
  Real.sqrt (c - 5) + 2 * Real.sqrt (10 - 2*c) = a - 4 →  -- Given equation
  ∃ b : ℝ, 
    b > 0 ∧  -- Positive side
    a^2 + b^2 = c^2 ∧  -- Pythagorean theorem
    a + b + c = 12 ∧  -- Perimeter
    (1/2) * a * b = 6  -- Area
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_area_l675_67531


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_sqrt_8_simplification_sqrt_1_3_simplification_sqrt_4_simplification_l675_67539

-- Define what it means for a quadratic radical to be simplest
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ≠ x → (∃ n : ℕ, x = Real.sqrt n) → 
    ¬(∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x = a * Real.sqrt b ∧ b < y)

-- State the theorem
theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 6) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 8) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/3)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 4) :=
by sorry

-- Define the simplification rules
theorem sqrt_8_simplification : Real.sqrt 8 = 2 * Real.sqrt 2 := by sorry
theorem sqrt_1_3_simplification : Real.sqrt (1/3) = Real.sqrt 3 / 3 := by sorry
theorem sqrt_4_simplification : Real.sqrt 4 = 2 := by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_sqrt_8_simplification_sqrt_1_3_simplification_sqrt_4_simplification_l675_67539


namespace NUMINAMATH_CALUDE_wendy_small_glasses_l675_67518

/-- The number of small glasses polished by Wendy -/
def small_glasses : ℕ := 50

/-- The number of large glasses polished by Wendy -/
def large_glasses : ℕ := small_glasses + 10

/-- The total number of glasses polished by Wendy -/
def total_glasses : ℕ := 110

/-- Proof that Wendy polished 50 small glasses -/
theorem wendy_small_glasses :
  small_glasses = 50 ∧
  large_glasses = small_glasses + 10 ∧
  small_glasses + large_glasses = total_glasses :=
by sorry

end NUMINAMATH_CALUDE_wendy_small_glasses_l675_67518


namespace NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_l675_67525

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_x2y2_in_expansion :
  let n : ℕ := 4
  let k : ℕ := 2
  let coefficient : ℤ := binomial_coefficient n k * (-2)^k
  coefficient = 24 := by sorry

end NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_l675_67525


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_sum_l675_67511

-- Define the ellipse
def ellipse (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

-- Define the property that the ellipse passes through (0, 4)
def passes_through_B (m : ℝ) : Prop :=
  ellipse 0 4 m

-- Define the sum of distances from any point to the foci
def sum_distances_to_foci (m : ℝ) : ℝ := 8

-- Theorem statement
theorem ellipse_foci_distance_sum (m : ℝ) 
  (h : passes_through_B m) : 
  sum_distances_to_foci m = 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_sum_l675_67511


namespace NUMINAMATH_CALUDE_probability_no_empty_boxes_l675_67567

/-- The number of distinct balls -/
def num_balls : ℕ := 3

/-- The number of distinct boxes -/
def num_boxes : ℕ := 3

/-- The probability of placing balls into boxes with no empty boxes -/
def prob_no_empty_boxes : ℚ := 2/9

/-- Theorem stating the probability of placing balls into boxes with no empty boxes -/
theorem probability_no_empty_boxes :
  (Nat.factorial num_balls : ℚ) / (num_boxes ^ num_balls : ℚ) = prob_no_empty_boxes :=
sorry

end NUMINAMATH_CALUDE_probability_no_empty_boxes_l675_67567


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l675_67509

theorem solution_satisfies_system :
  let eq1 (x y : ℝ) := y + Real.sqrt (y - 3 * x) + 3 * x = 12
  let eq2 (x y : ℝ) := y^2 + y - 3 * x - 9 * x^2 = 144
  (eq1 (-24) 72 ∧ eq2 (-24) 72) ∧
  (eq1 (-4/3) 12 ∧ eq2 (-4/3) 12) := by
  sorry

#check solution_satisfies_system

end NUMINAMATH_CALUDE_solution_satisfies_system_l675_67509


namespace NUMINAMATH_CALUDE_store_price_reduction_l675_67571

theorem store_price_reduction (original_price : ℝ) (first_reduction : ℝ) :
  first_reduction > 0 →
  first_reduction < 100 →
  let second_reduction := 10
  let final_price_percentage := 82.8
  (original_price * (1 - first_reduction / 100) * (1 - second_reduction / 100)) / original_price * 100 = final_price_percentage →
  first_reduction = 8 := by
sorry

end NUMINAMATH_CALUDE_store_price_reduction_l675_67571


namespace NUMINAMATH_CALUDE_problem_1_l675_67547

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

-- State the theorem
theorem problem_1 (m : ℝ) : (A.compl ∩ B m = ∅) → (m = 1 ∨ m = 2) :=
sorry

end NUMINAMATH_CALUDE_problem_1_l675_67547


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l675_67579

def A : Set ℤ := {0, 1, 2, 8}
def B : Set ℤ := {-1, 1, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {1, 8} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l675_67579


namespace NUMINAMATH_CALUDE_equal_expressions_l675_67558

theorem equal_expressions : (-2)^3 = -2^3 := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_l675_67558


namespace NUMINAMATH_CALUDE_black_squares_count_l675_67588

/-- Represents a checkerboard with side length n -/
structure Checkerboard (n : ℕ) where
  is_corner_black : Bool
  is_alternating : Bool

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard 33) : ℕ :=
  sorry

/-- Theorem: The number of black squares on a 33x33 alternating checkerboard with black corners is 545 -/
theorem black_squares_count : 
  ∀ (board : Checkerboard 33), 
  board.is_corner_black = true → 
  board.is_alternating = true → 
  count_black_squares board = 545 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_count_l675_67588


namespace NUMINAMATH_CALUDE_total_boxes_needed_l675_67503

-- Define the amounts of wooden sticks and box capacities
def taehyung_total : ℚ := 21 / 11
def taehyung_per_box : ℚ := 7 / 11
def hoseok_total : ℚ := 8 / 17
def hoseok_per_box : ℚ := 2 / 17

-- Define the function to calculate the number of boxes needed
def boxes_needed (total : ℚ) (per_box : ℚ) : ℕ :=
  (total / per_box).ceil.toNat

-- Theorem statement
theorem total_boxes_needed :
  boxes_needed taehyung_total taehyung_per_box +
  boxes_needed hoseok_total hoseok_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_needed_l675_67503


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l675_67538

theorem quadratic_root_proof :
  let x : ℝ := (-5 + Real.sqrt (5^2 + 4*3*1)) / (2*3)
  3 * x^2 + 5 * x - 1 = 0 ∨
  let x : ℝ := (-5 - Real.sqrt (5^2 + 4*3*1)) / (2*3)
  3 * x^2 + 5 * x - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l675_67538


namespace NUMINAMATH_CALUDE_inequality_proof_l675_67562

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * b^2 + b^2 * c^2 + a^2 * c^2 ≥ a * b * c * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l675_67562


namespace NUMINAMATH_CALUDE_angle_300_in_fourth_quadrant_l675_67551

/-- An angle is in the fourth quadrant if it's between 270° and 360° (exclusive) -/
def is_in_fourth_quadrant (angle : ℝ) : Prop :=
  270 < angle ∧ angle < 360

/-- Prove that 300° is in the fourth quadrant -/
theorem angle_300_in_fourth_quadrant :
  is_in_fourth_quadrant 300 := by
  sorry

end NUMINAMATH_CALUDE_angle_300_in_fourth_quadrant_l675_67551


namespace NUMINAMATH_CALUDE_same_solution_d_value_l675_67542

theorem same_solution_d_value (x : ℝ) (d : ℝ) : 
  (3 * x + 8 = 4) ∧ (d * x - 15 = -5) → d = -7.5 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_d_value_l675_67542


namespace NUMINAMATH_CALUDE_complex_equation_solution_l675_67561

theorem complex_equation_solution (z : ℂ) : (z + 1) * Complex.I = 1 - Complex.I → z = -2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l675_67561


namespace NUMINAMATH_CALUDE_distance_ratios_sum_to_one_l675_67507

theorem distance_ratios_sum_to_one (x y z : ℝ) :
  let r := Real.sqrt (x^2 + y^2 + z^2)
  let c := x / r
  let s := y / r
  let z_r := z / r
  s^2 - c^2 + z_r^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_ratios_sum_to_one_l675_67507


namespace NUMINAMATH_CALUDE_island_population_even_l675_67529

/-- Represents the type of inhabitants on the island -/
inductive Inhabitant
| Knight
| Liar

/-- Represents a claim about the number of inhabitants -/
inductive Claim
| EvenKnights
| OddLiars

/-- Function that determines if a given inhabitant tells the truth about a claim -/
def tellsTruth (i : Inhabitant) (c : Claim) : Prop :=
  match i, c with
  | Inhabitant.Knight, _ => true
  | Inhabitant.Liar, _ => false

/-- The island population -/
structure Island where
  inhabitants : List Inhabitant
  claims : List (Inhabitant × Claim)
  all_claimed : ∀ i ∈ inhabitants, ∃ c, (i, c) ∈ claims

theorem island_population_even (isle : Island) : Even (List.length isle.inhabitants) := by
  sorry


end NUMINAMATH_CALUDE_island_population_even_l675_67529


namespace NUMINAMATH_CALUDE_f_is_quadratic_l675_67510

-- Define what a quadratic equation is
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the function representing the equation x^2 = x + 1
def f (x : ℝ) : ℝ := x^2 - x - 1

-- Theorem stating that f is a quadratic equation
theorem f_is_quadratic : is_quadratic_equation f :=
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l675_67510


namespace NUMINAMATH_CALUDE_evaluate_expression_l675_67599

theorem evaluate_expression : 49^2 - 25^2 + 10^2 = 1876 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l675_67599


namespace NUMINAMATH_CALUDE_random_events_classification_l675_67566

-- Define the events
inductive Event
| E1 -- If a > b, then a - b > 0
| E2 -- For any real number a (a > 0 and a ≠ 1), the function y = log_a x is an increasing function
| E3 -- A person shoots once and hits the bullseye
| E4 -- Drawing a yellow ball from a bag containing one red and two white balls

-- Define the types of events
inductive EventType
| Certain
| Random
| Impossible

-- Function to classify events
def classifyEvent (e : Event) : EventType :=
  match e with
  | Event.E1 => EventType.Certain
  | Event.E2 => EventType.Random
  | Event.E3 => EventType.Random
  | Event.E4 => EventType.Impossible

-- Theorem statement
theorem random_events_classification :
  (classifyEvent Event.E2 = EventType.Random ∧
   classifyEvent Event.E3 = EventType.Random) ∧
  (classifyEvent Event.E1 ≠ EventType.Random ∧
   classifyEvent Event.E4 ≠ EventType.Random) :=
sorry

end NUMINAMATH_CALUDE_random_events_classification_l675_67566


namespace NUMINAMATH_CALUDE_water_displacement_cube_in_cylinder_l675_67505

theorem water_displacement_cube_in_cylinder (cube_side : ℝ) (cylinder_radius : ℝ) 
  (h_cube : cube_side = 12) (h_cylinder : cylinder_radius = 6) : ∃ v : ℝ, v^2 = 4374 :=
by
  sorry

end NUMINAMATH_CALUDE_water_displacement_cube_in_cylinder_l675_67505


namespace NUMINAMATH_CALUDE_cyclic_inequality_l675_67594

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (x^3 / (x^2 + y)) + (y^3 / (y^2 + z)) + (z^3 / (z^2 + x)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l675_67594


namespace NUMINAMATH_CALUDE_teapot_teacup_discount_l675_67576

/-- Represents the payment amount for a purchase of teapots and teacups under different discount methods -/
def payment_amount (x : ℝ) : Prop :=
  let teapot_price : ℝ := 20
  let teacup_price : ℝ := 5
  let num_teapots : ℝ := 4
  let discount_rate : ℝ := 0.92
  let y1 : ℝ := teapot_price * num_teapots + teacup_price * (x - num_teapots)
  let y2 : ℝ := (teapot_price * num_teapots + teacup_price * x) * discount_rate
  (4 ≤ x ∧ x < 34 → y1 < y2) ∧
  (x = 34 → y1 = y2) ∧
  (x > 34 → y1 > y2)

theorem teapot_teacup_discount (x : ℝ) (h : x ≥ 4) : payment_amount x := by
  sorry

end NUMINAMATH_CALUDE_teapot_teacup_discount_l675_67576


namespace NUMINAMATH_CALUDE_min_pizzas_to_break_even_l675_67549

def car_cost : ℕ := 6000
def bag_cost : ℕ := 200
def earning_per_pizza : ℕ := 12
def gas_cost_per_delivery : ℕ := 4

theorem min_pizzas_to_break_even :
  let total_cost := car_cost + bag_cost
  let net_earning_per_pizza := earning_per_pizza - gas_cost_per_delivery
  (∀ n : ℕ, n * net_earning_per_pizza < total_cost → n < 775) ∧
  775 * net_earning_per_pizza ≥ total_cost :=
sorry

end NUMINAMATH_CALUDE_min_pizzas_to_break_even_l675_67549


namespace NUMINAMATH_CALUDE_sqrt_56_58_fraction_existence_l675_67541

theorem sqrt_56_58_fraction_existence (q : ℕ+) :
  q ≠ 1 → q ≠ 3 → ∃ p : ℤ, Real.sqrt 56 < (p : ℚ) / q ∧ (p : ℚ) / q < Real.sqrt 58 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_56_58_fraction_existence_l675_67541


namespace NUMINAMATH_CALUDE_hiker_supply_per_mile_l675_67535

/-- A hiker's supply calculation problem -/
theorem hiker_supply_per_mile
  (hiking_rate : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (first_pack_weight : ℝ)
  (resupply_percentage : ℝ)
  (h1 : hiking_rate = 2.5)
  (h2 : hours_per_day = 8)
  (h3 : days = 5)
  (h4 : first_pack_weight = 40)
  (h5 : resupply_percentage = 0.25)
  : (first_pack_weight + first_pack_weight * resupply_percentage) / (hiking_rate * hours_per_day * days) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_hiker_supply_per_mile_l675_67535


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l675_67560

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y :=
by
  use (-2)
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l675_67560


namespace NUMINAMATH_CALUDE_student_divisor_problem_l675_67581

theorem student_divisor_problem (correct_divisor correct_quotient student_quotient : ℕ) 
  (h1 : correct_divisor = 36)
  (h2 : correct_quotient = 42)
  (h3 : student_quotient = 24)
  : ∃ student_divisor : ℕ, 
    student_divisor * student_quotient = correct_divisor * correct_quotient ∧ 
    student_divisor = 63 := by
  sorry

end NUMINAMATH_CALUDE_student_divisor_problem_l675_67581


namespace NUMINAMATH_CALUDE_min_payment_bound_l675_67598

/-- Tea set price in yuan -/
def tea_set_price : ℕ := 200

/-- Tea bowl price in yuan -/
def tea_bowl_price : ℕ := 20

/-- Number of tea sets purchased -/
def num_tea_sets : ℕ := 30

/-- Discount factor for Option 2 -/
def discount_factor : ℚ := 95 / 100

/-- Payment for Option 1 given x tea bowls -/
def payment_option1 (x : ℕ) : ℕ := 20 * x + 5400

/-- Payment for Option 2 given x tea bowls -/
def payment_option2 (x : ℕ) : ℕ := 19 * x + 5700

/-- Theorem: The minimum payment is less than or equal to the minimum of Option 1 and Option 2 -/
theorem min_payment_bound (x : ℕ) (hx : x > 30) :
  ∃ (y : ℕ), y ≤ min (payment_option1 x) (payment_option2 x) ∧
  y = num_tea_sets * tea_set_price + x * tea_bowl_price -
      (min num_tea_sets x) * tea_bowl_price +
      ((x - min num_tea_sets x) * tea_bowl_price * discount_factor).floor :=
sorry

end NUMINAMATH_CALUDE_min_payment_bound_l675_67598


namespace NUMINAMATH_CALUDE_exponential_and_logarithm_inequalities_l675_67557

-- Define the exponential function
noncomputable def exp (base : ℝ) (exponent : ℝ) : ℝ := Real.exp (exponent * Real.log base)

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem exponential_and_logarithm_inequalities :
  (exp 0.8 (-0.1) < exp 0.8 (-0.2)) ∧ (log 7 6 > log 8 6) := by
  sorry

end NUMINAMATH_CALUDE_exponential_and_logarithm_inequalities_l675_67557


namespace NUMINAMATH_CALUDE_present_ages_sum_l675_67586

theorem present_ages_sum (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : 4 * a = 3 * b) : 
  a + b = 35 := by
  sorry

end NUMINAMATH_CALUDE_present_ages_sum_l675_67586


namespace NUMINAMATH_CALUDE_total_chocolates_l675_67570

theorem total_chocolates (bags : ℕ) (chocolates_per_bag : ℕ) 
  (h1 : bags = 20) (h2 : chocolates_per_bag = 156) :
  bags * chocolates_per_bag = 3120 :=
by sorry

end NUMINAMATH_CALUDE_total_chocolates_l675_67570


namespace NUMINAMATH_CALUDE_log_equation_solution_l675_67516

theorem log_equation_solution (b x : ℝ) 
  (hb_pos : b > 0) 
  (hb_neq_one : b ≠ 1) 
  (hx_neq_one : x ≠ 1) 
  (h_eq : (Real.log x) / (Real.log (b^3)) + (Real.log b) / (Real.log (x^3)) + (Real.log x) / (Real.log b) = 2) : 
  x = b^((6 - 2 * Real.sqrt 5) / 8) :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l675_67516


namespace NUMINAMATH_CALUDE_polygon_has_five_sides_l675_67555

/-- The set T of points (x, y) satisfying the given conditions -/
def T (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    a / 3 ≤ x ∧ x ≤ 5 * a / 2 ∧
    a / 3 ≤ y ∧ y ≤ 5 * a / 2 ∧
    x + y ≥ 3 * a / 2 ∧
    x + 2 * a ≥ 2 * y ∧
    2 * y + 2 * a ≥ 3 * x}

/-- The theorem stating that the polygon formed by T has 5 sides -/
theorem polygon_has_five_sides (a : ℝ) (ha : a > 0) :
  ∃ (vertices : Finset (ℝ × ℝ)), vertices.card = 5 ∧
  (∀ p ∈ T a, p ∈ convexHull ℝ (↑vertices : Set (ℝ × ℝ))) ∧
  (∀ v ∈ vertices, v ∈ T a) :=
sorry

end NUMINAMATH_CALUDE_polygon_has_five_sides_l675_67555


namespace NUMINAMATH_CALUDE_javier_first_throw_l675_67519

/-- Represents the distances of three javelin throws -/
structure JavelinThrows where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Conditions for Javier's javelin throws -/
def javierThrows (t : JavelinThrows) : Prop :=
  t.first = 2 * t.second ∧
  t.first = 1/2 * t.third ∧
  t.first + t.second + t.third = 1050

/-- Theorem stating that Javier's first throw was 300 meters -/
theorem javier_first_throw :
  ∀ t : JavelinThrows, javierThrows t → t.first = 300 := by
  sorry

end NUMINAMATH_CALUDE_javier_first_throw_l675_67519


namespace NUMINAMATH_CALUDE_age_puzzle_l675_67578

/-- The age of a person satisfying a specific age-related equation --/
theorem age_puzzle : ∃ A : ℕ, 5 * (A + 5) - 5 * (A - 5) = A ∧ A = 50 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l675_67578


namespace NUMINAMATH_CALUDE_orange_juice_problem_l675_67592

theorem orange_juice_problem (jug_volume : ℚ) (portion_drunk : ℚ) :
  jug_volume = 2/7 →
  portion_drunk = 5/8 →
  portion_drunk * jug_volume = 5/28 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_problem_l675_67592


namespace NUMINAMATH_CALUDE_irrational_approximation_l675_67528

theorem irrational_approximation 
  (r₁ r₂ : ℝ) 
  (h_irrational : Irrational (r₁ / r₂)) :
  ∀ (x p : ℝ), p > 0 → ∃ (k₁ k₂ : ℤ), |x - (↑k₁ * r₁ + ↑k₂ * r₂)| < p := by
  sorry

end NUMINAMATH_CALUDE_irrational_approximation_l675_67528


namespace NUMINAMATH_CALUDE_charity_donation_proof_l675_67508

/-- Calculates the donation amount for a charity draw ticket given initial amount,
    winnings, purchases, and final amount. -/
def calculate_donation (initial_amount : ℤ) (prize : ℤ) (lottery_win : ℤ) 
                       (water_cost : ℤ) (lottery_cost : ℤ) (final_amount : ℤ) : ℤ :=
  initial_amount + prize + lottery_win - water_cost - lottery_cost - final_amount

/-- Proves that the donation for the charity draw ticket was $4 given the problem conditions. -/
theorem charity_donation_proof (initial_amount : ℤ) (prize : ℤ) (lottery_win : ℤ) 
                               (water_cost : ℤ) (lottery_cost : ℤ) (final_amount : ℤ) 
                               (h1 : initial_amount = 10)
                               (h2 : prize = 90)
                               (h3 : lottery_win = 65)
                               (h4 : water_cost = 1)
                               (h5 : lottery_cost = 1)
                               (h6 : final_amount = 94) :
  calculate_donation initial_amount prize lottery_win water_cost lottery_cost final_amount = 4 :=
by sorry

#eval calculate_donation 10 90 65 1 1 94

end NUMINAMATH_CALUDE_charity_donation_proof_l675_67508


namespace NUMINAMATH_CALUDE_solve_for_m_l675_67515

/-- Given that x = -2, y = 1, and mx + 3y = 7, prove that m = -2 -/
theorem solve_for_m (x y m : ℝ) 
  (hx : x = -2) 
  (hy : y = 1) 
  (heq : m * x + 3 * y = 7) : 
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_m_l675_67515


namespace NUMINAMATH_CALUDE_linear_function_inverse_sum_l675_67550

/-- Given a linear function f and its inverse f⁻¹, prove that a + b + c = 0 --/
theorem linear_function_inverse_sum (a b c : ℝ) 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x + b)
  (h2 : ∀ x, f_inv x = b * x + a + c)
  (h3 : ∀ x, f (f_inv x) = x) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_inverse_sum_l675_67550


namespace NUMINAMATH_CALUDE_fraction_problem_l675_67553

theorem fraction_problem (a b c : ℕ) : 
  a ≤ 10 ∧ b ≤ 10 ∧ c ≤ 10 →
  (3 * a + 2 : ℚ) / 3 = (4 * b + 3 : ℚ) / 4 ∧ 
  (3 * a + 2 : ℚ) / 3 = (5 * c + 3 : ℚ) / 5 →
  (2 * a + b : ℚ) / c = 19 / 4 := by
sorry

#eval (19 : ℚ) / 4  -- This should output 4.75

end NUMINAMATH_CALUDE_fraction_problem_l675_67553


namespace NUMINAMATH_CALUDE_tim_balloon_count_l675_67593

theorem tim_balloon_count (dan_balloons : ℕ) (tim_multiplier : ℕ) (h1 : dan_balloons = 29) (h2 : tim_multiplier = 7) : 
  dan_balloons * tim_multiplier = 203 := by
  sorry

end NUMINAMATH_CALUDE_tim_balloon_count_l675_67593


namespace NUMINAMATH_CALUDE_g_properties_l675_67564

def f (n : ℕ) : ℕ := (Nat.factorial n)^2

def g (x : ℕ+) : ℚ := (f (x + 1) : ℚ) / (f x : ℚ)

theorem g_properties :
  (g 1 = 4) ∧
  (g 2 = 9) ∧
  (g 3 = 16) ∧
  (∀ ε > 0, ∃ N : ℕ+, ∀ x ≥ N, g x > ε) :=
sorry

end NUMINAMATH_CALUDE_g_properties_l675_67564


namespace NUMINAMATH_CALUDE_operation_laws_l675_67595

theorem operation_laws (a b : ℝ) :
  ((25 * b) * 8 = b * (25 * 8)) ∧
  (a * 6 + 6 * 15 = 6 * (a + 15)) ∧
  (1280 / 16 / 8 = 1280 / (16 * 8)) := by
  sorry

end NUMINAMATH_CALUDE_operation_laws_l675_67595


namespace NUMINAMATH_CALUDE_jordan_rectangle_width_l675_67559

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem jordan_rectangle_width : 
  ∀ (carol_rect jordan_rect : Rectangle),
    carol_rect.length = 5 →
    carol_rect.width = 24 →
    jordan_rect.length = 12 →
    area carol_rect = area jordan_rect →
    jordan_rect.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_jordan_rectangle_width_l675_67559


namespace NUMINAMATH_CALUDE_parabola_equation_l675_67504

/-- Given a parabola with directrix x = -7, its standard equation is y^2 = 28x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ x = -7) →  -- directrix equation
  (∃ a b c : ℝ, ∀ x y, p (x, y) ↔ y^2 = 28*x + b*y + c) -- standard form of parabola
  :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l675_67504


namespace NUMINAMATH_CALUDE_doubling_condition_iff_triangle_or_quadrilateral_l675_67544

/-- The sum of interior angles of an n-sided polygon is (n-2) * 180°. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A polygon satisfies the doubling condition if the sum of angles after doubling
    the sides is an integer multiple of the original sum of angles. -/
def satisfies_doubling_condition (m : ℕ) : Prop :=
  ∃ k : ℕ, sum_interior_angles (2 * m) = k * sum_interior_angles m

/-- Theorem: A polygon satisfies the doubling condition if and only if
    it has 3 or 4 sides. -/
theorem doubling_condition_iff_triangle_or_quadrilateral (m : ℕ) :
  satisfies_doubling_condition m ↔ m = 3 ∨ m = 4 :=
sorry

end NUMINAMATH_CALUDE_doubling_condition_iff_triangle_or_quadrilateral_l675_67544


namespace NUMINAMATH_CALUDE_hockey_league_games_l675_67506

/-- Calculate the number of games in a hockey league season -/
theorem hockey_league_games (num_teams : ℕ) (face_times : ℕ) : 
  num_teams = 18 → face_times = 10 → 
  (num_teams * (num_teams - 1) / 2) * face_times = 1530 :=
by sorry

end NUMINAMATH_CALUDE_hockey_league_games_l675_67506


namespace NUMINAMATH_CALUDE_max_power_under_500_l675_67524

theorem max_power_under_500 :
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 1 ∧ 
    a^b < 500 ∧
    (∀ (c d : ℕ), c > 0 → d > 1 → c^d < 500 → c^d ≤ a^b) ∧
    a = 22 ∧ b = 2 ∧ 
    a + b = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_power_under_500_l675_67524


namespace NUMINAMATH_CALUDE_pens_and_pencils_equation_system_l675_67543

theorem pens_and_pencils_equation_system (x y : ℕ) : 
  (x + y = 30 ∧ x = 2 * y - 3) ↔ 
  (x + y = 30 ∧ x = 2 * y - 3 ∧ x < 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_pens_and_pencils_equation_system_l675_67543


namespace NUMINAMATH_CALUDE_quadratic_equation_with_absolute_roots_l675_67500

theorem quadratic_equation_with_absolute_roots 
  (x₁ x₂ m : ℝ) 
  (h₁ : x₁ > 0) 
  (h₂ : x₂ < 0) 
  (h₃ : ∃ (original_eq : ℝ → Prop), original_eq x₁ ∧ original_eq x₂) :
  ∃ (new_eq : ℝ → Prop), 
    new_eq (|x₁|) ∧ 
    new_eq (|x₂|) ∧ 
    ∀ x, new_eq x ↔ x^2 - (1 - 4*m)/x + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_absolute_roots_l675_67500


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l675_67536

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^2 - 22 * X + 64 = (X - 3) * q + 25 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l675_67536


namespace NUMINAMATH_CALUDE_factorization_problems_l675_67532

theorem factorization_problems (x y : ℝ) :
  (x^2 - 4 = (x + 2) * (x - 2)) ∧
  (3 * x^2 - 6 * x * y + 3 * y^2 = 3 * (x - y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l675_67532


namespace NUMINAMATH_CALUDE_horner_v₂_eq_40_l675_67523

/-- Horner's method for a polynomial of degree 6 -/
def horner (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) (x : ℝ) : ℝ :=
  a₀ + x * (a₁ + x * (a₂ + x * (a₃ + x * (a₄ + x * (a₅ + x * a₆)))))

/-- The second Horner value for a polynomial of degree 6 -/
def v₂ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) (x : ℝ) : ℝ :=
  a₂ + x * (a₃ + x * (a₄ + x * (a₅ + x * a₆))) - 
  x * (a₁ + x * (a₂ + x * (a₃ + x * (a₄ + x * (a₅ + x * a₆)))))

theorem horner_v₂_eq_40 :
  v₂ 64 (-192) 240 (-160) 60 (-12) 1 2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_horner_v₂_eq_40_l675_67523


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l675_67590

theorem sin_cos_sum_equals_half : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
  Real.sin (70 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l675_67590


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_250_125_l675_67574

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem largest_two_digit_prime_factor_of_binomial_250_125 :
  ∃ (p : ℕ), p = 83 ∧ 
  Nat.Prime p ∧
  10 ≤ p ∧ p < 100 ∧
  p ∣ binomial 250 125 ∧
  ∀ (q : ℕ), Nat.Prime q → 10 ≤ q → q < 100 → q ∣ binomial 250 125 → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_250_125_l675_67574


namespace NUMINAMATH_CALUDE_range_of_m_l675_67573

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m * |Real.sin x + 2| - 9 < 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 1 < 0) →
  m < -1 ∨ (1 < m ∧ m < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l675_67573


namespace NUMINAMATH_CALUDE_faucet_fill_time_l675_67512

/-- Proves that given four faucets can fill a 120-gallon tub in 8 minutes, 
    eight faucets will fill a 30-gallon tub in 60 seconds. -/
theorem faucet_fill_time : 
  ∀ (faucets_1 faucets_2 : ℕ) 
    (tub_1 tub_2 : ℝ) 
    (time_1 : ℝ) 
    (time_2 : ℝ),
  faucets_1 = 4 →
  faucets_2 = 8 →
  tub_1 = 120 →
  tub_2 = 30 →
  time_1 = 8 →
  (faucets_1 : ℝ) * tub_2 * time_1 = faucets_2 * tub_1 * time_2 →
  time_2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_faucet_fill_time_l675_67512


namespace NUMINAMATH_CALUDE_sine_sum_identity_l675_67587

theorem sine_sum_identity (α β γ : ℝ) (h : α + β + γ = 0) :
  Real.sin α + Real.sin β + Real.sin γ = -4 * Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_identity_l675_67587


namespace NUMINAMATH_CALUDE_sum_inequality_l675_67597

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  (1 / (b * c + a + 1 / a) + 1 / (a * c + b + 1 / b) + 1 / (a * b + c + 1 / c)) ≤ 27 / 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l675_67597


namespace NUMINAMATH_CALUDE_stratified_sampling_admin_count_l675_67513

theorem stratified_sampling_admin_count 
  (total_employees : ℕ) 
  (admin_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 120) 
  (h2 : admin_employees = 40) 
  (h3 : sample_size = 24) : 
  ℕ :=
  by
    sorry

#check stratified_sampling_admin_count

end NUMINAMATH_CALUDE_stratified_sampling_admin_count_l675_67513


namespace NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l675_67582

/-- Given a geometric sequence with first term 12 and second term 4,
    prove that its eighth term is 4/729. -/
theorem eighth_term_of_geometric_sequence : 
  ∀ (a : ℕ → ℚ), 
    (∀ n, a (n + 2) * a n = (a (n + 1))^2) →  -- geometric sequence condition
    a 1 = 12 →                                -- first term
    a 2 = 4 →                                 -- second term
    a 8 = 4/729 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l675_67582


namespace NUMINAMATH_CALUDE_expression_equality_1_expression_equality_2_l675_67526

-- Part 1
theorem expression_equality_1 : 
  2 * Real.sin (45 * π / 180) - (π - Real.sqrt 5) ^ 0 + (1/2)⁻¹ + |Real.sqrt 2 - 1| = 2 * Real.sqrt 2 := by
  sorry

-- Part 2
theorem expression_equality_2 (a b : ℝ) : 
  (2*a + 3*b) * (3*a - 2*b) = 6*a^2 + 5*a*b - 6*b^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_1_expression_equality_2_l675_67526
