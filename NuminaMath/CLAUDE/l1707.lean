import Mathlib

namespace NUMINAMATH_CALUDE_max_k_value_l1707_170705

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + 2 * k * (x / y + y / x)) :
  k ≤ Real.sqrt (5 / 6) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1707_170705


namespace NUMINAMATH_CALUDE_part_one_part_two_l1707_170787

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Part I
theorem part_one (m : ℝ) (h : m = 3) : A ∩ (U \ B m) = Set.Icc 3 4 := by sorry

-- Part II
theorem part_two (m : ℝ) (h : A ∩ B m = ∅) : m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1707_170787


namespace NUMINAMATH_CALUDE_equal_area_division_l1707_170785

/-- The value of c that divides the area of five unit squares into two equal regions -/
def c : ℝ := 1.75

/-- The total area of the five unit squares -/
def total_area : ℝ := 5

/-- The equation of the line passing through (c, 0) and (3, 4) -/
def line_equation (x y : ℝ) : Prop := y = (4 / (3 - c)) * (x - c)

/-- The area of the triangle formed by the line and the x-axis -/
def triangle_area : ℝ := 2 * (3 - c)

theorem equal_area_division :
  triangle_area = total_area / 2 ↔ c = 1.75 :=
sorry

end NUMINAMATH_CALUDE_equal_area_division_l1707_170785


namespace NUMINAMATH_CALUDE_unique_solution_l1707_170709

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, f (a^2) (f b c + 1) = a^2 * (b * c + 1)

/-- Theorem stating that the only function satisfying the equation is f(a,b) = a*b -/
theorem unique_solution {f : ℝ → ℝ → ℝ} (hf : SatisfiesEquation f) :
  ∀ a b : ℝ, f a b = a * b := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1707_170709


namespace NUMINAMATH_CALUDE_surf_festival_problem_l1707_170752

/-- The Rip Curl Myrtle Beach Surf Festival problem -/
theorem surf_festival_problem (total_surfers : ℝ) (S1 : ℝ) :
  total_surfers = 15000 ∧
  S1 + 0.9 * S1 + 1.5 * S1 + (S1 + 0.9 * S1) + 0.5 * (S1 + 0.9 * S1) = total_surfers →
  S1 = 2400 ∧ total_surfers / 5 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_surf_festival_problem_l1707_170752


namespace NUMINAMATH_CALUDE_max_pieces_with_three_cuts_l1707_170701

/-- Represents a cake that can be cut -/
structure Cake :=
  (volume : ℝ)
  (height : ℝ)
  (width : ℝ)
  (depth : ℝ)

/-- Represents a cut made to the cake -/
inductive Cut
  | Horizontal
  | Vertical
  | Parallel

/-- The number of pieces resulting from a series of cuts -/
def num_pieces (cuts : List Cut) : ℕ :=
  2 ^ (cuts.length)

/-- The maximum number of identical pieces obtainable with 3 cuts -/
def max_pieces : ℕ := 8

/-- Theorem: The maximum number of identical pieces obtainable from a cake with 3 cuts is 8 -/
theorem max_pieces_with_three_cuts (c : Cake) :
  ∀ (cuts : List Cut), cuts.length = 3 → num_pieces cuts ≤ max_pieces :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_with_three_cuts_l1707_170701


namespace NUMINAMATH_CALUDE_pie_sale_profit_l1707_170753

/-- Calculates the profit from selling pies given the number of pies, costs, and selling price -/
theorem pie_sale_profit
  (num_pumpkin : ℕ)
  (num_cherry : ℕ)
  (cost_pumpkin : ℕ)
  (cost_cherry : ℕ)
  (selling_price : ℕ)
  (h1 : num_pumpkin = 10)
  (h2 : num_cherry = 12)
  (h3 : cost_pumpkin = 3)
  (h4 : cost_cherry = 5)
  (h5 : selling_price = 5) :
  (num_pumpkin + num_cherry) * selling_price - (num_pumpkin * cost_pumpkin + num_cherry * cost_cherry) = 20 :=
by sorry

end NUMINAMATH_CALUDE_pie_sale_profit_l1707_170753


namespace NUMINAMATH_CALUDE_document_delivery_equation_l1707_170776

theorem document_delivery_equation (x : ℝ) (h : x > 3) : 
  let distance : ℝ := 900
  let slow_horse_time : ℝ := x + 1
  let fast_horse_time : ℝ := x - 3
  let slow_horse_speed : ℝ := distance / slow_horse_time
  let fast_horse_speed : ℝ := distance / fast_horse_time
  fast_horse_speed = 2 * slow_horse_speed →
  (distance / slow_horse_time) * 2 = distance / fast_horse_time :=
by sorry


end NUMINAMATH_CALUDE_document_delivery_equation_l1707_170776


namespace NUMINAMATH_CALUDE_bicycle_profit_calculation_l1707_170764

theorem bicycle_profit_calculation (profit_A profit_B final_price : ℝ) :
  profit_A = 0.60 ∧ profit_B = 0.25 ∧ final_price = 225 →
  ∃ cost_price_A : ℝ,
    cost_price_A * (1 + profit_A) * (1 + profit_B) = final_price ∧
    cost_price_A = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_profit_calculation_l1707_170764


namespace NUMINAMATH_CALUDE_basketball_court_equation_rewrite_l1707_170778

theorem basketball_court_equation_rewrite :
  ∃ (a b c : ℤ), a > 0 ∧
  (∀ x : ℝ, 16 * x^2 + 32 * x - 40 = 0 ↔ (a * x + b)^2 = c) ∧
  a + b + c = 64 := by
  sorry

end NUMINAMATH_CALUDE_basketball_court_equation_rewrite_l1707_170778


namespace NUMINAMATH_CALUDE_triangle_properties_l1707_170733

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 3 ∧
  (t.c - 2 * t.a) * Real.cos t.B + t.b * Real.cos t.C = 0

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π / 3 ∧ Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1707_170733


namespace NUMINAMATH_CALUDE_square_of_sum_fifteen_seven_l1707_170796

theorem square_of_sum_fifteen_seven : 15^2 + 2*(15*7) + 7^2 = 484 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_fifteen_seven_l1707_170796


namespace NUMINAMATH_CALUDE_prob_two_gold_given_at_least_one_gold_l1707_170792

/-- The probability of selecting two gold medals given that at least one gold medal is selected -/
theorem prob_two_gold_given_at_least_one_gold 
  (total_medals : ℕ) 
  (gold_medals : ℕ) 
  (silver_medals : ℕ) 
  (bronze_medals : ℕ) 
  (h1 : total_medals = gold_medals + silver_medals + bronze_medals)
  (h2 : total_medals = 10)
  (h3 : gold_medals = 5)
  (h4 : silver_medals = 3)
  (h5 : bronze_medals = 2) :
  (Nat.choose gold_medals 2 : ℚ) / (Nat.choose total_medals 2 - Nat.choose (silver_medals + bronze_medals) 2) = 2/7 :=
sorry

end NUMINAMATH_CALUDE_prob_two_gold_given_at_least_one_gold_l1707_170792


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1707_170717

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Define the line on which the center of the desired circle lies
def centerLine (x y : ℝ) : Prop := x + y = 0

-- Define the desired circle
def desiredCircle (x y : ℝ) : Prop := (x + 3)^2 + (y - 3)^2 = 10

-- Theorem statement
theorem circle_equation_proof :
  ∃ (cx cy : ℝ),
    -- The center is on the line x + y = 0
    centerLine cx cy ∧
    -- The circle passes through the intersection points of circle1 and circle2
    (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → desiredCircle x y) ∧
    -- The equation (x + 3)² + (y - 3)² = 10 represents the desired circle
    (∀ (x y : ℝ), desiredCircle x y ↔ (x - cx)^2 + (y - cy)^2 = 10) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1707_170717


namespace NUMINAMATH_CALUDE_angle_A_value_max_area_l1707_170713

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + (t.c - 2 * t.b) * Real.cos t.A = 0

-- Theorem 1: If the condition is satisfied, then A = π/3
theorem angle_A_value (t : Triangle) (h : satisfies_condition t) : t.A = π / 3 :=
sorry

-- Theorem 2: If a = 2 and the condition is satisfied, the maximum area is √3
theorem max_area (t : Triangle) (h1 : satisfies_condition t) (h2 : t.a = 2) :
  (∀ t' : Triangle, satisfies_condition t' → t'.a = 2 → 
    1 / 2 * t'.b * t'.c * Real.sin t'.A ≤ Real.sqrt 3) ∧
  (∃ t' : Triangle, satisfies_condition t' ∧ t'.a = 2 ∧
    1 / 2 * t'.b * t'.c * Real.sin t'.A = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_angle_A_value_max_area_l1707_170713


namespace NUMINAMATH_CALUDE_toms_video_game_spending_l1707_170746

/-- The cost of the Batman game in dollars -/
def batman_cost : ℚ := 13.60

/-- The cost of the Superman game in dollars -/
def superman_cost : ℚ := 5.06

/-- The number of games Tom already owns -/
def existing_games : ℕ := 2

/-- The total amount Tom spent on video games -/
def total_spent : ℚ := batman_cost + superman_cost

theorem toms_video_game_spending :
  total_spent = 18.66 := by sorry

end NUMINAMATH_CALUDE_toms_video_game_spending_l1707_170746


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1707_170798

/-- Given a geometric sequence {a_n} where a_2 and a_6 are roots of x^2 - 34x + 64 = 0, a_4 = 8 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- a is a geometric sequence
  (a 2 * a 2 - 34 * a 2 + 64 = 0) →  -- a_2 is a root of x^2 - 34x + 64 = 0
  (a 6 * a 6 - 34 * a 6 + 64 = 0) →  -- a_6 is a root of x^2 - 34x + 64 = 0
  (a 4 = 8) := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l1707_170798


namespace NUMINAMATH_CALUDE_function_satisfying_conditions_l1707_170755

theorem function_satisfying_conditions : ∃ (f : ℝ → ℝ), 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (2 - x) + f x = 0) ∧ 
  (∀ x, f x = Real.cos (Real.pi / 2 * x)) ∧
  (∃ x y, f x ≠ f y) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_conditions_l1707_170755


namespace NUMINAMATH_CALUDE_rational_division_equality_l1707_170758

theorem rational_division_equality : 
  (-2 / 21) / (1 / 6 - 3 / 14 + 2 / 3 - 9 / 7) = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_rational_division_equality_l1707_170758


namespace NUMINAMATH_CALUDE_find_A_l1707_170750

theorem find_A (A B : ℕ) (h1 : A + B = 1149) (h2 : A = 8 * B + 24) : A = 1024 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l1707_170750


namespace NUMINAMATH_CALUDE_melanie_turnips_count_l1707_170702

/-- The number of turnips Benny grew -/
def benny_turnips : ℕ := 113

/-- The additional number of turnips Melanie grew compared to Benny -/
def melanie_extra_turnips : ℕ := 26

/-- The number of turnips Melanie grew -/
def melanie_turnips : ℕ := benny_turnips + melanie_extra_turnips

theorem melanie_turnips_count : melanie_turnips = 139 := by
  sorry

end NUMINAMATH_CALUDE_melanie_turnips_count_l1707_170702


namespace NUMINAMATH_CALUDE_genetic_material_distribution_l1707_170768

/-- Represents a cell in a diploid organism -/
structure DiploidCell where
  genetic_material : Set (α : Type)
  cytoplasm : Set (α : Type)

/-- Represents the process of cell division -/
def cell_division (parent : DiploidCell) : DiploidCell × DiploidCell :=
  sorry

/-- Predicate to check if the distribution is random and unequal -/
def is_random_unequal_distribution (parent : DiploidCell) (daughter1 daughter2 : DiploidCell) : Prop :=
  sorry

/-- Theorem stating that genetic material in cytoplasm is distributed randomly and unequally during cell division -/
theorem genetic_material_distribution (cell : DiploidCell) :
  let (daughter1, daughter2) := cell_division cell
  is_random_unequal_distribution cell daughter1 daughter2 := by
  sorry

end NUMINAMATH_CALUDE_genetic_material_distribution_l1707_170768


namespace NUMINAMATH_CALUDE_subtraction_and_divisibility_imply_sum_l1707_170711

/-- A number is divisible by 11 if and only if the alternating sum of its digits is divisible by 11 -/
def divisible_by_11 (n : ℕ) : Prop :=
  (n % 11 = 0)

/-- Returns the hundreds digit of a three-digit number -/
def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

/-- Returns the tens digit of a three-digit number -/
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Returns the ones digit of a three-digit number -/
def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem subtraction_and_divisibility_imply_sum (c d : ℕ) :
  (745 - (300 + c * 10 + 4) = 400 + d * 10 + 1) →
  divisible_by_11 (400 + d * 10 + 1) →
  c + d = 14 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_and_divisibility_imply_sum_l1707_170711


namespace NUMINAMATH_CALUDE_median_and_midline_projection_l1707_170724

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a parallel projection
def ParallelProjection := (ℝ × ℝ) → (ℝ × ℝ)

-- Define a median of a triangle
def median (t : Triangle) : (ℝ × ℝ) := sorry

-- Define a midline of a triangle
def midline (t : Triangle) : (ℝ × ℝ) := sorry

-- Theorem statement
theorem median_and_midline_projection 
  (t : Triangle) 
  (p : ParallelProjection) 
  (h : ∃ t', t' = Triangle.mk (p t.A) (p t.B) (p t.C)) :
  (p (median t) = median (Triangle.mk (p t.A) (p t.B) (p t.C))) ∧
  (p (midline t) = midline (Triangle.mk (p t.A) (p t.B) (p t.C))) := by
  sorry

end NUMINAMATH_CALUDE_median_and_midline_projection_l1707_170724


namespace NUMINAMATH_CALUDE_find_lesser_number_l1707_170706

theorem find_lesser_number (x y : ℝ) : 
  x + y = 60 → 
  x - y = 10 → 
  min x y = 25 := by
sorry

end NUMINAMATH_CALUDE_find_lesser_number_l1707_170706


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l1707_170735

/-- Two arithmetic sequences and their sums -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → S n / T n = (2 * n + 1 : ℚ) / (n + 2 : ℚ)

theorem arithmetic_sequences_ratio 
  (a b : ℕ → ℚ) (S T : ℕ → ℚ) 
  (h : arithmetic_sequences a b S T) : 
  a 7 / b 7 = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l1707_170735


namespace NUMINAMATH_CALUDE_reception_friends_l1707_170747

def wedding_reception (total_attendees : ℕ) (bride_couples : ℕ) (groom_couples : ℕ) : Prop :=
  let family_members := 2 * (bride_couples + groom_couples)
  let friends := total_attendees - family_members
  friends = 100

theorem reception_friends :
  wedding_reception 180 20 20 := by
  sorry

end NUMINAMATH_CALUDE_reception_friends_l1707_170747


namespace NUMINAMATH_CALUDE_gcd_2048_2101_l1707_170765

theorem gcd_2048_2101 : Nat.gcd 2048 2101 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2048_2101_l1707_170765


namespace NUMINAMATH_CALUDE_gyeong_hun_climb_l1707_170723

/-- Gyeong-hun's mountain climbing problem -/
theorem gyeong_hun_climb (uphill_speed downhill_speed : ℝ)
                         (downhill_extra_distance total_time : ℝ)
                         (h1 : uphill_speed = 3)
                         (h2 : downhill_speed = 4)
                         (h3 : downhill_extra_distance = 2)
                         (h4 : total_time = 4) :
  ∃ (distance : ℝ),
    distance / uphill_speed + (distance + downhill_extra_distance) / downhill_speed = total_time ∧
    distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_gyeong_hun_climb_l1707_170723


namespace NUMINAMATH_CALUDE_max_speed_theorem_l1707_170718

/-- Represents a data point of (speed, defective items) -/
structure DataPoint where
  speed : ℝ
  defective : ℝ

/-- The set of observed data points -/
def observed_data : List DataPoint := [
  ⟨8, 5⟩, ⟨12, 8⟩, ⟨14, 9⟩, ⟨16, 11⟩
]

/-- Calculates the slope of the regression line -/
def calculate_slope (data : List DataPoint) : ℝ := sorry

/-- Calculates the y-intercept of the regression line -/
def calculate_intercept (data : List DataPoint) (slope : ℝ) : ℝ := sorry

/-- The maximum number of defective items allowed per hour -/
def max_defective : ℝ := 10

theorem max_speed_theorem (data : List DataPoint) 
    (h_linear : ∃ (m b : ℝ), ∀ point ∈ data, point.defective = m * point.speed + b) :
  let slope := calculate_slope data
  let intercept := calculate_intercept data slope
  let max_speed := (max_defective - intercept) / slope
  ⌊max_speed⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_max_speed_theorem_l1707_170718


namespace NUMINAMATH_CALUDE_largest_coin_distribution_l1707_170781

theorem largest_coin_distribution (n : ℕ) : n ≤ 111 ↔ 
  (∃ (k : ℕ), n = 12 * k + 3 ∧ n < 120) :=
sorry

end NUMINAMATH_CALUDE_largest_coin_distribution_l1707_170781


namespace NUMINAMATH_CALUDE_plane_properties_l1707_170799

structure Plane

-- Define parallel and perpendicular relations for planes
def parallel (p q : Plane) : Prop := sorry
def perpendicular (p q : Plane) : Prop := sorry

-- Define line as intersection of two planes
def line_intersection (p q : Plane) : Type := sorry

-- Define parallel relation for lines
def line_parallel (l m : Type) : Prop := sorry

theorem plane_properties (α β γ : Plane) (hd : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (parallel α β ∧ parallel β γ → parallel α γ) ∧
  (parallel α β → ∀ (a b : Type), a = line_intersection α γ → b = line_intersection β γ → line_parallel a b) ∧
  (parallel α β ∧ perpendicular β γ → perpendicular α γ) ∧
  ¬(∀ α β γ : Plane, perpendicular α β ∧ perpendicular β γ → perpendicular α γ) := by
  sorry

end NUMINAMATH_CALUDE_plane_properties_l1707_170799


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l1707_170775

theorem perfect_square_binomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 10*x + k = (x + a)^2) ↔ k = 25 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l1707_170775


namespace NUMINAMATH_CALUDE_opposite_of_negative_sqrt_three_l1707_170715

theorem opposite_of_negative_sqrt_three : -(-(Real.sqrt 3)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_sqrt_three_l1707_170715


namespace NUMINAMATH_CALUDE_center_square_area_ratio_l1707_170712

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  side : ℝ
  cross_width : ℝ
  center_side : ℝ
  cross_area_ratio : ℝ
  cross_symmetric : Bool
  cross_uniform : Bool

/-- The theorem stating that if a symmetric cross occupies 50% of a square flag's area, 
    the center square occupies 6.25% of the total area -/
theorem center_square_area_ratio (flag : SquareFlag) 
  (h1 : flag.cross_area_ratio = 0.5)
  (h2 : flag.cross_symmetric = true)
  (h3 : flag.cross_uniform = true)
  (h4 : flag.center_side = flag.side / 4) :
  (flag.center_side ^ 2) / (flag.side ^ 2) = 0.0625 := by
  sorry

end NUMINAMATH_CALUDE_center_square_area_ratio_l1707_170712


namespace NUMINAMATH_CALUDE_fourth_child_age_is_eight_l1707_170721

/-- The age of the first child -/
def first_child_age : ℕ := 15

/-- The age difference between the first and second child -/
def age_diff_first_second : ℕ := 1

/-- The age of the second child when the third child was born -/
def second_child_age_at_third_birth : ℕ := 4

/-- The age difference between the third and fourth child -/
def age_diff_third_fourth : ℕ := 2

/-- The age of the fourth child -/
def fourth_child_age : ℕ := first_child_age - age_diff_first_second - second_child_age_at_third_birth - age_diff_third_fourth

theorem fourth_child_age_is_eight : fourth_child_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_child_age_is_eight_l1707_170721


namespace NUMINAMATH_CALUDE_ellipse_b_squared_value_l1707_170795

/-- The squared semi-minor axis of an ellipse with equation (x^2/25) + (y^2/b^2) = 1,
    which has the same foci as a hyperbola with equation (x^2/225) - (y^2/144) = 1/36 -/
def ellipse_b_squared : ℝ := 14.75

/-- The equation of the ellipse -/
def is_on_ellipse (x y b : ℝ) : Prop :=
  x^2 / 25 + y^2 / b^2 = 1

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 225 - y^2 / 144 = 1 / 36

/-- The foci of the ellipse and hyperbola coincide -/
axiom foci_coincide : ∃ c : ℝ,
  c^2 = 25 - ellipse_b_squared ∧
  c^2 = 225 / 36 - 144 / 36

theorem ellipse_b_squared_value :
  ellipse_b_squared = 14.75 := by sorry

end NUMINAMATH_CALUDE_ellipse_b_squared_value_l1707_170795


namespace NUMINAMATH_CALUDE_final_silver_count_l1707_170763

/-- Represents the number of tokens Alex has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents a token exchange booth -/
structure Booth where
  redIn : ℕ
  blueIn : ℕ
  redOut : ℕ
  blueOut : ℕ
  silverOut : ℕ

/-- Checks if an exchange is possible at a given booth -/
def canExchange (tokens : TokenCount) (booth : Booth) : Prop :=
  tokens.red ≥ booth.redIn ∧ tokens.blue ≥ booth.blueIn

/-- Performs an exchange at a given booth -/
def exchange (tokens : TokenCount) (booth : Booth) : TokenCount :=
  { red := tokens.red - booth.redIn + booth.redOut,
    blue := tokens.blue - booth.blueIn + booth.blueOut,
    silver := tokens.silver + booth.silverOut }

/-- Checks if any exchanges are possible -/
def exchangesPossible (tokens : TokenCount) (booths : List Booth) : Prop :=
  ∃ b ∈ booths, canExchange tokens b

/-- The main theorem to prove -/
theorem final_silver_count 
  (initialTokens : TokenCount)
  (booth1 booth2 : Booth)
  (h_initial : initialTokens = ⟨75, 75, 0⟩)
  (h_booth1 : booth1 = ⟨2, 0, 0, 1, 1⟩)
  (h_booth2 : booth2 = ⟨0, 3, 1, 0, 1⟩)
  : ∃ (finalTokens : TokenCount), 
    (¬ exchangesPossible finalTokens [booth1, booth2]) ∧ 
    finalTokens.silver = 103 := by
  sorry

end NUMINAMATH_CALUDE_final_silver_count_l1707_170763


namespace NUMINAMATH_CALUDE_positive_y_squared_geq_2y_minus_1_l1707_170738

theorem positive_y_squared_geq_2y_minus_1 :
  ∀ y : ℝ, y > 0 → y^2 ≥ 2*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_y_squared_geq_2y_minus_1_l1707_170738


namespace NUMINAMATH_CALUDE_unanswered_completion_count_l1707_170794

/-- A structure representing a multiple choice test -/
structure MultipleChoiceTest where
  total_questions : Nat
  choices_per_question : Nat
  single_answer_questions : Nat
  multi_select_questions : Nat
  correct_choices_per_multi : Nat

/-- The number of ways to complete the test with all questions unanswered -/
def ways_to_complete_unanswered (test : MultipleChoiceTest) : Nat :=
  1

/-- Theorem stating that there is only one way to complete the test with all questions unanswered -/
theorem unanswered_completion_count (test : MultipleChoiceTest)
  (h1 : test.total_questions = 10)
  (h2 : test.choices_per_question = 8)
  (h3 : test.single_answer_questions = 6)
  (h4 : test.multi_select_questions = 4)
  (h5 : test.correct_choices_per_multi = 2)
  (h6 : test.total_questions = test.single_answer_questions + test.multi_select_questions) :
  ways_to_complete_unanswered test = 1 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_completion_count_l1707_170794


namespace NUMINAMATH_CALUDE_square_fitting_theorem_l1707_170769

theorem square_fitting_theorem :
  ∃ (N : ℕ), N > 0 ∧ (N : ℝ) * (1 / N) ≤ 1 ∧ 4 * N = 1992 := by
  sorry

end NUMINAMATH_CALUDE_square_fitting_theorem_l1707_170769


namespace NUMINAMATH_CALUDE_trevor_taxi_cost_l1707_170793

/-- Calculates the total cost of Trevor's taxi ride downtown -/
def total_taxi_cost (uber_cost lyft_cost taxi_cost detour_rate tip_rate : ℚ) : ℚ :=
  let detour_cost := taxi_cost * detour_rate
  let tip := taxi_cost * tip_rate
  taxi_cost + detour_cost + tip

/-- Proves that the total cost of Trevor's taxi ride downtown is $20.25 -/
theorem trevor_taxi_cost :
  let uber_cost : ℚ := 22
  let lyft_cost : ℚ := uber_cost - 3
  let taxi_cost : ℚ := lyft_cost - 4
  let detour_rate : ℚ := 15 / 100
  let tip_rate : ℚ := 20 / 100
  total_taxi_cost uber_cost lyft_cost taxi_cost detour_rate tip_rate = 8100 / 400 := by
  sorry

#eval total_taxi_cost 22 19 15 (15/100) (20/100)

end NUMINAMATH_CALUDE_trevor_taxi_cost_l1707_170793


namespace NUMINAMATH_CALUDE_solution_concentration_l1707_170703

theorem solution_concentration (x y : ℝ) : 
  (0.45 * x = 0.15 * (x + y + 1)) ∧ 
  (0.30 * y = 0.05 * (x + y + 1)) → 
  x = 2/3 ∧ y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_concentration_l1707_170703


namespace NUMINAMATH_CALUDE_product_real_implies_b_value_l1707_170788

/-- Given complex numbers z₁ and z₂, if their product is real, then b = -2 -/
theorem product_real_implies_b_value (z₁ z₂ : ℂ) (b : ℝ) 
  (h₁ : z₁ = 1 + I) 
  (h₂ : z₂ = 2 + b * I) 
  (h₃ : (z₁ * z₂).im = 0) : 
  b = -2 := by
  sorry

end NUMINAMATH_CALUDE_product_real_implies_b_value_l1707_170788


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l1707_170766

theorem quadratic_root_existence (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (a^2 - 4*b ≥ 0) ∨ (c^2 - 4*d ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l1707_170766


namespace NUMINAMATH_CALUDE_cos_B_in_triangle_l1707_170720

theorem cos_B_in_triangle (A B C : ℝ) (AC BC : ℝ) (angle_A : ℝ) :
  AC = Real.sqrt 2 →
  BC = Real.sqrt 3 →
  angle_A = π / 3 →
  Real.cos B = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cos_B_in_triangle_l1707_170720


namespace NUMINAMATH_CALUDE_tom_initial_money_l1707_170745

/-- Tom's initial amount of money -/
def initial_money : ℕ := sorry

/-- Cost of the game Tom bought -/
def game_cost : ℕ := 49

/-- Cost of each toy -/
def toy_cost : ℕ := 4

/-- Number of toys Tom could buy after purchasing the game -/
def num_toys : ℕ := 2

/-- Theorem stating that Tom's initial money was $57 -/
theorem tom_initial_money : 
  initial_money = game_cost + num_toys * toy_cost :=
sorry

end NUMINAMATH_CALUDE_tom_initial_money_l1707_170745


namespace NUMINAMATH_CALUDE_floor_sqrt_19_squared_l1707_170791

theorem floor_sqrt_19_squared : ⌊Real.sqrt 19⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_19_squared_l1707_170791


namespace NUMINAMATH_CALUDE_place_mat_length_l1707_170744

/-- The radius of the round table -/
def table_radius : ℝ := 5

/-- The width of each place mat -/
def mat_width : ℝ := 1.5

/-- The number of place mats -/
def num_mats : ℕ := 4

/-- The theorem stating the length of each place mat -/
theorem place_mat_length :
  ∃ y : ℝ,
    y > 0 ∧
    y = 0.75 ∧
    (y + 2.5 * Real.sqrt 2 - mat_width / 2)^2 + (mat_width / 2)^2 = table_radius^2 ∧
    ∀ (i : Fin num_mats),
      ∃ (x y : ℝ),
        x^2 + y^2 = table_radius^2 ∧
        (x - mat_width / 2)^2 + (y - (y + 2.5 * Real.sqrt 2 - mat_width / 2))^2 = table_radius^2 :=
by
  sorry

end NUMINAMATH_CALUDE_place_mat_length_l1707_170744


namespace NUMINAMATH_CALUDE_mans_rowing_rate_l1707_170749

/-- Proves that a man's rowing rate in still water is 11 km/h given his speeds with and against the stream. -/
theorem mans_rowing_rate (with_stream : ℝ) (against_stream : ℝ)
  (h_with : with_stream = 18)
  (h_against : against_stream = 4) :
  (with_stream + against_stream) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mans_rowing_rate_l1707_170749


namespace NUMINAMATH_CALUDE_fifteenth_term_ratio_l1707_170728

-- Define the sums of arithmetic sequences
def S (a d n : ℚ) : ℚ := n * (2 * a + (n - 1) * d) / 2
def T (b e n : ℚ) : ℚ := n * (2 * b + (n - 1) * e) / 2

-- Define the ratio condition
def ratio_condition (a d b e n : ℚ) : Prop :=
  S a d n / T b e n = (5 * n + 3) / (3 * n + 17)

-- Define the 15th term of each sequence
def term_15 (a d : ℚ) : ℚ := a + 14 * d

-- Theorem statement
theorem fifteenth_term_ratio 
  (a d b e : ℚ) 
  (h : ∀ n : ℚ, ratio_condition a d b e n) : 
  term_15 a d / term_15 b e = 44 / 95 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_ratio_l1707_170728


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_is_six_l1707_170762

/-- The cost of each chocolate bar, given the total number of bars, 
    the number of unsold bars, and the total revenue from sales. -/
def chocolate_bar_cost (total_bars : ℕ) (unsold_bars : ℕ) (revenue : ℕ) : ℚ :=
  revenue / (total_bars - unsold_bars)

/-- Theorem stating that the cost of each chocolate bar is $6 under the given conditions. -/
theorem chocolate_bar_cost_is_six : 
  chocolate_bar_cost 13 6 42 = 6 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_is_six_l1707_170762


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l1707_170739

theorem triangle_angle_relation (A B C : ℝ) (h1 : Real.cos A + Real.sin B = 1) 
  (h2 : Real.sin A + Real.cos B = Real.sqrt 3) : Real.cos (A - C) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l1707_170739


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l1707_170716

/-- Given four consecutive even numbers whose sum of squares is 344, their sum is 36 -/
theorem consecutive_even_numbers_sum (n : ℕ) : 
  (n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344) → 
  (n + (n + 2) + (n + 4) + (n + 6) = 36) :=
by
  sorry

#check consecutive_even_numbers_sum

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l1707_170716


namespace NUMINAMATH_CALUDE_integer_fraction_count_l1707_170757

theorem integer_fraction_count : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, 0 < n ∧ n < 50 ∧ ∃ k : ℕ, n = k * (50 - n)) ∧ 
    Finset.card S = 2 := by sorry

end NUMINAMATH_CALUDE_integer_fraction_count_l1707_170757


namespace NUMINAMATH_CALUDE_complex_magnitude_l1707_170700

theorem complex_magnitude (a b : ℝ) (i : ℂ) :
  (i * i = -1) →
  ((a + i) * i = b + a * i) →
  Complex.abs (a + b * i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1707_170700


namespace NUMINAMATH_CALUDE_mariela_get_well_cards_l1707_170743

theorem mariela_get_well_cards (hospital_cards : ℕ) (home_cards : ℕ) 
  (h1 : hospital_cards = 403) (h2 : home_cards = 287) : 
  hospital_cards + home_cards = 690 := by
  sorry

end NUMINAMATH_CALUDE_mariela_get_well_cards_l1707_170743


namespace NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l1707_170773

theorem abs_eq_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l1707_170773


namespace NUMINAMATH_CALUDE_savings_calculation_l1707_170772

theorem savings_calculation (savings : ℚ) (tv_cost : ℚ) : 
  (1 / 4 : ℚ) * savings = tv_cost → 
  tv_cost = 300 → 
  savings = 1200 :=
by sorry

end NUMINAMATH_CALUDE_savings_calculation_l1707_170772


namespace NUMINAMATH_CALUDE_balloon_count_theorem_l1707_170771

theorem balloon_count_theorem (fred sam dan total : ℕ) 
  (h1 : fred = 10)
  (h2 : sam = 46)
  (h3 : dan = 16)
  (h4 : total = 72) :
  fred + sam + dan = total :=
sorry

end NUMINAMATH_CALUDE_balloon_count_theorem_l1707_170771


namespace NUMINAMATH_CALUDE_boy_position_in_line_l1707_170725

/-- The position of a boy in a line of boys, where he is equidistant from both ends -/
def midPosition (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- Theorem: In a line of 37 boys, the boy who is equidistant from both ends is in position 19 -/
theorem boy_position_in_line :
  midPosition 37 = 19 := by
  sorry

end NUMINAMATH_CALUDE_boy_position_in_line_l1707_170725


namespace NUMINAMATH_CALUDE_eva_apple_count_l1707_170726

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Eva needs to eat an apple -/
def weeks : ℕ := 2

/-- The number of apples Eva should buy -/
def apples_to_buy : ℕ := days_per_week * weeks

theorem eva_apple_count : apples_to_buy = 14 := by
  sorry

end NUMINAMATH_CALUDE_eva_apple_count_l1707_170726


namespace NUMINAMATH_CALUDE_polynomial_A_and_difference_l1707_170783

/-- Given polynomials A and B where B = 4x² - 3y - 1 and A + B = 6x² - y -/
def B (x y : ℝ) : ℝ := 4 * x^2 - 3 * y - 1

/-- Definition of A based on the given condition A + B = 6x² - y -/
def A (x y : ℝ) : ℝ := 6 * x^2 - y - B x y

theorem polynomial_A_and_difference (x y : ℝ) :
  A x y = 2 * x^2 + 2 * y + 1 ∧
  (|x - 1| + (y + 1)^2 = 0 → A x y - B x y = -5) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_A_and_difference_l1707_170783


namespace NUMINAMATH_CALUDE_jean_spot_ratio_l1707_170719

/-- Represents the number of spots on different parts of Jean's body -/
structure SpotCount where
  upperTorso : ℕ
  sides : ℕ

/-- The ratio of spots on the upper torso to total spots -/
def spotRatio (s : SpotCount) : ℚ :=
  s.upperTorso / (s.upperTorso + s.sides)

/-- Theorem stating that the ratio of spots on the upper torso to total spots is 3/4 -/
theorem jean_spot_ratio :
  ∀ (s : SpotCount), s.upperTorso = 30 → s.sides = 10 → spotRatio s = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_jean_spot_ratio_l1707_170719


namespace NUMINAMATH_CALUDE_smallest_m_is_correct_l1707_170732

/-- The smallest positive value of m for which 10x^2 - mx + 180 = 0 has integral solutions -/
def smallest_m : ℕ := 90

/-- A function representing the quadratic equation 10x^2 - mx + 180 = 0 -/
def quadratic (m : ℕ) (x : ℤ) : ℤ := 10 * x^2 - m * x + 180

theorem smallest_m_is_correct : 
  (∃ x y : ℤ, x ≠ y ∧ quadratic smallest_m x = 0 ∧ quadratic smallest_m y = 0) ∧ 
  (∀ m : ℕ, m < smallest_m → 
    ¬∃ x y : ℤ, x ≠ y ∧ quadratic m x = 0 ∧ quadratic m y = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_correct_l1707_170732


namespace NUMINAMATH_CALUDE_find_S_l1707_170736

theorem find_S : ∃ S : ℝ, 
  (∀ a b c d : ℝ, 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ d ≠ 1 ∧
    a + b + c + d = S ∧
    1/a + 1/b + 1/c + 1/d = S ∧
    1/(a-1) + 1/(b-1) + 1/(c-1) + 1/(d-1) = S) →
  S = -2 :=
by sorry

end NUMINAMATH_CALUDE_find_S_l1707_170736


namespace NUMINAMATH_CALUDE_brother_catchup_l1707_170741

/-- The time it takes for the older brother to catch up with the younger brother -/
def catchup_time (older_time younger_time delay : ℚ) : ℚ :=
  let relative_speed := 1 / older_time - 1 / younger_time
  let distance_covered := delay / younger_time
  delay + distance_covered / relative_speed

theorem brother_catchup :
  let older_time : ℚ := 12
  let younger_time : ℚ := 20
  let delay : ℚ := 5
  catchup_time older_time younger_time delay = 25/2 := by sorry

end NUMINAMATH_CALUDE_brother_catchup_l1707_170741


namespace NUMINAMATH_CALUDE_g_value_at_5_l1707_170784

def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 3

theorem g_value_at_5 (a b c : ℝ) (h : g a b c (-5) = -3) : g a b c 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_5_l1707_170784


namespace NUMINAMATH_CALUDE_tara_book_sales_l1707_170754

/-- The number of books Tara needs to sell to reach her goal -/
def books_to_sell (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) (accessory_cost : ℕ) : ℕ :=
  let goal := clarinet_cost - initial_savings
  let halfway := goal / 2
  let books_to_halfway := halfway / book_price
  let new_goal := goal + accessory_cost
  let books_after_loss := new_goal / book_price
  books_to_halfway + books_after_loss

/-- Theorem stating that Tara needs to sell 35 books in total -/
theorem tara_book_sales :
  books_to_sell 10 90 4 20 = 35 := by
  sorry

end NUMINAMATH_CALUDE_tara_book_sales_l1707_170754


namespace NUMINAMATH_CALUDE_inequality_proof_l1707_170790

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c = 1 → a^2 + b^2 + c^2 ≥ 1/3) ∧
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1707_170790


namespace NUMINAMATH_CALUDE_expression_simplification_l1707_170729

theorem expression_simplification (x y m : ℝ) 
  (h1 : (x - 5)^2 + |m - 1| = 0)
  (h2 : y + 1 = 5) :
  (2*x^2 - 3*x*y - 4*y^2) - m*(3*x^2 - x*y + 9*y^2) = -273 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1707_170729


namespace NUMINAMATH_CALUDE_difference_of_differences_l1707_170779

theorem difference_of_differences (a b c : ℤ) 
  (h1 : a - b = 2) 
  (h2 : b - c = -3) : 
  a - c = -1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_differences_l1707_170779


namespace NUMINAMATH_CALUDE_isbn_check_digit_l1707_170760

/-- Calculates the sum S for an ISBN --/
def calculate_S (A B C D E F G H I : ℕ) : ℕ :=
  10 * A + 9 * B + 8 * C + 7 * D + 6 * E + 5 * F + 4 * G + 3 * H + 2 * I

/-- Determines the check digit J based on the remainder r --/
def determine_J (r : ℕ) : ℕ :=
  if r = 0 then 0
  else if r = 1 then 10  -- Representing 'x' as 10
  else 11 - r

/-- Theorem: For the ISBN 962y707015, y = 7 --/
theorem isbn_check_digit (y : ℕ) (hy : y < 10) :
  let S := calculate_S 9 6 2 y 7 0 7 0 1
  let r := S % 11
  determine_J r = 5 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_isbn_check_digit_l1707_170760


namespace NUMINAMATH_CALUDE_car_journey_digit_squares_sum_l1707_170761

/-- Represents a car journey with specific odometer conditions -/
structure CarJourney where
  a : ℕ
  b : ℕ
  c : ℕ
  hours : ℕ
  initialReading : ℕ
  finalReading : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem car_journey_digit_squares_sum
  (journey : CarJourney)
  (h1 : journey.a ≥ 1)
  (h2 : journey.a + journey.b + journey.c = 9)
  (h3 : journey.initialReading = 100 * journey.a + 10 * journey.b + journey.c)
  (h4 : journey.finalReading = 100 * journey.c + 10 * journey.b + journey.a)
  (h5 : journey.finalReading - journey.initialReading = 65 * journey.hours) :
  journey.a^2 + journey.b^2 + journey.c^2 = 53 :=
sorry

end NUMINAMATH_CALUDE_car_journey_digit_squares_sum_l1707_170761


namespace NUMINAMATH_CALUDE_decreasing_cubic_condition_l1707_170734

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1

-- Define what it means for a function to be decreasing on ℝ
def IsDecreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → g x > g y

-- Theorem statement
theorem decreasing_cubic_condition (a : ℝ) :
  IsDecreasing (f a) → a < -3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_cubic_condition_l1707_170734


namespace NUMINAMATH_CALUDE_conditional_probability_animal_longevity_l1707_170704

def prob_birth_to_20 : ℝ := 0.8
def prob_birth_to_25 : ℝ := 0.4

theorem conditional_probability_animal_longevity :
  (prob_birth_to_25 / prob_birth_to_20) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_animal_longevity_l1707_170704


namespace NUMINAMATH_CALUDE_megan_carrots_second_day_l1707_170767

/-- Calculates the number of carrots Megan picked on the second day -/
def carrots_picked_second_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : ℕ :=
  total_carrots - (initial_carrots - thrown_out)

/-- Proves that Megan picked 46 carrots on the second day -/
theorem megan_carrots_second_day : 
  carrots_picked_second_day 19 4 61 = 46 := by
  sorry

end NUMINAMATH_CALUDE_megan_carrots_second_day_l1707_170767


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l1707_170722

theorem point_coordinate_sum (a b : ℝ) : 
  (2 = b - 1 ∧ -1 = a + 3) → a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l1707_170722


namespace NUMINAMATH_CALUDE_lunch_packing_ratio_l1707_170786

def school_days : ℕ := 180
def aliyah_lunch_days : ℕ := school_days / 2
def becky_lunch_days : ℕ := 45

theorem lunch_packing_ratio :
  becky_lunch_days / aliyah_lunch_days = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_lunch_packing_ratio_l1707_170786


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_squared_l1707_170756

theorem complex_magnitude_sum_squared : (Complex.abs (3 - 6*Complex.I) + Complex.abs (3 + 6*Complex.I))^2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_squared_l1707_170756


namespace NUMINAMATH_CALUDE_expression_simplification_l1707_170737

theorem expression_simplification (x y : ℝ) :
  let P := 2 * x + 3 * y
  let Q := 3 * x + 2 * y
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (24 * x^2 + 52 * x * y + 24 * y^2) / (5 * x * y - 5 * y^2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1707_170737


namespace NUMINAMATH_CALUDE_find_q_l1707_170782

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_find_q_l1707_170782


namespace NUMINAMATH_CALUDE_perimeter_decreases_to_convex_hull_outer_perimeter_not_smaller_l1707_170797

-- Define a polygon as a list of points in 2D space
def Polygon := List (ℝ × ℝ)

-- Define a function to calculate the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := sorry

-- Define a predicate to check if a polygon is convex
def is_convex (p : Polygon) : Prop := sorry

-- Define the convex hull of a polygon
def convex_hull (p : Polygon) : Polygon := sorry

-- Define a predicate to check if one polygon is completely inside another
def is_inside (a b : Polygon) : Prop := sorry

theorem perimeter_decreases_to_convex_hull (p : Polygon) : 
  perimeter (convex_hull p) < perimeter p := sorry

theorem outer_perimeter_not_smaller (a b : Polygon) 
  (h1 : is_convex a) (h2 : is_convex b) (h3 : is_inside a b) : 
  perimeter b ≥ perimeter a := sorry

end NUMINAMATH_CALUDE_perimeter_decreases_to_convex_hull_outer_perimeter_not_smaller_l1707_170797


namespace NUMINAMATH_CALUDE_fifty_ring_squares_l1707_170742

/-- Calculate the number of squares in the nth ring around a 2x1 rectangle --/
def ring_squares (n : ℕ) : ℕ :=
  let outer_width := 2 + 2 * n
  let outer_height := 1 + 2 * n
  let inner_width := 2 + 2 * (n - 1)
  let inner_height := 1 + 2 * (n - 1)
  outer_width * outer_height - inner_width * inner_height

/-- The 50th ring around a 2x1 rectangle contains 402 unit squares --/
theorem fifty_ring_squares : ring_squares 50 = 402 := by
  sorry

#eval ring_squares 50  -- This line is optional, for verification purposes

end NUMINAMATH_CALUDE_fifty_ring_squares_l1707_170742


namespace NUMINAMATH_CALUDE_proj_equals_v_l1707_170780

/-- Given two 2D vectors v and w, prove that the projection of v onto w is equal to v itself. -/
theorem proj_equals_v (v w : Fin 2 → ℝ) (hv : v = ![- 3, 2]) (hw : w = ![4, - 2]) :
  (v • w / (w • w)) • w = v := by sorry

end NUMINAMATH_CALUDE_proj_equals_v_l1707_170780


namespace NUMINAMATH_CALUDE_dora_receives_two_packs_l1707_170710

/-- The number of packs of stickers Dora receives --/
def dora_sticker_packs (allowance : ℕ) (card_price : ℕ) (sticker_price : ℕ) (num_people : ℕ) : ℕ :=
  let total_money := allowance * num_people
  let remaining_money := total_money - card_price
  let total_sticker_packs := remaining_money / sticker_price
  total_sticker_packs / num_people

/-- Theorem stating that Dora receives 2 packs of stickers --/
theorem dora_receives_two_packs :
  dora_sticker_packs 9 10 2 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dora_receives_two_packs_l1707_170710


namespace NUMINAMATH_CALUDE_living_space_increase_l1707_170748

/-- Proves that the average annual increase in living space needed is approximately 12.05 ten thousand m² --/
theorem living_space_increase (initial_population : ℝ) (initial_space_per_person : ℝ)
  (target_space_per_person : ℝ) (growth_rate : ℝ) (years : ℕ)
  (h1 : initial_population = 20) -- in ten thousands
  (h2 : initial_space_per_person = 8)
  (h3 : target_space_per_person = 10)
  (h4 : growth_rate = 0.01)
  (h5 : years = 4) :
  ∃ x : ℝ, abs (x - 12.05) < 0.01 ∧ 
  x * years = target_space_per_person * (initial_population * (1 + growth_rate) ^ years) - 
              initial_space_per_person * initial_population :=
by sorry


end NUMINAMATH_CALUDE_living_space_increase_l1707_170748


namespace NUMINAMATH_CALUDE_A_simplified_A_value_when_x_plus_one_squared_is_six_l1707_170727

-- Define the polynomial A
def A (x : ℝ) : ℝ := (x + 2)^2 + (1 - x) * (2 + x) - 3

-- Theorem for the simplified form of A
theorem A_simplified (x : ℝ) : A x = 3 * x + 3 := by sorry

-- Theorem for the value of A when (x+1)^2 = 6
theorem A_value_when_x_plus_one_squared_is_six :
  ∃ x : ℝ, (x + 1)^2 = 6 ∧ (A x = 3 * Real.sqrt 6 ∨ A x = -3 * Real.sqrt 6) := by sorry

end NUMINAMATH_CALUDE_A_simplified_A_value_when_x_plus_one_squared_is_six_l1707_170727


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1707_170770

/-- The quadratic equation ax^2 - x + 1 = 0 has real roots if and only if a ≤ 1/4 and a ≠ 0 -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - x + 1 = 0) ↔ (a ≤ 1/4 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1707_170770


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1707_170751

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (3 + 4 * Complex.I) = 43/25 + (1/25) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1707_170751


namespace NUMINAMATH_CALUDE_appropriate_speech_lengths_l1707_170774

/-- Represents the duration of a speech in minutes -/
def SpeechDuration := { d : ℝ // 20 ≤ d ∧ d ≤ 40 }

/-- The recommended speech rate in words per minute -/
def SpeechRate : ℝ := 120

/-- Calculates the number of words for a given duration -/
def wordCount (d : SpeechDuration) : ℝ := d.val * SpeechRate

/-- Checks if a word count is appropriate for the speech -/
def isAppropriateLength (w : ℝ) : Prop :=
  ∃ (d : SpeechDuration), wordCount d = w

theorem appropriate_speech_lengths :
  isAppropriateLength 2500 ∧ 
  isAppropriateLength 3800 ∧ 
  isAppropriateLength 4600 := by sorry

end NUMINAMATH_CALUDE_appropriate_speech_lengths_l1707_170774


namespace NUMINAMATH_CALUDE_B_coordinates_l1707_170759

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Moves a point up by a given number of units -/
def moveUp (p : Point) (units : ℤ) : Point :=
  { x := p.x, y := p.y + units }

/-- Moves a point left by a given number of units -/
def moveLeft (p : Point) (units : ℤ) : Point :=
  { x := p.x - units, y := p.y }

/-- The initial point A -/
def A : Point := { x := -3, y := -5 }

/-- The final point B after moving A -/
def B : Point := moveLeft (moveUp A 4) 3

/-- Theorem stating that B has the correct coordinates -/
theorem B_coordinates : B.x = -6 ∧ B.y = -1 := by sorry

end NUMINAMATH_CALUDE_B_coordinates_l1707_170759


namespace NUMINAMATH_CALUDE_cart_max_speed_l1707_170707

/-- The maximum speed of a cart on a circular track -/
theorem cart_max_speed (R a : ℝ) (hR : R > 0) (ha : a > 0) :
  ∃ v_max : ℝ,
    v_max = (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2))^(1/4) ∧
    ∀ v : ℝ,
      v ≤ v_max →
      (v^2 / (4 * Real.pi * R))^2 + (v^2 / R)^2 ≤ a^2 :=
by sorry

end NUMINAMATH_CALUDE_cart_max_speed_l1707_170707


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1707_170789

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let equation := -48 * x^2 + 108 * x - 27 = 0
  let sum_of_solutions := -108 / (-48)
  sum_of_solutions = 9/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1707_170789


namespace NUMINAMATH_CALUDE_pears_minus_apples_equals_two_l1707_170740

/-- Represents a bowl of fruit containing apples, pears, and bananas. -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- A bowl of fruit satisfying the given conditions. -/
def specialBowl : FruitBowl :=
  { apples := 0,  -- Placeholder value, will be constrained by theorem
    pears := 0,   -- Placeholder value, will be constrained by theorem
    bananas := 9 }

theorem pears_minus_apples_equals_two (bowl : FruitBowl) :
  bowl.apples + bowl.pears + bowl.bananas = 19 →
  bowl.bananas = 9 →
  bowl.bananas = bowl.pears + 3 →
  bowl.pears > bowl.apples →
  bowl.pears - bowl.apples = 2 := by
  sorry

#check pears_minus_apples_equals_two specialBowl

end NUMINAMATH_CALUDE_pears_minus_apples_equals_two_l1707_170740


namespace NUMINAMATH_CALUDE_perfect_cube_prime_factor_addition_l1707_170777

theorem perfect_cube_prime_factor_addition (x : ℕ) : ∃ x, 
  (27 = 3^3) ∧ 
  (∃ p : ℕ, Prime p ∧ p = 3 + x) ∧ 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_perfect_cube_prime_factor_addition_l1707_170777


namespace NUMINAMATH_CALUDE_fabric_requirement_l1707_170708

/-- The number of dresses to be made -/
def num_dresses : ℕ := 4

/-- The amount of fabric available in feet -/
def fabric_available : ℝ := 7

/-- The additional amount of fabric needed in feet -/
def fabric_needed : ℝ := 59

/-- The number of feet in a yard -/
def feet_per_yard : ℝ := 3

/-- The amount of fabric required for one dress in yards -/
def fabric_per_dress : ℝ := 5.5

theorem fabric_requirement :
  (fabric_available + fabric_needed) / feet_per_yard / num_dresses = fabric_per_dress := by
  sorry

end NUMINAMATH_CALUDE_fabric_requirement_l1707_170708


namespace NUMINAMATH_CALUDE_geric_initial_bills_l1707_170730

/-- The number of bills Jessa had initially -/
def jessa_initial : ℕ := 7 + 3

/-- The number of bills Kyla had -/
def kyla : ℕ := jessa_initial - 2

/-- The number of bills Geric had initially -/
def geric_initial : ℕ := 2 * kyla

theorem geric_initial_bills : geric_initial = 16 := by
  sorry

end NUMINAMATH_CALUDE_geric_initial_bills_l1707_170730


namespace NUMINAMATH_CALUDE_circle_radius_range_l1707_170714

/-- The set of points (x, y) satisfying the given equation -/
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sin (3 * p.1 + 4 * p.2) = Real.sin (3 * p.1) + Real.sin (4 * p.2)}

/-- A circle with center c and radius r -/
def Circle (c : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

/-- The theorem stating the range of possible radii for non-intersecting circles -/
theorem circle_radius_range (c : ℝ × ℝ) (r : ℝ) :
  (∀ p ∈ M, p ∉ Circle c r) → 0 < r ∧ r < Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_circle_radius_range_l1707_170714


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1707_170731

/-- Proves that the speed of a boat in still water is 12 km/hr given the conditions -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 4) 
  (h2 : downstream_distance = 4.8) 
  (h3 : downstream_time = 18 / 60) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 12 ∧ 
    downstream_distance = (still_water_speed + current_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1707_170731
