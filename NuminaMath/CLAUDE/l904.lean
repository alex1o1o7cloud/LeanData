import Mathlib

namespace NUMINAMATH_CALUDE_trig_identity_l904_90401

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin ((5*π)/6 - x) + (Real.cos ((π/3) - x))^2 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l904_90401


namespace NUMINAMATH_CALUDE_ink_bottle_arrangement_l904_90414

-- Define the type for a row of bottles
def Row := Fin 7 → Bool

-- Define the type for the arrangement of bottles
def Arrangement := Fin 130 → Row

-- Theorem statement
theorem ink_bottle_arrangement (arr : Arrangement) :
  (∃ i j k : Fin 130, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ arr i = arr j ∧ arr j = arr k) ∨
  (∃ i₁ j₁ i₂ j₂ : Fin 130, i₁ ≠ j₁ ∧ i₂ ≠ j₂ ∧ i₁ ≠ i₂ ∧ i₁ ≠ j₂ ∧ j₁ ≠ i₂ ∧ j₁ ≠ j₂ ∧
    arr i₁ = arr j₁ ∧ arr i₂ = arr j₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ink_bottle_arrangement_l904_90414


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l904_90426

def A : Set ℝ := {x | x^2 - 4 < 0}

def B : Set ℝ := {x | ∃ n : ℤ, x = 2*n + 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l904_90426


namespace NUMINAMATH_CALUDE_no_nonzero_solutions_l904_90494

theorem no_nonzero_solutions (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (Real.sqrt (a^2 + b^2) = 0 ↔ a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = (a + b) / 2 ↔ a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = Real.sqrt a + Real.sqrt b ↔ a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = a + b - 1 ↔ a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_nonzero_solutions_l904_90494


namespace NUMINAMATH_CALUDE_farm_ratio_change_l904_90483

theorem farm_ratio_change (H C : ℕ) : 
  H = 6 * C →  -- Initial ratio of horses to cows is 6:1
  H - 15 = (C + 15) + 70 →  -- After transaction, 70 more horses than cows
  (H - 15) / (C + 15) = 3  -- New ratio of horses to cows is 3:1
  := by sorry

end NUMINAMATH_CALUDE_farm_ratio_change_l904_90483


namespace NUMINAMATH_CALUDE_soccer_game_total_goals_l904_90476

theorem soccer_game_total_goals :
  let team_a_first_half : ℕ := 8
  let team_b_first_half : ℕ := team_a_first_half / 2
  let team_b_second_half : ℕ := team_a_first_half
  let team_a_second_half : ℕ := team_b_second_half - 2
  team_a_first_half + team_b_first_half + team_a_second_half + team_b_second_half = 26 :=
by sorry

end NUMINAMATH_CALUDE_soccer_game_total_goals_l904_90476


namespace NUMINAMATH_CALUDE_intersection_M_N_l904_90470

-- Define set M
def M : Set ℝ := {x | x^2 - x ≤ 0}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ico 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l904_90470


namespace NUMINAMATH_CALUDE_divisibility_implies_equation_existence_l904_90455

theorem divisibility_implies_equation_existence (p x y : ℕ) (hp : Prime p) 
  (hp_form : ∃ k : ℕ, p = 4 * k + 3) (hx : x > 0) (hy : y > 0)
  (hdiv : p ∣ (x^2 - x*y + ((p+1)/4) * y^2)) :
  ∃ u v : ℤ, x^2 - x*y + ((p+1)/4) * y^2 = p * (u^2 - u*v + ((p+1)/4) * v^2) := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equation_existence_l904_90455


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l904_90432

/-- The distance between the foci of the ellipse 9x^2 + 16y^2 = 144 is 2√7 -/
theorem ellipse_foci_distance :
  let a : ℝ := 4
  let b : ℝ := 3
  ∀ x y : ℝ, 9 * x^2 + 16 * y^2 = 144 →
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l904_90432


namespace NUMINAMATH_CALUDE_triangle_angle_C_l904_90491

/-- Given a triangle with angle A = 30°, side a = 1, and side b = √2,
    prove that the angle C is either 105° or 15°. -/
theorem triangle_angle_C (A : Real) (a b : Real) :
  A = 30 * π / 180 →
  a = 1 →
  b = Real.sqrt 2 →
  ∃ (C : Real), (C = 105 * π / 180 ∨ C = 15 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l904_90491


namespace NUMINAMATH_CALUDE_log_square_ratio_l904_90419

theorem log_square_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (h2 : x * y = 243) : 
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_log_square_ratio_l904_90419


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l904_90492

theorem complex_product_quadrant : 
  let z₁ : ℂ := 1 - 2*I
  let z₂ : ℂ := 2 + I
  let product := z₁ * z₂
  (product.re > 0 ∧ product.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l904_90492


namespace NUMINAMATH_CALUDE_shifted_line_equation_l904_90493

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Shifts a linear function horizontally and vertically -/
def shiftLinearFunction (f : LinearFunction) (horizontalShift : ℝ) (verticalShift : ℝ) : LinearFunction :=
  { slope := f.slope
    yIntercept := f.slope * (-horizontalShift) + f.yIntercept + verticalShift }

theorem shifted_line_equation (f : LinearFunction) :
  let f' := shiftLinearFunction f 2 3
  f.slope = 2 ∧ f.yIntercept = -3 → f'.slope = 2 ∧ f'.yIntercept = 4 := by
  sorry

#check shifted_line_equation

end NUMINAMATH_CALUDE_shifted_line_equation_l904_90493


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l904_90471

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ), ∀ (x : ℚ), x ≠ 4 ∧ x ≠ 2 →
    (6 * x + 2) / ((x - 4) * (x - 2)^3) = 
    P / (x - 4) + Q / (x - 2) + R / (x - 2)^3 ∧
    P = 13 / 4 ∧ Q = -13 / 2 ∧ R = -7 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l904_90471


namespace NUMINAMATH_CALUDE_problem_statement_l904_90495

theorem problem_statement (θ : ℝ) (h : (Real.sin θ)^2 + 4 = 2 * (Real.cos θ + 1)) :
  (Real.cos θ + 1) * (Real.sin θ + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l904_90495


namespace NUMINAMATH_CALUDE_surface_area_bound_l904_90462

/-- A convex broken line -/
structure ConvexBrokenLine where
  points : List (ℝ × ℝ)
  is_convex : Bool
  length : ℝ

/-- The surface area of revolution of a convex broken line -/
def surface_area_of_revolution (line : ConvexBrokenLine) : ℝ := sorry

/-- Theorem: The surface area of revolution of a convex broken line
    is less than or equal to π * d² / 2, where d is the length of the line -/
theorem surface_area_bound (line : ConvexBrokenLine) :
  surface_area_of_revolution line ≤ Real.pi * line.length^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_bound_l904_90462


namespace NUMINAMATH_CALUDE_area_of_triangle_l904_90459

/-- The hyperbola with equation x^2 - y^2/12 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2/12 = 1}

/-- The foci of the hyperbola -/
def Foci : ℝ × ℝ × ℝ × ℝ := sorry

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- The distance ratio condition -/
axiom distance_ratio : 
  let (f1x, f1y, f2x, f2y) := Foci
  let (px, py) := P
  3 * ((f2x - px)^2 + (f2y - py)^2) = 2 * ((f1x - px)^2 + (f1y - py)^2)

/-- P is on the hyperbola -/
axiom P_on_hyperbola : P ∈ Hyperbola

/-- The theorem to be proved -/
theorem area_of_triangle : 
  let (f1x, f1y, f2x, f2y) := Foci
  let (px, py) := P
  (1/2) * |f1x - f2x| * |f1y - f2y| = 12 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_l904_90459


namespace NUMINAMATH_CALUDE_cat_count_correct_l904_90452

/-- The number of cats that can meow -/
def meow : ℕ := 70

/-- The number of cats that can jump -/
def jump : ℕ := 40

/-- The number of cats that can fetch -/
def fetch : ℕ := 30

/-- The number of cats that can roll -/
def roll : ℕ := 50

/-- The number of cats that can meow and jump -/
def meow_jump : ℕ := 25

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 15

/-- The number of cats that can fetch and roll -/
def fetch_roll : ℕ := 20

/-- The number of cats that can meow and roll -/
def meow_roll : ℕ := 28

/-- The number of cats that can meow, jump, and fetch -/
def meow_jump_fetch : ℕ := 5

/-- The number of cats that can jump, fetch, and roll -/
def jump_fetch_roll : ℕ := 10

/-- The number of cats that can fetch, roll, and meow -/
def fetch_roll_meow : ℕ := 12

/-- The number of cats that can do all four tricks -/
def all_four : ℕ := 8

/-- The number of cats that can do no tricks -/
def no_tricks : ℕ := 12

/-- The total number of cats in the studio -/
def total_cats : ℕ := 129

theorem cat_count_correct : 
  total_cats = meow + jump + fetch + roll - meow_jump - jump_fetch - fetch_roll - meow_roll + 
               meow_jump_fetch + jump_fetch_roll + fetch_roll_meow - 2 * all_four + no_tricks := by
  sorry

end NUMINAMATH_CALUDE_cat_count_correct_l904_90452


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l904_90441

theorem sum_of_numbers_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 4 * a →
  a^2 + b^2 + c^2 = 4725 →
  a + b + c = 105 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l904_90441


namespace NUMINAMATH_CALUDE_trigonometric_identity_l904_90467

theorem trigonometric_identity (c d : ℝ) (θ : ℝ) 
  (h : (Real.sin θ)^2 / c + (Real.cos θ)^2 / d = 1 / (c + d)) 
  (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : d ≠ 1) :
  (Real.sin θ)^4 / c^2 + (Real.cos θ)^4 / d^2 = 2 * (c - d)^2 / (c^2 * d^2 * (d - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l904_90467


namespace NUMINAMATH_CALUDE_factorization_of_16x_squared_minus_4_l904_90434

theorem factorization_of_16x_squared_minus_4 (x : ℝ) :
  16 * x^2 - 4 = 4 * (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_16x_squared_minus_4_l904_90434


namespace NUMINAMATH_CALUDE_bus_trip_speed_l904_90458

/-- Proves that for a trip of 880 miles, if increasing the speed by 10 mph
    reduces the trip time by 2 hours, then the original speed was 61.5 mph. -/
theorem bus_trip_speed (v : ℝ) (h : v > 0) : 
  (880 / v) - (880 / (v + 10)) = 2 → v = 61.5 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l904_90458


namespace NUMINAMATH_CALUDE_hiking_distance_sum_l904_90429

theorem hiking_distance_sum : 
  let leg1 : ℝ := 3.8
  let leg2 : ℝ := 1.75
  let leg3 : ℝ := 2.3
  let leg4 : ℝ := 0.45
  let leg5 : ℝ := 1.92
  leg1 + leg2 + leg3 + leg4 + leg5 = 10.22 := by
  sorry

end NUMINAMATH_CALUDE_hiking_distance_sum_l904_90429


namespace NUMINAMATH_CALUDE_three_friends_came_later_l904_90464

/-- The number of friends who came over later -/
def friends_came_later (initial_friends final_total : ℕ) : ℕ :=
  final_total - initial_friends

/-- Theorem: Given 4 initial friends and a final total of 7 people,
    prove that 3 friends came over later -/
theorem three_friends_came_later :
  friends_came_later 4 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_friends_came_later_l904_90464


namespace NUMINAMATH_CALUDE_distance_between_foci_l904_90450

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 5)^2) = 26

-- Define the foci
def focus1 : ℝ × ℝ := (4, 3)
def focus2 : ℝ × ℝ := (-6, 5)

-- Theorem statement
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 2 * Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l904_90450


namespace NUMINAMATH_CALUDE_max_students_is_eight_l904_90472

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

end NUMINAMATH_CALUDE_max_students_is_eight_l904_90472


namespace NUMINAMATH_CALUDE_union_equals_N_implies_a_in_range_l904_90477

/-- Given sets M and N, if their union equals N, then a is in the interval [-2, 2] -/
theorem union_equals_N_implies_a_in_range (a : ℝ) :
  let M := {x : ℝ | x * (x - a - 1) < 0}
  let N := {x : ℝ | x^2 - 2*x - 3 ≤ 0}
  (M ∪ N = N) → a ∈ Set.Icc (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_N_implies_a_in_range_l904_90477


namespace NUMINAMATH_CALUDE_A_intersect_complement_B_eq_a_l904_90451

-- Define the universal set U
def U : Set Char := {'{', 'a', 'b', 'c', 'd', 'e', '}'}

-- Define set A
def A : Set Char := {'{', 'a', 'b', '}'}

-- Define set B
def B : Set Char := {'{', 'b', 'c', 'd', '}'}

-- Theorem to prove
theorem A_intersect_complement_B_eq_a : A ∩ (U \ B) = {'{', 'a', '}'} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_complement_B_eq_a_l904_90451


namespace NUMINAMATH_CALUDE_no_cracked_seashells_l904_90453

theorem no_cracked_seashells (tom_shells fred_shells total_shells : ℕ) 
  (h1 : tom_shells = 15)
  (h2 : fred_shells = 43)
  (h3 : total_shells = 58)
  (h4 : tom_shells + fred_shells = total_shells) :
  total_shells - (tom_shells + fred_shells) = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_cracked_seashells_l904_90453


namespace NUMINAMATH_CALUDE_paper_flowers_per_hour_l904_90447

/-- The number of paper flowers Person B makes per hour -/
def flowers_per_hour_B : ℕ := 80

/-- The number of paper flowers Person A makes per hour -/
def flowers_per_hour_A : ℕ := flowers_per_hour_B - 20

/-- The time it takes Person A to make 120 flowers -/
def time_A : ℚ := 120 / flowers_per_hour_A

/-- The time it takes Person B to make 160 flowers -/
def time_B : ℚ := 160 / flowers_per_hour_B

theorem paper_flowers_per_hour :
  (flowers_per_hour_A = flowers_per_hour_B - 20) ∧
  (time_A = time_B) →
  flowers_per_hour_B = 80 := by
  sorry

end NUMINAMATH_CALUDE_paper_flowers_per_hour_l904_90447


namespace NUMINAMATH_CALUDE_product_invariance_l904_90481

theorem product_invariance (a b : ℝ) (h : a * b = 300) :
  (6 * a) * (b / 6) = 300 := by
  sorry

end NUMINAMATH_CALUDE_product_invariance_l904_90481


namespace NUMINAMATH_CALUDE_equal_interval_line_segments_l904_90460

/-- Given two line segments with equal interval spacing between points,
    where one segment has 10 points over length a and the other has 100 points over length b,
    prove that b = 11a. -/
theorem equal_interval_line_segments (a b : ℝ) : 
  (∃ (interval : ℝ), 
    a = 9 * interval ∧ 
    b = 99 * interval) → 
  b = 11 * a := by sorry

end NUMINAMATH_CALUDE_equal_interval_line_segments_l904_90460


namespace NUMINAMATH_CALUDE_actual_distance_l904_90486

/-- Calculates the actual distance between two cities given the map distance and scale. -/
theorem actual_distance (map_distance : ℝ) (scale_distance : ℝ) (scale_miles : ℝ) : 
  map_distance * (scale_miles / scale_distance) = 240 :=
  by
  -- Assuming map_distance = 20, scale_distance = 0.5, and scale_miles = 6
  have h1 : map_distance = 20 := by sorry
  have h2 : scale_distance = 0.5 := by sorry
  have h3 : scale_miles = 6 := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_actual_distance_l904_90486


namespace NUMINAMATH_CALUDE_triangle_area_l904_90442

theorem triangle_area (a b c : ℝ) (A : ℝ) :
  b^2 - b*c - 2*c^2 = 0 →
  a = Real.sqrt 6 →
  Real.cos A = 7/8 →
  (1/2) * b * c * Real.sin A = Real.sqrt 15 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l904_90442


namespace NUMINAMATH_CALUDE_min_area_special_square_l904_90444

/-- A square with one side on y = 2x - 17 and two vertices on y = x^2 -/
structure SpecialSquare where
  /-- Side length of the square -/
  a : ℝ
  /-- Parameter for the line y = 2x + b passing through two vertices on the parabola -/
  b : ℝ
  /-- The square has one side on y = 2x - 17 -/
  side_on_line : a = (17 + b) / Real.sqrt 5
  /-- Two vertices of the square are on y = x^2 -/
  vertices_on_parabola : a^2 = 20 * (1 + b)

/-- The minimum area of a SpecialSquare is 80 -/
theorem min_area_special_square :
  ∀ s : SpecialSquare, s.a^2 ≥ 80 := by
  sorry

#check min_area_special_square

end NUMINAMATH_CALUDE_min_area_special_square_l904_90444


namespace NUMINAMATH_CALUDE_no_prime_solution_l904_90402

-- Define a function to convert a number from base p to base 10
def to_base_10 (digits : List Nat) (p : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * p ^ i) 0

-- Define the left-hand side of the equation
def lhs (p : Nat) : Nat :=
  to_base_10 [6, 0, 0, 2] p +
  to_base_10 [4, 0, 4] p +
  to_base_10 [5, 1, 2] p +
  to_base_10 [2, 2, 2] p +
  to_base_10 [9] p

-- Define the right-hand side of the equation
def rhs (p : Nat) : Nat :=
  to_base_10 [3, 3, 4] p +
  to_base_10 [2, 7, 5] p +
  to_base_10 [1, 2, 3] p

-- State the theorem
theorem no_prime_solution :
  ¬ ∃ p : Nat, Nat.Prime p ∧ lhs p = rhs p :=
sorry

end NUMINAMATH_CALUDE_no_prime_solution_l904_90402


namespace NUMINAMATH_CALUDE_inequality_proof_l904_90437

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^4 + b^4 + c^4 = 3) :
  1 / (4 - a*b) + 1 / (4 - b*c) + 1 / (4 - c*a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l904_90437


namespace NUMINAMATH_CALUDE_two_part_journey_average_speed_l904_90439

/-- Calculates the average speed for a two-part journey -/
theorem two_part_journey_average_speed
  (total_distance : ℝ)
  (first_part_distance : ℝ)
  (first_part_speed : ℝ)
  (second_part_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : first_part_distance = 12)
  (h3 : first_part_speed = 24)
  (h4 : second_part_speed = 48)
  : (total_distance / (first_part_distance / first_part_speed +
     (total_distance - first_part_distance) / second_part_speed)) = 40 := by
  sorry

#check two_part_journey_average_speed

end NUMINAMATH_CALUDE_two_part_journey_average_speed_l904_90439


namespace NUMINAMATH_CALUDE_annual_grass_cutting_cost_l904_90420

/-- The annual cost of grass cutting given specific conditions -/
theorem annual_grass_cutting_cost
  (initial_height : ℝ)
  (growth_rate : ℝ)
  (cutting_threshold : ℝ)
  (cost_per_cut : ℝ)
  (h1 : initial_height = 2)
  (h2 : growth_rate = 0.5)
  (h3 : cutting_threshold = 4)
  (h4 : cost_per_cut = 100)
  : ℝ :=
by
  -- Prove that the annual cost of grass cutting is $300
  sorry

#check annual_grass_cutting_cost

end NUMINAMATH_CALUDE_annual_grass_cutting_cost_l904_90420


namespace NUMINAMATH_CALUDE_sum_of_roots_and_constant_l904_90473

theorem sum_of_roots_and_constant (a b c : ℝ) : 
  (1^2 + a*1 + 2 = 0) → 
  (a^2 + 5*a + c = 0) → 
  (b^2 + 5*b + c = 0) → 
  a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_and_constant_l904_90473


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l904_90418

/-- Given two lines that intersect at a point, prove the sum of their y-intercepts. -/
theorem intersection_y_intercept_sum (a b : ℝ) : 
  (∃ x y : ℝ, x = (1/3)*y + a ∧ y = (1/3)*x + b ∧ x = 3 ∧ y = -1) →
  a + b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l904_90418


namespace NUMINAMATH_CALUDE_min_shipping_cost_l904_90445

/-- Represents the shipping problem with given stock, demand, and costs. -/
structure ShippingProblem where
  shanghai_stock : ℕ
  nanjing_stock : ℕ
  suzhou_demand : ℕ
  changsha_demand : ℕ
  cost_shanghai_suzhou : ℕ
  cost_shanghai_changsha : ℕ
  cost_nanjing_suzhou : ℕ
  cost_nanjing_changsha : ℕ

/-- Calculates the total shipping cost given the number of units shipped from Shanghai to Suzhou. -/
def total_cost (problem : ShippingProblem) (x : ℕ) : ℕ :=
  problem.cost_shanghai_suzhou * x +
  problem.cost_shanghai_changsha * (problem.shanghai_stock - x) +
  problem.cost_nanjing_suzhou * (problem.suzhou_demand - x) +
  problem.cost_nanjing_changsha * (x - (problem.suzhou_demand - problem.nanjing_stock))

/-- Theorem stating that the minimum shipping cost is 8600 yuan for the given problem. -/
theorem min_shipping_cost (problem : ShippingProblem) 
  (h1 : problem.shanghai_stock = 12)
  (h2 : problem.nanjing_stock = 6)
  (h3 : problem.suzhou_demand = 10)
  (h4 : problem.changsha_demand = 8)
  (h5 : problem.cost_shanghai_suzhou = 400)
  (h6 : problem.cost_shanghai_changsha = 800)
  (h7 : problem.cost_nanjing_suzhou = 300)
  (h8 : problem.cost_nanjing_changsha = 500) :
  ∃ x : ℕ, x ≥ 4 ∧ x ≤ 10 ∧ total_cost problem x = 8600 ∧ 
  ∀ y : ℕ, y ≥ 4 → y ≤ 10 → total_cost problem y ≥ total_cost problem x :=
sorry

end NUMINAMATH_CALUDE_min_shipping_cost_l904_90445


namespace NUMINAMATH_CALUDE_intersection_condition_l904_90425

/-- The set A defined by the equation y = x^2 + mx + 2 -/
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2 + m * p.1 + 2}

/-- The set B defined by the equation y = x + 1 -/
def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 1}

/-- The theorem stating the condition for non-empty intersection of A and B -/
theorem intersection_condition (m : ℝ) :
  (A m ∩ B).Nonempty ↔ m ≤ -1 ∨ m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l904_90425


namespace NUMINAMATH_CALUDE_johns_hair_growth_l904_90440

/-- Represents John's hair growth and haircut information -/
structure HairGrowthInfo where
  cutFrom : ℝ  -- Length of hair before cut
  cutTo : ℝ    -- Length of hair after cut
  baseCost : ℝ -- Base cost of a haircut
  tipPercent : ℝ -- Tip percentage
  yearlySpend : ℝ -- Total spent on haircuts per year

/-- Calculates the monthly hair growth rate -/
def monthlyGrowthRate (info : HairGrowthInfo) : ℝ :=
  -- Definition of the function
  sorry

/-- Theorem stating that John's hair grows 1.5 inches per month -/
theorem johns_hair_growth (info : HairGrowthInfo) 
  (h1 : info.cutFrom = 9)
  (h2 : info.cutTo = 6)
  (h3 : info.baseCost = 45)
  (h4 : info.tipPercent = 0.2)
  (h5 : info.yearlySpend = 324) :
  monthlyGrowthRate info = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_johns_hair_growth_l904_90440


namespace NUMINAMATH_CALUDE_train_speed_proof_l904_90490

/-- Proves that a train with given parameters has a specific speed -/
theorem train_speed_proof (train_length bridge_length time_to_cross : Real)
  (h1 : train_length = 110)
  (h2 : bridge_length = 136)
  (h3 : time_to_cross = 12.299016078713702) :
  (train_length + bridge_length) / time_to_cross * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_proof_l904_90490


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l904_90400

/-- Two perpendicular lines with a given perpendicular foot -/
structure PerpendicularLines where
  a : ℝ
  b : ℝ
  c : ℝ
  line1 : ∀ x y : ℝ, a * x + 4 * y - 2 = 0
  line2 : ∀ x y : ℝ, 2 * x - 5 * y + b = 0
  perpendicular : (a / 4) * (2 / 5) = -1
  foot_on_line1 : a * 1 + 4 * c - 2 = 0
  foot_on_line2 : 2 * 1 - 5 * c + b = 0

/-- The sum of a, b, and c for perpendicular lines with given conditions is -4 -/
theorem perpendicular_lines_sum (l : PerpendicularLines) : l.a + l.b + l.c = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l904_90400


namespace NUMINAMATH_CALUDE_regular_tetrahedron_properties_l904_90479

-- Define a regular tetrahedron
structure RegularTetrahedron where
  -- Add any necessary fields here
  
-- Define the properties of a regular tetrahedron
def has_equal_edges_and_vertex_angles (t : RegularTetrahedron) : Prop :=
  sorry

def has_congruent_faces_and_equal_dihedral_angles (t : RegularTetrahedron) : Prop :=
  sorry

def has_congruent_faces_and_equal_vertex_angles (t : RegularTetrahedron) : Prop :=
  sorry

-- Theorem stating that a regular tetrahedron satisfies all three properties
theorem regular_tetrahedron_properties (t : RegularTetrahedron) :
  has_equal_edges_and_vertex_angles t ∧
  has_congruent_faces_and_equal_dihedral_angles t ∧
  has_congruent_faces_and_equal_vertex_angles t :=
sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_properties_l904_90479


namespace NUMINAMATH_CALUDE_subtraction_minimizes_l904_90487

-- Define the set of operators
inductive Operator : Type
  | add : Operator
  | sub : Operator
  | mul : Operator
  | div : Operator

-- Function to apply the operator
def apply_operator (op : Operator) (a b : ℤ) : ℤ :=
  match op with
  | Operator.add => a + b
  | Operator.sub => a - b
  | Operator.mul => a * b
  | Operator.div => a / b

-- Theorem statement
theorem subtraction_minimizes :
  ∀ op : Operator, apply_operator Operator.sub (-3) 1 ≤ apply_operator op (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_minimizes_l904_90487


namespace NUMINAMATH_CALUDE_accessories_total_cost_l904_90489

def mouse_cost : ℕ := 20

def keyboard_cost : ℕ := 2 * mouse_cost

def headphones_cost : ℕ := mouse_cost + 15

def usb_hub_cost : ℕ := 36 - mouse_cost

def total_cost : ℕ := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost

theorem accessories_total_cost : total_cost = 111 := by
  sorry

end NUMINAMATH_CALUDE_accessories_total_cost_l904_90489


namespace NUMINAMATH_CALUDE_equation_solution_exists_l904_90498

theorem equation_solution_exists (m : ℕ+) :
  ∃ n : ℕ+, (n : ℚ) / m = ⌊(n^2 : ℚ)^(1/3)⌋ + ⌊(n : ℚ)^(1/2)⌋ + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l904_90498


namespace NUMINAMATH_CALUDE_at_least_one_hits_target_l904_90449

theorem at_least_one_hits_target (p_both : ℝ) (h : p_both = 0.6) :
  1 - (1 - p_both) * (1 - p_both) = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_hits_target_l904_90449


namespace NUMINAMATH_CALUDE_money_ratio_proof_l904_90416

theorem money_ratio_proof (alison brittany brooke kent : ℕ) : 
  alison = brittany / 2 →
  brittany = 4 * brooke →
  kent = 1000 →
  alison = 4000 →
  brooke / kent = 2 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_proof_l904_90416


namespace NUMINAMATH_CALUDE_jacket_price_restoration_l904_90454

theorem jacket_price_restoration :
  ∀ (original_price : ℝ),
  original_price > 0 →
  let price_after_first_reduction := original_price * (1 - 0.2)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.25)
  let required_increase := (original_price / price_after_second_reduction) - 1
  abs (required_increase - 0.6667) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_restoration_l904_90454


namespace NUMINAMATH_CALUDE_chlorine_treatment_capacity_l904_90435

/-- Proves that given a rectangular pool with specified dimensions and chlorine costs,
    one quart of chlorine treats 120 cubic feet of water. -/
theorem chlorine_treatment_capacity
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (chlorine_cost : ℝ) (total_spent : ℝ)
  (h1 : length = 10)
  (h2 : width = 8)
  (h3 : depth = 6)
  (h4 : chlorine_cost = 3)
  (h5 : total_spent = 12) :
  (length * width * depth) / (total_spent / chlorine_cost) = 120 := by
  sorry


end NUMINAMATH_CALUDE_chlorine_treatment_capacity_l904_90435


namespace NUMINAMATH_CALUDE_sets_are_equal_l904_90446

-- Define the sets A and B
def A : Set ℝ := {1, Real.sqrt 3, Real.pi}
def B : Set ℝ := {Real.pi, 1, |-(Real.sqrt 3)|}

-- State the theorem
theorem sets_are_equal : A = B := by sorry

end NUMINAMATH_CALUDE_sets_are_equal_l904_90446


namespace NUMINAMATH_CALUDE_terrell_total_hike_distance_l904_90478

/-- Represents a hike with distance, duration, and calorie expenditure -/
structure Hike where
  distance : ℝ
  duration : ℝ
  calories : ℝ

/-- Calculates the total distance of two hikes -/
def total_distance (h1 h2 : Hike) : ℝ :=
  h1.distance + h2.distance

theorem terrell_total_hike_distance :
  let saturday_hike : Hike := { distance := 8.2, duration := 5, calories := 4000 }
  let sunday_hike : Hike := { distance := 1.6, duration := 2, calories := 1500 }
  total_distance saturday_hike sunday_hike = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_terrell_total_hike_distance_l904_90478


namespace NUMINAMATH_CALUDE_exponent_multiplication_l904_90499

theorem exponent_multiplication (a : ℝ) : a^2 * a^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l904_90499


namespace NUMINAMATH_CALUDE_sqrt_three_diamond_sqrt_three_l904_90443

-- Define the binary operation ¤
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- Theorem statement
theorem sqrt_three_diamond_sqrt_three : diamond (Real.sqrt 3) (Real.sqrt 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_diamond_sqrt_three_l904_90443


namespace NUMINAMATH_CALUDE_symmetric_line_theorem_l904_90422

/-- Given a line with equation ax + by + c = 0, 
    returns the equation of the line symmetric to it with respect to the x-axis -/
def symmetricLineEquation (a b c : ℝ) : ℝ × ℝ × ℝ := (a, -b, c)

/-- Proves that the equation of the line symmetric to 2x-y+4=0 
    with respect to the x-axis is 2x+y+4=0 -/
theorem symmetric_line_theorem :
  let original := (2, -1, 4)
  let symmetric := symmetricLineEquation 2 (-1) 4
  symmetric = (2, 1, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_line_theorem_l904_90422


namespace NUMINAMATH_CALUDE_min_score_is_45_l904_90430

/-- Represents the test scores and conditions -/
structure TestScores where
  num_tests : ℕ
  max_score : ℕ
  first_three : Fin 3 → ℕ
  target_average : ℕ

/-- Calculates the minimum score needed on one of the last two tests -/
def min_score (ts : TestScores) : ℕ :=
  let total_needed := ts.target_average * ts.num_tests
  let first_three_sum := (ts.first_three 0) + (ts.first_three 1) + (ts.first_three 2)
  let remaining_sum := total_needed - first_three_sum
  remaining_sum - ts.max_score

/-- Theorem stating the minimum score needed is 45 -/
theorem min_score_is_45 (ts : TestScores) 
  (h1 : ts.num_tests = 5)
  (h2 : ts.max_score = 120)
  (h3 : ts.first_three 0 = 86 ∧ ts.first_three 1 = 102 ∧ ts.first_three 2 = 97)
  (h4 : ts.target_average = 90) :
  min_score ts = 45 := by
  sorry

#eval min_score { num_tests := 5, max_score := 120, first_three := ![86, 102, 97], target_average := 90 }

end NUMINAMATH_CALUDE_min_score_is_45_l904_90430


namespace NUMINAMATH_CALUDE_inequality_relationship_l904_90406

theorem inequality_relationship (x : ℝ) : 
  (∀ x, x - 2 > 0 → (x - 2) * (x - 1) > 0) ∧ 
  (∃ x, (x - 2) * (x - 1) > 0 ∧ ¬(x - 2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relationship_l904_90406


namespace NUMINAMATH_CALUDE_candy_difference_example_l904_90403

/-- Given a total number of candies and the number of strawberry candies,
    calculate the difference between grape and strawberry candies. -/
def candy_difference (total : ℕ) (strawberry : ℕ) : ℕ :=
  (total - strawberry) - strawberry

/-- Theorem stating that given 821 total candies and 267 strawberry candies,
    the difference between grape and strawberry candies is 287. -/
theorem candy_difference_example : candy_difference 821 267 = 287 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_example_l904_90403


namespace NUMINAMATH_CALUDE_binomial_coefficient_x4_in_x_plus_1_to_10_l904_90410

theorem binomial_coefficient_x4_in_x_plus_1_to_10 :
  (Finset.range 11).sum (fun k => (Nat.choose 10 k) * (1^(10 - k)) * (1^k)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x4_in_x_plus_1_to_10_l904_90410


namespace NUMINAMATH_CALUDE_x_cubed_plus_x_cubed_l904_90466

theorem x_cubed_plus_x_cubed (x : ℝ) (h : x > 0) : 
  (x^3 + x^3 = 2*x^3) ∧ 
  (x^3 + x^3 ≠ x^6) ∧ 
  (x^3 + x^3 ≠ (3*x)^3) ∧ 
  (x^3 + x^3 ≠ (x^3)^2) :=
by sorry

end NUMINAMATH_CALUDE_x_cubed_plus_x_cubed_l904_90466


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l904_90482

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 ≥ 1) ↔ ∃ x₀ : ℝ, x₀^2 < 1 := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l904_90482


namespace NUMINAMATH_CALUDE_prob_two_gray_rabbits_l904_90431

/-- The probability of selecting 2 gray rabbits out of a group of 5 rabbits, 
    where 3 are gray and 2 are white, given that each rabbit has an equal 
    chance of being selected. -/
theorem prob_two_gray_rabbits (total : Nat) (gray : Nat) (white : Nat) 
    (h1 : total = gray + white) 
    (h2 : total = 5) 
    (h3 : gray = 3) 
    (h4 : white = 2) : 
  (Nat.choose gray 2 : ℚ) / (Nat.choose total 2) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_gray_rabbits_l904_90431


namespace NUMINAMATH_CALUDE_larger_number_l904_90469

theorem larger_number (P Q : ℝ) (h1 : P = Real.sqrt 2) (h2 : Q = Real.sqrt 6 - Real.sqrt 2) : P > Q := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l904_90469


namespace NUMINAMATH_CALUDE_complex_modulus_implies_real_value_l904_90423

theorem complex_modulus_implies_real_value (a : ℝ) : 
  Complex.abs ((a + 2 * Complex.I) * (1 + Complex.I)) = 4 → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_implies_real_value_l904_90423


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l904_90448

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem unique_solution_factorial_equation : 
  ∃! n : ℕ, n * factorial n - factorial n = 5040 - factorial n :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l904_90448


namespace NUMINAMATH_CALUDE_pizza_delivery_gas_theorem_l904_90480

/-- The amount of gas remaining after a pizza delivery route. -/
def gas_remaining (start : Float) (used : Float) : Float :=
  start - used

/-- Theorem stating that given the starting amount and used amount of gas,
    the remaining amount is correctly calculated. -/
theorem pizza_delivery_gas_theorem :
  gas_remaining 0.5 0.3333333333333333 = 0.1666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_pizza_delivery_gas_theorem_l904_90480


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l904_90463

theorem other_root_of_quadratic (c : ℝ) : 
  (∃ x : ℝ, 6 * x^2 + c * x = -3) → 
  (-1/2 : ℝ) ∈ {x : ℝ | 6 * x^2 + c * x = -3} →
  (-1 : ℝ) ∈ {x : ℝ | 6 * x^2 + c * x = -3} := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l904_90463


namespace NUMINAMATH_CALUDE_muffin_profit_l904_90411

/-- Bob's muffin business profit calculation -/
theorem muffin_profit : 
  ∀ (muffins_per_day : ℕ) 
    (cost_price selling_price : ℚ) 
    (days_in_week : ℕ),
  muffins_per_day = 12 →
  cost_price = 3/4 →
  selling_price = 3/2 →
  days_in_week = 7 →
  (selling_price - cost_price) * muffins_per_day * days_in_week = 63 := by
  sorry


end NUMINAMATH_CALUDE_muffin_profit_l904_90411


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l904_90497

theorem cubic_equation_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b*x + 25 = 0 := by
  sorry

#check cubic_equation_real_root

end NUMINAMATH_CALUDE_cubic_equation_real_root_l904_90497


namespace NUMINAMATH_CALUDE_absolute_value_plus_pi_minus_two_to_zero_l904_90474

theorem absolute_value_plus_pi_minus_two_to_zero :
  |(-3 : ℝ)| + (π - 2)^(0 : ℝ) = 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_plus_pi_minus_two_to_zero_l904_90474


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l904_90417

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a + 1 = 0) → 
  (b^3 - 2*b + 1 = 0) → 
  (c^3 - 2*c + 1 = 0) → 
  (a ≠ b) → (b ≠ c) → (c ≠ a) →
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 10 / 3) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l904_90417


namespace NUMINAMATH_CALUDE_bin_game_expected_value_l904_90488

theorem bin_game_expected_value (k : ℕ) : 
  let total_balls : ℕ := 10 + k
  let prob_green : ℚ := 10 / total_balls
  let prob_purple : ℚ := k / total_balls
  let expected_value : ℚ := 3 * prob_green - 1 * prob_purple
  (expected_value = 3/4) → (k = 13) :=
by sorry

end NUMINAMATH_CALUDE_bin_game_expected_value_l904_90488


namespace NUMINAMATH_CALUDE_largest_expression_l904_90421

def y : ℝ := 0.0002

theorem largest_expression (a b c d e : ℝ) 
  (ha : a = 5 + y)
  (hb : b = 5 - y)
  (hc : c = 5 * y)
  (hd : d = 5 / y)
  (he : e = y / 5) :
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l904_90421


namespace NUMINAMATH_CALUDE_regular_ngon_minimal_l904_90438

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an n-gon
structure NGon where
  n : ℕ
  vertices : List (ℝ × ℝ)

-- Function to check if an n-gon is inscribed in a circle
def isInscribed (ngon : NGon) (circle : Circle) : Prop :=
  ngon.vertices.length = ngon.n ∧
  ∀ v ∈ ngon.vertices, (v.1 - circle.center.1)^2 + (v.2 - circle.center.2)^2 = circle.radius^2

-- Function to check if an n-gon is regular
def isRegular (ngon : NGon) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ v ∈ ngon.vertices, (v.1 - center.1)^2 + (v.2 - center.2)^2 = radius^2

-- Function to calculate the area of an n-gon
noncomputable def area (ngon : NGon) : ℝ := sorry

-- Function to calculate the perimeter of an n-gon
noncomputable def perimeter (ngon : NGon) : ℝ := sorry

-- Theorem statement
theorem regular_ngon_minimal (circle : Circle) (n : ℕ) :
  ∀ (ngon : NGon), 
    ngon.n = n → 
    isInscribed ngon circle → 
    ∃ (regular_ngon : NGon), 
      regular_ngon.n = n ∧ 
      isInscribed regular_ngon circle ∧ 
      isRegular regular_ngon ∧ 
      area regular_ngon ≤ area ngon ∧ 
      perimeter regular_ngon ≤ perimeter ngon :=
by sorry

end NUMINAMATH_CALUDE_regular_ngon_minimal_l904_90438


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l904_90485

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2)
  (h_sin : Real.sin (θ / 2) = Real.sqrt ((x - 1) / (2 * x)))
  (h_x_pos : x > 0) : 
  Real.tan θ = Real.sqrt (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l904_90485


namespace NUMINAMATH_CALUDE_range_of_f_l904_90428

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -2} :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l904_90428


namespace NUMINAMATH_CALUDE_yellow_tiled_area_l904_90465

theorem yellow_tiled_area (length : ℝ) (width : ℝ) (yellow_ratio : ℝ) : 
  length = 3.6 → 
  width = 2.5 * length → 
  yellow_ratio = 1 / 2 → 
  yellow_ratio * (length * width) = 16.2 := by
sorry

end NUMINAMATH_CALUDE_yellow_tiled_area_l904_90465


namespace NUMINAMATH_CALUDE_z_share_per_x_rupee_l904_90407

/-- Given a total amount divided among three parties x, y, and z, 
    this theorem proves the ratio of z's share to x's share. -/
theorem z_share_per_x_rupee 
  (total : ℝ) 
  (x_share : ℝ) 
  (y_share : ℝ) 
  (z_share : ℝ) 
  (h1 : total = 78) 
  (h2 : y_share = 18) 
  (h3 : y_share = 0.45 * x_share) 
  (h4 : total = x_share + y_share + z_share) : 
  z_share / x_share = 0.5 := by
sorry

end NUMINAMATH_CALUDE_z_share_per_x_rupee_l904_90407


namespace NUMINAMATH_CALUDE_solution_to_polynomial_equation_l904_90456

theorem solution_to_polynomial_equation : ∃ x : ℤ, x^5 - 101*x^3 - 999*x^2 + 100900 = 0 :=
by
  use 10
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_solution_to_polynomial_equation_l904_90456


namespace NUMINAMATH_CALUDE_film_rewinding_time_l904_90424

/-- Time required to rewind a film onto a reel -/
theorem film_rewinding_time
  (a L S ω : ℝ)
  (ha : a > 0)
  (hL : L > 0)
  (hS : S > 0)
  (hω : ω > 0) :
  ∃ T : ℝ,
    T > 0 ∧
    T = (π / (S * ω)) * (Real.sqrt (a^2 + (4 * S * L / π)) - a) :=
by sorry

end NUMINAMATH_CALUDE_film_rewinding_time_l904_90424


namespace NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l904_90436

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point A -/
def point_A : Point :=
  { x := 5, y := -4 }

/-- Theorem: Point A is in the fourth quadrant -/
theorem point_A_in_fourth_quadrant : is_in_fourth_quadrant point_A := by
  sorry


end NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l904_90436


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l904_90427

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l904_90427


namespace NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l904_90412

/-- Calculates the number of pages to write per day given total pages and number of days -/
def pagesPerDay (totalPages : ℕ) (numDays : ℕ) : ℚ :=
  totalPages / numDays

theorem stacy_paper_pages_per_day :
  let totalPages : ℕ := 33
  let numDays : ℕ := 3
  pagesPerDay totalPages numDays = 11 := by
  sorry

end NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l904_90412


namespace NUMINAMATH_CALUDE_largest_among_expressions_l904_90409

theorem largest_among_expressions : 
  let a := -|(-3)|^3
  let b := -(-3)^3
  let c := (-3)^3
  let d := -(3^3)
  (b ≥ a) ∧ (b ≥ c) ∧ (b ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_largest_among_expressions_l904_90409


namespace NUMINAMATH_CALUDE_hall_area_l904_90461

/-- Proves that the area of a rectangular hall is 500 square meters, given that its length is 25 meters and 5 meters more than its breadth. -/
theorem hall_area : 
  ∀ (length breadth : ℝ),
  length = 25 →
  length = breadth + 5 →
  length * breadth = 500 := by
sorry

end NUMINAMATH_CALUDE_hall_area_l904_90461


namespace NUMINAMATH_CALUDE_sufficient_condition_for_p_l904_90457

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define what it means for a condition to be sufficient but not necessary
def sufficient_but_not_necessary (condition : ℝ → Prop) (proposition : ℝ → Prop) : Prop :=
  (∃ a : ℝ, condition a ∧ proposition a) ∧
  (∃ a : ℝ, ¬condition a ∧ proposition a)

-- Theorem statement
theorem sufficient_condition_for_p :
  sufficient_but_not_necessary (λ a : ℝ => a = 2) p :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_p_l904_90457


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_root_l904_90433

theorem cubic_polynomials_common_root :
  ∃ (c d : ℝ), c = -3 ∧ d = -4 ∧
  ∃ (x : ℝ), x^3 + c*x^2 + 15*x + 10 = 0 ∧ x^3 + d*x^2 + 17*x + 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_root_l904_90433


namespace NUMINAMATH_CALUDE_equation_solutions_inequality_solutions_l904_90475

/-- Part a: Solutions for 1/x + 1/y + 1/z = 1 where x, y, z are natural numbers -/
def solutions_a : Set (ℕ × ℕ × ℕ) :=
  {(3, 3, 3), (6, 3, 2), (4, 4, 2)}

/-- Part b: Solutions for 1/x + 1/y + 1/z > 1 where x, y, z are natural numbers greater than 1 -/
def solutions_b : Set (ℕ × ℕ × ℕ) :=
  {(x, 2, 2) | x > 1} ∪ {(3, 3, 2), (4, 3, 2), (5, 3, 2)}

theorem equation_solutions (x y z : ℕ) :
  (1 / x + 1 / y + 1 / z = 1) ↔ (x, y, z) ∈ solutions_a := by
  sorry

theorem inequality_solutions (x y z : ℕ) :
  (x > 1 ∧ y > 1 ∧ z > 1 ∧ 1 / x + 1 / y + 1 / z > 1) ↔ (x, y, z) ∈ solutions_b := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_inequality_solutions_l904_90475


namespace NUMINAMATH_CALUDE_drawer_probability_verify_drawer_probability_l904_90468

/-- The probability of selecting one shirt, one pair of shorts, and one pair of socks
    from a drawer with 6 shirts, 7 pairs of shorts, and 8 pairs of socks
    when randomly removing three articles of clothing. -/
theorem drawer_probability : ℕ → ℕ → ℕ → ℚ
  | 6, 7, 8 => 168/665
  | _, _, _ => 0

/-- Verifies that the probability is correct for the given problem. -/
theorem verify_drawer_probability :
  drawer_probability 6 7 8 = 168/665 := by sorry

end NUMINAMATH_CALUDE_drawer_probability_verify_drawer_probability_l904_90468


namespace NUMINAMATH_CALUDE_child_tickets_sold_l904_90408

/-- Proves the number of child tickets sold in a theater --/
theorem child_tickets_sold (total_tickets : ℕ) (adult_price child_price total_revenue : ℚ) :
  total_tickets = 80 →
  adult_price = 12 →
  child_price = 5 →
  total_revenue = 519 →
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 63 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l904_90408


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l904_90404

theorem algebraic_expression_value (x : ℝ) : 2 * x^2 + 2 * x + 5 = 9 → 3 * x^2 + 3 * x - 7 = -1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l904_90404


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l904_90405

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ r ≠ p) →
  (∀ (x : ℝ), x^3 - 20*x^2 + 99*x - 154 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ (t : ℝ), t ≠ p ∧ t ≠ q ∧ t ≠ r → 
    1 / (t^3 - 20*t^2 + 99*t - 154) = A / (t - p) + B / (t - q) + C / (t - r)) →
  1 / A + 1 / B + 1 / C = 245 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l904_90405


namespace NUMINAMATH_CALUDE_tea_price_calculation_l904_90484

/-- The price of the first variety of tea in rupees per kg -/
def first_tea_price : ℝ := 126

/-- The price of the second variety of tea in rupees per kg -/
def second_tea_price : ℝ := 135

/-- The price of the third variety of tea in rupees per kg -/
def third_tea_price : ℝ := 175.5

/-- The price of the mixture in rupees per kg -/
def mixture_price : ℝ := 153

/-- The ratio of the first variety in the mixture -/
def first_ratio : ℝ := 1

/-- The ratio of the second variety in the mixture -/
def second_ratio : ℝ := 1

/-- The ratio of the third variety in the mixture -/
def third_ratio : ℝ := 2

theorem tea_price_calculation :
  first_tea_price * first_ratio + 
  second_tea_price * second_ratio + 
  third_tea_price * third_ratio = 
  mixture_price * (first_ratio + second_ratio + third_ratio) := by
  sorry

#check tea_price_calculation

end NUMINAMATH_CALUDE_tea_price_calculation_l904_90484


namespace NUMINAMATH_CALUDE_sqrt_five_irrational_and_greater_than_two_l904_90413

theorem sqrt_five_irrational_and_greater_than_two :
  ∃ x : ℝ, Irrational x ∧ x > 2 ∧ x = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_five_irrational_and_greater_than_two_l904_90413


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l904_90496

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ n > 1 ∧ ∀ (m : ℕ), m ^ 2 ∣ n → m = 1

theorem simplest_quadratic_radical : 
  ¬ is_simplest_quadratic_radical (Real.sqrt 4) ∧ 
  is_simplest_quadratic_radical (Real.sqrt 5) ∧ 
  ¬ is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧ 
  ¬ is_simplest_quadratic_radical (Real.sqrt 8) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l904_90496


namespace NUMINAMATH_CALUDE_angle_measure_proof_l904_90415

theorem angle_measure_proof (x : ℝ) : 
  (180 - x) = 3 * (90 - x) + 10 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l904_90415
