import Mathlib

namespace tangent_length_to_given_circle_l3864_386411

/-- The circle passing through three given points -/
structure Circle where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The length of the tangent segment from a point to a circle -/
def tangentLength (p : ℝ × ℝ) (c : Circle) : ℝ := sorry

/-- The origin point (0,0) -/
def origin : ℝ × ℝ := (0, 0)

/-- The circle passing through (4,3), (8,6), and (9,12) -/
def givenCircle : Circle :=
  { point1 := (4, 3)
    point2 := (8, 6)
    point3 := (9, 12) }

theorem tangent_length_to_given_circle :
  tangentLength origin givenCircle = 5 * Real.sqrt 2 := by
  sorry

end tangent_length_to_given_circle_l3864_386411


namespace continuous_piecewise_function_l3864_386470

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then c * x + 2
  else if x ≥ -1 then 2 * x - 4
  else 3 * x - d

theorem continuous_piecewise_function (c d : ℝ) :
  Continuous (f c d) → c + d = -7 := by
  sorry

end continuous_piecewise_function_l3864_386470


namespace fixed_point_of_exponential_function_l3864_386433

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) + 2
  f 2 = 2 := by
  sorry

end fixed_point_of_exponential_function_l3864_386433


namespace point_on_line_segment_l3864_386410

structure Point where
  x : ℝ
  y : ℝ

def Triangle (A B C : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def OnSegment (D A B : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D.x = A.x + t * (B.x - A.x) ∧ D.y = A.y + t * (B.y - A.y)

theorem point_on_line_segment (A B C D : Point) :
  Triangle A B C →
  A = Point.mk 1 2 →
  B = Point.mk 4 6 →
  C = Point.mk 6 3 →
  OnSegment D A B →
  D.y = (4/3) * D.x - (2/3) →
  ∃ t : ℝ, 1 ≤ t ∧ t ≤ 4 ∧ D = Point.mk t ((4/3) * t - (2/3)) :=
by sorry

end point_on_line_segment_l3864_386410


namespace quadratic_one_solution_l3864_386427

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 16 = 0) ↔ (m = 8 * Real.sqrt 3 ∨ m = -8 * Real.sqrt 3) := by
  sorry

end quadratic_one_solution_l3864_386427


namespace inscribed_square_area_l3864_386412

/-- The area of a square inscribed in a circular segment with an arc of 60° and radius 2√3 + √17 is 1. -/
theorem inscribed_square_area (R : ℝ) (h : R = 2 * Real.sqrt 3 + Real.sqrt 17) : 
  let segment_arc : ℝ := 60 * π / 180
  let square_side : ℝ := (R * (Real.sqrt 17 - 2 * Real.sqrt 3)) / 5
  square_side ^ 2 = 1 := by
  sorry

end inscribed_square_area_l3864_386412


namespace project_scores_analysis_l3864_386406

def scores : List ℝ := [8, 10, 9, 7, 7, 9, 8, 9]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem project_scores_analysis :
  mode scores = 9 ∧
  median scores = 8.5 ∧
  range scores = 3 ∧
  mean scores ≠ 8.4 := by sorry

end project_scores_analysis_l3864_386406


namespace vector_calculation_l3864_386429

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_calculation (a b : V) : 
  (1 / 2 : ℝ) • ((2 : ℝ) • a - (4 : ℝ) • b) + (2 : ℝ) • b = a := by
  sorry

end vector_calculation_l3864_386429


namespace distance_between_trees_l3864_386445

/-- Given a yard of length 400 meters with 26 trees planted at equal distances,
    including one tree at each end, the distance between consecutive trees is 16 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 400 ∧ num_trees = 26 →
  (yard_length / (num_trees - 1 : ℝ)) = 16 := by
  sorry

end distance_between_trees_l3864_386445


namespace basketball_distribution_l3864_386448

theorem basketball_distribution (total : ℕ) (left : ℕ) (classes : ℕ) : 
  total = 54 → left = 5 → classes * ((total - left) / classes) = total - left → classes = 7 := by
  sorry

end basketball_distribution_l3864_386448


namespace two_books_different_subjects_l3864_386459

theorem two_books_different_subjects (math_books : ℕ) (chinese_books : ℕ) (english_books : ℕ) :
  math_books = 10 → chinese_books = 9 → english_books = 8 →
  (math_books * chinese_books) + (math_books * english_books) + (chinese_books * english_books) = 242 :=
by sorry

end two_books_different_subjects_l3864_386459


namespace solution_ratio_l3864_386464

/-- Given a system of linear equations with a non-zero solution (x, y, z) and parameter k:
    x + k*y + 4*z = 0
    4*x + k*y + z = 0
    3*x + 5*y - 2*z = 0
    Prove that xz/y^2 = 25 -/
theorem solution_ratio (k x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : x + k*y + 4*z = 0)
  (eq2 : 4*x + k*y + z = 0)
  (eq3 : 3*x + 5*y - 2*z = 0) :
  x*z / (y^2) = 25 := by
  sorry

end solution_ratio_l3864_386464


namespace max_value_of_f_l3864_386488

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the interval
def I : Set ℝ := Set.Icc (-4) 3

-- State the theorem
theorem max_value_of_f : 
  ∃ (x : ℝ), x ∈ I ∧ f x = 15 ∧ ∀ (y : ℝ), y ∈ I → f y ≤ f x :=
sorry

end max_value_of_f_l3864_386488


namespace same_parity_smallest_largest_l3864_386420

/-- A set with certain properties related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def smallest (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def largest (A : Set ℤ) : ℤ := sorry

/-- A function to determine if a number is even -/
def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_smallest_largest : 
  isEven (smallest A_P) ↔ isEven (largest A_P) := by sorry

end same_parity_smallest_largest_l3864_386420


namespace largest_d_for_g_range_contains_two_l3864_386435

/-- The function g(x) defined as x^2 - 6x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + d

/-- The theorem stating that the largest value of d such that 2 is in the range of g(x) is 11 -/
theorem largest_d_for_g_range_contains_two :
  ∃ (d_max : ℝ), d_max = 11 ∧
  (∀ d : ℝ, (∃ x : ℝ, g d x = 2) → d ≤ d_max) ∧
  (∃ x : ℝ, g d_max x = 2) :=
sorry

end largest_d_for_g_range_contains_two_l3864_386435


namespace n_range_theorem_l3864_386417

theorem n_range_theorem (x y m n : ℝ) :
  n ≤ x ∧ x < y ∧ y ≤ n + 1 ∧
  m ∈ Set.Ioo x y ∧
  |y| = |m| + |x| →
  -1 < n ∧ n < 1 :=
by sorry

end n_range_theorem_l3864_386417


namespace exponent_equality_l3864_386458

theorem exponent_equality (n : ℕ) : 4^8 = 4^n → n = 8 := by
  sorry

end exponent_equality_l3864_386458


namespace polygon_sides_from_interior_angle_sum_l3864_386472

-- Define a convex polygon
structure ConvexPolygon where
  sides : ℕ
  is_convex : sides ≥ 3

-- Define the sum of interior angles for a polygon
def sum_interior_angles (p : ConvexPolygon) : ℝ :=
  180 * (p.sides - 2 : ℝ)

-- Theorem statement
theorem polygon_sides_from_interior_angle_sum (p : ConvexPolygon) 
  (h : sum_interior_angles p - x = 2190)
  (hx : 0 < x ∧ x < 180) : p.sides = 15 := by
  sorry

#check polygon_sides_from_interior_angle_sum

end polygon_sides_from_interior_angle_sum_l3864_386472


namespace square_roots_problem_l3864_386437

theorem square_roots_problem (x : ℝ) (h1 : x > 0) 
  (h2 : ∃ a : ℝ, (2 - a)^2 = x ∧ (2*a + 1)^2 = x) :
  ∃ a : ℝ, a = -3 ∧ (17 - x)^(1/3 : ℝ) = -2 := by
sorry

end square_roots_problem_l3864_386437


namespace salary_increase_percentage_l3864_386424

/-- Given a salary that increases to $812 with a 16% raise, 
    prove that a 10% raise results in $770.0000000000001 -/
theorem salary_increase_percentage (S : ℝ) 
  (h1 : S + 0.16 * S = 812) 
  (h2 : S + 0.1 * S = 770.0000000000001) : 
  ∃ (P : ℝ), S + P * S = 770.0000000000001 ∧ P = 0.1 := by
  sorry

end salary_increase_percentage_l3864_386424


namespace range_of_function_l3864_386484

theorem range_of_function (x : ℝ) (h : x ≥ -1) :
  let y := (12 * Real.sqrt (x + 1)) / (3 * x + 4)
  0 ≤ y ∧ y ≤ 2 * Real.sqrt 3 :=
sorry


end range_of_function_l3864_386484


namespace function_value_problem_l3864_386481

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f ((x / 2) - 1) = 2 * x + 3)
  (h2 : f m = 6) : 
  m = -1/4 := by sorry

end function_value_problem_l3864_386481


namespace adam_current_age_l3864_386432

/-- Adam's current age -/
def adam_age : ℕ := sorry

/-- Tom's current age -/
def tom_age : ℕ := 12

/-- Years into the future -/
def years_future : ℕ := 12

/-- Combined age in the future -/
def combined_future_age : ℕ := 44

theorem adam_current_age :
  adam_age = 8 :=
by
  sorry

end adam_current_age_l3864_386432


namespace total_allocation_schemes_l3864_386483

def num_classes : ℕ := 4
def total_spots : ℕ := 5
def min_spots_class_a : ℕ := 2

def allocation_schemes (n c m : ℕ) : ℕ :=
  -- n: total spots
  -- c: number of classes
  -- m: minimum spots for Class A
  sorry

theorem total_allocation_schemes :
  allocation_schemes total_spots num_classes min_spots_class_a = 20 := by
  sorry

end total_allocation_schemes_l3864_386483


namespace equation_solution_l3864_386498

theorem equation_solution : ∃ x : ℝ, -200 * x = 1600 ∧ x = -8 := by
  sorry

end equation_solution_l3864_386498


namespace repeating_decimal_fraction_sum_l3864_386491

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 3 + 834 / 999 ∧
  n + d = 4830 := by
  sorry

end repeating_decimal_fraction_sum_l3864_386491


namespace max_gcd_of_consecutive_cubic_sequence_l3864_386463

theorem max_gcd_of_consecutive_cubic_sequence :
  let b : ℕ → ℕ := fun n => 150 + n^3
  let d : ℕ → ℕ := fun n => Nat.gcd (b n) (b (n + 1))
  ∀ n : ℕ, n ≥ 1 → d n ≤ 1 :=
by sorry

end max_gcd_of_consecutive_cubic_sequence_l3864_386463


namespace square_gt_one_vs_cube_gt_one_l3864_386401

theorem square_gt_one_vs_cube_gt_one :
  {a : ℝ | a^2 > 1} ⊃ {a : ℝ | a^3 > 1} ∧ {a : ℝ | a^2 > 1} ≠ {a : ℝ | a^3 > 1} :=
by sorry

end square_gt_one_vs_cube_gt_one_l3864_386401


namespace smallest_k_for_convergence_l3864_386496

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 3 * u n - 3 * (u n)^3

def L : ℚ := 1/3

theorem smallest_k_for_convergence :
  ∀ k : ℕ, k ≥ 1 → |u k - L| ≤ 1 / 3^300 ∧
  ∀ m : ℕ, m < k → |u m - L| > 1 / 3^300 →
  k = 1 := by sorry

end smallest_k_for_convergence_l3864_386496


namespace triangle_area_is_correct_l3864_386439

/-- The slope of the first line -/
def m₁ : ℚ := 1/3

/-- The slope of the second line -/
def m₂ : ℚ := 3

/-- The point of intersection of the first two lines -/
def A : ℚ × ℚ := (3, 3)

/-- The equation of the third line: x + y = 12 -/
def line3 (x y : ℚ) : Prop := x + y = 12

/-- The area of the triangle formed by the three lines -/
noncomputable def triangle_area : ℚ := sorry

theorem triangle_area_is_correct : triangle_area = 8625/1000 := by sorry

end triangle_area_is_correct_l3864_386439


namespace round_trip_ticket_percentage_l3864_386460

theorem round_trip_ticket_percentage (total_passengers : ℝ) 
  (h1 : (0.2 : ℝ) * total_passengers = (passengers_with_roundtrip_and_car : ℝ))
  (h2 : (0.5 : ℝ) * (passengers_with_roundtrip : ℝ) = passengers_with_roundtrip - passengers_with_roundtrip_and_car) :
  (passengers_with_roundtrip : ℝ) / total_passengers = (0.4 : ℝ) := by
sorry

end round_trip_ticket_percentage_l3864_386460


namespace unique_a_value_l3864_386450

def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem unique_a_value : ∃! a : ℝ, A a ∪ B a = {0, 1, 2, 3, 9} ∧ a = 3 := by
  sorry

end unique_a_value_l3864_386450


namespace max_tax_revenue_l3864_386476

-- Define the market conditions
def supply_function (P : ℝ) : ℝ := 6 * P - 312
def demand_slope : ℝ := 4
def tax_rate : ℝ := 30
def consumer_price : ℝ := 118

-- Define the demand function
def demand_function (P : ℝ) : ℝ := 688 - demand_slope * P

-- Define the tax revenue function
def tax_revenue (t : ℝ) : ℝ := (288 - 2.4 * t) * t

-- Theorem statement
theorem max_tax_revenue :
  ∃ (t : ℝ), ∀ (t' : ℝ), tax_revenue t ≥ tax_revenue t' ∧ tax_revenue t = 8640 := by
  sorry


end max_tax_revenue_l3864_386476


namespace darryl_melon_sales_l3864_386477

/-- Calculates the total money made from selling melons given the initial quantities,
    prices, dropped/rotten melons, and remaining melons. -/
def total_money_made (initial_cantaloupes initial_honeydews : ℕ)
                     (price_cantaloupe price_honeydew : ℕ)
                     (dropped_cantaloupes rotten_honeydews : ℕ)
                     (remaining_cantaloupes remaining_honeydews : ℕ) : ℕ :=
  let sold_cantaloupes := initial_cantaloupes - remaining_cantaloupes - dropped_cantaloupes
  let sold_honeydews := initial_honeydews - remaining_honeydews - rotten_honeydews
  sold_cantaloupes * price_cantaloupe + sold_honeydews * price_honeydew

/-- Theorem stating that Darryl made $85 from selling melons. -/
theorem darryl_melon_sales : 
  total_money_made 30 27 2 3 2 3 8 9 = 85 := by
  sorry


end darryl_melon_sales_l3864_386477


namespace inequality_proof_l3864_386428

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end inequality_proof_l3864_386428


namespace range_of_a_l3864_386440

-- Define the sets P and M
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}

-- State the theorem
theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ Set.Icc (-1) 1 := by
  sorry

end range_of_a_l3864_386440


namespace exists_complete_list_l3864_386469

-- Define the tournament structure
structure Tournament where
  players : Type
  played : players → players → Prop
  winner : players → players → Prop
  no_draw : ∀ (a b : players), played a b → (winner a b ∨ winner b a)
  all_play : ∀ (a b : players), a ≠ b → played a b
  no_self_play : ∀ (a : players), ¬played a a

-- Define the list of beaten players for each player
def beaten_list (t : Tournament) (a : t.players) : Set t.players :=
  {b | t.winner a b ∨ ∃ c, t.winner a c ∧ t.winner c b}

-- Theorem statement
theorem exists_complete_list (t : Tournament) :
  ∃ a : t.players, ∀ b : t.players, b ≠ a → b ∈ beaten_list t a :=
sorry

end exists_complete_list_l3864_386469


namespace tangent_line_implies_a_value_l3864_386434

/-- Given a curve y = ax - ln(x + 1), prove that if its tangent line at (0, 0) is y = 2x, then a = 3 -/
theorem tangent_line_implies_a_value (a : ℝ) : 
  (∀ x, ∃ y, y = a * x - Real.log (x + 1)) →  -- Curve equation
  (∃ m, ∀ x, 2 * x = m * x) →                 -- Tangent line at (0, 0) is y = 2x
  a = 3 := by
sorry

end tangent_line_implies_a_value_l3864_386434


namespace second_number_in_expression_l3864_386451

theorem second_number_in_expression : 
  ∃ x : ℝ, (26.3 * x * 20) / 3 + 125 = 2229 ∧ x = 12 := by
  sorry

end second_number_in_expression_l3864_386451


namespace vector_projection_on_x_axis_l3864_386461

theorem vector_projection_on_x_axis (a : ℝ) (φ : ℝ) :
  a = 5 →
  φ = Real.pi / 3 →
  a * Real.cos φ = 2.5 := by
  sorry

end vector_projection_on_x_axis_l3864_386461


namespace line_perp_to_parallel_planes_l3864_386409

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_to_parallel_planes 
  (l : Line) (α β : Plane) :
  perp l β → para α β → perp l α :=
sorry

end line_perp_to_parallel_planes_l3864_386409


namespace negative_m_exponent_division_l3864_386402

theorem negative_m_exponent_division (m : ℝ) : (-m)^6 / (-m)^3 = -m^3 := by sorry

end negative_m_exponent_division_l3864_386402


namespace final_bird_count_and_ratio_l3864_386407

/-- Represents the number of birds in the park -/
structure BirdCount where
  blackbirds : ℕ
  magpies : ℕ
  blueJays : ℕ
  robins : ℕ

/-- Calculates the initial bird count based on given conditions -/
def initialBirdCount : BirdCount :=
  { blackbirds := 3 * 7,
    magpies := 13,
    blueJays := 2 * 5,
    robins := 4 }

/-- Calculates the final bird count after changes -/
def finalBirdCount : BirdCount :=
  { blackbirds := initialBirdCount.blackbirds - 6,
    magpies := initialBirdCount.magpies + 8,
    blueJays := initialBirdCount.blueJays + 3,
    robins := initialBirdCount.robins }

/-- Calculates the total number of birds -/
def totalBirds (count : BirdCount) : ℕ :=
  count.blackbirds + count.magpies + count.blueJays + count.robins

/-- Theorem: The final number of birds is 53 and the ratio is 15:21:13:4 -/
theorem final_bird_count_and_ratio :
  totalBirds finalBirdCount = 53 ∧
  finalBirdCount.blackbirds = 15 ∧
  finalBirdCount.magpies = 21 ∧
  finalBirdCount.blueJays = 13 ∧
  finalBirdCount.robins = 4 := by
  sorry


end final_bird_count_and_ratio_l3864_386407


namespace horner_method_f_3_l3864_386400

/-- Horner's method for evaluating polynomials -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f (x : ℝ) : ℝ :=
  horner [7, 6, 5, 4, 3, 2, 1, 0] x

theorem horner_method_f_3 :
  f 3 = 21324 := by
  sorry

end horner_method_f_3_l3864_386400


namespace sphere_intersection_radius_l3864_386408

-- Define the sphere
def sphere_center : ℝ × ℝ × ℝ := (3, 5, -9)

-- Define the intersection circles
def xy_circle_center : ℝ × ℝ × ℝ := (3, 5, 0)
def xy_circle_radius : ℝ := 2

def xz_circle_center : ℝ × ℝ × ℝ := (0, 5, -9)

-- Theorem statement
theorem sphere_intersection_radius : 
  let s := Real.sqrt ((Real.sqrt 85)^2 - 3^2)
  s = Real.sqrt 76 :=
sorry

end sphere_intersection_radius_l3864_386408


namespace value_of_N_l3864_386493

theorem value_of_N : ∃ N : ℝ, (0.25 * N = 0.55 * 5000) ∧ (N = 11000) := by
  sorry

end value_of_N_l3864_386493


namespace cos_alpha_plus_pi_fourth_l3864_386447

theorem cos_alpha_plus_pi_fourth (α β : Real) : 
  (3 * Real.pi / 4 < α) ∧ (α < Real.pi) ∧
  (3 * Real.pi / 4 < β) ∧ (β < Real.pi) ∧
  (Real.sin (α + β) = -4/5) ∧
  (Real.sin (β - Real.pi/4) = 12/13) →
  Real.cos (α + Real.pi/4) = -63/65 := by
sorry

end cos_alpha_plus_pi_fourth_l3864_386447


namespace interest_equality_implies_second_sum_l3864_386430

/-- Given a total sum of 2769 divided into two parts, if the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    then the second part is 1704. -/
theorem interest_equality_implies_second_sum (total : ℝ) (first_part second_part : ℝ) :
  total = 2769 →
  first_part + second_part = total →
  (first_part * 3 * 8) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1704 :=
by sorry

end interest_equality_implies_second_sum_l3864_386430


namespace metro_line_stations_l3864_386404

theorem metro_line_stations (x : ℕ) (h : x * (x - 1) = 1482) :
  x * (x - 1) = 1482 ∧ x > 1 := by
  sorry

end metro_line_stations_l3864_386404


namespace dave_tickets_used_l3864_386478

/-- Given that Dave had 13 tickets initially and has 7 tickets left after buying toys,
    prove that he used 6 tickets to buy toys. -/
theorem dave_tickets_used (initial : ℕ) (left : ℕ) (used : ℕ) 
    (h1 : initial = 13) 
    (h2 : left = 7) 
    (h3 : used = initial - left) : 
  used = 6 := by
  sorry

end dave_tickets_used_l3864_386478


namespace overtime_pay_ratio_l3864_386499

/-- Calculates the ratio of overtime pay rate to regular pay rate -/
theorem overtime_pay_ratio (regular_rate : ℚ) (regular_hours : ℚ) (total_pay : ℚ) (overtime_hours : ℚ)
  (h1 : regular_rate = 3)
  (h2 : regular_hours = 40)
  (h3 : total_pay = 186)
  (h4 : overtime_hours = 11) :
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  let overtime_rate := overtime_pay / overtime_hours
  overtime_rate / regular_rate = 2 := by
  sorry


end overtime_pay_ratio_l3864_386499


namespace plan_A_fixed_charge_l3864_386475

/-- The fixed charge for the first 4 minutes in Plan A -/
def fixed_charge : ℝ := sorry

/-- The per-minute rate after the first 4 minutes in Plan A -/
def rate_A : ℝ := 0.06

/-- The per-minute rate for Plan B -/
def rate_B : ℝ := 0.08

/-- The duration at which both plans charge the same amount -/
def equal_duration : ℝ := 18

theorem plan_A_fixed_charge :
  fixed_charge = 0.60 :=
by
  have h1 : fixed_charge + rate_A * (equal_duration - 4) = rate_B * equal_duration :=
    sorry
  sorry


end plan_A_fixed_charge_l3864_386475


namespace point_coordinates_l3864_386452

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance to x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance to y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem statement -/
theorem point_coordinates (P : Point) 
  (h1 : isInSecondQuadrant P) 
  (h2 : distanceToXAxis P = 7) 
  (h3 : distanceToYAxis P = 3) : 
  P.x = -3 ∧ P.y = 7 := by
  sorry


end point_coordinates_l3864_386452


namespace estimate_population_characteristic_l3864_386492

/-- Given a population and a sample, estimate the total number with a certain characteristic -/
theorem estimate_population_characteristic
  (total_population : ℕ)
  (sample_size : ℕ)
  (sample_with_characteristic : ℕ)
  (sample_size_positive : sample_size > 0)
  (sample_size_le_total : sample_size ≤ total_population)
  (sample_with_characteristic_le_sample : sample_with_characteristic ≤ sample_size) :
  let estimated_total := (total_population * sample_with_characteristic) / sample_size
  estimated_total = 6000 ∧ 
  estimated_total ≤ total_population ∧
  (sample_with_characteristic : ℚ) / (sample_size : ℚ) = 1/5 :=
by sorry

end estimate_population_characteristic_l3864_386492


namespace sin_alpha_value_l3864_386419

theorem sin_alpha_value (α : Real) (h1 : π/2 < α ∧ α < π) (h2 : 3 * Real.sin (2 * α) = Real.cos α) : 
  Real.sin α = 1/6 := by
  sorry

end sin_alpha_value_l3864_386419


namespace remainder_theorem_polynomial_remainder_l3864_386479

def p (x : ℝ) : ℝ := 4*x^8 - 2*x^6 + 5*x^4 - x^3 + 3*x - 15

theorem remainder_theorem (p : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + p a := sorry

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, p x = (2*x - 6) * q x + 25158 := by
  sorry

end remainder_theorem_polynomial_remainder_l3864_386479


namespace inequality_and_equality_condition_l3864_386446

theorem inequality_and_equality_condition (x y z : ℝ) (h : x^2 + y^2 + z^2 = 3) :
  x^3 - (y^2 + y*z + z^2)*x + y^2*z + y*z^2 ≤ 3 * Real.sqrt 3 ∧
  (x^3 - (y^2 + y*z + z^2)*x + y^2*z + y*z^2 = 3 * Real.sqrt 3 ↔
    ((x = Real.sqrt 3 ∧ y = 0 ∧ z = 0) ∨
     (x = -Real.sqrt 3 / 3 ∧ y = 2 * Real.sqrt 3 / 3 ∧ z = 2 * Real.sqrt 3 / 3))) :=
by sorry

end inequality_and_equality_condition_l3864_386446


namespace x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l3864_386422

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∃ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ ¬(x > 1)) := by
  sorry

end x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l3864_386422


namespace morse_code_symbols_l3864_386471

theorem morse_code_symbols : 
  (Finset.range 5).sum (fun i => 2^(i+1)) = 62 :=
sorry

end morse_code_symbols_l3864_386471


namespace average_after_17th_inning_l3864_386418

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an inning -/
def newAverage (stats : BatsmanStats) (runsScored : Nat) : Rat :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 3 after scoring 66 runs in the 17th inning, 
    then his average after the 17th inning is 18 -/
theorem average_after_17th_inning 
  (stats : BatsmanStats) 
  (h1 : stats.innings = 16)
  (h2 : newAverage stats 66 = stats.average + 3) :
  newAverage stats 66 = 18 := by
  sorry

end average_after_17th_inning_l3864_386418


namespace square_difference_divided_l3864_386426

theorem square_difference_divided : (111^2 - 99^2) / 12 = 210 := by
  sorry

end square_difference_divided_l3864_386426


namespace inequality_solution_comparison_l3864_386473

theorem inequality_solution_comparison (m n : ℝ) 
  (hm : 5 * m - 2 ≥ 3) 
  (hn : ¬(5 * n - 2 ≥ 3)) : 
  m > n :=
sorry

end inequality_solution_comparison_l3864_386473


namespace hyperbola_vertex_distance_l3864_386438

/-- The distance between vertices of a hyperbola with equation x^2/16 - y^2/9 = 1 is 8 -/
theorem hyperbola_vertex_distance :
  let h : ℝ × ℝ → ℝ := fun (x, y) ↦ x^2/16 - y^2/9 - 1
  ∃ v₁ v₂ : ℝ × ℝ, h v₁ = 0 ∧ h v₂ = 0 ∧ ‖v₁ - v₂‖ = 8 :=
by sorry

end hyperbola_vertex_distance_l3864_386438


namespace circle_tangent_to_x_axis_at_origin_l3864_386405

theorem circle_tangent_to_x_axis_at_origin 
  (D E F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 → 
    (∃ r : ℝ, r > 0 ∧ 
      ∀ x y : ℝ, (x^2 + y^2 = r^2) ↔ (x^2 + y^2 + D*x + E*y + F = 0)) ∧
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x| < δ → 
      ∃ y : ℝ, |y| < ε ∧ x^2 + y^2 + D*x + E*y + F = 0) ∧
    (0^2 + 0^2 + D*0 + E*0 + F = 0)) →
  D = 0 ∧ F = 0 ∧ E ≠ 0 :=
by sorry

end circle_tangent_to_x_axis_at_origin_l3864_386405


namespace sine_theorem_trihedral_angle_first_cosine_theorem_trihedral_angle_second_cosine_theorem_trihedral_angle_l3864_386441

/-- A trihedral angle with face angles α, β, γ and opposite dihedral angles A, B, C. -/
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real
  A : Real
  B : Real
  C : Real

/-- The sine theorem for a trihedral angle holds. -/
theorem sine_theorem_trihedral_angle (t : TrihedralAngle) :
  (Real.sin t.α) / (Real.sin t.A) = (Real.sin t.β) / (Real.sin t.B) ∧
  (Real.sin t.β) / (Real.sin t.B) = (Real.sin t.γ) / (Real.sin t.C) :=
sorry

/-- The first cosine theorem for a trihedral angle holds. -/
theorem first_cosine_theorem_trihedral_angle (t : TrihedralAngle) :
  Real.cos t.α = Real.cos t.β * Real.cos t.γ + Real.sin t.β * Real.sin t.γ * Real.cos t.A :=
sorry

/-- The second cosine theorem for a trihedral angle holds. -/
theorem second_cosine_theorem_trihedral_angle (t : TrihedralAngle) :
  Real.cos t.A = -Real.cos t.B * Real.cos t.C + Real.sin t.B * Real.sin t.C * Real.cos t.α :=
sorry

end sine_theorem_trihedral_angle_first_cosine_theorem_trihedral_angle_second_cosine_theorem_trihedral_angle_l3864_386441


namespace sin_300_degrees_l3864_386403

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l3864_386403


namespace sqrt_a_div_sqrt_b_equals_five_halves_l3864_386465

theorem sqrt_a_div_sqrt_b_equals_five_halves (a b : ℝ) 
  (h : (1/3)^2 + (1/4)^2 = (25*a / 61*b) * ((1/5)^2 + (1/6)^2)) : 
  Real.sqrt a / Real.sqrt b = 5/2 := by
  sorry

end sqrt_a_div_sqrt_b_equals_five_halves_l3864_386465


namespace sqrt_equation_solution_l3864_386425

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (3 + n) = 8 → n = 61 := by
  sorry

end sqrt_equation_solution_l3864_386425


namespace absolute_value_inequality_l3864_386436

theorem absolute_value_inequality (x : ℝ) : 
  x ≠ 0 → (|((x - 2) / x)| > ((x - 2) / x) ↔ 0 < x ∧ x < 2) := by sorry

end absolute_value_inequality_l3864_386436


namespace phone_number_probability_l3864_386457

def first_three_digits : ℕ := 3
def last_five_digits_arrangements : ℕ := 300

theorem phone_number_probability :
  let total_possibilities := first_three_digits * last_five_digits_arrangements
  (1 : ℚ) / total_possibilities = (1 : ℚ) / 900 :=
by sorry

end phone_number_probability_l3864_386457


namespace exam_pupils_count_l3864_386486

theorem exam_pupils_count :
  ∀ (n : ℕ) (total_marks : ℕ),
    n > 4 →
    total_marks = 39 * n →
    (total_marks - 71) / (n - 4) = 44 →
    n = 21 := by
  sorry

end exam_pupils_count_l3864_386486


namespace smallest_positive_integer_ending_in_3_divisible_by_5_l3864_386421

theorem smallest_positive_integer_ending_in_3_divisible_by_5 : ∃ n : ℕ,
  (n % 10 = 3) ∧ 
  (n % 5 = 0) ∧ 
  (∀ m : ℕ, m < n → (m % 10 = 3 → m % 5 ≠ 0)) ∧
  n = 53 := by
sorry

end smallest_positive_integer_ending_in_3_divisible_by_5_l3864_386421


namespace plains_routes_count_l3864_386449

/-- Represents the number of routes between two types of cities -/
structure RouteCount where
  total : ℕ
  mountain : ℕ
  plain : ℕ

/-- Calculates the number of routes between plains cities -/
def plainsRoutes (cities : ℕ × ℕ) (routes : RouteCount) : ℕ :=
  routes.total - routes.mountain - (cities.1 * 3 - 2 * routes.mountain) / 2

/-- Theorem stating the number of routes between plains cities -/
theorem plains_routes_count 
  (cities : ℕ × ℕ) 
  (routes : RouteCount) 
  (h1 : cities.1 + cities.2 = 100)
  (h2 : cities.1 = 30)
  (h3 : cities.2 = 70)
  (h4 : routes.total = 150)
  (h5 : routes.mountain = 21) :
  plainsRoutes cities routes = 81 := by
sorry

end plains_routes_count_l3864_386449


namespace max_rented_trucks_24_l3864_386453

/-- Represents the truck rental scenario for a week -/
structure TruckRental where
  total_trucks : ℕ
  returned_ratio : ℚ
  saturday_trucks : ℕ

/-- The maximum number of trucks that could have been rented out during the week -/
def max_rented_trucks (rental : TruckRental) : ℕ :=
  min rental.total_trucks (2 * rental.saturday_trucks)

/-- Theorem stating the maximum number of rented trucks for the given scenario -/
theorem max_rented_trucks_24 (rental : TruckRental) 
    (h1 : rental.total_trucks = 24)
    (h2 : rental.returned_ratio = 1/2)
    (h3 : rental.saturday_trucks ≥ 12) :
  max_rented_trucks rental = 24 := by
  sorry

#eval max_rented_trucks ⟨24, 1/2, 12⟩

end max_rented_trucks_24_l3864_386453


namespace dartboard_central_angle_l3864_386413

theorem dartboard_central_angle (probability : ℝ) (central_angle : ℝ) : 
  probability = 1 / 8 → central_angle = 45 := by
  sorry

end dartboard_central_angle_l3864_386413


namespace a_values_l3864_386431

def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

theorem a_values (a : ℝ) : A ∪ B a = A → a = 0 ∨ a = -1 ∨ a = -2 := by
  sorry

end a_values_l3864_386431


namespace book_cost_l3864_386444

theorem book_cost (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : num_books = 9)
  (h3 : remaining_amount = 16) :
  (initial_amount - remaining_amount) / num_books = 7 := by
  sorry

end book_cost_l3864_386444


namespace euclid_schools_count_l3864_386443

theorem euclid_schools_count :
  ∀ (n : ℕ) (andrea_rank beth_rank carla_rank : ℕ),
    -- Each school sends 3 students
    -- Total number of students is 3n
    -- Andrea's score is the median
    andrea_rank = (3 * n + 1) / 2 →
    -- Andrea's score is highest on her team
    andrea_rank < beth_rank →
    andrea_rank < carla_rank →
    -- Beth and Carla's ranks
    beth_rank = 37 →
    carla_rank = 64 →
    -- Each participant received a different score
    andrea_rank ≠ beth_rank ∧ andrea_rank ≠ carla_rank ∧ beth_rank ≠ carla_rank →
    -- Prove that the number of schools is 23
    n = 23 := by
  sorry


end euclid_schools_count_l3864_386443


namespace integral_sin_cos_power_l3864_386462

theorem integral_sin_cos_power : ∫ x in (-π/2)..0, (2^8 * Real.sin x^4 * Real.cos x^4) = 3*π := by sorry

end integral_sin_cos_power_l3864_386462


namespace square_of_99_l3864_386415

theorem square_of_99 : 99 * 99 = 9801 := by
  sorry

end square_of_99_l3864_386415


namespace sean_total_spend_l3864_386454

def almond_croissant_price : ℚ := 4.5
def salami_cheese_croissant_price : ℚ := 4.5
def plain_croissant_price : ℚ := 3
def focaccia_price : ℚ := 4
def latte_price : ℚ := 2.5

def almond_croissant_quantity : ℕ := 1
def salami_cheese_croissant_quantity : ℕ := 1
def plain_croissant_quantity : ℕ := 1
def focaccia_quantity : ℕ := 1
def latte_quantity : ℕ := 2

theorem sean_total_spend :
  almond_croissant_price * almond_croissant_quantity +
  salami_cheese_croissant_price * salami_cheese_croissant_quantity +
  plain_croissant_price * plain_croissant_quantity +
  focaccia_price * focaccia_quantity +
  latte_price * latte_quantity = 21 := by
sorry

end sean_total_spend_l3864_386454


namespace tangerine_sum_l3864_386487

theorem tangerine_sum (initial_count : ℕ) (final_counts : List ℕ) : 
  initial_count = 20 →
  final_counts = [10, 18, 17, 13, 16] →
  (final_counts.filter (· ≤ 13)).sum = 23 := by
  sorry

end tangerine_sum_l3864_386487


namespace garden_length_theorem_l3864_386468

/-- Represents a rectangular garden with given dimensions and area allocations. -/
structure Garden where
  length : ℝ
  width : ℝ
  tilled_ratio : ℝ
  trellised_ratio : ℝ
  raised_bed_area : ℝ

/-- Theorem stating the conditions and conclusion about the garden's length. -/
theorem garden_length_theorem (g : Garden) : 
  g.width = 120 ∧ 
  g.tilled_ratio = 1/2 ∧ 
  g.trellised_ratio = 1/3 ∧ 
  g.raised_bed_area = 8800 →
  g.length = 220 := by
  sorry

#check garden_length_theorem

end garden_length_theorem_l3864_386468


namespace initial_points_count_l3864_386489

/-- The number of points after one operation -/
def points_after_one_op (n : ℕ) : ℕ := 2 * n - 1

/-- The number of points after two operations -/
def points_after_two_ops (n : ℕ) : ℕ := 2 * (points_after_one_op n) - 1

/-- The number of points after three operations -/
def points_after_three_ops (n : ℕ) : ℕ := 2 * (points_after_two_ops n) - 1

/-- 
Theorem: If we start with n points on a line, perform the operation of adding a point 
between each pair of neighboring points three times, and end up with 65 points, 
then n must be equal to 9.
-/
theorem initial_points_count : points_after_three_ops 9 = 65 ∧ 
  (∀ m : ℕ, points_after_three_ops m = 65 → m = 9) := by
  sorry

end initial_points_count_l3864_386489


namespace binomial_coefficient_equality_l3864_386414

theorem binomial_coefficient_equality (m : ℕ) : 
  (Nat.choose 17 (3*m - 1) = Nat.choose 17 (2*m + 3)) → (m = 3 ∨ m = 4) := by
  sorry

end binomial_coefficient_equality_l3864_386414


namespace specific_rectangle_measurements_l3864_386482

/-- A rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculate the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculate the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem stating the area and perimeter of a specific rectangle -/
theorem specific_rectangle_measurements :
  let r : Rectangle := { length := 0.5, width := 0.36 }
  area r = 0.18 ∧ perimeter r = 1.72 := by
  sorry

end specific_rectangle_measurements_l3864_386482


namespace triangle_properties_l3864_386466

theorem triangle_properties (a b c A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  Real.sqrt 3 * b = 2 * a * Real.sin B * Real.cos C + 2 * c * Real.sin B * Real.cos A →
  a = 3 →
  c = 4 →
  B = π/3 ∧ b = Real.sqrt 13 ∧ Real.cos (2 * A + B) = -23/26 := by
  sorry

#check triangle_properties

end triangle_properties_l3864_386466


namespace cos_330_degrees_l3864_386442

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l3864_386442


namespace odd_function_implies_a_eq_two_l3864_386416

/-- The function f(x) = (x + a - 2)(2x² + a - 1) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a - 2) * (2 * x^2 + a - 1)

/-- f is an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_implies_a_eq_two :
  ∀ a : ℝ, is_odd_function (f a) → a = 2 := by sorry

end odd_function_implies_a_eq_two_l3864_386416


namespace domino_grid_side_divisible_by_four_l3864_386485

/-- A rectangular grid that can be cut into 1x2 dominoes with the property that any straight line
    along the grid lines intersects a multiple of four dominoes. -/
structure DominoCoveredGrid where
  a : ℕ  -- length of the grid
  b : ℕ  -- width of the grid
  is_valid : (a * b) % 2 = 0  -- ensures the grid can be covered by 1x2 dominoes
  line_cuts_multiple_of_four : ∀ (line : ℕ), line ≤ a ∨ line ≤ b → (line * 2) % 4 = 0

/-- If a rectangular grid can be covered by 1x2 dominoes such that any straight line along
    the grid lines intersects a multiple of four dominoes, then one of its sides is divisible by 4. -/
theorem domino_grid_side_divisible_by_four (grid : DominoCoveredGrid) :
  4 ∣ grid.a ∨ 4 ∣ grid.b :=
sorry

end domino_grid_side_divisible_by_four_l3864_386485


namespace A_sufficient_not_necessary_l3864_386474

-- Define propositions A and B
def A (x y : ℝ) : Prop := x + y ≠ 8
def B (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 6

-- Theorem statement
theorem A_sufficient_not_necessary :
  (∀ x y : ℝ, A x y → B x y) ∧
  ¬(∀ x y : ℝ, B x y → A x y) :=
by sorry

end A_sufficient_not_necessary_l3864_386474


namespace gcd_lcm_product_90_150_l3864_386467

theorem gcd_lcm_product_90_150 : 
  (Nat.gcd 90 150) * (Nat.lcm 90 150) = 13500 := by
  sorry

end gcd_lcm_product_90_150_l3864_386467


namespace minimum_value_implies_a_l3864_386495

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem minimum_value_implies_a (a : ℝ) :
  a > 0 →
  (∀ x, x > 0 → x ≤ Real.exp 1 → f a x ≥ 3/2) →
  (∃ x, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) :=
by sorry

end minimum_value_implies_a_l3864_386495


namespace snail_meets_minute_hand_l3864_386497

/-- Represents the position on a clock face in minutes past 12 -/
def ClockPosition := ℕ

/-- Calculates the position of the snail at a given time -/
def snail_position (time : ℕ) : ClockPosition :=
  (3 * time) % 60

/-- Calculates the position of the minute hand at a given time -/
def minute_hand_position (time : ℕ) : ClockPosition :=
  time % 60

/-- Checks if the snail and minute hand meet at a given time -/
def meets_at (time : ℕ) : Prop :=
  snail_position time = minute_hand_position time

theorem snail_meets_minute_hand :
  meets_at 40 ∧ meets_at 80 :=
sorry

end snail_meets_minute_hand_l3864_386497


namespace place_two_after_two_digit_number_l3864_386480

theorem place_two_after_two_digit_number (a b : ℕ) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≤ 9) : 
  (10 * a + b) * 10 + 2 = 100 * a + 10 * b + 2 := by
  sorry

end place_two_after_two_digit_number_l3864_386480


namespace nina_widget_problem_l3864_386423

theorem nina_widget_problem (x : ℝ) (h1 : 6 * x = 8 * (x - 1)) : 6 * x = 24 := by
  sorry

end nina_widget_problem_l3864_386423


namespace mod_equivalence_2021_l3864_386456

theorem mod_equivalence_2021 :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -2021 [ZMOD 13] ∧ n = 7 := by
  sorry

end mod_equivalence_2021_l3864_386456


namespace train_length_calculation_second_train_length_l3864_386494

/-- Calculates the length of the second train given the conditions of the problem -/
theorem train_length_calculation (length1 : ℝ) (speed1 speed2 : ℝ) (crossing_time : ℝ) : ℝ :=
  let km_per_hr_to_m_per_s : ℝ := 1000 / 3600
  let speed1_m_per_s : ℝ := speed1 * km_per_hr_to_m_per_s
  let speed2_m_per_s : ℝ := speed2 * km_per_hr_to_m_per_s
  let relative_speed : ℝ := speed1_m_per_s + speed2_m_per_s
  let total_distance : ℝ := relative_speed * crossing_time
  total_distance - length1

/-- The length of the second train is approximately 160 meters -/
theorem second_train_length :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |train_length_calculation 140 60 40 10.799136069114471 - 160| < ε :=
sorry

end train_length_calculation_second_train_length_l3864_386494


namespace jason_oranges_l3864_386490

/-- 
Given that Mary picked 122 oranges and the total number of oranges picked by Mary and Jason is 227,
prove that Jason picked 105 oranges.
-/
theorem jason_oranges :
  let mary_oranges : ℕ := 122
  let total_oranges : ℕ := 227
  let jason_oranges : ℕ := total_oranges - mary_oranges
  jason_oranges = 105 := by sorry

end jason_oranges_l3864_386490


namespace valid_triples_l3864_386455

def is_valid_triple (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  (a^2 * b) ∣ (a^3 + b^3 + c^3) ∧
  (b^2 * c) ∣ (a^3 + b^3 + c^3) ∧
  (c^2 * a) ∣ (a^3 + b^3 + c^3)

theorem valid_triples :
  ∀ a b c : ℕ, is_valid_triple a b c ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 2 ∧ c = 3) :=
by sorry

end valid_triples_l3864_386455
