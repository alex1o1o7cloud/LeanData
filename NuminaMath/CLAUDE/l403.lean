import Mathlib

namespace smallest_integer_gcd_18_is_6_l403_40312

theorem smallest_integer_gcd_18_is_6 : 
  ∃ n : ℕ, n > 100 ∧ n.gcd 18 = 6 ∧ ∀ m : ℕ, m > 100 ∧ m.gcd 18 = 6 → n ≤ m := by
  sorry

end smallest_integer_gcd_18_is_6_l403_40312


namespace simplify_expression_l403_40369

theorem simplify_expression (x y z : ℝ) : 
  (3 * x - (2 * y - 4 * z)) - ((3 * x - 2 * y) - 5 * z) = 9 * z := by
  sorry

end simplify_expression_l403_40369


namespace xy_value_l403_40387

theorem xy_value (x y : ℝ) 
  (h1 : (4:ℝ)^x / (2:ℝ)^(x+y) = 16)
  (h2 : (9:ℝ)^(x+y) / (3:ℝ)^(5*y) = 81) : 
  x * y = 32 := by
sorry

end xy_value_l403_40387


namespace sqrt_product_equals_21_l403_40379

theorem sqrt_product_equals_21 (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (7 * x) * Real.sqrt (21 * x) = 21) : 
  x = 21 / 97 := by
sorry

end sqrt_product_equals_21_l403_40379


namespace shelter_dogs_count_l403_40329

theorem shelter_dogs_count :
  ∀ (dogs cats : ℕ),
  (dogs : ℚ) / cats = 15 / 7 →
  dogs / (cats + 20) = 15 / 11 →
  dogs = 75 :=
by
  sorry

end shelter_dogs_count_l403_40329


namespace arccos_range_for_sin_l403_40326

theorem arccos_range_for_sin (a : ℝ) (x : ℝ) (h1 : x = Real.sin a) (h2 : a ∈ Set.Icc (-π/4) (3*π/4)) :
  ∃ y ∈ Set.Icc 0 (3*π/4), y = Real.arccos x :=
sorry

end arccos_range_for_sin_l403_40326


namespace smallest_integer_l403_40351

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 45) :
  b ≥ 1080 ∧ ∀ c : ℕ, c < 1080 → Nat.lcm a c / Nat.gcd a c ≠ 45 := by
  sorry

end smallest_integer_l403_40351


namespace simplify_sqrt_sum_l403_40360

theorem simplify_sqrt_sum : 
  (Real.sqrt 726 / Real.sqrt 81) + (Real.sqrt 294 / Real.sqrt 49) = (33 * Real.sqrt 2 + 9 * Real.sqrt 6) / 9 := by
  sorry

end simplify_sqrt_sum_l403_40360


namespace meal_cost_calculation_l403_40374

/-- Proves that given a meal with specific tax and tip rates, and a total cost,
    the original meal cost can be determined. -/
theorem meal_cost_calculation (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
    (h_total : total_cost = 36.90)
    (h_tax : tax_rate = 0.09)
    (h_tip : tip_rate = 0.18) :
    ∃ (original_cost : ℝ), 
      original_cost * (1 + tax_rate + tip_rate) = total_cost ∧ 
      original_cost = 29 := by
  sorry

end meal_cost_calculation_l403_40374


namespace find_x_l403_40382

theorem find_x : ∃ x : ℤ, 9873 + x = 13800 ∧ x = 3927 := by
  sorry

end find_x_l403_40382


namespace vector_sum_zero_l403_40355

variable {V : Type*} [AddCommGroup V]

/-- Given four points A, B, C, and D in a vector space, 
    prove that AB + BD - AC - CD equals the zero vector -/
theorem vector_sum_zero (A B C D : V) : 
  (B - A) + (D - B) - (C - A) - (D - C) = (0 : V) := by
  sorry

end vector_sum_zero_l403_40355


namespace binary_111011_is_59_l403_40321

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.zip b (List.range b.length).reverse).foldl
    (fun acc (bit, power) => acc + if bit then 2^power else 0) 0

theorem binary_111011_is_59 :
  binary_to_decimal [true, true, true, false, true, true] = 59 := by
  sorry

end binary_111011_is_59_l403_40321


namespace linear_dependence_condition_l403_40332

def v1 : ℝ × ℝ × ℝ := (1, 2, 3)
def v2 (k : ℝ) : ℝ × ℝ × ℝ := (4, k, 6)

def is_linearly_dependent (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), (a, b) ≠ (0, 0) ∧ a • v1 + b • v2 = (0, 0, 0)

theorem linear_dependence_condition (k : ℝ) :
  is_linearly_dependent v1 (v2 k) ↔ k = 8 :=
sorry

end linear_dependence_condition_l403_40332


namespace two_dogs_weekly_distance_l403_40315

/-- The total distance walked by two dogs in a week, given their daily walking distances -/
def total_weekly_distance (dog1_daily : ℕ) (dog2_daily : ℕ) : ℕ :=
  (dog1_daily * 7) + (dog2_daily * 7)

/-- Theorem: The total distance walked by two dogs in a week is 70 miles -/
theorem two_dogs_weekly_distance :
  total_weekly_distance 2 8 = 70 := by
  sorry

end two_dogs_weekly_distance_l403_40315


namespace function_k_value_l403_40361

theorem function_k_value (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = k * x + 1) →
  f 2 = 3 →
  k = 1 := by
sorry

end function_k_value_l403_40361


namespace dodecahedron_path_count_l403_40366

/-- Represents a path on a dodecahedron -/
structure DodecahedronPath where
  start : (Int × Int × Int)
  finish : (Int × Int × Int)
  length : Nat
  visitsAllCorners : Bool
  cannotReturnToStart : Bool

/-- The number of valid paths on a dodecahedron meeting specific conditions -/
def countValidPaths : Nat :=
  sorry

theorem dodecahedron_path_count :
  let validPath : DodecahedronPath :=
    { start := (0, 0, 0),
      finish := (1, 1, 0),
      length := 19,
      visitsAllCorners := true,
      cannotReturnToStart := true }
  countValidPaths = 90 := by
  sorry

end dodecahedron_path_count_l403_40366


namespace intersection_of_M_and_N_l403_40309

def M : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (-1, 1) + x • (1, 2)}
def N : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (1, -2) + x • (2, 3)}

theorem intersection_of_M_and_N :
  M ∩ N = {(-13, -23)} := by sorry

end intersection_of_M_and_N_l403_40309


namespace susan_gave_eight_apples_l403_40328

/-- The number of apples Susan gave to Sean -/
def apples_from_susan (initial_apples final_apples : ℕ) : ℕ :=
  final_apples - initial_apples

/-- Theorem stating that Susan gave Sean 8 apples -/
theorem susan_gave_eight_apples (initial_apples final_apples : ℕ) 
  (h1 : initial_apples = 9)
  (h2 : final_apples = 17) :
  apples_from_susan initial_apples final_apples = 8 := by
  sorry

end susan_gave_eight_apples_l403_40328


namespace time_after_duration_l403_40372

/-- Represents time in a 12-hour format -/
structure Time12 where
  hour : Nat
  minute : Nat
  second : Nat
  isPM : Bool

/-- Adds a duration to a given time -/
def addDuration (t : Time12) (hours minutes seconds : Nat) : Time12 :=
  sorry

/-- Converts the hour component to 12-hour format -/
def to12Hour (h : Nat) : Nat :=
  sorry

theorem time_after_duration (initial : Time12) (final : Time12) :
  initial = Time12.mk 3 15 15 true →
  final = addDuration initial 196 58 16 →
  final.hour = 8 ∧ 
  final.minute = 13 ∧ 
  final.second = 31 ∧ 
  final.isPM = true ∧
  final.hour + final.minute + final.second = 52 :=
sorry

end time_after_duration_l403_40372


namespace quadratic_no_real_roots_l403_40354

theorem quadratic_no_real_roots (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a*x + 1 = 0) ↔ -2 < a ∧ a < 2 :=
sorry

end quadratic_no_real_roots_l403_40354


namespace parabola_standard_equation_l403_40339

/-- A parabola with directrix y = -4 has the standard equation x² = 16y -/
theorem parabola_standard_equation (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, y = -4 → (x^2 = 2*p*y ↔ x^2 = 16*y)) :=
by sorry

end parabola_standard_equation_l403_40339


namespace subtraction_result_l403_40324

theorem subtraction_result (x : ℝ) (h : x / 10 = 6) : x - 15 = 45 := by
  sorry

end subtraction_result_l403_40324


namespace dividend_calculation_l403_40337

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 158 := by
sorry

end dividend_calculation_l403_40337


namespace ceiling_floor_difference_l403_40392

theorem ceiling_floor_difference : ⌈((15 / 8 : ℚ) ^ 2 * (-34 / 4 : ℚ))⌉ - ⌊(15 / 8 : ℚ) * ⌊-34 / 4⌋⌋ = -12 := by
  sorry

end ceiling_floor_difference_l403_40392


namespace total_toys_l403_40319

theorem total_toys (mike_toys : ℕ) (annie_toys : ℕ) (tom_toys : ℕ) 
  (h1 : mike_toys = 6)
  (h2 : annie_toys = 3 * mike_toys)
  (h3 : tom_toys = annie_toys + 2) :
  mike_toys + annie_toys + tom_toys = 56 := by
  sorry

end total_toys_l403_40319


namespace solution_x_value_l403_40323

theorem solution_x_value (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 80) : x = 26 := by
  sorry

end solution_x_value_l403_40323


namespace divisible_by_nine_l403_40373

theorem divisible_by_nine : ∃ k : ℤ, 2^10 - 2^8 + 2^6 - 2^4 + 2^2 - 1 = 9 * k := by
  sorry

end divisible_by_nine_l403_40373


namespace rotation_result_l403_40306

/-- Applies a 270° counter-clockwise rotation to a complex number -/
def rotate270 (z : ℂ) : ℂ := -Complex.I * z

/-- The initial complex number -/
def initial : ℂ := 4 - 2 * Complex.I

/-- The result of rotating the initial complex number by 270° counter-clockwise -/
def rotated : ℂ := rotate270 initial

/-- Theorem stating that rotating 4 - 2i by 270° counter-clockwise results in -4i - 2 -/
theorem rotation_result : rotated = -4 * Complex.I - 2 := by sorry

end rotation_result_l403_40306


namespace power_product_evaluation_l403_40356

theorem power_product_evaluation (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 := by
  sorry

end power_product_evaluation_l403_40356


namespace addition_proof_l403_40389

theorem addition_proof : 72 + 15 = 87 := by
  sorry

end addition_proof_l403_40389


namespace spatial_relations_theorem_l403_40303

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Parallel relation between a line and a plane -/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two planes -/
def plane_parallel_plane (p1 p2 : Plane3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def line_parallel_line (l1 l2 : Line3D) : Prop :=
  sorry

/-- Perpendicular relation between a line and a plane -/
def line_perp_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem spatial_relations_theorem 
  (m n : Line3D) 
  (α β : Plane3D) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (∀ l : Line3D, line_parallel_plane m α → 
    (line_in_plane l α → line_parallel_line m l)) ∧
  (¬ (plane_parallel_plane α β → line_in_plane m α → 
    line_in_plane n β → line_parallel_line m n)) ∧
  (line_perp_plane m α → line_perp_plane n β → 
    line_parallel_line m n → plane_parallel_plane α β) ∧
  (plane_parallel_plane α β → line_in_plane m α → 
    line_parallel_plane m β) :=
by sorry

end spatial_relations_theorem_l403_40303


namespace sum_of_absolute_coefficients_l403_40365

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (1 - 2*x)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| = 3^8 := by
sorry

end sum_of_absolute_coefficients_l403_40365


namespace x_range_theorem_l403_40325

theorem x_range_theorem (x : ℝ) :
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) →
  x < 1 ∨ x > 3 :=
by sorry

end x_range_theorem_l403_40325


namespace circle_tangency_l403_40396

/-- Two circles are tangent internally if the distance between their centers
    equals the difference of their radii -/
def are_tangent_internally (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (r1 - r2)^2

theorem circle_tangency (m : ℝ) :
  are_tangent_internally (m, 0) (-1, 2*m) 2 3 →
  m = 0 ∨ m = -2/5 := by
  sorry

end circle_tangency_l403_40396


namespace plates_problem_l403_40320

theorem plates_problem (initial_plates added_plates total_plates : ℕ) 
  (h1 : added_plates = 37)
  (h2 : total_plates = 83)
  (h3 : initial_plates + added_plates = total_plates) :
  initial_plates = 46 := by
  sorry

end plates_problem_l403_40320


namespace complement_intersection_equality_l403_40307

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {2, 3, 5}

-- Define set N
def N : Set Nat := {4, 6}

-- Theorem statement
theorem complement_intersection_equality :
  (U \ M) ∩ N = {4, 6} := by
  sorry

end complement_intersection_equality_l403_40307


namespace cuts_equality_l403_40397

/-- Represents a bagel -/
structure Bagel :=
  (intact : Bool)

/-- Represents the result of cutting a bagel -/
inductive CutResult
  | Log
  | TwoSectors

/-- Function to cut a bagel -/
def cut_bagel (b : Bagel) (result : CutResult) : Nat :=
  match result with
  | CutResult.Log => 1
  | CutResult.TwoSectors => 1

/-- Theorem stating that the number of cuts is the same for both operations -/
theorem cuts_equality (b : Bagel) :
  cut_bagel b CutResult.Log = cut_bagel b CutResult.TwoSectors :=
by
  sorry

end cuts_equality_l403_40397


namespace reflection_across_y_axis_l403_40359

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem reflection_across_y_axis :
  let P : Point := { x := 4, y := -1 }
  reflectAcrossYAxis P = { x := -4, y := -1 } := by
  sorry

end reflection_across_y_axis_l403_40359


namespace xy_leq_half_sum_squares_l403_40317

theorem xy_leq_half_sum_squares (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := by
  sorry

end xy_leq_half_sum_squares_l403_40317


namespace walters_age_2001_l403_40331

theorem walters_age_2001 (walter_age_1996 : ℕ) (grandmother_age_1996 : ℕ) :
  (grandmother_age_1996 = 3 * walter_age_1996) →
  (1996 - walter_age_1996 + 1996 - grandmother_age_1996 = 3864) →
  (walter_age_1996 + (2001 - 1996) = 37) :=
by sorry

end walters_age_2001_l403_40331


namespace regular_star_points_l403_40305

/-- A p-pointed regular star with specific angle properties -/
structure RegularStar where
  p : ℕ
  angle_d : ℝ
  angle_c : ℝ
  angle_c_minus_d : angle_c = angle_d + 15
  sum_of_angles : p * angle_c + p * angle_d = 360

/-- The number of points in a regular star with given properties is 24 -/
theorem regular_star_points (star : RegularStar) : star.p = 24 := by
  sorry

end regular_star_points_l403_40305


namespace unique_solution_l403_40362

/-- Represents the ages of three brothers -/
structure BrothersAges where
  older : ℕ
  xiaoyong : ℕ
  younger : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : BrothersAges) : Prop :=
  ages.older = 20 ∧
  ages.older > ages.xiaoyong ∧
  ages.xiaoyong > ages.younger ∧
  ages.younger ≥ 1 ∧
  2 * ages.xiaoyong + 5 * ages.younger = 97

/-- The theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! ages : BrothersAges, satisfiesConditions ages ∧ ages.xiaoyong = 16 ∧ ages.younger = 13 :=
sorry

end unique_solution_l403_40362


namespace linear_function_existence_l403_40383

theorem linear_function_existence (k : ℝ) (h1 : k ≠ 0) 
  (h2 : ∀ x1 x2 : ℝ, x1 < x2 → k * x1 + 3 < k * x2 + 3) :
  ∃ y : ℝ, y = k * (-2) + 3 := by
  sorry

end linear_function_existence_l403_40383


namespace merchant_pricing_strategy_l403_40302

-- Define the discount rates and profit margin as real numbers between 0 and 1
def purchase_discount : Real := 0.3
def sale_discount : Real := 0.2
def profit_margin : Real := 0.3

-- Define the list price as an arbitrary positive real number
def list_price : Real := 100

-- Define the purchase price as a function of the list price and purchase discount
def purchase_price (lp : Real) : Real := lp * (1 - purchase_discount)

-- Define the marked price as a function of the list price
def marked_price (lp : Real) : Real := 1.25 * lp

-- Define the selling price as a function of the marked price and sale discount
def selling_price (mp : Real) : Real := mp * (1 - sale_discount)

-- Define the profit as a function of the selling price and purchase price
def profit (sp : Real) (pp : Real) : Real := sp - pp

-- Theorem statement
theorem merchant_pricing_strategy :
  profit (selling_price (marked_price list_price)) (purchase_price list_price) =
  profit_margin * selling_price (marked_price list_price) := by sorry

end merchant_pricing_strategy_l403_40302


namespace specific_arrangement_probability_l403_40301

def total_lamps : ℕ := 6
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 2
def lamps_turned_on : ℕ := 3

def probability_specific_arrangement : ℚ := 2 / 25

theorem specific_arrangement_probability :
  (Nat.choose total_lamps blue_lamps * Nat.choose total_lamps lamps_turned_on) *
  probability_specific_arrangement =
  (Nat.choose (total_lamps - 2) (blue_lamps - 1) * Nat.choose (total_lamps - 2) (lamps_turned_on - 1)) :=
by sorry

end specific_arrangement_probability_l403_40301


namespace southern_tents_l403_40343

/-- Represents the number of tents in different parts of the campsite -/
structure Campsite where
  total : ℕ
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Theorem stating the number of tents in the southern part of the campsite -/
theorem southern_tents (c : Campsite) 
  (h_total : c.total = 900)
  (h_north : c.north = 100)
  (h_east : c.east = 2 * c.north)
  (h_center : c.center = 4 * c.north)
  (h_sum : c.total = c.north + c.east + c.center + c.south) : 
  c.south = 200 := by
  sorry


end southern_tents_l403_40343


namespace factor_x_squared_minus_four_ninety_eight_squared_minus_four_exists_n_for_equation_three_nine_nine_nine_nine_nine_one_not_prime_l403_40363

-- Part (a)
theorem factor_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x - 2) * (x + 2) := by sorry

-- Part (b)
theorem ninety_eight_squared_minus_four : 98^2 - 4 = 100 * 96 := by sorry

-- Part (c)
theorem exists_n_for_equation : ∃ n : ℕ+, (20 - n) * (20 + n) = 391 ∧ n = 3 := by sorry

-- Part (d)
theorem three_nine_nine_nine_nine_nine_one_not_prime : ¬ Nat.Prime 3999991 := by sorry

end factor_x_squared_minus_four_ninety_eight_squared_minus_four_exists_n_for_equation_three_nine_nine_nine_nine_nine_one_not_prime_l403_40363


namespace arithmetic_sequence_ninth_term_l403_40381

theorem arithmetic_sequence_ninth_term
  (a : ℝ) (d : ℝ) -- first term and common difference
  (h1 : a + 2 * d = 23) -- third term is 23
  (h2 : a + 5 * d = 29) -- sixth term is 29
  : a + 8 * d = 35 := -- ninth term is 35
by sorry

end arithmetic_sequence_ninth_term_l403_40381


namespace great_eighteen_hockey_league_games_l403_40335

/-- Represents a sports league with the given structure -/
structure League where
  total_teams : ℕ
  divisions : ℕ
  teams_per_division : ℕ
  intra_division_games : ℕ
  inter_division_games : ℕ

/-- Calculates the total number of games in the league -/
def total_games (l : League) : ℕ :=
  (l.total_teams * (l.teams_per_division - 1) * l.intra_division_games +
   l.total_teams * (l.total_teams - l.teams_per_division) * l.inter_division_games) / 2

/-- Theorem stating that the given league structure results in 243 total games -/
theorem great_eighteen_hockey_league_games :
  ∃ (l : League),
    l.total_teams = 18 ∧
    l.divisions = 3 ∧
    l.teams_per_division = 6 ∧
    l.intra_division_games = 3 ∧
    l.inter_division_games = 1 ∧
    total_games l = 243 := by
  sorry


end great_eighteen_hockey_league_games_l403_40335


namespace tom_height_l403_40378

theorem tom_height (t m : ℝ) : 
  t = 0.75 * m →                     -- Tom was 25% shorter than Mary two years ago
  m + 4 = 1.2 * (1.2 * t) →          -- Mary is now 20% taller than Tom after both have grown
  1.2 * t = 45 :=                    -- Tom's current height is 45 inches
by sorry

end tom_height_l403_40378


namespace total_cats_l403_40334

theorem total_cats (asleep : ℕ) (awake : ℕ) (h1 : asleep = 92) (h2 : awake = 6) :
  asleep + awake = 98 := by
  sorry

end total_cats_l403_40334


namespace circle_area_theorem_l403_40376

theorem circle_area_theorem (r : ℝ) (h : 3 * (1 / (2 * Real.pi * r)) = r) : 
  Real.pi * r^2 = 3/2 := by
  sorry

end circle_area_theorem_l403_40376


namespace max_value_of_g_l403_40386

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end max_value_of_g_l403_40386


namespace fraction_zero_implies_x_zero_l403_40338

theorem fraction_zero_implies_x_zero (x : ℚ) : 
  x / (2 * x - 1) = 0 → x = 0 := by
  sorry

end fraction_zero_implies_x_zero_l403_40338


namespace knights_in_february_l403_40318

/-- Represents a city with knights and liars -/
structure City where
  inhabitants : ℕ
  knights_february : ℕ
  claims_february : ℕ
  claims_30th : ℕ

/-- The proposition that a city satisfies the given conditions -/
def satisfies_conditions (c : City) : Prop :=
  c.inhabitants = 366 ∧
  c.claims_february = 100 ∧
  c.claims_30th = 60 ∧
  c.knights_february ≤ 29

/-- The theorem stating that if a city satisfies the conditions, 
    then exactly 29 knights were born in February -/
theorem knights_in_february (c : City) :
  satisfies_conditions c → c.knights_february = 29 := by
  sorry

end knights_in_february_l403_40318


namespace pizza_theorem_l403_40353

def pizza_problem (total_served : ℕ) (successfully_served : ℕ) : Prop :=
  total_served - successfully_served = 6

theorem pizza_theorem : pizza_problem 9 3 := by
  sorry

end pizza_theorem_l403_40353


namespace share_distribution_l403_40327

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 578 →
  a = (2/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  a = 68 := by sorry

end share_distribution_l403_40327


namespace lowest_degree_is_four_l403_40330

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := ℕ → ℤ

/-- The degree of an IntPolynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- The set of coefficients of an IntPolynomial -/
def coeffSet (p : IntPolynomial) : Set ℤ := sorry

/-- Predicate for a polynomial satisfying the given conditions -/
def satisfiesCondition (p : IntPolynomial) : Prop :=
  ∃ b : ℤ, (∃ x ∈ coeffSet p, x < b) ∧ 
           (∃ y ∈ coeffSet p, y > b) ∧ 
           b ∉ coeffSet p

/-- The main theorem statement -/
theorem lowest_degree_is_four :
  ∃ p : IntPolynomial, satisfiesCondition p ∧ degree p = 4 ∧
  ∀ q : IntPolynomial, satisfiesCondition q → degree q ≥ 4 :=
sorry

end lowest_degree_is_four_l403_40330


namespace smallest_solution_floor_equation_l403_40384

theorem smallest_solution_floor_equation :
  ∃ x : ℝ, (∀ y : ℝ, (⌊y^2⌋ : ℤ) - (⌊y⌋ : ℤ)^2 = 21 → x ≤ y) ∧
            (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 21 ∧
            x > 11.5 ∧ x < 11.6 :=
by sorry

end smallest_solution_floor_equation_l403_40384


namespace sum_of_three_squares_l403_40314

theorem sum_of_three_squares (s t : ℝ) : 
  (3 * s + 2 * t = 27) → 
  (2 * s + 3 * t = 23) → 
  (s + 2 * t = 13) → 
  (3 * s = 21) := by
  sorry

end sum_of_three_squares_l403_40314


namespace equal_piece_length_equal_piece_length_proof_l403_40347

/-- Given a rope of 1165 cm cut into 154 pieces, where 4 pieces are 100mm each and the rest are equal,
    the length of each equal piece is 75 mm. -/
theorem equal_piece_length : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (special_pieces : ℕ) (special_length_mm : ℕ) =>
    total_length_cm = 1165 ∧
    total_pieces = 154 ∧
    equal_pieces = 150 ∧
    special_pieces = 4 ∧
    special_length_mm = 100 →
    (total_length_cm * 10 - special_pieces * special_length_mm) / equal_pieces = 75

/-- Proof of the theorem -/
theorem equal_piece_length_proof : equal_piece_length 1165 154 150 4 100 := by
  sorry

end equal_piece_length_equal_piece_length_proof_l403_40347


namespace smallest_largest_8digit_multiples_of_360_l403_40346

/-- Checks if a number has all unique digits --/
def hasUniqueDigits (n : Nat) : Bool :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- Checks if a number is a multiple of 360 --/
def isMultipleOf360 (n : Nat) : Bool :=
  n % 360 = 0

/-- Theorem: 12378960 and 98763120 are the smallest and largest 8-digit multiples of 360 with unique digits --/
theorem smallest_largest_8digit_multiples_of_360 :
  (∀ n : Nat, n ≥ 10000000 ∧ n < 100000000 ∧ isMultipleOf360 n ∧ hasUniqueDigits n →
    n ≥ 12378960) ∧
  (∀ n : Nat, n ≥ 10000000 ∧ n < 100000000 ∧ isMultipleOf360 n ∧ hasUniqueDigits n →
    n ≤ 98763120) ∧
  isMultipleOf360 12378960 ∧
  isMultipleOf360 98763120 ∧
  hasUniqueDigits 12378960 ∧
  hasUniqueDigits 98763120 :=
by sorry


end smallest_largest_8digit_multiples_of_360_l403_40346


namespace sufficient_not_necessary_condition_l403_40341

/-- A quadratic function f(x) = ax^2 + bx + c where a > 0 and f(1) = 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem sufficient_not_necessary_condition
  (a b c : ℝ)
  (h_a_pos : a > 0)
  (h_f_1_eq_0 : QuadraticFunction a b c 1 = 0) :
  (∀ a b c, b > 2 * a → QuadraticFunction a b c (-2) < 0) ∧
  (∃ a b c, QuadraticFunction a b c (-2) < 0 ∧ b ≤ 2 * a) :=
by sorry

end sufficient_not_necessary_condition_l403_40341


namespace largest_of_seven_consecutive_integers_l403_40333

theorem largest_of_seven_consecutive_integers (n : ℕ) :
  (n > 0) →
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 3010) →
  (n + 6 = 433) :=
by sorry

end largest_of_seven_consecutive_integers_l403_40333


namespace math_statements_l403_40371

theorem math_statements :
  (∃ x : ℚ, x < -1 ∧ x > 1/x) ∧
  (∃ y : ℝ, y ≥ 0 ∧ -y ≥ y) ∧
  (∀ z : ℚ, z < 0 → z^2 > z) :=
by sorry

end math_statements_l403_40371


namespace sin_315_degrees_l403_40380

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end sin_315_degrees_l403_40380


namespace physics_to_music_ratio_l403_40357

/-- Proves that the ratio of physics marks to music marks is 1:2 given the marks in other subjects and total marks -/
theorem physics_to_music_ratio (science music social_studies total : ℕ) (physics : ℚ) :
  science = 70 →
  music = 80 →
  social_studies = 85 →
  total = 275 →
  physics = music * (1 / 2) →
  science + music + social_studies + physics = total →
  physics / music = 1 / 2 := by
sorry

end physics_to_music_ratio_l403_40357


namespace division_problem_l403_40377

theorem division_problem : ∃ (a b c d : Nat), 
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧
  19858 / 102 = 1000 * a + 100 * b + 10 * c + d ∧
  19858 % 102 = 0 :=
by sorry

end division_problem_l403_40377


namespace remaining_tanning_time_l403_40393

/-- Calculates the remaining tanning time for the last two weeks of the month. -/
theorem remaining_tanning_time 
  (max_monthly_time : ℕ) 
  (daily_time : ℕ) 
  (days_per_week : ℕ) 
  (first_half_weeks : ℕ) 
  (h1 : max_monthly_time = 200)
  (h2 : daily_time = 30)
  (h3 : days_per_week = 2)
  (h4 : first_half_weeks = 2) :
  max_monthly_time - (daily_time * days_per_week * first_half_weeks) = 80 :=
by
  sorry

#check remaining_tanning_time

end remaining_tanning_time_l403_40393


namespace train_length_problem_l403_40349

/-- Proves that given two trains of equal length running on parallel lines in the same direction,
    with the faster train moving at 46 km/hr and the slower train at 36 km/hr,
    if the faster train passes the slower train in 72 seconds,
    then the length of each train is 100 meters. -/
theorem train_length_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 72 →
  (faster_speed - slower_speed) * passing_time * (5 / 18) = 2 * train_length →
  train_length = 100 := by
  sorry

end train_length_problem_l403_40349


namespace sum_reciprocals_bound_l403_40340

theorem sum_reciprocals_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 1/y ≥ 2 ∧ ∀ M : ℝ, ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 1/x + 1/y > M :=
by sorry

end sum_reciprocals_bound_l403_40340


namespace complex_number_opposites_l403_40398

theorem complex_number_opposites (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
sorry

end complex_number_opposites_l403_40398


namespace f_increasing_on_interval_f_max_on_interval_f_min_on_interval_l403_40395

-- Define the function f(x) = -x^2 + 2x
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem for monotonicity
theorem f_increasing_on_interval : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ < f x₂ := by sorry

-- Theorem for maximum value
theorem f_max_on_interval : 
  ∃ x : ℝ, x ∈ Set.Icc 0 5 ∧ f x = 1 ∧ ∀ y ∈ Set.Icc 0 5, f y ≤ f x := by sorry

-- Theorem for minimum value
theorem f_min_on_interval : 
  ∃ x : ℝ, x ∈ Set.Icc 0 5 ∧ f x = -15 ∧ ∀ y ∈ Set.Icc 0 5, f y ≥ f x := by sorry

end f_increasing_on_interval_f_max_on_interval_f_min_on_interval_l403_40395


namespace order_of_trigonometric_functions_l403_40300

theorem order_of_trigonometric_functions : 
  let a := Real.sin (Real.sin (2008 * π / 180))
  let b := Real.sin (Real.cos (2008 * π / 180))
  let c := Real.cos (Real.sin (2008 * π / 180))
  let d := Real.cos (Real.cos (2008 * π / 180))
  b < a ∧ a < d ∧ d < c := by sorry

end order_of_trigonometric_functions_l403_40300


namespace quadratic_solution_sum_l403_40342

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, (3 * x^2 + 8 = 4 * x - 7) ↔ (x = a + b * I ∨ x = a - b * I)) →
  a + b^2 = 47/9 := by
sorry

end quadratic_solution_sum_l403_40342


namespace f_of_two_eq_neg_eight_l403_40344

/-- Given a function f(x) = x^5 + ax^3 + bx + 1 where f(-2) = 10, prove that f(2) = -8 -/
theorem f_of_two_eq_neg_eight (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 + a*x^3 + b*x + 1)
    (h2 : f (-2) = 10) : 
  f 2 = -8 := by
  sorry

end f_of_two_eq_neg_eight_l403_40344


namespace difference_of_fractions_l403_40399

theorem difference_of_fractions : (7 / 8 : ℚ) * 320 - (11 / 16 : ℚ) * 144 = 181 := by
  sorry

end difference_of_fractions_l403_40399


namespace red_balls_unchanged_l403_40348

/-- A box containing colored balls -/
structure Box where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Remove one blue ball from the box -/
def removeOneBlueBall (b : Box) : Box :=
  { red := b.red, blue := b.blue - 1, yellow := b.yellow }

theorem red_balls_unchanged (initial : Box) (h : initial.blue ≥ 1) :
  (removeOneBlueBall initial).red = initial.red :=
by sorry

end red_balls_unchanged_l403_40348


namespace coefficient_of_expression_l403_40390

/-- The coefficient of a monomial is the numerical factor that multiplies the variables. -/
def coefficient (expression : ℚ) : ℚ := sorry

/-- The expression -2ab/3 -/
def expression : ℚ := -2 / 3

theorem coefficient_of_expression :
  coefficient expression = -2 / 3 := by sorry

end coefficient_of_expression_l403_40390


namespace geometric_arithmetic_progression_l403_40367

theorem geometric_arithmetic_progression (a b c : ℝ) (q : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positivity for decreasing sequence
  a > b ∧ b > c →  -- Decreasing sequence
  b = a * q ∧ c = a * q^2 →  -- Geometric progression
  2 * (2020 * b / 7) = 577 * a + c / 7 →  -- Arithmetic progression
  q = 1/2 := by sorry

end geometric_arithmetic_progression_l403_40367


namespace gcd_bound_for_special_lcm_l403_40336

theorem gcd_bound_for_special_lcm (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) → 
  (10^6 ≤ b ∧ b < 10^7) → 
  (10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) → 
  Nat.gcd a b < 1000 := by
sorry

end gcd_bound_for_special_lcm_l403_40336


namespace chess_team_photo_arrangements_l403_40394

def chess_team_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  2 * (Nat.factorial num_boys) * (Nat.factorial num_girls)

theorem chess_team_photo_arrangements :
  chess_team_arrangements 3 3 = 72 := by
  sorry

end chess_team_photo_arrangements_l403_40394


namespace quadratic_inequality_condition_l403_40313

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 > 0) → -2 < k ∧ k < 3 ∧ 
  ∃ k₀ : ℝ, -2 < k₀ ∧ k₀ < 3 ∧ ∃ x : ℝ, x^2 + k₀*x + 1 ≤ 0 :=
by sorry

end quadratic_inequality_condition_l403_40313


namespace fibonacci_inequality_l403_40388

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (min (fibonacci n / fibonacci (n - 1)) (fibonacci (n + 1) / fibonacci n) < a / b ∧
   a / b < max (fibonacci n / fibonacci (n - 1)) (fibonacci (n + 1) / fibonacci n)) →
  b ≥ fibonacci (n + 1) :=
by sorry

end fibonacci_inequality_l403_40388


namespace m_range_for_fourth_quadrant_l403_40391

/-- A point in the fourth quadrant has positive x-coordinate and negative y-coordinate -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The coordinates of point M are (m+2, m) -/
def point_M (m : ℝ) : ℝ × ℝ := (m + 2, m)

/-- Theorem stating the range of m for point M to be in the fourth quadrant -/
theorem m_range_for_fourth_quadrant :
  ∀ m : ℝ, is_in_fourth_quadrant (point_M m).1 (point_M m).2 ↔ -2 < m ∧ m < 0 := by
  sorry

end m_range_for_fourth_quadrant_l403_40391


namespace substitution_remainder_l403_40310

/-- Represents the number of players in a soccer team --/
def total_players : ℕ := 22

/-- Represents the number of starting players --/
def starting_players : ℕ := 11

/-- Represents the number of substitute players --/
def substitute_players : ℕ := 11

/-- Represents the maximum number of substitutions allowed --/
def max_substitutions : ℕ := 4

/-- Calculates the number of ways to make substitutions in a soccer game --/
def substitution_ways : ℕ := 
  1 + 11^2 + 11^2 * 10^2 + 11^2 * 10^2 * 9^2 + 11^2 * 10^2 * 9^2 * 8^2

/-- Theorem stating that the remainder when the number of substitution ways
    is divided by 1000 is 722 --/
theorem substitution_remainder :
  substitution_ways % 1000 = 722 := by sorry

end substitution_remainder_l403_40310


namespace total_players_l403_40385

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) 
  (h1 : kabadi = 10) 
  (h2 : kho_kho_only = 20) 
  (h3 : both = 5) : 
  kabadi + kho_kho_only - both = 25 := by
  sorry

end total_players_l403_40385


namespace distribute_4_3_l403_40368

/-- The number of ways to distribute n indistinguishable objects into k distinct containers,
    with each container receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 36 ways to distribute 4 indistinguishable objects into 3 distinct containers,
    with each container receiving at least one object. -/
theorem distribute_4_3 : distribute 4 3 = 36 := by sorry

end distribute_4_3_l403_40368


namespace alex_cell_phone_cost_l403_40304

/-- Represents the cell phone plan cost structure and usage --/
structure CellPhonePlan where
  baseCost : ℝ
  textCost : ℝ
  extraMinuteCost : ℝ
  freeHours : ℝ
  textsSent : ℝ
  hoursUsed : ℝ

/-- Calculates the total cost of the cell phone plan --/
def totalCost (plan : CellPhonePlan) : ℝ :=
  plan.baseCost +
  plan.textCost * plan.textsSent +
  plan.extraMinuteCost * (plan.hoursUsed - plan.freeHours) * 60

/-- Theorem stating that Alex's total cost is $45.00 --/
theorem alex_cell_phone_cost :
  let plan : CellPhonePlan := {
    baseCost := 30
    textCost := 0.04
    extraMinuteCost := 0.15
    freeHours := 25
    textsSent := 150
    hoursUsed := 26
  }
  totalCost plan = 45 := by
  sorry


end alex_cell_phone_cost_l403_40304


namespace willy_stuffed_animals_l403_40375

def stuffed_animals_total (initial : ℕ) (mom_gift : ℕ) (dad_multiplier : ℕ) : ℕ :=
  let after_mom := initial + mom_gift
  let dad_gift := after_mom * dad_multiplier
  after_mom + dad_gift

theorem willy_stuffed_animals :
  stuffed_animals_total 10 2 3 = 48 := by
  sorry

end willy_stuffed_animals_l403_40375


namespace book_sale_loss_percentage_l403_40352

/-- Proves that the loss percentage on the first book is 15% given the problem conditions --/
theorem book_sale_loss_percentage
  (total_cost : ℝ)
  (cost_book1 : ℝ)
  (gain_percentage : ℝ)
  (h1 : total_cost = 360)
  (h2 : cost_book1 = 210)
  (h3 : gain_percentage = 19)
  (h4 : ∃ (selling_price : ℝ),
    selling_price = cost_book1 * (1 - (loss_percentage / 100)) ∧
    selling_price = (total_cost - cost_book1) * (1 + (gain_percentage / 100))) :
  ∃ (loss_percentage : ℝ), loss_percentage = 15 := by
sorry


end book_sale_loss_percentage_l403_40352


namespace correct_classification_l403_40316

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the structure of a reasoning process
structure ReasoningProcess where
  description : String
  correct_type : ReasoningType

-- Define the three reasoning processes
def process1 : ReasoningProcess :=
  { description := "The probability of a coin landing heads up is determined to be 0.5 through numerous trials",
    correct_type := ReasoningType.Inductive }

def process2 : ReasoningProcess :=
  { description := "The function f(x) = x^2 - |x| is an even function",
    correct_type := ReasoningType.Deductive }

def process3 : ReasoningProcess :=
  { description := "Scientists invented the electronic eagle eye by studying the eyes of eagles",
    correct_type := ReasoningType.Analogical }

-- Theorem to prove
theorem correct_classification :
  (process1.correct_type = ReasoningType.Inductive) ∧
  (process2.correct_type = ReasoningType.Deductive) ∧
  (process3.correct_type = ReasoningType.Analogical) :=
by sorry

end correct_classification_l403_40316


namespace function_inequality_implies_m_bound_l403_40308

/-- Given a function f(x) = (1/2)x^4 - 2x^3 + 3m where x ∈ ℝ, 
    if f(x) + 12 ≥ 0 for all x, then m ≥ 1/2 -/
theorem function_inequality_implies_m_bound (m : ℝ) : 
  (∀ x : ℝ, (1/2) * x^4 - 2 * x^3 + 3 * m + 12 ≥ 0) → m ≥ 1/2 := by
  sorry

end function_inequality_implies_m_bound_l403_40308


namespace labourer_income_l403_40311

/-- Represents the financial situation of a labourer over a 10-month period. -/
structure LabourerFinances where
  monthly_income : ℝ
  first_period_length : ℕ := 6
  second_period_length : ℕ := 4
  first_period_expense : ℝ := 75
  second_period_expense : ℝ := 60
  savings : ℝ := 30

/-- The labourer's finances satisfy the given conditions. -/
def satisfies_conditions (f : LabourerFinances) : Prop :=
  (f.first_period_length * f.monthly_income < f.first_period_length * f.first_period_expense) ∧
  (f.second_period_length * f.monthly_income = 
    f.second_period_length * f.second_period_expense + 
    (f.first_period_length * f.first_period_expense - f.first_period_length * f.monthly_income) + 
    f.savings)

/-- The labourer's monthly income is 72 given the conditions. -/
theorem labourer_income (f : LabourerFinances) (h : satisfies_conditions f) : 
  f.monthly_income = 72 := by
  sorry


end labourer_income_l403_40311


namespace integer_triple_divisibility_l403_40358

theorem integer_triple_divisibility :
  ∀ p q r : ℕ,
    1 < p → p < q → q < r →
    (p * q * r - 1) % ((p - 1) * (q - 1) * (r - 1)) = 0 →
    ((p = 2 ∧ q = 4 ∧ r = 8) ∨ (p = 3 ∧ q = 5 ∧ r = 15)) :=
by sorry

end integer_triple_divisibility_l403_40358


namespace inequality_proof_l403_40350

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 3) : 
  1/a^3 + 1/b^3 + 1/c^3 + 1/d^3 ≤ 1/(a^3 * b^3 * c^3 * d^3) := by
  sorry

end inequality_proof_l403_40350


namespace employee_count_l403_40364

theorem employee_count (avg_salary : ℕ) (salary_increase : ℕ) (manager_salary : ℕ) :
  avg_salary = 1500 →
  salary_increase = 500 →
  manager_salary = 12000 →
  ∃ n : ℕ, n * avg_salary + manager_salary = (n + 1) * (avg_salary + salary_increase) ∧ n = 20 :=
by
  sorry

end employee_count_l403_40364


namespace range_of_a_l403_40370

def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0

def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a (a : ℝ) : ¬(p a ∨ q a) ↔ a ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 := by
  sorry

end range_of_a_l403_40370


namespace cubic_increasing_minor_premise_l403_40322

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the concept of a minor premise in a deduction
def IsMinorPremise (statement : Prop) (conclusion : Prop) : Prop :=
  statement → conclusion

-- Theorem statement
theorem cubic_increasing_minor_premise :
  IsMinorPremise (IsIncreasing f) (IsIncreasing f) :=
sorry

end cubic_increasing_minor_premise_l403_40322


namespace parabolas_intersection_l403_40345

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x | 3 * x^2 - 4 * x + 2 = -x^2 + 2 * x + 3}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y (x : ℝ) : ℝ :=
  3 * x^2 - 4 * x + 2

/-- The first parabola -/
def parabola1 (x : ℝ) : ℝ :=
  3 * x^2 - 4 * x + 2

/-- The second parabola -/
def parabola2 (x : ℝ) : ℝ :=
  -x^2 + 2 * x + 3

theorem parabolas_intersection :
  intersection_x = {(3 - Real.sqrt 13) / 4, (3 + Real.sqrt 13) / 4} ∧
  ∀ x ∈ intersection_x, intersection_y x = (74 + 14 * Real.sqrt 13 * (if x < 0 then -1 else 1)) / 16 ∧
  ∀ x : ℝ, parabola1 x = parabola2 x ↔ x ∈ intersection_x :=
by sorry


end parabolas_intersection_l403_40345
