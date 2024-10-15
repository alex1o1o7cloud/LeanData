import Mathlib

namespace NUMINAMATH_CALUDE_line_through_points_sum_of_coefficients_l2204_220496

-- Define the line equation
def line_equation (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem line_through_points_sum_of_coefficients :
  ∀ a b : ℝ,
  line_equation a b 2 = 3 →
  line_equation a b 10 = 19 →
  a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_line_through_points_sum_of_coefficients_l2204_220496


namespace NUMINAMATH_CALUDE_triangle_properties_l2204_220488

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (Real.cos t.A) / (1 + Real.sin t.A) = (Real.sin (2 * t.B)) / (1 + Real.cos (2 * t.B)))
  (h2 : t.C = 2 * Real.pi / 3)
  (h3 : t.A + t.B + t.C = Real.pi)
  (h4 : t.a / Real.sin t.A = t.b / Real.sin t.B)
  (h5 : t.b / Real.sin t.B = t.c / Real.sin t.C)
  : 
  (t.B = Real.pi / 6) ∧ 
  (∀ (x : Triangle), x.A + x.B + x.C = Real.pi → 
    (x.a^2 + x.b^2) / x.c^2 ≥ 4 * Real.sqrt 2 - 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2204_220488


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2204_220448

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2204_220448


namespace NUMINAMATH_CALUDE_twenty_eighth_term_of_sequence_l2204_220411

def sequence_term (n : ℕ) : ℚ :=
  1 / (2 ^ (sum_of_repeated_terms n))
where
  sum_of_repeated_terms : ℕ → ℕ
  | 0 => 0
  | k + 1 => if (sum_of_repeated_terms k + k + 1 < n) then k + 1 else k

theorem twenty_eighth_term_of_sequence :
  sequence_term 28 = 1 / (2 ^ 7) :=
sorry

end NUMINAMATH_CALUDE_twenty_eighth_term_of_sequence_l2204_220411


namespace NUMINAMATH_CALUDE_max_edges_100_vertices_triangle_free_l2204_220428

/-- The maximum number of edges in a triangle-free graph with n vertices -/
def maxEdgesTriangleFree (n : ℕ) : ℕ := n^2 / 4

/-- Theorem: In a graph with 100 vertices and no triangles, the maximum number of edges is 2500 -/
theorem max_edges_100_vertices_triangle_free :
  maxEdgesTriangleFree 100 = 2500 := by
  sorry

#eval maxEdgesTriangleFree 100  -- Should output 2500

end NUMINAMATH_CALUDE_max_edges_100_vertices_triangle_free_l2204_220428


namespace NUMINAMATH_CALUDE_gcd_multiple_equivalence_l2204_220418

theorem gcd_multiple_equivalence (d : ℕ) (h : d ≥ 1) :
  {m : ℕ | m ≥ 2 ∧ d ∣ m} =
  {m : ℕ | m ≥ 2 ∧ ∃ n : ℕ, n ≥ 1 ∧ Nat.gcd m n = d ∧ Nat.gcd m (4 * n + 1) = 1} :=
by sorry

end NUMINAMATH_CALUDE_gcd_multiple_equivalence_l2204_220418


namespace NUMINAMATH_CALUDE_parabola_intersection_implies_nonzero_c_l2204_220424

/-- Two points on a parabola -/
structure ParabolaPoints (a b c : ℝ) :=
  (x₁ : ℝ)
  (x₂ : ℝ)
  (y₁ : ℝ)
  (y₂ : ℝ)
  (on_parabola₁ : y₁ = x₁^2)
  (on_parabola₂ : y₂ = x₂^2)
  (on_quadratic₁ : y₁ = a * x₁^2 + b * x₁ + c)
  (on_quadratic₂ : y₂ = a * x₂^2 + b * x₂ + c)
  (opposite_sides : x₁ * x₂ < 0)
  (right_angle : (x₁ - x₂)^2 + (y₁ - y₂)^2 = x₁^2 + y₁^2 + x₂^2 + y₂^2)

theorem parabola_intersection_implies_nonzero_c (a b c : ℝ) :
  (∃ p : ParabolaPoints a b c, True) → c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_implies_nonzero_c_l2204_220424


namespace NUMINAMATH_CALUDE_wedding_rsvp_yes_percentage_l2204_220427

theorem wedding_rsvp_yes_percentage 
  (total_guests : ℕ) 
  (no_response_percentage : ℚ) 
  (no_reply_guests : ℕ) : 
  total_guests = 200 →
  no_response_percentage = 9 / 100 →
  no_reply_guests = 16 →
  (↑(total_guests - (total_guests * no_response_percentage).floor - no_reply_guests) / total_guests : ℚ) = 83 / 100 := by
sorry

end NUMINAMATH_CALUDE_wedding_rsvp_yes_percentage_l2204_220427


namespace NUMINAMATH_CALUDE_first_month_sale_proof_l2204_220406

/-- Calculates the sale in the first month given the sales for the next 5 months and the average sale for all 6 months. -/
def first_month_sale (month2 month3 month4 month5 month6 average : ℕ) : ℕ :=
  6 * average - (month2 + month3 + month4 + month5 + month6)

theorem first_month_sale_proof (month2 month3 month4 month5 month6 average : ℕ) :
  first_month_sale 6927 6855 7230 6562 7391 6900 = 6435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_proof_l2204_220406


namespace NUMINAMATH_CALUDE_range_of_a_for_single_root_l2204_220459

-- Define the function f(x) = 2x³ - 3x² + a
def f (x a : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

-- State the theorem
theorem range_of_a_for_single_root :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ Set.Icc (-2) 2 ∧ f x a = 0) →
  a ∈ Set.Ioo (-4) 0 ∪ Set.Ioo 1 28 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_single_root_l2204_220459


namespace NUMINAMATH_CALUDE_product_of_numbers_l2204_220453

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x - y = 4) (h3 : x / y = 5 / 4) :
  x * y = 320 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2204_220453


namespace NUMINAMATH_CALUDE_intersection_singleton_complement_intersection_l2204_220497

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

-- Part 1
theorem intersection_singleton (a : ℝ) : A ∩ B a = {2} → a = -1 ∨ a = -3 := by
  sorry

-- Part 2
theorem complement_intersection (a : ℝ) : A ∩ (Set.univ \ B a) = A →
  a < -3 ∨ (-3 < a ∧ a < -1 - Real.sqrt 3) ∨
  (-1 - Real.sqrt 3 < a ∧ a < -1) ∨
  (-1 < a ∧ a < -1 + Real.sqrt 3) ∨
  a > -1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_singleton_complement_intersection_l2204_220497


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l2204_220481

/-- Given a parabola y² = 4x with focus F(1, 0) and directrix x = -1,
    and a line through F with slope √3 intersecting the parabola above
    the x-axis at point A, prove that the area of triangle AFK is 4√3,
    where K is the foot of the perpendicular from A to the directrix. -/
theorem parabola_triangle_area :
  let parabola : ℝ × ℝ → Prop := λ p => p.2^2 = 4 * p.1
  let F : ℝ × ℝ := (1, 0)
  let directrix : ℝ → Prop := λ x => x = -1
  let line : ℝ → ℝ := λ x => Real.sqrt 3 * (x - 1)
  let A : ℝ × ℝ := (3, 2 * Real.sqrt 3)
  let K : ℝ × ℝ := (-1, 2 * Real.sqrt 3)
  parabola A ∧
  (∀ x, line x = A.2 ↔ x = A.1) ∧
  directrix K.1 ∧
  (A.2 - K.2) / (A.1 - K.1) * (F.2 - A.2) / (F.1 - A.1) = -1 →
  (1/2) * abs (A.1 - F.1) * abs (A.2 - K.2) = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l2204_220481


namespace NUMINAMATH_CALUDE_inequality_proof_l2204_220483

theorem inequality_proof (x : ℝ) : 4 ≤ (x + 1) / (3 * x - 7) ∧ (x + 1) / (3 * x - 7) < 9 ↔ x ∈ Set.Ioo (32 / 13) (29 / 11) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2204_220483


namespace NUMINAMATH_CALUDE_total_tickets_sold_l2204_220476

/-- Proves that the total number of tickets sold is 130 given the specified conditions -/
theorem total_tickets_sold (adult_price child_price total_receipts child_tickets : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_receipts = 840)
  (h4 : child_tickets = 90)
  : ∃ (adult_tickets : ℕ), adult_tickets * adult_price + child_tickets * child_price = total_receipts ∧ 
    adult_tickets + child_tickets = 130 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l2204_220476


namespace NUMINAMATH_CALUDE_cube_monotone_l2204_220484

theorem cube_monotone (a b : ℝ) : a > b → a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_monotone_l2204_220484


namespace NUMINAMATH_CALUDE_triangle_inequality_variant_l2204_220429

theorem triangle_inequality_variant (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_variant_l2204_220429


namespace NUMINAMATH_CALUDE_solve_for_y_l2204_220426

theorem solve_for_y (t : ℚ) (x y : ℚ) 
  (hx : x = 3 - 2 * t) 
  (hy : y = 3 * t + 10) 
  (hx_val : x = -4) : 
  y = 41 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2204_220426


namespace NUMINAMATH_CALUDE_music_tool_cost_proof_l2204_220482

/-- The cost of Joan's purchases at the music store -/
def total_spent : ℚ := 163.28

/-- The cost of the trumpet Joan bought -/
def trumpet_cost : ℚ := 149.16

/-- The cost of the song book Joan bought -/
def song_book_cost : ℚ := 4.14

/-- The cost of the music tool -/
def music_tool_cost : ℚ := total_spent - trumpet_cost - song_book_cost

theorem music_tool_cost_proof : music_tool_cost = 9.98 := by
  sorry

end NUMINAMATH_CALUDE_music_tool_cost_proof_l2204_220482


namespace NUMINAMATH_CALUDE_probability_red_black_white_l2204_220456

def total_balls : ℕ := 12
def red_balls : ℕ := 5
def black_balls : ℕ := 4
def white_balls : ℕ := 2
def green_balls : ℕ := 1

theorem probability_red_black_white :
  (red_balls + black_balls + white_balls : ℚ) / total_balls = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_black_white_l2204_220456


namespace NUMINAMATH_CALUDE_simplify_fraction_cube_l2204_220435

theorem simplify_fraction_cube (a b : ℝ) (ha : a ≠ 0) :
  (3 * b / (2 * a^2))^3 = 27 * b^3 / (8 * a^6) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_cube_l2204_220435


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l2204_220422

theorem complex_multiplication_result : (1 + Complex.I) * (-Complex.I) = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l2204_220422


namespace NUMINAMATH_CALUDE_valid_pairs_count_l2204_220461

/-- A function that checks if a positive integer has any zero digits -/
def has_zero_digit (n : ℕ+) : Bool :=
  sorry

/-- The set of positive integers less than or equal to 500 without zero digits -/
def valid_numbers : Set ℕ+ :=
  {n : ℕ+ | n ≤ 500 ∧ ¬(has_zero_digit n)}

/-- The number of ordered pairs (a, b) of positive integers where a + b = 500 
    and neither a nor b has a zero digit -/
def count_valid_pairs : ℕ :=
  sorry

theorem valid_pairs_count : count_valid_pairs = 93196 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l2204_220461


namespace NUMINAMATH_CALUDE_equation_general_form_l2204_220419

theorem equation_general_form :
  ∀ x : ℝ, (x + 8) * (x - 1) = -5 ↔ x^2 + 7*x - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_general_form_l2204_220419


namespace NUMINAMATH_CALUDE_root_reciprocal_relation_l2204_220433

theorem root_reciprocal_relation (p m q n : ℝ) : 
  (∃ x : ℝ, x^2 + p*x + q = 0 ∧ (1/x)^2 + m*(1/x) + n = 0) → 
  (p*n - m)*(q*m - p) = (q*n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_root_reciprocal_relation_l2204_220433


namespace NUMINAMATH_CALUDE_triangle_inequality_l2204_220466

theorem triangle_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (hab : a + b ≥ c) (hbc : b + c ≥ a) (hca : c + a ≥ b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2204_220466


namespace NUMINAMATH_CALUDE_group_size_calculation_l2204_220486

theorem group_size_calculation (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 55)
  (h2 : norway = 43)
  (h3 : both = 61)
  (h4 : neither = 63) :
  iceland + norway - both + neither = 161 :=
by sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2204_220486


namespace NUMINAMATH_CALUDE_f_geq_one_solution_set_g_max_value_l2204_220408

def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

def g (x : ℝ) : ℝ := f x - x^2 + x

theorem f_geq_one_solution_set :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

theorem g_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

end NUMINAMATH_CALUDE_f_geq_one_solution_set_g_max_value_l2204_220408


namespace NUMINAMATH_CALUDE_max_value_expression_l2204_220407

theorem max_value_expression (a b c : ℝ) 
  (ha : -1 ≤ a ∧ a ≤ 1) 
  (hb : -1 ≤ b ∧ b ≤ 1) 
  (hc : -1 ≤ c ∧ c ≤ 1) : 
  ∀ x y z : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ y ∧ y ≤ 1 → -1 ≤ z ∧ z ≤ 1 → 
  2 * Real.sqrt (a * b * c) + Real.sqrt ((1 - a^2) * (1 - b^2) * (1 - c^2)) ≤ 
  2 * Real.sqrt (x * y * z) + Real.sqrt ((1 - x^2) * (1 - y^2) * (1 - z^2)) → 
  2 * Real.sqrt (x * y * z) + Real.sqrt ((1 - x^2) * (1 - y^2) * (1 - z^2)) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2204_220407


namespace NUMINAMATH_CALUDE_shooting_game_equations_l2204_220441

/-- Represents the shooting game scenario -/
structure ShootingGame where
  x : ℕ  -- number of baskets Xiao Ming made
  y : ℕ  -- number of baskets his father made

/-- The conditions of the shooting game -/
def valid_game (g : ShootingGame) : Prop :=
  g.x + g.y = 20 ∧ 3 * g.x = g.y

theorem shooting_game_equations (g : ShootingGame) :
  valid_game g ↔ g.x + g.y = 20 ∧ 3 * g.x = g.y :=
sorry

end NUMINAMATH_CALUDE_shooting_game_equations_l2204_220441


namespace NUMINAMATH_CALUDE_millet_majority_on_sixth_day_l2204_220451

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  totalSeeds : ℚ
  milletSeeds : ℚ

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  let newTotalSeeds := state.totalSeeds + 2^(state.day - 1) / 2
  let newMilletSeeds := state.milletSeeds / 2 + 0.4 * 2^(state.day - 1) / 2
  { day := state.day + 1, totalSeeds := newTotalSeeds, milletSeeds := newMilletSeeds }

/-- The initial state of the feeder -/
def initialState : FeederState :=
  { day := 1, totalSeeds := 1/2, milletSeeds := 0.2 }

/-- Calculates the state of the feeder after n days -/
def stateAfterDays (n : Nat) : FeederState :=
  match n with
  | 0 => initialState
  | m + 1 => nextDay (stateAfterDays m)

/-- Theorem: On the 6th day, more than half of the seeds are millet -/
theorem millet_majority_on_sixth_day :
  let sixthDay := stateAfterDays 5
  sixthDay.milletSeeds > sixthDay.totalSeeds / 2 := by
  sorry

end NUMINAMATH_CALUDE_millet_majority_on_sixth_day_l2204_220451


namespace NUMINAMATH_CALUDE_sqrt_three_subset_M_l2204_220452

def M : Set ℝ := {x | x ≤ 3}

theorem sqrt_three_subset_M : {Real.sqrt 3} ⊆ M := by sorry

end NUMINAMATH_CALUDE_sqrt_three_subset_M_l2204_220452


namespace NUMINAMATH_CALUDE_office_average_age_l2204_220454

/-- The average age of all persons in an office, given specific conditions -/
theorem office_average_age :
  let total_persons : ℕ := 18
  let group1_size : ℕ := 5
  let group1_avg : ℚ := 14
  let group2_size : ℕ := 9
  let group2_avg : ℚ := 16
  let person15_age : ℕ := 56
  (total_persons : ℚ) * (average_age : ℚ) =
    (group1_size : ℚ) * group1_avg +
    (group2_size : ℚ) * group2_avg +
    (person15_age : ℚ) +
    ((total_persons - group1_size - group2_size - 1) : ℚ) * average_age →
  average_age = 270 / 14 := by
sorry

end NUMINAMATH_CALUDE_office_average_age_l2204_220454


namespace NUMINAMATH_CALUDE_pencil_distribution_ways_l2204_220445

/-- The number of ways to distribute n identical objects into k distinct groups --/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute pencils among friends --/
def distributePencils (totalPencils friendCount minPencils : ℕ) : ℕ :=
  let remainingPencils := totalPencils - friendCount * minPencils
  starsAndBars remainingPencils friendCount

theorem pencil_distribution_ways :
  distributePencils 12 4 2 = 35 := by
  sorry

#eval distributePencils 12 4 2

end NUMINAMATH_CALUDE_pencil_distribution_ways_l2204_220445


namespace NUMINAMATH_CALUDE_range_of_m_l2204_220443

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^2 - 4*x - 2*m + 1 ≤ 0) → m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2204_220443


namespace NUMINAMATH_CALUDE_least_value_quadratic_l2204_220403

theorem least_value_quadratic (y : ℝ) : 
  (2 * y^2 + 7 * y + 3 = 6) → 
  y ≥ (-7 - Real.sqrt 73) / 4 ∧ 
  ∃ (y_min : ℝ), 2 * y_min^2 + 7 * y_min + 3 = 6 ∧ y_min = (-7 - Real.sqrt 73) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l2204_220403


namespace NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l2204_220494

/-- Proves that the ratio of NY Mets fans to Boston Red Sox fans is 4:5 given the conditions -/
theorem mets_to_red_sox_ratio :
  ∀ (yankees mets red_sox : ℕ),
  yankees + mets + red_sox = 360 →
  3 * mets = 2 * yankees →
  mets = 96 →
  5 * mets = 4 * red_sox :=
by
  sorry

end NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l2204_220494


namespace NUMINAMATH_CALUDE_product_of_positive_real_solutions_l2204_220440

theorem product_of_positive_real_solutions (x : ℂ) : 
  (x^6 = -729) → 
  (∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^6 = -729 ∧ z.re > 0) ∧ 
    (∀ z, z^6 = -729 ∧ z.re > 0 → z ∈ S) ∧
    (S.prod id = 9)) := by
sorry

end NUMINAMATH_CALUDE_product_of_positive_real_solutions_l2204_220440


namespace NUMINAMATH_CALUDE_min_colors_l2204_220442

def is_divisor (a b : ℕ) : Prop := b % a = 0

def valid_coloring (f : ℕ → ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ 1000 ∧ 1 ≤ b ∧ b ≤ 1000 → 
    is_divisor a b → f a ≠ f b

theorem min_colors : 
  (∃ (n : ℕ) (f : ℕ → ℕ), n = 10 ∧ valid_coloring f ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 1000 → f i ≤ n)) ∧ 
  (∀ (m : ℕ) (g : ℕ → ℕ), m < 10 → 
    ¬(valid_coloring g ∧ (∀ i, 1 ≤ i ∧ i ≤ 1000 → g i ≤ m))) :=
by sorry

end NUMINAMATH_CALUDE_min_colors_l2204_220442


namespace NUMINAMATH_CALUDE_odd_function_value_l2204_220421

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the theorem
theorem odd_function_value (f g : ℝ → ℝ) 
  (h_odd : OddFunction f)
  (h_g : ∀ x, g x = f x + 6)
  (h_g_neg_one : g (-1) = 3) :
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l2204_220421


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l2204_220498

/-- A geometric sequence with sum S_n = (a-2)⋅3^(n+1) + 2 -/
def GeometricSequence (a : ℝ) (n : ℕ) : ℝ := (a - 2) * 3^(n + 1) + 2

/-- The difference between consecutive sums gives the n-th term -/
def NthTerm (a : ℝ) (n : ℕ) : ℝ := GeometricSequence a n - GeometricSequence a (n - 1)

/-- Theorem stating that the constant a in the given geometric sequence is 4/3 -/
theorem geometric_sequence_constant : 
  ∃ (a : ℝ), (∀ n : ℕ, n ≥ 2 → (NthTerm a n) / (NthTerm a (n-1)) = (NthTerm a (n-1)) / (NthTerm a (n-2))) ∧ 
  a = 4/3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l2204_220498


namespace NUMINAMATH_CALUDE_roots_equation_l2204_220446

theorem roots_equation (α β : ℝ) : 
  (α^2 - 3*α + 1 = 0) → 
  (β^2 - 3*β + 1 = 0) → 
  3*α^4 + 8*β^3 = 333 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l2204_220446


namespace NUMINAMATH_CALUDE_probability_green_or_blue_ten_sided_die_l2204_220412

/-- Represents a 10-sided die with colored faces -/
structure ColoredDie :=
  (total_sides : Nat)
  (red_faces : Nat)
  (yellow_faces : Nat)
  (green_faces : Nat)
  (blue_faces : Nat)
  (valid_die : total_sides = red_faces + yellow_faces + green_faces + blue_faces)

/-- Calculates the probability of rolling either a green or blue face -/
def probability_green_or_blue (die : ColoredDie) : Rat :=
  (die.green_faces + die.blue_faces : Rat) / die.total_sides

/-- Theorem stating the probability of rolling either a green or blue face -/
theorem probability_green_or_blue_ten_sided_die :
  ∃ (die : ColoredDie),
    die.total_sides = 10 ∧
    die.red_faces = 4 ∧
    die.yellow_faces = 3 ∧
    die.green_faces = 2 ∧
    die.blue_faces = 1 ∧
    probability_green_or_blue die = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_or_blue_ten_sided_die_l2204_220412


namespace NUMINAMATH_CALUDE_shopping_money_calculation_l2204_220489

theorem shopping_money_calculation (M : ℚ) : 
  (1 - 4/5 * (1 - 1/3 * (1 - 3/8))) * M = 1200 → M = 14400 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_calculation_l2204_220489


namespace NUMINAMATH_CALUDE_m_divided_by_8_l2204_220425

theorem m_divided_by_8 (m : ℕ) (h : m = 16^1500) : m / 8 = 2^5997 := by
  sorry

end NUMINAMATH_CALUDE_m_divided_by_8_l2204_220425


namespace NUMINAMATH_CALUDE_parabola_intersection_properties_l2204_220449

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2
def g (x : ℝ) : ℝ := 2 * x - 3
def h (x : ℝ) : ℝ := 2

-- Define the theorem
theorem parabola_intersection_properties
  (a : ℝ)
  (ha : a ≠ 0)
  (h_intersection : f a 1 = g 1) :
  (a = -1) ∧
  (∀ x, f a x = -x^2) ∧
  (∀ x, x < 0 → (∀ y, y < x → f a y < f a x)) ∧
  (let x₁ := Real.sqrt 2
   let x₂ := -Real.sqrt 2
   let area := (1/2) * (x₁ - x₂) * (h x₁ - f a 0)
   area = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_properties_l2204_220449


namespace NUMINAMATH_CALUDE_p_or_q_and_not_p_implies_q_l2204_220467

theorem p_or_q_and_not_p_implies_q (p q : Prop) :
  (p ∨ q) → ¬p → q := by sorry

end NUMINAMATH_CALUDE_p_or_q_and_not_p_implies_q_l2204_220467


namespace NUMINAMATH_CALUDE_mo_drinks_26_cups_l2204_220432

/-- Represents Mo's drinking habits and the weather conditions for a week -/
structure WeeklyDrinks where
  n : ℕ  -- Number of hot chocolate cups on a rainy day
  rainyDays : ℕ  -- Number of rainy days in the week
  teaPerNonRainyDay : ℕ  -- Number of tea cups on a non-rainy day
  teaExcess : ℕ  -- Excess of tea cups over hot chocolate cups

/-- Calculates the total number of cups (tea and hot chocolate) Mo drinks in a week -/
def totalCups (w : WeeklyDrinks) : ℕ :=
  w.n * w.rainyDays + w.teaPerNonRainyDay * (7 - w.rainyDays)

/-- Theorem stating that under the given conditions, Mo drinks 26 cups in total -/
theorem mo_drinks_26_cups (w : WeeklyDrinks)
  (h1 : w.rainyDays = 1)
  (h2 : w.teaPerNonRainyDay = 3)
  (h3 : w.teaPerNonRainyDay * (7 - w.rainyDays) = w.n * w.rainyDays + w.teaExcess)
  (h4 : w.teaExcess = 10) :
  totalCups w = 26 := by
  sorry

#check mo_drinks_26_cups

end NUMINAMATH_CALUDE_mo_drinks_26_cups_l2204_220432


namespace NUMINAMATH_CALUDE_smallest_a_value_l2204_220413

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) :
  a ≥ 15 ∧ ∀ a' ≥ 0, (∀ x : ℝ, Real.sin (a' * x + b) = Real.sin (15 * x)) → a' ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2204_220413


namespace NUMINAMATH_CALUDE_swamp_flies_eaten_l2204_220472

/-- Represents the number of animals in the swamp ecosystem -/
structure SwampPopulation where
  gharials : ℕ
  herons : ℕ
  caimans : ℕ
  fish : ℕ
  frogs : ℕ

/-- Calculates the total number of flies eaten daily in the swamp -/
def flies_eaten_daily (pop : SwampPopulation) : ℕ :=
  pop.frogs * 30 + pop.herons * 60

/-- Theorem stating the number of flies eaten daily in the given swamp ecosystem -/
theorem swamp_flies_eaten 
  (pop : SwampPopulation)
  (h_gharials : pop.gharials = 9)
  (h_herons : pop.herons = 12)
  (h_caimans : pop.caimans = 7)
  (h_fish : pop.fish = 20)
  (h_frogs : pop.frogs = 50) :
  flies_eaten_daily pop = 2220 := by
  sorry


end NUMINAMATH_CALUDE_swamp_flies_eaten_l2204_220472


namespace NUMINAMATH_CALUDE_worker_speed_comparison_l2204_220485

/-- Given that workers A and B can complete a work together in 18 days,
    and A alone can complete the work in 24 days,
    prove that A is 3 times faster than B. -/
theorem worker_speed_comparison (work : ℝ) (a_rate : ℝ) (b_rate : ℝ) :
  work > 0 →
  a_rate > 0 →
  b_rate > 0 →
  work / (a_rate + b_rate) = 18 →
  work / a_rate = 24 →
  a_rate / b_rate = 3 := by
  sorry

end NUMINAMATH_CALUDE_worker_speed_comparison_l2204_220485


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2204_220409

theorem trigonometric_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 3 / 5) : 
  ((2 * Real.sin α ^ 2 + Real.sin (2 * α)) / Real.cos (2 * α) = 24 / 7) ∧ 
  (Real.tan (α + 5 * Real.pi / 4) = 7) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2204_220409


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2204_220455

theorem imaginary_part_of_z (z : ℂ) : z = ((Complex.I - 1)^2 + 4) / (Complex.I + 1) → z.im = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2204_220455


namespace NUMINAMATH_CALUDE_smallest_k_inequality_l2204_220444

theorem smallest_k_inequality (k : ℝ) : k = 1 ↔ 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + k * (x - y)^2 ≥ Real.sqrt (x^2 + y^2)) ∧ 
  (∀ k' : ℝ, k' > 0 → k' < k → ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) + k' * (x - y)^2 < Real.sqrt (x^2 + y^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_l2204_220444


namespace NUMINAMATH_CALUDE_train_length_l2204_220495

/-- Given a train with speed 72 km/hr crossing a 260 m platform in 26 seconds, prove its length is 260 m -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * 1000 / 3600 →
  platform_length = 260 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 260 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2204_220495


namespace NUMINAMATH_CALUDE_robie_cards_count_l2204_220431

/-- The number of cards in each box -/
def cards_per_box : ℕ := 10

/-- The number of cards not placed in a box -/
def cards_outside_box : ℕ := 5

/-- The number of boxes Robie gave away -/
def boxes_given_away : ℕ := 2

/-- The number of boxes Robie has with him -/
def boxes_remaining : ℕ := 5

/-- The total number of cards Robie had in the beginning -/
def total_cards : ℕ := (boxes_given_away + boxes_remaining) * cards_per_box + cards_outside_box

theorem robie_cards_count : total_cards = 75 := by
  sorry

end NUMINAMATH_CALUDE_robie_cards_count_l2204_220431


namespace NUMINAMATH_CALUDE_cone_volume_given_sphere_l2204_220463

/-- Given a sphere and a cone with specific properties, prove that the volume of the cone is 12288π cm³ -/
theorem cone_volume_given_sphere (r_sphere : ℝ) (h_cone : ℝ) (r_cone : ℝ) :
  r_sphere = 24 →
  h_cone = 2 * r_sphere →
  π * r_cone * (r_cone + Real.sqrt (r_cone^2 + h_cone^2)) = 4 * π * r_sphere^2 →
  (1/3) * π * r_cone^2 * h_cone = 12288 * π := by
  sorry

#check cone_volume_given_sphere

end NUMINAMATH_CALUDE_cone_volume_given_sphere_l2204_220463


namespace NUMINAMATH_CALUDE_monkey_arrangements_l2204_220460

theorem monkey_arrangements :
  (Finset.range 6).prod (λ i => 6 - i) = 720 := by
  sorry

end NUMINAMATH_CALUDE_monkey_arrangements_l2204_220460


namespace NUMINAMATH_CALUDE_mary_regular_hours_l2204_220465

/-- Mary's work schedule and pay structure --/
structure MaryWork where
  max_hours : ℕ
  regular_rate : ℚ
  overtime_rate : ℚ
  total_earnings : ℚ

/-- Theorem stating Mary's regular work hours --/
theorem mary_regular_hours (w : MaryWork) 
  (h1 : w.max_hours = 60)
  (h2 : w.regular_rate = 8)
  (h3 : w.overtime_rate = w.regular_rate * (1 + 1/4))
  (h4 : w.total_earnings = 560) :
  ∃ (regular_hours overtime_hours : ℕ),
    regular_hours + overtime_hours = w.max_hours ∧
    regular_hours * w.regular_rate + overtime_hours * w.overtime_rate = w.total_earnings ∧
    regular_hours = 20 := by
  sorry

end NUMINAMATH_CALUDE_mary_regular_hours_l2204_220465


namespace NUMINAMATH_CALUDE_cereal_eating_time_l2204_220458

/-- The time it takes for Mr. Fat and Mr. Thin to eat 4 pounds of cereal together -/
def eating_time (fat_rate thin_rate total_cereal : ℚ) : ℕ :=
  (total_cereal / (fat_rate + thin_rate)).ceil.toNat

/-- Proves that Mr. Fat and Mr. Thin take 53 minutes to eat 4 pounds of cereal together -/
theorem cereal_eating_time :
  eating_time (1 / 20) (1 / 40) 4 = 53 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l2204_220458


namespace NUMINAMATH_CALUDE_equation_solution_l2204_220493

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  4 - 9 / x + 4 / (x^2) = 0 → (3 / x = 12 ∨ 3 / x = 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2204_220493


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l2204_220439

theorem quadrilateral_angle_measure (A B C D : ℝ) : 
  A + B = 180 →  -- ∠A + ∠B = 180°
  C = D →        -- ∠C = ∠D
  A = 40 →       -- ∠A = 40°
  B + C = 160 → -- ∠B + ∠C = 160°
  D = 20 :=      -- Prove that ∠D = 20°
by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l2204_220439


namespace NUMINAMATH_CALUDE_parallel_segments_length_l2204_220473

/-- Given three parallel line segments XY, UV, and PQ, where UV = 90 cm and XY = 120 cm,
    prove that the length of PQ is 360/7 cm. -/
theorem parallel_segments_length (XY UV PQ : ℝ) (h1 : XY = 120) (h2 : UV = 90)
    (h3 : ∃ (k : ℝ), XY = k * UV ∧ PQ = k * UV) : PQ = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_length_l2204_220473


namespace NUMINAMATH_CALUDE_triangle_sphere_distance_l2204_220490

/-- The distance between the plane containing a triangle inscribed on a sphere and the center of the sphere -/
theorem triangle_sphere_distance (a b c R : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) (hR : R = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area / s
  Real.sqrt (R^2 - r^2) = 2 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sphere_distance_l2204_220490


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2204_220430

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x + 15) - 4 / Real.sqrt (x + 15) = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2204_220430


namespace NUMINAMATH_CALUDE_four_solutions_l2204_220478

/-- The number of integer pairs (m, n) satisfying (m-1)(n-1) = 2 -/
def count_solutions : ℕ := 4

/-- A pair of integers (m, n) satisfies the equation (m-1)(n-1) = 2 -/
def is_solution (m n : ℤ) : Prop := (m - 1) * (n - 1) = 2

theorem four_solutions :
  (∃ (S : Finset (ℤ × ℤ)), S.card = count_solutions ∧
    (∀ (p : ℤ × ℤ), p ∈ S ↔ is_solution p.1 p.2) ∧
    (∀ (m n : ℤ), is_solution m n → (m, n) ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l2204_220478


namespace NUMINAMATH_CALUDE_angle_between_given_lines_l2204_220401

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 3 * y + 3 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the angle between two lines
def angle_between_lines (l1 l2 : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem angle_between_given_lines :
  angle_between_lines line1 line2 = Real.arctan (1 / 2) := by sorry

end NUMINAMATH_CALUDE_angle_between_given_lines_l2204_220401


namespace NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l2204_220492

/-- The shortest distance between a point and a parabola -/
theorem shortest_distance_point_to_parabola :
  let point := (6, 12)
  let parabola := {(x, y) : ℝ × ℝ | x = y^2 / 2}
  (shortest_distance : ℝ) →
  shortest_distance = 2 * Real.sqrt 17 ∧
  ∀ (p : ℝ × ℝ), p ∈ parabola → 
    Real.sqrt ((point.1 - p.1)^2 + (point.2 - p.2)^2) ≥ shortest_distance :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l2204_220492


namespace NUMINAMATH_CALUDE_no_integer_solution_l2204_220447

theorem no_integer_solution : ¬∃ (k l m n x : ℤ),
  (x = k * l * m * n) ∧
  (x - k = 1966) ∧
  (x - l = 966) ∧
  (x - m = 66) ∧
  (x - n = 6) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2204_220447


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2204_220479

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  fourth_quadrant 4 (-3) := by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2204_220479


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_m_l2204_220457

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem infinitely_many_divisible_by_m (m : ℤ) :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ m ∣ fib n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_m_l2204_220457


namespace NUMINAMATH_CALUDE_coltons_marbles_l2204_220417

theorem coltons_marbles (white_marbles : ℕ) : 
  (∃ (groups : ℕ), groups = 8 ∧ (16 + white_marbles) % groups = 0) →
  ∃ (k : ℕ), white_marbles = 8 * k :=
by sorry

end NUMINAMATH_CALUDE_coltons_marbles_l2204_220417


namespace NUMINAMATH_CALUDE_solve_equation_l2204_220471

theorem solve_equation : ∃ a : ℝ, -2 - a = 0 ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2204_220471


namespace NUMINAMATH_CALUDE_bacteria_growth_time_l2204_220480

theorem bacteria_growth_time (fill_time : ℕ) (initial_count : ℕ) : 
  (fill_time = 64 ∧ initial_count = 1) → 
  (∃ (new_fill_time : ℕ), new_fill_time = 62 ∧ 2^new_fill_time * initial_count * 4 = 2^fill_time) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_time_l2204_220480


namespace NUMINAMATH_CALUDE_exist_good_numbers_counterexample_l2204_220414

/-- A natural number is "good" if its decimal representation contains only 0s and 1s -/
def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- Sum of digits of a natural number in base 10 -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Statement: There exist two good numbers whose product is good, but the sum of digits
    of their product is not equal to the product of their sums of digits -/
theorem exist_good_numbers_counterexample :
  ∃ (A B : ℕ), is_good A ∧ is_good B ∧ is_good (A * B) ∧
    sum_of_digits (A * B) ≠ sum_of_digits A * sum_of_digits B :=
sorry

end NUMINAMATH_CALUDE_exist_good_numbers_counterexample_l2204_220414


namespace NUMINAMATH_CALUDE_sandy_sums_attempted_sandy_specific_case_l2204_220499

theorem sandy_sums_attempted (marks_per_correct : ℕ) (marks_per_incorrect : ℕ) 
  (total_marks : ℕ) (correct_sums : ℕ) : ℕ :=
  let total_sums := correct_sums + (marks_per_correct * correct_sums - total_marks) / marks_per_incorrect
  total_sums

theorem sandy_specific_case : sandy_sums_attempted 3 2 65 25 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sandy_sums_attempted_sandy_specific_case_l2204_220499


namespace NUMINAMATH_CALUDE_not_perfect_square_product_l2204_220423

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m

/-- The main theorem stating that 1, 2, and 4 are the only positive integers
    for which n(n+a) is not a perfect square for all positive integers n -/
theorem not_perfect_square_product (a : ℕ) : a > 0 →
  (∀ n : ℕ, n > 0 → ¬is_perfect_square (n * (n + a))) ↔ a = 1 ∨ a = 2 ∨ a = 4 :=
sorry

end NUMINAMATH_CALUDE_not_perfect_square_product_l2204_220423


namespace NUMINAMATH_CALUDE_increasing_function_parameter_range_l2204_220438

/-- Given a function f(x) = -1/3 * x^3 + 1/2 * x^2 + 2ax, 
    if f(x) is increasing on the interval (2/3, +∞), 
    then a ∈ (-1/9, +∞) -/
theorem increasing_function_parameter_range (a : ℝ) : 
  (∀ x > 2/3, (deriv (fun x => -1/3 * x^3 + 1/2 * x^2 + 2*a*x) x) > 0) →
  a > -1/9 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_parameter_range_l2204_220438


namespace NUMINAMATH_CALUDE_thousandth_term_of_sequence_l2204_220491

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem thousandth_term_of_sequence :
  arithmetic_sequence 1 3 1000 = 2998 := by
  sorry

end NUMINAMATH_CALUDE_thousandth_term_of_sequence_l2204_220491


namespace NUMINAMATH_CALUDE_box_weights_sum_l2204_220404

theorem box_weights_sum (box1 box2 box3 box4 box5 : ℝ) 
  (h1 : box1 = 2.5)
  (h2 : box2 = 11.3)
  (h3 : box3 = 5.75)
  (h4 : box4 = 7.2)
  (h5 : box5 = 3.25) :
  box1 + box2 + box3 + box4 + box5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_box_weights_sum_l2204_220404


namespace NUMINAMATH_CALUDE_octal_addition_example_l2204_220434

/-- Represents a digit in the octal number system -/
def OctalDigit : Type := Fin 8

/-- Represents an octal number as a list of octal digits -/
def OctalNumber : Type := List OctalDigit

/-- Addition operation for octal numbers -/
def octal_add : OctalNumber → OctalNumber → OctalNumber :=
  sorry

/-- Conversion from a natural number to an octal number -/
def nat_to_octal : Nat → OctalNumber :=
  sorry

/-- Theorem: 47 + 56 = 125 in the octal number system -/
theorem octal_addition_example :
  octal_add (nat_to_octal 47) (nat_to_octal 56) = nat_to_octal 125 := by
  sorry

end NUMINAMATH_CALUDE_octal_addition_example_l2204_220434


namespace NUMINAMATH_CALUDE_function_periodicity_l2204_220477

open Real

-- Define the function f and the constant a
variable (f : ℝ → ℝ) (a : ℝ)

-- State the theorem
theorem function_periodicity 
  (h : ∀ x, f (x + a) = (1 + f x) / (1 - f x)) 
  (ha : a ≠ 0) : 
  ∀ x, f (x + 4 * a) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l2204_220477


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2204_220405

theorem rectangle_diagonal (l w : ℝ) (h_area : l * w = 20) (h_perimeter : 2 * l + 2 * w = 18) :
  l^2 + w^2 = 41 :=
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2204_220405


namespace NUMINAMATH_CALUDE_folded_square_perimeter_l2204_220450

/-- A square with side length 2 is folded so that vertex A meets edge BC at A',
    and edge AB intersects edge CD at F. Given BA' = 1/2,
    prove that the perimeter of triangle CFA' is (3 + √17) / 2. -/
theorem folded_square_perimeter (A B C D A' F : ℝ × ℝ) : 
  let square_side : ℝ := 2
  let BA'_length : ℝ := 1/2
  -- Define the square
  (A = (0, square_side) ∧ B = (0, 0) ∧ C = (square_side, 0) ∧ D = (square_side, square_side)) →
  -- A' is on BC
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A' = (t * square_side, 0)) →
  -- BA' length is 1/2
  (Real.sqrt ((A'.1 - B.1)^2 + (A'.2 - B.2)^2) = BA'_length) →
  -- F is on CD
  (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (square_side, s * square_side)) →
  -- F is also on AB
  (∃ r : ℝ, 0 ≤ r ∧ r ≤ 1 ∧ F = ((1-r) * A.1 + r * B.1, (1-r) * A.2 + r * B.2)) →
  -- Conclusion: Perimeter of CFA' is (3 + √17) / 2
  let CF := Real.sqrt ((C.1 - F.1)^2 + (C.2 - F.2)^2)
  let FA' := Real.sqrt ((F.1 - A'.1)^2 + (F.2 - A'.2)^2)
  let CA' := Real.sqrt ((C.1 - A'.1)^2 + (C.2 - A'.2)^2)
  CF + FA' + CA' = (3 + Real.sqrt 17) / 2 := by
sorry

end NUMINAMATH_CALUDE_folded_square_perimeter_l2204_220450


namespace NUMINAMATH_CALUDE_max_sum_squared_sum_l2204_220402

theorem max_sum_squared_sum (a b c : ℝ) (h : a + b + c = a^2 + b^2 + c^2) :
  a + b + c ≤ 3 ∧ ∃ x y z : ℝ, x + y + z = x^2 + y^2 + z^2 ∧ x + y + z = 3 :=
sorry

end NUMINAMATH_CALUDE_max_sum_squared_sum_l2204_220402


namespace NUMINAMATH_CALUDE_circles_tangent_implies_a_value_l2204_220410

/-- Circle C1 with center (a, 0) and radius 2 -/
def C1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 4}

/-- Circle C2 with center (0, √5) and radius |a| -/
def C2 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - Real.sqrt 5)^2 = a^2}

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (C1 C2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ),
    C1 = {p : ℝ × ℝ | (p.1 - c1.1)^2 + (p.2 - c1.2)^2 = r1^2} ∧
    C2 = {p : ℝ × ℝ | (p.1 - c2.1)^2 + (p.2 - c2.2)^2 = r2^2} ∧
    (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circles_tangent_implies_a_value :
  ∀ a : ℝ, externally_tangent (C1 a) (C2 a) → a = 1/4 ∨ a = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_implies_a_value_l2204_220410


namespace NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l2204_220462

/-- Given a cubic polynomial q(x) with the following properties:
    1) It has roots at 2, -2, and 1
    2) The function f(x) = (x^3 - 2x^2 - 5x + 6) / q(x) has no horizontal asymptote
    3) q(4) = 24
    Then q(x) = (2/3)x^3 - (2/3)x^2 - (8/3)x + 8/3 -/
theorem cubic_polynomial_uniqueness (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = 2 ∨ x = -2 ∨ x = 1) →
  (∃ k, ∀ x, q x = k * (x - 2) * (x + 2) * (x - 1)) →
  q 4 = 24 →
  ∀ x, q x = (2/3) * x^3 - (2/3) * x^2 - (8/3) * x + 8/3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l2204_220462


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2204_220416

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_correct : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h1 : seq.S 6 = 8 * seq.S 3)
    (h2 : seq.a 3 - seq.a 5 = 8) :
  seq.a 20 = -74 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2204_220416


namespace NUMINAMATH_CALUDE_first_day_hike_distance_l2204_220468

/-- A hike with two participants -/
structure Hike where
  total_distance : ℕ
  distance_left : ℕ
  tripp_backpack_weight : ℕ
  charlotte_backpack_weight : ℕ
  (charlotte_lighter : charlotte_backpack_weight = tripp_backpack_weight - 7)

/-- The distance hiked on the first day -/
def distance_hiked_first_day (h : Hike) : ℕ :=
  h.total_distance - h.distance_left

theorem first_day_hike_distance (h : Hike) 
  (h_total : h.total_distance = 36) 
  (h_left : h.distance_left = 27) : 
  distance_hiked_first_day h = 9 := by
  sorry

end NUMINAMATH_CALUDE_first_day_hike_distance_l2204_220468


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l2204_220474

-- Define the conditions p and q
def p (x : ℝ) : Prop := x - 3 > 0
def q (x : ℝ) : Prop := (x - 3) * (x - 4) < 0

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ∃ x, ¬(q x) ∧ p x :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l2204_220474


namespace NUMINAMATH_CALUDE_integer_expression_l2204_220436

/-- Binomial coefficient -/
def binomial (m l : ℕ) : ℕ := Nat.choose m l

/-- The main theorem -/
theorem integer_expression (l m : ℤ) (h1 : 1 ≤ l) (h2 : l < m) :
  ∃ (k : ℤ), ((m - 3*l + 2) / (l + 2)) * binomial m.toNat l.toNat = k ↔ 
  ∃ (n : ℤ), m + 8 = n * (l + 2) := by sorry

end NUMINAMATH_CALUDE_integer_expression_l2204_220436


namespace NUMINAMATH_CALUDE_total_ridges_l2204_220470

/-- The number of ridges on a single vinyl record -/
def ridges_per_record : ℕ := 60

/-- The number of cases Jerry has -/
def num_cases : ℕ := 4

/-- The number of shelves in each case -/
def shelves_per_case : ℕ := 3

/-- The number of records each shelf can hold -/
def records_per_shelf : ℕ := 20

/-- The percentage of shelf capacity that is full, represented as a rational number -/
def shelf_fullness : ℚ := 60 / 100

/-- Theorem stating the total number of ridges on Jerry's records -/
theorem total_ridges : 
  ridges_per_record * num_cases * shelves_per_case * records_per_shelf * shelf_fullness = 8640 := by
  sorry

end NUMINAMATH_CALUDE_total_ridges_l2204_220470


namespace NUMINAMATH_CALUDE_translation_theorem_l2204_220475

/-- The original function -/
def f (x : ℝ) : ℝ := -(x - 1)^2 + 3

/-- The target function -/
def g (x : ℝ) : ℝ := -x^2

/-- The translation function -/
def translate (x : ℝ) : ℝ := x + 1

theorem translation_theorem :
  ∀ x : ℝ, f (translate x) - 3 = g x := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l2204_220475


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l2204_220437

theorem cheryl_material_usage 
  (bought : ℚ) 
  (left : ℚ) 
  (h1 : bought = 3/8 + 1/3) 
  (h2 : left = 15/40) : 
  bought - left = 1/3 := by
sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l2204_220437


namespace NUMINAMATH_CALUDE_digit_1257_of_7_19th_l2204_220487

/-- The decimal representation of 7/19 repeats every 18 digits -/
def period : ℕ := 18

/-- The repeating sequence of digits in the decimal representation of 7/19 -/
def repeating_sequence : List ℕ := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]

/-- The position we're interested in -/
def target_position : ℕ := 1257

/-- Theorem stating that the 1257th digit after the decimal point in 7/19 is 7 -/
theorem digit_1257_of_7_19th : 
  (repeating_sequence.get? ((target_position - 1) % period)) = some 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_1257_of_7_19th_l2204_220487


namespace NUMINAMATH_CALUDE_trinomial_product_degree_15_l2204_220415

def trinomial (p q : ℕ) (a : ℝ) (x : ℝ) : ℝ := x^p + a * x^q + 1

theorem trinomial_product_degree_15 :
  ∀ (p q r s : ℕ) (a b : ℝ),
    q < p → s < r → p + r = 15 →
    (∃ (t : ℕ) (c : ℝ), 
      trinomial p q a * trinomial r s b = trinomial 15 t c) ↔
    ((p = 5 ∧ q = 0 ∧ r = 10 ∧ s = 5 ∧ a = 1 ∧ b = -1) ∨
     (p = 9 ∧ q = 3 ∧ r = 6 ∧ s = 3 ∧ a = -1 ∧ b = 1) ∨
     (p = 9 ∧ q = 6 ∧ r = 6 ∧ s = 3 ∧ a = -1 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_trinomial_product_degree_15_l2204_220415


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2204_220469

/-- A line passing through a point -/
def line_passes_through (k : ℝ) (x y : ℝ) : Prop :=
  2 - k * x = -5 * y

/-- The theorem stating that the line passes through the given point when k = -0.5 -/
theorem line_passes_through_point :
  line_passes_through (-0.5) 6 (-1) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2204_220469


namespace NUMINAMATH_CALUDE_find_divisor_l2204_220400

theorem find_divisor : ∃ (d : ℕ), d > 1 ∧ 
  (3198 + 2) % d = 0 ∧ 
  3198 % d ≠ 0 ∧ 
  ∀ (k : ℕ), k > 1 → (3198 + 2) % k = 0 → 3198 % k ≠ 0 → d ≤ k :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l2204_220400


namespace NUMINAMATH_CALUDE_probability_of_target_letter_l2204_220464

def word : String := "CALCULATE"
def target_letters : String := "CUT"

def count_occurrences (c : Char) (s : String) : Nat :=
  s.toList.filter (· = c) |>.length

def favorable_outcomes : Nat :=
  target_letters.toList.map (λ c => count_occurrences c word) |>.sum

theorem probability_of_target_letter :
  (favorable_outcomes : ℚ) / word.length = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_of_target_letter_l2204_220464


namespace NUMINAMATH_CALUDE_line_bisected_by_point_m_prove_line_bisected_by_point_m_l2204_220420

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point is the midpoint of two other points -/
def Point.isMidpointOf (m : Point) (p1 p2 : Point) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- The theorem to be proved -/
theorem line_bisected_by_point_m (l1 l2 : Line) (m : Point) : Prop :=
  let desired_line : Line := { a := 1, b := 4, c := -4 }
  let point_m : Point := { x := 0, y := 1 }
  l1 = { a := 1, b := -3, c := 10 } →
  l2 = { a := 2, b := 1, c := -8 } →
  m = point_m →
  ∃ (p1 p2 : Point),
    p1.liesOn l1 ∧
    p2.liesOn l2 ∧
    m.isMidpointOf p1 p2 ∧
    p1.liesOn desired_line ∧
    p2.liesOn desired_line ∧
    m.liesOn desired_line

/-- Proof of the theorem -/
theorem prove_line_bisected_by_point_m (l1 l2 : Line) (m : Point) :
  line_bisected_by_point_m l1 l2 m := by
  sorry

end NUMINAMATH_CALUDE_line_bisected_by_point_m_prove_line_bisected_by_point_m_l2204_220420
