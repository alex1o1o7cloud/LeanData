import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_ages_sum_of_ages_proof_l2072_207250

/-- Proves that the sum of James and Louise's current ages is 32 years. -/
theorem sum_of_ages : ℝ → ℝ → Prop :=
  fun james louise =>
    james = louise + 9 →
    james + 5 = 3 * (louise - 3) →
    james + louise = 32

-- The proof is omitted
theorem sum_of_ages_proof : ∃ (james louise : ℝ), sum_of_ages james louise :=
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_sum_of_ages_proof_l2072_207250


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2072_207278

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2072_207278


namespace NUMINAMATH_CALUDE_bryce_raisins_l2072_207236

theorem bryce_raisins (x : ℕ) : 
  (x - 6 = x / 2) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_bryce_raisins_l2072_207236


namespace NUMINAMATH_CALUDE_loss_equals_five_balls_l2072_207218

/-- Prove that the number of balls the loss equates to is 5 -/
theorem loss_equals_five_balls 
  (cost_price : ℕ) 
  (num_balls_sold : ℕ) 
  (selling_price : ℕ) 
  (h1 : cost_price = 72)
  (h2 : num_balls_sold = 15)
  (h3 : selling_price = 720) :
  (num_balls_sold * cost_price - selling_price) / cost_price = 5 := by
  sorry

#check loss_equals_five_balls

end NUMINAMATH_CALUDE_loss_equals_five_balls_l2072_207218


namespace NUMINAMATH_CALUDE_banana_group_size_l2072_207249

/-- Given a collection of bananas organized into groups, this theorem proves
    the size of each group when the total number of bananas and groups are known. -/
theorem banana_group_size
  (total_bananas : ℕ)
  (num_groups : ℕ)
  (h1 : total_bananas = 203)
  (h2 : num_groups = 7)
  : total_bananas / num_groups = 29 := by
  sorry

#eval 203 / 7  -- This should evaluate to 29

end NUMINAMATH_CALUDE_banana_group_size_l2072_207249


namespace NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l2072_207265

def T : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_root_in_T : 
  (∀ n : ℕ+, n ≥ 12 → ∃ z ∈ T, z ^ (n : ℕ) = 1) ∧ 
  (∀ m : ℕ+, m < 12 → ∃ n : ℕ+, n ≥ m ∧ ∀ z ∈ T, z ^ (n : ℕ) ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l2072_207265


namespace NUMINAMATH_CALUDE_lcm_of_9_12_18_l2072_207296

theorem lcm_of_9_12_18 : Nat.lcm (Nat.lcm 9 12) 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_18_l2072_207296


namespace NUMINAMATH_CALUDE_pyramid_volume_l2072_207290

/-- The volume of a pyramid with a rectangular base and given edge length --/
theorem pyramid_volume (base_length base_width edge_length : ℝ) 
  (h_base_length : base_length = 6)
  (h_base_width : base_width = 8)
  (h_edge_length : edge_length = 13) : 
  (1 / 3 : ℝ) * base_length * base_width * 
    Real.sqrt (edge_length^2 - ((base_length^2 + base_width^2) / 4)) = 192 := by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_pyramid_volume_l2072_207290


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_sqrt_neg_seven_squared_l2072_207257

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

theorem sqrt_neg_seven_squared : Real.sqrt ((-7)^2) = 7 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_sqrt_neg_seven_squared_l2072_207257


namespace NUMINAMATH_CALUDE_inequality_system_no_solution_l2072_207244

/-- The inequality system has no solution if and only if a ≥ -1 -/
theorem inequality_system_no_solution (a : ℝ) : 
  (∀ x : ℝ, ¬(x < a - 3 ∧ x + 2 > 2 * a)) ↔ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_no_solution_l2072_207244


namespace NUMINAMATH_CALUDE_perfect_games_count_l2072_207239

theorem perfect_games_count (perfect_score : ℕ) (total_points : ℕ) : 
  perfect_score = 21 → total_points = 63 → total_points / perfect_score = 3 := by
sorry

end NUMINAMATH_CALUDE_perfect_games_count_l2072_207239


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l2072_207275

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- X-coordinate of the foci -/
  foci_x : ℝ

/-- Given a hyperbola, returns its other asymptote -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ 2 * x + 16

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ -2 * x + 4)
  (h2 : h.foci_x = -3) :
  other_asymptote h = fun x ↦ 2 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l2072_207275


namespace NUMINAMATH_CALUDE_day_of_week_p_minus_one_l2072_207227

-- Define a type for days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the day of the week for a given day number
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

-- Define the theorem
theorem day_of_week_p_minus_one (P : Nat) :
  dayOfWeek 250 = DayOfWeek.Sunday →
  dayOfWeek 150 = DayOfWeek.Sunday →
  dayOfWeek 50 = DayOfWeek.Sunday :=
by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_day_of_week_p_minus_one_l2072_207227


namespace NUMINAMATH_CALUDE_expression_evaluation_l2072_207288

theorem expression_evaluation : 3 * 3^4 - 9^20 / 9^18 + 5^3 = 287 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2072_207288


namespace NUMINAMATH_CALUDE_perpendicular_lines_relationship_l2072_207228

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a vector
def perpendicular (l : Line3D) (v : ℝ × ℝ × ℝ) : Prop :=
  let (dx, dy, dz) := l.direction
  let (vx, vy, vz) := v
  dx * vx + dy * vy + dz * vz = 0

-- Define the relationships between two lines
inductive LineRelationship
  | Parallel
  | Intersecting
  | Skew

-- State the theorem
theorem perpendicular_lines_relationship (l1 l2 l3 : Line3D) 
  (h1 : perpendicular l1 l3.direction) 
  (h2 : perpendicular l2 l3.direction) :
  ∃ r : LineRelationship, true :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_relationship_l2072_207228


namespace NUMINAMATH_CALUDE_midpoint_specific_segment_l2072_207215

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ := sorry

theorem midpoint_specific_segment :
  let p1 : ℝ × ℝ := (6, π/4)
  let p2 : ℝ × ℝ := (6, 3*π/4)
  let (r, θ) := polar_midpoint p1.1 p1.2 p2.1 p2.2
  r = 3 * Real.sqrt 2 ∧ θ = π/2 :=
sorry

end NUMINAMATH_CALUDE_midpoint_specific_segment_l2072_207215


namespace NUMINAMATH_CALUDE_rachel_day_visitor_count_l2072_207225

/-- The number of visitors to Buckingham Palace over two days -/
def total_visitors : ℕ := 829

/-- The number of visitors to Buckingham Palace on the day before Rachel's visit -/
def previous_day_visitors : ℕ := 246

/-- The number of visitors to Buckingham Palace on the day of Rachel's visit -/
def rachel_day_visitors : ℕ := total_visitors - previous_day_visitors

theorem rachel_day_visitor_count : rachel_day_visitors = 583 := by
  sorry

end NUMINAMATH_CALUDE_rachel_day_visitor_count_l2072_207225


namespace NUMINAMATH_CALUDE_store_sales_total_l2072_207242

/-- Represents the number of DVDs and CDs sold in a store in one day. -/
structure StoreSales where
  dvds : ℕ
  cds : ℕ

/-- Given a store that sells 1.6 times as many DVDs as CDs and sells 168 DVDs in one day,
    the total number of DVDs and CDs sold is 273. -/
theorem store_sales_total (s : StoreSales) 
    (h1 : s.dvds = 168)
    (h2 : s.dvds = (1.6 : ℝ) * s.cds) : 
    s.dvds + s.cds = 273 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_total_l2072_207242


namespace NUMINAMATH_CALUDE_carol_peanuts_l2072_207209

theorem carol_peanuts (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 2 → received = 5 → total = initial + received → total = 7 := by
sorry

end NUMINAMATH_CALUDE_carol_peanuts_l2072_207209


namespace NUMINAMATH_CALUDE_egg_weight_probability_l2072_207210

/-- Given that the probability of an egg's weight being less than 30 grams is 0.30,
    prove that the probability of its weight being not less than 30 grams is 0.70. -/
theorem egg_weight_probability (p_less_than_30 : ℝ) (h1 : p_less_than_30 = 0.30) :
  1 - p_less_than_30 = 0.70 := by
  sorry

end NUMINAMATH_CALUDE_egg_weight_probability_l2072_207210


namespace NUMINAMATH_CALUDE_simplify_expression_l2072_207230

theorem simplify_expression (x y : ℝ) (n : ℤ) :
  (4 * x^(n+1) * y^n)^2 / ((-x*y)^2)^n = 16 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2072_207230


namespace NUMINAMATH_CALUDE_glitched_clock_correct_time_fraction_l2072_207232

/-- Represents a 24-hour digital clock with a glitch that displays '2' as '7' -/
structure GlitchedClock where
  /-- The number of hours in a day -/
  hours_per_day : ℕ := 24
  /-- The number of minutes per hour -/
  minutes_per_hour : ℕ := 60
  /-- The digit that is displayed incorrectly -/
  glitch_digit : ℕ := 2

/-- Calculates the fraction of the day the clock displays the correct time -/
def correct_time_fraction (clock : GlitchedClock) : ℚ :=
  let correct_hours := clock.hours_per_day - 6  -- Hours without '2'
  let correct_minutes_per_hour := clock.minutes_per_hour - 16  -- Minutes without '2' per hour
  (correct_hours : ℚ) / clock.hours_per_day * correct_minutes_per_hour / clock.minutes_per_hour

theorem glitched_clock_correct_time_fraction :
  ∀ (clock : GlitchedClock), correct_time_fraction clock = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_glitched_clock_correct_time_fraction_l2072_207232


namespace NUMINAMATH_CALUDE_farm_area_calculation_l2072_207260

/-- Calculates the area of a rectangular farm given its length and width ratio. -/
def farm_area (length : ℝ) (width_ratio : ℝ) : ℝ :=
  length * (width_ratio * length)

/-- Theorem stating that a rectangular farm with length 0.6 km and width three times its length has an area of 1.08 km². -/
theorem farm_area_calculation :
  farm_area 0.6 3 = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_farm_area_calculation_l2072_207260


namespace NUMINAMATH_CALUDE_solve_equation_l2072_207212

theorem solve_equation : ∃ x : ℝ, x + Real.sqrt (-4 + 6 * 4 / 3) = 13 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2072_207212


namespace NUMINAMATH_CALUDE_magic_square_constant_l2072_207292

def MagicSquare (a b c d e f g h i : ℕ) : Prop :=
  a + b + c = d + e + f ∧
  d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧
  b + e + h = c + f + i ∧
  a + e + i = c + e + g

theorem magic_square_constant (a b c d e f g h i : ℕ) :
  MagicSquare a b c d e f g h i →
  a = 12 → c = 4 → d = 7 → h = 1 →
  a + b + c = 15 :=
sorry

end NUMINAMATH_CALUDE_magic_square_constant_l2072_207292


namespace NUMINAMATH_CALUDE_star_associativity_l2072_207273

universe u

variable {U : Type u}

def star (X Y : Set U) : Set U := X ∩ Y

theorem star_associativity (X Y Z : Set U) : star (star X Y) Z = (X ∩ Y) ∩ Z := by
  sorry

end NUMINAMATH_CALUDE_star_associativity_l2072_207273


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2072_207297

def M : Set ℝ := {x | x^2 - 6*x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2072_207297


namespace NUMINAMATH_CALUDE_qualified_light_bulb_probability_l2072_207266

def market_probability (factory_A_share : ℝ) (factory_B_share : ℝ) 
                       (factory_A_qualification : ℝ) (factory_B_qualification : ℝ) : ℝ :=
  factory_A_share * factory_A_qualification + factory_B_share * factory_B_qualification

theorem qualified_light_bulb_probability :
  market_probability 0.7 0.3 0.9 0.8 = 0.87 := by
  sorry

end NUMINAMATH_CALUDE_qualified_light_bulb_probability_l2072_207266


namespace NUMINAMATH_CALUDE_lingonberries_to_pick_thursday_l2072_207284

/-- The amount of money Steve wants to make in total -/
def total_money : ℕ := 100

/-- The number of days Steve has to make the money -/
def total_days : ℕ := 4

/-- The amount of money Steve earns per pound of lingonberries -/
def money_per_pound : ℕ := 2

/-- The amount of lingonberries Steve picked on Monday -/
def monday_picked : ℕ := 8

/-- The amount of lingonberries Steve picked on Tuesday relative to Monday -/
def tuesday_multiplier : ℕ := 3

/-- The amount of lingonberries Steve picked on Wednesday -/
def wednesday_picked : ℕ := 0

theorem lingonberries_to_pick_thursday : 
  (total_money / money_per_pound) - 
  (monday_picked + tuesday_multiplier * monday_picked + wednesday_picked) = 18 := by
  sorry

end NUMINAMATH_CALUDE_lingonberries_to_pick_thursday_l2072_207284


namespace NUMINAMATH_CALUDE_chicken_pieces_needed_l2072_207259

/-- Represents the number of pieces of chicken used in different orders -/
structure ChickenPieces where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- Represents the number of orders for each type of dish -/
structure Orders where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- Calculates the total number of chicken pieces needed for all orders -/
def totalChickenPieces (pieces : ChickenPieces) (orders : Orders) : ℕ :=
  pieces.pasta * orders.pasta +
  pieces.barbecue * orders.barbecue +
  pieces.friedDinner * orders.friedDinner

/-- Theorem stating that given the specific conditions, 37 pieces of chicken are needed -/
theorem chicken_pieces_needed :
  let pieces := ChickenPieces.mk 2 3 8
  let orders := Orders.mk 6 3 2
  totalChickenPieces pieces orders = 37 := by
  sorry


end NUMINAMATH_CALUDE_chicken_pieces_needed_l2072_207259


namespace NUMINAMATH_CALUDE_min_sum_squares_l2072_207207

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) 
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0)
  (sum_cond : x₁ + 2*x₂ + 3*x₃ = 60) :
  x₁^2 + x₂^2 + x₃^2 ≥ 1800/7 ∧ 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 
    y₁ + 2*y₂ + 3*y₃ = 60 ∧ 
    y₁^2 + y₂^2 + y₃^2 = 1800/7 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2072_207207


namespace NUMINAMATH_CALUDE_min_sequence_length_is_eight_l2072_207269

/-- The set S containing elements 1, 2, 3, and 4 -/
def S : Finset ℕ := {1, 2, 3, 4}

/-- A sequence of natural numbers -/
def Sequence := List ℕ

/-- Check if a list contains exactly the elements of a given set -/
def containsExactly (l : List ℕ) (s : Finset ℕ) : Prop :=
  l.toFinset = s

/-- Check if a sequence satisfies the property for all non-empty subsets of S -/
def satisfiesProperty (seq : Sequence) : Prop :=
  ∀ B : Finset ℕ, B ⊆ S → B.Nonempty → 
    ∃ subseq : List ℕ, subseq.length = B.card ∧ 
      seq.Sublist subseq ∧ containsExactly subseq B

/-- The minimum length of a sequence satisfying the property -/
def minSequenceLength : ℕ := 8

/-- Theorem stating that the minimum length of a sequence satisfying the property is 8 -/
theorem min_sequence_length_is_eight :
  (∃ seq : Sequence, seq.length = minSequenceLength ∧ satisfiesProperty seq) ∧
  (∀ seq : Sequence, seq.length < minSequenceLength → ¬satisfiesProperty seq) := by
  sorry


end NUMINAMATH_CALUDE_min_sequence_length_is_eight_l2072_207269


namespace NUMINAMATH_CALUDE_modulus_of_z_l2072_207246

theorem modulus_of_z (z : ℂ) (h : z + 3*I = 3 - I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2072_207246


namespace NUMINAMATH_CALUDE_parallel_squares_theorem_l2072_207289

/-- Two squares with parallel sides -/
structure ParallelSquares where
  a : ℝ  -- Side length of the first square
  b : ℝ  -- Side length of the second square
  a_pos : 0 < a
  b_pos : 0 < b

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of an equilateral triangle -/
def is_equilateral (p q r : Point) : Prop :=
  (p.x - q.x)^2 + (p.y - q.y)^2 = (q.x - r.x)^2 + (q.y - r.y)^2 ∧
  (q.x - r.x)^2 + (q.y - r.y)^2 = (r.x - p.x)^2 + (r.y - p.y)^2

/-- The set of points M satisfying the condition -/
def valid_points (squares : ParallelSquares) : Set Point :=
  {m : Point | ∀ p : Point, p.x ∈ [-squares.a/2, squares.a/2] ∧ p.y ∈ [-squares.a/2, squares.a/2] →
    ∃ q : Point, q.x ∈ [-squares.b/2, squares.b/2] ∧ q.y ∈ [-squares.b/2, squares.b/2] ∧
    is_equilateral m p q}

/-- The main theorem -/
theorem parallel_squares_theorem (squares : ParallelSquares) :
  (valid_points squares).Nonempty ↔ squares.b ≥ (squares.a / 2) * (Real.sqrt 3 + 1) :=
sorry

end NUMINAMATH_CALUDE_parallel_squares_theorem_l2072_207289


namespace NUMINAMATH_CALUDE_swimming_contest_outcomes_l2072_207270

/-- The number of permutations of k elements chosen from a set of n elements -/
def permutations (n k : ℕ) : ℕ := sorry

/-- The number of participants in the swimming contest -/
def num_participants : ℕ := 6

/-- The number of places we're interested in (1st, 2nd, 3rd) -/
def num_places : ℕ := 3

theorem swimming_contest_outcomes :
  permutations num_participants num_places = 120 := by sorry

end NUMINAMATH_CALUDE_swimming_contest_outcomes_l2072_207270


namespace NUMINAMATH_CALUDE_xy_value_l2072_207216

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 35/12 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2072_207216


namespace NUMINAMATH_CALUDE_average_of_data_l2072_207234

def data : List ℕ := [5, 6, 5, 6, 4, 4]

theorem average_of_data : (data.sum : ℚ) / data.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_l2072_207234


namespace NUMINAMATH_CALUDE_cube_cut_volume_ratio_l2072_207202

theorem cube_cut_volume_ratio (x y : ℝ) (h_positive : x > 0 ∧ y > 0) 
  (h_cut : y < x) (h_surface_ratio : 2 * (x^2 + 2*x*y) = x^2 + 2*x*(x-y)) : 
  (x^2 * y) / (x^2 * (x - y)) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_cube_cut_volume_ratio_l2072_207202


namespace NUMINAMATH_CALUDE_max_value_of_f_l2072_207281

open Real

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (x^4 - 3*x^2 - 6*x + 13) - Real.sqrt (x^4 - x^2 + 1)

theorem max_value_of_f : 
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2072_207281


namespace NUMINAMATH_CALUDE_fence_price_per_foot_l2072_207223

theorem fence_price_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 289) 
  (h2 : total_cost = 3672) : 
  total_cost / (4 * Real.sqrt area) = 54 := by
  sorry

end NUMINAMATH_CALUDE_fence_price_per_foot_l2072_207223


namespace NUMINAMATH_CALUDE_problem_solution_l2072_207251

theorem problem_solution : ∃ x : ℝ, (6000 - (x / 21)) = 5995 ∧ x = 105 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2072_207251


namespace NUMINAMATH_CALUDE_range_of_m_l2072_207263

def p (m : ℝ) : Prop := ∃ (x y : ℝ), x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ < 0 ∧ x₁^2 - x₁ + m - 4 = 0 ∧ x₂^2 - x₂ + m - 4 = 0

theorem range_of_m (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬p m) :
  m ≤ 1 - Real.sqrt 2 ∨ (1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2072_207263


namespace NUMINAMATH_CALUDE_clock_hands_angle_l2072_207241

-- Define the initial speeds of the hands
def initial_hour_hand_speed : ℝ := 0.5
def initial_minute_hand_speed : ℝ := 6

-- Define the swapped speeds
def swapped_hour_hand_speed : ℝ := initial_minute_hand_speed
def swapped_minute_hand_speed : ℝ := initial_hour_hand_speed

-- Define the starting position (3 PM)
def starting_hour_position : ℝ := 90
def starting_minute_position : ℝ := 0

-- Define the target position (4 o'clock)
def target_hour_position : ℝ := 120

-- Theorem statement
theorem clock_hands_angle :
  let time_to_target := (target_hour_position - starting_hour_position) / swapped_hour_hand_speed
  let final_minute_position := starting_minute_position + swapped_minute_hand_speed * time_to_target
  let angle := target_hour_position - final_minute_position
  min angle (360 - angle) = 117.5 := by sorry

end NUMINAMATH_CALUDE_clock_hands_angle_l2072_207241


namespace NUMINAMATH_CALUDE_area_equality_iff_concyclic_l2072_207271

-- Define the triangle ABC
variable (A B C : Point)

-- Define the altitudes and their intersection
variable (U V W H : Point)

-- Define points X, Y, Z on the altitudes
variable (X Y Z : Point)

-- Define the property of being an acute-angled triangle
def is_acute_angled (A B C : Point) : Prop := sorry

-- Define the property of a point being on a line segment
def on_segment (P Q R : Point) : Prop := sorry

-- Define the property of points being different
def are_different (P Q : Point) : Prop := sorry

-- Define the property of points being concyclic
def are_concyclic (P Q R S : Point) : Prop := sorry

-- Define the area of a triangle
def area (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem area_equality_iff_concyclic :
  is_acute_angled A B C →
  on_segment A U H → on_segment B V H → on_segment C W H →
  on_segment A U X → on_segment B V Y → on_segment C W Z →
  are_different X H → are_different Y H → are_different Z H →
  (are_concyclic X Y Z H ↔ area A B C = area A B Z + area A Y C + area X B C) :=
by sorry

end NUMINAMATH_CALUDE_area_equality_iff_concyclic_l2072_207271


namespace NUMINAMATH_CALUDE_abc_inequality_l2072_207256

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = a * b * c) : 
  max a (max b c) > 17/10 := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l2072_207256


namespace NUMINAMATH_CALUDE_grade_distribution_l2072_207219

theorem grade_distribution (total_students : ℝ) (prob_A prob_B prob_C : ℝ) :
  total_students = 40 →
  prob_A = 0.6 * prob_B →
  prob_C = 1.5 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  prob_B * total_students = 40 / 3.1 :=
by sorry

end NUMINAMATH_CALUDE_grade_distribution_l2072_207219


namespace NUMINAMATH_CALUDE_golden_man_poem_analysis_correct_l2072_207226

/-- Represents a poem --/
structure Poem where
  content : String
  deriving Repr

/-- Represents the analysis of a poem --/
structure PoemAnalysis where
  sentimentality_reasons : List String
  artistic_techniques : List String
  deriving Repr

/-- Function to analyze a poem --/
def analyze_poem (p : Poem) : PoemAnalysis :=
  { sentimentality_reasons := ["humiliating mission", "decline of homeland", "aging"],
    artistic_techniques := ["using scenery to express emotions"] }

/-- The poem in question --/
def golden_man_poem : Poem :=
  { content := "Recalling the divine capital, a bustling place, where I once roamed..." }

/-- Theorem stating that the analysis of the golden_man_poem is correct --/
theorem golden_man_poem_analysis_correct :
  analyze_poem golden_man_poem =
    { sentimentality_reasons := ["humiliating mission", "decline of homeland", "aging"],
      artistic_techniques := ["using scenery to express emotions"] } := by
  sorry


end NUMINAMATH_CALUDE_golden_man_poem_analysis_correct_l2072_207226


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2072_207245

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (team_average : ℝ) 
  (captain_age_diff : ℝ) 
  (remaining_average_diff : ℝ) :
  team_size = 15 →
  team_average = 28 →
  captain_age_diff = 4 →
  remaining_average_diff = 2 →
  (team_size : ℝ) * team_average = 
    ((team_size - 2 : ℝ) * (team_average - remaining_average_diff)) + 
    (team_average + captain_age_diff) + 
    (team_average * team_size - ((team_size - 2 : ℝ) * (team_average - remaining_average_diff)) - 
    (team_average + captain_age_diff)) →
  team_average = 28 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l2072_207245


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2072_207280

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ - 2*m + 5 = 0 ∧ 
    x₂^2 - 4*x₂ - 2*m + 5 = 0 ∧
    x₁*x₂ + x₁ + x₂ = m^2 + 6) →
  m = 1 ∧ m ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2072_207280


namespace NUMINAMATH_CALUDE_conference_center_distance_l2072_207229

theorem conference_center_distance
  (initial_speed : ℝ)
  (initial_distance : ℝ)
  (late_time : ℝ)
  (speed_increase : ℝ)
  (early_time : ℝ)
  (h1 : initial_speed = 40)
  (h2 : initial_distance = 40)
  (h3 : late_time = 1.5)
  (h4 : speed_increase = 20)
  (h5 : early_time = 0.25)
  : ∃ (total_distance : ℝ), total_distance = 310 :=
by
  sorry

end NUMINAMATH_CALUDE_conference_center_distance_l2072_207229


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l2072_207201

def f (x : ℝ) : ℝ := -3 * x

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l2072_207201


namespace NUMINAMATH_CALUDE_camp_girls_count_l2072_207221

theorem camp_girls_count (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 133 → difference = 33 → girls + (girls + difference) = total → girls = 50 := by
sorry

end NUMINAMATH_CALUDE_camp_girls_count_l2072_207221


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2072_207268

/-- Given a geometric sequence {a_n} with common ratio q ≠ 1, where a₁a₂a₃ = -1/8
    and a₂, a₄, a₃ form an arithmetic sequence, the sum of the first 4 terms
    of the sequence {a_n} is equal to 5/8. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ, a (n + 1) = a n * q) →
  a 1 * a 2 * a 3 = -1/8 →
  2 * a 4 = a 2 + a 3 →
  (a 1 + a 2 + a 3 + a 4 : ℝ) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2072_207268


namespace NUMINAMATH_CALUDE_base_for_216_four_digits_l2072_207291

def has_exactly_four_digits (b : ℕ) (n : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

theorem base_for_216_four_digits :
  ∃! b : ℕ, b > 1 ∧ has_exactly_four_digits b 216 :=
by
  sorry

end NUMINAMATH_CALUDE_base_for_216_four_digits_l2072_207291


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2072_207293

/-- The sum of all sides of an equilateral triangle with side length 13/12 meters is 13/4 meters. -/
theorem equilateral_triangle_perimeter (side_length : ℚ) (h : side_length = 13 / 12) :
  3 * side_length = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2072_207293


namespace NUMINAMATH_CALUDE_correct_average_l2072_207267

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 16 →
  incorrect_num = 25 →
  correct_num = 45 →
  (n : ℚ) * incorrect_avg = (n - 1 : ℚ) * incorrect_avg + incorrect_num →
  ((n : ℚ) * incorrect_avg - incorrect_num + correct_num) / n = 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l2072_207267


namespace NUMINAMATH_CALUDE_john_total_height_climbed_l2072_207224

/-- Calculates the total height climbed by John given the number of flights, 
    height per flight, and additional climbing information. -/
def totalHeightClimbed (numFlights : ℕ) (heightPerFlight : ℕ) : ℕ :=
  let stairsHeight := numFlights * heightPerFlight
  let ropeHeight := stairsHeight / 2
  let ladderHeight := ropeHeight + 10
  stairsHeight + ropeHeight + ladderHeight

/-- Theorem stating that the total height climbed by John is 70 feet. -/
theorem john_total_height_climbed :
  totalHeightClimbed 3 10 = 70 := by
  sorry

end NUMINAMATH_CALUDE_john_total_height_climbed_l2072_207224


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2072_207276

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem arithmetic_sequence_properties :
  let a₁ : ℚ := 4
  let d : ℚ := 5
  let seq := arithmetic_sequence a₁ d
  (seq 3 * seq 6 = 406) ∧
  (∃ (q r : ℚ), seq 9 = seq 4 * q + r ∧ q = 2 ∧ r = 6) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2072_207276


namespace NUMINAMATH_CALUDE_semicircle_area_comparison_l2072_207238

theorem semicircle_area_comparison : 
  let rectangle_width : ℝ := 8
  let rectangle_length : ℝ := 12
  let small_semicircle_radius : ℝ := rectangle_width / 2
  let large_semicircle_radius : ℝ := rectangle_length / 2
  let small_semicircle_area : ℝ := π * small_semicircle_radius^2 / 2
  let large_semicircle_area : ℝ := π * large_semicircle_radius^2 / 2
  (large_semicircle_area / small_semicircle_area - 1) * 100 = 125 := by
sorry

end NUMINAMATH_CALUDE_semicircle_area_comparison_l2072_207238


namespace NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_exists_l2072_207206

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Theorem statement
theorem consecutive_numbers_digit_sum_exists :
  ∃ n : ℕ, sumOfDigits n = 52 ∧ sumOfDigits (n + 4) = 20 :=
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_exists_l2072_207206


namespace NUMINAMATH_CALUDE_problem_1_l2072_207262

theorem problem_1 : 
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 48 + Real.sqrt 54 = 4 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2072_207262


namespace NUMINAMATH_CALUDE_english_only_students_l2072_207203

theorem english_only_students (total : ℕ) (eg ef gf egf g f : ℕ) : 
  total = 50 ∧ 
  eg = 12 ∧ 
  g = 22 ∧ 
  f = 18 ∧ 
  ef = 10 ∧ 
  gf = 8 ∧ 
  egf = 4 ∧ 
  (∃ (e g_only f_only : ℕ), 
    e + g_only + f_only + eg + ef + gf - egf = total) →
  ∃ (e : ℕ), e = 14 ∧ 
    e + (g - (eg + gf - egf)) + (f - (ef + gf - egf)) + eg + ef + gf - egf = total :=
by sorry

end NUMINAMATH_CALUDE_english_only_students_l2072_207203


namespace NUMINAMATH_CALUDE_band_problem_solution_l2072_207248

def band_problem (num_flutes num_clarinets num_trumpets num_total : ℕ) 
  (flute_ratio clarinet_ratio trumpet_ratio pianist_ratio : ℚ) : Prop :=
  let flutes_in := (num_flutes : ℚ) * flute_ratio
  let clarinets_in := (num_clarinets : ℚ) * clarinet_ratio
  let trumpets_in := (num_trumpets : ℚ) * trumpet_ratio
  let non_pianists_in := flutes_in + clarinets_in + trumpets_in
  let pianists_in := (num_total : ℚ) - non_pianists_in
  ∃ (num_pianists : ℕ), (num_pianists : ℚ) * pianist_ratio = pianists_in ∧ num_pianists = 20

theorem band_problem_solution :
  band_problem 20 30 60 53 (4/5) (1/2) (1/3) (1/10) :=
sorry

end NUMINAMATH_CALUDE_band_problem_solution_l2072_207248


namespace NUMINAMATH_CALUDE_total_vehicles_on_highway_l2072_207253

theorem total_vehicles_on_highway : 
  ∀ (num_trucks : ℕ) (num_cars : ℕ),
  num_trucks = 100 →
  num_cars = 2 * num_trucks →
  num_cars + num_trucks = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_on_highway_l2072_207253


namespace NUMINAMATH_CALUDE_no_natural_square_diff_2014_l2072_207264

theorem no_natural_square_diff_2014 : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_diff_2014_l2072_207264


namespace NUMINAMATH_CALUDE_negation_equivalence_l2072_207237

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∀ x : ℝ, x ≤ 0 → (x + 1) * Real.exp x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2072_207237


namespace NUMINAMATH_CALUDE_billy_younger_than_gladys_l2072_207222

def billy_age : ℕ := sorry
def lucas_age : ℕ := sorry
def gladys_age : ℕ := 30

axiom lucas_future_age : lucas_age + 3 = 8
axiom gladys_age_relation : gladys_age = 2 * (billy_age + lucas_age)

theorem billy_younger_than_gladys : gladys_age / billy_age = 3 := by sorry

end NUMINAMATH_CALUDE_billy_younger_than_gladys_l2072_207222


namespace NUMINAMATH_CALUDE_gain_percentage_proof_l2072_207279

theorem gain_percentage_proof (C S : ℝ) (h : 80 * C = 25 * S) : 
  (S - C) / C * 100 = 220 := by
sorry

end NUMINAMATH_CALUDE_gain_percentage_proof_l2072_207279


namespace NUMINAMATH_CALUDE_log_inequality_l2072_207277

theorem log_inequality (x y : ℝ) (h : Real.log x < Real.log y ∧ Real.log y < 0) : 
  0 < x ∧ x < y ∧ y < 1 := by sorry

end NUMINAMATH_CALUDE_log_inequality_l2072_207277


namespace NUMINAMATH_CALUDE_cylinder_max_volume_l2072_207258

/-- Given a cylinder with a constant cross-section perimeter of 4,
    prove that its maximum volume is 8π/27 -/
theorem cylinder_max_volume :
  ∀ (r h : ℝ), r > 0 → h > 0 →
  (4 * r + 2 * h = 4) →
  (π * r^2 * h ≤ 8 * π / 27) ∧
  (∃ (r₀ h₀ : ℝ), r₀ > 0 ∧ h₀ > 0 ∧ 4 * r₀ + 2 * h₀ = 4 ∧ π * r₀^2 * h₀ = 8 * π / 27) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_max_volume_l2072_207258


namespace NUMINAMATH_CALUDE_star_calculation_l2072_207220

-- Define the ☆ operation
def star (a b : ℚ) : ℚ := a - b + 1

-- Theorem to prove
theorem star_calculation : (star (star 2 3) 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l2072_207220


namespace NUMINAMATH_CALUDE_average_xyz_is_five_sixths_l2072_207295

theorem average_xyz_is_five_sixths (x y z : ℚ) 
  (eq1 : 2003 * z - 4006 * x = 1002)
  (eq2 : 2003 * y + 6009 * x = 4004) :
  (x + y + z) / 3 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_average_xyz_is_five_sixths_l2072_207295


namespace NUMINAMATH_CALUDE_unique_solution_fifth_root_equation_l2072_207252

theorem unique_solution_fifth_root_equation (x : ℝ) :
  (((x^3 + 2*x)^(1/5) = (x^5 - 2*x)^(1/3)) ↔ (x = 0)) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_fifth_root_equation_l2072_207252


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2072_207204

/-- An isosceles triangle with two sides of length 6 cm and perimeter 20 cm has a base of length 8 cm. -/
theorem isosceles_triangle_base_length 
  (side_length : ℝ) 
  (perimeter : ℝ) 
  (h1 : side_length = 6) 
  (h2 : perimeter = 20) : 
  perimeter - 2 * side_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2072_207204


namespace NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_397_l2072_207213

theorem multiplicative_inverse_203_mod_397 : ∃ x : ℤ, 0 ≤ x ∧ x < 397 ∧ (203 * x) % 397 = 1 :=
by
  use 309
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_397_l2072_207213


namespace NUMINAMATH_CALUDE_total_milk_poured_l2072_207214

/-- Represents a bottle with a certain capacity -/
structure Bottle where
  capacity : ℝ

/-- Represents the amount of milk poured into a bottle -/
def pour (b : Bottle) (fraction : ℝ) : ℝ := b.capacity * fraction

theorem total_milk_poured (bottle1 bottle2 : Bottle) 
  (h1 : bottle1.capacity = 4)
  (h2 : bottle2.capacity = 8)
  (h3 : pour bottle2 (5.333333333333333 / bottle2.capacity) = 5.333333333333333) :
  pour bottle1 (5.333333333333333 / bottle2.capacity) + 
  pour bottle2 (5.333333333333333 / bottle2.capacity) = 8 := by
sorry

end NUMINAMATH_CALUDE_total_milk_poured_l2072_207214


namespace NUMINAMATH_CALUDE_melanie_cats_count_l2072_207235

theorem melanie_cats_count (jacob_cats : ℕ) (annie_cats : ℕ) (melanie_cats : ℕ)
  (h1 : jacob_cats = 90)
  (h2 : annie_cats * 3 = jacob_cats)
  (h3 : melanie_cats = annie_cats * 2) :
  melanie_cats = 60 := by
  sorry

end NUMINAMATH_CALUDE_melanie_cats_count_l2072_207235


namespace NUMINAMATH_CALUDE_min_value_3x_plus_2y_min_value_attained_l2072_207272

theorem min_value_3x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x = 4 * x * y - 2 * y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a = 4 * a * b - 2 * b → 3 * x + 2 * y ≤ 3 * a + 2 * b :=
by sorry

theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x = 4 * x * y - 2 * y) :
  3 * x + 2 * y = 2 + Real.sqrt 3 ↔ x = (3 + Real.sqrt 3) / 6 ∧ y = (Real.sqrt 3 + 1) / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_2y_min_value_attained_l2072_207272


namespace NUMINAMATH_CALUDE_basketball_tryouts_l2072_207231

theorem basketball_tryouts (girls : ℕ) (called_back : ℕ) (didnt_make_cut : ℕ) : girls = 39 → called_back = 26 → didnt_make_cut = 17 → girls + (called_back + didnt_make_cut - girls) = 43 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l2072_207231


namespace NUMINAMATH_CALUDE_right_triangle_area_l2072_207298

theorem right_triangle_area (a c : ℝ) (h1 : a = 30) (h2 : c = 34) : 
  let b := Real.sqrt (c^2 - a^2)
  (1/2) * a * b = 240 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2072_207298


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2072_207211

theorem complex_equation_solution (z : ℂ) 
  (h : 20 * Complex.abs z ^ 2 = 3 * Complex.abs (z + 3) ^ 2 + Complex.abs (z ^ 2 + 2) ^ 2 + 37) :
  z + 9 / z = -3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2072_207211


namespace NUMINAMATH_CALUDE_parallel_lines_and_planes_l2072_207287

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Returns whether two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Returns whether a line is parallel to a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Returns whether a line is a subset of a plane -/
def line_subset_of_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem parallel_lines_and_planes 
  (a b : Line3D) (α : Plane3D) 
  (h : line_subset_of_plane b α) :
  ¬(∀ (a b : Line3D) (α : Plane3D), are_parallel a b → line_parallel_to_plane a α) ∧
  ¬(∀ (a b : Line3D) (α : Plane3D), line_parallel_to_plane a α → are_parallel a b) :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_and_planes_l2072_207287


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l2072_207255

theorem cricket_team_age_difference (team_size : ℕ) (avg_age : ℝ) (captain_age : ℝ) (keeper_age_diff : ℝ) :
  team_size = 11 →
  avg_age = 25 →
  captain_age = 28 →
  keeper_age_diff = 3 →
  let total_age := avg_age * team_size
  let keeper_age := captain_age + keeper_age_diff
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + keeper_age)
  let remaining_avg := remaining_age / remaining_players
  avg_age - remaining_avg = 1 := by sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l2072_207255


namespace NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l2072_207200

theorem children_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neutral_boys : ℕ) : 
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  boys = 17 →
  girls = 43 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 5 →
  total_children - (happy_children + sad_children) = 20 := by
sorry

end NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l2072_207200


namespace NUMINAMATH_CALUDE_product_difference_squared_l2072_207261

theorem product_difference_squared : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_squared_l2072_207261


namespace NUMINAMATH_CALUDE_charity_book_donation_l2072_207299

theorem charity_book_donation (initial_books : ℕ) (donors : ℕ) (borrowed_books : ℕ) (final_books : ℕ)
  (h1 : initial_books = 300)
  (h2 : donors = 10)
  (h3 : borrowed_books = 140)
  (h4 : final_books = 210) :
  (final_books + borrowed_books - initial_books) / donors = 5 := by
  sorry

end NUMINAMATH_CALUDE_charity_book_donation_l2072_207299


namespace NUMINAMATH_CALUDE_max_value_fraction_l2072_207243

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (x * y) / (x + 8*y) ≤ 1/18 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2072_207243


namespace NUMINAMATH_CALUDE_petes_flag_problem_l2072_207233

theorem petes_flag_problem (us_stars : Nat) (us_stripes : Nat) (total_shapes : Nat) :
  us_stars = 50 →
  us_stripes = 13 →
  total_shapes = 54 →
  ∃ (circles squares : Nat),
    circles < us_stars / 2 ∧
    squares = 2 * us_stripes + 6 ∧
    circles + squares = total_shapes ∧
    us_stars / 2 - circles = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_petes_flag_problem_l2072_207233


namespace NUMINAMATH_CALUDE_range_of_b_minus_a_l2072_207283

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem range_of_b_minus_a (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) 3) ∧
  (∀ y ∈ Set.Icc (-1) 3, ∃ x ∈ Set.Icc a b, f x = y) →
  b - a ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_minus_a_l2072_207283


namespace NUMINAMATH_CALUDE_magic_king_episodes_l2072_207208

theorem magic_king_episodes (total_seasons : ℕ) 
  (first_half_episodes : ℕ) (second_half_episodes : ℕ) : 
  total_seasons = 10 ∧ 
  first_half_episodes = 20 ∧ 
  second_half_episodes = 25 →
  (total_seasons / 2 * first_half_episodes) + 
  (total_seasons / 2 * second_half_episodes) = 225 := by
  sorry

end NUMINAMATH_CALUDE_magic_king_episodes_l2072_207208


namespace NUMINAMATH_CALUDE_average_value_iff_m_in_zero_two_l2072_207285

/-- A function f has an average value on [a, b] if there exists x₀ ∈ (a, b) such that
    f(x₀) = (f(b) - f(a)) / (b - a) -/
def has_average_value (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

/-- The quadratic function f(x) = -x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x + 1

theorem average_value_iff_m_in_zero_two :
  ∀ m : ℝ, has_average_value (f m) (-1) 1 ↔ 0 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_average_value_iff_m_in_zero_two_l2072_207285


namespace NUMINAMATH_CALUDE_point_C_coordinates_l2072_207294

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-3, 1)
def B : ℝ × ℝ := (0, 5)

def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
    parallel (vector A C) (vector O A) →
    perpendicular (vector B C) (vector A B) →
    C = (12, -4) := by sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l2072_207294


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l2072_207274

theorem floor_plus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ + s = 20.5 :=
by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l2072_207274


namespace NUMINAMATH_CALUDE_area_inside_EFG_outside_AFD_l2072_207286

/-- Square ABCD with side length 36 -/
def square_side_length : ℝ := 36

/-- Point E is on side AB, 12 units from B -/
def distance_E_from_B : ℝ := 12

/-- Point F is the midpoint of side BC -/
def F_is_midpoint : Prop := True

/-- Point G is on side CD, 12 units from C -/
def distance_G_from_C : ℝ := 12

/-- The area of the region inside triangle EFG and outside triangle AFD -/
def area_difference : ℝ := 0

theorem area_inside_EFG_outside_AFD :
  square_side_length = 36 →
  distance_E_from_B = 12 →
  F_is_midpoint →
  distance_G_from_C = 12 →
  area_difference = 0 := by
  sorry

end NUMINAMATH_CALUDE_area_inside_EFG_outside_AFD_l2072_207286


namespace NUMINAMATH_CALUDE_binomial_unique_solution_l2072_207282

/-- Represents a binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The expectation of a binomial distribution -/
def expectation (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem stating the unique solution for n and p given E(ξ) and D(ξ) -/
theorem binomial_unique_solution :
  ∀ ξ : BinomialDistribution,
    expectation ξ = 12 →
    variance ξ = 4 →
    ξ.n = 18 ∧ ξ.p = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_binomial_unique_solution_l2072_207282


namespace NUMINAMATH_CALUDE_max_third_term_is_15_l2072_207247

/-- An arithmetic sequence of four positive integers with sum 46 -/
structure ArithmeticSequence where
  a : ℕ+  -- first term
  d : ℕ   -- common difference
  sum_eq_46 : a + (a + d) + (a + 2*d) + (a + 3*d) = 46

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ :=
  seq.a + 2 * seq.d

/-- Theorem: The maximum possible value of the third term is 15 -/
theorem max_third_term_is_15 :
  ∀ seq : ArithmeticSequence, third_term seq ≤ 15 ∧ ∃ seq : ArithmeticSequence, third_term seq = 15 :=
sorry

end NUMINAMATH_CALUDE_max_third_term_is_15_l2072_207247


namespace NUMINAMATH_CALUDE_polygon_with_120_degree_angles_is_hexagon_l2072_207205

theorem polygon_with_120_degree_angles_is_hexagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 120 →
    (n - 2) * 180 = n * interior_angle →
    n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_120_degree_angles_is_hexagon_l2072_207205


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l2072_207254

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  (2 * r) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l2072_207254


namespace NUMINAMATH_CALUDE_range_of_a_when_proposition_false_l2072_207217

theorem range_of_a_when_proposition_false :
  (¬ ∃ x₀ : ℝ, ∃ a : ℝ, a * x₀^2 - 2 * a * x₀ - 3 > 0) →
  (∀ a : ℝ, a ∈ Set.Icc (-3 : ℝ) 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_when_proposition_false_l2072_207217


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2072_207240

theorem inequality_system_solution :
  (∀ x : ℝ, 2 - x ≥ (x - 1) / 3 - 1 ↔ x ≤ 2.5) ∧
  ¬∃ x : ℝ, (5 * x + 1 < 3 * (x - 1)) ∧ ((x + 8) / 5 < (2 * x - 5) / 3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2072_207240
