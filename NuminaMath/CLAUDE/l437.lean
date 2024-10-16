import Mathlib

namespace NUMINAMATH_CALUDE_puppies_left_l437_43767

def initial_puppies : ℕ := 12
def puppies_given_away : ℕ := 7

theorem puppies_left : initial_puppies - puppies_given_away = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_left_l437_43767


namespace NUMINAMATH_CALUDE_power_sum_and_quotient_l437_43740

theorem power_sum_and_quotient : 1^345 + 5^7 / 5^5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_and_quotient_l437_43740


namespace NUMINAMATH_CALUDE_polynomial_transformation_l437_43770

theorem polynomial_transformation (x y : ℝ) : 
  x^3 - 6*x^2 + 11*x - 6 = 0 → 
  y = x + 1/x → 
  x^2*(y^2 + y - 6) = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l437_43770


namespace NUMINAMATH_CALUDE_jason_initial_cards_l437_43768

/-- The number of Pokemon cards Jason gave away. -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left. -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had. -/
def initial_cards : ℕ := cards_given_away + cards_left

/-- Theorem stating that Jason initially had 13 Pokemon cards. -/
theorem jason_initial_cards : initial_cards = 13 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l437_43768


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l437_43778

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 60 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l437_43778


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l437_43701

theorem necessary_but_not_sufficient_condition :
  ∀ x : ℝ, (2 * x^2 - 5 * x - 3 ≥ 0) → (x < 0 ∨ x > 2) ∧
  ∃ y : ℝ, (y < 0 ∨ y > 2) ∧ ¬(2 * y^2 - 5 * y - 3 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l437_43701


namespace NUMINAMATH_CALUDE_apartment_utilities_cost_l437_43791

/-- Proves that the utilities cost for an apartment is $114 given specific conditions -/
theorem apartment_utilities_cost 
  (rent : ℕ) 
  (groceries : ℕ) 
  (one_roommate_payment : ℕ) 
  (h_rent : rent = 1100)
  (h_groceries : groceries = 300)
  (h_one_roommate : one_roommate_payment = 757)
  (h_equal_split : ∀ total_cost, one_roommate_payment * 2 = total_cost) :
  ∃ utilities : ℕ, utilities = 114 ∧ rent + utilities + groceries = one_roommate_payment * 2 :=
by sorry

end NUMINAMATH_CALUDE_apartment_utilities_cost_l437_43791


namespace NUMINAMATH_CALUDE_fraction_equality_l437_43752

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 2 / 3) :
  t / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l437_43752


namespace NUMINAMATH_CALUDE_bus_departure_interval_l437_43746

/-- Represents the scenario of Xiao Wang and the No. 18 buses -/
structure BusScenario where
  /-- Speed of Xiao Wang in meters per minute -/
  wang_speed : ℝ
  /-- Speed of the No. 18 buses in meters per minute -/
  bus_speed : ℝ
  /-- Distance between two adjacent buses traveling in the same direction in meters -/
  bus_distance : ℝ
  /-- Xiao Wang walks at a constant speed -/
  wang_constant_speed : wang_speed > 0
  /-- Buses travel at a constant speed -/
  bus_constant_speed : bus_speed > 0
  /-- A bus passes Xiao Wang from behind every 6 minutes -/
  overtake_condition : 6 * bus_speed - 6 * wang_speed = bus_distance
  /-- A bus comes towards Xiao Wang every 3 minutes -/
  approach_condition : 3 * bus_speed + 3 * wang_speed = bus_distance

/-- The interval between bus departures is 4 minutes -/
theorem bus_departure_interval (scenario : BusScenario) : 
  scenario.bus_distance = 4 * scenario.bus_speed := by
  sorry

#check bus_departure_interval

end NUMINAMATH_CALUDE_bus_departure_interval_l437_43746


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l437_43710

theorem sqrt_inequality_solution_set (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 3 ∧ y < 2) ↔ x ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l437_43710


namespace NUMINAMATH_CALUDE_grain_spilled_calculation_l437_43755

def original_grain : ℕ := 50870
def remaining_grain : ℕ := 918

theorem grain_spilled_calculation : original_grain - remaining_grain = 49952 := by
  sorry

end NUMINAMATH_CALUDE_grain_spilled_calculation_l437_43755


namespace NUMINAMATH_CALUDE_right_triangle_and_modular_inverse_l437_43794

theorem right_triangle_and_modular_inverse :
  -- Define the sides of the triangle
  let a : ℕ := 15
  let b : ℕ := 112
  let c : ℕ := 113
  -- Define the modulus
  let m : ℕ := 2799
  -- Define the number we're finding the inverse for
  let x : ℕ := 225
  -- Condition: a, b, c form a right triangle
  (a^2 + b^2 = c^2) →
  -- Conclusion: 1 is the multiplicative inverse of x modulo m
  (1 * x) % m = 1 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_and_modular_inverse_l437_43794


namespace NUMINAMATH_CALUDE_payment_for_C_is_250_l437_43785

/-- Calculates the payment for worker C given the work rates and total payment --/
def calculate_payment_C (a_rate : ℚ) (b_rate : ℚ) (total_rate : ℚ) (total_payment : ℚ) : ℚ :=
  let c_rate := total_rate - (a_rate + b_rate)
  c_rate * total_payment

/-- Theorem stating that C should be paid 250 given the problem conditions --/
theorem payment_for_C_is_250 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (total_rate : ℚ) 
  (total_payment : ℚ) 
  (h1 : a_rate = 1/6) 
  (h2 : b_rate = 1/8) 
  (h3 : total_rate = 1/3) 
  (h4 : total_payment = 6000) : 
  calculate_payment_C a_rate b_rate total_rate total_payment = 250 := by
  sorry

#eval calculate_payment_C (1/6) (1/8) (1/3) 6000

end NUMINAMATH_CALUDE_payment_for_C_is_250_l437_43785


namespace NUMINAMATH_CALUDE_line_parallel_or_in_plane_l437_43784

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Main theorem -/
theorem line_parallel_or_in_plane (a b : Line3D) (α : Plane3D) 
  (h1 : parallel_lines a b) (h2 : parallel_line_plane a α) :
  parallel_line_plane b α ∨ line_in_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_or_in_plane_l437_43784


namespace NUMINAMATH_CALUDE_symmetric_function_sum_zero_l437_43787

theorem symmetric_function_sum_zero 
  (v : ℝ → ℝ) 
  (h : ∀ x ∈ Set.Icc (-1.75) 1.75, v (-x) = -v x) : 
  v (-1.75) + v (-0.5) + v 0.5 + v 1.75 = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_sum_zero_l437_43787


namespace NUMINAMATH_CALUDE_equivalent_angle_exists_l437_43777

-- Define the angle in degrees
def angle : ℝ := -463

-- Theorem stating that there exists an equivalent angle in the form k·360° + 257°
theorem equivalent_angle_exists :
  ∃ (k : ℤ), (k : ℝ) * 360 + 257 = angle + 360 * ⌊angle / 360⌋ := by
  sorry

end NUMINAMATH_CALUDE_equivalent_angle_exists_l437_43777


namespace NUMINAMATH_CALUDE_probability_not_snow_l437_43744

theorem probability_not_snow (p : ℚ) (h : p = 2 / 5) : 1 - p = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snow_l437_43744


namespace NUMINAMATH_CALUDE_function_identity_l437_43760

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_identity (x : ℝ) : f (x + 1) = x^2 - 2*x ↔ f x = x^2 - 4*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l437_43760


namespace NUMINAMATH_CALUDE_seventh_flip_probability_l437_43774

/-- A fair coin is a coin where the probability of getting heads is 1/2. -/
def fair_coin (p : ℝ → ℝ) : Prop := p 1 = 1/2

/-- A sequence of coin flips is independent if the probability of any outcome
    is not affected by the previous flips. -/
def independent_flips (p : ℕ → ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∀ x : ℝ, p n x = p 0 x

/-- The probability of getting heads on the seventh flip of a fair coin is 1/2,
    regardless of the outcomes of the previous six flips. -/
theorem seventh_flip_probability (p : ℕ → ℝ → ℝ) :
  fair_coin (p 0) →
  independent_flips p →
  p 6 1 = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_seventh_flip_probability_l437_43774


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l437_43705

theorem rectangular_prism_diagonal (width length height : ℝ) 
  (hw : width = 12) (hl : length = 16) (hh : height = 9) : 
  Real.sqrt (width^2 + length^2 + height^2) = Real.sqrt 481 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l437_43705


namespace NUMINAMATH_CALUDE_mutually_exclusive_necessary_not_sufficient_l437_43789

open Set

universe u

variable {Ω : Type u} [MeasurableSpace Ω]
variable (A₁ A₂ : Set Ω)

def mutually_exclusive (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅

def complementary (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅ ∧ A₁ ∪ A₂ = univ

theorem mutually_exclusive_necessary_not_sufficient :
  (complementary A₁ A₂ → mutually_exclusive A₁ A₂) ∧
  ¬(mutually_exclusive A₁ A₂ → complementary A₁ A₂) := by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_necessary_not_sufficient_l437_43789


namespace NUMINAMATH_CALUDE_shooting_probabilities_l437_43729

/-- Represents the probability of hitting a specific ring in a shooting event -/
structure ShootingProbability where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ

/-- Calculates the probability of hitting either the 10-ring or the 9-ring -/
def prob_10_or_9 (p : ShootingProbability) : ℝ :=
  p.ring10 + p.ring9

/-- Calculates the probability of hitting below the 8-ring -/
def prob_below_8 (p : ShootingProbability) : ℝ :=
  1 - (p.ring10 + p.ring9 + p.ring8)

/-- Theorem stating the probabilities for a given shooting event -/
theorem shooting_probabilities (p : ShootingProbability)
  (h1 : p.ring10 = 0.24)
  (h2 : p.ring9 = 0.28)
  (h3 : p.ring8 = 0.19) :
  prob_10_or_9 p = 0.52 ∧ prob_below_8 p = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l437_43729


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l437_43743

/-- Given two quadratic equations with a specific relationship between their roots, 
    prove that the ratio of certain coefficients is 3. -/
theorem quadratic_root_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ (r₁ r₂ : ℝ), (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧
                  (3 * r₁ + 3 * r₂ = -m ∧ 9 * r₁ * r₂ = n)) →
  n / p = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l437_43743


namespace NUMINAMATH_CALUDE_danielle_popsicle_sticks_l437_43711

/-- Calculates the number of remaining popsicle sticks after making popsicles -/
def remaining_popsicle_sticks (total_money : ℕ) (mold_cost : ℕ) (stick_pack_cost : ℕ) 
  (stick_pack_size : ℕ) (juice_cost : ℕ) (popsicles_per_juice : ℕ) : ℕ :=
  let remaining_money := total_money - mold_cost - stick_pack_cost
  let juice_bottles := remaining_money / juice_cost
  let popsicles_made := juice_bottles * popsicles_per_juice
  stick_pack_size - popsicles_made

/-- Proves that Danielle will be left with 40 popsicle sticks -/
theorem danielle_popsicle_sticks : 
  remaining_popsicle_sticks 10 3 1 100 2 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_danielle_popsicle_sticks_l437_43711


namespace NUMINAMATH_CALUDE_no_base_for_630_four_digits_odd_final_l437_43706

theorem no_base_for_630_four_digits_odd_final : ¬ ∃ b : ℕ, 
  2 ≤ b ∧ 
  b^3 ≤ 630 ∧ 
  630 < b^4 ∧ 
  (630 % b) % 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_no_base_for_630_four_digits_odd_final_l437_43706


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l437_43718

/-- A geometric sequence. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ) (a₁ : ℝ), ∀ n, a n = a₁ * q ^ (n - 1)

/-- The theorem stating the properties of the specific geometric sequence. -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_diff1 : a 5 - a 1 = 15) 
    (h_diff2 : a 4 - a 2 = 6) : 
    a 3 = 4 ∨ a 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l437_43718


namespace NUMINAMATH_CALUDE_cost_per_person_l437_43712

def total_cost : ℚ := 13500
def num_friends : ℕ := 15

theorem cost_per_person :
  total_cost / num_friends = 900 :=
sorry

end NUMINAMATH_CALUDE_cost_per_person_l437_43712


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l437_43728

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem for the range of a when B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l437_43728


namespace NUMINAMATH_CALUDE_combined_efficiency_l437_43735

-- Define the variables
def ray_efficiency : ℚ := 50
def tom_efficiency : ℚ := 10
def ray_distance : ℚ := 50
def tom_distance : ℚ := 100

-- Define the theorem
theorem combined_efficiency :
  let total_distance := ray_distance + tom_distance
  let ray_fuel := ray_distance / ray_efficiency
  let tom_fuel := tom_distance / tom_efficiency
  let total_fuel := ray_fuel + tom_fuel
  total_distance / total_fuel = 150 / 11 :=
by sorry

end NUMINAMATH_CALUDE_combined_efficiency_l437_43735


namespace NUMINAMATH_CALUDE_six_throws_total_skips_l437_43783

def stone_skips (n : ℕ) : ℕ := n^2 + n

def total_skips (num_throws : ℕ) : ℕ :=
  (List.range num_throws).map stone_skips |>.sum

theorem six_throws_total_skips :
  total_skips 5 + 2 * stone_skips 6 = 154 := by
  sorry

end NUMINAMATH_CALUDE_six_throws_total_skips_l437_43783


namespace NUMINAMATH_CALUDE_hash_difference_l437_43798

-- Define the # operation
def hash (x y : ℝ) : ℝ := x * y - 3 * x

-- State the theorem
theorem hash_difference : hash 8 3 - hash 3 8 = -15 := by sorry

end NUMINAMATH_CALUDE_hash_difference_l437_43798


namespace NUMINAMATH_CALUDE_barney_towel_usage_l437_43731

/-- The number of towels Barney owns -/
def total_towels : ℕ := 18

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of days Barney will not have clean towels -/
def days_without_clean_towels : ℕ := 5

/-- The number of towels Barney uses at a time -/
def towels_per_use : ℕ := 2

theorem barney_towel_usage :
  ∃ (x : ℕ),
    x = towels_per_use ∧
    total_towels - days_per_week * x = (days_per_week - days_without_clean_towels) * x :=
by sorry

end NUMINAMATH_CALUDE_barney_towel_usage_l437_43731


namespace NUMINAMATH_CALUDE_unique_items_count_l437_43724

/-- Represents a Beatles fan's collection --/
structure BeatlesFan where
  albums : ℕ
  memorabilia : ℕ

/-- Given the information about Andrew and John's collections, prove that the number of items
    in either Andrew's or John's collection or memorabilia, but not both, is 24. --/
theorem unique_items_count (andrew john : BeatlesFan) 
  (h1 : andrew.albums = 23)
  (h2 : andrew.memorabilia = 5)
  (h3 : john.albums = andrew.albums - 12 + 8) : 
  (andrew.albums - 12) + (john.albums - (andrew.albums - 12)) + andrew.memorabilia = 24 := by
  sorry

end NUMINAMATH_CALUDE_unique_items_count_l437_43724


namespace NUMINAMATH_CALUDE_total_red_pencils_l437_43723

/-- The number of packs of colored pencils Johnny bought -/
def total_packs : ℕ := 35

/-- The number of packs with 3 extra red pencils -/
def packs_with_3_extra_red : ℕ := 7

/-- The number of packs with 2 extra blue pencils and 1 extra red pencil -/
def packs_with_2_extra_blue_1_extra_red : ℕ := 4

/-- The number of packs with 1 extra green pencil and 2 extra red pencils -/
def packs_with_1_extra_green_2_extra_red : ℕ := 10

/-- The number of red pencils in each pack without extra pencils -/
def red_pencils_per_pack : ℕ := 1

/-- Theorem: The total number of red pencils Johnny bought is 59 -/
theorem total_red_pencils : 
  total_packs * red_pencils_per_pack + 
  packs_with_3_extra_red * 3 + 
  packs_with_2_extra_blue_1_extra_red * 1 + 
  packs_with_1_extra_green_2_extra_red * 2 = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_red_pencils_l437_43723


namespace NUMINAMATH_CALUDE_polyhedron_ball_covering_inequality_l437_43713

/-- A non-degenerate polyhedron -/
structure Polyhedron where
  nondegenerate : Bool

/-- A collection of balls covering a polyhedron -/
structure BallCovering (P : Polyhedron) where
  n : ℕ
  V : ℝ
  covers_surface : Bool

/-- Theorem: For any non-degenerate polyhedron, there exists a positive constant
    such that any ball covering satisfies the given inequality -/
theorem polyhedron_ball_covering_inequality (P : Polyhedron) 
    (h : P.nondegenerate = true) :
    ∃ c : ℝ, c > 0 ∧ 
    ∀ (B : BallCovering P), B.covers_surface → B.n > c / (B.V ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_ball_covering_inequality_l437_43713


namespace NUMINAMATH_CALUDE_total_tickets_sold_l437_43772

theorem total_tickets_sold (student_price general_price total_amount general_tickets : ℕ)
  (h1 : student_price = 4)
  (h2 : general_price = 6)
  (h3 : total_amount = 2876)
  (h4 : general_tickets = 388)
  (h5 : ∃ student_tickets : ℕ, student_price * student_tickets + general_price * general_tickets = total_amount) :
  ∃ total_tickets : ℕ, total_tickets = general_tickets + (total_amount - general_price * general_tickets) / student_price ∧ total_tickets = 525 :=
by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l437_43772


namespace NUMINAMATH_CALUDE_tiles_count_l437_43709

/-- Represents a square floor tiled with white tiles on the perimeter and black tiles in the center -/
structure TiledSquare where
  side_length : ℕ
  white_tiles : ℕ
  black_tiles : ℕ

/-- The number of white tiles on the perimeter of a square floor -/
def perimeter_tiles (s : TiledSquare) : ℕ := 4 * (s.side_length - 1)

/-- The number of black tiles in the center of a square floor -/
def center_tiles (s : TiledSquare) : ℕ := (s.side_length - 2)^2

/-- Theorem stating that if there are 80 white tiles on the perimeter, there are 361 black tiles in the center -/
theorem tiles_count (s : TiledSquare) :
  perimeter_tiles s = 80 → center_tiles s = 361 := by
  sorry

end NUMINAMATH_CALUDE_tiles_count_l437_43709


namespace NUMINAMATH_CALUDE_min_distinct_lines_for_31_segments_l437_43702

/-- Represents a non-self-intersecting open polyline on a plane -/
structure OpenPolyline where
  segments : ℕ
  non_self_intersecting : Bool
  consecutive_segments_not_collinear : Bool

/-- The minimum number of distinct lines needed to contain all segments of the polyline -/
def min_distinct_lines (p : OpenPolyline) : ℕ :=
  sorry

/-- Theorem stating the minimum number of distinct lines for a specific polyline -/
theorem min_distinct_lines_for_31_segments (p : OpenPolyline) :
  p.segments = 31 ∧ p.non_self_intersecting ∧ p.consecutive_segments_not_collinear →
  min_distinct_lines p = 9 :=
sorry

end NUMINAMATH_CALUDE_min_distinct_lines_for_31_segments_l437_43702


namespace NUMINAMATH_CALUDE_isosceles_triangle_smallest_angle_measure_l437_43793

theorem isosceles_triangle_smallest_angle_measure :
  ∀ (a b c : ℝ),
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Isosceles triangle condition
  c = 90 + 0.4 * 90  -- One angle is 40% larger than a right angle
  →
  a = 27 :=          -- One of the two smallest angles is 27°
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_smallest_angle_measure_l437_43793


namespace NUMINAMATH_CALUDE_equation_solution_l437_43739

theorem equation_solution (x : ℝ) : 
  (5 * x^2 - 3) / (x + 3) - 5 / (x + 3) = 6 / (x + 3) → 
  x = Real.sqrt 70 / 5 ∨ x = -Real.sqrt 70 / 5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l437_43739


namespace NUMINAMATH_CALUDE_tangent_line_at_1_l437_43786

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * (x - 2) + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Theorem statement
theorem tangent_line_at_1 : 
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m*x + b ↔ x - y + 1 = 0) ∧ 
  (m = f' 1) ∧ 
  (f 1 = m*1 + b) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_1_l437_43786


namespace NUMINAMATH_CALUDE_system_solution_l437_43759

theorem system_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + y + x*y = 8)
  (eq2 : y + z + y*z = 15)
  (eq3 : z + x + z*x = 35) :
  x + y + z + x*y = 15 := by sorry

end NUMINAMATH_CALUDE_system_solution_l437_43759


namespace NUMINAMATH_CALUDE_determine_fifth_subject_marks_l437_43769

/-- Given the marks of a student in 4 subjects and the average marks of 5 subjects,
    this theorem proves that the marks in the fifth subject can be uniquely determined. -/
theorem determine_fifth_subject_marks
  (english : ℕ)
  (mathematics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℚ)
  (h1 : english = 70)
  (h2 : mathematics = 63)
  (h3 : chemistry = 63)
  (h4 : biology = 65)
  (h5 : average = 68.2)
  : ∃! physics : ℕ,
    (english + mathematics + physics + chemistry + biology : ℚ) / 5 = average :=
by sorry

end NUMINAMATH_CALUDE_determine_fifth_subject_marks_l437_43769


namespace NUMINAMATH_CALUDE_car_distance_proof_l437_43776

theorem car_distance_proof (speed1 speed2 speed3 : ℝ) 
  (h1 : speed1 = 180)
  (h2 : speed2 = 160)
  (h3 : speed3 = 220) :
  speed1 + speed2 + speed3 = 560 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l437_43776


namespace NUMINAMATH_CALUDE_prob_different_colors_for_given_box_l437_43734

/-- A box containing balls of two colors -/
structure Box where
  small_balls : ℕ
  black_balls : ℕ

/-- The probability of drawing two balls of different colors -/
def prob_different_colors (b : Box) : ℚ :=
  let total_balls := b.small_balls + b.black_balls
  let different_color_combinations := b.small_balls * b.black_balls
  let total_combinations := (total_balls * (total_balls - 1)) / 2
  different_color_combinations / total_combinations

/-- The theorem stating the probability of drawing two balls of different colors -/
theorem prob_different_colors_for_given_box :
  prob_different_colors { small_balls := 3, black_balls := 1 } = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_prob_different_colors_for_given_box_l437_43734


namespace NUMINAMATH_CALUDE_square_property_fourth_power_property_smallest_square_smallest_fourth_power_sum_is_1130_l437_43799

/-- The smallest positive integer x such that 720x is a perfect square -/
def smallest_square_factor : ℕ := 5

/-- The smallest positive integer y such that 720y is a perfect fourth power -/
def smallest_fourth_power_factor : ℕ := 1125

/-- 720 * smallest_square_factor is a perfect square -/
theorem square_property : ∃ (n : ℕ), 720 * smallest_square_factor = n^2 := by sorry

/-- 720 * smallest_fourth_power_factor is a perfect fourth power -/
theorem fourth_power_property : ∃ (n : ℕ), 720 * smallest_fourth_power_factor = n^4 := by sorry

/-- smallest_square_factor is the smallest positive integer with the square property -/
theorem smallest_square :
  ∀ (k : ℕ), k > 0 ∧ k < smallest_square_factor → ¬∃ (n : ℕ), 720 * k = n^2 := by sorry

/-- smallest_fourth_power_factor is the smallest positive integer with the fourth power property -/
theorem smallest_fourth_power :
  ∀ (k : ℕ), k > 0 ∧ k < smallest_fourth_power_factor → ¬∃ (n : ℕ), 720 * k = n^4 := by sorry

/-- The sum of smallest_square_factor and smallest_fourth_power_factor -/
def sum_of_factors : ℕ := smallest_square_factor + smallest_fourth_power_factor

/-- The sum of the factors is 1130 -/
theorem sum_is_1130 : sum_of_factors = 1130 := by sorry

end NUMINAMATH_CALUDE_square_property_fourth_power_property_smallest_square_smallest_fourth_power_sum_is_1130_l437_43799


namespace NUMINAMATH_CALUDE_reflection_in_fourth_quadrant_l437_43764

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Defines the fourth quadrant -/
def fourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Reflects a point across the y-axis -/
def reflectYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Theorem stating that if P is in the second quadrant, 
    then the reflection of Q across the y-axis is in the fourth quadrant -/
theorem reflection_in_fourth_quadrant (a b : ℝ) :
  let p : Point := { x := a, y := b }
  let q : Point := { x := a - 1, y := -b }
  secondQuadrant p → fourthQuadrant (reflectYAxis q) := by
  sorry


end NUMINAMATH_CALUDE_reflection_in_fourth_quadrant_l437_43764


namespace NUMINAMATH_CALUDE_lego_set_cost_lego_set_cost_is_20_l437_43716

/-- The cost of each lego set when Tonya buys Christmas gifts for her sisters -/
theorem lego_set_cost (doll_cost : ℕ) (num_dolls : ℕ) (num_lego_sets : ℕ) : ℕ :=
  let total_doll_cost := doll_cost * num_dolls
  total_doll_cost / num_lego_sets

/-- Proof that each lego set costs $20 -/
theorem lego_set_cost_is_20 :
  lego_set_cost 15 4 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_lego_set_cost_lego_set_cost_is_20_l437_43716


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l437_43726

theorem pentagon_angle_measure (P Q R S T : ℝ) : 
  -- Pentagon condition
  P + Q + R + S + T = 540 →
  -- Equal angles condition
  P = R ∧ P = T →
  -- Supplementary angles condition
  Q + S = 180 →
  -- Conclusion
  T = 120 := by
sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l437_43726


namespace NUMINAMATH_CALUDE_brick_weight_l437_43758

theorem brick_weight :
  ∀ x : ℝ, x = 2 + x / 2 → x = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_brick_weight_l437_43758


namespace NUMINAMATH_CALUDE_equation_solution_l437_43730

theorem equation_solution : 
  ∃ (x : ℝ), x ≠ 2 ∧ (4*x^2 + 3*x + 2) / (x - 2) = 4*x + 5 → x = -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l437_43730


namespace NUMINAMATH_CALUDE_circle_land_theorem_l437_43745

/-- Represents a digit with its associated number of circles in Circle Land notation -/
structure CircleLandDigit where
  digit : Nat
  circles : Nat

/-- Calculates the value of a CircleLandDigit in the Circle Land number system -/
def circleValue (d : CircleLandDigit) : Nat :=
  d.digit * (10 ^ d.circles)

/-- Represents a number in Circle Land notation as a list of CircleLandDigits -/
def CircleLandNumber := List CircleLandDigit

/-- Calculates the value of a CircleLandNumber -/
def circleLandValue (n : CircleLandNumber) : Nat :=
  n.foldl (fun acc d => acc + circleValue d) 0

/-- The Circle Land representation of the number in the problem -/
def problemNumber : CircleLandNumber :=
  [⟨3, 4⟩, ⟨1, 2⟩, ⟨5, 0⟩]

theorem circle_land_theorem :
  circleLandValue problemNumber = 30105 := by sorry

end NUMINAMATH_CALUDE_circle_land_theorem_l437_43745


namespace NUMINAMATH_CALUDE_sum_of_roots_l437_43763

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1716 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l437_43763


namespace NUMINAMATH_CALUDE_incorrect_statement_l437_43795

def A (k : ℕ) : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 4*n + k}

theorem incorrect_statement :
  ¬ (∀ a b : ℤ, (a + b) ∈ A 3 → (a ∈ A 1 ∧ b ∈ A 2)) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_l437_43795


namespace NUMINAMATH_CALUDE_solve_equation_l437_43737

theorem solve_equation : ∃ x : ℝ, 3 * x + 15 = (1 / 3) * (8 * x + 48) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l437_43737


namespace NUMINAMATH_CALUDE_tangent_perpendicular_l437_43742

-- Define the curve C
def C (x : ℝ) : ℝ := x^2 + x

-- Define the derivative of C
def C' (x : ℝ) : ℝ := 2*x + 1

-- Define the perpendicular line
def perp_line (a x y : ℝ) : Prop := a*x - y + 1 = 0

theorem tangent_perpendicular :
  ∀ a : ℝ, 
  (C' 1 = -1/a) →  -- The slope of the tangent at x=1 is the negative reciprocal of a
  a = -1/3 := by
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_l437_43742


namespace NUMINAMATH_CALUDE_tom_rental_hours_l437_43732

/-- Represents the rental fees and total amount paid --/
structure RentalInfo where
  baseFee : ℕ
  bikeHourlyFee : ℕ
  helmetFee : ℕ
  lockHourlyFee : ℕ
  totalPaid : ℕ

/-- Calculates the number of hours rented based on the rental information --/
def calculateHoursRented (info : RentalInfo) : ℕ :=
  ((info.totalPaid - info.baseFee - info.helmetFee) / (info.bikeHourlyFee + info.lockHourlyFee))

/-- Theorem stating that Tom rented the bike and accessories for 8 hours --/
theorem tom_rental_hours (info : RentalInfo) 
    (h1 : info.baseFee = 17)
    (h2 : info.bikeHourlyFee = 7)
    (h3 : info.helmetFee = 5)
    (h4 : info.lockHourlyFee = 2)
    (h5 : info.totalPaid = 95) : 
  calculateHoursRented info = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_rental_hours_l437_43732


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l437_43715

theorem sqrt_product_equality : 
  (Real.sqrt 8 - Real.sqrt 2) * (Real.sqrt 7 - Real.sqrt 3) = Real.sqrt 14 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l437_43715


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l437_43781

theorem geometric_series_first_term 
  (a r : ℝ) 
  (sum_condition : a / (1 - r) = 18) 
  (sum_squares_condition : a^2 / (1 - r^2) = 72) : 
  a = 7.2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l437_43781


namespace NUMINAMATH_CALUDE_tracy_candies_l437_43720

theorem tracy_candies : ∃ (initial : ℕ) (brother_took : ℕ), 
  initial % 20 = 0 ∧ 
  1 ≤ brother_took ∧ 
  brother_took ≤ 6 ∧
  (3 * initial) / 5 - 40 - brother_took = 4 ∧
  initial = 80 := by sorry

end NUMINAMATH_CALUDE_tracy_candies_l437_43720


namespace NUMINAMATH_CALUDE_flight_cost_B_to_C_l437_43700

/-- Represents a city in a triangular configuration -/
inductive City
| A
| B
| C

/-- Represents the distance between two cities in kilometers -/
def distance (x y : City) : ℝ :=
  match x, y with
  | City.A, City.C => 3000
  | City.B, City.C => 1000
  | _, _ => 0  -- We don't need other distances for this problem

/-- The booking fee for a flight in dollars -/
def bookingFee : ℝ := 100

/-- The cost per kilometer for a flight in dollars -/
def costPerKm : ℝ := 0.1

/-- Calculates the cost of a flight between two cities -/
def flightCost (x y : City) : ℝ :=
  bookingFee + costPerKm * distance x y

/-- States that cities A, B, and C form a right-angled triangle with C as the right angle -/
axiom right_angle_at_C : distance City.A City.B ^ 2 = distance City.A City.C ^ 2 + distance City.B City.C ^ 2

theorem flight_cost_B_to_C :
  flightCost City.B City.C = 200 := by
  sorry

end NUMINAMATH_CALUDE_flight_cost_B_to_C_l437_43700


namespace NUMINAMATH_CALUDE_swimmer_passes_l437_43771

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  delay : ℝ

/-- Calculates the number of times swimmers pass each other --/
def count_passes (pool_length : ℝ) (time : ℝ) (swimmer_a : Swimmer) (swimmer_b : Swimmer) : ℕ :=
  sorry

/-- Theorem stating the number of passes in the given scenario --/
theorem swimmer_passes :
  let pool_length : ℝ := 120
  let total_time : ℝ := 900
  let swimmer_a : Swimmer := { speed := 3, delay := 0 }
  let swimmer_b : Swimmer := { speed := 4, delay := 10 }
  count_passes pool_length total_time swimmer_a swimmer_b = 38 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_passes_l437_43771


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l437_43782

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 40 → area = (perimeter / 4) ^ 2 → area = 100 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l437_43782


namespace NUMINAMATH_CALUDE_peaches_left_l437_43762

/-- Given baskets of peaches with specific initial conditions, proves the number of peaches left after removal. -/
theorem peaches_left (initial_baskets : Nat) (initial_peaches : Nat) (added_baskets : Nat) (added_peaches : Nat) (removed_peaches : Nat) : 
  initial_baskets = 5 →
  initial_peaches = 20 →
  added_baskets = 4 →
  added_peaches = 25 →
  removed_peaches = 10 →
  (initial_baskets * initial_peaches + added_baskets * added_peaches) - 
  ((initial_baskets + added_baskets) * removed_peaches) = 110 := by
  sorry

end NUMINAMATH_CALUDE_peaches_left_l437_43762


namespace NUMINAMATH_CALUDE_coltons_stickers_coltons_initial_stickers_l437_43779

theorem coltons_stickers (friends_count : ℕ) (stickers_per_friend : ℕ) 
  (extra_for_mandy : ℕ) (less_for_justin : ℕ) (stickers_left : ℕ) : ℕ :=
  let friends_total := friends_count * stickers_per_friend
  let mandy_stickers := friends_total + extra_for_mandy
  let justin_stickers := mandy_stickers - less_for_justin
  let given_away := friends_total + mandy_stickers + justin_stickers
  given_away + stickers_left

theorem coltons_initial_stickers : 
  coltons_stickers 3 4 2 10 42 = 72 := by sorry

end NUMINAMATH_CALUDE_coltons_stickers_coltons_initial_stickers_l437_43779


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l437_43790

/-- Converts a base 7 number to decimal --/
def toDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- The number in base 7 --/
def base7Number : List Nat := [1, 2, 0, 2, 1, 0, 1, 2]

/-- The decimal representation of the base 7 number --/
def decimalNumber : Nat := toDecimal base7Number

/-- Predicate to check if a number is prime --/
def isPrime (n : Nat) : Prop := sorry

theorem largest_prime_divisor :
  ∃ (p : Nat), isPrime p ∧ p ∣ decimalNumber ∧ 
  ∀ (q : Nat), isPrime q → q ∣ decimalNumber → q ≤ p ∧ p = 397 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l437_43790


namespace NUMINAMATH_CALUDE_data_average_l437_43780

theorem data_average (a : ℝ) : 
  (1 + 3 + 2 + 5 + a) / 5 = 3 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_data_average_l437_43780


namespace NUMINAMATH_CALUDE_total_weight_problem_l437_43788

/-- The total weight problem -/
theorem total_weight_problem (a b c d : ℕ) 
  (h1 : a + b = 250)
  (h2 : b + c = 235)
  (h3 : c + d = 260)
  (h4 : a + d = 275) :
  a + b + c + d = 510 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_problem_l437_43788


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l437_43708

theorem quadratic_two_roots (a b c : ℝ) (h1 : b > a + c) (h2 : a + c > 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l437_43708


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l437_43738

theorem trigonometric_inequality (α β γ : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : 0 < γ ∧ γ < π/2)
  (h4 : Real.sin α ^ 3 + Real.sin β ^ 3 + Real.sin γ ^ 3 = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l437_43738


namespace NUMINAMATH_CALUDE_lemonade_sold_l437_43750

/-- Represents the number of cups of lemonade sold -/
def lemonade : ℕ := sorry

/-- Represents the number of cups of hot chocolate sold -/
def hotChocolate : ℕ := sorry

/-- The total number of cups sold -/
def totalCups : ℕ := 400

/-- The total money earned in yuan -/
def totalMoney : ℕ := 546

/-- The price of a cup of lemonade in yuan -/
def lemonadePrice : ℕ := 1

/-- The price of a cup of hot chocolate in yuan -/
def hotChocolatePrice : ℕ := 2

theorem lemonade_sold : 
  lemonade = 254 ∧ 
  lemonade + hotChocolate = totalCups ∧ 
  lemonade * lemonadePrice + hotChocolate * hotChocolatePrice = totalMoney :=
sorry

end NUMINAMATH_CALUDE_lemonade_sold_l437_43750


namespace NUMINAMATH_CALUDE_probability_both_selected_l437_43733

theorem probability_both_selected (prob_ram : ℚ) (prob_ravi : ℚ) 
  (h1 : prob_ram = 4/7) (h2 : prob_ravi = 1/5) : 
  prob_ram * prob_ravi = 4/35 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l437_43733


namespace NUMINAMATH_CALUDE_total_cost_correct_l437_43725

def makeup_palette_price : ℝ := 15
def lipstick_price : ℝ := 2.5
def hair_color_price : ℝ := 4

def makeup_palette_count : ℕ := 3
def lipstick_count : ℕ := 4
def hair_color_count : ℕ := 3

def makeup_palette_discount : ℝ := 0.2
def hair_color_coupon_discount : ℝ := 0.1
def reward_points_discount : ℝ := 5

def storewide_discount_threshold : ℝ := 50
def storewide_discount_rate : ℝ := 0.1

def sales_tax_threshold : ℝ := 25
def sales_tax_rate_low : ℝ := 0.05
def sales_tax_rate_high : ℝ := 0.08

def calculate_total_cost : ℝ := sorry

theorem total_cost_correct : 
  ∀ ε > 0, |calculate_total_cost - 47.41| < ε := by sorry

end NUMINAMATH_CALUDE_total_cost_correct_l437_43725


namespace NUMINAMATH_CALUDE_winnie_lollipops_l437_43754

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (total_lollipops friends : ℕ) : ℕ :=
  total_lollipops % friends

theorem winnie_lollipops :
  let cherry := 32
  let wintergreen := 105
  let grape := 7
  let shrimp := 198
  let friends := 12
  let total := cherry + wintergreen + grape + shrimp
  lollipops_kept total friends = 6 := by sorry

end NUMINAMATH_CALUDE_winnie_lollipops_l437_43754


namespace NUMINAMATH_CALUDE_equation_solution_l437_43721

theorem equation_solution :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 2 * x - 3 = 0 :=
by
  use 3/2
  sorry

end NUMINAMATH_CALUDE_equation_solution_l437_43721


namespace NUMINAMATH_CALUDE_wade_friday_customers_l437_43741

/-- Represents the number of customers Wade served on Friday -/
def F : ℕ := by sorry

/-- Wade's tip per customer in dollars -/
def tip_per_customer : ℚ := 2

/-- Total tips Wade made over the three days in dollars -/
def total_tips : ℚ := 296

theorem wade_friday_customers :
  F = 28 ∧
  tip_per_customer * (F + 3 * F + 36) = total_tips :=
by sorry

end NUMINAMATH_CALUDE_wade_friday_customers_l437_43741


namespace NUMINAMATH_CALUDE_square_of_binomial_coefficient_l437_43751

/-- If bx^2 + 18x + 9 is the square of a binomial, then b = 9 -/
theorem square_of_binomial_coefficient (b : ℝ) : 
  (∃ t u : ℝ, ∀ x : ℝ, bx^2 + 18*x + 9 = (t*x + u)^2) → b = 9 :=
by sorry

end NUMINAMATH_CALUDE_square_of_binomial_coefficient_l437_43751


namespace NUMINAMATH_CALUDE_x_in_M_l437_43747

def M : Set ℝ := {x | x ≤ 7}

theorem x_in_M : 4 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_x_in_M_l437_43747


namespace NUMINAMATH_CALUDE_card_house_47_floors_l437_43703

/-- The number of cards needed for the nth floor of a card house -/
def cards_for_floor (n : ℕ) : ℕ := 2 + (n - 1) * 3

/-- The total number of cards needed for a card house with n floors -/
def total_cards (n : ℕ) : ℕ := 
  n * (cards_for_floor 1 + cards_for_floor n) / 2

/-- Theorem: A card house with 47 floors requires 3337 cards -/
theorem card_house_47_floors : total_cards 47 = 3337 := by
  sorry

end NUMINAMATH_CALUDE_card_house_47_floors_l437_43703


namespace NUMINAMATH_CALUDE_card_game_unfair_l437_43773

/-- Represents a playing card with a rank and suit -/
structure Card :=
  (rank : Nat)
  (suit : Nat)

/-- Represents the deck of cards -/
def Deck : Finset Card := sorry

/-- The number of cards in the deck -/
def deckSize : Nat := Finset.card Deck

/-- Volodya's draw -/
def volodyaDraw : Deck → Card := sorry

/-- Masha's draw -/
def mashaDraw : Deck → Card → Card := sorry

/-- Masha wins if her card rank is higher than Volodya's -/
def mashaWins (vCard mCard : Card) : Prop := mCard.rank > vCard.rank

/-- The probability of Masha winning -/
def probMashaWins : ℝ := sorry

/-- Theorem: The card game is unfair (biased against Masha) -/
theorem card_game_unfair : probMashaWins < 1/2 := by sorry

end NUMINAMATH_CALUDE_card_game_unfair_l437_43773


namespace NUMINAMATH_CALUDE_sandwich_bread_count_l437_43765

/-- The number of pieces of bread needed for one double meat sandwich -/
def double_meat_bread : ℕ := 3

/-- The number of regular sandwiches -/
def regular_sandwiches : ℕ := 14

/-- The number of double meat sandwiches -/
def double_meat_sandwiches : ℕ := 12

/-- The number of pieces of bread needed for one regular sandwich -/
def regular_bread : ℕ := 2

/-- The total number of pieces of bread used -/
def total_bread : ℕ := 64

theorem sandwich_bread_count : 
  regular_sandwiches * regular_bread + double_meat_sandwiches * double_meat_bread = total_bread := by
  sorry

end NUMINAMATH_CALUDE_sandwich_bread_count_l437_43765


namespace NUMINAMATH_CALUDE_exists_same_answer_question_l437_43714

/-- A person who either always tells the truth or always lies -/
inductive Person
| TruthTeller
| Liar

/-- The answer a person gives to a question -/
inductive Answer
| Yes
| No

/-- A question that can be asked to a person -/
def Question := Person → Answer

/-- The actual answer to a question about a person -/
def actualAnswer (p : Person) (q : Question) : Answer :=
  match p with
  | Person.TruthTeller => q Person.TruthTeller
  | Person.Liar => match q Person.Liar with
    | Answer.Yes => Answer.No
    | Answer.No => Answer.Yes

/-- There exists a question that makes both a truth-teller and a liar give the same answer -/
theorem exists_same_answer_question : ∃ (q : Question),
  actualAnswer Person.TruthTeller q = actualAnswer Person.Liar q :=
sorry

end NUMINAMATH_CALUDE_exists_same_answer_question_l437_43714


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l437_43722

theorem quadratic_equation_m_value 
  (x₁ x₂ m : ℝ) 
  (h1 : x₁^2 - 5*x₁ + m = 0)
  (h2 : x₂^2 - 5*x₂ + m = 0)
  (h3 : 3*x₁ - 2*x₂ = 5) :
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l437_43722


namespace NUMINAMATH_CALUDE_angle_ADC_measure_l437_43766

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)

-- Define the point D
structure PointD (t : Triangle) :=
  (on_angle_bisector : True)  -- D is on the angle bisector of ∠ABC
  (on_perp_bisector : True)   -- D is on the perpendicular bisector of AC

-- Theorem statement
theorem angle_ADC_measure (t : Triangle) (d : PointD t) 
  (h1 : t.A = 44) (h2 : t.B = 66) (h3 : t.C = 70) : 
  ∃ (angle_ADC : ℝ), angle_ADC = 114 := by
  sorry

end NUMINAMATH_CALUDE_angle_ADC_measure_l437_43766


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l437_43761

theorem square_perimeter_ratio (s1 s2 : ℝ) (h : s1^2 / s2^2 = 16 / 81) :
  (4 * s1) / (4 * s2) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l437_43761


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l437_43727

theorem min_value_of_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 9/y ≥ 1/a + 9/b) ∧ 1/a + 9/b = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l437_43727


namespace NUMINAMATH_CALUDE_quadratic_even_iff_a_eq_zero_l437_43775

/-- A quadratic function f(x) = x^2 + ax + b is even if and only if a = 0 -/
theorem quadratic_even_iff_a_eq_zero (a b : ℝ) :
  (∀ x : ℝ, x^2 + a*x + b = (-x)^2 + a*(-x) + b) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_even_iff_a_eq_zero_l437_43775


namespace NUMINAMATH_CALUDE_nineteen_to_binary_l437_43792

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

-- State the theorem
theorem nineteen_to_binary :
  decimalToBinary 19 = [1, 0, 0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_nineteen_to_binary_l437_43792


namespace NUMINAMATH_CALUDE_fixed_intersection_point_l437_43756

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an angle with two sides -/
structure Angle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop

/-- Predicate to check if two circles are non-overlapping -/
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

/-- Predicate to check if a point is on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Predicate to check if the angle touches both circles -/
def touches_circles (a : Angle) (c1 c2 : Circle) : Prop :=
  ∃ p1 p2 : ℝ × ℝ,
    a.side1 p1 ∧ on_circle p1 c1 ∧
    a.side2 p2 ∧ on_circle p2 c2 ∧
    p1 ≠ a.vertex ∧ p2 ≠ a.vertex

/-- The main theorem -/
theorem fixed_intersection_point
  (c1 c2 : Circle)
  (h_non_overlapping : non_overlapping c1 c2) :
  ∃ p : ℝ × ℝ,
    ∀ a : Angle,
      touches_circles a c1 c2 →
      ∃ t : ℝ,
        p.1 = a.vertex.1 + t * (p.1 - a.vertex.1) ∧
        p.2 = a.vertex.2 + t * (p.2 - a.vertex.2) :=
  sorry

end NUMINAMATH_CALUDE_fixed_intersection_point_l437_43756


namespace NUMINAMATH_CALUDE_target_breaking_orders_l437_43707

theorem target_breaking_orders : 
  (Nat.factorial 8) / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 2) = 560 := by
  sorry

end NUMINAMATH_CALUDE_target_breaking_orders_l437_43707


namespace NUMINAMATH_CALUDE_regular_polyhedra_similarity_l437_43753

/-- A regular polyhedron -/
structure RegularPolyhedron where
  -- Add necessary fields here
  -- For example:
  vertices : Set (ℝ × ℝ × ℝ)
  faces : Set (Set (ℝ × ℝ × ℝ))
  -- Add more fields as needed

/-- Defines what it means for two polyhedra to be of the same combinatorial type -/
def same_combinatorial_type (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- Defines what it means for faces to be of the same kind -/
def same_kind_faces (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- Defines what it means for polyhedral angles to be of the same kind -/
def same_kind_angles (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- Defines similarity between two polyhedra -/
def similar (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- The main theorem: two regular polyhedra of the same combinatorial type
    with faces and polyhedral angles of the same kind are similar -/
theorem regular_polyhedra_similarity (P Q : RegularPolyhedron)
  (h1 : same_combinatorial_type P Q)
  (h2 : same_kind_faces P Q)
  (h3 : same_kind_angles P Q) :
  similar P Q :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polyhedra_similarity_l437_43753


namespace NUMINAMATH_CALUDE_carson_clawed_39_times_l437_43796

/-- The number of times Carson gets clawed in the zoo enclosure. -/
def total_claws (num_wombats : ℕ) (num_rheas : ℕ) (wombat_claws : ℕ) (rhea_claws : ℕ) : ℕ :=
  num_wombats * wombat_claws + num_rheas * rhea_claws

/-- Theorem stating that Carson gets clawed 39 times given the specific conditions. -/
theorem carson_clawed_39_times :
  total_claws 9 3 4 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_carson_clawed_39_times_l437_43796


namespace NUMINAMATH_CALUDE_sum_of_456_terms_l437_43704

/-- An arithmetic progression with first term 2 and sum of second and third terms 13 -/
def ArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, 
    (a 1 = 2) ∧ 
    (a 2 + a 3 = 13) ∧ 
    ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 4th, 5th, and 6th terms of the arithmetic progression is 42 -/
theorem sum_of_456_terms (a : ℕ → ℝ) (h : ArithmeticProgression a) : 
  a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_456_terms_l437_43704


namespace NUMINAMATH_CALUDE_distance_apex_to_circumsphere_center_l437_43797

/-- Represents a rectangular pyramid with a frustum -/
structure RectangularPyramidWithFrustum where
  /-- Length of the rectangle base -/
  baseLength : ℝ
  /-- Width of the rectangle base -/
  baseWidth : ℝ
  /-- Height of the pyramid -/
  pyramidHeight : ℝ
  /-- Ratio of the volume of the smaller pyramid to the whole pyramid -/
  volumeRatio : ℝ

/-- Theorem stating the distance between the apex and the center of the frustum's circumsphere -/
theorem distance_apex_to_circumsphere_center
  (p : RectangularPyramidWithFrustum)
  (h1 : p.baseLength = 15)
  (h2 : p.baseWidth = 20)
  (h3 : p.pyramidHeight = 30)
  (h4 : p.volumeRatio = 1/9) :
  let xt := p.pyramidHeight - (1 - p.volumeRatio^(1/3)) * p.pyramidHeight +
            (p.baseLength^2 + p.baseWidth^2) / (18 * p.pyramidHeight)
  xt = 425/9 := by
  sorry

#check distance_apex_to_circumsphere_center

end NUMINAMATH_CALUDE_distance_apex_to_circumsphere_center_l437_43797


namespace NUMINAMATH_CALUDE_gcd_triple_characterization_l437_43757

theorem gcd_triple_characterization (a b c : ℕ+) :
  Nat.gcd a.val 20 = b.val ∧
  Nat.gcd b.val 15 = c.val ∧
  Nat.gcd a.val c.val = 5 →
  ∃ k : ℕ+, (a = 5 * k ∧ b = 5 ∧ c = 5) ∨
            (a = 5 * k ∧ b = 10 ∧ c = 5) ∨
            (a = 5 * k ∧ b = 20 ∧ c = 5) := by
  sorry


end NUMINAMATH_CALUDE_gcd_triple_characterization_l437_43757


namespace NUMINAMATH_CALUDE_store_purchase_divisibility_l437_43749

theorem store_purchase_divisibility (m n k : ℕ) :
  ∃ p : ℕ, 3 * m + 4 * n + 5 * k = 11 * p →
  ∃ q : ℕ, 9 * m + n + 4 * k = 11 * q :=
by sorry

end NUMINAMATH_CALUDE_store_purchase_divisibility_l437_43749


namespace NUMINAMATH_CALUDE_complex_number_simplification_l437_43748

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) :
  (2 + i^3) / (1 - i) = (3 + i) / 2 := by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l437_43748


namespace NUMINAMATH_CALUDE_inequality_solution_l437_43736

theorem inequality_solution (x : ℝ) (h : x ≠ 1) :
  x / (x - 1) ≥ 2 * x ↔ x ≤ 0 ∨ (1 < x ∧ x ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l437_43736


namespace NUMINAMATH_CALUDE_flower_bed_fraction_l437_43719

/-- Represents a rectangular yard with two congruent isosceles right triangular flower beds -/
structure FlowerYard where
  length : ℝ
  width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  (short_side_positive : trapezoid_short_side > 0)
  (long_side_positive : trapezoid_long_side > 0)
  (short_less_than_long : trapezoid_short_side < trapezoid_long_side)
  (width_eq : width = (trapezoid_long_side - trapezoid_short_side) / 2)
  (length_eq : length = trapezoid_long_side)

/-- The fraction of the yard occupied by the flower beds is 1/5 -/
theorem flower_bed_fraction (yard : FlowerYard) (h1 : yard.trapezoid_short_side = 15) 
    (h2 : yard.trapezoid_long_side = 25) : 
  (2 * yard.width ^ 2) / (yard.length * yard.width) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_l437_43719


namespace NUMINAMATH_CALUDE_fraction_meaningful_l437_43717

theorem fraction_meaningful (m : ℝ) : 
  (∃ (x : ℝ), x = 4 / (m - 1)) ↔ m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l437_43717
