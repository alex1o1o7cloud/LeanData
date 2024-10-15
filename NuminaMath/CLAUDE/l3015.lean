import Mathlib

namespace NUMINAMATH_CALUDE_coordinate_sum_of_h_l3015_301524

/-- Given a function g where g(3) = 8, and a function h where h(x) = (g(x))^3 for all x,
    the sum of the coordinates of the point (3, h(3)) is 515. -/
theorem coordinate_sum_of_h (g : ℝ → ℝ) (h : ℝ → ℝ) 
    (hg : g 3 = 8) (hh : ∀ x, h x = (g x)^3) : 
    3 + h 3 = 515 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_h_l3015_301524


namespace NUMINAMATH_CALUDE_inequality_proof_l3015_301594

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3015_301594


namespace NUMINAMATH_CALUDE_function_value_sum_l3015_301597

/-- A quadratic function f(x) with specific properties -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 2)^2 + 4

/-- The theorem stating the value of a + b + 2c for the given function -/
theorem function_value_sum (a : ℝ) :
  f a 0 = 5 ∧ f a 2 = 5 → a + 0 + 2 * 4 = 8.25 := by sorry

end NUMINAMATH_CALUDE_function_value_sum_l3015_301597


namespace NUMINAMATH_CALUDE_max_episodes_l3015_301503

/-- Represents a character in the TV show -/
structure Character where
  id : Nat

/-- Represents the state of knowledge for each character -/
structure KnowledgeState where
  knows_mystery : Set Character
  knows_others_know : Set (Character × Character)
  knows_others_dont_know : Set (Character × Character)

/-- Represents an episode of the TV show -/
inductive Episode
  | LearnMystery (c : Character)
  | LearnSomeoneKnows (c1 c2 : Character)
  | LearnSomeoneDoesntKnow (c1 c2 : Character)

/-- The number of characters in the TV show -/
def num_characters : Nat := 20

/-- Theorem: The maximum number of unique episodes is 780 -/
theorem max_episodes :
  ∃ (episodes : List Episode),
    episodes.length = 780 ∧
    episodes.Nodup ∧
    (∀ e : Episode, e ∈ episodes) ∧
    (∀ c : List Character, c.length = num_characters →
      ∃ (initial_state : KnowledgeState),
        ∃ (final_state : KnowledgeState),
          episodes.foldl
            (fun state episode =>
              match episode with
              | Episode.LearnMystery c =>
                { state with knows_mystery := state.knows_mystery ∪ {c} }
              | Episode.LearnSomeoneKnows c1 c2 =>
                { state with knows_others_know := state.knows_others_know ∪ {(c1, c2)} }
              | Episode.LearnSomeoneDoesntKnow c1 c2 =>
                { state with knows_others_dont_know := state.knows_others_dont_know ∪ {(c1, c2)} })
            initial_state
          = final_state) :=
  sorry

end NUMINAMATH_CALUDE_max_episodes_l3015_301503


namespace NUMINAMATH_CALUDE_largest_red_socks_l3015_301519

/-- The largest number of red socks in a drawer with specific conditions -/
theorem largest_red_socks (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total ≤ 1991)
  (h2 : total = red + blue)
  (h3 : (red * (red - 1) + blue * (blue - 1)) / (total * (total - 1)) = 1/2) :
  red ≤ 990 ∧ ∃ (r : ℕ), r = 990 ∧ 
    ∃ (t b : ℕ), t ≤ 1991 ∧ t = r + b ∧ 
      (r * (r - 1) + b * (b - 1)) / (t * (t - 1)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_largest_red_socks_l3015_301519


namespace NUMINAMATH_CALUDE_music_spending_l3015_301561

theorem music_spending (total_allowance : ℝ) (music_fraction : ℝ) : 
  total_allowance = 50 → music_fraction = 3/10 → music_fraction * total_allowance = 15 := by
  sorry

end NUMINAMATH_CALUDE_music_spending_l3015_301561


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l3015_301534

theorem sum_of_coefficients_zero (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (x^2 + x + 1) * (2*x - a)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₀ = -32 →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l3015_301534


namespace NUMINAMATH_CALUDE_min_weighings_nine_medals_l3015_301539

/-- Represents a set of medals with one heavier than the others -/
structure MedalSet :=
  (total : Nat)
  (heavier_exists : total > 0)

/-- Represents a balance scale used for weighing -/
structure BalanceScale

/-- The minimum number of weighings required to find the heavier medal -/
def min_weighings (medals : MedalSet) (scale : BalanceScale) : Nat :=
  sorry

/-- Theorem stating that for 9 medals, the minimum number of weighings is 2 -/
theorem min_weighings_nine_medals :
  ∀ (scale : BalanceScale),
  min_weighings ⟨9, by norm_num⟩ scale = 2 :=
sorry

end NUMINAMATH_CALUDE_min_weighings_nine_medals_l3015_301539


namespace NUMINAMATH_CALUDE_bus_ride_cost_l3015_301501

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 3.75

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := bus_cost + 2.35

/-- The theorem stating the cost of a bus ride from town P to town Q -/
theorem bus_ride_cost : bus_cost = 3.75 := by sorry

/-- The condition that a train ride costs $2.35 more than a bus ride -/
axiom train_cost_difference : train_cost = bus_cost + 2.35

/-- The condition that the combined cost of one train ride and one bus ride is $9.85 -/
axiom combined_cost : train_cost + bus_cost = 9.85

end NUMINAMATH_CALUDE_bus_ride_cost_l3015_301501


namespace NUMINAMATH_CALUDE_minimum_value_of_expression_l3015_301510

theorem minimum_value_of_expression (x : ℝ) (h : x > 0) :
  x + 4 / x ≥ 4 ∧ (x + 4 / x = 4 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_expression_l3015_301510


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l3015_301596

/-- Represents a hyperbola with parameters m and n -/
structure Hyperbola (m n : ℝ) where
  eq : ∀ x y : ℝ, x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

/-- The distance between the foci of the hyperbola is 4 -/
def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

/-- The theorem stating the range of n for the given hyperbola -/
theorem hyperbola_n_range (m n : ℝ) (h : Hyperbola m n) (d : foci_distance m n) :
  -1 < n ∧ n < 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l3015_301596


namespace NUMINAMATH_CALUDE_nancy_bottle_caps_l3015_301581

/-- The number of bottle caps Nancy starts with -/
def initial_caps : ℕ := 91

/-- The number of bottle caps Nancy finds -/
def found_caps : ℕ := 88

/-- The total number of bottle caps Nancy ends with -/
def total_caps : ℕ := initial_caps + found_caps

theorem nancy_bottle_caps : total_caps = 179 := by sorry

end NUMINAMATH_CALUDE_nancy_bottle_caps_l3015_301581


namespace NUMINAMATH_CALUDE_infinite_divisible_factorial_exponents_l3015_301599

/-- νₚ(n) is the exponent of p in the prime factorization of n! -/
def ν (p : Nat) (n : Nat) : Nat :=
  sorry

theorem infinite_divisible_factorial_exponents
  (d : Nat) (primes : Finset Nat) (h_primes : ∀ p ∈ primes, Nat.Prime p) :
  ∃ S : Set Nat, Set.Infinite S ∧
    ∀ n ∈ S, ∀ p ∈ primes, d ∣ ν p n :=
  sorry

end NUMINAMATH_CALUDE_infinite_divisible_factorial_exponents_l3015_301599


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3015_301585

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) →
  a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3015_301585


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_squared_l3015_301564

theorem imaginary_part_of_z_squared (z : ℂ) (h : z * (1 - Complex.I) = 2) : 
  Complex.im (z^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_squared_l3015_301564


namespace NUMINAMATH_CALUDE_total_tips_calculation_l3015_301515

def lawn_price : ℕ := 33
def lawns_mowed : ℕ := 16
def total_earned : ℕ := 558

theorem total_tips_calculation : 
  total_earned - (lawn_price * lawns_mowed) = 30 := by sorry

end NUMINAMATH_CALUDE_total_tips_calculation_l3015_301515


namespace NUMINAMATH_CALUDE_number_calculation_l3015_301541

theorem number_calculation (x : ℝ) : 0.25 * x = 0.20 * 650 + 190 → x = 1280 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l3015_301541


namespace NUMINAMATH_CALUDE_vlad_sister_height_difference_l3015_301502

/-- Converts feet and inches to total inches -/
def heightToInches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Represents the height difference between two people -/
def heightDifference (person1_feet : ℕ) (person1_inches : ℕ) (person2_feet : ℕ) (person2_inches : ℕ) : ℕ :=
  heightToInches person1_feet person1_inches - heightToInches person2_feet person2_inches

theorem vlad_sister_height_difference :
  heightDifference 6 3 2 10 = 41 := by sorry

end NUMINAMATH_CALUDE_vlad_sister_height_difference_l3015_301502


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3015_301529

/-- Ellipse C with foci at (-2,0) and (2,0), passing through (0, √5) -/
def ellipse_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ x^2/9 + y^2/5 = 1}

/-- Line l passing through (-2,0) with slope 1 -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ y = x + 2}

/-- Theorem stating the standard equation of ellipse C and the length of PQ -/
theorem ellipse_and_line_intersection :
  (∀ (x y : ℝ), (x, y) ∈ ellipse_C ↔ x^2/9 + y^2/5 = 1) ∧
  (∃ (P Q : ℝ × ℝ), P ∈ ellipse_C ∧ Q ∈ ellipse_C ∧ P ∈ line_l ∧ Q ∈ line_l ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 30/7) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3015_301529


namespace NUMINAMATH_CALUDE_seven_from_five_twos_l3015_301548

theorem seven_from_five_twos : ∃ (a b c d e f g h i j : ℕ),
  (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2) ∧
  (f = 2 ∧ g = 2 ∧ h = 2 ∧ i = 2 ∧ j = 2) ∧
  (a * b * c - d / e = 7) ∧
  (f + g + h + i / j = 7) ∧
  ((10 * a + b) / c - d * e = 7) :=
by sorry

end NUMINAMATH_CALUDE_seven_from_five_twos_l3015_301548


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_regular_octagon_interior_angle_is_135_l3015_301584

theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_of_interior_angles : ℝ := 180 * (n - 2)
  let one_interior_angle : ℝ := sum_of_interior_angles / n
  135

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle_is_135 :
  regular_octagon_interior_angle = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_regular_octagon_interior_angle_is_135_l3015_301584


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3015_301573

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^12 + b^12) / (a + b)^9 = -2 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3015_301573


namespace NUMINAMATH_CALUDE_cookies_leftover_is_four_l3015_301520

/-- The number of cookies left over when selling in packs of 10 -/
def cookies_leftover (abigail beatrice carson : ℕ) : ℕ :=
  (abigail + beatrice + carson) % 10

/-- Theorem stating that the number of cookies left over is 4 -/
theorem cookies_leftover_is_four :
  cookies_leftover 53 65 26 = 4 := by
  sorry

end NUMINAMATH_CALUDE_cookies_leftover_is_four_l3015_301520


namespace NUMINAMATH_CALUDE_min_value_abc_minus_b_l3015_301527

def S : Finset Int := {-10, -7, -3, 0, 4, 6, 9}

theorem min_value_abc_minus_b :
  (∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a * b * c - b = -546) ∧
  (∀ (a b c : Int), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
    a * b * c - b ≥ -546) :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_minus_b_l3015_301527


namespace NUMINAMATH_CALUDE_sum_of_combined_sequence_l3015_301588

/-- Given geometric sequence {aₙ} and arithmetic sequence {bₙ} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Theorem stating the sum of the first n terms of the sequence cₙ = aₙ + bₙ -/
theorem sum_of_combined_sequence
  (a b : ℕ → ℝ)
  (h_a : geometric_sequence a)
  (h_b : arithmetic_sequence b)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 8)
  (h_b1 : b 1 = 3)
  (h_b4 : b 4 = 12) :
  ∃ S : ℕ → ℝ, ∀ n : ℕ, S n = 2^n - 1 + (3/2 * n^2) + (3/2 * n) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_combined_sequence_l3015_301588


namespace NUMINAMATH_CALUDE_scarf_cost_is_two_l3015_301522

/-- The cost of a single scarf given Kiki's spending habits -/
def scarf_cost (total_money : ℚ) (num_scarves : ℕ) (hat_percentage : ℚ) : ℚ :=
  (1 - hat_percentage) * total_money / num_scarves

/-- Proof that the cost of each scarf is $2 -/
theorem scarf_cost_is_two :
  scarf_cost 90 18 (3/5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_scarf_cost_is_two_l3015_301522


namespace NUMINAMATH_CALUDE_hemisphere_base_area_l3015_301576

/-- Given a hemisphere with total surface area 9, prove its base area is 3 -/
theorem hemisphere_base_area (r : ℝ) (h : 3 * π * r^2 = 9) : π * r^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_base_area_l3015_301576


namespace NUMINAMATH_CALUDE_vertices_must_be_even_l3015_301557

-- Define a polyhedron
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

-- Define a property for trihedral angles
def has_trihedral_angles (p : Polyhedron) : Prop :=
  3 * p.vertices = 2 * p.edges

-- Theorem statement
theorem vertices_must_be_even (p : Polyhedron) 
  (h : has_trihedral_angles p) : Even p.vertices := by
  sorry


end NUMINAMATH_CALUDE_vertices_must_be_even_l3015_301557


namespace NUMINAMATH_CALUDE_red_balls_count_l3015_301563

/-- The number of red balls in a bag with given conditions -/
theorem red_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (yellow : ℕ) (purple : ℕ) 
    (prob_not_red_purple : ℚ) (h1 : total = 100) (h2 : white = 50) (h3 : green = 30) 
    (h4 : yellow = 10) (h5 : purple = 3) (h6 : prob_not_red_purple = 9/10) : 
    total - (white + green + yellow + purple) = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3015_301563


namespace NUMINAMATH_CALUDE_second_third_ratio_l3015_301535

theorem second_third_ratio (A B C : ℚ) : 
  A + B + C = 98 →
  A / B = 2 / 3 →
  B = 30 →
  B / C = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_second_third_ratio_l3015_301535


namespace NUMINAMATH_CALUDE_shaded_area_is_one_third_l3015_301591

/-- Represents a 3x3 square quilt block -/
structure QuiltBlock :=
  (size : Nat)
  (shaded_area : ℚ)

/-- The size of the quilt block is 3 -/
def quilt_size : Nat := 3

/-- The quilt block with the given shaded pattern -/
def patterned_quilt : QuiltBlock :=
  { size := quilt_size,
    shaded_area := 1 }

/-- Theorem stating that the shaded area of the patterned quilt is 1/3 of the total area -/
theorem shaded_area_is_one_third (q : QuiltBlock) (h : q = patterned_quilt) :
  q.shaded_area / (q.size * q.size : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_one_third_l3015_301591


namespace NUMINAMATH_CALUDE_sine_phase_shift_l3015_301589

/-- The phase shift of the sine function y = sin(4x + π/2) is π/8 units to the left. -/
theorem sine_phase_shift :
  let f : ℝ → ℝ := λ x ↦ Real.sin (4 * x + π / 2)
  ∃ (φ : ℝ), φ = π / 8 ∧
    ∀ x, f x = Real.sin (4 * (x + φ)) := by
  sorry

end NUMINAMATH_CALUDE_sine_phase_shift_l3015_301589


namespace NUMINAMATH_CALUDE_project_completion_l3015_301572

/-- Represents the work rate of the men (amount of work done per day per person) -/
def work_rate : ℝ := 1

/-- Represents the total amount of work in the project -/
def total_work : ℝ := 1

/-- The number of days it takes the original group to complete the project -/
def original_days : ℕ := 40

/-- The number of days it takes the reduced group to complete the project -/
def reduced_days : ℕ := 50

/-- The number of men removed from the original group -/
def men_removed : ℕ := 5

theorem project_completion (M : ℕ) : 
  (M : ℝ) * work_rate * original_days = total_work ∧
  ((M : ℝ) - men_removed) * work_rate * reduced_days = total_work →
  M = 25 := by
sorry

end NUMINAMATH_CALUDE_project_completion_l3015_301572


namespace NUMINAMATH_CALUDE_f_is_linear_equation_l3015_301554

/-- A linear equation with two variables is of the form ax + by = c, where a, b, and c are constants. -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The specific equation we want to prove is linear. -/
def f (x y : ℝ) : ℝ := 4 * x - 5 * y - 5

theorem f_is_linear_equation : is_linear_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_linear_equation_l3015_301554


namespace NUMINAMATH_CALUDE_turns_result_in_opposite_direction_l3015_301521

/-- Two turns result in opposite direction if they are in the same direction and sum to 180 degrees -/
def opposite_direction (turn1 : ℝ) (turn2 : ℝ) : Prop :=
  (turn1 > 0 ∧ turn2 > 0) ∧ turn1 + turn2 = 180

/-- The specific turns given in the problem -/
def first_turn : ℝ := 53
def second_turn : ℝ := 127

/-- Theorem stating that the given turns result in opposite direction -/
theorem turns_result_in_opposite_direction :
  opposite_direction first_turn second_turn := by
  sorry

#check turns_result_in_opposite_direction

end NUMINAMATH_CALUDE_turns_result_in_opposite_direction_l3015_301521


namespace NUMINAMATH_CALUDE_complement_of_intersection_l3015_301514

def U : Set ℕ := {1,2,3,4,5}
def M : Set ℕ := {1,2,4}
def N : Set ℕ := {3,4,5}

theorem complement_of_intersection :
  (M ∩ N)ᶜ = {1,2,3,5} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l3015_301514


namespace NUMINAMATH_CALUDE_passengers_who_got_off_l3015_301545

theorem passengers_who_got_off (initial : ℕ) (got_on : ℕ) (final : ℕ) : 
  initial = 28 → got_on = 7 → final = 26 → initial + got_on - final = 9 := by
  sorry

end NUMINAMATH_CALUDE_passengers_who_got_off_l3015_301545


namespace NUMINAMATH_CALUDE_intersection_line_passes_through_intersection_points_l3015_301556

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 12 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 4 = 0

/-- The equation of the line passing through the intersection points -/
def intersection_line (x y : ℝ) : Prop := x - 4*y - 4 = 0

/-- Theorem stating that the intersection_line passes through the intersection points of circle1 and circle2 -/
theorem intersection_line_passes_through_intersection_points :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → intersection_line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_passes_through_intersection_points_l3015_301556


namespace NUMINAMATH_CALUDE_caterer_order_l3015_301540

theorem caterer_order (ice_cream_price sundae_price total_price : ℚ) 
  (h1 : ice_cream_price = 0.60)
  (h2 : sundae_price = 1.20)
  (h3 : total_price = 225.00)
  (h4 : ice_cream_price * x + sundae_price * x = total_price) :
  x = 125 :=
by
  sorry

#check caterer_order

end NUMINAMATH_CALUDE_caterer_order_l3015_301540


namespace NUMINAMATH_CALUDE_gwen_bookcase_total_l3015_301532

/-- The number of mystery books on each shelf -/
def mystery_books_per_shelf : Nat := 7

/-- The number of shelves for mystery books -/
def mystery_shelves : Nat := 6

/-- The number of picture books on each shelf -/
def picture_books_per_shelf : Nat := 5

/-- The number of shelves for picture books -/
def picture_shelves : Nat := 4

/-- The number of biographies on each shelf -/
def biography_books_per_shelf : Nat := 3

/-- The number of shelves for biographies -/
def biography_shelves : Nat := 3

/-- The number of sci-fi books on each shelf -/
def scifi_books_per_shelf : Nat := 9

/-- The number of shelves for sci-fi books -/
def scifi_shelves : Nat := 2

/-- The total number of books on Gwen's bookcase -/
def total_books : Nat :=
  mystery_books_per_shelf * mystery_shelves +
  picture_books_per_shelf * picture_shelves +
  biography_books_per_shelf * biography_shelves +
  scifi_books_per_shelf * scifi_shelves

theorem gwen_bookcase_total : total_books = 89 := by
  sorry

end NUMINAMATH_CALUDE_gwen_bookcase_total_l3015_301532


namespace NUMINAMATH_CALUDE_max_rabbits_with_traits_l3015_301569

theorem max_rabbits_with_traits (N : ℕ) 
  (long_ears : ℕ) (jump_far : ℕ) (both_traits : ℕ) :
  (long_ears = 13 ∧ jump_far = 17 ∧ both_traits ≥ 3) →
  (∀ n : ℕ, n > N → ∃ arrangement : ℕ × ℕ × ℕ, 
    arrangement.1 + arrangement.2.1 + arrangement.2.2 = n ∧
    arrangement.1 + arrangement.2.1 = long_ears ∧
    arrangement.1 + arrangement.2.2 = jump_far ∧
    arrangement.1 < both_traits) →
  N = 27 :=
sorry

end NUMINAMATH_CALUDE_max_rabbits_with_traits_l3015_301569


namespace NUMINAMATH_CALUDE_binomial_16_choose_12_l3015_301528

theorem binomial_16_choose_12 : Nat.choose 16 12 = 43680 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_choose_12_l3015_301528


namespace NUMINAMATH_CALUDE_compare_irrational_expressions_l3015_301568

theorem compare_irrational_expressions : 2 * Real.sqrt 2 - Real.sqrt 7 < Real.sqrt 6 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_compare_irrational_expressions_l3015_301568


namespace NUMINAMATH_CALUDE_solve_for_t_l3015_301525

theorem solve_for_t (p : ℝ) (t : ℝ) 
  (h1 : 5 = p * 3^t) 
  (h2 : 45 = p * 9^t) : 
  t = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l3015_301525


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3015_301593

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^2 - 3*x*y + 2*y^2 - z^2 = 39) ∧ 
  (-x^2 + 6*y*z + 2*z^2 = 40) ∧ 
  (x^2 + x*y + 8*z^2 = 96) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3015_301593


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3015_301523

def polynomial (x : ℝ) : ℝ := 5*x^8 + 2*x^7 - 3*x^4 + 4*x^3 - 5*x^2 + 6*x - 20

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, polynomial x = q x * divisor x + 1404 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3015_301523


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3015_301518

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (2, -1) and b = (k, 5/2), then k = -5. -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, -1) →
  b = (k, 5/2) →
  (∃ (t : ℝ), t ≠ 0 ∧ b = t • a) →
  k = -5 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3015_301518


namespace NUMINAMATH_CALUDE_binary_to_decimal_11001101_l3015_301571

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary_digits : List Bool) : ℕ :=
  binary_digits.enum.foldl
    (fun acc (i, b) => acc + (if b then 2^i else 0))
    0

/-- The binary representation of the number we want to convert -/
def binary_number : List Bool := [true, false, true, true, false, false, true, true]

/-- Theorem stating that the decimal equivalent of 11001101 (binary) is 205 -/
theorem binary_to_decimal_11001101 :
  binary_to_decimal (binary_number.reverse) = 205 := by
  sorry

#eval binary_to_decimal (binary_number.reverse)

end NUMINAMATH_CALUDE_binary_to_decimal_11001101_l3015_301571


namespace NUMINAMATH_CALUDE_count_propositions_is_two_l3015_301562

-- Define a type for statements
inductive Statement
| EmptySetProperSubset
| QuadraticInequality
| PerpendicularLinesQuestion
| NaturalNumberEven

-- Define a function to check if a statement is a proposition
def isProposition (s : Statement) : Bool :=
  match s with
  | Statement.EmptySetProperSubset => true
  | Statement.QuadraticInequality => false
  | Statement.PerpendicularLinesQuestion => false
  | Statement.NaturalNumberEven => true

-- Define a function to count propositions
def countPropositions (statements : List Statement) : Nat :=
  statements.filter isProposition |>.length

-- Theorem to prove
theorem count_propositions_is_two :
  let statements := [Statement.EmptySetProperSubset, Statement.QuadraticInequality,
                     Statement.PerpendicularLinesQuestion, Statement.NaturalNumberEven]
  countPropositions statements = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_propositions_is_two_l3015_301562


namespace NUMINAMATH_CALUDE_ladder_slide_l3015_301560

-- Define the ladder and wall setup
def ladder_length : ℝ := 25
def initial_distance : ℝ := 7
def top_slide : ℝ := 4

-- Theorem statement
theorem ladder_slide :
  let initial_height : ℝ := Real.sqrt (ladder_length^2 - initial_distance^2)
  let new_height : ℝ := initial_height - top_slide
  let new_distance : ℝ := Real.sqrt (ladder_length^2 - new_height^2)
  new_distance - initial_distance = 8 := by
  sorry

end NUMINAMATH_CALUDE_ladder_slide_l3015_301560


namespace NUMINAMATH_CALUDE_rate_of_current_l3015_301549

/-- Proves that given a man's downstream speed, upstream speed, and still water speed, 
    the rate of current can be calculated. -/
theorem rate_of_current 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : downstream_speed = 45) 
  (h2 : upstream_speed = 23) 
  (h3 : still_water_speed = 34) : 
  downstream_speed - still_water_speed = 11 := by
  sorry

#check rate_of_current

end NUMINAMATH_CALUDE_rate_of_current_l3015_301549


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3015_301500

/-- Given a quadratic function f(x) = ax² + bx where a > 0 and b > 0,
    and the slope of the tangent line at x = 1 is 2,
    prove that the minimum value of (8a + b) / (ab) is 9 -/
theorem min_value_quadratic (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : 2*a + b = 2) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (8*x + y) / (x*y) ≥ 9) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (8*x + y) / (x*y) = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3015_301500


namespace NUMINAMATH_CALUDE_inequality_proof_l3015_301538

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a)) + (1 / (2 * b)) + (1 / (2 * c)) ≥ (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3015_301538


namespace NUMINAMATH_CALUDE_tour_group_composition_l3015_301598

/-- Represents the number of people in a tour group -/
structure TourGroup where
  total : ℕ
  children : ℕ

/-- Represents the ticket prices -/
structure TicketPrices where
  adult : ℕ
  child : ℕ

/-- The main theorem statement -/
theorem tour_group_composition 
  (group_a group_b : TourGroup) 
  (prices : TicketPrices) : 
  (group_b.total = group_a.total + 4) →
  (group_a.total + group_b.total = 18 * (group_b.total - group_a.total)) →
  (group_b.children = 3 * group_a.children - 2) →
  (prices.adult = 100) →
  (prices.child = prices.adult * 3 / 5) →
  (prices.adult * (group_a.total - group_a.children) + prices.child * group_a.children = 
   prices.adult * (group_b.total - group_b.children) + prices.child * group_b.children) →
  (group_a.total = 34 ∧ group_a.children = 6 ∧ 
   group_b.total = 38 ∧ group_b.children = 16) :=
by sorry

end NUMINAMATH_CALUDE_tour_group_composition_l3015_301598


namespace NUMINAMATH_CALUDE_common_chord_length_l3015_301558

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 2*x - 4*y + 3 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (a b c d : ℝ),
    C₁ a b ∧ C₁ c d ∧ C₂ a b ∧ C₂ c d ∧
    common_chord a b ∧ common_chord c d ∧
    ((a - c)^2 + (b - d)^2) = 11 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l3015_301558


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3015_301537

/-- For a parabola given by the equation y^2 = 10x, the distance from its focus to its directrix is 5. -/
theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), y^2 = 10*x → (∃ (focus_x focus_y directrix_x : ℝ),
    (∀ (point_x point_y : ℝ), point_y^2 = 10*point_x ↔ 
      (point_x - focus_x)^2 + (point_y - focus_y)^2 = (point_x - directrix_x)^2) ∧
    |focus_x - directrix_x| = 5) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3015_301537


namespace NUMINAMATH_CALUDE_fraction_calculation_l3015_301546

theorem fraction_calculation (x y : ℚ) (hx : x = 4/6) (hy : y = 8/12) :
  (6*x + 8*y) / (48*x*y) = 7/16 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3015_301546


namespace NUMINAMATH_CALUDE_square_ratio_proof_l3015_301578

theorem square_ratio_proof (area_ratio : ℚ) :
  area_ratio = 300 / 75 →
  ∃ (a b c : ℕ), 
    (a * Real.sqrt b : ℝ) / c = Real.sqrt area_ratio ∧
    a = 2 ∧ b = 1 ∧ c = 1 ∧
    a + b + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_proof_l3015_301578


namespace NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l3015_301511

theorem diophantine_equation_unique_solution :
  ∀ x y z t : ℤ, x^2 + y^2 + z^2 + t^2 = 2*x*y*z*t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l3015_301511


namespace NUMINAMATH_CALUDE_vector_sum_l3015_301526

def vector1 : ℝ × ℝ × ℝ := (4, -9, 2)
def vector2 : ℝ × ℝ × ℝ := (-1, 16, 5)

theorem vector_sum : 
  (vector1.1 + vector2.1, vector1.2.1 + vector2.2.1, vector1.2.2 + vector2.2.2) = (3, 7, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_l3015_301526


namespace NUMINAMATH_CALUDE_fifteen_exponent_division_l3015_301583

theorem fifteen_exponent_division : (15 : ℕ)^11 / (15 : ℕ)^8 = 3375 := by sorry

end NUMINAMATH_CALUDE_fifteen_exponent_division_l3015_301583


namespace NUMINAMATH_CALUDE_real_part_of_z_z_equals_1_plus_2i_purely_imaginary_l3015_301587

-- Define a complex number z as x + yi
def z (x y : ℝ) : ℂ := Complex.mk x y

-- Statement 1: The real part of z is x
theorem real_part_of_z (x y : ℝ) : (z x y).re = x := by sorry

-- Statement 2: If z = 1 + 2i, then x = 1 and y = 2
theorem z_equals_1_plus_2i (x y : ℝ) : 
  z x y = Complex.mk 1 2 → x = 1 ∧ y = 2 := by sorry

-- Statement 3: When x = 0 and y ≠ 0, z is a purely imaginary number
theorem purely_imaginary (y : ℝ) : 
  y ≠ 0 → (z 0 y).re = 0 ∧ (z 0 y).im ≠ 0 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_z_equals_1_plus_2i_purely_imaginary_l3015_301587


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_same_is_thirteen_l3015_301513

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- Represents the minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFourSame (machine : GumballMachine) : ℕ := 13

/-- Theorem stating that for the given gumball machine configuration, 
    the minimum number of gumballs needed to guarantee four of the same color is 13 -/
theorem min_gumballs_for_four_same_is_thirteen (machine : GumballMachine) 
  (h1 : machine.red = 12)
  (h2 : machine.white = 10)
  (h3 : machine.blue = 9)
  (h4 : machine.green = 8) : 
  minGumballsForFourSame machine = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_four_same_is_thirteen_l3015_301513


namespace NUMINAMATH_CALUDE_point_on_line_l3015_301506

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let A : Point := ⟨0, 3⟩
  let B : Point := ⟨-8, 0⟩
  let C : Point := ⟨16/3, 5⟩
  collinear A B C := by
  sorry


end NUMINAMATH_CALUDE_point_on_line_l3015_301506


namespace NUMINAMATH_CALUDE_silver_dollars_problem_l3015_301544

theorem silver_dollars_problem (chiu phung ha : ℕ) : 
  phung = chiu + 16 →
  ha = phung + 5 →
  chiu + phung + ha = 205 →
  chiu = 56 := by
  sorry

end NUMINAMATH_CALUDE_silver_dollars_problem_l3015_301544


namespace NUMINAMATH_CALUDE_problem_statement_l3015_301552

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -9)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 10) :
  b / (a + b) + c / (b + c) + a / (c + a) = 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3015_301552


namespace NUMINAMATH_CALUDE_janice_overtime_earnings_l3015_301567

/-- Represents Janice's work schedule and earnings --/
structure WorkSchedule where
  regularDays : ℕ
  regularPayPerDay : ℕ
  overtimeShifts : ℕ
  totalEarnings : ℕ

/-- Calculates the extra amount earned per overtime shift --/
def extraPerOvertimeShift (w : WorkSchedule) : ℕ :=
  (w.totalEarnings - w.regularDays * w.regularPayPerDay) / w.overtimeShifts

/-- Theorem stating that Janice's extra earnings per overtime shift is $15 --/
theorem janice_overtime_earnings :
  let w : WorkSchedule := {
    regularDays := 5,
    regularPayPerDay := 30,
    overtimeShifts := 3,
    totalEarnings := 195
  }
  extraPerOvertimeShift w = 15 := by
  sorry

end NUMINAMATH_CALUDE_janice_overtime_earnings_l3015_301567


namespace NUMINAMATH_CALUDE_partition_properties_l3015_301595

/-- P k l n is the number of partitions of n into no more than k parts, each not exceeding l -/
def P (k l n : ℕ) : ℕ := sorry

/-- The four properties of the partition function P -/
theorem partition_properties (k l n : ℕ) :
  (P k l n - P k (l-1) n = P (k-1) l (n-l)) ∧
  (P k l n - P (k-1) l n = P k (l-1) (n-k)) ∧
  (P k l n = P l k n) ∧
  (P k l n = P k l (k*l-n)) := by sorry

end NUMINAMATH_CALUDE_partition_properties_l3015_301595


namespace NUMINAMATH_CALUDE_frog_jumped_farther_l3015_301516

/-- The distance jumped by the grasshopper in inches -/
def grasshopper_jump : ℕ := 9

/-- The distance jumped by the frog in inches -/
def frog_jump : ℕ := 12

/-- The difference between the frog's jump and the grasshopper's jump -/
def jump_difference : ℕ := frog_jump - grasshopper_jump

/-- Theorem stating that the frog jumped 3 inches farther than the grasshopper -/
theorem frog_jumped_farther : jump_difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_frog_jumped_farther_l3015_301516


namespace NUMINAMATH_CALUDE_stamps_received_l3015_301517

/-- Given Simon's initial and final stamp counts, prove he received 27 stamps from friends -/
theorem stamps_received (initial_stamps final_stamps : ℕ) 
  (h1 : initial_stamps = 34)
  (h2 : final_stamps = 61) :
  final_stamps - initial_stamps = 27 := by
  sorry

end NUMINAMATH_CALUDE_stamps_received_l3015_301517


namespace NUMINAMATH_CALUDE_remainder_3_153_mod_8_l3015_301559

theorem remainder_3_153_mod_8 : 3^153 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_153_mod_8_l3015_301559


namespace NUMINAMATH_CALUDE_whale_third_hour_consumption_l3015_301551

/-- Represents the whale's plankton consumption during a feeding frenzy -/
def WhaleFeedingFrenzy (x : ℕ) : Prop :=
  let first_hour := x
  let second_hour := x + 3
  let third_hour := x + 6
  let fourth_hour := x + 9
  let fifth_hour := x + 12
  (first_hour + second_hour + third_hour + fourth_hour + fifth_hour = 450) ∧
  (third_hour = 90)

/-- Theorem stating that the whale consumes 90 kilos in the third hour -/
theorem whale_third_hour_consumption : ∃ x : ℕ, WhaleFeedingFrenzy x := by
  sorry

end NUMINAMATH_CALUDE_whale_third_hour_consumption_l3015_301551


namespace NUMINAMATH_CALUDE_inequality_proof_l3015_301592

theorem inequality_proof (b : ℝ) : (3*b - 1)*(4*b + 1) > (2*b + 1)*(5*b - 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3015_301592


namespace NUMINAMATH_CALUDE_angstadt_seniors_l3015_301547

/-- Mr. Angstadt's class enrollment problem -/
theorem angstadt_seniors (total_students : ℕ) 
  (stats_percent geometry_percent : ℚ)
  (stats_calc_overlap geometry_calc_overlap : ℚ)
  (stats_senior_percent geometry_senior_percent calc_senior_percent : ℚ)
  (h1 : total_students = 240)
  (h2 : stats_percent = 45/100)
  (h3 : geometry_percent = 35/100)
  (h4 : stats_calc_overlap = 10/100)
  (h5 : geometry_calc_overlap = 5/100)
  (h6 : stats_senior_percent = 90/100)
  (h7 : geometry_senior_percent = 60/100)
  (h8 : calc_senior_percent = 80/100) :
  ∃ (senior_count : ℕ), senior_count = 161 := by
sorry


end NUMINAMATH_CALUDE_angstadt_seniors_l3015_301547


namespace NUMINAMATH_CALUDE_min_expression_bound_l3015_301570

theorem min_expression_bound (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  min
    (min (x^2 + x*y + y^2) (x^2 + x*(y-1) + (y-1)^2))
    (min ((x-1)^2 + (x-1)*y + y^2) ((x-1)^2 + (x-1)*(y-1) + (y-1)^2))
  ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_min_expression_bound_l3015_301570


namespace NUMINAMATH_CALUDE_reflect_P_across_y_axis_l3015_301509

/-- Reflects a point across the y-axis in a 2D Cartesian coordinate system -/
def reflect_across_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, 3)

/-- Theorem stating that reflecting P(-2,3) across the y-axis results in (2,3) -/
theorem reflect_P_across_y_axis :
  reflect_across_y_axis P = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_across_y_axis_l3015_301509


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l3015_301507

theorem shopkeeper_profit_percentage 
  (theft_percentage : ℝ) 
  (loss_percentage : ℝ) 
  (profit_percentage : ℝ) : 
  theft_percentage = 60 → 
  loss_percentage = 56 → 
  (1 - theft_percentage / 100) * (1 + profit_percentage / 100) = 1 - loss_percentage / 100 → 
  profit_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l3015_301507


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3015_301577

theorem remainder_divisibility (N : ℤ) : 
  (N % 779 = 47) → (N % 19 = 9) := by
sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3015_301577


namespace NUMINAMATH_CALUDE_cubic_and_square_sum_l3015_301574

theorem cubic_and_square_sum (x y : ℝ) : 
  x + y = 12 → xy = 20 → (x^3 + y^3 = 1008 ∧ x^2 + y^2 = 104) := by
  sorry

end NUMINAMATH_CALUDE_cubic_and_square_sum_l3015_301574


namespace NUMINAMATH_CALUDE_tim_seashells_l3015_301533

/-- The number of seashells Tim initially found -/
def initial_seashells : ℕ := 679

/-- The number of seashells Tim gave away -/
def seashells_given_away : ℕ := 172

/-- The number of seashells Tim has now -/
def remaining_seashells : ℕ := initial_seashells - seashells_given_away

theorem tim_seashells : remaining_seashells = 507 := by
  sorry

end NUMINAMATH_CALUDE_tim_seashells_l3015_301533


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3015_301512

theorem complex_fraction_equality : ∀ (i : ℂ), i^2 = -1 → (5*i)/(1+2*i) = 2+i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3015_301512


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3015_301531

def f (x : ℝ) := x^2

theorem f_satisfies_conditions :
  (∀ x y, x < y ∧ y < -1 → f x > f y) ∧
  (∀ x, f x = f (-x)) ∧
  (∃ m, ∀ x, f m ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3015_301531


namespace NUMINAMATH_CALUDE_perpendicular_parallel_imply_perpendicular_l3015_301575

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the relations
def parallel (l₁ l₂ : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l₁ l₂ : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define non-coincident
def non_coincident_lines (l₁ l₂ : Line) : Prop := sorry
def non_coincident_planes (p₁ p₂ : Plane) : Prop := sorry

theorem perpendicular_parallel_imply_perpendicular 
  (a b : Line) (α : Plane) 
  (h_non_coincident_lines : non_coincident_lines a b)
  (h_non_coincident_planes : non_coincident_planes α β)
  (h1 : perpendicular_line_plane a α) 
  (h2 : parallel_line_plane b α) : 
  perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_imply_perpendicular_l3015_301575


namespace NUMINAMATH_CALUDE_probability_select_leaders_l3015_301586

def club_sizes : List Nat := [6, 8, 9]

def num_clubs : Nat := 3

def num_selected : Nat := 4

def num_co_presidents : Nat := 2

def num_vice_presidents : Nat := 1

theorem probability_select_leaders (club_sizes : List Nat) 
  (h1 : club_sizes = [6, 8, 9]) 
  (h2 : num_clubs = 3) 
  (h3 : num_selected = 4) 
  (h4 : num_co_presidents = 2) 
  (h5 : num_vice_presidents = 1) : 
  (1 / num_clubs) * (club_sizes.map (λ n => Nat.choose (n - (num_co_presidents + num_vice_presidents)) 1 / Nat.choose n num_selected)).sum = 67 / 630 := by
  sorry

end NUMINAMATH_CALUDE_probability_select_leaders_l3015_301586


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3015_301504

/-- Proves that given an income of 15000 from an 80% stock and an investment of 37500,
    the price of the stock is 50% of its face value. -/
theorem stock_price_calculation 
  (income : ℝ) 
  (investment : ℝ) 
  (yield : ℝ) 
  (h1 : income = 15000) 
  (h2 : investment = 37500) 
  (h3 : yield = 80) : 
  (income * 100 / (investment * yield)) = 0.5 := by
sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l3015_301504


namespace NUMINAMATH_CALUDE_system_solution_l3015_301579

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, x - 3*y = k + 2 ∧ x - y = 4 ∧ 3*x + y = -8) → k = 12 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3015_301579


namespace NUMINAMATH_CALUDE_roots_of_polynomials_l3015_301505

theorem roots_of_polynomials (α : ℂ) : 
  α^2 + α - 1 = 0 → α^3 - 2*α + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomials_l3015_301505


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3015_301580

theorem triangle_angle_proof (a b : ℝ) (B : ℝ) (A : ℝ) : 
  a = 4 → b = 4 * Real.sqrt 2 → B = π/4 → A = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3015_301580


namespace NUMINAMATH_CALUDE_digits_of_2_pow_120_l3015_301582

theorem digits_of_2_pow_120 (h : ∃ n : ℕ, 10^60 ≤ 2^200 ∧ 2^200 < 10^61) :
  ∃ n : ℕ, 10^36 ≤ 2^120 ∧ 2^120 < 10^37 :=
sorry

end NUMINAMATH_CALUDE_digits_of_2_pow_120_l3015_301582


namespace NUMINAMATH_CALUDE_expression_evaluation_l3015_301566

theorem expression_evaluation : (100 - (5000 - 500)) * (5000 - (500 - 100)) = -20240000 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3015_301566


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l3015_301565

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6 + c

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (a b c : ℕ) : ℕ := a * 5^2 + b * 5 + c

theorem base_conversion_subtraction :
  base6ToBase10 3 5 4 - base5ToBase10 2 3 1 = 76 := by sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l3015_301565


namespace NUMINAMATH_CALUDE_special_triangle_properties_l3015_301590

/-- An acute triangle ABC with specific properties -/
structure SpecialTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Properties of the triangle
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  law_of_sines : a / Real.sin A = b / Real.sin B
  special_relation : 2 * a * Real.sin B = Real.sqrt 3 * b
  side_a : a = Real.sqrt 7
  side_c : c = 2

/-- The main theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  t.A = π/3 ∧ 
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l3015_301590


namespace NUMINAMATH_CALUDE_power_inequality_l3015_301543

theorem power_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^6 + b^6 ≥ a*b*(a^4 + b^4) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3015_301543


namespace NUMINAMATH_CALUDE_work_completion_time_l3015_301550

theorem work_completion_time (b c total_time : ℝ) (total_payment c_payment : ℕ) 
  (hb : b = 8)
  (hc : c = 3)
  (htotal_payment : total_payment = 3680)
  (hc_payment : c_payment = 460) :
  ∃ a : ℝ, a = 24 / 5 ∧ 1 / a + 1 / b = 1 / c := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3015_301550


namespace NUMINAMATH_CALUDE_equation_solution_difference_l3015_301508

theorem equation_solution_difference : ∃ (r s : ℝ),
  (∀ x, (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3 ↔ (x = r ∨ x = s)) ∧
  r > s ∧
  r - s = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l3015_301508


namespace NUMINAMATH_CALUDE_inequality_for_elements_in_M_l3015_301542

-- Define the set M
def M : Set ℝ := {x : ℝ | -1/2 < x ∧ x < 1}

-- State the theorem
theorem inequality_for_elements_in_M (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  |a - b| < |1 - a * b| := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_elements_in_M_l3015_301542


namespace NUMINAMATH_CALUDE_inequality_proof_l3015_301530

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  a * Real.sqrt b + b * Real.sqrt c + c * Real.sqrt a ≤ 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3015_301530


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3015_301555

/-- Given the equation (x-2)^2 + √(y+1) = 0, prove that the point (x,y) lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (x y : ℝ) (h : (x - 2)^2 + Real.sqrt (y + 1) = 0) :
  x > 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3015_301555


namespace NUMINAMATH_CALUDE_vertical_angles_are_equal_equal_angles_are_vertical_converse_of_vertical_angles_are_equal_l3015_301536

/-- Definition of vertical angles -/
def VerticalAngles (α β : Angle) : Prop := sorry

/-- The original proposition -/
theorem vertical_angles_are_equal (α β : Angle) : 
  VerticalAngles α β → α = β := sorry

/-- The converse proposition -/
theorem equal_angles_are_vertical (α β : Angle) : 
  α = β → VerticalAngles α β := sorry

/-- Theorem stating that the converse of "Vertical angles are equal" 
    is "Angles that are equal are vertical angles" -/
theorem converse_of_vertical_angles_are_equal :
  (∀ α β : Angle, VerticalAngles α β → α = β) ↔ 
  (∀ α β : Angle, α = β → VerticalAngles α β) :=
sorry

end NUMINAMATH_CALUDE_vertical_angles_are_equal_equal_angles_are_vertical_converse_of_vertical_angles_are_equal_l3015_301536


namespace NUMINAMATH_CALUDE_inequality_proof_l3015_301553

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  a^2 + a*b + b^2 ≤ 3*(a - Real.sqrt (a*b) + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3015_301553
