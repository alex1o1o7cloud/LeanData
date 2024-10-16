import Mathlib

namespace NUMINAMATH_CALUDE_company_employee_reduction_l1810_181043

theorem company_employee_reduction (reduction_percentage : ℝ) (final_employees : ℕ) : 
  reduction_percentage = 14 →
  final_employees = 195 →
  round (final_employees / (1 - reduction_percentage / 100)) = 227 :=
by sorry

end NUMINAMATH_CALUDE_company_employee_reduction_l1810_181043


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l1810_181077

/-- A proportional function where y increases as x increases -/
structure IncreasingProportionalFunction where
  k : ℝ
  increasing : ∀ x₁ x₂, x₁ < x₂ → k * x₁ < k * x₂

/-- The point (√3, k) is in the first quadrant for an increasing proportional function -/
theorem point_in_first_quadrant (f : IncreasingProportionalFunction) :
  f.k > 0 ∧ Real.sqrt 3 > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l1810_181077


namespace NUMINAMATH_CALUDE_inequality_preservation_l1810_181098

theorem inequality_preservation (a b : ℝ) (h : a > b) : 3 * a > 3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1810_181098


namespace NUMINAMATH_CALUDE_tangent_slope_range_implies_y_coordinate_range_l1810_181073

/-- The curve C defined by y = x^2 - x + 1 -/
def C : ℝ → ℝ := λ x => x^2 - x + 1

/-- The derivative of C -/
def C' : ℝ → ℝ := λ x => 2*x - 1

theorem tangent_slope_range_implies_y_coordinate_range :
  ∀ x y : ℝ,
  y = C x →
  -1 ≤ C' x ∧ C' x ≤ 3 →
  3/4 ≤ y ∧ y ≤ 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_range_implies_y_coordinate_range_l1810_181073


namespace NUMINAMATH_CALUDE_intersection_count_7_intersection_count_21_l1810_181036

-- Define the line equation
def line_equation (k : ℝ) (x y : ℝ) : Prop := k * x + y + k^3 = 0

-- Define the set of k values for the first case
def k_values_7 : Set ℝ := {0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9}

-- Define the set of k values for the second case
def k_values_21 : Set ℝ := {x : ℝ | ∃ n : ℤ, -10 ≤ n ∧ n ≤ 10 ∧ x = n / 10}

-- Define the function to count intersection points
noncomputable def count_intersections (k_values : Set ℝ) : ℕ := sorry

-- Theorem for the first case
theorem intersection_count_7 : 
  count_intersections k_values_7 = 11 := by sorry

-- Theorem for the second case
theorem intersection_count_21 :
  count_intersections k_values_21 = 110 := by sorry

end NUMINAMATH_CALUDE_intersection_count_7_intersection_count_21_l1810_181036


namespace NUMINAMATH_CALUDE_possible_polynomials_g_l1810_181079

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property that g must satisfy
def satisfies_condition (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 + 12 * x + 4

-- Theorem statement
theorem possible_polynomials_g :
  ∀ g : ℝ → ℝ, satisfies_condition g ↔ (∀ x, g x = 3 * x + 2 ∨ g x = -3 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_possible_polynomials_g_l1810_181079


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_reverse_square_l1810_181029

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a number is a perfect square -/
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Theorem stating that 19 is the smallest two-digit prime number
    whose reverse is a perfect square -/
theorem smallest_two_digit_prime_reverse_square :
  (∀ n : ℕ, 10 ≤ n ∧ n < 19 ∧ is_prime n → ¬(is_square (reverse_digits n))) ∧
  (19 ≤ 99 ∧ is_prime 19 ∧ is_square (reverse_digits 19)) :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_reverse_square_l1810_181029


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1810_181062

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ : ℚ) = 0.56) ∧ x = 56/99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1810_181062


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l1810_181017

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l1810_181017


namespace NUMINAMATH_CALUDE_frequency_not_always_equal_probability_l1810_181006

/-- A random simulation for selecting a group leader from 6 students -/
structure GroupLeaderSelection where
  total_students : ℕ
  total_trials : ℕ
  successful_trials : ℕ

/-- The frequency of a student being selected in the simulation -/
def frequency (s : GroupLeaderSelection) : ℚ :=
  s.successful_trials / s.total_trials

/-- The theoretical probability of a student being selected -/
def theoretical_probability (s : GroupLeaderSelection) : ℚ :=
  1 / s.total_students

/-- Theorem stating that the frequency does not necessarily equal the theoretical probability -/
theorem frequency_not_always_equal_probability (s : GroupLeaderSelection) :
  ¬ (∀ s : GroupLeaderSelection, s.total_students = 6 → frequency s = theoretical_probability s) :=
sorry

end NUMINAMATH_CALUDE_frequency_not_always_equal_probability_l1810_181006


namespace NUMINAMATH_CALUDE_simplest_fraction_of_decimal_l1810_181069

theorem simplest_fraction_of_decimal (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 0.428125 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.428125 → a * d ≤ b * c →
  a = 137 ∧ b = 320 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_of_decimal_l1810_181069


namespace NUMINAMATH_CALUDE_circle_B_radius_l1810_181025

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def internally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius - c2.radius)^2

def congruent (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

def passes_through_center (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = c1.radius^2

theorem circle_B_radius
  (A B C D E : Circle)
  (h1 : externally_tangent A B)
  (h2 : externally_tangent A C)
  (h3 : externally_tangent A E)
  (h4 : externally_tangent B C)
  (h5 : externally_tangent B E)
  (h6 : externally_tangent C E)
  (h7 : internally_tangent A D)
  (h8 : internally_tangent B D)
  (h9 : internally_tangent C D)
  (h10 : internally_tangent E D)
  (h11 : congruent B C)
  (h12 : congruent A E)
  (h13 : A.radius = 2)
  (h14 : passes_through_center A D)
  : B.radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_B_radius_l1810_181025


namespace NUMINAMATH_CALUDE_h_is_smallest_l1810_181041

/-- Definition of the partition property for h(n) -/
def has_partition_property (h n : ℕ) : Prop :=
  ∀ (A : Fin n → Set ℕ), 
    (∀ i j, i ≠ j → A i ∩ A j = ∅) → 
    (⋃ i, A i) = Finset.range h →
    ∃ (a x y : ℕ), 
      1 ≤ x ∧ x ≤ y ∧ y ≤ h ∧
      ∃ i, {a + x, a + y, a + x + y} ⊆ A i

/-- The function h(n) -/
def h (n : ℕ) : ℕ := Nat.choose n (n / 2)

/-- Main theorem: h(n) is the smallest positive integer satisfying the partition property -/
theorem h_is_smallest (n : ℕ) (hn : 0 < n) : 
  has_partition_property (h n) n ∧ 
  ∀ m, 0 < m ∧ m < h n → ¬has_partition_property m n :=
sorry

end NUMINAMATH_CALUDE_h_is_smallest_l1810_181041


namespace NUMINAMATH_CALUDE_no_real_solution_cos_sin_l1810_181008

theorem no_real_solution_cos_sin : ¬∃ (x : ℝ), (Real.cos x = 1/2) ∧ (Real.sin x = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_cos_sin_l1810_181008


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1810_181046

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1810_181046


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1810_181011

/-- Given a polar equation ρ = 4sin(θ), prove its equivalence to the Cartesian equation x² + (y-2)² = 4 -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1810_181011


namespace NUMINAMATH_CALUDE_quarter_circle_arc_sum_limit_l1810_181086

/-- The limit of the sum of quarter-circle arcs approaches a quarter of the original circle's circumference --/
theorem quarter_circle_arc_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * (D / n) / 4) - π * D / 4| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circle_arc_sum_limit_l1810_181086


namespace NUMINAMATH_CALUDE_find_a_l1810_181032

theorem find_a : ∀ a : ℚ, 
  (∀ y : ℚ, (y + a) / 2 = (2 * y - a) / 3 → y = 5 * a) →
  (∀ x : ℚ, 3 * a - x = x / 2 + 3 → x = 2 * a - 2) →
  (5 * a = (2 * a - 2) - 3) →
  a = -5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_find_a_l1810_181032


namespace NUMINAMATH_CALUDE_circle_radius_with_tangent_and_secant_l1810_181058

/-- A circle with a tangent and a secant drawn from an external point -/
structure CircleWithTangentAndSecant where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the tangent -/
  tangent_length : ℝ
  /-- The length of the internal segment of the secant -/
  secant_internal_length : ℝ
  /-- The tangent and secant are mutually perpendicular -/
  perpendicular : True

/-- Theorem: If a circle has a tangent of length 12 and a secant with internal segment of length 10,
    and the tangent and secant are mutually perpendicular, then the radius of the circle is 13 -/
theorem circle_radius_with_tangent_and_secant 
  (c : CircleWithTangentAndSecant) 
  (h1 : c.tangent_length = 12) 
  (h2 : c.secant_internal_length = 10) :
  c.radius = 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_tangent_and_secant_l1810_181058


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l1810_181050

theorem least_n_satisfying_inequality :
  ∀ n : ℕ+, n < 4 → (1 : ℚ) / n - (1 : ℚ) / (n + 1) ≥ (1 : ℚ) / 15 ∧
  (1 : ℚ) / 4 - (1 : ℚ) / 5 < (1 : ℚ) / 15 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l1810_181050


namespace NUMINAMATH_CALUDE_sfl_entrances_l1810_181010

/-- Given that there are 283 people waiting at each entrance and 1415 people in total,
    prove that the number of entrances is 5. -/
theorem sfl_entrances (people_per_entrance : ℕ) (total_people : ℕ) 
  (h1 : people_per_entrance = 283) 
  (h2 : total_people = 1415) :
  total_people / people_per_entrance = 5 := by
  sorry

end NUMINAMATH_CALUDE_sfl_entrances_l1810_181010


namespace NUMINAMATH_CALUDE_swan_percentage_among_non_ducks_l1810_181054

theorem swan_percentage_among_non_ducks (geese swan heron duck : ℚ) :
  geese = 1/5 →
  swan = 3/10 →
  heron = 1/4 →
  duck = 1/4 →
  geese + swan + heron + duck = 1 →
  swan / (geese + swan + heron) = 2/5 :=
sorry

end NUMINAMATH_CALUDE_swan_percentage_among_non_ducks_l1810_181054


namespace NUMINAMATH_CALUDE_reunion_attendees_l1810_181005

theorem reunion_attendees (total : ℕ) (oates : ℕ) (hall : ℕ) (both : ℕ) : 
  total = 100 → oates = 40 → hall = 70 → 
  total = oates + hall - both → both = 10 := by
sorry

end NUMINAMATH_CALUDE_reunion_attendees_l1810_181005


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_foci_parabola_equation_l1810_181061

-- Define the hyperbola and ellipse
def hyperbola (m : ℝ) := {(x, y) : ℝ × ℝ | x^2 - y^2 = m}
def ellipse := {(x, y) : ℝ × ℝ | 2*x^2 + 3*y^2 = 72}

-- Define the condition of same foci
def same_foci (h : Set (ℝ × ℝ)) (e : Set (ℝ × ℝ)) : Prop := sorry

-- Define a parabola
def parabola (p : ℝ) := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}

-- Define the condition for focus on positive x-axis
def focus_on_positive_x (p : Set (ℝ × ℝ)) : Prop := sorry

-- Define the condition for passing through a point
def passes_through (p : Set (ℝ × ℝ)) (point : ℝ × ℝ) : Prop := point ∈ p

-- Theorem 1
theorem hyperbola_ellipse_foci (m : ℝ) : 
  same_foci (hyperbola m) ellipse → m = 6 := sorry

-- Theorem 2
theorem parabola_equation : 
  ∃ p : ℝ, focus_on_positive_x (parabola p) ∧ 
  passes_through (parabola p) (2, -4) ∧ 
  p = 4 := sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_foci_parabola_equation_l1810_181061


namespace NUMINAMATH_CALUDE_always_on_iff_odd_l1810_181087

/-- Represents the state of a light bulb -/
inductive BulbState
| On
| Off

/-- Represents a configuration of light bulbs -/
def BulbConfiguration (n : ℕ) := Fin n → BulbState

/-- Function to update the state of bulbs according to the given rule -/
def updateBulbs (n : ℕ) (config : BulbConfiguration n) : BulbConfiguration n :=
  sorry

/-- Predicate to check if a configuration has at least one bulb on -/
def hasOnBulb (n : ℕ) (config : BulbConfiguration n) : Prop :=
  sorry

/-- Theorem stating that there exists a configuration that always has at least one bulb on
    if and only if n is odd -/
theorem always_on_iff_odd (n : ℕ) :
  (∃ (initial : BulbConfiguration n), ∀ (t : ℕ), hasOnBulb n ((updateBulbs n)^[t] initial)) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_always_on_iff_odd_l1810_181087


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1810_181051

theorem sum_of_a_and_b (a b : ℝ) : 
  |a - 1/2| + |b + 5| = 0 → a + b = -9/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1810_181051


namespace NUMINAMATH_CALUDE_pablo_puzzle_completion_time_l1810_181007

/-- The number of days Pablo needs to complete all puzzles -/
def days_to_complete_puzzles (
  pieces_per_hour : ℕ
  ) (
  puzzles_300 : ℕ
  ) (
  puzzles_500 : ℕ
  ) (
  max_hours_per_day : ℕ
  ) : ℕ :=
  let total_pieces := puzzles_300 * 300 + puzzles_500 * 500
  let pieces_per_day := pieces_per_hour * max_hours_per_day
  (total_pieces + pieces_per_day - 1) / pieces_per_day

theorem pablo_puzzle_completion_time :
  days_to_complete_puzzles 100 8 5 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_pablo_puzzle_completion_time_l1810_181007


namespace NUMINAMATH_CALUDE_G_20_diamonds_l1810_181059

/-- The number of diamonds in the nth figure of sequence G -/
def G (n : ℕ) : ℕ :=
  2 * n * (n + 1)

theorem G_20_diamonds : G 20 = 840 := by
  sorry

end NUMINAMATH_CALUDE_G_20_diamonds_l1810_181059


namespace NUMINAMATH_CALUDE_rockham_soccer_league_members_l1810_181012

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 4

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 5

/-- The total cost for all members in dollars -/
def total_cost : ℕ := 2366

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 2

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- The cost of equipment for one member -/
def member_cost : ℕ := socks_per_member * sock_cost + 
                       tshirts_per_member * (sock_cost + tshirt_additional_cost)

/-- The number of members in the Rockham Soccer League -/
def number_of_members : ℕ := total_cost / member_cost

theorem rockham_soccer_league_members : number_of_members = 91 := by
  sorry

end NUMINAMATH_CALUDE_rockham_soccer_league_members_l1810_181012


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l1810_181075

theorem cos_2alpha_value (α : Real) (h : Real.tan (π/4 - α) = -1/3) : 
  Real.cos (2*α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l1810_181075


namespace NUMINAMATH_CALUDE_face_value_calculation_l1810_181019

/-- Given a company's dividend rate, an investor's return on investment, and the purchase price of shares, 
    calculate the face value of the shares. -/
theorem face_value_calculation (dividend_rate : ℝ) (roi : ℝ) (purchase_price : ℝ) :
  dividend_rate = 0.185 →
  roi = 0.25 →
  purchase_price = 37 →
  ∃ (face_value : ℝ), face_value * dividend_rate = purchase_price * roi ∧ face_value = 50 := by
  sorry

end NUMINAMATH_CALUDE_face_value_calculation_l1810_181019


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l1810_181096

theorem fraction_meaningful_condition (x : ℝ) :
  (∃ y : ℝ, y = 3 / (x - 1)) ↔ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l1810_181096


namespace NUMINAMATH_CALUDE_grape_rate_proof_l1810_181014

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 70

/-- The weight of grapes purchased in kg -/
def grape_weight : ℝ := 7

/-- The weight of mangoes purchased in kg -/
def mango_weight : ℝ := 9

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 55

/-- The total amount paid -/
def total_paid : ℝ := 985

theorem grape_rate_proof : 
  grape_rate * grape_weight + mango_rate * mango_weight = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_proof_l1810_181014


namespace NUMINAMATH_CALUDE_divisible_by_eight_inductive_step_l1810_181052

theorem divisible_by_eight (n : ℕ) :
  ∃ m : ℤ, 3^(4*n+1) + 5^(2*n+1) = 8 * m :=
by
  sorry

theorem inductive_step (k : ℕ) :
  3^(4*(k+1)+1) + 5^(2*(k+1)+1) = 56 * 3^(4*k+1) + 25 * (3^(4*k+1) + 5^(2*k+1)) :=
by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_inductive_step_l1810_181052


namespace NUMINAMATH_CALUDE_chapters_read_l1810_181028

/-- Represents the number of pages in each chapter of the book --/
def pages_per_chapter : ℕ := 8

/-- Represents the total number of pages Tom read --/
def total_pages_read : ℕ := 24

/-- Theorem stating that the number of chapters Tom read is 3 --/
theorem chapters_read : (total_pages_read / pages_per_chapter : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chapters_read_l1810_181028


namespace NUMINAMATH_CALUDE_first_dog_consumption_l1810_181092

/-- Represents the weekly food consumption of three dogs -/
structure DogFoodConsumption where
  first_dog : ℝ
  second_dog : ℝ
  third_dog : ℝ

/-- The total weekly food consumption of the three dogs -/
def total_consumption (d : DogFoodConsumption) : ℝ :=
  d.first_dog + d.second_dog + d.third_dog

theorem first_dog_consumption :
  ∃ (d : DogFoodConsumption),
    total_consumption d = 15 ∧
    d.second_dog = 2 * d.first_dog ∧
    d.third_dog = 6 ∧
    d.first_dog = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_dog_consumption_l1810_181092


namespace NUMINAMATH_CALUDE_zarnin_battle_station_staffing_l1810_181034

/-- The number of ways to fill positions from a pool of candidates -/
def fill_positions (total_candidates : Nat) (positions_to_fill : Nat) : Nat :=
  List.range positions_to_fill
  |>.map (fun i => total_candidates - i)
  |>.prod

/-- The problem statement -/
theorem zarnin_battle_station_staffing :
  fill_positions 20 5 = 1860480 := by
  sorry

end NUMINAMATH_CALUDE_zarnin_battle_station_staffing_l1810_181034


namespace NUMINAMATH_CALUDE_problem_solution_l1810_181076

def U : Set ℕ := {1, 2, 3, 4, 5}

def A (q : ℤ) : Set ℕ := {x ∈ U | x^2 - 5*x + q = 0}

def B (p : ℤ) : Set ℕ := {x ∈ U | x^2 + p*x + 12 = 0}

theorem problem_solution :
  ∃ (p q : ℤ),
    (U \ A q) ∪ B p = {1, 3, 4, 5} ∧
    p = -7 ∧
    q = 6 ∧
    A q = {2, 3} ∧
    B p = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1810_181076


namespace NUMINAMATH_CALUDE_seaweed_for_human_consumption_l1810_181055

/-- Given that:
  - 400 pounds of seaweed are harvested
  - 50% of seaweed is used for starting fires
  - 150 pounds are fed to livestock
Prove that 25% of the remaining seaweed after starting fires can be eaten by humans -/
theorem seaweed_for_human_consumption 
  (total_seaweed : ℝ) 
  (fire_seaweed_percentage : ℝ) 
  (livestock_seaweed : ℝ) 
  (h1 : total_seaweed = 400)
  (h2 : fire_seaweed_percentage = 0.5)
  (h3 : livestock_seaweed = 150) :
  let remaining_seaweed := total_seaweed * (1 - fire_seaweed_percentage)
  let human_seaweed := remaining_seaweed - livestock_seaweed
  human_seaweed / remaining_seaweed = 0.25 := by
sorry

end NUMINAMATH_CALUDE_seaweed_for_human_consumption_l1810_181055


namespace NUMINAMATH_CALUDE_awards_distribution_theorem_l1810_181074

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 6 awards to 3 students results in 465 ways -/
theorem awards_distribution_theorem :
  distribute_awards 6 3 = 465 :=
by sorry

end NUMINAMATH_CALUDE_awards_distribution_theorem_l1810_181074


namespace NUMINAMATH_CALUDE_mean_of_fractions_l1810_181002

theorem mean_of_fractions (a b : ℚ) (ha : a = 2/3) (hb : b = 4/9) :
  (a + b) / 2 = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_fractions_l1810_181002


namespace NUMINAMATH_CALUDE_firm_employs_100_looms_l1810_181021

/-- Represents the number of looms employed by the textile manufacturing firm. -/
def number_of_looms : ℕ := sorry

/-- The aggregate sales value of the output of the looms in rupees. -/
def aggregate_sales : ℕ := 500000

/-- The monthly manufacturing expenses in rupees. -/
def manufacturing_expenses : ℕ := 150000

/-- The monthly establishment charges in rupees. -/
def establishment_charges : ℕ := 75000

/-- The decrease in profit when one loom breaks down for one month, in rupees. -/
def profit_decrease : ℕ := 3500

theorem firm_employs_100_looms :
  number_of_looms = 100 ∧
  aggregate_sales / number_of_looms - manufacturing_expenses / number_of_looms = profit_decrease :=
sorry

end NUMINAMATH_CALUDE_firm_employs_100_looms_l1810_181021


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1810_181057

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of the expression (35)^7 + (93)^45 -/
def expression : ℕ := 35^7 + 93^45

/-- Theorem stating that the units digit of (35)^7 + (93)^45 is 8 -/
theorem units_digit_of_expression : unitsDigit expression = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1810_181057


namespace NUMINAMATH_CALUDE_nested_square_root_value_l1810_181030

theorem nested_square_root_value : ∃ y : ℝ, y > 0 ∧ y = Real.sqrt (3 - y) ∧ y = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l1810_181030


namespace NUMINAMATH_CALUDE_china_forex_reserves_scientific_notation_l1810_181049

-- Define the original amount in billions of US dollars
def original_amount : ℚ := 10663

-- Define the number of significant figures to retain
def significant_figures : ℕ := 3

-- Define the function to convert to scientific notation with given significant figures
def to_scientific_notation (x : ℚ) (sig_figs : ℕ) : ℚ × ℤ := sorry

-- Theorem statement
theorem china_forex_reserves_scientific_notation :
  let (mantissa, exponent) := to_scientific_notation (original_amount * 1000000000) significant_figures
  mantissa = 1.07 ∧ exponent = 12 := by sorry

end NUMINAMATH_CALUDE_china_forex_reserves_scientific_notation_l1810_181049


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_l1810_181067

theorem unique_solution_lcm_gcd : 
  ∃! n : ℕ+, Nat.lcm n 180 = Nat.gcd n 180 + 630 ∧ n = 360 := by sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_l1810_181067


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l1810_181083

theorem max_value_of_sum_of_roots (a b c : ℝ) 
  (sum_eq_two : a + b + c = 2)
  (a_ge_neg_one : a ≥ -1)
  (b_ge_neg_two : b ≥ -2)
  (c_ge_neg_three : c ≥ -3) :
  ∃ (max : ℝ), max = 4 * Real.sqrt 6 ∧
    ∀ (x : ℝ), x = Real.sqrt (4 * a + 2) + Real.sqrt (4 * b + 8) + Real.sqrt (4 * c + 14) → x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l1810_181083


namespace NUMINAMATH_CALUDE_inverse_sum_theorem_l1810_181094

noncomputable def g (x : ℝ) : ℝ :=
  if x < 15 then x + 5 else 3 * x - 1

theorem inverse_sum_theorem : 
  (Function.invFun g) 10 + (Function.invFun g) 50 = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_theorem_l1810_181094


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1810_181004

theorem complex_equation_sum (a b : ℝ) :
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1810_181004


namespace NUMINAMATH_CALUDE_expression_evaluation_l1810_181080

theorem expression_evaluation : 4 * (8 - 2)^2 - 6 = 138 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1810_181080


namespace NUMINAMATH_CALUDE_company_merger_profit_l1810_181013

theorem company_merger_profit (x : ℝ) (h1 : 0.4 * x = 60000) (h2 : 0 < x) : 0.6 * x = 90000 := by
  sorry

end NUMINAMATH_CALUDE_company_merger_profit_l1810_181013


namespace NUMINAMATH_CALUDE_bill_red_mushrooms_l1810_181066

/-- Proves that Bill gathered 12 red mushrooms based on the given conditions --/
theorem bill_red_mushrooms :
  ∀ (red_mushrooms : ℕ) 
    (brown_mushrooms : ℕ)
    (blue_mushrooms : ℕ)
    (green_mushrooms : ℕ)
    (white_spotted_mushrooms : ℕ),
  brown_mushrooms = 6 →
  blue_mushrooms = 6 →
  green_mushrooms = 14 →
  white_spotted_mushrooms = 17 →
  (blue_mushrooms / 2 : ℚ) + brown_mushrooms + (2 * red_mushrooms / 3 : ℚ) = white_spotted_mushrooms →
  red_mushrooms = 12 := by
sorry

end NUMINAMATH_CALUDE_bill_red_mushrooms_l1810_181066


namespace NUMINAMATH_CALUDE_squares_in_35x2_grid_l1810_181089

/-- The number of squares in a rectangular grid --/
def count_squares (length width : ℕ) : ℕ :=
  -- Count 1x1 squares
  length * width +
  -- Count 2x2 squares
  (length - 1) * (width - 1)

/-- Theorem: The number of squares in a 35x2 grid is 104 --/
theorem squares_in_35x2_grid :
  count_squares 35 2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_35x2_grid_l1810_181089


namespace NUMINAMATH_CALUDE_library_books_total_l1810_181082

/-- The total number of books obtained from the library -/
def total_books (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 54 initial books and 23 additional books, the total is 77 -/
theorem library_books_total : total_books 54 23 = 77 := by
  sorry

end NUMINAMATH_CALUDE_library_books_total_l1810_181082


namespace NUMINAMATH_CALUDE_car_journey_time_l1810_181003

theorem car_journey_time (distance : ℝ) (new_speed : ℝ) (initial_time : ℝ) : 
  distance = 360 →
  new_speed = 40 →
  distance / new_speed = (3/2) * initial_time →
  initial_time = 6 := by
sorry

end NUMINAMATH_CALUDE_car_journey_time_l1810_181003


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l1810_181097

theorem quadratic_inequality_always_nonnegative (x : ℝ) : x^2 + 3 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l1810_181097


namespace NUMINAMATH_CALUDE_sin_derivative_l1810_181063

open Real

theorem sin_derivative (x : ℝ) : deriv sin x = cos x := by sorry

end NUMINAMATH_CALUDE_sin_derivative_l1810_181063


namespace NUMINAMATH_CALUDE_angle_sum_is_180_l1810_181020

-- Define angles as real numbers (representing degrees)
variable (angle1 angle2 angle3 : ℝ)

-- Define the properties of vertical angles and supplementary angles
def vertical_angles (a b : ℝ) : Prop := a = b
def supplementary_angles (a b : ℝ) : Prop := a + b = 180

-- State the theorem
theorem angle_sum_is_180 
  (h1 : vertical_angles angle1 angle2) 
  (h2 : supplementary_angles angle2 angle3) : 
  angle1 + angle3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_180_l1810_181020


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1810_181039

theorem polynomial_division_remainder : 
  let p (x : ℚ) := x^4 - 4*x^3 + 13*x^2 - 14*x + 4
  let d (x : ℚ) := x^2 - 3*x + 13/3
  let q (x : ℚ) := x^2 - x + 10/3
  let r (x : ℚ) := 2*x + 16/9
  ∀ x, p x = d x * q x + r x := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1810_181039


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l1810_181090

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l1810_181090


namespace NUMINAMATH_CALUDE_additional_peaches_l1810_181035

theorem additional_peaches (initial_peaches total_peaches : ℕ) 
  (h1 : initial_peaches = 20)
  (h2 : total_peaches = 45) :
  total_peaches - initial_peaches = 25 := by
  sorry

end NUMINAMATH_CALUDE_additional_peaches_l1810_181035


namespace NUMINAMATH_CALUDE_leticia_dish_cost_is_10_l1810_181040

/-- The cost of Leticia's dish -/
def leticia_dish_cost : ℝ := 10

/-- The cost of Scarlett's dish -/
def scarlett_dish_cost : ℝ := 13

/-- The cost of Percy's dish -/
def percy_dish_cost : ℝ := 17

/-- The tip rate -/
def tip_rate : ℝ := 0.10

/-- The total tip amount -/
def total_tip : ℝ := 4

/-- Theorem stating that Leticia's dish costs $10 given the conditions -/
theorem leticia_dish_cost_is_10 :
  leticia_dish_cost = 10 :=
by sorry

end NUMINAMATH_CALUDE_leticia_dish_cost_is_10_l1810_181040


namespace NUMINAMATH_CALUDE_stack_height_probability_l1810_181064

/-- Represents the possible heights of a crate -/
inductive CrateHeight : Type
  | Two : CrateHeight
  | Three : CrateHeight
  | Five : CrateHeight

/-- The number of crates in the stack -/
def numCrates : ℕ := 5

/-- The target height of the stack -/
def targetHeight : ℕ := 16

/-- Calculates the total number of possible arrangements -/
def totalArrangements : ℕ := 3^numCrates

/-- Calculates the number of valid arrangements that sum to the target height -/
def validArrangements : ℕ := 20

/-- The probability of achieving the target height -/
def probabilityTargetHeight : ℚ := validArrangements / totalArrangements

theorem stack_height_probability :
  probabilityTargetHeight = 20 / 243 := by sorry

end NUMINAMATH_CALUDE_stack_height_probability_l1810_181064


namespace NUMINAMATH_CALUDE_intersection_value_l1810_181070

theorem intersection_value (m : ℝ) (B : Set ℝ) : 
  ({1, m - 2} : Set ℝ) ∩ B = {2} → m = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_value_l1810_181070


namespace NUMINAMATH_CALUDE_cycle_iff_minimal_cut_l1810_181095

-- Define a planar multigraph
structure PlanarMultigraph where
  V : Type*  -- Vertex set
  E : Type*  -- Edge set
  is_planar : Bool
  is_connected : Bool

-- Define a dual graph
def DualGraph (G : PlanarMultigraph) : PlanarMultigraph := sorry

-- Define a cycle in a graph
def is_cycle (G : PlanarMultigraph) (E : Set G.E) : Prop := sorry

-- Define a cut in a graph
def is_cut (G : PlanarMultigraph) (E : Set G.E) : Prop := sorry

-- Define a minimal cut
def is_minimal_cut (G : PlanarMultigraph) (E : Set G.E) : Prop := sorry

-- Define the dual edge set
def dual_edge_set (G : PlanarMultigraph) (E : Set G.E) : Set (DualGraph G).E := sorry

-- Main theorem
theorem cycle_iff_minimal_cut (G : PlanarMultigraph) (E : Set G.E) :
  is_cycle G E ↔ is_minimal_cut (DualGraph G) (dual_edge_set G E) := by sorry

end NUMINAMATH_CALUDE_cycle_iff_minimal_cut_l1810_181095


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1810_181056

theorem negative_fraction_comparison : -3/5 > -5/7 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1810_181056


namespace NUMINAMATH_CALUDE_line_equation_l1810_181084

/-- A line passing through (1,1) with y-intercept 3 has equation 2x + y - 3 = 0 -/
theorem line_equation (x y : ℝ) : 
  (2 * 1 + 1 - 3 = 0) ∧ 
  (2 * 0 + 3 - 3 = 0) ∧ 
  (∀ x y, y = -2 * x + 3) → 
  2 * x + y - 3 = 0 := by sorry

end NUMINAMATH_CALUDE_line_equation_l1810_181084


namespace NUMINAMATH_CALUDE_airsickness_gender_related_l1810_181065

/-- Represents the contingency table data for airsickness and gender --/
structure AirsicknessData :=
  (male_sick : ℕ)
  (male_not_sick : ℕ)
  (female_sick : ℕ)
  (female_not_sick : ℕ)

/-- Calculates the K² value for the given airsickness data --/
def calculate_k_squared (data : AirsicknessData) : ℚ :=
  let n := data.male_sick + data.male_not_sick + data.female_sick + data.female_not_sick
  let ad := data.male_sick * data.female_not_sick
  let bc := data.male_not_sick * data.female_sick
  let numerator := n * (ad - bc) * (ad - bc)
  let denominator := (data.male_sick + data.male_not_sick) * 
                     (data.female_sick + data.female_not_sick) * 
                     (data.male_sick + data.female_sick) * 
                     (data.male_not_sick + data.female_not_sick)
  numerator / denominator

/-- Theorem stating that the K² value for the given data indicates a relationship between airsickness and gender --/
theorem airsickness_gender_related (data : AirsicknessData) 
  (h1 : data.male_sick = 28)
  (h2 : data.male_not_sick = 28)
  (h3 : data.female_sick = 28)
  (h4 : data.female_not_sick = 56) :
  calculate_k_squared data > 3841 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_airsickness_gender_related_l1810_181065


namespace NUMINAMATH_CALUDE_hidden_primes_average_l1810_181060

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def card_sum (visible hidden : ℕ) : ℕ := visible + hidden

theorem hidden_primes_average (visible_1 visible_2 visible_3 hidden_1 hidden_2 hidden_3 : ℕ) 
  (h1 : visible_1 = 42)
  (h2 : visible_2 = 59)
  (h3 : visible_3 = 36)
  (h4 : is_prime hidden_1)
  (h5 : is_prime hidden_2)
  (h6 : is_prime hidden_3)
  (h7 : card_sum visible_1 hidden_1 = card_sum visible_2 hidden_2)
  (h8 : card_sum visible_2 hidden_2 = card_sum visible_3 hidden_3)
  (h9 : hidden_1 ≠ hidden_2 ∧ hidden_2 ≠ hidden_3 ∧ hidden_1 ≠ hidden_3)
  : (hidden_1 + hidden_2 + hidden_3) / 3 = 56 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hidden_primes_average_l1810_181060


namespace NUMINAMATH_CALUDE_total_area_of_triangular_houses_l1810_181045

/-- The total area of three similar triangular houses -/
theorem total_area_of_triangular_houses (base : ℝ) (height : ℝ) (num_houses : ℕ) :
  base = 40 ∧ height = 20 ∧ num_houses = 3 →
  num_houses * (base * height / 2) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_area_of_triangular_houses_l1810_181045


namespace NUMINAMATH_CALUDE_certain_number_problem_l1810_181093

theorem certain_number_problem (x : ℝ) : 4 * (3 * x / 5 - 220) = 320 → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1810_181093


namespace NUMINAMATH_CALUDE_solution_comparison_l1810_181042

theorem solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0)
  (h_sol : -q / p > -q' / p') : q / p < q' / p' := by
  sorry

end NUMINAMATH_CALUDE_solution_comparison_l1810_181042


namespace NUMINAMATH_CALUDE_ball_box_difference_l1810_181068

theorem ball_box_difference : 
  let white_balls : ℕ := 30
  let red_balls : ℕ := 18
  let balls_per_box : ℕ := 6
  let white_boxes := white_balls / balls_per_box
  let red_boxes := red_balls / balls_per_box
  white_boxes - red_boxes = 2 := by
sorry

end NUMINAMATH_CALUDE_ball_box_difference_l1810_181068


namespace NUMINAMATH_CALUDE_min_product_of_squares_plus_one_l1810_181053

/-- The polynomial P(x) = x^4 + ax^3 + bx^2 + cx + d -/
def P (a b c d x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem min_product_of_squares_plus_one (a b c d : ℝ) (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : b - d ≥ 5)
  (h₂ : P a b c d x₁ = 0)
  (h₃ : P a b c d x₂ = 0)
  (h₄ : P a b c d x₃ = 0)
  (h₅ : P a b c d x₄ = 0) :
  (x₁^2 + 1) * (x₂^2 + 1) * (x₃^2 + 1) * (x₄^2 + 1) ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_squares_plus_one_l1810_181053


namespace NUMINAMATH_CALUDE_problem_statement_l1810_181022

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 4) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 74.0625 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1810_181022


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_24_l1810_181099

/-- Calculates the net rate of pay per hour for a driver given specific conditions --/
theorem driver_net_pay_rate (travel_time : ℝ) (travel_speed : ℝ) (car_efficiency : ℝ) 
  (pay_rate : ℝ) (gasoline_cost : ℝ) : ℝ :=
  let total_distance := travel_time * travel_speed
  let gasoline_used := total_distance / car_efficiency
  let earnings := pay_rate * total_distance
  let gasoline_expense := gasoline_cost * gasoline_used
  let net_earnings := earnings - gasoline_expense
  let net_pay_rate := net_earnings / travel_time
  net_pay_rate

/-- Proves that the driver's net rate of pay is $24 per hour under given conditions --/
theorem driver_net_pay_is_24 :
  driver_net_pay_rate 3 50 25 0.60 3.00 = 24 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_24_l1810_181099


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1810_181047

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {3, 4}

-- Theorem statement
theorem intersection_with_complement : A ∩ (U \ B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1810_181047


namespace NUMINAMATH_CALUDE_min_students_with_brown_eyes_and_lunch_box_l1810_181033

/-- Given a class with the following properties:
  * There are 30 students in total
  * 12 students have brown eyes
  * 20 students have a lunch box
  This theorem proves that the minimum number of students
  who have both brown eyes and a lunch box is 2. -/
theorem min_students_with_brown_eyes_and_lunch_box
  (total_students : ℕ)
  (brown_eyes : ℕ)
  (lunch_box : ℕ)
  (h1 : total_students = 30)
  (h2 : brown_eyes = 12)
  (h3 : lunch_box = 20) :
  brown_eyes + lunch_box - total_students ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_students_with_brown_eyes_and_lunch_box_l1810_181033


namespace NUMINAMATH_CALUDE_weight_problem_l1810_181081

/-- Given three weights a, b, and c, prove that their average weights satisfy certain conditions -/
theorem weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 31 →
  (a + b) / 2 = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_weight_problem_l1810_181081


namespace NUMINAMATH_CALUDE_existence_of_special_odd_numbers_l1810_181078

theorem existence_of_special_odd_numbers : ∃ m n : ℕ, 
  Odd m ∧ Odd n ∧ 
  m > 2009 ∧ n > 2009 ∧ 
  (n^2 + 8) % m = 0 ∧ 
  (m^2 + 8) % n = 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_odd_numbers_l1810_181078


namespace NUMINAMATH_CALUDE_first_month_sale_is_6400_l1810_181048

/-- Represents the sales data for a grocer over six months -/
structure GrocerSales where
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ
  average : ℕ

/-- Calculates the sale in the first month given the sales data -/
def firstMonthSale (sales : GrocerSales) : ℕ :=
  6 * sales.average - (sales.month2 + sales.month3 + sales.month4 + sales.month5 + sales.month6)

/-- Theorem stating that the first month's sale is 6400 given the specific sales data -/
theorem first_month_sale_is_6400 (sales : GrocerSales) 
  (h1 : sales.month2 = 7000)
  (h2 : sales.month3 = 6800)
  (h3 : sales.month4 = 7200)
  (h4 : sales.month5 = 6500)
  (h5 : sales.month6 = 5100)
  (h6 : sales.average = 6500) :
  firstMonthSale sales = 6400 := by
  sorry


end NUMINAMATH_CALUDE_first_month_sale_is_6400_l1810_181048


namespace NUMINAMATH_CALUDE_find_number_l1810_181072

theorem find_number (x n : ℚ) : 
  x = 4 → 
  5 * x + 3 = n * (x - 17) → 
  n = -23 / 13 := by
sorry

end NUMINAMATH_CALUDE_find_number_l1810_181072


namespace NUMINAMATH_CALUDE_sequence_general_term_l1810_181024

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := 2 * n.val - a n

theorem sequence_general_term (a : ℕ+ → ℚ)
  (h : ∀ n : ℕ+, S n a = (n.val : ℚ)) :
  ∀ n : ℕ+, a n = (2^n.val - 1) / 2^(n.val - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1810_181024


namespace NUMINAMATH_CALUDE_window_washing_time_l1810_181009

/-- The time it takes your friend to wash a window (in minutes) -/
def friend_time : ℝ := 3

/-- The total time it takes both of you to wash 25 windows (in minutes) -/
def total_time : ℝ := 30

/-- The number of windows you wash together -/
def num_windows : ℝ := 25

/-- Your time to wash a window (in minutes) -/
def your_time : ℝ := 2

theorem window_washing_time :
  (1 / friend_time + 1 / your_time) * total_time = num_windows :=
sorry

end NUMINAMATH_CALUDE_window_washing_time_l1810_181009


namespace NUMINAMATH_CALUDE_friday_pushups_l1810_181031

/-- Calculates the number of push-ups Miriam does on Friday given her workout schedule --/
theorem friday_pushups (monday : ℕ) : 
  let tuesday := (monday : ℚ) * (14 : ℚ) / 10
  let wednesday := (monday : ℕ) * 2
  let thursday := ((monday : ℚ) + tuesday + (wednesday : ℚ)) / 2
  let friday := (monday : ℚ) + tuesday + (wednesday : ℚ) + thursday
  monday = 5 → friday = 33 := by
sorry


end NUMINAMATH_CALUDE_friday_pushups_l1810_181031


namespace NUMINAMATH_CALUDE_circle_equation_l1810_181026

theorem circle_equation (x y : ℝ) :
  (x + 2)^2 + (y - 2)^2 = 25 ↔ 
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-2, 2) ∧ 
    radius = 5 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1810_181026


namespace NUMINAMATH_CALUDE_smith_payment_l1810_181023

-- Define the original balance
def original_balance : ℝ := 150

-- Define the finance charge rate
def finance_charge_rate : ℝ := 0.02

-- Define the finance charge calculation
def finance_charge : ℝ := original_balance * finance_charge_rate

-- Define the total payment calculation
def total_payment : ℝ := original_balance + finance_charge

-- Theorem to prove
theorem smith_payment : total_payment = 153 := by
  sorry

end NUMINAMATH_CALUDE_smith_payment_l1810_181023


namespace NUMINAMATH_CALUDE_prob_ace_king_queen_value_l1810_181088

/-- The probability of drawing an Ace, then a King, then a Queen from a standard deck of 52 cards without replacement -/
def prob_ace_king_queen : ℚ :=
  let total_cards : ℕ := 52
  let num_aces : ℕ := 4
  let num_kings : ℕ := 4
  let num_queens : ℕ := 4
  (num_aces : ℚ) / total_cards *
  (num_kings : ℚ) / (total_cards - 1) *
  (num_queens : ℚ) / (total_cards - 2)

theorem prob_ace_king_queen_value : prob_ace_king_queen = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_queen_value_l1810_181088


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1810_181000

theorem quadratic_equation_solution (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x : ℝ, x^2 + 14*x = 96 ∧ x > 0 ∧ x = Real.sqrt a - b) → a + b = 152 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1810_181000


namespace NUMINAMATH_CALUDE_solution_range_l1810_181085

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (x + m) / (x - 2) + (2 * m) / (2 - x) = 3) → 
  m < 6 ∧ m ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l1810_181085


namespace NUMINAMATH_CALUDE_inscribed_rectangle_pc_length_l1810_181044

-- Define the triangle ABC
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define the rectangle PQRS
structure InscribedRectangle (t : EquilateralTriangle) where
  ps : ℝ
  pq : ℝ
  ps_positive : ps > 0
  pq_positive : pq > 0
  on_sides : ps ≤ t.side ∧ pq ≤ t.side
  is_rectangle : pq = Real.sqrt 3 * ps
  area : ps * pq = 28 * Real.sqrt 3

-- Define the theorem
theorem inscribed_rectangle_pc_length 
  (t : EquilateralTriangle) 
  (r : InscribedRectangle t) : 
  ∃ (pc : ℝ), pc = 2 * Real.sqrt 7 ∧ 
  pc^2 = t.side^2 + (t.side - r.ps)^2 - 2 * t.side * (t.side - r.ps) * Real.cos (π/3) :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_pc_length_l1810_181044


namespace NUMINAMATH_CALUDE_square_sum_simplification_l1810_181091

theorem square_sum_simplification (a : ℝ) : a^2 + 2*a^2 = 3*a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_simplification_l1810_181091


namespace NUMINAMATH_CALUDE_alex_not_jogging_probability_l1810_181038

theorem alex_not_jogging_probability (p : ℚ) 
  (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_alex_not_jogging_probability_l1810_181038


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1810_181037

theorem quadratic_inequality_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 2*a ≥ 0) ↔ (-8 ≤ a ∧ a ≤ 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1810_181037


namespace NUMINAMATH_CALUDE_range_of_a_l1810_181016

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | x ≤ a}
def Q : Set ℝ := {y | ∃ θ : ℝ, y = Real.sin θ}

-- State the theorem
theorem range_of_a (a : ℝ) : P a ⊇ Q → a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1810_181016


namespace NUMINAMATH_CALUDE_open_book_is_random_event_l1810_181015

/-- Represents the possible classifications of events --/
inductive EventType
  | Certain
  | Random
  | Impossible
  | Determined

/-- Represents a book --/
structure Book where
  grade : Nat
  subject : String
  publisher : String

/-- Represents the event of opening a book to a specific page --/
structure OpenBookEvent where
  book : Book
  page : Nat
  intentional : Bool

/-- Definition of a certain event --/
def is_certain_event (e : OpenBookEvent) : Prop :=
  e.intentional ∧ ∀ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- Definition of a random event --/
def is_random_event (e : OpenBookEvent) : Prop :=
  ¬e.intentional ∧ ∃ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- Definition of an impossible event --/
def is_impossible_event (e : OpenBookEvent) : Prop :=
  ¬∃ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- Definition of a determined event --/
def is_determined_event (e : OpenBookEvent) : Prop :=
  e.intentional ∧ ∃ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- The main theorem to prove --/
theorem open_book_is_random_event (e : OpenBookEvent) 
  (h1 : e.book.grade = 9)
  (h2 : e.book.subject = "mathematics")
  (h3 : e.book.publisher = "East China Normal University")
  (h4 : e.page = 50)
  (h5 : ¬e.intentional) :
  is_random_event e :=
sorry

end NUMINAMATH_CALUDE_open_book_is_random_event_l1810_181015


namespace NUMINAMATH_CALUDE_final_salary_proof_l1810_181001

def original_salary : ℝ := 20000
def reduction_rate : ℝ := 0.1
def increase_rate : ℝ := 0.1

def salary_after_changes (s : ℝ) (r : ℝ) (i : ℝ) : ℝ :=
  s * (1 - r) * (1 + i)

theorem final_salary_proof :
  salary_after_changes original_salary reduction_rate increase_rate = 19800 := by
  sorry

end NUMINAMATH_CALUDE_final_salary_proof_l1810_181001


namespace NUMINAMATH_CALUDE_locust_jump_equivalence_l1810_181071

/-- A type representing the position of a locust on a line -/
def Position := ℝ

/-- A type representing a configuration of locusts -/
def Configuration := List Position

/-- A function that represents a jump to the right -/
def jumpRight (config : Configuration) (i j : ℕ) : Configuration :=
  sorry

/-- A function that represents a jump to the left -/
def jumpLeft (config : Configuration) (i j : ℕ) : Configuration :=
  sorry

/-- A predicate that checks if all locusts are 1 unit apart -/
def isUnitApart (config : Configuration) : Prop :=
  sorry

theorem locust_jump_equivalence (initial : Configuration) 
  (h : ∃ (final : Configuration), (∀ i j, jumpRight initial i j = final) ∧ isUnitApart final) :
  ∃ (final : Configuration), (∀ i j, jumpLeft initial i j = final) ∧ isUnitApart final :=
sorry

end NUMINAMATH_CALUDE_locust_jump_equivalence_l1810_181071


namespace NUMINAMATH_CALUDE_sum_of_penultimate_terms_l1810_181027

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : ℕ, i < n → a (i + 1) = a i + d

theorem sum_of_penultimate_terms (a : ℕ → ℕ) :
  arithmetic_sequence a 7 →
  a 0 = 3 →
  a 6 = 33 →
  a 4 + a 5 = 48 := by
sorry

end NUMINAMATH_CALUDE_sum_of_penultimate_terms_l1810_181027


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l1810_181018

/-- Given a person who runs at 8 km/hr and covers a total distance of 16 km
    (half walking, half running) in 3 hours, prove that the walking speed is 4 km/hr. -/
theorem walking_speed_calculation (running_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  running_speed = 8 →
  total_distance = 16 →
  total_time = 3 →
  ∃ (walking_speed : ℝ),
    walking_speed * (total_distance / 2 / walking_speed) +
    running_speed * (total_distance / 2 / running_speed) = total_time ∧
    walking_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l1810_181018
