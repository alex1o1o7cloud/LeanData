import Mathlib

namespace arithmetic_sequence_problem_l3661_366102

-- Define an arithmetic sequence and its sum
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence a →
  sum_of_arithmetic_sequence a S →
  m > 0 →
  S (m - 1) = -2 →
  S m = 0 →
  S (m + 1) = 3 →
  m = 5 := by
  sorry

end arithmetic_sequence_problem_l3661_366102


namespace equatorial_circumference_scientific_notation_l3661_366147

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ a
  h2 : a < 10

/-- Check if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (x : ℝ) : Prop :=
  sn.a * (10 : ℝ) ^ sn.n = x

/-- The equatorial circumference in meters -/
def equatorialCircumference : ℝ := 40000000

/-- Theorem stating that 4 × 10^7 is the correct scientific notation for the equatorial circumference -/
theorem equatorial_circumference_scientific_notation :
  ∃ sn : ScientificNotation, sn.a = 4 ∧ sn.n = 7 ∧ represents sn equatorialCircumference :=
sorry

end equatorial_circumference_scientific_notation_l3661_366147


namespace geometric_arithmetic_sequence_problem_l3661_366188

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_sequence_problem
  (a b : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_arithmetic : is_arithmetic_sequence b)
  (h_a_prod : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
  (h_b_sum : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 :=
sorry

end geometric_arithmetic_sequence_problem_l3661_366188


namespace pythagorean_from_law_of_cosines_l3661_366192

/-- The law of cosines for a triangle with sides a, b, c and angle γ opposite side c -/
def lawOfCosines (a b c : ℝ) (γ : Real) : Prop :=
  c^2 = a^2 + b^2 - 2*a*b*Real.cos γ

/-- The Pythagorean theorem for a right triangle with sides a, b, c where c is the hypotenuse -/
def pythagoreanTheorem (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

/-- Theorem stating that the Pythagorean theorem is a special case of the law of cosines -/
theorem pythagorean_from_law_of_cosines (a b c : ℝ) :
  lawOfCosines a b c (π/2) → pythagoreanTheorem a b c :=
by sorry

end pythagorean_from_law_of_cosines_l3661_366192


namespace min_beacons_proof_l3661_366157

/-- Represents a room in the maze --/
structure Room :=
  (x : Nat) (y : Nat)

/-- Represents the maze structure --/
def Maze := List Room

/-- Calculates the distance between two rooms --/
def distance (r1 r2 : Room) : Nat :=
  sorry

/-- Checks if a set of beacon positions allows unambiguous location determination --/
def is_unambiguous (maze : Maze) (beacons : List Room) : Prop :=
  sorry

/-- The minimum number of beacons required --/
def min_beacons : Nat := 3

/-- The specific beacon positions that work --/
def beacon_positions : List Room :=
  [⟨1, 1⟩, ⟨4, 3⟩, ⟨1, 5⟩]  -- Representing a1, d3, a5

theorem min_beacons_proof (maze : Maze) :
  (∀ beacons : List Room, beacons.length < min_beacons → ¬ is_unambiguous maze beacons) ∧
  is_unambiguous maze beacon_positions :=
sorry

end min_beacons_proof_l3661_366157


namespace rational_sum_l3661_366150

theorem rational_sum (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Finset ℚ) = 
    {-24, -2, -3/2, -1/8, 1, 3} →
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
sorry

end rational_sum_l3661_366150


namespace daniel_noodles_remaining_l3661_366108

/-- The number of noodles Daniel has now, given his initial count and the number he gave away. -/
def noodles_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Daniel has 54 noodles remaining. -/
theorem daniel_noodles_remaining :
  noodles_remaining 66 12 = 54 := by
  sorry

end daniel_noodles_remaining_l3661_366108


namespace cheapest_solution_for_1096_days_l3661_366155

/-- Represents the cost and coverage of a ticket type -/
structure Ticket where
  days : ℕ
  cost : ℚ

/-- Finds the minimum cost to cover at least a given number of days using two types of tickets -/
def minCost (ticket1 ticket2 : Ticket) (totalDays : ℕ) : ℚ :=
  sorry

theorem cheapest_solution_for_1096_days :
  let sevenDayTicket : Ticket := ⟨7, 703/100⟩
  let thirtyDayTicket : Ticket := ⟨30, 30⟩
  minCost sevenDayTicket thirtyDayTicket 1096 = 140134/100 := by sorry

end cheapest_solution_for_1096_days_l3661_366155


namespace quadratic_root_implies_p_value_l3661_366119

theorem quadratic_root_implies_p_value (q p : ℝ) (h : Complex.I * Complex.I = -1) :
  (3 : ℂ) * (4 + Complex.I)^2 - q * (4 + Complex.I) + p = 0 → p = 51 := by
  sorry

end quadratic_root_implies_p_value_l3661_366119


namespace unique_solution_for_P_equals_2C_l3661_366177

def P (r n : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

def C (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem unique_solution_for_P_equals_2C (n : ℕ+) : 
  P 8 n = 2 * C 8 2 ↔ n = 2 := by sorry

end unique_solution_for_P_equals_2C_l3661_366177


namespace factor_expression_l3661_366195

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := by
  sorry

end factor_expression_l3661_366195


namespace min_valid_positions_l3661_366168

/-- Represents a disk with a certain number of sectors and red sectors. -/
structure Disk :=
  (total_sectors : ℕ)
  (red_sectors : ℕ)
  (h_red_le_total : red_sectors ≤ total_sectors)

/-- Represents the configuration of two overlapping disks. -/
structure DiskOverlay :=
  (disk1 : Disk)
  (disk2 : Disk)
  (h_same_sectors : disk1.total_sectors = disk2.total_sectors)

/-- Calculates the number of positions with at most 20 overlapping red sectors. -/
def count_valid_positions (overlay : DiskOverlay) : ℕ :=
  overlay.disk1.total_sectors - (overlay.disk1.red_sectors * overlay.disk2.red_sectors) / 21 + 1

theorem min_valid_positions (overlay : DiskOverlay) 
  (h_total : overlay.disk1.total_sectors = 1965)
  (h_red1 : overlay.disk1.red_sectors = 200)
  (h_red2 : overlay.disk2.red_sectors = 200) :
  count_valid_positions overlay = 61 :=
sorry

end min_valid_positions_l3661_366168


namespace sammy_score_l3661_366156

theorem sammy_score (sammy_score : ℕ) (gab_score : ℕ) (cher_score : ℕ) (opponent_score : ℕ) :
  gab_score = 2 * sammy_score →
  cher_score = 2 * gab_score →
  opponent_score = 85 →
  sammy_score + gab_score + cher_score = opponent_score + 55 →
  sammy_score = 20 := by
sorry

end sammy_score_l3661_366156


namespace total_handshakes_l3661_366139

-- Define the number of twin sets and triplet sets
def twin_sets : ℕ := 12
def triplet_sets : ℕ := 8

-- Define the total number of twins and triplets
def total_twins : ℕ := twin_sets * 2
def total_triplets : ℕ := triplet_sets * 3

-- Define the number of handshakes for each twin and triplet
def twin_handshakes : ℕ := (total_twins - 2) + (total_triplets * 3 / 4)
def triplet_handshakes : ℕ := (total_triplets - 3) + (total_twins * 1 / 4)

-- Theorem to prove
theorem total_handshakes : 
  (total_twins * twin_handshakes + total_triplets * triplet_handshakes) / 2 = 804 := by
  sorry

end total_handshakes_l3661_366139


namespace assignment_plans_eq_48_l3661_366115

/-- Represents the number of umpires from each country -/
def umpires_per_country : ℕ := 2

/-- Represents the number of countries -/
def num_countries : ℕ := 3

/-- Represents the number of venues -/
def num_venues : ℕ := 3

/-- Calculates the number of ways to assign umpires to venues -/
def assignment_plans : ℕ := sorry

/-- Theorem stating that the number of assignment plans is 48 -/
theorem assignment_plans_eq_48 : assignment_plans = 48 := by sorry

end assignment_plans_eq_48_l3661_366115


namespace triangle_third_side_length_l3661_366191

theorem triangle_third_side_length (a b : ℝ) (cos_theta : ℝ) : 
  a = 5 → b = 3 → 
  (5 * cos_theta^2 - 7 * cos_theta - 6 = 0) →
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2 * a * b * cos_theta ∧ c = 2 * Real.sqrt 13 :=
by sorry

end triangle_third_side_length_l3661_366191


namespace total_towels_weight_lb_l3661_366184

-- Define the given conditions
def mary_towels : ℕ := 24
def frances_towels : ℕ := mary_towels / 4
def frances_towels_weight_oz : ℚ := 128

-- Define the weight of one towel in ounces
def towel_weight_oz : ℚ := frances_towels_weight_oz / frances_towels

-- Define the total number of towels
def total_towels : ℕ := mary_towels + frances_towels

-- Define the conversion factor from ounces to pounds
def oz_to_lb : ℚ := 1 / 16

-- Theorem to prove
theorem total_towels_weight_lb :
  (total_towels : ℚ) * towel_weight_oz * oz_to_lb = 40 :=
sorry

end total_towels_weight_lb_l3661_366184


namespace recycling_point_calculation_l3661_366127

/-- The number of pounds needed to recycle to earn one point -/
def pounds_per_point (zoe_pounds : ℕ) (friends_pounds : ℕ) (total_points : ℕ) : ℚ :=
  (zoe_pounds + friends_pounds : ℚ) / total_points

theorem recycling_point_calculation :
  pounds_per_point 25 23 6 = 8 := by
  sorry

end recycling_point_calculation_l3661_366127


namespace parabola_point_relationship_l3661_366160

-- Define the parabola function
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 7

-- Define the theorem
theorem parabola_point_relationship (a : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h_opens_down : a < 0)
  (h_y₁ : y₁ = parabola a (-4))
  (h_y₂ : y₂ = parabola a 2)
  (h_y₃ : y₃ = parabola a 3) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end parabola_point_relationship_l3661_366160


namespace x_intercept_ratio_l3661_366112

-- Define the slopes and y-intercept
def m₁ : ℝ := 12
def m₂ : ℝ := 8
def b : ℝ := sorry

-- Define the x-intercepts
def u : ℝ := sorry
def v : ℝ := sorry

-- Define the lines
def line₁ (x : ℝ) : ℝ := m₁ * x + b
def line₂ (x : ℝ) : ℝ := m₂ * x + b

-- State the theorem
theorem x_intercept_ratio : 
  b ≠ 0 ∧ 
  line₁ u = 0 ∧ 
  line₂ v = 0 → 
  u / v = 2 / 3 := by sorry

end x_intercept_ratio_l3661_366112


namespace fraction_division_equality_l3661_366174

theorem fraction_division_equality : (8/9 - 5/6 + 2/3) / (-5/18) = -13/5 := by
  sorry

end fraction_division_equality_l3661_366174


namespace floor_length_calculation_l3661_366111

/-- Given a rectangular floor with width 8 m, covered by a square carpet with 4 m sides,
    leaving 64 square meters uncovered, the length of the floor is 10 m. -/
theorem floor_length_calculation (floor_width : ℝ) (carpet_side : ℝ) (uncovered_area : ℝ) :
  floor_width = 8 →
  carpet_side = 4 →
  uncovered_area = 64 →
  (floor_width * (carpet_side ^ 2 + uncovered_area) / floor_width) = 10 :=
by
  sorry

#check floor_length_calculation

end floor_length_calculation_l3661_366111


namespace smallest_factor_of_32_with_sum_3_l3661_366124

theorem smallest_factor_of_32_with_sum_3 :
  ∃ (a b c : ℤ),
    a * b * c = 32 ∧
    a + b + c = 3 ∧
    (∀ (x y z : ℤ), x * y * z = 32 → x + y + z = 3 → min a (min b c) ≤ min x (min y z)) ∧
    min a (min b c) = -4 :=
by sorry

end smallest_factor_of_32_with_sum_3_l3661_366124


namespace no_real_solutions_l3661_366193

theorem no_real_solutions :
  ∀ x : ℝ, (3 * x) / (x^2 + 2*x + 4) + (4 * x) / (x^2 - 4*x + 5) ≠ 1 := by
  sorry

end no_real_solutions_l3661_366193


namespace circle_ratio_theorem_l3661_366100

noncomputable def circle_ratio (r : ℝ) (A B C : ℝ × ℝ) : Prop :=
  let O := (0, 0)  -- Center of the circle
  ∃ (θ : ℝ),
    -- Points A, B, and C are on a circle of radius r
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧
    -- AB = AC
    dist A B = dist A C ∧
    -- AB > r
    dist A B > r ∧
    -- Length of minor arc BC is r
    θ = 1 ∧
    -- Ratio AB/BC
    dist A B / dist B C = (1/2) * (1 / Real.sin (1/4))

theorem circle_ratio_theorem (r : ℝ) (A B C : ℝ × ℝ) 
  (h : circle_ratio r A B C) : 
  ∃ (θ : ℝ), dist A B / dist B C = (1/2) * (1 / Real.sin (1/4)) :=
sorry

end circle_ratio_theorem_l3661_366100


namespace aardvark_path_length_l3661_366161

/- Define the radii of the circles -/
def small_radius : ℝ := 15
def large_radius : ℝ := 30

/- Define pi as a real number -/
noncomputable def π : ℝ := Real.pi

/- Theorem statement -/
theorem aardvark_path_length :
  let small_arc := π * small_radius
  let large_arc := π * large_radius
  let radial_segment := large_radius - small_radius
  small_arc + large_arc + 2 * radial_segment = 45 * π + 30 := by
  sorry

#check aardvark_path_length

end aardvark_path_length_l3661_366161


namespace prob_at_least_one_heart_or_joker_correct_l3661_366158

/-- The number of cards in a standard deck plus two jokers -/
def total_cards : ℕ := 54

/-- The number of hearts and jokers combined -/
def heart_or_joker : ℕ := 15

/-- The probability of drawing at least one heart or joker in two draws with replacement -/
def prob_at_least_one_heart_or_joker : ℚ := 155 / 324

theorem prob_at_least_one_heart_or_joker_correct :
  (1 : ℚ) - (1 - heart_or_joker / total_cards) ^ 2 = prob_at_least_one_heart_or_joker :=
sorry

end prob_at_least_one_heart_or_joker_correct_l3661_366158


namespace quadratic_equation_solutions_l3661_366122

theorem quadratic_equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 5*x₁ + 6 = 0 ∧ x₂^2 - 5*x₂ + 6 = 0 ∧ x₁ = 2 ∧ x₂ = 3) ∧
  (∃ x₁ x₂ : ℝ, 2*x₁^2 - 4*x₁ - 1 = 0 ∧ 2*x₂^2 - 4*x₂ - 1 = 0 ∧ 
    x₁ = (4 + Real.sqrt 24) / 4 ∧ x₂ = (4 - Real.sqrt 24) / 4) :=
by sorry

end quadratic_equation_solutions_l3661_366122


namespace cost_for_holly_fence_l3661_366162

/-- The total cost to plant trees along a fence --/
def total_cost (fence_length_yards : ℕ) (tree_width_feet : ℚ) (cost_per_tree : ℚ) : ℚ :=
  let fence_length_feet := fence_length_yards * 3
  let num_trees := fence_length_feet / tree_width_feet
  num_trees * cost_per_tree

/-- Proof that the total cost to plant trees along a 25-yard fence,
    where each tree is 1.5 feet wide and costs $8.00, is $400.00 --/
theorem cost_for_holly_fence :
  total_cost 25 (3/2) 8 = 400 := by
  sorry

end cost_for_holly_fence_l3661_366162


namespace remainder_98765432_mod_25_l3661_366153

theorem remainder_98765432_mod_25 : 98765432 % 25 = 7 := by
  sorry

end remainder_98765432_mod_25_l3661_366153


namespace smallest_n_with_seven_in_squares_l3661_366181

def contains_seven (n : ℕ) : Prop :=
  ∃ d k, n = 10 * k + 7 * d ∧ d ≤ 9

theorem smallest_n_with_seven_in_squares : 
  ∀ n : ℕ, n < 26 → ¬(contains_seven (n^2) ∧ contains_seven ((n+1)^2)) ∧
  (contains_seven (26^2) ∧ contains_seven (27^2)) :=
by sorry

end smallest_n_with_seven_in_squares_l3661_366181


namespace sum_of_first_four_terms_l3661_366126

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_four_terms
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_8th : a 8 = 21)
  (h_9th : a 9 = 17)
  (h_10th : a 10 = 13) :
  (a 1) + (a 2) + (a 3) + (a 4) = 172 :=
sorry

end sum_of_first_four_terms_l3661_366126


namespace smallest_four_digit_divisible_by_four_and_five_l3661_366103

theorem smallest_four_digit_divisible_by_four_and_five : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ n % 5 = 0 → n ≥ 1020 :=
by sorry

end smallest_four_digit_divisible_by_four_and_five_l3661_366103


namespace base_conversion_sum_equality_l3661_366144

def base_to_decimal (digits : List Nat) (base : Nat) : Rat :=
  (digits.reverse.enum.map (λ (i, d) => d * base ^ i)).sum

theorem base_conversion_sum_equality : 
  let a := base_to_decimal [2, 5, 4] 8
  let b := base_to_decimal [1, 2] 4
  let c := base_to_decimal [1, 3, 2] 5
  let d := base_to_decimal [2, 2] 3
  a / b + c / d = 33.9167 := by sorry

end base_conversion_sum_equality_l3661_366144


namespace line_intercept_sum_l3661_366199

/-- Given a line with equation 3x + 5y + k = 0, where the sum of its x-intercept and y-intercept is 16, prove that k = -30. -/
theorem line_intercept_sum (k : ℝ) : 
  (∃ (x y : ℝ), 3 * x + 5 * y + k = 0 ∧ 
   (3 * 0 + 5 * y + k = 0 → 3 * x + 5 * 0 + k = 0 → x + y = 16)) → 
  k = -30 := by
sorry

end line_intercept_sum_l3661_366199


namespace bus_driver_rate_l3661_366110

theorem bus_driver_rate (hours_worked : ℕ) (total_compensation : ℚ) : 
  hours_worked = 50 →
  total_compensation = 920 →
  ∃ (regular_rate : ℚ),
    (40 * regular_rate + (hours_worked - 40) * (1.75 * regular_rate) = total_compensation) ∧
    regular_rate = 16 := by
  sorry

end bus_driver_rate_l3661_366110


namespace chicken_flock_ratio_l3661_366197

/-- Chicken flock problem -/
theorem chicken_flock_ratio : 
  ∀ (susie_rir susie_gc britney_gc britney_total : ℕ),
  susie_rir = 11 →
  susie_gc = 6 →
  britney_gc = susie_gc / 2 →
  britney_total = (susie_rir + susie_gc) + 8 →
  ∃ (britney_rir : ℕ),
    britney_rir + britney_gc = britney_total ∧
    britney_rir = 2 * susie_rir :=
by sorry

end chicken_flock_ratio_l3661_366197


namespace ginger_water_usage_l3661_366113

/-- Calculates the total cups of water used by Ginger in her garden --/
def total_water_used (hours_worked : ℕ) (cups_per_bottle : ℕ) (bottles_for_plants : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (bottles_for_plants * cups_per_bottle)

/-- Theorem stating that given the conditions, Ginger used 26 cups of water --/
theorem ginger_water_usage :
  let hours_worked : ℕ := 8
  let cups_per_bottle : ℕ := 2
  let bottles_for_plants : ℕ := 5
  total_water_used hours_worked cups_per_bottle bottles_for_plants = 26 := by
  sorry

end ginger_water_usage_l3661_366113


namespace prime_divisibility_equivalence_l3661_366106

theorem prime_divisibility_equivalence (p : ℕ) (hp : Nat.Prime p) :
  (∃ x : ℤ, ∃ d₁ : ℤ, x^2 - x + 3 = d₁ * p) ↔ 
  (∃ y : ℤ, ∃ d₂ : ℤ, y^2 - y + 25 = d₂ * p) := by
sorry

end prime_divisibility_equivalence_l3661_366106


namespace calculate_expression_l3661_366130

theorem calculate_expression : 5 + 4 * (4 - 9)^3 = -495 := by
  sorry

end calculate_expression_l3661_366130


namespace probability_gears_from_algebras_l3661_366159

/-- The set of letters in "ALGEBRAS" -/
def algebras : Finset Char := {'A', 'L', 'G', 'E', 'B', 'R', 'S'}

/-- The set of letters in "GEARS" -/
def gears : Finset Char := {'G', 'E', 'A', 'R', 'S'}

/-- The probability of selecting a tile with a letter from "GEARS" out of the tiles from "ALGEBRAS" -/
theorem probability_gears_from_algebras :
  (algebras.filter (λ c => c ∈ gears)).card / algebras.card = 3 / 4 := by
  sorry

end probability_gears_from_algebras_l3661_366159


namespace divisibility_sequence_eventually_periodic_l3661_366116

/-- A sequence of positive integers satisfying the given divisibility property -/
def DivisibilitySequence (a : ℕ → ℕ+) : Prop :=
  ∀ n m : ℕ, (a (n + 2*m)).val ∣ (a n).val + (a (n + m)).val

/-- The sequence is eventually periodic -/
def EventuallyPeriodic (a : ℕ → ℕ+) : Prop :=
  ∃ N d : ℕ, d > 0 ∧ ∀ n : ℕ, n > N → a n = a (n + d)

/-- Main theorem: A sequence satisfying the divisibility property is eventually periodic -/
theorem divisibility_sequence_eventually_periodic (a : ℕ → ℕ+) :
  DivisibilitySequence a → EventuallyPeriodic a := by
  sorry

end divisibility_sequence_eventually_periodic_l3661_366116


namespace largest_divisor_of_product_l3661_366198

theorem largest_divisor_of_product (n : ℕ) (h : Odd n) :
  (∃ (k : ℕ), (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) = 480 * k) ∧
  (∀ (m : ℕ), m > 480 → ∃ (n : ℕ), Odd n ∧ ¬(∃ (k : ℕ), (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) = m * k)) :=
sorry

end largest_divisor_of_product_l3661_366198


namespace fruit_purchase_total_l3661_366166

/-- Calculate the total amount paid for fruits given their quantities and rates --/
theorem fruit_purchase_total (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) :
  grape_quantity = 9 →
  grape_rate = 70 →
  mango_quantity = 9 →
  mango_rate = 55 →
  grape_quantity * grape_rate + mango_quantity * mango_rate = 1125 := by
sorry

end fruit_purchase_total_l3661_366166


namespace traffic_light_theorem_l3661_366175

structure TrafficLightSystem where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ
  h1 : 0 ≤ p1 ∧ p1 ≤ 1
  h2 : 0 ≤ p2 ∧ p2 ≤ 1
  h3 : 0 ≤ p3 ∧ p3 ≤ 1
  h4 : p1 < p2
  h5 : p2 < p3
  h6 : p1 = 1/2
  h7 : (1 - p1) * (1 - p2) * (1 - p3) = 1/24
  h8 : p1 * p2 * p3 = 1/4

def prob_first_red_at_third (s : TrafficLightSystem) : ℝ :=
  (1 - s.p1) * (1 - s.p2) * s.p3

def expected_red_lights (s : TrafficLightSystem) : ℝ :=
  s.p1 + s.p2 + s.p3

theorem traffic_light_theorem (s : TrafficLightSystem) :
  prob_first_red_at_third s = 1/8 ∧ expected_red_lights s = 23/12 := by
  sorry

end traffic_light_theorem_l3661_366175


namespace book_pages_l3661_366132

theorem book_pages : ∀ (total : ℕ), 
  (total : ℚ) * (1 - 2/5) * (1 - 5/8) = 36 → 
  total = 120 := by
  sorry

end book_pages_l3661_366132


namespace petrov_insurance_cost_l3661_366134

/-- Calculate the total insurance cost for the Petrov family's mortgage --/
def calculate_insurance_cost (apartment_cost loan_amount interest_rate property_rate
                              woman_rate man_rate title_rate maria_share vasily_share : ℝ) : ℝ :=
  let total_loan := loan_amount * (1 + interest_rate)
  let property_cost := total_loan * property_rate
  let title_cost := total_loan * title_rate
  let maria_cost := total_loan * maria_share * woman_rate
  let vasily_cost := total_loan * vasily_share * man_rate
  property_cost + title_cost + maria_cost + vasily_cost

/-- The total insurance cost for the Petrov family's mortgage is 47481.2 rubles --/
theorem petrov_insurance_cost :
  calculate_insurance_cost 13000000 8000000 0.095 0.0009 0.0017 0.0019 0.0027 0.4 0.6 = 47481.2 := by
  sorry

end petrov_insurance_cost_l3661_366134


namespace seating_arrangement_l3661_366118

theorem seating_arrangement (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end seating_arrangement_l3661_366118


namespace amount_over_limit_l3661_366182

/-- Calculates the amount spent over a given limit when purchasing a necklace and a book,
    where the book costs $5 more than the necklace. -/
theorem amount_over_limit (necklace_cost book_cost limit : ℕ) : 
  necklace_cost = 34 →
  book_cost = necklace_cost + 5 →
  limit = 70 →
  (necklace_cost + book_cost) - limit = 3 := by
sorry


end amount_over_limit_l3661_366182


namespace equivalent_operations_l3661_366145

theorem equivalent_operations (x : ℝ) : 
  (x * (2/5)) / (4/7) = x * (7/10) := by
sorry

end equivalent_operations_l3661_366145


namespace age_ratio_l3661_366178

def tom_age : ℝ := 40.5
def total_age : ℝ := 54

theorem age_ratio : 
  let antonette_age := total_age - tom_age
  tom_age / antonette_age = 3 := by sorry

end age_ratio_l3661_366178


namespace unique_solution_sequence_l3661_366173

theorem unique_solution_sequence (n : ℕ) (hn : n ≥ 4) :
  ∃! (a : ℕ → ℝ),
    (∀ i, i ∈ Finset.range (2 * n) → a i > 0) ∧
    (∀ k, k ∈ Finset.range n →
      a (2 * k) = a (2 * k - 1) + a ((2 * k + 1) % (2 * n))) ∧
    (∀ k, k ∈ Finset.range n →
      a (2 * k - 1) = 1 / a ((2 * k - 2) % (2 * n)) + 1 / a (2 * k)) ∧
    (∀ i, i ∈ Finset.range (2 * n) → a i = if i % 2 = 0 then 2 else 1) :=
by sorry

end unique_solution_sequence_l3661_366173


namespace number_of_divisors_l3661_366117

theorem number_of_divisors 
  (p q r : Nat) 
  (m : Nat) 
  (h_p_prime : Nat.Prime p) 
  (h_q_prime : Nat.Prime q) 
  (h_r_prime : Nat.Prime r) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) 
  (h_m_pos : m > 0) 
  (h_n_def : n = 7^m * p^2 * q * r) : 
  Nat.card (Nat.divisors n) = 12 * (m + 1) := by
sorry

end number_of_divisors_l3661_366117


namespace tysons_age_l3661_366194

/-- Given the ages and relationships between Kyle, Julian, Frederick, and Tyson, prove Tyson's age --/
theorem tysons_age (kyle_age julian_age frederick_age tyson_age : ℕ) : 
  kyle_age = 25 →
  kyle_age = julian_age + 5 →
  frederick_age = julian_age + 20 →
  frederick_age = 2 * tyson_age →
  tyson_age = 20 := by
  sorry

end tysons_age_l3661_366194


namespace M_subset_N_l3661_366180

def M : Set ℝ := {-1, 1}

def N : Set ℝ := {x | (1 / x) < 3}

theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l3661_366180


namespace divisor_problem_l3661_366137

theorem divisor_problem (x : ℝ) (d : ℝ) : 
  x = 22.142857142857142 →
  (7 * (x + 5)) / d - 5 = 33 →
  d = 5 := by
sorry

end divisor_problem_l3661_366137


namespace kyles_presents_cost_difference_kyles_presents_cost_difference_is_11_l3661_366164

/-- The cost difference between the first and third present given Kyle's purchases. -/
theorem kyles_presents_cost_difference : ℕ → Prop :=
  fun difference =>
    ∀ (cost_1 cost_2 cost_3 : ℕ),
      cost_1 = 18 →
      cost_2 = cost_1 + 7 →
      cost_3 < cost_1 →
      cost_1 + cost_2 + cost_3 = 50 →
      difference = cost_1 - cost_3

/-- The cost difference between the first and third present is 11. -/
theorem kyles_presents_cost_difference_is_11 : kyles_presents_cost_difference 11 := by
  sorry

end kyles_presents_cost_difference_kyles_presents_cost_difference_is_11_l3661_366164


namespace trigonometric_identity_l3661_366196

theorem trigonometric_identity (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : Real.sin x ^ 4 / a ^ 2 + Real.cos x ^ 4 / b ^ 2 = 1 / (a ^ 2 + b ^ 2)) :
  Real.sin x ^ 2008 / a ^ 2006 + Real.cos x ^ 2008 / b ^ 2006 = 1 / (a ^ 2 + b ^ 2) ^ 1003 := by
  sorry

end trigonometric_identity_l3661_366196


namespace percentage_problem_l3661_366101

theorem percentage_problem (x : ℝ) (h : 24 = 75 / 100 * x) : x = 32 := by
  sorry

end percentage_problem_l3661_366101


namespace coefficient_x_squared_expansion_l3661_366128

/-- The coefficient of x^2 in the expansion of (3x^3 + 5x^2 - 4x + 1)(2x^2 - 9x + 3) -/
def coefficient_x_squared : ℤ := 51

/-- The first polynomial in the product -/
def poly1 (x : ℚ) : ℚ := 3 * x^3 + 5 * x^2 - 4 * x + 1

/-- The second polynomial in the product -/
def poly2 (x : ℚ) : ℚ := 2 * x^2 - 9 * x + 3

/-- Theorem stating that the coefficient of x^2 in the expansion of (poly1 * poly2) is equal to coefficient_x_squared -/
theorem coefficient_x_squared_expansion :
  ∃ (a b c d e : ℚ), (poly1 * poly2) = (λ x => a * x^4 + b * x^3 + coefficient_x_squared * x^2 + d * x + e) :=
by sorry

end coefficient_x_squared_expansion_l3661_366128


namespace ellipse_theorem_l3661_366143

-- Define the ellipse M
def ellipse_M (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1 ∧ a > 0

-- Define the right focus F
def right_focus (a c : ℝ) : Prop :=
  c > 0 ∧ a^2 = 3 + c^2

-- Define the symmetric property
def symmetric_property (c : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_M (2*c) (-x+2*c) y ∧ x^2 + y^2 = 0

-- Main theorem
theorem ellipse_theorem (a c : ℝ) 
  (h1 : ellipse_M a 0 0)
  (h2 : right_focus a c)
  (h3 : symmetric_property c) :
  a^2 = 4 ∧ c = 1 ∧
  ∀ (k x₁ y₁ x₂ y₂ : ℝ),
    (ellipse_M a x₁ y₁ ∧ ellipse_M a x₂ y₂ ∧
     y₁ = k*(x₁ - 4) ∧ y₂ = k*(x₂ - 4) ∧ k ≠ 0) →
    ∃ (t : ℝ), t*(y₁ + y₂) + x₁ = 1 ∧ t*(x₁ - x₂) + y₁ = 0 :=
by sorry

end ellipse_theorem_l3661_366143


namespace toy_bridge_weight_l3661_366105

/-- The weight that a toy bridge must support -/
theorem toy_bridge_weight (full_cans : Nat) (soda_per_can : Nat) (empty_can_weight : Nat) (additional_empty_cans : Nat) : 
  full_cans * (soda_per_can + empty_can_weight) + additional_empty_cans * empty_can_weight = 88 :=
by
  sorry

#check toy_bridge_weight 6 12 2 2

end toy_bridge_weight_l3661_366105


namespace pencil_distribution_solution_l3661_366136

/-- Represents the pencil distribution problem --/
def PencilDistribution (initial_pencils : ℕ) (initial_containers : ℕ) (first_addition : ℕ) (second_addition : ℕ) (final_containers : ℕ) : Prop :=
  let total_pencils := initial_pencils + first_addition + second_addition
  ∃ (distributed_pencils : ℕ), 
    distributed_pencils ≤ total_pencils ∧
    distributed_pencils % final_containers = 0 ∧
    ∀ (n : ℕ), n > distributed_pencils → n % final_containers ≠ 0 ∨ n > total_pencils

/-- Theorem stating the solution to the pencil distribution problem --/
theorem pencil_distribution_solution :
  PencilDistribution 150 5 30 47 6 → 
  ∃ (distributed_pencils : ℕ), distributed_pencils = 222 ∧ distributed_pencils % 6 = 0 :=
sorry

end pencil_distribution_solution_l3661_366136


namespace rhombus_acute_angle_l3661_366185

/-- Given a rhombus, prove that its acute angle is arccos(1/9) when the ratio of volumes of rotation is 1:2√5 -/
theorem rhombus_acute_angle (a : ℝ) (h : a > 0) : 
  let α := Real.arccos (1/9)
  let V₁ := (1/3) * π * (a * Real.sin (α/2))^2 * (2 * a * Real.cos (α/2))
  let V₂ := π * (a * Real.sin α)^2 * a
  V₁ / V₂ = 1 / (2 * Real.sqrt 5) → 
  α = Real.arccos (1/9) := by
sorry

end rhombus_acute_angle_l3661_366185


namespace sum_of_common_divisors_l3661_366146

def number_list : List Int := [24, 48, -18, 108, 72]

def is_common_divisor (d : Nat) : Bool :=
  number_list.all (fun n => n % d == 0)

def common_divisors : List Nat :=
  (List.range 108).filter is_common_divisor

theorem sum_of_common_divisors : (common_divisors.sum) = 12 := by
  sorry

end sum_of_common_divisors_l3661_366146


namespace proportion_check_l3661_366142

/-- A set of four positive real numbers forms a proportion if the product of the first and last
    numbers equals the product of the middle two numbers. -/
def IsProportional (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * d = b * c

theorem proportion_check :
  IsProportional 5 15 3 9 ∧
  ¬IsProportional 4 5 6 7 ∧
  ¬IsProportional 3 4 5 8 ∧
  ¬IsProportional 8 4 1 3 :=
by sorry

end proportion_check_l3661_366142


namespace g_comp_four_roots_l3661_366189

/-- The function g(x) defined as x^2 + 8x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 8*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 4 distinct real roots iff d < 4 -/
theorem g_comp_four_roots (d : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g_comp d x₁ = 0 ∧ g_comp d x₂ = 0 ∧ g_comp d x₃ = 0 ∧ g_comp d x₄ = 0 ∧
    ∀ (y : ℝ), g_comp d y = 0 → y = x₁ ∨ y = x₂ ∨ y = x₃ ∨ y = x₄) ↔
  d < 4 :=
sorry

end g_comp_four_roots_l3661_366189


namespace functional_equation_solution_l3661_366165

/-- Given a function f : ℝ → ℝ that satisfies the functional equation
    3f(x) + 2f(1-x) = 4x for all x, prove that f(x) = 4x - 8/5 for all x. -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x, 3 * f x + 2 * f (1 - x) = 4 * x) :
  ∀ x, f x = 4 * x - 8 / 5 := by
  sorry

end functional_equation_solution_l3661_366165


namespace folded_square_distance_l3661_366183

/-- Given a square sheet of paper with area 18 cm², prove that when folded so that
    a corner touches the line from midpoint of adjacent side to opposite corner,
    creating equal visible areas, the distance from corner to original position is 3 cm. -/
theorem folded_square_distance (s : ℝ) (h1 : s^2 = 18) : 
  let d := s * Real.sqrt 2 / 2
  d = 3 := by sorry

end folded_square_distance_l3661_366183


namespace emily_candy_from_neighbors_l3661_366167

/-- The number of candy pieces Emily received from her sister -/
def candy_from_sister : ℕ := 13

/-- The number of candy pieces Emily ate per day -/
def candy_eaten_per_day : ℕ := 9

/-- The number of days Emily's candy lasted -/
def days_candy_lasted : ℕ := 2

/-- The number of candy pieces Emily received from neighbors -/
def candy_from_neighbors : ℕ := (candy_eaten_per_day * days_candy_lasted) - candy_from_sister

theorem emily_candy_from_neighbors : candy_from_neighbors = 5 := by
  sorry

end emily_candy_from_neighbors_l3661_366167


namespace max_sum_at_11_l3661_366163

/-- Arithmetic sequence with first term 21 and common difference -2 -/
def arithmetic_sequence (n : ℕ) : ℚ := 21 - 2 * (n - 1)

/-- Sum of first n terms of the arithmetic sequence -/
def sequence_sum (n : ℕ) : ℚ := (n : ℚ) * (21 + arithmetic_sequence n) / 2

/-- The sum reaches its maximum value when n = 11 -/
theorem max_sum_at_11 : 
  ∀ k : ℕ, k ≠ 0 → sequence_sum 11 ≥ sequence_sum k :=
sorry

end max_sum_at_11_l3661_366163


namespace cloth_selling_price_l3661_366140

/-- Calculates the total selling price of cloth given the quantity sold, profit per meter, and cost price per meter. -/
def total_selling_price (quantity : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  quantity * (profit_per_meter + cost_price_per_meter)

/-- Proves that the total selling price of 66 meters of cloth with a profit of Rs. 5 per meter and a cost price of Rs. 5 per meter is Rs. 660. -/
theorem cloth_selling_price :
  total_selling_price 66 5 5 = 660 := by
  sorry

end cloth_selling_price_l3661_366140


namespace triangle_angle_C_l3661_366107

theorem triangle_angle_C (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  Real.sin B + Real.sin A * (Real.sin C - Real.cos C) = 0 →
  a = 2 →
  c = Real.sqrt 2 →
  a / Real.sin A = c / Real.sin C →
  C = π / 6 := by
sorry

end triangle_angle_C_l3661_366107


namespace simplify_fraction_l3661_366148

theorem simplify_fraction : (126 : ℚ) / 11088 = 1 / 88 := by
  sorry

end simplify_fraction_l3661_366148


namespace fifteen_sided_polygon_diagonals_l3661_366141

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by
  sorry

end fifteen_sided_polygon_diagonals_l3661_366141


namespace intersection_point_modulo_9_l3661_366125

theorem intersection_point_modulo_9 :
  ∀ x : ℕ, (3 * x + 6) % 9 = (7 * x + 3) % 9 → x % 9 = 3 := by
  sorry

end intersection_point_modulo_9_l3661_366125


namespace hyperbola_center_trajectory_l3661_366109

/-- The hyperbola equation with parameter m -/
def hyperbola (x y m : ℝ) : Prop :=
  x^2 - y^2 - 6*m*x - 4*m*y + 5*m^2 - 1 = 0

/-- The trajectory equation of the center -/
def trajectory_equation (x y : ℝ) : Prop :=
  2*x + 3*y = 0

/-- Theorem stating that the trajectory equation of the center of the hyperbola
    is 2x + 3y = 0 for all real m -/
theorem hyperbola_center_trajectory :
  ∀ m : ℝ, ∃ x y : ℝ, hyperbola x y m ∧ trajectory_equation x y :=
sorry

end hyperbola_center_trajectory_l3661_366109


namespace equation_solutions_l3661_366172

def equation (x y : ℝ) : Prop :=
  x^2 - x*y + y^2 - x + 3*y - 7 = 0

def solution_set : Set (ℝ × ℝ) :=
  {(3,1), (-1,1), (3,-1), (-3,-1), (-1,-5)}

theorem equation_solutions :
  (∀ (x y : ℝ), (x, y) ∈ solution_set ↔ equation x y) ∧
  equation 3 1 :=
sorry

end equation_solutions_l3661_366172


namespace magnitude_of_sum_l3661_366131

-- Define the vectors
def vec_a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def vec_b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def vec_c : Fin 2 → ℝ := ![2, -6]

-- Define the conditions
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * v i

-- Theorem statement
theorem magnitude_of_sum (x y : ℝ) :
  perpendicular (vec_a x) vec_c →
  parallel (vec_b y) vec_c →
  ‖vec_a x + vec_b y‖ = 5 * Real.sqrt 2 := by
  sorry


end magnitude_of_sum_l3661_366131


namespace inequality_solution_set_l3661_366114

theorem inequality_solution_set (a b : ℝ) : 
  {x : ℝ | a * x > b} ≠ Set.Iio (-b/a) := by sorry

end inequality_solution_set_l3661_366114


namespace rectangle_hexagon_pqr_sum_l3661_366129

/-- A hexagon formed by three rectangles intersecting three straight lines -/
structure RectangleHexagon where
  -- External angles at S, T, U
  s : ℝ
  t : ℝ
  u : ℝ
  -- External angles at P, Q, R
  p : ℝ
  q : ℝ
  r : ℝ
  -- Conditions
  angle_s : s = 55
  angle_t : t = 60
  angle_u : u = 65
  sum_external : p + q + r + s + t + u = 360

/-- The sum of external angles at P, Q, and R in the RectangleHexagon is 180° -/
theorem rectangle_hexagon_pqr_sum (h : RectangleHexagon) : h.p + h.q + h.r = 180 := by
  sorry

end rectangle_hexagon_pqr_sum_l3661_366129


namespace interest_rate_problem_l3661_366121

/-- Given a sum P at simple interest rate R for 5 years, if increasing the rate by 5% 
    results in Rs. 250 more interest, then P = 1000 -/
theorem interest_rate_problem (P R : ℝ) (h : P > 0) (k : R > 0) :
  (P * (R + 5) * 5) / 100 - (P * R * 5) / 100 = 250 → P = 1000 := by
  sorry

end interest_rate_problem_l3661_366121


namespace points_three_units_from_origin_l3661_366171

theorem points_three_units_from_origin (a : ℝ) : 
  |a| = 3 → (a = 3 ∨ a = -3) := by
sorry

end points_three_units_from_origin_l3661_366171


namespace geometric_sequence_a1_value_l3661_366190

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a1_value
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a3 : a 3 = 1)
  (h_mean : (a 5 + (3/2) * a 4) / 2 = 1/2) :
  a 1 = 4 :=
sorry

end geometric_sequence_a1_value_l3661_366190


namespace complex_fraction_real_implies_a_equals_two_l3661_366135

theorem complex_fraction_real_implies_a_equals_two (a : ℝ) :
  (((a : ℂ) + 2 * Complex.I) / (1 + Complex.I)).im = 0 → a = 2 := by
  sorry

end complex_fraction_real_implies_a_equals_two_l3661_366135


namespace laura_debt_l3661_366154

/-- Calculates the total amount owed after applying simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Proves that Laura owes $36.40 after one year -/
theorem laura_debt : 
  let principal : ℝ := 35
  let rate : ℝ := 0.04
  let time : ℝ := 1
  total_amount_owed principal rate time = 36.40 := by
  sorry

end laura_debt_l3661_366154


namespace system_of_equations_solution_l3661_366120

theorem system_of_equations_solution (x y : ℚ) 
  (eq1 : 3 * x - 2 * y = 7)
  (eq2 : 2 * x + 3 * y = 8) : 
  x = 37 / 13 := by
  sorry

end system_of_equations_solution_l3661_366120


namespace hyperbola_point_k_l3661_366179

/-- Given a point P(-3, 1) on the hyperbola y = k/x where k ≠ 0, prove that k = -3 -/
theorem hyperbola_point_k (k : ℝ) (h1 : k ≠ 0) (h2 : (1 : ℝ) = k / (-3)) : k = -3 := by
  sorry

end hyperbola_point_k_l3661_366179


namespace sin_squared_minus_cos_squared_l3661_366187

theorem sin_squared_minus_cos_squared (α : Real) (h : Real.sin α = Real.sqrt 5 / 5) :
  Real.sin α ^ 2 - Real.cos α ^ 2 = -3/5 := by
  sorry

end sin_squared_minus_cos_squared_l3661_366187


namespace asian_games_competition_l3661_366176

/-- Represents a player in the competition -/
structure Player where
  prelim_prob : ℚ  -- Probability of passing preliminary round
  final_prob : ℚ   -- Probability of passing final round

/-- The three players in the competition -/
def players : List Player := [
  ⟨1/2, 1/3⟩,  -- Player A
  ⟨1/3, 1/3⟩,  -- Player B
  ⟨1/2, 1/3⟩   -- Player C
]

/-- Probability of a player participating in the city competition -/
def city_comp_prob (p : Player) : ℚ := p.prelim_prob * p.final_prob

/-- Probability of at least one player participating in the city competition -/
def at_least_one_prob : ℚ :=
  1 - (players.map (λ p => 1 - city_comp_prob p)).prod

/-- Expected value of Option 1 (Lottery) -/
def option1_expected : ℚ := 3 * (1/3) * 600

/-- Expected value of Option 2 (Fixed Rewards) -/
def option2_expected : ℚ := 700

/-- Main theorem to prove -/
theorem asian_games_competition :
  at_least_one_prob = 31/81 ∧ option2_expected > option1_expected := by
  sorry


end asian_games_competition_l3661_366176


namespace cereal_servings_l3661_366149

theorem cereal_servings (cups_per_serving : ℝ) (total_cups_needed : ℝ) 
  (h1 : cups_per_serving = 2.0)
  (h2 : total_cups_needed = 36) :
  total_cups_needed / cups_per_serving = 18 := by
  sorry

end cereal_servings_l3661_366149


namespace complement_intersection_empty_l3661_366152

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {2, 4, 5}

theorem complement_intersection_empty :
  (Aᶜ ∩ Bᶜ : Set Nat) = ∅ :=
by
  sorry

#check complement_intersection_empty

end complement_intersection_empty_l3661_366152


namespace sqrt_ab_max_value_l3661_366133

theorem sqrt_ab_max_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  ∃ (m : ℝ), m = 1/2 ∧ ∀ x, x = Real.sqrt (a * b) → x ≤ m :=
sorry

end sqrt_ab_max_value_l3661_366133


namespace total_packs_eq_51_l3661_366169

/-- The number of cookie packs sold in the first village -/
def village1_packs : ℕ := 23

/-- The number of cookie packs sold in the second village -/
def village2_packs : ℕ := 28

/-- The total number of cookie packs sold in both villages -/
def total_packs : ℕ := village1_packs + village2_packs

theorem total_packs_eq_51 : total_packs = 51 := by
  sorry

end total_packs_eq_51_l3661_366169


namespace rectangles_count_l3661_366123

/-- The number of rectangles in a strip of height 1 and width n --/
def rectanglesInStrip (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The total number of rectangles in the given grid --/
def totalRectangles : ℕ :=
  rectanglesInStrip 5 + rectanglesInStrip 4 - 1

theorem rectangles_count : totalRectangles = 24 := by
  sorry

end rectangles_count_l3661_366123


namespace noah_has_largest_result_l3661_366151

def starting_number : ℕ := 15

def liam_result : ℕ := ((starting_number - 2) * 3) + 3
def mia_result : ℕ := ((starting_number * 3) - 4) + 3
def noah_result : ℕ := ((starting_number - 3) + 4) * 3

theorem noah_has_largest_result :
  noah_result > liam_result ∧ noah_result > mia_result :=
by sorry

end noah_has_largest_result_l3661_366151


namespace music_students_l3661_366170

/-- Proves that the number of students taking music is 50 given the conditions of the problem -/
theorem music_students (total : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : art = 20)
  (h3 : both = 10)
  (h4 : neither = 440) :
  ∃ music : ℕ, music = 50 ∧ total = music + art - both + neither :=
by sorry

end music_students_l3661_366170


namespace randy_piggy_bank_l3661_366186

/-- Calculates the remaining money in Randy's piggy bank after a year -/
theorem randy_piggy_bank (initial_amount : ℕ) (spend_per_trip : ℕ) (trips_per_month : ℕ) (months : ℕ) :
  initial_amount = 200 →
  spend_per_trip = 2 →
  trips_per_month = 4 →
  months = 12 →
  initial_amount - (spend_per_trip * trips_per_month * months) = 104 := by
  sorry

#check randy_piggy_bank

end randy_piggy_bank_l3661_366186


namespace cube_cutting_l3661_366104

theorem cube_cutting (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^3 = 98 + b^3) : b = 3 :=
sorry

end cube_cutting_l3661_366104


namespace magical_stack_size_l3661_366138

/-- A magical stack is a stack of cards where at least one card from each pile retains its original position after restacking. -/
def MagicalStack (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≤ n ∧ b ≤ n ∧ a ≠ b

/-- The position of a card in the original stack. -/
def OriginalPosition (card : ℕ) (n : ℕ) : ℕ :=
  if card ≤ n then card else card - n

/-- The position of a card in the restacked stack. -/
def RestackedPosition (card : ℕ) (n : ℕ) : ℕ :=
  2 * card - 1 - (if card ≤ n then 0 else 1)

/-- A card retains its position if its original position equals its restacked position. -/
def RetainsPosition (card : ℕ) (n : ℕ) : Prop :=
  OriginalPosition card n = RestackedPosition card n

theorem magical_stack_size :
  ∀ n : ℕ,
    MagicalStack n →
    RetainsPosition 111 n →
    RetainsPosition 90 n →
    2 * n ≥ 332 →
    2 * n = 332 :=
sorry

end magical_stack_size_l3661_366138
