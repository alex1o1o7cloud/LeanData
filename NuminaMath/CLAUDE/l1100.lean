import Mathlib

namespace NUMINAMATH_CALUDE_earnings_difference_l1100_110082

/-- Proves the difference in earnings between Evan and Markese -/
theorem earnings_difference : 
  ∀ (E : ℕ), 
  E > 16 →  -- Evan earned more than Markese
  E + 16 = 37 →  -- Their combined earnings
  E - 16 = 5  -- The difference in earnings
  := by sorry

end NUMINAMATH_CALUDE_earnings_difference_l1100_110082


namespace NUMINAMATH_CALUDE_count_pairs_eq_three_l1100_110042

/-- The number of distinct ordered pairs of positive integers (m,n) satisfying 1/m + 1/n = 1/3 -/
def count_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 3)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

theorem count_pairs_eq_three : count_pairs = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_three_l1100_110042


namespace NUMINAMATH_CALUDE_herd_division_l1100_110008

theorem herd_division (total : ℕ) 
  (h1 : (1 : ℚ) / 3 * total + (1 : ℚ) / 6 * total + (1 : ℚ) / 9 * total + 12 = total) : 
  total = 54 := by
  sorry

end NUMINAMATH_CALUDE_herd_division_l1100_110008


namespace NUMINAMATH_CALUDE_largest_divisor_of_a_pow_25_minus_a_l1100_110026

theorem largest_divisor_of_a_pow_25_minus_a : 
  ∃ (n : ℕ), n = 2730 ∧ 
  (∀ (a : ℤ), (a^25 - a) % n = 0) ∧
  (∀ (m : ℕ), m > n → ∃ (a : ℤ), (a^25 - a) % m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_a_pow_25_minus_a_l1100_110026


namespace NUMINAMATH_CALUDE_fifth_grade_total_is_144_l1100_110076

/-- The number of students in the fifth grade of Longteng Primary School --/
def fifth_grade_total : ℕ :=
  let class1 : ℕ := 42
  let class2 : ℕ := (class1 * 6) / 7
  let class3 : ℕ := (class2 * 5) / 6
  let class4 : ℕ := (class3 * 12) / 10
  class1 + class2 + class3 + class4

theorem fifth_grade_total_is_144 : fifth_grade_total = 144 := by
  sorry

end NUMINAMATH_CALUDE_fifth_grade_total_is_144_l1100_110076


namespace NUMINAMATH_CALUDE_prob_three_diff_suits_is_169_425_l1100_110098

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suits in a standard deck -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- Function to get the suit of a card -/
def card_suit : Fin 52 → Suit := sorry

/-- Probability of drawing three cards of different suits -/
def prob_three_different_suits (d : Deck) : ℚ :=
  let first_draw := d.cards.card
  let second_draw := d.cards.card - 1
  let third_draw := d.cards.card - 2
  let diff_suit_second := 39
  let diff_suit_third := 26
  (diff_suit_second / second_draw) * (diff_suit_third / third_draw)

theorem prob_three_diff_suits_is_169_425 (d : Deck) :
  prob_three_different_suits d = 169 / 425 := by sorry

end NUMINAMATH_CALUDE_prob_three_diff_suits_is_169_425_l1100_110098


namespace NUMINAMATH_CALUDE_polynomial_value_constraint_l1100_110062

theorem polynomial_value_constraint (P : ℤ → ℤ) (a b c d : ℤ) :
  (∃ (n : ℤ → ℤ), P = n) →
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  P a = 1979 →
  P b = 1979 →
  P c = 1979 →
  P d = 1979 →
  ∀ (x : ℤ), P x ≠ 3958 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_constraint_l1100_110062


namespace NUMINAMATH_CALUDE_systematic_sampling_eighth_group_l1100_110064

/-- Systematic sampling function -/
def systematicSample (totalEmployees : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun n => firstSample + (n - 1) * (totalEmployees / sampleSize)

theorem systematic_sampling_eighth_group 
  (totalEmployees : ℕ) 
  (sampleSize : ℕ) 
  (firstSample : ℕ) :
  totalEmployees = 200 →
  sampleSize = 40 →
  firstSample = 22 →
  systematicSample totalEmployees sampleSize firstSample 8 = 37 :=
by
  sorry

#eval systematicSample 200 40 22 8

end NUMINAMATH_CALUDE_systematic_sampling_eighth_group_l1100_110064


namespace NUMINAMATH_CALUDE_bat_costs_60_l1100_110043

/-- The cost of a ball in pounds -/
def ball_cost : ℝ := sorry

/-- The cost of a bat in pounds -/
def bat_cost : ℝ := sorry

/-- The sum of the cost of a ball and a bat is £90 -/
axiom sum_ball_bat : ball_cost + bat_cost = 90

/-- The sum of the cost of three balls and two bats is £210 -/
axiom sum_three_balls_two_bats : 3 * ball_cost + 2 * bat_cost = 210

/-- The cost of a bat is £60 -/
theorem bat_costs_60 : bat_cost = 60 := by sorry

end NUMINAMATH_CALUDE_bat_costs_60_l1100_110043


namespace NUMINAMATH_CALUDE_bch_unique_product_l1100_110078

/-- Represents a letter of the alphabet -/
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z

/-- Assigns a numerical value to each letter -/
def letterValue : Letter → Nat
| Letter.A => 1
| Letter.B => 2
| Letter.C => 3
| Letter.D => 4
| Letter.E => 5
| Letter.F => 6
| Letter.G => 7
| Letter.H => 8
| Letter.I => 9
| Letter.J => 10
| Letter.K => 11
| Letter.L => 12
| Letter.M => 13
| Letter.N => 14
| Letter.O => 15
| Letter.P => 16
| Letter.Q => 17
| Letter.R => 18
| Letter.S => 19
| Letter.T => 20
| Letter.U => 21
| Letter.V => 22
| Letter.W => 23
| Letter.X => 24
| Letter.Y => 25
| Letter.Z => 26

/-- Calculates the product of a three-letter list -/
def productOfList (a b c : Letter) : Nat :=
  letterValue a * letterValue b * letterValue c

/-- Checks if three letters are in alphabetical order -/
def isAlphabeticalOrder (a b c : Letter) : Prop :=
  letterValue a ≤ letterValue b ∧ letterValue b ≤ letterValue c

/-- Main theorem: BCH is the only other three-letter list with product equal to BDF -/
theorem bch_unique_product :
  ∀ (x y z : Letter),
    x ≠ y ∧ y ≠ z ∧ x ≠ z →
    isAlphabeticalOrder x y z →
    productOfList x y z = productOfList Letter.B Letter.D Letter.F →
    x = Letter.B ∧ y = Letter.C ∧ z = Letter.H :=
by sorry


end NUMINAMATH_CALUDE_bch_unique_product_l1100_110078


namespace NUMINAMATH_CALUDE_equation_proof_l1100_110068

theorem equation_proof : 225 + 2 * 15 * 4 + 16 = 361 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1100_110068


namespace NUMINAMATH_CALUDE_hyperbola_eq_theorem_l1100_110022

/-- A hyperbola with the given properties -/
structure Hyperbola where
  -- The hyperbola is centered at the origin
  center_origin : True
  -- The foci are on the coordinate axes
  foci_on_axes : True
  -- One of the asymptotes has the equation y = (1/2)x
  asymptote_eq : ∀ x y : ℝ, y = (1/2) * x
  -- The point (2, 2) lies on the hyperbola
  point_on_hyperbola : True

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (y^2 / 3) - (x^2 / 12) = 1

/-- Theorem stating the equation of the hyperbola with the given properties -/
theorem hyperbola_eq_theorem (h : Hyperbola) :
  ∀ x y : ℝ, hyperbola_equation h x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eq_theorem_l1100_110022


namespace NUMINAMATH_CALUDE_number_multiple_problem_l1100_110075

theorem number_multiple_problem (A B k : ℕ) 
  (sum_cond : A + B = 77)
  (bigger_cond : A = 42)
  (multiple_cond : 6 * B = k * A) :
  k = 5 := by sorry

end NUMINAMATH_CALUDE_number_multiple_problem_l1100_110075


namespace NUMINAMATH_CALUDE_leadership_structure_count_correct_l1100_110096

def colony_size : Nat := 35
def num_deputy_governors : Nat := 3
def lieutenants_per_deputy : Nat := 3
def subordinates_per_lieutenant : Nat := 2

def leadership_structure_count : Nat :=
  colony_size * 
  Nat.choose (colony_size - 1) num_deputy_governors *
  Nat.choose (colony_size - 1 - num_deputy_governors) lieutenants_per_deputy *
  Nat.choose (colony_size - 1 - num_deputy_governors - lieutenants_per_deputy) lieutenants_per_deputy *
  Nat.choose (colony_size - 1 - num_deputy_governors - 2 * lieutenants_per_deputy) lieutenants_per_deputy *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 2) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 4) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 6) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 8) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 10) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 12) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 14) 2

theorem leadership_structure_count_correct : 
  leadership_structure_count = 35 * 5984 * 4495 * 3276 * 2300 * 120 * 91 * 66 * 45 * 28 * 15 * 6 * 1 :=
by sorry

end NUMINAMATH_CALUDE_leadership_structure_count_correct_l1100_110096


namespace NUMINAMATH_CALUDE_temperature_difference_is_8_l1100_110090

-- Define the temperatures
def temp_top : ℝ := -9
def temp_foot : ℝ := -1

-- Define the temperature difference
def temp_difference : ℝ := temp_foot - temp_top

-- Theorem statement
theorem temperature_difference_is_8 : temp_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_is_8_l1100_110090


namespace NUMINAMATH_CALUDE_containers_used_is_three_l1100_110089

/-- The number of posters that can be printed with the initial amount of ink -/
def initial_posters : ℕ := 60

/-- The number of posters that can be printed after losing one container of ink -/
def remaining_posters : ℕ := 45

/-- The number of posters that can be printed with one container of ink -/
def posters_per_container : ℕ := initial_posters - remaining_posters

/-- The number of containers used to print the remaining posters -/
def containers_used : ℕ := remaining_posters / posters_per_container

theorem containers_used_is_three :
  containers_used = 3 := by sorry

end NUMINAMATH_CALUDE_containers_used_is_three_l1100_110089


namespace NUMINAMATH_CALUDE_valid_draws_count_l1100_110084

/-- Represents the number of cards for each color --/
def cards_per_color : ℕ := 5

/-- Represents the number of colors --/
def num_colors : ℕ := 3

/-- Represents the number of cards drawn --/
def cards_drawn : ℕ := 4

/-- Represents the total number of cards --/
def total_cards : ℕ := cards_per_color * num_colors

/-- Represents the set of all possible draws --/
def all_draws : Set (Fin total_cards) := sorry

/-- Predicate to check if a draw contains all colors --/
def has_all_colors (draw : Fin total_cards) : Prop := sorry

/-- Predicate to check if a draw has all different letters --/
def has_different_letters (draw : Fin total_cards) : Prop := sorry

/-- The number of valid draws --/
def num_valid_draws : ℕ := sorry

theorem valid_draws_count : num_valid_draws = 360 := by sorry

end NUMINAMATH_CALUDE_valid_draws_count_l1100_110084


namespace NUMINAMATH_CALUDE_solutions_satisfy_system_system_implies_solutions_l1100_110091

/-- The system of equations we want to solve -/
def system (x y z : ℝ) : Prop :=
  x + y + z = 2 ∧ x^2 + y^2 + z^2 = 26 ∧ x^3 + y^3 + z^3 = 38

/-- The set of solutions to our system -/
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(1, 4, -3), (1, -3, 4), (4, 1, -3), (4, -3, 1), (-3, 1, 4), (-3, 4, 1)}

/-- Theorem stating that the solutions satisfy the system -/
theorem solutions_satisfy_system : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → system x y z := by
  sorry

/-- Theorem stating that any solution to the system is in our set of solutions -/
theorem system_implies_solutions : ∀ (x y z : ℝ), system x y z → (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_solutions_satisfy_system_system_implies_solutions_l1100_110091


namespace NUMINAMATH_CALUDE_prob_two_consecutive_sum_four_l1100_110046

-- Define a 3-sided die
def Die := Fin 3

-- Define the sum of two dice
def diceSum (d1 d2 : Die) : ℕ := d1.val + d2.val + 2

-- Define the probability of getting a sum of 4 on a single roll
def probSumFour : ℚ := 1 / 3

-- Theorem statement
theorem prob_two_consecutive_sum_four :
  (probSumFour * probSumFour : ℚ) = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_two_consecutive_sum_four_l1100_110046


namespace NUMINAMATH_CALUDE_journey_speeds_correct_l1100_110015

/-- Represents the journey details and speeds -/
structure Journey where
  uphill_distance : ℝ
  downhill_distance : ℝ
  flat_distance : ℝ
  time_ab : ℝ
  time_ba : ℝ
  flat_speed : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- Checks if the given speeds satisfy the journey conditions -/
def satisfies_conditions (j : Journey) : Prop :=
  let flat_time := j.flat_distance / j.flat_speed
  let hill_time_ab := j.time_ab - flat_time
  let hill_time_ba := j.time_ba - flat_time
  (j.uphill_distance / j.uphill_speed + j.downhill_distance / j.downhill_speed = hill_time_ab) ∧
  (j.uphill_distance / j.downhill_speed + j.downhill_distance / j.uphill_speed = hill_time_ba)

/-- Theorem stating that the given speeds satisfy the journey conditions -/
theorem journey_speeds_correct (j : Journey) 
  (h1 : j.uphill_distance = 3)
  (h2 : j.downhill_distance = 6)
  (h3 : j.flat_distance = 12)
  (h4 : j.time_ab = 67/60)
  (h5 : j.time_ba = 76/60)
  (h6 : j.flat_speed = 18)
  (h7 : j.uphill_speed = 12)
  (h8 : j.downhill_speed = 30) :
  satisfies_conditions j := by
  sorry

end NUMINAMATH_CALUDE_journey_speeds_correct_l1100_110015


namespace NUMINAMATH_CALUDE_circle_distance_extrema_l1100_110097

-- Define the circle C
def Circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function d
def d (x y : ℝ) : ℝ := 
  (x + 1)^2 + y^2 + (x - 1)^2 + y^2

-- Theorem statement
theorem circle_distance_extrema :
  (∃ x y : ℝ, Circle x y ∧ ∀ x' y' : ℝ, Circle x' y' → d x y ≥ d x' y') ∧
  (∃ x y : ℝ, Circle x y ∧ ∀ x' y' : ℝ, Circle x' y' → d x y ≤ d x' y') ∧
  (∀ x y : ℝ, Circle x y → d x y ≤ 14) ∧
  (∀ x y : ℝ, Circle x y → d x y ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_circle_distance_extrema_l1100_110097


namespace NUMINAMATH_CALUDE_ratio_change_after_addition_l1100_110048

theorem ratio_change_after_addition : 
  ∀ (a b : ℕ), 
    (a : ℚ) / b = 2 / 3 →
    b - a = 8 →
    (a + 4 : ℚ) / (b + 4) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_after_addition_l1100_110048


namespace NUMINAMATH_CALUDE_probability_ratio_l1100_110066

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of different numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The number of cards with each number -/
def cards_per_number : ℕ := 5

/-- The probability of drawing five cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (2250 : ℚ) / (Nat.choose total_cards cards_drawn)

theorem probability_ratio :
  q / p = 225 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l1100_110066


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l1100_110012

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l1100_110012


namespace NUMINAMATH_CALUDE_cube_frame_construction_l1100_110017

/-- Represents a cube -/
structure Cube where
  edge_length : ℝ

/-- Represents a wire -/
structure Wire where
  length : ℝ

/-- Represents the number of cuts needed to construct a cube frame -/
def num_cuts_needed (c : Cube) (w : Wire) : ℕ := sorry

theorem cube_frame_construction (c : Cube) (w : Wire) 
  (h1 : c.edge_length = 10)
  (h2 : w.length = 120) :
  ¬ (num_cuts_needed c w = 0) ∧ (num_cuts_needed c w = 3) := by sorry

end NUMINAMATH_CALUDE_cube_frame_construction_l1100_110017


namespace NUMINAMATH_CALUDE_nine_integer_chords_l1100_110019

/-- Represents a circle with a given radius and a point P at a distance from the center -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Counts the number of integer-length chords containing P in the given circle -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem stating that for a circle with radius 20 and P at distance 12,
    there are exactly 9 integer-length chords containing P -/
theorem nine_integer_chords :
  let c := CircleWithPoint.mk 20 12
  count_integer_chords c = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_integer_chords_l1100_110019


namespace NUMINAMATH_CALUDE_target_walmart_knife_ratio_l1100_110057

/-- Represents the number of tools in a multitool -/
structure Multitool where
  screwdrivers : ℕ
  knives : ℕ
  other_tools : ℕ

/-- The Walmart multitool -/
def walmart : Multitool :=
  { screwdrivers := 1
    knives := 3
    other_tools := 2 }

/-- The Target multitool -/
def target (k : ℕ) : Multitool :=
  { screwdrivers := 1
    knives := k
    other_tools := 4 }  -- 3 files + 1 pair of scissors

/-- Total number of tools in a multitool -/
def total_tools (m : Multitool) : ℕ :=
  m.screwdrivers + m.knives + m.other_tools

/-- The theorem to prove -/
theorem target_walmart_knife_ratio :
    ∃ k : ℕ, 
      total_tools (target k) = total_tools walmart + 5 ∧ 
      k = 2 * walmart.knives := by
  sorry


end NUMINAMATH_CALUDE_target_walmart_knife_ratio_l1100_110057


namespace NUMINAMATH_CALUDE_jimin_class_size_l1100_110036

/-- The number of students in Jimin's class -/
def total_students : ℕ := 45

/-- The number of students who like Korean -/
def korean_fans : ℕ := 38

/-- The number of students who like math -/
def math_fans : ℕ := 39

/-- The number of students who like both Korean and math -/
def both_fans : ℕ := 32

/-- There is no student who does not like both Korean and math -/
axiom no_other_students : total_students = korean_fans + math_fans - both_fans

theorem jimin_class_size :
  total_students = 45 :=
sorry

end NUMINAMATH_CALUDE_jimin_class_size_l1100_110036


namespace NUMINAMATH_CALUDE_company_fund_problem_l1100_110014

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (50 * n = initial_fund + 5) →
  (45 * n + 95 = initial_fund) →
  initial_fund = 995 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l1100_110014


namespace NUMINAMATH_CALUDE_power_zero_l1100_110040

theorem power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_l1100_110040


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_one_third_l1100_110099

-- Define the logarithm functions
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem logarithm_expression_equals_one_third :
  (lg (1/4) - lg 25) / (2 * log_base 5 10 + log_base 5 (1/4)) + log_base 3 4 * log_base 8 9 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_logarithm_expression_equals_one_third_l1100_110099


namespace NUMINAMATH_CALUDE_max_cells_crossed_cells_crossed_achievable_l1100_110073

/-- Represents a circle on a grid --/
structure GridCircle where
  radius : ℝ
  center : ℝ × ℝ

/-- Represents a cell on a grid --/
structure GridCell where
  x : ℤ
  y : ℤ

/-- Function to count the number of cells crossed by a circle --/
def countCrossedCells (c : GridCircle) : ℕ :=
  sorry

/-- Theorem stating the maximum number of cells crossed by a circle with radius 10 --/
theorem max_cells_crossed (c : GridCircle) (h : c.radius = 10) :
  countCrossedCells c ≤ 80 :=
sorry

/-- Theorem stating that 80 cells can be crossed --/
theorem cells_crossed_achievable :
  ∃ (c : GridCircle), c.radius = 10 ∧ countCrossedCells c = 80 :=
sorry

end NUMINAMATH_CALUDE_max_cells_crossed_cells_crossed_achievable_l1100_110073


namespace NUMINAMATH_CALUDE_line_chart_drawing_method_l1100_110092

/-- Represents a point on a grid -/
structure GridPoint where
  x : ℝ
  y : ℝ

/-- Represents a line chart -/
structure LineChart where
  points : List GridPoint
  unit_length : ℝ
  quantity : ℝ → ℝ

/-- The method of drawing a line chart -/
def draw_line_chart (chart : LineChart) : Prop :=
  ∃ (plotted_points : List GridPoint) (connected_points : List GridPoint),
    plotted_points = chart.points ∧
    connected_points = chart.points ∧
    (∀ p ∈ chart.points, p.y = chart.quantity (p.x * chart.unit_length))

theorem line_chart_drawing_method (chart : LineChart) :
  draw_line_chart chart ↔
  (∃ (plotted_points : List GridPoint) (connected_points : List GridPoint),
    plotted_points = chart.points ∧
    connected_points = chart.points) :=
sorry

end NUMINAMATH_CALUDE_line_chart_drawing_method_l1100_110092


namespace NUMINAMATH_CALUDE_algebra_to_calculus_ratio_l1100_110002

/-- Represents the number of years Devin taught each subject and in total -/
structure TeachingYears where
  calculus : ℕ
  algebra : ℕ
  statistics : ℕ
  total : ℕ

/-- Defines the conditions of Devin's teaching career -/
def devin_teaching (y : TeachingYears) : Prop :=
  y.calculus = 4 ∧
  y.statistics = 5 * y.algebra ∧
  y.total = y.calculus + y.algebra + y.statistics ∧
  y.total = 52

/-- Theorem stating the ratio of Algebra to Calculus teaching years -/
theorem algebra_to_calculus_ratio (y : TeachingYears) 
  (h : devin_teaching y) : y.algebra / y.calculus = 2 := by
  sorry

end NUMINAMATH_CALUDE_algebra_to_calculus_ratio_l1100_110002


namespace NUMINAMATH_CALUDE_f_increasing_implies_a_range_a_range_implies_f_increasing_f_increasing_iff_a_range_l1100_110013

/-- The function f(x) defined as 2x^2 - 4(1-a)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * (1 - a) * x + 1

/-- The theorem stating that if f(x) is increasing on [3,+∞), then a ≥ -2 -/
theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y, 3 ≤ x → x < y → f a x < f a y) → a ≥ -2 :=
sorry

/-- The theorem stating that if a ≥ -2, then f(x) is increasing on [3,+∞) -/
theorem a_range_implies_f_increasing (a : ℝ) :
  a ≥ -2 → (∀ x y, 3 ≤ x → x < y → f a x < f a y) :=
sorry

/-- The main theorem stating the equivalence between f(x) being increasing on [3,+∞) and a ≥ -2 -/
theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x y, 3 ≤ x → x < y → f a x < f a y) ↔ a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_implies_a_range_a_range_implies_f_increasing_f_increasing_iff_a_range_l1100_110013


namespace NUMINAMATH_CALUDE_max_npn_value_l1100_110059

def is_two_digit_same_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ (n / 10 = n % 10)

def is_one_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def is_three_digit_npn (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

theorem max_npn_value :
  ∀ (mm m npn : ℕ),
    is_two_digit_same_digits mm →
    is_one_digit m →
    is_three_digit_npn npn →
    mm * m = npn →
    npn ≤ 729 :=
sorry

end NUMINAMATH_CALUDE_max_npn_value_l1100_110059


namespace NUMINAMATH_CALUDE_marcella_shoes_theorem_l1100_110067

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_pairs_remaining (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

/-- Theorem stating that with 50 initial pairs and 15 individual shoes lost,
    the maximum number of complete pairs remaining is 35. -/
theorem marcella_shoes_theorem :
  max_pairs_remaining 50 15 = 35 := by
  sorry

end NUMINAMATH_CALUDE_marcella_shoes_theorem_l1100_110067


namespace NUMINAMATH_CALUDE_circle_intersection_distance_ellipse_standard_form_collinearity_condition_l1100_110018

noncomputable section

-- Define the curve C
def C (t : ℝ) (x y : ℝ) : Prop := (4 - t) * x^2 + t * y^2 = 12

-- Part 1
theorem circle_intersection_distance (t : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, C t x₁ y₁ ∧ C t x₂ y₂ ∧ 
   y₁ = x₁ - 2 ∧ y₂ = x₂ - 2 ∧ 
   ∀ x y : ℝ, C t x y → ∃ r : ℝ, x^2 + y^2 = r^2) →
  ∃ A B : ℝ × ℝ, C t A.1 A.2 ∧ C t B.1 B.2 ∧ 
            A.2 = A.1 - 2 ∧ B.2 = B.1 - 2 ∧
            (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16 :=
sorry

-- Part 2
theorem ellipse_standard_form (t : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧
   ∀ x y : ℝ, C t x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (1 - b^2 / a^2 = 2/3 ∨ 1 - a^2 / b^2 = 2/3) →
  (∀ x y : ℝ, C t x y ↔ x^2 / 12 + y^2 / 4 = 1) ∨
  (∀ x y : ℝ, C t x y ↔ x^2 / 4 + y^2 / 12 = 1) :=
sorry

-- Part 3
theorem collinearity_condition (k m s : ℝ) :
  let P := {p : ℝ × ℝ | p.1^2 + 3 * p.2^2 = 12 ∧ p.2 = k * p.1 + m}
  let Q := {q : ℝ × ℝ | q.1^2 + 3 * q.2^2 = 12 ∧ q.2 = k * q.1 + m ∧ q ≠ (0, 2) ∧ q ≠ (0, -2)}
  let G := {g : ℝ × ℝ | g.2 = s ∧ ∃ q ∈ Q, g.2 - (-2) = (g.1 - 0) * (q.2 - (-2)) / (q.1 - 0)}
  s * m = 4 →
  ∃ p ∈ P, ∃ g ∈ G, (2 - g.2) * (p.1 - g.1) = (p.2 - g.2) * (0 - g.1) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_distance_ellipse_standard_form_collinearity_condition_l1100_110018


namespace NUMINAMATH_CALUDE_no_valid_labeling_l1100_110034

/-- Represents a labeling of the hexagon vertices and center -/
def Labeling := Fin 7 → Fin 7

/-- The sum of labels on a line through the center -/
def lineSum (l : Labeling) (i j : Fin 7) : ℕ :=
  l i + l 6 + l j  -- Assuming index 6 represents the center J

/-- Checks if a labeling is valid according to the problem conditions -/
def isValidLabeling (l : Labeling) : Prop :=
  (Function.Injective l) ∧ 
  (lineSum l 0 4 = lineSum l 1 5) ∧
  (lineSum l 1 5 = lineSum l 2 3) ∧
  (lineSum l 2 3 = lineSum l 0 4)

/-- The main theorem: there are no valid labelings -/
theorem no_valid_labeling : 
  ¬ ∃ l : Labeling, isValidLabeling l :=
sorry

end NUMINAMATH_CALUDE_no_valid_labeling_l1100_110034


namespace NUMINAMATH_CALUDE_not_necessarily_square_l1100_110016

/-- A quadrilateral with four sides and two diagonals -/
structure Quadrilateral :=
  (side1 side2 side3 side4 diagonal1 diagonal2 : ℝ)

/-- Predicate to check if a quadrilateral has 4 equal sides and 2 equal diagonals -/
def has_equal_sides_and_diagonals (q : Quadrilateral) : Prop :=
  q.side1 = q.side2 ∧ q.side2 = q.side3 ∧ q.side3 = q.side4 ∧
  q.diagonal1 = q.diagonal2 ∧
  q.side1 ≠ q.diagonal1

/-- Predicate to check if a quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop :=
  q.side1 = q.side2 ∧ q.side2 = q.side3 ∧ q.side3 = q.side4 ∧
  q.diagonal1 = q.diagonal2 ∧
  q.diagonal1 = q.side1 * Real.sqrt 2

/-- Theorem stating that a quadrilateral with 4 equal sides and 2 equal diagonals
    is not necessarily a square -/
theorem not_necessarily_square :
  ∃ q : Quadrilateral, has_equal_sides_and_diagonals q ∧ ¬is_square q :=
sorry


end NUMINAMATH_CALUDE_not_necessarily_square_l1100_110016


namespace NUMINAMATH_CALUDE_classroom_books_l1100_110021

/-- The total number of books in a classroom -/
def total_books (num_children : ℕ) (books_per_child : ℕ) (teacher_books : ℕ) : ℕ :=
  num_children * books_per_child + teacher_books

/-- Theorem: The total number of books in the classroom is 202 -/
theorem classroom_books : 
  total_books 15 12 22 = 202 := by
  sorry

end NUMINAMATH_CALUDE_classroom_books_l1100_110021


namespace NUMINAMATH_CALUDE_circles_contained_l1100_110037

theorem circles_contained (r R d : ℝ) (hr : r = 1) (hR : R = 5) (hd : d = 3) :
  d < R - r ∧ d + r < R :=
sorry

end NUMINAMATH_CALUDE_circles_contained_l1100_110037


namespace NUMINAMATH_CALUDE_total_amount_paid_l1100_110020

def ticket_cost : ℕ := 44
def num_people : ℕ := 3
def service_fee : ℕ := 18

theorem total_amount_paid : 
  ticket_cost * num_people + service_fee = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1100_110020


namespace NUMINAMATH_CALUDE_find_S_l1100_110047

theorem find_S : ∃ S : ℝ, (1/3 * 1/8 * S = 1/4 * 1/6 * 120) ∧ (S = 120) := by
  sorry

end NUMINAMATH_CALUDE_find_S_l1100_110047


namespace NUMINAMATH_CALUDE_h_equation_l1100_110041

theorem h_equation (x : ℝ) (h : ℝ → ℝ) :
  (4 * x^4 + 11 * x^3 + h x = 10 * x^3 - x^2 + 4 * x - 7) →
  h x = -4 * x^4 - x^3 - x^2 + 4 * x - 7 := by
sorry

end NUMINAMATH_CALUDE_h_equation_l1100_110041


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1100_110095

theorem compound_interest_rate (P : ℝ) (h1 : P * (1 + r)^6 = 6000) (h2 : P * (1 + r)^7 = 7500) : r = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1100_110095


namespace NUMINAMATH_CALUDE_charles_initial_skittles_l1100_110081

theorem charles_initial_skittles (current_skittles taken_skittles : ℕ) 
  (h1 : current_skittles = 18)
  (h2 : taken_skittles = 7) :
  current_skittles + taken_skittles = 25 := by
  sorry

end NUMINAMATH_CALUDE_charles_initial_skittles_l1100_110081


namespace NUMINAMATH_CALUDE_cosine_sine_power_equation_l1100_110056

theorem cosine_sine_power_equation (x : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  (Real.cos x)^8 + (Real.sin x)^8 = 97 / 128 ↔ x = Real.pi / 12 ∨ x = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_power_equation_l1100_110056


namespace NUMINAMATH_CALUDE_average_weight_problem_l1100_110000

/-- The average weight problem -/
theorem average_weight_problem 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84)
  (h2 : A = 78)
  (h3 : E = D + 6)
  (h4 : (B + C + D + E) / 4 = 79) :
  (A + B + C + D) / 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1100_110000


namespace NUMINAMATH_CALUDE_translation_theorem_l1100_110052

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point left by a given amount -/
def translateLeft (p : Point) (amount : ℝ) : Point :=
  { x := p.x - amount, y := p.y }

/-- Translates a point down by a given amount -/
def translateDown (p : Point) (amount : ℝ) : Point :=
  { x := p.x, y := p.y - amount }

theorem translation_theorem :
  let p := Point.mk (-4) 3
  let p' := translateDown (translateLeft p 2) 2
  p'.x = -6 ∧ p'.y = 1 := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l1100_110052


namespace NUMINAMATH_CALUDE_sum_of_squares_representation_l1100_110025

theorem sum_of_squares_representation (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), (x : ℤ) = a^2 + b^2 ∧ (y : ℤ) = c^2 + d^2 ∧ (z : ℤ) = a * c + b * d := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_representation_l1100_110025


namespace NUMINAMATH_CALUDE_number_division_problem_l1100_110010

theorem number_division_problem (x : ℚ) : (x / 5 = 75 + x / 6) → x = 2250 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1100_110010


namespace NUMINAMATH_CALUDE_cyclic_inequality_l1100_110093

def cyclic_sum (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c + f b c a + f c a b

def cyclic_prod (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c * f b c a * f c a b

theorem cyclic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  cyclic_sum (fun x y z => x / (y + z)) a b c ≥ 2 - 4 * cyclic_prod (fun x y z => x / (y + z)) a b c := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l1100_110093


namespace NUMINAMATH_CALUDE_triangle_inequality_l1100_110003

/-- Given a triangle with side lengths a, b, c, semiperimeter p, inradius r, 
    and distances from incenter to sides l_a, l_b, l_c, prove that 
    l_a * l_b * l_c ≤ r * p^2 -/
theorem triangle_inequality (a b c p r l_a l_b l_c : ℝ) 
  (h1 : l_a * l_b * l_c ≤ Real.sqrt (p^3 * (p - a) * (p - b) * (p - c)))
  (h2 : ∃ S, S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h3 : ∃ S, S = r * p) :
  l_a * l_b * l_c ≤ r * p^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1100_110003


namespace NUMINAMATH_CALUDE_minimal_poster_area_l1100_110038

theorem minimal_poster_area (m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_m_ge_n : m ≥ n) : 
  ∃ (posters : Finset (ℕ × ℕ)), 
    (Finset.card posters = m * n) ∧ 
    (∀ (k l : ℕ), (k, l) ∈ posters → 1 ≤ k ∧ k ≤ m ∧ 1 ≤ l ∧ l ≤ n) →
    (minimal_area : ℕ) = m * (n * (n + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_minimal_poster_area_l1100_110038


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l1100_110053

-- Define the point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (m + 3, m - 1)

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant_m_range :
  ∀ m : ℝ, in_fourth_quadrant (P m) ↔ -3 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l1100_110053


namespace NUMINAMATH_CALUDE_small_bottles_sold_percentage_l1100_110071

theorem small_bottles_sold_percentage
  (initial_small : ℕ)
  (initial_big : ℕ)
  (big_sold_percentage : ℚ)
  (total_remaining : ℕ)
  (h1 : initial_small = 6000)
  (h2 : initial_big = 15000)
  (h3 : big_sold_percentage = 12 / 100)
  (h4 : total_remaining = 18540)
  (h5 : ∃ (x : ℚ), 
    initial_small - (x * initial_small) + 
    initial_big - (big_sold_percentage * initial_big) = total_remaining) :
  ∃ (x : ℚ), x = 11 / 100 ∧ 
    initial_small - (x * initial_small) + 
    initial_big - (big_sold_percentage * initial_big) = total_remaining :=
by sorry

end NUMINAMATH_CALUDE_small_bottles_sold_percentage_l1100_110071


namespace NUMINAMATH_CALUDE_mixture_problem_l1100_110086

/-- Represents the quantities of milk, water, and juice in a mixture --/
structure Mixture where
  milk : ℝ
  water : ℝ
  juice : ℝ

/-- Calculates the total quantity of a mixture --/
def totalQuantity (m : Mixture) : ℝ := m.milk + m.water + m.juice

/-- Checks if the given quantities form the specified ratio --/
def isRatio (m : Mixture) (r : Mixture) : Prop :=
  m.milk / r.milk = m.water / r.water ∧ m.milk / r.milk = m.juice / r.juice

/-- The main theorem to prove --/
theorem mixture_problem (initial : Mixture) (final : Mixture) : 
  isRatio initial ⟨5, 3, 4⟩ → 
  final.milk = initial.milk ∧ 
  final.water = initial.water + 12 ∧ 
  final.juice = initial.juice + 6 →
  isRatio final ⟨5, 9, 8⟩ →
  totalQuantity initial = 24 := by
  sorry

end NUMINAMATH_CALUDE_mixture_problem_l1100_110086


namespace NUMINAMATH_CALUDE_sin_pi_half_minus_x_is_even_l1100_110079

/-- The function f(x) = sin(π/2 - x) is even, implying symmetry about the y-axis -/
theorem sin_pi_half_minus_x_is_even :
  ∀ x : ℝ, Real.sin (π / 2 - x) = Real.sin (π / 2 - (-x)) :=
by sorry

end NUMINAMATH_CALUDE_sin_pi_half_minus_x_is_even_l1100_110079


namespace NUMINAMATH_CALUDE_wendy_glasses_difference_l1100_110058

/-- The number of glasses polished by Wendy -/
def total_glasses : ℕ := 110

/-- The number of small glasses polished by Wendy -/
def small_glasses : ℕ := 50

/-- The number of large glasses polished by Wendy -/
def large_glasses : ℕ := total_glasses - small_glasses

theorem wendy_glasses_difference :
  large_glasses > small_glasses ∧ large_glasses - small_glasses = 10 := by
  sorry

end NUMINAMATH_CALUDE_wendy_glasses_difference_l1100_110058


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l1100_110085

/-- Given vectors a and b where a is parallel to b, prove that 2sin(α)cos(α) = -4/5 -/
theorem parallel_vectors_trig_identity (α : ℝ) :
  let a : ℝ × ℝ := (Real.cos α, -2)
  let b : ℝ × ℝ := (Real.sin α, 1)
  (∃ (k : ℝ), a = k • b) →
  2 * Real.sin α * Real.cos α = -4/5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l1100_110085


namespace NUMINAMATH_CALUDE_original_triangle_area_l1100_110077

theorem original_triangle_area
  (original_area : ℝ)
  (new_area : ℝ)
  (h1 : new_area = 32)
  (h2 : new_area = 4 * original_area) :
  original_area = 8 :=
by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l1100_110077


namespace NUMINAMATH_CALUDE_prop2_prop4_l1100_110050

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem for proposition 2
theorem prop2 (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → plane_perpendicular α β := by sorry

-- Theorem for proposition 4
theorem prop4 (m : Line) (α β : Plane) :
  perpendicular m α → plane_parallel α β → perpendicular m β := by sorry

end NUMINAMATH_CALUDE_prop2_prop4_l1100_110050


namespace NUMINAMATH_CALUDE_impossible_digit_product_after_increment_l1100_110033

def digit_product (n : ℕ) : ℕ := sorry

theorem impossible_digit_product_after_increment :
  ∀ N : ℕ,
  N > 0 →
  digit_product N = 20 →
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 24) ∧
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 25) ∧
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 30) ∧
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 40) ∧
  ¬(∃ M : ℕ, M = N + 1 ∧ digit_product M = 35) :=
by sorry

end NUMINAMATH_CALUDE_impossible_digit_product_after_increment_l1100_110033


namespace NUMINAMATH_CALUDE_birds_in_tree_l1100_110060

theorem birds_in_tree (initial_birds : ℕ) : 
  initial_birds + 21 = 35 → initial_birds = 14 := by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l1100_110060


namespace NUMINAMATH_CALUDE_penny_frog_count_l1100_110011

/-- The number of tree frogs Penny counted -/
def tree_frogs : ℕ := 55

/-- The number of poison frogs Penny counted -/
def poison_frogs : ℕ := 10

/-- The number of wood frogs Penny counted -/
def wood_frogs : ℕ := 13

/-- The total number of frogs Penny counted -/
def total_frogs : ℕ := tree_frogs + poison_frogs + wood_frogs

theorem penny_frog_count : total_frogs = 78 := by
  sorry

end NUMINAMATH_CALUDE_penny_frog_count_l1100_110011


namespace NUMINAMATH_CALUDE_inequalities_solution_sets_l1100_110028

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x^2 > 1
def inequality2 (x : ℝ) : Prop := -x^2 + 2*x + 3 > 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | x < -1 ∨ x > 1}
def solution_set2 : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem inequalities_solution_sets :
  (∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1) ∧
  (∀ x : ℝ, inequality2 x ↔ x ∈ solution_set2) :=
sorry

end NUMINAMATH_CALUDE_inequalities_solution_sets_l1100_110028


namespace NUMINAMATH_CALUDE_balloon_arrangement_count_l1100_110029

/-- The number of letters in the word BALLOON -/
def n : ℕ := 7

/-- The number of times 'L' appears in BALLOON -/
def k₁ : ℕ := 2

/-- The number of times 'O' appears in BALLOON -/
def k₂ : ℕ := 2

/-- The number of unique arrangements of letters in BALLOON -/
def balloon_arrangements : ℕ := n.factorial / (k₁.factorial * k₂.factorial)

theorem balloon_arrangement_count : balloon_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangement_count_l1100_110029


namespace NUMINAMATH_CALUDE_odd_function_property_l1100_110009

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_period : ∀ x, f (x + 2) = -f x) : 
  f (-2) = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l1100_110009


namespace NUMINAMATH_CALUDE_fathers_age_twice_marikas_l1100_110088

/-- Marika's age in 2006 -/
def marika_age_2006 : ℕ := 10

/-- The year of Marika's 10th birthday -/
def birth_year : ℕ := 2006

/-- The ratio of father's age to Marika's age in 2006 -/
def age_ratio_2006 : ℕ := 5

/-- The year when father's age will be twice Marika's age -/
def target_year : ℕ := 2036

theorem fathers_age_twice_marikas : 
  target_year = birth_year + (age_ratio_2006 - 2) * marika_age_2006 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_twice_marikas_l1100_110088


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l1100_110045

theorem min_value_of_fraction (x a b : ℝ) (hx : 0 < x ∧ x < 1) (ha : a > 0) (hb : b > 0) :
  ∃ m : ℝ, m = (a + b)^2 ∧ ∀ y : ℝ, 0 < y ∧ y < 1 → 1 / (y^a * (1 - y)^b) ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l1100_110045


namespace NUMINAMATH_CALUDE_circle_properties_l1100_110004

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4

-- Define the line that contains the center of C
def center_line (x y : ℝ) : Prop :=
  2 * x - y + 2 = 0

-- Define the intersecting line
def intersecting_line (a x y : ℝ) : Prop :=
  x = a * y + 3

-- Theorem statement
theorem circle_properties :
  -- Given conditions
  (circle_C 1 0) ∧ 
  (circle_C (-1) 2) ∧
  (∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧ center_line x₀ y₀) →
  -- Conclusions
  (∀ (x y : ℝ), circle_C x y ↔ (x + 1)^2 + y^2 = 4) ∧
  (∃ (a : ℝ), (a = Real.sqrt 15 ∨ a = -Real.sqrt 15) ∧
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
      intersecting_line a x₁ y₁ ∧ intersecting_line a x₂ y₂ ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) :=
by sorry


end NUMINAMATH_CALUDE_circle_properties_l1100_110004


namespace NUMINAMATH_CALUDE_zoo_count_l1100_110080

theorem zoo_count (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 200) 
  (h2 : total_legs = 522) : 
  ∃ (birds mammals : ℕ), 
    birds + mammals = total_animals ∧ 
    2 * birds + 4 * mammals = total_legs ∧ 
    birds = 139 := by
  sorry

end NUMINAMATH_CALUDE_zoo_count_l1100_110080


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l1100_110024

theorem mean_of_added_numbers (original_numbers : List ℝ) (x y z : ℝ) :
  original_numbers.length = 7 →
  original_numbers.sum / original_numbers.length = 75 →
  (original_numbers.sum + x + y + z) / (original_numbers.length + 3) = 90 →
  (x + y + z) / 3 = 125 := by
sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l1100_110024


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1100_110039

/-- Given a cube with surface area 294 square centimeters, its volume is 343 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  s > 0 →
  6 * s^2 = 294 →
  s^3 = 343 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1100_110039


namespace NUMINAMATH_CALUDE_sqrt_7_simplest_l1100_110006

-- Define the concept of simplest square root
def is_simplest_sqrt (x : ℝ) : Prop :=
  x > 0 ∧ ∀ y : ℝ, y > 0 → y^2 = x → y = Real.sqrt x

-- Define the set of square roots to compare
def sqrt_set : Set ℝ := {Real.sqrt 24, Real.sqrt (1/3), Real.sqrt 7, Real.sqrt 0.2}

-- Theorem statement
theorem sqrt_7_simplest :
  ∀ x ∈ sqrt_set, x ≠ Real.sqrt 7 → ¬(is_simplest_sqrt x) ∧ is_simplest_sqrt (Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_7_simplest_l1100_110006


namespace NUMINAMATH_CALUDE_tomatoes_needed_fried_green_tomatoes_l1100_110027

theorem tomatoes_needed (slices_per_tomato : ℕ) (slices_per_meal : ℕ) (people : ℕ) : ℕ :=
  let total_slices := slices_per_meal * people
  total_slices / slices_per_tomato

theorem fried_green_tomatoes :
  tomatoes_needed 8 20 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_needed_fried_green_tomatoes_l1100_110027


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_A_complement_A_union_B_l1100_110055

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- State the theorems to be proved
theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 7} := by sorry

theorem complement_A : (Set.univ \ A) = {x | x < 3 ∨ 7 ≤ x} := by sorry

theorem complement_A_union_B : (Set.univ \ (A ∪ B)) = {x | x ≤ 2 ∨ 10 ≤ x} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_A_complement_A_union_B_l1100_110055


namespace NUMINAMATH_CALUDE_common_external_tangents_parallel_l1100_110032

/-- Two circles with equal radii -/
structure EqualRadiiCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  radius : ℝ

/-- A line representing a common external tangent -/
structure CommonExternalTangent where
  slope : ℝ
  intercept : ℝ

/-- The line connecting the centers of two circles -/
def centerLine (c : EqualRadiiCircles) : ℝ × ℝ → Prop :=
  fun p => ∃ t : ℝ, p = (1 - t) • c.center1 + t • c.center2

/-- Two lines are parallel -/
def parallel (l1 l2 : ℝ × ℝ → Prop) : Prop :=
  ∀ p q : ℝ × ℝ, l1 p → l1 q → l2 p → l2 q → 
    (p.1 - q.1) * (p.2 - q.2) = (p.1 - q.1) * (p.2 - q.2)

theorem common_external_tangents_parallel (c : EqualRadiiCircles) 
  (t1 t2 : CommonExternalTangent) : 
  parallel (fun p => p.2 = t1.slope * p.1 + t1.intercept) 
           (fun p => p.2 = t2.slope * p.1 + t2.intercept) ∧
  parallel (fun p => p.2 = t1.slope * p.1 + t1.intercept) 
           (centerLine c) := by
  sorry

end NUMINAMATH_CALUDE_common_external_tangents_parallel_l1100_110032


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1100_110001

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 13 + a 15 = 20) :
  a 10 - (1/5) * a 12 = 4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1100_110001


namespace NUMINAMATH_CALUDE_sequence_conclusions_l1100_110007

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1)^2 - a (n + 1) = a n)

theorem sequence_conclusions (a : ℕ → ℝ) (h : sequence_property a) :
  (∀ n ≥ 2, a n > 1) ∧
  ((0 < a 1 ∧ a 1 < 2) → (∀ n, a n < a (n + 1))) ∧
  (a 1 > 2 → ∀ n ≥ 2, 2 < a n ∧ a n < a 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_conclusions_l1100_110007


namespace NUMINAMATH_CALUDE_bacteria_growth_l1100_110023

/-- The time in minutes for the bacteria population to double -/
def doubling_time : ℝ := 6

/-- The initial population of bacteria -/
def initial_population : ℕ := 1000

/-- The total time of growth in minutes -/
def total_time : ℝ := 53.794705707972525

/-- The final population of bacteria after the given time -/
def final_population : ℕ := 495451

theorem bacteria_growth :
  let num_doublings : ℝ := total_time / doubling_time
  let theoretical_population : ℝ := initial_population * (2 ^ num_doublings)
  ⌊theoretical_population⌋ = final_population :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1100_110023


namespace NUMINAMATH_CALUDE_number_exists_l1100_110005

theorem number_exists : ∃ N : ℝ, 2.5 * N = 199.99999999999997 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l1100_110005


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l1100_110094

theorem sum_of_quadratic_solutions :
  let f (x : ℝ) := x^2 - 6*x - 8 - (2*x + 18)
  let solutions := {x : ℝ | f x = 0}
  (∃ x₁ x₂ : ℝ, solutions = {x₁, x₂}) →
  (∃ s : ℝ, ∀ x ∈ solutions, ∃ y ∈ solutions, x + y = s) →
  (∃ s : ℝ, ∀ x ∈ solutions, ∃ y ∈ solutions, x + y = s ∧ s = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l1100_110094


namespace NUMINAMATH_CALUDE_any_nonzero_to_zero_power_l1100_110030

theorem any_nonzero_to_zero_power (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_any_nonzero_to_zero_power_l1100_110030


namespace NUMINAMATH_CALUDE_f_composition_value_l1100_110069

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else (1/2) ^ x

theorem f_composition_value : f (f (1/27)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1100_110069


namespace NUMINAMATH_CALUDE_revenue_difference_l1100_110035

/-- Mr. Banks' number of investments -/
def banks_investments : ℕ := 8

/-- Revenue from each of Mr. Banks' investments -/
def banks_revenue_per_investment : ℕ := 500

/-- Ms. Elizabeth's number of investments -/
def elizabeth_investments : ℕ := 5

/-- Revenue from each of Ms. Elizabeth's investments -/
def elizabeth_revenue_per_investment : ℕ := 900

/-- The difference in total revenue between Ms. Elizabeth and Mr. Banks -/
theorem revenue_difference : 
  elizabeth_investments * elizabeth_revenue_per_investment - 
  banks_investments * banks_revenue_per_investment = 500 := by
  sorry

end NUMINAMATH_CALUDE_revenue_difference_l1100_110035


namespace NUMINAMATH_CALUDE_subset_implies_m_eq_two_l1100_110065

/-- The set A of solutions to the quadratic equation x^2 + 3x + 2 = 0 -/
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}

/-- The set B of solutions to the quadratic equation x^2 + (m+1)x + m = 0 -/
def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

/-- Theorem stating that if A is a subset of B, then m must equal 2 -/
theorem subset_implies_m_eq_two : A ⊆ B 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_eq_two_l1100_110065


namespace NUMINAMATH_CALUDE_shoe_pair_difference_l1100_110063

theorem shoe_pair_difference (ellie_shoes riley_shoes : ℕ) : 
  ellie_shoes = 8 →
  riley_shoes < ellie_shoes →
  ellie_shoes + riley_shoes = 13 →
  ellie_shoes - riley_shoes = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_shoe_pair_difference_l1100_110063


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l1100_110061

theorem algebraic_expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (a + 1) / a / (a - (1 + 2 * a^2) / (3 * a)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l1100_110061


namespace NUMINAMATH_CALUDE_haley_balls_count_l1100_110074

theorem haley_balls_count (balls_per_bag : ℕ) (num_bags : ℕ) (h1 : balls_per_bag = 4) (h2 : num_bags = 9) :
  balls_per_bag * num_bags = 36 := by
  sorry

end NUMINAMATH_CALUDE_haley_balls_count_l1100_110074


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1100_110087

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^9 + b^9) / (a + b)^9 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1100_110087


namespace NUMINAMATH_CALUDE_fourth_group_number_l1100_110031

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (secondNumber : ℕ) : ℕ → ℕ :=
  fun groupIndex => secondNumber + (groupIndex - 2) * (totalStudents / sampleSize)

theorem fourth_group_number
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (secondNumber : ℕ)
  (h1 : totalStudents = 60)
  (h2 : sampleSize = 5)
  (h3 : secondNumber = 16) :
  systematicSample totalStudents sampleSize secondNumber 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_number_l1100_110031


namespace NUMINAMATH_CALUDE_sally_orange_balloons_l1100_110044

/-- The number of orange balloons Sally has now, given her initial count and the number she lost. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Sally now has 7 orange balloons -/
theorem sally_orange_balloons : remaining_balloons 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sally_orange_balloons_l1100_110044


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1100_110070

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 1) :
  {x : ℝ | x^2 - (a + 1)*x + a < 0} = {x : ℝ | a < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1100_110070


namespace NUMINAMATH_CALUDE_fencing_cost_is_1634_l1100_110072

/-- Represents the cost of fencing for a single side -/
structure SideCost where
  length : ℝ
  costPerFoot : ℝ

/-- Calculates the total cost of fencing for an irregular four-sided plot -/
def totalFencingCost (sideA sideB sideC sideD : SideCost) : ℝ :=
  sideA.length * sideA.costPerFoot +
  sideB.length * sideB.costPerFoot +
  sideC.length * sideC.costPerFoot +
  sideD.length * sideD.costPerFoot

/-- Theorem stating that the total fencing cost for the given plot is 1634 -/
theorem fencing_cost_is_1634 :
  let sideA : SideCost := { length := 8, costPerFoot := 58 }
  let sideB : SideCost := { length := 5, costPerFoot := 62 }
  let sideC : SideCost := { length := 6, costPerFoot := 64 }
  let sideD : SideCost := { length := 7, costPerFoot := 68 }
  totalFencingCost sideA sideB sideC sideD = 1634 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_1634_l1100_110072


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1100_110051

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {2,3,4}
def B : Set Nat := {1,4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {1,4,5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1100_110051


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1100_110049

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |2*y - 44| + |y - 24| = |3*y - 66| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1100_110049


namespace NUMINAMATH_CALUDE_rectangle_horizontal_length_l1100_110054

theorem rectangle_horizontal_length 
  (perimeter : ℝ) 
  (h v : ℝ) 
  (perimeter_eq : perimeter = 2 * h + 2 * v) 
  (vertical_shorter : v = h - 3) 
  (perimeter_value : perimeter = 54) : h = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_horizontal_length_l1100_110054


namespace NUMINAMATH_CALUDE_football_equipment_cost_l1100_110083

/-- The cost of equipment for a football team -/
theorem football_equipment_cost : 
  let num_players : ℕ := 16
  let jersey_cost : ℚ := 25
  let shorts_cost : ℚ := 15.20
  let socks_cost : ℚ := 6.80
  let total_cost : ℚ := num_players * (jersey_cost + shorts_cost + socks_cost)
  total_cost = 752 := by sorry

end NUMINAMATH_CALUDE_football_equipment_cost_l1100_110083
