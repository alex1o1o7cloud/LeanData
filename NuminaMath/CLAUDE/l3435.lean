import Mathlib

namespace chocolate_syrup_usage_l3435_343564

/-- The number of ounces of chocolate syrup used in each shake -/
def syrup_per_shake : ℝ := 4

/-- The number of ounces of chocolate syrup used on each cone -/
def syrup_per_cone : ℝ := 6

/-- The number of shakes sold -/
def num_shakes : ℕ := 2

/-- The number of cones sold -/
def num_cones : ℕ := 1

/-- The total number of ounces of chocolate syrup used -/
def total_syrup : ℝ := 14

theorem chocolate_syrup_usage :
  syrup_per_shake * num_shakes + syrup_per_cone * num_cones = total_syrup :=
by sorry

end chocolate_syrup_usage_l3435_343564


namespace circle_equation_k_range_l3435_343591

theorem circle_equation_k_range (x y k : ℝ) :
  (∃ r : ℝ, r > 0 ∧ ∀ x y, x^2 + y^2 - 2*x + y + k = 0 ↔ (x - 1)^2 + (y + 1/2)^2 = r^2) →
  k < 5/4 :=
by sorry

end circle_equation_k_range_l3435_343591


namespace siblings_age_ratio_l3435_343599

theorem siblings_age_ratio : 
  ∀ (henry_age sister_age : ℕ),
  henry_age = 4 * sister_age →
  henry_age + sister_age + 15 = 240 →
  sister_age / 15 = 3 := by
sorry

end siblings_age_ratio_l3435_343599


namespace number_difference_l3435_343521

theorem number_difference (a b c : ℝ) : 
  a = 2 * b ∧ 
  a = 3 * c ∧ 
  (a + b + c) / 3 = 88 → 
  a - c = 96 := by
sorry

end number_difference_l3435_343521


namespace ball_max_height_l3435_343532

/-- The height function of a ball thrown upwards -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 50

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 130 := by
  sorry

end ball_max_height_l3435_343532


namespace hyperbola_theorem_l3435_343530

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The line y = x - 1 -/
def line_equation (p : Point) : Prop :=
  p.y = p.x - 1

/-- Theorem: Given the conditions, the hyperbola has the equation x²/2 - y²/5 = 1 -/
theorem hyperbola_theorem (h : Hyperbola) (f m n : Point) :
  -- Center at origin
  hyperbola_equation h ⟨0, 0⟩ →
  -- Focus at (√7, 0)
  f = ⟨Real.sqrt 7, 0⟩ →
  -- M and N are on the hyperbola and the line
  hyperbola_equation h m ∧ line_equation m →
  hyperbola_equation h n ∧ line_equation n →
  -- Midpoint x-coordinate is -2/3
  (m.x + n.x) / 2 = -2/3 →
  -- The hyperbola equation is x²/2 - y²/5 = 1
  h.a^2 = 2 ∧ h.b^2 = 5 :=
sorry

end hyperbola_theorem_l3435_343530


namespace boat_distance_along_stream_l3435_343546

/-- Represents the distance traveled by a boat in one hour -/
structure BoatTravel where
  speedStillWater : ℝ
  distanceAgainstStream : ℝ
  timeTravel : ℝ

/-- Calculates the distance traveled along the stream -/
def distanceAlongStream (bt : BoatTravel) : ℝ :=
  let streamSpeed := bt.speedStillWater - bt.distanceAgainstStream
  (bt.speedStillWater + streamSpeed) * bt.timeTravel

/-- Theorem: Given the conditions, the boat travels 13 km along the stream -/
theorem boat_distance_along_stream :
  ∀ (bt : BoatTravel),
    bt.speedStillWater = 11 ∧
    bt.distanceAgainstStream = 9 ∧
    bt.timeTravel = 1 →
    distanceAlongStream bt = 13 := by
  sorry


end boat_distance_along_stream_l3435_343546


namespace no_intersection_points_l3435_343537

theorem no_intersection_points (x y : ℝ) : 
  ¬∃ x y, (y = 3 * x^2 - 4 * x + 5) ∧ (y = -x^2 + 6 * x - 8) := by
  sorry

end no_intersection_points_l3435_343537


namespace ten_person_meeting_exchanges_l3435_343504

/-- The number of business card exchanges in a group meeting -/
def business_card_exchanges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 10 people, where each person exchanges business cards
    with every other person exactly once, the total number of exchanges is 45. -/
theorem ten_person_meeting_exchanges :
  business_card_exchanges 10 = 45 := by
  sorry

end ten_person_meeting_exchanges_l3435_343504


namespace min_ab_value_l3435_343533

theorem min_ab_value (a b : ℕ+) 
  (h : (fun x y : ℝ => x^2 + y^2 - 2*a*x + a^2*(1-b)) = 0 ↔ 
       (fun x y : ℝ => x^2 + y^2 - 2*y + 1 - a^2*b) = 0) : 
  (a : ℝ) * (b : ℝ) ≥ (1/2 : ℝ) := by
sorry

end min_ab_value_l3435_343533


namespace reflected_light_ray_equation_l3435_343502

/-- Given an incident light ray following the line y = 2x + 1 and reflecting on the line y = x,
    the equation of the reflected light ray is x - 2y - 1 = 0 -/
theorem reflected_light_ray_equation (x y : ℝ) :
  (y = 2*x + 1) →  -- Incident light ray equation
  (y = x) →        -- Reflection line equation
  (x - 2*y - 1 = 0) -- Reflected light ray equation
  := by sorry

end reflected_light_ray_equation_l3435_343502


namespace halves_to_one_and_half_l3435_343563

theorem halves_to_one_and_half :
  (3 : ℚ) / 2 / ((1 : ℚ) / 2) = 3 :=
sorry

end halves_to_one_and_half_l3435_343563


namespace polynomial_simplification_l3435_343510

theorem polynomial_simplification (x : ℝ) :
  (5 * x^10 + 8 * x^8 + 3 * x^6) + (2 * x^12 + 3 * x^10 + x^8 + 4 * x^6 + 2 * x^2 + 7) =
  2 * x^12 + 8 * x^10 + 9 * x^8 + 7 * x^6 + 2 * x^2 + 7 :=
by sorry

end polynomial_simplification_l3435_343510


namespace unique_common_one_position_l3435_343522

/-- A binary sequence of length n -/
def BinarySequence (n : ℕ) := Fin n → Bool

/-- The property that for any three sequences, there exists a position where all three have a 1 -/
def ThreeSequenceProperty (n : ℕ) (sequences : Finset (BinarySequence n)) : Prop :=
  ∀ s1 s2 s3 : BinarySequence n, s1 ∈ sequences → s2 ∈ sequences → s3 ∈ sequences →
    ∃ p : Fin n, s1 p = true ∧ s2 p = true ∧ s3 p = true

/-- The main theorem to be proved -/
theorem unique_common_one_position
  (n : ℕ) (sequences : Finset (BinarySequence n))
  (h_count : sequences.card = 2^(n-1))
  (h_three : ThreeSequenceProperty n sequences) :
  ∃! p : Fin n, ∀ s ∈ sequences, s p = true :=
sorry

end unique_common_one_position_l3435_343522


namespace probability_after_removal_l3435_343577

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : ℕ)
  (each : ℕ)
  (h1 : total = numbers * each)

/-- Calculates the number of ways to choose 2 cards from n cards -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the probability of selecting a pair from the deck after removing two pairs -/
def probability_of_pair (d : Deck) : ℚ :=
  let remaining := d.total - 4
  let total_choices := choose_two remaining
  let pair_choices := (d.numbers - 1) * choose_two d.each
  pair_choices / total_choices

theorem probability_after_removal (d : Deck) 
  (h2 : d.total = 60) 
  (h3 : d.numbers = 12) 
  (h4 : d.each = 5) : 
  probability_of_pair d = 11 / 154 := by
  sorry

end probability_after_removal_l3435_343577


namespace difference_is_ten_l3435_343518

/-- Properties of a rectangular plot -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_eq : area = 20 * breadth
  breadth_value : breadth = 10

/-- The area of a rectangle -/
def area (plot : RectangularPlot) : ℝ := plot.length * plot.breadth

/-- The difference between length and breadth -/
def length_breadth_difference (plot : RectangularPlot) : ℝ :=
  plot.length - plot.breadth

/-- Theorem: The difference between length and breadth is 10 meters -/
theorem difference_is_ten (plot : RectangularPlot) :
  length_breadth_difference plot = 10 := by
  sorry

end difference_is_ten_l3435_343518


namespace unit_digit_of_expression_l3435_343519

theorem unit_digit_of_expression : ∃ n : ℕ, n % 10 = 4 ∧ 
  n = (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) - 1 := by
  sorry

end unit_digit_of_expression_l3435_343519


namespace work_completion_time_l3435_343526

theorem work_completion_time (days_B : ℝ) (combined_work : ℝ) (combined_days : ℝ) (days_A : ℝ) : 
  days_B = 45 →
  combined_work = 7 / 18 →
  combined_days = 7 →
  (1 / days_A + 1 / days_B) * combined_days = combined_work →
  days_A = 90 := by
sorry

end work_completion_time_l3435_343526


namespace paul_picked_72_cans_l3435_343596

/-- The number of cans Paul picked up on Saturday and Sunday --/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem stating that Paul picked up 72 cans in total --/
theorem paul_picked_72_cans :
  total_cans 6 3 8 = 72 := by
  sorry

end paul_picked_72_cans_l3435_343596


namespace ranch_cows_count_l3435_343514

/-- Represents the number of cows and horses a rancher has -/
structure RanchAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the conditions of the ranch -/
def ranchConditions (animals : RanchAnimals) : Prop :=
  animals.cows = 5 * animals.horses ∧ animals.cows + animals.horses = 168

theorem ranch_cows_count :
  ∃ (animals : RanchAnimals), ranchConditions animals ∧ animals.cows = 140 := by
  sorry

end ranch_cows_count_l3435_343514


namespace vector_at_t_zero_l3435_343524

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point : ℝ → ℝ × ℝ × ℝ

/-- Given conditions for the parameterized line -/
def line_conditions (L : ParameterizedLine) : Prop :=
  L.point 1 = (2, 5, 7) ∧ L.point 4 = (8, -7, 1)

theorem vector_at_t_zero 
  (L : ParameterizedLine) 
  (h : line_conditions L) : 
  L.point 0 = (0, 9, 9) := by
  sorry

end vector_at_t_zero_l3435_343524


namespace remaining_area_approx_l3435_343542

/-- Represents a circular grass plot with a straight path cutting through it. -/
structure GrassPlot where
  diameter : ℝ
  pathWidth : ℝ
  pathEdgeDistance : ℝ

/-- Calculates the remaining grass area of the plot after the path is cut through. -/
def remainingGrassArea (plot : GrassPlot) : ℝ :=
  sorry

/-- Theorem stating that for the given dimensions, the remaining grass area is approximately 56π + 17 square feet. -/
theorem remaining_area_approx (plot : GrassPlot) 
  (h1 : plot.diameter = 16)
  (h2 : plot.pathWidth = 4)
  (h3 : plot.pathEdgeDistance = 2) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |remainingGrassArea plot - (56 * Real.pi + 17)| < ε :=
sorry

end remaining_area_approx_l3435_343542


namespace specific_tournament_balls_used_l3435_343534

/-- A tennis tournament with specific rules for ball usage -/
structure TennisTournament where
  rounds : Nat
  games_per_round : List Nat
  cans_per_game : Nat
  balls_per_can : Nat

/-- Calculate the total number of tennis balls used in a tournament -/
def total_balls_used (t : TennisTournament) : Nat :=
  (t.games_per_round.sum * t.cans_per_game * t.balls_per_can)

/-- Theorem: The total number of tennis balls used in the specific tournament is 225 -/
theorem specific_tournament_balls_used :
  let t : TennisTournament := {
    rounds := 4,
    games_per_round := [8, 4, 2, 1],
    cans_per_game := 5,
    balls_per_can := 3
  }
  total_balls_used t = 225 := by
  sorry


end specific_tournament_balls_used_l3435_343534


namespace system_solution_l3435_343516

theorem system_solution (x y : ℚ) : 
  (10 / (2 * x + 3 * y - 29) + 9 / (7 * x - 8 * y + 24) = 8) ∧ 
  ((2 * x + 3 * y - 29) / 2 = (7 * x - 8 * y) / 3 + 8) → 
  x = 5 ∧ y = 7 := by
sorry

end system_solution_l3435_343516


namespace village_population_equality_l3435_343523

/-- The initial population of Village 1 -/
def initial_population_village1 : ℕ := 68000

/-- The yearly decrease in population of Village 1 -/
def yearly_decrease_village1 : ℕ := 1200

/-- The initial population of Village 2 -/
def initial_population_village2 : ℕ := 42000

/-- The yearly increase in population of Village 2 -/
def yearly_increase_village2 : ℕ := 800

/-- The number of years after which the populations are equal -/
def years_until_equal : ℕ := 13

theorem village_population_equality :
  initial_population_village1 - yearly_decrease_village1 * years_until_equal =
  initial_population_village2 + yearly_increase_village2 * years_until_equal :=
by sorry

end village_population_equality_l3435_343523


namespace result_has_five_digits_l3435_343517

-- Define a nonzero digit type
def NonzeroDigit := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

-- Define the operation
def operation (A B C : NonzeroDigit) : ℕ :=
  9876 + A.val * 100 + 54 + B.val * 10 + 2 - C.val

-- Theorem statement
theorem result_has_five_digits (A B C : NonzeroDigit) :
  10000 ≤ operation A B C ∧ operation A B C < 100000 :=
sorry

end result_has_five_digits_l3435_343517


namespace half_animals_are_goats_l3435_343529

/-- The number of cows the farmer has initially -/
def initial_cows : ℕ := 7

/-- The number of sheep the farmer has initially -/
def initial_sheep : ℕ := 8

/-- The number of goats the farmer has initially -/
def initial_goats : ℕ := 6

/-- The total number of animals initially -/
def initial_total : ℕ := initial_cows + initial_sheep + initial_goats

/-- The number of goats to be bought -/
def goats_to_buy : ℕ := 9

/-- Theorem stating that buying 9 goats will make half of the animals goats -/
theorem half_animals_are_goats : 
  2 * (initial_goats + goats_to_buy) = initial_total + goats_to_buy := by
  sorry

#check half_animals_are_goats

end half_animals_are_goats_l3435_343529


namespace all_blue_figures_are_small_l3435_343547

-- Define the universe of shapes
inductive Shape
| Square
| Triangle

-- Define colors
inductive Color
| Blue
| Red

-- Define sizes
inductive Size
| Large
| Small

-- Define a figure as a combination of shape, color, and size
structure Figure where
  shape : Shape
  color : Color
  size : Size

-- State the conditions
axiom large_is_square : 
  ∀ (f : Figure), f.size = Size.Large → f.shape = Shape.Square

axiom blue_is_triangle : 
  ∀ (f : Figure), f.color = Color.Blue → f.shape = Shape.Triangle

-- Theorem to prove
theorem all_blue_figures_are_small : 
  ∀ (f : Figure), f.color = Color.Blue → f.size = Size.Small :=
sorry

end all_blue_figures_are_small_l3435_343547


namespace opposite_of_neg_six_l3435_343525

/-- The opposite of a real number -/
def opposite (a : ℝ) : ℝ := -a

/-- Theorem: The opposite of -6 is 6 -/
theorem opposite_of_neg_six : opposite (-6) = 6 := by
  sorry

end opposite_of_neg_six_l3435_343525


namespace original_number_proof_l3435_343559

theorem original_number_proof (x : ℝ) : 
  (x + 0.375 * x) - (x - 0.425 * x) = 85 → x = 106.25 := by
  sorry

end original_number_proof_l3435_343559


namespace knights_seating_probability_formula_l3435_343515

/-- The probability of three knights being seated at a round table with n chairs
    such that there is an empty chair on either side of each knight. -/
def knights_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4) * (n - 5) / ((n - 1) * (n - 2))
  else
    0

/-- Theorem stating that the probability of three knights being seated at a round table
    with n chairs (where n ≥ 6) such that there is an empty chair on either side of
    each knight is equal to (n-4)(n-5) / ((n-1)(n-2)). -/
theorem knights_seating_probability_formula (n : ℕ) (h : n ≥ 6) :
  knights_seating_probability n = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by sorry

end knights_seating_probability_formula_l3435_343515


namespace candy_cookies_per_tray_l3435_343527

/-- Represents the cookie distribution problem --/
structure CookieDistribution where
  num_trays : ℕ
  num_packs : ℕ
  cookies_per_pack : ℕ
  has_equal_trays : Bool

/-- The number of cookies in each tray given the distribution --/
def cookies_per_tray (d : CookieDistribution) : ℕ :=
  (d.num_packs * d.cookies_per_pack) / d.num_trays

/-- Theorem stating the number of cookies per tray in Candy's distribution --/
theorem candy_cookies_per_tray :
  let d : CookieDistribution := {
    num_trays := 4,
    num_packs := 8,
    cookies_per_pack := 12,
    has_equal_trays := true
  }
  cookies_per_tray d = 24 := by
  sorry


end candy_cookies_per_tray_l3435_343527


namespace divisibility_of_S_l3435_343584

-- Define the conditions
def is_valid_prime_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p > 3 ∧ q > 3 ∧ ∃ n : ℕ, q - p = 2^n ∨ p - q = 2^n

-- Define the function S
def S (p q m : ℕ) : ℕ := p^(2*m+1) + q^(2*m+1)

-- State the theorem
theorem divisibility_of_S (p q : ℕ) (h : is_valid_prime_pair p q) :
  ∀ m : ℕ, (3 : ℕ) ∣ S p q m :=
sorry

end divisibility_of_S_l3435_343584


namespace ball_drawing_problem_l3435_343513

theorem ball_drawing_problem (n : ℕ+) : 
  (3 * n) / ((n + 3) * (n + 2) : ℝ) = 7 / 30 → n = 7 := by
  sorry

end ball_drawing_problem_l3435_343513


namespace sum_local_values_2345_l3435_343539

/-- The local value of a digit in a number based on its position -/
def local_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

/-- The sum of local values of digits in a four-digit number -/
def sum_local_values (d₁ d₂ d₃ d₄ : ℕ) : ℕ :=
  local_value d₁ 3 + local_value d₂ 2 + local_value d₃ 1 + local_value d₄ 0

/-- Theorem: The sum of local values of digits in 2345 is 2345 -/
theorem sum_local_values_2345 : sum_local_values 2 3 4 5 = 2345 := by
  sorry

#eval sum_local_values 2 3 4 5

end sum_local_values_2345_l3435_343539


namespace determinant_of_specific_matrix_l3435_343544

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 0, -4; 3, -1, 5; 1, 2, 3]
  Matrix.det A = -54 := by
  sorry

end determinant_of_specific_matrix_l3435_343544


namespace ratio_b_to_a_is_one_l3435_343552

/-- An arithmetic sequence with first four terms a, b, x, and 2x - 1/2 -/
structure ArithmeticSequence (a b x : ℝ) : Prop where
  term1 : a = a
  term2 : b = b
  term3 : x = x
  term4 : 2 * x - 1/2 = 2 * x - 1/2
  is_arithmetic : ∃ (d : ℝ), b - a = d ∧ x - b = d ∧ (2 * x - 1/2) - x = d

/-- The ratio of b to a in the arithmetic sequence is 1 -/
theorem ratio_b_to_a_is_one {a b x : ℝ} (h : ArithmeticSequence a b x) : b / a = 1 := by
  sorry

end ratio_b_to_a_is_one_l3435_343552


namespace hypergeom_expected_and_variance_l3435_343586

/-- Hypergeometric distribution parameters -/
structure HyperGeomParams where
  N : ℕ  -- Population size
  K : ℕ  -- Number of success states in the population
  n : ℕ  -- Number of draws
  h1 : K ≤ N
  h2 : n ≤ N

/-- Expected value of a hypergeometric distribution -/
def expected_value (p : HyperGeomParams) : ℚ :=
  (p.n * p.K : ℚ) / p.N

/-- Variance of a hypergeometric distribution -/
def variance (p : HyperGeomParams) : ℚ :=
  (p.n * p.K * (p.N - p.K) * (p.N - p.n) : ℚ) / (p.N^2 * (p.N - 1))

/-- Theorem: Expected value and variance for the given problem -/
theorem hypergeom_expected_and_variance :
  ∃ (p : HyperGeomParams),
    p.N = 100 ∧ p.K = 10 ∧ p.n = 3 ∧
    expected_value p = 3/10 ∧
    variance p = 51/200 := by
  sorry

end hypergeom_expected_and_variance_l3435_343586


namespace linear_coefficient_of_quadratic_l3435_343565

/-- Given a quadratic equation equivalent to 5x - 2 = 3x^2, 
    prove that the coefficient of the linear term is -5 -/
theorem linear_coefficient_of_quadratic (a b c : ℝ) : 
  (5 : ℝ) * x - 2 = 3 * x^2 → 
  a * x^2 + b * x + c = 0 → 
  c = 2 →
  b = -5 := by
  sorry

#check linear_coefficient_of_quadratic

end linear_coefficient_of_quadratic_l3435_343565


namespace sisters_and_brothers_in_family_l3435_343595

/-- Represents a family with boys and girls -/
structure Family where
  boys : Nat
  girls : Nat

/-- Calculates the number of sisters a girl has in the family (excluding herself) -/
def sisters_of_girl (f : Family) : Nat :=
  f.girls - 1

/-- Calculates the number of brothers a girl has in the family -/
def brothers_of_girl (f : Family) : Nat :=
  f.boys

theorem sisters_and_brothers_in_family (harry_sisters : Nat) (harry_brothers : Nat) :
  harry_sisters = 4 → harry_brothers = 3 →
  ∃ (f : Family),
    f.girls = harry_sisters + 1 ∧
    f.boys = harry_brothers + 1 ∧
    sisters_of_girl f = 3 ∧
    brothers_of_girl f = 3 :=
by sorry

end sisters_and_brothers_in_family_l3435_343595


namespace profit_increase_1995_to_1997_l3435_343556

/-- Represents the financial data of a company over three years -/
structure CompanyFinances where
  R1 : ℝ  -- Revenue in 1995
  E1 : ℝ  -- Expenses in 1995
  P1 : ℝ  -- Profit in 1995
  R2 : ℝ  -- Revenue in 1996
  E2 : ℝ  -- Expenses in 1996
  P2 : ℝ  -- Profit in 1996
  R3 : ℝ  -- Revenue in 1997
  E3 : ℝ  -- Expenses in 1997
  P3 : ℝ  -- Profit in 1997

/-- The profit increase from 1995 to 1997 is 55.25% -/
theorem profit_increase_1995_to_1997 (cf : CompanyFinances)
  (h1 : cf.P1 = cf.R1 - cf.E1)
  (h2 : cf.R2 = 1.20 * cf.R1)
  (h3 : cf.E2 = 1.10 * cf.E1)
  (h4 : cf.P2 = 1.15 * cf.P1)
  (h5 : cf.R3 = 1.25 * cf.R2)
  (h6 : cf.E3 = 1.20 * cf.E2)
  (h7 : cf.P3 = 1.35 * cf.P2) :
  cf.P3 = 1.5525 * cf.P1 := by
  sorry

#check profit_increase_1995_to_1997

end profit_increase_1995_to_1997_l3435_343556


namespace square_sum_of_linear_equations_l3435_343594

theorem square_sum_of_linear_equations (x y : ℝ) 
  (eq1 : 3 * x + y = 20) 
  (eq2 : 4 * x + y = 25) : 
  x^2 + y^2 = 50 := by
  sorry

end square_sum_of_linear_equations_l3435_343594


namespace weekend_finances_correct_l3435_343578

/-- Represents Tom's financial situation over the weekend -/
structure WeekendFinances where
  initial : ℝ  -- Initial amount
  car_wash : ℝ  -- Amount earned from washing cars
  lawn_mow : ℝ  -- Amount earned from mowing lawns
  painting : ℝ  -- Amount earned from painting
  expenses : ℝ  -- Amount spent on gas and food
  final : ℝ  -- Final amount

/-- Theorem stating that Tom's final amount is correctly calculated -/
theorem weekend_finances_correct (tom : WeekendFinances) 
  (h1 : tom.initial = 74)
  (h2 : tom.final = 86) :
  tom.initial + tom.car_wash + tom.lawn_mow + tom.painting - tom.expenses = tom.final := by
  sorry

end weekend_finances_correct_l3435_343578


namespace two_digit_multiplication_sum_l3435_343557

theorem two_digit_multiplication_sum (a b : ℕ) : 
  a ≥ 10 ∧ a < 100 ∧ b ≥ 10 ∧ b < 100 →
  a * (b + 40) = 2496 →
  a * b = 936 →
  a + b = 63 := by
sorry

end two_digit_multiplication_sum_l3435_343557


namespace sixteen_percent_of_forty_percent_of_93_75_l3435_343507

theorem sixteen_percent_of_forty_percent_of_93_75 : 
  (0.16 * (0.4 * 93.75)) = 6 := by
  sorry

end sixteen_percent_of_forty_percent_of_93_75_l3435_343507


namespace P_sufficient_not_necessary_Q_l3435_343593

theorem P_sufficient_not_necessary_Q :
  (∀ x y : ℝ, x + y ≠ 5 → (x ≠ 2 ∨ y ≠ 3)) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 3) ∧ x + y = 5) :=
by sorry

end P_sufficient_not_necessary_Q_l3435_343593


namespace circle_area_diameter_4_l3435_343508

/-- The area of a circle with diameter 4 meters is 4π square meters. -/
theorem circle_area_diameter_4 :
  let diameter : ℝ := 4
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 4 * π :=
by sorry

end circle_area_diameter_4_l3435_343508


namespace range_of_a_l3435_343506

-- Define the conditions
def p (x : ℝ) : Prop := x^2 + 2*x > 3
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a (h : ∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ ∃ x a, ¬(p x) ∧ (q x a)) :
  ∀ a : ℝ, (∃ x : ℝ, q x a) → a ≥ 1 :=
by sorry

end range_of_a_l3435_343506


namespace rectangle_max_area_l3435_343540

/-- A rectangle with perimeter 40 meters has a maximum area of 100 square meters. -/
theorem rectangle_max_area :
  ∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ 2 * (w + h) = 40 ∧
  (∀ (w' h' : ℝ), w' > 0 → h' > 0 → 2 * (w' + h') = 40 → w' * h' ≤ w * h) ∧
  w * h = 100 := by
  sorry

end rectangle_max_area_l3435_343540


namespace final_ratio_is_11_to_14_l3435_343581

/-- Represents the number of students in a school --/
structure School where
  boys : ℕ
  girls : ℕ

def initial_school : School :=
  { boys := 120,
    girls := 160 }

def students_left : School :=
  { boys := 10,
    girls := 20 }

def final_school : School :=
  { boys := initial_school.boys - students_left.boys,
    girls := initial_school.girls - students_left.girls }

theorem final_ratio_is_11_to_14 :
  ∃ (k : ℕ), k > 0 ∧ final_school.boys = 11 * k ∧ final_school.girls = 14 * k :=
sorry

end final_ratio_is_11_to_14_l3435_343581


namespace expression_value_l3435_343589

theorem expression_value (p q r s : ℝ) 
  (h1 : p^2 / q^3 = 4 / 5)
  (h2 : r^3 / s^2 = 7 / 9) :
  11 / (7 - r^3 / s^2) + (2 * q^3 - p^2) / (2 * q^3 + p^2) = 123 / 56 := by
  sorry

end expression_value_l3435_343589


namespace alice_and_bob_savings_l3435_343568

theorem alice_and_bob_savings (alice_money : ℚ) (bob_money : ℚ) :
  alice_money = 2 / 5 →
  bob_money = 1 / 4 →
  2 * (alice_money + bob_money) = 13 / 10 := by
sorry

end alice_and_bob_savings_l3435_343568


namespace ollie_caught_five_fish_l3435_343535

/-- The number of fish caught by Ollie given the fishing results of Patrick and Angus -/
def ollies_fish (patrick_fish : ℕ) (angus_more_than_patrick : ℕ) (ollie_fewer_than_angus : ℕ) : ℕ :=
  patrick_fish + angus_more_than_patrick - ollie_fewer_than_angus

/-- Theorem stating that Ollie caught 5 fish given the problem conditions -/
theorem ollie_caught_five_fish :
  ollies_fish 8 4 7 = 5 := by
  sorry

end ollie_caught_five_fish_l3435_343535


namespace expand_expression_l3435_343501

theorem expand_expression (x y z : ℝ) :
  (2 * x + 15) * (3 * y + 20 * z + 25) = 6 * x * y + 40 * x * z + 50 * x + 45 * y + 300 * z + 375 := by
  sorry

end expand_expression_l3435_343501


namespace sqrt_difference_between_l3435_343541

theorem sqrt_difference_between (a b : ℝ) (h : a < b) : 
  ∃ (n k : ℕ), a < Real.sqrt n - Real.sqrt k ∧ Real.sqrt n - Real.sqrt k < b := by
  sorry

end sqrt_difference_between_l3435_343541


namespace intersection_of_A_and_B_l3435_343567

def A : Set Int := {-2, -1, 0, 1, 2}

def B : Set Int := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_of_A_and_B : A ∩ B = {-2, 0, 2} := by
  sorry

end intersection_of_A_and_B_l3435_343567


namespace convention_handshakes_l3435_343505

/-- The number of handshakes in a convention --/
def number_of_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes for the given convention --/
theorem convention_handshakes :
  number_of_handshakes 5 3 = 90 := by
  sorry

end convention_handshakes_l3435_343505


namespace quiz_show_win_probability_l3435_343558

def num_questions : ℕ := 4
def num_options : ℕ := 3
def min_correct : ℕ := 3

def probability_correct_guess : ℚ := 1 / num_options

/-- The probability of winning the quiz show by answering at least 3 out of 4 questions correctly,
    where each question has 3 options and guesses are random. -/
theorem quiz_show_win_probability :
  (Finset.sum (Finset.range (num_questions - min_correct + 1))
    (fun k => (Nat.choose num_questions (num_questions - k)) *
              (probability_correct_guess ^ (num_questions - k)) *
              ((1 - probability_correct_guess) ^ k))) = 1 / 9 := by
  sorry

end quiz_show_win_probability_l3435_343558


namespace wifes_ring_to_first_ring_ratio_l3435_343570

/-- The cost of Jim's first ring in dollars -/
def first_ring_cost : ℝ := 10000

/-- The cost of Jim's wife's ring in dollars -/
def wifes_ring_cost : ℝ := 20000

/-- The amount Jim is out of pocket in dollars -/
def out_of_pocket : ℝ := 25000

/-- Theorem stating the ratio of the cost of Jim's wife's ring to the cost of the first ring -/
theorem wifes_ring_to_first_ring_ratio :
  wifes_ring_cost / first_ring_cost = 2 :=
by
  have h1 : first_ring_cost + wifes_ring_cost - first_ring_cost / 2 = out_of_pocket := by sorry
  sorry

#check wifes_ring_to_first_ring_ratio

end wifes_ring_to_first_ring_ratio_l3435_343570


namespace cryptarithmetic_solution_l3435_343512

def is_distinct (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def is_digit (n : ℕ) : Prop := n < 10

def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem cryptarithmetic_solution :
  ∃! (X Y B M C : ℕ),
    is_distinct X Y B M C ∧
    is_nonzero_digit X ∧
    is_digit Y ∧
    is_nonzero_digit B ∧
    is_digit M ∧
    is_digit C ∧
    X * 1000 + Y * 100 + 70 + B * 100 + M * 10 + C =
    B * 1000 + M * 100 + C * 10 + 0 :=
by sorry

end cryptarithmetic_solution_l3435_343512


namespace polynomial_divisibility_l3435_343555

theorem polynomial_divisibility (a b c α β γ p : ℤ) (hp : Prime p)
  (h_div_α : p ∣ (a * α^2 + b * α + c))
  (h_div_β : p ∣ (a * β^2 + b * β + c))
  (h_div_γ : p ∣ (a * γ^2 + b * γ + c))
  (h_diff_αβ : ¬(p ∣ (α - β)))
  (h_diff_βγ : ¬(p ∣ (β - γ)))
  (h_diff_γα : ¬(p ∣ (γ - α))) :
  (p ∣ a) ∧ (p ∣ b) ∧ (p ∣ c) ∧ (∀ x : ℤ, p ∣ (a * x^2 + b * x + c)) :=
by sorry

end polynomial_divisibility_l3435_343555


namespace ethel_mental_math_l3435_343588

theorem ethel_mental_math (square_50 : 50^2 = 2500) :
  49^2 = 2500 - 99 := by
  sorry

end ethel_mental_math_l3435_343588


namespace certain_number_problem_l3435_343548

theorem certain_number_problem : ∃! x : ℝ, ((x - 50) / 4) * 3 + 28 = 73 := by
  sorry

end certain_number_problem_l3435_343548


namespace blocks_and_colors_l3435_343560

theorem blocks_and_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (colors_used : ℕ) : 
  total_blocks = 49 → 
  blocks_per_color = 7 → 
  total_blocks = blocks_per_color * colors_used → 
  colors_used = 7 := by
sorry

end blocks_and_colors_l3435_343560


namespace other_x_intercept_of_quadratic_l3435_343550

/-- Given a quadratic function f(x) = ax^2 + bx + c with vertex (5, -3) and
    one x-intercept at (1, 0), the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept_of_quadratic 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 5 = -3 ∧ ∀ x, f x ≤ f 5) -- Vertex condition
  (h3 : f 1 = 0) -- Given x-intercept
  : ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 9 :=
by sorry

end other_x_intercept_of_quadratic_l3435_343550


namespace trig_expression_equals_one_l3435_343503

theorem trig_expression_equals_one (d : ℝ) (h : d = 2 * Real.pi / 13) :
  (Real.sin (4 * d) * Real.sin (7 * d) * Real.sin (11 * d) * Real.sin (14 * d) * Real.sin (17 * d)) /
  (Real.sin d * Real.sin (2 * d) * Real.sin (4 * d) * Real.sin (5 * d) * Real.sin (6 * d)) = 1 := by
  sorry

end trig_expression_equals_one_l3435_343503


namespace nonagon_diagonals_count_l3435_343562

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ := 27

/-- A convex nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- The number of vertices each vertex can connect to (excluding itself and adjacent vertices) -/
def connections_per_vertex : ℕ := nonagon_sides - 3

theorem nonagon_diagonals_count :
  num_diagonals_nonagon = (nonagon_sides * connections_per_vertex) / 2 := by
  sorry

end nonagon_diagonals_count_l3435_343562


namespace no_function_satisfies_property_l3435_343538

-- Define the property that we want to disprove
def HasProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) = x^2 - 2

-- Theorem stating that no such function exists
theorem no_function_satisfies_property :
  ¬ ∃ f : ℝ → ℝ, HasProperty f :=
sorry

end no_function_satisfies_property_l3435_343538


namespace number_division_problem_l3435_343575

theorem number_division_problem (x : ℚ) : x / 5 = 70 + x / 6 → x = 2100 := by
  sorry

end number_division_problem_l3435_343575


namespace mixed_lubricant_price_l3435_343511

/-- Represents an oil type with its volume, price, and discount or tax -/
structure OilType where
  volume : ℝ
  price : ℝ
  discount_or_tax : ℝ
  is_discount : Bool

/-- Calculates the total cost of an oil type after applying discount or tax -/
def calculate_cost (oil : OilType) : ℝ :=
  let base_cost := oil.volume * oil.price
  if oil.is_discount then
    base_cost * (1 - oil.discount_or_tax)
  else
    base_cost * (1 + oil.discount_or_tax)

/-- Theorem stating that the final price per litre of the mixed lubricant oil is approximately 52.80 -/
theorem mixed_lubricant_price (oils : List OilType) 
  (h1 : oils.length = 6)
  (h2 : oils[0] = OilType.mk 70 43 0.15 true)
  (h3 : oils[1] = OilType.mk 50 51 0.10 false)
  (h4 : oils[2] = OilType.mk 15 60 0.08 true)
  (h5 : oils[3] = OilType.mk 25 62 0.12 false)
  (h6 : oils[4] = OilType.mk 40 67 0.05 true)
  (h7 : oils[5] = OilType.mk 10 75 0.18 true) :
  let total_cost := oils.map calculate_cost |>.sum
  let total_volume := oils.map (·.volume) |>.sum
  abs (total_cost / total_volume - 52.80) < 0.01 := by
  sorry

end mixed_lubricant_price_l3435_343511


namespace divisor_sum_implies_exponent_sum_l3435_343549

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ :=
  (a * (r^(n+1) - 1)) / (r - 1)

def sum_of_divisors (i j : ℕ) : ℕ :=
  (sum_of_geometric_series 1 2 i) * (sum_of_geometric_series 1 5 j)

theorem divisor_sum_implies_exponent_sum (i j : ℕ) :
  sum_of_divisors i j = 930 → i + j = 6 :=
by
  sorry

end divisor_sum_implies_exponent_sum_l3435_343549


namespace shanna_garden_harvest_l3435_343566

/-- Represents Shanna's garden --/
structure Garden where
  tomato : ℕ
  eggplant : ℕ
  pepper : ℕ

/-- Calculates the number of vegetables harvested from Shanna's garden --/
def harvest_vegetables (g : Garden) : ℕ :=
  let remaining_tomato := g.tomato / 2
  let remaining_pepper := g.pepper - 1
  let remaining_eggplant := g.eggplant
  let total_remaining := remaining_tomato + remaining_pepper + remaining_eggplant
  total_remaining * 7

/-- Theorem stating the total number of vegetables harvested from Shanna's garden --/
theorem shanna_garden_harvest :
  let initial_garden : Garden := ⟨6, 2, 4⟩
  harvest_vegetables initial_garden = 56 := by
  sorry

end shanna_garden_harvest_l3435_343566


namespace walk_distance_l3435_343580

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the final position after walking in four segments -/
def finalPosition (d : ℝ) : Point :=
  { x := d + d,  -- East distance: second segment + fourth segment
    y := -d + d + d }  -- South, then North, then North again

/-- Theorem stating that if the final position is 40 meters north of the start,
    then the distance walked in each segment must be 40 meters -/
theorem walk_distance (d : ℝ) :
  (finalPosition d).y = 40 → d = 40 := by sorry

end walk_distance_l3435_343580


namespace santa_candy_distribution_l3435_343543

theorem santa_candy_distribution (n : ℕ) (total_candies left_candies : ℕ) :
  3 < n ∧ n < 15 →
  total_candies = 195 →
  left_candies = 8 →
  ∃ k : ℕ, k * n = total_candies - left_candies ∧ k = 17 := by
  sorry

end santa_candy_distribution_l3435_343543


namespace f_even_implies_a_zero_min_value_when_a_greater_than_two_l3435_343569

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + |2*x - a|

-- Theorem 1: If f is even, then a = 0
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by sorry

-- Theorem 2: If a > 2, then the minimum value of f(x) is a - 1
theorem min_value_when_a_greater_than_two (a : ℝ) :
  a > 2 → ∃ m : ℝ, (∀ x : ℝ, f a x ≥ m) ∧ (∃ x : ℝ, f a x = m) ∧ m = a - 1 := by sorry

end f_even_implies_a_zero_min_value_when_a_greater_than_two_l3435_343569


namespace sum_of_squares_l3435_343597

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 72) : x^2 + y^2 = 180 := by
  sorry

end sum_of_squares_l3435_343597


namespace circle_radius_l3435_343585

theorem circle_radius (x y : ℝ) : 
  (x^2 - 8*x + y^2 + 6*y + 1 = 0) → 
  ∃ (h k r : ℝ), r = 2 * Real.sqrt 6 ∧ 
    ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 - 8*x + y^2 + 6*y + 1 = 0 :=
by sorry

end circle_radius_l3435_343585


namespace book_page_ratio_l3435_343587

/-- Given a set of books with specific page counts, prove the ratio of pages between middle and shortest books --/
theorem book_page_ratio (longest middle shortest : ℕ) : 
  longest = 396 → 
  shortest = longest / 4 → 
  middle = 297 → 
  middle / shortest = 3 := by
sorry

end book_page_ratio_l3435_343587


namespace quadratic_no_real_roots_l3435_343500

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (pos_p : p > 0) (pos_q : q > 0) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (p_neq_q : p ≠ q)
  (geom_seq : a^2 = p * q)
  (arith_seq : ∃ d : ℝ, b = p + d ∧ c = p + 2*d ∧ q = p + 3*d)
  : ∀ x : ℝ, b * x^2 - 2*a * x + c ≠ 0 :=
by sorry

end quadratic_no_real_roots_l3435_343500


namespace intersection_of_M_and_N_l3435_343590

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by sorry

end intersection_of_M_and_N_l3435_343590


namespace min_value_inequality_l3435_343574

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b + 2 * Real.sqrt (a * b) ≥ 4 := by
  sorry

end min_value_inequality_l3435_343574


namespace expression_value_l3435_343545

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = -4) :
  5 * (x - y)^2 - x * y = 12 := by sorry

end expression_value_l3435_343545


namespace quadratic_one_solution_l3435_343579

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, x^2 - k*x + 1 = 0) ↔ k = 2 ∨ k = -2 := by sorry

end quadratic_one_solution_l3435_343579


namespace ratio_equality_l3435_343561

theorem ratio_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  a / b = 4 / 3 := by sorry

end ratio_equality_l3435_343561


namespace number_equality_l3435_343598

theorem number_equality (x : ℚ) : 
  (35 / 100) * x = (30 / 100) * 50 → x = 300 / 7 := by
sorry

end number_equality_l3435_343598


namespace pictures_hanging_l3435_343571

theorem pictures_hanging (total : ℕ) (vertical : ℕ) (horizontal : ℕ) (haphazard : ℕ) : 
  total = 30 →
  vertical = 10 →
  horizontal = total / 2 →
  haphazard = total - vertical - horizontal →
  haphazard = 5 := by
sorry

end pictures_hanging_l3435_343571


namespace sum_of_roots_cubic_l3435_343592

theorem sum_of_roots_cubic : ∃ (A B C : ℝ),
  (3 * A^3 - 9 * A^2 + 6 * A - 4 = 0) ∧
  (3 * B^3 - 9 * B^2 + 6 * B - 4 = 0) ∧
  (3 * C^3 - 9 * C^2 + 6 * C - 4 = 0) ∧
  (A + B + C = 3) := by
  sorry

end sum_of_roots_cubic_l3435_343592


namespace nova_annual_donation_l3435_343509

/-- Nova's monthly donation in dollars -/
def monthly_donation : ℕ := 1707

/-- Number of months in a year -/
def months_in_year : ℕ := 12

/-- Nova's total annual donation in dollars -/
def annual_donation : ℕ := monthly_donation * months_in_year

theorem nova_annual_donation :
  annual_donation = 20484 := by
  sorry

end nova_annual_donation_l3435_343509


namespace remainder_7n_mod_5_l3435_343582

theorem remainder_7n_mod_5 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := by
  sorry

end remainder_7n_mod_5_l3435_343582


namespace expansion_simplification_l3435_343553

theorem expansion_simplification (x y : ℝ) :
  (2*x - y) * (2*x + 3*y) - (x + y) * (x - y) = 3*x^2 + 4*x*y - 2*y^2 := by
  sorry

end expansion_simplification_l3435_343553


namespace exactly_ten_naas_l3435_343536

-- Define the set S
variable (S : Type)

-- Define gib and naa as elements of S
variable (gib naa : S)

-- Define the collection relation
variable (is_collection_of : S → S → Prop)

-- Define the belonging relation
variable (belongs_to : S → S → Prop)

-- P1: Every gib is a collection of naas
axiom P1 : ∀ g : S, (g = gib) → ∃ n : S, (n = naa) ∧ is_collection_of g n

-- P2: Any two distinct gibs have two and only two naas in common
axiom P2 : ∀ g1 g2 : S, (g1 = gib) ∧ (g2 = gib) ∧ (g1 ≠ g2) →
  ∃! n1 n2 : S, (n1 = naa) ∧ (n2 = naa) ∧ (n1 ≠ n2) ∧
  is_collection_of g1 n1 ∧ is_collection_of g1 n2 ∧
  is_collection_of g2 n1 ∧ is_collection_of g2 n2

-- P3: Every naa belongs to three and only three gibs
axiom P3 : ∀ n : S, (n = naa) →
  ∃! g1 g2 g3 : S, (g1 = gib) ∧ (g2 = gib) ∧ (g3 = gib) ∧
  (g1 ≠ g2) ∧ (g2 ≠ g3) ∧ (g1 ≠ g3) ∧
  belongs_to n g1 ∧ belongs_to n g2 ∧ belongs_to n g3

-- P4: There are exactly five gibs
axiom P4 : ∃! g1 g2 g3 g4 g5 : S,
  (g1 = gib) ∧ (g2 = gib) ∧ (g3 = gib) ∧ (g4 = gib) ∧ (g5 = gib) ∧
  (g1 ≠ g2) ∧ (g1 ≠ g3) ∧ (g1 ≠ g4) ∧ (g1 ≠ g5) ∧
  (g2 ≠ g3) ∧ (g2 ≠ g4) ∧ (g2 ≠ g5) ∧
  (g3 ≠ g4) ∧ (g3 ≠ g5) ∧
  (g4 ≠ g5)

-- Theorem: There are exactly ten naas
theorem exactly_ten_naas : ∃! n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 : S,
  (n1 = naa) ∧ (n2 = naa) ∧ (n3 = naa) ∧ (n4 = naa) ∧ (n5 = naa) ∧
  (n6 = naa) ∧ (n7 = naa) ∧ (n8 = naa) ∧ (n9 = naa) ∧ (n10 = naa) ∧
  (n1 ≠ n2) ∧ (n1 ≠ n3) ∧ (n1 ≠ n4) ∧ (n1 ≠ n5) ∧ (n1 ≠ n6) ∧ (n1 ≠ n7) ∧ (n1 ≠ n8) ∧ (n1 ≠ n9) ∧ (n1 ≠ n10) ∧
  (n2 ≠ n3) ∧ (n2 ≠ n4) ∧ (n2 ≠ n5) ∧ (n2 ≠ n6) ∧ (n2 ≠ n7) ∧ (n2 ≠ n8) ∧ (n2 ≠ n9) ∧ (n2 ≠ n10) ∧
  (n3 ≠ n4) ∧ (n3 ≠ n5) ∧ (n3 ≠ n6) ∧ (n3 ≠ n7) ∧ (n3 ≠ n8) ∧ (n3 ≠ n9) ∧ (n3 ≠ n10) ∧
  (n4 ≠ n5) ∧ (n4 ≠ n6) ∧ (n4 ≠ n7) ∧ (n4 ≠ n8) ∧ (n4 ≠ n9) ∧ (n4 ≠ n10) ∧
  (n5 ≠ n6) ∧ (n5 ≠ n7) ∧ (n5 ≠ n8) ∧ (n5 ≠ n9) ∧ (n5 ≠ n10) ∧
  (n6 ≠ n7) ∧ (n6 ≠ n8) ∧ (n6 ≠ n9) ∧ (n6 ≠ n10) ∧
  (n7 ≠ n8) ∧ (n7 ≠ n9) ∧ (n7 ≠ n10) ∧
  (n8 ≠ n9) ∧ (n8 ≠ n10) ∧
  (n9 ≠ n10) :=
sorry

end exactly_ten_naas_l3435_343536


namespace neznaika_expression_problem_l3435_343554

theorem neznaika_expression_problem :
  ∃ (f : ℝ → ℝ → ℝ → ℝ),
    (∀ x y z, f x y z = x / (y - Real.sqrt z)) →
    f 20 2 2 > 30 := by
  sorry

end neznaika_expression_problem_l3435_343554


namespace max_imaginary_part_of_roots_l3435_343528

theorem max_imaginary_part_of_roots (z : ℂ) : 
  z^6 - z^4 + z^2 - z + 1 = 0 →
  ∃ (θ : ℝ), -π/2 ≤ θ ∧ θ ≤ π/2 ∧
  (∀ (w : ℂ), w^6 - w^4 + w^2 - w + 1 = 0 → Complex.im w ≤ Real.sin θ) ∧
  θ = 900 * π / (7 * 180) :=
by sorry

end max_imaginary_part_of_roots_l3435_343528


namespace boys_percentage_l3435_343520

/-- Given a class with a 2:3 ratio of boys to girls and 30 total students,
    prove that 40% of the students are boys. -/
theorem boys_percentage (total_students : ℕ) (boy_girl_ratio : ℚ) : 
  total_students = 30 →
  boy_girl_ratio = 2 / 3 →
  (boy_girl_ratio / (1 + boy_girl_ratio)) * 100 = 40 := by
sorry

end boys_percentage_l3435_343520


namespace annie_spending_l3435_343583

def television_count : ℕ := 5
def television_price : ℕ := 50
def figurine_count : ℕ := 10
def figurine_price : ℕ := 1

def total_spending : ℕ := television_count * television_price + figurine_count * figurine_price

theorem annie_spending :
  total_spending = 260 := by sorry

end annie_spending_l3435_343583


namespace vasims_share_l3435_343576

/-- Represents the share of money for each person -/
structure Share :=
  (amount : ℕ)

/-- Represents the distribution of money among three people -/
structure Distribution :=
  (faruk : Share)
  (vasim : Share)
  (ranjith : Share)

/-- The ratio of the distribution -/
def distribution_ratio (d : Distribution) : Prop :=
  5 * d.faruk.amount = 3 * d.vasim.amount ∧
  6 * d.faruk.amount = 3 * d.ranjith.amount

/-- The difference between the largest and smallest share is 900 -/
def share_difference (d : Distribution) : Prop :=
  d.ranjith.amount - d.faruk.amount = 900

theorem vasims_share (d : Distribution) 
  (h1 : distribution_ratio d) 
  (h2 : share_difference d) : 
  d.vasim.amount = 1500 :=
sorry

end vasims_share_l3435_343576


namespace age_difference_32_12_l3435_343531

/-- The difference in ages between two people given their present ages -/
def age_difference (elder_age younger_age : ℕ) : ℕ :=
  elder_age - younger_age

/-- Theorem stating the age difference between two people with given ages -/
theorem age_difference_32_12 :
  age_difference 32 12 = 20 := by
  sorry

end age_difference_32_12_l3435_343531


namespace complete_square_factorization_l3435_343551

theorem complete_square_factorization :
  ∀ (x : ℝ), x^2 + 2*x + 1 = (x + 1)^2 := by
  sorry

#check complete_square_factorization

end complete_square_factorization_l3435_343551


namespace division_problem_l3435_343573

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
    (h1 : dividend = 3086)
    (h2 : divisor = 85)
    (h3 : remainder = 26)
    (h4 : dividend = divisor * quotient + remainder) :
  quotient = 36 := by
  sorry

end division_problem_l3435_343573


namespace sum_of_special_numbers_l3435_343572

/-- Given one-digit numbers A and B satisfying 8AA4 - BBB = BBBB, prove their sum is 12 -/
theorem sum_of_special_numbers (A B : ℕ) : 
  A < 10 → B < 10 → 
  8000 + 100 * A + 10 * A + 4 - (100 * B + 10 * B + B) = 1000 * B + 100 * B + 10 * B + B →
  A + B = 12 := by
sorry

end sum_of_special_numbers_l3435_343572
