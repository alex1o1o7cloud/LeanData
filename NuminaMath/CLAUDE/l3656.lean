import Mathlib

namespace students_per_school_is_247_l3656_365681

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := 6175

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := total_students / num_schools

/-- Theorem stating that the number of students in each elementary school is 247 -/
theorem students_per_school_is_247 : students_per_school = 247 := by sorry

end students_per_school_is_247_l3656_365681


namespace quadratic_minimum_value_l3656_365617

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_minimum_value 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hmin : ∀ x ∈ Set.Icc (-b / (2 * a)) ((2 * a - b) / (2 * a)), 
    quadratic_function a b c x ≠ (4 * a * c - b^2) / (4 * a)) :
  ∃ x ∈ Set.Icc (-b / (2 * a)) ((2 * a - b) / (2 * a)), 
    quadratic_function a b c x = (4 * a^2 + 4 * a * c - b^2) / (4 * a) ∧
    ∀ y ∈ Set.Icc (-b / (2 * a)) ((2 * a - b) / (2 * a)), 
      quadratic_function a b c y ≥ (4 * a^2 + 4 * a * c - b^2) / (4 * a) :=
by
  sorry

end quadratic_minimum_value_l3656_365617


namespace cubic_roots_sum_l3656_365615

theorem cubic_roots_sum (a b c : ℝ) : 
  (10 * a^3 + 15 * a^2 + 2005 * a + 2010 = 0) →
  (10 * b^3 + 15 * b^2 + 2005 * b + 2010 = 0) →
  (10 * c^3 + 15 * c^2 + 2005 * c + 2010 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 907.125 := by
sorry

end cubic_roots_sum_l3656_365615


namespace doughnuts_eaten_l3656_365643

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of doughnuts in the box initially -/
def initial_doughnuts : ℕ := 2 * dozen

/-- The number of doughnuts remaining -/
def remaining_doughnuts : ℕ := 16

/-- The number of doughnuts eaten by the family -/
def eaten_doughnuts : ℕ := initial_doughnuts - remaining_doughnuts

theorem doughnuts_eaten : eaten_doughnuts = 8 := by
  sorry

end doughnuts_eaten_l3656_365643


namespace weekend_getaway_cost_sharing_l3656_365690

/-- A weekend getaway cost-sharing problem -/
theorem weekend_getaway_cost_sharing 
  (henry_paid linda_paid jack_paid : ℝ)
  (h l : ℝ)
  (henry_paid_amount : henry_paid = 120)
  (linda_paid_amount : linda_paid = 150)
  (jack_paid_amount : jack_paid = 210)
  (total_cost : henry_paid + linda_paid + jack_paid = henry_paid + linda_paid + jack_paid)
  (even_split : (henry_paid + linda_paid + jack_paid) / 3 = henry_paid + h)
  (even_split' : (henry_paid + linda_paid + jack_paid) / 3 = linda_paid + l)
  : h - l = 30 := by sorry

end weekend_getaway_cost_sharing_l3656_365690


namespace new_student_weight_l3656_365682

theorem new_student_weight (initial_count : ℕ) (initial_avg : ℝ) (new_avg : ℝ) :
  initial_count = 19 →
  initial_avg = 15 →
  new_avg = 14.6 →
  (initial_count : ℝ) * initial_avg + (initial_count + 1 : ℝ) * new_avg - (initial_count : ℝ) * initial_avg = 7 :=
by sorry

end new_student_weight_l3656_365682


namespace missing_fraction_sum_l3656_365669

theorem missing_fraction_sum (x : ℚ) : 
  (1/3 : ℚ) + (1/2 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-2/15 : ℚ) + (-17/30 : ℚ) = 
  (13333333333333333 : ℚ) / 100000000000000000 := by
  sorry

end missing_fraction_sum_l3656_365669


namespace number_divided_by_three_l3656_365600

theorem number_divided_by_three : ∃ x : ℤ, x / 3 = x - 24 ∧ x = 72 := by
  sorry

end number_divided_by_three_l3656_365600


namespace octagon_diagonals_l3656_365612

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l3656_365612


namespace smallest_overlap_percentage_l3656_365634

/-- The smallest possible percentage of a population playing both football and basketball,
    given that 85% play football and 75% play basketball. -/
theorem smallest_overlap_percentage (total population_football population_basketball : ℝ) :
  population_football = 0.85 * total →
  population_basketball = 0.75 * total →
  total > 0 →
  ∃ (overlap : ℝ), 
    overlap ≥ 0.60 * total ∧
    overlap ≤ population_football ∧
    overlap ≤ population_basketball ∧
    ∀ (x : ℝ), 
      x ≥ 0 ∧ 
      x ≤ population_football ∧ 
      x ≤ population_basketball ∧ 
      population_football + population_basketball - x ≤ total → 
      x ≥ overlap :=
by
  sorry

end smallest_overlap_percentage_l3656_365634


namespace intersection_complement_equals_set_l3656_365606

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1) / Real.log 10}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem intersection_complement_equals_set :
  N ∩ (Mᶜ) = {x | 0 < x ∧ x ≤ 1} := by sorry

end intersection_complement_equals_set_l3656_365606


namespace prime_pairs_divisibility_l3656_365640

theorem prime_pairs_divisibility (p q : ℕ) : 
  p.Prime ∧ q.Prime ∧ p < 2005 ∧ q < 2005 ∧ 
  (q ∣ p^2 + 8) ∧ (p ∣ q^2 + 8) → 
  ((p = 2 ∧ q = 2) ∨ (p = 881 ∧ q = 89) ∨ (p = 89 ∧ q = 881)) := by
  sorry

end prime_pairs_divisibility_l3656_365640


namespace geometric_sequence_ratio_l3656_365644

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The main theorem -/
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_a3 : a 3 = 2) 
  (h_a4a6 : a 4 * a 6 = 16) : 
  (a 9 - a 11) / (a 5 - a 7) = 4 := by
  sorry


end geometric_sequence_ratio_l3656_365644


namespace smallest_fraction_between_l3656_365602

theorem smallest_fraction_between (a₁ b₁ a₂ b₂ : ℕ) 
  (h₁ : a₁ < b₁) (h₂ : a₂ < b₂) 
  (h₃ : Nat.gcd a₁ b₁ = 1) (h₄ : Nat.gcd a₂ b₂ = 1)
  (h₅ : a₂ * b₁ - a₁ * b₂ = 1) :
  ∃ (n k : ℕ), 
    (∀ (n' k' : ℕ), a₁ * n' < b₁ * k' ∧ b₂ * k' < a₂ * n' → n ≤ n') ∧
    a₁ * n < b₁ * k ∧ b₂ * k < a₂ * n ∧
    n = b₁ + b₂ ∧ k = a₁ + a₂ := by
  sorry

end smallest_fraction_between_l3656_365602


namespace systematic_sampling_theorem_l3656_365667

/-- Calculates the number of sampled students within a given interval using systematic sampling -/
def sampledStudentsInInterval (totalStudents : ℕ) (sampleSize : ℕ) (intervalStart : ℕ) (intervalEnd : ℕ) : ℕ :=
  let intervalSize := intervalEnd - intervalStart + 1
  let samplingInterval := totalStudents / sampleSize
  intervalSize / samplingInterval

theorem systematic_sampling_theorem :
  sampledStudentsInInterval 1221 37 496 825 = 10 := by
  sorry

end systematic_sampling_theorem_l3656_365667


namespace divisibility_of_power_difference_l3656_365698

theorem divisibility_of_power_difference (a b : ℕ) (h : a + b = 61) :
  (61 : ℤ) ∣ (a^100 - b^100) :=
by
  sorry

end divisibility_of_power_difference_l3656_365698


namespace f_monotonicity_and_negativity_l3656_365671

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem f_monotonicity_and_negativity (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y → f a x < f a y) ∨
  (∃ c, c > 0 ∧ 
    (∀ x y, 0 < x ∧ x < y ∧ y < c → f a x < f a y) ∧
    (∀ x y, c < x ∧ x < y → f a y < f a x)) ∧
  (∀ x, x > 0 → f a x < 0) ↔ a > (Real.exp 1)⁻¹ :=
sorry

end f_monotonicity_and_negativity_l3656_365671


namespace positive_number_equation_l3656_365646

theorem positive_number_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^a) (h4 : b = 3*a) : a = Real.sqrt 3 := by
  sorry

end positive_number_equation_l3656_365646


namespace roots_less_than_one_l3656_365639

theorem roots_less_than_one (a b : ℝ) (h : abs a + abs b < 1) :
  ∀ x, x^2 + a*x + b = 0 → abs x < 1 :=
sorry

end roots_less_than_one_l3656_365639


namespace max_value_of_complex_number_l3656_365672

theorem max_value_of_complex_number (z : ℂ) : 
  Complex.abs (z - (3 - I)) = 2 → 
  (∀ w : ℂ, Complex.abs (w - (3 - I)) = 2 → Complex.abs (w + (1 + I)) ≤ Complex.abs (z + (1 + I))) → 
  Complex.abs (z + (1 + I)) = 6 :=
by sorry

end max_value_of_complex_number_l3656_365672


namespace wall_painting_fraction_l3656_365664

theorem wall_painting_fraction :
  let total_wall : ℚ := 1
  let matilda_half : ℚ := 1/2
  let ellie_half : ℚ := 1/2
  let matilda_painted : ℚ := matilda_half * (1/2)
  let ellie_painted : ℚ := ellie_half * (1/3)
  matilda_painted + ellie_painted = 5/12 := by
  sorry

end wall_painting_fraction_l3656_365664


namespace binary_1010_eq_10_l3656_365675

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010₍₂₎ -/
def binary_1010 : List Bool := [false, true, false, true]

theorem binary_1010_eq_10 : binary_to_decimal binary_1010 = 10 := by
  sorry

end binary_1010_eq_10_l3656_365675


namespace quadratic_equation_problem1_quadratic_equation_problem2_l3656_365661

-- Problem 1
theorem quadratic_equation_problem1 (x : ℝ) :
  (x - 5)^2 - 16 = 0 ↔ x = 9 ∨ x = 1 := by sorry

-- Problem 2
theorem quadratic_equation_problem2 (x : ℝ) :
  x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 := by sorry

end quadratic_equation_problem1_quadratic_equation_problem2_l3656_365661


namespace circle_centers_distance_bound_l3656_365652

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Sum of reciprocals of distances between circle centers -/
def sum_reciprocal_distances (circles : List Circle) : ℝ := sorry

/-- No line meets more than two circles -/
def no_line_meets_more_than_two (circles : List Circle) : Prop := sorry

theorem circle_centers_distance_bound (n : ℕ) (circles : List Circle) 
  (h1 : circles.length = n)
  (h2 : ∀ c ∈ circles, c.radius = 1)
  (h3 : no_line_meets_more_than_two circles) :
  sum_reciprocal_distances circles ≤ (n - 1 : ℝ) * Real.pi / 4 := by
  sorry

end circle_centers_distance_bound_l3656_365652


namespace pizza_combinations_l3656_365648

def number_of_toppings : ℕ := 8

theorem pizza_combinations : 
  (number_of_toppings) +                    -- one-topping pizzas
  (number_of_toppings.choose 2) +           -- two-topping pizzas
  (number_of_toppings.choose 3) = 92 :=     -- three-topping pizzas
by sorry

end pizza_combinations_l3656_365648


namespace quadratic_root_value_l3656_365637

theorem quadratic_root_value (c : ℚ) : 
  (∀ x : ℚ, (3/2 * x^2 + 13*x + c = 0) ↔ (x = (-13 + Real.sqrt 23)/3 ∨ x = (-13 - Real.sqrt 23)/3)) →
  c = 146/6 :=
by sorry

end quadratic_root_value_l3656_365637


namespace first_day_exceeding_threshold_l3656_365619

-- Define the growth function for the bacteria colony
def bacteriaCount (n : ℕ) : ℕ := 4 * 3^n

-- Define the threshold
def threshold : ℕ := 200

-- Theorem statement
theorem first_day_exceeding_threshold :
  (∃ n : ℕ, bacteriaCount n > threshold) ∧
  (∀ k : ℕ, k < 4 → bacteriaCount k ≤ threshold) ∧
  (bacteriaCount 4 > threshold) := by
  sorry

end first_day_exceeding_threshold_l3656_365619


namespace x_over_u_value_l3656_365604

theorem x_over_u_value (u v w x : ℝ) 
  (h1 : u / v = 5)
  (h2 : w / v = 3)
  (h3 : w / x = 2 / 3) :
  x / u = 9 / 10 := by
sorry

end x_over_u_value_l3656_365604


namespace least_eight_binary_digits_l3656_365631

def binary_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem least_eight_binary_digits : 
  ∀ k : ℕ, k > 0 → (binary_digits k ≥ 8 → k ≥ 128) ∧ binary_digits 128 = 8 :=
by sorry

end least_eight_binary_digits_l3656_365631


namespace trapezoid_shorter_base_l3656_365628

/-- Represents a trapezoid -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_diagonal_length : ℝ

/-- 
Given a trapezoid where:
- The line joining the midpoints of the diagonals has length 5
- The longer base is 105
Then the shorter base must be 95
-/
theorem trapezoid_shorter_base 
  (t : Trapezoid) 
  (h1 : t.midpoint_diagonal_length = 5) 
  (h2 : t.long_base = 105) : 
  t.short_base = 95 := by
sorry

end trapezoid_shorter_base_l3656_365628


namespace parabola_equation_l3656_365618

/-- Represents a parabola with focus (5,5) and directrix 4x + 9y = 36 -/
structure Parabola where
  focus : ℝ × ℝ := (5, 5)
  directrix : ℝ → ℝ → ℝ := fun x y => 4*x + 9*y - 36

/-- Represents the equation of a conic in general form -/
structure ConicEquation where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

def ConicEquation.isValid (eq : ConicEquation) : Prop :=
  eq.a > 0 ∧ Int.gcd eq.a.natAbs (Int.gcd eq.b.natAbs (Int.gcd eq.c.natAbs (Int.gcd eq.d.natAbs (Int.gcd eq.e.natAbs eq.f.natAbs)))) = 1

/-- The equation of the parabola matches the given conic equation -/
def equationMatches (p : Parabola) (eq : ConicEquation) : Prop :=
  ∀ x y : ℝ, eq.a * x^2 + eq.b * x * y + eq.c * y^2 + eq.d * x + eq.e * y + eq.f = 0 ↔
    (x - p.focus.1)^2 + (y - p.focus.2)^2 = ((4*x + 9*y - 36) / Real.sqrt 97)^2

theorem parabola_equation (p : Parabola) :
  ∃ eq : ConicEquation, eq.isValid ∧ equationMatches p eq ∧
    eq.a = 81 ∧ eq.b = -60 ∧ eq.c = 273 ∧ eq.d = -2162 ∧ eq.e = -5913 ∧ eq.f = 19407 :=
sorry

end parabola_equation_l3656_365618


namespace complement_intersection_M_N_l3656_365694

def M : Set ℝ := {x : ℝ | x ≥ 1}
def N : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}

theorem complement_intersection_M_N :
  (M ∩ N)ᶜ = {x : ℝ | x < 1 ∨ x > 3} := by sorry

end complement_intersection_M_N_l3656_365694


namespace hockey_league_teams_l3656_365678

/-- The number of teams in a hockey league -/
def num_teams : ℕ := 15

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season -/
def total_games : ℕ := 1050

/-- Theorem stating that the number of teams is correct given the conditions -/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) / 2) * games_per_pair = total_games :=
sorry

end hockey_league_teams_l3656_365678


namespace unique_prime_with_same_remainder_l3656_365696

theorem unique_prime_with_same_remainder : 
  ∃! n : ℕ, 
    Prime n ∧ 
    200 < n ∧ 
    n < 300 ∧ 
    ∃ r : ℕ, n % 7 = r ∧ n % 9 = r :=
by sorry

end unique_prime_with_same_remainder_l3656_365696


namespace comic_arrangement_count_l3656_365695

/-- The number of different Spiderman comic books --/
def spiderman_comics : ℕ := 8

/-- The number of different Archie comic books --/
def archie_comics : ℕ := 6

/-- The number of different Garfield comic books --/
def garfield_comics : ℕ := 7

/-- The number of ways to arrange the comic books --/
def arrange_comics : ℕ := spiderman_comics.factorial * (archie_comics - 1).factorial * garfield_comics.factorial * 2

theorem comic_arrangement_count :
  arrange_comics = 4864460800 :=
by sorry

end comic_arrangement_count_l3656_365695


namespace isabellas_haircut_l3656_365665

/-- Given an initial hair length and an amount cut off, 
    calculate the resulting hair length after a haircut. -/
def hair_length_after_cut (initial_length cut_length : ℕ) : ℕ :=
  initial_length - cut_length

/-- Theorem: Isabella's hair length after the haircut is 9 inches. -/
theorem isabellas_haircut : hair_length_after_cut 18 9 = 9 := by
  sorry

end isabellas_haircut_l3656_365665


namespace fraction_less_than_one_l3656_365654

theorem fraction_less_than_one (a b : ℝ) (h1 : a < b) (h2 : b < 0) : b / a < 1 := by
  sorry

end fraction_less_than_one_l3656_365654


namespace polynomial_simplification_l3656_365630

theorem polynomial_simplification (x : ℝ) :
  (3*x - 2) * (5*x^12 + 3*x^11 + 7*x^9 + 3*x^8) =
  15*x^13 - x^12 - 6*x^11 + 21*x^10 - 5*x^9 - 6*x^8 := by
  sorry

end polynomial_simplification_l3656_365630


namespace rachel_winter_clothing_l3656_365684

theorem rachel_winter_clothing (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) : 
  num_boxes = 7 → scarves_per_box = 3 → mittens_per_box = 4 → 
  num_boxes * scarves_per_box + num_boxes * mittens_per_box = 49 := by
  sorry

end rachel_winter_clothing_l3656_365684


namespace charles_share_l3656_365633

/-- The number of sheep in the inheritance problem -/
structure SheepInheritance where
  john : ℕ
  alfred : ℕ
  charles : ℕ
  alfred_more_than_john : alfred = (120 * john) / 100
  alfred_more_than_charles : alfred = (125 * charles) / 100
  john_share : john = 3600

/-- Theorem stating that Charles receives 3456 sheep -/
theorem charles_share (s : SheepInheritance) : s.charles = 3456 := by
  sorry

end charles_share_l3656_365633


namespace slope_of_line_AB_l3656_365653

/-- Given points A(2, 0) and B(3, √3), prove that the slope of line AB is √3 -/
theorem slope_of_line_AB (A B : ℝ × ℝ) : 
  A = (2, 0) → B = (3, Real.sqrt 3) → (B.2 - A.2) / (B.1 - A.1) = Real.sqrt 3 := by
  sorry

end slope_of_line_AB_l3656_365653


namespace dans_team_total_games_l3656_365674

/-- Represents a baseball team's game results -/
structure BaseballTeam where
  wins : ℕ
  losses : ℕ

/-- The total number of games played by a baseball team -/
def total_games (team : BaseballTeam) : ℕ :=
  team.wins + team.losses

/-- Theorem: Dan's high school baseball team played 18 games in total -/
theorem dans_team_total_games :
  ∃ (team : BaseballTeam), team.wins = 15 ∧ team.losses = 3 ∧ total_games team = 18 :=
sorry

end dans_team_total_games_l3656_365674


namespace skitties_remainder_l3656_365610

theorem skitties_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end skitties_remainder_l3656_365610


namespace food_supply_duration_l3656_365627

/-- Proves that given a food supply for 760 men that lasts for x days, 
    if after 2 days 1140 more men join and the food lasts for 8 more days, 
    then x = 20. -/
theorem food_supply_duration (x : ℝ) : 
  (760 * x = (760 + 1140) * 8) → x = 20 := by
  sorry

end food_supply_duration_l3656_365627


namespace mans_downstream_speed_l3656_365677

theorem mans_downstream_speed 
  (upstream_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : upstream_speed = 8) 
  (h2 : stream_speed = 2.5) : 
  upstream_speed + 2 * stream_speed = 13 := by
  sorry

end mans_downstream_speed_l3656_365677


namespace seed_survival_rate_l3656_365609

theorem seed_survival_rate 
  (germination_rate : ℝ) 
  (seedling_probability : ℝ) 
  (h1 : germination_rate = 0.9) 
  (h2 : seedling_probability = 0.81) : 
  ∃ p : ℝ, p = germination_rate ∧ p * germination_rate = seedling_probability :=
by
  sorry

end seed_survival_rate_l3656_365609


namespace street_trees_l3656_365622

theorem street_trees (road_length : ℕ) (tree_interval : ℕ) (h1 : road_length = 2575) (h2 : tree_interval = 25) : 
  (road_length / tree_interval) + 1 = 104 := by
  sorry

end street_trees_l3656_365622


namespace meetings_percentage_of_work_day_l3656_365666

/-- Represents the duration of a work day in minutes -/
def work_day_duration : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_duration : ℕ := 80

/-- Represents the duration of the break between meetings in minutes -/
def break_duration : ℕ := 15

/-- Calculates the total time spent in meetings and on break -/
def total_meeting_time : ℕ :=
  first_meeting_duration + (3 * first_meeting_duration) + break_duration

/-- Theorem stating that the percentage of work day spent in meetings and on break is 56% -/
theorem meetings_percentage_of_work_day :
  (total_meeting_time : ℚ) / work_day_duration * 100 = 56 := by
  sorry

end meetings_percentage_of_work_day_l3656_365666


namespace jason_music_store_expenses_l3656_365680

theorem jason_music_store_expenses :
  let flute : ℝ := 142.46
  let music_tool : ℝ := 8.89
  let song_book : ℝ := 7.00
  let flute_case : ℝ := 35.25
  let music_stand : ℝ := 12.15
  let cleaning_kit : ℝ := 14.99
  let sheet_protectors : ℝ := 3.29
  flute + music_tool + song_book + flute_case + music_stand + cleaning_kit + sheet_protectors = 224.03 := by
  sorry

end jason_music_store_expenses_l3656_365680


namespace nearest_integer_to_three_plus_sqrt_five_fourth_power_l3656_365641

theorem nearest_integer_to_three_plus_sqrt_five_fourth_power :
  ∃ n : ℤ, n = 752 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - (m : ℝ)| := by
  sorry

end nearest_integer_to_three_plus_sqrt_five_fourth_power_l3656_365641


namespace power_difference_mod_eight_l3656_365657

theorem power_difference_mod_eight : 
  (47^1235 - 22^1235) % 8 = 7 := by
  sorry

end power_difference_mod_eight_l3656_365657


namespace prob_not_all_same_five_8sided_dice_l3656_365629

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The number of dice rolled -/
def k : ℕ := 5

/-- The probability that not all k n-sided dice show the same number -/
def prob_not_all_same (n k : ℕ) : ℚ :=
  1 - (n : ℚ) / (n ^ k : ℚ)

/-- Theorem: The probability of not all five 8-sided dice showing the same number is 4095/4096 -/
theorem prob_not_all_same_five_8sided_dice :
  prob_not_all_same n k = 4095 / 4096 := by
  sorry

end prob_not_all_same_five_8sided_dice_l3656_365629


namespace percentage_difference_l3656_365616

theorem percentage_difference : (0.9 * 40) - (0.8 * 30) = 12 := by
  sorry

end percentage_difference_l3656_365616


namespace no_solution_fractional_equation_l3656_365613

theorem no_solution_fractional_equation :
  ∀ x : ℝ, (1 - x) / (x - 2) ≠ 1 / (2 - x) + 1 :=
by sorry

end no_solution_fractional_equation_l3656_365613


namespace unique_square_divisible_by_six_in_range_l3656_365658

theorem unique_square_divisible_by_six_in_range : ∃! x : ℕ, 
  (∃ n : ℕ, x = n^2) ∧ 
  (∃ k : ℕ, x = 6 * k) ∧ 
  50 ≤ x ∧ x ≤ 150 :=
by sorry

end unique_square_divisible_by_six_in_range_l3656_365658


namespace legacy_cleaning_time_l3656_365647

/-- The number of floors in the building -/
def num_floors : ℕ := 4

/-- The number of rooms per floor -/
def rooms_per_floor : ℕ := 10

/-- Legacy's hourly rate in dollars -/
def hourly_rate : ℕ := 15

/-- Total earnings from cleaning all floors in dollars -/
def total_earnings : ℕ := 3600

/-- Time to clean one room in hours -/
def time_per_room : ℚ := 6

theorem legacy_cleaning_time :
  time_per_room = (total_earnings : ℚ) / (hourly_rate * num_floors * rooms_per_floor : ℚ) :=
sorry

end legacy_cleaning_time_l3656_365647


namespace square_sum_equals_eight_l3656_365649

theorem square_sum_equals_eight (m : ℝ) 
  (h : (2018 + m) * (2020 + m) = 2) : 
  (2018 + m)^2 + (2020 + m)^2 = 8 := by
  sorry

end square_sum_equals_eight_l3656_365649


namespace gcd_lcm_pairs_l3656_365662

theorem gcd_lcm_pairs :
  (Nat.gcd 6 12 = 6 ∧ Nat.lcm 6 12 = 12) ∧
  (Nat.gcd 7 8 = 1 ∧ Nat.lcm 7 8 = 56) ∧
  (Nat.gcd 15 20 = 5 ∧ Nat.lcm 15 20 = 60) := by
  sorry

end gcd_lcm_pairs_l3656_365662


namespace min_gcd_of_primes_squared_minus_one_l3656_365693

theorem min_gcd_of_primes_squared_minus_one (p q : Nat) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hp_gt_100 : p > 100) (hq_gt_100 : q > 100) : 
  Nat.gcd (p^2 - 1) (q^2 - 1) ≥ 8 := by
  sorry

end min_gcd_of_primes_squared_minus_one_l3656_365693


namespace bike_price_l3656_365626

theorem bike_price (P : ℝ) : P + 0.1 * P = 82500 → P = 75000 := by
  sorry

end bike_price_l3656_365626


namespace inequality_proof_l3656_365659

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end inequality_proof_l3656_365659


namespace sum_remainder_mod_seven_l3656_365645

theorem sum_remainder_mod_seven
  (a b c : ℕ)
  (ha : 0 < a ∧ a < 7)
  (hb : 0 < b ∧ b < 7)
  (hc : 0 < c ∧ c < 7)
  (h1 : a * b * c % 7 = 1)
  (h2 : 4 * c % 7 = 3)
  (h3 : 5 * b % 7 = (4 + b) % 7) :
  (a + b + c) % 7 = 6 := by
sorry

end sum_remainder_mod_seven_l3656_365645


namespace equation_solution_l3656_365688

theorem equation_solution (x : ℝ) (h_pos : x > 0) :
  7.74 * Real.sqrt (Real.log x / Real.log 5) + (Real.log x / Real.log 5) ^ (1/3) = 2 →
  x = 5 := by
sorry

end equation_solution_l3656_365688


namespace distinct_remainders_mod_14_l3656_365679

theorem distinct_remainders_mod_14 : ∃ (a b c d e : ℕ),
  1 ≤ a ∧ a ≤ 13 ∧
  1 ≤ b ∧ b ≤ 13 ∧
  1 ≤ c ∧ c ≤ 13 ∧
  1 ≤ d ∧ d ≤ 13 ∧
  1 ≤ e ∧ e ≤ 13 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (a * b) % 14 ≠ (a * c) % 14 ∧
  (a * b) % 14 ≠ (a * d) % 14 ∧
  (a * b) % 14 ≠ (a * e) % 14 ∧
  (a * b) % 14 ≠ (b * c) % 14 ∧
  (a * b) % 14 ≠ (b * d) % 14 ∧
  (a * b) % 14 ≠ (b * e) % 14 ∧
  (a * b) % 14 ≠ (c * d) % 14 ∧
  (a * b) % 14 ≠ (c * e) % 14 ∧
  (a * b) % 14 ≠ (d * e) % 14 ∧
  (a * c) % 14 ≠ (a * d) % 14 ∧
  (a * c) % 14 ≠ (a * e) % 14 ∧
  (a * c) % 14 ≠ (b * c) % 14 ∧
  (a * c) % 14 ≠ (b * d) % 14 ∧
  (a * c) % 14 ≠ (b * e) % 14 ∧
  (a * c) % 14 ≠ (c * d) % 14 ∧
  (a * c) % 14 ≠ (c * e) % 14 ∧
  (a * c) % 14 ≠ (d * e) % 14 ∧
  (a * d) % 14 ≠ (a * e) % 14 ∧
  (a * d) % 14 ≠ (b * c) % 14 ∧
  (a * d) % 14 ≠ (b * d) % 14 ∧
  (a * d) % 14 ≠ (b * e) % 14 ∧
  (a * d) % 14 ≠ (c * d) % 14 ∧
  (a * d) % 14 ≠ (c * e) % 14 ∧
  (a * d) % 14 ≠ (d * e) % 14 ∧
  (a * e) % 14 ≠ (b * c) % 14 ∧
  (a * e) % 14 ≠ (b * d) % 14 ∧
  (a * e) % 14 ≠ (b * e) % 14 ∧
  (a * e) % 14 ≠ (c * d) % 14 ∧
  (a * e) % 14 ≠ (c * e) % 14 ∧
  (a * e) % 14 ≠ (d * e) % 14 ∧
  (b * c) % 14 ≠ (b * d) % 14 ∧
  (b * c) % 14 ≠ (b * e) % 14 ∧
  (b * c) % 14 ≠ (c * d) % 14 ∧
  (b * c) % 14 ≠ (c * e) % 14 ∧
  (b * c) % 14 ≠ (d * e) % 14 ∧
  (b * d) % 14 ≠ (b * e) % 14 ∧
  (b * d) % 14 ≠ (c * d) % 14 ∧
  (b * d) % 14 ≠ (c * e) % 14 ∧
  (b * d) % 14 ≠ (d * e) % 14 ∧
  (b * e) % 14 ≠ (c * d) % 14 ∧
  (b * e) % 14 ≠ (c * e) % 14 ∧
  (b * e) % 14 ≠ (d * e) % 14 ∧
  (c * d) % 14 ≠ (c * e) % 14 ∧
  (c * d) % 14 ≠ (d * e) % 14 ∧
  (c * e) % 14 ≠ (d * e) % 14 :=
by sorry

end distinct_remainders_mod_14_l3656_365679


namespace first_price_increase_l3656_365670

theorem first_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.15 = 1.38 → x = 20 := by
  sorry

end first_price_increase_l3656_365670


namespace megans_books_l3656_365620

theorem megans_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 7)
  (h2 : mystery_shelves = 8)
  (h3 : picture_shelves = 2) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 70 :=
by sorry

end megans_books_l3656_365620


namespace triangle_base_length_l3656_365689

/-- Given a triangle with area 36 cm² and height 8 cm, its base length is 9 cm. -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 36 →
  height = 8 →
  area = (base * height) / 2 →
  base = 9 := by
sorry

end triangle_base_length_l3656_365689


namespace tan_45_degrees_equals_one_l3656_365686

theorem tan_45_degrees_equals_one : 
  Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_equals_one_l3656_365686


namespace fraction_power_five_l3656_365638

theorem fraction_power_five : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end fraction_power_five_l3656_365638


namespace first_term_value_l3656_365668

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_value 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a5 : a 5 = 9) 
  (h_a3_a2 : 2 * a 3 = a 2 + 6) : 
  a 1 = -3 := by
sorry

end first_term_value_l3656_365668


namespace perfect_cubes_between_powers_of_three_l3656_365625

theorem perfect_cubes_between_powers_of_three : 
  (Finset.filter (fun n : ℕ => 
    3^5 - 1 ≤ n^3 ∧ n^3 ≤ 3^15 + 1) 
    (Finset.range (Nat.floor (Real.rpow 3 5) + 1))).card = 18 := by
  sorry

end perfect_cubes_between_powers_of_three_l3656_365625


namespace point_distance_inequality_l3656_365642

/-- Given points A(0,2), B(0,1), and D(t,0) with t > 0, and M(x,y) on line segment AD,
    if |AM| ≤ 2|BM| always holds, then t ≥ 2√3/3. -/
theorem point_distance_inequality (t : ℝ) (h_t : t > 0) :
  (∀ x y : ℝ, y = (2*t - 2*x)/t →
    x^2 + (y - 2)^2 ≤ 4 * (x^2 + (y - 1)^2)) →
  t ≥ 2 * Real.sqrt 3 / 3 :=
by sorry

end point_distance_inequality_l3656_365642


namespace product_sign_implication_l3656_365691

theorem product_sign_implication (a b c d : ℝ) :
  (a * b * c * d < 0) →
  (a > 0) →
  (b > c) →
  (d < 0) →
  ((0 < c ∧ c < b) ∨ (c < b ∧ b < 0)) :=
by sorry

end product_sign_implication_l3656_365691


namespace block_partition_l3656_365692

theorem block_partition (n : ℕ) (weights : List ℕ) : 
  weights.length = n →
  (∀ w ∈ weights, 1 ≤ w ∧ w < n) →
  weights.sum < 2 * n →
  ∃ (subset : List ℕ), subset ⊆ weights ∧ subset.sum = n := by
  sorry

end block_partition_l3656_365692


namespace intersection_M_N_l3656_365601

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 > 0}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 2} := by sorry

end intersection_M_N_l3656_365601


namespace complex_magnitude_l3656_365623

theorem complex_magnitude (b : ℝ) : 
  let z : ℂ := (3 - b * Complex.I) / Complex.I
  (z.re = z.im) → Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end complex_magnitude_l3656_365623


namespace cricket_team_age_difference_l3656_365676

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_average_age : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 25)
  (h3 : team_average_age = 23)
  (h4 : ∃ (wicket_keeper_age : ℕ), 
    wicket_keeper_age > captain_age ∧ 
    (team_size : ℝ) * team_average_age = 
      (team_size - 2 : ℝ) * (team_average_age - 1) + captain_age + wicket_keeper_age) :
  ∃ (wicket_keeper_age : ℕ), wicket_keeper_age = captain_age + 5 := by
  sorry

end cricket_team_age_difference_l3656_365676


namespace builder_wage_is_100_l3656_365663

/-- The daily wage of a builder given the construction rates and total cost -/
def builder_daily_wage (builders_per_floor : ℕ) (days_per_floor : ℕ) 
  (total_builders : ℕ) (total_houses : ℕ) (floors_per_house : ℕ) 
  (total_cost : ℕ) : ℚ :=
  (total_cost : ℚ) / (total_builders * total_houses * floors_per_house * days_per_floor : ℚ)

theorem builder_wage_is_100 :
  builder_daily_wage 3 30 6 5 6 270000 = 100 := by sorry

end builder_wage_is_100_l3656_365663


namespace triangle_area_l3656_365611

theorem triangle_area : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (6, 1)
  let C : ℝ × ℝ := (10, 6)
  let v : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)
  let w : ℝ × ℝ := (B.1 - C.1, B.2 - C.2)
  abs (v.1 * w.2 - v.2 * w.1) / 2 = 24 := by
  sorry

end triangle_area_l3656_365611


namespace smallest_perfect_square_tiling_l3656_365655

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length. -/
structure Square where
  side : ℕ

/-- The area of a rectangle. -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- The area of a square. -/
def Square.area (s : Square) : ℕ := s.side * s.side

/-- A rectangle fits in a square if its width and height are both less than or equal to the square's side length. -/
def fits_in (r : Rectangle) (s : Square) : Prop := r.width ≤ s.side ∧ r.height ≤ s.side

/-- A square is perfectly tiled by rectangles if the sum of the areas of the rectangles equals the area of the square. -/
def perfectly_tiled (s : Square) (rs : List Rectangle) : Prop :=
  (rs.map Rectangle.area).sum = s.area

theorem smallest_perfect_square_tiling :
  ∃ (s : Square) (rs : List Rectangle),
    (∀ r ∈ rs, r.width = 3 ∧ r.height = 4) ∧
    perfectly_tiled s rs ∧
    (∀ r ∈ rs, fits_in r s) ∧
    rs.length = 12 ∧
    s.side = 12 ∧
    (∀ (s' : Square) (rs' : List Rectangle),
      (∀ r ∈ rs', r.width = 3 ∧ r.height = 4) →
      perfectly_tiled s' rs' →
      (∀ r ∈ rs', fits_in r s') →
      s'.side ≥ s.side) := by
  sorry

#check smallest_perfect_square_tiling

end smallest_perfect_square_tiling_l3656_365655


namespace point_movement_to_x_axis_l3656_365624

/-- Given a point P with coordinates (m+2, 2m+4) that is moved 2 units up to point Q which lies on the x-axis, prove that the coordinates of Q are (-1, 0) -/
theorem point_movement_to_x_axis (m : ℝ) :
  let P : ℝ × ℝ := (m + 2, 2*m + 4)
  let Q : ℝ × ℝ := (P.1, P.2 + 2)
  Q.2 = 0 → Q = (-1, 0) := by sorry

end point_movement_to_x_axis_l3656_365624


namespace distance_city_A_to_C_l3656_365683

/-- Prove the distance between city A and city C given travel times and speeds -/
theorem distance_city_A_to_C 
  (time_Eddy : ℝ) 
  (time_Freddy : ℝ) 
  (distance_AB : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : time_Eddy = 3) 
  (h2 : time_Freddy = 4) 
  (h3 : distance_AB = 600) 
  (h4 : speed_ratio = 1.7391304347826086) : 
  ∃ distance_AC : ℝ, distance_AC = 460 := by
  sorry

end distance_city_A_to_C_l3656_365683


namespace odd_square_mod_eight_l3656_365621

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := by
  sorry

end odd_square_mod_eight_l3656_365621


namespace reflected_quadrilateral_area_l3656_365632

/-- Represents a convex quadrilateral -/
structure ConvexQuadrilateral where
  area : ℝ
  is_convex : Bool

/-- Represents a point inside a convex quadrilateral -/
structure PointInQuadrilateral where
  quad : ConvexQuadrilateral
  is_inside : Bool

/-- Represents the quadrilateral formed by reflecting a point with respect to the midpoints of a quadrilateral's sides -/
def ReflectedQuadrilateral (p : PointInQuadrilateral) : ConvexQuadrilateral :=
  sorry

/-- The theorem stating that the area of the reflected quadrilateral is twice the area of the original quadrilateral -/
theorem reflected_quadrilateral_area 
  (q : ConvexQuadrilateral) 
  (p : PointInQuadrilateral) 
  (h1 : p.quad = q) 
  (h2 : p.is_inside = true) 
  (h3 : q.is_convex = true) :
  (ReflectedQuadrilateral p).area = 2 * q.area :=
sorry

end reflected_quadrilateral_area_l3656_365632


namespace five_seventeenths_repetend_l3656_365636

def decimal_repetend (n d : ℕ) (repetend : List ℕ) : Prop :=
  ∃ (k : ℕ), (n : ℚ) / d = (k : ℚ) / 10^(repetend.length) + 
    (List.sum (List.zipWith (λ (digit place) => (digit : ℚ) / 10^place) repetend 
    (List.range repetend.length))) / (10^(repetend.length) - 1)

theorem five_seventeenths_repetend :
  decimal_repetend 5 17 [2, 9, 4, 1, 1, 7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5] :=
sorry

end five_seventeenths_repetend_l3656_365636


namespace triangle_perimeter_range_l3656_365687

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : Real :=
  t.a + t.b + t.c

-- Theorem statement
theorem triangle_perimeter_range (t : Triangle) 
  (h1 : t.B = π/3) 
  (h2 : t.b = 2 * Real.sqrt 3) 
  (h3 : t.A > 0) 
  (h4 : t.C > 0) 
  (h5 : t.A + t.B + t.C = π) :
  4 * Real.sqrt 3 < perimeter t ∧ perimeter t ≤ 6 * Real.sqrt 3 := by
  sorry

end triangle_perimeter_range_l3656_365687


namespace sum_of_solutions_x_squared_36_l3656_365656

theorem sum_of_solutions_x_squared_36 (x : ℝ) (h : x^2 = 36) :
  ∃ (y : ℝ), y^2 = 36 ∧ x + y = 0 :=
by
  sorry

end sum_of_solutions_x_squared_36_l3656_365656


namespace tagged_fish_in_second_catch_l3656_365635

/-- Represents the number of fish in a pond -/
def total_fish : ℕ := 3200

/-- Represents the number of fish initially tagged -/
def tagged_fish : ℕ := 80

/-- Represents the number of fish in the second catch -/
def second_catch : ℕ := 80

/-- Calculates the expected number of tagged fish in the second catch -/
def expected_tagged_in_second_catch : ℚ :=
  (tagged_fish : ℚ) * (second_catch : ℚ) / (total_fish : ℚ)

theorem tagged_fish_in_second_catch :
  ⌊expected_tagged_in_second_catch⌋ = 2 := by
  sorry

end tagged_fish_in_second_catch_l3656_365635


namespace complex_number_theorem_l3656_365685

theorem complex_number_theorem (z : ℂ) :
  (∃ (k : ℝ), z / 4 = k * I) →
  Complex.abs z = 2 * Real.sqrt 5 →
  z = 2 * I ∨ z = -2 * I := by sorry

end complex_number_theorem_l3656_365685


namespace card_collection_average_l3656_365603

def card_count (k : ℕ) : ℕ := 2 * k - 1

def total_cards (n : ℕ) : ℕ := n^2

def sum_of_values (n : ℕ) : ℕ := (n * (n + 1) / 2)^2 - (n * (n + 1) * (2 * n + 1) / 6)

def average_value (n : ℕ) : ℚ := (sum_of_values n : ℚ) / (total_cards n : ℚ)

theorem card_collection_average (n : ℕ) :
  n > 0 ∧ average_value n = 100 → n = 10 :=
by sorry

end card_collection_average_l3656_365603


namespace parallelogram_base_length_l3656_365608

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_ratio : ℝ) 
  (base_altitude_angle : ℝ) :
  area = 162 →
  altitude_base_ratio = 2 →
  base_altitude_angle = 60 * π / 180 →
  ∃ (base : ℝ), base = 9 ∧ area = base * (altitude_base_ratio * base) :=
by sorry

end parallelogram_base_length_l3656_365608


namespace sally_orange_balloons_l3656_365614

def initial_orange_balloons : ℕ := 9
def lost_orange_balloons : ℕ := 2

theorem sally_orange_balloons :
  initial_orange_balloons - lost_orange_balloons = 7 :=
by sorry

end sally_orange_balloons_l3656_365614


namespace similar_rectangle_ratio_l3656_365660

/-- Given a rectangle with length 40 meters and width 20 meters, 
    prove that a similar smaller rectangle with an area of 200 square meters 
    has dimensions that are 1/2 of the larger rectangle's dimensions. -/
theorem similar_rectangle_ratio (big_length big_width small_area : ℝ) 
  (h1 : big_length = 40)
  (h2 : big_width = 20)
  (h3 : small_area = 200)
  (h4 : small_area = (big_length * r) * (big_width * r)) 
  (r : ℝ) : r = 1 / 2 := by
  sorry

#check similar_rectangle_ratio

end similar_rectangle_ratio_l3656_365660


namespace center_cell_value_l3656_365651

theorem center_cell_value (a b c d e f g h i : ℝ) 
  (positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ i > 0)
  (row_products : a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1)
  (col_products : a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1)
  (square_products : a * d * e * b = 2 ∧ b * e * f * c = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2) :
  e = 1 := by
sorry

end center_cell_value_l3656_365651


namespace geometric_arithmetic_ratio_l3656_365697

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- Three terms form an arithmetic sequence -/
def arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  arithmetic_sequence (a 4) (a 5) (a 6) →
  q = 1 ∨ q = -2 := by
  sorry

end geometric_arithmetic_ratio_l3656_365697


namespace larger_pile_size_l3656_365673

/-- Given two piles of toys where the total number is 120 and the larger pile
    is twice as big as the smaller pile, the number of toys in the larger pile is 80. -/
theorem larger_pile_size (small : ℕ) (large : ℕ) : 
  small + large = 120 → large = 2 * small → large = 80 := by
  sorry

end larger_pile_size_l3656_365673


namespace school_play_ticket_sales_l3656_365607

/-- Calculates the total sales from school play tickets -/
def total_ticket_sales (student_price adult_price : ℕ) (student_tickets adult_tickets : ℕ) : ℕ :=
  student_price * student_tickets + adult_price * adult_tickets

/-- Theorem: The total sales from the school play tickets is $216 -/
theorem school_play_ticket_sales :
  total_ticket_sales 6 8 20 12 = 216 := by
  sorry

end school_play_ticket_sales_l3656_365607


namespace circle_radius_proof_l3656_365605

theorem circle_radius_proof (r₁ r₂ : ℝ) : 
  r₂ = 2 →                             -- The smaller circle has a radius of 2 cm
  (π * r₁^2) = 4 * (π * r₂^2) →        -- The area of one circle is four times the area of the other
  r₁ = 4 :=                            -- The radius of the larger circle is 4 cm
by sorry

end circle_radius_proof_l3656_365605


namespace quadratic_equal_roots_l3656_365650

theorem quadratic_equal_roots (b c : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + c = 0 ∧ (∀ y : ℝ, y^2 + b*y + c = 0 → y = x)) → 
  b^2 - 2*(1+2*c) = -2 := by
sorry

end quadratic_equal_roots_l3656_365650


namespace circle_and_tangent_lines_l3656_365699

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a : ℝ), (x - a)^2 + (y - 2*a)^2 = 5

-- Define the points A, B, and P
def point_A : ℝ × ℝ := (3, 2)
def point_B : ℝ × ℝ := (1, 6)
def point_P : ℝ × ℝ := (-1, 3)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := 2*x - y + 5 = 0
def tangent_line_2 (x y : ℝ) : Prop := x + 2*y - 5 = 0

theorem circle_and_tangent_lines :
  (circle_C point_A.1 point_A.2) ∧
  (circle_C point_B.1 point_B.2) ∧
  (∀ (x y : ℝ), circle_C x y → (x - 2)^2 + (y - 4)^2 = 5) ∧
  (∀ (x y : ℝ), (tangent_line_1 x y ∨ tangent_line_2 x y) →
    (∃ (t : ℝ), circle_C (t*x + (1-t)*point_P.1) (t*y + (1-t)*point_P.2)) ∧
    (∀ (s : ℝ), s ≠ t → ¬ circle_C (s*x + (1-s)*point_P.1) (s*y + (1-s)*point_P.2))) :=
by sorry

end circle_and_tangent_lines_l3656_365699
