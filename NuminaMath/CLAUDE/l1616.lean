import Mathlib

namespace eleventh_grade_sample_l1616_161641

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def grade_ratio : Fin 3 → ℕ
| 0 => 3  -- 10th grade
| 1 => 3  -- 11th grade
| 2 => 4  -- 12th grade

/-- The total sample size -/
def sample_size : ℕ := 50

/-- Calculates the number of students to be sampled from a specific grade -/
def students_to_sample (grade : Fin 3) : ℕ :=
  (grade_ratio grade * sample_size) / (grade_ratio 0 + grade_ratio 1 + grade_ratio 2)

theorem eleventh_grade_sample :
  students_to_sample 1 = 15 := by
  sorry

end eleventh_grade_sample_l1616_161641


namespace symmetric_points_range_l1616_161657

theorem symmetric_points_range (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, a - x^2 = -(2*x + 1)) → a ∈ Set.Icc (-2) (-1) := by
  sorry

end symmetric_points_range_l1616_161657


namespace circle_area_from_circumference_l1616_161661

/-- The area of a circle with circumference 24 cm is 144/π square centimeters. -/
theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 24 → π * r^2 = 144 / π := by
  sorry

end circle_area_from_circumference_l1616_161661


namespace inequality_proof_l1616_161650

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2*x^6*y^6 ≤ π/2 := by sorry

end inequality_proof_l1616_161650


namespace f_of_2_equals_2_l1616_161645

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

-- State the theorem
theorem f_of_2_equals_2 : f 2 = 2 := by
  sorry

end f_of_2_equals_2_l1616_161645


namespace variable_value_l1616_161610

theorem variable_value (x y : ℤ) (h1 : 2 * x - y = 11) (h2 : 4 * x + y ≠ 17) : y = -9 := by
  sorry

end variable_value_l1616_161610


namespace same_solution_systems_l1616_161604

theorem same_solution_systems (m n : ℝ) : 
  (∃ x y : ℝ, 5*x - 2*y = 3 ∧ m*x + 5*y = 4 ∧ x - 4*y = -3 ∧ 5*x + n*y = 1) →
  m = -1 ∧ n = -4 := by
  sorry

end same_solution_systems_l1616_161604


namespace min_value_given_max_l1616_161614

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem min_value_given_max (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 20) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = -7) :=
by sorry

end min_value_given_max_l1616_161614


namespace q_gt_one_neither_sufficient_nor_necessary_l1616_161664

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- Theorem: "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem q_gt_one_neither_sufficient_nor_necessary :
  ¬(∀ a : ℕ → ℝ, ∀ q : ℝ, GeometricSequence a q → (q > 1 → IncreasingSequence a)) ∧
  ¬(∀ a : ℕ → ℝ, ∀ q : ℝ, GeometricSequence a q → (IncreasingSequence a → q > 1)) :=
sorry

end q_gt_one_neither_sufficient_nor_necessary_l1616_161664


namespace arithmetic_sequence_sum_l1616_161662

/-- An arithmetic sequence with common ratio q ≠ 1 -/
def ArithmeticSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ ∀ n : ℕ, a (n + 1) - a n = q * (a n - a (n - 1))

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  ArithmeticSequence a q →
  (a 1 + a 2 + a 3 + a 4 + a 5 = 6) →
  (a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 = 18) →
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := by
  sorry

end arithmetic_sequence_sum_l1616_161662


namespace shortest_side_length_l1616_161629

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  /-- The length of the shortest side (opposite to 30° angle) -/
  short : ℝ
  /-- The length of the middle side (opposite to 60° angle) -/
  middle : ℝ
  /-- The length of the hypotenuse (opposite to 90° angle) -/
  hypotenuse : ℝ
  /-- The ratio of sides in a 30-60-90 triangle -/
  ratio_prop : short = middle / Real.sqrt 3 ∧ middle = hypotenuse / 2

/-- Theorem: In a 30-60-90 triangle with hypotenuse 30 units, the shortest side is 15 units -/
theorem shortest_side_length (t : Triangle30_60_90) (h : t.hypotenuse = 30) : t.short = 15 := by
  sorry

end shortest_side_length_l1616_161629


namespace product_36_sum_0_l1616_161683

theorem product_36_sum_0 (a b c d e f : ℤ) : 
  a * b * c * d * e * f = 36 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f →
  a + b + c + d + e + f = 0 :=
sorry

end product_36_sum_0_l1616_161683


namespace isosceles_triangle_third_side_l1616_161619

/-- An isosceles triangle with side lengths 4 and 8 has its third side equal to 8 -/
theorem isosceles_triangle_third_side : ∀ (a b c : ℝ),
  a = 4 ∧ b = 8 ∧ (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  (a + b > c ∧ b + c > a ∧ a + c > b) →      -- triangle inequality
  c = 8 := by
  sorry

end isosceles_triangle_third_side_l1616_161619


namespace tangent_problem_l1616_161695

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 1/2) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan (α - π/4) = -3/4 := by
  sorry

end tangent_problem_l1616_161695


namespace train_speed_l1616_161638

/-- Calculates the speed of a train in km/hr given its length and time to pass a tree -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 275) (h2 : time = 11) :
  (length / time) * 3.6 = 90 := by
  sorry

end train_speed_l1616_161638


namespace halloween_goodie_bags_cost_l1616_161668

/-- Represents the minimum cost to purchase Halloween goodie bags --/
def minimum_cost (total_students : ℕ) (vampire_students : ℕ) (pumpkin_students : ℕ) 
  (package_size : ℕ) (package_cost : ℚ) (individual_cost : ℚ) : ℚ :=
  let vampire_packages := (vampire_students + package_size - 1) / package_size
  let pumpkin_packages := pumpkin_students / package_size
  let pumpkin_individual := pumpkin_students % package_size
  let base_cost := (vampire_packages + pumpkin_packages) * package_cost + 
                   pumpkin_individual * individual_cost
  let discounted_cost := if base_cost > 10 then base_cost * (1 - 0.1) else base_cost
  ⌈discounted_cost * 100⌉ / 100

/-- Theorem stating the minimum cost for Halloween goodie bags --/
theorem halloween_goodie_bags_cost :
  minimum_cost 25 11 14 5 3 1 = 14.4 := by
  sorry

end halloween_goodie_bags_cost_l1616_161668


namespace cousins_ages_sum_l1616_161622

theorem cousins_ages_sum : ∃ (a b c d : ℕ), 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧  -- single-digit
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧      -- positive
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧  -- distinct
  ((a * b = 24 ∧ c * d = 35) ∨ (a * c = 24 ∧ b * d = 35) ∨ 
   (a * d = 24 ∧ b * c = 35) ∨ (b * c = 24 ∧ a * d = 35) ∨ 
   (b * d = 24 ∧ a * c = 35) ∨ (c * d = 24 ∧ a * b = 35)) →
  a + b + c + d = 23 := by
sorry

end cousins_ages_sum_l1616_161622


namespace max_blue_points_l1616_161631

theorem max_blue_points (total_spheres : ℕ) (h : total_spheres = 2016) :
  ∃ (red_spheres : ℕ), 
    red_spheres ≤ total_spheres ∧
    red_spheres * (total_spheres - red_spheres) = 1016064 ∧
    ∀ (x : ℕ), x ≤ total_spheres → 
      x * (total_spheres - x) ≤ 1016064 := by
  sorry

end max_blue_points_l1616_161631


namespace problem_solution_l1616_161626

theorem problem_solution (x y z : ℚ) 
  (sum_condition : x + y + z = 120)
  (equal_condition : x + 10 = y - 5 ∧ y - 5 = 4*z) : 
  y = 545/9 := by
sorry

end problem_solution_l1616_161626


namespace quadratic_function_properties_l1616_161676

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end quadratic_function_properties_l1616_161676


namespace conference_handshakes_l1616_161607

/-- The number of unique handshakes in a conference -/
def unique_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a conference of 12 people where each person shakes hands with 6 others,
    there are 36 unique handshakes -/
theorem conference_handshakes :
  unique_handshakes 12 6 = 36 := by
  sorry

end conference_handshakes_l1616_161607


namespace kerrys_age_l1616_161669

/-- Given Kerry's birthday celebration setup, prove his age. -/
theorem kerrys_age :
  ∀ (num_cakes : ℕ) 
    (candles_per_box : ℕ) 
    (cost_per_box : ℚ) 
    (total_cost : ℚ),
  num_cakes = 3 →
  candles_per_box = 12 →
  cost_per_box = 5/2 →
  total_cost = 5 →
  ∃ (age : ℕ),
    age * num_cakes = (total_cost / cost_per_box) * candles_per_box ∧
    age = 8 := by
sorry

end kerrys_age_l1616_161669


namespace simplify_and_evaluate_l1616_161616

theorem simplify_and_evaluate (a : ℕ) (ha : a = 2030) :
  (a + 1 : ℚ) / a - a / (a + 1) = (2 * a + 1 : ℚ) / (a * (a + 1)) ∧
  2 * a + 1 = 4061 := by
  sorry

end simplify_and_evaluate_l1616_161616


namespace games_needed_for_512_players_l1616_161635

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  initial_players : ℕ
  games_played : ℕ

/-- Calculates the number of games needed to declare a champion -/
def games_needed (tournament : SingleEliminationTournament) : ℕ :=
  tournament.initial_players - 1

/-- Theorem: In a single-elimination tournament with 512 initial players,
    511 games are needed to declare a champion -/
theorem games_needed_for_512_players :
  ∀ (tournament : SingleEliminationTournament),
    tournament.initial_players = 512 →
    games_needed tournament = 511 := by
  sorry

#check games_needed_for_512_players

end games_needed_for_512_players_l1616_161635


namespace equation_solutions_l1616_161696

theorem equation_solutions (x m : ℝ) : 
  ((3 * x - m) / 2 - (x + m) / 3 = 5 / 6) →
  (m = -1 → x = 0) ∧
  (x = 5 → (1 / 2) * m^2 + 2 * m = 30) := by
sorry

end equation_solutions_l1616_161696


namespace intersection_distance_l1616_161694

/-- The distance between the intersection points of a parabola and a circle -/
theorem intersection_distance (x1 y1 x2 y2 : ℝ) : 
  (y1^2 = 12*x1) →
  (x1^2 + y1^2 - 4*x1 - 6*y1 = 0) →
  (y2^2 = 12*x2) →
  (x2^2 + y2^2 - 4*x2 - 6*y2 = 0) →
  x1 ≠ x2 ∨ y1 ≠ y2 →
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2) = 3 * 13^(1/2) :=
by sorry

end intersection_distance_l1616_161694


namespace part1_part2_l1616_161602

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 6) (m^2 + m - 2)

-- Part 1: Prove that if z - 2m is purely imaginary, then m = 3
theorem part1 (m : ℝ) : (z m - 2 * m).re = 0 → m = 3 := by sorry

-- Part 2: Prove that if z is in the second quadrant, then m is in (-3, -2) ∪ (1, 2)
theorem part2 (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 → m ∈ Set.Ioo (-3) (-2) ∪ Set.Ioo 1 2 := by sorry

end part1_part2_l1616_161602


namespace henrys_money_l1616_161658

theorem henrys_money (x : ℤ) : 
  (x + 18 - 10 = 19) → (x = 11) := by
  sorry

end henrys_money_l1616_161658


namespace smallest_product_l1616_161601

def digits : List ℕ := [6, 7, 8, 9]

def is_valid_placement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : ℕ) : ℕ := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : ℕ, is_valid_placement a b c d →
  product a b c d ≥ 5372 :=
sorry

end smallest_product_l1616_161601


namespace largest_common_number_l1616_161689

def is_in_first_sequence (x : ℕ) : Prop := ∃ n : ℕ, x = 3 + 8 * n

def is_in_second_sequence (x : ℕ) : Prop := ∃ m : ℕ, x = 5 + 9 * m

def is_in_range (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 150

theorem largest_common_number :
  (is_in_first_sequence 131) ∧
  (is_in_second_sequence 131) ∧
  (is_in_range 131) ∧
  (∀ y : ℕ, y > 131 →
    ¬(is_in_first_sequence y ∧ is_in_second_sequence y ∧ is_in_range y)) :=
by sorry

end largest_common_number_l1616_161689


namespace disney_banquet_revenue_l1616_161637

/-- Calculates the total revenue from ticket sales for a Disney banquet --/
theorem disney_banquet_revenue :
  let total_attendees : ℕ := 586
  let resident_price : ℚ := 12.95
  let non_resident_price : ℚ := 17.95
  let num_residents : ℕ := 219
  let num_non_residents : ℕ := total_attendees - num_residents
  let resident_revenue : ℚ := num_residents * resident_price
  let non_resident_revenue : ℚ := num_non_residents * non_resident_price
  let total_revenue : ℚ := resident_revenue + non_resident_revenue
  total_revenue = 9423.70 := by
  sorry

end disney_banquet_revenue_l1616_161637


namespace equal_digit_probability_l1616_161665

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The probability of rolling a one-digit number on a single die -/
def prob_one_digit : ℚ := 9 / 20

/-- The probability of rolling a two-digit number on a single die -/
def prob_two_digit : ℚ := 11 / 20

/-- The number of ways to choose half the dice -/
def num_combinations : ℕ := (num_dice.choose (num_dice / 2))

/-- The probability of getting an equal number of one-digit and two-digit numbers when rolling 6 20-sided dice -/
theorem equal_digit_probability : 
  (num_combinations : ℚ) * (prob_one_digit ^ (num_dice / 2)) * (prob_two_digit ^ (num_dice / 2)) = 485264 / 1600000 := by
  sorry

end equal_digit_probability_l1616_161665


namespace rowers_who_voted_l1616_161611

theorem rowers_who_voted (num_coaches : ℕ) (votes_per_rower : ℕ) (votes_per_coach : ℕ) : 
  num_coaches = 36 → votes_per_rower = 3 → votes_per_coach = 5 → 
  (num_coaches * votes_per_coach) / votes_per_rower = 60 := by
  sorry

end rowers_who_voted_l1616_161611


namespace age_difference_proof_l1616_161671

theorem age_difference_proof (total_age : ℕ) (ratio_a ratio_b ratio_c ratio_d : ℕ) :
  total_age = 190 ∧ ratio_a = 4 ∧ ratio_b = 3 ∧ ratio_c = 7 ∧ ratio_d = 5 →
  ∃ (x : ℚ), x * (ratio_a + ratio_b + ratio_c + ratio_d) = total_age ∧
             x * ratio_a - x * ratio_b = 10 :=
by
  sorry

end age_difference_proof_l1616_161671


namespace x_axis_ellipse_iff_condition_l1616_161643

/-- An ellipse with foci on the x-axis -/
structure XAxisEllipse where
  k : ℝ
  eq : ∀ (x y : ℝ), x^2 / 2 + y^2 / k = 1

/-- The condition for an ellipse with foci on the x-axis -/
def is_x_axis_ellipse_condition (k : ℝ) : Prop :=
  0 < k ∧ k < 2

/-- The theorem stating that 0 < k < 2 is a necessary and sufficient condition 
    for the equation x^2/2 + y^2/k = 1 to represent an ellipse with foci on the x-axis -/
theorem x_axis_ellipse_iff_condition (e : XAxisEllipse) :
  is_x_axis_ellipse_condition e.k ↔ True :=
sorry

end x_axis_ellipse_iff_condition_l1616_161643


namespace birthday_crayons_count_l1616_161617

/-- The number of crayons Paul got for his birthday -/
def birthday_crayons : ℕ := sorry

/-- The number of crayons Paul got at the end of the school year -/
def school_year_crayons : ℕ := 134

/-- The total number of crayons Paul has now -/
def total_crayons : ℕ := 613

/-- Theorem stating that the number of crayons Paul got for his birthday is 479 -/
theorem birthday_crayons_count : birthday_crayons = 479 := by
  sorry

end birthday_crayons_count_l1616_161617


namespace eleventhDrawnNumber_l1616_161613

/-- Systematic sampling function -/
def systematicSample (totalParticipants : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) (n : ℕ) : ℕ :=
  firstDrawn + (n - 1) * (totalParticipants / sampleSize)

/-- Theorem: 11th number drawn in the systematic sampling -/
theorem eleventhDrawnNumber (totalParticipants : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) :
  totalParticipants = 1000 →
  sampleSize = 50 →
  firstDrawn = 15 →
  systematicSample totalParticipants sampleSize firstDrawn 11 = 215 := by
  sorry

#check eleventhDrawnNumber

end eleventhDrawnNumber_l1616_161613


namespace arithmetic_sequence_common_difference_l1616_161680

/-- An arithmetic sequence with a_2 = 1 and a_5 = 7 has common difference 2 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 2 = 1)  -- Given: a_2 = 1
  (h2 : a 5 = 7)  -- Given: a_5 = 7
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  : a 3 - a 2 = 2 :=  -- Conclusion: The common difference is 2
by
  sorry

end arithmetic_sequence_common_difference_l1616_161680


namespace proportion_equality_l1616_161636

theorem proportion_equality : (5 / 34) / (7 / 48) = (120 / 1547) / (1 / 13) := by
  sorry

end proportion_equality_l1616_161636


namespace rectangle_waste_area_l1616_161677

theorem rectangle_waste_area (x y : ℝ) (h1 : x + 2*y = 7) (h2 : 2*x + 3*y = 11) : 
  let a := Real.sqrt (x^2 + y^2)
  let total_area := 11 * 7
  let waste_area := total_area - 4 * a^2
  let waste_percentage := (waste_area / total_area) * 100
  ∃ ε > 0, abs (waste_percentage - 48) < ε :=
sorry

end rectangle_waste_area_l1616_161677


namespace product_of_real_parts_of_complex_solutions_l1616_161627

theorem product_of_real_parts_of_complex_solutions : ∃ (z₁ z₂ : ℂ),
  (z₁^2 + 2*z₁ = Complex.I) ∧ 
  (z₂^2 + 2*z₂ = Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (Complex.re z₁ * Complex.re z₂ = (1 - Real.sqrt 2) / 2) := by
  sorry

end product_of_real_parts_of_complex_solutions_l1616_161627


namespace tic_tac_toe_tournament_contradiction_l1616_161603

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (participants : ℕ)

/-- Calculates the total number of matches in a single-elimination tournament -/
def total_matches (t : Tournament) : ℕ := t.participants - 1

/-- Represents the claims of some participants -/
structure Claims :=
  (num_claimants : ℕ)
  (matches_per_claimant : ℕ)

/-- Calculates the total number of matches implied by the claims -/
def implied_matches (c : Claims) : ℕ := c.num_claimants * c.matches_per_claimant / 2

theorem tic_tac_toe_tournament_contradiction (t : Tournament) (c : Claims) 
  (h1 : t.participants = 18)
  (h2 : c.num_claimants = 6)
  (h3 : c.matches_per_claimant = 4) :
  implied_matches c ≠ total_matches t :=
sorry

end tic_tac_toe_tournament_contradiction_l1616_161603


namespace fraction_doubled_l1616_161688

theorem fraction_doubled (a b : ℝ) (h : a ≠ b) : 
  (2*a * 2*b) / (2*a - 2*b) = 2 * (a * b / (a - b)) := by
  sorry

end fraction_doubled_l1616_161688


namespace no_real_solutions_l1616_161623

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 4) ∧ (x * y - z^2 = 1) := by
  sorry

end no_real_solutions_l1616_161623


namespace large_doll_price_correct_l1616_161692

def total_spending : ℝ := 350
def price_difference : ℝ := 2
def extra_dolls : ℕ := 20

def large_doll_price : ℝ := 7
def small_doll_price : ℝ := large_doll_price - price_difference

theorem large_doll_price_correct :
  (total_spending / small_doll_price = total_spending / large_doll_price + extra_dolls) ∧
  (large_doll_price > 0) ∧
  (small_doll_price > 0) := by
  sorry

end large_doll_price_correct_l1616_161692


namespace complex_equation_solution_l1616_161685

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ) = (2 + Complex.I) / (1 + a * Complex.I) → a = -2 :=
by sorry

end complex_equation_solution_l1616_161685


namespace undefined_expression_expression_undefined_at_nine_l1616_161655

theorem undefined_expression (x : ℝ) : 
  (x^2 - 18*x + 81 = 0) ↔ (x = 9) := by sorry

theorem expression_undefined_at_nine : 
  ∃! x : ℝ, x^2 - 18*x + 81 = 0 := by sorry

end undefined_expression_expression_undefined_at_nine_l1616_161655


namespace present_age_of_B_present_age_of_B_proof_l1616_161699

/-- The present age of person B given the conditions -/
theorem present_age_of_B : ℕ → ℕ → Prop :=
  fun a b =>
    (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
    (a = b + 4) →              -- A is now 4 years older than B
    b = 34                     -- B's current age is 34

-- The proof is omitted
theorem present_age_of_B_proof : ∃ a b, present_age_of_B a b :=
  sorry

end present_age_of_B_present_age_of_B_proof_l1616_161699


namespace items_per_charge_l1616_161609

def total_items : ℕ := 20
def num_cards : ℕ := 4

theorem items_per_charge :
  total_items / num_cards = 5 := by
  sorry

end items_per_charge_l1616_161609


namespace passengers_taken_at_first_station_l1616_161656

/-- Represents the number of passengers on the train at various points --/
structure TrainPassengers where
  initial : ℕ
  afterFirstDrop : ℕ
  afterFirstPickup : ℕ
  afterSecondDrop : ℕ
  afterSecondPickup : ℕ
  final : ℕ

/-- Represents the passenger flow on the train's journey --/
def trainJourney (x : ℕ) : TrainPassengers :=
  { initial := 270,
    afterFirstDrop := 270 - (270 / 3),
    afterFirstPickup := 270 - (270 / 3) + x,
    afterSecondDrop := (270 - (270 / 3) + x) - ((270 - (270 / 3) + x) / 2),
    afterSecondPickup := (270 - (270 / 3) + x) - ((270 - (270 / 3) + x) / 2) + 12,
    final := 242 }

/-- Theorem stating that 280 passengers were taken at the first station --/
theorem passengers_taken_at_first_station :
  ∃ (x : ℕ), trainJourney x = trainJourney 280 ∧ 
  (trainJourney x).afterSecondPickup = (trainJourney x).final :=
sorry


end passengers_taken_at_first_station_l1616_161656


namespace solution_set_inequality_l1616_161646

/-- Given that the solution set of x^2 + ax + b > 0 is (-∞, -2) ∪ (-1/2, +∞),
    prove that the solution set of bx^2 + ax + 1 < 0 is (-2, -1/2) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, x^2 + a*x + b > 0 ↔ x < -2 ∨ x > -1/2) →
  (∀ x, b*x^2 + a*x + 1 < 0 ↔ -2 < x ∧ x < -1/2) :=
by sorry

end solution_set_inequality_l1616_161646


namespace imaginary_part_of_complex_product_l1616_161693

theorem imaginary_part_of_complex_product : 
  let z : ℂ := (1 - Complex.I) * (2 + Complex.I)
  Complex.im z = -1 := by sorry

end imaginary_part_of_complex_product_l1616_161693


namespace conference_arrangement_count_l1616_161615

/-- Represents the number of teachers from each school -/
structure SchoolTeachers :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)

/-- Calculates the number of ways to arrange teachers from different schools -/
def arrangementCount (teachers : SchoolTeachers) : ℕ :=
  sorry

/-- The specific arrangement of teachers from the problem -/
def conferenceTeachers : SchoolTeachers :=
  { A := 2, B := 2, C := 1 }

/-- Theorem stating that the number of valid arrangements is 48 -/
theorem conference_arrangement_count :
  arrangementCount conferenceTeachers = 48 :=
sorry

end conference_arrangement_count_l1616_161615


namespace max_player_salary_l1616_161634

theorem max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  n = 18 →
  min_salary = 20000 →
  total_cap = 600000 →
  ∃ (max_salary : ℕ),
    max_salary = 260000 ∧
    max_salary = total_cap - (n - 1) * min_salary ∧
    max_salary ≥ min_salary ∧
    (n - 1) * min_salary + max_salary ≤ total_cap :=
by sorry

end max_player_salary_l1616_161634


namespace exists_integer_divisible_by_24_with_cube_root_between_9_and_9_5_l1616_161624

theorem exists_integer_divisible_by_24_with_cube_root_between_9_and_9_5 :
  ∃ n : ℕ+, 24 ∣ n ∧ 9 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 9.5 := by
  sorry

end exists_integer_divisible_by_24_with_cube_root_between_9_and_9_5_l1616_161624


namespace sophomore_sample_size_l1616_161600

/-- Calculates the number of sophomores in a stratified sample -/
def sophomores_in_sample (total_students : ℕ) (total_sophomores : ℕ) (sample_size : ℕ) : ℕ :=
  (total_sophomores * sample_size) / total_students

theorem sophomore_sample_size :
  let total_students : ℕ := 4500
  let total_sophomores : ℕ := 1500
  let sample_size : ℕ := 600
  sophomores_in_sample total_students total_sophomores sample_size = 200 := by
  sorry

end sophomore_sample_size_l1616_161600


namespace container_production_l1616_161697

/-- Represents the production rate of containers per worker per hour -/
def container_rate : ℝ := by sorry

/-- Represents the production rate of covers per worker per hour -/
def cover_rate : ℝ := by sorry

/-- The number of containers produced by 80 workers in 2 hours -/
def containers_80_2 : ℝ := 320

/-- The number of covers produced by 80 workers in 2 hours -/
def covers_80_2 : ℝ := 160

/-- The number of containers produced by 100 workers in 3 hours -/
def containers_100_3 : ℝ := 450

/-- The number of covers produced by 100 workers in 3 hours -/
def covers_100_3 : ℝ := 300

/-- The number of covers produced by 40 workers in 4 hours -/
def covers_40_4 : ℝ := 160

theorem container_production :
  80 * 2 * container_rate = containers_80_2 ∧
  80 * 2 * cover_rate = covers_80_2 ∧
  100 * 3 * container_rate = containers_100_3 ∧
  100 * 3 * cover_rate = covers_100_3 ∧
  40 * 4 * cover_rate = covers_40_4 →
  40 * 4 * container_rate = 160 := by sorry

end container_production_l1616_161697


namespace sum_of_coefficients_equals_value_at_one_sum_of_coefficients_is_eight_l1616_161621

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 2 * (4 * x^6 + 9 * x^3 - 5) + 8 * (x^4 - 8 * x + 6)

/-- The sum of coefficients of a polynomial is equal to its value at x = 1 -/
theorem sum_of_coefficients_equals_value_at_one :
  (p 1) = 8 := by sorry

/-- The sum of coefficients of the given polynomial is 8 -/
theorem sum_of_coefficients_is_eight :
  ∃ (f : ℝ → ℝ), (∀ x, f x = p x) ∧ (f 1 = 8) := by sorry

end sum_of_coefficients_equals_value_at_one_sum_of_coefficients_is_eight_l1616_161621


namespace pairwise_product_signs_l1616_161648

theorem pairwise_product_signs (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let products := [a * b, b * c, c * a]
  (products.filter (· > 0)).length = 1 ∨ (products.filter (· > 0)).length = 3 :=
sorry

end pairwise_product_signs_l1616_161648


namespace mountain_trail_length_l1616_161620

/-- Represents the hike on the Mountain Trail -/
structure MountainTrail where
  -- Daily distances hiked
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ
  -- Conditions
  first_two_days : day1 + day2 = 30
  second_third_avg : (day2 + day3) / 2 = 16
  last_three_days : day3 + day4 + day5 = 45
  first_fourth_days : day1 + day4 = 32

/-- The theorem stating the total length of the Mountain Trail -/
theorem mountain_trail_length (hike : MountainTrail) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 107 := by
  sorry


end mountain_trail_length_l1616_161620


namespace carpentry_job_cost_l1616_161679

/-- Calculates the total cost of a carpentry job -/
theorem carpentry_job_cost 
  (hourly_rate : ℕ) 
  (material_cost : ℕ) 
  (estimated_hours : ℕ) 
  (h1 : hourly_rate = 28)
  (h2 : material_cost = 560)
  (h3 : estimated_hours = 15) :
  hourly_rate * estimated_hours + material_cost = 980 := by
sorry

end carpentry_job_cost_l1616_161679


namespace tan_945_degrees_l1616_161667

theorem tan_945_degrees (x : ℝ) : 
  (∀ x, Real.tan (x + 2 * Real.pi) = Real.tan x) → 
  Real.tan (945 * Real.pi / 180) = 1 := by
  sorry

end tan_945_degrees_l1616_161667


namespace ratio_x_to_y_l1616_161691

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 7 * y) / (17 * x - 3 * y) = 4 / 7) :
  x / y = 37 / 16 := by
  sorry

end ratio_x_to_y_l1616_161691


namespace arccos_negative_one_equals_pi_l1616_161672

theorem arccos_negative_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end arccos_negative_one_equals_pi_l1616_161672


namespace simplify_and_evaluate_1_l1616_161630

theorem simplify_and_evaluate_1 (m : ℤ) :
  m = -2023 → 7 * m^2 + 4 - 2 * m^2 - 3 * m - 5 * m^2 - 5 + 4 * m = -2024 := by
  sorry

end simplify_and_evaluate_1_l1616_161630


namespace fourth_root_equation_solution_l1616_161633

theorem fourth_root_equation_solution : 
  ∃ (p q r : ℕ+), 
    4 * (7^(1/4) - 6^(1/4))^(1/4) = p^(1/4) + q^(1/4) - r^(1/4) ∧ 
    p + q + r = 99 := by
  sorry

end fourth_root_equation_solution_l1616_161633


namespace sum_of_distinct_prime_factors_882_l1616_161644

def sum_of_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_distinct_prime_factors_882 :
  sum_of_distinct_prime_factors 882 = 12 := by sorry

end sum_of_distinct_prime_factors_882_l1616_161644


namespace cake_slices_kept_l1616_161687

theorem cake_slices_kept (total_slices : ℕ) (eaten_fraction : ℚ) (kept_slices : ℕ) : 
  total_slices = 12 →
  eaten_fraction = 1/4 →
  kept_slices = total_slices - (total_slices * (eaten_fraction.num / eaten_fraction.den).toNat) →
  kept_slices = 9 := by
  sorry

end cake_slices_kept_l1616_161687


namespace gcd_10010_15015_l1616_161642

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end gcd_10010_15015_l1616_161642


namespace negation_equivalence_l1616_161649

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (2 : ℝ) ^ x < x ^ 2) ↔ (∀ x : ℝ, (2 : ℝ) ^ x ≥ x ^ 2) :=
by sorry

end negation_equivalence_l1616_161649


namespace inequality_preservation_l1616_161628

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end inequality_preservation_l1616_161628


namespace weighted_average_combined_class_l1616_161639

/-- Given two classes of students, prove that the weighted average of the combined class
    is equal to the sum of the products of each class's student count and average mark,
    divided by the total number of students. -/
theorem weighted_average_combined_class
  (n₁ : ℕ) (n₂ : ℕ) (x₁ : ℚ) (x₂ : ℚ)
  (h₁ : n₁ = 58)
  (h₂ : n₂ = 52)
  (h₃ : x₁ = 67)
  (h₄ : x₂ = 82) :
  (n₁ * x₁ + n₂ * x₂) / (n₁ + n₂ : ℚ) = (58 * 67 + 52 * 82) / (58 + 52 : ℚ) :=
by sorry

end weighted_average_combined_class_l1616_161639


namespace chongqing_population_scientific_notation_l1616_161686

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Conversion from a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The population of Chongqing at the end of 2022 -/
def chongqingPopulation : ℕ := 32000000

theorem chongqing_population_scientific_notation :
  toScientificNotation (chongqingPopulation : ℝ) =
    ScientificNotation.mk 3.2 7 (by norm_num) :=
  sorry

end chongqing_population_scientific_notation_l1616_161686


namespace rye_flour_amount_l1616_161653

/-- The amount of rye flour Sarah bought -/
def rye_flour : ℝ := sorry

/-- The amount of whole-wheat bread flour Sarah bought -/
def whole_wheat_bread : ℝ := 10

/-- The amount of chickpea flour Sarah bought -/
def chickpea : ℝ := 3

/-- The amount of whole-wheat pastry flour Sarah had at home -/
def whole_wheat_pastry : ℝ := 2

/-- The total amount of flour Sarah has now -/
def total_flour : ℝ := 20

/-- Theorem stating that the amount of rye flour Sarah bought is 5 pounds -/
theorem rye_flour_amount : rye_flour = 5 := by
  sorry

end rye_flour_amount_l1616_161653


namespace equilateral_triangle_min_rotation_angle_l1616_161663

/-- An equilateral triangle with rotational symmetry -/
class EquilateralTriangle :=
  (rotation_symmetry : Bool)
  (is_equilateral : Bool)

/-- The minimum rotation angle (in degrees) for a shape with rotational symmetry -/
def min_rotation_angle (shape : EquilateralTriangle) : ℝ :=
  sorry

/-- Theorem: The minimum rotation angle for an equilateral triangle with rotational symmetry is 120 degrees -/
theorem equilateral_triangle_min_rotation_angle (t : EquilateralTriangle)
  (h1 : t.rotation_symmetry = true)
  (h2 : t.is_equilateral = true) :
  min_rotation_angle t = 120 :=
sorry

end equilateral_triangle_min_rotation_angle_l1616_161663


namespace sequence_property_l1616_161682

theorem sequence_property (m : ℤ) (a : ℕ → ℤ) (r s : ℕ) : 
  (|m| ≥ 2) →
  (∃ k, a k ≠ 0) →
  (∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n) →
  (r > s) →
  (s ≥ 2) →
  (a r = a s) →
  (a r = a 1) →
  (r - s ≥ |m|) := by
sorry

end sequence_property_l1616_161682


namespace dog_weights_l1616_161684

theorem dog_weights (y z : ℝ) : 
  let dog_weights : List ℝ := [25, 31, 35, 33, y, z]
  (dog_weights.take 4).sum / 4 = dog_weights.sum / 6 →
  y + z = 62 := by
sorry

end dog_weights_l1616_161684


namespace floor_ceil_sum_l1616_161659

theorem floor_ceil_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ + ⌈(0.001 : ℝ)⌉ = 6 := by
  sorry

end floor_ceil_sum_l1616_161659


namespace triangle_shortest_side_l1616_161632

theorem triangle_shortest_side (a b c : ℕ) (h : ℕ) : 
  a = 24 →                                  -- One side is 24
  a + b + c = 66 →                          -- Perimeter is 66
  b ≤ c →                                   -- b is the shortest side
  ∃ (A : ℕ), A * A = 297 * (33 - b) * (b - 9) →  -- Area is an integer (using Heron's formula)
  24 * h = 2 * A →                          -- Integer altitude condition
  b = 15 := by sorry

end triangle_shortest_side_l1616_161632


namespace contractor_payment_proof_l1616_161675

/-- Calculates the total amount received by a contractor given the contract terms and absences. -/
def contractor_payment (total_days : ℕ) (payment_per_day : ℚ) (fine_per_day : ℚ) (absent_days : ℕ) : ℚ :=
  let working_days := total_days - absent_days
  let total_payment := working_days * payment_per_day
  let total_fine := absent_days * fine_per_day
  total_payment - total_fine

/-- Proves that the contractor receives Rs. 425 given the specified conditions. -/
theorem contractor_payment_proof :
  contractor_payment 30 25 7.5 10 = 425 := by
  sorry

end contractor_payment_proof_l1616_161675


namespace blue_balls_count_l1616_161640

def probability_two_red (red green blue : ℕ) : ℚ :=
  (red.choose 2 : ℚ) / ((red + green + blue).choose 2 : ℚ)

theorem blue_balls_count (red green : ℕ) (prob : ℚ) :
  red = 7 →
  green = 4 →
  probability_two_red red green (blue : ℕ) = (175 : ℚ) / 1000 →
  blue = 5 := by
sorry

end blue_balls_count_l1616_161640


namespace softball_team_ratio_l1616_161605

/-- Represents a co-ed softball team -/
structure SoftballTeam where
  men : ℕ
  women : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of men to women on the softball team -/
theorem softball_team_ratio (team : SoftballTeam) : 
  team.women = team.men + 4 → 
  team.men + team.women = 14 → 
  ∃ (r : Ratio), r.numerator = team.men ∧ r.denominator = team.women ∧ r.numerator = 5 ∧ r.denominator = 9 :=
by sorry

end softball_team_ratio_l1616_161605


namespace garden_area_l1616_161666

theorem garden_area (total_posts : ℕ) (post_distance : ℕ) (longer_side_posts : ℕ) (shorter_side_posts : ℕ) :
  total_posts = 24 →
  post_distance = 4 →
  longer_side_posts = 2 * shorter_side_posts →
  longer_side_posts + shorter_side_posts = total_posts + 4 →
  (shorter_side_posts - 1) * post_distance * (longer_side_posts - 1) * post_distance = 576 :=
by sorry

end garden_area_l1616_161666


namespace solve_equation_l1616_161654

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := a * (b ^ (1/2))

-- Theorem statement
theorem solve_equation (x : ℝ) (h : at_op 4 x = 12) : x = 9 := by
  sorry

end solve_equation_l1616_161654


namespace fraction_inequality_l1616_161608

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (b + c) / (a + c) > b / a :=
by sorry

end fraction_inequality_l1616_161608


namespace carolyn_embroiders_50_flowers_l1616_161606

/-- Represents the embroidery problem with given conditions -/
structure EmbroideryProblem where
  stitches_per_minute : ℕ
  stitches_per_flower : ℕ
  stitches_per_unicorn : ℕ
  stitches_for_godzilla : ℕ
  num_unicorns : ℕ
  total_minutes : ℕ

/-- Calculates the number of flowers Carolyn wants to embroider -/
def flowers_to_embroider (p : EmbroideryProblem) : ℕ :=
  let total_stitches := p.stitches_per_minute * p.total_minutes
  let stitches_for_creatures := p.stitches_for_godzilla + p.num_unicorns * p.stitches_per_unicorn
  let remaining_stitches := total_stitches - stitches_for_creatures
  remaining_stitches / p.stitches_per_flower

/-- Theorem stating that given the problem conditions, Carolyn wants to embroider 50 flowers -/
theorem carolyn_embroiders_50_flowers :
  let p := EmbroideryProblem.mk 4 60 180 800 3 1085
  flowers_to_embroider p = 50 := by
  sorry


end carolyn_embroiders_50_flowers_l1616_161606


namespace lemonade_stand_cost_l1616_161612

theorem lemonade_stand_cost (net_profit babysitting_income : ℝ)
  (gross_revenue num_lemonades : ℕ)
  (lemon_cost sugar_cost ice_cost : ℝ)
  (bulk_discount sales_tax sunhat_cost : ℝ) :
  net_profit = 44 →
  babysitting_income = 31 →
  gross_revenue = 47 →
  num_lemonades = 50 →
  lemon_cost = 0.20 →
  sugar_cost = 0.15 →
  ice_cost = 0.05 →
  bulk_discount = 0.10 →
  sales_tax = 0.05 →
  sunhat_cost = 10 →
  ∃ (total_cost : ℝ),
    total_cost = (num_lemonades * (lemon_cost + sugar_cost + ice_cost) -
      num_lemonades * (lemon_cost + sugar_cost) * bulk_discount +
      gross_revenue * sales_tax + sunhat_cost) ∧
    total_cost = 30.60 :=
by sorry

end lemonade_stand_cost_l1616_161612


namespace diagonals_divisible_by_3_count_l1616_161651

/-- A convex polygon with 30 sides -/
structure ConvexPolygon30 where
  sides : ℕ
  convex : Bool
  sides_eq_30 : sides = 30

/-- The number of diagonals in a polygon that are divisible by 3 -/
def diagonals_divisible_by_3 (p : ConvexPolygon30) : ℕ := 17

/-- Theorem stating that the number of diagonals divisible by 3 in a convex 30-sided polygon is 17 -/
theorem diagonals_divisible_by_3_count (p : ConvexPolygon30) : 
  diagonals_divisible_by_3 p = 17 := by sorry

end diagonals_divisible_by_3_count_l1616_161651


namespace trig_expression_equality_l1616_161690

theorem trig_expression_equality : 
  1 / Real.sin (40 * π / 180) - Real.sqrt 2 / Real.cos (40 * π / 180) = 1 / 8 := by
  sorry

end trig_expression_equality_l1616_161690


namespace johns_naps_per_week_l1616_161625

/-- Given that John takes 60 hours of naps in 70 days, and each nap is 2 hours long,
    prove that he takes 3 naps per week. -/
theorem johns_naps_per_week 
  (nap_duration : ℝ) 
  (total_days : ℝ) 
  (total_nap_hours : ℝ) 
  (h1 : nap_duration = 2)
  (h2 : total_days = 70)
  (h3 : total_nap_hours = 60) :
  (total_nap_hours / (total_days / 7)) / nap_duration = 3 :=
by sorry

end johns_naps_per_week_l1616_161625


namespace tan_80_in_terms_of_cos_100_l1616_161618

theorem tan_80_in_terms_of_cos_100 (m : ℝ) (h : Real.cos (100 * π / 180) = m) :
  Real.tan (80 * π / 180) = Real.sqrt (1 - m^2) / (-m) := by
  sorry

end tan_80_in_terms_of_cos_100_l1616_161618


namespace complex_pure_imaginary_l1616_161673

theorem complex_pure_imaginary (m : ℝ) : 
  (m + (10 : ℂ) / (3 + Complex.I)).im ≠ 0 ∧ (m + (10 : ℂ) / (3 + Complex.I)).re = 0 → m = -3 := by
  sorry

end complex_pure_imaginary_l1616_161673


namespace sphere_volume_from_inscribed_cube_l1616_161674

theorem sphere_volume_from_inscribed_cube (s : Real) (r : Real) : 
  (6 * s^2 = 32) →  -- surface area of cube is 32
  (r = s * Real.sqrt 3 / 2) →  -- radius of sphere in terms of cube side length
  (4 / 3 * Real.pi * r^3 = 32 * Real.pi / 3) :=  -- volume of sphere
by sorry

end sphere_volume_from_inscribed_cube_l1616_161674


namespace product_of_numbers_l1616_161670

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end product_of_numbers_l1616_161670


namespace parallelograms_in_triangle_l1616_161652

/-- The number of parallelograms formed inside a triangle -/
def num_parallelograms (n : ℕ) : ℕ := 3 * Nat.choose (n + 2) 4

/-- 
Theorem: The number of parallelograms formed inside a triangle 
whose sides are divided into n equal parts with parallel lines 
drawn through these points is equal to 3 * (n+2 choose 4).
-/
theorem parallelograms_in_triangle (n : ℕ) : 
  num_parallelograms n = 3 * Nat.choose (n + 2) 4 := by
  sorry

end parallelograms_in_triangle_l1616_161652


namespace part_one_part_two_l1616_161678

/-- The function f(x) as defined in the problem -/
def f (a c x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

/-- Part 1 of the theorem -/
theorem part_one (a : ℝ) :
  f a 19 1 > 0 ↔ -2 < a ∧ a < 8 := by sorry

/-- Part 2 of the theorem -/
theorem part_two (a c : ℝ) :
  (∀ x : ℝ, f a c x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9) := by sorry

end part_one_part_two_l1616_161678


namespace consecutive_product_sum_l1616_161647

theorem consecutive_product_sum : ∃ (a b x y z : ℕ), 
  (a + 1 = b) ∧ 
  (x + 1 = y) ∧ 
  (y + 1 = z) ∧
  (a * b = 1320) ∧ 
  (x * y * z = 1320) ∧ 
  (a + b + x + y + z = 106) := by
sorry

end consecutive_product_sum_l1616_161647


namespace cubic_equation_roots_l1616_161698

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 11*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  k + m = 52 := by
sorry

end cubic_equation_roots_l1616_161698


namespace newspaper_probability_l1616_161681

-- Define the time intervals
def delivery_start : ℝ := 6.5
def delivery_end : ℝ := 7.5
def departure_start : ℝ := 7.0
def departure_end : ℝ := 8.0

-- Define the probability function
def probability_of_getting_newspaper : ℝ := sorry

-- Theorem statement
theorem newspaper_probability :
  probability_of_getting_newspaper = 7 / 8 := by sorry

end newspaper_probability_l1616_161681


namespace oil_floats_on_water_l1616_161660

-- Define the density of a substance
def density (substance : Type) : ℝ := sorry

-- Define what it means for a substance to float on another
def floats_on (a b : Type) : Prop := 
  density a < density b

-- Define oil and water as types
def oil : Type := sorry
def water : Type := sorry

-- State the theorem
theorem oil_floats_on_water : 
  (density oil < density water) → floats_on oil water := by sorry

end oil_floats_on_water_l1616_161660
