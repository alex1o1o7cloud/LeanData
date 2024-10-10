import Mathlib

namespace unique_number_with_remainders_l2701_270123

theorem unique_number_with_remainders : ∃! m : ℤ,
  (m % 13 = 12) ∧
  (m % 12 = 11) ∧
  (m % 11 = 10) ∧
  (m % 10 = 9) ∧
  (m % 9 = 8) ∧
  (m % 8 = 7) ∧
  (m % 7 = 6) ∧
  (m % 6 = 5) ∧
  (m % 5 = 4) ∧
  (m % 4 = 3) ∧
  (m % 3 = 2) ∧
  m = 360359 :=
by sorry

end unique_number_with_remainders_l2701_270123


namespace button_probability_theorem_l2701_270110

/-- Represents a jar containing buttons of different colors -/
structure Jar where
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Represents the state of both jars after the transfer -/
structure JarState where
  jarA : Jar
  jarB : Jar

def initial_jarA : Jar := { green := 6, red := 3, blue := 9 }

def button_transfer (x : ℕ) : JarState :=
  { jarA := { green := initial_jarA.green - x, red := initial_jarA.red, blue := initial_jarA.blue - 2*x },
    jarB := { green := x, red := 0, blue := 2*x } }

def total_buttons (jar : Jar) : ℕ := jar.green + jar.red + jar.blue

theorem button_probability_theorem (x : ℕ) (h1 : x > 0) 
  (h2 : total_buttons (button_transfer x).jarA = (total_buttons initial_jarA) / 2) :
  (((button_transfer x).jarA.blue : ℚ) / (total_buttons (button_transfer x).jarA)) * 
  (((button_transfer x).jarB.green : ℚ) / (total_buttons (button_transfer x).jarB)) = 1/9 := by
  sorry

end button_probability_theorem_l2701_270110


namespace worker_assessment_correct_l2701_270140

/-- Worker's skill assessment model -/
structure WorkerAssessment where
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- Probability of ending assessment with 10 products -/
def prob_end_10 (w : WorkerAssessment) : ℝ :=
  w.p^9 * (10 - 9 * w.p)

/-- Expected value of total products produced and debugged -/
def expected_total (w : WorkerAssessment) : ℝ :=
  20 - 10 * w.p - 10 * w.p^9 + 10 * w.p^10

/-- Main theorem: Correctness of worker assessment model -/
theorem worker_assessment_correct (w : WorkerAssessment) :
  (prob_end_10 w = w.p^9 * (10 - 9 * w.p)) ∧
  (expected_total w = 20 - 10 * w.p - 10 * w.p^9 + 10 * w.p^10) := by
  sorry

end worker_assessment_correct_l2701_270140


namespace min_value_of_f_f_is_even_f_monotone_increasing_l2701_270194

noncomputable section

-- Define the operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Define the function f
def f (x : ℝ) : ℝ := 1 + Real.exp x + 1 / Real.exp x

-- Theorem statements
theorem min_value_of_f : ∀ x : ℝ, f x ≥ 3 ∧ ∃ x₀ : ℝ, f x₀ = 3 := by sorry

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

theorem f_monotone_increasing : 
  ∀ x y : ℝ, x ≥ 0 → y ≥ x → f y ≥ f x := by sorry

end

end min_value_of_f_f_is_even_f_monotone_increasing_l2701_270194


namespace candy_distribution_l2701_270180

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 42 → num_bags = 2 → total_candy = num_bags * candy_per_bag → candy_per_bag = 21 := by
  sorry

end candy_distribution_l2701_270180


namespace picture_frame_perimeter_l2701_270193

theorem picture_frame_perimeter (width height : ℕ) (h1 : width = 6) (h2 : height = 9) :
  2 * width + 2 * height = 30 :=
by sorry

end picture_frame_perimeter_l2701_270193


namespace gym_time_calculation_l2701_270177

/-- Calculates the total time spent at the gym per week -/
def gym_time_per_week (visits_per_week : ℕ) (weightlifting_time : ℝ) (warmup_cardio_ratio : ℝ) : ℝ :=
  visits_per_week * (weightlifting_time + warmup_cardio_ratio * weightlifting_time)

/-- Theorem: Given the specified gym routine, the total time spent at the gym per week is 4 hours -/
theorem gym_time_calculation :
  let visits_per_week : ℕ := 3
  let weightlifting_time : ℝ := 1
  let warmup_cardio_ratio : ℝ := 1/3
  gym_time_per_week visits_per_week weightlifting_time warmup_cardio_ratio = 4 := by
  sorry

end gym_time_calculation_l2701_270177


namespace classic_rock_collections_l2701_270147

/-- The number of albums in either Andrew's or Bob's collection, but not both -/
def albums_not_shared (andrew_total : ℕ) (bob_not_andrew : ℕ) (shared : ℕ) : ℕ :=
  (andrew_total - shared) + bob_not_andrew

theorem classic_rock_collections :
  let andrew_total := 20
  let bob_not_andrew := 8
  let shared := 11
  albums_not_shared andrew_total bob_not_andrew shared = 17 := by sorry

end classic_rock_collections_l2701_270147


namespace certain_number_value_l2701_270181

/-- Given that the average of 100, 200, 300, and x is 250,
    and the average of 300, 150, 100, x, and y is 200,
    prove that y = 50 -/
theorem certain_number_value (x : ℝ) (y : ℝ) 
    (h1 : (100 + 200 + 300 + x) / 4 = 250)
    (h2 : (300 + 150 + 100 + x + y) / 5 = 200) : 
  y = 50 := by sorry

end certain_number_value_l2701_270181


namespace complex_number_in_first_quadrant_l2701_270107

/-- The complex number z defined as 2/(1-i) - 2i^3 is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant :
  let z : ℂ := 2 / (1 - Complex.I) - 2 * Complex.I^3
  0 < z.re ∧ 0 < z.im :=
by sorry

end complex_number_in_first_quadrant_l2701_270107


namespace project_duration_calculation_l2701_270168

/-- The number of weeks a project lasts based on breakfast expenses -/
def project_duration (people : ℕ) (days_per_week : ℕ) (meal_cost : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent / (people * days_per_week * meal_cost)

theorem project_duration_calculation :
  let people : ℕ := 4
  let days_per_week : ℕ := 5
  let meal_cost : ℚ := 4
  let total_spent : ℚ := 1280
  project_duration people days_per_week meal_cost total_spent = 16 := by
  sorry

end project_duration_calculation_l2701_270168


namespace dolphin_ratio_l2701_270178

theorem dolphin_ratio (initial_dolphins final_dolphins : ℕ) 
  (h1 : initial_dolphins = 65)
  (h2 : final_dolphins = 260) :
  (final_dolphins - initial_dolphins) / initial_dolphins = 3 := by
  sorry

end dolphin_ratio_l2701_270178


namespace set_operations_l2701_270106

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ (A ∪ B)) = {3}) ∧
  ((U \ A) ∩ (U \ B) = {3}) := by sorry

end set_operations_l2701_270106


namespace diophantine_equation_solution_l2701_270188

theorem diophantine_equation_solution (x y z : ℤ) :
  5 * x^3 + 11 * y^3 + 13 * z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end diophantine_equation_solution_l2701_270188


namespace average_growth_rate_equation_l2701_270145

/-- Represents the average monthly growth rate as a real number between 0 and 1 -/
def average_growth_rate : ℝ := sorry

/-- The initial output value in January in billions of yuan -/
def initial_output : ℝ := 50

/-- The final output value in March in billions of yuan -/
def final_output : ℝ := 60

/-- The number of months between January and March -/
def months : ℕ := 2

theorem average_growth_rate_equation :
  initial_output * (1 + average_growth_rate) ^ months = final_output :=
sorry

end average_growth_rate_equation_l2701_270145


namespace games_in_division_is_sixty_l2701_270143

/-- Represents a baseball league with specified conditions -/
structure BaseballLeague where
  n : ℕ  -- Number of games against each team in the same division
  m : ℕ  -- Number of games against each team in the other division
  h1 : n > 2 * m
  h2 : m > 5
  h3 : 4 * n + 5 * m = 100

/-- The number of games a team plays within its own division -/
def gamesInDivision (league : BaseballLeague) : ℕ := 4 * league.n

theorem games_in_division_is_sixty (league : BaseballLeague) :
  gamesInDivision league = 60 := by
  sorry

#check games_in_division_is_sixty

end games_in_division_is_sixty_l2701_270143


namespace farm_size_l2701_270158

/-- Represents a farm with sunflowers and flax -/
structure Farm where
  flax : ℕ
  sunflowers : ℕ

/-- The total size of the farm in acres -/
def Farm.total_size (f : Farm) : ℕ := f.flax + f.sunflowers

/-- Theorem: Given the conditions, the farm's total size is 240 acres -/
theorem farm_size (f : Farm) 
  (h1 : f.sunflowers = f.flax + 80)  -- 80 more acres of sunflowers than flax
  (h2 : f.flax = 80)                 -- 80 acres of flax
  : f.total_size = 240 := by
  sorry

end farm_size_l2701_270158


namespace prob_at_least_one_girl_l2701_270142

/-- The probability of selecting at least one girl when randomly choosing 2 people from a group of 3 boys and 2 girls is 7/10 -/
theorem prob_at_least_one_girl (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 2) :
  let total := boys + girls
  let prob_at_least_one_girl := 1 - (Nat.choose boys 2 : ℚ) / (Nat.choose total 2 : ℚ)
  prob_at_least_one_girl = 7/10 := by
sorry

end prob_at_least_one_girl_l2701_270142


namespace sqrt_five_minus_two_power_2023_times_sqrt_five_plus_two_power_2023_equals_one_l2701_270151

theorem sqrt_five_minus_two_power_2023_times_sqrt_five_plus_two_power_2023_equals_one :
  (Real.sqrt 5 - 2) ^ 2023 * (Real.sqrt 5 + 2) ^ 2023 = 1 := by
  sorry

end sqrt_five_minus_two_power_2023_times_sqrt_five_plus_two_power_2023_equals_one_l2701_270151


namespace paper_number_sum_paper_number_sum_proof_l2701_270171

/-- Given n pieces of paper, each containing 3 different positive integers no greater than n,
    and any two pieces sharing exactly one common number, prove that the sum of all numbers
    written on these pieces of paper is equal to 3 * n(n+1)/2. -/
theorem paper_number_sum (n : ℕ) : ℕ :=
  let paper_count := n
  let max_number := n
  let numbers_per_paper := 3
  let shared_number_count := 1
  3 * (n * (n + 1) / 2)

-- The proof is omitted as per instructions
theorem paper_number_sum_proof (n : ℕ) :
  paper_number_sum n = 3 * (n * (n + 1) / 2) := by sorry

end paper_number_sum_paper_number_sum_proof_l2701_270171


namespace specific_polyhedron_volume_l2701_270119

/-- Represents a polygon in the figure -/
inductive Polygon
| IsoscelesRightTriangle
| Square
| EquilateralTriangle

/-- Represents the figure that can be folded into a polyhedron -/
structure Figure where
  polygons : List Polygon
  can_fold_to_polyhedron : Bool

/-- Calculates the volume of the polyhedron formed by folding the figure -/
def polyhedron_volume (fig : Figure) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific polyhedron -/
theorem specific_polyhedron_volume :
  ∃ (fig : Figure),
    fig.polygons = [Polygon.IsoscelesRightTriangle, Polygon.IsoscelesRightTriangle, Polygon.IsoscelesRightTriangle,
                    Polygon.Square, Polygon.Square, Polygon.Square,
                    Polygon.EquilateralTriangle] ∧
    fig.can_fold_to_polyhedron = true ∧
    polyhedron_volume fig = 8 - (2 * Real.sqrt 2) / 3 :=
  sorry

end specific_polyhedron_volume_l2701_270119


namespace movie_children_count_l2701_270165

/-- Calculates the maximum number of children that can be taken to the movies given the ticket costs and total budget. -/
def max_children (adult_ticket_cost child_ticket_cost total_budget : ℕ) : ℕ :=
  ((total_budget - adult_ticket_cost) / child_ticket_cost)

/-- Theorem stating that given the specific costs and budget, the maximum number of children is 9. -/
theorem movie_children_count :
  let adult_ticket_cost := 8
  let child_ticket_cost := 3
  let total_budget := 35
  max_children adult_ticket_cost child_ticket_cost total_budget = 9 := by
  sorry

end movie_children_count_l2701_270165


namespace black_socks_bought_is_12_l2701_270108

/-- The number of pairs of black socks Dmitry bought -/
def black_socks_bought : ℕ := sorry

/-- The initial number of blue sock pairs -/
def initial_blue : ℕ := 14

/-- The initial number of black sock pairs -/
def initial_black : ℕ := 24

/-- The initial number of white sock pairs -/
def initial_white : ℕ := 10

/-- The total number of sock pairs after buying more black socks -/
def total_after : ℕ := initial_blue + initial_white + initial_black + black_socks_bought

/-- The number of black sock pairs after buying more -/
def black_after : ℕ := initial_black + black_socks_bought

theorem black_socks_bought_is_12 : 
  black_socks_bought = 12 ∧ 
  black_after = (3 : ℚ) / 5 * total_after := by sorry

end black_socks_bought_is_12_l2701_270108


namespace ceiling_negative_fraction_squared_l2701_270170

theorem ceiling_negative_fraction_squared : ⌈(-7/4)^2⌉ = 4 := by sorry

end ceiling_negative_fraction_squared_l2701_270170


namespace smallest_possible_a_l2701_270149

theorem smallest_possible_a (a b c d x : ℤ) 
  (h1 : (a - 2*b) * x = 1)
  (h2 : (b - 3*c) * x = 1)
  (h3 : (c - 4*d) * x = 1)
  (h4 : x + 100 = d)
  (h5 : x > 0) :
  a ≥ 2433 ∧ ∃ (a₀ b₀ c₀ d₀ x₀ : ℤ), 
    a₀ = 2433 ∧
    (a₀ - 2*b₀) * x₀ = 1 ∧
    (b₀ - 3*c₀) * x₀ = 1 ∧
    (c₀ - 4*d₀) * x₀ = 1 ∧
    x₀ + 100 = d₀ ∧
    x₀ > 0 :=
by sorry

end smallest_possible_a_l2701_270149


namespace dinitrogen_trioxide_weight_calculation_l2701_270113

/-- The atomic weight of Nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of Oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of Dinitrogen trioxide (N2O3) in g/mol -/
def dinitrogen_trioxide_weight : ℝ := 
  nitrogen_weight * nitrogen_count + oxygen_weight * oxygen_count

theorem dinitrogen_trioxide_weight_calculation : 
  dinitrogen_trioxide_weight = 76.02 := by
  sorry

end dinitrogen_trioxide_weight_calculation_l2701_270113


namespace shortest_distance_on_specific_cone_l2701_270134

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ
  angle : ℝ

/-- Calculates the shortest distance between two points on a cone's surface -/
def shortestDistanceOnCone (cone : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem shortest_distance_on_specific_cone :
  let cone : Cone := { baseRadius := 500, height := 300 * Real.sqrt 3 }
  let p1 : ConePoint := { distanceFromVertex := 150, angle := 0 }
  let p2 : ConePoint := { distanceFromVertex := 450 * Real.sqrt 2, angle := 5 * Real.pi / Real.sqrt 52 }
  shortestDistanceOnCone cone p1 p2 = 450 * Real.sqrt 2 - 150 := by
  sorry

end shortest_distance_on_specific_cone_l2701_270134


namespace race_length_is_1000_l2701_270121

/-- The length of a race, given the distance covered by one runner and their remaining distance when another runner finishes. -/
def race_length (distance_covered : ℕ) (distance_remaining : ℕ) : ℕ :=
  distance_covered + distance_remaining

/-- Theorem stating that the race length is 1000 meters under the given conditions. -/
theorem race_length_is_1000 :
  let ava_covered : ℕ := 833
  let ava_remaining : ℕ := 167
  race_length ava_covered ava_remaining = 1000 := by
  sorry

end race_length_is_1000_l2701_270121


namespace remainder_theorem_l2701_270138

theorem remainder_theorem (n : ℤ) (h : ∃ (a : ℤ), n = 100 * a - 1) : 
  (n^2 + 2*n + 3) % 100 = 2 := by
sorry

end remainder_theorem_l2701_270138


namespace line_perp_plane_implies_planes_perp_l2701_270141

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (α β : Plane) (l : Line)
  (h1 : subset l α)
  (h2 : perp l β) :
  perp_planes α β :=
sorry

end line_perp_plane_implies_planes_perp_l2701_270141


namespace upgraded_fraction_is_one_fourth_l2701_270144

/-- Represents a satellite with modular units and sensors -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on the satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

/-- Theorem: The fraction of upgraded sensors on the satellite is 1/4 -/
theorem upgraded_fraction_is_one_fourth (s : Satellite) 
  (h1 : s.units = 24)
  (h2 : s.non_upgraded_per_unit = s.total_upgraded / 8) :
  upgraded_fraction s = 1/4 := by
  sorry

end upgraded_fraction_is_one_fourth_l2701_270144


namespace square_sum_zero_implies_both_zero_l2701_270160

theorem square_sum_zero_implies_both_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l2701_270160


namespace certain_number_l2701_270192

theorem certain_number : ∃ x : ℕ, x - 2 - 2 = 5 ∧ x = 9 := by
  sorry

end certain_number_l2701_270192


namespace first_loan_amount_l2701_270164

/-- Represents a student loan -/
structure Loan where
  amount : ℝ
  rate : ℝ

/-- Calculates the interest paid on a loan -/
def interest_paid (loan : Loan) : ℝ :=
  loan.amount * loan.rate

theorem first_loan_amount
  (loan1 loan2 : Loan)
  (h1 : loan2.rate = 0.09)
  (h2 : loan1.amount = loan2.amount + 1500)
  (h3 : interest_paid loan1 + interest_paid loan2 = 617)
  (h4 : loan2.amount = 4700) :
  loan1.amount = 6200 := by
  sorry

end first_loan_amount_l2701_270164


namespace log_inequality_equivalence_l2701_270137

/-- A function that is even and monotonically increasing on [0,+∞) -/
def EvenMonoIncreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)

theorem log_inequality_equivalence (f : ℝ → ℝ) (h : EvenMonoIncreasing f) :
  (∀ x : ℝ, f 1 < f (Real.log x) ↔ (x > 10 ∨ 0 < x ∧ x < (1/10))) :=
sorry

end log_inequality_equivalence_l2701_270137


namespace prob_three_red_modified_deck_l2701_270129

/-- A deck of cards with red and black suits -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (h_red_cards : red_cards ≤ total_cards)

/-- The probability of drawing three red cards in a row -/
def prob_three_red (d : Deck) : ℚ :=
  (d.red_cards * (d.red_cards - 1) * (d.red_cards - 2)) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- The deck described in the problem -/
def modified_deck : Deck :=
  { total_cards := 60,
    red_cards := 36,
    h_red_cards := by norm_num }

theorem prob_three_red_modified_deck :
  prob_three_red modified_deck = 140 / 673 := by
  sorry

end prob_three_red_modified_deck_l2701_270129


namespace max_digit_sum_l2701_270133

def is_valid_number (n : ℕ) : Prop :=
  2000 ≤ n ∧ n ≤ 2999 ∧ n % 13 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem max_digit_sum :
  ∃ (n : ℕ), is_valid_number n ∧
  ∀ (m : ℕ), is_valid_number m → digit_sum m ≤ digit_sum n ∧
  digit_sum n = 26 :=
sorry

end max_digit_sum_l2701_270133


namespace intersection_M_N_l2701_270195

-- Define the set M
def M : Set ℝ := {x | Real.log (1 - x) < 0}

-- Define the set N
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end intersection_M_N_l2701_270195


namespace integral_of_f_l2701_270118

theorem integral_of_f (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 + 2 * ∫ x in (0:ℝ)..1, f x) → 
  ∫ x in (0:ℝ)..1, f x = -1/3 := by
  sorry

end integral_of_f_l2701_270118


namespace diamond_roof_diagonal_l2701_270126

/-- Given a diamond-shaped roof with area A and diagonals d1 and d2, 
    prove that if A = 80 and d1 = 16, then d2 = 10 -/
theorem diamond_roof_diagonal (A d1 d2 : ℝ) 
  (h_area : A = 80) 
  (h_diagonal : d1 = 16) 
  (h_shape : A = (d1 * d2) / 2) : 
  d2 = 10 := by
  sorry

end diamond_roof_diagonal_l2701_270126


namespace geometric_sequence_problem_l2701_270159

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence condition
  a 3 = 4 →
  a 7 = 12 →
  a 11 = 36 := by
sorry

end geometric_sequence_problem_l2701_270159


namespace motorcycle_toll_correct_l2701_270187

/-- Represents the weekly commute scenario for Geordie --/
structure CommuteScenario where
  workDaysPerWeek : ℕ
  carToll : ℚ
  mpg : ℚ
  commuteDistance : ℚ
  gasPrice : ℚ
  carTripsPerWeek : ℕ
  motorcycleTripsPerWeek : ℕ
  totalWeeklyCost : ℚ

/-- Calculates the motorcycle toll given a commute scenario --/
def calculateMotorcycleToll (scenario : CommuteScenario) : ℚ :=
  sorry

/-- Theorem stating that the calculated motorcycle toll is correct --/
theorem motorcycle_toll_correct (scenario : CommuteScenario) :
  scenario.workDaysPerWeek = 5 ∧
  scenario.carToll = 25/2 ∧
  scenario.mpg = 35 ∧
  scenario.commuteDistance = 14 ∧
  scenario.gasPrice = 15/4 ∧
  scenario.carTripsPerWeek = 3 ∧
  scenario.motorcycleTripsPerWeek = 2 ∧
  scenario.totalWeeklyCost = 118 →
  calculateMotorcycleToll scenario = 131/4 :=
sorry

end motorcycle_toll_correct_l2701_270187


namespace pie_chart_most_suitable_l2701_270124

-- Define the available graph types
inductive GraphType
| PieChart
| BarGraph
| LineGraph

-- Define the expenditure categories
inductive ExpenditureCategory
| Education
| Clothing
| Food
| Other

-- Define a function to determine if a graph type is suitable for representing percentages
def isSuitableForPercentages (g : GraphType) : Prop :=
  match g with
  | GraphType.PieChart => True
  | _ => False

-- Define a function to check if a graph type can effectively show parts of a whole
def showsPartsOfWhole (g : GraphType) : Prop :=
  match g with
  | GraphType.PieChart => True
  | _ => False

-- Theorem stating that a pie chart is the most suitable graph type
theorem pie_chart_most_suitable (categories : List ExpenditureCategory) 
  (h1 : categories.length > 1) 
  (h2 : categories.length ≤ 4) : 
  ∃ (g : GraphType), isSuitableForPercentages g ∧ showsPartsOfWhole g :=
by
  sorry

end pie_chart_most_suitable_l2701_270124


namespace cost_of_600_pages_l2701_270103

-- Define the cost per 5 pages in cents
def cost_per_5_pages : ℕ := 10

-- Define the number of pages to be copied
def pages_to_copy : ℕ := 600

-- Theorem to prove the cost of copying 600 pages
theorem cost_of_600_pages : 
  (pages_to_copy / 5) * cost_per_5_pages = 1200 :=
by
  sorry

#check cost_of_600_pages

end cost_of_600_pages_l2701_270103


namespace total_length_is_24_l2701_270161

/-- Represents a geometric figure with perpendicular adjacent sides -/
structure GeometricFigure where
  bottom : ℝ
  right : ℝ
  top_left : ℝ
  top_right : ℝ
  middle_horizontal : ℝ
  middle_vertical : ℝ
  left : ℝ

/-- Calculates the total length of visible segments in the transformed figure -/
def total_length_after_transform (fig : GeometricFigure) : ℝ :=
  fig.bottom + (fig.right - 2) + (fig.top_left - 3) + fig.left

/-- Theorem stating that the total length of segments in Figure 2 is 24 units -/
theorem total_length_is_24 (fig : GeometricFigure) 
  (h1 : fig.bottom = 5)
  (h2 : fig.right = 10)
  (h3 : fig.top_left = 4)
  (h4 : fig.top_right = 4)
  (h5 : fig.middle_horizontal = 3)
  (h6 : fig.middle_vertical = 3)
  (h7 : fig.left = 10) :
  total_length_after_transform fig = 24 := by
  sorry


end total_length_is_24_l2701_270161


namespace midpoint_coordinate_product_l2701_270120

/-- Given that N(4,7) is the midpoint of line segment CD and C(5,3) is one endpoint,
    prove that the product of the coordinates of point D is 33. -/
theorem midpoint_coordinate_product (D : ℝ × ℝ) : 
  let N : ℝ × ℝ := (4, 7)
  let C : ℝ × ℝ := (5, 3)
  (N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2) →
  D.1 * D.2 = 33 := by
sorry

end midpoint_coordinate_product_l2701_270120


namespace count_odd_numbers_less_than_400_l2701_270112

/-- The set of digits that can be used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4}

/-- A function that checks if a number is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- A function that checks if a three-digit number is less than 400 -/
def isLessThan400 (n : Nat) : Bool := n < 400 ∧ n ≥ 100

/-- The set of valid hundreds digits (1, 2, 3) -/
def validHundreds : Finset Nat := {1, 2, 3}

/-- The set of valid units digits for odd numbers (1, 3) -/
def validUnits : Finset Nat := {1, 3}

/-- The main theorem -/
theorem count_odd_numbers_less_than_400 :
  (validHundreds.card * digits.card * validUnits.card) = 24 := by
  sorry

#eval validHundreds.card * digits.card * validUnits.card

end count_odd_numbers_less_than_400_l2701_270112


namespace remainder_after_adding_2025_l2701_270104

theorem remainder_after_adding_2025 (n : ℤ) : n % 5 = 3 → (n + 2025) % 5 = 3 := by
  sorry

end remainder_after_adding_2025_l2701_270104


namespace f_difference_l2701_270111

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 5 * x - 4

theorem f_difference (x h : ℝ) : 
  f (x + h) - f x = h * (6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h - 5) := by
  sorry

end f_difference_l2701_270111


namespace barb_dress_fraction_l2701_270135

theorem barb_dress_fraction (original_price savings paid : ℝ) (f : ℝ) :
  original_price = 180 →
  savings = 80 →
  paid = original_price - savings →
  paid = f * original_price - 10 →
  f = 11 / 18 := by
  sorry

end barb_dress_fraction_l2701_270135


namespace five_pow_minus_two_pow_div_by_three_l2701_270152

theorem five_pow_minus_two_pow_div_by_three (n : ℕ) :
  ∃ k : ℤ, 5^n - 2^n = 3 * k :=
sorry

end five_pow_minus_two_pow_div_by_three_l2701_270152


namespace triangles_with_fixed_vertex_l2701_270184

/-- The number of triangles formed with a fixed vertex from 8 points on a circle -/
theorem triangles_with_fixed_vertex (n : ℕ) (h : n = 8) : 
  (Nat.choose (n - 1) 2) = 21 := by
  sorry

#check triangles_with_fixed_vertex

end triangles_with_fixed_vertex_l2701_270184


namespace smallest_solution_is_smaller_root_l2701_270190

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 + 9 * x - 92 = 0

-- Define the original equation
def original_eq (x : ℝ) : Prop := 3 * x^2 + 24 * x - 92 = x * (x + 15)

-- Theorem statement
theorem smallest_solution_is_smaller_root :
  ∃ (x : ℝ), quadratic_eq x ∧ 
  (∀ (y : ℝ), quadratic_eq y → x ≤ y) ∧
  (∀ (z : ℝ), original_eq z → x ≤ z) := by
  sorry

end smallest_solution_is_smaller_root_l2701_270190


namespace complex_power_220_36_l2701_270176

theorem complex_power_220_36 : (Complex.exp (220 * π / 180 * I))^36 = 1 := by
  sorry

end complex_power_220_36_l2701_270176


namespace base_number_proof_l2701_270189

theorem base_number_proof (base : ℝ) : base ^ 7 = 3 ^ 14 → base = 9 := by
  sorry

end base_number_proof_l2701_270189


namespace sin_2x_value_l2701_270101

theorem sin_2x_value (x : ℝ) (h : Real.tan (x + π/4) = 2) : Real.sin (2*x) = 3/5 := by
  sorry

end sin_2x_value_l2701_270101


namespace checkerboard_ratio_l2701_270169

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The sum of squares from 1 to n -/
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of rectangles on an n x n checkerboard -/
def num_rectangles (n : ℕ) : ℕ := (choose_2 (n + 1)) ^ 2

/-- The number of squares on an n x n checkerboard -/
def num_squares (n : ℕ) : ℕ := sum_squares n

theorem checkerboard_ratio :
  (num_squares 9 : ℚ) / (num_rectangles 9 : ℚ) = 19 / 135 := by sorry

end checkerboard_ratio_l2701_270169


namespace exists_N_average_ten_l2701_270114

theorem exists_N_average_ten :
  ∃ N : ℝ, 9 < N ∧ N < 17 ∧ (6 + 10 + N) / 3 = 10 := by
sorry

end exists_N_average_ten_l2701_270114


namespace v_4_value_l2701_270191

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecursiveSequence (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 2 * v (n + 1) + v n

theorem v_4_value (v : ℕ → ℝ) (h_rec : RecursiveSequence v) 
    (h_v2 : v 2 = 7) (h_v5 : v 5 = 53) : v 4 = 22.6 := by
  sorry

end v_4_value_l2701_270191


namespace certain_number_problem_l2701_270167

theorem certain_number_problem : ∃ x : ℝ, x * (5^4) = 70000 ∧ x = 112 := by
  sorry

end certain_number_problem_l2701_270167


namespace x_20_digits_l2701_270131

theorem x_20_digits (x : ℝ) (h1 : x > 0) (h2 : 10^7 ≤ x^4) (h3 : x^5 < 10^9) :
  10^35 ≤ x^20 ∧ x^20 < 10^36 := by
  sorry

end x_20_digits_l2701_270131


namespace longest_side_of_triangle_l2701_270174

theorem longest_side_of_triangle (y : ℚ) : 
  6 + (y + 3) + (3 * y - 2) = 40 →
  max 6 (max (y + 3) (3 * y - 2)) = 91 / 4 := by
sorry

end longest_side_of_triangle_l2701_270174


namespace polynomial_expansion_theorem_l2701_270183

theorem polynomial_expansion_theorem (a b : ℝ) : 
  a > 0 → b > 0 → a * b = 1/2 → 
  (28 : ℝ) * a^6 * b^2 = (56 : ℝ) * a^5 * b^3 → 
  a = 1 := by
sorry

end polynomial_expansion_theorem_l2701_270183


namespace at_most_one_square_l2701_270199

theorem at_most_one_square (a : ℕ → ℤ) (h : ∀ n : ℕ, a (n + 1) = (a n)^3 + 1999) :
  ∃! n : ℕ, ∃ k : ℤ, a n = k^2 :=
sorry

end at_most_one_square_l2701_270199


namespace total_boys_in_class_l2701_270146

/-- Given a circular arrangement of students, if the 10th and 40th positions
    are opposite each other and only every other student is counted,
    then the total number of boys in the class is 30. -/
theorem total_boys_in_class (n : ℕ) 
  (circular_arrangement : n > 0)
  (opposite_positions : 40 - 10 = n / 2)
  (count_every_other : n % 2 = 0) : 
  n / 2 = 30 := by
  sorry

end total_boys_in_class_l2701_270146


namespace max_value_of_t_l2701_270154

theorem max_value_of_t (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  min x (y / (x^2 + y^2)) ≤ 1 / Real.sqrt 2 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ min x (y / (x^2 + y^2)) = 1 / Real.sqrt 2 :=
by sorry

end max_value_of_t_l2701_270154


namespace alice_age_l2701_270132

/-- Prove that Alice's age is 20 years old given the conditions. -/
theorem alice_age : 
  ∀ (alice_pens : ℕ) (clara_pens : ℕ) (alice_age : ℕ) (clara_age : ℕ),
  alice_pens = 60 →
  clara_pens = (2 * alice_pens) / 5 →
  alice_pens - clara_pens = clara_age - alice_age →
  clara_age > alice_age →
  clara_age + 5 = 61 →
  alice_age = 20 := by
sorry

end alice_age_l2701_270132


namespace correct_operation_l2701_270157

theorem correct_operation (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = -x * y^2 := by
  sorry

end correct_operation_l2701_270157


namespace power_function_through_point_and_value_l2701_270139

/-- A power function that passes through the point (2,8) -/
def f : ℝ → ℝ := fun x ↦ x^3

theorem power_function_through_point_and_value :
  f 2 = 8 ∧ f 3 = 27 := by
  sorry

end power_function_through_point_and_value_l2701_270139


namespace smallest_product_l2701_270130

def digits : List Nat := [4, 5, 6, 7]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat, is_valid_arrangement a b c d →
    product a b c d ≥ 2622 :=
by sorry

end smallest_product_l2701_270130


namespace chocolate_kisses_bags_l2701_270196

theorem chocolate_kisses_bags (total_candy : ℕ) (total_bags : ℕ) (heart_bags : ℕ) (non_chocolate_pieces : ℕ) :
  total_candy = 63 →
  total_bags = 9 →
  heart_bags = 2 →
  non_chocolate_pieces = 28 →
  total_candy % total_bags = 0 →
  ∃ (kisses_bags : ℕ),
    kisses_bags = total_bags - heart_bags - (non_chocolate_pieces / (total_candy / total_bags)) ∧
    kisses_bags = 3 :=
by sorry

end chocolate_kisses_bags_l2701_270196


namespace correct_number_of_values_l2701_270172

theorem correct_number_of_values 
  (original_mean : ℝ) 
  (incorrect_value : ℝ) 
  (correct_value : ℝ) 
  (correct_mean : ℝ) 
  (h1 : original_mean = 190) 
  (h2 : incorrect_value = 130) 
  (h3 : correct_value = 165) 
  (h4 : correct_mean = 191.4) : 
  ∃ n : ℕ, n > 0 ∧ 
    n * original_mean + (correct_value - incorrect_value) = n * correct_mean ∧ 
    n = 25 := by
  sorry

end correct_number_of_values_l2701_270172


namespace x_gt_one_sufficient_not_necessary_l2701_270102

theorem x_gt_one_sufficient_not_necessary :
  (∃ x : ℝ, x > 1 → (1 / x) < 1) ∧
  (∃ x : ℝ, (1 / x) < 1 ∧ ¬(x > 1)) :=
by sorry

end x_gt_one_sufficient_not_necessary_l2701_270102


namespace ln_power_equality_l2701_270136

theorem ln_power_equality (x : ℝ) :
  (Real.log (x^4))^2 = (Real.log x)^6 ↔ x = 1 ∨ x = Real.exp 2 ∨ x = Real.exp (-2) :=
sorry

end ln_power_equality_l2701_270136


namespace point_coordinate_sum_l2701_270173

/-- Given points P and Q, where P is at the origin and Q is on the line y = 6,
    if the slope of PQ is 3/4, then the sum of Q's coordinates is 14. -/
theorem point_coordinate_sum (x : ℝ) : 
  let P : ℝ × ℝ := (0, 0)
  let Q : ℝ × ℝ := (x, 6)
  let slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  slope = 3/4 → Q.1 + Q.2 = 14 := by
  sorry

end point_coordinate_sum_l2701_270173


namespace quadratic_inequality_solution_l2701_270150

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 5*x - 14

-- Define the solution set
def solution_set : Set ℝ := {x | -2 < x ∧ x < 7}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set := by sorry

end quadratic_inequality_solution_l2701_270150


namespace problem_solution_l2701_270122

theorem problem_solution : (((3⁻¹ : ℚ) - 2 + 6^2 + 1)⁻¹ * 6 : ℚ) = 9 / 53 := by
  sorry

end problem_solution_l2701_270122


namespace isosceles_trapezoid_angles_l2701_270153

theorem isosceles_trapezoid_angles (a d : ℝ) : 
  -- The trapezoid is isosceles and angles form an arithmetic sequence
  a > 0 ∧ d > 0 ∧ 
  -- The sum of angles in a quadrilateral is 360°
  a + (a + d) + (a + 2*d) + 140 = 360 ∧ 
  -- The largest angle is 140°
  a + 3*d = 140 → 
  -- The smallest angle is 40°
  a = 40 := by sorry

end isosceles_trapezoid_angles_l2701_270153


namespace temperature_difference_product_product_of_possible_P_values_l2701_270166

theorem temperature_difference_product (P : ℝ) : 
  (∃ (A B : ℝ), A = B + P ∧ 
   ∃ (A_t B_t : ℝ), A_t = B + P - 8 ∧ B_t = B + 2 ∧ 
   |A_t - B_t| = 4) →
  (P = 12 ∨ P = 4) :=
by sorry

theorem product_of_possible_P_values : 
  (∀ P : ℝ, (∃ (A B : ℝ), A = B + P ∧ 
   ∃ (A_t B_t : ℝ), A_t = B + P - 8 ∧ B_t = B + 2 ∧ 
   |A_t - B_t| = 4) →
  (P = 12 ∨ P = 4)) →
  12 * 4 = 48 :=
by sorry

end temperature_difference_product_product_of_possible_P_values_l2701_270166


namespace race_distance_proof_l2701_270185

/-- The total distance of a race where:
  * A covers the distance in 20 seconds
  * B covers the distance in 25 seconds
  * A beats B by 14 meters
-/
def race_distance : ℝ := 56

/-- A's time to complete the race in seconds -/
def time_A : ℝ := 20

/-- B's time to complete the race in seconds -/
def time_B : ℝ := 25

/-- The distance by which A beats B in meters -/
def beat_distance : ℝ := 14

theorem race_distance_proof :
  race_distance = (time_B * beat_distance) / (time_B / time_A - 1) :=
by sorry

end race_distance_proof_l2701_270185


namespace smallest_integer_with_remainders_l2701_270109

theorem smallest_integer_with_remainders : ∃ (x : ℕ), 
  x > 0 ∧
  x % 2 = 0 ∧
  x % 4 = 1 ∧
  x % 5 = 2 ∧
  x % 7 = 3 ∧
  (∀ y : ℕ, y > 0 ∧ y % 2 = 0 ∧ y % 4 = 1 ∧ y % 5 = 2 ∧ y % 7 = 3 → x ≤ y) :=
by
  -- Proof goes here
  sorry

end smallest_integer_with_remainders_l2701_270109


namespace fourth_roll_prob_is_five_sixths_l2701_270162

-- Define the types of dice
inductive DieType
| Fair
| BiasedSix
| BiasedOne

-- Define the probability of rolling a six for each die type
def probSix (d : DieType) : ℚ :=
  match d with
  | DieType.Fair => 1/6
  | DieType.BiasedSix => 1/2
  | DieType.BiasedOne => 1/10

-- Define the probability of selecting each die
def probSelectDie (d : DieType) : ℚ := 1/3

-- Define the probability of rolling three sixes in a row for a given die
def probThreeSixes (d : DieType) : ℚ := (probSix d) ^ 3

-- Define the total probability of rolling three sixes
def totalProbThreeSixes : ℚ :=
  (probSelectDie DieType.Fair) * (probThreeSixes DieType.Fair) +
  (probSelectDie DieType.BiasedSix) * (probThreeSixes DieType.BiasedSix) +
  (probSelectDie DieType.BiasedOne) * (probThreeSixes DieType.BiasedOne)

-- Define the updated probability of having used each die type given three sixes were rolled
def updatedProbDie (d : DieType) : ℚ :=
  (probSelectDie d) * (probThreeSixes d) / totalProbThreeSixes

-- The main theorem
theorem fourth_roll_prob_is_five_sixths :
  (updatedProbDie DieType.Fair) * (probSix DieType.Fair) +
  (updatedProbDie DieType.BiasedSix) * (probSix DieType.BiasedSix) +
  (updatedProbDie DieType.BiasedOne) * (probSix DieType.BiasedOne) = 5/6 := by
  sorry

end fourth_roll_prob_is_five_sixths_l2701_270162


namespace stratified_sample_size_l2701_270128

/-- Given a stratified sample of three products with a quantity ratio of 2:3:5,
    prove that if 16 units of the first product are in the sample,
    then the total sample size is 80. -/
theorem stratified_sample_size
  (total_ratio : ℕ)
  (ratio_A : ℕ)
  (ratio_B : ℕ)
  (ratio_C : ℕ)
  (sample_A : ℕ)
  (h1 : total_ratio = ratio_A + ratio_B + ratio_C)
  (h2 : ratio_A = 2)
  (h3 : ratio_B = 3)
  (h4 : ratio_C = 5)
  (h5 : sample_A = 16) :
  (sample_A * total_ratio) / ratio_A = 80 := by
sorry

end stratified_sample_size_l2701_270128


namespace remaining_amount_is_99_l2701_270127

/-- Calculates the remaining amount in US dollars after transactions --/
def remaining_amount (initial_usd : ℝ) (initial_euro : ℝ) (exchange_rate : ℝ) 
  (supermarket_spend : ℝ) (book_cost_euro : ℝ) (lunch_cost : ℝ) : ℝ :=
  initial_usd + initial_euro * exchange_rate - supermarket_spend - book_cost_euro * exchange_rate - lunch_cost

/-- Proves that the remaining amount is 99 US dollars given the initial amounts and transactions --/
theorem remaining_amount_is_99 :
  remaining_amount 78 50 1.2 15 10 12 = 99 := by
  sorry

#eval remaining_amount 78 50 1.2 15 10 12

end remaining_amount_is_99_l2701_270127


namespace inscribed_sphere_radius_ratio_l2701_270175

/-- A hexahedron with equilateral triangle faces congruent to those of a regular octahedron -/
structure SpecialHexahedron where
  -- The faces are equilateral triangles
  faces_equilateral : Bool
  -- The faces are congruent to those of a regular octahedron
  faces_congruent_to_octahedron : Bool

/-- A regular octahedron -/
structure RegularOctahedron where

/-- The radius of the inscribed sphere in a polyhedron -/
def inscribed_sphere_radius (P : Type) : ℝ := sorry

/-- The theorem stating the ratio of inscribed sphere radii -/
theorem inscribed_sphere_radius_ratio 
  (h : SpecialHexahedron) 
  (o : RegularOctahedron) 
  (h_valid : h.faces_equilateral ∧ h.faces_congruent_to_octahedron) :
  inscribed_sphere_radius SpecialHexahedron / inscribed_sphere_radius RegularOctahedron = 2/3 :=
sorry

end inscribed_sphere_radius_ratio_l2701_270175


namespace arithmetic_sequence_second_term_l2701_270155

/-- An arithmetic sequence with first term 3 and sum of second and third terms 12 has second term equal to 5 -/
theorem arithmetic_sequence_second_term (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 3 →                                -- first term is 3
  a 2 + a 3 = 12 →                         -- sum of second and third terms is 12
  a 2 = 5 :=                               -- second term is 5
by sorry

end arithmetic_sequence_second_term_l2701_270155


namespace sum_of_coordinates_of_D_l2701_270156

/-- Given that M(4, 6) is the midpoint of CD and C has coordinates (10, 2),
    prove that the sum of the coordinates of point D is 8. -/
theorem sum_of_coordinates_of_D (C D M : ℝ × ℝ) : 
  C = (10, 2) →
  M = (4, 6) →
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 8 := by
  sorry

end sum_of_coordinates_of_D_l2701_270156


namespace quadratic_inequality_solution_set_l2701_270198

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 + 3*x - 4 > 0 ↔ x > 1 ∨ x < -4 := by
sorry

end quadratic_inequality_solution_set_l2701_270198


namespace circle_equation_l2701_270186

/-- Given a circle with center at (-2, 3) and tangent to the y-axis, 
    its equation is (x+2)^2+(y-3)^2=4 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (-2, 3)
  let tangent_to_y_axis : ℝ → Prop := λ r => r = 2
  tangent_to_y_axis (abs center.1) →
  (x + 2)^2 + (y - 3)^2 = 4 := by
sorry

end circle_equation_l2701_270186


namespace max_profit_is_21600_l2701_270179

/-- Represents the production quantities of products A and B -/
structure Production where
  a : ℝ
  b : ℝ

/-- Calculates the total profit for a given production quantity -/
def totalProfit (p : Production) : ℝ :=
  2100 * p.a + 900 * p.b

/-- Checks if a production quantity satisfies all constraints -/
def isValid (p : Production) : Prop :=
  p.a ≥ 0 ∧ p.b ≥ 0 ∧
  1.5 * p.a + 0.5 * p.b ≤ 150 ∧
  1 * p.a + 0.3 * p.b ≤ 90 ∧
  5 * p.a + 3 * p.b ≤ 600

/-- Theorem stating that the maximum total profit is 21600 yuan -/
theorem max_profit_is_21600 :
  ∃ (p : Production), isValid p ∧
    totalProfit p = 21600 ∧
    ∀ (q : Production), isValid q → totalProfit q ≤ 21600 := by
  sorry

end max_profit_is_21600_l2701_270179


namespace intersection_distance_squared_l2701_270148

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The squared distance between two points in a 2D plane --/
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The squared distance between the intersection points of two specific circles --/
theorem intersection_distance_squared (c1 c2 : Circle) 
  (h1 : c1 = ⟨(3, -2), 5⟩) 
  (h2 : c2 = ⟨(3, 4), 3⟩) : 
  ∃ (p1 p2 : ℝ × ℝ), 
    squaredDistance p1 c1.center = c1.radius^2 ∧ 
    squaredDistance p1 c2.center = c2.radius^2 ∧
    squaredDistance p2 c1.center = c1.radius^2 ∧ 
    squaredDistance p2 c2.center = c2.radius^2 ∧
    squaredDistance p1 p2 = 224/9 := by
  sorry

end intersection_distance_squared_l2701_270148


namespace derivative_sin_cos_l2701_270105

theorem derivative_sin_cos (x : Real) :
  deriv (fun x => 3 * Real.sin x - 4 * Real.cos x) x = 3 * Real.cos x + 4 * Real.sin x := by
  sorry

end derivative_sin_cos_l2701_270105


namespace tom_tim_ratio_l2701_270115

/-- The typing speeds of Tim and Tom, and their relationship -/
structure TypingSpeed where
  tim : ℝ
  tom : ℝ
  total_normal : tim + tom = 15
  total_increased : tim + 1.6 * tom = 18

/-- The ratio of Tom's normal typing speed to Tim's is 1:2 -/
theorem tom_tim_ratio (s : TypingSpeed) : s.tom / s.tim = 1 / 2 := by
  sorry

end tom_tim_ratio_l2701_270115


namespace percentage_calculation_l2701_270182

def total_population : ℕ := 40000
def part_population : ℕ := 36000

theorem percentage_calculation : 
  (part_population : ℚ) / (total_population : ℚ) * 100 = 90 := by
  sorry

end percentage_calculation_l2701_270182


namespace share_purchase_price_l2701_270100

/-- The price at which an investor bought shares, given dividend rate, face value, and return on investment. -/
theorem share_purchase_price 
  (dividend_rate : ℝ) 
  (face_value : ℝ) 
  (roi : ℝ) 
  (h1 : dividend_rate = 0.185) 
  (h2 : face_value = 50) 
  (h3 : roi = 0.25) : 
  ∃ (price : ℝ), price = 37 := by
sorry

end share_purchase_price_l2701_270100


namespace geometric_sequence_problem_l2701_270197

theorem geometric_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, a k > 0) →  -- positive terms
  (∀ k, a (k + 1) / a k = a (k + 2) / a (k + 1)) →  -- geometric sequence
  (a 3 = 4) →
  (a 4 * a 5 * a 6 = 2^12) →
  (S n = 2^10 - 1) →
  (S n = (a 1 * (1 - (a 2 / a 1)^n)) / (1 - (a 2 / a 1))) →  -- sum formula for geometric sequence
  (a 1 = 1 ∧ a 2 / a 1 = 2 ∧ n = 10) :=
by sorry

end geometric_sequence_problem_l2701_270197


namespace find_k_l2701_270125

-- Define the sets A and B
def A (k : ℝ) : Set ℝ := {x | 1 < x ∧ x < k}
def B (k : ℝ) : Set ℝ := {y | ∃ x ∈ A k, y = 2*x - 5}

-- Define the intersection set
def intersection_set : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem find_k (k : ℝ) : A k ∩ B k = intersection_set → k = 3.5 := by
  sorry

end find_k_l2701_270125


namespace quadratic_function_theorem_l2701_270116

def units_digit (n : ℕ) : ℕ := n % 10

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

theorem quadratic_function_theorem (a b c : ℤ) (p : ℕ) :
  is_positive_even p →
  10 ≤ p →
  p ≤ 50 →
  units_digit p > 0 →
  units_digit (p^3) - units_digit (p^2) = 0 →
  (a * p^2 + b * p + c : ℤ) = 0 →
  (a * p^4 + b * p^2 + c : ℤ) = (a * p^6 + b * p^3 + c : ℤ) →
  units_digit (p + 5) = 1 := by
  sorry

end quadratic_function_theorem_l2701_270116


namespace x_plus_y_squared_l2701_270163

theorem x_plus_y_squared (x y : ℝ) (h1 : 2 * x * (x + y) = 54) (h2 : 3 * y * (x + y) = 81) : 
  (x + y)^2 = 135 := by
sorry

end x_plus_y_squared_l2701_270163


namespace problem_statement_l2701_270117

theorem problem_statement (a b : ℝ) 
  (h1 : 0 < (1 : ℝ) / a) 
  (h2 : (1 : ℝ) / a < (1 : ℝ) / b) 
  (h3 : (1 : ℝ) / b < 1) 
  (h4 : Real.log a * Real.log b = 1) : 
  (2 : ℝ) ^ a > (2 : ℝ) ^ b ∧ 
  a * b > Real.exp 2 ∧ 
  Real.exp (a - b) > a / b := by
  sorry

end problem_statement_l2701_270117
