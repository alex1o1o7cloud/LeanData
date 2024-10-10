import Mathlib

namespace fourth_side_length_l3521_352139

/-- A quadrilateral inscribed in a circle with specific properties -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The length of three sides of the quadrilateral -/
  side_length : ℝ
  /-- Assertion that the quadrilateral is a kite with two equal consecutive sides -/
  is_kite : Prop
  /-- Assertion that one diagonal is a diameter of the circle -/
  diagonal_is_diameter : Prop

/-- The theorem stating the length of the fourth side of the quadrilateral -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
  (h1 : q.radius = 150 * Real.sqrt 2)
  (h2 : q.side_length = 150) :
  ∃ (fourth_side : ℝ), fourth_side = 150 := by
  sorry

end fourth_side_length_l3521_352139


namespace overtime_hours_l3521_352132

/-- Queenie's daily wage as a part-time clerk -/
def daily_wage : ℕ := 150

/-- Queenie's overtime pay rate per hour -/
def overtime_rate : ℕ := 5

/-- Number of days Queenie worked -/
def days_worked : ℕ := 5

/-- Total amount Queenie received -/
def total_pay : ℕ := 770

/-- Calculate the number of overtime hours Queenie worked -/
theorem overtime_hours : 
  (total_pay - daily_wage * days_worked) / overtime_rate = 4 := by
  sorry

end overtime_hours_l3521_352132


namespace area_of_R2_l3521_352138

/-- Rectangle R1 -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle R2 -/
structure Rectangle2 where
  diagonal : ℝ

/-- Given conditions -/
def given_conditions : Prop :=
  ∃ (R1 : Rectangle1) (R2 : Rectangle2),
    R1.side = 4 ∧
    R1.area = 32 ∧
    R2.diagonal = 20 ∧
    -- Similarity condition (ratio of sides is the same)
    ∃ (k : ℝ), k > 0 ∧ R2.diagonal = k * (R1.side * (R1.area / R1.side).sqrt)

/-- Theorem: Area of R2 is 160 square inches -/
theorem area_of_R2 : given_conditions → ∃ (R2 : Rectangle2), R2.diagonal = 20 ∧ R2.diagonal^2 / 2 = 160 :=
sorry

end area_of_R2_l3521_352138


namespace council_vote_change_l3521_352128

theorem council_vote_change (total_members : ℕ) 
  (initial_for initial_against : ℚ) 
  (revote_for revote_against : ℚ) : 
  total_members = 500 ∧ 
  initial_for + initial_against = total_members ∧
  initial_against > initial_for ∧
  revote_for + revote_against = total_members ∧
  revote_for > revote_against ∧
  revote_for - revote_against = (3/2) * (initial_against - initial_for) ∧
  revote_for = (11/10) * initial_against →
  revote_for - initial_for = 156.25 := by
sorry

end council_vote_change_l3521_352128


namespace filtration_theorem_l3521_352176

/-- The reduction rate of impurities per filtration -/
def reduction_rate : ℝ := 0.2

/-- The target percentage of impurities relative to the original amount -/
def target_percentage : ℝ := 0.05

/-- The logarithm of 2 -/
def log_2 : ℝ := 0.301

/-- The minimum number of filtrations required -/
def min_filtrations : ℕ := 14

theorem filtration_theorem : 
  ∀ n : ℕ, (1 - reduction_rate) ^ n < target_percentage ↔ n ≥ min_filtrations := by
  sorry

end filtration_theorem_l3521_352176


namespace total_subscription_is_50000_l3521_352171

/-- Represents the subscription amounts and profit distribution for a business --/
structure BusinessSubscription where
  /-- C's subscription amount --/
  c : ℕ
  /-- Total profit --/
  total_profit : ℕ
  /-- A's profit share --/
  a_profit : ℕ

/-- Calculates the total subscription amount given the business subscription details --/
def total_subscription (bs : BusinessSubscription) : ℕ :=
  3 * bs.c + 14000

/-- Theorem stating that the total subscription amount is 50000 given the problem conditions --/
theorem total_subscription_is_50000 (bs : BusinessSubscription)
  (h1 : bs.total_profit = 36000)
  (h2 : bs.a_profit = 15120)
  (h3 : bs.a_profit * (3 * bs.c + 14000) = bs.total_profit * (bs.c + 9000)) :
  total_subscription bs = 50000 := by
  sorry

#check total_subscription_is_50000

end total_subscription_is_50000_l3521_352171


namespace equal_selection_probability_l3521_352198

/-- Given a population size and sample size, prove that the probability of selection
    is equal for simple random sampling, systematic sampling, and stratified sampling. -/
theorem equal_selection_probability
  (N n : ℕ) -- Population size and sample size
  (h_N_pos : N > 0) -- Assumption: Population size is positive
  (h_n_le_N : n ≤ N) -- Assumption: Sample size is not greater than population size
  (P₁ P₂ P₃ : ℚ) -- Probabilities for each sampling method
  (h_P₁ : P₁ = n / N) -- Definition of P₁ for simple random sampling
  (h_P₂ : P₂ = n / N) -- Definition of P₂ for systematic sampling
  (h_P₃ : P₃ = n / N) -- Definition of P₃ for stratified sampling
  : P₁ = P₂ ∧ P₂ = P₃ := by
  sorry

end equal_selection_probability_l3521_352198


namespace definite_integral_sqrt_4_minus_x_squared_minus_2x_l3521_352199

theorem definite_integral_sqrt_4_minus_x_squared_minus_2x : 
  ∫ (x : ℝ) in (0)..(2), (Real.sqrt (4 - x^2) - 2*x) = π - 4 := by sorry

end definite_integral_sqrt_4_minus_x_squared_minus_2x_l3521_352199


namespace tennis_players_l3521_352123

theorem tennis_players (total : ℕ) (squash : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 38)
  (h2 : squash = 21)
  (h3 : neither = 10)
  (h4 : both = 12) :
  total - squash + both - neither = 19 :=
by sorry

end tennis_players_l3521_352123


namespace total_workers_is_214_l3521_352130

/-- Represents a workshop with its salary information -/
structure Workshop where
  avgSalary : ℕ
  techCount : ℕ
  techAvgSalary : ℕ
  otherSalary : ℕ

/-- Calculates the total number of workers in a workshop -/
def totalWorkers (w : Workshop) : ℕ :=
  let otherWorkers := (w.avgSalary * (w.techCount + 1) - w.techAvgSalary * w.techCount) / (w.avgSalary - w.otherSalary)
  w.techCount + otherWorkers

/-- The given workshops -/
def workshopA : Workshop := {
  avgSalary := 8000,
  techCount := 7,
  techAvgSalary := 20000,
  otherSalary := 6000
}

def workshopB : Workshop := {
  avgSalary := 9000,
  techCount := 10,
  techAvgSalary := 25000,
  otherSalary := 5000
}

def workshopC : Workshop := {
  avgSalary := 10000,
  techCount := 15,
  techAvgSalary := 30000,
  otherSalary := 7000
}

/-- The main theorem to prove -/
theorem total_workers_is_214 :
  totalWorkers workshopA + totalWorkers workshopB + totalWorkers workshopC = 214 := by
  sorry


end total_workers_is_214_l3521_352130


namespace symmetric_points_l3521_352165

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- The theorem stating that (4, -3) is symmetric to (-4, 3) with respect to the origin -/
theorem symmetric_points : symmetric_wrt_origin (4, -3) (-4, 3) := by
  sorry

end symmetric_points_l3521_352165


namespace mad_hatter_win_condition_l3521_352149

/-- Represents the fraction of voters for each candidate and undecided voters -/
structure VoteDistribution where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ
  undecided : ℝ

/-- Represents the final vote count after undecided voters have voted -/
structure FinalVotes where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ

def minimum_fraction_for_mad_hatter (initial_votes : VoteDistribution) : ℝ :=
  0.7

theorem mad_hatter_win_condition (initial_votes : VoteDistribution) 
  (h1 : initial_votes.mad_hatter = 0.2)
  (h2 : initial_votes.march_hare = 0.25)
  (h3 : initial_votes.dormouse = 0.3)
  (h4 : initial_votes.undecided = 0.25)
  (h5 : initial_votes.mad_hatter + initial_votes.march_hare + initial_votes.dormouse + initial_votes.undecided = 1) :
  ∀ (final_votes : FinalVotes),
    (final_votes.mad_hatter ≥ initial_votes.mad_hatter + initial_votes.undecided * minimum_fraction_for_mad_hatter initial_votes) →
    (final_votes.march_hare ≤ initial_votes.march_hare + initial_votes.undecided * (1 - minimum_fraction_for_mad_hatter initial_votes)) →
    (final_votes.dormouse ≤ initial_votes.dormouse + initial_votes.undecided * (1 - minimum_fraction_for_mad_hatter initial_votes)) →
    (final_votes.mad_hatter + final_votes.march_hare + final_votes.dormouse = 1) →
    (final_votes.mad_hatter ≥ final_votes.march_hare ∧ final_votes.mad_hatter ≥ final_votes.dormouse) :=
  sorry

end mad_hatter_win_condition_l3521_352149


namespace markers_multiple_of_four_l3521_352180

-- Define the types of items
structure Items where
  coloring_books : ℕ
  markers : ℕ
  crayons : ℕ

-- Define the function to calculate the maximum number of baskets
def max_baskets (items : Items) : ℕ :=
  min (min (items.coloring_books) (items.markers)) (items.crayons)

-- Theorem statement
theorem markers_multiple_of_four (items : Items) 
  (h1 : items.coloring_books = 12)
  (h2 : items.crayons = 36)
  (h3 : max_baskets items = 4) :
  ∃ k : ℕ, items.markers = 4 * k :=
sorry

end markers_multiple_of_four_l3521_352180


namespace max_value_tangent_l3521_352178

theorem max_value_tangent (x₀ : ℝ) : 
  (∀ x : ℝ, 3 * Real.sin x - 4 * Real.cos x ≤ 3 * Real.sin x₀ - 4 * Real.cos x₀) → 
  Real.tan x₀ = -3/4 := by
sorry

end max_value_tangent_l3521_352178


namespace geometric_sequence_problem_l3521_352129

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n, a (n + 1) = q * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, n < m → a n < a m

theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : increasing_sequence a)
  (h3 : a 5 ^ 2 = a 10)
  (h4 : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  a 5 = 32 := by
sorry

end geometric_sequence_problem_l3521_352129


namespace expression_simplification_and_evaluation_l3521_352119

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (1 - 1 / (x + 1)) / (x / (x - 1)) = (x - 1) / (x + 1) ∧
  (2 - 1) / (2 + 1) = 1 / 3 :=
by sorry

end expression_simplification_and_evaluation_l3521_352119


namespace expression_evaluation_l3521_352150

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^(y + 1) + 6 * y^(x + 1) + 2 * x * y = 2775 := by
  sorry

end expression_evaluation_l3521_352150


namespace max_temperature_range_l3521_352168

theorem max_temperature_range (temps : Fin 5 → ℝ) 
  (avg_temp : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 50)
  (min_temp : ∃ i, temps i = 45 ∧ ∀ j, temps j ≥ 45) :
  ∃ i j, temps i - temps j ≤ 25 ∧ 
  ∀ k l, temps k - temps l ≤ temps i - temps j :=
by sorry

end max_temperature_range_l3521_352168


namespace halloween_candy_l3521_352194

/-- The number of candy pieces Debby's sister had -/
def sister_candy : ℕ := 42

/-- The number of candy pieces eaten on the first night -/
def eaten_candy : ℕ := 35

/-- The number of candy pieces left after eating -/
def remaining_candy : ℕ := 39

/-- Debby's candy pieces -/
def debby_candy : ℕ := 32

theorem halloween_candy :
  debby_candy + sister_candy - eaten_candy = remaining_candy :=
by sorry

end halloween_candy_l3521_352194


namespace river_round_trip_time_l3521_352161

/-- The time taken for a round trip on a river with given conditions -/
theorem river_round_trip_time
  (rower_speed : ℝ)
  (river_speed : ℝ)
  (distance : ℝ)
  (h1 : rower_speed = 6)
  (h2 : river_speed = 1)
  (h3 : distance = 2.916666666666667)
  : (distance / (rower_speed - river_speed)) + (distance / (rower_speed + river_speed)) = 1 := by
  sorry

#eval (2.916666666666667 / (6 - 1)) + (2.916666666666667 / (6 + 1))

end river_round_trip_time_l3521_352161


namespace initial_liquid_x_percentage_l3521_352104

theorem initial_liquid_x_percentage
  (initial_water_percentage : Real)
  (initial_solution_weight : Real)
  (evaporated_water : Real)
  (added_solution : Real)
  (final_liquid_x_percentage : Real)
  (h1 : initial_water_percentage = 70)
  (h2 : initial_solution_weight = 8)
  (h3 : evaporated_water = 3)
  (h4 : added_solution = 3)
  (h5 : final_liquid_x_percentage = 41.25)
  : Real := by
  sorry

#check initial_liquid_x_percentage

end initial_liquid_x_percentage_l3521_352104


namespace smallest_multiple_of_3_4_5_l3521_352185

theorem smallest_multiple_of_3_4_5 : 
  ∀ n : ℕ, (3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n) → n ≥ 60 :=
by sorry

end smallest_multiple_of_3_4_5_l3521_352185


namespace parallel_line_through_point_l3521_352157

/-- A line in the xy-plane is represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the xy-plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- A point lies on a line if its coordinates satisfy the line's equation. -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The problem statement as a theorem. -/
theorem parallel_line_through_point :
  let l1 : Line := { slope := -2, intercept := 3 }
  let p : Point := { x := 1, y := 2 }
  ∃ l2 : Line, parallel l1 l2 ∧ pointOnLine p l2 ∧ l2.slope = -2 ∧ l2.intercept = 4 := by
  sorry

end parallel_line_through_point_l3521_352157


namespace parallelogram_base_length_l3521_352142

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 576) 
  (h2 : height = 48) 
  (h3 : area = base * height) : 
  base = 12 := by
sorry

end parallelogram_base_length_l3521_352142


namespace flag_arrangement_remainder_l3521_352108

/-- Number of blue flags -/
def blue_flags : ℕ := 11

/-- Number of green flags -/
def green_flags : ℕ := 10

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Number of distinguishable arrangements -/
def M : ℕ := 660

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem flag_arrangement_remainder :
  M % 1000 = 660 := by sorry

end flag_arrangement_remainder_l3521_352108


namespace bc_length_l3521_352192

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

-- Define the conditions of the problem
def problem_conditions (t : Triangle) : Prop :=
  t.AB = 5 ∧ t.AC = 6 ∧ Real.sin t.A = 3/5

-- Theorem statement
theorem bc_length (t : Triangle) 
  (h_acute : t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2)
  (h_cond : problem_conditions t) : 
  t.BC = Real.sqrt 13 := by
  sorry

end bc_length_l3521_352192


namespace least_three_digit_multiple_l3521_352151

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (2 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (3 ∣ n) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (2 ∣ m) ∧ (5 ∣ m) ∧ (7 ∣ m) ∧ (3 ∣ m) → m ≥ n) ∧
  n = 210 :=
by
  sorry

end least_three_digit_multiple_l3521_352151


namespace line_through_point_l3521_352188

/-- Given a line with equation y = 2x + b passing through the point (-4, 0), prove that b = 8 -/
theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b) → -- The line has equation y = 2x + b
  (0 = 2 * (-4) + b) →         -- The line passes through the point (-4, 0)
  b = 8 :=                     -- The value of b is 8
by sorry

end line_through_point_l3521_352188


namespace polar_equation_graph_l3521_352163

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a graph in polar coordinates -/
inductive PolarGraph
  | Circle : PolarGraph
  | Ray : PolarGraph
  | Both : PolarGraph

/-- The equation (ρ-3)(θ-π/2)=0 with ρ≥0 -/
def polarEquation (p : PolarPoint) : Prop :=
  (p.ρ - 3) * (p.θ - Real.pi / 2) = 0 ∧ p.ρ ≥ 0

/-- The theorem stating that the equation represents a circle and a ray -/
theorem polar_equation_graph : 
  (∃ p : PolarPoint, polarEquation p) → PolarGraph.Both = PolarGraph.Both :=
sorry

end polar_equation_graph_l3521_352163


namespace line_transformation_l3521_352154

/-- Given a line with equation y = -3/4x + 5, prove that a new line M with one-third the slope
and three times the y-intercept has the equation y = -1/4x + 15. -/
theorem line_transformation (x y : ℝ) :
  (y = -3/4 * x + 5) →
  ∃ (M : ℝ → ℝ),
    (∀ x, M x = -1/4 * x + 15) ∧
    (∀ x, M x = 1/3 * (-3/4) * x + 3 * 5) :=
by sorry

end line_transformation_l3521_352154


namespace anna_candy_store_l3521_352109

def candy_store_problem (initial_amount : ℚ) 
                        (gum_price : ℚ) (gum_quantity : ℕ)
                        (chocolate_price : ℚ) (chocolate_quantity : ℕ)
                        (candy_cane_price : ℚ) (candy_cane_quantity : ℕ) : Prop :=
  let total_spent := gum_price * gum_quantity + 
                     chocolate_price * chocolate_quantity + 
                     candy_cane_price * candy_cane_quantity
  initial_amount - total_spent = 1

theorem anna_candy_store : 
  candy_store_problem 10 1 3 1 5 (1/2) 2 := by
  sorry

end anna_candy_store_l3521_352109


namespace license_plate_count_l3521_352145

/-- The number of consonants available for the first character -/
def num_consonants : ℕ := 20

/-- The number of vowels available for the second and third characters -/
def num_vowels : ℕ := 6

/-- The number of digits and special symbols available for the fourth character -/
def num_digits_and_symbols : ℕ := 12

/-- The total number of possible license plates -/
def total_plates : ℕ := num_consonants * num_vowels * num_vowels * num_digits_and_symbols

theorem license_plate_count : total_plates = 103680 := by
  sorry

end license_plate_count_l3521_352145


namespace motel_pricing_solution_l3521_352126

/-- A motel pricing structure with a flat fee for the first night and a consistent nightly fee thereafter. -/
structure MotelPricing where
  flat_fee : ℝ
  nightly_fee : ℝ

/-- The total cost for a stay at the motel given the number of nights. -/
def total_cost (p : MotelPricing) (nights : ℕ) : ℝ :=
  p.flat_fee + p.nightly_fee * (nights - 1)

theorem motel_pricing_solution :
  ∃ (p : MotelPricing),
    total_cost p 4 = 215 ∧
    total_cost p 3 = 155 ∧
    p.flat_fee = 35 ∧
    p.nightly_fee = 60 := by
  sorry

end motel_pricing_solution_l3521_352126


namespace first_day_sale_is_30_percent_l3521_352170

/-- The percentage of apples sold on the first day -/
def first_day_sale_percentage : ℝ := sorry

/-- The percentage of apples thrown away on the first day -/
def first_day_throwaway_percentage : ℝ := 0.20

/-- The percentage of apples sold on the second day -/
def second_day_sale_percentage : ℝ := 0.50

/-- The total percentage of apples thrown away -/
def total_throwaway_percentage : ℝ := 0.42

/-- Theorem stating that the percentage of apples sold on the first day is 30% -/
theorem first_day_sale_is_30_percent :
  first_day_sale_percentage = 0.30 :=
by
  sorry

end first_day_sale_is_30_percent_l3521_352170


namespace keith_books_l3521_352159

theorem keith_books (jason_books : ℕ) (total_books : ℕ) (h1 : jason_books = 21) (h2 : total_books = 41) :
  total_books - jason_books = 20 := by
sorry

end keith_books_l3521_352159


namespace min_additional_coins_l3521_352125

def friends : ℕ := 15
def initial_coins : ℕ := 100

theorem min_additional_coins :
  let required_coins := (friends * (friends + 1)) / 2
  required_coins - initial_coins = 20 := by
sorry

end min_additional_coins_l3521_352125


namespace square_root_of_sqrt_16_l3521_352173

theorem square_root_of_sqrt_16 : 
  {x : ℝ | x^2 = Real.sqrt 16} = {2, -2} := by sorry

end square_root_of_sqrt_16_l3521_352173


namespace f_extrema_g_negativity_l3521_352102

noncomputable def f (x : ℝ) : ℝ := -x^2 + Real.log x

noncomputable def g (a x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x - 2*a*x

def interval : Set ℝ := Set.Icc (1/Real.exp 1) (Real.exp 1)

theorem f_extrema :
  ∃ (x_min x_max : ℝ), x_min ∈ interval ∧ x_max ∈ interval ∧
  (∀ x ∈ interval, f x ≥ f x_min) ∧
  (∀ x ∈ interval, f x ≤ f x_max) ∧
  f x_min = 1 - Real.exp 2 ∧
  f x_max = -1/2 - 1/2 * Real.log 2 :=
sorry

theorem g_negativity :
  ∀ a : ℝ, (∀ x > 2, g a x < 0) ↔ a ≤ 1/2 :=
sorry

end f_extrema_g_negativity_l3521_352102


namespace molecular_weight_C8H10N4O6_l3521_352166

/-- The atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Carbon atoms in C8H10N4O6 -/
def num_C : ℕ := 8

/-- The number of Hydrogen atoms in C8H10N4O6 -/
def num_H : ℕ := 10

/-- The number of Nitrogen atoms in C8H10N4O6 -/
def num_N : ℕ := 4

/-- The number of Oxygen atoms in C8H10N4O6 -/
def num_O : ℕ := 6

/-- The molecular weight of C8H10N4O6 in g/mol -/
def molecular_weight : ℝ :=
  num_C * atomic_weight_C +
  num_H * atomic_weight_H +
  num_N * atomic_weight_N +
  num_O * atomic_weight_O

theorem molecular_weight_C8H10N4O6 : molecular_weight = 258.22 := by
  sorry

end molecular_weight_C8H10N4O6_l3521_352166


namespace line_intersection_with_x_axis_l3521_352169

/-- A line passing through two given points intersects the x-axis at a specific point. -/
theorem line_intersection_with_x_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_point1 : x₁ = 2 ∧ y₁ = 3) 
  (h_point2 : x₂ = -4 ∧ y₂ = 9) : 
  ∃ x : ℝ, x = 5 ∧ (y₂ - y₁) * x + (x₁ * y₂ - x₂ * y₁) = (y₂ - y₁) * x₁ := by
  sorry

#check line_intersection_with_x_axis

end line_intersection_with_x_axis_l3521_352169


namespace complex_equation_solution_l3521_352183

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l3521_352183


namespace wally_bears_count_l3521_352146

def bear_price (n : ℕ) : ℚ :=
  4 - (n - 1) * (1/2)

def total_cost (num_bears : ℕ) : ℚ :=
  (num_bears : ℚ) / 2 * (2 * 4 + (num_bears - 1) * (-1/2))

theorem wally_bears_count : 
  ∃ (n : ℕ), n > 0 ∧ total_cost n = 354 :=
sorry

end wally_bears_count_l3521_352146


namespace difference_of_squares_special_case_l3521_352111

theorem difference_of_squares_special_case : (827 : ℤ) * 827 - 826 * 828 = 1 := by
  sorry

end difference_of_squares_special_case_l3521_352111


namespace problem_statement_l3521_352186

theorem problem_statement (a b : ℝ) (h : (a + 2)^2 + |b - 1| = 0) :
  (a + b)^2023 = -1 := by sorry

end problem_statement_l3521_352186


namespace geometric_sequence_sum_l3521_352137

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) (h_sum : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  a 5 + a 7 = 6 := by
sorry

end geometric_sequence_sum_l3521_352137


namespace two_absent_one_present_probability_l3521_352153

-- Define the probability of a student being absent
def p_absent : ℚ := 1 / 20

-- Define the probability of a student being present
def p_present : ℚ := 1 - p_absent

-- Define the number of students
def n_students : ℕ := 3

-- Define the number of absent students we're interested in
def n_absent : ℕ := 2

-- Theorem statement
theorem two_absent_one_present_probability :
  (n_students.choose n_absent : ℚ) * p_absent ^ n_absent * p_present ^ (n_students - n_absent) = 57 / 8000 := by
  sorry

end two_absent_one_present_probability_l3521_352153


namespace loan_duration_l3521_352156

/-- Given a loan split into two parts, prove the duration of the second part. -/
theorem loan_duration (total sum : ℕ) (second_part : ℕ) (first_rate second_rate : ℚ) (first_duration : ℕ) :
  total = 2691 →
  second_part = 1656 →
  first_rate = 3 / 100 →
  second_rate = 5 / 100 →
  first_duration = 8 →
  (total - second_part) * first_rate * first_duration = second_part * second_rate * 3 →
  3 = (total - second_part) * first_rate * first_duration / (second_part * second_rate) :=
by sorry

end loan_duration_l3521_352156


namespace lucky_lucy_calculation_l3521_352177

theorem lucky_lucy_calculation (p q r s t : ℤ) 
  (hp : p = 2) (hq : q = 3) (hr : r = 5) (hs : s = 8) : 
  (p - q - r - s + t = p - (q - (r - (s + t)))) ↔ t = 5 := by
  sorry

end lucky_lucy_calculation_l3521_352177


namespace additive_inverses_and_quadratic_roots_l3521_352106

theorem additive_inverses_and_quadratic_roots :
  (∀ x y : ℝ, (∃ z : ℝ, x + z = 0 ∧ y + z = 0) → x + y = 0) ∧
  (∀ q : ℝ, (∀ x : ℝ, x^2 + x + q ≠ 0) → q > -1) := by
  sorry

end additive_inverses_and_quadratic_roots_l3521_352106


namespace thompson_children_ages_l3521_352110

/-- Represents a 5-digit license plate number -/
structure LicensePlate where
  digits : Fin 5 → Nat
  sum_constraint : (digits 0) + (digits 1) + (digits 2) + (digits 3) + (digits 4) = 5
  format_constraint : ∃ a b c, 
    ((digits 0 = a ∧ digits 1 = a ∧ digits 2 = b ∧ digits 3 = b ∧ digits 4 = c) ∨
     (digits 0 = a ∧ digits 1 = b ∧ digits 2 = a ∧ digits 3 = b ∧ digits 4 = c))

/-- Represents the ages of Mr. Thompson's children -/
structure ChildrenAges where
  ages : Fin 6 → Nat
  oldest_12 : ∃ i, ages i = 12
  sum_40 : (ages 0) + (ages 1) + (ages 2) + (ages 3) + (ages 4) + (ages 5) = 40

theorem thompson_children_ages 
  (plate : LicensePlate) 
  (ages : ChildrenAges) 
  (divisibility : ∀ i, plate.digits 0 * 10000 + plate.digits 1 * 1000 + 
                       plate.digits 2 * 100 + plate.digits 3 * 10 + plate.digits 4 % ages.ages i = 0) :
  ¬(∃ i, ages.ages i = 10) :=
sorry

end thompson_children_ages_l3521_352110


namespace lcm_factor_problem_l3521_352167

/-- Given two positive integers with specific properties, prove that the second factor of their LCM is 13 -/
theorem lcm_factor_problem (A B : ℕ+) (X : ℕ+) : 
  (Nat.gcd A B = 23) →
  (Nat.lcm A B = 23 * 12 * X) →
  (A = 299) →
  X = 13 := by
sorry

end lcm_factor_problem_l3521_352167


namespace solve_for_a_l3521_352105

-- Define the operation *
def star (a b : ℝ) : ℝ := 2 * a - b^2

-- Theorem statement
theorem solve_for_a : ∃ (a : ℝ), star a 3 = 7 ∧ a = 8 := by
  sorry

end solve_for_a_l3521_352105


namespace part_one_part_two_l3521_352184

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Define set B
def B : Set ℝ := {x : ℝ | |2*x - 1| ≤ 3}

-- Part I
theorem part_one : 
  {x : ℝ | f 5 x > 9} = {x : ℝ | x < -6 ∨ x > 3} := by sorry

-- Part II
-- Define set A
def A (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ |x - 4|}

theorem part_two :
  {a : ℝ | A a ∪ B = A a} = {a : ℝ | -1 ≤ a ∧ a ≤ 0} := by sorry

end part_one_part_two_l3521_352184


namespace boat_speed_in_still_water_l3521_352144

/-- Proves that the speed of a boat in still water is 22 km/hr, given the conditions -/
theorem boat_speed_in_still_water :
  let stream_speed : ℝ := 5
  let downstream_distance : ℝ := 108
  let downstream_time : ℝ := 4
  let downstream_speed : ℝ := downstream_distance / downstream_time
  let boat_speed_still : ℝ := downstream_speed - stream_speed
  boat_speed_still = 22 := by
  sorry

end boat_speed_in_still_water_l3521_352144


namespace min_value_fraction_l3521_352114

theorem min_value_fraction (x y : ℝ) (h : (x + 2)^2 + y^2 = 1) :
  ∃ k : ℝ, k = (y - 1) / (x - 2) ∧ k ≥ 0 ∧ ∀ m : ℝ, m = (y - 1) / (x - 2) → m ≥ k :=
sorry

end min_value_fraction_l3521_352114


namespace song_count_proof_l3521_352116

def final_song_count (initial : ℕ) (deleted : ℕ) (added : ℕ) : ℕ :=
  initial - deleted + added

theorem song_count_proof (initial deleted added : ℕ) 
  (h1 : initial ≥ deleted) : 
  final_song_count initial deleted added = initial - deleted + added :=
by
  sorry

#eval final_song_count 34 14 44

end song_count_proof_l3521_352116


namespace total_equivalent_pencils_is_139_9_l3521_352172

/-- Calculates the total equivalent number of pencils in three drawers after additions and removals --/
def totalEquivalentPencils (
  initialPencils1 : Float
  ) (initialPencils2 : Float
  ) (initialPens3 : Float
  ) (mikeAddedPencils1 : Float
  ) (sarahAddedPencils2 : Float
  ) (sarahAddedPens2 : Float
  ) (joeRemovedPencils1 : Float
  ) (joeRemovedPencils2 : Float
  ) (joeRemovedPens3 : Float
  ) (exchangeRate : Float
  ) : Float :=
  let finalPencils1 := initialPencils1 + mikeAddedPencils1 - joeRemovedPencils1
  let finalPencils2 := initialPencils2 + sarahAddedPencils2 - joeRemovedPencils2
  let finalPens3 := initialPens3 + sarahAddedPens2 - joeRemovedPens3
  let totalPencils := finalPencils1 + finalPencils2 + (finalPens3 * exchangeRate)
  totalPencils

theorem total_equivalent_pencils_is_139_9 :
  totalEquivalentPencils 41.5 25.2 13.6 30.7 18.5 8.4 5.3 7.1 3.8 2 = 139.9 := by
  sorry

end total_equivalent_pencils_is_139_9_l3521_352172


namespace parabola_tangent_ellipse_l3521_352147

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define point A on the parabola
def point_A : ℝ × ℝ := (2, 4)

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := 4*x - 4

-- Define the ellipse
def ellipse (a b x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- State the theorem
theorem parabola_tangent_ellipse :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  parabola (point_A.1) = point_A.2 →
  tangent_line 1 = 0 →
  tangent_line 0 = -4 →
  ellipse a b 1 0 →
  ellipse a b 0 (-4) →
  ellipse (Real.sqrt 17) 4 1 0 ∧
  ellipse (Real.sqrt 17) 4 0 (-4) :=
sorry

end parabola_tangent_ellipse_l3521_352147


namespace joan_attended_395_games_l3521_352133

/-- The number of baseball games Joan attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

/-- Proof that Joan attended 395 baseball games -/
theorem joan_attended_395_games (total_games night_games missed_games : ℕ) 
  (h1 : total_games = 864)
  (h2 : night_games = 128)
  (h3 : missed_games = 469) :
  games_attended total_games missed_games = 395 := by
  sorry

#eval games_attended 864 469

end joan_attended_395_games_l3521_352133


namespace mo_drinking_difference_l3521_352164

/-- Mo's drinking habits and last week's data --/
structure MoDrinkingData where
  n : ℕ  -- Number of hot chocolate cups on rainy days
  total_cups : ℕ  -- Total cups of tea and hot chocolate last week
  rainy_days : ℕ  -- Number of rainy days last week

/-- Theorem stating the difference between tea and hot chocolate cups --/
theorem mo_drinking_difference (data : MoDrinkingData) : 
  data.n ≤ 2 ∧ 
  data.total_cups = 20 ∧ 
  data.rainy_days = 2 → 
  (7 - data.rainy_days) * 3 - data.rainy_days * data.n = 11 := by
  sorry

#check mo_drinking_difference

end mo_drinking_difference_l3521_352164


namespace helen_oranges_l3521_352182

/-- The number of oranges Helen started with -/
def initial_oranges : ℕ := sorry

/-- Helen gets 29 more oranges from Ann -/
def oranges_from_ann : ℕ := 29

/-- Helen ends up with 38 oranges -/
def final_oranges : ℕ := 38

/-- Theorem stating that the initial number of oranges plus the oranges from Ann equals the final number of oranges -/
theorem helen_oranges : initial_oranges + oranges_from_ann = final_oranges := by sorry

end helen_oranges_l3521_352182


namespace smallest_integer_with_remainders_l3521_352189

theorem smallest_integer_with_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧ 
  x % 6 = 5 ∧
  (∀ y : ℕ, y > 0 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 6 = 5 → x ≤ y) ∧
  x = 59 := by
  sorry

end smallest_integer_with_remainders_l3521_352189


namespace parallelepiped_volume_l3521_352190

/-- The volume of a rectangular parallelepiped with diagonal d, which forms angles of 60° and 45° with two of its edges, is equal to d³√2 / 8 -/
theorem parallelepiped_volume (d : ℝ) (h_d_pos : d > 0) : ∃ (V : ℝ),
  V = d^3 * Real.sqrt 2 / 8 ∧
  ∃ (a b h : ℝ),
    a > 0 ∧ b > 0 ∧ h > 0 ∧
    V = a * b * h ∧
    d^2 = a^2 + b^2 + h^2 ∧
    a / d = Real.cos (π / 4) ∧
    b / d = Real.cos (π / 3) :=
by sorry

end parallelepiped_volume_l3521_352190


namespace multiple_problem_l3521_352103

theorem multiple_problem (m : ℚ) : 38 + m * 43 = 124 → m = 2 := by
  sorry

end multiple_problem_l3521_352103


namespace width_of_right_triangle_in_square_l3521_352193

/-- A right triangle that fits inside a square -/
structure RightTriangleInSquare where
  height : ℝ
  width : ℝ
  square_side : ℝ
  is_right_triangle : True
  fits_in_square : height ≤ square_side ∧ width ≤ square_side

/-- Theorem: The width of a right triangle with height 2 that fits in a 2x2 square is 2 -/
theorem width_of_right_triangle_in_square
  (triangle : RightTriangleInSquare)
  (h_height : triangle.height = 2)
  (h_square : triangle.square_side = 2) :
  triangle.width = 2 :=
sorry

end width_of_right_triangle_in_square_l3521_352193


namespace right_triangle_max_area_l3521_352175

/-- Given a right triangle with perimeter 2, its maximum area is 3 - 2√2 -/
theorem right_triangle_max_area :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  a + b + Real.sqrt (a^2 + b^2) = 2 →
  (1/2) * a * b ≤ 3 - 2 * Real.sqrt 2 :=
by sorry

end right_triangle_max_area_l3521_352175


namespace least_possible_difference_l3521_352136

theorem least_possible_difference (x y z : ℤ) : 
  Even x → Odd y → Odd z → x < y → y < z → y - x > 5 → 
  ∀ (s : ℤ), z - x ≥ s → s ≥ 9 :=
by
  sorry

end least_possible_difference_l3521_352136


namespace fourth_power_of_square_of_fourth_prime_mod_seven_l3521_352160

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- Define the operation we're performing
def operation (n : ℕ) : ℕ := (n^2)^4

-- Theorem statement
theorem fourth_power_of_square_of_fourth_prime_mod_seven :
  operation fourth_smallest_prime % 7 = 0 := by
  sorry

end fourth_power_of_square_of_fourth_prime_mod_seven_l3521_352160


namespace escalator_standing_time_l3521_352115

/-- Represents the time it takes Clea to ride an escalator in different scenarios -/
structure EscalatorRide where
  nonOperatingWalkTime : ℝ
  operatingWalkTime : ℝ
  standingTime : ℝ

/-- Proves that given the conditions, the standing time on the operating escalator is 80 seconds -/
theorem escalator_standing_time (ride : EscalatorRide) 
  (h1 : ride.nonOperatingWalkTime = 120)
  (h2 : ride.operatingWalkTime = 48) :
  ride.standingTime = 80 := by
  sorry

#check escalator_standing_time

end escalator_standing_time_l3521_352115


namespace cube_sum_from_sum_and_square_sum_l3521_352124

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 6) (h2 : x^2 + y^2 = 18) : x^3 + y^3 = 54 := by
  sorry

end cube_sum_from_sum_and_square_sum_l3521_352124


namespace fraction_integer_iff_p_in_range_l3521_352148

theorem fraction_integer_iff_p_in_range (p : ℕ+) :
  (∃ (k : ℕ+), (4 * p + 17 : ℚ) / (3 * p - 7 : ℚ) = k) ↔ 3 ≤ p ∧ p ≤ 40 := by
sorry

end fraction_integer_iff_p_in_range_l3521_352148


namespace prove_triangle_cotangent_formula_l3521_352158

def triangle_cotangent_formula (A B C a b c p r S : Real) : Prop :=
  let ctg_half (x : Real) := (p - x) / r
  A + B + C = Real.pi ∧
  p = (a + b + c) / 2 ∧
  S = Real.sqrt (p * (p - a) * (p - b) * (p - c)) ∧
  S = p * r ∧
  ctg_half a + ctg_half b + ctg_half c = ctg_half a * ctg_half b * ctg_half c

theorem prove_triangle_cotangent_formula (A B C a b c p r S : Real) :
  triangle_cotangent_formula A B C a b c p r S := by
  sorry

end prove_triangle_cotangent_formula_l3521_352158


namespace poster_area_is_zero_l3521_352195

theorem poster_area_is_zero (x y : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (3 * x + 5) * (y + 3) = x * y + 57) : x * y = 0 := by
  sorry

end poster_area_is_zero_l3521_352195


namespace inequality_proof_l3521_352112

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end inequality_proof_l3521_352112


namespace inequality_proof_l3521_352174

theorem inequality_proof (a b c d : ℝ) :
  (a + b + c + d) * (a * b * (c + d) + (a + b) * c * d) - a * b * c * d ≤ 
  (1 / 2) * (a * (b + d) + b * (c + d) + c * (d + a))^2 := by
  sorry

end inequality_proof_l3521_352174


namespace measure_of_inequality_is_zero_l3521_352197

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]
variable (μ : Measure Ω)
variable (ξ η : Ω → ℝ)

theorem measure_of_inequality_is_zero 
  (hξ_integrable : IntegrableOn (|ξ|) Set.univ μ)
  (hη_integrable : IntegrableOn (|η|) Set.univ μ)
  (h_inequality : ∀ (A : Set Ω), MeasurableSet A → ∫ x in A, ξ x ∂μ ≤ ∫ x in A, η x ∂μ) :
  μ {x | ξ x > η x} = 0 := by
  sorry

end measure_of_inequality_is_zero_l3521_352197


namespace smallest_dual_base_representation_l3521_352187

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ), a > 3 ∧ b > 3 ∧
  (1 * a + 3 = 13) ∧
  (3 * b + 1 = 13) ∧
  (∀ (x y : ℕ), x > 3 → y > 3 →
    (1 * x + 3 = 3 * y + 1) →
    (1 * x + 3 ≥ 13)) := by
  sorry

end smallest_dual_base_representation_l3521_352187


namespace prob_three_diff_suits_probability_three_different_suits_l3521_352141

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- The number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- The number of cards in each suit -/
def CardsPerSuit : ℕ := StandardDeck / NumSuits

/-- The probability of picking three cards of different suits from a standard deck without replacement -/
theorem prob_three_diff_suits : 
  (39 / 51) * (26 / 50) = 169 / 425 := by sorry

/-- The main theorem: probability of picking three cards of different suits -/
theorem probability_three_different_suits :
  let p := (CardsPerSuit * (NumSuits - 1) / (StandardDeck - 1)) * 
           (CardsPerSuit * (NumSuits - 2) / (StandardDeck - 2))
  p = 169 / 425 := by sorry

end prob_three_diff_suits_probability_three_different_suits_l3521_352141


namespace polynomial_expansion_p_value_l3521_352131

/-- The value of p in the expansion of (x+y)^8 -/
theorem polynomial_expansion_p_value :
  ∀ (p q : ℝ),
  p > 0 →
  q > 0 →
  p + q = 1 →
  8 * p^7 * q = 28 * p^6 * q^2 →
  p = 7/9 := by
sorry

end polynomial_expansion_p_value_l3521_352131


namespace cosine_equality_l3521_352100

theorem cosine_equality (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (812 * π / 180) → n = 92 := by
  sorry

end cosine_equality_l3521_352100


namespace bruce_fruit_shopping_l3521_352162

def grapes_quantity : ℝ := 8
def grapes_price : ℝ := 70
def mangoes_quantity : ℝ := 11
def mangoes_price : ℝ := 55
def oranges_quantity : ℝ := 5
def oranges_price : ℝ := 45
def apples_quantity : ℝ := 3
def apples_price : ℝ := 90
def cherries_quantity : ℝ := 4.5
def cherries_price : ℝ := 120

def total_cost : ℝ := grapes_quantity * grapes_price + 
                      mangoes_quantity * mangoes_price + 
                      oranges_quantity * oranges_price + 
                      apples_quantity * apples_price + 
                      cherries_quantity * cherries_price

theorem bruce_fruit_shopping : total_cost = 2200 := by
  sorry

end bruce_fruit_shopping_l3521_352162


namespace sum_and_powers_equality_l3521_352134

theorem sum_and_powers_equality : (3 + 7)^3 + (3^2 + 7^2 + 3^3 + 7^3) = 1428 := by
  sorry

end sum_and_powers_equality_l3521_352134


namespace blender_sales_at_600_l3521_352152

/-- Represents the relationship between price and number of customers for blenders. -/
structure BlenderSales where
  price : ℝ
  customers : ℝ

/-- The inverse proportionality constant for blender sales. -/
def k : ℝ := 10 * 300

/-- Axiom: The number of customers is inversely proportional to the price of blenders. -/
axiom inverse_proportion (b : BlenderSales) : b.price * b.customers = k

/-- The theorem to be proved. -/
theorem blender_sales_at_600 :
  ∃ (b : BlenderSales), b.price = 600 ∧ b.customers = 5 :=
sorry

end blender_sales_at_600_l3521_352152


namespace min_value_quadratic_form_l3521_352135

theorem min_value_quadratic_form (x y : ℝ) :
  2 * x^2 + 3 * x * y + 2 * y^2 ≥ 0 ∧
  (2 * x^2 + 3 * x * y + 2 * y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end min_value_quadratic_form_l3521_352135


namespace trig_identities_l3521_352122

theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 ∧
  Real.sin α ^ 2 + Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 4/5 := by
  sorry

end trig_identities_l3521_352122


namespace wolf_sheep_problem_l3521_352113

theorem wolf_sheep_problem (x : ℕ) : 
  (∃ y : ℕ, y = 3 * x + 2 ∧ y = 8 * x - 8) → x = 2 :=
by sorry

end wolf_sheep_problem_l3521_352113


namespace dragon_jewel_ratio_l3521_352143

theorem dragon_jewel_ratio :
  ∀ (initial_jewels : ℕ),
    initial_jewels - 3 + 6 = 24 →
    (6 : ℚ) / initial_jewels = 2 / 7 :=
by
  sorry

end dragon_jewel_ratio_l3521_352143


namespace output_value_S_l3521_352191

theorem output_value_S : ∃ S : ℕ, S = 1 * 3^1 + 2 * 3^2 + 3 * 3^3 ∧ S = 102 := by
  sorry

end output_value_S_l3521_352191


namespace jons_website_hours_l3521_352117

theorem jons_website_hours (earnings_per_visit : ℚ) (visits_per_hour : ℕ) 
  (monthly_earnings : ℚ) (days_in_month : ℕ) 
  (h1 : earnings_per_visit = 1/10) 
  (h2 : visits_per_hour = 50) 
  (h3 : monthly_earnings = 3600) 
  (h4 : days_in_month = 30) : 
  (monthly_earnings / earnings_per_visit / visits_per_hour) / days_in_month = 24 := by
  sorry

end jons_website_hours_l3521_352117


namespace flooring_rate_calculation_l3521_352179

/-- Given a rectangular room with specified dimensions and total flooring cost,
    calculate the rate per square meter for flooring. -/
theorem flooring_rate_calculation
  (length : ℝ) (width : ℝ) (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500)
  : (total_cost / (length * width)) = 800 := by
  sorry

end flooring_rate_calculation_l3521_352179


namespace sequence_property_l3521_352140

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := 2 * a n - a 1

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n, sequence_sum a n = 2 * a n - a 1) →
  (2 * (a 2 + 1) = a 3 + a 1) →
  ∀ n, a n = 2^n :=
by sorry

end sequence_property_l3521_352140


namespace complex_absolute_value_l3521_352101

theorem complex_absolute_value (t : ℝ) : 
  t > 0 → Complex.abs (-5 + t * Complex.I) = 3 * Real.sqrt 13 → t = 2 * Real.sqrt 23 := by
  sorry

end complex_absolute_value_l3521_352101


namespace rect_to_cylindrical_l3521_352120

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical :
  let x : ℝ := -5
  let y : ℝ := 0
  let z : ℝ := -8
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.pi
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = 5 ∧ θ = Real.pi ∧ z = -8 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end rect_to_cylindrical_l3521_352120


namespace michael_truck_meet_once_l3521_352121

/-- Represents the meeting of Michael and the truck -/
structure Meeting where
  time : ℝ
  position : ℝ

/-- Represents the problem setup -/
structure Setup where
  michael_speed : ℝ
  pail_spacing : ℝ
  truck_speed : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck -/
def count_meetings (s : Setup) : ℕ :=
  sorry

/-- The main theorem stating that Michael and the truck meet exactly once -/
theorem michael_truck_meet_once (s : Setup) 
  (h1 : s.michael_speed = 6)
  (h2 : s.pail_spacing = 300)
  (h3 : s.truck_speed = 15)
  (h4 : s.truck_stop_time = 45)
  (h5 : s.initial_distance = 300) : 
  count_meetings s = 1 := by
  sorry

end michael_truck_meet_once_l3521_352121


namespace divisibility_implies_equality_l3521_352181

theorem divisibility_implies_equality (a b : ℕ) 
  (h : (a^2 + a*b + 1) % (b^2 + b*a + 1) = 0) : a = b := by
  sorry

end divisibility_implies_equality_l3521_352181


namespace train_crossing_time_l3521_352155

/-- A train problem -/
theorem train_crossing_time
  (train_speed : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_speed = 20)
  (h2 : platform_length = 300)
  (h3 : platform_crossing_time = 30) :
  let train_length := train_speed * platform_crossing_time - platform_length
  (train_length / train_speed) = 15 := by
sorry

end train_crossing_time_l3521_352155


namespace right_triangle_iff_sum_squares_eq_eight_R_squared_l3521_352196

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  circumradius_positive : 0 < R

/-- Definition of a right triangle -/
def IsRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

/-- Theorem: A triangle satisfies a² + b² + c² = 8R² if and only if it is a right triangle -/
theorem right_triangle_iff_sum_squares_eq_eight_R_squared (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 = 8 * t.R^2 ↔ IsRightTriangle t := by
  sorry

end right_triangle_iff_sum_squares_eq_eight_R_squared_l3521_352196


namespace quiz_competition_score_l3521_352127

/-- Calculates the final score in a quiz competition given the number of correct, incorrect, and unanswered questions. -/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - (incorrect : ℚ) * (1 / 4)

/-- Represents the quiz competition problem -/
theorem quiz_competition_score :
  let total_questions : ℕ := 35
  let correct_answers : ℕ := 17
  let incorrect_answers : ℕ := 12
  let unanswered_questions : ℕ := 6
  correct_answers + incorrect_answers + unanswered_questions = total_questions →
  calculate_score correct_answers incorrect_answers unanswered_questions = 14 := by
  sorry

end quiz_competition_score_l3521_352127


namespace teacher_work_days_l3521_352118

/-- Represents the number of days a teacher works in a month -/
def days_worked_per_month (periods_per_day : ℕ) (pay_per_period : ℕ) (months_worked : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings / months_worked) / (periods_per_day * pay_per_period)

/-- Theorem stating the number of days a teacher works in a month given specific conditions -/
theorem teacher_work_days :
  days_worked_per_month 5 5 6 3600 = 24 := by
  sorry

end teacher_work_days_l3521_352118


namespace freely_falling_body_time_l3521_352107

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 9.808

/-- The additional distance fallen in meters -/
def additional_distance : ℝ := 49.34

/-- The additional time of fall in seconds -/
def additional_time : ℝ := 1.3

/-- The initial time of fall in seconds -/
def initial_time : ℝ := 7.088

theorem freely_falling_body_time :
  g * (initial_time * additional_time + 0.5 * additional_time^2) = additional_distance := by
  sorry

end freely_falling_body_time_l3521_352107
