import Mathlib

namespace NUMINAMATH_CALUDE_largest_three_digit_and_smallest_four_digit_l1_121

theorem largest_three_digit_and_smallest_four_digit : 
  (∃ n : ℕ, n = 999 ∧ ∀ m : ℕ, m < 1000 → m ≤ n) ∧
  (∃ k : ℕ, k = 1000 ∧ ∀ l : ℕ, l ≥ 1000 → l ≥ k) ∧
  (1000 - 999 = 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_and_smallest_four_digit_l1_121


namespace NUMINAMATH_CALUDE_min_value_expression_l1_184

theorem min_value_expression (x : ℝ) (h : x > 0) : 
  9 * x + 1 / x^6 ≥ 10 ∧ ∃ y > 0, 9 * y + 1 / y^6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1_184


namespace NUMINAMATH_CALUDE_susie_vacuum_time_l1_178

/-- Calculates the time to vacuum a house given the time per room and number of rooms -/
def time_to_vacuum_house (time_per_room : ℕ) (num_rooms : ℕ) : ℚ :=
  (time_per_room * num_rooms : ℚ) / 60

/-- Proves that Susie's vacuuming time is 2 hours -/
theorem susie_vacuum_time :
  let time_per_room : ℕ := 20
  let num_rooms : ℕ := 6
  time_to_vacuum_house time_per_room num_rooms = 2 := by
  sorry

end NUMINAMATH_CALUDE_susie_vacuum_time_l1_178


namespace NUMINAMATH_CALUDE_negative_five_greater_than_negative_seven_l1_188

theorem negative_five_greater_than_negative_seven : -5 > -7 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_greater_than_negative_seven_l1_188


namespace NUMINAMATH_CALUDE_sum_vertices_is_nine_l1_141

/-- The number of vertices in a rectangle -/
def rectangle_vertices : ℕ := 4

/-- The number of vertices in a pentagon -/
def pentagon_vertices : ℕ := 5

/-- The sum of vertices of a rectangle and a pentagon -/
def sum_vertices : ℕ := rectangle_vertices + pentagon_vertices

theorem sum_vertices_is_nine : sum_vertices = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_vertices_is_nine_l1_141


namespace NUMINAMATH_CALUDE_initial_principal_is_500_l1_167

/-- Simple interest calculation function -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating that given the conditions, the initial principal must be $500 -/
theorem initial_principal_is_500 :
  ∃ (rate : ℝ),
    simpleInterest 500 rate 2 = 590 ∧
    simpleInterest 500 rate 7 = 815 :=
by
  sorry

#check initial_principal_is_500

end NUMINAMATH_CALUDE_initial_principal_is_500_l1_167


namespace NUMINAMATH_CALUDE_rider_distance_l1_191

/-- The distance traveled by a rider moving back and forth along a moving caravan -/
theorem rider_distance (caravan_length caravan_distance : ℝ) 
  (h_length : caravan_length = 1)
  (h_distance : caravan_distance = 1) : 
  ∃ (rider_speed : ℝ), 
    rider_speed > 0 ∧ 
    (1 / (rider_speed - 1) + 1 / (rider_speed + 1) = 1) ∧
    rider_speed * caravan_distance = 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_rider_distance_l1_191


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1_112

theorem quadratic_equations_solutions :
  (∀ x, x * (x - 3) + x = 3 ↔ x = 3 ∨ x = -1) ∧
  (∀ x, 3 * x^2 - 1 = 4 * x ↔ x = (2 + Real.sqrt 7) / 3 ∨ x = (2 - Real.sqrt 7) / 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1_112


namespace NUMINAMATH_CALUDE_new_person_age_l1_132

theorem new_person_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) :
  n = 8 ∧ initial_avg = 14 ∧ new_avg = 16 →
  ∃ new_age : ℝ,
    new_age = n * new_avg + new_avg - n * initial_avg ∧
    new_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l1_132


namespace NUMINAMATH_CALUDE_billy_decoration_rate_l1_154

/-- The number of eggs Mia can decorate per hour -/
def mia_rate : ℕ := 24

/-- The total number of eggs to be decorated -/
def total_eggs : ℕ := 170

/-- The time taken by Mia and Billy together to decorate all eggs (in hours) -/
def total_time : ℕ := 5

/-- Billy's decoration rate (in eggs per hour) -/
def billy_rate : ℕ := total_eggs / total_time - mia_rate

theorem billy_decoration_rate :
  billy_rate = 10 := by sorry

end NUMINAMATH_CALUDE_billy_decoration_rate_l1_154


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1_177

-- Define the universal set U
def U : Set Nat := {1, 2, 3}

-- Define the set M
def M : Set Nat := {1}

-- State the theorem
theorem complement_of_M_in_U : 
  (U \ M) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1_177


namespace NUMINAMATH_CALUDE_thirty_two_distributions_l1_111

/-- Represents a knockout tournament with 6 players. -/
structure Tournament :=
  (players : Fin 6 → ℕ)

/-- The number of possible outcomes for each match. -/
def match_outcomes : ℕ := 2

/-- The number of rounds in the tournament. -/
def num_rounds : ℕ := 5

/-- Calculates the total number of possible prize distribution orders. -/
def prize_distributions (t : Tournament) : ℕ :=
  match_outcomes ^ num_rounds

/-- Theorem stating that there are 32 possible prize distribution orders. -/
theorem thirty_two_distributions (t : Tournament) :
  prize_distributions t = 32 := by
  sorry

end NUMINAMATH_CALUDE_thirty_two_distributions_l1_111


namespace NUMINAMATH_CALUDE_vacation_savings_l1_161

def total_income : ℝ := 72800
def total_expenses : ℝ := 54200
def deposit_rate : ℝ := 0.1

theorem vacation_savings : 
  (total_income - total_expenses) * (1 - deposit_rate) = 16740 :=
by sorry

end NUMINAMATH_CALUDE_vacation_savings_l1_161


namespace NUMINAMATH_CALUDE_square_area_on_parabola_and_line_square_area_on_parabola_and_line_is_36_l1_181

theorem square_area_on_parabola_and_line : ℝ → Prop :=
  fun area =>
    ∃ (x₁ x₂ : ℝ),
      -- The endpoints lie on the parabola y = x^2 + 4x + 3
      8 = x₁^2 + 4*x₁ + 3 ∧
      8 = x₂^2 + 4*x₂ + 3 ∧
      -- The side length is the absolute difference between x-coordinates
      area = (x₁ - x₂)^2 ∧
      -- The area of the square is 36
      area = 36

-- The proof of the theorem
theorem square_area_on_parabola_and_line_is_36 :
  square_area_on_parabola_and_line 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_and_line_square_area_on_parabola_and_line_is_36_l1_181


namespace NUMINAMATH_CALUDE_rice_weight_l1_163

/-- Given rice divided equally into 4 containers, with 70 ounces in each container,
    and 1 pound equaling 16 ounces, the total amount of rice is 17.5 pounds. -/
theorem rice_weight (containers : Nat) (ounces_per_container : Nat) (ounces_per_pound : Nat)
    (h1 : containers = 4)
    (h2 : ounces_per_container = 70)
    (h3 : ounces_per_pound = 16) :
    (containers * ounces_per_container : Rat) / ounces_per_pound = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_rice_weight_l1_163


namespace NUMINAMATH_CALUDE_baker_problem_l1_107

def verify_cake_info (initial_cakes : ℕ) (sold_cakes : ℕ) (remaining_cakes : ℕ) : Prop :=
  initial_cakes - sold_cakes = remaining_cakes

def can_determine_initial_pastries (initial_cakes : ℕ) (sold_cakes : ℕ) (remaining_cakes : ℕ) 
    (sold_pastries : ℕ) : Prop :=
  false

theorem baker_problem (initial_cakes : ℕ) (sold_cakes : ℕ) (remaining_cakes : ℕ) 
    (sold_pastries : ℕ) :
  initial_cakes = 149 →
  sold_cakes = 10 →
  remaining_cakes = 139 →
  sold_pastries = 90 →
  verify_cake_info initial_cakes sold_cakes remaining_cakes ∧
  ¬can_determine_initial_pastries initial_cakes sold_cakes remaining_cakes sold_pastries :=
by
  sorry

end NUMINAMATH_CALUDE_baker_problem_l1_107


namespace NUMINAMATH_CALUDE_final_water_level_l1_147

/-- The final water level in a system of two connected cylindrical vessels -/
theorem final_water_level 
  (h : ℝ) -- Initial height of both liquids
  (ρ_water : ℝ) -- Density of water
  (ρ_oil : ℝ) -- Density of oil
  (h_pos : h > 0)
  (ρ_water_pos : ρ_water > 0)
  (ρ_oil_pos : ρ_oil > 0)
  (h_val : h = 40)
  (ρ_water_val : ρ_water = 1000)
  (ρ_oil_val : ρ_oil = 700) :
  ∃ (h_water : ℝ), h_water = 280 / 17 ∧ 
    ρ_water * h_water = ρ_oil * (h - h_water) ∧
    h_water > 0 ∧ h_water < h :=
by
  sorry


end NUMINAMATH_CALUDE_final_water_level_l1_147


namespace NUMINAMATH_CALUDE_squirrel_nut_difference_l1_114

theorem squirrel_nut_difference :
  let num_squirrels : ℕ := 4
  let num_nuts : ℕ := 2
  num_squirrels - num_nuts = 2 :=
by sorry

end NUMINAMATH_CALUDE_squirrel_nut_difference_l1_114


namespace NUMINAMATH_CALUDE_days_between_appointments_l1_142

/-- Represents the waiting periods for Mark's vaccine appointments -/
structure VaccineWaitingPeriod where
  totalWait : ℕ
  initialWait : ℕ
  finalWait : ℕ

/-- Theorem stating the number of days between first and second appointments -/
theorem days_between_appointments (mark : VaccineWaitingPeriod)
  (h1 : mark.totalWait = 38)
  (h2 : mark.initialWait = 4)
  (h3 : mark.finalWait = 14) :
  mark.totalWait - mark.initialWait - mark.finalWait = 20 := by
  sorry

#check days_between_appointments

end NUMINAMATH_CALUDE_days_between_appointments_l1_142


namespace NUMINAMATH_CALUDE_rectangular_prism_cutout_l1_196

theorem rectangular_prism_cutout (x y : ℕ) : 
  (15 * 5 * 4 - y * 5 * x = 120) → (x < 4 ∧ y < 15) → x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_cutout_l1_196


namespace NUMINAMATH_CALUDE_principal_calculation_l1_180

/-- The principal amount in dollars -/
def principal : ℝ := sorry

/-- The compounded amount after 7 years -/
def compounded_amount : ℝ := sorry

/-- The difference between compounded amount and principal -/
def difference : ℝ := 5000

/-- The interest rate for the first 2 years -/
def rate1 : ℝ := 0.03

/-- The interest rate for the next 3 years -/
def rate2 : ℝ := 0.04

/-- The interest rate for the last 2 years -/
def rate3 : ℝ := 0.05

/-- The number of years for each interest rate period -/
def years1 : ℕ := 2
def years2 : ℕ := 3
def years3 : ℕ := 2

theorem principal_calculation :
  principal * (1 + rate1) ^ years1 * (1 + rate2) ^ years2 * (1 + rate3) ^ years3 = principal + difference :=
by sorry

end NUMINAMATH_CALUDE_principal_calculation_l1_180


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1_190

theorem sqrt_expression_equality : 
  Real.sqrt 8 - 2 * Real.sqrt (1/2) + (Real.sqrt 27 + 2 * Real.sqrt 6) / Real.sqrt 3 = 3 * Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1_190


namespace NUMINAMATH_CALUDE_jezebel_bouquet_cost_l1_164

/-- The cost of a bouquet of flowers -/
def bouquet_cost (red_roses_per_dozen : ℕ) (red_rose_cost : ℚ) (sunflowers : ℕ) (sunflower_cost : ℚ) : ℚ :=
  (red_roses_per_dozen * 12 * red_rose_cost) + (sunflowers * sunflower_cost)

/-- Theorem: The cost of Jezebel's bouquet is $45 -/
theorem jezebel_bouquet_cost :
  bouquet_cost 2 (3/2) 3 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_jezebel_bouquet_cost_l1_164


namespace NUMINAMATH_CALUDE_product_ratio_l1_115

def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def first_three_primes : List ℕ := [2, 3, 5]
def next_three_composites : List ℕ := [14, 15, 16]

theorem product_ratio :
  (first_six_composites.prod) / ((first_three_primes ++ next_three_composites).prod) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_l1_115


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1_152

theorem weight_of_replaced_person (initial_count : ℕ) (average_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  let replaced_person_weight := new_person_weight - initial_count * average_increase
  replaced_person_weight

#check weight_of_replaced_person 8 5 75 -- Should evaluate to 35

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1_152


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1_134

theorem smallest_positive_solution (x : ℕ) : x = 30 ↔ 
  (x > 0 ∧ 
   (51 * x + 15) % 35 = 5 ∧ 
   ∀ y : ℕ, y > 0 → (51 * y + 15) % 35 = 5 → x ≤ y) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1_134


namespace NUMINAMATH_CALUDE_base_thirteen_unique_l1_198

/-- Converts a list of digits in base b to its decimal representation -/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Theorem stating that 13 is the unique base for which the equation holds -/
theorem base_thirteen_unique :
  ∃! b : Nat, b > 1 ∧ 
    toDecimal [5, 3, 2, 4] b + toDecimal [6, 4, 7, 3] b = toDecimal [1, 2, 5, 3, 2] b :=
by sorry

end NUMINAMATH_CALUDE_base_thirteen_unique_l1_198


namespace NUMINAMATH_CALUDE_strategy_exists_l1_193

/-- Represents a question of the form "Is n smaller than a?" --/
structure Question where
  a : ℕ
  deriving Repr

/-- Represents an answer to a question --/
inductive Answer
  | Yes
  | No
  deriving Repr

/-- Represents a strategy for determining n --/
structure Strategy where
  questions : List Question
  decisionFunction : List Answer → ℕ

/-- Theorem stating that a strategy exists to determine n within the given constraints --/
theorem strategy_exists :
  ∃ (s : Strategy),
    (s.questions.length ≤ 10) ∧
    (∀ n : ℕ,
      n > 0 ∧ n ≤ 144 →
      ∃ (answers : List Answer),
        answers.length = s.questions.length ∧
        s.decisionFunction answers = n) :=
  sorry


end NUMINAMATH_CALUDE_strategy_exists_l1_193


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1_144

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x - 3 > 0}

def N : Set ℝ := {x | |x| ≤ 3}

theorem complement_M_intersect_N :
  (Set.compl M ∩ N) = Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1_144


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l1_149

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

-- Statement to prove
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l1_149


namespace NUMINAMATH_CALUDE_office_viewing_time_l1_122

/-- The number of episodes in The Office series -/
def total_episodes : ℕ := 201

/-- The number of episodes watched per week -/
def episodes_per_week : ℕ := 3

/-- The number of weeks needed to watch all episodes -/
def weeks_to_watch : ℕ := 67

theorem office_viewing_time :
  (total_episodes + episodes_per_week - 1) / episodes_per_week = weeks_to_watch :=
sorry

end NUMINAMATH_CALUDE_office_viewing_time_l1_122


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1_182

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1_182


namespace NUMINAMATH_CALUDE_marble_draw_probability_l1_175

/-- Represents a bag of marbles -/
structure MarbleBag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0
  red : ℕ := 0
  green : ℕ := 0

/-- Calculate the total number of marbles in a bag -/
def MarbleBag.total (bag : MarbleBag) : ℕ :=
  bag.white + bag.black + bag.yellow + bag.blue + bag.red + bag.green

/-- Definition of Bag A -/
def bagA : MarbleBag := { white := 5, black := 5 }

/-- Definition of Bag B -/
def bagB : MarbleBag := { yellow := 8, blue := 7 }

/-- Definition of Bag C -/
def bagC : MarbleBag := { yellow := 3, blue := 7 }

/-- Definition of Bag D -/
def bagD : MarbleBag := { red := 4, green := 6 }

/-- Probability of drawing a yellow marble from a bag -/
def probYellow (bag : MarbleBag) : ℚ :=
  bag.yellow / bag.total

/-- Probability of drawing a green marble from a bag -/
def probGreen (bag : MarbleBag) : ℚ :=
  bag.green / bag.total

/-- Main theorem: Probability of drawing yellow as second and green as third marble -/
theorem marble_draw_probability : 
  (1/2 * probYellow bagB + 1/2 * probYellow bagC) * probGreen bagD = 17/50 := by
  sorry

end NUMINAMATH_CALUDE_marble_draw_probability_l1_175


namespace NUMINAMATH_CALUDE_zainab_hourly_wage_l1_159

/-- Zainab's work schedule and earnings -/
structure WorkSchedule where
  daysPerWeek : ℕ
  hoursPerDay : ℕ
  totalWeeks : ℕ
  totalEarnings : ℕ

/-- Calculate hourly wage given a work schedule -/
def hourlyWage (schedule : WorkSchedule) : ℚ :=
  schedule.totalEarnings / (schedule.daysPerWeek * schedule.hoursPerDay * schedule.totalWeeks)

/-- Zainab's specific work schedule -/
def zainabSchedule : WorkSchedule :=
  { daysPerWeek := 3
  , hoursPerDay := 4
  , totalWeeks := 4
  , totalEarnings := 96 }

/-- Theorem: Zainab's hourly wage is $2 -/
theorem zainab_hourly_wage :
  hourlyWage zainabSchedule = 2 := by
  sorry

end NUMINAMATH_CALUDE_zainab_hourly_wage_l1_159


namespace NUMINAMATH_CALUDE_milk_calculation_l1_110

theorem milk_calculation (initial : ℚ) (given : ℚ) (received : ℚ) :
  initial = 5 →
  given = 18 / 4 →
  received = 7 / 4 →
  initial - given + received = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_milk_calculation_l1_110


namespace NUMINAMATH_CALUDE_not_always_true_point_not_in_plane_l1_171

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the necessary relations
variable (belongs_to : Point → Line → Prop)
variable (belongs_to_plane : Point → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Define the theorem
theorem not_always_true_point_not_in_plane 
  (A : Point) (l : Line) (α : Plane) : 
  ¬(∀ A l α, ¬(line_in_plane l α) → belongs_to A l → ¬(belongs_to_plane A α)) :=
by sorry

end NUMINAMATH_CALUDE_not_always_true_point_not_in_plane_l1_171


namespace NUMINAMATH_CALUDE_game_lives_per_player_l1_165

theorem game_lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) :
  initial_players = 8 →
  additional_players = 2 →
  total_lives = 60 →
  (total_lives / (initial_players + additional_players) : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_game_lives_per_player_l1_165


namespace NUMINAMATH_CALUDE_square_area_on_parallel_lines_l1_131

/-- Represents a line in a 2D plane -/
structure Line where
  -- Add necessary fields for a line

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Checks if three lines are parallel -/
def are_parallel (l1 l2 l3 : Line) : Prop := sorry

/-- Calculates the perpendicular distance between two lines -/
def perpendicular_distance (l1 l2 : Line) : ℝ := sorry

/-- Checks if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

/-- Calculates the area of a square -/
def square_area (s : Square) : ℝ := sorry

/-- The main theorem -/
theorem square_area_on_parallel_lines 
  (l1 l2 l3 : Line) 
  (s : Square) :
  are_parallel l1 l2 l3 →
  perpendicular_distance l1 l2 = 3 →
  perpendicular_distance l2 l3 = 3 →
  point_on_line s.a l1 →
  point_on_line s.b l3 →
  point_on_line s.c l2 →
  square_area s = 45 := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parallel_lines_l1_131


namespace NUMINAMATH_CALUDE_basketball_fall_certain_l1_145

-- Define the type for events
inductive Event
  | RainTomorrow
  | RollEvenDice
  | TVAdvertisement
  | BasketballFall

-- Define a predicate for certain events
def IsCertain (e : Event) : Prop :=
  match e with
  | Event.BasketballFall => True
  | _ => False

-- Define the law of gravity (simplified)
axiom law_of_gravity : ∀ (object : Type), object → object → Prop

-- Theorem statement
theorem basketball_fall_certain :
  ∀ (e : Event), IsCertain e ↔ e = Event.BasketballFall :=
sorry

end NUMINAMATH_CALUDE_basketball_fall_certain_l1_145


namespace NUMINAMATH_CALUDE_sam_win_probability_proof_l1_170

/-- The probability of hitting the target with one shot -/
def hit_probability : ℚ := 2/5

/-- The probability of missing the target with one shot -/
def miss_probability : ℚ := 3/5

/-- Sam wins when the total number of shots (including the last successful one) is odd -/
axiom sam_wins_on_odd : True

/-- The probability that Sam wins the game -/
def sam_win_probability : ℚ := 5/8

theorem sam_win_probability_proof : 
  sam_win_probability = hit_probability + miss_probability * miss_probability * sam_win_probability :=
sorry

end NUMINAMATH_CALUDE_sam_win_probability_proof_l1_170


namespace NUMINAMATH_CALUDE_foreign_exchange_earnings_equation_l1_199

/-- Represents the monthly decline rate as a real number between 0 and 1 -/
def monthly_decline_rate : ℝ := sorry

/-- Initial foreign exchange earnings in July (in millions of USD) -/
def initial_earnings : ℝ := 200

/-- Foreign exchange earnings in September (in millions of USD) -/
def final_earnings : ℝ := 98

/-- The number of months between July and September -/
def months_elapsed : ℕ := 2

theorem foreign_exchange_earnings_equation :
  initial_earnings * (1 - monthly_decline_rate) ^ months_elapsed = final_earnings :=
sorry

end NUMINAMATH_CALUDE_foreign_exchange_earnings_equation_l1_199


namespace NUMINAMATH_CALUDE_median_and_perpendicular_bisector_equations_l1_102

/-- Given three points in a plane, prove the equations of the median and perpendicular bisector of a side -/
theorem median_and_perpendicular_bisector_equations 
  (A B C : ℝ × ℝ) 
  (hA : A = (1, 2)) 
  (hB : B = (-1, 4)) 
  (hC : C = (5, 2)) : 
  (∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) ↔ (y - 3 = -1 * (x - 0))) ∧ 
  (∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) ↔ (y = x + 3)) := by
  sorry


end NUMINAMATH_CALUDE_median_and_perpendicular_bisector_equations_l1_102


namespace NUMINAMATH_CALUDE_tan_neg_3780_degrees_l1_187

theorem tan_neg_3780_degrees : Real.tan ((-3780 : ℝ) * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_3780_degrees_l1_187


namespace NUMINAMATH_CALUDE_euler_family_mean_age_is_68_over_7_l1_130

/-- The mean age of the Euler family's children -/
def euler_family_mean_age : ℚ :=
  let ages : List ℕ := [6, 6, 6, 6, 12, 16, 16]
  (ages.sum : ℚ) / ages.length

/-- Theorem stating the mean age of the Euler family's children -/
theorem euler_family_mean_age_is_68_over_7 :
  euler_family_mean_age = 68 / 7 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_is_68_over_7_l1_130


namespace NUMINAMATH_CALUDE_players_per_group_l1_139

theorem players_per_group (new_players returning_players total_groups : ℕ) : 
  new_players = 48 → 
  returning_players = 6 → 
  total_groups = 9 → 
  (new_players + returning_players) / total_groups = 6 :=
by sorry

end NUMINAMATH_CALUDE_players_per_group_l1_139


namespace NUMINAMATH_CALUDE_password_count_correct_l1_126

/-- The number of letters in the English alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letters in the password -/
def num_password_letters : ℕ := 2

/-- The number of digits in the password -/
def num_password_digits : ℕ := 2

/-- The total number of possible passwords -/
def num_possible_passwords : ℕ := (num_letters * (num_letters - 1)) * (num_digits * (num_digits - 1))

theorem password_count_correct :
  num_possible_passwords = num_letters * (num_letters - 1) * num_digits * (num_digits - 1) := by
  sorry

end NUMINAMATH_CALUDE_password_count_correct_l1_126


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1_155

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1_155


namespace NUMINAMATH_CALUDE_total_clothing_pieces_l1_174

theorem total_clothing_pieces (shirts trousers : ℕ) 
  (h1 : shirts = 589) 
  (h2 : trousers = 345) : 
  shirts + trousers = 934 := by
  sorry

end NUMINAMATH_CALUDE_total_clothing_pieces_l1_174


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l1_113

/-- Represents the repair cost calculation for Ramu's car sale --/
theorem repair_cost_calculation (initial_cost selling_price profit_percent : ℝ) (R : ℝ) : 
  initial_cost = 34000 →
  selling_price = 65000 →
  profit_percent = 41.30434782608695 →
  profit_percent = ((selling_price - (initial_cost + R)) / (initial_cost + R)) * 100 :=
by sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l1_113


namespace NUMINAMATH_CALUDE_coefficient_a3b2_in_expansion_l1_194

theorem coefficient_a3b2_in_expansion : ∃ (coeff : ℕ),
  coeff = (Nat.choose 5 3) * (Nat.choose 8 4) ∧
  coeff = 700 := by sorry

end NUMINAMATH_CALUDE_coefficient_a3b2_in_expansion_l1_194


namespace NUMINAMATH_CALUDE_sin_A_value_l1_156

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem sin_A_value (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.b = Real.sqrt 3) 
  (h3 : t.A + t.C = 2 * t.B) 
  (h4 : t.A + t.B + t.C = Real.pi) 
  (h5 : t.a / Real.sin t.A = t.b / Real.sin t.B) : 
  Real.sin t.A = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_A_value_l1_156


namespace NUMINAMATH_CALUDE_max_value_xy_8x_y_l1_173

theorem max_value_xy_8x_y (x y : ℝ) (h : x^2 + y^2 = 20) :
  ∃ (M : ℝ), M = 42 ∧ xy + 8*x + y ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 20 ∧ x₀*y₀ + 8*x₀ + y₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_value_xy_8x_y_l1_173


namespace NUMINAMATH_CALUDE_rectangle_circle_union_area_l1_137

/-- The area of the union of a rectangle and a circle with specific dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := 8
  let circle_radius : ℝ := 12
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let circle_area : ℝ := π * circle_radius^2
  let quarter_circle_area : ℝ := circle_area / 4
  rectangle_area + (circle_area - quarter_circle_area) = 96 + 108 * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_union_area_l1_137


namespace NUMINAMATH_CALUDE_mean_interior_angle_quadrilateral_l1_128

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- Formula for the sum of interior angles of a polygon -/
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The mean value of interior angles of a quadrilateral -/
theorem mean_interior_angle_quadrilateral :
  (sum_of_interior_angles quadrilateral_sides) / quadrilateral_sides = 90 := by
  sorry

end NUMINAMATH_CALUDE_mean_interior_angle_quadrilateral_l1_128


namespace NUMINAMATH_CALUDE_removed_number_theorem_l1_169

theorem removed_number_theorem (n : ℕ) (m : ℕ) :
  m ≤ n →
  (n * (n + 1) / 2 - m) / (n - 1) = 163/4 →
  m = 61 := by
sorry

end NUMINAMATH_CALUDE_removed_number_theorem_l1_169


namespace NUMINAMATH_CALUDE_turtle_contradiction_l1_151

/-- Represents the position of a turtle in the line -/
inductive Position
  | Front
  | Middle
  | Back

/-- Represents a turtle with its position and statements about other turtles -/
structure Turtle where
  position : Position
  turtles_behind : Nat
  turtles_in_front : Nat

/-- The scenario of three turtles in a line -/
def turtle_scenario : List Turtle :=
  [ { position := Position.Front
    , turtles_behind := 2
    , turtles_in_front := 0 }
  , { position := Position.Middle
    , turtles_behind := 1
    , turtles_in_front := 1 }
  , { position := Position.Back
    , turtles_behind := 1
    , turtles_in_front := 1 } ]

/-- Theorem stating that the turtle scenario leads to a contradiction -/
theorem turtle_contradiction : 
  ∀ (t : Turtle), t ∈ turtle_scenario → 
    (t.position = Position.Front → t.turtles_behind = 2) ∧
    (t.position = Position.Middle → t.turtles_behind = 1 ∧ t.turtles_in_front = 1) ∧
    (t.position = Position.Back → t.turtles_behind = 1 ∧ t.turtles_in_front = 1) →
    False := by
  sorry

end NUMINAMATH_CALUDE_turtle_contradiction_l1_151


namespace NUMINAMATH_CALUDE_interval_of_decrease_l1_135

/-- The function f(x) = -x^2 + 4x - 3 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -2*x + 4

theorem interval_of_decrease (x : ℝ) :
  x ≥ 2 → (∀ y, y > x → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_interval_of_decrease_l1_135


namespace NUMINAMATH_CALUDE_third_line_through_integer_point_l1_123

theorem third_line_through_integer_point (a b c : ℝ) :
  (∃ y : ℝ, (y = a + b ∧ y = b + c) ∨ (y = a + b ∧ y = c + a) ∨ (y = b + c ∧ y = c + a)) →
  ∃ x y : ℤ, (y = a * x + a) ∨ (y = b * x + c) ∨ (y = c * x + a) :=
by sorry

end NUMINAMATH_CALUDE_third_line_through_integer_point_l1_123


namespace NUMINAMATH_CALUDE_solve_invitations_l1_136

def invitations_problem (I : ℝ) : Prop :=
  let rsvp_rate : ℝ := 0.9
  let show_up_rate : ℝ := 0.8
  let no_gift_attendees : ℕ := 10
  let thank_you_cards : ℕ := 134
  
  (rsvp_rate * show_up_rate * I - no_gift_attendees : ℝ) = thank_you_cards

theorem solve_invitations : ∃ I : ℝ, invitations_problem I ∧ I = 200 := by
  sorry

end NUMINAMATH_CALUDE_solve_invitations_l1_136


namespace NUMINAMATH_CALUDE_function_monotonically_increasing_l1_118

/-- The function f(x) = x^2 - 2x + 8 is monotonically increasing on the interval (1, +∞) -/
theorem function_monotonically_increasing (x y : ℝ) : x > 1 → y > 1 → x < y →
  (x^2 - 2*x + 8) < (y^2 - 2*y + 8) := by sorry

end NUMINAMATH_CALUDE_function_monotonically_increasing_l1_118


namespace NUMINAMATH_CALUDE_sqrt_plus_reciprocal_sqrt_l1_119

theorem sqrt_plus_reciprocal_sqrt (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 98) :
  Real.sqrt x + 1 / Real.sqrt x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_reciprocal_sqrt_l1_119


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l1_185

theorem binomial_expansion_example : 97^3 + 3*(97^2) + 3*97 + 1 = 940792 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l1_185


namespace NUMINAMATH_CALUDE_recycling_program_earnings_l1_162

/-- Represents the referral program structure and earnings --/
structure ReferralProgram where
  initial_signup_bonus : ℚ
  first_tier_referral_bonus : ℚ
  second_tier_referral_bonus : ℚ
  friend_signup_bonus : ℚ
  friend_referral_bonus : ℚ
  first_day_referrals : ℕ
  first_day_friends_referrals : ℕ
  week_end_friends_referrals : ℕ
  third_day_referrals : ℕ
  fourth_day_friends_referrals : ℕ

/-- Calculates the total earnings for Katrina and her friends --/
def total_earnings (program : ReferralProgram) : ℚ :=
  sorry

/-- The recycling program referral structure --/
def recycling_program : ReferralProgram := {
  initial_signup_bonus := 5,
  first_tier_referral_bonus := 8,
  second_tier_referral_bonus := 3/2,
  friend_signup_bonus := 5,
  friend_referral_bonus := 2,
  first_day_referrals := 5,
  first_day_friends_referrals := 3,
  week_end_friends_referrals := 2,
  third_day_referrals := 2,
  fourth_day_friends_referrals := 1
}

/-- Theorem stating that the total earnings for Katrina and her friends is $190.50 --/
theorem recycling_program_earnings :
  total_earnings recycling_program = 381/2 := by
  sorry

end NUMINAMATH_CALUDE_recycling_program_earnings_l1_162


namespace NUMINAMATH_CALUDE_modInverses_correct_l1_124

def modInverses (n : ℕ) : List ℕ :=
  match n with
  | 2 => [1]
  | 3 => [1, 2]
  | 4 => [1, 3]
  | 5 => [1, 3, 2, 4]
  | 6 => [1, 5]
  | 7 => [1, 4, 5, 2, 3, 6]
  | 8 => [1, 3, 5, 7]
  | 9 => [1, 5, 7, 2, 4, 8]
  | 10 => [1, 7, 3, 9]
  | _ => []

theorem modInverses_correct (n : ℕ) (h : 2 ≤ n ∧ n ≤ 10) :
  ∀ a ∈ modInverses n, ∃ b : ℕ, 
    1 ≤ b ∧ b < n ∧ 
    (a * b) % n = 1 ∧ 
    Nat.gcd b n = 1 :=
by sorry

end NUMINAMATH_CALUDE_modInverses_correct_l1_124


namespace NUMINAMATH_CALUDE_prob_second_red_three_two_l1_150

/-- Represents a bag of colored balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- Calculates the probability of drawing a red ball on the second draw,
    given that the first ball drawn is red -/
def prob_second_red_given_first_red (b : Bag) : ℚ :=
  if b.red > 0 then
    (b.red - 1) / (b.red + b.white - 1)
  else
    0

/-- Theorem stating that for a bag with 3 red and 2 white balls,
    the probability of drawing a red ball on the second draw,
    given that the first ball drawn is red, is 1/2 -/
theorem prob_second_red_three_two : 
  prob_second_red_given_first_red ⟨3, 2⟩ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_red_three_two_l1_150


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l1_148

/-- Given vectors a and b in R², prove that |a - b| = 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l1_148


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1_108

theorem geometric_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
  (h2 : a 2 = 2) 
  (h3 : q > 0) 
  (h4 : S 4 / S 2 = 10) : 
  a 4 = 18 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1_108


namespace NUMINAMATH_CALUDE_height_average_comparison_l1_176

theorem height_average_comparison 
  (h₁ : ℝ → ℝ → ℝ → ℝ → ℝ → Prop) 
  (a b c d : ℝ) 
  (h₂ : 3 * a + 2 * b = 2 * c + 3 * d) 
  (h₃ : a > d) : 
  |c + d| / 2 > |a + b| / 2 := by
sorry

end NUMINAMATH_CALUDE_height_average_comparison_l1_176


namespace NUMINAMATH_CALUDE_log_expression_equals_one_l1_104

theorem log_expression_equals_one : 
  (((1 - Real.log 3 / Real.log 6) ^ 2 + (Real.log 2 / Real.log 6) * (Real.log 18 / Real.log 6)) / (Real.log 4 / Real.log 6)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_one_l1_104


namespace NUMINAMATH_CALUDE_red_balls_count_l1_158

theorem red_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (yellow : ℕ) (purple : ℕ) 
  (h_total : total = 60)
  (h_white : white = 22)
  (h_green : green = 18)
  (h_yellow : yellow = 2)
  (h_purple : purple = 3)
  (h_prob : (white + green + yellow : ℚ) / total = 7/10) :
  total - (white + green + yellow + purple) = 15 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l1_158


namespace NUMINAMATH_CALUDE_first_pipe_fill_time_l1_138

/-- Given two pipes that can fill a tank, an outlet pipe that can empty it, 
    and the time it takes to fill the tank when all pipes are open, 
    this theorem proves the time it takes for the first pipe to fill the tank. -/
theorem first_pipe_fill_time (t : ℝ) (h1 : t > 0) 
  (h2 : 1/t + 1/30 - 1/45 = 1/15) : t = 18 := by
  sorry

end NUMINAMATH_CALUDE_first_pipe_fill_time_l1_138


namespace NUMINAMATH_CALUDE_find_number_l1_109

theorem find_number : ∃! x : ℝ, (8 * x + 5400) / 12 = 530 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1_109


namespace NUMINAMATH_CALUDE_lane_length_correct_l1_166

/-- Represents the length of a swimming lane in meters -/
def lane_length : ℝ := 100

/-- Represents the number of round trips swum -/
def round_trips : ℕ := 3

/-- Represents the total distance swum in meters -/
def total_distance : ℝ := 600

/-- Theorem stating that the lane length is correct given the conditions -/
theorem lane_length_correct : 
  lane_length * (2 * round_trips) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_lane_length_correct_l1_166


namespace NUMINAMATH_CALUDE_cauchy_functional_equation_verify_solution_l1_133

/-- A function satisfying the additive Cauchy equation -/
def is_additive (f : ℕ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

/-- A function satisfying f(nk) = n f(k) for all n, k ∈ ℕ -/
def satisfies_property (f : ℕ → ℝ) : Prop :=
  ∀ n k, f (n * k) = n * f k

theorem cauchy_functional_equation (f : ℕ → ℝ) 
  (h_additive : is_additive f) (h_property : satisfies_property f) :
  ∃ a : ℝ, ∀ n : ℕ, f n = a * n := by sorry

theorem verify_solution (a : ℝ) :
  let f : ℕ → ℝ := λ n ↦ a * n
  is_additive f ∧ satisfies_property f := by sorry

end NUMINAMATH_CALUDE_cauchy_functional_equation_verify_solution_l1_133


namespace NUMINAMATH_CALUDE_coffee_table_price_is_330_l1_120

/-- The price of the coffee table in a living room set purchase --/
def coffee_table_price (sofa_price armchair_price total_invoice : ℕ) (num_armchairs : ℕ) : ℕ :=
  total_invoice - (sofa_price + num_armchairs * armchair_price)

/-- Theorem stating the price of the coffee table in the given scenario --/
theorem coffee_table_price_is_330 :
  coffee_table_price 1250 425 2430 2 = 330 := by
  sorry

end NUMINAMATH_CALUDE_coffee_table_price_is_330_l1_120


namespace NUMINAMATH_CALUDE_simplify_product_of_radicals_l1_189

theorem simplify_product_of_radicals (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 120 * x * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_simplify_product_of_radicals_l1_189


namespace NUMINAMATH_CALUDE_original_acid_percentage_l1_186

theorem original_acid_percentage (x y : ℝ) :
  (y / (x + y + 1) = 1 / 5) →
  ((y + 1) / (x + y + 2) = 1 / 3) →
  (y / (x + y) = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_original_acid_percentage_l1_186


namespace NUMINAMATH_CALUDE_f_monotone_increasing_on_negative_l1_143

-- Define the function
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_monotone_increasing_on_negative : 
  MonotoneOn f (Set.Iic 0) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_on_negative_l1_143


namespace NUMINAMATH_CALUDE_product_one_sum_greater_than_inverses_l1_157

theorem product_one_sum_greater_than_inverses
  (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_product : a * b * c = 1) 
  (h_sum : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ 
  (b > 1 ∧ a < 1 ∧ c < 1) ∨ 
  (c > 1 ∧ a < 1 ∧ b < 1) :=
by sorry

end NUMINAMATH_CALUDE_product_one_sum_greater_than_inverses_l1_157


namespace NUMINAMATH_CALUDE_squats_calculation_l1_183

/-- 
Proves that if the number of squats increases by 5 each day for four consecutive days, 
and 45 squats are performed on the fourth day, then 30 squats were performed on the first day.
-/
theorem squats_calculation (initial_squats : ℕ) : 
  (∀ (day : ℕ), day < 4 → initial_squats + 5 * day = initial_squats + day * 5) →
  initial_squats + 5 * 3 = 45 →
  initial_squats = 30 := by
  sorry

end NUMINAMATH_CALUDE_squats_calculation_l1_183


namespace NUMINAMATH_CALUDE_valid_number_count_l1_101

/-- Represents a valid six-digit number configuration --/
structure ValidNumber :=
  (digits : Fin 6 → Fin 6)
  (no_repetition : Function.Injective digits)
  (one_not_at_ends : digits 0 ≠ 1 ∧ digits 5 ≠ 1)
  (one_adjacent_even_pair : ∃! (i : Fin 5), 
    (digits i).val % 2 = 0 ∧ (digits (i + 1)).val % 2 = 0 ∧
    (digits i).val ≠ (digits (i + 1)).val)

/-- The number of valid six-digit numbers --/
def count_valid_numbers : ℕ := sorry

/-- The main theorem stating the count of valid numbers --/
theorem valid_number_count : count_valid_numbers = 288 := by sorry

end NUMINAMATH_CALUDE_valid_number_count_l1_101


namespace NUMINAMATH_CALUDE_complementary_angle_triple_l1_129

theorem complementary_angle_triple (x y : ℝ) : 
  x + y = 90 ∧ x = 3 * y → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_triple_l1_129


namespace NUMINAMATH_CALUDE_third_root_of_polynomial_l1_195

theorem third_root_of_polynomial (a b : ℚ) :
  (∀ x : ℚ, a * x^3 + 2*(a + b) * x^2 + (b - 2*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 61/35) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_polynomial_l1_195


namespace NUMINAMATH_CALUDE_blake_bucket_water_l1_106

theorem blake_bucket_water (poured_out water_left : ℝ) 
  (h1 : poured_out = 0.2)
  (h2 : water_left = 0.6) :
  poured_out + water_left = 0.8 := by
sorry

end NUMINAMATH_CALUDE_blake_bucket_water_l1_106


namespace NUMINAMATH_CALUDE_total_fish_equation_l1_127

/-- The number of fish owned by four friends, given their relative quantities -/
def total_fish (x : ℝ) : ℝ :=
  let max_fish := x
  let sam_fish := 3.25 * max_fish
  let joe_fish := 9.5 * sam_fish
  let harry_fish := 5.5 * joe_fish
  max_fish + sam_fish + joe_fish + harry_fish

/-- Theorem stating that the total number of fish is 204.9375 times the number of fish Max has -/
theorem total_fish_equation (x : ℝ) : total_fish x = 204.9375 * x := by
  sorry

end NUMINAMATH_CALUDE_total_fish_equation_l1_127


namespace NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l1_146

-- Define the right triangle
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Theorem statement
theorem right_triangle_area_and_perimeter :
  ∀ (a b c : ℝ),
  right_triangle a b c →
  a = 36 →
  b = 48 →
  (1/2 * a * b = 864) ∧ (a + b + c = 144) := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l1_146


namespace NUMINAMATH_CALUDE_trig_expression_equals_eight_thirds_l1_153

theorem trig_expression_equals_eight_thirds :
  let sin30 : ℝ := 1/2
  let cos30 : ℝ := Real.sqrt 3 / 2
  (cos30^2 - sin30^2) / (cos30^2 * sin30^2) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_eight_thirds_l1_153


namespace NUMINAMATH_CALUDE_exists_double_application_square_l1_192

theorem exists_double_application_square : 
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 := by sorry

end NUMINAMATH_CALUDE_exists_double_application_square_l1_192


namespace NUMINAMATH_CALUDE_exponential_function_determined_l1_179

def is_exponential (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = a^x

theorem exponential_function_determined (f : ℝ → ℝ) :
  is_exponential f → f 3 = 8 → ∀ x, f x = 2^x := by sorry

end NUMINAMATH_CALUDE_exponential_function_determined_l1_179


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l1_103

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (y : ℕ), y = Nat.lcm x (Nat.lcm 8 12) ∧ y = 120) → 
  x ≤ 120 ∧ ∃ (z : ℕ), z = 120 ∧ Nat.lcm z (Nat.lcm 8 12) = 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l1_103


namespace NUMINAMATH_CALUDE_profit_calculation_l1_100

theorem profit_calculation (cost1 cost2 cost3 : ℝ) (profit_percentage : ℝ) :
  cost1 = 200 →
  cost2 = 300 →
  cost3 = 500 →
  profit_percentage = 0.1 →
  let total_cost := cost1 + cost2 + cost3
  let total_selling_price := total_cost + total_cost * profit_percentage
  total_selling_price = 1100 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l1_100


namespace NUMINAMATH_CALUDE_equal_star_set_eq_four_lines_l1_116

-- Define the operation ⋆
def star (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def equal_star_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

-- Define the four lines
def four_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 + p.2 = 0}

-- Theorem stating the equivalence of the two sets
theorem equal_star_set_eq_four_lines :
  equal_star_set = four_lines := by sorry

end NUMINAMATH_CALUDE_equal_star_set_eq_four_lines_l1_116


namespace NUMINAMATH_CALUDE_rectangle_and_parallelogram_area_l1_160

-- Define the shapes and their properties
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side ^ 2

structure Circle where
  radius : ℝ

structure Rectangle where
  length : ℝ
  breadth : ℝ
  area : ℝ
  area_eq : area = length * breadth

structure Parallelogram where
  base : ℝ
  height : ℝ
  diagonal : ℝ
  area : ℝ
  area_eq : area = base * height

-- Define the problem
def problem (s : Square) (c : Circle) (r : Rectangle) (p : Parallelogram) : Prop :=
  s.area = 3600 ∧
  s.side = c.radius ∧
  r.length = 2/5 * c.radius ∧
  r.breadth = 10 ∧
  r.breadth = 1/2 * p.diagonal ∧
  p.base = 20 * Real.sqrt 3 ∧
  p.height = r.breadth

-- Theorem to prove
theorem rectangle_and_parallelogram_area 
  (s : Square) (c : Circle) (r : Rectangle) (p : Parallelogram) 
  (h : problem s c r p) : 
  r.area = 240 ∧ p.area = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_and_parallelogram_area_l1_160


namespace NUMINAMATH_CALUDE_luggage_per_passenger_l1_140

theorem luggage_per_passenger (total_passengers : ℕ) (total_bags : ℕ) 
  (h1 : total_passengers = 4) (h2 : total_bags = 32) : 
  total_bags / total_passengers = 8 := by
  sorry

end NUMINAMATH_CALUDE_luggage_per_passenger_l1_140


namespace NUMINAMATH_CALUDE_correct_calculation_l1_172

theorem correct_calculation (x : ℝ) : 3 * x - 12 = 60 → (x / 3) + 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1_172


namespace NUMINAMATH_CALUDE_complex_fraction_real_implies_a_negative_one_l1_117

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition that (a+i)/(1-i) is real
def is_real (a : ℝ) : Prop := ∃ (r : ℝ), (a + i) / (1 - i) = r

-- Theorem statement
theorem complex_fraction_real_implies_a_negative_one (a : ℝ) :
  is_real a → a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_real_implies_a_negative_one_l1_117


namespace NUMINAMATH_CALUDE_solution_set_f_geq_neg_two_max_a_for_f_leq_x_minus_a_l1_197

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

-- Theorem for the solution set of f(x) ≥ -2
theorem solution_set_f_geq_neg_two :
  {x : ℝ | f x ≥ -2} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 6} := by sorry

-- Theorem for the maximum value of a
theorem max_a_for_f_leq_x_minus_a :
  ∀ a : ℝ, (∀ x : ℝ, f x ≤ x - a) ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_neg_two_max_a_for_f_leq_x_minus_a_l1_197


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1_105

theorem fractional_equation_solution :
  ∃! x : ℚ, (2 - x) / (x - 3) + 3 = 2 / (3 - x) ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1_105


namespace NUMINAMATH_CALUDE_all_students_visiting_one_student_visiting_l1_168

-- Define the probabilities of each student visiting
def prob_A : ℚ := 2/3
def prob_B : ℚ := 1/4
def prob_C : ℚ := 2/5

-- Theorem for the probability of all three students visiting
theorem all_students_visiting : 
  prob_A * prob_B * prob_C = 1/15 := by
  sorry

-- Theorem for the probability of exactly one student visiting
theorem one_student_visiting : 
  prob_A * (1 - prob_B) * (1 - prob_C) + 
  (1 - prob_A) * prob_B * (1 - prob_C) + 
  (1 - prob_A) * (1 - prob_B) * prob_C = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_all_students_visiting_one_student_visiting_l1_168


namespace NUMINAMATH_CALUDE_sams_books_l1_125

theorem sams_books (joan_books : ℕ) (total_books : ℕ) (h1 : joan_books = 102) (h2 : total_books = 212) :
  total_books - joan_books = 110 := by
  sorry

end NUMINAMATH_CALUDE_sams_books_l1_125
