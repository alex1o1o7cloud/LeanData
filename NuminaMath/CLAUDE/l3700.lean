import Mathlib

namespace equation_solution_l3700_370029

theorem equation_solution : 
  ∃ x : ℝ, (64 + 5 * 12 / (x / 3) = 65) ∧ (x = 180) := by
  sorry

end equation_solution_l3700_370029


namespace player_one_winning_strategy_l3700_370074

-- Define the chessboard
def Chessboard : Type := Fin 8 × Fin 8

-- Define the distance between two points on the chessboard
def distance (p1 p2 : Chessboard) : ℝ := sorry

-- Define a valid move
def validMove (prev curr next : Chessboard) : Prop :=
  distance curr next > distance prev curr

-- Define the game state
structure GameState :=
  (position : Chessboard)
  (lastMove : Option Chessboard)
  (playerTurn : Bool)  -- true for Player One, false for Player Two

-- Define the winning condition for Player One
def playerOneWins (game : GameState) : Prop :=
  ∀ move : Chessboard, ¬validMove (Option.getD game.lastMove game.position) game.position move

-- Theorem: Player One has a winning strategy
theorem player_one_winning_strategy :
  ∃ (strategy : GameState → Chessboard),
    ∀ (game : GameState),
      game.playerTurn → 
      validMove (Option.getD game.lastMove game.position) game.position (strategy game) ∧
      playerOneWins {
        position := strategy game,
        lastMove := some game.position,
        playerTurn := false
      } := sorry

end player_one_winning_strategy_l3700_370074


namespace quadratic_roots_relation_l3700_370063

theorem quadratic_roots_relation (m n p : ℝ) 
  (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0)
  (h : ∃ (r₁ r₂ : ℝ), (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧ 
                      (3*r₁ + 3*r₂ = -m ∧ 9*r₁*r₂ = n)) :
  n / p = -27 := by
sorry

end quadratic_roots_relation_l3700_370063


namespace specific_pairing_probability_l3700_370058

/-- Represents a classroom with male and female students -/
structure Classroom where
  female_count : ℕ
  male_count : ℕ

/-- Represents a pairing of students -/
structure Pairing where
  classroom : Classroom
  is_opposite_gender : Bool

/-- Calculates the probability of a specific pairing -/
def probability_of_specific_pairing (c : Classroom) (p : Pairing) : ℚ :=
  1 / c.male_count

/-- Theorem: The probability of a specific female-male pairing in a classroom
    with 20 female students and 18 male students is 1/18 -/
theorem specific_pairing_probability :
  let c : Classroom := { female_count := 20, male_count := 18 }
  let p : Pairing := { classroom := c, is_opposite_gender := true }
  probability_of_specific_pairing c p = 1 / 18 := by
    sorry

end specific_pairing_probability_l3700_370058


namespace nested_root_simplification_l3700_370056

theorem nested_root_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x * (x^3)^(1/4) := by
  sorry

end nested_root_simplification_l3700_370056


namespace range_of_a_l3700_370059

theorem range_of_a : ∀ a : ℝ, 
  (∀ x : ℝ, |x - a| < 1 ↔ (1/2 : ℝ) < x ∧ x < (3/2 : ℝ)) →
  ((1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ)) :=
by sorry

end range_of_a_l3700_370059


namespace initial_files_count_l3700_370027

theorem initial_files_count (organized_morning : ℕ) (to_organize_afternoon : ℕ) (missing : ℕ) :
  organized_morning = to_organize_afternoon ∧
  to_organize_afternoon = missing ∧
  to_organize_afternoon = 15 →
  2 * organized_morning + to_organize_afternoon + missing = 60 :=
by sorry

end initial_files_count_l3700_370027


namespace shortest_assembly_time_is_13_l3700_370061

/-- Represents the time taken for each step in the assembly process -/
structure AssemblyTimes where
  ac : ℕ -- Time from A to C
  cd : ℕ -- Time from C to D
  be : ℕ -- Time from B to E
  ed : ℕ -- Time from E to D
  df : ℕ -- Time from D to F

/-- Calculates the shortest assembly time given the times for each step -/
def shortestAssemblyTime (times : AssemblyTimes) : ℕ :=
  max (times.ac + times.cd) (times.be + times.ed + times.df)

/-- Theorem stating that for the given assembly times, the shortest assembly time is 13 hours -/
theorem shortest_assembly_time_is_13 :
  let times : AssemblyTimes := {
    ac := 3,
    cd := 4,
    be := 3,
    ed := 4,
    df := 2
  }
  shortestAssemblyTime times = 13 := by
  sorry

end shortest_assembly_time_is_13_l3700_370061


namespace divisor_sum_840_l3700_370057

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem divisor_sum_840 (i j : ℕ) (h : i > 0 ∧ j > 0) :
  sum_of_divisors (2^i * 3^j) = 840 → i + j = 5 := by
  sorry

end divisor_sum_840_l3700_370057


namespace imaginary_unit_powers_l3700_370006

theorem imaginary_unit_powers (i : ℂ) : i^2 = -1 → i^50 + i^105 = -1 + i := by
  sorry

end imaginary_unit_powers_l3700_370006


namespace largest_even_digit_multiple_of_5_l3700_370071

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

theorem largest_even_digit_multiple_of_5 :
  ∃ (n : ℕ), n = 6880 ∧
  has_only_even_digits n ∧
  n < 8000 ∧
  n % 5 = 0 ∧
  ∀ m : ℕ, has_only_even_digits m → m < 8000 → m % 5 = 0 → m ≤ n :=
by sorry

end largest_even_digit_multiple_of_5_l3700_370071


namespace sqrt_twelve_div_sqrt_three_equals_two_l3700_370060

theorem sqrt_twelve_div_sqrt_three_equals_two : Real.sqrt 12 / Real.sqrt 3 = 2 := by
  sorry

end sqrt_twelve_div_sqrt_three_equals_two_l3700_370060


namespace house_store_transaction_loss_l3700_370066

theorem house_store_transaction_loss (house_price store_price : ℝ) : 
  house_price * (1 - 0.2) = 12000 →
  store_price * (1 + 0.2) = 12000 →
  house_price + store_price - 2 * 12000 = 1000 := by
  sorry

end house_store_transaction_loss_l3700_370066


namespace distribute_5_2_l3700_370028

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_5_2 : distribute 5 2 = 3 := by sorry

end distribute_5_2_l3700_370028


namespace polynomials_equal_sum_of_squares_is_954_l3700_370072

/-- The original polynomial expression -/
def original_polynomial (x : ℝ) : ℝ := 5 * (x^3 - 3*x^2 + 4) - 8 * (2*x^4 - x^3 + x)

/-- The fully simplified polynomial -/
def simplified_polynomial (x : ℝ) : ℝ := -16*x^4 - 3*x^3 - 15*x^2 + 8*x + 20

/-- Theorem stating that the original and simplified polynomials are equal -/
theorem polynomials_equal : ∀ x : ℝ, original_polynomial x = simplified_polynomial x := by sorry

/-- The sum of squares of coefficients of the simplified polynomial -/
def sum_of_squares_of_coefficients : ℕ := 16^2 + 3^2 + 15^2 + 8^2 + 20^2

/-- Theorem stating that the sum of squares of coefficients is 954 -/
theorem sum_of_squares_is_954 : sum_of_squares_of_coefficients = 954 := by sorry

end polynomials_equal_sum_of_squares_is_954_l3700_370072


namespace expression_equals_four_l3700_370091

theorem expression_equals_four :
  let a := 7 + Real.sqrt 48
  let b := 7 - Real.sqrt 48
  (a^2023 + b^2023)^2 - (a^2023 - b^2023)^2 = 4 := by
  sorry

end expression_equals_four_l3700_370091


namespace sqrt_x_div_sqrt_y_l3700_370097

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (29*x)/(53*y)) :
  Real.sqrt x / Real.sqrt y = 91/42 := by
  sorry

end sqrt_x_div_sqrt_y_l3700_370097


namespace number_of_observations_l3700_370026

theorem number_of_observations (initial_mean old_value new_value new_mean : ℝ) : 
  initial_mean = 36 →
  old_value = 40 →
  new_value = 25 →
  new_mean = 34.9 →
  ∃ (n : ℕ), (n : ℝ) * initial_mean - old_value + new_value = (n : ℝ) * new_mean ∧ 
              n = 14 :=
by sorry

end number_of_observations_l3700_370026


namespace inequality_proof_l3700_370025

theorem inequality_proof (x a : ℝ) (h : |x - a| < 1) : 
  let f := fun (t : ℝ) => t^2 - 2*t
  |f x - f a| < 2*|a| + 3 := by
sorry

end inequality_proof_l3700_370025


namespace greatest_four_digit_divisible_by_55_and_11_l3700_370037

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_55_and_11 :
  ∃ (m : ℕ), is_four_digit m ∧
             m % 55 = 0 ∧
             (reverse_digits m) % 55 = 0 ∧
             m % 11 = 0 ∧
             (∀ (n : ℕ), is_four_digit n →
                         n % 55 = 0 →
                         (reverse_digits n) % 55 = 0 →
                         n % 11 = 0 →
                         n ≤ m) ∧
             m = 5445 :=
sorry

end greatest_four_digit_divisible_by_55_and_11_l3700_370037


namespace largest_number_l3700_370024

def a : ℚ := 883/1000
def b : ℚ := 8839/10000
def c : ℚ := 88/100
def d : ℚ := 839/1000
def e : ℚ := 889/1000

theorem largest_number : b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end largest_number_l3700_370024


namespace smallest_arithmetic_mean_of_nine_consecutive_naturals_l3700_370070

theorem smallest_arithmetic_mean_of_nine_consecutive_naturals (n : ℕ) : 
  (∀ k : ℕ, k ∈ Finset.range 9 → (n + k) > 0) →
  (((List.range 9).map (λ k => n + k)).prod) % 1111 = 0 →
  (((List.range 9).map (λ k => n + k)).sum / 9 : ℚ) ≥ 97 :=
by sorry

end smallest_arithmetic_mean_of_nine_consecutive_naturals_l3700_370070


namespace divisibility_condition_l3700_370032

theorem divisibility_condition (x y z k : ℤ) :
  (∃ q : ℤ, x^3 + y^3 + z^3 + k*x*y*z = (x + y + z) * q) ↔ k = -3 := by
  sorry

end divisibility_condition_l3700_370032


namespace quadratic_equation_condition_l3700_370077

/-- The equation (m-4)x^|m-2| + 2x - 5 = 0 is quadratic if and only if m = 0 -/
theorem quadratic_equation_condition (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, (m - 4) * x^(|m - 2|) + 2*x - 5 = a*x^2 + b*x + c) ↔ m = 0 :=
sorry

end quadratic_equation_condition_l3700_370077


namespace digital_earth_prospects_l3700_370030

/-- Represents the prospects of digital Earth applications -/
structure DigitalEarthProspects where
  spatialLab : Bool  -- Provides a digital spatial laboratory
  decisionMaking : Bool  -- Government decision-making can fully rely on it
  urbanManagement : Bool  -- Provides a basis for urban management
  predictable : Bool  -- The development is predictable

/-- The correct prospects of digital Earth applications -/
def correctProspects : DigitalEarthProspects :=
  { spatialLab := true
    decisionMaking := false
    urbanManagement := true
    predictable := false }

/-- Theorem stating the correct prospects of digital Earth applications -/
theorem digital_earth_prospects :
  (correctProspects.spatialLab = true) ∧
  (correctProspects.urbanManagement = true) ∧
  (correctProspects.decisionMaking = false) ∧
  (correctProspects.predictable = false) := by
  sorry


end digital_earth_prospects_l3700_370030


namespace first_player_winning_strategy_l3700_370008

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents the chessboard game -/
structure ChessboardGame :=
  (m : Nat) (n : Nat)

/-- Checks if a position is winning for the current player -/
def isWinningPosition (game : ChessboardGame) (pos : Position) : Prop :=
  pos.x ≠ pos.y

/-- Checks if the first player has a winning strategy -/
def firstPlayerWins (game : ChessboardGame) : Prop :=
  isWinningPosition game ⟨game.m - 1, game.n - 1⟩

/-- The main theorem: The first player wins iff m ≠ n -/
theorem first_player_winning_strategy (game : ChessboardGame) :
  firstPlayerWins game ↔ game.m ≠ game.n :=
sorry

end first_player_winning_strategy_l3700_370008


namespace first_term_of_geometric_sequence_l3700_370022

def geometric_sequence (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r^(n - 1)

theorem first_term_of_geometric_sequence 
  (a : ℝ) (r : ℝ) (h1 : r ≠ 0) (h2 : r ≠ 1) :
  (geometric_sequence a r 1 + geometric_sequence a r 2 + 
   geometric_sequence a r 3 + geometric_sequence a r 4 = 240) →
  (geometric_sequence a r 2 + geometric_sequence a r 4 = 180) →
  a = 6 := by
sorry

end first_term_of_geometric_sequence_l3700_370022


namespace possible_m_values_l3700_370004

theorem possible_m_values (x m a b : ℤ) : 
  (∀ x, x^2 + m*x - 14 = (x + a) * (x + b)) → 
  (m = 5 ∨ m = -5 ∨ m = 13 ∨ m = -13) :=
sorry

end possible_m_values_l3700_370004


namespace digit_sum_equation_l3700_370075

theorem digit_sum_equation (a : ℕ) : a * 1000 + a * 998 + a * 999 = 22997 → a = 7 := by
  sorry

end digit_sum_equation_l3700_370075


namespace container_emptying_possible_l3700_370048

/-- Represents a container with water -/
structure Container where
  water : ℕ

/-- Represents the state of three containers -/
structure ContainerState where
  a : Container
  b : Container
  c : Container

/-- Represents a transfer of water between containers -/
inductive Transfer : ContainerState → ContainerState → Prop where
  | ab (s : ContainerState) : 
      Transfer s ⟨⟨s.a.water + s.b.water⟩, ⟨0⟩, s.c⟩
  | ac (s : ContainerState) : 
      Transfer s ⟨⟨s.a.water + s.c.water⟩, s.b, ⟨0⟩⟩
  | ba (s : ContainerState) : 
      Transfer s ⟨⟨0⟩, ⟨s.a.water + s.b.water⟩, s.c⟩
  | bc (s : ContainerState) : 
      Transfer s ⟨s.a, ⟨s.b.water + s.c.water⟩, ⟨0⟩⟩
  | ca (s : ContainerState) : 
      Transfer s ⟨⟨0⟩, s.b, ⟨s.a.water + s.c.water⟩⟩
  | cb (s : ContainerState) : 
      Transfer s ⟨s.a, ⟨0⟩, ⟨s.b.water + s.c.water⟩⟩

/-- Represents a sequence of transfers -/
def TransferSeq := List (ContainerState → ContainerState)

/-- Applies a sequence of transfers to an initial state -/
def applyTransfers (initial : ContainerState) (seq : TransferSeq) : ContainerState :=
  seq.foldl (fun state transfer => transfer state) initial

/-- Predicate to check if a container is empty -/
def isEmptyContainer (c : Container) : Prop := c.water = 0

/-- Predicate to check if any container in the state is empty -/
def hasEmptyContainer (s : ContainerState) : Prop :=
  isEmptyContainer s.a ∨ isEmptyContainer s.b ∨ isEmptyContainer s.c

/-- The main theorem to prove -/
theorem container_emptying_possible (initial : ContainerState) : 
  ∃ (seq : TransferSeq), hasEmptyContainer (applyTransfers initial seq) := by
  sorry

end container_emptying_possible_l3700_370048


namespace cheetah_catch_fox_l3700_370042

/-- Represents the cheetah's speed in meters per second -/
def cheetah_speed : ℝ := 4

/-- Represents the fox's speed in meters per second -/
def fox_speed : ℝ := 3

/-- Represents the initial distance between the cheetah and the fox in meters -/
def initial_distance : ℝ := 30

/-- Theorem stating that the cheetah will catch the fox after running 120 meters -/
theorem cheetah_catch_fox : 
  cheetah_speed * (initial_distance / (cheetah_speed - fox_speed)) = 120 :=
sorry

end cheetah_catch_fox_l3700_370042


namespace vanya_masha_speed_ratio_l3700_370086

/-- Represents the scenario of Vanya and Masha's journey to school -/
structure SchoolJourney where
  d : ℝ  -- Total distance from home to school
  vanya_speed : ℝ  -- Vanya's speed
  masha_speed : ℝ  -- Masha's speed

/-- The theorem stating the relationship between Vanya and Masha's speeds -/
theorem vanya_masha_speed_ratio (journey : SchoolJourney) :
  journey.d > 0 →  -- Ensure the distance is positive
  (2/3 * journey.d) / journey.vanya_speed = (1/6 * journey.d) / journey.masha_speed →  -- Condition from overtaking point
  (1/2 * journey.d) / journey.masha_speed = journey.d / journey.vanya_speed →  -- Condition when Vanya reaches school
  journey.vanya_speed / journey.masha_speed = 4 := by
  sorry

end vanya_masha_speed_ratio_l3700_370086


namespace first_agency_daily_charge_is_correct_l3700_370084

/-- The daily charge of the first car rental agency -/
def first_agency_daily_charge : ℝ := 20.25

/-- The per-mile charge of the first car rental agency -/
def first_agency_mile_charge : ℝ := 0.14

/-- The daily charge of the second car rental agency -/
def second_agency_daily_charge : ℝ := 18.25

/-- The per-mile charge of the second car rental agency -/
def second_agency_mile_charge : ℝ := 0.22

/-- The number of miles at which the costs become equal -/
def miles_equal_cost : ℝ := 25

theorem first_agency_daily_charge_is_correct :
  first_agency_daily_charge + first_agency_mile_charge * miles_equal_cost =
  second_agency_daily_charge + second_agency_mile_charge * miles_equal_cost :=
by sorry

end first_agency_daily_charge_is_correct_l3700_370084


namespace segment_length_implies_product_l3700_370045

/-- Given that the length of the segment between the points (3a, 2a-5) and (5, 0) is 3√10 units,
    prove that the product of all possible values of a is -40/13. -/
theorem segment_length_implies_product (a : ℝ) : 
  (((3*a - 5)^2 + (2*a - 5)^2) = 90) → 
  (∃ b : ℝ, (a = b ∨ a = -8/13) ∧ a * b = -40/13) :=
by sorry

end segment_length_implies_product_l3700_370045


namespace pizza_consumption_order_l3700_370085

-- Define the siblings
inductive Sibling : Type
| Emily : Sibling
| Sam : Sibling
| Nora : Sibling
| Oliver : Sibling
| Jack : Sibling

-- Define the pizza consumption for each sibling
def pizza_consumption (s : Sibling) : Rat :=
  match s with
  | Sibling.Emily => 1/6
  | Sibling.Sam => 1/4
  | Sibling.Nora => 1/3
  | Sibling.Oliver => 1/8
  | Sibling.Jack => 1 - (1/6 + 1/4 + 1/3 + 1/8)

-- Define a function to compare pizza consumption
def consumes_more (s1 s2 : Sibling) : Prop :=
  pizza_consumption s1 > pizza_consumption s2

-- State the theorem
theorem pizza_consumption_order :
  consumes_more Sibling.Nora Sibling.Sam ∧
  consumes_more Sibling.Sam Sibling.Emily ∧
  consumes_more Sibling.Emily Sibling.Jack ∧
  consumes_more Sibling.Jack Sibling.Oliver :=
by sorry

end pizza_consumption_order_l3700_370085


namespace a_neg_sufficient_a_neg_not_necessary_a_neg_sufficient_not_necessary_l3700_370068

/-- Represents a quadratic equation ax^2 + 2x + 1 = 0 -/
structure QuadraticEquation (a : ℝ) where
  eq : ∀ x, a * x^2 + 2 * x + 1 = 0

/-- Predicate for an equation having at least one negative root -/
def has_negative_root {a : ℝ} (eq : QuadraticEquation a) : Prop :=
  ∃ x, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

/-- Statement that 'a < 0' is a sufficient condition -/
theorem a_neg_sufficient {a : ℝ} (h : a < 0) : 
  ∃ (eq : QuadraticEquation a), has_negative_root eq :=
sorry

/-- Statement that 'a < 0' is not a necessary condition -/
theorem a_neg_not_necessary : 
  ∃ a, ¬(a < 0) ∧ ∃ (eq : QuadraticEquation a), has_negative_root eq :=
sorry

/-- Main theorem stating that 'a < 0' is sufficient but not necessary -/
theorem a_neg_sufficient_not_necessary : 
  (∀ a, a < 0 → ∃ (eq : QuadraticEquation a), has_negative_root eq) ∧
  (∃ a, ¬(a < 0) ∧ ∃ (eq : QuadraticEquation a), has_negative_root eq) :=
sorry

end a_neg_sufficient_a_neg_not_necessary_a_neg_sufficient_not_necessary_l3700_370068


namespace max_intersections_15_10_l3700_370053

/-- The maximum number of intersection points for segments connecting points on x and y axes -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersections for 15 x-axis points and 10 y-axis points -/
theorem max_intersections_15_10 :
  max_intersections 15 10 = 4725 := by
  sorry

end max_intersections_15_10_l3700_370053


namespace barbell_cost_increase_l3700_370035

theorem barbell_cost_increase (old_cost new_cost : ℝ) (h1 : old_cost = 250) (h2 : new_cost = 325) :
  (new_cost - old_cost) / old_cost * 100 = 30 := by
  sorry

end barbell_cost_increase_l3700_370035


namespace problem_solution_l3700_370012

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 2 - t) 
  (h2 : y = 4 * t + 7) 
  (h3 : x = -3) : 
  y = 27 := by
  sorry

end problem_solution_l3700_370012


namespace cheryl_different_colors_probability_l3700_370088

/-- Represents the number of marbles of each color in the box -/
def initial_marbles : Nat := 2

/-- Represents the total number of colors -/
def total_colors : Nat := 4

/-- Represents the total number of marbles in the box -/
def total_marbles : Nat := initial_marbles * total_colors

/-- Represents the number of marbles each person draws -/
def marbles_drawn : Nat := 2

/-- Calculates the probability of Cheryl not getting two marbles of the same color -/
theorem cheryl_different_colors_probability :
  let total_outcomes := (total_marbles.choose marbles_drawn) * 
                        ((total_marbles - marbles_drawn).choose marbles_drawn) * 
                        ((total_marbles - 2*marbles_drawn).choose marbles_drawn)
  let favorable_outcomes := (total_colors.choose marbles_drawn) * 
                            ((total_marbles - marbles_drawn).choose marbles_drawn) * 
                            ((total_marbles - 2*marbles_drawn).choose marbles_drawn)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 12 := by
  sorry

#eval initial_marbles -- 2
#eval total_colors -- 4
#eval total_marbles -- 8
#eval marbles_drawn -- 2

end cheryl_different_colors_probability_l3700_370088


namespace sugar_price_inconsistency_l3700_370098

/-- Represents the price and consumption change of sugar -/
structure SugarPriceChange where
  p₀ : ℝ  -- Initial price
  p₁ : ℝ  -- New price
  r : ℝ   -- Reduction in consumption (as a decimal)

/-- Checks if the given sugar price change is consistent -/
def is_consistent (s : SugarPriceChange) : Prop :=
  s.r = (s.p₁ - s.p₀) / s.p₁

/-- Theorem stating that the given conditions are inconsistent -/
theorem sugar_price_inconsistency :
  ¬ is_consistent ⟨3, 5, 0.4⟩ := by
  sorry

end sugar_price_inconsistency_l3700_370098


namespace unique_function_satisfying_condition_l3700_370043

theorem unique_function_satisfying_condition :
  ∀ f : ℕ → ℕ,
    (f 1 > 0) →
    (∀ m n : ℕ, f (m^2 + 3*n^2) = (f m)^2 + 3*(f n)^2) →
    (∀ n : ℕ, f n = n) := by
  sorry

end unique_function_satisfying_condition_l3700_370043


namespace tea_price_correct_l3700_370067

/-- Represents the prices and quantities of tea in two purchases -/
structure TeaPurchases where
  first_quantity_A : ℕ
  first_quantity_B : ℕ
  first_total_cost : ℕ
  second_quantity_A : ℕ
  second_quantity_B : ℕ
  second_total_cost : ℕ
  price_increase : ℚ

/-- The solution to the tea pricing problem -/
def tea_price_solution (tp : TeaPurchases) : ℚ × ℚ :=
  (100, 200)

/-- Theorem stating that the given solution is correct for the specified tea purchases -/
theorem tea_price_correct (tp : TeaPurchases) 
  (h1 : tp.first_quantity_A = 30)
  (h2 : tp.first_quantity_B = 20)
  (h3 : tp.first_total_cost = 7000)
  (h4 : tp.second_quantity_A = 20)
  (h5 : tp.second_quantity_B = 15)
  (h6 : tp.second_total_cost = 6000)
  (h7 : tp.price_increase = 1/5) : 
  let (price_A, price_B) := tea_price_solution tp
  (tp.first_quantity_A : ℚ) * price_A + (tp.first_quantity_B : ℚ) * price_B = tp.first_total_cost ∧
  (tp.second_quantity_A : ℚ) * price_A * (1 + tp.price_increase) + 
  (tp.second_quantity_B : ℚ) * price_B * (1 + tp.price_increase) = tp.second_total_cost :=
by
  sorry

#check tea_price_correct

end tea_price_correct_l3700_370067


namespace distance_on_quadratic_curve_l3700_370018

/-- The distance between two points on a quadratic curve. -/
theorem distance_on_quadratic_curve (m n p x₁ x₂ : ℝ) :
  let y₁ := m * x₁^2 + n * x₁ + p
  let y₂ := m * x₂^2 + n * x₂ + p
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = (x₂ - x₁)^2 * (1 + m^2 * (x₂ + x₁)^2 + n^2) :=
by sorry

end distance_on_quadratic_curve_l3700_370018


namespace rectangle_circle_overlap_area_l3700_370090

/-- The area of overlap between a rectangle and a circle with shared center -/
theorem rectangle_circle_overlap_area 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_radius : ℝ) 
  (h1 : rectangle_width = 8) 
  (h2 : rectangle_height = 2 * Real.sqrt 2) 
  (h3 : circle_radius = 2) : 
  ∃ (overlap_area : ℝ), overlap_area = 2 * Real.pi + 4 := by
  sorry

end rectangle_circle_overlap_area_l3700_370090


namespace production_plan_equation_l3700_370064

/-- Represents a factory's production plan -/
structure ProductionPlan where
  original_days : ℕ
  original_parts_per_day : ℕ
  new_days : ℕ
  additional_parts_per_day : ℕ
  extra_parts : ℕ

/-- The equation holds for the given production plan -/
def equation_holds (plan : ProductionPlan) : Prop :=
  plan.original_days * plan.original_parts_per_day = 
  plan.new_days * (plan.original_parts_per_day + plan.additional_parts_per_day) - plan.extra_parts

theorem production_plan_equation (plan : ProductionPlan) 
  (h1 : plan.original_days = 20)
  (h2 : plan.new_days = 15)
  (h3 : plan.additional_parts_per_day = 4)
  (h4 : plan.extra_parts = 10) :
  equation_holds plan := by
  sorry

#check production_plan_equation

end production_plan_equation_l3700_370064


namespace contradiction_assumption_l3700_370094

theorem contradiction_assumption (a b c : ℝ) :
  (¬ (a > 0 ∨ b > 0 ∨ c > 0)) ↔ (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0) :=
by sorry

end contradiction_assumption_l3700_370094


namespace difference_of_squares_l3700_370093

theorem difference_of_squares (a : ℝ) : a^2 - 4 = (a + 2) * (a - 2) := by
  sorry

end difference_of_squares_l3700_370093


namespace band_second_set_songs_l3700_370049

/-- Proves the number of songs played in the second set given the band's repertoire and performance details -/
theorem band_second_set_songs 
  (total_songs : ℕ) 
  (first_set : ℕ) 
  (encore : ℕ) 
  (avg_third_fourth : ℕ) 
  (h1 : total_songs = 30)
  (h2 : first_set = 5)
  (h3 : encore = 2)
  (h4 : avg_third_fourth = 8) :
  ∃ (second_set : ℕ), 
    second_set = 7 ∧ 
    (total_songs - first_set - second_set - encore) / 2 = avg_third_fourth :=
by sorry

end band_second_set_songs_l3700_370049


namespace lollipop_sugar_calculation_l3700_370040

def chocolate_bars : ℕ := 14
def sugar_per_bar : ℕ := 10
def total_sugar : ℕ := 177

def sugar_in_lollipop : ℕ := total_sugar - (chocolate_bars * sugar_per_bar)

theorem lollipop_sugar_calculation :
  sugar_in_lollipop = 37 := by
  sorry

end lollipop_sugar_calculation_l3700_370040


namespace inequality_proof_l3700_370003

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
by sorry

end inequality_proof_l3700_370003


namespace square_field_area_l3700_370002

/-- The area of a square field with side length 25 meters is 625 square meters. -/
theorem square_field_area : 
  let side_length : ℝ := 25
  let area : ℝ := side_length * side_length
  area = 625 := by sorry

end square_field_area_l3700_370002


namespace bucket_full_weight_bucket_full_weight_proof_l3700_370096

/-- Given a bucket with known weights at different fill levels, 
    calculate the weight when completely full. -/
theorem bucket_full_weight (c d : ℝ) : ℝ :=
  let three_fourths_weight := c
  let one_third_weight := d
  let full_weight := (8/5 : ℝ) * c - (3/5 : ℝ) * d
  full_weight

/-- Prove that the calculated full weight is correct -/
theorem bucket_full_weight_proof (c d : ℝ) :
  bucket_full_weight c d = (8/5 : ℝ) * c - (3/5 : ℝ) * d := by
  sorry

end bucket_full_weight_bucket_full_weight_proof_l3700_370096


namespace average_age_problem_l3700_370080

theorem average_age_problem (total_students : Nat) (average_age : ℝ) 
  (group1_students : Nat) (group1_average : ℝ) (student12_age : ℝ) :
  total_students = 16 →
  average_age = 16 →
  group1_students = 5 →
  group1_average = 14 →
  student12_age = 42 →
  let remaining_students := total_students - group1_students - 1
  let total_age := average_age * total_students
  let group1_total_age := group1_average * group1_students
  let remaining_total_age := total_age - group1_total_age - student12_age
  remaining_total_age / remaining_students = 16 := by
  sorry

end average_age_problem_l3700_370080


namespace line_plane_relationships_l3700_370011

-- Define the basic structures
variable (α : Plane) (l m : Line)

-- Define the relationships
def not_contained_in (l : Line) (α : Plane) : Prop := sorry
def contained_in (m : Line) (α : Plane) : Prop := sorry
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (α : Plane) : Prop := sorry
def perpendicular_lines (l m : Line) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- State the theorem
theorem line_plane_relationships :
  not_contained_in l α →
  contained_in m α →
  ((perpendicular l α → perpendicular_lines l m) ∧
   (parallel_lines l m → parallel_line_plane l α)) :=
by sorry

end line_plane_relationships_l3700_370011


namespace rectangular_field_area_l3700_370099

/-- Represents a rectangular field with a width that is one-third of its length -/
structure RectangularField where
  length : ℝ
  width : ℝ
  width_is_third_of_length : width = length / 3
  perimeter_is_72 : 2 * (length + width) = 72

/-- The area of a rectangular field with the given conditions is 243 square meters -/
theorem rectangular_field_area (field : RectangularField) : field.length * field.width = 243 := by
  sorry

end rectangular_field_area_l3700_370099


namespace even_decreasing_implies_inequality_l3700_370020

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem even_decreasing_implies_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) (h_decr : decreasing_on_nonneg f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end even_decreasing_implies_inequality_l3700_370020


namespace line_through_point_l3700_370007

theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, b * x + (b + 2) * y = b - 1 → x = 3 ∧ y = -5) → b = -3 :=
by sorry

end line_through_point_l3700_370007


namespace abs_plus_square_zero_implies_product_l3700_370092

theorem abs_plus_square_zero_implies_product (a b : ℝ) : 
  |a - 1| + (b + 2)^2 = 0 → a * b^a = -2 := by
sorry

end abs_plus_square_zero_implies_product_l3700_370092


namespace victory_chain_exists_l3700_370041

/-- Represents a chess player in the tournament -/
structure Player :=
  (id : Nat)

/-- Represents the result of a match between two players -/
inductive MatchResult
  | Win
  | Loss
  | Draw

/-- The chess tournament with 2016 players -/
def Tournament := Fin 2016 → Player

/-- The result of a match between two players -/
def matchResult (p1 p2 : Player) : MatchResult := sorry

/-- Condition: If players A and B tie, then every other player loses to either A or B -/
def tieCondition (t : Tournament) : Prop :=
  ∀ a b : Player, matchResult a b = MatchResult.Draw →
    ∀ c : Player, c ≠ a ∧ c ≠ b →
      matchResult c a = MatchResult.Loss ∨ matchResult c b = MatchResult.Loss

/-- There are at least two draws in the tournament -/
def atLeastTwoDraws (t : Tournament) : Prop :=
  ∃ a b c d : Player, a ≠ b ∧ c ≠ d ∧ matchResult a b = MatchResult.Draw ∧ matchResult c d = MatchResult.Draw

/-- A permutation of players where each player defeats the next -/
def victoryChain (t : Tournament) (p : Fin 2016 → Fin 2016) : Prop :=
  ∀ i : Fin 2015, matchResult (t (p i)) (t (p (i + 1))) = MatchResult.Win

/-- Main theorem: If there are at least two draws and the tie condition holds,
    then there exists a permutation where each player defeats the next -/
theorem victory_chain_exists (t : Tournament)
  (h1 : tieCondition t) (h2 : atLeastTwoDraws t) :
  ∃ p : Fin 2016 → Fin 2016, Function.Bijective p ∧ victoryChain t p := by
  sorry

end victory_chain_exists_l3700_370041


namespace same_solution_equations_l3700_370000

theorem same_solution_equations (x b : ℝ) : 
  (2 * x + 7 = 3) ∧ (b * x - 10 = -2) → b = -4 := by
  sorry

end same_solution_equations_l3700_370000


namespace gcd_of_all_P_is_one_l3700_370087

-- Define P as a function of n, where n represents the first of the three consecutive even integers
def P (n : ℕ) : ℕ := 2 * n * (2 * n + 2) * (2 * n + 4) + 2

-- Theorem stating that the greatest common divisor of all P(n) is 1
theorem gcd_of_all_P_is_one : ∃ (d : ℕ), d > 0 ∧ (∀ (n : ℕ), n > 0 → d ∣ P n) → d = 1 := by
  sorry

end gcd_of_all_P_is_one_l3700_370087


namespace combined_girls_avg_is_88_l3700_370014

/-- Represents a high school with average scores for boys, girls, and combined -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two schools -/
structure CombinedSchools where
  school1 : School
  school2 : School
  combined_boys_avg : ℝ

/-- Calculates the combined average score for girls across two schools -/
def combined_girls_avg (schools : CombinedSchools) : ℝ :=
  sorry

/-- The theorem stating that the combined average score for girls is 88 -/
theorem combined_girls_avg_is_88 (schools : CombinedSchools) 
  (h1 : schools.school1 = { boys_avg := 74, girls_avg := 77, combined_avg := 75 })
  (h2 : schools.school2 = { boys_avg := 83, girls_avg := 94, combined_avg := 90 })
  (h3 : schools.combined_boys_avg = 80) :
  combined_girls_avg schools = 88 := by
  sorry

end combined_girls_avg_is_88_l3700_370014


namespace sin_30_plus_cos_60_l3700_370083

theorem sin_30_plus_cos_60 : Real.sin (30 * π / 180) + Real.cos (60 * π / 180) = 1 := by
  sorry

end sin_30_plus_cos_60_l3700_370083


namespace cos_equality_with_period_l3700_370047

theorem cos_equality_with_period (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → 
  Real.cos (n * π / 180) = Real.cos (845 * π / 180) → 
  n = 125 := by
sorry

end cos_equality_with_period_l3700_370047


namespace same_solution_equations_l3700_370036

theorem same_solution_equations (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 8 = 5 ∧ c * x - 7 = 1) → c = -8 := by
  sorry

end same_solution_equations_l3700_370036


namespace remainders_inequality_l3700_370044

theorem remainders_inequality (X Y M A B s t u : ℕ) : 
  X > Y →
  X % M = A →
  Y % M = B →
  X = Y + 8 →
  (X^2) % M = s →
  (Y^2) % M = t →
  ((A*B)^2) % M = u →
  (s ≠ t ∧ t ≠ u ∧ s ≠ u) :=
by sorry

end remainders_inequality_l3700_370044


namespace point_inside_circle_implies_a_range_l3700_370017

theorem point_inside_circle_implies_a_range (a : ℝ) :
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end point_inside_circle_implies_a_range_l3700_370017


namespace matrix_not_invertible_l3700_370019

def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2 + x, 9; 4 - x, 5]

theorem matrix_not_invertible (x : ℚ) :
  ¬(IsUnit (matrix x).det) ↔ x = 13/7 := by
  sorry

end matrix_not_invertible_l3700_370019


namespace min_fraction_sum_l3700_370054

def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_fraction_sum (A B C D : Nat) : 
  A ∈ Digits → B ∈ Digits → C ∈ Digits → D ∈ Digits →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  Nat.Prime B → Nat.Prime D →
  (∀ A' B' C' D' : Nat, 
    A' ∈ Digits → B' ∈ Digits → C' ∈ Digits → D' ∈ Digits →
    A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
    Nat.Prime B' → Nat.Prime D' →
    (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) ≤ (A' : ℚ) / (B' : ℚ) + (C' : ℚ) / (D' : ℚ)) →
  (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) = 1 / 5 := by
sorry

end min_fraction_sum_l3700_370054


namespace max_min_distance_on_sphere_l3700_370016

/-- A point on a unit sphere represented by its coordinates -/
def SpherePoint := ℝ × ℝ × ℝ

/-- The distance between two points on a unit sphere -/
def sphereDistance (p q : SpherePoint) : ℝ := sorry

/-- Checks if a point is on the unit sphere -/
def isOnUnitSphere (p : SpherePoint) : Prop := sorry

/-- Represents a configuration of five points on a unit sphere -/
def Configuration := Fin 5 → SpherePoint

/-- The minimum pairwise distance in a configuration -/
def minDistance (c : Configuration) : ℝ := sorry

/-- Checks if a configuration has two points at opposite poles and three equidistant points on the equator -/
def isOptimalConfiguration (c : Configuration) : Prop := sorry

theorem max_min_distance_on_sphere :
  ∀ c : Configuration, (∀ i, isOnUnitSphere (c i)) →
  minDistance c ≤ Real.sqrt 2 ∧
  (minDistance c = Real.sqrt 2 ↔ isOptimalConfiguration c) := by sorry

end max_min_distance_on_sphere_l3700_370016


namespace may_savings_l3700_370015

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (0-indexed)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end may_savings_l3700_370015


namespace half_plus_seven_equals_seventeen_l3700_370055

theorem half_plus_seven_equals_seventeen (x : ℝ) : (1/2) * x + 7 = 17 → x = 20 := by
  sorry

end half_plus_seven_equals_seventeen_l3700_370055


namespace inequality_proof_l3700_370031

theorem inequality_proof (x : ℝ) (h : x > 0) : 1/x + 4*x^2 ≥ 3 := by
  sorry

end inequality_proof_l3700_370031


namespace class_size_l3700_370073

theorem class_size (N : ℕ) (S D B : ℕ) : 
  S = (3 * N) / 5 →  -- 3/5 of the class swims
  D = (3 * N) / 5 →  -- 3/5 of the class dances
  B = 5 →            -- 5 pupils both swim and dance
  N = S + D - B →    -- Total is sum of swimmers and dancers minus overlap
  N = 25 := by sorry

end class_size_l3700_370073


namespace triangle_third_side_length_l3700_370081

theorem triangle_third_side_length 
  (a b : ℝ) 
  (γ : ℝ) 
  (ha : a = 10) 
  (hb : b = 15) 
  (hγ : γ = 150 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*Real.cos γ ∧ c = Real.sqrt (325 + 150 * Real.sqrt 3) :=
sorry

end triangle_third_side_length_l3700_370081


namespace divisibility_theorem_l3700_370038

theorem divisibility_theorem (a b c d m : ℤ) 
  (h_odd : Odd m)
  (h_div_sum : m ∣ (a + b + c + d))
  (h_div_sum_squares : m ∣ (a^2 + b^2 + c^2 + d^2)) :
  m ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) := by
  sorry

end divisibility_theorem_l3700_370038


namespace arithmetic_sequence_common_difference_l3700_370095

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is the arithmetic sequence
  (S : ℕ → ℝ)  -- S is the sum function
  (h1 : ∀ n, S n = -n^2 + 4*n)  -- Given condition
  (h2 : ∀ n, S (n+1) - S n = a (n+1))  -- Definition of sum of arithmetic sequence
  : ∃ d : ℝ, (∀ n, a (n+1) - a n = d) ∧ d = -2 :=
by sorry

end arithmetic_sequence_common_difference_l3700_370095


namespace A_intersect_B_l3700_370001

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem A_intersect_B : A ∩ B = {1} := by
  sorry

end A_intersect_B_l3700_370001


namespace trig_identity_l3700_370039

open Real

theorem trig_identity : 
  sin (150 * π / 180) * cos ((-420) * π / 180) + 
  cos ((-690) * π / 180) * sin (600 * π / 180) + 
  tan (405 * π / 180) = 1/2 := by
  sorry

end trig_identity_l3700_370039


namespace integer_solutions_of_equation_l3700_370065

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 7*x*y - 13*x + 15*y - 37 = 0 ↔ 
    ((x = -2 ∧ y = 11) ∨ (x = -1 ∧ y = 3) ∨ (x = 7 ∧ y = 2)) := by
  sorry

end integer_solutions_of_equation_l3700_370065


namespace multiplication_fraction_result_l3700_370034

theorem multiplication_fraction_result : 12 * (1 / 17) * 34 = 24 := by
  sorry

end multiplication_fraction_result_l3700_370034


namespace percentage_calculation_l3700_370051

theorem percentage_calculation (n : ℝ) (h : n = 5600) : 0.15 * (0.30 * (0.50 * n)) = 126 := by
  sorry

end percentage_calculation_l3700_370051


namespace license_plate_palindrome_probability_l3700_370069

/-- The number of possible letters in the license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in the license plate. -/
def num_digits : ℕ := 10

/-- The length of the letter sequence in the license plate. -/
def letter_seq_length : ℕ := 4

/-- The length of the digit sequence in the license plate. -/
def digit_seq_length : ℕ := 4

/-- The probability of a license plate containing at least one palindrome. -/
def palindrome_probability : ℚ := 775 / 67600

theorem license_plate_palindrome_probability :
  let letter_palindrome_prob := 1 / (num_letters ^ 2 : ℚ)
  let digit_palindrome_prob := 1 / (num_digits ^ 2 : ℚ)
  let total_prob := letter_palindrome_prob + digit_palindrome_prob - 
                    (letter_palindrome_prob * digit_palindrome_prob)
  total_prob = palindrome_probability := by sorry

end license_plate_palindrome_probability_l3700_370069


namespace highest_price_scheme_l3700_370079

theorem highest_price_scheme (n m : ℝ) (hn : 0 < n) (hnm : n < m) (hm : m < 100) :
  let price_A := 100 * (1 + m / 100) * (1 - n / 100)
  let price_B := 100 * (1 + n / 100) * (1 - m / 100)
  let price_C := 100 * (1 + (m + n) / 200) * (1 - (m + n) / 200)
  let price_D := 100 * (1 + m * n / 10000) * (1 - m * n / 10000)
  price_A ≥ price_B ∧ price_A ≥ price_C ∧ price_A ≥ price_D :=
by sorry

end highest_price_scheme_l3700_370079


namespace ben_win_probability_l3700_370078

theorem ben_win_probability (p_loss p_tie : ℚ) 
  (h_loss : p_loss = 5 / 12)
  (h_tie : p_tie = 1 / 6) :
  1 - p_loss - p_tie = 5 / 12 := by
  sorry

end ben_win_probability_l3700_370078


namespace net_income_calculation_l3700_370033

def calculate_net_income (spring_lawn : ℝ) (spring_garden : ℝ) (summer_lawn : ℝ) (summer_garden : ℝ)
  (fall_lawn : ℝ) (fall_garden : ℝ) (winter_snow : ℝ)
  (spring_lawn_supplies : ℝ) (spring_garden_supplies : ℝ)
  (summer_lawn_supplies : ℝ) (summer_garden_supplies : ℝ)
  (fall_lawn_supplies : ℝ) (fall_garden_supplies : ℝ)
  (winter_snow_supplies : ℝ)
  (advertising_percent : ℝ) (maintenance_percent : ℝ) : ℝ :=
  let total_earnings := spring_lawn + spring_garden + summer_lawn + summer_garden +
                        fall_lawn + fall_garden + winter_snow
  let total_supplies := spring_lawn_supplies + spring_garden_supplies +
                        summer_lawn_supplies + summer_garden_supplies +
                        fall_lawn_supplies + fall_garden_supplies +
                        winter_snow_supplies
  let total_gardening := spring_garden + summer_garden + fall_garden
  let total_lawn_mowing := spring_lawn + summer_lawn + fall_lawn
  let advertising_expenses := advertising_percent * total_gardening
  let maintenance_expenses := maintenance_percent * total_lawn_mowing
  total_earnings - total_supplies - advertising_expenses - maintenance_expenses

theorem net_income_calculation :
  calculate_net_income 200 150 600 450 300 350 100
                       80 50 150 100 75 75 25
                       0.15 0.10 = 1342.50 := by
  sorry

end net_income_calculation_l3700_370033


namespace transistors_in_2010_l3700_370076

/-- Moore's law doubling period in years -/
def doubling_period : ℕ := 2

/-- Initial year for the calculation -/
def initial_year : ℕ := 1995

/-- Final year for the calculation -/
def final_year : ℕ := 2010

/-- Initial number of transistors in 1995 -/
def initial_transistors : ℕ := 2000000

/-- Calculate the number of transistors based on Moore's law -/
def moores_law_transistors (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / doubling_period)

/-- Theorem stating the number of transistors in 2010 according to Moore's law -/
theorem transistors_in_2010 :
  moores_law_transistors (final_year - initial_year) = 256000000 := by
  sorry

end transistors_in_2010_l3700_370076


namespace unique_n_squared_plus_2n_prime_l3700_370050

theorem unique_n_squared_plus_2n_prime :
  ∃! (n : ℕ), n > 0 ∧ Nat.Prime (n^2 + 2*n) :=
sorry

end unique_n_squared_plus_2n_prime_l3700_370050


namespace not_prime_sum_product_l3700_370082

theorem not_prime_sum_product (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0)
  (h5 : a * c + b * d = (b + d + a - c) * (b + d - a + c)) :
  ¬ (Nat.Prime (a * b + c * d).natAbs) := by
sorry

end not_prime_sum_product_l3700_370082


namespace unique_solution_xyz_l3700_370005

theorem unique_solution_xyz (x y z : ℝ) 
  (eq1 : x + y = 4)
  (eq2 : x * y - z^2 = 4) :
  x = 2 ∧ y = 2 ∧ z = 0 :=
by sorry

end unique_solution_xyz_l3700_370005


namespace quadratic_inequality_solution_l3700_370023

theorem quadratic_inequality_solution (m : ℝ) :
  {x : ℝ | x^2 + (2*m+1)*x + m^2 + m > 0} = {x : ℝ | x > -m ∨ x < -m-1} := by
  sorry

end quadratic_inequality_solution_l3700_370023


namespace water_flow_problem_l3700_370013

/-- The water flow problem -/
theorem water_flow_problem (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive
  (h2 : (2 * (30 / x) + 2 * (30 / x) + 4 * (60 / x)) / 2 = 18) -- Total water collected and dumped
  : x = 10 := by
  sorry

end water_flow_problem_l3700_370013


namespace m_minus_n_values_l3700_370021

theorem m_minus_n_values (m n : ℤ) 
  (h1 : |m| = 3)
  (h2 : |n| = 5)
  (h3 : m + n > 0) :
  m - n = -2 ∨ m - n = -8 := by
  sorry

end m_minus_n_values_l3700_370021


namespace triangle_abc_properties_l3700_370062

/-- Triangle ABC with given conditions -/
structure TriangleABC where
  /-- Point B has coordinates (4,4) -/
  B : ℝ × ℝ
  hB : B = (4, 4)
  
  /-- The equation of the angle bisector of ∠A is y = 0 -/
  angle_bisector : ℝ → ℝ
  h_angle_bisector : ∀ x, angle_bisector x = 0
  
  /-- The equation of the altitude from B to AC is x - 2y + 2 = 0 -/
  altitude : ℝ → ℝ
  h_altitude : ∀ x, altitude x = (x + 2) / 2

/-- The coordinates of point C in triangle ABC -/
def point_C (t : TriangleABC) : ℝ × ℝ := (10, -8)

/-- The area of triangle ABC -/
def area (t : TriangleABC) : ℝ := 48

/-- Main theorem: The coordinates of C and the area of triangle ABC are correct -/
theorem triangle_abc_properties (t : TriangleABC) : 
  (point_C t = (10, -8)) ∧ (area t = 48) := by
  sorry

end triangle_abc_properties_l3700_370062


namespace restaurant_bill_proof_l3700_370046

theorem restaurant_bill_proof (n : ℕ) (extra : ℝ) (total : ℝ) : 
  n = 10 → 
  extra = 3 → 
  (n - 1) * (total / n + extra) = total → 
  total = 270 := by
sorry

end restaurant_bill_proof_l3700_370046


namespace matthew_friends_count_l3700_370010

/-- The number of friends Matthew gave crackers and cakes to -/
def num_friends : ℕ := 4

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 32

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := 98

/-- The number of crackers each person ate -/
def crackers_per_person : ℕ := 8

theorem matthew_friends_count :
  (initial_crackers / crackers_per_person = num_friends) ∧
  (initial_crackers % crackers_per_person = 0) :=
sorry

end matthew_friends_count_l3700_370010


namespace nonagon_diagonal_intersection_probability_is_two_sevenths_l3700_370009

/-- The probability of two randomly selected diagonals in a nonagon intersecting inside the nonagon -/
def nonagon_diagonal_intersection_probability : ℚ :=
  2 / 7

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of diagonals in a nonagon -/
def nonagon_diagonals : ℕ := nonagon_sides.choose 2 - nonagon_sides

/-- The number of ways to choose two diagonals in a nonagon -/
def diagonal_pairs : ℕ := nonagon_diagonals.choose 2

/-- The number of ways to choose four vertices in a nonagon -/
def four_vertex_selections : ℕ := nonagon_sides.choose 4

theorem nonagon_diagonal_intersection_probability_is_two_sevenths :
  nonagon_diagonal_intersection_probability = four_vertex_selections / diagonal_pairs :=
sorry

end nonagon_diagonal_intersection_probability_is_two_sevenths_l3700_370009


namespace equation_solution_l3700_370089

theorem equation_solution : 
  ∃ y : ℝ, 7 * (2 * y + 3) - 5 = -3 * (2 - 5 * y) ∧ y = 22 := by
  sorry

end equation_solution_l3700_370089


namespace balloon_count_l3700_370052

theorem balloon_count (my_balloons : ℕ) (friend_balloons : ℕ) 
  (h1 : friend_balloons = 5)
  (h2 : my_balloons - friend_balloons = 2) : 
  my_balloons = 7 := by
sorry

end balloon_count_l3700_370052
