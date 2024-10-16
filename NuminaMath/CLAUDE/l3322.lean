import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l3322_332238

theorem equation_solution : 
  ∃! x : ℝ, x > 0 ∧ Real.sqrt (3 * x - 2) + 9 / Real.sqrt (3 * x - 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3322_332238


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l3322_332206

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the bag of balls -/
def Bag : Multiset BallColor :=
  Multiset.replicate 3 BallColor.Red + Multiset.replicate 2 BallColor.Black

/-- Represents a draw of two balls from the bag -/
def Draw : Type := Fin 2 → BallColor

/-- The event of drawing at least one black ball -/
def AtLeastOneBlack (draw : Draw) : Prop :=
  ∃ i : Fin 2, draw i = BallColor.Black

/-- The event of drawing all red balls -/
def AllRed (draw : Draw) : Prop :=
  ∀ i : Fin 2, draw i = BallColor.Red

/-- The theorem stating that AtLeastOneBlack and AllRed are mutually exclusive -/
theorem mutually_exclusive_events :
  ∀ (draw : Draw), ¬(AtLeastOneBlack draw ∧ AllRed draw) :=
by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l3322_332206


namespace NUMINAMATH_CALUDE_frisbee_price_problem_l3322_332235

theorem frisbee_price_problem (total_frisbees : ℕ) (total_revenue : ℕ) 
  (price_some : ℕ) (min_sold_at_price_some : ℕ) :
  total_frisbees = 64 →
  total_revenue = 200 →
  price_some = 4 →
  min_sold_at_price_some = 8 →
  ∃ (price_others : ℕ), 
    price_others = 3 ∧
    ∃ (num_at_price_some : ℕ),
      num_at_price_some ≥ min_sold_at_price_some ∧
      price_some * num_at_price_some + price_others * (total_frisbees - num_at_price_some) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_frisbee_price_problem_l3322_332235


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3322_332290

theorem gcd_lcm_sum : Nat.gcd 40 72 + Nat.lcm 48 18 = 152 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3322_332290


namespace NUMINAMATH_CALUDE_survey_size_l3322_332214

-- Define the problem parameters
def percent_independent : ℚ := 752 / 1000
def percent_no_companionship : ℚ := 621 / 1000
def misinformed_students : ℕ := 41

-- Define the theorem
theorem survey_size :
  ∃ (total_students : ℕ),
    (total_students > 0) ∧
    (↑misinformed_students : ℚ) / (percent_independent * percent_no_companionship * ↑total_students) = 1 ∧
    total_students = 90 := by
  sorry

end NUMINAMATH_CALUDE_survey_size_l3322_332214


namespace NUMINAMATH_CALUDE_apples_per_person_is_two_l3322_332261

/-- Calculates the number of pounds of apples each person gets in a family -/
def applesPerPerson (originalPrice : ℚ) (priceIncrease : ℚ) (totalCost : ℚ) (familySize : ℕ) : ℚ :=
  let newPrice := originalPrice * (1 + priceIncrease)
  let costPerPerson := totalCost / familySize
  costPerPerson / newPrice

/-- Theorem stating that under the given conditions, each person gets 2 pounds of apples -/
theorem apples_per_person_is_two :
  applesPerPerson (8/5) (1/4) 16 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_person_is_two_l3322_332261


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3322_332256

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3322_332256


namespace NUMINAMATH_CALUDE_perimeter_AEC_l3322_332286

/-- A square with side length 2 and vertices A, B, C, D (in order) is folded so that C meets AB at C'.
    AC' = 1/4, and BC intersects AD at E. -/
structure FoldedSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  C' : ℝ × ℝ
  E : ℝ × ℝ
  h_square : A = (0, 2) ∧ B = (0, 0) ∧ C = (2, 0) ∧ D = (2, 2)
  h_C'_on_AB : C'.1 = 1/4 ∧ C'.2 = 0
  h_E_on_AD : E = (0, 2)

/-- The perimeter of triangle AEC' in a folded square is (√65 + 1)/4 -/
theorem perimeter_AEC'_folded_square (fs : FoldedSquare) :
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d fs.A fs.E + d fs.E fs.C' + d fs.C' fs.A = (Real.sqrt 65 + 1) / 4 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_AEC_l3322_332286


namespace NUMINAMATH_CALUDE_round_robin_tournament_games_l3322_332259

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of combinations of n things taken k at a time -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem round_robin_tournament_games :
  num_games 6 = binom 6 2 := by sorry

end NUMINAMATH_CALUDE_round_robin_tournament_games_l3322_332259


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_conditions_l3322_332202

theorem sufficient_but_not_necessary_conditions (a b : ℝ) :
  (∀ (a b : ℝ), a + b > 2 → a + b > 0) ∧
  (∀ (a b : ℝ), (a > 0 ∧ b > 0) → a + b > 0) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a + b > 2)) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a > 0 ∧ b > 0)) :=
by sorry


end NUMINAMATH_CALUDE_sufficient_but_not_necessary_conditions_l3322_332202


namespace NUMINAMATH_CALUDE_horse_and_saddle_value_l3322_332227

/-- The total value of a horse and saddle is $100, given that the horse is worth 7 times as much as the saddle, and the saddle is worth $12.5. -/
theorem horse_and_saddle_value :
  let saddle_value : ℝ := 12.5
  let horse_value : ℝ := 7 * saddle_value
  horse_value + saddle_value = 100 := by
  sorry

end NUMINAMATH_CALUDE_horse_and_saddle_value_l3322_332227


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3322_332215

/-- A quadratic function with axis of symmetry at x = 1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  axis_of_symmetry : -b / (2 * a) = 1

/-- Theorem: For a quadratic function with axis of symmetry at x = 1, c < 2b -/
theorem quadratic_inequality (f : QuadraticFunction) : f.c < 2 * f.b := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3322_332215


namespace NUMINAMATH_CALUDE_total_grading_time_l3322_332232

def math_worksheets : ℕ := 45
def science_worksheets : ℕ := 37
def history_worksheets : ℕ := 32

def math_grading_time : ℕ := 15
def science_grading_time : ℕ := 20
def history_grading_time : ℕ := 25

theorem total_grading_time :
  math_worksheets * math_grading_time +
  science_worksheets * science_grading_time +
  history_worksheets * history_grading_time = 2215 := by
sorry

end NUMINAMATH_CALUDE_total_grading_time_l3322_332232


namespace NUMINAMATH_CALUDE_paige_folders_l3322_332287

theorem paige_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 27 →
  deleted_files = 9 →
  files_per_folder = 6 →
  (initial_files - deleted_files) / files_per_folder = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_paige_folders_l3322_332287


namespace NUMINAMATH_CALUDE_specific_shiny_penny_last_probability_l3322_332223

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 4

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 4

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The probability of drawing a specific shiny penny last -/
def prob_specific_shiny_last : ℚ := 1 / 2

theorem specific_shiny_penny_last_probability :
  prob_specific_shiny_last = (Nat.choose (total_pennies - 1) (shiny_pennies - 1)) / (Nat.choose total_pennies shiny_pennies) :=
by sorry

end NUMINAMATH_CALUDE_specific_shiny_penny_last_probability_l3322_332223


namespace NUMINAMATH_CALUDE_solution_exists_l3322_332253

/-- The number of primes less than or equal to n -/
def ν (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem solution_exists (m : ℕ) (hm : m > 2) :
  (∃ n : ℕ, n > 1 ∧ n / ν n = m) → (∃ n : ℕ, n > 1 ∧ n / ν n = m - 1) :=
sorry

end NUMINAMATH_CALUDE_solution_exists_l3322_332253


namespace NUMINAMATH_CALUDE_application_methods_eq_540_l3322_332226

/-- The number of ways 6 students can apply to 3 colleges with restrictions -/
def application_methods : ℕ :=
  let total_ways := 3^6
  let no_applicants_one_college := 2^6
  let no_applicants_two_colleges := 1
  total_ways - 3 * no_applicants_one_college + 3 * no_applicants_two_colleges

/-- Theorem stating that the number of application methods is 540 -/
theorem application_methods_eq_540 : application_methods = 540 := by
  sorry

end NUMINAMATH_CALUDE_application_methods_eq_540_l3322_332226


namespace NUMINAMATH_CALUDE_hexagon_circumference_hexagon_circumference_proof_l3322_332229

/-- The circumference of a regular hexagon with side length 5 centimeters is 30 centimeters. -/
theorem hexagon_circumference : ℝ → Prop :=
  fun side_length =>
    side_length = 5 →
    (6 : ℝ) * side_length = 30

-- The proof is omitted
theorem hexagon_circumference_proof : hexagon_circumference 5 :=
  sorry

end NUMINAMATH_CALUDE_hexagon_circumference_hexagon_circumference_proof_l3322_332229


namespace NUMINAMATH_CALUDE_y_derivative_l3322_332212

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x + Real.sqrt x + 2

theorem y_derivative (x : ℝ) (h : x ≠ 0) :
  deriv y x = (x * Real.cos x - Real.sin x) / x^2 + 1 / (2 * Real.sqrt x) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3322_332212


namespace NUMINAMATH_CALUDE_plane_equation_from_point_and_normal_l3322_332254

/-- Given a point P₀ and a normal vector u⃗, prove that the equation
    ax + by + cz + d = 0 represents the plane passing through P₀ with normal vector u⃗. -/
theorem plane_equation_from_point_and_normal (P₀ : ℝ × ℝ × ℝ) (u : ℝ × ℝ × ℝ) 
  (a b c d : ℝ) :
  let (x₀, y₀, z₀) := P₀
  let (a', b', c') := u
  (a = 2 ∧ b = -1 ∧ c = -3 ∧ d = 3) →
  (x₀ = 1 ∧ y₀ = 2 ∧ z₀ = 1) →
  (a' = -2 ∧ b' = 1 ∧ c' = 3) →
  ∀ (x y z : ℝ), a*x + b*y + c*z + d = 0 ↔ 
    a'*(x - x₀) + b'*(y - y₀) + c'*(z - z₀) = 0 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_from_point_and_normal_l3322_332254


namespace NUMINAMATH_CALUDE_find_K_l3322_332216

theorem find_K : ∃ K : ℕ, 32^5 * 4^5 = 2^K ∧ K = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_K_l3322_332216


namespace NUMINAMATH_CALUDE_queen_then_club_probability_l3322_332246

-- Define a standard deck of cards
def standardDeck : ℕ := 52

-- Define the number of Queens in a standard deck
def numQueens : ℕ := 4

-- Define the number of clubs in a standard deck
def numClubs : ℕ := 13

-- Define the probability of drawing a Queen first and a club second
def probQueenThenClub : ℚ := 1 / 52

-- Theorem statement
theorem queen_then_club_probability :
  probQueenThenClub = (numQueens / standardDeck) * (numClubs / (standardDeck - 1)) :=
by sorry

end NUMINAMATH_CALUDE_queen_then_club_probability_l3322_332246


namespace NUMINAMATH_CALUDE_ramsey_theorem_l3322_332295

-- Define a type for people
variable (Person : Type)

-- Define the acquaintance relation
variable (knows : Person → Person → Prop)

-- Axiom: The acquaintance relation is symmetric (mutual)
axiom knows_symmetric : ∀ (a b : Person), knows a b ↔ knows b a

-- Define a group of 6 people
variable (group : Finset Person)
axiom group_size : group.card = 6

-- Main theorem
theorem ramsey_theorem :
  ∃ (subset : Finset Person),
    subset.card = 3 ∧
    subset ⊆ group ∧
    (∀ (a b : Person), a ∈ subset → b ∈ subset → a ≠ b → knows a b) ∨
    (∀ (a b : Person), a ∈ subset → b ∈ subset → a ≠ b → ¬knows a b) :=
sorry

end NUMINAMATH_CALUDE_ramsey_theorem_l3322_332295


namespace NUMINAMATH_CALUDE_equivalent_operation_l3322_332263

theorem equivalent_operation (x : ℚ) : 
  (x * (5 / 6)) / (2 / 3) = x * (5 / 4) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l3322_332263


namespace NUMINAMATH_CALUDE_allocation_count_l3322_332211

/-- The number of factories available for allocation --/
def num_factories : ℕ := 4

/-- The number of students to be allocated --/
def num_students : ℕ := 3

/-- The total number of possible allocations without restrictions --/
def total_allocations : ℕ := num_factories ^ num_students

/-- The number of allocations where no student goes to Factory A --/
def allocations_without_A : ℕ := (num_factories - 1) ^ num_students

/-- The number of valid allocations where at least one student goes to Factory A --/
def valid_allocations : ℕ := total_allocations - allocations_without_A

theorem allocation_count : valid_allocations = 37 := by
  sorry

end NUMINAMATH_CALUDE_allocation_count_l3322_332211


namespace NUMINAMATH_CALUDE_sum_of_min_max_FGH_is_23_l3322_332275

/-- Represents a single digit (0-9) -/
def SingleDigit : Type := { n : ℕ // n < 10 }

/-- Represents a number in the form F861G20H -/
def NumberFGH (F G H : SingleDigit) : ℕ := 
  F.1 * 100000000 + 861 * 100000 + G.1 * 10000 + 20 * 100 + H.1

/-- Condition that F861G20H is divisible by 11 -/
def IsDivisibleBy11 (F G H : SingleDigit) : Prop :=
  NumberFGH F G H % 11 = 0

theorem sum_of_min_max_FGH_is_23 :
  ∃ (Fmin Gmin Hmin Fmax Gmax Hmax : SingleDigit),
    (∀ F G H : SingleDigit, IsDivisibleBy11 F G H →
      Fmin.1 + Gmin.1 + Hmin.1 ≤ F.1 + G.1 + H.1 ∧
      F.1 + G.1 + H.1 ≤ Fmax.1 + Gmax.1 + Hmax.1) ∧
    Fmin.1 + Gmin.1 + Hmin.1 + Fmax.1 + Gmax.1 + Hmax.1 = 23 :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_max_FGH_is_23_l3322_332275


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3322_332291

/-- The maximum value of the quadratic function f(x) = -2x^2 + 4x - 18 is -16 -/
theorem quadratic_max_value :
  let f : ℝ → ℝ := fun x ↦ -2 * x^2 + 4 * x - 18
  ∃ M : ℝ, M = -16 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3322_332291


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l3322_332255

theorem max_sqrt_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 18) :
  ∃ (d : ℝ), d = 6 ∧ ∀ (a b : ℝ), a ≥ 0 → b ≥ 0 → a + b = 18 → Real.sqrt a + Real.sqrt b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l3322_332255


namespace NUMINAMATH_CALUDE_last_digit_is_four_l3322_332231

/-- Represents the process of repeatedly removing digits in odd positions -/
def remove_odd_positions (n : ℕ) : ℕ → ℕ
| 0 => 0
| 1 => n % 10
| m + 2 => remove_odd_positions (n / 100) m

/-- The initial 100-digit number -/
def initial_number : ℕ := 1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890

/-- The theorem stating that the last remaining digit is 4 -/
theorem last_digit_is_four :
  ∃ k, remove_odd_positions initial_number k = 4 ∧ 
       ∀ m > k, remove_odd_positions initial_number m = 0 :=
sorry

end NUMINAMATH_CALUDE_last_digit_is_four_l3322_332231


namespace NUMINAMATH_CALUDE_tony_money_left_l3322_332225

/-- The amount of money Tony has left after purchases at a baseball game. -/
def money_left (initial_amount ticket_cost hot_dog_cost drink_cost cap_cost : ℕ) : ℕ :=
  initial_amount - ticket_cost - hot_dog_cost - drink_cost - cap_cost

/-- Theorem stating that Tony has $13 left after his purchases. -/
theorem tony_money_left : 
  money_left 50 16 5 4 12 = 13 := by
  sorry

end NUMINAMATH_CALUDE_tony_money_left_l3322_332225


namespace NUMINAMATH_CALUDE_world_cup_knowledge_competition_l3322_332266

theorem world_cup_knowledge_competition (p_know : ℝ) (p_guess : ℝ) (num_options : ℕ) :
  p_know = 2/3 →
  p_guess = 1/3 →
  num_options = 4 →
  (p_know * 1 + p_guess * (1 / num_options)) / (p_know + p_guess * (1 / num_options)) = 8/9 :=
by sorry

end NUMINAMATH_CALUDE_world_cup_knowledge_competition_l3322_332266


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3322_332293

/-- A figure composed of square tiles -/
structure TiledFigure where
  tiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a figure, each sharing at least one side with the original figure -/
def add_tiles (figure : TiledFigure) (new_tiles : ℕ) : TiledFigure :=
  { tiles := figure.tiles + new_tiles,
    perimeter := figure.perimeter + 2 * new_tiles }

theorem perimeter_after_adding_tiles (initial_figure : TiledFigure) :
  initial_figure.tiles = 10 →
  initial_figure.perimeter = 16 →
  (add_tiles initial_figure 4).perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3322_332293


namespace NUMINAMATH_CALUDE_angle_B_measure_l3322_332243

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)
  (sum_angles : A + B + C + D = 360)

-- Define the theorem
theorem angle_B_measure (q : Quadrilateral) (h : q.A + q.C = 150) : q.B = 105 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l3322_332243


namespace NUMINAMATH_CALUDE_eighteenth_replacement_november_l3322_332241

/-- Represents months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Converts a number of months since January to a Month -/
def monthsToMonth (n : ℕ) : Month :=
  match n % 12 with
  | 0 => Month.December
  | 1 => Month.January
  | 2 => Month.February
  | 3 => Month.March
  | 4 => Month.April
  | 5 => Month.May
  | 6 => Month.June
  | 7 => Month.July
  | 8 => Month.August
  | 9 => Month.September
  | 10 => Month.October
  | _ => Month.November

/-- The month of the nth battery replacement, given replacements occur every 7 months starting from January -/
def batteryReplacementMonth (n : ℕ) : Month :=
  monthsToMonth (7 * (n - 1) + 1)

theorem eighteenth_replacement_november :
  batteryReplacementMonth 18 = Month.November := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_replacement_november_l3322_332241


namespace NUMINAMATH_CALUDE_slope_of_line_l3322_332204

/-- The slope of a line represented by the equation 3y = 4x + 9 is 4/3 -/
theorem slope_of_line (x y : ℝ) : 3 * y = 4 * x + 9 → (y - 3) / x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l3322_332204


namespace NUMINAMATH_CALUDE_number_to_billions_l3322_332205

/-- Converts a number to billions -/
def to_billions (n : ℕ) : ℚ :=
  (n : ℚ) / 1000000000

theorem number_to_billions :
  to_billions 640080000 = 0.64008 := by sorry

end NUMINAMATH_CALUDE_number_to_billions_l3322_332205


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3322_332272

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 1 → 1 / a + 1 / b ≤ 1 / x + 1 / y) ∧
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3322_332272


namespace NUMINAMATH_CALUDE_inequality_proof_l3322_332217

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a * b + b * c + c * a = 1) :
  Real.sqrt (a^3 + a) + Real.sqrt (b^3 + b) + Real.sqrt (c^3 + c) ≥ 2 * Real.sqrt (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3322_332217


namespace NUMINAMATH_CALUDE_inscriptions_exist_l3322_332271

/-- Represents the maker of a casket -/
inductive Maker
| Bellini
| Cellini

/-- Represents a casket with its inscription -/
structure Casket where
  maker : Maker
  inscription : Prop

/-- The pair of caskets satisfies the given conditions -/
def satisfies_conditions (golden silver : Casket) : Prop :=
  let P := (golden.maker = Maker.Bellini ∧ silver.maker = Maker.Cellini) ∨
           (golden.maker = Maker.Cellini ∧ silver.maker = Maker.Bellini)
  let Q := silver.maker = Maker.Cellini
  
  -- Condition 1: One can conclude that one casket is made by Bellini and the other by Cellini
  (golden.inscription ∧ silver.inscription → P) ∧
  
  -- Condition 1 (continued): But it's impossible to determine which casket is whose work
  (golden.inscription ∧ silver.inscription → ¬(golden.maker = Maker.Bellini ∨ golden.maker = Maker.Cellini)) ∧
  
  -- Condition 2: The inscription on either casket alone doesn't allow concluding about the makers
  (golden.inscription → ¬P) ∧
  (silver.inscription → ¬P)

/-- There exist inscriptions that satisfy the given conditions -/
theorem inscriptions_exist : ∃ (golden silver : Casket), satisfies_conditions golden silver := by
  sorry

end NUMINAMATH_CALUDE_inscriptions_exist_l3322_332271


namespace NUMINAMATH_CALUDE_complete_square_factorize_l3322_332257

-- Problem 1: Complete the square
theorem complete_square (x p : ℝ) : x^2 + 2*p*x + 1 = (x + p)^2 + (1 - p^2) := by sorry

-- Problem 2: Factorization
theorem factorize (a b : ℝ) : a^2 - b^2 + 4*a + 2*b + 3 = (a + b + 1)*(a - b + 3) := by sorry

end NUMINAMATH_CALUDE_complete_square_factorize_l3322_332257


namespace NUMINAMATH_CALUDE_chocolate_eggs_weight_l3322_332219

/-- Calculates the total weight of remaining chocolate eggs after one box is discarded -/
theorem chocolate_eggs_weight (total_eggs : ℕ) (egg_weight : ℕ) (num_boxes : ℕ) :
  total_eggs = 12 →
  egg_weight = 10 →
  num_boxes = 4 →
  (total_eggs * egg_weight) - (total_eggs / num_boxes * egg_weight) = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_eggs_weight_l3322_332219


namespace NUMINAMATH_CALUDE_workshop_workers_l3322_332233

theorem workshop_workers (average_salary : ℝ) (technician_salary : ℝ) (non_technician_salary : ℝ) 
  (num_technicians : ℕ) (h1 : average_salary = 8000) 
  (h2 : technician_salary = 10000) (h3 : non_technician_salary = 6000) 
  (h4 : num_technicians = 7) : 
  ∃ (total_workers : ℕ), total_workers = 14 ∧ 
  (num_technicians : ℝ) * technician_salary + 
  ((total_workers - num_technicians) : ℝ) * non_technician_salary = 
  (total_workers : ℝ) * average_salary :=
sorry

end NUMINAMATH_CALUDE_workshop_workers_l3322_332233


namespace NUMINAMATH_CALUDE_square_area_ratio_l3322_332200

theorem square_area_ratio : 
  let small_side : ℝ := 5
  let large_side : ℝ := small_side + 5
  let small_area : ℝ := small_side ^ 2
  let large_area : ℝ := large_side ^ 2
  large_area / small_area = 4 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3322_332200


namespace NUMINAMATH_CALUDE_composition_of_even_function_is_even_l3322_332248

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem composition_of_even_function_is_even (g : ℝ → ℝ) (h : is_even_function g) :
  is_even_function (fun x ↦ g (g (g x))) := by sorry

end NUMINAMATH_CALUDE_composition_of_even_function_is_even_l3322_332248


namespace NUMINAMATH_CALUDE_soccer_team_selection_l3322_332282

/-- The total number of players in the soccer team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of players to be chosen as starters -/
def num_starters : ℕ := 6

/-- The number of quadruplets to be chosen as starters -/
def num_quadruplets_chosen : ℕ := 1

/-- The number of ways to choose the starting lineup -/
def num_ways : ℕ := 3168

theorem soccer_team_selection :
  (num_quadruplets * Nat.choose (total_players - num_quadruplets) (num_starters - num_quadruplets_chosen)) = num_ways :=
sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l3322_332282


namespace NUMINAMATH_CALUDE_ant_on_red_after_six_moves_probability_on_red_after_six_moves_l3322_332242

/-- Represents the color of a dot on the lattice -/
inductive DotColor
| Red
| Blue

/-- Represents the state of the ant's position -/
structure AntState :=
  (color : DotColor)

/-- Defines a single move of the ant -/
def move (state : AntState) : AntState :=
  match state.color with
  | DotColor.Red => { color := DotColor.Blue }
  | DotColor.Blue => { color := DotColor.Red }

/-- Applies n moves to the initial state -/
def apply_moves (initial : AntState) (n : ℕ) : AntState :=
  match n with
  | 0 => initial
  | n + 1 => move (apply_moves initial n)

/-- The main theorem to prove -/
theorem ant_on_red_after_six_moves (initial : AntState) :
  initial.color = DotColor.Red →
  (apply_moves initial 6).color = DotColor.Red :=
sorry

/-- The probability of the ant being on a red dot after 6 moves -/
theorem probability_on_red_after_six_moves (initial : AntState) :
  initial.color = DotColor.Red →
  ∃ (p : ℝ), p = 1 ∧ 
  (∀ (final : AntState), (apply_moves initial 6).color = DotColor.Red → p = 1) :=
sorry

end NUMINAMATH_CALUDE_ant_on_red_after_six_moves_probability_on_red_after_six_moves_l3322_332242


namespace NUMINAMATH_CALUDE_smallest_prime_is_prime_q_value_l3322_332289

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def smallest_prime : ℕ := 2

theorem smallest_prime_is_prime : is_prime smallest_prime := by sorry

theorem q_value (p q : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (h_relation : q = 13 * p + 1) 
  (h_smallest : p = smallest_prime) : 
  q = 29 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_is_prime_q_value_l3322_332289


namespace NUMINAMATH_CALUDE_city_distance_proof_l3322_332292

/-- Calculates the actual distance between two cities given the map distance and scale. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Proves that the actual distance between two cities is 2400 km given the map conditions. -/
theorem city_distance_proof (map_distance : ℝ) (scale : ℝ) 
  (h1 : map_distance = 120)
  (h2 : scale = 20) : 
  actual_distance map_distance scale = 2400 := by
  sorry

#check city_distance_proof

end NUMINAMATH_CALUDE_city_distance_proof_l3322_332292


namespace NUMINAMATH_CALUDE_custom_mult_identity_value_l3322_332252

/-- Custom multiplication operation -/
def custom_mult (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c * x * y

theorem custom_mult_identity_value (a b c : ℝ) :
  (custom_mult a b c 1 2 = 4) →
  (custom_mult a b c 2 3 = 6) →
  (∃ m : ℝ, m ≠ 0 ∧ ∀ x : ℝ, custom_mult a b c x m = x) →
  ∃ m : ℝ, m = 13 ∧ m ≠ 0 ∧ ∀ x : ℝ, custom_mult a b c x m = x :=
by sorry

end NUMINAMATH_CALUDE_custom_mult_identity_value_l3322_332252


namespace NUMINAMATH_CALUDE_number_of_factors_14_pow_15_l3322_332249

def b : ℕ := 14
def n : ℕ := 15

theorem number_of_factors_14_pow_15 :
  (∀ x ∈ Finset.range 16, x > 0 → x.divisors.card ≤ (b^n).divisors.card) ∧
  (b^n).divisors.card = 256 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_14_pow_15_l3322_332249


namespace NUMINAMATH_CALUDE_compound_propositions_true_l3322_332222

-- Define proposition P
def P : Prop := ∀ x y : ℝ, x > y → -x > -y

-- Define proposition Q
def Q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem to prove
theorem compound_propositions_true : (¬P ∨ ¬Q) ∧ ((¬P) ∨ Q) := by
  sorry

end NUMINAMATH_CALUDE_compound_propositions_true_l3322_332222


namespace NUMINAMATH_CALUDE_projection_result_l3322_332230

/-- A projection that takes [2, -4] to [3, -3] -/
def projection (v : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The projection satisfies the given condition -/
axiom projection_condition : projection (2, -4) = (3, -3)

/-- Theorem: The projection takes [3, 5] to [-1, 1] -/
theorem projection_result : projection (3, 5) = (-1, 1) := by sorry

end NUMINAMATH_CALUDE_projection_result_l3322_332230


namespace NUMINAMATH_CALUDE_tan_is_odd_l3322_332281

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem tan_is_odd : ∀ x : ℝ, tan (-x) = -tan x := by sorry

end NUMINAMATH_CALUDE_tan_is_odd_l3322_332281


namespace NUMINAMATH_CALUDE_min_a_for_subset_l3322_332284

theorem min_a_for_subset (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 5, x^2 - 6*x ≤ a + 2) ↔ a ≥ -7 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_subset_l3322_332284


namespace NUMINAMATH_CALUDE_tetrahedron_division_l3322_332247

-- Define a regular tetrahedron
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_positive : edge_length > 0)

-- Define the division of edges
def divide_edges (t : RegularTetrahedron) : ℕ := 3

-- Define the planes drawn through division points
structure DivisionPlanes :=
  (tetrahedron : RegularTetrahedron)
  (num_divisions : ℕ)
  (parallel_to_faces : Bool)

-- Define the number of parts the tetrahedron is divided into
def num_parts (t : RegularTetrahedron) (d : DivisionPlanes) : ℕ := 15

-- Theorem statement
theorem tetrahedron_division (t : RegularTetrahedron) :
  let d := DivisionPlanes.mk t (divide_edges t) true
  num_parts t d = 15 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_division_l3322_332247


namespace NUMINAMATH_CALUDE_set_intersection_range_l3322_332207

theorem set_intersection_range (a : ℝ) : 
  let A : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
  let B : Set ℝ := {x | 0 < x ∧ x < 1}
  A ∩ B = ∅ → (a ≤ -1/2 ∨ a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_range_l3322_332207


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_l3322_332240

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

-- Define the dot product of two 2D vectors
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

-- Define the perpendicularity condition
def is_perpendicular (u v : Fin 2 → ℝ) : Prop := dot_product u v = 0

-- State the theorem
theorem perpendicular_vectors_k (k : ℝ) :
  is_perpendicular 
    (fun i => k * (a i) + (b i)) 
    (fun i => (a i) - 3 * (b i)) 
  → k = 19 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_l3322_332240


namespace NUMINAMATH_CALUDE_percentage_of_a_to_b_l3322_332201

theorem percentage_of_a_to_b (A B C D : ℝ) 
  (h1 : A = 0.125 * C)
  (h2 : B = 0.375 * D)
  (h3 : D = 1.225 * C)
  (h4 : C = 0.805 * B) :
  A = 0.100625 * B := by
sorry

end NUMINAMATH_CALUDE_percentage_of_a_to_b_l3322_332201


namespace NUMINAMATH_CALUDE_apps_deleted_l3322_332245

theorem apps_deleted (initial_apps final_apps : ℕ) (h1 : initial_apps = 12) (h2 : final_apps = 4) :
  initial_apps - final_apps = 8 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_l3322_332245


namespace NUMINAMATH_CALUDE_tan_alpha_eq_one_l3322_332268

theorem tan_alpha_eq_one (α : Real) 
  (h : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) : 
  Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_one_l3322_332268


namespace NUMINAMATH_CALUDE_factorization_2x_minus_x_squared_l3322_332236

theorem factorization_2x_minus_x_squared (x : ℝ) : 2*x - x^2 = x*(2-x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2x_minus_x_squared_l3322_332236


namespace NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l3322_332285

theorem polar_to_rectangular_coordinates :
  let r : ℝ := 2
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 1 ∧ y = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l3322_332285


namespace NUMINAMATH_CALUDE_sum_of_perimeters_l3322_332203

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) :
  4 * x + 4 * y = 4 * (Real.sqrt 63 + Real.sqrt 22) := by
sorry

end NUMINAMATH_CALUDE_sum_of_perimeters_l3322_332203


namespace NUMINAMATH_CALUDE_constant_speed_walking_time_l3322_332296

/-- Represents the time taken to walk a certain distance at a constant speed -/
structure WalkingTime where
  distance : ℝ
  time : ℝ

/-- Given a constant walking speed, prove that if it takes 30 minutes to walk 4 kilometers,
    then it will take 15 minutes to walk 2 kilometers -/
theorem constant_speed_walking_time 
  (speed : ℝ) 
  (library : WalkingTime) 
  (school : WalkingTime) 
  (h1 : speed > 0)
  (h2 : library.distance = 4)
  (h3 : library.time = 30)
  (h4 : school.distance = 2)
  (h5 : library.distance / library.time = speed)
  (h6 : school.distance / school.time = speed) :
  school.time = 15 := by
  sorry

end NUMINAMATH_CALUDE_constant_speed_walking_time_l3322_332296


namespace NUMINAMATH_CALUDE_triangle_side_length_l3322_332244

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 7)
  (h2 : c = 6)
  (h3 : Real.cos (B - C) = 37/40)
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h5 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h6 : A + B + C = π) :
  a = Real.sqrt 66.1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3322_332244


namespace NUMINAMATH_CALUDE_distance_between_roots_l3322_332299

/-- The distance between the roots of x^2 - 2x - 3 = 0 is 4 -/
theorem distance_between_roots : ∃ x₁ x₂ : ℝ, 
  x₁^2 - 2*x₁ - 3 = 0 ∧ 
  x₂^2 - 2*x₂ - 3 = 0 ∧ 
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 4 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_roots_l3322_332299


namespace NUMINAMATH_CALUDE_filled_circles_in_2009_l3322_332224

/-- Represents the cumulative number of circles (both filled and empty) after n filled circles -/
def s (n : ℕ) : ℕ := (n^2 + n) / 2

/-- Represents the pattern where the nth filled circle is followed by n empty circles -/
def circle_pattern (n : ℕ) : ℕ := n + 1

theorem filled_circles_in_2009 : 
  ∃ k : ℕ, k = 63 ∧ s k ≤ 2009 ∧ s (k + 1) > 2009 :=
sorry

end NUMINAMATH_CALUDE_filled_circles_in_2009_l3322_332224


namespace NUMINAMATH_CALUDE_ellipse_properties_l3322_332279

/-- An ellipse defined by the equation 25x^2 + 9y^2 = 225 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 25 * p.1^2 + 9 * p.2^2 = 225}

/-- The length of the semi-major axis of the ellipse -/
def semiMajorAxis : ℝ := 5

/-- The length of the semi-minor axis of the ellipse -/
def semiMinorAxis : ℝ := 3

/-- The distance from the center to a focus of the ellipse -/
def focalDistance : ℝ := 4

theorem ellipse_properties :
  (2 * semiMajorAxis = 10) ∧
  (2 * semiMinorAxis = 6) ∧
  (focalDistance / semiMajorAxis = 0.8) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3322_332279


namespace NUMINAMATH_CALUDE_initial_amount_of_liquid_A_l3322_332251

/-- Given a can with a mixture of liquids A and B, this theorem proves the initial amount of liquid A
    based on the given conditions and ratios. -/
theorem initial_amount_of_liquid_A (x : ℝ) : 
  -- Initial ratio of A to B is 7:5
  7 * x / (5 * x) = 7 / 5 →
  -- After removing 9 liters and adding B to make new ratio 7:9
  (7 * x - 9 * (7 / 12)) / (5 * x - 9 * (5 / 12) + 9) = 7 / 9 →
  -- The initial amount of liquid A was 21 liters
  7 * x = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_of_liquid_A_l3322_332251


namespace NUMINAMATH_CALUDE_player_b_wins_l3322_332297

/-- Represents a chessboard --/
def Chessboard := Fin 8 → Fin 8 → Option Bool

/-- Represents a position on the chessboard --/
def Position := Fin 8 × Fin 8

/-- Checks if a bishop can be captured at a given position --/
def canBeCaptured (board : Chessboard) (pos : Position) : Prop :=
  sorry

/-- Represents a valid move in the game --/
def ValidMove (board : Chessboard) (pos : Position) : Prop :=
  board pos.1 pos.2 = none ∧ ¬canBeCaptured board pos

/-- Represents the state of the game --/
structure GameState where
  board : Chessboard
  playerATurn : Bool

/-- Represents a strategy for a player --/
def Strategy := GameState → Position

/-- Checks if a strategy is winning for Player B --/
def isWinningStrategyForB (s : Strategy) : Prop :=
  sorry

/-- The main theorem stating that Player B has a winning strategy --/
theorem player_b_wins : ∃ s : Strategy, isWinningStrategyForB s :=
  sorry

end NUMINAMATH_CALUDE_player_b_wins_l3322_332297


namespace NUMINAMATH_CALUDE_leo_score_in_blackjack_l3322_332208

/-- In a blackjack game, given the scores of Caroline and Anthony, 
    and the fact that Leo is the winner with the winning score, 
    prove that Leo's score is 21. -/
theorem leo_score_in_blackjack 
  (caroline_score : ℕ) 
  (anthony_score : ℕ) 
  (winning_score : ℕ) 
  (leo_is_winner : Bool) : ℕ :=
by
  -- Define the given conditions
  have h1 : caroline_score = 13 := by sorry
  have h2 : anthony_score = 19 := by sorry
  have h3 : winning_score = 21 := by sorry
  have h4 : leo_is_winner = true := by sorry

  -- Prove that Leo's score is equal to the winning score
  sorry

#check leo_score_in_blackjack

end NUMINAMATH_CALUDE_leo_score_in_blackjack_l3322_332208


namespace NUMINAMATH_CALUDE_pedestrian_speed_theorem_l3322_332260

theorem pedestrian_speed_theorem :
  ∃ (v : ℝ), v > 0 ∧
  (∀ (t : ℝ), 0 ≤ t ∧ t ≤ 2.5 →
    (if t % 1 < 0.5 then (5 + v) else (5 - v)) * 0.5 +
    (if (t + 0.5) % 1 < 0.5 then (5 + v) else (5 - v)) * 0.5 = 5) ∧
  ((4 * (5 + v) * 0.5 + 3 * (5 - v) * 0.5) / 3.5 > 5) :=
by sorry

end NUMINAMATH_CALUDE_pedestrian_speed_theorem_l3322_332260


namespace NUMINAMATH_CALUDE_f_discontinuities_l3322_332273

noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then
    if x ≠ -2 then (x^2 + 7*x + 10) / (x^2 - 4)
    else 11/4
  else 4*x - 3

theorem f_discontinuities :
  (∃ (L : ℝ), ContinuousAt (fun x => if x ≠ -2 then f x else L) (-2)) ∧
  (¬ ContinuousAt f 2) := by
  sorry

end NUMINAMATH_CALUDE_f_discontinuities_l3322_332273


namespace NUMINAMATH_CALUDE_repair_cost_theorem_l3322_332283

def new_shoes_cost : ℝ := 28
def new_shoes_lifespan : ℝ := 2
def used_shoes_lifespan : ℝ := 1
def percentage_difference : ℝ := 0.2173913043478261

theorem repair_cost_theorem :
  ∃ (repair_cost : ℝ),
    repair_cost = 11.50 ∧
    (new_shoes_cost / new_shoes_lifespan) = repair_cost * (1 + percentage_difference) :=
by sorry

end NUMINAMATH_CALUDE_repair_cost_theorem_l3322_332283


namespace NUMINAMATH_CALUDE_expression_evaluation_l3322_332220

theorem expression_evaluation :
  let a : ℤ := -1
  (a^2 + 1) - 3*a*(a - 1) + 2*(a^2 + a - 1) = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3322_332220


namespace NUMINAMATH_CALUDE_chord_equation_through_bisection_point_l3322_332221

/-- Given a parabola y² = 6x and a chord passing through point P(4, 1) that is bisected at P,
    prove that the equation of the line l on which this chord lies is 3x - y - 11 = 0. -/
theorem chord_equation_through_bisection_point (x y : ℝ) :
  (∀ x y, y^2 = 6*x) →  -- Parabola equation
  (∃ x₁ y₁ x₂ y₂ : ℝ,   -- Existence of two points on the parabola
    y₁^2 = 6*x₁ ∧ y₂^2 = 6*x₂ ∧
    (4 = (x₁ + x₂) / 2) ∧ (1 = (y₁ + y₂) / 2)) →  -- P(4,1) is midpoint
  (3*x - y - 11 = 0) :=  -- Equation of the line
by sorry

end NUMINAMATH_CALUDE_chord_equation_through_bisection_point_l3322_332221


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l3322_332270

/-- Rectangle with known side length and area -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle similar to Rectangle1 with known diagonal -/
structure Rectangle2 where
  diagonal : ℝ

/-- The area of Rectangle2 given the properties of Rectangle1 and Rectangle2 -/
def area_rectangle2 (r1 : Rectangle1) (r2 : Rectangle2) : ℝ :=
  160

theorem rectangle_area_proof (r1 : Rectangle1) (r2 : Rectangle2) 
  (h1 : r1.side = 4)
  (h2 : r1.area = 32)
  (h3 : r2.diagonal = 20) :
  area_rectangle2 r1 r2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l3322_332270


namespace NUMINAMATH_CALUDE_water_requirement_proof_l3322_332250

/-- The water requirement per household per month in a village -/
def water_per_household (total_water : ℕ) (num_households : ℕ) : ℕ :=
  total_water / num_households

/-- Theorem: The water requirement per household per month is 200 litres -/
theorem water_requirement_proof :
  water_per_household 2000 10 = 200 := by
  sorry

end NUMINAMATH_CALUDE_water_requirement_proof_l3322_332250


namespace NUMINAMATH_CALUDE_car_speed_decrease_l3322_332210

/-- Proves that the speed decrease per interval is 3 mph given the conditions of the problem -/
theorem car_speed_decrease (initial_speed : ℝ) (distance_fifth : ℝ) (interval_duration : ℝ) :
  initial_speed = 45 →
  distance_fifth = 4.4 →
  interval_duration = 8 / 60 →
  ∃ (speed_decrease : ℝ),
    speed_decrease = 3 ∧
    initial_speed - 4 * speed_decrease = distance_fifth / interval_duration :=
by sorry

end NUMINAMATH_CALUDE_car_speed_decrease_l3322_332210


namespace NUMINAMATH_CALUDE_six_digit_square_numbers_l3322_332274

theorem six_digit_square_numbers : 
  ∀ n : ℕ, 
    (100000 ≤ n ∧ n < 1000000) → 
    (∃ m : ℕ, m < 1000 ∧ n = m^2) → 
    (n = 390625 ∨ n = 141376) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_square_numbers_l3322_332274


namespace NUMINAMATH_CALUDE_adjacent_same_tribe_l3322_332239

-- Define the four tribes
inductive Tribe
| Human
| Dwarf
| Elf
| Goblin

-- Define the seating arrangement
def Seating := Fin 33 → Tribe

-- Define the condition that humans cannot sit next to goblins
def NoHumanNextToGoblin (s : Seating) : Prop :=
  ∀ i : Fin 33, (s i = Tribe.Human ∧ s (i + 1) = Tribe.Goblin) ∨
                (s i = Tribe.Goblin ∧ s (i + 1) = Tribe.Human) → False

-- Define the condition that elves cannot sit next to dwarves
def NoElfNextToDwarf (s : Seating) : Prop :=
  ∀ i : Fin 33, (s i = Tribe.Elf ∧ s (i + 1) = Tribe.Dwarf) ∨
                (s i = Tribe.Dwarf ∧ s (i + 1) = Tribe.Elf) → False

-- Define the property of having adjacent same tribe representatives
def HasAdjacentSameTribe (s : Seating) : Prop :=
  ∃ i : Fin 33, s i = s (i + 1)

-- State the theorem
theorem adjacent_same_tribe (s : Seating) 
  (no_human_goblin : NoHumanNextToGoblin s) 
  (no_elf_dwarf : NoElfNextToDwarf s) : 
  HasAdjacentSameTribe s :=
sorry

end NUMINAMATH_CALUDE_adjacent_same_tribe_l3322_332239


namespace NUMINAMATH_CALUDE_new_average_after_removing_scores_l3322_332288

theorem new_average_after_removing_scores (n : ℕ) (original_avg : ℚ) (score1 score2 : ℕ) :
  n = 60 →
  original_avg = 82 →
  score1 = 95 →
  score2 = 97 →
  let total_sum := n * original_avg
  let remaining_sum := total_sum - (score1 + score2)
  let new_avg := remaining_sum / (n - 2)
  new_avg = 81.52 := by sorry

end NUMINAMATH_CALUDE_new_average_after_removing_scores_l3322_332288


namespace NUMINAMATH_CALUDE_connie_marbles_l3322_332262

/-- The number of marbles Connie started with -/
def initial_marbles : ℕ := 776

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℚ := 183.0

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := 593

theorem connie_marbles :
  (initial_marbles : ℚ) - marbles_given = remaining_marbles :=
by sorry

end NUMINAMATH_CALUDE_connie_marbles_l3322_332262


namespace NUMINAMATH_CALUDE_bugs_and_flowers_l3322_332264

/-- Given that 2.0 bugs ate 3.0 flowers in total, prove that each bug ate 1.5 flowers. -/
theorem bugs_and_flowers (total_bugs : ℝ) (total_flowers : ℝ) 
  (h1 : total_bugs = 2.0) 
  (h2 : total_flowers = 3.0) : 
  total_flowers / total_bugs = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_bugs_and_flowers_l3322_332264


namespace NUMINAMATH_CALUDE_binomial_divisibility_l3322_332277

theorem binomial_divisibility (k : ℕ) (hk : k ≥ 2) :
  (∃ m : ℕ, Nat.choose (2^k) 2 + Nat.choose (2^k) 3 = 2^(3*k) * m) ∧
  (∀ n : ℕ, Nat.choose (2^k) 2 + Nat.choose (2^k) 3 ≠ 2^(3*k + 1) * n) :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l3322_332277


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3322_332294

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_axes : Bool
  eccentricity : ℝ

/-- The equation of asymptotes for a hyperbola -/
def asymptote_equation (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = x ∨ y = -x}

theorem hyperbola_asymptotes (C : Hyperbola) 
  (h1 : C.center = (0, 0)) 
  (h2 : C.foci_on_axes = true) 
  (h3 : C.eccentricity = Real.sqrt 2) : 
  asymptote_equation C = {(x, y) | y = x ∨ y = -x} := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3322_332294


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3322_332298

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  let z : ℂ := 2 / (-1 + i)
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3322_332298


namespace NUMINAMATH_CALUDE_fraction_addition_l3322_332280

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l3322_332280


namespace NUMINAMATH_CALUDE_batsman_highest_score_l3322_332209

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (overall_average : ℚ) 
  (score_difference : ℕ) 
  (average_without_extremes : ℚ) 
  (h : total_innings = 46)
  (i : overall_average = 62)
  (j : score_difference = 150)
  (k : average_without_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    highest_score + lowest_score = total_innings * overall_average - (total_innings - 2) * average_without_extremes ∧
    highest_score = 221 :=
by sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l3322_332209


namespace NUMINAMATH_CALUDE_symmetric_trig_function_property_l3322_332218

/-- Given a function f(x) = a*sin(2x) + b*cos(2x) where a and b are real numbers,
    ab ≠ 0, and f is symmetric about x = π/6, prove that a = √3 * b. -/
theorem symmetric_trig_function_property (a b : ℝ) (h1 : a * b ≠ 0) :
  (∀ x, a * Real.sin (2 * x) + b * Real.cos (2 * x) = 
        a * Real.sin (2 * (Real.pi / 6 - x)) + b * Real.cos (2 * (Real.pi / 6 - x))) →
  a = Real.sqrt 3 * b := by
  sorry

end NUMINAMATH_CALUDE_symmetric_trig_function_property_l3322_332218


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_rope_l3322_332267

/-- Represents the sides of an isosceles triangle --/
structure IsoscelesTriangle where
  short : ℝ
  long : ℝ
  isIsosceles : long = 2 * short

/-- Checks if the given sides form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem to be proved --/
theorem isosceles_triangle_from_rope (t : IsoscelesTriangle) :
  t.short + t.long + t.long = 20 →
  is_valid_triangle t.short t.long t.long →
  t.short = 4 ∧ t.long = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_rope_l3322_332267


namespace NUMINAMATH_CALUDE_b_payment_l3322_332228

/-- Calculate the amount b should pay for renting a pasture -/
theorem b_payment (total_rent : ℕ) 
  (a_horses a_months a_rate : ℕ) 
  (b_horses b_months b_rate : ℕ)
  (c_horses c_months c_rate : ℕ)
  (d_horses d_months d_rate : ℕ)
  (h_total_rent : total_rent = 725)
  (h_a : a_horses = 12 ∧ a_months = 8 ∧ a_rate = 5)
  (h_b : b_horses = 16 ∧ b_months = 9 ∧ b_rate = 6)
  (h_c : c_horses = 18 ∧ c_months = 6 ∧ c_rate = 7)
  (h_d : d_horses = 20 ∧ d_months = 4 ∧ d_rate = 4) :
  ∃ (b_payment : ℕ), b_payment = 259 ∧ 
  b_payment = round ((b_horses * b_months * b_rate : ℚ) / 
    ((a_horses * a_months * a_rate + b_horses * b_months * b_rate + 
      c_horses * c_months * c_rate + d_horses * d_months * d_rate) : ℚ) * total_rent) :=
by sorry

#check b_payment

end NUMINAMATH_CALUDE_b_payment_l3322_332228


namespace NUMINAMATH_CALUDE_sin_cos_cube_difference_squared_l3322_332258

theorem sin_cos_cube_difference_squared (θ : Real) 
  (h : Real.sin θ - Real.cos θ = (Real.sqrt 6 - Real.sqrt 2) / 2) : 
  24 * (Real.sin θ ^ 3 - Real.cos θ ^ 3) ^ 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_cube_difference_squared_l3322_332258


namespace NUMINAMATH_CALUDE_gift_cost_l3322_332269

theorem gift_cost (half_cost : ℝ) (h : half_cost = 14) : 
  2 * half_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_l3322_332269


namespace NUMINAMATH_CALUDE_sin_cos_power_12_range_l3322_332234

theorem sin_cos_power_12_range (x : ℝ) : 
  (1 : ℝ) / 32 ≤ Real.sin x ^ 12 + Real.cos x ^ 12 ∧ 
  Real.sin x ^ 12 + Real.cos x ^ 12 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_power_12_range_l3322_332234


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3322_332237

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 1 / (2 * x + y) + 1 / (y + 1) = 1) : 
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 1 / (2 * x' + y') + 1 / (y' + 1) = 1 → x + 2 * y ≤ x' + 2 * y') ∧ 
  x + 2 * y = Real.sqrt 3 + 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3322_332237


namespace NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l3322_332213

/-- Given an association with total members, votes cast, and the winning candidate's votes as a percentage of total membership, calculate the percentage of votes cast that the winning candidate received. -/
theorem winning_candidate_vote_percentage
  (total_members : ℕ)
  (votes_cast : ℕ)
  (winning_votes_percentage_of_total : ℚ)
  (h1 : total_members = 1600)
  (h2 : votes_cast = 525)
  (h3 : winning_votes_percentage_of_total = 19.6875 / 100) :
  (winning_votes_percentage_of_total * total_members) / votes_cast = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l3322_332213


namespace NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l3322_332276

theorem cos_pi_sixth_minus_alpha (α : ℝ) (h : Real.sin (α + π / 3) = 1 / 2) :
  Real.cos (π / 6 - α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l3322_332276


namespace NUMINAMATH_CALUDE_matrix_cube_computation_l3322_332278

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_cube_computation :
  A ^ 3 = !![(-4), 2; (-2), 1] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_computation_l3322_332278


namespace NUMINAMATH_CALUDE_methods_B_and_D_are_correct_l3322_332265

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * x + 5 * y = 18
def equation2 (x y : ℝ) : Prop := 7 * x + 4 * y = 36

-- Define method B
def methodB (x y : ℝ) : Prop :=
  ∃ (z : ℝ), 9 * x + 9 * y = 54 ∧ z * (9 * x + 9 * y) - (2 * x + 5 * y) = z * 54 - 18

-- Define method D
def methodD (x y : ℝ) : Prop :=
  5 * (7 * x + 4 * y) - 4 * (2 * x + 5 * y) = 5 * 36 - 4 * 18

-- Theorem statement
theorem methods_B_and_D_are_correct :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y → methodB x y ∧ methodD x y :=
sorry

end NUMINAMATH_CALUDE_methods_B_and_D_are_correct_l3322_332265
