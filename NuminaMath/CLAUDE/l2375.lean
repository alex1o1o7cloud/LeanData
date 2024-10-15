import Mathlib

namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2375_237518

def U : Finset Int := {-3, -2, -1, 0, 1}
def A : Finset Int := {-2, -1}
def B : Finset Int := {-3, -1, 0}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {-3, 0} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2375_237518


namespace NUMINAMATH_CALUDE_hypothetical_town_population_l2375_237528

theorem hypothetical_town_population : ∃ n : ℕ, 
  (∃ m k : ℕ, 
    n^2 + 150 = m^2 + 1 ∧ 
    n^2 + 300 = k^2) ∧ 
  n^2 = 5476 := by
  sorry

end NUMINAMATH_CALUDE_hypothetical_town_population_l2375_237528


namespace NUMINAMATH_CALUDE_extended_hexagon_area_l2375_237585

structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ
  area : ℝ

def extend_hexagon (h : Hexagon) : Hexagon := sorry

theorem extended_hexagon_area (h : Hexagon) 
  (side_lengths : Fin 6 → ℝ)
  (h_sides : ∀ i, dist (h.vertices i) (h.vertices ((i + 1) % 6)) = side_lengths i)
  (h_area : h.area = 30)
  (h_side_lengths : side_lengths = ![3, 4, 5, 6, 7, 8]) :
  (extend_hexagon h).area = 90 := by
  sorry

end NUMINAMATH_CALUDE_extended_hexagon_area_l2375_237585


namespace NUMINAMATH_CALUDE_tagged_ratio_is_one_thirtieth_l2375_237545

/-- Represents the fish population in a pond -/
structure FishPopulation where
  initialTagged : ℕ
  secondCatchTotal : ℕ
  secondCatchTagged : ℕ
  estimatedTotal : ℕ

/-- Calculates the ratio of tagged fish to total fish in the second catch -/
def taggedRatio (fp : FishPopulation) : ℚ :=
  fp.secondCatchTagged / fp.secondCatchTotal

/-- The specific fish population described in the problem -/
def pondPopulation : FishPopulation :=
  { initialTagged := 60
  , secondCatchTotal := 60
  , secondCatchTagged := 2
  , estimatedTotal := 1800 }

/-- Theorem stating that the ratio of tagged fish to total fish in the second catch is 1/30 -/
theorem tagged_ratio_is_one_thirtieth :
  taggedRatio pondPopulation = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_tagged_ratio_is_one_thirtieth_l2375_237545


namespace NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l2375_237576

theorem p_or_q_necessary_not_sufficient (p q : Prop) : 
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l2375_237576


namespace NUMINAMATH_CALUDE_betty_age_l2375_237525

theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 14) :
  betty = 7 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l2375_237525


namespace NUMINAMATH_CALUDE_theater_revenue_l2375_237566

theorem theater_revenue (n : ℕ) (C : ℝ) :
  (∃ R : ℝ, R = 1.20 * C) →
  (∃ R_95 : ℝ, R_95 = 0.95 * 1.20 * C ∧ R_95 = 1.14 * C) :=
by sorry

end NUMINAMATH_CALUDE_theater_revenue_l2375_237566


namespace NUMINAMATH_CALUDE_three_digit_permutation_property_l2375_237533

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_permutations (n : ℕ) : List ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  [100*a + 10*b + c, 100*a + 10*c + b, 100*b + 10*a + c, 100*b + 10*c + a, 100*c + 10*a + b, 100*c + 10*b + a]

def satisfies_property (n : ℕ) : Prop :=
  is_three_digit n ∧ (List.sum (digit_permutations n)) / 6 = n

def solution_set : List ℕ := [111, 222, 333, 444, 555, 666, 777, 888, 999, 407, 518, 629, 370, 481, 592]

theorem three_digit_permutation_property :
  ∀ n : ℕ, satisfies_property n ↔ n ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_three_digit_permutation_property_l2375_237533


namespace NUMINAMATH_CALUDE_rational_cube_sum_zero_l2375_237515

theorem rational_cube_sum_zero (x y z : ℚ) 
  (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_sum_zero_l2375_237515


namespace NUMINAMATH_CALUDE_rogers_age_multiple_rogers_age_multiple_is_two_l2375_237573

/-- Proves that the multiple of Jill's age that relates to Roger's age is 2 -/
theorem rogers_age_multiple : ℕ → Prop := fun m =>
  let jill_age : ℕ := 20
  let finley_age : ℕ := 40
  let years_passed : ℕ := 15
  let roger_age : ℕ := m * jill_age + 5
  let jill_future_age : ℕ := jill_age + years_passed
  let roger_future_age : ℕ := roger_age + years_passed
  let finley_future_age : ℕ := finley_age + years_passed
  let future_age_difference : ℕ := roger_future_age - jill_future_age
  (future_age_difference = finley_future_age - 30) → (m = 2)

/-- The theorem holds for m = 2 -/
theorem rogers_age_multiple_is_two : rogers_age_multiple 2 := by
  sorry

end NUMINAMATH_CALUDE_rogers_age_multiple_rogers_age_multiple_is_two_l2375_237573


namespace NUMINAMATH_CALUDE_check_to_new_balance_ratio_l2375_237557

def initial_balance : ℚ := 150
def check_amount : ℚ := 50

def new_balance : ℚ := initial_balance + check_amount

theorem check_to_new_balance_ratio :
  check_amount / new_balance = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_check_to_new_balance_ratio_l2375_237557


namespace NUMINAMATH_CALUDE_slope_value_l2375_237549

/-- The slope of a line passing through a focus of the ellipse x^2 + 2y^2 = 3 
    and intersecting it at two points with distance 2 apart. -/
def slope_through_focus (k : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ),
    -- A and B are on the ellipse
    A.1^2 + 2*A.2^2 = 3 ∧ B.1^2 + 2*B.2^2 = 3 ∧
    -- The line passes through a focus
    ∃ (x : ℝ), x^2 = 3/2 ∧ (A.2 - 0) = k * (A.1 - x) ∧ (B.2 - 0) = k * (B.1 - x) ∧
    -- The distance between A and B is 2
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4

/-- The theorem stating the absolute value of the slope -/
theorem slope_value : ∀ k : ℝ, slope_through_focus k → k^2 = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_value_l2375_237549


namespace NUMINAMATH_CALUDE_shoe_selection_probability_l2375_237503

theorem shoe_selection_probability (num_pairs : ℕ) (prob : ℚ) : 
  num_pairs = 8 ∧ 
  prob = 1/15 ∧
  (∃ (total : ℕ), 
    (num_pairs * 2 : ℚ) / (total * (total - 1)) = prob) →
  ∃ (total : ℕ), total = 16 ∧ 
    (num_pairs * 2 : ℚ) / (total * (total - 1)) = prob :=
by sorry

end NUMINAMATH_CALUDE_shoe_selection_probability_l2375_237503


namespace NUMINAMATH_CALUDE_inequality_proof_l2375_237578

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2375_237578


namespace NUMINAMATH_CALUDE_spike_morning_crickets_l2375_237590

/-- The number of crickets Spike hunts in the morning. -/
def morning_crickets : ℕ := 5

/-- The number of crickets Spike hunts in the afternoon and evening. -/
def afternoon_evening_crickets : ℕ := 3 * morning_crickets

/-- The total number of crickets Spike hunts per day. -/
def total_crickets : ℕ := 20

/-- Theorem stating that the number of crickets Spike hunts in the morning is 5. -/
theorem spike_morning_crickets :
  morning_crickets = 5 ∧
  afternoon_evening_crickets = 3 * morning_crickets ∧
  total_crickets = morning_crickets + afternoon_evening_crickets ∧
  total_crickets = 20 :=
by sorry

end NUMINAMATH_CALUDE_spike_morning_crickets_l2375_237590


namespace NUMINAMATH_CALUDE_expression_simplification_l2375_237502

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) + ((x^3 - 1) / y) * ((y^3 - 1) / x) = 2 * x * y^2 + 2 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2375_237502


namespace NUMINAMATH_CALUDE_movie_deal_savings_l2375_237552

theorem movie_deal_savings : 
  let deal_price : ℚ := 20
  let movie_price : ℚ := 8
  let popcorn_price : ℚ := movie_price - 3
  let drink_price : ℚ := popcorn_price + 1
  let candy_price : ℚ := drink_price / 2
  let total_price : ℚ := movie_price + popcorn_price + drink_price + candy_price
  total_price - deal_price = 2 := by sorry

end NUMINAMATH_CALUDE_movie_deal_savings_l2375_237552


namespace NUMINAMATH_CALUDE_smith_family_mean_age_l2375_237531

def smith_family_ages : List ℕ := [8, 8, 8, 12, 11, 3, 4]

theorem smith_family_mean_age :
  (smith_family_ages.sum : ℚ) / smith_family_ages.length = 54 / 7 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_mean_age_l2375_237531


namespace NUMINAMATH_CALUDE_viewing_time_theorem_l2375_237598

/-- Represents the duration of the show in minutes -/
def show_duration : ℕ := 30

/-- Represents the number of days Max watches the show -/
def days_watched : ℕ := 4

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℚ :=
  (minutes : ℚ) / 60

/-- Theorem stating that watching a 30-minute show for 4 days results in 2 hours of viewing time -/
theorem viewing_time_theorem :
  minutes_to_hours (show_duration * days_watched) = 2 := by
  sorry

end NUMINAMATH_CALUDE_viewing_time_theorem_l2375_237598


namespace NUMINAMATH_CALUDE_cube_root_two_not_expressible_l2375_237567

theorem cube_root_two_not_expressible : ¬ ∃ (p q r : ℚ), (2 : ℝ)^(1/3) = p + q * (r^(1/2)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_two_not_expressible_l2375_237567


namespace NUMINAMATH_CALUDE_product_expansion_l2375_237532

theorem product_expansion (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2375_237532


namespace NUMINAMATH_CALUDE_solve_sqrt_equation_l2375_237599

theorem solve_sqrt_equation (x : ℝ) :
  Real.sqrt ((3 / x) + 3) = 5/3 → x = -27/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_sqrt_equation_l2375_237599


namespace NUMINAMATH_CALUDE_dennis_lives_on_sixth_floor_l2375_237572

def frank_floor : ℕ := 16

def charlie_floor (frank : ℕ) : ℕ := frank / 4

def dennis_floor (charlie : ℕ) : ℕ := charlie + 2

theorem dennis_lives_on_sixth_floor :
  dennis_floor (charlie_floor frank_floor) = 6 := by
  sorry

end NUMINAMATH_CALUDE_dennis_lives_on_sixth_floor_l2375_237572


namespace NUMINAMATH_CALUDE_screen_to_body_ratio_increases_l2375_237524

theorem screen_to_body_ratio_increases
  (b a m : ℝ)
  (h1 : 0 < b)
  (h2 : b < a)
  (h3 : 0 < m) :
  b / a < (b + m) / (a + m) :=
by sorry

end NUMINAMATH_CALUDE_screen_to_body_ratio_increases_l2375_237524


namespace NUMINAMATH_CALUDE_sqrt_x_minus_four_defined_l2375_237521

theorem sqrt_x_minus_four_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 4) ↔ x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_four_defined_l2375_237521


namespace NUMINAMATH_CALUDE_shaded_area_is_12_5_l2375_237500

-- Define the rectangle and its properties
def rectangle_JKLM (J K L M : ℝ × ℝ) : Prop :=
  K.1 = 0 ∧ K.2 = 0 ∧
  L.1 = 5 ∧ L.2 = 0 ∧
  M.1 = 5 ∧ M.2 = 6 ∧
  J.1 = 0 ∧ J.2 = 6

-- Define the additional points I, Q, and N
def point_I (I : ℝ × ℝ) : Prop := I.1 = 0 ∧ I.2 = 5
def point_Q (Q : ℝ × ℝ) : Prop := Q.1 = 5 ∧ Q.2 = 5
def point_N (N : ℝ × ℝ) : Prop := N.1 = 2.5 ∧ N.2 = 3

-- Define the lines JM and LK
def line_JM (J M : ℝ × ℝ) (x y : ℝ) : Prop :=
  y = (6 / 5) * x

def line_LK (L K : ℝ × ℝ) (x y : ℝ) : Prop :=
  y = -(6 / 5) * x + 6

-- Define the areas of trapezoid KQNM and triangle IKN
def area_KQNM (K Q N M : ℝ × ℝ) : ℝ := 11.25
def area_IKN (I K N : ℝ × ℝ) : ℝ := 1.25

-- Theorem statement
theorem shaded_area_is_12_5
  (J K L M I Q N : ℝ × ℝ)
  (h_rect : rectangle_JKLM J K L M)
  (h_I : point_I I)
  (h_Q : point_Q Q)
  (h_N : point_N N)
  (h_JM : line_JM J M N.1 N.2)
  (h_LK : line_LK L K N.1 N.2)
  : area_KQNM K Q N M + area_IKN I K N = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_is_12_5_l2375_237500


namespace NUMINAMATH_CALUDE_edward_total_money_l2375_237550

-- Define the variables
def dollars_per_lawn : ℕ := 8
def lawns_mowed : ℕ := 5
def initial_savings : ℕ := 7

-- Define the theorem
theorem edward_total_money :
  dollars_per_lawn * lawns_mowed + initial_savings = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_edward_total_money_l2375_237550


namespace NUMINAMATH_CALUDE_valid_configuration_iff_consecutive_adjacent_l2375_237509

/-- Represents a cell in the 4x4 grid --/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents a configuration of numbers in the 4x4 grid --/
def Configuration := Cell → Option ℕ

/-- Checks if two cells are adjacent --/
def adjacent (c1 c2 : Cell) : Bool :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- Checks if a configuration is valid --/
def is_valid (config : Configuration) : Prop :=
  ∀ (c1 c2 : Cell),
    match config c1, config c2 with
    | some n1, some n2 =>
        if n1 + 1 = n2 ∨ n2 + 1 = n1 then
          adjacent c1 c2
        else
          true
    | _, _ => true

/-- Theorem: A configuration is valid if and only if all pairs of consecutive numbers
    present in the grid are in adjacent cells --/
theorem valid_configuration_iff_consecutive_adjacent (config : Configuration) :
  is_valid config ↔
  (∀ (c1 c2 : Cell),
    match config c1, config c2 with
    | some n1, some n2 =>
        if n1 + 1 = n2 ∨ n2 + 1 = n1 then
          adjacent c1 c2
        else
          true
    | _, _ => true) :=
by sorry


end NUMINAMATH_CALUDE_valid_configuration_iff_consecutive_adjacent_l2375_237509


namespace NUMINAMATH_CALUDE_rectangular_prism_edge_pairs_l2375_237542

/-- A rectangular prism -/
structure RectangularPrism where
  edges : Finset Edge
  faces : Finset Face

/-- An edge of a rectangular prism -/
structure Edge where
  -- Add necessary fields

/-- A face of a rectangular prism -/
structure Face where
  -- Add necessary fields

/-- Two edges are parallel -/
def parallel (e1 e2 : Edge) : Prop := sorry

/-- Two edges are perpendicular -/
def perpendicular (e1 e2 : Edge) : Prop := sorry

/-- The set of pairs of parallel edges in a rectangular prism -/
def parallelEdgePairs (rp : RectangularPrism) : Finset (Edge × Edge) :=
  sorry

/-- The set of pairs of perpendicular edges in a rectangular prism -/
def perpendicularEdgePairs (rp : RectangularPrism) : Finset (Edge × Edge) :=
  sorry

theorem rectangular_prism_edge_pairs (rp : RectangularPrism) :
  (Finset.card (parallelEdgePairs rp) = 8) ∧
  (Finset.card (perpendicularEdgePairs rp) = 20) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_edge_pairs_l2375_237542


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2375_237513

/-- Given a circle with equation x^2 + y^2 + 2x - 4y - 6 = 0, 
    its center is at (-1, 2) and its radius is √11 -/
theorem circle_center_and_radius :
  let circle_eq := (fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y - 6 = 0)
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧ 
    radius = Real.sqrt 11 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2375_237513


namespace NUMINAMATH_CALUDE_difference_of_squares_l2375_237522

theorem difference_of_squares (x : ℝ) : x^2 - 25 = (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2375_237522


namespace NUMINAMATH_CALUDE_largest_root_equation_l2375_237596

theorem largest_root_equation (a b c d : ℝ) 
  (h1 : a + d = 2022)
  (h2 : b + c = 2022)
  (h3 : a ≠ c) :
  ∃ x : ℝ, (x - a) * (x - b) = (x - c) * (x - d) ∧ 
    x = 1011 ∧
    ∀ y : ℝ, (y - a) * (y - b) = (y - c) * (y - d) → y ≤ 1011 :=
by sorry

end NUMINAMATH_CALUDE_largest_root_equation_l2375_237596


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l2375_237594

theorem add_preserves_inequality (a b c : ℝ) : a < b → a + c < b + c := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l2375_237594


namespace NUMINAMATH_CALUDE_tournament_512_players_games_l2375_237577

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  initial_players : ℕ
  games_played : ℕ

/-- Calculates the number of games required to determine a champion. -/
def games_required (tournament : SingleEliminationTournament) : ℕ :=
  tournament.initial_players - 1

/-- Theorem stating that a tournament with 512 initial players requires 511 games. -/
theorem tournament_512_players_games (tournament : SingleEliminationTournament) 
    (h : tournament.initial_players = 512) : 
    games_required tournament = 511 := by
  sorry

#eval games_required ⟨512, 0⟩

end NUMINAMATH_CALUDE_tournament_512_players_games_l2375_237577


namespace NUMINAMATH_CALUDE_inequality_proof_l2375_237570

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 3) :
  Real.sqrt (1 + a) + Real.sqrt (1 + b) ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2375_237570


namespace NUMINAMATH_CALUDE_classroom_pencils_l2375_237530

theorem classroom_pencils (num_children : ℕ) (pencils_per_child : ℕ) 
  (h1 : num_children = 10) (h2 : pencils_per_child = 5) : 
  num_children * pencils_per_child = 50 := by
  sorry

end NUMINAMATH_CALUDE_classroom_pencils_l2375_237530


namespace NUMINAMATH_CALUDE_odds_against_C_winning_l2375_237504

-- Define the type for horses
inductive Horse : Type
| A
| B
| C

-- Define the function for odds against winning
def oddsAgainst (h : Horse) : ℚ :=
  match h with
  | Horse.A => 4 / 1
  | Horse.B => 3 / 2
  | Horse.C => 3 / 2  -- This is what we want to prove

-- State the theorem
theorem odds_against_C_winning :
  (∀ h₁ h₂ : Horse, h₁ ≠ h₂ → oddsAgainst h₁ ≠ oddsAgainst h₂) →  -- No ties
  oddsAgainst Horse.A = 4 / 1 →
  oddsAgainst Horse.B = 3 / 2 →
  oddsAgainst Horse.C = 3 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_odds_against_C_winning_l2375_237504


namespace NUMINAMATH_CALUDE_soldier_movement_l2375_237519

theorem soldier_movement (n : ℕ) :
  (∃ (initial_config : Fin (n + 2) → Fin n → Bool)
     (final_config : Fin n → Fin (n + 2) → Bool),
   (∀ i j, initial_config i j → 
     ∃ i' j', final_config i' j' ∧ 
       ((i' = i ∧ j' = j) ∨ 
        (i'.val + 1 = i.val ∧ j' = j) ∨ 
        (i'.val = i.val + 1 ∧ j' = j) ∨ 
        (i' = i ∧ j'.val + 1 = j.val) ∨ 
        (i' = i ∧ j'.val = j.val + 1))) ∧
   (∀ i j, initial_config i j ↔ true) ∧
   (∀ i j, final_config i j ↔ true)) →
  Even n :=
by sorry

end NUMINAMATH_CALUDE_soldier_movement_l2375_237519


namespace NUMINAMATH_CALUDE_square_difference_equality_l2375_237554

theorem square_difference_equality : (36 + 9)^2 - (9^2 + 36^2) = 648 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2375_237554


namespace NUMINAMATH_CALUDE_rotation_of_doubled_complex_l2375_237517

theorem rotation_of_doubled_complex :
  let z : ℂ := 3 - 4*I
  let doubled : ℂ := 2 * z
  let rotated : ℂ := -doubled
  rotated = -6 + 8*I :=
by
  sorry

end NUMINAMATH_CALUDE_rotation_of_doubled_complex_l2375_237517


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_quarter_sector_l2375_237593

/-- The radius of an inscribed circle in a quarter circle sector -/
theorem inscribed_circle_radius_quarter_sector (r : ℝ) (h : r = 5) :
  let inscribed_radius := r * (Real.sqrt 2 - 1)
  inscribed_radius = 5 * Real.sqrt 2 - 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_quarter_sector_l2375_237593


namespace NUMINAMATH_CALUDE_window_offer_savings_l2375_237505

/-- Represents the window offer structure -/
structure WindowOffer where
  normalPrice : ℕ
  purchaseCount : ℕ
  freeCount : ℕ

/-- Calculates the cost for a given number of windows under the offer -/
def costUnderOffer (offer : WindowOffer) (windowCount : ℕ) : ℕ :=
  let fullSets := windowCount / (offer.purchaseCount + offer.freeCount)
  let remainingWindows := windowCount % (offer.purchaseCount + offer.freeCount)
  (fullSets * offer.purchaseCount + min remainingWindows offer.purchaseCount) * offer.normalPrice

/-- Calculates the savings when purchasing windows together vs separately -/
def calculateSavings (offer : WindowOffer) (dave : ℕ) (doug : ℕ) : ℕ :=
  let separateCost := costUnderOffer offer dave + costUnderOffer offer doug
  let combinedCost := costUnderOffer offer (dave + doug)
  (dave + doug) * offer.normalPrice - combinedCost

/-- The main theorem stating the savings amount -/
theorem window_offer_savings :
  let offer : WindowOffer := ⟨100, 6, 2⟩
  let davesWindows : ℕ := 9
  let dougsWindows : ℕ := 10
  calculateSavings offer davesWindows dougsWindows = 400 := by
  sorry

end NUMINAMATH_CALUDE_window_offer_savings_l2375_237505


namespace NUMINAMATH_CALUDE_west_movement_representation_l2375_237562

/-- Represents direction of movement --/
inductive Direction
  | East
  | West

/-- Represents a movement with magnitude and direction --/
structure Movement where
  magnitude : ℝ
  direction : Direction

/-- Converts a movement to its coordinate representation --/
def toCoordinate (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.magnitude
  | Direction.West => -m.magnitude

theorem west_movement_representation :
  let westMovement : Movement := ⟨80, Direction.West⟩
  toCoordinate westMovement = -80 := by sorry

end NUMINAMATH_CALUDE_west_movement_representation_l2375_237562


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l2375_237587

/-- The volume of a sphere inscribed in a cube with edge length 8 inches -/
theorem volume_of_inscribed_sphere (π : ℝ) :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = (256 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l2375_237587


namespace NUMINAMATH_CALUDE_red_face_probability_l2375_237523

/-- A cube with colored faces -/
structure ColoredCube where
  redFaces : Nat
  blueFaces : Nat
  is_cube : redFaces + blueFaces = 6

/-- The probability of rolling a specific color on a colored cube -/
def rollProbability (cube : ColoredCube) (color : Nat) : Rat :=
  color / 6

/-- Theorem: The probability of rolling a red face on a cube with 5 red faces and 1 blue face is 5/6 -/
theorem red_face_probability :
  ∀ (cube : ColoredCube), cube.redFaces = 5 → cube.blueFaces = 1 →
  rollProbability cube cube.redFaces = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_red_face_probability_l2375_237523


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l2375_237541

/-- Calculates the time for a train to pass a platform -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (time_to_pass_point : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_to_pass_point = 120)
  (h3 : platform_length = 1000) :
  (train_length + platform_length) / (train_length / time_to_pass_point) = 220 := by
  sorry

#check train_platform_passing_time

end NUMINAMATH_CALUDE_train_platform_passing_time_l2375_237541


namespace NUMINAMATH_CALUDE_fifth_root_unity_product_l2375_237579

/-- Given a complex number z that is a fifth root of unity, 
    prove that the product (1 - z)(1 - z^2)(1 - z^3)(1 - z^4) equals 5 -/
theorem fifth_root_unity_product (z : ℂ) 
  (h : z = Complex.exp (2 * Real.pi * I / 5)) : 
  (1 - z) * (1 - z^2) * (1 - z^3) * (1 - z^4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_unity_product_l2375_237579


namespace NUMINAMATH_CALUDE_pizza_combinations_l2375_237544

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2375_237544


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l2375_237597

theorem unique_two_digit_integer (u : ℕ) : 
  (u ≥ 10 ∧ u ≤ 99) →  -- u is a two-digit positive integer
  (13 * u) % 100 = 52 →  -- when multiplied by 13, the last two digits are 52
  u = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l2375_237597


namespace NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l2375_237556

def point_A : ℝ × ℝ := (2, -3)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_A_in_fourth_quadrant : in_fourth_quadrant point_A := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l2375_237556


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2375_237546

/-- The eccentricity of an ellipse with given conditions is between 0 and 2√5/5 -/
theorem ellipse_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := Real.sqrt (1 - b^2 / a^2)
  let l := {p : ℝ × ℝ | p.2 = 1/2 * (p.1 + a)}
  let C₁ := {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  let C₂ := {p : ℝ × ℝ | p.1^2 + p.2^2 = b^2}
  (∃ p q : ℝ × ℝ, p ≠ q ∧ p ∈ l ∩ C₂ ∧ q ∈ l ∩ C₂) →
  0 < e ∧ e < 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2375_237546


namespace NUMINAMATH_CALUDE_candy_store_sales_theorem_l2375_237514

/-- Represents the sales data of a candy store -/
structure CandyStoreSales where
  fudgePounds : ℕ
  fudgePrice : ℚ
  trufflesDozens : ℕ
  trufflePrice : ℚ
  pretzelsDozens : ℕ
  pretzelPrice : ℚ

/-- Calculates the total money made by the candy store -/
def totalMoney (sales : CandyStoreSales) : ℚ :=
  sales.fudgePounds * sales.fudgePrice +
  sales.trufflesDozens * 12 * sales.trufflePrice +
  sales.pretzelsDozens * 12 * sales.pretzelPrice

/-- Theorem stating that the candy store made $212.00 -/
theorem candy_store_sales_theorem (sales : CandyStoreSales) 
  (h1 : sales.fudgePounds = 20)
  (h2 : sales.fudgePrice = 5/2)
  (h3 : sales.trufflesDozens = 5)
  (h4 : sales.trufflePrice = 3/2)
  (h5 : sales.pretzelsDozens = 3)
  (h6 : sales.pretzelPrice = 2) :
  totalMoney sales = 212 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_sales_theorem_l2375_237514


namespace NUMINAMATH_CALUDE_problem_1_l2375_237581

theorem problem_1 : (-1/12) / (-1/2 + 2/3 + 3/4) = -1/11 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2375_237581


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2375_237547

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let f (x y : ℝ) := x^2 / a^2 - y^2 / b^2 - 1
  ∃ (x y : ℝ), f x y = 0 ∧ x > 0 ∧ x * c = 0 ∧ 2 * c = y * a / b ∧ c^2 = a^2 + b^2
  → c / a = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2375_237547


namespace NUMINAMATH_CALUDE_apple_pricing_l2375_237565

/-- The price function for apples -/
noncomputable def price (l q x : ℝ) (k : ℝ) : ℝ :=
  if k ≤ x then l * k else l * x + q * (k - x)

theorem apple_pricing (l q x : ℝ) : 
  (price l q x 33 = 11.67) →
  (price l q x 36 = 12.48) →
  (price l q x 10 = 3.62) →
  (x = 30) := by
sorry

end NUMINAMATH_CALUDE_apple_pricing_l2375_237565


namespace NUMINAMATH_CALUDE_tiffany_max_points_l2375_237527

/-- Represents the ring toss game --/
structure RingTossGame where
  total_money : ℕ
  cost_per_play : ℕ
  rings_per_play : ℕ
  red_points : ℕ
  green_points : ℕ
  games_played : ℕ
  red_buckets_hit : ℕ
  green_buckets_hit : ℕ

/-- Calculates the maximum points possible for the given game state --/
def max_points (game : RingTossGame) : ℕ :=
  let points_so_far := game.red_buckets_hit * game.red_points + game.green_buckets_hit * game.green_points
  let remaining_games := game.total_money / game.cost_per_play - game.games_played
  let max_additional_points := remaining_games * game.rings_per_play * game.green_points
  points_so_far + max_additional_points

/-- Theorem stating that the maximum points Tiffany can get is 38 --/
theorem tiffany_max_points :
  let game := RingTossGame.mk 3 1 5 2 3 2 4 5
  max_points game = 38 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_max_points_l2375_237527


namespace NUMINAMATH_CALUDE_complement_union_equals_divisible_by_3_l2375_237506

-- Define the universal set U as the set of all integers
def U : Set ℤ := Set.univ

-- Define set A
def A : Set ℤ := {x | ∃ k : ℤ, x = 3*k + 1}

-- Define set B
def B : Set ℤ := {x | ∃ k : ℤ, x = 3*k + 2}

-- Define the set of integers divisible by 3
def DivisibleBy3 : Set ℤ := {x | ∃ k : ℤ, x = 3*k}

-- Theorem statement
theorem complement_union_equals_divisible_by_3 :
  (U \ (A ∪ B)) = DivisibleBy3 :=
sorry

end NUMINAMATH_CALUDE_complement_union_equals_divisible_by_3_l2375_237506


namespace NUMINAMATH_CALUDE_minimum_loaves_needed_l2375_237553

def slices_per_loaf : ℕ := 20
def regular_sandwich_slices : ℕ := 2
def double_meat_sandwich_slices : ℕ := 3
def triple_decker_sandwich_slices : ℕ := 4
def club_sandwich_slices : ℕ := 5
def regular_sandwiches : ℕ := 25
def double_meat_sandwiches : ℕ := 18
def triple_decker_sandwiches : ℕ := 12
def club_sandwiches : ℕ := 8

theorem minimum_loaves_needed : 
  ∃ (loaves : ℕ), 
    loaves * slices_per_loaf = 
      regular_sandwiches * regular_sandwich_slices +
      double_meat_sandwiches * double_meat_sandwich_slices +
      triple_decker_sandwiches * triple_decker_sandwich_slices +
      club_sandwiches * club_sandwich_slices ∧
    loaves = 10 := by
  sorry

end NUMINAMATH_CALUDE_minimum_loaves_needed_l2375_237553


namespace NUMINAMATH_CALUDE_smallest_sum_divisible_by_2016_l2375_237584

theorem smallest_sum_divisible_by_2016 :
  ∃ (n₁ n₂ n₃ n₄ n₅ n₆ n₇ : ℕ),
    0 < n₁ ∧ n₁ < n₂ ∧ n₂ < n₃ ∧ n₃ < n₄ ∧ n₄ < n₅ ∧ n₅ < n₆ ∧ n₆ < n₇ ∧
    (n₁ * n₂ * n₃ * n₄ * n₅ * n₆ * n₇) % 2016 = 0 ∧
    n₁ + n₂ + n₃ + n₄ + n₅ + n₆ + n₇ = 31 ∧
    ∀ (m₁ m₂ m₃ m₄ m₅ m₆ m₇ : ℕ),
      0 < m₁ ∧ m₁ < m₂ ∧ m₂ < m₃ ∧ m₃ < m₄ ∧ m₄ < m₅ ∧ m₅ < m₆ ∧ m₆ < m₇ →
      (m₁ * m₂ * m₃ * m₄ * m₅ * m₆ * m₇) % 2016 = 0 →
      m₁ + m₂ + m₃ + m₄ + m₅ + m₆ + m₇ ≥ 31 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_divisible_by_2016_l2375_237584


namespace NUMINAMATH_CALUDE_count_triangles_in_polygon_l2375_237580

/-- The number of triangles in a regular n-sided polygon (n ≥ 6) whose sides are formed by diagonals
    and whose vertices are vertices of the polygon -/
def triangles_in_polygon (n : ℕ) : ℕ :=
  n * (n - 4) * (n - 5) / 6

/-- Theorem stating the number of triangles in a regular n-sided polygon (n ≥ 6) whose sides are formed
    by diagonals and whose vertices are vertices of the polygon -/
theorem count_triangles_in_polygon (n : ℕ) (h : n ≥ 6) :
  triangles_in_polygon n = n * (n - 4) * (n - 5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_count_triangles_in_polygon_l2375_237580


namespace NUMINAMATH_CALUDE_average_chapters_per_book_l2375_237508

theorem average_chapters_per_book 
  (total_chapters : Float) 
  (total_books : Float) 
  (h1 : total_chapters = 17.0) 
  (h2 : total_books = 4.0) : 
  total_chapters / total_books = 4.25 := by
sorry

end NUMINAMATH_CALUDE_average_chapters_per_book_l2375_237508


namespace NUMINAMATH_CALUDE_square_root_of_16_l2375_237561

theorem square_root_of_16 : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_16_l2375_237561


namespace NUMINAMATH_CALUDE_shifted_quadratic_equation_solutions_l2375_237569

/-- Given an equation a(x+m)²+b=0 with solutions x₁=-2 and x₂=1, 
    prove that a(x+m+2)²+b=0 has solutions x₁=-4 and x₂=-1 -/
theorem shifted_quadratic_equation_solutions 
  (a m b : ℝ) 
  (ha : a ≠ 0) 
  (h1 : a * ((-2 : ℝ) + m)^2 + b = 0) 
  (h2 : a * ((1 : ℝ) + m)^2 + b = 0) :
  a * ((-4 : ℝ) + m + 2)^2 + b = 0 ∧ a * ((-1 : ℝ) + m + 2)^2 + b = 0 :=
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_equation_solutions_l2375_237569


namespace NUMINAMATH_CALUDE_f_deriv_l2375_237540

/-- The function f(x) = 2x + 3 -/
def f (x : ℝ) : ℝ := 2 * x + 3

/-- Theorem: The derivative of f(x) = 2x + 3 is equal to 2 -/
theorem f_deriv : deriv f = λ _ => 2 := by sorry

end NUMINAMATH_CALUDE_f_deriv_l2375_237540


namespace NUMINAMATH_CALUDE_m_range_l2375_237511

/-- The range of m given the specified conditions -/
theorem m_range (m : ℝ) : 
  (¬ ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∧ 
  ((∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∨ 
   (∀ x : ℝ, x ≥ 2 → x + m/x - 2 > 0)) →
  0 < m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_m_range_l2375_237511


namespace NUMINAMATH_CALUDE_pens_given_to_sharon_l2375_237507

/-- The number of pens given to Sharon -/
def pens_to_sharon (initial_pens : ℕ) (mike_pens : ℕ) (final_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - final_pens

theorem pens_given_to_sharon :
  pens_to_sharon 20 22 65 = 19 := by
  sorry

end NUMINAMATH_CALUDE_pens_given_to_sharon_l2375_237507


namespace NUMINAMATH_CALUDE_great_circle_bisects_angle_l2375_237568

/-- A point on a sphere -/
structure SpherePoint where
  -- Add necessary fields

/-- A great circle on a sphere -/
structure GreatCircle where
  -- Add necessary fields

/-- The North Pole -/
def NorthPole : SpherePoint :=
  sorry

/-- The equator -/
def Equator : GreatCircle :=
  sorry

/-- Check if a point is on a great circle -/
def isOnGreatCircle (p : SpherePoint) (gc : GreatCircle) : Prop :=
  sorry

/-- Check if two points are equidistant from a third point -/
def areEquidistant (p1 p2 p3 : SpherePoint) : Prop :=
  sorry

/-- Check if a point is on the equator -/
def isOnEquator (p : SpherePoint) : Prop :=
  sorry

/-- The great circle through two points -/
def greatCircleThrough (p1 p2 : SpherePoint) : GreatCircle :=
  sorry

/-- Check if a great circle bisects an angle in a spherical triangle -/
def bisectsAngle (gc : GreatCircle) (p1 p2 p3 : SpherePoint) : Prop :=
  sorry

/-- Main theorem -/
theorem great_circle_bisects_angle (A B C : SpherePoint) :
  isOnGreatCircle A (greatCircleThrough NorthPole B) →
  isOnGreatCircle B (greatCircleThrough NorthPole A) →
  areEquidistant A B NorthPole →
  isOnEquator C →
  bisectsAngle (greatCircleThrough C NorthPole) A C B :=
by
  sorry

end NUMINAMATH_CALUDE_great_circle_bisects_angle_l2375_237568


namespace NUMINAMATH_CALUDE_distance_on_line_l2375_237563

/-- The distance between two points on a line --/
theorem distance_on_line (n m p q r s : ℝ) :
  q = n * p + m →
  s = n * r + m →
  Real.sqrt ((r - p)^2 + (s - q)^2) = |r - p| * Real.sqrt (1 + n^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l2375_237563


namespace NUMINAMATH_CALUDE_ralph_tv_time_l2375_237595

/-- Represents Ralph's TV watching schedule for a week -/
structure TVSchedule where
  weekdayHours : ℝ
  weekdayShows : ℕ × ℕ  -- (number of 1-hour shows, number of 30-minute shows)
  videoGameDays : ℕ
  weekendHours : ℝ
  weekendShows : ℕ × ℕ  -- (number of 1-hour shows, number of 45-minute shows)
  weekendBreak : ℝ

/-- Calculates the total TV watching time for a week given a TV schedule -/
def totalTVTime (schedule : TVSchedule) : ℝ :=
  let weekdayTotal := schedule.weekdayHours * 5
  let weekendTotal := (schedule.weekendHours - schedule.weekendBreak) * 2
  weekdayTotal + weekendTotal

/-- Ralph's actual TV schedule -/
def ralphSchedule : TVSchedule :=
  { weekdayHours := 3
  , weekdayShows := (1, 4)
  , videoGameDays := 3
  , weekendHours := 6
  , weekendShows := (3, 4)
  , weekendBreak := 0.5 }

/-- Theorem stating that Ralph's total TV watching time in one week is 26 hours -/
theorem ralph_tv_time : totalTVTime ralphSchedule = 26 := by
  sorry


end NUMINAMATH_CALUDE_ralph_tv_time_l2375_237595


namespace NUMINAMATH_CALUDE_rock_climbing_participants_number_of_rock_climbing_participants_l2375_237583

/- Define the total number of students in the school -/
def total_students : ℕ := 800

/- Define the percentage of students who went on the camping trip -/
def camping_percentage : ℚ := 25 / 100

/- Define the percentage of camping students who took more than $100 -/
def more_than_100_percentage : ℚ := 15 / 100

/- Define the percentage of camping students who took exactly $100 -/
def exactly_100_percentage : ℚ := 30 / 100

/- Define the percentage of camping students who took between $50 and $100 -/
def between_50_and_100_percentage : ℚ := 40 / 100

/- Define the percentage of students with more than $100 who participated in rock climbing -/
def rock_climbing_participation_percentage : ℚ := 50 / 100

/- Theorem stating the number of students who participated in rock climbing -/
theorem rock_climbing_participants : ℕ := by
  sorry

/- Main theorem to prove -/
theorem number_of_rock_climbing_participants : rock_climbing_participants = 15 := by
  sorry

end NUMINAMATH_CALUDE_rock_climbing_participants_number_of_rock_climbing_participants_l2375_237583


namespace NUMINAMATH_CALUDE_equation_solutions_l2375_237560

theorem equation_solutions : 
  ∀ (m n : ℕ), 3 * 2^m + 1 = n^2 ↔ (m = 0 ∧ n = 2) ∨ (m = 3 ∧ n = 5) ∨ (m = 4 ∧ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2375_237560


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l2375_237537

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 3 * X^4 - 5 * X^3 + 6 * X^2 - 8 * X + 3
  let divisor : Polynomial ℚ := X^2 + X + 1
  let quotient : Polynomial ℚ := 3 * X^2 - 8 * X
  (dividend / divisor) = quotient := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l2375_237537


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l2375_237501

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := x^4 - 3*x^2 - 4
def g (x : ℝ) : ℝ := -x^4 + 3*x^2 + 2*x

-- State the theorem
theorem polynomial_sum_theorem : 
  ∀ x : ℝ, f x + g x = -4 + 2*x :=
by
  sorry

#check polynomial_sum_theorem

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l2375_237501


namespace NUMINAMATH_CALUDE_triangle_third_side_length_triangle_third_side_length_proof_l2375_237536

/-- Given a triangle with perimeter 160 and two sides of lengths 40 and 50,
    the length of the third side is 70. -/
theorem triangle_third_side_length : ℝ → ℝ → ℝ → Prop :=
  fun (perimeter side1 side2 : ℝ) =>
    perimeter = 160 ∧ side1 = 40 ∧ side2 = 50 →
    ∃ (side3 : ℝ), side3 = 70 ∧ perimeter = side1 + side2 + side3

/-- Proof of the theorem -/
theorem triangle_third_side_length_proof :
  triangle_third_side_length 160 40 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_triangle_third_side_length_proof_l2375_237536


namespace NUMINAMATH_CALUDE_inverse_composition_equals_target_l2375_237558

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 4

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := (x + 4) / 3

-- Theorem statement
theorem inverse_composition_equals_target : f_inv (f_inv 13) = 29 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_target_l2375_237558


namespace NUMINAMATH_CALUDE_weight_of_single_pencil_l2375_237543

/-- The weight of a single pencil given the weight of a dozen pencils -/
theorem weight_of_single_pencil (dozen_weight : ℝ) (h : dozen_weight = 182.88) :
  dozen_weight / 12 = 15.24 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_single_pencil_l2375_237543


namespace NUMINAMATH_CALUDE_solve_airport_distance_l2375_237555

/-- Represents the problem of calculating the distance to the airport --/
def airport_distance_problem (initial_speed : ℝ) (speed_increase : ℝ) (initial_time : ℝ) 
  (late_time : ℝ) (early_time : ℝ) : Prop :=
  ∃ (distance : ℝ) (total_time : ℝ),
    -- Initial part of the journey
    initial_speed * initial_time = initial_speed
    -- Total distance equation
    ∧ distance = initial_speed * (total_time + late_time)
    -- Remaining distance equation with increased speed
    ∧ distance - initial_speed * initial_time = (initial_speed + speed_increase) * (total_time - initial_time - early_time)
    -- The solution
    ∧ distance = 264

/-- The theorem stating the solution to the airport distance problem --/
theorem solve_airport_distance : 
  airport_distance_problem 45 20 1 0.75 0.75 := by
  sorry

end NUMINAMATH_CALUDE_solve_airport_distance_l2375_237555


namespace NUMINAMATH_CALUDE_remainder_mod_12_l2375_237589

theorem remainder_mod_12 : (1234^567 + 89^1011) % 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_12_l2375_237589


namespace NUMINAMATH_CALUDE_no_unbounded_phine_sequence_l2375_237548

/-- A phine sequence is a sequence of positive real numbers satisfying
    a_{n+2} = (a_{n+1} + a_{n-1}) / a_n for all n ≥ 2 -/
def IsPhine (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n ≥ 2, a (n + 2) = (a (n + 1) + a (n - 1)) / a n)

/-- There does not exist an unbounded phine sequence -/
theorem no_unbounded_phine_sequence :
  ¬ ∃ a : ℕ → ℝ, IsPhine a ∧ ∀ r : ℝ, ∃ n : ℕ, a n > r :=
sorry

end NUMINAMATH_CALUDE_no_unbounded_phine_sequence_l2375_237548


namespace NUMINAMATH_CALUDE_christine_distance_l2375_237571

/-- Calculates the distance traveled given speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Christine's distance traveled -/
theorem christine_distance :
  let speed : ℝ := 20
  let time : ℝ := 4
  distance_traveled speed time = 80 := by sorry

end NUMINAMATH_CALUDE_christine_distance_l2375_237571


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_neg_105_l2375_237574

/-- A quadratic function with given properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  has_minimum_20 : ∃ x, a * x^2 + b * x + c = 20 ∧ ∀ y, a * y^2 + b * y + c ≥ 20
  root_at_3 : a * 3^2 + b * 3 + c = 0
  root_at_7 : a * 7^2 + b * 7 + c = 0

/-- The sum of coefficients of a quadratic function with given properties is -105 -/
theorem sum_of_coefficients_is_neg_105 (f : QuadraticFunction) : 
  f.a + f.b + f.c = -105 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_neg_105_l2375_237574


namespace NUMINAMATH_CALUDE_ellipse_center_correct_l2375_237586

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  (3 * y + 6)^2 / 7^2 + (4 * x - 8)^2 / 6^2 = 1

/-- The center of the ellipse -/
def ellipse_center : ℝ × ℝ := (-2, 2)

/-- Theorem stating that ellipse_center is the center of the ellipse defined by ellipse_equation -/
theorem ellipse_center_correct :
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    ((y - ellipse_center.2)^2 / (7/3)^2 + (x - ellipse_center.1)^2 / (3/2)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_center_correct_l2375_237586


namespace NUMINAMATH_CALUDE_geography_quiz_correct_percentage_l2375_237510

theorem geography_quiz_correct_percentage (y : ℝ) (h : y > 0) :
  let total_questions := 8 * y
  let incorrect_answers := 2 * y - 3
  let correct_answers := total_questions - incorrect_answers
  let correct_percentage := (correct_answers / total_questions) * 100
  correct_percentage = 75 + 75 / (2 * y) :=
by sorry

end NUMINAMATH_CALUDE_geography_quiz_correct_percentage_l2375_237510


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptotes_l2375_237535

/-- The distance from the focus of the parabola x^2 = 8y to the asymptotes of the hyperbola x^2 - y^2/9 = 1 is √10 / 5 -/
theorem distance_focus_to_asymptotes :
  let parabola := {p : ℝ × ℝ | p.1^2 = 8 * p.2}
  let hyperbola := {p : ℝ × ℝ | p.1^2 - p.2^2 / 9 = 1}
  let focus : ℝ × ℝ := (0, 2)
  let asymptote (x : ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 ∨ p.2 = -3 * p.1}
  let distance (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) := 
    Real.sqrt (10) / 5
  ∀ p ∈ parabola, p.1^2 = 8 * p.2 →
  ∀ h ∈ hyperbola, h.1^2 - h.2^2 / 9 = 1 →
  distance focus (asymptote 0) = Real.sqrt 10 / 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptotes_l2375_237535


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l2375_237592

/-- Calculates the total number of penalty kicks in a soccer team drill -/
theorem soccer_penalty_kicks (total_players : ℕ) (goalies : ℕ) : 
  total_players = 25 → goalies = 4 → total_players * (goalies - 1) = 96 := by
  sorry

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l2375_237592


namespace NUMINAMATH_CALUDE_amusement_park_payment_l2375_237582

def regular_ticket_cost : ℕ := 109
def child_discount : ℕ := 5
def change_received : ℕ := 74

def family_ticket_cost : ℕ := 
  (regular_ticket_cost - child_discount) * 2 + regular_ticket_cost * 2

theorem amusement_park_payment : 
  family_ticket_cost + change_received = 500 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_payment_l2375_237582


namespace NUMINAMATH_CALUDE_albert_joshua_difference_l2375_237591

-- Define the number of rocks each person collected
def joshua_rocks : ℕ := 80
def jose_rocks : ℕ := joshua_rocks - 14
def albert_rocks : ℕ := jose_rocks + 20

-- Theorem statement
theorem albert_joshua_difference :
  albert_rocks - joshua_rocks = 6 := by
sorry

end NUMINAMATH_CALUDE_albert_joshua_difference_l2375_237591


namespace NUMINAMATH_CALUDE_exists_non_squareable_number_l2375_237538

/-- A complication is adding a single digit to a number. -/
def Complication := Nat → Nat

/-- Apply a sequence of complications to a number. -/
def applyComplications (n : Nat) (complications : List Complication) : Nat :=
  complications.foldl (fun acc c => c acc) n

/-- Check if a number is a perfect square. -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem exists_non_squareable_number : 
  ∃ n : Nat, ∀ complications : List Complication, 
    complications.length ≤ 100 → 
    ¬(isPerfectSquare (applyComplications n complications)) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_squareable_number_l2375_237538


namespace NUMINAMATH_CALUDE_zoo_problem_solution_l2375_237526

/-- Represents the number of animals in each exhibit -/
structure ZooExhibits where
  rainForest : ℕ
  reptileHouse : ℕ
  aquarium : ℕ
  aviary : ℕ
  mammalHouse : ℕ

/-- Checks if the given numbers of animals satisfy the conditions of the zoo problem -/
def satisfiesZooConditions (exhibits : ZooExhibits) : Prop :=
  exhibits.reptileHouse = 3 * exhibits.rainForest - 5 ∧
  exhibits.reptileHouse = 16 ∧
  exhibits.aquarium = 2 * exhibits.reptileHouse ∧
  exhibits.aviary = (exhibits.aquarium - exhibits.rainForest) + 3 ∧
  exhibits.mammalHouse = ((exhibits.rainForest + exhibits.aquarium + exhibits.aviary) / 3 + 2)

/-- The theorem stating that there exists a unique solution to the zoo problem -/
theorem zoo_problem_solution : 
  ∃! exhibits : ZooExhibits, satisfiesZooConditions exhibits ∧ 
    exhibits.rainForest = 7 ∧ 
    exhibits.aquarium = 32 ∧ 
    exhibits.aviary = 28 ∧ 
    exhibits.mammalHouse = 24 :=
  sorry

end NUMINAMATH_CALUDE_zoo_problem_solution_l2375_237526


namespace NUMINAMATH_CALUDE_smaugs_hoard_l2375_237588

theorem smaugs_hoard (gold_coins : ℕ) (silver_coins : ℕ) (copper_coins : ℕ) 
  (silver_to_copper : ℕ) (total_value : ℕ) :
  gold_coins = 100 →
  silver_coins = 60 →
  copper_coins = 33 →
  silver_to_copper = 8 →
  total_value = 2913 →
  total_value = gold_coins * silver_to_copper * (silver_coins / gold_coins) + 
                silver_coins * silver_to_copper + 
                copper_coins →
  silver_coins / gold_coins = 3 := by
  sorry

end NUMINAMATH_CALUDE_smaugs_hoard_l2375_237588


namespace NUMINAMATH_CALUDE_quadratic_real_root_l2375_237559

/-- 
For a quadratic equation x^2 + bx + 9, the equation has at least one real root 
if and only if b belongs to the set (-∞, -6] ∪ [6, ∞)
-/
theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 9 = 0) ↔ b ≤ -6 ∨ b ≥ 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l2375_237559


namespace NUMINAMATH_CALUDE_volunteer_allocation_schemes_l2375_237512

def num_volunteers : ℕ := 5
def num_projects : ℕ := 4

theorem volunteer_allocation_schemes :
  (num_volunteers.choose 2) * (num_projects!) = 240 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_allocation_schemes_l2375_237512


namespace NUMINAMATH_CALUDE_min_value_expression_l2375_237575

theorem min_value_expression (a b : ℝ) (h : a - b^2 = 4) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x y : ℝ), x - y^2 = 4 → x^2 - 3*y^2 + x - 14 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2375_237575


namespace NUMINAMATH_CALUDE_solutions_count_l2375_237529

/-- The number of solutions to the equation √(x+3) = ax + 2 depends on the value of parameter a -/
theorem solutions_count (a : ℝ) : 
  (∀ x, Real.sqrt (x + 3) ≠ a * x + 2) ∨ 
  (∃! x, Real.sqrt (x + 3) = a * x + 2) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ Real.sqrt (x₁ + 3) = a * x₁ + 2 ∧ Real.sqrt (x₂ + 3) = a * x₂ + 2 ∧ 
    ∀ x, Real.sqrt (x + 3) = a * x + 2 → x = x₁ ∨ x = x₂) ∨
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    Real.sqrt (x₁ + 3) = a * x₁ + 2 ∧ Real.sqrt (x₂ + 3) = a * x₂ + 2 ∧ Real.sqrt (x₃ + 3) = a * x₃ + 2 ∧
    ∀ x, Real.sqrt (x + 3) = a * x + 2 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∨
  (∃ x₁ x₂ x₃ x₄, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    Real.sqrt (x₁ + 3) = a * x₁ + 2 ∧ Real.sqrt (x₂ + 3) = a * x₂ + 2 ∧ 
    Real.sqrt (x₃ + 3) = a * x₃ + 2 ∧ Real.sqrt (x₄ + 3) = a * x₄ + 2 ∧
    ∀ x, Real.sqrt (x + 3) = a * x + 2 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) :=
by
  sorry

end NUMINAMATH_CALUDE_solutions_count_l2375_237529


namespace NUMINAMATH_CALUDE_gdp_equality_l2375_237564

/-- Represents the GDP value in billions of yuan -/
def gdp_billions : ℝ := 4504.5

/-- Represents the same GDP value in scientific notation -/
def gdp_scientific : ℝ := 4.5045 * (10 ^ 12)

/-- Theorem stating that the GDP value in billions is equal to its scientific notation representation -/
theorem gdp_equality : gdp_billions * (10 ^ 9) = gdp_scientific := by sorry

end NUMINAMATH_CALUDE_gdp_equality_l2375_237564


namespace NUMINAMATH_CALUDE_function_domain_implies_k_range_l2375_237539

theorem function_domain_implies_k_range
  (a : ℝ) (k : ℝ)
  (h_a_pos : a > 0)
  (h_a_neq_one : a ≠ 1)
  (h_defined : ∀ x : ℝ, x^2 - 2*k*x + 2*k + 3 > 0) :
  -1 < k ∧ k < 3 :=
sorry

end NUMINAMATH_CALUDE_function_domain_implies_k_range_l2375_237539


namespace NUMINAMATH_CALUDE_area_remaining_after_iterations_l2375_237551

/-- The fraction of area that remains after each iteration -/
def remaining_fraction : ℚ := 3 / 4

/-- The number of iterations -/
def num_iterations : ℕ := 5

/-- The final fraction of the original area remaining -/
def final_fraction : ℚ := 243 / 1024

theorem area_remaining_after_iterations :
  remaining_fraction ^ num_iterations = final_fraction := by
  sorry

end NUMINAMATH_CALUDE_area_remaining_after_iterations_l2375_237551


namespace NUMINAMATH_CALUDE_min_value_of_f_l2375_237534

def f (x : ℝ) := x^2 + 14*x + 3

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2375_237534


namespace NUMINAMATH_CALUDE_fifth_plot_excess_tiles_l2375_237516

def plot_width (n : ℕ) : ℕ := 3 + 2 * (n - 1)
def plot_length (n : ℕ) : ℕ := 4 + 3 * (n - 1)
def plot_area (n : ℕ) : ℕ := plot_width n * plot_length n

theorem fifth_plot_excess_tiles : plot_area 5 - plot_area 4 = 59 := by
  sorry

end NUMINAMATH_CALUDE_fifth_plot_excess_tiles_l2375_237516


namespace NUMINAMATH_CALUDE_factor_expression_l2375_237520

theorem factor_expression (x : ℝ) : 3*x*(x-4) + 5*(x-4) - 2*(x-4) = (3*x + 3)*(x-4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2375_237520
