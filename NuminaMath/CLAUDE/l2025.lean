import Mathlib

namespace NUMINAMATH_CALUDE_expand_binomial_product_l2025_202537

theorem expand_binomial_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomial_product_l2025_202537


namespace NUMINAMATH_CALUDE_brochure_calculation_l2025_202510

/-- Calculates the number of brochures created by a printing press given specific conditions -/
theorem brochure_calculation (single_page_spreads : ℕ) 
  (h1 : single_page_spreads = 20)
  (h2 : ∀ n : ℕ, n = single_page_spreads → 2 * n = number_of_double_page_spreads)
  (h3 : ∀ n : ℕ, n = total_spread_pages → n / 4 = number_of_ad_blocks)
  (h4 : ∀ n : ℕ, n = number_of_ad_blocks → 4 * n = total_ads)
  (h5 : ∀ n : ℕ, n = total_ads → n / 4 = ad_pages)
  (h6 : ∀ n : ℕ, n = total_pages → n / 5 = number_of_brochures)
  : number_of_brochures = 25 := by
  sorry

#check brochure_calculation

end NUMINAMATH_CALUDE_brochure_calculation_l2025_202510


namespace NUMINAMATH_CALUDE_picture_area_l2025_202585

theorem picture_area (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 2 * x * y + 9 * x + 4 * y = 42) : x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l2025_202585


namespace NUMINAMATH_CALUDE_length_ae_is_21_l2025_202534

/-- Given 5 consecutive points on a straight line, prove that under certain conditions, the length of ae is 21 -/
theorem length_ae_is_21
  (a b c d e : ℝ) -- Representing points as real numbers on a line
  (h_consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Consecutive points
  (h_bc_cd : c - b = 3 * (d - c)) -- bc = 3 cd
  (h_de : e - d = 8) -- de = 8
  (h_ab : b - a = 5) -- ab = 5
  (h_ac : c - a = 11) -- ac = 11
  : e - a = 21 := by
  sorry

end NUMINAMATH_CALUDE_length_ae_is_21_l2025_202534


namespace NUMINAMATH_CALUDE_total_flight_distance_l2025_202517

/-- Given the distances between Spain, Russia, and a stopover country, 
    calculate the total distance to fly from the stopover to Russia and back to Spain. -/
theorem total_flight_distance 
  (spain_russia : ℕ) 
  (spain_stopover : ℕ) 
  (h1 : spain_russia = 7019)
  (h2 : spain_stopover = 1615) : 
  spain_stopover + (spain_russia - spain_stopover) + spain_russia = 12423 :=
by sorry

#check total_flight_distance

end NUMINAMATH_CALUDE_total_flight_distance_l2025_202517


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l2025_202551

theorem difference_of_squares_special_case : (723 : ℤ) * 723 - 722 * 724 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l2025_202551


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l2025_202563

-- Define the sets A and B
def A : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def B : Set ℝ := {x | (x + 2) / (x - 14) < 0}

-- Define the set E
def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

-- Statement for the first part of the problem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 14} := by sorry

-- Statement for the second part of the problem
theorem E_subset_B_implies_a_geq_neg_one (a : ℝ) :
  E a ⊆ B → a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l2025_202563


namespace NUMINAMATH_CALUDE_pecan_amount_correct_l2025_202519

/-- Represents the composition of a nut mixture -/
structure NutMixture where
  pecan_pounds : ℝ
  cashew_pounds : ℝ
  pecan_price : ℝ
  mixture_price : ℝ

/-- Verifies if a given nut mixture satisfies the problem conditions -/
def is_valid_mixture (m : NutMixture) : Prop :=
  m.cashew_pounds = 2 ∧
  m.pecan_price = 5.60 ∧
  m.mixture_price = 4.34

/-- Calculates the total value of the mixture -/
def mixture_value (m : NutMixture) : ℝ :=
  (m.pecan_pounds + m.cashew_pounds) * m.mixture_price

/-- Calculates the value of pecans in the mixture -/
def pecan_value (m : NutMixture) : ℝ :=
  m.pecan_pounds * m.pecan_price

/-- The main theorem stating that the mixture with 1.33333333333 pounds of pecans
    satisfies the problem conditions -/
theorem pecan_amount_correct (m : NutMixture) 
  (h_valid : is_valid_mixture m)
  (h_pecan : m.pecan_pounds = 1.33333333333) :
  mixture_value m = pecan_value m + m.cashew_pounds * (mixture_value m / (m.pecan_pounds + m.cashew_pounds)) :=
by
  sorry


end NUMINAMATH_CALUDE_pecan_amount_correct_l2025_202519


namespace NUMINAMATH_CALUDE_system_solution_l2025_202598

theorem system_solution (k : ℚ) : 
  (∃ x y : ℚ, x + y = 5 * k ∧ x - y = 9 * k ∧ 2 * x + 3 * y = 6) → k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2025_202598


namespace NUMINAMATH_CALUDE_fourth_power_difference_l2025_202525

theorem fourth_power_difference (a b : ℝ) : 
  (a - b)^4 = a^4 - 4*a^3*b + 6*a^2*b^2 - 4*a*b^3 + b^4 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_power_difference_l2025_202525


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2025_202567

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {x : ℕ | ∃ n ∈ A, x = 2 * n}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2025_202567


namespace NUMINAMATH_CALUDE_rotation_90_clockwise_l2025_202503

-- Define the possible positions in the circle
inductive Position
  | Top
  | Left
  | Right

-- Define the shapes
inductive Shape
  | Pentagon
  | SmallerCircle
  | Rectangle

-- Define a function to represent the initial configuration
def initial_config : Position → Shape
  | Position.Top => Shape.Pentagon
  | Position.Left => Shape.SmallerCircle
  | Position.Right => Shape.Rectangle

-- Define a function to represent the configuration after 90° clockwise rotation
def rotated_config : Position → Shape
  | Position.Top => Shape.SmallerCircle
  | Position.Right => Shape.Pentagon
  | Position.Left => Shape.Rectangle

-- Theorem stating that the rotated configuration is correct
theorem rotation_90_clockwise :
  ∀ p : Position, rotated_config p = initial_config (match p with
    | Position.Top => Position.Right
    | Position.Right => Position.Left
    | Position.Left => Position.Top
  ) :=
by sorry

end NUMINAMATH_CALUDE_rotation_90_clockwise_l2025_202503


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l2025_202521

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 2 = 0 → x - y - 2 = 0 → 
   ((-a/2) * 1 = -1)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l2025_202521


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l2025_202583

/-- The quadratic equation ax^2 + 2x + 1 = 0 has at least one negative root
    if and only if a < 0 or 0 < a ≤ 1 -/
theorem quadratic_negative_root (a : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (a < 0 ∨ (0 < a ∧ a ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l2025_202583


namespace NUMINAMATH_CALUDE_mask_production_optimization_l2025_202502

/-- Represents the production plan for masks -/
structure MaskProduction where
  typeA : ℕ  -- Number of type A masks produced
  typeB : ℕ  -- Number of type B masks produced
  days : ℕ   -- Number of days used for production

/-- Checks if a production plan is valid according to the given conditions -/
def isValidProduction (p : MaskProduction) : Prop :=
  p.typeA + p.typeB = 50000 ∧
  p.typeA ≥ 18000 ∧
  p.days ≤ 8 ∧
  p.typeA ≤ 6000 * p.days ∧
  p.typeB ≤ 8000 * (p.days - (p.typeA / 6000))

/-- Calculates the profit for a given production plan -/
def profit (p : MaskProduction) : ℕ :=
  (p.typeA * 5 + p.typeB * 3) / 10

/-- Theorem stating the maximum profit and minimum production time -/
theorem mask_production_optimization :
  (∃ p : MaskProduction, isValidProduction p ∧
    (∀ q : MaskProduction, isValidProduction q → profit q ≤ profit p) ∧
    profit p = 23400) ∧
  (∃ p : MaskProduction, isValidProduction p ∧
    (∀ q : MaskProduction, isValidProduction q → p.days ≤ q.days) ∧
    p.days = 7) := by
  sorry


end NUMINAMATH_CALUDE_mask_production_optimization_l2025_202502


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2025_202530

open Set

theorem fixed_point_theorem (f g : (Set.Icc 0 1) → (Set.Icc 0 1))
  (hf_cont : Continuous f)
  (hg_cont : Continuous g)
  (h_comm : ∀ x ∈ Set.Icc 0 1, f (g x) = g (f x))
  (hf_incr : StrictMono f) :
  ∃ a ∈ Set.Icc 0 1, f a = a ∧ g a = a := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2025_202530


namespace NUMINAMATH_CALUDE_range_of_a_l2025_202552

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 4) → -3 ≤ a ∧ a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2025_202552


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2025_202504

def P : Set ℤ := {x | (x - 3) * (x - 6) ≤ 0}
def Q : Set ℤ := {5, 7}

theorem intersection_P_Q : P ∩ Q = {5} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2025_202504


namespace NUMINAMATH_CALUDE_ebook_count_l2025_202555

def total_ebooks (anna_bought : ℕ) (john_diff : ℕ) (john_lost : ℕ) (mary_factor : ℕ) (mary_gave : ℕ) : ℕ :=
  let john_bought := anna_bought - john_diff
  let john_has := john_bought - john_lost
  let mary_bought := mary_factor * john_bought
  let mary_has := mary_bought - mary_gave
  anna_bought + john_has + mary_has

theorem ebook_count :
  total_ebooks 50 15 3 2 7 = 145 := by
  sorry

end NUMINAMATH_CALUDE_ebook_count_l2025_202555


namespace NUMINAMATH_CALUDE_randy_initial_money_l2025_202595

/-- Randy's initial amount of money -/
def initial_money : ℕ := sorry

/-- Amount Smith gave to Randy -/
def smith_gave : ℕ := 200

/-- Amount Randy gave to Sally -/
def sally_received : ℕ := 1200

/-- Amount Randy kept after giving money to Sally -/
def randy_kept : ℕ := 2000

theorem randy_initial_money :
  initial_money + smith_gave - sally_received = randy_kept ∧
  initial_money = 3000 := by sorry

end NUMINAMATH_CALUDE_randy_initial_money_l2025_202595


namespace NUMINAMATH_CALUDE_four_squared_sum_equals_four_cubed_l2025_202553

theorem four_squared_sum_equals_four_cubed : 4^2 + 4^2 + 4^2 + 4^2 = 4^3 := by
  sorry

end NUMINAMATH_CALUDE_four_squared_sum_equals_four_cubed_l2025_202553


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2025_202507

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 2} = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2025_202507


namespace NUMINAMATH_CALUDE_xy_value_l2025_202566

theorem xy_value (x y : ℝ) 
  (h1 : (16 : ℝ)^x / (4 : ℝ)^(x + y) = 16)
  (h2 : (25 : ℝ)^(x + y) / (5 : ℝ)^(6 * y) = 625) :
  x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2025_202566


namespace NUMINAMATH_CALUDE_chess_tournament_max_matches_l2025_202564

/-- Represents a chess tournament. -/
structure ChessTournament where
  participants : Nat
  max_matches_per_pair : Nat
  no_triples : Bool

/-- The maximum number of matches any participant can play. -/
def max_matches_per_participant (t : ChessTournament) : Nat :=
  sorry

/-- The theorem stating the maximum number of matches per participant
    in a chess tournament with the given conditions. -/
theorem chess_tournament_max_matches
  (t : ChessTournament)
  (h1 : t.participants = 300)
  (h2 : t.max_matches_per_pair = 1)
  (h3 : t.no_triples = true) :
  max_matches_per_participant t = 200 :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_max_matches_l2025_202564


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2025_202527

theorem min_value_quadratic (x : ℝ) (y : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) →
  y = 2 * x^2 - 6 * x + 3 →
  ∃ (m : ℝ), m = -1 ∧ ∀ z ∈ Set.Icc (-1 : ℝ) (1 : ℝ), 2 * z^2 - 6 * z + 3 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2025_202527


namespace NUMINAMATH_CALUDE_g_range_values_l2025_202541

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - 3*x) / (2 + 3*x))

theorem g_range_values :
  {y | ∃ x, g x = y} = {-π/2, π/4} := by sorry

end NUMINAMATH_CALUDE_g_range_values_l2025_202541


namespace NUMINAMATH_CALUDE_sqrt3_cos_minus_sin_eq_sqrt2_l2025_202515

theorem sqrt3_cos_minus_sin_eq_sqrt2 :
  Real.sqrt 3 * Real.cos (π / 12) - Real.sin (π / 12) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_cos_minus_sin_eq_sqrt2_l2025_202515


namespace NUMINAMATH_CALUDE_rosencrans_wins_iff_odd_l2025_202593

/-- Represents a chord-drawing game on a circle with n points. -/
structure ChordGame where
  n : ℕ
  h : n ≥ 5

/-- Represents the outcome of the game. -/
inductive Outcome
  | RosencransWins
  | GildensternWins

/-- Determines the winner of the chord game based on the number of points. -/
def ChordGame.winner (game : ChordGame) : Outcome :=
  if game.n % 2 = 1 then Outcome.RosencransWins else Outcome.GildensternWins

/-- Theorem stating that Rosencrans wins if and only if n is odd. -/
theorem rosencrans_wins_iff_odd (game : ChordGame) :
  game.winner = Outcome.RosencransWins ↔ Odd game.n :=
sorry

end NUMINAMATH_CALUDE_rosencrans_wins_iff_odd_l2025_202593


namespace NUMINAMATH_CALUDE_smallest_positive_solution_of_equation_l2025_202557

theorem smallest_positive_solution_of_equation :
  ∃ (x : ℝ), x > 0 ∧ x^4 - 58*x^2 + 841 = 0 ∧ ∀ (y : ℝ), y > 0 ∧ y^4 - 58*y^2 + 841 = 0 → x ≤ y ∧ x = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_of_equation_l2025_202557


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2025_202558

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 9| = |x + 3| + 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2025_202558


namespace NUMINAMATH_CALUDE_y_divisibility_l2025_202560

def y : ℕ := 58 + 104 + 142 + 184 + 304 + 368 + 3304

theorem y_divisibility :
  (∃ k : ℕ, y = 2 * k) ∧
  (∃ k : ℕ, y = 4 * k) ∧
  ¬(∀ k : ℕ, y = 8 * k) ∧
  ¬(∀ k : ℕ, y = 16 * k) := by
  sorry

end NUMINAMATH_CALUDE_y_divisibility_l2025_202560


namespace NUMINAMATH_CALUDE_max_polygons_bound_l2025_202596

/-- The number of points marked on the circle. -/
def num_points : ℕ := 12

/-- The minimum allowed internal angle at the circle's center (in degrees). -/
def min_angle : ℝ := 30

/-- A function that calculates the maximum number of distinct convex polygons
    that can be formed under the given conditions. -/
def max_polygons (n : ℕ) (θ : ℝ) : ℕ :=
  2^n - (n.choose 0 + n.choose 1 + n.choose 2)

/-- Theorem stating that the maximum number of distinct convex polygons
    satisfying the conditions is less than or equal to 4017. -/
theorem max_polygons_bound :
  max_polygons num_points min_angle ≤ 4017 :=
sorry

end NUMINAMATH_CALUDE_max_polygons_bound_l2025_202596


namespace NUMINAMATH_CALUDE_compound_interest_rate_is_ten_percent_l2025_202599

/-- Given the conditions of the problem, prove that the compound interest rate is 10% --/
theorem compound_interest_rate_is_ten_percent
  (simple_principal : ℝ)
  (simple_rate : ℝ)
  (simple_time : ℝ)
  (compound_principal : ℝ)
  (compound_time : ℝ)
  (h1 : simple_principal = 1750.0000000000018)
  (h2 : simple_rate = 8)
  (h3 : simple_time = 3)
  (h4 : compound_principal = 4000)
  (h5 : compound_time = 2)
  (h6 : simple_principal * simple_rate * simple_time / 100 = 
        compound_principal * ((1 + compound_rate / 100) ^ compound_time - 1) / 2)
  : compound_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_is_ten_percent_l2025_202599


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2025_202579

/-- Given a quadratic equation (m-2)x^2 + 3x - m^2 - m + 6 = 0 where one root is 0,
    prove that m = -3 is the only valid solution. -/
theorem quadratic_root_zero (m : ℝ) : 
  (∀ x, (m - 2) * x^2 + 3 * x - m^2 - m + 6 = 0 ↔ x = 0 ∨ x = (m^2 + m - 6) / (2 * m - 4)) →
  m - 2 ≠ 0 →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2025_202579


namespace NUMINAMATH_CALUDE_steve_final_marbles_l2025_202592

theorem steve_final_marbles (sam_initial steve_initial sally_initial sam_final : ℕ) :
  sam_initial = 2 * steve_initial →
  sally_initial = sam_initial - 5 →
  sam_final = sam_initial - 6 →
  sam_final = 8 →
  steve_initial + 3 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_steve_final_marbles_l2025_202592


namespace NUMINAMATH_CALUDE_cube_construction_condition_l2025_202594

/-- A brick is composed of twelve unit cubes arranged in a three-step staircase of width 2. -/
def Brick : Type := Unit

/-- Predicate indicating whether it's possible to build a cube of side length n using Bricks. -/
def CanBuildCube (n : ℕ) : Prop := sorry

theorem cube_construction_condition (n : ℕ) : 
  CanBuildCube n ↔ 12 ∣ n :=
sorry

end NUMINAMATH_CALUDE_cube_construction_condition_l2025_202594


namespace NUMINAMATH_CALUDE_house_occupancy_l2025_202514

/-- The number of people in the house given specific room occupancies. -/
def people_in_house (bedroom living_room kitchen garage patio : ℕ) : ℕ :=
  bedroom + living_room + kitchen + garage + patio

/-- The problem statement as a theorem. -/
theorem house_occupancy : ∃ (bedroom living_room kitchen garage patio : ℕ),
  bedroom = 7 ∧
  living_room = 8 ∧
  kitchen = living_room + 3 ∧
  garage * 2 = kitchen ∧
  patio = garage * 2 ∧
  people_in_house bedroom living_room kitchen garage patio = 41 := by
  sorry

end NUMINAMATH_CALUDE_house_occupancy_l2025_202514


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l2025_202549

theorem chicken_wings_distribution (friends : ℕ) (pre_cooked : ℕ) (additional : ℕ) :
  friends = 3 →
  pre_cooked = 8 →
  additional = 10 →
  (pre_cooked + additional) / friends = 6 :=
by sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l2025_202549


namespace NUMINAMATH_CALUDE_digit_55_is_2_l2025_202578

/-- The decimal representation of 1/17 as a list of digits -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating sequence in the decimal representation of 1/17 -/
def repeat_length : Nat := 16

/-- The 55th digit after the decimal point in the decimal representation of 1/17 -/
def digit_55 : Nat := decimal_rep_1_17[(55 - 1) % repeat_length]

theorem digit_55_is_2 : digit_55 = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_55_is_2_l2025_202578


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2025_202542

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 30 → 2 * y - 3 * x = 5 → |y - x| = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2025_202542


namespace NUMINAMATH_CALUDE_min_value_expression_l2025_202501

theorem min_value_expression (x y : ℝ) : x^2 + x*y + y^2 - 3*y ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2025_202501


namespace NUMINAMATH_CALUDE_max_value_of_vector_difference_l2025_202548

/-- Given plane vectors a and b satisfying |b| = 2|a| = 2, 
    the maximum value of |a - 2b| is 5. -/
theorem max_value_of_vector_difference (a b : ℝ × ℝ) 
  (h1 : ‖b‖ = 2 * ‖a‖) (h2 : ‖b‖ = 2) : 
  ∃ (max : ℝ), max = 5 ∧ ∀ (x : ℝ × ℝ), x = a - 2 • b → ‖x‖ ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_vector_difference_l2025_202548


namespace NUMINAMATH_CALUDE_box_probability_l2025_202561

theorem box_probability (a : ℕ) (h1 : a > 0) : 
  (4 : ℝ) / a = (1 : ℝ) / 5 → a = 20 := by
sorry

end NUMINAMATH_CALUDE_box_probability_l2025_202561


namespace NUMINAMATH_CALUDE_reading_time_calculation_l2025_202590

theorem reading_time_calculation (total_time math_time spelling_time history_time science_time piano_time break_time : ℕ)
  (h1 : total_time = 180)
  (h2 : math_time = 25)
  (h3 : spelling_time = 30)
  (h4 : history_time = 20)
  (h5 : science_time = 15)
  (h6 : piano_time = 30)
  (h7 : break_time = 20) :
  total_time - (math_time + spelling_time + history_time + science_time + piano_time + break_time) = 40 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l2025_202590


namespace NUMINAMATH_CALUDE_cannot_reach_all_same_l2025_202500

/-- Represents the state of the circle of numbers -/
structure CircleState where
  ones : Nat
  zeros : Nat
  deriving Repr

/-- The operation performed on the circle each second -/
def next_state (s : CircleState) : CircleState :=
  sorry

/-- Predicate to check if all numbers in the circle are the same -/
def all_same (s : CircleState) : Prop :=
  s.ones = 0 ∨ s.zeros = 0

/-- The initial state of the circle -/
def initial_state : CircleState :=
  { ones := 4, zeros := 5 }

/-- Theorem stating that it's impossible to reach a state where all numbers are the same -/
theorem cannot_reach_all_same :
  ¬ ∃ (n : Nat), all_same (n.iterate next_state initial_state) :=
sorry

end NUMINAMATH_CALUDE_cannot_reach_all_same_l2025_202500


namespace NUMINAMATH_CALUDE_unique_years_arithmetic_sequence_l2025_202544

/-- A year in the 19th century -/
structure Year19thCentury where
  x : Nat
  y : Nat
  x_range : x ≤ 9
  y_range : y ≤ 9

/-- Check if the differences between adjacent digits form an arithmetic sequence -/
def isArithmeticSequence (year : Year19thCentury) : Prop :=
  ∃ d : Int, (year.x - 8 : Int) - 7 = d ∧ (year.y - year.x : Int) - (year.x - 8) = d

/-- The theorem stating that 1881 and 1894 are the only years satisfying the condition -/
theorem unique_years_arithmetic_sequence :
  ∀ year : Year19thCentury, isArithmeticSequence year ↔ (year.x = 8 ∧ year.y = 1) ∨ (year.x = 9 ∧ year.y = 4) :=
by sorry

end NUMINAMATH_CALUDE_unique_years_arithmetic_sequence_l2025_202544


namespace NUMINAMATH_CALUDE_sqrt_sum_fraction_simplification_l2025_202523

theorem sqrt_sum_fraction_simplification :
  Real.sqrt ((36 : ℝ) / 49 + 16 / 9 + 1 / 16) = 45 / 28 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fraction_simplification_l2025_202523


namespace NUMINAMATH_CALUDE_division_problem_l2025_202547

theorem division_problem (d q : ℚ) : 
  (100 / d = q) → 
  (d * ⌊q⌋ ≤ 100) → 
  (100 - d * ⌊q⌋ = 4) → 
  (d = 16 ∧ q = 6.65) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2025_202547


namespace NUMINAMATH_CALUDE_star_three_four_l2025_202529

def star (x y : ℝ) : ℝ := 4 * x + 6 * y

theorem star_three_four : star 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_star_three_four_l2025_202529


namespace NUMINAMATH_CALUDE_chairs_bought_l2025_202588

theorem chairs_bought (chair_cost : ℕ) (total_spent : ℕ) (num_chairs : ℕ) : 
  chair_cost = 15 → total_spent = 180 → num_chairs * chair_cost = total_spent → num_chairs = 12 := by
  sorry

end NUMINAMATH_CALUDE_chairs_bought_l2025_202588


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l2025_202513

theorem ceiling_floor_difference : 
  ⌈(18 : ℚ) / 11 * (-33 : ℚ) / 4⌉ - ⌊(18 : ℚ) / 11 * ⌊(-33 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l2025_202513


namespace NUMINAMATH_CALUDE_height_calculations_l2025_202572

-- Define the conversion rate
def inch_to_cm : ℝ := 2.54

-- Define heights in inches
def maria_height_inches : ℝ := 54
def samuel_height_inches : ℝ := 72

-- Define function to convert inches to centimeters
def inches_to_cm (inches : ℝ) : ℝ := inches * inch_to_cm

-- Theorem statement
theorem height_calculations :
  let maria_height_cm := inches_to_cm maria_height_inches
  let samuel_height_cm := inches_to_cm samuel_height_inches
  let height_difference := samuel_height_cm - maria_height_cm
  (maria_height_cm = 137.16) ∧
  (samuel_height_cm = 182.88) ∧
  (height_difference = 45.72) := by
  sorry

end NUMINAMATH_CALUDE_height_calculations_l2025_202572


namespace NUMINAMATH_CALUDE_wilted_ratio_after_first_night_l2025_202589

/-- Represents the number of roses at different stages --/
structure RoseCount where
  initial : ℕ
  afterFirstNight : ℕ
  afterSecondNight : ℕ

/-- Calculates the ratio of wilted flowers to total flowers after the first night --/
def wiltedRatio (rc : RoseCount) : Rat :=
  (rc.initial - rc.afterFirstNight) / rc.initial

theorem wilted_ratio_after_first_night
  (rc : RoseCount)
  (h1 : rc.initial = 36)
  (h2 : rc.afterSecondNight = 9)
  (h3 : rc.afterFirstNight = 2 * rc.afterSecondNight) :
  wiltedRatio rc = 1/2 := by
  sorry

#eval wiltedRatio { initial := 36, afterFirstNight := 18, afterSecondNight := 9 }

end NUMINAMATH_CALUDE_wilted_ratio_after_first_night_l2025_202589


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2025_202591

/-- Definition of line l₁ -/
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + (a - 1) * y - 1 = 0

/-- Definition of line l₂ -/
def l₂ (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x + (2 * a + 3) * y - 3 = 0

/-- Definition of perpendicular lines -/
def perpendicular (a : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ, 
  l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ ∧ 
  a * (a - 1) + (a - 1) * (2 * a + 3) = 0

/-- Theorem stating that a = 1 is sufficient but not necessary for perpendicularity -/
theorem a_eq_one_sufficient_not_necessary : 
  (∀ a : ℝ, a = 1 → perpendicular a) ∧ 
  ¬(∀ a : ℝ, perpendicular a → a = 1) := by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2025_202591


namespace NUMINAMATH_CALUDE_prism_to_spheres_waste_l2025_202506

/-- The volume of waste when polishing a regular triangular prism into spheres -/
theorem prism_to_spheres_waste (base_side : ℝ) (height : ℝ) (sphere_radius : ℝ) :
  base_side = 6 →
  height = 8 * Real.sqrt 3 →
  sphere_radius = Real.sqrt 3 →
  ((Real.sqrt 3 / 4) * base_side^2 * height) - (4 * (4 / 3) * Real.pi * sphere_radius^3) =
    216 - 16 * Real.sqrt 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_prism_to_spheres_waste_l2025_202506


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l2025_202535

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a⋅cos B = b⋅cos A, then the triangle is isosceles with A = B -/
theorem isosceles_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  a * Real.cos B = b * Real.cos A →
  A = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l2025_202535


namespace NUMINAMATH_CALUDE_multiply_52_48_l2025_202569

theorem multiply_52_48 : 52 * 48 = 2496 := by
  sorry

end NUMINAMATH_CALUDE_multiply_52_48_l2025_202569


namespace NUMINAMATH_CALUDE_farm_feet_count_l2025_202540

/-- A farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of heads in the farm -/
def total_heads (f : Farm) : ℕ := f.hens + f.cows

/-- The total number of feet in the farm -/
def total_feet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- Theorem: Given a farm with 48 heads and 28 hens, the total number of feet is 136 -/
theorem farm_feet_count :
  ∀ f : Farm, total_heads f = 48 → f.hens = 28 → total_feet f = 136 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_feet_count_l2025_202540


namespace NUMINAMATH_CALUDE_largest_multiple_of_18_with_8_and_0_l2025_202511

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 8 ∨ d = 0

theorem largest_multiple_of_18_with_8_and_0 :
  ∃ m : ℕ,
    m > 0 ∧
    m % 18 = 0 ∧
    is_valid_number m ∧
    (∀ k : ℕ, k > m → k % 18 = 0 → ¬is_valid_number k) ∧
    m / 18 = 493826048 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_18_with_8_and_0_l2025_202511


namespace NUMINAMATH_CALUDE_parallelogram_area_l2025_202570

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 9 inches and 12 inches is 54√3 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 54 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2025_202570


namespace NUMINAMATH_CALUDE_specific_ellipse_major_axis_l2025_202522

/-- An ellipse with specific properties -/
structure Ellipse where
  -- The ellipse is tangent to both x-axis and y-axis
  tangent_to_axes : Bool
  -- The x-coordinate of both foci
  focus_x : ℝ
  -- The y-coordinates of the foci
  focus_y1 : ℝ
  focus_y2 : ℝ

/-- The length of the major axis of the ellipse -/
def major_axis_length (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating the length of the major axis for a specific ellipse -/
theorem specific_ellipse_major_axis :
  ∃ (e : Ellipse), 
    e.tangent_to_axes = true ∧
    e.focus_x = 3 ∧
    e.focus_y1 = -4 + 2 * Real.sqrt 2 ∧
    e.focus_y2 = -4 - 2 * Real.sqrt 2 ∧
    major_axis_length e = 8 :=
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_major_axis_l2025_202522


namespace NUMINAMATH_CALUDE_multiplication_proof_l2025_202550

theorem multiplication_proof (m : ℕ) : m = 32505 → m * 9999 = 325027405 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_proof_l2025_202550


namespace NUMINAMATH_CALUDE_g_of_three_l2025_202576

/-- Given a function g : ℝ → ℝ satisfying g(x) - 3 * g(1/x) = 3^x + 1 for all x ≠ 0,
    prove that g(3) = -17/4 -/
theorem g_of_three (g : ℝ → ℝ) 
    (h : ∀ x ≠ 0, g x - 3 * g (1/x) = 3^x + 1) : 
    g 3 = -17/4 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_l2025_202576


namespace NUMINAMATH_CALUDE_die_throws_for_most_likely_two_l2025_202533

theorem die_throws_for_most_likely_two (n : ℕ) : 
  let p : ℚ := 1/6  -- probability of rolling a two
  let q : ℚ := 5/6  -- probability of not rolling a two
  let k₀ : ℕ := 32  -- most likely number of times a two is rolled
  (n * p - q ≤ k₀ ∧ k₀ ≤ n * p + p) → (191 ≤ n ∧ n ≤ 197) :=
by sorry

end NUMINAMATH_CALUDE_die_throws_for_most_likely_two_l2025_202533


namespace NUMINAMATH_CALUDE_num_aplus_needed_is_two_l2025_202587

/-- Represents the grading system and reward calculation for Paul's courses. -/
structure GradingSystem where
  numCourses : Nat
  bPlusReward : Nat
  aReward : Nat
  aPlusReward : Nat
  maxReward : Nat

/-- Calculates the number of A+ grades needed to double the previous rewards. -/
def numAPlusNeeded (gs : GradingSystem) : Nat :=
  sorry

/-- Theorem stating that the number of A+ grades needed is 2. -/
theorem num_aplus_needed_is_two (gs : GradingSystem) 
  (h1 : gs.numCourses = 10)
  (h2 : gs.bPlusReward = 5)
  (h3 : gs.aReward = 10)
  (h4 : gs.aPlusReward = 15)
  (h5 : gs.maxReward = 190) :
  numAPlusNeeded gs = 2 := by
  sorry

end NUMINAMATH_CALUDE_num_aplus_needed_is_two_l2025_202587


namespace NUMINAMATH_CALUDE_derek_initial_money_l2025_202505

theorem derek_initial_money (initial_money : ℚ) : 
  (initial_money / 2 - (initial_money / 2) / 4 = 360) → initial_money = 960 := by
  sorry

end NUMINAMATH_CALUDE_derek_initial_money_l2025_202505


namespace NUMINAMATH_CALUDE_reservoir_water_amount_l2025_202577

theorem reservoir_water_amount 
  (total_capacity : ℝ) 
  (end_amount : ℝ) 
  (normal_level : ℝ) 
  (h1 : end_amount = 2 * normal_level)
  (h2 : end_amount = 0.75 * total_capacity)
  (h3 : normal_level = total_capacity - 20) :
  end_amount = 24 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_water_amount_l2025_202577


namespace NUMINAMATH_CALUDE_pirate_coin_sharing_l2025_202536

/-- The number of coins Pete gives himself in the final round -/
def x : ℕ := 9

/-- The total number of coins Pete has at the end -/
def petes_coins (x : ℕ) : ℕ := x * (x + 1) / 2

/-- The total number of coins Paul has at the end -/
def pauls_coins (x : ℕ) : ℕ := x

/-- The condition that Pete has 5 times as many coins as Paul -/
def pete_five_times_paul (x : ℕ) : Prop :=
  petes_coins x = 5 * pauls_coins x

/-- The total number of coins shared -/
def total_coins (x : ℕ) : ℕ := petes_coins x + pauls_coins x

theorem pirate_coin_sharing :
  pete_five_times_paul x ∧ total_coins x = 54 := by
  sorry

end NUMINAMATH_CALUDE_pirate_coin_sharing_l2025_202536


namespace NUMINAMATH_CALUDE_negative_cube_squared_l2025_202528

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l2025_202528


namespace NUMINAMATH_CALUDE_not_true_from_false_premises_l2025_202516

theorem not_true_from_false_premises (p q : Prop) : 
  ¬ (∀ (p q : Prop), (p → q) → (¬p → q)) :=
sorry

end NUMINAMATH_CALUDE_not_true_from_false_premises_l2025_202516


namespace NUMINAMATH_CALUDE_mini_quiz_true_false_count_l2025_202580

/-- The number of true-false questions in the mini-quiz. -/
def n : ℕ := 3

/-- The number of multiple-choice questions. -/
def m : ℕ := 2

/-- The number of answer choices for each multiple-choice question. -/
def k : ℕ := 4

/-- The total number of ways to write the answer key. -/
def total_ways : ℕ := 96

theorem mini_quiz_true_false_count :
  (2^n - 2) * k^m = total_ways ∧ n > 0 := by sorry

end NUMINAMATH_CALUDE_mini_quiz_true_false_count_l2025_202580


namespace NUMINAMATH_CALUDE_parabola_translation_existence_l2025_202559

theorem parabola_translation_existence : ∃ (h k : ℝ),
  (0 = -(0 - h)^2 + k) ∧  -- passes through origin
  ((1/2) * (2*h) * k = 1) ∧  -- triangle area is 1
  (h^2 = k) ∧  -- vertex is (h, k)
  (h = 1 ∨ h = -1) ∧
  (k = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_existence_l2025_202559


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2025_202573

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Icc (-1/2 : ℝ) (-1/3) = {x | a * x^2 - b * x - 1 ≥ 0}) :
  {x : ℝ | x^2 - b*x - a < 0} = Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2025_202573


namespace NUMINAMATH_CALUDE_tree_planting_equation_system_l2025_202524

theorem tree_planting_equation_system :
  ∀ (x y : ℕ),
  (x + y = 20) →
  (3 * x + 2 * y = 52) →
  (∀ (total_pioneers total_trees boys_trees girls_trees : ℕ),
    total_pioneers = 20 →
    total_trees = 52 →
    boys_trees = 3 →
    girls_trees = 2 →
    x + y = total_pioneers ∧
    3 * x + 2 * y = total_trees) :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_equation_system_l2025_202524


namespace NUMINAMATH_CALUDE_x_equals_six_l2025_202597

theorem x_equals_six (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x) (h3 : 1/a + 1/b = 1) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_six_l2025_202597


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l2025_202539

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 435 → books_sold = 218 → books_per_shelf = 17 →
  (initial_stock - books_sold + books_per_shelf - 1) / books_per_shelf = 13 := by
sorry


end NUMINAMATH_CALUDE_coloring_book_shelves_l2025_202539


namespace NUMINAMATH_CALUDE_arrangement_count_four_objects_five_positions_l2025_202562

theorem arrangement_count : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * arrangement_count n

theorem four_objects_five_positions :
  arrangement_count 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_four_objects_five_positions_l2025_202562


namespace NUMINAMATH_CALUDE_john_investment_proof_l2025_202518

/-- The amount John invested in total -/
def total_investment : ℝ := 1200

/-- The annual interest rate for Bank A -/
def rate_A : ℝ := 0.04

/-- The annual interest rate for Bank B -/
def rate_B : ℝ := 0.06

/-- The number of years the money is invested -/
def years : ℕ := 2

/-- The total amount after two years -/
def final_amount : ℝ := 1300.50

/-- The amount John invested in Bank A -/
def investment_A : ℝ := 1138.57

theorem john_investment_proof :
  ∃ (x : ℝ), 
    x = investment_A ∧ 
    x ≥ 0 ∧ 
    x ≤ total_investment ∧
    x * (1 + rate_A) ^ years + (total_investment - x) * (1 + rate_B) ^ years = final_amount :=
by sorry

end NUMINAMATH_CALUDE_john_investment_proof_l2025_202518


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l2025_202546

theorem sum_of_fractions_geq_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (a + 2*b) + b / (b + 2*c) + c / (c + 2*a) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l2025_202546


namespace NUMINAMATH_CALUDE_shirt_ratio_l2025_202545

/-- Given that Hazel received 6 shirts and the total number of shirts is 18,
    prove that the ratio of Razel's shirts to Hazel's shirts is 2:1. -/
theorem shirt_ratio (hazel_shirts : ℕ) (total_shirts : ℕ) (razel_shirts : ℕ) : 
  hazel_shirts = 6 → total_shirts = 18 → razel_shirts = total_shirts - hazel_shirts →
  (razel_shirts : ℚ) / hazel_shirts = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_shirt_ratio_l2025_202545


namespace NUMINAMATH_CALUDE_triangle_perimeter_for_radius_3_l2025_202554

/-- A configuration of three circles and a triangle -/
structure CircleTriangleConfig where
  /-- The radius of each circle -/
  radius : ℝ
  /-- The circles are externally tangent to each other -/
  circles_tangent : Prop
  /-- Each side of the triangle is tangent to two of the circles -/
  triangle_tangent : Prop

/-- The perimeter of the triangle in the given configuration -/
def triangle_perimeter (config : CircleTriangleConfig) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the triangle for the given configuration -/
theorem triangle_perimeter_for_radius_3 :
  ∀ (config : CircleTriangleConfig),
    config.radius = 3 →
    config.circles_tangent →
    config.triangle_tangent →
    triangle_perimeter config = 18 + 18 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_for_radius_3_l2025_202554


namespace NUMINAMATH_CALUDE_leading_zeros_count_l2025_202532

theorem leading_zeros_count (n : ℕ) (h : n = 20^22) :
  (∃ k : ℕ, (1 : ℚ) / n = k / 10^28 ∧ k ≥ 10^27 ∧ k < 10^28) :=
sorry

end NUMINAMATH_CALUDE_leading_zeros_count_l2025_202532


namespace NUMINAMATH_CALUDE_prob_male_monday_female_tuesday_is_one_third_l2025_202568

/-- Represents the number of male volunteers -/
def num_men : ℕ := 2

/-- Represents the number of female volunteers -/
def num_women : ℕ := 2

/-- Represents the total number of volunteers -/
def total_volunteers : ℕ := num_men + num_women

/-- Represents the number of days for which volunteers are selected -/
def num_days : ℕ := 2

/-- Calculates the probability of selecting a male volunteer for Monday
    and a female volunteer for Tuesday -/
def prob_male_monday_female_tuesday : ℚ :=
  (num_men * num_women) / (total_volunteers * (total_volunteers - 1))

/-- Proves that the probability of selecting a male volunteer for Monday
    and a female volunteer for Tuesday is 1/3 -/
theorem prob_male_monday_female_tuesday_is_one_third :
  prob_male_monday_female_tuesday = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_male_monday_female_tuesday_is_one_third_l2025_202568


namespace NUMINAMATH_CALUDE_inscribed_circle_triangle_l2025_202581

theorem inscribed_circle_triangle (r : ℝ) (a b c : ℝ) :
  r = 3 →
  a + b = 7 →
  a = 3 →
  b = 4 →
  c^2 = a^2 + b^2 →
  (a + r)^2 + (b + r)^2 = c^2 →
  (a, b, c) = (3, 4, 5) ∨ (a, b, c) = (4, 3, 5) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_triangle_l2025_202581


namespace NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l2025_202582

theorem largest_angle_convex_pentagon (x : ℝ) : 
  (x + 2) + (2*x + 3) + (3*x - 5) + (4*x + 1) + (5*x - 1) = 540 →
  max (x + 2) (max (2*x + 3) (max (3*x - 5) (max (4*x + 1) (5*x - 1)))) = 179 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l2025_202582


namespace NUMINAMATH_CALUDE_min_value_of_u_l2025_202543

theorem min_value_of_u (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : 2 * x + y = 6) :
  ∃ (min_u : ℝ), min_u = 27 / 2 ∧ ∀ (u : ℝ), u = 4 * x^2 + 3 * x * y + y^2 - 6 * x - 3 * y → u ≥ min_u :=
sorry

end NUMINAMATH_CALUDE_min_value_of_u_l2025_202543


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2025_202508

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2025_202508


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l2025_202512

theorem polygon_interior_angles (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 720 → (180 * (n - 2) : ℝ) = sum_angles → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l2025_202512


namespace NUMINAMATH_CALUDE_install_time_proof_l2025_202575

/-- Calculates the time needed to install remaining windows -/
def time_to_install_remaining (total : ℕ) (installed : ℕ) (time_per_window : ℕ) : ℕ :=
  (total - installed) * time_per_window

/-- Proves that the time to install remaining windows is 48 hours -/
theorem install_time_proof (total : ℕ) (installed : ℕ) (time_per_window : ℕ) 
  (h1 : total = 14)
  (h2 : installed = 8)
  (h3 : time_per_window = 8) :
  time_to_install_remaining total installed time_per_window = 48 := by
  sorry

#eval time_to_install_remaining 14 8 8

end NUMINAMATH_CALUDE_install_time_proof_l2025_202575


namespace NUMINAMATH_CALUDE_contrapositive_sine_not_piecewise_l2025_202584

-- Define the universe of functions
variable (F : Type) [Nonempty F]

-- Define predicates for sine function and piecewise function
variable (is_sine : F → Prop)
variable (is_piecewise : F → Prop)

-- State the theorem
theorem contrapositive_sine_not_piecewise :
  (∀ f : F, is_sine f → ¬ is_piecewise f) ↔
  (∀ f : F, is_piecewise f → ¬ is_sine f) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_sine_not_piecewise_l2025_202584


namespace NUMINAMATH_CALUDE_quadratic_factoring_l2025_202526

theorem quadratic_factoring (x : ℝ) : x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factoring_l2025_202526


namespace NUMINAMATH_CALUDE_power_sum_equality_l2025_202538

theorem power_sum_equality : (-2)^2005 + (-2)^2006 = 2^2005 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2025_202538


namespace NUMINAMATH_CALUDE_intersection_contains_two_elements_l2025_202556

-- Define the sets P and Q
def P (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = k * (x - 1) + 1}
def Q : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 - 2*y = 0}

-- Theorem statement
theorem intersection_contains_two_elements :
  ∃ (k : ℝ), ∃ (a b : ℝ × ℝ), a ≠ b ∧ a ∈ P k ∩ Q ∧ b ∈ P k ∩ Q ∧
  ∀ (c : ℝ × ℝ), c ∈ P k ∩ Q → c = a ∨ c = b :=
sorry

end NUMINAMATH_CALUDE_intersection_contains_two_elements_l2025_202556


namespace NUMINAMATH_CALUDE_count_even_multiples_of_three_squares_l2025_202520

theorem count_even_multiples_of_three_squares (n : Nat) : 
  (∃ k, k ∈ Finset.range n ∧ 36 * k * k < 3000) ↔ n = 10 :=
sorry

end NUMINAMATH_CALUDE_count_even_multiples_of_three_squares_l2025_202520


namespace NUMINAMATH_CALUDE_dan_remaining_marbles_l2025_202531

def initial_marbles : ℕ := 64
def marbles_given_away : ℕ := 14

theorem dan_remaining_marbles :
  initial_marbles - marbles_given_away = 50 := by
  sorry

end NUMINAMATH_CALUDE_dan_remaining_marbles_l2025_202531


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l2025_202509

theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  let f : Polynomial ℤ := X^n + 5 * X^(n-1) + 3
  Irreducible f := by sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l2025_202509


namespace NUMINAMATH_CALUDE_smallest_angle_satisfies_equation_l2025_202565

/-- The smallest positive angle x in degrees that satisfies the given equation -/
def smallest_angle : ℝ := 11.25

theorem smallest_angle_satisfies_equation :
  let x := smallest_angle * Real.pi / 180
  Real.tan (6 * x) = (Real.cos (2 * x) - Real.sin (2 * x)) / (Real.cos (2 * x) + Real.sin (2 * x)) ∧
  ∀ y : ℝ, 0 < y ∧ y < smallest_angle →
    Real.tan (6 * y * Real.pi / 180) ≠ (Real.cos (2 * y * Real.pi / 180) - Real.sin (2 * y * Real.pi / 180)) /
                                       (Real.cos (2 * y * Real.pi / 180) + Real.sin (2 * y * Real.pi / 180)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfies_equation_l2025_202565


namespace NUMINAMATH_CALUDE_steve_earnings_l2025_202574

/-- Calculates the amount of money an author keeps after selling books and paying an agent. -/
def authorEarnings (totalCopies : ℕ) (advanceCopies : ℕ) (earningsPerCopy : ℚ) (agentPercentage : ℚ) : ℚ :=
  let copiesForEarnings := totalCopies - advanceCopies
  let totalEarnings := copiesForEarnings * earningsPerCopy
  let agentCut := totalEarnings * agentPercentage
  totalEarnings - agentCut

/-- Proves that given the conditions of Steve's book sales, he keeps $1,620,000 after paying his agent. -/
theorem steve_earnings :
  authorEarnings 1000000 100000 2 (1/10) = 1620000 := by
  sorry

end NUMINAMATH_CALUDE_steve_earnings_l2025_202574


namespace NUMINAMATH_CALUDE_triangle_angle_and_perimeter_l2025_202571

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_angle_and_perimeter (t : Triangle) (h : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) :
  t.A = π / 3 ∧ 
  (t.a = Real.sqrt 3 → ∃ y : ℝ, y > 2 * Real.sqrt 3 ∧ y ≤ 3 * Real.sqrt 3 ∧ y = t.a + t.b + t.c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_and_perimeter_l2025_202571


namespace NUMINAMATH_CALUDE_product_of_sums_powers_specific_product_evaluation_l2025_202586

theorem product_of_sums_powers (a b : ℕ) : 
  (a + b) * (a^2 + b^2) * (a^4 + b^4) * (a^8 + b^8) = 
  (1/2 : ℚ) * ((a^16 : ℚ) - (b^16 : ℚ)) :=
by sorry

theorem specific_product_evaluation : 
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^8 + 1^8) = 21523360 :=
by sorry

end NUMINAMATH_CALUDE_product_of_sums_powers_specific_product_evaluation_l2025_202586
