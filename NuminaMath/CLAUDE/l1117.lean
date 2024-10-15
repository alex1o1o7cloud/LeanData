import Mathlib

namespace NUMINAMATH_CALUDE_optimal_decomposition_2008_l1117_111746

theorem optimal_decomposition_2008 (decomp : List Nat) :
  (decomp.sum = 2008) →
  (decomp.prod ≤ (List.replicate 668 3 ++ List.replicate 2 2).prod) :=
by sorry

end NUMINAMATH_CALUDE_optimal_decomposition_2008_l1117_111746


namespace NUMINAMATH_CALUDE_seven_ways_to_make_eight_cents_l1117_111758

/-- Represents the number of ways to make a certain amount with given coins -/
def WaysToMakeAmount (oneCent twoCent fiveCent target : ℕ) : ℕ := sorry

/-- Theorem stating that there are exactly 7 ways to make 8 cents with the given coins -/
theorem seven_ways_to_make_eight_cents :
  WaysToMakeAmount 8 4 1 8 = 7 := by sorry

end NUMINAMATH_CALUDE_seven_ways_to_make_eight_cents_l1117_111758


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l1117_111794

theorem arithmetic_sequence_sum_divisibility :
  ∀ (a d : ℕ+), 
  ∃ (k : ℕ), (15 : ℕ) * (a + 7 * d) = k := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l1117_111794


namespace NUMINAMATH_CALUDE_range_of_a_l1117_111709

theorem range_of_a (x a : ℝ) : 
  (∀ x, x < 0 → x < a) ∧ (∃ x, x ≥ 0 ∧ x < a) → 
  a > 0 ∧ ∀ ε > 0, ∃ b, b > a ∧ b < a + ε :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1117_111709


namespace NUMINAMATH_CALUDE_neon_signs_blink_together_l1117_111784

theorem neon_signs_blink_together : Nat.lcm (Nat.lcm 7 11) 13 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_neon_signs_blink_together_l1117_111784


namespace NUMINAMATH_CALUDE_katie_cookies_l1117_111768

def pastry_sale (cupcakes sold leftover : ℕ) : Prop :=
  ∃ (cookies total : ℕ),
    total = sold + leftover ∧
    total = cupcakes + cookies ∧
    cupcakes = 7 ∧
    sold = 4 ∧
    leftover = 8 ∧
    cookies = 5

theorem katie_cookies : pastry_sale 7 4 8 := by
  sorry

end NUMINAMATH_CALUDE_katie_cookies_l1117_111768


namespace NUMINAMATH_CALUDE_correlation_theorem_l1117_111775

/-- A function to represent the relationship between x and y -/
def f (x : ℝ) : ℝ := 0.1 * x - 10

/-- Definition of positive correlation -/
def positively_correlated (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- Definition of negative correlation -/
def negatively_correlated (f g : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → g x₁ > g x₂

/-- The main theorem -/
theorem correlation_theorem (z : ℝ → ℝ) 
  (h : negatively_correlated f z) :
  positively_correlated f ∧ negatively_correlated id z := by
  sorry

end NUMINAMATH_CALUDE_correlation_theorem_l1117_111775


namespace NUMINAMATH_CALUDE_problem_solution_l1117_111729

theorem problem_solution (x y : ℝ) :
  (Real.sqrt (x - 3 * y) + |x^2 - 9|) / ((x + 3)^2) = 0 →
  Real.sqrt (x + 2) / Real.sqrt (y + 1) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1117_111729


namespace NUMINAMATH_CALUDE_xy_sum_l1117_111753

theorem xy_sum (x y : ℕ) : 
  0 < x ∧ x < 20 ∧ 0 < y ∧ y < 20 ∧ x + y + x * y = 95 → x + y = 18 ∨ x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_l1117_111753


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l1117_111736

theorem longest_side_of_triangle (x : ℝ) : 
  5 + (2*x + 3) + (3*x - 2) = 41 →
  max 5 (max (2*x + 3) (3*x - 2)) = 19 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l1117_111736


namespace NUMINAMATH_CALUDE_base_7_divisibility_l1117_111750

def base_7_to_decimal (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7 + d

def is_divisible_by_9 (n : ℕ) : Prop :=
  ∃ k, n = 9 * k

theorem base_7_divisibility (x : ℕ) :
  (x < 7) →
  (is_divisible_by_9 (base_7_to_decimal 4 5 x 2)) ↔ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_base_7_divisibility_l1117_111750


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1117_111771

theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let cylinder_volume := π * r^2 * h
  let cone_volume := (1/3) * π * r^2 * h
  cylinder_volume = 72 * π → cone_volume = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1117_111771


namespace NUMINAMATH_CALUDE_martha_has_19_butterflies_l1117_111796

/-- The number of butterflies in Martha's collection -/
structure ButterflyCollection where
  blue : ℕ
  yellow : ℕ
  black : ℕ

/-- Martha's butterfly collection satisfies the given conditions -/
def marthasCollection : ButterflyCollection where
  blue := 6
  yellow := 3
  black := 10

/-- The total number of butterflies in a collection -/
def totalButterflies (c : ButterflyCollection) : ℕ :=
  c.blue + c.yellow + c.black

/-- Theorem stating that Martha's collection has 19 butterflies in total -/
theorem martha_has_19_butterflies :
  totalButterflies marthasCollection = 19 ∧
  marthasCollection.blue = 2 * marthasCollection.yellow :=
by
  sorry


end NUMINAMATH_CALUDE_martha_has_19_butterflies_l1117_111796


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1117_111741

theorem quadratic_root_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + a = 0 → x > 1) → 
  (3 < a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1117_111741


namespace NUMINAMATH_CALUDE_product_decreasing_implies_inequality_l1117_111747

theorem product_decreasing_implies_inequality
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, (deriv f x) * g x + f x * (deriv g x) < 0)
  (a b x : ℝ)
  (h_x : a < x ∧ x < b) :
  f x * g x > f b * g b :=
sorry

end NUMINAMATH_CALUDE_product_decreasing_implies_inequality_l1117_111747


namespace NUMINAMATH_CALUDE_strongroom_keys_l1117_111788

theorem strongroom_keys (n : ℕ) : n > 0 → (
  (∃ (key_distribution : Fin 5 → Finset (Fin 10)),
    (∀ d : Fin 5, (key_distribution d).card = n) ∧
    (∀ majority : Finset (Fin 5), majority.card ≥ 3 →
      (majority.biUnion key_distribution).card = 10) ∧
    (∀ minority : Finset (Fin 5), minority.card ≤ 2 →
      (minority.biUnion key_distribution).card < 10))
  ↔ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_strongroom_keys_l1117_111788


namespace NUMINAMATH_CALUDE_power_sum_value_l1117_111783

theorem power_sum_value (a m n : ℝ) (h1 : a^m = 5) (h2 : a^n = 3) :
  a^(m + n) = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_value_l1117_111783


namespace NUMINAMATH_CALUDE_range_of_a_l1117_111776

theorem range_of_a (x : ℝ) (h : x > 1) :
  ∃ (S : Set ℝ), S = {a : ℝ | a ≤ x + 1 / (x - 1)} ∧ 
  ∀ (ε : ℝ), ε > 0 → ∃ (a : ℝ), a ∈ S ∧ a > 3 - ε :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1117_111776


namespace NUMINAMATH_CALUDE_line_perp_from_plane_perp_l1117_111705

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- State the theorem
theorem line_perp_from_plane_perp
  (m n : Line) (α β : Plane)
  (h1 : perp_line_plane m α)
  (h2 : perp_line_plane n β)
  (h3 : perp_plane α β) :
  perp_line m n :=
sorry

end NUMINAMATH_CALUDE_line_perp_from_plane_perp_l1117_111705


namespace NUMINAMATH_CALUDE_floor_tiles_l1117_111710

theorem floor_tiles (n : ℕ) : 
  n % 3 = 0 ∧ 
  2 * (2 * n / 3) - 1 = 49 → 
  n^2 - (n / 3)^2 = 1352 := by
sorry

end NUMINAMATH_CALUDE_floor_tiles_l1117_111710


namespace NUMINAMATH_CALUDE_speaker_cost_correct_l1117_111779

/-- The amount Keith spent on speakers -/
def speaker_cost : ℚ := 136.01

/-- The amount Keith spent on a CD player -/
def cd_player_cost : ℚ := 139.38

/-- The amount Keith spent on new tires -/
def tire_cost : ℚ := 112.46

/-- The total amount Keith spent -/
def total_cost : ℚ := 387.85

/-- Theorem stating that the speaker cost is correct given the other expenses -/
theorem speaker_cost_correct : 
  speaker_cost = total_cost - (cd_player_cost + tire_cost) := by
  sorry

end NUMINAMATH_CALUDE_speaker_cost_correct_l1117_111779


namespace NUMINAMATH_CALUDE_longest_side_range_l1117_111792

/-- An obtuse triangle with sides a, b, and c, where c is the longest side -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  c_longest : c ≥ max a b
  obtuse : c^2 > a^2 + b^2

/-- The theorem stating the range of the longest side in a specific obtuse triangle -/
theorem longest_side_range (t : ObtuseTriangle) (ha : t.a = 1) (hb : t.b = 2) :
  Real.sqrt 5 < t.c ∧ t.c < 3 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_range_l1117_111792


namespace NUMINAMATH_CALUDE_meaningful_fraction_l1117_111726

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l1117_111726


namespace NUMINAMATH_CALUDE_billy_final_lap_is_150_seconds_l1117_111799

/-- Represents the swimming competition between Billy and Margaret -/
structure SwimmingCompetition where
  billy_first_5_laps : ℕ  -- time in seconds
  billy_next_3_laps : ℕ  -- time in seconds
  billy_9th_lap : ℕ      -- time in seconds
  margaret_total_time : ℕ -- time in seconds
  billy_win_margin : ℕ   -- time in seconds

/-- Calculates Billy's final lap time given the competition details -/
def billy_final_lap_time (comp : SwimmingCompetition) : ℕ :=
  comp.margaret_total_time - comp.billy_win_margin - 
  (comp.billy_first_5_laps + comp.billy_next_3_laps + comp.billy_9th_lap)

/-- Theorem stating that Billy's final lap time is 150 seconds -/
theorem billy_final_lap_is_150_seconds (comp : SwimmingCompetition) 
  (h1 : comp.billy_first_5_laps = 120)
  (h2 : comp.billy_next_3_laps = 240)
  (h3 : comp.billy_9th_lap = 60)
  (h4 : comp.margaret_total_time = 600)
  (h5 : comp.billy_win_margin = 30) :
  billy_final_lap_time comp = 150 := by
  sorry

end NUMINAMATH_CALUDE_billy_final_lap_is_150_seconds_l1117_111799


namespace NUMINAMATH_CALUDE_complement_union_A_B_l1117_111730

-- Define the sets A and B
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_union_A_B : 
  (A ∪ B)ᶜ = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l1117_111730


namespace NUMINAMATH_CALUDE_matrix_multiplication_proof_l1117_111786

theorem matrix_multiplication_proof :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![2, -6; -1, 3]
  A * B = !![8, -24; 3, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_proof_l1117_111786


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l1117_111743

/-- The area of a square with adjacent vertices at (0,3) and (4,0) is 25. -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 3)
  let p2 : ℝ × ℝ := (4, 0)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l1117_111743


namespace NUMINAMATH_CALUDE_rectangle_length_l1117_111767

/-- Given a rectangle with a length to width ratio of 6:5 and a width of 20 inches,
    prove that its length is 24 inches. -/
theorem rectangle_length (width : ℝ) (length : ℝ) : 
  width = 20 → length / width = 6 / 5 → length = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l1117_111767


namespace NUMINAMATH_CALUDE_personal_trainer_cost_l1117_111702

-- Define the given conditions
def old_hourly_wage : ℚ := 40
def raise_percentage : ℚ := 5 / 100
def hours_per_day : ℚ := 8
def days_per_week : ℚ := 5
def old_bills : ℚ := 600
def leftover : ℚ := 980

-- Define the theorem
theorem personal_trainer_cost :
  let new_hourly_wage := old_hourly_wage * (1 + raise_percentage)
  let weekly_earnings := new_hourly_wage * hours_per_day * days_per_week
  let total_expenses := weekly_earnings - leftover
  total_expenses - old_bills = 100 := by sorry

end NUMINAMATH_CALUDE_personal_trainer_cost_l1117_111702


namespace NUMINAMATH_CALUDE_women_at_gathering_l1117_111790

/-- Represents a social gathering with men and women dancing -/
structure SocialGathering where
  men : ℕ
  women : ℕ
  man_partners : ℕ
  woman_partners : ℕ

/-- Calculates the total number of dance pairs -/
def total_pairs (g : SocialGathering) : ℕ := g.men * g.man_partners

/-- Theorem: In a social gathering where 15 men attended, each man danced with 4 women,
    and each woman danced with 3 men, the number of women who attended is 20. -/
theorem women_at_gathering (g : SocialGathering) 
  (h1 : g.men = 15)
  (h2 : g.man_partners = 4)
  (h3 : g.woman_partners = 3)
  (h4 : total_pairs g = g.women * g.woman_partners) :
  g.women = 20 := by
  sorry

end NUMINAMATH_CALUDE_women_at_gathering_l1117_111790


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l1117_111744

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the number of population changes per day -/
def changes_per_day : ℕ := seconds_per_day / 2

/-- Represents the death rate in people per two seconds -/
def death_rate : ℕ := 2

/-- Represents the daily net population increase -/
def daily_net_increase : ℕ := 345600

/-- Represents the average birth rate in people per two seconds -/
def birth_rate : ℕ := 10

theorem birth_rate_calculation :
  (birth_rate - death_rate) * changes_per_day = daily_net_increase :=
by sorry

end NUMINAMATH_CALUDE_birth_rate_calculation_l1117_111744


namespace NUMINAMATH_CALUDE_katarina_miles_l1117_111766

/-- The total miles run by all four runners -/
def total_miles : ℕ := 195

/-- The number of miles run by Harriet -/
def harriet_miles : ℕ := 48

/-- The number of runners who ran the same distance as Harriet -/
def same_distance_runners : ℕ := 3

theorem katarina_miles : 
  total_miles - harriet_miles * same_distance_runners = 51 := by sorry

end NUMINAMATH_CALUDE_katarina_miles_l1117_111766


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1117_111777

-- Define the variables
variable (a b c p q r : ℝ)

-- Define the conditions
axiom eq1 : 17 * p + b * q + c * r = 0
axiom eq2 : a * p + 29 * q + c * r = 0
axiom eq3 : a * p + b * q + 56 * r = 0
axiom a_ne_17 : a ≠ 17
axiom p_ne_0 : p ≠ 0

-- State the theorem
theorem sum_of_fractions_equals_one :
  a / (a - 17) + b / (b - 29) + c / (c - 56) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1117_111777


namespace NUMINAMATH_CALUDE_carlos_singles_percentage_l1117_111791

/-- Represents the number of hits for each type of hit in baseball --/
structure HitCounts where
  total : ℕ
  homeRuns : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the percentage of singles given the hit counts --/
def percentageSingles (hits : HitCounts) : ℚ :=
  let singles := hits.total - (hits.homeRuns + hits.triples + hits.doubles)
  (singles : ℚ) / hits.total * 100

/-- Carlos's hit counts for the baseball season --/
def carlosHits : HitCounts :=
  { total := 50
  , homeRuns := 3
  , triples := 2
  , doubles := 8 }

/-- Theorem stating that the percentage of Carlos's hits that were singles is 74% --/
theorem carlos_singles_percentage :
  percentageSingles carlosHits = 74 := by
  sorry


end NUMINAMATH_CALUDE_carlos_singles_percentage_l1117_111791


namespace NUMINAMATH_CALUDE_get_ready_time_l1117_111712

/-- The time it takes for Jack and his two toddlers to get ready -/
def total_time (jack_socks jack_shoes jack_jacket toddler_socks toddler_shoes toddler_shoelaces : ℕ) : ℕ :=
  let jack_time := jack_socks + jack_shoes + jack_jacket
  let toddler_time := toddler_socks + toddler_shoes + 2 * toddler_shoelaces
  jack_time + 2 * toddler_time

theorem get_ready_time :
  total_time 2 4 3 2 5 1 = 27 :=
by sorry

end NUMINAMATH_CALUDE_get_ready_time_l1117_111712


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l1117_111781

theorem termite_ridden_not_collapsing (total_homes : ℕ) (termite_ridden : ℕ) (collapsing : ℕ) :
  termite_ridden = total_homes / 3 →
  collapsing = termite_ridden / 4 →
  (termite_ridden - collapsing) = total_homes / 4 :=
by sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l1117_111781


namespace NUMINAMATH_CALUDE_platform_length_l1117_111714

/-- Given a train of length 600 m that takes 78 seconds to cross a platform
    and 52 seconds to cross a signal pole, prove that the length of the platform is 300 m. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ)
  (h1 : train_length = 600)
  (h2 : time_platform = 78)
  (h3 : time_pole = 52) :
  (train_length * time_platform / time_pole) - train_length = 300 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1117_111714


namespace NUMINAMATH_CALUDE_fraction_of_muscle_gain_as_fat_l1117_111720

/-- Calculates the fraction of muscle gain that is fat given initial weight, muscle gain percentage, and final weight. -/
theorem fraction_of_muscle_gain_as_fat 
  (initial_weight : ℝ) 
  (muscle_gain_percentage : ℝ) 
  (final_weight : ℝ) 
  (h1 : initial_weight = 120)
  (h2 : muscle_gain_percentage = 0.20)
  (h3 : final_weight = 150) :
  (final_weight - initial_weight - muscle_gain_percentage * initial_weight) / (muscle_gain_percentage * initial_weight) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_muscle_gain_as_fat_l1117_111720


namespace NUMINAMATH_CALUDE_even_and_increasing_order_l1117_111772

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem even_and_increasing_order (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : increasing_on_nonneg f) : 
  f (-2) < f 3 ∧ f 3 < f (-π) := by
  sorry

end NUMINAMATH_CALUDE_even_and_increasing_order_l1117_111772


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l1117_111700

theorem multiplication_subtraction_equality : 72 * 989 - 12 * 989 = 59340 := by sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l1117_111700


namespace NUMINAMATH_CALUDE_divisibility_property_l1117_111754

theorem divisibility_property (a b d : ℕ+) 
  (h1 : (a + b : ℕ) % d = 0)
  (h2 : (a * b : ℕ) % (d * d) = 0) :
  (a : ℕ) % d = 0 ∧ (b : ℕ) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1117_111754


namespace NUMINAMATH_CALUDE_min_rounds_for_sole_winner_l1117_111707

/-- Represents a chess tournament -/
structure ChessTournament where
  num_players : ℕ
  num_rounds : ℕ
  points_per_win : ℚ
  points_per_draw : ℚ
  points_per_loss : ℚ

/-- Checks if a tournament configuration allows for a sole winner -/
def has_sole_winner (t : ChessTournament) : Prop :=
  ∃ (leader_score : ℚ) (max_other_score : ℚ),
    leader_score > max_other_score ∧
    leader_score ≤ t.num_rounds * t.points_per_win ∧
    max_other_score ≤ (t.num_rounds - 1) * t.points_per_win + t.points_per_draw

/-- The main theorem stating the minimum number of rounds for a sole winner -/
theorem min_rounds_for_sole_winner :
  ∀ (t : ChessTournament),
    t.num_players = 10 →
    t.points_per_win = 1 →
    t.points_per_draw = 1/2 →
    t.points_per_loss = 0 →
    (∀ n : ℕ, n < 7 → ¬(has_sole_winner {num_players := t.num_players,
                                         num_rounds := n,
                                         points_per_win := t.points_per_win,
                                         points_per_draw := t.points_per_draw,
                                         points_per_loss := t.points_per_loss})) ∧
    (has_sole_winner {num_players := t.num_players,
                      num_rounds := 7,
                      points_per_win := t.points_per_win,
                      points_per_draw := t.points_per_draw,
                      points_per_loss := t.points_per_loss}) :=
by
  sorry

end NUMINAMATH_CALUDE_min_rounds_for_sole_winner_l1117_111707


namespace NUMINAMATH_CALUDE_student_calculation_difference_l1117_111706

theorem student_calculation_difference : 
  let number : ℝ := 100.00000000000003
  let correct_answer := number * (4/5 : ℝ)
  let student_answer := number / (4/5 : ℝ)
  student_answer - correct_answer = 45.00000000000002 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_difference_l1117_111706


namespace NUMINAMATH_CALUDE_first_valid_year_is_1979_l1117_111793

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 1970 ∧ sum_of_digits year = 15

theorem first_valid_year_is_1979 :
  (∀ y : ℕ, y < 1979 → ¬(is_valid_year y)) ∧ is_valid_year 1979 := by
  sorry

end NUMINAMATH_CALUDE_first_valid_year_is_1979_l1117_111793


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1117_111735

/-- Discriminant of a quadratic equation ax² + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Condition for a quadratic equation to have two equal real roots -/
def has_two_equal_real_roots (a b c : ℝ) : Prop := discriminant a b c = 0

theorem quadratic_equal_roots :
  has_two_equal_real_roots 1 (-2) 1 ∧
  ¬has_two_equal_real_roots 1 (-3) 2 ∧
  ¬has_two_equal_real_roots 1 (-2) 3 ∧
  ¬has_two_equal_real_roots 1 0 (-9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1117_111735


namespace NUMINAMATH_CALUDE_arithmetic_geometric_k4_l1117_111722

def arithmetic_geometric_sequence (a : ℕ → ℝ) (d k : ℕ → ℕ) : Prop :=
  (∃ (c : ℝ), c ≠ 0 ∧ ∀ n, a (n + 1) = a n + c) ∧
  (∃ (q : ℝ), q ≠ 0 ∧ q ≠ 1 ∧ ∀ n, a (k (n + 1)) = a (k n) * q) ∧
  k 1 = 1 ∧ k 2 = 2 ∧ k 3 = 6

theorem arithmetic_geometric_k4 (a : ℕ → ℝ) (d k : ℕ → ℕ) :
  arithmetic_geometric_sequence a d k → k 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_k4_l1117_111722


namespace NUMINAMATH_CALUDE_james_monthly_income_l1117_111721

/-- Represents a subscription tier with its subscriber count and price --/
structure SubscriptionTier where
  subscribers : ℕ
  price : ℚ

/-- Calculates the total monthly income for James from Twitch subscriptions --/
def calculate_monthly_income (tier1 tier2 tier3 : SubscriptionTier) : ℚ :=
  tier1.subscribers * tier1.price +
  tier2.subscribers * tier2.price +
  tier3.subscribers * tier3.price

/-- Theorem stating that James' monthly income from Twitch subscriptions is $2522.50 --/
theorem james_monthly_income :
  let tier1 := SubscriptionTier.mk (120 + 10) (499 / 100)
  let tier2 := SubscriptionTier.mk (50 + 25) (999 / 100)
  let tier3 := SubscriptionTier.mk (30 + 15) (2499 / 100)
  calculate_monthly_income tier1 tier2 tier3 = 252250 / 100 := by
  sorry


end NUMINAMATH_CALUDE_james_monthly_income_l1117_111721


namespace NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_ap_l1117_111773

/-- A cubic polynomial with coefficients a and b -/
def cubic_polynomial (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- Predicate for a cubic polynomial having three distinct roots in arithmetic progression -/
def has_three_distinct_roots_in_ap (a b : ℝ) : Prop :=
  ∃ (r d : ℝ), d ≠ 0 ∧
    cubic_polynomial a b (-d) = 0 ∧
    cubic_polynomial a b 0 = 0 ∧
    cubic_polynomial a b d = 0

/-- Theorem stating the condition for a cubic polynomial to have three distinct roots in arithmetic progression -/
theorem cubic_three_distinct_roots_in_ap (a b : ℝ) :
  has_three_distinct_roots_in_ap a b ↔ b = 0 ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_ap_l1117_111773


namespace NUMINAMATH_CALUDE_gravel_path_cost_l1117_111769

/-- Calculate the cost of gravelling a path around a rectangular plot -/
theorem gravel_path_cost 
  (plot_length : ℝ) 
  (plot_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm_paise : ℝ) : 
  plot_length = 150 ∧ 
  plot_width = 95 ∧ 
  path_width = 4.5 ∧ 
  cost_per_sqm_paise = 90 → 
  (((plot_length + 2 * path_width) * (plot_width + 2 * path_width) - 
    plot_length * plot_width) * 
   (cost_per_sqm_paise / 100)) = 2057.40 :=
by sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l1117_111769


namespace NUMINAMATH_CALUDE_total_buildable_area_l1117_111762

def num_sections : ℕ := 7
def section_area : ℝ := 9473
def open_space_percent : ℝ := 0.15

theorem total_buildable_area :
  (num_sections : ℝ) * section_area * (1 - open_space_percent) = 56364.35 := by
  sorry

end NUMINAMATH_CALUDE_total_buildable_area_l1117_111762


namespace NUMINAMATH_CALUDE_indefinite_integral_ln_4x2_plus_1_l1117_111711

open Real

theorem indefinite_integral_ln_4x2_plus_1 (x : ℝ) :
  (deriv fun x => x * log (4 * x^2 + 1) - 8 * x + 4 * arctan (2 * x)) x = log (4 * x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_ln_4x2_plus_1_l1117_111711


namespace NUMINAMATH_CALUDE_max_area_wire_rectangle_or_square_l1117_111765

/-- The maximum area enclosed by a rectangle or square formed from a wire of length 2 meters -/
theorem max_area_wire_rectangle_or_square : 
  let wire_length : ℝ := 2
  let max_area : ℝ := (1 : ℝ) / 4
  ∀ l w : ℝ, 
    0 < l ∧ 0 < w →  -- positive length and width
    2 * (l + w) ≤ wire_length →  -- perimeter constraint
    l * w ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_max_area_wire_rectangle_or_square_l1117_111765


namespace NUMINAMATH_CALUDE_floor_equation_solution_set_l1117_111727

theorem floor_equation_solution_set (x : ℝ) :
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 5/3 ≤ x ∧ x < 7/3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_set_l1117_111727


namespace NUMINAMATH_CALUDE_max_visible_cubes_12_l1117_111713

/-- Represents a cube formed by unit cubes --/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from any single point --/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  3 * cube.size^2 - 3 * (cube.size - 1) + 1

/-- Theorem stating that for a 12 × 12 × 12 cube, the maximum number of visible unit cubes is 400 --/
theorem max_visible_cubes_12 :
  max_visible_cubes { size := 12 } = 400 := by
  sorry

#eval max_visible_cubes { size := 12 }

end NUMINAMATH_CALUDE_max_visible_cubes_12_l1117_111713


namespace NUMINAMATH_CALUDE_angle_sum_90_l1117_111742

-- Define the necessary structures
structure Plane :=
(π : Type)

structure Line :=
(l : Type)

-- Define the perpendicular relation between a line and a plane
def perpendicular (p : Line) (π : Plane) : Prop :=
sorry

-- Define the angle between a line and a plane
def angle_line_plane (l : Line) (π : Plane) : ℝ :=
sorry

-- Define the angle between two lines
def angle_between_lines (l1 l2 : Line) : ℝ :=
sorry

-- State the theorem
theorem angle_sum_90 (p : Line) (π : Plane) (l : Line) 
  (h : perpendicular p π) :
  angle_line_plane l π + angle_between_lines l p = 90 :=
sorry

end NUMINAMATH_CALUDE_angle_sum_90_l1117_111742


namespace NUMINAMATH_CALUDE_gym_occupancy_l1117_111739

theorem gym_occupancy (initial_people : ℕ) (people_came_in : ℕ) (people_left : ℕ) 
  (h1 : initial_people = 16) 
  (h2 : people_came_in = 5) 
  (h3 : people_left = 2) : 
  initial_people + people_came_in - people_left = 19 :=
by sorry

end NUMINAMATH_CALUDE_gym_occupancy_l1117_111739


namespace NUMINAMATH_CALUDE_sequence_a_property_sequence_a_positive_sequence_a_first_two_terms_sequence_a_bounds_sequence_a_decreasing_l1117_111764

def sequence_a (n : ℕ+) : ℝ := sorry

theorem sequence_a_property (n : ℕ+) : 
  sequence_a n + n * sequence_a n - 1 = 0 := sorry

theorem sequence_a_positive (n : ℕ+) : 
  sequence_a n > 0 := sorry

theorem sequence_a_first_two_terms : 
  sequence_a 1 = 1/2 ∧ sequence_a 2 = 1/4 := sorry

theorem sequence_a_bounds (n : ℕ+) : 
  0 < sequence_a n ∧ sequence_a n < 1 := sorry

theorem sequence_a_decreasing (n : ℕ+) : 
  sequence_a n > sequence_a (n + 1) := sorry

end NUMINAMATH_CALUDE_sequence_a_property_sequence_a_positive_sequence_a_first_two_terms_sequence_a_bounds_sequence_a_decreasing_l1117_111764


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l1117_111715

theorem max_value_sum_of_roots (x y z : ℝ) 
  (sum_eq_two : x + y + z = 2)
  (x_geq_neg_one : x ≥ -1)
  (y_geq_neg_two : y ≥ -2)
  (z_geq_neg_one : z ≥ -1) :
  ∃ (M : ℝ), M = 4 * Real.sqrt 3 ∧ 
  ∀ (a b c : ℝ), a + b + c = 2 → a ≥ -1 → b ≥ -2 → c ≥ -1 →
  Real.sqrt (3 * a^2 + 3) + Real.sqrt (3 * b^2 + 6) + Real.sqrt (3 * c^2 + 3) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l1117_111715


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l1117_111763

theorem quadratic_rational_solutions (d : ℕ+) : 
  (∃ x : ℚ, 7 * x^2 + 13 * x + d.val = 0) ↔ d = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l1117_111763


namespace NUMINAMATH_CALUDE_max_sum_2023_factors_l1117_111789

theorem max_sum_2023_factors :
  ∃ (A B C : ℕ+), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A * B * C = 2023 ∧
    ∀ (X Y Z : ℕ+), 
      X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z →
      X * Y * Z = 2023 →
      X + Y + Z ≤ A + B + C ∧
      A + B + C = 297 := by
sorry

end NUMINAMATH_CALUDE_max_sum_2023_factors_l1117_111789


namespace NUMINAMATH_CALUDE_ratio_equality_l1117_111778

theorem ratio_equality (x : ℚ) : (x / (2 / 6)) = ((3 / 4) / (1 / 2)) → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1117_111778


namespace NUMINAMATH_CALUDE_sequence_minimum_term_minimum_term_value_minimum_term_occurs_at_24_l1117_111749

theorem sequence_minimum_term (n : ℕ) (h1 : 7 ≤ n) (h2 : n ≤ 95) :
  (Real.sqrt (n / 6) + Real.sqrt (96 / n : ℝ)) ≥ 4 :=
by sorry

theorem minimum_term_value (n : ℕ) (h1 : 7 ≤ n) (h2 : n ≤ 95) :
  (Real.sqrt (24 / 6) + Real.sqrt (96 / 24 : ℝ)) = 4 :=
by sorry

theorem minimum_term_occurs_at_24 :
  ∃ (n : ℕ), 7 ≤ n ∧ n ≤ 95 ∧
  (Real.sqrt (n / 6) + Real.sqrt (96 / n : ℝ)) = 4 ∧
  n = 24 :=
by sorry

end NUMINAMATH_CALUDE_sequence_minimum_term_minimum_term_value_minimum_term_occurs_at_24_l1117_111749


namespace NUMINAMATH_CALUDE_min_sum_given_log_sum_l1117_111782

theorem min_sum_given_log_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : Real.log m / Real.log 3 + Real.log n / Real.log 3 = 4) : 
  m + n ≥ 18 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 
    Real.log m₀ / Real.log 3 + Real.log n₀ / Real.log 3 = 4 ∧ m₀ + n₀ = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_given_log_sum_l1117_111782


namespace NUMINAMATH_CALUDE_correct_recommendation_count_l1117_111719

/-- Represents the number of recommendation spots for each language -/
structure SpotDistribution :=
  (korean : Nat)
  (japanese : Nat)
  (russian : Nat)

/-- Represents the gender distribution of candidates -/
structure CandidateDistribution :=
  (female : Nat)
  (male : Nat)

/-- Calculates the number of different recommendation methods -/
def recommendationMethods (spots : SpotDistribution) (candidates : CandidateDistribution) : Nat :=
  sorry

/-- Theorem stating the number of different recommendation methods -/
theorem correct_recommendation_count :
  let spots : SpotDistribution := ⟨2, 2, 1⟩
  let candidates : CandidateDistribution := ⟨3, 2⟩
  recommendationMethods spots candidates = 24 := by
  sorry

end NUMINAMATH_CALUDE_correct_recommendation_count_l1117_111719


namespace NUMINAMATH_CALUDE_binomial_coefficient_relation_l1117_111798

theorem binomial_coefficient_relation (n : ℕ) : 
  (n ≥ 2) →
  (Nat.choose n 2 * 3^(n-2) = 5 * 3^n) →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_relation_l1117_111798


namespace NUMINAMATH_CALUDE_monic_quartic_specific_values_l1117_111723

-- Define a monic quartic polynomial
def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_specific_values (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h_neg2 : f (-2) = 0)
  (h_1 : f 1 = -2)
  (h_3 : f 3 = -6)
  (h_5 : f 5 = -10) :
  f 0 = 29 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_specific_values_l1117_111723


namespace NUMINAMATH_CALUDE_exp_ln_five_l1117_111724

theorem exp_ln_five : Real.exp (Real.log 5) = 5 := by sorry

end NUMINAMATH_CALUDE_exp_ln_five_l1117_111724


namespace NUMINAMATH_CALUDE_num_paths_equals_1287_l1117_111701

/-- The number of blocks to the right -/
def blocks_right : ℕ := 8

/-- The number of blocks up -/
def blocks_up : ℕ := 5

/-- The total number of moves -/
def total_moves : ℕ := blocks_right + blocks_up

/-- The number of different shortest paths -/
def num_paths : ℕ := Nat.choose total_moves blocks_up

theorem num_paths_equals_1287 : num_paths = 1287 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_equals_1287_l1117_111701


namespace NUMINAMATH_CALUDE_sequence_problem_l1117_111761

def second_difference (a : ℕ → ℤ) : ℕ → ℤ := λ n => a (n + 2) - 2 * a (n + 1) + a n

theorem sequence_problem (a : ℕ → ℤ) 
  (h1 : ∀ n, second_difference a n = 16)
  (h2 : a 63 = 10)
  (h3 : a 89 = 10) :
  a 51 = 3658 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1117_111761


namespace NUMINAMATH_CALUDE_radical_product_simplification_l1117_111716

theorem radical_product_simplification (m : ℝ) (h : m > 0) :
  Real.sqrt (50 * m) * Real.sqrt (5 * m) * Real.sqrt (45 * m) = 15 * m * Real.sqrt (10 * m) :=
by sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l1117_111716


namespace NUMINAMATH_CALUDE_video_dislikes_calculation_l1117_111732

/-- Calculates the final number of dislikes for a video given initial likes, 
    initial dislikes formula, and additional dislikes. -/
def final_dislikes (initial_likes : ℕ) (additional_dislikes : ℕ) : ℕ :=
  (initial_likes / 2 + 100) + additional_dislikes

/-- Theorem stating that for a video with 3000 initial likes and 1000 additional dislikes,
    the final number of dislikes is 2600. -/
theorem video_dislikes_calculation :
  final_dislikes 3000 1000 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_video_dislikes_calculation_l1117_111732


namespace NUMINAMATH_CALUDE_three_dozen_quarters_value_l1117_111733

/-- Proves that 3 dozen quarters is equal to $9 --/
theorem three_dozen_quarters_value : 
  let dozen : ℕ := 12
  let quarter_value : ℕ := 25  -- in cents
  let cents_per_dollar : ℕ := 100
  (3 * dozen * quarter_value) / cents_per_dollar = 9 := by
  sorry

end NUMINAMATH_CALUDE_three_dozen_quarters_value_l1117_111733


namespace NUMINAMATH_CALUDE_trebled_result_l1117_111752

theorem trebled_result (x : ℕ) (h : x = 7) : 3 * ((2 * x) + 9) = 69 := by
  sorry

end NUMINAMATH_CALUDE_trebled_result_l1117_111752


namespace NUMINAMATH_CALUDE_larger_number_proof_l1117_111774

theorem larger_number_proof (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 23)
  (lcm_eq : Nat.lcm a b = 23 * 14 * 15) :
  max a b = 345 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1117_111774


namespace NUMINAMATH_CALUDE_lewis_items_count_l1117_111745

theorem lewis_items_count (tanya samantha lewis james : ℕ) : 
  tanya = 4 →
  samantha = 4 * tanya →
  lewis = samantha - (samantha / 3) →
  james = 2 * lewis →
  lewis = 11 := by
sorry

end NUMINAMATH_CALUDE_lewis_items_count_l1117_111745


namespace NUMINAMATH_CALUDE_poll_percentage_equal_l1117_111757

theorem poll_percentage_equal (total : ℕ) (women_favor_percent : ℚ) (women_opposed : ℕ) : 
  total = 120 → women_favor_percent = 35/100 → women_opposed = 39 →
  ∃ (women men : ℕ), 
    women + men = total ∧ 
    women_opposed = (1 - women_favor_percent) * women ∧
    women = men ∧ 
    women / total = 1/2 ∧ 
    men / total = 1/2 := by
  sorry

#check poll_percentage_equal

end NUMINAMATH_CALUDE_poll_percentage_equal_l1117_111757


namespace NUMINAMATH_CALUDE_membership_change_fall_increase_value_l1117_111718

/-- The percentage increase in membership during fall -/
def fall_increase : ℝ := sorry

/-- The percentage decrease in membership during spring -/
def spring_decrease : ℝ := 19

/-- The total percentage increase from original to spring membership -/
def total_increase : ℝ := 12.52

/-- Theorem stating the relationship between fall increase, spring decrease, and total increase -/
theorem membership_change :
  (1 + fall_increase / 100) * (1 - spring_decrease / 100) = 1 + total_increase / 100 :=
sorry

/-- The fall increase is approximately 38.91% -/
theorem fall_increase_value : 
  ∃ ε > 0, |fall_increase - 38.91| < ε :=
sorry

end NUMINAMATH_CALUDE_membership_change_fall_increase_value_l1117_111718


namespace NUMINAMATH_CALUDE_center_square_side_length_l1117_111725

theorem center_square_side_length : 
  let large_square_side : ℝ := 120
  let total_area : ℝ := large_square_side ^ 2
  let l_shape_area : ℝ := (1 / 5) * total_area
  let center_square_area : ℝ := total_area - 4 * l_shape_area
  let center_square_side : ℝ := Real.sqrt center_square_area
  center_square_side = 54 := by
  sorry

end NUMINAMATH_CALUDE_center_square_side_length_l1117_111725


namespace NUMINAMATH_CALUDE_nine_crosses_fit_on_chessboard_l1117_111731

/-- Represents a cross pentomino -/
structure CrossPentomino :=
  (size : ℕ := 5)

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- The area of a cross pentomino -/
def cross_pentomino_area (c : CrossPentomino) : ℕ := c.size

/-- The area of a chessboard -/
def chessboard_area (b : Chessboard) : ℕ := b.rows * b.cols

/-- Theorem: Nine cross pentominoes can fit on an 8x8 chessboard -/
theorem nine_crosses_fit_on_chessboard :
  ∃ (c : CrossPentomino) (b : Chessboard),
    b.rows = 8 ∧ b.cols = 8 ∧
    9 * (cross_pentomino_area c) ≤ chessboard_area b :=
by sorry

end NUMINAMATH_CALUDE_nine_crosses_fit_on_chessboard_l1117_111731


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l1117_111737

def monthly_salary : ℝ := 5750
def initial_savings_rate : ℝ := 0.20
def new_savings : ℝ := 230

theorem expense_increase_percentage :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_savings - new_savings
  (expense_increase / initial_expenses) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l1117_111737


namespace NUMINAMATH_CALUDE_store_purchase_cost_l1117_111755

/-- Given the prices of pens, notebooks, and pencils satisfying certain conditions,
    prove that 4 pens, 5 notebooks, and 5 pencils cost 73 rubles. -/
theorem store_purchase_cost (pen_price notebook_price pencil_price : ℚ) :
  (2 * pen_price + 3 * notebook_price + pencil_price = 33) →
  (pen_price + notebook_price + 2 * pencil_price = 20) →
  (4 * pen_price + 5 * notebook_price + 5 * pencil_price = 73) :=
by sorry

end NUMINAMATH_CALUDE_store_purchase_cost_l1117_111755


namespace NUMINAMATH_CALUDE_pencil_cost_l1117_111708

theorem pencil_cost (pen_price pencil_price : ℚ) : 
  3 * pen_price + 2 * pencil_price = 165/100 →
  4 * pen_price + 7 * pencil_price = 303/100 →
  pencil_price = 19155/100000 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_l1117_111708


namespace NUMINAMATH_CALUDE_square_numbers_between_20_and_120_divisible_by_3_l1117_111740

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem square_numbers_between_20_and_120_divisible_by_3 :
  {x : ℕ | is_square x ∧ x % 3 = 0 ∧ 20 < x ∧ x < 120} = {36, 81} := by
  sorry

end NUMINAMATH_CALUDE_square_numbers_between_20_and_120_divisible_by_3_l1117_111740


namespace NUMINAMATH_CALUDE_rachel_plant_arrangement_l1117_111751

/-- Represents the number of ways to arrange plants under lamps -/
def arrangement_count (cactus_count : ℕ) (orchid_count : ℕ) (yellow_lamp_count : ℕ) (blue_lamp_count : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating the number of arrangements for the given problem -/
theorem rachel_plant_arrangement :
  arrangement_count 3 1 3 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_rachel_plant_arrangement_l1117_111751


namespace NUMINAMATH_CALUDE_sum_of_even_integers_ranges_l1117_111748

def S1 : ℕ := (100 / 2) * (2 + 200)

def S2 : ℕ := (150 / 2) * (102 + 400)

theorem sum_of_even_integers_ranges (R : ℕ) : R = S1 + S2 → R = 47750 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_ranges_l1117_111748


namespace NUMINAMATH_CALUDE_negation_of_square_non_negative_l1117_111759

theorem negation_of_square_non_negative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_square_non_negative_l1117_111759


namespace NUMINAMATH_CALUDE_football_team_yardage_l1117_111734

theorem football_team_yardage (initial_loss : ℤ) (gain : ℤ) (final_progress : ℤ) : 
  gain = 9 ∧ final_progress = 4 → initial_loss = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_football_team_yardage_l1117_111734


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1117_111756

/-- A line passing through (4,1) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (4,1) -/
  point_condition : m * 4 + b = 1
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b ≠ 0 → -b = b / m

/-- The equation of an EqualInterceptLine is either x - 4y = 0 or x + y - 5 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 1/4 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 5) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1117_111756


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1117_111704

theorem unique_solution_for_equation (x y : ℝ) :
  (x - 6)^2 + (y - 7)^2 + (x - y)^2 = 1/3 ↔ x = 19/3 ∧ y = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1117_111704


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1117_111738

/-- The coordinates of a point in a 2D Cartesian coordinate system. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Theorem: The coordinates of the point (1, -2) with respect to the origin
    in a Cartesian coordinate system are (1, -2). -/
theorem point_coordinates_wrt_origin (p : Point2D) (h : p = ⟨1, -2⟩) :
  p.x = 1 ∧ p.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1117_111738


namespace NUMINAMATH_CALUDE_composite_and_infinite_x_l1117_111770

theorem composite_and_infinite_x (a : ℕ) :
  (∃ x : ℕ, ∃ y z : ℕ, y > 1 ∧ z > 1 ∧ a * x + 1 = y * z) ∧
  (∀ n : ℕ, ∃ x : ℕ, x > n ∧ ∃ y z : ℕ, y > 1 ∧ z > 1 ∧ a * x + 1 = y * z) :=
by sorry

end NUMINAMATH_CALUDE_composite_and_infinite_x_l1117_111770


namespace NUMINAMATH_CALUDE_complex_arithmetic_proof_l1117_111795

theorem complex_arithmetic_proof : ((2 : ℂ) + 5*I + (3 : ℂ) - 6*I) * ((1 : ℂ) + 2*I) = (7 : ℂ) + 9*I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_proof_l1117_111795


namespace NUMINAMATH_CALUDE_particle_speed_l1117_111797

/-- Given a particle with position (3t + 4, 5t - 9) at time t, 
    its speed after a time interval of 2 units is √136. -/
theorem particle_speed (t : ℝ) : 
  let pos (t : ℝ) := (3 * t + 4, 5 * t - 9)
  let Δt := 2
  let Δx := (pos (t + Δt)).1 - (pos t).1
  let Δy := (pos (t + Δt)).2 - (pos t).2
  Real.sqrt (Δx ^ 2 + Δy ^ 2) = Real.sqrt 136 := by
  sorry

end NUMINAMATH_CALUDE_particle_speed_l1117_111797


namespace NUMINAMATH_CALUDE_lg_properties_l1117_111703

-- Define the base 10 logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_properties :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 →
    (lg (x₁ * x₂) = lg x₁ + lg x₂) ∧
    (x₁ ≠ x₂ → (lg x₁ - lg x₂) / (x₁ - x₂) > 0) :=
by sorry

end NUMINAMATH_CALUDE_lg_properties_l1117_111703


namespace NUMINAMATH_CALUDE_jazmin_dolls_count_l1117_111780

/-- The number of dolls Geraldine has -/
def geraldine_dolls : ℕ := 2186

/-- The total number of dolls Jazmin and Geraldine have together -/
def total_dolls : ℕ := 3395

/-- The number of dolls Jazmin has -/
def jazmin_dolls : ℕ := total_dolls - geraldine_dolls

theorem jazmin_dolls_count : jazmin_dolls = 1209 := by
  sorry

end NUMINAMATH_CALUDE_jazmin_dolls_count_l1117_111780


namespace NUMINAMATH_CALUDE_two_solutions_l1117_111717

/-- The number of ordered pairs of integers (x, y) satisfying x^4 + y^2 = 4y -/
def count_solutions : ℕ := 2

/-- Predicate that checks if a pair of integers satisfies the equation -/
def satisfies_equation (x y : ℤ) : Prop :=
  x^4 + y^2 = 4*y

theorem two_solutions :
  (∃! (s : Finset (ℤ × ℤ)), s.card = count_solutions ∧ 
    ∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_equation p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_l1117_111717


namespace NUMINAMATH_CALUDE_bubble_pass_probability_specific_l1117_111787

/-- The probability that in a sequence of n distinct terms,
    the kth term ends up in the mth position after one bubble pass. -/
def bubble_pass_probability (n k m : ℕ) : ℚ :=
  if k ≤ m ∧ m < n then 1 / ((m - k + 2) * (m - k + 1))
  else 0

theorem bubble_pass_probability_specific :
  bubble_pass_probability 50 25 40 = 1 / 272 := by
  sorry

end NUMINAMATH_CALUDE_bubble_pass_probability_specific_l1117_111787


namespace NUMINAMATH_CALUDE_blue_face_ratio_one_third_l1117_111760

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- The number of blue faces after cutting the cube into unit cubes -/
def blue_faces (c : Cube n) : ℕ := 6 * n^2

/-- The total number of faces of all unit cubes -/
def total_faces (c : Cube n) : ℕ := 6 * n^3

/-- The theorem stating that the ratio of blue faces to total faces is 1/3 iff n = 3 -/
theorem blue_face_ratio_one_third (n : ℕ) (c : Cube n) :
  (blue_faces c : ℚ) / (total_faces c : ℚ) = 1/3 ↔ n = 3 := by sorry

end NUMINAMATH_CALUDE_blue_face_ratio_one_third_l1117_111760


namespace NUMINAMATH_CALUDE_inequalities_given_sum_positive_l1117_111728

theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_sum_positive_l1117_111728


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1117_111785

/-- The focal length of an ellipse with equation x^2 + 2y^2 = 2 is 2 -/
theorem ellipse_focal_length : 
  let ellipse_eq : ℝ → ℝ → Prop := λ x y => x^2 + 2*y^2 = 2
  ∃ a b c : ℝ, 
    (∀ x y, ellipse_eq x y ↔ (x^2 / a^2) + (y^2 / b^2) = 1) ∧ 
    c^2 = a^2 - b^2 ∧
    2 * c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1117_111785
