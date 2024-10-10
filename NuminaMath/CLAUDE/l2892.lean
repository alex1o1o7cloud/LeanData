import Mathlib

namespace min_games_for_condition_l2892_289239

/-- The number of teams in the tournament -/
def num_teams : ℕ := 16

/-- The total number of possible games in a round-robin tournament -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The maximum number of non-played games such that no three teams are mutually non-played -/
def max_non_played_games : ℕ := (num_teams / 2) ^ 2

/-- The minimum number of games that must be played to satisfy the condition -/
def min_games_played : ℕ := total_games - max_non_played_games

theorem min_games_for_condition : min_games_played = 56 := by sorry

end min_games_for_condition_l2892_289239


namespace percentage_problem_l2892_289280

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x * (x / 100) = 9) : x = 30 := by
  sorry

end percentage_problem_l2892_289280


namespace inverse_f_at_5_l2892_289293

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem inverse_f_at_5 :
  ∃ (f_inv : ℝ → ℝ), (∀ x ≥ 0, f_inv (f x) = x) ∧ f_inv 5 = 2 :=
by
  sorry

end inverse_f_at_5_l2892_289293


namespace final_comfortable_butterflies_l2892_289267

/-- Represents a point in the 2D lattice -/
structure LatticePoint where
  x : ℕ
  y : ℕ

/-- Represents the state of the lattice at any given time -/
def LatticeState := LatticePoint → Bool

/-- The neighborhood of a lattice point -/
def neighborhood (n : ℕ) (c : LatticePoint) : Set LatticePoint :=
  sorry

/-- Checks if a butterfly at a given point is lonely -/
def isLonely (n : ℕ) (state : LatticeState) (p : LatticePoint) : Bool :=
  sorry

/-- Simulates the process of lonely butterflies flying away -/
def simulateProcess (n : ℕ) (initialState : LatticeState) : LatticeState :=
  sorry

/-- Counts the number of comfortable butterflies in the final state -/
def countComfortableButterflies (n : ℕ) (finalState : LatticeState) : ℕ :=
  sorry

/-- The main theorem stating that the number of comfortable butterflies in the final state is n -/
theorem final_comfortable_butterflies (n : ℕ) (h : n > 0) :
  countComfortableButterflies n (simulateProcess n (λ _ => true)) = n :=
sorry

end final_comfortable_butterflies_l2892_289267


namespace compound_molar_mass_l2892_289208

/-- Given that 8 moles of a compound weigh 1600 grams, prove that its molar mass is 200 grams/mole -/
theorem compound_molar_mass (mass : ℝ) (moles : ℝ) (h1 : mass = 1600) (h2 : moles = 8) :
  mass / moles = 200 := by
  sorry

end compound_molar_mass_l2892_289208


namespace inverse_proportion_order_l2892_289282

-- Define the constants and variables
variable (a b c k : ℝ)
variable (y₁ y₂ y₃ : ℝ)

-- State the theorem
theorem inverse_proportion_order (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) 
  (h4 : k > 0)
  (h5 : y₁ = k / (a - b))
  (h6 : y₂ = k / (a - c))
  (h7 : y₃ = k / (c - a)) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end inverse_proportion_order_l2892_289282


namespace fern_fronds_l2892_289249

theorem fern_fronds (total_ferns : ℕ) (total_leaves : ℕ) (leaves_per_frond : ℕ) 
  (h1 : total_ferns = 6)
  (h2 : total_leaves = 1260)
  (h3 : leaves_per_frond = 30) :
  (total_leaves / leaves_per_frond) / total_ferns = 7 := by
sorry

end fern_fronds_l2892_289249


namespace max_value_x_plus_reciprocal_l2892_289234

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 10 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 12 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 12 :=
by sorry

end max_value_x_plus_reciprocal_l2892_289234


namespace pump_fill_time_pump_fill_time_proof_l2892_289287

theorem pump_fill_time (fill_time_with_leak : ℚ) (leak_drain_time : ℕ) : ℚ :=
  let fill_rate_with_leak : ℚ := 1 / fill_time_with_leak
  let leak_rate : ℚ := 1 / leak_drain_time
  let pump_rate : ℚ := fill_rate_with_leak + leak_rate
  1 / pump_rate

theorem pump_fill_time_proof :
  pump_fill_time (13/6) 26 = 2 := by sorry

end pump_fill_time_pump_fill_time_proof_l2892_289287


namespace largest_number_l2892_289259

/-- Converts a number from base b to decimal (base 10) --/
def to_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Theorem: 11 in base 3 is greater than 3 in base 10, 11 in base 2, and 3 in base 8 --/
theorem largest_number :
  (to_decimal 11 3 > to_decimal 3 10) ∧
  (to_decimal 11 3 > to_decimal 11 2) ∧
  (to_decimal 11 3 > to_decimal 3 8) :=
sorry

end largest_number_l2892_289259


namespace expansion_terms_count_l2892_289285

def factor1 : ℕ := 3
def factor2 : ℕ := 4
def factor3 : ℕ := 5

theorem expansion_terms_count : factor1 * factor2 * factor3 = 60 := by
  sorry

end expansion_terms_count_l2892_289285


namespace person_height_from_shadow_ratio_l2892_289231

/-- Proves that given a tree's height and shadow length, and a person's shadow length,
    we can determine the person's height assuming a constant ratio of height to shadow length. -/
theorem person_height_from_shadow_ratio (tree_height tree_shadow person_shadow : ℝ) 
  (h1 : tree_height = 50) 
  (h2 : tree_shadow = 25)
  (h3 : person_shadow = 20) :
  (tree_height / tree_shadow) * person_shadow = 40 := by
  sorry

end person_height_from_shadow_ratio_l2892_289231


namespace jose_bottle_caps_l2892_289223

/-- The number of bottle caps Jose ends up with after receiving more -/
def total_bottle_caps (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Jose ends up with 9 bottle caps -/
theorem jose_bottle_caps : total_bottle_caps 7 2 = 9 := by
  sorry

end jose_bottle_caps_l2892_289223


namespace omega_sum_equals_one_l2892_289294

theorem omega_sum_equals_one (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 + ω^48 + ω^51 + ω^54 + ω^57 + ω^60 + ω^63 = 1 := by
  sorry

end omega_sum_equals_one_l2892_289294


namespace sodaCans_theorem_l2892_289226

/-- The number of cans of soda that can be bought for a given amount of euros -/
def sodaCans (S Q E : ℚ) : ℚ :=
  10 * E * S / Q

/-- Theorem stating that the number of cans of soda that can be bought for E euros
    is equal to 10ES/Q, given that S cans can be purchased for Q dimes and
    1 euro is equivalent to 10 dimes -/
theorem sodaCans_theorem (S Q E : ℚ) (hS : S > 0) (hQ : Q > 0) (hE : E ≥ 0) :
  sodaCans S Q E = 10 * E * S / Q :=
by sorry

end sodaCans_theorem_l2892_289226


namespace total_amount_is_265_l2892_289269

/-- Represents the distribution of money among six individuals -/
structure MoneyDistribution where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  t : ℝ
  u : ℝ

/-- The theorem stating the total amount given the conditions -/
theorem total_amount_is_265 (dist : MoneyDistribution) : 
  (dist.p = 3 * (dist.s / 1.95)) →
  (dist.q = 2.70 * (dist.s / 1.95)) →
  (dist.r = 2.30 * (dist.s / 1.95)) →
  (dist.s = 39) →
  (dist.t = 1.80 * (dist.s / 1.95)) →
  (dist.u = 1.50 * (dist.s / 1.95)) →
  (dist.p + dist.q + dist.r + dist.s + dist.t + dist.u = 265) := by
  sorry


end total_amount_is_265_l2892_289269


namespace rectangle_perimeter_l2892_289299

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) (h4 : w = 7) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (rectangle_length + w) = 38 := by
sorry

end rectangle_perimeter_l2892_289299


namespace andrew_stamps_hundred_permits_l2892_289256

/-- The number of permits Andrew stamps in a day given his schedule and stamping rate -/
def permits_stamped (appointments : ℕ) (appointment_duration : ℕ) (workday_hours : ℕ) (stamping_rate : ℕ) : ℕ :=
  let total_appointment_hours := appointments * appointment_duration
  let stamping_hours := workday_hours - total_appointment_hours
  stamping_hours * stamping_rate

/-- Theorem stating that Andrew stamps 100 permits given his specific schedule and stamping rate -/
theorem andrew_stamps_hundred_permits :
  permits_stamped 2 3 8 50 = 100 := by
  sorry

end andrew_stamps_hundred_permits_l2892_289256


namespace rectangular_solid_on_sphere_l2892_289273

theorem rectangular_solid_on_sphere (x : ℝ) : 
  let surface_area : ℝ := 18 * Real.pi
  let radius : ℝ := Real.sqrt (surface_area / (4 * Real.pi))
  3^2 + 2^2 + x^2 = 4 * radius^2 → x = Real.sqrt 5 := by
  sorry

end rectangular_solid_on_sphere_l2892_289273


namespace min_y_is_e_l2892_289213

-- Define the function representing the given equation
def f (x y : ℝ) : Prop := Real.exp x = y * Real.log x + y * Real.log y

-- Theorem stating the minimum value of y
theorem min_y_is_e :
  ∃ (y_min : ℝ), y_min = Real.exp 1 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → f x y → y ≥ y_min :=
sorry

end min_y_is_e_l2892_289213


namespace arithmetic_sequence_formula_l2892_289286

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 1 + a 2 + a 3 = 12) 
  (h_prod : a 1 * a 2 * a 3 = 48) :
  ∀ n : ℕ, a n = 2 * n := by
sorry

end arithmetic_sequence_formula_l2892_289286


namespace sam_poured_buckets_l2892_289292

/-- The number of buckets Sam initially poured into the pool -/
def initial_buckets : ℝ := 1

/-- The number of buckets Sam added later -/
def additional_buckets : ℝ := 8.8

/-- The total number of buckets Sam poured into the pool -/
def total_buckets : ℝ := initial_buckets + additional_buckets

theorem sam_poured_buckets : total_buckets = 9.8 := by
  sorry

end sam_poured_buckets_l2892_289292


namespace complete_square_equivalence_l2892_289221

theorem complete_square_equivalence :
  let f₁ : ℝ → ℝ := λ x ↦ x^2 - 2*x + 3
  let f₂ : ℝ → ℝ := λ x ↦ 3*x^2 + 6*x - 1
  let f₃ : ℝ → ℝ := λ x ↦ -2*x^2 + 3*x - 2
  let g₁ : ℝ → ℝ := λ x ↦ (x - 1)^2 + 2
  let g₂ : ℝ → ℝ := λ x ↦ 3*(x + 1)^2 - 4
  let g₃ : ℝ → ℝ := λ x ↦ -2*(x - 3/4)^2 - 7/8
  (∀ x : ℝ, f₁ x = g₁ x) ∧
  (∀ x : ℝ, f₂ x = g₂ x) ∧
  (∀ x : ℝ, f₃ x = g₃ x) := by
  sorry

end complete_square_equivalence_l2892_289221


namespace mary_overtime_rate_increase_l2892_289275

/-- Represents Mary's work schedule and pay structure -/
structure WorkSchedule where
  max_hours : ℕ
  regular_hours : ℕ
  regular_rate : ℚ
  total_earnings : ℚ

/-- Calculates the percentage increase in overtime rate compared to regular rate -/
def overtime_rate_increase (w : WorkSchedule) : ℚ :=
  let regular_earnings := w.regular_hours * w.regular_rate
  let overtime_earnings := w.total_earnings - regular_earnings
  let overtime_hours := w.max_hours - w.regular_hours
  let overtime_rate := overtime_earnings / overtime_hours
  ((overtime_rate - w.regular_rate) / w.regular_rate) * 100

/-- Mary's work schedule -/
def mary_schedule : WorkSchedule :=
  { max_hours := 40
  , regular_hours := 20
  , regular_rate := 8
  , total_earnings := 360 }

/-- Theorem stating that Mary's overtime rate increase is 25% -/
theorem mary_overtime_rate_increase :
  overtime_rate_increase mary_schedule = 25 := by
  sorry

end mary_overtime_rate_increase_l2892_289275


namespace star_op_power_equality_l2892_289222

def star_op (a b : ℕ+) : ℕ+ := a ^ (b.val ^ 2)

theorem star_op_power_equality (a b n : ℕ+) :
  (star_op a b) ^ n.val = star_op a (n * b) ↔ n = 1 := by
  sorry

end star_op_power_equality_l2892_289222


namespace tan_alpha_value_l2892_289281

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 := by
sorry

end tan_alpha_value_l2892_289281


namespace smallest_n_for_g_equals_seven_g_of_eight_equals_seven_l2892_289219

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1 else n/2 + 3

theorem smallest_n_for_g_equals_seven :
  ∀ n : ℕ, n > 0 → g n = 7 → n ≥ 8 :=
by sorry

theorem g_of_eight_equals_seven : g 8 = 7 :=
by sorry

end smallest_n_for_g_equals_seven_g_of_eight_equals_seven_l2892_289219


namespace f_increasing_on_interval_l2892_289289

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

def domain (x : ℝ) : Prop := x^2 - 2*x - 3 > 0

theorem f_increasing_on_interval :
  ∀ x y, x < y → x < -1 → y < -1 → domain x → domain y → f x < f y :=
by sorry

end f_increasing_on_interval_l2892_289289


namespace tallest_player_height_l2892_289227

theorem tallest_player_height (shortest_height : ℝ) (height_difference : ℝ) 
  (h1 : shortest_height = 68.25)
  (h2 : height_difference = 9.5) :
  shortest_height + height_difference = 77.75 := by
  sorry

end tallest_player_height_l2892_289227


namespace final_sum_after_transformation_l2892_289238

theorem final_sum_after_transformation (x y T : ℝ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 := by
  sorry

end final_sum_after_transformation_l2892_289238


namespace perfect_squares_difference_l2892_289214

theorem perfect_squares_difference (n : ℕ) : 
  (∃ a : ℕ, n - 52 = a^2) ∧ (∃ b : ℕ, n + 37 = b^2) → n = 1988 := by
  sorry

end perfect_squares_difference_l2892_289214


namespace largest_n_for_equation_l2892_289202

theorem largest_n_for_equation : ∃ (x y z : ℕ+), 
  (100 : ℤ) = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 12 ∧ 
  ∀ (n : ℕ+), n > 10 → ¬∃ (a b c : ℕ+), 
    (n^2 : ℤ) = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 5*a + 5*b + 5*c - 12 :=
by sorry

end largest_n_for_equation_l2892_289202


namespace fourth_number_in_sequence_l2892_289291

theorem fourth_number_in_sequence (seq : Fin 6 → ℕ) 
  (h1 : seq 0 = 29)
  (h2 : seq 1 = 35)
  (h3 : seq 2 = 41)
  (h5 : seq 4 = 53)
  (h6 : seq 5 = 59)
  (h_arithmetic : ∀ i : Fin 4, seq (i + 1) - seq i = seq 1 - seq 0) :
  seq 3 = 47 := by
  sorry

end fourth_number_in_sequence_l2892_289291


namespace same_speed_problem_l2892_289212

theorem same_speed_problem (x : ℝ) : 
  let jack_speed := x^2 - 11*x - 22
  let jill_distance := x^2 - 5*x - 60
  let jill_time := x + 6
  let jill_speed := jill_distance / jill_time
  jack_speed = jill_speed → jack_speed = 4 :=
by sorry

end same_speed_problem_l2892_289212


namespace polygon_sides_from_exterior_angle_l2892_289260

theorem polygon_sides_from_exterior_angle (exterior_angle : ℝ) (n : ℕ) :
  exterior_angle = 36 →
  (360 : ℝ) / exterior_angle = n →
  n = 10 := by
  sorry

end polygon_sides_from_exterior_angle_l2892_289260


namespace solve_system_l2892_289237

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -25 / 11 := by
sorry

end solve_system_l2892_289237


namespace sequence_with_positive_triples_negative_sum_l2892_289200

theorem sequence_with_positive_triples_negative_sum : 
  ∃ (seq : Fin 20 → ℝ), 
    (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧ 
    (Finset.sum Finset.univ seq < 0) := by
  sorry

end sequence_with_positive_triples_negative_sum_l2892_289200


namespace total_dozens_shipped_l2892_289240

-- Define the number of boxes shipped last week
def boxes_last_week : ℕ := 10

-- Define the total number of pomelos shipped last week
def total_pomelos_last_week : ℕ := 240

-- Define the number of boxes shipped this week
def boxes_this_week : ℕ := 20

-- Theorem to prove
theorem total_dozens_shipped : ℕ := by
  -- The proof goes here
  sorry

-- Goal: prove that total_dozens_shipped = 60
example : total_dozens_shipped = 60 := by sorry

end total_dozens_shipped_l2892_289240


namespace grocer_bananas_purchase_l2892_289263

/-- The number of pounds of bananas purchased by a grocer -/
def bananas_purchased (buy_price : ℚ) (sell_price : ℚ) (total_profit : ℚ) : ℚ :=
  total_profit / (sell_price / 4 - buy_price / 3)

/-- Theorem stating that the grocer purchased 72 pounds of bananas -/
theorem grocer_bananas_purchase :
  bananas_purchased (1/2) (1/1) 6 = 72 := by
  sorry

end grocer_bananas_purchase_l2892_289263


namespace not_quadratic_radical_l2892_289277

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop := x ≥ 0

-- State the theorem
theorem not_quadratic_radical : ¬ is_quadratic_radical (-4) := by
  sorry

end not_quadratic_radical_l2892_289277


namespace donny_remaining_money_l2892_289283

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem donny_remaining_money :
  initial_amount - (kite_cost + frisbee_cost) = 61 := by
  sorry

end donny_remaining_money_l2892_289283


namespace plate_distance_to_bottom_l2892_289220

/-- Given a square table with a round plate, if the distances from the plate to the top, left, and right edges
    of the table are 10, 63, and 20 units respectively, then the distance from the plate to the bottom edge
    of the table is 73 units. -/
theorem plate_distance_to_bottom (d : ℝ) :
  let top_distance : ℝ := 10
  let left_distance : ℝ := 63
  let right_distance : ℝ := 20
  let bottom_distance : ℝ := left_distance + right_distance - top_distance
  bottom_distance = 73 := by
  sorry


end plate_distance_to_bottom_l2892_289220


namespace profit_percentage_is_36_percent_l2892_289233

def selling_price : ℝ := 850
def profit : ℝ := 225

theorem profit_percentage_is_36_percent :
  (profit / (selling_price - profit)) * 100 = 36 := by
  sorry

end profit_percentage_is_36_percent_l2892_289233


namespace zoo_animal_difference_l2892_289232

def zoo_problem (parrots snakes monkeys elephants zebras : ℕ) : Prop :=
  (parrots = 8) ∧
  (snakes = 3 * parrots) ∧
  (monkeys = 2 * snakes) ∧
  (elephants = (parrots + snakes) / 2) ∧
  (zebras = elephants - 3)

theorem zoo_animal_difference :
  ∀ parrots snakes monkeys elephants zebras : ℕ,
  zoo_problem parrots snakes monkeys elephants zebras →
  monkeys - zebras = 35 :=
by
  sorry

end zoo_animal_difference_l2892_289232


namespace cone_volume_l2892_289265

/-- Given a cone with slant height 5 and lateral surface area 20π, prove its volume is 16π -/
theorem cone_volume (s : ℝ) (l : ℝ) (v : ℝ) : 
  s = 5 → l = 20 * Real.pi → v = (16 : ℝ) * Real.pi → 
  (s^2 * Real.pi / l = s / 4) ∧ 
  (v = (1/3) * (l/s)^2 * (s^2 - (l/(Real.pi * s))^2)) := by
  sorry

#check cone_volume

end cone_volume_l2892_289265


namespace arithmetic_sequence_sin_sum_l2892_289246

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 7 + a 13 = 4 * Real.pi) :
  Real.sin (a 2 + a 12) = Real.sqrt 3 / 2 := by
  sorry

end arithmetic_sequence_sin_sum_l2892_289246


namespace rationalize_denominator_35_sqrt_35_l2892_289296

theorem rationalize_denominator_35_sqrt_35 :
  (35 : ℝ) / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end rationalize_denominator_35_sqrt_35_l2892_289296


namespace original_jellybeans_proof_l2892_289272

/-- The original number of jellybeans in Jenny's jar -/
def original_jellybeans : ℕ := 50

/-- The fraction of jellybeans remaining after each day -/
def daily_remaining_fraction : ℚ := 4/5

/-- The number of days that have passed -/
def days_passed : ℕ := 2

/-- The number of jellybeans remaining after two days -/
def remaining_jellybeans : ℕ := 32

/-- Theorem stating that the original number of jellybeans is correct -/
theorem original_jellybeans_proof :
  (daily_remaining_fraction ^ days_passed) * original_jellybeans = remaining_jellybeans := by
  sorry

end original_jellybeans_proof_l2892_289272


namespace intersection_of_A_and_B_l2892_289245

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1) 1 := by sorry

end intersection_of_A_and_B_l2892_289245


namespace equal_parts_complex_l2892_289253

/-- A complex number is an "equal parts complex number" if its real and imaginary parts are equal -/
def is_equal_parts (z : ℂ) : Prop := z.re = z.im

/-- Given that Z = (1+ai)i is an "equal parts complex number", prove that a = -1 -/
theorem equal_parts_complex (a : ℝ) :
  is_equal_parts ((1 + a * Complex.I) * Complex.I) → a = -1 := by
  sorry

end equal_parts_complex_l2892_289253


namespace negative_cube_squared_l2892_289261

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end negative_cube_squared_l2892_289261


namespace line_parameterization_l2892_289244

/-- Given a line y = -3x + 2 parameterized as [x; y] = [5; r] + t[k; 8], prove r = -13 and k = -4 -/
theorem line_parameterization (r k : ℝ) : 
  (∀ x y t : ℝ, y = -3 * x + 2 ↔ ∃ t, (x, y) = (5 + t * k, r + t * 8)) →
  r = -13 ∧ k = -4 := by
  sorry

end line_parameterization_l2892_289244


namespace allan_bought_three_balloons_l2892_289266

/-- The number of balloons Allan bought at the park -/
def balloons_bought_at_park (initial_balloons final_balloons : ℕ) : ℕ :=
  final_balloons - initial_balloons

/-- Theorem stating that Allan bought 3 balloons at the park -/
theorem allan_bought_three_balloons :
  balloons_bought_at_park 5 8 = 3 := by
  sorry

end allan_bought_three_balloons_l2892_289266


namespace base_length_is_double_half_length_l2892_289279

/-- An isosceles triangle with a line bisector from the vertex angle -/
structure IsoscelesTriangleWithBisector :=
  (base_half_length : ℝ)

/-- The theorem stating that the total base length is twice the length of each half -/
theorem base_length_is_double_half_length (triangle : IsoscelesTriangleWithBisector) 
  (h : triangle.base_half_length = 4) : 
  2 * triangle.base_half_length = 8 := by
  sorry

#check base_length_is_double_half_length

end base_length_is_double_half_length_l2892_289279


namespace product_of_fractions_equals_self_l2892_289247

theorem product_of_fractions_equals_self (n : ℝ) (h : n > 0) : 
  n = (4/5 * n) * (5/6 * n) → n = 3/2 := by
  sorry

end product_of_fractions_equals_self_l2892_289247


namespace max_cables_for_given_network_l2892_289229

/-- Represents a computer network with two brands of computers. -/
structure ComputerNetwork where
  brand_a : ℕ
  brand_b : ℕ

/-- Represents the number of cables in the network. -/
def cables (n : ComputerNetwork) : ℕ := sorry

/-- Checks if all computers in the network can communicate. -/
def all_communicate (n : ComputerNetwork) : Prop := sorry

/-- The maximum number of cables needed for full communication. -/
def max_cables (n : ComputerNetwork) : ℕ := sorry

/-- Theorem stating the maximum number of cables for the given network. -/
theorem max_cables_for_given_network :
  ∀ (n : ComputerNetwork),
    n.brand_a = 20 ∧ n.brand_b = 20 →
    max_cables n = 20 ∧ all_communicate n := by sorry

end max_cables_for_given_network_l2892_289229


namespace characterize_equal_prime_factors_l2892_289216

/-- The set of prime factors of a positive integer n -/
def primeDivisors (n : ℕ) : Set ℕ := sorry

theorem characterize_equal_prime_factors :
  ∀ (a m n : ℕ),
    a > 1 →
    m < n →
    (primeDivisors (a^m - 1) = primeDivisors (a^n - 1)) ↔
    (∃ l : ℕ, l ≥ 2 ∧ a = 2^l - 1 ∧ m = 1 ∧ n = 2) :=
by sorry

end characterize_equal_prime_factors_l2892_289216


namespace sin_36_degrees_l2892_289258

theorem sin_36_degrees : 
  Real.sin (36 * π / 180) = (1 / 4) * Real.sqrt (10 - 2 * Real.sqrt 5) := by sorry

end sin_36_degrees_l2892_289258


namespace vector_relations_l2892_289257

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Define vector a -/
def a : Vector2D := ⟨1, 1⟩

/-- Define vector b with parameter m -/
def b (m : ℝ) : Vector2D := ⟨2, m⟩

/-- Two vectors are parallel if their components are proportional -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : Vector2D) : Prop :=
  v.x * w.x + v.y * w.y = 0

/-- Main theorem -/
theorem vector_relations :
  (∀ m : ℝ, parallel a (b m) → m = 2) ∧
  (∀ m : ℝ, perpendicular a (b m) → m = -2) := by
  sorry

end vector_relations_l2892_289257


namespace square_sum_equals_z_squared_l2892_289248

theorem square_sum_equals_z_squared (x y z b a : ℝ) 
  (h1 : x * y + x^2 = b)
  (h2 : 1 / x^2 - 1 / y^2 = a)
  (h3 : z = x + y) :
  (x + y)^2 = z^2 := by
  sorry

end square_sum_equals_z_squared_l2892_289248


namespace exam_class_size_l2892_289217

/-- Represents a class of students with their exam marks. -/
structure ExamClass where
  totalStudents : ℕ
  averageMark : ℚ
  excludedStudents : ℕ
  excludedAverage : ℚ
  remainingAverage : ℚ

/-- Theorem stating the number of students in the class given the conditions. -/
theorem exam_class_size (c : ExamClass)
  (h1 : c.averageMark = 80)
  (h2 : c.excludedStudents = 5)
  (h3 : c.excludedAverage = 50)
  (h4 : c.remainingAverage = 90)
  (h5 : c.totalStudents * c.averageMark = 
        (c.totalStudents - c.excludedStudents) * c.remainingAverage + 
        c.excludedStudents * c.excludedAverage) :
  c.totalStudents = 20 := by
  sorry


end exam_class_size_l2892_289217


namespace cats_puppies_weight_difference_l2892_289236

/-- The number of puppies Hartley has -/
def num_puppies : ℕ := 4

/-- The weight of each puppy in kilograms -/
def puppy_weight : ℚ := 7.5

/-- The number of cats at the rescue center -/
def num_cats : ℕ := 14

/-- The weight of each cat in kilograms -/
def cat_weight : ℚ := 2.5

/-- The total weight of the puppies in kilograms -/
def total_puppy_weight : ℚ := num_puppies * puppy_weight

/-- The total weight of the cats in kilograms -/
def total_cat_weight : ℚ := num_cats * cat_weight

theorem cats_puppies_weight_difference :
  total_cat_weight - total_puppy_weight = 5 := by
  sorry

end cats_puppies_weight_difference_l2892_289236


namespace inequality_implication_l2892_289284

theorem inequality_implication (m n : ℝ) (h : m > n) : -3*m < -3*n := by
  sorry

end inequality_implication_l2892_289284


namespace fraction_meaningful_l2892_289225

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end fraction_meaningful_l2892_289225


namespace product_inequality_l2892_289205

theorem product_inequality (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) : 
  8 * (a₁ * a₂ * a₃ * a₄ + 1) ≥ (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) := by
  sorry

end product_inequality_l2892_289205


namespace milk_for_12_cookies_l2892_289206

/-- The number of cookies that can be baked with 5 liters of milk -/
def cookies_per_5_liters : ℕ := 30

/-- The number of cups in a liter -/
def cups_per_liter : ℕ := 4

/-- The number of cookies we want to bake -/
def target_cookies : ℕ := 12

/-- The function that calculates the number of cups of milk needed for a given number of cookies -/
def milk_needed (cookies : ℕ) : ℚ :=
  (cookies * cups_per_liter * 5 : ℚ) / cookies_per_5_liters

theorem milk_for_12_cookies :
  milk_needed target_cookies = 8 := by sorry

end milk_for_12_cookies_l2892_289206


namespace organization_size_after_five_years_l2892_289268

def organization_growth (initial_members : ℕ) (initial_leaders : ℕ) (years : ℕ) : ℕ :=
  let rec growth (year : ℕ) (members : ℕ) : ℕ :=
    if year = 0 then
      members
    else
      growth (year - 1) (4 * members - 18)
  growth years initial_members

theorem organization_size_after_five_years :
  organization_growth 12 6 5 = 6150 := by
  sorry

end organization_size_after_five_years_l2892_289268


namespace matrix_identity_l2892_289262

variable {n : Type*} [DecidableEq n] [Fintype n]

theorem matrix_identity (A : Matrix n n ℝ) (h_inv : IsUnit A) 
  (h_eq : (A - 3 • (1 : Matrix n n ℝ)) * (A - 5 • (1 : Matrix n n ℝ)) = 0) :
  A + 8 • A⁻¹ = (7 • A + 64 • (1 : Matrix n n ℝ)) / 15 := by
  sorry

end matrix_identity_l2892_289262


namespace bobby_candy_problem_l2892_289230

theorem bobby_candy_problem (initial_candy : ℕ) (chocolate : ℕ) (candy_chocolate_diff : ℕ) :
  initial_candy = 38 →
  chocolate = 16 →
  candy_chocolate_diff = 58 →
  (initial_candy + chocolate + candy_chocolate_diff) - initial_candy = 36 :=
by
  sorry

end bobby_candy_problem_l2892_289230


namespace negation_of_existence_square_leq_one_negation_l2892_289251

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, x < 1 ∧ p x) ↔ (∀ x, x < 1 → ¬ p x) :=
by sorry

theorem square_leq_one_negation :
  (¬ ∃ x : ℝ, x < 1 ∧ x^2 ≤ 1) ↔ (∀ x : ℝ, x < 1 → x^2 > 1) :=
by sorry

end negation_of_existence_square_leq_one_negation_l2892_289251


namespace stratified_sampling_theorem_l2892_289241

theorem stratified_sampling_theorem (total_sample : ℕ) 
  (school_A : ℕ) (school_B : ℕ) (school_C : ℕ) : 
  total_sample = 60 → 
  school_A = 180 → 
  school_B = 270 → 
  school_C = 90 → 
  (school_C * total_sample) / (school_A + school_B + school_C) = 10 := by
  sorry

end stratified_sampling_theorem_l2892_289241


namespace even_function_shift_l2892_289278

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is monotonically increasing on an interval (a,b) if
    for all x, y in (a,b), x < y implies f(x) < f(y) -/
def MonoIncOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

/-- A function has a symmetry axis at x = k if f(k + x) = f(k - x) for all x -/
def HasSymmetryAxis (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x, f (k + x) = f (k - x)

theorem even_function_shift (f : ℝ → ℝ) :
    IsEven f →
    MonoIncOn f 3 5 →
    HasSymmetryAxis (fun x ↦ f (x - 1)) 1 ∧
    MonoIncOn (fun x ↦ f (x - 1)) 4 6 := by
  sorry

end even_function_shift_l2892_289278


namespace total_candy_cases_is_80_l2892_289288

/-- The Sweet Shop gets a new candy shipment every 35 days. -/
def shipment_interval : ℕ := 35

/-- The number of cases of chocolate bars. -/
def chocolate_cases : ℕ := 25

/-- The number of cases of lollipops. -/
def lollipop_cases : ℕ := 55

/-- The total number of candy cases. -/
def total_candy_cases : ℕ := chocolate_cases + lollipop_cases

/-- Theorem stating that the total number of candy cases is 80. -/
theorem total_candy_cases_is_80 : total_candy_cases = 80 := by
  sorry

end total_candy_cases_is_80_l2892_289288


namespace trig_expression_equals_three_halves_l2892_289243

theorem trig_expression_equals_three_halves (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := by
  sorry

end trig_expression_equals_three_halves_l2892_289243


namespace smallest_shift_for_scaled_function_l2892_289235

-- Define a periodic function with period 30
def isPeriodic30 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 30) = g x

-- Define the property we're looking for
def hasProperty (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 3) = g (x / 3)

theorem smallest_shift_for_scaled_function (g : ℝ → ℝ) (h : isPeriodic30 g) :
  ∃ b : ℝ, b > 0 ∧ hasProperty g b ∧ ∀ b' : ℝ, b' > 0 → hasProperty g b' → b ≤ b' :=
sorry

end smallest_shift_for_scaled_function_l2892_289235


namespace optimal_sampling_methods_for_surveys_l2892_289264

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a survey with its characteristics -/
structure Survey where
  totalPopulation : ℕ
  sampleSize : ℕ
  hasDistinctGroups : Bool
  hasSmallDifferences : Bool

/-- Determines the optimal sampling method for a given survey -/
def optimalSamplingMethod (s : Survey) : SamplingMethod :=
  if s.hasDistinctGroups then SamplingMethod.Stratified
  else if s.hasSmallDifferences && s.sampleSize < 10 then SamplingMethod.SimpleRandom
  else SamplingMethod.Systematic

/-- The main theorem stating the optimal sampling methods for the two surveys -/
theorem optimal_sampling_methods_for_surveys :
  let survey1 : Survey := {
    totalPopulation := 500,
    sampleSize := 100,
    hasDistinctGroups := true,
    hasSmallDifferences := false
  }
  let survey2 : Survey := {
    totalPopulation := 15,
    sampleSize := 3,
    hasDistinctGroups := false,
    hasSmallDifferences := true
  }
  (optimalSamplingMethod survey1 = SamplingMethod.Stratified) ∧
  (optimalSamplingMethod survey2 = SamplingMethod.SimpleRandom) :=
by
  sorry

end optimal_sampling_methods_for_surveys_l2892_289264


namespace isosceles_triangle_area_l2892_289224

/-- An isosceles triangle with given height and median -/
structure IsoscelesTriangle where
  -- Height from the base to the vertex
  height : ℝ
  -- Median from a leg to the midpoint of the base
  median : ℝ

/-- The area of an isosceles triangle given its height and median -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with height 18 and median 15 is 144 -/
theorem isosceles_triangle_area :
  let t : IsoscelesTriangle := { height := 18, median := 15 }
  area t = 144 := by
  sorry

end isosceles_triangle_area_l2892_289224


namespace smallest_prime_twelve_less_prime_square_l2892_289203

theorem smallest_prime_twelve_less_prime_square : ∃ (p n : ℕ), 
  p = 13 ∧ 
  Nat.Prime p ∧ 
  Nat.Prime n ∧ 
  p = n^2 - 12 ∧
  ∀ (q m : ℕ), Nat.Prime q ∧ Nat.Prime m ∧ q = m^2 - 12 → p ≤ q :=
by sorry

end smallest_prime_twelve_less_prime_square_l2892_289203


namespace parallelogram_fourth_vertex_l2892_289218

-- Define the points
def A : ℝ × ℝ := (-6, -1)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (-3, -2)

-- Define the parallelogram property
def is_parallelogram (A B C M : ℝ × ℝ) : Prop :=
  (B.1 - A.1, B.2 - A.2) = (M.1 - C.1, M.2 - C.2)

-- Theorem statement
theorem parallelogram_fourth_vertex :
  ∃ M : ℝ × ℝ, is_parallelogram A B C M ∧ M = (4, 1) := by
  sorry

end parallelogram_fourth_vertex_l2892_289218


namespace town_population_distribution_l2892_289297

/-- Represents a category in the pie chart --/
structure Category where
  name : String
  percentage : ℝ

/-- Represents a pie chart with three categories --/
structure PieChart where
  categories : Fin 3 → Category
  sum_to_100 : (categories 0).percentage + (categories 1).percentage + (categories 2).percentage = 100

/-- The main theorem --/
theorem town_population_distribution (chart : PieChart) 
  (h1 : (chart.categories 0).name = "less than 5,000 residents")
  (h2 : (chart.categories 1).name = "5,000 to 20,000 residents")
  (h3 : (chart.categories 2).name = "20,000 or more residents")
  (h4 : (chart.categories 1).percentage = 40) :
  (chart.categories 1).percentage = 40 := by
  sorry

end town_population_distribution_l2892_289297


namespace dog_grouping_theorem_l2892_289270

/-- The number of ways to divide 12 dogs into specified groups -/
def dog_grouping_ways : ℕ :=
  let total_dogs : ℕ := 12
  let group1_size : ℕ := 4  -- Fluffy's group
  let group2_size : ℕ := 5  -- Nipper's group
  let group3_size : ℕ := 3
  let remaining_dogs : ℕ := total_dogs - 2  -- Excluding Fluffy and Nipper
  let ways_to_fill_group1 : ℕ := Nat.choose remaining_dogs (group1_size - 1)
  let ways_to_fill_group2 : ℕ := Nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1)
  ways_to_fill_group1 * ways_to_fill_group2

/-- Theorem stating the number of ways to divide the dogs into groups -/
theorem dog_grouping_theorem : dog_grouping_ways = 4200 := by
  sorry

end dog_grouping_theorem_l2892_289270


namespace shekars_english_score_l2892_289210

/-- Given Shekar's scores in four subjects and his average score, prove his English score --/
theorem shekars_english_score
  (math_score science_score social_score biology_score : ℕ)
  (average_score : ℚ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : social_score = 82)
  (h4 : biology_score = 85)
  (h5 : average_score = 71)
  (h6 : (math_score + science_score + social_score + biology_score + english_score : ℚ) / 5 = average_score) :
  english_score = 47 :=
by sorry

end shekars_english_score_l2892_289210


namespace cloth_sales_worth_l2892_289250

-- Define the commission rate as a percentage
def commission_rate : ℚ := 2.5

-- Define the commission earned on a particular day
def commission_earned : ℚ := 15

-- Define the function to calculate the total sales
def total_sales (rate : ℚ) (commission : ℚ) : ℚ :=
  commission / (rate / 100)

-- Theorem statement
theorem cloth_sales_worth :
  total_sales commission_rate commission_earned = 600 := by
  sorry

end cloth_sales_worth_l2892_289250


namespace fill_time_calculation_l2892_289252

/-- Represents the time to fill a leaky tank -/
def fill_time_with_leak : ℝ := 8

/-- Represents the time for the tank to empty due to the leak -/
def empty_time : ℝ := 56

/-- Represents the time to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 7

/-- Theorem stating that given the fill time with leak and empty time,
    the fill time without leak is 7 hours -/
theorem fill_time_calculation :
  (fill_time_with_leak * empty_time) / (empty_time - fill_time_with_leak) = fill_time_without_leak :=
sorry

end fill_time_calculation_l2892_289252


namespace reachability_l2892_289255

/-- Number of positive integer divisors of n -/
def τ (n : ℕ) : ℕ := sorry

/-- Sum of positive integer divisors of n -/
def σ (n : ℕ) : ℕ := sorry

/-- Number of positive integers less than or equal to n that are relatively prime to n -/
def φ (n : ℕ) : ℕ := sorry

/-- Represents the operation of applying τ, σ, or φ -/
inductive Operation
| tau : Operation
| sigma : Operation
| phi : Operation

/-- Applies an operation to a natural number -/
def applyOperation (op : Operation) (n : ℕ) : ℕ :=
  match op with
  | Operation.tau => τ n
  | Operation.sigma => σ n
  | Operation.phi => φ n

/-- Theorem: For any two integers a and b greater than 1, 
    there exists a finite sequence of operations that transforms a into b -/
theorem reachability (a b : ℕ) (ha : a > 1) (hb : b > 1) : 
  ∃ (ops : List Operation), 
    (ops.foldl (fun n op => applyOperation op n) a) = b :=
sorry

end reachability_l2892_289255


namespace solve_for_k_l2892_289271

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 6

def g (k x : ℝ) : ℝ := x^2 - k * x - 8

theorem solve_for_k : ∃ k : ℝ, f 5 - g k 5 = 20 ∧ k = -10.8 := by sorry

end solve_for_k_l2892_289271


namespace winning_pair_probability_l2892_289209

-- Define the deck
def deck_size : ℕ := 9
def num_colors : ℕ := 3
def num_letters : ℕ := 3

-- Define a winning pair
def is_winning_pair (card1 card2 : ℕ × ℕ) : Prop :=
  (card1.1 = card2.1) ∨ (card1.2 = card2.2)

-- Define the probability of drawing a winning pair
def prob_winning_pair : ℚ :=
  (num_colors * (num_letters.choose 2) + num_letters * (num_colors.choose 2)) / deck_size.choose 2

-- Theorem statement
theorem winning_pair_probability : prob_winning_pair = 1/2 := by
  sorry

end winning_pair_probability_l2892_289209


namespace three_digit_squares_ending_in_self_l2892_289276

theorem three_digit_squares_ending_in_self (A : ℕ) : 
  (100 ≤ A ∧ A < 1000) ∧ (A^2 ≡ A [ZMOD 1000]) ↔ (A = 376 ∨ A = 625) := by
  sorry

end three_digit_squares_ending_in_self_l2892_289276


namespace average_monthly_production_l2892_289215

/-- Calculates the average monthly salt production for a year given the initial production and monthly increase. -/
theorem average_monthly_production
  (initial_production : ℕ)
  (monthly_increase : ℕ)
  (months : ℕ)
  (h1 : initial_production = 1000)
  (h2 : monthly_increase = 100)
  (h3 : months = 12) :
  (initial_production + (initial_production + monthly_increase * (months - 1)) * (months - 1) / 2) / months = 9800 / 12 := by
  sorry

end average_monthly_production_l2892_289215


namespace candy_bars_purchased_l2892_289274

theorem candy_bars_purchased (total_cost : ℕ) (price_per_bar : ℕ) (h1 : total_cost = 6) (h2 : price_per_bar = 3) :
  total_cost / price_per_bar = 2 := by
  sorry

end candy_bars_purchased_l2892_289274


namespace area_of_region_R_l2892_289228

/-- A square with side length 3 -/
structure Square :=
  (side_length : ℝ)
  (is_three : side_length = 3)

/-- The region R in the square -/
def region_R (s : Square) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1^2 + p.2^2 ≤ (3*Real.sqrt 2/2)^2}

/-- The area of a region -/
noncomputable def area (r : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem to be proved -/
theorem area_of_region_R (s : Square) : area (region_R s) = 9 * Real.pi / 8 := by
  sorry

end area_of_region_R_l2892_289228


namespace a_4_equals_7_l2892_289295

-- Define the sequence sum function
def S (n : ℕ) : ℤ := n^2 - 1

-- Define the sequence term function
def a (n : ℕ) : ℤ := S n - S (n-1)

-- Theorem statement
theorem a_4_equals_7 : a 4 = 7 := by
  sorry

end a_4_equals_7_l2892_289295


namespace factor_t_squared_minus_64_l2892_289290

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_t_squared_minus_64_l2892_289290


namespace soil_bags_needed_l2892_289207

/-- Calculates the number of soil bags needed for raised beds -/
theorem soil_bags_needed
  (num_beds : ℕ)
  (length width height : ℝ)
  (soil_per_bag : ℝ)
  (h_num_beds : num_beds = 2)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_soil_per_bag : soil_per_bag = 4) :
  ⌈(num_beds * length * width * height) / soil_per_bag⌉ = 16 := by
  sorry

end soil_bags_needed_l2892_289207


namespace remainder_twelve_pow_2012_mod_5_l2892_289201

theorem remainder_twelve_pow_2012_mod_5 : 12^2012 % 5 = 1 := by
  sorry

end remainder_twelve_pow_2012_mod_5_l2892_289201


namespace part_one_part_two_l2892_289211

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides opposite to angles A, B, C respectively

-- Define the conditions
variable (h1 : 0 < A) -- A is acute
variable (h2 : A < π / 2) -- A is acute
variable (h3 : 3 * b = 5 * a * Real.sin B) -- Given condition

-- Part 1
theorem part_one : 
  Real.sin (2 * A) + Real.cos ((B + C) / 2) ^ 2 = 53 / 50 := by sorry

-- Part 2
theorem part_two (h4 : a = Real.sqrt 2) (h5 : 1 / 2 * b * c * Real.sin A = 3 / 2) :
  b = Real.sqrt 5 ∧ c = Real.sqrt 5 := by sorry

end

end part_one_part_two_l2892_289211


namespace probability_different_colors_is_three_fifths_l2892_289254

def num_red_balls : ℕ := 3
def num_white_balls : ℕ := 2
def total_balls : ℕ := num_red_balls + num_white_balls

def probability_different_colors : ℚ :=
  (num_red_balls * num_white_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2 : ℚ)

theorem probability_different_colors_is_three_fifths :
  probability_different_colors = 3 / 5 := by
  sorry

end probability_different_colors_is_three_fifths_l2892_289254


namespace sum_of_pairs_l2892_289242

theorem sum_of_pairs : (5 + 2) + (8 + 6) + (4 + 7) + (3 + 2) = 37 := by
  sorry

end sum_of_pairs_l2892_289242


namespace vector_ratio_theorem_l2892_289204

/-- Given points O, A, B, C in a Cartesian coordinate system where O is the origin,
    prove that the ratio of the magnitudes of BC to AC is 3,
    given that OC is a weighted sum of OA and OB. -/
theorem vector_ratio_theorem (O A B C : ℝ × ℝ) :
  O = (0, 0) →
  C - O = 3/4 • (A - O) + 1/4 • (B - O) →
  ‖C - B‖ / ‖C - A‖ = 3 := by
  sorry

end vector_ratio_theorem_l2892_289204


namespace sqrt_five_power_calculation_l2892_289298

theorem sqrt_five_power_calculation : (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 78125 * Real.sqrt 5 := by
  sorry

end sqrt_five_power_calculation_l2892_289298
