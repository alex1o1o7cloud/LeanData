import Mathlib

namespace NUMINAMATH_CALUDE_candy_probability_theorem_l2682_268254

/-- Represents a packet of candies -/
structure CandyPacket where
  blue : ℕ
  total : ℕ
  h_total_pos : total > 0
  h_blue_le_total : blue ≤ total

/-- Represents a box containing two packets of candies -/
structure CandyBox where
  packet1 : CandyPacket
  packet2 : CandyPacket

/-- The probability of drawing a blue candy from the box -/
def blue_probability (box : CandyBox) : ℚ :=
  (box.packet1.blue + box.packet2.blue : ℚ) / (box.packet1.total + box.packet2.total)

theorem candy_probability_theorem :
  (∃ box : CandyBox, blue_probability box = 5/13) ∧
  (∃ box : CandyBox, blue_probability box = 7/18) ∧
  (∀ box : CandyBox, blue_probability box ≠ 17/40) :=
sorry

end NUMINAMATH_CALUDE_candy_probability_theorem_l2682_268254


namespace NUMINAMATH_CALUDE_jake_earnings_l2682_268203

def calculate_earnings (viper_count cobra_count python_count : ℕ)
                       (viper_eggs cobra_eggs python_eggs : ℕ)
                       (viper_price cobra_price python_price : ℚ)
                       (viper_discount cobra_discount : ℚ) : ℚ :=
  let viper_babies := viper_count * viper_eggs
  let cobra_babies := cobra_count * cobra_eggs
  let python_babies := python_count * python_eggs
  let viper_earnings := viper_babies * (viper_price * (1 - viper_discount))
  let cobra_earnings := cobra_babies * (cobra_price * (1 - cobra_discount))
  let python_earnings := python_babies * python_price
  viper_earnings + cobra_earnings + python_earnings

theorem jake_earnings :
  calculate_earnings 2 3 1 3 2 4 300 250 450 (1/10) (1/20) = 4845 := by
  sorry

end NUMINAMATH_CALUDE_jake_earnings_l2682_268203


namespace NUMINAMATH_CALUDE_committee_selection_count_l2682_268234

theorem committee_selection_count : Nat.choose 12 5 = 792 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_count_l2682_268234


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2682_268219

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 6) ((Nat.factorial 12) / (Nat.factorial 4)) = 240 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2682_268219


namespace NUMINAMATH_CALUDE_tangent_line_slope_relation_l2682_268230

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem tangent_line_slope_relation (a b : ℝ) :
  a^2 + b = 0 →
  ∃ (m n : ℝ),
    let k1 := 3*m^2 + 2*a*m + b
    let k2 := 3*n^2 + 2*a*n + b
    k2 = 4*k1 →
    a^2 = 3*b :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_relation_l2682_268230


namespace NUMINAMATH_CALUDE_grady_blue_cubes_l2682_268278

theorem grady_blue_cubes (grady_red : ℕ) (gage_initial_red gage_initial_blue : ℕ) (gage_total : ℕ) :
  grady_red = 20 →
  gage_initial_red = 10 →
  gage_initial_blue = 12 →
  gage_total = 35 →
  ∃ (grady_blue : ℕ),
    (2 * grady_red / 5 + grady_blue / 3 + gage_initial_red + gage_initial_blue = gage_total) ∧
    grady_blue = 15 :=
by sorry

end NUMINAMATH_CALUDE_grady_blue_cubes_l2682_268278


namespace NUMINAMATH_CALUDE_stock_exchange_problem_l2682_268221

theorem stock_exchange_problem (total_stocks : ℕ) 
  (h_total : total_stocks = 1980) 
  (H L : ℕ) 
  (h_relation : H = L + L / 5) 
  (h_sum : H + L = total_stocks) : 
  H = 1080 := by
sorry

end NUMINAMATH_CALUDE_stock_exchange_problem_l2682_268221


namespace NUMINAMATH_CALUDE_stone_piles_problem_l2682_268268

theorem stone_piles_problem (x y : ℕ) : 
  (y + 100 = 2 * (x - 100)) → 
  (∃ z : ℕ, x + z = 5 * (y - z)) → 
  x ≥ 170 → 
  (x = 170 ∧ y = 40) ∨ x > 170 :=
sorry

end NUMINAMATH_CALUDE_stone_piles_problem_l2682_268268


namespace NUMINAMATH_CALUDE_perpendicular_diagonals_imply_cyclic_projections_l2682_268289

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the concept of perpendicular diagonals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  let AC := (q.C.1 - q.A.1, q.C.2 - q.A.2)
  let BD := (q.D.1 - q.B.1, q.D.2 - q.B.2)
  AC.1 * BD.1 + AC.2 * BD.2 = 0

-- Define the projection of a point onto a line segment
def project_point (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the intersection point of diagonals
def diagonal_intersection (q : Quadrilateral) : ℝ × ℝ :=
  sorry

-- Define a cyclic quadrilateral
def is_cyclic (A B C D : ℝ × ℝ) : Prop :=
  sorry

-- Main theorem
theorem perpendicular_diagonals_imply_cyclic_projections (q : Quadrilateral) :
  has_perpendicular_diagonals q →
  let I := diagonal_intersection q
  let A1 := project_point I q.A q.B
  let B1 := project_point I q.B q.C
  let C1 := project_point I q.C q.D
  let D1 := project_point I q.D q.A
  is_cyclic A1 B1 C1 D1 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_diagonals_imply_cyclic_projections_l2682_268289


namespace NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l2682_268275

theorem smallest_solution_for_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^3⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 18 ∧
  (∀ (y : ℝ), y > 0 → (⌊y^3⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 18 → y ≥ x) ∧
  x = 369 / 19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l2682_268275


namespace NUMINAMATH_CALUDE_area_between_curves_l2682_268252

-- Define the functions f and g
def f (x : ℝ) : ℝ := -2 * x^2 + 7 * x - 6
def g (x : ℝ) : ℝ := -x

-- Define the theorem
theorem area_between_curves : 
  ∃ (a b : ℝ), a < b ∧ 
  (∫ (x : ℝ) in a..b, f x - g x) = 8/3 := by
sorry

end NUMINAMATH_CALUDE_area_between_curves_l2682_268252


namespace NUMINAMATH_CALUDE_john_game_period_duration_l2682_268266

/-- Calculates the duration of each period in John's game --/
def period_duration (points_per_interval : ℕ) (total_points : ℕ) (num_periods : ℕ) : ℕ :=
  (total_points / points_per_interval * 4) / num_periods

/-- Proves that each period lasts 12 minutes given the game conditions --/
theorem john_game_period_duration :
  period_duration 7 42 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_john_game_period_duration_l2682_268266


namespace NUMINAMATH_CALUDE_ferris_wheel_small_seats_l2682_268235

/-- The number of small seats on the Ferris wheel -/
def small_seats : ℕ := 2

/-- The number of people each small seat can hold -/
def people_per_small_seat : ℕ := 14

/-- The total number of people who can ride on small seats -/
def total_people_on_small_seats : ℕ := small_seats * people_per_small_seat

theorem ferris_wheel_small_seats :
  total_people_on_small_seats = 28 := by sorry

end NUMINAMATH_CALUDE_ferris_wheel_small_seats_l2682_268235


namespace NUMINAMATH_CALUDE_board_cutting_l2682_268245

theorem board_cutting (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) : 
  total_length = 120 →
  shorter_piece + (2 * shorter_piece + difference) = total_length →
  difference = 15 →
  shorter_piece = 35 := by
sorry

end NUMINAMATH_CALUDE_board_cutting_l2682_268245


namespace NUMINAMATH_CALUDE_area_of_intersection_region_l2682_268237

noncomputable def f₀ (x : ℝ) : ℝ := |x|

noncomputable def f₁ (x : ℝ) : ℝ := |f₀ x - 1|

noncomputable def f₂ (x : ℝ) : ℝ := |f₁ x - 2|

theorem area_of_intersection_region (f₀ f₁ f₂ : ℝ → ℝ) :
  (f₀ = fun x ↦ |x|) →
  (f₁ = fun x ↦ |f₀ x - 1|) →
  (f₂ = fun x ↦ |f₁ x - 2|) →
  (∫ x in (-3)..(3), min (f₂ x) 0) = -7 :=
by sorry

end NUMINAMATH_CALUDE_area_of_intersection_region_l2682_268237


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2682_268236

-- Define the condition for m and n
def condition (m n : ℝ) : Prop := m < 0 ∧ 0 < n

-- Define what it means for the equation to represent a hyperbola
def is_hyperbola (m n : ℝ) : Prop := 
  ∃ (x y : ℝ), n * x^2 + m * y^2 = 1 ∧ (m < 0 ∧ n > 0) ∨ (m > 0 ∧ n < 0)

-- State the theorem
theorem condition_sufficient_not_necessary (m n : ℝ) :
  (condition m n → is_hyperbola m n) ∧ 
  ¬(is_hyperbola m n → condition m n) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2682_268236


namespace NUMINAMATH_CALUDE_sum_remainder_l2682_268262

theorem sum_remainder (m : ℤ) : ((9 - m) + (m + 5)) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l2682_268262


namespace NUMINAMATH_CALUDE_wage_increase_calculation_l2682_268288

theorem wage_increase_calculation (W H W' H' : ℝ) : 
  W > 0 → H > 0 → W' > W → -- Initial conditions
  H' = H * (1 - 0.20) → -- 20% reduction in hours
  W * H = W' * H' → -- Total weekly income remains the same
  (W' - W) / W = 0.25 := by -- The wage increase is 25%
  sorry

end NUMINAMATH_CALUDE_wage_increase_calculation_l2682_268288


namespace NUMINAMATH_CALUDE_alcohol_concentration_after_addition_l2682_268263

/-- Proves that adding 5.5 liters of alcohol and 4.5 liters of water to a 40-liter solution
    with 5% alcohol concentration results in a 15% alcohol solution. -/
theorem alcohol_concentration_after_addition :
  let initial_volume : ℝ := 40
  let initial_concentration : ℝ := 0.05
  let added_alcohol : ℝ := 5.5
  let added_water : ℝ := 4.5
  let final_concentration : ℝ := 0.15
  let final_volume : ℝ := initial_volume + added_alcohol + added_water
  initial_volume * initial_concentration + added_alcohol =
    final_volume * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_concentration_after_addition_l2682_268263


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_46_l2682_268293

theorem consecutive_integers_sum_46 :
  ∃ (x : ℕ), x > 0 ∧ x + (x + 1) + (x + 2) + (x + 3) = 46 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_46_l2682_268293


namespace NUMINAMATH_CALUDE_johans_house_rooms_l2682_268247

theorem johans_house_rooms (walls_per_room : ℕ) (green_ratio : ℚ) (purple_walls : ℕ) : 
  walls_per_room = 8 →
  green_ratio = 3/5 →
  purple_walls = 32 →
  ∃ (total_rooms : ℕ), total_rooms = 10 ∧ 
    (purple_walls : ℚ) / walls_per_room = (1 - green_ratio) * total_rooms :=
by sorry

end NUMINAMATH_CALUDE_johans_house_rooms_l2682_268247


namespace NUMINAMATH_CALUDE_euro_calculation_l2682_268281

/-- The € operation as defined in the problem -/
def euro (x y z : ℕ) : ℕ := (2 * x * y + y^2) * z^3

/-- The statement to be proven -/
theorem euro_calculation : euro 7 (euro 4 5 3) 2 = 24844760 := by
  sorry

end NUMINAMATH_CALUDE_euro_calculation_l2682_268281


namespace NUMINAMATH_CALUDE_sqrt_one_plus_xy_rational_l2682_268282

theorem sqrt_one_plus_xy_rational (x y : ℚ) 
  (h : (x^2 + y^2 - 2) * (x + y)^2 + (x*y + 1)^2 = 0) : 
  ∃ (q : ℚ), q^2 = 1 + x*y := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_plus_xy_rational_l2682_268282


namespace NUMINAMATH_CALUDE_triangle_height_proof_l2682_268217

/-- Given a triangle with base 4 meters and a constant k = 2 meters, 
    prove that its height is 4 meters when its area satisfies two equations. -/
theorem triangle_height_proof (height : ℝ) (k : ℝ) (base : ℝ) : 
  k = 2 →
  base = 4 →
  (base^2) / (4 * (height - k)) = (1/2) * base * height →
  height = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_proof_l2682_268217


namespace NUMINAMATH_CALUDE_reflection_result_l2682_268229

/-- Reflect a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflect a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The initial point C -/
def C : ℝ × ℝ := (-1, 4)

theorem reflection_result :
  (reflect_x (reflect_y C)) = (1, -4) := by
  sorry

end NUMINAMATH_CALUDE_reflection_result_l2682_268229


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l2682_268227

theorem triangle_angle_theorem (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : (a^2 + b^2) * (a^2 + b^2 - c^2) = 3 * a^2 * b^2) (h5 : a = b) :
  let C := Real.arccos ((a^4 + b^4 - a^2 * b^2) / (2 * a * b * (a^2 + b^2)))
  C = Real.arccos (1/4) :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l2682_268227


namespace NUMINAMATH_CALUDE_tan_150_degrees_l2682_268285

theorem tan_150_degrees : Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l2682_268285


namespace NUMINAMATH_CALUDE_jackson_meat_problem_l2682_268233

theorem jackson_meat_problem (M : ℝ) : 
  M > 0 → 
  M - (1/4 * M) - 3 = 12 → 
  M = 20 :=
by sorry

end NUMINAMATH_CALUDE_jackson_meat_problem_l2682_268233


namespace NUMINAMATH_CALUDE_ones_digit_73_pow_355_l2682_268228

theorem ones_digit_73_pow_355 : (73^355) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_73_pow_355_l2682_268228


namespace NUMINAMATH_CALUDE_room_selection_equivalence_l2682_268277

def total_rooms : ℕ := 6

def select_at_least_two (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ k => if k ≥ 2 then Nat.choose n k else 0)

def sum_of_combinations (n : ℕ) : ℕ :=
  Nat.choose n 3 + 2 * Nat.choose n 4 + Nat.choose n 5 + Nat.choose n 6

def power_minus_seven (n : ℕ) : ℕ :=
  2^n - 7

theorem room_selection_equivalence :
  select_at_least_two total_rooms = sum_of_combinations total_rooms ∧
  select_at_least_two total_rooms = power_minus_seven total_rooms := by
  sorry

end NUMINAMATH_CALUDE_room_selection_equivalence_l2682_268277


namespace NUMINAMATH_CALUDE_power_of_three_equation_l2682_268205

theorem power_of_three_equation (k : ℤ) : 
  3^1999 - 3^1998 - 3^1997 + 3^1996 = k * 3^1996 → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equation_l2682_268205


namespace NUMINAMATH_CALUDE_bus_speed_increase_l2682_268246

/-- The speed increase of a bus per hour, given initial speed and total distance traveled. -/
theorem bus_speed_increase 
  (S₀ : ℝ) 
  (total_distance : ℝ) 
  (x : ℝ) 
  (h1 : S₀ = 35) 
  (h2 : total_distance = 552) 
  (h3 : total_distance = S₀ * 12 + x * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11)) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_bus_speed_increase_l2682_268246


namespace NUMINAMATH_CALUDE_log_x2y2_l2682_268238

theorem log_x2y2 (x y : ℝ) (h1 : Real.log (x^2 * y^5) = 2) (h2 : Real.log (x^3 * y^2) = 2) :
  Real.log (x^2 * y^2) = 16/11 := by
sorry

end NUMINAMATH_CALUDE_log_x2y2_l2682_268238


namespace NUMINAMATH_CALUDE_class_composition_l2682_268296

theorem class_composition (total : ℕ) (girls boys : ℕ) : 
  girls = (6 : ℚ) / 10 * total →
  (girls - 1 : ℚ) / (total - 3) = 25 / 40 →
  girls = 21 ∧ boys = 14 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l2682_268296


namespace NUMINAMATH_CALUDE_last_three_digits_is_419_l2682_268231

/-- A function that generates the nth digit in the list of increasing positive integers starting with 2 -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1998th, 1999th, and 2000th digits -/
def lastThreeDigits : ℕ := 
  100 * (nthDigit 1998) + 10 * (nthDigit 1999) + (nthDigit 2000)

/-- Theorem stating that the last three digits form the number 419 -/
theorem last_three_digits_is_419 : lastThreeDigits = 419 := by sorry

end NUMINAMATH_CALUDE_last_three_digits_is_419_l2682_268231


namespace NUMINAMATH_CALUDE_a_2007_equals_4_l2682_268260

def f : ℕ → ℕ
  | 1 => 4
  | 2 => 1
  | 3 => 3
  | 4 => 5
  | 5 => 2
  | _ => 0

def a : ℕ → ℕ
  | 0 => 5
  | n + 1 => f (a n)

theorem a_2007_equals_4 : a 2007 = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_2007_equals_4_l2682_268260


namespace NUMINAMATH_CALUDE_cos_product_20_40_60_80_l2682_268280

theorem cos_product_20_40_60_80 : 
  Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (60 * π / 180) * Real.cos (80 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_20_40_60_80_l2682_268280


namespace NUMINAMATH_CALUDE_pushup_difference_l2682_268243

theorem pushup_difference (zachary_pushups : Real) (david_more_than_zachary : Real) (john_less_than_david : Real)
  (h1 : zachary_pushups = 15.5)
  (h2 : david_more_than_zachary = 39.2)
  (h3 : john_less_than_david = 9.3) :
  let david_pushups := zachary_pushups + david_more_than_zachary
  let john_pushups := david_pushups - john_less_than_david
  john_pushups - zachary_pushups = 29.9 := by
sorry

end NUMINAMATH_CALUDE_pushup_difference_l2682_268243


namespace NUMINAMATH_CALUDE_min_value_of_2x_plus_y_min_value_is_one_min_value_exists_l2682_268269

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + y + 2 * x * y = 5/4) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 2 * a + b + 2 * a * b = 5/4 → 2 * x + y ≤ 2 * a + b :=
by sorry

theorem min_value_is_one (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + y + 2 * x * y = 5/4) : 
  2 * x + y ≥ 1 :=
by sorry

theorem min_value_exists (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + y + 2 * x * y = 5/4) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b + 2 * a * b = 5/4 ∧ 2 * a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_2x_plus_y_min_value_is_one_min_value_exists_l2682_268269


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l2682_268206

def M : ℕ := sorry

theorem highest_power_of_three_dividing_M : 
  ∃ (k : ℕ), (3^1 ∣ M) ∧ ¬(3^(k+2) ∣ M) :=
sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l2682_268206


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2682_268215

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_def : ∀ n : ℕ, S n = (n : ℝ) * a 1 + (n : ℝ) * (n - 1) / 2 * (a 2 - a 1)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h : seq.a 1 + 5 * seq.a 3 = seq.S 8) :
    seq.a 10 = 0 ∧ seq.S 7 = seq.S 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2682_268215


namespace NUMINAMATH_CALUDE_expected_adjacent_pairs_standard_deck_l2682_268291

/-- A standard deck of cards. -/
structure Deck :=
  (total : Nat)
  (black : Nat)
  (red : Nat)
  (h_total : total = 52)
  (h_half : black = red)
  (h_sum : black + red = total)

/-- The expected number of adjacent pairs with one black and one red card
    in a circular arrangement of cards from a standard deck. -/
def expectedAdjacentPairs (d : Deck) : Rat :=
  (d.total : Rat) * (d.black : Rat) * (d.red : Rat) / ((d.total - 1) : Rat)

theorem expected_adjacent_pairs_standard_deck :
  ∃ (d : Deck), expectedAdjacentPairs d = 1352 / 51 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_pairs_standard_deck_l2682_268291


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2682_268223

/-- Given a geometric sequence with common ratio greater than 1,
    if the difference between the 5th and 1st term is 15,
    and the difference between the 4th and 2nd term is 6,
    then the 3rd term is 4. -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- The sequence
  (q : ℝ)      -- Common ratio
  (h_geom : ∀ n, a (n + 1) = a n * q)  -- Geometric sequence property
  (h_q : q > 1)  -- Common ratio > 1
  (h_diff1 : a 5 - a 1 = 15)  -- Given condition
  (h_diff2 : a 4 - a 2 = 6)   -- Given condition
  : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2682_268223


namespace NUMINAMATH_CALUDE_system_solution_unique_l2682_268207

theorem system_solution_unique :
  ∃! (x y : ℝ), x + 2*y = 9 ∧ 3*x - 2*y = -1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2682_268207


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2682_268241

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a - 15 → 
  (a + b + c) / 3 = c + 10 → 
  a + b + c = 45 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2682_268241


namespace NUMINAMATH_CALUDE_jade_savings_l2682_268257

/-- Calculates Jade's monthly savings given her earnings and spending patterns. -/
theorem jade_savings (monthly_earnings : ℝ) (living_expenses_ratio : ℝ) (insurance_ratio : ℝ) :
  monthly_earnings = 1600 →
  living_expenses_ratio = 0.75 →
  insurance_ratio = 1/5 →
  monthly_earnings * (1 - living_expenses_ratio - insurance_ratio) = 80 :=
by sorry

end NUMINAMATH_CALUDE_jade_savings_l2682_268257


namespace NUMINAMATH_CALUDE_particular_number_proof_l2682_268261

theorem particular_number_proof : ∃! x : ℚ, ((x + 2 - 6) * 3) / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_proof_l2682_268261


namespace NUMINAMATH_CALUDE_binary_1010101_equals_octal_125_l2682_268201

/-- Converts a binary number represented as a list of bits to a natural number -/
def binary_to_natural (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its octal representation -/
def natural_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: natural_to_octal (n / 8)

/-- The binary representation of 1010101₂ -/
def binary_1010101 : List Bool := [true, false, true, false, true, false, true]

/-- The octal representation of 125₈ -/
def octal_125 : List ℕ := [5, 2, 1]

theorem binary_1010101_equals_octal_125 :
  natural_to_octal (binary_to_natural binary_1010101) = octal_125 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010101_equals_octal_125_l2682_268201


namespace NUMINAMATH_CALUDE_max_points_scored_l2682_268239

-- Define the variables
def total_shots : ℕ := 50
def three_point_success_rate : ℚ := 3 / 10
def two_point_success_rate : ℚ := 4 / 10

-- Define the function to calculate points
def calculate_points (three_point_shots : ℕ) : ℚ :=
  let two_point_shots : ℕ := total_shots - three_point_shots
  (three_point_success_rate * 3 * three_point_shots) + (two_point_success_rate * 2 * two_point_shots)

-- Theorem statement
theorem max_points_scored :
  ∃ (max_points : ℚ), max_points = 45 ∧
  ∀ (x : ℕ), x ≤ total_shots → calculate_points x ≤ max_points :=
sorry

end NUMINAMATH_CALUDE_max_points_scored_l2682_268239


namespace NUMINAMATH_CALUDE_arcsin_neg_half_equals_neg_pi_sixth_l2682_268225

theorem arcsin_neg_half_equals_neg_pi_sixth : 
  Real.arcsin (-1/2) = -π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_half_equals_neg_pi_sixth_l2682_268225


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2682_268250

theorem square_area_from_diagonal (d : ℝ) (h : d = 2) : 
  (d^2 / 2) = 2 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2682_268250


namespace NUMINAMATH_CALUDE_debbys_museum_pictures_l2682_268290

theorem debbys_museum_pictures 
  (zoo_pictures : ℕ) 
  (deleted_pictures : ℕ) 
  (remaining_pictures : ℕ) 
  (h1 : zoo_pictures = 24)
  (h2 : deleted_pictures = 14)
  (h3 : remaining_pictures = 22)
  (h4 : remaining_pictures = zoo_pictures + museum_pictures - deleted_pictures) :
  museum_pictures = 12 := by
  sorry

#check debbys_museum_pictures

end NUMINAMATH_CALUDE_debbys_museum_pictures_l2682_268290


namespace NUMINAMATH_CALUDE_m_range_l2682_268284

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom even_func : ∀ x : ℝ, f (-x) = f x
axiom increasing_neg : ∀ a b : ℝ, a < b → b < 0 → (f a - f b) / (a - b) > 0
axiom condition : ∀ m : ℝ, f (2*m + 1) > f (2*m)

-- State the theorem
theorem m_range (m : ℝ) : 
  (∀ x : ℝ, f (-x) = f x) → 
  (∀ a b : ℝ, a < b → b < 0 → (f a - f b) / (a - b) > 0) → 
  (f (2*m + 1) > f (2*m)) → 
  m < -1/4 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2682_268284


namespace NUMINAMATH_CALUDE_largest_number_l2682_268242

def numbers : List ℝ := [0.988, 0.9808, 0.989, 0.9809, 0.998]

theorem largest_number (n : ℝ) (hn : n ∈ numbers) : n ≤ 0.998 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l2682_268242


namespace NUMINAMATH_CALUDE_hyperbola_and_line_equations_l2682_268297

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 49 + y^2 / 24 = 1

-- Define the asymptotes of the hyperbola
def asymptotes (x y : ℝ) : Prop := y = 4/3 * x ∨ y = -4/3 * x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the line passing through the right focus
def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 5 * Real.sqrt 3 = 0

-- State the theorem
theorem hyperbola_and_line_equations :
  (∀ x y : ℝ, ellipse x y → asymptotes x y → hyperbola x y) ∧
  (∀ x y : ℝ, ellipse x y → line x y) := by sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_equations_l2682_268297


namespace NUMINAMATH_CALUDE_house_bedrooms_count_l2682_268249

/-- A house with two floors and a certain number of bedrooms on each floor. -/
structure House where
  second_floor_bedrooms : ℕ
  first_floor_bedrooms : ℕ

/-- The total number of bedrooms in a house. -/
def total_bedrooms (h : House) : ℕ :=
  h.second_floor_bedrooms + h.first_floor_bedrooms

/-- Theorem stating that a house with 2 bedrooms on the second floor and 8 on the first floor has 10 bedrooms in total. -/
theorem house_bedrooms_count :
  ∀ (h : House), h.second_floor_bedrooms = 2 → h.first_floor_bedrooms = 8 →
  total_bedrooms h = 10 := by
  sorry

end NUMINAMATH_CALUDE_house_bedrooms_count_l2682_268249


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2682_268299

/-- Represents the number of students in a stratum -/
structure Stratum where
  size : ℕ

/-- Represents a sample from a stratum -/
structure Sample where
  size : ℕ

/-- Calculates the total sample size for stratified sampling -/
def calculateTotalSampleSize (male : Stratum) (female : Stratum) (femaleSample : Sample) : ℕ :=
  (femaleSample.size * (male.size + female.size)) / female.size

/-- Theorem: Given the specified conditions, the total sample size is 176 -/
theorem stratified_sampling_theorem (male : Stratum) (female : Stratum) (femaleSample : Sample)
    (h1 : male.size = 1200)
    (h2 : female.size = 1000)
    (h3 : femaleSample.size = 80) :
    calculateTotalSampleSize male female femaleSample = 176 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2682_268299


namespace NUMINAMATH_CALUDE_log_plus_fraction_gt_one_l2682_268295

theorem log_plus_fraction_gt_one (x a : ℝ) (hx : x > 1) (ha : a ≥ 1/2) :
  Real.log x + a / (x - 1) > 1 := by sorry

end NUMINAMATH_CALUDE_log_plus_fraction_gt_one_l2682_268295


namespace NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l2682_268264

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  /-- The base angle of the isosceles triangle -/
  base_angle : ℝ
  /-- The altitude of the isosceles triangle -/
  altitude : ℝ
  /-- The side length of the isosceles triangle -/
  side_length : ℝ
  /-- Constraint that the base angle is between 0 and π/2 -/
  angle_constraint : 0 < base_angle ∧ base_angle < π / 2
  /-- Constraint that the altitude is positive -/
  altitude_positive : altitude > 0
  /-- Constraint that the side length is positive -/
  side_positive : side_length > 0

/-- Theorem stating that there exist multiple non-congruent isosceles triangles
    with the same base angle and altitude -/
theorem isosceles_triangle_not_unique :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1.base_angle = t2.base_angle ∧
    t1.altitude = t2.altitude ∧
    t1.side_length ≠ t2.side_length :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l2682_268264


namespace NUMINAMATH_CALUDE_multiply_and_add_equality_l2682_268272

theorem multiply_and_add_equality : 24 * 42 + 58 * 24 + 12 * 24 = 2688 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_equality_l2682_268272


namespace NUMINAMATH_CALUDE_tournament_has_king_l2682_268276

/-- Represents the result of a match between two teams -/
inductive MatchResult
  | Win
  | Lose

/-- A tournament with m teams -/
structure Tournament (m : ℕ) where
  /-- The result of a match between two teams -/
  result : Fin m → Fin m → MatchResult
  /-- Each pair of teams has competed exactly once -/
  competed_once : ∀ i j : Fin m, i ≠ j → (result i j = MatchResult.Win ∧ result j i = MatchResult.Lose) ∨
                                        (result i j = MatchResult.Lose ∧ result j i = MatchResult.Win)

/-- Definition of a king in the tournament -/
def is_king (t : Tournament m) (x : Fin m) : Prop :=
  ∀ y : Fin m, y ≠ x → 
    (t.result x y = MatchResult.Win) ∨ 
    (∃ z : Fin m, t.result x z = MatchResult.Win ∧ t.result z y = MatchResult.Win)

/-- Theorem: Every tournament has a king -/
theorem tournament_has_king (m : ℕ) (t : Tournament m) : ∃ x : Fin m, is_king t x := by
  sorry

end NUMINAMATH_CALUDE_tournament_has_king_l2682_268276


namespace NUMINAMATH_CALUDE_julio_lime_cost_l2682_268210

/-- Represents the cost of limes for Julio's mocktails over 30 days -/
def lime_cost (mocktails_per_day : ℕ) (tbsp_per_mocktail : ℕ) (tbsp_per_lime : ℕ) (limes_per_dollar : ℕ) (days : ℕ) : ℚ :=
  let limes_needed := (mocktails_per_day * tbsp_per_mocktail * days) / tbsp_per_lime
  let lime_sets := (limes_needed + limes_per_dollar - 1) / limes_per_dollar
  lime_sets

theorem julio_lime_cost : 
  lime_cost 1 1 2 3 30 = 5 := by
  sorry

end NUMINAMATH_CALUDE_julio_lime_cost_l2682_268210


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2682_268256

/-- Given four points in 3D space, this theorem states that the intersection of the lines
    passing through the first two points and the last two points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) : 
  A = (5, -6, 8) →
  B = (15, -16, 13) →
  C = (1, 4, -5) →
  D = (3, -4, 11) →
  ∃ t s : ℝ, 
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (1 + 2*s, 4 - 8*s, -5 + 16*s) ∧
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (3, -4, 7) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2682_268256


namespace NUMINAMATH_CALUDE_range_of_a_l2682_268258

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, 2 * x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) : ¬(p a ∨ q a) → (a > 2 ∨ (-2 < a ∧ a < 1)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2682_268258


namespace NUMINAMATH_CALUDE_boys_age_problem_l2682_268286

theorem boys_age_problem (x : ℕ) : x + 4 = 2 * (x - 6) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_boys_age_problem_l2682_268286


namespace NUMINAMATH_CALUDE_five_people_arrangement_l2682_268292

/-- The number of arrangements of n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of arrangements of n people in a row where two specific people are next to each other -/
def adjacentArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of arrangements of n people in a row where two specific people are not next to each other -/
def nonAdjacentArrangements (n : ℕ) : ℕ := totalArrangements n - adjacentArrangements n

theorem five_people_arrangement :
  nonAdjacentArrangements 5 = 72 :=
by sorry

end NUMINAMATH_CALUDE_five_people_arrangement_l2682_268292


namespace NUMINAMATH_CALUDE_coin_value_calculation_l2682_268213

def total_value (num_dimes num_quarters : ℕ) (dime_value quarter_value : ℚ) : ℚ :=
  (num_dimes : ℚ) * dime_value + (num_quarters : ℚ) * quarter_value

theorem coin_value_calculation :
  total_value 22 10 (10 / 100) (25 / 100) = 470 / 100 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_calculation_l2682_268213


namespace NUMINAMATH_CALUDE_relationship_2x_3sinx_l2682_268214

theorem relationship_2x_3sinx :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧
  (∀ x : ℝ, 0 < x ∧ x < θ → 2 * x < 3 * Real.sin x) ∧
  (2 * θ = 3 * Real.sin θ) ∧
  (∀ x : ℝ, θ < x ∧ x < π / 2 → 2 * x > 3 * Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_relationship_2x_3sinx_l2682_268214


namespace NUMINAMATH_CALUDE_ellipse_equation_l2682_268271

/-- An ellipse with focal length 2 passing through (-√5, 0) has a standard equation of either x²/5 + y²/4 = 1 or y²/6 + x²/5 = 1 -/
theorem ellipse_equation (f : ℝ) (P : ℝ × ℝ) : 
  f = 2 → P = (-Real.sqrt 5, 0) → 
  (∃ (x y : ℝ), x^2/5 + y^2/4 = 1) ∨ (∃ (x y : ℝ), y^2/6 + x^2/5 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2682_268271


namespace NUMINAMATH_CALUDE_equation_solution_l2682_268298

theorem equation_solution (k : ℤ) : 
  let n : ℚ := -5 + 1024 * k
  (5/4) * n + 5/4 = n := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2682_268298


namespace NUMINAMATH_CALUDE_sampling_suitable_for_large_events_sampling_suitable_for_beijing_olympics_l2682_268270

/-- Represents a survey method -/
inductive SurveyMethod
  | Sampling
  | Comprehensive

/-- Represents a large-scale event -/
structure LargeEvent where
  name : String
  potential_viewers : ℕ

/-- Defines when a survey method is suitable for an event -/
def is_suitable_survey_method (method : SurveyMethod) (event : LargeEvent) : Prop :=
  match method with
  | SurveyMethod.Sampling => 
      event.potential_viewers > 1000000 ∧ 
      (∀ n : ℕ, n < event.potential_viewers → ∃ sample : Finset ℕ, sample.card = n)
  | SurveyMethod.Comprehensive => 
      event.potential_viewers ≤ 1000000

/-- The main theorem stating that sampling survey is suitable for large events -/
theorem sampling_suitable_for_large_events (event : LargeEvent) 
  (h1 : event.potential_viewers > 1000000) 
  (h2 : ∀ n : ℕ, n < event.potential_viewers → ∃ sample : Finset ℕ, sample.card = n) :
  is_suitable_survey_method SurveyMethod.Sampling event :=
by sorry

/-- The Beijing Winter Olympics as an instance of LargeEvent -/
def beijing_winter_olympics : LargeEvent :=
  { name := "Beijing Winter Olympics"
  , potential_viewers := 2000000000 }  -- An example large number

/-- Theorem stating that sampling survey is suitable for the Beijing Winter Olympics -/
theorem sampling_suitable_for_beijing_olympics :
  is_suitable_survey_method SurveyMethod.Sampling beijing_winter_olympics :=
by sorry

end NUMINAMATH_CALUDE_sampling_suitable_for_large_events_sampling_suitable_for_beijing_olympics_l2682_268270


namespace NUMINAMATH_CALUDE_square_roots_problem_l2682_268202

theorem square_roots_problem (m : ℝ) :
  (2*m - 4)^2 = (3*m - 1)^2 → (2*m - 4)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2682_268202


namespace NUMINAMATH_CALUDE_garden_perimeter_proof_l2682_268267

/-- The perimeter of a rectangular garden with given length and breadth. -/
def garden_perimeter (length breadth : ℝ) : ℝ :=
  2 * length + 2 * breadth

/-- Theorem: The perimeter of a rectangular garden with length 375 m and breadth 100 m is 950 m. -/
theorem garden_perimeter_proof :
  garden_perimeter 375 100 = 950 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_proof_l2682_268267


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2682_268244

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (17 * x - 3 * y) = 5 / 7) :
  x / y = -20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2682_268244


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2682_268274

theorem unique_solution_quadratic (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x - a = 0) ↔ a = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2682_268274


namespace NUMINAMATH_CALUDE_power_function_theorem_l2682_268279

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x ^ n

-- Define the theorem
theorem power_function_theorem (f : ℝ → ℝ) (h : isPowerFunction f) :
  f 2 = 1/4 → f (1/2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_function_theorem_l2682_268279


namespace NUMINAMATH_CALUDE_inequality_system_solution_implies_a_greater_than_negative_one_l2682_268273

theorem inequality_system_solution_implies_a_greater_than_negative_one :
  (∃ x : ℝ, x + a ≥ 0 ∧ 1 - 2*x > x - 2) → a > -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_implies_a_greater_than_negative_one_l2682_268273


namespace NUMINAMATH_CALUDE_grade_c_boxes_l2682_268209

theorem grade_c_boxes (total : ℕ) (m n t : ℕ) 
  (h1 : total = 420)
  (h2 : 2 * t = m + n) : 
  (total / 3 : ℕ) = 140 := by sorry

end NUMINAMATH_CALUDE_grade_c_boxes_l2682_268209


namespace NUMINAMATH_CALUDE_example_is_quadratic_l2682_268208

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 3x + 1 = 0 is a quadratic equation -/
theorem example_is_quadratic : is_quadratic_equation (fun x ↦ x^2 - 3*x + 1) := by
  sorry

end NUMINAMATH_CALUDE_example_is_quadratic_l2682_268208


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2682_268232

-- Problem 1
theorem problem_1 : (-1)^2023 + 2 * Real.cos (π / 4) - |Real.sqrt 2 - 2| - (1 / 2)⁻¹ = 2 * Real.sqrt 2 - 5 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx : x ≠ 0) : 
  (1 - 1 / (x + 1)) / ((x^2) / (x^2 + 2*x + 1)) = (x + 1) / x := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2682_268232


namespace NUMINAMATH_CALUDE_blanket_average_price_l2682_268212

theorem blanket_average_price : 
  let blanket_group1 := (3, 100)
  let blanket_group2 := (5, 150)
  let blanket_group3 := (2, 570)
  let total_blankets := blanket_group1.1 + blanket_group2.1 + blanket_group3.1
  let total_cost := blanket_group1.1 * blanket_group1.2 + blanket_group2.1 * blanket_group2.2 + blanket_group3.1 * blanket_group3.2
  total_cost / total_blankets = 219 := by
sorry

end NUMINAMATH_CALUDE_blanket_average_price_l2682_268212


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2682_268248

theorem sum_of_three_numbers : 3/8 + 0.125 + 9.51 = 10.01 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2682_268248


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l2682_268265

/-- Given a hyperbola with equation x²/32 - y²/4 = 1, the distance between its foci is 12 -/
theorem hyperbola_foci_distance (x y : ℝ) :
  x^2 / 32 - y^2 / 4 = 1 → ∃ c : ℝ, c = 6 ∧ 2 * c = 12 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l2682_268265


namespace NUMINAMATH_CALUDE_magic_square_sum_l2682_268253

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e f g h i : ℝ)
  (row_sum : a + b + c = d + e + f ∧ d + e + f = g + h + i)
  (col_sum : a + d + g = b + e + h ∧ b + e + h = c + f + i)
  (diag_sum : a + e + i = c + e + g)

/-- The sum of all elements in a magic square -/
def total_sum (m : MagicSquare) : ℝ := m.a + m.b + m.c + m.d + m.e + m.f + m.g + m.h + m.i

/-- Theorem: Sum of remaining squares in a specific magic square -/
theorem magic_square_sum :
  ∀ (m : MagicSquare),
    m.b = 7 ∧ m.c = 2018 ∧ m.g = 4 →
    (total_sum m) - (m.b + m.c + m.g) = -11042.5 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_sum_l2682_268253


namespace NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l2682_268240

theorem smallest_n_for_terminating_decimal : 
  ∃ (n : ℕ+), n = 24 ∧ 
  (∀ (m : ℕ+), m < n → ¬(∃ (a b : ℕ), (m : ℚ) / (m + 101 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0 ∧ 
  (∀ (p : ℕ), Prime p → p ∣ b → p = 2 ∨ p = 5))) ∧
  (∃ (a b : ℕ), (n : ℚ) / (n + 101 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0 ∧ 
  (∀ (p : ℕ), Prime p → p ∣ b → p = 2 ∨ p = 5)) :=
by sorry


end NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l2682_268240


namespace NUMINAMATH_CALUDE_swimmer_distance_l2682_268216

/-- Proves that the distance swam against the current is 6 km given the specified conditions -/
theorem swimmer_distance (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) 
  (h1 : swimmer_speed = 4)
  (h2 : current_speed = 1)
  (h3 : time = 2) :
  (swimmer_speed - current_speed) * time = 6 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_distance_l2682_268216


namespace NUMINAMATH_CALUDE_divisibility_property_l2682_268200

theorem divisibility_property (n : ℕ) : 
  ∃ (x y : ℤ), (x^2 + y^2 - 2018) % n = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l2682_268200


namespace NUMINAMATH_CALUDE_fraction_value_l2682_268220

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 5) (h3 : ∃ (n : ℤ), x / y = n) : x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2682_268220


namespace NUMINAMATH_CALUDE_microscope_magnification_factor_l2682_268218

/-- The magnification factor of an electron microscope, given the magnified image diameter and actual tissue diameter. -/
theorem microscope_magnification_factor 
  (magnified_diameter : ℝ) 
  (actual_diameter : ℝ) 
  (h1 : magnified_diameter = 2) 
  (h2 : actual_diameter = 0.002) : 
  magnified_diameter / actual_diameter = 1000 := by
sorry

end NUMINAMATH_CALUDE_microscope_magnification_factor_l2682_268218


namespace NUMINAMATH_CALUDE_fraction_problem_l2682_268222

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2682_268222


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2682_268255

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2682_268255


namespace NUMINAMATH_CALUDE_train_length_problem_l2682_268224

theorem train_length_problem (speed1 speed2 : ℝ) (pass_time : ℝ) (h1 : speed1 = 55) (h2 : speed2 = 50) (h3 : pass_time = 11.657142857142858) :
  let relative_speed := (speed1 + speed2) * (5 / 18)
  let total_distance := relative_speed * pass_time
  let train_length := total_distance / 2
  train_length = 170 := by sorry

end NUMINAMATH_CALUDE_train_length_problem_l2682_268224


namespace NUMINAMATH_CALUDE_girls_in_circle_l2682_268287

/-- The number of girls in a circular arrangement where one girl is
    both the fifth to the left and the eighth to the right of another girl. -/
def num_girls_in_circle : ℕ := 13

/-- Proposition: In a circular arrangement of girls, if one girl is both
    the fifth to the left and the eighth to the right of another girl,
    then the total number of girls in the circle is 13. -/
theorem girls_in_circle :
  ∀ (n : ℕ), n > 0 →
  (∃ (a b : ℕ), a < n ∧ b < n ∧ a ≠ b ∧
   ((a + 5) % n = b) ∧ ((b + 8) % n = a)) →
  n = num_girls_in_circle :=
sorry

end NUMINAMATH_CALUDE_girls_in_circle_l2682_268287


namespace NUMINAMATH_CALUDE_sum_of_remainders_consecutive_integers_l2682_268251

theorem sum_of_remainders_consecutive_integers (n : ℕ) : 
  (n % 4) + ((n + 1) % 4) + ((n + 2) % 4) + ((n + 3) % 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remainders_consecutive_integers_l2682_268251


namespace NUMINAMATH_CALUDE_opposite_of_negative_2022_l2682_268211

theorem opposite_of_negative_2022 : -((-2022 : ℤ)) = 2022 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2022_l2682_268211


namespace NUMINAMATH_CALUDE_union_of_sets_l2682_268226

theorem union_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | -2 < x ∧ x < 0} →
  B = {x : ℝ | -1 < x ∧ x < 1} →
  A ∪ B = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2682_268226


namespace NUMINAMATH_CALUDE_panda_bamboo_consumption_l2682_268259

/-- The amount of bamboo eaten by bigger pandas each day -/
def bigger_panda_bamboo : ℝ := 275

/-- The number of small pandas -/
def small_pandas : ℕ := 4

/-- The number of bigger pandas -/
def bigger_pandas : ℕ := 5

/-- The amount of bamboo eaten by small pandas each day -/
def small_panda_bamboo : ℝ := 25

/-- The total amount of bamboo eaten by all pandas in a week -/
def total_weekly_bamboo : ℝ := 2100

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem panda_bamboo_consumption :
  bigger_panda_bamboo * bigger_pandas * days_in_week +
  small_panda_bamboo * small_pandas * days_in_week =
  total_weekly_bamboo :=
sorry

end NUMINAMATH_CALUDE_panda_bamboo_consumption_l2682_268259


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l2682_268283

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2

-- State the theorem
theorem tangent_line_at_one (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2 * f (2 - x) - x^2 + 8*x - 8) : 
  ∃ m b, ∀ x, (x - 1) * (f 1) + m * (x - 1) = m * x + b := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l2682_268283


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l2682_268294

theorem sum_of_fractions_equals_seven : 
  1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
  1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 
  1 / (Real.sqrt 12 - 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l2682_268294


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2682_268204

/-- Given three lines in a plane, if they intersect at the same point, 
    we can determine the value of a parameter in one of the lines. -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃ (x y : ℝ), (a*x + 2*y + 6 = 0) ∧ (x + y - 4 = 0) ∧ (2*x - y + 1 = 0)) →
  a = -12 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2682_268204
