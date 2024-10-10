import Mathlib

namespace max_gross_profit_l2351_235153

/-- Represents the daily sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -x + 26

/-- Represents the gross profit as a function of selling price -/
def gross_profit (x : ℝ) : ℝ := x * (sales_volume x) - 4 * (sales_volume x)

/-- Theorem stating the maximum gross profit under given constraints -/
theorem max_gross_profit :
  ∃ (max_profit : ℝ),
    (∀ x : ℝ, 6 ≤ x ∧ x ≤ 12 ∧ sales_volume x ≤ 10 → gross_profit x ≤ max_profit) ∧
    (∃ x : ℝ, 6 ≤ x ∧ x ≤ 12 ∧ sales_volume x ≤ 10 ∧ gross_profit x = max_profit) ∧
    max_profit = 120 := by
  sorry

end max_gross_profit_l2351_235153


namespace base3_12012_equals_140_l2351_235190

def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem base3_12012_equals_140 :
  base3ToBase10 [2, 1, 0, 2, 1] = 140 := by
  sorry

end base3_12012_equals_140_l2351_235190


namespace isosceles_right_triangle_hypotenuse_l2351_235157

/-- An isosceles right triangle with perimeter 8 + 8√2 has a hypotenuse of length 8 -/
theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → -- side length is positive
  c > 0 → -- hypotenuse length is positive
  a + a + c = 8 + 8 * Real.sqrt 2 → -- perimeter condition
  a * a + a * a = c * c → -- Pythagorean theorem for isosceles right triangle
  c = 8 := by
sorry

end isosceles_right_triangle_hypotenuse_l2351_235157


namespace tank_full_time_l2351_235183

/-- Represents the state of a water tank system -/
structure TankSystem where
  capacity : ℕ
  fill_rate_a : ℕ
  fill_rate_b : ℕ
  drain_rate : ℕ

/-- Calculates the time needed to fill the tank -/
def time_to_fill (system : TankSystem) : ℕ :=
  let net_fill_per_cycle := system.fill_rate_a + system.fill_rate_b - system.drain_rate
  let cycles := system.capacity / net_fill_per_cycle
  cycles * 3

/-- Theorem stating that the tank will be full after 54 minutes -/
theorem tank_full_time (system : TankSystem) 
  (h1 : system.capacity = 900)
  (h2 : system.fill_rate_a = 40)
  (h3 : system.fill_rate_b = 30)
  (h4 : system.drain_rate = 20) :
  time_to_fill system = 54 := by
  sorry

#eval time_to_fill { capacity := 900, fill_rate_a := 40, fill_rate_b := 30, drain_rate := 20 }

end tank_full_time_l2351_235183


namespace least_possible_smallest_integer_l2351_235100

theorem least_possible_smallest_integer
  (a b c d : ℤ)
  (different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (average : (a + b + c + d) / 4 = 74)
  (largest : d = 90)
  (ordered : a ≤ b ∧ b ≤ c ∧ c ≤ d) :
  a ≥ 31 :=
by sorry

end least_possible_smallest_integer_l2351_235100


namespace gain_percent_calculation_l2351_235181

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 30 * S) :
  (S - C) / C * 100 = 200 / 3 := by
  sorry

end gain_percent_calculation_l2351_235181


namespace hayden_ironing_time_l2351_235106

/-- The total time Hayden spends ironing over 4 weeks -/
def total_ironing_time (shirt_time pants_time days_per_week num_weeks : ℕ) : ℕ :=
  (shirt_time + pants_time) * days_per_week * num_weeks

/-- Proof that Hayden spends 160 minutes ironing over 4 weeks -/
theorem hayden_ironing_time :
  total_ironing_time 5 3 5 4 = 160 := by
  sorry

end hayden_ironing_time_l2351_235106


namespace number_puzzle_l2351_235197

theorem number_puzzle : ∃! x : ℝ, x - 18 = 3 * (86 - x) := by sorry

end number_puzzle_l2351_235197


namespace discount_percentage_calculation_l2351_235129

theorem discount_percentage_calculation (washing_machine_cost dryer_cost total_paid : ℚ) : 
  washing_machine_cost = 100 →
  dryer_cost = washing_machine_cost - 30 →
  total_paid = 153 →
  (washing_machine_cost + dryer_cost - total_paid) / (washing_machine_cost + dryer_cost) * 100 = 10 := by
  sorry

end discount_percentage_calculation_l2351_235129


namespace quadratic_roots_difference_difference_of_roots_x2_minus_7x_plus_9_l2351_235175

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 →
  |x₁ - x₂| = Real.sqrt ((b^2 - 4*a*c) / a^2) := by
sorry

theorem difference_of_roots_x2_minus_7x_plus_9 :
  let x₁ := (7 + Real.sqrt 13) / 2
  let x₂ := (7 - Real.sqrt 13) / 2
  x₁^2 - 7*x₁ + 9 = 0 ∧ x₂^2 - 7*x₂ + 9 = 0 →
  |x₁ - x₂| = Real.sqrt 13 := by
sorry

end quadratic_roots_difference_difference_of_roots_x2_minus_7x_plus_9_l2351_235175


namespace total_scoops_needed_l2351_235187

/-- Calculates the total number of scoops needed for baking ingredients --/
theorem total_scoops_needed
  (flour_cups : ℚ)
  (sugar_cups : ℚ)
  (milk_cups : ℚ)
  (flour_scoop : ℚ)
  (sugar_scoop : ℚ)
  (milk_scoop : ℚ)
  (h_flour : flour_cups = 4)
  (h_sugar : sugar_cups = 3)
  (h_milk : milk_cups = 2)
  (h_flour_scoop : flour_scoop = 1/4)
  (h_sugar_scoop : sugar_scoop = 1/3)
  (h_milk_scoop : milk_scoop = 1/2) :
  ⌈flour_cups / flour_scoop⌉ + ⌈sugar_cups / sugar_scoop⌉ + ⌈milk_cups / milk_scoop⌉ = 29 :=
by sorry

end total_scoops_needed_l2351_235187


namespace inequality_implies_range_l2351_235189

/-- The inequality condition for all x > 1 -/
def inequality_condition (a : ℝ) : Prop :=
  ∀ x > 1, a * (x - 1) ≥ Real.log (x - 1)

/-- The range of a satisfying the inequality condition -/
def a_range (a : ℝ) : Prop :=
  a ≥ 1 / Real.exp 1

theorem inequality_implies_range :
  ∀ a : ℝ, inequality_condition a → a_range a :=
by sorry

end inequality_implies_range_l2351_235189


namespace cricket_team_right_handed_players_l2351_235110

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 150)
  (h2 : throwers = 90)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 5 = 0)
  : total_players - (total_players - throwers) / 5 = 138 := by
  sorry

end cricket_team_right_handed_players_l2351_235110


namespace train_length_l2351_235178

/-- Given a train with speed 50 km/hr crossing a pole in 9 seconds, its length is 125 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 50 → -- speed in km/hr
  time = 9 → -- time in seconds
  length = speed * (1000 / 3600) * time → -- length calculation
  length = 125 := by sorry

end train_length_l2351_235178


namespace eugene_pencils_eugene_final_pencils_l2351_235128

theorem eugene_pencils (initial_pencils : ℕ) (received_pencils : ℕ) 
  (pack_size : ℕ) (num_friends : ℕ) (given_away : ℕ) : ℕ :=
  let total_after_receiving := initial_pencils + received_pencils
  let total_in_packs := pack_size * (num_friends + 1)
  let total_before_giving := total_after_receiving + total_in_packs
  total_before_giving - given_away

theorem eugene_final_pencils :
  eugene_pencils 51 6 12 3 8 = 97 := by
  sorry

end eugene_pencils_eugene_final_pencils_l2351_235128


namespace cement_bags_calculation_l2351_235152

theorem cement_bags_calculation (cement_cost : ℕ) (sand_lorries : ℕ) (sand_tons_per_lorry : ℕ)
  (sand_cost_per_ton : ℕ) (total_payment : ℕ) :
  cement_cost = 10 →
  sand_lorries = 20 →
  sand_tons_per_lorry = 10 →
  sand_cost_per_ton = 40 →
  total_payment = 13000 →
  (total_payment - sand_lorries * sand_tons_per_lorry * sand_cost_per_ton) / cement_cost = 500 := by
  sorry

end cement_bags_calculation_l2351_235152


namespace x_squared_minus_y_squared_l2351_235199

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end x_squared_minus_y_squared_l2351_235199


namespace square_difference_of_sum_and_diff_l2351_235149

theorem square_difference_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 5) 
  (diff_eq : x - y = 10) : 
  x^2 - y^2 = 50 := by
sorry

end square_difference_of_sum_and_diff_l2351_235149


namespace min_shift_sinusoidal_graphs_l2351_235131

open Real

theorem min_shift_sinusoidal_graphs : 
  let f (x : ℝ) := 2 * sin (x + π/6)
  let g (x : ℝ) := 2 * sin (x - π/3)
  ∃ φ : ℝ, φ > 0 ∧ (∀ x : ℝ, f (x - φ) = g x) ∧
    (∀ ψ : ℝ, ψ > 0 ∧ (∀ x : ℝ, f (x - ψ) = g x) → φ ≤ ψ) ∧
    φ = π/2 :=
by sorry

end min_shift_sinusoidal_graphs_l2351_235131


namespace mod_congruence_unique_solution_l2351_235121

theorem mod_congruence_unique_solution :
  ∃! n : ℕ, n ≤ 6 ∧ n ≡ -7845 [ZMOD 7] ∧ n = 2 := by
  sorry

end mod_congruence_unique_solution_l2351_235121


namespace bread_in_pond_l2351_235176

/-- Proves that the total number of bread pieces thrown in a pond is 100 given the specified conditions --/
theorem bread_in_pond (duck1_half : ℕ → ℕ) (duck2_pieces duck3_pieces left_in_water : ℕ) : 
  duck1_half = (λ x => x / 2) ∧ 
  duck2_pieces = 13 ∧ 
  duck3_pieces = 7 ∧ 
  left_in_water = 30 → 
  ∃ total : ℕ, 
    total = 100 ∧ 
    duck1_half total + duck2_pieces + duck3_pieces + left_in_water = total :=
by
  sorry

end bread_in_pond_l2351_235176


namespace classroom_sum_problem_l2351_235116

theorem classroom_sum_problem (a b : ℤ) : 
  3 * a + 4 * b = 161 → (a = 17 ∨ b = 17) → (a = 31 ∨ b = 31) := by
  sorry

end classroom_sum_problem_l2351_235116


namespace quadratic_roots_range_l2351_235167

theorem quadratic_roots_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + (3*a - 1)*x + a + 8 = 0 ↔ x = x₁ ∨ x = x₂) →  -- quadratic equation with roots x₁ and x₂
  x₁ ≠ x₂ →  -- distinct roots
  x₁ < 1 →   -- x₁ < 1
  x₂ > 1 →   -- x₂ > 1
  a < -2 :=  -- range of a
by sorry

end quadratic_roots_range_l2351_235167


namespace sand_heap_radius_l2351_235132

/-- Given a cylindrical bucket of sand and a conical heap formed from it, 
    prove that the radius of the heap's base is 63 cm. -/
theorem sand_heap_radius : 
  ∀ (h_cylinder r_cylinder h_cone r_cone : ℝ),
  h_cylinder = 36 ∧ 
  r_cylinder = 21 ∧ 
  h_cone = 12 ∧
  π * r_cylinder^2 * h_cylinder = (1/3) * π * r_cone^2 * h_cone →
  r_cone = 63 := by
  sorry

end sand_heap_radius_l2351_235132


namespace number_calculation_l2351_235102

theorem number_calculation (x : Float) (h : x = 0.08999999999999998) :
  let number := x * 0.1
  number = 0.008999999999999999 := by
  sorry

end number_calculation_l2351_235102


namespace area_triangle_AOB_l2351_235160

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line intersecting a parabola -/
structure IntersectingLine (p : Parabola) where
  pointA : ℝ × ℝ
  pointB : ℝ × ℝ
  passes_through_focus : (pointA.1 - p.focus.1) * (pointB.2 - p.focus.2) = 
                         (pointB.1 - p.focus.1) * (pointA.2 - p.focus.2)

/-- Theorem: Area of triangle AOB for a specific parabola and intersecting line -/
theorem area_triangle_AOB 
  (p : Parabola) 
  (l : IntersectingLine p) 
  (h_parabola : p.equation = fun x y => y^2 = 4*x) 
  (h_focus : p.focus = (1, 0)) 
  (h_AF_length : Real.sqrt ((l.pointA.1 - p.focus.1)^2 + (l.pointA.2 - p.focus.2)^2) = 3) :
  let O : ℝ × ℝ := (0, 0)
  Real.sqrt (
    (l.pointA.1 * l.pointB.2 - l.pointB.1 * l.pointA.2)^2 +
    (l.pointA.1 * O.2 - O.1 * l.pointA.2)^2 +
    (O.1 * l.pointB.2 - l.pointB.1 * O.2)^2
  ) / 2 = 3 * Real.sqrt 2 / 2 := by
  sorry

end area_triangle_AOB_l2351_235160


namespace percent_only_cat_owners_l2351_235108

/-- Given a school with the following student statistics:
  * total_students: The total number of students
  * cat_owners: The number of students who own cats
  * dog_owners: The number of students who own dogs
  * both_owners: The number of students who own both cats and dogs

  This theorem proves that the percentage of students who own only cats is 8%.
-/
theorem percent_only_cat_owners
  (total_students : ℕ)
  (cat_owners : ℕ)
  (dog_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 80)
  (h3 : dog_owners = 150)
  (h4 : both_owners = 40) :
  (((cat_owners - both_owners : ℚ) / total_students) * 100 = 8) := by
  sorry

end percent_only_cat_owners_l2351_235108


namespace unique_x_with_three_prime_factors_l2351_235161

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 5^n - 1 ∧ 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ 
    x = 2^(Nat.log 2 x) * 11 * p * q) →
  x = 3124 :=
by sorry

end unique_x_with_three_prime_factors_l2351_235161


namespace right_handed_players_count_l2351_235169

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 52)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) :
  throwers + ((total_players - throwers) * 2 / 3) = 64 := by
  sorry

end right_handed_players_count_l2351_235169


namespace circle_center_l2351_235180

def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 4 * y^2 - 24 * y - 36 = 0

def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y → (x - h)^2 + (y - k)^2 = 1

theorem circle_center : is_center 1 3 := by
  sorry

end circle_center_l2351_235180


namespace boys_camp_problem_l2351_235172

theorem boys_camp_problem (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))
  (h2 : (total : ℚ) * (1/5) * (3/10) = (total : ℚ) * (1/5) * (30/100))
  (h3 : (total : ℚ) * (1/5) * (7/10) = 35) :
  total = 250 := by sorry

end boys_camp_problem_l2351_235172


namespace difference_of_squares_l2351_235130

theorem difference_of_squares (m : ℝ) : m^2 - 16 = (m + 4) * (m - 4) := by sorry

end difference_of_squares_l2351_235130


namespace modular_congruence_l2351_235150

theorem modular_congruence (x : ℤ) : 
  (5 * x + 9) % 18 = 4 → (3 * x + 15) % 18 = 12 := by sorry

end modular_congruence_l2351_235150


namespace max_value_sqrt_sum_l2351_235140

theorem max_value_sqrt_sum (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 
    Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b)) ≥ Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y))) ∧
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b)) ≤ 1 :=
by sorry

end max_value_sqrt_sum_l2351_235140


namespace next_square_number_proof_l2351_235165

/-- The next square number after 4356 composed of four consecutive digits -/
def next_square_number : ℕ := 5476

/-- The square root of the next square number -/
def square_root : ℕ := 74

/-- Predicate to check if a number is composed of four consecutive digits -/
def is_composed_of_four_consecutive_digits (n : ℕ) : Prop :=
  ∃ (a : ℕ), a > 0 ∧ a < 7 ∧
  (n = a * 1000 + (a + 1) * 100 + (a + 2) * 10 + (a + 3) ∨
   n = a * 1000 + (a + 1) * 100 + (a + 3) * 10 + (a + 2) ∨
   n = a * 1000 + (a + 2) * 100 + (a + 1) * 10 + (a + 3) ∨
   n = a * 1000 + (a + 2) * 100 + (a + 3) * 10 + (a + 1) ∨
   n = a * 1000 + (a + 3) * 100 + (a + 1) * 10 + (a + 2) ∨
   n = a * 1000 + (a + 3) * 100 + (a + 2) * 10 + (a + 1))

theorem next_square_number_proof :
  next_square_number = square_root ^ 2 ∧
  is_composed_of_four_consecutive_digits next_square_number ∧
  ∀ (n : ℕ), 4356 < n ∧ n < next_square_number →
    ¬(∃ (m : ℕ), n = m ^ 2 ∧ is_composed_of_four_consecutive_digits n) :=
by sorry

end next_square_number_proof_l2351_235165


namespace line_b_production_l2351_235168

/-- Represents a production line in a factory -/
inductive ProductionLine
| A
| B
| C

/-- Represents the production of a factory with three production lines -/
structure FactoryProduction where
  total : ℕ
  lines : ProductionLine → ℕ
  sum_eq_total : lines ProductionLine.A + lines ProductionLine.B + lines ProductionLine.C = total
  arithmetic_seq : ∃ d : ℤ, 
    (lines ProductionLine.B : ℤ) - (lines ProductionLine.A : ℤ) = d ∧
    (lines ProductionLine.C : ℤ) - (lines ProductionLine.B : ℤ) = d

/-- The theorem stating the production of line B given the conditions -/
theorem line_b_production (fp : FactoryProduction) 
  (h_total : fp.total = 16800) : 
  fp.lines ProductionLine.B = 5600 := by
  sorry

end line_b_production_l2351_235168


namespace tiles_for_dining_room_l2351_235163

/-- Calculates the number of tiles needed for a rectangular room with a border --/
def tiles_needed (room_length room_width border_width : ℕ) 
  (small_tile_size large_tile_size : ℕ) : ℕ :=
  let border_tiles := 
    2 * (2 * (room_length - 2 * border_width) + 2 * (room_width - 2 * border_width)) + 
    4 * border_width * border_width / (small_tile_size * small_tile_size)
  let inner_area := (room_length - 2 * border_width) * (room_width - 2 * border_width)
  let large_tiles := (inner_area + large_tile_size * large_tile_size - 1) / 
    (large_tile_size * large_tile_size)
  border_tiles + large_tiles

/-- Theorem stating that for the given room dimensions and tile sizes, 
    the total number of tiles needed is 144 --/
theorem tiles_for_dining_room : 
  tiles_needed 20 15 2 1 3 = 144 := by sorry

end tiles_for_dining_room_l2351_235163


namespace correct_observation_value_l2351_235156

/-- Proves that the correct value of a misrecorded observation is 48, given the conditions of the problem. -/
theorem correct_observation_value (n : ℕ) (original_mean corrected_mean wrong_value : ℚ)
  (h_n : n = 50)
  (h_original_mean : original_mean = 32)
  (h_corrected_mean : corrected_mean = 32.5)
  (h_wrong_value : wrong_value = 23) :
  let correct_value := (n : ℚ) * corrected_mean - ((n : ℚ) * original_mean - wrong_value)
  correct_value = 48 := by
  sorry

end correct_observation_value_l2351_235156


namespace exists_a_min_value_3_l2351_235136

open Real

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x - log x

theorem exists_a_min_value_3 :
  ∃ a : ℝ, ∀ x : ℝ, 0 < x → x ≤ exp 1 → g a x ≥ 3 ∧
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ exp 1 ∧ g a x₀ = 3 :=
by sorry

end exists_a_min_value_3_l2351_235136


namespace smallest_coin_count_l2351_235139

def is_valid_coin_combination (dimes quarters : ℕ) : Prop :=
  dimes * 10 + quarters * 25 = 265 ∧ dimes > quarters

def coin_count (dimes quarters : ℕ) : ℕ :=
  dimes + quarters

theorem smallest_coin_count : 
  (∃ d q : ℕ, is_valid_coin_combination d q) ∧ 
  (∀ d q : ℕ, is_valid_coin_combination d q → coin_count d q ≥ 16) ∧
  (∃ d q : ℕ, is_valid_coin_combination d q ∧ coin_count d q = 16) :=
sorry

end smallest_coin_count_l2351_235139


namespace total_volume_is_85_l2351_235124

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The total volume of n cubes, each with side length s -/
def total_volume (n : ℕ) (s : ℝ) : ℝ := n * (cube_volume s)

/-- Carl's cubes -/
def carl_cubes : ℕ := 3
def carl_side_length : ℝ := 3

/-- Kate's cubes -/
def kate_cubes : ℕ := 4
def kate_side_length : ℝ := 1

/-- The theorem stating that the total volume of Carl's and Kate's cubes is 85 -/
theorem total_volume_is_85 : 
  total_volume carl_cubes carl_side_length + total_volume kate_cubes kate_side_length = 85 := by
  sorry

end total_volume_is_85_l2351_235124


namespace smallest_multiple_sixty_four_satisfies_smallest_satisfying_number_l2351_235107

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 450 * x % 640 = 0 → x ≥ 64 := by
  sorry

theorem sixty_four_satisfies : 450 * 64 % 640 = 0 := by
  sorry

theorem smallest_satisfying_number : ∃ x : ℕ, x > 0 ∧ 450 * x % 640 = 0 ∧ ∀ y : ℕ, (y > 0 ∧ 450 * y % 640 = 0) → x ≤ y := by
  sorry

end smallest_multiple_sixty_four_satisfies_smallest_satisfying_number_l2351_235107


namespace complex_fraction_sum_l2351_235103

theorem complex_fraction_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (2 + i) / (1 + i) = a + b * i → a + b = 1 := by
  sorry

end complex_fraction_sum_l2351_235103


namespace circle_problem_l2351_235179

theorem circle_problem (P : ℝ × ℝ) (S : ℝ × ℝ) (k : ℝ) :
  P = (5, 12) →
  S = (0, k) →
  (∃ (O : ℝ × ℝ), O = (0, 0) ∧
    ∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧
      (P.1 - O.1)^2 + (P.2 - O.2)^2 = r₁^2 ∧
      (S.1 - O.1)^2 + (S.2 - O.2)^2 = r₂^2 ∧
      r₁ - r₂ = 4) →
  k = 9 := by
sorry

end circle_problem_l2351_235179


namespace empty_set_proof_l2351_235195

theorem empty_set_proof : {x : ℝ | x^2 - x + 1 = 0} = ∅ := by sorry

end empty_set_proof_l2351_235195


namespace negation_equivalence_l2351_235109

theorem negation_equivalence :
  (¬ ∃ x : ℤ, x^2 + 2*x + 1 ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + 1 > 0) := by
  sorry

end negation_equivalence_l2351_235109


namespace james_age_is_35_l2351_235119

/-- The age James turned when John turned 35 -/
def james_age : ℕ := sorry

/-- John's age when James turned james_age -/
def john_age : ℕ := 35

/-- Tim's current age -/
def tim_age : ℕ := 79

theorem james_age_is_35 : james_age = 35 :=
  by
    have h1 : tim_age = 2 * john_age - 5 := by sorry
    have h2 : james_age = john_age := by sorry
    sorry

#check james_age_is_35

end james_age_is_35_l2351_235119


namespace rates_sum_of_squares_l2351_235174

/-- Represents the rates of biking, jogging, and swimming -/
structure Rates where
  biking : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (r : Rates) : Prop :=
  3 * r.biking + 2 * r.jogging + 4 * r.swimming = 84 ∧
  4 * r.biking + 3 * r.jogging + 2 * r.swimming = 106

/-- The theorem to be proved -/
theorem rates_sum_of_squares (r : Rates) : 
  satisfies_conditions r → r.biking^2 + r.jogging^2 + r.swimming^2 = 1125 := by
  sorry


end rates_sum_of_squares_l2351_235174


namespace soccer_team_boys_l2351_235145

/-- Proves the number of boys on a soccer team given certain conditions -/
theorem soccer_team_boys (total : ℕ) (attendees : ℕ) : 
  total = 30 → 
  attendees = 20 → 
  ∃ (boys girls : ℕ), 
    boys + girls = total ∧ 
    boys + (girls / 3) = attendees ∧
    boys = 15 := by
  sorry

end soccer_team_boys_l2351_235145


namespace unique_solution_power_of_two_l2351_235113

theorem unique_solution_power_of_two (a b m : ℕ) : 
  a > 0 → b > 0 → (a + b^2) * (b + a^2) = 2^m → a = 1 ∧ b = 1 ∧ m = 2 := by
  sorry

end unique_solution_power_of_two_l2351_235113


namespace probability_not_ab_l2351_235159

def num_courses : ℕ := 4
def num_selected : ℕ := 2

def probability_not_selected_together : ℚ :=
  1 - (1 / (num_courses.choose num_selected))

theorem probability_not_ab : probability_not_selected_together = 5/6 := by
  sorry

end probability_not_ab_l2351_235159


namespace power_seven_mod_twelve_l2351_235184

theorem power_seven_mod_twelve : 7^150 % 12 = 1 := by sorry

end power_seven_mod_twelve_l2351_235184


namespace expression_value_at_three_l2351_235144

theorem expression_value_at_three : 
  let x : ℝ := 3
  x^6 - x^3 - 6*x = 684 := by sorry

end expression_value_at_three_l2351_235144


namespace S_is_infinite_l2351_235155

-- Define the set of points that satisfy the conditions
def S : Set (ℚ × ℚ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 5}

-- Theorem: The set S is infinite
theorem S_is_infinite : Set.Infinite S := by
  sorry

end S_is_infinite_l2351_235155


namespace sandy_molly_age_ratio_l2351_235158

/-- The ratio of ages between two people -/
def age_ratio (age1 : ℕ) (age2 : ℕ) : ℚ := age1 / age2

/-- Sandy's age -/
def sandy_age : ℕ := 49

/-- Age difference between Molly and Sandy -/
def age_difference : ℕ := 14

/-- Molly's age -/
def molly_age : ℕ := sandy_age + age_difference

theorem sandy_molly_age_ratio :
  age_ratio sandy_age molly_age = 7 / 9 := by
  sorry

end sandy_molly_age_ratio_l2351_235158


namespace triangle_side_length_l2351_235188

/-- Given a triangle ABC with the specified properties, prove that AC = 5√3 -/
theorem triangle_side_length (A B C : ℝ) (BC : ℝ) :
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  A + B + C = π →
  2 * Real.sin (A - B) + Real.cos (B + C) = 2 →
  BC = 5 →
  ∃ (AC : ℝ), AC = 5 * Real.sqrt 3 := by
  sorry

end triangle_side_length_l2351_235188


namespace circle_center_radius_sum_l2351_235185

/-- Given a circle C with equation x^2 + 8x + y^2 - 2y = -4, 
    prove that u + v + s = -3 + √13, where (u,v) is the center and s is the radius -/
theorem circle_center_radius_sum (x y : ℝ) : 
  (∃ (u v s : ℝ), x^2 + 8*x + y^2 - 2*y = -4 ∧ 
  (x - u)^2 + (y - v)^2 = s^2) → 
  (∃ (u v s : ℝ), x^2 + 8*x + y^2 - 2*y = -4 ∧ 
  (x - u)^2 + (y - v)^2 = s^2 ∧ 
  u + v + s = -3 + Real.sqrt 13) :=
by sorry

end circle_center_radius_sum_l2351_235185


namespace dog_tail_length_l2351_235112

theorem dog_tail_length (body_length : ℝ) (head_length : ℝ) (tail_length : ℝ) 
  (overall_length : ℝ) (width : ℝ) (height : ℝ) :
  tail_length = body_length / 2 →
  head_length = body_length / 6 →
  height = 1.5 * width →
  overall_length = 30 →
  width = 12 →
  overall_length = body_length + head_length + tail_length →
  tail_length = 15 := by
  sorry

end dog_tail_length_l2351_235112


namespace larger_number_problem_l2351_235146

theorem larger_number_problem (x y : ℕ) 
  (h1 : y - x = 1500)
  (h2 : y = 6 * x + 15) : 
  y = 1797 := by
sorry

end larger_number_problem_l2351_235146


namespace perimeter_710_implies_n_66_l2351_235134

/-- Represents the perimeter of the nth figure in the sequence -/
def perimeter (n : ℕ) : ℕ := 60 + (n - 1) * 10

/-- Theorem stating that if the perimeter of the nth figure is 710 cm, then n is 66 -/
theorem perimeter_710_implies_n_66 : ∃ n : ℕ, perimeter n = 710 ∧ n = 66 := by
  sorry

end perimeter_710_implies_n_66_l2351_235134


namespace factorial_expression_equals_2015_l2351_235196

-- Define factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_expression_equals_2015 : 
  (factorial (factorial 2014 - 1) * factorial 2015) / factorial (factorial 2014) = 2015 := by
  sorry

end factorial_expression_equals_2015_l2351_235196


namespace number_pattern_l2351_235111

/-- Represents a number as a string of consecutive '1' digits -/
def ones (n : ℕ) : ℕ :=
  (10 ^ n - 1) / 9

/-- The main theorem to be proved -/
theorem number_pattern (n : ℕ) (h : n ≤ 123456) :
  n * 9 + (n + 1) = ones (n + 1) :=
sorry

end number_pattern_l2351_235111


namespace intersection_complement_equality_l2351_235143

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 4}

-- Define set N
def N : Finset Nat := {1, 3, 5}

-- Theorem statement
theorem intersection_complement_equality :
  N ∩ (U \ M) = {3, 5} := by sorry

end intersection_complement_equality_l2351_235143


namespace regular_polygon_945_diagonals_has_45_sides_l2351_235123

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 945 diagonals has 45 sides -/
theorem regular_polygon_945_diagonals_has_45_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 945 → n = 45 := by
sorry

end regular_polygon_945_diagonals_has_45_sides_l2351_235123


namespace inverse_proportion_k_negative_l2351_235114

/-- Given two points A(-2, y₁) and B(5, y₂) on the graph of y = k/x (k ≠ 0),
    if y₁ > y₂, then k < 0. -/
theorem inverse_proportion_k_negative
  (k : ℝ) (y₁ y₂ : ℝ)
  (hk : k ≠ 0)
  (hA : y₁ = k / (-2))
  (hB : y₂ = k / 5)
  (hy : y₁ > y₂) :
  k < 0 :=
sorry

end inverse_proportion_k_negative_l2351_235114


namespace equal_winning_chance_l2351_235125

/-- Represents a lottery ticket -/
structure LotteryTicket where
  id : ℕ

/-- Represents a lottery -/
structure Lottery where
  winningProbability : ℝ
  totalTickets : ℕ

/-- The probability of a ticket winning is equal to the lottery's winning probability -/
def ticketWinningProbability (lottery : Lottery) (ticket : LotteryTicket) : ℝ :=
  lottery.winningProbability

theorem equal_winning_chance (lottery : Lottery) 
    (h1 : lottery.winningProbability = 0.002)
    (h2 : lottery.totalTickets = 1000) :
    ∀ (t1 t2 : LotteryTicket), ticketWinningProbability lottery t1 = ticketWinningProbability lottery t2 :=
  sorry


end equal_winning_chance_l2351_235125


namespace ten_men_and_boys_complete_in_ten_days_l2351_235194

/-- The number of days it takes for a group of men and boys to complete a work -/
def daysToComplete (numMen numBoys : ℕ) : ℚ :=
  10 / ((2 * numMen : ℚ) / 3 + (numBoys : ℚ) / 3)

/-- Theorem stating that 10 men and 10 boys will complete the work in 10 days -/
theorem ten_men_and_boys_complete_in_ten_days :
  daysToComplete 10 10 = 10 := by sorry

end ten_men_and_boys_complete_in_ten_days_l2351_235194


namespace square_area_from_diagonal_l2351_235126

theorem square_area_from_diagonal (d : ℝ) (h : d = 3.8) :
  (d^2 / 2) = 7.22 := by sorry

end square_area_from_diagonal_l2351_235126


namespace right_triangle_shorter_leg_l2351_235177

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 25 →           -- Hypotenuse is 25
  a ≤ b →            -- a is the shorter leg
  (a = 7 ∨ a = 20) := by
sorry

end right_triangle_shorter_leg_l2351_235177


namespace only_one_is_ultra_prime_l2351_235133

-- Define f(n) as the sum of all divisors of n
def f (n : ℕ) : ℕ := sorry

-- Define g(n) = n + f(n)
def g (n : ℕ) : ℕ := n + f n

-- Define ultra-prime
def is_ultra_prime (n : ℕ) : Prop := f (g n) = 2 * n + 3

-- Theorem statement
theorem only_one_is_ultra_prime :
  ∃! (n : ℕ), n < 100 ∧ is_ultra_prime n :=
sorry

end only_one_is_ultra_prime_l2351_235133


namespace batsman_running_percentage_l2351_235148

theorem batsman_running_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : 
  total_runs = 120 →
  boundaries = 6 →
  sixes = 4 →
  (total_runs - (boundaries * 4 + sixes * 6)) / total_runs * 100 = 60 := by
sorry

end batsman_running_percentage_l2351_235148


namespace train_passing_pole_time_l2351_235122

/-- Proves that a train of length 240 m takes 24 seconds to pass a pole, given that it takes 89 seconds to pass a 650 m platform -/
theorem train_passing_pole_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_pass_platform : ℝ) 
  (h1 : train_length = 240)
  (h2 : platform_length = 650)
  (h3 : time_to_pass_platform = 89) :
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 24 := by
  sorry

end train_passing_pole_time_l2351_235122


namespace complex_exp_210_deg_60th_power_l2351_235127

theorem complex_exp_210_deg_60th_power : 
  (Complex.exp (210 * π / 180 * I)) ^ 60 = 1 := by sorry

end complex_exp_210_deg_60th_power_l2351_235127


namespace special_rectangle_difference_l2351_235173

/-- A rectangle with perimeter 4r and diagonal k times the length of one side -/
structure SpecialRectangle (r k : ℝ) where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 2 * r
  diagonal_eq : length ^ 2 + width ^ 2 = (k * length) ^ 2

/-- The absolute difference between length and width is k times the length -/
theorem special_rectangle_difference (r k : ℝ) (rect : SpecialRectangle r k) :
  |rect.length - rect.width| = k * rect.length :=
sorry

end special_rectangle_difference_l2351_235173


namespace percentage_reduction_price_increase_for_target_profit_price_increase_for_max_profit_maximum_daily_profit_l2351_235142

-- Define the original price and final price
def original_price : ℝ := 50
def final_price : ℝ := 32

-- Define the profit per kilogram and initial daily sales
def profit_per_kg : ℝ := 10
def initial_daily_sales : ℝ := 500

-- Define the reduction in sales per yuan increase
def sales_reduction_per_yuan : ℝ := 20

-- Define the target daily profit
def target_daily_profit : ℝ := 6000

-- Part 1: Percentage reduction
theorem percentage_reduction : 
  ∃ x : ℝ, x > 0 ∧ x < 1 ∧ original_price * (1 - x)^2 = final_price ∧ x = 0.2 := by sorry

-- Part 2: Price increase for target profit
theorem price_increase_for_target_profit :
  ∃ x : ℝ, x > 0 ∧ (profit_per_kg + x) * (initial_daily_sales - sales_reduction_per_yuan * x) = target_daily_profit ∧
  (∀ y : ℝ, y > 0 ∧ (profit_per_kg + y) * (initial_daily_sales - sales_reduction_per_yuan * y) = target_daily_profit → x ≤ y) ∧
  x = 5 := by sorry

-- Part 3: Price increase for maximum profit
def profit_function (x : ℝ) : ℝ := (profit_per_kg + x) * (initial_daily_sales - sales_reduction_per_yuan * x)

theorem price_increase_for_max_profit :
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ x = 7.5 := by sorry

-- Part 4: Maximum daily profit
theorem maximum_daily_profit :
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ profit_function x = 6125 := by sorry

end percentage_reduction_price_increase_for_target_profit_price_increase_for_max_profit_maximum_daily_profit_l2351_235142


namespace john_vacation_expenses_l2351_235170

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after buying a ticket -/
def remaining_money (savings : ℕ) (ticket_cost : ℕ) : ℕ :=
  base8_to_base10 savings - ticket_cost

theorem john_vacation_expenses :
  remaining_money 5373 1500 = 1311 := by sorry

end john_vacation_expenses_l2351_235170


namespace min_value_of_some_expression_l2351_235147

-- Define the expression as a function of x and some_expression
def f (x : ℝ) (some_expression : ℝ) : ℝ :=
  |x - 4| + |x + 6| + |some_expression|

-- State the theorem
theorem min_value_of_some_expression :
  (∃ (some_expression : ℝ), ∀ (x : ℝ), f x some_expression ≥ 11) →
  (∃ (some_expression : ℝ), (∀ (x : ℝ), f x some_expression ≥ 11) ∧ |some_expression| = 1) :=
by sorry

end min_value_of_some_expression_l2351_235147


namespace f_bound_l2351_235137

/-- The function f(x) = (e^x - 1) / x -/
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / x

theorem f_bound (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, 0 < |x| ∧ |x| < Real.log (1 + a) → |f x - 1| < a :=
sorry

end f_bound_l2351_235137


namespace david_pushups_count_l2351_235105

def zachary_pushups : ℕ := 35

def david_pushups : ℕ := zachary_pushups + 9

theorem david_pushups_count : david_pushups = 44 := by sorry

end david_pushups_count_l2351_235105


namespace coin_probability_l2351_235118

theorem coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1/2) : 
  (Nat.choose 6 3 : ℝ) * p^3 * (1-p)^3 = 1/20 → p = 1/400 := by
  sorry

end coin_probability_l2351_235118


namespace second_term_of_geometric_series_l2351_235193

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    the second term of the sequence is 3. -/
theorem second_term_of_geometric_series (a : ℝ) :
  (∑' n, a * (1/4)^n : ℝ) = 16 →
  a * (1/4) = 3 := by sorry

end second_term_of_geometric_series_l2351_235193


namespace larger_number_problem_l2351_235135

theorem larger_number_problem (x y : ℕ) : 
  x * y = 30 → x + y = 13 → max x y = 10 := by
  sorry

end larger_number_problem_l2351_235135


namespace isosceles_triangle_perimeter_l2351_235166

/-- Represents the roots of the quadratic equation x^2 - 8x + 15 = 0 --/
def roots : Set ℝ := {x : ℝ | x^2 - 8*x + 15 = 0}

/-- Represents an isosceles triangle with side lengths from the roots --/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  h_roots : side1 ∈ roots ∧ side2 ∈ roots
  h_isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base

/-- The perimeter of an isosceles triangle --/
def perimeter (t : IsoscelesTriangle) : ℝ := t.side1 + t.side2 + t.base

/-- Theorem stating that the perimeter of the isosceles triangle is either 11 or 13 --/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  perimeter t = 11 ∨ perimeter t = 13 := by
  sorry

end isosceles_triangle_perimeter_l2351_235166


namespace square_sum_equality_l2351_235120

theorem square_sum_equality : 107 * 107 + 93 * 93 = 20098 := by
  sorry

end square_sum_equality_l2351_235120


namespace parallelogram_height_l2351_235182

/-- Given a parallelogram with sides 20 feet and 60 feet, and height 55 feet
    perpendicular to the 20-foot side, prove that the height perpendicular
    to the 60-foot side is 1100/60 feet. -/
theorem parallelogram_height (a b h : ℝ) (ha : a = 20) (hb : b = 60) (hh : h = 55) :
  a * h / b = 1100 / 60 := by
  sorry

end parallelogram_height_l2351_235182


namespace polynomial_roots_sum_l2351_235164

theorem polynomial_roots_sum (c d : ℝ) : 
  c^2 - 6*c + 10 = 0 ∧ d^2 - 6*d + 10 = 0 → c^3 + c^5*d^3 + c^3*d^5 + d^3 = 16156 := by
  sorry

end polynomial_roots_sum_l2351_235164


namespace min_value_expression_l2351_235191

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 4 / y) ≥ 9 := by
sorry

end min_value_expression_l2351_235191


namespace no_solution_absolute_value_equation_l2351_235115

theorem no_solution_absolute_value_equation :
  ¬∃ (x : ℝ), |2*x - 5| = 3*x + 1 := by
  sorry

end no_solution_absolute_value_equation_l2351_235115


namespace quadratic_roots_sum_l2351_235162

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + 2*m - 2022 = 0) → 
  (n^2 + 2*n - 2022 = 0) → 
  (m^2 + 3*m + n = 2020) := by
  sorry

end quadratic_roots_sum_l2351_235162


namespace parabola_axis_of_symmetry_l2351_235138

/-- A parabola passing through two points with the same y-coordinate -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  n : ℝ

/-- The x-coordinate of the axis of symmetry of a parabola -/
def axisOfSymmetry (p : Parabola) : ℝ := 2

/-- Theorem: The axis of symmetry of a parabola passing through (1,n) and (3,n) is x = 2 -/
theorem parabola_axis_of_symmetry (p : Parabola) : 
  p.n = p.a * 1^2 + p.b * 1 + p.c ∧ 
  p.n = p.a * 3^2 + p.b * 3 + p.c → 
  axisOfSymmetry p = 2 := by
  sorry

end parabola_axis_of_symmetry_l2351_235138


namespace birds_to_asia_count_l2351_235186

/-- The number of bird families that flew to Asia -/
def birds_to_asia (initial : ℕ) (to_africa : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_africa - remaining

/-- Theorem stating that 37 bird families flew to Asia -/
theorem birds_to_asia_count : birds_to_asia 85 23 25 = 37 := by
  sorry

end birds_to_asia_count_l2351_235186


namespace algebraic_expression_value_l2351_235154

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2023*a - 1 = 0) :
  a*(a+1)*(a-1) + 2023*a^2 + 1 = 1 := by
sorry

end algebraic_expression_value_l2351_235154


namespace cubic_function_property_l2351_235101

/-- Given a cubic function f(x) = ax³ + bx + 8, if f(-2) = 10, then f(2) = 6 -/
theorem cubic_function_property (a b : ℝ) : 
  let f := λ x : ℝ => a * x^3 + b * x + 8
  f (-2) = 10 → f 2 = 6 := by
sorry

end cubic_function_property_l2351_235101


namespace fayes_math_problems_l2351_235198

theorem fayes_math_problems :
  ∀ (total_problems math_problems science_problems finished_problems remaining_problems : ℕ),
    science_problems = 9 →
    finished_problems = 40 →
    remaining_problems = 15 →
    total_problems = math_problems + science_problems →
    total_problems = finished_problems + remaining_problems →
    math_problems = 46 := by
  sorry

end fayes_math_problems_l2351_235198


namespace angle_terminal_side_l2351_235117

theorem angle_terminal_side (θ : Real) (a : Real) : 
  (2 * Real.sin (π / 8) ^ 2 - 1, a) ∈ Set.range (λ t : Real × Real => (t.1 * Real.cos θ - t.2 * Real.sin θ, t.1 * Real.sin θ + t.2 * Real.cos θ)) ∧ 
  Real.sin θ = 2 * Real.sqrt 3 * Real.sin (13 * π / 12) * Real.cos (π / 12) →
  a = - Real.sqrt 6 / 2 := by
sorry

end angle_terminal_side_l2351_235117


namespace batsman_average_is_60_l2351_235151

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  total_innings : ℕ
  highest_score : ℕ
  lowest_score : ℕ
  average_excluding_extremes : ℚ

/-- Calculate the batting average given the batsman's statistics -/
def batting_average (stats : BatsmanStats) : ℚ :=
  let total_runs := (stats.total_innings - 2) * stats.average_excluding_extremes + stats.highest_score + stats.lowest_score
  total_runs / stats.total_innings

theorem batsman_average_is_60 (stats : BatsmanStats) :
  stats.total_innings = 46 ∧
  stats.highest_score - stats.lowest_score = 140 ∧
  stats.average_excluding_extremes = 58 ∧
  stats.highest_score = 174 →
  batting_average stats = 60 := by
  sorry

#eval batting_average {
  total_innings := 46,
  highest_score := 174,
  lowest_score := 34,
  average_excluding_extremes := 58
}

end batsman_average_is_60_l2351_235151


namespace coordinate_change_l2351_235192

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors a, b, c
variable (a b c : V)

-- Define that {a, b, c} is a basis
variable (h₁ : LinearIndependent ℝ ![a, b, c])
variable (h₂ : Submodule.span ℝ {a, b, c} = ⊤)

-- Define that {a+b, a-b, c} is also a basis
variable (h₃ : LinearIndependent ℝ ![a + b, a - b, c])
variable (h₄ : Submodule.span ℝ {a + b, a - b, c} = ⊤)

-- Define the vector p
variable (p : V)

-- State the theorem
theorem coordinate_change (hp : p = a - 2 • b + 3 • c) :
  p = (-1/2 : ℝ) • (a + b) + (3/2 : ℝ) • (a - b) + 3 • c := by sorry

end coordinate_change_l2351_235192


namespace sangita_flying_months_l2351_235171

/-- Calculates the number of months needed to complete flying hours for a pilot certificate. -/
def months_to_complete_flying (total_required : ℕ) (day_completed : ℕ) (night_completed : ℕ) (cross_country_completed : ℕ) (monthly_goal : ℕ) : ℕ :=
  let total_completed := day_completed + night_completed + cross_country_completed
  let remaining_hours := total_required - total_completed
  (remaining_hours + monthly_goal - 1) / monthly_goal

/-- Proves that Sangita needs 6 months to complete her flying hours. -/
theorem sangita_flying_months :
  months_to_complete_flying 1500 50 9 121 220 = 6 := by
  sorry

end sangita_flying_months_l2351_235171


namespace quadratic_one_root_l2351_235141

/-- A quadratic function f(x) = mx^2 - 4x + 1 has exactly one root if and only if m ≤ 4 -/
theorem quadratic_one_root (m : ℝ) :
  (∃! x, m * x^2 - 4 * x + 1 = 0) ↔ m ≤ 4 := by
  sorry

end quadratic_one_root_l2351_235141


namespace line_division_theorem_l2351_235104

/-- A line in the plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three lines divide the plane into six parts -/
def divides_into_six_parts (l₁ l₂ l₃ : Line) : Prop :=
  sorry

/-- The set of k values that satisfy the condition -/
def k_values : Set ℝ :=
  {0, -1, -2}

theorem line_division_theorem (k : ℝ) :
  let l₁ : Line := ⟨1, -2, 1⟩  -- x - 2y + 1 = 0
  let l₂ : Line := ⟨1, 0, -1⟩ -- x - 1 = 0
  let l₃ : Line := ⟨1, k, 0⟩  -- x + ky = 0
  divides_into_six_parts l₁ l₂ l₃ → k ∈ k_values := by
  sorry

end line_division_theorem_l2351_235104
