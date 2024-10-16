import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_value_l1393_139381

open Real

/-- Prove that in triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*sin(B) = √2*sin(C), cos(C) = 1/3, and the area of the triangle is 4, then c = 6. -/
theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) :
  a * sin B = sqrt 2 * sin C →
  cos C = 1 / 3 →
  1 / 2 * a * b * sin C = 4 →
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l1393_139381


namespace NUMINAMATH_CALUDE_bobs_current_time_l1393_139329

/-- Given that Bob's sister runs a mile in 320 seconds, and Bob needs to improve his time by 50% to match his sister's time, prove that Bob's current time is 480 seconds. -/
theorem bobs_current_time (sister_time : ℝ) (improvement_rate : ℝ) (bob_time : ℝ) 
  (h1 : sister_time = 320)
  (h2 : improvement_rate = 0.5)
  (h3 : bob_time = sister_time + sister_time * improvement_rate) :
  bob_time = 480 := by
  sorry

end NUMINAMATH_CALUDE_bobs_current_time_l1393_139329


namespace NUMINAMATH_CALUDE_black_pens_count_l1393_139301

theorem black_pens_count (total_pens blue_pens : ℕ) 
  (h1 : total_pens = 8)
  (h2 : blue_pens = 4) :
  total_pens - blue_pens = 4 := by
  sorry

end NUMINAMATH_CALUDE_black_pens_count_l1393_139301


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1393_139324

-- Problem 1
theorem factorization_problem_1 (a b x : ℝ) :
  x^2 * (a - b) + 4 * (b - a) = (a - b) * (x + 2) * (x - 2) := by sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  -a^3 + 6 * a^2 * b - 9 * a * b^2 = -a * (a - 3 * b)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1393_139324


namespace NUMINAMATH_CALUDE_fraction_simplification_l1393_139392

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hab : a ≠ b) :
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1393_139392


namespace NUMINAMATH_CALUDE_sin_cos_square_identity_l1393_139300

theorem sin_cos_square_identity (α : ℝ) : (Real.sin α + Real.cos α)^2 = 1 + Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_square_identity_l1393_139300


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1393_139394

theorem polynomial_remainder (x : ℝ) : 
  let Q := fun (x : ℝ) => 8*x^4 - 18*x^3 - 6*x^2 + 4*x - 30
  let divisor := fun (x : ℝ) => 2*x - 8
  Q 4 = 786 ∧ (∃ P : ℝ → ℝ, ∀ x, Q x = P x * divisor x + 786) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1393_139394


namespace NUMINAMATH_CALUDE_system_solution_l1393_139361

theorem system_solution :
  ∃ (a b c d : ℝ),
    (a + c = 4 ∧
     a * c + b + d = 6 ∧
     a * d + b * c = 5 ∧
     b * d = 2) ∧
    ((a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 3 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1393_139361


namespace NUMINAMATH_CALUDE_sum_of_specific_geometric_series_l1393_139375

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_specific_geometric_series :
  let a : ℚ := 1/2
  let r : ℚ := 1/2
  let n : ℕ := 7
  geometric_series_sum a r n = 127/128 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_geometric_series_l1393_139375


namespace NUMINAMATH_CALUDE_prob_good_friends_is_one_fourth_l1393_139340

/-- The number of balls in the pocket -/
def num_balls : ℕ := 4

/-- The set of possible ball numbers -/
def ball_numbers : Finset ℕ := Finset.range num_balls

/-- The probability space of drawing two balls with replacement -/
def draw_space : Finset (ℕ × ℕ) := ball_numbers.product ball_numbers

/-- The event of drawing the same number (becoming "good friends") -/
def good_friends : Finset (ℕ × ℕ) := 
  draw_space.filter (fun p => p.1 = p.2)

/-- The probability of becoming "good friends" -/
def prob_good_friends : ℚ :=
  good_friends.card / draw_space.card

theorem prob_good_friends_is_one_fourth : 
  prob_good_friends = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_good_friends_is_one_fourth_l1393_139340


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1393_139348

/-- The common difference of an arithmetic sequence with general term a_n = 5 - 4n is -4. -/
theorem arithmetic_sequence_common_difference :
  ∀ (a : ℕ → ℝ), (∀ n, a n = 5 - 4 * n) →
  ∃ d : ℝ, ∀ n, a (n + 1) - a n = d ∧ d = -4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1393_139348


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l1393_139316

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (2 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (3 ∣ n) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (2 ∣ m) ∧ (5 ∣ m) ∧ (7 ∣ m) ∧ (3 ∣ m) → m ≥ n) ∧
  n = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l1393_139316


namespace NUMINAMATH_CALUDE_dog_water_consumption_l1393_139332

/-- Calculates the water needed for a dog during a hike given the total water capacity,
    human water consumption rate, and duration of the hike. -/
theorem dog_water_consumption
  (total_water : ℝ)
  (human_rate : ℝ)
  (duration : ℝ)
  (h1 : total_water = 4.8 * 1000) -- 4.8 L converted to ml
  (h2 : human_rate = 800)
  (h3 : duration = 4) :
  (total_water - human_rate * duration) / duration = 400 :=
by sorry

end NUMINAMATH_CALUDE_dog_water_consumption_l1393_139332


namespace NUMINAMATH_CALUDE_ceiling_sqrt_sum_times_two_l1393_139317

theorem ceiling_sqrt_sum_times_two : 
  2 * (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉) = 54 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_sum_times_two_l1393_139317


namespace NUMINAMATH_CALUDE_painted_cubes_l1393_139360

theorem painted_cubes (n : ℕ) (h : n = 5) : 
  n^3 - (n - 2)^3 = 98 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_l1393_139360


namespace NUMINAMATH_CALUDE_pair_op_theorem_l1393_139342

/-- Definition of the custom operation ⊗ for pairs of real numbers -/
def pair_op (a b c d : ℝ) : ℝ × ℝ := (a * c - b * d, a * d + b * c)

/-- Theorem stating that if (1, 2) ⊗ (p, q) = (5, 0), then p + q equals some real number -/
theorem pair_op_theorem (p q : ℝ) :
  pair_op 1 2 p q = (5, 0) → ∃ r : ℝ, p + q = r := by
  sorry

end NUMINAMATH_CALUDE_pair_op_theorem_l1393_139342


namespace NUMINAMATH_CALUDE_total_weight_of_diamonds_and_jades_l1393_139377

/-- Given that 5 diamonds weigh 100 g and a jade is 10 g heavier than a diamond,
    prove that the total weight of 4 diamonds and 2 jades is 140 g. -/
theorem total_weight_of_diamonds_and_jades :
  let diamond_weight : ℚ := 100 / 5
  let jade_weight : ℚ := diamond_weight + 10
  4 * diamond_weight + 2 * jade_weight = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_diamonds_and_jades_l1393_139377


namespace NUMINAMATH_CALUDE_alloy_mixture_problem_l1393_139374

/-- Proves that the amount of the first alloy used is 15 kg given the conditions of the problem -/
theorem alloy_mixture_problem (x : ℝ) : 
  (0.10 * x + 0.08 * 35 = 0.086 * (x + 35)) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_problem_l1393_139374


namespace NUMINAMATH_CALUDE_hexagon_circle_area_ratio_l1393_139341

/-- Given a regular hexagon and a circle with equal perimeter/circumference,
    the ratio of the area of the hexagon to the area of the circle is π√3/6 -/
theorem hexagon_circle_area_ratio :
  ∀ (s r : ℝ),
  s > 0 → r > 0 →
  6 * s = 2 * Real.pi * r →
  (3 * Real.sqrt 3 / 2 * s^2) / (Real.pi * r^2) = Real.pi * Real.sqrt 3 / 6 := by
sorry

end NUMINAMATH_CALUDE_hexagon_circle_area_ratio_l1393_139341


namespace NUMINAMATH_CALUDE_trumpet_cost_l1393_139336

/-- The cost of a trumpet, given the total amount spent and the cost of a song book. -/
theorem trumpet_cost (total_spent song_book_cost : ℚ) 
  (h1 : total_spent = 151)
  (h2 : song_book_cost = 5.84) :
  total_spent - song_book_cost = 145.16 := by
  sorry

end NUMINAMATH_CALUDE_trumpet_cost_l1393_139336


namespace NUMINAMATH_CALUDE_xyz_value_l1393_139359

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1393_139359


namespace NUMINAMATH_CALUDE_curve_intersects_median_l1393_139388

theorem curve_intersects_median (a b c : ℝ) (h : a + c - 2*b ≠ 0) :
  ∃! p : ℂ, 
    (∀ t : ℝ, p ≠ Complex.I * a * (Real.cos t)^4 + 2 * (1/2 + Complex.I * b) * (Real.cos t)^2 * (Real.sin t)^2 + (1 + Complex.I * c) * (Real.sin t)^4) ∧
    (p.re = 1/2) ∧
    (p.im = (a + c + 2*b) / 4) ∧
    (∃ k : ℝ, p.im - (a + b)/2 = (c - a) * (p.re - 1/4) + k * ((3/4 - 1/4) * Complex.I - ((b + c)/2 - (a + b)/2))) := by
  sorry

end NUMINAMATH_CALUDE_curve_intersects_median_l1393_139388


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l1393_139357

theorem baker_remaining_cakes (total_cakes friend_bought : ℕ) 
  (h1 : total_cakes = 155)
  (h2 : friend_bought = 140) :
  total_cakes - friend_bought = 15 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l1393_139357


namespace NUMINAMATH_CALUDE_smallest_school_size_l1393_139339

theorem smallest_school_size : ∃ n : ℕ, n > 0 ∧ n % 4 = 0 ∧ (n / 4) % 10 = 0 ∧
  (∃ y z : ℕ, y > 0 ∧ z > 0 ∧ 2 * y = 3 * z ∧ y + z - (n / 40) = n / 4) ∧
  ∀ m : ℕ, m > 0 → m % 4 = 0 → (m / 4) % 10 = 0 →
    (∃ y z : ℕ, y > 0 ∧ z > 0 ∧ 2 * y = 3 * z ∧ y + z - (m / 40) = m / 4) →
    m ≥ 200 :=
by sorry

end NUMINAMATH_CALUDE_smallest_school_size_l1393_139339


namespace NUMINAMATH_CALUDE_billy_laundry_loads_l1393_139387

/-- Represents the time taken for each chore in minutes -/
structure ChoreTime where
  sweeping : ℕ  -- time to sweep one room
  dishwashing : ℕ  -- time to wash one dish
  laundry : ℕ  -- time to do one load of laundry

/-- Represents the chores done by each child -/
structure Chores where
  rooms_swept : ℕ
  dishes_washed : ℕ
  laundry_loads : ℕ

def total_time (ct : ChoreTime) (c : Chores) : ℕ :=
  ct.sweeping * c.rooms_swept + ct.dishwashing * c.dishes_washed + ct.laundry * c.laundry_loads

theorem billy_laundry_loads (ct : ChoreTime) (anna billy : Chores) :
  ct.sweeping = 3 →
  ct.dishwashing = 2 →
  ct.laundry = 9 →
  anna.rooms_swept = 10 →
  billy.dishes_washed = 6 →
  anna.dishes_washed = 0 →
  anna.laundry_loads = 0 →
  billy.rooms_swept = 0 →
  total_time ct anna = total_time ct billy →
  billy.laundry_loads = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_laundry_loads_l1393_139387


namespace NUMINAMATH_CALUDE_eighth_group_frequency_l1393_139334

theorem eighth_group_frequency 
  (total_sample : ℕ) 
  (num_groups : ℕ) 
  (freq_1 freq_2 freq_3 freq_4 : ℕ) 
  (sum_freq_5_to_7 : ℚ) :
  total_sample = 100 →
  num_groups = 8 →
  freq_1 = 15 →
  freq_2 = 17 →
  freq_3 = 11 →
  freq_4 = 13 →
  sum_freq_5_to_7 = 32 / 100 →
  (freq_1 + freq_2 + freq_3 + freq_4 + (sum_freq_5_to_7 * total_sample).num + 
    (total_sample - freq_1 - freq_2 - freq_3 - freq_4 - (sum_freq_5_to_7 * total_sample).num)) / total_sample = 1 →
  (total_sample - freq_1 - freq_2 - freq_3 - freq_4 - (sum_freq_5_to_7 * total_sample).num) / total_sample = 12 / 100 :=
by sorry

end NUMINAMATH_CALUDE_eighth_group_frequency_l1393_139334


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1393_139364

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating the total wet surface area of a specific cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 6 1.25 = 83 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1393_139364


namespace NUMINAMATH_CALUDE_matthew_stocks_solution_l1393_139307

def matthew_stocks (expensive_stock_price : ℕ) (cheap_stock_price : ℕ) (cheap_stock_shares : ℕ) (total_assets : ℕ) (expensive_stock_shares : ℕ) : Prop :=
  expensive_stock_price = 2 * cheap_stock_price ∧
  cheap_stock_shares = 26 ∧
  expensive_stock_price = 78 ∧
  total_assets = 2106 ∧
  expensive_stock_shares * expensive_stock_price + cheap_stock_shares * cheap_stock_price = total_assets

theorem matthew_stocks_solution :
  ∃ (expensive_stock_price cheap_stock_price cheap_stock_shares total_assets expensive_stock_shares : ℕ),
    matthew_stocks expensive_stock_price cheap_stock_price cheap_stock_shares total_assets expensive_stock_shares ∧
    expensive_stock_shares = 14 := by
  sorry

end NUMINAMATH_CALUDE_matthew_stocks_solution_l1393_139307


namespace NUMINAMATH_CALUDE_angle_is_120_degrees_l1393_139384

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_120_degrees (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 4)
  (h2 : b = (-1, 0))
  (h3 : (a.1 + 2 * b.1) * b.1 + (a.2 + 2 * b.2) * b.2 = 0) :
  angle_between_vectors a b = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_is_120_degrees_l1393_139384


namespace NUMINAMATH_CALUDE_subtract_square_thirty_l1393_139369

theorem subtract_square_thirty : 30 - 5^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_square_thirty_l1393_139369


namespace NUMINAMATH_CALUDE_joint_business_profit_l1393_139310

/-- Represents the profit distribution in a joint business venture -/
structure JointBusiness where
  a_investment : ℝ
  b_investment : ℝ
  a_period : ℝ
  b_period : ℝ
  b_profit : ℝ

/-- Calculates the total profit given the conditions of the joint business -/
def total_profit (jb : JointBusiness) : ℝ :=
  7 * jb.b_profit

/-- Theorem stating that under the given conditions, the total profit is 28000 -/
theorem joint_business_profit (jb : JointBusiness) 
  (h1 : jb.a_investment = 3 * jb.b_investment)
  (h2 : jb.a_period = 2 * jb.b_period)
  (h3 : jb.b_profit = 4000) :
  total_profit jb = 28000 := by
  sorry

#eval total_profit { a_investment := 3, b_investment := 1, a_period := 2, b_period := 1, b_profit := 4000 }

end NUMINAMATH_CALUDE_joint_business_profit_l1393_139310


namespace NUMINAMATH_CALUDE_third_player_games_l1393_139353

/-- Represents a table tennis game with three players -/
structure TableTennisGame where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The rules and conditions of the game -/
def valid_game (g : TableTennisGame) : Prop :=
  g.total_games = g.player1_games ∧
  g.total_games = g.player2_games + g.player3_games ∧
  g.player1_games = 21 ∧
  g.player2_games = 10

/-- Theorem stating that under the given conditions, the third player must have played 11 games -/
theorem third_player_games (g : TableTennisGame) (h : valid_game g) : 
  g.player3_games = 11 := by
  sorry

end NUMINAMATH_CALUDE_third_player_games_l1393_139353


namespace NUMINAMATH_CALUDE_polygon_diagonals_l1393_139337

theorem polygon_diagonals (n : ℕ) (h : n > 2) :
  (360 / (360 / n) : ℚ) - 3 = 9 :=
sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l1393_139337


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l1393_139312

theorem sqrt_difference_approximation : 
  |Real.sqrt 75 - Real.sqrt 72 - 0.17| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l1393_139312


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l1393_139325

theorem inequality_holds_iff (r : ℝ) : 
  (r ≥ 0 ∧ ∀ s > 0, (4 * (r * s^2 + r^2 * s + 4 * s^2 + 4 * r * s)) / (r + s) ≤ 3 * r^2 * s) ↔ 
  r ≥ (2 + 2 * Real.sqrt 13) / 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l1393_139325


namespace NUMINAMATH_CALUDE_moles_of_H2O_formed_l1393_139303

-- Define the chemical reaction
structure ChemicalReaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String
  ratio : ℕ → ℕ → ℕ

-- Define the problem setup
def reaction : ChemicalReaction := {
  reactant1 := "NaHCO3"
  reactant2 := "HC2H3O2"
  product1 := "NaC2H3O2"
  product2 := "CO2"
  product3 := "H2O"
  ratio := λ x y => min x y
}

def initial_moles_NaHCO3 : ℕ := 3
def initial_moles_HC2H3O2 : ℕ := 3

-- State the theorem
theorem moles_of_H2O_formed (r : ChemicalReaction) 
  (h1 : r = reaction) 
  (h2 : initial_moles_NaHCO3 = 3) 
  (h3 : initial_moles_HC2H3O2 = 3) : 
  r.ratio initial_moles_NaHCO3 initial_moles_HC2H3O2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_moles_of_H2O_formed_l1393_139303


namespace NUMINAMATH_CALUDE_ambiguous_date_characterization_max_consecutive_ambiguous_proof_l1393_139378

/-- Represents a date with day and month -/
structure Date where
  day : Nat
  month : Nat
  h1 : day ≥ 1 ∧ day ≤ 31
  h2 : month ≥ 1 ∧ month ≤ 12

/-- Defines when a date is ambiguous -/
def is_ambiguous (d : Date) : Prop :=
  d.day ≥ 1 ∧ d.day ≤ 12 ∧ d.day ≠ d.month

/-- The maximum number of consecutive ambiguous dates in any month -/
def max_consecutive_ambiguous : Nat := 11

theorem ambiguous_date_characterization (d : Date) :
  is_ambiguous d ↔ d.day ≥ 1 ∧ d.day ≤ 12 ∧ d.day ≠ d.month :=
sorry

theorem max_consecutive_ambiguous_proof :
  ∀ m : Nat, m ≥ 1 → m ≤ 12 →
    (∃ consecutive : List Date,
      consecutive.length = max_consecutive_ambiguous ∧
      (∀ d ∈ consecutive, d.month = m ∧ is_ambiguous d) ∧
      (∀ d : Date, d.month = m → is_ambiguous d → d ∈ consecutive)) :=
sorry

end NUMINAMATH_CALUDE_ambiguous_date_characterization_max_consecutive_ambiguous_proof_l1393_139378


namespace NUMINAMATH_CALUDE_calculation_proof_l1393_139302

theorem calculation_proof : ((-1/3)⁻¹ : ℝ) - (Real.sqrt 3 - 2)^0 + 4 * Real.cos (π/4) = -4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1393_139302


namespace NUMINAMATH_CALUDE_polygon_sides_l1393_139358

theorem polygon_sides (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1393_139358


namespace NUMINAMATH_CALUDE_psychology_class_pairs_l1393_139371

def number_of_students : ℕ := 12

-- Function to calculate the number of unique pairs
def unique_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem stating that for 12 students, the number of unique pairs is 66
theorem psychology_class_pairs : 
  unique_pairs number_of_students = 66 := by sorry

end NUMINAMATH_CALUDE_psychology_class_pairs_l1393_139371


namespace NUMINAMATH_CALUDE_always_true_inequalities_l1393_139382

theorem always_true_inequalities (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_always_true_inequalities_l1393_139382


namespace NUMINAMATH_CALUDE_james_beef_cost_l1393_139313

def beef_purchase (num_packs : ℕ) (weight_per_pack : ℝ) (price_per_pound : ℝ) : ℝ :=
  (num_packs : ℝ) * weight_per_pack * price_per_pound

theorem james_beef_cost :
  beef_purchase 5 4 5.50 = 110 := by
  sorry

end NUMINAMATH_CALUDE_james_beef_cost_l1393_139313


namespace NUMINAMATH_CALUDE_sum_of_digits_653xy_divisible_by_80_l1393_139355

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem sum_of_digits_653xy_divisible_by_80 (x y : ℕ) :
  x < 10 →
  y < 10 →
  is_divisible_by (653 * 100 + x * 10 + y) 80 →
  x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_653xy_divisible_by_80_l1393_139355


namespace NUMINAMATH_CALUDE_heating_pad_cost_per_use_l1393_139320

/-- The cost per use of a heating pad -/
def cost_per_use (total_cost : ℚ) (uses_per_week : ℕ) (num_weeks : ℕ) : ℚ :=
  total_cost / (uses_per_week * num_weeks)

/-- Theorem: The cost per use of a heating pad is $5 -/
theorem heating_pad_cost_per_use :
  cost_per_use 30 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_heating_pad_cost_per_use_l1393_139320


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1393_139350

theorem partial_fraction_decomposition 
  (a b c d : ℤ) (h : a * d ≠ b * c) :
  ∃ (r s : ℝ), ∀ (x : ℝ), 
    1 / ((a * x + b) * (c * x + d)) = 
    r / (a * x + b) + s / (c * x + d) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1393_139350


namespace NUMINAMATH_CALUDE_fraction_equality_l1393_139323

theorem fraction_equality (a b : ℚ) (h : a / 5 = b / 3) : (a - b) / (3 * a) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1393_139323


namespace NUMINAMATH_CALUDE_rectangular_plot_perimeter_l1393_139395

/-- Given a rectangular plot with length 10 meters more than width,
    and fencing cost of 1430 Rs at 6.5 Rs/meter,
    prove the perimeter is 220 meters. -/
theorem rectangular_plot_perimeter :
  ∀ (width length : ℝ),
  length = width + 10 →
  6.5 * (2 * (length + width)) = 1430 →
  2 * (length + width) = 220 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_perimeter_l1393_139395


namespace NUMINAMATH_CALUDE_trichotomy_of_reals_l1393_139319

theorem trichotomy_of_reals : ∀ a b : ℝ, (a > b ∨ a = b ∨ a < b) ∧ 
  (¬(a > b ∧ a = b) ∧ ¬(a > b ∧ a < b) ∧ ¬(a = b ∧ a < b)) := by
  sorry

end NUMINAMATH_CALUDE_trichotomy_of_reals_l1393_139319


namespace NUMINAMATH_CALUDE_algebraic_identities_l1393_139385

theorem algebraic_identities (x y : ℝ) : 
  (3 * x^2 * y * (-2 * x * y)^3 = -24 * x^5 * y^4) ∧ 
  ((5 * x + 2 * y) * (3 * x - 2 * y) = 15 * x^2 - 4 * x * y - 4 * y^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l1393_139385


namespace NUMINAMATH_CALUDE_f_at_negative_five_l1393_139380

/-- Given a function f(x) = x^2 + 2x - 3, prove that f(-5) = 12 -/
theorem f_at_negative_five (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x - 3) : f (-5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_five_l1393_139380


namespace NUMINAMATH_CALUDE_semicircle_overlap_width_l1393_139318

/-- Given a rectangle with two semicircles drawn inside, where each semicircle
    has a radius of 5 cm and the rectangle height is 8 cm, the width of the
    overlap between the semicircles is 6 cm. -/
theorem semicircle_overlap_width (r : ℝ) (h : ℝ) (w : ℝ) :
  r = 5 →
  h = 8 →
  w = 2 * Real.sqrt (r^2 - (h/2)^2) →
  w = 6 := by
  sorry

#check semicircle_overlap_width

end NUMINAMATH_CALUDE_semicircle_overlap_width_l1393_139318


namespace NUMINAMATH_CALUDE_star_inequality_l1393_139328

-- Define the * operation
def star (m n : Int) : Int := (m + 2) * 3 - n

-- Theorem statement
theorem star_inequality : star 2 (-2) > star (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_star_inequality_l1393_139328


namespace NUMINAMATH_CALUDE_opposite_numbers_problem_l1393_139397

theorem opposite_numbers_problem (x y : ℚ) 
  (h1 : x + y = 0)  -- x and y are opposite numbers
  (h2 : x - y = 3)  -- given condition
  : x^2 + 2*x*y + 1 = -5/4 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_problem_l1393_139397


namespace NUMINAMATH_CALUDE_two_primes_between_lower_limit_and_14_l1393_139354

theorem two_primes_between_lower_limit_and_14 : 
  ∃ (x : ℕ), x ≤ 7 ∧ 
  (∃ (p q : ℕ), Prime p ∧ Prime q ∧ x < p ∧ p < q ∧ q < 14) ∧
  (∀ (y : ℕ), y > 7 → ¬(∃ (p q : ℕ), Prime p ∧ Prime q ∧ y < p ∧ p < q ∧ q < 14)) :=
sorry

end NUMINAMATH_CALUDE_two_primes_between_lower_limit_and_14_l1393_139354


namespace NUMINAMATH_CALUDE_solution_to_equation_l1393_139335

theorem solution_to_equation (x y : ℝ) : 
  x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ↔ x = 1/3 ∧ y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1393_139335


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1393_139367

/-- Given a system of linear equations:
    3x₁ - x₂ + 3x₃ = 5
    2x₁ - x₂ + 4x₃ = 5
    x₁ + 2x₂ - 3x₃ = 0
    Prove that (1, 1, 1) is a solution. -/
theorem solution_satisfies_system :
  let x₁ : ℝ := 1
  let x₂ : ℝ := 1
  let x₃ : ℝ := 1
  (3 * x₁ - x₂ + 3 * x₃ = 5) ∧
  (2 * x₁ - x₂ + 4 * x₃ = 5) ∧
  (x₁ + 2 * x₂ - 3 * x₃ = 0) :=
by sorry

#check solution_satisfies_system

end NUMINAMATH_CALUDE_solution_satisfies_system_l1393_139367


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1393_139372

-- Define the type for sampling methods
inductive SamplingMethod
| Lottery
| RandomNumber
| Stratified
| Systematic

-- Define the company's production
structure Company where
  sedanModels : Nat
  significantDifferences : Bool

-- Define the appropriateness of a sampling method
def isAppropriate (method : SamplingMethod) (company : Company) : Prop :=
  method = SamplingMethod.Stratified ∧ 
  company.sedanModels > 1 ∧ 
  company.significantDifferences

-- Theorem statement
theorem stratified_sampling_most_appropriate (company : Company) 
  (h1 : company.sedanModels = 3) 
  (h2 : company.significantDifferences = true) :
  isAppropriate SamplingMethod.Stratified company := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1393_139372


namespace NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l1393_139347

theorem shaded_area_of_tiled_floor :
  let floor_length : ℝ := 12
  let floor_width : ℝ := 10
  let tile_length : ℝ := 2
  let tile_width : ℝ := 1
  let circle_radius : ℝ := 1/2
  let triangle_base : ℝ := 1/2
  let triangle_height : ℝ := 1/2
  let num_tiles : ℝ := (floor_length / tile_width) * (floor_width / tile_length)
  let tile_area : ℝ := tile_length * tile_width
  let white_circle_area : ℝ := π * circle_radius^2
  let white_triangle_area : ℝ := 1/2 * triangle_base * triangle_height
  let shaded_area_per_tile : ℝ := tile_area - white_circle_area - white_triangle_area
  let total_shaded_area : ℝ := num_tiles * shaded_area_per_tile
  total_shaded_area = 112.5 - 15 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l1393_139347


namespace NUMINAMATH_CALUDE_registration_combinations_l1393_139366

/-- The number of people signing up for sports competitions -/
def num_people : ℕ := 4

/-- The number of available sports competitions -/
def num_sports : ℕ := 3

/-- 
Theorem: Given 'num_people' people and 'num_sports' sports competitions, 
where each person must choose exactly one event, the total number of 
possible registration combinations is 'num_sports' raised to the power of 'num_people'.
-/
theorem registration_combinations : 
  (num_sports : ℕ) ^ (num_people : ℕ) = 81 := by
  sorry

#eval (num_sports : ℕ) ^ (num_people : ℕ)

end NUMINAMATH_CALUDE_registration_combinations_l1393_139366


namespace NUMINAMATH_CALUDE_sports_meeting_score_l1393_139304

/-- Represents the score for a single placement --/
inductive Placement
| first
| second
| third

/-- Calculates the score for a given placement --/
def score (p : Placement) : Nat :=
  match p with
  | .first => 5
  | .second => 3
  | .third => 1

/-- Represents the placements of a class --/
structure ClassPlacements where
  first : Nat
  second : Nat
  third : Nat

/-- Calculates the total score for a class given its placements --/
def totalScore (cp : ClassPlacements) : Nat :=
  cp.first * score Placement.first +
  cp.second * score Placement.second +
  cp.third * score Placement.third

/-- Calculates the total number of placements for a class --/
def totalPlacements (cp : ClassPlacements) : Nat :=
  cp.first + cp.second + cp.third

theorem sports_meeting_score (class1 class2 : ClassPlacements) :
  totalPlacements class1 = 2 →
  totalPlacements class2 = 4 →
  totalScore class1 = totalScore class2 →
  totalScore class1 + totalScore class2 + 7 = 27 :=
by sorry

end NUMINAMATH_CALUDE_sports_meeting_score_l1393_139304


namespace NUMINAMATH_CALUDE_problem_solution_l1393_139331

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 6}

def B (m : ℝ) : Set ℝ := {x | 6 - m < x ∧ x < m + 3}

theorem problem_solution :
  (∀ m : ℝ,
    (m = 6 → (Aᶜ ∪ B m) = {x | x < -3 ∨ x > 0})) ∧
  (∀ m : ℝ,
    (A ∪ B m = A) ↔ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1393_139331


namespace NUMINAMATH_CALUDE_total_games_attended_l1393_139379

def games_this_month : ℕ := 11
def games_last_month : ℕ := 17
def games_next_month : ℕ := 16

theorem total_games_attended : games_this_month + games_last_month + games_next_month = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_games_attended_l1393_139379


namespace NUMINAMATH_CALUDE_derivative_through_point_l1393_139383

theorem derivative_through_point (a : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + a*x + 1
  let f' : ℝ → ℝ := λ x => 2*x + a
  f' 2 = 4 → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_through_point_l1393_139383


namespace NUMINAMATH_CALUDE_money_distribution_l1393_139370

/-- Given three people A, B, and C with the following conditions:
  - The total amount between A, B, and C is 900
  - A and C together have 400
  - B and C together have 750
  Prove that C has 250. -/
theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 900)
  (h2 : A + C = 400)
  (h3 : B + C = 750) : 
  C = 250 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1393_139370


namespace NUMINAMATH_CALUDE_sandy_book_purchase_l1393_139338

/-- The number of books Sandy bought from the first shop -/
def books_first_shop : ℕ := 65

/-- The amount Sandy spent at the first shop (in cents) -/
def cost_first_shop : ℕ := 148000

/-- The amount Sandy spent at the second shop (in cents) -/
def cost_second_shop : ℕ := 92000

/-- The average price per book (in cents) -/
def average_price : ℕ := 2000

/-- The number of books Sandy bought from the second shop -/
def books_second_shop : ℕ := 55

theorem sandy_book_purchase :
  (cost_first_shop + cost_second_shop) / (books_first_shop + books_second_shop) = average_price ∧
  (cost_first_shop + cost_second_shop) = average_price * (books_first_shop + books_second_shop) :=
by sorry

end NUMINAMATH_CALUDE_sandy_book_purchase_l1393_139338


namespace NUMINAMATH_CALUDE_quadratic_general_form_l1393_139306

theorem quadratic_general_form :
  ∀ x : ℝ, (6 * x^2 = 5 * x - 4) ↔ (6 * x^2 - 5 * x + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_general_form_l1393_139306


namespace NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l1393_139368

theorem complex_arithmetic_evaluation : (7 - 3*I) - 3*(2 - 5*I) = 1 + 12*I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l1393_139368


namespace NUMINAMATH_CALUDE_max_consecutive_common_divisor_l1393_139326

def a (n : ℕ) : ℤ :=
  if 7 ∣ n then n^6 - 2017 else (n^6 - 2017) / 7

theorem max_consecutive_common_divisor :
  (∃ k : ℕ, ∀ i : ℕ, ∃ d > 1, ∀ j : ℕ, j < k → d ∣ a (i + j)) ∧
  (¬∃ k > 2, ∀ i : ℕ, ∃ d > 1, ∀ j : ℕ, j < k → d ∣ a (i + j)) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_common_divisor_l1393_139326


namespace NUMINAMATH_CALUDE_f_max_value_l1393_139349

/-- The quadratic function f(y) = -3y^2 + 18y - 7 -/
def f (y : ℝ) : ℝ := -3 * y^2 + 18 * y - 7

/-- The maximum value of f(y) is 20 -/
theorem f_max_value : ∃ (M : ℝ), M = 20 ∧ ∀ (y : ℝ), f y ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l1393_139349


namespace NUMINAMATH_CALUDE_equal_area_triangles_l1393_139390

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equal_area_triangles :
  triangleArea 25 25 30 = triangleArea 25 25 40 := by sorry

end NUMINAMATH_CALUDE_equal_area_triangles_l1393_139390


namespace NUMINAMATH_CALUDE_infinitely_many_special_n_l1393_139327

theorem infinitely_many_special_n : ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 
  (∃ m : ℕ, n * m = 2^(2^n + 1) + 1) ∧ 
  (∀ m : ℕ, n * m ≠ 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_special_n_l1393_139327


namespace NUMINAMATH_CALUDE_even_number_less_than_square_l1393_139389

theorem even_number_less_than_square (m : ℕ) (h1 : m > 1) (h2 : Even m) : m < m^2 := by
  sorry

end NUMINAMATH_CALUDE_even_number_less_than_square_l1393_139389


namespace NUMINAMATH_CALUDE_special_function_properties_l1393_139351

open Real

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y - 1) ∧
  (∀ x, x > 0 → f x > 1) ∧
  (f 3 = 4)

/-- The main theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (∀ x y, x < y → f x < f y) ∧ (f 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l1393_139351


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1393_139345

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ (n - 3) * (n + 5) < 0) ∧ Finset.card S = 7 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1393_139345


namespace NUMINAMATH_CALUDE_no_thirty_degree_angle_l1393_139346

structure Cube where
  vertices : Finset (Fin 8)

def skew_lines (c : Cube) (p1 p2 p3 p4 : Fin 8) : Prop :=
  p1 ∈ c.vertices ∧ p2 ∈ c.vertices ∧ p3 ∈ c.vertices ∧ p4 ∈ c.vertices ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4

def angle_between_lines (l1 l2 : Fin 8 × Fin 8) : ℝ :=
  sorry -- Definition of angle calculation between two lines in a cube

theorem no_thirty_degree_angle (c : Cube) :
  ∀ (p1 p2 p3 p4 : Fin 8),
    skew_lines c p1 p2 p3 p4 →
    angle_between_lines (p1, p2) (p3, p4) ≠ 30 :=
sorry

end NUMINAMATH_CALUDE_no_thirty_degree_angle_l1393_139346


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1393_139393

/-- Represents a seating arrangement in an examination room -/
structure ExamRoom where
  rows : Nat
  columns : Nat
  total_seats : Nat

/-- Calculates the number of possible seating arrangements for two students
    who cannot sit adjacent to each other in the given exam room -/
def count_seating_arrangements (room : ExamRoom) : Nat :=
  sorry

/-- Theorem stating that the number of seating arrangements for two students
    in a 5x6 exam room with 30 seats, where they cannot sit adjacent to each other,
    is 772 -/
theorem seating_arrangements_count :
  let exam_room : ExamRoom := ⟨5, 6, 30⟩
  count_seating_arrangements exam_room = 772 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1393_139393


namespace NUMINAMATH_CALUDE_base_10_to_base_4_123_l1393_139343

/-- Converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of digits is a valid base 4 representation -/
def isValidBase4 (digits : List ℕ) : Prop :=
  sorry

theorem base_10_to_base_4_123 :
  let base4Repr := toBase4 123
  isValidBase4 base4Repr ∧ base4Repr = [1, 3, 2, 3] :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_4_123_l1393_139343


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1393_139308

theorem quadratic_equation_solution :
  ∃! (x : ℚ), x > 0 ∧ 6 * x^2 + 9 * x - 24 = 0 ∧ x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1393_139308


namespace NUMINAMATH_CALUDE_area_is_seven_and_half_l1393_139305

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf_continuous : Continuous f)
variable (hf_monotone : Monotone f)
variable (hf_0 : f 0 = 0)
variable (hf_1 : f 1 = 1)

-- Define the area calculation function
def area_bounded (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem area_is_seven_and_half :
  area_bounded f = 7.5 :=
sorry

end NUMINAMATH_CALUDE_area_is_seven_and_half_l1393_139305


namespace NUMINAMATH_CALUDE_burger_cost_l1393_139330

/-- Proves that the cost of each burger is $3.50 given Selena's expenses --/
theorem burger_cost (tip : ℝ) (steak_price : ℝ) (ice_cream_price : ℝ) (remaining : ℝ) :
  tip = 99 →
  steak_price = 24 →
  ice_cream_price = 2 →
  remaining = 38 →
  ∃ (burger_price : ℝ),
    burger_price = 3.5 ∧
    tip = 2 * steak_price + 2 * burger_price + 3 * ice_cream_price + remaining :=
by
  sorry

end NUMINAMATH_CALUDE_burger_cost_l1393_139330


namespace NUMINAMATH_CALUDE_triangle_area_proof_l1393_139344

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define the side lengths
def SideLength (A B C : ℝ × ℝ) (a b c : ℝ) : Prop := 
  Triangle A B C ∧ 
  (a = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) ∧
  (b = Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) ∧
  (c = Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))

-- Define the angle C
def AngleC (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the area of the triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_proof 
  (A B C : ℝ × ℝ) (a b c : ℝ) 
  (h1 : SideLength A B C a b c)
  (h2 : b = 1)
  (h3 : c = Real.sqrt 3)
  (h4 : AngleC A B C = 2 * Real.pi / 3) :
  TriangleArea A B C = Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l1393_139344


namespace NUMINAMATH_CALUDE_problem_solution_l1393_139399

theorem problem_solution : (1 / ((-8^2)^4)) * (-8)^9 = -8 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1393_139399


namespace NUMINAMATH_CALUDE_projection_of_two_vectors_l1393_139365

/-- Given two vectors that project to the same vector, find the projection --/
theorem projection_of_two_vectors (v₁ v₂ v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  let p := (14/73, 214/73)
  (∃ (k₁ k₂ : ℝ), v₁ - k₁ • v = p ∧ v₂ - k₂ • v = p) →
  v₁ = (5, -2) →
  v₂ = (2, 6) →
  (∃ (k : ℝ), v₁ - k • v = p ∧ v₂ - k • v = p) :=
by sorry


end NUMINAMATH_CALUDE_projection_of_two_vectors_l1393_139365


namespace NUMINAMATH_CALUDE_triangle_problem_l1393_139321

theorem triangle_problem (a b c A B C : ℝ) (h1 : 2 * a * Real.cos B + b = 2 * c)
  (h2 : a = 2 * Real.sqrt 3) (h3 : (1 / 2) * b * c * Real.sin A = Real.sqrt 3) :
  A = π / 3 ∧ Real.sin B + Real.sin C = Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1393_139321


namespace NUMINAMATH_CALUDE_power_of_two_triplets_l1393_139352

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def satisfies_conditions (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

theorem power_of_two_triplets :
  ∀ a b c : ℕ,
    satisfies_conditions a b c ↔
      (a = 2 ∧ b = 2 ∧ c = 2) ∨
      (a = 2 ∧ b = 2 ∧ c = 3) ∨
      (a = 2 ∧ b = 3 ∧ c = 3) ∨
      (a = 2 ∧ b = 3 ∧ c = 6) ∨
      (a = 3 ∧ b = 5 ∧ c = 7) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_triplets_l1393_139352


namespace NUMINAMATH_CALUDE_tournament_players_count_l1393_139309

/-- Represents a tournament with the given conditions -/
structure Tournament where
  n : ℕ  -- Number of players not in the lowest 8
  total_players : ℕ := n + 8
  points_among_n : ℕ := n * (n - 1) / 2
  points_n_vs_lowest8 : ℕ := points_among_n / 3
  points_among_lowest8 : ℕ := 28
  total_points : ℕ := 4 * points_among_n / 3 + 2 * points_among_lowest8

/-- The theorem stating that the total number of players in the tournament is 50 -/
theorem tournament_players_count (t : Tournament) : t.total_players = 50 := by
  sorry

end NUMINAMATH_CALUDE_tournament_players_count_l1393_139309


namespace NUMINAMATH_CALUDE_parabola_equation_l1393_139356

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  p > 0 ∧ y^2 = 2 * p * x

-- Define the area of triangle AOB
def triangle_area (area : ℝ) : Prop := area = Real.sqrt 3

-- Theorem statement
theorem parabola_equation (a b p : ℝ) :
  (∃ x y : ℝ, hyperbola a b x y) →
  eccentricity 2 →
  (∃ x y : ℝ, parabola p x y) →
  triangle_area (Real.sqrt 3) →
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1393_139356


namespace NUMINAMATH_CALUDE_sum_and_subtract_l1393_139362

theorem sum_and_subtract : (2345 + 3452 + 4523 + 5234) - 1234 = 14320 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_subtract_l1393_139362


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1393_139376

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (5, -2)
  are_parallel a b → m = -10 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1393_139376


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1393_139315

/-- Given a circle (x+a)^2 + y^2 = 4 and a line x - y - 4 = 0 intersecting the circle
    to form a chord of length 2√2, prove that a = -2 or a = -6 -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, (x + a)^2 + y^2 = 4 ∧ x - y - 4 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + a)^2 + y₁^2 = 4 ∧ x₁ - y₁ - 4 = 0 ∧
    (x₂ + a)^2 + y₂^2 = 4 ∧ x₂ - y₂ - 4 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →
  a = -2 ∨ a = -6 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1393_139315


namespace NUMINAMATH_CALUDE_count_solutions_eq_288_l1393_139396

/-- The count of positive integers N less than 500 for which x^⌊x⌋ = N has a solution -/
def count_solutions : ℕ :=
  let floor_0_count := 1  -- N = 1 for ⌊x⌋ = 0
  let floor_1_count := 0  -- Already counted in floor_0_count
  let floor_2_count := 5  -- N = 4, 5, ..., 8
  let floor_3_count := 38 -- N = 27, 28, ..., 64
  let floor_4_count := 244 -- N = 256, 257, ..., 499
  floor_0_count + floor_1_count + floor_2_count + floor_3_count + floor_4_count

/-- The main theorem stating that the count of solutions is 288 -/
theorem count_solutions_eq_288 : count_solutions = 288 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_eq_288_l1393_139396


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l1393_139373

/-- The locus of points P(x,y) satisfying the given conditions forms an ellipse -/
theorem locus_is_ellipse (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 0)
  let directrix : ℝ → Prop := λ t => t = 9
  let dist_ratio : ℝ := 1/3
  let dist_to_A : ℝ := Real.sqrt ((x - A.1)^2 + (y - A.2)^2)
  let dist_to_directrix : ℝ := |x - 9|
  dist_to_A / dist_to_directrix = dist_ratio →
  x^2/9 + y^2/8 = 1 :=
by sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l1393_139373


namespace NUMINAMATH_CALUDE_area_triangle_DEF_is_seven_l1393_139386

/-- The area of triangle DEF in the given configuration --/
def area_triangle_DEF (side_length_PQRS : ℝ) (side_length_small_square : ℝ) : ℝ :=
  sorry

/-- The theorem stating the area of triangle DEF is 7 cm² --/
theorem area_triangle_DEF_is_seven
  (h1 : area_triangle_DEF 6 2 = 7) :
  ∃ (side_length_PQRS side_length_small_square : ℝ),
    side_length_PQRS^2 = 36 ∧
    side_length_small_square = 2 ∧
    area_triangle_DEF side_length_PQRS side_length_small_square = 7 :=
  sorry

end NUMINAMATH_CALUDE_area_triangle_DEF_is_seven_l1393_139386


namespace NUMINAMATH_CALUDE_sum_of_threes_place_values_l1393_139333

def number : ℕ := 63130

def first_three_place_value : ℕ := 3000
def second_three_place_value : ℕ := 30

theorem sum_of_threes_place_values :
  first_three_place_value + second_three_place_value = 3030 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_threes_place_values_l1393_139333


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l1393_139314

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_sum_of_powers (h : i^2 = -1) : (1 + i)^30 + (1 - i)^30 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l1393_139314


namespace NUMINAMATH_CALUDE_total_bill_correct_l1393_139322

/-- Represents the group composition and meal prices -/
structure GroupInfo where
  adults : Nat
  teenagers : Nat
  children : Nat
  adultMealPrice : ℚ
  teenagerMealPrice : ℚ
  childrenMealPrice : ℚ
  adultSodaPrice : ℚ
  childrenSodaPrice : ℚ
  appetizerPrice : ℚ
  dessertPrice : ℚ

/-- Represents the number of additional items ordered -/
structure AdditionalItems where
  appetizers : Nat
  desserts : Nat

/-- Represents the discount conditions -/
structure DiscountConditions where
  adultMealDiscount : ℚ
  childrenMealSodaDiscount : ℚ
  totalBillDiscount : ℚ
  minChildrenForDiscount : Nat
  teenagersPerFreeDessert : Nat
  minTotalForExtraDiscount : ℚ

/-- Calculates the total bill after all applicable discounts and special offers -/
def calculateTotalBill (group : GroupInfo) (items : AdditionalItems) (discounts : DiscountConditions) : ℚ :=
  sorry

/-- Theorem stating that the calculated total bill matches the expected result -/
theorem total_bill_correct (group : GroupInfo) (items : AdditionalItems) (discounts : DiscountConditions) :
  let expectedBill : ℚ := 230.70
  calculateTotalBill group items discounts = expectedBill :=
by
  sorry

end NUMINAMATH_CALUDE_total_bill_correct_l1393_139322


namespace NUMINAMATH_CALUDE_bee_count_correct_l1393_139311

def bee_count (day : Nat) : ℕ :=
  match day with
  | 0 => 144  -- Monday
  | 1 => 432  -- Tuesday
  | 2 => 216  -- Wednesday
  | 3 => 432  -- Thursday
  | 4 => 648  -- Friday
  | 5 => 486  -- Saturday
  | 6 => 1944 -- Sunday
  | _ => 0    -- Invalid day

def daily_multiplier (day : Nat) : ℚ :=
  match day with
  | 0 => 1    -- Monday (base)
  | 1 => 3    -- Tuesday
  | 2 => 1/2  -- Wednesday
  | 3 => 2    -- Thursday
  | 4 => 3/2  -- Friday
  | 5 => 3/4  -- Saturday
  | 6 => 4    -- Sunday
  | _ => 0    -- Invalid day

theorem bee_count_correct (day : Nat) :
  day < 7 →
  (day = 0 ∨ (bee_count day : ℚ) = (bee_count (day - 1) : ℚ) * daily_multiplier day) :=
by sorry

end NUMINAMATH_CALUDE_bee_count_correct_l1393_139311


namespace NUMINAMATH_CALUDE_rice_bags_sold_l1393_139398

theorem rice_bags_sold (initial_stock restocked final_stock : ℕ) 
  (h1 : initial_stock = 55)
  (h2 : restocked = 132)
  (h3 : final_stock = 164) :
  initial_stock + restocked - final_stock = 23 := by
  sorry

end NUMINAMATH_CALUDE_rice_bags_sold_l1393_139398


namespace NUMINAMATH_CALUDE_watermelon_puzzle_l1393_139363

theorem watermelon_puzzle (A B C : ℕ) 
  (h1 : C - (A + B) = 6)
  (h2 : (B + C) - A = 16)
  (h3 : (C + A) - B = 8) :
  A + B + C = 18 := by
sorry

end NUMINAMATH_CALUDE_watermelon_puzzle_l1393_139363


namespace NUMINAMATH_CALUDE_exam_failure_percentage_l1393_139391

theorem exam_failure_percentage (total_percentage : ℝ) 
  (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  total_percentage = 100 →
  failed_hindi = 30 →
  failed_both = 28 →
  passed_both = 56 →
  ∃ failed_english : ℝ,
    failed_english = 42 ∧
    total_percentage - passed_both = failed_hindi + failed_english - failed_both :=
by sorry

end NUMINAMATH_CALUDE_exam_failure_percentage_l1393_139391
