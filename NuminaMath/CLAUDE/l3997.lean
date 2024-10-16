import Mathlib

namespace NUMINAMATH_CALUDE_sandy_balloons_l3997_399781

/-- Given the total number of blue balloons and the number of balloons Alyssa and Sally have,
    calculate the number of balloons Sandy has. -/
theorem sandy_balloons (total : ℕ) (alyssa : ℕ) (sally : ℕ) (h1 : total = 104) (h2 : alyssa = 37) (h3 : sally = 39) :
  total - alyssa - sally = 28 := by
  sorry

end NUMINAMATH_CALUDE_sandy_balloons_l3997_399781


namespace NUMINAMATH_CALUDE_basketball_game_ratio_l3997_399789

theorem basketball_game_ratio :
  let girls : ℕ := 30
  let boys : ℕ := girls + 18
  let ratio : ℚ := boys / girls
  ratio = 8 / 5 := by sorry

end NUMINAMATH_CALUDE_basketball_game_ratio_l3997_399789


namespace NUMINAMATH_CALUDE_power_one_plus_power_five_quotient_l3997_399744

theorem power_one_plus_power_five_quotient (n : ℕ) :
  (1 : ℕ)^345 + 5^10 / 5^7 = 126 := by
  sorry

end NUMINAMATH_CALUDE_power_one_plus_power_five_quotient_l3997_399744


namespace NUMINAMATH_CALUDE_car_distance_l3997_399704

theorem car_distance (x : ℝ) (h : 12 * x = 10 * (x + 2)) : 12 * x = 120 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l3997_399704


namespace NUMINAMATH_CALUDE_park_rose_bushes_l3997_399791

/-- The number of rose bushes in a park after planting new ones -/
def total_rose_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The park will have 6 rose bushes after planting -/
theorem park_rose_bushes : total_rose_bushes 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_park_rose_bushes_l3997_399791


namespace NUMINAMATH_CALUDE_fourteen_n_divisibility_l3997_399772

theorem fourteen_n_divisibility (n d : ℕ) (p₁ p₂ p₃ : ℕ) 
  (h1 : 0 < n ∧ n < 200)
  (h2 : Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃)
  (h3 : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h4 : n = p₁ * p₂ * p₃)
  (h5 : (14 * n) % d = 0) : 
  d = n := by
sorry

end NUMINAMATH_CALUDE_fourteen_n_divisibility_l3997_399772


namespace NUMINAMATH_CALUDE_quintic_integer_root_counts_l3997_399773

/-- The set of possible numbers of integer roots (counting multiplicity) for a quintic polynomial with integer coefficients -/
def QuinticIntegerRootCounts : Set ℕ := {0, 1, 2, 3, 4, 5}

/-- A quintic polynomial with integer coefficients -/
structure QuinticPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The number of integer roots (counting multiplicity) of a quintic polynomial -/
def integerRootCount (p : QuinticPolynomial) : ℕ := sorry

theorem quintic_integer_root_counts (p : QuinticPolynomial) :
  integerRootCount p ∈ QuinticIntegerRootCounts := by sorry

end NUMINAMATH_CALUDE_quintic_integer_root_counts_l3997_399773


namespace NUMINAMATH_CALUDE_ghost_castle_paths_l3997_399722

theorem ghost_castle_paths (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ghost_castle_paths_l3997_399722


namespace NUMINAMATH_CALUDE_range_of_a_l3997_399754

theorem range_of_a (p : Prop) (h : p) : 
  (∀ x : ℝ, x ∈ Set.Ioo 1 2 → Real.exp x - a ≤ 0) → 
  a ∈ Set.Ici (Real.exp 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3997_399754


namespace NUMINAMATH_CALUDE_division_problem_l3997_399733

theorem division_problem (A : ℕ) : 
  (11 / A = 3) ∧ (11 % A = 2) → A = 3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3997_399733


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3997_399798

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3997_399798


namespace NUMINAMATH_CALUDE_pebble_throwing_difference_l3997_399745

/-- The number of pebbles Candy throws -/
def candy_pebbles : ℕ := 4

/-- The number of pebbles Lance throws -/
def lance_pebbles : ℕ := 3 * candy_pebbles

/-- The difference between Lance's pebbles and Candy's pebbles -/
def pebble_difference : ℕ := lance_pebbles - candy_pebbles

theorem pebble_throwing_difference :
  pebble_difference = 8 := by sorry

end NUMINAMATH_CALUDE_pebble_throwing_difference_l3997_399745


namespace NUMINAMATH_CALUDE_rhombus_area_l3997_399790

/-- The area of a rhombus with side length √145 and diagonals differing by 10 units is 100 square units. -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) : 
  s = Real.sqrt 145 →
  d₂ - d₁ = 10 →
  s^2 = (d₁/2)^2 + (d₂/2)^2 →
  (1/2) * d₁ * d₂ = 100 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3997_399790


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_negative_four_satisfies_l3997_399726

theorem quadratic_two_distinct_roots (c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + c = 0 ∧ y^2 - 4*y + c = 0) → c < 4 :=
by sorry

theorem negative_four_satisfies (c : ℝ) : 
  c = -4 → (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + c = 0 ∧ y^2 - 4*y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_negative_four_satisfies_l3997_399726


namespace NUMINAMATH_CALUDE_bulb_replacement_probabilities_l3997_399792

/-- Represents the probability of a bulb lasting more than a given number of years -/
def bulb_survival_prob (years : ℕ) : ℝ :=
  match years with
  | 1 => 0.8
  | 2 => 0.3
  | _ => 0 -- Assuming 0 probability for other years

/-- The number of lamps in the conference room -/
def num_lamps : ℕ := 3

/-- Calculates the probability of not replacing any bulbs during the first replacement -/
def prob_no_replace : ℝ := (bulb_survival_prob 1) ^ num_lamps

/-- Calculates the probability of replacing exactly 2 bulbs during the first replacement -/
def prob_replace_two : ℝ := 
  (Nat.choose num_lamps 2 : ℝ) * (bulb_survival_prob 1) * (1 - bulb_survival_prob 1)^2

/-- Calculates the probability that a bulb needs to be replaced during the second replacement -/
def prob_replace_second : ℝ := 
  (1 - bulb_survival_prob 1)^2 + (bulb_survival_prob 1) * (1 - bulb_survival_prob 2)

theorem bulb_replacement_probabilities :
  (prob_no_replace = 0.512) ∧
  (prob_replace_two = 0.096) ∧
  (prob_replace_second = 0.6) :=
sorry

end NUMINAMATH_CALUDE_bulb_replacement_probabilities_l3997_399792


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l3997_399797

theorem min_value_of_expression (x y : ℝ) : (x * y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

theorem lower_bound_achievable : ∃ x y : ℝ, (x * y - 1)^2 + (x + y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l3997_399797


namespace NUMINAMATH_CALUDE_quentavious_nickels_l3997_399770

/-- Proves the number of nickels Quentavious left with -/
theorem quentavious_nickels (initial_nickels : ℕ) (gum_per_nickel : ℕ) (gum_received : ℕ) :
  initial_nickels = 5 →
  gum_per_nickel = 2 →
  gum_received = 6 →
  initial_nickels - (gum_received / gum_per_nickel) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quentavious_nickels_l3997_399770


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3997_399735

-- System 1
theorem system_one_solution (x : ℝ) :
  (2 * x + 10 ≤ 5 * x + 1 ∧ 3 * (x - 1) > 9) ↔ x > 4 := by sorry

-- System 2
theorem system_two_solution (x : ℝ) :
  (3 * (x + 2) ≥ 2 * x + 5 ∧ 2 * x - (3 * x + 1) / 2 < 1) ↔ -1 ≤ x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3997_399735


namespace NUMINAMATH_CALUDE_scalene_to_right_triangle_l3997_399750

theorem scalene_to_right_triangle 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c) :
  ∃ x : ℝ, (a + x)^2 + (b + x)^2 = (c + x)^2 :=
sorry

end NUMINAMATH_CALUDE_scalene_to_right_triangle_l3997_399750


namespace NUMINAMATH_CALUDE_coin_triangle_border_mass_l3997_399757

/-- A configuration of coins in a triangular arrangement -/
structure CoinTriangle where
  total_coins : ℕ
  border_coins : ℕ
  trio_mass : ℝ

/-- The property that the total mass of border coins is a multiple of the trio mass -/
def border_mass_property (ct : CoinTriangle) : Prop :=
  ∃ k : ℕ, (ct.border_coins : ℝ) * ct.trio_mass / 3 = k * ct.trio_mass

/-- The theorem stating the total mass of border coins in the specific configuration -/
theorem coin_triangle_border_mass (ct : CoinTriangle) 
  (h1 : ct.total_coins = 28)
  (h2 : ct.border_coins = 18)
  (h3 : ct.trio_mass = 10) :
  (ct.border_coins : ℝ) * ct.trio_mass / 3 = 60 :=
sorry

end NUMINAMATH_CALUDE_coin_triangle_border_mass_l3997_399757


namespace NUMINAMATH_CALUDE_x_value_proof_l3997_399748

theorem x_value_proof (x : ℝ) (h : 0.65 * x = 0.20 * 487.50) : x = 150 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3997_399748


namespace NUMINAMATH_CALUDE_pencil_cost_with_discount_cost_of_3000_pencils_l3997_399701

/-- The cost of pencils with a bulk discount -/
theorem pencil_cost_with_discount (base_quantity : ℕ) (base_cost : ℚ) 
  (order_quantity : ℕ) (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let base_price_per_pencil := base_cost / base_quantity
  let discounted_price_per_pencil := base_price_per_pencil * (1 - discount_rate)
  let total_cost := if order_quantity > discount_threshold
                    then order_quantity * discounted_price_per_pencil
                    else order_quantity * base_price_per_pencil
  total_cost

/-- The cost of 3000 pencils with the given conditions -/
theorem cost_of_3000_pencils : 
  pencil_cost_with_discount 150 40 3000 1000 (5/100) = 760 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_with_discount_cost_of_3000_pencils_l3997_399701


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l3997_399736

theorem magnitude_of_complex_power : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l3997_399736


namespace NUMINAMATH_CALUDE_count_rectangles_l3997_399703

/-- The number of rectangles with sides parallel to the axes in an n×n grid -/
def num_rectangles (n : ℕ) : ℕ :=
  n^2 * (n-1)^2 / 4

/-- Theorem stating the number of rectangles in an n×n grid -/
theorem count_rectangles (n : ℕ) (h : n > 0) :
  num_rectangles n = (n.choose 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_count_rectangles_l3997_399703


namespace NUMINAMATH_CALUDE_exponential_function_at_zero_l3997_399712

theorem exponential_function_at_zero (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  (fun x : ℝ => a^x) 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_at_zero_l3997_399712


namespace NUMINAMATH_CALUDE_emma_share_l3997_399799

theorem emma_share (total : ℕ) (ratio_daniel ratio_emma ratio_fiona : ℕ) (h1 : total = 153) (h2 : ratio_daniel = 3) (h3 : ratio_emma = 5) (h4 : ratio_fiona = 9) : 
  (ratio_emma * total) / (ratio_daniel + ratio_emma + ratio_fiona) = 45 := by
sorry

end NUMINAMATH_CALUDE_emma_share_l3997_399799


namespace NUMINAMATH_CALUDE_boys_without_calculators_l3997_399729

theorem boys_without_calculators (total_boys : ℕ) (students_with_calc : ℕ) (girls_with_calc : ℕ) (forgot_calc : ℕ) :
  total_boys = 20 →
  students_with_calc = 26 →
  girls_with_calc = 15 →
  forgot_calc = 3 →
  total_boys - (students_with_calc - girls_with_calc + (forgot_calc * total_boys) / (students_with_calc + forgot_calc)) = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_boys_without_calculators_l3997_399729


namespace NUMINAMATH_CALUDE_pyramid_volume_l3997_399719

/-- The volume of a pyramid with a square base and given dimensions -/
theorem pyramid_volume (base_side : ℝ) (edge_length : ℝ) (h : base_side = 10 ∧ edge_length = 17) :
  (1 / 3 : ℝ) * base_side ^ 2 * Real.sqrt (edge_length ^ 2 - (base_side ^ 2 / 2)) = 
    (100 * Real.sqrt 239) / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3997_399719


namespace NUMINAMATH_CALUDE_tissue_cost_with_discount_l3997_399776

/-- Calculate the total cost of tissues with discount --/
theorem tissue_cost_with_discount
  (num_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (price_per_tissue : ℚ)
  (discount_rate : ℚ)
  (h_num_boxes : num_boxes = 25)
  (h_packs_per_box : packs_per_box = 18)
  (h_tissues_per_pack : tissues_per_pack = 150)
  (h_price_per_tissue : price_per_tissue = 6 / 100)
  (h_discount_rate : discount_rate = 1 / 10) :
  (num_boxes : ℚ) * (packs_per_box : ℚ) * (tissues_per_pack : ℚ) * price_per_tissue *
    (1 - discount_rate) = 3645 := by
  sorry

#check tissue_cost_with_discount

end NUMINAMATH_CALUDE_tissue_cost_with_discount_l3997_399776


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l3997_399718

/-- Calculates the total distance walked given the number of blocks and distance per block -/
def total_distance (blocks_east blocks_north : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * miles_per_block

/-- Proves that walking 8 blocks east and 10 blocks north, with each block being 1/4 mile, results in a total distance of 4.5 miles -/
theorem arthur_walk_distance :
  total_distance 8 10 (1/4) = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_arthur_walk_distance_l3997_399718


namespace NUMINAMATH_CALUDE_apples_per_box_l3997_399777

theorem apples_per_box (total_apples : ℕ) (num_boxes : ℕ) (apples_per_box : ℕ) : 
  total_apples = 49 → num_boxes = 7 → total_apples = num_boxes * apples_per_box → apples_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_box_l3997_399777


namespace NUMINAMATH_CALUDE_fraction_proportion_l3997_399759

theorem fraction_proportion (x y : ℚ) (h : y ≠ 0) :
  (x / y) / (2 / 5) = (3 / 7) / (6 / 5) → x / y = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proportion_l3997_399759


namespace NUMINAMATH_CALUDE_family_probability_l3997_399721

theorem family_probability :
  let p_boy : ℝ := 1/2
  let p_girl : ℝ := 1/2
  let num_children : ℕ := 4
  p_boy + p_girl = 1 →
  (1 : ℝ) - (p_boy ^ num_children + p_girl ^ num_children) = 7/8 :=
by sorry

end NUMINAMATH_CALUDE_family_probability_l3997_399721


namespace NUMINAMATH_CALUDE_perpendicular_vectors_sum_magnitude_l3997_399708

def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b : ℝ × ℝ := (1, -2)

theorem perpendicular_vectors_sum_magnitude (x : ℝ) :
  let a := vector_a x
  let b := vector_b
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- perpendicular condition
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_sum_magnitude_l3997_399708


namespace NUMINAMATH_CALUDE_sequence_a_10_l3997_399765

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ p q : ℕ, a (p + q) = a p * a q)

theorem sequence_a_10 (a : ℕ → ℝ) 
  (h_prop : sequence_property a) 
  (h_a8 : a 8 = 16) : 
  a 10 = 32 := by
sorry

end NUMINAMATH_CALUDE_sequence_a_10_l3997_399765


namespace NUMINAMATH_CALUDE_diamond_value_l3997_399749

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Converts a number in base 9 to base 10 -/
def base9To10 (d : Digit) : ℕ :=
  9 * d.val + 5

/-- Converts a number in base 10 to itself -/
def base10To10 (d : Digit) : ℕ :=
  10 * d.val + 2

theorem diamond_value :
  ∃ (d : Digit), base9To10 d = base10To10 d ∧ d.val = 3 := by sorry

end NUMINAMATH_CALUDE_diamond_value_l3997_399749


namespace NUMINAMATH_CALUDE_bella_steps_l3997_399755

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℕ := 15840

/-- Bella's speed relative to Ella's -/
def speed_ratio : ℕ := 4

/-- The number of feet Bella covers in one step -/
def feet_per_step : ℕ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps : ℕ := 1056

theorem bella_steps :
  distance * (speed_ratio + 1) / speed_ratio / feet_per_step = steps := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_l3997_399755


namespace NUMINAMATH_CALUDE_prob_same_outcome_equals_half_l3997_399761

-- Define the success probabilities for two independent events
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.8

-- Define the probability of both events resulting in the same outcome
def prob_same_outcome : ℝ := (prob_A * prob_B) + ((1 - prob_A) * (1 - prob_B))

-- Theorem statement
theorem prob_same_outcome_equals_half : prob_same_outcome = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_outcome_equals_half_l3997_399761


namespace NUMINAMATH_CALUDE_identify_genuine_coins_l3997_399715

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | Unequal : WeighResult

/-- Represents a set of coins -/
structure CoinSet where
  total : Nat
  fake : Nat
  h_fake_count : fake ≤ 1
  h_total : total = 101

/-- Represents a weighing action -/
def weighing (left right : Nat) : WeighResult :=
  sorry

/-- The main theorem to prove -/
theorem identify_genuine_coins (coins : CoinSet) :
  ∃ (genuine : Nat), genuine ≥ 50 ∧
    ∀ (left right : Nat),
      left + right ≤ coins.total →
      (weighing left right = WeighResult.Equal →
        genuine = left + right) ∧
      (weighing left right = WeighResult.Unequal →
        genuine = coins.total - (left + right)) :=
  sorry

end NUMINAMATH_CALUDE_identify_genuine_coins_l3997_399715


namespace NUMINAMATH_CALUDE_orchids_in_vase_l3997_399716

/-- Represents the number of roses initially in the vase -/
def initial_roses : ℕ := 9

/-- Represents the number of orchids initially in the vase -/
def initial_orchids : ℕ := 6

/-- Represents the number of roses in the vase now -/
def current_roses : ℕ := 3

/-- Represents the difference between the number of orchids and roses in the vase now -/
def orchid_rose_difference : ℕ := 10

/-- Represents the number of orchids in the vase now -/
def current_orchids : ℕ := current_roses + orchid_rose_difference

theorem orchids_in_vase : current_orchids = 13 := by sorry

end NUMINAMATH_CALUDE_orchids_in_vase_l3997_399716


namespace NUMINAMATH_CALUDE_problem_solution_l3997_399783

def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 2|

theorem problem_solution :
  (∃ (M : ℝ), (∀ x, f x ≥ M) ∧ (∃ x, f x = M) ∧ M = 3) ∧
  (∀ x, f x < 3 + |2*x + 2| ↔ -1 < x ∧ x < 2) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 2*b^2 = 3 → 2*a + b ≤ 3*Real.sqrt 6 / 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 2*b^2 = 3 ∧ 2*a + b = 3*Real.sqrt 6 / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3997_399783


namespace NUMINAMATH_CALUDE_simplify_fraction_l3997_399737

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3997_399737


namespace NUMINAMATH_CALUDE_predicted_distance_is_4km_l3997_399739

/-- Represents the cycling challenge scenario -/
structure CyclingChallenge where
  t : ℝ  -- Time taken to cycle first 1 km
  d : ℝ  -- Predicted distance for remaining time

/-- The cycling challenge satisfies the given conditions -/
def valid_challenge (c : CyclingChallenge) : Prop :=
  c.d = (60 - c.t) / c.t ∧  -- First prediction
  c.d = 384 / (c.t + 36)    -- Second prediction after cycling 15 km in 36 minutes

/-- The predicted distance is 4 km -/
theorem predicted_distance_is_4km (c : CyclingChallenge) 
  (h : valid_challenge c) : c.d = 4 := by
  sorry

#check predicted_distance_is_4km

end NUMINAMATH_CALUDE_predicted_distance_is_4km_l3997_399739


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3997_399714

theorem unique_solution_condition (a c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + 2) ↔ c ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3997_399714


namespace NUMINAMATH_CALUDE_topsoil_cost_l3997_399774

-- Define the cost per cubic foot of topsoil
def cost_per_cubic_foot : ℝ := 8

-- Define the conversion factor from cubic yards to cubic feet
def cubic_yards_to_cubic_feet : ℝ := 27

-- Define the volume in cubic yards
def volume_in_cubic_yards : ℝ := 7

-- Theorem statement
theorem topsoil_cost : 
  volume_in_cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l3997_399774


namespace NUMINAMATH_CALUDE_exclusive_or_implies_possible_p_true_q_false_l3997_399725

theorem exclusive_or_implies_possible_p_true_q_false (P Q : Prop) 
  (h1 : P ∨ Q) (h2 : ¬(P ∧ Q)) : 
  ∃ (p q : Prop), p = P ∧ q = Q ∧ p = true ∧ q = false :=
sorry

end NUMINAMATH_CALUDE_exclusive_or_implies_possible_p_true_q_false_l3997_399725


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3997_399771

def S (n : ℕ) (m : ℝ) : ℝ := 3^(n+1) + m

def a (n : ℕ) (m : ℝ) : ℝ :=
  if n = 1 then S 1 m
  else S n m - S (n-1) m

theorem geometric_sequence_condition (m : ℝ) :
  (∀ n : ℕ, n ≥ 2 → (a (n+1) m) * (a (n-1) m) = (a n m)^2) ↔ m = -3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3997_399771


namespace NUMINAMATH_CALUDE_beam_travel_time_l3997_399710

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length 4 -/
structure Square where
  A : Point := ⟨0, 0⟩
  B : Point := ⟨4, 0⟩
  C : Point := ⟨4, 4⟩
  D : Point := ⟨0, 4⟩

/-- The beam's path in the square -/
structure BeamPath (s : Square) where
  F : Point
  E : Point
  BE : ℝ
  EF : ℝ
  FC : ℝ
  speed : ℝ

/-- Theorem stating the time taken for the beam to travel from F to E -/
theorem beam_travel_time (s : Square) (path : BeamPath s) 
  (h1 : path.BE = 2)
  (h2 : path.EF = 2)
  (h3 : path.FC = 2)
  (h4 : path.speed = 1)
  (h5 : path.E = ⟨2, 0⟩) :
  ∃ t : ℝ, t = 2 * Real.sqrt 61 ∧ 
    t * path.speed = Real.sqrt ((10 - path.F.x)^2 + (6 - path.F.y)^2) := by
  sorry

end NUMINAMATH_CALUDE_beam_travel_time_l3997_399710


namespace NUMINAMATH_CALUDE_circle_radii_order_l3997_399766

theorem circle_radii_order (r_A r_B r_C : ℝ) : 
  r_A = 2 →
  2 * Real.pi * r_B = 10 * Real.pi →
  Real.pi * r_C^2 = 16 * Real.pi →
  r_A < r_C ∧ r_C < r_B := by
sorry

end NUMINAMATH_CALUDE_circle_radii_order_l3997_399766


namespace NUMINAMATH_CALUDE_equal_distance_point_sum_of_distances_equal_distance_time_l3997_399731

def A : ℝ := -1
def B : ℝ := 3

theorem equal_distance_point (x : ℝ) : 
  |x - A| = |x - B| → x = 1 := by sorry

theorem sum_of_distances (x : ℝ) : 
  (|x - A| + |x - B| = 5) ↔ (x = -3/2 ∨ x = 7/2) := by sorry

def P (t : ℝ) : ℝ := -t
def A' (t : ℝ) : ℝ := -1 - 5*t
def B' (t : ℝ) : ℝ := 3 - 20*t

theorem equal_distance_time (t : ℝ) : 
  |P t - A' t| = |P t - B' t| ↔ (t = 4/15 ∨ t = 2/23) := by sorry

end NUMINAMATH_CALUDE_equal_distance_point_sum_of_distances_equal_distance_time_l3997_399731


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_l3997_399705

theorem mean_equality_implies_y_value :
  let mean1 := (6 + 9 + 18) / 3
  let mean2 := (12 + y) / 2
  mean1 = mean2 → y = 10 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_l3997_399705


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3997_399732

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a = (1, m) and b = (3, 1), prove that if they are parallel, then m = 1/3 -/
theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (1, m) (3, 1) → m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3997_399732


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l3997_399794

/-- 
Calculates the number of games in a single-elimination tournament.
num_teams: The number of teams in the tournament.
-/
def num_games (num_teams : ℕ) : ℕ :=
  num_teams - 1

theorem single_elimination_tournament_games :
  num_games 16 = 15 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l3997_399794


namespace NUMINAMATH_CALUDE_park_area_l3997_399760

/-- The area of a rectangular park with a modified perimeter -/
theorem park_area (l w : ℝ) (h1 : 2 * l + 2 * w + 5 = 80) (h2 : l = 3 * w) :
  l * w = 263.6719 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l3997_399760


namespace NUMINAMATH_CALUDE_burger_calories_l3997_399709

/-- Calculates the number of calories per burger given the following conditions:
  * 10 burritos cost $6
  * Each burrito has 120 calories
  * 5 burgers cost $8
  * Burgers provide 50 more calories per dollar than burritos
-/
theorem burger_calories :
  let burrito_count : ℕ := 10
  let burrito_cost : ℚ := 6
  let burrito_calories : ℕ := 120
  let burger_count : ℕ := 5
  let burger_cost : ℚ := 8
  let calorie_difference_per_dollar : ℕ := 50
  
  let burrito_calories_per_dollar : ℚ := (burrito_count * burrito_calories : ℚ) / burrito_cost
  let burger_calories_per_dollar : ℚ := burrito_calories_per_dollar + calorie_difference_per_dollar
  let total_burger_calories : ℚ := burger_calories_per_dollar * burger_cost
  let calories_per_burger : ℚ := total_burger_calories / burger_count
  
  calories_per_burger = 400 := by
    sorry

end NUMINAMATH_CALUDE_burger_calories_l3997_399709


namespace NUMINAMATH_CALUDE_rafting_and_tubing_count_l3997_399795

theorem rafting_and_tubing_count (total_kids : ℕ) 
  (h1 : total_kids = 40) 
  (tubing_fraction : ℚ) 
  (h2 : tubing_fraction = 1/4) 
  (rafting_fraction : ℚ) 
  (h3 : rafting_fraction = 1/2) : ℕ :=
  let tubing_kids := (total_kids : ℚ) * tubing_fraction
  let rafting_and_tubing_kids := tubing_kids * rafting_fraction
  5

#check rafting_and_tubing_count

end NUMINAMATH_CALUDE_rafting_and_tubing_count_l3997_399795


namespace NUMINAMATH_CALUDE_same_direction_condition_l3997_399778

/-- Two vectors are in the same direction if one is a positive scalar multiple of the other -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m > 0 ∧ a = (m * b.1, m * b.2)

/-- The condition for vectors a and b to be in the same direction -/
theorem same_direction_condition (k : ℝ) :
  same_direction (k, 2) (1, 1) ↔ k = 2 := by
  sorry

#check same_direction_condition

end NUMINAMATH_CALUDE_same_direction_condition_l3997_399778


namespace NUMINAMATH_CALUDE_cost_per_topping_is_two_l3997_399711

/-- Represents the cost of a pizza order with toppings and tip --/
def pizza_order_cost (large_pizza_cost : ℝ) (num_pizzas : ℕ) (toppings_per_pizza : ℕ) 
  (tip_percentage : ℝ) (topping_cost : ℝ) : ℝ :=
  let base_cost := large_pizza_cost * num_pizzas
  let total_toppings := num_pizzas * toppings_per_pizza
  let toppings_cost := total_toppings * topping_cost
  let subtotal := base_cost + toppings_cost
  let tip := subtotal * tip_percentage
  subtotal + tip

/-- The cost per topping is $2 --/
theorem cost_per_topping_is_two :
  ∃ (topping_cost : ℝ),
    pizza_order_cost 14 2 3 0.25 topping_cost = 50 ∧ 
    topping_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_topping_is_two_l3997_399711


namespace NUMINAMATH_CALUDE_school_election_votes_l3997_399741

theorem school_election_votes (randy_votes shaun_votes eliot_votes : ℕ) : 
  randy_votes = 16 →
  shaun_votes = 5 * randy_votes →
  eliot_votes = 2 * shaun_votes →
  eliot_votes = 160 := by
  sorry

end NUMINAMATH_CALUDE_school_election_votes_l3997_399741


namespace NUMINAMATH_CALUDE_min_abs_z_l3997_399784

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 2) + Complex.abs (z - 7*I) = 10) :
  Complex.abs z ≥ 1.4 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_l3997_399784


namespace NUMINAMATH_CALUDE_students_liking_both_sports_l3997_399746

theorem students_liking_both_sports (basketball : ℕ) (cricket : ℕ) (total : ℕ) : 
  basketball = 12 → cricket = 8 → total = 17 → 
  basketball + cricket - total = 3 := by
sorry

end NUMINAMATH_CALUDE_students_liking_both_sports_l3997_399746


namespace NUMINAMATH_CALUDE_product_of_xy_l3997_399768

theorem product_of_xy (x y z w : ℕ+) 
  (h1 : x = w)
  (h2 : y = z)
  (h3 : w + w = w * w)
  (h4 : y = w)
  (h5 : z = 3) :
  x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_xy_l3997_399768


namespace NUMINAMATH_CALUDE_problem_1_l3997_399717

theorem problem_1 (x : ℝ) : 4 * (x + 1)^2 = 49 → x = 5/2 ∨ x = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3997_399717


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l3997_399720

-- Define the operation
noncomputable def bowtie (c x : ℝ) : ℝ := c + Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 5 x = 11 ∧ x = 30 := by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l3997_399720


namespace NUMINAMATH_CALUDE_eraser_buyers_difference_l3997_399751

theorem eraser_buyers_difference : ∀ (price : ℕ) (fifth_graders fourth_graders : ℕ),
  price > 0 →
  fifth_graders * price = 325 →
  fourth_graders * price = 460 →
  fourth_graders = 40 →
  fourth_graders - fifth_graders = 27 := by
sorry

end NUMINAMATH_CALUDE_eraser_buyers_difference_l3997_399751


namespace NUMINAMATH_CALUDE_cubic_expansion_simplification_l3997_399707

theorem cubic_expansion_simplification :
  (30 + 5)^3 - (30^3 + 3*30^2*5 + 3*30*5^2 + 5^3 - 5^3) = 125 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_simplification_l3997_399707


namespace NUMINAMATH_CALUDE_smallest_b_value_l3997_399780

theorem smallest_b_value (a b : ℝ) (h1 : 1 < a) (h2 : a < b)
  (h3 : b ≥ a + 1)
  (h4 : 1 / b + 1 / a ≤ 1) :
  b ≥ (3 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3997_399780


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3997_399779

/-- A rhombus with given diagonal lengths has a specific perimeter -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l3997_399779


namespace NUMINAMATH_CALUDE_tan_alpha_minus_2beta_l3997_399713

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2/5)
  (h2 : Real.tan β = 1/2) : 
  Real.tan (α - 2*β) = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_2beta_l3997_399713


namespace NUMINAMATH_CALUDE_soda_drinkers_l3997_399793

theorem soda_drinkers (total : ℕ) (wine : ℕ) (both : ℕ) (soda : ℕ) : 
  total = 31 → wine = 26 → both = 17 → soda = total + both - wine := by
  sorry

end NUMINAMATH_CALUDE_soda_drinkers_l3997_399793


namespace NUMINAMATH_CALUDE_rectangle_perimeter_problem_l3997_399740

theorem rectangle_perimeter_problem : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    a > 0 ∧ 
    b > 0 ∧ 
    (a * b : ℕ) = 2 * (2 * a + 2 * b) ∧ 
    2 * (a + b) = 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_problem_l3997_399740


namespace NUMINAMATH_CALUDE_cosine_value_l3997_399782

theorem cosine_value (α : Real) 
  (h : Real.cos (α - π/6) - Real.sin α = 2 * Real.sqrt 3 / 5) : 
  Real.cos (α + 7*π/6) = -(2 * Real.sqrt 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_l3997_399782


namespace NUMINAMATH_CALUDE_intersection_when_m_is_5_intersection_equals_B_iff_l3997_399747

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 9}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Statement 1: When m = 5, A ∩ B = {x | 6 ≤ x < 9}
theorem intersection_when_m_is_5 : 
  A ∩ B 5 = {x : ℝ | 6 ≤ x ∧ x < 9} := by sorry

-- Statement 2: A ∩ B = B if and only if m ∈ (-∞, 5)
theorem intersection_equals_B_iff :
  ∀ m : ℝ, A ∩ B m = B m ↔ m < 5 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_5_intersection_equals_B_iff_l3997_399747


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3997_399762

theorem smallest_positive_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 5 = 4) ∧ 
  (a % 7 = 6) ∧ 
  (∀ b : ℕ, b > 0 ∧ b % 5 = 4 ∧ b % 7 = 6 → a ≤ b) ∧
  (a = 34) := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3997_399762


namespace NUMINAMATH_CALUDE_three_digit_base15_double_l3997_399700

/-- A function that converts a number from base 10 to base 15 --/
def toBase15 (n : ℕ) : ℕ :=
  (n / 100) * 15^2 + ((n / 10) % 10) * 15 + (n % 10)

/-- The set of three-digit numbers that satisfy the condition --/
def validNumbers : Finset ℕ := {150, 145, 290}

/-- The property that a number, when converted to base 15, is twice its original value --/
def satisfiesCondition (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ toBase15 n = 2 * n

theorem three_digit_base15_double :
  ∀ n : ℕ, satisfiesCondition n ↔ n ∈ validNumbers :=
sorry


end NUMINAMATH_CALUDE_three_digit_base15_double_l3997_399700


namespace NUMINAMATH_CALUDE_age_difference_l3997_399730

/-- Proves that given a man and his son, where the son's present age is 24 and in two years
    the man's age will be twice his son's age, the difference between their present ages is 26 years. -/
theorem age_difference (man_age son_age : ℕ) : 
  son_age = 24 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3997_399730


namespace NUMINAMATH_CALUDE_shortest_distance_proof_l3997_399734

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 - y - 2 * Real.log (Real.sqrt x) = 0

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 4*y + 1 = 0

-- Define the shortest distance function
noncomputable def shortest_distance : ℝ := (Real.sqrt 2 / 2) * (1 + Real.log 2)

-- Theorem statement
theorem shortest_distance_proof :
  ∀ (x y : ℝ), curve x y →
  ∃ (d : ℝ), d ≥ 0 ∧ 
    (∀ (x' y' : ℝ), line x' y' → 
      d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) ∧
    d = shortest_distance :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_proof_l3997_399734


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3997_399785

-- Define set M
def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}

-- Define set N
def N : Set ℝ := {x : ℝ | x < -5 ∨ x > 5}

-- Theorem statement
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < -5 ∨ x > -3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3997_399785


namespace NUMINAMATH_CALUDE_part_one_part_two_l3997_399724

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (x : ℝ) : 
  let a := 2
  (x ≤ -1/2 ∨ x ≥ 7/2) ↔ f a x ≥ 4 - |x - 1| :=
sorry

-- Part II
theorem part_two (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f 1 x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  1/m + 1/(2*n) = 1 →
  m + 2*n ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3997_399724


namespace NUMINAMATH_CALUDE_parabola_one_x_intercept_l3997_399723

-- Define the parabola function
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- Theorem: The parabola has exactly one x-intercept
theorem parabola_one_x_intercept : 
  ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_one_x_intercept_l3997_399723


namespace NUMINAMATH_CALUDE_money_left_calculation_l3997_399787

theorem money_left_calculation (initial_amount spent_on_sweets given_to_each_friend : ℚ) 
  (number_of_friends : ℕ) (h1 : initial_amount = 200.50) 
  (h2 : spent_on_sweets = 35.25) (h3 : given_to_each_friend = 25.20) 
  (h4 : number_of_friends = 2) : 
  initial_amount - spent_on_sweets - (given_to_each_friend * number_of_friends) = 114.85 := by
  sorry

end NUMINAMATH_CALUDE_money_left_calculation_l3997_399787


namespace NUMINAMATH_CALUDE_trajectory_length_l3997_399758

/-- The curve y = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The line x = 2 on which point A moves -/
def line_x_eq_2 (x : ℝ) : Prop := x = 2

/-- The tangent line to the curve at point (x₀, f x₀) -/
def tangent_line (x₀ : ℝ) (x a : ℝ) : Prop :=
  a = (3 * x₀^2 - 1) * (x - x₀) + f x₀

/-- The condition for point A(2, a) to have a tangent line to the curve -/
def has_tangent (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, tangent_line x₀ 2 a

/-- The statement to be proved -/
theorem trajectory_length :
  ∀ a : ℝ, line_x_eq_2 2 →
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    has_tangent a ∧
    (∀ x : ℝ, has_tangent a → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  (∃ a_min a_max : ℝ, 
    (∀ a' : ℝ, has_tangent a' → a_min ≤ a' ∧ a' ≤ a_max) ∧
    a_max - a_min = 8) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_length_l3997_399758


namespace NUMINAMATH_CALUDE_fraction_power_four_l3997_399727

theorem fraction_power_four : (5 / 3 : ℚ) ^ 4 = 625 / 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_four_l3997_399727


namespace NUMINAMATH_CALUDE_functional_polynomial_form_l3997_399775

/-- A polynomial that satisfies the given functional equation. -/
structure FunctionalPolynomial where
  P : ℝ → ℝ
  nonzero : P ≠ 0
  satisfies_equation : ∀ x : ℝ, P (x^2 - 2*x) = (P (x - 2))^2

/-- The theorem stating the form of polynomials satisfying the functional equation. -/
theorem functional_polynomial_form (fp : FunctionalPolynomial) :
  ∃ n : ℕ, n > 0 ∧ ∀ x : ℝ, fp.P x = (x + 1)^n :=
sorry

end NUMINAMATH_CALUDE_functional_polynomial_form_l3997_399775


namespace NUMINAMATH_CALUDE_extended_altitude_triangle_l3997_399767

/-- Given a triangle ABC with sides a, b, c, angles α, β, γ, and area t,
    we extend its altitudes beyond the sides by their own lengths to form
    a new triangle A'B'C' with sides a', b', c' and area t'. -/
theorem extended_altitude_triangle
  (a b c a' b' c' : ℝ)
  (α β γ : ℝ)
  (t t' : ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧ β > 0 ∧ γ > 0)
  (h_angles : α + β + γ = Real.pi)
  (h_area : t = (1/2) * a * b * Real.sin γ)
  (h_extended : a' > a ∧ b' > b ∧ c' > c) :
  (a'^2 + b'^2 + c'^2 - (a^2 + b^2 + c^2) = 32 * t * Real.sin α * Real.sin β * Real.sin γ) ∧
  (t' = t * (3 + 8 * Real.cos α * Real.cos β * Real.cos γ)) := by
  sorry

end NUMINAMATH_CALUDE_extended_altitude_triangle_l3997_399767


namespace NUMINAMATH_CALUDE_mailman_theorem_l3997_399738

def mailman_problem (mails_per_block : ℕ) (houses_per_block : ℕ) : Prop :=
  mails_per_block / houses_per_block = 8

theorem mailman_theorem :
  mailman_problem 32 4 :=
by
  sorry

end NUMINAMATH_CALUDE_mailman_theorem_l3997_399738


namespace NUMINAMATH_CALUDE_three_in_A_not_in_B_l3997_399743

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define the complement of A in U
def complement_A : Finset Nat := {2, 4}

-- Define the complement of B in U
def complement_B : Finset Nat := {3, 4}

-- Define set A
def A : Finset Nat := U \ complement_A

-- Define set B
def B : Finset Nat := U \ complement_B

-- Theorem statement
theorem three_in_A_not_in_B : 3 ∈ A ∧ 3 ∉ B := by
  sorry

end NUMINAMATH_CALUDE_three_in_A_not_in_B_l3997_399743


namespace NUMINAMATH_CALUDE_alex_cell_phone_cost_l3997_399752

/-- Calculates the total cost of a cell phone plan -/
def total_cost (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
                (num_texts : ℕ) (hours_talked : ℕ) : ℚ :=
  let text_charge := (num_texts : ℚ) * text_cost
  let extra_hours := max ((hours_talked : ℤ) - 30) 0
  let extra_minutes := (extra_hours : ℚ) * 60
  let extra_minute_charge := extra_minutes * extra_minute_cost
  base_cost + text_charge + extra_minute_charge

/-- The total cost of Alex's cell phone plan is $48.50 -/
theorem alex_cell_phone_cost : 
  total_cost 20 (7/100) (15/100) 150 32 = 485/10 := by
  sorry

end NUMINAMATH_CALUDE_alex_cell_phone_cost_l3997_399752


namespace NUMINAMATH_CALUDE_prime_factors_of_1998_l3997_399706

theorem prime_factors_of_1998 (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a < b ∧ b < c ∧
  a * b * c = 1998 →
  (b + c)^a = 1600 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_of_1998_l3997_399706


namespace NUMINAMATH_CALUDE_color_copies_equation_l3997_399728

/-- The cost per color copy at print shop X -/
def cost_X : ℝ := 1.20

/-- The cost per color copy at print shop Y -/
def cost_Y : ℝ := 1.70

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℝ := 20

/-- 
Given:
- Print shop X charges $1.20 per color copy
- Print shop Y charges $1.70 per color copy
- The charge for a certain number of color copies at print shop Y is $20 greater than at print shop X

Prove that the number of color copies n satisfies the equation:
1.70n = 1.20n + 20
-/
theorem color_copies_equation (n : ℝ) : 
  cost_Y * n = cost_X * n + additional_charge ↔ n = 40 :=
sorry

end NUMINAMATH_CALUDE_color_copies_equation_l3997_399728


namespace NUMINAMATH_CALUDE_xy_and_x_minus_y_squared_l3997_399756

theorem xy_and_x_minus_y_squared (x y : ℝ) 
  (sum_eq : x + y = 5) 
  (sum_squares_eq : x^2 + y^2 = 15) : 
  x * y = 5 ∧ (x - y)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_xy_and_x_minus_y_squared_l3997_399756


namespace NUMINAMATH_CALUDE_binomial_20_choose_6_l3997_399702

theorem binomial_20_choose_6 : Nat.choose 20 6 = 19380 := by sorry

end NUMINAMATH_CALUDE_binomial_20_choose_6_l3997_399702


namespace NUMINAMATH_CALUDE_external_diagonals_theorem_l3997_399763

/-- Checks if a set of three numbers could be the lengths of external diagonals of a right regular prism -/
def valid_external_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 ≥ c^2 ∧
  b^2 + c^2 ≥ a^2 ∧
  a^2 + c^2 ≥ b^2

theorem external_diagonals_theorem :
  ¬(valid_external_diagonals 3 4 6) ∧
  valid_external_diagonals 3 4 5 ∧
  valid_external_diagonals 5 6 8 ∧
  valid_external_diagonals 5 8 9 ∧
  valid_external_diagonals 6 8 10 :=
by sorry

end NUMINAMATH_CALUDE_external_diagonals_theorem_l3997_399763


namespace NUMINAMATH_CALUDE_max_value_of_a_l3997_399796

theorem max_value_of_a (a b c d : ℕ) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 150) : 
  a ≤ 8924 ∧ ∃ (a' b' c' d' : ℕ), a' = 8924 ∧ a' < 3 * b' ∧ b' < 4 * c' ∧ c' < 5 * d' ∧ d' < 150 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3997_399796


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l3997_399753

/-- The line equation 5y - 6x = 15 intersects the x-axis at the point (-2.5, 0) -/
theorem line_intersects_x_axis :
  ∃ (x : ℚ), x = -2.5 ∧ 5 * 0 - 6 * x = 15 := by sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l3997_399753


namespace NUMINAMATH_CALUDE_inequalities_proof_l3997_399764

theorem inequalities_proof (a b : ℝ) (h : a > b) (h0 : b > 0) :
  (Real.sqrt a > Real.sqrt b) ∧ (a - 1/a > b - 1/b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3997_399764


namespace NUMINAMATH_CALUDE_hiking_trail_length_l3997_399769

/-- Represents the hiking trail problem -/
def HikingTrail :=
  {length : ℝ // length > 0}

/-- The total time for the round trip in hours -/
def totalTime : ℝ := 3

/-- The uphill speed in km/h -/
def uphillSpeed : ℝ := 2

/-- The downhill speed in km/h -/
def downhillSpeed : ℝ := 4

/-- Theorem stating that the length of the hiking trail is 4 km -/
theorem hiking_trail_length :
  ∃ (trail : HikingTrail),
    (trail.val / uphillSpeed + trail.val / downhillSpeed = totalTime) ∧
    trail.val = 4 := by sorry

end NUMINAMATH_CALUDE_hiking_trail_length_l3997_399769


namespace NUMINAMATH_CALUDE_x_value_proof_l3997_399788

theorem x_value_proof (x : ℝ) (h : 9 / x^3 = x / 27) : x = 3 * (3 ^ (1/4)) := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3997_399788


namespace NUMINAMATH_CALUDE_point_transformation_l3997_399786

/-- Rotation of a point (x, y) by 180° around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ := (2*h - x, 2*k - y)

/-- Reflection of a point (x, y) about y = x -/
def reflectYEqualX (x y : ℝ) : ℝ × ℝ := (y, x)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let Q : ℝ × ℝ := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectYEqualX rotated.1 rotated.2
  final = (3, -7) → a - b = 8 := by
sorry

end NUMINAMATH_CALUDE_point_transformation_l3997_399786


namespace NUMINAMATH_CALUDE_rachel_final_lives_l3997_399742

/-- Calculates the total number of lives after losing and gaining lives in a video game. -/
def totalLives (initialLives livesLost livesGained : ℕ) : ℕ :=
  initialLives - livesLost + livesGained

/-- Proves that given the initial conditions, Rachel ends up with 32 lives. -/
theorem rachel_final_lives :
  totalLives 10 4 26 = 32 := by
  sorry

#eval totalLives 10 4 26

end NUMINAMATH_CALUDE_rachel_final_lives_l3997_399742
