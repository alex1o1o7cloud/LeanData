import Mathlib

namespace ellipse_standard_equation_l3284_328443

/-- The standard equation of an ellipse with specific parameters. -/
theorem ellipse_standard_equation 
  (foci_on_y_axis : Bool) 
  (major_axis_length : ℝ) 
  (eccentricity : ℝ) : 
  foci_on_y_axis ∧ major_axis_length = 20 ∧ eccentricity = 2/5 → 
  ∃ (x y : ℝ), y^2/100 + x^2/84 = 1 :=
by sorry

end ellipse_standard_equation_l3284_328443


namespace limit_sine_cosine_ratio_l3284_328494

theorem limit_sine_cosine_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |(1 + Real.sin (2*x) - Real.cos (2*x)) / (1 - Real.sin (2*x) - Real.cos (2*x)) + 1| < ε :=
by sorry

end limit_sine_cosine_ratio_l3284_328494


namespace angle_supplement_l3284_328441

theorem angle_supplement (α : ℝ) : 
  (90 - α = 125) → (180 - α = 125) := by
  sorry

end angle_supplement_l3284_328441


namespace product_xy_on_line_k_l3284_328489

/-- A line passing through the origin with slope 1/4 -/
def line_k (x y : ℝ) : Prop := y = (1/4) * x

theorem product_xy_on_line_k :
  ∀ x y : ℝ,
  line_k x 8 → line_k 20 y →
  x * y = 160 := by
sorry

end product_xy_on_line_k_l3284_328489


namespace quadratic_form_sum_l3284_328476

theorem quadratic_form_sum (x : ℝ) : ∃ (b c : ℝ), 
  2*x^2 - 28*x + 50 = (x + b)^2 + c ∧ b + c = -55 := by
  sorry

end quadratic_form_sum_l3284_328476


namespace N_value_l3284_328434

theorem N_value : 
  let N := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2) - Real.sqrt (4 - 2 * Real.sqrt 3)
  N = (1 + Real.sqrt 6 - Real.sqrt 3) / 2 := by
sorry

end N_value_l3284_328434


namespace unique_positive_solution_l3284_328454

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ 3 * x^2 - 7 * x - 6 = 0 :=
by
  sorry

end unique_positive_solution_l3284_328454


namespace pirate_treasure_distribution_l3284_328409

/-- Represents the number of coins in the final distribution step -/
def x : ℕ := sorry

/-- Pete's coin distribution pattern -/
def petes_coins (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Polly's final coin count -/
def pollys_coins : ℕ := x

/-- Pete's final coin count -/
def petes_final_coins : ℕ := 3 * x

theorem pirate_treasure_distribution :
  petes_coins x = petes_final_coins ∧
  pollys_coins + petes_final_coins = 20 := by sorry

end pirate_treasure_distribution_l3284_328409


namespace underdog_wins_in_nine_games_l3284_328428

/- Define the probability of the favored team winning a single game -/
def p : ℚ := 3/4

/- Define the number of games needed to win the series -/
def games_to_win : ℕ := 5

/- Define the maximum number of games in the series -/
def max_games : ℕ := 9

/- Define the probability of the underdog team winning a single game -/
def q : ℚ := 1 - p

/- Define the number of ways to choose 4 games out of 8 -/
def ways_to_choose : ℕ := Nat.choose 8 4

theorem underdog_wins_in_nine_games :
  (ways_to_choose : ℚ) * q^4 * p^4 * q = 5670/262144 := by
  sorry

end underdog_wins_in_nine_games_l3284_328428


namespace square_area_after_cut_l3284_328424

theorem square_area_after_cut (x : ℝ) : 
  x > 0 → x * (x - 3) = 40 → x^2 = 64 := by
  sorry

end square_area_after_cut_l3284_328424


namespace staircase_perimeter_l3284_328491

/-- Represents a staircase-shaped region -/
structure StaircaseRegion where
  congruent_sides : ℕ
  rectangle_length : ℝ
  area : ℝ

/-- The perimeter of a staircase-shaped region -/
def perimeter (region : StaircaseRegion) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific staircase region -/
theorem staircase_perimeter : 
  ∀ (region : StaircaseRegion), 
    region.congruent_sides = 12 ∧ 
    region.rectangle_length = 12 ∧ 
    region.area = 85 → 
    perimeter region = 41 :=
by sorry

end staircase_perimeter_l3284_328491


namespace function_difference_implies_m_value_l3284_328462

theorem function_difference_implies_m_value :
  ∀ (m : ℝ),
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - 3 * x + 5
  let g : ℝ → ℝ := λ x ↦ x^2 - m * x - 8
  (f 5 - g 5 = 15) → (m = -15) :=
by
  sorry

end function_difference_implies_m_value_l3284_328462


namespace geometric_sequence_product_threshold_l3284_328417

theorem geometric_sequence_product_threshold (n : ℕ) : 
  (n > 0 ∧ 3^((n * (n + 1)) / 12) > 1000) ↔ n ≥ 6 := by
  sorry

end geometric_sequence_product_threshold_l3284_328417


namespace max_factors_bound_l3284_328401

/-- The number of positive factors of b^n, where b and n are positive integers with b ≤ 20 and n ≤ 20 -/
def max_factors (b n : ℕ+) : ℕ :=
  if b ≤ 20 ∧ n ≤ 20 then
    -- Placeholder for the actual calculation of factors
    0
  else
    0

/-- The maximum number of positive factors of b^n is 861, where b and n are positive integers with b ≤ 20 and n ≤ 20 -/
theorem max_factors_bound :
  ∃ (b n : ℕ+), b ≤ 20 ∧ n ≤ 20 ∧ max_factors b n = 861 ∧
  ∀ (b' n' : ℕ+), b' ≤ 20 → n' ≤ 20 → max_factors b' n' ≤ 861 :=
sorry

end max_factors_bound_l3284_328401


namespace range_of_m_l3284_328480

/-- A function f is decreasing on (0, +∞) -/
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f y < f x

/-- The solution set of (x-1)² > m is ℝ -/
def SolutionSetIsReals (m : ℝ) : Prop :=
  ∀ x, (x - 1)^2 > m

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : IsDecreasingOn f)
  (h2 : SolutionSetIsReals m)
  (h3 : (IsDecreasingOn f) ∨ (SolutionSetIsReals m))
  (h4 : ¬((IsDecreasingOn f) ∧ (SolutionSetIsReals m))) :
  0 ≤ m ∧ m < (1/2) := by
  sorry

end range_of_m_l3284_328480


namespace initial_books_in_bin_l3284_328498

theorem initial_books_in_bin (initial_books sold_books added_books final_books : ℕ) :
  sold_books = 3 →
  added_books = 10 →
  final_books = 11 →
  initial_books - sold_books + added_books = final_books →
  initial_books = 4 := by
  sorry

end initial_books_in_bin_l3284_328498


namespace village_population_is_72_l3284_328421

/-- The number of people a vampire drains per week -/
def vampire_drain_rate : ℕ := 3

/-- The number of people a werewolf eats per week -/
def werewolf_eat_rate : ℕ := 5

/-- The number of weeks the village lasts -/
def weeks_lasted : ℕ := 9

/-- The total number of people in the village -/
def village_population : ℕ := vampire_drain_rate * weeks_lasted + werewolf_eat_rate * weeks_lasted

theorem village_population_is_72 : village_population = 72 := by
  sorry

end village_population_is_72_l3284_328421


namespace volume_per_part_l3284_328437

/-- Given two rectangular prisms and a number of equal parts filling these prisms,
    calculate the volume of each part. -/
theorem volume_per_part
  (length width height : ℝ)
  (num_prisms num_parts : ℕ)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_num_prisms : num_prisms = 2)
  (h_num_parts : num_parts = 16) :
  (num_prisms * length * width * height) / num_parts = 4 := by
  sorry

end volume_per_part_l3284_328437


namespace min_rotation_angle_is_72_l3284_328486

/-- A regular five-pointed star -/
structure RegularFivePointedStar where
  -- Add necessary properties here

/-- The minimum rotation angle for a regular five-pointed star to coincide with its original position -/
def min_rotation_angle (star : RegularFivePointedStar) : ℝ :=
  72

/-- Theorem stating that the minimum rotation angle for a regular five-pointed star 
    to coincide with its original position is 72 degrees -/
theorem min_rotation_angle_is_72 (star : RegularFivePointedStar) :
  min_rotation_angle star = 72 := by
  sorry

end min_rotation_angle_is_72_l3284_328486


namespace greatest_common_multiple_9_15_under_150_l3284_328425

theorem greatest_common_multiple_9_15_under_150 :
  ∃ n : ℕ, n = 135 ∧ 
  (∀ m : ℕ, m < 150 → m % 9 = 0 → m % 15 = 0 → m ≤ n) ∧
  135 % 9 = 0 ∧ 135 % 15 = 0 ∧ 135 < 150 :=
by sorry

end greatest_common_multiple_9_15_under_150_l3284_328425


namespace quadratic_rewrite_sum_l3284_328468

/-- Given a quadratic expression x^2 - 16x + 15, when rewritten in the form (x+d)^2 + e,
    the sum of d and e is -57. -/
theorem quadratic_rewrite_sum (d e : ℝ) : 
  (∀ x, x^2 - 16*x + 15 = (x+d)^2 + e) → d + e = -57 := by
  sorry

end quadratic_rewrite_sum_l3284_328468


namespace reciprocal_of_product_l3284_328438

theorem reciprocal_of_product : (((1 : ℚ) / 3) * (3 / 4))⁻¹ = 4 := by sorry

end reciprocal_of_product_l3284_328438


namespace min_shots_to_hit_ship_l3284_328493

/-- Represents a point on the game board -/
structure Point where
  x : Fin 10
  y : Fin 10

/-- Represents a ship on the game board -/
inductive Ship
  | Horizontal : Fin 10 → Fin 7 → Ship
  | Vertical : Fin 7 → Fin 10 → Ship

/-- Checks if a point is on a ship -/
def pointOnShip (p : Point) (s : Ship) : Prop :=
  match s with
  | Ship.Horizontal row col => p.y = row ∧ col ≤ p.x ∧ p.x < col + 4
  | Ship.Vertical row col => p.x = col ∧ row ≤ p.y ∧ p.y < row + 4

/-- The theorem to be proved -/
theorem min_shots_to_hit_ship :
  ∃ (shots : Finset Point),
    shots.card = 14 ∧
    ∀ (s : Ship), ∃ (p : Point), p ∈ shots ∧ pointOnShip p s ∧
    ∀ (shots' : Finset Point),
      shots'.card < 14 →
      ∃ (s : Ship), ∀ (p : Point), p ∈ shots' → ¬pointOnShip p s :=
by sorry

end min_shots_to_hit_ship_l3284_328493


namespace sum_vector_magnitude_l3284_328423

/-- Given two vectors a and b in ℝ³, prove that their sum has magnitude √26 -/
theorem sum_vector_magnitude (a b : ℝ × ℝ × ℝ) : 
  a = (1, -1, 0) → b = (3, -2, 1) → ‖a + b‖ = Real.sqrt 26 := by
  sorry

end sum_vector_magnitude_l3284_328423


namespace book_sale_gain_percentage_l3284_328402

theorem book_sale_gain_percentage (initial_sale_price : ℝ) (loss_percentage : ℝ) (desired_sale_price : ℝ) : 
  initial_sale_price = 810 →
  loss_percentage = 10 →
  desired_sale_price = 990 →
  let cost_price := initial_sale_price / (1 - loss_percentage / 100)
  let gain := desired_sale_price - cost_price
  let gain_percentage := (gain / cost_price) * 100
  gain_percentage = 10 := by
sorry

end book_sale_gain_percentage_l3284_328402


namespace divisibility_equivalence_l3284_328467

theorem divisibility_equivalence (x y : ℤ) : 
  (7 ∣ (2*x + 3*y)) ↔ (7 ∣ (5*x + 4*y)) := by
  sorry

end divisibility_equivalence_l3284_328467


namespace lcm_of_18_and_50_l3284_328418

theorem lcm_of_18_and_50 : Nat.lcm 18 50 = 450 := by
  sorry

end lcm_of_18_and_50_l3284_328418


namespace max_zero_point_quadratic_l3284_328472

theorem max_zero_point_quadratic (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  let f := fun x : ℝ => a * x^2 + (3 + 1/b) * x - a
  let zero_points := {x : ℝ | f x = 0}
  ∃ (x : ℝ), x ∈ zero_points ∧ ∀ (y : ℝ), y ∈ zero_points → y ≤ x ∧ x = (-9 + Real.sqrt 85) / 2 :=
by sorry

end max_zero_point_quadratic_l3284_328472


namespace weightlifting_total_capacity_l3284_328477

/-- Represents a weightlifter's lifting capacities -/
structure LiftingCapacity where
  cleanAndJerk : ℝ
  snatch : ℝ

/-- Calculates the new lifting capacity after applying percentage increases -/
def newCapacity (initial : LiftingCapacity) (cjIncrease snatchIncrease : ℝ) : LiftingCapacity :=
  { cleanAndJerk := initial.cleanAndJerk * (1 + cjIncrease)
  , snatch := initial.snatch * (1 + snatchIncrease) }

/-- Calculates the total lifting capacity for a weightlifter -/
def totalCapacity (capacity : LiftingCapacity) : ℝ :=
  capacity.cleanAndJerk + capacity.snatch

/-- The theorem to be proved -/
theorem weightlifting_total_capacity : 
  let john_initial := LiftingCapacity.mk 80 50
  let alice_initial := LiftingCapacity.mk 90 55
  let mark_initial := LiftingCapacity.mk 100 65
  
  let john_final := newCapacity john_initial 1 0.8
  let alice_final := newCapacity alice_initial 0.5 0.9
  let mark_final := newCapacity mark_initial 0.75 0.7
  
  totalCapacity john_final + totalCapacity alice_final + totalCapacity mark_final = 775 := by
  sorry

end weightlifting_total_capacity_l3284_328477


namespace hexadecimal_to_decimal_l3284_328430

theorem hexadecimal_to_decimal (m : ℕ) : 
  1 * 6^5 + 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 12710 → m = 4 := by
  sorry

end hexadecimal_to_decimal_l3284_328430


namespace max_product_sum_l3284_328465

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 24) :
  (A * M * C + A * M + M * C + C * A) ≤ 704 :=
sorry

end max_product_sum_l3284_328465


namespace basket_weight_is_20_l3284_328474

/-- The weight of the basket in kilograms -/
def basket_weight : ℝ := 20

/-- The lifting capacity of one standard balloon in kilograms -/
def balloon_capacity : ℝ := 60

/-- One standard balloon can lift a basket with contents weighing not more than 80 kg -/
axiom one_balloon_limit : basket_weight + balloon_capacity ≤ 80

/-- Two standard balloons can lift the same basket with contents weighing not more than 180 kg -/
axiom two_balloon_limit : basket_weight + 2 * balloon_capacity ≤ 180

theorem basket_weight_is_20 : basket_weight = 20 := by sorry

end basket_weight_is_20_l3284_328474


namespace complex_number_relation_l3284_328440

theorem complex_number_relation (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) :
  (x^9 + y^9) / (x^9 - y^9) + (x^9 - y^9) / (x^9 + y^9) = 3/2 :=
by sorry

end complex_number_relation_l3284_328440


namespace massager_usage_time_l3284_328495

/-- The number of vibrations per second at the lowest setting -/
def lowest_vibrations_per_second : ℕ := 1600

/-- The percentage increase in vibrations at the highest setting -/
def highest_setting_increase : ℚ := 60 / 100

/-- The total number of vibrations experienced -/
def total_vibrations : ℕ := 768000

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Calculates the number of minutes Matt uses the massager at the highest setting -/
def usage_time_minutes : ℚ :=
  let highest_vibrations_per_second : ℚ := lowest_vibrations_per_second * (1 + highest_setting_increase)
  let usage_time_seconds : ℚ := total_vibrations / highest_vibrations_per_second
  usage_time_seconds / seconds_per_minute

theorem massager_usage_time :
  usage_time_minutes = 5 := by sorry

end massager_usage_time_l3284_328495


namespace power_digits_sum_l3284_328416

theorem power_digits_sum : ∃ (m n : ℕ), 
  (100 ≤ 2^m ∧ 2^m < 10000) ∧ 
  (100 ≤ 5^n ∧ 5^n < 10000) ∧ 
  (2^m / 100 % 10 = 5^n / 100 % 10) ∧
  (2^m / 100 % 10 + 5^n / 100 % 10 = 4) :=
sorry

end power_digits_sum_l3284_328416


namespace inverse_functions_values_l3284_328479

-- Define the inverse function relationship
def are_inverse_functions (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

-- Define the two linear functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

-- State the theorem
theorem inverse_functions_values :
  ∀ a b : ℝ, are_inverse_functions (f a) (g b) → a = 1/3 ∧ b = -6 :=
by sorry

end inverse_functions_values_l3284_328479


namespace untouchable_area_of_cube_l3284_328456

-- Define the cube and sphere
def cube_edge_length : ℝ := 4
def sphere_radius : ℝ := 1

-- Theorem statement
theorem untouchable_area_of_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) 
  (h1 : cube_edge_length = 4) (h2 : sphere_radius = 1) : 
  (6 * (cube_edge_length ^ 2 - (cube_edge_length - 2 * sphere_radius) ^ 2)) = 72 := by
  sorry

end untouchable_area_of_cube_l3284_328456


namespace two_numbers_difference_l3284_328445

theorem two_numbers_difference (a b : ℝ) : 
  a + b = 10 → a^2 - b^2 = 40 → |a - b| = 4 := by sorry

end two_numbers_difference_l3284_328445


namespace pie_shop_pricing_l3284_328487

/-- The number of slices per whole pie -/
def slices_per_pie : ℕ := 4

/-- The number of pies sold -/
def pies_sold : ℕ := 9

/-- The total revenue from selling all pies -/
def total_revenue : ℕ := 180

/-- The price per slice of pie -/
def price_per_slice : ℚ := 5

theorem pie_shop_pricing :
  price_per_slice = total_revenue / (pies_sold * slices_per_pie) := by
  sorry

end pie_shop_pricing_l3284_328487


namespace complex_magnitude_problem_l3284_328497

theorem complex_magnitude_problem (m : ℝ) : 
  (Complex.I * ((1 + m * Complex.I) * (3 + Complex.I))).re = 0 →
  Complex.abs ((m + 3 * Complex.I) / (1 - Complex.I)) = 3 := by
sorry

end complex_magnitude_problem_l3284_328497


namespace set_intersection_complement_equality_l3284_328484

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 3}

-- Define set N
def N : Set ℝ := {x | x ≤ 2}

-- Theorem statement
theorem set_intersection_complement_equality :
  M ∩ (U \ N) = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end set_intersection_complement_equality_l3284_328484


namespace distinct_sums_lower_bound_l3284_328453

theorem distinct_sums_lower_bound (n : ℕ) (a : Fin n → ℝ) 
  (h_pos : ∀ i, a i > 0) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  (Finset.powerset (Finset.range n)).card ≥ n * (n + 1) / 2 := by
  sorry

end distinct_sums_lower_bound_l3284_328453


namespace fraction_difference_times_two_l3284_328451

theorem fraction_difference_times_two :
  let a := 4 + 6 + 8 + 10
  let b := 3 + 5 + 7 + 9
  (a / b - b / a) * 2 = 13 / 21 := by sorry

end fraction_difference_times_two_l3284_328451


namespace line_passes_through_fixed_point_l3284_328455

/-- The line (2a+b)x + (a+b)y + a - b = 0 passes through (-2, 3) for all real a and b -/
theorem line_passes_through_fixed_point :
  ∀ (a b x y : ℝ), (2*a + b)*x + (a + b)*y + a - b = 0 ↔ (x = -2 ∧ y = 3) ∨ (x ≠ -2 ∨ y ≠ 3) :=
by sorry

end line_passes_through_fixed_point_l3284_328455


namespace mike_tv_hours_l3284_328457

-- Define the number of hours Mike watches TV daily
def tv_hours : ℝ := 4

-- Define the number of days in a week Mike plays video games
def gaming_days : ℕ := 3

-- Define the total hours spent on both activities in a week
def total_hours : ℝ := 34

-- Theorem statement
theorem mike_tv_hours :
  -- Condition: On gaming days, Mike plays for half as long as he watches TV
  (gaming_days * (tv_hours / 2) +
  -- Condition: Mike watches TV every day of the week
   7 * tv_hours = total_hours) →
  -- Conclusion: Mike watches TV for 4 hours every day
  tv_hours = 4 := by
sorry

end mike_tv_hours_l3284_328457


namespace range_of_f_l3284_328499

def f (x : ℝ) : ℝ := 4 * (x - 1)^2 - 1

theorem range_of_f : 
  ∀ y ∈ Set.Icc (-1 : ℝ) 15, ∃ x ∈ Set.Ico (-1 : ℝ) 2, f x = y ∧
  ∀ x ∈ Set.Ico (-1 : ℝ) 2, f x ∈ Set.Icc (-1 : ℝ) 15 :=
by sorry

end range_of_f_l3284_328499


namespace barneys_restock_order_l3284_328478

/-- Represents the number of items in Barney's grocery store --/
structure GroceryStore where
  sold : Nat        -- Number of items sold that day
  left : Nat        -- Number of items left in the store
  storeroom : Nat   -- Number of items in the storeroom

/-- Calculates the total number of items ordered to restock the shelves --/
def items_ordered (store : GroceryStore) : Nat :=
  store.sold + store.left + store.storeroom

/-- Theorem stating that for Barney's grocery store, the number of items
    ordered to restock the shelves is 5608 --/
theorem barneys_restock_order :
  let store : GroceryStore := {
    sold := 1561,
    left := 3472,
    storeroom := 575
  }
  items_ordered store = 5608 := by
  sorry

end barneys_restock_order_l3284_328478


namespace candy_distribution_l3284_328481

theorem candy_distribution (x : ℚ) 
  (laura_candies : x > 0)
  (mark_candies : ℚ → ℚ)
  (nina_candies : ℚ → ℚ)
  (oliver_candies : ℚ → ℚ)
  (mark_def : mark_candies x = 4 * x)
  (nina_def : nina_candies x = 2 * mark_candies x)
  (oliver_def : oliver_candies x = 6 * nina_candies x)
  (total_candies : x + mark_candies x + nina_candies x + oliver_candies x = 360) :
  x = 360 / 61 := by
  sorry

end candy_distribution_l3284_328481


namespace min_sum_reciprocal_product_l3284_328449

theorem min_sum_reciprocal_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end min_sum_reciprocal_product_l3284_328449


namespace june_score_l3284_328403

theorem june_score (april_may_avg : ℝ) (april_may_june_avg : ℝ) (june_score : ℝ) :
  april_may_avg = 89 →
  april_may_june_avg = 88 →
  june_score = 3 * april_may_june_avg - 2 * april_may_avg →
  june_score = 86 := by
sorry

end june_score_l3284_328403


namespace parallelogram_height_l3284_328446

theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) :
  area = 360 ∧ base = 30 ∧ area = base * height → height = 12 := by
  sorry

end parallelogram_height_l3284_328446


namespace cubic_equation_one_real_root_l3284_328410

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0 :=
by
  sorry

end cubic_equation_one_real_root_l3284_328410


namespace algebraic_simplification_l3284_328405

theorem algebraic_simplification (a : ℝ) : (-2*a)^3 * a^3 + (-3*a^3)^2 = a^6 := by
  sorry

end algebraic_simplification_l3284_328405


namespace ellipse_intersection_constant_sum_distance_l3284_328406

/-- The slope of a line that intersects an ellipse such that the sum of squared distances
    from any point on the major axis to the intersection points is constant. -/
theorem ellipse_intersection_constant_sum_distance (k : ℝ) : 
  (∀ a : ℝ, ∃ A B : ℝ × ℝ,
    (A.1^2 / 25 + A.2^2 / 16 = 1) ∧
    (B.1^2 / 25 + B.2^2 / 16 = 1) ∧
    (A.2 - 0 = k * (A.1 - a)) ∧
    (B.2 - 0 = k * (B.1 - a)) ∧
    ((A.1 - a)^2 + A.2^2 + (B.1 - a)^2 + B.2^2 = (512 - 800 * k^2) / (16 + 25 * k^2))) →
  k = 4/5 ∨ k = -4/5 := by
sorry

end ellipse_intersection_constant_sum_distance_l3284_328406


namespace opposite_pairs_l3284_328447

theorem opposite_pairs : 
  (3^2 = -(-3^2)) ∧ 
  (3^2 ≠ -2^3) ∧ 
  (3^2 ≠ -(-3)^2) ∧ 
  (-3^2 ≠ -(-3)^2) := by
  sorry

end opposite_pairs_l3284_328447


namespace min_k_value_l3284_328435

/-- A special number is a three-digit number with all digits different and non-zero -/
def is_special_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10 ∧
  ∀ i, (n / 10^i) % 10 ≠ 0

/-- F(n) is the sum of three new numbers obtained by swapping digits of n, divided by 111 -/
def F (n : ℕ) : ℚ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ((d2 * 100 + d1 * 10 + d3) + (d3 * 100 + d2 * 10 + d1) + (d1 * 100 + d3 * 10 + d2)) / 111

theorem min_k_value (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 1 ≤ y ∧ y ≤ 9)
  (hs : is_special_number (100 * x + 32)) (ht : is_special_number (150 + y))
  (h_sum : F (100 * x + 32) + F (150 + y) = 19) :
  let s := 100 * x + 32
  let t := 150 + y
  let k := F s - F t
  ∃ k₀, k ≥ k₀ ∧ k₀ = -7 :=
sorry

end min_k_value_l3284_328435


namespace max_profit_at_15_verify_conditions_l3284_328458

-- Define the relationship between price and sales quantity
def sales_quantity (x : ℤ) : ℤ := -5 * x + 150

-- Define the profit function
def profit (x : ℤ) : ℤ := sales_quantity x * (x - 8)

-- Theorem statement
theorem max_profit_at_15 :
  ∀ x : ℤ, 8 ≤ x → x ≤ 15 → profit x ≤ 525 ∧ profit 15 = 525 :=
by
  sorry

-- Verify the given conditions
theorem verify_conditions :
  sales_quantity 9 = 105 ∧ sales_quantity 11 = 95 :=
by
  sorry

end max_profit_at_15_verify_conditions_l3284_328458


namespace perfect_square_increased_by_prime_l3284_328483

theorem perfect_square_increased_by_prime (n : ℕ) : ∃ n : ℕ, 
  (∃ a : ℕ, n^2 = a^2) ∧ 
  (∃ b : ℕ, n^2 + 461 = b^2) ∧ 
  (∃ c : ℕ, n^2 = 5 * c) ∧ 
  (∃ d : ℕ, n^2 + 461 = 5 * d) ∧ 
  n^2 = 52900 := by
  sorry

end perfect_square_increased_by_prime_l3284_328483


namespace isosceles_trapezoid_area_l3284_328433

/-- An isosceles trapezoid circumscribed about a circle -/
structure IsoscelesTrapezoid where
  /-- The longer base of the trapezoid -/
  longerBase : ℝ
  /-- One base angle of the trapezoid -/
  baseAngle : ℝ

/-- The area of the isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem isosceles_trapezoid_area
  (t : IsoscelesTrapezoid)
  (h1 : t.longerBase = 16)
  (h2 : t.baseAngle = Real.arcsin 0.8) :
  area t = 80 :=
sorry

end isosceles_trapezoid_area_l3284_328433


namespace hyperbola_foci_l3284_328442

/-- The equation of a hyperbola in standard form -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 7 - y^2 / 9 = 1

/-- The coordinates of a focus of the hyperbola -/
def focus_coordinate : ℝ × ℝ := (4, 0)

/-- Theorem: The foci of the given hyperbola are located at (±4, 0) -/
theorem hyperbola_foci :
  let (a, b) := focus_coordinate
  (hyperbola_equation a b ∨ hyperbola_equation (-a) b) ∧
  ∀ (x y : ℝ), (x, y) ≠ (a, b) ∧ (x, y) ≠ (-a, b) →
    ¬(hyperbola_equation x y ∧ x^2 - y^2 = a^2) :=
by sorry


end hyperbola_foci_l3284_328442


namespace like_terms_exponent_sum_l3284_328420

theorem like_terms_exponent_sum (m n : ℤ) : 
  (∃ (x y : ℝ), -5 * x^m * y^(m+1) = x^(n-1) * y^3) → m + n = 5 := by
  sorry

end like_terms_exponent_sum_l3284_328420


namespace team_formation_count_l3284_328444

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of male doctors -/
def male_doctors : ℕ := 5

/-- The number of female doctors -/
def female_doctors : ℕ := 4

/-- The size of the team -/
def team_size : ℕ := 3

/-- The number of ways to form a team with both male and female doctors -/
def team_formations : ℕ := 
  choose male_doctors 2 * choose female_doctors 1 + 
  choose male_doctors 1 * choose female_doctors 2

theorem team_formation_count : team_formations = 70 := by sorry

end team_formation_count_l3284_328444


namespace percentage_difference_l3284_328463

theorem percentage_difference (C : ℝ) (A B : ℝ) 
  (hA : A = 0.75 * C) 
  (hB : B = 0.63 * C) : 
  (A - B) / A = 0.16 := by
  sorry

end percentage_difference_l3284_328463


namespace electrons_gained_by_oxidizing_agent_l3284_328411

-- Define the redox reaction components
structure RedoxReaction where
  cu_io3_2 : ℕ
  ki : ℕ
  h2so4 : ℕ
  cui : ℕ
  i2 : ℕ
  k2so4 : ℕ
  h2o : ℕ

-- Define the valence changes
structure ValenceChanges where
  cu_initial : ℤ
  cu_final : ℤ
  i_initial : ℤ
  i_final : ℤ

-- Define the function to calculate electron moles gained
def electronMolesGained (vc : ValenceChanges) : ℤ :=
  (vc.cu_initial - vc.cu_final) + 2 * (vc.i_initial - vc.i_final)

-- Theorem statement
theorem electrons_gained_by_oxidizing_agent 
  (reaction : RedoxReaction)
  (valence_changes : ValenceChanges)
  (h1 : reaction.cu_io3_2 = 2)
  (h2 : reaction.ki = 24)
  (h3 : reaction.h2so4 = 12)
  (h4 : reaction.cui = 2)
  (h5 : reaction.i2 = 13)
  (h6 : reaction.k2so4 = 12)
  (h7 : reaction.h2o = 12)
  (h8 : valence_changes.cu_initial = 2)
  (h9 : valence_changes.cu_final = 1)
  (h10 : valence_changes.i_initial = 5)
  (h11 : valence_changes.i_final = 0) :
  electronMolesGained valence_changes = 11 := by
  sorry

end electrons_gained_by_oxidizing_agent_l3284_328411


namespace ant_travel_distance_l3284_328464

theorem ant_travel_distance (planet_radius : ℝ) (observer_height : ℝ) : 
  planet_radius = 156 → observer_height = 13 → 
  let horizon_distance := Real.sqrt ((planet_radius + observer_height)^2 - planet_radius^2)
  (2 * Real.pi * horizon_distance) = 130 * Real.pi :=
by sorry

end ant_travel_distance_l3284_328464


namespace sum_of_max_min_F_l3284_328431

-- Define the function f as an odd function on [-a, a]
def f (a : ℝ) (x : ℝ) : ℝ := sorry

-- Define F(x) = f(x) + 1
def F (a : ℝ) (x : ℝ) : ℝ := f a x + 1

-- Theorem statement
theorem sum_of_max_min_F (a : ℝ) (h : a > 0) :
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc (-a) a ∧ x_min ∈ Set.Icc (-a) a ∧
  (∀ x ∈ Set.Icc (-a) a, F a x ≤ F a x_max) ∧
  (∀ x ∈ Set.Icc (-a) a, F a x_min ≤ F a x) ∧
  F a x_max + F a x_min = 2 :=
sorry

end sum_of_max_min_F_l3284_328431


namespace solution_set_f_leq_3_min_m_for_inequality_l3284_328407

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for part I
theorem solution_set_f_leq_3 :
  {x : ℝ | f x ≤ 3} = {x : ℝ | 0 ≤ x ∧ x ≤ 3} :=
by sorry

-- Theorem for part II
theorem min_m_for_inequality (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f x ≤ m - x - 4/x) ↔ m ≥ 5 :=
by sorry

end solution_set_f_leq_3_min_m_for_inequality_l3284_328407


namespace fixed_point_of_logarithmic_function_l3284_328436

/-- The logarithm function with base a -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The function f(x) = log_a(x + 3) - 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 3) - 1

theorem fixed_point_of_logarithmic_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-2) = -1 := by sorry

end fixed_point_of_logarithmic_function_l3284_328436


namespace streak_plate_method_claim_incorrect_l3284_328429

/-- Represents the capability of the streak plate method -/
structure StreakPlateMethod where
  can_separate : Bool
  can_count : Bool

/-- The actual capabilities of the streak plate method -/
def actual_streak_plate_method : StreakPlateMethod :=
  { can_separate := true
  , can_count := false }

/-- The claimed capabilities of the streak plate method in the statement -/
def claimed_streak_plate_method : StreakPlateMethod :=
  { can_separate := true
  , can_count := true }

/-- Theorem stating that the claim about the streak plate method is incorrect -/
theorem streak_plate_method_claim_incorrect :
  actual_streak_plate_method ≠ claimed_streak_plate_method :=
by sorry

end streak_plate_method_claim_incorrect_l3284_328429


namespace square_sum_problem_l3284_328460

theorem square_sum_problem (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -12) : 
  x^2 + 6*y^2 = 108 := by sorry

end square_sum_problem_l3284_328460


namespace expression_simplification_l3284_328415

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (2 * x - 6) / (x - 2) / (5 / (x - 2) - x - 2) = Real.sqrt 2 - 2 := by
  sorry

end expression_simplification_l3284_328415


namespace inequality_range_l3284_328496

theorem inequality_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (∀ m : ℝ, (1 / x + 4 / y ≥ m) ↔ m ≤ 9 / 4) := by
  sorry

end inequality_range_l3284_328496


namespace quadratic_factorization_l3284_328459

theorem quadratic_factorization (y A B : ℤ) : 
  (15 * y^2 - 94 * y + 56 = (A * y - 7) * (B * y - 8)) → 
  (A * B + A = 20) := by
sorry

end quadratic_factorization_l3284_328459


namespace percentage_difference_l3284_328413

theorem percentage_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.25 * y) : 
  x - y = 8 := by
  sorry

end percentage_difference_l3284_328413


namespace characterize_satisfying_polynomials_l3284_328422

/-- A polynomial satisfying the given inequality. -/
structure SatisfyingPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  h_c : |c| ≤ 1
  h_ab : (|a| = 1 ∧ b = 0) ∨ (|a| < 1 ∧ |b| ≤ 2 * Real.sqrt (1 + a * c - |a + c|))

/-- The main theorem statement. -/
theorem characterize_satisfying_polynomials :
  ∀ (P : ℝ → ℝ), (∀ x : ℝ, |P x - x| ≤ x^2 + 1) ↔
    ∃ (p : SatisfyingPolynomial), ∀ x : ℝ, P x = p.a * x^2 + (p.b + 1) * x + p.c :=
sorry

end characterize_satisfying_polynomials_l3284_328422


namespace intersection_of_A_and_B_l3284_328400

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l3284_328400


namespace symmetric_point_wrt_line_l3284_328439

/-- Given a line l: x - y - 1 = 0 and two points A(-1, 1) and B(2, -2),
    prove that B is symmetric to A with respect to l. -/
theorem symmetric_point_wrt_line :
  let l : ℝ → ℝ → Prop := λ x y => x - y - 1 = 0
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (2, -2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (∀ x y, l x y ↔ x - y - 1 = 0) →
  l midpoint.1 midpoint.2 ∧
  (B.2 - A.2) / (B.1 - A.1) = -((B.1 - A.1) / (B.2 - A.2)) :=
by sorry

end symmetric_point_wrt_line_l3284_328439


namespace straight_flush_probability_l3284_328426

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in a poker hand -/
def PokerHand : ℕ := 5

/-- Represents the number of possible starting ranks for a straight flush -/
def StartingRanks : ℕ := 10

/-- Represents the number of suits in a standard deck -/
def Suits : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the total number of possible 5-card hands -/
def TotalHands : ℕ := choose StandardDeck PokerHand

/-- Represents the total number of straight flushes -/
def StraightFlushes : ℕ := StartingRanks * Suits

/-- Theorem: The probability of drawing a straight flush is 1/64,974 -/
theorem straight_flush_probability :
  StraightFlushes / TotalHands = 1 / 64974 := by sorry

end straight_flush_probability_l3284_328426


namespace train_length_calculation_l3284_328492

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 80 * (1000 / 3600) →
  crossing_time = 10.889128869690424 →
  bridge_length = 142 →
  ∃ (train_length : ℝ), abs (train_length - 100.222) < 0.001 := by
  sorry

end train_length_calculation_l3284_328492


namespace sum_m_n_equals_19_l3284_328470

theorem sum_m_n_equals_19 (m n : ℕ+) 
  (h1 : (m.val.choose n.val) * 2 = 272)
  (h2 : (m.val.factorial / (m.val - n.val).factorial) = 272) :
  m + n = 19 := by sorry

end sum_m_n_equals_19_l3284_328470


namespace unique_integer_angle_geometric_progression_l3284_328461

theorem unique_integer_angle_geometric_progression :
  ∃! (a b c : ℕ+), a + b + c = 180 ∧ 
  ∃ (r : ℚ), r > 1 ∧ b = a * (r : ℚ) ∧ c = b * (r : ℚ) ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c :=
by sorry

end unique_integer_angle_geometric_progression_l3284_328461


namespace monthly_average_production_l3284_328482

/-- The daily average production for a month given production rates for different periods -/
theorem monthly_average_production 
  (days_first_period : ℕ) 
  (days_second_period : ℕ) 
  (avg_first_period : ℕ) 
  (avg_second_period : ℕ) 
  (h1 : days_first_period = 25)
  (h2 : days_second_period = 5)
  (h3 : avg_first_period = 50)
  (h4 : avg_second_period = 38) :
  (days_first_period * avg_first_period + days_second_period * avg_second_period) / 
  (days_first_period + days_second_period) = 48 := by
  sorry

#check monthly_average_production

end monthly_average_production_l3284_328482


namespace flu_transmission_rate_l3284_328469

theorem flu_transmission_rate : ∃ x : ℝ, 
  (x > 0) ∧ ((1 + x)^2 = 100) ∧ (x = 9) := by
  sorry

end flu_transmission_rate_l3284_328469


namespace xy_eq_x_plus_y_plus_3_l3284_328485

theorem xy_eq_x_plus_y_plus_3 (x y : ℕ) : 
  x * y = x + y + 3 ↔ (x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2) ∨ (x = 3 ∧ y = 3) := by
  sorry

end xy_eq_x_plus_y_plus_3_l3284_328485


namespace square_minus_floor_product_l3284_328412

/-- The floor function, which returns the greatest integer less than or equal to a given real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

/-- Theorem stating that for A = 50 + 19√7, A^2 - A⌊A⌋ = 27 -/
theorem square_minus_floor_product (A : ℝ) (h : A = 50 + 19 * Real.sqrt 7) :
  A^2 - A * (floor A) = 27 := by
  sorry

end square_minus_floor_product_l3284_328412


namespace unique_composite_with_special_divisor_property_l3284_328427

theorem unique_composite_with_special_divisor_property :
  ∃! (n : ℕ), 
    n > 1 ∧ 
    ¬(Nat.Prime n) ∧
    (∃ (k : ℕ) (d : ℕ → ℕ), 
      d 1 = 1 ∧ d k = n ∧
      (∀ i, 1 ≤ i → i < k → d i < d (i+1)) ∧
      (∀ i, 1 ≤ i → i < k → d i ∣ n) ∧
      (∀ i, 1 < i → i ≤ k → (d i - d (i-1)) = i * (d 2 - d 1))) ∧
    n = 4 := by
  sorry

end unique_composite_with_special_divisor_property_l3284_328427


namespace gcd_lcm_sum_implies_divisibility_l3284_328490

theorem gcd_lcm_sum_implies_divisibility (a b : ℤ) 
  (h : Nat.gcd a.natAbs b.natAbs + Nat.lcm a.natAbs b.natAbs = a.natAbs + b.natAbs) : 
  a ∣ b ∨ b ∣ a := by
  sorry

end gcd_lcm_sum_implies_divisibility_l3284_328490


namespace hexagon_triangle_count_l3284_328475

/-- Regular hexagon with area 6 -/
structure RegularHexagon :=
  (area : ℝ)
  (is_regular : area = 6)

/-- Equilateral triangle with area 4 -/
structure EquilateralTriangle :=
  (area : ℝ)
  (is_equilateral : area = 4)

/-- Configuration of four regular hexagons -/
def HexagonConfiguration := Fin 4 → RegularHexagon

/-- Count of equilateral triangles formed by vertices of hexagons -/
def count_triangles (config : HexagonConfiguration) : ℕ := sorry

/-- Main theorem: There are 12 equilateral triangles with area 4 -/
theorem hexagon_triangle_count (config : HexagonConfiguration) :
  count_triangles config = 12 := by sorry

end hexagon_triangle_count_l3284_328475


namespace simplify_fraction_l3284_328452

theorem simplify_fraction : (120 : ℚ) / 1320 = 1 / 11 := by sorry

end simplify_fraction_l3284_328452


namespace simplification_order_l3284_328466

-- Define the power operations
inductive PowerOperation
| MultiplicationOfPowers
| PowerOfPower
| PowerOfProduct

-- Define a function to simplify the expression
def simplify (a : ℕ) : ℕ := (a^2 * a^3)^2

-- Define a function to get the sequence of operations
def operationSequence : List PowerOperation :=
  [PowerOperation.PowerOfProduct, PowerOperation.PowerOfPower, PowerOperation.MultiplicationOfPowers]

-- State the theorem
theorem simplification_order :
  simplify a = a^10 ∧ operationSequence = [PowerOperation.PowerOfProduct, PowerOperation.PowerOfPower, PowerOperation.MultiplicationOfPowers] :=
sorry

end simplification_order_l3284_328466


namespace odd_sum_floor_power_l3284_328414

theorem odd_sum_floor_power (n : ℕ+) : 
  Odd (n + ⌊(Real.sqrt 2 + 1)^(n : ℝ)⌋) := by sorry

end odd_sum_floor_power_l3284_328414


namespace right_triangle_area_l3284_328432

theorem right_triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : c = 17) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 60 := by
  sorry

end right_triangle_area_l3284_328432


namespace sqrt_equality_l3284_328448

theorem sqrt_equality (x : ℝ) (h : x < -1) :
  Real.sqrt ((x + 2) / (1 - (x - 2) / (x + 1))) = Real.sqrt ((x^2 + 3*x + 2) / 3) := by
  sorry

end sqrt_equality_l3284_328448


namespace min_score_theorem_l3284_328450

/-- Represents the normal distribution parameters -/
structure NormalParams where
  μ : ℝ
  σ : ℝ

/-- Represents the problem parameters -/
structure ProblemParams where
  total_students : ℕ
  top_rank : ℕ
  normal_params : NormalParams

/-- The probability of being within one standard deviation of the mean -/
def prob_within_one_std : ℝ := 0.6827

/-- The probability of being within two standard deviations of the mean -/
def prob_within_two_std : ℝ := 0.9545

/-- Calculates the minimum score to be in the top rank -/
def min_score_for_top_rank (params : ProblemParams) : ℝ :=
  params.normal_params.μ + 2 * params.normal_params.σ

/-- Theorem stating the minimum score to be in the top 9100 out of 400,000 students -/
theorem min_score_theorem (params : ProblemParams)
  (h1 : params.total_students = 400000)
  (h2 : params.top_rank = 9100)
  (h3 : params.normal_params.μ = 98)
  (h4 : params.normal_params.σ = 10) :
  min_score_for_top_rank params = 118 := by
  sorry

#eval min_score_for_top_rank { total_students := 400000, top_rank := 9100, normal_params := { μ := 98, σ := 10 } }

end min_score_theorem_l3284_328450


namespace prob_five_odd_in_six_rolls_l3284_328471

def fair_six_sided_die : Fin 6 → ℚ
  | _ => 1 / 6

def is_odd (n : Fin 6) : Bool :=
  n.val % 2 = 1

def prob_exactly_k_odd (k : Nat) (n : Nat) : ℚ :=
  (Nat.choose n k) * (1/2)^k * (1/2)^(n-k)

theorem prob_five_odd_in_six_rolls :
  prob_exactly_k_odd 5 6 = 3/32 := by
  sorry

end prob_five_odd_in_six_rolls_l3284_328471


namespace product_of_special_set_l3284_328419

theorem product_of_special_set (n : ℕ) (M : Finset ℝ) (h_odd : Odd n) (h_n_gt_1 : n > 1) 
  (h_card : M.card = n) (h_sum_invariant : ∀ x ∈ M, M.sum id = (M.erase x).sum id + x) : 
  M.prod id = 0 := by
  sorry

end product_of_special_set_l3284_328419


namespace frisbee_sales_receipts_l3284_328408

theorem frisbee_sales_receipts :
  ∀ (x y : ℕ),
  x + y = 64 →
  y ≥ 4 →
  3 * x + 4 * y = 196 :=
by sorry

end frisbee_sales_receipts_l3284_328408


namespace area_of_intersection_l3284_328404

-- Define the square ABCD
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_unit : A = (0, 1) ∧ B = (1, 1) ∧ C = (1, 0) ∧ D = (0, 0))

-- Define the rotation
def rotate (p : ℝ × ℝ) (center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

-- Define the rotated square A'B'C'D'
def rotated_square (S : Square) (angle : ℝ) : Square := sorry

-- Define the intersection quadrilateral DALC'
structure Quadrilateral :=
  (D A L C' : ℝ × ℝ)

-- Define the area function for a quadrilateral
def area (Q : Quadrilateral) : ℝ := sorry

-- Main theorem
theorem area_of_intersection (S : Square) (α : ℝ) :
  let S' := rotated_square S α
  let Q := Quadrilateral.mk S.D S.A (Real.cos α, 1) (Real.cos α, Real.sin α)
  area Q = 1/2 * (1 - Real.sin α * Real.cos α) :=
sorry

end area_of_intersection_l3284_328404


namespace helga_extra_hours_thursday_l3284_328488

/-- Represents Helga's work schedule and article production --/
structure HelgaWorkSchedule where
  articles_per_30min : ℕ
  normal_hours_per_day : ℕ
  normal_days_per_week : ℕ
  articles_this_week : ℕ
  extra_hours_friday : ℕ

/-- Calculates the number of extra hours Helga worked on Thursday --/
def extra_hours_thursday (schedule : HelgaWorkSchedule) : ℕ :=
  sorry

/-- Theorem stating that given Helga's work schedule, she worked 2 extra hours on Thursday --/
theorem helga_extra_hours_thursday 
  (schedule : HelgaWorkSchedule)
  (h1 : schedule.articles_per_30min = 5)
  (h2 : schedule.normal_hours_per_day = 4)
  (h3 : schedule.normal_days_per_week = 5)
  (h4 : schedule.articles_this_week = 250)
  (h5 : schedule.extra_hours_friday = 3) :
  extra_hours_thursday schedule = 2 :=
sorry

end helga_extra_hours_thursday_l3284_328488


namespace road_repair_group_size_l3284_328473

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 15

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours : ℕ := 6

/-- The theorem stating that the first group size is 39 -/
theorem road_repair_group_size :
  first_group_size * first_group_days * first_group_hours =
  second_group_size * second_group_days * second_group_hours :=
by
  sorry

#check road_repair_group_size

end road_repair_group_size_l3284_328473
