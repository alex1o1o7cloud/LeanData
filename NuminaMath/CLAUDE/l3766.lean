import Mathlib

namespace log_equation_solution_l3766_376679

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 6 → x = 117649 := by
  sorry

end log_equation_solution_l3766_376679


namespace sqrt_3_minus_1_power_l3766_376667

theorem sqrt_3_minus_1_power (N : ℕ) : 
  (Real.sqrt 3 - 1) ^ N = 4817152 - 2781184 * Real.sqrt 3 → N = 16 := by
  sorry

end sqrt_3_minus_1_power_l3766_376667


namespace fraction2012_is_16_45_l3766_376623

/-- Represents a fraction in the sequence -/
structure Fraction :=
  (numerator : Nat)
  (denominator : Nat)
  (h1 : numerator ≤ denominator / 2)
  (h2 : numerator > 0)
  (h3 : denominator > 0)

/-- The sequence of fractions not exceeding 1/2 -/
def fractionSequence : Nat → Fraction := sorry

/-- The 2012th fraction in the sequence -/
def fraction2012 : Fraction := fractionSequence 2012

/-- Theorem stating that the 2012th fraction is 16/45 -/
theorem fraction2012_is_16_45 :
  fraction2012.numerator = 16 ∧ fraction2012.denominator = 45 := by sorry

end fraction2012_is_16_45_l3766_376623


namespace quadratic_point_m_value_l3766_376644

theorem quadratic_point_m_value (a m : ℝ) :
  a > 0 →
  m ≠ 0 →
  3 = -a * m^2 + 2 * a * m + 3 →
  m = 2 := by
  sorry

end quadratic_point_m_value_l3766_376644


namespace that_and_this_percentage_l3766_376699

/-- Proves that "that and this" plus half of "that and this" is 200% of three-quarters of "that and this" -/
theorem that_and_this_percentage : 
  ∀ x : ℝ, x > 0 → (x + 0.5 * x) / (0.75 * x) = 2 := by
  sorry

end that_and_this_percentage_l3766_376699


namespace moon_speed_conversion_l3766_376661

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.04

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The speed of the moon in kilometers per hour -/
def moon_speed_km_per_hour : ℝ := moon_speed_km_per_sec * seconds_per_hour

theorem moon_speed_conversion : 
  moon_speed_km_per_hour = 3744 := by sorry

end moon_speed_conversion_l3766_376661


namespace smallest_draw_for_red_apple_probability_l3766_376695

theorem smallest_draw_for_red_apple_probability (total_apples : Nat) (red_apples : Nat) 
  (h1 : total_apples = 15) (h2 : red_apples = 9) : 
  (∃ n : Nat, n = 5 ∧ 
    ∀ k : Nat, k < n → (red_apples - k : Rat) / (total_apples - k) ≥ 1/2 ∧
    (red_apples - n : Rat) / (total_apples - n) < 1/2) :=
by sorry

end smallest_draw_for_red_apple_probability_l3766_376695


namespace discount_percentage_l3766_376682

theorem discount_percentage (initial_amount : ℝ) 
  (h1 : initial_amount = 500)
  (h2 : ∃ (needed_before_discount : ℝ), needed_before_discount = initial_amount + 2/5 * initial_amount)
  (h3 : ∃ (amount_still_needed : ℝ), amount_still_needed = 95) : 
  ∃ (discount_percentage : ℝ), discount_percentage = 15 := by
sorry

end discount_percentage_l3766_376682


namespace square_side_length_l3766_376633

/-- Given a square with diagonal length 4, prove that its side length is 2√2. -/
theorem square_side_length (d : ℝ) (h : d = 4) : 
  ∃ s : ℝ, s > 0 ∧ s * s * 2 = d * d ∧ s = 2 * Real.sqrt 2 := by
  sorry

end square_side_length_l3766_376633


namespace eccentricity_ratio_range_l3766_376641

theorem eccentricity_ratio_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e₁ := Real.sqrt (a^2 - b^2) / a
  let e₂ := Real.sqrt (a^2 - b^2) / b
  (e₁ * e₂ < 1) →
  (∃ x, x > Real.sqrt 2 ∧ x < (1 + Real.sqrt 5) / 2 ∧ e₂ / e₁ = x) ∧
  (∀ y, y ≤ Real.sqrt 2 ∨ y ≥ (1 + Real.sqrt 5) / 2 → e₂ / e₁ ≠ y) :=
by sorry


end eccentricity_ratio_range_l3766_376641


namespace silver_coin_percentage_is_31_5_percent_l3766_376683

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  bead_percentage : ℝ
  gold_coin_percentage : ℝ

/-- Calculates the percentage of silver coins in the urn --/
def silver_coin_percentage (urn : UrnComposition) : ℝ :=
  (1 - urn.bead_percentage) * (1 - urn.gold_coin_percentage)

/-- Theorem stating that the percentage of silver coins is 31.5% --/
theorem silver_coin_percentage_is_31_5_percent :
  let urn : UrnComposition := ⟨0.3, 0.55⟩
  silver_coin_percentage urn = 0.315 := by sorry

end silver_coin_percentage_is_31_5_percent_l3766_376683


namespace sequence_formula_correct_l3766_376665

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℤ := n^2 - 10*n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℤ := 2*n - 11

/-- Theorem stating that the given formula for a_n is correct -/
theorem sequence_formula_correct (n : ℕ) : 
  n ≥ 1 → S n - S (n-1) = a n := by sorry

end sequence_formula_correct_l3766_376665


namespace binomial_29_5_l3766_376649

theorem binomial_29_5 (h1 : Nat.choose 27 3 = 2925)
                      (h2 : Nat.choose 27 4 = 17550)
                      (h3 : Nat.choose 27 5 = 80730) :
  Nat.choose 29 5 = 118755 := by
  sorry

end binomial_29_5_l3766_376649


namespace midpoint_pentagon_perimeter_l3766_376645

/-- A convex pentagon in a 2D plane. -/
structure ConvexPentagon where
  -- We don't need to define the specific properties of a convex pentagon for this problem

/-- The sum of all diagonals of a convex pentagon. -/
def sum_of_diagonals (p : ConvexPentagon) : ℝ := sorry

/-- The pentagon formed by connecting the midpoints of the sides of a convex pentagon. -/
def midpoint_pentagon (p : ConvexPentagon) : ConvexPentagon := sorry

/-- The perimeter of a pentagon. -/
def perimeter (p : ConvexPentagon) : ℝ := sorry

/-- 
Theorem: The perimeter of the pentagon formed by connecting the midpoints 
of the sides of a convex pentagon is equal to half the sum of all diagonals 
of the original pentagon.
-/
theorem midpoint_pentagon_perimeter (p : ConvexPentagon) : 
  perimeter (midpoint_pentagon p) = (1/2) * sum_of_diagonals p := by
  sorry

end midpoint_pentagon_perimeter_l3766_376645


namespace divisibility_by_seven_l3766_376627

theorem divisibility_by_seven (a b : ℕ) (h : 7 ∣ (a + b)) : 7 ∣ (101 * a + 10 * b) := by
  sorry

end divisibility_by_seven_l3766_376627


namespace modified_ohara_triple_49_64_l3766_376621

/-- Definition of a Modified O'Hara triple -/
def isModifiedOHaraTriple (a b x : ℕ+) : Prop :=
  (a.val : ℝ).sqrt + (b.val : ℝ).sqrt = (x.val : ℝ)^2

/-- Theorem: If (49, 64, x) is a Modified O'Hara triple, then x = √113 -/
theorem modified_ohara_triple_49_64 (x : ℕ+) :
  isModifiedOHaraTriple 49 64 x → x.val = Real.sqrt 113 := by
  sorry

end modified_ohara_triple_49_64_l3766_376621


namespace x_squared_minus_y_squared_l3766_376618

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 3/8) (h2 : x - y = 5/24) : x^2 - y^2 = 5/64 := by
  sorry

end x_squared_minus_y_squared_l3766_376618


namespace caterpillar_problem_l3766_376619

/-- The number of caterpillars remaining on a tree after population changes. -/
def caterpillarsRemaining (initial : ℕ) (hatched : ℕ) (left : ℕ) : ℕ :=
  initial + hatched - left

/-- Theorem stating that given the specific numbers in the problem, 
    the result is 10 caterpillars. -/
theorem caterpillar_problem : 
  caterpillarsRemaining 14 4 8 = 10 := by
  sorry

end caterpillar_problem_l3766_376619


namespace bulls_and_heat_games_l3766_376611

/-- Given that the Chicago Bulls won 70 games and the Miami Heat won 5 more games than the Bulls,
    prove that the total number of games won by both teams together is 145. -/
theorem bulls_and_heat_games (bulls_games : ℕ) (heat_games : ℕ) : 
  bulls_games = 70 → 
  heat_games = bulls_games + 5 → 
  bulls_games + heat_games = 145 := by
sorry

end bulls_and_heat_games_l3766_376611


namespace cube_sum_and_product_l3766_376698

theorem cube_sum_and_product (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  (a^3 + b^3 = 1008) ∧ ((a + b - (a - b)) * (a^3 + b^3) = 4032) := by
  sorry

end cube_sum_and_product_l3766_376698


namespace kevin_stuffed_animals_l3766_376680

/-- Represents the number of prizes Kevin collected. -/
def total_prizes : ℕ := 50

/-- Represents the number of frisbees Kevin collected. -/
def frisbees : ℕ := 18

/-- Represents the number of yo-yos Kevin collected. -/
def yo_yos : ℕ := 18

/-- Represents the number of stuffed animals Kevin collected. -/
def stuffed_animals : ℕ := total_prizes - frisbees - yo_yos

theorem kevin_stuffed_animals : stuffed_animals = 14 := by
  sorry

end kevin_stuffed_animals_l3766_376680


namespace power_function_through_point_l3766_376616

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x ^ α) →
  f 2 = Real.sqrt 2 →
  α = 1 / 2 := by
sorry

end power_function_through_point_l3766_376616


namespace min_sum_inequality_l3766_376603

theorem min_sum_inequality (a b μ : ℝ) (ha : a > 0) (hb : b > 0) (hμ : μ > 0) 
  (h : 1/a + 9/b = 1) : 
  (∀ a b, a > 0 → b > 0 → 1/a + 9/b = 1 → a + b ≥ μ) ↔ μ ∈ Set.Ioo 0 16 := by
  sorry

end min_sum_inequality_l3766_376603


namespace train_crossing_bridge_time_train_crossing_bridge_time_is_35_seconds_l3766_376684

/-- The time required for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 200) 
  (h2 : bridge_length = 150) 
  (h3 : train_speed_kmph = 36) : ℝ :=
let train_speed_mps := train_speed_kmph * (5/18)
let total_distance := train_length + bridge_length
let time := total_distance / train_speed_mps
35

theorem train_crossing_bridge_time_is_35_seconds 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 200) 
  (h2 : bridge_length = 150) 
  (h3 : train_speed_kmph = 36) : 
  train_crossing_bridge_time train_length bridge_length train_speed_kmph h1 h2 h3 = 35 := by
sorry

end train_crossing_bridge_time_train_crossing_bridge_time_is_35_seconds_l3766_376684


namespace rectangle_length_equality_l3766_376605

/-- Given two rectangles with equal area, where one rectangle measures 15 inches by 24 inches
    and the other is 45 inches wide, the length of the second rectangle is 8 inches. -/
theorem rectangle_length_equality (carol_length carol_width jordan_width : ℕ) 
    (jordan_length : ℚ) : 
  carol_length = 15 ∧ 
  carol_width = 24 ∧ 
  jordan_width = 45 ∧ 
  carol_length * carol_width = jordan_length * jordan_width →
  jordan_length = 8 := by
sorry

end rectangle_length_equality_l3766_376605


namespace sum_interior_angles_hexagon_l3766_376626

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon is a polygon with 6 sides. -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720 degrees. -/
theorem sum_interior_angles_hexagon :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end sum_interior_angles_hexagon_l3766_376626


namespace a_8_equals_14_l3766_376637

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℕ := n^2 - n

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℕ := S n - S (n-1)

theorem a_8_equals_14 : a 8 = 14 := by sorry

end a_8_equals_14_l3766_376637


namespace largest_angle_in_convex_pentagon_l3766_376687

/-- The largest angle in a convex pentagon with specific angle measures -/
theorem largest_angle_in_convex_pentagon : 
  ∀ x : ℝ,
  (x + 2) + (2*x + 3) + (3*x - 4) + (4*x + 5) + (5*x - 6) = 540 →
  max (x + 2) (max (2*x + 3) (max (3*x - 4) (max (4*x + 5) (5*x - 6)))) = 174 :=
by
  sorry

end largest_angle_in_convex_pentagon_l3766_376687


namespace total_cost_trick_decks_l3766_376638

/-- Calculates the cost of decks based on tiered pricing and promotion --/
def calculate_cost (num_decks : ℕ) : ℚ :=
  let base_price := if num_decks ≤ 3 then 8
                    else if num_decks ≤ 6 then 7
                    else 6
  let full_price_decks := num_decks / 2
  let discounted_decks := num_decks - full_price_decks
  (full_price_decks * base_price + discounted_decks * base_price / 2 : ℚ)

/-- The total cost of trick decks for Victor and his friend --/
theorem total_cost_trick_decks : 
  calculate_cost 6 + calculate_cost 2 = 43.5 := by
  sorry

#eval calculate_cost 6 + calculate_cost 2

end total_cost_trick_decks_l3766_376638


namespace participants_2003_l3766_376689

def initial_participants : ℕ := 500
def increase_2001 : ℚ := 1.3
def increase_2002 : ℚ := 1.4
def increase_2003 : ℚ := 1.5

theorem participants_2003 : 
  ⌊(((initial_participants : ℚ) * increase_2001) * increase_2002) * increase_2003⌋ = 1365 := by
  sorry

end participants_2003_l3766_376689


namespace select_at_most_one_ab_l3766_376666

def students : ℕ := 5
def selected : ℕ := 3
def competitions : ℕ := 3

def ways_to_select (n k : ℕ) : ℕ := Nat.choose n k

def ways_to_assign (n : ℕ) : ℕ := Nat.factorial n

def select_with_at_most_one_specific (total specific selected : ℕ) : ℕ :=
  -- Case 1: One specific student selected
  2 * (ways_to_select (total - 2) (selected - 1) * ways_to_assign competitions) +
  -- Case 2: Neither specific student selected
  (ways_to_select (total - 2) selected * ways_to_assign competitions)

theorem select_at_most_one_ab :
  select_with_at_most_one_specific students 2 selected = 42 := by
  sorry

end select_at_most_one_ab_l3766_376666


namespace expansion_properties_l3766_376643

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Sum of odd-indexed binomial coefficients for (a + b)^n -/
def sumOddCoeffs (n : ℕ) : ℕ := sorry

/-- The constant term in the expansion of (x + 1/√x)^n -/
def constantTerm (n : ℕ) : ℕ := sorry

/-- Theorem about the expansion of (x + 1/√x)^9 -/
theorem expansion_properties :
  (sumOddCoeffs 9 = 256) ∧ (constantTerm 9 = 84) := by sorry

end expansion_properties_l3766_376643


namespace parabola_parameter_is_two_l3766_376614

/-- Proves that given a hyperbola and a parabola with specific properties, 
    the parameter of the parabola is 2. -/
theorem parabola_parameter_is_two 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (eccentricity : Real.sqrt (1 + b^2 / a^2) = 2)
  (parabola : ∀ x y, y^2 = 2 * p * x)
  (triangle_area : 1/4 * p^2 * b / a = Real.sqrt 3) :
  p = 2 := by
  sorry

end parabola_parameter_is_two_l3766_376614


namespace dog_groom_time_l3766_376622

/-- The time (in hours) it takes to groom a cat -/
def cat_groom_time : ℝ := 0.5

/-- The total time (in hours) it takes to groom 5 dogs and 3 cats -/
def total_groom_time : ℝ := 14

/-- The number of dogs groomed -/
def num_dogs : ℕ := 5

/-- The number of cats groomed -/
def num_cats : ℕ := 3

/-- Theorem stating that the time to groom a dog is 2.5 hours -/
theorem dog_groom_time : 
  ∃ (dog_time : ℝ), 
    dog_time * num_dogs + cat_groom_time * num_cats = total_groom_time ∧ 
    dog_time = 2.5 := by
  sorry

end dog_groom_time_l3766_376622


namespace instantaneous_speed_at_3_seconds_l3766_376672

/-- The motion equation of an object -/
def s (t : ℝ) : ℝ := 1 - t + 2 * t^2

/-- The instantaneous speed (derivative of s) -/
def v (t : ℝ) : ℝ := -1 + 4 * t

theorem instantaneous_speed_at_3_seconds :
  v 3 = 11 := by sorry

end instantaneous_speed_at_3_seconds_l3766_376672


namespace product_of_cubic_fractions_l3766_376646

theorem product_of_cubic_fractions : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 4) * (f 5) * (f 6) * (f 7) * (f 8) = 73 / 312 := by
  sorry

end product_of_cubic_fractions_l3766_376646


namespace jenny_walking_distance_l3766_376659

theorem jenny_walking_distance (ran_distance : Real) (extra_ran_distance : Real) :
  ran_distance = 0.6 →
  extra_ran_distance = 0.2 →
  ∃ walked_distance : Real,
    walked_distance + extra_ran_distance = ran_distance ∧
    walked_distance = 0.4 :=
by sorry

end jenny_walking_distance_l3766_376659


namespace imaginary_part_of_complex_fraction_l3766_376642

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / (3 - i)
  Complex.im z = -1/5 := by
sorry

end imaginary_part_of_complex_fraction_l3766_376642


namespace minimum_selling_price_for_profit_margin_l3766_376630

/-- The minimum selling price for a small refrigerator to maintain a 20% profit margin --/
theorem minimum_selling_price_for_profit_margin
  (average_sales : ℕ)
  (refrigerator_cost : ℝ)
  (shipping_fee : ℝ)
  (storefront_fee : ℝ)
  (repair_cost : ℝ)
  (profit_margin : ℝ)
  (h_average_sales : average_sales = 50)
  (h_refrigerator_cost : refrigerator_cost = 1200)
  (h_shipping_fee : shipping_fee = 20)
  (h_storefront_fee : storefront_fee = 10000)
  (h_repair_cost : repair_cost = 5000)
  (h_profit_margin : profit_margin = 0.2)
  : ∃ (x : ℝ), x ≥ 1824 ∧
    (average_sales : ℝ) * x - (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) ≥
    (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) * profit_margin ∧
    ∀ (y : ℝ), y < x →
      (average_sales : ℝ) * y - (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) <
      (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) * profit_margin :=
by sorry

end minimum_selling_price_for_profit_margin_l3766_376630


namespace eldest_age_l3766_376640

theorem eldest_age (a b c d : ℕ) : 
  (∃ (x : ℕ), a = 5 * x ∧ b = 7 * x ∧ c = 8 * x ∧ d = 9 * x) →  -- ages are in ratio 5:7:8:9
  (a - 10) + (b - 10) + (c - 10) + (d - 10) = 107 →             -- sum of ages 10 years ago
  d = 45                                                        -- present age of eldest
  := by sorry

end eldest_age_l3766_376640


namespace julias_mean_score_l3766_376655

def scores : List ℝ := [88, 90, 92, 94, 95, 97, 98, 99]

def henry_mean : ℝ := 94

theorem julias_mean_score (h1 : scores.length = 8)
                          (h2 : ∃ henry_scores julia_scores : List ℝ,
                                henry_scores.length = 4 ∧
                                julia_scores.length = 4 ∧
                                henry_scores ++ julia_scores = scores)
                          (h3 : ∃ henry_scores : List ℝ,
                                henry_scores.length = 4 ∧
                                henry_scores.sum / 4 = henry_mean) :
  ∃ julia_scores : List ℝ,
    julia_scores.length = 4 ∧
    julia_scores.sum / 4 = 94.25 :=
sorry

end julias_mean_score_l3766_376655


namespace profit_percentage_calculation_l3766_376634

theorem profit_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 60 → selling_price = 63 → 
  (selling_price - cost_price) / cost_price * 100 = 5 := by
  sorry

end profit_percentage_calculation_l3766_376634


namespace planes_perpendicular_to_same_plane_are_parallel_l3766_376625

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields/axioms for a plane

/-- A line in 3D space -/
structure Line where
  -- Add necessary fields/axioms for a line

/-- Two planes are perpendicular -/
def perpendicular (p1 p2 : Plane) : Prop :=
  sorry

/-- Two planes are parallel -/
def parallel (p1 p2 : Plane) : Prop :=
  sorry

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_plane (l : Line) (p : Plane) : Prop :=
  sorry

theorem planes_perpendicular_to_same_plane_are_parallel 
  (α β γ : Plane) : perpendicular α γ → perpendicular β γ → parallel α β :=
sorry

end planes_perpendicular_to_same_plane_are_parallel_l3766_376625


namespace set_equality_implies_sum_l3766_376610

theorem set_equality_implies_sum (m n : ℝ) : 
  let P : Set ℝ := {m / n, 1}
  let Q : Set ℝ := {n, 0}
  P = Q → m + n = 1 := by
sorry

end set_equality_implies_sum_l3766_376610


namespace inequality_proof_l3766_376660

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end inequality_proof_l3766_376660


namespace cube_root_of_hundred_l3766_376673

theorem cube_root_of_hundred (x : ℝ) : (Real.sqrt x)^3 = 100 → x = 10^(4/3) := by
  sorry

end cube_root_of_hundred_l3766_376673


namespace value_of_expression_l3766_376604

theorem value_of_expression (h k : ℤ) : 
  (∃ a : ℤ, 3 * X^3 - h * X - k = a * (X - 3) * (X + 1)) →
  |3 * h - 2 * k| = 27 :=
by
  sorry

end value_of_expression_l3766_376604


namespace ellipse_circle_fixed_point_l3766_376671

/-- Given an ellipse with equation (x²/a²) + (y²/b²) = 1, where a > b > 0,
    and a point P(x₀, y₀) on the ellipse different from A₁(-a, 0) and A(a, 0),
    the circle with diameter MM₁ (where M and M₁ are intersections of PA and PA₁
    with the directrix x = a²/c) passes through a fixed point outside the ellipse. -/
theorem ellipse_circle_fixed_point
  (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_pos : c > 0) (h_a_gt_b : a > b)
  (x₀ y₀ : ℝ) (h_on_ellipse : x₀^2 / a^2 + y₀^2 / b^2 = 1)
  (h_not_A : x₀ ≠ a ∨ y₀ ≠ 0) (h_not_A₁ : x₀ ≠ -a ∨ y₀ ≠ 0) :
  ∃ (x y : ℝ), x = (a^2 + b^2) / c ∧ y = 0 ∧
  (x - a^2 / c)^2 + (y + b^2 * (x₀ - c) / (c * y₀))^2 = (b^2 * (c * x₀ - a^2) / (a * c * y₀))^2 :=
sorry

end ellipse_circle_fixed_point_l3766_376671


namespace system_unique_solution_l3766_376629

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  Real.arccos ((4 - y) / 4) = Real.arccos ((a + x) / 2) ∧
  x^2 + y^2 + 2*x - 8*y = b

-- Define the condition for a
def a_condition (a : ℝ) : Prop :=
  a ≤ -9 ∨ a ≥ 11

-- Theorem statement
theorem system_unique_solution (a : ℝ) :
  (∀ b : ℝ, (∃! p : ℝ × ℝ, system a b p.1 p.2) ∨ (¬ ∃ p : ℝ × ℝ, system a b p.1 p.2)) ↔
  a_condition a :=
sorry

end system_unique_solution_l3766_376629


namespace average_calls_proof_l3766_376664

def average_calls (mon tue wed thu fri : ℕ) : ℚ :=
  (mon + tue + wed + thu + fri : ℚ) / 5

theorem average_calls_proof (mon tue wed thu fri : ℕ) :
  average_calls mon tue wed thu fri = (mon + tue + wed + thu + fri : ℚ) / 5 := by
  sorry

end average_calls_proof_l3766_376664


namespace correct_calculation_l3766_376652

theorem correct_calculation (square : ℕ) (h : (325 - square) * 5 = 1500) : 
  325 - square * 5 = 200 := by
  sorry

end correct_calculation_l3766_376652


namespace remove_parentheses_l3766_376674

theorem remove_parentheses (a : ℝ) : -(2*a - 1) = -2*a + 1 := by
  sorry

end remove_parentheses_l3766_376674


namespace homework_scenarios_count_l3766_376678

/-- The number of subjects available for homework -/
def num_subjects : ℕ := 4

/-- The number of students doing homework -/
def num_students : ℕ := 3

/-- The number of possible scenarios for homework assignment -/
def num_scenarios : ℕ := num_subjects ^ num_students

/-- Theorem stating that the number of possible scenarios is 64 -/
theorem homework_scenarios_count : num_scenarios = 64 := by
  sorry

end homework_scenarios_count_l3766_376678


namespace median_triangle_area_l3766_376668

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (ma mb mc : ℝ)
  (area : ℝ)

-- Define the triangle formed by medians
def MedianTriangle (t : Triangle) : Triangle :=
  { a := t.ma,
    b := t.mb,
    c := t.mc,
    ma := 0,  -- We don't need these values for the median triangle
    mb := 0,
    mc := 0,
    area := 0 }  -- We'll prove this is 3/4 * t.area

-- Theorem statement
theorem median_triangle_area (t : Triangle) :
  (MedianTriangle t).area = 3/4 * t.area :=
sorry

end median_triangle_area_l3766_376668


namespace company_employees_count_l3766_376636

theorem company_employees_count :
  let total_employees : ℝ := 140
  let prefer_x : ℝ := 0.6
  let prefer_y : ℝ := 0.4
  let max_satisfied : ℝ := 140
  prefer_x + prefer_y = 1 →
  prefer_x * total_employees + prefer_y * total_employees = max_satisfied →
  total_employees = 140 :=
by sorry

end company_employees_count_l3766_376636


namespace min_value_theorem_min_value_achieved_l3766_376631

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + a * b = 3) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y + x * y = 3 → 2 * a + b ≤ 2 * x + y :=
by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + a * b = 3) :
  2 * a + b = 4 * Real.sqrt 2 - 3 ↔ a = Real.sqrt 2 - 1 ∧ b = 2 * Real.sqrt 2 - 1 :=
by sorry

end min_value_theorem_min_value_achieved_l3766_376631


namespace solution_difference_l3766_376670

theorem solution_difference (a b : ℝ) (ha : a ≠ 0) 
  (h : a^2 - b*a - 4*a = 0) : a - b = 4 := by
  sorry

end solution_difference_l3766_376670


namespace b_value_range_l3766_376606

theorem b_value_range (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 18) : 
  ∃ (b_min b_max : ℝ), 
    (∀ b', (∃ a' c', a' + b' + c' = 3 ∧ a'^2 + b'^2 + c'^2 = 18) → b_min ≤ b' ∧ b' ≤ b_max) ∧
    b_max - b_min = 2 * Real.sqrt (45/4) :=
sorry

end b_value_range_l3766_376606


namespace reciprocal_F_location_l3766_376648

/-- A complex number in the first quadrant outside the unit circle -/
def F : ℂ :=
  sorry

/-- Theorem: The reciprocal of F is in the fourth quadrant inside the unit circle -/
theorem reciprocal_F_location :
  let z := F⁻¹
  0 < z.re ∧ z.im < 0 ∧ Complex.abs z < 1 :=
by sorry

end reciprocal_F_location_l3766_376648


namespace lcm_24_36_42_l3766_376609

theorem lcm_24_36_42 : Nat.lcm 24 (Nat.lcm 36 42) = 504 := by
  sorry

end lcm_24_36_42_l3766_376609


namespace infinite_geometric_series_common_ratio_l3766_376617

theorem infinite_geometric_series_common_ratio 
  (a : ℝ) (S : ℝ) (h1 : a = 500) (h2 : S = 2500) :
  let r := 1 - a / S
  r = 4 / 5 := by
  sorry

end infinite_geometric_series_common_ratio_l3766_376617


namespace triangle_inequality_cube_l3766_376697

/-- Triangle inequality for sides a, b, c -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- The inequality to be proven -/
theorem triangle_inequality_cube (a b c : ℝ) (h : is_triangle a b c) :
  a^3 + b^3 + c^3 + 4*a*b*c ≤ 9/32 * (a + b + c)^3 :=
sorry

end triangle_inequality_cube_l3766_376697


namespace square_difference_l3766_376602

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end square_difference_l3766_376602


namespace boxes_filled_in_five_minutes_l3766_376654

/-- Given a machine that fills boxes at a constant rate, 
    this theorem proves how many boxes it can fill in 5 minutes. -/
theorem boxes_filled_in_five_minutes 
  (boxes_per_hour : ℚ) 
  (h1 : boxes_per_hour = 24 / 60) : 
  boxes_per_hour * 5 = 2 := by
  sorry

#check boxes_filled_in_five_minutes

end boxes_filled_in_five_minutes_l3766_376654


namespace second_part_speed_l3766_376624

/-- Proves that given a trip of 70 kilometers, where the first 35 kilometers are traveled at 48 km/h
    and the average speed of the entire trip is 32 km/h, the speed of the second part of the trip is 24 km/h. -/
theorem second_part_speed (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (average_speed : ℝ)
    (h1 : total_distance = 70)
    (h2 : first_part_distance = 35)
    (h3 : first_part_speed = 48)
    (h4 : average_speed = 32) :
    let second_part_distance := total_distance - first_part_distance
    let total_time := total_distance / average_speed
    let first_part_time := first_part_distance / first_part_speed
    let second_part_time := total_time - first_part_time
    let second_part_speed := second_part_distance / second_part_time
    second_part_speed = 24 := by
  sorry

end second_part_speed_l3766_376624


namespace coefficient_of_b_fourth_l3766_376653

theorem coefficient_of_b_fourth (b : ℝ) : 
  (∃ b : ℝ, b^4 - 41*b^2 + 100 = 0) ∧ 
  (∃ b₁ b₂ : ℝ, b₁ ≥ b₂ ∧ b₂ ≥ 0 ∧ b₁ + b₂ = 4.5 ∧ 
    b₁^4 - 41*b₁^2 + 100 = 0 ∧ b₂^4 - 41*b₂^2 + 100 = 0) →
  (∃ a : ℝ, ∀ b : ℝ, a*b^4 - 41*b^2 + 100 = 0 → a = 1) :=
by sorry

end coefficient_of_b_fourth_l3766_376653


namespace min_value_A_l3766_376669

theorem min_value_A (x y z w : ℝ) :
  ∃ (A : ℝ), A = (1 + Real.sqrt 2) / 2 ∧
  (∀ (B : ℝ), (x*y + 2*y*z + z*w ≤ B*(x^2 + y^2 + z^2 + w^2)) → A ≤ B) ∧
  (x*y + 2*y*z + z*w ≤ A*(x^2 + y^2 + z^2 + w^2)) :=
by sorry

end min_value_A_l3766_376669


namespace line_circle_intersection_l3766_376675

/-- Given a line and a circle that intersect at two points with a specific distance between them, 
    prove that the slope of the line has a specific value. -/
theorem line_circle_intersection (m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (∀ x y : ℝ, m * x + y + 3 * m - Real.sqrt 3 = 0 → (x, y) ∈ {(x, y) | m * x + y + 3 * m - Real.sqrt 3 = 0}) ∧ 
    (∀ x y : ℝ, x^2 + y^2 = 12 → (x, y) ∈ {(x, y) | x^2 + y^2 = 12}) ∧
    A ∈ {(x, y) | m * x + y + 3 * m - Real.sqrt 3 = 0} ∧
    A ∈ {(x, y) | x^2 + y^2 = 12} ∧
    B ∈ {(x, y) | m * x + y + 3 * m - Real.sqrt 3 = 0} ∧
    B ∈ {(x, y) | x^2 + y^2 = 12} ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12) →
  m = -Real.sqrt 3 / 3 := by
sorry

end line_circle_intersection_l3766_376675


namespace function_comparison_and_maximum_l3766_376693

def f (x : ℝ) := abs (x - 1)
def g (x : ℝ) := -x^2 + 6*x - 5

theorem function_comparison_and_maximum :
  (∀ x : ℝ, g x ≥ f x ↔ x ∈ Set.Icc 1 4) ∧
  (∃ M : ℝ, M = 9/4 ∧ ∀ x : ℝ, g x - f x ≤ M) :=
sorry

end function_comparison_and_maximum_l3766_376693


namespace cube_surface_area_increase_l3766_376685

/-- Represents a cube with a given side length -/
structure Cube where
  side_length : ℝ

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℝ :=
  6 * c.side_length * c.side_length

/-- Calculates the increase in surface area after making cuts -/
def surface_area_increase (c : Cube) (num_cuts : ℕ) : ℝ :=
  2 * c.side_length * c.side_length * num_cuts

/-- Theorem: The increase in surface area of a 10 cm cube after three cuts is 600 cm² -/
theorem cube_surface_area_increase :
  let c := Cube.mk 10
  surface_area_increase c 3 = 600 := by
  sorry

end cube_surface_area_increase_l3766_376685


namespace perimeter_of_shaded_region_l3766_376601

/-- The perimeter of the shaded region formed by three touching circles -/
theorem perimeter_of_shaded_region (circle_circumference : ℝ) :
  circle_circumference = 36 →
  (3 : ℝ) * (circle_circumference / 6) = 18 := by
  sorry

end perimeter_of_shaded_region_l3766_376601


namespace union_of_P_and_Q_l3766_376608

def P : Set ℕ := {1, 2, 3, 4}

def Q : Set ℕ := {x : ℕ | 3 ≤ x ∧ x < 7}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3, 4, 5, 6} := by
  sorry

end union_of_P_and_Q_l3766_376608


namespace natalia_crates_l3766_376628

/-- Calculates the number of crates needed for a given number of items and crate capacity -/
def crates_needed (items : ℕ) (capacity : ℕ) : ℕ :=
  (items + capacity - 1) / capacity

/-- The total number of crates needed for Natalia's items -/
def total_crates : ℕ :=
  crates_needed 145 12 + crates_needed 271 8 + crates_needed 419 10 + crates_needed 209 14

theorem natalia_crates :
  total_crates = 104 := by
  sorry

end natalia_crates_l3766_376628


namespace quadratic_equation_rational_roots_l3766_376690

theorem quadratic_equation_rational_roots (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y : ℚ, x^2 + p^2 * x + q^3 = 0 ∧ y^2 + p^2 * y + q^3 = 0 ∧ x ≠ y) ↔ 
  (p = 3 ∧ q = 2 ∧ 
   ∃ x y : ℚ, x = -1 ∧ y = -8 ∧ 
   x^2 + p^2 * x + q^3 = 0 ∧ y^2 + p^2 * y + q^3 = 0) :=
by sorry

end quadratic_equation_rational_roots_l3766_376690


namespace tan_plus_4sin_20_deg_equals_sqrt3_l3766_376635

theorem tan_plus_4sin_20_deg_equals_sqrt3 :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_plus_4sin_20_deg_equals_sqrt3_l3766_376635


namespace sudoku_like_puzzle_l3766_376686

/-- A 2x2 grid filled with numbers from 1 to 4 -/
def Grid := Fin 2 → Fin 2 → Fin 4

/-- Check if all numbers in a list are distinct -/
def all_distinct (l : List (Fin 4)) : Prop :=
  l.length = 4 ∧ l.Nodup

/-- Check if a grid satisfies Sudoku-like conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ i, all_distinct [g i 0, g i 1]) ∧  -- rows
  (∀ j, all_distinct [g 0 j, g 1 j]) ∧  -- columns
  all_distinct [g 0 0, g 0 1, g 1 0, g 1 1]  -- entire 2x2 grid

theorem sudoku_like_puzzle :
  ∀ g : Grid,
    valid_grid g →
    g 0 0 = 0 →  -- 1 in top-left (0-indexed)
    g 1 1 = 3 →  -- 4 in bottom-right (0-indexed)
    g 0 1 = 2    -- 3 in top-right (0-indexed)
    := by sorry

end sudoku_like_puzzle_l3766_376686


namespace perimeter_of_larger_square_l3766_376615

-- Define the side lengths of the small squares
def small_squares : List ℕ := [1, 1, 2, 3, 5, 8, 13]

-- Define the property that these squares form a larger square
def forms_larger_square (squares : List ℕ) : Prop := sorry

-- Define the perimeter calculation function
def calculate_perimeter (squares : List ℕ) : ℕ := sorry

-- Theorem statement
theorem perimeter_of_larger_square :
  forms_larger_square small_squares →
  calculate_perimeter small_squares = 68 := by
  sorry

end perimeter_of_larger_square_l3766_376615


namespace N_not_cube_l3766_376639

/-- Represents a number of the form 10...050...01 with 100 zeros in each group -/
def N : ℕ := 10^201 + 5 * 10^100 + 1

/-- Theorem stating that N is not a perfect cube -/
theorem N_not_cube : ¬ ∃ (m : ℕ), m^3 = N := by
  sorry

end N_not_cube_l3766_376639


namespace russia_us_size_ratio_l3766_376681

theorem russia_us_size_ratio :
  ∀ (us canada russia : ℝ),
    us > 0 →
    canada = 1.5 * us →
    russia = (4/3) * canada →
    russia / us = 8/3 := by
  sorry

end russia_us_size_ratio_l3766_376681


namespace hyperbola_eccentricity_l3766_376658

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) →
  Real.sqrt (1 + (b/a)^2) = 2 :=
sorry

end hyperbola_eccentricity_l3766_376658


namespace log_sum_abs_l3766_376600

theorem log_sum_abs (x : ℝ) (θ : ℝ) (h : Real.log x / Real.log 3 = 1 + Real.sin θ) :
  |x - 1| + |x - 9| = 8 := by
  sorry

end log_sum_abs_l3766_376600


namespace probability_at_least_one_white_ball_l3766_376647

/-- Given a bag with 3 red balls and 2 white balls, the probability of drawing
    at least one white ball when 3 balls are randomly drawn is 9/10. -/
theorem probability_at_least_one_white_ball
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (drawn_balls : ℕ)
  (h1 : total_balls = red_balls + white_balls)
  (h2 : red_balls = 3)
  (h3 : white_balls = 2)
  (h4 : drawn_balls = 3) :
  (1 - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ)) = 9/10 :=
by sorry

end probability_at_least_one_white_ball_l3766_376647


namespace brads_money_l3766_376662

theorem brads_money (total : ℚ) (josh_brad_ratio : ℚ) (josh_doug_ratio : ℚ)
  (h1 : total = 68)
  (h2 : josh_brad_ratio = 2)
  (h3 : josh_doug_ratio = 3/4) :
  ∃ (brad : ℚ), brad = 12 ∧ 
    ∃ (josh doug : ℚ), 
      josh = josh_brad_ratio * brad ∧
      josh = josh_doug_ratio * doug ∧
      josh + doug + brad = total :=
by sorry

end brads_money_l3766_376662


namespace birth_interval_is_3_7_l3766_376651

/-- Represents the ages of 5 children -/
structure ChildrenAges where
  ages : Fin 5 → ℕ
  sum_65 : ages 0 + ages 1 + ages 2 + ages 3 + ages 4 = 65
  youngest_7 : ages 0 = 7

/-- The interval between births, assuming equal spacing -/
def birthInterval (c : ChildrenAges) : ℚ :=
  ((c.ages 4 - c.ages 0) : ℚ) / 4

/-- Theorem stating the birth interval is 3.7 years -/
theorem birth_interval_is_3_7 (c : ChildrenAges) : birthInterval c = 37/10 := by
  sorry

end birth_interval_is_3_7_l3766_376651


namespace hyperbola_standard_equation_l3766_376650

/-- Given a hyperbola with foci on the y-axis, real axis length of 6,
    and asymptotes y = ± 3/2 x, its standard equation is y²/9 - x²/4 = 1 -/
theorem hyperbola_standard_equation
  (foci_on_y_axis : Bool)
  (real_axis_length : ℝ)
  (asymptote_slope : ℝ)
  (h_real_axis : real_axis_length = 6)
  (h_asymptote : asymptote_slope = 3/2) :
  ∃ (a b : ℝ),
    a = real_axis_length / 2 ∧
    b = a / asymptote_slope ∧
    (λ (x y : ℝ) => y^2 / a^2 - x^2 / b^2 = 1) =
    (λ (x y : ℝ) => y^2 / 9 - x^2 / 4 = 1) :=
by sorry

end hyperbola_standard_equation_l3766_376650


namespace father_ate_chocolates_father_ate_two_chocolates_l3766_376656

theorem father_ate_chocolates (total_chocolates : ℕ) (num_sisters : ℕ) (given_to_mother : ℕ) (father_left : ℕ) : ℕ :=
  let num_people := num_sisters + 1
  let chocolates_per_person := total_chocolates / num_people
  let given_to_father := num_people * (chocolates_per_person / 2)
  let father_initial := given_to_father - given_to_mother
  father_initial - father_left

theorem father_ate_two_chocolates :
  father_ate_chocolates 20 4 3 5 = 2 := by
  sorry

end father_ate_chocolates_father_ate_two_chocolates_l3766_376656


namespace arithmetic_sequence_properties_l3766_376663

/-- An arithmetic sequence with sum of first n terms S_n = -2n^2 + 15n -/
def S (n : ℕ+) : ℤ := -2 * n.val ^ 2 + 15 * n.val

/-- The general term of the arithmetic sequence -/
def a (n : ℕ+) : ℤ := 17 - 4 * n.val

theorem arithmetic_sequence_properties :
  ∀ n : ℕ+,
  -- The general term of the sequence is a_n = 17 - 4n
  (∀ k : ℕ+, S k - S (k - 1) = a k) ∧
  -- S_n achieves its maximum value when n = 4
  (∀ k : ℕ+, S k ≤ S 4) ∧
  -- The maximum value of S_n is 28
  S 4 = 28 := by
  sorry

end arithmetic_sequence_properties_l3766_376663


namespace pizza_order_problem_l3766_376613

theorem pizza_order_problem (slices_per_pizza : ℕ) (james_fraction : ℚ) (james_slices : ℕ) :
  slices_per_pizza = 6 →
  james_fraction = 2 / 3 →
  james_slices = 8 →
  (james_slices : ℚ) / james_fraction / slices_per_pizza = 2 :=
by sorry

end pizza_order_problem_l3766_376613


namespace digit_difference_750_150_l3766_376696

/-- The number of digits in the base-2 representation of a positive integer -/
def numDigitsBase2 (n : ℕ+) : ℕ :=
  Nat.log2 n + 1

/-- The difference in the number of digits between 750 and 150 in base 2 -/
theorem digit_difference_750_150 : numDigitsBase2 750 - numDigitsBase2 150 = 1 := by
  sorry

end digit_difference_750_150_l3766_376696


namespace no_integer_solution_l3766_376692

theorem no_integer_solution (a b c d : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) :
  ¬∃ x : ℤ, x^4 - a*x^3 - b*x^2 - c*x - d = 0 := by
  sorry

end no_integer_solution_l3766_376692


namespace max_quarters_l3766_376676

theorem max_quarters (total : ℚ) (q : ℕ) : 
  total = 4.55 →
  (0.25 * q + 0.05 * q + 0.1 * (q / 2 : ℚ) = total) →
  (∀ n : ℕ, (0.25 * n + 0.05 * n + 0.1 * (n / 2 : ℚ) ≤ total)) →
  q = 13 :=
by sorry

end max_quarters_l3766_376676


namespace sin_600_degrees_l3766_376677

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_600_degrees_l3766_376677


namespace dima_lives_on_seventh_floor_l3766_376620

/-- Represents the floor where Dima lives -/
def dimas_floor : ℕ := 7

/-- Represents the highest floor button Dima can reach -/
def max_reachable_floor : ℕ := 6

/-- The number of stories in the building -/
def building_stories : ℕ := 9

/-- Time (in seconds) it takes to descend from Dima's floor to the first floor -/
def descent_time : ℕ := 60

/-- Total time (in seconds) for the upward journey -/
def ascent_time : ℕ := 70

/-- Proposition stating that Dima lives on the 7th floor given the conditions -/
theorem dima_lives_on_seventh_floor :
  dimas_floor = 7 ∧
  max_reachable_floor = 6 ∧
  building_stories = 9 ∧
  descent_time = 60 ∧
  ascent_time = 70 ∧
  (5 * dimas_floor = 6 * max_reachable_floor + 1) :=
by sorry

end dima_lives_on_seventh_floor_l3766_376620


namespace complement_of_A_in_I_l3766_376612

def I : Set ℕ := {1,2,3,4,5,6,7}
def A : Set ℕ := {2,4,6,7}

theorem complement_of_A_in_I :
  I \ A = {1,3,5} := by sorry

end complement_of_A_in_I_l3766_376612


namespace shopkeeper_profit_l3766_376657

/-- Calculates the profit percentage when selling n articles at the cost price of m articles -/
def profit_percentage (n m : ℕ) : ℚ :=
  (m - n) / n * 100

/-- Theorem: When a shopkeeper sells 10 articles at the cost price of 12 articles, the profit percentage is 20% -/
theorem shopkeeper_profit : profit_percentage 10 12 = 20 := by
  sorry

end shopkeeper_profit_l3766_376657


namespace number_division_problem_l3766_376607

theorem number_division_problem (x y : ℚ) 
  (h1 : (x - 5) / 7 = 7) 
  (h2 : (x - 6) / y = 6) : 
  y = 8 := by sorry

end number_division_problem_l3766_376607


namespace jonah_profit_l3766_376691

/-- Calculates the profit from selling pineapple rings given the following conditions:
  * Number of pineapples bought
  * Cost per pineapple
  * Number of rings per pineapple
  * Number of rings sold as a set
  * Price per set of rings
-/
def calculate_profit (num_pineapples : ℕ) (cost_per_pineapple : ℕ) 
                     (rings_per_pineapple : ℕ) (rings_per_set : ℕ) 
                     (price_per_set : ℕ) : ℕ :=
  let total_cost := num_pineapples * cost_per_pineapple
  let total_rings := num_pineapples * rings_per_pineapple
  let num_sets := total_rings / rings_per_set
  let total_revenue := num_sets * price_per_set
  total_revenue - total_cost

/-- Proves that Jonah's profit is $342 given the specified conditions -/
theorem jonah_profit : 
  calculate_profit 6 3 12 4 5 = 342 := by
  sorry

end jonah_profit_l3766_376691


namespace inequality_preservation_l3766_376632

theorem inequality_preservation (a b : ℝ) (h : a > b) : (1/3)*a - 1 > (1/3)*b - 1 := by
  sorry

end inequality_preservation_l3766_376632


namespace koala_fiber_intake_l3766_376688

/-- Given that a koala absorbs 30% of the fiber it eats and absorbed 12 ounces of fiber in one day,
    prove that the total amount of fiber eaten by the koala that day was 40 ounces. -/
theorem koala_fiber_intake (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_amount : ℝ) 
    (h1 : absorption_rate = 0.30)
    (h2 : absorbed_amount = 12)
    (h3 : absorbed_amount = absorption_rate * total_amount) :
  total_amount = 40 := by
  sorry

end koala_fiber_intake_l3766_376688


namespace first_number_proof_l3766_376694

theorem first_number_proof (x : ℝ) : x + 33 + 333 + 3.33 = 369.63 → x = 0.30 := by
  sorry

end first_number_proof_l3766_376694
