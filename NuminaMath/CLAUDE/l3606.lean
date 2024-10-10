import Mathlib

namespace factor_count_8100_l3606_360684

def number_to_factor : ℕ := 8100

/-- The number of positive factors of a natural number n -/
def count_factors (n : ℕ) : ℕ := sorry

theorem factor_count_8100 : count_factors number_to_factor = 45 := by sorry

end factor_count_8100_l3606_360684


namespace function_comparison_l3606_360634

theorem function_comparison
  (a : ℝ)
  (h_a_lower : -3 < a)
  (h_a_upper : a < 0)
  (x₁ x₂ : ℝ)
  (h_x_order : x₁ < x₂)
  (h_x_sum : x₁ + x₂ ≠ 1 + a)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + 2 * a * x + 4)
  : f x₁ > f x₂ := by
  sorry

end function_comparison_l3606_360634


namespace circular_rug_middle_ring_area_l3606_360673

theorem circular_rug_middle_ring_area :
  ∀ (inner_radius middle_radius outer_radius : ℝ)
    (inner_area middle_area outer_area : ℝ),
  inner_radius = 1 →
  middle_radius = inner_radius + 1 →
  outer_radius = middle_radius + 1 →
  inner_area = π * inner_radius^2 →
  middle_area = π * middle_radius^2 →
  outer_area = π * outer_radius^2 →
  middle_area - inner_area = 3 * π := by
sorry

end circular_rug_middle_ring_area_l3606_360673


namespace midpoint_of_complex_line_segment_l3606_360604

theorem midpoint_of_complex_line_segment :
  let z₁ : ℂ := -7 + 5*I
  let z₂ : ℂ := 5 - 3*I
  let midpoint := (z₁ + z₂) / 2
  midpoint = -1 + I :=
by sorry

end midpoint_of_complex_line_segment_l3606_360604


namespace parabola_shift_theorem_l3606_360613

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := p.b - 2 * p.a * h
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 1 ∧ p.b = -4 ∧ p.c = -4 →
  let p' := shift p 3 3
  p'.a = 1 ∧ p'.b = 2 ∧ p'.c = -5 := by sorry

end parabola_shift_theorem_l3606_360613


namespace cornbread_pieces_count_l3606_360618

/-- Represents the dimensions of a rectangular pan --/
structure PanDimensions where
  length : ℕ
  width : ℕ

/-- Represents the dimensions of a piece of cornbread --/
structure PieceDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of whole pieces that can be cut from a pan --/
def maxPieces (pan : PanDimensions) (piece : PieceDimensions) (margin : ℕ) : ℕ :=
  ((pan.length / piece.length) * ((pan.width - margin) / piece.width))

/-- Theorem stating that the maximum number of pieces for the given dimensions is 72 --/
theorem cornbread_pieces_count :
  let pan := PanDimensions.mk 24 20
  let piece := PieceDimensions.mk 3 2
  let margin := 1
  maxPieces pan piece margin = 72 := by
  sorry


end cornbread_pieces_count_l3606_360618


namespace inverse_proportion_problem_l3606_360640

theorem inverse_proportion_problem (k : ℝ) (a b : ℝ → ℝ) :
  (∀ x, a x * (b x)^2 = k) →  -- Inverse proportion relationship
  (∃ x, a x = 40) →           -- a = 40 for some value of b
  (a (b 10) = 10) →           -- When a = 10
  b 10 = 2                    -- b = 2
:= by sorry

end inverse_proportion_problem_l3606_360640


namespace inverse_89_mod_90_l3606_360693

theorem inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 89 ∧ (89 * x) % 90 = 1 := by
  sorry

end inverse_89_mod_90_l3606_360693


namespace alpha_range_l3606_360624

theorem alpha_range (α : Real) (k : Int) 
  (h1 : Real.sin α > 0)
  (h2 : Real.cos α < 0)
  (h3 : Real.sin (α / 3) > Real.cos (α / 3)) :
  ∃ k, (α / 3 ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 4) (2 * k * Real.pi + Real.pi / 3)) ∨
       (α / 3 ∈ Set.Ioo (2 * k * Real.pi + 5 * Real.pi / 6) (2 * k * Real.pi + Real.pi)) :=
by sorry

end alpha_range_l3606_360624


namespace coefficient_x5_in_expansion_l3606_360600

/-- The coefficient of x^5 in the expansion of (x^3 + 1/x)^7 is 35 -/
theorem coefficient_x5_in_expansion : Nat := by
  sorry

end coefficient_x5_in_expansion_l3606_360600


namespace smallest_non_factor_product_of_72_l3606_360695

theorem smallest_non_factor_product_of_72 (x y : ℕ+) : 
  x ≠ y →
  x ∣ 72 →
  y ∣ 72 →
  ¬(x * y ∣ 72) →
  (∀ a b : ℕ+, a ≠ b → a ∣ 72 → b ∣ 72 → ¬(a * b ∣ 72) → x * y ≤ a * b) →
  x * y = 32 := by
sorry

end smallest_non_factor_product_of_72_l3606_360695


namespace mice_pairing_impossible_l3606_360697

/-- Represents the number of mice in the family -/
def total_mice : ℕ := 24

/-- Represents the number of mice that go to the warehouse each night -/
def mice_per_night : ℕ := 4

/-- Represents the number of new pairings a mouse makes each night -/
def new_pairings_per_night : ℕ := mice_per_night - 1

/-- Represents the number of pairings each mouse needs to make -/
def required_pairings : ℕ := total_mice - 1

/-- Theorem stating that it's impossible for each mouse to pair with every other mouse exactly once -/
theorem mice_pairing_impossible : 
  ¬(required_pairings % new_pairings_per_night = 0) := by sorry

end mice_pairing_impossible_l3606_360697


namespace negation_of_exists_square_nonpositive_l3606_360629

theorem negation_of_exists_square_nonpositive :
  (¬ ∃ x : ℝ, x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0) := by
  sorry

end negation_of_exists_square_nonpositive_l3606_360629


namespace farmers_herd_size_l3606_360683

theorem farmers_herd_size :
  ∀ (total : ℚ),
  (2 / 5 : ℚ) * total + (1 / 5 : ℚ) * total + (1 / 10 : ℚ) * total + 9 = total →
  total = 30 := by
sorry

end farmers_herd_size_l3606_360683


namespace arithmetic_sequence_general_term_l3606_360646

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1) :=
by sorry

end arithmetic_sequence_general_term_l3606_360646


namespace unique_odd_number_with_remainder_l3606_360687

theorem unique_odd_number_with_remainder : 
  ∃! n : ℕ, 30 < n ∧ n < 50 ∧ n % 2 = 1 ∧ n % 7 = 2 :=
by
  -- The proof goes here
  sorry

end unique_odd_number_with_remainder_l3606_360687


namespace albino_deer_antlers_l3606_360690

theorem albino_deer_antlers (total_deer : ℕ) (albino_deer : ℕ) 
  (h1 : total_deer = 920)
  (h2 : albino_deer = 23)
  (h3 : albino_deer = (total_deer * 10 / 100) / 4) : 
  albino_deer = 23 := by
  sorry

end albino_deer_antlers_l3606_360690


namespace summer_jolly_degrees_l3606_360643

/-- The combined number of degrees for Summer and Jolly -/
def combined_degrees (summer_degrees : ℕ) (difference : ℕ) : ℕ :=
  summer_degrees + (summer_degrees - difference)

/-- Theorem: Given Summer has 150 degrees and 5 more degrees than Jolly,
    the combined number of degrees they both have is 295. -/
theorem summer_jolly_degrees :
  combined_degrees 150 5 = 295 := by
  sorry

end summer_jolly_degrees_l3606_360643


namespace exists_subset_with_unique_sum_representation_l3606_360609

theorem exists_subset_with_unique_sum_representation : 
  ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n :=
sorry

end exists_subset_with_unique_sum_representation_l3606_360609


namespace toms_promotion_expenses_l3606_360694

/-- Represents the problem of calculating Tom's promotion expenses --/
def TomsDoughBallPromotion (flour_needed : ℕ) (flour_bag_size : ℕ) (flour_bag_cost : ℕ)
  (salt_needed : ℕ) (salt_cost_per_pound : ℚ) (tickets_sold : ℕ) (ticket_price : ℕ) (profit : ℕ) : Prop :=
  let flour_bags := flour_needed / flour_bag_size
  let flour_cost := flour_bags * flour_bag_cost
  let salt_cost := (salt_needed : ℚ) * salt_cost_per_pound
  let revenue := tickets_sold * ticket_price
  let promotion_cost := revenue - profit - flour_cost - (salt_cost.num / salt_cost.den)
  promotion_cost = 1000

/-- The theorem stating that Tom's promotion expenses are $1000 --/
theorem toms_promotion_expenses :
  TomsDoughBallPromotion 500 50 20 10 (1/5) 500 20 8798 :=
sorry

end toms_promotion_expenses_l3606_360694


namespace sum_a_b_equals_negative_one_l3606_360622

theorem sum_a_b_equals_negative_one (a b : ℝ) (h : |a + 3| + (b - 2)^2 = 0) : a + b = -1 := by
  sorry

end sum_a_b_equals_negative_one_l3606_360622


namespace peanut_butter_jar_size_l3606_360681

/-- Calculates the size of the third jar given the total amount of peanut butter,
    the sizes of two jars, and the total number of jars. -/
def third_jar_size (total_peanut_butter : ℕ) (jar1_size jar2_size : ℕ) (total_jars : ℕ) : ℕ :=
  let jars_per_size := total_jars / 3
  let remaining_peanut_butter := total_peanut_butter - (jar1_size + jar2_size) * jars_per_size
  remaining_peanut_butter / jars_per_size

/-- Proves that given the conditions, the size of the third jar is 40 ounces. -/
theorem peanut_butter_jar_size :
  third_jar_size 252 16 28 9 = 40 := by
  sorry

end peanut_butter_jar_size_l3606_360681


namespace fraction_multiplication_equality_l3606_360644

theorem fraction_multiplication_equality : (1 / 2) * (1 / 3) * (1 / 4) * (1 / 6) * 144 = 1 := by
  sorry

end fraction_multiplication_equality_l3606_360644


namespace square_plus_inverse_square_l3606_360653

theorem square_plus_inverse_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  x^2 + 1/x^2 = 11 := by
  sorry

end square_plus_inverse_square_l3606_360653


namespace johns_expenses_exceed_earnings_l3606_360656

/-- Represents the percentage of John's earnings spent on each category -/
structure Expenses where
  rent : ℝ
  dishwasher : ℝ
  groceries : ℝ

/-- Calculates John's expenses based on the given conditions -/
def calculate_expenses (rent_percent : ℝ) : Expenses :=
  { rent := rent_percent,
    dishwasher := rent_percent - (0.3 * rent_percent),
    groceries := rent_percent + (0.15 * rent_percent) }

/-- Theorem stating that John's expenses exceed his earnings -/
theorem johns_expenses_exceed_earnings (rent_percent : ℝ) 
  (h1 : rent_percent = 0.4)  -- John spent 40% of his earnings on rent
  (h2 : rent_percent > 0)    -- Rent percentage is positive
  (h3 : rent_percent < 1)    -- Rent percentage is less than 100%
  : (calculate_expenses rent_percent).rent + 
    (calculate_expenses rent_percent).dishwasher + 
    (calculate_expenses rent_percent).groceries > 1 := by
  sorry

#check johns_expenses_exceed_earnings

end johns_expenses_exceed_earnings_l3606_360656


namespace price_per_ring_is_correct_l3606_360632

/-- Calculates the price per pineapple ring given the following conditions:
  * Number of pineapples bought
  * Cost per pineapple
  * Number of rings per pineapple
  * Number of rings sold together
  * Total profit
-/
def price_per_ring (num_pineapples : ℕ) (cost_per_pineapple : ℚ) 
                   (rings_per_pineapple : ℕ) (rings_per_set : ℕ) 
                   (total_profit : ℚ) : ℚ :=
  let total_cost := num_pineapples * cost_per_pineapple
  let total_rings := num_pineapples * rings_per_pineapple
  let total_revenue := total_cost + total_profit
  let num_sets := total_rings / rings_per_set
  let price_per_set := total_revenue / num_sets
  price_per_set / rings_per_set

/-- Theorem stating that the price per pineapple ring is $1.25 under the given conditions -/
theorem price_per_ring_is_correct : 
  price_per_ring 6 3 12 4 72 = 5/4 := by
  sorry

end price_per_ring_is_correct_l3606_360632


namespace point_difference_l3606_360623

/-- The value of a touchdown in points -/
def touchdown_value : ℕ := 7

/-- The number of touchdowns scored by Brayden and Gavin -/
def brayden_gavin_touchdowns : ℕ := 7

/-- The number of touchdowns scored by Cole and Freddy -/
def cole_freddy_touchdowns : ℕ := 9

/-- The point difference between Cole and Freddy's team and Brayden and Gavin's team -/
theorem point_difference : 
  cole_freddy_touchdowns * touchdown_value - brayden_gavin_touchdowns * touchdown_value = 14 :=
by sorry

end point_difference_l3606_360623


namespace louisa_second_day_travel_l3606_360659

/-- Proves that Louisa traveled 280 miles on the second day of her vacation --/
theorem louisa_second_day_travel :
  ∀ (first_day_distance second_day_distance : ℝ) 
    (speed : ℝ) 
    (time_difference : ℝ),
  first_day_distance = 160 →
  speed = 40 →
  time_difference = 3 →
  first_day_distance / speed + time_difference = second_day_distance / speed →
  second_day_distance = 280 := by
sorry

end louisa_second_day_travel_l3606_360659


namespace boat_travel_distance_l3606_360601

/-- Proves that a boat traveling upstream and downstream with given conditions travels 91.25 miles -/
theorem boat_travel_distance (v : ℝ) (d : ℝ) : 
  d / (v - 3) = d / (v + 3) + 0.5 →
  d / (v + 3) = 2.5191640969412834 →
  d = 91.25 := by
  sorry

end boat_travel_distance_l3606_360601


namespace fraction_equality_l3606_360620

theorem fraction_equality (a b c d : ℝ) : 
  (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4 →
  (b - a) * (d - c) / ((c - b) * (a - d)) = -4 / 3 := by
sorry

end fraction_equality_l3606_360620


namespace quadratic_residue_criterion_l3606_360602

theorem quadratic_residue_criterion (p a : ℕ) (hp : Prime p) (hp2 : p ≠ 2) (ha : a ≠ 0) :
  (∃ x, x^2 ≡ a [ZMOD p]) → a^((p-1)/2) ≡ 1 [ZMOD p] ∧
  (¬∃ x, x^2 ≡ a [ZMOD p]) → a^((p-1)/2) ≡ -1 [ZMOD p] := by
  sorry

end quadratic_residue_criterion_l3606_360602


namespace cylinder_surface_area_l3606_360669

/-- The total surface area of a cylinder with height 12 and radius 5 is 170π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 12
  let r : ℝ := 5
  let lateral_area := 2 * Real.pi * r * h
  let base_area := 2 * Real.pi * r^2
  lateral_area + base_area = 170 * Real.pi :=
by
  sorry

end cylinder_surface_area_l3606_360669


namespace angle_equation_l3606_360621

theorem angle_equation (α β : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : (Real.sin α + Real.cos α) * (Real.sin β + Real.cos β) = 2) :
  (Real.sin (2 * α) + Real.cos (3 * β))^2 + (Real.sin (2 * β) + Real.cos (3 * α))^2 = 3 - 2 * Real.sqrt 2 := by
  sorry

end angle_equation_l3606_360621


namespace magician_payment_calculation_l3606_360610

/-- The total amount paid to a magician given their hourly rate, daily hours, and number of weeks worked -/
def magician_payment (hourly_rate : ℕ) (daily_hours : ℕ) (weeks : ℕ) : ℕ :=
  hourly_rate * daily_hours * 7 * weeks

/-- Theorem stating that a magician charging $60 per hour, working 3 hours daily for 2 weeks, earns $2520 -/
theorem magician_payment_calculation :
  magician_payment 60 3 2 = 2520 := by
  sorry

#eval magician_payment 60 3 2

end magician_payment_calculation_l3606_360610


namespace santiago_has_58_roses_l3606_360661

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The number of additional roses Mrs. Santiago has compared to Mrs. Garrett -/
def additional_roses : ℕ := 34

/-- The total number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := garrett_roses + additional_roses

/-- Theorem stating that Mrs. Santiago has 58 red roses -/
theorem santiago_has_58_roses : santiago_roses = 58 := by
  sorry

end santiago_has_58_roses_l3606_360661


namespace M_intersect_N_is_empty_l3606_360663

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x + 2}

-- Define the set N
def N : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Theorem statement
theorem M_intersect_N_is_empty : M ∩ (N.image Prod.snd) = ∅ := by
  sorry

end M_intersect_N_is_empty_l3606_360663


namespace stone_reduction_moves_l3606_360605

theorem stone_reduction_moves (n : ℕ) (h : n = 2005) : 
  ∃ (m : ℕ), m = 11 ∧ 
  (∀ (k : ℕ), k < m → 2^(m - k) ≥ n) ∧
  (2^(m - m) < n) :=
sorry

end stone_reduction_moves_l3606_360605


namespace angle_D_measure_l3606_360698

-- Define a scalene triangle DEF
structure ScaleneTriangle where
  D : ℝ
  E : ℝ
  F : ℝ
  scalene : D ≠ E ∧ E ≠ F ∧ D ≠ F
  sum_180 : D + E + F = 180

-- Theorem statement
theorem angle_D_measure (t : ScaleneTriangle) 
  (h1 : t.D = 2 * t.E) 
  (h2 : t.F = t.E - 20) : 
  t.D = 100 := by
  sorry

end angle_D_measure_l3606_360698


namespace laptop_price_l3606_360666

theorem laptop_price (deposit : ℝ) (deposit_percentage : ℝ) (full_price : ℝ) 
  (h1 : deposit = 400)
  (h2 : deposit_percentage = 25)
  (h3 : deposit = (deposit_percentage / 100) * full_price) :
  full_price = 1600 := by
sorry

end laptop_price_l3606_360666


namespace quadratic_coefficient_determination_l3606_360642

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- Theorem: If a quadratic function has vertex at (2, 5) and passes through (1, 2), then a = -3 -/
theorem quadratic_coefficient_determination (q : QuadraticFunction) 
  (vertex : q.f 2 = 5) 
  (point : q.f 1 = 2) : 
  q.a = -3 := by
  sorry

end quadratic_coefficient_determination_l3606_360642


namespace integer_roots_l3606_360630

/-- A polynomial of degree 3 with integer coefficients -/
def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂ * x^2 + a₁ * x - 13

/-- The set of possible integer roots -/
def possible_roots : Set ℤ := {-13, -1, 1, 13}

/-- Theorem stating that the possible integer roots of the polynomial are -13, -1, 1, and 13 -/
theorem integer_roots (a₂ a₁ : ℤ) :
  ∀ x : ℤ, polynomial a₂ a₁ x = 0 → x ∈ possible_roots :=
by sorry

end integer_roots_l3606_360630


namespace angle_AFE_measure_l3606_360606

-- Define the points
variable (A B C D E F : Point)

-- Define the rectangle ABCD
def is_rectangle (A B C D : Point) : Prop := sorry

-- Define the relationship AB = 2BC
def AB_twice_BC (A B C : Point) : Prop := sorry

-- Define E on the opposite half-plane from A with respect to CD
def E_opposite_halfplane (A C D E : Point) : Prop := sorry

-- Define angle CDE = 120°
def angle_CDE_120 (C D E : Point) : Prop := sorry

-- Define F as midpoint of AD
def F_midpoint_AD (A D F : Point) : Prop := sorry

-- Define the measure of an angle
def angle_measure (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem angle_AFE_measure
  (h_rect : is_rectangle A B C D)
  (h_AB_BC : AB_twice_BC A B C)
  (h_E_opp : E_opposite_halfplane A C D E)
  (h_CDE : angle_CDE_120 C D E)
  (h_F_mid : F_midpoint_AD A D F) :
  angle_measure A F E = 150 := by sorry

end angle_AFE_measure_l3606_360606


namespace quadratic_inequality_implies_not_roots_l3606_360615

theorem quadratic_inequality_implies_not_roots (a b x : ℝ) :
  x^2 - (a + b)*x + a*b ≠ 0 → ¬(x = a ∨ x = b) := by
  sorry

end quadratic_inequality_implies_not_roots_l3606_360615


namespace x_minus_y_equals_four_l3606_360692

theorem x_minus_y_equals_four (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
  sorry

end x_minus_y_equals_four_l3606_360692


namespace moon_speed_km_per_hour_l3606_360664

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 0.2

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: The moon's speed in kilometers per hour -/
theorem moon_speed_km_per_hour :
  moon_speed_km_per_sec * (seconds_per_hour : ℝ) = 720 := by
  sorry

end moon_speed_km_per_hour_l3606_360664


namespace remainder_problem_l3606_360612

theorem remainder_problem (N : ℤ) : ∃ (k : ℤ), N = 296 * k + 75 → ∃ (m : ℤ), N = 37 * m + 1 := by
  sorry

end remainder_problem_l3606_360612


namespace base9_addition_theorem_l3606_360689

/-- Addition of numbers in base 9 --/
def base9_add (a b c : ℕ) : ℕ :=
  sorry

/-- Conversion from base 9 to base 10 --/
def base9_to_base10 (n : ℕ) : ℕ :=
  sorry

theorem base9_addition_theorem :
  base9_add 2175 1714 406 = 4406 :=
by sorry

end base9_addition_theorem_l3606_360689


namespace solve_for_p_l3606_360660

theorem solve_for_p (n m p : ℚ) : 
  (5 / 6 : ℚ) = n / 72 ∧ 
  (5 / 6 : ℚ) = (m + n) / 84 ∧ 
  (5 / 6 : ℚ) = (p - m) / 120 → 
  p = 110 := by
sorry

end solve_for_p_l3606_360660


namespace fiftieth_parenthesis_sum_l3606_360641

def sequence_term (n : ℕ) : ℕ := 24 * ((n - 1) / 4) + 1

def parenthesis_sum (n : ℕ) : ℕ :=
  if n % 4 = 1 then sequence_term n
  else if n % 4 = 2 then sequence_term n + (sequence_term n + 2)
  else if n % 4 = 3 then sequence_term n + (sequence_term n + 2) + (sequence_term n + 4)
  else sequence_term n

theorem fiftieth_parenthesis_sum : parenthesis_sum 50 = 392 := by sorry

end fiftieth_parenthesis_sum_l3606_360641


namespace probability_at_least_one_female_l3606_360668

theorem probability_at_least_one_female (total : ℕ) (male : ℕ) (female : ℕ) (selected : ℕ) :
  total = male + female →
  selected = 3 →
  male = 6 →
  female = 4 →
  (1 - (Nat.choose male selected / Nat.choose total selected : ℚ)) = 5/6 :=
sorry

end probability_at_least_one_female_l3606_360668


namespace factorial_starts_with_1966_l3606_360674

theorem factorial_starts_with_1966 : ∃ k : ℕ, ∃ n : ℕ, 
  1966 * 10^n ≤ k! ∧ k! < 1967 * 10^n :=
sorry

end factorial_starts_with_1966_l3606_360674


namespace max_value_theorem_l3606_360635

theorem max_value_theorem (a : ℝ) (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ (max_val : ℝ), max_val = 1/4 ∧ (3 * a + 1 ≤ max_val) :=
by sorry

end max_value_theorem_l3606_360635


namespace simplify_expression_l3606_360658

theorem simplify_expression (x : ℝ) : 
  x * (3 * x^2 - 2) - 5 * (x^2 - 2*x + 7) = 3 * x^3 - 5 * x^2 + 8 * x - 35 := by
  sorry

end simplify_expression_l3606_360658


namespace range_of_a_l3606_360678

theorem range_of_a (x a : ℝ) : 
  (∀ x, (|x + a| < 3 ↔ 2 < x ∧ x < 3)) → 
  -5 ≤ a ∧ a ≤ 0 :=
by sorry

end range_of_a_l3606_360678


namespace basketball_probabilities_l3606_360657

/-- Represents a basketball player's shooting accuracy -/
structure Player where
  accuracy : ℝ
  accuracy_nonneg : 0 ≤ accuracy
  accuracy_le_one : accuracy ≤ 1

/-- The probability of a player hitting at least one shot in two attempts -/
def prob_hit_at_least_one (player : Player) : ℝ :=
  1 - (1 - player.accuracy)^2

/-- The probability of two players making exactly three shots in four attempts -/
def prob_three_out_of_four (player_a player_b : Player) : ℝ :=
  2 * (player_a.accuracy * (1 - player_a.accuracy) * player_b.accuracy^2 +
       player_b.accuracy * (1 - player_b.accuracy) * player_a.accuracy^2)

theorem basketball_probabilities 
  (player_a : Player) 
  (player_b : Player) 
  (h_a : player_a.accuracy = 1/2) 
  (h_b : (1 - player_b.accuracy)^2 = 1/16) :
  prob_hit_at_least_one player_a = 3/4 ∧ 
  prob_three_out_of_four player_a player_b = 3/8 := by
  sorry

end basketball_probabilities_l3606_360657


namespace sin_cos_identity_l3606_360672

theorem sin_cos_identity (x : ℝ) (h : 3 * Real.sin x + Real.cos x = 0) :
  Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + Real.cos x ^ 2 = 2 / 5 := by
  sorry

end sin_cos_identity_l3606_360672


namespace angle_between_vectors_l3606_360671

/-- The angle between two vectors in degrees -/
def angle_between (u v : ℝ × ℝ) : ℝ := sorry

/-- The dot product of two 2D vectors -/
def dot_product (u v : ℝ × ℝ) : ℝ := sorry

/-- The magnitude (length) of a 2D vector -/
def magnitude (u : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, -4)
  ∀ c : ℝ × ℝ,
    magnitude c = Real.sqrt 5 →
    dot_product (a.1 + b.1, a.2 + b.2) c = 5/2 →
    angle_between a c = 120 := by sorry

end angle_between_vectors_l3606_360671


namespace smallest_other_integer_l3606_360676

theorem smallest_other_integer (m n x : ℕ+) : 
  m = 36 → 
  Nat.gcd m n = x + 5 → 
  Nat.lcm m n = x * (x + 5) → 
  ∃ (n_min : ℕ+), n_min ≤ n ∧ n_min = 1 := by
  sorry

end smallest_other_integer_l3606_360676


namespace equation_equals_twentyfour_l3606_360677

theorem equation_equals_twentyfour : 8 / (3 - 8 / 3) = 24 := by
  sorry

#check equation_equals_twentyfour

end equation_equals_twentyfour_l3606_360677


namespace largest_c_for_max_function_l3606_360665

open Real

/-- Given positive real numbers a and b, the largest real c such that 
    c ≤ max(ax + 1/(ax), bx + 1/(bx)) for all positive real x is 2. -/
theorem largest_c_for_max_function (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ c : ℝ, ∀ x : ℝ, x > 0 → c ≤ max (a * x + 1 / (a * x)) (b * x + 1 / (b * x))) ∧
  (∀ c : ℝ, (∀ x : ℝ, x > 0 → c ≤ max (a * x + 1 / (a * x)) (b * x + 1 / (b * x))) → c ≤ 2) :=
by sorry

end largest_c_for_max_function_l3606_360665


namespace sarah_picked_45_apples_l3606_360628

/-- The number of apples Sarah's brother picked -/
def brother_apples : ℝ := 9.0

/-- The factor by which Sarah picked more apples than her brother -/
def sarah_factor : ℕ := 5

/-- The number of apples Sarah picked -/
def sarah_apples : ℝ := brother_apples * sarah_factor

theorem sarah_picked_45_apples : sarah_apples = 45 := by
  sorry

end sarah_picked_45_apples_l3606_360628


namespace percentage_of_singles_is_70_percent_l3606_360685

def total_hits : ℕ := 50
def home_runs : ℕ := 3
def triples : ℕ := 2
def doubles : ℕ := 10

def singles : ℕ := total_hits - (home_runs + triples + doubles)

theorem percentage_of_singles_is_70_percent :
  (singles : ℚ) / total_hits * 100 = 70 := by sorry

end percentage_of_singles_is_70_percent_l3606_360685


namespace ratio_difference_l3606_360607

theorem ratio_difference (a b c : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : ∃ (x : ℕ), a = 3 * x ∧ b = 5 * x ∧ c = 7 * x) (h3 : c = 70) :
  c - a = 40 := by
  sorry

end ratio_difference_l3606_360607


namespace parabola_y_intercepts_l3606_360655

/-- The number of y-intercepts of the parabola x = 3y^2 - 4y + 2 -/
theorem parabola_y_intercepts :
  let f : ℝ → ℝ := fun y => 3 * y^2 - 4 * y + 2
  (∃ y, f y = 0) = false :=
by sorry

end parabola_y_intercepts_l3606_360655


namespace system_solution_l3606_360608

theorem system_solution : ∃ (x y : ℚ), 
  (7 * x - 14 * y = 3) ∧ 
  (3 * y - x = 5) ∧ 
  (x = 79/7) ∧ 
  (y = 38/7) := by
  sorry

end system_solution_l3606_360608


namespace no_sum_2017_double_digits_l3606_360631

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the impossibility of expressing 2017 as the sum of two natural numbers
    where the sum of digits of one is twice the sum of digits of the other -/
theorem no_sum_2017_double_digits : ¬ ∃ (A B : ℕ), 
  (A + B = 2017) ∧ (sumOfDigits A = 2 * sumOfDigits B) := by
  sorry

end no_sum_2017_double_digits_l3606_360631


namespace constant_sum_inverse_lengths_l3606_360647

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- Define a line passing through F
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define a perpendicular line passing through F
def perp_line_through_F (k : ℝ) (x y : ℝ) : Prop := y = -(1/k) * (x - 1)

-- Define the theorem
theorem constant_sum_inverse_lengths 
  (k : ℝ) 
  (A B C D : ℝ × ℝ) 
  (hA : curve A.1 A.2) (hB : curve B.1 B.2) (hC : curve C.1 C.2) (hD : curve D.1 D.2)
  (hAB : line_through_F k A.1 A.2 ∧ line_through_F k B.1 B.2)
  (hCD : perp_line_through_F k C.1 C.2 ∧ perp_line_through_F k D.1 D.2)
  (hk : k ≠ 0) :
  1 / Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
  1 / Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 17/48 :=
sorry

end constant_sum_inverse_lengths_l3606_360647


namespace smallest_covering_radius_l3606_360679

theorem smallest_covering_radius :
  let r : ℝ := Real.sqrt 3 / 2
  ∀ s : ℝ, s < r → ¬(∃ (c₁ c₂ c₃ : ℝ × ℝ),
    (∀ p : ℝ × ℝ, Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ 1 →
      Real.sqrt ((p.1 - c₁.1)^2 + (p.2 - c₁.2)^2) ≤ s ∨
      Real.sqrt ((p.1 - c₂.1)^2 + (p.2 - c₂.2)^2) ≤ s ∨
      Real.sqrt ((p.1 - c₃.1)^2 + (p.2 - c₃.2)^2) ≤ s)) ∧
  ∃ (c₁ c₂ c₃ : ℝ × ℝ),
    (∀ p : ℝ × ℝ, Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ 1 →
      Real.sqrt ((p.1 - c₁.1)^2 + (p.2 - c₁.2)^2) ≤ r ∨
      Real.sqrt ((p.1 - c₂.1)^2 + (p.2 - c₂.2)^2) ≤ r ∨
      Real.sqrt ((p.1 - c₃.1)^2 + (p.2 - c₃.2)^2) ≤ r) :=
by sorry

end smallest_covering_radius_l3606_360679


namespace f_ln_2_equals_neg_1_l3606_360611

-- Define the base of natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_ln_2_equals_neg_1 
  (h_monotonic : Monotone f)
  (h_condition : ∀ x : ℝ, f (f x + Real.exp x) = 1 - e) :
  f (Real.log 2) = -1 := by sorry

end f_ln_2_equals_neg_1_l3606_360611


namespace topsoil_cost_l3606_360662

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards : ℝ := 8

/-- The total cost of topsoil in dollars -/
def total_cost : ℝ := cost_per_cubic_foot * cubic_feet_per_cubic_yard * cubic_yards

theorem topsoil_cost : total_cost = 1728 := by
  sorry

end topsoil_cost_l3606_360662


namespace min_value_expression_l3606_360603

theorem min_value_expression (u v : ℝ) : 
  (u - v)^2 + (Real.sqrt (4 - u^2) - 2*v - 5)^2 ≥ 9 - 4 * Real.sqrt 5 := by
sorry

end min_value_expression_l3606_360603


namespace problem_1_problem_2_l3606_360614

-- Problem 1
theorem problem_1 : 
  (Real.sqrt 24 - Real.sqrt (1/2)) - (Real.sqrt (1/8) + Real.sqrt 6) = Real.sqrt 6 - (3 * Real.sqrt 2) / 4 := by
  sorry

-- Problem 2
theorem problem_2 : 
  ∀ x : ℝ, (x - 2)^2 = 3*(x - 2) ↔ x = 2 ∨ x = 5 := by
  sorry

end problem_1_problem_2_l3606_360614


namespace interest_percentage_of_face_value_l3606_360651

-- Define the bond parameters
def face_value : ℝ := 5000
def selling_price : ℝ := 6153.846153846153
def interest_rate_of_selling_price : ℝ := 0.065

-- Define the theorem
theorem interest_percentage_of_face_value :
  let interest := interest_rate_of_selling_price * selling_price
  let interest_percentage_of_face := (interest / face_value) * 100
  interest_percentage_of_face = 8 := by sorry

end interest_percentage_of_face_value_l3606_360651


namespace certain_number_proof_l3606_360633

theorem certain_number_proof : ∃ X : ℝ, 
  0.8 * X = 0.7 * 60.00000000000001 + 30 ∧ X = 90.00000000000001 := by
  sorry

end certain_number_proof_l3606_360633


namespace probability_three_primes_is_correct_l3606_360619

def num_dice : ℕ := 7
def faces_per_die : ℕ := 10
def num_primes_per_die : ℕ := 4

def probability_exactly_three_primes : ℚ :=
  (num_dice.choose 3) *
  (num_primes_per_die / faces_per_die) ^ 3 *
  ((faces_per_die - num_primes_per_die) / faces_per_die) ^ (num_dice - 3)

theorem probability_three_primes_is_correct :
  probability_exactly_three_primes = 9072 / 31250 := by
  sorry

end probability_three_primes_is_correct_l3606_360619


namespace tan_alpha_minus_2beta_l3606_360680

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2/5)
  (h2 : Real.tan β = 1/2) :
  Real.tan (α - 2*β) = -1/12 := by
  sorry

end tan_alpha_minus_2beta_l3606_360680


namespace presidentAndCommittee_ten_l3606_360686

/-- The number of ways to choose a president and a 3-person committee from a group of n people,
    where the president cannot be on the committee and the order of choosing committee members does not matter. -/
def presidentAndCommittee (n : ℕ) : ℕ :=
  n * (n - 1).choose 3

/-- Theorem stating that for a group of 10 people, there are 840 ways to choose a president
    and a 3-person committee under the given conditions. -/
theorem presidentAndCommittee_ten :
  presidentAndCommittee 10 = 840 := by
  sorry

end presidentAndCommittee_ten_l3606_360686


namespace ratio_problem_l3606_360627

theorem ratio_problem (a b c : ℝ) : 
  a / b = 2 / 3 ∧ b / c = 3 / 4 ∧ a^2 + c^2 = 180 → b = 9 := by
sorry

end ratio_problem_l3606_360627


namespace square_inequality_l3606_360691

theorem square_inequality (a b : ℝ) : a > b ∧ b > 0 → a^2 > b^2 ∧ ¬(∀ x y : ℝ, x^2 > y^2 → x > y ∧ y > 0) := by
  sorry

end square_inequality_l3606_360691


namespace car_speed_comparison_l3606_360616

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  let x := 2 / (1 / u + 1 / v)
  let y := (u + v) / 2
  x ≤ y := by
sorry

end car_speed_comparison_l3606_360616


namespace school_gender_ratio_l3606_360652

/-- Given a school with a 5:4 ratio of boys to girls and 1500 boys, prove there are 1200 girls -/
theorem school_gender_ratio (num_boys : ℕ) (num_girls : ℕ) : 
  num_boys = 1500 →
  (5 : ℚ) / 4 = num_boys / num_girls →
  num_girls = 1200 := by
sorry

end school_gender_ratio_l3606_360652


namespace farm_legs_count_l3606_360625

/-- Represents the number of legs for each animal type -/
def legs_per_animal (animal : String) : Nat :=
  match animal with
  | "chicken" => 2
  | "buffalo" => 4
  | _ => 0

/-- Calculates the total number of legs in the farm -/
def total_legs (total_animals : Nat) (chickens : Nat) : Nat :=
  let buffalos := total_animals - chickens
  chickens * legs_per_animal "chicken" + buffalos * legs_per_animal "buffalo"

/-- Theorem: In a farm with 9 animals, including 5 chickens and the rest buffalos, there are 26 legs in total -/
theorem farm_legs_count : total_legs 9 5 = 26 := by
  sorry

end farm_legs_count_l3606_360625


namespace solution_set_for_t_equals_one_range_of_t_l3606_360696

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - t| + |x + t|

-- Part 1
theorem solution_set_for_t_equals_one :
  {x : ℝ | f 1 x ≤ 8 - x^2} = Set.Icc (-2) 2 := by sorry

-- Part 2
theorem range_of_t (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 4) :
  (∀ x : ℝ, f t x = (4 * m^2 + n) / (m * n)) →
  t ∈ Set.Iic (-9/8) ∪ Set.Ici (9/8) := by sorry

end solution_set_for_t_equals_one_range_of_t_l3606_360696


namespace isosceles_triangle_base_angle_l3606_360638

theorem isosceles_triangle_base_angle (α : ℝ) (h1 : α = 42) :
  let β := (180 - α) / 2
  (α = β) ∨ (β = 69) :=
sorry

end isosceles_triangle_base_angle_l3606_360638


namespace cards_given_to_jeff_l3606_360699

def initial_cards : ℕ := 573
def bought_cards : ℕ := 127
def cards_to_john : ℕ := 195
def cards_to_jimmy : ℕ := 75
def percentage_to_jeff : ℚ := 6 / 100
def final_cards : ℕ := 210

theorem cards_given_to_jeff :
  let total_cards := initial_cards + bought_cards
  let cards_after_john_jimmy := total_cards - (cards_to_john + cards_to_jimmy)
  let cards_to_jeff := (percentage_to_jeff * cards_after_john_jimmy).ceil
  cards_to_jeff = 26 ∧ 
  final_cards + cards_to_jeff = cards_after_john_jimmy :=
sorry

end cards_given_to_jeff_l3606_360699


namespace calculate_speed_l3606_360637

/-- Given two people moving in opposite directions, calculate the speed of one person given the other's speed and their final distance after a certain time. -/
theorem calculate_speed 
  (roja_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : roja_speed = 7)
  (h2 : time = 4)
  (h3 : final_distance = 40) :
  ∃ (pooja_speed : ℝ), pooja_speed = 3 ∧ (roja_speed + pooja_speed) * time = final_distance :=
by sorry

end calculate_speed_l3606_360637


namespace sum_of_fifth_powers_l3606_360688

theorem sum_of_fifth_powers (a b c : ℝ) (h : a + b + c = 0) :
  2 * (a^5 + b^5 + c^5) = 5 * a * b * c * (a^2 + b^2 + c^2) := by
  sorry

end sum_of_fifth_powers_l3606_360688


namespace tissue_paper_usage_l3606_360667

theorem tissue_paper_usage (initial : ℕ) (remaining : ℕ) (used : ℕ) : 
  initial = 97 → remaining = 93 → used = initial - remaining → used = 4 := by
  sorry

end tissue_paper_usage_l3606_360667


namespace cos_difference_given_sum_l3606_360682

theorem cos_difference_given_sum (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 0.75)
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = -0.21875 := by
sorry

end cos_difference_given_sum_l3606_360682


namespace repeating_decimal_56_equals_fraction_l3606_360654

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_56_equals_fraction :
  RepeatingDecimal 5 6 = 56 / 99 := by
  sorry

end repeating_decimal_56_equals_fraction_l3606_360654


namespace tim_golf_balls_l3606_360649

/-- The number of items in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of golf balls Tim has -/
def golf_ball_dozens : ℕ := 13

/-- The total number of golf balls Tim has -/
def total_golf_balls : ℕ := golf_ball_dozens * dozen

theorem tim_golf_balls : total_golf_balls = 156 := by
  sorry

end tim_golf_balls_l3606_360649


namespace largest_circle_area_l3606_360675

theorem largest_circle_area (length width : ℝ) (h1 : length = 18) (h2 : width = 8) :
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  (Real.pi * radius ^ 2) = 676 / Real.pi :=
by sorry

end largest_circle_area_l3606_360675


namespace apples_in_basket_A_l3606_360648

/-- The number of baskets -/
def num_baskets : ℕ := 5

/-- The average number of fruits per basket -/
def avg_fruits_per_basket : ℕ := 25

/-- The number of mangoes in basket B -/
def mangoes_in_B : ℕ := 30

/-- The number of peaches in basket C -/
def peaches_in_C : ℕ := 20

/-- The number of pears in basket D -/
def pears_in_D : ℕ := 25

/-- The number of bananas in basket E -/
def bananas_in_E : ℕ := 35

/-- The number of apples in basket A -/
def apples_in_A : ℕ := num_baskets * avg_fruits_per_basket - (mangoes_in_B + peaches_in_C + pears_in_D + bananas_in_E)

theorem apples_in_basket_A : apples_in_A = 15 := by
  sorry

end apples_in_basket_A_l3606_360648


namespace profit_maximization_l3606_360617

def profit_function (x : ℝ) : ℝ := -2 * x^2 + 200 * x - 3200

theorem profit_maximization (x : ℝ) :
  35 ≤ x ∧ x ≤ 45 →
  (∀ y : ℝ, 35 ≤ y ∧ y ≤ 45 → profit_function y ≤ profit_function 45) ∧
  profit_function 45 = 1750 ∧
  (∀ z : ℝ, 35 ≤ z ∧ z ≤ 45 ∧ profit_function z ≥ 1600 → 40 ≤ z ∧ z ≤ 45) :=
by sorry

end profit_maximization_l3606_360617


namespace sufficient_not_necessary_condition_l3606_360650

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) :=
sorry

end sufficient_not_necessary_condition_l3606_360650


namespace sum_of_four_squares_equals_prime_multiple_l3606_360645

theorem sum_of_four_squares_equals_prime_multiple (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (m : Nat) (x₁ x₂ x₃ x₄ : Int), 
    m < p ∧ 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 = m * p ∧ 
    (∀ (m' : Nat) (y₁ y₂ y₃ y₄ : Int), 
      m' < p → 
      y₁^2 + y₂^2 + y₃^2 + y₄^2 = m' * p → 
      m ≤ m') ∧
    m = 1 := by
  sorry

end sum_of_four_squares_equals_prime_multiple_l3606_360645


namespace math_contest_problem_count_l3606_360626

/-- Represents the number of problems solved by each participant -/
structure ParticipantSolutions where
  neznayka : ℕ
  pilyulkin : ℕ
  knopochka : ℕ
  vintik : ℕ
  znayka : ℕ

/-- Defines the conditions of the math contest -/
def MathContest (n : ℕ) (solutions : ParticipantSolutions) : Prop :=
  solutions.neznayka = 6 ∧
  solutions.znayka = 10 ∧
  solutions.pilyulkin > solutions.neznayka ∧
  solutions.pilyulkin < solutions.znayka ∧
  solutions.knopochka > solutions.neznayka ∧
  solutions.knopochka < solutions.znayka ∧
  solutions.vintik > solutions.neznayka ∧
  solutions.vintik < solutions.znayka ∧
  solutions.neznayka + solutions.pilyulkin + solutions.knopochka + solutions.vintik + solutions.znayka = 4 * n

theorem math_contest_problem_count (solutions : ParticipantSolutions) :
  ∃ n : ℕ, MathContest n solutions → n = 10 :=
by sorry

end math_contest_problem_count_l3606_360626


namespace hat_pairs_l3606_360636

/-- Given a group of 12 people where exactly 4 are wearing hats, 
    the number of pairs where at least one person is wearing a hat is 38. -/
theorem hat_pairs (total : ℕ) (hat_wearers : ℕ) (h1 : total = 12) (h2 : hat_wearers = 4) :
  (total.choose 2) - ((total - hat_wearers).choose 2) = 38 := by
  sorry

end hat_pairs_l3606_360636


namespace complex_equation_solution_l3606_360670

theorem complex_equation_solution :
  ∀ z : ℂ, (z - Complex.I) / z = Complex.I → z = -1/2 + Complex.I/2 := by
  sorry

end complex_equation_solution_l3606_360670


namespace passing_percentage_is_30_l3606_360639

def max_marks : ℕ := 600
def student_marks : ℕ := 80
def fail_margin : ℕ := 100

def passing_percentage : ℚ :=
  (student_marks + fail_margin : ℚ) / max_marks * 100

theorem passing_percentage_is_30 : passing_percentage = 30 := by
  sorry

end passing_percentage_is_30_l3606_360639
