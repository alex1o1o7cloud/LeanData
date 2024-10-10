import Mathlib

namespace fountain_position_l1060_106085

/-- Two towers with a fountain between them -/
structure TowerSetup where
  tower1_height : ℝ
  tower2_height : ℝ
  distance_between_towers : ℝ
  fountain_distance : ℝ

/-- The setup satisfies the problem conditions -/
def valid_setup (s : TowerSetup) : Prop :=
  s.tower1_height = 30 ∧
  s.tower2_height = 40 ∧
  s.distance_between_towers = 50 ∧
  0 < s.fountain_distance ∧
  s.fountain_distance < s.distance_between_towers

/-- The birds' flight paths are equal -/
def equal_flight_paths (s : TowerSetup) : Prop :=
  s.tower1_height^2 + s.fountain_distance^2 =
  s.tower2_height^2 + (s.distance_between_towers - s.fountain_distance)^2

theorem fountain_position (s : TowerSetup) 
  (h1 : valid_setup s) (h2 : equal_flight_paths s) :
  s.fountain_distance = 32 ∧ 
  s.distance_between_towers - s.fountain_distance = 18 := by
  sorry

end fountain_position_l1060_106085


namespace job_completion_time_l1060_106044

/-- The number of days it takes for two workers to complete a job together,
    given their individual work rates. -/
def days_to_complete (rate_a rate_b : ℚ) : ℚ :=
  1 / (rate_a + rate_b)

theorem job_completion_time 
  (rate_a rate_b : ℚ) 
  (h1 : rate_a = rate_b) 
  (h2 : rate_b = 1 / 12) : 
  days_to_complete rate_a rate_b = 6 := by
  sorry

end job_completion_time_l1060_106044


namespace geometric_sequence_a9_l1060_106031

/-- A geometric sequence with a_3 = 2 and a_5 = 6 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 3 = 2 ∧ a 5 = 6

theorem geometric_sequence_a9 (a : ℕ → ℝ) 
  (h : geometric_sequence a) : a 9 = 54 := by
  sorry

end geometric_sequence_a9_l1060_106031


namespace distance_between_points_l1060_106045

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 0)
  let p2 : ℝ × ℝ := (5, 9)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 10 := by
  sorry

end distance_between_points_l1060_106045


namespace parallel_line_through_point_l1060_106079

/-- A line in the 2D plane represented by its slope-intercept form y = mx + b -/
structure Line where
  slope : ℚ
  intercept : ℚ

def Line.through_point (l : Line) (x y : ℚ) : Prop :=
  y = l.slope * x + l.intercept

def Line.parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem parallel_line_through_point (given_line target_line : Line) 
    (h_parallel : given_line.parallel target_line)
    (h_through_point : target_line.through_point 3 0) :
  target_line = Line.mk (1/2) (-3/2) :=
sorry

end parallel_line_through_point_l1060_106079


namespace max_value_b_plus_c_l1060_106091

theorem max_value_b_plus_c (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a + c) * (b^2 + a*c) = 4*a) : 
  b + c ≤ 2 := by
sorry

end max_value_b_plus_c_l1060_106091


namespace gcf_lcm_sum_3_6_l1060_106058

theorem gcf_lcm_sum_3_6 : Nat.gcd 3 6 + Nat.lcm 3 6 = 9 := by
  sorry

end gcf_lcm_sum_3_6_l1060_106058


namespace percentage_of_indian_children_l1060_106052

theorem percentage_of_indian_children (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (percent_indian_men : ℚ) (percent_indian_women : ℚ) (percent_not_indian : ℚ)
  (h1 : total_men = 700)
  (h2 : total_women = 500)
  (h3 : total_children = 800)
  (h4 : percent_indian_men = 20 / 100)
  (h5 : percent_indian_women = 40 / 100)
  (h6 : percent_not_indian = 79 / 100) :
  (((1 - percent_not_indian) * (total_men + total_women + total_children) -
    percent_indian_men * total_men - percent_indian_women * total_women) /
    total_children : ℚ) = 10 / 100 := by
  sorry

end percentage_of_indian_children_l1060_106052


namespace rectangle_length_proof_l1060_106094

theorem rectangle_length_proof (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let new_area := (l - 5) * (w + 5)
  new_area = l * w + 75 → l = 40 := by
sorry

end rectangle_length_proof_l1060_106094


namespace range_of_a_for_increasing_f_l1060_106087

/-- A function f(x) = x³ + ax - 2 that is increasing on (1, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x - 2

/-- The derivative of f(x) -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x > 1, ∀ y > x, f a y > f a x) ↔ a ≥ -3 :=
sorry

end range_of_a_for_increasing_f_l1060_106087


namespace train_passing_platform_l1060_106096

/-- Calculates the time for a train to pass a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1500) 
  (h2 : tree_crossing_time = 120) 
  (h3 : platform_length = 500) : 
  (train_length + platform_length) / (train_length / tree_crossing_time) = 160 := by
  sorry

#check train_passing_platform

end train_passing_platform_l1060_106096


namespace this_year_sales_calculation_l1060_106077

def last_year_sales : ℝ := 320
def percent_increase : ℝ := 0.25

theorem this_year_sales_calculation :
  last_year_sales * (1 + percent_increase) = 400 := by
  sorry

end this_year_sales_calculation_l1060_106077


namespace complex_cube_absolute_value_l1060_106017

theorem complex_cube_absolute_value : 
  Complex.abs ((1 + 2 * Complex.I + 3 - Real.sqrt 3 * Complex.I) ^ 3) = 
  (23 - 4 * Real.sqrt 3) ^ (3/2) := by
  sorry

end complex_cube_absolute_value_l1060_106017


namespace intersection_with_complement_l1060_106019

def U : Set Nat := {2, 3, 4, 5, 6}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {2, 3, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {4} := by
  sorry

end intersection_with_complement_l1060_106019


namespace max_value_sin_function_l1060_106068

theorem max_value_sin_function :
  ∀ x : ℝ, -π/2 ≤ x ∧ x ≤ 0 →
  ∃ y_max : ℝ, y_max = 5 ∧
  ∀ y : ℝ, y = 3 * Real.sin x + 5 → y ≤ y_max :=
by sorry

end max_value_sin_function_l1060_106068


namespace subtracted_value_proof_l1060_106059

theorem subtracted_value_proof (x : ℕ) (h : x = 124) :
  ∃! y : ℕ, 2 * x - y = 110 :=
sorry

end subtracted_value_proof_l1060_106059


namespace unique_solution_condition_l1060_106033

theorem unique_solution_condition (k : ℚ) : 
  (∃! x : ℚ, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by
sorry

end unique_solution_condition_l1060_106033


namespace fraction_nonzero_digits_l1060_106086

def fraction := 800 / (2^5 * 5^11)

def count_nonzero_decimal_digits (x : ℚ) : ℕ := sorry

theorem fraction_nonzero_digits :
  count_nonzero_decimal_digits fraction = 3 := by sorry

end fraction_nonzero_digits_l1060_106086


namespace equation_system_solution_l1060_106049

theorem equation_system_solution : 
  ∀ (x y z : ℝ), 
    z ≠ 0 →
    3 * x - 4 * y - 2 * z = 0 →
    x - 2 * y + 5 * z = 0 →
    (2 * x^2 - x * y) / (y^2 + 4 * z^2) = 744 / 305 := by
  sorry

end equation_system_solution_l1060_106049


namespace base6_to_base10_conversion_l1060_106097

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The base 6 representation of the number -/
def base6Number : List Nat := [1, 2, 5, 4, 3]

theorem base6_to_base10_conversion :
  base6ToBase10 base6Number = 4945 := by
  sorry

end base6_to_base10_conversion_l1060_106097


namespace work_completion_time_l1060_106083

theorem work_completion_time (a_time b_time : ℝ) (work_left : ℝ) : 
  a_time = 15 → b_time = 20 → work_left = 0.7666666666666666 →
  (1 / a_time + 1 / b_time) * 2 = 1 - work_left := by sorry

end work_completion_time_l1060_106083


namespace poker_hand_probabilities_l1060_106051

-- Define the total number of possible 5-card hands
def total_hands : ℕ := 2598960

-- Define the number of ways to get each hand type
def pair_ways : ℕ := 1098240
def two_pair_ways : ℕ := 123552
def three_of_a_kind_ways : ℕ := 54912
def straight_ways : ℕ := 10000
def flush_ways : ℕ := 5108
def full_house_ways : ℕ := 3744
def four_of_a_kind_ways : ℕ := 624
def straight_flush_ways : ℕ := 40

-- Define the probability of each hand type
def prob_pair : ℚ := pair_ways / total_hands
def prob_two_pair : ℚ := two_pair_ways / total_hands
def prob_three_of_a_kind : ℚ := three_of_a_kind_ways / total_hands
def prob_straight : ℚ := straight_ways / total_hands
def prob_flush : ℚ := flush_ways / total_hands
def prob_full_house : ℚ := full_house_ways / total_hands
def prob_four_of_a_kind : ℚ := four_of_a_kind_ways / total_hands
def prob_straight_flush : ℚ := straight_flush_ways / total_hands

-- Theorem stating the probabilities of different poker hands
theorem poker_hand_probabilities :
  (prob_pair = 1098240 / 2598960) ∧
  (prob_two_pair = 123552 / 2598960) ∧
  (prob_three_of_a_kind = 54912 / 2598960) ∧
  (prob_straight = 10000 / 2598960) ∧
  (prob_flush = 5108 / 2598960) ∧
  (prob_full_house = 3744 / 2598960) ∧
  (prob_four_of_a_kind = 624 / 2598960) ∧
  (prob_straight_flush = 40 / 2598960) :=
by sorry

end poker_hand_probabilities_l1060_106051


namespace total_crayons_l1060_106018

theorem total_crayons (billy_crayons jane_crayons : ℝ) 
  (h1 : billy_crayons = 62.0) 
  (h2 : jane_crayons = 52.0) : 
  billy_crayons + jane_crayons = 114.0 := by
  sorry

end total_crayons_l1060_106018


namespace two_complex_roots_iff_k_values_l1060_106029

/-- The equation has exactly two complex roots if and only if k is 0, 2i, or -2i -/
theorem two_complex_roots_iff_k_values (k : ℂ) : 
  (∃! (r₁ r₂ : ℂ), ∀ (x : ℂ), x ≠ -3 ∧ x ≠ -4 → 
    (x / (x + 3) + x / (x + 4) = k * x ↔ x = 0 ∨ x = r₁ ∨ x = r₂)) ↔ 
  (k = 0 ∨ k = 2*I ∨ k = -2*I) :=
sorry

end two_complex_roots_iff_k_values_l1060_106029


namespace third_number_proof_l1060_106057

theorem third_number_proof (x : ℝ) : 0.3 * 0.8 + x * 0.5 = 0.29 → x = 0.1 := by
  sorry

end third_number_proof_l1060_106057


namespace tablespoons_in_half_cup_l1060_106095

/-- Proves that there are 8 tablespoons in half a cup of rice -/
theorem tablespoons_in_half_cup (grains_per_cup : ℕ) (teaspoons_per_tablespoon : ℕ) (grains_per_teaspoon : ℕ)
  (h1 : grains_per_cup = 480)
  (h2 : teaspoons_per_tablespoon = 3)
  (h3 : grains_per_teaspoon = 10) :
  (grains_per_cup / 2) / (grains_per_teaspoon * teaspoons_per_tablespoon) = 8 := by
  sorry

#check tablespoons_in_half_cup

end tablespoons_in_half_cup_l1060_106095


namespace complex_real_condition_l1060_106041

theorem complex_real_condition (m : ℝ) : 
  (∃ (z : ℂ), z = (m^2 - 5*m + 6 : ℝ) + (m - 3 : ℝ)*I ∧ z.im = 0) → m = 3 := by
  sorry

end complex_real_condition_l1060_106041


namespace last_two_digits_of_expression_l1060_106042

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_expression : 
  last_two_digits (sum_of_factorials 15 - factorial 5 * factorial 10 * factorial 15) = 13 := by
sorry

end last_two_digits_of_expression_l1060_106042


namespace polynomial_expansion_l1060_106037

theorem polynomial_expansion (x : ℝ) :
  (3*x^2 + 2*x - 5)*(x - 2) - (x - 2)*(x^2 - 5*x + 28) + (4*x - 7)*(x - 2)*(x + 4) =
  6*x^3 + 4*x^2 - 93*x + 122 := by
  sorry

end polynomial_expansion_l1060_106037


namespace variance_of_data_l1060_106006

/-- Given a list of 5 real numbers with an average of 5 and an average of squares of 33,
    prove that the variance of the list is 8. -/
theorem variance_of_data (x : List ℝ) (hx : x.length = 5)
  (h_avg : x.sum / 5 = 5)
  (h_avg_sq : (x.map (λ xi => xi^2)).sum / 5 = 33) :
  (x.map (λ xi => (xi - 5)^2)).sum / 5 = 8 := by
  sorry

end variance_of_data_l1060_106006


namespace sum_of_square_areas_l1060_106061

-- Define the triangle XYZ
def triangle_XYZ (X Y Z : ℝ × ℝ) : Prop :=
  let d := λ a b : ℝ × ℝ => Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  d X Z = 13 ∧ d X Y = 12 ∧ (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

-- Define the theorem
theorem sum_of_square_areas (X Y Z : ℝ × ℝ) (h : triangle_XYZ X Y Z) :
  (13 : ℝ)^2 + (Real.sqrt ((13 : ℝ)^2 - 12^2))^2 = 194 := by
  sorry


end sum_of_square_areas_l1060_106061


namespace x_to_y_value_l1060_106028

theorem x_to_y_value (x y : ℝ) (h : (x + 2)^2 + |y - 3| = 0) : x^y = -8 := by
  sorry

end x_to_y_value_l1060_106028


namespace parallel_vectors_x_value_l1060_106004

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b are parallel, prove that x = -6 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, 4)
  parallel a b → x = -6 := by sorry

end parallel_vectors_x_value_l1060_106004


namespace floor_expression_equals_twelve_l1060_106054

theorem floor_expression_equals_twelve (n : ℕ) (h : n = 1006) : 
  ⌊((n + 1)^3 / ((n - 1) * n) - (n - 1)^3 / (n * (n + 1)) + 5 : ℝ)⌋ = 12 := by
  sorry

end floor_expression_equals_twelve_l1060_106054


namespace square_has_perpendicular_diagonals_but_parallelogram_not_l1060_106000

-- Define a square
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define a parallelogram
structure Parallelogram :=
  (base : ℝ)
  (height : ℝ)
  (base_positive : base > 0)
  (height_positive : height > 0)

-- Define the property of perpendicular diagonals
def has_perpendicular_diagonals (S : Type) : Prop :=
  ∀ s : S, ∃ d₁ d₂ : ℝ × ℝ, d₁.1 * d₂.1 + d₁.2 * d₂.2 = 0

-- Theorem statement
theorem square_has_perpendicular_diagonals_but_parallelogram_not :
  (has_perpendicular_diagonals Square) ∧ ¬(has_perpendicular_diagonals Parallelogram) :=
sorry

end square_has_perpendicular_diagonals_but_parallelogram_not_l1060_106000


namespace house_wall_planks_l1060_106035

/-- Given the total number of nails, nails per plank, and additional nails used,
    calculate the number of planks needed. -/
def planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  (total_nails - additional_nails) / nails_per_plank

/-- Theorem stating that given the specific conditions, the number of planks needed is 1. -/
theorem house_wall_planks :
  planks_needed 11 3 8 = 1 := by
  sorry

end house_wall_planks_l1060_106035


namespace max_principals_l1060_106002

/-- Represents the number of years in a period -/
def period : ℕ := 10

/-- Represents the minimum term length for a principal -/
def minTerm : ℕ := 3

/-- Represents the maximum term length for a principal -/
def maxTerm : ℕ := 5

/-- Represents a valid principal term length -/
def ValidTerm (t : ℕ) : Prop := minTerm ≤ t ∧ t ≤ maxTerm

/-- 
Theorem: The maximum number of principals during a continuous 10-year period is 3,
given that each principal's term can be between 3 and 5 years.
-/
theorem max_principals :
  ∃ (n : ℕ) (terms : List ℕ),
    (∀ t ∈ terms, ValidTerm t) ∧ 
    (terms.sum ≥ period) ∧
    (terms.length = n) ∧
    (∀ m : ℕ, m > n → 
      ¬∃ (terms' : List ℕ), 
        (∀ t ∈ terms', ValidTerm t) ∧ 
        (terms'.sum ≥ period) ∧
        (terms'.length = m)) ∧
    n = 3 :=
  sorry

end max_principals_l1060_106002


namespace min_socks_for_twelve_pairs_l1060_106010

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (purple : ℕ)

/-- Calculates the minimum number of socks needed to guarantee a certain number of pairs -/
def minSocksForPairs (drawer : SockDrawer) (pairs : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 27 socks are needed to guarantee 12 pairs in the given drawer -/
theorem min_socks_for_twelve_pairs :
  let drawer : SockDrawer := { red := 90, green := 70, blue := 50, purple := 30 }
  minSocksForPairs drawer 12 = 27 := by sorry

end min_socks_for_twelve_pairs_l1060_106010


namespace hotel_room_charges_l1060_106043

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * (1 - 0.4))
  (h2 : P = G * (1 - 0.1)) :
  R = G * 1.5 := by
  sorry

end hotel_room_charges_l1060_106043


namespace hotel_flat_fee_calculation_l1060_106064

/-- A hotel charging system with a flat fee for the first night and a separate rate for additional nights. -/
structure HotelCharges where
  flatFee : ℝ  -- Flat fee for the first night
  nightlyRate : ℝ  -- Rate for each additional night

/-- Calculate the total cost for a given number of nights -/
def totalCost (h : HotelCharges) (nights : ℕ) : ℝ :=
  h.flatFee + h.nightlyRate * (nights - 1)

/-- Theorem stating the flat fee for the first night given the conditions -/
theorem hotel_flat_fee_calculation (h : HotelCharges) :
  totalCost h 2 = 120 ∧ totalCost h 5 = 255 → h.flatFee = 75 := by
  sorry

#check hotel_flat_fee_calculation

end hotel_flat_fee_calculation_l1060_106064


namespace fraction_equality_l1060_106016

theorem fraction_equality (a b : ℝ) (h : (1 / a) + (1 / (2 * b)) = 3) :
  (2 * a - 5 * a * b + 4 * b) / (4 * a * b - 3 * a - 6 * b) = -1/2 := by
  sorry

end fraction_equality_l1060_106016


namespace recycling_theorem_l1060_106098

def recycle (n : ℕ) : ℕ :=
  if n < 5 then 0 else n / 5 + recycle (n / 5)

theorem recycling_theorem :
  recycle 3125 = 781 :=
by
  sorry

end recycling_theorem_l1060_106098


namespace number_of_valid_divisors_l1060_106012

def total_marbles : ℕ := 720

theorem number_of_valid_divisors :
  (Finset.filter (fun m => m > 1 ∧ m < total_marbles ∧ total_marbles % m = 0) 
    (Finset.range (total_marbles + 1))).card = 28 := by
  sorry

end number_of_valid_divisors_l1060_106012


namespace extreme_value_implies_fourth_quadrant_l1060_106084

/-- A function f(x) = x^3 - ax^2 - bx has an extreme value of 10 at x = 1. -/
def has_extreme_value (a b : ℝ) : Prop :=
  let f := fun x : ℝ => x^3 - a*x^2 - b*x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  f 1 = 10

/-- The point (a, b) lies in the fourth quadrant. -/
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a < 0 ∧ b > 0

/-- Theorem: If f(x) = x^3 - ax^2 - bx has an extreme value of 10 at x = 1,
    then the point (a, b) lies in the fourth quadrant. -/
theorem extreme_value_implies_fourth_quadrant (a b : ℝ) :
  has_extreme_value a b → in_fourth_quadrant a b :=
by sorry

end extreme_value_implies_fourth_quadrant_l1060_106084


namespace function_increasing_iff_m_in_range_l1060_106001

/-- The function f(x) = (1/3)x³ - mx² - 3m²x + 1 is increasing on (1, 2) if and only if m is in [-1, 1/3] -/
theorem function_increasing_iff_m_in_range (m : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, StrictMono (fun x => (1/3) * x^3 - m * x^2 - 3 * m^2 * x + 1)) ↔
  m ∈ Set.Icc (-1) (1/3) :=
sorry

end function_increasing_iff_m_in_range_l1060_106001


namespace largest_solution_of_equation_l1060_106070

theorem largest_solution_of_equation : 
  let f : ℝ → ℝ := λ b => (3*b + 7)*(b - 2) - 9*b
  let largest_solution : ℝ := (4 + Real.sqrt 58) / 3
  (f largest_solution = 0) ∧ 
  (∀ b : ℝ, f b = 0 → b ≤ largest_solution) := by
  sorry

end largest_solution_of_equation_l1060_106070


namespace centroid_curve_area_centroid_curve_area_for_diameter_30_l1060_106034

/-- The area of the region bounded by the curve traced by the centroid of a triangle,
    where two vertices of the triangle are the endpoints of a circle's diameter,
    and the third vertex moves along the circle's circumference. -/
theorem centroid_curve_area (diameter : ℝ) : ℝ :=
  let radius := diameter / 2
  let centroid_radius := radius / 3
  let area := Real.pi * centroid_radius ^ 2
  ⌊area + 0.5⌋

/-- The area of the region bounded by the curve traced by the centroid of triangle ABC,
    where AB is a diameter of a circle with length 30 and C is a point on the circle,
    is approximately 79 (to the nearest positive integer). -/
theorem centroid_curve_area_for_diameter_30 :
  centroid_curve_area 30 = 79 := by
  sorry

end centroid_curve_area_centroid_curve_area_for_diameter_30_l1060_106034


namespace container_volume_ratio_l1060_106092

theorem container_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/5 : ℝ) * V₁ = (2/3 : ℝ) * V₂ →
  V₁ / V₂ = 10/9 := by
sorry

end container_volume_ratio_l1060_106092


namespace sqrt_one_sixty_four_l1060_106090

theorem sqrt_one_sixty_four : Real.sqrt (1 / 64) = 1 / 8 := by sorry

end sqrt_one_sixty_four_l1060_106090


namespace increasing_f_implies_a_range_l1060_106088

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem increasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  (3/2 < a ∧ a < 3) :=
sorry

end increasing_f_implies_a_range_l1060_106088


namespace complex_squared_plus_2i_l1060_106075

theorem complex_squared_plus_2i (i : ℂ) : i^2 = -1 → (1 + i)^2 + 2*i = 4*i := by
  sorry

end complex_squared_plus_2i_l1060_106075


namespace certain_number_proof_l1060_106023

theorem certain_number_proof (m : ℕ) : 9999 * m = 724827405 → m = 72483 := by
  sorry

end certain_number_proof_l1060_106023


namespace smallest_c_for_inverse_l1060_106025

def f (x : ℝ) := (x - 3)^2 - 4

theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → f x = f y → x = y) ↔ c ≥ 3 :=
sorry

end smallest_c_for_inverse_l1060_106025


namespace arithmetic_progression_of_primes_l1060_106026

theorem arithmetic_progression_of_primes (p q r d : ℕ) : 
  Prime p → Prime q → Prime r → 
  p > 3 → q > 3 → r > 3 →
  q = p + d → r = p + 2*d → 
  6 ∣ d :=
by sorry

end arithmetic_progression_of_primes_l1060_106026


namespace unique_six_digit_square_split_l1060_106015

def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2

def contains_no_zero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d ≠ 10

theorem unique_six_digit_square_split :
  ∃! n : ℕ,
    100000 ≤ n ∧ n ≤ 999999 ∧
    (∃ k : ℕ, n = k^2) ∧
    (∃ a b : ℕ, n = a * 1000 + b ∧
      is_three_digit_square a ∧
      is_three_digit_square b ∧
      contains_no_zero a ∧
      contains_no_zero b) :=
sorry

end unique_six_digit_square_split_l1060_106015


namespace tan_22_5_deg_sum_l1060_106056

theorem tan_22_5_deg_sum (a b c d : ℕ+) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - (b : ℝ).sqrt + (c : ℝ).sqrt - (d : ℝ)) :
  a + b + c + d = 3 := by
sorry

end tan_22_5_deg_sum_l1060_106056


namespace sequences_properties_l1060_106048

def sequence1 (n : ℕ) : ℤ := (-3)^n
def sequence2 (n : ℕ) : ℤ := -2 * (-3)^n
def sequence3 (n : ℕ) : ℤ := (-3)^n + 2

theorem sequences_properties :
  (∃ k : ℕ, sequence2 k + sequence2 (k+1) + sequence2 (k+2) = 378) ∧
  (sequence1 2024 + sequence2 2024 + sequence3 2024 = 2) := by
  sorry

end sequences_properties_l1060_106048


namespace origin_and_slope_condition_vertical_tangent_condition_l1060_106040

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + (1 - a)*x^2 - a*(a + 2)*x + b

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3*x^2 + 2*(1 - a)*x - a*(a + 2)

-- Theorem 1: If f(0) = 0 and f'(0) = -3, then (a = -3 or a = 1) and b = 0
theorem origin_and_slope_condition (a b : ℝ) :
  f a b 0 = 0 ∧ f' a 0 = -3 → (a = -3 ∨ a = 1) ∧ b = 0 := by sorry

-- Theorem 2: The curve y = f(x) has two vertical tangent lines iff a ∈ (-∞, -1/2) ∪ (-1/2, +∞)
theorem vertical_tangent_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0) ↔ 
  a < -1/2 ∨ a > -1/2 := by sorry

end origin_and_slope_condition_vertical_tangent_condition_l1060_106040


namespace solutions_x_fourth_plus_81_l1060_106022

theorem solutions_x_fourth_plus_81 :
  {x : ℂ | x^4 + 81 = 0} = {3 + 3*I, -3 - 3*I, -3 + 3*I, 3 - 3*I} := by
  sorry

end solutions_x_fourth_plus_81_l1060_106022


namespace set_equality_implies_sum_l1060_106020

/-- Given that the set {1, a, b/a} equals the set {0, a², a+b}, prove that a²⁰¹³ + b²⁰¹² = -1 -/
theorem set_equality_implies_sum (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2013 + b^2012 = -1 := by
  sorry

end set_equality_implies_sum_l1060_106020


namespace largest_consecutive_sum_120_l1060_106030

/-- Given a sequence of consecutive natural numbers with sum 120, 
    the largest number in the sequence is 26 -/
theorem largest_consecutive_sum_120 (n : ℕ) (a : ℕ) (h1 : n > 1) 
  (h2 : (n : ℝ) * (2 * a + n - 1) / 2 = 120) :
  a + n - 1 ≤ 26 := by
  sorry

end largest_consecutive_sum_120_l1060_106030


namespace math_basketball_count_l1060_106007

/-- Represents the number of students in a school with various club and team memberships -/
structure SchoolMembership where
  total : ℕ
  science_club : ℕ
  math_club : ℕ
  football_team : ℕ
  basketball_team : ℕ
  science_football : ℕ

/-- Conditions for the school membership problem -/
def school_conditions (s : SchoolMembership) : Prop :=
  s.total = 60 ∧
  s.science_club + s.math_club = s.total ∧
  s.football_team + s.basketball_team = s.total ∧
  s.science_football = 20 ∧
  s.math_club = 36 ∧
  s.basketball_team = 22

/-- Theorem stating the number of students in both math club and basketball team -/
theorem math_basketball_count (s : SchoolMembership) 
  (h : school_conditions s) : 
  s.math_club + s.basketball_team - s.total = 18 := by
  sorry

#check math_basketball_count

end math_basketball_count_l1060_106007


namespace complex_number_with_prime_modulus_exists_l1060_106076

theorem complex_number_with_prime_modulus_exists : ∃ (z : ℂ), 
  z^2 = (3 + Complex.I) * z - 24 + 15 * Complex.I ∧ 
  ∃ (p : ℕ), Nat.Prime p ∧ (z.re^2 + z.im^2 : ℝ) = p :=
sorry

end complex_number_with_prime_modulus_exists_l1060_106076


namespace parabola_translation_l1060_106009

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (p : Parabola) :
  p.a = -2 ∧ p.b = -4 ∧ p.c = -6 →
  translate p 1 3 = Parabola.mk (-2) 0 (-1) := by
  sorry

end parabola_translation_l1060_106009


namespace twelve_chairs_subsets_l1060_106024

/-- The number of chairs in the circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets with at least four adjacent chairs -/
def subsetsWithAdjacentChairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs, the number of subsets with at least four adjacent chairs is 1776 -/
theorem twelve_chairs_subsets : subsetsWithAdjacentChairs n = 1776 := by sorry

end twelve_chairs_subsets_l1060_106024


namespace parallel_transitivity_l1060_106089

-- Define a type for lines in a plane
structure Line where
  -- You can add more specific properties here if needed
  mk :: 

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  -- The definition of parallel lines
  sorry

-- State the theorem
theorem parallel_transitivity (l1 l2 l3 : Line) : 
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 := by
  sorry

end parallel_transitivity_l1060_106089


namespace tony_water_consumption_l1060_106074

/-- 
Given that Tony drank 48 ounces of water yesterday, which is 4% less than 
what he drank two days ago, prove that he drank 50 ounces of water two days ago.
-/
theorem tony_water_consumption (yesterday : ℝ) (two_days_ago : ℝ) 
  (h1 : yesterday = 48)
  (h2 : yesterday = two_days_ago * (1 - 0.04)) : 
  two_days_ago = 50 := by
  sorry

end tony_water_consumption_l1060_106074


namespace bill_difference_l1060_106073

theorem bill_difference (anna_tip bob_tip cindy_tip : ℝ)
  (anna_percent bob_percent cindy_percent : ℝ)
  (h_anna : anna_tip = 3 ∧ anna_percent = 0.15)
  (h_bob : bob_tip = 4 ∧ bob_percent = 0.10)
  (h_cindy : cindy_tip = 5 ∧ cindy_percent = 0.25)
  (h_anna_bill : anna_tip = anna_percent * (anna_tip / anna_percent))
  (h_bob_bill : bob_tip = bob_percent * (bob_tip / bob_percent))
  (h_cindy_bill : cindy_tip = cindy_percent * (cindy_tip / cindy_percent)) :
  max (anna_tip / anna_percent) (max (bob_tip / bob_percent) (cindy_tip / cindy_percent)) -
  min (anna_tip / anna_percent) (min (bob_tip / bob_percent) (cindy_tip / cindy_percent)) = 20 :=
by sorry

end bill_difference_l1060_106073


namespace sum_exradii_equals_four_circumradius_plus_inradius_l1060_106078

/-- Given a triangle with exradii r_a, r_b, r_c, circumradius R, and inradius r,
    prove that the sum of the exradii equals four times the circumradius plus the inradius. -/
theorem sum_exradii_equals_four_circumradius_plus_inradius 
  (r_a r_b r_c R r : ℝ) :
  r_a > 0 → r_b > 0 → r_c > 0 → R > 0 → r > 0 →
  r_a + r_b + r_c = 4 * R + r := by
  sorry

end sum_exradii_equals_four_circumradius_plus_inradius_l1060_106078


namespace basketball_club_girls_l1060_106080

theorem basketball_club_girls (total_members : ℕ) (attendance : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_members = 30 →
  attendance = 18 →
  boys + girls = total_members →
  boys + (1/3 : ℚ) * girls = attendance →
  girls = 18 :=
by sorry

end basketball_club_girls_l1060_106080


namespace f_properties_l1060_106093

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2/x - a*x + 5|

theorem f_properties :
  ∀ a : ℝ,
  (∃ x : ℝ, f a x = 0) ∧
  (a = 3 → ∀ x y : ℝ, x < y → y < -1 → f a x > f a y) ∧
  (a > 0 → ∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 2 ∧ f a x₀ = 8/3 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≤ 8/3) :=
by sorry

end f_properties_l1060_106093


namespace cube_side_length_l1060_106071

theorem cube_side_length (surface_area : ℝ) (side_length : ℝ) : 
  surface_area = 600 → 
  6 * side_length^2 = surface_area → 
  side_length = 10 := by
sorry

end cube_side_length_l1060_106071


namespace square_difference_divided_by_ten_l1060_106038

theorem square_difference_divided_by_ten : (305^2 - 295^2) / 10 = 600 := by sorry

end square_difference_divided_by_ten_l1060_106038


namespace jills_bus_journey_ratio_l1060_106072

/-- Represents the time in minutes for various parts of Jill's bus journey -/
structure BusJourney where
  first_bus_wait : ℕ
  first_bus_ride : ℕ
  second_bus_ride : ℕ

/-- Calculates the ratio of the second bus ride time to the combined wait and trip time of the first bus -/
def bus_time_ratio (journey : BusJourney) : ℚ :=
  journey.second_bus_ride / (journey.first_bus_wait + journey.first_bus_ride)

/-- Theorem stating that for Jill's specific journey, the bus time ratio is 1/2 -/
theorem jills_bus_journey_ratio :
  let journey : BusJourney := { first_bus_wait := 12, first_bus_ride := 30, second_bus_ride := 21 }
  bus_time_ratio journey = 1/2 := by
  sorry


end jills_bus_journey_ratio_l1060_106072


namespace missing_number_proof_l1060_106039

theorem missing_number_proof (x : ℝ) : 11 + Real.sqrt (-4 + x * 4 / 3) = 13 ↔ x = 6 := by
  sorry

end missing_number_proof_l1060_106039


namespace square_area_from_vertices_l1060_106062

/-- The area of a square with adjacent vertices at (0,3) and (4,0) is 25. -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 3)
  let p2 : ℝ × ℝ := (4, 0)
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  d^2 = 25 := by sorry

end square_area_from_vertices_l1060_106062


namespace imaginary_part_of_complex_fraction_l1060_106046

theorem imaginary_part_of_complex_fraction : Complex.im (5 * Complex.I / (1 + 2 * Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l1060_106046


namespace mass_of_man_is_180_l1060_106053

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating that the mass of the man is 180 kg under the given conditions. -/
theorem mass_of_man_is_180 :
  let boat_length : ℝ := 6
  let boat_breadth : ℝ := 3
  let boat_sink_height : ℝ := 0.01
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 180 := by
  sorry

#eval mass_of_man 6 3 0.01 1000

end mass_of_man_is_180_l1060_106053


namespace vector_operation_l1060_106013

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)

theorem vector_operation : 
  (3 • a - 2 • b : ℝ × ℝ) = (1, 5) := by sorry

end vector_operation_l1060_106013


namespace parallelogram_altitude_base_ratio_l1060_106055

/-- Given a parallelogram with area 128 sq m and base 8 m, prove the ratio of altitude to base is 2 -/
theorem parallelogram_altitude_base_ratio :
  ∀ (area base altitude : ℝ),
  area = 128 ∧ base = 8 ∧ area = base * altitude →
  altitude / base = 2 := by
sorry

end parallelogram_altitude_base_ratio_l1060_106055


namespace chemistry_marks_proof_l1060_106066

def english_marks : ℕ := 91
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def biology_marks : ℕ := 85
def average_marks : ℕ := 78
def total_subjects : ℕ := 5

theorem chemistry_marks_proof :
  ∃ (chemistry_marks : ℕ),
    (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks) / total_subjects = average_marks ∧
    chemistry_marks = 67 := by
  sorry

end chemistry_marks_proof_l1060_106066


namespace bananas_cantaloupe_cost_l1060_106050

/-- Represents the cost of groceries -/
structure GroceryCost where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ

/-- The total cost of all items is $40 -/
def total_cost (g : GroceryCost) : Prop :=
  g.apples + g.bananas + g.cantaloupe + g.dates = 40

/-- A carton of dates costs three times as much as a sack of apples -/
def dates_cost (g : GroceryCost) : Prop :=
  g.dates = 3 * g.apples

/-- The price of a cantaloupe is equal to half the sum of the price of a sack of apples and a bunch of bananas -/
def cantaloupe_cost (g : GroceryCost) : Prop :=
  g.cantaloupe = (g.apples + g.bananas) / 2

/-- The main theorem: Given the conditions, the cost of a bunch of bananas and a cantaloupe is $8 -/
theorem bananas_cantaloupe_cost (g : GroceryCost) 
  (h1 : total_cost g) 
  (h2 : dates_cost g) 
  (h3 : cantaloupe_cost g) : 
  g.bananas + g.cantaloupe = 8 := by
  sorry

end bananas_cantaloupe_cost_l1060_106050


namespace samuel_has_five_birds_l1060_106067

/-- The number of berries a single bird eats per day -/
def berries_per_bird_per_day : ℕ := 7

/-- The total number of berries eaten by all birds in 4 days -/
def total_berries_in_four_days : ℕ := 140

/-- The number of days over which the total berries are consumed -/
def days : ℕ := 4

/-- The number of birds Samuel has -/
def samuels_birds : ℕ := total_berries_in_four_days / (days * berries_per_bird_per_day)

theorem samuel_has_five_birds : samuels_birds = 5 := by
  sorry

end samuel_has_five_birds_l1060_106067


namespace seventh_term_of_arithmetic_sequence_l1060_106065

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem seventh_term_of_arithmetic_sequence 
  (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : sum_of_arithmetic_sequence a 13 = 39) : 
  a 7 = 3 := by
sorry

end seventh_term_of_arithmetic_sequence_l1060_106065


namespace rogers_remaining_years_l1060_106060

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions of the coworkers' experience -/
def valid_experience (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.peter = 12 ∧
  e.tom = 2 * e.robert ∧
  e.robert = e.peter - 4 ∧
  e.robert = e.mike + 2

/-- Roger's retirement years -/
def retirement_years : ℕ := 50

/-- Theorem stating that Roger needs to work 8 more years before retirement -/
theorem rogers_remaining_years (e : Experience) (h : valid_experience e) :
  retirement_years - e.roger = 8 := by
  sorry


end rogers_remaining_years_l1060_106060


namespace triangle_formation_l1060_106047

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  ¬(can_form_triangle 2 3 5) ∧
  can_form_triangle 5 6 10 ∧
  ¬(can_form_triangle 1 1 3) ∧
  ¬(can_form_triangle 3 4 9) :=
sorry

end triangle_formation_l1060_106047


namespace fixed_point_on_line_fixed_point_unique_l1060_106069

/-- The line equation passing through a fixed point -/
def line_equation (k x y : ℝ) : Prop :=
  y = k * (x - 2) + 3

/-- The fixed point through which the line always passes -/
def fixed_point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the fixed point satisfies the line equation for all k -/
theorem fixed_point_on_line :
  ∀ k : ℝ, line_equation k (fixed_point.1) (fixed_point.2) :=
sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_unique :
  ∀ x y : ℝ, (∀ k : ℝ, line_equation k x y) → (x, y) = fixed_point :=
sorry

end fixed_point_on_line_fixed_point_unique_l1060_106069


namespace sqrt_inequality_equivalence_l1060_106021

theorem sqrt_inequality_equivalence :
  (Real.sqrt 2 - Real.sqrt 3 < Real.sqrt 6 - Real.sqrt 7) ↔
  ((Real.sqrt 2 + Real.sqrt 7)^2 < (Real.sqrt 6 + Real.sqrt 3)^2) := by
  sorry

end sqrt_inequality_equivalence_l1060_106021


namespace gcd_of_45_and_75_l1060_106011

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_of_45_and_75_l1060_106011


namespace patio_rearrangement_l1060_106032

/-- Represents a rectangular patio layout --/
structure PatioLayout where
  rows : ℕ
  columns : ℕ
  total_tiles : ℕ

/-- Defines the conditions for a valid patio layout --/
def is_valid_layout (layout : PatioLayout) : Prop :=
  layout.total_tiles = layout.rows * layout.columns

/-- Defines the rearrangement of the patio --/
def rearranged_layout (original : PatioLayout) : PatioLayout :=
  { rows := original.total_tiles / (original.columns - 2)
  , columns := original.columns - 2
  , total_tiles := original.total_tiles }

/-- The main theorem to prove --/
theorem patio_rearrangement 
  (original : PatioLayout)
  (h_valid : is_valid_layout original)
  (h_rows : original.rows = 6)
  (h_total : original.total_tiles = 48) :
  (rearranged_layout original).rows - original.rows = 2 :=
sorry

end patio_rearrangement_l1060_106032


namespace complex_modulus_constraint_l1060_106036

theorem complex_modulus_constraint (a : ℝ) :
  (∀ θ : ℝ, Complex.abs ((a - Real.cos θ) + (a - 1 - Real.sin θ) * Complex.I) ≤ 2) →
  0 ≤ a ∧ a ≤ 1 := by
  sorry

end complex_modulus_constraint_l1060_106036


namespace adams_dog_food_packages_l1060_106014

theorem adams_dog_food_packages (cat_packages : ℕ) (cat_cans_per_package : ℕ) (dog_cans_per_package : ℕ) (cat_dog_can_difference : ℕ) :
  cat_packages = 9 →
  cat_cans_per_package = 10 →
  dog_cans_per_package = 5 →
  cat_dog_can_difference = 55 →
  ∃ (dog_packages : ℕ),
    cat_packages * cat_cans_per_package = dog_packages * dog_cans_per_package + cat_dog_can_difference ∧
    dog_packages = 7 :=
by
  sorry


end adams_dog_food_packages_l1060_106014


namespace divisibility_implies_equality_l1060_106082

theorem divisibility_implies_equality (a b : ℕ+) 
  (h : ∀ n : ℕ+, (a.val^n.val + n.val) ∣ (b.val^n.val + n.val)) : a = b := by
  sorry

end divisibility_implies_equality_l1060_106082


namespace animal_ratio_proof_l1060_106003

/-- Given ratios between animals, prove the final ratio of all animals -/
theorem animal_ratio_proof 
  (chicken_pig_ratio : ℚ × ℚ)
  (sheep_horse_ratio : ℚ × ℚ)
  (pig_horse_ratio : ℚ × ℚ)
  (h1 : chicken_pig_ratio = (26, 5))
  (h2 : sheep_horse_ratio = (25, 9))
  (h3 : pig_horse_ratio = (10, 3)) :
  ∃ (k : ℚ), k > 0 ∧ 
    k * 156 = chicken_pig_ratio.1 * pig_horse_ratio.2 ∧
    k * 30 = chicken_pig_ratio.2 * pig_horse_ratio.2 ∧
    k * 9 = pig_horse_ratio.2 ∧
    k * 25 = sheep_horse_ratio.1 * pig_horse_ratio.2 / sheep_horse_ratio.2 :=
by
  sorry

end animal_ratio_proof_l1060_106003


namespace sum_of_solutions_is_six_l1060_106027

theorem sum_of_solutions_is_six : 
  ∃ (x₁ x₂ : ℂ), 
    (2 : ℂ) ^ (x₁^2 - 3*x₁ - 2) = (8 : ℂ) ^ (x₁ - 5) ∧
    (2 : ℂ) ^ (x₂^2 - 3*x₂ - 2) = (8 : ℂ) ^ (x₂ - 5) ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 6 ∧
    ∀ (y : ℂ), (2 : ℂ) ^ (y^2 - 3*y - 2) = (8 : ℂ) ^ (y - 5) → y = x₁ ∨ y = x₂ :=
by sorry

end sum_of_solutions_is_six_l1060_106027


namespace complex_magnitude_problem_l1060_106008

theorem complex_magnitude_problem (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = (1 + 2*i) / i → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l1060_106008


namespace find_number_l1060_106063

theorem find_number (A B : ℕ) (hA : A > 0) (hB : B > 0) : 
  Nat.gcd A B = 15 → Nat.lcm A B = 312 → B = 195 → A = 24 := by
  sorry

end find_number_l1060_106063


namespace reading_speed_ratio_l1060_106005

/-- Given that Emery takes 20 days to read a book and the average number of days
    for Emery and Serena to read the book is 60, prove that the ratio of
    Emery's reading speed to Serena's reading speed is 5:1 -/
theorem reading_speed_ratio
  (emery_days : ℕ)
  (average_days : ℚ)
  (h_emery : emery_days = 20)
  (h_average : average_days = 60) :
  ∃ (emery_speed serena_speed : ℚ), 
    emery_speed / serena_speed = 5 / 1 :=
by sorry

end reading_speed_ratio_l1060_106005


namespace no_geometric_mean_opposite_signs_l1060_106099

/-- The geometric mean of two real numbers does not exist if they have opposite signs -/
theorem no_geometric_mean_opposite_signs (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  ¬∃ (x : ℝ), x^2 = a * b :=
by sorry

end no_geometric_mean_opposite_signs_l1060_106099


namespace product_of_numbers_l1060_106081

theorem product_of_numbers (x y : ℝ) 
  (h1 : (x + y) / (x - y) = 7)
  (h2 : (x * y) / (x - y) = 24) : 
  x * y = 48 := by
sorry

end product_of_numbers_l1060_106081
