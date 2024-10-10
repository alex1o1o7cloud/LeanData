import Mathlib

namespace triangle_inequalities_l141_14109

/-- Theorem about triangle inequalities involving area, side lengths, altitudes, and excircle radii -/
theorem triangle_inequalities (a b c : ℝ) (S : ℝ) (h_a h_b h_c : ℝ) (r_a r_b r_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_area : S > 0)
  (h_altitudes : h_a > 0 ∧ h_b > 0 ∧ c > 0)
  (h_radii : r_a > 0 ∧ r_b > 0 ∧ r_c > 0) :
  S^3 ≤ (Real.sqrt 3 / 4)^3 * (a * b * c)^2 ∧ 
  3 * h_a * h_b * h_c ≤ 3 * Real.sqrt 3 * S ∧
  3 * Real.sqrt 3 * S ≤ 3 * r_a * r_b * r_c := by
  sorry


end triangle_inequalities_l141_14109


namespace square_sum_equals_25_l141_14174

theorem square_sum_equals_25 (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) :
  x^2 + y^2 = 25 := by
  sorry

end square_sum_equals_25_l141_14174


namespace triangle_inequality_cube_l141_14187

theorem triangle_inequality_cube (a b c : ℝ) 
  (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^3 + b^3 + 3*a*b*c > c^3 := by
  sorry

end triangle_inequality_cube_l141_14187


namespace quadratic_minimum_l141_14135

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end quadratic_minimum_l141_14135


namespace tunnel_length_proof_l141_14122

/-- The length of a train in miles -/
def train_length : ℝ := 1.5

/-- The time difference in minutes between the front of the train entering the tunnel and the tail exiting -/
def time_difference : ℝ := 4

/-- The speed of the train in miles per hour -/
def train_speed : ℝ := 45

/-- The length of the tunnel in miles -/
def tunnel_length : ℝ := 1.5

theorem tunnel_length_proof :
  tunnel_length = train_speed * (time_difference / 60) - train_length :=
by sorry

end tunnel_length_proof_l141_14122


namespace batsman_average_theorem_l141_14183

/-- Represents a batsman's score history -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningScore : Nat

/-- Calculates the new average after the latest inning -/
def newAverage (b : Batsman) : Rat :=
  (b.totalRuns + b.lastInningScore) / (b.innings + 1)

/-- Theorem: A batsman who scores 100 runs in his 17th inning and increases his average by 5 runs will have a new average of 20 runs -/
theorem batsman_average_theorem (b : Batsman) 
  (h1 : b.innings = 16)
  (h2 : b.lastInningScore = 100)
  (h3 : b.averageIncrease = 5)
  (h4 : newAverage b = (b.totalRuns + b.lastInningScore) / (b.innings + 1)) :
  newAverage b = 20 := by
  sorry

#check batsman_average_theorem

end batsman_average_theorem_l141_14183


namespace hundreds_digit_of_factorial_difference_l141_14100

theorem hundreds_digit_of_factorial_difference : (25 - 20).factorial ≡ 0 [ZMOD 1000] := by
  sorry

end hundreds_digit_of_factorial_difference_l141_14100


namespace city_population_increase_l141_14104

/-- Represents the net population increase in a city over one day -/
def net_population_increase (birth_rate : ℕ) (death_rate : ℕ) (time_interval : ℕ) (seconds_per_day : ℕ) : ℕ :=
  let net_rate_per_interval := birth_rate - death_rate
  let net_rate_per_second := net_rate_per_interval / time_interval
  net_rate_per_second * seconds_per_day

/-- Theorem stating the net population increase in a day given specific birth and death rates -/
theorem city_population_increase : 
  net_population_increase 6 2 2 86400 = 172800 := by
  sorry

#eval net_population_increase 6 2 2 86400

end city_population_increase_l141_14104


namespace equation_implies_equilateral_l141_14179

/-- A triangle with sides a, b, c and opposite angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- The radius of the circumscribed circle of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- The equation that the triangle satisfies -/
def satisfies_equation (t : Triangle) : Prop :=
  (t.a * Real.cos t.α + t.b * Real.cos t.β + t.c * Real.cos t.γ) /
  (t.a * Real.sin t.β + t.b * Real.sin t.γ + t.c * Real.sin t.α) =
  perimeter t / (9 * circumradius t)

/-- A triangle is equilateral if all its sides are equal -/
def is_equilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

/-- The main theorem -/
theorem equation_implies_equilateral (t : Triangle) :
  satisfies_equation t → is_equilateral t :=
by sorry

end equation_implies_equilateral_l141_14179


namespace cut_rectangle_corners_l141_14133

/-- A shape created by cutting off one corner of a rectangle --/
structure CutRectangle where
  originalCorners : Nat
  cutCorners : Nat
  newCorners : Nat

/-- Properties of a rectangle with one corner cut off --/
def isValidCutRectangle (r : CutRectangle) : Prop :=
  r.originalCorners = 4 ∧
  r.cutCorners = 1 ∧
  r.newCorners = r.originalCorners + r.cutCorners

/-- Theorem: A rectangle with one corner cut off has 5 corners --/
theorem cut_rectangle_corners (r : CutRectangle) (h : isValidCutRectangle r) :
  r.newCorners = 5 := by
  sorry

#check cut_rectangle_corners

end cut_rectangle_corners_l141_14133


namespace floor_painting_theorem_l141_14168

/-- The number of ordered pairs (a,b) satisfying the floor painting conditions -/
def floor_painting_solutions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    b > a ∧ (a - 4) * (b - 4) = 2 * a * b / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 3 solutions to the floor painting problem -/
theorem floor_painting_theorem : floor_painting_solutions = 3 := by
  sorry

end floor_painting_theorem_l141_14168


namespace blue_candles_l141_14134

/-- The number of blue candles on a birthday cake -/
theorem blue_candles (total : ℕ) (yellow : ℕ) (red : ℕ) (blue : ℕ)
  (h1 : total = 79)
  (h2 : yellow = 27)
  (h3 : red = 14)
  (h4 : blue = total - yellow - red) :
  blue = 38 := by
  sorry

end blue_candles_l141_14134


namespace percentage_relation_l141_14180

theorem percentage_relation (j p t m n : ℕ+) (r : ℚ) : 
  (j : ℚ) = 0.75 * p ∧
  (j : ℚ) = 0.80 * t ∧
  (t : ℚ) = p - (r / 100) * p ∧
  (m : ℚ) = 1.10 * p ∧
  (n : ℚ) = 0.70 * m ∧
  (j : ℚ) + p + t = m * n →
  r = 6.25 := by
sorry

end percentage_relation_l141_14180


namespace simple_interest_time_calculation_l141_14106

/-- Simple interest calculation for a given principal, rate, and interest amount -/
theorem simple_interest_time_calculation 
  (P : ℝ) (R : ℝ) (SI : ℝ) (h1 : P = 10000) (h2 : R = 5) (h3 : SI = 500) : 
  (SI * 100) / (P * R) * 12 = 12 := by
sorry

end simple_interest_time_calculation_l141_14106


namespace maria_trip_portion_l141_14198

theorem maria_trip_portion (total_distance : ℝ) (first_stop_fraction : ℝ) (remaining_distance : ℝ)
  (h1 : total_distance = 560)
  (h2 : first_stop_fraction = 1 / 2)
  (h3 : remaining_distance = 210) :
  (total_distance * (1 - first_stop_fraction) - remaining_distance) / (total_distance * (1 - first_stop_fraction)) = 1 / 4 := by
  sorry

end maria_trip_portion_l141_14198


namespace sufficient_not_necessary_l141_14139

/-- A hyperbola is represented by an equation of the form a*x² + b*y² = c, where a and b have opposite signs and c ≠ 0 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0
  h3 : c ≠ 0
  h4 : a * b < 0

/-- The equation x²/(9-k) + y²/(k-4) = 1 -/
def equation (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (9 - k) + y^2 / (k - 4) = 1

/-- The condition k > 9 is sufficient but not necessary for the equation to represent a hyperbola -/
theorem sufficient_not_necessary (k : ℝ) :
  (k > 9 → ∃ h : Hyperbola, equation k = λ x y ↦ h.a * x^2 + h.b * y^2 = h.c) ∧
  ¬(∀ h : Hyperbola, equation k = λ x y ↦ h.a * x^2 + h.b * y^2 = h.c → k > 9) :=
by sorry

end sufficient_not_necessary_l141_14139


namespace ice_cream_cost_l141_14101

theorem ice_cream_cost (pierre_scoops mom_scoops : ℕ) (total_bill : ℚ) 
  (h1 : pierre_scoops = 3)
  (h2 : mom_scoops = 4)
  (h3 : total_bill = 14) :
  ∃ (scoop_cost : ℚ), scoop_cost * (pierre_scoops + mom_scoops : ℚ) = total_bill ∧ scoop_cost = 2 := by
  sorry

end ice_cream_cost_l141_14101


namespace betty_oranges_l141_14190

theorem betty_oranges (emily sandra betty : ℕ) 
  (h1 : emily = 7 * sandra) 
  (h2 : sandra = 3 * betty) 
  (h3 : emily = 252) : 
  betty = 12 := by
  sorry

end betty_oranges_l141_14190


namespace rotation_direction_undetermined_l141_14110

-- Define a type for rotation direction
inductive RotationDirection
| Clockwise
| Counterclockwise

-- Define a type for a quadrilateral
structure Quadrilateral where
  -- We don't need to specify the exact structure of a quadrilateral for this problem
  mk :: 

-- Define a point Z
def Z : ℝ × ℝ := sorry

-- Define the rotation transformation
def rotate (q : Quadrilateral) (center : ℝ × ℝ) (angle : ℝ) : Quadrilateral := sorry

-- State the theorem
theorem rotation_direction_undetermined 
  (q1 q2 : Quadrilateral) 
  (h1 : rotate q1 Z (270 : ℝ) = q2) : 
  ¬ ∃ (d : RotationDirection), d = RotationDirection.Clockwise ∨ d = RotationDirection.Counterclockwise := 
sorry

end rotation_direction_undetermined_l141_14110


namespace diamond_calculation_l141_14159

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_calculation : 
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 := by
  sorry

end diamond_calculation_l141_14159


namespace total_spider_legs_l141_14191

/-- The number of spiders in the room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs is 32 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 32 := by
  sorry

end total_spider_legs_l141_14191


namespace cylinder_volume_ratio_l141_14103

/-- The ratio of volumes of two cylinders formed from a 6x9 rectangle -/
theorem cylinder_volume_ratio : 
  let rect_width : ℝ := 6
  let rect_height : ℝ := 9
  let cylinder1_height : ℝ := rect_height
  let cylinder1_circumference : ℝ := rect_width
  let cylinder2_height : ℝ := rect_width
  let cylinder2_circumference : ℝ := rect_height
  let cylinder1_volume : ℝ := π * (cylinder1_circumference / (2 * π))^2 * cylinder1_height
  let cylinder2_volume : ℝ := π * (cylinder2_circumference / (2 * π))^2 * cylinder2_height
  let max_volume : ℝ := max cylinder1_volume cylinder2_volume
  let min_volume : ℝ := min cylinder1_volume cylinder2_volume
  (max_volume / min_volume) = 3/4 := by
sorry

end cylinder_volume_ratio_l141_14103


namespace largest_x_floor_div_l141_14136

theorem largest_x_floor_div (x : ℝ) : 
  (⌊x⌋ : ℝ) / x = 9 / 10 → x ≤ 80 / 9 := by
  sorry

end largest_x_floor_div_l141_14136


namespace sum_and_reciprocal_square_l141_14160

theorem sum_and_reciprocal_square (m : ℝ) (h : m + 1/m = 5) : 
  m^2 + 1/m^2 + m + 1/m = 28 := by
  sorry

end sum_and_reciprocal_square_l141_14160


namespace james_height_l141_14181

theorem james_height (tree_height : ℝ) (tree_shadow : ℝ) (james_shadow : ℝ) :
  tree_height = 60 →
  tree_shadow = 20 →
  james_shadow = 25 →
  (tree_height / tree_shadow) * james_shadow = 75 := by
  sorry

end james_height_l141_14181


namespace simplify_sqrt_expression_l141_14151

theorem simplify_sqrt_expression :
  Real.sqrt (75 - 30 * Real.sqrt 5) = 5 - 3 * Real.sqrt 5 := by
  sorry

end simplify_sqrt_expression_l141_14151


namespace concert_ticket_discount_l141_14177

theorem concert_ticket_discount (normal_price : ℝ) (scalper_markup : ℝ) (scalper_discount : ℝ) (total_paid : ℝ) :
  normal_price = 50 →
  scalper_markup = 2.4 →
  scalper_discount = 10 →
  total_paid = 360 →
  ∃ (discounted_price : ℝ),
    2 * normal_price + 2 * (scalper_markup * normal_price - scalper_discount / 2) + discounted_price = total_paid ∧
    discounted_price / normal_price = 0.6 := by
  sorry

end concert_ticket_discount_l141_14177


namespace fraction_problem_l141_14155

theorem fraction_problem (f : ℚ) : f = 1/3 → 0.75 * 264 = f * 264 + 110 := by
  sorry

end fraction_problem_l141_14155


namespace product_equality_l141_14127

theorem product_equality (a b c : ℝ) (h : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) :
  6 * 15 * 3 = 3 := by
  sorry

end product_equality_l141_14127


namespace consecutive_non_primes_l141_14128

theorem consecutive_non_primes (n : ℕ) : 
  ∃ k : ℕ, ∀ i : ℕ, i ∈ Finset.range n → ¬ Prime (k + i + 2) := by
  sorry

end consecutive_non_primes_l141_14128


namespace base4_10203_equals_291_l141_14152

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_10203_equals_291 :
  base4_to_decimal [3, 0, 2, 0, 1] = 291 := by
  sorry

end base4_10203_equals_291_l141_14152


namespace units_digit_of_m_squared_plus_three_to_m_l141_14193

def m : ℕ := 2021^3 + 3^2021

theorem units_digit_of_m_squared_plus_three_to_m (m : ℕ := 2021^3 + 3^2021) :
  (m^2 + 3^m) % 10 = 7 :=
by sorry

end units_digit_of_m_squared_plus_three_to_m_l141_14193


namespace student_arrangements_eq_20_l141_14138

/-- The number of ways to arrange 7 students of different heights in a row,
    with the tallest in the middle and the others decreasing in height towards both ends. -/
def student_arrangements : ℕ :=
  Nat.choose 6 3

/-- Theorem stating that the number of student arrangements is 20. -/
theorem student_arrangements_eq_20 : student_arrangements = 20 := by
  sorry

end student_arrangements_eq_20_l141_14138


namespace fraction_equality_l141_14166

theorem fraction_equality (x y : ℝ) (h : x ≠ y) : -x / (x - y) = x / (-x + y) := by
  sorry

end fraction_equality_l141_14166


namespace cubic_equation_solution_l141_14195

theorem cubic_equation_solution : ∃ x : ℝ, (x - 2)^3 + (x - 6)^3 = 54 ∧ x = 7 := by
  sorry

end cubic_equation_solution_l141_14195


namespace taco_castle_parking_lot_l141_14162

theorem taco_castle_parking_lot : 
  ∀ (volkswagen ford toyota dodge : ℕ),
    volkswagen = 5 →
    toyota = 2 * volkswagen →
    ford = 2 * toyota →
    3 * ford = dodge →
    dodge = 60 := by
  sorry

end taco_castle_parking_lot_l141_14162


namespace last_three_digits_proof_l141_14171

theorem last_three_digits_proof : 
  (301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 3 % 1000 = 976 := by
  sorry

end last_three_digits_proof_l141_14171


namespace max_sum_of_factors_l141_14112

theorem max_sum_of_factors (p q : ℕ+) (h : p * q = 100) : 
  ∃ (a b : ℕ+), a * b = 100 ∧ a + b ≤ p + q ∧ a + b = 101 := by
  sorry

end max_sum_of_factors_l141_14112


namespace f_range_contains_interval_f_range_may_extend_l141_14117

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos (x/3))^2 + 2*Real.pi * Real.arcsin (x/3) - 3*(Real.arcsin (x/3))^2 + 
  (Real.pi^2/4)*(x^3 - x^2 + 4*x - 8)

theorem f_range_contains_interval :
  ∀ y ∈ Set.Icc (Real.pi^2/4) ((9*Real.pi^2)/4),
  ∃ x ∈ Set.Icc (-3) 3, f x = y :=
by sorry

theorem f_range_may_extend :
  ∃ y, (y < Real.pi^2/4 ∨ y > (9*Real.pi^2)/4) ∧
  ∃ x ∈ Set.Icc (-3) 3, f x = y :=
by sorry

end f_range_contains_interval_f_range_may_extend_l141_14117


namespace machine_B_performs_better_l141_14167

def machineA : List ℕ := [0, 1, 0, 2, 2, 0, 3, 1, 2, 4]
def machineB : List ℕ := [2, 3, 1, 1, 0, 2, 1, 1, 0, 1]

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let avg := average l
  (l.map (fun x => ((x : ℚ) - avg) ^ 2)).sum / l.length

theorem machine_B_performs_better :
  average machineB < average machineA ∧
  variance machineB < variance machineA := by
  sorry

end machine_B_performs_better_l141_14167


namespace max_togs_value_l141_14197

def tag_price : ℕ := 3
def tig_price : ℕ := 4
def tog_price : ℕ := 8
def total_budget : ℕ := 100

def max_togs (x y z : ℕ) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧
  x * tag_price + y * tig_price + z * tog_price = total_budget ∧
  ∀ (a b c : ℕ), a ≥ 1 → b ≥ 1 → c ≥ 1 →
    a * tag_price + b * tig_price + c * tog_price = total_budget →
    c ≤ z

theorem max_togs_value : ∃ (x y : ℕ), max_togs x y 11 := by
  sorry

end max_togs_value_l141_14197


namespace quadratic_equation_solution_l141_14145

theorem quadratic_equation_solution (h : (63 * (5/7)^2 + 36) = (100 * (5/7) - 9)) :
  (63 * 1^2 + 36) = (100 * 1 - 9) ∧ 
  ∀ x : ℚ, x ≠ 5/7 → x ≠ 1 → (63 * x^2 + 36) ≠ (100 * x - 9) :=
by sorry

end quadratic_equation_solution_l141_14145


namespace ticket_price_ratio_l141_14108

/-- Proves that the ratio of adult to child ticket prices is 2:1 given the problem conditions --/
theorem ticket_price_ratio :
  ∀ (adult_price child_price : ℚ),
    adult_price = 32 →
    400 * adult_price + 200 * child_price = 16000 →
    adult_price / child_price = 2 := by
  sorry

end ticket_price_ratio_l141_14108


namespace point_in_first_quadrant_l141_14143

theorem point_in_first_quadrant (m : ℝ) : 
  let P : ℝ × ℝ := (3, m^2 + 1)
  P.1 > 0 ∧ P.2 > 0 := by
sorry

end point_in_first_quadrant_l141_14143


namespace sum_of_edge_lengths_pyramid_volume_l141_14172

-- Define the pyramid
def pyramid_base_side : ℝ := 8
def pyramid_height : ℝ := 15

-- Theorem for the sum of edge lengths
theorem sum_of_edge_lengths :
  let diagonal := pyramid_base_side * Real.sqrt 2
  let slant_edge := Real.sqrt (pyramid_height^2 + (diagonal / 2)^2)
  4 * pyramid_base_side + 4 * slant_edge = 32 + 4 * Real.sqrt 257 :=
sorry

-- Theorem for the volume
theorem pyramid_volume :
  (1 / 3) * pyramid_base_side^2 * pyramid_height = 320 :=
sorry

end sum_of_edge_lengths_pyramid_volume_l141_14172


namespace smallest_digit_divisible_by_three_l141_14115

theorem smallest_digit_divisible_by_three : 
  ∃ (x : ℕ), x < 10 ∧ 
  (526000 + x * 100 + 18) % 3 = 0 ∧
  ∀ (y : ℕ), y < x → y < 10 → (526000 + y * 100 + 18) % 3 ≠ 0 :=
by sorry

end smallest_digit_divisible_by_three_l141_14115


namespace smallest_four_digit_multiple_of_17_l141_14178

theorem smallest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → 1013 ≤ n := by
  sorry

end smallest_four_digit_multiple_of_17_l141_14178


namespace quadratic_inequality_solution_l141_14118

theorem quadratic_inequality_solution (a b : ℝ) (h : ∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :
  (a = 1 ∧ b = 2) ∧
  (∀ m : ℝ,
    (m = 2 → ∀ x, ¬(x^2 - (m + 2) * x + 2 * m < 0)) ∧
    (m < 2 → ∀ x, x^2 - (m + 2) * x + 2 * m < 0 ↔ m < x ∧ x < 2) ∧
    (m > 2 → ∀ x, x^2 - (m + 2) * x + 2 * m < 0 ↔ 2 < x ∧ x < m)) :=
by sorry

end quadratic_inequality_solution_l141_14118


namespace max_value_of_f_in_interval_l141_14157

def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x + 1

theorem max_value_of_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 3 ∧ 
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f c) ∧
  f c = 10 := by
sorry

end max_value_of_f_in_interval_l141_14157


namespace parabola_circle_tangency_l141_14130

/-- A parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an intersection point between a parabola and a circle -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is a tangent point between a parabola and a circle -/
def is_tangent_point (p : Parabola) (c : Circle) (point : IntersectionPoint) : Prop :=
  sorry

/-- Theorem stating that if a circle and a parabola intersect at exactly two points,
    and one is a tangent point, then the other must also be a tangent point -/
theorem parabola_circle_tangency
  (p : Parabola) (c : Circle) 
  (i1 i2 : IntersectionPoint) 
  (h_distinct : i1 ≠ i2)
  (h_only_two : ∀ i : IntersectionPoint, i = i1 ∨ i = i2)
  (h_tangent : is_tangent_point p c i1) :
  is_tangent_point p c i2 :=
sorry

end parabola_circle_tangency_l141_14130


namespace boys_in_third_group_l141_14113

/-- Represents the work rate of a single person --/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers --/
structure WorkGroup where
  boys : ℕ
  girls : ℕ

/-- Calculates the total work done by a group in a given number of days --/
def totalWork (group : WorkGroup) (boyRate girlRate : WorkRate) (days : ℕ) : ℝ :=
  (group.boys : ℝ) * boyRate.rate * (days : ℝ) + (group.girls : ℝ) * girlRate.rate * (days : ℝ)

/-- The main theorem stating that the number of boys in the third group is 26 --/
theorem boys_in_third_group : 
  ∀ (x : ℕ) (boyRate girlRate : WorkRate),
  let group1 := WorkGroup.mk x 20
  let group2 := WorkGroup.mk 6 8
  let group3 := WorkGroup.mk 26 48
  totalWork group1 boyRate girlRate 4 = totalWork group2 boyRate girlRate 10 ∧
  totalWork group1 boyRate girlRate 4 = totalWork group3 boyRate girlRate 2 →
  group3.boys = 26 := by
sorry

end boys_in_third_group_l141_14113


namespace median_angle_relation_l141_14111

/-- Represents a triangle with sides a, b, c, angle γ opposite to side c, and median sc to side c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  γ : ℝ
  sc : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_γ : 0 < γ
  pos_sc : 0 < sc
  tri_ineq : a + b > c ∧ b + c > a ∧ c + a > b

theorem median_angle_relation (t : Triangle) :
  (t.γ < 90 ↔ t.sc > t.c / 2) ∧
  (t.γ = 90 ↔ t.sc = t.c / 2) ∧
  (t.γ > 90 ↔ t.sc < t.c / 2) :=
sorry

end median_angle_relation_l141_14111


namespace mary_warmth_hours_l141_14126

/-- The number of sticks of wood produced by chopping up furniture and the number of hours Mary can keep warm. -/
def furniture_to_warmth (chair_sticks table_sticks cabinet_sticks stool_sticks : ℕ)
  (chairs tables cabinets stools : ℕ) (sticks_per_hour : ℕ) : ℕ :=
  let total_sticks := chair_sticks * chairs + table_sticks * tables + 
                      cabinet_sticks * cabinets + stool_sticks * stools
  total_sticks / sticks_per_hour

/-- Theorem stating that Mary can keep warm for 64 hours given the specified conditions. -/
theorem mary_warmth_hours : 
  furniture_to_warmth 8 12 16 3 25 12 5 8 7 = 64 := by
  sorry

end mary_warmth_hours_l141_14126


namespace gcd_of_B_is_two_l141_14119

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_two :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end gcd_of_B_is_two_l141_14119


namespace min_value_trig_expression_l141_14161

theorem min_value_trig_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 ≥ 236 := by
  sorry

end min_value_trig_expression_l141_14161


namespace magnitude_of_b_l141_14153

/-- Given two vectors a and b in ℝ², prove that the magnitude of b is 2√2 -/
theorem magnitude_of_b (a b : ℝ × ℝ) : 
  let angle := 3 * π / 4
  a = (-3, 4) →
  a.fst * b.fst + a.snd * b.snd = -10 →
  Real.sqrt (b.fst ^ 2 + b.snd ^ 2) = 2 * Real.sqrt 2 := by
  sorry


end magnitude_of_b_l141_14153


namespace dice_rolling_expectation_l141_14158

/-- The expected value of 6^D after n steps in the dice rolling process -/
def expected_value (n : ℕ) : ℝ :=
  6 + 5 * n

/-- The number of steps in the process -/
def num_steps : ℕ := 2013

theorem dice_rolling_expectation :
  expected_value num_steps = 10071 := by
  sorry

end dice_rolling_expectation_l141_14158


namespace negative_modulus_of_complex_l141_14175

theorem negative_modulus_of_complex (z : ℂ) (h : z = 6 + 8*I) : -Complex.abs z = -10 := by
  sorry

end negative_modulus_of_complex_l141_14175


namespace monotonic_quadratic_l141_14156

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 3

theorem monotonic_quadratic (m : ℝ) :
  (∀ x₁ x₂, 1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 → (f m x₁ < f m x₂ ∨ f m x₁ > f m x₂)) →
  m ≤ 1 ∨ m ≥ 3 := by sorry

end monotonic_quadratic_l141_14156


namespace cosine_arithmetic_sequence_product_l141_14150

theorem cosine_arithmetic_sequence_product (a : ℕ → ℝ) (S : Set ℝ) (a₀ b₀ : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + 2 * π / 3) →
  S = {x | ∃ n : ℕ, x = Real.cos (a n)} →
  S = {a₀, b₀} →
  a₀ * b₀ = -1/2 :=
sorry

end cosine_arithmetic_sequence_product_l141_14150


namespace subtraction_result_l141_14169

theorem subtraction_result (number : ℝ) (percentage : ℝ) (subtrahend : ℝ) : 
  number = 200 → 
  percentage = 95 → 
  subtrahend = 12 → 
  (percentage / 100) * number - subtrahend = 178 := by
sorry

end subtraction_result_l141_14169


namespace first_part_count_l141_14173

theorem first_part_count (total_count : Nat) (total_avg : Nat) (first_avg : Nat) (last_avg : Nat) (thirteenth_result : Nat) :
  total_count = 25 →
  total_avg = 18 →
  first_avg = 10 →
  last_avg = 20 →
  thirteenth_result = 90 →
  ∃ n : Nat, n = 14 ∧ 
    n * first_avg + thirteenth_result + (total_count - n) * last_avg = total_count * total_avg :=
by sorry

end first_part_count_l141_14173


namespace smallest_disjoint_r_l141_14147

def A : Set ℤ := {n | ∃ k : ℕ, (n = 3 + 10 * k) ∨ (n = 6 + 26 * k) ∨ (n = 5 + 29 * k)}

def is_disjoint (r b : ℤ) : Prop :=
  ∀ k l : ℕ, (b + r * k) ∉ A

theorem smallest_disjoint_r : 
  (∃ b : ℤ, is_disjoint 290 b) ∧ 
  (∀ r : ℕ, r < 290 → ¬∃ b : ℤ, is_disjoint r b) :=
sorry

end smallest_disjoint_r_l141_14147


namespace percentage_equality_l141_14185

theorem percentage_equality (x y : ℝ) (h1 : 2 * x = 0.5 * y) (h2 : x = 16) : y = 64 := by
  sorry

end percentage_equality_l141_14185


namespace stating_min_messages_proof_l141_14131

/-- Represents the minimum number of messages needed for information distribution -/
def min_messages (n : ℕ) : ℕ := 2 * (n - 1)

/-- 
Theorem stating that the minimum number of messages needed for n people 
to share all information is 2(n-1)
-/
theorem min_messages_proof (n : ℕ) (h : n > 0) : 
  ∀ (f : ℕ → ℕ), 
  (∀ i : ℕ, i < n → f i ≥ min_messages n) → 
  (∃ g : ℕ → ℕ → Bool, 
    (∀ i j : ℕ, i < n ∧ j < n → g i j = true) ∧ 
    (∀ i : ℕ, i < n → ∃ k : ℕ, k < f i ∧ 
      (∀ j : ℕ, j < n → ∃ m : ℕ, m ≤ k ∧ g i j = true))) :=
sorry

#check min_messages_proof

end stating_min_messages_proof_l141_14131


namespace sphere_volume_ratio_l141_14148

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 8 := by
sorry

end sphere_volume_ratio_l141_14148


namespace train_length_proof_l141_14142

/-- Proves that a train crossing a 550-meter platform in 51 seconds and a signal pole in 18 seconds has a length of 300 meters. -/
theorem train_length_proof (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 550)
  (h2 : platform_time = 51)
  (h3 : pole_time = 18) :
  let train_length := (platform_length * pole_time) / (platform_time - pole_time)
  train_length = 300 := by
  sorry

end train_length_proof_l141_14142


namespace equation_solution_l141_14116

theorem equation_solution :
  ∃ x : ℝ, (5 * x - 8 * (2 * x + 3) = 4 * (x - 3 * (2 * x - 5)) + 7 * (2 * x - 5)) ∧ x = -9.8 := by
  sorry

end equation_solution_l141_14116


namespace exam_girls_count_l141_14188

theorem exam_girls_count (total : ℕ) (pass_rate_boys : ℚ) (pass_rate_girls : ℚ) (fail_rate_total : ℚ) :
  total = 2000 ∧
  pass_rate_boys = 30 / 100 ∧
  pass_rate_girls = 32 / 100 ∧
  fail_rate_total = 691 / 1000 →
  ∃ (girls : ℕ), girls = 900 ∧ girls ≤ total ∧
    (girls : ℚ) * pass_rate_girls + (total - girls : ℚ) * pass_rate_boys = (1 - fail_rate_total) * total :=
by sorry

end exam_girls_count_l141_14188


namespace prime_power_plus_144_square_l141_14125

theorem prime_power_plus_144_square (p n m : ℕ) : 
  p.Prime → p > 0 → n > 0 → m > 0 → p^n + 144 = m^2 → 
  (p = 5 ∧ n = 2 ∧ m = 13) ∨ (p = 2 ∧ n = 8 ∧ m = 20) ∨ (p = 3 ∧ n = 4 ∧ m = 15) := by
  sorry

end prime_power_plus_144_square_l141_14125


namespace best_meeting_days_l141_14186

-- Define the days of the week
inductive Day
| Mon
| Tue
| Wed
| Thu
| Fri

-- Define the team members
inductive Member
| Anna
| Bill
| Carl
| Dana

-- Define the availability function
def availability (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Anna, Day.Mon => false
  | Member.Anna, Day.Wed => false
  | Member.Bill, Day.Tue => false
  | Member.Bill, Day.Thu => false
  | Member.Bill, Day.Fri => false
  | Member.Carl, Day.Mon => false
  | Member.Carl, Day.Tue => false
  | Member.Carl, Day.Thu => false
  | Member.Carl, Day.Fri => false
  | Member.Dana, Day.Wed => false
  | Member.Dana, Day.Thu => false
  | _, _ => true

-- Count available members for a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => availability m d) [Member.Anna, Member.Bill, Member.Carl, Member.Dana]).length

-- Define the maximum availability
def maxAvailability : Nat :=
  List.foldl max 0 (List.map availableCount [Day.Mon, Day.Tue, Day.Wed, Day.Thu, Day.Fri])

-- Theorem statement
theorem best_meeting_days :
  (availableCount Day.Mon = maxAvailability) ∧
  (availableCount Day.Tue = maxAvailability) ∧
  (availableCount Day.Wed = maxAvailability) ∧
  (availableCount Day.Thu < maxAvailability) ∧
  (availableCount Day.Fri = maxAvailability) := by
  sorry

end best_meeting_days_l141_14186


namespace mango_boxes_count_l141_14165

/-- Given a number of mangoes per dozen, total mangoes, and mangoes per box,
    calculate the number of boxes. -/
def calculate_boxes (mangoes_per_dozen : ℕ) (total_mangoes : ℕ) (dozens_per_box : ℕ) : ℕ :=
  total_mangoes / (mangoes_per_dozen * dozens_per_box)

/-- Prove that there are 36 boxes of mangoes given the problem conditions. -/
theorem mango_boxes_count :
  let mangoes_per_dozen : ℕ := 12
  let total_mangoes : ℕ := 4320
  let dozens_per_box : ℕ := 10
  calculate_boxes mangoes_per_dozen total_mangoes dozens_per_box = 36 := by
  sorry

#eval calculate_boxes 12 4320 10

end mango_boxes_count_l141_14165


namespace probability_of_selecting_A_and_B_l141_14196

/-- The number of students in total -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B when choosing 3 students out of 5 -/
theorem probability_of_selecting_A_and_B :
  (Nat.choose (total_students - 2) (selected_students - 2)) / 
  (Nat.choose total_students selected_students : ℚ) = 3 / 10 := by
sorry

end probability_of_selecting_A_and_B_l141_14196


namespace flyers_left_to_hand_out_l141_14114

theorem flyers_left_to_hand_out 
  (total_flyers : ℕ) 
  (jack_handed : ℕ) 
  (rose_handed : ℕ) 
  (h1 : total_flyers = 1236)
  (h2 : jack_handed = 120)
  (h3 : rose_handed = 320) : 
  total_flyers - (jack_handed + rose_handed) = 796 :=
by sorry

end flyers_left_to_hand_out_l141_14114


namespace subset_condition_disjoint_condition_l141_14189

-- Define the sets A and S
def A : Set ℝ := {x | -7 ≤ 2*x - 5 ∧ 2*x - 5 ≤ 9}
def S (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2*k - 1}

-- Statement 1
theorem subset_condition (k : ℝ) : 
  (S k).Nonempty ∧ S k ⊆ A ↔ 2 ≤ k ∧ k ≤ 4 := by sorry

-- Statement 2
theorem disjoint_condition (k : ℝ) : 
  A ∩ S k = ∅ ↔ k < 2 ∨ k > 6 := by sorry

end subset_condition_disjoint_condition_l141_14189


namespace kendy_transfer_proof_l141_14123

-- Define the initial balance
def initial_balance : ℚ := 190

-- Define the remaining balance
def remaining_balance : ℚ := 100

-- Define the amount transferred to mom
def amount_to_mom : ℚ := 60

-- Define the amount transferred to sister
def amount_to_sister : ℚ := amount_to_mom / 2

-- Theorem statement
theorem kendy_transfer_proof :
  initial_balance - (amount_to_mom + amount_to_sister) = remaining_balance :=
by sorry

end kendy_transfer_proof_l141_14123


namespace no_rational_solution_l141_14163

theorem no_rational_solution : ¬∃ (a b : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6) := by
  sorry

end no_rational_solution_l141_14163


namespace units_digit_of_seven_to_ten_l141_14144

theorem units_digit_of_seven_to_ten (n : ℕ) : 7^10 ≡ 9 [MOD 10] := by
  sorry

end units_digit_of_seven_to_ten_l141_14144


namespace sum_of_ages_five_children_l141_14176

/-- Calculates the sum of ages for a group of children born at regular intervals -/
def sumOfAges (numChildren : ℕ) (ageInterval : ℕ) (youngestAge : ℕ) : ℕ :=
  let ages := List.range numChildren |>.map (fun i => youngestAge + i * ageInterval)
  ages.sum

/-- Proves that the sum of ages for 5 children born at 2-year intervals, with the youngest being 6, is 50 -/
theorem sum_of_ages_five_children :
  sumOfAges 5 2 6 = 50 := by
  sorry

end sum_of_ages_five_children_l141_14176


namespace unique_B_for_divisibility_l141_14132

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def number_4BBB2 (B : ℕ) : ℕ := 40000 + 1000 * B + 100 * B + 10 * B + 2

theorem unique_B_for_divisibility :
  ∃! B : ℕ, digit B ∧ is_divisible_by_9 (number_4BBB2 B) ∧ B = 4 := by
  sorry

end unique_B_for_divisibility_l141_14132


namespace square_perimeter_with_area_9_l141_14146

theorem square_perimeter_with_area_9 (s : ℝ) (h1 : s^2 = 9) (h2 : ∃ k : ℕ, 4 * s = 4 * k) : 4 * s = 12 := by
  sorry

end square_perimeter_with_area_9_l141_14146


namespace intersection_implies_C_value_l141_14192

/-- Two lines intersect on the y-axis iff their intersection point has x-coordinate 0 -/
def intersect_on_y_axis (A C : ℝ) : Prop :=
  ∃ y : ℝ, A * 0 + 3 * y + C = 0 ∧ 2 * 0 - 3 * y + 4 = 0

/-- If the lines Ax + 3y + C = 0 and 2x - 3y + 4 = 0 intersect on the y-axis, then C = -4 -/
theorem intersection_implies_C_value (A : ℝ) :
  intersect_on_y_axis A C → C = -4 :=
sorry

end intersection_implies_C_value_l141_14192


namespace art_project_marker_distribution_l141_14194

/-- Proves that each student in the last group receives 5 markers given the conditions of the art project. -/
theorem art_project_marker_distribution :
  let total_students : ℕ := 68
  let total_groups : ℕ := 5
  let total_marker_boxes : ℕ := 48
  let markers_per_box : ℕ := 6
  let group1_students : ℕ := 12
  let group1_markers_per_student : ℕ := 2
  let group2_students : ℕ := 20
  let group2_markers_per_student : ℕ := 3
  let group3_students : ℕ := 15
  let group3_markers_per_student : ℕ := 5
  let group4_students : ℕ := 8
  let group4_markers_per_student : ℕ := 8
  let total_markers : ℕ := total_marker_boxes * markers_per_box
  let used_markers : ℕ := group1_students * group1_markers_per_student +
                          group2_students * group2_markers_per_student +
                          group3_students * group3_markers_per_student +
                          group4_students * group4_markers_per_student
  let remaining_markers : ℕ := total_markers - used_markers
  let last_group_students : ℕ := total_students - (group1_students + group2_students + group3_students + group4_students)
  remaining_markers / last_group_students = 5 :=
by sorry

end art_project_marker_distribution_l141_14194


namespace max_value_at_two_l141_14140

/-- The function f(x) = x(x-m)² -/
def f (m : ℝ) (x : ℝ) : ℝ := x * (x - m)^2

/-- The derivative of f(x) -/
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*m*x + m^2

theorem max_value_at_two (m : ℝ) :
  (∀ x : ℝ, f m x ≤ f m 2) →
  (m = 6 ∧
   ∀ a : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                             f m x₁ = a ∧ f m x₂ = a ∧ f m x₃ = a) ↔
             (0 < a ∧ a < 32)) :=
by sorry

end max_value_at_two_l141_14140


namespace tan_intersection_distance_l141_14141

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem tan_intersection_distance (a : ℝ) :
  ∃ (d : ℝ), d > 0 ∧
  ∀ (x : ℝ), tan x = a → tan (x + d) = a ∧
  ∀ (y : ℝ), 0 < y ∧ y < d → tan (x + y) ≠ a :=
by
  -- The proof would go here
  sorry

end tan_intersection_distance_l141_14141


namespace jumping_contest_l141_14164

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper frog mouse : ℕ) 
  (h1 : grasshopper = 14)
  (h2 : mouse = frog - 16)
  (h3 : mouse = grasshopper + 21) :
  frog - grasshopper = 37 := by
  sorry

end jumping_contest_l141_14164


namespace fraction_equality_l141_14102

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x + 2 * y) / (x - 5 * y) = 3) : 
  (x + 5 * y) / (5 * x - y) = 7 / 87 := by
  sorry

end fraction_equality_l141_14102


namespace reduced_price_per_dozen_l141_14120

/-- Represents the price reduction percentage -/
def price_reduction : ℝ := 0.40

/-- Represents the additional number of apples that can be bought after the price reduction -/
def additional_apples : ℕ := 64

/-- Represents the fixed amount of money spent on apples -/
def fixed_amount : ℝ := 40

/-- Represents the number of apples in a dozen -/
def apples_per_dozen : ℕ := 12

/-- Theorem stating that given the conditions, the reduced price per dozen apples is Rs. 3 -/
theorem reduced_price_per_dozen (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := original_price * (1 - price_reduction)
  let original_quantity := fixed_amount / original_price
  let new_quantity := original_quantity + additional_apples
  (new_quantity : ℝ) * reduced_price = fixed_amount →
  apples_per_dozen * (fixed_amount / new_quantity) = 3 :=
by sorry

end reduced_price_per_dozen_l141_14120


namespace division_remainder_proof_l141_14124

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 689 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end division_remainder_proof_l141_14124


namespace isosceles_triangle_perimeter_l141_14149

/-- An isosceles triangle with side lengths a, b, and c, where two sides are 11 and one side is 5 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_isosceles : (a = b ∧ a = 11 ∧ c = 5) ∨ (a = c ∧ a = 11 ∧ b = 5) ∨ (b = c ∧ b = 11 ∧ a = 5)
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of an isosceles triangle with two sides of length 11 and one side of length 5 is 27 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : t.a + t.b + t.c = 27 := by
  sorry

end isosceles_triangle_perimeter_l141_14149


namespace circle_center_and_point_check_l141_14199

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 11 = 0

def center : ℝ × ℝ := (3, -1)

def point : ℝ × ℝ := (5, -1)

theorem circle_center_and_point_check :
  (∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = 21) ∧
  ¬ circle_equation point.1 point.2 := by
  sorry

end circle_center_and_point_check_l141_14199


namespace teacher_health_survey_l141_14107

theorem teacher_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : high_bp = 80)
  (h3 : heart_trouble = 50)
  (h4 : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 100/3 :=
sorry

end teacher_health_survey_l141_14107


namespace valid_tiling_exists_l141_14154

/-- Represents a point on the infinite 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a domino piece on the grid -/
inductive Domino
  | Horizontal (topLeft : GridPoint)
  | Vertical (topLeft : GridPoint)

/-- Represents a tiling of the infinite grid with dominos -/
def Tiling := GridPoint → Domino

/-- Checks if a given point is covered by a domino in the tiling -/
def isCovered (t : Tiling) (p : GridPoint) : Prop := 
  ∃ d : Domino, d ∈ Set.range t ∧ 
    match d with
    | Domino.Horizontal tl => p.x = tl.x ∧ (p.y = tl.y ∨ p.y = tl.y + 1)
    | Domino.Vertical tl => p.y = tl.y ∧ (p.x = tl.x ∨ p.x = tl.x + 1)

/-- Checks if a horizontal line intersects a finite number of dominos -/
def finiteHorizontalIntersections (t : Tiling) : Prop :=
  ∀ y : ℤ, ∃ n : ℕ, ∀ x : ℤ, x > n → 
    (t ⟨x, y⟩ = t ⟨x - 1, y⟩ ∨ t ⟨x, y⟩ = t ⟨x - 1, y - 1⟩)

/-- Checks if a vertical line intersects a finite number of dominos -/
def finiteVerticalIntersections (t : Tiling) : Prop :=
  ∀ x : ℤ, ∃ n : ℕ, ∀ y : ℤ, y > n → 
    (t ⟨x, y⟩ = t ⟨x, y - 1⟩ ∨ t ⟨x, y⟩ = t ⟨x - 1, y - 1⟩)

/-- The main theorem stating that a valid tiling with the required properties exists -/
theorem valid_tiling_exists : 
  ∃ t : Tiling, 
    (∀ p : GridPoint, isCovered t p) ∧ 
    finiteHorizontalIntersections t ∧
    finiteVerticalIntersections t := by
  sorry

end valid_tiling_exists_l141_14154


namespace four_digit_number_not_divisible_by_11_l141_14105

def is_not_divisible_by_11 (n : ℕ) : Prop := ¬(n % 11 = 0)

theorem four_digit_number_not_divisible_by_11 :
  ∀ B : ℕ, B < 10 →
  (∃ A : ℕ, A < 10 ∧ 
    (∀ B : ℕ, B < 10 → is_not_divisible_by_11 (9000 + 100 * A + 10 * B))) ↔ 
  (∃ A : ℕ, A = 1) :=
by sorry

end four_digit_number_not_divisible_by_11_l141_14105


namespace stratified_sample_size_l141_14121

/-- Represents the ratio of students in three schools -/
structure SchoolRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total sample size given the number of students sampled from the smallest school -/
def totalSampleSize (ratio : SchoolRatio) (smallestSchoolSample : ℕ) : ℕ :=
  smallestSchoolSample * (ratio.a + ratio.b + ratio.c) / ratio.a

/-- Theorem: For schools with ratio 2:3:5, if 10 students are sampled from the smallest school, the total sample is 50 -/
theorem stratified_sample_size (ratio : SchoolRatio) (h1 : ratio.a = 2) (h2 : ratio.b = 3) (h3 : ratio.c = 5) :
  totalSampleSize ratio 10 = 50 := by
  sorry

#eval totalSampleSize ⟨2, 3, 5⟩ 10

end stratified_sample_size_l141_14121


namespace max_m_value_min_quadratic_sum_l141_14129

-- Part 1
theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x + m| ≥ 2*m) → m ≤ 1 :=
sorry

-- Part 2
theorem min_quadratic_sum {a b c : ℝ} 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  2*a^2 + 3*b^2 + 4*c^2 ≥ 12/13 ∧ 
  (2*a^2 + 3*b^2 + 4*c^2 = 12/13 ↔ a = 6/13 ∧ b = 4/13 ∧ c = 3/13) :=
sorry

end max_m_value_min_quadratic_sum_l141_14129


namespace tangent_line_to_circle_l141_14182

theorem tangent_line_to_circle (m : ℝ) : 
  m > 0 → 
  (∀ x y : ℝ, x + y = 0 → (x - m)^2 + y^2 = 2) → 
  m = 2 := by
sorry

end tangent_line_to_circle_l141_14182


namespace even_monotone_decreasing_inequality_l141_14184

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f y < f x

-- State the theorem
theorem even_monotone_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_mono : monotone_decreasing_on_pos f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end even_monotone_decreasing_inequality_l141_14184


namespace number_of_candles_l141_14170

def candle_weight : ℕ := 9
def total_weight : ℕ := 63

theorem number_of_candles : (total_weight / candle_weight = 7) := by
  sorry

end number_of_candles_l141_14170


namespace parking_garage_floors_l141_14137

/-- Represents a parking garage with the given properties -/
structure ParkingGarage where
  floors : ℕ
  drive_time : ℕ
  id_check_time : ℕ
  total_time : ℕ

/-- Calculates the number of ID checks required -/
def id_checks (g : ParkingGarage) : ℕ := (g.floors - 1) / 3

/-- Calculates the total time to traverse the garage -/
def calculate_total_time (g : ParkingGarage) : ℕ :=
  g.drive_time * (g.floors - 1) + g.id_check_time * id_checks g

/-- Theorem stating that a parking garage with the given properties has 13 floors -/
theorem parking_garage_floors :
  ∃ (g : ParkingGarage), 
    g.drive_time = 80 ∧ 
    g.id_check_time = 120 ∧ 
    g.total_time = 1440 ∧ 
    calculate_total_time g = g.total_time ∧ 
    g.floors = 13 := by
  sorry

end parking_garage_floors_l141_14137
