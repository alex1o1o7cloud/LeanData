import Mathlib

namespace no_factors_of_p_l2481_248180

def p (x : ℝ) : ℝ := x^4 - 3*x^2 + 5

theorem no_factors_of_p :
  (∀ x, p x ≠ (x^2 + 1) * (x^2 - 3*x + 5)) ∧
  (∀ x, p x ≠ (x - 1) * (x^3 + x^2 - 2*x - 5)) ∧
  (∀ x, p x ≠ (x^2 + 5) * (x^2 - 5)) ∧
  (∀ x, p x ≠ (x^2 + 2*x + 1) * (x^2 - 2*x + 4)) :=
by
  sorry

#check no_factors_of_p

end no_factors_of_p_l2481_248180


namespace consecutive_factorials_divisible_by_61_l2481_248132

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem consecutive_factorials_divisible_by_61 (k : ℕ) :
  (∃ m : ℕ, factorial (k - 2) + factorial (k - 1) + factorial k = 61 * m) →
  k ≥ 61 := by
sorry

end consecutive_factorials_divisible_by_61_l2481_248132


namespace solution_set_inequality_l2481_248176

theorem solution_set_inequality (x : ℝ) : 
  Set.Icc (-2 : ℝ) 1 = {x | (1 - x) / (2 + x) ≥ 0} := by sorry

end solution_set_inequality_l2481_248176


namespace square_sum_geq_two_l2481_248157

theorem square_sum_geq_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^2 + b^2 ≥ 2 := by
  sorry

end square_sum_geq_two_l2481_248157


namespace seven_eighths_of_48_l2481_248182

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end seven_eighths_of_48_l2481_248182


namespace xyz_product_abs_l2481_248128

theorem xyz_product_abs (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h_eq : x + 2/y = y + 2/z ∧ y + 2/z = z + 2/x) :
  |x * y * z| = 2 := by sorry

end xyz_product_abs_l2481_248128


namespace surface_area_after_vertex_removal_l2481_248195

/-- The surface area of a cube after removing unit cubes from its vertices -/
theorem surface_area_after_vertex_removal (side_length : ℝ) (h : side_length = 4) :
  6 * side_length^2 = 6 * side_length^2 := by sorry

end surface_area_after_vertex_removal_l2481_248195


namespace reciprocal_of_negative_three_l2481_248159

theorem reciprocal_of_negative_three :
  (1 : ℝ) / (-3 : ℝ) = -1/3 := by sorry

end reciprocal_of_negative_three_l2481_248159


namespace right_triangle_arithmetic_sequence_l2481_248110

theorem right_triangle_arithmetic_sequence (a b c : ℝ) (area : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a < b → b < c →
  c - b = b - a →
  a^2 + b^2 = c^2 →
  area = (1/2) * a * b →
  area = 1350 →
  (a, b, c) = (45, 60, 75) := by sorry

end right_triangle_arithmetic_sequence_l2481_248110


namespace g_15_equals_274_l2481_248191

/-- The function g defined for all natural numbers -/
def g (n : ℕ) : ℕ := n^2 + 2*n + 19

/-- Theorem stating that g(15) equals 274 -/
theorem g_15_equals_274 : g 15 = 274 := by
  sorry

end g_15_equals_274_l2481_248191


namespace soccer_team_red_cards_l2481_248134

theorem soccer_team_red_cards 
  (total_players : ℕ) 
  (players_without_cautions : ℕ) 
  (yellow_cards_per_cautioned_player : ℕ) 
  (yellow_cards_per_red_card : ℕ) 
  (h1 : total_players = 11) 
  (h2 : players_without_cautions = 5) 
  (h3 : yellow_cards_per_cautioned_player = 1) 
  (h4 : yellow_cards_per_red_card = 2) : 
  (total_players - players_without_cautions) * yellow_cards_per_cautioned_player / yellow_cards_per_red_card = 3 := by
  sorry

end soccer_team_red_cards_l2481_248134


namespace lizzie_has_27_crayons_l2481_248168

def billie_crayons : ℕ := 18

def bobbie_crayons (billie : ℕ) : ℕ := 3 * billie

def lizzie_crayons (bobbie : ℕ) : ℕ := bobbie / 2

theorem lizzie_has_27_crayons :
  lizzie_crayons (bobbie_crayons billie_crayons) = 27 :=
by sorry

end lizzie_has_27_crayons_l2481_248168


namespace biggest_measure_for_containers_l2481_248152

theorem biggest_measure_for_containers (a b c : ℕ) 
  (ha : a = 496) (hb : b = 403) (hc : c = 713) : 
  Nat.gcd a (Nat.gcd b c) = 31 := by
  sorry

end biggest_measure_for_containers_l2481_248152


namespace product_of_numbers_l2481_248179

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 := by
  sorry

end product_of_numbers_l2481_248179


namespace complement_union_A_B_l2481_248148

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

theorem complement_union_A_B : 
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end complement_union_A_B_l2481_248148


namespace floor_neg_three_point_seven_l2481_248142

/-- The floor function, which returns the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem: The floor of -3.7 is -4 -/
theorem floor_neg_three_point_seven :
  floor (-3.7) = -4 := by
  sorry

end floor_neg_three_point_seven_l2481_248142


namespace eleventh_number_with_digit_sum_12_l2481_248158

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ+) : ℕ+ := sorry

theorem eleventh_number_with_digit_sum_12 :
  nth_number_with_digit_sum_12 11 = 147 := by sorry

end eleventh_number_with_digit_sum_12_l2481_248158


namespace percentage_increase_l2481_248109

theorem percentage_increase (original : ℝ) (new : ℝ) : 
  original = 60 → new = 150 → (new - original) / original * 100 = 150 := by
  sorry

end percentage_increase_l2481_248109


namespace probability_of_graduate_degree_l2481_248129

theorem probability_of_graduate_degree (G C N : ℕ) : 
  G * 8 = N →
  C * 3 = N * 2 →
  (G : ℚ) / (G + C) = 3 / 19 :=
by sorry

end probability_of_graduate_degree_l2481_248129


namespace equilateral_triangle_exists_l2481_248194

-- Define a point in a plane
def Point := ℝ × ℝ

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define a predicate for an equilateral triangle
def is_equilateral_triangle (p1 p2 p3 : Point) : Prop :=
  distance p1 p2 = 1 ∧ distance p2 p3 = 1 ∧ distance p3 p1 = 1

-- Main theorem
theorem equilateral_triangle_exists 
  (points : Finset Point) 
  (h1 : points.card = 6) 
  (h2 : ∃ (pairs : Finset (Point × Point)), 
    pairs.card = 8 ∧ 
    ∀ (pair : Point × Point), pair ∈ pairs → distance pair.1 pair.2 = 1) :
  ∃ (p1 p2 p3 : Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
    is_equilateral_triangle p1 p2 p3 :=
sorry

end equilateral_triangle_exists_l2481_248194


namespace range_of_m_l2481_248104

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m + 3) + y^2 / (7*m - 3) = 1 ∧ m + 3 > 0 ∧ 7*m - 3 < 0

def q (m : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (5 - 2*m)^x₁ < (5 - 2*m)^x₂

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (m ≤ -3 ∨ (3/7 ≤ m ∧ m < 2)) :=
sorry

end range_of_m_l2481_248104


namespace sunglasses_hat_probability_l2481_248187

theorem sunglasses_hat_probability 
  (total_sunglasses : ℕ) 
  (total_hats : ℕ) 
  (hat_also_sunglasses_prob : ℚ) :
  total_sunglasses = 80 →
  total_hats = 45 →
  hat_also_sunglasses_prob = 1/3 →
  (total_hats * hat_also_sunglasses_prob : ℚ) / total_sunglasses = 3/16 := by
sorry

end sunglasses_hat_probability_l2481_248187


namespace license_plate_count_l2481_248122

/-- The number of digits in a license plate -/
def num_digits : ℕ := 5

/-- The number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- The number of consonants in the alphabet -/
def num_consonants : ℕ := 21

/-- The number of positions where the consonant pair can be placed -/
def consonant_pair_positions : ℕ := 6

/-- The number of distinct license plates -/
def num_license_plates : ℕ := 
  consonant_pair_positions * digit_choices ^ num_digits * (num_consonants * (num_consonants - 1))

theorem license_plate_count : num_license_plates = 2520000000 := by
  sorry

end license_plate_count_l2481_248122


namespace stocking_stuffers_l2481_248130

theorem stocking_stuffers (num_kids : ℕ) (candy_canes_per_stocking : ℕ) (beanie_babies_per_stocking : ℕ) (total_stuffers : ℕ) : 
  num_kids = 3 → 
  candy_canes_per_stocking = 4 → 
  beanie_babies_per_stocking = 2 → 
  total_stuffers = 21 → 
  (total_stuffers - (candy_canes_per_stocking + beanie_babies_per_stocking) * num_kids) / num_kids = 1 :=
by sorry

end stocking_stuffers_l2481_248130


namespace cube_sum_theorem_l2481_248178

theorem cube_sum_theorem (a b c : ℕ) : 
  a^3 = 1 + 7 ∧ 
  3^3 = 1 + 7 + b ∧ 
  4^3 = 1 + 7 + c → 
  a + b + c = 77 := by
  sorry

end cube_sum_theorem_l2481_248178


namespace total_weight_is_120_pounds_l2481_248174

/-- The weight of a single dumbbell in pounds -/
def dumbbell_weight : ℕ := 20

/-- The number of dumbbells initially set up -/
def initial_dumbbells : ℕ := 4

/-- The number of additional dumbbells Parker adds -/
def added_dumbbells : ℕ := 2

/-- The total number of dumbbells Parker uses -/
def total_dumbbells : ℕ := initial_dumbbells + added_dumbbells

/-- Theorem: The total weight of dumbbells Parker is using is 120 pounds -/
theorem total_weight_is_120_pounds :
  total_dumbbells * dumbbell_weight = 120 := by
  sorry


end total_weight_is_120_pounds_l2481_248174


namespace polynomial_division_l2481_248121

theorem polynomial_division (x : ℝ) : 
  x^6 - 14*x^4 + 8*x^3 - 26*x^2 + 14*x - 3 = 
  (x - 3) * (x^5 + 3*x^4 - 5*x^3 - 7*x^2 - 47*x - 7) + (-24) := by
sorry

end polynomial_division_l2481_248121


namespace smallest_prime_divisor_of_sum_l2481_248165

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  2 = Nat.minFac (5^23 + 7^17) := by sorry

end smallest_prime_divisor_of_sum_l2481_248165


namespace least_apples_total_l2481_248198

/-- Represents the number of apples a monkey initially takes --/
structure MonkeyTake where
  apples : ℕ

/-- Represents the final distribution of apples for each monkey --/
structure MonkeyFinal where
  apples : ℕ

/-- Calculates the final number of apples for each monkey based on initial takes --/
def calculateFinal (m1 m2 m3 : MonkeyTake) : (MonkeyFinal × MonkeyFinal × MonkeyFinal) :=
  let f1 := MonkeyFinal.mk ((m1.apples / 2) + (m2.apples / 3) + (5 * m3.apples / 12))
  let f2 := MonkeyFinal.mk ((m1.apples / 4) + (m2.apples / 3) + (5 * m3.apples / 12))
  let f3 := MonkeyFinal.mk ((m1.apples / 4) + (m2.apples / 3) + (m3.apples / 6))
  (f1, f2, f3)

/-- Checks if the final distribution satisfies the 4:3:2 ratio --/
def satisfiesRatio (f1 f2 f3 : MonkeyFinal) : Prop :=
  4 * f2.apples = 3 * f1.apples ∧ 3 * f3.apples = 2 * f2.apples

/-- The main theorem stating the least possible total number of apples --/
theorem least_apples_total : 
  ∃ (m1 m2 m3 : MonkeyTake), 
    let (f1, f2, f3) := calculateFinal m1 m2 m3
    satisfiesRatio f1 f2 f3 ∧ 
    m1.apples + m2.apples + m3.apples = 336 ∧
    (∀ (n1 n2 n3 : MonkeyTake),
      let (g1, g2, g3) := calculateFinal n1 n2 n3
      satisfiesRatio g1 g2 g3 → 
      n1.apples + n2.apples + n3.apples ≥ 336) :=
sorry

end least_apples_total_l2481_248198


namespace correct_quadratic_equation_l2481_248162

theorem correct_quadratic_equation :
  ∀ (a b c : ℝ),
  (∃ (a' : ℝ), a' ≠ a ∧ (a' * 4 * 4 + b * 4 + c = 0) ∧ (a' * (-3) * (-3) + b * (-3) + c = 0)) →
  (∃ (c' : ℝ), c' ≠ c ∧ (a * 7 * 7 + b * 7 + c' = 0) ∧ (a * 3 * 3 + b * 3 + c' = 0)) →
  (a = 1 ∧ b = 10 ∧ c = 21) :=
by sorry

end correct_quadratic_equation_l2481_248162


namespace total_knitting_time_l2481_248123

/-- Represents the time in hours to knit each item of clothing --/
structure KnittingTime where
  hat : ℝ
  scarf : ℝ
  sweater : ℝ
  mitten : ℝ
  sock : ℝ

/-- Calculates the total time to knit one complete outfit --/
def outfitTime (t : KnittingTime) : ℝ :=
  t.hat + t.scarf + t.sweater + 2 * t.mitten + 2 * t.sock

/-- Theorem stating the total time to knit 3 outfits --/
theorem total_knitting_time (t : KnittingTime)
  (hat_time : t.hat = 2)
  (scarf_time : t.scarf = 3)
  (sweater_time : t.sweater = 6)
  (mitten_time : t.mitten = 1)
  (sock_time : t.sock = 1.5) :
  3 * outfitTime t = 48 := by
  sorry


end total_knitting_time_l2481_248123


namespace february_2013_days_l2481_248156

/-- A function that determines if a given year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  year % 4 = 0

/-- A function that returns the number of days in February for a given year -/
def daysInFebruary (year : ℕ) : ℕ :=
  if isLeapYear year then 29 else 28

/-- Theorem stating that February in 2013 has 28 days -/
theorem february_2013_days : daysInFebruary 2013 = 28 := by
  sorry

#eval daysInFebruary 2013

end february_2013_days_l2481_248156


namespace vectors_perpendicular_distance_AB_l2481_248119

-- Define the line and parabola
def line (x y : ℝ) : Prop := y = x - 2
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define points A and B as intersections
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define O as the origin
def O : ℝ × ℝ := (0, 0)

-- Vector from O to A
def OA : ℝ × ℝ := (A.1 - O.1, A.2 - O.2)

-- Vector from O to B
def OB : ℝ × ℝ := (B.1 - O.1, B.2 - O.2)

-- Theorem 1: OA ⊥ OB
theorem vectors_perpendicular : OA.1 * OB.1 + OA.2 * OB.2 = 0 := by sorry

-- Theorem 2: |AB| = 2√10
theorem distance_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 10 := by sorry

end vectors_perpendicular_distance_AB_l2481_248119


namespace rectangle_area_l2481_248190

theorem rectangle_area (w : ℝ) (h : w > 0) : 
  (4 * w = 4 * w) ∧ (2 * (4 * w) + 2 * w = 200) → 4 * w * w = 1600 := by
  sorry

end rectangle_area_l2481_248190


namespace raviraj_journey_l2481_248160

def journey (initial_south distance_after_first_turn second_north final_west distance_to_home : ℝ) : Prop :=
  initial_south = 20 ∧
  second_north = 20 ∧
  final_west = 20 ∧
  distance_to_home = 30 ∧
  distance_after_first_turn + final_west = distance_to_home

theorem raviraj_journey :
  ∀ initial_south distance_after_first_turn second_north final_west distance_to_home,
    journey initial_south distance_after_first_turn second_north final_west distance_to_home →
    distance_after_first_turn = 10 :=
by
  sorry

end raviraj_journey_l2481_248160


namespace vector_addition_l2481_248127

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![1, -1]

theorem vector_addition :
  (vector_a + vector_b) = ![2, 1] := by sorry

end vector_addition_l2481_248127


namespace cubic_function_property_l2481_248167

/-- Given a cubic function f(x) = ax³ + bx + 1 where ab ≠ 0, 
    if f(2016) = k, then f(-2016) = 2-k -/
theorem cubic_function_property (a b k : ℝ) (h1 : a * b ≠ 0) :
  let f := λ x : ℝ => a * x^3 + b * x + 1
  f 2016 = k → f (-2016) = 2 - k := by
sorry

end cubic_function_property_l2481_248167


namespace a_7_not_prime_l2481_248151

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Sequence defined by the recursive formula -/
def a : ℕ → ℕ
  | 0 => 1  -- a_1 is a positive integer
  | n + 1 => a n + reverseDigits (a n)

theorem a_7_not_prime : ∃ (k : ℕ), k > 1 ∧ k < a 7 ∧ a 7 % k = 0 := by sorry

end a_7_not_prime_l2481_248151


namespace average_height_calculation_l2481_248125

theorem average_height_calculation (north_count : ℕ) (north_avg : ℝ) 
  (south_count : ℕ) (south_avg : ℝ) : 
  north_count = 300 → 
  south_count = 200 → 
  north_avg = 1.60 → 
  south_avg = 1.50 → 
  let total_count := north_count + south_count
  let total_height := north_count * north_avg + south_count * south_avg
  (total_height / total_count : ℝ) = 1.56 := by sorry

end average_height_calculation_l2481_248125


namespace inequality_1_inequality_2_l2481_248116

-- First inequality
theorem inequality_1 (x : ℝ) : (2*x - 1)/3 - (9*x + 2)/6 ≤ 1 ↔ x ≥ -2 := by sorry

-- Second system of inequalities
theorem inequality_2 (x : ℝ) : 
  (x - 3*(x - 2) ≥ 4 ∧ (2*x - 1)/5 < (x + 1)/2) ↔ -7 < x ∧ x ≤ 1 := by sorry

end inequality_1_inequality_2_l2481_248116


namespace angle_c_is_right_angle_l2481_248186

theorem angle_c_is_right_angle 
  (A B C : ℝ) 
  (triangle_condition : A + B + C = Real.pi)
  (condition1 : Real.sin A + Real.cos B = Real.sqrt 2)
  (condition2 : Real.cos A + Real.sin B = Real.sqrt 2) : 
  C = Real.pi / 2 := by
  sorry

end angle_c_is_right_angle_l2481_248186


namespace negation_of_universal_proposition_l2481_248131

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l2481_248131


namespace trig_identity_l2481_248111

theorem trig_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin x + Real.cos (x + y) * Real.cos x = Real.cos y := by
  sorry

end trig_identity_l2481_248111


namespace quadratic_inequality_solution_quadratic_inequality_range_l2481_248161

-- Problem 1
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ -1 < x ∧ x < 2) →
  a = 1 ∧ b = -2 :=
sorry

-- Problem 2
theorem quadratic_inequality_range (c : ℝ) :
  (∀ x : ℝ, x ≥ 1 → x^2 + 3*x - c > 0) →
  c < 4 :=
sorry

end quadratic_inequality_solution_quadratic_inequality_range_l2481_248161


namespace inequality_not_always_hold_l2481_248106

theorem inequality_not_always_hold (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) :
  ¬ (∀ a b c, a > b ∧ b > 0 ∧ c ≠ 0 → (a - b) / c > 0) := by
  sorry

end inequality_not_always_hold_l2481_248106


namespace tangent_line_at_origin_l2481_248188

/-- Given a real number a, a function f, and its derivative f', 
    prove that the tangent line at the origin has slope -3 
    when f'(x) is an even function. -/
theorem tangent_line_at_origin (a : ℝ) 
  (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a*x^2 + (a-3)*x) 
  (h2 : ∀ x, (deriv f) x = f' x) 
  (h3 : ∀ x, f' x = f' (-x)) : 
  (deriv f) 0 = -3 := by
  sorry

end tangent_line_at_origin_l2481_248188


namespace pencil_ratio_l2481_248126

/-- Given the number of pencils for Tyrah, Tim, and Sarah, prove the ratio of Tim's to Sarah's pencils -/
theorem pencil_ratio (sarah_pencils tyrah_pencils tim_pencils : ℕ) 
  (h1 : tyrah_pencils = 6 * sarah_pencils)
  (h2 : tyrah_pencils = 12)
  (h3 : tim_pencils = 16) :
  tim_pencils / sarah_pencils = 8 := by
sorry

end pencil_ratio_l2481_248126


namespace count_numbers_with_three_between_100_and_499_l2481_248173

def count_numbers_with_three (lower_bound upper_bound : ℕ) : ℕ :=
  let first_digit_three := 100
  let second_digit_three := 40
  let third_digit_three := 40
  let all_digits_three := 1
  first_digit_three + second_digit_three + third_digit_three - all_digits_three

theorem count_numbers_with_three_between_100_and_499 :
  count_numbers_with_three 100 499 = 181 := by
  sorry

#eval count_numbers_with_three 100 499

end count_numbers_with_three_between_100_and_499_l2481_248173


namespace geometric_sequence_property_l2481_248136

/-- A geometric sequence with common ratio q > 1 and positive first term -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ a 1 > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) 
    (h : GeometricSequence a q) 
    (eq : a 2 * a 4 + a 4 * a 10 - a 4 * a 6 - a 5 * a 5 = 9) :
  a 3 - a 7 = -3 := by
  sorry

end geometric_sequence_property_l2481_248136


namespace fraction_problem_l2481_248169

theorem fraction_problem : ∃ (a b : ℤ), 
  (a - 1 : ℚ) / b = 2/3 ∧ 
  (a - 2 : ℚ) / b = 1/2 ∧ 
  (a : ℚ) / b = 5/6 := by
sorry

end fraction_problem_l2481_248169


namespace katie_new_games_l2481_248192

/-- Given that Katie has some new games and 39 old games,
    her friends have 34 new games, and Katie has 62 more games than her friends,
    prove that Katie has 57 new games. -/
theorem katie_new_games :
  ∀ (new_games : ℕ),
  new_games + 39 = 34 + 62 →
  new_games = 57 :=
by
  sorry

end katie_new_games_l2481_248192


namespace f_monotonicity_and_max_k_l2481_248150

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + Real.log x) / x

theorem f_monotonicity_and_max_k :
  (∀ m : ℝ, m ≥ 1 → ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f m x₁ > f m x₂) ∧
  (∀ m : ℝ, m < 1 → ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → x₂ ≤ Real.exp (1 - m) → f m x₁ < f m x₂) ∧
  (∀ m : ℝ, m < 1 → ∀ x₁ x₂ : ℝ, Real.exp (1 - m) < x₁ → x₁ < x₂ → f m x₁ > f m x₂) ∧
  (∀ x : ℝ, x > 1 → 6 / (x + 1) < f 4 x) ∧
  (∀ k : ℕ, k > 6 → ∃ x : ℝ, x > 1 ∧ k / (x + 1) ≥ f 4 x) :=
by sorry

end f_monotonicity_and_max_k_l2481_248150


namespace tom_payment_l2481_248120

/-- The total amount Tom paid to the shopkeeper for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 1235 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 75 = 1235 := by
  sorry

end tom_payment_l2481_248120


namespace fraction_division_simplify_fraction_divide_fractions_l2481_248140

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction (n d : ℤ) (hd : d ≠ 0) :
  (n / d : ℚ) = (n / gcd n d) / (d / gcd n d) :=
by sorry

theorem divide_fractions : (5 / 6 : ℚ) / (-9 / 10) = -25 / 27 :=
by sorry

end fraction_division_simplify_fraction_divide_fractions_l2481_248140


namespace sqrt_x_minus_one_meaningful_l2481_248138

theorem sqrt_x_minus_one_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_meaningful_l2481_248138


namespace pizza_toppings_combinations_l2481_248107

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end pizza_toppings_combinations_l2481_248107


namespace eighth_term_value_l2481_248175

/-- An arithmetic sequence with the given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, 
    a 1 = 1 ∧ 
    (∀ n, a (n + 1) = a n + d) ∧
    a 3 + a 4 + a 5 + a 6 = 20

theorem eighth_term_value (a : ℕ → ℚ) (h : arithmetic_sequence a) : 
  a 8 = 9 := by
  sorry

end eighth_term_value_l2481_248175


namespace different_course_selections_count_l2481_248100

/-- The number of courses available to choose from -/
def num_courses : ℕ := 4

/-- The number of courses each person must choose -/
def courses_per_person : ℕ := 2

/-- The number of people choosing courses -/
def num_people : ℕ := 2

/-- Represents the ways two people can choose courses differently -/
def different_course_selections : ℕ := 30

/-- Theorem stating the number of ways two people can choose courses differently -/
theorem different_course_selections_count :
  (num_courses = 4) →
  (courses_per_person = 2) →
  (num_people = 2) →
  (different_course_selections = 30) :=
by sorry

end different_course_selections_count_l2481_248100


namespace point_in_region_range_l2481_248172

theorem point_in_region_range (a : ℝ) : 
  (2 * a + 3 < 3) → a < 0 := by
  sorry

end point_in_region_range_l2481_248172


namespace janelle_gave_six_green_marbles_l2481_248196

/-- Represents the number of marbles Janelle has and gives away. -/
structure MarbleCount where
  initialGreen : Nat
  blueBags : Nat
  marblesPerBag : Nat
  giftBlue : Nat
  finalTotal : Nat

/-- Calculates the number of green marbles Janelle gave to her friend. -/
def greenMarblesGiven (m : MarbleCount) : Nat :=
  m.initialGreen - (m.finalTotal - (m.blueBags * m.marblesPerBag - m.giftBlue))

/-- Theorem stating that Janelle gave 6 green marbles to her friend. -/
theorem janelle_gave_six_green_marbles (m : MarbleCount) 
    (h1 : m.initialGreen = 26)
    (h2 : m.blueBags = 6)
    (h3 : m.marblesPerBag = 10)
    (h4 : m.giftBlue = 8)
    (h5 : m.finalTotal = 72) :
  greenMarblesGiven m = 6 := by
  sorry

#eval greenMarblesGiven { initialGreen := 26, blueBags := 6, marblesPerBag := 10, giftBlue := 8, finalTotal := 72 }

end janelle_gave_six_green_marbles_l2481_248196


namespace no_solution_for_certain_a_l2481_248153

-- Define the equation
def equation (x a : ℝ) : ℝ := 6 * abs (x - 4*a) + abs (x - a^2) + 5*x - 4*a

-- State the theorem
theorem no_solution_for_certain_a :
  ∀ a : ℝ, (a < -12 ∨ a > 0) → ¬∃ x : ℝ, equation x a = 0 := by
  sorry

end no_solution_for_certain_a_l2481_248153


namespace loan_interest_rate_l2481_248146

/-- Given a loan of $220 repaid with $242 after one year, prove the annual interest rate is 10% -/
theorem loan_interest_rate : 
  let principal : ℝ := 220
  let total_repayment : ℝ := 242
  let interest_rate : ℝ := (total_repayment - principal) / principal * 100
  interest_rate = 10 := by
  sorry

end loan_interest_rate_l2481_248146


namespace rhombus_diagonal_l2481_248199

theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h1 : area = 80)
  (h2 : d1 = 16)
  (h3 : area = (d1 * d2) / 2) :
  d2 = 10 := by
  sorry

end rhombus_diagonal_l2481_248199


namespace area_of_triangle_FNV_l2481_248118

-- Define the rectangle EFGH
structure Rectangle where
  EF : ℝ
  EH : ℝ

-- Define the trapezoid KWFG
structure Trapezoid where
  KF : ℝ
  WG : ℝ
  height : ℝ
  area : ℝ

-- Define the theorem
theorem area_of_triangle_FNV (rect : Rectangle) (trap : Trapezoid) :
  rect.EF = 15 ∧
  trap.KF = 5 ∧
  trap.WG = 5 ∧
  trap.area = 150 ∧
  trap.KF = trap.WG →
  (1 / 2 : ℝ) * (1 / 2 : ℝ) * (trap.KF + rect.EF) * rect.EH = 125 := by
  sorry


end area_of_triangle_FNV_l2481_248118


namespace prime_square_plus_twelve_mod_twelve_l2481_248163

theorem prime_square_plus_twelve_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  (p^2 + 12) % 12 = 1 := by
  sorry

end prime_square_plus_twelve_mod_twelve_l2481_248163


namespace triangle_area_l2481_248137

theorem triangle_area (base height : ℝ) (h1 : base = 4) (h2 : height = 6) :
  (base * height) / 2 = 12 := by
  sorry

end triangle_area_l2481_248137


namespace seven_balls_three_boxes_l2481_248144

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 160 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 160 := by sorry

end seven_balls_three_boxes_l2481_248144


namespace bus_stop_theorem_l2481_248193

/-- Represents the number of passengers boarding at stop i and alighting at stop j -/
def passenger_count (i j : Fin 6) : ℕ := sorry

/-- The total number of passengers on the bus between stops i and j -/
def bus_load (i j : Fin 6) : ℕ := sorry

theorem bus_stop_theorem :
  ∀ (passenger_count : Fin 6 → Fin 6 → ℕ),
  (∀ (i j : Fin 6), i < j → bus_load i j ≤ 5) →
  ∃ (A₁ B₁ A₂ B₂ : Fin 6),
    A₁ < B₁ ∧ A₂ < B₂ ∧ A₁ ≠ A₂ ∧ B₁ ≠ B₂ ∧
    passenger_count A₁ B₁ = 0 ∧ passenger_count A₂ B₂ = 0 := by
  sorry

end bus_stop_theorem_l2481_248193


namespace sin_cos_pi_12_l2481_248108

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end sin_cos_pi_12_l2481_248108


namespace value_144_is_square_iff_b_gt_4_l2481_248141

/-- The value of 144 in base b -/
def value_in_base_b (b : ℕ) : ℕ := b^2 + 4*b + 4

/-- A number is a perfect square if it has an integer square root -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m^2 = n

theorem value_144_is_square_iff_b_gt_4 (b : ℕ) :
  is_perfect_square (value_in_base_b b) ↔ b > 4 :=
sorry

end value_144_is_square_iff_b_gt_4_l2481_248141


namespace quadratic_inequality_empty_solution_l2481_248184

/-- The quadratic inequality -2 + 3x - 2x^2 > 0 has an empty solution set -/
theorem quadratic_inequality_empty_solution : 
  ∀ x : ℝ, ¬(-2 + 3*x - 2*x^2 > 0) := by
  sorry

end quadratic_inequality_empty_solution_l2481_248184


namespace remainder_2345678901_mod_102_l2481_248124

theorem remainder_2345678901_mod_102 : 2345678901 % 102 = 65 := by
  sorry

end remainder_2345678901_mod_102_l2481_248124


namespace quadratic_inequality_solution_l2481_248185

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x : ℝ, (1/3 < x ∧ x < 1/2) ↔ (a*x^2 + 5*x + c > 0)) → 
  (a = -6 ∧ c = -1) :=
by sorry

end quadratic_inequality_solution_l2481_248185


namespace lcm_150_540_l2481_248189

theorem lcm_150_540 : Nat.lcm 150 540 = 2700 := by
  sorry

end lcm_150_540_l2481_248189


namespace cosine_of_angle_l2481_248114

/-- Given two vectors a and b in ℝ², prove that the cosine of the angle between them is (3√10) / 10 -/
theorem cosine_of_angle (a b : ℝ × ℝ) (h1 : a = (3, 3)) (h2 : (2 • b) - a = (-1, 1)) :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = 3 * Real.sqrt 10 / 10 := by
  sorry

end cosine_of_angle_l2481_248114


namespace sum_of_reciprocals_equals_256_l2481_248166

/-- Given a cubic polynomial with roots p, q, r, prove that the sum of reciprocals of 
    partial fraction decomposition coefficients equals 256. -/
theorem sum_of_reciprocals_equals_256 
  (p q r : ℝ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_roots : x^3 - 27*x^2 + 98*x - 72 = (x - p) * (x - q) * (x - r)) 
  (A B C : ℝ) 
  (h_partial_fraction : ∀ (s : ℝ), s ≠ p → s ≠ q → s ≠ r → 
    1 / (s^3 - 27*s^2 + 98*s - 72) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1/A + 1/B + 1/C = 256 := by
  sorry

end sum_of_reciprocals_equals_256_l2481_248166


namespace log_order_l2481_248133

theorem log_order (a b c : ℝ) (ha : a = Real.log 3 / Real.log 4)
  (hb : b = Real.log 4 / Real.log 3) (hc : c = Real.log (4/3) / Real.log (3/4)) :
  b > a ∧ a > c :=
by sorry

end log_order_l2481_248133


namespace quadratic_root_m_l2481_248112

theorem quadratic_root_m (m : ℝ) : ((-1 : ℝ)^2 + m * (-1) - 5 = 0) → m = -4 := by
  sorry

end quadratic_root_m_l2481_248112


namespace middle_number_value_l2481_248117

theorem middle_number_value 
  (a b c d e f g h i j k : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 10.5)
  (h2 : (f + g + h + i + j + k) / 6 = 11.4)
  (h3 : (a + b + c + d + e + f + g + h + i + j + k) / 11 = 9.9)
  (h4 : a + b + c = i + j + k) :
  f = 22.5 := by
    sorry

end middle_number_value_l2481_248117


namespace gcd_count_for_product_180_l2481_248143

theorem gcd_count_for_product_180 : 
  ∃ (S : Finset ℕ), 
    (∀ a b : ℕ, a > 0 → b > 0 → Nat.gcd a b * Nat.lcm a b = 180 → 
      Nat.gcd a b ∈ S) ∧ 
    (∀ n ∈ S, ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ Nat.gcd a b * Nat.lcm a b = 180 ∧ 
      Nat.gcd a b = n) ∧
    Finset.card S = 7 :=
by sorry

end gcd_count_for_product_180_l2481_248143


namespace correct_selection_ways_l2481_248102

/-- The number of university graduates --/
def total_graduates : ℕ := 10

/-- The number of graduates to be selected --/
def selected_graduates : ℕ := 3

/-- The function that calculates the number of ways to select graduates --/
def selection_ways (total : ℕ) (select : ℕ) (at_least_AB : Bool) (exclude_C : Bool) : ℕ := sorry

/-- The theorem stating the correct number of selection ways --/
theorem correct_selection_ways : 
  selection_ways total_graduates selected_graduates true true = 49 := by sorry

end correct_selection_ways_l2481_248102


namespace certain_number_proof_l2481_248115

theorem certain_number_proof (original : ℝ) (certain : ℝ) : 
  original = 50 → (1/5 : ℝ) * original - 5 = certain → certain = 5 := by
  sorry

end certain_number_proof_l2481_248115


namespace c_investment_time_l2481_248197

/-- Represents the investment details of a partnership --/
structure Partnership where
  x : ℝ  -- A's investment amount
  m : ℝ  -- Number of months after which C invests
  annual_gain : ℝ 
  a_share : ℝ 

/-- Calculates the investment share of partner A --/
def a_investment_share (p : Partnership) : ℝ := p.x * 12

/-- Calculates the investment share of partner B --/
def b_investment_share (p : Partnership) : ℝ := 2 * p.x * 6

/-- Calculates the investment share of partner C --/
def c_investment_share (p : Partnership) : ℝ := 3 * p.x * (12 - p.m)

/-- Calculates the total investment share --/
def total_investment_share (p : Partnership) : ℝ :=
  a_investment_share p + b_investment_share p + c_investment_share p

/-- The main theorem stating that C invests after 3 months --/
theorem c_investment_time (p : Partnership) 
  (h1 : p.annual_gain = 18300)
  (h2 : p.a_share = 6100)
  (h3 : a_investment_share p / total_investment_share p = p.a_share / p.annual_gain) :
  p.m = 3 := by
  sorry

end c_investment_time_l2481_248197


namespace square_and_cube_difference_l2481_248171

theorem square_and_cube_difference (a b : ℝ) 
  (sum_eq : a + b = 8) 
  (diff_eq : a - b = 4) : 
  a^2 - b^2 = 32 ∧ a^3 - b^3 = 208 := by
  sorry

end square_and_cube_difference_l2481_248171


namespace polynomial_factorization_l2481_248164

theorem polynomial_factorization (y : ℝ) : 
  y^4 - 4*y^2 + 4 + 49*y^2 = (y^2 + 1) * (y^2 + 13) := by
  sorry

end polynomial_factorization_l2481_248164


namespace sum_of_three_numbers_l2481_248149

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 99) 
  (h2 : a*b + b*c + a*c = 131) : 
  a + b + c = 19 := by
  sorry

end sum_of_three_numbers_l2481_248149


namespace sequence_inequality_l2481_248181

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_pos : ∀ k, 0 ≤ k ∧ k ≤ n → 0 < a k)
  (h_eq : ∀ k, 1 ≤ k ∧ k < n → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) - a (k+1)) :
  a n < 1 / (n - 1) := by
sorry

end sequence_inequality_l2481_248181


namespace sweater_cost_l2481_248170

theorem sweater_cost (initial_amount : ℕ) (tshirt_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 91 → 
  tshirt_cost = 6 → 
  shoes_cost = 11 → 
  remaining_amount = 50 → 
  initial_amount - remaining_amount - tshirt_cost - shoes_cost = 24 := by
sorry

end sweater_cost_l2481_248170


namespace tan_105_degrees_l2481_248147

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l2481_248147


namespace tangent_equation_solution_l2481_248105

open Real

theorem tangent_equation_solution :
  ∃! y : ℝ, 0 ≤ y ∧ y < 2 * π ∧
  tan (150 * π / 180 - y) = (sin (150 * π / 180) - sin y) / (cos (150 * π / 180) - cos y) →
  y = 0 ∨ y = 2 * π := by
sorry

end tangent_equation_solution_l2481_248105


namespace plane_distance_ratio_l2481_248135

/-- Proves the ratio of plane distance to total distance -/
theorem plane_distance_ratio (total : ℝ) (bus : ℝ) (train : ℝ) (plane : ℝ) :
  total = 900 →
  train = (2/3) * bus →
  bus = 360 →
  plane = total - (bus + train) →
  plane / total = 1/3 := by
sorry

end plane_distance_ratio_l2481_248135


namespace a_plus_b_value_l2481_248101

theorem a_plus_b_value (a b : ℝ) (h1 : a^2 = 16) (h2 : |b| = 3) (h3 : a + b < 0) :
  a + b = -1 ∨ a + b = -7 :=
by sorry

end a_plus_b_value_l2481_248101


namespace intersection_points_min_distance_l2481_248155

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x + y + 1 = 0
def C₃ (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Theorem for intersection points
theorem intersection_points :
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y ↔ (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = -1)) :=
sorry

-- Theorem for minimum distance
theorem min_distance :
  (∃ d : ℝ, d = Real.sqrt 2 - 1 ∧
    ∀ x₁ y₁ x₂ y₂ : ℝ, C₂ x₁ y₁ → C₃ x₂ y₂ →
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ d) :=
sorry

end intersection_points_min_distance_l2481_248155


namespace employee_earnings_l2481_248154

/-- Calculates the total earnings for an employee based on their work schedule and pay rates. -/
theorem employee_earnings (regular_rate : ℝ) (overtime_multiplier : ℝ) (regular_hours : ℝ) 
  (first_three_days_hours : ℝ) (last_two_days_multiplier : ℝ) : 
  regular_rate = 30 →
  overtime_multiplier = 1.5 →
  regular_hours = 40 →
  first_three_days_hours = 6 →
  last_two_days_multiplier = 2 →
  let overtime_rate := regular_rate * overtime_multiplier
  let last_two_days_hours := first_three_days_hours * last_two_days_multiplier
  let total_hours := first_three_days_hours * 3 + last_two_days_hours * 2
  let overtime_hours := max (total_hours - regular_hours) 0
  let regular_pay := min total_hours regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  regular_pay + overtime_pay = 1290 := by
sorry

end employee_earnings_l2481_248154


namespace amount_saved_calculation_l2481_248113

def initial_amount : ℕ := 6000
def pen_cost : ℕ := 3200
def eraser_cost : ℕ := 1000
def candy_cost : ℕ := 500

theorem amount_saved_calculation :
  initial_amount - (pen_cost + eraser_cost + candy_cost) = 1300 := by
  sorry

end amount_saved_calculation_l2481_248113


namespace thirtieth_number_in_base12_l2481_248145

/-- Converts a decimal number to its base 12 representation --/
def toBase12 (n : ℕ) : List ℕ :=
  if n < 12 then [n]
  else (n % 12) :: toBase12 (n / 12)

/-- Interprets a list of digits as a number in base 12 --/
def fromBase12 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 12 * acc) 0

theorem thirtieth_number_in_base12 :
  toBase12 30 = [6, 2] ∧ fromBase12 [6, 2] = 30 := by
  sorry

end thirtieth_number_in_base12_l2481_248145


namespace parabola_and_line_theorem_l2481_248103

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem parabola_and_line_theorem (para : Parabola) (l : Line) (P A B : Point) :
  (para.p * 2 = 4) →  -- Distance between focus and directrix is 4
  (P.x = 1 ∧ P.y = -1) →  -- P is at (1, -1)
  (A.y ^ 2 = 2 * para.p * A.x ∧ B.y ^ 2 = 2 * para.p * B.x) →  -- A and B are on the parabola
  (l.a * A.x + l.b * A.y + l.c = 0 ∧ l.a * B.x + l.b * B.y + l.c = 0) →  -- A and B are on the line
  (P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2) →  -- P is midpoint of AB
  (∀ x y, y ^ 2 = 8 * x ↔ y ^ 2 = 2 * para.p * x) ∧  -- Parabola equation is y² = 8x
  (∀ x y, 4 * x + y - 3 = 0 ↔ l.a * x + l.b * y + l.c = 0)  -- Line equation is 4x + y - 3 = 0
:= by sorry

end parabola_and_line_theorem_l2481_248103


namespace binary_11011_equals_27_l2481_248177

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end binary_11011_equals_27_l2481_248177


namespace distribute_6_4_l2481_248139

/-- The number of ways to distribute n identical objects among k classes,
    with each class receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- Theorem stating that there are 10 ways to distribute 6 spots among 4 classes,
    with each class receiving at least one spot. -/
theorem distribute_6_4 : distribute 6 4 = 10 := by sorry

end distribute_6_4_l2481_248139


namespace rectangle_area_l2481_248183

/-- Given a rectangle with length four times its width and perimeter 200 cm, its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 →
  l * w = 1600 := by
sorry

end rectangle_area_l2481_248183
