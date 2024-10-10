import Mathlib

namespace binary_110011_equals_51_l12_1207

/-- Converts a binary digit to its decimal value -/
def binaryToDecimal (digit : Nat) (position : Nat) : Nat :=
  digit * (2 ^ position)

/-- Represents the binary number 110011 -/
def binaryNumber : List Nat := [1, 1, 0, 0, 1, 1]

/-- Converts a list of binary digits to its decimal representation -/
def listBinaryToDecimal (bits : List Nat) : Nat :=
  (List.zipWith binaryToDecimal bits (List.range bits.length)).sum

theorem binary_110011_equals_51 : listBinaryToDecimal binaryNumber = 51 := by
  sorry

end binary_110011_equals_51_l12_1207


namespace beverage_mix_ratio_l12_1251

theorem beverage_mix_ratio : 
  ∀ (x y : ℝ), 
  x > 0 → y > 0 →
  (5 * x + 4 * y = 5.5 * x + 3.6 * y) →
  (x / y = 4 / 5) := by
sorry

end beverage_mix_ratio_l12_1251


namespace product_of_six_consecutive_divisible_by_ten_l12_1262

theorem product_of_six_consecutive_divisible_by_ten (n : ℕ+) :
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5)) := by
  sorry

end product_of_six_consecutive_divisible_by_ten_l12_1262


namespace find_a_l12_1273

def A (a : ℝ) : Set ℝ := {4, a^2}
def B (a : ℝ) : Set ℝ := {a-6, a+1, 9}

theorem find_a : ∀ a : ℝ, A a ∩ B a = {9} → a = -3 := by
  sorry

end find_a_l12_1273


namespace prob_four_red_cards_standard_deck_l12_1233

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- Probability of drawing n red cards in a row from a deck -/
def prob_n_red_cards (d : Deck) (n : ℕ) : ℚ :=
  sorry

theorem prob_four_red_cards_standard_deck :
  let standard_deck : Deck := ⟨52, 26, 26⟩
  prob_n_red_cards standard_deck 4 = 276 / 9801 := by
  sorry

end prob_four_red_cards_standard_deck_l12_1233


namespace three_heads_probability_l12_1205

def prob_heads : ℚ := 1/2

theorem three_heads_probability :
  let prob_three_heads := prob_heads * prob_heads * prob_heads
  prob_three_heads = 1/8 := by sorry

end three_heads_probability_l12_1205


namespace distance_is_27_l12_1268

/-- The distance between two locations A and B, where two people walk towards each other, 
    meet, continue to their destinations, turn back, and meet again. -/
def distance_between_locations : ℝ :=
  let first_meeting_distance_from_A : ℝ := 10
  let second_meeting_distance_from_B : ℝ := 3
  first_meeting_distance_from_A + (2 * first_meeting_distance_from_A - second_meeting_distance_from_B)

/-- Theorem stating that the distance between locations A and B is 27 kilometers. -/
theorem distance_is_27 : distance_between_locations = 27 := by
  sorry

end distance_is_27_l12_1268


namespace square_of_binomial_c_value_l12_1293

theorem square_of_binomial_c_value (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + c = (a * x + b)^2) → c = 25 := by
  sorry

end square_of_binomial_c_value_l12_1293


namespace exam_score_calculation_l12_1237

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) :
  total_questions = 50 →
  correct_answers = 36 →
  marks_per_correct = 4 →
  marks_lost_per_wrong = 1 →
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong) = 130 := by
  sorry

end exam_score_calculation_l12_1237


namespace range_of_a_l12_1270

-- Define the sets A and B
def A (a : ℝ) := {x : ℝ | a < x ∧ x < a + 1}
def B := {x : ℝ | 3 + 2*x - x^2 > 0}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, A a ∩ B = A a) ↔ (∀ a : ℝ, -1 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l12_1270


namespace arc_length_of_sector_l12_1203

/-- Given a circular sector with radius 5 cm and area 11.25 cm², 
    prove that the length of the arc is 4.5 cm. -/
theorem arc_length_of_sector (r : ℝ) (area : ℝ) (arc_length : ℝ) : 
  r = 5 → 
  area = 11.25 → 
  arc_length = r * (2 * area / (r * r)) → 
  arc_length = 4.5 := by
sorry

end arc_length_of_sector_l12_1203


namespace rectangle_original_length_l12_1218

theorem rectangle_original_length :
  ∀ (original_length : ℝ),
    (original_length * 10 = 25 * 7.2) →
    original_length = 18 :=
by sorry

end rectangle_original_length_l12_1218


namespace cube_split_2017_l12_1263

/-- The function that gives the first odd number in the split for m^3 -/
def first_split (m : ℕ) : ℕ := 2 * m * (m - 1) + 1

/-- The predicate that checks if a number is in the split for m^3 -/
def in_split (n m : ℕ) : Prop :=
  ∃ k, 0 < k ∧ k ≤ m^3 ∧ n = first_split m + 2 * (k - 1)

theorem cube_split_2017 :
  ∀ m : ℕ, m > 1 → (in_split 2017 m ↔ m = 47) :=
sorry

end cube_split_2017_l12_1263


namespace fraction_inequality_conditions_l12_1235

theorem fraction_inequality_conditions (a b : ℝ) :
  (∀ x : ℝ, |((x^2 + a*x + b) / (x^2 + 2*x + 2))| < 1) ↔ (a = 2 ∧ 0 < b ∧ b < 2) := by
  sorry

end fraction_inequality_conditions_l12_1235


namespace range_of_sum_l12_1295

theorem range_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 1) :
  a + b ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end range_of_sum_l12_1295


namespace fractional_parts_inequality_l12_1296

theorem fractional_parts_inequality (q : ℕ+) (hq : ¬ ∃ (m : ℕ), q = m^3) :
  ∃ (c : ℝ), c > 0 ∧
  ∀ (n : ℕ+), 
    (n : ℝ) * q.val ^ (1/3 : ℝ) - ⌊(n : ℝ) * q.val ^ (1/3 : ℝ)⌋ +
    (n : ℝ) * q.val ^ (2/3 : ℝ) - ⌊(n : ℝ) * q.val ^ (2/3 : ℝ)⌋ ≥
    c * (n : ℝ) ^ (-1/2 : ℝ) :=
by sorry

end fractional_parts_inequality_l12_1296


namespace xyz_ratio_l12_1222

theorem xyz_ratio (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (diff_xy : x ≠ y) (diff_xz : x ≠ z) (diff_yz : y ≠ z)
  (eq1 : y / (x - z) = (x + y) / z)
  (eq2 : (x + y) / z = x / y) : 
  x / y = 2 := by sorry

end xyz_ratio_l12_1222


namespace largest_divisible_n_l12_1216

theorem largest_divisible_n : ∀ n : ℕ+, n^3 + 144 ∣ n + 12 → n ≤ 780 :=
sorry

end largest_divisible_n_l12_1216


namespace students_not_enrolled_l12_1259

theorem students_not_enrolled (total : ℕ) (football : ℕ) (swimming : ℕ) (both : ℕ) :
  total = 100 →
  football = 37 →
  swimming = 40 →
  both = 15 →
  total - (football + swimming - both) = 38 := by
sorry

end students_not_enrolled_l12_1259


namespace triangle_area_l12_1241

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if A = 2π/3, a = 7, and b = 3, then the area of the triangle S_ABC = 15√3/4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A = 2 * Real.pi / 3 →
  a = 7 →
  b = 3 →
  (∃ (S_ABC : ℝ), S_ABC = (15 * Real.sqrt 3) / 4 ∧ S_ABC = (1/2) * b * c * Real.sin A) :=
by sorry

end triangle_area_l12_1241


namespace factorization_problem_1_factorization_problem_2_l12_1224

-- Problem 1
theorem factorization_problem_1 (y : ℝ) : y^3 - y^2 + (1/4)*y = y*(y - 1/2)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (m n : ℝ) : m^4 - n^4 = (m - n)*(m + n)*(m^2 + n^2) := by sorry

end factorization_problem_1_factorization_problem_2_l12_1224


namespace min_value_problem_l12_1298

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  ∃ (m : ℝ), m = (1 : ℝ)/5184 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 
    1/x + 1/y + 1/z = 9 → x^4 * y^3 * z^2 ≥ m ∧
    a^4 * b^3 * c^2 = m :=
sorry

end min_value_problem_l12_1298


namespace confucius_travel_equation_l12_1247

/-- Represents the scenario of Confucius and his students traveling to a school -/
def confucius_travel (x : ℝ) : Prop :=
  let student_speed := x
  let cart_speed := 1.5 * x
  let distance := 30
  let student_time := distance / student_speed
  let confucius_time := distance / cart_speed + 1
  student_time = confucius_time

/-- Theorem stating the equation that holds true for the travel scenario -/
theorem confucius_travel_equation (x : ℝ) (hx : x > 0) :
  confucius_travel x ↔ 30 / x = 30 / (1.5 * x) + 1 :=
sorry

end confucius_travel_equation_l12_1247


namespace smallest_n_is_correct_l12_1294

/-- The smallest positive integer n such that all roots of z^5 - z^3 + z = 0 are n^th roots of unity -/
def smallest_n : ℕ := 12

/-- The complex polynomial z^5 - z^3 + z -/
def f (z : ℂ) : ℂ := z^5 - z^3 + z

theorem smallest_n_is_correct :
  ∀ z : ℂ, f z = 0 → ∃ k : ℕ, z^smallest_n = 1 ∧
  ∀ m : ℕ, (∀ w : ℂ, f w = 0 → w^m = 1) → smallest_n ≤ m :=
by sorry

end smallest_n_is_correct_l12_1294


namespace common_remainder_proof_l12_1280

theorem common_remainder_proof : 
  let n := 1398 - 7
  (n % 7 = 5) ∧ (n % 9 = 5) ∧ (n % 11 = 5) :=
by sorry

end common_remainder_proof_l12_1280


namespace line_direction_vector_l12_1226

/-- Prove that for a line passing through (1, -3) and (5, 3), 
    if its direction vector is of the form (3, c), then c = 9/2 -/
theorem line_direction_vector (c : ℚ) : 
  (∃ (t : ℚ), (1 + 3*t = 5) ∧ (-3 + c*t = 3)) → c = 9/2 := by
  sorry

end line_direction_vector_l12_1226


namespace regular_octagon_extended_sides_angle_l12_1220

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Point Q where extended sides BC and DE meet -/
def Q (octagon : RegularOctagon) : ℝ × ℝ := sorry

/-- Angle measure in degrees -/
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem regular_octagon_extended_sides_angle 
  (octagon : RegularOctagon) : 
  angle_measure (octagon.vertices 3) (Q octagon) (octagon.vertices 4) = 90 := by
  sorry

end regular_octagon_extended_sides_angle_l12_1220


namespace cafe_chairs_count_l12_1250

/-- Calculates the total number of chairs in a cafe given the number of indoor and outdoor tables
    and the number of chairs per table type. -/
def total_chairs (indoor_tables : ℕ) (outdoor_tables : ℕ) (chairs_per_indoor_table : ℕ) (chairs_per_outdoor_table : ℕ) : ℕ :=
  indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table

/-- Theorem stating that the total number of chairs in the cafe is 123. -/
theorem cafe_chairs_count :
  total_chairs 9 11 10 3 = 123 := by
  sorry

#eval total_chairs 9 11 10 3

end cafe_chairs_count_l12_1250


namespace cube_sum_reciprocal_l12_1275

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) : 
  a^3 + 1/a^3 = 2 * Real.sqrt 5 := by
  sorry

end cube_sum_reciprocal_l12_1275


namespace complex_sum_magnitude_l12_1217

theorem complex_sum_magnitude (a b c : ℂ) :
  Complex.abs a = 1 →
  Complex.abs b = 1 →
  Complex.abs c = 1 →
  a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = -3 →
  Complex.abs (a + b + c) = 3 := by
sorry

end complex_sum_magnitude_l12_1217


namespace sum_of_coefficients_l12_1236

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 729 * x^3 + 8 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 78 := by
sorry

end sum_of_coefficients_l12_1236


namespace subtraction_property_l12_1214

theorem subtraction_property : 12.56 - (5.56 - 2.63) = 12.56 - 5.56 + 2.63 := by
  sorry

end subtraction_property_l12_1214


namespace intersection_of_M_and_N_l12_1271

def M : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def N : Set ℝ := {x | -4 < x ∧ x ≤ 2}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by sorry

end intersection_of_M_and_N_l12_1271


namespace xy_yz_zx_bounds_l12_1297

theorem xy_yz_zx_bounds (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2 + 1) : 
  ∃ (N n : ℝ), (∀ t : ℝ, t = x*y + y*z + z*x → t ≤ N ∧ n ≤ t) ∧ 11 < N + 6*n ∧ N + 6*n < 12 := by
  sorry

end xy_yz_zx_bounds_l12_1297


namespace prism_with_five_faces_has_nine_edges_l12_1204

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  faces : ℕ

/-- The number of edges in a prism given its number of faces. -/
def num_edges (p : Prism) : ℕ :=
  if p.faces = 5 then 9 else 0  -- We only define it for the case of 5 faces

theorem prism_with_five_faces_has_nine_edges (p : Prism) (h : p.faces = 5) : 
  num_edges p = 9 := by
  sorry

#check prism_with_five_faces_has_nine_edges

end prism_with_five_faces_has_nine_edges_l12_1204


namespace sine_cosine_inequality_l12_1258

-- Define a periodic function with period 2
def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f x

-- Define symmetry around x = 2
def symmetric_around_two (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) = f (2 - x)

-- Define decreasing on an interval
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Define acute angle
def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

-- Theorem statement
theorem sine_cosine_inequality
  (f : ℝ → ℝ)
  (h_periodic : periodic_function f)
  (h_symmetric : symmetric_around_two f)
  (h_decreasing : decreasing_on f (-3) (-2))
  (A B : ℝ)
  (h_acute_A : acute_angle A)
  (h_acute_B : acute_angle B)
  (h_triangle : A + B ≤ Real.pi / 2) :
  f (Real.sin A) > f (Real.cos B) :=
sorry

end sine_cosine_inequality_l12_1258


namespace axis_of_symmetry_l12_1281

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 3) * (x + 1)

-- Theorem statement
theorem axis_of_symmetry :
  ∃ (a b c : ℝ), (∀ x, f x = a * x^2 + b * x + c) ∧ 
  (a ≠ 0) ∧
  (- b / (2 * a) = -1) := by
  sorry

end axis_of_symmetry_l12_1281


namespace min_trips_for_28_containers_l12_1202

/-- The minimum number of trips required to transport a given number of containers -/
def min_trips (total_containers : ℕ) (max_per_trip : ℕ) : ℕ :=
  (total_containers + max_per_trip - 1) / max_per_trip

theorem min_trips_for_28_containers :
  min_trips 28 5 = 6 := by
  sorry

#eval min_trips 28 5

end min_trips_for_28_containers_l12_1202


namespace triangle_area_l12_1285

/-- Given a triangle with sides in ratio 5:12:13 and perimeter 300, prove its area is 3000 -/
theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (5 * (300 / 30), 12 * (300 / 30), 13 * (300 / 30))) 
  (h_perimeter : a + b + c = 300) : 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 3000 := by
sorry

end triangle_area_l12_1285


namespace axisymmetric_triangle_is_isosceles_l12_1227

/-- A triangle is axisymmetric if it has an axis of symmetry. -/
def IsAxisymmetric (t : Triangle) : Prop := sorry

/-- A triangle is isosceles if it has at least two sides of equal length. -/
def IsIsosceles (t : Triangle) : Prop := sorry

/-- If a triangle is axisymmetric, then it is isosceles. -/
theorem axisymmetric_triangle_is_isosceles (t : Triangle) :
  IsAxisymmetric t → IsIsosceles t := by
  sorry

end axisymmetric_triangle_is_isosceles_l12_1227


namespace sum_and_double_l12_1255

theorem sum_and_double : 2 * (1324 + 4231 + 3124 + 2413) = 22184 := by
  sorry

end sum_and_double_l12_1255


namespace pulley_centers_distance_l12_1239

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (hr₁ : r₁ = 10) 
  (hr₂ : r₂ = 6) 
  (hd : contact_distance = 30) : 
  ∃ (center_distance : ℝ), center_distance = 2 * Real.sqrt 229 :=
by sorry

end pulley_centers_distance_l12_1239


namespace factorization_1_l12_1200

theorem factorization_1 (m n : ℤ) : 3 * m * n - 6 * m^2 * n^2 = 3 * m * n * (1 - 2 * m * n) :=
by sorry

end factorization_1_l12_1200


namespace live_streaming_fee_strategy2_revenue_total_profit_l12_1264

-- Define the problem parameters
def total_items : ℕ := 600
def strategy1_items : ℕ := 200
def strategy2_items : ℕ := 400
def strategy2_phase1_items : ℕ := 100
def strategy2_phase2_items : ℕ := 300

-- Define the strategies
def strategy1_price (m : ℝ) : ℝ := 2 * m - 5
def strategy1_fee_rate : ℝ := 0.01
def strategy2_base_price (m : ℝ) : ℝ := 2.5 * m
def strategy2_discount1 : ℝ := 0.8
def strategy2_discount2 : ℝ := 0.8

-- Theorem statements
theorem live_streaming_fee (m : ℝ) :
  strategy1_items * strategy1_price m * strategy1_fee_rate = 4 * m - 10 := by sorry

theorem strategy2_revenue (m : ℝ) :
  strategy2_phase1_items * strategy2_base_price m * strategy2_discount1 +
  strategy2_phase2_items * strategy2_base_price m * strategy2_discount1 * strategy2_discount2 = 680 * m := by sorry

theorem total_profit (m : ℝ) :
  strategy1_items * strategy1_price m +
  (strategy2_phase1_items * strategy2_base_price m * strategy2_discount1 +
   strategy2_phase2_items * strategy2_base_price m * strategy2_discount1 * strategy2_discount2) -
  (strategy1_items * strategy1_price m * strategy1_fee_rate) -
  (total_items * m) = 476 * m - 990 := by sorry

end live_streaming_fee_strategy2_revenue_total_profit_l12_1264


namespace statement_1_statement_2_statement_3_l12_1288

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (5*a + 1) * x + 4*a + 4

-- Statement 1
theorem statement_1 (a : ℝ) (h : a < -1) : f a 0 < 0 := by sorry

-- Statement 2
theorem statement_2 (a : ℝ) (h : a > 0) : 
  ∃ (y : ℝ), y = 3 ∧ ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → f a x ≤ y := by sorry

-- Statement 3
theorem statement_3 (a : ℝ) (h : a < 0) : 
  f a 2 > f a 3 ∧ f a 3 > f a 4 := by sorry

end statement_1_statement_2_statement_3_l12_1288


namespace factor_expression_l12_1223

theorem factor_expression (x : ℝ) : 6 * x^3 - 54 * x = 6 * x * (x + 3) * (x - 3) := by
  sorry

end factor_expression_l12_1223


namespace no_exact_change_for_57_can_make_change_for_15_l12_1249

/-- Represents the available Tyro bill denominations -/
def tyro_bills : List ℕ := [35, 80]

/-- Checks if a given amount can be represented as a sum of available Tyro bills -/
def can_make_exact_change (amount : ℕ) : Prop :=
  ∃ (a b : ℕ), a * 35 + b * 80 = amount

/-- Checks if a given amount can be represented as a difference of sums of available Tyro bills -/
def can_make_change_with_subtraction (amount : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a * 35 + b * 80 - (c * 35 + d * 80) = amount

/-- Theorem stating that exact change cannot be made for 57 Tyros -/
theorem no_exact_change_for_57 : ¬ can_make_exact_change 57 := by sorry

/-- Theorem stating that change can be made for 15 Tyros using subtraction -/
theorem can_make_change_for_15 : can_make_change_with_subtraction 15 := by sorry

end no_exact_change_for_57_can_make_change_for_15_l12_1249


namespace increasing_f_implies_t_geq_5_l12_1269

/-- A cubic function with a parameter t -/
def f (t : ℝ) (x : ℝ) : ℝ := -x^3 + x^2 + t*x + t

/-- The derivative of f with respect to x -/
def f' (t : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*x + t

theorem increasing_f_implies_t_geq_5 :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, Monotone (f t)) →
  t ≥ 5 := by sorry

end increasing_f_implies_t_geq_5_l12_1269


namespace coin_difference_l12_1287

/-- Represents the number of coins of each denomination in Tom's collection -/
structure CoinCollection where
  fiveCent : ℚ
  tenCent : ℚ
  twentyCent : ℚ

/-- Conditions for Tom's coin collection -/
def validCollection (c : CoinCollection) : Prop :=
  c.fiveCent + c.tenCent + c.twentyCent = 30 ∧
  c.tenCent = 2 * c.fiveCent ∧
  5 * c.fiveCent + 10 * c.tenCent + 20 * c.twentyCent = 340

/-- The main theorem to prove -/
theorem coin_difference (c : CoinCollection) 
  (h : validCollection c) : c.twentyCent - c.fiveCent = 2/7 := by
  sorry

end coin_difference_l12_1287


namespace eesha_late_arrival_l12_1225

/-- Eesha's commute problem -/
theorem eesha_late_arrival (usual_time : ℕ) (late_start : ℕ) (speed_reduction : ℚ) : 
  usual_time = 60 → late_start = 30 → speed_reduction = 1/4 →
  (usual_time : ℚ) / (1 - speed_reduction) + late_start - usual_time = 15 := by
  sorry

#check eesha_late_arrival

end eesha_late_arrival_l12_1225


namespace dereks_savings_l12_1289

theorem dereks_savings (n : ℕ) (a : ℝ) (r : ℝ) : 
  n = 12 → a = 2 → r = 2 → 
  a * (1 - r^n) / (1 - r) = 8190 := by
  sorry

end dereks_savings_l12_1289


namespace f_increasing_on_interval_l12_1215

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2*x) / Real.log (1/2)

theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Iio (1/2)) := by sorry

end f_increasing_on_interval_l12_1215


namespace mean_of_three_numbers_l12_1278

theorem mean_of_three_numbers (n : ℤ) : 
  n = (17 + 23 + 2*n) / 3 → n = 40 := by
sorry

end mean_of_three_numbers_l12_1278


namespace valid_numbers_count_l12_1277

/-- A function that generates all valid five-digit numbers satisfying the conditions -/
def validNumbers : List Nat := sorry

/-- A predicate that checks if a number satisfies all conditions -/
def isValid (n : Nat) : Bool := sorry

/-- The main theorem stating that there are exactly 20 valid numbers -/
theorem valid_numbers_count : (validNumbers.filter isValid).length = 20 := by sorry

end valid_numbers_count_l12_1277


namespace smallest_m_for_integral_solutions_l12_1234

theorem smallest_m_for_integral_solutions : 
  ∃ (m : ℕ), m > 0 ∧ 
  (∃ (x : ℤ), 12 * x^2 - m * x + 360 = 0) ∧
  (∀ (k : ℕ), 0 < k ∧ k < m → ¬∃ (x : ℤ), 12 * x^2 - k * x + 360 = 0) ∧
  m = 132 := by
sorry

end smallest_m_for_integral_solutions_l12_1234


namespace region_is_rectangle_l12_1299

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The region defined by the given inequalities -/
def Region : Set Point2D :=
  {p : Point2D | -1 ≤ p.x ∧ p.x ≤ 1 ∧ 2 ≤ p.y ∧ p.y ≤ 4}

/-- Definition of a rectangle in 2D -/
def IsRectangle (S : Set Point2D) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), x1 < x2 ∧ y1 < y2 ∧
    S = {p : Point2D | x1 ≤ p.x ∧ p.x ≤ x2 ∧ y1 ≤ p.y ∧ p.y ≤ y2}

/-- Theorem: The defined region is a rectangle -/
theorem region_is_rectangle : IsRectangle Region := by
  sorry

end region_is_rectangle_l12_1299


namespace geometric_sequence_product_l12_1219

/-- Given that -1, a, b, c, -4 form a geometric sequence, prove that a * b * c = -8 -/
theorem geometric_sequence_product (a b c : ℝ) 
  (h : ∃ (r : ℝ), a = -1 * r ∧ b = a * r ∧ c = b * r ∧ -4 = c * r) : 
  a * b * c = -8 := by
  sorry

end geometric_sequence_product_l12_1219


namespace school_population_l12_1245

theorem school_population (total_sample : ℕ) (first_year_sample : ℕ) (third_year_sample : ℕ) (second_year_total : ℕ) :
  total_sample = 45 →
  first_year_sample = 20 →
  third_year_sample = 10 →
  second_year_total = 300 →
  ∃ (total_students : ℕ), total_students = 900 :=
by
  sorry

end school_population_l12_1245


namespace number_of_pens_purchased_l12_1206

/-- Given the total cost of pens and pencils, the number of pencils, and the prices of pens and pencils,
    prove that the number of pens purchased is 30. -/
theorem number_of_pens_purchased 
  (total_cost : ℝ) 
  (num_pencils : ℕ) 
  (price_pencil : ℝ) 
  (price_pen : ℝ) 
  (h1 : total_cost = 510)
  (h2 : num_pencils = 75)
  (h3 : price_pencil = 2)
  (h4 : price_pen = 12) :
  (total_cost - num_pencils * price_pencil) / price_pen = 30 := by
  sorry


end number_of_pens_purchased_l12_1206


namespace lauri_eating_days_l12_1231

/-- The number of days Lauri ate apples -/
def lauriDays : ℕ := 15

/-- The fraction of an apple Simone ate per day -/
def simonePerDay : ℚ := 1/2

/-- The number of days Simone ate apples -/
def simoneDays : ℕ := 16

/-- The fraction of an apple Lauri ate per day -/
def lauriPerDay : ℚ := 1/3

/-- The total number of apples both girls ate -/
def totalApples : ℕ := 13

theorem lauri_eating_days : 
  simonePerDay * simoneDays + lauriPerDay * lauriDays = totalApples := by
  sorry

end lauri_eating_days_l12_1231


namespace deck_size_l12_1254

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1 / 5 →
  (r : ℚ) / (r + b + 6) = 1 / 7 →
  r + b = 15 := by
  sorry

end deck_size_l12_1254


namespace pin_combinations_l12_1292

/-- The number of unique permutations of a multiset with elements {5, 3, 3, 7} -/
def pinPermutations : ℕ :=
  Nat.factorial 4 / (Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 1)

theorem pin_combinations : pinPermutations = 12 := by
  sorry

end pin_combinations_l12_1292


namespace smallest_whole_number_with_odd_factors_l12_1260

theorem smallest_whole_number_with_odd_factors : ∃ n : ℕ, 
  n > 100 ∧ 
  (∀ m : ℕ, m > 100 → (∃ k : ℕ, k * k = m) → m ≥ n) ∧
  (∃ k : ℕ, k * k = n) :=
by sorry

end smallest_whole_number_with_odd_factors_l12_1260


namespace inequality_problem_l12_1246

theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * y > x * z) ∧
  (∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * (y - z) > 0) ∧
  (∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * z * (z - x) < 0) ∧
  ¬(∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * y^2 < z * y^2) :=
by sorry

end inequality_problem_l12_1246


namespace roots_equation_value_l12_1248

theorem roots_equation_value (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 4 = 0 → 
  x₂^2 - 3*x₂ - 4 = 0 → 
  x₁^2 - 4*x₁ - x₂ + 2*x₁*x₂ = -7 := by
  sorry

end roots_equation_value_l12_1248


namespace can_empty_table_l12_1256

/-- Represents a 2x2 table of natural numbers -/
def Table := Fin 2 → Fin 2 → ℕ

/-- Represents a move on the table -/
inductive Move
| RemoveRow (row : Fin 2) : Move
| DoubleColumn (col : Fin 2) : Move

/-- Applies a move to a table -/
def applyMove (t : Table) (m : Move) : Table :=
  match m with
  | Move.RemoveRow row => fun i j => if i = row ∧ t i j > 0 then t i j - 1 else t i j
  | Move.DoubleColumn col => fun i j => if j = col then 2 * t i j else t i j

/-- Checks if a table is empty (all cells are zero) -/
def isEmptyTable (t : Table) : Prop :=
  ∀ i j, t i j = 0

/-- The main theorem: any non-empty table can be emptied -/
theorem can_empty_table (t : Table) (h : ∀ i j, t i j > 0) :
  ∃ (moves : List Move), isEmptyTable (moves.foldl applyMove t) :=
sorry

end can_empty_table_l12_1256


namespace nine_times_nines_digit_sum_l12_1229

/-- Represents a number consisting of n nines -/
def nines (n : ℕ) : ℕ := (10^n - 1)

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Theorem stating that the product of 9 and a number with 120 nines has a digit sum of 1080 -/
theorem nine_times_nines_digit_sum :
  sumOfDigits (9 * nines 120) = 1080 := by sorry

end nine_times_nines_digit_sum_l12_1229


namespace assignment_ways_l12_1240

def total_students : ℕ := 30
def selected_students : ℕ := 10
def group_size : ℕ := 5

def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem assignment_ways :
  (combination total_students selected_students * combination selected_students group_size) / 2 =
  (combination total_students selected_students * combination selected_students group_size) / 2 := by
  sorry

end assignment_ways_l12_1240


namespace waiter_problem_l12_1284

/-- Calculates the number of men at tables given the number of tables, women, and average customers per table. -/
def number_of_men (tables : Float) (women : Float) (avg_customers : Float) : Float :=
  tables * avg_customers - women

/-- Theorem stating that given 9.0 tables, 7.0 women, and an average of 1.111111111 customers per table, the number of men at the tables is 3.0. -/
theorem waiter_problem :
  number_of_men 9.0 7.0 1.111111111 = 3.0 := by
  sorry

end waiter_problem_l12_1284


namespace parallel_vectors_x_value_l12_1201

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = 1/2 := by
sorry

end parallel_vectors_x_value_l12_1201


namespace P_in_first_quadrant_l12_1242

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The point P with coordinates (2,1) -/
def P : Point :=
  { x := 2, y := 1 }

/-- Theorem stating that P lies in the first quadrant -/
theorem P_in_first_quadrant : isInFirstQuadrant P := by
  sorry

end P_in_first_quadrant_l12_1242


namespace money_sharing_l12_1208

theorem money_sharing (emma finn grace total : ℕ) : 
  emma = 45 →
  emma + finn + grace = total →
  3 * finn = 4 * emma →
  3 * grace = 5 * emma →
  total = 180 := by
sorry

end money_sharing_l12_1208


namespace conic_section_foci_l12_1283

-- Define the polar equation of the conic section
def polar_equation (ρ θ : ℝ) : Prop := ρ = 16 / (5 - 3 * Real.cos θ)

-- Define the focus coordinates
def focus1 : ℝ × ℝ := (0, 0)
def focus2 : ℝ × ℝ := (6, 0)

-- Theorem statement
theorem conic_section_foci (ρ θ : ℝ) :
  polar_equation ρ θ → (focus1 = (0, 0) ∧ focus2 = (6, 0)) :=
sorry

end conic_section_foci_l12_1283


namespace simplify_fraction_l12_1221

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end simplify_fraction_l12_1221


namespace students_not_enrolled_l12_1212

theorem students_not_enrolled (total : ℕ) (math : ℕ) (chem : ℕ) (both : ℕ) :
  total = 60 ∧ math = 40 ∧ chem = 30 ∧ both = 25 →
  total - (math + chem - both) = 15 :=
by sorry

end students_not_enrolled_l12_1212


namespace third_quadrant_condition_l12_1265

def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem third_quadrant_condition (a : ℝ) :
  is_in_third_quadrant ((1 + I) * (a - I)) ↔ a < -1 := by
  sorry

end third_quadrant_condition_l12_1265


namespace remainder_theorem_l12_1252

def polynomial (x : ℝ) : ℝ := 4*x^6 - x^5 - 8*x^4 + 3*x^2 + 5*x - 15

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (x - 3) * q x + 2079 :=
by
  sorry

end remainder_theorem_l12_1252


namespace problem_solution_l12_1267

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) (sum : x + y = 5) :
  x = (7 + Real.sqrt 5) / 2 :=
by sorry

end problem_solution_l12_1267


namespace geometric_sequence_property_l12_1232

/-- Given a geometric sequence {a_n} where a_4 = 4, prove that a_3 * a_5 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h4 : a 4 = 4) : a 3 * a 5 = 16 := by
  sorry

end geometric_sequence_property_l12_1232


namespace absolute_value_equality_l12_1272

theorem absolute_value_equality (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) ↔ 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 0 ∧ c = 1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = -1)) :=
by sorry

end absolute_value_equality_l12_1272


namespace archery_probabilities_l12_1253

/-- Represents the probabilities of hitting different rings in archery --/
structure ArcheryProbabilities where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ
  ring7 : ℝ
  below7 : ℝ

/-- The given probabilities for archer Zhang Qiang --/
def zhangQiang : ArcheryProbabilities :=
  { ring10 := 0.24
  , ring9 := 0.28
  , ring8 := 0.19
  , ring7 := 0.16
  , below7 := 0.13 }

/-- The sum of all probabilities should be 1 --/
axiom probSum (p : ArcheryProbabilities) : p.ring10 + p.ring9 + p.ring8 + p.ring7 + p.below7 = 1

theorem archery_probabilities (p : ArcheryProbabilities) 
  (h : p = zhangQiang) : 
  (p.ring10 + p.ring9 = 0.52) ∧ 
  (p.ring10 + p.ring9 + p.ring8 + p.ring7 = 0.87) ∧ 
  (p.ring7 + p.below7 = 0.29) := by
  sorry


end archery_probabilities_l12_1253


namespace weaving_problem_l12_1257

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℕ  -- The sequence
  first_three_sum : a 1 + a 2 + a 3 = 9
  second_fourth_sixth_sum : a 2 + a 4 + a 6 = 15

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℕ :=
  (List.range n).map seq.a |>.sum

theorem weaving_problem (seq : ArithmeticSequence) : sum_n seq 7 = 35 := by
  sorry

end weaving_problem_l12_1257


namespace solve_system_l12_1291

theorem solve_system (a b : ℚ) 
  (eq1 : 2020 * a + 2024 * b = 2028)
  (eq2 : 2022 * a + 2026 * b = 2030) : 
  a - b = -3 := by
  sorry

end solve_system_l12_1291


namespace bluegrass_percentage_in_x_l12_1213

/-- Represents a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
def finalMixture (x y : SeedMixture) (xWeight : ℝ) : SeedMixture :=
  { ryegrass := xWeight * x.ryegrass + (1 - xWeight) * y.ryegrass,
    bluegrass := xWeight * x.bluegrass + (1 - xWeight) * y.bluegrass,
    fescue := xWeight * x.fescue + (1 - xWeight) * y.fescue }

theorem bluegrass_percentage_in_x 
  (x : SeedMixture)
  (y : SeedMixture)
  (h1 : x.ryegrass = 0.4)
  (h2 : y.ryegrass = 0.25)
  (h3 : y.fescue = 0.75)
  (h4 : (finalMixture x y 0.6667).ryegrass = 0.35)
  : x.bluegrass = 0.6 := by
  sorry

end bluegrass_percentage_in_x_l12_1213


namespace speed_difference_l12_1282

/-- Two cars traveling in opposite directions -/
structure TwoCars where
  fast_speed : ℝ
  slow_speed : ℝ
  time : ℝ
  distance : ℝ

/-- The conditions of the problem -/
def problem_conditions (cars : TwoCars) : Prop :=
  cars.fast_speed = 55 ∧
  cars.time = 5 ∧
  cars.distance = 500 ∧
  cars.distance = (cars.fast_speed + cars.slow_speed) * cars.time

/-- The theorem to prove -/
theorem speed_difference (cars : TwoCars) 
  (h : problem_conditions cars) : 
  cars.fast_speed - cars.slow_speed = 10 := by
  sorry


end speed_difference_l12_1282


namespace square_roots_of_four_l12_1274

-- Define the square root property
def is_square_root (x y : ℝ) : Prop := y ^ 2 = x

-- Theorem statement
theorem square_roots_of_four :
  ∃ (a b : ℝ), a ≠ b ∧ is_square_root 4 a ∧ is_square_root 4 b ∧
  ∀ (c : ℝ), is_square_root 4 c → (c = a ∨ c = b) :=
sorry

end square_roots_of_four_l12_1274


namespace negative_five_greater_than_negative_sqrt_26_l12_1243

theorem negative_five_greater_than_negative_sqrt_26 :
  -5 > -Real.sqrt 26 := by
  sorry

end negative_five_greater_than_negative_sqrt_26_l12_1243


namespace conference_theorem_l12_1276

def conference_problem (total : ℕ) (coffee tea soda : ℕ) 
  (coffee_tea tea_soda coffee_soda : ℕ) (all_three : ℕ) : Prop :=
  let drank_at_least_one := coffee + tea + soda - coffee_tea - tea_soda - coffee_soda + all_three
  total - drank_at_least_one = 5

theorem conference_theorem : 
  conference_problem 30 15 13 9 7 4 3 2 := by sorry

end conference_theorem_l12_1276


namespace unique_reverse_double_minus_one_l12_1290

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Theorem stating that 37 is the unique two-digit number that satisfies the given condition -/
theorem unique_reverse_double_minus_one :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 2 * n - 1 = reverse_digits n :=
by
  sorry

end unique_reverse_double_minus_one_l12_1290


namespace angle_A_is_pi_third_triangle_area_l12_1261

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the vectors
def m (t : Triangle) : ℝ × ℝ := (t.a + t.b + t.c, 3 * t.c)
def n (t : Triangle) : ℝ × ℝ := (t.b, t.c + t.b - t.a)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem 1
theorem angle_A_is_pi_third (t : Triangle) 
  (h : parallel (m t) (n t)) : t.A = π / 3 := by
  sorry

-- Theorem 2
theorem triangle_area (t : Triangle)
  (h1 : t.a = Real.sqrt 3)
  (h2 : t.b = 1)
  (h3 : t.A = π / 3) :
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2 := by
  sorry

end angle_A_is_pi_third_triangle_area_l12_1261


namespace arithmetic_sequence_length_l12_1228

theorem arithmetic_sequence_length (a₁ aₙ d n : ℕ) : 
  a₁ = 6 → aₙ = 154 → d = 4 → aₙ = a₁ + (n - 1) * d → n = 38 :=
by sorry

end arithmetic_sequence_length_l12_1228


namespace y_not_less_than_four_by_at_least_one_l12_1238

theorem y_not_less_than_four_by_at_least_one (y : ℝ) :
  (y ≥ 5) ↔ (y - 4 ≥ 1) :=
by sorry

end y_not_less_than_four_by_at_least_one_l12_1238


namespace rationalize_denominator_l12_1230

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℚ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧
    B = 7 ∧
    C = 9 ∧
    D = 13 ∧
    E = 5 :=
by sorry

end rationalize_denominator_l12_1230


namespace exact_three_green_probability_l12_1211

def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def num_trials : ℕ := 7
def num_green_selected : ℕ := 3

def probability_green : ℚ := green_marbles / total_marbles
def probability_purple : ℚ := purple_marbles / total_marbles

theorem exact_three_green_probability :
  (Nat.choose num_trials num_green_selected : ℚ) *
  (probability_green ^ num_green_selected) *
  (probability_purple ^ (num_trials - num_green_selected)) =
  860818 / 3421867 := by sorry

end exact_three_green_probability_l12_1211


namespace sara_hotdog_cost_l12_1279

/-- The cost of Sara's lunch items -/
structure LunchCost where
  total : ℝ
  salad : ℝ
  hotdog : ℝ

/-- Sara's lunch satisfies the given conditions -/
def sara_lunch : LunchCost where
  total := 10.46
  salad := 5.10
  hotdog := 10.46 - 5.10

/-- Theorem: Sara spent $5.36 on the hotdog -/
theorem sara_hotdog_cost : sara_lunch.hotdog = 5.36 := by
  sorry

end sara_hotdog_cost_l12_1279


namespace car_ownership_theorem_l12_1286

/-- The number of cars owned by Cathy, Lindsey, Carol, and Susan -/
def total_cars (cathy lindsey carol susan : ℕ) : ℕ :=
  cathy + lindsey + carol + susan

/-- Theorem stating the total number of cars owned by the four people -/
theorem car_ownership_theorem (cathy lindsey carol susan : ℕ) 
  (h1 : cathy = 5)
  (h2 : lindsey = cathy + 4)
  (h3 : carol = 2 * cathy)
  (h4 : susan = carol - 2) :
  total_cars cathy lindsey carol susan = 32 := by
  sorry

#check car_ownership_theorem

end car_ownership_theorem_l12_1286


namespace min_additional_games_correct_l12_1244

/-- The minimum number of additional games needed for the Cheetahs to win at least 80% of all games -/
def min_additional_games : ℕ := 15

/-- The initial number of games played -/
def initial_games : ℕ := 5

/-- The initial number of games won by the Cheetahs -/
def initial_cheetah_wins : ℕ := 1

/-- Checks if the given number of additional games satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  (initial_cheetah_wins + n : ℚ) / (initial_games + n : ℚ) ≥ 4/5

theorem min_additional_games_correct :
  satisfies_condition min_additional_games ∧
  ∀ m : ℕ, m < min_additional_games → ¬satisfies_condition m :=
by sorry

end min_additional_games_correct_l12_1244


namespace max_value_of_g_l12_1266

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3 ∧ ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → g y ≤ g x :=
by sorry

end max_value_of_g_l12_1266


namespace increasing_power_function_l12_1209

/-- A function f(x) = (m^2 - 2m - 2)x^(m^2 + m - 1) is increasing on (0, +∞) if and only if
    m^2 - 2m - 2 > 0 and m^2 + m - 1 > 0 -/
theorem increasing_power_function (m : ℝ) :
  let f := fun (x : ℝ) => (m^2 - 2*m - 2) * x^(m^2 + m - 1)
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ↔ 
  (m^2 - 2*m - 2 > 0 ∧ m^2 + m - 1 > 0) :=
by sorry

end increasing_power_function_l12_1209


namespace hole_depth_proof_l12_1210

/-- The depth of the hole Mat is digging -/
def hole_depth : ℝ := 120

/-- Mat's height in cm -/
def mat_height : ℝ := 90

theorem hole_depth_proof :
  (mat_height = (3/4) * hole_depth) ∧
  (hole_depth - mat_height = mat_height - (1/2) * hole_depth) :=
by sorry

end hole_depth_proof_l12_1210
