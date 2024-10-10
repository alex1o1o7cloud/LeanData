import Mathlib

namespace difference_of_squares_51_50_l3091_309164

-- Define the function for squaring a number
def square (n : ℕ) : ℕ := n * n

-- State the theorem
theorem difference_of_squares_51_50 : square 51 - square 50 = 101 := by
  sorry

end difference_of_squares_51_50_l3091_309164


namespace pareto_principle_implies_key_parts_l3091_309166

/-- Represents the Pareto Principle applied to business management -/
structure ParetoPrinciple where
  core_business : ℝ
  total_business : ℝ
  core_result : ℝ
  total_result : ℝ
  efficiency_improvement : Bool
  core_business_ratio : core_business / total_business = 0.2
  result_ratio : core_result / total_result = 0.8
  focus_on_core : efficiency_improvement = true

/-- The conclusion drawn from the Pareto Principle -/
def emphasis_on_key_parts : Prop := True

/-- Theorem stating that the Pareto Principle implies emphasis on key parts -/
theorem pareto_principle_implies_key_parts (p : ParetoPrinciple) : 
  emphasis_on_key_parts :=
sorry

end pareto_principle_implies_key_parts_l3091_309166


namespace polygon_150_sides_diagonals_l3091_309195

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 150 sides has 11025 diagonals -/
theorem polygon_150_sides_diagonals : num_diagonals 150 = 11025 := by
  sorry

end polygon_150_sides_diagonals_l3091_309195


namespace inverse_variation_solution_l3091_309198

/-- Represents the inverse variation relationship between 5y and x^3 -/
def inverse_variation (x y : ℝ) : Prop :=
  ∃ k : ℝ, 5 * y = k / (x^3)

theorem inverse_variation_solution :
  ∀ f : ℝ → ℝ,
  (∀ x, inverse_variation x (f x)) →  -- Condition: 5y varies inversely as the cube of x
  f 2 = 4 →                           -- Condition: When y = 4, x = 2
  f 4 = 1/2                           -- Conclusion: y = 1/2 when x = 4
:= by sorry

end inverse_variation_solution_l3091_309198


namespace similar_not_congruent_l3091_309197

/-- Two triangles with sides a1, b1, c1 and a2, b2, c2 respectively -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of similar triangles -/
def similar (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

/-- Definition of congruent triangles -/
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

/-- Theorem: There exist two triangles with 3 equal angles (similar) 
    and 2 equal sides that are not congruent -/
theorem similar_not_congruent : ∃ (t1 t2 : Triangle), 
  similar t1 t2 ∧ t1.c = t2.c ∧ t1.a = t2.a ∧ ¬congruent t1 t2 := by
  sorry


end similar_not_congruent_l3091_309197


namespace scientific_notation_of_280000_l3091_309194

theorem scientific_notation_of_280000 :
  280000 = 2.8 * (10 ^ 5) := by
  sorry

end scientific_notation_of_280000_l3091_309194


namespace athletes_with_four_points_after_seven_rounds_l3091_309130

/-- The number of athletes with k points after m rounds in a tournament of 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n-m) * (m.choose k)

/-- The total number of athletes with 4 points after 7 rounds in a tournament of 2^n + 6 participants -/
def athletes_with_four_points (n : ℕ) : ℕ := 35 * 2^(n-7) + 2

theorem athletes_with_four_points_after_seven_rounds (n : ℕ) (h : n > 7) :
  athletes_with_four_points n = f n 7 4 + 2 :=
sorry

#check athletes_with_four_points_after_seven_rounds

end athletes_with_four_points_after_seven_rounds_l3091_309130


namespace hexagon_area_in_circle_l3091_309183

/-- The area of a regular hexagon inscribed in a circle -/
theorem hexagon_area_in_circle (circle_area : ℝ) (hexagon_area : ℝ) : 
  circle_area = 400 * Real.pi →
  hexagon_area = 600 * Real.sqrt 3 :=
by sorry

end hexagon_area_in_circle_l3091_309183


namespace load_truck_time_proof_l3091_309125

/-- The time taken for three workers to load one truck simultaneously -/
def time_to_load_truck (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating the time taken for the given workers to load one truck -/
theorem load_truck_time_proof :
  let rate1 : ℚ := 1 / 5
  let rate2 : ℚ := 1 / 4
  let rate3 : ℚ := 1 / 6
  time_to_load_truck rate1 rate2 rate3 = 60 / 37 := by
  sorry

#eval time_to_load_truck (1/5) (1/4) (1/6)

end load_truck_time_proof_l3091_309125


namespace product_sum_reciprocals_l3091_309177

theorem product_sum_reciprocals : (3 * 5 * 7) * (1/3 + 1/5 + 1/7) = 71 := by
  sorry

end product_sum_reciprocals_l3091_309177


namespace min_value_theorem_l3091_309115

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  ∃ (min : ℝ), min = 9/2 ∧ ∀ x, x = 1/(2*a) + 2/(b-1) → x ≥ min :=
sorry

end min_value_theorem_l3091_309115


namespace no_isosceles_triangle_l3091_309171

/-- The set of stick lengths -/
def stickLengths : Set ℝ :=
  {x : ℝ | ∃ n : ℕ, n < 100 ∧ x = (0.9 : ℝ) ^ n}

/-- Definition of an isosceles triangle formed by three sticks -/
def isIsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ a + b > c ∧ b + c > a ∧ a + c > b

/-- Theorem stating the impossibility of forming an isosceles triangle -/
theorem no_isosceles_triangle :
  ¬ ∃ a b c : ℝ, a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧
    isIsoscelesTriangle a b c :=
sorry

end no_isosceles_triangle_l3091_309171


namespace tripled_base_doubled_exponent_l3091_309107

theorem tripled_base_doubled_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (3 * a) ^ (2 * b) = a ^ (2 * b) * y ^ b → y = 9 := by
  sorry

end tripled_base_doubled_exponent_l3091_309107


namespace exists_subset_with_unique_sum_representation_l3091_309139

-- Define the property for the subset X
def has_unique_sum_representation (X : Set ℤ) : Prop :=
  ∀ n : ℤ, ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n

-- Theorem statement
theorem exists_subset_with_unique_sum_representation :
  ∃ X : Set ℤ, has_unique_sum_representation X :=
sorry

end exists_subset_with_unique_sum_representation_l3091_309139


namespace x_not_equal_one_l3091_309154

theorem x_not_equal_one (x : ℝ) (h : (x - 1)^0 = 1) : x ≠ 1 := by
  sorry

end x_not_equal_one_l3091_309154


namespace root_sum_l3091_309184

-- Define the complex number 2i-3
def z : ℂ := -3 + 2*Complex.I

-- Define the quadratic equation
def quadratic (p q : ℝ) (x : ℂ) : ℂ := 2*x^2 + p*x + q

-- State the theorem
theorem root_sum (p q : ℝ) : 
  quadratic p q z = 0 → p + q = 38 := by sorry

end root_sum_l3091_309184


namespace smallest_k_for_zero_difference_l3091_309193

def u (n : ℕ) := n^4 + n^2 + n

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => fun n => f (n + 1) - f n

def iteratedΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ iteratedΔ k

theorem smallest_k_for_zero_difference :
  ∃ k, k = 5 ∧ 
    (∀ n, iteratedΔ k u n = 0) ∧
    (∀ j < k, ∃ n, iteratedΔ j u n ≠ 0) :=
sorry

end smallest_k_for_zero_difference_l3091_309193


namespace fair_selection_condition_l3091_309149

/-- Fairness condition for ball selection --/
def is_fair_selection (b c : ℕ) : Prop :=
  (b - c)^2 = b + c

/-- The probability of selecting same color balls --/
def prob_same_color (b c : ℕ) : ℚ :=
  (b * (b - 1) + c * (c - 1)) / ((b + c) * (b + c - 1))

/-- The probability of selecting different color balls --/
def prob_diff_color (b c : ℕ) : ℚ :=
  (2 * b * c) / ((b + c) * (b + c - 1))

/-- Theorem stating the fairness condition for ball selection --/
theorem fair_selection_condition (b c : ℕ) :
  prob_same_color b c = prob_diff_color b c ↔ is_fair_selection b c :=
sorry

end fair_selection_condition_l3091_309149


namespace units_digit_of_4569_pow_804_l3091_309162

theorem units_digit_of_4569_pow_804 : (4569^804) % 10 = 1 := by
  sorry

end units_digit_of_4569_pow_804_l3091_309162


namespace square_difference_pattern_l3091_309117

theorem square_difference_pattern (n : ℕ+) : (n + 2)^2 - n^2 = 4 * (n + 1) := by
  sorry

end square_difference_pattern_l3091_309117


namespace crop_ratio_l3091_309158

theorem crop_ratio (corn_rows : ℕ) (potato_rows : ℕ) (corn_per_row : ℕ) (potatoes_per_row : ℕ) (remaining_crops : ℕ) : 
  corn_rows = 10 →
  potato_rows = 5 →
  corn_per_row = 9 →
  potatoes_per_row = 30 →
  remaining_crops = 120 →
  (remaining_crops : ℚ) / ((corn_rows * corn_per_row + potato_rows * potatoes_per_row) : ℚ) = 1 / 2 := by
sorry

end crop_ratio_l3091_309158


namespace fraction_equalities_l3091_309124

theorem fraction_equalities (x y : ℚ) (h : x / y = 5 / 6) : 
  ((3 * x + 2 * y) / y = 9 / 2) ∧ 
  (y / (2 * x - y) = 3 / 2) ∧ 
  ((x - 3 * y) / y = -13 / 6) ∧ 
  ((2 * x) / (3 * y) = 5 / 9) ∧ 
  ((x + y) / (2 * y) = 11 / 12) := by
  sorry


end fraction_equalities_l3091_309124


namespace chocolate_squares_l3091_309196

theorem chocolate_squares (jenny_squares mike_squares : ℕ) : 
  jenny_squares = 65 → 
  jenny_squares = 3 * mike_squares + 5 → 
  mike_squares = 20 := by
sorry

end chocolate_squares_l3091_309196


namespace tim_balloon_count_l3091_309159

/-- Given that Dan has 58.0 violet balloons and 10.0 times more violet balloons than Tim,
    prove that Tim has 5.8 violet balloons. -/
theorem tim_balloon_count : 
  ∀ (dan_balloons tim_balloons : ℝ),
    dan_balloons = 58.0 →
    dan_balloons = 10.0 * tim_balloons →
    tim_balloons = 5.8 := by
  sorry

end tim_balloon_count_l3091_309159


namespace four_noncoplanar_points_determine_four_planes_l3091_309104

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if points are coplanar
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a function to count the number of unique planes determined by four points
def countPlanesFromPoints (p1 p2 p3 p4 : Point3D) : ℕ := sorry

-- Theorem statement
theorem four_noncoplanar_points_determine_four_planes 
  (p1 p2 p3 p4 : Point3D) 
  (h : ¬ areCoplanar p1 p2 p3 p4) : 
  countPlanesFromPoints p1 p2 p3 p4 = 4 := by sorry

end four_noncoplanar_points_determine_four_planes_l3091_309104


namespace fraction_value_l3091_309179

theorem fraction_value (a b c : Int) (h1 : a = 5) (h2 : b = -3) (h3 : c = 2) :
  3 / (a + b + c : ℚ) = 3 / 4 := by sorry

end fraction_value_l3091_309179


namespace aziz_parents_years_in_america_before_birth_l3091_309161

theorem aziz_parents_years_in_america_before_birth :
  let current_year : ℕ := 2021
  let aziz_age : ℕ := 36
  let parents_move_year : ℕ := 1982
  let aziz_birth_year : ℕ := current_year - aziz_age
  let years_before_birth : ℕ := aziz_birth_year - parents_move_year
  years_before_birth = 3 :=
by sorry

end aziz_parents_years_in_america_before_birth_l3091_309161


namespace impossible_tiling_l3091_309112

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_square : size * size = size^2)

/-- Represents a tile -/
structure Tile :=
  (length : Nat)
  (width : Nat)

/-- Represents a tiling configuration -/
structure TilingConfiguration :=
  (board : Chessboard)
  (tile : Tile)
  (num_tiles : Nat)
  (central_square_uncovered : Bool)

/-- Main theorem: Impossibility of specific tiling -/
theorem impossible_tiling (config : TilingConfiguration) : 
  config.board.size = 13 ∧ 
  config.tile.length = 4 ∧ 
  config.tile.width = 1 ∧
  config.num_tiles = 42 ∧
  config.central_square_uncovered = true
  → False :=
sorry

end impossible_tiling_l3091_309112


namespace triangle_side_difference_minimum_l3091_309121

theorem triangle_side_difference_minimum (x : ℝ) : 
  (5/3 < x) →
  (x < 11/3) →
  (x + 6 + (4*x - 1) > x + 10) →
  (x + 6 + (x + 10) > 4*x - 1) →
  ((4*x - 1) + (x + 10) > x + 6) →
  (x + 10 > x + 6) →
  (x + 10 > 4*x - 1) →
  (x + 10) - (x + 6) ≥ 4 :=
by sorry

end triangle_side_difference_minimum_l3091_309121


namespace sum_of_fractions_minus_eight_l3091_309114

theorem sum_of_fractions_minus_eight (a b c d e f : ℚ) : 
  a = 4 / 2 →
  b = 7 / 4 →
  c = 11 / 8 →
  d = 21 / 16 →
  e = 41 / 32 →
  f = 81 / 64 →
  a + b + c + d + e + f - 8 = 63 / 64 := by
  sorry

end sum_of_fractions_minus_eight_l3091_309114


namespace line_minimum_sum_l3091_309106

theorem line_minimum_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : (1 : ℝ) / a + (1 : ℝ) / b = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ (1 : ℝ) / x + (1 : ℝ) / y = 1 → a + b ≤ x + y) ∧ a + b = 4 :=
by sorry

end line_minimum_sum_l3091_309106


namespace delegation_selection_ways_l3091_309185

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of men in the brigade -/
def num_men : ℕ := 10

/-- The number of women in the brigade -/
def num_women : ℕ := 8

/-- The number of men to be selected for the delegation -/
def men_in_delegation : ℕ := 3

/-- The number of women to be selected for the delegation -/
def women_in_delegation : ℕ := 2

/-- The theorem stating the number of ways to select the delegation -/
theorem delegation_selection_ways :
  (choose num_men men_in_delegation) * (choose num_women women_in_delegation) = 3360 := by
  sorry

end delegation_selection_ways_l3091_309185


namespace food_fraction_proof_l3091_309118

def initial_amount : ℝ := 499.9999999999999

theorem food_fraction_proof (clothes_fraction : ℝ) (travel_fraction : ℝ) (food_fraction : ℝ) 
  (h1 : clothes_fraction = 1/3)
  (h2 : travel_fraction = 1/4)
  (h3 : initial_amount * (1 - clothes_fraction) * (1 - food_fraction) * (1 - travel_fraction) = 200) :
  food_fraction = 1/5 := by
  sorry

end food_fraction_proof_l3091_309118


namespace min_value_3a_plus_b_min_value_exists_min_value_equality_l3091_309126

theorem min_value_3a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = a*b) :
  ∀ x y, x > 0 → y > 0 → x + 2*y = x*y → 3*a + b ≤ 3*x + y :=
by sorry

theorem min_value_exists (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = a*b) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = x*y ∧ 3*x + y = 7 + 2*Real.sqrt 6 :=
by sorry

theorem min_value_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = a*b) :
  3*a + b ≥ 7 + 2*Real.sqrt 6 :=
by sorry

end min_value_3a_plus_b_min_value_exists_min_value_equality_l3091_309126


namespace positive_solution_range_l3091_309101

theorem positive_solution_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ a / (x + 3) = 1 / 2 ∧ x = a) → a > 3 / 2 := by
  sorry

end positive_solution_range_l3091_309101


namespace problem_statement_l3091_309152

theorem problem_statement (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x + 2*y + 3*z = 1) : 
  (∃ (m : ℝ), m = 6 + 2*Real.sqrt 2 + 2*Real.sqrt 3 + 2*Real.sqrt 6 ∧ 
   (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + 2*b + 3*c = 1 → 
    1/a + 1/b + 1/c ≥ m) ∧
   1/x + 1/y + 1/z = m) ∧ 
  x^2 + y^2 + z^2 ≥ 1/14 := by sorry

end problem_statement_l3091_309152


namespace congruence_solution_l3091_309174

theorem congruence_solution (n : ℤ) : (13 * n) % 47 = 8 ↔ n % 47 = 4 := by
  sorry

end congruence_solution_l3091_309174


namespace repeating_decimal_difference_l3091_309141

/-- Represents a repeating decimal with a single digit repeating part -/
def SingleDigitRepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with a two-digit repeating part -/
def TwoDigitRepeatingDecimal (n : ℕ) : ℚ := n / 99

theorem repeating_decimal_difference (h1 : 0 < 99) (h2 : 0 < 9) :
  99 * (TwoDigitRepeatingDecimal 49 - SingleDigitRepeatingDecimal 4) = 5 := by
  sorry

end repeating_decimal_difference_l3091_309141


namespace cosine_of_angle_between_vectors_l3091_309105

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_of_angle_between_vectors
  (c d : V)
  (h1 : ‖c‖ = 5)
  (h2 : ‖d‖ = 7)
  (h3 : ‖c + d‖ = 10) :
  inner c d / (‖c‖ * ‖d‖) = 13 / 35 := by
  sorry

end cosine_of_angle_between_vectors_l3091_309105


namespace line_x_coordinate_l3091_309165

/-- Given a line passing through (x, -4) and (10, 3) with x-intercept 4, 
    prove that x = -4 -/
theorem line_x_coordinate (x : ℝ) : 
  (∃ (m b : ℝ), (∀ (t : ℝ), -4 = m * x + b) ∧ 
                 (3 = m * 10 + b) ∧ 
                 (0 = m * 4 + b)) →
  x = -4 := by
  sorry

end line_x_coordinate_l3091_309165


namespace range_of_a_l3091_309163

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ {x : ℝ | -4*x + 4*a < 0} → x ≠ 2) → 
  a ∈ {x : ℝ | x ≥ 2} := by
  sorry

end range_of_a_l3091_309163


namespace jordan_run_time_l3091_309188

/-- Given that Jordan runs 4 miles in 1/3 of the time Steve takes to run 6 miles,
    and Steve takes 36 minutes to run 6 miles, prove that Jordan would take 30 minutes to run 10 miles. -/
theorem jordan_run_time (jordan_distance : ℝ) (steve_distance : ℝ) (steve_time : ℝ) 
  (h1 : jordan_distance = 4)
  (h2 : steve_distance = 6)
  (h3 : steve_time = 36)
  (h4 : jordan_distance * steve_time = steve_distance * (steve_time / 3)) :
  (10 : ℝ) * (steve_time / jordan_distance) = 30 := by
  sorry

end jordan_run_time_l3091_309188


namespace line_point_z_coordinate_l3091_309191

/-- Given a line passing through two points in 3D space, find the z-coordinate of a point on the line with a specific x-coordinate. -/
theorem line_point_z_coordinate 
  (p1 : ℝ × ℝ × ℝ) 
  (p2 : ℝ × ℝ × ℝ) 
  (x : ℝ) : 
  p1 = (1, 3, 2) → 
  p2 = (4, 2, -1) → 
  x = 3 → 
  ∃ (y z : ℝ), (∃ (t : ℝ), 
    (1 + 3*t, 3 - t, 2 - 3*t) = (x, y, z) ∧ 
    z = 0) := by
  sorry

#check line_point_z_coordinate

end line_point_z_coordinate_l3091_309191


namespace Q_subset_P_l3091_309127

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def Q : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem Q_subset_P : Q ⊆ P := by
  sorry

end Q_subset_P_l3091_309127


namespace contest_score_difference_l3091_309187

def score_65_percent : ℝ := 0.15
def score_85_percent : ℝ := 0.20
def score_95_percent : ℝ := 0.40
def score_110_percent : ℝ := 1 - (score_65_percent + score_85_percent + score_95_percent)

def score_65 : ℝ := 65
def score_85 : ℝ := 85
def score_95 : ℝ := 95
def score_110 : ℝ := 110

def mean_score : ℝ := 
  score_65_percent * score_65 + 
  score_85_percent * score_85 + 
  score_95_percent * score_95 + 
  score_110_percent * score_110

def median_score : ℝ := score_95

theorem contest_score_difference : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.25 ∧ |median_score - mean_score - 3| < ε :=
sorry

end contest_score_difference_l3091_309187


namespace unique_solution_l3091_309189

def machine_step (N : ℕ) : ℕ :=
  if N % 2 = 1 then 5 * N + 3
  else if N % 3 = 0 then N / 3
  else N + 1

def machine_process (N : ℕ) : ℕ :=
  (machine_step ∘ machine_step ∘ machine_step ∘ machine_step ∘ machine_step) N

theorem unique_solution :
  ∀ N : ℕ, N > 0 → (machine_process N = 1 ↔ N = 6) :=
sorry

end unique_solution_l3091_309189


namespace min_plates_min_plates_achieved_l3091_309170

theorem min_plates (m n : ℕ) : 
  2 * m + n ≥ 15 ∧ 
  m + 2 * n ≥ 18 ∧ 
  m + 3 * n ≥ 27 →
  m + n ≥ 12 :=
by
  sorry

theorem min_plates_achieved : 
  ∃ (m n : ℕ), 
    2 * m + n ≥ 15 ∧ 
    m + 2 * n ≥ 18 ∧ 
    m + 3 * n ≥ 27 ∧
    m + n = 12 :=
by
  sorry

end min_plates_min_plates_achieved_l3091_309170


namespace point_distance_from_origin_l3091_309140

theorem point_distance_from_origin (A : ℝ) : 
  (|A - 0| = 4) → (A = 4 ∨ A = -4) := by
  sorry

end point_distance_from_origin_l3091_309140


namespace train_length_problem_l3091_309150

/-- The length of Train 2 given the following conditions:
    - Train 1 length is 290 meters
    - Train 1 speed is 120 km/h
    - Train 2 speed is 80 km/h
    - Trains are running in opposite directions
    - Time to cross each other is 9 seconds
-/
theorem train_length_problem (train1_length : ℝ) (train1_speed : ℝ) (train2_speed : ℝ) (crossing_time : ℝ) :
  train1_length = 290 →
  train1_speed = 120 →
  train2_speed = 80 →
  crossing_time = 9 →
  ∃ train2_length : ℝ,
    (train1_length + train2_length) / crossing_time = (train1_speed + train2_speed) * (1000 / 3600) ∧
    abs (train2_length - 209.95) < 0.01 := by
  sorry

end train_length_problem_l3091_309150


namespace arthur_reading_challenge_l3091_309147

/-- Arthur's summer reading challenge -/
theorem arthur_reading_challenge 
  (total_goal : ℕ) 
  (book1_pages : ℕ) 
  (book1_read_percent : ℚ) 
  (book2_pages : ℕ) 
  (book2_read_fraction : ℚ) 
  (h1 : total_goal = 800)
  (h2 : book1_pages = 500)
  (h3 : book1_read_percent = 80 / 100)
  (h4 : book2_pages = 1000)
  (h5 : book2_read_fraction = 1 / 5)
  : ℕ := by
  sorry

#check arthur_reading_challenge

end arthur_reading_challenge_l3091_309147


namespace blackboard_numbers_l3091_309155

/-- Represents the state of the blackboard after n steps -/
def BlackboardState (n : ℕ) : Type := List ℕ

/-- The rule for updating the blackboard -/
def updateBlackboard (state : BlackboardState n) : BlackboardState (n + 1) :=
  sorry

/-- The number of numbers on the blackboard after n steps -/
def f (n : ℕ) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem blackboard_numbers (n : ℕ) : 
  f n = (1 / 2 : ℚ) * Nat.choose (2 * n + 2) (n + 1) := by
  sorry

end blackboard_numbers_l3091_309155


namespace right_triangle_cone_volumes_l3091_309131

/-- Given a right triangle with legs a and b, if the volume of the cone formed by
    rotating about leg a is 675π cm³ and the volume of the cone formed by rotating
    about leg b is 1215π cm³, then the length of the hypotenuse is 3√106 cm. -/
theorem right_triangle_cone_volumes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / 3 : ℝ) * π * b^2 * a = 675 * π →
  (1 / 3 : ℝ) * π * a^2 * b = 1215 * π →
  Real.sqrt (a^2 + b^2) = 3 * Real.sqrt 106 := by
sorry


end right_triangle_cone_volumes_l3091_309131


namespace royalties_sales_ratio_decrease_l3091_309178

/-- Calculate the percentage decrease in the ratio of royalties to sales --/
theorem royalties_sales_ratio_decrease (first_royalties second_royalties : ℝ)
  (first_sales second_sales : ℝ) :
  first_royalties = 6 →
  first_sales = 20 →
  second_royalties = 9 →
  second_sales = 108 →
  let first_ratio := first_royalties / first_sales
  let second_ratio := second_royalties / second_sales
  let percentage_decrease := (first_ratio - second_ratio) / first_ratio * 100
  abs (percentage_decrease - 72.23) < 0.01 := by
sorry

end royalties_sales_ratio_decrease_l3091_309178


namespace ellipse_condition_equiv_k_range_ellipse_standard_equation_l3091_309123

-- Define the curve C
def curve_C (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (4 - k) - y^2 / (1 - k) = 1

-- Define the condition for an ellipse with foci on the x-axis
def is_ellipse_x_axis (k : ℝ) : Prop :=
  4 - k > 0 ∧ k - 1 > 0 ∧ 4 - k > k - 1

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  1 < k ∧ k < 5/2

-- Define the ellipse passing through (√6, 2) with foci at (-2,0) and (2,0)
def ellipse_through_point (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 8 = 1

-- Theorem 1: Equivalence of ellipse condition and k range
theorem ellipse_condition_equiv_k_range (k : ℝ) :
  is_ellipse_x_axis k ↔ k_range k :=
sorry

-- Theorem 2: Standard equation of the ellipse
theorem ellipse_standard_equation :
  ellipse_through_point (Real.sqrt 6) 2 :=
sorry

end ellipse_condition_equiv_k_range_ellipse_standard_equation_l3091_309123


namespace alligator_journey_time_l3091_309192

/-- The additional time taken for the return journey of alligators -/
def additional_time (initial_time : ℕ) (total_alligators : ℕ) (total_time : ℕ) : ℕ :=
  (total_time - initial_time) / total_alligators - initial_time

/-- Theorem stating that the additional time for the return journey is 2 hours -/
theorem alligator_journey_time : additional_time 4 7 46 = 2 := by
  sorry

end alligator_journey_time_l3091_309192


namespace volume_to_surface_area_ratio_l3091_309119

/-- Represents a T-shaped structure made of unit cubes -/
structure TCube where
  base : Nat
  stack : Nat

/-- Calculates the volume of the T-shaped structure -/
def volume (t : TCube) : Nat :=
  t.base + t.stack

/-- Calculates the surface area of the T-shaped structure -/
def surfaceArea (t : TCube) : Nat :=
  2 * (5 + 3) + 1 + 3 * 5

/-- The specific T-shaped structure described in the problem -/
def specificT : TCube :=
  { base := 4, stack := 4 }

theorem volume_to_surface_area_ratio :
  (volume specificT : ℚ) / (surfaceArea specificT : ℚ) = 1 / 4 := by
  sorry

end volume_to_surface_area_ratio_l3091_309119


namespace hot_sauce_duration_l3091_309142

-- Define the volume of a quart in ounces
def quart_volume : ℝ := 32

-- Define the size of the hot sauce container
def container_size : ℝ := quart_volume - 2

-- Define the size of one serving in ounces
def serving_size : ℝ := 0.5

-- Define the number of servings James uses per day
def servings_per_day : ℝ := 3

-- Define the amount of hot sauce James uses per day
def daily_usage : ℝ := serving_size * servings_per_day

-- Theorem: The hot sauce will last 20 days
theorem hot_sauce_duration :
  container_size / daily_usage = 20 := by sorry

end hot_sauce_duration_l3091_309142


namespace smallest_sum_of_squares_l3091_309172

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 143 → x^2 + y^2 ≥ 145 := by
sorry

end smallest_sum_of_squares_l3091_309172


namespace another_root_of_p_l3091_309137

-- Define the polynomials p and q
def p (a b : ℤ) (x : ℂ) : ℂ := x^3 + a*x^2 + b*x - 1
def q (c d : ℤ) (x : ℂ) : ℂ := x^3 + c*x^2 + d*x + 1

-- State the theorem
theorem another_root_of_p (a b c d : ℤ) (α : ℂ) :
  (∃ (a b : ℤ), p a b α = 0) →  -- α is a root of p(x) = 0
  (∀ (r : ℚ), p a b r ≠ 0) →  -- p(x) is irreducible over the rationals
  (∃ (c d : ℤ), q c d (α + 1) = 0) →  -- α + 1 is a root of q(x) = 0
  (∃ β : ℂ, p a b β = 0 ∧ (β = -1/(α+1) ∨ β = -(α+1)/α)) :=
by sorry

end another_root_of_p_l3091_309137


namespace d_properties_l3091_309168

/-- Given a nonnegative integer c, define sequences a_n and d_n -/
def a (c n : ℕ) : ℕ := n^2 + c

def d (c n : ℕ) : ℕ := Nat.gcd (a c n) (a c (n + 1))

/-- Theorem stating the properties of d_n for different values of c -/
theorem d_properties (c : ℕ) :
  (∀ n : ℕ, n ≥ 1 → c = 0 → d c n = 1) ∧
  (∀ n : ℕ, n ≥ 1 → c = 1 → d c n = 1 ∨ d c n = 5) ∧
  (∀ n : ℕ, n ≥ 1 → d c n ≤ 4 * c + 1) :=
sorry

end d_properties_l3091_309168


namespace f_derivative_at_2_when_a_0_f_minimum_at_0_iff_a_lt_2_g_not_tangent_to_line_with_slope_3_2_l3091_309135

noncomputable section

open Real

/-- The base of the natural logarithm -/
def e : ℝ := exp 1

/-- The function f(x) = (x^2 + ax + a)e^(-x) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x + a) * (e^(-x))

/-- The function g(x) = (4 - x)e^(x - 2) for x < 2 -/
def g (x : ℝ) : ℝ := (4 - x) * (e^(x - 2))

theorem f_derivative_at_2_when_a_0 :
  (deriv (f 0)) 2 = 0 := by sorry

theorem f_minimum_at_0_iff_a_lt_2 (a : ℝ) :
  (∀ x, f a 0 ≤ f a x) ↔ a < 2 := by sorry

theorem g_not_tangent_to_line_with_slope_3_2 :
  ¬ ∃ (c : ℝ), ∃ (x : ℝ), x < 2 ∧ g x = (3/2) * x + c ∧ (deriv g) x = 3/2 := by sorry

end f_derivative_at_2_when_a_0_f_minimum_at_0_iff_a_lt_2_g_not_tangent_to_line_with_slope_3_2_l3091_309135


namespace club_members_remainder_l3091_309157

theorem club_members_remainder (N : ℕ) : 
  50 < N → N < 80 → 
  N % 5 = 0 → (N % 8 = 0 ∨ N % 7 = 0) → 
  N % 9 = 6 ∨ N % 9 = 7 := by
sorry

end club_members_remainder_l3091_309157


namespace common_ratio_of_geometric_sequence_l3091_309133

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, geometric_sequence a q)
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 1/4) :
  ∃ q : ℝ, geometric_sequence a q ∧ q = 1/2 := by
  sorry

end common_ratio_of_geometric_sequence_l3091_309133


namespace value_of_X_l3091_309113

theorem value_of_X : ∃ X : ℚ, (1/4 : ℚ) * (1/8 : ℚ) * X = (1/2 : ℚ) * (1/6 : ℚ) * 120 ∧ X = 320 := by
  sorry

end value_of_X_l3091_309113


namespace turtle_count_l3091_309173

/-- Represents the number of turtles in the lake -/
def total_turtles : ℕ := 100

/-- Percentage of female turtles -/
def female_percentage : ℚ := 60 / 100

/-- Percentage of male turtles with stripes -/
def male_striped_percentage : ℚ := 25 / 100

/-- Number of baby male turtles with stripes -/
def baby_striped_males : ℕ := 4

/-- Percentage of adult male turtles with stripes -/
def adult_striped_percentage : ℚ := 60 / 100

theorem turtle_count :
  total_turtles = 100 :=
sorry

end turtle_count_l3091_309173


namespace point_coordinates_l3091_309175

-- Define a point in the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the second quadrant
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- Define the distance from a point to the x-axis
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

-- Define the distance from a point to the y-axis
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

-- Theorem statement
theorem point_coordinates (p : Point) 
  (h1 : secondQuadrant p)
  (h2 : distanceToXAxis p = 4)
  (h3 : distanceToYAxis p = 5) :
  p.x = -5 ∧ p.y = 4 := by
  sorry

end point_coordinates_l3091_309175


namespace M_remainder_l3091_309132

/-- The number of positive integers less than or equal to 2010 whose base-2 representation has more 1's than 0's -/
def M : ℕ := 1162

/-- 2010 is less than 2^11 - 1 -/
axiom h1 : 2010 < 2^11 - 1

/-- The sum of binary numbers where the number of 1's is more than 0's up to 2^11 - 1 -/
def total_sum : ℕ := 2^11 - 1

/-- The number of binary numbers more than 2010 and ≤ 2047 -/
def excess : ℕ := 37

/-- The sum of center elements in Pascal's Triangle rows 0 to 5 -/
def center_sum : ℕ := 351

theorem M_remainder (h2 : M = (total_sum + center_sum) / 2 - excess) :
  M % 1000 = 162 := by sorry

end M_remainder_l3091_309132


namespace f_monotone_iff_m_range_l3091_309144

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - m) * x - m else Real.log x / Real.log m

-- State the theorem
theorem f_monotone_iff_m_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) ↔ (3/2 ≤ m ∧ m < 3) :=
sorry

end f_monotone_iff_m_range_l3091_309144


namespace vertical_distance_to_charlie_l3091_309181

/-- The vertical distance between the midpoint of the line segment connecting
    (8, -15) and (2, 10), and the point (5, 3) is 5.5 units. -/
theorem vertical_distance_to_charlie : 
  let annie : ℝ × ℝ := (8, -15)
  let barbara : ℝ × ℝ := (2, 10)
  let charlie : ℝ × ℝ := (5, 3)
  let midpoint : ℝ × ℝ := ((annie.1 + barbara.1) / 2, (annie.2 + barbara.2) / 2)
  charlie.2 - midpoint.2 = 5.5 := by sorry

end vertical_distance_to_charlie_l3091_309181


namespace swimmer_speed_in_still_water_l3091_309129

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmerSpeeds where
  man : ℝ  -- Speed of the man in still water
  stream : ℝ  -- Speed of the stream

/-- Calculates the effective speed when swimming downstream -/
def downstream_speed (s : SwimmerSpeeds) : ℝ := s.man + s.stream

/-- Calculates the effective speed when swimming upstream -/
def upstream_speed (s : SwimmerSpeeds) : ℝ := s.man - s.stream

/-- Theorem: Given the conditions of the swimming problem, the man's speed in still water is 8 km/h -/
theorem swimmer_speed_in_still_water :
  ∃ (s : SwimmerSpeeds),
    (downstream_speed s * 4 = 48) ∧
    (upstream_speed s * 6 = 24) ∧
    (s.man = 8) := by
  sorry

#check swimmer_speed_in_still_water

end swimmer_speed_in_still_water_l3091_309129


namespace hidden_sum_is_55_l3091_309151

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The total number of dice -/
def num_dice : ℕ := 4

/-- The list of visible numbers on the dice -/
def visible_numbers : List ℕ := [1, 2, 3, 3, 4, 5, 5, 6]

/-- The sum of all numbers on all dice -/
def total_sum : ℕ := die_sum * num_dice

/-- The sum of visible numbers -/
def visible_sum : ℕ := visible_numbers.sum

theorem hidden_sum_is_55 : total_sum - visible_sum = 55 := by
  sorry

end hidden_sum_is_55_l3091_309151


namespace ranch_minimum_animals_l3091_309128

theorem ranch_minimum_animals (ponies horses : ℕ) : 
  ponies > 0 →
  horses = ponies + 3 →
  ∃ (ponies_with_horseshoes icelandic_ponies : ℕ),
    ponies_with_horseshoes = (3 * ponies) / 10 ∧
    icelandic_ponies = (5 * ponies_with_horseshoes) / 8 →
  ponies + horses ≥ 35 :=
by
  sorry

end ranch_minimum_animals_l3091_309128


namespace coffee_shop_spending_prove_coffee_shop_spending_l3091_309103

theorem coffee_shop_spending : ℝ → ℝ → Prop :=
  fun (ben_spent david_spent : ℝ) =>
    (david_spent = ben_spent / 2) →
    (ben_spent = david_spent + 15) →
    (ben_spent + david_spent = 45)

/-- Proof of the coffee shop spending theorem -/
theorem prove_coffee_shop_spending :
  ∃ (ben_spent david_spent : ℝ),
    coffee_shop_spending ben_spent david_spent :=
by
  sorry

end coffee_shop_spending_prove_coffee_shop_spending_l3091_309103


namespace determinant_scaling_l3091_309120

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = -3 →
  Matrix.det ![![3*x, 3*y], ![5*z, 5*w]] = -45 := by
  sorry

end determinant_scaling_l3091_309120


namespace remaining_candies_formula_l3091_309100

/-- Represents the remaining number of candies after the first night -/
def remaining_candies (K S m n : ℕ) : ℚ :=
  (K + S : ℚ) * (1 - m / n)

/-- Theorem stating that the remaining number of candies after the first night
    is equal to (K + S) * (1 - m/n) -/
theorem remaining_candies_formula (K S m n : ℕ) (h : n ≠ 0) :
  remaining_candies K S m n = (K + S : ℚ) * (1 - m / n) :=
by sorry

end remaining_candies_formula_l3091_309100


namespace joan_balloons_l3091_309110

/-- The number of blue balloons Joan has after gaining more -/
def total_balloons (initial : ℕ) (gained : ℕ) : ℕ :=
  initial + gained

theorem joan_balloons :
  total_balloons 9 2 = 11 := by
  sorry

end joan_balloons_l3091_309110


namespace stating_num_elective_ways_l3091_309148

/-- Represents the number of elective courses -/
def num_courses : ℕ := 4

/-- Represents the number of academic years -/
def num_years : ℕ := 3

/-- Represents the maximum number of courses a student can take per year -/
def max_courses_per_year : ℕ := 3

/-- 
Calculates the number of ways to distribute distinct courses over years
-/
def distribute_courses : ℕ := sorry

/-- 
Theorem stating that the number of ways to distribute the courses is 78
-/
theorem num_elective_ways : distribute_courses = 78 := by sorry

end stating_num_elective_ways_l3091_309148


namespace die_roll_probabilities_l3091_309138

-- Define the sample space for rolling a fair six-sided die twice
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events
def A : Set Ω := {ω | ω.1 + ω.2 = 4}
def B : Set Ω := {ω | ω.2 % 2 = 0}
def C : Set Ω := {ω | ω.1 = ω.2}
def D : Set Ω := {ω | ω.1 % 2 = 1 ∨ ω.2 % 2 = 1}

-- Theorem statement
theorem die_roll_probabilities :
  P D = 3/4 ∧
  P (B ∩ D) = 1/4 ∧
  P (B ∩ C) = P B * P C := by sorry

end die_roll_probabilities_l3091_309138


namespace pentagonal_cross_section_exists_regular_pentagonal_cross_section_impossible_l3091_309109

/-- A cube in 3D space -/
structure Cube :=
  (side : ℝ)
  (side_pos : side > 0)

/-- A plane in 3D space -/
structure Plane :=
  (normal : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)

/-- A pentagon in 2D space -/
structure Pentagon :=
  (vertices : Finset (ℝ × ℝ))
  (is_pentagon : vertices.card = 5)

/-- A regular pentagon in 2D space -/
structure RegularPentagon extends Pentagon :=
  (is_regular : ∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 → 
    ∃ (rotation : ℝ × ℝ → ℝ × ℝ), rotation v1 = v2 ∧ rotation '' vertices = vertices)

/-- The cross-section formed by intersecting a cube with a plane -/
def crossSection (c : Cube) (p : Plane) : Set (ℝ × ℝ) :=
  sorry

theorem pentagonal_cross_section_exists (c : Cube) : 
  ∃ (p : Plane), ∃ (pent : Pentagon), crossSection c p = ↑pent.vertices :=
sorry

theorem regular_pentagonal_cross_section_impossible (c : Cube) : 
  ¬∃ (p : Plane), ∃ (reg_pent : RegularPentagon), crossSection c p = ↑reg_pent.vertices :=
sorry

end pentagonal_cross_section_exists_regular_pentagonal_cross_section_impossible_l3091_309109


namespace division_problem_l3091_309186

theorem division_problem (n : ℕ) : 
  (n / 20 = 9) ∧ (n % 20 = 1) → n = 181 := by
  sorry

end division_problem_l3091_309186


namespace paint_intensity_problem_l3091_309167

theorem paint_intensity_problem (original_intensity new_intensity replacement_fraction : ℝ) 
  (h1 : original_intensity = 0.1)
  (h2 : new_intensity = 0.15)
  (h3 : replacement_fraction = 0.5) :
  let added_intensity := (new_intensity - (1 - replacement_fraction) * original_intensity) / replacement_fraction
  added_intensity = 0.2 := by
sorry

end paint_intensity_problem_l3091_309167


namespace solution_l3091_309153

def problem (m n : ℕ) : Prop :=
  m + n = 80 ∧ 
  Nat.gcd m n = 6 ∧ 
  Nat.lcm m n = 210

theorem solution (m n : ℕ) (h : problem m n) : 
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 15.75 := by
  sorry

end solution_l3091_309153


namespace simplify_square_roots_l3091_309143

theorem simplify_square_roots : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end simplify_square_roots_l3091_309143


namespace caitlin_age_l3091_309134

theorem caitlin_age (anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ)
  (h1 : anna_age = 42)
  (h2 : brianna_age = anna_age / 2)
  (h3 : caitlin_age = brianna_age - 5) :
  caitlin_age = 16 := by
sorry

end caitlin_age_l3091_309134


namespace no_2016_subsequence_l3091_309169

-- Define the sequence
def seq : ℕ → ℕ
  | 0 => 2
  | 1 => 0
  | 2 => 1
  | 3 => 7
  | 4 => 0
  | n + 5 => (seq n + seq (n + 1) + seq (n + 2) + seq (n + 3)) % 10

-- Define a function to check if a subsequence appears at a given position
def subsequenceAt (start : ℕ) (subseq : List ℕ) : Prop :=
  ∀ i, i < subseq.length → seq (start + i) = subseq.get ⟨i, by sorry⟩

-- Theorem statement
theorem no_2016_subsequence :
  ¬ ∃ start : ℕ, start ≥ 4 ∧ subsequenceAt start [2, 0, 1, 6] :=
by sorry

end no_2016_subsequence_l3091_309169


namespace power_of_four_remainder_l3091_309199

theorem power_of_four_remainder (a : ℕ+) (p : ℕ) :
  p = 4^(a : ℕ) → p % 10 = 6 → ∃ k : ℕ, (a : ℕ) = 2 * k := by
  sorry

end power_of_four_remainder_l3091_309199


namespace smallest_n_for_seating_arrangement_l3091_309190

theorem smallest_n_for_seating_arrangement (k : ℕ) : 
  (2 ≤ k) → 
  (∃ n : ℕ, 
    k < n ∧ 
    (2 * (n - 1).factorial * (n - k + 2) = n * (n - 1).factorial) ∧
    (∀ m : ℕ, m < n → 
      (2 ≤ m ∧ k < m) → 
      (2 * (m - 1).factorial * (m - k + 2) ≠ m * (m - 1).factorial))) → 
  (∃ n : ℕ, 
    k < n ∧ 
    (2 * (n - 1).factorial * (n - k + 2) = n * (n - 1).factorial) ∧
    n = 12) :=
sorry

end smallest_n_for_seating_arrangement_l3091_309190


namespace b_formula_l3091_309156

/-- Sequence a_n defined recursively --/
def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 / (a n + 1)

/-- Sequence b_n defined in terms of a_n --/
def b (n : ℕ) : ℚ := |((a n + 2) / (a n - 1))|

/-- The main theorem to be proved --/
theorem b_formula (n : ℕ) : b n = 2^(n + 1) := by
  sorry

end b_formula_l3091_309156


namespace sum_of_integers_l3091_309160

theorem sum_of_integers : (-25) + 34 + 156 + (-65) = 100 := by
  sorry

end sum_of_integers_l3091_309160


namespace brainiacs_liking_neither_count_l3091_309111

/-- The number of brainiacs who like neither rebus teasers nor math teasers -/
def brainiacs_liking_neither (total : ℕ) (rebus : ℕ) (math : ℕ) (both : ℕ) : ℕ :=
  total - (rebus + math - both)

/-- Theorem stating the number of brainiacs liking neither type of teaser -/
theorem brainiacs_liking_neither_count :
  let total := 100
  let rebus := 2 * math
  let both := 18
  let math_not_rebus := 20
  let math := both + math_not_rebus
  brainiacs_liking_neither total rebus math both = 4 := by
  sorry

#eval brainiacs_liking_neither 100 76 38 18

end brainiacs_liking_neither_count_l3091_309111


namespace exists_multiple_in_ascending_sequence_l3091_309108

/-- Definition of an ascending sequence -/
def IsAscending (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1) ∧ a (2 * n) = 2 * a n

/-- Theorem: For any ascending sequence of positive integers and prime p > a₁,
    there exists a term in the sequence divisible by p -/
theorem exists_multiple_in_ascending_sequence
    (a : ℕ → ℕ)
    (h_ascending : IsAscending a)
    (h_positive : ∀ n, a n > 0)
    (p : ℕ)
    (h_prime : Nat.Prime p)
    (h_p_gt_a1 : p > a 1) :
    ∃ n, p ∣ a n := by
  sorry

end exists_multiple_in_ascending_sequence_l3091_309108


namespace doubling_points_theorem_l3091_309136

/-- Definition of a "doubling point" -/
def is_doubling_point (P Q : ℝ × ℝ) : Prop :=
  2 * (P.1 + Q.1) = P.2 + Q.2

/-- The point P₁ -/
def P₁ : ℝ × ℝ := (1, 0)

/-- Q₁ and Q₂ are specified points -/
def Q₁ : ℝ × ℝ := (3, 8)
def Q₂ : ℝ × ℝ := (-2, -2)

/-- The parabola y = x² - 2x - 3 -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem doubling_points_theorem :
  (is_doubling_point P₁ Q₁) ∧
  (is_doubling_point P₁ Q₂) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    is_doubling_point P₁ (x₁, parabola x₁) ∧
    is_doubling_point P₁ (x₂, parabola x₂)) ∧
  (∀ (Q : ℝ × ℝ), is_doubling_point P₁ Q → 
    Real.sqrt ((Q.1 - P₁.1)^2 + (Q.2 - P₁.2)^2) ≥ 4 * Real.sqrt 5 / 5) ∧
  (∃ (Q : ℝ × ℝ), is_doubling_point P₁ Q ∧
    Real.sqrt ((Q.1 - P₁.1)^2 + (Q.2 - P₁.2)^2) = 4 * Real.sqrt 5 / 5) := by
  sorry

end doubling_points_theorem_l3091_309136


namespace frustum_sphere_equal_volume_l3091_309102

/-- Given a frustum of a cone with small radius 2 inches, large radius 3 inches,
    and height 5 inches, the radius of a sphere with the same volume is ∛(95/4) inches. -/
theorem frustum_sphere_equal_volume :
  let r₁ : ℝ := 2  -- small radius of frustum
  let r₂ : ℝ := 3  -- large radius of frustum
  let h : ℝ := 5   -- height of frustum
  let V_frustum := (1/3) * π * h * (r₁^2 + r₁*r₂ + r₂^2)
  let r_sphere := (95/4)^(1/3 : ℝ)
  let V_sphere := (4/3) * π * r_sphere^3
  V_frustum = V_sphere := by sorry

end frustum_sphere_equal_volume_l3091_309102


namespace exponent_multiplication_l3091_309182

theorem exponent_multiplication (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by
  sorry

end exponent_multiplication_l3091_309182


namespace cloth_cost_price_calculation_l3091_309176

/-- The cost price of one metre of cloth, given the selling details --/
def cost_price_per_metre (cloth_length : ℕ) (selling_price : ℚ) (profit_per_metre : ℚ) : ℚ :=
  (selling_price - cloth_length * profit_per_metre) / cloth_length

theorem cloth_cost_price_calculation :
  let cloth_length : ℕ := 92
  let selling_price : ℚ := 9890
  let profit_per_metre : ℚ := 24
  cost_price_per_metre cloth_length selling_price profit_per_metre = 83.5 := by
sorry


end cloth_cost_price_calculation_l3091_309176


namespace angle_measure_when_sine_is_half_l3091_309116

/-- If ∠A is an acute angle in a triangle and sin A = 1/2, then ∠A = 30°. -/
theorem angle_measure_when_sine_is_half (A : Real) (h_acute : 0 < A ∧ A < π / 2) 
  (h_sin : Real.sin A = 1 / 2) : A = π / 6 := by
  sorry

end angle_measure_when_sine_is_half_l3091_309116


namespace min_value_expression_min_value_attainable_l3091_309122

theorem min_value_expression (x y : ℝ) : 
  (x * y - 1/2)^2 + (x - y)^2 ≥ 1/4 :=
sorry

theorem min_value_attainable : 
  ∃ x y : ℝ, (x * y - 1/2)^2 + (x - y)^2 = 1/4 :=
sorry

end min_value_expression_min_value_attainable_l3091_309122


namespace no_positive_roots_l3091_309146

theorem no_positive_roots :
  ∀ x : ℝ, x^3 + 6*x^2 + 11*x + 6 = 0 → x ≤ 0 := by
  sorry

end no_positive_roots_l3091_309146


namespace geometric_sequence_sum_l3091_309145

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric a →
  (∀ n : ℕ+, a n > 0) →
  a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36 →
  a 2 + a 5 = 6 := by
sorry

end geometric_sequence_sum_l3091_309145


namespace remainder_polynomial_l3091_309180

theorem remainder_polynomial (p : ℝ → ℝ) (h1 : p 2 = 4) (h2 : p 4 = 8) :
  ∃ (q r : ℝ → ℝ), (∀ x, p x = q x * (x - 2) * (x - 4) + r x) ∧
                    (∀ x, r x = 2 * x) :=
by sorry

end remainder_polynomial_l3091_309180
