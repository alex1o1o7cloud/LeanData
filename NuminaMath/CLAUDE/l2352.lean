import Mathlib

namespace vector_sum_parallel_l2352_235231

/-- Given two parallel vectors a and b in R², prove that their linear combination results in (-4, -8) -/
theorem vector_sum_parallel (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  (2 • a + 3 • b : Fin 2 → ℝ) = ![-4, -8] := by
sorry

end vector_sum_parallel_l2352_235231


namespace no_two_different_three_digit_cubes_l2352_235262

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, i < digits.length → j < digits.length → i ≠ j → digits.get ⟨i, by sorry⟩ ≠ digits.get ⟨j, by sorry⟩

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem no_two_different_three_digit_cubes :
  ∀ KUB SHAR : ℕ,
  is_three_digit KUB →
  is_three_digit SHAR →
  all_digits_different KUB →
  all_digits_different SHAR →
  is_cube KUB →
  (∀ d : ℕ, d < 10 → (d ∈ KUB.digits 10 → d ∉ SHAR.digits 10) ∧ (d ∈ SHAR.digits 10 → d ∉ KUB.digits 10)) →
  ¬ is_cube SHAR :=
by sorry

end no_two_different_three_digit_cubes_l2352_235262


namespace smallest_valid_m_l2352_235264

def T : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 3 / 2 ∧ Real.sqrt 2 / 2 ≤ y ∧ y ≤ 1}

def is_valid_m (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z ∈ T, z^n = Complex.I

theorem smallest_valid_m : 
  (is_valid_m 6) ∧ (∀ m : ℕ, m < 6 → ¬(is_valid_m m)) :=
sorry

end smallest_valid_m_l2352_235264


namespace fifth_term_is_two_l2352_235228

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying specific conditions, prove that its fifth term is 2. -/
theorem fifth_term_is_two (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_2 : a 2 = 2 * a 3 + 1)
  (h_4 : a 4 = 2 * a 3 + 7) : 
  a 5 = 2 := by
  sorry

end fifth_term_is_two_l2352_235228


namespace high_scam_probability_l2352_235293

/-- Represents an email message -/
structure Email :=
  (claims_prize : Bool)
  (asks_for_phone : Bool)
  (requests_payment : Bool)
  (payment_amount : ℕ)

/-- Represents the probability of an email being a scam -/
def scam_probability (e : Email) : ℝ := sorry

/-- Theorem: Given an email with specific characteristics, the probability of it being a scam is high -/
theorem high_scam_probability (e : Email) 
  (h1 : e.claims_prize = true)
  (h2 : e.asks_for_phone = true)
  (h3 : e.requests_payment = true)
  (h4 : e.payment_amount = 150) :
  scam_probability e > 0.9 := by sorry

end high_scam_probability_l2352_235293


namespace geometric_sum_first_seven_terms_l2352_235268

/-- Sum of a finite geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/5

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/3

/-- The number of terms to sum -/
def n : ℕ := 7

theorem geometric_sum_first_seven_terms :
  geometric_sum a r n = 2186/3645 := by
  sorry

end geometric_sum_first_seven_terms_l2352_235268


namespace purely_imaginary_product_l2352_235285

theorem purely_imaginary_product (x : ℝ) : 
  (Complex.I : ℂ).im * ((x + 2 * Complex.I) * ((x + 3) + 2 * Complex.I) * ((x + 5) + 2 * Complex.I)).re = 0 ↔ 
  x = -5 ∨ x = -4 ∨ x = 1 := by
  sorry

end purely_imaginary_product_l2352_235285


namespace largest_divisor_of_product_l2352_235210

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h2 : n > 0) :
  ∃ (k : ℕ), k ≤ 15 ∧ 
  (∀ (m : ℕ), m ≤ k → (m ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+11))) ∧
  (∀ (m : ℕ), m > 15 → ∃ (p : ℕ), Even p ∧ p > 0 ∧ ¬(m ∣ (p+1)*(p+3)*(p+5)*(p+7)*(p+11))) :=
by sorry

end largest_divisor_of_product_l2352_235210


namespace distance_to_focus_l2352_235265

/-- Given a parabola y² = 8x and a point P(4, y) on it, 
    the distance from P to the focus of the parabola is 6. -/
theorem distance_to_focus (y : ℝ) : 
  y^2 = 32 →  -- Point P(4, y) is on the parabola y² = 8x
  let F := (2, 0)  -- Focus of the parabola
  Real.sqrt ((4 - 2)^2 + y^2) = 6 := by sorry

end distance_to_focus_l2352_235265


namespace percentage_relationship_l2352_235256

theorem percentage_relationship (A B T : ℝ) 
  (h1 : B = 0.14 * T) 
  (h2 : A = 0.5 * B) : 
  A = 0.07 * T := by
sorry

end percentage_relationship_l2352_235256


namespace quadratic_inequality_range_l2352_235289

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), x^2 + m*x - 1 < 0) → 
  m ∈ Set.Ioo (-Real.sqrt 2 / 2) 0 :=
by sorry

end quadratic_inequality_range_l2352_235289


namespace min_roots_count_l2352_235248

/-- A function satisfying the given symmetry conditions -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (2 - x) = f (2 + x)) ∧ (∀ x : ℝ, f (7 - x) = f (7 + x))

/-- The theorem stating the minimum number of roots -/
theorem min_roots_count
  (f : ℝ → ℝ)
  (h_symmetric : SymmetricFunction f)
  (h_root_zero : f 0 = 0) :
  (∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, f x = 0 ∧ x ∈ Set.Icc (-1000) 1000) ∧
    (∀ roots' : Finset ℝ, (∀ x ∈ roots', f x = 0 ∧ x ∈ Set.Icc (-1000) 1000) → 
      roots'.card ≤ roots.card) ∧
    roots.card = 401) :=
  sorry

end min_roots_count_l2352_235248


namespace smallest_perfect_squares_l2352_235242

theorem smallest_perfect_squares (a b : ℕ+) 
  (h1 : ∃ x : ℕ, (15 * a + 16 * b : ℕ) = x^2)
  (h2 : ∃ y : ℕ, (16 * a - 15 * b : ℕ) = y^2) :
  ∃ (x y : ℕ), x^2 = 231361 ∧ y^2 = 231361 ∧ 
    (∀ (x' y' : ℕ), (15 * a + 16 * b : ℕ) = x'^2 → (16 * a - 15 * b : ℕ) = y'^2 → 
      x'^2 ≥ 231361 ∧ y'^2 ≥ 231361) :=
by sorry

end smallest_perfect_squares_l2352_235242


namespace power_equality_l2352_235267

theorem power_equality (n b : ℝ) : n = 2 ^ (1/4) → n ^ b = 8 → b = 12 := by
  sorry

end power_equality_l2352_235267


namespace water_percentage_is_15_l2352_235229

/-- Calculates the percentage of water in a mixture of three liquids -/
def water_percentage_in_mixture (a_percentage : ℚ) (b_percentage : ℚ) (c_percentage : ℚ) 
  (a_parts : ℚ) (b_parts : ℚ) (c_parts : ℚ) : ℚ :=
  ((a_percentage * a_parts + b_percentage * b_parts + c_percentage * c_parts) / 
   (a_parts + b_parts + c_parts)) * 100

/-- Theorem stating that the percentage of water in the given mixture is 15% -/
theorem water_percentage_is_15 : 
  water_percentage_in_mixture (10/100) (15/100) (25/100) 4 3 2 = 15 := by
  sorry

end water_percentage_is_15_l2352_235229


namespace imaginary_part_of_one_minus_i_cubed_l2352_235211

theorem imaginary_part_of_one_minus_i_cubed (i : ℂ) : 
  i^2 = -1 → Complex.im ((1 - i)^3) = -2 := by sorry

end imaginary_part_of_one_minus_i_cubed_l2352_235211


namespace local_minimum_value_inequality_condition_l2352_235284

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x

-- Theorem for part (1)
theorem local_minimum_value (a : ℝ) :
  (∃ k, ∀ x, x ≠ 1 → (f a x - f a 1) / (x - 1) = k) →
  (∃ x₀, ∀ x, x ≠ x₀ → f a x > f a x₀) →
  f a x₀ = -Real.log 2 - 5/4 := by sorry

-- Theorem for part (2)
theorem inequality_condition (x₁ x₂ m : ℝ) :
  1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 →
  (∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 →
    f 1 x₁ - f 1 x₂ > m * (x₂ - x₁) / (x₁ * x₂)) ↔
  m ≤ -6 := by sorry

end local_minimum_value_inequality_condition_l2352_235284


namespace cos_sin_sum_l2352_235244

theorem cos_sin_sum (α : Real) (h : Real.cos (π/6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5*π/6 + α) + (Real.sin (α - π/6))^2 = (2 - Real.sqrt 3) / 2 := by
sorry

end cos_sin_sum_l2352_235244


namespace production_equation_l2352_235200

/-- Represents the equation for a production scenario where increasing the daily rate by 20%
    completes the task 4 days earlier. -/
theorem production_equation (x : ℝ) (h : x > 0) :
  (3000 : ℝ) / x = 4 + (3000 : ℝ) / (x * (1 + 20 / 100)) :=
sorry

end production_equation_l2352_235200


namespace fifteenth_base_five_number_l2352_235239

/-- Represents a number in base 5 --/
def BaseFive : Type := Nat

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : Nat) : BaseFive :=
  sorry

/-- The sequence of numbers in base 5 --/
def baseFiveSequence : Nat → BaseFive :=
  sorry

theorem fifteenth_base_five_number :
  baseFiveSequence 15 = toBaseFive 30 := by
  sorry

end fifteenth_base_five_number_l2352_235239


namespace marley_has_31_fruits_l2352_235249

-- Define the number of fruits for Louis and Samantha
def louis_oranges : ℕ := 5
def louis_apples : ℕ := 3
def samantha_oranges : ℕ := 8
def samantha_apples : ℕ := 7

-- Define Marley's fruits in terms of Louis and Samantha
def marley_oranges : ℕ := 2 * louis_oranges
def marley_apples : ℕ := 3 * samantha_apples

-- Define the total number of Marley's fruits
def marley_total_fruits : ℕ := marley_oranges + marley_apples

-- Theorem statement
theorem marley_has_31_fruits : marley_total_fruits = 31 := by
  sorry

end marley_has_31_fruits_l2352_235249


namespace tetrahedron_face_sum_squares_l2352_235252

/-- A tetrahedron with circumradius 1 and face triangles with sides a, b, and c -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  circumradius : ℝ
  circumradius_eq_one : circumradius = 1

/-- The sum of squares of the face triangle sides of a tetrahedron with circumradius 1 is equal to 8 -/
theorem tetrahedron_face_sum_squares (t : Tetrahedron) : t.a^2 + t.b^2 + t.c^2 = 8 := by
  sorry

end tetrahedron_face_sum_squares_l2352_235252


namespace slope_product_l2352_235257

/-- Given two lines L₁ and L₂ with equations y = mx and y = nx respectively,
    where L₁ makes three times as large an angle with the horizontal as L₂,
    L₁ has 5 times the slope of L₂, and L₁ is not horizontal,
    prove that mn = 5/7. -/
theorem slope_product (m n : ℝ) : 
  m ≠ 0 →  -- L₁ is not horizontal
  (∃ θ₁ θ₂ : ℝ, 
    θ₁ = 3 * θ₂ ∧  -- L₁ makes three times as large an angle with the horizontal as L₂
    m = Real.tan θ₁ ∧ 
    n = Real.tan θ₂ ∧
    m = 5 * n) →  -- L₁ has 5 times the slope of L₂
  m * n = 5 / 7 := by
  sorry

end slope_product_l2352_235257


namespace probability_three_red_one_blue_l2352_235296

theorem probability_three_red_one_blue (total_red : Nat) (total_blue : Nat) 
  (draw_count : Nat) (red_count : Nat) (blue_count : Nat) :
  total_red = 10 →
  total_blue = 5 →
  draw_count = 4 →
  red_count = 3 →
  blue_count = 1 →
  (Nat.choose total_red red_count * Nat.choose total_blue blue_count : ℚ) / 
  (Nat.choose (total_red + total_blue) draw_count) = 40 / 91 :=
by sorry

end probability_three_red_one_blue_l2352_235296


namespace specific_box_volume_l2352_235237

/-- The volume of an open box created from a rectangular sheet --/
def box_volume (sheet_length sheet_width x : ℝ) : ℝ :=
  (sheet_length - 2*x) * (sheet_width - 2*x) * x

/-- Theorem: The volume of the specific box is 4x^3 - 60x^2 + 216x --/
theorem specific_box_volume :
  ∀ x : ℝ, box_volume 18 12 x = 4*x^3 - 60*x^2 + 216*x :=
by
  sorry

end specific_box_volume_l2352_235237


namespace unique_number_exists_l2352_235226

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a + b + c = 10 ∧
    b = a + c ∧
    100 * c + 10 * b + a = 100 * a + 10 * b + c + 99

theorem unique_number_exists :
  ∃! n : ℕ, is_valid_number n ∧ n = 203 :=
sorry

end unique_number_exists_l2352_235226


namespace raisin_distribution_l2352_235274

/-- The number of raisins Bryce received -/
def bryce_raisins : ℕ := 15

/-- The number of raisins Carter received -/
def carter_raisins : ℕ := bryce_raisins - 10

theorem raisin_distribution : 
  (bryce_raisins = 15) ∧ 
  (carter_raisins = bryce_raisins - 10) ∧ 
  (carter_raisins = bryce_raisins / 3) := by
  sorry

end raisin_distribution_l2352_235274


namespace sum_of_interior_angles_is_360_or_540_l2352_235286

/-- A regular polygon with all diagonals equal -/
structure EqualDiagonalRegularPolygon where
  /-- Number of sides of the polygon -/
  n : ℕ
  /-- Condition that the polygon has at least 3 sides -/
  h_n : n ≥ 3
  /-- Condition that all diagonals are equal -/
  all_diagonals_equal : True

/-- The sum of interior angles of a regular polygon with all diagonals equal -/
def sum_of_interior_angles (p : EqualDiagonalRegularPolygon) : ℝ :=
  (p.n - 2) * 180

/-- Theorem stating that the sum of interior angles is either 360° or 540° -/
theorem sum_of_interior_angles_is_360_or_540 (p : EqualDiagonalRegularPolygon) :
  sum_of_interior_angles p = 360 ∨ sum_of_interior_angles p = 540 := by
  sorry

end sum_of_interior_angles_is_360_or_540_l2352_235286


namespace angle_Q_measure_l2352_235294

-- Define a regular octagon
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

-- Define the extended sides and point Q
def extended_sides (octagon : RegularOctagon) : sorry := sorry

def point_Q (octagon : RegularOctagon) : ℝ × ℝ := sorry

-- Define the angle at Q
def angle_Q (octagon : RegularOctagon) : ℝ := sorry

-- Theorem statement
theorem angle_Q_measure (octagon : RegularOctagon) : 
  angle_Q octagon = 22.5 := by sorry

end angle_Q_measure_l2352_235294


namespace odd_function_property_l2352_235288

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : OddFunction f) (h_fa : f a = 11) : f (-a) = -11 := by
  sorry

end odd_function_property_l2352_235288


namespace function_inequality_l2352_235299

theorem function_inequality (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x) :
  ∀ x y : ℝ, x > y → f x + y ≤ f y + x := by
sorry

end function_inequality_l2352_235299


namespace cheese_purchase_l2352_235246

theorem cheese_purchase (initial_amount : ℕ) (cheese_cost beef_cost : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 87)
  (h2 : cheese_cost = 7)
  (h3 : beef_cost = 5)
  (h4 : remaining_amount = 61) :
  (initial_amount - remaining_amount - beef_cost) / cheese_cost = 3 :=
by sorry

end cheese_purchase_l2352_235246


namespace complex_fraction_evaluation_l2352_235201

theorem complex_fraction_evaluation : (1 - I) / (2 + I) = 1/5 - 3/5 * I := by
  sorry

end complex_fraction_evaluation_l2352_235201


namespace pump_fill_time_proof_l2352_235203

/-- The time it takes to fill the tank with the leak (in hours) -/
def fill_time_with_leak : ℝ := 20

/-- The time it takes for the leak to empty the tank (in hours) -/
def leak_empty_time : ℝ := 5

/-- The time it takes for the pump to fill the tank without the leak (in hours) -/
def pump_fill_time : ℝ := 4

theorem pump_fill_time_proof : 
  (1 / pump_fill_time - 1 / leak_empty_time) * fill_time_with_leak = 1 := by
  sorry

end pump_fill_time_proof_l2352_235203


namespace solution_set_f_geq_2_max_a_value_l2352_235275

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x - 1|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set_f_geq_2 :
  {x : ℝ | f x ≥ 2} = {x : ℝ | x ≤ 0 ∨ x ≥ 4/3} :=
sorry

-- Theorem for the maximum value of a
theorem max_a_value (a : ℝ) :
  (∀ x : ℝ, f x ≥ a * |x|) ↔ a ≤ 1 :=
sorry

end solution_set_f_geq_2_max_a_value_l2352_235275


namespace hexagon_side_length_l2352_235251

/-- A regular hexagon with six segments drawn inside -/
structure SegmentedHexagon where
  /-- The side length of the hexagon -/
  side_length : ℝ
  /-- The lengths of the six segments -/
  segment_lengths : Fin 6 → ℝ
  /-- The segments are drawn sequentially with right angles between them -/
  segments_right_angled : Bool
  /-- The segments have lengths from 1 to 6 -/
  segment_lengths_valid : ∀ i, segment_lengths i = (i : ℝ) + 1

/-- The theorem stating that the side length of the hexagon is 15/2 -/
theorem hexagon_side_length (h : SegmentedHexagon) : h.side_length = 15 / 2 := by
  sorry


end hexagon_side_length_l2352_235251


namespace conditional_prob_specific_given_different_l2352_235298

/-- The number of attractions available for tourists to choose from. -/
def num_attractions : ℕ := 5

/-- The probability that two tourists choose different attractions. -/
def prob_different_attractions : ℚ := 4 / 5

/-- The probability that one tourist chooses a specific attraction and the other chooses any of the remaining attractions. -/
def prob_one_specific_others_different : ℚ := 8 / 25

/-- Theorem stating the conditional probability of both tourists choosing a specific attraction given they choose different attractions. -/
theorem conditional_prob_specific_given_different :
  prob_one_specific_others_different / prob_different_attractions = 2 / 5 := by
  sorry

end conditional_prob_specific_given_different_l2352_235298


namespace y_intercept_for_specific_line_l2352_235277

/-- A line in the two-dimensional plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope -3 and x-intercept (7, 0), the y-intercept is (0, 21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := (7, 0) }
  y_intercept l = (0, 21) := by sorry

end y_intercept_for_specific_line_l2352_235277


namespace chess_tournament_games_l2352_235278

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) * 2

/-- Theorem: In a chess tournament with 16 players, where each player plays twice with every other player, the total number of games is 480 -/
theorem chess_tournament_games :
  tournament_games 16 = 480 := by
  sorry

end chess_tournament_games_l2352_235278


namespace adam_change_l2352_235279

-- Define the given amounts
def adam_money : ℚ := 5.00
def airplane_cost : ℚ := 4.28

-- Define the change function
def change (money cost : ℚ) : ℚ := money - cost

-- Theorem statement
theorem adam_change :
  change adam_money airplane_cost = 0.72 := by
  sorry

end adam_change_l2352_235279


namespace jasmine_percentage_l2352_235269

/-- Calculates the percentage of jasmine in a solution after adding jasmine and water -/
theorem jasmine_percentage
  (initial_volume : ℝ)
  (initial_jasmine_percentage : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 90)
  (h2 : initial_jasmine_percentage = 5)
  (h3 : added_jasmine = 8)
  (h4 : added_water = 2) :
  let initial_jasmine := initial_volume * (initial_jasmine_percentage / 100)
  let total_jasmine := initial_jasmine + added_jasmine
  let total_volume := initial_volume + added_jasmine + added_water
  let final_percentage := (total_jasmine / total_volume) * 100
  final_percentage = 12.5 := by
sorry


end jasmine_percentage_l2352_235269


namespace eleven_one_base_three_is_perfect_square_l2352_235243

/-- Represents a number in a given base --/
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- Checks if a number is a perfect square --/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- The main theorem --/
theorem eleven_one_base_three_is_perfect_square :
  isPerfectSquare (toDecimal [1, 1, 1, 1, 1] 3) := by
  sorry

end eleven_one_base_three_is_perfect_square_l2352_235243


namespace parabola_c_value_l2352_235235

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 3 = 4 →  -- vertex at (4, 3)
  p.x_coord 5 = 2 →  -- passes through (2, 5)
  p.c = 0.5 := by
  sorry

end parabola_c_value_l2352_235235


namespace fraction_of_students_with_B_l2352_235290

theorem fraction_of_students_with_B (fraction_A : Real) (fraction_A_or_B : Real) 
  (h1 : fraction_A = 0.7)
  (h2 : fraction_A_or_B = 0.9) :
  fraction_A_or_B - fraction_A = 0.2 := by
  sorry

end fraction_of_students_with_B_l2352_235290


namespace square_field_area_l2352_235272

/-- The area of a square field with a diagonal of 30 meters is 450 square meters. -/
theorem square_field_area (diagonal : ℝ) (h : diagonal = 30) : 
  (diagonal ^ 2) / 2 = 450 := by
  sorry

end square_field_area_l2352_235272


namespace min_distance_to_parabola_l2352_235225

/-- Rectilinear distance between two points -/
def rectilinear_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- A point on the parabola x² = y -/
def parabola_point (t : ℝ) : ℝ × ℝ := (t, t^2)

/-- The fixed point M(-1, 0) -/
def M : ℝ × ℝ := (-1, 0)

/-- Theorem: The minimum rectilinear distance from M(-1, 0) to the parabola x² = y is 3/4 -/
theorem min_distance_to_parabola :
  ∀ t : ℝ, rectilinear_distance (M.1) (M.2) (parabola_point t).1 (parabola_point t).2 ≥ 3/4 ∧
  ∃ t₀ : ℝ, rectilinear_distance (M.1) (M.2) (parabola_point t₀).1 (parabola_point t₀).2 = 3/4 :=
sorry

end min_distance_to_parabola_l2352_235225


namespace total_points_earned_l2352_235214

/-- The number of pounds required to earn one point -/
def pounds_per_point : ℕ := 4

/-- The number of pounds Paige recycled -/
def paige_pounds : ℕ := 14

/-- The number of pounds Paige's friends recycled -/
def friends_pounds : ℕ := 2

/-- The total number of pounds recycled -/
def total_pounds : ℕ := paige_pounds + friends_pounds

/-- The theorem stating that the total points earned is 4 -/
theorem total_points_earned : (total_pounds / pounds_per_point : ℕ) = 4 := by
  sorry

end total_points_earned_l2352_235214


namespace marble_count_l2352_235234

theorem marble_count (total : ℕ) (blue : ℕ) (prob_red_or_white : ℚ) 
  (h1 : total = 50)
  (h2 : blue = 5)
  (h3 : prob_red_or_white = 9/10) :
  total - blue = 45 := by
sorry

end marble_count_l2352_235234


namespace percentage_problem_l2352_235213

theorem percentage_problem (x : ℝ) :
  (15 / 100) * (30 / 100) * (50 / 100) * x = 108 → x = 4800 := by
  sorry

end percentage_problem_l2352_235213


namespace smallest_multiple_with_factors_l2352_235292

theorem smallest_multiple_with_factors : ∃ (n : ℕ+), 
  (∀ (m : ℕ+), (1452 * m : ℕ) % 2^4 = 0 ∧ 
                (1452 * m : ℕ) % 3^3 = 0 ∧ 
                (1452 * m : ℕ) % 13^3 = 0 → n ≤ m) ∧
  (1452 * n : ℕ) % 2^4 = 0 ∧ 
  (1452 * n : ℕ) % 3^3 = 0 ∧ 
  (1452 * n : ℕ) % 13^3 = 0 ∧
  n = 79092 := by
  sorry

end smallest_multiple_with_factors_l2352_235292


namespace missing_fraction_sum_l2352_235238

theorem missing_fraction_sum (x : ℚ) :
  1/3 + 1/2 + 1/5 + 1/4 + (-9/20) + (-9/20) + x = 9/20 →
  x = 1/15 := by
sorry

end missing_fraction_sum_l2352_235238


namespace dracula_is_alive_l2352_235255

-- Define the propositions
variable (T : Prop) -- "The Transylvanian is human"
variable (D : Prop) -- "Count Dracula is alive"

-- Define the Transylvanian's statements
variable (statement1 : T)
variable (statement2 : T → D)

-- Define the Transylvanian's ability to reason logically
variable (logical_reasoning : T)

-- Theorem to prove
theorem dracula_is_alive : D := by
  sorry

end dracula_is_alive_l2352_235255


namespace m_range_characterization_l2352_235212

/-- Proposition p: For all x ∈ ℝ, x² + 2x > m -/
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x > m

/-- Proposition q: There exists x₀ ∈ ℝ, such that x₀² + 2mx₀ + 2 - m ≤ 0 -/
def q (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 - m ≤ 0

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := (m > -2 ∧ m < -1) ∨ m ≥ 1

theorem m_range_characterization (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
sorry

end m_range_characterization_l2352_235212


namespace parallel_transitivity_l2352_235271

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define a relation for parallel lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (a b c : Line) :
  Parallel a c → Parallel b c → Parallel a b := by
  sorry

end parallel_transitivity_l2352_235271


namespace distance_on_quadratic_curve_l2352_235202

/-- The distance between two points on a quadratic curve -/
theorem distance_on_quadratic_curve (m k a b c d : ℝ) :
  b = m * a^2 + k →
  d = m * c^2 + k →
  Real.sqrt ((c - a)^2 + (d - b)^2) = |c - a| * Real.sqrt (1 + m^2 * (c + a)^2) := by
  sorry

end distance_on_quadratic_curve_l2352_235202


namespace different_chord_length_l2352_235259

-- Define the ellipse
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / m + y^2 / 4 = 1

-- Define the chord length for a line y = ax + b on the ellipse
noncomputable def chordLength (m a b : ℝ) : ℝ :=
  let A := 4 + m * a^2
  let B := 2 * m * a
  let C := m * (b^2 - 1)
  Real.sqrt ((B^2 - 4*A*C) / A^2)

-- Theorem statement
theorem different_chord_length (k m : ℝ) (hm : m > 0) :
  chordLength m k 1 ≠ chordLength m (-k) 2 :=
sorry

end different_chord_length_l2352_235259


namespace cloth_cost_per_meter_l2352_235222

theorem cloth_cost_per_meter (total_length : ℝ) (total_cost : ℝ) 
  (h1 : total_length = 9.25)
  (h2 : total_cost = 434.75) :
  total_cost / total_length = 47 := by
  sorry

end cloth_cost_per_meter_l2352_235222


namespace complex_division_l2352_235253

theorem complex_division (i : ℂ) : i^2 = -1 → (2 : ℂ) / (1 + i) = 1 - i := by
  sorry

end complex_division_l2352_235253


namespace range_of_f_range_of_a_l2352_235215

-- Define the functions
def f (x : ℝ) : ℝ := 2 * abs (x - 1) - abs (x - 4)

def g (x a : ℝ) : ℝ := 2 * abs (x - 1) - abs (x - a)

-- State the theorems
theorem range_of_f : Set.range f = Set.Ici (-3) := by sorry

theorem range_of_a (h : ∀ x : ℝ, g x a ≥ -1) : a ∈ Set.Icc 0 2 := by sorry

end range_of_f_range_of_a_l2352_235215


namespace max_value_z_minus_i_l2352_235205

theorem max_value_z_minus_i (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (M : ℝ), M = 3 ∧ ∀ w, Complex.abs w = 2 → Complex.abs (w - Complex.I) ≤ M :=
sorry

end max_value_z_minus_i_l2352_235205


namespace complement_intersection_theorem_l2352_235233

def U : Set Nat := {1, 2, 3, 4, 5}
def P : Set Nat := {2, 4}
def Q : Set Nat := {1, 3, 4, 6}

theorem complement_intersection_theorem :
  (U \ P) ∩ Q = {1, 3} := by sorry

end complement_intersection_theorem_l2352_235233


namespace convertible_count_l2352_235230

theorem convertible_count (total : ℕ) (regular_percent : ℚ) (truck_percent : ℚ) :
  total = 125 →
  regular_percent = 64 / 100 →
  truck_percent = 8 / 100 →
  (total : ℚ) * regular_percent + (total : ℚ) * truck_percent + 35 = total :=
by sorry

end convertible_count_l2352_235230


namespace cookies_for_guests_l2352_235273

/-- Given the total number of cookies and cookies per guest, calculate the number of guests. -/
def number_of_guests (total_cookies : ℕ) (cookies_per_guest : ℕ) : ℕ :=
  total_cookies / cookies_per_guest

/-- Theorem stating that the number of guests is 2 when there are 38 total cookies and 19 cookies per guest. -/
theorem cookies_for_guests : number_of_guests 38 19 = 2 := by
  sorry

end cookies_for_guests_l2352_235273


namespace mars_visibility_time_l2352_235260

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : ℕ :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  if totalMinutes2 ≥ totalMinutes1 then
    totalMinutes2 - totalMinutes1
  else
    (24 * 60) - (totalMinutes1 - totalMinutes2)

/-- Subtracts a given number of minutes from a time -/
def subtractMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes
  let newTotalMinutes := (totalMinutes + 24 * 60 - m) % (24 * 60)
  { hours := newTotalMinutes / 60,
    minutes := newTotalMinutes % 60,
    h_valid := by sorry }

theorem mars_visibility_time 
  (jupiter_after_mars : ℕ) 
  (uranus_after_jupiter : ℕ)
  (uranus_appearance : Time)
  (h1 : jupiter_after_mars = 2 * 60 + 41)
  (h2 : uranus_after_jupiter = 3 * 60 + 16)
  (h3 : uranus_appearance = { hours := 6, minutes := 7, h_valid := by sorry }) :
  let jupiter_time := subtractMinutes uranus_appearance uranus_after_jupiter
  let mars_time := subtractMinutes jupiter_time jupiter_after_mars
  mars_time = { hours := 0, minutes := 10, h_valid := by sorry } :=
by sorry

end mars_visibility_time_l2352_235260


namespace height_prediction_approximate_l2352_235254

/-- Represents a linear regression model for height prediction -/
structure HeightModel where
  slope : ℝ
  intercept : ℝ

/-- Predicts height based on the model and age -/
def predict_height (model : HeightModel) (age : ℝ) : ℝ :=
  model.slope * age + model.intercept

/-- The given height prediction model -/
def given_model : HeightModel := { slope := 7.19, intercept := 73.93 }

/-- Theorem stating that the predicted height at age 10 is approximately 145.83cm -/
theorem height_prediction_approximate :
  ∃ ε > 0, ∀ δ > 0, δ < ε → 
    |predict_height given_model 10 - 145.83| < δ :=
sorry

end height_prediction_approximate_l2352_235254


namespace circle_center_radius_sum_l2352_235206

/-- Given a circle C' with equation x^2 - 14x + y^2 + 16y + 100 = 0,
    prove that the sum of its center coordinates and radius is -1 + √13 -/
theorem circle_center_radius_sum :
  ∃ (a' b' r' : ℝ),
    (∀ (x y : ℝ), x^2 - 14*x + y^2 + 16*y + 100 = 0 ↔ (x - a')^2 + (y - b')^2 = r'^2) →
    a' + b' + r' = -1 + Real.sqrt 13 := by
  sorry

end circle_center_radius_sum_l2352_235206


namespace expression_evaluation_l2352_235207

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y) :
  (x * (1 / x)) * (y / (1 / y)) = 1 / x^2 := by
  sorry

end expression_evaluation_l2352_235207


namespace unique_equidistant_point_l2352_235282

/-- The line equation 4x + 3y = 12 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y = 12

/-- A point (x, y) is on the line if it satisfies the line equation -/
def point_on_line (x y : ℝ) : Prop := line_equation x y

/-- The point (x, y) is in the first quadrant -/
def in_first_quadrant (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0

/-- The point (x, y) is equidistant from coordinate axes -/
def equidistant_from_axes (x y : ℝ) : Prop := x = y

/-- The theorem stating that (12/7, 12/7) is the unique point satisfying all conditions -/
theorem unique_equidistant_point :
  ∃! (x y : ℝ), point_on_line x y ∧ in_first_quadrant x y ∧ equidistant_from_axes x y ∧ x = 12/7 ∧ y = 12/7 := by
  sorry

end unique_equidistant_point_l2352_235282


namespace jimmy_father_emails_l2352_235223

/-- The number of emails Jimmy's father receives per day before subscribing to the news channel -/
def initial_emails_per_day : ℕ := 20

/-- The number of additional emails per day after subscribing to the news channel -/
def additional_emails_per_day : ℕ := 5

/-- The total number of days in April -/
def days_in_april : ℕ := 30

/-- The day in April when Jimmy's father subscribed to the news channel -/
def subscription_day : ℕ := days_in_april / 2

theorem jimmy_father_emails :
  (subscription_day * initial_emails_per_day) +
  ((days_in_april - subscription_day) * (initial_emails_per_day + additional_emails_per_day)) = 675 := by
  sorry

end jimmy_father_emails_l2352_235223


namespace modifiedLucas_100th_term_divisible_by_5_l2352_235250

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 4
  | (n + 2) => (modifiedLucas n + modifiedLucas (n + 1)) % 5

theorem modifiedLucas_100th_term_divisible_by_5 : modifiedLucas 99 % 5 = 0 := by
  sorry

end modifiedLucas_100th_term_divisible_by_5_l2352_235250


namespace infinite_circles_inside_l2352_235232

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define what it means for a point to be inside a circle
def isInside (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define what it means for a circle to be entirely inside another circle
def isEntirelyInside (c1 c2 : Circle) : Prop :=
  ∀ p : Point, isInside p c1 → isInside p c2

-- The main theorem
theorem infinite_circles_inside (C : Circle) (A B : Point) 
  (hA : isInside A C) (hB : isInside B C) :
  ∃ f : ℕ → Circle, (∀ n : ℕ, isEntirelyInside (f n) C ∧ 
                               isInside A (f n) ∧ 
                               isInside B (f n)) ∧
                     (∀ m n : ℕ, m ≠ n → f m ≠ f n) := by
  sorry

end infinite_circles_inside_l2352_235232


namespace simplify_expression_l2352_235295

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (25 * x^3) * (8 * x^4) * (1 / (4 * x^2)^3) = 25/8 * x := by
  sorry

end simplify_expression_l2352_235295


namespace least_amount_to_add_l2352_235266

def savings : ℕ := 642986
def children : ℕ := 9

theorem least_amount_to_add : 
  (∃ (x : ℕ), (savings + x) % children = 0 ∧ 
  ∀ (y : ℕ), y < x → (savings + y) % children ≠ 0) → 
  (∃ (x : ℕ), (savings + x) % children = 0 ∧ 
  ∀ (y : ℕ), y < x → (savings + y) % children ≠ 0 ∧ x = 1) :=
sorry

end least_amount_to_add_l2352_235266


namespace mathematics_magnet_problem_l2352_235270

/-- The number of letters in 'MATHEMATICS' -/
def total_letters : ℕ := 11

/-- The number of vowels in 'MATHEMATICS' -/
def num_vowels : ℕ := 4

/-- The number of consonants in 'MATHEMATICS' -/
def num_consonants : ℕ := 7

/-- The number of vowels selected -/
def selected_vowels : ℕ := 3

/-- The number of consonants selected -/
def selected_consonants : ℕ := 4

/-- The number of distinct possible collections of letters -/
def distinct_collections : ℕ := 490

theorem mathematics_magnet_problem :
  (total_letters = num_vowels + num_consonants) →
  (distinct_collections = 490) :=
by sorry

end mathematics_magnet_problem_l2352_235270


namespace vector_simplification_l2352_235219

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem vector_simplification :
  (1 / 2 : ℝ) • ((2 : ℝ) • a + (8 : ℝ) • b) - ((4 : ℝ) • a - (2 : ℝ) • b) = (6 : ℝ) • b - (3 : ℝ) • a :=
by sorry

end vector_simplification_l2352_235219


namespace remainder_cd_mod_m_l2352_235227

theorem remainder_cd_mod_m (m c d : ℤ) : 
  m > 0 → 
  ∃ c_inv d_inv : ℤ, (c * c_inv ≡ 1 [ZMOD m]) ∧ (d * d_inv ≡ 1 [ZMOD m]) →
  d ≡ 2 * c_inv [ZMOD m] →
  c * d ≡ 2 [ZMOD m] := by
sorry

end remainder_cd_mod_m_l2352_235227


namespace gcf_of_60_90_150_l2352_235204

theorem gcf_of_60_90_150 : Nat.gcd 60 (Nat.gcd 90 150) = 30 := by
  sorry

end gcf_of_60_90_150_l2352_235204


namespace redox_agents_identification_l2352_235291

/-- Represents a chemical species with its oxidation state -/
structure Species where
  element : String
  oxidation_state : Int

/-- Represents a half-reaction in a redox reaction -/
structure HalfReaction where
  reactant : Species
  product : Species
  electrons : Int
  is_reduction : Bool

/-- Represents a full redox reaction -/
structure RedoxReaction where
  oxidation : HalfReaction
  reduction : HalfReaction

def is_oxidizing_agent (s : Species) (r : RedoxReaction) : Prop :=
  s = r.reduction.reactant

def is_reducing_agent (s : Species) (r : RedoxReaction) : Prop :=
  s = r.oxidation.reactant

theorem redox_agents_identification
  (s0 : Species)
  (h20 : Species)
  (h2plus : Species)
  (s2minus : Species)
  (reduction : HalfReaction)
  (oxidation : HalfReaction)
  (full_reaction : RedoxReaction)
  (h_s0 : s0 = ⟨"S", 0⟩)
  (h_h20 : h20 = ⟨"H2", 0⟩)
  (h_h2plus : h2plus = ⟨"H2", 1⟩)
  (h_s2minus : s2minus = ⟨"S", -2⟩)
  (h_reduction : reduction = ⟨s0, s2minus, 2, true⟩)
  (h_oxidation : oxidation = ⟨h20, h2plus, -2, false⟩)
  (h_full_reaction : full_reaction = ⟨oxidation, reduction⟩)
  : is_oxidizing_agent s0 full_reaction ∧ is_reducing_agent h20 full_reaction := by
  sorry


end redox_agents_identification_l2352_235291


namespace pyramid_volume_is_2_root2_div_3_l2352_235241

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with vertex P and base ABC -/
structure Pyramid where
  P : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Check if a triangle is equilateral -/
def isEquilateral (A B C : Point3D) : Prop := 
  distance A B = distance B C ∧ distance B C = distance C A

/-- Calculate the angle between three points -/
def angle (A P C : Point3D) : ℝ := sorry

/-- Calculate the volume of a pyramid -/
def pyramidVolume (p : Pyramid) : ℝ := sorry

theorem pyramid_volume_is_2_root2_div_3 (p : Pyramid) : 
  isEquilateral p.A p.B p.C →
  distance p.P p.A = distance p.P p.B ∧ 
  distance p.P p.B = distance p.P p.C →
  distance p.A p.B = 2 →
  angle p.A p.P p.C = Real.pi / 2 →
  pyramidVolume p = 2 * Real.sqrt 2 / 3 := by
  sorry

end pyramid_volume_is_2_root2_div_3_l2352_235241


namespace animals_fiber_intake_l2352_235287

-- Define the absorption rates and absorbed amounts
def koala_absorption_rate : ℝ := 0.30
def koala_absorbed_amount : ℝ := 12
def kangaroo_absorption_rate : ℝ := 0.40
def kangaroo_absorbed_amount : ℝ := 16

-- Define the theorem
theorem animals_fiber_intake :
  ∃ (koala_intake kangaroo_intake : ℝ),
    koala_intake * koala_absorption_rate = koala_absorbed_amount ∧
    kangaroo_intake * kangaroo_absorption_rate = kangaroo_absorbed_amount ∧
    koala_intake = 40 ∧
    kangaroo_intake = 40 := by
  sorry

end animals_fiber_intake_l2352_235287


namespace sum_of_coefficients_zero_l2352_235258

theorem sum_of_coefficients_zero 
  (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*(x + 1) + a₂*(x + 1)^2 + a₃*(x + 1)^3 + a₄*(x + 1)^4) → 
  a₁ + a₂ + a₃ + a₄ = 0 := by
sorry

end sum_of_coefficients_zero_l2352_235258


namespace special_set_is_all_reals_l2352_235261

/-- A subset of real numbers with a special property -/
def SpecialSet (A : Set ℝ) : Prop :=
  (∀ x y : ℝ, x + y ∈ A → x * y ∈ A) ∧ Set.Nonempty A

/-- The main theorem: Any special set of real numbers is equal to the entire set of real numbers -/
theorem special_set_is_all_reals (A : Set ℝ) (h : SpecialSet A) : A = Set.univ :=
sorry

end special_set_is_all_reals_l2352_235261


namespace work_days_of_a_l2352_235276

/-- Represents the number of days worked by each person -/
structure WorkDays where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total earnings given work days and daily wages -/
def totalEarnings (days : WorkDays) (wages : DailyWages) : ℕ :=
  days.a * wages.a + days.b * wages.b + days.c * wages.c

/-- The main theorem to prove -/
theorem work_days_of_a (days : WorkDays) (wages : DailyWages) : 
  days.b = 9 ∧ 
  days.c = 4 ∧ 
  wages.a * 4 = wages.b * 3 ∧ 
  wages.b * 5 = wages.c * 4 ∧ 
  wages.c = 125 ∧ 
  totalEarnings days wages = 1850 → 
  days.a = 6 := by
  sorry


end work_days_of_a_l2352_235276


namespace power_division_23_l2352_235236

theorem power_division_23 : (23 ^ 11 : ℕ) / (23 ^ 5) = 148035889 := by sorry

end power_division_23_l2352_235236


namespace credit_card_balance_calculation_l2352_235281

/-- Calculates the final balance on a credit card after two interest applications and an additional charge. -/
def finalBalance (initialBalance : ℝ) (interestRate : ℝ) (additionalCharge : ℝ) : ℝ :=
  let balanceAfterFirstInterest := initialBalance * (1 + interestRate)
  let balanceAfterCharge := balanceAfterFirstInterest + additionalCharge
  balanceAfterCharge * (1 + interestRate)

/-- Theorem stating that given the specific conditions, the final balance is $96.00 -/
theorem credit_card_balance_calculation :
  finalBalance 50 0.2 20 = 96 := by
  sorry

end credit_card_balance_calculation_l2352_235281


namespace max_students_distribution_l2352_235218

theorem max_students_distribution (pens pencils erasers notebooks : ℕ) 
  (h_pens : pens = 4261)
  (h_pencils : pencils = 2677)
  (h_erasers : erasers = 1759)
  (h_notebooks : notebooks = 1423) :
  (∃ (n : ℕ), n > 0 ∧ 
    pens % n = 0 ∧ pencils % n = 0 ∧ erasers % n = 0 ∧ notebooks % n = 0 ∧
    (∀ m : ℕ, m > n → (pens % m ≠ 0 ∨ pencils % m ≠ 0 ∨ erasers % m ≠ 0 ∨ notebooks % m ≠ 0))) →
  1 = Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks :=
by sorry

end max_students_distribution_l2352_235218


namespace fraction_problem_l2352_235216

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  N = 180 →
  6 + (1/2) * (1/3) * f * N = (1/15) * N →
  f = 1/5 := by
sorry

end fraction_problem_l2352_235216


namespace lcm_gcd_product_15_75_l2352_235245

theorem lcm_gcd_product_15_75 : Nat.lcm 15 75 * Nat.gcd 15 75 = 1125 := by
  sorry

end lcm_gcd_product_15_75_l2352_235245


namespace gcd_of_840_and_1764_l2352_235240

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_of_840_and_1764_l2352_235240


namespace function_values_impossibility_l2352_235280

theorem function_values_impossibility (a b c : ℝ) (d : ℤ) :
  ¬∃ (m : ℝ), (a * m^3 + b * m - c / m + d = -1) ∧
              (a * (-m)^3 + b * (-m) - c / (-m) + d = 4) := by
  sorry

end function_values_impossibility_l2352_235280


namespace largest_constant_inequality_l2352_235217

theorem largest_constant_inequality (C : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z)) ↔ C ≤ Real.sqrt 6 := by
  sorry

end largest_constant_inequality_l2352_235217


namespace mrs_thomson_savings_l2352_235263

theorem mrs_thomson_savings (incentive : ℝ) (food_fraction : ℝ) (clothes_fraction : ℝ) (saved_amount : ℝ)
  (h1 : incentive = 240)
  (h2 : food_fraction = 1/3)
  (h3 : clothes_fraction = 1/5)
  (h4 : saved_amount = 84) :
  let remaining := incentive - (food_fraction * incentive) - (clothes_fraction * incentive)
  saved_amount / remaining = 3/4 := by
sorry

end mrs_thomson_savings_l2352_235263


namespace rectangular_to_polar_l2352_235221

theorem rectangular_to_polar :
  let x : ℝ := 2 * Real.sqrt 3
  let y : ℝ := -2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 ∧ y < 0 then 2 * π + Real.arctan (y / x) else Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = 4 ∧ θ = 11 * π / 6 := by
  sorry

end rectangular_to_polar_l2352_235221


namespace gcd_204_85_l2352_235224

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l2352_235224


namespace problem_statement_l2352_235283

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 * (b^2 + 1) + b * (b + 2*a) = 40)
  (h2 : a * (b + 1) + b = 8) : 
  1 / a^2 + 1 / b^2 = 8 := by
sorry

end problem_statement_l2352_235283


namespace rectangle_area_l2352_235208

theorem rectangle_area (x y : ℝ) 
  (h_perimeter : x + y = 5)
  (h_diagonal : x^2 + y^2 = 15) : 
  x * y = 5 := by
sorry

end rectangle_area_l2352_235208


namespace circle_radius_with_chords_l2352_235220

/-- A circle with three parallel chords -/
structure CircleWithChords where
  -- The radius of the circle
  radius : ℝ
  -- The distance from the center to the closest chord
  x : ℝ
  -- The common distance between the chords
  y : ℝ
  -- Conditions on the chords
  chord_condition : radius^2 = x^2 + 100 ∧ 
                    radius^2 = (x + y)^2 + 64 ∧ 
                    radius^2 = (x + 2*y)^2 + 16

/-- The theorem stating that the radius of the circle with the given chord configuration is 5√22/2 -/
theorem circle_radius_with_chords (c : CircleWithChords) : c.radius = 5 * Real.sqrt 22 / 2 := by
  sorry

end circle_radius_with_chords_l2352_235220


namespace carla_smoothie_cream_l2352_235247

/-- Given information about Carla's smoothie recipe, prove the amount of cream used. -/
theorem carla_smoothie_cream (watermelon_puree : ℕ) (num_servings : ℕ) (serving_size : ℕ) 
  (h1 : watermelon_puree = 500)
  (h2 : num_servings = 4)
  (h3 : serving_size = 150) :
  num_servings * serving_size - watermelon_puree = 100 := by
  sorry

#check carla_smoothie_cream

end carla_smoothie_cream_l2352_235247


namespace right_triangle_area_l2352_235297

/-- The area of a right triangle with hypotenuse 10√2 and one 45° angle is 50 square inches. -/
theorem right_triangle_area (h : ℝ) (α : ℝ) (A : ℝ) : 
  h = 10 * Real.sqrt 2 →  -- hypotenuse length
  α = 45 * π / 180 →      -- one angle in radians
  A = 50 →                -- area
  ∃ (a b : ℝ), 
    a^2 + b^2 = h^2 ∧     -- Pythagorean theorem
    Real.cos α = a / h ∧  -- cosine of the angle
    A = (1/2) * a * b     -- area formula
  := by sorry

end right_triangle_area_l2352_235297


namespace union_and_intersection_range_of_a_l2352_235209

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 5 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | a * x + 1 > 0}

-- Theorem for part (I)
theorem union_and_intersection :
  (A ∪ B = {x | 1 ≤ x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | 6 ≤ x ∧ x < 10}) := by sorry

-- Theorem for part (II)
theorem range_of_a :
  ∀ a : ℝ, (A ∩ C a = A) → a ∈ Set.Ici (-1/6) := by sorry

end union_and_intersection_range_of_a_l2352_235209
