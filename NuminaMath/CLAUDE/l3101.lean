import Mathlib

namespace vector_coordinates_l3101_310125

/-- Given two vectors a and b in ℝ², prove that if a is parallel to b, 
    a = (2, -1), and the magnitude of b is 2√5, then b is either (-4, 2) or (4, -2) -/
theorem vector_coordinates (a b : ℝ × ℝ) : 
  (∃ (k : ℝ), b = k • a) →  -- a is parallel to b
  a = (2, -1) →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 2 * Real.sqrt 5 →  -- magnitude of b is 2√5
  (b = (-4, 2) ∨ b = (4, -2)) :=
by sorry

end vector_coordinates_l3101_310125


namespace second_term_of_sequence_l3101_310144

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem second_term_of_sequence (a d : ℤ) :
  arithmetic_sequence a d 12 = 11 →
  arithmetic_sequence a d 13 = 14 →
  arithmetic_sequence a d 2 = -19 :=
by
  sorry

end second_term_of_sequence_l3101_310144


namespace expression_simplification_and_evaluation_expression_evaluation_at_3_l3101_310142

theorem expression_simplification_and_evaluation (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (1 - x / (x + 1)) / ((x^2 - 2*x + 1) / (x^2 - 1)) = 1 / (x - 1) :=
sorry

theorem expression_evaluation_at_3 :
  (1 - 3 / (3 + 1)) / ((3^2 - 2*3 + 1) / (3^2 - 1)) = 1 / 2 :=
sorry

end expression_simplification_and_evaluation_expression_evaluation_at_3_l3101_310142


namespace diameter_endpoints_form_trapezoid_l3101_310159

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (c₁ c₂ : Circle) (d₁ d₂ : Set (ℝ × ℝ)) : Prop :=
  -- Circles are external to each other
  let (x₁, y₁) := c₁.center
  let (x₂, y₂) := c₂.center
  (x₁ - x₂)^2 + (y₁ - y₂)^2 > (c₁.radius + c₂.radius)^2 ∧
  -- d₁ and d₂ are diameters of c₁ and c₂ respectively
  (∀ p ∈ d₁, dist p c₁.center ≤ c₁.radius) ∧
  (∀ p ∈ d₂, dist p c₂.center ≤ c₂.radius) ∧
  -- The line through one diameter is tangent to the other circle
  (∃ p ∈ d₁, dist p c₂.center = c₂.radius) ∧
  (∃ p ∈ d₂, dist p c₁.center = c₁.radius)

-- Define a trapezoid
def is_trapezoid (quadrilateral : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d : ℝ × ℝ), quadrilateral = {a, b, c, d} ∧
  (∃ (m : ℝ), (c.1 - d.1 = m * (a.1 - b.1) ∧ c.2 - d.2 = m * (a.2 - b.2)) ∨
              (b.1 - c.1 = m * (a.1 - d.1) ∧ b.2 - c.2 = m * (a.2 - d.2)))

-- Theorem statement
theorem diameter_endpoints_form_trapezoid (c₁ c₂ : Circle) (d₁ d₂ : Set (ℝ × ℝ)) :
  problem_setup c₁ c₂ d₁ d₂ →
  is_trapezoid (d₁ ∪ d₂) :=
sorry

end diameter_endpoints_form_trapezoid_l3101_310159


namespace abc_base16_to_base4_l3101_310174

/-- Converts a base 16 digit to its decimal representation -/
def hexToDecimal (x : Char) : ℕ :=
  match x with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | _ => 0  -- This case should not occur for our specific problem

/-- Converts a decimal number to its base 4 representation -/
def decimalToBase4 (x : ℕ) : List ℕ :=
  [x / 4, x % 4]

/-- Converts a base 16 number to base 4 -/
def hexToBase4 (x : String) : List ℕ :=
  x.data.map hexToDecimal |>.bind decimalToBase4

theorem abc_base16_to_base4 :
  hexToBase4 "ABC" = [2, 2, 2, 3, 3, 0] := by sorry

end abc_base16_to_base4_l3101_310174


namespace sqrt_64_times_sqrt_25_l3101_310161

theorem sqrt_64_times_sqrt_25 : Real.sqrt (64 * Real.sqrt 25) = 8 * Real.sqrt 5 := by
  sorry

end sqrt_64_times_sqrt_25_l3101_310161


namespace landscape_length_l3101_310157

/-- Given a rectangular landscape with a playground, calculate its length -/
theorem landscape_length (breadth : ℝ) (playground_area : ℝ) : 
  breadth > 0 →
  playground_area = 1200 →
  playground_area = (1 / 6) * (8 * breadth * breadth) →
  8 * breadth = 240 := by
  sorry

end landscape_length_l3101_310157


namespace n_value_l3101_310173

-- Define the cubic polynomial
def cubic_poly (x m : ℝ) : ℝ := x^3 - 3*x^2 + m*x + 24

-- Define the quadratic polynomial
def quad_poly (x n : ℝ) : ℝ := x^2 + n*x - 6

theorem n_value (a b c m n : ℝ) : 
  (cubic_poly a m = 0) ∧ 
  (cubic_poly b m = 0) ∧ 
  (cubic_poly c m = 0) ∧
  (quad_poly (-a) n = 0) ∧ 
  (quad_poly (-b) n = 0) →
  n = -1 := by
sorry

end n_value_l3101_310173


namespace solution_set_f_greater_than_2_range_of_m_l3101_310182

-- Define the function f(x) = |2x-1|
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -1/2 ∨ x > 3/2} := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x + 2 * |x + 3| - 4 > m * x) → m ≤ -11 := by sorry

end solution_set_f_greater_than_2_range_of_m_l3101_310182


namespace negation_of_absolute_value_less_than_zero_l3101_310176

theorem negation_of_absolute_value_less_than_zero :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x₀ : ℝ, |x₀| ≥ 0) := by sorry

end negation_of_absolute_value_less_than_zero_l3101_310176


namespace even_decreasing_implies_increasing_l3101_310177

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f y < f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem even_decreasing_implies_increasing
  (f : ℝ → ℝ) (h_even : is_even f) (h_decr : decreasing_on f (Set.Ici 0)) :
  increasing_on f (Set.Iic 0) :=
sorry

end even_decreasing_implies_increasing_l3101_310177


namespace translation_of_line_segment_l3101_310130

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

/-- Theorem: Translation of line segment AB to A'B' -/
theorem translation_of_line_segment (A B A' : Point) (t : Translation) :
  A.x = -2 ∧ A.y = 0 ∧
  B.x = 0 ∧ B.y = 3 ∧
  A'.x = 2 ∧ A'.y = 1 ∧
  A' = applyTranslation A t →
  applyTranslation B t = { x := 4, y := 4 } :=
by sorry

end translation_of_line_segment_l3101_310130


namespace quadratic_solution_l3101_310180

theorem quadratic_solution (x : ℝ) (h1 : x^2 - 3*x - 6 = 0) (h2 : x ≠ 0) :
  x = (3 + Real.sqrt 33) / 2 ∨ x = (3 - Real.sqrt 33) / 2 := by
  sorry

end quadratic_solution_l3101_310180


namespace line_equation_l3101_310141

/-- Given a line passing through (b, 0) and (0, h), forming a triangle with area T' in the second quadrant where b > 0, prove that the equation of the line is -2T'x + b²y + 2T'b = 0. -/
theorem line_equation (b T' : ℝ) (h : ℝ) (hb : b > 0) : 
  ∃ (x y : ℝ → ℝ), ∀ t, -2 * T' * x t + b^2 * y t + 2 * T' * b = 0 :=
by sorry

end line_equation_l3101_310141


namespace quadratic_function_sum_l3101_310195

theorem quadratic_function_sum (a b c : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + 2 ≤ a*x^2 + b*x + c) ∧
  (∀ x : ℝ, a*x^2 + b*x + c ≤ 2*x^2 - 4*x + 3) →
  a + b + c = 1 := by
  sorry

end quadratic_function_sum_l3101_310195


namespace bryden_quarter_sale_l3101_310179

/-- The amount a collector pays for state quarters as a percentage of face value -/
def collector_offer_percentage : ℚ := 2500

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 5

/-- The face value of a single state quarter in dollars -/
def quarter_face_value : ℚ := 1/4

/-- The amount Bryden will receive for his quarters in dollars -/
def bryden_received_amount : ℚ := 31.25

theorem bryden_quarter_sale :
  (collector_offer_percentage / 100) * (bryden_quarters : ℚ) * quarter_face_value = bryden_received_amount := by
  sorry

end bryden_quarter_sale_l3101_310179


namespace derivative_of_sin_over_x_l3101_310193

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / x

theorem derivative_of_sin_over_x :
  deriv f = fun x => (x * Real.cos x - Real.sin x) / (x^2) :=
sorry

end derivative_of_sin_over_x_l3101_310193


namespace calculation_proof_l3101_310117

theorem calculation_proof : 
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 10.5 = 10.5 := by
  sorry

end calculation_proof_l3101_310117


namespace cistern_fill_time_l3101_310198

-- Define the rates of the pipes
def rateA : ℚ := 1 / 12
def rateB : ℚ := 1 / 18
def rateC : ℚ := -(1 / 15)

-- Define the combined rate
def combinedRate : ℚ := rateA + rateB + rateC

-- Define the time to fill the cistern
def timeToFill : ℚ := 1 / combinedRate

-- Theorem statement
theorem cistern_fill_time :
  timeToFill = 180 / 13 :=
sorry

end cistern_fill_time_l3101_310198


namespace coefficient_of_x_cubed_is_94_l3101_310132

-- Define the polynomials
def p (x : ℝ) : ℝ := 3 * x^3 + 4 * x^2 + 5 * x + 6
def q (x : ℝ) : ℝ := 7 * x^2 + 8 * x + 9

-- Theorem statement
theorem coefficient_of_x_cubed_is_94 :
  ∃ a b c d e : ℝ, p * q = (λ x => a * x^5 + b * x^4 + 94 * x^3 + c * x^2 + d * x + e) :=
sorry

end coefficient_of_x_cubed_is_94_l3101_310132


namespace perfect_cube_factors_of_72_is_two_l3101_310119

/-- A function that returns the number of positive factors of 72 that are perfect cubes -/
def perfect_cube_factors_of_72 : ℕ :=
  -- The function should return the number of positive factors of 72 that are perfect cubes
  sorry

/-- Theorem stating that the number of positive factors of 72 that are perfect cubes is 2 -/
theorem perfect_cube_factors_of_72_is_two : perfect_cube_factors_of_72 = 2 := by
  sorry

end perfect_cube_factors_of_72_is_two_l3101_310119


namespace correct_total_amount_l3101_310108

/-- Calculate the total amount paid for grapes and mangoes -/
def totalAmountPaid (grapeQuantity : ℕ) (grapeRate : ℕ) (mangoQuantity : ℕ) (mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Theorem stating that the total amount paid is correct -/
theorem correct_total_amount :
  totalAmountPaid 10 70 9 55 = 1195 := by
  sorry

#eval totalAmountPaid 10 70 9 55

end correct_total_amount_l3101_310108


namespace rectangle_perimeter_l3101_310152

theorem rectangle_perimeter (length width : ℝ) : 
  length / width = 4 / 3 →
  length * width = 972 →
  2 * (length + width) = 126 :=
by sorry

end rectangle_perimeter_l3101_310152


namespace chris_bill_calculation_l3101_310158

/-- Calculates the total internet bill based on base charge, overage rate, and data usage over the limit. -/
def total_bill (base_charge : ℝ) (overage_rate : ℝ) (data_over_limit : ℝ) : ℝ :=
  base_charge + overage_rate * data_over_limit

/-- Theorem stating that Chris's total bill is equal to the sum of the base charge and overage charge. -/
theorem chris_bill_calculation (base_charge overage_rate data_over_limit : ℝ) :
  total_bill base_charge overage_rate data_over_limit = base_charge + overage_rate * data_over_limit :=
by sorry

end chris_bill_calculation_l3101_310158


namespace afternoon_rowers_count_l3101_310110

/-- The number of campers who went rowing in the afternoon -/
def afternoon_rowers (morning_rowers total_rowers : ℕ) : ℕ :=
  total_rowers - morning_rowers

/-- Proof that 21 campers went rowing in the afternoon -/
theorem afternoon_rowers_count :
  afternoon_rowers 13 34 = 21 := by
  sorry

end afternoon_rowers_count_l3101_310110


namespace point_on_transformed_plane_l3101_310112

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def applySimilarity (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem point_on_transformed_plane :
  let A : Point3D := ⟨1, 2, 2⟩
  let a : Plane := ⟨3, 0, -1, 5⟩
  let k : ℝ := -1/5
  let a' : Plane := applySimilarity a k
  pointOnPlane A a' := by sorry

end point_on_transformed_plane_l3101_310112


namespace train_platform_passing_time_l3101_310168

theorem train_platform_passing_time :
  let train_length : ℝ := 360
  let platform_length : ℝ := 390
  let train_speed_kmh : ℝ := 45
  let total_distance : ℝ := train_length + platform_length
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let time : ℝ := total_distance / train_speed_ms
  time = 60 := by sorry

end train_platform_passing_time_l3101_310168


namespace hari_contribution_is_8280_l3101_310121

/-- Represents the business partnership between Praveen and Hari -/
structure Partnership where
  praveen_initial : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's contribution to the partnership -/
def hari_contribution (p : Partnership) : ℕ :=
  (p.praveen_initial * p.praveen_months * p.profit_ratio_hari) / (p.hari_months * p.profit_ratio_praveen)

/-- Theorem stating Hari's contribution in the given scenario -/
theorem hari_contribution_is_8280 :
  let p : Partnership := {
    praveen_initial := 3220,
    praveen_months := 12,
    hari_months := 7,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  hari_contribution p = 8280 := by
  sorry

end hari_contribution_is_8280_l3101_310121


namespace roses_mother_age_l3101_310167

theorem roses_mother_age (rose_age mother_age : ℕ) : 
  rose_age = mother_age / 3 →
  rose_age + mother_age = 100 →
  mother_age = 75 := by
sorry

end roses_mother_age_l3101_310167


namespace officers_selection_count_l3101_310148

/-- The number of ways to choose officers from a club -/
def choose_officers (total_members : ℕ) (senior_members : ℕ) (positions : ℕ) : ℕ :=
  senior_members * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose officers under given conditions -/
theorem officers_selection_count :
  choose_officers 12 4 5 = 31680 := by
  sorry

end officers_selection_count_l3101_310148


namespace perpendicular_bisector_c_value_l3101_310100

/-- The perpendicular bisector of a line segment from (2, 5) to (8, 11) has equation 2x - y = c. -/
theorem perpendicular_bisector_c_value :
  ∃ (c : ℝ), 
    (∀ (x y : ℝ), (2 * x - y = c) ↔ 
      (x - 5)^2 + (y - 8)^2 = (5 - 2)^2 + (8 - 5)^2 ∧ 
      (x - 5) * (8 - 2) = -(y - 8) * (11 - 5)) → 
    c = 2 := by
  sorry

end perpendicular_bisector_c_value_l3101_310100


namespace books_per_child_l3101_310181

theorem books_per_child (num_children : ℕ) (teacher_books : ℕ) (total_books : ℕ) :
  num_children = 10 →
  teacher_books = 8 →
  total_books = 78 →
  ∃ (books_per_child : ℕ), books_per_child * num_children + teacher_books = total_books ∧ books_per_child = 7 :=
by sorry

end books_per_child_l3101_310181


namespace jasons_music_store_spending_l3101_310187

/-- The problem of calculating Jason's total spending at the music store -/
theorem jasons_music_store_spending
  (flute_cost : ℝ)
  (music_stand_cost : ℝ)
  (song_book_cost : ℝ)
  (h1 : flute_cost = 142.46)
  (h2 : music_stand_cost = 8.89)
  (h3 : song_book_cost = 7.00) :
  flute_cost + music_stand_cost + song_book_cost = 158.35 := by
  sorry

end jasons_music_store_spending_l3101_310187


namespace quadratic_equation_roots_ratio_l3101_310154

theorem quadratic_equation_roots_ratio (c : ℚ) : 
  (∃ x y : ℚ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + c = 0 ∧ y^2 + 10*y + c = 0) → 
  c = 75/4 := by
sorry

end quadratic_equation_roots_ratio_l3101_310154


namespace squares_in_4x2023_grid_l3101_310169

/-- The number of squares with vertices on grid points in a 4 x 2023 grid -/
def squaresInGrid (rows : ℕ) (cols : ℕ) : ℕ :=
  let type_a := rows * cols
  let type_b := (rows - 1) * (cols - 1)
  let type_c := (rows - 2) * (cols - 2)
  let type_d := (rows - 3) * (cols - 3)
  type_a + 2 * type_b + 3 * type_c + 4 * type_d

/-- Theorem stating that the number of squares in a 4 x 2023 grid is 40430 -/
theorem squares_in_4x2023_grid :
  squaresInGrid 4 2023 = 40430 := by
  sorry

end squares_in_4x2023_grid_l3101_310169


namespace product_of_squares_l3101_310196

theorem product_of_squares (x : ℝ) :
  (Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) →
  (7 + x) * (28 - x) = 529 := by
sorry

end product_of_squares_l3101_310196


namespace probability_of_one_each_l3101_310131

def drawer_contents : ℕ := 7

def total_items : ℕ := 4 * drawer_contents

def ways_to_select_one_of_each : ℕ := drawer_contents^4

def total_selections : ℕ := (total_items.choose 4)

theorem probability_of_one_each : 
  (ways_to_select_one_of_each : ℚ) / total_selections = 2401 / 20475 :=
by sorry

end probability_of_one_each_l3101_310131


namespace green_dots_third_row_l3101_310192

/-- Represents a sequence of rows with green dots -/
def GreenDotSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem green_dots_third_row
  (a : ℕ → ℕ)
  (seq : GreenDotSequence a)
  (h1 : a 1 = 3)
  (h2 : a 2 = 6)
  (h4 : a 4 = 12)
  (h5 : a 5 = 15) :
  a 3 = 9 := by
sorry

end green_dots_third_row_l3101_310192


namespace complete_square_quadratic_l3101_310172

theorem complete_square_quadratic (a b c : ℝ) (h : a = 1 ∧ b = 6 ∧ c = 5) :
  ∃ (k : ℝ), (x + k)^2 - (x^2 + b*x + c) = 4 := by
  sorry

end complete_square_quadratic_l3101_310172


namespace no_rational_solution_for_odd_coeff_quadratic_l3101_310186

theorem no_rational_solution_for_odd_coeff_quadratic
  (a b c : ℤ)
  (ha : Odd a)
  (hb : Odd b)
  (hc : Odd c) :
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end no_rational_solution_for_odd_coeff_quadratic_l3101_310186


namespace binomial_rv_unique_params_l3101_310107

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: For a binomial random variable ξ with E(ξ) = 5/3 and D(ξ) = 10/9, n = 5 and p = 1/3 -/
theorem binomial_rv_unique_params (ξ : BinomialRV) 
  (h_exp : expected_value ξ = 5/3)
  (h_var : variance ξ = 10/9) :
  ξ.n = 5 ∧ ξ.p = 1/3 := by
  sorry

end binomial_rv_unique_params_l3101_310107


namespace distance_height_relation_l3101_310146

/-- An equilateral triangle with an arbitrary line in its plane -/
structure TriangleWithLine where
  /-- The side length of the equilateral triangle -/
  side : ℝ
  /-- The height of the equilateral triangle -/
  height : ℝ
  /-- The distance from the first vertex to the line -/
  m : ℝ
  /-- The distance from the second vertex to the line -/
  n : ℝ
  /-- The distance from the third vertex to the line -/
  p : ℝ
  /-- The side length is positive -/
  side_pos : 0 < side
  /-- The height is related to the side length as in an equilateral triangle -/
  height_eq : height = (Real.sqrt 3 / 2) * side

/-- The main theorem stating the relationship between distances and height -/
theorem distance_height_relation (t : TriangleWithLine) :
  (t.m - t.n)^2 + (t.n - t.p)^2 + (t.p - t.m)^2 = 2 * t.height^2 := by
  sorry

end distance_height_relation_l3101_310146


namespace tic_tac_toe_rounds_difference_l3101_310122

theorem tic_tac_toe_rounds_difference 
  (total_rounds : ℕ) 
  (william_wins : ℕ) 
  (h1 : total_rounds = 15) 
  (h2 : william_wins = 10) 
  (h3 : william_wins > total_rounds - william_wins) : 
  william_wins - (total_rounds - william_wins) = 5 := by
sorry

end tic_tac_toe_rounds_difference_l3101_310122


namespace tangent_line_at_zero_range_of_b_when_a_zero_sum_of_squares_greater_than_e_l3101_310128

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - a * Real.sin x
def g (x : ℝ) : ℝ := b * Real.sqrt x

-- Tangent line equation
theorem tangent_line_at_zero :
  ∃ m c : ℝ, ∀ x : ℝ, m * x + c = (1 - a) * x + 1 :=
sorry

-- Range of b when a = 0
theorem range_of_b_when_a_zero (h : a = 0) :
  ∃ x > 0, f x = g x ↔ b ≥ Real.sqrt (2 * Real.exp 1) :=
sorry

-- Proof of a^2 + b^2 > e
theorem sum_of_squares_greater_than_e (h : ∃ x > 0, f x = g x) :
  a^2 + b^2 > Real.exp 1 :=
sorry

end tangent_line_at_zero_range_of_b_when_a_zero_sum_of_squares_greater_than_e_l3101_310128


namespace perfect_cube_property_l3101_310120

theorem perfect_cube_property (x y : ℕ+) (h : ∃ k : ℕ+, x * y^2 = k^3) :
  ∃ m : ℕ+, x^2 * y = m^3 := by
  sorry

end perfect_cube_property_l3101_310120


namespace expression_equality_l3101_310162

theorem expression_equality (n : ℕ) (h : n ≥ 1) :
  (5^(n+1) + 6^(n+2))^2 - (5^(n+1) - 6^(n+2))^2 = 144 * 30^(n+1) := by
  sorry

end expression_equality_l3101_310162


namespace books_taken_out_monday_l3101_310163

/-- The number of books taken out on Monday from a library -/
def books_taken_out (initial_books : ℕ) (books_returned : ℕ) (final_books : ℕ) : ℕ :=
  initial_books + books_returned - final_books

/-- Theorem stating that 124 books were taken out on Monday -/
theorem books_taken_out_monday : books_taken_out 336 22 234 = 124 := by
  sorry

end books_taken_out_monday_l3101_310163


namespace prob_less_than_three_heads_in_eight_flips_prob_less_than_three_heads_in_eight_flips_proof_l3101_310175

/-- The probability of getting fewer than 3 heads in 8 fair coin flips -/
theorem prob_less_than_three_heads_in_eight_flips : ℚ :=
  37 / 256

/-- Proof that the probability of getting fewer than 3 heads in 8 fair coin flips is 37/256 -/
theorem prob_less_than_three_heads_in_eight_flips_proof :
  prob_less_than_three_heads_in_eight_flips = 37 / 256 := by
  sorry


end prob_less_than_three_heads_in_eight_flips_prob_less_than_three_heads_in_eight_flips_proof_l3101_310175


namespace brady_record_theorem_l3101_310194

/-- The minimum average yards per game needed to beat the record -/
def min_avg_yards_per_game (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ) : ℚ :=
  (current_record + 1 - current_yards) / games_left

/-- Theorem stating the minimum average yards per game needed to beat the record -/
theorem brady_record_theorem (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ)
  (h1 : current_record = 5999)
  (h2 : current_yards = 4200)
  (h3 : games_left = 6) :
  min_avg_yards_per_game current_record current_yards games_left = 300 := by
  sorry

end brady_record_theorem_l3101_310194


namespace no_integer_solutions_for_equation_l3101_310126

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y z : ℤ), x^2 + y^2 = 4*z - 1 := by
  sorry

end no_integer_solutions_for_equation_l3101_310126


namespace missing_number_equation_l3101_310199

theorem missing_number_equation (x : ℤ) : 1234562 - 12 * x * 2 = 1234490 ↔ x = 3 := by
  sorry

end missing_number_equation_l3101_310199


namespace quadratic_roots_range_l3101_310188

theorem quadratic_roots_range (m : ℝ) : 
  (¬ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + m = 0 ∧ x₂^2 + 2*x₂ + m = 0) →
  (5 - 2*m > 1) →
  1 ≤ m ∧ m < 2 :=
by sorry

end quadratic_roots_range_l3101_310188


namespace simplify_expression_l3101_310135

theorem simplify_expression : (6 * 10^10) / (2 * 10^4) = 3000000 := by
  sorry

end simplify_expression_l3101_310135


namespace sad_probability_value_l3101_310104

/-- Represents a person in the company -/
inductive Person : Type
| boy : Fin 3 → Person
| girl : Fin 3 → Person

/-- Represents the love relation between people -/
def loves : Person → Person → Prop := sorry

/-- The sad circumstance where no one is loved by the one they love -/
def sad_circumstance (loves : Person → Person → Prop) : Prop :=
  ∀ p : Person, ∃ q : Person, loves p q ∧ ¬loves q p

/-- The total number of possible love arrangements -/
def total_arrangements : ℕ := 729

/-- The number of sad arrangements -/
def sad_arrangements : ℕ := 156

/-- The probability of the sad circumstance -/
def sad_probability : ℚ := sad_arrangements / total_arrangements

theorem sad_probability_value : sad_probability = 156 / 729 :=
sorry

end sad_probability_value_l3101_310104


namespace total_unread_books_is_17_l3101_310150

/-- Represents a book series with total books and read books -/
structure BookSeries where
  total : ℕ
  read : ℕ

/-- Calculates the number of unread books in a series -/
def unread_books (series : BookSeries) : ℕ :=
  series.total - series.read

/-- The three book series -/
def series1 : BookSeries := ⟨14, 8⟩
def series2 : BookSeries := ⟨10, 5⟩
def series3 : BookSeries := ⟨18, 12⟩

/-- Theorem stating that the total number of unread books is 17 -/
theorem total_unread_books_is_17 :
  unread_books series1 + unread_books series2 + unread_books series3 = 17 := by
  sorry

end total_unread_books_is_17_l3101_310150


namespace exists_right_triangles_form_consecutive_l3101_310106

/-- A right-angled triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  right_angle : a^2 + b^2 = c^2

/-- A triangle with consecutive natural number side lengths -/
structure ConsecutiveTriangle where
  n : ℕ
  sides : Fin 3 → ℕ
  consecutive : sides = fun i => 2*n + i.val - 1

theorem exists_right_triangles_form_consecutive (A : ℕ) :
  ∃ (t : ConsecutiveTriangle) (rt1 rt2 : RightTriangle),
    t.sides 0 = rt1.a + rt2.a ∧
    t.sides 1 = rt1.b ∧
    t.sides 1 = rt2.b ∧
    t.sides 2 = rt1.c + rt2.c ∧
    A = (t.sides 0 * t.sides 1) / 2 :=
sorry

end exists_right_triangles_form_consecutive_l3101_310106


namespace min_value_sin_function_l3101_310101

theorem min_value_sin_function : 
  ∀ x : ℝ, -Real.sin x ^ 3 - 2 * Real.sin x ≥ -3 := by
  sorry

end min_value_sin_function_l3101_310101


namespace geometric_sequence_general_term_l3101_310115

theorem geometric_sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 3 = 5/2 →
  S 3 = 15/2 →
  (∀ n, a n = 5/2) ∨ (∀ n, a n = 10 * (-1/2)^(n-1)) :=
sorry

end geometric_sequence_general_term_l3101_310115


namespace pool_filling_cost_l3101_310136

/-- Proves that the cost to fill a pool is $5 given the specified conditions -/
theorem pool_filling_cost (fill_time : ℕ) (flow_rate : ℕ) (water_cost : ℚ) : 
  fill_time = 50 → 
  flow_rate = 100 → 
  water_cost = 1 / 1000 → 
  (fill_time * flow_rate : ℚ) * water_cost = 5 := by
  sorry

#check pool_filling_cost

end pool_filling_cost_l3101_310136


namespace power_equation_implies_m_equals_one_l3101_310153

theorem power_equation_implies_m_equals_one (s m : ℕ) :
  (2^16) * (25^s) = 5 * (10^m) → m = 1 := by
  sorry

end power_equation_implies_m_equals_one_l3101_310153


namespace union_of_M_and_N_l3101_310147

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by
  sorry

end union_of_M_and_N_l3101_310147


namespace rotation_symmetry_l3101_310129

-- Define the directions
inductive Direction
  | Up
  | Down
  | Left
  | Right

-- Define a square configuration
def SquareConfig := List Direction

-- Define a rotation function
def rotate90Clockwise (config : SquareConfig) : SquareConfig :=
  match config with
  | [a, b, c, d] => [d, a, b, c]
  | _ => []  -- Return empty list for invalid configurations

-- Theorem statement
theorem rotation_symmetry (original : SquareConfig) :
  original = [Direction.Up, Direction.Right, Direction.Down, Direction.Left] →
  rotate90Clockwise original = [Direction.Right, Direction.Down, Direction.Left, Direction.Up] :=
by
  sorry


end rotation_symmetry_l3101_310129


namespace factorial_divisibility_l3101_310164

theorem factorial_divisibility (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 100) :
  ∃ k : ℕ, (Nat.factorial (n^2 + 1)) = k * (Nat.factorial n)^(n + 2) := by
  sorry

end factorial_divisibility_l3101_310164


namespace brandon_textbook_weight_l3101_310166

def jon_textbook_weights : List ℝ := [2, 8, 5, 9]

theorem brandon_textbook_weight (brandon_weight : ℝ) : 
  (List.sum jon_textbook_weights = 3 * brandon_weight) → brandon_weight = 8 := by
  sorry

end brandon_textbook_weight_l3101_310166


namespace geometry_theorem_l3101_310133

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)

-- Define the non-coincidence of lines and planes
variable (non_coincident_lines : Line → Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n l : Line)
variable (α β : Plane)

-- State the theorem
theorem geometry_theorem 
  (h_non_coincident_lines : non_coincident_lines m n l)
  (h_non_coincident_planes : non_coincident_planes α β) :
  (∀ (l m : Line) (α β : Plane),
    line_perpendicular_to_plane l α →
    line_perpendicular_to_plane m β →
    parallel l m →
    plane_parallel α β) ∧
  (∀ (m n : Line) (α β : Plane),
    plane_perpendicular α β →
    plane_intersection α β = m →
    line_in_plane n β →
    perpendicular n m →
    line_perpendicular_to_plane n α) :=
by sorry

end geometry_theorem_l3101_310133


namespace factorial_series_diverges_l3101_310189

/-- The series Σ(k!/(2^k)) for k from 1 to infinity -/
def factorial_series (k : ℕ) : ℚ := (Nat.factorial k : ℚ) / (2 ^ k : ℚ)

/-- The statement that the factorial series diverges -/
theorem factorial_series_diverges : ¬ Summable factorial_series := by
  sorry

end factorial_series_diverges_l3101_310189


namespace smallest_sum_of_two_primes_above_70_l3101_310190

theorem smallest_sum_of_two_primes_above_70 : 
  ∃ (p q : Nat), 
    Prime p ∧ 
    Prime q ∧ 
    p > 70 ∧ 
    q > 70 ∧ 
    p ≠ q ∧ 
    p + q = 144 ∧ 
    (∀ (r s : Nat), Prime r → Prime s → r > 70 → s > 70 → r ≠ s → r + s ≥ 144) := by
  sorry

end smallest_sum_of_two_primes_above_70_l3101_310190


namespace nail_salon_fingers_l3101_310116

theorem nail_salon_fingers (total_earnings : ℚ) (cost_per_manicure : ℚ) (total_fingers : ℕ) (non_clients : ℕ) :
  total_earnings = 200 →
  cost_per_manicure = 20 →
  total_fingers = 210 →
  non_clients = 11 →
  ∃ (fingers_per_person : ℕ), 
    fingers_per_person = 10 ∧
    (total_earnings / cost_per_manicure + non_clients : ℚ) * fingers_per_person = total_fingers := by
  sorry

end nail_salon_fingers_l3101_310116


namespace modulo_residue_problem_l3101_310165

theorem modulo_residue_problem : (348 + 8 * 58 + 9 * 195 + 6 * 29) % 19 = 5 := by
  sorry

end modulo_residue_problem_l3101_310165


namespace mary_milk_weight_l3101_310103

/-- Proves that the weight of milk Mary bought is 6 pounds -/
theorem mary_milk_weight (bag_capacity : ℕ) (green_beans_weight : ℕ) (remaining_capacity : ℕ) : 
  bag_capacity = 20 →
  green_beans_weight = 4 →
  remaining_capacity = 2 →
  6 = bag_capacity - remaining_capacity - (green_beans_weight + 2 * green_beans_weight) :=
by sorry

end mary_milk_weight_l3101_310103


namespace remainder_theorem_l3101_310138

theorem remainder_theorem (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end remainder_theorem_l3101_310138


namespace line_segment_length_l3101_310127

/-- The hyperbola C with equation x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The line l with equation y = 2√3x + m -/
def line (x y m : ℝ) : Prop := y = 2 * Real.sqrt 3 * x + m

/-- The right vertex of the hyperbola -/
def right_vertex (x y : ℝ) : Prop := hyperbola x y ∧ x > 0 ∧ y = 0

/-- The asymptotes of the hyperbola -/
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- The intersection points of the line and the asymptotes -/
def intersection_points (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  line x₁ y₁ m ∧ asymptote x₁ y₁ ∧
  line x₂ y₂ m ∧ asymptote x₂ y₂ ∧
  x₁ ≠ x₂

/-- The theorem statement -/
theorem line_segment_length 
  (x y m x₁ y₁ x₂ y₂ : ℝ) :
  right_vertex x y →
  line x y m →
  intersection_points x₁ y₁ x₂ y₂ m →
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 * Real.sqrt 13 / 3 :=
sorry

end line_segment_length_l3101_310127


namespace three_plants_three_colors_l3101_310118

/-- Represents the number of ways to assign plants to colored lamps -/
def plant_lamp_assignments (num_plants : ℕ) (num_identical_plants : ℕ) (num_colors : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of ways to assign 3 plants to 3 colors of lamps -/
theorem three_plants_three_colors :
  plant_lamp_assignments 3 2 3 = 27 := by
  sorry

end three_plants_three_colors_l3101_310118


namespace division_problem_l3101_310123

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 127 ∧ quotient = 9 ∧ remainder = 1 ∧ 
  dividend = divisor * quotient + remainder →
  divisor = 14 := by sorry

end division_problem_l3101_310123


namespace coefficient_x_squared_l3101_310185

theorem coefficient_x_squared (m n : ℕ+) : 
  (2 * m.val + 3 * n.val = 13) → 
  (∃ k, k = Nat.choose m.val 2 * 2^2 + Nat.choose n.val 2 * 3^2 ∧ (k = 31 ∨ k = 40)) :=
by sorry

end coefficient_x_squared_l3101_310185


namespace subtraction_problem_l3101_310156

theorem subtraction_problem (M N : ℕ) : 
  M < 10 → N < 10 → M * 10 + 4 - (30 + N) = 16 → M + N = 13 := by
sorry

end subtraction_problem_l3101_310156


namespace sanitizer_sprays_common_kill_percentage_l3101_310155

theorem sanitizer_sprays_common_kill_percentage 
  (spray1_kill : Real) 
  (spray2_kill : Real) 
  (combined_survival : Real) 
  (h1 : spray1_kill = 0.5) 
  (h2 : spray2_kill = 0.25) 
  (h3 : combined_survival = 0.3) : 
  spray1_kill + spray2_kill - (1 - combined_survival) = 0.05 := by
  sorry

end sanitizer_sprays_common_kill_percentage_l3101_310155


namespace quadratic_root_conditions_l3101_310184

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 + (a^2 + 1) * x + a - 2

-- Define the roots of the quadratic equation
def roots (a : ℝ) : Set ℝ := {x : ℝ | quadratic a x = 0}

-- Theorem statement
theorem quadratic_root_conditions (a : ℝ) :
  (∃ x ∈ roots a, x > 1) ∧ (∃ y ∈ roots a, y < -1) → 0 < a ∧ a < 2 :=
by sorry

end quadratic_root_conditions_l3101_310184


namespace expression_evaluation_l3101_310197

theorem expression_evaluation :
  let m : ℚ := -1/2
  let f (x : ℚ) := (5 / (x - 2) - x - 2) * ((2 * x - 4) / (3 - x))
  f m = 5 := by
  sorry

end expression_evaluation_l3101_310197


namespace smallest_odd_n_l3101_310160

def is_smallest_odd (n : ℕ) : Prop :=
  Odd n ∧ 
  (3 : ℝ) ^ ((n + 1)^2 / 5) > 500 ∧ 
  ∀ m : ℕ, Odd m ∧ m < n → (3 : ℝ) ^ ((m + 1)^2 / 5) ≤ 500

theorem smallest_odd_n : is_smallest_odd 6 := by sorry

end smallest_odd_n_l3101_310160


namespace complex_equation_product_l3101_310109

theorem complex_equation_product (x y : ℝ) : 
  (Complex.I : ℂ) * x - (Complex.I : ℂ) * y + x + y = 2 → x * y = 1 := by
sorry

end complex_equation_product_l3101_310109


namespace sqrt_sum_inequality_l3101_310183

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 21) :
  Real.sqrt a + Real.sqrt b < 2 * Real.sqrt 11 := by
sorry

end sqrt_sum_inequality_l3101_310183


namespace simplify_fraction_l3101_310134

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by sorry

end simplify_fraction_l3101_310134


namespace floor_of_neg_two_point_seven_l3101_310102

-- Define the greatest integer function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem floor_of_neg_two_point_seven :
  floor (-2.7) = -3 := by sorry

end floor_of_neg_two_point_seven_l3101_310102


namespace exactly_two_correct_statements_l3101_310139

theorem exactly_two_correct_statements : 
  let f : ℝ → ℝ := λ x => x + 1/x
  let statement1 := ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ (∃ (x₀ : ℝ), f x₀ = m) ∧ m = 2
  let statement2 := ∀ (a b : ℝ), a^2 + b^2 ≥ 2*a*b
  let statement3 := ∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d ∧ d > 0 → a*c > b*d
  let statement4 := (¬∃ (x : ℝ), x^2 + x + 1 ≥ 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0)
  let statement5 := ∀ (x y : ℝ), x > y ↔ 1/x < 1/y
  let statement6 := ∀ (p q : Prop), (¬(p ∨ q)) → (¬(¬p ∨ ¬q))
  (statement2 ∧ statement3 ∧ ¬statement1 ∧ ¬statement4 ∧ ¬statement5 ∧ ¬statement6) := by sorry

end exactly_two_correct_statements_l3101_310139


namespace logical_equivalence_l3101_310113

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ R) → ¬Q) ↔ (Q → (¬P ∨ ¬R)) := by sorry

end logical_equivalence_l3101_310113


namespace ellipse_foci_distance_l3101_310124

/-- The distance between the foci of the ellipse (x²/36) + (y²/9) = 9 is 2√3 -/
theorem ellipse_foci_distance :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 36 + y^2 / 9 = 9}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ 
    ∀ p ∈ ellipse, dist p f₁ + dist p f₂ = 2 * Real.sqrt 36 ∧
    dist f₁ f₂ = 2 * Real.sqrt 3 :=
by sorry

end ellipse_foci_distance_l3101_310124


namespace proportion_solution_l3101_310114

-- Define the conversion factor from minutes to seconds
def minutes_to_seconds (minutes : ℚ) : ℚ := 60 * minutes

-- Define the proportion
def proportion (x : ℚ) : Prop :=
  x / 4 = 8 / (minutes_to_seconds 4)

-- Theorem statement
theorem proportion_solution :
  ∃ (x : ℚ), proportion x ∧ x = 1 / 7.5 := by
  sorry

end proportion_solution_l3101_310114


namespace cookies_leftover_l3101_310145

/-- The number of cookies Amelia has -/
def ameliaCookies : ℕ := 52

/-- The number of cookies Benjamin has -/
def benjaminCookies : ℕ := 63

/-- The number of cookies Chloe has -/
def chloeCookies : ℕ := 25

/-- The number of cookies in each package -/
def packageSize : ℕ := 15

/-- The total number of cookies -/
def totalCookies : ℕ := ameliaCookies + benjaminCookies + chloeCookies

/-- The number of cookies left over after packaging -/
def leftoverCookies : ℕ := totalCookies % packageSize

theorem cookies_leftover : leftoverCookies = 5 := by
  sorry

end cookies_leftover_l3101_310145


namespace unique_intersection_characterization_l3101_310140

/-- A line that has only one common point (-1, -1) with the parabola y = 8x^2 + 10x + 1 -/
def uniqueIntersectionLine (f : ℝ → ℝ) : Prop :=
  (∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -1 ∧ p.2 = f p.1 ∧ p.2 = 8 * p.1^2 + 10 * p.1 + 1) ∧
  (∀ x : ℝ, f x = -6 * x - 7 ∨ (∀ y : ℝ, f y = -1))

/-- The theorem stating that a line has a unique intersection with the parabola
    if and only if it's either y = -6x - 7 or x = -1 -/
theorem unique_intersection_characterization :
  ∀ f : ℝ → ℝ, uniqueIntersectionLine f ↔ 
    (∀ x : ℝ, f x = -6 * x - 7) ∨ (∀ x : ℝ, f x = -1) :=
sorry

end unique_intersection_characterization_l3101_310140


namespace fraction_of_third_is_eighth_l3101_310143

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by sorry

end fraction_of_third_is_eighth_l3101_310143


namespace probability_theorem_l3101_310105

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  faces : Finset (Finset (Fin 5))
  num_faces : faces.card = 12

/-- Three distinct vertices of a regular dodecahedron -/
def ThreeVertices (d : RegularDodecahedron) := Finset (Fin 3)

/-- The probability that a plane determined by three randomly chosen distinct
    vertices of a regular dodecahedron contains points inside the dodecahedron -/
def probability_plane_intersects_interior (d : RegularDodecahedron) : ℚ :=
  1 - 1 / 9.5

/-- Theorem stating the probability of a plane determined by three randomly chosen
    distinct vertices of a regular dodecahedron containing points inside the dodecahedron -/
theorem probability_theorem (d : RegularDodecahedron) :
  probability_plane_intersects_interior d = 1 - 1 / 9.5 := by
  sorry

end probability_theorem_l3101_310105


namespace dinner_fraction_l3101_310178

theorem dinner_fraction (total_money : ℚ) (ice_cream_cost : ℚ) (money_left : ℚ) :
  total_money = 80 ∧ ice_cream_cost = 18 ∧ money_left = 2 →
  (total_money - ice_cream_cost - money_left) / total_money = 3/4 := by
  sorry

end dinner_fraction_l3101_310178


namespace w_coordinate_of_point_on_line_l3101_310170

/-- A 4D point -/
structure Point4D where
  x : ℝ
  y : ℝ
  z : ℝ
  w : ℝ

/-- Definition of the line passing through two points -/
def line_through (p q : Point4D) (t : ℝ) : Point4D :=
  { x := p.x + t * (q.x - p.x),
    y := p.y + t * (q.y - p.y),
    z := p.z + t * (q.z - p.z),
    w := p.w + t * (q.w - p.w) }

/-- The theorem to be proved -/
theorem w_coordinate_of_point_on_line : 
  let p1 : Point4D := {x := 3, y := 3, z := 2, w := 1}
  let p2 : Point4D := {x := 6, y := 2, z := 1, w := -1}
  ∃ t : ℝ, 
    let point := line_through p1 p2 t
    point.y = 4 ∧ point.w = 3 := by
  sorry

end w_coordinate_of_point_on_line_l3101_310170


namespace house_wall_planks_l3101_310151

theorem house_wall_planks (total_planks small_planks : ℕ) 
  (h1 : total_planks = 29)
  (h2 : small_planks = 17) :
  total_planks - small_planks = 12 := by
  sorry

end house_wall_planks_l3101_310151


namespace triangle_heights_l3101_310111

theorem triangle_heights (ha hb : ℝ) (d : ℕ) :
  ha = 3 →
  hb = 7 →
  (∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a * ha = b * hb ∧
    b * hb = c * d ∧
    a * ha = c * d ∧
    a + b > c ∧ a + c > b ∧ b + c > a) →
  d = 3 ∨ d = 4 ∨ d = 5 :=
by sorry

end triangle_heights_l3101_310111


namespace temperature_function_and_max_l3101_310149

-- Define the temperature function
def T (a b c d : ℝ) (t : ℝ) : ℝ := a * t^3 + b * t^2 + c * t + d

-- Define the derivative of the temperature function
def T_prime (a b c : ℝ) (t : ℝ) : ℝ := 3 * a * t^2 + 2 * b * t + c

-- State the theorem
theorem temperature_function_and_max (a b c d : ℝ) 
  (ha : a ≠ 0)
  (h1 : T a b c d (-4) = 8)
  (h2 : T a b c d 0 = 60)
  (h3 : T a b c d 1 = 58)
  (h4 : T_prime a b c (-4) = T_prime a b c 4) :
  (∃ (t : ℝ), t ≥ -2 ∧ t ≤ 2 ∧ 
    (∀ (s : ℝ), s ≥ -2 ∧ s ≤ 2 → T 1 0 (-3) 60 t ≥ T 1 0 (-3) 60 s) ∧
    T 1 0 (-3) 60 t = 62) ∧
  (∀ (t : ℝ), T 1 0 (-3) 60 t = t^3 - 3*t + 60) := by
  sorry


end temperature_function_and_max_l3101_310149


namespace systematic_sampling_theorem_l3101_310191

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalStudents : ℕ
  groupSize : ℕ
  numGroups : ℕ
  sampleSize : ℕ
  initialSample : ℕ
  initialGroup : ℕ

/-- Given a systematic sampling scheme, calculate the sample from a specific group -/
def sampleFromGroup (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.initialSample + s.groupSize * (group - s.initialGroup)

theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.totalStudents = 50)
  (h2 : s.groupSize = 5)
  (h3 : s.numGroups = 10)
  (h4 : s.sampleSize = 10)
  (h5 : s.initialSample = 12)
  (h6 : s.initialGroup = 3) :
  sampleFromGroup s 8 = 37 := by
  sorry

end systematic_sampling_theorem_l3101_310191


namespace sequences_properties_l3101_310171

/-- Definition of the first sequence -/
def seq1 (n : ℕ) : ℤ := (-2)^n

/-- Definition of the second sequence -/
def seq2 (m : ℕ) : ℤ := (-2)^(m-1)

/-- Definition of the third sequence -/
def seq3 (m : ℕ) : ℤ := (-2)^(m-1) - 1

/-- Theorem stating the properties of the sequences -/
theorem sequences_properties :
  (∀ n : ℕ, seq1 n = (-2)^n) ∧
  (∀ m : ℕ, seq3 m = seq2 m - 1) ∧
  (seq1 2019 + seq2 2019 + seq3 2019 = -1) :=
by sorry

end sequences_properties_l3101_310171


namespace largest_multiple_of_seven_as_sum_of_three_squares_l3101_310137

theorem largest_multiple_of_seven_as_sum_of_three_squares :
  ∃ n : ℕ, 
    (∃ a : ℕ, n = a^2 + (a+1)^2 + (a+2)^2) ∧ 
    7 ∣ n ∧
    n < 10000 ∧
    (∀ m : ℕ, (∃ b : ℕ, m = b^2 + (b+1)^2 + (b+2)^2) → 7 ∣ m → m < 10000 → m ≤ n) ∧
    n = 8750 :=
by
  sorry

end largest_multiple_of_seven_as_sum_of_three_squares_l3101_310137
