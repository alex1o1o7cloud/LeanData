import Mathlib

namespace carved_arc_angle_l3228_322899

/-- An equilateral triangle -/
structure EquilateralTriangle where
  height : ℝ
  height_pos : height > 0

/-- A circle rolling along the side of an equilateral triangle -/
structure RollingCircle (triangle : EquilateralTriangle) where
  radius : ℝ
  radius_eq_height : radius = triangle.height

/-- The arc carved out from the circle by the sides of the triangle -/
def carved_arc (triangle : EquilateralTriangle) (circle : RollingCircle triangle) : ℝ := sorry

/-- Theorem: The arc carved out from the circle subtends an angle of 60° at the center -/
theorem carved_arc_angle (triangle : EquilateralTriangle) (circle : RollingCircle triangle) :
  carved_arc triangle circle = 60 * π / 180 := by sorry

end carved_arc_angle_l3228_322899


namespace odd_function_a_value_l3228_322848

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_a_value
  (f : ℝ → ℝ)
  (h_odd : isOddFunction f)
  (h_neg : ∀ x, x < 0 → f x = x^2 + a*x)
  (h_f2 : f 2 = 6)
  (a : ℝ) :
  a = 5 := by
sorry

end odd_function_a_value_l3228_322848


namespace fraction_problem_l3228_322875

theorem fraction_problem (x : ℚ) : x * 45 - 5 = 10 → x = 1/3 := by
  sorry

end fraction_problem_l3228_322875


namespace triangle_angle_ratio_range_l3228_322833

theorem triangle_angle_ratio_range (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  A + B + C = π →
  S = (1/2) * b * c * Real.sin A →
  a^2 = 2*S + (b-c)^2 →
  1 - (1/2) * Real.sin A = (b^2 + c^2 - a^2) / (2*b*c) →
  ∃ (l u : ℝ), l = 2 * Real.sqrt 2 ∧ u = 59/15 ∧
    (∀ x, l ≤ x ∧ x < u ↔ 
      ∃ (B' C' : ℝ), 0 < B' ∧ B' < π/2 ∧ 0 < C' ∧ C' < π/2 ∧
        x = (2 * Real.sin B'^2 + Real.sin C'^2) / (Real.sin B' * Real.sin C')) :=
by sorry

end triangle_angle_ratio_range_l3228_322833


namespace embankment_height_bounds_l3228_322881

/-- Represents the properties of a trapezoidal embankment -/
structure Embankment where
  length : ℝ
  lower_base : ℝ
  slope_angle : ℝ
  volume_min : ℝ
  volume_max : ℝ

/-- Theorem stating the height bounds for the embankment -/
theorem embankment_height_bounds (e : Embankment)
  (h_length : e.length = 100)
  (h_lower_base : e.lower_base = 5)
  (h_slope_angle : e.slope_angle = π/4)
  (h_volume : e.volume_min = 400 ∧ e.volume_max = 500)
  (h_upper_base_min : ∀ b, b ≥ 2 → 
    400 ≤ 25 * (5^2 - b^2) ∧ 25 * (5^2 - b^2) ≤ 500) :
  ∃ (h : ℝ), 1 ≤ h ∧ h ≤ (5 - Real.sqrt 5) / 2 :=
by sorry

end embankment_height_bounds_l3228_322881


namespace BF_length_is_10_8_l3228_322835

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  right_angle_A : True  -- Represents the right angle at A
  right_angle_C : True  -- Represents the right angle at C
  E_on_AC : True  -- Represents that E is on AC
  F_on_AC : True  -- Represents that F is on AC
  DE_perp_AC : True  -- Represents that DE is perpendicular to AC
  BF_perp_AC : True  -- Represents that BF is perpendicular to AC
  AE_length : Real
  DE_length : Real
  CE_length : Real
  h_AE : AE_length = 4
  h_DE : DE_length = 6
  h_CE : CE_length = 8

/-- Calculate the length of BF in the given quadrilateral -/
def calculate_BF_length (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that BF length is 10.8 -/
theorem BF_length_is_10_8 (q : Quadrilateral) : calculate_BF_length q = 10.8 := by sorry

end BF_length_is_10_8_l3228_322835


namespace volleyball_teams_l3228_322885

theorem volleyball_teams (total_people : ℕ) (people_per_team : ℕ) (h1 : total_people = 6) (h2 : people_per_team = 2) :
  total_people / people_per_team = 3 := by
sorry

end volleyball_teams_l3228_322885


namespace prob_at_least_one_boy_and_girl_l3228_322811

def prob_boy_or_girl : ℚ := 1 / 2

def family_size : ℕ := 4

theorem prob_at_least_one_boy_and_girl :
  (1 : ℚ) - (prob_boy_or_girl ^ family_size + prob_boy_or_girl ^ family_size) = 7 / 8 := by
  sorry

end prob_at_least_one_boy_and_girl_l3228_322811


namespace number_of_girls_l3228_322816

theorem number_of_girls (total_children : Nat) (happy_children : Nat) (sad_children : Nat) 
  (neutral_children : Nat) (boys : Nat) (happy_boys : Nat) (sad_girls : Nat) (neutral_boys : Nat) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 16 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 4 →
  total_children = happy_children + sad_children + neutral_children →
  total_children - boys = 44 := by
  sorry

#check number_of_girls

end number_of_girls_l3228_322816


namespace calculate_a10_l3228_322888

/-- A sequence satisfying the given property -/
def special_sequence (a : ℕ+ → ℤ) : Prop :=
  ∀ (p q : ℕ+), a (p + q) = a p + a q

/-- The theorem to prove -/
theorem calculate_a10 (a : ℕ+ → ℤ) 
  (h1 : special_sequence a) 
  (h2 : a 2 = -6) : 
  a 10 = -30 := by
sorry

end calculate_a10_l3228_322888


namespace non_congruent_squares_count_l3228_322819

/-- Represents a square on a lattice grid -/
structure LatticeSquare where
  -- We'll represent a square by its side length and orientation
  side_length : ℕ
  is_rotated : Bool

/-- The size of the grid -/
def grid_size : ℕ := 6

/-- Counts the number of squares of a given side length on the grid -/
def count_squares (side_length : ℕ) : ℕ :=
  (grid_size - side_length) * (grid_size - side_length)

/-- Counts all non-congruent squares on the 6x6 grid -/
def count_all_squares : ℕ :=
  -- Count regular squares
  (count_squares 1) + (count_squares 2) + (count_squares 3) + 
  (count_squares 4) + (count_squares 5) +
  -- Count rotated squares (same formula as regular squares)
  (count_squares 1) + (count_squares 2) + (count_squares 3) + 
  (count_squares 4) + (count_squares 5)

/-- Theorem: The number of non-congruent squares on a 6x6 grid is 110 -/
theorem non_congruent_squares_count : count_all_squares = 110 := by
  sorry

end non_congruent_squares_count_l3228_322819


namespace consecutive_numbers_theorem_l3228_322891

theorem consecutive_numbers_theorem (n : ℕ) (avg : ℚ) (largest : ℕ) : 
  n > 0 ∧ 
  avg = 20 ∧ 
  largest = 23 ∧ 
  (↑largest - ↑(n - 1) + ↑largest) / 2 = avg → 
  n = 7 :=
by sorry

end consecutive_numbers_theorem_l3228_322891


namespace min_bushes_cover_alley_l3228_322804

/-- The length of the alley in meters -/
def alley_length : ℝ := 400

/-- The radius of scent spread for each lily of the valley bush in meters -/
def scent_radius : ℝ := 20

/-- The minimum number of bushes needed to cover the alley with scent -/
def min_bushes : ℕ := 10

/-- Theorem stating that the minimum number of bushes needed to cover the alley is correct -/
theorem min_bushes_cover_alley :
  ∀ (n : ℕ), n ≥ min_bushes → n * (2 * scent_radius) ≥ alley_length :=
by sorry

end min_bushes_cover_alley_l3228_322804


namespace simplify_fraction_product_l3228_322844

theorem simplify_fraction_product : 8 * (15 / 4) * (-40 / 45) = -2 / 3 := by
  sorry

end simplify_fraction_product_l3228_322844


namespace range_of_x_when_proposition_false_l3228_322870

theorem range_of_x_when_proposition_false (x : ℝ) :
  x^2 - 5*x + 4 ≤ 0 → 1 ≤ x ∧ x ≤ 4 := by
  sorry

end range_of_x_when_proposition_false_l3228_322870


namespace complex_imaginary_part_l3228_322828

theorem complex_imaginary_part (z : ℂ) : (3 - 4*I) * z = Complex.abs (4 + 3*I) → Complex.im z = 4/5 := by
  sorry

end complex_imaginary_part_l3228_322828


namespace citrus_yield_probability_l3228_322853

/-- Represents the yield recovery rates in the first year -/
def first_year_rates : List ℝ := [1.0, 0.9, 0.8]

/-- Represents the probabilities of yield recovery rates in the first year -/
def first_year_probs : List ℝ := [0.2, 0.4, 0.4]

/-- Represents the growth rates in the second year -/
def second_year_rates : List ℝ := [1.5, 1.25, 1.0]

/-- Represents the probabilities of growth rates in the second year -/
def second_year_probs : List ℝ := [0.3, 0.3, 0.4]

/-- Calculates the probability of reaching exactly the pre-disaster yield after two years -/
def probability_pre_disaster_yield (f_rates : List ℝ) (f_probs : List ℝ) (s_rates : List ℝ) (s_probs : List ℝ) : ℝ :=
  sorry

theorem citrus_yield_probability :
  probability_pre_disaster_yield first_year_rates first_year_probs second_year_rates second_year_probs = 0.2 := by
  sorry

end citrus_yield_probability_l3228_322853


namespace area_ratio_is_one_l3228_322809

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the right triangle PQR with given side lengths
def rightTrianglePQR : Triangle :=
  { P := (0, 15),
    Q := (0, 0),
    R := (20, 0) }

-- Define midpoints S and T
def S : ℝ × ℝ := (0, 7.5)
def T : ℝ × ℝ := (12.5, 12.5)

-- Define point Y as the intersection of RT and QS
def Y : ℝ × ℝ := sorry

-- Define the areas of quadrilateral PSYT and triangle QYR
def areaPSYT : ℝ := sorry
def areaQYR : ℝ := sorry

-- Theorem statement
theorem area_ratio_is_one :
  areaPSYT = areaQYR :=
sorry

end area_ratio_is_one_l3228_322809


namespace income_ratio_l3228_322867

/-- Proof of the ratio of monthly incomes --/
theorem income_ratio (c_income b_income a_annual_income : ℝ) 
  (hb : b_income = c_income * 1.12)
  (hc : c_income = 12000)
  (ha : a_annual_income = 403200.0000000001) :
  (a_annual_income / 12) / b_income = 2.5 := by
  sorry

end income_ratio_l3228_322867


namespace no_divisibility_l3228_322890

theorem no_divisibility (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) :
  ¬(d ∣ a^(2^n) + 1) := by
  sorry

end no_divisibility_l3228_322890


namespace largest_A_at_125_l3228_322858

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence A_k -/
def A (k : ℕ) : ℝ := binomial 500 k * (0.3 ^ k)

theorem largest_A_at_125 : 
  ∀ k ∈ Finset.range 501, A 125 ≥ A k :=
sorry

end largest_A_at_125_l3228_322858


namespace hyperbola_focal_length_l3228_322834

theorem hyperbola_focal_length (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 20 = 1 → y = 2*x) → 
  let c := Real.sqrt (a^2 + 20)
  2 * c = 10 := by sorry

end hyperbola_focal_length_l3228_322834


namespace line_through_parabola_vertex_l3228_322892

theorem line_through_parabola_vertex (a : ℝ) : 
  let line := fun x => x + a
  let parabola := fun x => x^2 + a^2
  let vertex_x := 0
  let vertex_y := parabola vertex_x
  (∃! (a1 a2 : ℝ), a1 ≠ a2 ∧ 
    line vertex_x = vertex_y ∧ 
    ∀ a', line vertex_x = vertex_y → (a' = a1 ∨ a' = a2)) := by
  sorry

end line_through_parabola_vertex_l3228_322892


namespace circle_C_equation_l3228_322893

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = 13

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  2*x - 7*y + 8 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (6, 0)
def point_B : ℝ × ℝ := (1, 5)

-- Theorem statement
theorem circle_C_equation :
  ∀ (x y : ℝ),
    (∃ (cx cy : ℝ), line_l cx cy ∧ 
      (x - cx)^2 + (y - cy)^2 = (point_A.1 - cx)^2 + (point_A.2 - cy)^2 ∧
      (x - cx)^2 + (y - cy)^2 = (point_B.1 - cx)^2 + (point_B.2 - cy)^2) →
    circle_C x y :=
by
  sorry

end circle_C_equation_l3228_322893


namespace homework_theorem_l3228_322825

def homework_problem (total_time math_percentage other_time : ℝ) : Prop :=
  let math_time := math_percentage * total_time
  let science_time := total_time - math_time - other_time
  (science_time / total_time) * 100 = 40

theorem homework_theorem :
  ∀ (total_time math_percentage other_time : ℝ),
    total_time = 150 →
    math_percentage = 0.3 →
    other_time = 45 →
    homework_problem total_time math_percentage other_time :=
by sorry

end homework_theorem_l3228_322825


namespace expression_undefined_l3228_322884

theorem expression_undefined (a : ℝ) : 
  ¬∃x, x = (a + 3) / (a^2 - 9*a + 20) ↔ a = 4 ∨ a = 5 := by
  sorry

end expression_undefined_l3228_322884


namespace specific_tile_arrangement_l3228_322815

/-- The number of distinguishable arrangements for a row of tiles -/
def tileArrangements (brown purple green yellow blue : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow + blue) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green *
   Nat.factorial yellow * Nat.factorial blue)

/-- Theorem: The number of distinguishable arrangements for a row consisting of
    1 brown tile, 1 purple tile, 3 green tiles, 3 yellow tiles, and 2 blue tiles
    is equal to 50400. -/
theorem specific_tile_arrangement :
  tileArrangements 1 1 3 3 2 = 50400 := by
  sorry

end specific_tile_arrangement_l3228_322815


namespace initial_disappearance_percentage_l3228_322872

/-- Represents the population changes in a village --/
def village_population (initial_population : ℕ) (final_population : ℕ) (panic_exodus_percent : ℚ) : Prop :=
  ∃ (initial_disappearance_percent : ℚ),
    final_population = initial_population * (1 - initial_disappearance_percent / 100) * (1 - panic_exodus_percent / 100) ∧
    initial_disappearance_percent = 10

/-- Theorem stating the initial disappearance percentage in the village --/
theorem initial_disappearance_percentage :
  village_population 7800 5265 25 := by sorry

end initial_disappearance_percentage_l3228_322872


namespace golden_delicious_per_pint_l3228_322838

/-- The number of pink lady apples required to make one pint of cider -/
def pink_lady_per_pint : ℕ := 40

/-- The number of apples a farmhand can pick per hour -/
def apples_per_hour : ℕ := 240

/-- The number of farmhands -/
def num_farmhands : ℕ := 6

/-- The number of hours worked -/
def hours_worked : ℕ := 5

/-- The ratio of golden delicious to pink lady apples -/
def apple_ratio : ℚ := 1 / 3

/-- The number of pints of cider that can be made -/
def pints_of_cider : ℕ := 120

theorem golden_delicious_per_pint : ℕ := by
  sorry

end golden_delicious_per_pint_l3228_322838


namespace walters_coins_value_l3228_322810

/-- Represents the value of a coin in cents -/
def coin_value : String → Nat
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "half_dollar" => 50
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes half_dollars : Nat) : Nat :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime" +
  half_dollars * coin_value "half_dollar"

/-- Converts a number of cents to a percentage of a dollar -/
def cents_to_percentage (cents : Nat) : Nat :=
  cents

theorem walters_coins_value :
  total_value 2 1 2 1 = 77 ∧ cents_to_percentage (total_value 2 1 2 1) = 77 := by
  sorry

end walters_coins_value_l3228_322810


namespace triangle_side_length_l3228_322864

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 2 →
  b = Real.sqrt 6 →
  A = π / 6 →
  c^2 - 2 * Real.sqrt 6 * c * Real.cos A + 2 = 6 →
  c = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l3228_322864


namespace percentage_subtraction_equivalence_l3228_322883

theorem percentage_subtraction_equivalence :
  ∀ (a : ℝ), a - (0.07 * a) = 0.93 * a :=
by
  sorry

end percentage_subtraction_equivalence_l3228_322883


namespace journey_time_calculation_l3228_322855

theorem journey_time_calculation (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
    (h1 : total_distance = 560)
    (h2 : speed1 = 21)
    (h3 : speed2 = 24) : 
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 25 := by
  sorry

end journey_time_calculation_l3228_322855


namespace expression_value_l3228_322861

theorem expression_value : 
  let x : ℝ := 4
  let y : ℝ := -3
  let z : ℝ := 5
  x^2 + y^2 - z^2 + 2*y*z = -30 :=
by sorry

end expression_value_l3228_322861


namespace job_completion_time_l3228_322894

theorem job_completion_time 
  (initial_men : ℕ) 
  (initial_days : ℕ) 
  (new_men : ℕ) 
  (prep_days : ℕ) 
  (h1 : initial_men = 10) 
  (h2 : initial_days = 15) 
  (h3 : new_men = 15) 
  (h4 : prep_days = 2) : 
  (initial_men * initial_days) / new_men + prep_days = 12 := by
sorry

end job_completion_time_l3228_322894


namespace solve_linear_equation_l3228_322849

theorem solve_linear_equation (x : ℚ) (h : 3 * x - 4 = -2 * x + 11) : x = 3 := by
  sorry

end solve_linear_equation_l3228_322849


namespace power_sixteen_div_sixteen_squared_l3228_322812

theorem power_sixteen_div_sixteen_squared : 2^16 / 16^2 = 256 := by
  sorry

end power_sixteen_div_sixteen_squared_l3228_322812


namespace ivan_milkshake_cost_l3228_322873

/-- The cost of Ivan's milkshake -/
def milkshake_cost (initial_amount : ℚ) (cupcake_fraction : ℚ) (final_amount : ℚ) : ℚ :=
  initial_amount - initial_amount * cupcake_fraction - final_amount

/-- Theorem: The cost of Ivan's milkshake is $5 -/
theorem ivan_milkshake_cost :
  milkshake_cost 10 (1/5) 3 = 5 := by
  sorry

end ivan_milkshake_cost_l3228_322873


namespace candy_mixture_problem_l3228_322882

theorem candy_mixture_problem (X Y : ℝ) : 
  X + Y = 10 →
  3.50 * X + 4.30 * Y = 40 →
  Y = 6.25 := by
sorry

end candy_mixture_problem_l3228_322882


namespace arithmetic_computation_l3228_322845

theorem arithmetic_computation : 12 + 4 * (5 - 9)^2 / 2 = 44 := by sorry

end arithmetic_computation_l3228_322845


namespace triangle_proof_l3228_322852

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, BD = b and cos(ABC) = 7/12 -/
theorem triangle_proof (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  b^2 = a * c →
  D.1 ≥ 0 ∧ D.1 ≤ c →  -- D lies on AC
  b * Real.sin B = a * Real.sin C →
  2 * (c - D.1) = D.1 →  -- AD = 2DC
  (b = Real.sqrt (a * c)) ∧
  (Real.cos B = 7 / 12) :=
by sorry

end triangle_proof_l3228_322852


namespace distribution_property_l3228_322851

-- Define a type for our distribution
def Distribution (α : Type*) := α → ℝ

-- Define properties of our distribution
def IsSymmetric (f : Distribution ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def StandardDeviationProperty (f : Distribution ℝ) (a d : ℝ) : Prop :=
  ∫ x in Set.Icc (a - d) (a + d), f x = 0.68

-- Main theorem
theorem distribution_property (f : Distribution ℝ) (a d : ℝ) 
  (h_symmetric : IsSymmetric f a) 
  (h_std_dev : StandardDeviationProperty f a d) :
  ∫ x in Set.Iic (a + d), f x = 0.84 := by
  sorry

end distribution_property_l3228_322851


namespace solve_equation_l3228_322897

theorem solve_equation :
  ∃! y : ℚ, 2 * y + 3 * y = 200 - (4 * y + 10 * y / 2) ∧ y = 100 / 7 := by
  sorry

end solve_equation_l3228_322897


namespace sum_of_coefficients_l3228_322887

-- Define the polynomial g(x)
def g (p q r s : ℝ) (x : ℂ) : ℂ := x^4 + p*x^3 + q*x^2 + r*x + s

-- State the theorem
theorem sum_of_coefficients (p q r s : ℝ) :
  (g p q r s (3*I) = 0) →
  (g p q r s (1 + 3*I) = 0) →
  p + q + r + s = 89 :=
by sorry

end sum_of_coefficients_l3228_322887


namespace largest_prime_divisor_of_sum_of_squares_l3228_322868

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, 
    Nat.Prime p ∧ 
    p ∣ (36^2 + 45^2) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (36^2 + 45^2) → q ≤ p :=
by
  -- The proof goes here
  sorry

end largest_prime_divisor_of_sum_of_squares_l3228_322868


namespace marias_age_l3228_322808

theorem marias_age (cooper dante maria : ℕ) 
  (sum_ages : cooper + dante + maria = 31)
  (dante_twice_cooper : dante = 2 * cooper)
  (dante_younger_maria : dante = maria - 1) : 
  maria = 13 := by
  sorry

end marias_age_l3228_322808


namespace only_statement3_correct_l3228_322859

-- Define even and odd functions
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the statements
def Statement1 : Prop := ∀ f : ℝ → ℝ, EvenFunction f → ∃ y, f 0 = y
def Statement2 : Prop := ∀ f : ℝ → ℝ, OddFunction f → f 0 = 0
def Statement3 : Prop := ∀ f : ℝ → ℝ, EvenFunction f → ∀ x, f x = f (-x)
def Statement4 : Prop := ∀ f : ℝ → ℝ, (EvenFunction f ∧ OddFunction f) → ∀ x, f x = 0

-- Theorem stating that only Statement3 is correct
theorem only_statement3_correct :
  ¬Statement1 ∧ ¬Statement2 ∧ Statement3 ∧ ¬Statement4 :=
sorry

end only_statement3_correct_l3228_322859


namespace reciprocal_inequality_l3228_322821

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end reciprocal_inequality_l3228_322821


namespace cube_space_division_theorem_l3228_322865

/-- The number of parts that space is divided into by the planes containing the faces of a cube -/
def cube_space_division : ℕ := 33

/-- The number of faces a cube has -/
def cube_faces : ℕ := 6

/-- Theorem stating that the planes containing the faces of a cube divide space into 33 parts -/
theorem cube_space_division_theorem :
  cube_space_division = 33 ∧ cube_faces = 6 :=
sorry

end cube_space_division_theorem_l3228_322865


namespace a_value_l3228_322886

-- Define the system of inequalities
def system (x a : ℝ) : Prop :=
  3 * x + a < 0 ∧ 2 * x + 7 > 4 * x - 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 0

-- Theorem statement
theorem a_value (a : ℝ) :
  (∀ x, system x a ↔ solution_set x) → a = 0 := by
  sorry

end a_value_l3228_322886


namespace quadratic_roots_ratio_l3228_322832

theorem quadratic_roots_ratio (q : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r + s = -8 ∧ r * s = q ∧ 
   ∀ x : ℝ, x^2 + 8*x + q = 0 ↔ (x = r ∨ x = s)) → 
  q = 12 := by
sorry

end quadratic_roots_ratio_l3228_322832


namespace stock_price_loss_l3228_322843

theorem stock_price_loss (n : ℕ) (P : ℝ) (h : P > 0) : 
  P * (1.1 ^ n) * (0.9 ^ n) < P := by
  sorry

#check stock_price_loss

end stock_price_loss_l3228_322843


namespace ab_equals_seventeen_l3228_322839

theorem ab_equals_seventeen
  (h1 : a - b = 5)
  (h2 : a^2 + b^2 = 34)
  (h3 : a^3 - b^3 = 30)
  (h4 : a^2 + b^2 - c^2 = 50)
  (h5 : c = 2*a - b)
  : a * b = 17 := by
  sorry

end ab_equals_seventeen_l3228_322839


namespace composition_equation_solution_l3228_322836

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 5 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 7 * x + 6
  ∃ x : ℝ, δ (φ x) = -4 ∧ x = -43/35 := by
  sorry

end composition_equation_solution_l3228_322836


namespace uranium_conductivity_is_deductive_reasoning_l3228_322818

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)

-- Define uranium as a constant in our universe
variable (uranium : U)

-- Define what deductive reasoning is
def is_deductive_reasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

-- State the theorem
theorem uranium_conductivity_is_deductive_reasoning :
  is_deductive_reasoning
    (∀ x : U, Metal x → ConductsElectricity x)
    (Metal uranium)
    (ConductsElectricity uranium) :=
by
  sorry


end uranium_conductivity_is_deductive_reasoning_l3228_322818


namespace large_cube_volume_l3228_322820

theorem large_cube_volume (die_surface_area : ℝ) (h : die_surface_area = 96) :
  let die_face_area := die_surface_area / 6
  let large_cube_face_area := 4 * die_face_area
  let large_cube_side_length := Real.sqrt large_cube_face_area
  large_cube_side_length ^ 3 = 512 := by
  sorry

end large_cube_volume_l3228_322820


namespace segment_length_l3228_322830

/-- Given 5 points on a line, prove that PQ = 11 -/
theorem segment_length (P Q R S T : ℝ) : 
  P < Q ∧ Q < R ∧ R < S ∧ S < T →
  (Q - P) + (R - P) + (S - P) + (T - P) = 67 →
  (Q - P) + (R - Q) + (S - Q) + (T - Q) = 34 →
  Q - P = 11 := by
  sorry

end segment_length_l3228_322830


namespace complex_set_characterization_l3228_322869

theorem complex_set_characterization (z : ℂ) :
  (z - 1)^2 = Complex.abs (z - 1)^2 ↔ z.im = 0 :=
sorry

end complex_set_characterization_l3228_322869


namespace common_point_in_intervals_l3228_322863

theorem common_point_in_intervals (n : ℕ) (a b : Fin n → ℝ) 
  (h_closed : ∀ i, a i ≤ b i) 
  (h_intersect : ∀ i j, ∃ x, a i ≤ x ∧ x ≤ b i ∧ a j ≤ x ∧ x ≤ b j) : 
  ∃ p, ∀ i, a i ≤ p ∧ p ≤ b i :=
sorry

end common_point_in_intervals_l3228_322863


namespace spinner_probability_l3228_322889

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 := by
  sorry

end spinner_probability_l3228_322889


namespace max_sum_of_digits_l3228_322874

/-- A function that checks if a number is a digit (0-9) -/
def isDigit (n : ℕ) : Prop := n ≤ 9

/-- A function that checks if four numbers are distinct -/
def areDistinct (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem max_sum_of_digits (A B C D : ℕ) :
  isDigit A → isDigit B → isDigit C → isDigit D →
  areDistinct A B C D →
  A + B + C + D = 17 →
  (A + B) % (C + D) = 0 →
  A + B ≤ 16 ∧ ∃ (A' B' C' D' : ℕ), 
    isDigit A' ∧ isDigit B' ∧ isDigit C' ∧ isDigit D' ∧
    areDistinct A' B' C' D' ∧
    A' + B' + C' + D' = 17 ∧
    (A' + B') % (C' + D') = 0 ∧
    A' + B' = 16 :=
by sorry

end max_sum_of_digits_l3228_322874


namespace unique_distance_l3228_322806

def is_valid_distance (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  (n / 10) = n - ((n % 10) * 10 + (n / 10))

theorem unique_distance : ∃! n : ℕ, is_valid_distance n ∧ n = 98 :=
sorry

end unique_distance_l3228_322806


namespace ninth_grade_math_only_l3228_322898

theorem ninth_grade_math_only (total : ℕ) (math science foreign : ℕ) 
  (h_total : total = 120)
  (h_math : math = 85)
  (h_science : science = 70)
  (h_foreign : foreign = 54) :
  ∃ (math_science math_foreign science_foreign math_science_foreign : ℕ),
    math_science + math_foreign + science_foreign - math_science_foreign ≤ math ∧
    math_science + math_foreign + science_foreign - math_science_foreign ≤ science ∧
    math_science + math_foreign + science_foreign - math_science_foreign ≤ foreign ∧
    total = math + science + foreign - math_science - math_foreign - science_foreign + math_science_foreign ∧
    math - (math_science + math_foreign - math_science_foreign) = 45 := by
  sorry

end ninth_grade_math_only_l3228_322898


namespace apples_found_l3228_322878

def initial_apples : ℕ := 7
def final_apples : ℕ := 81

theorem apples_found (found : ℕ) : found = final_apples - initial_apples := by
  sorry

end apples_found_l3228_322878


namespace elderly_selected_l3228_322871

/-- Given a population with the following properties:
  - Total population of 1500
  - Divided into three equal groups (children, elderly, middle-aged)
  - 60 people are selected using stratified sampling
  This theorem proves that the number of elderly people selected is 20. -/
theorem elderly_selected (total_population : ℕ) (sample_size : ℕ) (num_groups : ℕ) :
  total_population = 1500 →
  sample_size = 60 →
  num_groups = 3 →
  (total_population / num_groups : ℚ) * (sample_size / total_population : ℚ) = 20 := by
  sorry

end elderly_selected_l3228_322871


namespace unique_solution_for_P_squared_prime_l3228_322807

/-- The polynomial P(n) = n^3 - n^2 - 5n + 2 -/
def P (n : ℤ) : ℤ := n^3 - n^2 - 5*n + 2

/-- A predicate to check if a number is prime -/
def isPrime (p : ℤ) : Prop := Nat.Prime p.natAbs

theorem unique_solution_for_P_squared_prime :
  ∃! n : ℤ, ∃ p : ℤ, isPrime p ∧ (P n)^2 = p^2 ∧ n = -3 :=
sorry

end unique_solution_for_P_squared_prime_l3228_322807


namespace tommy_order_cost_and_percentages_l3228_322831

/-- Represents the weight of each fruit in kilograms -/
structure FruitOrder where
  apples : ℝ
  oranges : ℝ
  grapes : ℝ
  strawberries : ℝ
  bananas : ℝ
  pineapples : ℝ

/-- Represents the price of each fruit per kilogram -/
structure FruitPrices where
  apples : ℝ
  oranges : ℝ
  grapes : ℝ
  strawberries : ℝ
  bananas : ℝ
  pineapples : ℝ

def totalWeight (order : FruitOrder) : ℝ :=
  order.apples + order.oranges + order.grapes + order.strawberries + order.bananas + order.pineapples

def totalCost (order : FruitOrder) (prices : FruitPrices) : ℝ :=
  order.apples * prices.apples +
  order.oranges * prices.oranges +
  order.grapes * prices.grapes +
  order.strawberries * prices.strawberries +
  order.bananas * prices.bananas +
  order.pineapples * prices.pineapples

theorem tommy_order_cost_and_percentages 
  (order : FruitOrder)
  (prices : FruitPrices)
  (h1 : totalWeight order = 20)
  (h2 : order.apples = 4)
  (h3 : order.oranges = 2)
  (h4 : order.grapes = 4)
  (h5 : order.strawberries = 3)
  (h6 : order.bananas = 1)
  (h7 : order.pineapples = 3)
  (h8 : prices.apples = 2)
  (h9 : prices.oranges = 3)
  (h10 : prices.grapes = 2.5)
  (h11 : prices.strawberries = 4)
  (h12 : prices.bananas = 1.5)
  (h13 : prices.pineapples = 3.5) :
  totalCost order prices = 48 ∧
  order.apples / totalWeight order = 0.2 ∧
  order.oranges / totalWeight order = 0.1 ∧
  order.grapes / totalWeight order = 0.2 ∧
  order.strawberries / totalWeight order = 0.15 ∧
  order.bananas / totalWeight order = 0.05 ∧
  order.pineapples / totalWeight order = 0.15 := by
  sorry


end tommy_order_cost_and_percentages_l3228_322831


namespace simplify_cube_roots_l3228_322854

theorem simplify_cube_roots : 
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 28 ^ (1/3) * 4 ^ (1/3) := by
  sorry

end simplify_cube_roots_l3228_322854


namespace complex_real_condition_l3228_322896

theorem complex_real_condition (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (2 - Complex.I)
  (∃ (x : ℝ), z = x) → a = -2 := by
sorry

end complex_real_condition_l3228_322896


namespace power_sum_equals_two_l3228_322841

theorem power_sum_equals_two : (-1 : ℝ)^2 + (1/3 : ℝ)^0 = 2 := by
  sorry

end power_sum_equals_two_l3228_322841


namespace remaining_pie_portion_l3228_322895

theorem remaining_pie_portion (carlos_share maria_fraction : ℝ) : 
  carlos_share = 0.6 →
  maria_fraction = 0.25 →
  (1 - carlos_share) * (1 - maria_fraction) = 0.3 := by
  sorry

end remaining_pie_portion_l3228_322895


namespace cylinder_not_triangular_front_view_l3228_322840

-- Define a spatial geometric body
structure SpatialBody where
  name : String

-- Define the front view of a spatial body
inductive FrontView
  | Triangle
  | Rectangle
  | Other

-- Define a function that returns the front view of a spatial body
def frontViewOf (body : SpatialBody) : FrontView :=
  sorry

-- Define a cylinder
def cylinder : SpatialBody :=
  { name := "Cylinder" }

-- Theorem: A cylinder cannot have a triangular front view
theorem cylinder_not_triangular_front_view :
  frontViewOf cylinder ≠ FrontView.Triangle :=
sorry

end cylinder_not_triangular_front_view_l3228_322840


namespace geometry_relations_l3228_322800

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : contains β m) :
  (parallel α β → perpendicular_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) ∧
  ¬(perpendicular_planes α β → parallel_lines l m) ∧
  ¬(perpendicular_lines l m → parallel α β) :=
by sorry

end geometry_relations_l3228_322800


namespace digital_earth_storage_technologies_l3228_322802

-- Define the set of all possible technologies
inductive Technology
| Nano
| LaserHolographic
| Protein
| Distributed
| Virtual
| Spatial
| Visualization

-- Define the property of contributing to digital Earth data storage
def contributesToDigitalEarthStorage (tech : Technology) : Prop :=
  match tech with
  | Technology.Nano => true
  | Technology.LaserHolographic => true
  | Technology.Protein => true
  | Technology.Distributed => true
  | _ => false

-- Define the set of technologies that contribute to digital Earth storage
def contributingTechnologies : Set Technology :=
  {tech | contributesToDigitalEarthStorage tech}

-- Theorem statement
theorem digital_earth_storage_technologies :
  contributingTechnologies = {Technology.Nano, Technology.LaserHolographic, Technology.Protein, Technology.Distributed} :=
by sorry

end digital_earth_storage_technologies_l3228_322802


namespace height_difference_l3228_322879

/-- Given three people A, B, and C, where A's height is 30% less than B's,
    and C's height is 20% more than A's, prove that the percentage difference
    between B's height and C's height is 16%. -/
theorem height_difference (h_b : ℝ) (h_b_pos : h_b > 0) : 
  let h_a := 0.7 * h_b
  let h_c := 1.2 * h_a
  ((h_b - h_c) / h_b) * 100 = 16 := by sorry

end height_difference_l3228_322879


namespace band_arrangement_l3228_322823

theorem band_arrangement (total_members : Nat) (min_row : Nat) (max_row : Nat) : 
  total_members = 108 → min_row = 10 → max_row = 18 → 
  (∃! n : Nat, n = (Finset.filter (λ x : Nat => min_row ≤ x ∧ x ≤ max_row ∧ total_members % x = 0) 
    (Finset.range (max_row - min_row + 1))).card ∧ n = 2) := by
  sorry

end band_arrangement_l3228_322823


namespace shopkeeper_profit_percentage_l3228_322877

theorem shopkeeper_profit_percentage
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (discount : ℝ)
  (h_cp : cost_price = 180)
  (h_mp : markup_percentage = 45)
  (h_d : discount = 45) :
  let markup := cost_price * (markup_percentage / 100)
  let marked_price := cost_price + markup
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 := by
sorry

end shopkeeper_profit_percentage_l3228_322877


namespace functional_equation_solution_l3228_322880

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (f : FunctionType) : Prop :=
  ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y

-- Define the solution conditions
def IsSolution (f : FunctionType) : Prop :=
  (∀ x : ℝ, f x = 0) ∨
  ((∀ x : ℝ, x ≠ 0 → f x = 1) ∧ ∃ c : ℝ, f 0 = c)

-- Theorem statement
theorem functional_equation_solution (f : FunctionType) :
  SatisfiesEquation f → IsSolution f :=
sorry

end functional_equation_solution_l3228_322880


namespace perpendicular_line_equation_l3228_322827

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line = Line.mk 1 (-2) (-3) →
  point = Point.mk 3 1 →
  ∃ (result_line : Line),
    perpendicular given_line result_line ∧
    on_line point result_line ∧
    result_line = Line.mk 2 1 (-7) := by
  sorry

end perpendicular_line_equation_l3228_322827


namespace hyperbola_eccentricity_l3228_322866

/-- Given a parabola and a hyperbola with specific properties, prove that the eccentricity of the hyperbola is 1 + √2 -/
theorem hyperbola_eccentricity 
  (p a b : ℝ) 
  (hp : p > 0) 
  (ha : a > 0) 
  (hb : b > 0) 
  (parabola : ℝ → ℝ → Prop) 
  (hyperbola : ℝ → ℝ → Prop) 
  (parabola_eq : ∀ x y, parabola x y ↔ y^2 = 2*p*x) 
  (hyperbola_eq : ∀ x y, hyperbola x y ↔ x^2/a^2 - y^2/b^2 = 1) 
  (focus_shared : ∃ F : ℝ × ℝ, F.1 = p/2 ∧ F.2 = 0 ∧ 
    F.1^2/a^2 + F.2^2/b^2 = (a^2 + b^2)/a^2) 
  (intersection_line : ∃ I₁ I₂ : ℝ × ℝ, 
    parabola I₁.1 I₁.2 ∧ hyperbola I₁.1 I₁.2 ∧ 
    parabola I₂.1 I₂.2 ∧ hyperbola I₂.1 I₂.2 ∧ 
    (I₂.2 - I₁.2) * (p/2 - I₁.1) = (I₂.1 - I₁.1) * (0 - I₁.2)) :
  (a^2 + b^2)/a^2 = (1 + Real.sqrt 2)^2 := by
sorry

end hyperbola_eccentricity_l3228_322866


namespace brown_paint_red_pigment_weight_l3228_322814

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue : Real
  red : Real
  yellow : Real

/-- Represents the weight of each paint in the mixture -/
structure MixtureWeights where
  maroon : Real
  green : Real

theorem brown_paint_red_pigment_weight
  (maroon : PaintMixture)
  (green : PaintMixture)
  (weights : MixtureWeights)
  (h_maroon_comp : maroon.blue = 0.5 ∧ maroon.red = 0.5 ∧ maroon.yellow = 0)
  (h_green_comp : green.blue = 0.3 ∧ green.red = 0 ∧ green.yellow = 0.7)
  (h_total_weight : weights.maroon + weights.green = 10)
  (h_brown_blue : weights.maroon * maroon.blue + weights.green * green.blue = 4) :
  weights.maroon * maroon.red = 2.5 := by
  sorry

#check brown_paint_red_pigment_weight

end brown_paint_red_pigment_weight_l3228_322814


namespace student_number_problem_l3228_322817

theorem student_number_problem (x : ℝ) : 5 * x - 138 = 102 → x = 48 := by
  sorry

end student_number_problem_l3228_322817


namespace isabellas_total_items_l3228_322822

/-- Given that Alexis bought 3 times more pants and dresses than Isabella,
    prove that Isabella bought 13 items in total. -/
theorem isabellas_total_items
  (alexis_pants : ℕ)
  (alexis_dresses : ℕ)
  (h1 : alexis_pants = 21)
  (h2 : alexis_dresses = 18)
  (h3 : ∃ (k : ℕ), k > 0 ∧ alexis_pants = 3 * k ∧ alexis_dresses = 3 * (alexis_dresses / 3)) :
  alexis_pants / 3 + alexis_dresses / 3 = 13 :=
by sorry

end isabellas_total_items_l3228_322822


namespace bob_distance_when_meeting_l3228_322876

/-- Prove that Bob walked 35 miles when he met Yolanda, given the following conditions:
  - The total distance between X and Y is 65 miles
  - Yolanda starts walking from X to Y
  - Bob starts walking from Y to X one hour after Yolanda
  - Yolanda's walking rate is 5 miles per hour
  - Bob's walking rate is 7 miles per hour
-/
theorem bob_distance_when_meeting (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ)
  (h1 : total_distance = 65)
  (h2 : yolanda_rate = 5)
  (h3 : bob_rate = 7) :
  let time_to_meet := (total_distance - yolanda_rate) / (yolanda_rate + bob_rate)
  bob_rate * time_to_meet = 35 := by
  sorry

end bob_distance_when_meeting_l3228_322876


namespace august_mail_total_l3228_322860

/-- The number of pieces of mail Vivian sent in a given month -/
def mail_sent (month : String) : ℕ :=
  match month with
  | "April" => 5
  | "May" => 10
  | "June" => 20
  | "July" => 40
  | _ => 0

/-- The number of business days in August -/
def august_business_days : ℕ := 23

/-- The number of holidays in August -/
def august_holidays : ℕ := 8

/-- The amount of mail sent on a business day in August -/
def august_business_day_mail : ℕ := 2 * mail_sent "July"

/-- The amount of mail sent on a holiday in August -/
def august_holiday_mail : ℕ := mail_sent "July" / 2

theorem august_mail_total :
  august_business_days * august_business_day_mail +
  august_holidays * august_holiday_mail = 2000 := by
  sorry

end august_mail_total_l3228_322860


namespace quadratic_form_ratio_l3228_322829

theorem quadratic_form_ratio (b c : ℝ) : 
  (∀ x, x^2 + 1500*x + 2400 = (x + b)^2 + c) → 
  c / b = -746.8 := by
sorry

end quadratic_form_ratio_l3228_322829


namespace prob_rolling_doubles_l3228_322846

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice being rolled -/
def numDice : ℕ := 3

/-- The number of favorable outcomes (rolling the same number on all dice) -/
def favorableOutcomes : ℕ := numSides

/-- The total number of possible outcomes when rolling the dice -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The probability of rolling doubles with three six-sided dice -/
theorem prob_rolling_doubles : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 36 := by sorry

end prob_rolling_doubles_l3228_322846


namespace least_m_is_207_l3228_322824

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 8) / (x n + 7)

def is_least_m (m : ℕ) : Prop :=
  x m ≤ 5 + 1 / 2^15 ∧ ∀ k < m, x k > 5 + 1 / 2^15

theorem least_m_is_207 : is_least_m 207 := by
  sorry

end least_m_is_207_l3228_322824


namespace least_possible_n_l3228_322837

/-- The type of rational coefficients for the polynomial terms -/
structure Coefficient where
  a : ℚ
  b : ℚ

/-- 
Checks if a list of coefficients satisfies the equation
x^2 + x + 4 = ∑(i=1 to n) (a_i * x + b_i)^2 for all real x
-/
def satisfies_equation (coeffs : List Coefficient) : Prop :=
  ∀ (x : ℝ), x^2 + x + 4 = (coeffs.map (fun c => (c.a * x + c.b)^2)).sum

/-- The main theorem stating that 5 is the least possible value of n -/
theorem least_possible_n :
  (∃ (coeffs : List Coefficient), coeffs.length = 5 ∧ satisfies_equation coeffs) ∧
  (∀ (n : ℕ) (coeffs : List Coefficient), n < 5 → coeffs.length = n → ¬satisfies_equation coeffs) :=
sorry

end least_possible_n_l3228_322837


namespace cube_difference_problem_l3228_322857

theorem cube_difference_problem (a b c : ℝ) 
  (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c)
  (sum_of_squares : a^2 + b^2 + c^2 = 160)
  (largest_sum : a = b + c)
  (difference : b - c = 4) :
  |b^3 - c^3| = 320 := by
sorry

end cube_difference_problem_l3228_322857


namespace correct_stratified_sample_l3228_322862

/-- Represents the staff categories in the unit -/
inductive StaffCategory
  | Business
  | Management
  | Logistics

/-- Represents the staff distribution in the unit -/
structure StaffDistribution where
  total : ℕ
  business : ℕ
  management : ℕ
  logistics : ℕ
  sum_eq_total : business + management + logistics = total

/-- Represents the sample size and distribution -/
structure Sample where
  size : ℕ
  business : ℕ
  management : ℕ
  logistics : ℕ
  sum_eq_size : business + management + logistics = size

/-- Checks if a sample is proportionally correct for a given staff distribution -/
def is_proportional_sample (staff : StaffDistribution) (sample : Sample) : Prop :=
  staff.business * sample.size = sample.business * staff.total ∧
  staff.management * sample.size = sample.management * staff.total ∧
  staff.logistics * sample.size = sample.logistics * staff.total

/-- Theorem: The given sample is proportionally correct for the given staff distribution -/
theorem correct_stratified_sample :
  let staff : StaffDistribution := ⟨160, 112, 16, 32, rfl⟩
  let sample : Sample := ⟨20, 14, 2, 4, rfl⟩
  is_proportional_sample staff sample := by sorry


end correct_stratified_sample_l3228_322862


namespace negation_of_universal_proposition_l3228_322801

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x - 1| - |x + 1| ≤ 3) ↔ (∃ x₀ : ℝ, |x₀ - 1| - |x₀ + 1| > 3) := by
  sorry

end negation_of_universal_proposition_l3228_322801


namespace positive_expression_l3228_322805

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 1 < z ∧ z < 2) : 
  y + z^2 > 0 := by
  sorry

end positive_expression_l3228_322805


namespace money_division_l3228_322856

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 3200 →
  r - q = 4000 := by
sorry

end money_division_l3228_322856


namespace polynomial_factorization_l3228_322813

theorem polynomial_factorization (x : ℝ) :
  x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end polynomial_factorization_l3228_322813


namespace smallest_common_multiple_of_8_and_6_l3228_322803

theorem smallest_common_multiple_of_8_and_6 : 
  ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m > 0 → 8 ∣ m → 6 ∣ m → n ≤ m :=
by sorry

end smallest_common_multiple_of_8_and_6_l3228_322803


namespace problem_1_l3228_322842

theorem problem_1 (x : ℝ) (a : ℝ) : x - 1/x = 3 → a = x^2 + 1/x^2 → a = 11 := by
  sorry


end problem_1_l3228_322842


namespace line_equation_correct_l3228_322826

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def pointOnLine (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a vector is parallel to a line --/
def vectorParallelToLine (l : Line2D) (v : Vector2D) : Prop :=
  l.a * v.y = l.b * v.x

/-- The main theorem --/
theorem line_equation_correct (l : Line2D) (p : Point2D) (v : Vector2D) : 
  l.a = 1 ∧ l.b = 2 ∧ l.c = -1 ∧
  p.x = 1 ∧ p.y = 0 ∧
  v.x = 2 ∧ v.y = -1 →
  pointOnLine l p ∧ vectorParallelToLine l v := by
  sorry

end line_equation_correct_l3228_322826


namespace is_vertex_of_parabola_l3228_322847

/-- The parabola equation -/
def parabola_equation (x : ℝ) : ℝ := -2 * x^2 - 20 * x - 50

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-5, 0)

/-- Theorem stating that the given point is the vertex of the parabola -/
theorem is_vertex_of_parabola :
  let (m, n) := vertex
  ∀ x : ℝ, parabola_equation x ≤ parabola_equation m :=
by sorry

end is_vertex_of_parabola_l3228_322847


namespace quadratic_system_solution_l3228_322850

theorem quadratic_system_solution (a b c : ℝ) :
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧
             b * x^2 + c * x + a = 0 ∧
             c * x^2 + a * x + b = 0) ↔
  a + b + c = 0 := by
  sorry

end quadratic_system_solution_l3228_322850
