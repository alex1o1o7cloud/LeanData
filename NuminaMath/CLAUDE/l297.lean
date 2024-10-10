import Mathlib

namespace f_monotone_increasing_l297_29756

/-- The function f(x) = x^3 - 1/x is monotonically increasing for x > 0 -/
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → x₁^3 - 1/x₁ < x₂^3 - 1/x₂ := by
  sorry

end f_monotone_increasing_l297_29756


namespace cube_root_unity_polynomial_identity_l297_29790

theorem cube_root_unity_polynomial_identity
  (a b c : ℂ) (n m : ℕ) :
  (∃ x : ℂ, x^3 = 1 ∧ a * x^(3*n + 2) + b * x^(3*m + 1) + c = 0) →
  a^3 + b^3 + c^3 - 3*a*b*c = 0 :=
by
  sorry

end cube_root_unity_polynomial_identity_l297_29790


namespace x_varies_with_z_l297_29701

theorem x_varies_with_z (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
sorry

end x_varies_with_z_l297_29701


namespace total_players_on_ground_l297_29780

def cricket_players : ℕ := 35
def hockey_players : ℕ := 28
def football_players : ℕ := 42
def softball_players : ℕ := 25
def basketball_players : ℕ := 18
def volleyball_players : ℕ := 30

theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + 
  softball_players + basketball_players + volleyball_players = 178 := by
  sorry

end total_players_on_ground_l297_29780


namespace sum_of_fractions_equals_target_l297_29757

theorem sum_of_fractions_equals_target : 
  (1/3 : ℚ) + (1/2 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (2/15 : ℚ) = 
  (13333333333333333 : ℚ) / (100000000000000000 : ℚ) := by sorry

end sum_of_fractions_equals_target_l297_29757


namespace min_value_of_f_l297_29704

def f (x : ℝ) := x^2 - 4*x + 7

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 3 :=
sorry

end min_value_of_f_l297_29704


namespace average_speed_calculation_l297_29761

/-- Calculate the average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_calculation (speed1 speed2 : ℝ) (h1 : speed1 = 90) (h2 : speed2 = 55) :
  (speed1 + speed2) / 2 = 72.5 := by
  sorry

end average_speed_calculation_l297_29761


namespace problem_solution_l297_29709

theorem problem_solution (a b : ℝ) (h : |a - 1| + (2 + b)^2 = 0) : 
  (a + b)^2009 = -1 := by
  sorry

end problem_solution_l297_29709


namespace opposite_teal_is_violet_l297_29716

-- Define the colors
inductive Color
  | Blue
  | Orange
  | Yellow
  | Violet
  | Teal
  | Pink

-- Define a cube as a function from face positions to colors
def Cube := Fin 6 → Color

-- Define face positions
def top : Fin 6 := 0
def bottom : Fin 6 := 1
def left : Fin 6 := 2
def right : Fin 6 := 3
def front : Fin 6 := 4
def back : Fin 6 := 5

-- Define the theorem
theorem opposite_teal_is_violet (cube : Cube) :
  (∀ (view : Fin 3), cube top = Color.Violet) →
  (∀ (view : Fin 3), cube left = Color.Orange) →
  (cube front = Color.Blue ∨ cube front = Color.Yellow ∨ cube front = Color.Pink) →
  (∃ (face : Fin 6), cube face = Color.Teal) →
  (∀ (face1 face2 : Fin 6), face1 ≠ face2 → cube face1 ≠ cube face2) →
  (cube bottom = Color.Teal → cube top = Color.Violet) :=
by sorry

end opposite_teal_is_violet_l297_29716


namespace city_population_theorem_l297_29742

/-- Given three cities with populations H, L, and C, where H > L and C = H - 5000,
    prove that if L + C = H + C - 5000, then L = H - 5000. -/
theorem city_population_theorem (H L C : ℕ) 
    (h1 : H > L) 
    (h2 : C = H - 5000) 
    (h3 : L + C = H + C - 5000) : 
  L = H - 5000 := by
  sorry

end city_population_theorem_l297_29742


namespace pie_division_l297_29755

theorem pie_division (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 5/6 ∧ num_people = 4 → (total_pie / num_people : ℚ) = 5/24 := by
  sorry

end pie_division_l297_29755


namespace probability_not_purple_l297_29795

/-- Given a bag of marbles where the odds of pulling a purple marble are 5:6,
    prove that the probability of not pulling a purple marble is 6/11. -/
theorem probability_not_purple (total : ℕ) (purple : ℕ) (not_purple : ℕ) :
  total = purple + not_purple →
  purple = 5 →
  not_purple = 6 →
  (not_purple : ℚ) / total = 6 / 11 :=
by sorry

end probability_not_purple_l297_29795


namespace unique_prime_solution_l297_29772

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_prime_solution :
  ∀ p q r : ℕ,
    is_prime p ∧ is_prime q ∧ is_prime r ∧
    p * (q - r) = q + r →
    p = 5 ∧ q = 3 ∧ r = 2 :=
by sorry

end unique_prime_solution_l297_29772


namespace negative_two_equals_negative_abs_two_l297_29789

theorem negative_two_equals_negative_abs_two : -2 = -|-2| := by
  sorry

end negative_two_equals_negative_abs_two_l297_29789


namespace range_of_a_l297_29707

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- Define the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingFunction f) 
  (h_inequality : f (1 - a) < f (2 * a - 1)) :
  0 < a ∧ a < 2/3 := by
  sorry

end range_of_a_l297_29707


namespace two_numbers_sum_and_difference_l297_29737

theorem two_numbers_sum_and_difference (x y : ℝ) : 
  x + y = 18 ∧ x - y = 24 → x = 21 ∧ y = -3 := by
  sorry

end two_numbers_sum_and_difference_l297_29737


namespace custom_mul_result_l297_29776

/-- Custom multiplication operation for rational numbers -/
noncomputable def custom_mul (a b : ℚ) (x y : ℚ) : ℚ := a * x + b * y

theorem custom_mul_result 
  (a b : ℚ) 
  (h1 : custom_mul a b 1 2 = 1) 
  (h2 : custom_mul a b (-3) 3 = 6) :
  custom_mul a b 2 (-5) = -7 := by
  sorry

end custom_mul_result_l297_29776


namespace sum_equals_rounded_sum_jo_equals_alex_sum_l297_29759

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def sum_to_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_rounded_to_five (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_five |>.sum

theorem sum_equals_rounded_sum (n : ℕ) : sum_to_n n = sum_rounded_to_five n := by
  sorry

-- The main theorem
theorem jo_equals_alex_sum : sum_to_n 200 = sum_rounded_to_five 200 := by
  sorry

end sum_equals_rounded_sum_jo_equals_alex_sum_l297_29759


namespace inequality_solution_l297_29736

theorem inequality_solution (x : ℝ) :
  x ≠ 5 →
  (x * (x + 2) / (x - 5)^2 ≥ 15 ↔ 
    x ≤ 3.71 ∨ (x ≥ 7.14 ∧ x < 5) ∨ x > 5) :=
by sorry

end inequality_solution_l297_29736


namespace bombardment_percentage_l297_29781

/-- Proves that the percentage of people who died by bombardment is 10% --/
theorem bombardment_percentage (initial_population : ℕ) (final_population : ℕ) 
  (h1 : initial_population = 4500)
  (h2 : final_population = 3240)
  (h3 : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ 
    final_population = initial_population - (x / 100 * initial_population) - 
    (1/5 * (initial_population - (x / 100 * initial_population)))) :
  ∃ x : ℝ, x = 10 ∧ 
    final_population = initial_population - (x / 100 * initial_population) - 
    (1/5 * (initial_population - (x / 100 * initial_population))) :=
by sorry

#check bombardment_percentage

end bombardment_percentage_l297_29781


namespace parallelogram_area_26_14_l297_29706

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 26 cm and height 14 cm is 364 square centimeters -/
theorem parallelogram_area_26_14 : parallelogram_area 26 14 = 364 := by
  sorry

end parallelogram_area_26_14_l297_29706


namespace parallel_segment_length_l297_29760

/-- Given a triangle ABC with side AC = 8 cm, if two segments parallel to AC divide the triangle
    into three equal areas, then the length of the parallel segment closest to AC is 8√3/3. -/
theorem parallel_segment_length (A B C : ℝ × ℝ) (a b : ℝ) :
  let triangle_area := (4 : ℝ) * b
  let segment_de_length := (8 : ℝ) * Real.sqrt 6 / 3
  let segment_fg_length := (8 : ℝ) * Real.sqrt 3 / 3
  A = (0, 0) →
  B = (a, b) →
  C = (8, 0) →
  triangle_area / 3 = b * (a * Real.sqrt (8 / 3))^2 / (2 * a) →
  segment_de_length > segment_fg_length →
  segment_fg_length = 8 * Real.sqrt 3 / 3 :=
by sorry

end parallel_segment_length_l297_29760


namespace pet_shop_theorem_l297_29732

def pet_shop_problem (parakeet_cost : ℕ) : Prop :=
  let puppy_cost := 3 * parakeet_cost
  let kitten_cost := 2 * parakeet_cost
  let total_cost := 2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost
  parakeet_cost = 10 → total_cost = 130

theorem pet_shop_theorem : pet_shop_problem 10 := by
  sorry

end pet_shop_theorem_l297_29732


namespace complex_multiplication_l297_29703

theorem complex_multiplication (i : ℂ) : i * i = -1 → (1 - i) * (1 + 2*i) = 3 + i := by
  sorry

end complex_multiplication_l297_29703


namespace arithmetic_sequence_property_l297_29714

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_condition : a 2^2 + 2*a 2*a 8 + a 6*a 10 = 16) : 
  a 4 * a 6 = 4 := by
sorry

end arithmetic_sequence_property_l297_29714


namespace correct_sample_size_l297_29722

/-- Given a population with total students and girls, and a sample size,
    calculate the number of girls in the sample using stratified sampling. -/
def girlsInSample (totalStudents girls sampleSize : ℕ) : ℕ :=
  (girls * sampleSize) / totalStudents

/-- Theorem stating that for the given population and sample size,
    the number of girls in the sample should be 20. -/
theorem correct_sample_size :
  girlsInSample 30000 4000 150 = 20 := by
  sorry

end correct_sample_size_l297_29722


namespace correct_stratified_sample_l297_29715

structure University :=
  (total_students : ℕ)
  (freshmen : ℕ)
  (sophomores : ℕ)
  (juniors : ℕ)
  (seniors : ℕ)
  (sample_size : ℕ)

def stratified_sample (u : University) : Vector ℕ 4 :=
  let sampling_ratio := u.sample_size / u.total_students
  ⟨[u.freshmen * sampling_ratio,
    u.sophomores * sampling_ratio,
    u.juniors * sampling_ratio,
    u.seniors * sampling_ratio],
   by simp⟩

theorem correct_stratified_sample (u : University) 
  (h1 : u.total_students = 8000)
  (h2 : u.freshmen = 1600)
  (h3 : u.sophomores = 3200)
  (h4 : u.juniors = 2000)
  (h5 : u.seniors = 1200)
  (h6 : u.sample_size = 400)
  (h7 : u.total_students = u.freshmen + u.sophomores + u.juniors + u.seniors) :
  stratified_sample u = ⟨[80, 160, 100, 60], by simp⟩ := by
  sorry

#check correct_stratified_sample

end correct_stratified_sample_l297_29715


namespace fraction_sum_equals_decimal_l297_29785

theorem fraction_sum_equals_decimal : (3 / 50) + (5 / 500) + (7 / 5000) = 0.0714 := by
  sorry

end fraction_sum_equals_decimal_l297_29785


namespace tangent_line_slope_l297_29791

/-- Given a differentiable function f, prove that its derivative at x = 1 is 2,
    given that the tangent line equation at (1, f(1)) is 2x - y + 2 = 0. -/
theorem tangent_line_slope (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x y, x = 1 ∧ y = f 1 → 2 * x - y + 2 = 0) →
  deriv f 1 = 2 := by
sorry

end tangent_line_slope_l297_29791


namespace vector_simplification_l297_29771

variable {V : Type*} [AddCommGroup V]
variable (A B C D F : V)

theorem vector_simplification :
  (C - D) + (B - C) + (A - B) = A - D ∧
  (A - B) + (D - F) + (C - D) + (B - C) + (F - A) = 0 := by
  sorry

end vector_simplification_l297_29771


namespace min_reciprocal_sum_l297_29754

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 4 * Real.sqrt 3 :=
sorry

end min_reciprocal_sum_l297_29754


namespace parabola_y_axis_intersection_l297_29792

/-- The equation of the parabola -/
def parabola (x y : ℝ) : Prop := y = x^2 + 2*x - 3

/-- Definition of a point on the y-axis -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the parabola with the y-axis -/
def intersection_point : ℝ × ℝ := (0, -3)

/-- Theorem stating that the intersection_point is on the parabola and the y-axis -/
theorem parabola_y_axis_intersection :
  let (x, y) := intersection_point
  parabola x y ∧ on_y_axis x y :=
by sorry

end parabola_y_axis_intersection_l297_29792


namespace cans_collection_proof_l297_29740

/-- The number of cans collected on a given day -/
def cans_on_day (a b : ℚ) (d : ℕ) : ℚ := a * d^2 + b

theorem cans_collection_proof (a b : ℚ) :
  cans_on_day a b 1 = 4 ∧
  cans_on_day a b 2 = 9 ∧
  cans_on_day a b 3 = 14 →
  a = 5/3 ∧ b = 7/3 ∧ cans_on_day a b 7 = 84 := by
  sorry

end cans_collection_proof_l297_29740


namespace cat_whiskers_ratio_l297_29782

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  juniper : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- The theorem stating the relationship between the cats' whiskers and their ratio -/
theorem cat_whiskers_ratio (c : CatWhiskers) : 
  c.juniper = 12 →
  c.buffy = 40 →
  c.puffy = 3 * c.juniper →
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3 →
  (Ratio.mk c.puffy c.scruffy) = (Ratio.mk 1 2) := by
  sorry


end cat_whiskers_ratio_l297_29782


namespace jack_additional_sweets_l297_29712

theorem jack_additional_sweets (initial_sweets : ℕ) (remaining_sweets : ℕ) : 
  initial_sweets = 22 →
  remaining_sweets = 7 →
  (initial_sweets / 2 + (initial_sweets - remaining_sweets - initial_sweets / 2) = initial_sweets - remaining_sweets) →
  initial_sweets - remaining_sweets - initial_sweets / 2 = 4 :=
by sorry

end jack_additional_sweets_l297_29712


namespace quadratic_roots_l297_29799

theorem quadratic_roots (a b c : ℝ) (h : (b^3)^2 - 4*(a^3)*(c^3) > 0) :
  (b^5)^2 - 4*(a^5)*(c^5) > 0 := by
  sorry

end quadratic_roots_l297_29799


namespace unique_solution_absolute_value_equation_l297_29705

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 3| = |x + 2| ∧ |x + 2| = |x - 5| ∧ x = 2 := by
  sorry

end unique_solution_absolute_value_equation_l297_29705


namespace vector_sign_sum_l297_29713

/-- Given a 3-dimensional vector with nonzero components, the sum of the signs of its components
    plus the sign of their product can only be 4, 0, or -2. -/
theorem vector_sign_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x / |x| + y / |y| + z / |z| + (x * y * z) / |x * y * z|) ∈ ({4, 0, -2} : Set ℝ) := by
  sorry

end vector_sign_sum_l297_29713


namespace intersection_complement_theorem_l297_29773

def U : Set Nat := {0, 1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {3, 4, 5, 6}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 2} := by sorry

end intersection_complement_theorem_l297_29773


namespace even_perfect_square_factors_count_l297_29702

/-- The number of even perfect square factors of 2^6 * 5^4 * 7^3 -/
def num_even_perfect_square_factors : ℕ :=
  sorry

/-- The given number -/
def given_number : ℕ :=
  2^6 * 5^4 * 7^3

theorem even_perfect_square_factors_count :
  num_even_perfect_square_factors = 18 :=
by sorry

end even_perfect_square_factors_count_l297_29702


namespace solve_equation_l297_29721

theorem solve_equation (x : ℝ) : 
  (x^4)^(1/3) = 32 * 32^(1/12) → x = 16 * 2^(1/4) := by
  sorry

end solve_equation_l297_29721


namespace patio_surrounded_by_bushes_l297_29783

/-- The side length of the square patio in feet -/
def patio_side_length : ℝ := 20

/-- The spacing between rose bushes in feet -/
def bush_spacing : ℝ := 2

/-- The number of rose bushes needed to surround the patio -/
def num_bushes : ℕ := 40

/-- Theorem stating that the number of rose bushes needed to surround the square patio is 40 -/
theorem patio_surrounded_by_bushes :
  (4 * patio_side_length) / bush_spacing = num_bushes := by sorry

end patio_surrounded_by_bushes_l297_29783


namespace unique_minimum_condition_l297_29779

/-- The objective function z(x,y) = ax + 2y has its unique minimum at (1,0) for all real x and y
    if and only if a is in the open interval (-4, -2) -/
theorem unique_minimum_condition (a : ℝ) :
  (∀ x y : ℝ, (a * x + 2 * y) ≥ (a * 1 + 2 * 0) ∧ 
   (∀ x' y' : ℝ, (x', y') ≠ (1, 0) → (a * x' + 2 * y') > (a * 1 + 2 * 0)))
  ↔ 
  (-4 < a ∧ a < -2) :=
sorry

end unique_minimum_condition_l297_29779


namespace f_four_equals_thirtysix_l297_29735

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_four_equals_thirtysix
  (f : ℝ → ℝ)
  (h_not_zero : ∃ x, f x ≠ 0)
  (h_eq : FunctionalEquation f)
  (h_f_two : f 2 = 9) :
  f 4 = 36 := by
  sorry

end f_four_equals_thirtysix_l297_29735


namespace simplify_and_evaluate_l297_29739

theorem simplify_and_evaluate (x : ℝ) (h : x ≠ 3) :
  (x^2 - x - 6) / (x - 3) = x + 2 ∧ 
  (4^2 - 4 - 6) / (4 - 3) = 6 :=
sorry

end simplify_and_evaluate_l297_29739


namespace prime_divisibility_l297_29727

theorem prime_divisibility (p q : Nat) : 
  Nat.Prime p → Nat.Prime q → q ∣ (3^p - 2^p) → p ∣ (q - 1) := by
  sorry

end prime_divisibility_l297_29727


namespace accident_insurance_probability_l297_29786

theorem accident_insurance_probability (p1 p2 : ℝ) 
  (h1 : p1 = 1 / 20)
  (h2 : p2 = 1 / 21)
  (h3 : 0 ≤ p1 ∧ p1 ≤ 1)
  (h4 : 0 ≤ p2 ∧ p2 ≤ 1) :
  1 - (1 - p1) * (1 - p2) = 2 / 21 := by
sorry


end accident_insurance_probability_l297_29786


namespace exists_same_color_distance_one_l297_29768

/-- A coloring of the plane using three colors -/
def Coloring := ℝ × ℝ → Fin 3

/-- Two points in the plane -/
def TwoPoints := (ℝ × ℝ) × (ℝ × ℝ)

/-- The distance between two points is 1 -/
def DistanceOne (p : TwoPoints) : Prop :=
  let (p1, p2) := p
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = 1

/-- Two points have the same color -/
def SameColor (c : Coloring) (p : TwoPoints) : Prop :=
  let (p1, p2) := p
  c p1 = c p2

/-- Main theorem: In any three-coloring of the plane, there exist two points
    of the same color with distance 1 between them -/
theorem exists_same_color_distance_one :
  ∀ c : Coloring, ∃ p : TwoPoints, DistanceOne p ∧ SameColor c p := by
  sorry

end exists_same_color_distance_one_l297_29768


namespace restoration_time_is_minimum_l297_29708

/-- Represents the time required for a process on a handicraft -/
structure ProcessTime :=
  (shaping : ℕ)
  (painting : ℕ)

/-- The set of handicrafts -/
inductive Handicraft
  | A
  | B
  | C

/-- The time required for each handicraft -/
def handicraftTime : Handicraft → ProcessTime
  | Handicraft.A => ⟨9, 15⟩
  | Handicraft.B => ⟨16, 8⟩
  | Handicraft.C => ⟨10, 14⟩

/-- The minimum time required to complete the restoration work -/
def minRestorationTime : ℕ := 46

theorem restoration_time_is_minimum :
  minRestorationTime = 46 ∧
  ∀ (order : List Handicraft), order.length = 3 →
    (order.foldl (λ acc h => acc + (handicraftTime h).shaping) 0) +
    (List.maximum (order.map (λ h => (handicraftTime h).painting)) ).getD 0 ≥ minRestorationTime :=
  sorry

#check restoration_time_is_minimum

end restoration_time_is_minimum_l297_29708


namespace sum_of_three_equal_numbers_l297_29730

theorem sum_of_three_equal_numbers (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 → 
  a = 12 → 
  b = 24 → 
  c = d → 
  d = e → 
  c + d + e = 64 := by
sorry

end sum_of_three_equal_numbers_l297_29730


namespace fourth_term_equals_eleven_l297_29741

/-- Given a sequence {aₙ} where Sₙ = 2n² - 3n, prove that a₄ = 11 -/
theorem fourth_term_equals_eleven (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = 2 * n^2 - 3 * n) →
  (∀ n, a n = S n - S (n-1)) →
  a 4 = 11 := by
sorry

end fourth_term_equals_eleven_l297_29741


namespace adi_change_l297_29700

/-- The change Adi receives when buying a pencil -/
theorem adi_change (pencil_cost : ℕ) (payment : ℕ) (change : ℕ) : 
  pencil_cost = 35 →
  payment = 100 →
  change = payment - pencil_cost →
  change = 65 :=
by
  sorry

end adi_change_l297_29700


namespace plane_contains_points_and_satisfies_constraints_l297_29767

def point1 : ℝ × ℝ × ℝ := (2, -1, 3)
def point2 : ℝ × ℝ × ℝ := (0, -1, 5)
def point3 : ℝ × ℝ × ℝ := (-2, -3, 4)

def plane_equation (x y z : ℝ) : Prop := 2*x + 5*y - 2*z + 7 = 0

theorem plane_contains_points_and_satisfies_constraints :
  (plane_equation point1.1 point1.2.1 point1.2.2) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2) ∧
  (2 > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd 2 5) 2) 7 = 1) :=
sorry

end plane_contains_points_and_satisfies_constraints_l297_29767


namespace smallest_sum_of_digits_of_sum_l297_29748

/-- A function that returns true if a number has unique digits -/
def hasUniqueDigits (n : Nat) : Bool :=
  sorry

/-- A function that returns the sum of digits of a number -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

/-- A function that checks if two numbers use all digits from 1 to 9 exactly once between them -/
def useAllDigitsOnce (x y : Nat) : Bool :=
  sorry

theorem smallest_sum_of_digits_of_sum (x y : Nat) : 
  x ≥ 100 ∧ x < 1000 ∧ 
  y ≥ 100 ∧ y < 1000 ∧ 
  hasUniqueDigits x ∧ 
  hasUniqueDigits y ∧ 
  useAllDigitsOnce x y ∧
  x + y < 1000 →
  ∃ (T : Nat), T = x + y ∧ sumOfDigits T ≥ 21 :=
by sorry

end smallest_sum_of_digits_of_sum_l297_29748


namespace M_intersect_N_eq_M_l297_29764

open Set Real

def M : Set ℝ := {x | ∃ y, y = log (x - 1)}
def N : Set ℝ := {y | ∃ x, y = x^2 + 1}

theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end M_intersect_N_eq_M_l297_29764


namespace neither_a_nor_b_probability_l297_29745

def prob_a : ℝ := 0.20
def prob_b : ℝ := 0.40
def prob_a_and_b : ℝ := 0.15

theorem neither_a_nor_b_probability :
  1 - (prob_a + prob_b - prob_a_and_b) = 0.55 := by
  sorry

end neither_a_nor_b_probability_l297_29745


namespace concert_ticket_price_l297_29744

theorem concert_ticket_price (num_tickets : ℕ) (total_spent : ℚ) (h1 : num_tickets = 8) (h2 : total_spent = 32) : 
  total_spent / num_tickets = 4 := by
  sorry

end concert_ticket_price_l297_29744


namespace max_soap_boxes_in_carton_l297_29719

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Carton dimensions -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- Soap box dimensions -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 6 }

/-- Theorem: The maximum number of soap boxes that can be placed in the carton is 250 -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 250 := by
  sorry

end max_soap_boxes_in_carton_l297_29719


namespace kendra_hat_purchase_l297_29797

theorem kendra_hat_purchase (toy_price hat_price initial_money change toys_bought : ℕ) 
  (h1 : toy_price = 20)
  (h2 : hat_price = 10)
  (h3 : initial_money = 100)
  (h4 : toys_bought = 2)
  (h5 : change = 30) :
  (initial_money - change - toy_price * toys_bought) / hat_price = 3 := by
  sorry

end kendra_hat_purchase_l297_29797


namespace wood_burning_problem_l297_29711

/-- Wood burning problem -/
theorem wood_burning_problem (initial_bundles morning_burned end_bundles : ℕ) 
  (h1 : initial_bundles = 10)
  (h2 : morning_burned = 4)
  (h3 : end_bundles = 3) :
  initial_bundles - morning_burned - end_bundles = 3 :=
by sorry

end wood_burning_problem_l297_29711


namespace xy_value_l297_29796

theorem xy_value (x y : ℝ) (h : |x - 1| + (y + 2)^2 = 0) : x * y = -2 := by
  sorry

end xy_value_l297_29796


namespace x_varies_as_square_of_sin_z_l297_29724

/-- Given that x is directly proportional to the square of y, and y is directly proportional to sin(z),
    prove that x varies as the 2nd power of sin(z). -/
theorem x_varies_as_square_of_sin_z
  (x y z : ℝ)
  (hxy : ∃ k : ℝ, x = k * y^2)
  (hyz : ∃ j : ℝ, y = j * Real.sin z) :
  ∃ m : ℝ, x = m * (Real.sin z)^2 :=
sorry

end x_varies_as_square_of_sin_z_l297_29724


namespace contrapositive_equivalence_l297_29793

theorem contrapositive_equivalence :
  (∀ x : ℝ, x > 10 → x > 1) ↔ (∀ x : ℝ, x ≤ 1 → x ≤ 10) :=
by sorry

end contrapositive_equivalence_l297_29793


namespace initial_distance_problem_l297_29763

theorem initial_distance_problem (enrique_speed jamal_speed meeting_time : ℝ) 
  (h1 : enrique_speed = 16)
  (h2 : jamal_speed = 23)
  (h3 : meeting_time = 8) :
  enrique_speed * meeting_time + jamal_speed * meeting_time = 312 := by
  sorry

end initial_distance_problem_l297_29763


namespace five_digit_sum_l297_29729

theorem five_digit_sum (x : ℕ) : 
  (1 + 3 + 4 + 6 + x) * (5 * 4 * 3 * 2 * 1) = 2640 → x = 8 := by
  sorry

end five_digit_sum_l297_29729


namespace pyramid_volume_scaling_l297_29733

/-- Given a pyramid with a rectangular base and initial volume,
    calculate the new volume after scaling its dimensions. -/
theorem pyramid_volume_scaling (l w h : ℝ) (V : ℝ) :
  V = (1 / 3) * l * w * h →
  V = 60 →
  (1 / 3) * (3 * l) * (2 * w) * (2 * h) = 720 :=
by sorry

end pyramid_volume_scaling_l297_29733


namespace vector_decomposition_l297_29766

/-- Given vectors in R^3 -/
def x : Fin 3 → ℝ := ![5, 15, 0]
def p : Fin 3 → ℝ := ![1, 0, 5]
def q : Fin 3 → ℝ := ![-1, 3, 2]
def r : Fin 3 → ℝ := ![0, -1, 1]

/-- The decomposition of x in terms of p, q, and r -/
theorem vector_decomposition :
  x = (4 : ℝ) • p - (1 : ℝ) • q - (18 : ℝ) • r := by sorry

end vector_decomposition_l297_29766


namespace terrell_weight_lifting_l297_29775

/-- The number of times Terrell lifts the weights -/
def usual_lifts : ℕ := 10

/-- The weight of each dumbbell Terrell usually uses (in pounds) -/
def usual_weight : ℕ := 25

/-- The weight of each new dumbbell Terrell wants to use (in pounds) -/
def new_weight : ℕ := 20

/-- The number of dumbbells Terrell lifts each time -/
def num_dumbbells : ℕ := 2

/-- Calculates the total weight lifted -/
def total_weight (weight : ℕ) (lifts : ℕ) : ℕ :=
  num_dumbbells * weight * lifts

/-- The number of times Terrell needs to lift the new weights to achieve the same total weight -/
def required_lifts : ℚ :=
  (total_weight usual_weight usual_lifts : ℚ) / (num_dumbbells * new_weight)

theorem terrell_weight_lifting :
  required_lifts = 12.5 := by sorry

end terrell_weight_lifting_l297_29775


namespace total_cash_reward_l297_29758

/-- Represents a subject with its grade, credit hours, and cash reward per grade point -/
structure Subject where
  name : String
  grade : Nat
  creditHours : Nat
  cashRewardPerPoint : Nat

/-- Calculates the total cash reward for a given subject -/
def subjectReward (s : Subject) : Nat :=
  s.grade * s.cashRewardPerPoint

/-- Represents the artwork reward -/
def artworkReward : Nat := 20

/-- List of subjects with their respective information -/
def subjects : List Subject := [
  ⟨"Mathematics", 2, 5, 5⟩,
  ⟨"English", 3, 4, 4⟩,
  ⟨"Spanish", 3, 4, 4⟩,
  ⟨"Physics", 3, 4, 3⟩,
  ⟨"Chemistry", 3, 3, 3⟩,
  ⟨"History", 4, 3, 5⟩
]

/-- Calculates the total cash reward for all subjects -/
def totalSubjectsReward : Nat :=
  (subjects.map subjectReward).sum

/-- Theorem: The total cash reward Milo gets is $92 -/
theorem total_cash_reward : totalSubjectsReward + artworkReward = 92 := by
  sorry

end total_cash_reward_l297_29758


namespace centerville_snail_count_l297_29731

/-- The number of snails removed from Centerville -/
def snails_removed : ℕ := 3482

/-- The number of snails remaining in Centerville -/
def snails_remaining : ℕ := 8278

/-- The original number of snails in Centerville -/
def original_snails : ℕ := snails_removed + snails_remaining

theorem centerville_snail_count : original_snails = 11760 := by
  sorry

end centerville_snail_count_l297_29731


namespace power_multiplication_l297_29784

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l297_29784


namespace intersection_implies_m_value_l297_29747

def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

theorem intersection_implies_m_value (m : ℝ) :
  A m ∩ B = {3} → m = 3 := by
  sorry

end intersection_implies_m_value_l297_29747


namespace polynomial_simplification_l297_29717

theorem polynomial_simplification (x : ℝ) :
  3 + 5*x - 7*x^2 - 9 + 11*x - 13*x^2 + 15 - 17*x + 19*x^2 = 9 - x - x^2 :=
by sorry

end polynomial_simplification_l297_29717


namespace exactly_five_numbers_l297_29750

/-- A function that returns the number of ways a positive integer can be written as the sum of consecutive positive odd integers -/
def numConsecutiveOddSums (n : ℕ) : ℕ := sorry

/-- A function that checks if a positive integer is less than 100 and can be written as the sum of consecutive positive odd integers in exactly 3 different ways -/
def isValidNumber (n : ℕ) : Prop :=
  n < 100 ∧ numConsecutiveOddSums n = 3

/-- The main theorem stating that there are exactly 5 numbers satisfying the conditions -/
theorem exactly_five_numbers :
  ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n, n ∈ S ↔ isValidNumber n :=
sorry

end exactly_five_numbers_l297_29750


namespace even_function_k_value_l297_29718

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The function f(x) = kx^2 + (k-1)x + 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem even_function_k_value :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by sorry

end even_function_k_value_l297_29718


namespace trishas_walking_distance_l297_29728

/-- Trisha's walking distances in New York City -/
theorem trishas_walking_distance
  (total_distance : ℝ)
  (hotel_to_tshirt : ℝ)
  (h1 : total_distance = 0.8888888888888888)
  (h2 : hotel_to_tshirt = 0.6666666666666666)
  (h3 : ∃ x : ℝ, total_distance = x + x + hotel_to_tshirt) :
  ∃ x : ℝ, x = 0.1111111111111111 ∧ total_distance = x + x + hotel_to_tshirt :=
by sorry

end trishas_walking_distance_l297_29728


namespace even_function_implies_a_eq_neg_one_l297_29746

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+1)(x+a) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (x + 1) * (x + a)

/-- If f(x) = (x+1)(x+a) is an even function, then a = -1 -/
theorem even_function_implies_a_eq_neg_one :
  ∃ a : ℝ, IsEven (f a) → a = -1 := by
  sorry

end even_function_implies_a_eq_neg_one_l297_29746


namespace min_natural_numbers_for_prime_products_l297_29734

theorem min_natural_numbers_for_prime_products (p : Fin 100 → ℕ) (a : ℕ → ℕ) :
  (∀ i j, i ≠ j → p i ≠ p j) →  -- p₁, ..., p₁₀₀ are distinct
  (∀ i, Prime (p i)) →  -- p₁, ..., p₁₀₀ are prime
  (∀ i, a i > 1) →  -- Each a_i is greater than 1
  (∀ i : Fin 100, ∃ j k, a j * a k = p i * p ((i + 1) % 100)^3) →  -- Each p_i * p_{i+1}³ is a product of two a_i's
  (∃ k, ∀ i, a i ≠ 0 → i < k) →  -- There are finitely many non-zero a_i's
  (∃ k, k ≥ 150 ∧ ∀ i, a i ≠ 0 → i < k) :=  -- There are at least 150 non-zero a_i's
by sorry

end min_natural_numbers_for_prime_products_l297_29734


namespace inequality_proof_l297_29723

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  ((a^2 / b) + (b^2 / c) + (c^2 / a) ≥ 1) ∧ (a * b + b * c + a * c ≤ 1/3) := by
  sorry

end inequality_proof_l297_29723


namespace f_min_at_one_l297_29778

/-- The quadratic function that we're analyzing -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 3

/-- Theorem stating that f reaches its minimum value when x = 1 -/
theorem f_min_at_one : ∀ x : ℝ, f x ≥ f 1 := by
  sorry

end f_min_at_one_l297_29778


namespace three_large_five_small_capacity_l297_29794

/-- Represents the capacity of a large truck in tons -/
def large_truck_capacity : ℝ := sorry

/-- Represents the capacity of a small truck in tons -/
def small_truck_capacity : ℝ := sorry

/-- The total capacity of 2 large trucks and 3 small trucks is 15.5 tons -/
axiom condition1 : 2 * large_truck_capacity + 3 * small_truck_capacity = 15.5

/-- The total capacity of 5 large trucks and 6 small trucks is 35 tons -/
axiom condition2 : 5 * large_truck_capacity + 6 * small_truck_capacity = 35

/-- Theorem: 3 large trucks and 5 small trucks can transport 24.5 tons -/
theorem three_large_five_small_capacity : 
  3 * large_truck_capacity + 5 * small_truck_capacity = 24.5 := by sorry

end three_large_five_small_capacity_l297_29794


namespace translation_company_min_employees_l297_29777

/-- The number of languages offered by the company -/
def num_languages : ℕ := 4

/-- The number of languages each employee must learn -/
def languages_per_employee : ℕ := 2

/-- The minimum number of employees with identical training -/
def min_identical_training : ℕ := 5

/-- The number of possible language combinations -/
def num_combinations : ℕ := Nat.choose num_languages languages_per_employee

/-- The minimum number of employees in the company -/
def min_employees : ℕ := 25

theorem translation_company_min_employees :
  ∀ n : ℕ, n ≥ min_employees →
    ∃ (group : Finset (Finset (Fin num_languages))),
      (∀ e ∈ group, Finset.card e = languages_per_employee) ∧
      (Finset.card group ≥ min_identical_training) :=
by sorry

end translation_company_min_employees_l297_29777


namespace min_rental_cost_l297_29788

/-- Represents the rental arrangement for buses --/
structure RentalArrangement where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental arrangement is valid according to the given constraints --/
def is_valid_arrangement (arr : RentalArrangement) : Prop :=
  36 * arr.typeA + 60 * arr.typeB ≥ 900 ∧
  arr.typeA + arr.typeB ≤ 21 ∧
  arr.typeB - arr.typeA ≤ 7

/-- Calculates the total cost for a given rental arrangement --/
def total_cost (arr : RentalArrangement) : ℕ :=
  1600 * arr.typeA + 2400 * arr.typeB

/-- Theorem stating that the minimum rental cost is 36800 yuan --/
theorem min_rental_cost :
  ∃ (arr : RentalArrangement),
    is_valid_arrangement arr ∧
    total_cost arr = 36800 ∧
    ∀ (other : RentalArrangement),
      is_valid_arrangement other →
      total_cost other ≥ 36800 :=
sorry

end min_rental_cost_l297_29788


namespace functional_equation_solution_l297_29749

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x * f y) : 
  (∀ x : ℚ, f x = 0) ∨ (∀ x : ℚ, f x = 1) := by
  sorry

end functional_equation_solution_l297_29749


namespace real_part_of_2_minus_i_l297_29753

theorem real_part_of_2_minus_i : Complex.re (2 - Complex.I) = 2 := by sorry

end real_part_of_2_minus_i_l297_29753


namespace parallel_vectors_x_value_l297_29752

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The theorem statement -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (-1, x)
  let b : ℝ × ℝ := (x, -4)
  parallel a b → x = 2 ∨ x = -2 := by
sorry

end parallel_vectors_x_value_l297_29752


namespace vertical_asymptote_at_four_sevenths_l297_29725

/-- The function f(x) = (2x+3)/(7x-4) has a vertical asymptote at x = 4/7 -/
theorem vertical_asymptote_at_four_sevenths :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f x = (2*x + 3) / (7*x - 4)) →
  ∃! a : ℝ, a = 4/7 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε := by
  sorry

end vertical_asymptote_at_four_sevenths_l297_29725


namespace min_value_theorem_min_value_achievable_l297_29751

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) + (x/y * y/z * z/x) ≥ 44 := by
  sorry

-- Optionally, we can add a statement to show that the lower bound is tight
theorem min_value_achievable : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x/y + y/z + z/x + y/x + z/y + x/z = 10 ∧
    (x/y + y/z + z/x) * (y/x + z/y + x/z) + (x/y * y/z * z/x) = 44 := by
  sorry

end min_value_theorem_min_value_achievable_l297_29751


namespace hyperbola_foci_l297_29743

/-- The hyperbola equation -/
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (1 + k^2) - y^2 / (8 - k^2) = 1

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(-3, 0), (3, 0)}

theorem hyperbola_foci (k : ℝ) (h : 1 + k^2 > 0) :
  ∃ (x y : ℝ), hyperbola_equation x y k →
  (x, y) ∈ foci_coordinates :=
sorry

end hyperbola_foci_l297_29743


namespace longest_side_of_obtuse_consecutive_integer_triangle_l297_29710

-- Define a triangle with consecutive integer side lengths
def ConsecutiveIntegerSidedTriangle (a b c : ℕ) : Prop :=
  (b = a + 1) ∧ (c = b + 1) ∧ (a ≥ 1)

-- Define an obtuse triangle
def ObtuseTriangle (a b c : ℕ) : Prop :=
  (a^2 + b^2 < c^2) ∨ (a^2 + c^2 < b^2) ∨ (b^2 + c^2 < a^2)

theorem longest_side_of_obtuse_consecutive_integer_triangle :
  ∀ a b c : ℕ,
  ConsecutiveIntegerSidedTriangle a b c →
  ObtuseTriangle a b c →
  c = 4 :=
sorry

end longest_side_of_obtuse_consecutive_integer_triangle_l297_29710


namespace greatest_valid_integer_l297_29720

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 24 = 4

theorem greatest_valid_integer : 
  (∀ m, is_valid m → m ≤ 140) ∧ is_valid 140 :=
sorry

end greatest_valid_integer_l297_29720


namespace divisibility_of_947B_l297_29726

-- Define a function to check if a number is divisible by 3
def divisible_by_three (n : ℕ) : Prop := n % 3 = 0

-- Define a function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Theorem statement
theorem divisibility_of_947B (B : ℕ) : 
  B < 10 →  -- B is a single digit
  (∀ (n : ℕ), divisible_by_three n ↔ divisible_by_three (sum_of_digits n)) →  -- Divisibility rule
  (divisible_by_three (9000 + 400 + 70 + B) ↔ (B = 1 ∨ B = 4 ∨ B = 7)) :=
by sorry

end divisibility_of_947B_l297_29726


namespace sum_product_inequality_l297_29738

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end sum_product_inequality_l297_29738


namespace necessary_not_sufficient_condition_l297_29787

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b → a > b - 1) ∧
  (∃ a b : ℝ, a > b - 1 ∧ ¬(a > b)) :=
by sorry

end necessary_not_sufficient_condition_l297_29787


namespace tom_ran_median_distance_l297_29770

def runners : Finset String := {"Phil", "Tom", "Pete", "Amal", "Sanjay"}

def distance : String → ℝ
| "Phil" => 4
| "Tom" => 6
| "Pete" => 2
| "Amal" => 8
| "Sanjay" => 7
| _ => 0

def isMedian (x : ℝ) (s : Finset ℝ) : Prop :=
  2 * (s.filter (· ≤ x)).card ≥ s.card ∧
  2 * (s.filter (· ≥ x)).card ≥ s.card

theorem tom_ran_median_distance :
  isMedian (distance "Tom") (runners.image distance) :=
sorry

end tom_ran_median_distance_l297_29770


namespace triangle_perimeter_is_50_l297_29798

/-- Represents a triangle with specific side lengths -/
structure Triangle where
  left_side : ℝ
  right_side : ℝ
  base : ℝ

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  t.left_side + t.right_side + t.base

/-- Theorem: The perimeter of a triangle with given conditions is 50 cm -/
theorem triangle_perimeter_is_50 :
  ∀ t : Triangle,
    t.left_side = 12 →
    t.right_side = t.left_side + 2 →
    t.base = 24 →
    perimeter t = 50 := by
  sorry

end triangle_perimeter_is_50_l297_29798


namespace calculator_decimal_correction_l297_29769

theorem calculator_decimal_correction (x y : ℚ) (z : ℕ) :
  x = 0.065 →
  y = 3.25 →
  z = 21125 →
  (x * y : ℚ) = 0.21125 :=
by
  sorry

end calculator_decimal_correction_l297_29769


namespace cow_calf_cost_problem_l297_29765

theorem cow_calf_cost_problem (total_cost calf_cost cow_cost : ℕ) : 
  total_cost = 990 →
  cow_cost = 8 * calf_cost →
  total_cost = cow_cost + calf_cost →
  cow_cost = 880 := by
sorry

end cow_calf_cost_problem_l297_29765


namespace concyclic_projections_l297_29774

/-- Four points are concyclic if they lie on the same circle. -/
def Concyclic (A B C D : Point) : Prop := sorry

/-- The orthogonal projection of a point onto a line. -/
def OrthogonalProjection (P Q R : Point) : Point := sorry

/-- The main theorem: if A, B, C, D are concyclic, and A', C' are orthogonal projections of A, C 
    onto BD, and B', D' are orthogonal projections of B, D onto AC, then A', B', C', D' are concyclic. -/
theorem concyclic_projections 
  (A B C D : Point) 
  (h_concyclic : Concyclic A B C D) 
  (A' : Point) (h_A' : A' = OrthogonalProjection A B D)
  (C' : Point) (h_C' : C' = OrthogonalProjection C B D)
  (B' : Point) (h_B' : B' = OrthogonalProjection B A C)
  (D' : Point) (h_D' : D' = OrthogonalProjection D A C) :
  Concyclic A' B' C' D' :=
sorry

end concyclic_projections_l297_29774


namespace geometric_sequence_property_not_necessary_condition_l297_29762

/-- A sequence is geometric if the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_sequence_property :
  ∀ a b c d : ℝ,
  (∃ s : ℕ → ℝ, IsGeometricSequence s ∧ s 0 = a ∧ s 1 = b ∧ s 2 = c ∧ s 3 = d) →
  a * d = b * c :=
sorry

theorem not_necessary_condition :
  ∃ a b c d : ℝ, a * d = b * c ∧
  ¬(∃ s : ℕ → ℝ, IsGeometricSequence s ∧ s 0 = a ∧ s 1 = b ∧ s 2 = c ∧ s 3 = d) :=
sorry

end geometric_sequence_property_not_necessary_condition_l297_29762
