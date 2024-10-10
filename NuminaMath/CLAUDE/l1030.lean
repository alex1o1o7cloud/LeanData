import Mathlib

namespace fuel_cost_savings_l1030_103075

theorem fuel_cost_savings 
  (old_efficiency : ℝ) 
  (old_fuel_cost : ℝ) 
  (trip_distance : ℝ) 
  (efficiency_improvement : ℝ) 
  (fuel_cost_increase : ℝ) 
  (h1 : old_efficiency > 0)
  (h2 : old_fuel_cost > 0)
  (h3 : trip_distance = 1000)
  (h4 : efficiency_improvement = 0.6)
  (h5 : fuel_cost_increase = 0.25) :
  let new_efficiency := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost := old_fuel_cost * (1 + fuel_cost_increase)
  let old_trip_cost := (trip_distance / old_efficiency) * old_fuel_cost
  let new_trip_cost := (trip_distance / new_efficiency) * new_fuel_cost
  let savings_percentage := (old_trip_cost - new_trip_cost) / old_trip_cost * 100
  savings_percentage = 21.875 := by
sorry

end fuel_cost_savings_l1030_103075


namespace problem_statement_l1030_103007

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem problem_statement (a b : ℝ) 
  (h1 : 0 < a ∧ a < 1/2) 
  (h2 : 0 < b ∧ b < 1/2) 
  (h3 : f (1/a) + f (2/b) = 10) : 
  a + b/2 ≥ 2/7 := by
sorry

end problem_statement_l1030_103007


namespace absolute_value_comparison_l1030_103094

theorem absolute_value_comparison (m n : ℝ) : m < n → n < 0 → abs m > abs n := by
  sorry

end absolute_value_comparison_l1030_103094


namespace x_values_l1030_103047

def U : Set ℕ := Set.univ

def A (x : ℕ) : Set ℕ := {1, 4, x}

def B (x : ℕ) : Set ℕ := {1, x^2}

theorem x_values (x : ℕ) : (Set.compl (A x) ⊂ Set.compl (B x)) → (x = 0 ∨ x = 2) :=
by
  sorry

end x_values_l1030_103047


namespace load_capacity_calculation_l1030_103084

theorem load_capacity_calculation (T H : ℝ) (L : ℝ) : 
  T = 3 → H = 9 → L = (35 * T^3) / H^3 → L = 35 / 27 := by
  sorry

end load_capacity_calculation_l1030_103084


namespace perpendicular_bisector_trajectory_l1030_103005

theorem perpendicular_bisector_trajectory (Z₁ Z₂ : ℂ) (h : Z₁ ≠ Z₂) :
  {Z : ℂ | Complex.abs (Z - Z₁) = Complex.abs (Z - Z₂)} =
  {Z : ℂ | (Z - (Z₁ + Z₂) / 2) • (Z₂ - Z₁) = 0} :=
by sorry

end perpendicular_bisector_trajectory_l1030_103005


namespace circle_land_diagram_value_l1030_103076

/-- Represents a digit with circles in Circle Land -/
structure CircleDigit where
  digit : Nat
  circles : Nat

/-- Calculates the value of a CircleDigit -/
def circleValue (cd : CircleDigit) : Nat :=
  cd.digit * (10 ^ cd.circles)

/-- Represents a number in Circle Land -/
def CircleLandNumber (cds : List CircleDigit) : Nat :=
  cds.map circleValue |>.sum

/-- The specific diagram given in the problem -/
def problemDiagram : List CircleDigit :=
  [⟨3, 4⟩, ⟨1, 2⟩, ⟨5, 0⟩]

theorem circle_land_diagram_value :
  CircleLandNumber problemDiagram = 30105 := by
  sorry

end circle_land_diagram_value_l1030_103076


namespace triangle_special_angle_l1030_103074

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a^2 + b^2 - √2ab = c^2, then the measure of angle C is π/4 -/
theorem triangle_special_angle (a b c : ℝ) (h : a^2 + b^2 - Real.sqrt 2 * a * b = c^2) :
  let angle_C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  angle_C = π / 4 := by
sorry


end triangle_special_angle_l1030_103074


namespace string_average_length_l1030_103081

theorem string_average_length (s₁ s₂ s₃ : ℝ) (h₁ : s₁ = 2) (h₂ : s₂ = 5) (h₃ : s₃ = 7) :
  (s₁ + s₂ + s₃) / 3 = 14 / 3 := by
  sorry

#check string_average_length

end string_average_length_l1030_103081


namespace cubic_sum_over_product_l1030_103024

theorem cubic_sum_over_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 18)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 21 := by
sorry

end cubic_sum_over_product_l1030_103024


namespace handshake_count_gathering_handshakes_l1030_103082

theorem handshake_count (twin_sets : ℕ) (triplet_sets : ℕ) : ℕ :=
  let twins := 2 * twin_sets
  let triplets := 3 * triplet_sets
  let twin_handshakes := twins * (twins - 2) / 2
  let cross_handshakes := twins * triplets
  twin_handshakes + cross_handshakes

theorem gathering_handshakes :
  handshake_count 8 5 = 352 := by
  sorry

end handshake_count_gathering_handshakes_l1030_103082


namespace train_crossing_time_l1030_103067

/-- The time taken for a train to cross a man walking in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 600 →
  train_speed = 56 * 1000 / 3600 →
  man_speed = 2 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 40 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1030_103067


namespace min_value_problem_l1030_103010

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1/a + 1/b) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1/x + 1/y → 1/x + 2/y ≥ min :=
by sorry

end min_value_problem_l1030_103010


namespace sum_of_squares_of_roots_l1030_103014

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 2 * x₁ - 15 = 0) →
  (3 * x₂^2 - 2 * x₂ - 15 = 0) →
  (x₁^2 + x₂^2 = 94/9) := by
sorry

end sum_of_squares_of_roots_l1030_103014


namespace function_properties_l1030_103013

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + 2*a*x

def g (a b : ℝ) (x : ℝ) : ℝ := 3*a^2 * Real.log x + b

-- State the theorem
theorem function_properties (a b : ℝ) :
  (a > 0) →
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    f a x₀ = g a b x₀ ∧ 
    (deriv (f a)) x₀ = (deriv (g a b)) x₀ ∧
    a = Real.exp 1) →
  (b = -(Real.exp 1)^2 / 2) ∧
  (∀ x > 0, f a x ≥ g a b x - b) →
  (0 < a ∧ a ≤ Real.exp ((5:ℝ)/6)) :=
sorry

end function_properties_l1030_103013


namespace probability_all_visible_faces_same_color_l1030_103061

/-- Represents the three possible colors for painting the cube faces -/
inductive Color
| Red
| Blue
| Green

/-- A cube with 6 faces, each painted with a color -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- The probability of a specific color arrangement on the cube -/
def colorArrangementProbability : ℚ := (1 : ℚ) / 729

/-- Predicate to check if a cube can be placed with all visible vertical faces the same color -/
def hasAllVisibleFacesSameColor (c : Cube) : Prop := sorry

/-- The number of color arrangements where all visible vertical faces can be the same color -/
def numValidArrangements : ℕ := 57

/-- Theorem stating the probability of a cube having all visible vertical faces the same color -/
theorem probability_all_visible_faces_same_color :
  (numValidArrangements : ℚ) * colorArrangementProbability = 57 / 729 := by sorry

end probability_all_visible_faces_same_color_l1030_103061


namespace sports_books_count_l1030_103020

theorem sports_books_count (total_books school_books : ℕ) 
  (h1 : total_books = 58) 
  (h2 : school_books = 19) : 
  total_books - school_books = 39 := by
sorry

end sports_books_count_l1030_103020


namespace units_digit_of_special_two_digit_number_l1030_103083

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def units_digit (n : ℕ) : ℕ := n % 10

def digit_product (n : ℕ) : ℕ := tens_digit n * units_digit n

def digit_sum (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem units_digit_of_special_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ N = digit_product N * digit_sum N ∧ units_digit N = 8 :=
sorry

end units_digit_of_special_two_digit_number_l1030_103083


namespace min_value_zero_l1030_103054

theorem min_value_zero (x y : ℕ) (hx : x ≤ 2) (hy : y ≤ 3) :
  (x^2 * y^2 : ℝ) / (x^2 + y^2)^2 ≥ 0 ∧
  ∃ (a b : ℕ), a ≤ 2 ∧ b ≤ 3 ∧ (a^2 * b^2 : ℝ) / (a^2 + b^2)^2 = 0 :=
by sorry

end min_value_zero_l1030_103054


namespace binary_1011001100_equals_octal_5460_l1030_103043

def binary_to_octal (b : ℕ) : ℕ :=
  sorry

theorem binary_1011001100_equals_octal_5460 :
  binary_to_octal 1011001100 = 5460 := by
  sorry

end binary_1011001100_equals_octal_5460_l1030_103043


namespace simplify_expression_evaluate_expression_l1030_103039

-- Part 1
theorem simplify_expression (x y : ℝ) :
  (3 * x^2 - 2 * x * y + 5 * y^2) - 2 * (x^2 - x * y - 2 * y^2) = x^2 + 9 * y^2 := by
  sorry

-- Part 2
theorem evaluate_expression (x y : ℝ) (A B : ℝ) 
  (h1 : A = -x - 2*y - 1)
  (h2 : B = x + 2*y + 2)
  (h3 : x + 2*y = 6) :
  A + 3*B = 17 := by
  sorry

end simplify_expression_evaluate_expression_l1030_103039


namespace imaginary_part_of_complex_product_l1030_103057

theorem imaginary_part_of_complex_product (i : ℂ) :
  i * i = -1 →
  (Complex.im ((2 - 3 * i) * i) = 2) :=
by
  sorry

end imaginary_part_of_complex_product_l1030_103057


namespace square_difference_formula_l1030_103006

theorem square_difference_formula (x y A : ℝ) : 
  (3*x + 2*y)^2 = (3*x - 2*y)^2 + A → A = 24*x*y := by sorry

end square_difference_formula_l1030_103006


namespace combined_travel_time_l1030_103015

/-- 
Given a car that takes 4.5 hours to reach station B, and a train that takes 2 hours longer 
than the car to travel the same distance, the combined time for both to reach station B is 11 hours.
-/
theorem combined_travel_time (car_time train_time : ℝ) : 
  car_time = 4.5 → 
  train_time = car_time + 2 → 
  car_time + train_time = 11 := by
sorry

end combined_travel_time_l1030_103015


namespace instantaneous_velocity_at_2_l1030_103080

-- Define the displacement function
def s (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 6 * t^2

-- Theorem statement
theorem instantaneous_velocity_at_2 :
  v 2 = 24 := by
  sorry

end instantaneous_velocity_at_2_l1030_103080


namespace expression_equality_l1030_103030

theorem expression_equality : -1^4 + |1 - Real.sqrt 2| - (Real.pi - 3.14)^0 = Real.sqrt 2 - 3 := by
  sorry

end expression_equality_l1030_103030


namespace atomic_number_relationship_l1030_103036

/-- Given three elements with atomic numbers R, M, and Z, if their ions R^(X-), M^(n+), and Z^(m+) 
    have the same electronic structure, and n > m, then M > Z > R. -/
theorem atomic_number_relationship (R M Z n m x : ℤ) 
  (h1 : R + x = M - n) 
  (h2 : R + x = Z - m) 
  (h3 : n > m) : 
  M > Z ∧ Z > R := by sorry

end atomic_number_relationship_l1030_103036


namespace transformation_result_l1030_103069

def rotate_180_degrees (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

def reflect_y_equals_x (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.2, point.1)

theorem transformation_result (Q : ℝ × ℝ) :
  let rotated := rotate_180_degrees (2, 3) Q
  let reflected := reflect_y_equals_x rotated
  reflected = (4, -1) → Q.1 - Q.2 = 3 := by
sorry

end transformation_result_l1030_103069


namespace unique_solution_l1030_103046

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  x > 0 ∧ (x - 2) > 0 ∧ (x + 2) > 0 ∧
  log10 x + log10 (x - 2) = log10 3 + log10 (x + 2)

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 6 :=
sorry

end unique_solution_l1030_103046


namespace currant_yield_increase_l1030_103091

theorem currant_yield_increase (initial_yield_per_bush : ℝ) : 
  let total_yield := 15 * initial_yield_per_bush
  let new_yield_per_bush := total_yield / 12
  (new_yield_per_bush - initial_yield_per_bush) / initial_yield_per_bush * 100 = 25 := by
sorry

end currant_yield_increase_l1030_103091


namespace min_max_z_values_l1030_103011

theorem min_max_z_values (x y z : ℝ) 
  (h1 : x^2 ≤ y + z) 
  (h2 : y^2 ≤ z + x) 
  (h3 : z^2 ≤ x + y) : 
  (-1/4 : ℝ) ≤ z ∧ z ≤ 2 := by
  sorry

end min_max_z_values_l1030_103011


namespace harmonic_sum_divisibility_l1030_103008

theorem harmonic_sum_divisibility (p : ℕ) (m n : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) 
  (h_sum : (m : ℚ) / n = (Finset.range (p - 1)).sum (λ i => 1 / (i + 1 : ℚ))) :
  p ∣ m := by
  sorry

end harmonic_sum_divisibility_l1030_103008


namespace pencil_count_l1030_103028

theorem pencil_count (rows : ℕ) (pencils_per_row : ℕ) (h1 : rows = 2) (h2 : pencils_per_row = 3) :
  rows * pencils_per_row = 6 := by
  sorry

end pencil_count_l1030_103028


namespace smallest_factorization_coefficient_l1030_103066

theorem smallest_factorization_coefficient : 
  ∃ (b : ℕ), b = 95 ∧ 
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 2016 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), b' < b → 
    ¬(∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b'*x + 2016 = (x + p) * (x + q))) :=
by sorry

end smallest_factorization_coefficient_l1030_103066


namespace unruly_max_sum_squares_l1030_103035

/-- A quadratic polynomial q(x) with real coefficients a and b -/
def q (a b x : ℝ) : ℝ := x^2 - (a+b)*x + a*b - 1

/-- The condition for q to be unruly -/
def is_unruly (a b : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ∀ w, q a b (q a b w) = 0 ↔ w = x ∨ w = y ∨ w = z

/-- The sum of squares of roots of q(x) -/
def sum_of_squares (a b : ℝ) : ℝ := (a+b)^2 + 2*(a*b - 1)

/-- Theorem stating that the unruly polynomial maximizing the sum of squares of its roots satisfies q(1) = -3 -/
theorem unruly_max_sum_squares :
  ∃ (a b : ℝ), is_unruly a b ∧
    (∀ (c d : ℝ), is_unruly c d → sum_of_squares c d ≤ sum_of_squares a b) ∧
    q a b 1 = -3 :=
sorry

end unruly_max_sum_squares_l1030_103035


namespace choose_four_captains_from_twelve_l1030_103078

theorem choose_four_captains_from_twelve (n : ℕ) (k : ℕ) : n = 12 ∧ k = 4 → Nat.choose n k = 990 := by
  sorry

end choose_four_captains_from_twelve_l1030_103078


namespace remainder_equality_l1030_103068

theorem remainder_equality (a b k : ℤ) (h : k ∣ (a - b)) : a % k = b % k := by
  sorry

end remainder_equality_l1030_103068


namespace coin_flip_probability_l1030_103064

theorem coin_flip_probability :
  let p : ℝ := 1/3  -- Probability of getting heads in a single flip
  let q : ℝ := 1 - p  -- Probability of getting tails in a single flip
  let num_players : ℕ := 4  -- Number of players
  let prob_same_flips : ℝ := (p^num_players) * (∑' n, q^(num_players * n)) -- Probability all players flip same number of times
  prob_same_flips = 1/65
  := by sorry

end coin_flip_probability_l1030_103064


namespace trigonometric_sum_equals_three_halves_l1030_103099

theorem trigonometric_sum_equals_three_halves :
  Real.sin (π / 24) ^ 4 + Real.cos (5 * π / 24) ^ 4 + 
  Real.sin (19 * π / 24) ^ 4 + Real.cos (23 * π / 24) ^ 4 = 3 / 2 := by
  sorry

end trigonometric_sum_equals_three_halves_l1030_103099


namespace sqrt_inequality_l1030_103000

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end sqrt_inequality_l1030_103000


namespace library_books_theorem_l1030_103049

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being a new arrival
variable (is_new_arrival : Book → Prop)

-- Define the theorem
theorem library_books_theorem (h : ¬ (∀ b : Book, is_new_arrival b)) :
  (∃ b : Book, ¬ is_new_arrival b) ∧ (¬ ∀ b : Book, is_new_arrival b) := by
  sorry

end library_books_theorem_l1030_103049


namespace root_difference_squared_l1030_103033

theorem root_difference_squared (f g : ℝ) : 
  (6 * f^2 + 13 * f - 28 = 0) → 
  (6 * g^2 + 13 * g - 28 = 0) → 
  (f - g)^2 = 169 / 9 := by
sorry

end root_difference_squared_l1030_103033


namespace max_xy_value_l1030_103001

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 18) :
  ∀ z w : ℝ, z > 0 → w > 0 → z + w = 18 → x * y ≥ z * w ∧ x * y ≤ 81 :=
by sorry

end max_xy_value_l1030_103001


namespace inequality_range_l1030_103031

theorem inequality_range (m : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ m ∈ Set.Iio (-4) ∪ Set.Ioi 2 := by
  sorry

end inequality_range_l1030_103031


namespace stack_probability_exact_l1030_103085

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates n! (n factorial) -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of crates in the stack -/
def numCrates : ℕ := 15

/-- The dimensions of each crate -/
def crateDim : CrateDimensions := ⟨2, 5, 7⟩

/-- The target height of the stack -/
def targetHeight : ℕ := 60

/-- The total number of possible orientations for the stack -/
def totalOrientations : ℕ := 3^numCrates

/-- The number of valid orientations that result in the target height -/
def validOrientations : ℕ := 
  choose numCrates 5 * choose 10 10 +
  choose numCrates 7 * choose 8 5 * choose 3 3 +
  choose numCrates 9 * choose 6 6

/-- The probability of the stack being exactly 60ft tall -/
def stackProbability : ℚ := validOrientations / totalOrientations

theorem stack_probability_exact : 
  stackProbability = 158158 / 14348907 := by sorry

end stack_probability_exact_l1030_103085


namespace line_equation_through_point_with_angle_specific_line_equation_l1030_103032

/-- The equation of a line passing through a given point with a given inclination angle -/
theorem line_equation_through_point_with_angle (x₀ y₀ : ℝ) (θ : ℝ) :
  (x₀ = Real.sqrt 3) →
  (y₀ = -2 * Real.sqrt 3) →
  (θ = 135 * π / 180) →
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧
                 ∀ (x y : ℝ), a * x + b * y + c = 0 ↔
                               y - y₀ = Real.tan θ * (x - x₀) :=
by sorry

/-- The specific equation of the line in the problem -/
theorem specific_line_equation :
  ∃ (x y : ℝ), x + y + Real.sqrt 3 = 0 ↔
                y - (-2 * Real.sqrt 3) = Real.tan (135 * π / 180) * (x - Real.sqrt 3) :=
by sorry

end line_equation_through_point_with_angle_specific_line_equation_l1030_103032


namespace valid_speaking_orders_eq_600_l1030_103077

/-- The number of students in the class --/
def total_students : ℕ := 7

/-- The number of students to be selected for speaking --/
def selected_speakers : ℕ := 4

/-- The number of special students (A and B) --/
def special_students : ℕ := 2

/-- Function to calculate the number of valid speaking orders --/
def valid_speaking_orders : ℕ :=
  let one_special := special_students * (total_students - special_students).choose (selected_speakers - 1) * (selected_speakers).factorial
  let both_special := special_students.choose 2 * (total_students - special_students).choose (selected_speakers - 2) * (selected_speakers).factorial
  let adjacent := special_students.choose 2 * (total_students - special_students).choose (selected_speakers - 2) * (selected_speakers - 1).factorial * 2
  one_special + both_special - adjacent

/-- Theorem stating that the number of valid speaking orders is 600 --/
theorem valid_speaking_orders_eq_600 : valid_speaking_orders = 600 := by
  sorry

end valid_speaking_orders_eq_600_l1030_103077


namespace pattern_calculation_main_calculation_l1030_103002

theorem pattern_calculation : ℕ → Prop :=
  fun n => n * (n + 1) + (n + 1) * (n + 2) = 2 * (n + 1) * (n + 1)

theorem main_calculation : 
  75 * 222 + 76 * 225 - 25 * 14 * 15 - 25 * 15 * 16 = 302 := by
  sorry

end pattern_calculation_main_calculation_l1030_103002


namespace jason_commute_distance_l1030_103042

/-- Represents Jason's commute with convenience stores and a detour --/
structure JasonCommute where
  distance_house_to_first : ℝ
  distance_first_to_second : ℝ
  distance_second_to_third : ℝ
  distance_third_to_work : ℝ
  detour_distance : ℝ

/-- Calculates the total commute distance with detour --/
def total_commute_with_detour (j : JasonCommute) : ℝ :=
  j.distance_house_to_first + j.distance_first_to_second + 
  (j.distance_second_to_third + j.detour_distance) + j.distance_third_to_work

/-- Theorem stating Jason's commute distance with detour --/
theorem jason_commute_distance :
  ∀ j : JasonCommute,
  j.distance_house_to_first = 4 →
  j.distance_first_to_second = 6 →
  j.distance_second_to_third = j.distance_first_to_second + (2/3 * j.distance_first_to_second) →
  j.distance_third_to_work = j.distance_house_to_first →
  j.detour_distance = 3 →
  total_commute_with_detour j = 27 := by
  sorry

end jason_commute_distance_l1030_103042


namespace equation_holds_iff_specific_values_l1030_103055

theorem equation_holds_iff_specific_values :
  ∀ (a b p q : ℝ),
  (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
  (a = 2 ∧ b = -1 ∧ p = -1 ∧ q = 1/4) :=
by sorry

end equation_holds_iff_specific_values_l1030_103055


namespace shaded_area_theorem_l1030_103003

/-- The fraction of the total area shaded in each iteration -/
def shaded_fraction : ℚ := 4 / 6

/-- The fraction of the remaining area subdivided in each iteration -/
def subdivision_fraction : ℚ := 1 / 6

/-- The sum of the shaded areas in an infinitely divided rectangle -/
def shaded_area_sum : ℚ := shaded_fraction / (1 - subdivision_fraction)

/-- 
Theorem: The sum of the shaded area in an infinitely divided rectangle, 
where 4/6 of each central subdivision is shaded in each iteration, 
is equal to 4/5 of the total area.
-/
theorem shaded_area_theorem : shaded_area_sum = 4 / 5 := by
  sorry

end shaded_area_theorem_l1030_103003


namespace birdhouse_planks_l1030_103004

/-- The number of planks required to build one birdhouse -/
def planks_per_birdhouse : ℕ := sorry

/-- The number of nails required to build one birdhouse -/
def nails_per_birdhouse : ℕ := 20

/-- The cost of one nail in cents -/
def nail_cost : ℕ := 5

/-- The cost of one plank in cents -/
def plank_cost : ℕ := 300

/-- The total cost to build 4 birdhouses in cents -/
def total_cost_4_birdhouses : ℕ := 8800

theorem birdhouse_planks :
  planks_per_birdhouse = 7 ∧
  4 * (nails_per_birdhouse * nail_cost + planks_per_birdhouse * plank_cost) = total_cost_4_birdhouses :=
sorry

end birdhouse_planks_l1030_103004


namespace volunteers_distribution_count_l1030_103029

def number_of_volunteers : ℕ := 5
def number_of_schools : ℕ := 3

/-- The number of ways to distribute volunteers to schools -/
def distribute_volunteers : ℕ := sorry

theorem volunteers_distribution_count :
  distribute_volunteers = 150 := by sorry

end volunteers_distribution_count_l1030_103029


namespace locus_is_straight_line_l1030_103058

-- Define the fixed point A
def A : ℝ × ℝ := (1, 1)

-- Define the line l
def l (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the locus of points equidistant from A and l
def locus (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x - A.1)^2 + (y - A.2)^2 = (x + y - 2)^2 / 2

-- Theorem statement
theorem locus_is_straight_line :
  ∃ (a b c : ℝ), ∀ (x y : ℝ), locus (x, y) ↔ a*x + b*y + c = 0 :=
sorry

end locus_is_straight_line_l1030_103058


namespace units_digit_of_15_to_15_l1030_103089

theorem units_digit_of_15_to_15 : ∃ n : ℕ, 15^15 ≡ 5 [ZMOD 10] :=
sorry

end units_digit_of_15_to_15_l1030_103089


namespace souvenir_shop_properties_l1030_103027

/-- Represents the cost and profit structure of souvenirs --/
structure SouvenirShop where
  costA : ℕ → ℕ  -- Cost function for type A
  costB : ℕ → ℕ  -- Cost function for type B
  profitA : ℕ    -- Profit per piece of type A
  profitB : ℕ    -- Profit per piece of type B

/-- Theorem stating the properties of the souvenir shop problem --/
theorem souvenir_shop_properties (shop : SouvenirShop) :
  (shop.costA 7 + shop.costB 4 = 760) ∧
  (shop.costA 5 + shop.costB 8 = 800) ∧
  (shop.profitA = 30) ∧
  (shop.profitB = 20) →
  (∃ (x y : ℕ), 
    (∀ n : ℕ, shop.costA n = n * x) ∧
    (∀ n : ℕ, shop.costB n = n * y) ∧
    x = 80 ∧ 
    y = 50) ∧
  (∃ (plans : List ℕ),
    plans.length = 7 ∧
    (∀ a ∈ plans, 
      80 * a + 50 * (100 - a) ≥ 7000 ∧
      80 * a + 50 * (100 - a) ≤ 7200)) ∧
  (∃ (maxA : ℕ) (maxB : ℕ),
    maxA + maxB = 100 ∧
    maxA = 73 ∧
    maxB = 27 ∧
    ∀ a b : ℕ, 
      a + b = 100 →
      shop.profitA * a + shop.profitB * b ≤ shop.profitA * maxA + shop.profitB * maxB) :=
by sorry


end souvenir_shop_properties_l1030_103027


namespace min_mozart_bach_not_beethoven_l1030_103065

theorem min_mozart_bach_not_beethoven 
  (total : ℕ) 
  (mozart : ℕ) 
  (bach : ℕ) 
  (beethoven : ℕ) 
  (h1 : total = 200)
  (h2 : mozart = 160)
  (h3 : bach = 120)
  (h4 : beethoven = 90)
  : ∃ (x : ℕ), x ≥ 10 ∧ 
    x ≤ mozart - beethoven ∧ 
    x ≤ bach - beethoven ∧ 
    x ≤ total - beethoven ∧
    x = min (mozart - beethoven) (min (bach - beethoven) (total - beethoven)) :=
by sorry

end min_mozart_bach_not_beethoven_l1030_103065


namespace negative_five_is_square_root_of_twenty_five_l1030_103017

theorem negative_five_is_square_root_of_twenty_five : ∃ x : ℝ, x^2 = 25 ∧ x = -5 := by
  sorry

end negative_five_is_square_root_of_twenty_five_l1030_103017


namespace sqrt_19_minus_1_between_3_and_4_l1030_103026

theorem sqrt_19_minus_1_between_3_and_4 :
  let a := Real.sqrt 19 - 1
  3 < a ∧ a < 4 := by sorry

end sqrt_19_minus_1_between_3_and_4_l1030_103026


namespace sum_of_roots_quadratic_l1030_103079

theorem sum_of_roots_quadratic (x : ℝ) : 
  (2 * x^2 - 5 * x + 3 = 9) → 
  (∃ y : ℝ, 2 * y^2 - 5 * y + 3 = 9 ∧ x + y = 5/2) :=
by sorry

end sum_of_roots_quadratic_l1030_103079


namespace real_roots_of_p_l1030_103095

def p (x : ℝ) : ℝ := x^4 - 3*x^3 + 3*x^2 - x - 6

theorem real_roots_of_p :
  ∃ (a b c d : ℝ), (∀ x, p x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
                   (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 1) := by
  sorry

end real_roots_of_p_l1030_103095


namespace college_students_fraction_l1030_103060

theorem college_students_fraction (total : ℕ) (h_total : total > 0) :
  let third_year := (total : ℚ) / 2
  let not_second_year := (total : ℚ) * 7 / 10
  let second_year := total - not_second_year
  let not_third_year := total - third_year
  second_year / not_third_year = 3 / 5 :=
by sorry

end college_students_fraction_l1030_103060


namespace unique_solution_power_equation_l1030_103098

theorem unique_solution_power_equation :
  ∃! (x y m n : ℕ), x > y ∧ y > 0 ∧ m > 1 ∧ n > 1 ∧ (x + y)^n = x^m + y^m :=
by
  sorry

end unique_solution_power_equation_l1030_103098


namespace sum_of_bases_l1030_103040

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The value of digit C in base 13 -/
def C : Nat := 12

theorem sum_of_bases :
  to_base_10 [7, 5, 3] 9 + to_base_10 [2, C, 4] 13 = 1129 :=
by sorry

end sum_of_bases_l1030_103040


namespace odd_sum_power_divisibility_l1030_103092

theorem odd_sum_power_divisibility
  (a b l : ℕ) 
  (h_odd_a : Odd a) 
  (h_odd_b : Odd b)
  (h_a_gt_1 : a > 1)
  (h_b_gt_1 : b > 1)
  (h_sum : a + b = 2^l) :
  ∀ k : ℕ, k > 0 → (k^2 ∣ a^k + b^k) → k = 1 :=
sorry

end odd_sum_power_divisibility_l1030_103092


namespace equation_has_two_real_roots_l1030_103071

theorem equation_has_two_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (x₁ - Real.sqrt (2 * x₁ + 6) = 2) ∧
  (x₂ - Real.sqrt (2 * x₂ + 6) = 2) ∧
  (∀ x : ℝ, x - Real.sqrt (2 * x + 6) = 2 → x = x₁ ∨ x = x₂) :=
by sorry

end equation_has_two_real_roots_l1030_103071


namespace shaded_area_of_square_with_rectangles_shaded_area_is_22_l1030_103044

/-- The area of the shaded L-shaped region in a square with three rectangles removed -/
theorem shaded_area_of_square_with_rectangles (side_length : ℝ) 
  (rect1_length rect1_width : ℝ) 
  (rect2_length rect2_width : ℝ) 
  (rect3_length rect3_width : ℝ) : ℝ :=
  side_length * side_length - (rect1_length * rect1_width + rect2_length * rect2_width + rect3_length * rect3_width)

/-- The area of the shaded L-shaped region is 22 square units -/
theorem shaded_area_is_22 :
  shaded_area_of_square_with_rectangles 6 3 1 4 2 1 3 = 22 := by
  sorry

end shaded_area_of_square_with_rectangles_shaded_area_is_22_l1030_103044


namespace parallelogram_area_24_16_l1030_103097

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_24_16 :
  parallelogramArea 24 16 = 384 := by
  sorry

end parallelogram_area_24_16_l1030_103097


namespace problem_statement_l1030_103016

theorem problem_statement (a : ℝ) (h : a = 5 - 2 * Real.sqrt 6) : a^2 - 10*a + 1 = 0 := by
  sorry

end problem_statement_l1030_103016


namespace cylinder_height_difference_l1030_103021

theorem cylinder_height_difference (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end cylinder_height_difference_l1030_103021


namespace cross_in_square_l1030_103072

/-- Given a square with side length s containing a cross made up of two squares
    with side length s/2 and two squares with side length s/4, if the total area
    of the cross is 810 cm², then s = 36 cm. -/
theorem cross_in_square (s : ℝ) :
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → s = 36 := by sorry

end cross_in_square_l1030_103072


namespace sum_of_solutions_quadratic_l1030_103052

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let a : ℝ := -32
  let b : ℝ := 84
  let c : ℝ := 135
  let eq := a * x^2 + b * x + c = 0
  let sum_of_roots := -b / a
  sum_of_roots = 21 / 8 := by
  sorry

end sum_of_solutions_quadratic_l1030_103052


namespace absolute_value_comparison_l1030_103087

theorem absolute_value_comparison (m n : ℝ) : m < n → n < 0 → abs m > abs n := by
  sorry

end absolute_value_comparison_l1030_103087


namespace carlas_apples_l1030_103038

/-- The number of apples Carla put in her backpack in the morning. -/
def initial_apples : ℕ := sorry

/-- The number of apples stolen by Buffy. -/
def stolen_apples : ℕ := 45

/-- The number of apples that fell out of the backpack. -/
def fallen_apples : ℕ := 26

/-- The number of apples remaining at lunchtime. -/
def remaining_apples : ℕ := 8

theorem carlas_apples : initial_apples = stolen_apples + fallen_apples + remaining_apples := by
  sorry

end carlas_apples_l1030_103038


namespace least_divisible_by_second_primes_l1030_103059

/-- The second set of four consecutive prime numbers -/
def second_consecutive_primes : Finset Nat := {11, 13, 17, 19}

/-- The product of the second set of four consecutive prime numbers -/
def product_of_primes : Nat := 46219

/-- Theorem stating that the product of the second set of four consecutive primes
    is the least positive whole number divisible by all of them -/
theorem least_divisible_by_second_primes :
  (∀ p ∈ second_consecutive_primes, product_of_primes % p = 0) ∧
  (∀ n : Nat, 0 < n ∧ n < product_of_primes →
    ∃ p ∈ second_consecutive_primes, n % p ≠ 0) :=
sorry

end least_divisible_by_second_primes_l1030_103059


namespace proposition_a_is_true_l1030_103041

theorem proposition_a_is_true : ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0 := by
  sorry

#check proposition_a_is_true

end proposition_a_is_true_l1030_103041


namespace eight_player_tournament_l1030_103050

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 8 players, the total number of matches is 28. -/
theorem eight_player_tournament : num_matches 8 = 28 := by
  sorry

end eight_player_tournament_l1030_103050


namespace work_completion_time_l1030_103037

/-- The number of days y needs to finish the work -/
def y_days : ℝ := 15

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 10

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 6.000000000000001

/-- The number of days x needs to finish the entire work alone -/
def x_days : ℝ := 18

theorem work_completion_time :
  y_days = 15 ∧ y_worked = 10 ∧ x_remaining = 6.000000000000001 →
  x_days = 18 :=
by sorry

end work_completion_time_l1030_103037


namespace winnie_the_pooh_honey_l1030_103093

theorem winnie_the_pooh_honey (a b c d e : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0) 
  (total : a + b + c + d + e = 3) : 
  max (a + b) (max (b + c) (max (c + d) (d + e))) ≥ 1 := by
  sorry

end winnie_the_pooh_honey_l1030_103093


namespace distance_to_circle_center_l1030_103090

/-- The distance from a point on y = 2x to the center of (x-8)^2 + (y-1)^2 = 2,
    given symmetric tangents -/
theorem distance_to_circle_center (P : ℝ × ℝ) : 
  (∃ t : ℝ, P.1 = t ∧ P.2 = 2*t) →  -- P is on the line y = 2x
  (∃ l₁ l₂ : ℝ × ℝ → Prop,  -- l₁ and l₂ are tangent lines
    (∀ Q : ℝ × ℝ, l₁ Q → (Q.1 - 8)^2 + (Q.2 - 1)^2 = 2) ∧
    (∀ Q : ℝ × ℝ, l₂ Q → (Q.1 - 8)^2 + (Q.2 - 1)^2 = 2) ∧
    l₁ P ∧ l₂ P ∧
    (∀ Q : ℝ × ℝ, l₁ Q ↔ l₂ (2*P.1 - Q.1, 2*P.2 - Q.2))) →  -- l₁ and l₂ are symmetric about y = 2x
  Real.sqrt ((P.1 - 8)^2 + (P.2 - 1)^2) = 3 * Real.sqrt 5 := by
  sorry


end distance_to_circle_center_l1030_103090


namespace base5_divisible_by_31_l1030_103073

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (a b c d : ℕ) : ℕ := a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

/-- Checks if a number is divisible by 31 --/
def isDivisibleBy31 (n : ℕ) : Prop := ∃ k : ℕ, n = 31 * k

/-- The main theorem --/
theorem base5_divisible_by_31 (x : ℕ) : 
  x < 5 → (isDivisibleBy31 (base5ToBase10 3 4 x 1) ↔ x = 4) := by
  sorry

end base5_divisible_by_31_l1030_103073


namespace integer_solutions_for_k_l1030_103063

theorem integer_solutions_for_k (k : ℤ) : 
  (∃ x : ℤ, 9 * x - 3 = k * x + 14) ↔ k ∈ ({8, 10, -8, 26} : Set ℤ) := by
sorry

end integer_solutions_for_k_l1030_103063


namespace circle_properties_l1030_103062

-- Define the line
def line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the first circle (the one we're proving)
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- Define what it means for a point to be on a circle
def on_circle (x y : ℝ) (circle : ℝ → ℝ → Prop) : Prop := circle x y

-- Define what it means for a line to be tangent to a circle
def is_tangent (circle : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), on_circle x y circle ∧ line x y ∧
  ∀ (x' y' : ℝ), line x' y' → (x' - x)^2 + (y' - y)^2 ≥ 0

-- Define what it means for two circles to intersect
def circles_intersect (circle1 circle2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), on_circle x y circle1 ∧ on_circle x y circle2

-- State the theorem
theorem circle_properties :
  is_tangent circle1 line ∧ circles_intersect circle1 circle2 := by sorry

end circle_properties_l1030_103062


namespace sum_of_squared_even_differences_l1030_103086

theorem sum_of_squared_even_differences : 
  (20^2 - 18^2) + (16^2 - 14^2) + (12^2 - 10^2) + (8^2 - 6^2) + (4^2 - 2^2) = 200 := by
  sorry

end sum_of_squared_even_differences_l1030_103086


namespace percentage_of_female_employees_l1030_103045

theorem percentage_of_female_employees (total_employees : ℕ) 
  (computer_literate_percentage : ℚ) (female_computer_literate : ℕ) 
  (male_computer_literate_percentage : ℚ) :
  total_employees = 1100 →
  computer_literate_percentage = 62 / 100 →
  female_computer_literate = 462 →
  male_computer_literate_percentage = 1 / 2 →
  (↑female_computer_literate + (male_computer_literate_percentage * ↑(total_employees - female_computer_literate / computer_literate_percentage))) / ↑total_employees = 3 / 5 := by
  sorry

#check percentage_of_female_employees

end percentage_of_female_employees_l1030_103045


namespace complex_expression_equality_logarithmic_expression_equality_l1030_103070

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

theorem complex_expression_equality : 
  (1) * (2^(1/3) * 3^(1/2))^6 + (2 * 2^(1/2))^(4/3) - 4 * (16/49)^(-1/2) - 2^(1/4) * 8^0.25 - (-2009)^0 = 100 :=
sorry

theorem logarithmic_expression_equality :
  2 * (lg (2^(1/2)))^2 + lg (2^(1/2)) + lg 5 + ((lg (2^(1/2)))^2 - lg 2 + 1)^(1/2) = 1 :=
sorry

end complex_expression_equality_logarithmic_expression_equality_l1030_103070


namespace completing_square_result_l1030_103009

theorem completing_square_result (x : ℝ) : 
  x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
  sorry

end completing_square_result_l1030_103009


namespace luke_stickers_l1030_103048

theorem luke_stickers (initial bought gift given_away used remaining : ℕ) : 
  bought = 12 →
  gift = 20 →
  given_away = 5 →
  used = 8 →
  remaining = 39 →
  initial + bought + gift - given_away - used = remaining →
  initial = 20 := by
  sorry

end luke_stickers_l1030_103048


namespace cube_volume_ratio_l1030_103018

theorem cube_volume_ratio (q p : ℝ) (h : p = 3 * q) : q^3 / p^3 = 1 / 27 := by
  sorry

end cube_volume_ratio_l1030_103018


namespace least_months_to_triple_l1030_103053

def interest_rate : ℝ := 1.06

def amount_owed (t : ℕ) : ℝ := interest_rate ^ t

def exceeds_triple (t : ℕ) : Prop := amount_owed t > 3

theorem least_months_to_triple : 
  (∀ n < 20, ¬(exceeds_triple n)) ∧ (exceeds_triple 20) :=
sorry

end least_months_to_triple_l1030_103053


namespace animal_count_l1030_103056

theorem animal_count (dogs cats frogs : ℕ) : 
  cats = (80 * dogs) / 100 →
  frogs = 2 * dogs →
  frogs = 160 →
  dogs + cats + frogs = 304 := by
sorry

end animal_count_l1030_103056


namespace license_plate_count_l1030_103096

def letter_choices : ℕ := 26
def odd_digits : ℕ := 5
def all_digits : ℕ := 10
def even_digits : ℕ := 4

theorem license_plate_count : 
  letter_choices^3 * odd_digits * all_digits * even_digits = 3514400 := by
  sorry

end license_plate_count_l1030_103096


namespace quarter_circle_radius_l1030_103019

theorem quarter_circle_radius (x y z : ℝ) (h_right_angle : x^2 + y^2 = z^2)
  (h_xy_area : π * x^2 / 4 = 2 * π) (h_xz_arc : π * y / 2 = 6 * π) :
  z / 2 = Real.sqrt 152 := by
  sorry

end quarter_circle_radius_l1030_103019


namespace perpendicular_vectors_m_equals_6_l1030_103012

def vector_a (m : ℝ) : ℝ × ℝ := (2, m)
def vector_b : ℝ × ℝ := (1, -1)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

def vector_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

theorem perpendicular_vectors_m_equals_6 :
  ∀ m : ℝ, 
    let a := vector_a m
    let b := vector_b
    let sum := vector_add a (vector_scale 2 b)
    dot_product b sum = 0 → m = 6 := by
  sorry

end perpendicular_vectors_m_equals_6_l1030_103012


namespace theater_parking_increase_l1030_103088

/-- Calculates the net increase in vehicles during a theater play --/
def net_increase_vehicles (play_duration : ℝ) 
  (car_arrival_rate car_departure_rate : ℝ)
  (motorcycle_arrival_rate motorcycle_departure_rate : ℝ)
  (van_arrival_rate van_departure_rate : ℝ) :
  (ℝ × ℝ × ℝ) :=
  let net_car_increase := (car_arrival_rate - car_departure_rate) * play_duration
  let net_motorcycle_increase := (motorcycle_arrival_rate - motorcycle_departure_rate) * play_duration
  let net_van_increase := (van_arrival_rate - van_departure_rate) * play_duration
  (net_car_increase, net_motorcycle_increase, net_van_increase)

/-- Theorem stating the net increase in vehicles during the theater play --/
theorem theater_parking_increase :
  let play_duration : ℝ := 2.5
  let car_arrival_rate : ℝ := 70
  let car_departure_rate : ℝ := 40
  let motorcycle_arrival_rate : ℝ := 120
  let motorcycle_departure_rate : ℝ := 60
  let van_arrival_rate : ℝ := 30
  let van_departure_rate : ℝ := 20
  net_increase_vehicles play_duration 
    car_arrival_rate car_departure_rate
    motorcycle_arrival_rate motorcycle_departure_rate
    van_arrival_rate van_departure_rate = (75, 150, 25) := by
  sorry

end theater_parking_increase_l1030_103088


namespace variance_linear_transformation_l1030_103025

def variance (data : List ℝ) : ℝ := sorry

theorem variance_linear_transformation 
  (data : List ℝ) 
  (h : variance data = 1/3) : 
  variance (data.map (λ x => 3*x - 1)) = 3 := by sorry

end variance_linear_transformation_l1030_103025


namespace james_tshirt_cost_l1030_103023

def calculate_total_cost (num_shirts : ℕ) (discount_rate : ℚ) (original_price : ℚ) : ℚ :=
  num_shirts * (original_price * (1 - discount_rate))

theorem james_tshirt_cost :
  calculate_total_cost 6 (1/2) 20 = 60 := by
  sorry

end james_tshirt_cost_l1030_103023


namespace cos_2x_minus_pi_4_graph_translation_l1030_103034

open Real

theorem cos_2x_minus_pi_4_graph_translation (x : ℝ) : 
  cos (2*x - π/4) = sin (2*(x + π/8)) := by sorry

end cos_2x_minus_pi_4_graph_translation_l1030_103034


namespace division_problem_l1030_103022

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 2944) (h2 : quotient = 40) (h3 : remainder = 64) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 72 :=
by sorry

end division_problem_l1030_103022


namespace sequence_growth_l1030_103051

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, i ≥ 1 → Nat.gcd (a i) (a (i + 1)) > a (i - 1)

theorem sequence_growth (a : ℕ → ℕ) (h : sequence_property a) :
  ∀ n : ℕ, a n ≥ 2^n :=
by sorry

end sequence_growth_l1030_103051
