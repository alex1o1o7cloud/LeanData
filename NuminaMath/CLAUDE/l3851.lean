import Mathlib

namespace min_reciprocal_sum_l3851_385196

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x + 1/y) ≥ 3 + 2*Real.sqrt 2 :=
by sorry

end min_reciprocal_sum_l3851_385196


namespace mildred_blocks_l3851_385114

/-- The number of blocks Mildred ends up with -/
def total_blocks (initial : ℕ) (found : ℕ) : ℕ :=
  initial + found

/-- Theorem stating that Mildred's total blocks is the sum of initial and found blocks -/
theorem mildred_blocks (initial : ℕ) (found : ℕ) :
  total_blocks initial found = initial + found := by
  sorry

end mildred_blocks_l3851_385114


namespace maximum_assignment_x_plus_v_l3851_385177

def Values : Finset ℕ := {2, 3, 4, 5}

structure Assignment where
  V : ℕ
  W : ℕ
  X : ℕ
  Y : ℕ
  h1 : V ∈ Values
  h2 : W ∈ Values
  h3 : X ∈ Values
  h4 : Y ∈ Values
  h5 : V ≠ W ∧ V ≠ X ∧ V ≠ Y ∧ W ≠ X ∧ W ≠ Y ∧ X ≠ Y

def ExpressionValue (a : Assignment) : ℕ := a.Y^a.X - a.W^a.V

def MaximumAssignment : Assignment → Prop := λ a => 
  ∀ b : Assignment, ExpressionValue a ≥ ExpressionValue b

theorem maximum_assignment_x_plus_v (a : Assignment) 
  (h : MaximumAssignment a) : a.X + a.V = 8 := by
  sorry

end maximum_assignment_x_plus_v_l3851_385177


namespace incorrect_proposition_l3851_385103

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem incorrect_proposition
  (m n : Line) (α β : Plane)
  (h1 : parallel m α)
  (h2 : perpendicular n β)
  (h3 : perpendicular_planes α β) :
  ¬ (parallel_lines m n) :=
sorry

end incorrect_proposition_l3851_385103


namespace evaluate_expression_l3851_385167

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/2) (hz : z = 8) :
  x^3 * y^4 * z = 1/128 := by sorry

end evaluate_expression_l3851_385167


namespace school_population_l3851_385130

theorem school_population (total_students : ℕ) : total_students = 400 :=
  sorry

end school_population_l3851_385130


namespace product_evaluation_l3851_385152

def product_term (n : ℕ) : ℚ := (n * (n + 2) + n) / ((n + 1)^2 : ℚ)

def product_series : ℕ → ℚ
  | 0 => 1
  | n + 1 => product_series n * product_term (n + 1)

theorem product_evaluation : 
  product_series 98 = 9800 / 9801 := by sorry

end product_evaluation_l3851_385152


namespace hyperbola_sum_l3851_385169

/-- Given a hyperbola with center (1, -1), one focus at (1, 5), one vertex at (1, 2),
    and equation ((y-k)^2 / a^2) - ((x-h)^2 / b^2) = 1,
    prove that h + k + a + b = 3√3 + 3 -/
theorem hyperbola_sum (h k a b : ℝ) : 
  h = 1 ∧ k = -1 ∧  -- center at (1, -1)
  ∃ (x y : ℝ), x = 1 ∧ y = 5 ∧  -- one focus at (1, 5)
    (y - k)^2 = (x - h)^2 + a^2 ∧  -- relationship between focus, center, and a
  ∃ (x y : ℝ), x = 1 ∧ y = 2 ∧  -- one vertex at (1, 2)
    (y - k)^2 = a^2 ∧  -- relationship between vertex, center, and a
  ∀ (x y : ℝ), ((y - k)^2 / a^2) - ((x - h)^2 / b^2) = 1  -- equation of hyperbola
  →
  h + k + a + b = 3 * Real.sqrt 3 + 3 := by
sorry

end hyperbola_sum_l3851_385169


namespace hydrochloric_acid_mixture_l3851_385181

def total_mass : ℝ := 600
def final_concentration : ℝ := 0.15
def concentration_1 : ℝ := 0.3
def concentration_2 : ℝ := 0.1
def mass_1 : ℝ := 150
def mass_2 : ℝ := 450

theorem hydrochloric_acid_mixture :
  mass_1 + mass_2 = total_mass ∧
  (concentration_1 * mass_1 + concentration_2 * mass_2) / total_mass = final_concentration :=
by sorry

end hydrochloric_acid_mixture_l3851_385181


namespace river_depth_problem_l3851_385173

/-- River depth problem -/
theorem river_depth_problem (may_depth june_depth july_depth : ℕ) : 
  may_depth = 5 →
  june_depth = may_depth + 10 →
  july_depth = june_depth * 3 →
  july_depth = 45 := by
  sorry

end river_depth_problem_l3851_385173


namespace equation_solution_l3851_385110

theorem equation_solution (m : ℤ) : 
  (∃ x : ℕ+, 2 * m * x - 8 = (m + 2) * x) → 
  m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 10 := by
  sorry

end equation_solution_l3851_385110


namespace color_p_gon_l3851_385116

theorem color_p_gon (p a : ℕ) (hp : Nat.Prime p) :
  let total_colorings := a^p
  let monochromatic_colorings := a
  let distinct_non_monochromatic := (total_colorings - monochromatic_colorings) / p
  distinct_non_monochromatic + monochromatic_colorings = (a^p - a) / p + a := by
  sorry

end color_p_gon_l3851_385116


namespace max_min_value_l3851_385191

theorem max_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let s := min x (min (y + 1/x) (1/y))
  ∃ (max_s : ℝ), max_s = Real.sqrt 2 ∧ 
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → min x' (min (y' + 1/x') (1/y')) ≤ max_s) ∧
    (s = max_s ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2) :=
by sorry

end max_min_value_l3851_385191


namespace units_digit_problem_l3851_385183

theorem units_digit_problem : ∃ n : ℕ, n % 10 = 7 ∧ 
  (((2008^2 + 2^2008)^2 + 2^(2008^2 + 2^2008)) % 10 = n % 10) := by
  sorry

end units_digit_problem_l3851_385183


namespace largest_y_coordinate_degenerate_ellipse_l3851_385118

theorem largest_y_coordinate_degenerate_ellipse :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ (x^2 / 49) + ((y - 3)^2 / 25)
  ∀ (x y : ℝ), f (x, y) = 0 → y ≤ 3 :=
by sorry

end largest_y_coordinate_degenerate_ellipse_l3851_385118


namespace longest_side_of_special_triangle_l3851_385157

-- Define a triangle with sides in arithmetic progression
structure ArithmeticTriangle where
  a : ℝ
  d : ℝ
  angle : ℝ

-- Theorem statement
theorem longest_side_of_special_triangle (t : ArithmeticTriangle) 
  (h1 : t.d = 2)
  (h2 : t.angle = 2 * π / 3) -- 120° in radians
  (h3 : (t.a + t.d)^2 = (t.a - t.d)^2 + t.a^2 - 2*(t.a - t.d)*t.a*(- 1/2)) -- Law of Cosines for 120°
  : t.a + t.d = 7 := by
  sorry

end longest_side_of_special_triangle_l3851_385157


namespace f_two_l3851_385156

/-- A linear function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The inverse function of f -/
def f_inv (x : ℝ) : ℝ := sorry

/-- f is a linear function -/
axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- f satisfies the equation f(x) = 3f^(-1)(x) + 5 -/
axiom f_equation : ∀ x, f x = 3 * f_inv x + 5

/-- f(1) = 5 -/
axiom f_one : f 1 = 5

/-- The main theorem: f(2) = 3 -/
theorem f_two : f 2 = 3 := by sorry

end f_two_l3851_385156


namespace integer_fraction_sum_l3851_385102

theorem integer_fraction_sum (n : ℕ) : n > 0 →
  (∃ (x y z : ℤ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
    x + y + z = 0 ∧ 
    (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = (1 : ℚ) / n) ↔ 
  ∃ (k : ℕ), n = 2 * k ∧ k > 0 :=
by sorry

end integer_fraction_sum_l3851_385102


namespace inequality_solution_implies_a_value_l3851_385105

/-- Given that the solution set of the inequality (ax)/(x-1) > 1 is (1, 2), prove that a = 1/2 --/
theorem inequality_solution_implies_a_value (a : ℝ) :
  (∀ x : ℝ, (1 < x ∧ x < 2) ↔ (a * x) / (x - 1) > 1) →
  a = 1/2 :=
by sorry

end inequality_solution_implies_a_value_l3851_385105


namespace sum_of_coefficients_l3851_385144

/-- The coefficient of x^2 in the original function -/
def α : ℝ := 3

/-- The coefficient of x in the original function -/
def β : ℝ := -2

/-- The constant term in the original function -/
def γ : ℝ := 4

/-- The horizontal shift of the graph (to the left) -/
def h : ℝ := 2

/-- The vertical shift of the graph (upwards) -/
def k : ℝ := 5

/-- The coefficient of x^2 in the transformed function -/
def a : ℝ := α

/-- The coefficient of x in the transformed function -/
def b : ℝ := 2 * α * h - β

/-- The constant term in the transformed function -/
def c : ℝ := α * h^2 - β * h + γ + k

/-- Theorem stating that the sum of coefficients in the transformed function equals 30 -/
theorem sum_of_coefficients : a + b + c = 30 := by sorry

end sum_of_coefficients_l3851_385144


namespace inequality_proof_l3851_385122

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * (x - y - 1) + 1 / (x^2 - 2*x*y + y^2) ≥ 1 := by
  sorry

end inequality_proof_l3851_385122


namespace sine_cosine_ratio_equals_tangent_l3851_385148

theorem sine_cosine_ratio_equals_tangent :
  (Real.sin (10 * π / 180) + Real.sin (20 * π / 180)) / 
  (Real.cos (10 * π / 180) + Real.cos (20 * π / 180)) = 
  Real.tan (15 * π / 180) := by
  sorry

end sine_cosine_ratio_equals_tangent_l3851_385148


namespace tangent_lines_theorem_l3851_385159

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

theorem tangent_lines_theorem :
  let l1 : ℝ → ℝ → Prop := λ x y => x - y - 2 = 0
  let l2 : ℝ → ℝ → Prop := λ x y => x + y + 3 = 0
  (∀ x, deriv f x = 2*x + 1) ∧
  (l1 0 (-2)) ∧
  (∃ a b, f a = b ∧ l2 a b) ∧
  (∀ x y, l1 x y → ∀ x' y', l2 x' y' → (y - (-2)) / (x - 0) * (y' - y) / (x' - x) = -1) →
  (∀ x y, l1 x y ↔ (x - y - 2 = 0)) ∧
  (∀ x y, l2 x y ↔ (x + y + 3 = 0))
:= by sorry

end tangent_lines_theorem_l3851_385159


namespace blood_expires_same_day_l3851_385119

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The factorial of 8 -/
def blood_expiration_seconds : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

/-- The day a unit of blood expires when donated at noon -/
def blood_expiration_day (donation_day : ℕ) : ℕ :=
  donation_day + (blood_expiration_seconds / seconds_per_day)

theorem blood_expires_same_day (donation_day : ℕ) :
  blood_expiration_day donation_day = donation_day := by
  sorry

end blood_expires_same_day_l3851_385119


namespace rational_cosine_summands_l3851_385107

theorem rational_cosine_summands (x : ℝ) 
  (h_S : ∃ q : ℚ, q = Real.sin (64 * x) + Real.sin (65 * x))
  (h_C : ∃ q : ℚ, q = Real.cos (64 * x) + Real.cos (65 * x)) :
  ∃ (q1 q2 : ℚ), q1 = Real.cos (64 * x) ∧ q2 = Real.cos (65 * x) :=
sorry

end rational_cosine_summands_l3851_385107


namespace probability_king_of_diamonds_l3851_385133

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- The game setup with two standard decks -/
def game_setup (d : Deck) : Prop :=
  d.cards = 52 ∧ d.ranks = 13 ∧ d.suits = 4

/-- The total number of cards in the combined deck -/
def total_cards (d : Deck) : Nat :=
  2 * d.cards

/-- The number of Kings of Diamonds in the combined deck -/
def kings_of_diamonds : Nat := 2

/-- The probability of drawing a King of Diamonds from the top of the combined deck -/
theorem probability_king_of_diamonds (d : Deck) :
  game_setup d →
  (kings_of_diamonds : ℚ) / (total_cards d) = 1 / 52 :=
by sorry

end probability_king_of_diamonds_l3851_385133


namespace permutations_theorem_l3851_385104

def alphabet_size : ℕ := 26

def excluded_words : List String := ["dog", "god", "gum", "depth", "thing"]

def permutations_without_substrings (n : ℕ) (words : List String) : ℕ :=
  n.factorial - 3 * (n - 2).factorial + 3 * (n - 6).factorial + 2 * (n - 7).factorial - (n - 9).factorial

theorem permutations_theorem :
  permutations_without_substrings alphabet_size excluded_words =
  alphabet_size.factorial - 3 * (alphabet_size - 2).factorial + 3 * (alphabet_size - 6).factorial +
  2 * (alphabet_size - 7).factorial - (alphabet_size - 9).factorial :=
by sorry

end permutations_theorem_l3851_385104


namespace shifted_sine_equals_cosine_l3851_385195

theorem shifted_sine_equals_cosine (φ : Real) (h : 0 < φ ∧ φ < π) :
  (∀ x, 2 * Real.sin (2 * x - π / 3 + φ) = 2 * Real.cos (2 * x)) ↔ φ = 5 * π / 6 := by
  sorry

end shifted_sine_equals_cosine_l3851_385195


namespace slip_4_5_in_R_l3851_385123

-- Define the set of slips
def slips : List ℝ := [1, 1.5, 1.5, 2, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 5, 5.5]

-- Define the boxes
inductive Box
| P | Q | R | S | T | U

-- Define a distribution of slips to boxes
def Distribution := Box → List ℝ

-- Define the constraint that the sum in each box is an integer
def sumIsInteger (d : Distribution) : Prop :=
  ∀ b : Box, ∃ n : ℤ, (d b).sum = n

-- Define the constraint that the sums are consecutive integers
def consecutiveSums (d : Distribution) : Prop :=
  ∃ n : ℤ, (d Box.P).sum = n ∧
           (d Box.Q).sum = n + 1 ∧
           (d Box.R).sum = n + 2 ∧
           (d Box.S).sum = n + 3 ∧
           (d Box.T).sum = n + 4 ∧
           (d Box.U).sum = n + 5

-- Define the constraint that 1 is in box U and 2 is in box Q
def fixedSlips (d : Distribution) : Prop :=
  1 ∈ d Box.U ∧ 2 ∈ d Box.Q

-- Main theorem
theorem slip_4_5_in_R (d : Distribution) 
  (h1 : d Box.P ++ d Box.Q ++ d Box.R ++ d Box.S ++ d Box.T ++ d Box.U = slips)
  (h2 : sumIsInteger d)
  (h3 : consecutiveSums d)
  (h4 : fixedSlips d) :
  4.5 ∈ d Box.R :=
sorry

end slip_4_5_in_R_l3851_385123


namespace fifth_root_division_l3851_385134

theorem fifth_root_division (x : ℝ) (h : x > 0) :
  (x ^ (1 / 3)) / (x ^ (1 / 5)) = x ^ (2 / 15) :=
sorry

end fifth_root_division_l3851_385134


namespace sin_seven_pi_sixths_l3851_385194

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1/2 := by
  sorry

end sin_seven_pi_sixths_l3851_385194


namespace time_to_bernards_house_l3851_385108

/-- Given June's biking rate and the distance to Bernard's house, prove the time to bike there --/
theorem time_to_bernards_house 
  (distance_to_julia : ℝ) 
  (time_to_julia : ℝ) 
  (distance_to_bernard : ℝ) 
  (h1 : distance_to_julia = 2) 
  (h2 : time_to_julia = 8) 
  (h3 : distance_to_bernard = 6) : 
  (time_to_julia / distance_to_julia) * distance_to_bernard = 24 := by
  sorry

end time_to_bernards_house_l3851_385108


namespace min_value_of_f_on_interval_l3851_385131

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the interval [0, 3]
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem min_value_of_f_on_interval :
  ∃ (min : ℝ), min = 0 ∧ ∀ x ∈ interval, f x ≥ min :=
sorry

end min_value_of_f_on_interval_l3851_385131


namespace bus_breakdown_time_correct_l3851_385139

/-- Represents the scenario of a school trip with a bus breakdown -/
structure BusBreakdown where
  S : ℝ  -- Distance between school and county town in km
  x : ℝ  -- Walking speed in km/minute
  t : ℝ  -- Walking time of teachers and students in minutes
  a : ℝ  -- Bus breakdown time in minutes

/-- The bus speed is 5 times the walking speed -/
def bus_speed (bd : BusBreakdown) : ℝ := 5 * bd.x

/-- The walking time satisfies the equation derived from the problem conditions -/
def walking_time_equation (bd : BusBreakdown) : Prop :=
  bd.t = bd.S / (5 * bd.x) + 20 - (bd.S - bd.x * bd.t) / (5 * bd.x)

/-- The bus breakdown time satisfies the equation derived from the problem conditions -/
def breakdown_time_equation (bd : BusBreakdown) : Prop :=
  bd.a + (2 * (bd.S - bd.x * bd.t)) / (5 * bd.x) = (2 * bd.S) / (5 * bd.x) + 30

/-- Theorem stating that given the conditions, the bus breakdown time equation holds -/
theorem bus_breakdown_time_correct (bd : BusBreakdown) 
  (h_walking_time : walking_time_equation bd) :
  breakdown_time_equation bd :=
sorry

end bus_breakdown_time_correct_l3851_385139


namespace unique_square_with_special_property_l3851_385164

/-- Checks if a number uses exactly 5 different non-zero digits in base 6 --/
def hasFiveDifferentNonZeroDigitsBase6 (n : ℕ) : Prop := sorry

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : List ℕ := sorry

/-- Moves the last digit of a number to the front --/
def moveLastToFront (n : ℕ) : ℕ := sorry

/-- Reverses the digits of a number --/
def reverseDigits (n : ℕ) : ℕ := sorry

theorem unique_square_with_special_property :
  ∃! n : ℕ,
    n ^ 2 ≤ 54321 ∧
    n ^ 2 ≥ 12345 ∧
    hasFiveDifferentNonZeroDigitsBase6 (n ^ 2) ∧
    (∃ m : ℕ, m ^ 2 = moveLastToFront (n ^ 2) ∧
              m = reverseDigits n) ∧
    n = 221 := by sorry

end unique_square_with_special_property_l3851_385164


namespace f_eval_at_one_l3851_385163

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + 2*x^3 + b*x^2 + 150*x + c

-- State the theorem
theorem f_eval_at_one (a b c : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0) →
  f b c 1 = -15640 :=
by sorry

end f_eval_at_one_l3851_385163


namespace bus_stop_problem_l3851_385165

theorem bus_stop_problem (boys girls : ℕ) : 
  (boys = 2 * (girls - 15)) →
  (girls - 15 = 5 * (boys - 45)) →
  (boys = 50 ∧ girls = 40) := by
  sorry

end bus_stop_problem_l3851_385165


namespace find_FC_l3851_385124

/-- Given a triangle ABC with point D on AC and point E on AD, prove the length of FC. -/
theorem find_FC (DC CB : ℝ) (h1 : DC = 10) (h2 : CB = 12)
  (AB AD ED : ℝ) (h3 : AB = 1/3 * AD) (h4 : ED = 2/3 * AD) : 
  ∃ (FC : ℝ), FC = 506/33 := by
  sorry

end find_FC_l3851_385124


namespace equation_solution_l3851_385101

theorem equation_solution :
  ∃ x : ℝ, (4 / 7) * (1 / 9) * x = 14 ∧ x = 220.5 := by
  sorry

end equation_solution_l3851_385101


namespace polynomial_properties_l3851_385158

def f (x : ℝ) : ℝ := x^3 - 2*x

theorem polynomial_properties :
  (∀ x y : ℚ, f x = f y → x = y) ∧
  (∃ a b : ℝ, a ≠ b ∧ f a = f b) := by
  sorry

end polynomial_properties_l3851_385158


namespace cubic_root_sum_squares_l3851_385180

theorem cubic_root_sum_squares (a b c : ℂ) : 
  (a^3 + 3*a^2 - 10*a + 5 = 0) →
  (b^3 + 3*b^2 - 10*b + 5 = 0) →
  (c^3 + 3*c^2 - 10*c + 5 = 0) →
  a^2*b^2 + b^2*c^2 + c^2*a^2 = 70 := by
sorry

end cubic_root_sum_squares_l3851_385180


namespace system_solution_l3851_385120

theorem system_solution :
  ∃ (x y : ℚ),
    (16 * x^2 + 8 * x * y + 4 * y^2 + 20 * x + 2 * y = -7) ∧
    (8 * x^2 - 16 * x * y + 2 * y^2 + 20 * x - 14 * y = -11) ∧
    (x = -3/4) ∧ (y = 1/2) := by
  sorry

end system_solution_l3851_385120


namespace calculation_proof_l3851_385155

theorem calculation_proof : 
  (5 : ℚ) / 19 * ((3 + 4 / 5) * (5 + 1 / 3) + (4 + 2 / 3) * (19 / 5)) = 10 := by
  sorry

end calculation_proof_l3851_385155


namespace new_game_cost_new_game_cost_is_8_l3851_385128

def initial_money : ℕ := 57
def toy_cost : ℕ := 4
def num_toys : ℕ := 2

theorem new_game_cost : ℕ :=
  initial_money - (toy_cost * num_toys)

#check new_game_cost

theorem new_game_cost_is_8 : new_game_cost = 8 := by
  sorry

end new_game_cost_new_game_cost_is_8_l3851_385128


namespace total_amount_proof_l3851_385171

/-- Given that r has two-thirds of the total amount with p and q, and r has 1600,
    prove that the total amount T with p, q, and r is 4000. -/
theorem total_amount_proof (T : ℝ) (r : ℝ) 
  (h1 : r = 2/3 * T)
  (h2 : r = 1600) : 
  T = 4000 := by
  sorry

end total_amount_proof_l3851_385171


namespace arithmetic_mean_odd_eq_n_l3851_385132

/-- The sum of the first n odd positive integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n^2

/-- The arithmetic mean of the first n odd positive integers -/
def arithmetic_mean_odd (n : ℕ) : ℚ := (sum_first_n_odd n : ℚ) / n

/-- Theorem: The arithmetic mean of the first n odd positive integers is equal to n -/
theorem arithmetic_mean_odd_eq_n (n : ℕ) (h : n > 0) : 
  arithmetic_mean_odd n = n := by sorry

end arithmetic_mean_odd_eq_n_l3851_385132


namespace minji_clothes_combinations_l3851_385146

theorem minji_clothes_combinations (tops : ℕ) (bottoms : ℕ) 
  (h1 : tops = 3) (h2 : bottoms = 5) : tops * bottoms = 15 := by
  sorry

end minji_clothes_combinations_l3851_385146


namespace mike_ride_distance_l3851_385185

/-- Represents the taxi fare structure -/
structure TaxiFare where
  start_fee : ℝ
  per_mile_fee : ℝ
  bridge_toll : ℝ

/-- Calculates the total fare for a given distance -/
def total_fare (fare : TaxiFare) (distance : ℝ) : ℝ :=
  fare.start_fee + fare.per_mile_fee * distance + fare.bridge_toll

theorem mike_ride_distance (mike_fare annie_fare : TaxiFare) 
  (annie_distance : ℝ) (h1 : mike_fare.start_fee = 2.5) 
  (h2 : mike_fare.per_mile_fee = 0.25) (h3 : mike_fare.bridge_toll = 0)
  (h4 : annie_fare.start_fee = 2.5) (h5 : annie_fare.per_mile_fee = 0.25) 
  (h6 : annie_fare.bridge_toll = 5) (h7 : annie_distance = 26) :
  ∃ mike_distance : ℝ, 
    total_fare mike_fare mike_distance = total_fare annie_fare annie_distance ∧ 
    mike_distance = 46 := by
  sorry

end mike_ride_distance_l3851_385185


namespace power_function_k_values_l3851_385160

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

theorem power_function_k_values (k : ℝ) :
  is_power_function (λ x => (k^2 - k - 5) * x^3) → k = 3 ∨ k = -2 := by
  sorry

end power_function_k_values_l3851_385160


namespace quiche_theorem_l3851_385126

/-- Quiche ingredients and their properties --/
structure QuicheIngredients where
  spinach_initial : ℝ
  mushrooms_initial : ℝ
  onions_initial : ℝ
  spinach_reduction : ℝ
  mushrooms_reduction : ℝ
  onions_reduction : ℝ
  cream_cheese_volume : ℝ
  cream_cheese_calories : ℝ
  eggs_volume : ℝ
  eggs_calories : ℝ
  oz_to_cup_conversion : ℝ

/-- Calculate the total volume and calorie content of the quiche --/
def quiche_properties (ingredients : QuicheIngredients) : ℝ × ℝ :=
  let cooked_spinach := ingredients.spinach_initial * ingredients.spinach_reduction
  let cooked_mushrooms := ingredients.mushrooms_initial * ingredients.mushrooms_reduction
  let cooked_onions := ingredients.onions_initial * ingredients.onions_reduction
  let total_volume_oz := cooked_spinach + cooked_mushrooms + cooked_onions + 
                         ingredients.cream_cheese_volume + ingredients.eggs_volume
  let total_volume_cups := total_volume_oz * ingredients.oz_to_cup_conversion
  let total_calories := ingredients.cream_cheese_volume * ingredients.cream_cheese_calories + 
                        ingredients.eggs_volume * ingredients.eggs_calories
  (total_volume_cups, total_calories)

/-- Theorem stating the properties of the quiche --/
theorem quiche_theorem (ingredients : QuicheIngredients) 
  (h1 : ingredients.spinach_initial = 40)
  (h2 : ingredients.mushrooms_initial = 25)
  (h3 : ingredients.onions_initial = 15)
  (h4 : ingredients.spinach_reduction = 0.2)
  (h5 : ingredients.mushrooms_reduction = 0.65)
  (h6 : ingredients.onions_reduction = 0.5)
  (h7 : ingredients.cream_cheese_volume = 6)
  (h8 : ingredients.cream_cheese_calories = 80)
  (h9 : ingredients.eggs_volume = 4)
  (h10 : ingredients.eggs_calories = 70)
  (h11 : ingredients.oz_to_cup_conversion = 0.125) :
  quiche_properties ingredients = (5.21875, 760) := by
  sorry

#eval quiche_properties {
  spinach_initial := 40,
  mushrooms_initial := 25,
  onions_initial := 15,
  spinach_reduction := 0.2,
  mushrooms_reduction := 0.65,
  onions_reduction := 0.5,
  cream_cheese_volume := 6,
  cream_cheese_calories := 80,
  eggs_volume := 4,
  eggs_calories := 70,
  oz_to_cup_conversion := 0.125
}

end quiche_theorem_l3851_385126


namespace rachel_essay_time_l3851_385154

/-- Rachel's essay writing process -/
def essay_time (pages_per_30min : ℕ) (research_time : ℕ) (total_pages : ℕ) (editing_time : ℕ) : ℕ :=
  let writing_time := 30 * total_pages / pages_per_30min
  (research_time + writing_time + editing_time) / 60

/-- Theorem: Rachel spends 5 hours completing her essay -/
theorem rachel_essay_time :
  essay_time 1 45 6 75 = 5 := by
  sorry

end rachel_essay_time_l3851_385154


namespace inequality_of_positive_reals_l3851_385175

theorem inequality_of_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  b * c / a + a * c / b + a * b / c ≥ a + b + c := by
  sorry

end inequality_of_positive_reals_l3851_385175


namespace special_sequence_characterization_l3851_385179

/-- A sequence of real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n ≤ a (n + 1)) ∧ 
  (∀ m n : ℕ, a (m^2 + n^2) = (a m)^2 + (a n)^2)

/-- The theorem stating the only possible sequences satisfying the conditions -/
theorem special_sequence_characterization (a : ℕ → ℝ) :
  SpecialSequence a →
  ((∀ n, a n = 0) ∨ (∀ n, a n = 1/2) ∨ (∀ n, a n = n)) :=
by sorry

end special_sequence_characterization_l3851_385179


namespace intersection_proof_l3851_385170

def S : Set Nat := {0, 1, 3, 5, 7, 9}

theorem intersection_proof (A B : Set Nat) 
  (h1 : S = {0, 1, 3, 5, 7, 9})
  (h2 : (S \ A) = {0, 5, 9})
  (h3 : B = {3, 5, 7}) :
  A ∩ B = {3, 7} := by
sorry

end intersection_proof_l3851_385170


namespace negation_of_universal_proposition_l3851_385197

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 < 0) := by
  sorry

end negation_of_universal_proposition_l3851_385197


namespace water_polo_team_selection_l3851_385193

/-- The number of ways to select a starting team in a water polo club. -/
theorem water_polo_team_selection (total_members : Nat) (team_size : Nat) (h1 : total_members = 20) (h2 : team_size = 9) :
  (total_members * Nat.choose (total_members - 1) (team_size - 1) * (team_size - 1)) = 12093120 := by
  sorry

end water_polo_team_selection_l3851_385193


namespace guitar_savings_l3851_385121

/-- The suggested retail price of the guitar -/
def suggested_price : ℝ := 1000

/-- The discount percentage offered by Guitar Center -/
def gc_discount : ℝ := 0.15

/-- The shipping fee charged by Guitar Center -/
def gc_shipping : ℝ := 100

/-- The discount percentage offered by Sweetwater -/
def sw_discount : ℝ := 0.10

/-- The cost of the guitar at Guitar Center -/
def gc_cost : ℝ := suggested_price * (1 - gc_discount) + gc_shipping

/-- The cost of the guitar at Sweetwater -/
def sw_cost : ℝ := suggested_price * (1 - sw_discount)

/-- The savings when buying from the cheaper store (Sweetwater) -/
theorem guitar_savings : gc_cost - sw_cost = 50 := by
  sorry

end guitar_savings_l3851_385121


namespace solution_set_when_a_neg_one_range_of_a_l3851_385136

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1|
def g (a : ℝ) (x : ℝ) : ℝ := 2 * |x| + a

-- Theorem for part (1)
theorem solution_set_when_a_neg_one :
  {x : ℝ | f x ≤ g (-1) x} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ ≥ (1/2) * g a x₀) → a ≤ 2 := by sorry

end solution_set_when_a_neg_one_range_of_a_l3851_385136


namespace problem_statement_l3851_385125

def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≤ 0}

theorem problem_statement :
  (∀ a : ℝ, B a ⊆ A → a ∈ Set.Icc 1 2) ∧
  (∀ a : ℝ, A ∩ B a = {1} → a ∈ Set.Iic 1) := by
  sorry

end problem_statement_l3851_385125


namespace share_calculation_l3851_385189

theorem share_calculation (total : ℚ) (a b c : ℚ) 
  (h_total : total = 578)
  (h_a : a = (2/3) * b)
  (h_b : b = (1/4) * c)
  (h_sum : a + b + c = total) :
  b = 102 := by
  sorry

end share_calculation_l3851_385189


namespace candy_groups_l3851_385113

theorem candy_groups (total_candies : ℕ) (group_size : ℕ) (h1 : total_candies = 30) (h2 : group_size = 3) :
  total_candies / group_size = 10 := by
  sorry

end candy_groups_l3851_385113


namespace oil_in_peanut_butter_l3851_385199

/-- Given a ratio of oil to peanuts and the total weight of peanut butter,
    calculate the amount of oil used. -/
def oil_amount (oil_ratio : ℚ) (peanut_ratio : ℚ) (total_weight : ℚ) : ℚ :=
  (oil_ratio / (oil_ratio + peanut_ratio)) * total_weight

/-- Theorem stating that for the given ratios and total weight,
    the amount of oil used is 4 ounces. -/
theorem oil_in_peanut_butter :
  oil_amount 2 8 20 = 4 := by
  sorry

end oil_in_peanut_butter_l3851_385199


namespace noahs_sales_ratio_l3851_385100

/-- Noah's painting sales problem -/
theorem noahs_sales_ratio :
  let large_price : ℕ := 60
  let small_price : ℕ := 30
  let last_month_large : ℕ := 8
  let last_month_small : ℕ := 4
  let this_month_sales : ℕ := 1200
  let last_month_sales : ℕ := large_price * last_month_large + small_price * last_month_small
  (this_month_sales : ℚ) / (last_month_sales : ℚ) = 2 := by
  sorry

end noahs_sales_ratio_l3851_385100


namespace train_length_calculation_l3851_385135

/-- The length of two trains given their speeds and overtaking time -/
theorem train_length_calculation (v1 v2 t : ℝ) (h1 : v1 = 46) (h2 : v2 = 36) (h3 : t = 27) :
  let relative_speed := (v1 - v2) * (5 / 18)
  let distance := relative_speed * t
  let train_length := distance / 2
  train_length = 37.5 := by sorry

end train_length_calculation_l3851_385135


namespace difference_sum_of_powers_of_three_l3851_385186

def S : Finset ℕ := Finset.range 11

def difference_sum (S : Finset ℕ) : ℕ :=
  S.sum (λ i => S.sum (λ j => if i < j then 3^j - 3^i else 0))

theorem difference_sum_of_powers_of_three : difference_sum S = 787484 := by
  sorry

end difference_sum_of_powers_of_three_l3851_385186


namespace expression_equality_l3851_385138

theorem expression_equality : 12 * 171 + 29 * 9 + 171 * 13 + 29 * 16 = 5000 := by
  sorry

end expression_equality_l3851_385138


namespace min_sum_with_constraint_l3851_385174

theorem min_sum_with_constraint (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 + x + 2*y + 3*z = 13/4) :
  x + y + z ≥ (-3 + Real.sqrt 22) / 2 ∧
  ∃ (x' y' z' : ℝ), 0 ≤ x' ∧ 0 ≤ y' ∧ 0 ≤ z' ∧
    x'^2 + y'^2 + z'^2 + x' + 2*y' + 3*z' = 13/4 ∧
    x' + y' + z' = (-3 + Real.sqrt 22) / 2 :=
by sorry

end min_sum_with_constraint_l3851_385174


namespace correct_calculation_l3851_385150

theorem correct_calculation : (-9)^2 / (-3)^2 = -9 := by
  sorry

end correct_calculation_l3851_385150


namespace arctan_equation_solution_l3851_385115

theorem arctan_equation_solution :
  ∃ x : ℝ, x > 0 ∧ Real.arctan (1/x) + Real.arctan (1/x^2) + Real.arctan (1/x^3) = π/4 ∧ x = 1 :=
by sorry

end arctan_equation_solution_l3851_385115


namespace clara_loses_prob_l3851_385147

/-- The probability of Clara's coin landing heads -/
def clara_heads_prob : ℚ := 2/3

/-- The probability of Ethan's coin landing heads -/
def ethan_heads_prob : ℚ := 1/4

/-- The probability of both Clara and Ethan getting tails in one round -/
def both_tails_prob : ℚ := (1 - clara_heads_prob) * (1 - ethan_heads_prob)

/-- The game where Clara and Ethan alternately toss coins until one gets a head and loses -/
def coin_toss_game : Prop :=
  ∃ (p : ℚ), p = clara_heads_prob * (1 / (1 - both_tails_prob))

/-- The theorem stating that the probability of Clara losing is 8/9 -/
theorem clara_loses_prob : 
  coin_toss_game → (∃ (p : ℚ), p = 8/9 ∧ p = clara_heads_prob * (1 / (1 - both_tails_prob))) :=
by sorry

end clara_loses_prob_l3851_385147


namespace cyclic_quadrilateral_symmetry_l3851_385153

-- Define the points
variable (A B C D A₁ B₁ C₁ D₁ P : Point)

-- Define the property of being cyclic
def is_cyclic (A B C D : Point) : Prop := sorry

-- Define symmetry with respect to a point
def symmetrical_wrt (A B : Point) (P : Point) : Prop := sorry

-- State the theorem
theorem cyclic_quadrilateral_symmetry 
  (h1 : symmetrical_wrt A A₁ P) 
  (h2 : symmetrical_wrt B B₁ P) 
  (h3 : symmetrical_wrt C C₁ P) 
  (h4 : symmetrical_wrt D D₁ P)
  (h5 : is_cyclic A₁ B C D)
  (h6 : is_cyclic A B₁ C D)
  (h7 : is_cyclic A B C₁ D) :
  is_cyclic A B C D₁ := by sorry

end cyclic_quadrilateral_symmetry_l3851_385153


namespace tied_rope_length_l3851_385112

/-- Calculates the length of a rope made by tying multiple shorter ropes together. -/
def ropeLength (n : ℕ) (ropeLength : ℕ) (knotReduction : ℕ) : ℕ :=
  n * ropeLength - (n - 1) * knotReduction

/-- Proves that tying 64 ropes of 25 cm each, with 3 cm reduction per knot, results in a 1411 cm rope. -/
theorem tied_rope_length :
  ropeLength 64 25 3 = 1411 := by
  sorry

end tied_rope_length_l3851_385112


namespace elise_remaining_money_l3851_385109

/-- Calculates the remaining money in dollars for Elise --/
def remaining_money (initial_amount : ℝ) (saved_euros : ℝ) (euro_to_dollar : ℝ) 
                    (comic_cost : ℝ) (puzzle_cost_pounds : ℝ) (pound_to_dollar : ℝ) : ℝ :=
  initial_amount + saved_euros * euro_to_dollar - comic_cost - puzzle_cost_pounds * pound_to_dollar

/-- Theorem stating that Elise's remaining money is $1.04 --/
theorem elise_remaining_money :
  remaining_money 8 11 1.18 2 13 1.38 = 1.04 := by
  sorry

end elise_remaining_money_l3851_385109


namespace pass_percentage_l3851_385172

theorem pass_percentage 
  (passed_english : Real) 
  (passed_math : Real) 
  (failed_both : Real) 
  (h1 : passed_english = 63) 
  (h2 : passed_math = 65) 
  (h3 : failed_both = 27) : 
  100 - failed_both = 73 := by
  sorry

end pass_percentage_l3851_385172


namespace at_least_two_primes_of_form_l3851_385162

theorem at_least_two_primes_of_form (n : ℕ) : ∃ (a b : ℕ), 2 ≤ a ∧ 2 ≤ b ∧ a ≠ b ∧ 
  Nat.Prime (a^3 + a^2 + 1) ∧ Nat.Prime (b^3 + b^2 + 1) := by
  sorry

end at_least_two_primes_of_form_l3851_385162


namespace reflection_coordinate_sum_l3851_385178

/-- Given a point A with coordinates (3, y) and its reflection B over the x-axis,
    the sum of all four coordinate values is 6. -/
theorem reflection_coordinate_sum (y : ℝ) : 
  let A : ℝ × ℝ := (3, y)
  let B : ℝ × ℝ := (3, -y)
  (A.1 + A.2 + B.1 + B.2) = 6 := by
  sorry

end reflection_coordinate_sum_l3851_385178


namespace larger_number_problem_l3851_385140

theorem larger_number_problem (x y : ℝ) : 4 * y = 5 * x → x + y = 54 → y = 30 := by
  sorry

end larger_number_problem_l3851_385140


namespace quadratic_equations_solutions_l3851_385168

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1 = 4 + 3 * Real.sqrt 2 ∧ x2 = 4 - 3 * Real.sqrt 2 ∧ 
    x1^2 - 8*x1 - 2 = 0 ∧ x2^2 - 8*x2 - 2 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 3/2 ∧ x2 = -1 ∧ 
    2*x1^2 - x1 - 3 = 0 ∧ 2*x2^2 - x2 - 3 = 0) :=
by sorry

end quadratic_equations_solutions_l3851_385168


namespace age_ratio_proof_l3851_385176

/-- Given three people a, b, and c, with the following conditions:
  1. a is two years older than b
  2. The total of the ages of a, b, and c is 72
  3. b is 28 years old
Prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 72 →
  b = 28 →
  b / c = 2 := by
  sorry

end age_ratio_proof_l3851_385176


namespace fraction_simplification_l3851_385182

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hsum : x + 1/y ≠ 0) :
  (x + 1/y) / (y + 1/x) = x/y := by
  sorry

end fraction_simplification_l3851_385182


namespace triangle_square_apothem_equality_l3851_385198

/-- Theorem: Value of k for a specific right triangle and square configuration -/
theorem triangle_square_apothem_equality (x : ℝ) (k : ℝ) : 
  x > 0 →  -- Ensure positive side lengths
  (3*x)^2 + (4*x)^2 = (5*x)^2 →  -- Pythagorean theorem for right triangle
  12*x = k * (6*x^2) →  -- Perimeter = k * Area for triangle
  4*x = 5 →  -- Apothem equality
  100 = 3 * 40 →  -- Square area = 3 * Square perimeter
  k = 8/5 := by sorry

end triangle_square_apothem_equality_l3851_385198


namespace real_solutions_condition_l3851_385161

theorem real_solutions_condition (a : ℝ) :
  (∃ x : ℝ, |x| + x^2 = a) ↔ a ≥ 0 := by
  sorry

end real_solutions_condition_l3851_385161


namespace course_selection_ways_l3851_385192

theorem course_selection_ways (type_a : ℕ) (type_b : ℕ) (total_selection : ℕ) : 
  type_a = 4 → type_b = 2 → total_selection = 3 →
  (Nat.choose type_a 1 * Nat.choose type_b 2) + (Nat.choose type_a 2 * Nat.choose type_b 1) = 16 := by
  sorry

end course_selection_ways_l3851_385192


namespace fourth_power_of_cube_of_third_smallest_prime_l3851_385117

def third_smallest_prime : ℕ := 5

theorem fourth_power_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 4 = 244140625 := by
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l3851_385117


namespace kayla_total_is_15_l3851_385129

def theresa_chocolate : ℕ := 12
def theresa_soda : ℕ := 18

def kayla_chocolate : ℕ := theresa_chocolate / 2
def kayla_soda : ℕ := theresa_soda / 2

def kayla_total : ℕ := kayla_chocolate + kayla_soda

theorem kayla_total_is_15 : kayla_total = 15 := by
  sorry

end kayla_total_is_15_l3851_385129


namespace expression_simplification_l3851_385190

theorem expression_simplification (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - b^2) / (a - b) = (a^3 - 3*a*b + b^3) / (a * b) := by
  sorry

end expression_simplification_l3851_385190


namespace multiples_of_12_between_15_and_250_l3851_385127

theorem multiples_of_12_between_15_and_250 : 
  (Finset.filter (λ x => x > 15 ∧ x < 250 ∧ x % 12 = 0) (Finset.range 251)).card = 19 := by
  sorry

end multiples_of_12_between_15_and_250_l3851_385127


namespace items_sold_increase_after_discount_l3851_385111

/-- Theorem: Increase in items sold after discount
  If a store offers a 10% discount on all items and their gross income increases by 3.5%,
  then the number of items sold increases by 15%.
-/
theorem items_sold_increase_after_discount (P N : ℝ) (N' : ℝ) :
  P > 0 → N > 0 →
  (0.9 * P * N' = 1.035 * P * N) →
  (N' - N) / N * 100 = 15 := by
  sorry

end items_sold_increase_after_discount_l3851_385111


namespace cards_per_player_l3851_385188

/-- Proves that evenly distributing 54 cards among 3 players results in 18 cards per player -/
theorem cards_per_player (initial_cards : ℕ) (added_cards : ℕ) (num_players : ℕ) :
  initial_cards = 52 →
  added_cards = 2 →
  num_players = 3 →
  (initial_cards + added_cards) / num_players = 18 := by
sorry

end cards_per_player_l3851_385188


namespace distance_AB_is_40_l3851_385184

/-- The distance between two points A and B -/
def distance_AB : ℝ := 40

/-- The remaining distance for the second cyclist when the first cyclist has traveled half the total distance -/
def remaining_distance_second : ℝ := 24

/-- The remaining distance for the first cyclist when the second cyclist has traveled half the total distance -/
def remaining_distance_first : ℝ := 15

/-- The theorem stating that the distance between points A and B is 40 km -/
theorem distance_AB_is_40 :
  (distance_AB / 2 + remaining_distance_second = distance_AB) ∧
  (distance_AB / 2 + remaining_distance_first = distance_AB) →
  distance_AB = 40 := by
  sorry

end distance_AB_is_40_l3851_385184


namespace function_monotonicity_l3851_385145

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_monotonicity (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : periodic_two f)
  (h_decreasing : decreasing_on f 1 2) :
  increasing_on f (-2) (-1) ∧ decreasing_on f 3 4 :=
by sorry

end function_monotonicity_l3851_385145


namespace quadratic_inequality_l3851_385151

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end quadratic_inequality_l3851_385151


namespace students_behind_yoongi_l3851_385106

/-- Given a line of students, calculates the number of students behind a specific student -/
def studentsInBack (totalStudents : ℕ) (studentsBetween : ℕ) : ℕ :=
  totalStudents - (studentsBetween + 2)

theorem students_behind_yoongi :
  let totalStudents : ℕ := 20
  let studentsBetween : ℕ := 5
  studentsInBack totalStudents studentsBetween = 13 := by
  sorry

end students_behind_yoongi_l3851_385106


namespace emily_chairs_l3851_385143

/-- The number of chairs Emily bought -/
def num_chairs : ℕ := sorry

/-- The number of tables Emily bought -/
def num_tables : ℕ := 2

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_furniture : ℕ := 8

/-- The total time spent (in minutes) -/
def total_time : ℕ := 48

theorem emily_chairs : 
  num_chairs = 4 ∧ 
  time_per_furniture * (num_chairs + num_tables) = total_time :=
sorry

end emily_chairs_l3851_385143


namespace hyperbola_k_range_l3851_385187

-- Define the equation of the hyperbola
def hyperbola_eq (x y k : ℝ) : Prop :=
  x^2 / (3 - k) - y^2 / (k - 1) = 1

-- Define the condition for k to represent a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, hyperbola_eq x y k

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ (1 < k ∧ k < 3) :=
sorry

end hyperbola_k_range_l3851_385187


namespace altitudes_constructible_l3851_385142

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle in a 2D plane -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents a circle in a 2D plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents construction tools -/
inductive ConstructionTool
  | Straightedge
  | Protractor

/-- Represents an altitude of a triangle -/
structure Altitude :=
  (base : Point)
  (apex : Point)

/-- Function to construct altitudes of a triangle -/
def constructAltitudes (t : Triangle) (c : Circle) (tools : List ConstructionTool) : 
  List Altitude :=
  sorry

/-- Theorem stating that altitudes can be constructed -/
theorem altitudes_constructible (t : Triangle) (c : Circle) : 
  ∃ (tools : List ConstructionTool), 
    (ConstructionTool.Straightedge ∈ tools) ∧ 
    (ConstructionTool.Protractor ∈ tools) ∧ 
    (constructAltitudes t c tools).length = 3 :=
  sorry

end altitudes_constructible_l3851_385142


namespace bicycling_problem_l3851_385137

/-- The number of days after which the condition is satisfied -/
def days : ℕ := 12

/-- The total distance between points A and B in kilometers -/
def total_distance : ℕ := 600

/-- The distance person A travels per day in kilometers -/
def person_a_speed : ℕ := 40

/-- The effective daily distance person B travels in kilometers -/
def person_b_speed : ℕ := 30

/-- The remaining distance for person A after the given number of days -/
def remaining_distance_a : ℕ := total_distance - person_a_speed * days

/-- The remaining distance for person B after the given number of days -/
def remaining_distance_b : ℕ := total_distance - person_b_speed * days

theorem bicycling_problem :
  remaining_distance_b = 2 * remaining_distance_a :=
sorry

end bicycling_problem_l3851_385137


namespace time_to_install_one_window_l3851_385149

theorem time_to_install_one_window
  (total_windows : ℕ)
  (installed_windows : ℕ)
  (time_for_remaining : ℕ)
  (h1 : total_windows = 14)
  (h2 : installed_windows = 5)
  (h3 : time_for_remaining = 36)
  : (time_for_remaining : ℚ) / (total_windows - installed_windows : ℚ) = 4 := by
  sorry

end time_to_install_one_window_l3851_385149


namespace tic_tac_toe_tie_probability_l3851_385166

theorem tic_tac_toe_tie_probability (amy_win : ℚ) (lily_win : ℚ) (john_win : ℚ)
  (h_amy : amy_win = 4/9)
  (h_lily : lily_win = 1/3)
  (h_john : john_win = 1/6) :
  1 - (amy_win + lily_win + john_win) = 1/18 := by
sorry

end tic_tac_toe_tie_probability_l3851_385166


namespace reflection_of_C_l3851_385141

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point C -/
def C : ℝ × ℝ := (3, 1)

theorem reflection_of_C :
  (reflect_x ∘ reflect_y) C = (-3, -1) := by sorry

end reflection_of_C_l3851_385141
