import Mathlib

namespace NUMINAMATH_CALUDE_candy_bar_cost_l463_46368

theorem candy_bar_cost (soft_drink_cost candy_bar_count total_spent : ℕ) :
  soft_drink_cost = 2 →
  candy_bar_count = 5 →
  total_spent = 27 →
  ∃ (candy_bar_cost : ℕ), candy_bar_cost * candy_bar_count + soft_drink_cost = total_spent ∧ candy_bar_cost = 5 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l463_46368


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l463_46319

/-- 
Given a two-digit number where the difference between the original number 
and the number with interchanged digits is 27, prove that the difference 
between the two digits of the number is 3.
-/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 27 → x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l463_46319


namespace NUMINAMATH_CALUDE_odd_square_not_sum_of_five_odd_squares_l463_46333

theorem odd_square_not_sum_of_five_odd_squares :
  ∀ n a b c d e : ℤ,
  Odd n → Odd a → Odd b → Odd c → Odd d → Odd e →
  ¬(n^2 ≡ a^2 + b^2 + c^2 + d^2 + e^2 [ZMOD 8]) :=
by sorry

end NUMINAMATH_CALUDE_odd_square_not_sum_of_five_odd_squares_l463_46333


namespace NUMINAMATH_CALUDE_arithmetic_sequence_specific_terms_l463_46341

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + seq.diff * (n - 1)

/-- Theorem: In an arithmetic sequence where the 4th term is 23 and the 6th term is 47, the 8th term is 71 -/
theorem arithmetic_sequence_specific_terms
  (seq : ArithmeticSequence)
  (h4 : seq.nthTerm 4 = 23)
  (h6 : seq.nthTerm 6 = 47) :
  seq.nthTerm 8 = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_specific_terms_l463_46341


namespace NUMINAMATH_CALUDE_sandy_puppies_count_l463_46332

/-- Given that Sandy initially had 8 puppies and gave away 4,
    prove that she now has 4 puppies. -/
theorem sandy_puppies_count (initial_puppies : Nat) (given_away : Nat) :
  initial_puppies = 8 → given_away = 4 → initial_puppies - given_away = 4 := by
  sorry

end NUMINAMATH_CALUDE_sandy_puppies_count_l463_46332


namespace NUMINAMATH_CALUDE_sandys_shorts_expense_l463_46328

/-- Given Sandy's shopping expenses, calculate the amount spent on shorts -/
theorem sandys_shorts_expense (total shirt jacket : ℚ)
  (h_total : total = 33.56)
  (h_shirt : shirt = 12.14)
  (h_jacket : jacket = 7.43) :
  total - shirt - jacket = 13.99 := by
  sorry

end NUMINAMATH_CALUDE_sandys_shorts_expense_l463_46328


namespace NUMINAMATH_CALUDE_cubic_equation_properties_l463_46348

/-- Theorem about cubic equations and their roots -/
theorem cubic_equation_properties (p q x₀ a b : ℝ) 
  (h1 : x₀^3 + p*x₀ + q = 0)  -- x₀ is a root of the cubic equation
  (h2 : ∀ x, x^3 + p*x + q = (x - x₀)*(x^2 + a*x + b)) :  -- Factorization of the cubic
  (a = x₀) ∧ (p^2 ≥ 4*x₀*q) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_properties_l463_46348


namespace NUMINAMATH_CALUDE_pepper_difference_l463_46345

/-- Represents the types of curry based on spice level -/
inductive CurryType
| VerySpicy
| Spicy
| Mild

/-- Returns the number of peppers needed for a given curry type -/
def peppersNeeded (c : CurryType) : ℕ :=
  match c with
  | .VerySpicy => 3
  | .Spicy => 2
  | .Mild => 1

/-- Calculates the total number of peppers needed for a given number of curries of each type -/
def totalPeppers (verySpicy spicy mild : ℕ) : ℕ :=
  verySpicy * peppersNeeded CurryType.VerySpicy +
  spicy * peppersNeeded CurryType.Spicy +
  mild * peppersNeeded CurryType.Mild

/-- The main theorem stating the difference in peppers bought -/
theorem pepper_difference : 
  totalPeppers 30 30 10 - totalPeppers 0 15 90 = 40 := by
  sorry

#eval totalPeppers 30 30 10 - totalPeppers 0 15 90

end NUMINAMATH_CALUDE_pepper_difference_l463_46345


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l463_46305

/-- A structure representing a line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A structure representing a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Containment relation of a line in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Theorem: If line a is parallel to line b, line a is not contained in plane α,
    and line b is contained in plane α, then line a is parallel to plane α -/
theorem line_parallel_to_plane (a b : Line3D) (α : Plane3D) :
  parallel_lines a b → ¬line_in_plane a α → line_in_plane b α → parallel_line_plane a α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l463_46305


namespace NUMINAMATH_CALUDE_maximize_x_cube_y_fourth_l463_46386

/-- 
Given positive real numbers x and y such that x + y = 50,
x^3 * y^4 is maximized when x = 150/7 and y = 200/7.
-/
theorem maximize_x_cube_y_fourth (x y : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (sum_xy : x + y = 50) :
  x^3 * y^4 ≤ (150/7)^3 * (200/7)^4 ∧ 
  x^3 * y^4 = (150/7)^3 * (200/7)^4 ↔ x = 150/7 ∧ y = 200/7 := by
  sorry

#check maximize_x_cube_y_fourth

end NUMINAMATH_CALUDE_maximize_x_cube_y_fourth_l463_46386


namespace NUMINAMATH_CALUDE_cos_300_degrees_l463_46358

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l463_46358


namespace NUMINAMATH_CALUDE_jenny_travel_distance_l463_46301

/-- The distance from Jenny's home to her friend's place in miles -/
def total_distance : ℝ := 155

/-- Jenny's initial speed in miles per hour -/
def initial_speed : ℝ := 45

/-- Jenny's increased speed in miles per hour -/
def increased_speed : ℝ := 65

/-- The time Jenny stops at the store in hours -/
def stop_time : ℝ := 0.25

/-- The total travel time in hours -/
def total_time : ℝ := 3.4375

theorem jenny_travel_distance :
  (initial_speed * (total_time + 1) = total_distance) ∧
  (total_distance - initial_speed = increased_speed * (total_time - stop_time - 1)) ∧
  (total_distance = initial_speed * (total_time + 1)) :=
sorry

end NUMINAMATH_CALUDE_jenny_travel_distance_l463_46301


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l463_46379

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - 4*m + 3)
  z.re = 0 ∧ z.im ≠ 0 → m = -1 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l463_46379


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l463_46360

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, 3 * x^2 - 2 * x - 8 < 0 ↔ -4/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l463_46360


namespace NUMINAMATH_CALUDE_count_triangles_eq_29_l463_46381

/-- The number of non-similar triangles with angles (in degrees) that are distinct
    positive integers in an arithmetic progression with an even common difference -/
def count_triangles : ℕ :=
  let angle_sum := 180
  let middle_angle := angle_sum / 3
  let max_difference := middle_angle - 1
  (max_difference / 2)

theorem count_triangles_eq_29 : count_triangles = 29 := by
  sorry

end NUMINAMATH_CALUDE_count_triangles_eq_29_l463_46381


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l463_46350

theorem complex_magnitude_problem (z : ℂ) : z = (5 * Complex.I) / (2 + Complex.I) - 3 * Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l463_46350


namespace NUMINAMATH_CALUDE_sin_2x_derivative_l463_46327

theorem sin_2x_derivative (x : ℝ) : 
  deriv (λ x => Real.sin (2 * x)) x = 2 * Real.cos (2 * x) := by
sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_l463_46327


namespace NUMINAMATH_CALUDE_fraction_simplification_l463_46307

theorem fraction_simplification (a b c : ℕ+) (h : Nat.gcd (a * b) c = Nat.gcd a (b * c)) :
  let (a', c') := (a.val / Nat.gcd a.val c.val, c.val / Nat.gcd a.val c.val)
  Nat.gcd a' c' = 1 ∧ Nat.gcd a' b.val = 1 ∧ Nat.gcd c' b.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l463_46307


namespace NUMINAMATH_CALUDE_triangle_area_l463_46365

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  1/2 * b * Real.cos A = Real.sin B →
  a = 2 * Real.sqrt 3 →
  b + c = 6 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l463_46365


namespace NUMINAMATH_CALUDE_one_square_remains_l463_46361

/-- Represents a grid with its dimensions and number of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (squares : ℕ)

/-- Represents the state of points on the grid -/
structure GridState :=
  (grid : Grid)
  (removed_points : ℕ)
  (remaining_squares : ℕ)

/-- Function to calculate the number of additional points to remove -/
def pointsToRemove (initial : GridState) (target : ℕ) : ℕ :=
  sorry

theorem one_square_remains (g : Grid) (initial : GridState) : 
  g.rows = 4 ∧ g.cols = 4 ∧ g.squares = 30 ∧ 
  initial.grid = g ∧ initial.removed_points = 4 →
  pointsToRemove initial 1 = 4 :=
sorry

end NUMINAMATH_CALUDE_one_square_remains_l463_46361


namespace NUMINAMATH_CALUDE_triangle_properties_l463_46354

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = 3)
  (h2 : (Real.cos t.A / Real.cos t.B) + (Real.sin t.A / Real.sin t.B) = 2 * t.c / t.b)
  (h3 : t.A + t.B + t.C = Real.pi)
  (h4 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h5 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) :
  t.B = Real.pi / 3 ∧ 
  (∀ (t' : Triangle), t'.b = 3 → t'.a + t'.b + t'.c ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l463_46354


namespace NUMINAMATH_CALUDE_weight_of_larger_square_l463_46322

/-- Represents the properties of a square piece of wood -/
structure WoodSquare where
  side : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two wood squares of different sizes -/
theorem weight_of_larger_square
  (small : WoodSquare)
  (large : WoodSquare)
  (h1 : small.side = 4)
  (h2 : small.weight = 16)
  (h3 : large.side = 6)
  (h4 : large.weight = (large.side^2 / small.side^2) * small.weight) :
  large.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_larger_square_l463_46322


namespace NUMINAMATH_CALUDE_room_length_calculation_l463_46325

/-- Given a room with specified width, total paving cost, and paving rate per square meter,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 3.75 →
  total_cost = 16500 →
  rate_per_sqm = 800 →
  (total_cost / rate_per_sqm) / width = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_room_length_calculation_l463_46325


namespace NUMINAMATH_CALUDE_student_allowance_l463_46393

/-- Proves that the student's weekly allowance is $3.00 given the spending pattern described. -/
theorem student_allowance (allowance : ℝ) : 
  (2/5 : ℝ) * allowance + 
  (1/3 : ℝ) * ((3/5 : ℝ) * allowance) + 
  1.20 = allowance → 
  allowance = 3 :=
by sorry

end NUMINAMATH_CALUDE_student_allowance_l463_46393


namespace NUMINAMATH_CALUDE_inequality_proof_l463_46337

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l463_46337


namespace NUMINAMATH_CALUDE_alphabet_letter_count_l463_46304

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (line_only : ℕ) :
  total = 50 →
  both = 16 →
  line_only = 30 →
  ∃ (dot_only : ℕ),
    dot_only = total - (both + line_only) ∧
    dot_only = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_alphabet_letter_count_l463_46304


namespace NUMINAMATH_CALUDE_fourth_vertex_exists_l463_46349

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a quadrilateral is inscribed in a circle -/
def isInscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is circumscribed around a circle -/
def isCircumscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

theorem fourth_vertex_exists (A B C : Point) 
  (h_inscribed : isInscribed ⟨A, B, C, sorry⟩)
  (h_circumscribed : isCircumscribed ⟨A, B, C, sorry⟩)
  (h_AB_ge_BC : distance A B ≥ distance B C) :
  ∃ (D : Point), 
    isInscribed ⟨A, B, C, D⟩ ∧ 
    isCircumscribed ⟨A, B, C, D⟩ :=
by
  sorry

#check fourth_vertex_exists

end NUMINAMATH_CALUDE_fourth_vertex_exists_l463_46349


namespace NUMINAMATH_CALUDE_log_domain_intersection_l463_46318

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem log_domain_intersection :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_log_domain_intersection_l463_46318


namespace NUMINAMATH_CALUDE_sequence_property_l463_46312

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n > 0) →
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → a n * (n * a n - a (n + 1)) = (n + 1) * (a (n + 1))^2) →
  ∀ n : ℕ, n > 0 → a n = 1 / n :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l463_46312


namespace NUMINAMATH_CALUDE_percent_relation_l463_46344

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 2.5 * a) :
  c = 0.1 * b := by sorry

end NUMINAMATH_CALUDE_percent_relation_l463_46344


namespace NUMINAMATH_CALUDE_expression1_equals_4_expression2_equals_neg6_l463_46302

-- Define the expressions
def expression1 : ℚ := (-36) * (1/3 - 1/2) + 16 / ((-2)^3)
def expression2 : ℚ := (-5 + 2) * (1/3) + 5^2 / (-5)

-- Theorem statements
theorem expression1_equals_4 : expression1 = 4 := by sorry

theorem expression2_equals_neg6 : expression2 = -6 := by sorry

end NUMINAMATH_CALUDE_expression1_equals_4_expression2_equals_neg6_l463_46302


namespace NUMINAMATH_CALUDE_tan_difference_absolute_value_l463_46351

theorem tan_difference_absolute_value (α β : Real) : 
  (∃ x y : Real, x^2 - 2*x - 4 = 0 ∧ y^2 - 2*y - 4 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  |Real.tan (α - β)| = 2 * Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_absolute_value_l463_46351


namespace NUMINAMATH_CALUDE_game_ends_after_17_rounds_l463_46375

/-- Represents a player in the token game -/
structure Player where
  name : String
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : List Player
  rounds : ℕ

/-- Determines if the game has ended -/
def gameEnded (state : GameState) : Bool :=
  state.players.any (fun p => p.tokens = 0)

/-- Updates the game state for one round -/
def updateGameState (state : GameState) : GameState :=
  sorry -- Implementation details omitted

/-- Runs the game until it ends -/
def runGame (initialState : GameState) : ℕ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the game ends after 17 rounds -/
theorem game_ends_after_17_rounds :
  let initialState := GameState.mk
    [Player.mk "A" 20, Player.mk "B" 18, Player.mk "C" 16]
    0
  runGame initialState = 17 := by
  sorry

end NUMINAMATH_CALUDE_game_ends_after_17_rounds_l463_46375


namespace NUMINAMATH_CALUDE_b_recurrence_l463_46347

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | 2 => 5
  | (n + 3) => (a (n + 2) * a (n + 1) - 2) / a n

def b (n : ℕ) : ℚ := a (2 * n)

theorem b_recurrence : ∀ n : ℕ, b (n + 2) - 4 * b (n + 1) + b n = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_recurrence_l463_46347


namespace NUMINAMATH_CALUDE_regular_square_prism_volume_l463_46359

theorem regular_square_prism_volume (h : ℝ) (sa : ℝ) (v : ℝ) : 
  h = 2 →
  sa = 12 * Real.pi →
  (∃ (r : ℝ), sa = 4 * Real.pi * r^2 ∧ 
    ∃ (a : ℝ), (2*r)^2 = 2*a^2 + h^2 ∧ 
    v = a^2 * h) →
  v = 8 := by sorry

end NUMINAMATH_CALUDE_regular_square_prism_volume_l463_46359


namespace NUMINAMATH_CALUDE_complex_equation_solution_l463_46382

theorem complex_equation_solution (a : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a + i) * (1 + i) = 2 * i) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l463_46382


namespace NUMINAMATH_CALUDE_robin_gum_count_l463_46385

/-- The number of gum pieces Robin has after his purchases -/
def total_gum_pieces : ℕ :=
  let initial_packages := 27
  let initial_pieces_per_package := 18
  let additional_packages_1 := 15
  let additional_pieces_per_package_1 := 12
  let additional_packages_2 := 8
  let additional_pieces_per_package_2 := 25
  initial_packages * initial_pieces_per_package +
  additional_packages_1 * additional_pieces_per_package_1 +
  additional_packages_2 * additional_pieces_per_package_2

theorem robin_gum_count : total_gum_pieces = 866 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l463_46385


namespace NUMINAMATH_CALUDE_probability_both_selected_l463_46370

theorem probability_both_selected (X Y : ℝ) (hX : X = 1/7) (hY : Y = 2/9) :
  X * Y = 2/63 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l463_46370


namespace NUMINAMATH_CALUDE_no_function_satisfies_property_l463_46387

-- Define the property that we want to disprove
def HasProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) = x^2 - 2

-- Theorem stating that no such function exists
theorem no_function_satisfies_property :
  ¬ ∃ f : ℝ → ℝ, HasProperty f :=
sorry

end NUMINAMATH_CALUDE_no_function_satisfies_property_l463_46387


namespace NUMINAMATH_CALUDE_intersection_M_N_l463_46308

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 2) ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l463_46308


namespace NUMINAMATH_CALUDE_sum_local_values_2345_l463_46388

/-- The local value of a digit in a number based on its position -/
def local_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

/-- The sum of local values of digits in a four-digit number -/
def sum_local_values (d₁ d₂ d₃ d₄ : ℕ) : ℕ :=
  local_value d₁ 3 + local_value d₂ 2 + local_value d₃ 1 + local_value d₄ 0

/-- Theorem: The sum of local values of digits in 2345 is 2345 -/
theorem sum_local_values_2345 : sum_local_values 2 3 4 5 = 2345 := by
  sorry

#eval sum_local_values 2 3 4 5

end NUMINAMATH_CALUDE_sum_local_values_2345_l463_46388


namespace NUMINAMATH_CALUDE_equation_solution_l463_46397

theorem equation_solution (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  (x + y) / x = 2 * y / (x + y) + 1 ↔ x = y ∨ x = -3 * y :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l463_46397


namespace NUMINAMATH_CALUDE_two_rotations_top_left_to_top_right_l463_46376

/-- Represents the corners of a rectangle --/
inductive Corner
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the rotation of a rectangle around a regular pentagon --/
def rotateAroundPentagon (n : ℕ) (startCorner : Corner) : Corner :=
  match n % 4 with
  | 0 => startCorner
  | 1 => match startCorner with
    | Corner.TopLeft => Corner.TopRight
    | Corner.TopRight => Corner.BottomRight
    | Corner.BottomRight => Corner.BottomLeft
    | Corner.BottomLeft => Corner.TopLeft
  | 2 => match startCorner with
    | Corner.TopLeft => Corner.BottomRight
    | Corner.TopRight => Corner.BottomLeft
    | Corner.BottomRight => Corner.TopLeft
    | Corner.BottomLeft => Corner.TopRight
  | 3 => match startCorner with
    | Corner.TopLeft => Corner.BottomLeft
    | Corner.TopRight => Corner.TopLeft
    | Corner.BottomRight => Corner.TopRight
    | Corner.BottomLeft => Corner.BottomRight
  | _ => startCorner  -- This case should never occur due to % 4

/-- Theorem stating that after two full rotations, an object at the top left corner ends up at the top right corner --/
theorem two_rotations_top_left_to_top_right :
  rotateAroundPentagon 2 Corner.TopLeft = Corner.TopRight :=
by sorry


end NUMINAMATH_CALUDE_two_rotations_top_left_to_top_right_l463_46376


namespace NUMINAMATH_CALUDE_book_pricing_problem_l463_46329

/-- Proves that the cost price is approximately 64% of the marked price
    given the conditions of the book pricing problem. -/
theorem book_pricing_problem (MP CP : ℝ) : 
  MP > 0 → -- Marked price is positive
  CP > 0 → -- Cost price is positive
  MP * 0.88 = 1.375 * CP → -- Condition after applying discount and gain
  ∃ ε > 0, |CP / MP - 0.64| < ε := by
sorry


end NUMINAMATH_CALUDE_book_pricing_problem_l463_46329


namespace NUMINAMATH_CALUDE_stacy_heather_walking_problem_l463_46355

/-- The problem of Stacy and Heather walking towards each other -/
theorem stacy_heather_walking_problem 
  (total_distance : ℝ) 
  (heather_speed : ℝ) 
  (stacy_speed : ℝ) 
  (heather_distance : ℝ) :
  total_distance = 15 →
  heather_speed = 5 →
  stacy_speed = heather_speed + 1 →
  heather_distance = 5.7272727272727275 →
  ∃ (time_difference : ℝ), 
    time_difference = 24 / 60 ∧ 
    time_difference * stacy_speed = total_distance - (heather_distance + stacy_speed * (heather_distance / heather_speed)) :=
by sorry

end NUMINAMATH_CALUDE_stacy_heather_walking_problem_l463_46355


namespace NUMINAMATH_CALUDE_relation_abc_l463_46362

theorem relation_abc : 
  let a := (2 : ℝ) ^ (1/5 : ℝ)
  let b := (2/5 : ℝ) ^ (1/5 : ℝ)
  let c := (2/5 : ℝ) ^ (3/5 : ℝ)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relation_abc_l463_46362


namespace NUMINAMATH_CALUDE_f_decreasing_iff_b_leq_neg_two_l463_46377

/-- A piecewise function f parameterized by b -/
noncomputable def f (b : ℝ) : ℝ → ℝ := fun x =>
  if x < 0 then x^2 + (2+b)*x - 1 else (2*b-1)*x + b - 2

/-- f is decreasing on ℝ if and only if b ≤ -2 -/
theorem f_decreasing_iff_b_leq_neg_two (b : ℝ) :
  (∀ x y : ℝ, x < y → f b x > f b y) ↔ b ≤ -2 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_iff_b_leq_neg_two_l463_46377


namespace NUMINAMATH_CALUDE_probability_of_double_domino_l463_46334

/-- Represents a domino tile with two integers -/
structure Domino :=
  (a b : ℕ)

/-- The set of all possible domino tiles -/
def dominoSet : Set Domino :=
  {d : Domino | d.a ≤ 12 ∧ d.b ≤ 12}

/-- A domino is considered a double if both numbers are the same -/
def isDouble (d : Domino) : Prop :=
  d.a = d.b

/-- The number of unique domino tiles in the complete set -/
def totalDominos : ℕ :=
  (13 * 14) / 2

/-- The number of double dominos in the complete set -/
def doubleDominos : ℕ := 13

theorem probability_of_double_domino :
  (doubleDominos : ℚ) / totalDominos = 13 / 91 :=
sorry

end NUMINAMATH_CALUDE_probability_of_double_domino_l463_46334


namespace NUMINAMATH_CALUDE_simplify_inverse_sum_product_l463_46339

theorem simplify_inverse_sum_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplify_inverse_sum_product_l463_46339


namespace NUMINAMATH_CALUDE_workshop_task_completion_l463_46373

/-- Calculates the average number of parts to be processed per day for the remaining days -/
def average_parts_per_day (total_parts : ℕ) (total_days : ℕ) (initial_days : ℕ) (initial_parts_per_day : ℕ) : ℚ :=
  let remaining_parts := total_parts - initial_days * initial_parts_per_day
  let remaining_days := total_days - initial_days
  (remaining_parts : ℚ) / remaining_days

theorem workshop_task_completion :
  average_parts_per_day 190 10 2 15 = 20 := by
  sorry

#eval average_parts_per_day 190 10 2 15

end NUMINAMATH_CALUDE_workshop_task_completion_l463_46373


namespace NUMINAMATH_CALUDE_negative_third_greater_than_negative_half_l463_46326

theorem negative_third_greater_than_negative_half : -1/3 > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_negative_third_greater_than_negative_half_l463_46326


namespace NUMINAMATH_CALUDE_norris_balance_proof_l463_46314

/-- Calculates the total savings with interest for Norris --/
def total_savings_with_interest (savings : List ℚ) (interest_rate : ℚ) : ℚ :=
  let base_savings := savings.sum
  let interest := 
    savings.take 4 -- Exclude January's savings from interest calculation
      |> List.scanl (λ acc x => acc + x) 0
      |> List.tail!
      |> List.map (λ x => x * interest_rate)
      |> List.sum
  base_savings + interest

/-- Calculates Norris's final balance --/
def norris_final_balance (savings : List ℚ) (interest_rate : ℚ) (loan_amount : ℚ) (repayment : ℚ) : ℚ :=
  total_savings_with_interest savings interest_rate + (loan_amount - repayment)

theorem norris_balance_proof (savings : List ℚ) (interest_rate : ℚ) (loan_amount : ℚ) (repayment : ℚ) :
  savings = [29, 25, 31, 35, 40] ∧ 
  interest_rate = 2 / 100 ∧
  loan_amount = 20 ∧
  repayment = 10 →
  norris_final_balance savings interest_rate loan_amount repayment = 175.76 := by
  sorry

end NUMINAMATH_CALUDE_norris_balance_proof_l463_46314


namespace NUMINAMATH_CALUDE_sojas_book_progress_l463_46306

theorem sojas_book_progress (pages_finished : ℕ) (total_pages : ℕ) : 
  (pages_finished = total_pages - pages_finished + 100) →
  (total_pages = 300) →
  (pages_finished : ℚ) / total_pages = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sojas_book_progress_l463_46306


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_25_l463_46336

theorem smallest_divisible_by_18_and_25 : Nat.lcm 18 25 = 450 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_25_l463_46336


namespace NUMINAMATH_CALUDE_star_four_eight_two_l463_46317

/-- The ⋆ operation for positive integers -/
def star (a b c : ℕ+) : ℚ :=
  (a * b + c) / (a + b + c)

/-- Theorem stating that 4 ⋆ 8 ⋆ 2 = 17/7 -/
theorem star_four_eight_two :
  star 4 8 2 = 17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_star_four_eight_two_l463_46317


namespace NUMINAMATH_CALUDE_functional_equation_solution_l463_46340

theorem functional_equation_solution (f : ℚ → ℝ) 
  (h : ∀ x y : ℚ, f (x + y) = f x + f y + 2 * x * y) : 
  ∃ k : ℝ, ∀ x : ℚ, f x = x^2 + k * x := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l463_46340


namespace NUMINAMATH_CALUDE_cos_180_degrees_l463_46303

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l463_46303


namespace NUMINAMATH_CALUDE_leifs_apples_l463_46363

def num_oranges : ℕ := 24 -- 2 dozen oranges

theorem leifs_apples :
  ∃ (num_apples : ℕ), num_apples = num_oranges - 10 ∧ num_apples = 14 :=
by sorry

end NUMINAMATH_CALUDE_leifs_apples_l463_46363


namespace NUMINAMATH_CALUDE_average_rounds_is_three_l463_46367

/-- Represents the number of golfers who played a certain number of rounds -/
def GolferDistribution := List (ℕ × ℕ)

/-- Calculates the total number of rounds played by all golfers -/
def totalRounds (dist : GolferDistribution) : ℕ :=
  dist.foldl (fun acc (rounds, golfers) => acc + rounds * golfers) 0

/-- Calculates the total number of golfers -/
def totalGolfers (dist : GolferDistribution) : ℕ :=
  dist.foldl (fun acc (_, golfers) => acc + golfers) 0

/-- Rounds a rational number to the nearest integer -/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem average_rounds_is_three (golfData : GolferDistribution) 
  (h : golfData = [(1, 4), (2, 3), (3, 6), (4, 2), (5, 4), (6, 1)]) : 
  roundToNearest (totalRounds golfData / totalGolfers golfData) = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_rounds_is_three_l463_46367


namespace NUMINAMATH_CALUDE_people_liking_neither_sport_l463_46389

/-- Given a class with the following properties:
  * There are 16 people in total
  * 5 people like both baseball and football
  * 2 people only like baseball
  * 3 people only like football
  Prove that 6 people like neither baseball nor football -/
theorem people_liking_neither_sport (total : Nat) (both : Nat) (only_baseball : Nat) (only_football : Nat)
  (h_total : total = 16)
  (h_both : both = 5)
  (h_only_baseball : only_baseball = 2)
  (h_only_football : only_football = 3) :
  total - (both + only_baseball + only_football) = 6 := by
sorry

end NUMINAMATH_CALUDE_people_liking_neither_sport_l463_46389


namespace NUMINAMATH_CALUDE_archery_score_proof_l463_46371

/-- Represents the score of hitting a region in the archery target -/
structure RegionScore where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the score of an archer -/
def archerScore (rs : RegionScore) (r1 r2 : Fin 3) : ℕ :=
  match r1, r2 with
  | 0, 0 => rs.A + rs.A
  | 0, 1 | 1, 0 => rs.A + rs.B
  | 0, 2 | 2, 0 => rs.A + rs.C
  | 1, 1 => rs.B + rs.B
  | 1, 2 | 2, 1 => rs.B + rs.C
  | 2, 2 => rs.C + rs.C

theorem archery_score_proof (rs : RegionScore) 
  (h1 : archerScore rs 2 0 = 15)  -- First archer: C and A
  (h2 : archerScore rs 2 1 = 18)  -- Second archer: C and B
  (h3 : archerScore rs 1 0 = 13)  -- Third archer: B and A
  : archerScore rs 1 1 = 16 :=    -- Fourth archer: B and B
by sorry

end NUMINAMATH_CALUDE_archery_score_proof_l463_46371


namespace NUMINAMATH_CALUDE_siblings_age_ratio_l463_46372

theorem siblings_age_ratio : 
  ∀ (aaron_age henry_age sister_age : ℕ),
  aaron_age = 15 →
  sister_age = 3 * aaron_age →
  aaron_age + henry_age + sister_age = 240 →
  henry_age / sister_age = 4 := by
sorry

end NUMINAMATH_CALUDE_siblings_age_ratio_l463_46372


namespace NUMINAMATH_CALUDE_minimum_speed_x_l463_46395

/-- Minimum speed problem for vehicle X --/
theorem minimum_speed_x (distance_xy distance_xz speed_y speed_z : ℝ) 
  (h1 : distance_xy = 500)
  (h2 : distance_xz = 300)
  (h3 : speed_y = 40)
  (h4 : speed_z = 30)
  (h5 : speed_y > speed_z)
  (speed_x : ℝ) :
  speed_x > 135 ↔ distance_xz / (speed_x - speed_z) < distance_xy / (speed_x + speed_y) :=
by sorry

end NUMINAMATH_CALUDE_minimum_speed_x_l463_46395


namespace NUMINAMATH_CALUDE_last_number_systematic_sampling_l463_46357

/-- Systematic sampling function -/
def systematicSampling (totalEmployees : ℕ) (sampleSize : ℕ) (firstNumber : ℕ) : ℕ :=
  let interval := totalEmployees / sampleSize
  firstNumber + (sampleSize - 1) * interval

/-- Theorem: Last number in systematic sampling -/
theorem last_number_systematic_sampling :
  systematicSampling 1000 50 15 = 995 := by
  sorry

#eval systematicSampling 1000 50 15

end NUMINAMATH_CALUDE_last_number_systematic_sampling_l463_46357


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l463_46338

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  (n = 3^15 + 11^13) → (Nat.minFac n = 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l463_46338


namespace NUMINAMATH_CALUDE_necklaces_given_to_friends_l463_46342

theorem necklaces_given_to_friends (initial : ℕ) (sold : ℕ) (remaining : ℕ) :
  initial = 60 →
  sold = 16 →
  remaining = 26 →
  initial - sold - remaining = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_necklaces_given_to_friends_l463_46342


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_1_is_45_degrees_l463_46392

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_angle_at_1_is_45_degrees :
  let slope := f' 1
  let angle := Real.arctan slope
  angle = π/4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_1_is_45_degrees_l463_46392


namespace NUMINAMATH_CALUDE_curve_is_parabola_l463_46343

theorem curve_is_parabola (θ : Real) (r : Real → Real) (x y : Real) :
  (r θ = 1 / (1 - Real.sin θ)) →
  (x^2 + y^2 = r θ^2) →
  (y = r θ * Real.sin θ) →
  (x^2 = 2*y + 1) :=
by
  sorry

#check curve_is_parabola

end NUMINAMATH_CALUDE_curve_is_parabola_l463_46343


namespace NUMINAMATH_CALUDE_base_prime_repr_450_l463_46309

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- The base prime representation of 450 is [1, 2, 2] -/
theorem base_prime_repr_450 : base_prime_repr 450 = [1, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_repr_450_l463_46309


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l463_46374

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (4 * x - 7 * y = -20) ∧ 
    (9 * x + 3 * y = -21) ∧ 
    (x = -69/25) ∧ 
    (y = 32/25) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l463_46374


namespace NUMINAMATH_CALUDE_monica_savings_l463_46323

def weekly_savings : ℕ := 15
def weeks_to_fill : ℕ := 60
def repetitions : ℕ := 5

theorem monica_savings : weekly_savings * weeks_to_fill * repetitions = 4500 := by
  sorry

end NUMINAMATH_CALUDE_monica_savings_l463_46323


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l463_46366

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (abs x - 2) / (x^2 - 4*x + 4) = 0 → x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l463_46366


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l463_46311

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the events
def EventAGetsRed (d : Distribution) : Prop := d Person.A = Card.Red
def EventBGetsBlue (d : Distribution) : Prop := d Person.B = Card.Blue

-- State the theorem
theorem events_mutually_exclusive_not_opposite :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventAGetsRed d ∧ EventBGetsBlue d)) ∧
  -- The events are not opposite
  (∃ d : Distribution, ¬EventAGetsRed d ∧ ¬EventBGetsBlue d) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l463_46311


namespace NUMINAMATH_CALUDE_average_lawn_cuts_per_month_l463_46353

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months Mr. Roper cuts his lawn 15 times per month -/
def high_frequency_months : ℕ := 6

/-- The number of months Mr. Roper cuts his lawn 3 times per month -/
def low_frequency_months : ℕ := 6

/-- The number of times Mr. Roper cuts his lawn in high frequency months -/
def high_frequency_cuts : ℕ := 15

/-- The number of times Mr. Roper cuts his lawn in low frequency months -/
def low_frequency_cuts : ℕ := 3

/-- Theorem stating that the average number of times Mr. Roper cuts his lawn per month is 9 -/
theorem average_lawn_cuts_per_month :
  (high_frequency_months * high_frequency_cuts + low_frequency_months * low_frequency_cuts) / months_in_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_lawn_cuts_per_month_l463_46353


namespace NUMINAMATH_CALUDE_mona_unique_players_l463_46300

/-- Represents the number of groups Mona joined --/
def total_groups : ℕ := 18

/-- Represents the number of groups where Mona encountered 2 previous players --/
def groups_with_two_previous : ℕ := 6

/-- Represents the number of groups where Mona encountered 1 previous player --/
def groups_with_one_previous : ℕ := 4

/-- Represents the number of players in the first large group --/
def first_large_group : ℕ := 9

/-- Represents the number of previous players in the first large group --/
def previous_in_first_large : ℕ := 4

/-- Represents the number of players in the second large group --/
def second_large_group : ℕ := 12

/-- Represents the number of previous players in the second large group --/
def previous_in_second_large : ℕ := 5

/-- Theorem stating that Mona grouped with at least 20 unique players --/
theorem mona_unique_players : ℕ := by
  sorry

end NUMINAMATH_CALUDE_mona_unique_players_l463_46300


namespace NUMINAMATH_CALUDE_conic_section_properties_l463_46331

-- Define the equation C
def C (x y k : ℝ) : Prop := x^2 / (16 + k) - y^2 / (9 - k) = 1

-- Theorem statement
theorem conic_section_properties :
  -- The equation cannot represent a circle
  (∀ k : ℝ, ¬∃ r : ℝ, ∀ x y : ℝ, C x y k ↔ x^2 + y^2 = r^2) ∧
  -- When k > 9, the equation represents an ellipse with foci on the x-axis
  (∀ k : ℝ, k > 9 → ∃ a b : ℝ, a > b ∧ b > 0 ∧ ∀ x y : ℝ, C x y k ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  -- When -16 < k < 9, the equation represents a hyperbola with foci on the x-axis
  (∀ k : ℝ, -16 < k ∧ k < 9 → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, C x y k ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
  -- When the equation represents an ellipse or a hyperbola, the focal distance is always 10
  (∀ k : ℝ, (k > 9 ∨ (-16 < k ∧ k < 9)) → 
    ∃ c : ℝ, c = 5 ∧ 
    (∀ x y : ℝ, C x y k → 
      (k > 9 → ∃ a b : ℝ, a > b ∧ b > 0 ∧ c^2 = a^2 - b^2) ∧
      (-16 < k ∧ k < 9 → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ c^2 = a^2 + b^2))) :=
sorry

end NUMINAMATH_CALUDE_conic_section_properties_l463_46331


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l463_46335

theorem subtraction_of_decimals : (25.50 : ℝ) - 3.245 = 22.255 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l463_46335


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l463_46380

theorem quadratic_equation_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k^2 * x₁^2 + (2*k - 1) * x₁ + 1 = 0 ∧ k^2 * x₂^2 + (2*k - 1) * x₂ + 1 = 0) →
  (k < 1/4 ∧ k ≠ 0) ∧
  ¬∃ (k : ℝ), ∃ (x : ℝ), k^2 * x^2 + (2*k - 1) * x + 1 = 0 ∧ k^2 * (-x)^2 + (2*k - 1) * (-x) + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l463_46380


namespace NUMINAMATH_CALUDE_basketball_team_selection_l463_46321

theorem basketball_team_selection (n m k : ℕ) (h1 : n = 18) (h2 : m = 2) (h3 : k = 8) :
  Nat.choose (n - m) (k - m) = 8008 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l463_46321


namespace NUMINAMATH_CALUDE_length_AB_with_given_eccentricity_and_focal_length_max_major_axis_length_with_perpendicular_vectors_l463_46346

noncomputable section

-- Define the line and ellipse
def line (x : ℝ) : ℝ := -x + 1
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the intersection points
def intersection_points (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse a b p.1 p.2 ∧ p.2 = line p.1}

-- Define the eccentricity and focal length
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)
def focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

-- Define the length of a line segment
def segment_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define perpendicularity of vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Part 1
theorem length_AB_with_given_eccentricity_and_focal_length :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  eccentricity a b = Real.sqrt 3 / 3 →
  focal_length a b = 1 →
  ∃ A B : ℝ × ℝ,
    A ∈ intersection_points a b ∧
    B ∈ intersection_points a b ∧
    segment_length A B = 8 * Real.sqrt 3 / 5 :=
sorry

-- Part 2
theorem max_major_axis_length_with_perpendicular_vectors :
  ∃ max_length : ℝ,
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  (∃ A B : ℝ × ℝ,
    A ∈ intersection_points a b ∧
    B ∈ intersection_points a b ∧
    perpendicular A B) →
  (1/2 : ℝ) ≤ eccentricity a b ∧ eccentricity a b ≤ Real.sqrt 2 / 2 →
  2 * a ≤ max_length ∧
  max_length = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_length_AB_with_given_eccentricity_and_focal_length_max_major_axis_length_with_perpendicular_vectors_l463_46346


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l463_46356

theorem sqrt_product_equality (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (16 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (30 * x) = 30) : 
  x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l463_46356


namespace NUMINAMATH_CALUDE_salary_after_raise_l463_46330

/-- 
Given an original salary and a percentage increase, 
calculate the new salary after a raise.
-/
theorem salary_after_raise 
  (original_salary : ℝ) 
  (percentage_increase : ℝ) 
  (new_salary : ℝ) : 
  original_salary = 55 ∧ 
  percentage_increase = 9.090909090909092 ∧
  new_salary = original_salary * (1 + percentage_increase / 100) →
  new_salary = 60 :=
by sorry

end NUMINAMATH_CALUDE_salary_after_raise_l463_46330


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l463_46391

/-- Given a polynomial function g(x) = ax^7 + bx^3 + dx^2 + cx - 8,
    prove that if g(-7) = 3 and d = 0, then g(7) = -19 -/
theorem polynomial_symmetry (a b c d : ℝ) :
  let g := λ x : ℝ => a * x^7 + b * x^3 + d * x^2 + c * x - 8
  (g (-7) = 3) → (d = 0) → (g 7 = -19) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l463_46391


namespace NUMINAMATH_CALUDE_library_loans_l463_46315

theorem library_loans (init_a init_b current_a current_b : ℕ) 
  (return_rate_a return_rate_b : ℚ) : 
  init_a = 75 → 
  init_b = 100 → 
  current_a = 54 → 
  current_b = 82 → 
  return_rate_a = 65 / 100 → 
  return_rate_b = 1 / 2 → 
  ∃ (loaned_a loaned_b : ℕ), 
    loaned_a + loaned_b = 96 ∧ 
    (init_a - current_a : ℚ) = (1 - return_rate_a) * loaned_a ∧
    (init_b - current_b : ℚ) = (1 - return_rate_b) * loaned_b :=
by sorry

end NUMINAMATH_CALUDE_library_loans_l463_46315


namespace NUMINAMATH_CALUDE_average_price_per_movie_l463_46384

theorem average_price_per_movie :
  let dvd_count : ℕ := 8
  let dvd_price : ℚ := 12
  let bluray_count : ℕ := 4
  let bluray_price : ℚ := 18
  let total_spent : ℚ := dvd_count * dvd_price + bluray_count * bluray_price
  let total_movies : ℕ := dvd_count + bluray_count
  (total_spent / total_movies : ℚ) = 14 := by
sorry

end NUMINAMATH_CALUDE_average_price_per_movie_l463_46384


namespace NUMINAMATH_CALUDE_jenny_mike_chocolate_l463_46316

theorem jenny_mike_chocolate (mike_squares jenny_squares : ℕ) 
  (h1 : mike_squares = 20) 
  (h2 : jenny_squares = 65) : 
  jenny_squares - 3 * mike_squares = 5 := by
  sorry

end NUMINAMATH_CALUDE_jenny_mike_chocolate_l463_46316


namespace NUMINAMATH_CALUDE_three_Z_five_equals_fourteen_l463_46352

-- Define the operation Z
def Z (a b : ℤ) : ℤ := b + 11*a - a^2 - a*b

-- Theorem statement
theorem three_Z_five_equals_fourteen : Z 3 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_five_equals_fourteen_l463_46352


namespace NUMINAMATH_CALUDE_monica_students_l463_46369

/-- Represents the number of students in each class and the overlaps between classes -/
structure ClassData where
  class1 : ℕ
  class2 : ℕ
  class3 : ℕ
  class4 : ℕ
  class5 : ℕ
  class6 : ℕ
  overlap12 : ℕ
  overlap45 : ℕ
  overlap236 : ℕ
  overlap56 : ℕ

/-- Calculates the number of individual students Monica sees each day -/
def individualStudents (data : ClassData) : ℕ :=
  data.class1 + data.class2 + data.class3 + data.class4 + data.class5 + data.class6 -
  (data.overlap12 + data.overlap45 + data.overlap236 + data.overlap56)

/-- Theorem stating that Monica sees 114 individual students each day -/
theorem monica_students :
  ∀ (data : ClassData),
    data.class1 = 20 ∧
    data.class2 = 25 ∧
    data.class3 = 25 ∧
    data.class4 = 10 ∧
    data.class5 = 28 ∧
    data.class6 = 28 ∧
    data.overlap12 = 5 ∧
    data.overlap45 = 3 ∧
    data.overlap236 = 6 ∧
    data.overlap56 = 8 →
    individualStudents data = 114 :=
by
  sorry


end NUMINAMATH_CALUDE_monica_students_l463_46369


namespace NUMINAMATH_CALUDE_abs_sum_lt_abs_diff_when_product_negative_l463_46383

theorem abs_sum_lt_abs_diff_when_product_negative (a b : ℝ) (h : a * b < 0) :
  |a + b| < |a - b| := by sorry

end NUMINAMATH_CALUDE_abs_sum_lt_abs_diff_when_product_negative_l463_46383


namespace NUMINAMATH_CALUDE_largest_angle_of_convex_hexagon_consecutive_angles_l463_46313

-- Define a type for convex hexagons with consecutive integer angle measures
structure ConvexHexagonConsecutiveAngles where
  -- The smallest angle measure
  smallest_angle : ℕ
  -- Ensure the hexagon is convex (all angles are less than 180°)
  convex : smallest_angle + 5 < 180

-- Define the sum of interior angles of a hexagon
def hexagon_angle_sum : ℕ := 720

-- Theorem statement
theorem largest_angle_of_convex_hexagon_consecutive_angles 
  (h : ConvexHexagonConsecutiveAngles) : 
  h.smallest_angle + 5 = 122 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_of_convex_hexagon_consecutive_angles_l463_46313


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l463_46398

theorem perfect_square_quadratic (m : ℝ) : 
  (∀ x, ∃ y, 9 - m * x + x^2 = y^2) → (m = 6 ∨ m = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l463_46398


namespace NUMINAMATH_CALUDE_jennifer_score_l463_46320

/-- Calculates the score for a modified AMC 8 contest -/
def calculateScore (totalQuestions correctAnswers incorrectAnswers unansweredQuestions : ℕ) : ℤ :=
  2 * correctAnswers - incorrectAnswers

/-- Proves that Jennifer's score in the modified AMC 8 contest is 20 points -/
theorem jennifer_score :
  let totalQuestions : ℕ := 30
  let correctAnswers : ℕ := 15
  let incorrectAnswers : ℕ := 10
  let unansweredQuestions : ℕ := 5
  calculateScore totalQuestions correctAnswers incorrectAnswers unansweredQuestions = 20 := by
  sorry

#eval calculateScore 30 15 10 5

end NUMINAMATH_CALUDE_jennifer_score_l463_46320


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l463_46394

theorem smallest_number_divisibility (x : ℕ) : x = 6297 ↔ 
  (∀ y : ℕ, (y + 3) % 18 = 0 ∧ (y + 3) % 70 = 0 ∧ (y + 3) % 100 = 0 ∧ (y + 3) % 84 = 0 → y ≥ x) ∧
  (x + 3) % 18 = 0 ∧ (x + 3) % 70 = 0 ∧ (x + 3) % 100 = 0 ∧ (x + 3) % 84 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l463_46394


namespace NUMINAMATH_CALUDE_letters_difference_l463_46399

/-- The number of letters Jack received in the morning -/
def morning_letters : ℕ := 8

/-- The number of letters Jack received in the afternoon -/
def afternoon_letters : ℕ := 7

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 2

theorem letters_difference : morning_letters - afternoon_letters = 1 := by
  sorry

end NUMINAMATH_CALUDE_letters_difference_l463_46399


namespace NUMINAMATH_CALUDE_range_of_a_l463_46324

/-- A function f : ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f : ℝ → ℝ is increasing on [0, +∞) -/
def IsIncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

/-- The condition that f(ax+1) ≤ f(x-2) holds for all x in [1/2, 1] -/
def Condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, 1/2 ≤ x → x ≤ 1 → f (a*x + 1) ≤ f (x - 2)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
    (h1 : IsEven f)
    (h2 : IsIncreasingOnNonnegative f)
    (h3 : Condition f a) :
  -2 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l463_46324


namespace NUMINAMATH_CALUDE_simplify_expression_l463_46390

theorem simplify_expression : (8 * 10^7) / (4 * 10^2) = 200000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l463_46390


namespace NUMINAMATH_CALUDE_pillar_height_D_equals_19_l463_46364

/-- Regular hexagon with pillars -/
structure HexagonWithPillars where
  sideLength : ℝ
  pillarHeightA : ℝ
  pillarHeightB : ℝ
  pillarHeightC : ℝ

/-- Calculate the height of pillar D given a hexagon with pillars -/
def pillarHeightD (h : HexagonWithPillars) : ℝ :=
  sorry

/-- Theorem stating the height of pillar D for the given hexagon -/
theorem pillar_height_D_equals_19 (h : HexagonWithPillars) 
  (h_side : h.sideLength = 10)
  (h_A : h.pillarHeightA = 15)
  (h_B : h.pillarHeightB = 12)
  (h_C : h.pillarHeightC = 11) :
  pillarHeightD h = 19 :=
sorry

end NUMINAMATH_CALUDE_pillar_height_D_equals_19_l463_46364


namespace NUMINAMATH_CALUDE_correct_prime_sum_l463_46310

def isPrime (n : ℕ) : Prop :=
  ∃ m : ℕ, n + 2 = 2^m

def primeSum : ℕ := sorry

theorem correct_prime_sum : primeSum = 2026 := by sorry

end NUMINAMATH_CALUDE_correct_prime_sum_l463_46310


namespace NUMINAMATH_CALUDE_minutes_ratio_to_hour_l463_46396

theorem minutes_ratio_to_hour (minutes_in_hour : ℕ) (ratio : ℚ) (result : ℕ) : 
  minutes_in_hour = 60 →
  ratio = 1/5 →
  result = minutes_in_hour * ratio →
  result = 12 := by sorry

end NUMINAMATH_CALUDE_minutes_ratio_to_hour_l463_46396


namespace NUMINAMATH_CALUDE_train_length_l463_46378

/-- The length of a train given its crossing times over two platforms -/
theorem train_length (platform1_length platform2_length : ℝ)
                     (crossing_time1 crossing_time2 : ℝ)
                     (h1 : platform1_length = 200)
                     (h2 : platform2_length = 300)
                     (h3 : crossing_time1 = 15)
                     (h4 : crossing_time2 = 20) :
  ∃ (train_length : ℝ),
    train_length = 100 ∧
    (train_length + platform1_length) / crossing_time1 =
    (train_length + platform2_length) / crossing_time2 :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_l463_46378
