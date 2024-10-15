import Mathlib

namespace NUMINAMATH_CALUDE_no_pascal_row_with_four_distinct_elements_l1159_115994

theorem no_pascal_row_with_four_distinct_elements : 
  ¬ ∃ (n : ℕ) (k m : ℕ) (a b c d : ℕ), 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (b = Nat.choose n k) ∧
    (d = Nat.choose n m) ∧
    (a = b / 2) ∧
    (c = d / 2) :=
by sorry

end NUMINAMATH_CALUDE_no_pascal_row_with_four_distinct_elements_l1159_115994


namespace NUMINAMATH_CALUDE_third_grade_girls_l1159_115903

theorem third_grade_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 123 → boys = 66 → total = boys + girls → girls = 57 := by
  sorry

end NUMINAMATH_CALUDE_third_grade_girls_l1159_115903


namespace NUMINAMATH_CALUDE_min_cars_correct_l1159_115925

/-- Represents the minimum number of cars needed for a given number of adults -/
def min_cars (adults : ℕ) : ℕ :=
  if adults ≤ 5 then 6 else 10

/-- Each car must rest one day a week -/
axiom car_rest_day : ∀ (c : ℕ), c > 0 → ∃ (d : ℕ), d ≤ 7 ∧ c % 7 = d

/-- All adults wish to drive daily -/
axiom adults_drive_daily : ∀ (a : ℕ), a > 0 → ∀ (d : ℕ), d ≤ 7 → ∃ (c : ℕ), c > 0

theorem min_cars_correct (adults : ℕ) (h : adults > 0) :
  ∀ (cars : ℕ), cars < min_cars adults →
    ∃ (d : ℕ), d ≤ 7 ∧ cars - (cars / 7) < adults :=
by sorry

#check min_cars_correct

end NUMINAMATH_CALUDE_min_cars_correct_l1159_115925


namespace NUMINAMATH_CALUDE_polyhedron_property_l1159_115969

/-- Represents a convex polyhedron with the given properties -/
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  t : ℕ  -- Number of triangular faces
  s : ℕ  -- Number of square faces
  euler_formula : V - E + F = 2
  face_count : F = 42
  face_types : F = t + s
  edge_relation : E = (3 * t + 4 * s) / 2
  vertex_degree : 13 * V = 2 * E

/-- The main theorem to be proved -/
theorem polyhedron_property (p : ConvexPolyhedron) : 100 * 3 + 10 * 2 + p.V = 337 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_property_l1159_115969


namespace NUMINAMATH_CALUDE_product_as_sum_of_squares_l1159_115918

theorem product_as_sum_of_squares : 85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 := by
  sorry

end NUMINAMATH_CALUDE_product_as_sum_of_squares_l1159_115918


namespace NUMINAMATH_CALUDE_handshake_count_l1159_115983

theorem handshake_count (n : ℕ) (total_handshakes : ℕ) : 
  n = 7 ∧ total_handshakes = n * (n - 1) / 2 → total_handshakes = 21 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l1159_115983


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l1159_115976

theorem arctan_equation_solution :
  ∀ x : ℝ, 2 * Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/x) = π/4 → x = -19/5 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l1159_115976


namespace NUMINAMATH_CALUDE_illuminated_area_of_cube_l1159_115958

/-- The area of the illuminated part of a cube's surface when illuminated by a cylindrical beam -/
theorem illuminated_area_of_cube (a ρ : ℝ) (h_a : a = Real.sqrt (2 + Real.sqrt 3)) (h_ρ : ρ = Real.sqrt 2) :
  let S := ρ^2 * Real.sqrt 3 * (Real.pi - 6 * Real.arccos (a / (ρ * Real.sqrt 2)) + 
           6 * (a / (ρ * Real.sqrt 2)) * Real.sqrt (1 - (a / (ρ * Real.sqrt 2))^2))
  S = Real.sqrt 3 * (Real.pi + 3) :=
sorry

end NUMINAMATH_CALUDE_illuminated_area_of_cube_l1159_115958


namespace NUMINAMATH_CALUDE_max_xy_value_l1159_115965

theorem max_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : Real.sqrt 3 = Real.sqrt (9^x * 3^y)) : 
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ Real.sqrt 3 = Real.sqrt (9^a * 3^b) → x * y ≥ a * b) ∧ 
  x * y = 1/8 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l1159_115965


namespace NUMINAMATH_CALUDE_solution_to_system_l1159_115955

theorem solution_to_system (x y : ℝ) :
  x^5 + y^5 = 33 ∧ x + y = 3 →
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
sorry

end NUMINAMATH_CALUDE_solution_to_system_l1159_115955


namespace NUMINAMATH_CALUDE_evaluate_expression_l1159_115993

theorem evaluate_expression (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (4 * y^2) * (1 / (2*y)^3) = 9 * y^2 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1159_115993


namespace NUMINAMATH_CALUDE_smallest_cookie_count_l1159_115968

theorem smallest_cookie_count : ∃ (x : ℕ), x > 0 ∧
  x % 6 = 5 ∧ x % 8 = 7 ∧ x % 9 = 2 ∧
  ∀ (y : ℕ), y > 0 → y % 6 = 5 → y % 8 = 7 → y % 9 = 2 → x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_count_l1159_115968


namespace NUMINAMATH_CALUDE_f_increasing_and_odd_l1159_115997

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_and_odd_l1159_115997


namespace NUMINAMATH_CALUDE_factorization_valid_l1159_115941

-- Define the left-hand side of the equation
def lhs (x : ℝ) : ℝ := -8 * x^2 + 8 * x - 2

-- Define the right-hand side of the equation
def rhs (x : ℝ) : ℝ := -2 * (2 * x - 1)^2

-- Theorem stating that the left-hand side equals the right-hand side for all real x
theorem factorization_valid (x : ℝ) : lhs x = rhs x := by
  sorry

end NUMINAMATH_CALUDE_factorization_valid_l1159_115941


namespace NUMINAMATH_CALUDE_angle_bisectors_rational_l1159_115981

/-- Given a triangle with sides a = 84, b = 125, and c = 169, 
    the lengths of all angle bisectors are rational numbers -/
theorem angle_bisectors_rational (a b c : ℚ) (h1 : a = 84) (h2 : b = 125) (h3 : c = 169) :
  ∃ (fa fb fc : ℚ), 
    (fa = 2 * b * c / (b + c) * (((b^2 + c^2 - a^2) / (2 * b * c) + 1) / 2).sqrt) ∧
    (fb = 2 * a * c / (a + c) * (((a^2 + c^2 - b^2) / (2 * a * c) + 1) / 2).sqrt) ∧
    (fc = 2 * a * b / (a + b) * (((a^2 + b^2 - c^2) / (2 * a * b) + 1) / 2).sqrt) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisectors_rational_l1159_115981


namespace NUMINAMATH_CALUDE_cost_price_equation_l1159_115910

/-- The cost price of a watch satisfying the given conditions -/
def cost_price : ℝ := 
  let C : ℝ := 2070.31
  C

/-- Theorem stating the equation that the cost price must satisfy -/
theorem cost_price_equation : 
  3 * (0.925 * cost_price + 265) = 3 * cost_price * 1.053 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_equation_l1159_115910


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l1159_115992

theorem incorrect_observation_value (n : ℕ) (original_mean corrected_mean correct_value : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 36)
  (h3 : corrected_mean = 36.5)
  (h4 : correct_value = 43) :
  ∃ x : ℝ, 
    (n : ℝ) * original_mean = (n : ℝ) * corrected_mean - correct_value + x :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l1159_115992


namespace NUMINAMATH_CALUDE_tom_video_game_spending_l1159_115944

/-- The total amount Tom spent on new video games --/
def total_spent (batman_price superman_price discount_rate tax_rate : ℚ) : ℚ :=
  let discounted_batman := batman_price * (1 - discount_rate)
  let discounted_superman := superman_price * (1 - discount_rate)
  let total_before_tax := discounted_batman + discounted_superman
  total_before_tax * (1 + tax_rate)

/-- Theorem stating the total amount Tom spent on new video games --/
theorem tom_video_game_spending :
  total_spent 13.60 5.06 0.20 0.08 = 16.12 := by
  sorry

#eval total_spent 13.60 5.06 0.20 0.08

end NUMINAMATH_CALUDE_tom_video_game_spending_l1159_115944


namespace NUMINAMATH_CALUDE_equation_solution_l1159_115962

theorem equation_solution : ∃! x : ℝ, (3 : ℝ) / (x - 3) = (4 : ℝ) / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1159_115962


namespace NUMINAMATH_CALUDE_only_D_opposite_sign_l1159_115919

-- Define the pairs of numbers
def pair_A : ℤ × ℤ := (-(-1), 1)
def pair_B : ℤ × ℤ := ((-1)^2, 1)
def pair_C : ℤ × ℤ := (|(-1)|, 1)
def pair_D : ℤ × ℤ := (-1, 1)

-- Define a function to check if two numbers are opposite in sign
def opposite_sign (a b : ℤ) : Prop := a * b < 0

-- Theorem stating that only pair D contains numbers with opposite signs
theorem only_D_opposite_sign :
  ¬(opposite_sign pair_A.1 pair_A.2) ∧
  ¬(opposite_sign pair_B.1 pair_B.2) ∧
  ¬(opposite_sign pair_C.1 pair_C.2) ∧
  (opposite_sign pair_D.1 pair_D.2) :=
sorry

end NUMINAMATH_CALUDE_only_D_opposite_sign_l1159_115919


namespace NUMINAMATH_CALUDE_prob_skew_lines_l1159_115930

/-- A cube with 8 vertices -/
structure Cube :=
  (vertices : Finset (Fin 8))

/-- A line determined by two vertices of the cube -/
structure Line (c : Cube) :=
  (v1 v2 : Fin 8)
  (h1 : v1 ∈ c.vertices)
  (h2 : v2 ∈ c.vertices)
  (h3 : v1 ≠ v2)

/-- Two lines are skew if they are non-coplanar and non-intersecting -/
def are_skew (c : Cube) (l1 l2 : Line c) : Prop :=
  sorry

/-- The set of all lines determined by any two vertices of the cube -/
def all_lines (c : Cube) : Finset (Line c) :=
  sorry

/-- The probability of an event occurring when choosing two lines from all_lines -/
def probability (c : Cube) (event : Line c → Line c → Prop) : ℚ :=
  sorry

theorem prob_skew_lines (c : Cube) :
  probability c (λ l1 l2 => are_skew c l1 l2) = 29 / 63 :=
sorry

end NUMINAMATH_CALUDE_prob_skew_lines_l1159_115930


namespace NUMINAMATH_CALUDE_max_at_two_implies_c_six_l1159_115913

/-- The function f(x) defined as x(x-c)^2 --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- Theorem stating that if f(x) has a maximum at x = 2, then c = 6 --/
theorem max_at_two_implies_c_six :
  ∀ c : ℝ, (∀ x : ℝ, f c x ≤ f c 2) → c = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_at_two_implies_c_six_l1159_115913


namespace NUMINAMATH_CALUDE_fifth_term_of_special_sequence_l1159_115911

/-- A sequence where each term after the first is 1/4 of the sum of the term before it and the term after it -/
def SpecialSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = (1 : ℚ) / 4 * (a n + a (n + 2))

theorem fifth_term_of_special_sequence
  (a : ℕ → ℚ)
  (h_seq : SpecialSequence a)
  (h_first : a 1 = 2)
  (h_fourth : a 4 = 50) :
  a 5 = 2798 / 15 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_special_sequence_l1159_115911


namespace NUMINAMATH_CALUDE_next_perfect_square_sum_of_digits_l1159_115960

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def begins_with_three_twos (n : ℕ) : Prop :=
  n ≥ 222000 ∧ n < 223000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem next_perfect_square_sum_of_digits :
  ∃ n : ℕ, is_perfect_square n ∧ 
           begins_with_three_twos n ∧
           (∀ m : ℕ, is_perfect_square m ∧ begins_with_three_twos m → n ≤ m) ∧
           sum_of_digits n = 18 :=
sorry

end NUMINAMATH_CALUDE_next_perfect_square_sum_of_digits_l1159_115960


namespace NUMINAMATH_CALUDE_log_sqrt8_512sqrt8_l1159_115978

theorem log_sqrt8_512sqrt8 : Real.log (512 * Real.sqrt 8) / Real.log (Real.sqrt 8) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt8_512sqrt8_l1159_115978


namespace NUMINAMATH_CALUDE_train_length_calculation_l1159_115902

/-- The length of a train given its speed, the speed of a trolley moving in the opposite direction, and the time it takes for the train to pass the trolley. -/
theorem train_length_calculation (train_speed : ℝ) (trolley_speed : ℝ) (passing_time : ℝ) : 
  train_speed = 60 →
  trolley_speed = 12 →
  passing_time = 5.4995600351971845 →
  ∃ (train_length : ℝ), abs (train_length - 109.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1159_115902


namespace NUMINAMATH_CALUDE_burritos_per_box_burritos_problem_l1159_115927

theorem burritos_per_box (total_boxes : ℕ) (fraction_given_away : ℚ) 
  (burritos_eaten_per_day : ℕ) (days_eaten : ℕ) (burritos_left : ℕ) : ℕ :=
let burritos_per_box := 
  (burritos_left + burritos_eaten_per_day * days_eaten) / 
  (total_boxes * (1 - fraction_given_away))
20

theorem burritos_problem : 
  burritos_per_box 3 (1/3) 3 10 10 = 20 := by
sorry

end NUMINAMATH_CALUDE_burritos_per_box_burritos_problem_l1159_115927


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1159_115924

-- Define set A
def A : Set ℝ := {x | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1159_115924


namespace NUMINAMATH_CALUDE_cubic_roots_product_l1159_115935

theorem cubic_roots_product (a b c : ℝ) : 
  (x^3 - 15*x^2 + 25*x - 12 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (2 + a) * (2 + b) * (2 + c) = 130 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_product_l1159_115935


namespace NUMINAMATH_CALUDE_equation_solution_l1159_115990

theorem equation_solution (x : ℝ) (h : x ≠ 2/3) :
  (6*x + 2) / (3*x^2 + 6*x - 4) = 3*x / (3*x - 2) ↔ x = 1 / Real.sqrt 3 ∨ x = -1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1159_115990


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l1159_115905

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  fociAxis : ℝ × ℝ
  eccentricity : ℝ
  passingPoint : ℝ × ℝ

/-- The equation of an ellipse given its properties -/
def ellipseEquation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 45 + y^2 / 36 = 1

/-- Theorem stating that an ellipse with the given properties has the specified equation -/
theorem ellipse_equation_proof (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.fociAxis.2 = 0)
  (h3 : e.eccentricity = Real.sqrt 5 / 5)
  (h4 : e.passingPoint = (-5, 4)) :
  ellipseEquation e = fun x y => x^2 / 45 + y^2 / 36 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_proof_l1159_115905


namespace NUMINAMATH_CALUDE_cos_five_pi_sixth_plus_alpha_l1159_115916

theorem cos_five_pi_sixth_plus_alpha (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) = -(Real.sqrt 3 / 3) := by
sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixth_plus_alpha_l1159_115916


namespace NUMINAMATH_CALUDE_max_take_home_pay_l1159_115934

/-- Represents the income in thousands of dollars -/
def income (x : ℝ) : ℝ := x + 10

/-- Represents the tax rate as a percentage -/
def taxRate (x : ℝ) : ℝ := x

/-- Calculates the take-home pay given the income parameter x -/
def takeHomePay (x : ℝ) : ℝ := 30250 - 10 * (x - 45)^2

/-- Theorem stating that the income yielding the maximum take-home pay is $55,000 -/
theorem max_take_home_pay :
  ∃ (x : ℝ), (∀ (y : ℝ), takeHomePay y ≤ takeHomePay x) ∧ income x = 55 := by
  sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l1159_115934


namespace NUMINAMATH_CALUDE_fraction_simplification_l1159_115912

theorem fraction_simplification (a : ℝ) (h : a ≠ 2) :
  (3 - a) / (a - 2) + 1 = 1 / (a - 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1159_115912


namespace NUMINAMATH_CALUDE_multiples_of_four_between_100_and_300_l1159_115954

theorem multiples_of_four_between_100_and_300 :
  (Finset.filter (fun n => n % 4 = 0) (Finset.range 300 \ Finset.range 101)).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_between_100_and_300_l1159_115954


namespace NUMINAMATH_CALUDE_flower_count_l1159_115991

/-- The number of pots -/
def num_pots : ℕ := 141

/-- The number of flowers in each pot -/
def flowers_per_pot : ℕ := 71

/-- The total number of flowers -/
def total_flowers : ℕ := num_pots * flowers_per_pot

theorem flower_count : total_flowers = 10011 := by
  sorry

end NUMINAMATH_CALUDE_flower_count_l1159_115991


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1159_115963

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
    b := p.b
    c := p.c + v }

/-- The theorem to be proved -/
theorem parabola_shift_theorem :
  let original := Parabola.mk (-2) 0 0
  let shifted_left := shift_horizontal original 3
  let final := shift_vertical shifted_left (-1)
  final = Parabola.mk (-2) 12 (-19) := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1159_115963


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_l1159_115961

theorem quadratic_roots_opposite (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (k^2 - 4)*x₁ + k - 1 = 0 ∧
    x₂^2 + (k^2 - 4)*x₂ + k - 1 = 0 ∧
    x₁ = -x₂) →
  k = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_opposite_l1159_115961


namespace NUMINAMATH_CALUDE_dougs_age_l1159_115943

theorem dougs_age (betty_age : ℕ) (doug_age : ℕ) (pack_cost : ℕ) :
  2 * betty_age = pack_cost →
  betty_age + doug_age = 90 →
  20 * pack_cost = 2000 →
  doug_age = 40 := by
sorry

end NUMINAMATH_CALUDE_dougs_age_l1159_115943


namespace NUMINAMATH_CALUDE_pieces_from_rod_l1159_115970

/-- The number of pieces of a given length that can be cut from a rod. -/
def number_of_pieces (rod_length_m : ℕ) (piece_length_cm : ℕ) : ℕ :=
  (rod_length_m * 100) / piece_length_cm

/-- Theorem: The number of 85 cm pieces that can be cut from a 34-meter rod is 40. -/
theorem pieces_from_rod : number_of_pieces 34 85 = 40 := by
  sorry

end NUMINAMATH_CALUDE_pieces_from_rod_l1159_115970


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_planes_perp_to_line_are_parallel_l1159_115987

/-- A 3D space -/
structure Space3D where
  -- Add necessary structure here

/-- A line in 3D space -/
structure Line3D (S : Space3D) where
  -- Add necessary structure here

/-- A plane in 3D space -/
structure Plane3D (S : Space3D) where
  -- Add necessary structure here

/-- Perpendicularity between a line and a plane -/
def perpendicular_line_plane (S : Space3D) (l : Line3D S) (p : Plane3D S) : Prop :=
  sorry

/-- Perpendicularity between a plane and a line -/
def perpendicular_plane_line (S : Space3D) (p : Plane3D S) (l : Line3D S) : Prop :=
  sorry

/-- Parallelism between two lines -/
def parallel_lines (S : Space3D) (l1 l2 : Line3D S) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel_planes (S : Space3D) (p1 p2 : Plane3D S) : Prop :=
  sorry

/-- Theorem: Two lines perpendicular to the same plane are parallel to each other -/
theorem lines_perp_to_plane_are_parallel (S : Space3D) (l1 l2 : Line3D S) (p : Plane3D S)
  (h1 : perpendicular_line_plane S l1 p) (h2 : perpendicular_line_plane S l2 p) :
  parallel_lines S l1 l2 :=
sorry

/-- Theorem: Two planes perpendicular to the same line are parallel to each other -/
theorem planes_perp_to_line_are_parallel (S : Space3D) (p1 p2 : Plane3D S) (l : Line3D S)
  (h1 : perpendicular_plane_line S p1 l) (h2 : perpendicular_plane_line S p2 l) :
  parallel_planes S p1 p2 :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_planes_perp_to_line_are_parallel_l1159_115987


namespace NUMINAMATH_CALUDE_midpoint_coord_sum_l1159_115908

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (10, 3) and (-4, -7) is 1 -/
theorem midpoint_coord_sum : 
  let x1 : ℝ := 10
  let y1 : ℝ := 3
  let x2 : ℝ := -4
  let y2 : ℝ := -7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 1 := by sorry

end NUMINAMATH_CALUDE_midpoint_coord_sum_l1159_115908


namespace NUMINAMATH_CALUDE_v_shaped_to_log_v_shaped_l1159_115939

/-- Definition of a V-shaped function -/
def is_v_shaped (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f (x₁ + x₂) ≤ f x₁ + f x₂

/-- Definition of a Logarithmic V-shaped function -/
def is_log_v_shaped (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f x > 0) ∧
  (∀ x₁ x₂ : ℝ, Real.log (f (x₁ + x₂)) < Real.log (f x₁) + Real.log (f x₂))

/-- Theorem: If f is V-shaped and f(x) ≥ 2 for all x, then f is Logarithmic V-shaped -/
theorem v_shaped_to_log_v_shaped (f : ℝ → ℝ) 
    (hv : is_v_shaped f) (hf : ∀ x : ℝ, f x ≥ 2) : 
    is_log_v_shaped f := by
  sorry

end NUMINAMATH_CALUDE_v_shaped_to_log_v_shaped_l1159_115939


namespace NUMINAMATH_CALUDE_min_value_polynomial_min_value_achieved_l1159_115915

theorem min_value_polynomial (x : ℝ) : (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 ≥ 2034 := by
  sorry

theorem min_value_achieved : ∃ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 = 2034 := by
  sorry

end NUMINAMATH_CALUDE_min_value_polynomial_min_value_achieved_l1159_115915


namespace NUMINAMATH_CALUDE_tomato_count_l1159_115933

theorem tomato_count (plant1 plant2 plant3 plant4 : ℕ) : 
  plant1 = 8 →
  plant2 = plant1 + 4 →
  plant3 = 3 * (plant1 + plant2) →
  plant4 = plant3 →
  plant1 + plant2 + plant3 + plant4 = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_count_l1159_115933


namespace NUMINAMATH_CALUDE_original_polygon_sides_l1159_115957

theorem original_polygon_sides (n : ℕ) : 
  (n + 1 - 2) * 180 = 1620 → n = 10 := by sorry

end NUMINAMATH_CALUDE_original_polygon_sides_l1159_115957


namespace NUMINAMATH_CALUDE_stratified_sampling_result_count_l1159_115975

def junior_population : ℕ := 400
def senior_population : ℕ := 200
def total_sample_size : ℕ := 60

def proportional_allocation (total_pop : ℕ) (stratum_pop : ℕ) (sample_size : ℕ) : ℕ :=
  (stratum_pop * sample_size) / total_pop

theorem stratified_sampling_result_count :
  Nat.choose junior_population (proportional_allocation (junior_population + senior_population) junior_population total_sample_size) *
  Nat.choose senior_population (proportional_allocation (junior_population + senior_population) senior_population total_sample_size) =
  Nat.choose junior_population 40 * Nat.choose senior_population 20 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_count_l1159_115975


namespace NUMINAMATH_CALUDE_min_towns_for_22_routes_l1159_115988

/-- A graph representing a country's airline network -/
structure AirlineNetwork where
  towns : Finset ℕ
  connections : towns → towns → Bool
  paid_direction : towns → towns → Bool

/-- The number of free routes between two towns in an airline network -/
def free_routes (g : AirlineNetwork) (a b : g.towns) : ℕ :=
  sorry

/-- The theorem stating that the minimum number of towns for 22 free routes is 7 -/
theorem min_towns_for_22_routes :
  ∃ (g : AirlineNetwork) (a b : g.towns),
    free_routes g a b = 22 ∧
    g.towns.card = 7 ∧
    (∀ (h : AirlineNetwork) (x y : h.towns),
      free_routes h x y = 22 → h.towns.card ≥ 7) :=
  sorry

end NUMINAMATH_CALUDE_min_towns_for_22_routes_l1159_115988


namespace NUMINAMATH_CALUDE_product_is_twice_square_l1159_115950

theorem product_is_twice_square (a b c d : ℕ+) (h : a * b = 2 * c * d) :
  ∃ (n : ℕ), a * b * c * d = 2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_product_is_twice_square_l1159_115950


namespace NUMINAMATH_CALUDE_x_value_proof_l1159_115952

theorem x_value_proof (x : ℝ) : 
  (⌊x⌋ + ⌈x⌉ = 7) ∧ (10 ≤ 3*x - 5 ∧ 3*x - 5 ≤ 13) → 3 < x ∧ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_x_value_proof_l1159_115952


namespace NUMINAMATH_CALUDE_circle_equation_through_points_l1159_115936

theorem circle_equation_through_points :
  let equation (x y : ℝ) := x^2 + y^2 - 4*x - 6*y
  ∀ (x y : ℝ), equation x y = 0 →
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_l1159_115936


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1159_115986

theorem quadratic_equation_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 - (k-3)*x₁ - k + 1 = 0) ∧ 
  (x₂^2 - (k-3)*x₂ - k + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1159_115986


namespace NUMINAMATH_CALUDE_log_sum_fifty_twenty_l1159_115922

theorem log_sum_fifty_twenty : Real.log 50 / Real.log 10 + Real.log 20 / Real.log 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_fifty_twenty_l1159_115922


namespace NUMINAMATH_CALUDE_milestone_number_l1159_115959

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  ones : ℕ
  h1 : tens ≥ 1 ∧ tens ≤ 9
  h2 : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to a natural number -/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : ℕ := 10 * n.tens + n.ones

/-- Theorem: Given the conditions of the problem, the initial number must be 16 -/
theorem milestone_number (initial : TwoDigitNumber) 
  (h1 : initial.toNat + initial.toNat = 100 * initial.ones + initial.tens + 100 * initial.tens + initial.ones) :
  initial.tens = 1 ∧ initial.ones = 6 := by
  sorry

#check milestone_number

end NUMINAMATH_CALUDE_milestone_number_l1159_115959


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1159_115938

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  m = (Real.sqrt 3, 1) →
  n = (Real.cos A + 1, Real.sin A) →
  m.1 * n.1 + m.2 * n.2 = 2 + Real.sqrt 3 →
  a = Real.sqrt 3 →
  Real.cos B = Real.sqrt 3 / 3 →
  A = π / 6 ∧ 
  (1 / 2 : ℝ) * a * b * Real.sin C = Real.sqrt 2 / 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1159_115938


namespace NUMINAMATH_CALUDE_square_area_proof_l1159_115904

theorem square_area_proof (side_length : ℝ) (rectangle_perimeter : ℝ) : 
  side_length = 8 →
  rectangle_perimeter = 20 →
  (side_length * side_length) = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l1159_115904


namespace NUMINAMATH_CALUDE_division_problem_l1159_115979

theorem division_problem (divisor : ℕ) : 
  (15 / divisor = 4) ∧ (15 % divisor = 3) → divisor = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1159_115979


namespace NUMINAMATH_CALUDE_original_number_exists_and_unique_l1159_115900

theorem original_number_exists_and_unique : 
  ∃! x : ℝ, 3 * (2 * x + 9) = 63 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_and_unique_l1159_115900


namespace NUMINAMATH_CALUDE_remainder_of_S_mod_512_l1159_115901

def R : Finset ℕ := Finset.image (λ n => (3^n) % 512) (Finset.range 12)

def S : ℕ := Finset.sum R id

theorem remainder_of_S_mod_512 : S % 512 = 72 := by sorry

end NUMINAMATH_CALUDE_remainder_of_S_mod_512_l1159_115901


namespace NUMINAMATH_CALUDE_field_trip_cost_l1159_115907

def total_cost (students : ℕ) (teachers : ℕ) (bus_capacity : ℕ) (rental_cost : ℕ) (toll_cost : ℕ) : ℕ :=
  let total_people := students + teachers
  let buses_needed := (total_people + bus_capacity - 1) / bus_capacity
  buses_needed * (rental_cost + toll_cost)

theorem field_trip_cost :
  total_cost 252 8 41 300000 7500 = 2152500 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_cost_l1159_115907


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1159_115951

def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x > 1}

theorem union_of_A_and_B : A ∪ B = {x | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1159_115951


namespace NUMINAMATH_CALUDE_renovation_project_dirt_calculation_l1159_115984

theorem renovation_project_dirt_calculation (total material sand cement : ℚ)
  (h1 : sand = 0.17)
  (h2 : cement = 0.17)
  (h3 : total = 0.67)
  (h4 : material = total - (sand + cement)) :
  material = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_renovation_project_dirt_calculation_l1159_115984


namespace NUMINAMATH_CALUDE_sum_odd_integers_11_to_39_l1159_115949

/-- The sum of odd integers from 11 to 39 (inclusive) is 375 -/
theorem sum_odd_integers_11_to_39 : 
  (Finset.range 15).sum (fun i => 2 * i + 11) = 375 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_11_to_39_l1159_115949


namespace NUMINAMATH_CALUDE_coordinate_sum_of_h_l1159_115926

theorem coordinate_sum_of_h (g : ℝ → ℝ) (h : ℝ → ℝ) : 
  g 4 = 8 → 
  (∀ x, h x = (g x)^2) → 
  4 + h 4 = 68 := by
sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_h_l1159_115926


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l1159_115995

/-- Given a hyperbola with equation x²/m - y²/(m+6) = 1 where m > 0,
    and its conjugate axis is twice the length of its transverse axis,
    prove that the standard form of the hyperbola's equation is x²/2 - y²/8 = 1 -/
theorem hyperbola_standard_form (m : ℝ) (h_m : m > 0) 
  (h_eq : ∀ x y : ℝ, x^2 / m - y^2 / (m + 6) = 1)
  (h_axis : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = m ∧ b^2 = m + 6 ∧ b = 2*a) :
  ∀ x y : ℝ, x^2 / 2 - y^2 / 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l1159_115995


namespace NUMINAMATH_CALUDE_franks_age_l1159_115971

theorem franks_age (frank gabriel lucy : ℕ) : 
  gabriel = frank - 3 →
  frank + gabriel = 17 →
  lucy = gabriel + 5 →
  lucy = gabriel + frank →
  frank = 10 := by
sorry

end NUMINAMATH_CALUDE_franks_age_l1159_115971


namespace NUMINAMATH_CALUDE_julie_initial_savings_l1159_115917

/-- The amount of money Julie saved initially before doing jobs to buy a mountain bike. -/
def initial_savings : ℕ := sorry

/-- The cost of the mountain bike Julie wants to buy. -/
def bike_cost : ℕ := 2345

/-- The number of lawns Julie plans to mow. -/
def lawns_to_mow : ℕ := 20

/-- The payment Julie receives for mowing each lawn. -/
def payment_per_lawn : ℕ := 20

/-- The number of newspapers Julie plans to deliver. -/
def newspapers_to_deliver : ℕ := 600

/-- The payment Julie receives for delivering each newspaper (in cents). -/
def payment_per_newspaper : ℕ := 40

/-- The number of dogs Julie plans to walk. -/
def dogs_to_walk : ℕ := 24

/-- The payment Julie receives for walking each dog. -/
def payment_per_dog : ℕ := 15

/-- The amount of money Julie has left after purchasing the bike. -/
def money_left : ℕ := 155

/-- Theorem stating that Julie's initial savings were $1190. -/
theorem julie_initial_savings :
  initial_savings = 1190 :=
by sorry

end NUMINAMATH_CALUDE_julie_initial_savings_l1159_115917


namespace NUMINAMATH_CALUDE_symmetric_matrix_square_sum_l1159_115977

theorem symmetric_matrix_square_sum (x y z : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; y, z]
  (∀ i j, B i j = B j i) →  -- B is symmetric
  B * B = (1 : Matrix (Fin 2) (Fin 2) ℝ) →  -- B^2 = I
  x^2 + 2*y^2 + z^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_matrix_square_sum_l1159_115977


namespace NUMINAMATH_CALUDE_icecream_cost_theorem_l1159_115989

def chapati_count : ℕ := 16
def rice_count : ℕ := 5
def vegetable_count : ℕ := 7
def icecream_count : ℕ := 6

def chapati_cost : ℕ := 6
def rice_cost : ℕ := 45
def vegetable_cost : ℕ := 70

def total_paid : ℕ := 1015

theorem icecream_cost_theorem : 
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / icecream_count = 34 := by
  sorry

end NUMINAMATH_CALUDE_icecream_cost_theorem_l1159_115989


namespace NUMINAMATH_CALUDE_jellybean_problem_l1159_115998

/-- Calculates the number of jellybeans removed after adding some back -/
def jellybeans_removed_after_adding_back (initial : ℕ) (removed : ℕ) (added_back : ℕ) (final : ℕ) : ℕ :=
  initial - removed + added_back - final

theorem jellybean_problem (initial : ℕ) (removed : ℕ) (added_back : ℕ) (final : ℕ)
  (h1 : initial = 37)
  (h2 : removed = 15)
  (h3 : added_back = 5)
  (h4 : final = 23) :
  jellybeans_removed_after_adding_back initial removed added_back final = 4 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l1159_115998


namespace NUMINAMATH_CALUDE_puppy_weight_l1159_115923

/-- Given the weights of a puppy, a smaller cat, and a larger cat, prove that the puppy weighs 5 pounds. -/
theorem puppy_weight (p s l : ℝ) 
  (total_weight : p + s + l = 30)
  (puppy_larger_cat : p + l = 3 * s)
  (puppy_smaller_cat : p + s = l - 5) :
  p = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l1159_115923


namespace NUMINAMATH_CALUDE_divisible_by_72_implies_a3_b2_l1159_115931

/-- Represents a five-digit number in the form a679b -/
def five_digit_number (a b : ℕ) : ℕ := a * 10000 + 6790 + b

/-- Checks if a natural number is divisible by another natural number -/
def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem divisible_by_72_implies_a3_b2 :
  ∀ a b : ℕ, 
    a < 10 → b < 10 →
    is_divisible_by (five_digit_number a b) 72 →
    a = 3 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_72_implies_a3_b2_l1159_115931


namespace NUMINAMATH_CALUDE_cos_50_tan_40_equals_sqrt_3_l1159_115985

theorem cos_50_tan_40_equals_sqrt_3 : 
  4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_50_tan_40_equals_sqrt_3_l1159_115985


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1159_115920

theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![3, -1]
  let b : Fin 2 → ℝ := ![1, x]
  (∀ i, i < 2 → a i * b i = 0) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1159_115920


namespace NUMINAMATH_CALUDE_num_solutions_for_quadratic_l1159_115974

/-- A line y = x + a that does not pass through the second quadrant -/
structure Line where
  a : ℝ
  not_in_second_quadrant : a ≤ 0

/-- The number of real solutions for a quadratic equation -/
inductive NumRealSolutions
  | zero
  | one
  | two

/-- The theorem stating the number of real solutions for the equation ax^2 + 2x + 1 = 0 -/
theorem num_solutions_for_quadratic (l : Line) :
  ∃ n : NumRealSolutions, (n = NumRealSolutions.one ∨ n = NumRealSolutions.two) ∧
  (∃ x : ℝ, l.a * x^2 + 2*x + 1 = 0) ∧
  (n = NumRealSolutions.two → ∃ x y : ℝ, x ≠ y ∧ l.a * x^2 + 2*x + 1 = 0 ∧ l.a * y^2 + 2*y + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_num_solutions_for_quadratic_l1159_115974


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l1159_115999

theorem largest_multiple_of_15_under_400 : 
  ∀ n : ℕ, n * 15 < 400 → n * 15 ≤ 390 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l1159_115999


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l1159_115956

/-- The area of a square with adjacent vertices at (1,3) and (4,6) is 18 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (4, 6)
  let distance_squared := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
  distance_squared = 18 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l1159_115956


namespace NUMINAMATH_CALUDE_karens_order_cost_l1159_115966

/-- The cost of Karen's fast-food order -/
def fast_food_order_cost (burger_cost sandwich_cost smoothie_cost : ℕ) 
  (burger_quantity sandwich_quantity smoothie_quantity : ℕ) : ℕ :=
  burger_cost * burger_quantity + sandwich_cost * sandwich_quantity + smoothie_cost * smoothie_quantity

/-- Theorem stating that Karen's fast-food order costs $17 -/
theorem karens_order_cost : 
  fast_food_order_cost 5 4 4 1 1 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_karens_order_cost_l1159_115966


namespace NUMINAMATH_CALUDE_trihedral_angle_sum_l1159_115972

/-- Represents a trihedral angle with plane angles α, β, and γ. -/
structure TrihedralAngle where
  α : ℝ
  β : ℝ
  γ : ℝ
  α_pos : 0 < α
  β_pos : 0 < β
  γ_pos : 0 < γ

/-- The sum of any two plane angles of a trihedral angle is greater than the third. -/
theorem trihedral_angle_sum (t : TrihedralAngle) : t.α + t.β > t.γ ∧ t.β + t.γ > t.α ∧ t.α + t.γ > t.β := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_sum_l1159_115972


namespace NUMINAMATH_CALUDE_ship_meetings_count_l1159_115929

/-- The number of ships sailing in each direction -/
def num_ships : ℕ := 5

/-- The total number of meetings between two groups of ships -/
def total_meetings (n : ℕ) : ℕ := n * n

/-- Theorem stating that the total number of meetings is 25 -/
theorem ship_meetings_count : total_meetings num_ships = 25 := by
  sorry

end NUMINAMATH_CALUDE_ship_meetings_count_l1159_115929


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1159_115996

/-- Simple interest calculation --/
theorem simple_interest_rate_calculation 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 5000)
  (h2 : interest = 2500)
  (h3 : time = 5)
  : (interest * 100) / (principal * time) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1159_115996


namespace NUMINAMATH_CALUDE_stratified_sampling_properties_l1159_115914

structure School where
  first_year_students : ℕ
  second_year_students : ℕ

def stratified_sample (school : School) (sample_size : ℕ) : 
  ℕ × ℕ :=
  let total_students := school.first_year_students + school.second_year_students
  let first_year_sample := (school.first_year_students * sample_size) / total_students
  let second_year_sample := sample_size - first_year_sample
  (first_year_sample, second_year_sample)

theorem stratified_sampling_properties 
  (school : School)
  (sample_size : ℕ)
  (h1 : school.first_year_students = 1000)
  (h2 : school.second_year_students = 1080)
  (h3 : sample_size = 208) :
  let (first_sample, second_sample) := stratified_sample school sample_size
  -- 1. Students from different grades can be selected simultaneously
  (first_sample > 0 ∧ second_sample > 0) ∧
  -- 2. The number of students selected from each grade is proportional to the grade's population
  (first_sample = 100 ∧ second_sample = 108) ∧
  -- 3. The probability of selection for any student is equal across both grades
  (first_sample / school.first_year_students = second_sample / school.second_year_students) :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_properties_l1159_115914


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1159_115932

theorem rectangle_perimeter (area : ℝ) (side_difference : ℝ) (perimeter : ℝ) : 
  area = 500 →
  side_difference = 5 →
  (∃ x : ℝ, x > 0 ∧ x * (x + side_difference) = area) →
  perimeter = 90 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1159_115932


namespace NUMINAMATH_CALUDE_hannahs_peppers_l1159_115973

theorem hannahs_peppers (green_peppers red_peppers : ℝ) 
  (h1 : green_peppers = 0.3333333333333333)
  (h2 : red_peppers = 0.3333333333333333) :
  green_peppers + red_peppers = 0.6666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_peppers_l1159_115973


namespace NUMINAMATH_CALUDE_cab_driver_income_l1159_115921

theorem cab_driver_income (income1 income2 income3 income4 : ℕ) (average : ℚ) :
  income1 = 45 →
  income2 = 50 →
  income3 = 60 →
  income4 = 65 →
  average = 58 →
  ∃ income5 : ℕ, 
    (income1 + income2 + income3 + income4 + income5 : ℚ) / 5 = average ∧
    income5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l1159_115921


namespace NUMINAMATH_CALUDE_complex_number_existence_l1159_115940

theorem complex_number_existence : ∃ z : ℂ, 
  (z + 10 / z).im = 0 ∧ (z + 4).re = (z + 4).im :=
sorry

end NUMINAMATH_CALUDE_complex_number_existence_l1159_115940


namespace NUMINAMATH_CALUDE_circle_symmetry_l1159_115980

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), original_circle x y ∧ symmetry_line x y → symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1159_115980


namespace NUMINAMATH_CALUDE_prob_two_cards_sum_17_l1159_115928

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of each specific card (8 or 9) in the deck
def cards_of_each_value : ℕ := 4

-- Define the probability of choosing two specific cards
def prob_two_specific_cards : ℚ := (cards_of_each_value : ℚ) / total_cards * (cards_of_each_value : ℚ) / (total_cards - 1)

-- Define the number of ways to choose two cards that sum to 17 (8+9 or 9+8)
def ways_to_sum_17 : ℕ := 2

theorem prob_two_cards_sum_17 : 
  prob_two_specific_cards * ways_to_sum_17 = 8 / 663 := by sorry

end NUMINAMATH_CALUDE_prob_two_cards_sum_17_l1159_115928


namespace NUMINAMATH_CALUDE_diana_etienne_money_comparison_l1159_115953

/-- Proves that Diana's money is 21.25% greater than Etienne's after euro appreciation --/
theorem diana_etienne_money_comparison :
  let initial_rate : ℝ := 1.25  -- 1 euro = 1.25 dollars
  let diana_dollars : ℝ := 600
  let etienne_euros : ℝ := 350
  let appreciation_rate : ℝ := 1.08  -- 8% appreciation
  let new_rate : ℝ := initial_rate * appreciation_rate
  let etienne_dollars : ℝ := etienne_euros * new_rate
  let difference_percent : ℝ := (diana_dollars - etienne_dollars) / etienne_dollars * 100
  difference_percent = 21.25 := by
sorry

end NUMINAMATH_CALUDE_diana_etienne_money_comparison_l1159_115953


namespace NUMINAMATH_CALUDE_square_park_fencing_cost_l1159_115909

/-- Given a square park with a total fencing cost, calculate the cost per side -/
theorem square_park_fencing_cost (total_cost : ℝ) (h_total_cost : total_cost = 172) :
  total_cost / 4 = 43 := by
  sorry

end NUMINAMATH_CALUDE_square_park_fencing_cost_l1159_115909


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1159_115948

theorem complex_equation_solution (m : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (↑m + 2 * Complex.I) * (2 - Complex.I) = 4 + 3 * Complex.I →
  m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1159_115948


namespace NUMINAMATH_CALUDE_cube_volume_problem_l1159_115937

theorem cube_volume_problem (V₁ : ℝ) (V₂ : ℝ) (A₁ : ℝ) (A₂ : ℝ) (s₁ : ℝ) (s₂ : ℝ) :
  V₁ = 8 →
  s₁ ^ 3 = V₁ →
  A₁ = 6 * s₁ ^ 2 →
  A₂ = 3 * A₁ →
  A₂ = 6 * s₂ ^ 2 →
  V₂ = s₂ ^ 3 →
  V₂ = 24 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l1159_115937


namespace NUMINAMATH_CALUDE_remainder_2519_div_4_l1159_115967

theorem remainder_2519_div_4 : 2519 % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_4_l1159_115967


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_8154_l1159_115906

/-- Calculates the cost of white washing a room with given dimensions and openings -/
def whitewashingCost (length width height : ℝ) (doorLength doorWidth : ℝ)
  (windowLength windowWidth : ℝ) (windowCount : ℕ) (costPerSquareFoot : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let doorArea := doorLength * doorWidth
  let windowArea := windowLength * windowWidth * windowCount
  let netArea := wallArea - doorArea - windowArea
  netArea * costPerSquareFoot

/-- Theorem stating the cost of white washing the room with given specifications -/
theorem whitewashing_cost_is_8154 :
  whitewashingCost 25 15 12 6 3 4 3 3 9 = 8154 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_8154_l1159_115906


namespace NUMINAMATH_CALUDE_consecutive_numbers_percentage_l1159_115964

theorem consecutive_numbers_percentage (a b c d e f g : ℤ) : 
  (a + b + c + d + e + f + g) / 7 = 9 ∧ 
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧ f = e + 1 ∧ g = f + 1 →
  a * 100 / g = 50 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_percentage_l1159_115964


namespace NUMINAMATH_CALUDE_outfits_count_l1159_115947

/-- The number of different outfits that can be made from a given number of shirts, ties, and belts. -/
def number_of_outfits (shirts ties belts : ℕ) : ℕ := shirts * ties * belts

/-- Theorem stating that the number of outfits from 8 shirts, 7 ties, and 4 belts is 224. -/
theorem outfits_count : number_of_outfits 8 7 4 = 224 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1159_115947


namespace NUMINAMATH_CALUDE_arrangement_count_l1159_115946

/-- Represents the number of different seed types -/
def num_seed_types : ℕ := 5

/-- Represents the number of experimental fields -/
def num_fields : ℕ := 5

/-- Represents the number of seed types that can be placed at the ends -/
def num_end_seeds : ℕ := 3

/-- Represents the number of positions for the A-B pair -/
def num_ab_positions : ℕ := 3

/-- Calculates the number of ways to arrange seeds under the given conditions -/
def calculate_arrangements : ℕ :=
  (num_end_seeds * (num_end_seeds - 1)) * -- Arrangements for the ends
  (num_ab_positions * 2)                  -- Arrangements for A-B pair and remaining seed

/-- Theorem stating that the number of arrangement methods is 24 -/
theorem arrangement_count : calculate_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1159_115946


namespace NUMINAMATH_CALUDE_conference_season_games_l1159_115942

/-- Calculates the number of games in a complete season for a sports conference. -/
def games_in_season (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let divisions := total_teams / teams_per_division
  let intra_division_total := divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams / 2) * teams_per_division * inter_division_games
  intra_division_total + inter_division_total

/-- Theorem stating the number of games in a complete season for the given conference structure. -/
theorem conference_season_games : 
  games_in_season 14 7 3 1 = 175 := by
  sorry

end NUMINAMATH_CALUDE_conference_season_games_l1159_115942


namespace NUMINAMATH_CALUDE_race_speed_ratio_l1159_115982

theorem race_speed_ratio (course_length : ℝ) (head_start : ℝ) 
  (h1 : course_length = 84)
  (h2 : head_start = 63)
  (h3 : course_length > head_start)
  (h4 : head_start > 0) :
  ∃ (speed_a speed_b : ℝ),
    speed_a > 0 ∧ speed_b > 0 ∧
    (course_length / speed_a = (course_length - head_start) / speed_b) ∧
    speed_a = 4 * speed_b :=
by sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l1159_115982


namespace NUMINAMATH_CALUDE_banana_problem_l1159_115945

/-- Represents the number of bananas eaten on a given day -/
def bananas_eaten (day : ℕ) (first_day : ℕ) : ℕ :=
  first_day + 6 * (day - 1)

/-- The total number of bananas eaten over 5 days -/
def total_bananas (first_day : ℕ) : ℕ :=
  (bananas_eaten 1 first_day) + (bananas_eaten 2 first_day) + 
  (bananas_eaten 3 first_day) + (bananas_eaten 4 first_day) + 
  (bananas_eaten 5 first_day)

theorem banana_problem : 
  ∃ (first_day : ℕ), total_bananas first_day = 100 ∧ first_day = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_problem_l1159_115945
