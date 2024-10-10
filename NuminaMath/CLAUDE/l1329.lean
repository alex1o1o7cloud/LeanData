import Mathlib

namespace square_roots_theorem_l1329_132928

theorem square_roots_theorem (x a : ℝ) (hx : x > 0) :
  (∃ (r₁ r₂ : ℝ), r₁ = 2*a - 1 ∧ r₂ = -a + 2 ∧ r₁^2 = x ∧ r₂^2 = x) →
  a = -1 ∧ x = 9 :=
by sorry

end square_roots_theorem_l1329_132928


namespace mile_to_rod_l1329_132906

-- Define the conversion factors
def mile_to_furlong : ℝ := 8
def furlong_to_pace : ℝ := 220
def pace_to_rod : ℝ := 0.2

-- Theorem statement
theorem mile_to_rod : 
  1 * mile_to_furlong * furlong_to_pace * pace_to_rod = 352 := by
  sorry

end mile_to_rod_l1329_132906


namespace fifth_student_stickers_l1329_132902

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fifth_student_stickers :
  let a₁ := 29  -- first term
  let d := 6    -- common difference
  let n := 5    -- position of the term we're looking for
  arithmetic_sequence a₁ d n = 53 := by sorry

end fifth_student_stickers_l1329_132902


namespace triangle_altitude_l1329_132985

theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) : 
  area = 500 → base = 50 → area = (1/2) * base * altitude → altitude = 20 := by
  sorry

end triangle_altitude_l1329_132985


namespace newer_car_distance_l1329_132923

theorem newer_car_distance (older_distance : ℝ) (percentage_increase : ℝ) 
  (h1 : older_distance = 150)
  (h2 : percentage_increase = 0.30) : 
  older_distance * (1 + percentage_increase) = 195 :=
by sorry

end newer_car_distance_l1329_132923


namespace apps_deleted_l1329_132955

/-- Given that Dave initially had 150 apps on his phone and 65 apps remained after deletion,
    prove that the number of apps deleted is 85. -/
theorem apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) (h1 : initial_apps = 150) (h2 : remaining_apps = 65) :
  initial_apps - remaining_apps = 85 := by
  sorry

end apps_deleted_l1329_132955


namespace log_equation_l1329_132972

theorem log_equation : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end log_equation_l1329_132972


namespace constant_term_expansion_l1329_132998

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (x + 2 / Real.sqrt x) ^ 6
  ∃ (coefficient : ℝ), coefficient = 240 ∧ 
    (∃ (other_terms : ℝ → ℝ), expansion = coefficient + other_terms x ∧ 
      (∀ y : ℝ, other_terms y ≠ 0 → y ≠ 0)) :=
by sorry

end constant_term_expansion_l1329_132998


namespace certain_number_problem_l1329_132950

theorem certain_number_problem (x : ℝ) : 
  (10 + x + 60) / 3 = (10 + 40 + 25) / 3 + 5 → x = 20 := by
  sorry

end certain_number_problem_l1329_132950


namespace three_intersection_points_l1329_132968

-- Define the three lines
def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 8 * x - 12 * y = 9

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
by
  sorry

end three_intersection_points_l1329_132968


namespace rachel_distance_to_nicholas_l1329_132933

/-- The distance between two points given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Rachel's distance to Nicholas's house -/
theorem rachel_distance_to_nicholas : distance 2 5 = 10 := by
  sorry

end rachel_distance_to_nicholas_l1329_132933


namespace symmetric_point_theorem_l1329_132915

/-- Given a point P in the Cartesian coordinate system, 
    find its symmetric point with respect to the x-axis. -/
def symmetric_point_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

/-- Theorem: The coordinates of the point symmetric to P (-1, 2) 
    with respect to the x-axis are (-1, -2). -/
theorem symmetric_point_theorem : 
  symmetric_point_x_axis (-1, 2) = (-1, -2) := by
  sorry

end symmetric_point_theorem_l1329_132915


namespace cds_per_rack_is_eight_l1329_132912

/-- The number of racks a shelf can hold -/
def num_racks : ℕ := 4

/-- The total number of CDs a shelf can hold -/
def total_cds : ℕ := 32

/-- The number of CDs each rack can hold -/
def cds_per_rack : ℕ := total_cds / num_racks

theorem cds_per_rack_is_eight : cds_per_rack = 8 := by
  sorry

end cds_per_rack_is_eight_l1329_132912


namespace min_monochromatic_triangles_K15_l1329_132958

/-- A coloring of the edges of a complete graph using two colors. -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- The number of monochromatic triangles in a two-colored complete graph. -/
def monochromaticTriangles (n : ℕ) (c : TwoColoring n) : ℕ := sorry

/-- Theorem: The minimum number of monochromatic triangles in K₁₅ is 88. -/
theorem min_monochromatic_triangles_K15 :
  (∃ c : TwoColoring 15, monochromaticTriangles 15 c = 88) ∧
  (∀ c : TwoColoring 15, monochromaticTriangles 15 c ≥ 88) := by
  sorry

end min_monochromatic_triangles_K15_l1329_132958


namespace sequence_divisibility_implies_zero_l1329_132919

theorem sequence_divisibility_implies_zero (x : ℕ → ℤ) :
  (∀ i j : ℕ, i ≠ j → (i * j : ℤ) ∣ (x i + x j)) →
  ∀ n : ℕ, x n = 0 :=
by sorry

end sequence_divisibility_implies_zero_l1329_132919


namespace dodecagon_diagonals_l1329_132911

/-- The number of diagonals in a convex n-gon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A convex dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by sorry

end dodecagon_diagonals_l1329_132911


namespace iAmALiar_false_for_knights_and_knaves_iAmALiar_identifies_spy_l1329_132932

-- Define the types of characters
inductive Character : Type
| Knight : Character
| Knave : Character
| Spy : Character

-- Define the property of telling the truth
def tellsTruth (c : Character) : Prop :=
  match c with
  | Character.Knight => true
  | Character.Knave => false
  | Character.Spy => false

-- Define the statement "I am a liar"
def iAmALiar (c : Character) : Prop :=
  ¬(tellsTruth c)

-- Theorem: The statement "I am a liar" is false for both knights and knaves
theorem iAmALiar_false_for_knights_and_knaves :
  ∀ c : Character, c ≠ Character.Spy → ¬(iAmALiar c = tellsTruth c) :=
by sorry

-- Theorem: The statement "I am a liar" immediately identifies the speaker as a spy
theorem iAmALiar_identifies_spy :
  ∀ c : Character, iAmALiar c = tellsTruth c → c = Character.Spy :=
by sorry

end iAmALiar_false_for_knights_and_knaves_iAmALiar_identifies_spy_l1329_132932


namespace continuous_multiplicative_function_is_exponential_l1329_132952

open Real

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def ContinuousMultiplicativeFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ f 0 = 1 ∧ ∀ x y, f (x + y) ≥ f x * f y

/-- The main theorem statement -/
theorem continuous_multiplicative_function_is_exponential
  (f : ℝ → ℝ) (hf : ContinuousMultiplicativeFunction f) :
  ∃ a : ℝ, a > 0 ∧ ∀ x, f x = a^x :=
sorry

end continuous_multiplicative_function_is_exponential_l1329_132952


namespace x_value_theorem_l1329_132980

theorem x_value_theorem (x : ℝ) : x * (x * (x + 1) + 2) + 3 = x^3 + x^2 + x - 6 → x = -9 := by
  sorry

end x_value_theorem_l1329_132980


namespace equilateral_triangle_side_length_l1329_132922

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define the perimeter of an equilateral triangle
def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.side_length

-- Theorem statement
theorem equilateral_triangle_side_length 
  (t : EquilateralTriangle) 
  (h : perimeter t = 18) : 
  t.side_length = 6 := by
  sorry

end equilateral_triangle_side_length_l1329_132922


namespace juniper_bones_l1329_132974

theorem juniper_bones (initial_bones doubled_bones stolen_bones : ℕ) 
  (h1 : initial_bones = 4)
  (h2 : doubled_bones = initial_bones * 2)
  (h3 : stolen_bones = 2) :
  doubled_bones - stolen_bones = 6 :=
by
  sorry

end juniper_bones_l1329_132974


namespace robin_gum_count_l1329_132936

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 9

/-- The number of gum pieces in each package -/
def pieces_per_package : ℕ := 15

/-- The total number of gum pieces Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 135 := by
  sorry

end robin_gum_count_l1329_132936


namespace total_cost_of_toys_l1329_132945

-- Define the costs of the toys
def marbles_cost : ℚ := 9.05
def football_cost : ℚ := 4.95
def baseball_cost : ℚ := 6.52

-- Theorem stating the total cost
theorem total_cost_of_toys :
  marbles_cost + football_cost + baseball_cost = 20.52 := by
  sorry

end total_cost_of_toys_l1329_132945


namespace total_oranges_count_l1329_132979

def initial_oranges : ℕ := 5
def received_oranges : ℕ := 3
def bought_skittles : ℕ := 9

theorem total_oranges_count : 
  initial_oranges + received_oranges = 8 :=
by sorry

end total_oranges_count_l1329_132979


namespace exists_concave_to_convex_map_not_exists_convex_to_concave_map_l1329_132905

-- Define the plane
def Plane := ℝ × ℝ

-- Define a polygon as a list of points in the plane
def Polygon := List Plane

-- Define a simple polygon
def SimplePolygon (p : Polygon) : Prop := sorry

-- Define a convex polygon
def ConvexPolygon (p : Polygon) : Prop := sorry

-- Define a concave polygon
def ConcavePolygon (p : Polygon) : Prop := ¬ConvexPolygon p

-- State the existence of the function for part (a)
theorem exists_concave_to_convex_map :
  ∃ (f : Plane → Plane), ∀ (n : ℕ) (p : Polygon),
    n ≥ 4 →
    SimplePolygon p →
    ConcavePolygon p →
    p.length = n →
    ∃ (q : Polygon), SimplePolygon q ∧ ConvexPolygon q ∧ q = p.map f :=
sorry

-- State the non-existence of the function for part (b)
theorem not_exists_convex_to_concave_map :
  ¬∃ (f : Plane → Plane), ∀ (n : ℕ) (p : Polygon),
    n ≥ 4 →
    SimplePolygon p →
    ConvexPolygon p →
    p.length = n →
    ∃ (q : Polygon), SimplePolygon q ∧ ConcavePolygon q ∧ q = p.map f :=
sorry

end exists_concave_to_convex_map_not_exists_convex_to_concave_map_l1329_132905


namespace complex_magnitude_problem_l1329_132935

theorem complex_magnitude_problem (x y : ℝ) (h : (2 + Complex.I) * y = x + y * Complex.I) (hy : y ≠ 0) :
  Complex.abs ((x / y) + Complex.I) = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l1329_132935


namespace definite_integral_3x_minus_sinx_l1329_132961

theorem definite_integral_3x_minus_sinx : 
  ∫ x in (0)..(π/2), (3*x - Real.sin x) = 3*π^2/8 - 1 := by sorry

end definite_integral_3x_minus_sinx_l1329_132961


namespace pizza_size_relation_l1329_132918

theorem pizza_size_relation (r : ℝ) (h : r > 0) :
  let R := r * Real.sqrt (1 + 156 / 100)
  (R - r) / r * 100 = 60 := by sorry

end pizza_size_relation_l1329_132918


namespace symmetric_points_difference_l1329_132913

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

/-- Given two points M(a, 3) and N(5, b) that are symmetric with respect to the y-axis,
    prove that a - b = -8 -/
theorem symmetric_points_difference (a b : ℝ) 
    (h : symmetric_wrt_y_axis (a, 3) (5, b)) : a - b = -8 := by
  sorry

end symmetric_points_difference_l1329_132913


namespace greatest_k_value_l1329_132909

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 145) →
  k ≤ Real.sqrt 177 :=
sorry

end greatest_k_value_l1329_132909


namespace greatest_n_is_5_l1329_132910

/-- A coloring of a board is a function that assigns a color to each square. -/
def Coloring (m n : ℕ) := Fin m → Fin n → Fin 3

/-- A board has a valid coloring if no rectangle has all four corners of the same color. -/
def ValidColoring (m n : ℕ) (c : Coloring m n) : Prop :=
  ∀ (r1 r2 : Fin m) (c1 c2 : Fin n),
    r1 ≠ r2 → c1 ≠ c2 →
    (c r1 c1 = c r1 c2 ∧ c r1 c1 = c r2 c1 ∧ c r1 c1 = c r2 c2) → False

/-- The main theorem: The greatest possible value of n is 5. -/
theorem greatest_n_is_5 :
  (∃ (c : Coloring 6 4), ValidColoring 6 4 c) ∧
  (∀ n > 5, ¬∃ (c : Coloring (n+1) (n-1)), ValidColoring (n+1) (n-1) c) :=
sorry

end greatest_n_is_5_l1329_132910


namespace lucky_lucy_theorem_l1329_132970

/-- The expression with parentheses -/
def expr_with_parentheses (a b c d e : ℤ) : ℤ := a + (b - (c + (d - e)))

/-- The expression without parentheses -/
def expr_without_parentheses (a b c d e : ℤ) : ℤ := a + b - c + d - e

/-- The theorem stating that the expressions are equal when e = 8 -/
theorem lucky_lucy_theorem (a b c d : ℤ) (ha : a = 2) (hb : b = 4) (hc : c = 6) (hd : d = 8) :
  ∃ e : ℤ, expr_with_parentheses a b c d e = expr_without_parentheses a b c d e ∧ e = 8 := by
  sorry

end lucky_lucy_theorem_l1329_132970


namespace no_diametrical_opposition_possible_l1329_132994

/-- Represents a circular arrangement of numbers from 1 to 2014 -/
def CircularArrangement := Fin 2014 → Fin 2014

/-- Checks if a swap between two adjacent positions is valid -/
def validSwap (arr : CircularArrangement) (pos : Fin 2014) : Prop :=
  arr pos + arr ((pos + 1) % 2014) ≠ 2015

/-- Represents a sequence of swaps -/
def SwapSequence := List (Fin 2014)

/-- Applies a sequence of swaps to an arrangement -/
def applySwaps (initial : CircularArrangement) (swaps : SwapSequence) : CircularArrangement :=
  sorry

/-- Checks if a number is diametrically opposite its initial position -/
def isDiametricallyOpposite (initial final : CircularArrangement) (pos : Fin 2014) : Prop :=
  final pos = initial ((pos + 1007) % 2014)

/-- The main theorem stating that it's impossible to achieve diametrical opposition for all numbers -/
theorem no_diametrical_opposition_possible :
  ∀ (initial : CircularArrangement) (swaps : SwapSequence),
    (∀ (pos : Fin 2014), validSwap (applySwaps initial swaps) pos) →
    ¬(∀ (pos : Fin 2014), isDiametricallyOpposite initial (applySwaps initial swaps) pos) :=
  sorry

end no_diametrical_opposition_possible_l1329_132994


namespace invalid_altitudes_l1329_132982

/-- A triple of positive real numbers represents valid altitudes of a triangle if and only if
    the sum of the reciprocals of any two is greater than the reciprocal of the third. -/
def ValidAltitudes (h₁ h₂ h₃ : ℝ) : Prop :=
  h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧
  1/h₁ + 1/h₂ > 1/h₃ ∧
  1/h₂ + 1/h₃ > 1/h₁ ∧
  1/h₃ + 1/h₁ > 1/h₂

/-- The triple (5, 12, 13) cannot be the lengths of the three altitudes of a triangle. -/
theorem invalid_altitudes : ¬ ValidAltitudes 5 12 13 := by
  sorry

end invalid_altitudes_l1329_132982


namespace power_function_conditions_l1329_132995

def α_set : Set ℚ := {-1, 1, 2, 3/5, 7/2}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_domain_R (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ y, f x = y

def satisfies_conditions (α : ℚ) : Prop :=
  let f := fun x => x ^ (α : ℝ)
  has_domain_R f ∧ is_odd_function f

theorem power_function_conditions :
  ∀ α ∈ α_set, satisfies_conditions α ↔ α ∈ ({1, 3/5} : Set ℚ) :=
sorry

end power_function_conditions_l1329_132995


namespace p_necessary_not_sufficient_for_q_l1329_132966

theorem p_necessary_not_sufficient_for_q :
  (∃ x : ℝ, x < 1 ∧ ¬(x^2 + x - 2 < 0)) ∧
  (∀ x : ℝ, x^2 + x - 2 < 0 → x < 1) :=
by sorry

end p_necessary_not_sufficient_for_q_l1329_132966


namespace south_american_stamps_cost_l1329_132937

structure Country where
  name : String
  continent : String
  price : Rat
  stamps_50s : Nat
  stamps_60s : Nat

def brazil : Country := {
  name := "Brazil"
  continent := "South America"
  price := 6/100
  stamps_50s := 4
  stamps_60s := 7
}

def peru : Country := {
  name := "Peru"
  continent := "South America"
  price := 4/100
  stamps_50s := 6
  stamps_60s := 4
}

def france : Country := {
  name := "France"
  continent := "Europe"
  price := 6/100
  stamps_50s := 8
  stamps_60s := 4
}

def spain : Country := {
  name := "Spain"
  continent := "Europe"
  price := 5/100
  stamps_50s := 3
  stamps_60s := 9
}

def south_american_countries : List Country := [brazil, peru]

def total_cost (countries : List Country) : Rat :=
  countries.foldl (fun acc country => 
    acc + (country.price * (country.stamps_50s + country.stamps_60s : Rat))) 0

theorem south_american_stamps_cost :
  total_cost south_american_countries = 106/100 := by
  sorry

#eval total_cost south_american_countries

end south_american_stamps_cost_l1329_132937


namespace quadratic_equation_roots_l1329_132957

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ + 3 = 0 ∧ k * x₂^2 - 2 * x₂ + 3 = 0) ↔ 
  (k ≤ 1/3 ∧ k ≠ 0) :=
by sorry

end quadratic_equation_roots_l1329_132957


namespace f_at_two_l1329_132949

noncomputable section

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_monotonic : Monotone f
axiom f_condition : ∀ x : ℝ, f (f x - exp x) = exp 1 + 1

-- Theorem to prove
theorem f_at_two : f 2 = exp 2 + 1 := by sorry

end f_at_two_l1329_132949


namespace difference_solution_equation_problems_l1329_132993

/-- Definition of a difference solution equation -/
def is_difference_solution_equation (a b : ℝ) : Prop :=
  ∃ x : ℝ, a * x = b ∧ x = b - a

theorem difference_solution_equation_problems :
  -- Part 1
  is_difference_solution_equation 2 4 ∧
  -- Part 2
  (∀ a b : ℝ, is_difference_solution_equation 4 (a * b + a) →
    3 * (a * b + a) = 16) ∧
  -- Part 3
  (∀ m n : ℝ, is_difference_solution_equation 4 (m * n + m) ∧
    is_difference_solution_equation (-2) (m * n + n) →
    3 * (m * n + m) - 9 * (m * n + n)^2 = 0) :=
by
  sorry

end difference_solution_equation_problems_l1329_132993


namespace complex_trajectory_l1329_132938

/-- The trajectory of a complex number with given modulus -/
theorem complex_trajectory (x y : ℝ) (h : Complex.abs (x - 2 + y * Complex.I) = 2 * Real.sqrt 2) :
  (x - 2)^2 + y^2 = 8 := by
  sorry

end complex_trajectory_l1329_132938


namespace jaydee_typing_speed_l1329_132940

def typing_speed (hours : ℕ) (words : ℕ) : ℕ :=
  words / (hours * 60)

theorem jaydee_typing_speed :
  typing_speed 2 4560 = 38 :=
by sorry

end jaydee_typing_speed_l1329_132940


namespace carpet_width_in_cm_l1329_132981

/-- Proves that the width of the carpet is 1000 centimeters given the room dimensions and carpeting costs. -/
theorem carpet_width_in_cm (room_length room_breadth carpet_cost_per_meter total_cost : ℝ) 
  (h1 : room_length = 18)
  (h2 : room_breadth = 7.5)
  (h3 : carpet_cost_per_meter = 4.5)
  (h4 : total_cost = 810) : 
  (total_cost / carpet_cost_per_meter) / room_length * 100 = 1000 := by
  sorry

#check carpet_width_in_cm

end carpet_width_in_cm_l1329_132981


namespace det_value_for_quadratic_root_l1329_132939

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem det_value_for_quadratic_root (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  det (x + 1) (3*x) (x - 2) (x - 1) = 1 := by
  sorry

end det_value_for_quadratic_root_l1329_132939


namespace circle_center_l1329_132930

/-- The center of a circle defined by the equation (x+2)^2 + (y-1)^2 = 1 is at the point (-2, 1) -/
theorem circle_center (x y : ℝ) : 
  ((x + 2)^2 + (y - 1)^2 = 1) → ((-2, 1) : ℝ × ℝ) = (x, y) := by
sorry

end circle_center_l1329_132930


namespace adam_cat_food_packages_l1329_132992

theorem adam_cat_food_packages : 
  ∀ (c : ℕ), -- c represents the number of packages of cat food
  (10 * c = 7 * 5 + 55) → c = 9 :=
by
  sorry

end adam_cat_food_packages_l1329_132992


namespace function_symmetry_l1329_132967

theorem function_symmetry (a : ℝ) : 
  let f : ℝ → ℝ := λ x => (x^2 + x + 1) / (x^2 + 1)
  f a = 2/3 → f (-a) = 4/3 := by
sorry

end function_symmetry_l1329_132967


namespace multiplication_proof_l1329_132904

theorem multiplication_proof :
  ∀ (a b c : ℕ),
  a = 60 + b →
  c = 14 →
  a * c = 882 ∧
  68 * 14 = 952 :=
by
  sorry

end multiplication_proof_l1329_132904


namespace cos_equality_implies_angle_l1329_132953

theorem cos_equality_implies_angle (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (315 * π / 180) → n = 45 := by
  sorry

end cos_equality_implies_angle_l1329_132953


namespace fashion_pricing_increase_l1329_132941

theorem fashion_pricing_increase (C : ℝ) : 
  let retailer_price := 1.40 * C
  let customer_price := 1.6100000000000001 * C
  ((customer_price - retailer_price) / retailer_price) * 100 = 15 := by
sorry

end fashion_pricing_increase_l1329_132941


namespace correct_calculation_l1329_132917

theorem correct_calculation (x : ℝ) : x * 7 = 126 → x / 6 = 3 := by
  sorry

end correct_calculation_l1329_132917


namespace cheesecake_factory_savings_l1329_132934

/-- Calculates the combined savings of three employees over a period of time. -/
def combinedSavings (hourlyWage : ℚ) (hoursPerDay : ℚ) (daysPerWeek : ℚ) (weeks : ℚ)
  (savingsRate1 savingsRate2 savingsRate3 : ℚ) : ℚ :=
  let monthlyEarnings := hourlyWage * hoursPerDay * daysPerWeek * weeks
  let savings1 := monthlyEarnings * savingsRate1
  let savings2 := monthlyEarnings * savingsRate2
  let savings3 := monthlyEarnings * savingsRate3
  savings1 + savings2 + savings3

/-- The combined savings of three employees at a Cheesecake factory after four weeks. -/
theorem cheesecake_factory_savings :
  combinedSavings 10 10 5 4 (2/5) (3/5) (1/2) = 3000 := by
  sorry

end cheesecake_factory_savings_l1329_132934


namespace change_percentage_closest_to_five_l1329_132927

def item_prices : List ℚ := [12.99, 9.99, 7.99, 6.50, 4.99, 3.75, 1.27]
def payment : ℚ := 50

def total_price : ℚ := item_prices.sum
def change : ℚ := payment - total_price
def change_percentage : ℚ := (change / payment) * 100

theorem change_percentage_closest_to_five :
  ∀ x ∈ [3, 5, 7, 10, 12], |change_percentage - 5| ≤ |change_percentage - x| :=
by sorry

end change_percentage_closest_to_five_l1329_132927


namespace root_range_implies_a_range_l1329_132907

theorem root_range_implies_a_range :
  ∀ a : ℝ,
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ x^2 - 4*x + 3*a^2 - 2 = 0) →
  a ∈ Set.Icc (-Real.sqrt (5/3)) (Real.sqrt (5/3)) :=
by sorry

end root_range_implies_a_range_l1329_132907


namespace sets_intersection_and_complement_l1329_132973

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def B : Set ℝ := {x | (x - 2) / (x + 3) > 0}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- State the theorem
theorem sets_intersection_and_complement (a : ℝ) 
  (h : A ∩ C a = C a) : 
  (A ∩ B = Set.Ioc 2 3) ∧ 
  ((Set.univ \ A) ∪ (Set.univ \ B) = Set.Iic 2 ∪ Set.Ioi 3) ∧
  (a ≤ 3) := by sorry

end sets_intersection_and_complement_l1329_132973


namespace extra_page_number_l1329_132956

theorem extra_page_number (n : ℕ) (k : ℕ) : 
  n = 62 → 
  (n * (n + 1)) / 2 + k = 1986 → 
  k = 33 := by
sorry

end extra_page_number_l1329_132956


namespace necessary_but_not_sufficient_l1329_132951

/-- The function f(x) = -x³ + 2ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 2*a*x

/-- f is monotonically decreasing on (-∞, 1] -/
def is_monotone_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ 1 → f a x ≥ f a y

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, is_monotone_decreasing_on_interval a → a < 3/2) ∧
  (∃ a : ℝ, a < 3/2 ∧ ¬is_monotone_decreasing_on_interval a) :=
sorry

end necessary_but_not_sufficient_l1329_132951


namespace smallest_integer_satisfying_inequalities_l1329_132996

theorem smallest_integer_satisfying_inequalities :
  ∀ x : ℤ, (x + 8 > 10 ∧ -3*x < -9) → x ≥ 4 :=
by sorry

end smallest_integer_satisfying_inequalities_l1329_132996


namespace ab_value_l1329_132924

theorem ab_value (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 150) : a * b = 25 := by
  sorry

end ab_value_l1329_132924


namespace arithmetic_sequence_terms_l1329_132987

/-- An arithmetic sequence with first term 11, common difference 4, and last term 107 has 25 terms -/
theorem arithmetic_sequence_terms (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 11 → d = 4 → aₙ = 107 → aₙ = a₁ + (n - 1) * d → n = 25 := by
sorry

end arithmetic_sequence_terms_l1329_132987


namespace statement_1_statement_4_l1329_132916

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contained : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Statement ①
theorem statement_1 (m n : Line) (α : Plane) :
  perpendicular m n → perpendicularLP m α → ¬contained n α → parallel n α :=
sorry

-- Statement ④
theorem statement_4 (m n : Line) (α β : Plane) :
  perpendicular m n → perpendicularLP m α → perpendicularLP n β → perpendicularPP α β :=
sorry

end statement_1_statement_4_l1329_132916


namespace grandsons_age_l1329_132984

theorem grandsons_age (grandson_age grandfather_age : ℕ) : 
  grandfather_age = 6 * grandson_age →
  grandfather_age + 4 + grandson_age + 4 = 78 →
  grandson_age = 10 := by
sorry

end grandsons_age_l1329_132984


namespace square_root_of_nine_l1329_132997

theorem square_root_of_nine : 
  ∃ (x : ℝ), x^2 = 9 ↔ x = 3 ∨ x = -3 :=
by sorry

end square_root_of_nine_l1329_132997


namespace tshirt_profit_calculation_l1329_132929

-- Define the profit per jersey
def profit_per_jersey : ℕ := 5

-- Define the profit per t-shirt
def profit_per_tshirt : ℕ := 215

-- Define the number of t-shirts sold
def tshirts_sold : ℕ := 20

-- Define the number of jerseys sold
def jerseys_sold : ℕ := 64

-- Theorem to prove
theorem tshirt_profit_calculation :
  tshirts_sold * profit_per_tshirt = 4300 := by
  sorry

end tshirt_profit_calculation_l1329_132929


namespace sum_of_digits_l1329_132962

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    b = a + c ∧
    100 * c + 10 * b + a = n + 99 ∧
    n = 253

theorem sum_of_digits (n : ℕ) (h : is_valid_number n) : 
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 10 := by
  sorry

end sum_of_digits_l1329_132962


namespace hyperbola_center_l1329_132943

/-- The center of a hyperbola with foci at (3, 6) and (11, 10) is at (7, 8) -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) (h1 : f1 = (3, 6)) (h2 : f2 = (11, 10)) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  center = (7, 8) := by sorry

end hyperbola_center_l1329_132943


namespace apple_boxes_weights_l1329_132969

theorem apple_boxes_weights (a b c d : ℝ) 
  (h1 : a + b + c = 70)
  (h2 : a + b + d = 80)
  (h3 : a + c + d = 73)
  (h4 : b + c + d = 77)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0) :
  a = 23 ∧ b = 27 ∧ c = 20 ∧ d = 30 := by
sorry

end apple_boxes_weights_l1329_132969


namespace max_participants_answering_A_l1329_132991

theorem max_participants_answering_A (total : ℕ) (a b c : ℕ) : 
  total = 39 →
  a + b + c + (a + 3*b - 5) + 3*b + (total - (2*a + 6*b - 5)) = total →
  a = b + c →
  2*(total - (2*a + 6*b - 5)) = 3*b →
  (2*a + 9*b = 44 ∧ a ≥ 0 ∧ b ≥ 0) →
  (∃ max_A : ℕ, max_A = 2*a + 3*b - 5 ∧ max_A ≤ 23 ∧ 
   ∀ other_A : ℕ, other_A = 2*a' + 3*b' - 5 → 
   (2*a' + 9*b' = 44 ∧ a' ≥ 0 ∧ b' ≥ 0) → other_A ≤ max_A) :=
by sorry

end max_participants_answering_A_l1329_132991


namespace largest_three_digit_multiple_of_13_l1329_132965

theorem largest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 13 = 0 → n ≤ 987 :=
by
  sorry

end largest_three_digit_multiple_of_13_l1329_132965


namespace quadratic_function_properties_l1329_132921

/-- Given a quadratic function f(x) = x^2 - 2ax + b where a and b are real numbers,
    and the solution set of f(x) ≤ 0 is [-1, 2], this theorem proves two statements about f. -/
theorem quadratic_function_properties (a b : ℝ) 
    (f : ℝ → ℝ) 
    (h_f : ∀ x, f x = x^2 - 2*a*x + b) 
    (h_solution_set : Set.Icc (-1 : ℝ) 2 = {x | f x ≤ 0}) : 
  (∀ x, b*x^2 - 2*a*x + 1 ≤ 0 ↔ x ≤ -1 ∨ x ≥ 1/2) ∧ 
  (b = a^2 → 
   (∀ x₁ ∈ Set.Icc 2 4, ∃ x₂ ∈ Set.Icc 2 4, f x₁ * f x₂ = 1) → 
   a = 3 + Real.sqrt 2 ∨ a = 3 - Real.sqrt 2) :=
sorry

end quadratic_function_properties_l1329_132921


namespace sin_two_x_value_l1329_132963

theorem sin_two_x_value (x : ℝ) 
  (h : Real.sin (Real.pi + x) + Real.sin ((3 * Real.pi) / 2 + x) = 1/2) : 
  Real.sin (2 * x) = -(3/4) := by
  sorry

end sin_two_x_value_l1329_132963


namespace equation_solution_l1329_132920

theorem equation_solution (y : ℚ) : 
  (8 * y^2 + 127 * y + 5) / (4 * y + 41) = 2 * y + 3 → y = 118 / 33 := by
  sorry

end equation_solution_l1329_132920


namespace remainder_zero_l1329_132908

def nines : ℕ := 10^20089 - 1
def threes : ℕ := 3 * (10^20083 - 1) / 9

theorem remainder_zero :
  (nines^2007 - threes^2007) % 11 = 0 := by sorry

end remainder_zero_l1329_132908


namespace sin_30_degrees_l1329_132946

theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by sorry

end sin_30_degrees_l1329_132946


namespace P_less_than_Q_l1329_132959

theorem P_less_than_Q (a : ℝ) (ha : a ≥ 0) : 
  Real.sqrt a + Real.sqrt (a + 7) < Real.sqrt (a + 3) + Real.sqrt (a + 4) :=
by sorry

end P_less_than_Q_l1329_132959


namespace melissa_games_played_l1329_132947

/-- Given that Melissa scored 12 points in each game and a total of 36 points,
    prove that she played 3 games. -/
theorem melissa_games_played (points_per_game : ℕ) (total_points : ℕ) 
  (h1 : points_per_game = 12) 
  (h2 : total_points = 36) : 
  total_points / points_per_game = 3 := by
  sorry

end melissa_games_played_l1329_132947


namespace hyperbola_asymptote_l1329_132971

/-- The asymptote equation of a hyperbola with specific properties -/
theorem hyperbola_asymptote (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_focal : 2 * Real.sqrt (a^2 + b^2) = Real.sqrt 3 * (2 * a)) :
  ∃ (k : ℝ), k = Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → y = k * x ∨ y = -k * x) :=
sorry

end hyperbola_asymptote_l1329_132971


namespace pure_imaginary_condition_l1329_132914

theorem pure_imaginary_condition (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x - 1) * Complex.I).re = 0 ∧ 
  (((x^2 - 1) : ℂ) + (x - 1) * Complex.I).im ≠ 0 → 
  x = -1 := by sorry

end pure_imaginary_condition_l1329_132914


namespace vector_expression_evaluation_l1329_132926

/-- Given the vector expression, prove that it equals the result vector. -/
theorem vector_expression_evaluation :
  (⟨3, -2⟩ : ℝ × ℝ) - 5 • ⟨2, -6⟩ + ⟨0, 3⟩ = ⟨-7, 31⟩ := by
  sorry

end vector_expression_evaluation_l1329_132926


namespace math_club_exclusive_members_l1329_132976

theorem math_club_exclusive_members :
  ∀ (total_students : ℕ) (both_clubs : ℕ) (math_club : ℕ) (science_club : ℕ),
    total_students = 30 →
    both_clubs = 2 →
    math_club = 3 * science_club →
    total_students = math_club + science_club - both_clubs →
    math_club - both_clubs = 20 :=
by sorry

end math_club_exclusive_members_l1329_132976


namespace equation_solution_range_l1329_132960

theorem equation_solution_range (x a : ℝ) : 
  (2 * x + a) / (x - 1) = 1 → x > 0 → x ≠ 1 → a < -1 ∧ a ≠ -2 :=
by sorry

end equation_solution_range_l1329_132960


namespace otimes_composition_l1329_132990

-- Define the new operation
def otimes (x y : ℝ) : ℝ := x^2 + y^2

-- State the theorem
theorem otimes_composition (x : ℝ) : otimes x (otimes x x) = x^2 + 4*x^4 := by
  sorry

end otimes_composition_l1329_132990


namespace clay_pot_earnings_l1329_132900

/-- Calculate the money earned from selling clay pots --/
theorem clay_pot_earnings (total_pots : ℕ) (cracked_fraction : ℚ) (price_per_pot : ℕ) : 
  total_pots = 80 →
  cracked_fraction = 2 / 5 →
  price_per_pot = 40 →
  (total_pots : ℚ) * (1 - cracked_fraction) * price_per_pot = 1920 := by
  sorry

end clay_pot_earnings_l1329_132900


namespace house_height_l1329_132942

theorem house_height (house_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ)
  (h1 : house_shadow = 84)
  (h2 : pole_height = 14)
  (h3 : pole_shadow = 28) :
  (house_shadow / pole_shadow) * pole_height = 42 :=
by sorry

end house_height_l1329_132942


namespace arrangements_count_l1329_132931

/-- The number of applicants --/
def num_applicants : ℕ := 5

/-- The number of students to be selected --/
def num_selected : ℕ := 3

/-- The number of events --/
def num_events : ℕ := 3

/-- Function to calculate the number of arrangements --/
def num_arrangements (n_applicants : ℕ) (n_selected : ℕ) (n_events : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of arrangements --/
theorem arrangements_count :
  num_arrangements num_applicants num_selected num_events = 48 :=
sorry

end arrangements_count_l1329_132931


namespace regular_polygons_covering_plane_l1329_132986

/-- A function that returns true if a regular n-gon can completely and tightly cover a plane without gaps -/
def can_cover_plane (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ k : ℕ, k ≥ 3 ∧ k * (1 - 2 / n) = 2

/-- The theorem stating which regular polygons can completely and tightly cover a plane without gaps -/
theorem regular_polygons_covering_plane :
  ∀ n : ℕ, can_cover_plane n ↔ n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

end regular_polygons_covering_plane_l1329_132986


namespace hexagon_angle_Q_l1329_132944

/-- A hexagon with specified interior angles -/
structure Hexagon :=
  (angleS : ℝ)
  (angleT : ℝ)
  (angleU : ℝ)
  (angleV : ℝ)
  (angleW : ℝ)
  (h_angleS : angleS = 120)
  (h_angleT : angleT = 130)
  (h_angleU : angleU = 140)
  (h_angleV : angleV = 100)
  (h_angleW : angleW = 85)

/-- The measure of angle Q in the hexagon -/
def angleQ (h : Hexagon) : ℝ := 720 - (h.angleS + h.angleT + h.angleU + h.angleV + h.angleW)

theorem hexagon_angle_Q (h : Hexagon) : angleQ h = 145 := by
  sorry

end hexagon_angle_Q_l1329_132944


namespace parallelogram_area_calculation_l1329_132988

/-- The area of the parallelogram formed by vectors u and z -/
def parallelogramArea (u z : Fin 2 → ℝ) : ℝ :=
  |u 0 * z 1 - u 1 * z 0|

/-- The problem statement -/
theorem parallelogram_area_calculation :
  let u : Fin 2 → ℝ := ![3, 4]
  let z : Fin 2 → ℝ := ![8, -1]
  parallelogramArea u z = 35 := by
sorry

end parallelogram_area_calculation_l1329_132988


namespace cow_chicken_problem_l1329_132925

theorem cow_chicken_problem (c h : ℕ) : 
  4 * c + 2 * h = 2 * (c + h) + 16 → c = 8 := by
  sorry

end cow_chicken_problem_l1329_132925


namespace prob_second_day_A_l1329_132964

-- Define the probabilities
def prob_A_given_A : ℝ := 0.7
def prob_A_given_B : ℝ := 0.5
def prob_first_day_A : ℝ := 0.5
def prob_first_day_B : ℝ := 0.5

-- State the theorem
theorem prob_second_day_A :
  prob_first_day_A * prob_A_given_A + prob_first_day_B * prob_A_given_B = 0.6 := by
  sorry

end prob_second_day_A_l1329_132964


namespace nice_sequence_classification_l1329_132954

/-- A sequence of integers -/
def IntegerSequence := ℕ → ℤ

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- A nice sequence satisfies the given condition for some function f -/
def IsNice (a : IntegerSequence) : Prop :=
  ∃ f : PositiveIntFunction, ∀ i j n : ℕ+,
    (a i.val - a j.val) % n.val = 0 ↔ (i.val - j.val) % f n = 0

/-- A sequence is periodic with period k -/
def IsPeriodic (a : IntegerSequence) (k : ℕ+) : Prop :=
  ∀ i : ℕ, a (i + k) = a i

/-- A sequence is an arithmetic sequence -/
def IsArithmetic (a : IntegerSequence) : Prop :=
  ∃ d : ℤ, ∀ i : ℕ, a (i + 1) = a i + d

/-- The main theorem: nice sequences are either constant, periodic with period 2, or arithmetic -/
theorem nice_sequence_classification (a : IntegerSequence) :
  IsNice a → (IsPeriodic a 1 ∨ IsPeriodic a 2 ∨ IsArithmetic a) :=
sorry

end nice_sequence_classification_l1329_132954


namespace quadratic_equation_solution_l1329_132948

theorem quadratic_equation_solution :
  let x₁ : ℝ := (2 + Real.sqrt 3) / 2
  let x₂ : ℝ := (2 - Real.sqrt 3) / 2
  4 * x₁^2 - 8 * x₁ + 1 = 0 ∧ 4 * x₂^2 - 8 * x₂ + 1 = 0 := by
  sorry

end quadratic_equation_solution_l1329_132948


namespace lasagna_mince_amount_l1329_132999

/-- Proves that the amount of ground mince used for each lasagna is 2 pounds -/
theorem lasagna_mince_amount 
  (total_dishes : ℕ) 
  (cottage_pie_mince : ℕ) 
  (total_mince : ℕ) 
  (cottage_pies : ℕ) 
  (h1 : total_dishes = 100)
  (h2 : cottage_pie_mince = 3)
  (h3 : total_mince = 500)
  (h4 : cottage_pies = 100) :
  (total_mince - cottage_pies * cottage_pie_mince) / (total_dishes - cottage_pies) = 2 := by
  sorry

#check lasagna_mince_amount

end lasagna_mince_amount_l1329_132999


namespace absolute_value_expression_l1329_132989

theorem absolute_value_expression (x : ℤ) (h : x = -2023) :
  |abs (abs x - x) - abs x| - x = 4046 := by
  sorry

end absolute_value_expression_l1329_132989


namespace divisibility_of_x_l1329_132977

theorem divisibility_of_x (x y : ℕ+) (h1 : 2 * x ^ 2 - 1 = y ^ 15) (h2 : x > 1) :
  5 ∣ x.val := by
  sorry

end divisibility_of_x_l1329_132977


namespace pet_store_cages_l1329_132903

theorem pet_store_cages (total_birds : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) 
  (h1 : total_birds = 54)
  (h2 : parrots_per_cage = 2)
  (h3 : parakeets_per_cage = 7) :
  total_birds / (parrots_per_cage + parakeets_per_cage) = 6 := by
  sorry

end pet_store_cages_l1329_132903


namespace different_color_chip_probability_l1329_132978

theorem different_color_chip_probability :
  let total_chips : ℕ := 12
  let red_chips : ℕ := 7
  let green_chips : ℕ := 5
  let prob_red : ℚ := red_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  let prob_different_colors : ℚ := prob_red * prob_green + prob_green * prob_red
  prob_different_colors = 35 / 72 := by sorry

end different_color_chip_probability_l1329_132978


namespace housing_development_l1329_132983

theorem housing_development (total : ℕ) (garage : ℕ) (pool : ℕ) (neither : ℕ) 
  (h_total : total = 90)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_neither : neither = 35) :
  garage + pool - (total - neither) = 35 := by
  sorry

end housing_development_l1329_132983


namespace rayden_lily_duck_ratio_l1329_132975

/-- Proves the ratio of Rayden's ducks to Lily's ducks is 3:1 -/
theorem rayden_lily_duck_ratio :
  let lily_ducks : ℕ := 20
  let lily_geese : ℕ := 10
  let rayden_geese : ℕ := 4 * lily_geese
  let total_difference : ℕ := 70
  let rayden_total : ℕ := lily_ducks + lily_geese + total_difference
  let rayden_ducks : ℕ := rayden_total - rayden_geese
  (rayden_ducks : ℚ) / lily_ducks = 3 / 1 := by sorry

end rayden_lily_duck_ratio_l1329_132975


namespace quadratic_equation_solution_l1329_132901

theorem quadratic_equation_solution : ∃ (a b : ℝ), 
  (a^2 - 6*a + 11 = 27) ∧ 
  (b^2 - 6*b + 11 = 27) ∧ 
  (a ≥ b) ∧ 
  (3*a - 2*b = 28) := by
  sorry

end quadratic_equation_solution_l1329_132901
