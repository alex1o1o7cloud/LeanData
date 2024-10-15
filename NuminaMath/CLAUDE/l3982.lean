import Mathlib

namespace NUMINAMATH_CALUDE_sin_alpha_on_ray_l3982_398277

/-- If the terminal side of angle α lies on the ray y = -√3x (x < 0), then sin α = √3/2 -/
theorem sin_alpha_on_ray (α : Real) : 
  (∃ (x y : Real), x < 0 ∧ y = -Real.sqrt 3 * x ∧ 
   (∃ (r : Real), x^2 + y^2 = r^2 ∧ Real.sin α = y / r)) →
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_on_ray_l3982_398277


namespace NUMINAMATH_CALUDE_julian_airplane_models_l3982_398259

theorem julian_airplane_models : 
  ∀ (total_legos : ℕ) (legos_per_model : ℕ) (additional_legos_needed : ℕ),
    total_legos = 400 →
    legos_per_model = 240 →
    additional_legos_needed = 80 →
    (total_legos + additional_legos_needed) / legos_per_model = 2 := by
  sorry

end NUMINAMATH_CALUDE_julian_airplane_models_l3982_398259


namespace NUMINAMATH_CALUDE_number_problem_l3982_398278

theorem number_problem : ∃ x : ℚ, x = 15 + (x * 9/64) + (x * 1/2) ∧ x = 960/23 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3982_398278


namespace NUMINAMATH_CALUDE_peters_marbles_l3982_398221

theorem peters_marbles (n : ℕ) (orange purple silver white : ℕ) : 
  n > 0 →
  orange = n / 2 →
  purple = n / 5 →
  silver = 8 →
  white = n - (orange + purple + silver) →
  n = orange + purple + silver + white →
  (∀ m : ℕ, m > 0 ∧ m < n → 
    m / 2 + m / 5 + 8 + (m - (m / 2 + m / 5 + 8)) ≠ m) →
  white = 1 := by
sorry

end NUMINAMATH_CALUDE_peters_marbles_l3982_398221


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3982_398286

theorem repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = 0.1 * c + 0.01 * d + 0.001 * c + 0.0001 * d + 0.00001 * c + 0.000001 * d →
  c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3982_398286


namespace NUMINAMATH_CALUDE_CCl4_formation_l3982_398230

-- Define the initial amounts of reactants
def initial_C2H6 : ℝ := 2
def initial_Cl2 : ℝ := 14

-- Define the stoichiometric ratio for each step
def stoichiometric_ratio : ℝ := 1

-- Define the number of reaction steps
def num_steps : ℕ := 4

-- Theorem statement
theorem CCl4_formation (remaining_Cl2 : ℝ → ℝ) 
  (h1 : remaining_Cl2 0 = initial_Cl2)
  (h2 : ∀ n : ℕ, n < num_steps → 
    remaining_Cl2 (n + 1) = remaining_Cl2 n - stoichiometric_ratio * initial_C2H6)
  (h3 : ∀ n : ℕ, n ≤ num_steps → remaining_Cl2 n ≥ 0) :
  remaining_Cl2 num_steps = initial_Cl2 - num_steps * stoichiometric_ratio * initial_C2H6 ∧
  initial_C2H6 = initial_C2H6 :=
by sorry

end NUMINAMATH_CALUDE_CCl4_formation_l3982_398230


namespace NUMINAMATH_CALUDE_inequality_solution_l3982_398294

theorem inequality_solution :
  {x : ℝ | |(6 - x) / 4| < 3 ∧ x ≥ 2} = Set.Ici 2 ∩ Set.Iio 18 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3982_398294


namespace NUMINAMATH_CALUDE_actor_stage_time_l3982_398208

theorem actor_stage_time (actors_at_once : ℕ) (total_actors : ℕ) (show_duration : ℕ) : 
  actors_at_once = 5 → total_actors = 20 → show_duration = 60 → 
  (show_duration / (total_actors / actors_at_once) : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_actor_stage_time_l3982_398208


namespace NUMINAMATH_CALUDE_max_value_of_function_l3982_398274

theorem max_value_of_function (x : ℝ) (h : x < 0) :
  3 * x + 4 / x ≤ -4 * Real.sqrt 3 ∧ ∃ y < 0, 3 * y + 4 / y = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3982_398274


namespace NUMINAMATH_CALUDE_variance_of_surviving_trees_l3982_398252

/-- The number of osmanthus trees transplanted -/
def n : ℕ := 4

/-- The probability of survival for each tree -/
def p : ℚ := 4/5

/-- The random variable representing the number of surviving trees -/
def X : ℕ → ℚ := sorry

/-- The expected value of X -/
def E_X : ℚ := n * p

/-- The variance of X -/
def Var_X : ℚ := n * p * (1 - p)

theorem variance_of_surviving_trees :
  Var_X = 16/25 := by sorry

end NUMINAMATH_CALUDE_variance_of_surviving_trees_l3982_398252


namespace NUMINAMATH_CALUDE_x_minus_y_equals_half_l3982_398245

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {2, 0, x}
def B (x y : ℝ) : Set ℝ := {1/x, |x|, y/x}

-- State the theorem
theorem x_minus_y_equals_half (x y : ℝ) : A x = B x y → x - y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_half_l3982_398245


namespace NUMINAMATH_CALUDE_probability_of_meeting_l3982_398262

/-- Two people arrive independently and uniformly at random within a 2-hour interval -/
def arrival_interval : ℝ := 2

/-- Each person stays for 30 minutes after arrival -/
def stay_duration : ℝ := 0.5

/-- The maximum arrival time for each person is 30 minutes before the end of the 2-hour interval -/
def max_arrival_time : ℝ := arrival_interval - stay_duration

/-- The probability of two people seeing each other given the conditions -/
theorem probability_of_meeting :
  let total_area : ℝ := arrival_interval ^ 2
  let overlap_area : ℝ := total_area - 2 * (stay_duration ^ 2 / 2)
  overlap_area / total_area = 15 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_of_meeting_l3982_398262


namespace NUMINAMATH_CALUDE_distinct_roots_of_f_l3982_398279

-- Define the function
def f (x : ℝ) : ℝ := (x - 5) * (x + 3)^2

-- Theorem statement
theorem distinct_roots_of_f :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ ∀ x, f x = 0 → x = r₁ ∨ x = r₂ :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_of_f_l3982_398279


namespace NUMINAMATH_CALUDE_root_product_value_l3982_398253

theorem root_product_value (m n : ℝ) : 
  m^2 - 2019*m - 1 = 0 → 
  n^2 - 2019*n - 1 = 0 → 
  (m^2 - 2019*m + 3) * (n^2 - 2019*n + 4) = 20 := by
sorry

end NUMINAMATH_CALUDE_root_product_value_l3982_398253


namespace NUMINAMATH_CALUDE_parabola_directrix_l3982_398227

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = 16 * x^2

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -1/64

-- Theorem statement
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ d = -1/64 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3982_398227


namespace NUMINAMATH_CALUDE_third_candidate_votes_l3982_398298

theorem third_candidate_votes : 
  let total_votes : ℕ := 23400
  let candidate1_votes : ℕ := 7636
  let candidate2_votes : ℕ := 11628
  let winning_percentage : ℚ := 49.69230769230769 / 100
  ∀ (third_candidate_votes : ℕ),
    (candidate1_votes + candidate2_votes + third_candidate_votes = total_votes) ∧
    (candidate2_votes = (winning_percentage * total_votes).floor) →
    third_candidate_votes = 4136 := by
sorry

end NUMINAMATH_CALUDE_third_candidate_votes_l3982_398298


namespace NUMINAMATH_CALUDE_max_pages_copied_l3982_398285

-- Define the cost per page in cents
def cost_per_page : ℕ := 3

-- Define the budget in dollars
def budget : ℕ := 15

-- Define the function to calculate the number of pages
def pages_copied (cost : ℕ) (budget : ℕ) : ℕ :=
  (budget * 100) / cost

-- Theorem statement
theorem max_pages_copied :
  pages_copied cost_per_page budget = 500 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_copied_l3982_398285


namespace NUMINAMATH_CALUDE_base_10_to_base_5_88_l3982_398284

def to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_10_to_base_5_88 : to_base_5 88 = [3, 2, 3] := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_5_88_l3982_398284


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l3982_398267

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem complement_intersection_equals_set : 
  (A ∪ B)ᶜ ∩ (A ∩ B)ᶜ = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l3982_398267


namespace NUMINAMATH_CALUDE_meghan_weight_conversion_l3982_398296

/-- Given a base b, this function calculates the value of a number represented as 451 in base b -/
def value_in_base_b (b : ℕ) : ℕ := 4 * b^2 + 5 * b + 1

/-- Given a base b, this function calculates the value of a number represented as 127 in base 2b -/
def value_in_base_2b (b : ℕ) : ℕ := 1 * (2*b)^2 + 2 * (2*b) + 7

/-- Theorem stating that if a number is represented as 451 in base b and 127 in base 2b, 
    then it is equal to 175 in base 10 -/
theorem meghan_weight_conversion (b : ℕ) : 
  value_in_base_b b = value_in_base_2b b → value_in_base_b b = 175 := by
  sorry

#eval value_in_base_b 6  -- Should output 175
#eval value_in_base_2b 6  -- Should also output 175

end NUMINAMATH_CALUDE_meghan_weight_conversion_l3982_398296


namespace NUMINAMATH_CALUDE_coffee_blend_price_l3982_398219

/-- Proves the price of the second blend of coffee given the conditions of the problem -/
theorem coffee_blend_price
  (total_blend : ℝ)
  (target_price : ℝ)
  (first_blend_price : ℝ)
  (first_blend_amount : ℝ)
  (h1 : total_blend = 20)
  (h2 : target_price = 8.4)
  (h3 : first_blend_price = 9)
  (h4 : first_blend_amount = 8)
  : ∃ (second_blend_price : ℝ),
    second_blend_price = 8 ∧
    total_blend * target_price =
      first_blend_amount * first_blend_price +
      (total_blend - first_blend_amount) * second_blend_price :=
by sorry

end NUMINAMATH_CALUDE_coffee_blend_price_l3982_398219


namespace NUMINAMATH_CALUDE_base_6_addition_l3982_398247

/-- Addition in base 6 -/
def add_base_6 (a b : ℕ) : ℕ :=
  (a + b) % 36

/-- Conversion from base 6 to decimal -/
def base_6_to_decimal (n : ℕ) : ℕ :=
  (n / 10) * 6 + (n % 10)

theorem base_6_addition :
  add_base_6 (base_6_to_decimal 4) (base_6_to_decimal 14) = base_6_to_decimal 22 := by
  sorry

#eval add_base_6 (base_6_to_decimal 4) (base_6_to_decimal 14)
#eval base_6_to_decimal 22

end NUMINAMATH_CALUDE_base_6_addition_l3982_398247


namespace NUMINAMATH_CALUDE_same_club_probability_l3982_398290

theorem same_club_probability (n : ℕ) (h : n = 8) :
  let p := 1 / n
  (n : ℝ) * p * p = 1 / n :=
by
  sorry

end NUMINAMATH_CALUDE_same_club_probability_l3982_398290


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l3982_398264

-- Define repeating decimal 0.overline{36}
def repeating_36 : ℚ := 36 / 99

-- Define repeating decimal 0.overline{09}
def repeating_09 : ℚ := 9 / 99

-- Theorem statement
theorem repeating_decimal_ratio : repeating_36 / repeating_09 = 4 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l3982_398264


namespace NUMINAMATH_CALUDE_major_axis_length_for_given_conditions_l3982_398265

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def majorAxisLength (cylinderRadius : ℝ) (majorToMinorRatio : ℝ) : ℝ :=
  2 * cylinderRadius * majorToMinorRatio

/-- Theorem: The major axis length is 6 for a cylinder of radius 2 and 50% longer major axis --/
theorem major_axis_length_for_given_conditions :
  majorAxisLength 2 1.5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_for_given_conditions_l3982_398265


namespace NUMINAMATH_CALUDE_twice_slope_line_equation_l3982_398200

/-- Given a line L1: 2x + 3y + 3 = 0, prove that the line L2 passing through (1,0) 
    with a slope twice that of L1 has the equation 4x + 3y = 4. -/
theorem twice_slope_line_equation : 
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 2 * x + 3 * y + 3 = 0
  let m1 : ℝ := -2 / 3  -- slope of L1
  let m2 : ℝ := 2 * m1  -- slope of L2
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 3 * y = 4
  (∀ x y, L2 x y ↔ y - 0 = m2 * (x - 1)) ∧ L2 1 0 := by
  sorry


end NUMINAMATH_CALUDE_twice_slope_line_equation_l3982_398200


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3982_398258

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3982_398258


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l3982_398229

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = π) : -- condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
    sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l3982_398229


namespace NUMINAMATH_CALUDE_square_sum_given_linear_and_product_l3982_398213

theorem square_sum_given_linear_and_product (x y : ℝ) 
  (h1 : x + 2*y = 6) (h2 : x*y = -12) : x^2 + 4*y^2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_linear_and_product_l3982_398213


namespace NUMINAMATH_CALUDE_percentage_less_l3982_398276

theorem percentage_less (q y w z : ℝ) : 
  w = 0.6 * q →
  z = 0.54 * y →
  z = 1.5 * w →
  q = 0.6 * y :=
by sorry

end NUMINAMATH_CALUDE_percentage_less_l3982_398276


namespace NUMINAMATH_CALUDE_transportation_is_car_l3982_398203

/-- Represents different modes of transportation -/
inductive TransportMode
  | Walking
  | Bicycle
  | Car

/-- Definition of a transportation mode with its speed -/
structure Transportation where
  mode : TransportMode
  speed : ℝ  -- Speed in kilometers per hour

/-- Theorem stating that a transportation with speed 70 km/h is a car -/
theorem transportation_is_car (t : Transportation) (h : t.speed = 70) : t.mode = TransportMode.Car := by
  sorry


end NUMINAMATH_CALUDE_transportation_is_car_l3982_398203


namespace NUMINAMATH_CALUDE_climbing_time_problem_l3982_398282

/-- The time it takes for Jason to be 42 feet higher than Matt, given their climbing rates. -/
theorem climbing_time_problem (matt_rate jason_rate : ℝ) (height_difference : ℝ)
  (h_matt : matt_rate = 6)
  (h_jason : jason_rate = 12)
  (h_diff : height_difference = 42) :
  (height_difference / (jason_rate - matt_rate)) = 7 :=
by sorry

end NUMINAMATH_CALUDE_climbing_time_problem_l3982_398282


namespace NUMINAMATH_CALUDE_amusement_park_groups_l3982_398209

theorem amusement_park_groups (n : ℕ) (k : ℕ) : n = 7 ∧ k = 4 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_groups_l3982_398209


namespace NUMINAMATH_CALUDE_min_value_inequality_l3982_398272

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + 2 * y + 6) :
  1 / x + 1 / (2 * y) ≥ 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3982_398272


namespace NUMINAMATH_CALUDE_function_value_l3982_398263

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (1/4 - a)*x + 2*a

theorem function_value (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_function_value_l3982_398263


namespace NUMINAMATH_CALUDE_hanna_erasers_l3982_398234

/-- Given information about erasers owned by Tanya, Rachel, and Hanna -/
theorem hanna_erasers (tanya_erasers : ℕ) (tanya_red_erasers : ℕ) (rachel_erasers : ℕ) (hanna_erasers : ℕ) : 
  tanya_erasers = 20 →
  tanya_red_erasers = tanya_erasers / 2 →
  rachel_erasers = tanya_red_erasers / 2 - 3 →
  hanna_erasers = 2 * rachel_erasers →
  hanna_erasers = 4 := by
sorry

end NUMINAMATH_CALUDE_hanna_erasers_l3982_398234


namespace NUMINAMATH_CALUDE_correct_pairings_l3982_398224

/-- The number of possible pairings for the first round of a tennis tournament with 2n players -/
def numPairings (n : ℕ) : ℚ :=
  (Nat.factorial (2 * n)) / ((2 ^ n) * Nat.factorial n)

/-- Theorem stating that numPairings gives the correct number of possible pairings -/
theorem correct_pairings (n : ℕ) :
  numPairings n = (Nat.factorial (2 * n)) / ((2 ^ n) * Nat.factorial n) := by
  sorry

end NUMINAMATH_CALUDE_correct_pairings_l3982_398224


namespace NUMINAMATH_CALUDE_sum_range_l3982_398232

theorem sum_range : ∃ (x : ℚ), 10.5 < x ∧ x < 11 ∧ x = 2 + 1/8 + 3 + 1/3 + 5 + 1/18 := by
  sorry

end NUMINAMATH_CALUDE_sum_range_l3982_398232


namespace NUMINAMATH_CALUDE_triangle_area_tangent_circles_l3982_398249

/-- Given two non-overlapping circles with radii r₁ and r₂, where one common internal tangent
    is perpendicular to one common external tangent, the area S of the triangle formed by
    these tangents and the third common tangent satisfies one of two formulas. -/
theorem triangle_area_tangent_circles (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₁ ≠ r₂) :
  ∃ S : ℝ, (S = (r₁ * r₂ * (r₁ + r₂)) / |r₁ - r₂|) ∨ (S = (r₁ * r₂ * |r₁ - r₂|) / (r₁ + r₂)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_tangent_circles_l3982_398249


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3982_398217

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

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_length 
  (q : Quadrilateral) 
  (hConvex : isConvex q) 
  (hAB : distance q.A q.B = 8)
  (hCD : distance q.C q.D = 18)
  (hAC : distance q.A q.C = 20)
  (E : Point)
  (hE : E = lineIntersection q.A q.C q.B q.D)
  (hAreas : triangleArea q.A E q.D = triangleArea q.B E q.C) :
  distance q.A E = 80 / 13 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3982_398217


namespace NUMINAMATH_CALUDE_cubic_equation_consequence_l3982_398239

theorem cubic_equation_consequence (y : ℝ) (h : y^3 - 3*y = 9) : 
  y^5 - 10*y^2 = -y^2 + 9*y + 27 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_consequence_l3982_398239


namespace NUMINAMATH_CALUDE_tetrahedron_divides_space_l3982_398269

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  faces : Fin 4 → Plane
  edges : Fin 6 → Line
  vertices : Fin 4 → Point

/-- The number of regions formed by the planes of a tetrahedron's faces -/
def num_regions (t : Tetrahedron) : ℕ := 15

/-- Theorem stating that the planes of a tetrahedron's faces divide space into 15 regions -/
theorem tetrahedron_divides_space (t : Tetrahedron) : 
  num_regions t = 15 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_divides_space_l3982_398269


namespace NUMINAMATH_CALUDE_max_apartments_in_complex_l3982_398244

/-- Represents an apartment complex -/
structure ApartmentComplex where
  num_buildings : ℕ
  num_floors : ℕ
  apartments_per_floor : ℕ

/-- The maximum number of apartments in the complex -/
def max_apartments (complex : ApartmentComplex) : ℕ :=
  complex.num_buildings * complex.num_floors * complex.apartments_per_floor

/-- Theorem stating the maximum number of apartments in the given complex -/
theorem max_apartments_in_complex :
  ∃ (complex : ApartmentComplex),
    complex.num_buildings ≤ 22 ∧
    complex.num_buildings > 0 ∧
    complex.num_floors ≤ 6 ∧
    complex.apartments_per_floor = 5 ∧
    max_apartments complex = 660 := by
  sorry

end NUMINAMATH_CALUDE_max_apartments_in_complex_l3982_398244


namespace NUMINAMATH_CALUDE_justin_tim_same_game_l3982_398241

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of players to be selected (excluding Justin and Tim) -/
def players_to_select : ℕ := players_per_game - 2

/-- The number of remaining players after excluding Justin and Tim -/
def remaining_players : ℕ := total_players - 2

/-- The number of times Justin and Tim play in the same game -/
def same_game_count : ℕ := Nat.choose remaining_players players_to_select

theorem justin_tim_same_game :
  same_game_count = 210 := by sorry

end NUMINAMATH_CALUDE_justin_tim_same_game_l3982_398241


namespace NUMINAMATH_CALUDE_walking_speed_l3982_398205

/-- Given a constant walking speed, prove that traveling 30 km in 6 hours results in a speed of 5 kmph -/
theorem walking_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 30) 
    (h2 : time = 6) 
    (h3 : speed = distance / time) : speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_l3982_398205


namespace NUMINAMATH_CALUDE_annulus_area_l3982_398242

/-- An annulus is the region between two concentric circles. -/
structure Annulus where
  b : ℝ  -- radius of the larger circle
  c : ℝ  -- radius of the smaller circle
  h : b > c

/-- The configuration of the annulus with a tangent line. -/
structure AnnulusConfig extends Annulus where
  a : ℝ  -- length of the tangent line XZ
  d : ℝ  -- length of YZ
  e : ℝ  -- length of XY

/-- The area of an annulus is πa², where a is the length of a tangent line
    from a point on the smaller circle to the larger circle. -/
theorem annulus_area (config : AnnulusConfig) : 
  (config.b ^ 2 - config.c ^ 2) * π = config.a ^ 2 * π := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l3982_398242


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3982_398280

theorem profit_percentage_calculation (cost_price selling_price : ℝ) :
  cost_price = 240 →
  selling_price = 288 →
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3982_398280


namespace NUMINAMATH_CALUDE_johns_grass_height_l3982_398287

/-- The height to which John cuts his grass -/
def cut_height : ℝ := 2

/-- The monthly growth rate of the grass in inches -/
def growth_rate : ℝ := 0.5

/-- The maximum height of the grass before cutting in inches -/
def max_height : ℝ := 4

/-- The number of times John cuts his grass per year -/
def cuts_per_year : ℕ := 3

/-- The number of months between each cutting -/
def months_between_cuts : ℕ := 4

theorem johns_grass_height :
  cut_height + growth_rate * months_between_cuts = max_height :=
sorry

end NUMINAMATH_CALUDE_johns_grass_height_l3982_398287


namespace NUMINAMATH_CALUDE_expression_evaluation_l3982_398289

theorem expression_evaluation : 
  let a : ℚ := 7
  let b : ℚ := 11
  let c : ℚ := 13
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3982_398289


namespace NUMINAMATH_CALUDE_triangle_inequality_l3982_398225

theorem triangle_inequality (a b c S r R : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0 ∧ r > 0 ∧ R > 0) :
  (9 * r) / (2 * S) ≤ (1 / a + 1 / b + 1 / c) ∧ (1 / a + 1 / b + 1 / c) ≤ (9 * R) / (4 * S) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3982_398225


namespace NUMINAMATH_CALUDE_gcd_180_450_l3982_398226

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_450_l3982_398226


namespace NUMINAMATH_CALUDE_factorization_problems_l3982_398292

theorem factorization_problems :
  (∀ a : ℝ, 18 * a^2 - 32 = 2 * (3*a + 4) * (3*a - 4)) ∧
  (∀ x y : ℝ, y - 6*x*y + 9*x^2*y = y * (1 - 3*x)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l3982_398292


namespace NUMINAMATH_CALUDE_jelly_bean_ratio_l3982_398206

def napoleon_jelly_beans : ℕ := 17
def mikey_jelly_beans : ℕ := 19

def sedrich_jelly_beans : ℕ := napoleon_jelly_beans + 4

def total_jelly_beans : ℕ := napoleon_jelly_beans + sedrich_jelly_beans

theorem jelly_bean_ratio :
  total_jelly_beans = 2 * mikey_jelly_beans :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_ratio_l3982_398206


namespace NUMINAMATH_CALUDE_polynomial_value_l3982_398299

theorem polynomial_value (a b c d : ℝ) : 
  (∀ x, a * x^5 + b * x^3 + c * x + d = 
    (fun x => a * x^5 + b * x^3 + c * x + d) x) →
  (a * 0^5 + b * 0^3 + c * 0 + d = -5) →
  (a * (-3)^5 + b * (-3)^3 + c * (-3) + d = 7) →
  (a * 3^5 + b * 3^3 + c * 3 + d = -17) := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l3982_398299


namespace NUMINAMATH_CALUDE_data_transmission_time_l3982_398223

theorem data_transmission_time (blocks : Nat) (chunks_per_block : Nat) (transmission_rate : Nat) :
  blocks = 50 →
  chunks_per_block = 1024 →
  transmission_rate = 100 →
  (blocks * chunks_per_block : ℚ) / transmission_rate / 60 = 9 := by
  sorry

end NUMINAMATH_CALUDE_data_transmission_time_l3982_398223


namespace NUMINAMATH_CALUDE_square_value_proof_l3982_398283

theorem square_value_proof : ∃ (square : ℚ), 
  (13.5 / (11 + (2.25 / (1 - square))) - 1 / 7) * (7/6) = 1 ∧ square = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_square_value_proof_l3982_398283


namespace NUMINAMATH_CALUDE_outfit_combinations_l3982_398231

theorem outfit_combinations : 
  let total_items : ℕ := 3  -- shirts, pants, hats
  let colors_per_item : ℕ := 5
  let total_combinations := colors_per_item ^ total_items
  let same_color_combinations := 
    (total_items * colors_per_item * (colors_per_item - 1)) + colors_per_item
  total_combinations - same_color_combinations = 60 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3982_398231


namespace NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l3982_398236

/-- Given an object with acceleration a, initial velocity V₀, 
    final velocity V, and displacement S, prove the time t 
    taken to reach V from V₀ -/
theorem time_from_velocity_and_displacement 
  (a V₀ V S t : ℝ) 
  (hv : V = a * t + V₀) 
  (hs : S = (1/3) * a * t^3 + V₀ * t) :
  t = (V - V₀) / a :=
sorry

end NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l3982_398236


namespace NUMINAMATH_CALUDE_quartic_root_ratio_l3982_398273

theorem quartic_root_ratio (a b c d e : ℝ) (h : ∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = -1 ∨ x = 2 ∨ x = 3) : 
  d / e = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_ratio_l3982_398273


namespace NUMINAMATH_CALUDE_semesters_per_year_l3982_398222

/-- Given the cost per semester and total cost for 13 years of school,
    prove that there are 2 semesters in a year. -/
theorem semesters_per_year :
  let cost_per_semester : ℕ := 20000
  let total_cost : ℕ := 520000
  let years : ℕ := 13
  let total_semesters : ℕ := total_cost / cost_per_semester
  total_semesters / years = 2 := by
  sorry

end NUMINAMATH_CALUDE_semesters_per_year_l3982_398222


namespace NUMINAMATH_CALUDE_monday_sales_correct_l3982_398257

/-- Represents the inventory and sales of hand sanitizer bottles at Danivan Drugstore --/
structure DrugstoreInventory where
  initial_inventory : ℕ
  tuesday_sales : ℕ
  daily_sales_wed_to_sun : ℕ
  saturday_delivery : ℕ
  end_week_inventory : ℕ

/-- Calculates the number of bottles sold on Monday --/
def monday_sales (d : DrugstoreInventory) : ℕ :=
  d.initial_inventory - d.tuesday_sales - (5 * d.daily_sales_wed_to_sun) + d.saturday_delivery - d.end_week_inventory

/-- Theorem stating that the number of bottles sold on Monday is 2445 --/
theorem monday_sales_correct (d : DrugstoreInventory) 
  (h1 : d.initial_inventory = 4500)
  (h2 : d.tuesday_sales = 900)
  (h3 : d.daily_sales_wed_to_sun = 50)
  (h4 : d.saturday_delivery = 650)
  (h5 : d.end_week_inventory = 1555) :
  monday_sales d = 2445 := by
  sorry

end NUMINAMATH_CALUDE_monday_sales_correct_l3982_398257


namespace NUMINAMATH_CALUDE_three_Y_five_l3982_398218

def Y (a b : ℤ) : ℤ := b + 10*a - a^2 - b^2

theorem three_Y_five : Y 3 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_Y_five_l3982_398218


namespace NUMINAMATH_CALUDE_turquoise_survey_result_l3982_398212

/-- Represents the survey about turquoise color perception -/
structure TurquoiseSurvey where
  total : ℕ
  blue : ℕ
  both : ℕ
  neither : ℕ

/-- The number of people who believe turquoise is "green-ish" -/
def green_count (s : TurquoiseSurvey) : ℕ :=
  s.total - (s.blue - s.both) - s.both - s.neither

/-- Theorem stating the result of the survey -/
theorem turquoise_survey_result (s : TurquoiseSurvey) 
  (h1 : s.total = 150)
  (h2 : s.blue = 90)
  (h3 : s.both = 40)
  (h4 : s.neither = 30) :
  green_count s = 70 := by
  sorry

#eval green_count ⟨150, 90, 40, 30⟩

end NUMINAMATH_CALUDE_turquoise_survey_result_l3982_398212


namespace NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l3982_398251

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_60th_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic a)
  (h_first : a 1 = 3)
  (h_fifteenth : a 15 = 31) :
  a 60 = 121 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l3982_398251


namespace NUMINAMATH_CALUDE_new_cards_count_l3982_398293

theorem new_cards_count (cards_per_page : ℕ) (old_cards : ℕ) (total_pages : ℕ) : 
  cards_per_page = 3 → old_cards = 16 → total_pages = 8 → 
  total_pages * cards_per_page - old_cards = 8 := by
  sorry

end NUMINAMATH_CALUDE_new_cards_count_l3982_398293


namespace NUMINAMATH_CALUDE_slide_boys_count_l3982_398255

/-- The number of boys who initially went down the slide -/
def initial_boys : ℕ := 22

/-- The number of additional boys who went down the slide -/
def additional_boys : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_boys : ℕ := initial_boys + additional_boys

theorem slide_boys_count : total_boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_slide_boys_count_l3982_398255


namespace NUMINAMATH_CALUDE_probability_of_selecting_one_each_l3982_398228

/-- The probability of selecting one shirt, one pair of shorts, and one pair of socks
    when randomly choosing three items from a drawer containing 4 shirts, 5 pairs of shorts,
    and 6 pairs of socks. -/
theorem probability_of_selecting_one_each (num_shirts : ℕ) (num_shorts : ℕ) (num_socks : ℕ) :
  num_shirts = 4 →
  num_shorts = 5 →
  num_socks = 6 →
  (num_shirts * num_shorts * num_socks : ℚ) / (Nat.choose (num_shirts + num_shorts + num_socks) 3) = 24 / 91 := by
  sorry

#check probability_of_selecting_one_each

end NUMINAMATH_CALUDE_probability_of_selecting_one_each_l3982_398228


namespace NUMINAMATH_CALUDE_dance_pairs_correct_l3982_398207

/-- The number of ways to form dance pairs given specific knowledge constraints -/
def dance_pairs (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then
    Nat.choose n r * (Nat.factorial n / Nat.factorial (n - r))
  else 0

/-- Theorem stating the correct number of dance pairs -/
theorem dance_pairs_correct (n : ℕ) (r : ℕ) (h : r ≤ n) :
  dance_pairs n r = Nat.choose n r * (Nat.factorial n / Nat.factorial (n - r)) :=
by sorry

end NUMINAMATH_CALUDE_dance_pairs_correct_l3982_398207


namespace NUMINAMATH_CALUDE_dividend_calculation_l3982_398235

theorem dividend_calculation (quotient divisor : ℝ) (h1 : quotient = 0.0012000000000000001) (h2 : divisor = 17) :
  quotient * divisor = 0.0204000000000000027 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3982_398235


namespace NUMINAMATH_CALUDE_carlos_gummy_worms_l3982_398271

/-- The number of gummy worms remaining after eating half for a given number of days -/
def gummy_worms_remaining (initial : ℕ) (days : ℕ) : ℕ :=
  initial / (2 ^ days)

/-- Theorem stating that Carlos has 4 gummy worms left after 4 days -/
theorem carlos_gummy_worms :
  gummy_worms_remaining 64 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_carlos_gummy_worms_l3982_398271


namespace NUMINAMATH_CALUDE_special_function_value_l3982_398291

/-- A function satisfying f(xy) = f(x)/y for all positive real numbers x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y

theorem special_function_value 
  (f : ℝ → ℝ) 
  (h1 : special_function f) 
  (h2 : f 45 = 15) : 
  f 60 = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l3982_398291


namespace NUMINAMATH_CALUDE_polygon_sides_l3982_398268

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1260 → ∃ n : ℕ, n = 9 ∧ sum_interior_angles = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3982_398268


namespace NUMINAMATH_CALUDE_sin_cube_identity_l3982_398266

theorem sin_cube_identity (θ : ℝ) : 
  Real.sin θ ^ 3 = (-1/4 : ℝ) * Real.sin (3 * θ) + (3/4 : ℝ) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l3982_398266


namespace NUMINAMATH_CALUDE_inequality_range_l3982_398211

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2*x + 1| - |2*x - 1| < a) → a > 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3982_398211


namespace NUMINAMATH_CALUDE_rancher_problem_solution_l3982_398220

/-- Represents the rancher's cattle problem -/
structure CattleProblem where
  initial_cattle : ℕ
  dead_cattle : ℕ
  price_reduction : ℚ
  loss_amount : ℚ

/-- Calculates the original price per head of cattle -/
def original_price (p : CattleProblem) : ℚ :=
  p.loss_amount / (p.initial_cattle - p.dead_cattle : ℚ)

/-- Calculates the total amount the rancher would have made -/
def total_amount (p : CattleProblem) : ℚ :=
  (p.initial_cattle : ℚ) * original_price p

/-- Theorem stating the solution to the rancher's problem -/
theorem rancher_problem_solution (p : CattleProblem) 
  (h1 : p.initial_cattle = 340)
  (h2 : p.dead_cattle = 172)
  (h3 : p.price_reduction = 150)
  (h4 : p.loss_amount = 25200) :
  total_amount p = 49813.40 := by
  sorry

end NUMINAMATH_CALUDE_rancher_problem_solution_l3982_398220


namespace NUMINAMATH_CALUDE_max_third_term_arithmetic_sequence_greatest_third_term_l3982_398295

theorem max_third_term_arithmetic_sequence (a d : ℕ) (h1 : 0 < a) (h2 : 0 < d) 
  (h3 : a + (a + d) + (a + 2*d) + (a + 3*d) = 52) : 
  ∀ (x y : ℕ), 0 < x → 0 < y → 
  x + (x + y) + (x + 2*y) + (x + 3*y) = 52 → 
  x + 2*y ≤ a + 2*d := by
sorry

theorem greatest_third_term : 
  ∃ (a d : ℕ), 0 < a ∧ 0 < d ∧ 
  a + (a + d) + (a + 2*d) + (a + 3*d) = 52 ∧
  a + 2*d = 17 ∧
  (∀ (x y : ℕ), 0 < x → 0 < y → 
   x + (x + y) + (x + 2*y) + (x + 3*y) = 52 → 
   x + 2*y ≤ 17) := by
sorry

end NUMINAMATH_CALUDE_max_third_term_arithmetic_sequence_greatest_third_term_l3982_398295


namespace NUMINAMATH_CALUDE_negation_equivalence_l3982_398202

/-- A number is even if it's divisible by 2 -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The negation of "There is an even number that is a prime number" is equivalent to "No even number is a prime number" -/
theorem negation_equivalence : 
  (¬ ∃ n : ℕ, IsEven n ∧ IsPrime n) ↔ (∀ n : ℕ, IsEven n → ¬ IsPrime n) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3982_398202


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3982_398261

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3982_398261


namespace NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l3982_398214

/-- Given a mixture of water and alcohol, calculate the new alcohol percentage after adding water. -/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 15)
  (h2 : initial_percentage = 20)
  (h3 : added_water = 5)
  : (initial_volume * initial_percentage / 100) / (initial_volume + added_water) * 100 = 15 := by
  sorry

#check alcohol_percentage_after_dilution

end NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l3982_398214


namespace NUMINAMATH_CALUDE_students_only_english_l3982_398233

theorem students_only_english (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 32)
  (h2 : both = 12)
  (h3 : german = 22)
  (h4 : total = (german - both) + both + (total - german)) :
  total - german = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_only_english_l3982_398233


namespace NUMINAMATH_CALUDE_infinitely_many_primes_with_large_primitive_root_l3982_398270

theorem infinitely_many_primes_with_large_primitive_root (n : ℕ) (hn : n > 0) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ p ∈ S,
    Nat.Prime p ∧ ∀ m ∈ Finset.range n, ∃ x, x^2 ≡ m [MOD p] :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_with_large_primitive_root_l3982_398270


namespace NUMINAMATH_CALUDE_chantel_bracelet_count_l3982_398297

/-- The number of bracelets Chantel has at the end of the process --/
def final_bracelet_count (initial_daily_production : ℕ) (initial_days : ℕ) 
  (first_giveaway : ℕ) (second_daily_production : ℕ) (second_days : ℕ) 
  (second_giveaway : ℕ) : ℕ :=
  initial_daily_production * initial_days - first_giveaway + 
  second_daily_production * second_days - second_giveaway

/-- Theorem stating that Chantel ends up with 13 bracelets --/
theorem chantel_bracelet_count : 
  final_bracelet_count 2 5 3 3 4 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_chantel_bracelet_count_l3982_398297


namespace NUMINAMATH_CALUDE_no_common_root_with_specific_values_l3982_398248

theorem no_common_root_with_specific_values : ¬ ∃ (P₁ P₂ : ℤ → ℤ) (a b : ℤ),
  (∀ x, ∃ (c : ℤ), P₁ x = c) ∧  -- P₁ has integer coefficients
  (∀ x, ∃ (c : ℤ), P₂ x = c) ∧  -- P₂ has integer coefficients
  a < 0 ∧                       -- a is strictly negative
  P₁ a = 0 ∧                    -- a is a root of P₁
  P₂ a = 0 ∧                    -- a is a root of P₂
  b > 0 ∧                       -- b is positive
  P₁ b = 2007 ∧                 -- P₁ evaluates to 2007 at b
  P₂ b = 2008                   -- P₂ evaluates to 2008 at b
  := by sorry

end NUMINAMATH_CALUDE_no_common_root_with_specific_values_l3982_398248


namespace NUMINAMATH_CALUDE_soccer_players_count_l3982_398250

def total_students : ℕ := 400
def sports_proportion : ℚ := 52 / 100
def soccer_proportion : ℚ := 125 / 1000

theorem soccer_players_count :
  ⌊(total_students : ℚ) * sports_proportion * soccer_proportion⌋ = 26 := by
  sorry

end NUMINAMATH_CALUDE_soccer_players_count_l3982_398250


namespace NUMINAMATH_CALUDE_function_inequality_l3982_398281

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, HasDerivAt f (f' x) x) 
  (h' : ∀ x, f' x > f x) : f (Real.log 2022) > 2022 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3982_398281


namespace NUMINAMATH_CALUDE_initial_selling_price_theorem_l3982_398210

/-- The number of articles sold at a gain -/
def articles_sold_gain : ℝ := 20

/-- The gain percentage -/
def gain_percentage : ℝ := 0.20

/-- The number of articles that would be sold at a loss -/
def articles_sold_loss : ℝ := 29.99999625000047

/-- The loss percentage -/
def loss_percentage : ℝ := 0.20

/-- Theorem stating that the initial selling price for articles sold at a gain
    is 24 times the cost price of one article -/
theorem initial_selling_price_theorem (cost_price : ℝ) :
  let selling_price_gain := cost_price * (1 + gain_percentage)
  let selling_price_loss := cost_price * (1 - loss_percentage)
  articles_sold_gain * selling_price_gain = articles_sold_loss * selling_price_loss →
  articles_sold_gain * selling_price_gain = 24 * cost_price :=
by sorry

end NUMINAMATH_CALUDE_initial_selling_price_theorem_l3982_398210


namespace NUMINAMATH_CALUDE_smallest_perimeter_l3982_398254

/- Define the triangle PQR -/
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  isosceles : dist P Q = dist P R

/- Define the intersection point J -/
def J (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

/- Define the perimeter of the triangle -/
def perimeter (P Q R : ℝ × ℝ) : ℝ :=
  dist P Q + dist Q R + dist R P

/- Main theorem -/
theorem smallest_perimeter (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  dist Q (J P Q R) = 10 →
  ∃ (P' Q' R' : ℝ × ℝ), Triangle P' Q' R' ∧
    dist Q' (J P' Q' R') = 10 ∧
    perimeter P' Q' R' = 198 ∧
    ∀ (P'' Q'' R'' : ℝ × ℝ), Triangle P'' Q'' R'' →
      dist Q'' (J P'' Q'' R'') = 10 →
      perimeter P'' Q'' R'' ≥ 198 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l3982_398254


namespace NUMINAMATH_CALUDE_sam_gave_thirteen_cards_l3982_398204

/-- The number of baseball cards Sam gave to Mike -/
def cards_from_sam (initial_cards final_cards : ℕ) : ℕ :=
  final_cards - initial_cards

theorem sam_gave_thirteen_cards :
  let initial_cards : ℕ := 87
  let final_cards : ℕ := 100
  cards_from_sam initial_cards final_cards = 13 := by
  sorry

end NUMINAMATH_CALUDE_sam_gave_thirteen_cards_l3982_398204


namespace NUMINAMATH_CALUDE_prime_power_theorem_l3982_398246

theorem prime_power_theorem (p q : ℕ) : 
  p > 1 → q > 1 → 
  Nat.Prime p → Nat.Prime q → 
  Nat.Prime (7 * p + q) → Nat.Prime (p * q + 11) → 
  p^q = 8 ∨ p^q = 9 := by
sorry

end NUMINAMATH_CALUDE_prime_power_theorem_l3982_398246


namespace NUMINAMATH_CALUDE_locus_definition_correct_l3982_398275

-- Define the space we're working in (e.g., a metric space)
variable {X : Type*} [MetricSpace X]

-- Define the locus and the distance
variable (P : X) (r : ℝ) (locus : Set X)

-- Define the condition for a point to be at distance r from P
def atDistanceR (x : X) := dist x P = r

-- State the theorem
theorem locus_definition_correct :
  (∀ x : X, atDistanceR P r x → x ∈ locus) ∧
  (∀ x : X, x ∈ locus → atDistanceR P r x) :=
sorry

end NUMINAMATH_CALUDE_locus_definition_correct_l3982_398275


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3982_398237

theorem sum_of_squared_coefficients : 
  let expression := fun x : ℝ => 5 * (x^3 - x) - 3 * (x^2 - 4*x + 3)
  let simplified := fun x : ℝ => 5*x^3 - 3*x^2 + 7*x - 9
  (∀ x : ℝ, expression x = simplified x) →
  (5^2 + (-3)^2 + 7^2 + (-9)^2 = 164) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3982_398237


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l3982_398243

/-- Given two vectors a and b in R², where a = (2,1) and b = (k,3),
    if a + 2b is parallel to 2a - b, then k = 6 -/
theorem vector_parallel_condition (k : ℝ) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![k, 3]
  (∃ (t : ℝ), t ≠ 0 ∧ (a + 2 • b) = t • (2 • a - b)) →
  k = 6 :=
by sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l3982_398243


namespace NUMINAMATH_CALUDE_football_season_games_l3982_398238

/-- Represents a football team's season statistics -/
structure SeasonStats where
  totalGames : ℕ
  tieGames : ℕ
  firstHundredWins : ℕ
  remainingWins : ℕ
  maxConsecutiveLosses : ℕ
  minConsecutiveWins : ℕ

/-- Calculates the total wins for a season -/
def totalWins (stats : SeasonStats) : ℕ :=
  stats.firstHundredWins + stats.remainingWins

/-- Calculates the win percentage for a season, excluding tie games -/
def winPercentage (stats : SeasonStats) : ℚ :=
  (totalWins stats : ℚ) / ((stats.totalGames - stats.tieGames) : ℚ)

/-- Theorem stating the total number of games played in the season -/
theorem football_season_games (stats : SeasonStats) 
  (h1 : 150 ≤ stats.totalGames ∧ stats.totalGames ≤ 200)
  (h2 : stats.firstHundredWins = 63)
  (h3 : stats.remainingWins = (stats.totalGames - 100) * 48 / 100)
  (h4 : stats.tieGames = 5)
  (h5 : winPercentage stats = 58 / 100)
  (h6 : stats.minConsecutiveWins ≥ 20)
  (h7 : stats.maxConsecutiveLosses ≤ 10) :
  stats.totalGames = 179 := by
  sorry


end NUMINAMATH_CALUDE_football_season_games_l3982_398238


namespace NUMINAMATH_CALUDE_trapezium_height_l3982_398201

theorem trapezium_height (a b area : ℝ) (ha : a = 20) (hb : b = 16) (harea : area = 270) :
  (2 * area) / (a + b) = 15 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l3982_398201


namespace NUMINAMATH_CALUDE_sqrt_function_theorem_linear_function_theorem_l3982_398215

-- Problem 1
theorem sqrt_function_theorem (f : ℝ → ℝ) :
  (∀ x ≥ 0, f (Real.sqrt x) = x - 1) →
  (∀ x ≥ 0, f x = x^2 - 1) :=
by sorry

-- Problem 2
theorem linear_function_theorem (f : ℝ → ℝ) :
  (∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b) →
  (∀ x, f (f x) = f x + 2) →
  (∀ x, f x = x + 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_function_theorem_linear_function_theorem_l3982_398215


namespace NUMINAMATH_CALUDE_parallelepiped_surface_area_l3982_398288

/-- Represents a parallelepiped composed of white and black unit cubes -/
structure Parallelepiped where
  white_cubes : ℕ
  black_cubes : ℕ
  length : ℕ
  width : ℕ
  height : ℕ

/-- Conditions for the parallelepiped -/
def valid_parallelepiped (p : Parallelepiped) : Prop :=
  p.white_cubes > 0 ∧
  p.black_cubes = p.white_cubes * 53 / 52 ∧
  p.length > 1 ∧ p.width > 1 ∧ p.height > 1 ∧
  p.length * p.width * p.height = p.white_cubes + p.black_cubes

/-- Surface area of a parallelepiped -/
def surface_area (p : Parallelepiped) : ℕ :=
  2 * (p.length * p.width + p.width * p.height + p.height * p.length)

/-- Theorem stating the surface area of the parallelepiped is 142 -/
theorem parallelepiped_surface_area (p : Parallelepiped) 
  (h : valid_parallelepiped p) : surface_area p = 142 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_surface_area_l3982_398288


namespace NUMINAMATH_CALUDE_school_picnic_volunteers_l3982_398256

theorem school_picnic_volunteers (total_parents supervise_parents refreshment_parents : ℕ) 
  (h1 : total_parents = 84)
  (h2 : supervise_parents = 25)
  (h3 : refreshment_parents = 42)
  (h4 : refreshment_parents = (3/2 : ℚ) * (total_parents - supervise_parents - refreshment_parents + both_parents)) :
  ∃ both_parents : ℕ, both_parents = 11 := by
  sorry

end NUMINAMATH_CALUDE_school_picnic_volunteers_l3982_398256


namespace NUMINAMATH_CALUDE_problem_solution_l3982_398240

theorem problem_solution (x y : ℝ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : 1/x + 1/y = 1) 
  (h4 : x * y = 9) : 
  y = (9 + 3 * Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3982_398240


namespace NUMINAMATH_CALUDE_current_age_ratio_l3982_398216

def age_ratio (p q : ℕ) : ℚ := p / q

theorem current_age_ratio :
  ∀ (p q : ℕ),
  (∃ k : ℕ, p = k * q) →
  (p + 11 = 2 * (q + 11)) →
  (p = 30 + 3) →
  age_ratio p q = 3 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_current_age_ratio_l3982_398216


namespace NUMINAMATH_CALUDE_weight_of_eight_moles_l3982_398260

/-- The total weight of a given number of moles of a compound -/
def total_weight (molecular_weight : ℝ) (moles : ℝ) : ℝ :=
  molecular_weight * moles

/-- Proof that 8 moles of a compound with molecular weight 496 g/mol has a total weight of 3968 g -/
theorem weight_of_eight_moles :
  let molecular_weight : ℝ := 496
  let moles : ℝ := 8
  total_weight molecular_weight moles = 3968 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_eight_moles_l3982_398260
