import Mathlib

namespace NUMINAMATH_CALUDE_even_function_inequality_l2141_214181

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is monotonic on (-∞, 0] if it's either
    nondecreasing or nonincreasing on that interval -/
def IsMonotonicOnNegative (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y) ∨ (∀ x y, x ≤ y ∧ y ≤ 0 → f y ≤ f x)

theorem even_function_inequality (f : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_monotonic : IsMonotonicOnNegative f)
  (h_inequality : f (-2) < f 1) :
  f 5 < f (-3) ∧ f (-3) < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l2141_214181


namespace NUMINAMATH_CALUDE_sock_pair_count_l2141_214179

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_sock_pairs (black white blue : ℕ) : ℕ :=
  black * white + black * blue + white * blue

/-- Theorem: There are 107 ways to choose a pair of socks of different colors
    from a drawer containing 5 black socks, 6 white socks, and 7 blue socks -/
theorem sock_pair_count :
  different_color_sock_pairs 5 6 7 = 107 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l2141_214179


namespace NUMINAMATH_CALUDE_binary_rep_of_31_l2141_214183

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Theorem: The binary representation of 31 is [true, true, true, true, true] -/
theorem binary_rep_of_31 : toBinary 31 = [true, true, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_rep_of_31_l2141_214183


namespace NUMINAMATH_CALUDE_banana_mango_equivalence_l2141_214199

/-- Represents the cost relationship between fruits -/
structure FruitCost where
  banana : ℝ
  pear : ℝ
  mango : ℝ

/-- The given cost relationships -/
def cost_relation (c : FruitCost) : Prop :=
  4 * c.banana = 3 * c.pear ∧ 8 * c.pear = 5 * c.mango

/-- The theorem to prove -/
theorem banana_mango_equivalence (c : FruitCost) (h : cost_relation c) :
  20 * c.banana = 9.375 * c.mango :=
sorry

end NUMINAMATH_CALUDE_banana_mango_equivalence_l2141_214199


namespace NUMINAMATH_CALUDE_num_lists_15_4_l2141_214130

/-- The number of elements in the set to draw from -/
def n : ℕ := 15

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from a set of n elements -/
def num_lists (n k : ℕ) : ℕ := n^k

/-- Theorem: The number of possible lists when drawing 4 times with replacement from a set of 15 elements is 50625 -/
theorem num_lists_15_4 : num_lists n k = 50625 := by
  sorry

end NUMINAMATH_CALUDE_num_lists_15_4_l2141_214130


namespace NUMINAMATH_CALUDE_perpendicular_distance_is_six_l2141_214178

/-- A rectangular parallelepiped with dimensions 6 × 5 × 4 -/
structure Parallelepiped where
  length : ℝ := 6
  width : ℝ := 5
  height : ℝ := 4

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The perpendicular distance from a point to a plane -/
def perpendicularDistance (S : Point3D) (P Q R : Point3D) : ℝ := sorry

theorem perpendicular_distance_is_six :
  let p : Parallelepiped := { }
  let S : Point3D := ⟨6, 0, 0⟩
  let P : Point3D := ⟨0, 0, 0⟩
  let Q : Point3D := ⟨0, 5, 0⟩
  let R : Point3D := ⟨0, 0, 4⟩
  perpendicularDistance S P Q R = 6 := by sorry

end NUMINAMATH_CALUDE_perpendicular_distance_is_six_l2141_214178


namespace NUMINAMATH_CALUDE_dealership_sales_prediction_l2141_214104

/-- The number of sports cars predicted to be sold -/
def sports_cars : ℕ := 45

/-- The ratio of sports cars to sedans -/
def ratio : ℚ := 3 / 5

/-- The minimum difference between sedans and sports cars -/
def min_difference : ℕ := 20

/-- The number of sedans expected to be sold -/
def sedans : ℕ := 75

theorem dealership_sales_prediction :
  (sedans : ℚ) = sports_cars / ratio ∧ 
  sedans ≥ sports_cars + min_difference := by
  sorry

end NUMINAMATH_CALUDE_dealership_sales_prediction_l2141_214104


namespace NUMINAMATH_CALUDE_handshake_arrangement_count_l2141_214113

/-- A handshake arrangement for a group of people -/
structure HandshakeArrangement (n : ℕ) :=
  (shakes : Fin n → Finset (Fin n))
  (shake_count : ∀ i, (shakes i).card = 3)
  (symmetry : ∀ i j, i ∈ shakes j ↔ j ∈ shakes i)

/-- The number of distinct handshake arrangements for 12 people -/
def M : ℕ := sorry

/-- The main theorem: M is congruent to 850 modulo 1000 -/
theorem handshake_arrangement_count :
  M ≡ 850 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_handshake_arrangement_count_l2141_214113


namespace NUMINAMATH_CALUDE_average_of_four_digits_l2141_214167

theorem average_of_four_digits 
  (total_digits : Nat)
  (total_average : ℚ)
  (five_digits : Nat)
  (five_average : ℚ)
  (h1 : total_digits = 9)
  (h2 : total_average = 18)
  (h3 : five_digits = 5)
  (h4 : five_average = 26)
  : (total_digits * total_average - five_digits * five_average) / (total_digits - five_digits) = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_digits_l2141_214167


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2141_214150

/-- A geometric sequence with positive terms satisfying a certain relation -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ q : ℝ, q > 1 ∧ ∀ n, a (n + 1) = q * a n) ∧
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))

/-- The general term of the geometric sequence -/
def GeneralTerm (a : ℕ → ℝ) : Prop :=
  ∃ a₁ : ℝ, ∀ n, a n = a₁ * 2^(n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  GeometricSequence a → GeneralTerm a := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2141_214150


namespace NUMINAMATH_CALUDE_cost_per_sqm_intersecting_roads_l2141_214151

/-- The cost per square meter for traveling two intersecting roads on a rectangular lawn. -/
theorem cost_per_sqm_intersecting_roads 
  (lawn_length : ℝ) 
  (lawn_width : ℝ) 
  (road_width : ℝ) 
  (total_cost : ℝ) : 
  lawn_length = 80 ∧ 
  lawn_width = 40 ∧ 
  road_width = 10 ∧ 
  total_cost = 3300 → 
  (total_cost / ((lawn_length * road_width + lawn_width * road_width) - road_width * road_width)) = 3 := by
sorry

end NUMINAMATH_CALUDE_cost_per_sqm_intersecting_roads_l2141_214151


namespace NUMINAMATH_CALUDE_interior_angle_sum_l2141_214164

theorem interior_angle_sum (n : ℕ) : 
  (180 * (n - 2) = 2160) → (180 * ((n + 3) - 2) = 2700) := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_l2141_214164


namespace NUMINAMATH_CALUDE_pizza_slices_remaining_l2141_214138

/-- The number of remaining pizza slices after eating some -/
def remaining_slices (slices_per_pizza : ℕ) (pizzas_ordered : ℕ) (slices_eaten : ℕ) : ℕ :=
  slices_per_pizza * pizzas_ordered - slices_eaten

/-- Theorem: Given the conditions, the number of remaining slices is 9 -/
theorem pizza_slices_remaining :
  remaining_slices 8 2 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_remaining_l2141_214138


namespace NUMINAMATH_CALUDE_essay_word_count_excess_l2141_214184

theorem essay_word_count_excess (word_limit : ℕ) (saturday_words : ℕ) (sunday_words : ℕ) :
  word_limit = 1000 →
  saturday_words = 450 →
  sunday_words = 650 →
  (saturday_words + sunday_words) - word_limit = 100 := by
  sorry

end NUMINAMATH_CALUDE_essay_word_count_excess_l2141_214184


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2141_214108

theorem sum_of_cubes_of_roots (p q r : ℝ) : 
  p^3 - 2*p^2 + 3*p - 4 = 0 →
  q^3 - 2*q^2 + 3*q - 4 = 0 →
  r^3 - 2*r^2 + 3*r - 4 = 0 →
  p^3 + q^3 + r^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2141_214108


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2141_214126

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point on the ellipse
def point_on_ellipse (p : ℝ × ℝ) : Prop :=
  ellipse p.1 p.2

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (h₁ : point_on_ellipse A) 
  (h₂ : point_on_ellipse B) 
  (h₃ : collinear A B F₂) :
  distance F₁ A + distance F₁ B + distance A B = 20 := 
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2141_214126


namespace NUMINAMATH_CALUDE_no_solution_to_fractional_equation_l2141_214125

theorem no_solution_to_fractional_equation :
  ¬ ∃ (x : ℝ), x ≠ 3 ∧ (2 - x) / (x - 3) = 1 + 1 / (3 - x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_fractional_equation_l2141_214125


namespace NUMINAMATH_CALUDE_valid_input_statement_l2141_214135

/-- Represents a programming language construct --/
inductive ProgramConstruct
| Input : String → String → ProgramConstruct
| Other : ProgramConstruct

/-- Checks if a given ProgramConstruct is a valid INPUT statement --/
def isValidInputStatement (stmt : ProgramConstruct) : Prop :=
  match stmt with
  | ProgramConstruct.Input prompt var => true
  | _ => false

/-- Theorem: An INPUT statement with a prompt and variable is valid --/
theorem valid_input_statement (prompt var : String) :
  isValidInputStatement (ProgramConstruct.Input prompt var) := by
  sorry

#check valid_input_statement

end NUMINAMATH_CALUDE_valid_input_statement_l2141_214135


namespace NUMINAMATH_CALUDE_quadratic_equation_necessary_not_sufficient_l2141_214191

theorem quadratic_equation_necessary_not_sufficient :
  ∀ x : ℝ, 
    (x = 5 → x^2 - 4*x - 5 = 0) ∧ 
    ¬(x^2 - 4*x - 5 = 0 → x = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_necessary_not_sufficient_l2141_214191


namespace NUMINAMATH_CALUDE_k_h_symmetry_l2141_214106

-- Define the function h
def h (x : ℝ) : ℝ := 4 * x^2 - 12

-- Define a variable k as a function from ℝ to ℝ
variable (k : ℝ → ℝ)

-- State the theorem
theorem k_h_symmetry (h_def : ∀ x, h x = 4 * x^2 - 12) 
                     (k_h_3 : k (h 3) = 16) : 
  k (h (-3)) = 16 := by
  sorry


end NUMINAMATH_CALUDE_k_h_symmetry_l2141_214106


namespace NUMINAMATH_CALUDE_range_of_m_l2141_214155

/-- The proposition p: x^2 - 7x + 10 ≤ 0 -/
def p (x : ℝ) : Prop := x^2 - 7*x + 10 ≤ 0

/-- The proposition q: m ≤ x ≤ m + 1 -/
def q (m x : ℝ) : Prop := m ≤ x ∧ x ≤ m + 1

/-- q is a sufficient condition for p -/
def q_sufficient_for_p (m : ℝ) : Prop := ∀ x, q m x → p x

theorem range_of_m (m : ℝ) : 
  q_sufficient_for_p m → 2 ≤ m ∧ m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2141_214155


namespace NUMINAMATH_CALUDE_fifth_sphere_radius_l2141_214172

/-- Represents a cone with height and base radius 7 -/
structure Cone :=
  (height : ℝ := 7)
  (base_radius : ℝ := 7)

/-- Represents a sphere with a center and radius -/
structure Sphere :=
  (center : ℝ × ℝ × ℝ)
  (radius : ℝ)

/-- Represents the configuration of spheres in the cone -/
structure SphereConfiguration :=
  (cone : Cone)
  (base_spheres : Fin 4 → Sphere)
  (top_sphere : Sphere)

/-- Checks if two spheres are externally touching -/
def externally_touching (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 = (s1.radius + s2.radius)^2

/-- Checks if a sphere touches the lateral surface of the cone -/
def touches_lateral_surface (s : Sphere) (c : Cone) : Prop :=
  sorry -- Definition omitted for brevity

/-- Checks if a sphere touches the base of the cone -/
def touches_base (s : Sphere) (c : Cone) : Prop :=
  sorry -- Definition omitted for brevity

/-- Theorem stating the radius of the fifth sphere -/
theorem fifth_sphere_radius (config : SphereConfiguration) :
  (∀ i j : Fin 4, i ≠ j → externally_touching (config.base_spheres i) (config.base_spheres j)) →
  (∀ i : Fin 4, touches_lateral_surface (config.base_spheres i) config.cone) →
  (∀ i : Fin 4, touches_base (config.base_spheres i) config.cone) →
  (∀ i : Fin 4, externally_touching (config.base_spheres i) config.top_sphere) →
  touches_lateral_surface config.top_sphere config.cone →
  config.top_sphere.radius = 2 * Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_fifth_sphere_radius_l2141_214172


namespace NUMINAMATH_CALUDE_exist_consecutive_lucky_years_l2141_214190

/-- Returns the first two digits of a four-digit number -/
def firstTwoDigits (n : ℕ) : ℕ := n / 100

/-- Returns the last two digits of a four-digit number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- Checks if a year is lucky -/
def isLuckyYear (year : ℕ) : Prop :=
  year % (firstTwoDigits year + lastTwoDigits year) = 0

/-- Theorem: There exist two consecutive lucky years -/
theorem exist_consecutive_lucky_years :
  ∃ (y : ℕ), 1000 ≤ y ∧ y < 9999 ∧ isLuckyYear y ∧ isLuckyYear (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_exist_consecutive_lucky_years_l2141_214190


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l2141_214124

theorem polynomial_root_problem (a b c d : ℤ) : 
  a * (3 + Complex.I)^4 + b * (3 + Complex.I)^3 + c * (3 + Complex.I)^2 + d * (3 + Complex.I) + a = 0 →
  Int.gcd (Int.gcd (Int.gcd a b) c) d = 1 →
  d.natAbs = 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l2141_214124


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l2141_214128

theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 8) (h3 : EF = 9) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let diameter := 2 * area / s
  diameter = 4 * Real.sqrt 35 / 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l2141_214128


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2141_214110

theorem quadratic_inequality_solution (x : ℝ) :
  -4 * x^2 + 7 * x + 2 < 0 ↔ x < -1/4 ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2141_214110


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l2141_214198

theorem stratified_sampling_problem (high_school_students : ℕ) (middle_school_students : ℕ)
  (middle_school_sample : ℕ) (total_sample : ℕ) :
  high_school_students = 3500 →
  middle_school_students = 1500 →
  middle_school_sample = 30 →
  (middle_school_students : ℚ) / (high_school_students + middle_school_students : ℚ) * middle_school_sample = total_sample →
  total_sample = 100 := by
sorry


end NUMINAMATH_CALUDE_stratified_sampling_problem_l2141_214198


namespace NUMINAMATH_CALUDE_train_length_l2141_214174

/-- Given a train traveling at a certain speed that crosses a bridge of known length in a specific time,
    this theorem proves the length of the train. -/
theorem train_length
  (train_speed : ℝ)
  (bridge_length : ℝ)
  (crossing_time : ℝ)
  (h1 : train_speed = 45)  -- km/hr
  (h2 : bridge_length = 235)  -- meters
  (h3 : crossing_time = 30)  -- seconds
  : ∃ (train_length : ℝ), train_length = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2141_214174


namespace NUMINAMATH_CALUDE_final_amount_is_47500_l2141_214102

def income : ℝ := 200000

def children_share : ℝ := 0.15
def num_children : ℕ := 3
def wife_share : ℝ := 0.30
def donation_rate : ℝ := 0.05

def final_amount : ℝ :=
  let children_total := children_share * num_children * income
  let wife_total := wife_share * income
  let remaining_after_family := income - children_total - wife_total
  let donation := donation_rate * remaining_after_family
  remaining_after_family - donation

theorem final_amount_is_47500 :
  final_amount = 47500 := by sorry

end NUMINAMATH_CALUDE_final_amount_is_47500_l2141_214102


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2141_214159

theorem solution_set_of_inequality (x : ℝ) :
  (((2 * x - 1) / (x + 2)) > 1) ↔ (x < -2 ∨ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2141_214159


namespace NUMINAMATH_CALUDE_sequence_nature_l2141_214116

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem sequence_nature (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n^2) →
  (∀ n, a n = S n - S (n-1)) →
  arithmetic_sequence a 2 :=
sorry

end NUMINAMATH_CALUDE_sequence_nature_l2141_214116


namespace NUMINAMATH_CALUDE_pie_shop_revenue_l2141_214189

/-- The revenue calculation for a pie shop --/
theorem pie_shop_revenue : 
  (price_per_slice : ℕ) → 
  (slices_per_pie : ℕ) → 
  (number_of_pies : ℕ) → 
  price_per_slice = 5 →
  slices_per_pie = 4 →
  number_of_pies = 9 →
  price_per_slice * slices_per_pie * number_of_pies = 180 := by
  sorry

end NUMINAMATH_CALUDE_pie_shop_revenue_l2141_214189


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2141_214131

theorem fraction_sum_equality : (3 / 10 : ℚ) + (2 / 100 : ℚ) + (8 / 1000 : ℚ) + (8 / 10000 : ℚ) = 0.3288 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2141_214131


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2141_214177

def set_A : Set ℝ := {x | x < -1 ∨ x > 3}
def set_B : Set ℝ := {x | x - 2 ≥ 0}

theorem union_of_A_and_B : set_A ∪ set_B = {x | x < -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2141_214177


namespace NUMINAMATH_CALUDE_negation_equivalence_l2141_214148

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2141_214148


namespace NUMINAMATH_CALUDE_x_squared_coefficient_l2141_214109

/-- The coefficient of x^2 in the expansion of (x^2+x+1)(1-x)^6 is 10 -/
theorem x_squared_coefficient : Int := by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_l2141_214109


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l2141_214137

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- Theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  let initialHeight : ℝ := 80
  let reboundFactor : ℝ := 3/4
  let bounces : ℕ := 4
  totalDistance initialHeight reboundFactor bounces = 357.5 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l2141_214137


namespace NUMINAMATH_CALUDE_cooking_cleaning_combinations_l2141_214129

-- Define the number of friends
def total_friends : ℕ := 5

-- Define the number of cooks
def num_cooks : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem cooking_cleaning_combinations :
  combination total_friends num_cooks = 10 := by
  sorry

end NUMINAMATH_CALUDE_cooking_cleaning_combinations_l2141_214129


namespace NUMINAMATH_CALUDE_rectangle_breadth_l2141_214195

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) :
  square_area = 3600 →
  rectangle_area = 240 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  rectangle_area = rectangle_length * (rectangle_area / rectangle_length) →
  rectangle_area / rectangle_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l2141_214195


namespace NUMINAMATH_CALUDE_pythagorean_triple_parity_l2141_214171

theorem pythagorean_triple_parity (x y z : ℤ) (h : x^2 + y^2 = z^2) :
  Even x ∨ Even y := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_parity_l2141_214171


namespace NUMINAMATH_CALUDE_deposit_calculation_l2141_214187

theorem deposit_calculation (total_price : ℝ) (deposit_percentage : ℝ) (remaining_amount : ℝ) : 
  deposit_percentage = 0.1 →
  remaining_amount = 945 →
  total_price * (1 - deposit_percentage) = remaining_amount →
  total_price * deposit_percentage = 105 := by
sorry

end NUMINAMATH_CALUDE_deposit_calculation_l2141_214187


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l2141_214146

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 12*n + 28 ≤ 0 ∧ 
  ∀ (m : ℤ), m^2 - 12*m + 28 ≤ 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l2141_214146


namespace NUMINAMATH_CALUDE_rook_placement_modulo_four_l2141_214112

/-- The color of a cell on the board -/
def cellColor (n i j : ℕ) : ℕ := min (i + j - 1) (2 * n - i - j + 1)

/-- A valid rook placement function -/
def IsValidRookPlacement (n : ℕ) (f : ℕ → ℕ) : Prop :=
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  (∀ i, i ∈ Finset.range n → f i ∈ Finset.range n) ∧
  (∀ i j, i ≠ j → cellColor n i (f i) ≠ cellColor n j (f j))

theorem rook_placement_modulo_four (n : ℕ) :
  (∃ f, IsValidRookPlacement n f) →
  n % 4 = 0 ∨ n % 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_rook_placement_modulo_four_l2141_214112


namespace NUMINAMATH_CALUDE_soda_price_l2141_214123

/-- The cost of a burger in cents -/
def burger_cost : ℕ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

/-- Four burgers and three sodas cost 540 cents -/
axiom alice_purchase : 4 * burger_cost + 3 * soda_cost = 540

/-- Three burgers and two sodas cost 390 cents -/
axiom bill_purchase : 3 * burger_cost + 2 * soda_cost = 390

/-- The cost of a soda is 60 cents -/
theorem soda_price : soda_cost = 60 := by sorry

end NUMINAMATH_CALUDE_soda_price_l2141_214123


namespace NUMINAMATH_CALUDE_segments_form_triangle_l2141_214157

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem segments_form_triangle :
  can_form_triangle 5 6 10 :=
by sorry

end NUMINAMATH_CALUDE_segments_form_triangle_l2141_214157


namespace NUMINAMATH_CALUDE_product_of_parts_of_complex_square_l2141_214173

theorem product_of_parts_of_complex_square : ∃ (a b : ℝ), (Complex.mk 1 2)^2 = Complex.mk a b ∧ a * b = -12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_parts_of_complex_square_l2141_214173


namespace NUMINAMATH_CALUDE_square_stack_area_l2141_214168

theorem square_stack_area (blue_exposed red_exposed yellow_exposed : ℝ) 
  (h1 : blue_exposed = 25)
  (h2 : red_exposed = 19)
  (h3 : yellow_exposed = 11) :
  let blue_side := Real.sqrt blue_exposed
  let red_uncovered := red_exposed / blue_side
  let large_side := blue_side + red_uncovered
  large_side ^ 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_stack_area_l2141_214168


namespace NUMINAMATH_CALUDE_find_M_l2141_214107

theorem find_M : ∃ M : ℚ, (25 / 100) * M = (35 / 100) * 4025 ∧ M = 5635 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l2141_214107


namespace NUMINAMATH_CALUDE_trivia_team_absentees_l2141_214194

theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 5 → 
  points_per_member = 6 → 
  total_points = 18 → 
  total_members - (total_points / points_per_member) = 2 :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_absentees_l2141_214194


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l2141_214170

/-- The set of possible slopes for a line with y-intercept (0,3) that intersects
    the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt 2 / Real.sqrt 110 ∨ m ≥ Real.sqrt 2 / Real.sqrt 110}

/-- The equation of the line with slope m and y-intercept 3 -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem line_intersects_ellipse (m : ℝ) :
  m ∈ possible_slopes ↔
  ∃ x : ℝ, ellipse_equation x (line_equation m x) := by
  sorry

#check line_intersects_ellipse

end NUMINAMATH_CALUDE_line_intersects_ellipse_l2141_214170


namespace NUMINAMATH_CALUDE_derivative_at_one_l2141_214120

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem derivative_at_one (x : ℝ) :
  ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → |((f (1 + h) - f 1) / h) - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2141_214120


namespace NUMINAMATH_CALUDE_probability_sum_thirty_l2141_214127

/-- Die A is a 30-faced die numbered 1-25 and 27-31 -/
def DieA : Finset ℕ := Finset.filter (fun n => n ≠ 26) (Finset.range 32 \ Finset.range 1)

/-- Die B is a 30-faced die numbered 1-20 and 26-31 -/
def DieB : Finset ℕ := (Finset.range 21 \ Finset.range 1) ∪ (Finset.range 32 \ Finset.range 26)

/-- The set of all possible outcomes when rolling both dice -/
def AllOutcomes : Finset (ℕ × ℕ) := DieA.product DieB

/-- The set of outcomes where the sum of the rolled numbers is 30 -/
def SumThirty : Finset (ℕ × ℕ) := AllOutcomes.filter (fun p => p.1 + p.2 = 30)

/-- The probability of rolling a sum of 30 with the given dice -/
def ProbabilitySumThirty : ℚ := (SumThirty.card : ℚ) / (AllOutcomes.card : ℚ)

theorem probability_sum_thirty : ProbabilitySumThirty = 59 / 900 := by sorry

end NUMINAMATH_CALUDE_probability_sum_thirty_l2141_214127


namespace NUMINAMATH_CALUDE_A_B_white_mutually_exclusive_l2141_214160

/-- Represents a person who can receive a ball -/
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person

/-- Represents a ball color -/
inductive BallColor : Type
  | Red : BallColor
  | Black : BallColor
  | White : BallColor

/-- Represents a distribution of balls to people -/
def Distribution := Person → BallColor

/-- The event that person A receives the white ball -/
def A_receives_white (d : Distribution) : Prop := d Person.A = BallColor.White

/-- The event that person B receives the white ball -/
def B_receives_white (d : Distribution) : Prop := d Person.B = BallColor.White

/-- Each person receives exactly one ball -/
def valid_distribution (d : Distribution) : Prop :=
  ∀ (c : BallColor), ∃! (p : Person), d p = c

theorem A_B_white_mutually_exclusive :
  ∀ (d : Distribution), valid_distribution d →
    ¬(A_receives_white d ∧ B_receives_white d) :=
sorry

end NUMINAMATH_CALUDE_A_B_white_mutually_exclusive_l2141_214160


namespace NUMINAMATH_CALUDE_circle_tangent_to_axes_l2141_214185

/-- A circle in the Cartesian plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The standard equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- A circle is tangent to the x-axis -/
def Circle.tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

/-- A circle is tangent to the y-axis -/
def Circle.tangentToYAxis (c : Circle) : Prop :=
  c.center.1 = c.radius

/-- The main theorem -/
theorem circle_tangent_to_axes (c : Circle) :
  c.radius = 2 ∧ c.tangentToXAxis ∧ c.tangentToYAxis ↔ 
  ∀ x y : ℝ, c.equation x y ↔ (x - 2)^2 + (y - 2)^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_axes_l2141_214185


namespace NUMINAMATH_CALUDE_alice_bob_race_difference_l2141_214158

/-- The time difference between two runners finishing a race -/
def race_time_difference (alice_speed bob_speed race_distance : ℝ) : ℝ :=
  bob_speed * race_distance - alice_speed * race_distance

/-- Theorem stating the time difference between Alice and Bob in a 12-mile race -/
theorem alice_bob_race_difference :
  race_time_difference 7 9 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_race_difference_l2141_214158


namespace NUMINAMATH_CALUDE_car_price_before_discount_l2141_214197

theorem car_price_before_discount 
  (discount_percentage : ℝ) 
  (price_after_discount : ℝ) 
  (h1 : discount_percentage = 55) 
  (h2 : price_after_discount = 450000) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_percentage / 100) = price_after_discount ∧ 
    original_price = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_car_price_before_discount_l2141_214197


namespace NUMINAMATH_CALUDE_triangle_equals_four_l2141_214182

/-- Given that △ is a digit and △7₁₂ = △3₁₃, prove that △ = 4 -/
theorem triangle_equals_four (triangle : ℕ) 
  (h1 : triangle < 10) 
  (h2 : triangle * 12 + 7 = triangle * 13 + 3) : 
  triangle = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_equals_four_l2141_214182


namespace NUMINAMATH_CALUDE_donald_oranges_l2141_214145

theorem donald_oranges (initial : ℕ) (total : ℕ) (found : ℕ) : 
  initial = 4 → total = 9 → found = total - initial → found = 5 := by sorry

end NUMINAMATH_CALUDE_donald_oranges_l2141_214145


namespace NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l2141_214147

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum square side length needed to fit two rectangles -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r2.width) (r1.height + r2.height)

/-- The theorem stating the smallest possible square area -/
theorem smallest_square_area_for_rectangles :
  let r1 : Rectangle := ⟨2, 5⟩
  let r2 : Rectangle := ⟨4, 3⟩
  (minSquareSide r1 r2) ^ 2 = 36 := by sorry

end NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l2141_214147


namespace NUMINAMATH_CALUDE_simplify_expression_l2141_214186

theorem simplify_expression (b : ℝ) : (1)*(2*b)*(3*b^2)*(4*b^3)*(5*b^4)*(6*b^5) = 720*b^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2141_214186


namespace NUMINAMATH_CALUDE_death_rate_is_three_per_two_seconds_l2141_214154

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per second -/
def birth_rate : ℚ := 3

/-- Represents the net population increase per day -/
def net_increase_per_day : ℕ := 129600

/-- Calculates the death rate in people per second -/
def death_rate : ℚ := birth_rate - (net_increase_per_day : ℚ) / seconds_per_day

/-- Theorem stating that the death rate is 3 people every two seconds -/
theorem death_rate_is_three_per_two_seconds : 
  death_rate * 2 = 3 := by sorry

end NUMINAMATH_CALUDE_death_rate_is_three_per_two_seconds_l2141_214154


namespace NUMINAMATH_CALUDE_min_value_trigonometric_function_l2141_214196

theorem min_value_trigonometric_function :
  ∀ x : ℝ, 0 < x → x < π / 2 →
    1 / (Real.sin x)^2 + 12 * Real.sqrt 3 / Real.cos x ≥ 28 ∧
    ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < π / 2 ∧
      1 / (Real.sin x₀)^2 + 12 * Real.sqrt 3 / Real.cos x₀ = 28 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_function_l2141_214196


namespace NUMINAMATH_CALUDE_smallest_divisor_after_221_next_divisor_is_289_l2141_214103

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem smallest_divisor_after_221 (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000)  -- m is a 4-digit number
  (h2 : is_even m)             -- m is even
  (h3 : m % 221 = 0)           -- 221 is a divisor of m
  : (∃ d : ℕ, d ∣ m ∧ 221 < d ∧ d < 289) → False :=
by sorry

theorem next_divisor_is_289 (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000)  -- m is a 4-digit number
  (h2 : is_even m)             -- m is even
  (h3 : m % 221 = 0)           -- 221 is a divisor of m
  : 289 ∣ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_221_next_divisor_is_289_l2141_214103


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2141_214136

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3 * y = 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2141_214136


namespace NUMINAMATH_CALUDE_ellipse_theorem_l2141_214180

/-- The equation of the given ellipse -/
def given_ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

/-- The equation of the new ellipse -/
def new_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

/-- The point through which the new ellipse passes -/
def point : ℝ × ℝ := (3, -2)

theorem ellipse_theorem :
  (∃ (c : ℝ), c > 0 ∧
    (∀ (x y : ℝ), given_ellipse x y → 
      ∃ (f1 f2 : ℝ × ℝ), (f1.1 = c ∧ f1.2 = 0) ∧ (f2.1 = -c ∧ f2.2 = 0) ∧
        (x - f1.1)^2 + y^2 + (x - f2.1)^2 + y^2 = 
        ((x - f1.1)^2 + y^2)^(1/2) + ((x - f2.1)^2 + y^2)^(1/2)) ∧
    (∀ (x y : ℝ), new_ellipse x y →
      ∃ (f1 f2 : ℝ × ℝ), (f1.1 = c ∧ f1.2 = 0) ∧ (f2.1 = -c ∧ f2.2 = 0) ∧
        (x - f1.1)^2 + y^2 + (x - f2.1)^2 + y^2 = 
        ((x - f1.1)^2 + y^2)^(1/2) + ((x - f2.1)^2 + y^2)^(1/2))) ∧
  new_ellipse point.1 point.2 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_theorem_l2141_214180


namespace NUMINAMATH_CALUDE_three_faces_colored_count_l2141_214100

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  totalSmallCubes : ℕ
  smallCubesPerEdge : ℕ

/-- Calculates the number of small cubes with exactly three faces colored -/
def threeFacesColored (c : CutCube) : ℕ := 8

/-- Theorem: In a cube cut into 216 equal smaller cubes, 
    the number of small cubes with exactly three faces colored is 8 -/
theorem three_faces_colored_count :
  ∀ (c : CutCube), c.totalSmallCubes = 216 → threeFacesColored c = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_faces_colored_count_l2141_214100


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l2141_214153

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := -x^2 + 5*x - 6 ≥ 0

-- Part 1
theorem range_of_x (x : ℝ) (h : p 1 x ∧ q x) : 2 ≤ x ∧ x < 3 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, ¬(p a x) → ¬(q x)) 
  (h3 : ∃ x, q x ∧ p a x) : 
  1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l2141_214153


namespace NUMINAMATH_CALUDE_expression_simplification_l2141_214143

theorem expression_simplification (x y : ℝ) 
  (hx : x = Real.sqrt 2) 
  (hy : y = Real.sqrt 3) : 
  (x + y) * (x - y) - y * (2 * x - y) = 2 - 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2141_214143


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2141_214162

open Set Real

-- Define the universal set I as the set of real numbers
def I : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x + 1 ≥ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 2 ≥ 0}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (Bᶜ) = {x : ℝ | -1 ≤ x ∧ x < sqrt 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2141_214162


namespace NUMINAMATH_CALUDE_certain_number_calculation_l2141_214192

theorem certain_number_calculation (y : ℝ) : (0.65 * 210 = 0.20 * y) → y = 682.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l2141_214192


namespace NUMINAMATH_CALUDE_like_terms_proof_l2141_214105

/-- Two algebraic expressions are like terms if they have the same variables with the same exponents. -/
def like_terms (expr1 expr2 : String) : Prop := sorry

theorem like_terms_proof :
  (like_terms "3a³b" "-3ba³") ∧
  ¬(like_terms "a³" "b³") ∧
  ¬(like_terms "abc" "ac") ∧
  ¬(like_terms "a⁵" "2⁵") := by sorry

end NUMINAMATH_CALUDE_like_terms_proof_l2141_214105


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2141_214121

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 3| + |x + 3| < 7} = Set.Icc (-1) (7/3) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2141_214121


namespace NUMINAMATH_CALUDE_sum_x_y_equals_nine_fifths_l2141_214165

theorem sum_x_y_equals_nine_fifths (x y : ℝ) 
  (eq1 : x + |x| + y = 5)
  (eq2 : x + |y| - y = 6) : 
  x + y = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_nine_fifths_l2141_214165


namespace NUMINAMATH_CALUDE_no_root_in_interval_l2141_214118

-- Define the function f(x) = x^5 - 3x - 1
def f (x : ℝ) : ℝ := x^5 - 3*x - 1

-- State the theorem
theorem no_root_in_interval :
  ∀ x ∈ Set.Ioo 2 3, f x ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_root_in_interval_l2141_214118


namespace NUMINAMATH_CALUDE_circle_slash_problem_l2141_214133

/-- Custom operation ⊘ defined as (a ⊘ b) = (√(k*a + b))^3 -/
noncomputable def circle_slash (k : ℝ) (a b : ℝ) : ℝ := (Real.sqrt (k * a + b)) ^ 3

/-- Theorem: If 9 ⊘ x = 64 and k = 3, then x = -11 -/
theorem circle_slash_problem (x : ℝ) (h1 : circle_slash 3 9 x = 64) : x = -11 := by
  sorry

end NUMINAMATH_CALUDE_circle_slash_problem_l2141_214133


namespace NUMINAMATH_CALUDE_orange_packing_l2141_214122

theorem orange_packing (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 2650) (h2 : oranges_per_box = 10) :
  total_oranges / oranges_per_box = 265 := by
  sorry

end NUMINAMATH_CALUDE_orange_packing_l2141_214122


namespace NUMINAMATH_CALUDE_problem_statement_l2141_214175

theorem problem_statement (p q r : ℝ) 
  (h1 : p * q / (p + r) + q * r / (q + p) + r * p / (r + q) = -7)
  (h2 : p * r / (p + r) + q * p / (q + p) + r * q / (r + q) = 8) :
  q / (p + q) + r / (q + r) + p / (r + p) = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2141_214175


namespace NUMINAMATH_CALUDE_reflection_over_x_axis_l2141_214149

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]

def reflects_over_x_axis (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ (x y : ℝ), M.mulVec ![x, y] = ![x, -y]

theorem reflection_over_x_axis :
  reflects_over_x_axis reflection_matrix := by sorry

end NUMINAMATH_CALUDE_reflection_over_x_axis_l2141_214149


namespace NUMINAMATH_CALUDE_early_arrival_l2141_214119

/-- Given a boy who usually takes 14 minutes to reach school, if he walks at 7/6 of his usual rate, he will arrive 2 minutes early. -/
theorem early_arrival (usual_time : ℝ) (new_rate : ℝ) : 
  usual_time = 14 → new_rate = 7/6 → usual_time - (usual_time / new_rate) = 2 :=
by sorry

end NUMINAMATH_CALUDE_early_arrival_l2141_214119


namespace NUMINAMATH_CALUDE_vhs_to_dvd_cost_l2141_214114

def replace_cost (num_movies : ℕ) (trade_in_price : ℕ) (dvd_price : ℕ) : ℕ :=
  num_movies * dvd_price - num_movies * trade_in_price

theorem vhs_to_dvd_cost :
  replace_cost 100 2 10 = 800 := by
  sorry

end NUMINAMATH_CALUDE_vhs_to_dvd_cost_l2141_214114


namespace NUMINAMATH_CALUDE_factorial_difference_sum_l2141_214140

theorem factorial_difference_sum : Nat.factorial 10 - Nat.factorial 8 + Nat.factorial 6 = 3589200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_sum_l2141_214140


namespace NUMINAMATH_CALUDE_pencil_difference_l2141_214166

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The time frame in minutes -/
def time_frame : ℕ := 6

/-- The time it takes for a hand-crank sharpener to sharpen one pencil (in seconds) -/
def hand_crank_time : ℕ := 45

/-- The time it takes for an electric sharpener to sharpen one pencil (in seconds) -/
def electric_time : ℕ := 20

/-- The difference in the number of pencils sharpened between the electric and hand-crank sharpeners -/
theorem pencil_difference : 
  (time_frame * seconds_per_minute) / electric_time - 
  (time_frame * seconds_per_minute) / hand_crank_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_difference_l2141_214166


namespace NUMINAMATH_CALUDE_optimal_allocation_l2141_214111

/-- Represents the allocation of workers in a furniture factory -/
structure WorkerAllocation where
  total_workers : ℕ
  tabletop_workers : ℕ
  tableleg_workers : ℕ
  tabletops_per_worker : ℕ
  tablelegs_per_worker : ℕ
  legs_per_table : ℕ

/-- Checks if the allocation produces matching numbers of tabletops and table legs -/
def is_matching_production (w : WorkerAllocation) : Prop :=
  w.tabletop_workers * w.tabletops_per_worker * w.legs_per_table = 
  w.tableleg_workers * w.tablelegs_per_worker

/-- The theorem stating the optimal worker allocation -/
theorem optimal_allocation :
  ∀ w : WorkerAllocation,
    w.total_workers = 60 ∧
    w.tabletops_per_worker = 3 ∧
    w.tablelegs_per_worker = 6 ∧
    w.legs_per_table = 4 ∧
    w.tabletop_workers + w.tableleg_workers = w.total_workers →
    (w.tabletop_workers = 20 ∧ w.tableleg_workers = 40) ↔ 
    is_matching_production w :=
by sorry

end NUMINAMATH_CALUDE_optimal_allocation_l2141_214111


namespace NUMINAMATH_CALUDE_apple_probability_l2141_214169

def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def chosen_apples : ℕ := 3

theorem apple_probability :
  (Nat.choose red_apples chosen_apples +
   Nat.choose green_apples chosen_apples +
   (Nat.choose red_apples 2 * Nat.choose green_apples 1) +
   (Nat.choose green_apples 2 * Nat.choose red_apples 1)) /
  Nat.choose total_apples chosen_apples = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_probability_l2141_214169


namespace NUMINAMATH_CALUDE_berry_temperature_proof_l2141_214115

theorem berry_temperature_proof (temps : List Float) (avg : Float) : 
  temps = [99.1, 98.2, 98.7, 99.8, 99, 98.9] →
  avg = 99 →
  ∃ (wed_temp : Float), 
    wed_temp = 99.3 ∧ 
    (temps.sum + wed_temp) / 7 = avg :=
by sorry

end NUMINAMATH_CALUDE_berry_temperature_proof_l2141_214115


namespace NUMINAMATH_CALUDE_base_7_addition_problem_l2141_214156

/-- Given an addition problem in base 7, prove that X + Y = 10 in base 10 --/
theorem base_7_addition_problem (X Y : ℕ) : 
  (5 * 7^2 + X * 7 + Y) + (5 * 7 + 2) = 6 * 7^2 + 4 * 7 + X → X + Y = 10 := by
sorry

end NUMINAMATH_CALUDE_base_7_addition_problem_l2141_214156


namespace NUMINAMATH_CALUDE_jane_toy_bear_production_l2141_214161

/-- Jane's toy bear production problem -/
theorem jane_toy_bear_production 
  (base_output : ℝ) 
  (base_hours : ℝ) 
  (assistant_output_increase : ℝ) 
  (assistant_hours_decrease : ℝ) 
  (assistant_A_increase : ℝ) 
  (assistant_B_increase : ℝ) 
  (assistant_C_increase : ℝ) 
  (h1 : assistant_output_increase = 0.8) 
  (h2 : assistant_hours_decrease = 0.1) 
  (h3 : assistant_A_increase = 1.0) 
  (h4 : assistant_B_increase = 0.75) 
  (h5 : assistant_C_increase = 0.5) :
  let output_A := (1 + assistant_A_increase) * base_output / ((1 - assistant_hours_decrease) * base_hours)
  let output_B := (1 + assistant_B_increase) * base_output / ((1 - assistant_hours_decrease) * base_hours)
  let output_C := (1 + assistant_C_increase) * base_output / ((1 - assistant_hours_decrease) * base_hours)
  let increase_A := (output_A / (base_output / base_hours) - 1) * 100
  let increase_B := (output_B / (base_output / base_hours) - 1) * 100
  let increase_C := (output_C / (base_output / base_hours) - 1) * 100
  let average_increase := (increase_A + increase_B + increase_C) / 3
  ∃ ε > 0, |average_increase - 94.43| < ε :=
by sorry

end NUMINAMATH_CALUDE_jane_toy_bear_production_l2141_214161


namespace NUMINAMATH_CALUDE_insects_eaten_by_geckos_and_lizards_l2141_214117

/-- The number of insects eaten by geckos and lizards -/
def total_insects_eaten (num_geckos : ℕ) (insects_per_gecko : ℕ) (num_lizards : ℕ) : ℕ :=
  num_geckos * insects_per_gecko + num_lizards * (2 * insects_per_gecko)

/-- Theorem stating the total number of insects eaten in the given scenario -/
theorem insects_eaten_by_geckos_and_lizards :
  total_insects_eaten 5 6 3 = 66 := by
  sorry


end NUMINAMATH_CALUDE_insects_eaten_by_geckos_and_lizards_l2141_214117


namespace NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l2141_214141

theorem largest_of_eight_consecutive_integers (a : ℕ) 
  (h1 : a > 0) 
  (h2 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6) + (a + 7)) = 5400) : 
  (a + 7) = 678 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l2141_214141


namespace NUMINAMATH_CALUDE_merry_and_brother_lambs_l2141_214176

/-- The number of lambs Merry and her brother have -/
theorem merry_and_brother_lambs :
  let merry_lambs : ℕ := 10
  let brother_lambs : ℕ := merry_lambs + 3
  merry_lambs + brother_lambs = 23 :=
by sorry

end NUMINAMATH_CALUDE_merry_and_brother_lambs_l2141_214176


namespace NUMINAMATH_CALUDE_days_missed_proof_l2141_214132

/-- The total number of days missed by Vanessa, Mike, and Sarah -/
def total_days_missed (v m s : ℕ) : ℕ := v + m + s

/-- Theorem: Given the conditions, the total number of days missed is 17 -/
theorem days_missed_proof (v m s : ℕ) 
  (h1 : v + m = 14)  -- Vanessa and Mike have missed 14 days total
  (h2 : m + s = 12)  -- Mike and Sarah have missed 12 days total
  (h3 : v = 5)       -- Vanessa missed 5 days of school alone
  : total_days_missed v m s = 17 := by
  sorry

#check days_missed_proof

end NUMINAMATH_CALUDE_days_missed_proof_l2141_214132


namespace NUMINAMATH_CALUDE_unique_solution_for_abc_l2141_214152

theorem unique_solution_for_abc : ∃! (a b c : ℝ),
  a < b ∧ b < c ∧
  a + b + c = 21 / 4 ∧
  1 / a + 1 / b + 1 / c = 21 / 4 ∧
  a * b * c = 1 ∧
  a = 1 / 4 ∧ b = 1 ∧ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_abc_l2141_214152


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2141_214139

theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  (‖a‖ = 1) → (‖b‖ = 1) → (a • (a - 2 • b) = 0) → ‖a + b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2141_214139


namespace NUMINAMATH_CALUDE_function_inequality_solution_set_l2141_214142

open Real

-- Define the function f and its properties
theorem function_inequality_solution_set 
  (f : ℝ → ℝ) 
  (f' : ℝ → ℝ) 
  (h1 : ∀ x > 0, HasDerivAt f (f' x) x)
  (h2 : ∀ x > 0, x * f' x + f x = (log x) / x)
  (h3 : f (exp 1) = (exp 1)⁻¹) :
  {x : ℝ | f (x + 1) - f ((exp 1) + 1) > x - (exp 1)} = Set.Ioo (-1) (exp 1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_solution_set_l2141_214142


namespace NUMINAMATH_CALUDE_may_blue_yarns_l2141_214188

/-- The number of scarves May can knit using one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := 4

/-- The total number of scarves May will be able to make -/
def total_scarves : ℕ := 36

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

theorem may_blue_yarns : 
  scarves_per_yarn * (red_yarns + yellow_yarns + blue_yarns) = total_scarves :=
by sorry

end NUMINAMATH_CALUDE_may_blue_yarns_l2141_214188


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2141_214134

theorem arithmetic_calculations :
  ((1 : ℤ) * (-5) - (-6) + (-7) = -6) ∧
  ((-1 : ℤ)^2021 + (-18) * |(-2 : ℚ) / 9| - 4 / (-2 : ℤ) = -3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2141_214134


namespace NUMINAMATH_CALUDE_julia_garden_area_l2141_214163

/-- Represents a rectangular garden with given walking constraints -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  length_walk : length * 30 = 1500
  perimeter_walk : (length + width) * 2 * 12 = 1500

/-- The area of Julia's garden is 625 square meters -/
theorem julia_garden_area (garden : RectangularGarden) : garden.length * garden.width = 625 := by
  sorry

#check julia_garden_area

end NUMINAMATH_CALUDE_julia_garden_area_l2141_214163


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l2141_214144

def arithmetic_sequence_sum (a : ℕ) (d : ℕ) (last : ℕ) : ℕ :=
  let n := (last - a) / d + 1
  n * (a + last) / 2

theorem arithmetic_sequence_sum_remainder
  (h1 : arithmetic_sequence_sum 3 6 279 % 8 = 3) : 
  arithmetic_sequence_sum 3 6 279 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l2141_214144


namespace NUMINAMATH_CALUDE_tile_arrangement_count_l2141_214193

/-- The number of distinguishable arrangements of tiles -/
def tileArrangements (brown purple green yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 brown, 1 purple, 3 green, and 2 yellow tiles is 420 -/
theorem tile_arrangement_count :
  tileArrangements 1 1 3 2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangement_count_l2141_214193


namespace NUMINAMATH_CALUDE_tangent_at_one_two_tangent_through_one_one_l2141_214101

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- Theorem for the tangent line at (1, 2)
theorem tangent_at_one_two :
  ∃ (m b : ℝ), (∀ x y, y = m * x + b ↔ 2 * x - y = 0) ∧
  f 1 = 2 ∧ f' 1 = m := by sorry

-- Theorem for the tangent lines through (1, 1)
theorem tangent_through_one_one :
  ∃ (x₀ : ℝ), (x₀ = 0 ∨ x₀ = 2) ∧
  (∀ x y, y = 1 ↔ x₀ = 0 ∧ y = f x₀ + f' x₀ * (x - x₀)) ∧
  (∀ x y, 4 * x - y - 3 = 0 ↔ x₀ = 2 ∧ y = f x₀ + f' x₀ * (x - x₀)) ∧
  f x₀ + f' x₀ * (1 - x₀) = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_at_one_two_tangent_through_one_one_l2141_214101
