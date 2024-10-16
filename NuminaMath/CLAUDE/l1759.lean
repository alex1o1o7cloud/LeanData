import Mathlib

namespace NUMINAMATH_CALUDE_cost_per_deck_is_8_l1759_175963

/-- The cost of a single trick deck -/
def cost_per_deck : ℝ := sorry

/-- The number of decks Victor bought -/
def victor_decks : ℕ := 6

/-- The number of decks Victor's friend bought -/
def friend_decks : ℕ := 2

/-- The total amount spent -/
def total_spent : ℝ := 64

/-- Theorem stating that the cost per deck is 8 dollars -/
theorem cost_per_deck_is_8 : cost_per_deck = 8 :=
  by sorry

end NUMINAMATH_CALUDE_cost_per_deck_is_8_l1759_175963


namespace NUMINAMATH_CALUDE_textbook_delivery_problem_l1759_175915

theorem textbook_delivery_problem (x y : ℝ) : 
  (0.5 * x + 0.2 * y = 390) ∧ 
  (0.5 * x = 3 * 0.8 * y) →
  (x = 720 ∧ y = 150) := by
sorry

end NUMINAMATH_CALUDE_textbook_delivery_problem_l1759_175915


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1759_175947

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 2 + a 3 + a 10 + a 11 = 48) →
  a 6 + a 7 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1759_175947


namespace NUMINAMATH_CALUDE_point_relationship_l1759_175929

/-- Given two points A(-1/2, m) and B(2, n) on the line y = 3x + b, prove that m < n -/
theorem point_relationship (m n b : ℝ) : 
  ((-1/2 : ℝ), m) ∈ {(x, y) | y = 3*x + b} →
  ((2 : ℝ), n) ∈ {(x, y) | y = 3*x + b} →
  m < n :=
by sorry

end NUMINAMATH_CALUDE_point_relationship_l1759_175929


namespace NUMINAMATH_CALUDE_person_is_knight_l1759_175970

-- Define the type of person
inductive Person : Type
  | Knight : Person
  | Liar : Person

-- Define the statements
def lovesLinda (p : Person) : Prop := 
  match p with
  | Person.Knight => true
  | Person.Liar => false

def lovesKatie (p : Person) : Prop :=
  match p with
  | Person.Knight => true
  | Person.Liar => false

-- Define the theorem
theorem person_is_knight : 
  ∀ (p : Person), 
    (lovesLinda p = true ∨ lovesLinda p = false) → 
    (lovesLinda p → lovesKatie p) → 
    p = Person.Knight :=
by
  sorry


end NUMINAMATH_CALUDE_person_is_knight_l1759_175970


namespace NUMINAMATH_CALUDE_other_solution_quadratic_equation_l1759_175916

theorem other_solution_quadratic_equation :
  let f : ℚ → ℚ := λ x ↦ 45 * x^2 - 56 * x + 31
  ∃ x : ℚ, x ≠ 2/5 ∧ f x = 0 ∧ x = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_equation_l1759_175916


namespace NUMINAMATH_CALUDE_modular_inverse_of_2_mod_199_l1759_175959

theorem modular_inverse_of_2_mod_199 : ∃ x : ℤ, 2 * x ≡ 1 [ZMOD 199] ∧ 0 ≤ x ∧ x < 199 :=
  by sorry

end NUMINAMATH_CALUDE_modular_inverse_of_2_mod_199_l1759_175959


namespace NUMINAMATH_CALUDE_dogs_liking_no_food_l1759_175906

def total_dogs : ℕ := 80
def watermelon_dogs : ℕ := 18
def salmon_dogs : ℕ := 58
def chicken_dogs : ℕ := 16
def watermelon_and_salmon : ℕ := 7
def chicken_and_salmon : ℕ := 6
def chicken_and_watermelon : ℕ := 4
def all_three : ℕ := 3

theorem dogs_liking_no_food : 
  total_dogs - (watermelon_dogs + salmon_dogs + chicken_dogs
              - watermelon_and_salmon - chicken_and_salmon - chicken_and_watermelon
              + all_three) = 2 := by
  sorry

end NUMINAMATH_CALUDE_dogs_liking_no_food_l1759_175906


namespace NUMINAMATH_CALUDE_function_properties_l1759_175937

def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (-(x + 1))

def is_odd_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f (-(x + 2))

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def equation_solutions (f : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x = x ∧ f y = y ∧ f z = z ∧
    ∀ w, f w = w → w = x ∨ w = y ∨ w = z

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even_shifted f)
  (h2 : is_odd_shifted f)
  (h3 : ∀ x ∈ Set.Icc 0 1, f x = 2^x - 1) :
  symmetric_about f 1 ∧ equation_solutions f := by
sorry

end NUMINAMATH_CALUDE_function_properties_l1759_175937


namespace NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l1759_175954

theorem ice_cream_scoop_arrangements (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l1759_175954


namespace NUMINAMATH_CALUDE_radius_of_polar_circle_is_three_l1759_175920

/-- The radius of a circle defined by the polar equation ρ = 6cosθ -/
def radius_of_polar_circle : ℝ := 3

/-- The polar equation of the circle -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 6 * Real.cos θ

theorem radius_of_polar_circle_is_three :
  ∀ ρ θ : ℝ, polar_equation ρ θ → 
  ∃ x y : ℝ, x^2 + y^2 = radius_of_polar_circle^2 ∧ 
             x = ρ * Real.cos θ ∧ 
             y = ρ * Real.sin θ :=
sorry

end NUMINAMATH_CALUDE_radius_of_polar_circle_is_three_l1759_175920


namespace NUMINAMATH_CALUDE_shems_earnings_l1759_175938

/-- Proves that Shem's earnings for an 8-hour workday is $80, given the conditions. -/
theorem shems_earnings (kem_hourly_rate : ℝ) (shem_multiplier : ℝ) (workday_hours : ℕ) :
  kem_hourly_rate = 4 →
  shem_multiplier = 2.5 →
  workday_hours = 8 →
  kem_hourly_rate * shem_multiplier * workday_hours = 80 := by
  sorry

#check shems_earnings

end NUMINAMATH_CALUDE_shems_earnings_l1759_175938


namespace NUMINAMATH_CALUDE_jenny_sold_192_packs_l1759_175931

/-- The number of boxes Jenny sold -/
def boxes_sold : ℝ := 24.0

/-- The number of packs per box -/
def packs_per_box : ℝ := 8.0

/-- The total number of packs Jenny sold -/
def total_packs : ℝ := boxes_sold * packs_per_box

theorem jenny_sold_192_packs : total_packs = 192.0 := by
  sorry

end NUMINAMATH_CALUDE_jenny_sold_192_packs_l1759_175931


namespace NUMINAMATH_CALUDE_unique_solution_l1759_175911

/-- The functional equation satisfied by g --/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x + y) * g (x - y) = (g x - g y)^2 - 6 * x^2 * g y

/-- There is only one function satisfying the functional equation --/
theorem unique_solution :
  ∃! g : ℝ → ℝ, FunctionalEquation g ∧ ∀ x : ℝ, g x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1759_175911


namespace NUMINAMATH_CALUDE_no_matching_pyramids_l1759_175969

/-- Represents a convex n-sided pyramid -/
structure NSidedPyramid (n : ℕ) :=
  (convex : Bool)
  (dihedralAngles : Fin n → ℝ)

/-- Represents a triangular pyramid -/
structure TriangularPyramid :=
  (dihedralAngles : Fin 4 → ℝ)

/-- The theorem stating that no such pair of pyramids exists -/
theorem no_matching_pyramids :
  ∀ (n : ℕ) (nPyramid : NSidedPyramid n) (tPyramid : TriangularPyramid),
    n ≥ 4 →
    nPyramid.convex = true →
    (∃ (i j k l : Fin n),
      i ≠ j ∧ i ≠ k ∧ i ≠ l ∧
      j ≠ k ∧ j ≠ l ∧
      k ≠ l ∧
      nPyramid.dihedralAngles i = tPyramid.dihedralAngles 0 ∧
      nPyramid.dihedralAngles j = tPyramid.dihedralAngles 1 ∧
      nPyramid.dihedralAngles k = tPyramid.dihedralAngles 2 ∧
      nPyramid.dihedralAngles l = tPyramid.dihedralAngles 3) →
    False :=
by sorry

end NUMINAMATH_CALUDE_no_matching_pyramids_l1759_175969


namespace NUMINAMATH_CALUDE_unique_solution_mn_l1759_175986

theorem unique_solution_mn (m n : ℕ+) 
  (h1 : (n^4 : ℕ) ∣ 2*m^5 - 1)
  (h2 : (m^4 : ℕ) ∣ 2*n^5 + 1) :
  m = 1 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l1759_175986


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1759_175944

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ) * (2 + a * Complex.I) * (a - 2 * Complex.I) = -4 * Complex.I → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1759_175944


namespace NUMINAMATH_CALUDE_power_division_rule_l1759_175957

theorem power_division_rule (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1759_175957


namespace NUMINAMATH_CALUDE_problem_distribution_count_l1759_175975

def distribute_problems (n_problems : ℕ) (n_friends : ℕ) (max_recipients : ℕ) : ℕ :=
  (n_friends.choose max_recipients) * (max_recipients ^ n_problems)

theorem problem_distribution_count :
  distribute_problems 7 10 3 = 262440 :=
by sorry

end NUMINAMATH_CALUDE_problem_distribution_count_l1759_175975


namespace NUMINAMATH_CALUDE_parallel_intersections_l1759_175962

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation for planes and lines
variable (intersect : Plane → Plane → Line → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_intersections
  (α β γ : Plane) (m n : Line)
  (h1 : parallel_planes α β)
  (h2 : intersect α γ m)
  (h3 : intersect β γ n) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_parallel_intersections_l1759_175962


namespace NUMINAMATH_CALUDE_derivative_at_negative_third_l1759_175994

/-- Given a function f(x) = x^2 + 2f'(-1/3)x, prove that f'(-1/3) = 2/3 -/
theorem derivative_at_negative_third (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + 2 * (deriv f (-1/3)) * x) :
  deriv f (-1/3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_third_l1759_175994


namespace NUMINAMATH_CALUDE_train_length_l1759_175988

/-- Given a train that crosses a platform of length 350 meters in 39 seconds
    and crosses a signal pole in 18 seconds, the length of the train is 300 meters. -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
    (h1 : platform_length = 350)
    (h2 : platform_time = 39)
    (h3 : pole_time = 18) :
    (platform_length * pole_time) / (platform_time - pole_time) = 300 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1759_175988


namespace NUMINAMATH_CALUDE_union_equals_reals_l1759_175981

-- Define sets A and B
def A : Set ℝ := {x | Real.log x > 0}
def B : Set ℝ := {x | x ≤ 1}

-- State the theorem
theorem union_equals_reals : A ∪ B = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_union_equals_reals_l1759_175981


namespace NUMINAMATH_CALUDE_existence_of_x_y_l1759_175978

theorem existence_of_x_y : ∃ (x y : ℝ), 3*x + y > 0 ∧ 4*x + y > 0 ∧ 6*x + 5*y < 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_y_l1759_175978


namespace NUMINAMATH_CALUDE_milk_tea_sales_distribution_l1759_175909

/-- Represents the sales distribution of milk tea flavors -/
structure MilkTeaSales where
  total : ℕ
  winterMelon : ℕ
  okinawa : ℕ
  chocolate : ℕ
  thai : ℕ
  taro : ℕ

/-- Conditions for the milk tea sales problem -/
def salesConditions (s : MilkTeaSales) : Prop :=
  s.total = 100 ∧
  s.winterMelon = (35 * s.total) / 100 ∧
  s.okinawa = s.total / 4 ∧
  s.taro = 12 ∧
  3 * s.chocolate = 7 * s.thai ∧
  s.chocolate + s.thai = s.total - s.winterMelon - s.okinawa - s.taro

/-- Theorem stating the correct distribution of milk tea sales -/
theorem milk_tea_sales_distribution :
  ∃ (s : MilkTeaSales),
    salesConditions s ∧
    s.winterMelon = 35 ∧
    s.okinawa = 25 ∧
    s.chocolate = 8 ∧
    s.thai = 20 ∧
    s.taro = 12 ∧
    s.winterMelon + s.okinawa + s.chocolate + s.thai + s.taro = s.total :=
by
  sorry

end NUMINAMATH_CALUDE_milk_tea_sales_distribution_l1759_175909


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l1759_175953

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_are_different (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens ≠ ones

def sum_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens + ones

theorem unique_two_digit_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
            digits_are_different n ∧ 
            n^2 = (sum_of_digits n)^3 ∧
            n = 27 :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l1759_175953


namespace NUMINAMATH_CALUDE_exam_correct_answers_l1759_175900

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  total_questions : Nat
  correct_score : Int
  wrong_score : Int

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  total_score : Int

/-- Calculates the number of correctly answered questions. -/
def correct_answers (result : ExamResult) : Nat :=
  sorry

/-- Theorem stating that for the given exam conditions, 
    the number of correct answers is 42. -/
theorem exam_correct_answers 
  (e : Exam) 
  (r : ExamResult) 
  (h1 : e.total_questions = 80) 
  (h2 : e.correct_score = 4) 
  (h3 : e.wrong_score = -1) 
  (h4 : r.exam = e) 
  (h5 : r.total_score = 130) : 
  correct_answers r = 42 := by
  sorry

end NUMINAMATH_CALUDE_exam_correct_answers_l1759_175900


namespace NUMINAMATH_CALUDE_rectangle_side_length_l1759_175961

/-- Given two rectangles A and B, where A has sides of length 3 and 6,
    and the ratio of corresponding sides of A to B is 3/4,
    prove that the length of side c in Rectangle B is 4. -/
theorem rectangle_side_length (a b c d : ℝ) : 
  a = 3 → b = 6 → a / c = 3 / 4 → b / d = 3 / 4 → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l1759_175961


namespace NUMINAMATH_CALUDE_factorial_plus_one_eq_power_l1759_175993

theorem factorial_plus_one_eq_power (n p : ℕ) : 
  (Nat.factorial (p - 1) + 1 = p ^ n) ↔ 
  ((n = 1 ∧ p = 2) ∨ (n = 1 ∧ p = 3) ∨ (n = 2 ∧ p = 5)) :=
sorry

end NUMINAMATH_CALUDE_factorial_plus_one_eq_power_l1759_175993


namespace NUMINAMATH_CALUDE_octagon_interior_angle_l1759_175942

/-- The measure of each interior angle in a regular octagon -/
def interior_angle_octagon : ℝ := 135

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem octagon_interior_angle :
  interior_angle_octagon = (sum_interior_angles octagon_sides) / octagon_sides :=
by sorry

end NUMINAMATH_CALUDE_octagon_interior_angle_l1759_175942


namespace NUMINAMATH_CALUDE_cosine_inequality_equivalence_l1759_175985

theorem cosine_inequality_equivalence (x : Real) : 
  (∃ k : Int, (-π/6 + 2*π*k < x ∧ x < π/6 + 2*π*k) ∨ 
              (2*π/3 + 2*π*k < x ∧ x < 4*π/3 + 2*π*k)) ↔ 
  (Real.cos (2*x) - 4*Real.cos (π/4)*Real.cos (5*π/12)*Real.cos x + 
   Real.cos (5*π/6) + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_equivalence_l1759_175985


namespace NUMINAMATH_CALUDE_f_g_f_3_equals_101_l1759_175919

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 5 * x + 4

-- State the theorem
theorem f_g_f_3_equals_101 : f (g (f 3)) = 101 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_3_equals_101_l1759_175919


namespace NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l1759_175987

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let ball_diameter : ℝ := 40
  let ball_radius : ℝ := ball_diameter / 2
  let hole1_diameter : ℝ := 4
  let hole1_radius : ℝ := hole1_diameter / 2
  let hole2_diameter : ℝ := 2.5
  let hole2_radius : ℝ := hole2_diameter / 2
  let hole_depth : ℝ := 8
  let ball_volume : ℝ := (4 / 3) * π * (ball_radius ^ 3)
  let hole1_volume : ℝ := π * (hole1_radius ^ 2) * hole_depth
  let hole2_volume : ℝ := π * (hole2_radius ^ 2) * hole_depth
  ball_volume - hole1_volume - 2 * hole2_volume = 10609.67 * π :=
by sorry

end NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l1759_175987


namespace NUMINAMATH_CALUDE_product_closest_to_2500_l1759_175967

def options : List ℝ := [2500, 2600, 250, 260, 25000]

def product : ℝ := 0.0003125 * 8125312

theorem product_closest_to_2500 : 
  ∀ x ∈ options, |product - 2500| ≤ |product - x| :=
sorry

end NUMINAMATH_CALUDE_product_closest_to_2500_l1759_175967


namespace NUMINAMATH_CALUDE_max_obtuse_triangles_four_points_l1759_175914

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle formed by three points -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Predicate to check if a triangle is obtuse -/
def isObtuse (t : Triangle) : Prop :=
  sorry

/-- The set of all possible triangles formed by 4 points -/
def allTriangles (p1 p2 p3 p4 : Point) : Set Triangle :=
  sorry

/-- The number of obtuse triangles in a set of triangles -/
def numObtuseTriangles (ts : Set Triangle) : ℕ :=
  sorry

/-- Theorem: The maximum number of obtuse triangles formed by 4 points is 4 -/
theorem max_obtuse_triangles_four_points (p1 p2 p3 p4 : Point) :
  ∃ (arrangement : Point → Point),
    numObtuseTriangles (allTriangles (arrangement p1) (arrangement p2) (arrangement p3) (arrangement p4)) ≤ 4 ∧
    ∃ (q1 q2 q3 q4 : Point),
      numObtuseTriangles (allTriangles q1 q2 q3 q4) = 4 :=
sorry

end NUMINAMATH_CALUDE_max_obtuse_triangles_four_points_l1759_175914


namespace NUMINAMATH_CALUDE_new_ellipse_and_hyperbola_l1759_175910

/-- New distance between two points -/
def new_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- New ellipse -/
def on_new_ellipse (x y c d a : ℝ) : Prop :=
  new_distance x y c d + new_distance x y (-c) (-d) = 2 * a

/-- New hyperbola -/
def on_new_hyperbola (x y c d a : ℝ) : Prop :=
  |new_distance x y c d - new_distance x y (-c) (-d)| = 2 * a

/-- Main theorem for new ellipse and hyperbola -/
theorem new_ellipse_and_hyperbola (x y c d a : ℝ) :
  (on_new_ellipse x y c d a ↔ 
    |x - c| + |y - d| + |x + c| + |y + d| = 2 * a) ∧
  (on_new_hyperbola x y c d a ↔ 
    |(|x - c| + |y - d|) - (|x + c| + |y + d|)| = 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_new_ellipse_and_hyperbola_l1759_175910


namespace NUMINAMATH_CALUDE_tan_135_degrees_l1759_175902

theorem tan_135_degrees : 
  let angle : Real := 135 * Real.pi / 180
  let point : Fin 2 → Real := ![-(Real.sqrt 2) / 2, (Real.sqrt 2) / 2]
  Real.tan angle = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_135_degrees_l1759_175902


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l1759_175943

theorem sum_of_squares_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l1759_175943


namespace NUMINAMATH_CALUDE_power_of_power_l1759_175940

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1759_175940


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1759_175939

theorem arithmetic_calculations : 
  (12 - (-18) + (-7) + (-15) = 8) ∧ 
  (-2^3 + (-5)^2 * (2/5) - |(-3)| = -1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1759_175939


namespace NUMINAMATH_CALUDE_square_triangle_area_equality_l1759_175945

theorem square_triangle_area_equality (x : ℝ) (h : x > 0) :
  let square_area := x^2
  let triangle_base := x
  let triangle_altitude := 2 * x
  let triangle_area := (1 / 2) * triangle_base * triangle_altitude
  square_area = triangle_area := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_area_equality_l1759_175945


namespace NUMINAMATH_CALUDE_walking_running_distance_ratio_l1759_175923

/-- Proves that the ratio of distance walked to distance run is 1:1 given the specified conditions --/
theorem walking_running_distance_ratio
  (walking_speed : ℝ)
  (running_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (h1 : walking_speed = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 3)
  (h4 : total_distance = 16)
  : ∃ (distance_walked distance_run : ℝ),
    distance_walked / walking_speed + distance_run / running_speed = total_time ∧
    distance_walked + distance_run = total_distance ∧
    distance_walked / distance_run = 1 := by
  sorry

end NUMINAMATH_CALUDE_walking_running_distance_ratio_l1759_175923


namespace NUMINAMATH_CALUDE_milk_cost_is_1_15_l1759_175955

/-- The cost of Anna's breakfast items and lunch sandwich, and the difference between lunch and breakfast costs -/
structure AnnasMeals where
  bagel_cost : ℚ
  juice_cost : ℚ
  sandwich_cost : ℚ
  lunch_breakfast_diff : ℚ

/-- Calculate the cost of the milk carton based on Anna's meal expenses -/
def milk_cost (meals : AnnasMeals) : ℚ :=
  let breakfast_cost := meals.bagel_cost + meals.juice_cost
  let lunch_cost := breakfast_cost + meals.lunch_breakfast_diff
  lunch_cost - meals.sandwich_cost

/-- Theorem stating that the cost of the milk carton is $1.15 -/
theorem milk_cost_is_1_15 (meals : AnnasMeals) 
  (h1 : meals.bagel_cost = 95/100)
  (h2 : meals.juice_cost = 85/100)
  (h3 : meals.sandwich_cost = 465/100)
  (h4 : meals.lunch_breakfast_diff = 4) :
  milk_cost meals = 115/100 := by
  sorry

end NUMINAMATH_CALUDE_milk_cost_is_1_15_l1759_175955


namespace NUMINAMATH_CALUDE_triangle_altitude_excircle_radii_inequality_l1759_175935

/-- Given a triangle ABC with sides a, b, and c, altitude mc from vertex C to side AB,
    and radii ra and rb of the excircles opposite to vertices A and B respectively,
    prove that the altitude mc is at most the geometric mean of ra and rb. -/
theorem triangle_altitude_excircle_radii_inequality 
  (a b c mc ra rb : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ mc > 0 ∧ ra > 0 ∧ rb > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_altitude : mc = (2 * (a * b * c).sqrt) / (a + b + c)) 
  (h_excircle_a : ra = (a * b * c).sqrt / (b + c - a)) 
  (h_excircle_b : rb = (a * b * c).sqrt / (a + c - b)) : 
  mc ≤ Real.sqrt (ra * rb) :=
sorry

end NUMINAMATH_CALUDE_triangle_altitude_excircle_radii_inequality_l1759_175935


namespace NUMINAMATH_CALUDE_polynomial_problem_l1759_175949

theorem polynomial_problem (n : ℕ) (p : ℝ → ℝ) : 
  (∀ k : ℕ, k ≤ n → p (2 * k) = 0) →
  (∀ k : ℕ, k < n → p (2 * k + 1) = 2) →
  p (2 * n + 1) = -30 →
  (∃ c : ℝ → ℝ, ∀ x, p x = c x * x * (x - 2)^2 * (x - 4)) →
  (n = 2 ∧ ∀ x, p x = -2/3 * x * (x - 2)^2 * (x - 4)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_problem_l1759_175949


namespace NUMINAMATH_CALUDE_unique_pair_divisibility_l1759_175948

theorem unique_pair_divisibility (a b : ℕ) : 
  (7^a - 3^b) ∣ (a^4 + b^2) → a = 2 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_divisibility_l1759_175948


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l1759_175922

theorem fourth_root_equation_solution (x : ℝ) :
  (x * (x^4)^(1/2))^(1/4) = 4 → x = 2^(8/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l1759_175922


namespace NUMINAMATH_CALUDE_banana_permutations_l1759_175980

theorem banana_permutations : 
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  (total_letters.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
sorry

end NUMINAMATH_CALUDE_banana_permutations_l1759_175980


namespace NUMINAMATH_CALUDE_product_of_differences_divisible_by_twelve_l1759_175956

theorem product_of_differences_divisible_by_twelve 
  (a b c d : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  12 ∣ ((a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_differences_divisible_by_twelve_l1759_175956


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1759_175983

-- Define the quadratic function f
def f : ℝ → ℝ := λ x => x^2 - x + 1

-- State the theorem
theorem quadratic_function_properties :
  (f 0 = 1) ∧
  (∀ x, f (x + 1) - f x = 2 * x) ∧
  (f = λ x => x^2 - x + 1) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≥ 3/4) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3/4) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1759_175983


namespace NUMINAMATH_CALUDE_expression_values_l1759_175936

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := (a / abs a) + (b / abs b) + (c / abs c) + (d / abs d) + ((a * b * c * d) / abs (a * b * c * d))
  expr = 5 ∨ expr = 1 ∨ expr = -3 ∨ expr = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l1759_175936


namespace NUMINAMATH_CALUDE_line_parameterization_l1759_175924

/-- Given a line y = 2x - 30 parameterized by (x,y) = (f(t), 20t - 14), prove that f(t) = 10t + 8 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 2 * f t - 30 = 20 * t - 14) → 
  (∀ t : ℝ, f t = 10 * t + 8) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l1759_175924


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l1759_175966

/-- The number of trailing zeros in n -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- 20 raised to the power of 50 -/
def a : ℕ := 20^50

/-- 50 raised to the power of 20 -/
def b : ℕ := 50^20

/-- The main theorem stating that the number of trailing zeros
    in the product of 20^50 and 50^20 is 90 -/
theorem product_trailing_zeros : trailingZeros (a * b) = 90 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l1759_175966


namespace NUMINAMATH_CALUDE_bottle_capacity_proof_l1759_175903

theorem bottle_capacity_proof (x : ℚ) : 
  (16/3 : ℚ) / 8 * x + 16/3 = 8 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottle_capacity_proof_l1759_175903


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1759_175996

/-- The constant term in the expansion of (√x + 3/x)^12 -/
def constantTerm : ℕ := 40095

/-- The binomial coefficient (12 choose 8) -/
def binomialCoeff : ℕ := 495

theorem constant_term_expansion :
  constantTerm = binomialCoeff * 3^4 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1759_175996


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1759_175946

theorem quadratic_roots_relation : 
  ∀ x₁ x₂ : ℝ, 
  (x₁^2 - 2*x₁ - 4 = 0) → 
  (x₂^2 - 2*x₂ - 4 = 0) → 
  x₁ ≠ x₂ →
  (x₁ + x₂) / (x₁ * x₂) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1759_175946


namespace NUMINAMATH_CALUDE_circle_ellipse_tangent_l1759_175901

-- Define the circle M
def circle_M (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*m*x - 3 = 0

-- Define the ellipse C
def ellipse_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1

-- Define the focus F
def focus_F (c : ℝ) : ℝ × ℝ :=
  (-c, 0)

-- Define the line l
def line_l (c : ℝ) (x y : ℝ) : Prop :=
  x = -c

theorem circle_ellipse_tangent (m a c : ℝ) :
  m < 0 →
  (∀ x y, circle_M m x y → (x - 1)^2 + y^2 = 4) →
  (∃ x y, circle_M m x y ∧ line_l c x y) →
  (∀ x y, circle_M m x y ∧ line_l c x y → (x - 1)^2 + y^2 = 4) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_ellipse_tangent_l1759_175901


namespace NUMINAMATH_CALUDE_parallel_vectors_x_values_l1759_175918

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_values :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (x - 1, 2)
  are_parallel a b → x = -1 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_values_l1759_175918


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1759_175982

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a₁ d : ℝ, ∀ n : ℕ, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 5 + a 10 = 12 →
  3 * a 7 + a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1759_175982


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1759_175999

theorem sqrt_sum_equality : ∃ (a b c : ℕ+), 
  (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) + Real.sqrt 11 * Real.sqrt 3 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) + Real.sqrt 11 * Real.sqrt 3 = (a' * Real.sqrt 3 + b' * Real.sqrt 11) / c') → 
    c ≤ c') ∧
  a = 84 ∧ b = 44 ∧ c = 33 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1759_175999


namespace NUMINAMATH_CALUDE_double_angle_sine_fifteen_degrees_l1759_175971

theorem double_angle_sine_fifteen_degrees :
  2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_double_angle_sine_fifteen_degrees_l1759_175971


namespace NUMINAMATH_CALUDE_final_water_percentage_l1759_175905

/-- Calculates the final percentage of water in a mixture after adding water -/
theorem final_water_percentage
  (initial_mixture : ℝ)
  (initial_water_percentage : ℝ)
  (added_water : ℝ)
  (h_initial_mixture : initial_mixture = 50)
  (h_initial_water_percentage : initial_water_percentage = 10)
  (h_added_water : added_water = 25) :
  let initial_water := initial_mixture * (initial_water_percentage / 100)
  let final_water := initial_water + added_water
  let final_mixture := initial_mixture + added_water
  (final_water / final_mixture) * 100 = 40 := by
sorry


end NUMINAMATH_CALUDE_final_water_percentage_l1759_175905


namespace NUMINAMATH_CALUDE_min_value_exp_ln_squared_l1759_175926

theorem min_value_exp_ln_squared (a : ℝ) (b : ℝ) (h : b > 0) :
  (Real.exp a - Real.log b)^2 + (a - b)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exp_ln_squared_l1759_175926


namespace NUMINAMATH_CALUDE_rice_dumpling_max_profit_l1759_175928

/-- A structure representing the rice dumpling problem -/
structure RiceDumplingProblem where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  selling_price_A : ℝ
  selling_price_B : ℝ
  total_purchase_cost : ℝ
  total_boxes : ℕ

/-- The profit function for the rice dumpling problem -/
def profit (p : RiceDumplingProblem) (x y : ℕ) : ℝ :=
  (p.selling_price_A - p.purchase_price_A) * x + (p.selling_price_B - p.purchase_price_B) * y

/-- Theorem stating the maximum profit for the rice dumpling problem -/
theorem rice_dumpling_max_profit (p : RiceDumplingProblem) 
  (h1 : p.purchase_price_A = 25)
  (h2 : p.purchase_price_B = 30)
  (h3 : p.selling_price_A = 32)
  (h4 : p.selling_price_B = 40)
  (h5 : p.total_purchase_cost = 1500)
  (h6 : p.total_boxes = 60) :
  ∃ (x y : ℕ), x + y = p.total_boxes ∧ x ≥ 2 * y ∧ profit p x y = 480 ∧ 
  ∀ (a b : ℕ), a + b = p.total_boxes → a ≥ 2 * b → profit p a b ≤ 480 :=
by sorry

#check rice_dumpling_max_profit

end NUMINAMATH_CALUDE_rice_dumpling_max_profit_l1759_175928


namespace NUMINAMATH_CALUDE_product_cost_l1759_175977

/-- The cost of a product given its selling price and profit margin -/
theorem product_cost (x a : ℝ) (h : a > 0) :
  let selling_price := x
  let profit_margin := a / 100
  selling_price = (1 + profit_margin) * (selling_price / (1 + profit_margin)) :=
by sorry

end NUMINAMATH_CALUDE_product_cost_l1759_175977


namespace NUMINAMATH_CALUDE_abs_value_complex_l1759_175995

/-- The absolute value of ((1+i)³)/2 is equal to √2 -/
theorem abs_value_complex : Complex.abs ((1 + Complex.I) ^ 3 / 2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_complex_l1759_175995


namespace NUMINAMATH_CALUDE_tomato_seed_planting_l1759_175904

theorem tomato_seed_planting (mike_morning mike_afternoon ted_morning ted_afternoon total : ℕ) : 
  mike_morning = 50 →
  ted_morning = 2 * mike_morning →
  mike_afternoon = 60 →
  ted_afternoon < mike_afternoon →
  total = mike_morning + ted_morning + mike_afternoon + ted_afternoon →
  total = 250 →
  mike_afternoon - ted_afternoon = 20 := by
sorry

end NUMINAMATH_CALUDE_tomato_seed_planting_l1759_175904


namespace NUMINAMATH_CALUDE_quadratic_square_completion_l1759_175964

theorem quadratic_square_completion (a b c : ℤ) : 
  (∀ x : ℝ, 64 * x^2 - 96 * x - 48 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 86 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_square_completion_l1759_175964


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_81_l1759_175960

theorem arithmetic_square_root_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_81_l1759_175960


namespace NUMINAMATH_CALUDE_pool_water_volume_l1759_175930

/-- Calculates the remaining water volume in a pool after evaporation --/
def remaining_water_volume (initial_volume : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_volume - evaporation_rate * days

/-- Theorem: The remaining water volume after 45 days is 355 gallons --/
theorem pool_water_volume : 
  remaining_water_volume 400 1 45 = 355 := by
  sorry

end NUMINAMATH_CALUDE_pool_water_volume_l1759_175930


namespace NUMINAMATH_CALUDE_carols_rectangle_width_l1759_175913

theorem carols_rectangle_width (carol_length jordan_length jordan_width : ℝ) 
  (h1 : carol_length = 15)
  (h2 : jordan_length = 6)
  (h3 : jordan_width = 50)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  carol_width = 20 := by
  sorry

end NUMINAMATH_CALUDE_carols_rectangle_width_l1759_175913


namespace NUMINAMATH_CALUDE_infinitely_many_common_terms_l1759_175992

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 14 * a (n + 1) + a n

/-- Sequence b_n defined recursively -/
def b : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 6 * b (n + 1) - b n

/-- There are infinitely many common terms in sequences a and b -/
theorem infinitely_many_common_terms :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ a (2 * k + 1) = b (3 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_common_terms_l1759_175992


namespace NUMINAMATH_CALUDE_cat_eye_movement_l1759_175968

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the transformation (moving 3 units to the right)
def moveRight (p : Point) : Point :=
  (p.1 + 3, p.2)

-- Define the original points
def eye1 : Point := (-4, 3)
def eye2 : Point := (-2, 3)

-- State the theorem
theorem cat_eye_movement :
  (moveRight eye1 = (-1, 3)) ∧ (moveRight eye2 = (1, 3)) := by
  sorry

end NUMINAMATH_CALUDE_cat_eye_movement_l1759_175968


namespace NUMINAMATH_CALUDE_magician_card_decks_l1759_175984

theorem magician_card_decks (price : ℕ) (decks_left : ℕ) (earnings : ℕ) : 
  price = 7 → decks_left = 8 → earnings = 56 → 
  ∃ (initial_decks : ℕ), initial_decks = decks_left + earnings / price :=
by sorry

end NUMINAMATH_CALUDE_magician_card_decks_l1759_175984


namespace NUMINAMATH_CALUDE_order_of_expressions_l1759_175912

theorem order_of_expressions : 
  let a : ℝ := (0.2 : ℝ)^2
  let b : ℝ := 2^(0.3 : ℝ)
  let c : ℝ := Real.log 2 / Real.log 0.2
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l1759_175912


namespace NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l1759_175979

/-- Given a principal amount and an interest rate satisfying the given conditions,
    prove that the interest rate is 10% --/
theorem interest_rate_is_ten_percent (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 2420)
  (h2 : P * (1 + r)^3 = 2662) :
  r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l1759_175979


namespace NUMINAMATH_CALUDE_work_increase_with_absence_l1759_175958

theorem work_increase_with_absence (p : ℕ) (W : ℝ) (h : p > 0) :
  let original_work := W / p
  let remaining_workers := (3 : ℝ) / 4 * p
  let new_work := W / remaining_workers
  new_work - original_work = (1 : ℝ) / 3 * original_work :=
by sorry

end NUMINAMATH_CALUDE_work_increase_with_absence_l1759_175958


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l1759_175934

theorem base_2_representation_of_123 : 
  ∃ (a b c d e f g : ℕ), 
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1) ∧
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l1759_175934


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1759_175974

def U : Set ℕ := {1,2,3,4,5,6,7}
def A : Set ℕ := {1,3,5,7}

theorem complement_of_A_in_U : 
  (U \ A) = {2,4,6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1759_175974


namespace NUMINAMATH_CALUDE_language_group_partition_l1759_175925

/-- Represents a student and the languages they speak -/
structure Student where
  speaks_english : Bool
  speaks_french : Bool
  speaks_spanish : Bool

/-- Represents a group of students -/
def StudentGroup := List Student

/-- Check if a group satisfies the language requirements -/
def validGroup (group : StudentGroup) : Bool :=
  (group.filter (·.speaks_english)).length = 10 ∧
  (group.filter (·.speaks_french)).length = 10 ∧
  (group.filter (·.speaks_spanish)).length = 10

theorem language_group_partition (students : List Student) :
  (students.filter (·.speaks_english)).length = 50 →
  (students.filter (·.speaks_french)).length = 50 →
  (students.filter (·.speaks_spanish)).length = 50 →
  ∃ (groups : List StudentGroup), 
    groups.length = 5 ∧ 
    (groups.join = students) ∧
    (∀ group ∈ groups, validGroup group) :=
by
  sorry

end NUMINAMATH_CALUDE_language_group_partition_l1759_175925


namespace NUMINAMATH_CALUDE_arc_cover_theorem_l1759_175972

/-- Represents an arc on a circle -/
structure Arc where
  start : ℝ  -- Start angle in degrees
  length : ℝ  -- Length of the arc in degrees

/-- A set of arcs covering a circle -/
def ArcCover := Set Arc

/-- Predicate to check if a set of arcs covers the entire circle -/
def covers_circle (cover : ArcCover) : Prop := sorry

/-- Predicate to check if any single arc in the set covers the entire circle -/
def has_complete_arc (cover : ArcCover) : Prop := sorry

/-- Calculate the total measure of a set of arcs -/
def total_measure (arcs : Set Arc) : ℝ := sorry

/-- Main theorem -/
theorem arc_cover_theorem (cover : ArcCover) 
  (h1 : covers_circle cover) 
  (h2 : ¬ has_complete_arc cover) : 
  ∃ (subset : Set Arc), subset ⊆ cover ∧ covers_circle subset ∧ total_measure subset ≤ 720 := by
  sorry

end NUMINAMATH_CALUDE_arc_cover_theorem_l1759_175972


namespace NUMINAMATH_CALUDE_line_does_not_intersect_curve_l1759_175933

/-- The function representing the curve y = (|x|-1)/(|x-1|) -/
noncomputable def f (x : ℝ) : ℝ := (abs x - 1) / (abs (x - 1))

/-- The theorem stating the condition for non-intersection -/
theorem line_does_not_intersect_curve (m : ℝ) :
  (∀ x : ℝ, m * x ≠ f x) ↔ (-1 ≤ m ∧ m < -3 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_line_does_not_intersect_curve_l1759_175933


namespace NUMINAMATH_CALUDE_job_crop_production_l1759_175941

/-- Represents the land allocation of Job's farm --/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  future_expansion : ℕ
  cattle : ℕ

/-- Calculates the land used for crop production --/
def crop_production (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.future_expansion + farm.cattle)

/-- Theorem stating that Job's land used for crop production is 70 hectares --/
theorem job_crop_production :
  let job_farm : FarmLand := {
    total := 150,
    house_and_machinery := 25,
    future_expansion := 15,
    cattle := 40
  }
  crop_production job_farm = 70 := by sorry

end NUMINAMATH_CALUDE_job_crop_production_l1759_175941


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l1759_175998

def B_inverse : Matrix (Fin 2) (Fin 2) ℝ :=
  !![3, 4;
    -2, -3]

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = B_inverse) : 
  (B^3)⁻¹ = B_inverse := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l1759_175998


namespace NUMINAMATH_CALUDE_medals_award_ways_l1759_175989

-- Define the total number of sprinters
def total_sprinters : ℕ := 10

-- Define the number of Canadian sprinters
def canadian_sprinters : ℕ := 4

-- Define the number of non-Canadian sprinters
def non_canadian_sprinters : ℕ := total_sprinters - canadian_sprinters

-- Define the number of medals
def num_medals : ℕ := 3

-- Function to calculate the number of ways to award medals
def ways_to_award_medals : ℕ := 
  -- Case 1: No Canadians get a medal
  (non_canadian_sprinters * (non_canadian_sprinters - 1) * (non_canadian_sprinters - 2)) + 
  -- Case 2: Exactly one Canadian gets a medal
  (canadian_sprinters * num_medals * (non_canadian_sprinters) * (non_canadian_sprinters - 1))

-- Theorem statement
theorem medals_award_ways : 
  ways_to_award_medals = 360 := by sorry

end NUMINAMATH_CALUDE_medals_award_ways_l1759_175989


namespace NUMINAMATH_CALUDE_viewers_of_program_A_l1759_175927

theorem viewers_of_program_A (total_viewers : ℕ) (ratio_both ratio_A ratio_B : ℕ) : 
  total_viewers = 560 →
  ratio_both = 1 →
  ratio_A = 2 →
  ratio_B = 3 →
  (ratio_both + ratio_A + ratio_B) * (ratio_both + ratio_A) * total_viewers / ((ratio_both + ratio_A + ratio_B) * (ratio_both + ratio_A + ratio_B)) = 280 :=
by sorry

end NUMINAMATH_CALUDE_viewers_of_program_A_l1759_175927


namespace NUMINAMATH_CALUDE_chord_length_square_of_quarter_circle_l1759_175952

/-- Given a circular sector with central angle 90° and radius 10 cm,
    the square of the chord length connecting the arc endpoints is 200 cm². -/
theorem chord_length_square_of_quarter_circle (r : ℝ) (h : r = 10) :
  let chord_length_square := 2 * r^2
  chord_length_square = 200 := by sorry

end NUMINAMATH_CALUDE_chord_length_square_of_quarter_circle_l1759_175952


namespace NUMINAMATH_CALUDE_oxen_count_l1759_175917

/-- The number of oxen in the first group that can plough 1/7th of a field in 2 days. -/
def first_group : ℕ := sorry

/-- The time it takes for the first group to plough the entire field. -/
def total_time : ℕ := 14

/-- The fraction of the field ploughed by the first group. -/
def ploughed_fraction : ℚ := 1/7

/-- The number of oxen in the second group. -/
def second_group : ℕ := 18

/-- The time it takes for the second group to plough the remaining field. -/
def remaining_time : ℕ := 20

/-- The fraction of the field ploughed by the second group. -/
def remaining_fraction : ℚ := 6/7

theorem oxen_count :
  (first_group * total_time) / 1 = (second_group * remaining_time) / remaining_fraction →
  first_group = 30 := by sorry

end NUMINAMATH_CALUDE_oxen_count_l1759_175917


namespace NUMINAMATH_CALUDE_hans_reservation_deposit_l1759_175973

/-- Calculates the total deposit for a restaurant reservation with given guest counts and fees -/
def calculate_deposit (num_kids num_adults num_seniors num_students num_employees : ℕ)
  (flat_fee kid_fee adult_fee senior_fee student_fee employee_fee : ℚ)
  (service_charge_rate : ℚ) : ℚ :=
  let base_deposit := flat_fee + 
    num_kids * kid_fee + 
    num_adults * adult_fee + 
    num_seniors * senior_fee + 
    num_students * student_fee + 
    num_employees * employee_fee
  let service_charge := base_deposit * service_charge_rate
  base_deposit + service_charge

/-- The total deposit for Hans' reservation is $128.63 -/
theorem hans_reservation_deposit :
  calculate_deposit 2 8 5 3 2 30 3 6 4 (9/2) (5/2) (1/20) = 12863/100 := by
  sorry

end NUMINAMATH_CALUDE_hans_reservation_deposit_l1759_175973


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_l1759_175976

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of a point on the ellipse -/
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse_C P.1 P.2

/-- Definition of the right focus -/
def right_focus : ℝ × ℝ := (1, 0)

/-- Definition of a line passing through the right focus -/
def line_through_focus (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), A = right_focus + t • (B - right_focus) ∨
             B = right_focus + t • (A - right_focus)

/-- Definition of the dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem to be proved -/
theorem ellipse_fixed_point :
  ∃ (M : ℝ × ℝ), M.1 = 5/4 ∧ M.2 = 0 ∧
  ∀ (A B : ℝ × ℝ), point_on_ellipse A → point_on_ellipse B →
  line_through_focus A B →
  dot_product (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2) = -7/16 :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_l1759_175976


namespace NUMINAMATH_CALUDE_evaluate_expression_l1759_175997

theorem evaluate_expression : (4^150 * 9^152) / 6^301 = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1759_175997


namespace NUMINAMATH_CALUDE_milk_mixture_price_l1759_175991

/-- Calculate the selling price of a milk-water mixture per litre -/
theorem milk_mixture_price (pure_milk_cost : ℝ) (pure_milk_volume : ℝ) (water_volume : ℝ) :
  pure_milk_cost = 3.60 →
  pure_milk_volume = 25 →
  water_volume = 5 →
  (pure_milk_cost * pure_milk_volume) / (pure_milk_volume + water_volume) = 3 := by
sorry


end NUMINAMATH_CALUDE_milk_mixture_price_l1759_175991


namespace NUMINAMATH_CALUDE_circle_radius_with_tangent_l1759_175908

/-- The radius of a circle with equation x^2 + y^2 = 25 and a tangent at y = 5 is 5 -/
theorem circle_radius_with_tangent (x y : ℝ) :
  x^2 + y^2 = 25 → ∃ (x₀ : ℝ), x₀^2 + 5^2 = 25 → 
  Real.sqrt ((0 - x₀)^2 + (5 - 0)^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_with_tangent_l1759_175908


namespace NUMINAMATH_CALUDE_jellybean_count_l1759_175990

/-- The number of jellybeans needed to fill a large drinking glass -/
def large_glass : ℕ := sorry

/-- The number of jellybeans needed to fill a small drinking glass -/
def small_glass : ℕ := sorry

/-- The total number of large glasses -/
def num_large_glasses : ℕ := 5

/-- The total number of small glasses -/
def num_small_glasses : ℕ := 3

/-- The total number of jellybeans needed to fill all glasses -/
def total_jellybeans : ℕ := 325

theorem jellybean_count :
  (small_glass = large_glass / 2) →
  (num_large_glasses * large_glass + num_small_glasses * small_glass = total_jellybeans) →
  large_glass = 50 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l1759_175990


namespace NUMINAMATH_CALUDE_most_probable_occurrences_l1759_175932

theorem most_probable_occurrences (p : ℝ) (k₀ : ℕ) (h_p : p = 0.4) (h_k₀ : k₀ = 25) :
  ∃ n : ℕ, 62 ≤ n ∧ n ≤ 64 ∧
  (∀ m : ℕ, (m * p - (1 - p) ≤ k₀ ∧ k₀ < m * p + p) → m = n) :=
by sorry

end NUMINAMATH_CALUDE_most_probable_occurrences_l1759_175932


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1759_175921

theorem dice_roll_probability : 
  let num_dice : ℕ := 8
  let num_sides : ℕ := 6
  num_dice > num_sides →
  (probability_at_least_two_same : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1759_175921


namespace NUMINAMATH_CALUDE_binomial_sum_of_even_coefficients_l1759_175907

theorem binomial_sum_of_even_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = 121 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_of_even_coefficients_l1759_175907


namespace NUMINAMATH_CALUDE_matthew_score_proof_l1759_175951

def basket_value : ℕ := 3
def total_baskets : ℕ := 5
def shawn_points : ℕ := 6

def matthew_points : ℕ := 9

theorem matthew_score_proof :
  matthew_points = basket_value * total_baskets - shawn_points :=
by sorry

end NUMINAMATH_CALUDE_matthew_score_proof_l1759_175951


namespace NUMINAMATH_CALUDE_median_sum_ge_four_circumradius_l1759_175965

/-- A triangle is represented by its three vertices in the real plane -/
structure Triangle where
  A : Real × Real
  B : Real × Real
  C : Real × Real

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : Triangle) : Real := sorry

/-- The length of a median in a triangle -/
noncomputable def median_length (t : Triangle) (vertex : Fin 3) : Real := sorry

/-- Predicate to check if a triangle is not obtuse -/
def is_not_obtuse (t : Triangle) : Prop := sorry

/-- Theorem: For any non-obtuse triangle, the sum of its median lengths
    is greater than or equal to four times its circumradius -/
theorem median_sum_ge_four_circumradius (t : Triangle) 
  (h : is_not_obtuse t) : 
  (median_length t 0) + (median_length t 1) + (median_length t 2) ≥ 4 * (circumradius t) := by
  sorry

end NUMINAMATH_CALUDE_median_sum_ge_four_circumradius_l1759_175965


namespace NUMINAMATH_CALUDE_recurring_fraction_equality_l1759_175950

-- Define the recurring decimal 0.812812...
def recurring_812 : ℚ := 812 / 999

-- Define the recurring decimal 2.406406...
def recurring_2406 : ℚ := 2404 / 999

-- Theorem statement
theorem recurring_fraction_equality : 
  recurring_812 / recurring_2406 = 203 / 601 := by
  sorry

end NUMINAMATH_CALUDE_recurring_fraction_equality_l1759_175950
