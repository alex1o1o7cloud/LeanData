import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_four_fourth_power_sum_l3501_350135

theorem sqrt_four_fourth_power_sum : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_fourth_power_sum_l3501_350135


namespace NUMINAMATH_CALUDE_pages_per_notepad_l3501_350147

/-- Proves that given the total cost, cost per notepad, and total pages,
    the number of pages per notepad can be determined. -/
theorem pages_per_notepad
  (total_cost : ℝ)
  (cost_per_notepad : ℝ)
  (total_pages : ℕ)
  (h1 : total_cost = 10)
  (h2 : cost_per_notepad = 1.25)
  (h3 : total_pages = 480) :
  (total_pages : ℝ) / (total_cost / cost_per_notepad) = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_pages_per_notepad_l3501_350147


namespace NUMINAMATH_CALUDE_monotonicity_condition_l3501_350172

/-- The function f(x) = √(x² + 1) - ax is monotonic on [0,+∞) if and only if a ≥ 1, given that a > 0 -/
theorem monotonicity_condition (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → (Real.sqrt (x^2 + 1) - a * x < Real.sqrt (y^2 + 1) - a * y ∨
                               Real.sqrt (x^2 + 1) - a * x > Real.sqrt (y^2 + 1) - a * y)) ↔
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l3501_350172


namespace NUMINAMATH_CALUDE_teachers_at_queen_high_school_l3501_350120

-- Define the given conditions
def total_students : ℕ := 1500
def classes_per_student : ℕ := 6
def classes_per_teacher : ℕ := 5
def students_per_class : ℕ := 35

-- Define the theorem
theorem teachers_at_queen_high_school :
  (total_students * classes_per_student / students_per_class + 4) / classes_per_teacher = 52 := by
  sorry


end NUMINAMATH_CALUDE_teachers_at_queen_high_school_l3501_350120


namespace NUMINAMATH_CALUDE_rotate_angle_result_l3501_350137

/-- Given an initial angle of 30 degrees and a 450-degree counterclockwise rotation,
    the resulting acute angle measures 60 degrees. -/
theorem rotate_angle_result (initial_angle rotation : ℝ) (h1 : initial_angle = 30)
    (h2 : rotation = 450) : 
    (initial_angle + rotation) % 360 = 60 ∨ 360 - (initial_angle + rotation) % 360 = 60 :=
by sorry

end NUMINAMATH_CALUDE_rotate_angle_result_l3501_350137


namespace NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l3501_350113

/-- The 'Twirly Tea Cups' ride problem -/
theorem twirly_tea_cups_capacity 
  (total_capacity : ℕ) 
  (num_teacups : ℕ) 
  (h1 : total_capacity = 63) 
  (h2 : num_teacups = 7) : 
  total_capacity / num_teacups = 9 := by
  sorry

end NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l3501_350113


namespace NUMINAMATH_CALUDE_min_value_f_in_interval_l3501_350116

def f (x : ℝ) : ℝ := x^4 - 4*x + 3

theorem min_value_f_in_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 3 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2 : ℝ) 3 → f x ≤ f y) ∧
  f x = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_in_interval_l3501_350116


namespace NUMINAMATH_CALUDE_inverse_proportion_point_order_l3501_350185

/-- Prove that for points on an inverse proportion function, their y-coordinates follow a specific order -/
theorem inverse_proportion_point_order (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h_pos : k > 0)
  (h_A : y₁ = k / (-1))
  (h_B : y₂ = k / 2)
  (h_C : y₃ = k / 3) :
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_order_l3501_350185


namespace NUMINAMATH_CALUDE_intersection_A_B_l3501_350142

def A : Set ℝ := {x | (x + 3) * (2 - x) > 0}
def B : Set ℝ := {-5, -4, 0, 1, 4}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3501_350142


namespace NUMINAMATH_CALUDE_line_equation_in_triangle_l3501_350124

/-- Given a line passing through (-2b, 0) forming a triangular region in the second quadrant with area S, 
    its equation is 2Sx - b^2y + 4bS = 0 --/
theorem line_equation_in_triangle (b S : ℝ) (h_b : b ≠ 0) (h_S : S > 0) : 
  ∃ (m k : ℝ), 
    (∀ (x y : ℝ), y = m * x + k → 
      (x = -2*b ∧ y = 0) ∨ 
      (x = 0 ∧ y = 0) ∨ 
      (x = 0 ∧ y > 0)) ∧
    (1/2 * 2*b * (S/b) = S) ∧
    (∀ (x y : ℝ), 2*S*x - b^2*y + 4*b*S = 0 ↔ y = m * x + k) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_in_triangle_l3501_350124


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l3501_350158

/-- Given a square C with perimeter 40 cm and a square D with area equal to one-third the area of square C, 
    the perimeter of square D is (40√3)/3 cm. -/
theorem square_perimeter_relation (C D : Real) : 
  (C * 4 = 40) →  -- Perimeter of square C is 40 cm
  (D^2 = (C^2) / 3) →  -- Area of square D is one-third the area of square C
  (D * 4 = 40 * Real.sqrt 3 / 3) :=  -- Perimeter of square D is (40√3)/3 cm
by sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l3501_350158


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3501_350180

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

/-- The main theorem stating that functions satisfying the equation are either the identity or absolute value function. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = |x|) := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l3501_350180


namespace NUMINAMATH_CALUDE_nicole_cookies_l3501_350104

theorem nicole_cookies (N : ℚ) : 
  (((1 - N) * (1 - 3/5)) = 6/25) → N = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_nicole_cookies_l3501_350104


namespace NUMINAMATH_CALUDE_inequality_proof_l3501_350184

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3501_350184


namespace NUMINAMATH_CALUDE_average_rate_of_change_f_on_0_2_l3501_350122

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the interval [0, 2]
def a : ℝ := 0
def b : ℝ := 2

-- State the theorem
theorem average_rate_of_change_f_on_0_2 :
  (f b - f a) / (b - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_f_on_0_2_l3501_350122


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l3501_350118

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | 2*x < 2}

theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l3501_350118


namespace NUMINAMATH_CALUDE_same_color_probability_l3501_350170

/-- The probability of drawing 2 balls of the same color from a bag -/
theorem same_color_probability (total : ℕ) (red : ℕ) (yellow : ℕ) (blue : ℕ) :
  total = red + yellow + blue →
  red = 3 →
  yellow = 2 →
  blue = 1 →
  (Nat.choose red 2 + Nat.choose yellow 2) / Nat.choose total 2 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3501_350170


namespace NUMINAMATH_CALUDE_boys_tried_out_l3501_350107

/-- The number of boys who tried out for the basketball team -/
def num_boys : ℕ := sorry

/-- The number of girls who tried out for the basketball team -/
def num_girls : ℕ := 39

/-- The number of students who got called back -/
def called_back : ℕ := 26

/-- The number of students who didn't make the cut -/
def didnt_make_cut : ℕ := 17

theorem boys_tried_out : num_boys = 4 := by
  sorry

end NUMINAMATH_CALUDE_boys_tried_out_l3501_350107


namespace NUMINAMATH_CALUDE_princes_wish_fulfilled_l3501_350175

/-- Represents a knight at the round table -/
structure Knight where
  city : Nat
  hasGoldGoblet : Bool

/-- The state of the round table at any given moment -/
def RoundTable := Vector Knight 13

/-- Checks if two knights from the same city both have gold goblets -/
def sameCity2GoldGoblets (table : RoundTable) : Bool :=
  sorry

/-- Passes goblets to the right -/
def passGoblets (table : RoundTable) : RoundTable :=
  sorry

/-- The main theorem to be proved -/
theorem princes_wish_fulfilled (k : Nat) (h1 : 1 < k) (h2 : k < 13)
  (initial_table : RoundTable)
  (h3 : (initial_table.toList.filter Knight.hasGoldGoblet).length = k)
  (h4 : (initial_table.toList.map Knight.city).toFinset.card = k) :
  ∃ n : Nat, sameCity2GoldGoblets (n.iterate passGoblets initial_table) := by
  sorry

end NUMINAMATH_CALUDE_princes_wish_fulfilled_l3501_350175


namespace NUMINAMATH_CALUDE_chef_cakes_problem_l3501_350138

def chef_cakes (total_eggs : ℕ) (fridge_eggs : ℕ) (eggs_per_cake : ℕ) : ℕ :=
  (total_eggs - fridge_eggs) / eggs_per_cake

theorem chef_cakes_problem :
  chef_cakes 60 10 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chef_cakes_problem_l3501_350138


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l3501_350197

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ), 3 * x - 4 * y = -7 ∧ 6 * x - 5 * y = 5 :=
by
  use 7, 7
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l3501_350197


namespace NUMINAMATH_CALUDE_angle_subtraction_l3501_350129

/-- Represents an angle in degrees, minutes, and seconds -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

/-- Converts an Angle to seconds -/
def angleToSeconds (a : Angle) : ℕ :=
  a.degrees * 3600 + a.minutes * 60 + a.seconds

/-- Converts seconds to an Angle -/
def secondsToAngle (s : ℕ) : Angle :=
  let d := s / 3600
  let m := (s % 3600) / 60
  let sec := s % 60
  ⟨d, m, sec⟩

theorem angle_subtraction :
  let a₁ : Angle := ⟨90, 0, 0⟩
  let a₂ : Angle := ⟨78, 28, 56⟩
  let result : Angle := ⟨11, 31, 4⟩
  angleToSeconds a₁ - angleToSeconds a₂ = angleToSeconds result := by
  sorry

end NUMINAMATH_CALUDE_angle_subtraction_l3501_350129


namespace NUMINAMATH_CALUDE_area_bounded_region_l3501_350145

/-- The area of a region bounded by horizontal lines, a vertical line, and a semicircle -/
theorem area_bounded_region (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let rectangular_area := (a + b) * d
  let semicircle_area := (1 / 2) * Real.pi * c^2
  rectangular_area + semicircle_area = (a + b) * d + (1 / 2) * Real.pi * c^2 := by
  sorry

end NUMINAMATH_CALUDE_area_bounded_region_l3501_350145


namespace NUMINAMATH_CALUDE_group_trip_cost_l3501_350187

/-- The total cost for a group trip given the number of people and cost per person -/
def total_cost (num_people : ℕ) (cost_per_person : ℕ) : ℕ :=
  num_people * cost_per_person

/-- Proof that the total cost for 11 people at $1100 each is $12100 -/
theorem group_trip_cost : total_cost 11 1100 = 12100 := by
  sorry

end NUMINAMATH_CALUDE_group_trip_cost_l3501_350187


namespace NUMINAMATH_CALUDE_monicas_savings_l3501_350155

theorem monicas_savings (weekly_savings : ℕ) (weeks_to_fill : ℕ) (repetitions : ℕ) : 
  weekly_savings = 15 → weeks_to_fill = 60 → repetitions = 5 →
  weekly_savings * weeks_to_fill * repetitions = 4500 := by
  sorry

end NUMINAMATH_CALUDE_monicas_savings_l3501_350155


namespace NUMINAMATH_CALUDE_pyramid_sphere_inequality_l3501_350115

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- The radius of the circumscribed sphere -/
  R : ℝ
  /-- The radius of the inscribed sphere -/
  r : ℝ
  /-- R is positive -/
  R_pos : 0 < R
  /-- r is positive -/
  r_pos : 0 < r

/-- 
For a regular quadrilateral pyramid inscribed in a sphere with radius R 
and circumscribed around a sphere with radius r, R ≥ (√2 + 1)r holds.
-/
theorem pyramid_sphere_inequality (p : RegularQuadrilateralPyramid) : 
  p.R ≥ (Real.sqrt 2 + 1) * p.r := by
  sorry

end NUMINAMATH_CALUDE_pyramid_sphere_inequality_l3501_350115


namespace NUMINAMATH_CALUDE_unit_digit_of_8_power_1533_l3501_350154

theorem unit_digit_of_8_power_1533 : (8^1533 : ℕ) % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_unit_digit_of_8_power_1533_l3501_350154


namespace NUMINAMATH_CALUDE_complex_subtraction_l3501_350196

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + 4*I) :
  a - 3*b = -1 - 15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3501_350196


namespace NUMINAMATH_CALUDE_equilateral_triangle_condition_l3501_350114

/-- A function that checks if a natural number n satisfies the condition for forming an equilateral triangle with sticks of lengths 1 to n -/
def canFormEquilateralTriangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- The sum of the first n natural numbers -/
def sumFirstN (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating the necessary and sufficient condition for forming an equilateral triangle -/
theorem equilateral_triangle_condition (n : ℕ) :
  (∃ (a b c : ℕ), a + b + c = sumFirstN n ∧ a = b ∧ b = c) ↔ canFormEquilateralTriangle n :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_condition_l3501_350114


namespace NUMINAMATH_CALUDE_complementary_fraction_irreducible_l3501_350152

theorem complementary_fraction_irreducible (a b : ℤ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : Nat.gcd a.natAbs b.natAbs = 1) : 
  Nat.gcd (b - a).natAbs b.natAbs = 1 := by
sorry

end NUMINAMATH_CALUDE_complementary_fraction_irreducible_l3501_350152


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3501_350119

theorem rectangle_area_theorem (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧
  l / w = 5 / 2 ∧
  d ^ 2 = (l / 2) ^ 2 + w ^ 2 ∧
  l * w = (5 / 13) * d ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3501_350119


namespace NUMINAMATH_CALUDE_age_ratio_sandy_molly_l3501_350192

/-- Given that Sandy is 70 years old and Molly is 20 years older than Sandy,
    prove that the ratio of their ages is 7:9. -/
theorem age_ratio_sandy_molly :
  let sandy_age : ℕ := 70
  let age_difference : ℕ := 20
  let molly_age : ℕ := sandy_age + age_difference
  (sandy_age : ℚ) / (molly_age : ℚ) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_age_ratio_sandy_molly_l3501_350192


namespace NUMINAMATH_CALUDE_diamond_area_is_50_l3501_350151

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Represents the diamond-shaped region in the square -/
structure DiamondRegion where
  square : Square
  pointA : Point
  pointB : Point

/-- The area of the diamond-shaped region in a 10x10 square -/
def diamondArea (d : DiamondRegion) : ℝ :=
  sorry

theorem diamond_area_is_50 (d : DiamondRegion) : 
  d.square.side = 10 →
  d.pointA.x = 5 ∧ d.pointA.y = 10 →
  d.pointB.x = 5 ∧ d.pointB.y = 0 →
  diamondArea d = 50 := by
  sorry

end NUMINAMATH_CALUDE_diamond_area_is_50_l3501_350151


namespace NUMINAMATH_CALUDE_fraction_simplification_l3501_350156

theorem fraction_simplification : (144 : ℚ) / 1296 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3501_350156


namespace NUMINAMATH_CALUDE_inscribed_prism_surface_area_l3501_350188

/-- The surface area of a right square prism inscribed in a sphere -/
theorem inscribed_prism_surface_area (r h : ℝ) (a : ℝ) :
  r = Real.sqrt 6 →
  h = 4 →
  2 * a^2 + h^2 = 4 * r^2 →
  2 * a^2 + 4 * a * h = 40 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_prism_surface_area_l3501_350188


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l3501_350121

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l3501_350121


namespace NUMINAMATH_CALUDE_meal_cost_45_dollars_l3501_350160

/-- The cost of a meal consisting of one pizza and three burgers -/
def meal_cost (burger_price : ℝ) : ℝ :=
  let pizza_price := 2 * burger_price
  pizza_price + 3 * burger_price

/-- Theorem: The cost of one pizza and three burgers is $45 when a burger costs $9 -/
theorem meal_cost_45_dollars :
  meal_cost 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_45_dollars_l3501_350160


namespace NUMINAMATH_CALUDE_work_completion_time_l3501_350140

/-- Given that:
  - A can do a work in 4 days
  - A and B together can finish the work in 3 days
  Prove that B can do the work alone in 12 days -/
theorem work_completion_time (a_time b_time combined_time : ℝ) 
  (ha : a_time = 4)
  (hc : combined_time = 3)
  (h_combined : 1 / a_time + 1 / b_time = 1 / combined_time) :
  b_time = 12 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3501_350140


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_relationship_l3501_350103

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (skew_lines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_relationship
  (a b : Line) (α : Plane)
  (h1 : parallel_line_plane a α)
  (h2 : contained_in_plane b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_relationship_l3501_350103


namespace NUMINAMATH_CALUDE_line_through_point_l3501_350164

/-- Given a line represented by the equation 3kx - k = -4y - 2 that contains the point (2, 1),
    prove that k = -6/5 -/
theorem line_through_point (k : ℚ) :
  (3 * k * 2 - k = -4 * 1 - 2) → k = -6/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3501_350164


namespace NUMINAMATH_CALUDE_tan_pi_sixth_minus_alpha_l3501_350101

theorem tan_pi_sixth_minus_alpha (α : ℝ) (h : Real.sin α = 3 * Real.sin (α - π / 3)) :
  Real.tan (π / 6 - α) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_sixth_minus_alpha_l3501_350101


namespace NUMINAMATH_CALUDE_absolute_value_five_l3501_350163

theorem absolute_value_five (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_five_l3501_350163


namespace NUMINAMATH_CALUDE_log_inequality_l3501_350141

theorem log_inequality (x y z : ℝ) 
  (hx : x = 6 * (Real.log 3 / Real.log 64))
  (hy : y = (1/3) * (Real.log 64 / Real.log 3))
  (hz : z = (3/2) * (Real.log 3 / Real.log 8)) :
  x > y ∧ y > z := by sorry

end NUMINAMATH_CALUDE_log_inequality_l3501_350141


namespace NUMINAMATH_CALUDE_parabola_b_value_l3501_350169

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem parabola_b_value :
  ∀ b c : ℝ,
  Parabola b c 1 = 2 →
  Parabola b c 5 = 2 →
  b = -6 := by
sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3501_350169


namespace NUMINAMATH_CALUDE_triangular_square_iff_pell_solution_l3501_350108

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- A solution to the Pell's equation X^2 - 8Y^2 = 1 -/
def pell_solution (x : ℕ) : Prop := ∃ y : ℕ, x^2 - 8*y^2 = 1

/-- The main theorem: a triangular number is a perfect square iff it has the form (x^2 - 1)/8
    where x is a solution to the Pell's equation X^2 - 8Y^2 = 1 -/
theorem triangular_square_iff_pell_solution :
  ∀ n : ℕ, (∃ k : ℕ, triangular_number n = k^2) ↔ 
  (∃ x : ℕ, pell_solution x ∧ triangular_number n = (x^2 - 1) / 8) :=
sorry

end NUMINAMATH_CALUDE_triangular_square_iff_pell_solution_l3501_350108


namespace NUMINAMATH_CALUDE_crate_height_difference_l3501_350143

/-- The height difference between two crate packing methods for cylindrical pipes -/
theorem crate_height_difference (n : ℕ) (d : ℝ) :
  let h_direct := n * d
  let h_staggered := (n / 2) * (d + d * Real.sqrt 3 / 2)
  n = 200 ∧ d = 12 →
  h_direct - h_staggered = 120 - 60 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_crate_height_difference_l3501_350143


namespace NUMINAMATH_CALUDE_seeds_per_flowerbed_is_four_l3501_350189

/-- The number of flowerbeds -/
def num_flowerbeds : ℕ := 8

/-- The total number of seeds planted -/
def total_seeds : ℕ := 32

/-- The number of seeds in each flowerbed -/
def seeds_per_flowerbed : ℕ := total_seeds / num_flowerbeds

/-- Theorem: The number of seeds per flowerbed is 4 -/
theorem seeds_per_flowerbed_is_four :
  seeds_per_flowerbed = 4 :=
by sorry

end NUMINAMATH_CALUDE_seeds_per_flowerbed_is_four_l3501_350189


namespace NUMINAMATH_CALUDE_time_difference_l3501_350166

def brian_time : ℕ := 96
def todd_time : ℕ := 88

theorem time_difference : brian_time - todd_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_l3501_350166


namespace NUMINAMATH_CALUDE_triangle_side_length_l3501_350130

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  a = 3 → b = Real.sqrt 13 → B = π / 3 → 
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3501_350130


namespace NUMINAMATH_CALUDE_min_faces_to_paint_correct_faces_to_paint_less_than_total_l3501_350149

/-- The minimum number of cube faces Vasya needs to paint to prevent Petya from assembling
    an nxnxn cube that is completely white on the outside, given n^3 white 1x1x1 cubes. -/
def min_faces_to_paint (n : ℕ) : ℕ :=
  match n with
  | 2 => 2
  | 3 => 12
  | _ => 0  -- undefined for other values of n

/-- Theorem stating the correct minimum number of faces to paint for n=2 and n=3 -/
theorem min_faces_to_paint_correct :
  (min_faces_to_paint 2 = 2) ∧ (min_faces_to_paint 3 = 12) :=
by sorry

/-- Helper function to calculate the total number of small cubes -/
def total_small_cubes (n : ℕ) : ℕ := n^3

/-- Theorem stating that the number of faces to paint is less than the total number of cube faces -/
theorem faces_to_paint_less_than_total (n : ℕ) :
  n = 2 ∨ n = 3 → min_faces_to_paint n < 6 * total_small_cubes n :=
by sorry

end NUMINAMATH_CALUDE_min_faces_to_paint_correct_faces_to_paint_less_than_total_l3501_350149


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3501_350190

theorem parallel_vectors_k_value (k : ℝ) : 
  let a : Fin 2 → ℝ := ![3*k + 1, 2]
  let b : Fin 2 → ℝ := ![k, 1]
  (∃ (c : ℝ), a = c • b) → k = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3501_350190


namespace NUMINAMATH_CALUDE_tournament_games_per_pair_l3501_350146

/-- Represents a chess tournament --/
structure ChessTournament where
  num_players : ℕ
  total_games : ℕ
  games_per_pair : ℕ
  h_players : num_players = 19
  h_total_games : total_games = 342
  h_games_formula : total_games = (num_players * (num_players - 1) * games_per_pair) / 2

/-- Theorem stating that in the given tournament, each player plays against each opponent twice --/
theorem tournament_games_per_pair (t : ChessTournament) : t.games_per_pair = 2 := by
  sorry

#check tournament_games_per_pair

end NUMINAMATH_CALUDE_tournament_games_per_pair_l3501_350146


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3501_350117

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3501_350117


namespace NUMINAMATH_CALUDE_equation_solution_l3501_350195

theorem equation_solution (x : ℝ) (h : 9 - 4/x = 7 + 8/x) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3501_350195


namespace NUMINAMATH_CALUDE_y_percent_of_x_l3501_350157

theorem y_percent_of_x (y x : ℕ+) (h1 : y = (125 : ℕ+)) (h2 : (y : ℝ) = 0.125 * (x : ℝ)) :
  (y : ℝ) / 100 * (x : ℝ) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_y_percent_of_x_l3501_350157


namespace NUMINAMATH_CALUDE_hexagon_with_90_degree_angle_l3501_350186

/-- A hexagon with angles in geometric progression has an angle of 90 degrees. -/
theorem hexagon_with_90_degree_angle :
  ∃ (a r : ℝ), 
    a > 0 ∧ r > 0 ∧
    a + a*r + a*r^2 + a*r^3 + a*r^4 + a*r^5 = 720 ∧
    (a = 90 ∨ a*r = 90 ∨ a*r^2 = 90 ∨ a*r^3 = 90 ∨ a*r^4 = 90 ∨ a*r^5 = 90) :=
by sorry

end NUMINAMATH_CALUDE_hexagon_with_90_degree_angle_l3501_350186


namespace NUMINAMATH_CALUDE_total_cost_l3501_350173

/-- The cost of a bottle of soda -/
def soda_cost : ℚ := sorry

/-- The cost of a bottle of mineral water -/
def mineral_cost : ℚ := sorry

/-- First condition: 2 bottles of soda and 1 bottle of mineral water cost 7 yuan -/
axiom condition1 : 2 * soda_cost + mineral_cost = 7

/-- Second condition: 4 bottles of soda and 3 bottles of mineral water cost 16 yuan -/
axiom condition2 : 4 * soda_cost + 3 * mineral_cost = 16

/-- Theorem: The cost of 10 bottles of soda and 10 bottles of mineral water is 45 yuan -/
theorem total_cost : 10 * soda_cost + 10 * mineral_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_l3501_350173


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l3501_350131

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 3)
  (sum_cubes : a^3 + b^3 + c^3 = 4) :
  a^4 + b^4 + c^4 = 6.833 := by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l3501_350131


namespace NUMINAMATH_CALUDE_donna_dog_walking_rate_l3501_350167

def dog_walking_hours : ℕ := 2 * 7
def card_shop_earnings : ℚ := 2 * 5 * 12.5
def babysitting_earnings : ℚ := 4 * 10
def total_earnings : ℚ := 305

theorem donna_dog_walking_rate : 
  ∃ (rate : ℚ), rate * dog_walking_hours + card_shop_earnings + babysitting_earnings = total_earnings ∧ rate = 10 := by
sorry

end NUMINAMATH_CALUDE_donna_dog_walking_rate_l3501_350167


namespace NUMINAMATH_CALUDE_employee_average_salary_l3501_350133

theorem employee_average_salary 
  (num_employees : ℕ) 
  (manager_salary : ℕ) 
  (average_increase : ℕ) 
  (h1 : num_employees = 18)
  (h2 : manager_salary = 5800)
  (h3 : average_increase = 200) :
  let total_with_manager := (num_employees + 1) * (average_employee_salary + average_increase)
  let total_without_manager := num_employees * average_employee_salary + manager_salary
  total_with_manager = total_without_manager →
  average_employee_salary = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_employee_average_salary_l3501_350133


namespace NUMINAMATH_CALUDE_triangle_theorem_l3501_350159

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def given_conditions (t : Triangle) : Prop :=
  t.b = 3 ∧
  t.b * t.c * Real.cos t.A = -6 ∧
  1/2 * t.b * t.c * Real.sin t.A = 3

-- State the theorem
theorem triangle_theorem (t : Triangle) :
  given_conditions t →
  t.A = Real.pi * 3/4 ∧ t.a = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3501_350159


namespace NUMINAMATH_CALUDE_problem_statement_l3501_350176

def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, 2 * x - 2 ≥ m^2 - 3 * m

def q (m a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1) 1, m ≤ a * x

theorem problem_statement (m : ℝ) :
  (p m ↔ m ∈ Set.Icc 1 2) ∧
  ((¬(p m) ∧ ¬(q m 1)) ∧ (p m ∨ q m 1) ↔ m ∈ Set.Ioi 1 ∪ Set.Iic 2 \ {1}) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3501_350176


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3501_350179

theorem min_value_trig_expression (x : ℝ) :
  (Real.sin x)^8 + (Real.cos x)^8 + 2 ≥ 5/4 * ((Real.sin x)^6 + (Real.cos x)^6 + 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3501_350179


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3501_350198

theorem solution_set_inequality (x : ℝ) : 
  (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3501_350198


namespace NUMINAMATH_CALUDE_largest_reciprocal_l3501_350109

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/7 → b = 3/4 → c = 2 → d = 8 → e = 100 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l3501_350109


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l3501_350161

def is_fibonacci_like (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) = b (n + 1) + b n

theorem fibonacci_like_sequence (b : ℕ → ℕ) :
  is_fibonacci_like b →
  (∀ n m : ℕ, n < m → b n < b m) →
  b 6 = 96 →
  b 7 = 184 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l3501_350161


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l3501_350165

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 81 → x - y = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l3501_350165


namespace NUMINAMATH_CALUDE_continuity_at_two_l3501_350105

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / (x^2 - 4)

theorem continuity_at_two :
  ∀ (c : ℝ), ContinuousAt f 2 ↔ c = 7/4 := by sorry

end NUMINAMATH_CALUDE_continuity_at_two_l3501_350105


namespace NUMINAMATH_CALUDE_apartment_occupancy_l3501_350171

/-- Calculates the number of people in an apartment building given specific conditions. -/
def people_in_building (total_floors : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  let full_floors := total_floors / 2
  let half_full_floors := total_floors - full_floors
  let full_apartments := full_floors * apartments_per_floor
  let half_full_apartments := half_full_floors * (apartments_per_floor / 2)
  let total_apartments := full_apartments + half_full_apartments
  total_apartments * people_per_apartment

/-- Theorem stating that under given conditions, the number of people in the building is 360. -/
theorem apartment_occupancy : 
  people_in_building 12 10 4 = 360 := by
  sorry


end NUMINAMATH_CALUDE_apartment_occupancy_l3501_350171


namespace NUMINAMATH_CALUDE_stating_sandy_comic_books_l3501_350148

/-- 
Given a person with an initial number of comic books, who sells half of them and then buys more,
this function calculates the final number of comic books.
-/
def final_comic_books (initial : ℕ) (bought : ℕ) : ℕ :=
  initial / 2 + bought

/-- 
Theorem stating that if Sandy starts with 14 comic books, sells half, and buys 6 more,
she will end up with 13 comic books.
-/
theorem sandy_comic_books : final_comic_books 14 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_stating_sandy_comic_books_l3501_350148


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3501_350106

theorem perfect_square_condition (n : ℤ) : 
  (∃ k : ℤ, n^2 + 6*n + 1 = k^2) ↔ (n = -6 ∨ n = 0) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3501_350106


namespace NUMINAMATH_CALUDE_james_purchase_cost_l3501_350127

/-- The total cost of James' purchase of shirts and pants -/
def total_cost (num_shirts : ℕ) (shirt_price : ℕ) (pant_price : ℕ) : ℕ :=
  let num_pants := num_shirts / 2
  num_shirts * shirt_price + num_pants * pant_price

/-- Theorem stating that James' purchase costs $100 -/
theorem james_purchase_cost : total_cost 10 6 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_james_purchase_cost_l3501_350127


namespace NUMINAMATH_CALUDE_tunnel_length_l3501_350123

/-- Calculates the length of a tunnel given train and travel information -/
theorem tunnel_length (train_length : ℝ) (exit_time : ℝ) (train_speed : ℝ) :
  train_length = 2 →
  exit_time = 4 →
  train_speed = 90 →
  (train_speed / 60) * exit_time - train_length = 4 := by
  sorry

#check tunnel_length

end NUMINAMATH_CALUDE_tunnel_length_l3501_350123


namespace NUMINAMATH_CALUDE_y_not_between_l3501_350199

theorem y_not_between (a b x y : ℝ) (ha : a > 0) (hb : b > 0) 
  (hy : y = (a * Real.sin x + b) / (a * Real.sin x - b)) :
  y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_y_not_between_l3501_350199


namespace NUMINAMATH_CALUDE_probability_4_vertices_in_same_plane_l3501_350177

-- Define a cube type
def Cube := Unit

-- Define a function to represent the number of vertices in a cube
def num_vertices (c : Cube) : ℕ := 8

-- Define a function to represent the number of ways to select 4 vertices from 8
def ways_to_select_4_from_8 (c : Cube) : ℕ := 70

-- Define a function to represent the number of ways 4 vertices can lie in the same plane
def ways_4_vertices_in_same_plane (c : Cube) : ℕ := 12

-- Theorem statement
theorem probability_4_vertices_in_same_plane (c : Cube) :
  (ways_4_vertices_in_same_plane c : ℚ) / (ways_to_select_4_from_8 c : ℚ) = 6 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_4_vertices_in_same_plane_l3501_350177


namespace NUMINAMATH_CALUDE_zero_product_implies_zero_factor_l3501_350194

/-- p-arithmetic, where p is prime -/
structure PArithmetic (p : ℕ) :=
  (carrier : Type)
  (add : carrier → carrier → carrier)
  (mul : carrier → carrier → carrier)
  (zero : carrier)
  (isPrime : Nat.Prime p)

/-- Statement: In p-arithmetic, if the product of two numbers is zero, then at least one of the numbers must be zero -/
theorem zero_product_implies_zero_factor {p : ℕ} (parith : PArithmetic p) :
  ∀ (a b : parith.carrier), parith.mul a b = parith.zero → a = parith.zero ∨ b = parith.zero :=
sorry

end NUMINAMATH_CALUDE_zero_product_implies_zero_factor_l3501_350194


namespace NUMINAMATH_CALUDE_smallest_divisible_n_l3501_350132

theorem smallest_divisible_n : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m^3 % 450 = 0 ∧ m^4 % 2560 = 0 → n ≤ m) ∧
  n^3 % 450 = 0 ∧ n^4 % 2560 = 0 ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_n_l3501_350132


namespace NUMINAMATH_CALUDE_correct_num_recipes_l3501_350181

/-- The number of recipes to be made for a chocolate chip cookie bake sale. -/
def num_recipes : ℕ := 23

/-- The number of cups of chocolate chips required for one recipe. -/
def cups_per_recipe : ℕ := 2

/-- The total number of cups of chocolate chips needed for all recipes. -/
def total_cups_needed : ℕ := 46

/-- Theorem stating that the number of recipes is correct given the conditions. -/
theorem correct_num_recipes : 
  num_recipes * cups_per_recipe = total_cups_needed :=
by sorry

end NUMINAMATH_CALUDE_correct_num_recipes_l3501_350181


namespace NUMINAMATH_CALUDE_root_cube_sum_condition_l3501_350128

theorem root_cube_sum_condition (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁^3 - 6*x₁^2 + a*x₁ + a = 0) ∧ 
    (x₂^3 - 6*x₂^2 + a*x₂ + a = 0) ∧ 
    (x₃^3 - 6*x₃^2 + a*x₃ + a = 0) ∧ 
    ((x₁-3)^3 + (x₂-3)^3 + (x₃-3)^3 = 0)) ↔ 
  (a = -9) :=
sorry

end NUMINAMATH_CALUDE_root_cube_sum_condition_l3501_350128


namespace NUMINAMATH_CALUDE_no_solution_arccos_equation_l3501_350174

theorem no_solution_arccos_equation : ¬∃ x : ℝ, Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_arccos_equation_l3501_350174


namespace NUMINAMATH_CALUDE_carol_final_gold_tokens_l3501_350153

/-- Represents the state of Carol's tokens -/
structure TokenState where
  purple : ℕ
  green : ℕ
  gold : ℕ

/-- Defines the exchange rules -/
def exchange1 (state : TokenState) : TokenState :=
  { purple := state.purple - 3, green := state.green + 2, gold := state.gold + 1 }

def exchange2 (state : TokenState) : TokenState :=
  { purple := state.purple + 1, green := state.green - 4, gold := state.gold + 1 }

/-- Checks if an exchange is possible -/
def canExchange (state : TokenState) : Bool :=
  state.purple ≥ 3 ∨ state.green ≥ 4

/-- The initial state of Carol's tokens -/
def initialState : TokenState :=
  { purple := 100, green := 85, gold := 0 }

/-- The theorem to prove -/
theorem carol_final_gold_tokens :
  ∃ (finalState : TokenState),
    (¬canExchange finalState) ∧
    (finalState.gold = 90) ∧
    (∃ (n m : ℕ),
      finalState = (exchange2^[m] ∘ exchange1^[n]) initialState) :=
sorry

end NUMINAMATH_CALUDE_carol_final_gold_tokens_l3501_350153


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l3501_350191

-- Define the markup percentage
def markup : ℚ := 24 / 100

-- Define the selling price
def selling_price : ℚ := 8215

-- Define the cost price calculation function
def cost_price (sp : ℚ) (m : ℚ) : ℚ := sp / (1 + m)

-- Theorem statement
theorem computer_table_cost_price : 
  cost_price selling_price markup = 6625 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l3501_350191


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3501_350134

theorem fraction_subtraction : 3 / 5 - (2 / 15 + 1 / 3) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3501_350134


namespace NUMINAMATH_CALUDE_five_divided_triangle_has_48_triangles_l3501_350144

/-- Represents an equilateral triangle with sides divided into n equal parts -/
structure DividedEquilateralTriangle where
  n : ℕ
  n_pos : 0 < n

/-- Counts the number of distinct equilateral triangles in a divided equilateral triangle -/
def count_distinct_triangles (t : DividedEquilateralTriangle) : ℕ :=
  sorry

/-- Theorem stating that a 5-divided equilateral triangle contains 48 distinct equilateral triangles -/
theorem five_divided_triangle_has_48_triangles :
  ∀ (t : DividedEquilateralTriangle), t.n = 5 → count_distinct_triangles t = 48 :=
by sorry

end NUMINAMATH_CALUDE_five_divided_triangle_has_48_triangles_l3501_350144


namespace NUMINAMATH_CALUDE_decimal_expansion_irrational_l3501_350125

/-- Decimal expansion function -/
def decimal_expansion (f : ℕ → ℕ) : ℚ :=
  sorry

/-- Power function -/
def f (n : ℕ) (x : ℕ) : ℕ :=
  x^n

/-- Theorem: The decimal expansion α is irrational for all positive integers n -/
theorem decimal_expansion_irrational (n : ℕ) (h : n > 0) :
  ¬ ∃ (q : ℚ), q = decimal_expansion (f n) :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_irrational_l3501_350125


namespace NUMINAMATH_CALUDE_power_four_times_four_equals_square_to_fourth_l3501_350112

theorem power_four_times_four_equals_square_to_fourth (a : ℝ) : a^4 * a^4 = (a^2)^4 := by
  sorry

end NUMINAMATH_CALUDE_power_four_times_four_equals_square_to_fourth_l3501_350112


namespace NUMINAMATH_CALUDE_car_fuel_tank_cost_l3501_350150

/-- Proves that the cost to fill a car fuel tank is $45 given specific conditions -/
theorem car_fuel_tank_cost : ∃ (F : ℚ),
  (2000 / 500 : ℚ) * F + (3/5) * ((2000 / 500 : ℚ) * F) = 288 ∧ F = 45 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_tank_cost_l3501_350150


namespace NUMINAMATH_CALUDE_sphere_volume_to_surface_area_l3501_350102

theorem sphere_volume_to_surface_area :
  ∀ (r : ℝ), 
    (4 / 3 : ℝ) * π * r^3 = 4 * Real.sqrt 3 * π → 
    4 * π * r^2 = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_to_surface_area_l3501_350102


namespace NUMINAMATH_CALUDE_coefficients_of_our_equation_l3501_350111

/-- Given a quadratic equation ax^2 + bx + c = 0, 
    returns the coefficients a, b, and c as a triple -/
def quadratic_coefficients (a b c : ℚ) : ℚ × ℚ × ℚ := (a, b, c)

/-- The quadratic equation 3x^2 - 6x - 1 = 0 -/
def our_equation := quadratic_coefficients 3 (-6) (-1)

theorem coefficients_of_our_equation :
  our_equation.2.1 = -6 ∧ our_equation.2.2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_our_equation_l3501_350111


namespace NUMINAMATH_CALUDE_soccer_ball_price_is_40_l3501_350183

def soccer_ball_price (total_balls : ℕ) (amount_given : ℕ) (change_received : ℕ) : ℕ :=
  (amount_given - change_received) / total_balls

theorem soccer_ball_price_is_40 :
  soccer_ball_price 2 100 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_price_is_40_l3501_350183


namespace NUMINAMATH_CALUDE_max_regions_correct_max_regions_recurrence_max_regions_is_maximum_l3501_350126

/-- The maximum number of regions a plane can be divided into by n rectangles with parallel sides -/
def max_regions (n : ℕ) : ℕ := 2*n^2 - 2*n + 2

/-- Theorem stating that max_regions gives the correct number of regions for n rectangles -/
theorem max_regions_correct (n : ℕ) : 
  max_regions n = 2*n^2 - 2*n + 2 := by sorry

/-- Theorem stating that max_regions satisfies the recurrence relation -/
theorem max_regions_recurrence (n : ℕ) : 
  max_regions (n + 1) = max_regions n + 4*n := by sorry

/-- Theorem stating that max_regions gives the maximum possible number of regions -/
theorem max_regions_is_maximum (n : ℕ) (k : ℕ) :
  k ≤ max_regions n := by sorry

end NUMINAMATH_CALUDE_max_regions_correct_max_regions_recurrence_max_regions_is_maximum_l3501_350126


namespace NUMINAMATH_CALUDE_boys_in_class_l3501_350193

theorem boys_in_class (total_students : ℕ) (girls_ratio : ℚ) : 
  total_students = 160 → girls_ratio = 5/8 → total_students - (girls_ratio * total_students).num = 60 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l3501_350193


namespace NUMINAMATH_CALUDE_jakesDrinkVolume_l3501_350110

/-- Represents the composition of a drink mixture -/
structure DrinkMixture where
  coke : ℕ
  sprite : ℕ
  mountainDew : ℕ

/-- Calculates the total parts in a drink mixture -/
def totalParts (d : DrinkMixture) : ℕ := d.coke + d.sprite + d.mountainDew

/-- Represents Jake's drink mixture -/
def jakesDrink : DrinkMixture := { coke := 2, sprite := 1, mountainDew := 3 }

/-- The volume of Coke in Jake's drink in ounces -/
def cokeVolume : ℕ := 6

/-- Theorem: Jake's drink has a total volume of 18 ounces -/
theorem jakesDrinkVolume : 
  (cokeVolume * totalParts jakesDrink) / jakesDrink.coke = 18 := by
  sorry

end NUMINAMATH_CALUDE_jakesDrinkVolume_l3501_350110


namespace NUMINAMATH_CALUDE_exactly_one_line_through_6_5_l3501_350162

/-- Represents a line in the xy-plane with given x and y intercepts -/
structure Line where
  x_intercept : ℝ
  y_intercept : ℝ

/-- Checks if a real number is a positive even integer -/
def is_positive_even (n : ℝ) : Prop :=
  n > 0 ∧ ∃ k : ℤ, n = 2 * k

/-- Checks if a real number is a positive odd integer -/
def is_positive_odd (n : ℝ) : Prop :=
  n > 0 ∧ ∃ k : ℤ, n = 2 * k + 1

/-- Checks if a line passes through the point (6,5) -/
def passes_through_6_5 (l : Line) : Prop :=
  6 / l.x_intercept + 5 / l.y_intercept = 1

/-- The main theorem to be proved -/
theorem exactly_one_line_through_6_5 :
  ∃! l : Line,
    is_positive_even l.x_intercept ∧
    is_positive_odd l.y_intercept ∧
    passes_through_6_5 l :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_line_through_6_5_l3501_350162


namespace NUMINAMATH_CALUDE_total_cost_is_2495_l3501_350100

/-- Represents the quantity of each fruit in kilograms -/
def apple_qty : ℕ := 8
def mango_qty : ℕ := 9
def banana_qty : ℕ := 6
def grape_qty : ℕ := 4
def cherry_qty : ℕ := 3

/-- Represents the rate of each fruit per kilogram -/
def apple_rate : ℕ := 70
def mango_rate : ℕ := 75
def banana_rate : ℕ := 40
def grape_rate : ℕ := 120
def cherry_rate : ℕ := 180

/-- Calculates the total cost of all fruits -/
def total_cost : ℕ := 
  apple_qty * apple_rate + 
  mango_qty * mango_rate + 
  banana_qty * banana_rate + 
  grape_qty * grape_rate + 
  cherry_qty * cherry_rate

/-- Theorem stating that the total cost of all fruits is 2495 -/
theorem total_cost_is_2495 : total_cost = 2495 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_2495_l3501_350100


namespace NUMINAMATH_CALUDE_rational_function_zero_l3501_350168

-- Define the numerator and denominator of the rational function
def numerator (x : ℝ) : ℝ := x^2 - x - 6
def denominator (x : ℝ) : ℝ := 5*x - 15

-- Define the domain of the function (all real numbers except 3)
def domain (x : ℝ) : Prop := x ≠ 3

-- State the theorem
theorem rational_function_zero (x : ℝ) (h : domain x) : 
  (numerator x) / (denominator x) = 0 ↔ x = -2 :=
sorry

end NUMINAMATH_CALUDE_rational_function_zero_l3501_350168


namespace NUMINAMATH_CALUDE_consecutive_sets_sum_150_l3501_350139

/-- A structure representing a set of consecutive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum_is_150 : (length * (2 * start + length - 1)) / 2 = 150
  length_ge_2 : length ≥ 2

/-- The theorem stating that there are exactly 5 sets of consecutive integers summing to 150 -/
theorem consecutive_sets_sum_150 :
  (∃ (sets : Finset ConsecutiveSet), sets.card = 5 ∧
    (∀ s : ConsecutiveSet, s ∈ sets ↔ 
      (s.length * (2 * s.start + s.length - 1)) / 2 = 150 ∧ 
      s.length ≥ 2)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_sets_sum_150_l3501_350139


namespace NUMINAMATH_CALUDE_fraction_problem_l3501_350178

theorem fraction_problem (A B x : ℝ) : 
  A + B = 27 → 
  B = 15 → 
  0.5 * A + x * B = 11 → 
  x = 1/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3501_350178


namespace NUMINAMATH_CALUDE_regular_tile_area_l3501_350136

/-- Represents the properties of tiles used to cover a wall -/
structure TileInfo where
  regularLength : ℝ
  regularWidth : ℝ
  jumboLength : ℝ
  jumboWidth : ℝ
  totalTiles : ℝ
  regularTiles : ℝ
  jumboTiles : ℝ

/-- Theorem stating the area covered by regular tiles on a wall -/
theorem regular_tile_area (t : TileInfo) (h1 : t.jumboLength = 3 * t.regularLength)
    (h2 : t.jumboWidth = t.regularWidth)
    (h3 : t.jumboTiles = (1/3) * t.totalTiles)
    (h4 : t.regularTiles = (2/3) * t.totalTiles)
    (h5 : t.regularLength * t.regularWidth * t.regularTiles +
          t.jumboLength * t.jumboWidth * t.jumboTiles = 385) :
    t.regularLength * t.regularWidth * t.regularTiles = 154 := by
  sorry

end NUMINAMATH_CALUDE_regular_tile_area_l3501_350136


namespace NUMINAMATH_CALUDE_angle_sine_relation_l3501_350182

open Real

/-- For angles α and β in the first quadrant, "α > β" is neither a sufficient nor a necessary condition for "sin α > sin β". -/
theorem angle_sine_relation (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) :
  (∃ α' β' : ℝ, α' > β' ∧ sin α' = sin β') ∧
  (∃ α'' β'' : ℝ, sin α'' > sin β'' ∧ ¬(α'' > β'')) := by
  sorry


end NUMINAMATH_CALUDE_angle_sine_relation_l3501_350182
