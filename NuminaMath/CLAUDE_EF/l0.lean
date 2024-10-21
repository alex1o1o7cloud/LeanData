import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_solve_inequality_system_l0_38

-- Problem 1
theorem calculate_expression : 
  Real.sqrt (9 / 4) + |2 - Real.sqrt 3| - (64 : ℝ) ^ (1/3) + 2⁻¹ = -Real.sqrt 3 := by sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) :
  (x + 5 < 4 ∧ (3 * x + 1) / 2 ≥ 2 * x - 1) ↔ x < -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_solve_inequality_system_l0_38


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_for_spheres_l0_14

/-- The edge length of a tetrahedron circumscribed around four mutually tangent spheres -/
noncomputable def tetrahedron_edge_length (sphere_radius : ℝ) (elevation : ℝ) : ℝ :=
  8 * Real.sqrt 3 / 3

/-- The configuration of four mutually tangent spheres with one elevated -/
structure SpheresConfiguration where
  sphere_radius : ℝ
  elevation : ℝ
  spheres_tangent : Bool
  elevated_sphere : Bool
  floor_spheres : Bool
  floor_spheres_touch_elevated : Bool

/-- The theorem stating the edge length of the circumscribed tetrahedron -/
theorem tetrahedron_edge_length_for_spheres 
  (config : SpheresConfiguration) 
  (h_radius : config.sphere_radius = 2) 
  (h_elevation : config.elevation = 2) 
  (h_tangent : config.spheres_tangent = true) 
  (h_elevated : config.elevated_sphere = true) 
  (h_floor : config.floor_spheres = true) 
  (h_touch : config.floor_spheres_touch_elevated = true) :
  tetrahedron_edge_length config.sphere_radius config.elevation = 8 * Real.sqrt 3 / 3 := by
  sorry

#check tetrahedron_edge_length_for_spheres

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_for_spheres_l0_14


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_maximum_marks_correct_l0_25

/-- Calculates the maximum marks for a test given the pass percentage, student's score, and fail margin. -/
def test_maximum_marks (pass_percentage : ℚ) (student_score : ℕ) (fail_margin : ℕ) : ℚ :=
  let pass_threshold : ℚ := (student_score + fail_margin : ℚ)
  pass_threshold / pass_percentage

/-- Proves that the calculated maximum marks are correct. -/
theorem test_maximum_marks_correct (pass_percentage : ℚ) (student_score : ℕ) (fail_margin : ℕ) 
    (h : 0 < pass_percentage) :
    pass_percentage * test_maximum_marks pass_percentage student_score fail_margin = 
    (student_score + fail_margin : ℚ) := by
  unfold test_maximum_marks
  field_simp
  ring

#eval test_maximum_marks (3/10) 80 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_maximum_marks_correct_l0_25


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_and_triangle_perimeter_l0_44

theorem quadratic_roots_and_triangle_perimeter (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - (2*k + 1)*x + 4*(k - 1/2)
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ∧
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x = 2 ∧ y = 4 → x + y + 4 = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_and_triangle_perimeter_l0_44


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_sum_l0_56

/-- A point on the right branch of the hyperbola x^2 - y^2 = 1 -/
structure HyperbolaPoint where
  a : ℝ
  b : ℝ
  on_hyperbola : a^2 - b^2 = 1
  right_branch : a > b

/-- The distance from a point (x, y) to the line y = x is |x - y| / √2 -/
noncomputable def distanceToLine (x y : ℝ) : ℝ := |x - y| / Real.sqrt 2

theorem hyperbola_point_sum (A : HyperbolaPoint) 
  (h : distanceToLine A.a A.b = Real.sqrt 2) : A.a + A.b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_sum_l0_56


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_l0_12

/-- Defines an isosceles triangle -/
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)

/-- Defines the base of an isosceles triangle -/
noncomputable def BaseOfIsoscelesTriangle (a b c : ℝ) : ℝ :=
  if a = b then c
  else if b = c then a
  else b

/-- An isosceles triangle with side lengths 4, 4, and 8 has a base of 4 -/
theorem isosceles_triangle_base (a b c : ℝ) : 
  a = 4 → b = 4 → c = 8 → 
  IsoscelesTriangle a b c → 
  BaseOfIsoscelesTriangle a b c = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_l0_12


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_divisor_l0_58

theorem remainder_divisor (n : ℕ) (h1 : n > 0) (h2 : n % 18 = 3) :
  (∃ (d : ℕ), d < 18 ∧ d > 1 ∧ n % d = 3) →
  (∀ (d : ℕ), d < 18 ∧ d > 1 ∧ n % d = 3 → d ≤ 9) ∧
  n % 9 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_divisor_l0_58


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l0_3

def T (n : ℕ) (b : ℝ) : ℝ :=
  b^n * 2^((n-1)*n/2)

theorem geometric_sequence_property (b : ℝ) (h : b ≠ 0) :
  let seq := λ i => T (3 * (i + 2)) b / T (3 * (i + 1)) b
  ∀ i, seq (i + 1) / seq i = 2^9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l0_3


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_inequality_min_volume_ratio_min_ratio_apex_angle_l0_73

/-- Represents a configuration of a right circular cone with an inscribed sphere 
    and a circumscribed right circular cylinder. -/
structure ConeSphereCylinder where
  R : ℝ  -- radius of the inscribed sphere
  x : ℝ  -- semivertical angle of the cone

/-- Volume of the cone -/
noncomputable def cone_volume (c : ConeSphereCylinder) : ℝ :=
  (1/3) * Real.pi * c.R^3 * (1 + Real.sin c.x)^3 / (Real.cos c.x)^2 / Real.sin c.x

/-- Volume of the cylinder -/
noncomputable def cylinder_volume (c : ConeSphereCylinder) : ℝ :=
  2 * Real.pi * c.R^3

/-- Ratio of cone volume to cylinder volume -/
noncomputable def volume_ratio (c : ConeSphereCylinder) : ℝ :=
  cone_volume c / cylinder_volume c

theorem cone_cylinder_volume_inequality (c : ConeSphereCylinder) :
  cone_volume c ≠ cylinder_volume c := by
  sorry

theorem min_volume_ratio :
  ∃ c : ConeSphereCylinder, volume_ratio c = 4/3 ∧ 
  ∀ c' : ConeSphereCylinder, volume_ratio c' ≥ 4/3 := by
  sorry

theorem min_ratio_apex_angle (c : ConeSphereCylinder) :
  volume_ratio c = 4/3 → c.x = Real.arcsin (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_inequality_min_volume_ratio_min_ratio_apex_angle_l0_73


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_solution_l0_84

noncomputable def f (a b x : ℝ) : ℝ := x / (a * x + b)

theorem function_solution (a b : ℝ) (h_a : a ≠ 0) :
  (f a b 2 = 1) ∧ 
  (∃! x, f a b x = x) →
  ((∀ x, f a b x = 2 * x / (x + 2)) ∨ (∀ x, f a b x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_solution_l0_84


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_345_ratio_triangle_l0_54

theorem largest_angle_in_345_ratio_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles in a triangle
  a / 3 = b / 4 ∧ b / 4 = c / 5 →  -- ratio of angles
  max a (max b c) = 75 :=  -- largest angle is 75°
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_345_ratio_triangle_l0_54


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_perimeter_ratio_l0_24

theorem equilateral_triangle_area_perimeter_ratio (s : ℝ) (h : s = 6) :
  (s^2 * Real.sqrt 3 / 4) / (3 * s) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_perimeter_ratio_l0_24


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l0_1

def is_valid_number (n : ℕ) : Bool :=
  100 ≤ n ∧ n ≤ 999 ∧  -- three-digit number
  (n / 100 > 5) ∧      -- hundreds digit > 5
  ((n / 10) % 10 > 5) ∧ -- tens digit > 5
  (n % 10 > 5) ∧       -- ones digit > 5
  n % 5 = 0            -- divisible by 5

theorem count_valid_numbers : 
  (Finset.filter (fun n => is_valid_number n = true) (Finset.range 1000)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l0_1


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_concentration_after_drinking_and_refilling_l0_37

/-- Calculates the concentration of milk after drinking some and refilling with water -/
noncomputable def milk_concentration (initial_volume : ℝ) (volume_drunk : ℝ) : ℝ :=
  (initial_volume - volume_drunk) / initial_volume * 100

/-- The concentration of milk in a 20-liter cup is 90% after drinking 2 liters and refilling -/
theorem milk_concentration_after_drinking_and_refilling :
  milk_concentration 20 2 = 90 := by
  unfold milk_concentration
  norm_num

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check milk_concentration 20 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_concentration_after_drinking_and_refilling_l0_37


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_area_theorem_l0_95

/-- The number of circular arcs in the curve -/
def num_arcs : ℕ := 10

/-- The length of each circular arc -/
noncomputable def arc_length : ℝ := Real.pi / 3

/-- The side length of the regular pentagon -/
def pentagon_side : ℝ := 3

/-- The radius of each circular arc -/
noncomputable def arc_radius : ℝ := 1 / 2

/-- The area of the regular pentagon -/
noncomputable def pentagon_area : ℝ := (1 / 4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * pentagon_side^2

/-- The area of a single sector -/
noncomputable def sector_area : ℝ := (arc_length * arc_radius^2) / 2

/-- The total area enclosed by the curve -/
noncomputable def enclosed_area : ℝ := pentagon_area + num_arcs * sector_area

theorem curve_area_theorem :
  enclosed_area = 15.484 + 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_area_theorem_l0_95


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_inlet_fills_in_two_hours_l0_7

/-- Represents the time in hours for the first inlet to fill the cistern -/
noncomputable def T : ℝ := sorry

/-- Rate at which the first inlet fills the cistern (fraction per hour) -/
noncomputable def rate_inlet1 : ℝ := 1 / T

/-- Rate at which the second inlet fills the cistern (fraction per hour) -/
noncomputable def rate_inlet2 : ℝ := 1 / (2 * T)

/-- Rate at which the outlet empties the cistern (fraction per hour) -/
noncomputable def rate_outlet : ℝ := 1 / 2

/-- Combined rate of both inlets (fraction per hour) -/
noncomputable def rate_both_inlets : ℝ := rate_inlet1 + rate_inlet2

/-- Amount filled in the first hour (9:00 am to 10:00 am) -/
noncomputable def amount_filled_first_hour : ℝ := rate_both_inlets

/-- Combined rate of the system after outlet is opened (fraction per hour) -/
noncomputable def rate_combined : ℝ := rate_both_inlets - rate_outlet

/-- Remaining amount to be filled after the first hour -/
noncomputable def amount_remaining : ℝ := 1 - amount_filled_first_hour

theorem first_inlet_fills_in_two_hours :
  rate_combined = amount_remaining → T = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_inlet_fills_in_two_hours_l0_7


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equiangular_star_inner_pentagon_perimeter_l0_33

/-- The perimeter of the inner pentagon in an equiangular star with total segment length 1 -/
noncomputable def inner_pentagon_perimeter : ℝ :=
  1 - 1 / (1 + Real.sin (18 * Real.pi / 180))

/-- Theorem: The perimeter of the inner pentagon in an equiangular star with total segment length 1 -/
theorem equiangular_star_inner_pentagon_perimeter :
  let star_total_length : ℝ := 1
  let star_segments : ℕ := 5
  let star_is_equiangular : Prop := True -- placeholder for the equiangular property
  inner_pentagon_perimeter = 1 - 1 / (1 + Real.sin (18 * Real.pi / 180)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equiangular_star_inner_pentagon_perimeter_l0_33


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moe_has_least_money_l0_79

structure Friend where
  name : String
  money : ℕ

def bo : Friend := ⟨"Bo", 0⟩
def coe : Friend := ⟨"Coe", 0⟩
def flo : Friend := ⟨"Flo", 0⟩
def jo : Friend := ⟨"Jo", 0⟩
def moe : Friend := ⟨"Moe", 0⟩
def zoe : Friend := ⟨"Zoe", 0⟩

def friends : List Friend := [bo, coe, flo, jo, moe, zoe]

axiom different_amounts : ∀ f g : Friend, f ≠ g → f.money ≠ g.money

axiom flo_more_than_jo_bo : flo.money > jo.money ∧ flo.money > bo.money

axiom bo_coe_more_than_moe : bo.money > moe.money ∧ coe.money > moe.money

axiom jo_more_than_moe_less_than_bo : jo.money > moe.money ∧ jo.money < bo.money

axiom zoe_more_than_jo_less_than_coe : zoe.money > jo.money ∧ zoe.money < coe.money

theorem moe_has_least_money : ∀ f : Friend, f ∈ friends → f ≠ moe → moe.money < f.money := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moe_has_least_money_l0_79


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_no_french_percentage_l0_66

/-- Represents the percentage of employees who are men -/
noncomputable def percentMen : ℝ := 45

/-- Represents the percentage of men who speak French -/
noncomputable def percentMenFrench : ℝ := 60

/-- Represents the percentage of all employees who speak French -/
noncomputable def percentTotalFrench : ℝ := 40

/-- Calculates the percentage of women who do not speak French -/
noncomputable def percentWomenNoFrench : ℝ :=
  let percentWomen := 100 - percentMen
  let menFrench := percentMen * percentMenFrench / 100
  let womenFrench := percentTotalFrench - menFrench
  (percentWomen - womenFrench) / percentWomen * 100

theorem women_no_french_percentage :
  ∀ ε > 0, |percentWomenNoFrench - 76.36| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_no_french_percentage_l0_66


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l0_74

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (k r : ℝ), ∀ x, x > 0 → f x = k * x^r

def is_strictly_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

theorem problem_statement :
  (∃ n : ℝ, is_power_function (λ x ↦ n * x^(n^2 + 2*n)) ∧
    is_strictly_increasing (λ x ↦ n * x^(n^2 + 2*n)) (Set.Ioi 0)) ∧
  ¬(¬(∃ x : ℝ, x^2 + 2 > 3*x) ↔ (∀ x : ℝ, x^2 + 2 < 3*x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l0_74


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_assignment_count_l0_43

/-- A type representing the vertices and center of a regular octagon -/
inductive OctagonPoint
| A | B | C | D | E | F | G | H | J

/-- A function type that assigns a digit to each point of the octagon -/
def OctagonAssignment := OctagonPoint → Fin 9

/-- Checks if an assignment is valid (each digit used exactly once) -/
def is_valid_assignment (f : OctagonAssignment) : Prop :=
  Function.Injective f

/-- Checks if the sums along the specified lines are equal -/
def has_equal_sums (f : OctagonAssignment) : Prop :=
  let sum_line (p q : OctagonPoint) := (f p).val + (f OctagonPoint.J).val + (f q).val
  sum_line OctagonPoint.A OctagonPoint.E = sum_line OctagonPoint.B OctagonPoint.F ∧
  sum_line OctagonPoint.A OctagonPoint.E = sum_line OctagonPoint.C OctagonPoint.G ∧
  sum_line OctagonPoint.A OctagonPoint.E = sum_line OctagonPoint.D OctagonPoint.H

/-- The main theorem stating that there are 1152 valid assignments -/
theorem octagon_assignment_count :
  ∃ (s : Finset OctagonAssignment), 
    (∀ f ∈ s, is_valid_assignment f ∧ has_equal_sums f) ∧
    s.card = 1152 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_assignment_count_l0_43


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_binomial_sum_nonzero_l0_72

theorem alternating_binomial_sum_nonzero (m : ℕ) (h : m % 6 = 5) :
  1 + (-3 : ℤ) ^ ((m - 1) / 2 : ℕ) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_binomial_sum_nonzero_l0_72


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_inequality_l0_77

theorem max_value_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_condition : a + b + c + d ≤ 4) :
  ((a^2 + 3*a*b : ℝ) ^ (1/4 : ℝ)) + ((b^2 + 3*b*c : ℝ) ^ (1/4 : ℝ)) + 
  ((c^2 + 3*c*d : ℝ) ^ (1/4 : ℝ)) + ((d^2 + 3*d*a : ℝ) ^ (1/4 : ℝ)) ≤ 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_inequality_l0_77


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_approx_l0_2

/-- The number of circles described about the corners of a square -/
def num_circles : ℕ := 4

/-- The side length of the square in centimeters -/
noncomputable def square_side : ℝ := 14

/-- The radius of each circle in centimeters -/
noncomputable def circle_radius : ℝ := square_side / 2

/-- The area of the square in square centimeters -/
noncomputable def square_area : ℝ := square_side ^ 2

/-- The area of one circle in square centimeters -/
noncomputable def circle_area : ℝ := Real.pi * circle_radius ^ 2

/-- The area enclosed between the circumferences of the circles in square centimeters -/
noncomputable def enclosed_area : ℝ := square_area - num_circles * (circle_area / 4)

/-- Theorem stating that the enclosed area is approximately 42.06195997410015 square centimeters -/
theorem enclosed_area_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1e-10 ∧ |enclosed_area - 42.06195997410015| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_approx_l0_2


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_integer_l0_88

theorem quadratic_root_integer (n : ℕ+) :
  (∃ x : ℤ, 4 * x^2 - (4 * Real.sqrt 3 + 4) * x + Real.sqrt 3 * (n : ℝ) - 24 = 0) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_integer_l0_88


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_of_f_f_involutive_g_involutive_condition_h_involutive_condition_l0_62

noncomputable def f (x : ℝ) : ℝ := x / (x - 1)

theorem fixed_points_of_f :
  ∀ r : ℝ, r ≠ 1 → (f r = r ↔ r = 0 ∨ r = 2) := by sorry

theorem f_involutive :
  ∀ x : ℝ, x ≠ 1 → f (f x) = x := by sorry

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (2 * x) / (x + k)

theorem g_involutive_condition :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ -k → g k (g k x) = x) ↔ k = -2 := by sorry

noncomputable def h (a b c : ℝ) (x : ℝ) : ℝ := (a * x + b) / (b * x + c)

theorem h_involutive_condition :
  ∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (∀ x : ℝ, x ≠ -c/b → h a b c (h a b c x) = x) ↔ c = -a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_of_f_f_involutive_g_involutive_condition_h_involutive_condition_l0_62


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_field_area_l0_65

/-- The cost of fencing per meter in Rupees -/
noncomputable def fencing_cost_per_meter : ℝ := 4.40

/-- The total cost of fencing in Rupees -/
noncomputable def total_fencing_cost : ℝ := 5806.831494371739

/-- The circumference of the circular field in meters -/
noncomputable def circumference : ℝ := total_fencing_cost / fencing_cost_per_meter

/-- The radius of the circular field in meters -/
noncomputable def radius : ℝ := circumference / (2 * Real.pi)

/-- The area of the circular field in square meters -/
noncomputable def area_sq_meters : ℝ := Real.pi * radius^2

/-- The area of the circular field in hectares -/
noncomputable def area_hectares : ℝ := area_sq_meters / 10000

/-- Theorem stating that the area of the circular field is approximately 13.85 hectares -/
theorem circular_field_area : 
  abs (area_hectares - 13.85) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_field_area_l0_65


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l0_26

noncomputable def g (x : ℝ) : ℝ := (Real.cos (6 * x) + 2 * Real.sin (3 * x) ^ 2) / (2 - 2 * Real.cos (3 * x))

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - g x ^ 2)

theorem f_range : Set.range f = Set.Icc 0 (Real.sqrt 15 / 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l0_26


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_pentominoes_with_both_symmetries_l0_31

/-- A pentomino is a shape formed by joining five squares edge to edge. -/
structure Pentomino where
  -- Add necessary fields to represent a pentomino
  -- This is a placeholder and may need to be adjusted based on how pentominoes are represented
  id : Nat

/-- Checks if a pentomino has at least one line of reflectional symmetry -/
def has_reflectional_symmetry (p : Pentomino) : Bool :=
  sorry -- Define the condition for reflectional symmetry

/-- Checks if a pentomino has rotational symmetry of order 2 (180 degrees) -/
def has_rotational_symmetry_order2 (p : Pentomino) : Bool :=
  sorry -- Define the condition for rotational symmetry of order 2

/-- The set of all 18 pentominoes -/
def all_pentominoes : Finset Pentomino :=
  sorry -- Define or assume the set of all 18 pentominoes

/-- The main theorem stating that exactly 4 pentominoes have both types of symmetry -/
theorem four_pentominoes_with_both_symmetries :
  (all_pentominoes.filter (λ p => has_reflectional_symmetry p ∧ has_rotational_symmetry_order2 p)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_pentominoes_with_both_symmetries_l0_31


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_heads_in_four_flips_prob_two_heads_in_four_flips_proof_l0_97

/-- The probability of getting exactly 2 heads in 4 flips of a fair coin -/
theorem prob_two_heads_in_four_flips : ℝ := 3/8

/-- Proof of the theorem -/
theorem prob_two_heads_in_four_flips_proof : prob_two_heads_in_four_flips = 
  let n : ℕ := 4  -- number of flips
  let k : ℕ := 2  -- number of heads we want
  let p : ℝ := 1/2  -- probability of heads on a single flip
  (n.choose k : ℝ) * p^k * (1-p)^(n-k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_heads_in_four_flips_prob_two_heads_in_four_flips_proof_l0_97


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_felicity_gasoline_amount_l0_92

-- Define the fuel amounts for each person
variable (adhira_diesel : ℝ)
variable (felicity_gasoline : ℝ)
variable (benjamin_ethanol : ℝ)

-- Define the relationships between fuel amounts
axiom felicity_ratio : felicity_gasoline = 2.2 * adhira_diesel
axiom benjamin_ratio : benjamin_ethanol = adhira_diesel / 1.5

-- Define the total fuel consumption for Felicity and Adhira
axiom total_felicity_adhira : felicity_gasoline + adhira_diesel = 30

-- Define Benjamin's fuel consumption
axiom benjamin_consumption : benjamin_ethanol = 35

-- Define fuel consumption rates (not used in the proof, but included for completeness)
def felicity_rate : ℝ := 2.5
def adhira_rate : ℝ := 1.8
def benjamin_rate : ℝ := 3

-- Theorem to prove
theorem felicity_gasoline_amount : felicity_gasoline = 20.625 := by
  have h1 : adhira_diesel = 30 / 3.2 := by
    -- Proof steps would go here
    sorry
  
  have h2 : felicity_gasoline = 2.2 * (30 / 3.2) := by
    -- Proof steps would go here
    sorry
  
  -- Final calculation
  calc
    felicity_gasoline = 2.2 * (30 / 3.2) := h2
    _ = 20.625 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_felicity_gasoline_amount_l0_92


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_m_value_l0_42

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 4^x / (4^x + m)

theorem symmetric_function_m_value (m : ℝ) :
  (∀ x : ℝ, f m x + f m (1 - x) = 1) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_m_value_l0_42


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l0_96

-- Define the function
noncomputable def f (x a : ℝ) := (Real.sin x + a) * (Real.cos x + a)

-- State the theorem
theorem min_value_of_f (a : ℝ) (h : a > Real.sqrt 2) :
  ∃ m : ℝ, (∀ x : ℝ, f x a ≥ m) ∧ (∃ x : ℝ, f x a = m) ∧ m = (a - Real.sqrt 2 / 2)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l0_96


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_finish_distance_l0_11

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  runnerA : Runner
  runnerB : Runner
  runnerC : Runner

/-- The theorem statement -/
theorem race_finish_distance (race : Race) 
  (h1 : race.distance = 1000)
  (h2 : race.runnerB.speed = 0.95 * race.runnerA.speed)
  (h3 : race.runnerC.speed = 0.96 * race.runnerB.speed) :
  race.distance - (race.runnerC.speed * (race.distance / race.runnerA.speed)) = 88 := by
  sorry

#check race_finish_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_finish_distance_l0_11


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacque_suitcase_weight_l0_53

/-- Calculates the final weight of Jacque's suitcase after his trip to France -/
noncomputable def final_suitcase_weight (initial_weight : ℝ) (perfume_weight : ℝ) (perfume_count : ℕ)
  (chocolate_weight : ℝ) (soap_weight : ℝ) (soap_count : ℕ) (jam_weight : ℝ) (jam_count : ℕ)
  (sculpture_weight : ℝ) (shirt_weight : ℝ) (shirt_count : ℕ) : ℝ :=
  let ounce_to_pound := 1 / 16
  let kg_to_pound := 2.20462
  let gram_to_kg := 1 / 1000
  initial_weight +
  (perfume_weight * (perfume_count : ℝ) * ounce_to_pound) +
  chocolate_weight +
  (soap_weight * (soap_count : ℝ) * ounce_to_pound) +
  (jam_weight * (jam_count : ℝ) * ounce_to_pound) +
  (sculpture_weight * kg_to_pound) +
  (shirt_weight * (shirt_count : ℝ) * gram_to_kg * kg_to_pound)

/-- Theorem stating that Jacque's suitcase weight on the return flight is approximately 27.70 pounds -/
theorem jacque_suitcase_weight :
  ∃ ε > 0, |final_suitcase_weight 12 1.2 5 4 5 2 8 2 3.5 300 3 - 27.70| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacque_suitcase_weight_l0_53


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_inequality_l0_35

def f : Nat → Nat := sorry

axiom f_property : ∀ w x y z : Nat, f (f (f z)) * f (w * x * f (y * f z)) = z^2 * f (x * f y) * f w

theorem factorial_inequality : ∀ n : Nat, n > 0 → f (Nat.factorial n) ≥ Nat.factorial n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_inequality_l0_35


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_is_C_l0_29

-- Define the expressions as noncomputable
noncomputable def A : ℝ := 30 - 4 * Real.sqrt 14
noncomputable def B : ℝ := 4 * Real.sqrt 14 - 30
noncomputable def C : ℝ := 25 - 6 * Real.sqrt 15
noncomputable def D : ℝ := 75 - 15 * Real.sqrt 30
noncomputable def E : ℝ := 15 * Real.sqrt 30 - 75

-- Theorem statement
theorem smallest_positive_is_C :
  C > 0 ∧ (A ≤ 0 ∨ C < A) ∧ (B ≤ 0 ∨ C < B) ∧ (D ≤ 0 ∨ C < D) ∧ (E ≤ 0 ∨ C < E) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_is_C_l0_29


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_perimeter_is_28_l0_36

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a polygon defined by its vertices -/
structure Polygon where
  vertices : List Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

/-- Calculate the perimeter of a polygon -/
noncomputable def perimeter (poly : Polygon) : ℝ :=
  let pairs := poly.vertices.zip (poly.vertices.rotate 1)
  (pairs.map (fun (p1, p2) => distance p1 p2)).sum

/-- Main theorem -/
theorem polygon_perimeter_is_28 
  (p q r s t u : Point)
  (pqrstu : Polygon)
  (right_angle_p : distance p t * distance p q = distance t q * distance p p)
  (right_angle_q : distance q p * distance q r = distance p r * distance q q)
  (right_angle_t : distance t p * distance t u = distance p u * distance t t)
  (pq_length : distance p q = 4)
  (ts_length : distance t s = 8)
  (u_on_ts : ∃ (k : ℝ), 0 ≤ k ∧ k ≤ 1 ∧ u.x = k * s.x + (1 - k) * t.x ∧ u.y = k * s.y + (1 - k) * t.y)
  (pqrstu_def : pqrstu.vertices = [p, q, r, s, t, u]) :
  perimeter pqrstu = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_perimeter_is_28_l0_36


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_shift_overlap_sine_l0_52

-- Define the range for φ
def phi_range (φ : ℝ) : Prop := -Real.pi ≤ φ ∧ φ < Real.pi

-- Define the original cosine function
noncomputable def original_func (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

-- Define the shifted cosine function
noncomputable def shifted_func (x φ : ℝ) : ℝ := original_func (x - Real.pi/2) φ

-- Define the sine function to compare with
noncomputable def compare_func (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi/3)

theorem cosine_shift_overlap_sine (φ : ℝ) 
  (h_range : phi_range φ) 
  (h_overlap : ∀ x, shifted_func x φ = compare_func x) : 
  φ = -5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_shift_overlap_sine_l0_52


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_reservoir_problem_l0_47

-- Define the water amount function
noncomputable def water_amount (t : ℝ) : ℝ :=
  400 + 60 * t - 120 * Real.sqrt (6 * t)

-- Define the water supply tension condition
def water_supply_tension (t : ℝ) : Prop :=
  water_amount t ≤ 80

theorem water_reservoir_problem :
  -- The time when the water amount is minimum
  (∃ t_min : ℝ, ∀ t : ℝ, 0 ≤ t ∧ t ≤ 24 → water_amount t_min ≤ water_amount t) ∧
  (∃ t_min : ℝ, t_min = 6) ∧
  -- The duration of water supply tension
  (∃ duration : ℝ, 
    (∀ t : ℝ, 0 ≤ t ∧ t ≤ 24 → water_supply_tension t ↔ 8/3 < t ∧ t < 32/3) ∧
    duration = 32/3 - 8/3) ∧
  (∃ duration : ℝ, duration = 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_reservoir_problem_l0_47


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pdf_constant_l0_19

/-- A probability density function p(x) = c / (1 + x^2) -/
noncomputable def p (c : ℝ) (x : ℝ) : ℝ := c / (1 + x^2)

/-- The integral of a probability density function over its entire domain is 1 -/
axiom pdf_integral (c : ℝ) : ∫ (x : ℝ), p c x = 1

/-- The value of c in the probability density function p(x) = c / (1 + x^2) is 1/π -/
theorem pdf_constant : ∃ c : ℝ, ∀ x : ℝ, p c x = 1 / (π * (1 + x^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pdf_constant_l0_19


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_five_l0_98

/-- The count of integers in (400, 800] divisible by 5 allowing digit repetition -/
def count_with_repetition : ℕ := 81

/-- The count of integers in (400, 800] divisible by 5 without digit repetition -/
def count_without_repetition : ℕ := 56

/-- The lower bound of the interval -/
def lower_bound : ℕ := 400

/-- The upper bound of the interval -/
def upper_bound : ℕ := 800

theorem count_divisible_by_five (n : ℕ) (h1 : lower_bound < n) (h2 : n ≤ upper_bound) :
  (∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a ≤ 7 ∧ (c = 0 ∨ c = 5) ∧ n % 5 = 0) →
  count_with_repetition = 81 ∧ count_without_repetition = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_five_l0_98


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_condition_implies_a_bound_l0_16

open Real

/-- The function f(x) = 2ln(x) + x^2 - 2ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x + x^2 - 2 * a * x

theorem extreme_points_condition_implies_a_bound 
  (a : ℝ) 
  (ha : a > 0) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ 
    (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → (x = x₁ ∨ x = x₂)) ∧
    f a x₁ - f a x₂ ≥ 3/2 - 2 * Real.log 2) →
  a ≥ 3/2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_condition_implies_a_bound_l0_16


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_height_l0_27

/-- Proves that given a sculpture of height 2 feet 10 inches and a total height (sculpture + base) of 3.1666666666666665 feet, the height of the base is 1/3 feet. -/
theorem base_height (sculpture_height_feet : ℚ) (sculpture_height_inches : ℚ) (total_height : ℚ) : 
  sculpture_height_feet = 2 →
  sculpture_height_inches = 10 →
  total_height = 3.1666666666666665 →
  total_height - (sculpture_height_feet + sculpture_height_inches / 12) = 1 / 3 := by
  sorry

#check base_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_height_l0_27


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_is_inscribed_circle_center_l0_61

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The angle bisector of a triangle at a given vertex -/
def angleBisector (t : Triangle) (vertex : Fin 3) : Set (ℝ × ℝ) :=
  sorry

/-- The incenter of a triangle -/
noncomputable def incenter (t : Triangle) : ℝ × ℝ :=
  sorry

/-- The inscribed circle of a triangle -/
def inscribedCircle (t : Triangle) : Set (ℝ × ℝ) :=
  sorry

/-- Theorem: The incenter (intersection of angle bisectors) is the center of the inscribed circle -/
theorem incenter_is_inscribed_circle_center (t : Triangle) :
  incenter t ∈ inscribedCircle t ∧
  ∀ p ∈ inscribedCircle t, ∃ r : ℝ, ∀ q ∈ inscribedCircle t, dist p q ≤ r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_is_inscribed_circle_center_l0_61


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_elements_calculation_l0_86

theorem set_elements_calculation (A B : Finset ℕ) :
  (A.card = (3/2 : ℚ) * B.card) →
  ((A ∪ B).card = 4500) →
  ((A ∩ B).card = 1200) →
  A.card = 3420 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_elements_calculation_l0_86


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l0_15

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

theorem symmetry_implies_phi (φ : ℝ) :
  (∀ x, f (x + φ) = f (-x + φ)) →
  ∃ k : ℤ, φ = π / 12 + k * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l0_15


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_turn_five_minutes_l0_17

/-- Represents the number of radians a minute hand turns in a given number of minutes -/
noncomputable def minuteHandTurn (minutes : ℝ) : ℝ :=
  (2 * Real.pi * minutes) / 60

/-- Theorem stating that moving the minute hand back by 5 minutes results in a turn of -π/6 radians -/
theorem minute_hand_turn_five_minutes :
  minuteHandTurn (-5) = -Real.pi / 6 := by
  -- Unfold the definition of minuteHandTurn
  unfold minuteHandTurn
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_turn_five_minutes_l0_17


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_area_l0_20

-- Define the sphere and points
def Sphere : Type := Unit -- We don't need to define the sphere explicitly
def Point : Type := Sphere

-- Define the distance function
noncomputable def distance (a b : Point) : ℝ := 1 -- All distances are 1

-- Define the cone height
noncomputable def coneHeight : ℝ := Real.sqrt 6 / 2

-- Helper function (not part of the problem, but needed for the statement)
noncomputable def area_of_trajectory (P : Point) : ℝ := sorry

-- Theorem statement
theorem trajectory_area (A B C P : Point) : 
  distance A B = 1 → 
  distance A C = 1 → 
  distance B C = 1 → 
  (∃ (h : ℝ), h = coneHeight) →  -- This represents the cone height condition
  area_of_trajectory P = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_area_l0_20


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_polynomials_l0_81

/-- A structure representing a quadratic polynomial ax^2 + bx + c with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The set of quadratic polynomials satisfying the given conditions -/
def ValidPolynomials : Set QuadraticPolynomial :=
  {p : QuadraticPolynomial | 
    p.a ≠ 0 ∧
    ∃ (r s : ℝ), 
      p.b = p.a * r ∧
      p.c = p.a * s ∧
      r + s = -p.b / p.a ∧
      r * s = p.c / p.a}

/-- The theorem stating that there are infinitely many valid quadratic polynomials -/
theorem infinitely_many_valid_polynomials : 
  Set.Infinite ValidPolynomials :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_polynomials_l0_81


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l0_9

noncomputable def f (x : ℝ) := -2 * Real.log (abs x)

theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, x < y ∧ y < 0 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_increasing_l0_9


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_select_A_and_B_l0_22

/-- The number of students in the group -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The number of students that must be selected (A and B) -/
def must_select : ℕ := 2

/-- The probability of selecting both A and B when choosing 3 students from a group of 5 -/
theorem probability_select_A_and_B :
  (Nat.choose (total_students - must_select) (selected_students - must_select)) / 
  (Nat.choose total_students selected_students : ℚ) = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_select_A_and_B_l0_22


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_lids_solution_l0_8

/-- Represents the number of can lids in each of the first three equal-sized boxes -/
def first_three_boxes : ℕ := sorry

/-- Represents the number of can lids in the fourth box -/
def fourth_box : ℕ := sorry

/-- The total number of can lids Aaron already had -/
def initial_lids : ℕ := 14

/-- The total number of can lids Aaron is taking to the recycling center -/
def total_lids : ℕ := 75

theorem can_lids_solution :
  (3 * first_three_boxes + fourth_box + initial_lids = total_lids) →
  (first_three_boxes = 20 ∧ fourth_box = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_lids_solution_l0_8


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_packing_l0_39

theorem circle_packing (R r : ℝ) (n m : ℕ) (s : ℝ) 
  (h1 : R = 18)
  (h2 : r = 3)
  (h3 : n = 16)
  (h4 : s = 1)
  (h5 : m = 9)
  (h6 : 0 < R ∧ 0 < r ∧ 0 < s)
  (h7 : n * (r + s)^2 + m * s^2 ≤ (R - s)^2) :
  ∃ (centers : Fin m → ℝ × ℝ), 
    (∀ i : Fin m, (centers i).1^2 + (centers i).2^2 ≤ (R - s)^2) ∧ 
    (∀ i j : Fin m, i ≠ j → 
      ((centers i).1 - (centers j).1)^2 + ((centers i).2 - (centers j).2)^2 ≥ (2*s)^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_packing_l0_39


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_25_factorial_l0_99

theorem largest_power_of_18_dividing_25_factorial :
  (∀ k : ℕ, k > 5 → ¬(Nat.factorial 25 % 18^k = 0)) ∧ (Nat.factorial 25 % 18^5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_25_factorial_l0_99


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l0_18

open Real

/-- The differential equation y'' - 12y' + 36y = sin(3x) -/
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (deriv^[2] y) x - 12 * (deriv y) x + 36 * y x = sin (3 * x)

/-- The general solution of the differential equation -/
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  (C₁ + C₂ * x) * exp (6 * x) + (4 / 225) * cos (3 * x) + (1 / 75) * sin (3 * x)

/-- Theorem stating that the general_solution satisfies the differential equation -/
theorem general_solution_satisfies_diff_eq (C₁ C₂ : ℝ) :
  ∀ x, diff_eq (general_solution C₁ C₂) x := by
  intro x
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l0_18


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l0_68

-- Define the inequality function
def f (x : ℝ) : Prop := 1 / (x^2 + 4) < 2 / x + 27 / 10

-- Define the solution set
def S : Set ℝ := Set.Ioo (-2) 0 ∪ Set.Ioo 0 (-10/27)

-- Theorem statement
theorem inequality_solution : ∀ x : ℝ, f x ↔ x ∈ S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l0_68


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_of_given_series_l0_90

def geometric_series : List ℚ := [-3/5, -5/3, -125/27]

def common_ratio (series : List ℚ) : Option ℚ :=
  if h : series.length ≥ 2
  then some (series[1]! / series[0]!)
  else none

theorem common_ratio_of_given_series :
  common_ratio geometric_series = some (25/9) := by
  rw [common_ratio, geometric_series]
  simp
  norm_num
  sorry

#eval common_ratio geometric_series

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_of_given_series_l0_90


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_rounded_to_nearest_tenth_l0_49

def total_students : ℕ := 22
def votes_in_favor : ℕ := 15

def ratio : ℚ := votes_in_favor / total_students

def round_to_nearest_tenth (x : ℚ) : ℚ := 
  ⌊(x * 10 + 1/2)⌋ / 10

theorem ratio_rounded_to_nearest_tenth :
  round_to_nearest_tenth ratio = 7/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_rounded_to_nearest_tenth_l0_49


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_rounded_l0_10

def mixed_number_to_fraction (whole : ℤ) (numerator : ℤ) (denominator : ℕ) : ℚ :=
  (whole : ℚ) + (numerator : ℚ) / (denominator : ℚ)

noncomputable def round_to_decimal_places (x : ℚ) (places : ℕ) : ℚ :=
  ((x * (10 : ℚ)^places).floor / (10 : ℚ)^places : ℚ)

theorem division_rounded (a b c d : ℤ) (e f : ℕ) :
  round_to_decimal_places ((mixed_number_to_fraction a b e) / ((c : ℚ) / (d : ℚ))) 3 = 39/10 :=
by
  sorry

#eval (39 : ℚ) / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_rounded_l0_10


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l0_63

/-- A circle with center on the y-axis, passing through (3, 1), and tangent to x-axis has the equation x^2 + (y-5)^2 = 25 -/
theorem circle_equation (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) :
  (∃ k, center = (0, k)) → -- center is on y-axis
  ((3 : ℝ), 1) ∈ C → -- circle passes through (3, 1)
  (∀ x : ℝ, (x, 0) ∉ C ∨ (∃ ε > 0, ∀ y : ℝ, 0 < |y| ∧ |y| < ε → (x, y) ∉ C)) → -- circle is tangent to x-axis
  C = {p : ℝ × ℝ | (p.1^2 + (p.2 - 5)^2 : ℝ) = 25} →
  ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + (y-5)^2 = 25 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l0_63


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desk_length_proof_l0_57

/-- Proves that the length of each desk is 5.5 meters under given conditions --/
theorem desk_length_proof (wall_length : ℝ) (bookcase_length : ℝ) (leftover_space : ℝ)
  (h_wall : wall_length = 15)
  (h_bookcase : bookcase_length = 1.5)
  (h_leftover : leftover_space = 1)
  (h_max_placement : ∃ (n : ℕ), 
    n > 0 ∧ 
    n * (desk_length + bookcase_length) = wall_length - leftover_space ∧
    ∀ (m : ℕ), m > n → m * (desk_length + bookcase_length) > wall_length - leftover_space) :
  desk_length = 5.5 := by
  sorry

where
  desk_length : ℝ := 5.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_desk_length_proof_l0_57


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evelyn_average_eric_average_kate_average_john_average_l0_78

-- Define the number of weeks
def num_weeks : ℕ := 3

-- Define the viewing hours for each family member
def evelyn_hours : Vector ℝ num_weeks := ⟨[10, 8, 6], by rfl⟩
def eric_hours : Vector ℝ num_weeks := ⟨[8, 6, 5], by rfl⟩
def kate_hours : Vector ℝ num_weeks := ⟨[0, 8, 4], by rfl⟩
def john_hours : Vector ℝ num_weeks := ⟨[0, 8, 8], by rfl⟩

-- Define a function to calculate average viewing hours
noncomputable def average_viewing_hours (hours : Vector ℝ num_weeks) : ℝ :=
  (hours.toList.sum) / num_weeks

-- Theorem statements
theorem evelyn_average :
  average_viewing_hours evelyn_hours = 8 := by sorry

theorem eric_average :
  average_viewing_hours eric_hours = 19 / 3 := by sorry

theorem kate_average :
  average_viewing_hours kate_hours = 4 := by sorry

theorem john_average :
  average_viewing_hours john_hours = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evelyn_average_eric_average_kate_average_john_average_l0_78


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vector_value_l0_94

/-- Given two vectors a and b in ℝ², where a = (2, -1) and b is collinear with a
    with magnitude 3√5, prove that b is either (6, -3) or (-6, 3). -/
theorem collinear_vector_value (a b : ℝ × ℝ) : 
  a = (2, -1) →
  (∃ (k : ℝ), b = k • a) →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 3 * Real.sqrt 5 →
  b = (6, -3) ∨ b = (-6, 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vector_value_l0_94


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jennas_eel_length_l0_48

/-- The length of Jenna's eel in inches -/
def jennas_eel : ℝ := sorry

/-- The length of Bill's eel in inches -/
def bills_eel : ℝ := sorry

/-- The length of Lucy's eel in inches -/
def lucys_eel : ℝ := sorry

/-- Jenna's eel is 2/5 as long as Bill's eel -/
axiom jenna_bill_ratio : jennas_eel = (2/5) * bills_eel

/-- Bill's eel is 4/3 the length of Lucy's eel -/
axiom bill_lucy_ratio : bills_eel = (4/3) * lucys_eel

/-- The combined length of their eels is 310 inches -/
axiom total_length : jennas_eel + bills_eel + lucys_eel = 310

/-- Lucy's eel is 5 inches shorter than twice the length of Jenna's eel -/
axiom lucy_jenna_relation : lucys_eel = 2 * jennas_eel - 5

theorem jennas_eel_length : jennas_eel = 965 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jennas_eel_length_l0_48


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_C_range_of_t_l0_4

-- Define points A, B, and C
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (3, -2)

-- Define vector AB and BC
def vecAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vecBC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define point P
def P (t : ℝ) : ℝ × ℝ := (3, t)

-- Define vectors PA and PB
def vecPA (t : ℝ) : ℝ × ℝ := (A.1 - (P t).1, A.2 - (P t).2)
def vecPB (t : ℝ) : ℝ × ℝ := (B.1 - (P t).1, B.2 - (P t).2)

-- Theorem for the coordinates of point C
theorem coordinates_of_C : vecAB = (2 * vecBC.1, 2 * vecBC.2) → C = (3, -2) := by
  sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) : 
  (vecPA t).1 * (vecPB t).1 + (vecPA t).2 * (vecPB t).2 < 0 ∧ 
  (vecPA t).1 * (vecPB t).2 ≠ (vecPA t).2 * (vecPB t).1 → 
  1 < t ∧ t < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_of_C_range_of_t_l0_4


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_AD_l0_64

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a broken line ABCD in space -/
structure BrokenLine where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Calculates the angle between three points -/
noncomputable def angle (p q r : Point3D) : ℝ := sorry

/-- Theorem: Shortest distance between A and D in a broken line ABCD -/
theorem shortest_distance_AD (line : BrokenLine) (a b : ℝ) 
    (h_angle_ABC : angle line.A line.B line.C = 120 * π / 180)
    (h_angle_BCD : 0 ≤ angle line.B line.C line.D ∧ angle line.B line.C line.D ≤ 60 * π / 180)
    (h_AB : distance line.A line.B = a)
    (h_BC : distance line.B line.C = a)
    (h_CD : distance line.C line.D = b)
    (h_a_gt_b : a > b)
    (h_b_pos : b > 0) :
  distance line.A line.D = Real.sqrt (3 * a^2 + b^2 - 2 * Real.sqrt 3 * a * b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_AD_l0_64


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l0_85

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in the xy-plane -/
structure Triangle where
  P : Point
  Q : Point
  R : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point lies on a line given by y = mx + b -/
def onLine (p : Point) (m b : ℝ) : Prop :=
  p.y = m * p.x + b

/-- Check if a triangle is right-angled at a given vertex -/
def isRightAngle (t : Triangle) (vertex : Point) : Prop :=
  let v1 := (t.P.x - vertex.x, t.P.y - vertex.y)
  let v2 := (t.Q.x - vertex.x, t.Q.y - vertex.y)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- Calculate the area of a triangle given two side lengths -/
noncomputable def triangleArea (a b : ℝ) : ℝ :=
  (1/2) * a * b

theorem triangle_area_theorem (t : Triangle) :
  isRightAngle t t.R ∧
  distance t.P t.R = 24 ∧
  distance t.R t.Q = 73 ∧
  distance t.P t.Q = 75 ∧
  (∃ p : ℝ, onLine ⟨p, 3*p + 4⟩ 3 4 ∧ distance ⟨p, 3*p + 4⟩ t.R = distance t.P t.R / 2) ∧
  (∃ q : ℝ, onLine ⟨q, -q + 5⟩ (-1) 5 ∧ distance ⟨q, -q + 5⟩ t.R = distance t.Q t.R / 2) →
  triangleArea (distance t.P t.R) (distance t.R t.Q) = 876 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l0_85


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l0_40

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℝ, (X : Polynomial ℝ)^2047 + 1 = 
    ((X : Polynomial ℝ)^12 - (X : Polynomial ℝ)^9 + (X : Polynomial ℝ)^6 - (X : Polynomial ℝ)^3 + 1) * q + 
    (-(X : Polynomial ℝ)^7 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l0_40


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_contradiction_assumption_l0_91

/-- A triangle is a geometric figure with three sides and three angles. -/
structure Triangle where
  sides : Fin 3 → ℝ
  angles : Fin 3 → ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) = Real.pi

/-- An angle is obtuse if it is greater than π/2. -/
def is_obtuse (angle : ℝ) : Prop := angle > Real.pi / 2

/-- The method of contradiction for proving "A triangle has at most one obtuse angle". -/
def contradiction_assumption (t : Triangle) : Prop :=
  ∃ i j : Fin 3, i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)

/-- The theorem stating that the correct assumption for the method of contradiction
    in proving "A triangle has at most one obtuse angle" is "There are at least two obtuse angles". -/
theorem correct_contradiction_assumption :
  ∀ t : Triangle, 
    (¬ ∃ i : Fin 3, ∀ j : Fin 3, j ≠ i → ¬(is_obtuse (t.angles j))) ↔ 
    contradiction_assumption t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_contradiction_assumption_l0_91


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_difference_l0_13

noncomputable def curve (a : ℝ) : ℝ → ℝ := λ x => a * x + 2 * Real.log (abs x)

theorem tangent_line_difference (a : ℝ) (k₁ k₂ : ℝ) :
  k₁ > k₂ →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    HasDerivAt (curve a) k₁ x₁ ∧
    HasDerivAt (curve a) k₂ x₂) →
  k₁ - k₂ = 4 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_difference_l0_13


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x5y_plus_xy5_value_l0_23

theorem x5y_plus_xy5_value (x y : ℝ) :
  (∃ q : ℚ, x + y = q) →
  (|x + 1| + (2*x - y + 4)^2 = 0) →
  x^5 * y + x * y^5 = -34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x5y_plus_xy5_value_l0_23


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_APB_l0_67

-- Define the square ABCD
noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (8, 0)
noncomputable def F : ℝ × ℝ := (0, 8)
noncomputable def D : ℝ × ℝ := (8, 8)

-- Define point P as the midpoint of AF
noncomputable def P : ℝ × ℝ := ((A.1 + F.1) / 2, (A.2 + F.2) / 2)

-- State the theorem
theorem area_of_triangle_APB :
  let triangle_area := (B.1 - A.1) * (P.2 - A.2) / 2
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 →
  triangle_area = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_APB_l0_67


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_yellow_balls_l0_60

/-- Given a bag with yellow and white balls, calculate the probability of drawing two yellow balls successively without replacement. -/
theorem probability_two_yellow_balls 
  (total_yellow : ℕ) 
  (total_white : ℕ) 
  (h_yellow : total_yellow = 6) 
  (h_white : total_white = 4) :
  (total_yellow - 1 : ℚ) / (total_yellow + total_white - 1) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_yellow_balls_l0_60


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_approximation_l0_28

/-- Represents a frustum-shaped basin -/
structure Basin where
  mouth_diameter : ℝ
  bottom_diameter : ℝ
  depth : ℝ

/-- Calculates the rainfall given a basin and water depth -/
noncomputable def calculate_rainfall (basin : Basin) (water_depth : ℝ) : ℝ :=
  let water_volume := sorry
  let mouth_area := Real.pi * (basin.mouth_diameter / 2) ^ 2
  water_volume / mouth_area * 10 -- Convert from feet to inches

/-- The main theorem stating that the calculated rainfall is approximately 12 inches -/
theorem rainfall_approximation (basin : Basin) (water_depth : ℝ) :
  basin.mouth_diameter = 28 ∧
  basin.bottom_diameter = 12 ∧
  basin.depth = 18 ∧
  water_depth = 9 →
  |calculate_rainfall basin water_depth - 12| < 0.01 := by
  sorry

#check rainfall_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_approximation_l0_28


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_universal_acquaintance_l0_51

/-- Represents the "knows" relation between people -/
def knows (people : Type*) (a b : people) : Prop := sorry

/-- The main theorem -/
theorem exists_universal_acquaintance
  (n : ℕ) -- n is a natural number
  (hn : n > 0) -- n is strictly positive
  (people : Type*) -- Type representing people
  [Fintype people] -- people is a finite type
  (h_card : Fintype.card people = 2 * n + 1) -- there are 2n + 1 people
  (h_symmetric : ∀ (a b : people), knows people a b ↔ knows people b a) -- knows is symmetric
  (h_existence : ∀ (S : Finset people), S.card = n →
    ∃ (p : people), p ∉ S ∧ ∀ (s : people), s ∈ S → knows people p s) :
  ∃ (p : people), ∀ (q : people), p ≠ q → knows people p q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_universal_acquaintance_l0_51


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_at_one_l0_0

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / (x^2 - 1)

theorem continuity_at_one : 
  ∃ (L : ℝ), ContinuousAt f 1 ↔ L = 3/2 := by
  sorry

#check continuity_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_at_one_l0_0


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l0_89

-- Define a regular tetrahedron with edge length 1
structure RegularTetrahedron where
  edge_length : ℝ
  edge_eq_one : edge_length = 1

-- Theorem statement
theorem tetrahedron_properties (t : RegularTetrahedron) :
  let r := (t.edge_length * Real.sqrt 6) / 4
  let v := t.edge_length^3 / (6 * Real.sqrt 2)
  let a := Real.sqrt 3 * t.edge_length^2
  let s := 4 * Real.pi * r^2
  (r ≠ Real.sqrt 3 / 3) ∧
  (v ≠ Real.sqrt 3) ∧
  (a ≠ Real.sqrt 6 + Real.sqrt 3 + 1) ∧
  (s ≠ 16 * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l0_89


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_existence_l0_83

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  not_all_zero : a ≠ 0 ∨ b ≠ 0

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ
  radius_pos : radius > 0

/-- The set of points on a circle -/
def Circle.points (S : Circle) : Set Point :=
  {p : Point | (p.x - S.center.x)^2 + (p.y - S.center.y)^2 = S.radius^2}

/-- The set of points on a line -/
def Line.points (l : Line) : Set Point :=
  {p : Point | l.a * p.x + l.b * p.y + l.c = 0}

/-- A line is tangent to a circle at a point -/
def TangentLine (S : Circle) (l : Line) (P : Point) : Prop :=
  P ∈ S.points ∧ P ∈ l.points ∧
  ∀ (Q : Point), Q ∈ l.points ∧ Q ≠ P → Q ∉ S.points

/-- Two circles are tangent at a point -/
def CirclesTangent (S1 S2 : Circle) (P : Point) : Prop :=
  P ∈ S1.points ∧ P ∈ S2.points ∧
  ∃ (l : Line), TangentLine S1 l P ∧ TangentLine S2 l P

/-- A circle is tangent to a line at a point -/
def CircleLineTangent (S : Circle) (l : Line) (P : Point) : Prop :=
  P ∈ S.points ∧ P ∈ l.points ∧ TangentLine S l P

/-- Given a circle S, a point A on S, and a line l, there exists a circle S'
    that is tangent to S at A and tangent to l at some point B. -/
theorem tangent_circle_existence (S : Circle) (A : Point) (l : Line) 
    (h1 : A ∈ S.points) : 
  ∃ (S' : Circle) (B : Point),
    CirclesTangent S S' A ∧ CircleLineTangent S' l B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_existence_l0_83


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_sequence_l0_6

def factorial_plus_n (n : ℕ) : ℕ := n.factorial + n

def sum_sequence : ℕ := (Finset.range 11).sum (λ i => factorial_plus_n (i + 1))

theorem units_digit_of_sum_sequence : sum_sequence % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_sequence_l0_6


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l0_82

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a line segment AB of length 2, P is the golden section point where AP > PB -/
def isGoldenSectionPoint (AP PB : ℝ) : Prop :=
  AP > PB ∧ AP / PB = φ ∧ AP + PB = 2

theorem golden_section_length :
  ∀ AP PB : ℝ, isGoldenSectionPoint AP PB → AP = Real.sqrt 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l0_82


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_left_for_monica_l0_55

theorem cookies_left_for_monica : ℕ := by
  let total_cookies := 120
  let father_cookies := 12
  let mother_cookies := father_cookies / 2
  let brother_cookies := mother_cookies + 2
  let sister_cookies := brother_cookies * 3
  let aunt_cookies := father_cookies * 2
  let cousin_cookies := aunt_cookies - 5
  
  let total_eaten := father_cookies + mother_cookies + brother_cookies + 
                     sister_cookies + aunt_cookies + cousin_cookies
  
  let monica_cookies := total_cookies - total_eaten
  
  have h1 : monica_cookies = 27 := by sorry
  
  exact monica_cookies


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_left_for_monica_l0_55


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l0_34

/-- The vertices of the triangle -/
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 6)
def C : ℝ × ℝ := (7, 3)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The lengths of the sides of the triangle -/
noncomputable def sideAB : ℝ := distance A B
noncomputable def sideBC : ℝ := distance B C
noncomputable def sideCA : ℝ := distance C A

/-- The theorem stating that the longest side of the triangle has length √26 -/
theorem longest_side_length :
  max sideAB (max sideBC sideCA) = Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l0_34


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_monomial_l0_21

/-- The coefficient of a monomial is the numerical factor that multiplies the variables. -/
noncomputable def coefficient (m : ℝ → ℝ → ℝ) : ℝ :=
  sorry

/-- The given monomial -/
noncomputable def monomial (x y : ℝ) : ℝ := -(2 * Real.pi * x^2 * y^2) / 3

theorem coefficient_of_monomial :
  coefficient monomial = -(2 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_monomial_l0_21


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l0_46

/-- The cosine function with angular frequency ω and phase φ -/
noncomputable def f (ω φ x : ℝ) : ℝ := Real.cos (ω * x + φ)

/-- The smallest positive period of f -/
noncomputable def T (ω : ℝ) : ℝ := 2 * Real.pi / ω

theorem min_omega_value (ω φ : ℝ) :
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  f ω φ (T ω) = Real.sqrt 3 / 2 →
  f ω φ (Real.pi / 9) = 0 →
  ∀ ω' > 0, (f ω' φ (T ω') = Real.sqrt 3 / 2 ∧ f ω' φ (Real.pi / 9) = 0) → ω' ≥ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l0_46


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l0_5

theorem problem_statement : (∃ x : ℝ, (2 : ℝ)^x ≤ (3 : ℝ)^x) ∨ (∀ x : ℝ, Real.exp x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l0_5


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_in_interval_l0_32

noncomputable def min_roots (g : ℝ → ℝ) (a b : ℝ) : ℕ := sorry

theorem min_roots_in_interval 
  (g : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, g (3 + x) = g (3 - x))
  (h2 : ∀ x : ℝ, g (8 + x) = g (8 - x))
  (h3 : g 0 = 0) :
  min_roots g (-1500) 1500 ≥ 690 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_in_interval_l0_32


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_short_side_length_is_five_l0_71

/-- Represents the dimensions and velvet requirements of a box --/
structure Box where
  long_side_length : ℚ
  long_side_width : ℚ
  short_side_width : ℚ
  top_bottom_area : ℚ
  total_velvet : ℚ

/-- Calculates the length of the short sides of the box --/
def short_side_length (b : Box) : ℚ :=
  let long_sides_area := 2 * b.long_side_length * b.long_side_width
  let top_bottom_total_area := 2 * b.top_bottom_area
  let short_sides_total_area := b.total_velvet - long_sides_area - top_bottom_total_area
  short_sides_total_area / (2 * b.short_side_width)

/-- Theorem stating that the short side length is 5 inches for the given box dimensions --/
theorem short_side_length_is_five (b : Box) 
  (h1 : b.long_side_length = 8)
  (h2 : b.long_side_width = 6)
  (h3 : b.short_side_width = 6)
  (h4 : b.top_bottom_area = 40)
  (h5 : b.total_velvet = 236) :
  short_side_length b = 5 := by
  sorry

def main : IO Unit := do
  let result := short_side_length { 
    long_side_length := 8,
    long_side_width := 6,
    short_side_width := 6,
    top_bottom_area := 40,
    total_velvet := 236
  }
  IO.println s!"The short side length is {result}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_short_side_length_is_five_l0_71


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_AP_PB_l0_69

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  coordinates : ℝ × ℝ

-- Define the problem setup
def problem_setup (circle1 circle2 : Circle) (P : Point) : Prop :=
  ∃ (Q : Point), 
    (P.coordinates ∈ Metric.sphere circle1.center circle1.radius) ∧
    (P.coordinates ∈ Metric.sphere circle2.center circle2.radius) ∧
    (Q.coordinates ∈ Metric.sphere circle1.center circle1.radius) ∧
    (Q.coordinates ∈ Metric.sphere circle2.center circle2.radius) ∧
    P ≠ Q

-- Define the segment AB
def segment_AB (A B : Point) : Set (ℝ × ℝ) :=
  {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • A.coordinates + t • B.coordinates}

-- Define the product AP·PB
noncomputable def product_AP_PB (A B P : Point) : ℝ :=
  ‖A.coordinates - P.coordinates‖ * ‖P.coordinates - B.coordinates‖

-- Define the angle between two vectors
noncomputable def angle (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (‖v1‖ * ‖v2‖))

-- Theorem statement
theorem max_product_AP_PB 
  (circle1 circle2 : Circle) (P : Point) 
  (h : problem_setup circle1 circle2 P) :
  ∃ (A B : Point),
    (A.coordinates ∈ Metric.sphere circle1.center circle1.radius) ∧
    (B.coordinates ∈ Metric.sphere circle2.center circle2.radius) ∧
    (P.coordinates ∈ segment_AB A B) ∧
    (∀ (A' B' : Point),
      (A'.coordinates ∈ Metric.sphere circle1.center circle1.radius) →
      (B'.coordinates ∈ Metric.sphere circle2.center circle2.radius) →
      (P.coordinates ∈ segment_AB A' B') →
      product_AP_PB A B P ≥ product_AP_PB A' B' P) ∧
    (angle (A.coordinates - P.coordinates) (circle1.center - P.coordinates) =
     angle (B.coordinates - P.coordinates) (circle2.center - P.coordinates)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_AP_PB_l0_69


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_l0_87

-- Define a rectangle on a coordinate plane
structure Rectangle where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ
  d : ℤ × ℤ
  is_rectangle : (b.1 - a.1) * (c.2 - a.2) = (c.1 - a.1) * (b.2 - a.2)
  not_square : (b.1 - a.1)^2 + (b.2 - a.2)^2 ≠ (c.1 - a.1)^2 + (c.2 - a.2)^2

-- Define a function that represents the division of a rectangle into squares
def divide_into_squares (r : Rectangle) : Set ((ℤ × ℤ) × (ℤ × ℤ)) := sorry

-- The main theorem
theorem rectangle_division (r : Rectangle) :
  ∃ (squares : Set ((ℤ × ℤ) × (ℤ × ℤ))),
    squares = divide_into_squares r ∧
    (∀ s ∈ squares, (s.1.1 - s.2.1)^2 = (s.1.2 - s.2.2)^2) ∧
    (∀ s ∈ squares, s.1 ∈ Set.univ ∧ s.2 ∈ Set.univ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_l0_87


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAB_l0_70

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y = 3
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 4*y

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := (1 + Real.sqrt 2, 2 - Real.sqrt 2)
noncomputable def B : ℝ × ℝ := (1 - Real.sqrt 2, 2 + Real.sqrt 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

-- Theorem statement
theorem area_of_triangle_OAB :
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 →
  triangleArea O A B = 2*(Real.pi - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAB_l0_70


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l0_93

theorem no_valid_n : ¬∃ (n : ℕ), (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l0_93


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sqrt_l0_45

open Real

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem derivative_of_sqrt (x : ℝ) (h : x > 0) :
  deriv f x = 1 / (2 * Real.sqrt x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sqrt_l0_45


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rockets_won_30_games_l0_75

-- Define the set of teams
inductive Team : Type
| Hawks : Team
| Wolves : Team
| Rockets : Team
| Knicks : Team
| Lakers : Team
| Clippers : Team

-- Define the function that assigns games won to each team
def games_won : Team → ℕ := sorry

-- Define the set of possible game counts
def game_counts : Set ℕ := {15, 20, 25, 30, 35, 40}

-- State the theorem
theorem rockets_won_30_games 
  (h1 : games_won Team.Hawks > games_won Team.Wolves)
  (h2 : games_won Team.Rockets > games_won Team.Knicks)
  (h3 : games_won Team.Rockets < games_won Team.Lakers)
  (h4 : games_won Team.Knicks ≥ 15)
  (h5 : games_won Team.Clippers < games_won Team.Lakers)
  (h6 : ∀ t : Team, games_won t ∈ game_counts)
  (h7 : ∀ x y : Team, x ≠ y → games_won x ≠ games_won y) :
  games_won Team.Rockets = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rockets_won_30_games_l0_75


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_reducible_permutation_l0_76

/-- A word is a list of characters, either 'A' or 'B' -/
def Word := List Char

/-- Reduction rule: ABBB = B -/
def reduce : Word → Word
  | [] => []
  | 'A'::'B'::'B'::'B'::rest => 'B' :: reduce rest
  | x::xs => x :: reduce xs

/-- Cyclic permutation of a word -/
def cyclicPermutation (w : Word) (k : Nat) : Word :=
  (w.drop k ++ w.take k)

/-- A word is reducible if it can be reduced to "B" -/
def isReducible (w : Word) : Prop :=
  ∃ k, reduce (cyclicPermutation w k) = ['B']

/-- Count occurrences of a character in a word -/
def countChar (w : Word) (c : Char) : Nat :=
  w.filter (· = c) |>.length

/-- Main theorem -/
theorem unique_reducible_permutation (n : Nat) (w : Word) 
    (h1 : w.length = 3 * n + 1)
    (h2 : countChar w 'A' = n)
    (h3 : countChar w 'B' = 2 * n + 1) :
    ∃! k, reduce (cyclicPermutation w k) = ['B'] := by
  sorry

#check unique_reducible_permutation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_reducible_permutation_l0_76


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_is_1050_l0_50

def digits : List Nat := [2, 0, 1, 5]

def isValidArrangement (a b : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ (a.repr.length + b.repr.length = 4) ∧
  (∀ d, d ∈ digits ↔ d ∈ a.repr.toList.map Char.toNat ∨
                     d ∈ b.repr.toList.map Char.toNat)

theorem max_product_is_1050 :
  ∀ a b : Nat, isValidArrangement a b → a * b ≤ 1050 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_is_1050_l0_50


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l0_30

-- Define the train's length in meters
noncomputable def train_length : ℝ := 120

-- Define the train's speed in km/h
noncomputable def train_speed_kmh : ℝ := 27

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the theorem
theorem train_crossing_time :
  (train_length / (train_speed_kmh * kmh_to_ms)) = 16 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l0_30


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l0_59

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 - x else (1/2)^x

-- Define a, b, and c
noncomputable def a : ℝ := f (Real.log 2 / Real.log 3)
noncomputable def b : ℝ := f (2^(-(1/2 : ℝ)))
noncomputable def c : ℝ := f (3^(1/2 : ℝ))

-- Theorem statement
theorem a_greater_than_b_greater_than_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l0_59


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_triangle_area_l0_80

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (0, -2)
def C : ℝ × ℝ := (-2, 3)

-- Define the median CM
noncomputable def M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Theorem for the equation of median CM
theorem median_equation :
  ∀ (x y : ℝ), (2 * x + 3 * y - 5 = 0) ↔ (∃ t : ℝ, x = M.1 + t * (C.1 - M.1) ∧ y = M.2 + t * (C.2 - M.2)) :=
by sorry

-- Function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem for the area of triangle ABC
theorem triangle_area :
  (1/2) * distance A B * distance A C * Real.sqrt (1 - ((distance A B^2 + distance A C^2 - distance B C^2) / (2 * distance A B * distance A C))^2) = 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_triangle_area_l0_80


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_slope_range_l0_41

/-- Represents an ellipse with center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  h_focus : c^2 = a^2 - b^2
  h_point : 1^2 / a^2 + (3/2)^2 / b^2 = 1
  h_symmetric : ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ 2*c - x = 0 ∧ y = 0

/-- The slope of line MA where M is midpoint of chord passing through (1/2, 0) -/
noncomputable def slope_MA (e : Ellipse) (m : ℝ) : ℝ :=
  m / (4 * m^2 + 4)

theorem ellipse_equation_and_slope_range (e : Ellipse) :
  e.a = 2 ∧ e.b = Real.sqrt 3 ∧ ∀ k : ℝ, k ∈ Set.Icc (-1/8 : ℝ) (1/8 : ℝ) ↔ ∃ m : ℝ, k = slope_MA e m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_slope_range_l0_41
