import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_parabola_vertex_locus_l312_31201

/-- Given a real constant p and two parabolas, one rolling without slipping around the other,
    this theorem states the equation of the locus of the vertex of the rolling parabola. -/
theorem rolling_parabola_vertex_locus (p : ℝ) :
  let fixed_parabola := {(x, y) : ℝ × ℝ | y^2 = 4*p*x}
  let rolling_parabola := {(x, y) : ℝ × ℝ | y^2 = -4*p*x}
  let vertex_locus := {(x, y) : ℝ × ℝ | x*(x^2 + y^2) + 2*p*y^2 = 0}
  (∀ (x y : ℝ), (x, y) ∈ vertex_locus ↔ 
    ∃ (t : ℝ), (x = -2*p*t^2/(1 + t^2) ∧ y = 2*p*t^3/(1 + t^2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_parabola_vertex_locus_l312_31201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_result_l312_31253

/-- The sum of alternating signs applied to positive integers less than 200, excluding multiples of 10 -/
def alternating_sum : ℕ → ℤ
| 0 => 0
| (n+1) => if (n+1) % 10 ≠ 0 ∧ n+1 < 200 then
             alternating_sum n + (if n % 2 = 0 then (n+1 : ℤ) else -(n+1 : ℤ))
           else
             alternating_sum n

theorem alternating_sum_result : alternating_sum 199 = 109 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_result_l312_31253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_mixture_price_l312_31254

/-- The selling price per pound of a candy mixture -/
noncomputable def selling_price_per_pound (total_weight : ℝ) (weight1 : ℝ) (weight2 : ℝ) (price1 : ℝ) (price2 : ℝ) : ℝ :=
  (weight1 * price1 + weight2 * price2) / total_weight

/-- Theorem: The selling price per pound of the candy mixture is $3.00 -/
theorem candy_mixture_price :
  selling_price_per_pound 30 20 10 2.95 3.10 = 3 := by
  -- Unfold the definition of selling_price_per_pound
  unfold selling_price_per_pound
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_mixture_price_l312_31254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_margin_price_is_120_l312_31206

/-- Price of each set of "Water Margin" comic books -/
def water_margin_price : ℝ := 120

/-- Price of each set of "Romance of the Three Kingdoms" comic books -/
def three_kingdoms_price : ℝ := water_margin_price + 60

/-- Total spent on "Romance of the Three Kingdoms" comic books -/
def total_three_kingdoms : ℝ := 3600

/-- Total spent on "Water Margin" comic books -/
def total_water_margin : ℝ := 4800

/-- Number of sets of "Romance of the Three Kingdoms" is half of "Water Margin" -/
axiom half_sets : (total_three_kingdoms / three_kingdoms_price) = (1/2) * (total_water_margin / water_margin_price)

theorem water_margin_price_is_120 : water_margin_price = 120 := by
  -- The proof goes here
  sorry

#check water_margin_price_is_120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_margin_price_is_120_l312_31206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l312_31214

-- Define the hyperbola structure
structure Hyperbola where
  center : ℝ × ℝ
  passes_through : List (ℝ × ℝ)
  opens : String

-- Define our specific hyperbola as a function of s
def our_hyperbola (s : ℝ) : Hyperbola where
  center := (0, 0)
  passes_through := [(1, 3), (5, 0), (s, -3)]
  opens := "horizontally or vertically"

-- Theorem statement
theorem hyperbola_s_squared (s : ℝ) :
  (our_hyperbola s).passes_through = [(1, 3), (5, 0), (s, -3)] →
  s^2 = 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l312_31214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_is_six_l312_31266

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

/-- Counts the maximum number of intersection points between a line and two distinct circles -/
def max_intersection_points (l : Line) (c1 c2 : Circle) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of intersection points is 6 -/
theorem max_intersections_is_six :
  ∀ (l : Line) (c1 c2 : Circle), c1 ≠ c2 → 
    (max_intersection_points l c1 c2 ≤ 6 ∧ 
     ∃ (l' : Line) (c1' c2' : Circle), c1' ≠ c2' ∧ max_intersection_points l' c1' c2' = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_is_six_l312_31266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_eating_days_l312_31259

/-- Represents the initial fruit quantities for each person -/
structure FruitInventory where
  apples : Nat
  pears : Nat
  peaches : Nat

/-- Represents the eating pattern over the 18 days -/
structure EatingPattern where
  both_apple_days : Nat
  both_pear_days : Nat
  apple_pear_days : Nat
  total_days : Nat

/-- The main theorem to prove -/
theorem peach_eating_days 
  (xiaoming_inventory : FruitInventory)
  (xiaohong_inventory : FruitInventory)
  (pattern : EatingPattern)
  (h1 : xiaoming_inventory.apples = 4)
  (h2 : xiaoming_inventory.pears = 6)
  (h3 : xiaoming_inventory.peaches = 8)
  (h4 : xiaohong_inventory.apples = 5)
  (h5 : xiaohong_inventory.pears = 7)
  (h6 : xiaohong_inventory.peaches = 6)
  (h7 : pattern.both_apple_days = 3)
  (h8 : pattern.both_pear_days = 2)
  (h9 : pattern.apple_pear_days = 3)
  (h10 : pattern.total_days = 18)
  : Nat := by
  -- The proof steps would go here
  sorry

-- Remove the #eval statement as it was causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_eating_days_l312_31259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_linear_combination_l312_31285

/-- Given vectors in ℝ², prove that a specific linear combination exists --/
theorem vector_linear_combination (e₁ e₂ a : ℝ × ℝ) 
  (h₁ : e₁ = (2, 1)) 
  (h₂ : e₂ = (1, 3)) 
  (h₃ : a = (-1, 2)) :
  ∃! (l₁ l₂ : ℝ), a = l₁ • e₁ + l₂ • e₂ ∧ l₁ = -1 ∧ l₂ = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_linear_combination_l312_31285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_ball_configuration_l312_31221

/-- Represents a cone with a given base radius -/
structure Cone where
  baseRadius : ℝ

/-- Represents a ball with a given radius -/
structure Ball where
  radius : ℝ

/-- Represents a configuration of three cones and a ball on a table -/
structure Configuration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  ball : Ball
  touchesAllCones : Prop
  centerEquidistant : Prop

/-- The theorem stating that under the given conditions, r must equal 1 -/
theorem cone_ball_configuration (r : ℝ) :
  let config : Configuration := {
    cone1 := { baseRadius := 2 * r }
    cone2 := { baseRadius := 3 * r }
    cone3 := { baseRadius := 10 * r }
    ball := { radius := 2 }
    touchesAllCones := True
    centerEquidistant := True
  }
  r = 1 := by
  sorry

#check cone_ball_configuration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_ball_configuration_l312_31221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximation_l312_31219

/-- The speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 36

/-- The time it takes for the train to pass the oak tree in seconds -/
noncomputable def passing_time : ℝ := 17.998560115190784

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_per_hr_to_m_per_s : ℝ := 1000 / 3600

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := train_speed * km_per_hr_to_m_per_s * passing_time

theorem train_length_approximation :
  ∃ ε > 0, |train_length - 179.99| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximation_l312_31219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_implies_a_equals_one_l312_31239

/-- The constant term in the expansion of (ax+1)(2x- 1/x)^5 -/
def constant_term (a : ℝ) : ℝ := -40 * a

/-- The theorem stating that if the constant term is -40, then a = 1 -/
theorem constant_term_implies_a_equals_one : 
  constant_term 1 = -40 → 1 = (fun a ↦ if constant_term a = -40 then a else 0) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_implies_a_equals_one_l312_31239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_distribution_l312_31289

theorem income_distribution (total_income : ℝ) (food_percent : ℝ) (education_percent : ℝ) (rent_percent : ℝ) :
  food_percent = 50 →
  education_percent = 15 →
  rent_percent = 50 →
  let remaining_after_food_education := total_income * (1 - (food_percent + education_percent) / 100);
  let rent_amount := remaining_after_food_education * (rent_percent / 100);
  let final_remaining := remaining_after_food_education - rent_amount;
  (final_remaining / total_income) * 100 = 17.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_distribution_l312_31289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l312_31267

/-- Given a train with speed excluding stoppages and stoppage time per hour,
    calculate the speed including stoppages -/
noncomputable def speed_including_stoppages (speed_excluding_stoppages : ℝ) (stoppage_time_minutes : ℝ) : ℝ :=
  let stoppage_time_hours := stoppage_time_minutes / 60
  let moving_time := 1 - stoppage_time_hours
  let distance_covered := speed_excluding_stoppages * moving_time
  distance_covered

/-- Theorem stating that for a train with speed 42 kmph excluding stoppages
    and stopping for 21.428571428571423 minutes per hour,
    the speed including stoppages is 27 kmph -/
theorem train_speed_theorem :
  speed_including_stoppages 42 21.428571428571423 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l312_31267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_is_twelve_l312_31230

/-- Represents a triangle with a given base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Represents a trapezoid with smaller base, larger base, and height -/
structure Trapezoid where
  smallerBase : ℝ
  largerBase : ℝ
  height : ℝ

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1 / 2) * t.base * t.height

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ := ((t.smallerBase + t.largerBase) / 2) * t.height

/-- Calculates the median of a trapezoid -/
noncomputable def trapezoidMedian (t : Trapezoid) : ℝ := (t.smallerBase + t.largerBase) / 2

theorem trapezoid_median_is_twelve (t : Triangle) (z : Trapezoid) 
    (h1 : t.base = 24)
    (h2 : t.height = z.height)
    (h3 : triangleArea t = trapezoidArea z)
    (h4 : z.smallerBase = z.largerBase / 2) : 
  trapezoidMedian z = 12 := by
  sorry

#check trapezoid_median_is_twelve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_median_is_twelve_l312_31230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_replacement_theorem_l312_31245

/-- Represents a month in a year, where January is 0 and December is 11 --/
def Month := Fin 12

/-- Calculates the number of months between two replacements --/
def months_between_replacements : ℕ := 7

/-- Calculates the total number of months for n replacements --/
def total_months (n : ℕ) : ℕ := months_between_replacements * (n - 1)

/-- Calculates the number of years passed for n replacements --/
def years_passed (n : ℕ) : ℕ := (total_months n) / 12

/-- Calculates the month of the nth replacement --/
def replacement_month (n : ℕ) : Month := 
  ⟨(total_months n) % 12, by
    apply Nat.mod_lt
    exact Nat.zero_lt_succ 11⟩

theorem battery_replacement_theorem (n : ℕ) :
  n = 30 → 
  years_passed n = 16 ∧ 
  replacement_month n = ⟨11, by norm_num⟩ := by
  sorry

#eval years_passed 30
#eval (replacement_month 30).val

end NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_replacement_theorem_l312_31245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_cylinder_volume_ratio_l312_31290

/-- Represents a rectangular parallelepiped. -/
structure RectangularParallelepiped where

/-- Represents a cylinder. -/
structure Cylinder where

/-- A rectangular parallelepiped is inscribed in a cylinder. -/
structure RectangularParallelepiped.InscribedInCylinder where

/-- The diagonal of the parallelepiped forms angles α and β with the adjacent sides of the base. -/
structure DiagonalFormAngles (α β : Real) where

/-- The ratio of volumes between two shapes. -/
def VolumeRatio (shape1 shape2 : Type) : Real := sorry

/-- The ratio of the volume of a rectangular parallelepiped inscribed in a cylinder
    to the volume of the cylinder, given the angles between the diagonal and the base sides. -/
theorem parallelepiped_cylinder_volume_ratio
  (α β : Real)
  (h_inscribed : RectangularParallelepiped.InscribedInCylinder)
  (h_diagonal_angles : DiagonalFormAngles α β) :
  VolumeRatio RectangularParallelepiped Cylinder =
    (4 * Real.cos α * Real.cos β) / (Real.pi * ((Real.cos α)^2 + (Real.cos β)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_cylinder_volume_ratio_l312_31290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l312_31272

theorem tan_triple_angle (θ : Real) (h1 : Real.sin θ = 5/13) (h2 : 0 < θ ∧ θ < Real.pi/2) :
  Real.tan (3*θ) = 145/78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l312_31272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_sum_of_squares_l312_31264

noncomputable def f (a b x : ℝ) : ℝ :=
  |a * Real.sin x + b * Real.cos x - 1| + |b * Real.sin x - a * Real.cos x|

theorem max_value_implies_sum_of_squares (a b : ℝ) :
  (∀ x, f a b x ≤ 11) ∧ (∃ x, f a b x = 11) → a^2 + b^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_sum_of_squares_l312_31264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_logarithms_and_root_l312_31233

theorem compare_logarithms_and_root : 
  (Real.log 0.2 / Real.log 10) < (Real.log 2 / Real.log 3) ∧ (Real.log 2 / Real.log 3) < 5^(1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_logarithms_and_root_l312_31233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_self_descriptive_numbers_l312_31262

def is_self_descriptive (n : ℕ) (num : List ℕ) : Prop :=
  num.length = n ∧ 
  (∀ i < n, num.get? i = some (num.count i))

def construct_number (n : ℕ) : List ℕ :=
  if n ≥ 7 then
    (n-4) :: 2 :: 1 :: (List.replicate (n-7) 0) ++ [1, 0, 0, 0]
  else if n = 5 then
    [2, 1, 2, 0, 0]
  else if n = 4 then
    [1, 2, 1, 0]  -- We choose one of the two possible solutions
  else
    []  -- Empty list for n = 2, 3, 6

theorem self_descriptive_numbers (n : ℕ) :
  (n ≥ 2) →
  (∃ num : List ℕ, is_self_descriptive n num) ↔
  (n ≥ 7 ∨ n = 5 ∨ n = 4) ∧
  is_self_descriptive n (construct_number n) :=
by
  sorry

#check self_descriptive_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_self_descriptive_numbers_l312_31262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_vertex_l312_31278

/-- The ellipse C defined by x^2/5 + y^2/4 = 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 / 4 = 1}

/-- The upper vertex B of the ellipse C -/
def B : ℝ × ℝ := (0, 2)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance_to_vertex (P : ℝ × ℝ) (h : P ∈ C) :
  distance P B ≤ 4 ∧ ∃ Q ∈ C, distance Q B = 4 := by
  sorry

#check max_distance_to_vertex

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_vertex_l312_31278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_and_increasing_l312_31297

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (1/2)^(|x| - 3)
noncomputable def g (x : ℝ) : ℝ := Real.log (x^2 - 4*x)

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, f x = y}
def B : Set ℝ := {x : ℝ | x^2 - 4*x > 0}

-- State the theorem
theorem union_and_increasing :
  (A ∪ B = {x : ℝ | x ≠ 0}) ∧
  (∀ x₁ x₂, x₁ > 4 → x₂ > x₁ → g x₂ > g x₁) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_and_increasing_l312_31297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reflected_midpoint_coords_specific_l312_31295

/-- Given two points A and B in the plane, this function calculates the sum of coordinates
    of the midpoint of segment AB after reflection over the y-axis. -/
noncomputable def sum_of_reflected_midpoint_coords (A B : ℝ × ℝ) : ℝ :=
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let reflected_midpoint := (-midpoint.1, midpoint.2)
  reflected_midpoint.1 + reflected_midpoint.2

/-- Theorem stating that the sum of coordinates of the midpoint of segment AB
    after reflection over the y-axis is -5, given A(3, -2) and B(15, 10). -/
theorem sum_of_reflected_midpoint_coords_specific : 
  sum_of_reflected_midpoint_coords (3, -2) (15, 10) = -5 := by
  -- Unfold the definition and simplify
  unfold sum_of_reflected_midpoint_coords
  -- Perform the arithmetic
  simp [add_div, sub_div]
  -- The result should now be obvious to Lean
  norm_num

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check sum_of_reflected_midpoint_coords (3, -2) (15, 10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reflected_midpoint_coords_specific_l312_31295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_roots_l312_31222

noncomputable def f (x : ℝ) : ℝ := |4 * x * (1 - x)|

theorem three_distinct_roots (t : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ∈ Set.Ioi 0 ∧
                      x₂ ∈ Set.Ioi 0 ∧
                      x₃ ∈ Set.Ioi 0 ∧
                      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                      f x₁ ^ 2 + (t - 3) * f x₁ + t - 2 = 0 ∧
                      f x₂ ^ 2 + (t - 3) * f x₂ + t - 2 = 0 ∧
                      f x₃ ^ 2 + (t - 3) * f x₃ + t - 2 = 0 ∧
                      ∀ x ∈ Set.Ioi 0,
                        f x ^ 2 + (t - 3) * f x + t - 2 = 0 →
                        x = x₁ ∨ x = x₂ ∨ x = x₃) ↔
  t = 2 ∨ t = 5 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_roots_l312_31222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_m_l312_31250

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  point : ℝ × ℝ
  equation : (point.1 / a) ^ 2 + (point.2 / b) ^ 2 = 1

/-- The focal length of an ellipse -/
noncomputable def FocalLength (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

/-- Theorem: For an ellipse with equation x^2/m + y^2/4 = 1 and focal length 2, m is either 3 or 5 -/
theorem ellipse_focal_length_m (m : ℝ) :
  (∃ a b : ℝ, Ellipse a b = Ellipse (Real.sqrt m) 2 ∧ FocalLength a b = 2) →
  (m = 3 ∨ m = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_m_l312_31250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l312_31213

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 3)

theorem omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ 
    (∀ x : ℝ, x₁ < x ∧ x < x₂ → f ω x ≠ 0) → x₂ - x₁ = 2) :
  ω = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l312_31213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l312_31249

noncomputable section

-- Define the triangle ABC
variable (A B C : Real) -- Angles
variable (a b c : Real) -- Sides

-- Given conditions
axiom side_a : a = 2
axiom side_b : b = 1
axiom angle_sum : A + B + C = Real.pi
axiom sine_rule : a / Real.sin A = b / Real.sin B
axiom cosine_rule : c^2 = a^2 + b^2 - 2*a*b*Real.cos C
axiom given_equation : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 1/2 * b

-- Statements to prove
theorem triangle_properties :
  Real.sin B = 1/2 ∧
  Real.cos B = Real.sqrt 3 / 2 ∧
  A = Real.pi/2 ∧
  (1/2 * a * b * Real.sin C = Real.sqrt 3 / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l312_31249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_divisors_of_1728_power_l312_31240

noncomputable def count_special_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => (Nat.divisors d).card = 1728) (Nat.divisors (n^n))).card

theorem special_divisors_of_1728_power : count_special_divisors 1728 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_divisors_of_1728_power_l312_31240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l312_31200

theorem cos_double_angle_special_case (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α - π / 4) = -3 / 5) : 
  Real.cos (2 * α) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l312_31200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_equation_l312_31220

/-- A line passing through point (2, 3) with intercepts on the axes that are opposite numbers -/
structure SpecialLine where
  -- The line passes through point (2, 3)
  passes_through : Set (ℝ × ℝ)
  -- The line has intercepts on the axes that are opposite numbers
  opposite_intercepts : ∃ a : ℝ, x_intercept = a ∧ y_intercept = -a
  -- Ensure the point (2, 3) is on the line
  point_on_line : (2, 3) ∈ passes_through

/-- The equation of the special line -/
def line_equation (l : SpecialLine) : Set (ℝ × ℝ) :=
  {(x, y) | (3 * x - 2 * y = 0) ∨ (x - y + 1 = 0)}

/-- Theorem stating that the equation of the special line is correct -/
theorem special_line_equation (l : SpecialLine) :
  l.passes_through = line_equation l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_equation_l312_31220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l312_31261

theorem triangle_ratio (A B C a b c : ℝ) : 
  A > 0 → B > 0 → C > 0 → 
  a > 0 → b > 0 → c > 0 →
  A + B + C = Real.pi →
  C = 2 * (A + B) →
  2 * b = a + c →
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →
  b / a = 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l312_31261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_house_number_l312_31226

def phone_number : List Nat := [3, 6, 4, 1, 5, 2, 8]

def is_valid_house_number (n : Nat) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n.repr.toList.map Char.toNat).sum = phone_number.sum ∧
  (n.repr.toList.map Char.toNat).Nodup

theorem largest_house_number :
  ∀ n : Nat, is_valid_house_number n → n ≤ 9875 :=
by
  intro n hn
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_house_number_l312_31226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l312_31256

/-- Represents the volume of a geometric shape -/
def Volume := ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (radius height : ℝ) : Volume :=
  (1/3) * Real.pi * radius^2 * height

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (radius height : ℝ) : Volume :=
  Real.pi * radius^2 * height

/-- Theorem: The height of water in the cylinder -/
theorem water_height_in_cylinder 
  (cone_radius : ℝ) 
  (cone_height : ℝ) 
  (cylinder_radius : ℝ) : 
  cone_radius = 12 → 
  cone_height = 18 → 
  cylinder_radius = 24 → 
  ∃ (cylinder_height : ℝ), 
    cylinderVolume cylinder_radius cylinder_height = coneVolume cone_radius cone_height ∧ 
    cylinder_height = 1.5 := by
  sorry

#check water_height_in_cylinder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l312_31256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l312_31247

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the problem
theorem triangle_problem (abc : Triangle) (m n : ℝ × ℝ) :
  -- Given conditions
  abc.A + abc.B + abc.C = π →
  m = (-Real.cos abc.B, Real.sin abc.C) →
  n = (-Real.cos abc.C, -Real.sin abc.B) →
  m.1 * n.1 + m.2 * n.2 = 1/2 →
  abc.b + abc.c = 4 →
  1/2 * abc.b * abc.c * Real.sin abc.A = Real.sqrt 3 →
  -- Conclusions
  abc.A = 2*π/3 ∧ abc.a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l312_31247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_exponential_equation_l312_31275

theorem solution_of_exponential_equation : ∃ x : ℝ, (2 : ℝ)^x = 8 ∧ x = 3 := by
  use 3
  constructor
  · simp [Real.rpow_def, Real.exp_log]
    norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_exponential_equation_l312_31275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_X_percent_B_approx_l312_31225

noncomputable def solution_A_weight : ℝ := 600
noncomputable def solution_B_weight : ℝ := 700
noncomputable def liquid_X_percent_A : ℝ := 0.008
noncomputable def liquid_X_percent_mixture : ℝ := 0.0174

noncomputable def liquid_X_percent_B : ℝ := 
  (liquid_X_percent_mixture * (solution_A_weight + solution_B_weight) - 
   liquid_X_percent_A * solution_A_weight) / solution_B_weight

theorem liquid_X_percent_B_approx :
  abs (liquid_X_percent_B - 0.0255) < 0.0001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_X_percent_B_approx_l312_31225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_midpoint_ratio_l312_31269

/-- Represents a trapezoid with bases of length a and b -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  ha : a > 0
  hb : b > 0
  hh : h > 0
  hab : a > b

/-- The area of the trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ := (t.a + t.b) * t.h / 2

/-- The area of the quadrilateral formed by joining the midpoints -/
noncomputable def midpoint_quad_area (t : Trapezoid) : ℝ := (t.a - t.b) * t.h / 4

/-- The theorem stating the relationship between the areas and the ratio of bases -/
theorem trapezoid_midpoint_ratio (t : Trapezoid) 
  (h : midpoint_quad_area t = trapezoid_area t / 4) : 
  t.a / t.b = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_midpoint_ratio_l312_31269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_relative_to_U_l312_31286

-- Define the set U
def U : Set ℝ := {y | ∃ x : ℝ, y = 2^x ∧ x ≥ -1}

-- Define the set A
def A : Set ℝ := {x | 1/(x-1) ≥ 1}

-- Define the complement of A relative to U
def C_U_A : Set ℝ := U \ A

-- Theorem statement
theorem complement_of_A_relative_to_U :
  C_U_A = Set.Icc (1/2) 1 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_relative_to_U_l312_31286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_terms_l312_31288

noncomputable def binomial_expr (x : ℝ) := (2 * x^2 - 1/x)^6

noncomputable def general_term (r : ℕ) (x : ℝ) : ℝ := 
  (-1)^r * 2^(6-r) * (Nat.choose 6 r) * x^(12-3*r)

theorem binomial_expansion_terms :
  ∃ (constant_term middle_term : ℝ),
    (∀ x, general_term 4 x = constant_term) ∧
    (constant_term = 60) ∧
    (∀ x, general_term 3 x = middle_term * x^3) ∧
    (middle_term = -160) := by
  sorry

#check binomial_expansion_terms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_terms_l312_31288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_owes_22264_l312_31236

/-- Represents an employee with their working hours and hourly rate -/
structure Employee where
  days_per_month : ℕ
  hours_per_day : ℕ
  hourly_rate : ℕ

/-- Calculates the total amount Janet owes for one month -/
def total_amount_owed (employees : List Employee) (fica_rate : ℚ) (retirement_rate : ℚ) : ℚ :=
  let total_wages := (employees.map (fun e => (e.days_per_month : ℚ) * (e.hours_per_day : ℚ) * (e.hourly_rate : ℚ))).sum
  let fica_tax := fica_rate * total_wages
  let retirement_contribution := retirement_rate * total_wages
  total_wages + fica_tax + retirement_contribution

/-- The main theorem stating the total amount Janet owes -/
theorem janet_owes_22264 :
  let employees : List Employee := [
    ⟨20, 6, 15⟩, ⟨25, 8, 15⟩, ⟨18, 7, 15⟩, ⟨22, 9, 15⟩,  -- Warehouse workers
    ⟨26, 10, 20⟩, ⟨25, 9, 20⟩  -- Managers
  ]
  let fica_rate : ℚ := 1/10
  let retirement_rate : ℚ := 1/20
  total_amount_owed employees fica_rate retirement_rate = 22264 := by
  sorry

#eval total_amount_owed [
  ⟨20, 6, 15⟩, ⟨25, 8, 15⟩, ⟨18, 7, 15⟩, ⟨22, 9, 15⟩,  -- Warehouse workers
  ⟨26, 10, 20⟩, ⟨25, 9, 20⟩  -- Managers
] (1/10) (1/20)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_owes_22264_l312_31236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l312_31208

theorem sum_of_squares_of_roots : 
  ∀ r₁ r₂ : ℝ, (r₁^2 - 11*r₁ + 12 = 0) ∧ (r₂^2 - 11*r₂ + 12 = 0) → r₁^2 + r₂^2 = 97 :=
by
  intros r₁ r₂ h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l312_31208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_2015_factorial_l312_31207

def trailingZeros (n : ℕ) : ℕ := sorry

def vp (p n : ℕ) : ℕ := sorry

theorem zeros_2015_factorial :
  trailingZeros 2015 = vp 5 (Nat.factorial 2015) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_2015_factorial_l312_31207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_april_price_is_one_twenty_l312_31270

/-- Represents the sales and pricing data for eggs over three months -/
structure EggSalesData where
  april_price : ℚ
  may_price : ℚ
  june_price : ℚ
  may_sales : ℚ
  april_sales : ℚ
  june_sales : ℚ

/-- The average price of eggs over the three-month period -/
noncomputable def average_price (data : EggSalesData) : ℚ :=
  (data.april_price * data.april_sales + data.may_price * data.may_sales + data.june_price * data.june_sales) /
  (data.april_sales + data.may_sales + data.june_sales)

/-- Theorem stating that given the conditions, the price per dozen in April was $1.20 -/
theorem april_price_is_one_twenty
  (data : EggSalesData)
  (h1 : data.may_price = 6/5)
  (h2 : data.june_price = 3)
  (h3 : data.april_sales = 2/3 * data.may_sales)
  (h4 : data.june_sales = 2 * data.april_sales)
  (h5 : average_price data = 2) :
  data.april_price = 6/5 := by
  sorry

#check april_price_is_one_twenty

end NUMINAMATH_CALUDE_ERRORFEEDBACK_april_price_is_one_twenty_l312_31270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l312_31271

/-- The probability of success on each turn -/
noncomputable def p : ℝ := 1/3

/-- The number of players -/
def n : ℕ := 3

/-- The probability that the first player wins the game -/
noncomputable def first_player_wins : ℝ := 9/19

theorem game_probability :
  first_player_wins = p * (1 - p^n)⁻¹ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l312_31271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l312_31255

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (-1 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Define the curve C
def curve_C (x : ℝ) : ℝ := x^2

-- Define point M
def point_M : ℝ × ℝ := (-1, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_product : ∃ (t1 t2 : ℝ),
  (line_l t1).2 = curve_C (line_l t1).1 ∧
  (line_l t2).2 = curve_C (line_l t2).1 ∧
  t1 ≠ t2 ∧
  distance point_M (line_l t1) * distance point_M (line_l t2) = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l312_31255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_equation_l312_31251

noncomputable def real_sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem function_satisfying_equation (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), (f x * f y - f (x * y)) / 5 = x + y + 2) :
  ∀ (x : ℝ), f x = (10 * x + 21 + real_sqrt 41) / (1 + real_sqrt 41) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_equation_l312_31251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l312_31276

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r 
    such that a(n+1) = r * a(n) for all n -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence {a_n} where a₄ + a₆ = π, 
    prove that a₅a₃ + 2a₅² + a₅a₇ = π² -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : IsGeometricSequence a) 
    (sum_condition : a 4 + a 6 = Real.pi) : 
    a 5 * a 3 + 2 * (a 5)^2 + a 5 * a 7 = Real.pi^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l312_31276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_equals_singleton_S_1024_equals_singleton_1024_l312_31293

def S : ℕ → Set ℕ
  | 0 => ∅  -- Add a case for 0
  | 1 => {1}
  | 2 => {2}
  | (n + 3) => {k : ℕ | (k - 1 ∈ S (n + 2)) ≠ (k ∈ S (n + 1))}

theorem S_equals_singleton (n : ℕ) (h : n ≥ 1) : S n = {n} := by
  sorry

theorem S_1024_equals_singleton_1024 : S 1024 = {1024} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_equals_singleton_S_1024_equals_singleton_1024_l312_31293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extract_amount_second_tube_l312_31212

noncomputable def initial_volume : ℝ := 200

noncomputable def first_tube_concentration (n : ℕ) : ℝ := (3/4) ^ n

noncomputable def second_tube_concentration (x : ℝ) : ℝ := ((initial_volume - x) / initial_volume) ^ 2

theorem extract_amount_second_tube :
  ∃ x : ℝ, x > 0 ∧ x < initial_volume ∧
  (first_tube_concentration 4) / (second_tube_concentration x) = 9/16 →
  x = 50 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extract_amount_second_tube_l312_31212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_lattice_hexagon_with_consecutive_squared_sides_l312_31202

/-- A point on an integer lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A hexagon with vertices on an integer lattice -/
structure LatticeHexagon where
  vertices : Vector LatticePoint 6

/-- The squared distance between two lattice points -/
def squared_distance (p q : LatticePoint) : ℤ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- The squared lengths of the sides of a hexagon -/
def side_lengths_squared (h : LatticeHexagon) : Vector ℤ 6 :=
  Vector.ofFn fun i =>
    squared_distance (h.vertices.get i) (h.vertices.get ((i + 1) % 6))

/-- Six consecutive positive integers, starting from n+1 -/
def consecutive_integers (n : ℕ) : Vector ℤ 6 :=
  Vector.ofFn fun i => (n + i + 1 : ℤ)

theorem no_lattice_hexagon_with_consecutive_squared_sides :
  ¬∃ (h : LatticeHexagon) (n : ℕ), side_lengths_squared h = consecutive_integers n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_lattice_hexagon_with_consecutive_squared_sides_l312_31202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_six_five_chain_l312_31218

/-- Represents a domino tile with two numbers -/
structure Domino :=
  (first : Nat)
  (second : Nat)
  (valid : first ≤ 6 ∧ second ≤ 6)

/-- The set of all domino tiles in a standard double-six set -/
def StandardDominoSet : Finset Domino := sorry

/-- The number of tiles in a standard domino set -/
axiom standard_domino_set_size : StandardDominoSet.card = 28

/-- Each number (0 through 6) appears exactly 7 times in the standard domino set -/
axiom number_frequency (n : Nat) (h : n ≤ 6) :
  (StandardDominoSet.filter (λ d => d.first = n ∨ d.second = n)).card = 7

/-- A domino chain is a list of domino tiles where adjacent tiles match -/
def DominoChain : List Domino → Prop := sorry

/-- Theorem: It's impossible to arrange all 28 domino tiles in a chain
    with a six on one end and a five on the other -/
theorem impossible_six_five_chain :
  ¬ ∃ (chain : List Domino),
    DominoChain chain ∧
    chain.length = 28 ∧
    (chain.head?.map (λ d => d.first = 6 ∨ d.second = 6)).isSome ∧
    (chain.get? (chain.length - 1)).map (λ d => d.first = 5 ∨ d.second = 5) = some true ∧
    (∀ d : Domino, d ∈ chain ↔ d ∈ StandardDominoSet) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_six_five_chain_l312_31218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_p_value_l312_31203

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  mean : ℝ
  variance : ℝ

/-- Properties of the binomial variable ξ -/
def ξ (n : ℕ) (p : ℝ) : BinomialVariable where
  n := n
  p := p
  mean := 300
  variance := 200

/-- Theorem stating that p = 1/3 for the given binomial variable -/
theorem binomial_p_value (n : ℕ) (p : ℝ) :
  (ξ n p).mean = n * p → (ξ n p).variance = n * p * (1 - p) → p = 1/3 := by
  sorry

#check binomial_p_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_p_value_l312_31203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreasing_l312_31243

-- Define the function f(x) = 3/x
noncomputable def f (x : ℝ) : ℝ := 3 / x

-- Theorem statement
theorem inverse_proportion_decreasing (x1 x2 : ℝ) :
  x1 ≠ 0 → x2 ≠ 0 → ((x1 < x2 ∧ 0 < x1 ∧ 0 < x2) ∨ (x1 < x2 ∧ x1 < 0 ∧ x2 < 0)) →
  f x2 < f x1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreasing_l312_31243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l312_31229

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0  -- We define a₀ as 0 for convenience
  | n + 1 => f (sequence_a n)

theorem sequence_a_properties (a₁ : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < 1) :
  (∀ n : ℕ, 0 < sequence_a (n + 1) ∧ sequence_a (n + 1) < sequence_a n ∧ sequence_a n < 1) ∧
  (∀ n : ℕ, sequence_a (n + 1) < (1/6) * (sequence_a n)^3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l312_31229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_calculation_l312_31209

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: For a trapezium with parallel sides of 20 cm and 18 cm, and an area of 285 square centimeters, the distance between the parallel sides is 15 cm. -/
theorem trapezium_height_calculation :
  ∃ h : ℝ, trapezium_area 20 18 h = 285 ∧ h = 15 :=
by
  use 15
  constructor
  · simp [trapezium_area]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_calculation_l312_31209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l312_31248

def number : ℕ := 8^4 * 7^3 * 9^1 * 5^5

theorem number_of_factors (n : ℕ) (h : n = number) : 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 936 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l312_31248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_sqrt_2_l312_31244

-- Define the circle C in Cartesian coordinates
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y = 0

-- Define the line l in parametric form
def line_l (t x y : ℝ) : Prop := x = -1 + t ∧ y = t

-- Define the ray OM in polar coordinates
def ray_OM (θ : ℝ) : Prop := θ = 3*Real.pi/4

-- Define the conversion between Cartesian and polar coordinates
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance between two points in polar coordinates
noncomputable def polar_distance (ρ₁ θ₁ ρ₂ θ₂ : ℝ) : ℝ := 
  Real.sqrt ((ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂)^2 + (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)^2)

theorem length_PQ_is_sqrt_2 :
  ∃ (ρ_P ρ_Q : ℝ),
    let (x_P, y_P) := polar_to_cartesian ρ_P (3*Real.pi/4)
    let (x_Q, y_Q) := polar_to_cartesian ρ_Q (3*Real.pi/4)
    circle_C x_P y_P ∧
    (∃ t, line_l t x_Q y_Q) ∧
    polar_distance ρ_P (3*Real.pi/4) ρ_Q (3*Real.pi/4) = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_sqrt_2_l312_31244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_minus_floor_product_l312_31299

/-- Floor function definition -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The problem statement -/
theorem power_minus_floor_product (n : ℕ) (h : n > 0) :
  (n : ℝ)^(n+1) - floor ((n : ℝ)^(n+1) / (n+1 : ℝ)) * (n+1 : ℝ) = n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_minus_floor_product_l312_31299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l312_31277

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_ratio :
  let si := simple_interest 2800 5 3
  let ci := compound_interest 4000 10 2
  si / ci = 1 / 2 := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l312_31277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_fraction_values_l312_31231

open Real

-- Define the set of fractions n/3 where n is an integer
def FractionsThirds : Set ℝ := {x | ∃ n : ℤ, x = n / 3}

-- Define the dot product for 2D vectors
def dot (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Define the magnitude of a 2D vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (dot v v)

-- Define the angle between two 2D vectors
noncomputable def angle (a b : ℝ × ℝ) : ℝ := Real.arccos ((dot a b) / (magnitude a * magnitude b))

-- Define ab and ba
noncomputable def ab (a b : ℝ × ℝ) : ℝ := (dot a b) / (dot b b)
noncomputable def ba (a b : ℝ × ℝ) : ℝ := (dot b a) / (dot a a)

theorem vector_fraction_values (a b : ℝ × ℝ) :
  (magnitude a ≥ magnitude b) →
  (magnitude b > 0) →
  (0 < angle a b) →
  (angle a b < π / 6) →
  (ab a b ∈ FractionsThirds) →
  (ba a b ∈ FractionsThirds) →
  (ab a b = 4/3 ∨ ab a b = 7/3 ∨ ab a b = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_fraction_values_l312_31231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rawot_market_properties_l312_31224

/-- Represents the seasons --/
inductive Season
| Spring
| Summer
| Autumn
| Winter

/-- Represents the supply-demand model for 'rawot' --/
structure RawotMarket where
  demandSlope : ℝ
  demandIntercept : Season → ℝ
  supplySlope : ℝ
  supplyIntercept : Season → ℝ

/-- Calculates the equilibrium price for a given season --/
noncomputable def equilibriumPrice (market : RawotMarket) (s : Season) : ℝ :=
  (market.demandIntercept s - market.supplyIntercept s) / (market.supplySlope + market.demandSlope)

/-- Calculates the equilibrium quantity for a given season --/
noncomputable def equilibriumQuantity (market : RawotMarket) (s : Season) : ℝ :=
  market.supplySlope * (equilibriumPrice market s) + market.supplyIntercept s

/-- Theorem stating the properties of the Rawot market --/
theorem rawot_market_properties (market : RawotMarket) :
  -- Spring total revenue is 90
  equilibriumPrice market Season.Spring * equilibriumQuantity market Season.Spring = 90 ∧
  -- Winter quantity is 5
  equilibriumQuantity market Season.Winter = 5 ∧
  -- Summer deficit when price is frozen at winter price is 10
  (market.demandIntercept Season.Summer - market.demandSlope * equilibriumPrice market Season.Winter) -
    (market.supplySlope * equilibriumPrice market Season.Winter + market.supplyIntercept Season.Summer) = 10 ∧
  -- Summer total expenditure is 172.5
  equilibriumPrice market Season.Summer * equilibriumQuantity market Season.Summer = 172.5 ∧
  -- Minimum price for zero demand is 19 and is season-independent
  (∀ s : Season, market.demandIntercept s / market.demandSlope = 19) ∧
  -- Producers only sell if price > 4
  (∀ s : Season, market.supplyIntercept s / market.supplySlope + 4 > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rawot_market_properties_l312_31224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gear_speed_proportion_l312_31217

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  angularSpeed : ℝ

/-- Represents a system of four interconnected gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  meshingCondition : A.teeth * A.angularSpeed = B.teeth * B.angularSpeed ∧
                     B.teeth * B.angularSpeed = C.teeth * C.angularSpeed ∧
                     C.teeth * C.angularSpeed = D.teeth * D.angularSpeed

theorem gear_speed_proportion (sys : GearSystem) :
  ∃ (k : ℝ), k ≠ 0 ∧ 
     sys.A.angularSpeed = k * (sys.B.teeth * sys.C.teeth * sys.D.teeth) ∧
     sys.B.angularSpeed = k * (sys.A.teeth * sys.C.teeth * sys.D.teeth) ∧
     sys.C.angularSpeed = k * (sys.A.teeth * sys.B.teeth * sys.D.teeth) ∧
     sys.D.angularSpeed = k * (sys.A.teeth * sys.B.teeth * sys.C.teeth) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gear_speed_proportion_l312_31217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l312_31281

/-- A function is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The given function -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 3) + φ)

theorem phi_value (φ : ℝ) (h1 : IsEven (f φ)) (h2 : 0 < φ) (h3 : φ < Real.pi) :
  φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l312_31281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l312_31265

noncomputable def v : Fin 3 → ℝ := ![3, -2, 4]

noncomputable def P : Matrix (Fin 3) (Fin 3) ℝ :=
  let vNormSquared := (v 0) ^ 2 + (v 1) ^ 2 + (v 2) ^ 2
  Matrix.of (λ i j => (v i * v j) / vNormSquared)

theorem det_projection_matrix_zero :
  Matrix.det P = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l312_31265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_shapes_l312_31283

/-- Given a right triangle ABC with legs a and b, prove the properties of the largest inscribed square and rectangle -/
theorem largest_inscribed_shapes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let triangle := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 / a + p.2 / b ≤ 1}
  ∃ (s : ℝ), s = a * b / (a + b) ∧
    (∀ (t : ℝ), t > 0 → {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ t ∧ 0 ≤ p.2 ∧ p.2 ≤ t} ⊆ triangle → t ≤ s) ∧
  ∃ (w h : ℝ), w = a / 2 ∧ h = b / 2 ∧
    (∀ (u v : ℝ), u > 0 → v > 0 → {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ u ∧ 0 ≤ p.2 ∧ p.2 ≤ v} ⊆ triangle → u * v ≤ w * h) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_shapes_l312_31283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2022_l312_31296

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => -1/4
  | n+1 => 1 - 1 / mySequence n

theorem mySequence_2022 : mySequence 2021 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2022_l312_31296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sum_equals_pi_over_four_plus_ln_two_l312_31273

theorem integral_sum_equals_pi_over_four_plus_ln_two :
  ∫ (x : ℝ) in (Set.Icc 0 1), Real.sqrt (1 - x^2) + ∫ (x : ℝ) in (Set.Icc 1 2), 1 / x = π / 4 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sum_equals_pi_over_four_plus_ln_two_l312_31273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_leaves_after_two_days_l312_31242

-- Define the work rates and durations
noncomputable def work_rate_A : ℝ := 1 / 20
noncomputable def work_rate_B : ℝ := 1 / 30
noncomputable def work_rate_C : ℝ := 1 / 10
def total_duration : ℝ := 15.000000000000002
def C_duration : ℝ := 4

-- Define the theorem
theorem A_leaves_after_two_days :
  ∃ (x : ℝ),
    x > 0 ∧
    x < total_duration ∧
    work_rate_C * C_duration +
    (work_rate_A + work_rate_B) * x +
    work_rate_B * (total_duration - x) = 1 ∧
    x = 2 := by
  -- Proof goes here
  sorry

#eval total_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_leaves_after_two_days_l312_31242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_20_primes_l312_31268

-- Define a function to get the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Define a function to sum the first n prime numbers
def sumFirstNPrimes (n : ℕ) : ℕ :=
  (List.range n).map (fun i => nthPrime (i + 1)) |>.sum

-- Theorem statement
theorem sum_first_20_primes : sumFirstNPrimes 20 = 639 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_20_primes_l312_31268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l312_31252

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

-- Define the equation
def equation (x : ℝ) : Prop :=
  (floor x : ℝ) * frac x = 2005 * x

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 0 ∨ x = -1 / 2006) :=
by
  sorry

#check equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l312_31252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_l312_31234

theorem cone_angle (R : ℝ) (r : ℝ) (θ : ℝ) : 
  (π * R = 2 * π * r) →  -- lateral surface unfolded forms semicircle with circumference 2πr
  (R = r / Real.cos θ) →      -- relationship between slant height, base radius, and angle
  (θ = Real.arccos (1/2)) -- angle is arccos(1/2), which is 60°
  :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_l312_31234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_CD_BD_l312_31274

noncomputable section

open EuclideanGeometry

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points D and E on BC and AC respectively
variable (D E : EuclideanSpace ℝ (Fin 2))

-- Define the intersection point T of AD and BE
variable (T : EuclideanSpace ℝ (Fin 2))

-- Axiom: D is on BC
axiom D_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • B + t • C

-- Axiom: E is on AC
axiom E_on_AC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (1 - s) • A + s • C

-- Axiom: T is on AD
axiom T_on_AD : ∃ u : ℝ, 0 ≤ u ∧ u ≤ 1 ∧ T = (1 - u) • A + u • D

-- Axiom: T is on BE
axiom T_on_BE : ∃ v : ℝ, 0 ≤ v ∧ v ≤ 1 ∧ T = (1 - v) • B + v • E

-- Axiom: AT/DT = 2
axiom ratio_AT_DT : dist A T / dist D T = 2

-- Axiom: BT/ET = 3
axiom ratio_BT_ET : dist B T / dist E T = 3

-- Theorem: CD/BD = 3/5
theorem ratio_CD_BD : dist C D / dist B D = 3/5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_CD_BD_l312_31274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l312_31216

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a line parameterization -/
structure LineParam where
  point : Vector2D
  direction : Vector2D

/-- The line equation y = (5/3)x - 17/3 -/
def lineEquation (v : Vector2D) : Prop :=
  v.y = (5/3) * v.x - 17/3

/-- Checks if a vector is a scalar multiple of [3, 5] or represents the correct slope -/
def isValidDirection (v : Vector2D) : Prop :=
  ∃ (k : ℝ), v = Vector2D.mk (3 * k) (5 * k) ∨ v.y / v.x = 5/3

/-- Checks if a parameterization is valid for the given line -/
def isValidParam (p : LineParam) : Prop :=
  lineEquation p.point ∧ isValidDirection p.direction

/-- The parameterizations to be checked -/
noncomputable def paramA : LineParam := LineParam.mk (Vector2D.mk 4 1) (Vector2D.mk (-3) (-5))
noncomputable def paramB : LineParam := LineParam.mk (Vector2D.mk 17 5) (Vector2D.mk 6 10)
noncomputable def paramC : LineParam := LineParam.mk (Vector2D.mk 2 (-7/3)) (Vector2D.mk (3/5) 1)
noncomputable def paramD : LineParam := LineParam.mk (Vector2D.mk (14/5) (-1)) (Vector2D.mk 1 (3/5))
noncomputable def paramE : LineParam := LineParam.mk (Vector2D.mk 0 (-17/3)) (Vector2D.mk 15 (-25))

/-- The main theorem to be proved -/
theorem valid_parameterizations :
  isValidParam paramA ∧
  isValidParam paramC ∧
  ¬isValidParam paramB ∧
  ¬isValidParam paramD ∧
  ¬isValidParam paramE := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l312_31216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_approx_l312_31215

/-- The length of the shortest side in a right triangle with other sides 5 and 12 -/
noncomputable def shortest_side : ℝ :=
  Real.sqrt 119

theorem shortest_side_approx :
  |shortest_side - 10.91| < 0.005 := by
  sorry

#eval Float.sqrt 119

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_approx_l312_31215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_closed_under_mult_l312_31223

-- Define the property of being closed under multiplication
def ClosedUnderMult (S : Set Int) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → (a * b) ∈ S

-- Define the main theorem
theorem at_least_one_closed_under_mult
  (T V : Set Int)
  (h1 : T ∪ V = Set.univ)
  (h2 : T ∩ V = ∅)
  (h3 : T.Nonempty)
  (h4 : V.Nonempty)
  (h5 : ∀ a b c, a ∈ T → b ∈ T → c ∈ T → (a * b * c) ∈ T)
  (h6 : ∀ x y z, x ∈ V → y ∈ V → z ∈ V → (x * y * z) ∈ V) :
  ClosedUnderMult T ∨ ClosedUnderMult V :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_closed_under_mult_l312_31223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l312_31282

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, |x| < 0)) ↔ (∃ x : ℝ, |x| ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l312_31282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l312_31205

/-- The hyperbola and line properties -/
structure HyperbolaConfig where
  b : ℝ
  /-- The hyperbola equation: x^2 - y^2/b^2 = 1 -/
  hyperbola : (x y : ℝ) → Prop := λ x y ↦ x^2 - y^2/b^2 = 1
  /-- The line equation: y = x + 1 -/
  line : (x y : ℝ) → Prop := λ x y ↦ y = x + 1
  /-- The left vertex A is at (-1, 0) -/
  left_vertex : ℝ × ℝ := (-1, 0)
  /-- B is the midpoint of AC -/
  b_midpoint : (xB yB xC yC : ℝ) → Prop := λ xB yB xC yC ↦ 2*xB = -1 + xC ∧ 2*yB = yC

/-- The theorem stating the eccentricity of the hyperbola -/
theorem hyperbola_eccentricity (config : HyperbolaConfig) : 
  ∃ (xB yB xC yC : ℝ), 
    config.line xB yB ∧ 
    config.line xC yC ∧ 
    config.b_midpoint xB yB xC yC ∧
    let e := Real.sqrt 10
    (∀ (x y : ℝ), config.hyperbola x y → 
      (x^2 + y^2) = e^2 * x^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l312_31205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_curve_and_slope_l312_31294

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  let ρ := 2 * Real.cos θ - 4 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos α, 1 + t * Real.sin α)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem line_intersects_curve_and_slope :
  ∃ (t1 t2 α : ℝ), 
    (∃ (θ1 θ2 : ℝ), curve_C θ1 = line_l t1 α ∧ curve_C θ2 = line_l t2 α) ∧ 
    distance (line_l t1 α) (line_l t2 α) = 3 * Real.sqrt 2 →
    (Real.tan α = 1 ∨ Real.tan α = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_curve_and_slope_l312_31294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l312_31238

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (ω * x + φ)

-- State the theorem
theorem function_properties 
  (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : 0 < φ ∧ φ < Real.pi) 
  (h3 : ∀ x : ℝ, f ω φ (x + 4 * Real.pi / ω) = f ω φ x) 
  (h4 : ∀ x : ℝ, f ω φ x = f ω φ (-x)) 
  (h5 : ∀ a b c A B C : ℝ, 
    (2 * a - c) * Real.cos B = b * Real.cos C → 
    (0 < A ∧ A < 2 * Real.pi / 3) → 
    (0 < B ∧ B < Real.pi) → 
    (0 < C ∧ C < Real.pi) → 
    A + B + C = Real.pi → 
    (5 / 2 : ℝ) < (f ω φ A)^2 + (f ω φ C)^2 ∧ (f ω φ A)^2 + (f ω φ C)^2 ≤ 3) :
  ω = 1 / 2 ∧ φ = Real.pi / 2 := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l312_31238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_increase_factor_l312_31232

/-- Represents a cylinder with a given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Represents the transformation of the cylinder -/
def transform (c : Cylinder) : Cylinder :=
  { radius := 2.5 * c.radius,  -- Increased by 150%
    height := 3 * c.height }   -- Tripled

theorem volume_increase_factor (c : Cylinder) :
  volume (transform c) = 18.75 * volume c := by
  sorry

#check volume_increase_factor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_increase_factor_l312_31232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2021_bounds_l312_31279

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => sequence_a (n + 2) * (sequence_a (n + 2)^2 + 1) / (sequence_a (n + 1)^2 + 1)

theorem a_2021_bounds : 63 ≤ sequence_a 2021 ∧ sequence_a 2021 < 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2021_bounds_l312_31279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l312_31227

/-- The ellipse on which point P moves -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

/-- The circle on which point A moves -/
def circle_A (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 16

/-- The circle on which point B moves -/
def circle_B (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 4

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The theorem stating the maximum value of PA + PB -/
theorem max_sum_distances :
  ∃ (xP yP xA yA xB yB : ℝ),
    ellipse xP yP ∧
    circle_A xA yA ∧
    circle_B xB yB ∧
    (∀ (xP' yP' xA' yA' xB' yB' : ℝ),
      ellipse xP' yP' →
      circle_A xA' yA' →
      circle_B xB' yB' →
      distance xP yP xA yA + distance xP yP xB yB ≥
      distance xP' yP' xA' yA' + distance xP' yP' xB' yB') ∧
    distance xP yP xA yA + distance xP yP xB yB = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l312_31227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l312_31246

theorem problem_solution :
  ∀ (a : ℤ) (b : ℚ) (c : ℕ),
    (∀ n : ℤ, n < 0 → n ≤ a) →
    (∀ q : ℚ, q ≠ 0 → abs b ≤ abs q) →
    (c⁻¹ : ℚ) = c →
    a^2024 + 2023 * b - c^2023 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l312_31246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_distances_l312_31280

-- Define the circle on which M moves
def circleM (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point N
def N : ℝ × ℝ := (4, 0)

-- Define P as the midpoint of MN
noncomputable def P (x₀ y₀ : ℝ) : ℝ × ℝ := ((x₀ + 4) / 2, y₀ / 2)

-- Define the line 3x + 4y - 86 = 0
def linePQ (x y : ℝ) : Prop := 3 * x + 4 * y - 86 = 0

-- State the theorem
theorem trajectory_and_distances :
  ∀ x₀ y₀ : ℝ, circleM x₀ y₀ →
  let (x, y) := P x₀ y₀
  (x - 2)^2 + y^2 = 1 ∧
  (∃ d : ℝ, d = 17 ∧ ∀ x' y' : ℝ, linePQ x' y' → d ≥ Real.sqrt ((x - x')^2 + (y - y')^2)) ∧
  (∃ d : ℝ, d = 15 ∧ ∀ x' y' : ℝ, linePQ x' y' → d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_distances_l312_31280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_arithmetic_sequence_l312_31235

/-- 
Given a triangle ABC with internal angles A, B, and C,
if tan A, tan B, tan C, 2tan B form an arithmetic sequence in that order,
then sin 2B = 4/5
-/
theorem triangle_tan_arithmetic_sequence (A B C : ℝ) :
  (∃ d : ℝ, Real.tan A = Real.tan B - d ∧ 
            Real.tan C = Real.tan B + d ∧ 
            2 * Real.tan B = Real.tan C + d) →
  Real.sin (2 * B) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tan_arithmetic_sequence_l312_31235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_sufficient_not_necessary_l312_31298

/-- A curve passes through the origin if there exists an x such that y = 0 when that x is substituted into the equation of the curve. -/
def passes_through_origin (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

/-- The curve y = sin(2x + φ) -/
noncomputable def curve (φ : ℝ) : ℝ → ℝ :=
  λ x => Real.sin (2 * x + φ)

/-- φ = π is a sufficient but not necessary condition for the curve to pass through the origin -/
theorem pi_sufficient_not_necessary :
  (∀ φ, φ = Real.pi → passes_through_origin (curve φ)) ∧
  ¬(∀ φ, passes_through_origin (curve φ) → φ = Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_sufficient_not_necessary_l312_31298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l312_31257

/-- A point in a plane -/
structure Point :=
  (x y : ℝ)

/-- A quadrilateral in a plane -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- The area of a quadrilateral -/
noncomputable def area (quad : Quadrilateral) : ℝ := sorry

/-- The length of a line segment between two points -/
noncomputable def length (A B : Point) : ℝ := sorry

/-- The angle between three points -/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- Parallel line segments -/
def parallel (A B C D : Point) : Prop := sorry

theorem quadrilateral_area (PQRS : Quadrilateral) :
  parallel PQRS.P PQRS.Q PQRS.R PQRS.S →
  parallel PQRS.P PQRS.R PQRS.Q PQRS.S →
  length PQRS.P PQRS.Q = 1 →
  length PQRS.R PQRS.S = 1 →
  length PQRS.P PQRS.R = 2 →
  length PQRS.Q PQRS.S = 2 →
  angle PQRS.Q PQRS.P PQRS.S = Real.pi / 2 →
  area PQRS = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l312_31257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_value_l312_31260

/-- A line in 2D space defined by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A line in 2D space defined by a standard equation ax + by = c -/
structure StandardLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the slope of a parametric line -/
noncomputable def paramSlope (l : ParametricLine) : ℝ :=
  sorry

/-- Check if two lines are perpendicular -/
def isPerpendicular (l1 : ParametricLine) (l2 : StandardLine) : Prop :=
  sorry

theorem perpendicular_lines_k_value :
  let l1 : ParametricLine := { x := λ t => 1 - 2*t, y := λ t => 2 + 3*t }
  let l2 : StandardLine := { a := 4, b := k, c := 1 }
  isPerpendicular l1 l2 → k = -6 :=
by
  sorry

#check perpendicular_lines_k_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_value_l312_31260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l312_31263

theorem divisibility_condition (n p : ℕ) (h_prime : Nat.Prime p) (h_bound : n ≤ 2 * p) :
  (((p : ℕ) - 1) * n + 1) % (n ^ (p - 1)) = 0 ↔
    (n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l312_31263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modelA_better_fit_l312_31204

-- Define the two models
def modelA (x : ℝ) : ℝ := x^2 + 1
def modelB (x : ℝ) : ℝ := 3*x - 1

-- Define the data points
def dataPoints : List (ℝ × ℝ) := [(1, 2), (2, 5), (3, 10.2)]

-- Define a function to calculate the sum of squared errors
def sumSquaredErrors (model : ℝ → ℝ) (points : List (ℝ × ℝ)) : ℝ :=
  points.foldl (fun acc (x, y) => acc + (model x - y)^2) 0

-- Theorem statement
theorem modelA_better_fit :
  sumSquaredErrors modelA dataPoints < sumSquaredErrors modelB dataPoints :=
by
  -- Evaluate both sides
  have h1 : sumSquaredErrors modelA dataPoints = 0.04 := by sorry
  have h2 : sumSquaredErrors modelB dataPoints = 2.04 := by sorry
  -- Compare the results
  rw [h1, h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_modelA_better_fit_l312_31204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_quarters_after_conversion_l312_31237

/-- Represents the number of quarters Sara has after converting euros -/
def total_quarters (initial_quarters : ℕ) (euros : ℕ) (exchange_rate : ℚ) : ℕ :=
  initial_quarters + Int.toNat ((euros : ℚ) * exchange_rate * 4).floor

/-- Theorem stating the total number of quarters Sara has after conversion -/
theorem sara_quarters_after_conversion :
  total_quarters 783 250 (118/100) = 1963 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_quarters_after_conversion_l312_31237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_theorem_l312_31258

def dot_product_problem (a b : ℝ × ℝ) : Prop :=
  let norm := λ v : ℝ × ℝ => Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
  norm a = 4 ∧ norm b = 5 ∧ norm (a.1 + b.1, a.2 + b.2) = Real.sqrt 21 →
  a.1 * b.1 + a.2 * b.2 = -10

theorem dot_product_theorem :
  ∀ a b : ℝ × ℝ, dot_product_problem a b := by
  sorry

#check dot_product_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_theorem_l312_31258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_APF_l312_31292

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the right focus F
def F : ℝ × ℝ := (2, 0)

-- Define point A
def A : ℝ × ℝ := (0, 3)

-- Define point P on the hyperbola
def P : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem area_of_triangle_APF :
  hyperbola P.1 P.2 ∧  -- P is on the hyperbola
  (P.1 - F.1) = 0 →    -- PF is perpendicular to x-axis
  (1/2 : ℝ) * |A.1 * P.2 + F.1 * A.2 + P.1 * F.2 - A.2 * P.1 - F.2 * A.1 - P.2 * F.1| = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_APF_l312_31292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_l312_31211

-- Define the constant speed of Car B
noncomputable def speed_B : ℝ := 50

-- Define the initial distance between cars
noncomputable def initial_distance : ℝ := 40

-- Define the time taken for Car A to overtake Car B
noncomputable def time : ℝ := 6

-- Define the distance Car A is ahead after overtaking
noncomputable def ahead_distance : ℝ := 8

-- Define the speed of Car A
noncomputable def speed_A : ℝ := (initial_distance + speed_B * time + ahead_distance) / time

-- Theorem statement
theorem car_speed_problem :
  speed_A = 58 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_l312_31211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digit_count_l312_31287

theorem product_digit_count (n : ℕ) (a b c : ℕ) :
  (10^(n-1) ≤ a ∧ a < 10^n) →
  (10^(n-1) ≤ b ∧ b < 10^n) →
  (10^(n-1) ≤ c ∧ c < 10^n) →
  (∃ k : ℕ, k ∈ ({3*n-2, 3*n-1, 3*n} : Set ℕ) ∧ 10^(k-1) ≤ a*b*c ∧ a*b*c < 10^k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digit_count_l312_31287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentyfourth_digit_of_sum_l312_31291

/-- decimal_sum a b n returns the nth digit after the decimal point of a + b -/
def decimal_sum (a b : ℚ) : ℕ → ℕ :=
  sorry -- Implementation details omitted for brevity

theorem twentyfourth_digit_of_sum :
  decimal_sum (1 / 13) (1 / 8) 24 = 3 := by
  sorry -- Proof details omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentyfourth_digit_of_sum_l312_31291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_x_intercepts_l312_31241

-- Define the curve C
noncomputable def C (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos θ, Real.sin θ)

-- Define points A and B
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, -1)

-- Define a moving point M on curve C
noncomputable def M (θ : ℝ) : ℝ × ℝ := C θ

-- Define the x-intercepts a and b
noncomputable def a (θ : ℝ) : ℝ := (1 - (C θ).1) / (C θ).2
noncomputable def b (θ : ℝ) : ℝ := ((C θ).1 + 1) / (C θ).2

-- Theorem statement
theorem min_sum_x_intercepts :
  ∀ θ : ℝ, M θ ≠ A → M θ ≠ B → |a θ + b θ| ≥ 2 ∧ ∃ θ₀ : ℝ, |a θ₀ + b θ₀| = 2 :=
by
  sorry

#check min_sum_x_intercepts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_x_intercepts_l312_31241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_on_chessboard_l312_31210

/-- A chessboard is represented as an 8x8 grid. -/
def Chessboard : Type := Fin 8 → Fin 8 → Bool

/-- A king's position on the chessboard. -/
structure KingPosition where
  row : Fin 8
  col : Fin 8

/-- Check if two king positions are attacking each other. -/
def are_attacking (k1 k2 : KingPosition) : Bool :=
  (abs (k1.row - k2.row) ≤ 1) && (abs (k1.col - k2.col) ≤ 1)

/-- A valid placement of kings is one where no two kings are attacking each other. -/
def is_valid_placement (kings : List KingPosition) : Prop :=
  ∀ k1 k2, k1 ∈ kings → k2 ∈ kings → k1 ≠ k2 → ¬(are_attacking k1 k2)

/-- The maximum number of non-attacking kings on an 8x8 chessboard is 16. -/
theorem max_kings_on_chessboard :
  ∃ (kings : List KingPosition),
    kings.length = 16 ∧ is_valid_placement kings ∧
    ∀ (other_kings : List KingPosition),
      is_valid_placement other_kings → other_kings.length ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_on_chessboard_l312_31210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l312_31284

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (↑(⌊x * 100 + 0.5⌋) : ℝ) / 100

/-- The sum of 132.478 and 56.925 rounded to the nearest hundredth is 189.40 -/
theorem sum_and_round :
  round_to_hundredth (132.478 + 56.925) = 189.40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l312_31284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_division_l312_31228

/-- The fraction of cookies remaining after three winners take their assumed shares --/
theorem cookie_division (total : ℚ) : 
  let al_share : ℚ := 4 / 9 * total
  let bert_share : ℚ := 3 / 9 * total
  let carl_share : ℚ := 2 / 9 * total

  let al_takes : ℚ := 4 / 9 * total
  let bert_takes : ℚ := 3 / 9 * (total - al_takes)
  let carl_takes : ℚ := 2 / 9 * (total - al_takes - bert_takes)

  let remaining : ℚ := (total - al_takes - bert_takes - carl_takes) / total

  remaining = 230 / 243 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_division_l312_31228
