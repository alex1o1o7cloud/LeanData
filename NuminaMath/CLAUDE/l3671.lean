import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l3671_367108

theorem sum_of_specific_numbers : 3 + 33 + 333 + 33.3 = 402.3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l3671_367108


namespace NUMINAMATH_CALUDE_joan_found_70_seashells_l3671_367101

/-- The number of seashells Sam gave to Joan -/
def seashells_from_sam : ℕ := 27

/-- The total number of seashells Joan has now -/
def total_seashells : ℕ := 97

/-- The number of seashells Joan found on the beach -/
def seashells_found_on_beach : ℕ := total_seashells - seashells_from_sam

theorem joan_found_70_seashells : seashells_found_on_beach = 70 := by
  sorry

end NUMINAMATH_CALUDE_joan_found_70_seashells_l3671_367101


namespace NUMINAMATH_CALUDE_flyers_left_to_hand_out_l3671_367152

theorem flyers_left_to_hand_out 
  (total_flyers : ℕ) 
  (jack_handed_out : ℕ) 
  (rose_handed_out : ℕ) 
  (h1 : total_flyers = 1236)
  (h2 : jack_handed_out = 120)
  (h3 : rose_handed_out = 320) :
  total_flyers - (jack_handed_out + rose_handed_out) = 796 :=
by sorry

end NUMINAMATH_CALUDE_flyers_left_to_hand_out_l3671_367152


namespace NUMINAMATH_CALUDE_fredrickson_chickens_l3671_367196

/-- Given a total number of chickens, calculates the number of chickens that do not lay eggs. -/
def chickens_not_laying_eggs (total : ℕ) : ℕ :=
  let roosters := total / 4
  let hens := total - roosters
  let laying_hens := (hens * 3) / 4
  roosters + (hens - laying_hens)

/-- Theorem stating that for 80 chickens, where 1/4 are roosters and 3/4 of hens lay eggs,
    the number of chickens not laying eggs is 35. -/
theorem fredrickson_chickens :
  chickens_not_laying_eggs 80 = 35 := by
  sorry

#eval chickens_not_laying_eggs 80

end NUMINAMATH_CALUDE_fredrickson_chickens_l3671_367196


namespace NUMINAMATH_CALUDE_toms_hourly_wage_l3671_367174

/-- Tom's hourly wage calculation --/
theorem toms_hourly_wage :
  let item_cost : ℝ := 25.35 + 70.69 + 85.96
  let hours_worked : ℕ := 31
  let savings_rate : ℝ := 0.1
  let hourly_wage : ℝ := item_cost / ((1 - savings_rate) * hours_worked)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |hourly_wage - 6.52| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_toms_hourly_wage_l3671_367174


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_l3671_367111

/-- Conversion from spherical coordinates to rectangular coordinates -/
theorem spherical_to_rectangular (ρ θ φ : ℝ) :
  ρ = 5 ∧ θ = π/4 ∧ φ = π/3 →
  (ρ * Real.sin φ * Real.cos θ = 5 * Real.sqrt 6 / 4) ∧
  (ρ * Real.sin φ * Real.sin θ = 5 * Real.sqrt 6 / 4) ∧
  (ρ * Real.cos φ = 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_l3671_367111


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3671_367109

/-- A sequence of 8 positive real numbers -/
structure Sequence :=
  (terms : Fin 8 → ℝ)
  (positive : ∀ i, terms i > 0)

/-- Predicate for a geometric sequence -/
def is_geometric (s : Sequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ i : Fin 7, s.terms i.succ = q * s.terms i

theorem sufficient_not_necessary_condition (s : Sequence) :
  (s.terms 0 + s.terms 7 < s.terms 3 + s.terms 4 → ¬is_geometric s) ∧
  ∃ s' : Sequence, ¬is_geometric s' ∧ s'.terms 0 + s'.terms 7 ≥ s'.terms 3 + s'.terms 4 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3671_367109


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l3671_367123

theorem mean_proportional_problem (B : ℝ) :
  (56.5 : ℝ) = Real.sqrt (49 * B) → B = 64.9 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l3671_367123


namespace NUMINAMATH_CALUDE_no_equal_roots_for_quadratic_l3671_367173

theorem no_equal_roots_for_quadratic :
  ¬ ∃ k : ℝ, ∃ x : ℝ, x^2 - (k + 1) * x + (k - 3) = 0 ∧
    ∀ y : ℝ, y^2 - (k + 1) * y + (k - 3) = 0 → y = x := by
  sorry

end NUMINAMATH_CALUDE_no_equal_roots_for_quadratic_l3671_367173


namespace NUMINAMATH_CALUDE_opposite_reciprocal_abs_l3671_367142

theorem opposite_reciprocal_abs (x : ℚ) (h : x = -4/3) : 
  (-x = 4/3) ∧ (x⁻¹ = -3/4) ∧ (|x| = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_abs_l3671_367142


namespace NUMINAMATH_CALUDE_problem_solution_l3671_367177

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem problem_solution :
  arithmetic_sequence 2 5 150 = 747 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3671_367177


namespace NUMINAMATH_CALUDE_simplify_expression_l3671_367144

theorem simplify_expression (a b : ℝ) : 
  (15*a + 45*b) + (20*a + 35*b) - (25*a + 55*b) + (30*a - 5*b) = 40*a + 20*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3671_367144


namespace NUMINAMATH_CALUDE_inequality_proof_l3671_367184

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 9*b*c) + b / Real.sqrt (b^2 + 9*c*a) + c / Real.sqrt (c^2 + 9*a*b) ≥ 3 / Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3671_367184


namespace NUMINAMATH_CALUDE_rectangle_area_l3671_367130

theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  length = 3 * width →
  perimeter = 2 * (length + width) →
  perimeter = 160 →
  area = length * width →
  area = 1200 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3671_367130


namespace NUMINAMATH_CALUDE_no_intersection_eq_two_five_l3671_367119

theorem no_intersection_eq_two_five : ¬∃ a : ℝ, 
  ({2, 4, a^3 - 2*a^2 - a + 7} : Set ℝ) ∩ 
  ({1, 5*a - 5, -1/2*a^2 + 3/2*a + 4, a^3 + a^2 + 3*a + 7} : Set ℝ) = 
  ({2, 5} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_no_intersection_eq_two_five_l3671_367119


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l3671_367114

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d : ℝ} : 
  (∀ x y, a*x + b*y + c = 0 ↔ d*x - y = 0) → a/b = -d

/-- The value of m for parallel lines -/
theorem parallel_lines_m_value : 
  (∀ x y, x + 2*y - 1 = 0 ↔ m*x - y = 0) → m = -1/2 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l3671_367114


namespace NUMINAMATH_CALUDE_expected_sides_after_cutting_l3671_367116

/-- The expected number of sides of a randomly picked polygon after cutting -/
def expected_sides (n : ℕ) : ℚ :=
  (n + 7200 : ℚ) / 3601

/-- Theorem stating the expected number of sides after cutting an n-sided polygon for 3600 seconds -/
theorem expected_sides_after_cutting (n : ℕ) :
  let initial_sides := n
  let num_cuts := 3600
  let total_sides := initial_sides + 2 * num_cuts
  let num_polygons := num_cuts + 1
  expected_sides n = total_sides / num_polygons :=
by
  sorry

#eval expected_sides 3  -- For a triangle
#eval expected_sides 4  -- For a quadrilateral

end NUMINAMATH_CALUDE_expected_sides_after_cutting_l3671_367116


namespace NUMINAMATH_CALUDE_sum_of_roots_is_negative_one_l3671_367128

-- Define the ∇ operation
def nabla (a b : ℝ) : ℝ := a * b - b * a^2

-- Theorem statement
theorem sum_of_roots_is_negative_one :
  let f : ℝ → ℝ := λ x => (nabla 2 x) - 8 - (nabla x 6)
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0) ∧ x₁ + x₂ = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_negative_one_l3671_367128


namespace NUMINAMATH_CALUDE_flower_garden_width_l3671_367188

/-- The width of a rectangular flower garden with given area and length -/
theorem flower_garden_width (area : ℝ) (length : ℝ) (h_area : area = 48.6) (h_length : length = 5.4) :
  area / length = 9 := by
  sorry

end NUMINAMATH_CALUDE_flower_garden_width_l3671_367188


namespace NUMINAMATH_CALUDE_equation_solution_l3671_367178

theorem equation_solution (x : ℝ) :
  (x / 5) / 3 = 9 / (x / 3) → x = 15 * Real.sqrt 1.8 ∨ x = -15 * Real.sqrt 1.8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3671_367178


namespace NUMINAMATH_CALUDE_star_sum_minus_emilio_sum_l3671_367110

def star_list : List Nat := List.range 50

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem star_sum_minus_emilio_sum : 
  star_list.sum - emilio_list.sum = 105 := by
  sorry

end NUMINAMATH_CALUDE_star_sum_minus_emilio_sum_l3671_367110


namespace NUMINAMATH_CALUDE_adults_who_ate_correct_l3671_367168

/-- Represents the number of adults who had their meal -/
def adults_who_ate : ℕ := 21

/-- The total number of adults -/
def total_adults : ℕ := 55

/-- The total number of children -/
def total_children : ℕ := 70

/-- The number of adults the meal can fully cater for -/
def meal_capacity_adults : ℕ := 70

/-- The number of children the meal can fully cater for -/
def meal_capacity_children : ℕ := 90

/-- The number of children that can be catered with the remaining food after some adults eat -/
def remaining_children_capacity : ℕ := 63

theorem adults_who_ate_correct :
  adults_who_ate * meal_capacity_children / meal_capacity_adults +
  remaining_children_capacity = meal_capacity_children :=
by sorry

end NUMINAMATH_CALUDE_adults_who_ate_correct_l3671_367168


namespace NUMINAMATH_CALUDE_lcm_of_18_50_120_l3671_367181

theorem lcm_of_18_50_120 : Nat.lcm (Nat.lcm 18 50) 120 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_18_50_120_l3671_367181


namespace NUMINAMATH_CALUDE_curve_translation_l3671_367189

-- Define the original curve
def original_curve (x y : ℝ) : Prop :=
  y * Real.cos x + 2 * y - 1 = 0

-- Define the translated curve
def translated_curve (x y : ℝ) : Prop :=
  (y - 1) * Real.sin x + 2 * y - 3 = 0

-- State the theorem
theorem curve_translation :
  ∀ (x y : ℝ),
    original_curve (x - π/2) (y + 1) ↔ translated_curve x y :=
by sorry

end NUMINAMATH_CALUDE_curve_translation_l3671_367189


namespace NUMINAMATH_CALUDE_same_type_as_3a2b_l3671_367145

/-- Two terms are of the same type if they have the same variables with the same exponents. -/
def same_type (t1 t2 : ℕ → ℕ → ℕ) : Prop :=
  ∀ a b, t1 a b ≠ 0 ∧ t2 a b ≠ 0 → 
    (t1 a b).factors.toFinset = (t2 a b).factors.toFinset

/-- The term $3a^2b$ -/
def term1 (a b : ℕ) : ℕ := 3 * a^2 * b

/-- The term $2ab^2$ -/
def term2 (a b : ℕ) : ℕ := 2 * a * b^2

/-- The term $-a^2b$ -/
def term3 (a b : ℕ) : ℕ := a^2 * b

/-- The term $-2ab$ -/
def term4 (a b : ℕ) : ℕ := 2 * a * b

/-- The term $5a^2$ -/
def term5 (a b : ℕ) : ℕ := 5 * a^2

theorem same_type_as_3a2b :
  same_type term1 term3 ∧
  ¬ same_type term1 term2 ∧
  ¬ same_type term1 term4 ∧
  ¬ same_type term1 term5 :=
sorry

end NUMINAMATH_CALUDE_same_type_as_3a2b_l3671_367145


namespace NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l3671_367180

-- First expression
theorem simplify_first_expression (a b : ℝ) :
  6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1/2) * a * b) = -a * b := by
  sorry

-- Second expression
theorem simplify_second_expression (t : ℝ) :
  -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l3671_367180


namespace NUMINAMATH_CALUDE_beyonce_songs_count_l3671_367171

/-- The number of songs Beyonce has released in total -/
def total_songs : ℕ :=
  let singles := 12
  let albums := 4
  let songs_first_cd := 18
  let songs_second_cd := 14
  let songs_per_album := songs_first_cd + songs_second_cd
  singles + albums * songs_per_album

/-- Theorem stating that Beyonce has released 140 songs in total -/
theorem beyonce_songs_count : total_songs = 140 := by
  sorry

end NUMINAMATH_CALUDE_beyonce_songs_count_l3671_367171


namespace NUMINAMATH_CALUDE_plane_equation_correct_l3671_367127

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- The plane equation we want to prove -/
def targetEquation : PlaneEquation :=
  { A := 15, B := 7, C := 17, D := -26 }

/-- The three given points -/
def p1 : Point3D := { x := 2, y := -3, z := 1 }
def p2 : Point3D := { x := -1, y := 1, z := 2 }
def p3 : Point3D := { x := 4, y := 0, z := -2 }

theorem plane_equation_correct :
  (pointOnPlane p1 targetEquation) ∧
  (pointOnPlane p2 targetEquation) ∧
  (pointOnPlane p3 targetEquation) ∧
  (targetEquation.A > 0) ∧
  (Nat.gcd (Nat.gcd (Int.natAbs targetEquation.A) (Int.natAbs targetEquation.B))
           (Nat.gcd (Int.natAbs targetEquation.C) (Int.natAbs targetEquation.D)) = 1) :=
by sorry


end NUMINAMATH_CALUDE_plane_equation_correct_l3671_367127


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3671_367192

/-- An ellipse with axes parallel to the coordinate axes, tangent to the x-axis at (3, 0) and tangent to the y-axis at (0, 2) -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ
  tangent_x : center.1 - semi_major_axis = 3
  tangent_y : center.2 - semi_minor_axis = 2

/-- The distance between the foci of the ellipse is 2√5 -/
theorem ellipse_foci_distance (e : Ellipse) : 
  Real.sqrt (4 * (e.semi_major_axis^2 - e.semi_minor_axis^2)) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3671_367192


namespace NUMINAMATH_CALUDE_largest_quantity_l3671_367133

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2012 / 2011 + 2010 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : C > A ∧ C > B := by sorry

end NUMINAMATH_CALUDE_largest_quantity_l3671_367133


namespace NUMINAMATH_CALUDE_circle_tangency_l3671_367103

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circle_tangency (m : ℝ) : 
  externally_tangent (0, 0) (2, 4) (Real.sqrt 5) (Real.sqrt (20 + m)) → m = -15 := by
  sorry

#check circle_tangency

end NUMINAMATH_CALUDE_circle_tangency_l3671_367103


namespace NUMINAMATH_CALUDE_roots_of_unity_quadratic_count_l3671_367170

/-- A complex number z is a root of unity if there exists a positive integer n such that z^n = 1 -/
def is_root_of_unity (z : ℂ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ z^n = 1

/-- The quadratic equation z^2 + az - 1 = 0 for some integer a -/
def quadratic_equation (z : ℂ) : Prop :=
  ∃ (a : ℤ), z^2 + a*z - 1 = 0

/-- The number of roots of unity that are also roots of the quadratic equation is exactly two -/
theorem roots_of_unity_quadratic_count :
  ∃! (S : Finset ℂ), (∀ z ∈ S, is_root_of_unity z ∧ quadratic_equation z) ∧ S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_roots_of_unity_quadratic_count_l3671_367170


namespace NUMINAMATH_CALUDE_min_price_theorem_l3671_367136

/-- Represents the manufacturing scenario with two components -/
structure ManufacturingScenario where
  prod_cost_A : ℝ  -- Production cost for component A
  ship_cost_A : ℝ  -- Shipping cost for component A
  prod_cost_B : ℝ  -- Production cost for component B
  ship_cost_B : ℝ  -- Shipping cost for component B
  fixed_costs : ℝ  -- Fixed costs per month
  units_A : ℕ      -- Number of units of component A produced and sold
  units_B : ℕ      -- Number of units of component B produced and sold

/-- Calculates the total cost for the given manufacturing scenario -/
def total_cost (s : ManufacturingScenario) : ℝ :=
  s.fixed_costs +
  s.units_A * (s.prod_cost_A + s.ship_cost_A) +
  s.units_B * (s.prod_cost_B + s.ship_cost_B)

/-- Theorem: The minimum price per unit that ensures total revenue is at least equal to total costs is $103 -/
theorem min_price_theorem (s : ManufacturingScenario)
  (h1 : s.prod_cost_A = 80)
  (h2 : s.ship_cost_A = 2)
  (h3 : s.prod_cost_B = 60)
  (h4 : s.ship_cost_B = 3)
  (h5 : s.fixed_costs = 16200)
  (h6 : s.units_A = 200)
  (h7 : s.units_B = 300) :
  ∃ (P : ℝ), P ≥ 103 ∧ P * (s.units_A + s.units_B) ≥ total_cost s ∧
  ∀ (Q : ℝ), Q * (s.units_A + s.units_B) ≥ total_cost s → Q ≥ P :=
sorry


end NUMINAMATH_CALUDE_min_price_theorem_l3671_367136


namespace NUMINAMATH_CALUDE_day_50_of_previous_year_is_thursday_l3671_367122

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year -/
structure Year where
  number : ℕ

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek :=
  sorry

/-- Returns the number of days in a year -/
def daysInYear (y : Year) : ℕ :=
  sorry

theorem day_50_of_previous_year_is_thursday
  (N : Year)
  (h1 : dayOfWeek N 250 = DayOfWeek.Friday)
  (h2 : dayOfWeek (Year.mk (N.number + 1)) 150 = DayOfWeek.Friday) :
  dayOfWeek (Year.mk (N.number - 1)) 50 = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_day_50_of_previous_year_is_thursday_l3671_367122


namespace NUMINAMATH_CALUDE_correct_batteries_in_toys_l3671_367138

/-- The number of batteries Tom used in his toys -/
def batteries_in_toys : ℕ := 15

/-- The number of batteries Tom used in his flashlights -/
def batteries_in_flashlights : ℕ := 2

/-- Theorem stating that the number of batteries in toys is correct -/
theorem correct_batteries_in_toys :
  batteries_in_toys = batteries_in_flashlights + 13 :=
by sorry

end NUMINAMATH_CALUDE_correct_batteries_in_toys_l3671_367138


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3671_367137

theorem triangle_perimeter (a b c : ℝ) : 
  (a - 2)^2 + |b - 4| = 0 → 
  c > 0 →
  c < a + b →
  c > |a - b| →
  ∃ (n : ℕ), c = 2 * n →
  a + b + c = 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3671_367137


namespace NUMINAMATH_CALUDE_binary_1100_is_12_l3671_367175

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + (if bit then 2^i else 0)) 0

theorem binary_1100_is_12 :
  binary_to_decimal [true, true, false, false] = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100_is_12_l3671_367175


namespace NUMINAMATH_CALUDE_quadratic_real_root_l3671_367176

theorem quadratic_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l3671_367176


namespace NUMINAMATH_CALUDE_inquisitive_tourist_ratio_l3671_367135

/-- Represents a tour group with a number of people -/
structure TourGroup where
  people : ℕ

/-- Represents a day of tours -/
structure TourDay where
  groups : List TourGroup
  usualQuestionsPerTourist : ℕ
  totalQuestionsAnswered : ℕ
  inquisitiveTouristGroup : ℕ  -- Index of the group with the inquisitive tourist

def calculateRatio (day : TourDay) : ℚ :=
  let regularQuestions := day.groups.enum.foldl
    (fun acc (i, group) =>
      if i = day.inquisitiveTouristGroup
      then acc + (group.people - 1) * day.usualQuestionsPerTourist
      else acc + group.people * day.usualQuestionsPerTourist)
    0
  let inquisitiveQuestions := day.totalQuestionsAnswered - regularQuestions
  inquisitiveQuestions / day.usualQuestionsPerTourist

theorem inquisitive_tourist_ratio (day : TourDay)
  (h1 : day.groups = [⟨6⟩, ⟨11⟩, ⟨8⟩, ⟨7⟩])
  (h2 : day.usualQuestionsPerTourist = 2)
  (h3 : day.totalQuestionsAnswered = 68)
  (h4 : day.inquisitiveTouristGroup = 2)  -- 0-based index for the third group
  : calculateRatio day = 3 := by
  sorry

#eval calculateRatio {
  groups := [⟨6⟩, ⟨11⟩, ⟨8⟩, ⟨7⟩],
  usualQuestionsPerTourist := 2,
  totalQuestionsAnswered := 68,
  inquisitiveTouristGroup := 2
}

end NUMINAMATH_CALUDE_inquisitive_tourist_ratio_l3671_367135


namespace NUMINAMATH_CALUDE_reverse_product_inequality_l3671_367118

/-- Reverses the digits and decimal point of a positive real number with finitely many decimal places -/
noncomputable def reverse (x : ℝ) : ℝ := sorry

/-- Predicate to check if a real number has finitely many decimal places -/
def has_finite_decimals (x : ℝ) : Prop := sorry

/-- The main theorem to be proved -/
theorem reverse_product_inequality {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (hfx : has_finite_decimals x) (hfy : has_finite_decimals y) : 
  reverse (x * y) ≤ 10 * reverse x * reverse y := by sorry

end NUMINAMATH_CALUDE_reverse_product_inequality_l3671_367118


namespace NUMINAMATH_CALUDE_arrangement_count_is_correct_l3671_367153

/-- The number of ways to arrange 4 volunteers and 1 elder in a row, with the elder in the middle -/
def arrangementCount : ℕ := 24

/-- The number of volunteers -/
def numVolunteers : ℕ := 4

/-- The number of elders -/
def numElders : ℕ := 1

/-- The total number of people -/
def totalPeople : ℕ := numVolunteers + numElders

theorem arrangement_count_is_correct :
  arrangementCount = Nat.factorial numVolunteers := by
  sorry


end NUMINAMATH_CALUDE_arrangement_count_is_correct_l3671_367153


namespace NUMINAMATH_CALUDE_square_cut_into_three_rectangles_l3671_367151

theorem square_cut_into_three_rectangles (square_side : ℝ) (cut_length : ℝ) : 
  square_side = 36 →
  ∃ (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℝ),
    -- The three rectangles have equal areas
    rect1_width * rect1_height = rect2_width * rect2_height ∧
    rect2_width * rect2_height = rect3_width * rect3_height ∧
    -- The rectangles fit within the square
    rect1_width + rect2_width ≤ square_side ∧
    rect1_height ≤ square_side ∧
    rect2_height ≤ square_side ∧
    rect3_width ≤ square_side ∧
    rect3_height ≤ square_side ∧
    -- The rectangles have common boundaries
    (rect1_width = rect2_width ∨ rect1_height = rect2_height) ∧
    (rect2_width = rect3_width ∨ rect2_height = rect3_height) ∧
    (rect1_width = rect3_width ∨ rect1_height = rect3_height) →
  cut_length = 60 :=
by sorry

end NUMINAMATH_CALUDE_square_cut_into_three_rectangles_l3671_367151


namespace NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_l3671_367140

theorem x_fourth_plus_y_fourth (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x * y = 48) : 
  x^4 + y^4 = 5392 := by
sorry

end NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_l3671_367140


namespace NUMINAMATH_CALUDE_combination_5_choose_3_l3671_367159

/-- The number of combinations of n things taken k at a time -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Proof that C(5,3) equals 10 -/
theorem combination_5_choose_3 : combination 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_combination_5_choose_3_l3671_367159


namespace NUMINAMATH_CALUDE_incorrect_statement_l3671_367182

theorem incorrect_statement (P Q : Prop) (h1 : P ↔ (2 + 2 = 5)) (h2 : Q ↔ (3 > 2)) : 
  ¬((¬(P ∧ Q)) ∧ (¬¬P)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_statement_l3671_367182


namespace NUMINAMATH_CALUDE_john_total_pay_this_year_l3671_367194

/-- John's annual bonus calculation -/
def johnBonus (baseSalaryLastYear : ℝ) (firstBonusLastYear : ℝ) (baseSalaryThisYear : ℝ) 
              (bonusGrowthRate : ℝ) (projectBonus : ℝ) (projectsCompleted : ℕ) : ℝ :=
  let firstBonusThisYear := firstBonusLastYear * (1 + bonusGrowthRate)
  let secondBonus := projectBonus * projectsCompleted
  baseSalaryThisYear + firstBonusThisYear + secondBonus

theorem john_total_pay_this_year :
  johnBonus 100000 10000 200000 0.05 2000 8 = 226500 := by
  sorry

end NUMINAMATH_CALUDE_john_total_pay_this_year_l3671_367194


namespace NUMINAMATH_CALUDE_solve_bag_problem_l3671_367169

def bag_problem (total_balls : ℕ) (prob_two_red : ℚ) (red_balls : ℕ) : Prop :=
  total_balls = 10 ∧
  prob_two_red = 1 / 15 ∧
  (red_balls : ℚ) / total_balls * (red_balls - 1) / (total_balls - 1) = prob_two_red

theorem solve_bag_problem :
  ∃ (red_balls : ℕ), bag_problem 10 (1 / 15) red_balls ∧ red_balls = 3 :=
sorry

end NUMINAMATH_CALUDE_solve_bag_problem_l3671_367169


namespace NUMINAMATH_CALUDE_product_divisible_by_14_l3671_367143

theorem product_divisible_by_14 (a b c d : ℤ) (h : 7*a + 8*b = 14*c + 28*d) : 
  14 ∣ (a * b) := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_14_l3671_367143


namespace NUMINAMATH_CALUDE_remainder_sum_reverse_order_l3671_367167

theorem remainder_sum_reverse_order (n : ℕ) : 
  n % 12 = 56 → n % 34 = 78 → (n % 34) % 12 + n % 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_reverse_order_l3671_367167


namespace NUMINAMATH_CALUDE_three_cakes_cooking_time_l3671_367193

/-- Represents the cooking process for cakes -/
structure CookingProcess where
  pot_capacity : ℕ
  cooking_time_per_cake : ℕ
  num_cakes : ℕ

/-- Calculates the minimum time needed to cook all cakes -/
def min_cooking_time (process : CookingProcess) : ℕ :=
  sorry

/-- Theorem stating the minimum time to cook 3 cakes -/
theorem three_cakes_cooking_time :
  ∀ (process : CookingProcess),
    process.pot_capacity = 2 →
    process.cooking_time_per_cake = 5 →
    process.num_cakes = 3 →
    min_cooking_time process = 15 :=
by sorry

end NUMINAMATH_CALUDE_three_cakes_cooking_time_l3671_367193


namespace NUMINAMATH_CALUDE_g_15_equals_281_l3671_367199

/-- The function g(n) = n^2 + n + 41 -/
def g (n : ℕ) : ℕ := n^2 + n + 41

/-- Theorem: g(15) = 281 -/
theorem g_15_equals_281 : g 15 = 281 := by
  sorry

end NUMINAMATH_CALUDE_g_15_equals_281_l3671_367199


namespace NUMINAMATH_CALUDE_video_streaming_cost_theorem_l3671_367163

/-- The total cost per person for a video streaming service after one year -/
def video_streaming_cost (subscription_cost : ℚ) (num_people : ℕ) (connection_fee : ℚ) (tax_rate : ℚ) : ℚ :=
  let monthly_subscription_per_person := subscription_cost / num_people
  let monthly_cost_before_tax := monthly_subscription_per_person + connection_fee
  let monthly_tax := monthly_cost_before_tax * tax_rate
  let monthly_total := monthly_cost_before_tax + monthly_tax
  12 * monthly_total

/-- Theorem stating the total cost per person for a specific video streaming service after one year -/
theorem video_streaming_cost_theorem :
  video_streaming_cost 14 4 2 (1/10) = 726/10 := by
  sorry

#eval video_streaming_cost 14 4 2 (1/10)

end NUMINAMATH_CALUDE_video_streaming_cost_theorem_l3671_367163


namespace NUMINAMATH_CALUDE_inner_polygon_smaller_perimeter_l3671_367187

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool
  
/-- Calculate the perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : Real :=
  sorry

/-- Check if one polygon is contained within another -/
def is_contained_in (inner outer : ConvexPolygon) : Prop :=
  sorry

/-- Theorem: The perimeter of an inner convex polygon is smaller than that of the outer convex polygon -/
theorem inner_polygon_smaller_perimeter
  (inner outer : ConvexPolygon)
  (h_inner_convex : inner.is_convex = true)
  (h_outer_convex : outer.is_convex = true)
  (h_contained : is_contained_in inner outer) :
  perimeter inner < perimeter outer :=
sorry

end NUMINAMATH_CALUDE_inner_polygon_smaller_perimeter_l3671_367187


namespace NUMINAMATH_CALUDE_loan_b_more_cost_effective_l3671_367139

/-- Calculates the total repayable amount for a loan -/
def totalRepayable (principal : ℝ) (interestRate : ℝ) (years : ℝ) : ℝ :=
  principal + principal * interestRate * years

/-- Represents the loan options available to Mike -/
structure LoanOption where
  principal : ℝ
  interestRate : ℝ
  years : ℝ

/-- Theorem stating that Loan B is more cost-effective than Loan A -/
theorem loan_b_more_cost_effective (carPrice savings : ℝ) (loanA loanB : LoanOption) :
  carPrice = 35000 ∧
  savings = 5000 ∧
  loanA.principal = 25000 ∧
  loanA.interestRate = 0.07 ∧
  loanA.years = 5 ∧
  loanB.principal = 20000 ∧
  loanB.interestRate = 0.05 ∧
  loanB.years = 4 →
  totalRepayable loanB.principal loanB.interestRate loanB.years <
  totalRepayable loanA.principal loanA.interestRate loanA.years :=
by sorry

end NUMINAMATH_CALUDE_loan_b_more_cost_effective_l3671_367139


namespace NUMINAMATH_CALUDE_yoongi_has_fewest_apples_l3671_367148

def jungkook_apples : ℕ := 6 * 3
def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5

theorem yoongi_has_fewest_apples :
  yoongi_apples ≤ jungkook_apples ∧ yoongi_apples ≤ yuna_apples := by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_fewest_apples_l3671_367148


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l3671_367132

def A : Set ℝ := {-1, 2, 3}
def B (a : ℝ) : Set ℝ := {a + 1, a^2 + 3}

theorem intersection_implies_a_values :
  ∀ a : ℝ, (A ∩ B a = {3}) → (a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l3671_367132


namespace NUMINAMATH_CALUDE_means_of_reciprocals_of_first_four_primes_l3671_367165

def first_four_primes : List Nat := [2, 3, 5, 7]

def reciprocals (lst : List Nat) : List Rat :=
  lst.map (λ x => 1 / x)

def arithmetic_mean (lst : List Rat) : Rat :=
  lst.sum / lst.length

def harmonic_mean (lst : List Rat) : Rat :=
  lst.length / (lst.map (λ x => 1 / x)).sum

theorem means_of_reciprocals_of_first_four_primes :
  let recip := reciprocals first_four_primes
  arithmetic_mean recip = 247 / 840 ∧
  harmonic_mean recip = 4 / 17 := by
  sorry

#eval arithmetic_mean (reciprocals first_four_primes)
#eval harmonic_mean (reciprocals first_four_primes)

end NUMINAMATH_CALUDE_means_of_reciprocals_of_first_four_primes_l3671_367165


namespace NUMINAMATH_CALUDE_shoes_to_belts_ratio_l3671_367155

def number_of_hats : ℕ := 5
def number_of_shoes : ℕ := 14
def belt_hat_difference : ℕ := 2

def number_of_belts : ℕ := number_of_hats + belt_hat_difference

theorem shoes_to_belts_ratio :
  (number_of_shoes : ℚ) / (number_of_belts : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_shoes_to_belts_ratio_l3671_367155


namespace NUMINAMATH_CALUDE_inequality_proof_l3671_367162

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄)
  (h4 : x₂ + x₃ + x₄ ≥ x₁) :
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3671_367162


namespace NUMINAMATH_CALUDE_polynomial_fits_data_l3671_367126

def f (x : ℝ) : ℝ := x^3 + 2*x^2 + x + 1

theorem polynomial_fits_data : 
  f 1 = 5 ∧ f 2 = 15 ∧ f 3 = 35 ∧ f 4 = 69 ∧ f 5 = 119 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_fits_data_l3671_367126


namespace NUMINAMATH_CALUDE_lesser_number_problem_l3671_367179

theorem lesser_number_problem (x y : ℝ) 
  (sum_eq : x + y = 60) 
  (diff_eq : 4 * y - x = 10) : 
  y = 14 := by sorry

end NUMINAMATH_CALUDE_lesser_number_problem_l3671_367179


namespace NUMINAMATH_CALUDE_ed_hotel_stay_l3671_367125

def hotel_problem (night_rate : ℚ) (morning_rate : ℚ) (initial_money : ℚ) (night_hours : ℚ) (money_left : ℚ) : ℚ :=
  let total_spent := initial_money - money_left
  let night_cost := night_rate * night_hours
  let morning_spent := total_spent - night_cost
  morning_spent / morning_rate

theorem ed_hotel_stay :
  hotel_problem 1.5 2 80 6 63 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ed_hotel_stay_l3671_367125


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_intersection_equals_B_iff_l3671_367131

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a+3}

-- Statement I
theorem intersection_when_a_is_one :
  (Set.univ \ A) ∩ (B 1) = {x | 3 < x ∧ x < 4} := by sorry

-- Statement II
theorem intersection_equals_B_iff (a : ℝ) :
  (Set.univ \ A) ∩ (B a) = B a ↔ a ≤ -2 ∨ a ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_intersection_equals_B_iff_l3671_367131


namespace NUMINAMATH_CALUDE_will_initial_candy_l3671_367141

/-- The amount of candy Will gave to Haley -/
def candy_given : ℕ := 6

/-- The amount of candy Will had left after giving some to Haley -/
def candy_left : ℕ := 9

/-- The initial amount of candy Will had -/
def initial_candy : ℕ := candy_given + candy_left

theorem will_initial_candy : initial_candy = 15 := by
  sorry

end NUMINAMATH_CALUDE_will_initial_candy_l3671_367141


namespace NUMINAMATH_CALUDE_radio_show_duration_is_three_hours_l3671_367124

/-- Calculates the total duration of a radio show in hours -/
def radio_show_duration (
  talking_segment_duration : ℕ)
  (ad_break_duration : ℕ)
  (num_talking_segments : ℕ)
  (num_ad_breaks : ℕ)
  (song_duration : ℕ) : ℚ :=
  let total_minutes : ℕ := 
    talking_segment_duration * num_talking_segments +
    ad_break_duration * num_ad_breaks +
    song_duration
  (total_minutes : ℚ) / 60

/-- Proves that given the specified conditions, the radio show duration is 3 hours -/
theorem radio_show_duration_is_three_hours :
  radio_show_duration 10 5 3 5 125 = 3 := by
  sorry

end NUMINAMATH_CALUDE_radio_show_duration_is_three_hours_l3671_367124


namespace NUMINAMATH_CALUDE_river_current_speed_l3671_367156

/-- Proves that the speed of the river's current is half the swimmer's speed in still water --/
theorem river_current_speed (x y : ℝ) : 
  x > 0 → -- swimmer's speed in still water is positive
  x = 10 → -- swimmer's speed in still water is 10 km/h
  (x + y) > 0 → -- downstream speed is positive
  (x - y) > 0 → -- upstream speed is positive
  (1 / (x - y)) = (3 * (1 / (x + y))) → -- upstream time is 3 times downstream time
  y = x / 2 := by
  sorry

end NUMINAMATH_CALUDE_river_current_speed_l3671_367156


namespace NUMINAMATH_CALUDE_max_value_theorem_l3671_367198

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 3) :
  3 * a * b * Real.sqrt 2 + 9 * b * c ≤ 3 * Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3671_367198


namespace NUMINAMATH_CALUDE_sequence_inequality_l3671_367146

def a : ℕ → ℚ
  | 0 => 1
  | (n + 1) => a n - (a n)^2 / 2019

theorem sequence_inequality : a 2019 < 1/2 ∧ 1/2 < a 2018 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3671_367146


namespace NUMINAMATH_CALUDE_probability_not_paying_cash_l3671_367121

theorem probability_not_paying_cash (p_only_cash p_both : ℝ) 
  (h1 : p_only_cash = 0.45)
  (h2 : p_both = 0.15) : 
  1 - (p_only_cash + p_both) = 0.4 := by
sorry

end NUMINAMATH_CALUDE_probability_not_paying_cash_l3671_367121


namespace NUMINAMATH_CALUDE_parallelogram_area_is_3sqrt14_l3671_367149

-- Define the complex equations
def equation1 (z : ℂ) : Prop := z^2 = 9 + 9 * Complex.I * Real.sqrt 7
def equation2 (z : ℂ) : Prop := z^2 = 3 + 3 * Complex.I * Real.sqrt 2

-- Define the solutions
def solutions : Set ℂ := {z : ℂ | equation1 z ∨ equation2 z}

-- Define the parallelogram area function
noncomputable def parallelogramArea (vertices : Set ℂ) : ℝ :=
  sorry -- Actual implementation would go here

-- Theorem statement
theorem parallelogram_area_is_3sqrt14 :
  parallelogramArea solutions = 3 * Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_is_3sqrt14_l3671_367149


namespace NUMINAMATH_CALUDE_correct_small_glasses_l3671_367105

/-- Calculates the number of small drinking glasses given the following conditions:
  * 50 jelly beans fill a large glass
  * 25 jelly beans fill a small glass
  * There are 5 large glasses
  * A total of 325 jelly beans are used
-/
def number_of_small_glasses (large_glass_beans : ℕ) (small_glass_beans : ℕ) 
  (num_large_glasses : ℕ) (total_beans : ℕ) : ℕ :=
  (total_beans - large_glass_beans * num_large_glasses) / small_glass_beans

theorem correct_small_glasses : 
  number_of_small_glasses 50 25 5 325 = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_small_glasses_l3671_367105


namespace NUMINAMATH_CALUDE_candy_distribution_l3671_367186

theorem candy_distribution (total_candy : ℕ) (candy_per_bag : ℕ) (h1 : total_candy = 648) (h2 : candy_per_bag = 81) :
  total_candy / candy_per_bag = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3671_367186


namespace NUMINAMATH_CALUDE_valid_queue_arrangements_correct_l3671_367191

/-- Represents the number of valid queue arrangements for a concert ticket purchase scenario. -/
def validQueueArrangements (m n : ℕ) : ℚ :=
  if n ≥ m then
    (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1))
  else 0

/-- Theorem stating the correctness of the validQueueArrangements function. -/
theorem valid_queue_arrangements_correct (m n : ℕ) (h : n ≥ m) :
  validQueueArrangements m n = (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_valid_queue_arrangements_correct_l3671_367191


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_l3671_367115

theorem sally_pokemon_cards (initial : ℕ) (new : ℕ) (lost : ℕ) : 
  initial = 27 → new = 41 → lost = 20 → initial + new - lost = 48 := by
  sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_l3671_367115


namespace NUMINAMATH_CALUDE_magician_trick_exists_strategy_l3671_367112

/-- Represents a card placement strategy for the magician's trick -/
structure CardPlacementStrategy (n : ℕ) :=
  (place_cards : Fin n → Fin n)
  (deduce_card1 : Fin n → Fin n → Fin n)
  (deduce_card2 : Fin n → Fin n → Fin n)

/-- The main theorem stating that a successful strategy exists for all n ≥ 3 -/
theorem magician_trick_exists_strategy (n : ℕ) (h : n ≥ 3) :
  ∃ (strategy : CardPlacementStrategy n),
    ∀ (card1_pos card2_pos : Fin n),
      card1_pos ≠ card2_pos →
      ∀ (magician_reveal spectator_reveal : Fin n),
        magician_reveal ≠ spectator_reveal →
        strategy.deduce_card1 magician_reveal spectator_reveal = card1_pos ∧
        strategy.deduce_card2 magician_reveal spectator_reveal = card2_pos :=
sorry

end NUMINAMATH_CALUDE_magician_trick_exists_strategy_l3671_367112


namespace NUMINAMATH_CALUDE_find_x2_l3671_367120

theorem find_x2 (x₁ x₂ x₃ : ℝ) 
  (h_order : x₁ < x₂ ∧ x₂ < x₃)
  (h_sum1 : x₁ + x₂ = 14)
  (h_sum2 : x₁ + x₃ = 17)
  (h_sum3 : x₂ + x₃ = 33) : 
  x₂ = 15 := by
sorry

end NUMINAMATH_CALUDE_find_x2_l3671_367120


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3671_367100

theorem inequality_system_solution :
  ∃ (x y : ℝ), 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧ 
    (2 * x - 4 * y ≤ -3) ∧
    (x = -1/3) ∧ 
    (y = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3671_367100


namespace NUMINAMATH_CALUDE_combined_paint_cost_l3671_367134

/-- Represents the dimensions and painting cost of a rectangular floor -/
structure Floor :=
  (length : ℝ)
  (breadth : ℝ)
  (paint_rate : ℝ)

/-- Calculates the area of a rectangular floor -/
def floor_area (f : Floor) : ℝ := f.length * f.breadth

/-- Calculates the cost to paint a floor -/
def paint_cost (f : Floor) : ℝ := floor_area f * f.paint_rate

/-- Represents the two-story building -/
structure Building :=
  (first_floor : Floor)
  (second_floor : Floor)

/-- The main theorem to prove -/
theorem combined_paint_cost (b : Building) : ℝ :=
  let f1 := b.first_floor
  let f2 := b.second_floor
  have h1 : f1.length = 3 * f1.breadth := by sorry
  have h2 : paint_cost f1 = 484 := by sorry
  have h3 : f1.paint_rate = 3 := by sorry
  have h4 : f2.length = 0.8 * f1.length := by sorry
  have h5 : f2.breadth = 1.3 * f1.breadth := by sorry
  have h6 : f2.paint_rate = 5 := by sorry
  have h7 : paint_cost f1 + paint_cost f2 = 1320.8 := by sorry
  1320.8

#check combined_paint_cost

end NUMINAMATH_CALUDE_combined_paint_cost_l3671_367134


namespace NUMINAMATH_CALUDE_increasing_interval_implies_a_l3671_367185

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a|

-- State the theorem
theorem increasing_interval_implies_a (a : ℝ) :
  (∀ x ≥ 3, ∀ y > x, f a y > f a x) ∧
  (∀ x < 3, ∃ y > x, f a y ≤ f a x) →
  a = -6 := by
  sorry

end NUMINAMATH_CALUDE_increasing_interval_implies_a_l3671_367185


namespace NUMINAMATH_CALUDE_macaroon_ratio_is_two_to_one_l3671_367190

/-- Represents the numbers of macaroons in different states --/
structure MacaroonCounts where
  initial_red : ℕ
  initial_green : ℕ
  green_eaten : ℕ
  total_remaining : ℕ

/-- Calculates the ratio of red macaroons eaten to green macaroons eaten --/
def macaroon_ratio (m : MacaroonCounts) : ℚ :=
  let red_eaten := m.initial_red - (m.total_remaining - (m.initial_green - m.green_eaten))
  red_eaten / m.green_eaten

/-- Theorem stating that given the specific conditions, the ratio is 2:1 --/
theorem macaroon_ratio_is_two_to_one (m : MacaroonCounts) 
  (h1 : m.initial_red = 50)
  (h2 : m.initial_green = 40)
  (h3 : m.green_eaten = 15)
  (h4 : m.total_remaining = 45) :
  macaroon_ratio m = 2 := by
  sorry

end NUMINAMATH_CALUDE_macaroon_ratio_is_two_to_one_l3671_367190


namespace NUMINAMATH_CALUDE_equation_solution_l3671_367166

theorem equation_solution : ∃! y : ℚ, 4 * (5 * y + 3) - 3 = -3 * (2 - 8 * y) ∧ y = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3671_367166


namespace NUMINAMATH_CALUDE_smallest_integer_fourth_root_l3671_367164

theorem smallest_integer_fourth_root (p : ℕ) (q : ℕ) (s : ℝ) : 
  (0 < q) → 
  (0 < s) → 
  (s < 1 / 2000) → 
  (p^(1/4 : ℝ) = q + s) → 
  (∀ (p' : ℕ) (q' : ℕ) (s' : ℝ), 
    0 < q' → 0 < s' → s' < 1 / 2000 → p'^(1/4 : ℝ) = q' + s' → p' ≥ p) →
  q = 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_fourth_root_l3671_367164


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3671_367106

theorem gcd_lcm_sum : Nat.gcd 45 125 + Nat.lcm 50 15 = 155 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3671_367106


namespace NUMINAMATH_CALUDE_triangle_function_properties_l3671_367195

/-- Given a triangle ABC with side lengths a, b, c, where c > a > 0 and c > b > 0,
    and a function f(x) = a^x + b^x - c^x, prove that:
    1. For all x < 1, f(x) > 0
    2. There exists x > 0 such that xa^x, b^x, c^x cannot form a triangle
    3. If ABC is obtuse, then there exists x ∈ (1, 2) such that f(x) = 0 -/
theorem triangle_function_properties (a b c : ℝ) (h1 : c > a) (h2 : a > 0) (h3 : c > b) (h4 : b > 0)
  (h5 : a + b > c) (f : ℝ → ℝ) (hf : ∀ x, f x = a^x + b^x - c^x) :
  (∀ x < 1, f x > 0) ∧
  (∃ x > 0, ¬ (xa^x + b^x > c^x ∧ xa^x + c^x > b^x ∧ b^x + c^x > xa^x)) ∧
  (a^2 + b^2 < c^2 → ∃ x ∈ Set.Ioo 1 2, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_function_properties_l3671_367195


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_square_l3671_367113

/-- A quadrilateral with equal length diagonals that are perpendicular to each other -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has diagonals of equal length -/
  equal_diagonals : Bool
  /-- The diagonals are perpendicular to each other -/
  perpendicular_diagonals : Bool

/-- Definition of a square -/
def is_square (q : SpecialQuadrilateral) : Prop :=
  q.equal_diagonals ∧ q.perpendicular_diagonals

/-- Theorem stating that a quadrilateral with equal length diagonals that are perpendicular to each other is a square -/
theorem special_quadrilateral_is_square (q : SpecialQuadrilateral) 
  (h1 : q.equal_diagonals = true) 
  (h2 : q.perpendicular_diagonals = true) : 
  is_square q := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_is_square_l3671_367113


namespace NUMINAMATH_CALUDE_min_value_theorem_l3671_367104

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9 * x + 1 / x^6 ≥ 10 ∧ (9 * x + 1 / x^6 = 10 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3671_367104


namespace NUMINAMATH_CALUDE_gymnastics_students_count_l3671_367172

/-- The position of a student in a rectangular formation. -/
structure Position where
  column_from_right : ℕ
  column_from_left : ℕ
  row_from_back : ℕ
  row_from_front : ℕ

/-- The gymnastics formation. -/
structure GymnasticsFormation where
  eunji_position : Position
  equal_students_per_row : Bool

/-- Calculate the total number of students in the gymnastics formation. -/
def total_students (formation : GymnasticsFormation) : ℕ :=
  let total_columns := formation.eunji_position.column_from_right +
                       formation.eunji_position.column_from_left - 1
  let total_rows := formation.eunji_position.row_from_back +
                    formation.eunji_position.row_from_front - 1
  total_columns * total_rows

/-- The main theorem stating the total number of students in the given formation. -/
theorem gymnastics_students_count :
  ∀ (formation : GymnasticsFormation),
    formation.eunji_position.column_from_right = 8 →
    formation.eunji_position.column_from_left = 14 →
    formation.eunji_position.row_from_back = 7 →
    formation.eunji_position.row_from_front = 15 →
    formation.equal_students_per_row = true →
    total_students formation = 441 := by
  sorry

end NUMINAMATH_CALUDE_gymnastics_students_count_l3671_367172


namespace NUMINAMATH_CALUDE_parallel_tangents_imply_m_values_l3671_367150

-- Define the line and curve
def line (x y : ℝ) : Prop := x - 9 * y - 8 = 0
def curve (x y m : ℝ) : Prop := y = x^3 - m * x^2 + 3 * x

-- Define the tangent slope at a point on the curve
def tangent_slope (x m : ℝ) : ℝ := 3 * x^2 - 2 * m * x + 3

-- State the theorem
theorem parallel_tangents_imply_m_values (m : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    line x₁ y₁ ∧ line x₂ y₂ ∧
    curve x₁ y₁ m ∧ curve x₂ y₂ m ∧
    x₁ ≠ x₂ ∧
    tangent_slope x₁ m = tangent_slope x₂ m) →
  m = 4 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_parallel_tangents_imply_m_values_l3671_367150


namespace NUMINAMATH_CALUDE_class_selection_theorem_l3671_367197

theorem class_selection_theorem (n m k a : ℕ) (h1 : n = 10) (h2 : m = 4) (h3 : k = 4) (h4 : a = 2) :
  (Nat.choose m a) * (Nat.choose (n - m) (k - a)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_selection_theorem_l3671_367197


namespace NUMINAMATH_CALUDE_second_file_size_is_90_l3671_367147

/-- Represents the download scenario with given conditions -/
structure DownloadScenario where
  internetSpeed : ℕ  -- in megabits per minute
  totalTime : ℕ      -- in minutes
  fileCount : ℕ
  firstFileSize : ℕ  -- in megabits
  thirdFileSize : ℕ  -- in megabits

/-- Calculates the size of the second file given a download scenario -/
def secondFileSize (scenario : DownloadScenario) : ℕ :=
  scenario.internetSpeed * scenario.totalTime - scenario.firstFileSize - scenario.thirdFileSize

/-- Theorem stating that the size of the second file is 90 megabits -/
theorem second_file_size_is_90 (scenario : DownloadScenario) 
  (h1 : scenario.internetSpeed = 2)
  (h2 : scenario.totalTime = 120)
  (h3 : scenario.fileCount = 3)
  (h4 : scenario.firstFileSize = 80)
  (h5 : scenario.thirdFileSize = 70) :
  secondFileSize scenario = 90 := by
  sorry

#eval secondFileSize { 
  internetSpeed := 2, 
  totalTime := 120, 
  fileCount := 3, 
  firstFileSize := 80, 
  thirdFileSize := 70 
}

end NUMINAMATH_CALUDE_second_file_size_is_90_l3671_367147


namespace NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l3671_367161

/-- Represents the game setup and elimination process -/
structure DealOrNoDeal where
  totalBoxes : Nat
  highValueBoxes : Nat
  eliminatedBoxes : Nat

/-- Checks if the chance of holding a high-value box is at least 1/2 -/
def hasAtLeastHalfChance (game : DealOrNoDeal) : Prop :=
  let remainingBoxes := game.totalBoxes - game.eliminatedBoxes
  2 * game.highValueBoxes ≥ remainingBoxes

/-- The main theorem to prove -/
theorem deal_or_no_deal_elimination 
  (game : DealOrNoDeal) 
  (h1 : game.totalBoxes = 26)
  (h2 : game.highValueBoxes = 6)
  (h3 : game.eliminatedBoxes = 15) : 
  hasAtLeastHalfChance game :=
sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l3671_367161


namespace NUMINAMATH_CALUDE_function_increasing_condition_l3671_367183

theorem function_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 2 ∧ x₂ > 2 →
    let f := fun x => x^2 - 2*a*x + 3
    (f x₁ - f x₂) / (x₁ - x₂) > 0) →
  a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_function_increasing_condition_l3671_367183


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l3671_367160

theorem triangle_determinant_zero (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) : 
  let matrix : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det matrix = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l3671_367160


namespace NUMINAMATH_CALUDE_two_numbers_satisfy_property_l3671_367157

/-- Given a two-digit positive integer, return the integer obtained by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The property that we're checking for each two-digit number -/
def has_property (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_perfect_square (n + (reverse_digits n)^3)

/-- The main theorem stating that exactly two numbers satisfy the property -/
theorem two_numbers_satisfy_property :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n, n ∈ s ↔ has_property n :=
sorry

end NUMINAMATH_CALUDE_two_numbers_satisfy_property_l3671_367157


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3671_367154

-- Define the "graph number" type
def GraphNumber := ℝ × ℝ × ℝ

-- Define a function to get the graph number of a quadratic function
def getGraphNumber (a b c : ℝ) : GraphNumber :=
  (a, b, c)

-- Define a predicate for when a quadratic function intersects x-axis at one point
def intersectsXAxisOnce (a b c : ℝ) : Prop :=
  b ^ 2 - 4 * a * c = 0

theorem quadratic_function_properties :
  -- Part 1
  getGraphNumber (1/3) (-1) (-1) = (1/3, -1, -1) ∧
  -- Part 2
  ∀ m : ℝ, intersectsXAxisOnce m (m+1) (m+1) → (m = -1 ∨ m = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3671_367154


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3671_367129

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3*y = 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3671_367129


namespace NUMINAMATH_CALUDE_max_curved_sides_is_2n_minus_2_l3671_367158

/-- A type representing a figure formed by the intersection of circles -/
structure IntersectionFigure where
  n : ℕ
  h_n : n ≥ 2

/-- The number of curved sides in an intersection figure -/
def curved_sides (F : IntersectionFigure) : ℕ := sorry

/-- The maximum number of curved sides for a given number of circles -/
def max_curved_sides (n : ℕ) : ℕ := 2 * n - 2

/-- Theorem stating that the maximum number of curved sides is 2n - 2 -/
theorem max_curved_sides_is_2n_minus_2 (F : IntersectionFigure) :
  curved_sides F ≤ max_curved_sides F.n := by
  sorry

end NUMINAMATH_CALUDE_max_curved_sides_is_2n_minus_2_l3671_367158


namespace NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l3671_367102

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

theorem probability_of_letter_in_mathematics :
  (mathematics.toList.toFinset.card : ℚ) / alphabet.card = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l3671_367102


namespace NUMINAMATH_CALUDE_intersection_in_second_quadrant_l3671_367107

/-- The intersection point of two lines is in the second quadrant if and only if k is in the open interval (0, 1/2) -/
theorem intersection_in_second_quadrant (k : ℝ) :
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y - x = 2 * k ∧ x < 0 ∧ y > 0) ↔ 0 < k ∧ k < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_in_second_quadrant_l3671_367107


namespace NUMINAMATH_CALUDE_prob_select_AB_correct_l3671_367117

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of students to be selected
def selected_students : ℕ := 3

-- Define the probability of selecting both A and B
def prob_select_AB : ℚ := 3 / 10

-- Theorem statement
theorem prob_select_AB_correct :
  (Nat.choose (total_students - 2) (selected_students - 2)) / (Nat.choose total_students selected_students) = prob_select_AB :=
sorry

end NUMINAMATH_CALUDE_prob_select_AB_correct_l3671_367117
