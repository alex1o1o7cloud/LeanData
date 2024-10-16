import Mathlib

namespace NUMINAMATH_CALUDE_burger_combinations_count_l2183_218369

/-- The number of condiment options available. -/
def num_condiments : ℕ := 10

/-- The number of choices for meat patties. -/
def num_patty_choices : ℕ := 3

/-- The number of choices for bun types. -/
def num_bun_choices : ℕ := 2

/-- Calculates the total number of different burger combinations. -/
def total_burger_combinations : ℕ := 2^num_condiments * num_patty_choices * num_bun_choices

/-- Theorem stating that the total number of different burger combinations is 6144. -/
theorem burger_combinations_count : total_burger_combinations = 6144 := by
  sorry

end NUMINAMATH_CALUDE_burger_combinations_count_l2183_218369


namespace NUMINAMATH_CALUDE_scheduling_methods_count_l2183_218315

/-- The number of days for scheduling --/
def num_days : ℕ := 7

/-- The number of volunteers --/
def num_volunteers : ℕ := 4

/-- Function to calculate the number of scheduling methods --/
def scheduling_methods : ℕ := 
  (num_days.choose num_volunteers) * (num_volunteers.factorial / 2)

/-- Theorem stating that the number of scheduling methods is 420 --/
theorem scheduling_methods_count : scheduling_methods = 420 := by
  sorry

end NUMINAMATH_CALUDE_scheduling_methods_count_l2183_218315


namespace NUMINAMATH_CALUDE_charlottes_schedule_is_correct_l2183_218309

/-- Represents the number of hours it takes to walk each type of dog -/
structure WalkingTime where
  poodle : ℕ
  chihuahua : ℕ
  labrador : ℕ

/-- Represents the schedule for the week -/
structure Schedule where
  monday_poodles : ℕ
  monday_chihuahuas : ℕ
  tuesday_chihuahuas : ℕ
  wednesday_labradors : ℕ

/-- The total available hours for dog-walking in the week -/
def total_hours : ℕ := 32

/-- The walking times for each type of dog -/
def walking_times : WalkingTime := {
  poodle := 2,
  chihuahua := 1,
  labrador := 3
}

/-- Charlotte's schedule for the week -/
def charlottes_schedule : Schedule := {
  monday_poodles := 8,  -- This is what we want to prove
  monday_chihuahuas := 2,
  tuesday_chihuahuas := 2,
  wednesday_labradors := 4
}

/-- Calculate the total hours spent walking dogs based on the schedule and walking times -/
def calculate_total_hours (s : Schedule) (w : WalkingTime) : ℕ :=
  s.monday_poodles * w.poodle +
  s.monday_chihuahuas * w.chihuahua +
  s.tuesday_chihuahuas * w.chihuahua +
  s.wednesday_labradors * w.labrador

/-- Theorem stating that Charlotte's schedule is correct -/
theorem charlottes_schedule_is_correct :
  calculate_total_hours charlottes_schedule walking_times = total_hours :=
by sorry

end NUMINAMATH_CALUDE_charlottes_schedule_is_correct_l2183_218309


namespace NUMINAMATH_CALUDE_complex_division_equality_l2183_218384

theorem complex_division_equality : ∀ (i : ℂ), i^2 = -1 →
  (3 - 2*i) / (2 + i) = 4/5 - 7/5*i :=
by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l2183_218384


namespace NUMINAMATH_CALUDE_circle_equation_l2183_218347

/-- The standard equation of a circle with center (-3, 4) and radius 2 is (x+3)^2 + (y-4)^2 = 4 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (-3, 4)
  let radius : ℝ := 2
  (x + 3)^2 + (y - 4)^2 = 4 ↔ 
    ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2183_218347


namespace NUMINAMATH_CALUDE_victors_class_size_l2183_218335

theorem victors_class_size (total_skittles : ℕ) (skittles_per_classmate : ℕ) 
  (h1 : total_skittles = 25)
  (h2 : skittles_per_classmate = 5) :
  total_skittles / skittles_per_classmate = 5 :=
by sorry

end NUMINAMATH_CALUDE_victors_class_size_l2183_218335


namespace NUMINAMATH_CALUDE_crabapple_recipients_count_l2183_218307

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of different sequences of crabapple recipients in a week -/
def crabapple_sequences : ℕ := num_students * (num_students - 1) * (num_students - 2)

/-- Theorem stating the number of different sequences of crabapple recipients -/
theorem crabapple_recipients_count :
  crabapple_sequences = 2730 :=
by sorry

end NUMINAMATH_CALUDE_crabapple_recipients_count_l2183_218307


namespace NUMINAMATH_CALUDE_modified_triangle_pieces_count_l2183_218330

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a1 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

/-- Represents the modified triangle construction -/
structure ModifiedTriangle where
  rows : ℕ
  rodStart : ℕ
  rodIncrease : ℕ
  connectorStart : ℕ
  connectorIncrease : ℕ
  supportStart : ℕ
  supportIncrease : ℕ
  supportStartRow : ℕ

/-- Calculates the total number of pieces in the modified triangle -/
def totalPieces (t : ModifiedTriangle) : ℕ :=
  let rods := arithmeticSum t.rodStart t.rodIncrease t.rows
  let connectors := arithmeticSum t.connectorStart t.connectorIncrease (t.rows + 1)
  let supports := arithmeticSum t.supportStart t.supportIncrease (t.rows - t.supportStartRow + 1)
  rods + connectors + supports

/-- The theorem to be proved -/
theorem modified_triangle_pieces_count :
  let t : ModifiedTriangle := {
    rows := 10,
    rodStart := 4,
    rodIncrease := 5,
    connectorStart := 1,
    connectorIncrease := 1,
    supportStart := 2,
    supportIncrease := 2,
    supportStartRow := 3
  }
  totalPieces t = 395 := by sorry

end NUMINAMATH_CALUDE_modified_triangle_pieces_count_l2183_218330


namespace NUMINAMATH_CALUDE_crayons_remaining_l2183_218341

theorem crayons_remaining (initial : ℕ) (lost : ℕ) (remaining : ℕ) : 
  initial = 253 → lost = 70 → remaining = initial - lost → remaining = 183 := by
  sorry

end NUMINAMATH_CALUDE_crayons_remaining_l2183_218341


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2183_218342

/-- Given a quadratic equation (k-1)x^2 - 2x + 1 = 0 with two real roots,
    the range of values for k is k ≤ 2 and k ≠ 1 -/
theorem quadratic_equation_roots_range (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (k - 1) * x^2 - 2 * x + 1 = 0 ∧ (k - 1) * y^2 - 2 * y + 1 = 0) →
  (k ≤ 2 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2183_218342


namespace NUMINAMATH_CALUDE_problem_solution_l2183_218300

theorem problem_solution (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 6) : 
  (a^2 + b^2 = 13) ∧ ((a - b)^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2183_218300


namespace NUMINAMATH_CALUDE_parabola_circle_theorem_l2183_218373

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℝ
  eq : (y : ℝ) → (x : ℝ) → Prop := fun y x => y^2 = 4 * a * x

/-- Represents a circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (p.a, 0)

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop := fun x => x = -p.a

/-- The chord length of the intersection between a parabola and its directrix -/
def chordLength (p : Parabola) : ℝ := sorry

/-- The standard equation of a circle -/
def standardEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem parabola_circle_theorem (p : Parabola) (c : Circle) :
  p.a = 1 →
  c.center = focus p →
  chordLength p = 6 →
  ∀ x y, standardEquation c x y ↔ (x - 1)^2 + y^2 = 13 := by sorry

end NUMINAMATH_CALUDE_parabola_circle_theorem_l2183_218373


namespace NUMINAMATH_CALUDE_percentage_less_than_150000_l2183_218313

/-- Represents the percentage of counties in a specific population range -/
structure PopulationRange where
  percentage : ℝ
  lower_bound : ℕ
  upper_bound : Option ℕ

/-- Proves that the percentage of counties with fewer than 150,000 residents is 83% -/
theorem percentage_less_than_150000 
  (less_than_10000 : PopulationRange)
  (between_10000_and_49999 : PopulationRange)
  (between_50000_and_149999 : PopulationRange)
  (more_than_150000 : PopulationRange)
  (h1 : less_than_10000.percentage = 21)
  (h2 : between_10000_and_49999.percentage = 44)
  (h3 : between_50000_and_149999.percentage = 18)
  (h4 : more_than_150000.percentage = 17)
  (h5 : less_than_10000.upper_bound = some 9999)
  (h6 : between_10000_and_49999.lower_bound = 10000 ∧ between_10000_and_49999.upper_bound = some 49999)
  (h7 : between_50000_and_149999.lower_bound = 50000 ∧ between_50000_and_149999.upper_bound = some 149999)
  (h8 : more_than_150000.lower_bound = 150000 ∧ more_than_150000.upper_bound = none)
  (h9 : less_than_10000.percentage + between_10000_and_49999.percentage + between_50000_and_149999.percentage + more_than_150000.percentage = 100) :
  less_than_10000.percentage + between_10000_and_49999.percentage + between_50000_and_149999.percentage = 83 := by
  sorry


end NUMINAMATH_CALUDE_percentage_less_than_150000_l2183_218313


namespace NUMINAMATH_CALUDE_extra_bottles_eq_three_l2183_218382

/-- The number of juice bottles Paul drinks per day -/
def paul_bottles : ℕ := 3

/-- The number of juice bottles Donald drinks per day -/
def donald_bottles : ℕ := 9

/-- The difference between Donald's daily juice consumption and twice Paul's daily juice consumption -/
def extra_bottles : ℕ := donald_bottles - 2 * paul_bottles

theorem extra_bottles_eq_three : extra_bottles = 3 := by
  sorry

end NUMINAMATH_CALUDE_extra_bottles_eq_three_l2183_218382


namespace NUMINAMATH_CALUDE_cube_surface_area_l2183_218376

theorem cube_surface_area (V : ℝ) (h : V = 64) : 
  6 * (V ^ (1/3))^2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2183_218376


namespace NUMINAMATH_CALUDE_quadratic_properties_l2183_218316

/-- Quadratic function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_down : a < 0)
  (h_b : b < 0)
  (h_c : c > 0)
  (h_sym : ∀ x, f a b c (x - 1) = f a b c (-x - 1)) :
  abc > 0 ∧
  (∀ x, -3 < x ∧ x < 1 → f a b c x > 0) ∧
  f a b c (-4) = -10/3 ∧
  f a b c 2 = -10/3 ∧
  f a b c 1 = 0 ∧
  f a b c (-3/2) = 5/2 ∧
  f a b c (-1/2) = 5/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2183_218316


namespace NUMINAMATH_CALUDE_emily_marbles_problem_l2183_218398

theorem emily_marbles_problem (emily_initial : ℕ) (emily_final : ℕ) : 
  emily_initial = 6 →
  emily_final = 8 →
  ∃ (additional_marbles : ℕ),
    emily_final = emily_initial + 2 * emily_initial - 
      ((emily_initial + 2 * emily_initial) / 2 + additional_marbles) ∧
    additional_marbles = 1 :=
by sorry

end NUMINAMATH_CALUDE_emily_marbles_problem_l2183_218398


namespace NUMINAMATH_CALUDE_area_is_two_side_a_value_l2183_218336

/-- Triangle ABC with given properties -/
structure TriangleABC where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Condition: cos A = 3/5
  cos_A : a^2 = b^2 + c^2 - 2*b*c*(3/5)
  -- Condition: AB · AC = 3
  dot_product : b*c*(3/5) = 3
  -- Condition: b - c = 3
  side_diff : b - c = 3

/-- The area of triangle ABC is 2 -/
theorem area_is_two (t : TriangleABC) : (1/2) * t.b * t.c * (4/5) = 2 := by sorry

/-- The value of side a is √13 -/
theorem side_a_value (t : TriangleABC) : t.a = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_area_is_two_side_a_value_l2183_218336


namespace NUMINAMATH_CALUDE_circle_tangent_vector_theorem_l2183_218319

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the point A
def A : ℝ × ℝ := (3, 4)

-- Define the vector equation
def VectorEquation (P M N : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), A.1 = x * M.1 + y * N.1 ∧ A.2 = x * M.2 + y * N.2

-- Define the trajectory equation
def TrajectoryEquation (P : ℝ × ℝ) : Prop :=
  P.1 ≠ 0 ∧ P.2 ≠ 0 ∧ P.1^2 / 16 + P.2^2 / 9 = (P.1 + P.2 - 1)^2

theorem circle_tangent_vector_theorem :
  ∀ (P M N : ℝ × ℝ),
    P ∈ Circle (0, 0) 1 →
    VectorEquation P M N →
    TrajectoryEquation P ∧ (∀ x y : ℝ, 9 * x^2 + 16 * y^2 ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_vector_theorem_l2183_218319


namespace NUMINAMATH_CALUDE_sector_area_l2183_218301

/-- The area of a circular sector with central angle π/3 and radius 4 is 8π/3 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 4) :
  (1 / 2) * θ * r^2 = 8 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2183_218301


namespace NUMINAMATH_CALUDE_sun_rise_set_differences_l2183_218368

/-- Represents a geographical location with latitude and longitude -/
structure Location where
  latitude : Real
  longitude : Real

/-- Calculates the time difference of sunrise between two locations given a solar declination -/
def sunriseTimeDifference (loc1 loc2 : Location) (solarDeclination : Real) : Real :=
  sorry

/-- Calculates the time difference of sunset between two locations given a solar declination -/
def sunsetTimeDifference (loc1 loc2 : Location) (solarDeclination : Real) : Real :=
  sorry

def szeged : Location := { latitude := 46.25, longitude := 20.1667 }
def nyiregyhaza : Location := { latitude := 47.9667, longitude := 21.75 }
def winterSolsticeDeclination : Real := -23.5

theorem sun_rise_set_differences (ε : Real) :
  (ε > 0) →
  (∃ d : Real, abs (d - winterSolsticeDeclination) < ε ∧
    sunriseTimeDifference szeged nyiregyhaza d > 0) ∧
  (∃ d : Real, sunsetTimeDifference szeged nyiregyhaza d < 0) :=
sorry

end NUMINAMATH_CALUDE_sun_rise_set_differences_l2183_218368


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2183_218337

theorem sphere_surface_area (cube_surface_area : ℝ) (sphere_radius : ℝ) : 
  cube_surface_area = 24 →
  (2 * sphere_radius) ^ 2 = 3 * (cube_surface_area / 6) →
  4 * Real.pi * sphere_radius ^ 2 = 12 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2183_218337


namespace NUMINAMATH_CALUDE_min_value_xyz_l2183_218333

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 27) :
  2 * x + 3 * y + 6 * z ≥ 54 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' * y' * z' = 27 ∧ 2 * x' + 3 * y' + 6 * z' = 54 := by
sorry

end NUMINAMATH_CALUDE_min_value_xyz_l2183_218333


namespace NUMINAMATH_CALUDE_cookies_remaining_l2183_218322

-- Define the given conditions
def pieces_per_pack : ℕ := 3
def original_packs : ℕ := 226
def packs_given_away : ℕ := 3

-- Define the theorem
theorem cookies_remaining :
  (original_packs - packs_given_away) * pieces_per_pack = 669 := by
  sorry

end NUMINAMATH_CALUDE_cookies_remaining_l2183_218322


namespace NUMINAMATH_CALUDE_arc_length_45_degrees_l2183_218348

theorem arc_length_45_degrees (circle_circumference : Real) (central_angle : Real) (arc_length : Real) : 
  circle_circumference = 72 →
  central_angle = 45 →
  arc_length = circle_circumference * (central_angle / 360) →
  arc_length = 9 :=
by sorry

end NUMINAMATH_CALUDE_arc_length_45_degrees_l2183_218348


namespace NUMINAMATH_CALUDE_molecular_weight_BaF2_is_175_l2183_218394

/-- The molecular weight of BaF2 in grams per mole. -/
def molecular_weight_BaF2 : ℝ := 175

/-- The number of moles of BaF2 in the given condition. -/
def moles_BaF2 : ℝ := 6

/-- The total weight of the given moles of BaF2 in grams. -/
def total_weight_BaF2 : ℝ := 1050

/-- Theorem stating that the molecular weight of BaF2 is 175 grams/mole. -/
theorem molecular_weight_BaF2_is_175 :
  molecular_weight_BaF2 = total_weight_BaF2 / moles_BaF2 :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_BaF2_is_175_l2183_218394


namespace NUMINAMATH_CALUDE_no_sum_of_150_consecutive_integers_l2183_218324

theorem no_sum_of_150_consecutive_integers : ¬ ∃ (k : ℤ),
  (150 * k + 11325 = 678900) ∨
  (150 * k + 11325 = 1136850) ∨
  (150 * k + 11325 = 1000000) ∨
  (150 * k + 11325 = 2251200) ∨
  (150 * k + 11325 = 1876800) :=
by sorry

end NUMINAMATH_CALUDE_no_sum_of_150_consecutive_integers_l2183_218324


namespace NUMINAMATH_CALUDE_g_odd_g_strictly_increasing_l2183_218399

/-- The function g(x) = lg(x + √(x^2 + 1)) -/
noncomputable def g (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1))

/-- g is an odd function -/
theorem g_odd : ∀ x, g (-x) = -g x := by sorry

/-- g is strictly increasing on ℝ -/
theorem g_strictly_increasing : StrictMono g := by sorry

end NUMINAMATH_CALUDE_g_odd_g_strictly_increasing_l2183_218399


namespace NUMINAMATH_CALUDE_min_value_theorem_l2183_218327

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  x^2 * y^3 * z^2 ≥ 1/2268 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    1/x₀ + 1/y₀ + 1/z₀ = 9 ∧ 
    x₀^2 * y₀^3 * z₀^2 = 1/2268 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2183_218327


namespace NUMINAMATH_CALUDE_a_eq_two_sufficient_not_necessary_l2183_218303

/-- A quadratic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The property that f is increasing on [-1,∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, -1 ≤ x ∧ x < y → f a x < f a y

/-- The statement that a=2 is sufficient but not necessary for f to be increasing on [-1,∞) -/
theorem a_eq_two_sufficient_not_necessary :
  (is_increasing_on_interval 2) ∧
  (∃ a : ℝ, a ≠ 2 ∧ is_increasing_on_interval a) :=
sorry

end NUMINAMATH_CALUDE_a_eq_two_sufficient_not_necessary_l2183_218303


namespace NUMINAMATH_CALUDE_special_ellipse_major_axis_length_l2183_218325

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : Bool
  /-- The x-coordinate of both foci -/
  foci_x : ℝ
  /-- The y-coordinate of the first focus -/
  focus1_y : ℝ
  /-- The y-coordinate of the second focus -/
  focus2_y : ℝ

/-- The length of the major axis of the special ellipse -/
def majorAxisLength (e : SpecialEllipse) : ℝ := 2

/-- Theorem stating that the length of the major axis is 2 for the given ellipse -/
theorem special_ellipse_major_axis_length (e : SpecialEllipse) 
  (h1 : e.tangent_to_axes = true)
  (h2 : e.foci_x = 4)
  (h3 : e.focus1_y = 1 + 2 * Real.sqrt 2)
  (h4 : e.focus2_y = 1 - 2 * Real.sqrt 2) :
  majorAxisLength e = 2 := by sorry

end NUMINAMATH_CALUDE_special_ellipse_major_axis_length_l2183_218325


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l2183_218380

theorem right_triangle_perimeter_equals_area :
  ∀ a b c : ℕ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = (a * b) / 2 →
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨
   (a = 12 ∧ b = 5 ∧ c = 13) ∨
   (a = 6 ∧ b = 8 ∧ c = 10) ∨
   (a = 8 ∧ b = 6 ∧ c = 10)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l2183_218380


namespace NUMINAMATH_CALUDE_ab_zero_necessary_not_sufficient_l2183_218386

def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

theorem ab_zero_necessary_not_sufficient (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) → a * b = 0 ∧
  ∃ a b, a * b = 0 ∧ ¬(∀ x, f a b x = -f a b (-x)) :=
sorry

end NUMINAMATH_CALUDE_ab_zero_necessary_not_sufficient_l2183_218386


namespace NUMINAMATH_CALUDE_ice_block_volume_l2183_218390

theorem ice_block_volume (V : ℝ) : 
  V > 0 →
  (8/35 : ℝ) * V = 0.15 →
  V = 0.65625 := by sorry

end NUMINAMATH_CALUDE_ice_block_volume_l2183_218390


namespace NUMINAMATH_CALUDE_min_sum_squares_over_a_squared_l2183_218339

theorem min_sum_squares_over_a_squared (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a ≠ 0) : 
  ((a + b)^2 + (b + c)^2 + (c + a)^2) / a^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_over_a_squared_l2183_218339


namespace NUMINAMATH_CALUDE_min_value_theorem_l2183_218331

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : 2*m + 2*n = 2) : 
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧ 
    (∀ (x y : ℝ), x > 0 → y > 0 → 2*x + 2*y = 2 → 1/x + 2/y ≥ min_val) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2183_218331


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_three_l2183_218305

theorem sqrt_difference_equals_two_sqrt_three :
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_three_l2183_218305


namespace NUMINAMATH_CALUDE_arc_length_ln_sin_l2183_218366

open Real MeasureTheory

/-- The arc length of the curve y = ln(sin x) from x = π/3 to x = π/2 is (1/2) ln 3 -/
theorem arc_length_ln_sin (f : ℝ → ℝ) (h : ∀ x, f x = Real.log (Real.sin x)) :
  ∫ x in Set.Icc (π/3) (π/2), sqrt (1 + (deriv f x)^2) = (1/2) * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_ln_sin_l2183_218366


namespace NUMINAMATH_CALUDE_randys_trip_length_l2183_218351

theorem randys_trip_length :
  ∀ (total_length : ℚ),
  (1 / 4 : ℚ) * total_length +  -- First part (gravel road)
  30 +                          -- Second part (pavement)
  (1 / 6 : ℚ) * total_length    -- Third part (dirt road)
  = total_length                -- Sum of all parts equals total length
  →
  total_length = 360 / 7 := by
sorry

end NUMINAMATH_CALUDE_randys_trip_length_l2183_218351


namespace NUMINAMATH_CALUDE_composite_without_prime_factor_of_same_form_l2183_218344

theorem composite_without_prime_factor_of_same_form :
  ∃ (k : ℕ), 
    (∃ (n : ℤ), k = 8 * n + 3) ∧ 
    (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ k = a * b) ∧
    (∀ (p : ℕ), Prime p → p ∣ k → ¬∃ (m : ℤ), p = 8 * m + 3) :=
sorry

end NUMINAMATH_CALUDE_composite_without_prime_factor_of_same_form_l2183_218344


namespace NUMINAMATH_CALUDE_team_selection_ways_l2183_218350

def total_boys : ℕ := 10
def total_girls : ℕ := 12
def team_size : ℕ := 8
def required_boys : ℕ := 5
def required_girls : ℕ := 3

theorem team_selection_ways :
  (Nat.choose total_boys required_boys) * (Nat.choose total_girls required_girls) = 55440 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_ways_l2183_218350


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l2183_218387

/-- Calculates the man's speed with the current given his speed against the current and the current's speed. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the man's speed against the current and the current's speed,
    the man's speed with the current is 12 km/hr. -/
theorem mans_speed_with_current :
  speed_with_current 8 2 = 12 := by
  sorry

#eval speed_with_current 8 2

end NUMINAMATH_CALUDE_mans_speed_with_current_l2183_218387


namespace NUMINAMATH_CALUDE_not_perfect_square_l2183_218321

-- Define the number with 300 ones followed by zeros
def number_with_300_ones : ℕ → ℕ 
  | n => 10^n * (10^300 - 1) / 9

-- Theorem statement
theorem not_perfect_square (n : ℕ) : 
  ¬ ∃ (m : ℕ), number_with_300_ones n = m^2 := by
  sorry


end NUMINAMATH_CALUDE_not_perfect_square_l2183_218321


namespace NUMINAMATH_CALUDE_jessica_bank_balance_l2183_218397

theorem jessica_bank_balance (B : ℝ) : 
  B > 0 → 
  200 = (2/5) * B → 
  let remaining := B - 200
  let deposit := (1/5) * remaining
  remaining + deposit = 360 := by
sorry

end NUMINAMATH_CALUDE_jessica_bank_balance_l2183_218397


namespace NUMINAMATH_CALUDE_train_length_calculation_l2183_218343

/-- The length of a train given its speed, a man's speed in the opposite direction, and the time it takes to pass the man -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (pass_time : ℝ) :
  train_speed = 60 →
  man_speed = 6 →
  pass_time = 32.99736021118311 →
  ∃ (train_length : ℝ), 
    (train_length ≥ 604.99 ∧ train_length ≤ 605.01) ∧
    train_length = (train_speed + man_speed) * (5 / 18) * pass_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2183_218343


namespace NUMINAMATH_CALUDE_binomial_20_19_l2183_218357

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_19_l2183_218357


namespace NUMINAMATH_CALUDE_solve_for_q_l2183_218396

theorem solve_for_q : ∀ (k r q : ℚ),
  (4 / 5 : ℚ) = k / 90 →
  (4 / 5 : ℚ) = (k + r) / 105 →
  (4 / 5 : ℚ) = (q - r) / 150 →
  q = 132 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l2183_218396


namespace NUMINAMATH_CALUDE_line_equation_l2183_218381

/-- The ellipse in the first quadrant -/
def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/3 = 1 ∧ x > 0 ∧ y > 0

/-- The line l -/
def line_l (x y : ℝ) : Prop := ∃ (k m : ℝ), y = k*x + m ∧ k < 0 ∧ m > 0

/-- Points A and B on the ellipse and line l -/
def point_A_B (xA yA xB yB : ℝ) : Prop :=
  ellipse xA yA ∧ ellipse xB yB ∧ line_l xA yA ∧ line_l xB yB

/-- Points M and N on the axes -/
def point_M_N (xM yM xN yN : ℝ) : Prop :=
  xM < 0 ∧ yM = 0 ∧ xN = 0 ∧ yN > 0 ∧ line_l xM yM ∧ line_l xN yN

/-- Equal distances |MA| = |NB| -/
def equal_distances (xA yA xB yB xM yM xN yN : ℝ) : Prop :=
  (xA - xM)^2 + yA^2 = xB^2 + (yB - yN)^2

/-- Distance |MN| = 2√3 -/
def distance_MN (xM yM xN yN : ℝ) : Prop :=
  (xM - xN)^2 + (yM - yN)^2 = 12

theorem line_equation (xA yA xB yB xM yM xN yN : ℝ) :
  point_A_B xA yA xB yB →
  point_M_N xM yM xN yN →
  equal_distances xA yA xB yB xM yM xN yN →
  distance_MN xM yM xN yN →
  ∃ (x y : ℝ), x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0 ∧ line_l x y :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2183_218381


namespace NUMINAMATH_CALUDE_apartment_price_ratio_l2183_218323

theorem apartment_price_ratio :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  1.21 * a + 1.11 * b = 1.15 * (a + b) →
  b / a = 1.5 := by
sorry

end NUMINAMATH_CALUDE_apartment_price_ratio_l2183_218323


namespace NUMINAMATH_CALUDE_min_beacons_for_unique_determination_l2183_218346

/-- Represents a room in the maze -/
structure Room where
  x : ℕ
  y : ℕ

/-- Represents the maze structure -/
structure Maze where
  rooms : List Room
  corridors : List (Room × Room)

/-- Represents a beacon in the maze -/
structure Beacon where
  location : Room

/-- Calculate the distance between two rooms -/
def distance (maze : Maze) (r1 r2 : Room) : ℕ := sorry

/-- Check if a room's location can be uniquely determined -/
def isUniquelyDetermined (maze : Maze) (beacons : List Beacon) (room : Room) : Bool := sorry

/-- The main theorem: At least 3 beacons are needed for unique determination -/
theorem min_beacons_for_unique_determination (maze : Maze) :
  ∀ (beacons : List Beacon),
    (∀ (room : Room), room ∈ maze.rooms → isUniquelyDetermined maze beacons room) →
    beacons.length ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_beacons_for_unique_determination_l2183_218346


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2183_218378

theorem unique_solution_condition (a b c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = (b + 1) * x + c) ↔ b ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2183_218378


namespace NUMINAMATH_CALUDE_x_one_value_l2183_218354

theorem x_one_value (x₁ x₂ x₃ : ℝ) 
  (h_order : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_sum : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/3) : 
  x₁ = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_x_one_value_l2183_218354


namespace NUMINAMATH_CALUDE_rice_and_flour_weights_l2183_218312

/-- The weight of a bag of rice in kilograms -/
def rice_weight : ℝ := 50

/-- The weight of a bag of flour in kilograms -/
def flour_weight : ℝ := 25

/-- The total weight of 8 bags of rice and 6 bags of flour in kilograms -/
def weight1 : ℝ := 550

/-- The total weight of 4 bags of rice and 7 bags of flour in kilograms -/
def weight2 : ℝ := 375

theorem rice_and_flour_weights :
  (8 * rice_weight + 6 * flour_weight = weight1) ∧
  (4 * rice_weight + 7 * flour_weight = weight2) := by
  sorry

end NUMINAMATH_CALUDE_rice_and_flour_weights_l2183_218312


namespace NUMINAMATH_CALUDE_no_solution_equation_l2183_218360

theorem no_solution_equation :
  ∀ x : ℝ, x ≠ 4 → x - 9 / (x - 4) ≠ 4 - 9 / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2183_218360


namespace NUMINAMATH_CALUDE_no_roots_of_composite_l2183_218302

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating that if f(x) = 2x has no real roots, then f(f(x)) = 4x has no real roots -/
theorem no_roots_of_composite (a b c : ℝ) (ha : a ≠ 0) 
  (h : ∀ x : ℝ, f a b c x ≠ 2 * x) : 
  ∀ x : ℝ, f a b c (f a b c x) ≠ 4 * x := by
  sorry

end NUMINAMATH_CALUDE_no_roots_of_composite_l2183_218302


namespace NUMINAMATH_CALUDE_new_average_price_six_toys_average_l2183_218304

def average_price (n : ℕ) (total_cost : ℚ) : ℚ := total_cost / n

theorem new_average_price 
  (n : ℕ) 
  (old_avg : ℚ) 
  (additional_cost : ℚ) : 
  average_price (n + 1) (n * old_avg + additional_cost) = 
    (n * old_avg + additional_cost) / (n + 1) :=
by
  sorry

theorem six_toys_average 
  (dhoni_toys : ℕ) 
  (dhoni_avg : ℚ) 
  (david_toy_price : ℚ) 
  (h1 : dhoni_toys = 5) 
  (h2 : dhoni_avg = 10) 
  (h3 : david_toy_price = 16) :
  average_price (dhoni_toys + 1) (dhoni_toys * dhoni_avg + david_toy_price) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_new_average_price_six_toys_average_l2183_218304


namespace NUMINAMATH_CALUDE_delta_value_l2183_218375

theorem delta_value : ∀ Δ : ℤ, 5 * (-3) = Δ - 3 → Δ = -12 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l2183_218375


namespace NUMINAMATH_CALUDE_pony_discount_rate_l2183_218392

/-- Represents the discount rate for Fox jeans -/
def F : ℝ := sorry

/-- Represents the discount rate for Pony jeans -/
def P : ℝ := sorry

/-- Regular price of Fox jeans -/
def fox_price : ℝ := 15

/-- Regular price of Pony jeans -/
def pony_price : ℝ := 20

/-- Number of Fox jeans purchased -/
def fox_count : ℕ := 3

/-- Number of Pony jeans purchased -/
def pony_count : ℕ := 2

/-- Total savings -/
def total_savings : ℝ := 9

/-- Sum of discount rates -/
def discount_sum : ℝ := 22

theorem pony_discount_rate :
  F + P = discount_sum ∧
  (fox_count * fox_price * F / 100 + pony_count * pony_price * P / 100 = total_savings) →
  P = 18 := by sorry

end NUMINAMATH_CALUDE_pony_discount_rate_l2183_218392


namespace NUMINAMATH_CALUDE_jens_birds_multiple_l2183_218314

theorem jens_birds_multiple (ducks chickens total_birds M : ℕ) : 
  ducks = 150 →
  total_birds = 185 →
  ducks = M * chickens + 10 →
  total_birds = ducks + chickens →
  M = 4 := by
sorry

end NUMINAMATH_CALUDE_jens_birds_multiple_l2183_218314


namespace NUMINAMATH_CALUDE_range_of_m_plus_n_l2183_218358

noncomputable def f (m n x : ℝ) : ℝ := m * 2^x + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x, f m n x = 0) ∧ 
  (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_plus_n_l2183_218358


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l2183_218374

/-- An isosceles triangle with side lengths 3 and 7 has a base of length 3. -/
theorem isosceles_triangle_base (a b : ℝ) (h1 : a = 3 ∨ a = 7) (h2 : b = 3 ∨ b = 7) (h3 : a ≠ b) :
  ∃ (x y : ℝ), x = y ∧ x + y > b ∧ x = 7 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l2183_218374


namespace NUMINAMATH_CALUDE_fraction_unchanged_l2183_218334

theorem fraction_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x) / (2 * (x + y)) = x / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l2183_218334


namespace NUMINAMATH_CALUDE_optimal_strategy_l2183_218311

/-- Represents the cosmetics store problem -/
structure CosmeticsStore where
  m : ℝ  -- Purchase price of cosmetic A
  n : ℝ  -- Purchase price of cosmetic B
  total_items : ℕ  -- Total number of items to purchase

/-- Conditions for the cosmetics store problem -/
def valid_store (store : CosmeticsStore) : Prop :=
  3 * store.m + 4 * store.n = 620 ∧
  5 * store.m + 3 * store.n = 740 ∧
  store.total_items = 200

/-- Calculate the profit for a given purchase strategy -/
def profit (store : CosmeticsStore) (items_a : ℕ) : ℝ :=
  (250 - store.m) * items_a + (200 - store.n) * (store.total_items - items_a)

/-- Check if a purchase strategy is valid -/
def valid_strategy (store : CosmeticsStore) (items_a : ℕ) : Prop :=
  store.m * items_a + store.n * (store.total_items - items_a) ≤ 18100 ∧
  profit store items_a ≥ 27000

/-- Theorem stating the optimal strategy and maximum profit -/
theorem optimal_strategy (store : CosmeticsStore) :
  valid_store store →
  (∃ (items_a : ℕ), valid_strategy store items_a) →
  (∃ (max_items_a : ℕ), 
    valid_strategy store max_items_a ∧
    ∀ (items_a : ℕ), valid_strategy store items_a → 
      profit store max_items_a ≥ profit store items_a) ∧
  (let max_items_a := 105
   profit store max_items_a = 27150 ∧
   valid_strategy store max_items_a ∧
   ∀ (items_a : ℕ), valid_strategy store items_a → 
     profit store max_items_a ≥ profit store items_a) :=
by sorry


end NUMINAMATH_CALUDE_optimal_strategy_l2183_218311


namespace NUMINAMATH_CALUDE_first_berry_count_l2183_218377

/-- A sequence of berry counts where the difference between consecutive counts increases by 2 -/
def BerrySequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = (a (n + 1) - a n) + 2

theorem first_berry_count
  (a : ℕ → ℕ)
  (h_seq : BerrySequence a)
  (h_2 : a 2 = 4)
  (h_3 : a 3 = 7)
  (h_4 : a 4 = 12)
  (h_5 : a 5 = 19) :
  a 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_berry_count_l2183_218377


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l2183_218367

variables (a b x : ℝ)

def f (a b x : ℝ) : ℝ := |2*x - a^4 + (1 - 6*a^2*b^2 - b^4)| + 2*|x - (2*a^3*b + 2*a*b^3 - 1)|

theorem inequality_and_minimum_value :
  (a^4 + 6*a^2*b^2 + b^4 ≥ 4*a*b*(a^2 + b^2)) ∧
  (∀ x, f a b x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l2183_218367


namespace NUMINAMATH_CALUDE_floor_product_equals_45_l2183_218363

theorem floor_product_equals_45 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 45 ↔ x ∈ Set.Icc (7.5) (7 + 2/3) := by sorry

end NUMINAMATH_CALUDE_floor_product_equals_45_l2183_218363


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l2183_218326

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : total_marks = 60) 
  (h3 : correct_marks = 3) 
  (h4 : incorrect_marks = 2) : 
  ∃ (correct : ℕ), correct = 24 ∧ 
    correct + (total_sums - correct) = total_sums ∧ 
    (correct_marks : ℤ) * correct - incorrect_marks * (total_sums - correct) = total_marks :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l2183_218326


namespace NUMINAMATH_CALUDE_train_distance_proof_l2183_218349

/-- The initial distance between two trains -/
def initial_distance : ℝ := 13

/-- The speed of Train A in miles per hour -/
def speed_A : ℝ := 37

/-- The speed of Train B in miles per hour -/
def speed_B : ℝ := 43

/-- The time it takes for Train B to overtake and be ahead of Train A, in hours -/
def overtake_time : ℝ := 5

/-- The distance Train B is ahead of Train A after overtaking, in miles -/
def ahead_distance : ℝ := 17

theorem train_distance_proof :
  initial_distance = (speed_B - speed_A) * overtake_time - ahead_distance :=
by sorry

end NUMINAMATH_CALUDE_train_distance_proof_l2183_218349


namespace NUMINAMATH_CALUDE_inequality_proof_l2183_218308

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2183_218308


namespace NUMINAMATH_CALUDE_minimize_y_l2183_218317

variable (a b c : ℝ)

def y (x : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2

theorem minimize_y :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y a b c x_min ≤ y a b c x ∧ x_min = (a + b + c) / 3 :=
sorry

end NUMINAMATH_CALUDE_minimize_y_l2183_218317


namespace NUMINAMATH_CALUDE_area_common_part_squares_l2183_218329

/-- The area of the common part of two squares -/
theorem area_common_part_squares (small_side : ℝ) (large_side : ℝ) 
  (h1 : small_side = 1)
  (h2 : large_side = 4 * small_side)
  (h3 : small_side > 0)
  (h4 : large_side > small_side) : 
  large_side^2 - (1/2 * small_side^2 + 1/2 * large_side^2) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_area_common_part_squares_l2183_218329


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l2183_218306

theorem polynomial_sum_of_coefficients :
  ∀ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, x^5 + 1 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 33 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l2183_218306


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2183_218332

/-- The measure of the largest angle in a pentagon ABCDE with specific angle conditions -/
theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 108 ∧ 
  B = 72 ∧ 
  C = D ∧ 
  E = 3 * C ∧ 
  A + B + C + D + E = 540 →
  (max A (max B (max C (max D E)))) = 216 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2183_218332


namespace NUMINAMATH_CALUDE_sweet_potato_sharing_l2183_218352

theorem sweet_potato_sharing (total : ℝ) (per_person : ℝ) (h1 : total = 52.5) (h2 : per_person = 5) :
  total - (⌊total / per_person⌋ * per_person) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_sweet_potato_sharing_l2183_218352


namespace NUMINAMATH_CALUDE_equal_angles_with_vectors_l2183_218370

/-- Given two vectors a and b in ℝ², prove that the vector c satisfies the condition
    that the angle between c and a is equal to the angle between c and b. -/
theorem equal_angles_with_vectors (a b c : ℝ × ℝ) : 
  a = (1, 0) → b = (1, -Real.sqrt 3) → c = (Real.sqrt 3, -1) →
  (c.1 * a.1 + c.2 * a.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (a.1^2 + a.2^2)) =
  (c.1 * b.1 + c.2 * b.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (b.1^2 + b.2^2)) := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_with_vectors_l2183_218370


namespace NUMINAMATH_CALUDE_mila_trip_distance_l2183_218355

/-- Represents the details of Mila's trip -/
structure MilaTrip where
  /-- Miles per gallon of Mila's car -/
  mpg : ℝ
  /-- Capacity of Mila's gas tank in gallons -/
  tankCapacity : ℝ
  /-- Miles driven in the first leg of the trip -/
  firstLegMiles : ℝ
  /-- Gallons of gas refueled -/
  refueledGallons : ℝ
  /-- Fraction of tank full upon arrival -/
  finalTankFraction : ℝ

/-- Calculates the total distance of Mila's trip -/
def totalDistance (trip : MilaTrip) : ℝ :=
  trip.firstLegMiles + (trip.tankCapacity - trip.finalTankFraction * trip.tankCapacity) * trip.mpg

/-- Theorem stating that Mila's total trip distance is 826 miles -/
theorem mila_trip_distance :
  ∀ (trip : MilaTrip),
    trip.mpg = 40 ∧
    trip.tankCapacity = 16 ∧
    trip.firstLegMiles = 400 ∧
    trip.refueledGallons = 10 ∧
    trip.finalTankFraction = 1/3 →
    totalDistance trip = 826 := by
  sorry

end NUMINAMATH_CALUDE_mila_trip_distance_l2183_218355


namespace NUMINAMATH_CALUDE_glenburgh_parade_squad_l2183_218389

theorem glenburgh_parade_squad (m : ℕ) : 
  (∃ k : ℕ, 20 * m = 28 * k + 6) → 
  20 * m < 1200 →
  (∀ n : ℕ, (∃ j : ℕ, 20 * n = 28 * j + 6) → 20 * n < 1200 → 20 * n ≤ 20 * m) →
  20 * m = 1160 := by
sorry

end NUMINAMATH_CALUDE_glenburgh_parade_squad_l2183_218389


namespace NUMINAMATH_CALUDE_fraction_simplification_l2183_218340

theorem fraction_simplification (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  (x^8 + 2*x^4*y^2 + y^4) / (x^4 + y^2) = 85 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2183_218340


namespace NUMINAMATH_CALUDE_working_light_bulbs_l2183_218320

theorem working_light_bulbs (total_lamps : ℕ) (bulbs_per_lamp : ℕ) 
  (quarter_lamps : ℕ) (half_lamps : ℕ) (remaining_lamps : ℕ) :
  total_lamps = 20 →
  bulbs_per_lamp = 7 →
  quarter_lamps = total_lamps / 4 →
  half_lamps = total_lamps / 2 →
  remaining_lamps = total_lamps - quarter_lamps - half_lamps →
  (quarter_lamps * (bulbs_per_lamp - 2) + 
   half_lamps * (bulbs_per_lamp - 1) + 
   remaining_lamps * (bulbs_per_lamp - 3)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_working_light_bulbs_l2183_218320


namespace NUMINAMATH_CALUDE_lens_discount_percentage_l2183_218365

theorem lens_discount_percentage (original_price : ℝ) (discounted_price : ℝ) (saving : ℝ) :
  original_price = 300 ∧ 
  discounted_price = 220 ∧ 
  saving = 20 →
  (original_price - (discounted_price + saving)) / original_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_lens_discount_percentage_l2183_218365


namespace NUMINAMATH_CALUDE_f_2_eq_0_f_positive_solution_set_l2183_218393

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (3-a)*x + 2*(1-a)

-- Theorem for f(2) = 0
theorem f_2_eq_0 (a : ℝ) : f a 2 = 0 := by sorry

-- Define the solution set for f(x) > 0
def solution_set (a : ℝ) : Set ℝ :=
  if a < -1 then {x | x < 2 ∨ x > 1-a}
  else if a = -1 then ∅
  else {x | 1-a < x ∧ x < 2}

-- Theorem for the solution set of f(x) > 0
theorem f_positive_solution_set (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ f a x > 0 := by sorry

end NUMINAMATH_CALUDE_f_2_eq_0_f_positive_solution_set_l2183_218393


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2183_218318

theorem unique_solution_exists (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃! x : ℝ, a^x = Real.log x / Real.log (1/4) := by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2183_218318


namespace NUMINAMATH_CALUDE_sector_area_l2183_218345

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 16) (h2 : central_angle = 2) :
  let radius := perimeter / (2 + central_angle)
  (1 / 2) * radius^2 * central_angle = 16 := by sorry

end NUMINAMATH_CALUDE_sector_area_l2183_218345


namespace NUMINAMATH_CALUDE_train_passing_time_l2183_218362

/-- The time taken for a train to pass a telegraph post -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 60 →
  train_speed_kmph = 36 →
  (train_length / (train_speed_kmph * (5/18))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2183_218362


namespace NUMINAMATH_CALUDE_prob_red_or_green_l2183_218388

/-- The probability of drawing a red or green marble from a bag -/
theorem prob_red_or_green (red green yellow : ℕ) (h : red = 4 ∧ green = 3 ∧ yellow = 6) :
  (red + green : ℚ) / (red + green + yellow) = 7 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_or_green_l2183_218388


namespace NUMINAMATH_CALUDE_sqrt_seven_simplification_l2183_218353

theorem sqrt_seven_simplification : 3 * Real.sqrt 7 - Real.sqrt 7 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_simplification_l2183_218353


namespace NUMINAMATH_CALUDE_hello_arrangements_l2183_218338

theorem hello_arrangements : ℕ := by
  -- Define the word length
  let word_length : ℕ := 5

  -- Define the number of repeated letters
  let repeated_letters : ℕ := 1

  -- Define the number of repetitions of the repeated letter
  let repetitions : ℕ := 2

  -- Calculate total permutations
  let total_permutations : ℕ := Nat.factorial word_length

  -- Calculate unique permutations
  let unique_permutations : ℕ := total_permutations / Nat.factorial repeated_letters

  -- Calculate incorrect arrangements
  let incorrect_arrangements : ℕ := unique_permutations - 1

  -- Prove that the number of incorrect arrangements is 59
  sorry

end NUMINAMATH_CALUDE_hello_arrangements_l2183_218338


namespace NUMINAMATH_CALUDE_yah_to_bah_conversion_l2183_218395

/-- Conversion rate between bahs and rahs -/
def bah_to_rah_rate : ℚ := 30 / 20

/-- Conversion rate between rahs and yahs -/
def rah_to_yah_rate : ℚ := 20 / 12

/-- The number of yahs we want to convert -/
def target_yahs : ℕ := 500

/-- The equivalent number of bahs -/
def equivalent_bahs : ℕ := 200

theorem yah_to_bah_conversion :
  (target_yahs : ℚ) / (rah_to_yah_rate * bah_to_rah_rate) = equivalent_bahs := by
  sorry

end NUMINAMATH_CALUDE_yah_to_bah_conversion_l2183_218395


namespace NUMINAMATH_CALUDE_full_price_revenue_l2183_218379

/-- Represents the price of a full-price ticket -/
def full_price : ℝ := sorry

/-- Represents the number of full-price tickets sold -/
def full_price_tickets : ℕ := sorry

/-- Represents the number of discounted tickets sold -/
def discounted_tickets : ℕ := sorry

/-- The total number of tickets sold is 160 -/
axiom total_tickets : full_price_tickets + discounted_tickets = 160

/-- The total revenue is $2400 -/
axiom total_revenue : full_price * full_price_tickets + (full_price / 3) * discounted_tickets = 2400

/-- Theorem stating that the revenue from full-price tickets is $400 -/
theorem full_price_revenue : full_price * full_price_tickets = 400 := by sorry

end NUMINAMATH_CALUDE_full_price_revenue_l2183_218379


namespace NUMINAMATH_CALUDE_series_equals_ten_implies_k_equals_sixteen_l2183_218359

/-- The sum of the infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The series in question -/
noncomputable def series (k : ℝ) : ℝ := 
  4 + geometric_sum ((4 + k) / 5) (1 / 5)

theorem series_equals_ten_implies_k_equals_sixteen :
  ∃ k : ℝ, series k = 10 ∧ k = 16 := by sorry

end NUMINAMATH_CALUDE_series_equals_ten_implies_k_equals_sixteen_l2183_218359


namespace NUMINAMATH_CALUDE_speed_with_stream_is_ten_l2183_218356

/-- The speed of a man rowing a boat with and against a stream. -/
structure BoatSpeed where
  /-- Speed against the stream in km/h -/
  against_stream : ℝ
  /-- Speed in still water in km/h -/
  still_water : ℝ

/-- Calculate the speed with the stream given speeds against stream and in still water -/
def speed_with_stream (bs : BoatSpeed) : ℝ :=
  2 * bs.still_water - bs.against_stream

/-- Theorem stating that given the specified conditions, the speed with the stream is 10 km/h -/
theorem speed_with_stream_is_ten (bs : BoatSpeed) 
    (h1 : bs.against_stream = 10) 
    (h2 : bs.still_water = 7) : 
    speed_with_stream bs = 10 := by
  sorry

end NUMINAMATH_CALUDE_speed_with_stream_is_ten_l2183_218356


namespace NUMINAMATH_CALUDE_farmer_earnings_proof_l2183_218391

/-- Calculates the farmer's earnings after the market fee -/
def farmer_earnings (potatoes carrots tomatoes : ℕ) 
  (potato_bundle_size potato_bundle_price : ℚ)
  (carrot_bundle_size carrot_bundle_price : ℚ)
  (tomato_price canned_tomato_set_size canned_tomato_set_price : ℚ)
  (market_fee_rate : ℚ) : ℚ :=
  let potato_sales := (potatoes / potato_bundle_size) * potato_bundle_price
  let carrot_sales := (carrots / carrot_bundle_size) * carrot_bundle_price
  let fresh_tomato_sales := (tomatoes / 2) * tomato_price
  let canned_tomato_sales := ((tomatoes / 2) / canned_tomato_set_size) * canned_tomato_set_price
  let total_sales := potato_sales + carrot_sales + fresh_tomato_sales + canned_tomato_sales
  let market_fee := total_sales * market_fee_rate
  total_sales - market_fee

/-- The farmer's earnings after the market fee is $618.45 -/
theorem farmer_earnings_proof :
  farmer_earnings 250 320 480 25 1.9 20 2 1 10 15 0.05 = 618.45 := by
  sorry

end NUMINAMATH_CALUDE_farmer_earnings_proof_l2183_218391


namespace NUMINAMATH_CALUDE_power_of_81_l2183_218328

theorem power_of_81 : (81 : ℝ) ^ (5/4) = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l2183_218328


namespace NUMINAMATH_CALUDE_population_halving_time_island_l2183_218372

/-- The time it takes for a population to halve given initial population and net emigration rate -/
def time_to_halve_population (initial_population : ℕ) (net_emigration_rate_per_500 : ℚ) : ℚ :=
  let net_emigration_rate := (initial_population : ℚ) / 500 * net_emigration_rate_per_500
  (initial_population : ℚ) / (2 * net_emigration_rate)

theorem population_halving_time_island (ε : ℚ) :
  ∃ (δ : ℚ), δ > 0 ∧ |time_to_halve_population 5000 35 - 7.14| < δ → δ < ε :=
sorry

end NUMINAMATH_CALUDE_population_halving_time_island_l2183_218372


namespace NUMINAMATH_CALUDE_triangle_problem_l2183_218364

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_problem (t : Triangle) :
  t.b * sin (2 * t.A) - t.a * sin (t.A + t.C) = 0 →
  t.a = 3 →
  (1 / 2) * t.b * t.c * sin t.A = (3 * sqrt 3) / 2 →
  t.A = π / 3 ∧ 1 / t.b + 1 / t.c = sqrt 3 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_problem_l2183_218364


namespace NUMINAMATH_CALUDE_quadratic_intersection_points_specific_quadratic_roots_l2183_218383

theorem quadratic_intersection_points (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔
  b^2 - 4*a*c > 0 :=
by sorry

theorem specific_quadratic_roots :
  ∃ x y : ℝ, x ≠ y ∧ 2 * x^2 + 3 * x - 2 = 0 ∧ 2 * y^2 + 3 * y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_points_specific_quadratic_roots_l2183_218383


namespace NUMINAMATH_CALUDE_decimal_100_to_base_4_has_four_digits_l2183_218310

def to_base_4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem decimal_100_to_base_4_has_four_digits :
  (to_base_4 100).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_decimal_100_to_base_4_has_four_digits_l2183_218310


namespace NUMINAMATH_CALUDE_wilson_pays_twelve_l2183_218361

/-- The total cost of Wilson's purchase at a fast-food restaurant --/
def total_cost (hamburger_price cola_price hamburger_quantity cola_quantity discount : ℕ) : ℕ :=
  hamburger_price * hamburger_quantity + cola_price * cola_quantity - discount

/-- Theorem stating that Wilson pays $12 in total --/
theorem wilson_pays_twelve :
  ∀ (hamburger_price cola_price hamburger_quantity cola_quantity discount : ℕ),
    hamburger_price = 5 →
    cola_price = 2 →
    hamburger_quantity = 2 →
    cola_quantity = 3 →
    discount = 4 →
    total_cost hamburger_price cola_price hamburger_quantity cola_quantity discount = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_wilson_pays_twelve_l2183_218361


namespace NUMINAMATH_CALUDE_pyramid_edges_cannot_form_closed_polygon_l2183_218385

/-- Represents a line segment in 3D space -/
structure Segment3D where
  parallel_to_plane : Bool

/-- Represents a collection of line segments in 3D space -/
structure SegmentCollection where
  segments : List Segment3D
  parallel_count : Nat
  non_parallel_count : Nat

/-- Checks if a collection of segments can form a closed polygon -/
def can_form_closed_polygon (collection : SegmentCollection) : Prop :=
  collection.parallel_count = collection.non_parallel_count ∧
  collection.parallel_count + collection.non_parallel_count = collection.segments.length

theorem pyramid_edges_cannot_form_closed_polygon :
  ¬ ∃ (collection : SegmentCollection),
    collection.parallel_count = 171 ∧
    collection.non_parallel_count = 171 ∧
    can_form_closed_polygon collection :=
by sorry

end NUMINAMATH_CALUDE_pyramid_edges_cannot_form_closed_polygon_l2183_218385


namespace NUMINAMATH_CALUDE_teacher_school_arrangements_l2183_218371

theorem teacher_school_arrangements :
  let n : ℕ := 4  -- number of teachers and schools
  let arrangements := {f : Fin n → Fin n | Function.Surjective f}  -- surjective functions represent valid arrangements
  Fintype.card arrangements = 24 := by
sorry

end NUMINAMATH_CALUDE_teacher_school_arrangements_l2183_218371
