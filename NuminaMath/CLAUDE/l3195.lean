import Mathlib

namespace NUMINAMATH_CALUDE_sin_two_x_value_l3195_319578

theorem sin_two_x_value (x : ℝ) (h : Real.sin (π / 4 - x) = 1 / 3) : 
  Real.sin (2 * x) = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_sin_two_x_value_l3195_319578


namespace NUMINAMATH_CALUDE_nine_integer_segments_l3195_319501

/-- Right triangle XYZ with integer leg lengths -/
structure RightTriangle where
  xy : ℕ
  yz : ℕ

/-- The number of different integer length line segments from Y to XZ -/
def countIntegerSegments (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle with XY = 15 and YZ = 20 -/
def specialTriangle : RightTriangle :=
  { xy := 15, yz := 20 }

/-- Theorem stating that the number of integer length segments is 9 -/
theorem nine_integer_segments :
  countIntegerSegments specialTriangle = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_integer_segments_l3195_319501


namespace NUMINAMATH_CALUDE_m_negative_two_sufficient_l3195_319582

/-- Two lines are perpendicular if their slopes are negative reciprocals of each other -/
def are_perpendicular (m : ℝ) : Prop :=
  let k1 := -(m + 1)
  let k2 := -m / (2 * m + 2)
  k1 * k2 = -1

/-- The statement that m = -2 is a sufficient condition for the lines to be perpendicular -/
theorem m_negative_two_sufficient :
  are_perpendicular (-2) := by sorry

end NUMINAMATH_CALUDE_m_negative_two_sufficient_l3195_319582


namespace NUMINAMATH_CALUDE_notebooks_distribution_l3195_319549

theorem notebooks_distribution (C : ℕ) (N : ℕ) : 
  (N / C = C / 8) →  -- Condition 1
  (N / (C / 2) = 16) →  -- Condition 2
  N = 512 := by sorry

end NUMINAMATH_CALUDE_notebooks_distribution_l3195_319549


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3195_319508

def p (x₁ x₂ : ℝ) : Prop := x₁^2 + 5*x₁ - 6 = 0 ∧ x₂^2 + 5*x₂ - 6 = 0

def q (x₁ x₂ : ℝ) : Prop := x₁ + x₂ = -5

theorem p_sufficient_not_necessary :
  (∀ x₁ x₂, p x₁ x₂ → q x₁ x₂) ∧ (∃ y₁ y₂, q y₁ y₂ ∧ ¬p y₁ y₂) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3195_319508


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3195_319557

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 3 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 3 = 0 → y = x) → 
  m = 6 ∨ m = -6 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3195_319557


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3195_319536

theorem no_solution_for_equation : ¬∃ (x : ℝ), (x - 1) / (x - 3) = 2 - 2 / (3 - x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3195_319536


namespace NUMINAMATH_CALUDE_tenth_group_sample_l3195_319524

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_employees : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_sample : ℕ

/-- The number drawn from a specific group in systematic sampling -/
def group_sample (s : SystematicSampling) (group : ℕ) : ℕ :=
  (group - 1) * s.group_size + s.first_sample

/-- Theorem stating the relationship between samples from different groups -/
theorem tenth_group_sample (s : SystematicSampling) 
  (h1 : s.total_employees = 200)
  (h2 : s.sample_size = 40)
  (h3 : s.group_size = 5)
  (h4 : group_sample s 5 = 22) :
  group_sample s 10 = 47 := by
  sorry

#check tenth_group_sample

end NUMINAMATH_CALUDE_tenth_group_sample_l3195_319524


namespace NUMINAMATH_CALUDE_field_trip_buses_l3195_319555

/-- The number of students in the school -/
def total_students : ℕ := 11210

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 118

/-- The number of school buses needed for the field trip -/
def buses_needed : ℕ := total_students / seats_per_bus

theorem field_trip_buses : buses_needed = 95 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_buses_l3195_319555


namespace NUMINAMATH_CALUDE_point_quadrant_l3195_319561

/-- Given that point A(a, -b) is in the first quadrant, prove that point B(a, b) is in the fourth quadrant -/
theorem point_quadrant (a b : ℝ) : 
  (a > 0 ∧ -b > 0) → (a > 0 ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_quadrant_l3195_319561


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_2550_l3195_319586

def sum_of_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_prime_factors_2550 : sum_of_prime_factors 2550 = 27 := by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_2550_l3195_319586


namespace NUMINAMATH_CALUDE_reasoning_forms_mapping_l3195_319562

/-- Represents the different forms of reasoning -/
inductive ReasoningForm
  | Inductive
  | Deductive
  | Analogical

/-- Represents the different reasoning descriptions -/
inductive ReasoningDescription
  | SpecificToSpecific
  | PartToWholeOrIndividualToGeneral
  | GeneralToSpecific

/-- Maps a reasoning description to its corresponding reasoning form -/
def descriptionToForm (d : ReasoningDescription) : ReasoningForm :=
  match d with
  | ReasoningDescription.SpecificToSpecific => ReasoningForm.Analogical
  | ReasoningDescription.PartToWholeOrIndividualToGeneral => ReasoningForm.Inductive
  | ReasoningDescription.GeneralToSpecific => ReasoningForm.Deductive

theorem reasoning_forms_mapping :
  (descriptionToForm ReasoningDescription.SpecificToSpecific = ReasoningForm.Analogical) ∧
  (descriptionToForm ReasoningDescription.PartToWholeOrIndividualToGeneral = ReasoningForm.Inductive) ∧
  (descriptionToForm ReasoningDescription.GeneralToSpecific = ReasoningForm.Deductive) :=
sorry

end NUMINAMATH_CALUDE_reasoning_forms_mapping_l3195_319562


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l3195_319556

def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / k + y^2 / (k - 3) = 1 ∧ k ≠ 0 ∧ k ≠ 3

theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ 0 < k ∧ k < 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l3195_319556


namespace NUMINAMATH_CALUDE_solution_set_of_even_decreasing_function_l3195_319571

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 1) * (a * x + b)

-- State the theorem
theorem solution_set_of_even_decreasing_function (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- f is even
  (∀ x y, 0 < x ∧ x < y → f a b y < f a b x) →  -- f is decreasing on (0, +∞)
  {x : ℝ | f a b (2 - x) < 0} = {x : ℝ | x < 1 ∨ x > 3} :=
by sorry


end NUMINAMATH_CALUDE_solution_set_of_even_decreasing_function_l3195_319571


namespace NUMINAMATH_CALUDE_seating_arrangements_l3195_319597

/-- The number of seats in the row -/
def num_seats : ℕ := 8

/-- The number of people to be seated -/
def num_people : ℕ := 3

/-- The number of empty seats required between people and at the ends -/
def num_partitions : ℕ := 4

/-- The number of ways to arrange the double empty seat -/
def double_seat_arrangements : ℕ := 4

/-- The number of ways to arrange the people -/
def people_arrangements : ℕ := 6

/-- The total number of seating arrangements -/
def total_arrangements : ℕ := double_seat_arrangements * people_arrangements

theorem seating_arrangements :
  total_arrangements = 24 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3195_319597


namespace NUMINAMATH_CALUDE_impossibility_of_tiling_101_square_l3195_319531

theorem impossibility_of_tiling_101_square : ¬ ∃ (a b : ℕ), 4*a + 9*b = 101*101 := by sorry

end NUMINAMATH_CALUDE_impossibility_of_tiling_101_square_l3195_319531


namespace NUMINAMATH_CALUDE_carmichael_family_children_l3195_319560

/-- The Carmichael family problem -/
theorem carmichael_family_children (f : ℝ) (x : ℝ) (y : ℝ) : 
  (45 + f + x * y) / (2 + x) = 25 →   -- average age of the family
  (f + x * y) / (1 + x) = 20 →        -- average age of father and children
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_carmichael_family_children_l3195_319560


namespace NUMINAMATH_CALUDE_initial_trees_count_park_initial_trees_l3195_319538

/-- The number of walnut trees in a park before and after planting. -/
structure ParkTrees where
  initial : ℕ
  planted : ℕ
  final : ℕ

/-- The properties of the park's walnut tree planting scenario. -/
def park_scenario : ParkTrees :=
  { initial := 0,  -- We don't know this value initially
    planted := 33,
    final := 55 }

/-- Theorem stating the relationship between initial, planted, and final number of trees. -/
theorem initial_trees_count (p : ParkTrees) (h1 : p.final = p.initial + p.planted) :
  p.initial = p.final - p.planted := by
  sorry

/-- The main theorem proving the initial number of trees in the park. -/
theorem park_initial_trees :
  park_scenario.initial = 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_trees_count_park_initial_trees_l3195_319538


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3195_319574

theorem right_triangle_hypotenuse (base height hypotenuse : ℝ) : 
  base = 8 →
  (1/2) * base * height = 24 →
  base^2 + height^2 = hypotenuse^2 →
  hypotenuse = 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3195_319574


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3195_319567

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, (1 < x ∧ x < 3) → (1 < x ∧ x < 4)) ∧
  ¬(∀ x : ℝ, (1 < x ∧ x < 4) → (1 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3195_319567


namespace NUMINAMATH_CALUDE_purchase_price_calculation_l3195_319584

theorem purchase_price_calculation (markup : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) :
  markup = 45 ∧ 
  overhead_percentage = 0.20 ∧ 
  net_profit = 12 →
  ∃ purchase_price : ℝ, 
    markup = overhead_percentage * purchase_price + net_profit ∧
    purchase_price = 165 := by
  sorry

end NUMINAMATH_CALUDE_purchase_price_calculation_l3195_319584


namespace NUMINAMATH_CALUDE_revenue_change_l3195_319564

theorem revenue_change 
  (T : ℝ) -- Original tax rate
  (C : ℝ) -- Original consumption
  (h1 : T > 0) -- Assumption: tax rate is positive
  (h2 : C > 0) -- Assumption: consumption is positive
  : 
  let T_new := T * (1 - 0.15) -- New tax rate after 15% decrease
  let C_new := C * (1 + 0.10) -- New consumption after 10% increase
  let R := T * C -- Original revenue
  let R_new := T_new * C_new -- New revenue
  (R_new / R) = 0.935 -- Ratio of new revenue to original revenue
  :=
by sorry

end NUMINAMATH_CALUDE_revenue_change_l3195_319564


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3195_319525

theorem trigonometric_equation_solution (x : Real) :
  (2 * Real.sin (17 * x) + Real.sqrt 3 * Real.cos (5 * x) + Real.sin (5 * x) = 0) ↔
  (∃ k : Int, x = π / 66 * (6 * k - 1) ∨ x = π / 18 * (3 * k + 2)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3195_319525


namespace NUMINAMATH_CALUDE_product_inequality_with_sum_constraint_l3195_319532

theorem product_inequality_with_sum_constraint (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_constraint : x + y + z = 1) :
  (1 + 1/x) * (1 + 1/y) * (1 + 1/z) ≥ 64 ∧
  ((1 + 1/x) * (1 + 1/y) * (1 + 1/z) = 64 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_with_sum_constraint_l3195_319532


namespace NUMINAMATH_CALUDE_sum_of_cyclic_equations_l3195_319540

theorem sum_of_cyclic_equations (p q r : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ r ≠ p →
  q = p * (4 - p) →
  r = q * (4 - q) →
  p = r * (4 - r) →
  p + q + r = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_equations_l3195_319540


namespace NUMINAMATH_CALUDE_parabola_range_l3195_319541

/-- The parabola y = x^2 + 2x + 4 -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem parabola_range :
  ∀ a b : ℝ, -2 ≤ a → a < 3 → b = parabola a → 3 ≤ b ∧ b < 19 :=
by sorry

end NUMINAMATH_CALUDE_parabola_range_l3195_319541


namespace NUMINAMATH_CALUDE_intersection_count_l3195_319569

-- Define the line L
def line_L (x y : ℝ) : Prop := y = 2 + Real.sqrt 3 - Real.sqrt 3 * x

-- Define the ellipse C'
def ellipse_C' (x y : ℝ) : Prop := 4 * x^2 + y^2 = 16

-- Define a point on the line L
def point_on_L : Prop := line_L 1 2

-- Theorem statement
theorem intersection_count :
  point_on_L →
  ∃ (p q : ℝ × ℝ),
    p ≠ q ∧
    line_L p.1 p.2 ∧
    line_L q.1 q.2 ∧
    ellipse_C' p.1 p.2 ∧
    ellipse_C' q.1 q.2 ∧
    ∀ (r : ℝ × ℝ), line_L r.1 r.2 ∧ ellipse_C' r.1 r.2 → r = p ∨ r = q :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l3195_319569


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l3195_319519

theorem chocolate_gain_percent :
  ∀ (C S : ℝ),
  C > 0 →
  S > 0 →
  35 * C = 21 * S →
  ((S - C) / C) * 100 = 200 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l3195_319519


namespace NUMINAMATH_CALUDE_smallest_side_length_is_correct_l3195_319535

/-- Represents a triangle ABC with a point D on AC --/
structure TriangleABCD where
  -- The side length of the equilateral triangle
  side_length : ℕ
  -- The length of CD
  cd_length : ℕ
  -- Ensures that CD is not longer than AC
  h_cd_le_side : cd_length ≤ side_length

/-- The smallest possible side length of an equilateral triangle ABC 
    with a point D on AC such that BD is perpendicular to AC, 
    BD² = 65, and AC and CD are integers --/
def smallest_side_length : ℕ := 8

theorem smallest_side_length_is_correct (t : TriangleABCD) : 
  (t.side_length : ℝ)^2 / 4 + 65 = (t.side_length : ℝ)^2 →
  t.side_length ≥ smallest_side_length := by
  sorry

#check smallest_side_length_is_correct

end NUMINAMATH_CALUDE_smallest_side_length_is_correct_l3195_319535


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_l3195_319594

theorem sum_of_quadratic_roots (x : ℝ) : 
  (2 * x^2 - 8 * x - 10 = 0) → 
  (∃ r s : ℝ, (2 * r^2 - 8 * r - 10 = 0) ∧ 
              (2 * s^2 - 8 * s - 10 = 0) ∧ 
              (r + s = 4)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_l3195_319594


namespace NUMINAMATH_CALUDE_total_cost_plates_and_spoons_l3195_319548

theorem total_cost_plates_and_spoons :
  let num_plates : ℕ := 9
  let price_per_plate : ℚ := 2
  let num_spoons : ℕ := 4
  let price_per_spoon : ℚ := 3/2
  (num_plates : ℚ) * price_per_plate + (num_spoons : ℚ) * price_per_spoon = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_plates_and_spoons_l3195_319548


namespace NUMINAMATH_CALUDE_intersection_range_l3195_319591

def set_A (a : ℝ) : Set ℝ := {x | |x - a| < 1}
def set_B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem intersection_range (a : ℝ) : (set_A a ∩ set_B).Nonempty → 0 < a ∧ a < 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_range_l3195_319591


namespace NUMINAMATH_CALUDE_quadratic_roots_and_inequality_l3195_319589

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop := x^2 + a*x + a - 1/2 = 0

-- Define the set of possible values for a
def valid_a_set : Set ℝ := {a | a ≤ 2 - Real.sqrt 2 ∨ a ≥ 2 + Real.sqrt 2}

-- Define the inequality
def inequality (m t a x₁ x₂ : ℝ) : Prop :=
  m^2 + t*m + 4*Real.sqrt 2 + 6 ≥ (x₁ - 3*x₂)*(x₂ - 3*x₁)

theorem quadratic_roots_and_inequality :
  ∀ a ∈ valid_a_set,
  ∀ x₁ x₂ : ℝ,
  quadratic_equation a x₁ ∧ quadratic_equation a x₂ →
  (∀ t ∈ Set.Icc (-1 : ℝ) 1,
    ∃ m : ℝ, inequality m t a x₁ x₂) ↔
  ∃ m : ℝ, m ≤ -1 ∨ m = 0 ∨ m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_inequality_l3195_319589


namespace NUMINAMATH_CALUDE_det_of_matrix_is_one_l3195_319598

theorem det_of_matrix_is_one : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 7; 2, 3]
  Matrix.det A = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_of_matrix_is_one_l3195_319598


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3195_319539

theorem perfect_square_trinomial (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3195_319539


namespace NUMINAMATH_CALUDE_tetrahedron_volume_lower_bound_l3195_319527

/-- Theorem: The volume of a tetrahedron is at least one-third the product of its opposite edge distances. -/
theorem tetrahedron_volume_lower_bound (d₁ d₂ d₃ V : ℝ) (h₁ : d₁ > 0) (h₂ : d₂ > 0) (h₃ : d₃ > 0) (hV : V > 0) :
  V ≥ (1/3) * d₁ * d₂ * d₃ := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_lower_bound_l3195_319527


namespace NUMINAMATH_CALUDE_pauls_toy_boxes_l3195_319511

theorem pauls_toy_boxes (toys_per_box : ℕ) (total_toys : ℕ) (h1 : toys_per_box = 8) (h2 : total_toys = 32) :
  total_toys / toys_per_box = 4 := by
sorry

end NUMINAMATH_CALUDE_pauls_toy_boxes_l3195_319511


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3195_319553

/-- A parabola with equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- A point on the parabola -/
structure PointOnParabola (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * para.p * x

theorem parabola_focus_distance (para : Parabola) 
  (A : PointOnParabola para) (h : A.y = Real.sqrt 2) :
  (3 * A.x = A.x + para.p / 2) → para.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3195_319553


namespace NUMINAMATH_CALUDE_part_one_calculation_part_two_calculation_l3195_319587

-- Part I
theorem part_one_calculation : -(-1)^1000 - 2.45 * 8 + 2.55 * (-8) = -41 := by
  sorry

-- Part II
theorem part_two_calculation : (1/6 - 1/3 + 0.25) / (-1/12) = -1 := by
  sorry

end NUMINAMATH_CALUDE_part_one_calculation_part_two_calculation_l3195_319587


namespace NUMINAMATH_CALUDE_sin_deg_rad_solutions_l3195_319537

def sin_deg_rad_eq (x : ℝ) : Prop := Real.sin x = Real.sin (x * Real.pi / 180)

theorem sin_deg_rad_solutions :
  ∃ (S : Finset ℝ), S.card = 10 ∧
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 90 ∧ sin_deg_rad_eq x) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 90 ∧ sin_deg_rad_eq x → x ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_sin_deg_rad_solutions_l3195_319537


namespace NUMINAMATH_CALUDE_point_position_on_line_l3195_319516

/-- Given five points on a line, prove the position of a point P satisfying a specific ratio condition -/
theorem point_position_on_line (a b c d : ℝ) :
  let O := (0 : ℝ)
  let A := (2*a : ℝ)
  let B := (3*b : ℝ)
  let C := (4*c : ℝ)
  let D := (5*d : ℝ)
  ∃ P : ℝ, B ≤ P ∧ P ≤ C ∧
    (P - A)^2 * (C - P) = (D - P)^2 * (P - B) →
    P = (8*a*c - 15*b*d) / (8*c - 15*d - 6*b + 4*a) :=
by sorry

end NUMINAMATH_CALUDE_point_position_on_line_l3195_319516


namespace NUMINAMATH_CALUDE_concert_earnings_l3195_319544

/-- Calculates the earnings of each band member from a concert --/
def band_member_earnings (attendees : ℕ) (ticket_price : ℚ) (band_percentage : ℚ) (band_members : ℕ) : ℚ :=
  (attendees : ℚ) * ticket_price * band_percentage / (band_members : ℚ)

/-- Theorem: Given the concert conditions, each band member earns $2,625 --/
theorem concert_earnings :
  band_member_earnings 500 30 (70/100) 4 = 2625 := by
  sorry

end NUMINAMATH_CALUDE_concert_earnings_l3195_319544


namespace NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l3195_319517

/-- Represents a labeling of a tetrahedron's vertices -/
def TetrahedronLabeling := Fin 4 → Fin 4

/-- Checks if a labeling uses each number exactly once -/
def is_valid_labeling (l : TetrahedronLabeling) : Prop :=
  ∀ i : Fin 4, ∃! j : Fin 4, l j = i

/-- Represents a face of the tetrahedron -/
def Face := Fin 3 → Fin 4

/-- Gets the sum of labels on a face -/
def face_sum (l : TetrahedronLabeling) (f : Face) : Nat :=
  (f 0).val + (f 1).val + (f 2).val

/-- Checks if all faces have the same sum -/
def all_faces_equal_sum (l : TetrahedronLabeling) : Prop :=
  ∀ f₁ f₂ : Face, face_sum l f₁ = face_sum l f₂

theorem no_valid_tetrahedron_labeling :
  ¬∃ l : TetrahedronLabeling, is_valid_labeling l ∧ all_faces_equal_sum l := by
  sorry

end NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l3195_319517


namespace NUMINAMATH_CALUDE_boys_in_row_l3195_319579

theorem boys_in_row (left_position right_position between : ℕ) : 
  left_position = 6 →
  right_position = 10 →
  between = 8 →
  left_position - 1 + between + right_position = 24 :=
by sorry

end NUMINAMATH_CALUDE_boys_in_row_l3195_319579


namespace NUMINAMATH_CALUDE_number_of_divisors_180_l3195_319570

theorem number_of_divisors_180 : Nat.card (Nat.divisors 180) = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_180_l3195_319570


namespace NUMINAMATH_CALUDE_floor_sum_equals_129_l3195_319506

theorem floor_sum_equals_129 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + 2*b^2 = 2016)
  (h2 : c^2 + 2*d^2 = 2016)
  (h3 : a*c = 1024)
  (h4 : b*d = 1024) :
  ⌊a + b + c + d⌋ = 129 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_equals_129_l3195_319506


namespace NUMINAMATH_CALUDE_important_rectangle_difference_l3195_319572

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (isBlack : Nat → Nat → Bool)

/-- Represents a rectangle on the chessboard -/
structure Rectangle :=
  (top : Nat)
  (left : Nat)
  (bottom : Nat)
  (right : Nat)

/-- Checks if a rectangle is important -/
def isImportantRectangle (board : Chessboard) (rect : Rectangle) : Bool :=
  board.isBlack rect.top rect.left &&
  board.isBlack rect.top rect.right &&
  board.isBlack rect.bottom rect.left &&
  board.isBlack rect.bottom rect.right

/-- Counts the number of important rectangles containing a square -/
def countImportantRectangles (board : Chessboard) (row : Nat) (col : Nat) : Nat :=
  sorry

/-- Sums the counts for all squares of a given color -/
def sumCounts (board : Chessboard) (isBlack : Bool) : Nat :=
  sorry

/-- The main theorem -/
theorem important_rectangle_difference (board : Chessboard) :
  board.size = 8 →
  (∀ i j, board.isBlack i j = ((i + j) % 2 = 0)) →
  (sumCounts board true) - (sumCounts board false) = 36 :=
sorry

end NUMINAMATH_CALUDE_important_rectangle_difference_l3195_319572


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l3195_319563

theorem walnut_trees_planted (current : ℕ) (final : ℕ) (planted : ℕ) : 
  current = 4 → final = 10 → planted = final - current → planted = 6 := by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l3195_319563


namespace NUMINAMATH_CALUDE_intersection_distance_l3195_319526

/-- The distance between intersection points of a line and circle -/
theorem intersection_distance (x y : ℝ) : 
  -- Line equation
  (y = Real.sqrt 3 * x + Real.sqrt 2 / 2) →
  -- Circle equation
  ((x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 1) →
  -- Distance between intersection points
  ∃ (a b : ℝ × ℝ), 
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l3195_319526


namespace NUMINAMATH_CALUDE_proposition_logic_l3195_319520

theorem proposition_logic (p q : Prop) :
  (p ∨ q) → ¬p → (¬p ∧ q) := by sorry

end NUMINAMATH_CALUDE_proposition_logic_l3195_319520


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3195_319509

/-- Definition of the diamond operation -/
def diamond (a b : ℚ) : ℚ := a * b + 3 * b - a

/-- Theorem stating that if 4 ◇ y = 44, then y = 48/7 -/
theorem diamond_equation_solution :
  diamond 4 y = 44 → y = 48 / 7 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3195_319509


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_l3195_319503

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant. -/
def FourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P. -/
def P : Point :=
  { x := 3, y := -2 }

/-- Theorem stating that P is in the fourth quadrant. -/
theorem P_in_fourth_quadrant : FourthQuadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_l3195_319503


namespace NUMINAMATH_CALUDE_rectangle_k_value_l3195_319551

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the properties of the rectangle
def isValidRectangle (rect : Rectangle) : Prop :=
  rect.A.1 = -3 ∧ 
  rect.A.2 = 1 ∧
  rect.B.1 = 4 ∧
  rect.D.2 = rect.A.2 + (rect.B.1 - rect.A.1)

-- Define the area of the rectangle
def rectangleArea (rect : Rectangle) : ℝ :=
  (rect.B.1 - rect.A.1) * (rect.D.2 - rect.A.2)

-- Theorem statement
theorem rectangle_k_value (rect : Rectangle) (k : ℝ) :
  isValidRectangle rect →
  rectangleArea rect = 70 →
  k > 0 →
  rect.D.2 = k →
  k = 11 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_k_value_l3195_319551


namespace NUMINAMATH_CALUDE_boat_current_rate_l3195_319575

/-- Proves that given a boat with a speed of 15 km/hr in still water, 
    traveling 3.6 km downstream in 12 minutes, the rate of the current is 3 km/hr. -/
theorem boat_current_rate 
  (boat_speed : ℝ) 
  (distance_downstream : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 15) 
  (h2 : distance_downstream = 3.6) 
  (h3 : time_minutes = 12) : 
  ∃ (current_rate : ℝ), current_rate = 3 ∧ 
    distance_downstream = (boat_speed + current_rate) * (time_minutes / 60) := by
  sorry

end NUMINAMATH_CALUDE_boat_current_rate_l3195_319575


namespace NUMINAMATH_CALUDE_simplify_expression_l3195_319502

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  ((m^2 - 3*m + 1) / m + 1) / ((m^2 - 1) / m) = (m - 1) / (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3195_319502


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l3195_319530

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle2D where
  v1 : Point2D
  v2 : Point2D
  v3 : Point2D

/-- Check if three lines are parallel -/
def are_parallel (l1 l2 l3 : Line2D) : Prop :=
  ∃ (k1 k2 : ℝ), l1.a = k1 * l2.a ∧ l1.b = k1 * l2.b ∧
                 l1.a = k2 * l3.a ∧ l1.b = k2 * l3.b

/-- Check if a point lies on a line -/
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a line passes through a point -/
def line_through_point (l : Line2D) (p : Point2D) : Prop :=
  point_on_line p l

/-- Check if a triangle's vertices lie on given lines -/
def triangle_vertices_on_lines (t : Triangle2D) (l1 l2 l3 : Line2D) : Prop :=
  (point_on_line t.v1 l1 ∨ point_on_line t.v1 l2 ∨ point_on_line t.v1 l3) ∧
  (point_on_line t.v2 l1 ∨ point_on_line t.v2 l2 ∨ point_on_line t.v2 l3) ∧
  (point_on_line t.v3 l1 ∨ point_on_line t.v3 l2 ∨ point_on_line t.v3 l3)

/-- Check if a triangle's sides (or extensions) pass through given points -/
def triangle_sides_through_points (t : Triangle2D) (p1 p2 p3 : Point2D) : Prop :=
  ∃ (l1 l2 l3 : Line2D),
    (point_on_line t.v1 l1 ∧ point_on_line t.v2 l1 ∧ line_through_point l1 p1) ∧
    (point_on_line t.v2 l2 ∧ point_on_line t.v3 l2 ∧ line_through_point l2 p2) ∧
    (point_on_line t.v3 l3 ∧ point_on_line t.v1 l3 ∧ line_through_point l3 p3)

theorem triangle_construction_theorem 
  (l1 l2 l3 : Line2D) 
  (p1 p2 p3 : Point2D) 
  (h_parallel : are_parallel l1 l2 l3) :
  ∃ (t : Triangle2D), 
    triangle_vertices_on_lines t l1 l2 l3 ∧ 
    triangle_sides_through_points t p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l3195_319530


namespace NUMINAMATH_CALUDE_equation_solutions_count_l3195_319547

theorem equation_solutions_count : 
  ∃! n : ℕ, n = (Finset.filter 
    (λ (p : ℕ × ℕ) => (p.1 - 4)^2 - 35 = (p.2 - 3)^2) 
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l3195_319547


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3195_319552

/-- Represents a parallelogram with given properties -/
structure Parallelogram where
  area : ℝ
  base : ℝ
  altitude : ℝ
  angle : ℝ
  area_eq : area = 200
  altitude_eq : altitude = 2 * base
  angle_eq : angle = 60

/-- Theorem: The base of the parallelogram with given properties is 10 meters -/
theorem parallelogram_base_length (p : Parallelogram) : p.base = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3195_319552


namespace NUMINAMATH_CALUDE_odd_integers_equality_l3195_319565

theorem odd_integers_equality (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_integers_equality_l3195_319565


namespace NUMINAMATH_CALUDE_jones_earnings_proof_l3195_319507

/-- Dr. Jones' monthly earnings in dollars -/
def monthly_earnings : ℝ := 6000

/-- Dr. Jones' monthly expenses and savings -/
theorem jones_earnings_proof :
  monthly_earnings - (
    640 +  -- House rental
    380 +  -- Food expense
    (monthly_earnings / 4) +  -- Electric and water bill
    (monthly_earnings / 5)  -- Insurances
  ) = 2280  -- Remaining money after expenses
  := by sorry

end NUMINAMATH_CALUDE_jones_earnings_proof_l3195_319507


namespace NUMINAMATH_CALUDE_tom_seashells_l3195_319596

theorem tom_seashells (yesterday : ℕ) (today : ℕ) 
  (h1 : yesterday = 7) (h2 : today = 4) : 
  yesterday + today = 11 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l3195_319596


namespace NUMINAMATH_CALUDE_math_competition_problem_solving_l3195_319528

theorem math_competition_problem_solving (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.85) 
  (h2 : p2 = 0.80) 
  (h3 : p3 = 0.75) : 
  (p1 + p2 + p3 - 2) ≥ 0.40 := by
sorry

end NUMINAMATH_CALUDE_math_competition_problem_solving_l3195_319528


namespace NUMINAMATH_CALUDE_total_sequences_count_l3195_319590

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 5

/-- The total number of possible sequences of student selections for one week -/
def total_sequences : ℕ := num_students ^ meetings_per_week

/-- Theorem stating that the total number of sequences is 759,375 -/
theorem total_sequences_count : total_sequences = 759375 := by
  sorry

end NUMINAMATH_CALUDE_total_sequences_count_l3195_319590


namespace NUMINAMATH_CALUDE_tank_capacity_l3195_319521

/-- The capacity of a tank with specific inlet and outlet pipe properties -/
theorem tank_capacity
  (outlet_time : ℝ)
  (inlet_rate : ℝ)
  (both_pipes_time : ℝ)
  (h1 : outlet_time = 5)
  (h2 : inlet_rate = 8)
  (h3 : both_pipes_time = 8) :
  ∃ (capacity : ℝ), capacity = 1280 ∧
    capacity / outlet_time - (inlet_rate * 60) = capacity / both_pipes_time :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l3195_319521


namespace NUMINAMATH_CALUDE_rectangular_hyperbola_foci_distance_l3195_319577

/-- The distance between foci of a rectangular hyperbola xy = 4 -/
theorem rectangular_hyperbola_foci_distance : 
  ∀ (x y : ℝ), x * y = 4 → ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_hyperbola_foci_distance_l3195_319577


namespace NUMINAMATH_CALUDE_cafeteria_pies_problem_l3195_319514

def cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies_problem :
  cafeteria_pies 75 19 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_problem_l3195_319514


namespace NUMINAMATH_CALUDE_complex_location_l3195_319599

theorem complex_location (z : ℂ) (h : (1 + Complex.I * Real.sqrt 3) * z = Complex.I * (2 * Real.sqrt 3)) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_location_l3195_319599


namespace NUMINAMATH_CALUDE_calculation_proof_l3195_319545

theorem calculation_proof :
  ((-11 : ℤ) + 8 + (-4) = -7) ∧
  (-1^2023 - |1 - (1/3 : ℚ)| * (-3/2)^2 = -5/2) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3195_319545


namespace NUMINAMATH_CALUDE_umbrella_boots_probability_l3195_319522

theorem umbrella_boots_probability
  (total_umbrellas : ℕ)
  (total_boots : ℕ)
  (prob_boots_and_umbrella : ℚ)
  (h1 : total_umbrellas = 40)
  (h2 : total_boots = 60)
  (h3 : prob_boots_and_umbrella = 1/3) :
  (prob_boots_and_umbrella * total_boots : ℚ) / total_umbrellas = 1/2 :=
sorry

end NUMINAMATH_CALUDE_umbrella_boots_probability_l3195_319522


namespace NUMINAMATH_CALUDE_sector_central_angle_l3195_319581

/-- Given a sector with perimeter 4 and area 1, its central angle is 2 radians. -/
theorem sector_central_angle (r θ : ℝ) : 
  (2 * r + r * θ = 4) →  -- perimeter condition
  ((1 / 2) * r^2 * θ = 1) →  -- area condition
  θ = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3195_319581


namespace NUMINAMATH_CALUDE_initial_marble_difference_l3195_319566

/-- The number of marbles Ed and Doug initially had, and the number Ed currently has -/
structure MarbleCount where
  ed_initial : ℕ
  doug_initial : ℕ
  ed_current : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : MarbleCount) : Prop :=
  m.ed_current = 45 ∧
  m.ed_initial > m.doug_initial ∧
  m.ed_current = m.doug_initial - 11 + 21

/-- The theorem stating the initial difference in marbles -/
theorem initial_marble_difference (m : MarbleCount) 
  (h : marble_problem m) : m.ed_initial - m.doug_initial = 10 := by
  sorry


end NUMINAMATH_CALUDE_initial_marble_difference_l3195_319566


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3195_319534

theorem quadratic_equation_solution : 
  let x₁ := 2 + Real.sqrt 5
  let x₂ := 2 - Real.sqrt 5
  (x₁^2 - 4*x₁ - 1 = 0) ∧ (x₂^2 - 4*x₂ - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3195_319534


namespace NUMINAMATH_CALUDE_typist_salary_problem_l3195_319559

theorem typist_salary_problem (original_salary : ℝ) : 
  (original_salary * 1.1 * 0.95 = 5225) → original_salary = 5000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l3195_319559


namespace NUMINAMATH_CALUDE_three_digit_distinct_sum_remainder_l3195_319588

def S : ℕ := sorry

theorem three_digit_distinct_sum_remainder : S % 1000 = 680 := by sorry

end NUMINAMATH_CALUDE_three_digit_distinct_sum_remainder_l3195_319588


namespace NUMINAMATH_CALUDE_misha_notebooks_l3195_319513

theorem misha_notebooks (a b c : ℕ) 
  (h1 : a + 6 = b + c)  -- Vera bought 6 notebooks less than Misha and Vasya together
  (h2 : b + 10 = a + c) -- Vasya bought 10 notebooks less than Vera and Misha together
  : c = 8 := by  -- Misha bought 8 notebooks
  sorry

end NUMINAMATH_CALUDE_misha_notebooks_l3195_319513


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3195_319592

theorem log_equality_implies_golden_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 18) ∧
  (Real.log q / Real.log 18 = Real.log (p + q) / Real.log 32) →
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3195_319592


namespace NUMINAMATH_CALUDE_graveyard_bones_count_l3195_319500

/-- Represents the number of bones in a skeleton based on its type -/
def bonesInSkeleton (type : String) : ℕ :=
  match type with
  | "woman" => 20
  | "man" => 25
  | "child" => 10
  | _ => 0

/-- Calculates the total number of bones in the graveyard -/
def totalBonesInGraveyard : ℕ :=
  let totalSkeletons : ℕ := 20
  let womenSkeletons : ℕ := totalSkeletons / 2
  let menSkeletons : ℕ := (totalSkeletons - womenSkeletons) / 2
  let childrenSkeletons : ℕ := totalSkeletons - womenSkeletons - menSkeletons
  
  womenSkeletons * bonesInSkeleton "woman" +
  menSkeletons * bonesInSkeleton "man" +
  childrenSkeletons * bonesInSkeleton "child"

theorem graveyard_bones_count :
  totalBonesInGraveyard = 375 := by
  sorry

#eval totalBonesInGraveyard

end NUMINAMATH_CALUDE_graveyard_bones_count_l3195_319500


namespace NUMINAMATH_CALUDE_square_with_semicircles_perimeter_l3195_319533

theorem square_with_semicircles_perimeter (π : Real) (h : π > 0) :
  let side_length := 2 / π
  let semicircle_radius := side_length / 2
  let semicircle_arc_length := π * semicircle_radius
  4 * semicircle_arc_length = 4 := by sorry

end NUMINAMATH_CALUDE_square_with_semicircles_perimeter_l3195_319533


namespace NUMINAMATH_CALUDE_shortest_path_is_3_sqrt_2_l3195_319543

/-- A polyhedron with right dihedral angles that unfolds into three adjacent unit squares -/
structure RightAnglePolyhedron where
  -- We don't need to define the full structure, just the properties we need
  unfoldsToThreeUnitSquares : Bool

/-- Two vertices on the polyhedron -/
structure Vertex where
  -- We don't need to define the full structure, just declare it exists

/-- The shortest path between two vertices on the surface of the polyhedron -/
def shortestPath (p : RightAnglePolyhedron) (v1 v2 : Vertex) : ℝ :=
  sorry

/-- Theorem: The shortest path between opposite corners of the unfolded net is 3√2 -/
theorem shortest_path_is_3_sqrt_2 (p : RightAnglePolyhedron) (x y : Vertex) :
  p.unfoldsToThreeUnitSquares → shortestPath p x y = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_is_3_sqrt_2_l3195_319543


namespace NUMINAMATH_CALUDE_square_of_198_l3195_319585

theorem square_of_198 : 
  (198 : ℕ)^2 = 200^2 - 2 * 200 * 2 + 2^2 := by
  have h1 : 198 = 200 - 2 := by sorry
  have h2 : ∀ (a b : ℕ), (a - b)^2 = a^2 - 2*a*b + b^2 := by sorry
  sorry

end NUMINAMATH_CALUDE_square_of_198_l3195_319585


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l3195_319510

theorem fraction_sum_integer (n : ℕ+) : 
  (1/2 + 1/3 + 1/5 + 1/n : ℚ).isInt → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l3195_319510


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3195_319518

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β)
  (h3 : perpendicular m α) (h4 : parallel m β) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3195_319518


namespace NUMINAMATH_CALUDE_fraction_simplification_l3195_319542

theorem fraction_simplification (x y : ℝ) (h : x^2 ≠ 4*y^2) :
  (-x + 2*y) / (x^2 - 4*y^2) = -1 / (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3195_319542


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_example_l3195_319554

/-- The sum of an arithmetic sequence with given parameters. -/
def arithmetic_sequence_sum (n : ℕ) (a l : ℤ) : ℤ :=
  n * (a + l) / 2

/-- Theorem stating that the sum of the given arithmetic sequence is 175. -/
theorem arithmetic_sequence_sum_example :
  arithmetic_sequence_sum 10 (-5) 40 = 175 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_example_l3195_319554


namespace NUMINAMATH_CALUDE_circle_properties_l3195_319505

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 0)

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- Theorem stating that the given equation describes a circle with the specified center and radius -/
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3195_319505


namespace NUMINAMATH_CALUDE_clusters_per_spoonful_l3195_319573

/-- Represents the number of clusters of oats in a box of cereal -/
def clusters_per_box : ℕ := 500

/-- Represents the number of bowlfuls in a box of cereal -/
def bowlfuls_per_box : ℕ := 5

/-- Represents the number of spoonfuls in a bowl of cereal -/
def spoonfuls_per_bowl : ℕ := 25

/-- Theorem stating that the number of clusters of oats in each spoonful is 4 -/
theorem clusters_per_spoonful :
  clusters_per_box / (bowlfuls_per_box * spoonfuls_per_bowl) = 4 := by
  sorry

end NUMINAMATH_CALUDE_clusters_per_spoonful_l3195_319573


namespace NUMINAMATH_CALUDE_gcd_6Tn_nplus1_le_3_exists_gcd_6Tn_nplus1_eq_3_l3195_319595

/-- The nth triangular number -/
def triangular_number (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

/-- Theorem: The greatest common divisor of 6Tn and n+1 is at most 3 -/
theorem gcd_6Tn_nplus1_le_3 (n : ℕ+) :
  Nat.gcd (6 * triangular_number n) (n + 1) ≤ 3 :=
sorry

/-- Theorem: There exists an n such that the greatest common divisor of 6Tn and n+1 is exactly 3 -/
theorem exists_gcd_6Tn_nplus1_eq_3 :
  ∃ n : ℕ+, Nat.gcd (6 * triangular_number n) (n + 1) = 3 :=
sorry

end NUMINAMATH_CALUDE_gcd_6Tn_nplus1_le_3_exists_gcd_6Tn_nplus1_eq_3_l3195_319595


namespace NUMINAMATH_CALUDE_hundred_passengers_sixteen_stops_l3195_319593

/-- The number of ways passengers can disembark from a train -/
def ways_to_disembark (num_passengers : ℕ) (num_stops : ℕ) : ℕ :=
  num_stops ^ num_passengers

/-- Theorem: 100 passengers disembarking at 16 stops results in 16^100 possibilities -/
theorem hundred_passengers_sixteen_stops :
  ways_to_disembark 100 16 = 16^100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_passengers_sixteen_stops_l3195_319593


namespace NUMINAMATH_CALUDE_tan_theta_eq_two_implies_expression_eq_neg_two_l3195_319512

theorem tan_theta_eq_two_implies_expression_eq_neg_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / 
  (Real.sin (π / 2 - θ) - Real.sin (π - θ)) = -2 := by sorry

end NUMINAMATH_CALUDE_tan_theta_eq_two_implies_expression_eq_neg_two_l3195_319512


namespace NUMINAMATH_CALUDE_half_page_ad_cost_l3195_319576

/-- Calculates the cost of a half-page advertisement in Math Magazine -/
theorem half_page_ad_cost : 
  let page_length : ℝ := 9
  let page_width : ℝ := 12
  let cost_per_square_inch : ℝ := 8
  let full_page_area : ℝ := page_length * page_width
  let half_page_area : ℝ := full_page_area / 2
  let total_cost : ℝ := half_page_area * cost_per_square_inch
  total_cost = 432 :=
by sorry

end NUMINAMATH_CALUDE_half_page_ad_cost_l3195_319576


namespace NUMINAMATH_CALUDE_dream_number_k_value_l3195_319558

def is_dream_number (p : ℕ) : Prop :=
  p ≥ 100 ∧ p < 1000 ∧
  let h := p / 100
  let t := (p / 10) % 10
  let u := p % 10
  h ≠ 0 ∧ t ≠ 0 ∧ u ≠ 0 ∧
  (h - t : ℤ) = (t - u : ℤ)

def m (p : ℕ) : ℕ :=
  let h := p / 100
  let t := (p / 10) % 10
  let u := p % 10
  (10 * h + t) + (10 * t + u)

def n (p : ℕ) : ℕ :=
  let h := p / 100
  let u := p % 10
  (10 * h + u) + (10 * u + h)

def F (p : ℕ) : ℚ :=
  (m p - n p : ℚ) / 9

def s (x y : ℕ) : ℕ := 10 * x + y + 502

def t (a b : ℕ) : ℕ := 10 * a + b + 200

theorem dream_number_k_value
  (x y a b : ℕ)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 7)
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 1 ≤ b ∧ b ≤ 9)
  (hs : is_dream_number (s x y))
  (ht : is_dream_number (t a b))
  (h_eq : 2 * F (s x y) + F (t a b) = -1)
  : F (s x y) / F (s x y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_dream_number_k_value_l3195_319558


namespace NUMINAMATH_CALUDE_five_pairs_l3195_319568

/-- The number of pairs of natural numbers (a, b) satisfying the given conditions -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    a ≥ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 6)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 5 pairs of natural numbers satisfying the conditions -/
theorem five_pairs : count_pairs = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_pairs_l3195_319568


namespace NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l3195_319504

/-- Given x = (3 + √8)^1000, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1 -/
theorem x_times_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 8) ^ 1000
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l3195_319504


namespace NUMINAMATH_CALUDE_trapezoid_area_is_correct_l3195_319580

/-- The area of a trapezoid bounded by y = x, y = 10, y = 5, and the y-axis -/
def trapezoidArea : ℝ := 37.5

/-- The line y = x -/
def lineYeqX (x : ℝ) : ℝ := x

/-- The line y = 10 -/
def lineY10 (x : ℝ) : ℝ := 10

/-- The line y = 5 -/
def lineY5 (x : ℝ) : ℝ := 5

/-- The y-axis (x = 0) -/
def yAxis : Set ℝ := {x | x = 0}

theorem trapezoid_area_is_correct :
  trapezoidArea = 37.5 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_correct_l3195_319580


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l3195_319546

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem murtha_pebble_collection :
  arithmetic_sum 1 1 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l3195_319546


namespace NUMINAMATH_CALUDE_sum_distinct_prime_divisors_1800_l3195_319550

def sum_of_distinct_prime_divisors (n : Nat) : Nat :=
  (Nat.factors n).toFinset.sum id

theorem sum_distinct_prime_divisors_1800 :
  sum_of_distinct_prime_divisors 1800 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_divisors_1800_l3195_319550


namespace NUMINAMATH_CALUDE_exam_pass_rate_l3195_319515

theorem exam_pass_rate (hindi : ℝ) (english : ℝ) (math : ℝ) 
  (hindi_english : ℝ) (hindi_math : ℝ) (english_math : ℝ) (all_three : ℝ)
  (h1 : hindi = 25) (h2 : english = 48) (h3 : math = 35)
  (h4 : hindi_english = 27) (h5 : hindi_math = 20) (h6 : english_math = 15)
  (h7 : all_three = 10) :
  100 - (hindi + english + math - hindi_english - hindi_math - english_math + all_three) = 44 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_rate_l3195_319515


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3195_319529

theorem quadratic_equation_solution : 
  ∀ y : ℝ, y^2 - 2*y + 1 = -(y - 1)*(y - 3) → y = 1 ∨ y = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3195_319529


namespace NUMINAMATH_CALUDE_problem_statement_l3195_319523

theorem problem_statement (x Q : ℝ) (h : 2 * (5 * x + 3 * Real.pi) = Q) :
  4 * (10 * x + 6 * Real.pi + 2) = 4 * Q + 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3195_319523


namespace NUMINAMATH_CALUDE_interest_difference_approximation_l3195_319583

/-- Calculates the balance after compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Calculates the balance difference between two accounts --/
def balance_difference (
  principal : ℝ)
  (rate1 : ℝ) (periods1 : ℕ)
  (rate2 : ℝ) (periods2 : ℕ) : ℝ :=
  compound_interest principal rate1 periods1 - compound_interest principal rate2 periods2

theorem interest_difference_approximation :
  let principal := 10000
  let rate_alice := 0.03  -- 6% / 2 for semiannual compounding
  let periods_alice := 20  -- 2 * 10 years
  let rate_bob := 0.04
  let periods_bob := 10
  abs (balance_difference principal rate_alice periods_alice rate_bob periods_bob - 3259) < 1 := by
  sorry

#eval balance_difference 10000 0.03 20 0.04 10

end NUMINAMATH_CALUDE_interest_difference_approximation_l3195_319583
