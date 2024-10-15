import Mathlib

namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3741_374168

theorem square_ratio_side_length_sum (area_ratio : ℚ) : 
  area_ratio = 50 / 98 →
  ∃ (a b c : ℕ), 
    (a * (b.sqrt : ℝ) / c : ℝ) ^ 2 = area_ratio ∧
    a = 5 ∧ b = 14 ∧ c = 49 ∧
    a + b + c = 68 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3741_374168


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3741_374107

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)
variable (para : Line → Plane → Prop)

-- Define the given objects
variable (l m : Line) (α : Plane)

-- Define the condition that l is perpendicular to α
variable (h : perpToPlane l α)

-- State the theorem
theorem sufficient_not_necessary :
  (∀ m, para m α → perp m l) ∧
  (∃ m, perp m l ∧ ¬para m α) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3741_374107


namespace NUMINAMATH_CALUDE_student_count_l3741_374110

theorem student_count (right_rank left_rank : ℕ) 
  (h1 : right_rank = 13) 
  (h2 : left_rank = 8) : 
  right_rank + left_rank - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l3741_374110


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3741_374199

theorem smallest_n_for_roots_of_unity : ∃ (n : ℕ), n > 0 ∧ (∀ (z : ℂ), z^4 + z^3 + 1 = 0 → z^n = 1) ∧ (∀ (m : ℕ), m > 0 → (∀ (z : ℂ), z^4 + z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3741_374199


namespace NUMINAMATH_CALUDE_triangle_abc_area_l3741_374123

/-- Reflection of a point over the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflection of a point over the line y = -x -/
def reflect_neg_x (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)

/-- Calculate the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * abs ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

theorem triangle_abc_area :
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := reflect_y_axis A
  let C : ℝ × ℝ := reflect_neg_x B
  triangle_area A B C = 10 := by sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l3741_374123


namespace NUMINAMATH_CALUDE_bad_carrots_count_l3741_374174

theorem bad_carrots_count (carol_carrots mom_carrots brother_carrots good_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : brother_carrots = 23)
  (h4 : good_carrots = 52) :
  carol_carrots + mom_carrots + brother_carrots - good_carrots = 16 := by
sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l3741_374174


namespace NUMINAMATH_CALUDE_sum_of_ab_l3741_374134

theorem sum_of_ab (a b : ℝ) (h1 : a * b = 5) (h2 : 1 / a^2 + 1 / b^2 = 0.6) : 
  a + b = 5 ∨ a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ab_l3741_374134


namespace NUMINAMATH_CALUDE_fifth_result_proof_l3741_374139

theorem fifth_result_proof (total_average : ℚ) (first_five_average : ℚ) (last_seven_average : ℚ) 
  (h1 : total_average = 42)
  (h2 : first_five_average = 49)
  (h3 : last_seven_average = 52) :
  ∃ (fifth_result : ℚ), fifth_result = 147 ∧ 
    (5 * first_five_average + 7 * last_seven_average - fifth_result) / 11 = total_average := by
  sorry

end NUMINAMATH_CALUDE_fifth_result_proof_l3741_374139


namespace NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_product_of_c_values_l3741_374155

theorem quadratic_equation_rational_solutions (c : ℕ+) : 
  (∃ x : ℚ, 3 * x^2 + 17 * x + c.val = 0) ↔ (c.val = 14 ∨ c.val = 24) :=
sorry

theorem product_of_c_values : 
  (∃ c₁ c₂ : ℕ+, c₁ ≠ c₂ ∧ 
    (∃ x : ℚ, 3 * x^2 + 17 * x + c₁.val = 0) ∧ 
    (∃ x : ℚ, 3 * x^2 + 17 * x + c₂.val = 0) ∧
    c₁.val * c₂.val = 336) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_product_of_c_values_l3741_374155


namespace NUMINAMATH_CALUDE_max_a_cubic_function_l3741_374192

/-- Given a cubic function f(x) = a x^3 + b x^2 + c x + d where a ≠ 0,
    and |f'(x)| ≤ 1 for 0 ≤ x ≤ 1, the maximum value of a is 8/3. -/
theorem max_a_cubic_function (a b c d : ℝ) (h₁ : a ≠ 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |3 * a * x^2 + 2 * b * x + c| ≤ 1) →
  a ≤ 8/3 ∧ ∃ b c : ℝ, a = 8/3 ∧ 
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |3 * (8/3) * x^2 + 2 * b * x + c| ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_a_cubic_function_l3741_374192


namespace NUMINAMATH_CALUDE_pizza_party_group_size_l3741_374185

/-- Given a group of people consisting of children and adults, where the number of children
    is twice the number of adults and there are 80 children, prove that the total number
    of people in the group is 120. -/
theorem pizza_party_group_size :
  ∀ (num_children num_adults : ℕ),
    num_children = 80 →
    num_children = 2 * num_adults →
    num_children + num_adults = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_party_group_size_l3741_374185


namespace NUMINAMATH_CALUDE_reservoir_overflow_time_l3741_374194

/-- Represents the state of a reservoir with four pipes -/
structure ReservoirSystem where
  fill_rate_a : ℚ  -- Rate at which Pipe A fills the reservoir (in reservoir/hour)
  fill_rate_c : ℚ  -- Rate at which Pipe C fills the reservoir (in reservoir/hour)
  drain_rate_b : ℚ  -- Rate at which Pipe B drains the reservoir (in reservoir/hour)
  drain_rate_d : ℚ  -- Rate at which Pipe D drains the reservoir (in reservoir/hour)
  initial_level : ℚ  -- Initial water level in the reservoir (as a fraction of full)

/-- Calculates the time until the reservoir overflows -/
def time_to_overflow (sys : ReservoirSystem) : ℚ :=
  sorry

/-- Theorem stating the time to overflow for the given reservoir system -/
theorem reservoir_overflow_time : 
  let sys : ReservoirSystem := {
    fill_rate_a := 1/3,
    fill_rate_c := 1/5,
    drain_rate_b := -1/4,
    drain_rate_d := -1/6,
    initial_level := 1/6
  }
  time_to_overflow sys = 83/4 := by
  sorry


end NUMINAMATH_CALUDE_reservoir_overflow_time_l3741_374194


namespace NUMINAMATH_CALUDE_power_function_through_point_l3741_374126

/-- Given a power function f(x) = x^n that passes through (2, √2/2), prove f(4) = 1/2 -/
theorem power_function_through_point (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 2 = Real.sqrt 2 / 2 →
  f 4 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3741_374126


namespace NUMINAMATH_CALUDE_new_average_is_65_l3741_374102

/-- Calculates the new average marks per paper after additional marks are added. -/
def new_average_marks (num_papers : ℕ) (original_average : ℚ) (additional_marks_geo : ℕ) (additional_marks_hist : ℕ) : ℚ :=
  (num_papers * original_average + additional_marks_geo + additional_marks_hist) / num_papers

/-- Proves that the new average marks per paper is 65 given the specified conditions. -/
theorem new_average_is_65 :
  new_average_marks 11 63 20 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_new_average_is_65_l3741_374102


namespace NUMINAMATH_CALUDE_extra_crayons_l3741_374151

theorem extra_crayons (packs : ℕ) (crayons_per_pack : ℕ) (total_crayons : ℕ) : 
  packs = 4 → crayons_per_pack = 10 → total_crayons = 46 → 
  total_crayons - (packs * crayons_per_pack) = 6 := by
  sorry

end NUMINAMATH_CALUDE_extra_crayons_l3741_374151


namespace NUMINAMATH_CALUDE_sequence_first_element_l3741_374115

def sequence_property (a b c d e : ℚ) : Prop :=
  c = a * b ∧ d = b * c ∧ e = c * d

theorem sequence_first_element :
  ∀ a b c d e : ℚ,
    sequence_property a b c d e →
    c = 3 →
    e = 18 →
    a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_first_element_l3741_374115


namespace NUMINAMATH_CALUDE_function_difference_l3741_374196

theorem function_difference (F : ℝ → ℤ) (h1 : F 3 = 3) (h2 : F 1 = 2) :
  F 3 - F 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_difference_l3741_374196


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l3741_374167

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l3741_374167


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3741_374172

theorem binomial_expansion_coefficient (n : ℕ) : 
  (Nat.choose n 2) * 3^2 = 54 → n = 4 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3741_374172


namespace NUMINAMATH_CALUDE_classroom_gpa_l3741_374100

theorem classroom_gpa (gpa_two_thirds : ℝ) (gpa_whole : ℝ) (gpa_one_third : ℝ) :
  gpa_two_thirds = 66 →
  gpa_whole = 64 →
  (1/3 : ℝ) * gpa_one_third + (2/3 : ℝ) * gpa_two_thirds = gpa_whole →
  gpa_one_third = 60 := by
sorry

end NUMINAMATH_CALUDE_classroom_gpa_l3741_374100


namespace NUMINAMATH_CALUDE_enemy_plane_hit_probability_l3741_374154

theorem enemy_plane_hit_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h_prob_A : prob_A = 0.7) 
  (h_prob_B : prob_B = 0.5) : 
  1 - (1 - prob_A) * (1 - prob_B) = 0.85 := by
sorry

end NUMINAMATH_CALUDE_enemy_plane_hit_probability_l3741_374154


namespace NUMINAMATH_CALUDE_team_size_is_eight_l3741_374112

/-- The number of players in the basketball team -/
def n : ℕ := sorry

/-- The initial average height of the team in centimeters -/
def initial_average : ℝ := 190

/-- The height of the player leaving the team in centimeters -/
def height_leaving : ℝ := 197

/-- The height of the player joining the team in centimeters -/
def height_joining : ℝ := 181

/-- The new average height of the team after the player change in centimeters -/
def new_average : ℝ := 188

/-- Theorem stating that the number of players in the team is 8 -/
theorem team_size_is_eight :
  (n : ℝ) * initial_average - (height_leaving - height_joining) = n * new_average ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_team_size_is_eight_l3741_374112


namespace NUMINAMATH_CALUDE_complex_modulus_10_minus_26i_l3741_374164

theorem complex_modulus_10_minus_26i :
  Complex.abs (10 - 26 * Complex.I) = 2 * Real.sqrt 194 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_10_minus_26i_l3741_374164


namespace NUMINAMATH_CALUDE_night_day_crew_ratio_l3741_374187

theorem night_day_crew_ratio (D N : ℕ) (B : ℝ) : 
  (D * B = (3/4) * (D * B + N * ((3/4) * B))) →
  (N : ℝ) / D = 4/3 := by
sorry

end NUMINAMATH_CALUDE_night_day_crew_ratio_l3741_374187


namespace NUMINAMATH_CALUDE_visual_range_increase_l3741_374137

theorem visual_range_increase (original_range new_range : ℝ) (h1 : original_range = 60) (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_visual_range_increase_l3741_374137


namespace NUMINAMATH_CALUDE_changfei_class_problem_l3741_374173

theorem changfei_class_problem (m n : ℕ+) 
  (h : m.val * (m.val - 1) + m.val * n.val + n.val = 51) : 
  m.val + n.val = 9 := by
sorry

end NUMINAMATH_CALUDE_changfei_class_problem_l3741_374173


namespace NUMINAMATH_CALUDE_percentage_of_democratic_voters_l3741_374159

theorem percentage_of_democratic_voters :
  ∀ (d r : ℝ),
    d + r = 100 →
    0.8 * d + 0.3 * r = 65 →
    d = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_democratic_voters_l3741_374159


namespace NUMINAMATH_CALUDE_congruent_implies_similar_similar_scale_one_implies_congruent_congruent_subset_similar_l3741_374104

-- Define geometric figures
structure GeometricFigure where
  -- Add necessary properties for a geometric figure
  -- This is a simplified representation
  shape : ℕ
  size : ℝ

-- Define congruence relation
def congruent (a b : GeometricFigure) : Prop :=
  a.shape = b.shape ∧ a.size = b.size

-- Define similarity relation with scale factor
def similar (a b : GeometricFigure) (scale : ℝ) : Prop :=
  a.shape = b.shape ∧ a.size = scale * b.size

-- Theorem: Congruent figures are similar with scale factor 1
theorem congruent_implies_similar (a b : GeometricFigure) :
  congruent a b → similar a b 1 := by
  sorry

-- Theorem: Similar figures with scale factor 1 are congruent
theorem similar_scale_one_implies_congruent (a b : GeometricFigure) :
  similar a b 1 → congruent a b := by
  sorry

-- Theorem: Congruent figures are a subset of similar figures
theorem congruent_subset_similar (a b : GeometricFigure) :
  congruent a b → ∃ scale, similar a b scale := by
  sorry

end NUMINAMATH_CALUDE_congruent_implies_similar_similar_scale_one_implies_congruent_congruent_subset_similar_l3741_374104


namespace NUMINAMATH_CALUDE_green_ball_probability_l3741_374118

/-- Represents a container with balls of different colors -/
structure Container where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- Calculates the total number of balls in a container -/
def Container.total (c : Container) : ℕ := c.red + c.green + c.blue

/-- Represents the problem setup with three containers -/
structure BallProblem where
  container1 : Container
  container2 : Container
  container3 : Container

/-- The specific problem instance as described -/
def problem : BallProblem :=
  { container1 := { red := 10, green := 2, blue := 3 }
  , container2 := { red := 5, green := 4, blue := 2 }
  , container3 := { red := 3, green := 5, blue := 3 }
  }

/-- Calculates the probability of selecting a green ball given the problem setup -/
def probabilityGreenBall (p : BallProblem) : ℚ :=
  let p1 := (p.container1.green : ℚ) / p.container1.total
  let p2 := (p.container2.green : ℚ) / p.container2.total
  let p3 := (p.container3.green : ℚ) / p.container3.total
  (p1 + p2 + p3) / 3

theorem green_ball_probability :
  probabilityGreenBall problem = 157 / 495 := by sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3741_374118


namespace NUMINAMATH_CALUDE_two_digit_addition_equation_l3741_374140

theorem two_digit_addition_equation (A B : ℕ) : 
  A ≠ B →
  A < 10 →
  B < 10 →
  6 * A + 10 * B + 2 = 77 →
  B = 1 := by sorry

end NUMINAMATH_CALUDE_two_digit_addition_equation_l3741_374140


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l3741_374181

def arithmetic_sequence (n : ℕ) : ℝ := 2 * n - 1

def kth_order_derivative_sequence (k m n : ℕ) : ℝ :=
  2^(k+2) * m - 2^(k+2) + 1

theorem geometric_sequence_proof (m : ℕ) (hm : m ≥ 2) :
  ∀ n : ℕ, n ≥ 1 → 
    (kth_order_derivative_sequence n m (n+1) - 1) / (kth_order_derivative_sequence n m n - 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l3741_374181


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3741_374150

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I : ℂ) * 2 + 1 * (a : ℂ) + (b : ℂ) = Complex.I * 2 →
  a = 1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3741_374150


namespace NUMINAMATH_CALUDE_ticket123123123_is_red_l3741_374180

-- Define the color type
inductive Color
| Red
| Green
| Blue

-- Define a ticket as a 9-digit number and a color
structure Ticket :=
  (number : Fin 9 → Fin 3)
  (color : Color)

-- Function to check if two tickets have no matching digits
def noMatchingDigits (t1 t2 : Ticket) : Prop :=
  ∀ i : Fin 9, t1.number i ≠ t2.number i

-- Define the given conditions
axiom different_colors (t1 t2 : Ticket) :
  noMatchingDigits t1 t2 → t1.color ≠ t2.color

-- Define the specific tickets mentioned in the problem
def ticket122222222 : Ticket :=
  { number := λ i => if i = 0 then 0 else 1,
    color := Color.Red }

def ticket222222222 : Ticket :=
  { number := λ _ => 1,
    color := Color.Green }

def ticket123123123 : Ticket :=
  { number := λ i => i % 3,
    color := Color.Red }  -- We'll prove this color

-- The theorem to prove
theorem ticket123123123_is_red :
  ticket123123123.color = Color.Red :=
sorry

end NUMINAMATH_CALUDE_ticket123123123_is_red_l3741_374180


namespace NUMINAMATH_CALUDE_distance_between_intersections_l3741_374188

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^4
def curve2 (x y : ℝ) : Prop := x - y^2 = 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve1 x y ∧ curve2 x y}

-- Theorem statement
theorem distance_between_intersections :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt (1 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_distance_between_intersections_l3741_374188


namespace NUMINAMATH_CALUDE_equality_identity_l3741_374120

theorem equality_identity (a : ℝ) : 
  (∃ a : ℝ, (a^4 - 1)^6 ≠ (a^6 - 1)^4) ∧ 
  (∀ a : ℝ, a = -1 ∨ a = 0 ∨ a = 1 → (a^4 - 1)^6 = (a^6 - 1)^4) := by
  sorry


end NUMINAMATH_CALUDE_equality_identity_l3741_374120


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l3741_374171

/-- A rectangular box with a square base -/
structure Box where
  base_side : ℝ
  height : ℝ
  height_eq_double_base : height = 2 * base_side

/-- A square sheet of wrapping paper -/
structure WrappingPaper where
  side_length : ℝ

/-- The configuration of the box on the wrapping paper -/
structure BoxWrappingConfiguration where
  box : Box
  paper : WrappingPaper
  box_centrally_placed : True
  vertices_on_midlines : True
  paper_folds_to_top_center : True

theorem wrapping_paper_area (config : BoxWrappingConfiguration) :
  config.paper.side_length ^ 2 = 16 * config.box.base_side ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l3741_374171


namespace NUMINAMATH_CALUDE_power_of_product_l3741_374135

theorem power_of_product (a b : ℝ) (m : ℕ+) : (a * b) ^ (m : ℕ) = a ^ (m : ℕ) * b ^ (m : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3741_374135


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3741_374136

theorem problem_1 (α β : Real) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := by
sorry

theorem problem_2 (α β : Real)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = Real.sqrt 5 / 5)
  (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  β = π / 8 := by
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3741_374136


namespace NUMINAMATH_CALUDE_cone_base_radius_l3741_374142

/-- Given a cone formed from a sector of a circle with a central angle of 120° and a radius of 6,
    the radius of the base circle of the cone is 2. -/
theorem cone_base_radius (sector_angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) : 
  sector_angle = 120 * π / 180 ∧ 
  sector_radius = 6 ∧ 
  base_radius = sector_angle / (2 * π) * sector_radius → 
  base_radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3741_374142


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l3741_374161

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 46 ∧ 
  corrected_mean = 36.5 →
  (n * corrected_mean - (correct_value - wrong_value)) / n = 36.04 := by
  sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l3741_374161


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l3741_374166

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 6 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l3741_374166


namespace NUMINAMATH_CALUDE_closest_point_on_line_l3741_374116

/-- The line y = -2x + 3 --/
def line (x : ℝ) : ℝ := -2 * x + 3

/-- The point we're finding the closest point to --/
def point : ℝ × ℝ := (2, -1)

/-- The squared distance between two points --/
def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem closest_point_on_line :
  ∀ x : ℝ, squared_distance (x, line x) point ≥ squared_distance (2, line 2) point :=
by sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l3741_374116


namespace NUMINAMATH_CALUDE_fraction_product_l3741_374119

theorem fraction_product : (2 : ℚ) / 9 * (5 : ℚ) / 8 = (5 : ℚ) / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3741_374119


namespace NUMINAMATH_CALUDE_probability_of_arithmetic_progression_l3741_374109

/-- Represents an 8-sided die -/
def Die := Fin 8

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- Checks if a list of four numbers forms an arithmetic progression with common difference 2 -/
def isArithmeticProgression (nums : List ℕ) : Prop :=
  nums.length = numDice ∧
  ∃ a : ℕ, nums = [a, a + 2, a + 4, a + 6]

/-- The set of all possible outcomes when rolling four 8-sided dice -/
def allOutcomes : Finset (List Die) :=
  sorry

/-- The set of favorable outcomes (those forming the desired arithmetic progression) -/
def favorableOutcomes : Finset (List Die) :=
  sorry

/-- The probability of obtaining a favorable outcome -/
theorem probability_of_arithmetic_progression :
  (Finset.card favorableOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 3 / 256 :=
sorry

end NUMINAMATH_CALUDE_probability_of_arithmetic_progression_l3741_374109


namespace NUMINAMATH_CALUDE_smallest_m_is_24_l3741_374146

/-- The set of complex numbers with real part between 1/2 and 2/3 -/
def S : Set ℂ :=
  {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ 2/3}

/-- Definition of the property we want to prove for m -/
def has_nth_root_of_unity (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z ∈ S, z^n = 1

/-- The theorem stating that 24 is the smallest positive integer satisfying the property -/
theorem smallest_m_is_24 :
  has_nth_root_of_unity 24 ∧ ∀ m : ℕ, 0 < m → m < 24 → ¬has_nth_root_of_unity m :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_24_l3741_374146


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3741_374121

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  (a^2 + 2*a*b) - 2*(a^2 + 4*a*b - b) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3741_374121


namespace NUMINAMATH_CALUDE_x_value_l3741_374186

theorem x_value : ∃ x : ℝ, (3 * x) / 7 = 21 ∧ x = 49 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3741_374186


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3741_374182

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < -15 ↔ x ∈ Set.Ioo (-5/2) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3741_374182


namespace NUMINAMATH_CALUDE_ceiling_floor_calculation_l3741_374113

theorem ceiling_floor_calculation : 
  ⌈(15 : ℝ) / 8 * (-45 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-45 : ℝ) / 4⌋⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_calculation_l3741_374113


namespace NUMINAMATH_CALUDE_possible_S_n_plus_1_l3741_374124

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Property: n ≡ S(n) (mod 9) for all natural numbers n -/
axiom S_mod_9 (n : ℕ) : n % 9 = S n % 9

theorem possible_S_n_plus_1 (n : ℕ) (h : S n = 3096) : 
  ∃ m : ℕ, m = n + 1 ∧ S m = 3097 := by sorry

end NUMINAMATH_CALUDE_possible_S_n_plus_1_l3741_374124


namespace NUMINAMATH_CALUDE_percentage_increase_l3741_374149

theorem percentage_increase (initial : ℝ) (final : ℝ) (percentage : ℝ) : 
  initial = 240 → final = 288 → percentage = 20 →
  (final - initial) / initial * 100 = percentage := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3741_374149


namespace NUMINAMATH_CALUDE_inequality_proof_l3741_374184

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3741_374184


namespace NUMINAMATH_CALUDE_florist_chrysanthemums_l3741_374178

theorem florist_chrysanthemums (narcissus : ℕ) (bouquets : ℕ) (flowers_per_bouquet : ℕ) 
  (h1 : narcissus = 75)
  (h2 : bouquets = 33)
  (h3 : flowers_per_bouquet = 5)
  (h4 : narcissus + chrysanthemums = bouquets * flowers_per_bouquet) :
  chrysanthemums = 90 :=
by sorry

end NUMINAMATH_CALUDE_florist_chrysanthemums_l3741_374178


namespace NUMINAMATH_CALUDE_surface_sum_bounds_l3741_374117

/-- Represents a small cube with numbers on its faces -/
structure SmallCube :=
  (faces : Fin 6 → Nat)
  (opposite_sum_seven : ∀ i : Fin 3, faces i + faces (i + 3) = 7)
  (valid_numbers : ∀ i : Fin 6, 1 ≤ faces i ∧ faces i ≤ 6)

/-- Represents the larger cube assembled from 64 small cubes -/
structure LargeCube :=
  (small_cubes : Fin 64 → SmallCube)

/-- The sum of visible numbers on the surface of the larger cube -/
def surface_sum (lc : LargeCube) : Nat :=
  sorry

theorem surface_sum_bounds (lc : LargeCube) :
  144 ≤ surface_sum lc ∧ surface_sum lc ≤ 528 := by
  sorry

end NUMINAMATH_CALUDE_surface_sum_bounds_l3741_374117


namespace NUMINAMATH_CALUDE_joan_gained_two_balloons_l3741_374128

/-- The number of blue balloons Joan gained -/
def balloons_gained (initial final : ℕ) : ℕ := final - initial

/-- Proof that Joan gained 2 blue balloons -/
theorem joan_gained_two_balloons :
  let initial : ℕ := 9
  let final : ℕ := 11
  balloons_gained initial final = 2 := by sorry

end NUMINAMATH_CALUDE_joan_gained_two_balloons_l3741_374128


namespace NUMINAMATH_CALUDE_complex_fraction_power_l3741_374195

theorem complex_fraction_power (i : ℂ) : i^2 = -1 → ((1 + i) / (1 - i))^2013 = i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l3741_374195


namespace NUMINAMATH_CALUDE_original_cost_price_correct_l3741_374193

/-- Represents the original cost price in euros -/
def original_cost_price : ℝ := 55.50

/-- Represents the selling price in dollars -/
def selling_price : ℝ := 100

/-- Represents the profit percentage -/
def profit_percentage : ℝ := 0.30

/-- Represents the exchange rate (dollars per euro) -/
def exchange_rate : ℝ := 1.2

/-- Represents the maintenance cost percentage -/
def maintenance_cost_percentage : ℝ := 0.05

/-- Represents the tax rate for the first 50 euros -/
def tax_rate_first_50 : ℝ := 0.10

/-- Represents the tax rate for amounts above 50 euros -/
def tax_rate_above_50 : ℝ := 0.15

/-- Represents the threshold for the tiered tax system -/
def tax_threshold : ℝ := 50

theorem original_cost_price_correct :
  let cost_price_dollars := selling_price / (1 + profit_percentage)
  let cost_price_euros := cost_price_dollars / exchange_rate
  let maintenance_cost := original_cost_price * maintenance_cost_percentage
  let tax_first_50 := min original_cost_price tax_threshold * tax_rate_first_50
  let tax_above_50 := max (original_cost_price - tax_threshold) 0 * tax_rate_above_50
  cost_price_euros = original_cost_price + maintenance_cost + tax_first_50 + tax_above_50 :=
by sorry

#check original_cost_price_correct

end NUMINAMATH_CALUDE_original_cost_price_correct_l3741_374193


namespace NUMINAMATH_CALUDE_video_recorder_price_l3741_374175

def employee_price (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : ℝ :=
  let retail_price := wholesale_cost * (1 + markup_percentage)
  let discount := retail_price * discount_percentage
  retail_price - discount

theorem video_recorder_price :
  employee_price 200 0.2 0.2 = 192 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_price_l3741_374175


namespace NUMINAMATH_CALUDE_boxes_sold_tuesday_l3741_374190

/-- The number of boxes Kim sold on different days of the week -/
structure BoxesSold where
  friday : ℕ
  thursday : ℕ
  wednesday : ℕ
  tuesday : ℕ

/-- The conditions of the problem -/
def problem_conditions (b : BoxesSold) : Prop :=
  b.friday = 600 ∧
  b.thursday = (3/2 : ℚ) * b.friday ∧
  b.wednesday = 2 * b.thursday ∧
  b.tuesday = 3 * b.wednesday

/-- The theorem stating that under the given conditions, Kim sold 5400 boxes on Tuesday -/
theorem boxes_sold_tuesday (b : BoxesSold) (h : problem_conditions b) : b.tuesday = 5400 := by
  sorry

end NUMINAMATH_CALUDE_boxes_sold_tuesday_l3741_374190


namespace NUMINAMATH_CALUDE_road_graveling_cost_l3741_374169

theorem road_graveling_cost (lawn_length lawn_width road_width gravel_cost : ℝ) :
  lawn_length = 80 ∧
  lawn_width = 60 ∧
  road_width = 10 ∧
  gravel_cost = 5 →
  (lawn_length * road_width + (lawn_width - road_width) * road_width) * gravel_cost = 6500 :=
by sorry

end NUMINAMATH_CALUDE_road_graveling_cost_l3741_374169


namespace NUMINAMATH_CALUDE_geometric_series_problem_l3741_374197

theorem geometric_series_problem (a r : ℝ) 
  (h1 : |r| < 1) 
  (h2 : a / (1 - r) = 7)
  (h3 : a * r / (1 - r^2) = 3) : 
  a + r = 5/2 := by sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l3741_374197


namespace NUMINAMATH_CALUDE_fraction_to_percentage_decimal_seven_fifteenths_to_decimal_l3741_374165

theorem fraction_to_percentage_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = (n : ℚ) / (d : ℚ) := by sorry

theorem seven_fifteenths_to_decimal :
  (7 : ℚ) / 15 = 0.4666666666666667 := by sorry

end NUMINAMATH_CALUDE_fraction_to_percentage_decimal_seven_fifteenths_to_decimal_l3741_374165


namespace NUMINAMATH_CALUDE_sum_ages_in_three_years_l3741_374153

/-- The sum of Josiah and Hans' ages in three years -/
def sum_ages (hans_age : ℕ) (josiah_multiplier : ℕ) (years_later : ℕ) : ℕ :=
  (hans_age * josiah_multiplier + hans_age) + 2 * years_later

/-- Theorem stating the sum of Josiah and Hans' ages in three years -/
theorem sum_ages_in_three_years :
  sum_ages 15 3 3 = 66 := by
  sorry

#eval sum_ages 15 3 3

end NUMINAMATH_CALUDE_sum_ages_in_three_years_l3741_374153


namespace NUMINAMATH_CALUDE_problem_statement_l3741_374130

theorem problem_statement (m n : ℝ) (h : 5 * m + 3 * n = 2) : 
  10 * m + 6 * n - 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3741_374130


namespace NUMINAMATH_CALUDE_parabola_vertex_l3741_374133

/-- The parabola is defined by the equation y = (x - 1)^2 - 2 -/
def parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

/-- The vertex of a parabola is the point where it reaches its maximum or minimum -/
def is_vertex (x y : ℝ) : Prop :=
  ∀ t : ℝ, parabola t ≥ parabola x ∨ parabola t ≤ parabola x

theorem parabola_vertex :
  is_vertex 1 (-2) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3741_374133


namespace NUMINAMATH_CALUDE_ab_value_l3741_374183

theorem ab_value (a b c : ℝ) 
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 27)
  (h3 : a + b + c = 10)
  (h4 : a^3 - b^3 = 36) :
  a * b = -15 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3741_374183


namespace NUMINAMATH_CALUDE_shape_relationships_l3741_374170

-- Define the basic geometric shapes
class Shape

-- Define specific shapes
class Rectangle extends Shape
class Rhombus extends Shape
class Triangle extends Shape
class Parallelogram extends Shape
class Square extends Shape
class Polygon extends Shape

-- Define specific types of triangles
class RightTriangle extends Triangle
class IsoscelesTriangle extends Triangle
class AcuteTriangle extends Triangle
class EquilateralTriangle extends IsoscelesTriangle
class ObtuseTriangle extends Triangle
class ScaleneTriangle extends Triangle

-- Define the relationships between shapes
theorem shape_relationships :
  -- Case 1
  (∃ x : Rectangle, ∃ y : Rhombus, True) ∧
  -- Case 2
  (∃ x : RightTriangle, ∃ y : IsoscelesTriangle, ∃ z : AcuteTriangle, True) ∧
  -- Case 3
  (∃ x : Parallelogram, ∃ y : Rectangle, ∃ z : Square, ∃ u : Rhombus, True) ∧
  -- Case 4
  (∃ x : Polygon, ∃ y : Triangle, ∃ z : IsoscelesTriangle, ∃ u : EquilateralTriangle, ∃ t : RightTriangle, True) ∧
  -- Case 5
  (∃ x : RightTriangle, ∃ y : IsoscelesTriangle, ∃ z : ObtuseTriangle, ∃ u : ScaleneTriangle, True) :=
by
  sorry


end NUMINAMATH_CALUDE_shape_relationships_l3741_374170


namespace NUMINAMATH_CALUDE_trout_ratio_is_three_to_one_l3741_374106

/-- The ratio of trouts caught by Caleb's dad to those caught by Caleb -/
def trout_ratio (caleb_trouts : ℕ) (dad_extra_trouts : ℕ) : ℚ :=
  (caleb_trouts + dad_extra_trouts) / caleb_trouts

theorem trout_ratio_is_three_to_one :
  trout_ratio 2 4 = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_trout_ratio_is_three_to_one_l3741_374106


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_five_l3741_374143

def roll_die : Finset ℕ := Finset.range 6

theorem probability_sum_greater_than_five :
  let outcomes := (roll_die.product roll_die).filter (λ p => p.1 + p.2 > 5)
  (outcomes.card : ℚ) / (roll_die.card * roll_die.card) = 13 / 18 := by
sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_five_l3741_374143


namespace NUMINAMATH_CALUDE_january_salary_l3741_374103

/-- Represents the salary structure for five months --/
structure SalaryStructure where
  jan : ℕ
  feb : ℕ
  mar : ℕ
  apr : ℕ
  may : ℕ

/-- Theorem stating the salary for January given the conditions --/
theorem january_salary (s : SalaryStructure) : s.jan = 4000 :=
  sorry

/-- The average salary for the first four months is 8000 --/
axiom avg_first_four (s : SalaryStructure) : 
  (s.jan + s.feb + s.mar + s.apr) / 4 = 8000

/-- The average salary for the last four months (including bonus) is 8800 --/
axiom avg_last_four (s : SalaryStructure) : 
  (s.feb + s.mar + s.apr + s.may + 1500) / 4 = 8800

/-- The salary for May (excluding bonus) is 6500 --/
axiom may_salary (s : SalaryStructure) : s.may = 6500

/-- February had a deduction of 700 --/
axiom feb_deduction (s : SalaryStructure) (feb_original : ℕ) : 
  s.feb = feb_original - 700

/-- No deductions in other months --/
axiom no_other_deductions (s : SalaryStructure) 
  (jan_original mar_original apr_original : ℕ) : 
  s.jan = jan_original ∧ s.mar = mar_original ∧ s.apr = apr_original

end NUMINAMATH_CALUDE_january_salary_l3741_374103


namespace NUMINAMATH_CALUDE_unique_solution_l3741_374145

/-- Two lines in a 2D plane -/
structure TwoLines where
  l₁ : ℝ → ℝ → ℝ  -- represents ax - by + 4 = 0
  l₂ : ℝ → ℝ → ℝ  -- represents (a - 1)x + y + b = 0
  a : ℝ
  b : ℝ

/-- Condition that l₁ is perpendicular to l₂ -/
def perpendicular (lines : TwoLines) : Prop :=
  lines.a * (lines.a - 1) - lines.b = 0

/-- Condition that l₁ passes through point (-3, -1) -/
def passes_through (lines : TwoLines) : Prop :=
  lines.l₁ (-3) (-1) = 0

/-- Condition that l₁ is parallel to l₂ -/
def parallel (lines : TwoLines) : Prop :=
  lines.a / lines.b = 1 - lines.a

/-- Condition that the distance from origin to both lines is equal -/
def equal_distance (lines : TwoLines) : Prop :=
  4 / lines.b = -lines.b

/-- The main theorem -/
theorem unique_solution (lines : TwoLines) :
  perpendicular lines ∧ passes_through lines →
  parallel lines ∧ equal_distance lines →
  lines.a = 2 ∧ lines.b = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3741_374145


namespace NUMINAMATH_CALUDE_max_distance_between_circles_l3741_374141

/-- The maximum distance between the centers of two circles with 6-inch diameters
    placed within a 12-inch by 14-inch rectangle without extending beyond it. -/
def max_circle_centers_distance : ℝ := 10

/-- The width of the rectangle -/
def rectangle_width : ℝ := 12

/-- The height of the rectangle -/
def rectangle_height : ℝ := 14

/-- The diameter of each circle -/
def circle_diameter : ℝ := 6

/-- Theorem stating that the maximum distance between the centers of the circles is 10 inches -/
theorem max_distance_between_circles :
  ∀ (center1 center2 : ℝ × ℝ),
  (0 ≤ center1.1 ∧ center1.1 ≤ rectangle_width) →
  (0 ≤ center1.2 ∧ center1.2 ≤ rectangle_height) →
  (0 ≤ center2.1 ∧ center2.1 ≤ rectangle_width) →
  (0 ≤ center2.2 ∧ center2.2 ≤ rectangle_height) →
  (∀ (x y : ℝ), (x - center1.1)^2 + (y - center1.2)^2 ≤ (circle_diameter / 2)^2 →
    0 ≤ x ∧ x ≤ rectangle_width ∧ 0 ≤ y ∧ y ≤ rectangle_height) →
  (∀ (x y : ℝ), (x - center2.1)^2 + (y - center2.2)^2 ≤ (circle_diameter / 2)^2 →
    0 ≤ x ∧ x ≤ rectangle_width ∧ 0 ≤ y ∧ y ≤ rectangle_height) →
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 ≤ max_circle_centers_distance^2 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_circles_l3741_374141


namespace NUMINAMATH_CALUDE_line_through_points_with_45_degree_slope_l3741_374157

/-- Given a line passing through points (3, m) and (2, 4) with a slope angle of 45°, prove that m = 5. -/
theorem line_through_points_with_45_degree_slope (m : ℝ) :
  (∃ (line : Set (ℝ × ℝ)), 
    (3, m) ∈ line ∧ 
    (2, 4) ∈ line ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → y - 4 = x - 2)) → 
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_with_45_degree_slope_l3741_374157


namespace NUMINAMATH_CALUDE_lin_peeled_fifteen_l3741_374131

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  lin_rate : ℕ
  christen_join_time : ℕ
  lin_join_time : ℕ

/-- Calculates the number of potatoes Lin peeled -/
def lin_potatoes_peeled (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Lin peeled 15 potatoes -/
theorem lin_peeled_fifteen (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 60)
  (h2 : scenario.homer_rate = 2)
  (h3 : scenario.christen_rate = 3)
  (h4 : scenario.lin_rate = 4)
  (h5 : scenario.christen_join_time = 6)
  (h6 : scenario.lin_join_time = 9) :
  lin_potatoes_peeled scenario = 15 := by
  sorry

end NUMINAMATH_CALUDE_lin_peeled_fifteen_l3741_374131


namespace NUMINAMATH_CALUDE_combined_original_price_l3741_374132

/-- Given a pair of shoes with a 20% discount sold for $480 and a dress with a 30% discount sold for $350,
    prove that the combined original price of the shoes and dress is $1100. -/
theorem combined_original_price 
  (shoes_discount : Real) (dress_discount : Real)
  (shoes_discounted_price : Real) (dress_discounted_price : Real)
  (h1 : shoes_discount = 0.2)
  (h2 : dress_discount = 0.3)
  (h3 : shoes_discounted_price = 480)
  (h4 : dress_discounted_price = 350) :
  (shoes_discounted_price / (1 - shoes_discount)) + (dress_discounted_price / (1 - dress_discount)) = 1100 := by
  sorry

end NUMINAMATH_CALUDE_combined_original_price_l3741_374132


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l3741_374160

def solution_set : Set (Nat × Nat) :=
  {(2, 1), (3, 1), (2, 2), (5, 2), (5, 3), (1, 2), (1, 3)}

theorem integer_fraction_characterization (m n : Nat) :
  m > 0 ∧ n > 0 →
  (∃ k : Int, (n^3 + 1 : Int) = k * (m^2 - 1)) ↔ (m, n) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l3741_374160


namespace NUMINAMATH_CALUDE_line_param_correct_l3741_374127

/-- The line y = 2x - 6 parameterized as (x, y) = (r, 2) + t(3, k) -/
def line_param (r k t : ℝ) : ℝ × ℝ :=
  (r + 3 * t, 2 + k * t)

/-- The line equation y = 2x - 6 -/
def line_eq (x y : ℝ) : Prop :=
  y = 2 * x - 6

theorem line_param_correct (r k : ℝ) : 
  (∀ t, line_eq (line_param r k t).1 (line_param r k t).2) ↔ r = 4 ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_param_correct_l3741_374127


namespace NUMINAMATH_CALUDE_line_inclination_45_degrees_l3741_374179

/-- Given a line passing through points (-2, 1) and (m, 3) with an inclination angle of 45°, prove that m = 0 -/
theorem line_inclination_45_degrees (m : ℝ) : 
  (3 - 1) / (m + 2) = Real.tan (π / 4) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_45_degrees_l3741_374179


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l3741_374147

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 14 = 0

-- Define the line
def line (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l3741_374147


namespace NUMINAMATH_CALUDE_not_fifteen_percent_less_l3741_374148

theorem not_fifteen_percent_less (A B : ℝ) (h : A = B * (1 + 0.15)) : 
  B ≠ A * (1 - 0.15) := by
sorry

end NUMINAMATH_CALUDE_not_fifteen_percent_less_l3741_374148


namespace NUMINAMATH_CALUDE_small_circle_radius_l3741_374191

/-- A design consisting of a small circle surrounded by four equal quarter-circle arcs -/
structure CircleDesign where
  /-- The radius of the large arcs -/
  R : ℝ
  /-- The radius of the small circle -/
  r : ℝ
  /-- The width of the design is 2 cm -/
  width_eq : R + r = 2

/-- The radius of the small circle in a CircleDesign with width 2 cm is 2 - √2 cm -/
theorem small_circle_radius (d : CircleDesign) : d.r = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_small_circle_radius_l3741_374191


namespace NUMINAMATH_CALUDE_crayon_selection_count_l3741_374101

-- Define the number of crayons of each color
def red_crayons : ℕ := 4
def blue_crayons : ℕ := 5
def green_crayons : ℕ := 3
def yellow_crayons : ℕ := 3

-- Define the total number of crayons
def total_crayons : ℕ := red_crayons + blue_crayons + green_crayons + yellow_crayons

-- Define the number of crayons to be selected
def select_count : ℕ := 5

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Define the theorem
theorem crayon_selection_count :
  ∃ (x : ℕ),
    x = combination total_crayons select_count -
        (combination (total_crayons - red_crayons) select_count +
         combination (total_crayons - blue_crayons) select_count +
         combination (total_crayons - green_crayons) select_count +
         combination (total_crayons - yellow_crayons) select_count) +
        -- Placeholder for corrections due to over-subtraction
        0 :=
by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_count_l3741_374101


namespace NUMINAMATH_CALUDE_hybrid_rice_yield_and_conversion_l3741_374122

-- Define the yield per acre of ordinary rice
def ordinary_yield : ℝ := 600

-- Define the yield per acre of hybrid rice
def hybrid_yield : ℝ := 1200

-- Define the acreage difference between fields
def acreage_difference : ℝ := 4

-- Define the harvest of field A (hybrid rice)
def field_A_harvest : ℝ := 9600

-- Define the harvest of field B (ordinary rice)
def field_B_harvest : ℝ := 7200

-- Define the total yield goal
def total_yield_goal : ℝ := 17700

-- Define the minimum acres to be converted
def min_acres_converted : ℝ := 1.5

-- Theorem statement
theorem hybrid_rice_yield_and_conversion :
  (hybrid_yield = 2 * ordinary_yield) ∧
  (field_B_harvest / ordinary_yield - field_A_harvest / hybrid_yield = acreage_difference) ∧
  (field_A_harvest + ordinary_yield * (field_B_harvest / ordinary_yield - min_acres_converted) + hybrid_yield * min_acres_converted ≥ total_yield_goal) := by
  sorry

end NUMINAMATH_CALUDE_hybrid_rice_yield_and_conversion_l3741_374122


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l3741_374162

theorem logarithm_equation_solution (x : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  (Real.log 2 / Real.log x) * (Real.log 2 / Real.log (2 * x)) = Real.log 2 / Real.log (4 * x) →
  x = 2 ^ Real.sqrt 2 ∨ x = 2 ^ (-Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l3741_374162


namespace NUMINAMATH_CALUDE_second_group_has_ten_students_l3741_374138

/-- The number of students in the second kindergartner group -/
def second_group_size : ℕ := 10

/-- The number of students in the first kindergartner group -/
def first_group_size : ℕ := 9

/-- The number of students in the third kindergartner group -/
def third_group_size : ℕ := 11

/-- The number of tissues in each mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The total number of tissues brought by all kindergartner groups -/
def total_tissues : ℕ := 1200

/-- Theorem stating that the second group has 10 students -/
theorem second_group_has_ten_students :
  second_group_size = 10 :=
by
  sorry

#check second_group_has_ten_students

end NUMINAMATH_CALUDE_second_group_has_ten_students_l3741_374138


namespace NUMINAMATH_CALUDE_peters_age_one_third_of_jacobs_l3741_374158

/-- Proves the number of years ago when Peter's age was one-third of Jacob's age -/
theorem peters_age_one_third_of_jacobs (peter_current_age jacob_current_age years_ago : ℕ) :
  peter_current_age = 16 →
  jacob_current_age = peter_current_age + 12 →
  peter_current_age - years_ago = (jacob_current_age - years_ago) / 3 →
  years_ago = 10 := by sorry

end NUMINAMATH_CALUDE_peters_age_one_third_of_jacobs_l3741_374158


namespace NUMINAMATH_CALUDE_cosine_symmetry_axis_l3741_374189

/-- Given a function f(x) = cos(x - π/4), prove that its axis of symmetry is x = π/4 + kπ where k ∈ ℤ -/
theorem cosine_symmetry_axis (f : ℝ → ℝ) (k : ℤ) :
  (∀ x, f x = Real.cos (x - π/4)) →
  (∀ x, f (π/4 + k * π + x) = f (π/4 + k * π - x)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_symmetry_axis_l3741_374189


namespace NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l3741_374108

theorem modular_inverse_28_mod_29 : ∃ x : ℕ, x ≤ 28 ∧ (28 * x) % 29 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l3741_374108


namespace NUMINAMATH_CALUDE_trains_crossing_time_l3741_374163

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (train_length : ℝ) (faster_speed : ℝ) : 
  train_length = 100 →
  faster_speed = 40 →
  (10 : ℝ) / 3 = (2 * train_length) / (faster_speed + faster_speed / 2) := by
  sorry

#check trains_crossing_time

end NUMINAMATH_CALUDE_trains_crossing_time_l3741_374163


namespace NUMINAMATH_CALUDE_hcd_problem_l3741_374152

theorem hcd_problem : (Nat.gcd 2548 364 + 8) - 12 = 360 := by sorry

end NUMINAMATH_CALUDE_hcd_problem_l3741_374152


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3741_374177

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

-- Theorem statement
theorem quadratic_max_value :
  ∃ (max : ℝ), max = -3 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3741_374177


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3741_374144

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 20 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 20 → (x : ℕ) + y ≤ (a : ℕ) + b) →
  (x : ℕ) + y = 81 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3741_374144


namespace NUMINAMATH_CALUDE_total_letters_received_l3741_374129

theorem total_letters_received (brother_letters : ℕ) (greta_extra : ℕ) (mother_multiplier : ℕ) : 
  brother_letters = 40 → 
  greta_extra = 10 → 
  mother_multiplier = 2 → 
  (brother_letters + (brother_letters + greta_extra) + 
   mother_multiplier * (brother_letters + (brother_letters + greta_extra))) = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_total_letters_received_l3741_374129


namespace NUMINAMATH_CALUDE_runners_speed_ratio_l3741_374198

/-- Two runners with different speeds start d miles apart. When running towards each other,
    they meet in s hours. When running in the same direction, the faster runner catches up
    to the slower one in u hours. This theorem proves that the ratio of their speeds is 2. -/
theorem runners_speed_ratio
  (d : ℝ) -- distance between starting points
  (s : ℝ) -- time to meet when running towards each other
  (u : ℝ) -- time for faster runner to catch up when running in same direction
  (h_d : d > 0)
  (h_s : s > 0)
  (h_u : u > 0) :
  ∃ (v_f v_s : ℝ), v_f > v_s ∧ v_f / v_s = 2 ∧
    v_f + v_s = d / s ∧
    (v_f - v_s) * u = v_s * u :=
by sorry

end NUMINAMATH_CALUDE_runners_speed_ratio_l3741_374198


namespace NUMINAMATH_CALUDE_lcm_hcf_relation_l3741_374176

theorem lcm_hcf_relation (d c : ℕ) (h1 : d > 0) (h2 : Nat.lcm 76 d = 456) (h3 : Nat.gcd 76 d = c) : d = 24 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_relation_l3741_374176


namespace NUMINAMATH_CALUDE_scrabble_champions_years_l3741_374114

theorem scrabble_champions_years (total_champions : ℕ) 
  (women_percent : ℚ) (men_with_beard_percent : ℚ) (men_with_beard : ℕ) : 
  women_percent = 3/5 →
  men_with_beard_percent = 2/5 →
  men_with_beard = 4 →
  total_champions = 25 := by
sorry

end NUMINAMATH_CALUDE_scrabble_champions_years_l3741_374114


namespace NUMINAMATH_CALUDE_siblings_comparison_l3741_374125

/-- Given information about siblings of Masud, Janet, Carlos, and Stella, prove that Janet has 16 fewer siblings than Carlos and Stella combined. -/
theorem siblings_comparison (masud janet carlos stella : ℕ) : 
  masud = 40 →
  janet = 4 * masud - 60 →
  carlos = 3 * masud / 4 + 12 →
  stella = 2 * (carlos - 12) - 8 →
  janet = 100 →
  carlos = 64 →
  stella = 52 →
  janet = carlos + stella - 16 := by
  sorry


end NUMINAMATH_CALUDE_siblings_comparison_l3741_374125


namespace NUMINAMATH_CALUDE_parabola_tangent_secant_relation_l3741_374111

/-- A parabola with its axis parallel to the y-axis -/
structure Parabola where
  a : ℝ
  f : ℝ → ℝ
  f_eq : f = fun x ↦ a * x^2

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.f x

/-- Tangent of the angle of inclination of the tangent at a point -/
def tangentSlope (p : Parabola) (point : PointOnParabola p) : ℝ :=
  2 * p.a * point.x

/-- Tangent of the angle of inclination of the secant line between two points -/
def secantSlope (p : Parabola) (p1 p2 : PointOnParabola p) : ℝ :=
  p.a * (p1.x + p2.x)

/-- The main theorem -/
theorem parabola_tangent_secant_relation (p : Parabola) 
    (A1 A2 A3 : PointOnParabola p) : 
    tangentSlope p A1 = secantSlope p A1 A2 + secantSlope p A1 A3 - secantSlope p A2 A3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_secant_relation_l3741_374111


namespace NUMINAMATH_CALUDE_range_of_sine_function_l3741_374105

open Set
open Real

theorem range_of_sine_function (x : ℝ) (h : 0 < x ∧ x < 2*π/3) :
  ∃ y, y ∈ Ioo 0 1 ∧ y = 2 * sin (x + π/6) - 1 ∧
  ∀ z, z = 2 * sin (x + π/6) - 1 → z ∈ Ioc 0 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_sine_function_l3741_374105


namespace NUMINAMATH_CALUDE_expand_product_l3741_374156

theorem expand_product (x : ℝ) : (x + 3) * (x^2 + 4*x + 6) = x^3 + 7*x^2 + 18*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3741_374156
