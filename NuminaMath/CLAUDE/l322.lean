import Mathlib

namespace modulus_of_complex_l322_32242

def i : ℂ := Complex.I

theorem modulus_of_complex (z : ℂ) : z = (2 + i) / (1 - i) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end modulus_of_complex_l322_32242


namespace boys_walking_speed_l322_32243

/-- 
Given two boys walking in the same direction for 7 hours, with one boy walking at 5.5 km/h 
and ending up 10.5 km apart, prove that the speed of the other boy is 7 km/h.
-/
theorem boys_walking_speed 
  (time : ℝ) 
  (distance_apart : ℝ) 
  (speed_second_boy : ℝ) 
  (speed_first_boy : ℝ) 
  (h1 : time = 7) 
  (h2 : distance_apart = 10.5) 
  (h3 : speed_second_boy = 5.5) 
  (h4 : distance_apart = (speed_first_boy - speed_second_boy) * time) : 
  speed_first_boy = 7 := by
  sorry

end boys_walking_speed_l322_32243


namespace line_moved_down_l322_32244

/-- Given a line y = -x + 1 moved down 3 units, prove that the resulting line is y = -x - 2 -/
theorem line_moved_down (x y : ℝ) :
  (y = -x + 1) → (y - 3 = -x - 2) := by
  sorry

end line_moved_down_l322_32244


namespace cone_sphere_volume_equality_implies_lateral_area_l322_32273

/-- Given a cone with base radius 1 and a sphere with radius 1, if their volumes are equal,
    then the lateral surface area of the cone is √17π. -/
theorem cone_sphere_volume_equality_implies_lateral_area (π : ℝ) (h : ℝ) :
  (1/3 : ℝ) * π * 1^2 * h = (4/3 : ℝ) * π * 1^3 →
  π * 1 * (1^2 + h^2).sqrt = π * Real.sqrt 17 := by sorry

end cone_sphere_volume_equality_implies_lateral_area_l322_32273


namespace no_nontrivial_integer_solution_l322_32284

theorem no_nontrivial_integer_solution (a b c d : ℤ) :
  6 * (6 * a ^ 2 + 3 * b ^ 2 + c ^ 2) = 5 * d ^ 2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end no_nontrivial_integer_solution_l322_32284


namespace complement_A_union_B_eq_B_l322_32285

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x > 1}
def B : Set ℝ := {x | x ≥ -1}

-- State the theorem
theorem complement_A_union_B_eq_B : (Aᶜ ∪ B) = B := by sorry

end complement_A_union_B_eq_B_l322_32285


namespace tan_product_simplification_l322_32252

theorem tan_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end tan_product_simplification_l322_32252


namespace sqrt_three_squared_five_fourth_l322_32219

theorem sqrt_three_squared_five_fourth (x : ℝ) : 
  x = Real.sqrt (3^2 * 5^4) → x = 75 := by
  sorry

end sqrt_three_squared_five_fourth_l322_32219


namespace equal_sampling_most_representative_l322_32235

/-- Represents a school in the survey --/
inductive School
| A
| B
| C
| D

/-- Represents a survey method --/
structure SurveyMethod where
  schools : List School
  studentsPerSchool : Nat

/-- Defines the representativeness of a survey method --/
def representativeness (method : SurveyMethod) : ℝ :=
  sorry

/-- The number of schools in the survey --/
def totalSchools : Nat := 4

/-- The survey method that samples from all schools equally --/
def equalSamplingMethod : SurveyMethod :=
  { schools := [School.A, School.B, School.C, School.D],
    studentsPerSchool := 150 }

/-- Theorem stating that the equal sampling method is the most representative --/
theorem equal_sampling_most_representative :
  ∀ (method : SurveyMethod),
    method.schools.length = totalSchools →
    representativeness equalSamplingMethod ≥ representativeness method :=
  sorry

end equal_sampling_most_representative_l322_32235


namespace quadratic_rational_root_even_coefficient_l322_32214

theorem quadratic_rational_root_even_coefficient
  (a b c : ℤ) (h_a : a ≠ 0)
  (h_root : ∃ (x : ℚ), a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end quadratic_rational_root_even_coefficient_l322_32214


namespace nested_expression_value_l322_32297

theorem nested_expression_value : (3*(3*(3*(3*(3*(3+2)+2)+2)+2)+2)+2) = 1457 := by
  sorry

end nested_expression_value_l322_32297


namespace latus_rectum_of_parabola_l322_32226

/-- The latus rectum of a parabola x^2 = -2y is y = 1/2 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  x^2 = -2*y → (∃ (x₀ : ℝ), x₀^2 = -2*(1/2) ∧ x₀ ≠ 0) :=
by sorry

end latus_rectum_of_parabola_l322_32226


namespace no_solution_sqrt_eq_negative_l322_32278

theorem no_solution_sqrt_eq_negative :
  ¬∃ x : ℝ, Real.sqrt (5 - x) = -3 := by
  sorry

end no_solution_sqrt_eq_negative_l322_32278


namespace proportion_third_term_l322_32251

/-- Given a proportion 0.75 : 1.65 :: y : 11, prove that y = 5 -/
theorem proportion_third_term (y : ℝ) : 
  (0.75 : ℝ) / 1.65 = y / 11 → y = 5 := by
  sorry

end proportion_third_term_l322_32251


namespace productivity_increase_l322_32229

/-- Represents the productivity level during a work shift -/
structure Productivity where
  planned : ℝ
  reduced : ℝ

/-- Represents a work shift with its duration and productivity levels -/
structure Shift where
  duration : ℝ
  plannedHours : ℝ
  productivity : Productivity

/-- Calculates the total work done during a shift -/
def totalWork (s : Shift) : ℝ :=
  s.plannedHours * s.productivity.planned +
  (s.duration - s.plannedHours) * s.productivity.reduced

/-- Theorem stating the productivity increase when extending the workday -/
theorem productivity_increase
  (initialShift : Shift)
  (extendedShift : Shift)
  (h1 : initialShift.duration = 8)
  (h2 : initialShift.plannedHours = 6)
  (h3 : initialShift.productivity.planned = 1)
  (h4 : initialShift.productivity.reduced = 0.75)
  (h5 : extendedShift.duration = 9)
  (h6 : extendedShift.plannedHours = 6)
  (h7 : extendedShift.productivity.planned = 1)
  (h8 : extendedShift.productivity.reduced = 0.7) :
  (totalWork extendedShift - totalWork initialShift) / totalWork initialShift = 0.08 := by
  sorry


end productivity_increase_l322_32229


namespace power_division_equality_l322_32270

theorem power_division_equality (a : ℝ) : (-2 * a^2)^3 / (2 * a^2) = -4 * a^4 := by
  sorry

end power_division_equality_l322_32270


namespace pentagon_smallest_angle_l322_32215

theorem pentagon_smallest_angle 
  (angles : Fin 5 → ℝ)
  (arithmetic_sequence : ∀ i : Fin 4, angles (i + 1) - angles i = angles (i + 2) - angles (i + 1))
  (largest_angle : angles 4 = 150)
  (angle_sum : angles 0 + angles 1 + angles 2 + angles 3 + angles 4 = 540) :
  angles 0 = 66 := by
sorry

end pentagon_smallest_angle_l322_32215


namespace fourth_term_of_specific_sequence_l322_32241

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℝ
  common_ratio : ℝ

/-- The nth term of a geometric sequence -/
def nth_term (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem fourth_term_of_specific_sequence :
  ∃ (seq : GeometricSequence),
    seq.first_term = 512 ∧
    nth_term seq 6 = 32 ∧
    nth_term seq 4 = 64 :=
by
  sorry

end fourth_term_of_specific_sequence_l322_32241


namespace emily_card_collection_l322_32228

/-- Emily's card collection problem -/
theorem emily_card_collection (initial_cards : ℕ) (additional_cards : ℕ) :
  initial_cards = 63 → additional_cards = 7 → initial_cards + additional_cards = 70 := by
  sorry

end emily_card_collection_l322_32228


namespace ten_thousandths_digit_of_seven_thirty_seconds_l322_32262

theorem ten_thousandths_digit_of_seven_thirty_seconds (f : ℚ) (d : ℕ) : 
  f = 7 / 32 →
  d = (⌊f * 10000⌋ % 10) →
  d = 8 :=
by sorry

end ten_thousandths_digit_of_seven_thirty_seconds_l322_32262


namespace f_value_at_3_l322_32233

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 3

-- State the theorem
theorem f_value_at_3 (a b c : ℝ) : f a b c (-3) = 7 → f a b c 3 = -13 := by
  sorry

end f_value_at_3_l322_32233


namespace batsman_110_run_inning_l322_32246

/-- Represents a batsman's scoring history -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  deriving Repr

/-- Calculate the average score of a batsman -/
def average (b : Batsman) : ℚ :=
  b.totalRuns / b.innings

/-- The inning where the batsman scores 110 runs -/
def scoreInning (b : Batsman) : ℕ :=
  b.innings + 1

theorem batsman_110_run_inning (b : Batsman) 
  (h1 : average (⟨b.innings + 1, b.totalRuns + 110⟩ : Batsman) = 60)
  (h2 : average (⟨b.innings + 1, b.totalRuns + 110⟩ : Batsman) - average b = 5) :
  scoreInning b = 11 := by
  sorry

#eval scoreInning ⟨10, 550⟩

end batsman_110_run_inning_l322_32246


namespace possible_values_of_a_l322_32286

theorem possible_values_of_a (a b c : ℝ) 
  (eq1 : a * b + a + b = c)
  (eq2 : b * c + b + c = a)
  (eq3 : c * a + c + a = b) :
  a = 0 ∨ a = -1 ∨ a = -2 := by
  sorry

end possible_values_of_a_l322_32286


namespace least_five_digit_square_cube_l322_32257

theorem least_five_digit_square_cube : 
  (∀ n : ℕ, n < 15625 → ¬(∃ a b : ℕ, n = a^2 ∧ n = b^3 ∧ n ≥ 10000)) ∧ 
  (∃ a b : ℕ, 15625 = a^2 ∧ 15625 = b^3) ∧ 
  15625 ≥ 10000 :=
sorry

end least_five_digit_square_cube_l322_32257


namespace mildreds_oranges_l322_32200

/-- The number of oranges Mildred's father gave her -/
def oranges_given (initial final : ℕ) : ℕ := final - initial

theorem mildreds_oranges : oranges_given 77 79 = 2 := by
  sorry

end mildreds_oranges_l322_32200


namespace smaller_solution_quadratic_l322_32269

theorem smaller_solution_quadratic (x : ℝ) : 
  x^2 + 9*x - 22 = 0 ∧ (∀ y : ℝ, y^2 + 9*y - 22 = 0 → x ≤ y) → x = -11 :=
by sorry

end smaller_solution_quadratic_l322_32269


namespace tenth_term_of_sequence_l322_32265

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 2/3) (h₃ : a₃ = 5/6) :
  arithmetic_sequence a₁ (a₂ - a₁) 10 = 2 := by
sorry

end tenth_term_of_sequence_l322_32265


namespace largest_four_digit_congruent_to_17_mod_28_l322_32276

theorem largest_four_digit_congruent_to_17_mod_28 :
  ∃ (n : ℕ), n = 9982 ∧ n < 10000 ∧ n ≡ 17 [MOD 28] ∧
  ∀ (m : ℕ), m < 10000 ∧ m ≡ 17 [MOD 28] → m ≤ n :=
by sorry

end largest_four_digit_congruent_to_17_mod_28_l322_32276


namespace geometric_sequence_ratio_l322_32240

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def has_four_consecutive_terms (b : ℕ → ℝ) (S : Set ℝ) : Prop :=
  ∃ k : ℕ, (b k ∈ S) ∧ (b (k + 1) ∈ S) ∧ (b (k + 2) ∈ S) ∧ (b (k + 3) ∈ S)

theorem geometric_sequence_ratio (a b : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  (∀ n : ℕ, b n = a n + 1) →
  has_four_consecutive_terms b {-53, -23, 19, 37, 82} →
  abs q > 1 →
  q = -3/2 :=
sorry

end geometric_sequence_ratio_l322_32240


namespace largest_angle_hexagon_l322_32205

/-- The largest interior angle of a convex hexagon with six consecutive integer angles -/
def largest_hexagon_angle : ℝ := 122.5

/-- A hexagon with six consecutive integer angles -/
structure ConsecutiveAngleHexagon where
  -- The smallest angle of the hexagon
  base_angle : ℝ
  -- Predicate ensuring the angles are consecutive integers
  consecutive_integers : ∀ i : Fin 6, (base_angle + i) = ↑(⌊base_angle⌋ + i)

/-- Theorem stating that the largest angle in a convex hexagon with six consecutive integer angles is 122.5° -/
theorem largest_angle_hexagon (h : ConsecutiveAngleHexagon) : 
  (h.base_angle + 5) = largest_hexagon_angle := by
  sorry

/-- The sum of interior angles of a hexagon is 720° -/
axiom sum_hexagon_angles : ∀ (h : ConsecutiveAngleHexagon), 
  (h.base_angle * 6 + 15) = 720

end largest_angle_hexagon_l322_32205


namespace min_ear_sightings_l322_32279

/-- Represents the direction a child is facing -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the grid of children -/
def Grid (n : Nat) := Position → Direction

/-- Counts the number of children seeing an ear in the given grid -/
def countEarSightings (n : Nat) (grid : Grid n) : Nat :=
  sorry

/-- Theorem stating the minimal number of children seeing an ear -/
theorem min_ear_sightings (n : Nat) :
  (∃ (grid : Grid n), countEarSightings n grid = n + 2) ∧
  (∀ (grid : Grid n), countEarSightings n grid ≥ n + 2) :=
sorry

end min_ear_sightings_l322_32279


namespace unique_solution_l322_32296

def original_number : Nat := 20222023

theorem unique_solution (n : Nat) :
  (n ≥ 1000000000 ∧ n < 10000000000) ∧  -- 10-digit number
  (∃ (a b : Nat), n = a * 1000000000 + original_number * 10 + b) ∧  -- Formed by adding digits to left and right
  (n % 72 = 0) →  -- Divisible by 72
  n = 3202220232 :=
sorry

end unique_solution_l322_32296


namespace circle_passes_through_fixed_points_l322_32287

/-- Quadratic function f(x) = 3x^2 - 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 4 * x + c

/-- Circle equation: x^2 + y^2 + Dx + Ey + F = 0 -/
def circle_equation (D E F x y : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

/-- Theorem: The circle passing through the intersection points of f(x) with the axes
    also passes through the fixed points (0, 1/3) and (4/3, 1/3) -/
theorem circle_passes_through_fixed_points (c : ℝ) 
  (h1 : 0 < c) (h2 : c < 4/3) : 
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, f c x = 0 ∧ y = 0 → circle_equation D E F x y) ∧ 
    (∀ x y : ℝ, x = 0 ∧ f c 0 = y → circle_equation D E F x y) ∧
    circle_equation D E F 0 (1/3) ∧ 
    circle_equation D E F (4/3) (1/3) := by
  sorry

end circle_passes_through_fixed_points_l322_32287


namespace first_sampling_immediate_l322_32298

/-- Represents the stages of the yeast population experiment -/
inductive ExperimentStage
  | Inoculation
  | Sampling
  | Counting

/-- Represents the timing of the first sampling test -/
inductive SamplingTiming
  | Immediate
  | Delayed

/-- The correct procedure for the yeast population experiment -/
def correctYeastExperimentProcedure : ExperimentStage → SamplingTiming
  | ExperimentStage.Inoculation => SamplingTiming.Immediate
  | _ => SamplingTiming.Delayed

/-- Theorem stating that the first sampling test should be conducted immediately after inoculation -/
theorem first_sampling_immediate :
  correctYeastExperimentProcedure ExperimentStage.Inoculation = SamplingTiming.Immediate :=
by sorry

end first_sampling_immediate_l322_32298


namespace quadratic_sum_l322_32218

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, 5 * x^2 - 30 * x - 45 = a * (x + b)^2 + c) → 
  a + b + c = -88 := by
sorry

end quadratic_sum_l322_32218


namespace intersection_product_l322_32295

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a parabola -/
structure Parabola where
  focus : Point

/-- Checks if a point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Checks if a point is on the parabola -/
def on_parabola (p : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 4 * p.focus.x * pt.x

/-- Theorem statement -/
theorem intersection_product (e : Ellipse) (p : Parabola) 
  (A B P : Point) (h_A : on_ellipse e A ∧ on_parabola p A)
  (h_B : on_ellipse e B ∧ on_parabola p B)
  (h_P : on_ellipse e P)
  (h_quad : A.y > 0 ∧ B.y < 0)
  (h_focus : p.focus.y = 0 ∧ p.focus.x > 0)
  (h_vertex : p.focus.x = e.a^2 / (4 * e.b^2))
  (M N : ℝ) (h_M : ∃ t, A.x + t * (P.x - A.x) = M ∧ A.y + t * (P.y - A.y) = 0)
  (h_N : ∃ t, B.x + t * (P.x - B.x) = N ∧ B.y + t * (P.y - B.y) = 0) :
  M * N = e.a^2 := by
  sorry

end intersection_product_l322_32295


namespace hannah_dog_food_l322_32271

/-- The amount of dog food Hannah needs to prepare for her three dogs in a day -/
def total_dog_food (first_dog_food : ℝ) (second_dog_multiplier : ℝ) (third_dog_extra : ℝ) : ℝ :=
  first_dog_food + 
  (first_dog_food * second_dog_multiplier) + 
  (first_dog_food * second_dog_multiplier + third_dog_extra)

/-- Theorem stating that Hannah needs to prepare 10 cups of dog food for her three dogs -/
theorem hannah_dog_food : total_dog_food 1.5 2 2.5 = 10 := by
  sorry

end hannah_dog_food_l322_32271


namespace rotated_square_height_l322_32237

theorem rotated_square_height (square_side : Real) (rotation_angle : Real) : 
  square_side = 2 ∧ rotation_angle = π / 6 →
  let diagonal := square_side * Real.sqrt 2
  let height_above_center := (diagonal / 2) * Real.sin rotation_angle
  let initial_center_height := square_side / 2
  initial_center_height + height_above_center = 1 + Real.sqrt 2 / 2 := by
sorry

end rotated_square_height_l322_32237


namespace axis_of_symmetry_l322_32231

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem axis_of_symmetry 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 ≤ φ ∧ φ < Real.pi) 
  (h_even : ∀ x, f ω φ x = f ω φ (-x)) 
  (h_distance : ∃ (a b : ℝ), b - a = 4 * Real.sqrt 2 ∧ f ω φ b = f ω φ a) :
  ∃ (x : ℝ), x = 4 ∧ ∀ y, f ω φ (x + y) = f ω φ (x - y) :=
sorry

end axis_of_symmetry_l322_32231


namespace daughter_weight_l322_32239

/-- Represents the weights of family members -/
structure FamilyWeights where
  grandmother : ℝ
  daughter : ℝ
  child : ℝ

/-- The conditions of the family weight problem -/
def FamilyWeightProblem (w : FamilyWeights) : Prop :=
  w.grandmother + w.daughter + w.child = 110 ∧
  w.daughter + w.child = 60 ∧
  w.child = (1 / 5) * w.grandmother

/-- The theorem stating that given the conditions, the daughter's weight is 50 kg -/
theorem daughter_weight (w : FamilyWeights) : 
  FamilyWeightProblem w → w.daughter = 50 := by
  sorry


end daughter_weight_l322_32239


namespace power_of_power_l322_32208

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by sorry

end power_of_power_l322_32208


namespace new_shoes_cost_proof_l322_32204

/-- The cost of repairing used shoes -/
def repair_cost : ℝ := 10.50

/-- The duration (in years) that repaired used shoes last -/
def repair_duration : ℝ := 1

/-- The duration (in years) that new shoes last -/
def new_duration : ℝ := 2

/-- The percentage increase in average cost per year of new shoes compared to repaired shoes -/
def cost_increase_percentage : ℝ := 42.857142857142854

/-- The cost of purchasing new shoes -/
def new_shoes_cost : ℝ := 30.00

/-- Theorem stating that the cost of new shoes is $30.00 given the problem conditions -/
theorem new_shoes_cost_proof :
  new_shoes_cost = (repair_cost / repair_duration + cost_increase_percentage / 100 * repair_cost) * new_duration :=
by sorry

end new_shoes_cost_proof_l322_32204


namespace starting_lineup_combinations_l322_32292

def total_players : ℕ := 15
def lineup_size : ℕ := 6
def pre_selected_players : ℕ := 2

theorem starting_lineup_combinations :
  Nat.choose (total_players - pre_selected_players) (lineup_size - pre_selected_players) = 715 := by
  sorry

end starting_lineup_combinations_l322_32292


namespace fence_cost_circular_plot_l322_32220

/-- The cost of building a fence around a circular plot -/
theorem fence_cost_circular_plot (area : ℝ) (price_per_foot : ℝ) : 
  area = 289 → price_per_foot = 58 → 
  (2 * Real.sqrt area * price_per_foot : ℝ) = 1972 := by
  sorry

#check fence_cost_circular_plot

end fence_cost_circular_plot_l322_32220


namespace rectangle_dimension_change_l322_32232

theorem rectangle_dimension_change (original_length original_width : ℝ) 
  (new_length new_width : ℝ) (h_positive : original_length > 0 ∧ original_width > 0) :
  new_width = 1.5 * original_width ∧ 
  original_length * original_width = new_length * new_width →
  (original_length - new_length) / original_length = 1 / 3 := by
sorry

end rectangle_dimension_change_l322_32232


namespace regular_polygon_exterior_angle_l322_32201

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 72 → n = 5 := by
  sorry

end regular_polygon_exterior_angle_l322_32201


namespace seven_day_payment_possible_l322_32248

/-- Represents the state of rings at any given time --/
structure RingState :=
  (single : ℕ)    -- number of single rings
  (double : ℕ)    -- number of chains with 2 rings
  (quadruple : ℕ) -- number of chains with 4 rings

/-- Represents a daily transaction --/
inductive Transaction
  | give_single
  | give_double
  | give_quadruple
  | return_single
  | return_double

/-- Applies a transaction to a RingState --/
def apply_transaction (state : RingState) (t : Transaction) : RingState :=
  match t with
  | Transaction.give_single => ⟨state.single - 1, state.double, state.quadruple⟩
  | Transaction.give_double => ⟨state.single, state.double - 1, state.quadruple⟩
  | Transaction.give_quadruple => ⟨state.single, state.double, state.quadruple - 1⟩
  | Transaction.return_single => ⟨state.single + 1, state.double, state.quadruple⟩
  | Transaction.return_double => ⟨state.single, state.double + 1, state.quadruple⟩

/-- Checks if a sequence of transactions is valid for a given initial state --/
def is_valid_sequence (initial : RingState) (transactions : List Transaction) : Prop :=
  ∀ (n : ℕ), n < transactions.length →
    let state := transactions.take n.succ
      |> List.foldl apply_transaction initial
    state.single ≥ 0 ∧ state.double ≥ 0 ∧ state.quadruple ≥ 0

/-- Checks if a sequence of transactions results in a net payment of one ring per day --/
def is_daily_payment (transactions : List Transaction) : Prop :=
  transactions.foldl (λ acc t =>
    match t with
    | Transaction.give_single => acc + 1
    | Transaction.give_double => acc + 2
    | Transaction.give_quadruple => acc + 4
    | Transaction.return_single => acc - 1
    | Transaction.return_double => acc - 2
  ) 0 = 1

/-- The main theorem: it is possible to pay for 7 days using a chain of 7 rings, cutting only one --/
theorem seven_day_payment_possible : ∃ (transactions : List Transaction),
  transactions.length = 7 ∧
  is_valid_sequence ⟨1, 1, 1⟩ transactions ∧
  (∀ (n : ℕ), n < 7 → is_daily_payment (transactions.take (n + 1))) :=
sorry

end seven_day_payment_possible_l322_32248


namespace specific_value_problem_l322_32299

theorem specific_value_problem (x : ℕ) (specific_value : ℕ) 
  (h1 : 25 * x = specific_value) 
  (h2 : x = 27) : 
  specific_value = 675 := by
sorry

end specific_value_problem_l322_32299


namespace square_difference_l322_32283

theorem square_difference (x y k c : ℝ) 
  (h1 : x * y = k) 
  (h2 : 1 / x^2 + 1 / y^2 = c) : 
  (x - y)^2 = c * k^2 - 2 * k := by
sorry

end square_difference_l322_32283


namespace intersection_points_on_line_l322_32294

/-- The slope of the line containing all intersection points of the given parametric lines -/
def intersection_line_slope : ℚ := 10/31

/-- The first line equation: 2x + 3y = 8u + 4 -/
def line1 (u x y : ℝ) : Prop := 2*x + 3*y = 8*u + 4

/-- The second line equation: 3x - 2y = 5u - 3 -/
def line2 (u x y : ℝ) : Prop := 3*x - 2*y = 5*u - 3

/-- The theorem stating that all intersection points lie on a line with slope 10/31 -/
theorem intersection_points_on_line :
  ∀ (u x y : ℝ), line1 u x y → line2 u x y →
  ∃ (k b : ℝ), y = intersection_line_slope * x + b :=
sorry

end intersection_points_on_line_l322_32294


namespace hot_dogs_remainder_l322_32202

theorem hot_dogs_remainder : 25197638 % 4 = 2 := by
  sorry

end hot_dogs_remainder_l322_32202


namespace proof_by_contradiction_method_l322_32207

-- Define what proof by contradiction means
def proof_by_contradiction (P : Prop) : Prop :=
  ∃ (proof : ¬P → False), P

-- State the theorem
theorem proof_by_contradiction_method :
  ¬(∀ (P Q : Prop), proof_by_contradiction P ↔ (¬P ∧ ¬Q → False)) :=
sorry

end proof_by_contradiction_method_l322_32207


namespace simultaneous_equations_solution_l322_32238

/-- The simultaneous equations y = kx + 5 and y = (3k - 2)x + 6 have at least one solution
    in terms of real numbers (x, y) if and only if k ≠ 1 -/
theorem simultaneous_equations_solution (k : ℝ) :
  (∃ x y : ℝ, y = k * x + 5 ∧ y = (3 * k - 2) * x + 6) ↔ k ≠ 1 := by
  sorry

end simultaneous_equations_solution_l322_32238


namespace fractional_equation_solution_l322_32213

theorem fractional_equation_solution :
  ∀ x : ℝ, (4 / x = 2 / (x + 1)) ↔ (x = -2) :=
by sorry

end fractional_equation_solution_l322_32213


namespace initial_candies_count_l322_32277

/-- The number of candies initially in the box -/
def initial_candies : ℕ := sorry

/-- The number of candies Diana took from the box -/
def candies_taken : ℕ := 6

/-- The number of candies left in the box after Diana took some -/
def candies_left : ℕ := 82

/-- Theorem stating that the initial number of candies is 88 -/
theorem initial_candies_count : initial_candies = 88 :=
  by sorry

end initial_candies_count_l322_32277


namespace pizza_consumption_order_l322_32261

def pizza_shares (total : ℚ) (ali : ℚ) (bea : ℚ) (chris : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  let dan := total - (ali + bea + chris)
  (dan, ali, chris, bea)

theorem pizza_consumption_order (total : ℚ) :
  let (dan, ali, chris, bea) := pizza_shares total (1/6) (1/8) (1/7)
  dan > ali ∧ ali > chris ∧ chris > bea := by
  sorry

#check pizza_consumption_order

end pizza_consumption_order_l322_32261


namespace function_value_symmetry_l322_32291

/-- Given a function f(x) = ax^7 + bx - 2 where f(2008) = 10, prove that f(-2008) = -12 -/
theorem function_value_symmetry (a b : ℝ) :
  let f := λ x : ℝ => a * x^7 + b * x - 2
  f 2008 = 10 → f (-2008) = -12 := by
sorry

end function_value_symmetry_l322_32291


namespace f_is_h_function_l322_32249

def is_h_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x₁ x₂, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁)

def f (x : ℝ) : ℝ := x * |x|

theorem f_is_h_function : is_h_function f := by sorry

end f_is_h_function_l322_32249


namespace speed_conversion_proof_l322_32259

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℚ := 3.6

/-- Given speed in km/h -/
def given_speed_kmph : ℝ := 1.5428571428571427

/-- Speed in m/s as a fraction -/
def speed_mps : ℚ := 3/7

theorem speed_conversion_proof :
  (speed_mps : ℝ) * mps_to_kmph = given_speed_kmph := by
  sorry

end speed_conversion_proof_l322_32259


namespace bouncy_balls_shipment_l322_32272

theorem bouncy_balls_shipment (displayed_percentage : ℚ) (warehouse_count : ℕ) : 
  displayed_percentage = 1/4 →
  warehouse_count = 90 →
  ∃ total : ℕ, total = 120 ∧ (1 - displayed_percentage) * total = warehouse_count :=
by sorry

end bouncy_balls_shipment_l322_32272


namespace consecutive_numbers_probability_l322_32255

def set_size : ℕ := 20
def selection_size : ℕ := 5

def prob_consecutive_numbers : ℚ :=
  1 - (Nat.choose (set_size - selection_size + 1) selection_size : ℚ) / (Nat.choose set_size selection_size : ℚ)

theorem consecutive_numbers_probability :
  prob_consecutive_numbers = 232 / 323 := by sorry

end consecutive_numbers_probability_l322_32255


namespace expression_simplification_l322_32260

theorem expression_simplification (x y b c d : ℝ) (h : c * y + d * x ≠ 0) :
  (c * y * (b * x^3 + 3 * b * x^2 * y + 3 * b * x * y^2 + b * y^3) + 
   d * x * (c * x^3 + 3 * c * x^2 * y + 3 * c * x * y^2 + c * y^3)) / 
  (c * y + d * x) = (c * x + y)^3 := by
  sorry

end expression_simplification_l322_32260


namespace greatest_n_with_perfect_square_property_l322_32268

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k^2

theorem greatest_n_with_perfect_square_property :
  ∃ (n : ℕ), n = 1921 ∧ n ≤ 2008 ∧
  (∀ m : ℕ, m ≤ 2008 → m > n →
    ¬ is_perfect_square ((sum_of_squares n) * (sum_of_squares (2 * n) - sum_of_squares n))) ∧
  is_perfect_square ((sum_of_squares n) * (sum_of_squares (2 * n) - sum_of_squares n)) :=
sorry

end greatest_n_with_perfect_square_property_l322_32268


namespace horner_rule_evaluation_l322_32254

def horner_polynomial (x : ℝ) : ℝ :=
  (((((2 * x - 0) * x - 3) * x + 2) * x + 7) * x + 6) * x + 3

theorem horner_rule_evaluation :
  horner_polynomial 2 = 5 := by
  sorry

end horner_rule_evaluation_l322_32254


namespace uniform_scores_smaller_variance_l322_32281

/-- Class scores data -/
structure ClassScores where
  mean : ℝ
  variance : ℝ

/-- Uniformity of scores -/
def more_uniform (a b : ClassScores) : Prop :=
  a.variance < b.variance

/-- Theorem: Class with smaller variance has more uniform scores -/
theorem uniform_scores_smaller_variance 
  (class_a class_b : ClassScores) 
  (h_mean : class_a.mean = class_b.mean) 
  (h_var : class_a.variance > class_b.variance) : 
  more_uniform class_b class_a :=
by sorry

end uniform_scores_smaller_variance_l322_32281


namespace smallest_n_sum_squares_over_n_is_square_l322_32274

/-- Sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Predicate to check if a number is a perfect square -/
def is_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

/-- Predicate to check if the sum of squares divided by n is a square -/
def is_sum_of_squares_over_n_square (n : ℕ) : Prop :=
  is_square (sum_of_squares n / n)

theorem smallest_n_sum_squares_over_n_is_square :
  (∀ m : ℕ, m > 1 ∧ m < 337 → ¬is_sum_of_squares_over_n_square m) ∧
  is_sum_of_squares_over_n_square 337 := by
  sorry

end smallest_n_sum_squares_over_n_is_square_l322_32274


namespace polygon_interior_angles_sum_l322_32280

theorem polygon_interior_angles_sum (n : ℕ) (sum : ℝ) : 
  sum = 900 → (n - 2) * 180 = sum → n = 7 :=
by sorry

end polygon_interior_angles_sum_l322_32280


namespace irregular_shape_area_l322_32290

/-- The area of an irregular shape consisting of a rectangle connected to a semi-circle -/
theorem irregular_shape_area (square_area : ℝ) (rect_length : ℝ) : 
  square_area = 2025 →
  rect_length = 10 →
  let circle_radius := Real.sqrt square_area
  let rect_breadth := (3 / 5) * circle_radius
  let rect_area := rect_length * rect_breadth
  let semicircle_area := (1 / 2) * Real.pi * circle_radius ^ 2
  rect_area + semicircle_area = 270 + 1012.5 * Real.pi :=
by sorry

end irregular_shape_area_l322_32290


namespace root_product_equals_21_l322_32209

theorem root_product_equals_21 (x₁ x₂ x₃ : ℝ) :
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x, Real.sqrt 100 * x^3 - 210 * x^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 21 := by
sorry

end root_product_equals_21_l322_32209


namespace max_students_is_eight_l322_32288

def knows (n : ℕ) : (Fin n → Fin n → Prop) → Prop :=
  λ f => ∀ (i j : Fin n), i ≠ j → f i j = f j i

def satisfies_conditions (n : ℕ) (f : Fin n → Fin n → Prop) : Prop :=
  knows n f ∧
  (∀ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    f a b ∨ f b c ∨ f a c) ∧
  (∀ (a b c d : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d → 
    (¬f a b ∧ ¬f c d) ∨ (¬f a c ∧ ¬f b d) ∨ (¬f a d ∧ ¬f b c))

theorem max_students_is_eight :
  (∃ (f : Fin 8 → Fin 8 → Prop), satisfies_conditions 8 f) ∧
  (∀ n > 8, ¬∃ (f : Fin n → Fin n → Prop), satisfies_conditions n f) :=
sorry

end max_students_is_eight_l322_32288


namespace division_result_l322_32236

theorem division_result : (210 : ℚ) / (15 + 12 * 3 - 6) = 14 / 3 := by
  sorry

end division_result_l322_32236


namespace angle_F_is_60_l322_32206

/-- A trapezoid with specific angle relationships -/
structure SpecialTrapezoid where
  -- Angles of the trapezoid
  angleE : ℝ
  angleF : ℝ
  angleG : ℝ
  angleH : ℝ
  -- Conditions given in the problem
  parallel_sides : True  -- Represents that EF and GH are parallel
  angle_E_triple_H : angleE = 3 * angleH
  angle_G_double_F : angleG = 2 * angleF
  -- Properties of a trapezoid
  sum_angles : angleE + angleF + angleG + angleH = 360
  opposite_angles_sum : angleF + angleG = 180

/-- Theorem stating that in the special trapezoid, angle F measures 60 degrees -/
theorem angle_F_is_60 (t : SpecialTrapezoid) : t.angleF = 60 := by
  sorry

end angle_F_is_60_l322_32206


namespace isosceles_triangle_perimeter_l322_32217

/-- A triangle with two equal sides and side lengths of 3 and 5 has a perimeter of either 11 or 13 -/
theorem isosceles_triangle_perimeter : ∀ (a b : ℝ), 
  a = 3 ∧ b = 5 →
  (∃ (p : ℝ), (p = 11 ∨ p = 13) ∧ 
   ((2 * a + b = p ∧ a + a > b) ∨ (2 * b + a = p ∧ b + b > a))) :=
by sorry


end isosceles_triangle_perimeter_l322_32217


namespace painting_discount_l322_32293

theorem painting_discount (x : ℝ) (h1 : x / 5 = 15) : x * (1 - 1/3) = 50 := by
  sorry

end painting_discount_l322_32293


namespace lowest_class_size_class_size_120_lowest_class_size_is_120_l322_32230

theorem lowest_class_size (n : ℕ) : n > 0 ∧ 6 ∣ n ∧ 8 ∣ n ∧ 12 ∣ n ∧ 15 ∣ n → n ≥ 120 := by
  sorry

theorem class_size_120 : 6 ∣ 120 ∧ 8 ∣ 120 ∧ 12 ∣ 120 ∧ 15 ∣ 120 := by
  sorry

theorem lowest_class_size_is_120 : ∃! n : ℕ, n > 0 ∧ 6 ∣ n ∧ 8 ∣ n ∧ 12 ∣ n ∧ 15 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 6 ∣ m ∧ 8 ∣ m ∧ 12 ∣ m ∧ 15 ∣ m) → n ≤ m := by
  sorry

end lowest_class_size_class_size_120_lowest_class_size_is_120_l322_32230


namespace base_faces_area_sum_l322_32289

/-- A pentagonal prism with given surface area and lateral area -/
structure PentagonalPrism where
  surfaceArea : ℝ
  lateralArea : ℝ

/-- Theorem: For a pentagonal prism with surface area 30 and lateral area 25,
    the sum of the areas of the two base faces equals 5 -/
theorem base_faces_area_sum (prism : PentagonalPrism)
    (h1 : prism.surfaceArea = 30)
    (h2 : prism.lateralArea = 25) :
    prism.surfaceArea - prism.lateralArea = 5 := by
  sorry

end base_faces_area_sum_l322_32289


namespace hyperbola_focal_distance_l322_32227

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_focal_distance 
  (P : ℝ × ℝ) 
  (h_on_hyperbola : is_on_hyperbola P.1 P.2) 
  (h_left_distance : distance P left_focus = 3) : 
  distance P right_focus = 9 := by sorry

end hyperbola_focal_distance_l322_32227


namespace sum_of_digits_of_4444_power_4444_l322_32216

-- Define the sum of digits function
def S (n : ℕ) : ℕ := sorry

-- State the theorem
theorem sum_of_digits_of_4444_power_4444 :
  ∃ (S : ℕ → ℕ),
    (∀ n : ℕ, S n % 9 = n % 9) →
    S (S (S (4444^4444))) = 7 := by sorry

end sum_of_digits_of_4444_power_4444_l322_32216


namespace initial_stock_value_l322_32224

/-- Represents the daily change in stock value -/
def daily_change : ℤ := 1

/-- Represents the number of days until the stock reaches $200 -/
def days_to_target : ℕ := 100

/-- Represents the target value of the stock -/
def target_value : ℤ := 200

/-- Theorem stating that the initial stock value is $101 -/
theorem initial_stock_value (V : ℤ) :
  V + (days_to_target - 1) * daily_change = target_value →
  V = 101 := by
  sorry

end initial_stock_value_l322_32224


namespace xy_inequality_l322_32203

theorem xy_inequality (x y : ℝ) (h : x^2 + y^2 - x*y = 1) :
  (-2 : ℝ) ≤ x + y ∧ x + y ≤ 2 ∧ 2/3 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2 := by
  sorry

end xy_inequality_l322_32203


namespace melanie_plums_l322_32250

/-- The number of plums picked by different people and in total -/
structure PlumPicking where
  dan : ℕ
  sally : ℕ
  total : ℕ

/-- The theorem stating how many plums Melanie picked -/
theorem melanie_plums (p : PlumPicking) (h1 : p.dan = 9) (h2 : p.sally = 3) (h3 : p.total = 16) :
  p.total - (p.dan + p.sally) = 4 := by
  sorry

end melanie_plums_l322_32250


namespace allowance_spent_at_toy_store_l322_32282

theorem allowance_spent_at_toy_store 
  (total_allowance : ℚ)
  (arcade_fraction : ℚ)
  (remaining_after_toy_store : ℚ)
  (h1 : total_allowance = 9/4)  -- $2.25 as a fraction
  (h2 : arcade_fraction = 3/5)
  (h3 : remaining_after_toy_store = 3/5)  -- $0.60 as a fraction
  : (total_allowance - arcade_fraction * total_allowance - remaining_after_toy_store) / 
    (total_allowance - arcade_fraction * total_allowance) = 1/3 := by
  sorry

end allowance_spent_at_toy_store_l322_32282


namespace opera_ticket_price_increase_l322_32266

theorem opera_ticket_price_increase (old_price new_price : ℝ) 
  (h1 : old_price = 85)
  (h2 : new_price = 102) : 
  (new_price - old_price) / old_price * 100 = 20 := by
  sorry

end opera_ticket_price_increase_l322_32266


namespace fourth_hexagon_dots_l322_32256

/-- Calculates the number of dots in the nth hexagon of the pattern. -/
def hexagonDots (n : ℕ) : ℕ :=
  if n = 0 then 1
  else hexagonDots (n - 1) + 6 * n

/-- The number of dots in the fourth hexagon is 55. -/
theorem fourth_hexagon_dots : hexagonDots 4 = 55 := by sorry

end fourth_hexagon_dots_l322_32256


namespace total_revenue_is_146475_l322_32264

/-- The number of cookies baked by Clementine -/
def C : ℕ := 72

/-- The number of cookies baked by Jake -/
def J : ℕ := (5 * C) / 2

/-- The number of cookies baked by Tory -/
def T : ℕ := (J + C) / 2

/-- The number of cookies baked by Spencer -/
def S : ℕ := (3 * (J + T)) / 2

/-- The price of each cookie in cents -/
def price_per_cookie : ℕ := 175

/-- The total revenue in cents -/
def total_revenue : ℕ := (C + J + T + S) * price_per_cookie

theorem total_revenue_is_146475 : total_revenue = 146475 := by
  sorry

end total_revenue_is_146475_l322_32264


namespace sector_area_l322_32223

/-- The area of a sector with central angle 2π/3 and radius 3 is 3π. -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 2 * Real.pi / 3) (h2 : r = 3) :
  (θ / (2 * Real.pi)) * Real.pi * r^2 = 3 * Real.pi := by
  sorry

end sector_area_l322_32223


namespace jerrys_debt_l322_32210

theorem jerrys_debt (total_debt : ℕ) (first_payment : ℕ) (additional_payment : ℕ) 
  (h1 : total_debt = 50)
  (h2 : first_payment = 12)
  (h3 : additional_payment = 3) : 
  total_debt - (first_payment + (first_payment + additional_payment)) = 23 :=
by sorry

end jerrys_debt_l322_32210


namespace estate_area_calculation_l322_32225

/-- Represents the side length of the square on the map in inches -/
def map_side_length : ℝ := 12

/-- Represents the scale of the map in miles per inch -/
def map_scale : ℝ := 100

/-- Calculates the actual side length of the estate in miles -/
def actual_side_length : ℝ := map_side_length * map_scale

/-- Calculates the actual area of the estate in square miles -/
def actual_area : ℝ := actual_side_length ^ 2

/-- Theorem stating that the actual area of the estate is 1440000 square miles -/
theorem estate_area_calculation : actual_area = 1440000 := by
  sorry

end estate_area_calculation_l322_32225


namespace complex_product_magnitude_l322_32267

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 :=
by sorry

end complex_product_magnitude_l322_32267


namespace cost_of_dozen_pens_l322_32247

theorem cost_of_dozen_pens 
  (total_cost : ℕ) 
  (pencil_count : ℕ) 
  (pen_cost : ℕ) 
  (pen_pencil_ratio : ℚ) :
  total_cost = 260 →
  pencil_count = 5 →
  pen_cost = 65 →
  pen_pencil_ratio = 5 / 1 →
  (12 : ℕ) * pen_cost = 780 :=
by sorry

end cost_of_dozen_pens_l322_32247


namespace no_perfect_cube_solution_l322_32253

theorem no_perfect_cube_solution : ¬∃ (n : ℕ), n > 0 ∧ ∃ (y : ℕ), 3 * n^2 + 3 * n + 7 = y^3 := by
  sorry

end no_perfect_cube_solution_l322_32253


namespace irrational_and_no_negative_square_l322_32245

-- Define p: 2+√2 is irrational
def p : Prop := Irrational (2 + Real.sqrt 2)

-- Define q: ∃ x ∈ ℝ, x^2 < 0
def q : Prop := ∃ x : ℝ, x^2 < 0

-- Theorem statement
theorem irrational_and_no_negative_square : p ∧ ¬q := by sorry

end irrational_and_no_negative_square_l322_32245


namespace double_counted_is_eight_l322_32263

/-- The number of double-counted toddlers in Bill's count -/
def double_counted : ℕ := 26 - 21 + 3

/-- Proof that the number of double-counted toddlers is 8 -/
theorem double_counted_is_eight : double_counted = 8 := by
  sorry

#eval double_counted

end double_counted_is_eight_l322_32263


namespace lights_at_top_point_l322_32211

/-- Represents the number of layers in the structure -/
def num_layers : ℕ := 7

/-- Represents the common ratio of the geometric sequence -/
def common_ratio : ℕ := 2

/-- Represents the total number of lights -/
def total_lights : ℕ := 381

/-- Theorem stating that the number of lights at the topmost point is 3 -/
theorem lights_at_top_point : 
  ∃ (a : ℕ), a * (common_ratio ^ num_layers - 1) / (common_ratio - 1) = total_lights ∧ a = 3 :=
sorry

end lights_at_top_point_l322_32211


namespace correct_selection_count_l322_32221

/-- Represents a basketball team with twins -/
structure BasketballTeam where
  total_players : Nat
  twin_sets : Nat
  non_twins : Nat

/-- Calculates the number of ways to select players for a game -/
def select_players (team : BasketballTeam) (to_select : Nat) : Nat :=
  sorry

/-- The specific basketball team from the problem -/
def our_team : BasketballTeam := {
  total_players := 16,
  twin_sets := 3,
  non_twins := 10
}

/-- Theorem stating the correct number of ways to select players -/
theorem correct_selection_count :
  select_players our_team 7 = 1380 := by sorry

end correct_selection_count_l322_32221


namespace soap_cost_l322_32212

/-- The total cost of soap given the number of bars, weight per bar, and price per pound -/
theorem soap_cost (num_bars : ℕ) (weight_per_bar : ℝ) (price_per_pound : ℝ) :
  num_bars = 20 →
  weight_per_bar = 1.5 →
  price_per_pound = 0.5 →
  num_bars * weight_per_bar * price_per_pound = 15 := by
  sorry

end soap_cost_l322_32212


namespace pencil_distribution_l322_32222

theorem pencil_distribution (initial_pencils : ℕ) (kept_pencils : ℕ) (extra_to_nilo : ℕ) 
  (h1 : initial_pencils = 50)
  (h2 : kept_pencils = 20)
  (h3 : extra_to_nilo = 10) : 
  ∃ (pencils_to_manny : ℕ), 
    pencils_to_manny + (pencils_to_manny + extra_to_nilo) = initial_pencils - kept_pencils ∧ 
    pencils_to_manny = 10 := by
  sorry

end pencil_distribution_l322_32222


namespace rational_function_decomposition_l322_32234

theorem rational_function_decomposition :
  ∃ (P Q R : ℝ), 
    (∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 →
      (-x^3 + 4*x^2 - 5*x + 3) / (x^4 + x^2) = P/x^2 + (Q*x + R)/(x^2 + 1)) ∧
    P = 3 ∧ Q = -1 ∧ R = 1 := by
  sorry

end rational_function_decomposition_l322_32234


namespace inverse_sum_lower_bound_l322_32275

theorem inverse_sum_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  1/x + 1/y ≥ 3 + 2*Real.sqrt 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 1 ∧ 1/x₀ + 1/y₀ = 3 + 2*Real.sqrt 2 :=
sorry

end inverse_sum_lower_bound_l322_32275


namespace intersection_and_union_of_sets_l322_32258

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {-3+a, 2*a-1, a^2+1}

theorem intersection_and_union_of_sets :
  ∃ (a : ℝ), (A a ∩ B a = {-3}) ∧ (a = -1) ∧ (A a ∪ B a = {-4, -3, 0, 1, 2}) := by
  sorry

end intersection_and_union_of_sets_l322_32258
