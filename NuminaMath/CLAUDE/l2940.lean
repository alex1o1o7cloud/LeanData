import Mathlib

namespace NUMINAMATH_CALUDE_basketball_probabilities_l2940_294096

/-- A series of 6 independent Bernoulli trials with probability of success 1/3 -/
def bernoulli_trials (n : ℕ) (p : ℝ) := n = 6 ∧ p = 1/3

/-- Probability of two failures before the first success -/
def prob_two_failures_before_success (p : ℝ) : ℝ := (1 - p)^2 * p

/-- Probability of exactly k successes in n trials -/
def prob_exactly_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- Expected number of successes -/
def expected_successes (n : ℕ) (p : ℝ) : ℝ := n * p

/-- Variance of the number of successes -/
def variance_successes (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem basketball_probabilities (n : ℕ) (p : ℝ) 
  (h : bernoulli_trials n p) : 
  prob_two_failures_before_success p = 4/27 ∧
  prob_exactly_k_successes n 3 p = 160/729 ∧
  expected_successes n p = 2 ∧
  variance_successes n p = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probabilities_l2940_294096


namespace NUMINAMATH_CALUDE_percentage_4_plus_years_l2940_294032

/-- Represents the number of employees in each year group -/
structure EmployeeDistribution :=
  (less_than_1 : ℕ)
  (one_to_2 : ℕ)
  (two_to_3 : ℕ)
  (three_to_4 : ℕ)
  (four_to_5 : ℕ)
  (five_to_6 : ℕ)
  (six_to_7 : ℕ)
  (seven_to_8 : ℕ)
  (eight_to_9 : ℕ)
  (nine_to_10 : ℕ)
  (ten_plus : ℕ)

/-- Calculates the total number of employees -/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1 + d.one_to_2 + d.two_to_3 + d.three_to_4 + d.four_to_5 + 
  d.five_to_6 + d.six_to_7 + d.seven_to_8 + d.eight_to_9 + d.nine_to_10 + d.ten_plus

/-- Calculates the number of employees who have worked for 4 years or more -/
def employees_4_plus_years (d : EmployeeDistribution) : ℕ :=
  d.four_to_5 + d.five_to_6 + d.six_to_7 + d.seven_to_8 + d.eight_to_9 + d.nine_to_10 + d.ten_plus

/-- Theorem: The percentage of employees who have worked for 4 years or more is 37.5% -/
theorem percentage_4_plus_years (d : EmployeeDistribution) : 
  (employees_4_plus_years d : ℚ) / (total_employees d : ℚ) = 375 / 1000 :=
sorry


end NUMINAMATH_CALUDE_percentage_4_plus_years_l2940_294032


namespace NUMINAMATH_CALUDE_merchandise_profit_analysis_l2940_294033

/-- Represents a store's merchandise sales model -/
structure MerchandiseModel where
  original_cost : ℝ
  original_price : ℝ
  original_sales : ℝ
  price_decrease_step : ℝ
  sales_increase_step : ℝ

/-- Calculate the profit given a price decrease -/
def profit (model : MerchandiseModel) (price_decrease : ℝ) : ℝ :=
  (model.original_sales + model.sales_increase_step * price_decrease) *
  (model.original_price - price_decrease - model.original_cost)

theorem merchandise_profit_analysis (model : MerchandiseModel)
  (h1 : model.original_cost = 80)
  (h2 : model.original_price = 100)
  (h3 : model.original_sales = 100)
  (h4 : model.price_decrease_step = 1)
  (h5 : model.sales_increase_step = 10) :
  profit model 0 = 2000 ∧
  (∀ x, profit model x = -10 * x^2 + 100 * x + 2000) ∧
  (∃ x, profit model x = 2250 ∧ ∀ y, profit model y ≤ profit model x) ∧
  (∀ p, 92 ≤ p ∧ p ≤ 98 ↔ profit model (100 - p) ≥ 2160) :=
by sorry

end NUMINAMATH_CALUDE_merchandise_profit_analysis_l2940_294033


namespace NUMINAMATH_CALUDE_negative_390_same_terminal_side_as_330_l2940_294057

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem negative_390_same_terminal_side_as_330 :
  same_terminal_side (-390) 330 :=
sorry

end NUMINAMATH_CALUDE_negative_390_same_terminal_side_as_330_l2940_294057


namespace NUMINAMATH_CALUDE_set_intersection_example_l2940_294053

theorem set_intersection_example :
  let M : Set ℕ := {2, 3, 4, 5}
  let N : Set ℕ := {3, 4, 5}
  M ∩ N = {3, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l2940_294053


namespace NUMINAMATH_CALUDE_zoo_viewing_time_is_75_minutes_l2940_294051

/-- Calculates the total viewing time for a zoo visit -/
def total_zoo_viewing_time (original_times new_times : List ℕ) (break_time : ℕ) : ℕ :=
  let total_viewing_time := original_times.sum + new_times.sum
  let total_break_time := break_time * (original_times.length + new_times.length - 1)
  total_viewing_time + total_break_time

/-- Theorem: The total time required to see all 9 animal types is 75 minutes -/
theorem zoo_viewing_time_is_75_minutes :
  total_zoo_viewing_time [4, 6, 7, 5, 9] [3, 7, 8, 10] 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_zoo_viewing_time_is_75_minutes_l2940_294051


namespace NUMINAMATH_CALUDE_logarithm_product_theorem_l2940_294054

theorem logarithm_product_theorem (c d : ℕ+) : 
  (d - c = 840) →
  (Real.log d / Real.log c = 3) →
  (c + d : ℕ) = 1010 := by sorry

end NUMINAMATH_CALUDE_logarithm_product_theorem_l2940_294054


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2940_294072

-- Define the quadratic polynomial q(x)
def q (x : ℚ) : ℚ := -15/14 * x^2 - 75/14 * x + 180/7

-- Theorem stating that q(x) satisfies the given conditions
theorem quadratic_polynomial_satisfies_conditions :
  q (-8) = 0 ∧ q 3 = 0 ∧ q 6 = -45 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2940_294072


namespace NUMINAMATH_CALUDE_train_passing_time_l2940_294063

/-- Theorem: Time taken for slower train to pass faster train's driver -/
theorem train_passing_time
  (train_length : ℝ)
  (fast_train_speed slow_train_speed : ℝ)
  (h1 : train_length = 500)
  (h2 : fast_train_speed = 45)
  (h3 : slow_train_speed = 15) :
  (train_length / ((fast_train_speed + slow_train_speed) * (1000 / 3600))) = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2940_294063


namespace NUMINAMATH_CALUDE_remainder_of_P_div_Q_l2940_294045

/-- P(x) is a polynomial defined as x^(6n) + x^(5n) + x^(4n) + x^(3n) + x^(2n) + x^n + 1 -/
def P (x n : ℕ) : ℕ := x^(6*n) + x^(5*n) + x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1

/-- Q(x) is a polynomial defined as x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 -/
def Q (x : ℕ) : ℕ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

/-- Theorem stating that the remainder of P(x) divided by Q(x) is 7 when n is a multiple of 7 -/
theorem remainder_of_P_div_Q (x n : ℕ) (h : ∃ k, n = 7 * k) :
  P x n % Q x = 7 := by sorry

end NUMINAMATH_CALUDE_remainder_of_P_div_Q_l2940_294045


namespace NUMINAMATH_CALUDE_three_part_division_l2940_294078

theorem three_part_division (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) 
  (h4 : A + B + C = 782) (h5 : C = 306) : 
  ∃ (k : ℝ), k > 0 ∧ A = k * A ∧ B = k * B ∧ C = k * 306 ∧ A + B = 476 := by
  sorry

end NUMINAMATH_CALUDE_three_part_division_l2940_294078


namespace NUMINAMATH_CALUDE_cord_length_proof_l2940_294090

/-- Given a cord divided into 19 equal parts, which when cut results in 20 pieces
    with the longest piece being 8 meters and the shortest being 2 meters,
    prove that the original length of the cord is 114 meters. -/
theorem cord_length_proof (n : ℕ) (longest shortest : ℝ) :
  n = 19 ∧
  longest = 8 ∧
  shortest = 2 →
  n * ((longest + shortest) / 2 + 1) = 114 :=
by sorry

end NUMINAMATH_CALUDE_cord_length_proof_l2940_294090


namespace NUMINAMATH_CALUDE_truncated_pyramid_diagonal_l2940_294044

/-- Regular truncated quadrilateral pyramid -/
structure TruncatedPyramid where
  height : ℝ
  lower_base_side : ℝ
  upper_base_side : ℝ

/-- Diagonal of a truncated pyramid -/
def diagonal (p : TruncatedPyramid) : ℝ :=
  sorry

/-- Theorem: The diagonal of the specified truncated pyramid is 6 -/
theorem truncated_pyramid_diagonal :
  let p : TruncatedPyramid := ⟨2, 5, 3⟩
  diagonal p = 6 := by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_diagonal_l2940_294044


namespace NUMINAMATH_CALUDE_actual_distance_travelled_l2940_294050

/-- The actual distance travelled by a person under specific conditions -/
theorem actual_distance_travelled (normal_speed fast_speed additional_distance : ℝ) 
  (h1 : normal_speed = 10)
  (h2 : fast_speed = 14)
  (h3 : additional_distance = 20)
  (h4 : (actual_distance / normal_speed) = ((actual_distance + additional_distance) / fast_speed)) :
  actual_distance = 50 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_travelled_l2940_294050


namespace NUMINAMATH_CALUDE_max_value_expression_l2940_294095

theorem max_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + y - Real.sqrt (x^4 + y^2)) / x ≤ (1 : ℝ) / 2 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (x₀^2 + y₀ - Real.sqrt (x₀^4 + y₀^2)) / x₀ = (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2940_294095


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2940_294026

theorem quadratic_inequality (a b c : ℝ) : a^2 + a*b + a*c < 0 → b^2 > 4*a*c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2940_294026


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l2940_294000

theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 5 / 6
  let arc_length : ℝ := sector_fraction * (2 * π * r)
  let cone_base_radius : ℝ := arc_length / (2 * π)
  let cone_height : ℝ := Real.sqrt (r^2 - cone_base_radius^2)
  let cone_volume : ℝ := (1/3) * π * cone_base_radius^2 * cone_height
  cone_volume = (25/3) * π * Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l2940_294000


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l2940_294088

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : balls_in_boxes 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l2940_294088


namespace NUMINAMATH_CALUDE_lesser_fraction_l2940_294034

theorem lesser_fraction (x y : ℝ) (h_sum : x + y = 13/14) (h_prod : x * y = 1/8) :
  min x y = (13 - Real.sqrt 57) / 28 := by sorry

end NUMINAMATH_CALUDE_lesser_fraction_l2940_294034


namespace NUMINAMATH_CALUDE_sum_of_ages_is_100_l2940_294086

/-- Given the conditions about Alice, Ben, and Charlie's ages, prove that the sum of their ages is 100. -/
theorem sum_of_ages_is_100 (A B C : ℕ) 
  (h1 : A = 20 + B + C) 
  (h2 : A^2 = 2000 + (B + C)^2) : 
  A + B + C = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_100_l2940_294086


namespace NUMINAMATH_CALUDE_constant_regular_cells_problem_solution_l2940_294013

/-- Represents the number of regular cells capable of division after a given number of days -/
def regular_cells (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells

/-- Theorem stating that the number of regular cells remains constant -/
theorem constant_regular_cells (initial_cells : ℕ) (days : ℕ) :
  regular_cells initial_cells days = initial_cells :=
by sorry

/-- The specific case for the problem with 4 initial cells and 10 days -/
theorem problem_solution :
  regular_cells 4 10 = 4 :=
by sorry

end NUMINAMATH_CALUDE_constant_regular_cells_problem_solution_l2940_294013


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l2940_294029

theorem square_area_error_percentage (s : ℝ) (h : s > 0) :
  let measured_side := 1.06 * s
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let error_percentage := (area_error / actual_area) * 100
  error_percentage = 12.36 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l2940_294029


namespace NUMINAMATH_CALUDE_max_sides_with_four_obtuse_angles_l2940_294052

-- Define a convex polygon type
structure ConvexPolygon where
  sides : ℕ
  interior_angles : Fin sides → Real
  is_convex : Bool
  obtuse_count : ℕ

-- Define the theorem
theorem max_sides_with_four_obtuse_angles 
  (p : ConvexPolygon) 
  (h1 : p.is_convex = true) 
  (h2 : p.obtuse_count = 4) 
  (h3 : ∀ i, 0 < p.interior_angles i ∧ p.interior_angles i < 180) 
  (h4 : (Finset.sum Finset.univ p.interior_angles) = (p.sides - 2) * 180) :
  p.sides ≤ 7 := by
  sorry


end NUMINAMATH_CALUDE_max_sides_with_four_obtuse_angles_l2940_294052


namespace NUMINAMATH_CALUDE_escalator_speed_l2940_294035

theorem escalator_speed (escalator_speed : ℝ) (escalator_length : ℝ) (time_taken : ℝ) 
  (h1 : escalator_speed = 11)
  (h2 : escalator_length = 126)
  (h3 : time_taken = 9) :
  escalator_speed + (escalator_length - escalator_speed * time_taken) / time_taken = 14 :=
by sorry

end NUMINAMATH_CALUDE_escalator_speed_l2940_294035


namespace NUMINAMATH_CALUDE_no_arithmetic_sequence_with_sum_n_cubed_l2940_294094

theorem no_arithmetic_sequence_with_sum_n_cubed :
  ¬ ∃ (a₁ d : ℝ), ∀ (n : ℕ), n > 0 →
    (n : ℝ) / 2 * (2 * a₁ + (n - 1) * d) = (n : ℝ)^3 :=
sorry

end NUMINAMATH_CALUDE_no_arithmetic_sequence_with_sum_n_cubed_l2940_294094


namespace NUMINAMATH_CALUDE_cylinder_radius_in_cone_l2940_294009

/-- 
Given a right circular cone with diameter 14 and altitude 16, and an inscribed right circular 
cylinder whose diameter equals its height, prove that the radius of the cylinder is 56/15.
-/
theorem cylinder_radius_in_cone (r : ℚ) : 
  (16 : ℚ) - 2 * r = (16 : ℚ) / 7 * r → r = 56 / 15 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_in_cone_l2940_294009


namespace NUMINAMATH_CALUDE_officer_selection_count_l2940_294048

/-- The number of members in the club -/
def club_size : ℕ := 30

/-- The number of officer positions to be filled -/
def officer_positions : ℕ := 4

/-- Function to calculate the number of ways to choose officers -/
def choose_officers (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem stating the number of ways to choose officers under given conditions -/
theorem officer_selection_count :
  (choose_officers (club_size - 2) officer_positions) +
  (choose_officers 2 2 * choose_officers (club_size - 2) (officer_positions - 2)) = 495936 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_count_l2940_294048


namespace NUMINAMATH_CALUDE_candle_recycling_l2940_294062

def original_candle_weight : ℝ := 20
def wax_percentage : ℝ := 0.1
def num_candles : ℕ := 5
def new_candle_weight : ℝ := 5

theorem candle_recycling :
  (↑num_candles * original_candle_weight * wax_percentage) / new_candle_weight = 3 := by
  sorry

end NUMINAMATH_CALUDE_candle_recycling_l2940_294062


namespace NUMINAMATH_CALUDE_mark_recapture_not_suitable_for_centipedes_l2940_294074

/-- Represents a method for population quantity experiments -/
inductive PopulationExperimentMethod
| MarkRecapture
| Sampling

/-- Represents an animal type -/
inductive AnimalType
| Centipede
| Rodent

/-- Represents the size of an animal -/
inductive AnimalSize
| Small
| Large

/-- Function to determine if a method is suitable for an animal type -/
def methodSuitability (method : PopulationExperimentMethod) (animal : AnimalType) : Prop :=
  match method, animal with
  | PopulationExperimentMethod.MarkRecapture, AnimalType.Centipede => False
  | PopulationExperimentMethod.Sampling, AnimalType.Centipede => True
  | _, _ => True

/-- Function to determine the size of an animal -/
def animalSize (animal : AnimalType) : AnimalSize :=
  match animal with
  | AnimalType.Centipede => AnimalSize.Small
  | AnimalType.Rodent => AnimalSize.Large

/-- Theorem stating that the mark-recapture method is not suitable for investigating centipedes -/
theorem mark_recapture_not_suitable_for_centipedes :
  ¬(methodSuitability PopulationExperimentMethod.MarkRecapture AnimalType.Centipede) :=
by sorry

end NUMINAMATH_CALUDE_mark_recapture_not_suitable_for_centipedes_l2940_294074


namespace NUMINAMATH_CALUDE_paul_lost_crayons_l2940_294080

/-- Given information about Paul's crayons --/
def paul_crayons (initial : ℕ) (given_to_friends : ℕ) (difference : ℕ) : Prop :=
  ∃ (lost : ℕ), 
    initial ≥ given_to_friends + lost ∧
    given_to_friends = lost + difference

/-- Theorem stating the number of crayons Paul lost --/
theorem paul_lost_crayons :
  paul_crayons 589 571 410 → ∃ (lost : ℕ), lost = 161 := by
  sorry

end NUMINAMATH_CALUDE_paul_lost_crayons_l2940_294080


namespace NUMINAMATH_CALUDE_sector_area_l2940_294010

/-- The area of a sector with central angle 2π/3 and radius √3 is π -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 2 * Real.pi / 3) (h2 : r = Real.sqrt 3) :
  1/2 * r^2 * θ = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2940_294010


namespace NUMINAMATH_CALUDE_square_difference_equal_l2940_294015

theorem square_difference_equal (a b : ℝ) : (a - b)^2 = (b - a)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equal_l2940_294015


namespace NUMINAMATH_CALUDE_probability_theorem_l2940_294019

-- Define the total number of students
def total_students : ℕ := 20

-- Define the fraction of students interested in the career
def interested_fraction : ℚ := 4 / 5

-- Define the number of interested students
def interested_students : ℕ := (interested_fraction * total_students).num.toNat

-- Define the function to calculate the probability
def probability_at_least_one_interested : ℚ :=
  1 - (total_students - interested_students) * (total_students - interested_students - 1) /
      (total_students * (total_students - 1))

-- Theorem statement
theorem probability_theorem :
  probability_at_least_one_interested = 92 / 95 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l2940_294019


namespace NUMINAMATH_CALUDE_square_circle_union_area_l2940_294093

/-- The area of the union of a square with side length 8 and a circle with radius 8
    centered at one of the square's vertices is 64 + 48π square units. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 8
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  let overlap_area := (1/4 : ℝ) * circle_area
  square_area + circle_area - overlap_area = 64 + 48 * π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l2940_294093


namespace NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l2940_294001

theorem ceiling_negative_seven_fourths_squared : ⌈(-(7/4))^2⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l2940_294001


namespace NUMINAMATH_CALUDE_johns_age_is_20_l2940_294073

-- Define John's age and his dad's age
def johns_age : ℕ := sorry
def dads_age : ℕ := sorry

-- State the theorem
theorem johns_age_is_20 :
  (johns_age + 30 = dads_age) →
  (johns_age + dads_age = 70) →
  johns_age = 20 := by
sorry

end NUMINAMATH_CALUDE_johns_age_is_20_l2940_294073


namespace NUMINAMATH_CALUDE_sphere_identical_views_l2940_294025

/-- A geometric body in 3D space -/
inductive GeometricBody
  | Sphere
  | Cylinder
  | TriangularPrism
  | Cone

/-- Represents a 2D view of a geometric body -/
structure View where
  shape : Type
  size : ℝ

/-- Returns true if all views are identical -/
def identicalViews (front side top : View) : Prop :=
  front = side ∧ side = top

/-- Returns the front, side, and top views of a geometric body -/
def getViews (body : GeometricBody) : (View × View × View) :=
  sorry

theorem sphere_identical_views :
  ∀ (body : GeometricBody),
    (∃ (front side top : View), 
      getViews body = (front, side, top) ∧ 
      identicalViews front side top) 
    ↔ 
    body = GeometricBody.Sphere :=
  sorry

end NUMINAMATH_CALUDE_sphere_identical_views_l2940_294025


namespace NUMINAMATH_CALUDE_triangle_projection_relation_l2940_294040

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  h_a_gt_b : a > b
  h_b_gt_c : b > c
  h_a_eq_2bc : a = 2 * (b - c)
  h_p_projection : p ≥ 0 ∧ p ≤ a -- Assuming p is a valid projection

-- State the theorem
theorem triangle_projection_relation (t : Triangle) : 4 * t.c + 8 * t.p = 3 * t.a := by
  sorry

end NUMINAMATH_CALUDE_triangle_projection_relation_l2940_294040


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_3_root_6_over_2_l2940_294037

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a quadrilateral in 3D space -/
structure Quadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Given a cube of side length 2, calculates the area of quadrilateral ABCD where
    A and C are diagonally opposite vertices, and B and D are quarter points on
    opposite edges not containing A or C -/
def quadrilateralArea (cube : Point3D → Bool) (A B C D : Point3D) : ℝ :=
  sorry

/-- Theorem stating that the area of the quadrilateral ABCD in the given conditions is 3√6/2 -/
theorem quadrilateral_area_is_3_root_6_over_2 
  (cube : Point3D → Bool) 
  (A B C D : Point3D) 
  (h_cube_side : ∀ (p q : Point3D), cube p ∧ cube q → 
    (p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2 ≤ 4)
  (h_A_C_diagonal : cube A ∧ cube C ∧ 
    (A.x - C.x)^2 + (A.y - C.y)^2 + (A.z - C.z)^2 = 12)
  (h_B_quarter : ∃ (p q : Point3D), cube p ∧ cube q ∧
    (B.x - p.x)^2 + (B.y - p.y)^2 + (B.z - p.z)^2 = 1/4 ∧
    (B.x - q.x)^2 + (B.y - q.y)^2 + (B.z - q.z)^2 = 9/4)
  (h_D_quarter : ∃ (r s : Point3D), cube r ∧ cube s ∧
    (D.x - r.x)^2 + (D.y - r.y)^2 + (D.z - r.z)^2 = 1/4 ∧
    (D.x - s.x)^2 + (D.y - s.y)^2 + (D.z - s.z)^2 = 9/4)
  : quadrilateralArea cube A B C D = 3 * Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_3_root_6_over_2_l2940_294037


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_when_sum_geq_5_l2940_294067

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 3|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≤ 6} = {x : ℝ | 0 ≤ x ∧ x ≤ 3} :=
by sorry

-- Part 2
theorem range_of_a_when_sum_geq_5 :
  ∀ x : ℝ, f a x + g x ≥ 5 → a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_when_sum_geq_5_l2940_294067


namespace NUMINAMATH_CALUDE_triangle_area_special_conditions_l2940_294011

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the area of the triangle under given conditions -/
theorem triangle_area_special_conditions (t : Triangle) 
  (h1 : (t.a - t.c)^2 = t.b^2 - 3/4 * t.a * t.c)
  (h2 : t.b = Real.sqrt 13)
  (h3 : ∃ (d : ℝ), Real.sin t.A + Real.sin t.C = 2 * Real.sin t.B) :
  (1/2 * t.a * t.c * Real.sin t.B) = (3 * Real.sqrt 39) / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_special_conditions_l2940_294011


namespace NUMINAMATH_CALUDE_equation_represents_parallel_lines_l2940_294097

theorem equation_represents_parallel_lines :
  ∃ (m c₁ c₂ : ℝ), m ≠ 0 ∧ c₁ ≠ c₂ ∧
  ∀ (x y : ℝ), x^2 - 9*y^2 + 3*x = 0 ↔ (y = m*x + c₁ ∨ y = m*x + c₂) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_parallel_lines_l2940_294097


namespace NUMINAMATH_CALUDE_alyssa_gave_away_seven_puppies_l2940_294075

/-- The number of puppies Alyssa gave to her friends -/
def puppies_given_away (initial : ℕ) (current : ℕ) : ℕ :=
  initial - current

/-- Theorem stating that Alyssa gave away 7 puppies -/
theorem alyssa_gave_away_seven_puppies :
  puppies_given_away 12 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_gave_away_seven_puppies_l2940_294075


namespace NUMINAMATH_CALUDE_phil_change_is_seven_l2940_294047

/-- The change Phil received after buying apples -/
def change_received : ℚ :=
  let number_of_apples : ℕ := 4
  let cost_per_apple : ℚ := 75 / 100
  let amount_paid : ℚ := 10
  amount_paid - (number_of_apples * cost_per_apple)

/-- Proof that Phil received $7.00 in change -/
theorem phil_change_is_seven : change_received = 7 := by
  sorry

end NUMINAMATH_CALUDE_phil_change_is_seven_l2940_294047


namespace NUMINAMATH_CALUDE_problem_solution_l2940_294070

theorem problem_solution (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + e^2 + 1 = d + Real.sqrt (a + b + c + e - 2*d)) : 
  d = -23/8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2940_294070


namespace NUMINAMATH_CALUDE_intersection_points_form_rectangle_l2940_294042

/-- The set of points satisfying xy = 18 and x^2 + y^2 = 45 -/
def IntersectionPoints : Set (ℝ × ℝ) :=
  {p | p.1 * p.2 = 18 ∧ p.1^2 + p.2^2 = 45}

/-- A function to check if four points form a rectangle -/
def IsRectangle (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let d12 := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2
  let d23 := (p2.1 - p3.1)^2 + (p2.2 - p3.2)^2
  let d34 := (p3.1 - p4.1)^2 + (p3.2 - p4.2)^2
  let d41 := (p4.1 - p1.1)^2 + (p4.2 - p1.2)^2
  let d13 := (p1.1 - p3.1)^2 + (p1.2 - p3.2)^2
  let d24 := (p2.1 - p4.1)^2 + (p2.2 - p4.2)^2
  (d12 = d34 ∧ d23 = d41) ∧ (d13 = d24)

theorem intersection_points_form_rectangle :
  ∃ p1 p2 p3 p4 : ℝ × ℝ, p1 ∈ IntersectionPoints ∧ p2 ∈ IntersectionPoints ∧
    p3 ∈ IntersectionPoints ∧ p4 ∈ IntersectionPoints ∧
    IsRectangle p1 p2 p3 p4 :=
  sorry

end NUMINAMATH_CALUDE_intersection_points_form_rectangle_l2940_294042


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2940_294060

/-- The solution set of the inequality x^2 - ax + a - 1 ≤ 0 for real a -/
def solution_set (a : ℝ) : Set ℝ :=
  if a < 2 then Set.Icc (a - 1) 1
  else if a = 2 then {1}
  else Set.Icc 1 (a - 1)

/-- Theorem stating the solution set of the inequality x^2 - ax + a - 1 ≤ 0 -/
theorem inequality_solution_set (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ x^2 - a*x + a - 1 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2940_294060


namespace NUMINAMATH_CALUDE_prob_second_white_given_first_white_l2940_294003

/-- Represents the color of a ball -/
inductive BallColor
| White
| Black

/-- Represents the state of the bag -/
structure BagState where
  white : ℕ
  black : ℕ

/-- The initial state of the bag -/
def initial_bag : BagState := ⟨3, 2⟩

/-- The probability of drawing a white ball given the bag state -/
def prob_white (bag : BagState) : ℚ :=
  bag.white / (bag.white + bag.black)

/-- The probability of drawing a specific color given the bag state -/
def prob_draw (bag : BagState) (color : BallColor) : ℚ :=
  match color with
  | BallColor.White => prob_white bag
  | BallColor.Black => 1 - prob_white bag

/-- The new bag state after drawing a ball of a given color -/
def draw_ball (bag : BagState) (color : BallColor) : BagState :=
  match color with
  | BallColor.White => ⟨bag.white - 1, bag.black⟩
  | BallColor.Black => ⟨bag.white, bag.black - 1⟩

theorem prob_second_white_given_first_white :
  prob_draw (draw_ball initial_bag BallColor.White) BallColor.White = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_white_given_first_white_l2940_294003


namespace NUMINAMATH_CALUDE_exists_cubic_polynomial_with_positive_roots_and_negative_derivative_roots_l2940_294089

/-- A cubic polynomial -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a*x^3 + b*x^2 + c*x + d

/-- The derivative of a cubic polynomial -/
def DerivativeCubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ 3*a*x^2 + 2*b*x + c

/-- All roots of a function are positive -/
def AllRootsPositive (f : ℝ → ℝ) : Prop := ∀ x, f x = 0 → x > 0

/-- All roots of a function are negative -/
def AllRootsNegative (f : ℝ → ℝ) : Prop := ∀ x, f x = 0 → x < 0

/-- A function has at least one unique root -/
def HasUniqueRoot (f : ℝ → ℝ) : Prop := ∃ x, f x = 0 ∧ ∀ y, f y = 0 → y = x

theorem exists_cubic_polynomial_with_positive_roots_and_negative_derivative_roots :
  ∃ (a b c d : ℝ), 
    let P := CubicPolynomial a b c d
    let P' := DerivativeCubicPolynomial (3*a) (2*b) c
    AllRootsPositive P ∧
    AllRootsNegative P' ∧
    HasUniqueRoot P ∧
    HasUniqueRoot P' :=
sorry

end NUMINAMATH_CALUDE_exists_cubic_polynomial_with_positive_roots_and_negative_derivative_roots_l2940_294089


namespace NUMINAMATH_CALUDE_equation_solution_l2940_294017

theorem equation_solution : ∃ x : ℝ, (3 / 4 - 1 / x = 1 / 2) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2940_294017


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l2940_294012

theorem triangle_angle_problem (a b c : ℝ) (A : ℝ) :
  b = c →
  a^2 = 2 * b^2 * (1 - Real.sin A) →
  A = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l2940_294012


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2940_294014

/-- Simple interest calculation -/
theorem simple_interest_principal (interest : ℚ) (rate : ℚ) (time : ℚ) :
  interest = 8625 →
  rate = 50 / 3 →
  time = 3 / 4 →
  ∃ principal : ℚ, principal = 69000 ∧ interest = principal * rate * time / 100 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2940_294014


namespace NUMINAMATH_CALUDE_sum_of_n_factorial_times_n_l2940_294006

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_n_factorial_times_n (a : ℕ) (h : factorial 1580 = a) :
  (Finset.range 1581).sum (λ n => n * factorial n) = 1581 * a - 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_n_factorial_times_n_l2940_294006


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l2940_294066

/-- Given a parabola with focus F(a,0) where a < 0, its standard equation is y^2 = 4ax -/
theorem parabola_standard_equation (a : ℝ) (h : a < 0) :
  ∃ (x y : ℝ), y^2 = 4*a*x :=
sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l2940_294066


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l2940_294007

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ice cream flavors that can be created -/
def ice_cream_flavors : ℕ := distribute 5 4

theorem ice_cream_flavors_count : ice_cream_flavors = 56 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l2940_294007


namespace NUMINAMATH_CALUDE_oldest_youngest_sum_l2940_294004

def age_problem (a b c d : ℕ) : Prop :=
  a + b + c + d = 100 ∧
  a = 32 ∧
  a + b = 3 * (c + d) ∧
  c = d + 3

theorem oldest_youngest_sum (a b c d : ℕ) 
  (h : age_problem a b c d) : 
  max a (max b (max c d)) + min a (min b (min c d)) = 54 := by
  sorry

end NUMINAMATH_CALUDE_oldest_youngest_sum_l2940_294004


namespace NUMINAMATH_CALUDE_cycle_original_price_l2940_294069

/-- Proves that given a cycle sold for 1260 with a 40% gain, the original price was 900 --/
theorem cycle_original_price (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1260)
  (h2 : gain_percentage = 40) : 
  (selling_price / (1 + gain_percentage / 100)) = 900 := by
  sorry

end NUMINAMATH_CALUDE_cycle_original_price_l2940_294069


namespace NUMINAMATH_CALUDE_intersecting_lines_l2940_294021

theorem intersecting_lines (k : ℚ) : 
  (∃! p : ℚ × ℚ, 
    p.1 + k * p.2 = 0 ∧ 
    2 * p.1 + 3 * p.2 + 8 = 0 ∧ 
    p.1 - p.2 - 1 = 0) → 
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_intersecting_lines_l2940_294021


namespace NUMINAMATH_CALUDE_nicks_chocolate_oranges_l2940_294049

/-- Proves the number of chocolate oranges Nick had initially -/
theorem nicks_chocolate_oranges 
  (candy_bar_price : ℕ) 
  (chocolate_orange_price : ℕ) 
  (fundraising_goal : ℕ) 
  (candy_bars_to_sell : ℕ) 
  (h1 : candy_bar_price = 5)
  (h2 : chocolate_orange_price = 10)
  (h3 : fundraising_goal = 1000)
  (h4 : candy_bars_to_sell = 160)
  (h5 : candy_bar_price * candy_bars_to_sell + chocolate_orange_price * chocolate_oranges = fundraising_goal) :
  chocolate_oranges = 20 := by
  sorry

end NUMINAMATH_CALUDE_nicks_chocolate_oranges_l2940_294049


namespace NUMINAMATH_CALUDE_perfect_numbers_mn_value_S_is_perfect_min_sum_value_l2940_294031

/-- Definition of a perfect number -/
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

/-- Statement 1: 29 and 13 are perfect numbers -/
theorem perfect_numbers : is_perfect_number 29 ∧ is_perfect_number 13 := by sorry

/-- Statement 2: Given equation has mn = ±4 -/
theorem mn_value (m n : ℝ) : 
  (∀ a : ℝ, a^2 - 4*a + 8 = (a - m)^2 + n^2) → m*n = 4 ∨ m*n = -4 := by sorry

/-- Statement 3: S is a perfect number when k = 36 -/
theorem S_is_perfect (a b : ℤ) :
  let S := a^2 + 4*a*b + 5*b^2 - 12*b + 36
  ∃ x y : ℤ, S = x^2 + y^2 := by sorry

/-- Statement 4: Minimum value of a + b is 3 -/
theorem min_sum_value (a b : ℝ) :
  -a^2 + 5*a + b - 7 = 0 → a + b ≥ 3 := by sorry

end NUMINAMATH_CALUDE_perfect_numbers_mn_value_S_is_perfect_min_sum_value_l2940_294031


namespace NUMINAMATH_CALUDE_i_13_times_1_plus_i_l2940_294043

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem i_13_times_1_plus_i : i^13 * (1 + i) = -1 + i := by sorry

end NUMINAMATH_CALUDE_i_13_times_1_plus_i_l2940_294043


namespace NUMINAMATH_CALUDE_harmonic_division_in_circumscribed_square_l2940_294024

-- Define the square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define the circle
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the configuration
structure Configuration where
  square : Square
  circle : Circle
  tangent : Line
  P : Point
  Q : Point
  R : Point
  S : Point

-- Define the property of being circumscribed
def is_circumscribed (s : Square) (c : Circle) : Prop :=
  s.side = 2 * c.radius ∧ s.center = c.center

-- Define the property of being a tangent
def is_tangent (l : Line) (c : Circle) : Prop :=
  ∃ (p : Point), p.x^2 + p.y^2 = c.radius^2 ∧ l.a * p.x + l.b * p.y + l.c = 0

-- Define the property of points being on the square or its extensions
def on_square_or_extension (p : Point) (s : Square) : Prop :=
  (p.x = s.center.1 - s.side/2 ∨ p.x = s.center.1 + s.side/2) ∨
  (p.y = s.center.2 - s.side/2 ∨ p.y = s.center.2 + s.side/2)

-- Define the harmonic division property
def harmonic_division (a : Point) (b : Point) (c : Point) (d : Point) : Prop :=
  (a.x - c.x) / (b.x - c.x) = (a.x - d.x) / (b.x - d.x)

-- Main theorem
theorem harmonic_division_in_circumscribed_square (cfg : Configuration) 
  (h1 : is_circumscribed cfg.square cfg.circle)
  (h2 : is_tangent cfg.tangent cfg.circle)
  (h3 : on_square_or_extension cfg.P cfg.square)
  (h4 : on_square_or_extension cfg.Q cfg.square)
  (h5 : on_square_or_extension cfg.R cfg.square)
  (h6 : on_square_or_extension cfg.S cfg.square) :
  harmonic_division cfg.P cfg.R cfg.Q cfg.S ∧ harmonic_division cfg.Q cfg.S cfg.P cfg.R :=
sorry

end NUMINAMATH_CALUDE_harmonic_division_in_circumscribed_square_l2940_294024


namespace NUMINAMATH_CALUDE_cube_root_of_square_l2940_294046

theorem cube_root_of_square (x : ℝ) : x > 0 → (x^2)^(1/3) = x^(2/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_square_l2940_294046


namespace NUMINAMATH_CALUDE_money_exchange_problem_l2940_294077

/-- Proves that given 100 one-hundred-yuan bills exchanged for twenty-yuan and fifty-yuan bills
    totaling 260 bills, the number of twenty-yuan bills is 100 and the number of fifty-yuan bills is 160. -/
theorem money_exchange_problem (x y : ℕ) 
  (h1 : x + y = 260)
  (h2 : 20 * x + 50 * y = 100 * 100) :
  x = 100 ∧ y = 160 := by
  sorry

end NUMINAMATH_CALUDE_money_exchange_problem_l2940_294077


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_l2940_294092

theorem count_four_digit_numbers : 
  (Finset.range 4001).card = (Finset.Icc 1000 5000).card := by sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_l2940_294092


namespace NUMINAMATH_CALUDE_uniform_random_transformation_l2940_294039

/-- A uniform random number on an interval -/
def UniformRandom (a b : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ b}

theorem uniform_random_transformation (b₁ : ℝ) (b : ℝ) :
  b₁ ∈ UniformRandom 0 1 →
  b = (b₁ - 0.5) * 6 →
  b ∈ UniformRandom (-3) 3 := by
  sorry

end NUMINAMATH_CALUDE_uniform_random_transformation_l2940_294039


namespace NUMINAMATH_CALUDE_product_and_quotient_cube_square_l2940_294058

theorem product_and_quotient_cube_square (a b k : ℕ) : 
  100 ≤ a * b ∧ a * b < 1000 →  -- three-digit number condition
  a * b = k^3 →                 -- product is cube of k
  (a : ℚ) / b = k^2 →           -- quotient is square of k
  a = 243 ∧ b = 3 ∧ k = 9 := by
sorry

end NUMINAMATH_CALUDE_product_and_quotient_cube_square_l2940_294058


namespace NUMINAMATH_CALUDE_simone_apple_days_l2940_294071

theorem simone_apple_days (d : ℕ) : 
  (1/2 : ℚ) * d + (1/3 : ℚ) * 15 = 13 → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_simone_apple_days_l2940_294071


namespace NUMINAMATH_CALUDE_function_inequality_l2940_294091

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)
variable (hf' : ∀ x, deriv f x < f x)

-- Define the theorem
theorem function_inequality (a : ℝ) (ha : a > 0) :
  f a < Real.exp a * f 0 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l2940_294091


namespace NUMINAMATH_CALUDE_probability_all_yellow_apples_l2940_294059

def total_apples : ℕ := 8
def yellow_apples : ℕ := 3
def red_apples : ℕ := 5
def apples_chosen : ℕ := 3

theorem probability_all_yellow_apples :
  (Nat.choose yellow_apples apples_chosen) / (Nat.choose total_apples apples_chosen) = 1 / 56 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_yellow_apples_l2940_294059


namespace NUMINAMATH_CALUDE_fran_speed_to_match_joann_l2940_294027

/-- Proves that Fran needs to ride at 30 mph to cover the same distance as Joann -/
theorem fran_speed_to_match_joann (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) :
  joann_speed = 15 →
  joann_time = 4 →
  fran_time = 2 →
  (fran_time * (joann_speed * joann_time / fran_time) = joann_speed * joann_time) ∧
  (joann_speed * joann_time / fran_time = 30) := by
  sorry

end NUMINAMATH_CALUDE_fran_speed_to_match_joann_l2940_294027


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2940_294041

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is perpendicular to c, then k = -3 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (k : ℝ) : 
  a = (Real.sqrt 3, 1) → 
  b = (0, 1) → 
  c = (k, Real.sqrt 3) → 
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = 0 → 
  k = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2940_294041


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l2940_294083

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-player round-robin tournament has 45 matches. -/
theorem ten_player_tournament_matches :
  num_matches 10 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ten_player_tournament_matches_l2940_294083


namespace NUMINAMATH_CALUDE_point_coordinates_sum_of_coordinates_l2940_294020

/-- Given three points X, Y, and Z in the plane satisfying certain ratios,
    prove that X has specific coordinates. -/
theorem point_coordinates (X Y Z : ℝ × ℝ) : 
  Y = (2, 3) →
  Z = (5, 1) →
  (dist X Z) / (dist X Y) = 1/3 →
  (dist Z Y) / (dist X Y) = 2/3 →
  X = (6.5, 0) :=
by sorry

/-- The sum of coordinates of point X -/
def sum_coordinates (X : ℝ × ℝ) : ℝ :=
  X.1 + X.2

/-- Prove that the sum of coordinates of X is 6.5 -/
theorem sum_of_coordinates (X : ℝ × ℝ) :
  X = (6.5, 0) →
  sum_coordinates X = 6.5 :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_sum_of_coordinates_l2940_294020


namespace NUMINAMATH_CALUDE_flight_duration_theorem_l2940_294036

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hk : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Converts minutes to hours and minutes -/
def minutesToTime (totalMinutes : ℕ) : Time :=
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60,
    hk := by sorry }

theorem flight_duration_theorem (departureTime arrivalTime : Time) 
    (hDepart : departureTime.hours = 9 ∧ departureTime.minutes = 20)
    (hArrive : arrivalTime.hours = 13 ∧ arrivalTime.minutes = 45)
    (delay : ℕ)
    (hDelay : delay = 25) :
  let actualDuration := minutesToTime (timeDifferenceInMinutes departureTime arrivalTime + delay)
  actualDuration.hours + actualDuration.minutes = 29 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_theorem_l2940_294036


namespace NUMINAMATH_CALUDE_exponential_inequality_l2940_294087

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2940_294087


namespace NUMINAMATH_CALUDE_odd_function_with_minimum_l2940_294022

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem odd_function_with_minimum (a b c d : ℝ) :
  (∀ x, f a b c d x = -f a b c d (-x)) →  -- f is an odd function
  (∀ x, f a b c d x ≥ f a b c d (-1)) →   -- f(-1) is the minimum value
  f a b c d (-1) = -1 →                   -- f(-1) = -1
  (∀ x, f a b c d x = -x^3 + x) :=        -- Conclusion: f(x) = -x³ + x
by
  sorry


end NUMINAMATH_CALUDE_odd_function_with_minimum_l2940_294022


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2940_294016

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 - 3•X + 1 : Polynomial ℝ) = (X^2 - X - 1) * q + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2940_294016


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2940_294068

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {y | ∃ x, y = x^2}

-- State the theorem
theorem M_intersect_N_eq_M : M ∩ N = M := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2940_294068


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l2940_294005

theorem estimate_sqrt_expression :
  ∀ (x : ℝ), (1.4 < Real.sqrt 2 ∧ Real.sqrt 2 < 1.5) →
  (6 < (3 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt (1/3) ∧
   (3 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt (1/3) < 7) := by
  sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l2940_294005


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l2940_294008

/-- The number of visitors to Buckingham Palace on different days --/
structure PalaceVisitors where
  dayOfVisit : ℕ
  previousDay : ℕ
  twoDaysPrior : ℕ

/-- Calculates the difference in visitors between the day of visit and the sum of the previous two days --/
def visitorDifference (v : PalaceVisitors) : ℕ :=
  v.dayOfVisit - (v.previousDay + v.twoDaysPrior)

/-- Theorem stating the difference in visitors for the given data --/
theorem buckingham_palace_visitors :
  ∃ (v : PalaceVisitors),
    v.dayOfVisit = 8333 ∧
    v.previousDay = 3500 ∧
    v.twoDaysPrior = 2500 ∧
    visitorDifference v = 2333 := by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l2940_294008


namespace NUMINAMATH_CALUDE_car_cost_equation_l2940_294099

/-- Proves that the original cost of the car satisfies the given equation -/
theorem car_cost_equation (repair_cost selling_price profit_percent : ℝ) 
  (h1 : repair_cost = 15000)
  (h2 : selling_price = 64900)
  (h3 : profit_percent = 13.859649122807017) :
  ∃ C : ℝ, (1 + profit_percent / 100) * C = selling_price - repair_cost :=
sorry

end NUMINAMATH_CALUDE_car_cost_equation_l2940_294099


namespace NUMINAMATH_CALUDE_raccoon_nut_distribution_l2940_294055

/-- Represents the number of nuts taken by each raccoon -/
structure NutDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given distribution satisfies all conditions -/
def isValidDistribution (d : NutDistribution) : Prop :=
  -- First raccoon's final nuts
  let first_final := d.first * 5 / 6 + d.second / 18 + d.third * 7 / 48
  -- Second raccoon's final nuts
  let second_final := d.first / 9 + d.second / 3 + d.third * 7 / 48
  -- Third raccoon's final nuts
  let third_final := d.first / 9 + d.second / 9 + d.third / 8
  -- All distributions result in whole numbers
  (d.first * 5 % 6 = 0) ∧ (d.second % 18 = 0) ∧ (d.third * 7 % 48 = 0) ∧
  (d.first % 9 = 0) ∧ (d.second % 3 = 0) ∧
  (d.first % 9 = 0) ∧ (d.second % 9 = 0) ∧ (d.third % 8 = 0) ∧
  -- Final ratio is 4:3:2
  (3 * first_final = 4 * second_final) ∧ (3 * first_final = 6 * third_final)

/-- The minimum total number of nuts -/
def minTotalNuts : ℕ := 864

theorem raccoon_nut_distribution :
  ∃ (d : NutDistribution), isValidDistribution d ∧
    d.first + d.second + d.third = minTotalNuts ∧
    (∀ (d' : NutDistribution), isValidDistribution d' →
      d'.first + d'.second + d'.third ≥ minTotalNuts) :=
  sorry


end NUMINAMATH_CALUDE_raccoon_nut_distribution_l2940_294055


namespace NUMINAMATH_CALUDE_game_configurations_l2940_294030

/-- The number of rows in the grid -/
def m : ℕ := 5

/-- The number of columns in the grid -/
def n : ℕ := 7

/-- The total number of steps needed to reach from bottom-left to top-right -/
def total_steps : ℕ := m + n

/-- The number of unique paths from bottom-left to top-right of an m × n grid -/
def num_paths : ℕ := Nat.choose total_steps n

theorem game_configurations : num_paths = 792 := by sorry

end NUMINAMATH_CALUDE_game_configurations_l2940_294030


namespace NUMINAMATH_CALUDE_alcohol_remaining_l2940_294064

/-- The amount of alcohol remaining after a series of pours and refills -/
def remaining_alcohol (initial_volume : ℚ) (pour_out1 : ℚ) (refill1 : ℚ) 
  (pour_out2 : ℚ) (refill2 : ℚ) (pour_out3 : ℚ) (refill3 : ℚ) : ℚ :=
  initial_volume * (1 - pour_out1) * (1 - pour_out2) * (1 - pour_out3)

/-- Theorem stating the final amount of alcohol in the bottle -/
theorem alcohol_remaining :
  remaining_alcohol 1 (1/3) (1/3) (1/3) (1/3) 1 1 = 8/27 := by
  sorry


end NUMINAMATH_CALUDE_alcohol_remaining_l2940_294064


namespace NUMINAMATH_CALUDE_vector_dot_product_l2940_294056

/-- Given two 2D vectors a and b, prove that their dot product is -18. -/
theorem vector_dot_product (a b : ℝ × ℝ) : 
  a = (1, -3) → b = (3, 7) → a.1 * b.1 + a.2 * b.2 = -18 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l2940_294056


namespace NUMINAMATH_CALUDE_root_sum_ratio_l2940_294082

theorem root_sum_ratio (m₁ m₂ : ℝ) : 
  (∃ p q : ℝ, 
    (∀ m : ℝ, m * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    p / q + q / p = 2 ∧
    (m₁ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₁ * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    (m₂ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₂ * (q^2 - 3*q) + 2*q + 7 = 0)) →
  m₁ / m₂ + m₂ / m₁ = 85/2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l2940_294082


namespace NUMINAMATH_CALUDE_teal_more_blue_l2940_294038

/-- The number of people surveyed -/
def total_surveyed : ℕ := 150

/-- The number of people who believe teal is "more green" -/
def more_green : ℕ := 90

/-- The number of people who believe teal is both "more green" and "more blue" -/
def both : ℕ := 35

/-- The number of people who believe teal is neither "more green" nor "more blue" -/
def neither : ℕ := 25

/-- The theorem stating that 70 people believe teal is "more blue" -/
theorem teal_more_blue : 
  ∃ (more_blue : ℕ), more_blue = 70 ∧ 
  more_blue + (more_green - both) + both + neither = total_surveyed :=
sorry

end NUMINAMATH_CALUDE_teal_more_blue_l2940_294038


namespace NUMINAMATH_CALUDE_only_solution_is_five_l2940_294084

theorem only_solution_is_five (n : ℤ) : 
  (⌊(n^2 : ℚ) / 5⌋ - ⌊(n : ℚ) / 2⌋^2 = 1) ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_only_solution_is_five_l2940_294084


namespace NUMINAMATH_CALUDE_unique_ages_l2940_294002

/-- Represents the ages of Abe, Beth, and Charlie -/
structure Ages where
  abe : ℕ
  beth : ℕ
  charlie : ℕ

/-- Checks if the given ages satisfy all the conditions -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.abe + ages.beth = 45 ∧
  ages.abe - ages.beth = 9 ∧
  (ages.abe - 7) + (ages.beth - 7) = 31 ∧
  ages.charlie - ages.abe = 5 ∧
  ages.charlie + ages.beth = 56

/-- Theorem stating that the ages 27, 18, and 38 are the unique solution -/
theorem unique_ages : ∃! ages : Ages, satisfiesConditions ages ∧ 
  ages.abe = 27 ∧ ages.beth = 18 ∧ ages.charlie = 38 := by
  sorry

end NUMINAMATH_CALUDE_unique_ages_l2940_294002


namespace NUMINAMATH_CALUDE_repetitions_for_99_cubes_impossible_2016_cubes_l2940_294081

/-- The number of cubes after x repetitions of the cutting process -/
def num_cubes (x : ℕ) : ℕ := 7 * x + 1

/-- Theorem stating that 14 repetitions are needed to obtain 99 cubes -/
theorem repetitions_for_99_cubes : ∃ x : ℕ, num_cubes x = 99 ∧ x = 14 := by sorry

/-- Theorem stating that it's impossible to obtain 2016 cubes -/
theorem impossible_2016_cubes : ¬∃ x : ℕ, num_cubes x = 2016 := by sorry

end NUMINAMATH_CALUDE_repetitions_for_99_cubes_impossible_2016_cubes_l2940_294081


namespace NUMINAMATH_CALUDE_impossible_closed_line_l2940_294098

/-- Represents a prism with a given number of lateral edges and total edges. -/
structure Prism where
  lateral_edges : ℕ
  total_edges : ℕ

/-- Represents the possibility of forming a closed broken line from translated edges of a prism. -/
def can_form_closed_line (p : Prism) : Prop :=
  ∃ (arrangement : List ℝ), 
    arrangement.length = p.total_edges ∧ 
    arrangement.sum = 0 ∧
    (∀ i ∈ arrangement, i = 0 ∨ i = 1 ∨ i = -1)

/-- Theorem stating that it's impossible to form a closed broken line from the given prism's edges. -/
theorem impossible_closed_line (p : Prism) 
  (h1 : p.lateral_edges = 373) 
  (h2 : p.total_edges = 1119) : 
  ¬ can_form_closed_line p := by
  sorry

end NUMINAMATH_CALUDE_impossible_closed_line_l2940_294098


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l2940_294023

def james_age_3_years_ago : ℕ := 27
def matt_current_age : ℕ := 65
def years_since_james_27 : ℕ := 3
def years_to_future : ℕ := 5

def james_future_age : ℕ := james_age_3_years_ago + years_since_james_27 + years_to_future
def matt_future_age : ℕ := matt_current_age + years_to_future

theorem age_ratio_is_two_to_one :
  (matt_future_age : ℚ) / james_future_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l2940_294023


namespace NUMINAMATH_CALUDE_halfway_between_one_sixth_and_one_fourth_l2940_294018

theorem halfway_between_one_sixth_and_one_fourth : 
  (1/6 : ℚ) / 2 + (1/4 : ℚ) / 2 = 5/24 := by sorry

end NUMINAMATH_CALUDE_halfway_between_one_sixth_and_one_fourth_l2940_294018


namespace NUMINAMATH_CALUDE_dvd_cost_l2940_294085

/-- Given that two identical DVDs cost $40, prove that five DVDs cost $100. -/
theorem dvd_cost (cost_of_two : ℝ) (h : cost_of_two = 40) :
  5 / 2 * cost_of_two = 100 := by
  sorry

end NUMINAMATH_CALUDE_dvd_cost_l2940_294085


namespace NUMINAMATH_CALUDE_clara_stickers_l2940_294079

def stickers_left (initial : ℕ) (given_to_boy : ℕ) : ℕ := 
  (initial - given_to_boy) / 2

theorem clara_stickers : stickers_left 100 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_clara_stickers_l2940_294079


namespace NUMINAMATH_CALUDE_C_power_50_l2940_294065

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 4; -8, -10]

theorem C_power_50 : C^50 = !![201, 200; -400, -449] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l2940_294065


namespace NUMINAMATH_CALUDE_correct_product_l2940_294061

theorem correct_product (x : ℝ) (h : 21 * x = 27 * x - 48) : 27 * x = 27 * x := by
  sorry

end NUMINAMATH_CALUDE_correct_product_l2940_294061


namespace NUMINAMATH_CALUDE_album_slots_equal_sum_of_photos_l2940_294028

/-- The number of photos brought by each person --/
def cristina_photos : ℕ := 7
def john_photos : ℕ := 10
def sarah_photos : ℕ := 9
def clarissa_photos : ℕ := 14

/-- The total number of slots in the photo album --/
def album_slots : ℕ := cristina_photos + john_photos + sarah_photos + clarissa_photos

/-- Theorem stating that the number of slots in the photo album
    is equal to the sum of photos brought by all four people --/
theorem album_slots_equal_sum_of_photos :
  album_slots = cristina_photos + john_photos + sarah_photos + clarissa_photos :=
by sorry

end NUMINAMATH_CALUDE_album_slots_equal_sum_of_photos_l2940_294028


namespace NUMINAMATH_CALUDE_inequality_proof_l2940_294076

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : a + 2*b + 3*c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2940_294076
