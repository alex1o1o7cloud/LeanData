import Mathlib

namespace spinner_direction_l4_405

-- Define the four cardinal directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define a function to rotate a direction
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  match (revolutions % 1).num.mod 4 with
  | 0 => d
  | 1 => match d with
    | Direction.North => Direction.East
    | Direction.East => Direction.South
    | Direction.South => Direction.West
    | Direction.West => Direction.North
  | 2 => match d with
    | Direction.North => Direction.South
    | Direction.East => Direction.West
    | Direction.South => Direction.North
    | Direction.West => Direction.East
  | 3 => match d with
    | Direction.North => Direction.West
    | Direction.East => Direction.North
    | Direction.South => Direction.East
    | Direction.West => Direction.South
  | _ => d  -- This case should never occur due to mod 4

theorem spinner_direction :
  let initial_direction := Direction.North
  let clockwise_rotation := 7/2  -- 3.5 revolutions
  let counterclockwise_rotation := 7/4  -- 1.75 revolutions
  let final_direction := rotate (rotate initial_direction clockwise_rotation) (-counterclockwise_rotation)
  final_direction = Direction.West := by
  sorry

end spinner_direction_l4_405


namespace distribute_four_to_three_l4_483

/-- The number of ways to distribute n students to k villages, where each village gets at least one student -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 students to 3 villages results in 36 different plans -/
theorem distribute_four_to_three : distribute_students 4 3 = 36 := by
  sorry

end distribute_four_to_three_l4_483


namespace weekly_diaper_sales_revenue_l4_468

/-- Represents the weekly diaper sales revenue calculation --/
theorem weekly_diaper_sales_revenue :
  let boxes_per_week : ℕ := 30
  let packs_per_box : ℕ := 40
  let diapers_per_pack : ℕ := 160
  let price_per_diaper : ℚ := 4
  let bundle_discount : ℚ := 0.05
  let special_discount : ℚ := 0.05
  let tax_rate : ℚ := 0.10

  let total_diapers : ℕ := boxes_per_week * packs_per_box * diapers_per_pack
  let base_revenue : ℚ := total_diapers * price_per_diaper
  let after_bundle_discount : ℚ := base_revenue * (1 - bundle_discount)
  let after_special_discount : ℚ := after_bundle_discount * (1 - special_discount)
  let final_revenue : ℚ := after_special_discount * (1 + tax_rate)

  final_revenue = 762432 :=
by sorry


end weekly_diaper_sales_revenue_l4_468


namespace range_of_k_for_inequality_l4_460

/-- Given functions f and g, prove the range of k for which g(x) ≥ k(x) holds. -/
theorem range_of_k_for_inequality (f g : ℝ → ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f x = Real.log x) →
  (∀ x : ℝ, g x = x - 1) →
  (∀ x : ℝ, x ≥ 0 → g x ≥ k * x) ↔ k ≤ 1 :=
by sorry

end range_of_k_for_inequality_l4_460


namespace sqrt_13_minus_3_bounds_l4_435

theorem sqrt_13_minus_3_bounds : 0 < Real.sqrt 13 - 3 ∧ Real.sqrt 13 - 3 < 1 := by
  sorry

end sqrt_13_minus_3_bounds_l4_435


namespace divisors_of_8820_multiple_of_3_and_5_l4_441

def number_of_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_8820_multiple_of_3_and_5 : 
  number_of_divisors 8820 = 18 := by sorry

end divisors_of_8820_multiple_of_3_and_5_l4_441


namespace grocer_sales_problem_l4_429

/-- Calculates the first month's sale given sales for the next 4 months and desired average -/
def first_month_sale (month2 month3 month4 month5 desired_average : ℕ) : ℕ :=
  5 * desired_average - (month2 + month3 + month4 + month5)

/-- Proves that the first month's sale is 6790 given the problem conditions -/
theorem grocer_sales_problem : 
  first_month_sale 5660 6200 6350 6500 6300 = 6790 := by
  sorry

#eval first_month_sale 5660 6200 6350 6500 6300

end grocer_sales_problem_l4_429


namespace special_natural_numbers_l4_445

theorem special_natural_numbers : 
  {x : ℕ | ∃ (y z : ℤ), x = 2 * y^2 - 1 ∧ x^2 = 2 * z^2 - 1} = {1, 7} := by
  sorry

end special_natural_numbers_l4_445


namespace banana_bread_pieces_l4_492

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of banana bread -/
structure BananaBreadPan where
  dimensions : Dimensions

/-- Represents a piece of banana bread -/
structure BananaBreadPiece where
  dimensions : Dimensions

/-- Calculates the number of pieces that can be cut from a pan -/
def num_pieces (pan : BananaBreadPan) (piece : BananaBreadPiece) : ℕ :=
  (area pan.dimensions) / (area piece.dimensions)

theorem banana_bread_pieces : 
  let pan := BananaBreadPan.mk (Dimensions.mk 24 20)
  let piece := BananaBreadPiece.mk (Dimensions.mk 3 4)
  num_pieces pan piece = 40 := by
  sorry

end banana_bread_pieces_l4_492


namespace circle_radius_c_value_l4_477

theorem circle_radius_c_value :
  ∀ c : ℝ,
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ↔ (x + 4)^2 + (y + 1)^2 = 25) →
  c = -8 :=
by
  sorry

end circle_radius_c_value_l4_477


namespace solution_satisfies_system_l4_434

theorem solution_satisfies_system :
  let x : ℚ := 7/2
  let y : ℚ := 1/2
  (2 * x + 4 * y = 9) ∧ (3 * x - 5 * y = 8) := by
  sorry

end solution_satisfies_system_l4_434


namespace perpendicular_lines_from_parallel_planes_l4_479

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (l m : Line) (α β : Plane) 
  (h1 : parallel_plane α β) 
  (h2 : perpendicular_line_plane l α) 
  (h3 : parallel_line_plane m β) : 
  perpendicular_line l m :=
sorry

end perpendicular_lines_from_parallel_planes_l4_479


namespace actual_speed_is_30_l4_494

/-- Given:
  1. Increasing speed by 10 mph reduces time by 1/4
  2. Increasing speed by 20 mph reduces time by an additional 1/3
  Prove that the actual average speed is 30 mph
-/
theorem actual_speed_is_30 (v : ℝ) (t : ℝ) (d : ℝ) :
  (d = v * t) →
  (d / (v + 10) = 3 / 4 * t) →
  (d / (v + 20) = 1 / 2 * t) →
  v = 30 := by
  sorry

end actual_speed_is_30_l4_494


namespace operations_for_106_triangles_l4_499

/-- The number of triangles after n operations -/
def num_triangles (n : ℕ) : ℕ := 4 + 3 * (n - 1)

theorem operations_for_106_triangles :
  ∃ n : ℕ, n > 0 ∧ num_triangles n = 106 ∧ n = 35 := by sorry

end operations_for_106_triangles_l4_499


namespace new_person_weight_l4_450

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (leaving_weight : ℝ) (average_increase : ℝ) : ℝ :=
  leaving_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 65 kg -/
theorem new_person_weight :
  weight_of_new_person 8 45 2.5 = 65 := by
  sorry

#eval weight_of_new_person 8 45 2.5

end new_person_weight_l4_450


namespace fraction_simplification_l4_443

theorem fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (2 / y) / (3 / x^2) = 3 / 2 := by
  sorry

end fraction_simplification_l4_443


namespace max_sum_of_factors_l4_417

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A * B * C = 1764 →
  A + B + C ≤ 33 :=
by sorry

end max_sum_of_factors_l4_417


namespace new_person_weight_l4_475

theorem new_person_weight (n : ℕ) (avg_increase weight_replaced : ℝ) :
  n = 7 →
  avg_increase = 6.2 →
  weight_replaced = 76 →
  n * avg_increase + weight_replaced = 119.4 :=
by
  sorry

end new_person_weight_l4_475


namespace consumer_installment_credit_l4_463

theorem consumer_installment_credit (total_credit : ℝ) : 
  (0.36 * total_credit = 3 * 57) → total_credit = 475 := by
  sorry

end consumer_installment_credit_l4_463


namespace reading_time_difference_l4_412

/-- Proves that given Xanthia's and Molly's reading speeds and a book's page count,
    the difference in reading time is 240 minutes. -/
theorem reading_time_difference
  (xanthia_speed : ℕ)
  (molly_speed : ℕ)
  (book_pages : ℕ)
  (h1 : xanthia_speed = 80)
  (h2 : molly_speed = 40)
  (h3 : book_pages = 320) :
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 240 :=
by sorry

end reading_time_difference_l4_412


namespace simplify_fraction_l4_424

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end simplify_fraction_l4_424


namespace common_external_tangent_y_intercept_is_11_l4_490

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the y-intercept of the common external tangent to two circles -/
noncomputable def commonExternalTangentYIntercept (c1 c2 : Circle) : ℝ :=
  sorry

/-- Theorem stating that the y-intercept of the common external tangent is 11 -/
theorem common_external_tangent_y_intercept_is_11 :
  let c1 : Circle := { center := (1, 3), radius := 3 }
  let c2 : Circle := { center := (10, 6), radius := 5 }
  commonExternalTangentYIntercept c1 c2 = 11 := by
  sorry

end common_external_tangent_y_intercept_is_11_l4_490


namespace saree_original_price_l4_448

theorem saree_original_price (P : ℝ) : 
  (P * (1 - 0.2) * (1 - 0.3) = 313.6) → P = 560 := by
  sorry

end saree_original_price_l4_448


namespace sufficient_not_necessary_l4_481

/-- The function f(x) = ax^2 + 4x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x - 1

/-- Predicate indicating that the graph of f has only one common point with the x-axis -/
def has_one_common_point (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Statement: a = -4 is a sufficient but not necessary condition for
    the graph of f to have only one common point with the x-axis -/
theorem sufficient_not_necessary :
  (has_one_common_point (-4)) ∧ 
  (∃ a : ℝ, a ≠ -4 ∧ has_one_common_point a) :=
sorry

end sufficient_not_necessary_l4_481


namespace train_speed_with_36_coaches_l4_427

/-- Represents the speed of a train given the number of coaches attached. -/
noncomputable def train_speed (initial_speed : ℝ) (k : ℝ) (coaches : ℝ) : ℝ :=
  initial_speed - k * Real.sqrt coaches

/-- The theorem states that given the initial conditions, 
    the speed of the train with 36 coaches is 48 kmph. -/
theorem train_speed_with_36_coaches 
  (initial_speed : ℝ) 
  (k : ℝ) 
  (speed_reduction : ∀ (c : ℝ), train_speed initial_speed k c = initial_speed - k * Real.sqrt c) 
  (h1 : initial_speed = 60) 
  (h2 : train_speed initial_speed k 36 = 48) :
  train_speed initial_speed k 36 = 48 := by
  sorry

#check train_speed_with_36_coaches

end train_speed_with_36_coaches_l4_427


namespace collinear_points_m_value_l4_452

/-- Given a line containing points (2, 9), (10, m), and (25, 4), prove that m = 167/23 -/
theorem collinear_points_m_value : 
  ∀ m : ℚ, 
  (∃ (line : Set (ℚ × ℚ)), 
    (2, 9) ∈ line ∧ 
    (10, m) ∈ line ∧ 
    (25, 4) ∈ line ∧ 
    (∀ (x y z : ℚ × ℚ), x ∈ line → y ∈ line → z ∈ line → 
      (z.2 - y.2) * (y.1 - x.1) = (y.2 - x.2) * (z.1 - y.1))) →
  m = 167 / 23 := by
  sorry

end collinear_points_m_value_l4_452


namespace quadratic_discriminant_l4_471

theorem quadratic_discriminant :
  let a : ℚ := 5
  let b : ℚ := 5 + 1/5
  let c : ℚ := 1/5
  let discriminant := b^2 - 4*a*c
  discriminant = 576/25 := by sorry

end quadratic_discriminant_l4_471


namespace min_k_value_l4_469

theorem min_k_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ k : ℝ, (1/a + 1/b + k/(a+b) ≥ 0)) : 
  ∃ k_min : ℝ, k_min = -4 ∧ ∀ k : ℝ, (1/a + 1/b + k/(a+b) ≥ 0) → k ≥ k_min :=
sorry

end min_k_value_l4_469


namespace hat_count_l4_482

/-- The number of hats in the box -/
def num_hats : ℕ := 3

/-- The set of all hats in the box -/
def Hats : Type := Fin num_hats

/-- A hat is red -/
def is_red : Hats → Prop := sorry

/-- A hat is blue -/
def is_blue : Hats → Prop := sorry

/-- A hat is yellow -/
def is_yellow : Hats → Prop := sorry

/-- All but 2 hats are red -/
axiom red_condition : ∃ (a b : Hats), a ≠ b ∧ ∀ (h : Hats), h ≠ a ∧ h ≠ b → is_red h

/-- All but 2 hats are blue -/
axiom blue_condition : ∃ (a b : Hats), a ≠ b ∧ ∀ (h : Hats), h ≠ a ∧ h ≠ b → is_blue h

/-- All but 2 hats are yellow -/
axiom yellow_condition : ∃ (a b : Hats), a ≠ b ∧ ∀ (h : Hats), h ≠ a ∧ h ≠ b → is_yellow h

/-- The main theorem: There are exactly 3 hats in the box -/
theorem hat_count : num_hats = 3 := by sorry

end hat_count_l4_482


namespace piggy_bank_savings_l4_432

/-- Calculates the remaining amount in a piggy bank after a year of regular spending -/
theorem piggy_bank_savings (initial_amount : ℕ) (spending_per_trip : ℕ) (trips_per_month : ℕ) (months_per_year : ℕ) :
  initial_amount = 200 →
  spending_per_trip = 2 →
  trips_per_month = 4 →
  months_per_year = 12 →
  initial_amount - (spending_per_trip * trips_per_month * months_per_year) = 104 := by
  sorry

end piggy_bank_savings_l4_432


namespace special_line_equation_l4_411

/-- A line passing through a point and intersecting coordinate axes at points with negative reciprocal intercepts -/
structure SpecialLine where
  a : ℝ × ℝ  -- The point A that the line passes through
  eq : ℝ → ℝ → Prop  -- The equation of the line

/-- The condition for the line to have negative reciprocal intercepts -/
def hasNegativeReciprocalIntercepts (l : SpecialLine) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (l.eq k 0 ∧ l.eq 0 (-k) ∨ l.eq (-k) 0 ∧ l.eq 0 k)

/-- The main theorem stating the equation of the special line -/
theorem special_line_equation (l : SpecialLine) 
    (h1 : l.a = (5, 2))
    (h2 : hasNegativeReciprocalIntercepts l) :
    (∀ x y, l.eq x y ↔ 2*x - 5*y = -8) ∨
    (∀ x y, l.eq x y ↔ x - y = 3) := by
  sorry


end special_line_equation_l4_411


namespace train_passing_platform_l4_431

/-- Given a train of length 240 meters passing a pole in 24 seconds,
    prove that it takes 89 seconds to pass a platform of length 650 meters. -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (pole_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 240)
  (h2 : pole_passing_time = 24)
  (h3 : platform_length = 650) :
  (train_length + platform_length) / (train_length / pole_passing_time) = 89 :=
by sorry

end train_passing_platform_l4_431


namespace coefficient_x_cubed_in_binomial_expansion_l4_449

theorem coefficient_x_cubed_in_binomial_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) * (1 ^ (5 - k)) * (1 ^ k) * (if k = 3 then 1 else 0)) = 10 := by
  sorry

end coefficient_x_cubed_in_binomial_expansion_l4_449


namespace max_xy_value_fraction_inequality_l4_470

-- Part I
theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 5 * y = 20) :
  ∃ (max_val : ℝ), max_val = 10 ∧ ∀ (z : ℝ), x * y ≤ z → z ≤ max_val :=
sorry

-- Part II
theorem fraction_inequality (a b c d k : ℝ) (hab : a > b) (hb : b > 0) (hcd : c < d) (hd : d < 0) (hk : k < 0) :
  k / (a - c) > k / (b - d) :=
sorry

end max_xy_value_fraction_inequality_l4_470


namespace remainder_theorem_l4_425

theorem remainder_theorem (x : ℤ) (h : (x + 2) % 45 = 7) : 
  ((x + 2) % 20 = 7) ∧ (x % 19 = 5) := by
  sorry

end remainder_theorem_l4_425


namespace three_numbers_sum_l4_456

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ordering of numbers
  b = 10 ∧  -- Median is 10
  (a + b + c) / 3 = a + 20 ∧  -- Mean is 20 more than least
  (a + b + c) / 3 = c - 25  -- Mean is 25 less than greatest
  → a + b + c = 45 := by
  sorry

end three_numbers_sum_l4_456


namespace function_value_at_a_plus_one_l4_438

theorem function_value_at_a_plus_one (a : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + 1
  f (a + 1) = a^2 + 2*a + 2 := by
  sorry

end function_value_at_a_plus_one_l4_438


namespace inequality_proof_l4_444

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by sorry

end inequality_proof_l4_444


namespace zeros_sum_greater_than_twice_intersection_l4_453

noncomputable section

variables (a : ℝ) (x₁ x₂ x₀ : ℝ)

def f (x : ℝ) : ℝ := a * Real.log x - (1/2) * x^2

theorem zeros_sum_greater_than_twice_intersection
  (h₁ : a > Real.exp 1)
  (h₂ : f a x₁ = 0)
  (h₃ : f a x₂ = 0)
  (h₄ : x₁ ≠ x₂)
  (h₅ : x₀ = (x₁ + x₂) / ((a / (x₁ * x₂)) + 1)) :
  x₁ + x₂ > 2 * x₀ := by
  sorry

end zeros_sum_greater_than_twice_intersection_l4_453


namespace parallelogram_area_l4_489

theorem parallelogram_area (base height : ℝ) (h1 : base = 36) (h2 : height = 18) :
  base * height = 648 := by
  sorry

end parallelogram_area_l4_489


namespace intersection_of_three_lines_l4_495

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y : ℝ) :
  (y = 4 * x + 3) ∧ 
  (y = -2 * x - 25) ∧ 
  (y = 3 * x + k) →
  k = -5/3 := by
  sorry

end intersection_of_three_lines_l4_495


namespace power_zero_eq_one_l4_421

theorem power_zero_eq_one (a b : ℝ) (h : a - b ≠ 0) : (a - b)^0 = 1 := by
  sorry

end power_zero_eq_one_l4_421


namespace quadratic_equation_solutions_l4_451

theorem quadratic_equation_solutions
  (a b c : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a - b + c = 0)
  (h3 : a ≠ 0) :
  ∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -1 :=
by sorry

end quadratic_equation_solutions_l4_451


namespace vector_BA_l4_403

def complex_vector (a b : ℂ) : ℂ := a - b

theorem vector_BA (OA OB : ℂ) :
  OA = 2 - 3*I ∧ OB = -3 + 2*I →
  complex_vector OA OB = 5 - 5*I :=
by sorry

end vector_BA_l4_403


namespace total_spent_on_games_l4_440

def batman_game_cost : ℚ := 13.60
def superman_game_cost : ℚ := 5.06

theorem total_spent_on_games : batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end total_spent_on_games_l4_440


namespace fourth_vertex_of_rectangle_l4_426

/-- Given a circle and three points forming part of a rectangle, 
    this theorem proves the coordinates of the fourth vertex. -/
theorem fourth_vertex_of_rectangle 
  (O : ℝ × ℝ) (R : ℝ) 
  (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) 
  (h_circle : (x₁ - O.1)^2 + (y₁ - O.2)^2 = R^2 ∧ (x₂ - O.1)^2 + (y₂ - O.2)^2 = R^2)
  (h_inside : (x₀ - O.1)^2 + (y₀ - O.2)^2 < R^2) :
  ∃ (x₄ y₄ : ℝ), 
    (x₄ = x₁ + x₂ - x₀ ∧ y₄ = y₁ + y₂ - y₀) ∧
    ((x₄ - O.1)^2 + (y₄ - O.2)^2 = R^2) ∧
    ((x₄ - x₀)^2 + (y₄ - y₀)^2 = (x₁ - x₂)^2 + (y₁ - y₂)^2) :=
by sorry

end fourth_vertex_of_rectangle_l4_426


namespace min_value_trigonometric_function_solution_set_quadratic_inequality_l4_488

-- Problem 1
theorem min_value_trigonometric_function (x : ℝ) (hx : 0 < x ∧ x < π / 2) :
  (1 / Real.sin x ^ 2) + (4 / Real.cos x ^ 2) ≥ 9 :=
sorry

-- Problem 2
theorem solution_set_quadratic_inequality (a b c α β : ℝ) 
  (h_sol : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β)
  (h_pos : 0 < α ∧ α < β) :
  ∀ x, c * x^2 + b * x + a < 0 ↔ x < 1/β ∨ x > 1/α :=
sorry

end min_value_trigonometric_function_solution_set_quadratic_inequality_l4_488


namespace group_b_inspected_products_group_b_inspectors_l4_472

-- Define the number of workshops
def num_workshops : ℕ := 9

-- Define the number of inspectors in Group A
def group_a_inspectors : ℕ := 8

-- Define the initial number of finished products per workshop
variable (a : ℕ)

-- Define the daily production of finished products per workshop
variable (b : ℕ)

-- Define the number of days Group A inspects workshops 1 and 2
def days_group_a_1_2 : ℕ := 2

-- Define the number of days Group A inspects workshops 3 and 4
def days_group_a_3_4 : ℕ := 3

-- Define the total number of days for inspection
def total_inspection_days : ℕ := 5

-- Define the number of workshops inspected by Group B
def workshops_group_b : ℕ := 5

-- Theorem for the total number of finished products inspected by Group B
theorem group_b_inspected_products (a b : ℕ) :
  workshops_group_b * a + workshops_group_b * total_inspection_days * b = 5 * a + 25 * b :=
sorry

-- Theorem for the number of inspectors in Group B
theorem group_b_inspectors (a b : ℕ) (h : a = 4 * b) :
  (workshops_group_b * a + workshops_group_b * total_inspection_days * b) /
  ((3 / 4 : ℚ) * b * total_inspection_days) = 12 :=
sorry

end group_b_inspected_products_group_b_inspectors_l4_472


namespace symmetric_point_complex_l4_420

def symmetric_about_imaginary_axis (z : ℂ) : ℂ := -Complex.re z + Complex.im z * Complex.I

theorem symmetric_point_complex : 
  let A : ℂ := 2 + Complex.I
  let B : ℂ := symmetric_about_imaginary_axis A
  B = -2 + Complex.I := by
sorry

end symmetric_point_complex_l4_420


namespace min_mines_is_23_l4_414

/-- Represents the state of a square in the minesweeper grid -/
inductive SquareState
  | Unopened
  | Opened (n : Nat)

/-- Represents the minesweeper grid -/
def MinesweeperGrid := Matrix (Fin 11) (Fin 13) SquareState

/-- Checks if a given position is valid on the grid -/
def isValidPosition (row : Fin 11) (col : Fin 13) : Bool := true

/-- Returns the number of mines in the neighboring squares -/
def neighboringMines (grid : MinesweeperGrid) (row : Fin 11) (col : Fin 13) : Nat :=
  sorry

/-- Checks if the grid satisfies all opened square conditions -/
def satisfiesConditions (grid : MinesweeperGrid) : Prop :=
  sorry

/-- Counts the total number of mines in the grid -/
def countMines (grid : MinesweeperGrid) : Nat :=
  sorry

/-- The specific minesweeper grid layout from the problem -/
def problemGrid : MinesweeperGrid :=
  sorry

/-- Theorem stating that the minimum number of mines is 23 -/
theorem min_mines_is_23 :
  ∀ (grid : MinesweeperGrid),
    satisfiesConditions grid →
    grid = problemGrid →
    countMines grid ≥ 23 ∧
    ∃ (minGrid : MinesweeperGrid),
      satisfiesConditions minGrid ∧
      minGrid = problemGrid ∧
      countMines minGrid = 23 :=
sorry

end min_mines_is_23_l4_414


namespace power_multiplication_problem_solution_l4_484

theorem power_multiplication (n : ℕ) : n * (n ^ n) = n ^ (n + 1) := by sorry

theorem problem_solution : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  apply power_multiplication

end power_multiplication_problem_solution_l4_484


namespace min_value_expression_min_value_achievable_l4_486

theorem min_value_expression (x y : ℝ) : x^2 + y^2 - 8*x - 6*y + 30 ≥ 5 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, x^2 + y^2 - 8*x - 6*y + 30 = 5 := by
  sorry

end min_value_expression_min_value_achievable_l4_486


namespace fence_painting_combinations_l4_462

def number_of_colors : ℕ := 5
def number_of_tools : ℕ := 4

theorem fence_painting_combinations :
  number_of_colors * number_of_tools = 20 := by
  sorry

end fence_painting_combinations_l4_462


namespace consecutive_naturals_properties_l4_415

theorem consecutive_naturals_properties (n k : ℕ) (h : k > 0) :
  (∃ m ∈ Finset.range k, 2 ∣ (n + m)) ∧ 
  (k % 2 = 0 → 2 ∣ (k * n + k * (k - 1) / 2)) :=
sorry

end consecutive_naturals_properties_l4_415


namespace cube_root_sum_l4_413

theorem cube_root_sum (u v w : ℝ) : 
  (∃ x y z : ℝ, x^3 = 8 ∧ y^3 = 27 ∧ z^3 = 64 ∧
   (u - x) * (u - y) * (u - z) = 1/2 ∧
   (v - x) * (v - y) * (v - z) = 1/2 ∧
   (w - x) * (w - y) * (w - z) = 1/2 ∧
   u ≠ v ∧ u ≠ w ∧ v ≠ w) →
  u^3 + v^3 + w^3 = -42 := by
sorry

end cube_root_sum_l4_413


namespace dining_group_size_l4_461

theorem dining_group_size (total_bill : ℝ) (tip_percentage : ℝ) (individual_payment : ℝ) : 
  total_bill = 139 ∧ tip_percentage = 0.1 ∧ individual_payment = 50.97 →
  ∃ n : ℕ, n = 3 ∧ n * individual_payment = total_bill * (1 + tip_percentage) := by
sorry

end dining_group_size_l4_461


namespace negative_correlation_from_negative_coefficient_given_equation_negative_correlation_l4_458

/-- Represents a linear regression equation -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- Defines negative correlation between x and y -/
def negatively_correlated (eq : LinearRegression) : Prop :=
  eq.b < 0

/-- Theorem: If the coefficient of x in a linear regression equation is negative,
    then x and y are negatively correlated -/
theorem negative_correlation_from_negative_coefficient (eq : LinearRegression) :
  eq.b < 0 → negatively_correlated eq :=
by
  sorry

/-- The given empirical regression equation -/
def given_equation : LinearRegression :=
  { a := 2, b := -1 }

/-- Theorem: The given equation represents a negative correlation between x and y -/
theorem given_equation_negative_correlation :
  negatively_correlated given_equation :=
by
  sorry

end negative_correlation_from_negative_coefficient_given_equation_negative_correlation_l4_458


namespace perimeter_difference_l4_466

/-- Calculates the perimeter of a rectangle given its width and height. -/
def rectangle_perimeter (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width + height)

/-- Calculates the perimeter of a cross-shaped figure composed of 5 unit squares. -/
def cross_perimeter : ℕ := 8

/-- Theorem stating the difference between the perimeters of a 4x3 rectangle and a cross-shaped figure. -/
theorem perimeter_difference : 
  (rectangle_perimeter 4 3) - cross_perimeter = 6 := by sorry

end perimeter_difference_l4_466


namespace tank_emptying_l4_436

theorem tank_emptying (tank_capacity : ℝ) : 
  (3/4 * tank_capacity - 1/3 * tank_capacity = 15) → 
  (1/3 * tank_capacity = 12) :=
by sorry

end tank_emptying_l4_436


namespace kenneth_rowing_speed_l4_400

/-- Calculates the rowing speed of Kenneth given the race conditions -/
theorem kenneth_rowing_speed 
  (race_distance : ℝ) 
  (biff_speed : ℝ) 
  (kenneth_extra_distance : ℝ) 
  (h1 : race_distance = 500) 
  (h2 : biff_speed = 50) 
  (h3 : kenneth_extra_distance = 10) : 
  (race_distance + kenneth_extra_distance) / (race_distance / biff_speed) = 51 := by
  sorry

end kenneth_rowing_speed_l4_400


namespace cos_30_minus_cos_60_l4_491

theorem cos_30_minus_cos_60 :
  Real.cos (30 * π / 180) - Real.cos (60 * π / 180) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end cos_30_minus_cos_60_l4_491


namespace coin_flip_probability_is_two_elevenths_l4_442

/-- The probability of getting 4 consecutive heads before 3 consecutive tails
    when repeatedly flipping a fair coin -/
def coin_flip_probability : ℚ :=
  2/11

/-- Theorem stating that the probability of getting 4 consecutive heads
    before 3 consecutive tails when repeatedly flipping a fair coin is 2/11 -/
theorem coin_flip_probability_is_two_elevenths :
  coin_flip_probability = 2/11 := by
  sorry

end coin_flip_probability_is_two_elevenths_l4_442


namespace fraction_decimal_digits_l4_404

-- Define the fraction
def fraction : ℚ := 987654321 / (2^30 * 5^5)

-- Define the function to calculate the minimum number of decimal digits
def min_decimal_digits (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem fraction_decimal_digits :
  min_decimal_digits fraction = 30 := by sorry

end fraction_decimal_digits_l4_404


namespace claire_photos_l4_497

/-- Given that:
    - Lisa and Robert have taken the same number of photos
    - Lisa has taken 3 times as many photos as Claire
    - Robert has taken 28 more photos than Claire
    Prove that Claire has taken 14 photos. -/
theorem claire_photos (claire lisa robert : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 28) :
  claire = 14 := by
  sorry

end claire_photos_l4_497


namespace kody_half_age_of_mohamed_l4_433

def years_ago (mohamed_current_age kody_current_age : ℕ) : ℕ :=
  let x : ℕ := 4
  x

theorem kody_half_age_of_mohamed (mohamed_current_age kody_current_age : ℕ)
  (h1 : mohamed_current_age = 2 * 30)
  (h2 : kody_current_age = 32)
  (h3 : ∃ x : ℕ, kody_current_age - x = (mohamed_current_age - x) / 2) :
  years_ago mohamed_current_age kody_current_age = 4 := by
sorry

end kody_half_age_of_mohamed_l4_433


namespace complement_A_union_B_when_a_is_5_A_union_B_equals_A_iff_l4_430

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | a - 2 ≤ x ∧ x ≤ 2*a - 3}

-- Statement for part 1
theorem complement_A_union_B_when_a_is_5 :
  (Set.univ \ A) ∪ B 5 = {x : ℝ | x ≤ -3 ∨ x ≥ 3} := by sorry

-- Statement for part 2
theorem A_union_B_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ a < 9/2 := by sorry

end complement_A_union_B_when_a_is_5_A_union_B_equals_A_iff_l4_430


namespace nicky_pace_is_3_l4_455

/-- Nicky's pace in meters per second -/
def nicky_pace : ℝ := 3

/-- Cristina's pace in meters per second -/
def cristina_pace : ℝ := 5

/-- Head start given to Nicky in meters -/
def head_start : ℝ := 48

/-- Time it takes Cristina to catch up to Nicky in seconds -/
def catch_up_time : ℝ := 24

/-- Theorem stating that Nicky's pace is 3 meters per second given the conditions -/
theorem nicky_pace_is_3 :
  cristina_pace > nicky_pace ∧
  cristina_pace * catch_up_time = nicky_pace * catch_up_time + head_start →
  nicky_pace = 3 := by
  sorry


end nicky_pace_is_3_l4_455


namespace CH4_required_for_CCl4_l4_407

-- Define the chemical species as real numbers (representing moles)
variable (CH4 CH2Cl2 CCl4 CHCl3 HCl Cl2 CH3Cl : ℝ)

-- Define the equilibrium constants
def K1 : ℝ := 1.2 * 10^2
def K2 : ℝ := 1.5 * 10^3
def K3 : ℝ := 3.4 * 10^4

-- Define the initial amounts of species
def initial_CH2Cl2 : ℝ := 2.5
def initial_CHCl3 : ℝ := 1.5
def initial_HCl : ℝ := 0.5
def initial_Cl2 : ℝ := 10
def initial_CH3Cl : ℝ := 0.2

-- Define the target amount of CCl4
def target_CCl4 : ℝ := 5

-- Theorem statement
theorem CH4_required_for_CCl4 :
  ∃ (required_CH4 : ℝ),
    required_CH4 = 2.5 ∧
    required_CH4 + initial_CH2Cl2 = target_CCl4 :=
sorry

end CH4_required_for_CCl4_l4_407


namespace expression_evaluation_l4_409

theorem expression_evaluation : (1 + 2 + 3) * (1 + 1/2 + 1/3) = 11 := by
  sorry

end expression_evaluation_l4_409


namespace max_lg_product_l4_498

open Real

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := log x / log 10

-- State the theorem
theorem max_lg_product (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) 
  (h : lg x ^ 2 + lg y ^ 2 = lg (10 * x ^ 2) + lg (10 * y ^ 2)) :
  ∃ (max : ℝ), max = 2 + 2 * sqrt 2 ∧ lg (x * y) ≤ max := by
  sorry

end max_lg_product_l4_498


namespace constant_term_expansion_l4_473

/-- The constant term in the expansion of (1+2x^2)(x-1/x)^8 is -42 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (1 + 2*x^2) * (x - 1/x)^8
  ∃ g : ℝ → ℝ, (∀ x ≠ 0, f x = g x) ∧ g 0 = -42 :=
sorry

end constant_term_expansion_l4_473


namespace first_player_always_wins_l4_464

/-- Represents a cubic polynomial of the form x^3 + ax^2 + bx + c -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determines if a cubic polynomial has exactly one real root -/
def has_exactly_one_real_root (p : CubicPolynomial) : Prop :=
  ∃! x : ℝ, x^3 + p.a * x^2 + p.b * x + p.c = 0

/-- Represents a strategy for the first player -/
def first_player_strategy : CubicPolynomial → CubicPolynomial → CubicPolynomial :=
  sorry

/-- Represents a strategy for the second player -/
def second_player_strategy : CubicPolynomial → CubicPolynomial :=
  sorry

/-- The main theorem stating that the first player can always win -/
theorem first_player_always_wins :
  ∀ (second_strategy : CubicPolynomial → CubicPolynomial),
    ∃ (first_strategy : CubicPolynomial → CubicPolynomial → CubicPolynomial),
      ∀ (initial : CubicPolynomial),
        has_exactly_one_real_root (first_strategy initial (second_strategy initial)) :=
sorry

end first_player_always_wins_l4_464


namespace sum_of_xyz_l4_419

theorem sum_of_xyz (x y z : ℕ+) 
  (h1 : x * y = 18)
  (h2 : x * z = 3)
  (h3 : y * z = 6) :
  x + y + z = 10 := by
  sorry

end sum_of_xyz_l4_419


namespace units_digit_G_1000_l4_408

def G (n : ℕ) : ℕ := 2^(3^n) + 1

theorem units_digit_G_1000 : G 1000 % 10 = 3 := by sorry

end units_digit_G_1000_l4_408


namespace solution_values_l4_487

-- Define the solution sets
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the equation with parameters a and b
def equation (a b : ℝ) (x : ℝ) : Prop := x^2 + a*x + b = 0

-- Theorem statement
theorem solution_values :
  ∃ (a b : ℝ), A_intersect_B = {x | equation a b x ∧ x^2 + a*x + b < 0} ∧ a = -1 ∧ b = -2 := by
  sorry

end solution_values_l4_487


namespace smallest_number_with_remainder_l4_457

theorem smallest_number_with_remainder (n : ℕ) : 
  300 % 25 = 0 →
  324 > 300 ∧
  324 % 25 = 24 ∧
  ∀ m : ℕ, m > 300 ∧ m % 25 = 24 → m ≥ 324 :=
by sorry

end smallest_number_with_remainder_l4_457


namespace acute_triangle_contains_grid_point_l4_406

-- Define a point on a graph paper grid
structure GridPoint where
  x : ℤ
  y : ℤ

-- Define a triangle on a graph paper grid
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

-- Define what it means for a triangle to be acute
def isAcute (t : GridTriangle) : Prop :=
  sorry -- Definition of acute triangle on a grid

-- Define what it means for a point to be inside or on the sides of a triangle
def isInsideOrOnSides (p : GridPoint) (t : GridTriangle) : Prop :=
  sorry -- Definition of a point being inside or on the sides of a triangle

-- The main theorem
theorem acute_triangle_contains_grid_point (t : GridTriangle) :
  isAcute t →
  ∃ p : GridPoint, p ≠ t.A ∧ p ≠ t.B ∧ p ≠ t.C ∧ isInsideOrOnSides p t :=
sorry

end acute_triangle_contains_grid_point_l4_406


namespace fifty_second_card_is_ten_l4_485

def card_sequence : Fin 14 → String
| 0 => "A"
| 1 => "2"
| 2 => "3"
| 3 => "4"
| 4 => "5"
| 5 => "6"
| 6 => "7"
| 7 => "8"
| 8 => "9"
| 9 => "10"
| 10 => "J"
| 11 => "Q"
| 12 => "K"
| 13 => "Joker"

def nth_card (n : Nat) : String :=
  card_sequence (n % 14)

theorem fifty_second_card_is_ten :
  nth_card 51 = "10" := by
  sorry

end fifty_second_card_is_ten_l4_485


namespace product_of_largest_primes_l4_476

def largest_one_digit_primes : List Nat := [7, 5]
def largest_two_digit_prime : Nat := 97
def largest_three_digit_prime : Nat := 997

theorem product_of_largest_primes : 
  (List.prod largest_one_digit_primes) * largest_two_digit_prime * largest_three_digit_prime = 3383815 := by
  sorry

end product_of_largest_primes_l4_476


namespace min_omega_two_max_sine_l4_459

theorem min_omega_two_max_sine (ω : Real) : ω > 0 → (∃ x₁ x₂ : Real, 
  0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧ 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → Real.sin (ω * x) ≤ Real.sin (ω * x₁)) ∧
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → Real.sin (ω * x) ≤ Real.sin (ω * x₂))) → 
  ω ≥ 4 * Real.pi :=
by sorry

end min_omega_two_max_sine_l4_459


namespace f_properties_l4_480

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

-- State the theorem
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f a (-x) = -(f a x)) ∧
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 1) :=
sorry

end

end f_properties_l4_480


namespace unbroken_seashells_l4_418

theorem unbroken_seashells (total : ℕ) (broken : ℕ) (h1 : total = 7) (h2 : broken = 4) :
  total - broken = 3 := by
  sorry

end unbroken_seashells_l4_418


namespace factorization_theorem1_factorization_theorem2_l4_493

-- For the first expression
theorem factorization_theorem1 (x y : ℝ) :
  3 * (x + y) * (x - y) - (x - y)^2 = 2 * (x - y) * (x + 2*y) := by sorry

-- For the second expression
theorem factorization_theorem2 (x y : ℝ) :
  x^2 * (y^2 - 1) + 2*x * (y^2 - 1) = x * (y + 1) * (y - 1) * (x + 2) := by sorry

end factorization_theorem1_factorization_theorem2_l4_493


namespace brett_marbles_l4_402

theorem brett_marbles (red : ℕ) (blue : ℕ) : 
  blue = red + 24 → 
  blue = 5 * red → 
  red = 6 := by
  sorry

end brett_marbles_l4_402


namespace sum_of_altitudes_for_specific_line_l4_447

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Creates a triangle from a line and coordinate axes -/
def triangleFromLine (l : Line) : Triangle :=
  sorry

/-- Calculates the sum of altitudes of a triangle -/
def sumOfAltitudes (t : Triangle) : ℝ :=
  sorry

/-- The main theorem -/
theorem sum_of_altitudes_for_specific_line :
  let l : Line := { a := 15, b := 8, c := 120 }
  let t : Triangle := triangleFromLine l
  sumOfAltitudes t = 391 / 17 := by
  sorry

end sum_of_altitudes_for_specific_line_l4_447


namespace quadratic_no_real_roots_condition_l4_467

theorem quadratic_no_real_roots_condition (m x : ℝ) : 
  (∀ x, x^2 - 2*x + m ≠ 0) → m ≥ 0 ∧ 
  ∃ m₀ ≥ 0, ∃ x₀, x₀^2 - 2*x₀ + m₀ = 0 :=
by sorry

end quadratic_no_real_roots_condition_l4_467


namespace golden_ratio_trigonometry_l4_465

theorem golden_ratio_trigonometry (m n : ℝ) : 
  m = 2 * Real.sin (18 * π / 180) →
  m^2 + n = 4 →
  (m * Real.sqrt n) / (2 * (Real.cos (27 * π / 180))^2 - 1) = 2 := by
  sorry

end golden_ratio_trigonometry_l4_465


namespace inequality_proof_l4_474

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2*x + y + z)^2 * (2*x^2 + (y + z)^2) +
  (2*y + z + x)^2 * (2*y^2 + (z + x)^2) +
  (2*z + x + y)^2 * (2*z^2 + (x + y)^2) ≤ 8 :=
by sorry

end inequality_proof_l4_474


namespace opposite_of_reciprocal_of_negative_five_l4_478

theorem opposite_of_reciprocal_of_negative_five :
  -(1 / -5) = 1 / 5 := by sorry

end opposite_of_reciprocal_of_negative_five_l4_478


namespace vacuum_cost_proof_l4_496

/-- The cost of the vacuum cleaner Daria is saving for -/
def vacuum_cost : ℕ := 120

/-- The initial amount Daria has collected -/
def initial_amount : ℕ := 20

/-- The amount Daria adds to her savings each week -/
def weekly_savings : ℕ := 10

/-- The number of weeks Daria needs to save -/
def weeks_to_save : ℕ := 10

/-- Theorem stating that the vacuum cost is correct given the initial amount,
    weekly savings, and number of weeks to save -/
theorem vacuum_cost_proof :
  vacuum_cost = initial_amount + weekly_savings * weeks_to_save := by
  sorry

end vacuum_cost_proof_l4_496


namespace cube_volume_from_surface_area_l4_437

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), 
  s > 0 → 
  6 * s^2 = 150 → 
  s^3 = 125 :=
by
  sorry

end cube_volume_from_surface_area_l4_437


namespace probability_factor_less_than_eight_l4_428

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem probability_factor_less_than_eight :
  let f := factors 120
  (f.filter (· < 8)).card / f.card = 3 / 8 := by sorry

end probability_factor_less_than_eight_l4_428


namespace one_third_displayed_l4_439

/-- Represents an art gallery with paintings and sculptures -/
structure ArtGallery where
  total_pieces : ℕ
  displayed_pieces : ℕ
  displayed_sculptures : ℕ
  not_displayed_paintings : ℕ
  not_displayed_sculptures : ℕ

/-- Conditions for the art gallery problem -/
def gallery_conditions (g : ArtGallery) : Prop :=
  g.total_pieces = 900 ∧
  g.not_displayed_sculptures = 400 ∧
  g.displayed_sculptures = g.displayed_pieces / 6 ∧
  g.not_displayed_paintings = (g.total_pieces - g.displayed_pieces) / 3

/-- Theorem stating that 1/3 of the pieces are displayed -/
theorem one_third_displayed (g : ArtGallery) 
  (h : gallery_conditions g) : 
  g.displayed_pieces = g.total_pieces / 3 := by
  sorry

#check one_third_displayed

end one_third_displayed_l4_439


namespace negation_unique_solution_equivalence_l4_422

theorem negation_unique_solution_equivalence :
  ¬(∀ a : ℝ, ∃! x : ℝ, a * x + 1 = 0) ↔
  (∃ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ a * x + 1 = 0 ∧ a * y + 1 = 0) ∨ (∀ x : ℝ, a * x + 1 ≠ 0)) :=
by sorry

end negation_unique_solution_equivalence_l4_422


namespace michaels_brother_money_l4_401

theorem michaels_brother_money (michael_initial : ℕ) (brother_initial : ℕ) (candy_cost : ℕ) : 
  michael_initial = 42 →
  brother_initial = 17 →
  candy_cost = 3 →
  brother_initial + michael_initial / 2 - candy_cost = 35 :=
by
  sorry

end michaels_brother_money_l4_401


namespace election_winner_votes_l4_423

theorem election_winner_votes (total_votes : ℝ) (winner_votes : ℝ) : 
  (winner_votes = 0.62 * total_votes) →
  (winner_votes - (total_votes - winner_votes) = 384) →
  (winner_votes = 992) :=
by
  sorry

end election_winner_votes_l4_423


namespace deandre_jordan_free_throws_l4_454

/-- The probability of scoring at least one point in two free throw attempts -/
def prob_at_least_one_point (success_rate : ℝ) : ℝ :=
  1 - (1 - success_rate) ^ 2

theorem deandre_jordan_free_throws :
  let success_rate : ℝ := 0.4
  prob_at_least_one_point success_rate = 0.64 := by
  sorry

end deandre_jordan_free_throws_l4_454


namespace max_gcd_of_three_digit_numbers_l4_416

theorem max_gcd_of_three_digit_numbers :
  ∀ a b : ℕ,
  a ≠ b →
  a < 10 →
  b < 10 →
  (∃ (x y : ℕ), x = 100 * a + 11 * b ∧ y = 101 * b + 10 * a ∧ Nat.gcd x y ≤ 45) ∧
  (∃ (a' b' : ℕ), a' ≠ b' ∧ a' < 10 ∧ b' < 10 ∧
    Nat.gcd (100 * a' + 11 * b') (101 * b' + 10 * a') = 45) :=
by sorry

end max_gcd_of_three_digit_numbers_l4_416


namespace half_radius_circle_y_l4_410

-- Define the circles
def circle_x : Real → Prop := λ r => 2 * Real.pi * r = 14 * Real.pi
def circle_y : Real → Prop := λ r => True  -- We don't have specific information about y's circumference

-- Theorem statement
theorem half_radius_circle_y : 
  ∃ (rx ry : Real), 
    circle_x rx ∧ 
    circle_y ry ∧ 
    (Real.pi * rx^2 = Real.pi * ry^2) ∧  -- Same area
    (ry / 2 = 3.5) := by
  sorry

end half_radius_circle_y_l4_410


namespace sum_mod_seven_l4_446

theorem sum_mod_seven : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := by
  sorry

end sum_mod_seven_l4_446
