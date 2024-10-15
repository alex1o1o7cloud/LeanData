import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_planes_not_transitive_l1524_152457

-- Define the type for planes
variable (Plane : Type)

-- Define the perpendicular relation between planes
variable (perp : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_not_transitive :
  ∃ (α β γ : Plane),
    α ≠ β ∧ β ≠ γ ∧ α ≠ γ ∧
    perp α β ∧ perp β γ ∧
    ¬(∀ (α β γ : Plane), perp α β → perp β γ → perp α γ) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_not_transitive_l1524_152457


namespace NUMINAMATH_CALUDE_decorations_count_l1524_152422

/-- The number of pieces of tinsel in each box -/
def tinsel_per_box : ℕ := 4

/-- The number of Christmas trees in each box -/
def trees_per_box : ℕ := 1

/-- The number of snow globes in each box -/
def globes_per_box : ℕ := 5

/-- The number of families receiving a box -/
def families : ℕ := 11

/-- The number of boxes given to the community center -/
def community_boxes : ℕ := 1

/-- The total number of decorations handed out -/
def total_decorations : ℕ := (tinsel_per_box + trees_per_box + globes_per_box) * (families + community_boxes)

theorem decorations_count : total_decorations = 120 := by
  sorry

end NUMINAMATH_CALUDE_decorations_count_l1524_152422


namespace NUMINAMATH_CALUDE_wholesale_price_is_correct_l1524_152427

/-- The wholesale price of a pen -/
def wholesale_price : ℝ := 2.5

/-- The retail price of one pen -/
def retail_price_one : ℝ := 5

/-- The retail price of three pens -/
def retail_price_three : ℝ := 10

/-- Theorem stating that the wholesale price of a pen is 2.5 rubles -/
theorem wholesale_price_is_correct : 
  (retail_price_one - wholesale_price = retail_price_three - 3 * wholesale_price) ∧
  wholesale_price = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_wholesale_price_is_correct_l1524_152427


namespace NUMINAMATH_CALUDE_factorization_equality_l1524_152492

theorem factorization_equality (b : ℝ) : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1524_152492


namespace NUMINAMATH_CALUDE_variation_problem_l1524_152409

theorem variation_problem (c : ℝ) (R S T : ℝ → ℝ) (t : ℝ) :
  (∀ t, R t = c * (S t)^2 / (T t)^2) →
  R 0 = 2 ∧ S 0 = 1 ∧ T 0 = 2 →
  R t = 50 ∧ T t = 5 →
  S t = 12.5 := by
sorry

end NUMINAMATH_CALUDE_variation_problem_l1524_152409


namespace NUMINAMATH_CALUDE_parity_relation_l1524_152495

theorem parity_relation (a b : ℤ) : 
  (Even (5*b + a) → Even (a - 3*b)) ∧ 
  (Odd (5*b + a) → Odd (a - 3*b)) := by sorry

end NUMINAMATH_CALUDE_parity_relation_l1524_152495


namespace NUMINAMATH_CALUDE_profit_rate_change_is_three_percent_l1524_152487

/-- Represents the change in profit rate that causes A's income to increase by 300 --/
def profit_rate_change (a_share : ℚ) (a_capital : ℕ) (income_increase : ℕ) : ℚ :=
  (income_increase : ℚ) / a_capital / a_share * 100

/-- Theorem stating the change in profit rate given the problem conditions --/
theorem profit_rate_change_is_three_percent :
  profit_rate_change (2/3) 15000 300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_rate_change_is_three_percent_l1524_152487


namespace NUMINAMATH_CALUDE_cube_with_tunnel_surface_area_l1524_152444

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a tunnel drilled through it -/
structure CubeWithTunnel where
  sideLength : ℝ
  tunnelVertices : Fin 3 → Point3D

/-- Calculates the surface area of a cube with a tunnel drilled through it -/
def surfaceArea (cube : CubeWithTunnel) : ℝ := sorry

/-- The main theorem stating that the surface area of the cube with tunnel is 864 -/
theorem cube_with_tunnel_surface_area :
  ∃ (cube : CubeWithTunnel),
    cube.sideLength = 12 ∧
    (cube.tunnelVertices 0).x = 3 ∧ (cube.tunnelVertices 0).y = 0 ∧ (cube.tunnelVertices 0).z = 0 ∧
    (cube.tunnelVertices 1).x = 0 ∧ (cube.tunnelVertices 1).y = 12 ∧ (cube.tunnelVertices 1).z = 0 ∧
    (cube.tunnelVertices 2).x = 0 ∧ (cube.tunnelVertices 2).y = 0 ∧ (cube.tunnelVertices 2).z = 3 ∧
    surfaceArea cube = 864 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_tunnel_surface_area_l1524_152444


namespace NUMINAMATH_CALUDE_job_choice_diploma_percentage_l1524_152476

theorem job_choice_diploma_percentage :
  let total_population : ℝ := 100
  let no_diploma_with_job : ℝ := 12
  let with_job_choice : ℝ := 40
  let with_diploma : ℝ := 43
  let without_job_choice : ℝ := total_population - with_job_choice
  let with_diploma_and_job : ℝ := with_job_choice - no_diploma_with_job
  let with_diploma_without_job : ℝ := with_diploma - with_diploma_and_job
  (with_diploma_without_job / without_job_choice) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_job_choice_diploma_percentage_l1524_152476


namespace NUMINAMATH_CALUDE_problem_statement_l1524_152462

theorem problem_statement (a b c d : ℕ) 
  (h1 : d ∣ a^(2*b) + c) 
  (h2 : d ≥ a + c) : 
  d ≥ a + a^(1/(2*b)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1524_152462


namespace NUMINAMATH_CALUDE_tangent_line_passes_through_point_l1524_152493

/-- The function f(x) = ax³ + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_line_passes_through_point (a : ℝ) :
  (f_derivative a 1) * (2 - 1) + (f a 1) = 7 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_passes_through_point_l1524_152493


namespace NUMINAMATH_CALUDE_average_hamburgers_per_day_l1524_152489

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7

theorem average_hamburgers_per_day :
  (total_hamburgers : ℚ) / (days_in_week : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_hamburgers_per_day_l1524_152489


namespace NUMINAMATH_CALUDE_continuous_compound_interest_rate_l1524_152406

/-- Continuous compound interest rate calculation -/
theorem continuous_compound_interest_rate 
  (P : ℝ) -- Principal amount
  (A : ℝ) -- Total amount after interest
  (t : ℝ) -- Time in years
  (h1 : P = 600)
  (h2 : A = 760)
  (h3 : t = 4)
  : ∃ r : ℝ, (A = P * Real.exp (r * t)) ∧ (abs (r - 0.05909725) < 0.00000001) :=
sorry

end NUMINAMATH_CALUDE_continuous_compound_interest_rate_l1524_152406


namespace NUMINAMATH_CALUDE_lewis_age_l1524_152479

theorem lewis_age (ages : List Nat) 
  (h1 : ages = [4, 6, 8, 10, 12])
  (h2 : ∃ (a b : Nat), a ∈ ages ∧ b ∈ ages ∧ a + b = 18 ∧ a ≠ b)
  (h3 : ∃ (c d : Nat), c ∈ ages ∧ d ∈ ages ∧ c > 5 ∧ c < 11 ∧ d > 5 ∧ d < 11 ∧ c ≠ d)
  (h4 : 6 ∈ ages)
  (h5 : ∀ (x : Nat), x ∈ ages → x = 4 ∨ x = 6 ∨ x = 8 ∨ x = 10 ∨ x = 12) :
  4 ∈ ages := by
  sorry

end NUMINAMATH_CALUDE_lewis_age_l1524_152479


namespace NUMINAMATH_CALUDE_right_angled_figure_l1524_152498

def top_side (X : ℝ) : ℝ := 2 + 1 + 3 + X
def bottom_side : ℝ := 3 + 4 + 5

theorem right_angled_figure (X : ℝ) : 
  top_side X = bottom_side → X = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_figure_l1524_152498


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_m_range_l1524_152496

-- Define the sets A and B
def A (m : ℝ) := {x : ℝ | x^2 - 4*m*x + 2*m + 6 = 0}
def B := {x : ℝ | x < 0}

-- State the theorem
theorem intersection_nonempty_implies_m_range (m : ℝ) :
  (A m ∩ B).Nonempty → m ≤ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_m_range_l1524_152496


namespace NUMINAMATH_CALUDE_parabola_focus_line_intersection_l1524_152441

/-- Represents a parabola with equation y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line passing through the focus of a parabola -/
structure FocusLine where
  angle : ℝ
  h_angle_eq : angle = π / 4

/-- Represents the intersection points of a line with a parabola -/
structure Intersection where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The main theorem -/
theorem parabola_focus_line_intersection
  (para : Parabola) (line : FocusLine) (inter : Intersection) :
  let midpoint_x := (inter.A.1 + inter.B.1) / 2
  let axis_distance := midpoint_x - para.p / 2
  axis_distance = 4 → para.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_line_intersection_l1524_152441


namespace NUMINAMATH_CALUDE_inequality_proof_l1524_152470

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b > 1) : a * b < a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1524_152470


namespace NUMINAMATH_CALUDE_slope_product_theorem_l1524_152404

theorem slope_product_theorem (m n p : ℝ) : 
  m ≠ 0 ∧ n ≠ 0 ∧ p ≠ 0 →  -- none of the lines are horizontal
  (∃ θ₁ θ₂ θ₃ : ℝ, 
    θ₁ = 3 * θ₂ ∧  -- L₁ makes three times the angle with the horizontal as L₂
    θ₃ = θ₁ / 2 ∧  -- L₃ makes half the angle of L₁
    m = Real.tan θ₁ ∧ 
    n = Real.tan θ₂ ∧ 
    p = Real.tan θ₃) →
  m = 3 * n →  -- L₁ has 3 times the slope of L₂
  m = 5 * p →  -- L₁ has 5 times the slope of L₃
  m * n * p = Real.sqrt 3 / 15 := by
sorry

end NUMINAMATH_CALUDE_slope_product_theorem_l1524_152404


namespace NUMINAMATH_CALUDE_evaluate_expression_l1524_152491

theorem evaluate_expression : (2 : ℕ)^(3^2) + 1^(3^3) = 513 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1524_152491


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1524_152415

theorem hyperbola_focal_length : 
  let a : ℝ := Real.sqrt 10
  let b : ℝ := Real.sqrt 2
  let c : ℝ := Real.sqrt (a^2 + b^2)
  2 * c = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1524_152415


namespace NUMINAMATH_CALUDE_train_speed_proof_l1524_152461

/-- The speed of the train from city A -/
def speed_train_A : ℝ := 60

/-- The speed of the train from city B -/
def speed_train_B : ℝ := 75

/-- The total distance between cities A and B in km -/
def total_distance : ℝ := 465

/-- The time in hours that the train from A travels before meeting -/
def time_train_A : ℝ := 4

/-- The time in hours that the train from B travels before meeting -/
def time_train_B : ℝ := 3

theorem train_speed_proof : 
  speed_train_A * time_train_A + speed_train_B * time_train_B = total_distance :=
by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l1524_152461


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1524_152455

/-- The distance between the foci of the hyperbola xy = 2 is 2√2 -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ),
    (∀ (x y : ℝ), x * y = 2 → (x - f₁.1) * (y - f₁.2) = (x - f₂.1) * (y - f₂.2)) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1524_152455


namespace NUMINAMATH_CALUDE_not_in_S_iff_one_or_multiple_of_five_l1524_152482

def S : Set Nat := sorry

axiom two_in_S : 2 ∈ S

axiom square_in_S_implies_n_in_S : ∀ n : Nat, n^2 ∈ S → n ∈ S

axiom n_in_S_implies_n_plus_5_squared_in_S : ∀ n : Nat, n ∈ S → (n + 5)^2 ∈ S

axiom S_is_smallest : ∀ T : Set Nat, 
  (2 ∈ T ∧ 
   (∀ n : Nat, n^2 ∈ T → n ∈ T) ∧ 
   (∀ n : Nat, n ∈ T → (n + 5)^2 ∈ T)) → 
  S ⊆ T

theorem not_in_S_iff_one_or_multiple_of_five (n : Nat) :
  n ∉ S ↔ n = 1 ∨ ∃ k : Nat, n = 5 * k :=
sorry

end NUMINAMATH_CALUDE_not_in_S_iff_one_or_multiple_of_five_l1524_152482


namespace NUMINAMATH_CALUDE_max_charge_at_150_l1524_152431

-- Define the charge function
def charge (x : ℝ) : ℝ := 1000 * x - 5 * (x - 100)^2

-- State the theorem
theorem max_charge_at_150 :
  ∀ x ∈ Set.Icc 100 180,
    charge x ≤ charge 150 ∧
    charge 150 = 112500 := by
  sorry

-- Note: Set.Icc 100 180 represents the closed interval [100, 180]

end NUMINAMATH_CALUDE_max_charge_at_150_l1524_152431


namespace NUMINAMATH_CALUDE_marble_distribution_l1524_152416

theorem marble_distribution (capacity_second : ℝ) : 
  capacity_second > 0 →
  capacity_second + (3/4 * capacity_second) = 1050 →
  capacity_second = 600 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l1524_152416


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l1524_152453

theorem cos_2alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α - Real.cos α = Real.sqrt 5 / 5) : 
  Real.cos (2 * α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l1524_152453


namespace NUMINAMATH_CALUDE_water_in_large_bottle_sport_formulation_l1524_152448

/-- Represents a flavored drink formulation -/
structure Formulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard_formulation : Formulation :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_formulation : Formulation :=
  { flavoring := 1, corn_syrup := 4, water := 60 }

/-- The amount of corn syrup in the large bottle (in ounces) -/
def large_bottle_corn_syrup : ℚ := 8

/-- Theorem stating the amount of water in the large bottle of sport formulation -/
theorem water_in_large_bottle_sport_formulation :
  (large_bottle_corn_syrup * sport_formulation.water) / sport_formulation.corn_syrup = 120 := by
  sorry

end NUMINAMATH_CALUDE_water_in_large_bottle_sport_formulation_l1524_152448


namespace NUMINAMATH_CALUDE_jude_change_l1524_152483

def chair_price : ℕ := 13
def table_price : ℕ := 50
def plate_set_price : ℕ := 20
def num_chairs : ℕ := 3
def num_plate_sets : ℕ := 2
def total_paid : ℕ := 130

def total_cost : ℕ := chair_price * num_chairs + table_price + plate_set_price * num_plate_sets

theorem jude_change : 
  total_paid - total_cost = 1 :=
sorry

end NUMINAMATH_CALUDE_jude_change_l1524_152483


namespace NUMINAMATH_CALUDE_circular_pool_volume_l1524_152458

/-- The volume of a circular pool with given dimensions -/
theorem circular_pool_volume (diameter : ℝ) (depth1 : ℝ) (depth2 : ℝ) :
  diameter = 20 →
  depth1 = 3 →
  depth2 = 5 →
  (π * (diameter / 2)^2 * depth1 + π * (diameter / 2)^2 * depth2) = 800 * π := by
  sorry

end NUMINAMATH_CALUDE_circular_pool_volume_l1524_152458


namespace NUMINAMATH_CALUDE_cubic_real_root_l1524_152419

/-- Given a cubic polynomial with real coefficients c and d, 
    if -3 - 4i is a root, then the real root is 25/3 -/
theorem cubic_real_root (c d : ℝ) : 
  (c * (Complex.I ^ 3 + (-3 - 4*Complex.I) ^ 3) + 
   4 * (Complex.I ^ 2 + (-3 - 4*Complex.I) ^ 2) + 
   d * (Complex.I + (-3 - 4*Complex.I)) - 100 = 0) →
  (∃ (x : ℝ), c * x^3 + 4 * x^2 + d * x - 100 = 0 ∧ x = 25/3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_real_root_l1524_152419


namespace NUMINAMATH_CALUDE_not_right_triangle_l1524_152466

theorem not_right_triangle (a b c : ℕ) (h : a = 3 ∧ b = 4 ∧ c = 6) : 
  ¬(a^2 + b^2 = c^2) := by
  sorry

#check not_right_triangle

end NUMINAMATH_CALUDE_not_right_triangle_l1524_152466


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l1524_152437

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 →
  (a - (b - (c - (d + e))) = a - b - c - d + e) →
  e = 3 := by
sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l1524_152437


namespace NUMINAMATH_CALUDE_continuous_fraction_solution_l1524_152474

theorem continuous_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (6 + 2 * Real.sqrt 39) / 4 := by
  sorry

end NUMINAMATH_CALUDE_continuous_fraction_solution_l1524_152474


namespace NUMINAMATH_CALUDE_integer_solution_system_l1524_152469

theorem integer_solution_system :
  ∀ x y z : ℤ,
  (x * y + y * z + z * x = -4) →
  (x^2 + y^2 + z^2 = 24) →
  (x^3 + y^3 + z^3 + 3*x*y*z = 16) →
  ((x = 2 ∧ y = -2 ∧ z = 4) ∨
   (x = 2 ∧ y = 4 ∧ z = -2) ∨
   (x = -2 ∧ y = 2 ∧ z = 4) ∨
   (x = -2 ∧ y = 4 ∧ z = 2) ∨
   (x = 4 ∧ y = 2 ∧ z = -2) ∨
   (x = 4 ∧ y = -2 ∧ z = 2)) :=
by sorry


end NUMINAMATH_CALUDE_integer_solution_system_l1524_152469


namespace NUMINAMATH_CALUDE_decimal_division_subtraction_l1524_152452

theorem decimal_division_subtraction : (0.24 / 0.004) - 0.1 = 59.9 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_subtraction_l1524_152452


namespace NUMINAMATH_CALUDE_cats_sold_during_sale_l1524_152459

/-- The number of cats sold during a pet store sale -/
theorem cats_sold_during_sale 
  (siamese_initial : ℕ) 
  (house_initial : ℕ) 
  (cats_left : ℕ) 
  (h1 : siamese_initial = 13)
  (h2 : house_initial = 5)
  (h3 : cats_left = 8) :
  siamese_initial + house_initial - cats_left = 10 :=
by sorry

end NUMINAMATH_CALUDE_cats_sold_during_sale_l1524_152459


namespace NUMINAMATH_CALUDE_steve_juice_consumption_l1524_152447

theorem steve_juice_consumption (don_juice : ℚ) (steve_fraction : ℚ) :
  don_juice = 1/4 →
  steve_fraction = 3/4 →
  steve_fraction * don_juice = 3/16 := by
sorry

end NUMINAMATH_CALUDE_steve_juice_consumption_l1524_152447


namespace NUMINAMATH_CALUDE_runner_position_l1524_152426

theorem runner_position (track_circumference : ℝ) (distance_run : ℝ) : 
  track_circumference = 100 →
  distance_run = 10560 →
  ∃ (n : ℕ) (remainder : ℝ), 
    distance_run = n * track_circumference + remainder ∧
    75 < remainder ∧ remainder ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_runner_position_l1524_152426


namespace NUMINAMATH_CALUDE_eight_positions_l1524_152463

def number : ℚ := 38.82

theorem eight_positions (n : ℚ) (h : n = number) : 
  (n - 38 = 0.82) ∧ 
  (n - 38.8 = 0.02) :=
by sorry

end NUMINAMATH_CALUDE_eight_positions_l1524_152463


namespace NUMINAMATH_CALUDE_min_value_expression_l1524_152468

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) :
  (1 / a^2) + 2 * a^2 + 3 * b^2 + 4 * a * b ≥ Real.sqrt (8/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1524_152468


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l1524_152471

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 300) (h2 : cat_owners = 30) : 
  (cat_owners : ℚ) / total_students * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l1524_152471


namespace NUMINAMATH_CALUDE_condition_for_inequality_l1524_152475

theorem condition_for_inequality (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_condition_for_inequality_l1524_152475


namespace NUMINAMATH_CALUDE_opposite_of_abs_negative_2023_l1524_152494

theorem opposite_of_abs_negative_2023 : -(|-2023|) = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_abs_negative_2023_l1524_152494


namespace NUMINAMATH_CALUDE_saras_remaining_money_l1524_152424

/-- Calculates Sara's remaining money after her first paycheck and expenses --/
theorem saras_remaining_money :
  let week1_hours : ℕ := 40
  let week1_rate : ℚ := 11.5
  let week2_regular_hours : ℕ := 40
  let week2_overtime_hours : ℕ := 10
  let week2_rate : ℚ := 12
  let overtime_multiplier : ℚ := 1.5
  let sales : ℚ := 1000
  let commission_rate : ℚ := 0.05
  let tax_rate : ℚ := 0.15
  let insurance_cost : ℚ := 60
  let misc_fees : ℚ := 20
  let tire_cost : ℚ := 410

  let week1_earnings := week1_hours * week1_rate
  let week2_regular_earnings := week2_regular_hours * week2_rate
  let week2_overtime_earnings := week2_overtime_hours * (week2_rate * overtime_multiplier)
  let total_hourly_earnings := week1_earnings + week2_regular_earnings + week2_overtime_earnings
  let commission := sales * commission_rate
  let total_earnings := total_hourly_earnings + commission
  let taxes := total_earnings * tax_rate
  let total_deductions := taxes + insurance_cost + misc_fees
  let net_earnings := total_earnings - total_deductions
  let remaining_money := net_earnings - tire_cost

  remaining_money = 504.5 :=
by sorry

end NUMINAMATH_CALUDE_saras_remaining_money_l1524_152424


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l1524_152446

theorem mayoral_election_votes (candidate_x candidate_y other_candidate : ℕ) : 
  candidate_x = candidate_y + (candidate_y / 2) →
  candidate_y = other_candidate - (other_candidate * 2 / 5) →
  candidate_x = 22500 →
  other_candidate = 25000 := by
sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l1524_152446


namespace NUMINAMATH_CALUDE_tan_fifteen_fraction_equals_sqrt_three_over_three_l1524_152460

theorem tan_fifteen_fraction_equals_sqrt_three_over_three :
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_fraction_equals_sqrt_three_over_three_l1524_152460


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l1524_152490

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, 
  n > 0 ∧
  n % 6 = 2 ∧
  n % 7 = 3 ∧
  n % 8 = 4 ∧
  (∀ m : ℕ, m > 0 → m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m) ∧
  n = 164 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l1524_152490


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1524_152436

/-- A quadratic function with axis of symmetry at x = 1 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The axis of symmetry of f is at x = 1 -/
axiom axis_of_symmetry (b c : ℝ) : ∀ x, f b c (1 + x) = f b c (1 - x)

/-- The inequality f(1) < f(2) < f(-1) holds for the quadratic function f -/
theorem quadratic_inequality (b c : ℝ) : f b c 1 < f b c 2 ∧ f b c 2 < f b c (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1524_152436


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1524_152411

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) (k : ℝ) :
  isGeometricSequence a →
  a 5 * a 8 * a 11 = k →
  k^2 = a 5 * a 6 * a 7 * a 9 * a 10 * a 11 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1524_152411


namespace NUMINAMATH_CALUDE_ryan_analysis_time_l1524_152405

/-- The number of individuals Ryan is analyzing -/
def num_individuals : ℕ := 3

/-- The number of bones in each individual -/
def bones_per_individual : ℕ := 206

/-- The time (in hours) Ryan spends on initial analysis per bone -/
def initial_analysis_time : ℚ := 1

/-- The additional time (in hours) Ryan spends on research per bone -/
def additional_research_time : ℚ := 1/2

/-- The total time Ryan needs for his analysis -/
def total_analysis_time : ℚ :=
  (num_individuals * bones_per_individual) * (initial_analysis_time + additional_research_time)

theorem ryan_analysis_time : total_analysis_time = 927 := by
  sorry

end NUMINAMATH_CALUDE_ryan_analysis_time_l1524_152405


namespace NUMINAMATH_CALUDE_three_digit_permutation_sum_divisible_by_37_l1524_152423

theorem three_digit_permutation_sum_divisible_by_37 (a b c : ℕ) 
  (h1 : 0 < a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9) :
  37 ∣ (100*a + 10*b + c) + 
       (100*a + 10*c + b) + 
       (100*b + 10*a + c) + 
       (100*b + 10*c + a) + 
       (100*c + 10*a + b) + 
       (100*c + 10*b + a) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_permutation_sum_divisible_by_37_l1524_152423


namespace NUMINAMATH_CALUDE_max_min_product_l1524_152472

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 12) (h_prod_sum : x*y + y*z + z*x = 30) :
  ∃ (n : ℝ), n = min (x*y) (min (y*z) (z*x)) ∧ n ≤ 2 ∧ 
  ∀ (m : ℝ), (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = 12 ∧ a*b + b*c + c*a = 30 ∧ 
    m = min (a*b) (min (b*c) (c*a))) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l1524_152472


namespace NUMINAMATH_CALUDE_x_interval_l1524_152442

theorem x_interval (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -4) (h3 : 2*x - 1 > 0) : x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_interval_l1524_152442


namespace NUMINAMATH_CALUDE_simplify_expression_l1524_152497

theorem simplify_expression (x : ℝ) : (3 * x + 20) - (7 * x - 5) = -4 * x + 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1524_152497


namespace NUMINAMATH_CALUDE_percentage_difference_l1524_152478

theorem percentage_difference (x y : ℝ) (h : y = x + 0.6 * x) :
  (y - x) / y = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1524_152478


namespace NUMINAMATH_CALUDE_square_root_equality_l1524_152488

theorem square_root_equality (m : ℝ) (x : ℝ) (h1 : m > 0) 
  (h2 : Real.sqrt m = x + 1) (h3 : Real.sqrt m = 5 + 2*x) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l1524_152488


namespace NUMINAMATH_CALUDE_tan_cos_sum_identity_l1524_152433

theorem tan_cos_sum_identity : 
  Real.tan (30 * π / 180) * Real.cos (60 * π / 180) + 
  Real.tan (45 * π / 180) * Real.cos (30 * π / 180) = 
  2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_cos_sum_identity_l1524_152433


namespace NUMINAMATH_CALUDE_dans_initial_marbles_l1524_152408

/-- The number of marbles Dan gave to Mary -/
def marbles_given : ℕ := 14

/-- The number of marbles Dan has now -/
def marbles_remaining : ℕ := 50

/-- The initial number of marbles Dan had -/
def initial_marbles : ℕ := marbles_given + marbles_remaining

theorem dans_initial_marbles : initial_marbles = 64 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_marbles_l1524_152408


namespace NUMINAMATH_CALUDE_students_in_score_range_l1524_152418

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  total_students : ℕ
  mean : ℝ
  variance : ℝ
  prob_above_140 : ℝ

/-- Calculates the number of students within a given score range -/
def students_in_range (dist : ScoreDistribution) (lower upper : ℝ) : ℕ :=
  sorry

theorem students_in_score_range (dist : ScoreDistribution) 
  (h1 : dist.total_students = 50)
  (h2 : dist.mean = 120)
  (h3 : dist.prob_above_140 = 0.2) :
  students_in_range dist 100 140 = 30 :=
sorry

end NUMINAMATH_CALUDE_students_in_score_range_l1524_152418


namespace NUMINAMATH_CALUDE_divisibility_condition_l1524_152417

/-- A pair of positive integers (m, n) satisfies the divisibility condition if and only if
    it is of the form (k^2 + 1, k) or (k, k^2 + 1) for some positive integer k. -/
theorem divisibility_condition (m n : ℕ+) : 
  (∃ d : ℕ+, d * (m * n - 1) = (n^2 - n + 1)^2) ↔ 
  (∃ k : ℕ+, (m = k^2 + 1 ∧ n = k) ∨ (m = k ∧ n = k^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1524_152417


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1524_152421

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 4 →
  downstream_distance = 9.6 →
  downstream_time = 24 / 60 →
  ∃ (boat_speed : ℝ), boat_speed = 20 ∧ (boat_speed + current_speed) * downstream_time = downstream_distance :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1524_152421


namespace NUMINAMATH_CALUDE_smallest_positive_e_value_l1524_152429

theorem smallest_positive_e_value (a b c d e : ℤ) :
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -3 ∨ x = 4 ∨ x = 8 ∨ x = -1/4) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∃ a' b' c' d' : ℤ, ∀ x : ℚ, a' * x^4 + b' * x^3 + c' * x^2 + d' * x + e' = 0 ↔ 
      x = -3 ∨ x = 4 ∨ x = 8 ∨ x = -1/4) →
    e ≤ e') →
  e = 96 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_e_value_l1524_152429


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1524_152486

theorem fraction_equation_solution (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 5 →
    (P / (x + 7 : ℝ)) + (Q / (x^2 - 6*x : ℝ)) = ((x^2 - 6*x + 14) / (x^3 + x^2 - 30*x) : ℝ)) →
  (Q : ℚ) / (P : ℚ) = 12 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1524_152486


namespace NUMINAMATH_CALUDE_inequality_problem_l1524_152484

theorem inequality_problem :
  (∀ (x : ℝ), (∀ (m : ℝ), -2 ≤ m ∧ m ≤ 2 → 2 * x - 1 > m * (x^2 - 1)) ↔ 
    ((Real.sqrt 7 - 1) / 2 < x ∧ x < (Real.sqrt 3 + 1) / 2)) ∧
  (¬ ∃ (m : ℝ), ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → 2 * x - 1 > m * (x^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l1524_152484


namespace NUMINAMATH_CALUDE_quadratic_root_form_l1524_152438

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 + 6*x + d = 0 ↔ x = (-6 + Real.sqrt d) / 2 ∨ x = (-6 - Real.sqrt d) / 2) →
  d = 36/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l1524_152438


namespace NUMINAMATH_CALUDE_total_investment_proof_l1524_152425

def bank_investment : ℝ := 6000
def bond_investment : ℝ := 6000
def bank_interest_rate : ℝ := 0.05
def bond_return_rate : ℝ := 0.09
def annual_income : ℝ := 660

theorem total_investment_proof :
  bank_investment + bond_investment = 12000 :=
by sorry

end NUMINAMATH_CALUDE_total_investment_proof_l1524_152425


namespace NUMINAMATH_CALUDE_two_greater_than_negative_three_l1524_152456

theorem two_greater_than_negative_three : 2 > -3 := by
  sorry

end NUMINAMATH_CALUDE_two_greater_than_negative_three_l1524_152456


namespace NUMINAMATH_CALUDE_terrier_hush_interval_terrier_hush_interval_is_two_l1524_152434

/-- The interval at which a terrier's owner hushes it, given the following conditions:
  - The poodle barks twice for every one time the terrier barks.
  - The terrier's owner says "hush" six times before the dogs stop barking.
  - The poodle barked 24 times. -/
theorem terrier_hush_interval : ℕ :=
  let poodle_barks : ℕ := 24
  let poodle_to_terrier_ratio : ℕ := 2
  let total_hushes : ℕ := 6
  let terrier_barks : ℕ := poodle_barks / poodle_to_terrier_ratio
  terrier_barks / total_hushes

/-- Proof that the terrier_hush_interval is equal to 2 -/
theorem terrier_hush_interval_is_two : terrier_hush_interval = 2 := by
  sorry

end NUMINAMATH_CALUDE_terrier_hush_interval_terrier_hush_interval_is_two_l1524_152434


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1524_152413

/-- The equation of circle C is x^2 + y^2 + 8x + 15 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 8*x + 15 = 0

/-- The equation of the line is y = kx - 2 -/
def line (k x y : ℝ) : Prop := y = k*x - 2

/-- A point (x, y) is on the line y = kx - 2 -/
def point_on_line (k x y : ℝ) : Prop := line k x y

/-- The distance between two points (x1, y1) and (x2, y2) is less than or equal to r -/
def distance_le (x1 y1 x2 y2 r : ℝ) : Prop :=
  (x1 - x2)^2 + (y1 - y2)^2 ≤ r^2

theorem circle_line_intersection (k : ℝ) : 
  (∃ x y : ℝ, point_on_line k x y ∧ 
    (∃ x0 y0 : ℝ, circle_C x0 y0 ∧ distance_le x y x0 y0 1)) →
  -4/3 ≤ k ∧ k ≤ 0 := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1524_152413


namespace NUMINAMATH_CALUDE_centroid_quadrilateral_area_ratio_l1524_152430

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is inside a quadrilateral -/
def isInterior (p : Point) (q : Quadrilateral) : Prop := sorry

/-- Calculates the centroid of a triangle -/
def centroid (a b c : Point) : Point := sorry

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Main theorem -/
theorem centroid_quadrilateral_area_ratio 
  (ABCD : Quadrilateral) 
  (P : Point) 
  (h1 : isConvex ABCD) 
  (h2 : isInterior P ABCD) : 
  let G1 := centroid ABCD.A ABCD.B P
  let G2 := centroid ABCD.B ABCD.C P
  let G3 := centroid ABCD.C ABCD.D P
  let G4 := centroid ABCD.D ABCD.A P
  let centroidQuad : Quadrilateral := ⟨G1, G2, G3, G4⟩
  area centroidQuad / area ABCD = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_centroid_quadrilateral_area_ratio_l1524_152430


namespace NUMINAMATH_CALUDE_f_max_value_l1524_152450

noncomputable def f (x : ℝ) : ℝ := x^6 / (x^12 + 3*x^9 - 6*x^6 + 12*x^3 + 27)

theorem f_max_value :
  (∀ x : ℝ, f x ≤ 1 / (12 * Real.sqrt 3)) ∧
  (∃ x : ℝ, f x = 1 / (12 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_f_max_value_l1524_152450


namespace NUMINAMATH_CALUDE_equation_solution_l1524_152432

theorem equation_solution (x : ℝ) : 
  x ≠ 3 → ((2 - x) / (x - 3) = 1 / (x - 3) - 2) → x = 5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1524_152432


namespace NUMINAMATH_CALUDE_probability_b_greater_than_a_l1524_152480

def A : Finset ℕ := {2, 3, 4, 5, 6}
def B : Finset ℕ := {1, 2, 3, 5}

theorem probability_b_greater_than_a : 
  (Finset.filter (λ (p : ℕ × ℕ) => p.2 > p.1) (A.product B)).card / (A.card * B.card : ℚ) = 1/5 :=
sorry

end NUMINAMATH_CALUDE_probability_b_greater_than_a_l1524_152480


namespace NUMINAMATH_CALUDE_pedestrian_meets_cart_time_l1524_152402

/-- Represents a participant in the scenario -/
inductive Participant
| Pedestrian
| Cyclist
| Cart
| Car

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents an event involving two participants -/
structure Event where
  participant1 : Participant
  participant2 : Participant
  time : Time

/-- The scenario with all participants and events -/
structure Scenario where
  cyclist_overtakes_pedestrian : Event
  pedestrian_meets_car : Event
  cyclist_meets_cart : Event
  cyclist_meets_car : Event
  car_meets_cyclist : Event
  car_meets_pedestrian : Event
  car_overtakes_cart : Event

def is_valid_scenario (s : Scenario) : Prop :=
  s.cyclist_overtakes_pedestrian.time = Time.mk 10 0 ∧
  s.pedestrian_meets_car.time = Time.mk 11 0 ∧
  s.cyclist_meets_cart.time.hours - s.cyclist_overtakes_pedestrian.time.hours = 
    s.cyclist_meets_car.time.hours - s.cyclist_meets_cart.time.hours ∧
  s.cyclist_meets_cart.time.minutes - s.cyclist_overtakes_pedestrian.time.minutes = 
    s.cyclist_meets_car.time.minutes - s.cyclist_meets_cart.time.minutes ∧
  s.car_meets_pedestrian.time.hours - s.car_meets_cyclist.time.hours = 
    s.car_overtakes_cart.time.hours - s.car_meets_pedestrian.time.hours ∧
  s.car_meets_pedestrian.time.minutes - s.car_meets_cyclist.time.minutes = 
    s.car_overtakes_cart.time.minutes - s.car_meets_pedestrian.time.minutes

theorem pedestrian_meets_cart_time (s : Scenario) (h : is_valid_scenario s) :
  ∃ (t : Event), t.participant1 = Participant.Pedestrian ∧ 
                 t.participant2 = Participant.Cart ∧ 
                 t.time = Time.mk 10 40 :=
sorry

end NUMINAMATH_CALUDE_pedestrian_meets_cart_time_l1524_152402


namespace NUMINAMATH_CALUDE_missing_shirts_is_eight_l1524_152445

/-- Represents the laundry problem with given conditions -/
structure LaundryProblem where
  trousers_count : ℕ
  total_bill : ℕ
  shirt_cost : ℕ
  trouser_cost : ℕ
  claimed_shirts : ℕ

/-- Calculates the number of missing shirts -/
def missing_shirts (p : LaundryProblem) : ℕ :=
  let total_trouser_cost := p.trousers_count * p.trouser_cost
  let total_shirt_cost := p.total_bill - total_trouser_cost
  let actual_shirts := total_shirt_cost / p.shirt_cost
  actual_shirts - p.claimed_shirts

/-- Theorem stating that the number of missing shirts is 8 -/
theorem missing_shirts_is_eight :
  ∃ (p : LaundryProblem),
    p.trousers_count = 10 ∧
    p.total_bill = 140 ∧
    p.shirt_cost = 5 ∧
    p.trouser_cost = 9 ∧
    p.claimed_shirts = 2 ∧
    missing_shirts p = 8 := by
  sorry

end NUMINAMATH_CALUDE_missing_shirts_is_eight_l1524_152445


namespace NUMINAMATH_CALUDE_carnation_percentage_l1524_152473

/-- Represents a flower bouquet with various types of flowers -/
structure Bouquet where
  total : ℝ
  pink_roses : ℝ
  red_roses : ℝ
  pink_carnations : ℝ
  red_carnations : ℝ
  yellow_tulips : ℝ

/-- Conditions for the flower bouquet problem -/
def bouquet_conditions (b : Bouquet) : Prop :=
  b.pink_roses + b.red_roses + b.pink_carnations + b.red_carnations + b.yellow_tulips = b.total ∧
  b.pink_roses + b.pink_carnations = 0.4 * b.total ∧
  b.red_roses + b.red_carnations = 0.4 * b.total ∧
  b.yellow_tulips = 0.2 * b.total ∧
  b.pink_roses = (2/5) * (b.pink_roses + b.pink_carnations) ∧
  b.red_carnations = (1/2) * (b.red_roses + b.red_carnations)

/-- Theorem stating that the percentage of carnations is 44% -/
theorem carnation_percentage (b : Bouquet) (h : bouquet_conditions b) : 
  (b.pink_carnations + b.red_carnations) / b.total = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_carnation_percentage_l1524_152473


namespace NUMINAMATH_CALUDE_marathon_duration_in_minutes_l1524_152481

-- Define the duration of the marathon
def marathon_hours : ℕ := 15
def marathon_minutes : ℕ := 35

-- Theorem to prove
theorem marathon_duration_in_minutes :
  marathon_hours * 60 + marathon_minutes = 935 := by
  sorry

end NUMINAMATH_CALUDE_marathon_duration_in_minutes_l1524_152481


namespace NUMINAMATH_CALUDE_opposite_numbers_system_solution_l1524_152440

theorem opposite_numbers_system_solution :
  ∀ (x y k : ℝ),
  (x = -y) →
  (2 * x + 5 * y = k) →
  (x - 3 * y = 16) →
  (k = -12) :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_system_solution_l1524_152440


namespace NUMINAMATH_CALUDE_canoe_kayak_ratio_l1524_152435

/-- Represents the rental prices and quantities of canoes and kayaks --/
structure RentalInfo where
  canoePrice : ℕ
  kayakPrice : ℕ
  canoeCount : ℕ
  kayakCount : ℕ

/-- Calculates the total revenue from canoe and kayak rentals --/
def totalRevenue (info : RentalInfo) : ℕ :=
  info.canoePrice * info.canoeCount + info.kayakPrice * info.kayakCount

/-- Theorem stating the ratio of canoes to kayaks given the rental conditions --/
theorem canoe_kayak_ratio (info : RentalInfo) :
  info.canoePrice = 15 →
  info.kayakPrice = 18 →
  totalRevenue info = 405 →
  info.canoeCount = info.kayakCount + 5 →
  (info.canoeCount : ℚ) / info.kayakCount = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_canoe_kayak_ratio_l1524_152435


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1524_152454

theorem complex_fraction_simplification :
  (Complex.mk 3 (-5)) / (Complex.mk 2 (-7)) = Complex.mk (-41/45) (-11/45) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1524_152454


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l1524_152412

theorem discount_percentage_calculation (original_price : ℝ) 
  (john_tip_rate : ℝ) (jane_tip_rate : ℝ) (price_difference : ℝ) 
  (h1 : original_price = 24.00000000000002)
  (h2 : john_tip_rate = 0.15)
  (h3 : jane_tip_rate = 0.15)
  (h4 : price_difference = 0.36)
  (h5 : original_price * (1 + john_tip_rate) - 
        original_price * (1 - D) * (1 + jane_tip_rate) = price_difference) :
  D = price_difference / (original_price * (1 + john_tip_rate)) := by
sorry

#eval (0.36 / 27.600000000000024 : Float)

end NUMINAMATH_CALUDE_discount_percentage_calculation_l1524_152412


namespace NUMINAMATH_CALUDE_image_of_two_zero_l1524_152464

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Theorem statement
theorem image_of_two_zero :
  f (2, 0) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_image_of_two_zero_l1524_152464


namespace NUMINAMATH_CALUDE_boat_speed_l1524_152420

/-- Given a boat that travels 11 km/h downstream and 5 km/h upstream, 
    its speed in still water is 8 km/h. -/
theorem boat_speed (downstream upstream : ℝ) 
  (h1 : downstream = 11) 
  (h2 : upstream = 5) : 
  (downstream + upstream) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l1524_152420


namespace NUMINAMATH_CALUDE_equal_probability_for_first_ace_l1524_152428

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ace_count : ℕ)
  (hc : ace_count ≤ total_cards)

/-- Represents a card game -/
structure CardGame :=
  (deck : Deck)
  (player_count : ℕ)
  (hpc : player_count > 0)

/-- The probability of a player receiving the first Ace -/
def first_ace_probability (game : CardGame) (player : ℕ) : ℚ :=
  1 / game.player_count

/-- Theorem stating that in a specific card game, each player has an equal probability of receiving the first Ace -/
theorem equal_probability_for_first_ace (game : CardGame) 
    (h1 : game.deck.total_cards = 32) 
    (h2 : game.deck.ace_count = 4) 
    (h3 : game.player_count = 4) : 
    ∀ (player : ℕ), player > 0 → player ≤ game.player_count → 
    first_ace_probability game player = 1 / 8 := by
  sorry

#check equal_probability_for_first_ace

end NUMINAMATH_CALUDE_equal_probability_for_first_ace_l1524_152428


namespace NUMINAMATH_CALUDE_congruence_problem_l1524_152403

theorem congruence_problem (x : ℤ) : 
  (5 * x + 11) % 19 = 3 → (3 * x + 7) % 19 = 6 := by sorry

end NUMINAMATH_CALUDE_congruence_problem_l1524_152403


namespace NUMINAMATH_CALUDE_polygon_sides_proof_l1524_152400

theorem polygon_sides_proof (n : ℕ) : 
  let sides1 := n
  let sides2 := n + 4
  let sides3 := n + 12
  let sides4 := n + 13
  let diagonals (m : ℕ) := m * (m - 3) / 2
  diagonals sides1 + diagonals sides4 = diagonals sides2 + diagonals sides3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_proof_l1524_152400


namespace NUMINAMATH_CALUDE_money_redistribution_theorem_l1524_152499

/-- Represents the money redistribution problem with four friends --/
def MoneyRedistribution (a b j t : ℕ) : Prop :=
  -- Initial conditions
  a ≠ b ∧ a ≠ j ∧ a ≠ t ∧ b ≠ j ∧ b ≠ t ∧ j ≠ t ∧
  -- Toy starts and ends with the same amount
  t = 48 ∧
  -- After four rounds of redistribution
  ∃ (a₁ b₁ j₁ t₁ : ℕ),
    -- First round (Amy redistributes)
    a₁ + b₁ + j₁ + t₁ = a + b + j + t ∧
    b₁ = 2 * b ∧ j₁ = 2 * j ∧ t₁ = 2 * t ∧
    ∃ (a₂ b₂ j₂ t₂ : ℕ),
      -- Second round (Beth redistributes)
      a₂ + b₂ + j₂ + t₂ = a₁ + b₁ + j₁ + t₁ ∧
      a₂ = 2 * a₁ ∧ j₂ = 2 * j₁ ∧ t₂ = 2 * t₁ ∧
      ∃ (a₃ b₃ j₃ t₃ : ℕ),
        -- Third round (Jan redistributes)
        a₃ + b₃ + j₃ + t₃ = a₂ + b₂ + j₂ + t₂ ∧
        a₃ = 2 * a₂ ∧ b₃ = 2 * b₂ ∧ t₃ = 2 * t₂ ∧
        ∃ (a₄ b₄ j₄ t₄ : ℕ),
          -- Fourth round (Toy redistributes)
          a₄ + b₄ + j₄ + t₄ = a₃ + b₃ + j₃ + t₃ ∧
          a₄ = 2 * a₃ ∧ b₄ = 2 * b₃ ∧ j₄ = 2 * j₃ ∧
          -- Toy ends with the same amount
          t₄ = t

/-- The theorem stating that the total money is 15 times Toy's amount --/
theorem money_redistribution_theorem {a b j t : ℕ} (h : MoneyRedistribution a b j t) :
  a + b + j + t = 15 * t :=
sorry

end NUMINAMATH_CALUDE_money_redistribution_theorem_l1524_152499


namespace NUMINAMATH_CALUDE_inequality_condition_l1524_152485

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < -1 → (a*x + 1)*(1 + x) < 0) ∧
  (∃ x : ℝ, (a*x + 1)*(1 + x) < 0 ∧ (x ≤ -2 ∨ x ≥ -1)) →
  0 ≤ a ∧ a < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1524_152485


namespace NUMINAMATH_CALUDE_red_cube_latin_square_bijection_l1524_152465

/-- A Latin square of order n is an n × n array filled with n different symbols, 
    each occurring exactly once in each row and exactly once in each column. -/
def is_latin_square (s : Fin 4 → Fin 4 → Fin 4) : Prop :=
  ∀ i j k : Fin 4, 
    (∀ j' : Fin 4, j ≠ j' → s i j ≠ s i j') ∧ 
    (∀ i' : Fin 4, i ≠ i' → s i j ≠ s i' j)

/-- The number of 4 × 4 Latin squares -/
def num_latin_squares : ℕ := sorry

/-- A configuration of red cubes in a 4 × 4 × 4 cube -/
def red_cube_config : Type := Fin 4 → Fin 4 → Fin 4

/-- A valid configuration of red cubes satisfies the constraint that
    in every 1 × 1 × 4 rectangular prism, exactly 1 unit cube is red -/
def is_valid_config (c : red_cube_config) : Prop :=
  ∀ i j : Fin 4, ∃! k : Fin 4, c i j = k

/-- The number of valid red cube configurations -/
def num_valid_configs : ℕ := sorry

theorem red_cube_latin_square_bijection :
  num_valid_configs = num_latin_squares :=
sorry

end NUMINAMATH_CALUDE_red_cube_latin_square_bijection_l1524_152465


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l1524_152477

theorem unequal_gender_probability (n : ℕ) (p : ℚ) : 
  n = 12 → p = 1/2 → 
  (1 - (Nat.choose n (n/2) : ℚ) * p^(n/2) * (1-p)^(n/2)) = 793/1024 := by
  sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l1524_152477


namespace NUMINAMATH_CALUDE_division_result_l1524_152401

theorem division_result : (0.075 : ℚ) / (0.005 : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1524_152401


namespace NUMINAMATH_CALUDE_cube_root_of_product_rewrite_cube_root_l1524_152414

theorem cube_root_of_product (a b c : ℕ) : 
  (a^9 * b^3 * c^3 : ℝ)^(1/3) = (a^3 * b * c : ℝ) :=
by sorry

theorem rewrite_cube_root : (2^9 * 5^3 * 7^3 : ℝ)^(1/3) = 280 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_product_rewrite_cube_root_l1524_152414


namespace NUMINAMATH_CALUDE_wheel_rotation_on_moving_car_l1524_152443

/-- A wheel is a circular object that can rotate. -/
structure Wheel :=
  (radius : ℝ)
  (center : ℝ × ℝ)

/-- A car is a vehicle with wheels. -/
structure Car :=
  (wheels : List Wheel)

/-- Motion types that an object can exhibit. -/
inductive MotionType
  | Rotation
  | Translation
  | Other

/-- A moving car is a car with a velocity. -/
structure MovingCar extends Car :=
  (velocity : ℝ × ℝ)

/-- The motion type exhibited by a wheel on a moving car. -/
def wheelMotionType (mc : MovingCar) (w : Wheel) : MotionType :=
  sorry

/-- Theorem: The wheels of a moving car exhibit rotational motion. -/
theorem wheel_rotation_on_moving_car (mc : MovingCar) (w : Wheel) 
  (h : w ∈ mc.wheels) : 
  wheelMotionType mc w = MotionType.Rotation :=
sorry

end NUMINAMATH_CALUDE_wheel_rotation_on_moving_car_l1524_152443


namespace NUMINAMATH_CALUDE_binomial_30_3_l1524_152410

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l1524_152410


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_area_l1524_152451

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line in the form y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Calculate the area of triangle AOB formed by the intersection of an ellipse and a line -/
def area_triangle_AOB (e : Ellipse) (l : Line) : ℝ :=
  sorry

theorem ellipse_line_intersection_area :
  ∀ (e : Ellipse) (l : Line),
    e.b = 1 →
    e.a^2 = 2 →
    l.m = 1 ∧ l.c = Real.sqrt 2 →
    area_triangle_AOB e l = 2/3 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_area_l1524_152451


namespace NUMINAMATH_CALUDE_point_coordinates_l1524_152439

/-- Given point M(5, -6) and vector a = (1, -2), if NM = 3a, then N has coordinates (2, 0) -/
theorem point_coordinates (M N : ℝ × ℝ) (a : ℝ × ℝ) : 
  M = (5, -6) → 
  a = (1, -2) → 
  N.1 - M.1 = 3 * a.1 ∧ N.2 - M.2 = 3 * a.2 → 
  N = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1524_152439


namespace NUMINAMATH_CALUDE_ellipse_properties_l1524_152467

/-- Given an ellipse with the following properties:
  * Equation: x²/a² + y²/b² = 1, where a > b > 0
  * Vertices: A(0,b) and C(0,-b)
  * Foci: F₁(-c,0) and F₂(c,0), where c > 0
  * A line through point E(3c,0) intersects the ellipse at another point B
  * F₁A ∥ F₂B
-/
theorem ellipse_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  let e := c / a -- eccentricity
  let m := (5/3) * c
  let n := (2*Real.sqrt 2/3) * c
  ∃ (x y : ℝ),
    -- Point B on the ellipse
    x^2/a^2 + y^2/b^2 = 1 ∧
    -- B is on the line through E(3c,0)
    ∃ (t : ℝ), x = 3*c*(1-t) ∧ y = 3*c*t ∧
    -- F₁A ∥ F₂B
    (b / (-c)) = (y - 0) / (x - c) ∧
    -- Eccentricity is √3/3
    e = Real.sqrt 3 / 3 ∧
    -- Point H(m,n) is on F₂B
    n / (m - c) = y / (x - c) ∧
    -- H is on the circumcircle of AF₁C
    (m - c/2)^2 + n^2 = (3*c/2)^2 ∧
    -- Ratio n/m
    n / m = 2 * Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1524_152467


namespace NUMINAMATH_CALUDE_secretary_project_hours_l1524_152407

theorem secretary_project_hours (total_hours : ℕ) (ratio_1 ratio_2 ratio_3 ratio_4 : ℕ) :
  total_hours = 2080 →
  ratio_1 = 3 →
  ratio_2 = 5 →
  ratio_3 = 7 →
  ratio_4 = 11 →
  (ratio_1 + ratio_2 + ratio_3 + ratio_4) * (total_hours / (ratio_1 + ratio_2 + ratio_3 + ratio_4)) = total_hours →
  ratio_4 * (total_hours / (ratio_1 + ratio_2 + ratio_3 + ratio_4)) = 880 :=
by sorry

end NUMINAMATH_CALUDE_secretary_project_hours_l1524_152407


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1524_152449

theorem arithmetic_calculations :
  (23 + (-13) + (-17) + 8 = 1) ∧
  (-2^3 - (1 + 0.5) / (1/3) * (-3) = 11/2) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1524_152449
