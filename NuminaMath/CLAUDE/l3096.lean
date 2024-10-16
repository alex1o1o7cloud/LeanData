import Mathlib

namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_four_times_exterior_l3096_309650

theorem polygon_sides_when_interior_four_times_exterior : 
  ∀ n : ℕ, n > 2 →
  (n - 2) * 180 = 4 * 360 →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_four_times_exterior_l3096_309650


namespace NUMINAMATH_CALUDE_rachel_colored_pictures_l3096_309672

/-- The number of pictures Rachel has colored -/
def pictures_colored (book1_pictures book2_pictures remaining_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - remaining_pictures

theorem rachel_colored_pictures :
  pictures_colored 23 32 11 = 44 := by
  sorry

end NUMINAMATH_CALUDE_rachel_colored_pictures_l3096_309672


namespace NUMINAMATH_CALUDE_floor_sum_equals_n_l3096_309615

theorem floor_sum_equals_n (N : ℕ+) :
  N = ∑' n : ℕ, ⌊(N : ℝ) / (2 ^ n : ℝ)⌋ := by sorry

end NUMINAMATH_CALUDE_floor_sum_equals_n_l3096_309615


namespace NUMINAMATH_CALUDE_cone_height_for_given_volume_and_angle_l3096_309646

/-- Represents a cone with given volume and vertex angle -/
structure Cone where
  volume : ℝ
  vertexAngle : ℝ

/-- Calculates the height of a cone given its volume and vertex angle -/
def coneHeight (c : Cone) : ℝ :=
  sorry

/-- Theorem stating that a cone with volume 19683π and vertex angle 90° has height 39 -/
theorem cone_height_for_given_volume_and_angle :
  let c : Cone := { volume := 19683 * Real.pi, vertexAngle := 90 }
  coneHeight c = 39 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_for_given_volume_and_angle_l3096_309646


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l3096_309697

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_b_value :
  ∀ b : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, b⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → b = -9 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l3096_309697


namespace NUMINAMATH_CALUDE_binomial_60_3_l3096_309669

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l3096_309669


namespace NUMINAMATH_CALUDE_inequality_of_four_positives_l3096_309689

theorem inequality_of_four_positives (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((abc + abd + acd + bcd) / 4) ^ (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_of_four_positives_l3096_309689


namespace NUMINAMATH_CALUDE_car_travel_theorem_l3096_309643

/-- Represents the distance-time relationship for a car traveling between two points --/
def distance_function (initial_distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  initial_distance - speed * time

theorem car_travel_theorem (initial_distance speed time : ℝ) 
  (h1 : initial_distance = 120)
  (h2 : speed = 80)
  (h3 : 0 ≤ time)
  (h4 : time ≤ 1.5) :
  let y := distance_function initial_distance speed time
  ∀ x, x = time → y = 120 - 80 * x ∧ 
  (x = 0.8 → y = 56) := by
  sorry

#check car_travel_theorem

end NUMINAMATH_CALUDE_car_travel_theorem_l3096_309643


namespace NUMINAMATH_CALUDE_tic_tac_toe_wins_l3096_309645

theorem tic_tac_toe_wins (total_rounds harry_wins william_wins : ℕ) :
  total_rounds = 15 →
  william_wins = harry_wins + 5 →
  william_wins = 10 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_wins_l3096_309645


namespace NUMINAMATH_CALUDE_pi_estimate_l3096_309673

theorem pi_estimate (n : ℕ) (m : ℕ) (h1 : n = 200) (h2 : m = 56) :
  let p := m / n
  let estimate := (4 * p + 2) / 1
  estimate = 78 / 25 := by
sorry

end NUMINAMATH_CALUDE_pi_estimate_l3096_309673


namespace NUMINAMATH_CALUDE_fraction_product_cubed_simplify_fraction_cube_l3096_309609

theorem fraction_product_cubed (a b c d : ℚ) :
  (a / b) ^ 3 * (c / d) ^ 3 = ((a * c) / (b * d)) ^ 3 :=
by sorry

theorem simplify_fraction_cube :
  (5 / 8) ^ 3 * (2 / 3) ^ 3 = 125 / 1728 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_cubed_simplify_fraction_cube_l3096_309609


namespace NUMINAMATH_CALUDE_apple_weight_l3096_309698

/-- Given a basket of apples, prove the weight of apples alone -/
theorem apple_weight (total : ℝ) (half : ℝ) (h1 : total = 52) (h2 : half = 28) :
  ∃ (basket : ℝ), basket ≥ 0 ∧ total = 2 * (half - basket) + basket ∧ total - basket = 48 := by
  sorry

end NUMINAMATH_CALUDE_apple_weight_l3096_309698


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l3096_309607

theorem modulo_eleven_residue :
  (332 + 6 * 44 + 8 * 176 + 3 * 22) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l3096_309607


namespace NUMINAMATH_CALUDE_evaluate_expression_l3096_309649

theorem evaluate_expression : 3^0 + 9^5 / 9^3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3096_309649


namespace NUMINAMATH_CALUDE_profit_increase_march_to_june_l3096_309641

/-- Calculates the total percent increase in profits from March to June given monthly changes -/
theorem profit_increase_march_to_june 
  (march_profit : ℝ) 
  (april_increase : ℝ) 
  (may_decrease : ℝ) 
  (june_increase : ℝ) 
  (h1 : april_increase = 0.4) 
  (h2 : may_decrease = 0.2) 
  (h3 : june_increase = 0.5) : 
  (((1 + june_increase) * (1 - may_decrease) * (1 + april_increase) - 1) * 100 = 68) := by
sorry

end NUMINAMATH_CALUDE_profit_increase_march_to_june_l3096_309641


namespace NUMINAMATH_CALUDE_min_max_x_given_xy_eq_nx_plus_ny_l3096_309670

theorem min_max_x_given_xy_eq_nx_plus_ny (n x y : ℕ+) (h : x * y = n * x + n * y) :
  x ≥ n + 1 ∧ x ≤ n * (n + 1) :=
sorry

end NUMINAMATH_CALUDE_min_max_x_given_xy_eq_nx_plus_ny_l3096_309670


namespace NUMINAMATH_CALUDE_statements_about_positive_numbers_l3096_309605

theorem statements_about_positive_numbers (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a > b → a^2 > b^2) ∧ 
  ((2 * a * b) / (a + b) < (a + b) / 2) ∧ 
  ((a^2 + b^2) / 2 > ((a + b) / 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_statements_about_positive_numbers_l3096_309605


namespace NUMINAMATH_CALUDE_redox_agents_identification_l3096_309695

/-- Represents a chemical species with its oxidation state -/
structure Species where
  element : String
  oxidation_state : Int

/-- Represents a half-reaction in a redox reaction -/
structure HalfReaction where
  reactant : Species
  product : Species
  electrons : Int
  is_reduction : Bool

/-- Represents a full redox reaction -/
structure RedoxReaction where
  oxidation : HalfReaction
  reduction : HalfReaction

def is_oxidizing_agent (s : Species) (r : RedoxReaction) : Prop :=
  s = r.reduction.reactant

def is_reducing_agent (s : Species) (r : RedoxReaction) : Prop :=
  s = r.oxidation.reactant

theorem redox_agents_identification
  (s0 : Species)
  (h20 : Species)
  (h2plus : Species)
  (s2minus : Species)
  (reduction : HalfReaction)
  (oxidation : HalfReaction)
  (full_reaction : RedoxReaction)
  (h_s0 : s0 = ⟨"S", 0⟩)
  (h_h20 : h20 = ⟨"H2", 0⟩)
  (h_h2plus : h2plus = ⟨"H2", 1⟩)
  (h_s2minus : s2minus = ⟨"S", -2⟩)
  (h_reduction : reduction = ⟨s0, s2minus, 2, true⟩)
  (h_oxidation : oxidation = ⟨h20, h2plus, -2, false⟩)
  (h_full_reaction : full_reaction = ⟨oxidation, reduction⟩)
  : is_oxidizing_agent s0 full_reaction ∧ is_reducing_agent h20 full_reaction := by
  sorry


end NUMINAMATH_CALUDE_redox_agents_identification_l3096_309695


namespace NUMINAMATH_CALUDE_num_paths_correct_l3096_309678

/-- The number of paths from (0,0) to (m,n) on Z^2 using only steps of +(1,0) or +(0,1) -/
def num_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem stating that num_paths gives the correct number of paths -/
theorem num_paths_correct (m n : ℕ) : 
  num_paths m n = Nat.choose (m + n) m := by
  sorry

end NUMINAMATH_CALUDE_num_paths_correct_l3096_309678


namespace NUMINAMATH_CALUDE_candy_pebbles_l3096_309671

theorem candy_pebbles (candy : ℕ) (lance : ℕ) : 
  lance = 3 * candy ∧ lance = candy + 8 → candy = 4 :=
by sorry

end NUMINAMATH_CALUDE_candy_pebbles_l3096_309671


namespace NUMINAMATH_CALUDE_tangent_length_circle_l3096_309630

/-- The length of the tangent line from a point on a circle to the circle itself -/
theorem tangent_length_circle (x y : ℝ) : 
  x^2 + (y - 2)^2 = 4 → 
  x = 2 → 
  y = 2 → 
  Real.sqrt ((x - 0)^2 + (y - 2)^2 - 4) = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_length_circle_l3096_309630


namespace NUMINAMATH_CALUDE_solution_to_system_l3096_309663

theorem solution_to_system :
  ∃ (x y : ℚ), 3 * x - 24 * y = 3 ∧ x - 3 * y = 4 ∧ x = 29/5 ∧ y = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l3096_309663


namespace NUMINAMATH_CALUDE_alpha_plus_beta_value_l3096_309629

theorem alpha_plus_beta_value (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 54*x + 621) / (x^2 + 42*x - 1764)) →
  α + β = 86 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_value_l3096_309629


namespace NUMINAMATH_CALUDE_ellipse_properties_l3096_309642

/-- Ellipse C in the Cartesian coordinate system α -/
def C (b : ℝ) (x y : ℝ) : Prop :=
  0 < b ∧ b < 2 ∧ x^2 / 4 + y^2 / b^2 = 1

/-- Point A is the right vertex of C -/
def A : ℝ × ℝ := (2, 0)

/-- Line l passing through O with non-zero slope -/
def l (m : ℝ) (x y : ℝ) : Prop :=
  m ≠ 0 ∧ y = m * x

/-- P and Q are intersection points of l and C -/
def intersectionPoints (b m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C b x₁ y₁ ∧ C b x₂ y₂ ∧ l m x₁ y₁ ∧ l m x₂ y₂

/-- M and N are intersections of AP, AQ with y-axis -/
def MN (b m : ℝ) : Prop :=
  ∃ (y₁ y₂ : ℝ), (∃ (x₁ y₁' : ℝ), C b x₁ y₁' ∧ l m x₁ y₁' ∧ y₁ = (y₁' / (x₁ - 2)) * (-2)) ∧
                 (∃ (x₂ y₂' : ℝ), C b x₂ y₂' ∧ l m x₂ y₂' ∧ y₂ = (y₂' / (x₂ - 2)) * (-2))

theorem ellipse_properties (b m : ℝ) :
  C b 1 1 → intersectionPoints b m → (∃ (x y : ℝ), C b x y ∧ l m x y ∧ (x^2 + y^2)^(1/2) = 2 * ((x - 2)^2 + y^2)^(1/2)) →
  b^2 = 4/3 ∧ (∀ y₁ y₂ : ℝ, MN b m → y₁ * y₂ = b^2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3096_309642


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3096_309618

theorem unique_solution_exponential_equation :
  ∃! (x y z t : ℕ+), 12^(x:ℕ) + 13^(y:ℕ) - 14^(z:ℕ) = 2013^(t:ℕ) ∧ 
    x = 1 ∧ y = 3 ∧ z = 2 ∧ t = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3096_309618


namespace NUMINAMATH_CALUDE_stratified_sampling_l3096_309625

theorem stratified_sampling (n : ℕ) (a b c : ℕ) :
  a = 16 ∧ 2 * a = 3 * b ∧ 3 * b = 5 * c ∧ n = a + b + c → n = 80 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_l3096_309625


namespace NUMINAMATH_CALUDE_slope_of_solutions_l3096_309658

/-- The equation that defines the relationship between x and y -/
def equation (x y : ℝ) : Prop := 2 / x + 3 / y = 0

/-- Theorem: The slope of the line determined by any two distinct solutions to the equation is -3/2 -/
theorem slope_of_solutions (x₁ y₁ x₂ y₂ : ℝ) (h₁ : equation x₁ y₁) (h₂ : equation x₂ y₂) (h_dist : (x₁, y₁) ≠ (x₂, y₂)) :
  (y₂ - y₁) / (x₂ - x₁) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_solutions_l3096_309658


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_intersection_and_double_angle_line_l3096_309651

-- Define the two original lines
def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l₂ (x y : ℝ) : Prop := x + 2*y - 9 = 0

-- Define the intersection point A
def A : ℝ × ℝ := (3, 3)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2*x + 3*y - 1 = 0

theorem intersection_and_parallel_line :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y → (2*x + 3*y - 15 = 0) := by sorry

theorem intersection_and_double_angle_line :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y → (4*x - 3*y - 3 = 0) := by sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_line_intersection_and_double_angle_line_l3096_309651


namespace NUMINAMATH_CALUDE_zero_in_set_A_l3096_309617

theorem zero_in_set_A : 
  let A : Set ℕ := {0, 1, 2}
  0 ∈ A := by
sorry

end NUMINAMATH_CALUDE_zero_in_set_A_l3096_309617


namespace NUMINAMATH_CALUDE_fourth_student_number_l3096_309665

def systematicSampling (totalStudents : Nat) (samplesToSelect : Nat) (selected : List Nat) : Nat :=
  sorry

theorem fourth_student_number
  (totalStudents : Nat)
  (samplesToSelect : Nat)
  (selected : List Nat)
  (h1 : totalStudents = 54)
  (h2 : samplesToSelect = 4)
  (h3 : selected = [3, 29, 42])
  : systematicSampling totalStudents samplesToSelect selected = 16 :=
by sorry

end NUMINAMATH_CALUDE_fourth_student_number_l3096_309665


namespace NUMINAMATH_CALUDE_mark_second_play_time_l3096_309652

/-- Calculates the time Mark played in the second part of a soccer game. -/
def second_play_time (total_time initial_play sideline : ℕ) : ℕ :=
  total_time - initial_play - sideline

/-- Theorem: Mark played 35 minutes in the second part of the game. -/
theorem mark_second_play_time :
  let total_time : ℕ := 90
  let initial_play : ℕ := 20
  let sideline : ℕ := 35
  second_play_time total_time initial_play sideline = 35 := by
  sorry

end NUMINAMATH_CALUDE_mark_second_play_time_l3096_309652


namespace NUMINAMATH_CALUDE_journey_time_ratio_and_sum_l3096_309676

/-- Represents the ratio of road segments --/
def road_ratio : Fin 3 → ℚ
| 0 => 1
| 1 => 2
| 2 => 1

/-- Represents the ratio of speeds on different road types --/
def speed_ratio : Fin 3 → ℚ
| 0 => 3
| 1 => 2
| 2 => 4

/-- Calculates the time taken for a journey --/
def journey_time (r : Fin 3 → ℚ) (s : Fin 3 → ℚ) : ℚ :=
  (r 0 / s 0) + (r 1 / s 1) + (r 2 / s 2)

/-- Theorem stating the ratio of journey times and the sum of m and n --/
theorem journey_time_ratio_and_sum :
  let to_school := journey_time road_ratio speed_ratio
  let return_home := journey_time road_ratio (fun i => speed_ratio (2 - i))
  let ratio := to_school / return_home
  ∃ (m n : ℕ), m.Coprime n ∧ ratio = n / m ∧ m + n = 35 := by
  sorry


end NUMINAMATH_CALUDE_journey_time_ratio_and_sum_l3096_309676


namespace NUMINAMATH_CALUDE_amusement_park_admission_l3096_309610

theorem amusement_park_admission (child_fee : ℚ) (adult_fee : ℚ) (total_fee : ℚ) (num_children : ℕ) :
  child_fee = 3/2 →
  adult_fee = 4 →
  total_fee = 810 →
  num_children = 180 →
  ∃ (num_adults : ℕ), 
    (child_fee * num_children + adult_fee * num_adults = total_fee) ∧
    (num_children + num_adults = 315) :=
by
  sorry

end NUMINAMATH_CALUDE_amusement_park_admission_l3096_309610


namespace NUMINAMATH_CALUDE_tripod_height_after_break_l3096_309656

theorem tripod_height_after_break (original_leg_length original_height broken_leg_length : ℝ) 
  (h : ℝ) (m n : ℕ) :
  original_leg_length = 6 →
  original_height = 5 →
  broken_leg_length = 4 →
  h = 12 →
  h = m / Real.sqrt n →
  m = 168 →
  n = 169 →
  ⌊m + Real.sqrt n⌋ = 181 :=
by sorry

end NUMINAMATH_CALUDE_tripod_height_after_break_l3096_309656


namespace NUMINAMATH_CALUDE_bakery_boxes_sold_l3096_309644

/-- Calculates the number of boxes of doughnuts sold by a bakery -/
theorem bakery_boxes_sold
  (doughnuts_per_box : ℕ)
  (total_doughnuts : ℕ)
  (doughnuts_given_away : ℕ)
  (h1 : doughnuts_per_box = 10)
  (h2 : total_doughnuts = 300)
  (h3 : doughnuts_given_away = 30) :
  (total_doughnuts / doughnuts_per_box) - (doughnuts_given_away / doughnuts_per_box) = 27 :=
by sorry

end NUMINAMATH_CALUDE_bakery_boxes_sold_l3096_309644


namespace NUMINAMATH_CALUDE_triangle_side_value_l3096_309693

/-- In a triangle ABC, given specific conditions, prove that a = 2√3 -/
theorem triangle_side_value (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  A = 2 * C ∧  -- Given condition
  c = 2 ∧  -- Given condition
  a^2 = 4*b - 4 ∧  -- Given condition
  a / (Real.sin A) = b / (Real.sin B) ∧  -- Sine law
  a / (Real.sin A) = c / (Real.sin C) ∧  -- Sine law
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) ∧  -- Cosine law
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧  -- Cosine law
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)  -- Cosine law
  → a = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_value_l3096_309693


namespace NUMINAMATH_CALUDE_equation_solution_l3096_309611

theorem equation_solution (x : ℝ) : 
  12 * Real.sin x - 5 * Real.cos x = 13 ↔ 
  ∃ k : ℤ, x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3096_309611


namespace NUMINAMATH_CALUDE_tens_digit_of_23_to_1987_l3096_309600

theorem tens_digit_of_23_to_1987 : ∃ k : ℕ, 23^1987 ≡ 40 + k [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_to_1987_l3096_309600


namespace NUMINAMATH_CALUDE_calculation_proof_l3096_309602

theorem calculation_proof : 56.8 * 35.7 + 56.8 * 28.5 + 64.2 * 43.2 = 6420 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3096_309602


namespace NUMINAMATH_CALUDE_triangle_properties_l3096_309680

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  cos B = 4/5 →
  b = 2 →
  (a = 5/3 → A = π/6) ∧
  (∀ a c, a > 0 → c > 0 → (1/2) * a * c * (3/5) ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3096_309680


namespace NUMINAMATH_CALUDE_least_non_special_fraction_l3096_309628

/-- Represents a fraction in the form (2^a - 2^b) / (2^c - 2^d) where a, b, c, d are positive integers -/
def SpecialFraction (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ+), n = (2^a.val - 2^b.val) / (2^c.val - 2^d.val)

/-- The least positive integer that cannot be represented as a SpecialFraction is 11 -/
theorem least_non_special_fraction : (∀ k < 11, SpecialFraction k) ∧ ¬SpecialFraction 11 := by
  sorry

end NUMINAMATH_CALUDE_least_non_special_fraction_l3096_309628


namespace NUMINAMATH_CALUDE_circle_radius_bounds_l3096_309688

/-- Given a quadrilateral ABCD circumscribed around a circle, where the
    tangency points divide AB into segments a and b, and AD into segments a and c,
    prove that the radius r of the circle satisfies the given inequality. -/
theorem circle_radius_bounds (a b c r : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) : 
  Real.sqrt ((a * b * c) / (a + b + c)) < r ∧ 
  r < Real.sqrt (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_bounds_l3096_309688


namespace NUMINAMATH_CALUDE_volume_polynomial_coefficients_ratio_l3096_309627

/-- A right rectangular prism with edge lengths 2, 2, and 5 -/
structure Prism where
  length : ℝ := 2
  width : ℝ := 2
  height : ℝ := 5

/-- The set of points within distance r of any point in the prism -/
def S (B : Prism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume (B : Prism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume polynomial -/
structure VolumeCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem volume_polynomial_coefficients_ratio (B : Prism) (coeff : VolumeCoefficients) :
  (∀ r : ℝ, volume B r = coeff.a * r^3 + coeff.b * r^2 + coeff.c * r + coeff.d) →
  (coeff.a > 0 ∧ coeff.b > 0 ∧ coeff.c > 0 ∧ coeff.d > 0) →
  (coeff.b * coeff.c) / (coeff.a * coeff.d) = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_volume_polynomial_coefficients_ratio_l3096_309627


namespace NUMINAMATH_CALUDE_ad_purchase_cost_is_108000_l3096_309639

/-- Represents the dimensions of an ad space -/
structure AdSpace where
  length : ℝ
  width : ℝ

/-- Represents the cost and quantity information for ad purchases -/
structure AdPurchase where
  numCompanies : ℕ
  adSpacesPerCompany : ℕ
  adSpace : AdSpace
  costPerSquareFoot : ℝ

/-- Calculates the total cost of ad purchases for multiple companies -/
def totalAdCost (purchase : AdPurchase) : ℝ :=
  purchase.numCompanies * purchase.adSpacesPerCompany * 
  purchase.adSpace.length * purchase.adSpace.width * 
  purchase.costPerSquareFoot

/-- Theorem stating that the total cost for the given ad purchase scenario is $108,000 -/
theorem ad_purchase_cost_is_108000 : 
  totalAdCost {
    numCompanies := 3,
    adSpacesPerCompany := 10,
    adSpace := { length := 12, width := 5 },
    costPerSquareFoot := 60
  } = 108000 := by
  sorry

end NUMINAMATH_CALUDE_ad_purchase_cost_is_108000_l3096_309639


namespace NUMINAMATH_CALUDE_corrected_mean_l3096_309601

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 30 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * (30.5 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l3096_309601


namespace NUMINAMATH_CALUDE_radii_ratio_in_regular_hexagonal_pyramid_l3096_309681

/-- A regular hexagonal pyramid with circumscribed and inscribed spheres -/
structure RegularHexagonalPyramid where
  /-- Radius of the circumscribed sphere -/
  R : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- The center of the circumscribed sphere lies on the surface of the inscribed sphere -/
  center_on_surface : R = r * (1 + Real.sqrt 21 / 3)

/-- The ratio of the radii of the circumscribed sphere to the inscribed sphere
    in a regular hexagonal pyramid where the center of the circumscribed sphere
    lies on the surface of the inscribed sphere is (3 + √21) / 3 -/
theorem radii_ratio_in_regular_hexagonal_pyramid (p : RegularHexagonalPyramid) :
  p.R / p.r = (3 + Real.sqrt 21) / 3 := by
  sorry

end NUMINAMATH_CALUDE_radii_ratio_in_regular_hexagonal_pyramid_l3096_309681


namespace NUMINAMATH_CALUDE_eleven_one_base_three_is_perfect_square_l3096_309637

/-- Represents a number in a given base --/
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- Checks if a number is a perfect square --/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- The main theorem --/
theorem eleven_one_base_three_is_perfect_square :
  isPerfectSquare (toDecimal [1, 1, 1, 1, 1] 3) := by
  sorry

end NUMINAMATH_CALUDE_eleven_one_base_three_is_perfect_square_l3096_309637


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3096_309647

theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4*x + 3 = 0 ∧ a * y^2 - 4*y + 3 = 0) ↔ 
  (a < 4/3 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3096_309647


namespace NUMINAMATH_CALUDE_female_employees_count_l3096_309632

/-- Given a company with the following properties:
  1. There are 280 female managers.
  2. 2/5 of all employees are managers.
  3. 2/5 of all male employees are managers.
  Prove that the total number of female employees is 700. -/
theorem female_employees_count (total_employees : ℕ) (male_employees : ℕ) :
  let female_managers : ℕ := 280
  let total_managers : ℕ := (2 * total_employees) / 5
  let male_managers : ℕ := (2 * male_employees) / 5
  total_managers = female_managers + male_managers →
  total_employees - male_employees = 700 := by
  sorry

end NUMINAMATH_CALUDE_female_employees_count_l3096_309632


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_5_range_of_m_for_f_geq_g_set_iic_1_equiv_interval_l3096_309659

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| + 2
def g (m : ℝ) (x : ℝ) : ℝ := m * |x|

-- Theorem for part I
theorem solution_set_f_greater_than_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -1 ∨ x > 5} := by sorry

-- Theorem for part II
theorem range_of_m_for_f_geq_g :
  {m : ℝ | ∀ x, f x ≥ g m x} = Set.Iic 1 := by sorry

-- Additional helper theorem to show that Set.Iic 1 is equivalent to (-∞, 1]
theorem set_iic_1_equiv_interval :
  Set.Iic 1 = {m : ℝ | m ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_5_range_of_m_for_f_geq_g_set_iic_1_equiv_interval_l3096_309659


namespace NUMINAMATH_CALUDE_symmetric_angles_theorem_l3096_309633

-- Define the property of terminal sides being symmetric with respect to x + y = 0
def symmetric_terminal_sides (α β : Real) : Prop := sorry

-- Define the set of angles β
def angle_set : Set Real := {β | ∃ k : Int, β = 2 * k * Real.pi - Real.pi / 6}

-- State the theorem
theorem symmetric_angles_theorem (α β : Real) 
  (h_symmetric : symmetric_terminal_sides α β) 
  (h_alpha : α = -Real.pi / 3) : 
  β ∈ angle_set := by sorry

end NUMINAMATH_CALUDE_symmetric_angles_theorem_l3096_309633


namespace NUMINAMATH_CALUDE_wednesday_spending_multiple_l3096_309624

def monday_spending : ℝ := 60
def tuesday_spending : ℝ := 4 * monday_spending
def total_spending : ℝ := 600

theorem wednesday_spending_multiple : 
  ∃ x : ℝ, 
    monday_spending + tuesday_spending + x * monday_spending = total_spending ∧ 
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_spending_multiple_l3096_309624


namespace NUMINAMATH_CALUDE_number_of_shooting_orders_l3096_309654

/-- Represents the number of targets in each column -/
def targets_per_column : Fin 3 → ℕ
  | 0 => 4  -- Column A
  | 1 => 3  -- Column B
  | 2 => 3  -- Column C

/-- The total number of targets -/
def total_targets : ℕ := 10

/-- The number of initial shooting sequences -/
def initial_sequences : ℕ := 2

/-- Calculates the number of permutations for the remaining shots -/
def remaining_permutations : ℕ :=
  Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 3)

/-- Theorem stating the total number of different orders to break the targets -/
theorem number_of_shooting_orders :
  initial_sequences * remaining_permutations = 1120 := by sorry

end NUMINAMATH_CALUDE_number_of_shooting_orders_l3096_309654


namespace NUMINAMATH_CALUDE_smallest_perfect_squares_l3096_309636

theorem smallest_perfect_squares (a b : ℕ+) 
  (h1 : ∃ x : ℕ, (15 * a + 16 * b : ℕ) = x^2)
  (h2 : ∃ y : ℕ, (16 * a - 15 * b : ℕ) = y^2) :
  ∃ (x y : ℕ), x^2 = 231361 ∧ y^2 = 231361 ∧ 
    (∀ (x' y' : ℕ), (15 * a + 16 * b : ℕ) = x'^2 → (16 * a - 15 * b : ℕ) = y'^2 → 
      x'^2 ≥ 231361 ∧ y'^2 ≥ 231361) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_squares_l3096_309636


namespace NUMINAMATH_CALUDE_chocolate_purchase_shortage_l3096_309683

theorem chocolate_purchase_shortage (chocolate_cost : ℕ) (initial_money : ℕ) (borrowed_money : ℕ) : 
  chocolate_cost = 500 ∧ initial_money = 400 ∧ borrowed_money = 59 →
  chocolate_cost - (initial_money + borrowed_money) = 41 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_purchase_shortage_l3096_309683


namespace NUMINAMATH_CALUDE_henry_skittles_count_l3096_309699

/-- The number of Skittles Bridget has initially -/
def bridget_initial : ℕ := 4

/-- The number of Skittles Bridget has after receiving Henry's Skittles -/
def bridget_final : ℕ := 8

/-- The number of Skittles Henry has -/
def henry_skittles : ℕ := bridget_final - bridget_initial

theorem henry_skittles_count : henry_skittles = 4 := by
  sorry

end NUMINAMATH_CALUDE_henry_skittles_count_l3096_309699


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3096_309640

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 + 4*I) / (1 + I) ∧ (z.re > 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3096_309640


namespace NUMINAMATH_CALUDE_jellybean_probability_l3096_309619

/-- Probability of picking exactly 2 red jellybeans from a bowl -/
theorem jellybean_probability :
  let total_jellybeans : ℕ := 10
  let red_jellybeans : ℕ := 4
  let blue_jellybeans : ℕ := 1
  let white_jellybeans : ℕ := 5
  let picks : ℕ := 3
  
  -- Ensure the total number of jellybeans is correct
  total_jellybeans = red_jellybeans + blue_jellybeans + white_jellybeans →
  
  -- Calculate the probability
  (Nat.choose red_jellybeans 2 * (blue_jellybeans + white_jellybeans)) / 
  Nat.choose total_jellybeans picks = 3 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l3096_309619


namespace NUMINAMATH_CALUDE_family_museum_cost_calculation_l3096_309664

/-- Calculates the discounted ticket price based on age --/
def discountedPrice (age : ℕ) (basePrice : ℚ) : ℚ :=
  if age ≥ 65 then basePrice * (1 - 0.2)
  else if age ≥ 12 ∧ age ≤ 18 then basePrice * (1 - 0.3)
  else if age ≥ 0 ∧ age ≤ 11 then basePrice * (1 - 0.5)
  else basePrice

/-- Calculates the total cost for a family museum trip --/
def familyMuseumCost (ages : List ℕ) (regularPrice specialPrice taxRate : ℚ) : ℚ :=
  let totalBeforeTax := (ages.map (fun age => discountedPrice age regularPrice + specialPrice)).sum
  totalBeforeTax * (1 + taxRate)

theorem family_museum_cost_calculation :
  let ages := [15, 10, 40, 42, 65]
  let regularPrice := 10
  let specialPrice := 5
  let taxRate := 0.1
  familyMuseumCost ages regularPrice specialPrice taxRate = 71.5 := by sorry

end NUMINAMATH_CALUDE_family_museum_cost_calculation_l3096_309664


namespace NUMINAMATH_CALUDE_veronica_photos_l3096_309687

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem veronica_photos (x : ℕ) 
  (h1 : choose x 3 + choose x 4 = 15) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_veronica_photos_l3096_309687


namespace NUMINAMATH_CALUDE_max_score_15_cards_l3096_309684

/-- Represents the score for a hand of cards -/
def score (r b y : ℕ) : ℕ := r + 2 * r * b + 3 * b * y

/-- The maximum score achievable with 15 cards -/
theorem max_score_15_cards : 
  ∃ (r b y : ℕ), r + b + y = 15 ∧ ∀ (r' b' y' : ℕ), r' + b' + y' = 15 → score r' b' y' ≤ score r b y ∧ score r b y = 168 := by
  sorry

end NUMINAMATH_CALUDE_max_score_15_cards_l3096_309684


namespace NUMINAMATH_CALUDE_crossed_out_number_is_21_l3096_309657

def first_n_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem crossed_out_number_is_21 :
  ∀ a : ℕ, 
    a > 0 ∧ a ≤ 20 →
    (∃ k : ℕ, k > 0 ∧ k ≤ 20 ∧ k ≠ a ∧ 
      k = (first_n_sum 20 - a) / 19 ∧ 
      (first_n_sum 20 - a) % 19 = 0) →
    a = 21 :=
by sorry

end NUMINAMATH_CALUDE_crossed_out_number_is_21_l3096_309657


namespace NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l3096_309648

/-- Given a total sum split into two parts, if the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years
    at 5% per annum, then the second part is 1664 Rs. -/
theorem interest_equality_implies_second_sum (total : ℝ) (first second : ℝ) :
  total = 2704 →
  first + second = total →
  (first * 3 * 8) / 100 = (second * 5 * 3) / 100 →
  second = 1664 := by
  sorry

end NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l3096_309648


namespace NUMINAMATH_CALUDE_suzy_book_count_l3096_309662

/-- Calculates the final number of books Suzy has after three days of transactions. -/
def final_book_count (initial : ℕ) (wed_out : ℕ) (thur_in : ℕ) (thur_out : ℕ) (fri_in : ℕ) : ℕ :=
  initial - wed_out + thur_in - thur_out + fri_in

/-- Theorem stating that given the specific transactions over three days, 
    Suzy ends up with 80 books. -/
theorem suzy_book_count : 
  final_book_count 98 43 23 5 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_suzy_book_count_l3096_309662


namespace NUMINAMATH_CALUDE_ball_probabilities_l3096_309682

/-- Represents a box of balls -/
structure Box where
  red : ℕ
  blue : ℕ

/-- The initial state of Box A -/
def box_a : Box := ⟨2, 4⟩

/-- The initial state of Box B -/
def box_b : Box := ⟨3, 3⟩

/-- The number of balls drawn from Box A -/
def balls_drawn : ℕ := 2

/-- Probability of drawing at least one blue ball from Box A -/
def prob_blue_from_a : ℚ := 14/15

/-- Probability of drawing a blue ball from Box B after transfer -/
def prob_blue_from_b : ℚ := 13/24

theorem ball_probabilities :
  (prob_blue_from_a = 14/15) ∧ (prob_blue_from_b = 13/24) := by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l3096_309682


namespace NUMINAMATH_CALUDE_trip_equation_correct_l3096_309638

/-- Represents a car trip with a stop -/
structure CarTrip where
  totalDistance : ℝ
  totalTime : ℝ
  stopDuration : ℝ
  speedBefore : ℝ
  speedAfter : ℝ

/-- The equation for the trip is correct -/
theorem trip_equation_correct (trip : CarTrip) 
    (h1 : trip.totalDistance = 300)
    (h2 : trip.totalTime = 4)
    (h3 : trip.stopDuration = 0.5)
    (h4 : trip.speedBefore = 60)
    (h5 : trip.speedAfter = 90) :
  ∃ t : ℝ, 
    t ≥ 0 ∧ 
    t ≤ trip.totalTime - trip.stopDuration ∧
    trip.speedBefore * t + trip.speedAfter * (trip.totalTime - trip.stopDuration - t) = trip.totalDistance :=
by sorry

end NUMINAMATH_CALUDE_trip_equation_correct_l3096_309638


namespace NUMINAMATH_CALUDE_f_of_1_plus_g_of_3_l3096_309635

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 5
def g (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_of_1_plus_g_of_3 : f (1 + g 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_f_of_1_plus_g_of_3_l3096_309635


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3096_309675

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_multiplier_for_perfect_square :
  ∃ (k : ℕ), k > 0 ∧ 
  (∃ (m : ℕ), k * y = m^2) ∧
  (∀ (j : ℕ), 0 < j ∧ j < k → ¬∃ (n : ℕ), j * y = n^2) ∧
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3096_309675


namespace NUMINAMATH_CALUDE_expand_product_l3096_309634

theorem expand_product (x : ℝ) : (2 * x + 3) * (x - 4) = 2 * x^2 - 5 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3096_309634


namespace NUMINAMATH_CALUDE_olivias_house_height_l3096_309614

/-- The height of Olivia's house in feet -/
def house_height : ℕ := 81

/-- The length of the shadow cast by Olivia's house in feet -/
def house_shadow : ℕ := 70

/-- The height of the flagpole in feet -/
def flagpole_height : ℕ := 35

/-- The length of the shadow cast by the flagpole in feet -/
def flagpole_shadow : ℕ := 30

/-- The height of the bush in feet -/
def bush_height : ℕ := 14

/-- The length of the shadow cast by the bush in feet -/
def bush_shadow : ℕ := 12

theorem olivias_house_height :
  (house_height : ℚ) / house_shadow = flagpole_height / flagpole_shadow ∧
  (house_height : ℚ) / house_shadow = bush_height / bush_shadow ∧
  house_height = 81 :=
sorry

end NUMINAMATH_CALUDE_olivias_house_height_l3096_309614


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_seventh_term_l3096_309623

theorem largest_binomial_coefficient_seventh_term :
  let n : ℕ := 8
  let k : ℕ := 6  -- 7th term corresponds to choosing 6 out of 8
  ∀ i : ℕ, i ≤ n → (n.choose k) ≥ (n.choose i) :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_seventh_term_l3096_309623


namespace NUMINAMATH_CALUDE_original_average_calculation_l3096_309622

theorem original_average_calculation (total_pupils : ℕ) 
  (removed_pupils : ℕ) (removed_total : ℕ) (new_average : ℕ) : 
  total_pupils = 21 →
  removed_pupils = 4 →
  removed_total = 71 →
  new_average = 44 →
  (total_pupils * (total_pupils - removed_pupils) * new_average + 
   total_pupils * removed_total) / (total_pupils * total_pupils) = 39 :=
by sorry

end NUMINAMATH_CALUDE_original_average_calculation_l3096_309622


namespace NUMINAMATH_CALUDE_preference_change_difference_l3096_309660

theorem preference_change_difference (initial_online initial_traditional final_online final_traditional : ℚ) 
  (h_initial_sum : initial_online + initial_traditional = 1)
  (h_final_sum : final_online + final_traditional = 1)
  (h_initial_online : initial_online = 2/5)
  (h_initial_traditional : initial_traditional = 3/5)
  (h_final_online : final_online = 4/5)
  (h_final_traditional : final_traditional = 1/5) :
  let min_change := |final_online - initial_online|
  let max_change := min initial_traditional (1 - initial_online)
  max_change - min_change = 2/5 := by
sorry

#eval (2 : ℚ) / 5 -- This should evaluate to 0.4, which is 40%

end NUMINAMATH_CALUDE_preference_change_difference_l3096_309660


namespace NUMINAMATH_CALUDE_exists_zero_implies_a_range_l3096_309696

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * abs x - 3 * a - 1

-- State the theorem
theorem exists_zero_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1) 1 ∧ f a x₀ = 0) →
  a ∈ Set.Icc (-1/2) (-1/3) :=
by sorry

end NUMINAMATH_CALUDE_exists_zero_implies_a_range_l3096_309696


namespace NUMINAMATH_CALUDE_smallest_square_ending_2016_l3096_309668

theorem smallest_square_ending_2016 : ∃ (n : ℕ), n = 996 ∧ 
  (∀ (m : ℕ), m < n → m^2 % 10000 ≠ 2016) ∧ n^2 % 10000 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_ending_2016_l3096_309668


namespace NUMINAMATH_CALUDE_mean_home_runs_l3096_309679

def player_count : ℕ := 6 + 4 + 3 + 1

def total_home_runs : ℕ := 6 * 6 + 7 * 4 + 8 * 3 + 10 * 1

theorem mean_home_runs : (total_home_runs : ℚ) / player_count = 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l3096_309679


namespace NUMINAMATH_CALUDE_f_at_six_l3096_309621

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 2 * x^4 + 5 * x^3 - x^2 + 3 * x + 4

-- Theorem stating that f(6) = 3658
theorem f_at_six : f 6 = 3658 := by sorry

end NUMINAMATH_CALUDE_f_at_six_l3096_309621


namespace NUMINAMATH_CALUDE_function_values_impossibility_l3096_309653

theorem function_values_impossibility (a b c : ℝ) (d : ℤ) :
  ¬∃ (m : ℝ), (a * m^3 + b * m - c / m + d = -1) ∧
              (a * (-m)^3 + b * (-m) - c / (-m) + d = 4) := by
  sorry

end NUMINAMATH_CALUDE_function_values_impossibility_l3096_309653


namespace NUMINAMATH_CALUDE_total_lunch_spending_l3096_309690

def lunch_problem (your_spending friend_spending total_spending : ℕ) : Prop :=
  friend_spending = 11 ∧
  friend_spending = your_spending + 3 ∧
  total_spending = your_spending + friend_spending

theorem total_lunch_spending : ∃ (your_spending friend_spending total_spending : ℕ),
  lunch_problem your_spending friend_spending total_spending ∧ total_spending = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_lunch_spending_l3096_309690


namespace NUMINAMATH_CALUDE_f_properties_l3096_309655

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - 1) / x - a * x + a

theorem f_properties (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y → a ≤ 0 → f a x < f a y) ∧ 
  (∃ x₀, 0 < x₀ ∧ x₀ < 1 ∧ f a x₀ = Real.exp 1 - 1 → a < 1) ∧
  ¬(a < 1 → ∃ x₀, 0 < x₀ ∧ x₀ < 1 ∧ f a x₀ = Real.exp 1 - 1 ∧ 
    ∀ x, 0 < x ∧ x < 1 ∧ x ≠ x₀ → f a x ≠ Real.exp 1 - 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3096_309655


namespace NUMINAMATH_CALUDE_louise_pencil_boxes_l3096_309677

def pencil_problem (box_capacity : ℕ) (red_pencils : ℕ) (yellow_pencils : ℕ) : Prop :=
  let blue_pencils := 2 * red_pencils
  let green_pencils := red_pencils + blue_pencils
  let total_boxes := 
    (red_pencils + blue_pencils + yellow_pencils + green_pencils) / box_capacity
  total_boxes = 8

theorem louise_pencil_boxes : 
  pencil_problem 20 20 40 :=
sorry

end NUMINAMATH_CALUDE_louise_pencil_boxes_l3096_309677


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_other_numbers_rational_sqrt_three_unique_irrational_l3096_309613

theorem sqrt_three_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ)) :=
by sorry

theorem other_numbers_rational :
  ∃ (a b c d e f : ℤ), 
    b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0 ∧
    (-32 : ℚ) / 7 = (a : ℚ) / (b : ℚ) ∧
    (0 : ℚ) = (c : ℚ) / (d : ℚ) ∧
    (3.5 : ℚ) = (e : ℚ) / (f : ℚ) :=
by sorry

theorem sqrt_three_unique_irrational 
  (h1 : ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ)))
  (h2 : ∃ (a b c d e f : ℤ), 
    b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0 ∧
    (-32 : ℚ) / 7 = (a : ℚ) / (b : ℚ) ∧
    (0 : ℚ) = (c : ℚ) / (d : ℚ) ∧
    (3.5 : ℚ) = (e : ℚ) / (f : ℚ)) :
  Real.sqrt 3 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_other_numbers_rational_sqrt_three_unique_irrational_l3096_309613


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_root_l3096_309612

theorem quadratic_equation_integer_root (k : ℕ) : 
  (∃ x : ℕ, x^2 - 34*x + 34*k - 1 = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_root_l3096_309612


namespace NUMINAMATH_CALUDE_ellipse_properties_l3096_309692

/-- Properties of a specific ellipse -/
structure EllipseC where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_c : 2 = Real.sqrt (a^2 - b^2)
  h_slope : (b - 0) / (0 - a) = -Real.sqrt 3 / 3

/-- Theorem about the standard equation and a geometric property of the ellipse -/
theorem ellipse_properties (e : EllipseC) :
  (∃ (x y : ℝ), x^2 / 6 + y^2 / 2 = 1) ∧
  (∃ (F P M N : ℝ × ℝ),
    F.1 = 2 ∧ F.2 = 0 ∧
    P.1 = 3 ∧
    (M.1^2 / 6 + M.2^2 / 2 = 1) ∧
    (N.1^2 / 6 + N.2^2 / 2 = 1) ∧
    (M.2 - N.2) * (P.1 - F.1) = (P.2 - F.2) * (M.1 - N.1) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) / Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3096_309692


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3096_309608

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope -2 and x-intercept (5,0), the y-intercept is (0,10). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -2, x_intercept := (5, 0) }
  y_intercept l = (0, 10) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3096_309608


namespace NUMINAMATH_CALUDE_arthur_total_distance_l3096_309691

/-- Calculates the total distance walked by Arthur in miles -/
def arthur_walk (block_length : ℚ) (east west north south : ℕ) : ℚ :=
  ((east + west + north + south) : ℚ) * block_length

/-- Theorem: Arthur's total walk distance is 4.5 miles -/
theorem arthur_total_distance :
  arthur_walk (1/4) 8 0 15 5 = 4.5 := by sorry

end NUMINAMATH_CALUDE_arthur_total_distance_l3096_309691


namespace NUMINAMATH_CALUDE_vector_b_coordinates_l3096_309603

/-- Given a vector a = (-1, 2) and a vector b with magnitude 3√5,
    if the cosine of the angle between a and b is -1,
    then b = (3, -6) -/
theorem vector_b_coordinates (a b : ℝ × ℝ) (θ : ℝ) : 
  a = (-1, 2) →
  ‖b‖ = 3 * Real.sqrt 5 →
  θ = Real.arccos (-1) →
  Real.cos θ = -1 →
  b = (3, -6) := by sorry

end NUMINAMATH_CALUDE_vector_b_coordinates_l3096_309603


namespace NUMINAMATH_CALUDE_k_at_negative_eight_l3096_309685

-- Define the polynomial h
def h (x : ℝ) : ℝ := x^3 - x - 2

-- Define the property that k is a cubic polynomial with the given conditions
def is_valid_k (k : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, h x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    (∀ x, k x = 0 ↔ x = a^2 ∨ x = b^2 ∨ x = c^2) ∧
    k 0 = 2

-- Theorem statement
theorem k_at_negative_eight (k : ℝ → ℝ) (hk : is_valid_k k) : k (-8) = -20 := by
  sorry

end NUMINAMATH_CALUDE_k_at_negative_eight_l3096_309685


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l3096_309631

/-- Proves that the cost price of a computer table is 2500 when the selling price is 3000 
    and the markup is 20% -/
theorem computer_table_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) : 
  markup_percentage = 20 →
  selling_price = 3000 →
  (100 + markup_percentage) / 100 * (selling_price / (1 + markup_percentage / 100)) = 2500 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l3096_309631


namespace NUMINAMATH_CALUDE_unique_seven_l3096_309661

/-- A function that returns true if the given positive integer n results in
    exactly one term with a rational coefficient in the binomial expansion
    of (√3x + ∛2)^n -/
def has_one_rational_term (n : ℕ+) : Prop :=
  ∃! r : ℕ, r ≤ n ∧ 3 ∣ r ∧ 2 ∣ (n - r)

/-- Theorem stating that 7 is the only positive integer satisfying the condition -/
theorem unique_seven : ∀ n : ℕ+, has_one_rational_term n ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_seven_l3096_309661


namespace NUMINAMATH_CALUDE_interest_rate_equivalence_l3096_309674

/-- Given an amount A that produces the same interest in 12 years as Rs 1000 produces in 2 years at 12%,
    prove that the interest rate R for amount A is 12%. -/
theorem interest_rate_equivalence (A : ℝ) (R : ℝ) : A > 0 →
  A * R * 12 = 1000 * 12 * 2 →
  R = 12 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equivalence_l3096_309674


namespace NUMINAMATH_CALUDE_bobbo_river_crossing_l3096_309666

/-- Bobbo's river crossing problem -/
theorem bobbo_river_crossing 
  (river_width : ℝ)
  (initial_speed : ℝ)
  (current_speed : ℝ)
  (waterfall_distance : ℝ)
  (h_river_width : river_width = 100)
  (h_initial_speed : initial_speed = 2)
  (h_current_speed : current_speed = 5)
  (h_waterfall_distance : waterfall_distance = 175) :
  let midway_distance := river_width / 2
  let time_to_midway := midway_distance / initial_speed
  let downstream_distance := current_speed * time_to_midway
  let remaining_distance := waterfall_distance - downstream_distance
  let time_left := remaining_distance / current_speed
  let required_speed := midway_distance / time_left
  required_speed - initial_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_bobbo_river_crossing_l3096_309666


namespace NUMINAMATH_CALUDE_square_root_of_8_factorial_over_70_l3096_309667

theorem square_root_of_8_factorial_over_70 : 
  let factorial_8 := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  Real.sqrt (factorial_8 / 70) = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_8_factorial_over_70_l3096_309667


namespace NUMINAMATH_CALUDE_sin_minus_cos_for_specific_tan_l3096_309616

theorem sin_minus_cos_for_specific_tan (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_for_specific_tan_l3096_309616


namespace NUMINAMATH_CALUDE_fraction_equality_y_value_l3096_309620

theorem fraction_equality_y_value (a b c d y : ℚ) 
  (h1 : a ≠ b) 
  (h2 : a ≠ 0) 
  (h3 : c ≠ d) 
  (h4 : (b + y) / (a + y) = d / c) : 
  y = (a * d - b * c) / (c - d) := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_y_value_l3096_309620


namespace NUMINAMATH_CALUDE_coordinate_line_segments_l3096_309626

theorem coordinate_line_segments (n : ℕ) (h : n > 2) :
  ∃ (points : Fin n → ℝ),
    (∀ i : Fin n, i.val ≠ 0 → i.val ≠ n - 1 →
      ∃ j k : Fin n, points i = (points j + points k) / 2) ∧
    (∀ i j : Fin n, i.val + 1 = j.val →
      ∀ k l : Fin n, k.val + 1 = l.val →
        i ≠ k → points j - points i ≠ points l - points k) :=
by sorry

end NUMINAMATH_CALUDE_coordinate_line_segments_l3096_309626


namespace NUMINAMATH_CALUDE_fraction_of_students_with_B_l3096_309694

theorem fraction_of_students_with_B (fraction_A : Real) (fraction_A_or_B : Real) 
  (h1 : fraction_A = 0.7)
  (h2 : fraction_A_or_B = 0.9) :
  fraction_A_or_B - fraction_A = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_students_with_B_l3096_309694


namespace NUMINAMATH_CALUDE_certain_number_equation_l3096_309686

theorem certain_number_equation (x : ℝ) : 5100 - (x / 20.4) = 5095 ↔ x = 102 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3096_309686


namespace NUMINAMATH_CALUDE_hair_cut_second_day_l3096_309606

/-- The amount of hair cut off on the second day, given the total amount cut off and the amount cut off on the first day. -/
theorem hair_cut_second_day 
  (total_cut : ℝ) 
  (first_day_cut : ℝ) 
  (h1 : total_cut = 0.875) 
  (h2 : first_day_cut = 0.375) : 
  total_cut - first_day_cut = 0.500 := by
sorry

end NUMINAMATH_CALUDE_hair_cut_second_day_l3096_309606


namespace NUMINAMATH_CALUDE_toms_family_stay_l3096_309604

/-- Calculates the number of days Tom's family stayed at his house -/
def days_at_toms_house (total_people : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ) (total_plates_used : ℕ) : ℕ :=
  total_plates_used / (total_people * meals_per_day * plates_per_meal)

/-- Proves that Tom's family stayed for 4 days given the problem conditions -/
theorem toms_family_stay : 
  days_at_toms_house 6 3 2 144 = 4 := by
  sorry

end NUMINAMATH_CALUDE_toms_family_stay_l3096_309604
