import Mathlib

namespace NUMINAMATH_CALUDE_max_value_proof_l2093_209309

theorem max_value_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  (abcd : ℝ) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_proof_l2093_209309


namespace NUMINAMATH_CALUDE_family_boys_count_l2093_209300

/-- A family where one child has 3 brothers and 6 sisters, and another child has 4 brothers and 5 sisters -/
structure Family where
  total_children : ℕ
  child1_brothers : ℕ
  child1_sisters : ℕ
  child2_brothers : ℕ
  child2_sisters : ℕ
  h1 : child1_brothers = 3
  h2 : child1_sisters = 6
  h3 : child2_brothers = 4
  h4 : child2_sisters = 5

/-- The number of boys in the family -/
def num_boys (f : Family) : ℕ := f.child1_brothers + 1

theorem family_boys_count (f : Family) : num_boys f = 4 := by
  sorry

end NUMINAMATH_CALUDE_family_boys_count_l2093_209300


namespace NUMINAMATH_CALUDE_product_of_good_sequences_is_good_l2093_209325

/-- A sequence is a function from natural numbers to real numbers. -/
def Sequence := ℕ → ℝ

/-- The first derivative of a sequence. -/
def firstDerivative (a : Sequence) : Sequence :=
  λ n => a (n + 1) - a n

/-- The k-th derivative of a sequence. -/
def kthDerivative : ℕ → Sequence → Sequence
  | 0, a => a
  | k + 1, a => firstDerivative (kthDerivative k a)

/-- A sequence is good if it and all its derivatives consist of positive numbers. -/
def isGoodSequence (a : Sequence) : Prop :=
  ∀ k n, kthDerivative k a n > 0

/-- The element-wise product of two sequences. -/
def productSequence (a b : Sequence) : Sequence :=
  λ n => a n * b n

/-- Theorem: The element-wise product of two good sequences is also a good sequence. -/
theorem product_of_good_sequences_is_good (a b : Sequence) 
  (ha : isGoodSequence a) (hb : isGoodSequence b) : 
  isGoodSequence (productSequence a b) := by
  sorry

end NUMINAMATH_CALUDE_product_of_good_sequences_is_good_l2093_209325


namespace NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_perpendicular_parallel_planes_l2093_209350

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_perpendicular
  (α : Plane) (m n : Line)
  (h1 : parallel_line_plane m α)
  (h2 : perpendicular_line_plane n α) :
  perpendicular_line_line m n :=
sorry

-- Theorem 2
theorem perpendicular_parallel_planes
  (α β : Plane) (m : Line)
  (h1 : perpendicular_line_plane m α)
  (h2 : parallel_plane_plane α β) :
  perpendicular_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_perpendicular_parallel_planes_l2093_209350


namespace NUMINAMATH_CALUDE_maximum_value_of_x_plus_reciprocal_x_l2093_209333

theorem maximum_value_of_x_plus_reciprocal_x (x : ℝ) :
  x < 0 → ∃ (max : ℝ), (∀ y, y < 0 → y + 1/y ≤ max) ∧ max = -2 :=
sorry


end NUMINAMATH_CALUDE_maximum_value_of_x_plus_reciprocal_x_l2093_209333


namespace NUMINAMATH_CALUDE_candied_grape_price_l2093_209342

-- Define the number of candied apples
def num_apples : ℕ := 15

-- Define the price of each candied apple
def price_apple : ℚ := 2

-- Define the number of candied grapes
def num_grapes : ℕ := 12

-- Define the total revenue
def total_revenue : ℚ := 48

-- Define the price of each candied grape
def price_grape : ℚ := 1.5

theorem candied_grape_price :
  price_grape * num_grapes + price_apple * num_apples = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_candied_grape_price_l2093_209342


namespace NUMINAMATH_CALUDE_ball_weight_problem_l2093_209332

theorem ball_weight_problem (R W : ℚ) 
  (eq1 : 7 * R + 5 * W = 43)
  (eq2 : 5 * R + 7 * W = 47) :
  4 * R + 8 * W = 49 := by
  sorry

end NUMINAMATH_CALUDE_ball_weight_problem_l2093_209332


namespace NUMINAMATH_CALUDE_surfer_wave_height_l2093_209315

/-- Represents the height of the highest wave caught by a surfer. -/
def highest_wave (H : ℝ) : ℝ := 4 * H + 2

/-- Represents the height of the shortest wave caught by a surfer. -/
def shortest_wave (H : ℝ) : ℝ := H + 4

theorem surfer_wave_height (H : ℝ) 
  (h1 : shortest_wave H = 7 + 3) 
  (h2 : shortest_wave H = H + 4) : 
  highest_wave H = 26 := by
  sorry

end NUMINAMATH_CALUDE_surfer_wave_height_l2093_209315


namespace NUMINAMATH_CALUDE_garden_length_l2093_209323

theorem garden_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = 2 + 3 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 100 →
  length = 38 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l2093_209323


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l2093_209349

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_arithmetic_sequence_ones_digit 
  (p q r : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hr : is_prime r) 
  (h_seq : q = p + 4 ∧ r = q + 4) 
  (h_p_gt_5 : p > 5) :
  ones_digit p = 3 ∨ ones_digit p = 9 :=
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l2093_209349


namespace NUMINAMATH_CALUDE_circle_equation_l2093_209389

/-- The equation of a circle with center (0, 4) passing through (3, 0) is x² + (y - 4)² = 25 -/
theorem circle_equation (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ x^2 + (y - 4)^2 = r^2) ∧ 
  (3^2 + (0 - 4)^2 = x^2 + (y - 4)^2) → 
  x^2 + (y - 4)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l2093_209389


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l2093_209307

/-- Proves that a cyclist traveling 136.4 km in 6 hours and 30 minutes has an average speed of approximately 5.83 m/s -/
theorem cyclist_average_speed :
  let distance_km : ℝ := 136.4
  let time_hours : ℝ := 6.5
  let distance_m : ℝ := distance_km * 1000
  let time_s : ℝ := time_hours * 3600
  let average_speed : ℝ := distance_m / time_s
  ∃ ε > 0, |average_speed - 5.83| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_cyclist_average_speed_l2093_209307


namespace NUMINAMATH_CALUDE_positive_integers_equality_l2093_209304

theorem positive_integers_equality (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (4 * a * b - 1) ∣ ((4 * a^2 - 1)^2) → a = b := by
  sorry

end NUMINAMATH_CALUDE_positive_integers_equality_l2093_209304


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2093_209388

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}

theorem intersection_with_complement : S ∩ (U \ T) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2093_209388


namespace NUMINAMATH_CALUDE_house_elves_do_not_exist_l2093_209353

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (HouseElf : U → Prop)
variable (LovesPranks : U → Prop)
variable (LovesCleanlinessAndOrder : U → Prop)

-- State the theorem
theorem house_elves_do_not_exist :
  (∀ x, HouseElf x → LovesPranks x) →
  (∀ x, HouseElf x → LovesCleanlinessAndOrder x) →
  (∀ x, LovesCleanlinessAndOrder x → ¬LovesPranks x) →
  ¬(∃ x, HouseElf x) :=
by
  sorry

end NUMINAMATH_CALUDE_house_elves_do_not_exist_l2093_209353


namespace NUMINAMATH_CALUDE_g_composition_result_l2093_209301

noncomputable def g (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^3 else -z^3

theorem g_composition_result :
  g (g (g (g (1 + I)))) = -134217728 - 134217728 * I :=
by sorry

end NUMINAMATH_CALUDE_g_composition_result_l2093_209301


namespace NUMINAMATH_CALUDE_janes_trip_distance_l2093_209366

theorem janes_trip_distance :
  ∀ (total_distance : ℝ),
  (1/4 : ℝ) * total_distance +     -- First part (highway)
  30 +                             -- Second part (city streets)
  (1/6 : ℝ) * total_distance       -- Third part (country roads)
  = total_distance                 -- Sum of all parts equals total distance
  →
  total_distance = 360/7 := by
sorry

end NUMINAMATH_CALUDE_janes_trip_distance_l2093_209366


namespace NUMINAMATH_CALUDE_semiperimeter_radius_sum_eq_legs_sum_l2093_209311

/-- A right triangle with legs a and b, hypotenuse c, semiperimeter p, and inscribed circle radius r -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  r : ℝ
  right_angle : c^2 = a^2 + b^2
  semiperimeter : p = (a + b + c) / 2
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The sum of the semiperimeter and the radius of the inscribed circle is equal to the sum of the legs -/
theorem semiperimeter_radius_sum_eq_legs_sum (t : RightTriangle) : t.p + t.r = t.a + t.b := by
  sorry

end NUMINAMATH_CALUDE_semiperimeter_radius_sum_eq_legs_sum_l2093_209311


namespace NUMINAMATH_CALUDE_triangle_side_length_l2093_209334

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 4)
  (h2 : t.b = 5)
  (h3 : t.S = 5 * Real.sqrt 3) :
  t.c = Real.sqrt 21 ∨ t.c = Real.sqrt 61 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2093_209334


namespace NUMINAMATH_CALUDE_triangle_area_l2093_209385

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  (1/2) * a * b = 54 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2093_209385


namespace NUMINAMATH_CALUDE_sin_symmetric_angles_l2093_209338

def symmetric_angles (α β : Real) : Prop :=
  ∃ k : Int, α + β = Real.pi + 2 * k * Real.pi

theorem sin_symmetric_angles (α β : Real) 
  (h_symmetric : symmetric_angles α β) (h_sin_α : Real.sin α = 1/3) : 
  Real.sin β = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_symmetric_angles_l2093_209338


namespace NUMINAMATH_CALUDE_donation_to_first_home_l2093_209314

theorem donation_to_first_home 
  (total_donation : ℝ) 
  (second_home_donation : ℝ) 
  (third_home_donation : ℝ) 
  (h1 : total_donation = 700)
  (h2 : second_home_donation = 225)
  (h3 : third_home_donation = 230) :
  total_donation - second_home_donation - third_home_donation = 245 :=
by sorry

end NUMINAMATH_CALUDE_donation_to_first_home_l2093_209314


namespace NUMINAMATH_CALUDE_parabola_equation_l2093_209308

/-- A vertical line passing through a point (x₀, y₀) -/
structure VerticalLine where
  x₀ : ℝ
  y₀ : ℝ

/-- A parabola with vertical axis and vertex at origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := λ x y => y^2 = -2 * p * x

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation (l : VerticalLine) (para : Parabola) :
  l.x₀ = 3/2 ∧ l.y₀ = 2 ∧ para.eq = λ x y => y^2 = -6*x := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2093_209308


namespace NUMINAMATH_CALUDE_tan_graph_product_l2093_209396

theorem tan_graph_product (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = 3 → x = π / 8) →
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π / 2))) →
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_tan_graph_product_l2093_209396


namespace NUMINAMATH_CALUDE_intersection_polygon_exists_and_unique_l2093_209397

/- Define the cube and points -/
def cube_edge_length : ℝ := 30

/- Define points on cube edges -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨cube_edge_length, 0, 0⟩
def C : Point3D := ⟨cube_edge_length, 0, cube_edge_length⟩
def D : Point3D := ⟨cube_edge_length, cube_edge_length, cube_edge_length⟩

def P : Point3D := ⟨10, 0, 0⟩
def Q : Point3D := ⟨cube_edge_length, 0, 10⟩
def R : Point3D := ⟨cube_edge_length, 15, cube_edge_length⟩

/- Define the plane PQR -/
def plane_PQR (x y z : ℝ) : Prop := 2*x + y - 2*z = 15

/- Define the cube -/
def in_cube (p : Point3D) : Prop :=
  0 ≤ p.x ∧ p.x ≤ cube_edge_length ∧
  0 ≤ p.y ∧ p.y ≤ cube_edge_length ∧
  0 ≤ p.z ∧ p.z ≤ cube_edge_length

/- Theorem statement -/
theorem intersection_polygon_exists_and_unique :
  ∃! polygon : Set Point3D,
    (∀ p ∈ polygon, in_cube p ∧ plane_PQR p.x p.y p.z) ∧
    (∀ p, in_cube p ∧ plane_PQR p.x p.y p.z → p ∈ polygon) :=
sorry

end NUMINAMATH_CALUDE_intersection_polygon_exists_and_unique_l2093_209397


namespace NUMINAMATH_CALUDE_expression_simplification_l2093_209369

theorem expression_simplification :
  (1 / ((1 / (Real.sqrt 2 + 1)) + (1 / (Real.sqrt 5 - 2)))) = 
  ((Real.sqrt 2 + Real.sqrt 5 - 1) / (6 + 2 * Real.sqrt 10)) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2093_209369


namespace NUMINAMATH_CALUDE_valid_allocations_count_l2093_209341

/-- The number of student volunteers --/
def num_students : ℕ := 5

/-- The number of display boards --/
def num_boards : ℕ := 2

/-- A function that calculates the number of ways to allocate students to display boards --/
def allocation_schemes (n : ℕ) (k : ℕ) (min_per_board : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid allocation schemes --/
theorem valid_allocations_count :
  allocation_schemes num_students num_boards 2 = 12 :=
sorry

end NUMINAMATH_CALUDE_valid_allocations_count_l2093_209341


namespace NUMINAMATH_CALUDE_max_investment_at_lower_rate_l2093_209378

theorem max_investment_at_lower_rate 
  (total_investment : ℝ) 
  (low_rate high_rate : ℝ) 
  (min_interest : ℝ) 
  (h1 : total_investment = 25000)
  (h2 : low_rate = 0.07)
  (h3 : high_rate = 0.12)
  (h4 : min_interest = 2450) :
  let max_low_investment := 11000
  ∀ x : ℝ, 
    0 ≤ x ∧ 
    x ≤ total_investment ∧ 
    low_rate * x + high_rate * (total_investment - x) ≥ min_interest →
    x ≤ max_low_investment := by
sorry

end NUMINAMATH_CALUDE_max_investment_at_lower_rate_l2093_209378


namespace NUMINAMATH_CALUDE_integer_power_sum_l2093_209375

theorem integer_power_sum (x : ℝ) (h : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l2093_209375


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2093_209376

/-- Define the @ operation for real numbers -/
def at_op (p q : ℝ) : ℝ := p + q - p * q

/-- The theorem statement -/
theorem inequality_system_solution_range (m : ℝ) :
  (∃! (a b : ℤ), (a ≠ b) ∧ 
    (at_op 2 (a : ℝ) > 0) ∧ (at_op (a : ℝ) 3 ≤ m) ∧
    (at_op 2 (b : ℝ) > 0) ∧ (at_op (b : ℝ) 3 ≤ m) ∧
    (∀ x : ℤ, x ≠ a ∧ x ≠ b → 
      ¬((at_op 2 (x : ℝ) > 0) ∧ (at_op (x : ℝ) 3 ≤ m))))
  → 3 ≤ m ∧ m < 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2093_209376


namespace NUMINAMATH_CALUDE_subtract_point_five_from_47_point_two_l2093_209379

theorem subtract_point_five_from_47_point_two : 47.2 - 0.5 = 46.7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_point_five_from_47_point_two_l2093_209379


namespace NUMINAMATH_CALUDE_curve_symmetric_about_origin_l2093_209356

/-- A curve defined by the equation xy - x^2 = 1 -/
def curve (x y : ℝ) : Prop := x * y - x^2 = 1

/-- Symmetry about the origin for the curve -/
theorem curve_symmetric_about_origin :
  ∀ x y : ℝ, curve x y ↔ curve (-x) (-y) :=
sorry

end NUMINAMATH_CALUDE_curve_symmetric_about_origin_l2093_209356


namespace NUMINAMATH_CALUDE_sequence_difference_l2093_209322

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

def geometric_sequence (g₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := g₁ * r^(n - 1 : ℕ)

theorem sequence_difference : 
  let a₁ := 3
  let a₂ := 11
  let g₁ := 2
  let g₂ := 10
  let d := a₂ - a₁
  let r := g₂ / g₁
  |arithmetic_sequence a₁ d 100 - geometric_sequence g₁ r 4| = 545 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l2093_209322


namespace NUMINAMATH_CALUDE_kenny_basketball_time_l2093_209393

def trumpet_practice : ℕ := 40

theorem kenny_basketball_time (run_time trumpet_time basketball_time : ℕ) 
  (h1 : trumpet_time = trumpet_practice)
  (h2 : trumpet_time = 2 * run_time)
  (h3 : run_time = 2 * basketball_time) : 
  basketball_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_kenny_basketball_time_l2093_209393


namespace NUMINAMATH_CALUDE_museum_art_count_l2093_209361

theorem museum_art_count (total : ℕ) (asian : ℕ) (egyptian : ℕ) (european : ℕ) 
  (h1 : total = 2500)
  (h2 : asian = 465)
  (h3 : egyptian = 527)
  (h4 : european = 320) :
  total - (asian + egyptian + european) = 1188 := by
  sorry

end NUMINAMATH_CALUDE_museum_art_count_l2093_209361


namespace NUMINAMATH_CALUDE_spring_experiment_l2093_209360

/-- Spring experiment data points -/
def spring_data : List (ℝ × ℝ) := [(0, 20), (1, 22), (2, 24), (3, 26), (4, 28), (5, 30)]

/-- The relationship between spring length y (in cm) and weight x (in kg) -/
def spring_relation (x y : ℝ) : Prop := y = 2 * x + 20

/-- Theorem stating that the spring_relation holds for all data points in spring_data -/
theorem spring_experiment :
  ∀ (point : ℝ × ℝ), point ∈ spring_data → spring_relation point.1 point.2 := by
  sorry

end NUMINAMATH_CALUDE_spring_experiment_l2093_209360


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2093_209305

theorem cubic_roots_sum_cubes (α β γ : ℂ) : 
  (8 * α^3 + 2012 * α + 2013 = 0) →
  (8 * β^3 + 2012 * β + 2013 = 0) →
  (8 * γ^3 + 2012 * γ + 2013 = 0) →
  (α + β)^3 + (β + γ)^3 + (γ + α)^3 = 6039 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2093_209305


namespace NUMINAMATH_CALUDE_log_relation_l2093_209371

theorem log_relation (a b : ℝ) (h1 : a = Real.log 256 / Real.log 4) (h2 : b = Real.log 27 / Real.log 3) :
  a = (4/3) * b := by sorry

end NUMINAMATH_CALUDE_log_relation_l2093_209371


namespace NUMINAMATH_CALUDE_ron_eats_24_slices_l2093_209362

/-- The number of pickle slices Sammy can eat -/
def sammy_slices : ℕ := 15

/-- Tammy can eat twice as many pickle slices as Sammy -/
def tammy_slices : ℕ := 2 * sammy_slices

/-- Ron eats 20% fewer pickle slices than Tammy -/
def ron_slices : ℕ := tammy_slices - (tammy_slices * 20 / 100)

/-- Theorem stating that Ron eats 24 pickle slices -/
theorem ron_eats_24_slices : ron_slices = 24 := by sorry

end NUMINAMATH_CALUDE_ron_eats_24_slices_l2093_209362


namespace NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l2093_209382

/-- Represents a gathering of people -/
structure Gathering where
  total : Nat
  group1 : Nat
  group2 : Nat
  group1_knows_each_other : Bool
  group2_knows_no_one : Bool

/-- Calculates the number of handshakes in a gathering -/
def count_handshakes (g : Gathering) : Nat :=
  if g.group1_knows_each_other && g.group2_knows_no_one then
    (g.group2 * (g.total - 1)) / 2
  else
    0  -- This case is not relevant for our specific problem

theorem handshakes_in_specific_gathering :
  let g : Gathering := {
    total := 30,
    group1 := 20,
    group2 := 10,
    group1_knows_each_other := true,
    group2_knows_no_one := true
  }
  count_handshakes g = 145 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l2093_209382


namespace NUMINAMATH_CALUDE_train_platform_ratio_l2093_209321

/-- Proves that the ratio of train length to platform length is 1:1 given the specified conditions --/
theorem train_platform_ratio (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 216 * (1000 / 3600) →
  train_length = 1800 →
  crossing_time = 60 →
  ∃ (platform_length : ℝ), train_length / platform_length = 1 := by
  sorry


end NUMINAMATH_CALUDE_train_platform_ratio_l2093_209321


namespace NUMINAMATH_CALUDE_managers_salary_l2093_209326

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (salary_increase : ℚ) :
  num_employees = 20 →
  avg_salary = 1300 →
  salary_increase = 100 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) / (num_employees + 1) - avg_salary = salary_increase →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) - (num_employees * avg_salary) = 3400 :=
by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l2093_209326


namespace NUMINAMATH_CALUDE_floor_tiles_count_l2093_209339

/-- Represents a square floor tiled with square tiles -/
structure SquareFloor where
  side_length : ℕ
  is_square : side_length > 0

/-- The number of black tiles on the diagonals of a square floor -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- The total number of tiles on a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

theorem floor_tiles_count (floor : SquareFloor) 
  (h : black_tiles floor = 101) : 
  total_tiles floor = 2601 := by
  sorry

end NUMINAMATH_CALUDE_floor_tiles_count_l2093_209339


namespace NUMINAMATH_CALUDE_no_such_polyhedron_l2093_209391

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ
  is_convex : Bool
  no_triangular_faces : Bool
  no_three_valent_vertices : Bool

/-- Euler's formula for polyhedra -/
def euler_formula (p : ConvexPolyhedron) : Prop :=
  p.faces + p.vertices - p.edges = 2

/-- Theorem: A convex polyhedron with no triangular faces and no three-valent vertices violates Euler's formula -/
theorem no_such_polyhedron (p : ConvexPolyhedron) 
  (h_convex : p.is_convex = true) 
  (h_no_tri : p.no_triangular_faces = true) 
  (h_no_three : p.no_three_valent_vertices = true) : 
  ¬(euler_formula p) := by
  sorry

end NUMINAMATH_CALUDE_no_such_polyhedron_l2093_209391


namespace NUMINAMATH_CALUDE_set_intersection_problem_l2093_209336

def A : Set ℝ := {1, 2, 6}
def B : Set ℝ := {2, 4}
def C : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 5}

theorem set_intersection_problem : (A ∪ B) ∩ C = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l2093_209336


namespace NUMINAMATH_CALUDE_sum_odd_and_even_integers_l2093_209399

def sum_odd_integers (n : ℕ) : ℕ := 
  (n^2 + n) / 2

def sum_even_integers (n : ℕ) : ℕ := 
  n * (n + 1)

theorem sum_odd_and_even_integers : 
  sum_odd_integers 111 + sum_even_integers 25 = 3786 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_and_even_integers_l2093_209399


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2093_209372

theorem shaded_area_calculation (R : ℝ) (d : ℝ) (h1 : R = 10) (h2 : d = 8) : 
  let r : ℝ := Real.sqrt (R^2 - d^2)
  let large_circle_area : ℝ := π * R^2
  let small_circle_area : ℝ := 2 * π * r^2
  large_circle_area - small_circle_area = 28 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2093_209372


namespace NUMINAMATH_CALUDE_water_jar_problem_l2093_209367

theorem water_jar_problem (small_jar large_jar : ℝ) (h1 : small_jar > 0) (h2 : large_jar > 0) 
  (h3 : small_jar ≠ large_jar) (water : ℝ) (h4 : water > 0)
  (h5 : water / small_jar = 1 / 7) (h6 : water / large_jar = 1 / 6) :
  (2 * water) / large_jar = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_water_jar_problem_l2093_209367


namespace NUMINAMATH_CALUDE_orange_pricing_theorem_l2093_209335

/-- Represents a pricing scheme for oranges -/
structure PricingScheme where
  oranges : ℕ
  price : ℕ

/-- Calculates the minimum cost for buying a given number of oranges -/
def minCost (schemes : List PricingScheme) (totalOranges : ℕ) : ℕ :=
  sorry

/-- Calculates the average cost per orange -/
def avgCost (totalCost : ℕ) (totalOranges : ℕ) : ℚ :=
  sorry

theorem orange_pricing_theorem (schemes : List PricingScheme) (totalOranges : ℕ) :
  schemes = [⟨4, 12⟩, ⟨7, 30⟩] →
  totalOranges = 20 →
  avgCost (minCost schemes totalOranges) totalOranges = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_pricing_theorem_l2093_209335


namespace NUMINAMATH_CALUDE_profit_7500_at_65_max_profit_at_70_max_profit_is_8000_l2093_209327

/-- Represents the online store's pricing and sales model -/
structure Store where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ

/-- Calculates the number of items sold based on the current price -/
def items_sold (s : Store) (price : ℝ) : ℝ :=
  s.initial_sales + s.price_sensitivity * (s.initial_price - price)

/-- Calculates the weekly profit based on the current price -/
def weekly_profit (s : Store) (price : ℝ) : ℝ :=
  (price - s.cost_price) * (items_sold s price)

/-- The store's pricing and sales model -/
def children_clothing_store : Store :=
  { cost_price := 50
  , initial_price := 80
  , initial_sales := 200
  , price_sensitivity := 20 }

/-- Theorem: The selling price of 65 yuan achieves a weekly profit of 7500 yuan while maximizing customer benefits -/
theorem profit_7500_at_65 :
  weekly_profit children_clothing_store 65 = 7500 ∧
  ∀ p, p < 65 → weekly_profit children_clothing_store p < 7500 :=
sorry

/-- Theorem: The selling price of 70 yuan maximizes the weekly profit -/
theorem max_profit_at_70 :
  ∀ p, weekly_profit children_clothing_store p ≤ weekly_profit children_clothing_store 70 :=
sorry

/-- Theorem: The maximum weekly profit is 8000 yuan -/
theorem max_profit_is_8000 :
  weekly_profit children_clothing_store 70 = 8000 :=
sorry

end NUMINAMATH_CALUDE_profit_7500_at_65_max_profit_at_70_max_profit_is_8000_l2093_209327


namespace NUMINAMATH_CALUDE_largest_base_for_12_cubed_digit_sum_base_8_sum_not_9_l2093_209320

def base_representation (n : ℕ) (b : ℕ) : List ℕ :=
  sorry

def sum_of_digits (n : ℕ) (b : ℕ) : ℕ :=
  (base_representation n b).sum

def power_in_base (base : ℕ) (n : ℕ) (power : ℕ) : ℕ :=
  sorry

theorem largest_base_for_12_cubed_digit_sum :
  ∀ b : ℕ, b > 8 → sum_of_digits (power_in_base b 12 3) b = 9 :=
by sorry

theorem base_8_sum_not_9 :
  sum_of_digits (power_in_base 8 12 3) 8 ≠ 9 :=
by sorry

end NUMINAMATH_CALUDE_largest_base_for_12_cubed_digit_sum_base_8_sum_not_9_l2093_209320


namespace NUMINAMATH_CALUDE_coefficient_a2_l2093_209374

/-- Given z = 1/2 + (√3/2)i and (x-z)^4 = a₀x^4 + a₁x^3 + a₂x^2 + a₃x + a₄, prove that a₂ = -3 + 3√3i. -/
theorem coefficient_a2 (z : ℂ) (a₀ a₁ a₂ a₃ a₄ : ℂ) :
  z = (1 : ℂ) / 2 + (Complex.I * Real.sqrt 3) / 2 →
  (∀ x : ℂ, (x - z)^4 = a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) →
  a₂ = -3 + 3 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a2_l2093_209374


namespace NUMINAMATH_CALUDE_quadratic_rational_root_implies_even_coefficient_l2093_209302

theorem quadratic_rational_root_implies_even_coefficient
  (a b c : ℤ)
  (h_a_nonzero : a ≠ 0)
  (h_rational_root : ∃ (x : ℚ), a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_implies_even_coefficient_l2093_209302


namespace NUMINAMATH_CALUDE_multiplication_commutativity_certainty_l2093_209394

theorem multiplication_commutativity_certainty :
  ∀ (a b : ℝ), a * b = b * a := by
  sorry

end NUMINAMATH_CALUDE_multiplication_commutativity_certainty_l2093_209394


namespace NUMINAMATH_CALUDE_ratio_problem_l2093_209345

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) :
  x / y = 11 / 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2093_209345


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2093_209352

theorem sum_of_four_numbers (a b c d : ℤ) 
  (sum_abc : a + b + c = 415)
  (sum_abd : a + b + d = 442)
  (sum_acd : a + c + d = 396)
  (sum_bcd : b + c + d = 325) :
  a + b + c + d = 526 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2093_209352


namespace NUMINAMATH_CALUDE_no_solution_greater_than_two_l2093_209348

theorem no_solution_greater_than_two (n : ℕ) (h : n > 2) :
  ¬ (3^(n-1) + 5^(n-1) ∣ 3^n + 5^n) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_greater_than_two_l2093_209348


namespace NUMINAMATH_CALUDE_sanchez_rope_theorem_l2093_209392

/-- Represents the number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Represents the number of feet of rope bought last week -/
def last_week_feet : ℕ := 6

/-- Represents the difference in feet between last week's and this week's purchase -/
def difference_feet : ℕ := 4

/-- Calculates the total inches of rope bought by Mr. Sanchez -/
def total_inches : ℕ := 
  (last_week_feet * inches_per_foot) + ((last_week_feet - difference_feet) * inches_per_foot)

/-- Theorem stating that the total inches of rope bought is 96 -/
theorem sanchez_rope_theorem : total_inches = 96 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_rope_theorem_l2093_209392


namespace NUMINAMATH_CALUDE_complement_of_P_l2093_209319

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x | x - 1 < 0}

-- State the theorem
theorem complement_of_P : Set.compl P = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_l2093_209319


namespace NUMINAMATH_CALUDE_f_sum_2006_2007_l2093_209331

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_periodic : ∀ x : ℝ, f (x + 4) = f x + f 2
axiom f_1 : f 1 = 2

-- State the theorem
theorem f_sum_2006_2007 : f 2006 + f 2007 = 2 := by sorry

end NUMINAMATH_CALUDE_f_sum_2006_2007_l2093_209331


namespace NUMINAMATH_CALUDE_clothes_pricing_l2093_209312

/-- Given a total spend and a price relation between shirt and trousers,
    prove the individual costs of the shirt and trousers. -/
theorem clothes_pricing (total : ℕ) (shirt_price trousers_price : ℕ) 
    (h1 : total = 185)
    (h2 : shirt_price = 2 * trousers_price + 5)
    (h3 : total = shirt_price + trousers_price) :
    trousers_price = 60 ∧ shirt_price = 125 := by
  sorry

end NUMINAMATH_CALUDE_clothes_pricing_l2093_209312


namespace NUMINAMATH_CALUDE_vector_addition_l2093_209313

def a : Fin 2 → ℝ := ![3, 1]
def b : Fin 2 → ℝ := ![-2, 5]

theorem vector_addition : 2 • a + b = ![4, 7] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l2093_209313


namespace NUMINAMATH_CALUDE_product_301_52_base7_units_digit_l2093_209377

theorem product_301_52_base7_units_digit (a b : ℕ) (ha : a = 301) (hb : b = 52) :
  (a * b) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_301_52_base7_units_digit_l2093_209377


namespace NUMINAMATH_CALUDE_negation_of_even_sum_false_l2093_209383

theorem negation_of_even_sum_false : 
  ¬(∀ a b : ℤ, (¬(Even a ∧ Even b) → ¬Even (a + b))) := by sorry

end NUMINAMATH_CALUDE_negation_of_even_sum_false_l2093_209383


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l2093_209306

/-- Given a geometric sequence {a_n} where the sum of its first n terms is 2^n - 1,
    this function computes the sum of the first n terms of the sequence {a_n^2}. -/
def sum_of_squares (n : ℕ) : ℚ :=
  (4^n - 1) / 3

/-- The sum of the first n terms of the original geometric sequence {a_n}. -/
def sum_of_original (n : ℕ) : ℕ :=
  2^n - 1

theorem sum_of_squares_theorem (n : ℕ) :
  sum_of_squares n = (4^n - 1) / 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l2093_209306


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2093_209381

theorem consecutive_integers_square_sum : 
  ∀ a : ℤ, a > 0 → 
  ((a - 1) * a * (a + 1) = 12 * (3 * a)) → 
  ((a - 1)^2 + a^2 + (a + 1)^2 = 77) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2093_209381


namespace NUMINAMATH_CALUDE_common_terms_k_polygonal_fermat_l2093_209357

/-- k-polygonal number sequence -/
def kPolygonalSeq (k : ℕ) (n : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

/-- Fermat number sequence -/
def fermatSeq (m : ℕ) : ℕ :=
  2^(2^m) + 1

/-- Proposition: The only positive integers k > 2 for which there exist common terms
    between the k-polygonal numbers sequence and the Fermat numbers sequence are 3 and 5 -/
theorem common_terms_k_polygonal_fermat :
  {k : ℕ | k > 2 ∧ ∃ (n m : ℕ), kPolygonalSeq k n = fermatSeq m} = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_common_terms_k_polygonal_fermat_l2093_209357


namespace NUMINAMATH_CALUDE_rabbits_total_distance_l2093_209329

/-- The total distance hopped by two rabbits in a given time -/
def total_distance (white_speed brown_speed time : ℕ) : ℕ :=
  (white_speed * time) + (brown_speed * time)

/-- Theorem: The total distance hopped by the white and brown rabbits in 5 minutes is 135 meters -/
theorem rabbits_total_distance :
  total_distance 15 12 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_total_distance_l2093_209329


namespace NUMINAMATH_CALUDE_vanya_finished_first_l2093_209363

/-- Represents a participant in the competition -/
structure Participant where
  name : String
  predicted_position : Nat
  actual_position : Nat

/-- The competition setup and results -/
structure Competition where
  participants : List Participant
  vanya : Participant

/-- Axioms for the competition -/
axiom all_positions_different (c : Competition) :
  ∀ p1 p2 : Participant, p1 ∈ c.participants → p2 ∈ c.participants → p1 ≠ p2 →
    p1.actual_position ≠ p2.actual_position

axiom vanya_predicted_last (c : Competition) :
  c.vanya.predicted_position = c.participants.length

axiom others_worse_than_predicted (c : Competition) :
  ∀ p : Participant, p ∈ c.participants → p ≠ c.vanya →
    p.actual_position > p.predicted_position

/-- Theorem: Vanya must have finished first -/
theorem vanya_finished_first (c : Competition) :
  c.vanya.actual_position = 1 :=
sorry

end NUMINAMATH_CALUDE_vanya_finished_first_l2093_209363


namespace NUMINAMATH_CALUDE_intercepts_correct_l2093_209395

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

/-- Theorem stating that the x-intercept and y-intercept are correct for the given line equation -/
theorem intercepts_correct : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept :=
sorry

end NUMINAMATH_CALUDE_intercepts_correct_l2093_209395


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l2093_209310

theorem sqrt_sum_equality : 
  let a : ℕ := 49
  let b : ℕ := 64
  let c : ℕ := 100
  Real.sqrt a + Real.sqrt b + Real.sqrt c = 
    Real.sqrt (219 + Real.sqrt 10080 + Real.sqrt 12600 + Real.sqrt 35280) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l2093_209310


namespace NUMINAMATH_CALUDE_work_completion_l2093_209324

/-- Represents the total amount of work in man-days -/
def total_work : ℕ := 10 * 80

/-- The number of days it takes for the second group to complete the work -/
def days_second_group : ℕ := 40

/-- Calculates the number of men needed to complete the work in a given number of days -/
def men_needed (days : ℕ) : ℕ := total_work / days

theorem work_completion :
  men_needed days_second_group = 20 :=
sorry

end NUMINAMATH_CALUDE_work_completion_l2093_209324


namespace NUMINAMATH_CALUDE_right_shift_two_units_l2093_209343

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Transformation that moves a function horizontally -/
def horizontalShift (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b - f.m * shift }

theorem right_shift_two_units (f : LinearFunction) :
  f.m = 2 ∧ f.b = 1 →
  (horizontalShift f 2).m = 2 ∧ (horizontalShift f 2).b = -3 := by
  sorry

end NUMINAMATH_CALUDE_right_shift_two_units_l2093_209343


namespace NUMINAMATH_CALUDE_jills_earnings_ratio_l2093_209346

/-- Jill's earnings over three months --/
def total_earnings : ℝ := 1200

/-- Jill's daily earnings in the first month --/
def first_month_daily : ℝ := 10

/-- Number of days in each month --/
def days_per_month : ℕ := 30

/-- Ratio of second month's daily earnings to first month's daily earnings --/
def earnings_ratio : ℝ := 2

theorem jills_earnings_ratio : 
  ∃ (second_month_daily : ℝ),
    first_month_daily * days_per_month +
    second_month_daily * days_per_month +
    second_month_daily * (days_per_month / 2) = total_earnings ∧
    second_month_daily / first_month_daily = earnings_ratio :=
by sorry

end NUMINAMATH_CALUDE_jills_earnings_ratio_l2093_209346


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l2093_209368

/-- Given an isosceles triangle and a similar triangle, calculates the perimeter of the larger triangle -/
theorem similar_triangle_perimeter (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a = b →
  c > a →
  c > b →
  d > c →
  (a + b + c) * (d / c) = 100 :=
by
  sorry

#check similar_triangle_perimeter

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l2093_209368


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2093_209364

theorem simplify_sqrt_expression :
  Real.sqrt 8 - Real.sqrt 50 + Real.sqrt 72 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2093_209364


namespace NUMINAMATH_CALUDE_number_relationship_l2093_209359

theorem number_relationship (n : ℚ) : n = 25 / 3 → (6 * n - 10) - 3 * n = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l2093_209359


namespace NUMINAMATH_CALUDE_triathlon_bicycle_speed_specific_triathlon_problem_l2093_209351

/-- Triathlon problem -/
theorem triathlon_bicycle_speed 
  (total_time : ℝ) 
  (swim_distance swim_speed : ℝ) 
  (run_distance run_speed : ℝ) 
  (bike_distance : ℝ) : ℝ :=
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

/-- The specific triathlon problem -/
theorem specific_triathlon_problem : 
  triathlon_bicycle_speed 1.75 (1/3) 1.5 2.5 8 12 = 1728/175 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_bicycle_speed_specific_triathlon_problem_l2093_209351


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2093_209354

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 - x^2/4 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = x/2 ∨ y = -x/2

-- Theorem statement
theorem hyperbola_properties :
  ∀ (x y : ℝ),
  hyperbola_equation x y →
  (∃ (a : ℝ), hyperbola_equation 0 a) ∧
  (∀ (x' y' : ℝ), x' ≠ 0 → asymptote_equation x' y' → 
    ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), δ > ε → 
      ∃ (x'' y'' : ℝ), hyperbola_equation x'' y'' ∧ 
      abs (x'' - x') < δ ∧ abs (y'' - y') < δ) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2093_209354


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l2093_209358

-- Define the points
variable (A B C D P Q E : Point)

-- Define the conditions
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry
def is_inside (P Q : Point) (A B C D : Point) : Prop := sorry
def is_cyclic_quadrilateral (P Q D A : Point) : Prop := sorry
def point_on_line (E P Q : Point) : Prop := sorry
def angle_eq (P A E Q D : Point) : Prop := sorry

-- State the theorem
theorem cyclic_quadrilateral_theorem 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_inside P Q A B C D)
  (h3 : is_cyclic_quadrilateral P Q D A)
  (h4 : is_cyclic_quadrilateral Q P B C)
  (h5 : point_on_line E P Q)
  (h6 : angle_eq P A E Q D)
  (h7 : angle_eq P B E Q C) :
  is_cyclic_quadrilateral A B C D :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l2093_209358


namespace NUMINAMATH_CALUDE_unique_sine_solution_l2093_209347

theorem unique_sine_solution : ∃! x : Real, 0 ≤ x ∧ x < Real.pi ∧ Real.sin x = -0.45 := by
  sorry

end NUMINAMATH_CALUDE_unique_sine_solution_l2093_209347


namespace NUMINAMATH_CALUDE_original_list_size_l2093_209387

theorem original_list_size (n : ℕ) (m : ℚ) : 
  (m + 3) * (n + 1) = m * n + 20 →
  (m + 1) * (n + 2) = m * n + 21 →
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_original_list_size_l2093_209387


namespace NUMINAMATH_CALUDE_james_muffins_count_l2093_209386

def arthur_muffins : ℕ := 115
def james_multiplier : ℚ := 12.5

theorem james_muffins_count :
  ⌈(arthur_muffins : ℚ) * james_multiplier⌉ = 1438 := by
  sorry

end NUMINAMATH_CALUDE_james_muffins_count_l2093_209386


namespace NUMINAMATH_CALUDE_volunteer_assignment_problem_l2093_209330

/-- The number of ways to assign n volunteers to k venues with at least one volunteer at each venue -/
def assignment_count (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

/-- The problem statement -/
theorem volunteer_assignment_problem :
  assignment_count 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_assignment_problem_l2093_209330


namespace NUMINAMATH_CALUDE_solve_trading_card_problem_l2093_209344

def trading_card_problem (initial_cards : ℕ) (brother_sets : ℕ) (friend_sets : ℕ) 
  (total_given : ℕ) (cards_per_set : ℕ) : ℕ :=
  let cards_to_brother := brother_sets * cards_per_set
  let cards_to_friend := friend_sets * cards_per_set
  let remaining_cards := total_given - (cards_to_brother + cards_to_friend)
  remaining_cards / cards_per_set

theorem solve_trading_card_problem :
  trading_card_problem 365 8 2 195 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_trading_card_problem_l2093_209344


namespace NUMINAMATH_CALUDE_tims_books_l2093_209398

theorem tims_books (mike_books : ℕ) (total_books : ℕ) (h1 : mike_books = 20) (h2 : total_books = 42) :
  total_books - mike_books = 22 := by
sorry

end NUMINAMATH_CALUDE_tims_books_l2093_209398


namespace NUMINAMATH_CALUDE_marble_return_condition_l2093_209317

/-- Represents the motion of a marble on a horizontal table with elastic collision -/
structure MarbleMotion where
  v₀ : ℝ  -- Initial speed
  h : ℝ   -- Initial height
  D : ℝ   -- Distance to vertical wall
  g : ℝ   -- Acceleration due to gravity

/-- The condition for the marble to return to the edge of the table -/
def returns_to_edge (m : MarbleMotion) : Prop :=
  m.v₀ = 2 * m.D * Real.sqrt (m.g / (2 * m.h))

/-- Theorem stating the condition for the marble to return to the edge of the table -/
theorem marble_return_condition (m : MarbleMotion) :
  returns_to_edge m ↔ m.v₀ = 2 * m.D * Real.sqrt (m.g / (2 * m.h)) :=
by sorry

end NUMINAMATH_CALUDE_marble_return_condition_l2093_209317


namespace NUMINAMATH_CALUDE_lollipops_per_boy_l2093_209340

theorem lollipops_per_boy (total_candies : ℕ) (total_children : ℕ) 
  (h1 : total_candies = 90)
  (h2 : total_children = 40)
  (h3 : ∃ (num_lollipops : ℕ), num_lollipops = total_candies / 3)
  (h4 : ∃ (num_candy_canes : ℕ), num_candy_canes = total_candies - total_candies / 3)
  (h5 : ∃ (num_girls : ℕ), num_girls = (total_candies - total_candies / 3) / 2)
  (h6 : ∃ (num_boys : ℕ), num_boys = total_children - (total_candies - total_candies / 3) / 2) :
  (total_candies / 3) / (total_children - (total_candies - total_candies / 3) / 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_lollipops_per_boy_l2093_209340


namespace NUMINAMATH_CALUDE_vector_range_l2093_209337

/-- Given unit vectors i and j along x and y axes respectively, and a vector a satisfying 
    |a - i| + |a - 2j| = √5, prove that the range of |a + 2i| is [6√5/5, 3]. -/
theorem vector_range (i j a : ℝ × ℝ) : 
  i = (1, 0) → 
  j = (0, 1) → 
  ‖a - i‖ + ‖a - 2 • j‖ = Real.sqrt 5 → 
  6 * Real.sqrt 5 / 5 ≤ ‖a + 2 • i‖ ∧ ‖a + 2 • i‖ ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_range_l2093_209337


namespace NUMINAMATH_CALUDE_mosquito_lethal_feedings_l2093_209390

/-- The number of mosquito feedings required to reach lethal blood loss -/
def lethal_feedings (drops_per_feeding : ℕ) (drops_per_liter : ℕ) (lethal_liters : ℕ) : ℕ :=
  (lethal_liters * drops_per_liter) / drops_per_feeding

theorem mosquito_lethal_feedings :
  lethal_feedings 20 5000 3 = 750 := by
  sorry

#eval lethal_feedings 20 5000 3

end NUMINAMATH_CALUDE_mosquito_lethal_feedings_l2093_209390


namespace NUMINAMATH_CALUDE_relationship_abc_l2093_209303

theorem relationship_abc (a b c : ℝ) :
  (∃ u v : ℝ, u - v = a ∧ u^2 - v^2 = b ∧ u^3 - v^3 = c) →
  3 * b^2 + a^4 = 4 * a * c := by
sorry

end NUMINAMATH_CALUDE_relationship_abc_l2093_209303


namespace NUMINAMATH_CALUDE_starters_with_twin_restriction_l2093_209370

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of players in the team -/
def total_players : ℕ := 16

/-- The number of starters to be chosen -/
def starters : ℕ := 5

/-- The number of players excluding both sets of twins -/
def players_excluding_twins : ℕ := total_players - 4

/-- The number of ways to choose starters with the twin restriction -/
def ways_to_choose_starters : ℕ :=
  binomial total_players starters -
  2 * binomial (total_players - 2) (starters - 2) +
  binomial (total_players - 4) (starters - 4)

theorem starters_with_twin_restriction :
  ways_to_choose_starters = 3652 :=
sorry

end NUMINAMATH_CALUDE_starters_with_twin_restriction_l2093_209370


namespace NUMINAMATH_CALUDE_election_vote_count_l2093_209373

theorem election_vote_count : 
  -- Define the total number of votes
  ∀ V : ℕ,
  -- First round vote percentages
  let a1 := (27 : ℚ) / 100 * V
  let b1 := (24 : ℚ) / 100 * V
  let c1 := (20 : ℚ) / 100 * V
  let d1 := (18 : ℚ) / 100 * V
  let e1 := V - (a1 + b1 + c1 + d1)
  -- Second round vote percentages
  let a2 := (30 : ℚ) / 100 * V
  let b2 := (27 : ℚ) / 100 * V
  let c2 := (22 : ℚ) / 100 * V
  let d2 := V - (a2 + b2 + c2)
  -- Final round
  let additional_votes := (10 : ℚ) / 100 * V  -- 5% each from C and D supporters
  let a_final := a2 + (5 : ℚ) / 100 * V
  let b_final := b2 + d2 + (5 : ℚ) / 100 * V
  -- B wins by 1350 votes
  b_final - a_final = 1350 →
  V = 7500 := by
sorry

end NUMINAMATH_CALUDE_election_vote_count_l2093_209373


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_15_mod_24_l2093_209318

theorem largest_five_digit_congruent_to_15_mod_24 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n ≡ 15 [MOD 24] → n ≤ 99999 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_15_mod_24_l2093_209318


namespace NUMINAMATH_CALUDE_ap_sum_100_l2093_209365

/-- Given an arithmetic progression where:
    - The sum of the first 15 terms is 45
    - The sum of the first 85 terms is 255
    This theorem proves that the sum of the first 100 terms is 300. -/
theorem ap_sum_100 (a d : ℝ) 
  (sum_15 : (15 : ℝ) / 2 * (2 * a + (15 - 1) * d) = 45)
  (sum_85 : (85 : ℝ) / 2 * (2 * a + (85 - 1) * d) = 255) :
  (100 : ℝ) / 2 * (2 * a + (100 - 1) * d) = 300 :=
by sorry

end NUMINAMATH_CALUDE_ap_sum_100_l2093_209365


namespace NUMINAMATH_CALUDE_square_sum_theorem_l2093_209384

theorem square_sum_theorem (x y : ℝ) 
  (h1 : x * (x + y) = 40)
  (h2 : y * (x + y) = 90) :
  (x + y)^2 = 130 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l2093_209384


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2093_209316

theorem unknown_number_proof (x : ℝ) : x + 5 * 12 / (180 / 3) = 66 → x = 65 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2093_209316


namespace NUMINAMATH_CALUDE_positive_number_square_root_l2093_209380

theorem positive_number_square_root (x : ℝ) : 
  x > 0 → (Real.sqrt ((4 * x) / 3) = x) → x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_square_root_l2093_209380


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l2093_209355

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of distinct letters in 'MATHEMATICS' -/
def distinct_letters : ℕ := 8

/-- The word we're considering -/
def word : String := "MATHEMATICS"

/-- Theorem: The probability of randomly selecting a letter from the alphabet
    that appears in 'MATHEMATICS' is 4/13 -/
theorem mathematics_letter_probability :
  (distinct_letters : ℚ) / (alphabet_size : ℚ) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l2093_209355


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2093_209328

/-- Given vectors a and b, if (-2a + b) is parallel to (a + kb), then k = -1/2 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (-3, 1)) 
  (h2 : b = (1, -2)) 
  (h_parallel : ∃ (t : ℝ), t • (-2 • a + b) = (a + k • b)) :
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2093_209328
