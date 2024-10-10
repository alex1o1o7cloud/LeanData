import Mathlib

namespace angle_AEC_l336_33675

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)
  (sum_360 : A + B + C + D = 360)

-- Define the exterior angle bisector point
def exterior_angle_bisector_point (q : Quadrilateral) : ℝ :=
  sorry

-- Theorem statement
theorem angle_AEC (q : Quadrilateral) :
  let E := exterior_angle_bisector_point q
  (360 - (q.B + q.D)) / 2 = E := by
  sorry

end angle_AEC_l336_33675


namespace handshake_count_l336_33630

/-- Represents the number of students in the class -/
def num_students : ℕ := 40

/-- Represents the length of the counting sequence -/
def sequence_length : ℕ := 4

/-- Calculates the number of initial pairs facing each other -/
def initial_pairs : ℕ := num_students / sequence_length

/-- Calculates the sum of handshakes in subsequent rounds -/
def subsequent_handshakes : ℕ := (initial_pairs * (initial_pairs + 1)) / 2

/-- Calculates the total number of handshakes -/
def total_handshakes : ℕ := initial_pairs + 3 * subsequent_handshakes

/-- Theorem stating that the total number of handshakes is 175 -/
theorem handshake_count : total_handshakes = 175 := by sorry

end handshake_count_l336_33630


namespace sufficient_condition_for_inequality_l336_33616

theorem sufficient_condition_for_inequality (a : ℝ) (h : a ≥ 5) :
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 := by
  sorry

end sufficient_condition_for_inequality_l336_33616


namespace x_value_l336_33614

theorem x_value : ∃ x : ℝ, (x = 88 * (1 + 0.20)) ∧ (x = 105.6) := by
  sorry

end x_value_l336_33614


namespace min_value_problem_l336_33687

theorem min_value_problem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h_sum : x + 2*y = 1) :
  ∃ (m : ℝ), m = 3/4 ∧ ∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*y'^2 ≥ m :=
sorry

end min_value_problem_l336_33687


namespace min_value_zero_at_one_sixth_l336_33690

/-- The quadratic expression as a function of x, y, and c -/
def f (x y c : ℝ) : ℝ :=
  2 * x^2 - 4 * c * x * y + (2 * c^2 + 1) * y^2 - 2 * x - 6 * y + 9

/-- Theorem stating that 1/6 is the value of c that makes the minimum of f zero -/
theorem min_value_zero_at_one_sixth :
  ∃ (x y : ℝ), f x y (1/6) = 0 ∧ ∀ (x' y' : ℝ), f x' y' (1/6) ≥ 0 :=
sorry

end min_value_zero_at_one_sixth_l336_33690


namespace cubic_fraction_value_l336_33692

theorem cubic_fraction_value : 
  let a : ℝ := 8
  let b : ℝ := 8 - 1
  (a^3 + b^3) / (a^2 - a*b + b^2) = 15 := by sorry

end cubic_fraction_value_l336_33692


namespace line_through_first_and_third_quadrants_has_positive_slope_l336_33636

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  passes_through_first_quadrant : Bool
  passes_through_third_quadrant : Bool

/-- Definition of a line passing through first and third quadrants -/
def passes_through_first_and_third (l : Line) : Prop :=
  l.passes_through_first_quadrant ∧ l.passes_through_third_quadrant

/-- Theorem: If a line y = kx (k ≠ 0) passes through the first and third quadrants, then k > 0 -/
theorem line_through_first_and_third_quadrants_has_positive_slope (l : Line) 
    (h1 : l.slope ≠ 0) 
    (h2 : passes_through_first_and_third l) : 
    l.slope > 0 := by
  sorry

end line_through_first_and_third_quadrants_has_positive_slope_l336_33636


namespace complement_of_union_equals_five_l336_33651

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_five :
  (U \ (M ∪ N)) = {5} := by sorry

end complement_of_union_equals_five_l336_33651


namespace sin_power_five_expansion_l336_33602

theorem sin_power_five_expansion (b₁ b₂ b₃ b₄ b₅ : ℝ) :
  (∀ θ : ℝ, Real.sin θ ^ 5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + 
    b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 = 63 / 512 := by
  sorry

end sin_power_five_expansion_l336_33602


namespace simplify_sqrt_sum_l336_33664

theorem simplify_sqrt_sum (h : π / 2 < 2 ∧ 2 < 3 * π / 4) :
  Real.sqrt (1 + Real.sin 4) + Real.sqrt (1 - Real.sin 4) = 2 * Real.sin 2 := by
  sorry

end simplify_sqrt_sum_l336_33664


namespace cage_cost_calculation_l336_33635

/-- The cost of Keith's purchases -/
def total_cost : ℝ := 24.81

/-- The cost of the rabbit toy -/
def rabbit_toy_cost : ℝ := 6.51

/-- The cost of the pet food -/
def pet_food_cost : ℝ := 5.79

/-- The amount of money Keith found -/
def found_money : ℝ := 1.00

/-- The cost of the cage -/
def cage_cost : ℝ := total_cost - (rabbit_toy_cost + pet_food_cost) + found_money

theorem cage_cost_calculation : cage_cost = 13.51 := by
  sorry

end cage_cost_calculation_l336_33635


namespace cubic_equation_real_root_l336_33620

theorem cubic_equation_real_root (k : ℝ) : ∃ x : ℝ, x^3 + 3*k*x^2 + 3*k^2*x + k^3 = 0 := by
  sorry

end cubic_equation_real_root_l336_33620


namespace initial_speed_is_60_l336_33666

/-- Represents the initial speed of a traveler given specific journey conditions -/
def initial_speed (D T : ℝ) : ℝ :=
  let remaining_time := T - T / 3
  let remaining_distance := D / 3
  60

/-- Theorem stating the initial speed under given conditions -/
theorem initial_speed_is_60 (D T : ℝ) (h1 : D > 0) (h2 : T > 0) :
  initial_speed D T = 60 := by
  sorry

#check initial_speed_is_60

end initial_speed_is_60_l336_33666


namespace quadratic_equation_integer_roots_l336_33600

theorem quadratic_equation_integer_roots :
  let S : Set ℝ := {a : ℝ | a > 0 ∧ ∃ x y : ℤ, x ≠ y ∧ a^2 * x^2 + a * x + 1 - 13 * a^2 = 0 ∧ a^2 * y^2 + a * y + 1 - 13 * a^2 = 0}
  S = {1, 1/3, 1/4} := by
  sorry

end quadratic_equation_integer_roots_l336_33600


namespace valid_division_exists_l336_33660

/-- Represents a grid cell that can contain a symbol -/
inductive Cell
  | Empty
  | Star
  | Cross

/-- Represents a 7x7 grid -/
def Grid := Fin 7 → Fin 7 → Cell

/-- Represents a matchstick placement -/
structure Matchstick where
  row : Fin 8
  col : Fin 8
  horizontal : Bool

/-- Counts the number of matchsticks in a list -/
def count_matchsticks (placements : List Matchstick) : Nat :=
  placements.length

/-- Checks if two parts of the grid are of equal size and shape -/
def equal_parts (g : Grid) (placements : List Matchstick) : Prop :=
  sorry

/-- Checks if the symbols (stars and crosses) are placed correctly -/
def correct_symbol_placement (g : Grid) : Prop :=
  sorry

/-- The main theorem stating that a valid division exists -/
theorem valid_division_exists : ∃ (g : Grid) (placements : List Matchstick),
  count_matchsticks placements = 26 ∧
  equal_parts g placements ∧
  correct_symbol_placement g :=
  sorry

end valid_division_exists_l336_33660


namespace trig_identity_l336_33601

theorem trig_identity (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.sin (α - π / 6) ^ 2 - Real.cos (5 * π / 6 + α) = (2 + Real.sqrt 3) / 3 := by
  sorry

end trig_identity_l336_33601


namespace point_D_value_l336_33641

/-- The number corresponding to point D on a number line, given that:
    - A corresponds to 5
    - B corresponds to 8
    - C corresponds to -10
    - The sum of the four numbers remains unchanged when the direction of the number line is reversed
-/
def point_D : ℝ := -3

/-- The sum of the numbers corresponding to points A, B, C, and D -/
def sum_forward (d : ℝ) : ℝ := 5 + 8 + (-10) + d

/-- The sum of the numbers corresponding to points A, B, C, and D when the direction is reversed -/
def sum_reversed (d : ℝ) : ℝ := (-5) + (-8) + 10 + (-d)

/-- Theorem stating that point D corresponds to -3 -/
theorem point_D_value : 
  sum_forward point_D = sum_reversed point_D :=
by sorry

end point_D_value_l336_33641


namespace unique_n_satisfying_conditions_l336_33661

def is_divisible_by_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ n ∧ q ∣ n

def unit_digit (n : ℕ) : ℕ := n % 10

def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

def uniquely_determined_by_divisors (n : ℕ) : Prop :=
  ∀ m : ℕ, m < 60 → unit_digit m = unit_digit n → num_divisors m = num_divisors n → m = n

theorem unique_n_satisfying_conditions :
  ∃! n : ℕ, n < 60 ∧
    is_divisible_by_two_primes n ∧
    uniquely_determined_by_divisors n ∧
    n = 10 := by
  sorry

end unique_n_satisfying_conditions_l336_33661


namespace length_PI_is_five_l336_33606

/-- A right triangle with given side lengths and its incenter -/
structure RightTriangleWithIncenter where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Length of side QR (hypotenuse) -/
  qr : ℝ
  /-- The triangle is right-angled -/
  is_right : pq ^ 2 + pr ^ 2 = qr ^ 2
  /-- The given side lengths form a valid triangle -/
  triangle_inequality : pq + pr > qr ∧ pr + qr > pq ∧ qr + pq > pr
  /-- The incenter of the triangle -/
  incenter : ℝ × ℝ

/-- The length of segment PI in a right triangle with incenter -/
def length_PI (t : RightTriangleWithIncenter) : ℝ :=
  sorry

/-- Theorem: The length of segment PI is 5 for the given triangle -/
theorem length_PI_is_five (t : RightTriangleWithIncenter) 
  (h1 : t.pq = 15) (h2 : t.pr = 20) (h3 : t.qr = 25) : length_PI t = 5 := by
  sorry

end length_PI_is_five_l336_33606


namespace three_integers_product_2008th_power_l336_33639

theorem three_integers_product_2008th_power :
  ∃ (x y z : ℕ), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧  -- distinct
    x > 0 ∧ y > 0 ∧ z > 0 ∧  -- positive
    y = (x + z) / 2 ∧        -- one is average of other two
    ∃ (k : ℕ), x * y * z = k^2008 := by
  sorry

end three_integers_product_2008th_power_l336_33639


namespace sum_of_areas_equals_AD_squared_l336_33607

/-- A right-angled quadrilateral with the golden ratio property -/
structure GoldenQuadrilateral where
  AB : ℝ
  AD : ℝ
  right_angled : AB > 0 ∧ AD > 0
  shorter_side : AB < AD
  golden_ratio : AB / AD = (AD - AB) / AB

/-- The sum of areas of an infinite series of similar quadrilaterals -/
def sum_of_areas (q : GoldenQuadrilateral) : ℝ := q.AD ^ 2

/-- The main theorem: the sum of areas equals AD^2 -/
theorem sum_of_areas_equals_AD_squared (q : GoldenQuadrilateral) :
  sum_of_areas q = q.AD ^ 2 := by sorry

end sum_of_areas_equals_AD_squared_l336_33607


namespace common_tangents_count_l336_33621

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 10*y + 13 = 0

-- Define a function to count common tangents
def count_common_tangents (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents C1 C2 = 1 := by sorry

end common_tangents_count_l336_33621


namespace student_speed_ratio_l336_33658

theorem student_speed_ratio :
  ∀ (distance_A distance_B time_A time_B : ℚ),
    distance_A = (6 / 5) * distance_B →
    time_B = (10 / 11) * time_A →
    (distance_A / time_A) / (distance_B / time_B) = 12 / 11 :=
by
  sorry

end student_speed_ratio_l336_33658


namespace janes_pudding_purchase_l336_33623

theorem janes_pudding_purchase (ice_cream_count : ℕ) (ice_cream_cost pudding_cost : ℚ) 
  (ice_cream_pudding_diff : ℚ) :
  ice_cream_count = 15 →
  ice_cream_cost = 5 →
  pudding_cost = 2 →
  ice_cream_count * ice_cream_cost = ice_cream_pudding_diff + pudding_count * pudding_cost →
  pudding_count = 5 :=
by
  sorry

#check janes_pudding_purchase

end janes_pudding_purchase_l336_33623


namespace vacuum_time_calculation_l336_33609

theorem vacuum_time_calculation (total_free_time dusting_time mopping_time cat_brushing_time num_cats remaining_free_time : ℕ) 
  (h1 : total_free_time = 180)
  (h2 : dusting_time = 60)
  (h3 : mopping_time = 30)
  (h4 : cat_brushing_time = 5)
  (h5 : num_cats = 3)
  (h6 : remaining_free_time = 30) :
  total_free_time - remaining_free_time - (dusting_time + mopping_time + cat_brushing_time * num_cats) = 45 := by
  sorry

end vacuum_time_calculation_l336_33609


namespace factorial_sum_remainder_l336_33611

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_remainder : (List.sum (List.map factorial [1, 2, 3, 4, 5, 6])) % 24 = 9 := by
  sorry

end factorial_sum_remainder_l336_33611


namespace no_distinct_integers_divisibility_l336_33678

theorem no_distinct_integers_divisibility : ¬∃ (a : Fin 2001 → ℕ+), 
  (∀ (i j : Fin 2001), i ≠ j → (a i).val * (a j).val ∣ 
    ((a i).val ^ 2000 - (a i).val ^ 1000 + 1) * 
    ((a j).val ^ 2000 - (a j).val ^ 1000 + 1)) ∧ 
  (∀ (i j : Fin 2001), i ≠ j → a i ≠ a j) :=
by sorry

end no_distinct_integers_divisibility_l336_33678


namespace inscribed_cube_volume_l336_33622

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_diagonal : ℝ := sphere_diameter
  let inner_cube_edge : ℝ := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume : ℝ := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l336_33622


namespace closest_ratio_is_one_to_one_l336_33629

def admission_fee (adults children : ℕ) : ℕ := 30 * adults + 15 * children

def is_valid_combination (adults children : ℕ) : Prop :=
  adults ≥ 2 ∧ children ≥ 2 ∧ admission_fee adults children = 2250

def ratio_difference (adults children : ℕ) : ℚ :=
  |((adults : ℚ) / (children : ℚ)) - 1|

theorem closest_ratio_is_one_to_one :
  ∃ (a c : ℕ), is_valid_combination a c ∧
    ∀ (x y : ℕ), is_valid_combination x y →
      ratio_difference a c ≤ ratio_difference x y :=
sorry

end closest_ratio_is_one_to_one_l336_33629


namespace equal_roots_values_l336_33608

theorem equal_roots_values (x m : ℝ) : 
  (x^2 * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = x^2 / m → 
  (∀ x, 2*x^2 - 4*x - m^2 - 2*m = 0) → 
  (m = -1 + Real.sqrt 3 ∨ m = -1 - Real.sqrt 3) :=
by sorry

end equal_roots_values_l336_33608


namespace line_passes_through_point_l336_33603

/-- The line equation kx - y + 1 - 3k = 0 passes through the point (3, 1) for all k. -/
theorem line_passes_through_point :
  ∀ (k : ℝ), k * 3 - 1 + 1 - 3 * k = 0 :=
by
  sorry

end line_passes_through_point_l336_33603


namespace x_eq_2_sufficient_not_necessary_for_x_geq_1_l336_33632

theorem x_eq_2_sufficient_not_necessary_for_x_geq_1 :
  (∀ x : ℝ, x = 2 → x ≥ 1) ∧ ¬(∀ x : ℝ, x ≥ 1 → x = 2) :=
by sorry

end x_eq_2_sufficient_not_necessary_for_x_geq_1_l336_33632


namespace lions_count_l336_33695

theorem lions_count (lions tigers cougars : ℕ) : 
  tigers = 14 →
  cougars = (lions + tigers) / 2 →
  lions + tigers + cougars = 39 →
  lions = 12 := by
sorry

end lions_count_l336_33695


namespace geometric_mean_of_1_and_9_l336_33625

def geometric_mean (a b : ℝ) : Set ℝ :=
  {x | x ^ 2 = a * b}

theorem geometric_mean_of_1_and_9 :
  geometric_mean 1 9 = {3, -3} := by sorry

end geometric_mean_of_1_and_9_l336_33625


namespace tim_pencil_count_l336_33654

/-- Given that Tyrah has six times as many pencils as Sarah, Tim has eight times as many pencils as Sarah, and Tyrah has 12 pencils, prove that Tim has 16 pencils. -/
theorem tim_pencil_count (sarah_pencils : ℕ) 
  (h1 : 6 * sarah_pencils = 12)  -- Tyrah has six times as many pencils as Sarah and has 12 pencils
  (h2 : 8 * sarah_pencils = tim_pencils) : -- Tim has eight times as many pencils as Sarah
  tim_pencils = 16 := by
  sorry

end tim_pencil_count_l336_33654


namespace distributive_property_l336_33697

theorem distributive_property (a b c : ℝ) : -2 * (a + b - 3 * c) = -2 * a - 2 * b + 6 * c := by
  sorry

end distributive_property_l336_33697


namespace sum_of_products_zero_l336_33645

theorem sum_of_products_zero 
  (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 108)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 117) :
  x*y + y*z + x*z = 0 := by
sorry

end sum_of_products_zero_l336_33645


namespace ink_blot_is_circle_l336_33652

/-- A closed, bounded set in a plane -/
def InkBlot : Type := Set (ℝ × ℝ)

/-- The minimum distance from a point to the boundary of the ink blot -/
def min_distance (S : InkBlot) (p : ℝ × ℝ) : ℝ := sorry

/-- The maximum distance from a point to the boundary of the ink blot -/
def max_distance (S : InkBlot) (p : ℝ × ℝ) : ℝ := sorry

/-- The largest of all minimum distances -/
def largest_min_distance (S : InkBlot) : ℝ := sorry

/-- The smallest of all maximum distances -/
def smallest_max_distance (S : InkBlot) : ℝ := sorry

/-- A circle in the plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : InkBlot := sorry

theorem ink_blot_is_circle (S : InkBlot) :
  largest_min_distance S = smallest_max_distance S →
  ∃ (center : ℝ × ℝ) (radius : ℝ), S = Circle center radius :=
sorry

end ink_blot_is_circle_l336_33652


namespace reflection_line_sum_l336_33643

/-- Given a reflection of point (-2, 3) across line y = mx + b to point (4, -5), prove m + b = -1 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 4 ∧ y = -5 ∧ 
    (x - (-2))^2 + (y - 3)^2 = (x - (-2))^2 + (m * (x - (-2)) + b - 3)^2 ∧
    y = m * x + b) →
  m + b = -1 := by
sorry

end reflection_line_sum_l336_33643


namespace max_product_sum_246_l336_33619

theorem max_product_sum_246 : 
  ∀ x y : ℤ, x + y = 246 → x * y ≤ 15129 :=
by
  sorry

end max_product_sum_246_l336_33619


namespace problem_solution_l336_33684

theorem problem_solution :
  ∀ (x a b : ℝ),
  (∃ y : ℝ, y^2 = x ∧ y = a + 3) ∧
  (∃ z : ℝ, z^2 = x ∧ z = 2*a - 15) ∧
  (3^2 = 2*b - 1) →
  (a = 4 ∧ b = 5 ∧ (a + b - 1)^(1/3) = 2) :=
by sorry

end problem_solution_l336_33684


namespace f_at_two_equals_one_fourth_l336_33648

/-- Given a function f(x) = 2^x + 2^(-x) - 4, prove that f(2) = 1/4 -/
theorem f_at_two_equals_one_fourth :
  let f : ℝ → ℝ := λ x ↦ 2^x + 2^(-x) - 4
  f 2 = 1/4 := by
  sorry

end f_at_two_equals_one_fourth_l336_33648


namespace geometric_sequence_product_l336_33691

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 2 →
  a 6 = 8 →
  a 3 * a 4 * a 5 = 64 := by
sorry

end geometric_sequence_product_l336_33691


namespace hiking_distance_proof_l336_33699

theorem hiking_distance_proof (total_distance car_to_stream stream_to_meadow : ℝ) 
  (h1 : total_distance = 0.7)
  (h2 : car_to_stream = 0.2)
  (h3 : stream_to_meadow = 0.4) :
  total_distance - (car_to_stream + stream_to_meadow) = 0.1 := by
  sorry

end hiking_distance_proof_l336_33699


namespace arithmetic_sequence_ratio_l336_33612

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = n * (a 0 + a (n-1)) / 2

/-- Theorem: If S_6/S_3 = 4 for an arithmetic sequence, then S_5/S_6 = 25/36 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 6 / seq.S 3 = 4) : 
  seq.S 5 / seq.S 6 = 25/36 := by
  sorry

end arithmetic_sequence_ratio_l336_33612


namespace negation_of_existence_exp_l336_33627

theorem negation_of_existence_exp (p : Prop) : 
  (p ↔ ∃ x : ℝ, Real.exp x < 0) → 
  (¬p ↔ ∀ x : ℝ, Real.exp x ≥ 0) := by
  sorry

end negation_of_existence_exp_l336_33627


namespace trig_values_special_angles_l336_33638

theorem trig_values_special_angles :
  (Real.sin (π/6) = 1/2) ∧
  (Real.cos (π/6) = Real.sqrt 3 / 2) ∧
  (Real.tan (π/6) = Real.sqrt 3 / 3) ∧
  (Real.sin (π/4) = Real.sqrt 2 / 2) ∧
  (Real.cos (π/4) = Real.sqrt 2 / 2) ∧
  (Real.tan (π/4) = 1) ∧
  (Real.sin (π/3) = Real.sqrt 3 / 2) ∧
  (Real.cos (π/3) = 1/2) ∧
  (Real.tan (π/3) = Real.sqrt 3) ∧
  (Real.sin (π/2) = 1) ∧
  (Real.cos (π/2) = 0) := by
  sorry

-- Note: tan(π/2) is undefined, so it's not included in the theorem statement

end trig_values_special_angles_l336_33638


namespace savings_percentage_second_year_l336_33659

/-- Proves that under the given conditions, the savings percentage in the second year is 15% -/
theorem savings_percentage_second_year 
  (salary_first_year : ℝ) 
  (savings_rate_first_year : ℝ) 
  (salary_increase_rate : ℝ) 
  (savings_increase_rate : ℝ) : 
  savings_rate_first_year = 0.1 →
  salary_increase_rate = 0.1 →
  savings_increase_rate = 1.65 →
  (savings_increase_rate * savings_rate_first_year * salary_first_year) / 
  ((1 + salary_increase_rate) * salary_first_year) = 0.15 := by
sorry


end savings_percentage_second_year_l336_33659


namespace min_sum_of_squares_l336_33670

theorem min_sum_of_squares (x y : ℝ) (h : 4 * x^2 + 5 * x * y + 4 * y^2 = 5) :
  ∃ (S_min : ℝ), S_min = 10/13 ∧ x^2 + y^2 ≥ S_min :=
by sorry

end min_sum_of_squares_l336_33670


namespace circle_area_is_one_l336_33671

theorem circle_area_is_one (r : ℝ) (h : r > 0) :
  (8 / (2 * Real.pi * r) + 2 * r = 6 * r) → π * r^2 = 1 := by
  sorry

end circle_area_is_one_l336_33671


namespace equation_is_quadratic_l336_33642

/-- Proves that the equation 3(x+1)² = 2(x+1) is equivalent to a quadratic equation in the standard form ax² + bx + c = 0, where a ≠ 0 -/
theorem equation_is_quadratic (x : ℝ) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (3 * (x + 1)^2 = 2 * (x + 1)) ↔ (a * x^2 + b * x + c = 0) :=
sorry

end equation_is_quadratic_l336_33642


namespace rotateSemicircleDiameter_is_eight_l336_33667

/-- The diameter of a solid figure obtained by rotating a semicircle around its diameter -/
def rotateSemicircleDiameter (radius : ℝ) : ℝ :=
  2 * radius

/-- Theorem: The diameter of a solid figure obtained by rotating a semicircle 
    with a radius of 4 centimeters once around its diameter is 8 centimeters -/
theorem rotateSemicircleDiameter_is_eight :
  rotateSemicircleDiameter 4 = 8 := by
  sorry

end rotateSemicircleDiameter_is_eight_l336_33667


namespace quadratic_roots_sum_squares_bounds_l336_33681

theorem quadratic_roots_sum_squares_bounds (k : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 - (k - 2) * x₁ + (k^2 + 3 * k + 5) = 0 →
  x₂^2 - (k - 2) * x₂ + (k^2 + 3 * k + 5) = 0 →
  x₁ ≠ x₂ →
  ∃ (y : ℝ), y = x₁^2 + x₂^2 ∧ y ≤ 18 ∧ y ≥ 50/9 ∧
  (∃ (k₁ : ℝ), x₁^2 - (k₁ - 2) * x₁ + (k₁^2 + 3 * k₁ + 5) = 0 ∧
               x₂^2 - (k₁ - 2) * x₂ + (k₁^2 + 3 * k₁ + 5) = 0 ∧
               x₁^2 + x₂^2 = 18) ∧
  (∃ (k₂ : ℝ), x₁^2 - (k₂ - 2) * x₁ + (k₂^2 + 3 * k₂ + 5) = 0 ∧
               x₂^2 - (k₂ - 2) * x₂ + (k₂^2 + 3 * k₂ + 5) = 0 ∧
               x₁^2 + x₂^2 = 50/9) :=
by sorry

end quadratic_roots_sum_squares_bounds_l336_33681


namespace order_relation_l336_33640

theorem order_relation (a b c : ℝ) : 
  a = Real.exp 0.2 → b = 0.2 ^ Real.exp 1 → c = Real.log 2 → b < c ∧ c < a := by
  sorry

end order_relation_l336_33640


namespace actual_tax_expectation_l336_33672

/-- Represents the fraction of the population that are liars -/
def fraction_liars : ℝ := 0.1

/-- Represents the fraction of the population that are economists -/
def fraction_economists : ℝ := 1 - fraction_liars

/-- Represents the fraction of affirmative answers for raising taxes -/
def affirmative_taxes : ℝ := 0.4

/-- Represents the fraction of affirmative answers for increasing money supply -/
def affirmative_money : ℝ := 0.3

/-- Represents the fraction of affirmative answers for issuing bonds -/
def affirmative_bonds : ℝ := 0.5

/-- Represents the fraction of affirmative answers for spending gold reserves -/
def affirmative_gold : ℝ := 0

/-- The theorem stating that 30% of the population actually expects raising taxes -/
theorem actual_tax_expectation : 
  affirmative_taxes - fraction_liars = 0.3 := by sorry

end actual_tax_expectation_l336_33672


namespace kyle_total_laps_l336_33662

-- Define the number of laps jogged in P.E. class
def pe_laps : ℝ := 1.12

-- Define the number of laps jogged during track practice
def track_laps : ℝ := 2.12

-- Define the total number of laps
def total_laps : ℝ := pe_laps + track_laps

-- Theorem statement
theorem kyle_total_laps : total_laps = 3.24 := by
  sorry

end kyle_total_laps_l336_33662


namespace intersection_distance_l336_33617

/-- The distance between the intersection points of y = 5 and y = 5x^2 + 2x - 2 is 2.4 -/
theorem intersection_distance : 
  let f (x : ℝ) := 5*x^2 + 2*x - 2
  let g (x : ℝ) := 5
  let roots := {x : ℝ | f x = g x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 2.4 :=
by sorry

end intersection_distance_l336_33617


namespace spaghetti_pizza_ratio_l336_33693

/-- The total number of students surveyed -/
def total_students : ℕ := 800

/-- The number of students preferring lasagna -/
def lasagna_preference : ℕ := 150

/-- The number of students preferring manicotti -/
def manicotti_preference : ℕ := 120

/-- The number of students preferring ravioli -/
def ravioli_preference : ℕ := 180

/-- The number of students preferring spaghetti -/
def spaghetti_preference : ℕ := 200

/-- The number of students preferring pizza -/
def pizza_preference : ℕ := 150

/-- Theorem stating that the ratio of students preferring spaghetti to students preferring pizza is 4/3 -/
theorem spaghetti_pizza_ratio : 
  (spaghetti_preference : ℚ) / (pizza_preference : ℚ) = 4 / 3 := by
  sorry

end spaghetti_pizza_ratio_l336_33693


namespace train_passengers_l336_33683

theorem train_passengers (initial_passengers : ℕ) (num_stops : ℕ) : 
  initial_passengers = 64 → num_stops = 4 → 
  (initial_passengers : ℚ) * (2/3)^num_stops = 1024/81 := by
  sorry

end train_passengers_l336_33683


namespace a_51_equals_101_l336_33631

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem a_51_equals_101 (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 51 = 101 := by
  sorry

end a_51_equals_101_l336_33631


namespace child_tickets_sold_l336_33676

theorem child_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ)
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_receipts ∧
    child_tickets = 90 := by
  sorry

end child_tickets_sold_l336_33676


namespace set_inclusion_theorem_l336_33649

def A (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def B : Set ℝ := {x | -1/2 < x ∧ x ≤ 2}

theorem set_inclusion_theorem :
  (∀ x ∈ B, x ∈ A 1) ∧
  (∀ a : ℝ, (∀ x ∈ A a, x ∈ B) ↔ a < -8 ∨ a ≥ 2) := by
  sorry

end set_inclusion_theorem_l336_33649


namespace second_month_sale_l336_33628

def average_sale : ℕ := 6600
def num_months : ℕ := 6
def sale_month1 : ℕ := 6435
def sale_month3 : ℕ := 7230
def sale_month4 : ℕ := 6562
def sale_month5 : ℕ := 6855
def sale_month6 : ℕ := 5591

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    sale_month2 = average_sale * num_months - (sale_month1 + sale_month3 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month2 = 6927 := by
  sorry

end second_month_sale_l336_33628


namespace mod_equivalence_unique_solution_l336_33626

theorem mod_equivalence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -3402 [ZMOD 10] ∧ n = 8 := by
  sorry

end mod_equivalence_unique_solution_l336_33626


namespace right_triangle_sin_d_l336_33680

theorem right_triangle_sin_d (D E F : Real) (h1 : 4 * Real.sin D = 5 * Real.cos D) :
  Real.sin D = 5 * Real.sqrt 41 / 41 := by
  sorry

end right_triangle_sin_d_l336_33680


namespace sphere_division_theorem_l336_33637

/-- The maximum number of regions into which a sphere can be divided by n great circles -/
def sphere_regions (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem: The maximum number of regions into which a sphere can be divided by n great circles is n^2 - n + 2 -/
theorem sphere_division_theorem (n : ℕ) : 
  sphere_regions n = n^2 - n + 2 := by
  sorry

end sphere_division_theorem_l336_33637


namespace train_distance_difference_l336_33610

/-- Proves that the difference in distance traveled by two trains is 60 km -/
theorem train_distance_difference :
  ∀ (speed1 speed2 total_distance : ℝ),
  speed1 = 20 →
  speed2 = 25 →
  total_distance = 540 →
  ∃ (time : ℝ),
    time > 0 ∧
    speed1 * time + speed2 * time = total_distance ∧
    speed2 * time - speed1 * time = 60 :=
by sorry

end train_distance_difference_l336_33610


namespace percent_of_percent_l336_33677

theorem percent_of_percent (y : ℝ) (h : y ≠ 0) :
  (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end percent_of_percent_l336_33677


namespace necessary_to_sufficient_negation_l336_33646

theorem necessary_to_sufficient_negation (A B : Prop) :
  (B → A) → (¬A → ¬B) := by sorry

end necessary_to_sufficient_negation_l336_33646


namespace machines_completion_time_l336_33679

theorem machines_completion_time 
  (time_A time_B time_C time_D time_E : ℝ) 
  (h_A : time_A = 4)
  (h_B : time_B = 12)
  (h_C : time_C = 6)
  (h_D : time_D = 8)
  (h_E : time_E = 18) :
  (1 / (1/time_A + 1/time_B + 1/time_C + 1/time_D + 1/time_E)) = 72/49 := by
  sorry

end machines_completion_time_l336_33679


namespace nine_payment_methods_l336_33615

/-- Represents the number of ways to pay an amount using given denominations -/
def paymentMethods (amount : ℕ) (denominations : List ℕ) : ℕ := sorry

/-- The cost of the book in yuan -/
def bookCost : ℕ := 20

/-- Available note denominations in yuan -/
def availableNotes : List ℕ := [10, 5, 1]

/-- Theorem stating that there are 9 ways to pay for the book -/
theorem nine_payment_methods : paymentMethods bookCost availableNotes = 9 := by sorry

end nine_payment_methods_l336_33615


namespace trapezoid_area_l336_33673

/-- A trapezoid with the given properties has an area of 260.4 square centimeters. -/
theorem trapezoid_area (h : ℝ) (b₁ b₂ : ℝ) :
  h = 12 →
  b₁ = 15 →
  b₂ = 13 →
  (b₁^2 - h^2).sqrt + (b₂^2 - h^2).sqrt = 14 →
  (1/2) * (((b₁^2 - h^2).sqrt + (b₂^2 - h^2).sqrt + b₁) + 
           ((b₁^2 - h^2).sqrt + (b₂^2 - h^2).sqrt + b₂)) * h = 260.4 :=
by sorry

end trapezoid_area_l336_33673


namespace soccer_league_female_fraction_l336_33668

theorem soccer_league_female_fraction :
  let last_year_males : ℕ := 30
  let male_increase_rate : ℚ := 11/10
  let female_increase_rate : ℚ := 5/4
  let total_increase_rate : ℚ := 23/20
  let this_year_males : ℚ := last_year_males * male_increase_rate
  let last_year_females : ℚ := (total_increase_rate * (last_year_males : ℚ) - this_year_males) / (female_increase_rate - total_increase_rate)
  let this_year_females : ℚ := last_year_females * female_increase_rate
  let this_year_total : ℚ := this_year_males + this_year_females
  
  (this_year_females / this_year_total) = 75/207 := by
  sorry

end soccer_league_female_fraction_l336_33668


namespace quadratic_coefficients_l336_33674

-- Define the original equation
def original_equation (x : ℝ) : Prop := 3 * x^2 - 2 = 4 * x

-- Define the general form of a quadratic equation
def general_form (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Theorem statement
theorem quadratic_coefficients :
  ∃ (c : ℝ), ∀ (x : ℝ), 
    (original_equation x ↔ general_form 3 (-4) c x) :=
sorry

end quadratic_coefficients_l336_33674


namespace solution_set_when_a_is_3_a_range_for_negative_f_l336_33613

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |2 * x - a|

-- Theorem for part (1)
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x > 0} = {x : ℝ | 1 < x ∧ x < 5/3} := by sorry

-- Theorem for part (2)
theorem a_range_for_negative_f :
  (∀ x < 2, f a x < 0) ↔ a ≥ 4 := by sorry

end solution_set_when_a_is_3_a_range_for_negative_f_l336_33613


namespace min_value_of_expression_l336_33698

theorem min_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 6 ∧ ∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 1/x + 1/y = 1 → 1/(x-1) + 9/(y-1) ≥ min :=
sorry

end min_value_of_expression_l336_33698


namespace gcd_90_405_l336_33647

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l336_33647


namespace prob_ace_king_queen_value_l336_33604

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Probability of drawing Ace, King, Queen in order without replacement -/
def prob_ace_king_queen : ℚ :=
  (NumAces : ℚ) / StandardDeck *
  NumKings / (StandardDeck - 1) *
  NumQueens / (StandardDeck - 2)

theorem prob_ace_king_queen_value :
  prob_ace_king_queen = 8 / 16575 := by
  sorry

end prob_ace_king_queen_value_l336_33604


namespace sphere_surface_area_with_inscribed_parallelepiped_l336_33669

theorem sphere_surface_area_with_inscribed_parallelepiped (a b c : ℝ) (S : ℝ) :
  a = 1 →
  b = 2 →
  c = 2 →
  S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
  S = 9 * Real.pi :=
by sorry

end sphere_surface_area_with_inscribed_parallelepiped_l336_33669


namespace final_alcohol_percentage_l336_33685

/-- Calculates the final alcohol percentage after adding pure alcohol to a solution -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_alcohol : ℝ)
  (h_initial_volume : initial_volume = 6)
  (h_initial_percentage : initial_percentage = 35)
  (h_added_alcohol : added_alcohol = 1.8) :
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol
  final_alcohol / final_volume * 100 = 50 := by
sorry

end final_alcohol_percentage_l336_33685


namespace cakes_served_total_l336_33650

/-- The number of cakes served over two days in a restaurant -/
theorem cakes_served_total (lunch_today : ℕ) (dinner_today : ℕ) (yesterday : ℕ)
  (h1 : lunch_today = 5)
  (h2 : dinner_today = 6)
  (h3 : yesterday = 3) :
  lunch_today + dinner_today + yesterday = 14 :=
by sorry

end cakes_served_total_l336_33650


namespace function_equality_l336_33696

theorem function_equality :
  (∀ x : ℝ, x^2 = (x^6)^(1/3)) ∧
  (∀ x : ℝ, x = (x^3)^(1/3)) := by
  sorry

end function_equality_l336_33696


namespace poverty_definition_l336_33682

-- Define poverty as a string
def poverty : String := "poverty"

-- State the theorem
theorem poverty_definition : poverty = "poverty" := by
  sorry

end poverty_definition_l336_33682


namespace expression_equals_sum_l336_33694

theorem expression_equals_sum (a b c : ℝ) (ha : a = 13) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end expression_equals_sum_l336_33694


namespace regular_polygon_with_150_degree_angles_has_12_sides_l336_33653

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 150) →
  (180 * (n - 2) : ℝ) = (150 * n : ℝ) →
  n = 12 := by
sorry

end regular_polygon_with_150_degree_angles_has_12_sides_l336_33653


namespace dave_total_earnings_l336_33644

def dave_earnings (hourly_wage : ℝ) (monday_hours : ℝ) (tuesday_hours : ℝ) : ℝ :=
  hourly_wage * (monday_hours + tuesday_hours)

theorem dave_total_earnings :
  dave_earnings 6 6 2 = 48 := by
  sorry

end dave_total_earnings_l336_33644


namespace luke_weed_eating_money_l336_33624

def mowing_money : ℕ := 9
def weeks_lasting : ℕ := 9
def weekly_spending : ℕ := 3

theorem luke_weed_eating_money :
  mowing_money + (weeks_lasting * weekly_spending - mowing_money) = 18 := by
  sorry

end luke_weed_eating_money_l336_33624


namespace law_firm_associates_tenure_l336_33657

/-- 
Given a law firm where:
- 30% of associates are second-year associates
- 60% of associates are not first-year associates

This theorem proves that 30% of associates have been at the firm for more than two years.
-/
theorem law_firm_associates_tenure (total : ℝ) (second_year : ℝ) (not_first_year : ℝ) 
  (h1 : second_year = 0.3 * total) 
  (h2 : not_first_year = 0.6 * total) : 
  total - (second_year + (total - not_first_year)) = 0.3 * total := by
  sorry

end law_firm_associates_tenure_l336_33657


namespace paula_paint_theorem_l336_33686

/-- Represents the number of rooms that can be painted with a given amount of paint -/
structure PaintCapacity where
  total_rooms : ℕ
  cans : ℕ

/-- Calculates the number of cans needed to paint a given number of rooms -/
def cans_needed (initial : PaintCapacity) (lost_cans : ℕ) (rooms_to_paint : ℕ) : ℕ :=
  let rooms_per_can := initial.total_rooms / initial.cans
  rooms_to_paint / rooms_per_can

theorem paula_paint_theorem (initial : PaintCapacity) (lost_cans : ℕ) :
  initial.total_rooms = 40 →
  initial.cans = initial.cans - lost_cans + lost_cans →
  lost_cans = 6 →
  cans_needed initial lost_cans 30 = 18 := by
  sorry

#check paula_paint_theorem

end paula_paint_theorem_l336_33686


namespace value_of_a_l336_33656

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

-- State the theorem
theorem value_of_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 3 * a * x^2 + 6 * x) → 
  deriv (f a) (-1) = 3 → 
  a = 3 := by
  sorry

end value_of_a_l336_33656


namespace opposite_sign_power_l336_33634

theorem opposite_sign_power (x y : ℝ) : 
  (|x + 3| * (y - 2)^2 ≤ 0 ∧ |x + 3| + (y - 2)^2 = 0) → x^y = 9 := by sorry

end opposite_sign_power_l336_33634


namespace unique_solution_divisibility_l336_33665

theorem unique_solution_divisibility : ∀ a b : ℕ+,
  (∃ k l : ℕ+, (a^2 + b^2 : ℕ) * k = a^3 + 1 ∧ (a^2 + b^2 : ℕ) * l = b^3 + 1) →
  a = 1 ∧ b = 1 := by
sorry

end unique_solution_divisibility_l336_33665


namespace alcohol_dilution_l336_33605

/-- Proves that adding 3 liters of water to 11 liters of a 42% alcohol solution 
    results in a new mixture with 33% alcohol. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 11 ∧ 
  initial_concentration = 0.42 ∧ 
  added_water = 3 ∧ 
  final_concentration = 0.33 →
  initial_volume * initial_concentration = 
    (initial_volume + added_water) * final_concentration :=
by sorry

end alcohol_dilution_l336_33605


namespace frog_grasshopper_jump_difference_l336_33655

theorem frog_grasshopper_jump_difference :
  let grasshopper_jump : ℕ := 9
  let frog_jump : ℕ := 12
  frog_jump - grasshopper_jump = 3 := by sorry

end frog_grasshopper_jump_difference_l336_33655


namespace winter_olympics_merchandise_l336_33633

def total_items : ℕ := 180
def figurine_cost : ℕ := 80
def pendant_cost : ℕ := 50
def total_spent : ℕ := 11400
def figurine_price : ℕ := 100
def pendant_price : ℕ := 60
def min_profit : ℕ := 2900

theorem winter_olympics_merchandise (x y : ℕ) (m : ℕ) : 
  x + y = total_items ∧ 
  figurine_cost * x + pendant_cost * y = total_spent ∧
  (pendant_price - pendant_cost) * m + (figurine_price - figurine_cost) * (total_items - m) ≥ min_profit →
  x = 80 ∧ y = 100 ∧ m ≤ 70 := by
  sorry

end winter_olympics_merchandise_l336_33633


namespace sqrt_expression_equality_l336_33688

theorem sqrt_expression_equality : 
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6 := by
  sorry

end sqrt_expression_equality_l336_33688


namespace composition_of_even_is_even_l336_33618

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) : 
  EvenFunction (g ∘ g) := by
  sorry

end composition_of_even_is_even_l336_33618


namespace intersection_empty_iff_a_in_range_l336_33689

/-- Given sets A and B, prove that their intersection is empty if and only if a is in the specified range -/
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  let A := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
  let B := {x : ℝ | x < -1 ∨ x > 5}
  (A ∩ B = ∅) ↔ (a > 3 ∨ (-1/2 ≤ a ∧ a ≤ 2)) :=
by sorry

end intersection_empty_iff_a_in_range_l336_33689


namespace oranges_left_l336_33663

/-- Proves that the number of oranges Joan is left with is equal to the number she picked minus the number Sara sold. -/
theorem oranges_left (joan_picked : ℕ) (sara_sold : ℕ) (joan_left : ℕ)
  (h1 : joan_picked = 37)
  (h2 : sara_sold = 10)
  (h3 : joan_left = 27) :
  joan_left = joan_picked - sara_sold :=
by sorry

end oranges_left_l336_33663
