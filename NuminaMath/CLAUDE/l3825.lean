import Mathlib

namespace NUMINAMATH_CALUDE_max_min_values_on_interval_l3825_382574

-- Define the function f(x) = 3x - x³
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define the interval [2, 3]
def interval : Set ℝ := { x | 2 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem max_min_values_on_interval :
  (∀ x ∈ interval, f x ≤ f 2) ∧
  (∀ x ∈ interval, f 3 ≤ f x) ∧
  (f 2 = -2) ∧
  (f 3 = -18) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_on_interval_l3825_382574


namespace NUMINAMATH_CALUDE_max_cubes_fit_l3825_382537

def small_cube_edge : ℝ := 10.7
def large_cube_edge : ℝ := 100

theorem max_cubes_fit (small_cube_edge : ℝ) (large_cube_edge : ℝ) :
  small_cube_edge = 10.7 →
  large_cube_edge = 100 →
  ⌊(large_cube_edge ^ 3) / (small_cube_edge ^ 3)⌋ = 816 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_fit_l3825_382537


namespace NUMINAMATH_CALUDE_remi_water_consumption_l3825_382513

/-- The amount of water Remi drinks in a week, given his bottle capacity, refill frequency, and spills. -/
def water_consumed (bottle_capacity : ℕ) (refills_per_day : ℕ) (days : ℕ) (spill1 : ℕ) (spill2 : ℕ) : ℕ :=
  bottle_capacity * refills_per_day * days - (spill1 + spill2)

/-- Theorem stating that Remi drinks 407 ounces of water in 7 days under the given conditions. -/
theorem remi_water_consumption :
  water_consumed 20 3 7 5 8 = 407 := by
  sorry

#eval water_consumed 20 3 7 5 8

end NUMINAMATH_CALUDE_remi_water_consumption_l3825_382513


namespace NUMINAMATH_CALUDE_no_real_roots_l3825_382507

theorem no_real_roots : ¬∃ x : ℝ, x + Real.sqrt (2 * x - 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3825_382507


namespace NUMINAMATH_CALUDE_fair_remaining_money_l3825_382545

/-- Calculates the remaining money after purchases at a fair --/
theorem fair_remaining_money 
  (initial_amount : ℝ) 
  (toy_cost : ℝ) 
  (hot_dog_cost : ℝ) 
  (candy_apple_cost : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : initial_amount = 15)
  (h2 : toy_cost = 2)
  (h3 : hot_dog_cost = 3.5)
  (h4 : candy_apple_cost = 1.5)
  (h5 : discount_percentage = 0.5)
  (h6 : hot_dog_cost ≥ toy_cost ∧ hot_dog_cost ≥ candy_apple_cost) :
  initial_amount - (toy_cost + hot_dog_cost * (1 - discount_percentage) + candy_apple_cost) = 9.75 := by
  sorry


end NUMINAMATH_CALUDE_fair_remaining_money_l3825_382545


namespace NUMINAMATH_CALUDE_third_place_prize_l3825_382597

theorem third_place_prize (total_prize : ℕ) (num_novels : ℕ) (first_prize : ℕ) (second_prize : ℕ) (other_prize : ℕ) :
  total_prize = 800 →
  num_novels = 18 →
  first_prize = 200 →
  second_prize = 150 →
  other_prize = 22 →
  (num_novels - 3) * other_prize + first_prize + second_prize + 120 = total_prize :=
by sorry

end NUMINAMATH_CALUDE_third_place_prize_l3825_382597


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l3825_382560

theorem pure_imaginary_product (a : ℝ) : 
  (∃ b : ℝ, (1 - Complex.I) * (a + Complex.I) = Complex.I * b) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l3825_382560


namespace NUMINAMATH_CALUDE_disjunction_true_l3825_382504

theorem disjunction_true : 
  (∀ x : ℝ, x < 0 → 2^x > x) ∨ (∃ x : ℝ, x^2 + x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_disjunction_true_l3825_382504


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3825_382553

/-- Given two orthogonal vectors a = (x-1, y) and b = (1, 2), with x > 0 and y > 0,
    the minimum value of 1/x + 1/y is 3 + 2√2 -/
theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h_orthogonal : (x - 1) * 1 + y * 2 = 0) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → (x' - 1) * 1 + y' * 2 = 0 → 1 / x' + 1 / y' ≥ 1 / x + 1 / y) ∧
  1 / x + 1 / y = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3825_382553


namespace NUMINAMATH_CALUDE_problem_statement_l3825_382571

theorem problem_statement (x y : ℝ) (h1 : x = 2 * y) (h2 : y ≠ 0) :
  (x + 2 * y) - (2 * x + y) = -y := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3825_382571


namespace NUMINAMATH_CALUDE_two_from_same_class_three_from_same_class_l3825_382555

/-- A function representing the distribution of students among classes -/
def Distribution (n : ℕ) := Fin 3 → ℕ

/-- The sum of students in all classes equals the total number of students -/
def valid_distribution (n : ℕ) (d : Distribution n) : Prop :=
  (d 0) + (d 1) + (d 2) = n

/-- There exists a class with at least k students -/
def exists_class_with_k_students (n k : ℕ) (d : Distribution n) : Prop :=
  ∃ i : Fin 3, d i ≥ k

theorem two_from_same_class (n : ℕ) (h : n ≥ 4) :
  ∀ d : Distribution n, valid_distribution n d → exists_class_with_k_students n 2 d :=
sorry

theorem three_from_same_class (n : ℕ) (h : n ≥ 7) :
  ∀ d : Distribution n, valid_distribution n d → exists_class_with_k_students n 3 d :=
sorry

end NUMINAMATH_CALUDE_two_from_same_class_three_from_same_class_l3825_382555


namespace NUMINAMATH_CALUDE_estate_area_calculation_l3825_382506

-- Define the scale conversion factor
def scale : ℝ := 500

-- Define the map dimensions
def map_width : ℝ := 5
def map_height : ℝ := 3

-- Define the actual dimensions
def actual_width : ℝ := scale * map_width
def actual_height : ℝ := scale * map_height

-- Define the actual area
def actual_area : ℝ := actual_width * actual_height

-- Theorem to prove
theorem estate_area_calculation :
  actual_area = 3750000 := by
  sorry

end NUMINAMATH_CALUDE_estate_area_calculation_l3825_382506


namespace NUMINAMATH_CALUDE_x_convergence_interval_l3825_382505

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 8) / (x n + 7)

theorem x_convergence_interval :
  ∃ m : ℕ, 81 ≤ m ∧ m ≤ 242 ∧ x m ≤ 4 + 1 / (2^18) ∧
  ∀ k : ℕ, 0 < k ∧ k < 81 → x k > 4 + 1 / (2^18) := by
  sorry

end NUMINAMATH_CALUDE_x_convergence_interval_l3825_382505


namespace NUMINAMATH_CALUDE_carlton_outfits_l3825_382586

/-- Represents Carlton's wardrobe and outfit combinations -/
structure Wardrobe where
  button_up_shirts : ℕ
  sweater_vests : ℕ
  outfits : ℕ

/-- Calculates the number of outfits for Carlton -/
def calculate_outfits (w : Wardrobe) : Prop :=
  w.button_up_shirts = 3 ∧
  w.sweater_vests = 2 * w.button_up_shirts ∧
  w.outfits = w.button_up_shirts * w.sweater_vests

/-- Theorem stating that Carlton has 18 outfits -/
theorem carlton_outfits :
  ∃ w : Wardrobe, calculate_outfits w ∧ w.outfits = 18 := by
  sorry


end NUMINAMATH_CALUDE_carlton_outfits_l3825_382586


namespace NUMINAMATH_CALUDE_opposite_pairs_l3825_382559

theorem opposite_pairs :
  (- (-2) = - (- (-2))) ∧
  ((-1)^2 = - ((-1)^2)) ∧
  ((-2)^3 ≠ -6) ∧
  ((-2)^7 = -2^7) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l3825_382559


namespace NUMINAMATH_CALUDE_range_of_3a_minus_b_l3825_382538

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : 1 ≤ a + b) (h2 : a + b ≤ 4) 
  (h3 : -1 ≤ a - b) (h4 : a - b ≤ 2) : 
  (∃ (x y : ℝ), (1 ≤ x + y ∧ x + y ≤ 4 ∧ -1 ≤ x - y ∧ x - y ≤ 2 ∧ 3*x - y = -1)) ∧
  (∃ (x y : ℝ), (1 ≤ x + y ∧ x + y ≤ 4 ∧ -1 ≤ x - y ∧ x - y ≤ 2 ∧ 3*x - y = 8)) ∧
  (∀ (x y : ℝ), 1 ≤ x + y → x + y ≤ 4 → -1 ≤ x - y → x - y ≤ 2 → -1 ≤ 3*x - y ∧ 3*x - y ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_b_l3825_382538


namespace NUMINAMATH_CALUDE_lower_right_is_four_l3825_382558

def Grid := Fin 4 → Fin 4 → Fin 4

def valid_grid (g : Grid) : Prop :=
  (∀ i j k, i ≠ j → g i k ≠ g j k) ∧
  (∀ i j k, i ≠ j → g k i ≠ g k j)

def initial_conditions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 3 = 2 ∧ g 1 2 = 3 ∧ g 2 0 = 3 ∧ g 3 1 = 0

theorem lower_right_is_four (g : Grid) 
  (h1 : valid_grid g) 
  (h2 : initial_conditions g) : 
  g 3 3 = 3 := by sorry

end NUMINAMATH_CALUDE_lower_right_is_four_l3825_382558


namespace NUMINAMATH_CALUDE_larger_integer_problem_l3825_382532

theorem larger_integer_problem (a b : ℕ+) : 
  a * b = 168 → 
  (a : ℤ) - (b : ℤ) = 4 ∨ (b : ℤ) - (a : ℤ) = 4 → 
  max a b = 14 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l3825_382532


namespace NUMINAMATH_CALUDE_circle_center_from_intersection_l3825_382511

/-- Given a parabola y = k x^2 and a circle x^2 - 2px + y^2 - 2qy = 0,
    if the abscissas of their intersection points are the roots of x^3 + ax + b = 0,
    then the center of the circle is (-b/2, (1-a)/2). -/
theorem circle_center_from_intersection (k a b : ℝ) :
  ∃ (p q : ℝ),
    (∀ x y : ℝ, y = k * x^2 ∧ x^2 - 2*p*x + y^2 - 2*q*y = 0 →
      x^3 + a*x + b = 0) →
    (p = b/2 ∧ q = (a-1)/2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_from_intersection_l3825_382511


namespace NUMINAMATH_CALUDE_staff_members_count_correct_staff_count_l3825_382519

theorem staff_members_count (allowance_days : ℕ) (allowance_rate : ℕ) 
  (accountant_amount : ℕ) (petty_cash : ℕ) : ℕ :=
  let allowance_per_staff := allowance_days * allowance_rate
  let total_amount := accountant_amount + petty_cash
  total_amount / allowance_per_staff

theorem correct_staff_count : 
  staff_members_count 30 100 65000 1000 = 22 := by sorry

end NUMINAMATH_CALUDE_staff_members_count_correct_staff_count_l3825_382519


namespace NUMINAMATH_CALUDE_inequality_proof_l3825_382501

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/2 + z/3 ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3825_382501


namespace NUMINAMATH_CALUDE_sqrt_pattern_l3825_382561

theorem sqrt_pattern (n : ℕ+) : 
  Real.sqrt (1 + 1 / n^2 + 1 / (n + 1)^2) = 1 + 1 / (n * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l3825_382561


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l3825_382524

theorem hexagon_perimeter (AB BC CD DE EF : ℝ) (AC AD AE AF : ℝ) : 
  AB = 1 →
  BC = 1 →
  CD = 1 →
  DE = 2 →
  EF = 1 →
  AC^2 = AB^2 + BC^2 →
  AD^2 = AC^2 + CD^2 →
  AE^2 = AD^2 + DE^2 →
  AF^2 = AE^2 + EF^2 →
  AB + BC + CD + DE + EF + AF = 6 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l3825_382524


namespace NUMINAMATH_CALUDE_unit_digit_sum_factorials_l3825_382530

def factorial (n : ℕ) : ℕ := Nat.factorial n

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def unit_digit (n : ℕ) : ℕ := n % 10

theorem unit_digit_sum_factorials :
  unit_digit (sum_factorials 2012) = unit_digit (sum_factorials 4) :=
sorry

end NUMINAMATH_CALUDE_unit_digit_sum_factorials_l3825_382530


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3825_382576

/-- Systematic sampling function that generates a sample number for a given group -/
def sampleNumber (x : ℕ) (k : ℕ) : ℕ :=
  (x + 33 * k) % 100 + 100 * k

/-- Generates the full sample of 10 numbers given an initial value x -/
def generateSample (x : ℕ) : List ℕ :=
  List.range 10 |>.map (sampleNumber x)

/-- Checks if a number ends with the digits 87 -/
def endsWith87 (n : ℕ) : Bool :=
  n % 100 = 87

/-- Set of possible x values that result in a sample number ending with 87 -/
def possibleXValues : Set ℕ :=
  {x | x ∈ Finset.range 100 ∧ ∃ k, k ∈ Finset.range 10 ∧ endsWith87 (sampleNumber x k)}

theorem systematic_sampling_theorem :
  (generateSample 24 = [24, 157, 290, 423, 556, 689, 822, 955, 88, 221]) ∧
  (possibleXValues = {21, 22, 23, 54, 55, 56, 87, 88, 89, 90}) := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3825_382576


namespace NUMINAMATH_CALUDE_matthews_crayons_count_l3825_382542

/-- The number of crayons Annie starts with -/
def initial_crayons : ℕ := 4

/-- The number of crayons Annie ends with -/
def final_crayons : ℕ := 40

/-- The number of crayons Matthew gave to Annie -/
def matthews_crayons : ℕ := final_crayons - initial_crayons

theorem matthews_crayons_count : matthews_crayons = 36 := by
  sorry

end NUMINAMATH_CALUDE_matthews_crayons_count_l3825_382542


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3825_382534

theorem trigonometric_identities :
  ∀ α : ℝ,
  (((Real.sqrt 3 * Real.sin (-1200 * π / 180)) / Real.tan (11 * π / 3)) - 
   (Real.cos (585 * π / 180) * Real.tan (-37 * π / 4)) = 
   Real.sqrt 3 / 2 - Real.sqrt 2 / 2) ∧
  ((Real.cos (α - π / 2) / Real.sin (5 * π / 2 + α)) * 
   Real.sin (α - 2 * π) * Real.cos (2 * π - α) = 
   Real.sin α ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3825_382534


namespace NUMINAMATH_CALUDE_line_perp_plane_necessity_not_sufficiency_l3825_382502

-- Define the types for lines and planes
variable (L : Type*) [NormedAddCommGroup L] [InnerProductSpace ℝ L]
variable (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P]

-- Define the perpendicular relation between lines and between a line and a plane
variable (perpendicular_lines : L → L → Prop)
variable (perpendicular_line_plane : L → P → Prop)

-- Define the containment relation between a line and a plane
variable (contained_in : L → P → Prop)

-- State the theorem
theorem line_perp_plane_necessity_not_sufficiency
  (m n : L) (α : P) (h_contained : contained_in n α) :
  (perpendicular_line_plane m α → perpendicular_lines m n) ∧
  ∃ (m' n' : L) (α' : P),
    contained_in n' α' ∧
    perpendicular_lines m' n' ∧
    ¬perpendicular_line_plane m' α' :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_necessity_not_sufficiency_l3825_382502


namespace NUMINAMATH_CALUDE_rhombus_area_l3825_382500

/-- The area of a rhombus with vertices at (0, 3.5), (8, 0), (0, -3.5), and (-8, 0) is 56 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (8, 0), (0, -3.5), (-8, 0)]
  let diag1 : ℝ := |3.5 - (-3.5)|
  let diag2 : ℝ := |8 - (-8)|
  (diag1 * diag2) / 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3825_382500


namespace NUMINAMATH_CALUDE_jerry_candy_problem_l3825_382581

/-- Given a total number of candy pieces, number of bags, and the distribution of chocolate types,
    calculate the number of non-chocolate candy pieces. -/
def non_chocolate_candy (total_candy : ℕ) (total_bags : ℕ) (heart_bags : ℕ) (kiss_bags : ℕ) : ℕ :=
  total_candy - (heart_bags + kiss_bags) * (total_candy / total_bags)

/-- Theorem stating that given 63 pieces of candy divided into 9 bags,
    with 2 bags of chocolate hearts and 3 bags of chocolate kisses,
    the number of non-chocolate candies is 28. -/
theorem jerry_candy_problem :
  non_chocolate_candy 63 9 2 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_jerry_candy_problem_l3825_382581


namespace NUMINAMATH_CALUDE_binomial_expansion_special_case_l3825_382549

theorem binomial_expansion_special_case : 7^4 + 4*(7^3) + 6*(7^2) + 4*7 + 1 = 8^4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_special_case_l3825_382549


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3825_382522

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ, are_parallel (1, m) (m, 4) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3825_382522


namespace NUMINAMATH_CALUDE_cosine_sum_theorem_l3825_382521

theorem cosine_sum_theorem (x m : Real) (h : Real.cos (x - Real.pi/6) = m) :
  Real.cos x + Real.cos (x - Real.pi/3) = Real.sqrt 3 * m := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_theorem_l3825_382521


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l3825_382525

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents the resulting structure after modifications -/
structure ModifiedCube where
  original : Cube 9
  small_cubes : ℕ := 27
  removed_corners : ℕ := 8

/-- Calculates the surface area of the modified cube structure -/
def surface_area (mc : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that the surface area of the modified cube is 1056 -/
theorem modified_cube_surface_area :
  ∀ (mc : ModifiedCube), surface_area mc = 1056 :=
sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l3825_382525


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_inequality_l3825_382595

theorem integer_pairs_satisfying_inequality :
  ∀ a b : ℕ+, 
    (11 * a * b ≤ a^3 - b^3 ∧ a^3 - b^3 ≤ 12 * a * b) ↔ 
    ((a = 30 ∧ b = 25) ∨ (a = 8 ∧ b = 4)) := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_inequality_l3825_382595


namespace NUMINAMATH_CALUDE_patrick_pencil_purchase_l3825_382536

/-- The number of pencils Patrick purchased -/
def num_pencils : ℕ := 60

/-- The ratio of cost price to selling price -/
def cost_to_sell_ratio : ℚ := 1.3333333333333333

/-- The number of pencils whose selling price equals the total loss -/
def loss_in_pencils : ℕ := 20

theorem patrick_pencil_purchase :
  num_pencils = 60 ∧
  (cost_to_sell_ratio - 1) * num_pencils = loss_in_pencils :=
sorry

end NUMINAMATH_CALUDE_patrick_pencil_purchase_l3825_382536


namespace NUMINAMATH_CALUDE_seashell_collection_l3825_382593

theorem seashell_collection (x y : ℝ) : 
  let initial := x
  let additional := y
  let total := initial + additional
  let after_jessica := (2/3) * total
  let after_henry := (3/4) * after_jessica
  after_henry = (1/2) * total
  := by sorry

end NUMINAMATH_CALUDE_seashell_collection_l3825_382593


namespace NUMINAMATH_CALUDE_f_properties_l3825_382567

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def decreasing_on_8_to_inf (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 8 → y > x → f y < f x

def f_plus_8_is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 8) = f (-x + 8)

-- State the theorem
theorem f_properties (h1 : decreasing_on_8_to_inf f) (h2 : f_plus_8_is_even f) :
  f 7 = f 9 ∧ f 7 > f 10 := by sorry

end NUMINAMATH_CALUDE_f_properties_l3825_382567


namespace NUMINAMATH_CALUDE_sphere_radius_l3825_382520

theorem sphere_radius (r_A : ℝ) : 
  let r_B : ℝ := 10
  (r_A^2 / r_B^2 = 16) → r_A = 40 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_l3825_382520


namespace NUMINAMATH_CALUDE_whale_ratio_theorem_l3825_382518

/-- The ratio of male whales on the third trip to the first trip -/
def whale_ratio : ℚ := 1 / 2

/-- The number of male whales on the first trip -/
def first_trip_males : ℕ := 28

/-- The number of female whales on the first trip -/
def first_trip_females : ℕ := 2 * first_trip_males

/-- The number of baby whales on the second trip -/
def second_trip_babies : ℕ := 8

/-- The total number of whales observed -/
def total_whales : ℕ := 178

/-- The number of male whales on the third trip -/
def third_trip_males : ℕ := total_whales - (first_trip_males + first_trip_females + second_trip_babies + 2 * second_trip_babies + first_trip_females)

theorem whale_ratio_theorem : 
  (third_trip_males : ℚ) / first_trip_males = whale_ratio := by
  sorry

end NUMINAMATH_CALUDE_whale_ratio_theorem_l3825_382518


namespace NUMINAMATH_CALUDE_sqrt_sum_expression_l3825_382539

theorem sqrt_sum_expression (a : ℝ) (h : a ≥ 1) :
  Real.sqrt (a + 2 * Real.sqrt (a - 1)) + Real.sqrt (a - 2 * Real.sqrt (a - 1)) =
    if 1 ≤ a ∧ a ≤ 2 then 2 else 2 * Real.sqrt (a - 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_expression_l3825_382539


namespace NUMINAMATH_CALUDE_power_of_two_with_three_identical_digits_l3825_382527

theorem power_of_two_with_three_identical_digits :
  ∃ (k : ℕ), k = 39 ∧ ∃ (d : ℕ), d < 10 ∧ (2^k) % 1000 = d * 111 :=
sorry

end NUMINAMATH_CALUDE_power_of_two_with_three_identical_digits_l3825_382527


namespace NUMINAMATH_CALUDE_weight_of_four_cakes_l3825_382535

/-- The weight of a cake in grams -/
def cake_weight : ℕ := sorry

/-- The weight of a piece of bread in grams -/
def bread_weight : ℕ := sorry

/-- Theorem stating the weight of 4 cakes -/
theorem weight_of_four_cakes : 4 * cake_weight = 800 :=
by
  have h1 : 3 * cake_weight + 5 * bread_weight = 1100 := sorry
  have h2 : cake_weight = bread_weight + 100 := sorry
  sorry

#check weight_of_four_cakes

end NUMINAMATH_CALUDE_weight_of_four_cakes_l3825_382535


namespace NUMINAMATH_CALUDE_book_pages_count_l3825_382569

/-- The number of pages Lance read on the first day -/
def pages_day1 : ℕ := 35

/-- The number of pages Lance read on the second day -/
def pages_day2 : ℕ := pages_day1 - 5

/-- The number of pages Lance will read on the third day -/
def pages_day3 : ℕ := 35

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_day1 + pages_day2 + pages_day3

theorem book_pages_count : total_pages = 100 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l3825_382569


namespace NUMINAMATH_CALUDE_rabbit_carrots_l3825_382583

/-- Represents the number of carrots in each burrow -/
def carrots_per_burrow : ℕ := 2

/-- Represents the number of apples in each tree -/
def apples_per_tree : ℕ := 3

/-- Represents the difference between the number of burrows and trees -/
def burrow_tree_difference : ℕ := 3

theorem rabbit_carrots (burrows trees : ℕ) : 
  burrows = trees + burrow_tree_difference →
  carrots_per_burrow * burrows = apples_per_tree * trees →
  carrots_per_burrow * burrows = 18 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrots_l3825_382583


namespace NUMINAMATH_CALUDE_sum_of_possible_radii_l3825_382564

/-- A circle with center C(r, r) is tangent to the positive x-axis and y-axis,
    and externally tangent to another circle centered at (3,3) with radius 2.
    This theorem states that the sum of all possible radii r is 16. -/
theorem sum_of_possible_radii : ∃ r₁ r₂ : ℝ,
  (r₁ - 3)^2 + (r₁ - 3)^2 = (r₁ + 2)^2 ∧
  (r₂ - 3)^2 + (r₂ - 3)^2 = (r₂ + 2)^2 ∧
  r₁ + r₂ = 16 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_radii_l3825_382564


namespace NUMINAMATH_CALUDE_drug_molecule_diameter_scientific_notation_l3825_382566

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem drug_molecule_diameter_scientific_notation :
  toScientificNotation 0.00000008 = ScientificNotation.mk 8 (-8) sorry := by
  sorry

end NUMINAMATH_CALUDE_drug_molecule_diameter_scientific_notation_l3825_382566


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_neg_four_squared_l3825_382590

theorem arithmetic_sqrt_of_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_neg_four_squared_l3825_382590


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3825_382548

/-- Given that 8 bowling balls weigh the same as 5 kayaks, and 4 kayaks weigh 120 pounds,
    prove that one bowling ball weighs 18.75 pounds. -/
theorem bowling_ball_weight :
  ∀ (bowl_weight kayak_weight : ℝ),
    8 * bowl_weight = 5 * kayak_weight →
    4 * kayak_weight = 120 →
    bowl_weight = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3825_382548


namespace NUMINAMATH_CALUDE_decreasing_function_condition_l3825_382568

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x

-- State the theorem
theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ (1 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_condition_l3825_382568


namespace NUMINAMATH_CALUDE_five_Y_three_equals_64_l3825_382570

-- Define the Y operation
def Y (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2

-- Theorem statement
theorem five_Y_three_equals_64 : Y 5 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_five_Y_three_equals_64_l3825_382570


namespace NUMINAMATH_CALUDE_and_false_necessary_not_sufficient_for_or_false_l3825_382551

theorem and_false_necessary_not_sufficient_for_or_false (p q : Prop) :
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ 
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_and_false_necessary_not_sufficient_for_or_false_l3825_382551


namespace NUMINAMATH_CALUDE_triangle_area_l3825_382577

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 6 → 
  a = 2 * c → 
  B = π / 3 → 
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3825_382577


namespace NUMINAMATH_CALUDE_gwen_final_amount_l3825_382552

def initial_amount : ℚ := 5.00
def candy_expense : ℚ := 3.25
def recycling_income : ℚ := 1.50
def card_expense : ℚ := 0.70

theorem gwen_final_amount :
  initial_amount - candy_expense + recycling_income - card_expense = 2.55 := by
  sorry

end NUMINAMATH_CALUDE_gwen_final_amount_l3825_382552


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_tangent_line_l3825_382512

/-- The ellipse E -/
def E (x y : ℝ) : Prop := y^2 / 8 + x^2 / 4 = 1

/-- The line l -/
def l (x y : ℝ) : Prop := x + y - 3 = 0

/-- The function f -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

theorem ellipse_line_intersection (A B : ℝ × ℝ) :
  E A.1 A.2 ∧ E B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2 ∧ A ≠ B ∧
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 2 →
  ∀ x y, l x y ↔ x + y - 3 = 0 :=
sorry

theorem tangent_line (P : ℝ × ℝ) :
  P = (1, 2) ∧ (∀ x, f x = x^2 - 3*x + 4) ∧
  (∀ x y, l x y ↔ x + y - 3 = 0) →
  ∃ a b, f P.1 = P.2 ∧ (deriv f) P.1 = -1 ∧
  ∀ x, f x = x^2 - a*x + b :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_tangent_line_l3825_382512


namespace NUMINAMATH_CALUDE_sets_problem_l3825_382544

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 9*x + 18 ≥ 0}
def B : Set ℝ := {x | -2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem sets_problem :
  (A ∪ B = Set.univ) ∧
  ((Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6}) ∧
  (∀ a : ℝ, C a ⊆ B → -2 ≤ a ∧ a ≤ 8) := by
  sorry


end NUMINAMATH_CALUDE_sets_problem_l3825_382544


namespace NUMINAMATH_CALUDE_sign_determination_l3825_382599

theorem sign_determination (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a / 5 > 0)
  (h2 : -b / (7*a) > 0)
  (h3 : 11 / (a*b*c) > 0)
  (h4 : -18 / (a*b*c*d) > 0) :
  a > 0 ∧ b < 0 ∧ c < 0 ∧ d < 0 := by
sorry

end NUMINAMATH_CALUDE_sign_determination_l3825_382599


namespace NUMINAMATH_CALUDE_A_union_complement_B_eq_A_l3825_382503

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {1,3,5}
def B : Set Nat := {2,4}

theorem A_union_complement_B_eq_A : A ∪ (U \ B) = A := by
  sorry

end NUMINAMATH_CALUDE_A_union_complement_B_eq_A_l3825_382503


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3825_382510

theorem hyperbola_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m + y^2 / 2 = 1) →
  (∃ a b c : ℝ, a^2 = 2 ∧ b^2 = -m ∧ c^2 = a^2 + b^2 ∧ c^2 / a^2 = 4) →
  m = -6 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3825_382510


namespace NUMINAMATH_CALUDE_sqrt_sum_rational_implies_both_rational_l3825_382592

theorem sqrt_sum_rational_implies_both_rational 
  (a b : ℚ) 
  (h : ∃ (q : ℚ), q = Real.sqrt a + Real.sqrt b) : 
  (∃ (r : ℚ), r = Real.sqrt a) ∧ (∃ (s : ℚ), s = Real.sqrt b) := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_rational_implies_both_rational_l3825_382592


namespace NUMINAMATH_CALUDE_juice_problem_l3825_382514

/-- Given the number of oranges per glass and the total number of oranges,
    calculate the number of glasses of juice. -/
def glasses_of_juice (oranges_per_glass : ℕ) (total_oranges : ℕ) : ℕ :=
  total_oranges / oranges_per_glass

theorem juice_problem :
  glasses_of_juice 2 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_juice_problem_l3825_382514


namespace NUMINAMATH_CALUDE_matthew_initial_cakes_l3825_382572

/-- The number of friends Matthew has -/
def num_friends : ℕ := 4

/-- The initial number of crackers Matthew has -/
def initial_crackers : ℕ := 10

/-- The number of cakes each person eats -/
def cakes_eaten_per_person : ℕ := 2

/-- The number of crackers given to each friend -/
def crackers_per_friend : ℕ := initial_crackers / num_friends

/-- The initial number of cakes Matthew had -/
def initial_cakes : ℕ := 2 * num_friends * crackers_per_friend

theorem matthew_initial_cakes :
  initial_cakes = 16 :=
sorry

end NUMINAMATH_CALUDE_matthew_initial_cakes_l3825_382572


namespace NUMINAMATH_CALUDE_dartboard_central_angles_l3825_382578

/-- Represents a region on a circular dartboard -/
structure DartboardRegion where
  probability : ℚ
  centralAngle : ℚ

/-- Theorem: Given the probabilities of hitting regions A and B on a circular dartboard,
    prove that their central angles are 45° and 30° respectively -/
theorem dartboard_central_angles 
  (regionA regionB : DartboardRegion)
  (hA : regionA.probability = 1/8)
  (hB : regionB.probability = 1/12)
  (h_total : regionA.centralAngle + regionB.centralAngle ≤ 360) :
  regionA.centralAngle = 45 ∧ regionB.centralAngle = 30 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_central_angles_l3825_382578


namespace NUMINAMATH_CALUDE_cubic_roots_difference_squared_l3825_382531

theorem cubic_roots_difference_squared (r s : ℝ) : 
  (∃ c : ℝ, r^3 - 2*r + c = 0 ∧ s^3 - 2*s + c = 0 ∧ 1^3 - 2*1 + c = 0) →
  (r - s)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_difference_squared_l3825_382531


namespace NUMINAMATH_CALUDE_triangle_area_l3825_382540

theorem triangle_area (a b c : ℝ) (h₁ : a = 15) (h₂ : b = 36) (h₃ : c = 39) :
  (1/2) * a * b = 270 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3825_382540


namespace NUMINAMATH_CALUDE_last_three_average_l3825_382541

theorem last_three_average (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  a + d = 11 →
  d = 4 →
  (b + c + d) / 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l3825_382541


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3825_382554

/-- The equation (x+y)^2 = x^2 + y^2 + 2 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (x y : ℝ), (x + y)^2 = x^2 + y^2 + 2 ↔ x * y = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3825_382554


namespace NUMINAMATH_CALUDE_cone_base_circumference_l3825_382594

/-- The circumference of the base of a right circular cone formed from a circular piece of paper 
    with radius 6 inches, after removing a 180° sector, is equal to 6π inches. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) : 
  (2 * π * r) * (1/2) = 6 * π := by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l3825_382594


namespace NUMINAMATH_CALUDE_largest_angle_obtuse_isosceles_triangle_l3825_382526

/-- An obtuse isosceles triangle with one angle measuring 20 degrees has its largest angle measuring 140 degrees. -/
theorem largest_angle_obtuse_isosceles_triangle (A B C : ℝ) :
  A = 20 → -- Angle A measures 20 degrees
  A + B + C = 180 → -- Sum of angles in a triangle is 180 degrees
  (A = C ∨ A = B) → -- Isosceles triangle condition
  A < 90 ∧ B < 90 ∧ C < 90 → -- Obtuse triangle condition (no right angle)
  A ≤ B ∧ A ≤ C → -- A is not the largest angle
  max B C = 140 := by -- The largest angle (either B or C) is 140 degrees
sorry

end NUMINAMATH_CALUDE_largest_angle_obtuse_isosceles_triangle_l3825_382526


namespace NUMINAMATH_CALUDE_rectangular_area_equation_l3825_382596

/-- Represents a rectangular area with length and width in meters -/
structure RectangularArea where
  length : ℝ
  width : ℝ

/-- The area of a rectangle is the product of its length and width -/
def area (r : RectangularArea) : ℝ := r.length * r.width

theorem rectangular_area_equation (x : ℝ) :
  let r : RectangularArea := { length := x, width := x - 6 }
  area r = 720 → x * (x - 6) = 720 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_area_equation_l3825_382596


namespace NUMINAMATH_CALUDE_same_shape_proof_l3825_382589

/-- A quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two quadratic functions have the same shape if the absolute values of their x² coefficients are equal -/
def same_shape (f g : QuadraticFunction) : Prop :=
  |f.a| = |g.a|

/-- The original function y = 5x² -/
def f : QuadraticFunction :=
  { a := 5, b := 0, c := 0 }

/-- The function y = -5x² + 2 -/
def g : QuadraticFunction :=
  { a := -5, b := 0, c := 2 }

theorem same_shape_proof : same_shape f g := by
  sorry

end NUMINAMATH_CALUDE_same_shape_proof_l3825_382589


namespace NUMINAMATH_CALUDE_third_trapezoid_largest_area_l3825_382556

-- Define the lengths of the segments
def a : ℝ := 2.12
def b : ℝ := 2.71
def c : ℝ := 3.53

-- Define the area calculation function for a trapezoid
def trapezoidArea (top bottom height : ℝ) : ℝ := (top + bottom) * height

-- Define the three possible trapezoids
def trapezoid1 : ℝ := trapezoidArea a c b
def trapezoid2 : ℝ := trapezoidArea b c a
def trapezoid3 : ℝ := trapezoidArea a b c

-- Theorem statement
theorem third_trapezoid_largest_area :
  trapezoid3 > trapezoid1 ∧ trapezoid3 > trapezoid2 :=
by sorry

end NUMINAMATH_CALUDE_third_trapezoid_largest_area_l3825_382556


namespace NUMINAMATH_CALUDE_student_arrangements_l3825_382598

def num_students : ℕ := 6

-- Condition 1: A not at head, B not at tail
def condition1 (arrangements : ℕ) : Prop :=
  arrangements = 504

-- Condition 2: A, B, and C not adjacent
def condition2 (arrangements : ℕ) : Prop :=
  arrangements = 144

-- Condition 3: A and B adjacent, C and D adjacent
def condition3 (arrangements : ℕ) : Prop :=
  arrangements = 96

-- Condition 4: Neither A nor B adjacent to C
def condition4 (arrangements : ℕ) : Prop :=
  arrangements = 288

theorem student_arrangements :
  ∃ (arr1 arr2 arr3 arr4 : ℕ),
    condition1 arr1 ∧
    condition2 arr2 ∧
    condition3 arr3 ∧
    condition4 arr4 :=
  by sorry

end NUMINAMATH_CALUDE_student_arrangements_l3825_382598


namespace NUMINAMATH_CALUDE_problem_solution_l3825_382528

def set_A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 1}

def set_B : Set ℝ := {x | (4 - x) / (x + 2) ≥ 0}

theorem problem_solution :
  (∀ a : ℝ, a = 2 → (set_A a)ᶜ ∩ set_B = {x | -2 < x ∧ x < 1}) ∧
  (∀ a : ℝ, set_A a ∪ set_B = set_B ↔ a < -2 ∨ (-1 < a ∧ a ≤ 3/2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3825_382528


namespace NUMINAMATH_CALUDE_right_triangle_tangent_sum_l3825_382587

theorem right_triangle_tangent_sum (α β : Real) (k : Real) : 
  α > 0 → β > 0 → α + β = π / 2 →
  (1 / 2) * Real.cos α * Real.cos β = k →
  Real.tan α + Real.tan β = 2 * k := by
sorry

end NUMINAMATH_CALUDE_right_triangle_tangent_sum_l3825_382587


namespace NUMINAMATH_CALUDE_expression_evaluation_l3825_382547

theorem expression_evaluation : 
  let a : ℤ := -2
  (a - 1)^2 - a*(a + 3) + 2*(a + 2)*(a - 2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3825_382547


namespace NUMINAMATH_CALUDE_temperature_conversion_l3825_382562

theorem temperature_conversion (t k r : ℝ) 
  (eq1 : t = 5/9 * (k - 32))
  (eq2 : r = 3*t)
  (eq3 : r = 150) : 
  k = 122 := by
sorry

end NUMINAMATH_CALUDE_temperature_conversion_l3825_382562


namespace NUMINAMATH_CALUDE_unique_triple_l3825_382579

/-- A function that checks if a number is divisible by any prime less than 2014 -/
def not_divisible_by_small_primes (n : ℕ) : Prop :=
  ∀ p, p < 2014 → Nat.Prime p → ¬(p ∣ n)

/-- The main theorem statement -/
theorem unique_triple : 
  ∃! (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ n : ℕ, n > 0 → not_divisible_by_small_primes n → 
      (n + c) ∣ (a^n + b^n + n)) ∧
    a = 1 ∧ b = 1 ∧ c = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_triple_l3825_382579


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3825_382582

theorem exponential_equation_solution :
  ∃! x : ℝ, 3^(2*x + 2) = (1 : ℝ) / 9 :=
by
  use -2
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3825_382582


namespace NUMINAMATH_CALUDE_multiply_specific_numbers_l3825_382529

theorem multiply_specific_numbers : 469138 * 9999 = 4690692862 := by
  sorry

end NUMINAMATH_CALUDE_multiply_specific_numbers_l3825_382529


namespace NUMINAMATH_CALUDE_function_determination_l3825_382509

theorem function_determination (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 1) :
  ∀ x, f x = x^2 - 2*x := by
sorry

end NUMINAMATH_CALUDE_function_determination_l3825_382509


namespace NUMINAMATH_CALUDE_rectangle_dimensions_area_l3825_382585

theorem rectangle_dimensions_area (x : ℝ) : 
  (x - 2) * (2 * x + 5) = 8 * x - 6 → x = 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_area_l3825_382585


namespace NUMINAMATH_CALUDE_rectangle_area_l3825_382565

/-- A rectangle with length thrice its breadth and perimeter 56 meters has an area of 147 square meters. -/
theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 56) : l * b = 147 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3825_382565


namespace NUMINAMATH_CALUDE_nested_sqrt_equality_l3825_382533

theorem nested_sqrt_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ 11) ^ (1/8) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_equality_l3825_382533


namespace NUMINAMATH_CALUDE_c_rent_share_l3825_382563

/-- Represents a person's pasture usage -/
structure PastureUsage where
  oxen : ℕ
  months : ℕ

/-- Calculates the total oxen-months for a given pasture usage -/
def oxenMonths (usage : PastureUsage) : ℕ :=
  usage.oxen * usage.months

/-- Calculates the share of rent for a given usage and total usage -/
def rentShare (usage : PastureUsage) (totalUsage : ℕ) (totalRent : ℕ) : ℚ :=
  (oxenMonths usage : ℚ) / (totalUsage : ℚ) * (totalRent : ℚ)

theorem c_rent_share :
  let a := PastureUsage.mk 10 7
  let b := PastureUsage.mk 12 5
  let c := PastureUsage.mk 15 3
  let totalRent := 175
  let totalUsage := oxenMonths a + oxenMonths b + oxenMonths c
  rentShare c totalUsage totalRent = 45 := by
  sorry

end NUMINAMATH_CALUDE_c_rent_share_l3825_382563


namespace NUMINAMATH_CALUDE_root_value_theorem_l3825_382550

theorem root_value_theorem (a : ℝ) (h : a^2 + 2*a - 1 = 0) : 2*a^2 + 4*a - 2024 = -2022 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l3825_382550


namespace NUMINAMATH_CALUDE_journey_distance_l3825_382515

/-- Proves that the total distance of a journey is 35 miles given specific conditions -/
theorem journey_distance (speed : ℝ) (time : ℝ) (total_portions : ℕ) (covered_portions : ℕ) :
  speed = 40 →
  time = 0.7 →
  total_portions = 5 →
  covered_portions = 4 →
  (speed * time) / covered_portions * total_portions = 35 :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l3825_382515


namespace NUMINAMATH_CALUDE_min_value_z_l3825_382508

theorem min_value_z (x y : ℝ) (h1 : x - y + 1 ≥ 0) (h2 : x + y - 1 ≥ 0) (h3 : x ≤ 3) :
  ∃ (z : ℝ), z = 2*x - 3*y ∧ z ≥ -6 ∧ (∀ (x' y' : ℝ), x' - y' + 1 ≥ 0 → x' + y' - 1 ≥ 0 → x' ≤ 3 → 2*x' - 3*y' ≥ z) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l3825_382508


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_xy_squared_l3825_382591

theorem factorization_cubic_minus_xy_squared (x y : ℝ) : 
  x^3 - x*y^2 = x*(x+y)*(x-y) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_xy_squared_l3825_382591


namespace NUMINAMATH_CALUDE_ordered_triples_satisfying_equation_l3825_382546

theorem ordered_triples_satisfying_equation :
  ∀ m n p : ℕ,
    m > 0 ∧ n > 0 ∧ Nat.Prime p ∧ p^n + 144 = m^2 →
    ((m = 13 ∧ n = 2 ∧ p = 5) ∨
     (m = 20 ∧ n = 8 ∧ p = 2) ∨
     (m = 15 ∧ n = 4 ∧ p = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ordered_triples_satisfying_equation_l3825_382546


namespace NUMINAMATH_CALUDE_average_rstp_l3825_382523

theorem average_rstp (r s t u : ℝ) (h : (5 / 2) * (r + s + t + u) = 20) :
  (r + s + t + u) / 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_average_rstp_l3825_382523


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3825_382588

/-- Given that x³ and y vary inversely, x and y are always positive, and y = 8 when x = 2,
    prove that x = 2/5 when y = 500. -/
theorem inverse_variation_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : ∃ (k : ℝ), ∀ (x y : ℝ), x^3 * y = k) 
  (h4 : 2^3 * 8 = (2 : ℝ)^3 * 8) : 
  (y = 500 → x = 2/5) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3825_382588


namespace NUMINAMATH_CALUDE_value_of_a_l3825_382580

def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}
def P (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

theorem value_of_a : 
  ∃ (a : ℝ), (U a).diff (P a) = {-1} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3825_382580


namespace NUMINAMATH_CALUDE_max_product_of_sum_constrained_naturals_l3825_382516

theorem max_product_of_sum_constrained_naturals
  (n k : ℕ) (h : k > 0) :
  let t : ℕ := n / k
  let r : ℕ := n % k
  ∃ (l : List ℕ),
    (l.length = k) ∧
    (l.sum = n) ∧
    (∀ (m : List ℕ), m.length = k → m.sum = n → l.prod ≥ m.prod) ∧
    (l.prod = (t + 1)^r * t^(k - r)) := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_sum_constrained_naturals_l3825_382516


namespace NUMINAMATH_CALUDE_max_sin_sum_60_degrees_l3825_382557

open Real

theorem max_sin_sum_60_degrees (x y : ℝ) : 
  0 < x → x < π/2 →
  0 < y → y < π/2 →
  x + y = π/3 →
  (∀ a b : ℝ, 0 < a → a < π/2 → 0 < b → b < π/2 → a + b = π/3 → sin a + sin b ≤ sin x + sin y) →
  sin x + sin y = 1 := by
sorry


end NUMINAMATH_CALUDE_max_sin_sum_60_degrees_l3825_382557


namespace NUMINAMATH_CALUDE_subtraction_problem_solution_l3825_382517

theorem subtraction_problem_solution :
  ∀ h t u : ℕ,
  h > u →
  h < 10 ∧ t < 10 ∧ u < 10 →
  (100 * h + 10 * t + u) - (100 * t + 10 * h + u) = 553 →
  h = 9 ∧ t = 4 ∧ u = 3 := by
sorry

end NUMINAMATH_CALUDE_subtraction_problem_solution_l3825_382517


namespace NUMINAMATH_CALUDE_boys_age_l3825_382584

theorem boys_age (current_age : ℕ) : 
  (current_age = 2 * (current_age - 5)) → current_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_age_l3825_382584


namespace NUMINAMATH_CALUDE_range_of_a_l3825_382575

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, x - a ≤ 0) → a ∈ Set.Ici 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3825_382575


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3825_382573

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  let a := -3
  let b := 0
  ∃ (x_max x_min : ℝ),
    x_max ∈ Set.Icc a b ∧
    x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 3 ∧
    f x_min = -17 :=
by sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3825_382573


namespace NUMINAMATH_CALUDE_square_of_105_l3825_382543

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end NUMINAMATH_CALUDE_square_of_105_l3825_382543
