import Mathlib

namespace NUMINAMATH_CALUDE_symposium_pair_selection_l365_36523

theorem symposium_pair_selection (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 2) :
  Nat.choose n k = 435 := by
  sorry

end NUMINAMATH_CALUDE_symposium_pair_selection_l365_36523


namespace NUMINAMATH_CALUDE_expression_value_l365_36539

theorem expression_value (x : ℝ) (h : 5 * x^2 - x - 1 = 0) :
  (3*x + 2) * (3*x - 2) + x * (x - 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l365_36539


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l365_36529

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def is_interesting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a ^ 2 ∧ 15 * n = b ^ 3

/-- The smallest interesting number is 1800. -/
theorem smallest_interesting_number : 
  (is_interesting 1800 ∧ ∀ m < 1800, ¬ is_interesting m) :=
sorry

end NUMINAMATH_CALUDE_smallest_interesting_number_l365_36529


namespace NUMINAMATH_CALUDE_number_puzzle_2016_l365_36515

theorem number_puzzle_2016 : ∃ (x y : ℕ), ∃ (z : ℕ), 
  x + y = 2016 ∧ 
  x = 10 * y + z ∧ 
  z < 10 ∧
  x = 1833 ∧ 
  y = 183 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_2016_l365_36515


namespace NUMINAMATH_CALUDE_quadratic_vertex_range_l365_36525

/-- A quadratic function of the form y = (a-1)x^2 + 3 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + 3

/-- The condition for the quadratic function to open downwards -/
def opens_downwards (a : ℝ) : Prop := a - 1 < 0

theorem quadratic_vertex_range (a : ℝ) :
  (∃ x, ∃ y, quadratic_function a x = y ∧ 
    ∀ z, quadratic_function a z ≤ y) →
  opens_downwards a →
  a < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_range_l365_36525


namespace NUMINAMATH_CALUDE_f_plus_f_neg_l365_36518

def f (x : ℝ) : ℝ := 5 * x^3

theorem f_plus_f_neg (x : ℝ) : f x + f (-x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_plus_f_neg_l365_36518


namespace NUMINAMATH_CALUDE_norma_laundry_problem_l365_36503

/-- The number of T-shirts Norma left in the washer -/
def t_shirts_left : ℕ := 9

/-- The number of sweaters Norma left in the washer -/
def sweaters_left : ℕ := 2 * t_shirts_left

/-- The total number of clothes Norma left in the washer -/
def total_left : ℕ := t_shirts_left + sweaters_left

/-- The number of sweaters Norma found when she returned -/
def sweaters_found : ℕ := 3

/-- The number of T-shirts Norma found when she returned -/
def t_shirts_found : ℕ := 3 * t_shirts_left

/-- The total number of clothes Norma found when she returned -/
def total_found : ℕ := sweaters_found + t_shirts_found

/-- The number of missing items -/
def missing_items : ℕ := total_left - total_found

theorem norma_laundry_problem : missing_items = 15 := by
  sorry

end NUMINAMATH_CALUDE_norma_laundry_problem_l365_36503


namespace NUMINAMATH_CALUDE_sum_m_n_equals_five_l365_36505

theorem sum_m_n_equals_five (m n : ℚ) (h : (m - 3) * Real.sqrt 5 + 2 - n = 0) : m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_five_l365_36505


namespace NUMINAMATH_CALUDE_product_xyz_l365_36559

theorem product_xyz (x y z k : ℝ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x)
  (h1 : x^3 + y^3 + k*(x^2 + y^2) = 2008)
  (h2 : y^3 + z^3 + k*(y^2 + z^2) = 2008)
  (h3 : z^3 + x^3 + k*(z^2 + x^2) = 2008) :
  x * y * z = -1004 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l365_36559


namespace NUMINAMATH_CALUDE_power_sum_sequence_l365_36555

theorem power_sum_sequence (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h_rec : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^10 + b^10 = 123 := by
sorry

end NUMINAMATH_CALUDE_power_sum_sequence_l365_36555


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l365_36508

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (1 + i) / (1 + i^3) = i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l365_36508


namespace NUMINAMATH_CALUDE_pipe_fill_rate_l365_36570

theorem pipe_fill_rate (slow_time fast_time combined_time : ℝ) 
  (h1 : slow_time = 160)
  (h2 : combined_time = 40)
  (h3 : slow_time > 0)
  (h4 : fast_time > 0)
  (h5 : combined_time > 0)
  (h6 : 1 / combined_time = 1 / fast_time + 1 / slow_time) :
  fast_time = slow_time / 3 :=
sorry

end NUMINAMATH_CALUDE_pipe_fill_rate_l365_36570


namespace NUMINAMATH_CALUDE_hcf_36_84_l365_36573

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_36_84_l365_36573


namespace NUMINAMATH_CALUDE_sum_of_possible_x_values_l365_36567

/-- An isosceles triangle with two angles of 60° and x° -/
structure IsoscelesTriangle60X where
  /-- The measure of angle x in degrees -/
  x : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True
  /-- One angle of the triangle is 60° -/
  has60Angle : True
  /-- Another angle of the triangle is x° -/
  hasXAngle : True
  /-- The sum of angles in a triangle is 180° -/
  angleSum : True

/-- The sum of all possible values of x in an isosceles triangle with angles 60° and x° is 180° -/
theorem sum_of_possible_x_values (t : IsoscelesTriangle60X) : 
  ∃ (x₁ x₂ x₃ : ℝ), (x₁ + x₂ + x₃ = 180 ∧ 
    (t.x = x₁ ∨ t.x = x₂ ∨ t.x = x₃)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_values_l365_36567


namespace NUMINAMATH_CALUDE_binary_to_base4_conversion_l365_36547

/-- Converts a binary number represented as a list of bits to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation. -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary representation of 1101101001₂ -/
def binary_num : List Bool :=
  [true, true, false, true, true, false, true, false, false, true]

/-- The base 4 representation of 13201₄ -/
def base4_num : List ℕ := [1, 3, 2, 0, 1]

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal binary_num) = base4_num := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base4_conversion_l365_36547


namespace NUMINAMATH_CALUDE_sector_angle_in_unit_circle_l365_36585

theorem sector_angle_in_unit_circle (sector_area : ℝ) (central_angle : ℝ) : 
  sector_area = 1 → central_angle = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_in_unit_circle_l365_36585


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l365_36580

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l365_36580


namespace NUMINAMATH_CALUDE_min_value_sin_product_l365_36599

theorem min_value_sin_product (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h_pos : θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧ θ₄ > 0) 
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) : 
  (2 * Real.sin θ₁ ^ 2 + 1 / Real.sin θ₁ ^ 2) * 
  (2 * Real.sin θ₂ ^ 2 + 1 / Real.sin θ₂ ^ 2) * 
  (2 * Real.sin θ₃ ^ 2 + 1 / Real.sin θ₃ ^ 2) * 
  (2 * Real.sin θ₄ ^ 2 + 1 / Real.sin θ₄ ^ 2) ≥ 81 := by
  sorry

#check min_value_sin_product

end NUMINAMATH_CALUDE_min_value_sin_product_l365_36599


namespace NUMINAMATH_CALUDE_johns_donation_l365_36513

/-- Given 6 initial contributions and a new contribution that increases the average by 50% to $75, prove that the new contribution is $225. -/
theorem johns_donation (initial_contributions : ℕ) (new_average : ℚ) : 
  initial_contributions = 6 ∧ 
  new_average = 75 ∧ 
  new_average = (3/2) * (300 / initial_contributions) →
  ∃ (johns_contribution : ℚ), 
    johns_contribution = 225 ∧
    new_average = (300 + johns_contribution) / (initial_contributions + 1) :=
by sorry

end NUMINAMATH_CALUDE_johns_donation_l365_36513


namespace NUMINAMATH_CALUDE_rational_equation_solution_l365_36592

theorem rational_equation_solution :
  ∃ (x : ℝ), (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21) ∧ x = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l365_36592


namespace NUMINAMATH_CALUDE_symmetric_line_proof_l365_36579

/-- Given two lines in a 2D plane, this function returns the equation of the line symmetric to the first line with respect to the second line. -/
def symmetricLine (l1 l2 : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The line x - y - 2 = 0 -/
def line1 : ℝ → ℝ → Prop :=
  λ x y ↦ x - y - 2 = 0

/-- The line x - 2y + 2 = 0 -/
def line2 : ℝ → ℝ → Prop :=
  λ x y ↦ x - 2*y + 2 = 0

/-- The line x - 7y + 22 = 0 -/
def resultLine : ℝ → ℝ → Prop :=
  λ x y ↦ x - 7*y + 22 = 0

theorem symmetric_line_proof : 
  symmetricLine line1 line2 = resultLine := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_proof_l365_36579


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l365_36551

theorem cyclic_sum_inequality (a b c : ℝ) : 
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧ 
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l365_36551


namespace NUMINAMATH_CALUDE_cos_seventeen_pi_sixths_l365_36527

theorem cos_seventeen_pi_sixths : 
  Real.cos (17 * π / 6) = - Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seventeen_pi_sixths_l365_36527


namespace NUMINAMATH_CALUDE_min_platforms_proof_l365_36583

/-- The minimum number of platforms required to transport all granite slabs -/
def min_platforms : ℕ := 40

/-- The number of 7-ton granite slabs -/
def slabs_7ton : ℕ := 120

/-- The number of 9-ton granite slabs -/
def slabs_9ton : ℕ := 80

/-- The maximum weight a platform can carry (in tons) -/
def max_platform_weight : ℕ := 40

/-- The weight of a 7-ton slab -/
def weight_7ton : ℕ := 7

/-- The weight of a 9-ton slab -/
def weight_9ton : ℕ := 9

theorem min_platforms_proof :
  min_platforms * 3 ≥ slabs_7ton ∧
  min_platforms * 2 ≥ slabs_9ton ∧
  3 * weight_7ton + 2 * weight_9ton ≤ max_platform_weight ∧
  ∀ n : ℕ, n < min_platforms →
    n * 3 < slabs_7ton ∨ n * 2 < slabs_9ton :=
by sorry

end NUMINAMATH_CALUDE_min_platforms_proof_l365_36583


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l365_36540

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 900 is 40√2 -/
theorem ellipse_foci_distance :
  let a : ℝ := Real.sqrt 100
  let b : ℝ := Real.sqrt 900
  let c : ℝ := Real.sqrt (b^2 - a^2)
  2 * c = 40 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l365_36540


namespace NUMINAMATH_CALUDE_problem_solution_l365_36591

theorem problem_solution (x : ℝ) (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 12) :
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 200/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l365_36591


namespace NUMINAMATH_CALUDE_tan_plus_cot_equals_three_l365_36543

theorem tan_plus_cot_equals_three (α : Real) (h : Real.sin (2 * α) = 2/3) :
  Real.tan α + 1 / Real.tan α = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_cot_equals_three_l365_36543


namespace NUMINAMATH_CALUDE_garden_length_l365_36550

/-- A rectangular garden with length twice the width and 180 yards of fencing. -/
structure Garden where
  width : ℝ
  length : ℝ
  fencing : ℝ
  twice_width : length = 2 * width
  total_fencing : 2 * length + 2 * width = fencing

/-- The length of a garden with 180 yards of fencing is 60 yards. -/
theorem garden_length (g : Garden) (h : g.fencing = 180) : g.length = 60 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l365_36550


namespace NUMINAMATH_CALUDE_min_sum_positive_integers_l365_36595

theorem min_sum_positive_integers (a b x y z : ℕ+) 
  (h : (3 : ℕ) * a.val = (7 : ℕ) * b.val ∧ 
       (7 : ℕ) * b.val = (5 : ℕ) * x.val ∧ 
       (5 : ℕ) * x.val = (4 : ℕ) * y.val ∧ 
       (4 : ℕ) * y.val = (6 : ℕ) * z.val) : 
  a.val + b.val + x.val + y.val + z.val ≥ 459 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_positive_integers_l365_36595


namespace NUMINAMATH_CALUDE_max_pens_173_l365_36545

/-- Represents a package of pens with its size and cost -/
structure PenPackage where
  size : Nat
  cost : Nat

/-- Finds the maximum number of pens that can be purchased with a given budget -/
def maxPens (budget : Nat) (packages : List PenPackage) : Nat :=
  sorry

/-- The specific problem setup -/
def problemSetup : List PenPackage := [
  ⟨12, 10⟩,
  ⟨20, 15⟩
]

/-- The theorem stating that the maximum number of pens purchasable with $173 is 224 -/
theorem max_pens_173 : maxPens 173 problemSetup = 224 := by
  sorry

end NUMINAMATH_CALUDE_max_pens_173_l365_36545


namespace NUMINAMATH_CALUDE_sqrt_inequality_l365_36549

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) + Real.sqrt (x - 2) > Real.sqrt (x - 4) + Real.sqrt (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l365_36549


namespace NUMINAMATH_CALUDE_sample_size_theorem_l365_36533

/-- Represents the types of products produced by the factory -/
inductive ProductType
  | A
  | B
  | C

/-- Represents the quantity ratio of products -/
def quantity_ratio : ProductType → ℕ
  | ProductType.A => 2
  | ProductType.B => 3
  | ProductType.C => 5

/-- Calculates the total ratio sum -/
def total_ratio : ℕ := quantity_ratio ProductType.A + quantity_ratio ProductType.B + quantity_ratio ProductType.C

/-- Represents the number of Type B products in the sample -/
def type_b_sample : ℕ := 24

/-- Theorem: If 24 units of Type B are drawn in a stratified random sample 
    from a production with ratio 2:3:5, then the total sample size is 80 -/
theorem sample_size_theorem : 
  (type_b_sample * total_ratio) / quantity_ratio ProductType.B = 80 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_theorem_l365_36533


namespace NUMINAMATH_CALUDE_map_scale_conversion_l365_36582

theorem map_scale_conversion (map_cm : ℝ) (real_km : ℝ) : 
  (20 : ℝ) * real_km = 100 * map_cm → 25 * real_km = 125 * map_cm := by
  sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l365_36582


namespace NUMINAMATH_CALUDE_smallest_odd_abundant_number_l365_36507

def is_abundant (n : ℕ) : Prop :=
  n < (Finset.sum (Finset.filter (λ x => x < n ∧ n % x = 0) (Finset.range n)) id)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem smallest_odd_abundant_number :
  (∀ n : ℕ, n < 945 → ¬(is_odd n ∧ is_abundant n ∧ is_composite n)) ∧
  (is_odd 945 ∧ is_abundant 945 ∧ is_composite 945) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_abundant_number_l365_36507


namespace NUMINAMATH_CALUDE_chinese_dream_essay_contest_l365_36528

theorem chinese_dream_essay_contest (total : ℕ) (seventh : ℕ) (eighth : ℕ) :
  total = 118 →
  seventh = eighth / 2 - 2 →
  total = seventh + eighth →
  seventh = 38 := by
sorry

end NUMINAMATH_CALUDE_chinese_dream_essay_contest_l365_36528


namespace NUMINAMATH_CALUDE_fraction_problem_l365_36596

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x = 140) (h2 : 0.65 * x = f * x - 21) : f = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l365_36596


namespace NUMINAMATH_CALUDE_cos_225_degrees_l365_36594

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l365_36594


namespace NUMINAMATH_CALUDE_simplify_expression_l365_36587

theorem simplify_expression (p : ℝ) :
  ((6*p + 2) - 3*p*3)*4 + (5 - 2/4)*(8*p - 12) = 24*p - 46 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l365_36587


namespace NUMINAMATH_CALUDE_find_set_B_l365_36524

-- Define the universal set U (we'll use ℤ for integers)
def U : Set ℤ := sorry

-- Define set A
def A : Set ℤ := {0, 2, 4}

-- Define the complement of A with respect to U
def C_UA : Set ℤ := {-1, 1}

-- Define the complement of B with respect to U
def C_UB : Set ℤ := {-1, 0, 2}

-- Define set B
def B : Set ℤ := {1, 4}

-- Theorem to prove
theorem find_set_B : B = {1, 4} := by sorry

end NUMINAMATH_CALUDE_find_set_B_l365_36524


namespace NUMINAMATH_CALUDE_inequality_and_fraction_properties_l365_36501

theorem inequality_and_fraction_properties (a : ℝ) (h : 2 < a ∧ a < 4) : 
  (3 * a - 2 > 2 * a ∧ 4 * (a - 1) < 3 * a) ∧ 
  (a - (a + 4) / (a + 1) = (a^2 - 4) / (a + 1)) ∧ 
  ((a^2 - 4) / (a + 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_fraction_properties_l365_36501


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l365_36588

/-- Given an ellipse and a hyperbola with the same foci, prove that the parameter a of the ellipse is 4 -/
theorem ellipse_hyperbola_same_foci (a : ℝ) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1 → a > 0) → -- Ellipse equation condition
  (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1) → -- Hyperbola equation
  (∃ c : ℝ, c^2 = 7 ∧ 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1 → x^2 + y^2 = a^2 + c^2) ∧ 
    (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1 → x^2 - y^2 = 4 + c^2)) → -- Same foci condition
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l365_36588


namespace NUMINAMATH_CALUDE_three_boys_three_girls_arrangements_l365_36554

/-- The number of possible arrangements for 3 boys and 3 girls in an alternating pattern -/
def alternating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  2 * (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating that the number of arrangements for 3 boys and 3 girls is 72 -/
theorem three_boys_three_girls_arrangements :
  alternating_arrangements 3 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_three_boys_three_girls_arrangements_l365_36554


namespace NUMINAMATH_CALUDE_david_cell_phone_cost_l365_36520

/-- Calculates the total cost of a cell phone plan. -/
def cell_phone_plan_cost (base_fee : ℝ) (text_cost : ℝ) (extra_minute_cost : ℝ) 
                         (standard_hours : ℝ) (texts_sent : ℕ) (hours_used : ℝ) : ℝ :=
  base_fee + 
  (text_cost * texts_sent) + 
  (extra_minute_cost * (hours_used - standard_hours) * 60)

/-- Theorem stating that David's cell phone plan cost is $54. -/
theorem david_cell_phone_cost : 
  cell_phone_plan_cost 30 0.1 0.15 20 150 21 = 54 := by
  sorry


end NUMINAMATH_CALUDE_david_cell_phone_cost_l365_36520


namespace NUMINAMATH_CALUDE_infinite_a_without_solution_l365_36564

/-- τ(n) denotes the number of positive divisors of the positive integer n -/
def tau (n : ℕ+) : ℕ := sorry

/-- The set of positive integers a for which τ(an) = n has no solution -/
def A : Set ℕ+ := {a : ℕ+ | ∀ n : ℕ+, tau (a * n) ≠ n}

/-- There exist infinitely many positive integers a such that τ(an) = n has no solution -/
theorem infinite_a_without_solution : Set.Infinite A := by sorry

end NUMINAMATH_CALUDE_infinite_a_without_solution_l365_36564


namespace NUMINAMATH_CALUDE_detergent_for_clothes_l365_36597

-- Define the detergent usage rate
def detergent_per_pound : ℝ := 2

-- Define the amount of clothes to be washed
def clothes_weight : ℝ := 9

-- Theorem to prove
theorem detergent_for_clothes : detergent_per_pound * clothes_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_detergent_for_clothes_l365_36597


namespace NUMINAMATH_CALUDE_exactly_four_points_C_l365_36558

/-- Given two points A and B in a plane that are 12 units apart, this function
    returns the number of points C such that the perimeter of triangle ABC is 60 units
    and the area of triangle ABC is 72 square units. -/
def count_points_C (A B : ℝ × ℝ) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 4 points C satisfying the conditions. -/
theorem exactly_four_points_C (A B : ℝ × ℝ) (h : dist A B = 12) :
  count_points_C A B = 4 :=
sorry

end NUMINAMATH_CALUDE_exactly_four_points_C_l365_36558


namespace NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l365_36535

/-- Represents the number of apples of each color in the basket -/
structure Basket :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Represents the number of apples taken from the basket -/
structure TakenApples :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Checks if the condition for stopping is met -/
def stoppingCondition (taken : TakenApples) : Prop :=
  taken.green < taken.yellow ∧ taken.yellow < taken.red

/-- The initial basket of apples -/
def initialBasket : Basket :=
  { green := 10, yellow := 13, red := 18 }

/-- Theorem for the maximum number of yellow apples that can be taken -/
theorem max_yellow_apples :
  ∃ (taken : TakenApples),
    taken.yellow = 13 ∧
    taken.yellow ≤ taken.red ∧
    taken.green ≤ taken.yellow ∧
    ∀ (other : TakenApples),
      other.yellow > 13 →
      other.yellow > other.red ∨ other.green > other.yellow :=
sorry

/-- Theorem for the maximum number of apples that can be taken in total -/
theorem max_total_apples :
  ∃ (taken : TakenApples),
    taken.green + taken.yellow + taken.red = 39 ∧
    ¬(stoppingCondition taken) ∧
    ∀ (other : TakenApples),
      other.green + other.yellow + other.red > 39 →
      stoppingCondition other :=
sorry

end NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l365_36535


namespace NUMINAMATH_CALUDE_range_of_a_l365_36538

theorem range_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_one : a^2 + b^2 + c^2 = 1) :
  -Real.sqrt 6 / 3 ≤ a ∧ a ≤ Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l365_36538


namespace NUMINAMATH_CALUDE_university_visit_probability_l365_36530

theorem university_visit_probability : 
  let n : ℕ := 4  -- number of students
  let k : ℕ := 2  -- number of universities
  let p : ℚ := (k^n - 2) / k^n  -- probability formula
  p = 7/8 := by sorry

end NUMINAMATH_CALUDE_university_visit_probability_l365_36530


namespace NUMINAMATH_CALUDE_theresas_sons_l365_36526

theorem theresas_sons (meatballs_per_plate : ℕ) (fraction_eaten : ℚ) (meatballs_left : ℕ) :
  meatballs_per_plate = 3 →
  fraction_eaten = 2/3 →
  meatballs_left = 3 →
  (meatballs_left : ℚ) / (1 - fraction_eaten) = 9 :=
by sorry

end NUMINAMATH_CALUDE_theresas_sons_l365_36526


namespace NUMINAMATH_CALUDE_sum_of_squares_difference_l365_36514

theorem sum_of_squares_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : Real.sqrt x / Real.sqrt y - Real.sqrt y / Real.sqrt x = 7/12)
  (h2 : x - y = 7) :
  x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_difference_l365_36514


namespace NUMINAMATH_CALUDE_line_b_production_l365_36519

/-- Represents the production of cement bags by three production lines -/
structure CementProduction where
  total : ℕ
  lineA : ℕ
  lineB : ℕ
  lineC : ℕ
  sum_eq_total : lineA + lineB + lineC = total
  arithmetic_sequence : lineA - lineB = lineB - lineC

/-- Theorem stating that under given conditions, production line B produces 6500 bags -/
theorem line_b_production (prod : CementProduction) (h : prod.total = 19500) : 
  prod.lineB = 6500 := by
  sorry

end NUMINAMATH_CALUDE_line_b_production_l365_36519


namespace NUMINAMATH_CALUDE_min_value_expression_l365_36561

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 4) * (b^2 + 4*b + 4) * (c^2 + 4*c + 4) / (a*b*c) ≥ 64 ∧
  (∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    (a₀^2 + 4*a₀ + 4) * (b₀^2 + 4*b₀ + 4) * (c₀^2 + 4*c₀ + 4) / (a₀*b₀*c₀) = 64) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l365_36561


namespace NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_l365_36572

/-- The least number of digits in the repeating block of the decimal expansion of 7/13 -/
def repeating_block_length : ℕ := 6

/-- Theorem stating that the least number of digits in the repeating block 
    of the decimal expansion of 7/13 is equal to repeating_block_length -/
theorem seven_thirteenths_repeating_block : 
  (Nat.lcm 13 10 : ℕ).factorization 2 + (Nat.lcm 13 10 : ℕ).factorization 5 = repeating_block_length :=
sorry

end NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_l365_36572


namespace NUMINAMATH_CALUDE_min_sums_theorem_l365_36560

def min_sums_for_unique_determination (n : ℕ) : ℕ :=
  Nat.choose (n - 1) 2 + 1

theorem min_sums_theorem (n : ℕ) (h : n ≥ 3) :
  ∀ (a : Fin n → ℝ),
  ∀ (k : ℕ),
  (k < min_sums_for_unique_determination n →
    ∃ (b₁ b₂ : Fin n → ℝ),
      b₁ ≠ b₂ ∧
      (∀ (i j : Fin n), i.val > j.val →
        (∃ (S : Finset (Fin n × Fin n)),
          S.card = k ∧
          (∀ (p : Fin n × Fin n), p ∈ S → p.1.val > p.2.val) ∧
          (∀ (p : Fin n × Fin n), p ∈ S → a (p.1) + a (p.2) = b₁ (p.1) + b₁ (p.2)) ∧
          (∀ (p : Fin n × Fin n), p ∈ S → a (p.1) + a (p.2) = b₂ (p.1) + b₂ (p.2))))) ∧
  (k ≥ min_sums_for_unique_determination n →
    ∀ (b : Fin n → ℝ),
    (∀ (S : Finset (Fin n × Fin n)),
      S.card = k →
      (∀ (p : Fin n × Fin n), p ∈ S → p.1.val > p.2.val) →
      (∃! (c : Fin n → ℝ), ∀ (p : Fin n × Fin n), p ∈ S → c (p.1) + c (p.2) = b (p.1) + b (p.2))))
  := by sorry

end NUMINAMATH_CALUDE_min_sums_theorem_l365_36560


namespace NUMINAMATH_CALUDE_jellybean_count_l365_36590

/-- The number of blue jellybeans in the jar -/
def blue_jellybeans : ℕ := 14

/-- The number of purple jellybeans in the jar -/
def purple_jellybeans : ℕ := 26

/-- The number of orange jellybeans in the jar -/
def orange_jellybeans : ℕ := 40

/-- The number of red jellybeans in the jar -/
def red_jellybeans : ℕ := 120

/-- The total number of jellybeans in the jar -/
def total_jellybeans : ℕ := blue_jellybeans + purple_jellybeans + orange_jellybeans + red_jellybeans

theorem jellybean_count : total_jellybeans = 200 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l365_36590


namespace NUMINAMATH_CALUDE_max_sum_of_factors_48_l365_36598

theorem max_sum_of_factors_48 : 
  ∃ (a b : ℕ), a * b = 48 ∧ a + b = 49 ∧ ∀ (x y : ℕ), x * y = 48 → x + y ≤ 49 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_48_l365_36598


namespace NUMINAMATH_CALUDE_at_most_two_special_numbers_l365_36506

/-- A positive integer n is special if it can be expressed as 2^a * 3^b for some nonnegative integers a and b. -/
def is_special (n : ℕ+) : Prop :=
  ∃ a b : ℕ, n = 2^a * 3^b

/-- For any positive integer k, there are at most two special numbers in the range (k^2, k^2 + 2k + 1). -/
theorem at_most_two_special_numbers (k : ℕ+) :
  ∃ n₁ n₂ : ℕ+, ∀ n : ℕ+,
    k^2 < n ∧ n < k^2 + 2*k + 1 ∧ is_special n →
    n = n₁ ∨ n = n₂ :=
  sorry

end NUMINAMATH_CALUDE_at_most_two_special_numbers_l365_36506


namespace NUMINAMATH_CALUDE_percentage_difference_l365_36556

theorem percentage_difference (x : ℝ) : x = 30 → 0.9 * 40 = 0.8 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l365_36556


namespace NUMINAMATH_CALUDE_hyperbola_sum_a_h_l365_36504

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  -- Asymptote equations
  asymptote1 : ℝ → ℝ
  asymptote2 : ℝ → ℝ
  -- Point the hyperbola passes through
  point : ℝ × ℝ
  -- Standard form parameters
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  -- Conditions
  asymptote1_eq : ∀ x, asymptote1 x = 3 * x + 2
  asymptote2_eq : ∀ x, asymptote2 x = -3 * x + 8
  point_on_hyperbola : point = (1, 6)
  standard_form : ∀ x y, (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1
  positive_params : a > 0 ∧ b > 0

/-- Theorem: For the given hyperbola, a + h = 2 -/
theorem hyperbola_sum_a_h (hyp : Hyperbola) : hyp.a + hyp.h = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_a_h_l365_36504


namespace NUMINAMATH_CALUDE_star_equation_solution_l365_36553

-- Define the star operation
noncomputable def star (x y : ℝ) : ℝ :=
  x + Real.sqrt (y + Real.sqrt (y + Real.sqrt y))

-- State the theorem
theorem star_equation_solution :
  ∃ h : ℝ, star 3 h = 8 ∧ h = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l365_36553


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l365_36517

theorem fixed_point_parabola :
  ∀ (k : ℝ), 3 * (5 : ℝ)^2 + k * 5 - 5 * k = 75 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l365_36517


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_intersections_l365_36541

/-- The number of unit cubes a space diagonal passes through in a rectangular prism -/
def spaceDiagonalIntersections (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd c a + Nat.gcd a (Nat.gcd b c)

/-- Theorem: For a 150 × 324 × 375 rectangular prism, the space diagonal passes through 768 unit cubes -/
theorem rectangular_prism_diagonal_intersections :
  spaceDiagonalIntersections 150 324 375 = 768 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_intersections_l365_36541


namespace NUMINAMATH_CALUDE_monotone_function_a_bound_l365_36586

/-- Given a function f(x) = x² + a/x that is monotonically increasing on [2, +∞),
    prove that a ≤ 16 -/
theorem monotone_function_a_bound (a : ℝ) :
  (∀ x ≥ 2, Monotone (fun x => x^2 + a/x)) →
  a ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_monotone_function_a_bound_l365_36586


namespace NUMINAMATH_CALUDE_kangaroo_equality_days_l365_36531

/-- The number of days required for Bert to have the same number of kangaroos as Kameron -/
def days_to_equal_kangaroos (kameron_kangaroos bert_kangaroos bert_daily_purchase : ℕ) : ℕ :=
  (kameron_kangaroos - bert_kangaroos) / bert_daily_purchase

/-- Theorem stating that it takes 40 days for Bert to have the same number of kangaroos as Kameron -/
theorem kangaroo_equality_days :
  days_to_equal_kangaroos 100 20 2 = 40 := by
  sorry

#eval days_to_equal_kangaroos 100 20 2

end NUMINAMATH_CALUDE_kangaroo_equality_days_l365_36531


namespace NUMINAMATH_CALUDE_combination_sum_l365_36581

theorem combination_sum : Nat.choose 5 2 + Nat.choose 5 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_l365_36581


namespace NUMINAMATH_CALUDE_selina_leaves_with_30_l365_36575

/-- The amount of money Selina leaves the store with after selling and buying clothes -/
def selina_final_money (pants_price shorts_price shirts_price : ℕ) 
  (pants_sold shorts_sold shirts_sold : ℕ) 
  (shirts_bought new_shirt_price : ℕ) : ℕ :=
  pants_price * pants_sold + shorts_price * shorts_sold + shirts_price * shirts_sold - 
  shirts_bought * new_shirt_price

/-- Theorem stating that Selina leaves the store with $30 -/
theorem selina_leaves_with_30 : 
  selina_final_money 5 3 4 3 5 5 2 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_selina_leaves_with_30_l365_36575


namespace NUMINAMATH_CALUDE_fraction_addition_l365_36574

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l365_36574


namespace NUMINAMATH_CALUDE_trigonometric_identity_l365_36584

theorem trigonometric_identity (α : ℝ) : 
  (Real.cos (2 * α - π / 2) + Real.sin (3 * π - 4 * α) - Real.cos (5 * π / 2 + 6 * α)) / 
  (4 * Real.sin (5 * π - 3 * α) * Real.cos (α - 2 * π)) = Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l365_36584


namespace NUMINAMATH_CALUDE_max_value_of_a_l365_36577

theorem max_value_of_a (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  a ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l365_36577


namespace NUMINAMATH_CALUDE_water_duration_village_water_duration_l365_36576

/-- Calculates how long water will last in a village given specific conditions. -/
theorem water_duration (water_per_person : ℝ) (small_households : ℕ) (large_households : ℕ) 
  (small_household_size : ℕ) (large_household_size : ℕ) (total_water : ℝ) : ℝ :=
  let water_usage_per_month := 
    (small_households * small_household_size * water_per_person) + 
    (large_households * large_household_size * water_per_person)
  total_water / water_usage_per_month

/-- Proves that the water lasts approximately 4.31 months under given conditions. -/
theorem village_water_duration : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |water_duration 20 7 3 2 5 2500 - 4.31| < ε :=
sorry

end NUMINAMATH_CALUDE_water_duration_village_water_duration_l365_36576


namespace NUMINAMATH_CALUDE_walt_investment_l365_36500

theorem walt_investment (amount_at_8_percent : ℝ) (total_interest : ℝ) : 
  amount_at_8_percent = 4000 →
  total_interest = 770 →
  ∃ (amount_at_9_percent : ℝ),
    0.09 * amount_at_9_percent + 0.08 * amount_at_8_percent = total_interest ∧
    amount_at_9_percent + amount_at_8_percent = 9000 :=
by sorry

end NUMINAMATH_CALUDE_walt_investment_l365_36500


namespace NUMINAMATH_CALUDE_exam_questions_l365_36568

theorem exam_questions (correct_score : ℕ) (wrong_penalty : ℕ) (total_score : ℕ) (correct_answers : ℕ) : ℕ :=
  let total_questions := correct_answers + (correct_score * correct_answers - total_score)
  50

#check exam_questions 4 1 130 36

end NUMINAMATH_CALUDE_exam_questions_l365_36568


namespace NUMINAMATH_CALUDE_rational_solutions_count_l365_36512

theorem rational_solutions_count (p : ℕ) (hp : Prime p) :
  let f : ℚ → ℚ := λ x => x^4 + (2 - p : ℚ)*x^3 + (2 - 2*p : ℚ)*x^2 + (1 - 2*p : ℚ)*x - p
  (∃ (s : Finset ℚ), s.card = 2 ∧ (∀ x ∈ s, f x = 0) ∧ (∀ x, f x = 0 → x ∈ s)) := by
  sorry

end NUMINAMATH_CALUDE_rational_solutions_count_l365_36512


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l365_36571

/-- Given two circles in the xy-plane:
    Circle1: x^2 + y^2 - x + y - 2 = 0
    Circle2: x^2 + y^2 = 5
    This theorem states that the line x - y - 3 = 0 passes through their intersection points. -/
theorem intersection_line_of_circles (x y : ℝ) :
  (x^2 + y^2 - x + y - 2 = 0 ∧ x^2 + y^2 = 5) → (x - y - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l365_36571


namespace NUMINAMATH_CALUDE_product_of_complex_polars_l365_36544

/-- Represents a complex number in polar form -/
structure ComplexPolar where
  magnitude : ℝ
  angle : ℝ

/-- Multiplication of complex numbers in polar form -/
def mul_complex_polar (z₁ z₂ : ComplexPolar) : ComplexPolar :=
  { magnitude := z₁.magnitude * z₂.magnitude,
    angle := z₁.angle + z₂.angle }

theorem product_of_complex_polars :
  let z₁ : ComplexPolar := { magnitude := 5, angle := 30 }
  let z₂ : ComplexPolar := { magnitude := 4, angle := 45 }
  let product := mul_complex_polar z₁ z₂
  product.magnitude = 20 ∧ product.angle = 75 := by sorry

end NUMINAMATH_CALUDE_product_of_complex_polars_l365_36544


namespace NUMINAMATH_CALUDE_factor_in_range_l365_36593

theorem factor_in_range : ∃ m : ℕ, 
  (201212200619 : ℕ) % m = 0 ∧ 
  (6 * 10^9 : ℕ) < m ∧ 
  m < (13 * 10^9 : ℕ) / 2 ∧
  m = 6490716149 := by
sorry

end NUMINAMATH_CALUDE_factor_in_range_l365_36593


namespace NUMINAMATH_CALUDE_pear_problem_l365_36589

theorem pear_problem (alyssa_pears nancy_pears carlos_pears given_away : ℕ) 
  (h1 : alyssa_pears = 42)
  (h2 : nancy_pears = 17)
  (h3 : carlos_pears = 25)
  (h4 : given_away = 5) :
  alyssa_pears + nancy_pears + carlos_pears - 3 * given_away = 69 := by
  sorry

end NUMINAMATH_CALUDE_pear_problem_l365_36589


namespace NUMINAMATH_CALUDE_line_not_intersecting_segment_l365_36542

/-- Given points P and Q, and a line l that does not intersect line segment PQ,
    prove that the parameter m in the line equation satisfies m < -2/3 or m > 1/2 -/
theorem line_not_intersecting_segment (m : ℝ) :
  let P : ℝ × ℝ := (-1, 1)
  let Q : ℝ × ℝ := (2, 2)
  let l := {(x, y) : ℝ × ℝ | x + m * y + m = 0}
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • P + t • Q ∉ l) →
  m < -2/3 ∨ m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_line_not_intersecting_segment_l365_36542


namespace NUMINAMATH_CALUDE_four_birdhouses_built_l365_36537

/-- The number of birdhouses that can be built with a given budget -/
def num_birdhouses (plank_cost nail_cost planks_per_house nails_per_house budget : ℚ) : ℚ :=
  budget / (plank_cost * planks_per_house + nail_cost * nails_per_house)

/-- Theorem stating that 4 birdhouses can be built with $88 given the specified costs and materials -/
theorem four_birdhouses_built :
  num_birdhouses 3 0.05 7 20 88 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_birdhouses_built_l365_36537


namespace NUMINAMATH_CALUDE_kiera_muffins_count_l365_36548

/-- Represents the number of items in an order -/
structure Order :=
  (muffins : ℕ)
  (fruitCups : ℕ)

/-- Calculates the cost of an order given the prices -/
def orderCost (order : Order) (muffinPrice fruitCupPrice : ℕ) : ℕ :=
  order.muffins * muffinPrice + order.fruitCups * fruitCupPrice

theorem kiera_muffins_count 
  (muffinPrice fruitCupPrice : ℕ)
  (francis : Order)
  (kiera : Order)
  (h1 : muffinPrice = 2)
  (h2 : fruitCupPrice = 3)
  (h3 : francis.muffins = 2)
  (h4 : francis.fruitCups = 2)
  (h5 : kiera.fruitCups = 1)
  (h6 : orderCost francis muffinPrice fruitCupPrice + 
        orderCost kiera muffinPrice fruitCupPrice = 17) :
  kiera.muffins = 2 := by
sorry

end NUMINAMATH_CALUDE_kiera_muffins_count_l365_36548


namespace NUMINAMATH_CALUDE_common_root_of_polynomials_l365_36546

/-- Given three polynomials P, Q, and R, prove that 7 is their common root. -/
theorem common_root_of_polynomials :
  let P : ℝ → ℝ := λ x => x^3 + 41*x^2 - 49*x - 2009
  let Q : ℝ → ℝ := λ x => x^3 + 5*x^2 - 49*x - 245
  let R : ℝ → ℝ := λ x => x^3 + 39*x^2 - 117*x - 1435
  P 7 = 0 ∧ Q 7 = 0 ∧ R 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_root_of_polynomials_l365_36546


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l365_36552

/-- Given a nonreal cube root of unity ω, prove that (ω - 2ω^2 + 2)^4 + (2 + 2ω - ω^2)^4 = -257 -/
theorem cube_root_unity_sum (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (ω - 2*ω^2 + 2)^4 + (2 + 2*ω - ω^2)^4 = -257 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l365_36552


namespace NUMINAMATH_CALUDE_approximation_accuracy_l365_36502

/-- The actual number of students --/
def actual_number : ℕ := 76500

/-- The approximate number in scientific notation --/
def approximate_number : ℝ := 7.7 * 10^4

/-- Definition of accuracy to thousands place --/
def accurate_to_thousands (x y : ℝ) : Prop :=
  ∃ k : ℤ, x = k * 1000 ∧ |y - x| < 500

/-- Theorem stating the approximation is accurate to the thousands place --/
theorem approximation_accuracy :
  accurate_to_thousands (↑actual_number) approximate_number :=
sorry

end NUMINAMATH_CALUDE_approximation_accuracy_l365_36502


namespace NUMINAMATH_CALUDE_no_integer_solutions_l365_36569

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 19 * x^3 - 84 * y^2 = 1984 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l365_36569


namespace NUMINAMATH_CALUDE_game_strategy_sum_final_result_l365_36511

theorem game_strategy_sum (R S : ℕ) : R - S = 1010 :=
  by
  have h1 : R = (1010 : ℕ) * 2022 / 2 := by sorry
  have h2 : S = (1010 : ℕ) * 2020 / 2 := by sorry
  sorry

theorem final_result : (R - S) / 10 = 101 :=
  by
  have h : R - S = 1010 := game_strategy_sum R S
  sorry

end NUMINAMATH_CALUDE_game_strategy_sum_final_result_l365_36511


namespace NUMINAMATH_CALUDE_prob_ace_king_same_suit_standard_deck_l365_36534

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (aces_per_suit : Nat)
  (kings_per_suit : Nat)

/-- Probability of drawing an Ace then a King of the same suit -/
def prob_ace_then_king_same_suit (d : Deck) : ℚ :=
  (d.aces_per_suit : ℚ) / d.total_cards * (d.kings_per_suit : ℚ) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing an Ace then a King of the same suit in a standard deck -/
theorem prob_ace_king_same_suit_standard_deck :
  let standard_deck : Deck :=
    { total_cards := 52
    , suits := 4
    , cards_per_suit := 13
    , aces_per_suit := 1
    , kings_per_suit := 1
    }
  prob_ace_then_king_same_suit standard_deck = 1 / 663 := by
  sorry


end NUMINAMATH_CALUDE_prob_ace_king_same_suit_standard_deck_l365_36534


namespace NUMINAMATH_CALUDE_gdp_growth_problem_l365_36521

/-- Calculates the final GDP after compound growth -/
def finalGDP (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate) ^ years

/-- The problem statement -/
theorem gdp_growth_problem :
  let initial_gdp : ℝ := 9593.3
  let growth_rate : ℝ := 0.073
  let years : ℕ := 4
  let final_gdp := finalGDP initial_gdp growth_rate years
  ∃ ε > 0, |final_gdp - 127165| < ε :=
sorry

end NUMINAMATH_CALUDE_gdp_growth_problem_l365_36521


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_reciprocal_l365_36562

theorem imaginary_part_of_complex_reciprocal (z : ℂ) (h : z = 1 + 2*I) : 
  Complex.im (z⁻¹) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_reciprocal_l365_36562


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l365_36532

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (p^3 - 25*p^2 + 90*p - 73 = 0) →
  (q^3 - 25*q^2 + 90*q - 73 = 0) →
  (r^3 - 25*r^2 + 90*r - 73 = 0) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 25*s^2 + 90*s - 73) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 256 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l365_36532


namespace NUMINAMATH_CALUDE_subway_passenger_decrease_l365_36566

theorem subway_passenger_decrease (initial : ℕ) (got_off : ℕ) (got_on : ℕ)
  (h1 : initial = 35)
  (h2 : got_off = 18)
  (h3 : got_on = 15) :
  initial - (initial - got_off + got_on) = 3 :=
by sorry

end NUMINAMATH_CALUDE_subway_passenger_decrease_l365_36566


namespace NUMINAMATH_CALUDE_probability_divisor_of_12_l365_36509

/-- A fair 6-sided die -/
def Die : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of divisors of 12 that appear on the die -/
def DivisorsOf12OnDie : Finset ℕ := {1, 2, 3, 4, 6}

/-- The probability of an event on a fair die -/
def probability (event : Finset ℕ) : ℚ :=
  (event ∩ Die).card / Die.card

theorem probability_divisor_of_12 :
  probability DivisorsOf12OnDie = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisor_of_12_l365_36509


namespace NUMINAMATH_CALUDE_triangle_perimeter_bounds_l365_36578

theorem triangle_perimeter_bounds (a b c : ℝ) (h : a * b + b * c + c * a = 12) :
  let k := a + b + c
  6 ≤ k ∧ k ≤ 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bounds_l365_36578


namespace NUMINAMATH_CALUDE_trig_computation_l365_36510

theorem trig_computation : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_computation_l365_36510


namespace NUMINAMATH_CALUDE_square_difference_equals_810_l365_36565

theorem square_difference_equals_810 : (27 + 15)^2 - (27^2 + 15^2) = 810 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_810_l365_36565


namespace NUMINAMATH_CALUDE_base9_to_base10_conversion_l365_36522

/-- Converts a base-9 number represented as a list of digits to its base-10 equivalent -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The base-9 representation of the number -/
def base9Number : List Nat := [7, 4, 8, 2]

theorem base9_to_base10_conversion :
  base9ToBase10 base9Number = 2149 := by
  sorry

end NUMINAMATH_CALUDE_base9_to_base10_conversion_l365_36522


namespace NUMINAMATH_CALUDE_barrel_division_exists_l365_36536

/-- Represents the fill state of a barrel -/
inductive BarrelState
  | Empty
  | Half
  | Full

/-- Represents a distribution of barrels to an heir -/
structure Distribution where
  empty : Nat
  half : Nat
  full : Nat

/-- Calculates the total wine in a distribution -/
def wineAmount (d : Distribution) : Nat :=
  d.full * 2 + d.half

/-- Checks if a distribution is valid (8 barrels total) -/
def isValidDistribution (d : Distribution) : Prop :=
  d.empty + d.half + d.full = 8

/-- Represents a complete division of barrels among three heirs -/
structure BarrelDivision where
  heir1 : Distribution
  heir2 : Distribution
  heir3 : Distribution

/-- Checks if a barrel division is valid -/
def isValidDivision (div : BarrelDivision) : Prop :=
  isValidDistribution div.heir1 ∧
  isValidDistribution div.heir2 ∧
  isValidDistribution div.heir3 ∧
  div.heir1.empty + div.heir2.empty + div.heir3.empty = 8 ∧
  div.heir1.half + div.heir2.half + div.heir3.half = 8 ∧
  div.heir1.full + div.heir2.full + div.heir3.full = 8 ∧
  wineAmount div.heir1 = wineAmount div.heir2 ∧
  wineAmount div.heir2 = wineAmount div.heir3

theorem barrel_division_exists : ∃ (div : BarrelDivision), isValidDivision div := by
  sorry

end NUMINAMATH_CALUDE_barrel_division_exists_l365_36536


namespace NUMINAMATH_CALUDE_augmented_matrix_solution_l365_36563

theorem augmented_matrix_solution (c₁ c₂ : ℝ) : 
  (∃ (x y : ℝ), 2 * x + 3 * y = c₁ ∧ y = c₂ ∧ x = 3 ∧ y = 5) → c₁ - c₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_augmented_matrix_solution_l365_36563


namespace NUMINAMATH_CALUDE_group_size_proof_l365_36557

/-- The number of people in a group where:
    1) Replacing a 60 kg person with a 110 kg person increases the total weight by 50 kg.
    2) The average weight increase is 5 kg.
-/
def group_size : ℕ :=
  10

theorem group_size_proof :
  (group_size : ℝ) * 5 = 110 - 60 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l365_36557


namespace NUMINAMATH_CALUDE_zoo_animal_count_l365_36516

/-- Calculates the total number of animals in a zoo with specific enclosure arrangements. -/
def total_animals_in_zoo : ℕ :=
  let tiger_enclosures : ℕ := 4
  let zebra_enclosures : ℕ := tiger_enclosures * 2
  let elephant_enclosures : ℕ := zebra_enclosures + 1
  let giraffe_enclosures : ℕ := elephant_enclosures * 3
  let rhino_enclosures : ℕ := 4

  let tigers : ℕ := tiger_enclosures * 4
  let zebras : ℕ := zebra_enclosures * 10
  let elephants : ℕ := elephant_enclosures * 3
  let giraffes : ℕ := giraffe_enclosures * 2
  let rhinos : ℕ := rhino_enclosures * 1

  tigers + zebras + elephants + giraffes + rhinos

/-- Theorem stating that the total number of animals in the zoo is 181. -/
theorem zoo_animal_count : total_animals_in_zoo = 181 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l365_36516
