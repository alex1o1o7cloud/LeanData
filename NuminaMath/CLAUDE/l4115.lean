import Mathlib

namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l4115_411585

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l4115_411585


namespace NUMINAMATH_CALUDE_equation_solution_l4115_411562

theorem equation_solution : ∀ x y : ℕ, 
  (x - 1) / (1 + (x - 1) * y) + (y - 1) / (2 * y - 1) = x / (x + 1) → 
  x = 2 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4115_411562


namespace NUMINAMATH_CALUDE_constant_term_value_l4115_411533

theorem constant_term_value (x y z : ℤ) (k : ℤ) : 
  4 * x + y + z = 80 → 
  3 * x + y - z = 20 → 
  x = 20 → 
  2 * x - y - z = k → 
  k = 40 := by sorry

end NUMINAMATH_CALUDE_constant_term_value_l4115_411533


namespace NUMINAMATH_CALUDE_wooden_statue_cost_l4115_411566

/-- The cost of a wooden statue given Theodore's production and earnings. -/
theorem wooden_statue_cost :
  let stone_statues : ℕ := 10
  let wooden_statues : ℕ := 20
  let stone_cost : ℚ := 20
  let tax_rate : ℚ := 1/10
  let total_earnings : ℚ := 270
  ∃ (wooden_cost : ℚ),
    (1 - tax_rate) * (stone_statues * stone_cost + wooden_statues * wooden_cost) = total_earnings ∧
    wooden_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_wooden_statue_cost_l4115_411566


namespace NUMINAMATH_CALUDE_some_number_value_l4115_411582

theorem some_number_value (x : ℝ) : 3034 - (1002 / x) = 3029 → x = 200.4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l4115_411582


namespace NUMINAMATH_CALUDE_minimize_S_l4115_411575

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := 2 * n^2 - 30 * n

/-- n minimizes S if S(n) is less than or equal to S(k) for all natural numbers k -/
def Minimizes (n : ℕ) : Prop :=
  ∀ k : ℕ, S n ≤ S k

theorem minimize_S :
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ Minimizes n :=
sorry

end NUMINAMATH_CALUDE_minimize_S_l4115_411575


namespace NUMINAMATH_CALUDE_penelope_savings_l4115_411530

/-- The amount of money Penelope saves daily, in dollars. -/
def daily_savings : ℕ := 24

/-- The number of days in a year (assuming it's not a leap year). -/
def days_in_year : ℕ := 365

/-- The total amount Penelope saves in a year. -/
def total_savings : ℕ := daily_savings * days_in_year

/-- Theorem: Penelope's total savings after one year is $8,760. -/
theorem penelope_savings : total_savings = 8760 := by
  sorry

end NUMINAMATH_CALUDE_penelope_savings_l4115_411530


namespace NUMINAMATH_CALUDE_three_intersections_iff_zero_l4115_411544

/-- The number of distinct intersection points between the curves x^2 - y^2 = a^2 and (x-1)^2 + y^2 = 1 -/
def intersection_count (a : ℝ) : ℕ :=
  sorry

/-- The condition for exactly three distinct intersection points -/
def has_three_intersections (a : ℝ) : Prop :=
  intersection_count a = 3

theorem three_intersections_iff_zero (a : ℝ) :
  has_three_intersections a ↔ a = 0 :=
sorry

end NUMINAMATH_CALUDE_three_intersections_iff_zero_l4115_411544


namespace NUMINAMATH_CALUDE_exists_hyperbola_segment_with_midpoint_l4115_411567

/-- The hyperbola equation -/
def on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

/-- The midpoint of two points -/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

theorem exists_hyperbola_segment_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    on_hyperbola x₁ y₁ ∧
    on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
  sorry

end NUMINAMATH_CALUDE_exists_hyperbola_segment_with_midpoint_l4115_411567


namespace NUMINAMATH_CALUDE_tilde_r_24_l4115_411583

def tilde_r (n : ℕ) : ℕ :=
  (Nat.factors n).sum + (Nat.factors n).toFinset.card

theorem tilde_r_24 : tilde_r 24 = 11 := by sorry

end NUMINAMATH_CALUDE_tilde_r_24_l4115_411583


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l4115_411501

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l4115_411501


namespace NUMINAMATH_CALUDE_jason_egg_consumption_l4115_411556

/-- The number of eggs Jason consumes in two weeks -/
def eggs_consumed_in_two_weeks : ℕ :=
  let eggs_per_omelet : ℕ := 3
  let days_in_two_weeks : ℕ := 14
  eggs_per_omelet * days_in_two_weeks

/-- Theorem stating that Jason consumes 42 eggs in two weeks -/
theorem jason_egg_consumption :
  eggs_consumed_in_two_weeks = 42 := by
  sorry

end NUMINAMATH_CALUDE_jason_egg_consumption_l4115_411556


namespace NUMINAMATH_CALUDE_log_equation_solution_l4115_411502

theorem log_equation_solution : 
  ∃ y : ℝ, (Real.log y - 3 * Real.log 5 = -3) ∧ (y = 0.125) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4115_411502


namespace NUMINAMATH_CALUDE_point_B_coordinates_l4115_411597

-- Define the point A and vector a
def A : ℝ × ℝ := (2, 4)
def a : ℝ × ℝ := (3, 4)

-- Define the relation between AB and a
def AB_relation (B : ℝ × ℝ) : Prop :=
  (B.1 - A.1, B.2 - A.2) = (2 * a.1, 2 * a.2)

-- Theorem stating that B has coordinates (8, 12)
theorem point_B_coordinates :
  ∃ B : ℝ × ℝ, AB_relation B ∧ B = (8, 12) := by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l4115_411597


namespace NUMINAMATH_CALUDE_smallest_c_value_l4115_411594

theorem smallest_c_value (a b c : ℤ) : 
  a < b → b < c → 
  (c - b = b - a) →  -- arithmetic progression
  (b * b = a * c) →  -- geometric progression
  b = 3 * a → 
  (∀ x : ℤ, (x < a ∨ x = a) → 
    ¬(x < 3*x → 3*x < 9*x → 
      (9*x - 3*x = 3*x - x) → 
      ((3*x) * (3*x) = x * (9*x)))) → 
  c = 9 * a :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l4115_411594


namespace NUMINAMATH_CALUDE_fixed_fee_is_9_39_l4115_411535

/-- Represents a cloud storage service billing system -/
structure CloudStorageBilling where
  fixed_fee : ℝ
  feb_usage_fee : ℝ
  feb_total : ℝ
  mar_total : ℝ

/-- The cloud storage billing satisfies the given conditions -/
def satisfies_conditions (bill : CloudStorageBilling) : Prop :=
  bill.feb_total = bill.fixed_fee + bill.feb_usage_fee ∧
  bill.mar_total = bill.fixed_fee + 3 * bill.feb_usage_fee ∧
  bill.feb_total = 15.80 ∧
  bill.mar_total = 28.62

/-- The fixed monthly fee is 9.39 given the conditions -/
theorem fixed_fee_is_9_39 (bill : CloudStorageBilling) 
  (h : satisfies_conditions bill) : bill.fixed_fee = 9.39 := by
  sorry

end NUMINAMATH_CALUDE_fixed_fee_is_9_39_l4115_411535


namespace NUMINAMATH_CALUDE_prime_diff_cubes_sum_squares_l4115_411538

theorem prime_diff_cubes_sum_squares (p : ℕ) (a b : ℕ) :
  Prime p → p = a^3 - b^3 → ∃ (c d : ℤ), p = c^2 + 3 * d^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_diff_cubes_sum_squares_l4115_411538


namespace NUMINAMATH_CALUDE_problem_1_l4115_411558

theorem problem_1 : (1) - 4^2 / (-32) * (2/3)^2 = 2/9 := by sorry

end NUMINAMATH_CALUDE_problem_1_l4115_411558


namespace NUMINAMATH_CALUDE_sum_4_inclusive_numbers_eq_1883_l4115_411598

/-- Returns true if the number contains the digit 4 -/
def contains4 (n : ℕ) : Bool :=
  n.repr.contains '4'

/-- Returns true if the number is 4-inclusive (multiple of 4 or contains 4) -/
def is4Inclusive (n : ℕ) : Bool :=
  n % 4 = 0 || contains4 n

/-- The sum of all 4-inclusive numbers in the range [0, 100] -/
def sum4InclusiveNumbers : ℕ :=
  (List.range 101).filter is4Inclusive |>.sum

theorem sum_4_inclusive_numbers_eq_1883 : sum4InclusiveNumbers = 1883 := by
  sorry

end NUMINAMATH_CALUDE_sum_4_inclusive_numbers_eq_1883_l4115_411598


namespace NUMINAMATH_CALUDE_constrained_line_generates_surface_l4115_411591

/-- A line parallel to the plane y=z, intersecting two parabolas -/
structure ConstrainedLine where
  /-- The line is parallel to the plane y=z -/
  parallel_to_yz : ℝ → ℝ → ℝ → Prop
  /-- The line intersects the parabola 2x=y², z=0 -/
  meets_parabola1 : ℝ → ℝ → ℝ → Prop
  /-- The line intersects the parabola 3x=z², y=0 -/
  meets_parabola2 : ℝ → ℝ → ℝ → Prop

/-- The surface generated by the constrained line -/
def generated_surface (x y z : ℝ) : Prop :=
  x = (y - z) * (y / 2 - z / 3)

/-- Theorem stating that the constrained line generates the specified surface -/
theorem constrained_line_generates_surface (L : ConstrainedLine) :
  ∀ x y z, L.parallel_to_yz x y z → L.meets_parabola1 x y z → L.meets_parabola2 x y z →
  generated_surface x y z :=
sorry

end NUMINAMATH_CALUDE_constrained_line_generates_surface_l4115_411591


namespace NUMINAMATH_CALUDE_extreme_value_condition_negative_interval_condition_l4115_411511

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Part I
theorem extreme_value_condition (a b : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = 10 →
  a = 4 ∧ b = -11 :=
sorry

-- Part II
theorem negative_interval_condition (b : ℝ) :
  (∀ (x : ℝ), x ∈ Set.Icc 1 2 → f (-1) b x < 0) →
  b < -5/2 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_negative_interval_condition_l4115_411511


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_plus_five_l4115_411563

theorem sum_of_three_numbers_plus_five (a b c : ℤ) 
  (h1 : a + b = 31) 
  (h2 : b + c = 48) 
  (h3 : c + a = 55) : 
  a + b + c + 5 = 72 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_plus_five_l4115_411563


namespace NUMINAMATH_CALUDE_quadratic_root_and_coefficient_l4115_411565

theorem quadratic_root_and_coefficient (m : ℝ) :
  (∃ x, x^2 + m*x + 2 = 0 ∧ x = -2) →
  (∃ y, y^2 + m*y + 2 = 0 ∧ y = -1) ∧ m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_and_coefficient_l4115_411565


namespace NUMINAMATH_CALUDE_max_banner_area_l4115_411503

/-- Represents the cost per meter of length -/
def cost_length : ℕ := 330

/-- Represents the cost per meter of width -/
def cost_width : ℕ := 450

/-- Represents the total budget in dollars -/
def budget : ℕ := 10000

/-- Proves that the maximum area of a rectangular banner with integer dimensions
    is 165 square meters, given the budget and cost constraints. -/
theorem max_banner_area :
  ∀ x y : ℕ,
    (cost_length * x + cost_width * y ≤ budget) →
    (x * y ≤ 165) :=
by sorry

end NUMINAMATH_CALUDE_max_banner_area_l4115_411503


namespace NUMINAMATH_CALUDE_mono_decreasing_g_l4115_411560

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being monotonically increasing on [1, 2]
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x ≤ y → f x ≤ f y

-- Define the function g(x) = f(1-x)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (1 - x)

-- State the theorem
theorem mono_decreasing_g (h : MonoIncreasing f) :
  ∀ x y, x ∈ Set.Icc (-1) 0 → y ∈ Set.Icc (-1) 0 → x ≤ y → g f y ≤ g f x :=
sorry

end NUMINAMATH_CALUDE_mono_decreasing_g_l4115_411560


namespace NUMINAMATH_CALUDE_angle_range_given_sine_l4115_411599

theorem angle_range_given_sine (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : Real.sin α = 0.58) :
  Real.pi / 6 < α ∧ α < Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_range_given_sine_l4115_411599


namespace NUMINAMATH_CALUDE_shirt_ratio_l4115_411581

theorem shirt_ratio (brian_shirts andrew_shirts steven_shirts : ℕ) :
  brian_shirts = 3 →
  andrew_shirts = 6 * brian_shirts →
  steven_shirts = 72 →
  steven_shirts / andrew_shirts = 4 :=
by sorry

end NUMINAMATH_CALUDE_shirt_ratio_l4115_411581


namespace NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l4115_411521

theorem three_fourths_to_fifth_power :
  (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l4115_411521


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_in_range_l4115_411507

theorem inequality_holds_iff_p_in_range :
  ∀ p : ℝ, p ≥ 0 →
  (∀ q : ℝ, q > 0 → (4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔
  p ∈ Set.Ici 0 ∩ Set.Iio 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_in_range_l4115_411507


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l4115_411508

-- Problem 1
theorem problem_1 : (12 : ℤ) - (-18) + (-7) - 15 = 8 := by sorry

-- Problem 2
theorem problem_2 : (-81 : ℚ) / (9/4) * (4/9) / (-16) = 1 := by sorry

-- Problem 3
theorem problem_3 : ((1/3 : ℚ) - 5/6 + 7/9) * (-18) = -5 := by sorry

-- Problem 4
theorem problem_4 : -(1 : ℚ)^4 - (1/5) * (2 - (-3))^2 = -6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l4115_411508


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_l4115_411531

theorem polynomial_nonnegative (x : ℝ) : x^4 - x^3 + 3*x^2 - 2*x + 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_l4115_411531


namespace NUMINAMATH_CALUDE_cubic_extreme_values_l4115_411588

/-- Given a cubic function f(x) = x^3 - px^2 - qx that passes through (1,0),
    prove that its maximum value is 4/27 and its minimum value is 0. -/
theorem cubic_extreme_values (p q : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 - p*x^2 - q*x
  (f 1 = 0) →
  (∃ x, f x = 4/27) ∧ (∀ y, f y ≤ 4/27) ∧ (∃ z, f z = 0) ∧ (∀ w, f w ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_cubic_extreme_values_l4115_411588


namespace NUMINAMATH_CALUDE_other_number_proof_l4115_411522

/-- Prove that given two positive integers, 24 and x, if their HCF (h) is 17 and their LCM (l) is 312, then x = 221. -/
theorem other_number_proof (x : ℕ) (h l : ℕ) : 
  x > 0 ∧ h > 0 ∧ l > 0 ∧ 
  h = Nat.gcd 24 x ∧ 
  l = Nat.lcm 24 x ∧ 
  h = 17 ∧ 
  l = 312 → 
  x = 221 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l4115_411522


namespace NUMINAMATH_CALUDE_power_inequality_l4115_411555

theorem power_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^6 + b^6 ≥ a*b*(a^4 + b^4) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l4115_411555


namespace NUMINAMATH_CALUDE_pairing_theorem_l4115_411589

def is_valid_pairing (n : ℕ) (pairing : List (ℕ × ℕ)) : Prop :=
  pairing.length = n ∧
  pairing.all (λ p => p.1 ≤ 2*n ∧ p.2 ≤ 2*n) ∧
  (List.range (2*n)).all (λ i => pairing.any (λ p => p.1 = i+1 ∨ p.2 = i+1))

def pairing_product (pairing : List (ℕ × ℕ)) : ℕ :=
  pairing.foldl (λ acc p => acc * (p.1 + p.2)) 1

theorem pairing_theorem (n : ℕ) (h : n > 1) :
  ∃ pairing : List (ℕ × ℕ), is_valid_pairing n pairing ∧
  ∃ m : ℕ, pairing_product pairing = m * m :=
sorry

end NUMINAMATH_CALUDE_pairing_theorem_l4115_411589


namespace NUMINAMATH_CALUDE_initial_cherries_l4115_411519

theorem initial_cherries (eaten : ℕ) (left : ℕ) (h1 : eaten = 25) (h2 : left = 42) :
  eaten + left = 67 := by
  sorry

end NUMINAMATH_CALUDE_initial_cherries_l4115_411519


namespace NUMINAMATH_CALUDE_work_completion_equivalence_l4115_411595

/-- The number of days needed for the first group to complete the work -/
def days_first_group : ℕ := 96

/-- The number of men in the second group -/
def men_second_group : ℕ := 40

/-- The number of days needed for the second group to complete the work -/
def days_second_group : ℕ := 60

/-- The number of men in the first group -/
def men_first_group : ℕ := 25

theorem work_completion_equivalence :
  men_first_group * days_first_group = men_second_group * days_second_group :=
by sorry

#check work_completion_equivalence

end NUMINAMATH_CALUDE_work_completion_equivalence_l4115_411595


namespace NUMINAMATH_CALUDE_wedge_volume_l4115_411524

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h θ : ℝ) (hd : d = 20) (hh : h = 20) (hθ : θ = 30 * π / 180) :
  let r := d / 2
  let cylinder_volume := π * r^2 * h
  let wedge_volume := (θ / (2 * π)) * cylinder_volume
  wedge_volume = 250 * π := by sorry

end NUMINAMATH_CALUDE_wedge_volume_l4115_411524


namespace NUMINAMATH_CALUDE_unique_bijective_function_satisfying_equation_l4115_411526

-- Define the property that a function f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = f x + y

-- State the theorem
theorem unique_bijective_function_satisfying_equation :
  ∃! f : ℝ → ℝ, Function.Bijective f ∧ SatisfiesEquation f ∧ (∀ x : ℝ, f x = x) :=
sorry

end NUMINAMATH_CALUDE_unique_bijective_function_satisfying_equation_l4115_411526


namespace NUMINAMATH_CALUDE_problem_triangle_integer_segments_l4115_411590

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments
    from vertex E to points on the hypotenuse DF -/
def countIntegerSegments (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle in the problem -/
def problemTriangle : RightTriangle :=
  { de := 24, ef := 25 }

/-- The main theorem stating that the number of distinct integer lengths
    of line segments from E to DF in the problem triangle is 14 -/
theorem problem_triangle_integer_segments :
  countIntegerSegments problemTriangle = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_triangle_integer_segments_l4115_411590


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l4115_411548

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 99) : 
  x^2 + y^2 = 5745 / 169 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l4115_411548


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_greater_than_neg_four_l4115_411577

def A (a : ℝ) : Set ℝ := {x | x^2 + (a + 2) * x + 1 = 0}
def B : Set ℝ := {x | x > 0}

theorem intersection_empty_implies_a_greater_than_neg_four (a : ℝ) :
  A a ∩ B = ∅ → a > -4 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_greater_than_neg_four_l4115_411577


namespace NUMINAMATH_CALUDE_flamingo_tail_feathers_l4115_411506

/-- The number of tail feathers per flamingo given the conditions for making feather boas --/
theorem flamingo_tail_feathers 
  (num_boas : ℕ) 
  (feathers_per_boa : ℕ) 
  (num_flamingoes : ℕ) 
  (safe_pluck_percentage : ℚ) : ℕ :=
  sorry

#check flamingo_tail_feathers 12 200 480 (1/4) = 20

end NUMINAMATH_CALUDE_flamingo_tail_feathers_l4115_411506


namespace NUMINAMATH_CALUDE_competition_configs_l4115_411587

/-- Represents a valid competition configuration -/
structure CompetitionConfig where
  n : ℕ
  k : ℕ
  h_n_ge_2 : n ≥ 2
  h_k_ge_1 : k ≥ 1
  h_total_score : k * (n * (n + 1) / 2) = 26 * n

/-- The set of all valid competition configurations -/
def ValidConfigs : Set CompetitionConfig := {c | c.n ≥ 2 ∧ c.k ≥ 1 ∧ c.k * (c.n * (c.n + 1) / 2) = 26 * c.n}

/-- The theorem stating the possible values of (n, k) -/
theorem competition_configs : ValidConfigs = {⟨25, 2, by norm_num, by norm_num, by norm_num⟩, 
                                              ⟨12, 4, by norm_num, by norm_num, by norm_num⟩, 
                                              ⟨3, 13, by norm_num, by norm_num, by norm_num⟩} := by
  sorry

end NUMINAMATH_CALUDE_competition_configs_l4115_411587


namespace NUMINAMATH_CALUDE_lcm_12_18_25_l4115_411557

theorem lcm_12_18_25 : Nat.lcm (Nat.lcm 12 18) 25 = 900 := by sorry

end NUMINAMATH_CALUDE_lcm_12_18_25_l4115_411557


namespace NUMINAMATH_CALUDE_alphabet_letter_count_l4115_411536

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (straight_only : ℕ) :
  total = 40 →
  both = 8 →
  straight_only = 24 →
  total = both + straight_only + (total - (both + straight_only)) →
  (total - (both + straight_only)) = 8 :=
by sorry

end NUMINAMATH_CALUDE_alphabet_letter_count_l4115_411536


namespace NUMINAMATH_CALUDE_lisa_dvd_rental_l4115_411534

theorem lisa_dvd_rental (total_spent : ℚ) (cost_per_dvd : ℚ) (h1 : total_spent = 4.8) (h2 : cost_per_dvd = 1.2) :
  total_spent / cost_per_dvd = 4 := by
  sorry

end NUMINAMATH_CALUDE_lisa_dvd_rental_l4115_411534


namespace NUMINAMATH_CALUDE_circle_intersection_distance_l4115_411553

-- Define the circles and their properties
variable (r R : ℝ)
variable (d : ℝ)

-- Hypotheses
variable (h1 : r > 0)
variable (h2 : R > 0)
variable (h3 : r < R)
variable (h4 : d > 0)

-- Define the intersection property
variable (intersection : ∃ (x : ℝ × ℝ), (x.1^2 + x.2^2 = r^2) ∧ ((x.1 - d)^2 + x.2^2 = R^2))

-- Theorem statement
theorem circle_intersection_distance : R - r < d ∧ d < r + R := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_distance_l4115_411553


namespace NUMINAMATH_CALUDE_exam_probabilities_l4115_411527

/-- Represents the probabilities of scoring in different ranges in a math exam --/
structure ExamProbabilities where
  above90 : ℝ
  between80and89 : ℝ
  between70and79 : ℝ
  between60and69 : ℝ

/-- Calculates the probability of scoring 80 or above --/
def prob_80_or_above (p : ExamProbabilities) : ℝ :=
  p.above90 + p.between80and89

/-- Calculates the probability of failing the exam (scoring below 60) --/
def prob_fail (p : ExamProbabilities) : ℝ :=
  1 - (p.above90 + p.between80and89 + p.between70and79 + p.between60and69)

/-- Theorem stating the probabilities of scoring 80 or above and failing the exam --/
theorem exam_probabilities 
  (p : ExamProbabilities) 
  (h1 : p.above90 = 0.18) 
  (h2 : p.between80and89 = 0.51) 
  (h3 : p.between70and79 = 0.15) 
  (h4 : p.between60and69 = 0.09) : 
  prob_80_or_above p = 0.69 ∧ prob_fail p = 0.07 := by
  sorry

end NUMINAMATH_CALUDE_exam_probabilities_l4115_411527


namespace NUMINAMATH_CALUDE_areas_theorem_l4115_411578

-- Define the areas A, B, and C
def A : ℝ := sorry
def B : ℝ := sorry
def C : ℝ := sorry

-- State the theorem
theorem areas_theorem :
  -- Condition for A: square with diagonal 2√2
  (∃ (s : ℝ), s * s = A ∧ s * Real.sqrt 2 = 2 * Real.sqrt 2) →
  -- Condition for B: rectangle with given vertices
  (∃ (w h : ℝ), w * h = B ∧ w = 4 ∧ h = 2) →
  -- Condition for C: triangle formed by axes and line y = -x/2 + 2
  (∃ (base height : ℝ), (1/2) * base * height = C ∧ base = 4 ∧ height = 2) →
  -- Conclusion
  A = 4 ∧ B = 8 ∧ C = 4 := by
sorry


end NUMINAMATH_CALUDE_areas_theorem_l4115_411578


namespace NUMINAMATH_CALUDE_lattice_point_proximity_probability_l4115_411549

theorem lattice_point_proximity_probability (r : ℝ) : 
  (r > 0) → 
  (π * r^2 = 1/3) → 
  (∃ (p : ℝ × ℝ), p.1 ≥ 0 ∧ p.1 ≤ 1 ∧ p.2 ≥ 0 ∧ p.2 ≤ 1 ∧ 
    ((p.1^2 + p.2^2 ≤ r^2) ∨ 
     ((1 - p.1)^2 + p.2^2 ≤ r^2) ∨ 
     (p.1^2 + (1 - p.2)^2 ≤ r^2) ∨ 
     ((1 - p.1)^2 + (1 - p.2)^2 ≤ r^2))) = 
  (r = Real.sqrt (1 / (3 * π))) :=
sorry

end NUMINAMATH_CALUDE_lattice_point_proximity_probability_l4115_411549


namespace NUMINAMATH_CALUDE_sum_in_base6_l4115_411551

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (a b : ℕ) : ℕ := a * 6 + b

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 36
  let remainder := n % 36
  let tens := remainder / 6
  let ones := remainder % 6
  (hundreds, tens, ones)

theorem sum_in_base6 :
  let a := base6ToBase10 3 5
  let b := base6ToBase10 2 5
  let sum := a + b
  let (h, t, o) := base10ToBase6 sum
  h = 1 ∧ t = 0 ∧ o = 4 := by sorry

end NUMINAMATH_CALUDE_sum_in_base6_l4115_411551


namespace NUMINAMATH_CALUDE_kindergarten_count_l4115_411547

/-- Given the ratio of boys to girls and girls to teachers in a kindergarten,
    along with the number of boys, prove the total number of students and teachers. -/
theorem kindergarten_count (boys girls teachers : ℕ) : 
  (boys : ℚ) / girls = 3 / 4 →
  (girls : ℚ) / teachers = 5 / 2 →
  boys = 18 →
  boys + girls + teachers = 53 := by
sorry

end NUMINAMATH_CALUDE_kindergarten_count_l4115_411547


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l4115_411564

theorem remainder_of_large_number : 2345678901 % 101 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l4115_411564


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l4115_411515

theorem largest_solution_of_equation (x : ℚ) :
  (8 * (9 * x^2 + 10 * x + 15) = x * (9 * x - 45)) →
  x ≤ -5/3 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l4115_411515


namespace NUMINAMATH_CALUDE_correct_result_l4115_411518

/-- Represents a five-digit number -/
structure FiveDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9
  h4 : d ≥ 0 ∧ d ≤ 9
  h5 : e ≥ 0 ∧ e ≤ 9

def reverseNumber (n : FiveDigitNumber) : Nat :=
  n.e * 10000 + n.d * 1000 + n.c * 100 + n.b * 10 + n.a

def originalNumber (n : FiveDigitNumber) : Nat :=
  n.a * 10000 + n.b * 1000 + n.c * 100 + n.d * 10 + n.e

theorem correct_result (n : FiveDigitNumber) 
  (h : reverseNumber n - originalNumber n = 34056) :
  n.e > n.a ∧ 
  n.e - n.a = 3 ∧ 
  (n.a - n.e) % 10 = 6 ∧ 
  n.b > n.d :=
sorry

end NUMINAMATH_CALUDE_correct_result_l4115_411518


namespace NUMINAMATH_CALUDE_no_prime_roots_for_specific_quadratic_l4115_411550

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_prime_roots_for_specific_quadratic :
  ¬ ∃ (k : ℤ) (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p ≠ q ∧
    p + q = 97 ∧ 
    p * q = k ∧
    ∀ (x : ℤ), x^2 - 97*x + k = 0 ↔ (x = p ∨ x = q) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_specific_quadratic_l4115_411550


namespace NUMINAMATH_CALUDE_poles_count_l4115_411554

/-- The number of telephone poles given the interval distance and total distance -/
def num_poles (interval : ℕ) (total_distance : ℕ) : ℕ :=
  (total_distance / interval) + 1

/-- Theorem stating that the number of poles is 61 given the specific conditions -/
theorem poles_count : num_poles 25 1500 = 61 := by
  sorry

end NUMINAMATH_CALUDE_poles_count_l4115_411554


namespace NUMINAMATH_CALUDE_principal_amount_proof_l4115_411523

-- Define the parameters of the investment
def interest_rate : ℚ := 5 / 100
def investment_duration : ℕ := 5
def final_amount : ℚ := 10210.25

-- Define the compound interest formula
def compound_interest (principal : ℚ) : ℚ :=
  principal * (1 + interest_rate) ^ investment_duration

-- State the theorem
theorem principal_amount_proof :
  ∃ (principal : ℚ), 
    compound_interest principal = final_amount ∧ 
    (principal ≥ 7999.5 ∧ principal ≤ 8000.5) := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l4115_411523


namespace NUMINAMATH_CALUDE_parallel_case_perpendicular_case_l4115_411574

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Define perpendicular condition
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Theorem for parallel case
theorem parallel_case :
  ∃ (x : ℝ), parallel (a + 2 • (b x)) (2 • a - b x) ∧ x = 1/2 := by sorry

-- Theorem for perpendicular case
theorem perpendicular_case :
  ∃ (x : ℝ), perpendicular (a + 2 • (b x)) (2 • a - b x) ∧ (x = -2 ∨ x = 7/2) := by sorry

end NUMINAMATH_CALUDE_parallel_case_perpendicular_case_l4115_411574


namespace NUMINAMATH_CALUDE_multiply_by_fraction_l4115_411580

theorem multiply_by_fraction (a b c : ℝ) (h : a * b = c) :
  (b / 10) * a = c / 10 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_fraction_l4115_411580


namespace NUMINAMATH_CALUDE_stock_price_change_l4115_411537

theorem stock_price_change (x : ℝ) : 
  (1 - x / 100) * 1.10 = 1.012 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l4115_411537


namespace NUMINAMATH_CALUDE_area_is_33_l4115_411520

/-- A line with slope -3 intersecting positive x and y axes -/
structure Line1 where
  slope : ℝ
  x_intercept : ℝ
  y_intercept : ℝ

/-- A line intersecting x and y axes -/
structure Line2 where
  x_intercept : ℝ
  y_intercept : ℝ

/-- The intersection point of two lines -/
structure Intersection where
  x : ℝ
  y : ℝ

/-- Definition of the problem setup -/
def problem_setup (l1 : Line1) (l2 : Line2) (e : Intersection) : Prop :=
  l1.slope = -3 ∧
  l1.x_intercept > 0 ∧
  l1.y_intercept > 0 ∧
  l2.x_intercept = 10 ∧
  e.x = 3 ∧
  e.y = 3

/-- The area of quadrilateral OBEC -/
def area_OBEC (l1 : Line1) (l2 : Line2) (e : Intersection) : ℝ := sorry

/-- Theorem stating the area of quadrilateral OBEC is 33 -/
theorem area_is_33 (l1 : Line1) (l2 : Line2) (e : Intersection) :
  problem_setup l1 l2 e → area_OBEC l1 l2 e = 33 := by sorry

end NUMINAMATH_CALUDE_area_is_33_l4115_411520


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l4115_411540

theorem quadratic_solution_property (x₁ x₂ : ℝ) : 
  x₁^2 - 5*x₁ + 2 = 0 → x₂^2 - 5*x₂ + 2 = 0 → 2*x₁ - x₁*x₂ + 2*x₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l4115_411540


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_l4115_411528

theorem triangle_arithmetic_sequence (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- c cos A, b cos B, a cos C form an arithmetic sequence
  2 * b * Real.cos B = c * Real.cos A + a * Real.cos C →
  -- Given conditions
  a + c = 3 * Real.sqrt 3 / 2 →
  b = Real.sqrt 3 →
  -- Conclusions
  B = π / 3 ∧
  (1 / 2 * a * c * Real.sin B = 5 * Real.sqrt 3 / 16) :=
by sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_l4115_411528


namespace NUMINAMATH_CALUDE_suraj_innings_l4115_411505

/-- Represents the cricket problem for Suraj's innings --/
def cricket_problem (n : ℕ) : Prop :=
  let A : ℚ := 10  -- Initial average (derived from the new average minus the increase)
  let new_average : ℚ := 16  -- New average after the last innings
  let runs_increase : ℚ := 6  -- Increase in average
  let last_innings_runs : ℕ := 112  -- Runs scored in the last innings
  
  -- The equation representing the new average
  (n * A + last_innings_runs) / (n + 1) = new_average ∧
  -- The equation representing the increase in average
  new_average = A + runs_increase

/-- Theorem stating that the number of innings before the last one is 16 --/
theorem suraj_innings : cricket_problem 16 := by sorry

end NUMINAMATH_CALUDE_suraj_innings_l4115_411505


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l4115_411539

theorem smallest_k_no_real_roots : 
  ∃ (k : ℤ), k = 2 ∧ 
  (∀ x : ℝ, 2 * x * (k * x - 4) - x^2 + 6 ≠ 0) ∧
  (∀ m : ℤ, m < k → ∃ x : ℝ, 2 * x * (m * x - 4) - x^2 + 6 = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l4115_411539


namespace NUMINAMATH_CALUDE_base_ratio_l4115_411541

/-- An isosceles trapezoid with bases a and b (a > b) and height h -/
structure IsoscelesTrapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  a_gt_b : a > b
  h_gt_zero : h > 0

/-- The property that the height divides the larger base in ratio 1:3 -/
def height_divides_base (t : IsoscelesTrapezoid) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (3 * x = t.a - x)

/-- The theorem stating the ratio of bases -/
theorem base_ratio (t : IsoscelesTrapezoid) 
  (h : height_divides_base t) : t.a / t.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_ratio_l4115_411541


namespace NUMINAMATH_CALUDE_f_positive_range_min_k_for_f_plus_k_positive_l4115_411572

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x

theorem f_positive_range (a : ℝ) :
  (∀ x, f a x > 0 ↔ x > 0) ∨
  (∀ x, f a x > 0 ↔ (x > 0 ∨ x < Real.log a)) ∨
  (∀ x, f a x > 0 ↔ (x > Real.log a ∨ x < 0)) :=
sorry

theorem min_k_for_f_plus_k_positive :
  ∃! k : ℕ, k > 0 ∧ ∀ x, f 2 x + k > 0 ∧ ∀ m : ℕ, m < k → ∃ y, f 2 y + m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_f_positive_range_min_k_for_f_plus_k_positive_l4115_411572


namespace NUMINAMATH_CALUDE_triangle_area_with_median_l4115_411552

/-- Given a triangle PQR with side lengths and median, calculate its area -/
theorem triangle_area_with_median (P Q R M : ℝ × ℝ) : 
  let PQ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let PR := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  let PM := Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  let QR := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  PQ = 9 →
  PR = 17 →
  PM = 13 →
  M = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  area = A :=
by sorry

#check triangle_area_with_median

end NUMINAMATH_CALUDE_triangle_area_with_median_l4115_411552


namespace NUMINAMATH_CALUDE_alpha_value_l4115_411514

theorem alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (2 * Real.pi)) 
  (h2 : ∃ (x y : Real), x = Real.sin (Real.pi / 6) ∧ y = Real.cos (5 * Real.pi / 6) ∧ 
    x = Real.sin α ∧ y = Real.cos α) : 
  α = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l4115_411514


namespace NUMINAMATH_CALUDE_racket_price_l4115_411570

theorem racket_price (total_spent sneakers_cost outfit_cost : ℕ) 
  (h1 : total_spent = 750)
  (h2 : sneakers_cost = 200)
  (h3 : outfit_cost = 250) :
  total_spent - sneakers_cost - outfit_cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_racket_price_l4115_411570


namespace NUMINAMATH_CALUDE_f_sum_equals_six_l4115_411513

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 9
  else 4^(-x) + 3/2

-- Theorem statement
theorem f_sum_equals_six :
  f 27 + f (-Real.log 3 / Real.log 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_six_l4115_411513


namespace NUMINAMATH_CALUDE_gwen_book_count_l4115_411509

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 4

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 3

/-- The total number of books Gwen has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_book_count : total_books = 32 := by
  sorry

end NUMINAMATH_CALUDE_gwen_book_count_l4115_411509


namespace NUMINAMATH_CALUDE_open_box_volume_formula_l4115_411573

/-- The volume of an open box constructed from a rectangular sheet of metal -/
def openBoxVolume (length width x : ℝ) : ℝ :=
  (length - 2*x) * (width - 2*x) * x

theorem open_box_volume_formula :
  ∀ x : ℝ, openBoxVolume 14 10 x = 140*x - 48*x^2 + 4*x^3 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_formula_l4115_411573


namespace NUMINAMATH_CALUDE_probability_at_least_one_girl_l4115_411525

def number_of_boys : ℕ := 3
def number_of_girls : ℕ := 2
def total_students : ℕ := number_of_boys + number_of_girls
def students_selected : ℕ := 2

theorem probability_at_least_one_girl :
  (Nat.choose total_students students_selected - Nat.choose number_of_boys students_selected) /
  Nat.choose total_students students_selected = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_girl_l4115_411525


namespace NUMINAMATH_CALUDE_incorrect_expression_l4115_411542

theorem incorrect_expression (x y : ℝ) (h : x / y = 2 / 5) :
  (x + 3 * y) / x ≠ 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l4115_411542


namespace NUMINAMATH_CALUDE_transformed_is_ellipse_l4115_411569

-- Define the original circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaling transformation
def scaling_transformation (x y : ℝ) : ℝ × ℝ := (5*x, 4*y)

-- Define the resulting equation after transformation
def transformed_equation (x' y' : ℝ) : Prop :=
  ∃ x y, circle_equation x y ∧ scaling_transformation x y = (x', y')

-- Statement to prove
theorem transformed_is_ellipse :
  ∃ a b, a > b ∧ a = 5 ∧
  ∀ x' y', transformed_equation x' y' ↔ (x'^2 / a^2) + (y'^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_transformed_is_ellipse_l4115_411569


namespace NUMINAMATH_CALUDE_rectangle_length_width_difference_l4115_411561

theorem rectangle_length_width_difference 
  (length width : ℝ) 
  (h1 : length = 6)
  (h2 : width = 4)
  (h3 : 2 * (length + width) = 20)
  (h4 : ∃ d : ℝ, length = width + d) : 
  length - width = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_width_difference_l4115_411561


namespace NUMINAMATH_CALUDE_divisible_by_six_l4115_411510

theorem divisible_by_six (a : ℤ) : ∃ k : ℤ, a^3 + 11*a = 6*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l4115_411510


namespace NUMINAMATH_CALUDE_two_language_speakers_l4115_411576

/-- Represents the number of students who can speak a given language -/
structure LanguageSpeakers where
  gujarati : ℕ
  hindi : ℕ
  marathi : ℕ

/-- Represents the number of students who can speak exactly two languages -/
structure BilingualStudents where
  gujarati_hindi : ℕ
  gujarati_marathi : ℕ
  hindi_marathi : ℕ

/-- The theorem to be proved -/
theorem two_language_speakers
  (total_students : ℕ)
  (speakers : LanguageSpeakers)
  (trilingual : ℕ)
  (h_total : total_students = 22)
  (h_gujarati : speakers.gujarati = 6)
  (h_hindi : speakers.hindi = 15)
  (h_marathi : speakers.marathi = 6)
  (h_trilingual : trilingual = 1)
  : ∃ (bilingual : BilingualStudents),
    bilingual.gujarati_hindi + bilingual.gujarati_marathi + bilingual.hindi_marathi = 6 ∧
    total_students = speakers.gujarati + speakers.hindi + speakers.marathi -
      (bilingual.gujarati_hindi + bilingual.gujarati_marathi + bilingual.hindi_marathi) +
      trilingual :=
by sorry

end NUMINAMATH_CALUDE_two_language_speakers_l4115_411576


namespace NUMINAMATH_CALUDE_chord_length_theorem_l4115_411579

theorem chord_length_theorem (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x + y = 2*k - 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = 1 ∧ 
    x₂^2 + y₂^2 = 1 ∧ 
    x₁ + y₁ = 2*k - 1 ∧ 
    x₂ + y₂ = 2*k - 1 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2) →
  k = 0 ∨ k = 1 := by
sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l4115_411579


namespace NUMINAMATH_CALUDE_sum_of_binary_numbers_l4115_411559

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101₍₂₎ -/
def binary_101 : List Bool := [true, false, true]

/-- The binary representation of 110₍₂₎ -/
def binary_110 : List Bool := [false, true, true]

theorem sum_of_binary_numbers :
  binary_to_decimal binary_101 + binary_to_decimal binary_110 = 11 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_binary_numbers_l4115_411559


namespace NUMINAMATH_CALUDE_min_width_proof_l4115_411516

/-- The minimum width of a rectangular area satisfying given conditions -/
def min_width : ℝ := 5

/-- The length of the rectangular area in terms of its width -/
def length (w : ℝ) : ℝ := 2 * w + 10

/-- The area of the rectangular region -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 120 → w ≥ min_width) ∧
  (area min_width ≥ 120) ∧
  (min_width > 0) :=
sorry

end NUMINAMATH_CALUDE_min_width_proof_l4115_411516


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l4115_411504

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ + (1/4 : ℝ) ≤ 0) ↔ 
  (∀ x : ℝ, x^2 - x + (1/4 : ℝ) > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l4115_411504


namespace NUMINAMATH_CALUDE_decimal_addition_subtraction_l4115_411593

theorem decimal_addition_subtraction : 0.5 + 0.03 - 0.004 + 0.007 = 0.533 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_subtraction_l4115_411593


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_z_times_i_real_implies_modulus_l4115_411584

-- Define the complex number z as a function of k
def z (k : ℝ) : ℂ := (k^2 - 3*k - 4 : ℝ) + (k - 1 : ℝ) * Complex.I

-- Theorem for the first part of the problem
theorem z_in_second_quadrant (k : ℝ) :
  (z k).re < 0 ∧ (z k).im > 0 ↔ 1 < k ∧ k < 4 := by sorry

-- Theorem for the second part of the problem
theorem z_times_i_real_implies_modulus (k : ℝ) :
  (z k * Complex.I).im = 0 → Complex.abs (z k) = 2 ∨ Complex.abs (z k) = 3 := by sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_z_times_i_real_implies_modulus_l4115_411584


namespace NUMINAMATH_CALUDE_triangle_inequality_with_semiperimeter_l4115_411568

theorem triangle_inequality_with_semiperimeter (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  Real.sqrt (b + c - a) + Real.sqrt (c + a - b) + Real.sqrt (a + b - c) ≤ 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ∧ 
  (Real.sqrt (b + c - a) + Real.sqrt (c + a - b) + Real.sqrt (a + b - c) = 
   Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_semiperimeter_l4115_411568


namespace NUMINAMATH_CALUDE_existence_of_four_integers_l4115_411529

theorem existence_of_four_integers (a : Fin 97 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  ∃ w x y z : Fin 97, w ≠ x ∧ y ≠ z ∧ 1984 ∣ ((a w).val - (a x).val) * ((a y).val - (a z).val) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_four_integers_l4115_411529


namespace NUMINAMATH_CALUDE_a_six_plus_seven_l4115_411545

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the sequence a_n
def a (n : ℕ) : ℝ := f n

-- State the properties of f
axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_periodic (x : ℝ) : f (x + 3) = f x
axiom f_neg_two : f (-2) = -3

-- State the theorem
theorem a_six_plus_seven : a f 6 + a f 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_six_plus_seven_l4115_411545


namespace NUMINAMATH_CALUDE_hotel_assignment_count_l4115_411517

/-- Represents the number of rooms in the hotel -/
def num_rooms : ℕ := 4

/-- Represents the number of friends arriving -/
def num_friends : ℕ := 6

/-- Represents the maximum number of friends allowed per room -/
def max_per_room : ℕ := 3

/-- Calculates the number of ways to assign friends to rooms -/
def num_assignments : ℕ :=
  -- The actual calculation is not provided here
  1560

/-- Theorem stating that the number of assignments is 1560 -/
theorem hotel_assignment_count :
  num_assignments = 1560 :=
sorry

end NUMINAMATH_CALUDE_hotel_assignment_count_l4115_411517


namespace NUMINAMATH_CALUDE_cricketer_new_average_l4115_411512

/-- Represents a cricketer's performance -/
structure CricketerPerformance where
  innings : ℕ
  lastInningScore : ℕ
  averageIncrease : ℕ

/-- Calculates the new average score after the last inning -/
def newAverageScore (performance : CricketerPerformance) : ℚ :=
  sorry

/-- Theorem stating the cricketer's new average score -/
theorem cricketer_new_average
  (performance : CricketerPerformance)
  (h1 : performance.innings = 19)
  (h2 : performance.lastInningScore = 99)
  (h3 : performance.averageIncrease = 4) :
  newAverageScore performance = 27 :=
sorry

end NUMINAMATH_CALUDE_cricketer_new_average_l4115_411512


namespace NUMINAMATH_CALUDE_sum_after_transformation_l4115_411571

theorem sum_after_transformation (S a b : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_transformation_l4115_411571


namespace NUMINAMATH_CALUDE_doughnut_machine_completion_time_l4115_411586

-- Define the start time and one-third completion time
def start_time : ℕ := 7 * 60  -- 7:00 AM in minutes
def one_third_time : ℕ := 10 * 60 + 15  -- 10:15 AM in minutes

-- Define the time taken for one-third of the job
def one_third_duration : ℕ := one_third_time - start_time

-- Define the total duration of the job
def total_duration : ℕ := 3 * one_third_duration

-- Define the completion time
def completion_time : ℕ := start_time + total_duration

-- Theorem to prove
theorem doughnut_machine_completion_time :
  completion_time = 16 * 60 + 45  -- 4:45 PM in minutes
:= by sorry

end NUMINAMATH_CALUDE_doughnut_machine_completion_time_l4115_411586


namespace NUMINAMATH_CALUDE_sams_dimes_proof_l4115_411546

/-- The number of dimes Sam's dad gave him -/
def dimes_from_dad (initial_dimes final_dimes : ℕ) : ℕ :=
  final_dimes - initial_dimes

theorem sams_dimes_proof (initial_dimes final_dimes : ℕ) 
  (h1 : initial_dimes = 9)
  (h2 : final_dimes = 16) : 
  dimes_from_dad initial_dimes final_dimes = 7 := by
  sorry

end NUMINAMATH_CALUDE_sams_dimes_proof_l4115_411546


namespace NUMINAMATH_CALUDE_unpainted_area_l4115_411532

/-- The area of the unpainted region on a 5-inch wide board when crossed with a 7-inch wide board at a 45-degree angle -/
theorem unpainted_area (board1_width board2_width crossing_angle : ℝ) : 
  board1_width = 5 →
  board2_width = 7 →
  crossing_angle = 45 →
  ∃ (area : ℝ), area = 35 * Real.sqrt 2 ∧ 
    area = board1_width * Real.sqrt 2 * board2_width := by
  sorry

end NUMINAMATH_CALUDE_unpainted_area_l4115_411532


namespace NUMINAMATH_CALUDE_high_school_ratio_problem_l4115_411543

theorem high_school_ratio_problem (initial_boys initial_girls : ℕ) 
  (boys_left girls_left : ℕ) (final_boys final_girls : ℕ) : 
  (initial_boys : ℚ) / initial_girls = 3 / 4 →
  girls_left = 2 * boys_left →
  boys_left = 10 →
  (final_boys : ℚ) / final_girls = 4 / 5 →
  final_boys = initial_boys - boys_left →
  final_girls = initial_girls - girls_left →
  initial_boys = 90 := by
sorry


end NUMINAMATH_CALUDE_high_school_ratio_problem_l4115_411543


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l4115_411592

theorem fraction_sum_theorem (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l4115_411592


namespace NUMINAMATH_CALUDE_wide_flag_height_l4115_411596

/-- Given the following conditions:
  - Total fabric: 1000 square feet
  - Square flags: 4 feet by 4 feet
  - Wide rectangular flags: 5 feet by unknown height
  - Tall rectangular flags: 3 feet by 5 feet
  - 16 square flags made
  - 20 wide flags made
  - 10 tall flags made
  - 294 square feet of fabric left
Prove that the height of the wide rectangular flags is 3 feet. -/
theorem wide_flag_height (total_fabric : ℝ) (square_side : ℝ) (wide_width : ℝ) 
  (tall_width tall_height : ℝ) (num_square num_wide num_tall : ℕ) (fabric_left : ℝ)
  (h_total : total_fabric = 1000)
  (h_square : square_side = 4)
  (h_wide_width : wide_width = 5)
  (h_tall : tall_width = 3 ∧ tall_height = 5)
  (h_num_square : num_square = 16)
  (h_num_wide : num_wide = 20)
  (h_num_tall : num_tall = 10)
  (h_fabric_left : fabric_left = 294) :
  ∃ (wide_height : ℝ), wide_height = 3 ∧ 
  total_fabric = num_square * square_side^2 + num_wide * wide_width * wide_height + 
                 num_tall * tall_width * tall_height + fabric_left :=
by sorry

end NUMINAMATH_CALUDE_wide_flag_height_l4115_411596


namespace NUMINAMATH_CALUDE_expression_equals_one_l4115_411500

theorem expression_equals_one (a b : ℝ) 
  (ha : a = Real.sqrt 2 + 0.8)
  (hb : b = Real.sqrt 2 - 0.2) :
  (((2-b)/(b-1)) + 2*((a-1)/(a-2))) / (b*((a-1)/(b-1)) + a*((2-b)/(a-2))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l4115_411500
