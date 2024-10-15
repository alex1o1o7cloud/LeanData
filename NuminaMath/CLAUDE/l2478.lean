import Mathlib

namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2478_247846

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) :
  (1/a + 9/b + 25/c + 49/d + 81/e + 121/f) ≥ 286^2/9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2478_247846


namespace NUMINAMATH_CALUDE_combinatorial_equality_l2478_247879

theorem combinatorial_equality (n : ℕ) : 
  (Nat.choose n 3 = Nat.choose n 5) → n = 8 := by
sorry

end NUMINAMATH_CALUDE_combinatorial_equality_l2478_247879


namespace NUMINAMATH_CALUDE_square_of_2m2_plus_n2_l2478_247837

theorem square_of_2m2_plus_n2 (m n : ℤ) :
  ∃ k l : ℤ, (2 * m^2 + n^2)^2 = 2 * k^2 + l^2 := by
sorry

end NUMINAMATH_CALUDE_square_of_2m2_plus_n2_l2478_247837


namespace NUMINAMATH_CALUDE_max_cake_pieces_l2478_247804

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 15

/-- The size of the small cake piece in inches -/
def small_piece_size : ℕ := 5

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := (large_cake_size * large_cake_size) / (small_piece_size * small_piece_size)

theorem max_cake_pieces : max_pieces = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_cake_pieces_l2478_247804


namespace NUMINAMATH_CALUDE_chipmunk_acorns_l2478_247864

/-- Represents the number of acorns hidden in each hole by an animal -/
structure AcornsPerHole where
  chipmunk : ℕ
  squirrel : ℕ

/-- Represents the number of holes dug by each animal -/
structure HolesDug where
  chipmunk : ℕ
  squirrel : ℕ

/-- The main theorem about the number of acorns hidden by the chipmunk -/
theorem chipmunk_acorns (aph : AcornsPerHole) (h : HolesDug) : 
  aph.chipmunk = 3 → 
  aph.squirrel = 4 → 
  h.chipmunk = h.squirrel + 4 → 
  aph.chipmunk * h.chipmunk = aph.squirrel * h.squirrel → 
  aph.chipmunk * h.chipmunk = 48 := by
  sorry

#check chipmunk_acorns

end NUMINAMATH_CALUDE_chipmunk_acorns_l2478_247864


namespace NUMINAMATH_CALUDE_package_weight_is_five_l2478_247888

/-- Calculates the weight of a package given the total shipping cost, flat fee, and cost per pound. -/
def package_weight (total_cost flat_fee cost_per_pound : ℚ) : ℚ :=
  (total_cost - flat_fee) / cost_per_pound

/-- Theorem stating that given the specific shipping costs, the package weighs 5 pounds. -/
theorem package_weight_is_five :
  package_weight 9 5 (4/5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_package_weight_is_five_l2478_247888


namespace NUMINAMATH_CALUDE_triangle_x_coordinate_sum_l2478_247890

/-- Given two triangles ABC and ADF with specific areas and coordinates,
    prove that the sum of all possible x-coordinates of A is -635.6 -/
theorem triangle_x_coordinate_sum :
  let triangle_ABC_area : ℝ := 2010
  let triangle_ADF_area : ℝ := 8020
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (226, 0)
  let D : ℝ × ℝ := (680, 380)
  let F : ℝ × ℝ := (700, 400)
  ∃ (x₁ x₂ : ℝ), 
    (∃ (y₁ : ℝ), triangle_ABC_area = (1/2) * 226 * |y₁|) ∧
    (∃ (y₂ : ℝ), triangle_ADF_area = (1/2) * 20 * |x₁ - y₂ + 300| / Real.sqrt 2) ∧
    (∃ (y₃ : ℝ), triangle_ADF_area = (1/2) * 20 * |x₂ - y₃ + 300| / Real.sqrt 2) ∧
    x₁ + x₂ = -635.6 := by
  sorry

#check triangle_x_coordinate_sum

end NUMINAMATH_CALUDE_triangle_x_coordinate_sum_l2478_247890


namespace NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_z_l2478_247823

/-- Given that z = (2 + mi) / (1 + i) is a pure imaginary number, 
    prove that the imaginary part of z is -2. -/
theorem imaginary_part_of_pure_imaginary_z (m : ℝ) : 
  let z : ℂ := (2 + m * Complex.I) / (1 + Complex.I)
  (∃ (y : ℝ), z = y * Complex.I) → Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_z_l2478_247823


namespace NUMINAMATH_CALUDE_perp_condition_l2478_247889

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The first line x + y = 0 -/
def line1 : Line := { slope := -1, intercept := 0 }

/-- The second line x - ay = 0 -/
def line2 (a : ℝ) : Line := { slope := a, intercept := 0 }

/-- Theorem: a = 1 is necessary and sufficient for perpendicularity -/
theorem perp_condition (a : ℝ) :
  perpendicular line1 (line2 a) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perp_condition_l2478_247889


namespace NUMINAMATH_CALUDE_quadratic_function_zeros_range_l2478_247815

theorem quadratic_function_zeros_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    (x₁ ∈ Set.Ioo (-2) 0 ∧ x₂ ∈ Set.Ioo 2 3) ∧
    (x₁^2 - 2*x₁ + a = 0 ∧ x₂^2 - 2*x₂ + a = 0)) →
  a ∈ Set.Ioo (-3) 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_zeros_range_l2478_247815


namespace NUMINAMATH_CALUDE_pineapples_sold_l2478_247860

theorem pineapples_sold (initial : ℕ) (rotten : ℕ) (fresh : ℕ) : 
  initial = 86 → rotten = 9 → fresh = 29 → initial - (fresh + rotten) = 48 := by
sorry

end NUMINAMATH_CALUDE_pineapples_sold_l2478_247860


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2478_247870

/-- Factorization of a quadratic expression -/
theorem quadratic_factorization (a : ℝ) : a^2 - 8*a + 16 = (a - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2478_247870


namespace NUMINAMATH_CALUDE_parent_teacher_night_duration_l2478_247898

def time_to_school : ℕ := 20
def time_from_school : ℕ := 20
def total_time : ℕ := 110

theorem parent_teacher_night_duration :
  total_time - (time_to_school + time_from_school) = 70 :=
by sorry

end NUMINAMATH_CALUDE_parent_teacher_night_duration_l2478_247898


namespace NUMINAMATH_CALUDE_alice_walking_time_l2478_247841

/-- Given Bob's walking time and distance, and the relationship between Alice and Bob's walking times and distances, prove that Alice would take 21 minutes to walk 7 miles. -/
theorem alice_walking_time 
  (bob_distance : ℝ) 
  (bob_time : ℝ) 
  (alice_distance : ℝ) 
  (alice_bob_time_ratio : ℝ) 
  (alice_target_distance : ℝ) 
  (h1 : bob_distance = 6) 
  (h2 : bob_time = 36) 
  (h3 : alice_distance = 4) 
  (h4 : alice_bob_time_ratio = 1/3) 
  (h5 : alice_target_distance = 7) : 
  (alice_target_distance / (alice_distance / (alice_bob_time_ratio * bob_time))) = 21 :=
by sorry

end NUMINAMATH_CALUDE_alice_walking_time_l2478_247841


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l2478_247824

def U : Finset Nat := {1,2,3,4,5}
def M : Finset Nat := {1,4}
def N : Finset Nat := {1,3,5}

theorem intersection_complement_equal : N ∩ (U \ M) = {3,5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l2478_247824


namespace NUMINAMATH_CALUDE_deposit_withdrawal_ratio_l2478_247805

/-- Prove that the ratio of the deposited amount to the withdrawn amount is 2:1 --/
theorem deposit_withdrawal_ratio (initial_savings withdrawal final_balance : ℚ) 
  (h1 : initial_savings = 230)
  (h2 : withdrawal = 60)
  (h3 : final_balance = 290) : 
  (final_balance - (initial_savings - withdrawal)) / withdrawal = 2 := by
  sorry

end NUMINAMATH_CALUDE_deposit_withdrawal_ratio_l2478_247805


namespace NUMINAMATH_CALUDE_sandwich_cost_l2478_247816

theorem sandwich_cost (total_cost soda_cost : ℝ) 
  (h1 : total_cost = 10.46)
  (h2 : soda_cost = 0.87) : 
  ∃ sandwich_cost : ℝ, 
    sandwich_cost = 3.49 ∧ 
    2 * sandwich_cost + 4 * soda_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_sandwich_cost_l2478_247816


namespace NUMINAMATH_CALUDE_marvin_bottle_caps_l2478_247845

theorem marvin_bottle_caps (initial : ℕ) (remaining : ℕ) (taken : ℕ) : 
  initial = 16 → remaining = 10 → taken = initial - remaining → taken = 6 := by
  sorry

end NUMINAMATH_CALUDE_marvin_bottle_caps_l2478_247845


namespace NUMINAMATH_CALUDE_continuous_function_on_T_has_fixed_point_l2478_247811

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ) (q : ℚ), t ∈ Set.Icc 0 1 ∧ p = (t * q, 1 - t)}

-- State the theorem
theorem continuous_function_on_T_has_fixed_point
  (f : T → T) (hf : Continuous f) :
  ∃ x : T, f x = x := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_on_T_has_fixed_point_l2478_247811


namespace NUMINAMATH_CALUDE_simplify_exponential_fraction_l2478_247869

theorem simplify_exponential_fraction (n : ℕ) :
  (3^(n+5) - 3 * 3^n) / (3 * 3^(n+4)) = 240 / 81 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponential_fraction_l2478_247869


namespace NUMINAMATH_CALUDE_cookie_cost_calculation_l2478_247877

def cookies_per_dozen : ℕ := 12

-- Define the problem parameters
def total_dozens : ℕ := 6
def selling_price : ℚ := 3/2
def charity_share : ℚ := 45

-- Theorem to prove
theorem cookie_cost_calculation :
  let total_cookies := total_dozens * cookies_per_dozen
  let total_revenue := total_cookies * selling_price
  let total_profit := 2 * charity_share
  let total_cost := total_revenue - total_profit
  (total_cost / total_cookies : ℚ) = 1/4 := by sorry

end NUMINAMATH_CALUDE_cookie_cost_calculation_l2478_247877


namespace NUMINAMATH_CALUDE_average_position_l2478_247834

def fractions : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]

theorem average_position :
  let avg := (List.sum fractions) / fractions.length
  avg = 223 / 840 ∧ 1/4 < avg ∧ avg < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_average_position_l2478_247834


namespace NUMINAMATH_CALUDE_sum_ages_after_ten_years_l2478_247886

/-- Given Ann's age and Tom's age relative to Ann's, calculate the sum of their ages after a certain number of years. -/
def sum_ages_after_years (ann_age : ℕ) (tom_age_multiplier : ℕ) (years_later : ℕ) : ℕ :=
  (ann_age + years_later) + (ann_age * tom_age_multiplier + years_later)

/-- Theorem stating that given Ann's age is 6 and Tom's age is twice Ann's, the sum of their ages 10 years later is 38. -/
theorem sum_ages_after_ten_years :
  sum_ages_after_years 6 2 10 = 38 := by
  sorry

end NUMINAMATH_CALUDE_sum_ages_after_ten_years_l2478_247886


namespace NUMINAMATH_CALUDE_dvd_price_ratio_l2478_247829

theorem dvd_price_ratio (mike_price steve_total_price : ℝ) 
  (h1 : mike_price = 5)
  (h2 : steve_total_price = 18)
  (h3 : ∃ (steve_online_price : ℝ), 
    steve_total_price = steve_online_price + 0.8 * steve_online_price) :
  ∃ (steve_online_price : ℝ), 
    steve_online_price / mike_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_dvd_price_ratio_l2478_247829


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_perpendicular_to_all_lines_l2478_247882

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_to_plane_perpendicular_to_all_lines
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contained_in n α) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_perpendicular_to_all_lines_l2478_247882


namespace NUMINAMATH_CALUDE_curve_C_polar_equation_l2478_247872

/-- Given a curve C with parametric equations x = 1 + cos α and y = sin α, 
    its polar equation is ρ = 2cos θ -/
theorem curve_C_polar_equation (α θ : Real) (ρ x y : Real) :
  (x = 1 + Real.cos α ∧ y = Real.sin α) →
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ρ = 2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_curve_C_polar_equation_l2478_247872


namespace NUMINAMATH_CALUDE_alcohol_concentration_after_mixing_l2478_247825

/-- Represents a vessel with a given capacity and alcohol concentration -/
structure Vessel where
  capacity : ℝ
  alcoholConcentration : ℝ

/-- Calculates the amount of alcohol in a vessel -/
def alcoholAmount (v : Vessel) : ℝ := v.capacity * v.alcoholConcentration

/-- Theorem: The alcohol concentration in vessel D after mixing is 35% -/
theorem alcohol_concentration_after_mixing 
  (vesselA : Vessel)
  (vesselB : Vessel)
  (vesselC : Vessel)
  (vesselD : Vessel)
  (h1 : vesselA.capacity = 5)
  (h2 : vesselA.alcoholConcentration = 0.25)
  (h3 : vesselB.capacity = 12)
  (h4 : vesselB.alcoholConcentration = 0.45)
  (h5 : vesselC.capacity = 7)
  (h6 : vesselC.alcoholConcentration = 0.35)
  (h7 : vesselD.capacity = 26) :
  let totalAlcohol := alcoholAmount vesselA + alcoholAmount vesselB + alcoholAmount vesselC
  let totalVolume := vesselA.capacity + vesselB.capacity + vesselC.capacity + (vesselD.capacity - (vesselA.capacity + vesselB.capacity + vesselC.capacity))
  totalAlcohol / totalVolume = 0.35 := by
    sorry

end NUMINAMATH_CALUDE_alcohol_concentration_after_mixing_l2478_247825


namespace NUMINAMATH_CALUDE_product_from_gcd_lcm_l2478_247812

theorem product_from_gcd_lcm (a b : ℤ) : 
  Int.gcd a b = 8 → Int.lcm a b = 24 → a * b = 192 := by
  sorry

end NUMINAMATH_CALUDE_product_from_gcd_lcm_l2478_247812


namespace NUMINAMATH_CALUDE_tourist_contact_probability_l2478_247822

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ := 1 - (1 - p)^42

/-- Theorem stating the probability of contact between two groups of tourists -/
theorem tourist_contact_probability 
  (p : ℝ) 
  (h1 : 0 ≤ p) 
  (h2 : p ≤ 1) 
  (group1 : Fin 6 → Type) 
  (group2 : Fin 7 → Type) :
  contact_probability p = 1 - (1 - p)^42 := by
sorry

end NUMINAMATH_CALUDE_tourist_contact_probability_l2478_247822


namespace NUMINAMATH_CALUDE_sweets_spending_proof_l2478_247836

/-- Calculates the amount spent on sweets given a weekly allowance, junk food spending ratio, and savings amount. -/
def amount_spent_on_sweets (allowance : ℚ) (junk_food_ratio : ℚ) (savings : ℚ) : ℚ :=
  allowance - allowance * junk_food_ratio - savings

/-- Proves that given a weekly allowance of $30, spending 1/3 on junk food, and saving $12, the amount spent on sweets is $8. -/
theorem sweets_spending_proof :
  amount_spent_on_sweets 30 (1/3) 12 = 8 := by
  sorry

#eval amount_spent_on_sweets 30 (1/3) 12

end NUMINAMATH_CALUDE_sweets_spending_proof_l2478_247836


namespace NUMINAMATH_CALUDE_power_zero_of_three_minus_pi_l2478_247851

theorem power_zero_of_three_minus_pi : (3 - Real.pi) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_of_three_minus_pi_l2478_247851


namespace NUMINAMATH_CALUDE_sum_of_primes_equals_210_l2478_247899

theorem sum_of_primes_equals_210 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h : 100^2 + 1^2 = 65^2 + 76^2 ∧ 100^2 + 1^2 = p * q) : 
  p + q = 210 := by
sorry

end NUMINAMATH_CALUDE_sum_of_primes_equals_210_l2478_247899


namespace NUMINAMATH_CALUDE_binomial_expansion_101_2_l2478_247814

theorem binomial_expansion_101_2 : 
  101^3 + 3*(101^2)*2 + 3*101*(2^2) + 2^3 = 1092727 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_101_2_l2478_247814


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l2478_247874

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l2478_247874


namespace NUMINAMATH_CALUDE_shiny_igneous_rocks_l2478_247821

theorem shiny_igneous_rocks (total : ℕ) (sedimentary : ℕ) (igneous : ℕ) :
  total = 270 →
  igneous = sedimentary / 2 →
  total = sedimentary + igneous →
  (igneous / 3 : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_shiny_igneous_rocks_l2478_247821


namespace NUMINAMATH_CALUDE_solution_set_l2478_247831

-- Define the condition
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*a*x + a > 0

-- Define the inequality
def inequality (a t : ℝ) : Prop :=
  a^(2*t + 1) < a^(t^2 + 2*t - 3)

-- State the theorem
theorem solution_set (a : ℝ) (h : always_positive a) :
  {t : ℝ | inequality a t} = {t : ℝ | -2 < t ∧ t < 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_l2478_247831


namespace NUMINAMATH_CALUDE_parallelogram_angle_ratio_l2478_247878

-- Define the parallelogram ABCD and point O
variable (A B C D O : Point)

-- Define the property of being a parallelogram
def is_parallelogram (A B C D : Point) : Prop := sorry

-- Define the property of O being the intersection of diagonals
def is_diagonal_intersection (A B C D O : Point) : Prop := sorry

-- Define angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem parallelogram_angle_ratio 
  (h_para : is_parallelogram A B C D)
  (h_diag : is_diagonal_intersection A B C D O)
  (h_cab : angle_measure C A B = 3 * angle_measure D B A)
  (h_dbc : angle_measure D B C = 3 * angle_measure D B A)
  (h_acb : ∃ r : ℝ, angle_measure A C B = r * angle_measure A O B) :
  ∃ r : ℝ, angle_measure A C B = r * angle_measure A O B ∧ r = 2 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_ratio_l2478_247878


namespace NUMINAMATH_CALUDE_intersecting_squares_area_difference_l2478_247885

theorem intersecting_squares_area_difference :
  let A : ℝ := 12^2
  let B : ℝ := 9^2
  let C : ℝ := 7^2
  let D : ℝ := 3^2
  ∀ (E F G : ℝ),
  (A + E - (B + F)) - (C + G - (B + D + F)) = 103 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_squares_area_difference_l2478_247885


namespace NUMINAMATH_CALUDE_right_triangle_smaller_angle_l2478_247865

theorem right_triangle_smaller_angle (a b c : Real) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  c = 90 →           -- One angle is 90° (right angle)
  b = 2 * a →        -- One angle is twice the other
  a = 30 :=          -- The smaller angle is 30°
by sorry

end NUMINAMATH_CALUDE_right_triangle_smaller_angle_l2478_247865


namespace NUMINAMATH_CALUDE_three_distinct_triangles60_l2478_247891

/-- A triangle with integer side lengths and one 60° angle -/
structure Triangle60 where
  a : ℕ
  b : ℕ
  c : ℕ
  coprime : Nat.gcd a (Nat.gcd b c) = 1
  angle60 : a^2 + b^2 + c^2 = 2 * max a (max b c)^2

/-- The existence of at least three distinct triangles with integer side lengths and one 60° angle -/
theorem three_distinct_triangles60 : ∃ (t1 t2 t3 : Triangle60),
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧
  (t1.a, t1.b, t1.c) ≠ (5, 7, 8) ∧
  (t2.a, t2.b, t2.c) ≠ (5, 7, 8) ∧
  (t3.a, t3.b, t3.c) ≠ (5, 7, 8) :=
sorry

end NUMINAMATH_CALUDE_three_distinct_triangles60_l2478_247891


namespace NUMINAMATH_CALUDE_unique_input_for_542_l2478_247852

def machine_operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then 5 * n else 3 * n + 2

def iterate_machine (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => machine_operation (iterate_machine n k)

theorem unique_input_for_542 :
  ∃! n : ℕ, n > 0 ∧ iterate_machine n 5 = 542 :=
by
  -- The proof would go here
  sorry

#eval iterate_machine 112500 5  -- Should output 542

end NUMINAMATH_CALUDE_unique_input_for_542_l2478_247852


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l2478_247883

/-- Proves that given a distance traveled at a certain speed, if the person were to travel an additional distance in the same time at a faster speed, we can calculate that faster speed. -/
theorem faster_speed_calculation (D : ℝ) (v_original : ℝ) (additional_distance : ℝ) (v_faster : ℝ) :
  D = 33.333333333333336 →
  v_original = 10 →
  additional_distance = 20 →
  D / v_original = (D + additional_distance) / v_faster →
  v_faster = 16 := by
  sorry

end NUMINAMATH_CALUDE_faster_speed_calculation_l2478_247883


namespace NUMINAMATH_CALUDE_cost_verification_max_purchase_l2478_247893

/-- Represents the cost of a single bat -/
def bat_cost : ℝ := 70

/-- Represents the cost of a single ball -/
def ball_cost : ℝ := 20

/-- Represents the discount rate when purchasing at least 3 bats and 3 balls -/
def discount_rate : ℝ := 0.10

/-- Represents the sales tax rate -/
def sales_tax_rate : ℝ := 0.08

/-- Represents the budget -/
def budget : ℝ := 270

/-- Verifies that the given costs satisfy the conditions -/
theorem cost_verification : 
  2 * bat_cost + 4 * ball_cost = 220 ∧ 
  bat_cost + 6 * ball_cost = 190 := by sorry

/-- Proves that the maximum number of bats and balls that can be purchased is 3 -/
theorem max_purchase : 
  ∀ n : ℕ, 
    n ≥ 3 → 
    n * (bat_cost + ball_cost) * (1 - discount_rate) * (1 + sales_tax_rate) ≤ budget → 
    n ≤ 3 := by sorry

end NUMINAMATH_CALUDE_cost_verification_max_purchase_l2478_247893


namespace NUMINAMATH_CALUDE_exists_balanced_partition_l2478_247868

/-- An undirected graph represented by its vertex set and edge relation -/
structure Graph (V : Type) where
  edge : V → V → Prop
  symm : ∀ u v, edge u v → edge v u

/-- The neighborhood of a vertex v in a set S -/
def neighborhood {V : Type} (G : Graph V) (S : Set V) (v : V) : Set V :=
  {u ∈ S | G.edge v u}

/-- A partition of a set into two disjoint subsets -/
structure Partition (V : Type) where
  A : Set V
  B : Set V
  disjoint : A ∩ B = ∅
  complete : A ∪ B = Set.univ

/-- The main theorem statement -/
theorem exists_balanced_partition {V : Type} (G : Graph V) :
  ∃ (P : Partition V), 
    (∀ v ∈ P.A, (neighborhood G P.B v).ncard ≥ (neighborhood G P.A v).ncard) ∧
    (∀ v ∈ P.B, (neighborhood G P.A v).ncard ≥ (neighborhood G P.B v).ncard) := by
  sorry

end NUMINAMATH_CALUDE_exists_balanced_partition_l2478_247868


namespace NUMINAMATH_CALUDE_f_properties_l2478_247820

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * Real.log x

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := a + Real.log x + 1

-- State the theorem
theorem f_properties (a : ℝ) :
  (f_deriv a (Real.exp 1) = 3) →
  (∃ (k : ℤ), k = 3 ∧ 
    (∀ x > 1, f 1 x - ↑k * x + ↑k > 0) ∧
    (∀ k' > ↑k, ∃ x > 1, f 1 x - ↑k' * x + ↑k' ≤ 0)) →
  (a = 1) ∧
  (∀ x ∈ Set.Ioo 0 (Real.exp (-2)), ∀ y ∈ Set.Ioo x (Real.exp (-2)), f 1 y < f 1 x) ∧
  (∀ x ∈ Set.Ioi (Real.exp (-2)), ∀ y ∈ Set.Ioi x, f 1 y > f 1 x) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l2478_247820


namespace NUMINAMATH_CALUDE_infinitely_many_minimal_points_l2478_247828

/-- Distance function from origin to point (x, y) -/
def distance (x y : ℝ) : ℝ := |x| + |y|

/-- The set of points (x, y) on the line y = x + 1 that minimize the distance from the origin -/
def minimal_distance_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 1 ∧ ∀ q : ℝ × ℝ, q.2 = q.1 + 1 → distance p.1 p.2 ≤ distance q.1 q.2}

/-- Theorem stating that there are infinitely many points that minimize the distance -/
theorem infinitely_many_minimal_points : Set.Infinite minimal_distance_points := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_minimal_points_l2478_247828


namespace NUMINAMATH_CALUDE_cube_edge_probability_cube_edge_probability_proof_l2478_247818

/-- The probability of randomly selecting two vertices that form an edge in a cube -/
theorem cube_edge_probability : ℚ :=
let num_vertices : ℕ := 8
let num_edges : ℕ := 12
let total_pairs : ℕ := num_vertices.choose 2
3 / 7

/-- Proof that the probability of randomly selecting two vertices that form an edge in a cube is 3/7 -/
theorem cube_edge_probability_proof :
  cube_edge_probability = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_probability_cube_edge_probability_proof_l2478_247818


namespace NUMINAMATH_CALUDE_volume_of_specific_room_l2478_247803

/-- The volume of a rectangular room -/
def room_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a room with dimensions 100 m x 10 m x 10 m is 100,000 cubic meters -/
theorem volume_of_specific_room : 
  room_volume 100 10 10 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_room_l2478_247803


namespace NUMINAMATH_CALUDE_problem_statement_l2478_247850

theorem problem_statement : (0.125 : ℝ)^2012 * (2^2012)^3 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2478_247850


namespace NUMINAMATH_CALUDE_ball_probabilities_l2478_247856

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 6

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 4

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- Calculates the probability of drawing two red balls -/
def prob_two_red : ℚ := (red_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls : ℚ)

/-- Calculates the probability of drawing at least one red ball -/
def prob_at_least_one_red : ℚ := 1 - (white_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls : ℚ)

theorem ball_probabilities :
  prob_two_red = 1/15 ∧ prob_at_least_one_red = 3/5 := by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2478_247856


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l2478_247855

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y n : ℝ) : Prop := x^2 - n*(y-1)^2 = 4

-- Define the tangency condition
def are_tangent (n : ℝ) : Prop := 
  ∃ x y, ellipse x y ∧ hyperbola x y n

-- Theorem statement
theorem ellipse_hyperbola_tangency (n : ℝ) :
  are_tangent n → n = 45/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l2478_247855


namespace NUMINAMATH_CALUDE_boys_exam_pass_count_l2478_247857

theorem boys_exam_pass_count :
  ∀ (total_boys : ℕ) 
    (avg_all avg_pass avg_fail : ℚ)
    (pass_count : ℕ),
  total_boys = 120 →
  avg_all = 35 →
  avg_pass = 39 →
  avg_fail = 15 →
  pass_count ≤ total_boys →
  (pass_count : ℚ) * avg_pass + (total_boys - pass_count : ℚ) * avg_fail = (total_boys : ℚ) * avg_all →
  pass_count = 100 := by
sorry

end NUMINAMATH_CALUDE_boys_exam_pass_count_l2478_247857


namespace NUMINAMATH_CALUDE_minimal_fraction_difference_l2478_247842

theorem minimal_fraction_difference (p q : ℕ+) : 
  (4 : ℚ) / 7 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < 7 / 12 ∧ 
  (∀ p' q' : ℕ+, (4 : ℚ) / 7 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < 7 / 12 → q ≤ q') →
  q - p = 8 := by
sorry

end NUMINAMATH_CALUDE_minimal_fraction_difference_l2478_247842


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2478_247800

theorem unique_prime_solution :
  ∃! (p q r : ℕ),
    Prime p ∧ Prime q ∧ Prime r ∧
    p < q ∧ q < r ∧
    25 * p * q + r = 2004 ∧
    ∃ m : ℕ, p * q * r + 1 = m * m ∧
    p = 7 ∧ q = 11 ∧ r = 79 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2478_247800


namespace NUMINAMATH_CALUDE_monday_rainfall_duration_l2478_247847

/-- Represents the rainfall data for three days -/
structure RainfallData where
  monday_rate : ℝ
  monday_duration : ℝ
  tuesday_rate : ℝ
  tuesday_duration : ℝ
  wednesday_rate : ℝ
  wednesday_duration : ℝ
  total_rainfall : ℝ

/-- Theorem: The duration of rainfall on Monday is 7 hours -/
theorem monday_rainfall_duration (data : RainfallData) : data.monday_duration = 7 :=
  by
  have h1 : data.monday_rate = 1 := by sorry
  have h2 : data.tuesday_rate = 2 := by sorry
  have h3 : data.tuesday_duration = 4 := by sorry
  have h4 : data.wednesday_rate = 2 * data.tuesday_rate := by sorry
  have h5 : data.wednesday_duration = 2 := by sorry
  have h6 : data.total_rainfall = 23 := by sorry
  have h7 : data.total_rainfall = 
    data.monday_rate * data.monday_duration + 
    data.tuesday_rate * data.tuesday_duration + 
    data.wednesday_rate * data.wednesday_duration := by sorry
  sorry

end NUMINAMATH_CALUDE_monday_rainfall_duration_l2478_247847


namespace NUMINAMATH_CALUDE_no_definitive_conclusion_l2478_247809

-- Define the sets
variable (Beta Zeta Yota : Set α)

-- Define the hypotheses
variable (h1 : ∃ x, x ∈ Beta ∧ x ∉ Zeta)
variable (h2 : Zeta ⊆ Yota)

-- Define the statements that cannot be conclusively proven
def statement_A := ∃ x, x ∈ Beta ∧ x ∉ Yota
def statement_B := Beta ⊆ Yota
def statement_C := Beta ∩ Yota = ∅
def statement_D := ∃ x, x ∈ Beta ∧ x ∈ Yota

-- Theorem stating that none of the statements can be definitively concluded
theorem no_definitive_conclusion :
  ¬(statement_A Beta Yota ∨ statement_B Beta Yota ∨ statement_C Beta Yota ∨ statement_D Beta Yota) :=
sorry

end NUMINAMATH_CALUDE_no_definitive_conclusion_l2478_247809


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l2478_247807

def total_potatoes : ℕ := 13
def cooked_potatoes : ℕ := 5
def cooking_time_per_potato : ℕ := 6

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 48 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l2478_247807


namespace NUMINAMATH_CALUDE_sequence_sum_l2478_247896

def is_six_digit_number (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def sequence_property (x : ℕ → ℕ) : Prop :=
  is_six_digit_number (x 1) ∧
  ∀ n : ℕ, n ≥ 1 → Nat.Prime (x (n + 1)) ∧ (x (n + 1) ∣ x n + 1)

theorem sequence_sum (x : ℕ → ℕ) (h : sequence_property x) : x 19 + x 20 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2478_247896


namespace NUMINAMATH_CALUDE_loads_to_wash_l2478_247887

theorem loads_to_wash (total : ℕ) (washed : ℕ) (h1 : total = 14) (h2 : washed = 8) :
  total - washed = 6 := by
  sorry

end NUMINAMATH_CALUDE_loads_to_wash_l2478_247887


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2478_247808

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^2 - 4*x*y + 3*y^2 - z^2 = 25) ∧ 
  (-x^2 + 5*y*z + 3*z^2 = 55) ∧ 
  (x^2 + 2*x*y + 9*z^2 = 150) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2478_247808


namespace NUMINAMATH_CALUDE_inequality_proof_l2478_247853

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + 1/a) * (1 + 1/b) ≥ 8 / (1 + a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2478_247853


namespace NUMINAMATH_CALUDE_inequality_proof_l2478_247813

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 0.5) :
  (x * y^2) / (x^3 + 1) + (y * z^2) / (y^3 + 1) + (z * x^2) / (z^3 + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2478_247813


namespace NUMINAMATH_CALUDE_soda_cost_calculation_l2478_247861

def restaurant_bill (num_adults num_children : ℕ) (adult_meal_cost child_meal_cost soda_cost : ℚ) : Prop :=
  let total_people := num_adults + num_children
  let meal_cost := (num_adults * adult_meal_cost) + (num_children * child_meal_cost)
  let total_bill := meal_cost + (total_people * soda_cost)
  (num_adults = 6) ∧ 
  (num_children = 2) ∧ 
  (adult_meal_cost = 6) ∧ 
  (child_meal_cost = 4) ∧ 
  (total_bill = 60)

theorem soda_cost_calculation :
  ∃ (soda_cost : ℚ), restaurant_bill 6 2 6 4 soda_cost ∧ soda_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_calculation_l2478_247861


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_equals_62_l2478_247826

theorem root_sum_reciprocal_equals_62 :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x^3 - 9*x^2 + 9*x = 1 ↔ x = a ∨ x = b ∨ x = 1) ∧
    a > b ∧
    a > 1 ∧
    b < 1 ∧
    a/b + b/a = 62 :=
by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_equals_62_l2478_247826


namespace NUMINAMATH_CALUDE_triple_sum_arithmetic_sequence_l2478_247849

def arithmetic_sequence (a₁ l n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * ((l - a₁) / (n - 1)))

def sum_arithmetic_sequence (a₁ l n : ℕ) : ℕ :=
  (n * (a₁ + l)) / 2

theorem triple_sum_arithmetic_sequence :
  let a₁ := 74
  let l := 107
  let n := 12
  3 * (sum_arithmetic_sequence a₁ l n) = 3258 := by
  sorry

end NUMINAMATH_CALUDE_triple_sum_arithmetic_sequence_l2478_247849


namespace NUMINAMATH_CALUDE_inequality_system_solution_iff_l2478_247854

theorem inequality_system_solution_iff (a : ℝ) :
  (∃ x : ℝ, x ≥ -1 ∧ 2 * x < a) ↔ a > -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_iff_l2478_247854


namespace NUMINAMATH_CALUDE_pen_profit_percentage_l2478_247897

/-- Calculates the profit percentage for a retailer selling pens --/
theorem pen_profit_percentage
  (num_pens : ℕ)
  (price_pens : ℕ)
  (discount_percent : ℚ)
  (h1 : num_pens = 120)
  (h2 : price_pens = 36)
  (h3 : discount_percent = 1/100)
  : ∃ (profit_percent : ℚ), profit_percent = 230/100 :=
by sorry

end NUMINAMATH_CALUDE_pen_profit_percentage_l2478_247897


namespace NUMINAMATH_CALUDE_hexagon_angle_sum_l2478_247839

-- Define the angles
def angle_A : ℝ := 35
def angle_B : ℝ := 80
def angle_C : ℝ := 30

-- Define the hexagon
def is_hexagon (x y : ℝ) : Prop :=
  angle_A + angle_B + (360 - x) + 90 + 60 + y = 720

-- Theorem statement
theorem hexagon_angle_sum (x y : ℝ) (h : is_hexagon x y) : x + y = 95 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_sum_l2478_247839


namespace NUMINAMATH_CALUDE_checkerboard_coverage_uncoverable_boards_l2478_247801

/-- Represents a rectangular checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Checks if a checkerboard can be completely covered by non-overlapping dominos -/
def can_cover (board : Checkerboard) : Prop :=
  Even (board.rows * board.cols)

/-- Theorem: A checkerboard can be covered iff its total number of squares is even -/
theorem checkerboard_coverage (board : Checkerboard) :
  can_cover board ↔ Even (board.rows * board.cols) :=
sorry

/-- Examples of checkerboards -/
def board1 := Checkerboard.mk 4 6
def board2 := Checkerboard.mk 5 5
def board3 := Checkerboard.mk 4 7
def board4 := Checkerboard.mk 5 6
def board5 := Checkerboard.mk 3 7

/-- Theorem: Specific boards that cannot be covered -/
theorem uncoverable_boards :
  ¬(can_cover board2) ∧ ¬(can_cover board5) :=
sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_uncoverable_boards_l2478_247801


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2478_247843

theorem product_sum_theorem (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = 120 →
  p + q + r + s + t = 32 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2478_247843


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l2478_247838

/-- Two lines are parallel if and only if their slopes are equal and they don't coincide -/
def parallel (m : ℝ) : Prop :=
  (m / 3 = 1 / (m - 2)) ∧ (-5 ≠ -1 / (m - 2))

/-- The theorem states that if the lines are parallel, then m is either 3 or -1 -/
theorem parallel_lines_m_values (m : ℝ) :
  parallel m → m = 3 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l2478_247838


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_14_9_l2478_247895

/-- A quadratic function f(x) = ax^2 + bx + c with vertex at (6, -2) and passing through (3, 0) -/
def QuadraticFunction (a b c : ℚ) : ℚ → ℚ :=
  fun x ↦ a * x^2 + b * x + c

theorem sum_of_coefficients_equals_14_9 (a b c : ℚ) :
  (QuadraticFunction a b c 6 = -2) →  -- vertex at (6, -2)
  (QuadraticFunction a b c 3 = 0) →   -- passes through (3, 0)
  (∀ x, QuadraticFunction a b c (12 - x) = QuadraticFunction a b c x) →  -- vertical symmetry
  a + b + c = 14 / 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_14_9_l2478_247895


namespace NUMINAMATH_CALUDE_min_value_implies_a_range_l2478_247873

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2*a*x - 2 else x + 36/x - 6*a

/-- Theorem stating that if f(2) is the minimum value of f(x), then a ∈ [2, 5] -/
theorem min_value_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 2) → 2 ≤ a ∧ a ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_min_value_implies_a_range_l2478_247873


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2478_247833

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 5 ∧ min = -15 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2478_247833


namespace NUMINAMATH_CALUDE_equation_solution_l2478_247858

theorem equation_solution : ∃ x : ℝ, 15 * 2 = x - 3 + 5 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2478_247858


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_min_value_reciprocal_min_value_sqrt_l2478_247848

-- Statement 1
theorem inequality_one (x : ℝ) (h : x ≥ 0) : x + 1 + 1 / (x + 1) ≥ 2 := by sorry

-- Statement 2
theorem inequality_two (x : ℝ) (h : x > 0) : (x + 1) / Real.sqrt x ≥ 2 := by sorry

-- Statement 3
theorem min_value_reciprocal : ∃ (m : ℝ), ∀ (x : ℝ), x + 1/x ≥ m ∧ ∃ (y : ℝ), y + 1/y = m := by sorry

-- Statement 4
theorem min_value_sqrt : ∃ (m : ℝ), ∀ (x : ℝ), Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) ≥ m ∧ 
  ∃ (y : ℝ), Real.sqrt (y^2 + 2) + 1 / Real.sqrt (y^2 + 2) = m := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_min_value_reciprocal_min_value_sqrt_l2478_247848


namespace NUMINAMATH_CALUDE_total_money_sam_and_billy_l2478_247810

/-- Given Sam has $75 and Billy has $25 less than twice the money Sam has, 
    their total money together is $200. -/
theorem total_money_sam_and_billy : 
  ∀ (sam_money billy_money : ℕ),
  sam_money = 75 →
  billy_money = 2 * sam_money - 25 →
  sam_money + billy_money = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_money_sam_and_billy_l2478_247810


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2478_247880

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a > b ∧ b > 0 → a^2 > b^2) ∧
  ∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2478_247880


namespace NUMINAMATH_CALUDE_minimum_value_and_inequality_l2478_247871

def f (x : ℝ) : ℝ := |x + 3| + |x - 1|

theorem minimum_value_and_inequality (p q r : ℝ) :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 4) ∧
  (p^2 + 2*q^2 + r^2 = 4 → q*(p + r) ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_and_inequality_l2478_247871


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2478_247835

theorem no_integer_solutions : ¬∃ (x y : ℤ), (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2478_247835


namespace NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_roots_l2478_247832

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

-- Define the quadratic equation
def quadratic_equation (a : ℕ → ℝ) (k : ℕ) (x : ℝ) :=
  a k * x^2 + 2 * a (k + 1) * x + a (k + 2) = 0

-- Main theorem
theorem arithmetic_sequence_quadratic_roots
  (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_a : ∀ n, a n ≠ 0)
  (h_arith : arithmetic_sequence a d) :
  (∀ k, ∃ x, quadratic_equation a k x ∧ x = -1) ∧
  (∃ f : ℕ → ℝ, ∀ n, f (n + 1) - f n = -1/2 ∧
    ∃ x, quadratic_equation a n x ∧ f n = 1 / (x + 1)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_roots_l2478_247832


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2478_247817

theorem partial_fraction_decomposition (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ 1 → 
    1 / (x^3 - 2*x^2 - 13*x + 10) = A / (x + 2) + B / (x - 1) + C / ((x - 1)^2)) →
  A = 1/9 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2478_247817


namespace NUMINAMATH_CALUDE_salaria_tree_count_l2478_247862

/-- Represents the types of orange trees --/
inductive TreeType
| A
| B

/-- Calculates the number of good oranges per tree per month --/
def goodOrangesPerTree (t : TreeType) : ℚ :=
  match t with
  | TreeType.A => 10 * (60 / 100)
  | TreeType.B => 15 * (1 / 3)

/-- Calculates the average number of good oranges per tree per month --/
def avgGoodOrangesPerTree : ℚ :=
  (goodOrangesPerTree TreeType.A + goodOrangesPerTree TreeType.B) / 2

/-- The total number of good oranges Salaria gets per month --/
def totalGoodOranges : ℚ := 55

/-- Theorem stating that the total number of trees Salaria has is 10 --/
theorem salaria_tree_count :
  totalGoodOranges / avgGoodOrangesPerTree = 10 := by
  sorry


end NUMINAMATH_CALUDE_salaria_tree_count_l2478_247862


namespace NUMINAMATH_CALUDE_least_five_digit_divisible_by_12_15_18_l2478_247863

theorem least_five_digit_divisible_by_12_15_18 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- 5-digit number
  (n % 12 = 0 ∧ n % 15 = 0 ∧ n % 18 = 0) ∧  -- divisible by 12, 15, and 18
  (∀ m : ℕ, m ≥ 10000 ∧ m < n ∧ m % 12 = 0 ∧ m % 15 = 0 ∧ m % 18 = 0 → false) ∧  -- least such number
  n = 10080 :=  -- the answer
by sorry

end NUMINAMATH_CALUDE_least_five_digit_divisible_by_12_15_18_l2478_247863


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2478_247894

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

-- Theorem statement
theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2478_247894


namespace NUMINAMATH_CALUDE_women_who_left_l2478_247827

theorem women_who_left (initial_men : ℕ) (initial_women : ℕ) (final_men : ℕ) (final_women : ℕ) :
  initial_men * 5 = initial_women * 4 →
  final_men = initial_men + 2 →
  final_men = 14 →
  final_women = 24 →
  final_women = 2 * (initial_women - (initial_women - final_women / 2)) →
  initial_women - final_women / 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_women_who_left_l2478_247827


namespace NUMINAMATH_CALUDE_man_work_time_l2478_247881

/-- The time taken by a man to complete a work given the following conditions:
    - A man, a woman, and a boy together complete the work in 3 days
    - A woman alone can do the work in 6 days
    - A boy alone can do the work in 18 days -/
theorem man_work_time (work : ℝ) (man_rate woman_rate boy_rate : ℝ) :
  work > 0 ∧
  man_rate > 0 ∧ woman_rate > 0 ∧ boy_rate > 0 ∧
  man_rate + woman_rate + boy_rate = work / 3 ∧
  woman_rate = work / 6 ∧
  boy_rate = work / 18 →
  work / man_rate = 9 := by
  sorry

#check man_work_time

end NUMINAMATH_CALUDE_man_work_time_l2478_247881


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2478_247867

theorem opposite_of_2023 : 
  -(2023 : ℤ) = -2023 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2478_247867


namespace NUMINAMATH_CALUDE_price_after_nine_years_l2478_247875

/-- The price of a product after a certain number of three-year periods, given an initial price and a decay rate. -/
def price_after_periods (initial_price : ℝ) (decay_rate : ℝ) (periods : ℕ) : ℝ :=
  initial_price * (1 - decay_rate) ^ periods

/-- Theorem stating that if a product's price decreases by 25% every three years and its current price is 640 yuan, then its price after 9 years will be 270 yuan. -/
theorem price_after_nine_years :
  let initial_price : ℝ := 640
  let decay_rate : ℝ := 0.25
  let periods : ℕ := 3
  price_after_periods initial_price decay_rate periods = 270 := by
  sorry


end NUMINAMATH_CALUDE_price_after_nine_years_l2478_247875


namespace NUMINAMATH_CALUDE_direction_field_properties_l2478_247859

open Real

-- Define the differential equation
def y' (x y : ℝ) : ℝ := x^2 + y^2

-- Theorem statement
theorem direction_field_properties :
  -- 1. Slope at origin is 0
  y' 0 0 = 0 ∧
  -- 2. Slope at (1, 0) is 1
  y' 1 0 = 1 ∧
  -- 3. Slope is 1 for any point on the unit circle
  (∀ x y : ℝ, x^2 + y^2 = 1 → y' x y = 1) ∧
  -- 4. Slope increases as distance from origin increases
  (∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 < x2^2 + y2^2 → y' x1 y1 < y' x2 y2) :=
by sorry

end NUMINAMATH_CALUDE_direction_field_properties_l2478_247859


namespace NUMINAMATH_CALUDE_tv_cost_l2478_247840

theorem tv_cost (mixer_cost tv_cost : ℕ) : 
  (2 * mixer_cost + tv_cost = 7000) → 
  (mixer_cost + 2 * tv_cost = 9800) → 
  tv_cost = 4200 := by
sorry

end NUMINAMATH_CALUDE_tv_cost_l2478_247840


namespace NUMINAMATH_CALUDE_intersection_distance_implies_omega_l2478_247866

/-- Given a function f(x) = 2sin(ωx + φ) where ω > 0, if the curve y = f(x) intersects
    the line y = √3 and the distance between two adjacent intersection points is π/6,
    then ω = 2 or ω = 10. -/
theorem intersection_distance_implies_omega (ω φ : ℝ) (h1 : ω > 0) :
  (∃ x1 x2 : ℝ, x2 - x1 = π / 6 ∧
    2 * Real.sin (ω * x1 + φ) = Real.sqrt 3 ∧
    2 * Real.sin (ω * x2 + φ) = Real.sqrt 3) →
  ω = 2 ∨ ω = 10 := by
  sorry


end NUMINAMATH_CALUDE_intersection_distance_implies_omega_l2478_247866


namespace NUMINAMATH_CALUDE_dance_off_ratio_l2478_247876

-- Define the dancing times and break time
def john_first_dance : ℕ := 3
def john_break : ℕ := 1
def john_second_dance : ℕ := 5
def combined_dance_time : ℕ := 20

-- Define John's total dancing and resting time
def john_total_time : ℕ := john_first_dance + john_break + john_second_dance

-- Define John's dancing time
def john_dance_time : ℕ := john_first_dance + john_second_dance

-- Define James' dancing time
def james_dance_time : ℕ := combined_dance_time - john_dance_time

-- Define James' additional dancing time
def james_additional_time : ℕ := james_dance_time - john_dance_time

-- Theorem to prove
theorem dance_off_ratio : 
  (james_additional_time : ℚ) / john_total_time = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_dance_off_ratio_l2478_247876


namespace NUMINAMATH_CALUDE_count_arrangements_eq_21_l2478_247884

/-- A function that counts the number of valid arrangements of digits 1, 1, 2, 5, 0 -/
def countArrangements : ℕ :=
  let digits : List ℕ := [1, 1, 2, 5, 0]
  let isValidArrangement (arr : List ℕ) : Bool :=
    arr.length = 5 ∧ 
    arr.head? ≠ some 0 ∧ 
    (arr.getLast? = some 0 ∨ arr.getLast? = some 5)

  -- Count valid arrangements
  sorry

/-- The theorem stating that the number of valid arrangements is 21 -/
theorem count_arrangements_eq_21 : countArrangements = 21 := by
  sorry

end NUMINAMATH_CALUDE_count_arrangements_eq_21_l2478_247884


namespace NUMINAMATH_CALUDE_b_bounds_l2478_247830

theorem b_bounds (a : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) :
  let b := a^3 + 1 / (1 + a)
  (b ≥ 1 - a + a^2) ∧ (3/4 < b) ∧ (b ≤ 3/2) := by sorry

end NUMINAMATH_CALUDE_b_bounds_l2478_247830


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2478_247892

theorem triangle_angle_measure (A B C : ℝ) (exterior_angle : ℝ) :
  -- An exterior angle of triangle ABC is 110°
  exterior_angle = 110 →
  -- ∠A = ∠B
  A = B →
  -- Triangle inequality (to ensure it's a valid triangle)
  A + B + C = 180 →
  -- Prove that ∠A is either 70° or 55°
  A = 70 ∨ A = 55 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2478_247892


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2478_247806

/-- 
Given a line y = mx + b, if the reflection of point (-4, 0) across this line 
is (2, 6), then m + b = 1.
-/
theorem reflection_line_sum (m b : ℝ) : 
  (∀ (x y : ℝ), y = m * x + b → 
    (x = -1 ∧ y = 3) ↔ 
    (x = ((-4 + 2) / 2) ∧ y = ((0 + 6) / 2))) →
  (m = -1) →
  (m + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2478_247806


namespace NUMINAMATH_CALUDE_combined_teaching_experience_l2478_247844

/-- Given two teachers, James and his partner, where James has taught for 40 years
    and his partner has taught for 10 years less than James, 
    their combined teaching experience is 70 years. -/
theorem combined_teaching_experience : 
  ∀ (james_experience partner_experience : ℕ),
  james_experience = 40 →
  partner_experience = james_experience - 10 →
  james_experience + partner_experience = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_teaching_experience_l2478_247844


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_6_sqrt_3_l2478_247802

theorem sqrt_sum_equals_6_sqrt_3 :
  Real.sqrt (31 - 12 * Real.sqrt 3) + Real.sqrt (31 + 12 * Real.sqrt 3) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_6_sqrt_3_l2478_247802


namespace NUMINAMATH_CALUDE_correct_product_after_reversal_error_l2478_247819

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ := 
  (n % 10) * 10 + (n / 10)

theorem correct_product_after_reversal_error (a b : ℕ) : 
  is_two_digit a → 
  is_two_digit b → 
  reverse_digits a * b = 378 → 
  a * b = 504 := by
  sorry

end NUMINAMATH_CALUDE_correct_product_after_reversal_error_l2478_247819
