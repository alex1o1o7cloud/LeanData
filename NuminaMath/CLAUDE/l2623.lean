import Mathlib

namespace NUMINAMATH_CALUDE_percent_to_decimal_five_percent_to_decimal_l2623_262309

theorem percent_to_decimal (p : ℝ) : p / 100 = p * 0.01 := by sorry

theorem five_percent_to_decimal : (5 : ℝ) / 100 = 0.05 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_five_percent_to_decimal_l2623_262309


namespace NUMINAMATH_CALUDE_range_of_a_in_p_a_neither_necessary_nor_sufficient_for_b_l2623_262399

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define sets A and B
def A : Set ℝ := {a | a ≤ 1}
def B : Set ℝ := {a | a ≥ 1 ∨ a ≤ -2}

-- Theorem for the range of a in proposition p
theorem range_of_a_in_p : ∀ a : ℝ, p a ↔ a ∈ A := by sorry

-- Theorem for the relationship between A and B
theorem a_neither_necessary_nor_sufficient_for_b :
  (¬∀ a : ℝ, a ∈ B → a ∈ A) ∧ (¬∀ a : ℝ, a ∈ A → a ∈ B) := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_p_a_neither_necessary_nor_sufficient_for_b_l2623_262399


namespace NUMINAMATH_CALUDE_correct_system_l2623_262301

/-- Represents the price of a horse in taels -/
def horse_price : ℝ := sorry

/-- Represents the price of a head of cattle in taels -/
def cattle_price : ℝ := sorry

/-- The total price of 4 horses and 6 heads of cattle is 48 taels -/
axiom eq1 : 4 * horse_price + 6 * cattle_price = 48

/-- The total price of 3 horses and 5 heads of cattle is 38 taels -/
axiom eq2 : 3 * horse_price + 5 * cattle_price = 38

/-- The system of equations correctly represents the given conditions -/
theorem correct_system : 
  (4 * horse_price + 6 * cattle_price = 48) ∧ 
  (3 * horse_price + 5 * cattle_price = 38) :=
sorry

end NUMINAMATH_CALUDE_correct_system_l2623_262301


namespace NUMINAMATH_CALUDE_sample_size_calculation_l2623_262366

/-- Represents the staff composition in a company -/
structure StaffComposition where
  sales : ℕ
  management : ℕ
  logistics : ℕ

/-- Represents a stratified sample from the company -/
structure StratifiedSample where
  total_size : ℕ
  sales_size : ℕ

/-- The theorem stating the relationship between the company's staff composition,
    the number of sales staff in the sample, and the total sample size -/
theorem sample_size_calculation 
  (company : StaffComposition)
  (sample : StratifiedSample)
  (h1 : company.sales = 15)
  (h2 : company.management = 3)
  (h3 : company.logistics = 2)
  (h4 : sample.sales_size = 30) :
  sample.total_size = 40 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l2623_262366


namespace NUMINAMATH_CALUDE_heartsuit_problem_l2623_262300

def heartsuit (a b : ℝ) : ℝ := |a + b|

theorem heartsuit_problem : heartsuit (-3) (heartsuit 5 (-8)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_problem_l2623_262300


namespace NUMINAMATH_CALUDE_probability_at_least_one_non_defective_l2623_262323

theorem probability_at_least_one_non_defective (p_defective : ℝ) (h_p : p_defective = 0.3) :
  let p_all_defective := p_defective ^ 3
  let p_at_least_one_non_defective := 1 - p_all_defective
  p_at_least_one_non_defective = 0.973 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_non_defective_l2623_262323


namespace NUMINAMATH_CALUDE_trig_expression_value_l2623_262362

theorem trig_expression_value (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l2623_262362


namespace NUMINAMATH_CALUDE_unique_nine_digit_number_l2623_262386

def is_nine_digit (n : ℕ) : Prop := 100000000 ≤ n ∧ n ≤ 999999999

def sum_of_digits (n : ℕ) : ℕ := sorry

def product_of_digits (n : ℕ) : ℕ := sorry

def round_to_millions (n : ℕ) : ℕ := sorry

theorem unique_nine_digit_number :
  ∃! n : ℕ,
    is_nine_digit n ∧
    n % 2 = 1 ∧
    sum_of_digits n = 10 ∧
    product_of_digits n ≠ 0 ∧
    n % 7 = 0 ∧
    round_to_millions n = 112 :=
by sorry

end NUMINAMATH_CALUDE_unique_nine_digit_number_l2623_262386


namespace NUMINAMATH_CALUDE_fixed_point_on_moving_line_intersecting_parabola_l2623_262395

/-- Theorem: Fixed point on a moving line intersecting a parabola -/
theorem fixed_point_on_moving_line_intersecting_parabola
  (p : ℝ) (k : ℝ) (b : ℝ)
  (hp : p > 0)
  (hk : k ≠ 0)
  (hb : b ≠ 0)
  (h_slope_product : ∀ x₁ y₁ x₂ y₂ : ℝ,
    y₁^2 = 2*p*x₁ → y₂^2 = 2*p*x₂ →
    y₁ = k*x₁ + b → y₂ = k*x₂ + b →
    (y₁ / x₁) * (y₂ / x₂) = Real.sqrt 3) :
  let fixed_point : ℝ × ℝ := (-2*p/Real.sqrt 3, 0)
  ∃ b' : ℝ, k * fixed_point.1 + b' = fixed_point.2 ∧ b' = 2*p*k/Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_fixed_point_on_moving_line_intersecting_parabola_l2623_262395


namespace NUMINAMATH_CALUDE_circular_garden_area_l2623_262326

/-- Proves that a circular garden with radius 6 and fence length equal to 1/3 of its area has an area of 36π square units -/
theorem circular_garden_area (r : ℝ) (h1 : r = 6) : 
  (2 * Real.pi * r = (1/3) * Real.pi * r^2) → Real.pi * r^2 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_area_l2623_262326


namespace NUMINAMATH_CALUDE_total_rice_weight_l2623_262341

-- Define the number of containers
def num_containers : ℕ := 4

-- Define the weight of rice in each container (in ounces)
def rice_per_container : ℝ := 29

-- Define the conversion rate from ounces to pounds
def ounces_per_pound : ℝ := 16

-- State the theorem
theorem total_rice_weight :
  (num_containers : ℝ) * rice_per_container / ounces_per_pound = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_total_rice_weight_l2623_262341


namespace NUMINAMATH_CALUDE_wednesday_earnings_l2623_262347

/-- Represents the types of lawns --/
inductive LawnType
| Small
| Medium
| Large

/-- Represents the charge rates for different lawn types --/
def charge_rate (lt : LawnType) : ℕ :=
  match lt with
  | LawnType.Small => 5
  | LawnType.Medium => 7
  | LawnType.Large => 10

/-- Extra fee for lawns with large piles of leaves --/
def large_pile_fee : ℕ := 3

/-- Calculates the earnings for a given day --/
def daily_earnings (small_bags medium_bags large_bags large_piles : ℕ) : ℕ :=
  small_bags * charge_rate LawnType.Small +
  medium_bags * charge_rate LawnType.Medium +
  large_bags * charge_rate LawnType.Large +
  large_piles * large_pile_fee

/-- Represents the work done on Monday --/
def monday_work : ℕ := daily_earnings 4 2 1 1

/-- Represents the work done on Tuesday --/
def tuesday_work : ℕ := daily_earnings 2 1 2 1

/-- Total earnings after three days --/
def total_earnings : ℕ := 163

/-- Theorem stating that Wednesday's earnings are $76 --/
theorem wednesday_earnings :
  total_earnings - (monday_work + tuesday_work) = 76 := by sorry

end NUMINAMATH_CALUDE_wednesday_earnings_l2623_262347


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2623_262340

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (n + (n + 4) = 130) → (n + (n + 2) + (n + 4) = 195) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2623_262340


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l2623_262371

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ - a₁ = d ∧ a₁ - (-9) = d ∧ (-1) - a₂ = d

-- Define the geometric sequence
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ / (-9) = r ∧ b₂ / b₁ = r ∧ b₃ / b₂ = r ∧ (-1) / b₃ = r

-- State the theorem
theorem arithmetic_geometric_sequence_relation :
  ∀ a₁ a₂ b₁ b₂ b₃ : ℝ,
  arithmetic_sequence a₁ a₂ →
  geometric_sequence b₁ b₂ b₃ →
  a₂ * b₂ - a₁ * b₂ = -8 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l2623_262371


namespace NUMINAMATH_CALUDE_quadratic_unique_root_l2623_262394

/-- Given real numbers p, q, r forming an arithmetic sequence with p ≥ q ≥ r ≥ 0,
    if the quadratic px^2 + qx + r has exactly one root, then this root is equal to 1 - √6/2 -/
theorem quadratic_unique_root (p q r : ℝ) 
  (arith_seq : ∃ k, q = p - k ∧ r = p - 2*k)
  (order : p ≥ q ∧ q ≥ r ∧ r ≥ 0)
  (unique_root : ∃! x, p*x^2 + q*x + r = 0) :
  ∃ x, p*x^2 + q*x + r = 0 ∧ x = 1 - Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_unique_root_l2623_262394


namespace NUMINAMATH_CALUDE_map_width_l2623_262319

/-- Given a rectangular map with area 10 square meters and length 5 meters, prove its width is 2 meters. -/
theorem map_width (area : ℝ) (length : ℝ) (width : ℝ) 
    (h_area : area = 10) 
    (h_length : length = 5) 
    (h_rectangle : area = length * width) : width = 2 := by
  sorry

end NUMINAMATH_CALUDE_map_width_l2623_262319


namespace NUMINAMATH_CALUDE_non_seniors_playing_instrument_l2623_262374

theorem non_seniors_playing_instrument (total_students : ℕ) 
  (senior_play_percent : ℚ) (non_senior_not_play_percent : ℚ) 
  (total_not_play_percent : ℚ) :
  total_students = 500 →
  senior_play_percent = 2/5 →
  non_senior_not_play_percent = 3/10 →
  total_not_play_percent = 234/500 →
  ∃ (seniors non_seniors : ℕ),
    seniors + non_seniors = total_students ∧
    (seniors : ℚ) * (1 - senior_play_percent) + 
    (non_seniors : ℚ) * non_senior_not_play_percent = 
    (total_students : ℚ) * total_not_play_percent ∧
    (non_seniors : ℚ) * (1 - non_senior_not_play_percent) = 154 :=
by sorry

end NUMINAMATH_CALUDE_non_seniors_playing_instrument_l2623_262374


namespace NUMINAMATH_CALUDE_amount_r_has_l2623_262338

theorem amount_r_has (total : ℝ) (r_fraction : ℝ) (h1 : total = 4000) (h2 : r_fraction = 2/3) : 
  let amount_pq := total / (1 + r_fraction)
  let amount_r := r_fraction * amount_pq
  amount_r = 1600 := by sorry

end NUMINAMATH_CALUDE_amount_r_has_l2623_262338


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2623_262379

theorem quadratic_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≠ 0) → -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2623_262379


namespace NUMINAMATH_CALUDE_complex_simplification_l2623_262358

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- The statement to prove -/
theorem complex_simplification : 7 * (2 - 3 * i) + 4 * i * (3 - 2 * i) = 22 - 9 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2623_262358


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2623_262392

theorem complex_equation_solution (z : ℂ) :
  (2 - 3*I)*z = 5 - I → z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2623_262392


namespace NUMINAMATH_CALUDE_hit_rate_problem_l2623_262348

theorem hit_rate_problem (p_at_least_one p_a p_b : ℝ) : 
  p_at_least_one = 0.7 →
  p_a = 0.4 →
  p_at_least_one = p_a + p_b - p_a * p_b →
  p_b = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_hit_rate_problem_l2623_262348


namespace NUMINAMATH_CALUDE_system_solution_unique_l2623_262387

theorem system_solution_unique (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 / x + 2 / y = 4 ∧ 5 / x - 6 / y = 2) ↔ (x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2623_262387


namespace NUMINAMATH_CALUDE_woman_work_time_l2623_262339

-- Define the work rate of one man
def man_rate : ℚ := 1 / 100

-- Define the total work (1 unit)
def total_work : ℚ := 1

-- Define the time taken by 10 men and 15 women
def combined_time : ℚ := 5

-- Define the number of men and women
def num_men : ℕ := 10
def num_women : ℕ := 15

-- Define the work rate of one woman
noncomputable def woman_rate : ℚ := 
  (total_work / combined_time - num_men * man_rate) / num_women

-- Theorem: One woman alone will take 150 days to complete the work
theorem woman_work_time : total_work / woman_rate = 150 := by sorry

end NUMINAMATH_CALUDE_woman_work_time_l2623_262339


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2623_262382

-- Define the universal set U
def U : Set ℝ := {x : ℝ | x ≤ 1}

-- Define set A
def A : Set ℝ := {x : ℝ | x < 0}

-- Theorem statement
theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2623_262382


namespace NUMINAMATH_CALUDE_max_integer_difference_l2623_262368

theorem max_integer_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) : 
  (∀ a b : ℤ, 7 < a ∧ a < 9 ∧ 9 < b ∧ b < 15 → y - x ≥ b - a) ∧ y - x = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_integer_difference_l2623_262368


namespace NUMINAMATH_CALUDE_fraction_enlargement_l2623_262302

theorem fraction_enlargement (x y : ℝ) (h : x ≠ y) :
  (3 * x) * (3 * y) / ((3 * x) - (3 * y)) = 3 * (x * y / (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_enlargement_l2623_262302


namespace NUMINAMATH_CALUDE_cos_angle_relation_l2623_262336

theorem cos_angle_relation (α : ℝ) (h : Real.cos (α + π/3) = 4/5) :
  Real.cos (π/3 - 2*α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_relation_l2623_262336


namespace NUMINAMATH_CALUDE_profit_percent_for_given_ratio_l2623_262375

/-- If the ratio of cost price to selling price is 4:5, then the profit percent is 25% -/
theorem profit_percent_for_given_ratio : 
  ∀ (cp sp : ℝ), cp > 0 → sp > 0 → cp / sp = 4 / 5 → (sp - cp) / cp * 100 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_percent_for_given_ratio_l2623_262375


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l2623_262327

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Defines a line in 2D space using the equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : IntersectionPoint) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- The theorem stating that (2, 1) is the unique intersection point of y = 3 - x and y = 3x - 5 -/
theorem intersection_point_of_lines :
  ∃! p : IntersectionPoint, 
    (pointOnLine p ⟨-1, 3⟩) ∧ (pointOnLine p ⟨3, -5⟩) ∧ p.x = 2 ∧ p.y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l2623_262327


namespace NUMINAMATH_CALUDE_equal_triangle_areas_l2623_262380

-- Define the points
variable (A B C D E F K L : Point)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the intersection points E and F
def E_is_intersection (A B C D E : Point) : Prop := sorry
def F_is_intersection (A B C D F : Point) : Prop := sorry

-- Define K and L as midpoints of diagonals
def K_is_midpoint (A C K : Point) : Prop := sorry
def L_is_midpoint (B D L : Point) : Prop := sorry

-- Define the area of a triangle
def area (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem equal_triangle_areas 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : E_is_intersection A B C D E)
  (h3 : F_is_intersection A B C D F)
  (h4 : K_is_midpoint A C K)
  (h5 : L_is_midpoint B D L) :
  area E K L = area F K L := by sorry

end NUMINAMATH_CALUDE_equal_triangle_areas_l2623_262380


namespace NUMINAMATH_CALUDE_unique_virtual_square_plus_one_l2623_262356

def is_virtual (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * a + b

def is_square_plus_one (n : ℕ) : Prop :=
  ∃ (m : ℕ), n = m^2 + 1

theorem unique_virtual_square_plus_one :
  ∃! (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ is_virtual n ∧ is_square_plus_one n ∧ n = 8282 :=
sorry

end NUMINAMATH_CALUDE_unique_virtual_square_plus_one_l2623_262356


namespace NUMINAMATH_CALUDE_square_9801_property_l2623_262352

theorem square_9801_property (y : ℤ) (h : y^2 = 9801) : (y + 2) * (y - 2) = 9797 := by
  sorry

end NUMINAMATH_CALUDE_square_9801_property_l2623_262352


namespace NUMINAMATH_CALUDE_remainder_8547_mod_9_l2623_262313

theorem remainder_8547_mod_9 : 8547 % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8547_mod_9_l2623_262313


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2623_262397

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 2 * Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = Real.sqrt 76 ∧ θ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2623_262397


namespace NUMINAMATH_CALUDE_reflection_of_M_l2623_262333

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (- p.1, p.2)

/-- The original point M -/
def M : ℝ × ℝ := (-5, 2)

theorem reflection_of_M :
  reflect_y M = (5, 2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_l2623_262333


namespace NUMINAMATH_CALUDE_intersection_equals_N_l2623_262364

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem intersection_equals_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_intersection_equals_N_l2623_262364


namespace NUMINAMATH_CALUDE_arrangement_remainder_l2623_262351

/-- The number of green marbles -/
def green_marbles : ℕ := 5

/-- The maximum number of blue marbles that satisfies the arrangement condition -/
def max_blue_marbles : ℕ := 15

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + max_blue_marbles

/-- The number of ways to arrange the marbles satisfying the condition -/
def arrangement_count : ℕ := (Nat.choose (max_blue_marbles + green_marbles) green_marbles)

/-- Theorem stating that the remainder when dividing the number of arrangements by 1000 is 3 -/
theorem arrangement_remainder : arrangement_count % 1000 = 3 := by sorry

end NUMINAMATH_CALUDE_arrangement_remainder_l2623_262351


namespace NUMINAMATH_CALUDE_park_not_crowded_implies_cool_or_rain_l2623_262303

variable (day : Type) -- Type representing days

-- Define predicates for weather conditions and park status
variable (temp_at_least_70 : day → Prop) -- Temperature is at least 70°F
variable (raining : day → Prop) -- It is raining
variable (crowded : day → Prop) -- The park is crowded

-- Given condition: If temp ≥ 70°F and not raining, then the park is crowded
variable (h : ∀ d : day, (temp_at_least_70 d ∧ ¬raining d) → crowded d)

theorem park_not_crowded_implies_cool_or_rain :
  ∀ d : day, ¬crowded d → (¬temp_at_least_70 d ∨ raining d) :=
by
  sorry

#check park_not_crowded_implies_cool_or_rain

end NUMINAMATH_CALUDE_park_not_crowded_implies_cool_or_rain_l2623_262303


namespace NUMINAMATH_CALUDE_m_range_l2623_262359

def p (m : ℝ) : Prop := ∀ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x > m

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m*x + 1 ≤ 0

theorem m_range (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m = -2 ∨ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_m_range_l2623_262359


namespace NUMINAMATH_CALUDE_water_amount_depends_on_time_l2623_262328

/-- Represents the water amount in the reservoir -/
def water_amount (t : ℝ) : ℝ := 50 - 2 * t

/-- States that water_amount is a function of time -/
theorem water_amount_depends_on_time :
  ∃ (f : ℝ → ℝ), ∀ t, water_amount t = f t :=
sorry

end NUMINAMATH_CALUDE_water_amount_depends_on_time_l2623_262328


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2623_262324

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2623_262324


namespace NUMINAMATH_CALUDE_odd_number_1991_in_group_32_l2623_262381

/-- The n-th group of odd numbers contains (2n-1) numbers -/
def group_size (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of odd numbers up to the n-th group -/
def sum_up_to_group (n : ℕ) : ℕ := n^2

/-- The position of 1991 in the sequence of odd numbers -/
def target : ℕ := 1991

/-- The theorem stating that 1991 is in the 32nd group -/
theorem odd_number_1991_in_group_32 :
  ∃ (n : ℕ), n = 32 ∧ 
  sum_up_to_group (n - 1) < target ∧ 
  target ≤ sum_up_to_group n :=
sorry

end NUMINAMATH_CALUDE_odd_number_1991_in_group_32_l2623_262381


namespace NUMINAMATH_CALUDE_sets_intersection_empty_l2623_262325

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x < 2}

theorem sets_intersection_empty : A ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_sets_intersection_empty_l2623_262325


namespace NUMINAMATH_CALUDE_equation_solution_set_l2623_262315

theorem equation_solution_set : ∀ (x y : ℝ), 
  ((x = 1 ∧ y = 3/2) ∨ 
   (x = 1 ∧ y = -1/2) ∨ 
   (x = -1 ∧ y = 3/2) ∨ 
   (x = -1 ∧ y = -1/2)) ↔ 
  4 * x^2 * y^2 = 4 * x * y + 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_set_l2623_262315


namespace NUMINAMATH_CALUDE_curvilinear_trapezoid_area_l2623_262349

-- Define the function representing the parabola
def f (x : ℝ) : ℝ := 9 - x^2

-- Define the integral bounds
def a : ℝ := -1
def b : ℝ := 2

-- State the theorem
theorem curvilinear_trapezoid_area : 
  ∫ x in a..b, f x = 24 := by sorry

end NUMINAMATH_CALUDE_curvilinear_trapezoid_area_l2623_262349


namespace NUMINAMATH_CALUDE_y_days_to_finish_work_l2623_262391

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 18

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 10

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 6

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

/-- Theorem stating that given the conditions, y needs 15 days to finish the work alone -/
theorem y_days_to_finish_work : 
  (1 / x_days) * x_remaining = 1 - (y_worked / y_days) := by sorry

end NUMINAMATH_CALUDE_y_days_to_finish_work_l2623_262391


namespace NUMINAMATH_CALUDE_pentagon_cannot_tile_l2623_262365

/-- Represents a regular polygon --/
inductive RegularPolygon
  | EquilateralTriangle
  | Square
  | Pentagon
  | Hexagon

/-- Calculates the interior angle of a regular polygon with n sides --/
def interiorAngle (n : ℕ) : ℚ :=
  180 - (360 / n)

/-- Checks if a polygon can tile the plane --/
def canTilePlane (p : RegularPolygon) : Prop :=
  match p with
  | RegularPolygon.EquilateralTriangle => (360 / interiorAngle 3).isInt
  | RegularPolygon.Square => (360 / interiorAngle 4).isInt
  | RegularPolygon.Pentagon => (360 / interiorAngle 5).isInt
  | RegularPolygon.Hexagon => (360 / interiorAngle 6).isInt

/-- Theorem stating that only the pentagon cannot tile the plane --/
theorem pentagon_cannot_tile :
  ∀ p : RegularPolygon,
    ¬(canTilePlane p) ↔ p = RegularPolygon.Pentagon :=
by sorry

end NUMINAMATH_CALUDE_pentagon_cannot_tile_l2623_262365


namespace NUMINAMATH_CALUDE_payment_plan_difference_l2623_262337

theorem payment_plan_difference (original_price down_payment num_payments payment_amount : ℕ) :
  original_price = 1500 ∧
  down_payment = 200 ∧
  num_payments = 24 ∧
  payment_amount = 65 →
  (down_payment + num_payments * payment_amount) - original_price = 260 := by
  sorry

end NUMINAMATH_CALUDE_payment_plan_difference_l2623_262337


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l2623_262306

theorem smallest_gcd_bc (a b c : ℕ+) (hab : Nat.gcd a b = 210) (hac : Nat.gcd a c = 770) :
  (∀ d : ℕ+, ∃ a' b' c' : ℕ+, Nat.gcd a' b' = 210 ∧ Nat.gcd a' c' = 770 ∧ Nat.gcd b' c' = d) →
  10 ≤ Nat.gcd b c :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l2623_262306


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2623_262376

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- State the theorem
theorem tangent_line_sum (h : tangent_line 1 (f 1)) : f 1 + deriv f 1 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2623_262376


namespace NUMINAMATH_CALUDE_probability_green_in_specific_bag_l2623_262353

structure Bag where
  total_balls : ℕ
  green_balls : ℕ
  white_balls : ℕ

def probability_green (b : Bag) : ℚ :=
  b.green_balls / b.total_balls

theorem probability_green_in_specific_bag : 
  ∃ (b : Bag), b.total_balls = 9 ∧ b.green_balls = 7 ∧ b.white_balls = 2 ∧ 
    probability_green b = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_in_specific_bag_l2623_262353


namespace NUMINAMATH_CALUDE_problem_statement_l2623_262370

theorem problem_statement :
  (∀ x : ℝ, (2/3 ≤ x ∧ x ≤ 2) → (-1 ≤ x ∧ x ≤ 2)) ∧
  (∃ x : ℝ, (-1 ≤ x ∧ x ≤ 2) ∧ ¬(2/3 ≤ x ∧ x ≤ 2)) ∧
  (∀ a : ℝ, (∀ x : ℝ, (x ≤ a ∨ x ≥ a + 1) → (2/3 ≤ x ∧ x ≤ 2)) ↔ (a ≥ 2 ∨ a ≤ -1/3)) :=
by sorry


end NUMINAMATH_CALUDE_problem_statement_l2623_262370


namespace NUMINAMATH_CALUDE_value_of_a_l2623_262311

theorem value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 7 * a) : a = 15 / 11 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2623_262311


namespace NUMINAMATH_CALUDE_solution_l2623_262396

def problem (f : ℝ → ℝ) : Prop :=
  (∀ x, f x * f (x + 2) = 13) ∧ f 1 = 2

theorem solution (f : ℝ → ℝ) (h : problem f) : f 2015 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_l2623_262396


namespace NUMINAMATH_CALUDE_max_regions_formula_l2623_262398

/-- The maximum number of regions delimited by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem stating the maximum number of regions delimited by n lines in a plane -/
theorem max_regions_formula (n : ℕ) :
  max_regions n = 1 + n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_formula_l2623_262398


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_415_l2623_262322

-- Define the set of angles with the same terminal side as -415°
def same_terminal_side (β : ℝ) : Prop :=
  ∃ k : ℤ, β = k * 360 - 415

-- Define the condition for the angle to be between 0° and 360°
def between_0_and_360 (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

-- Theorem statement
theorem angle_with_same_terminal_side_as_negative_415 :
  ∃ θ : ℝ, same_terminal_side θ ∧ between_0_and_360 θ ∧ θ = 305 :=
by sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_415_l2623_262322


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2623_262350

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a)
  (h_a1 : a 1 = 2)
  (h_sum : a 3 + a 5 = 10) :
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2623_262350


namespace NUMINAMATH_CALUDE_probability_two_heads_in_three_flips_l2623_262345

/-- The probability of getting exactly k successes in n trials with probability p for each trial -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- A fair coin has probability 0.5 of landing heads -/
def fair_coin_probability : ℝ := 0.5

/-- The number of flips -/
def num_flips : ℕ := 3

/-- The number of heads we want -/
def num_heads : ℕ := 2

theorem probability_two_heads_in_three_flips :
  binomial_probability num_flips num_heads fair_coin_probability = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_in_three_flips_l2623_262345


namespace NUMINAMATH_CALUDE_bridge_crossing_time_l2623_262378

/-- Proves that a man walking at 9 km/hr takes 15 minutes to cross a bridge of 2250 meters in length -/
theorem bridge_crossing_time (walking_speed : ℝ) (bridge_length : ℝ) :
  walking_speed = 9 →
  bridge_length = 2250 →
  (bridge_length / (walking_speed * 1000 / 60)) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_crossing_time_l2623_262378


namespace NUMINAMATH_CALUDE_find_z_when_y_is_4_l2623_262330

-- Define the relationship between y and z
def inverse_variation (y z : ℝ) (k : ℝ) : Prop :=
  y^3 * z^(1/3) = k

-- Theorem statement
theorem find_z_when_y_is_4 (y z : ℝ) (k : ℝ) :
  inverse_variation 2 1 k →
  inverse_variation 4 z k →
  z = 1 / 512 :=
by
  sorry

end NUMINAMATH_CALUDE_find_z_when_y_is_4_l2623_262330


namespace NUMINAMATH_CALUDE_solution_set_f_gt_3_l2623_262320

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x else Real.log x + 2

theorem solution_set_f_gt_3 :
  {x : ℝ | f x > 3} = {x : ℝ | x < -3 ∨ x > Real.exp 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_3_l2623_262320


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l2623_262360

/-- Proves that the volume of cylinder C is (1/9) π h³ given the specified conditions --/
theorem cylinder_volume_relation (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  h = 3 * r →  -- Height of C is three times radius of D
  r = h →      -- Radius of D is equal to height of C
  π * r^2 * h = 3 * (π * h^2 * r) →  -- Volume of C is three times volume of D
  π * r^2 * h = (1/9) * π * h^3 :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l2623_262360


namespace NUMINAMATH_CALUDE_sum_of_integers_l2623_262305

theorem sum_of_integers (a b : ℕ) (h1 : a > b) (h2 : a - b = 5) (h3 : a * b = 84) : a + b = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2623_262305


namespace NUMINAMATH_CALUDE_turtle_problem_l2623_262307

theorem turtle_problem (initial_turtles : ℕ) (h1 : initial_turtles = 9) : 
  let new_turtles := 3 * initial_turtles - 2
  let total_turtles := initial_turtles + new_turtles
  let remaining_turtles := total_turtles / 2
  remaining_turtles = 17 := by
sorry

end NUMINAMATH_CALUDE_turtle_problem_l2623_262307


namespace NUMINAMATH_CALUDE_micah_envelope_count_l2623_262393

def envelope_count (total_stamps : ℕ) (light_envelopes : ℕ) (stamps_per_light : ℕ) (stamps_per_heavy : ℕ) : ℕ :=
  let heavy_stamps := total_stamps - light_envelopes * stamps_per_light
  let heavy_envelopes := heavy_stamps / stamps_per_heavy
  light_envelopes + heavy_envelopes

theorem micah_envelope_count :
  envelope_count 52 6 2 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_micah_envelope_count_l2623_262393


namespace NUMINAMATH_CALUDE_equal_coins_after_transfer_l2623_262388

/-- Represents the amount of gold coins each merchant has -/
structure Merchants where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (m : Merchants) : Prop :=
  (m.ierema + 70 = m.yuliy) ∧ (m.foma - 40 = m.yuliy)

/-- The theorem to prove -/
theorem equal_coins_after_transfer (m : Merchants) 
  (h : satisfies_conditions m) : 
  m.foma - 55 = m.ierema + 55 := by
  sorry

#check equal_coins_after_transfer

end NUMINAMATH_CALUDE_equal_coins_after_transfer_l2623_262388


namespace NUMINAMATH_CALUDE_blood_expires_same_day_l2623_262343

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The factorial of 7 -/
def blood_expiration_seconds : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

/-- Proposition: Blood donated at noon expires on the same day -/
theorem blood_expires_same_day : blood_expiration_seconds < seconds_per_day := by
  sorry

#eval blood_expiration_seconds
#eval seconds_per_day

end NUMINAMATH_CALUDE_blood_expires_same_day_l2623_262343


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_min_value_sum_fractions_achieved_l2623_262308

theorem min_value_sum_fractions (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c) / d + (a + b + d) / c + (a + c + d) / b + (b + c + d) / a ≥ 12 :=
by sorry

theorem min_value_sum_fractions_achieved (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∃ (x : ℝ), x > 0 ∧ 
    (x + x + x) / x + (x + x + x) / x + (x + x + x) / x + (x + x + x) / x = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_min_value_sum_fractions_achieved_l2623_262308


namespace NUMINAMATH_CALUDE_sharpshooter_target_orders_l2623_262389

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multisetPermutations (total : ℕ) (counts : List ℕ) : ℕ :=
  factorial total / (counts.map factorial).prod

theorem sharpshooter_target_orders : 
  let total_targets : ℕ := 8
  let column_targets : List ℕ := [2, 3, 2, 1]
  multisetPermutations total_targets column_targets = 1680 := by
  sorry

end NUMINAMATH_CALUDE_sharpshooter_target_orders_l2623_262389


namespace NUMINAMATH_CALUDE_b_range_l2623_262384

theorem b_range (b : ℝ) : (∀ x : ℝ, x^2 + b*x + b > 0) → b ∈ Set.Ioo 0 4 := by
  sorry

end NUMINAMATH_CALUDE_b_range_l2623_262384


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l2623_262354

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_m_value :
  ∀ m : ℝ, collinear (-2) 12 1 3 m (-6) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l2623_262354


namespace NUMINAMATH_CALUDE_car_collision_frequency_l2623_262385

theorem car_collision_frequency :
  ∀ (x : ℝ),
    (x > 0) →
    (240 / x + 240 / 20 = 36) →
    x = 10 :=
by
  sorry

#check car_collision_frequency

end NUMINAMATH_CALUDE_car_collision_frequency_l2623_262385


namespace NUMINAMATH_CALUDE_correct_factorizations_l2623_262321

theorem correct_factorizations (x y : ℝ) : 
  (x^2 + x*y + y^2 ≠ (x + y)^2) ∧ 
  (-x^2 + 2*x*y - y^2 = -(x - y)^2) ∧ 
  (x^2 + 6*x*y - 9*y^2 ≠ (x - 3*y)^2) ∧ 
  (-x^2 + 1/4 = (1/2 + x)*(1/2 - x)) := by
sorry

end NUMINAMATH_CALUDE_correct_factorizations_l2623_262321


namespace NUMINAMATH_CALUDE_movie_theater_revenue_l2623_262383

/-- Calculates the total revenue of a movie theater given ticket sales and pricing information. -/
def calculate_total_revenue (
  matinee_price : ℚ)
  (evening_price : ℚ)
  (threeD_price : ℚ)
  (evening_group_discount : ℚ)
  (threeD_online_surcharge : ℚ)
  (early_bird_discount : ℚ)
  (matinee_tickets : ℕ)
  (early_bird_tickets : ℕ)
  (evening_tickets : ℕ)
  (evening_group_tickets : ℕ)
  (threeD_tickets : ℕ)
  (threeD_online_tickets : ℕ) : ℚ :=
  sorry

theorem movie_theater_revenue :
  let matinee_price : ℚ := 5
  let evening_price : ℚ := 12
  let threeD_price : ℚ := 20
  let evening_group_discount : ℚ := 0.1
  let threeD_online_surcharge : ℚ := 2
  let early_bird_discount : ℚ := 0.5
  let matinee_tickets : ℕ := 200
  let early_bird_tickets : ℕ := 20
  let evening_tickets : ℕ := 300
  let evening_group_tickets : ℕ := 150
  let threeD_tickets : ℕ := 100
  let threeD_online_tickets : ℕ := 60
  calculate_total_revenue
    matinee_price evening_price threeD_price
    evening_group_discount threeD_online_surcharge early_bird_discount
    matinee_tickets early_bird_tickets evening_tickets
    evening_group_tickets threeD_tickets threeD_online_tickets = 6490 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_revenue_l2623_262383


namespace NUMINAMATH_CALUDE_parallel_condition_l2623_262369

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y + 6 = 0

-- Define the parallel relation between two lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_condition (a : ℝ) :
  (parallel (l₁ a) (l₂ a) → (a = 2 ∨ a = -2)) ∧
  ¬(a = 2 ∨ a = -2 → parallel (l₁ a) (l₂ a)) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l2623_262369


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l2623_262390

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_t_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, t)
  parallel a b → t = -4 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l2623_262390


namespace NUMINAMATH_CALUDE_x_y_inequality_l2623_262367

theorem x_y_inequality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + 2 * |y| = 2 * x * y) : 
  (x > 0 → x + y > 3) ∧ (x < 0 → x + y < -3) := by
  sorry

end NUMINAMATH_CALUDE_x_y_inequality_l2623_262367


namespace NUMINAMATH_CALUDE_distribute_four_men_five_women_l2623_262357

/-- The number of ways to distribute men and women into groups -/
def distribute_people (num_men num_women : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating the correct number of distributions for 4 men and 5 women -/
theorem distribute_four_men_five_women :
  distribute_people 4 5 = 560 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_men_five_women_l2623_262357


namespace NUMINAMATH_CALUDE_desired_interest_rate_l2623_262318

/-- Calculate the desired interest rate (dividend yield) for a share -/
theorem desired_interest_rate (face_value : ℝ) (dividend_rate : ℝ) (market_value : ℝ) :
  face_value = 48 →
  dividend_rate = 0.09 →
  market_value = 36.00000000000001 →
  (face_value * dividend_rate) / market_value * 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_desired_interest_rate_l2623_262318


namespace NUMINAMATH_CALUDE_at_most_one_triangle_l2623_262331

/-- Represents a city in Euleria -/
def City : Type := Fin 101

/-- Represents an airline in Euleria -/
def Airline : Type := Fin 99

/-- Represents a flight between two cities operated by an airline -/
def Flight : Type := City × City × Airline

/-- The set of all flights in Euleria -/
def AllFlights : Set Flight := sorry

/-- A function that returns the airline operating a flight between two cities -/
def flightOperator : City → City → Airline := sorry

/-- A predicate that checks if three cities form a triangle -/
def isTriangle (a b c : City) : Prop :=
  flightOperator a b = flightOperator b c ∧ flightOperator b c = flightOperator c a

/-- The main theorem stating that there is at most one triangle in Euleria -/
theorem at_most_one_triangle :
  ∀ a b c d e f : City,
    isTriangle a b c → isTriangle d e f → a = d ∧ b = e ∧ c = f := by sorry

end NUMINAMATH_CALUDE_at_most_one_triangle_l2623_262331


namespace NUMINAMATH_CALUDE_foreign_stamps_count_l2623_262361

/-- Represents a stamp collection with various properties -/
structure StampCollection where
  total : ℕ
  old : ℕ
  foreignAndOld : ℕ
  neitherForeignNorOld : ℕ

/-- Calculates the number of foreign stamps in the collection -/
def foreignStamps (sc : StampCollection) : ℕ :=
  sc.total - sc.neitherForeignNorOld - (sc.old - sc.foreignAndOld)

/-- Theorem stating the number of foreign stamps in the given collection -/
theorem foreign_stamps_count (sc : StampCollection) 
    (h1 : sc.total = 200)
    (h2 : sc.old = 70)
    (h3 : sc.foreignAndOld = 20)
    (h4 : sc.neitherForeignNorOld = 60) :
    foreignStamps sc = 90 := by
  sorry

#eval foreignStamps { total := 200, old := 70, foreignAndOld := 20, neitherForeignNorOld := 60 }

end NUMINAMATH_CALUDE_foreign_stamps_count_l2623_262361


namespace NUMINAMATH_CALUDE_rice_division_l2623_262372

theorem rice_division (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 29 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight / num_containers) * ounces_per_pound = 29 := by
sorry

end NUMINAMATH_CALUDE_rice_division_l2623_262372


namespace NUMINAMATH_CALUDE_horses_meet_on_day_9_l2623_262317

/-- Represents the day on which the horses meet --/
def meetingDay : ℕ := 9

/-- The distance between Chang'an and Qi in li --/
def totalDistance : ℚ := 1125

/-- The initial distance covered by the good horse on the first day --/
def goodHorseInitial : ℚ := 103

/-- The daily increase in distance for the good horse --/
def goodHorseIncrease : ℚ := 13

/-- The initial distance covered by the mediocre horse on the first day --/
def mediocreHorseInitial : ℚ := 97

/-- The daily decrease in distance for the mediocre horse --/
def mediocreHorseDecrease : ℚ := 1/2

/-- Theorem stating that the horses meet on the 9th day --/
theorem horses_meet_on_day_9 :
  (meetingDay : ℚ) * (goodHorseInitial + mediocreHorseInitial) +
  (meetingDay * (meetingDay - 1) / 2) * (goodHorseIncrease - mediocreHorseDecrease) =
  2 * totalDistance := by
  sorry

#check horses_meet_on_day_9

end NUMINAMATH_CALUDE_horses_meet_on_day_9_l2623_262317


namespace NUMINAMATH_CALUDE_optimal_distribution_l2623_262335

/-- Represents the profit distribution problem for a fruit distributor. -/
structure FruitDistribution where
  /-- Total number of boxes of each fruit type -/
  total_boxes : ℕ
  /-- Profit per box of A fruit at Store A -/
  profit_a_store_a : ℕ
  /-- Profit per box of B fruit at Store A -/
  profit_b_store_a : ℕ
  /-- Profit per box of A fruit at Store B -/
  profit_a_store_b : ℕ
  /-- Profit per box of B fruit at Store B -/
  profit_b_store_b : ℕ
  /-- Minimum profit required for Store B -/
  min_profit_store_b : ℕ

/-- Theorem stating the optimal distribution and maximum profit -/
theorem optimal_distribution (fd : FruitDistribution)
  (h1 : fd.total_boxes = 10)
  (h2 : fd.profit_a_store_a = 11)
  (h3 : fd.profit_b_store_a = 17)
  (h4 : fd.profit_a_store_b = 9)
  (h5 : fd.profit_b_store_b = 13)
  (h6 : fd.min_profit_store_b = 115) :
  ∃ (a_store_a b_store_a a_store_b b_store_b : ℕ),
    a_store_a + b_store_a = fd.total_boxes ∧
    a_store_b + b_store_b = fd.total_boxes ∧
    a_store_a + a_store_b = fd.total_boxes ∧
    b_store_a + b_store_b = fd.total_boxes ∧
    fd.profit_a_store_b * a_store_b + fd.profit_b_store_b * b_store_b ≥ fd.min_profit_store_b ∧
    fd.profit_a_store_a * a_store_a + fd.profit_b_store_a * b_store_a +
    fd.profit_a_store_b * a_store_b + fd.profit_b_store_b * b_store_b = 246 ∧
    a_store_a = 7 ∧ b_store_a = 3 ∧ a_store_b = 3 ∧ b_store_b = 7 ∧
    ∀ (x y z w : ℕ),
      x + y = fd.total_boxes →
      z + w = fd.total_boxes →
      x + z = fd.total_boxes →
      y + w = fd.total_boxes →
      fd.profit_a_store_b * z + fd.profit_b_store_b * w ≥ fd.min_profit_store_b →
      fd.profit_a_store_a * x + fd.profit_b_store_a * y +
      fd.profit_a_store_b * z + fd.profit_b_store_b * w ≤ 246 :=
by sorry


end NUMINAMATH_CALUDE_optimal_distribution_l2623_262335


namespace NUMINAMATH_CALUDE_joe_weight_lifting_problem_l2623_262332

theorem joe_weight_lifting_problem (total_weight first_lift_weight : ℕ) 
  (h1 : total_weight = 900)
  (h2 : first_lift_weight = 400) : 
  2 * first_lift_weight - (total_weight - first_lift_weight) = 300 := by
  sorry

end NUMINAMATH_CALUDE_joe_weight_lifting_problem_l2623_262332


namespace NUMINAMATH_CALUDE_giraffe_contest_minimum_voters_l2623_262355

structure VotingSystem where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat

def minimum_voters_to_win (vs : VotingSystem) : Nat :=
  2 * ((vs.num_districts + 1) / 2) * ((vs.sections_per_district + 1) / 2)

theorem giraffe_contest_minimum_voters 
  (vs : VotingSystem)
  (h1 : vs.total_voters = 105)
  (h2 : vs.num_districts = 5)
  (h3 : vs.sections_per_district = 7)
  (h4 : vs.voters_per_section = 3)
  (h5 : vs.total_voters = vs.num_districts * vs.sections_per_district * vs.voters_per_section) :
  minimum_voters_to_win vs = 24 := by
  sorry

#eval minimum_voters_to_win ⟨105, 5, 7, 3⟩

end NUMINAMATH_CALUDE_giraffe_contest_minimum_voters_l2623_262355


namespace NUMINAMATH_CALUDE_smallest_percentage_for_90_percent_l2623_262346

/-- Represents the distribution of money in a population -/
structure MoneyDistribution where
  /-- Percentage of people owning the majority of money -/
  rich_percentage : ℝ
  /-- Percentage of money owned by the rich -/
  rich_money_percentage : ℝ
  /-- Percentage of people needed to own a target percentage of money -/
  target_percentage : ℝ
  /-- Target percentage of money to be owned -/
  target_money_percentage : ℝ

/-- Theorem stating the smallest percentage of people that can be guaranteed to own 90% of all money -/
theorem smallest_percentage_for_90_percent 
  (d : MoneyDistribution) 
  (h1 : d.rich_percentage = 20)
  (h2 : d.rich_money_percentage ≥ 80)
  (h3 : d.target_money_percentage = 90) :
  d.target_percentage = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_percentage_for_90_percent_l2623_262346


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_products_l2623_262312

theorem largest_divisor_of_consecutive_even_products (n : ℕ+) : 
  let Q := (2 * n) * (2 * n + 2) * (2 * n + 4)
  ∃ k : ℕ, Q = 12 * k ∧ ∀ m : ℕ, m > 12 → ¬(∀ n : ℕ+, m ∣ ((2 * n) * (2 * n + 2) * (2 * n + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_products_l2623_262312


namespace NUMINAMATH_CALUDE_constant_term_is_160_l2623_262363

/-- The constant term in the binomial expansion of (x + 2/x)^6 -/
def constant_term : ℕ :=
  (Nat.choose 6 3) * (2^3)

/-- Theorem: The constant term in the binomial expansion of (x + 2/x)^6 is 160 -/
theorem constant_term_is_160 : constant_term = 160 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_160_l2623_262363


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l2623_262316

def total_players : ℕ := 18
def quintuplets : ℕ := 5
def lineup_size : ℕ := 7
def quintuplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations :
  (Nat.choose quintuplets quintuplets_in_lineup) *
  (Nat.choose (total_players - quintuplets) (lineup_size - quintuplets_in_lineup)) = 12870 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l2623_262316


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2623_262344

theorem trigonometric_identities : 
  (2 * Real.sin (67.5 * π / 180) * Real.cos (67.5 * π / 180) = Real.sqrt 2 / 2) ∧
  (1 - 2 * (Real.sin (22.5 * π / 180))^2 = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2623_262344


namespace NUMINAMATH_CALUDE_expression_evaluation_l2623_262342

theorem expression_evaluation (x y : ℚ) (hx : x = 4/3) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 13/40 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2623_262342


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2623_262373

/-- Given a parabola y² = 4x and a point A on the parabola,
    if the distance from A to the focus is 4,
    then the distance from A to the origin is √21. -/
theorem parabola_point_distance (A : ℝ × ℝ) :
  A.1 ≥ 0 →  -- Ensure x-coordinate is non-negative
  A.2^2 = 4 * A.1 →  -- A is on the parabola
  (A.1 - 1)^2 + A.2^2 = 16 →  -- Distance from A to focus (1, 0) is 4
  A.1^2 + A.2^2 = 21 :=  -- Distance from A to origin is √21
by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2623_262373


namespace NUMINAMATH_CALUDE_odd_periodic2_sum_zero_l2623_262304

/-- A function that is odd and has a period of 2 -/
def OddPeriodic2 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = f x)

/-- Theorem: For any odd function with period 2, f(1) + f(4) + f(7) = 0 -/
theorem odd_periodic2_sum_zero (f : ℝ → ℝ) (h : OddPeriodic2 f) :
  f 1 + f 4 + f 7 = 0 := by
  sorry


end NUMINAMATH_CALUDE_odd_periodic2_sum_zero_l2623_262304


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2623_262377

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := y^2 / a^2 - x^2 / 3 = 1

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Define asymptotes
def asymptotes (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x

-- Define the trajectory of midpoint M
def trajectory (x y : ℝ) : Prop := x^2 / 75 + 3 * y^2 / 25 = 1

-- Define the theorem
theorem hyperbola_properties :
  ∀ (a : ℝ), 
  (∃ (x y : ℝ), hyperbola x y a) →
  eccentricity 2 →
  (∀ (x y : ℝ), asymptotes x y) ∧
  (∀ (x y : ℝ), trajectory x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2623_262377


namespace NUMINAMATH_CALUDE_smallest_vector_norm_l2623_262334

open Vector

theorem smallest_vector_norm (v : ℝ × ℝ) (h : ‖v + (-2, 4)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ u : ℝ × ℝ, ‖u + (-2, 4)‖ = 10 → ‖w‖ ≤ ‖u‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_vector_norm_l2623_262334


namespace NUMINAMATH_CALUDE_cos_sum_eleventh_pi_l2623_262329

open Complex

theorem cos_sum_eleventh_pi : 
  cos (π / 11) + cos (3 * π / 11) + cos (7 * π / 11) + cos (9 * π / 11) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_eleventh_pi_l2623_262329


namespace NUMINAMATH_CALUDE_marble_bowls_theorem_l2623_262314

theorem marble_bowls_theorem (capacity_ratio : Rat) (second_bowl_marbles : Nat) : 
  capacity_ratio = 3/4 → second_bowl_marbles = 600 →
  capacity_ratio * second_bowl_marbles + second_bowl_marbles = 1050 := by
  sorry

end NUMINAMATH_CALUDE_marble_bowls_theorem_l2623_262314


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l2623_262310

theorem sqrt_difference_equals_seven_sqrt_two_over_six :
  Real.sqrt (9 / 2) - Real.sqrt (2 / 9) = 7 * Real.sqrt 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l2623_262310
