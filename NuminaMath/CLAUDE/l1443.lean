import Mathlib

namespace NUMINAMATH_CALUDE_roberto_outfits_l1443_144383

/-- The number of different outfits Roberto can assemble -/
def number_of_outfits : ℕ :=
  let trousers : ℕ := 4
  let shirts : ℕ := 8
  let jackets : ℕ := 3
  let belts : ℕ := 2
  trousers * shirts * jackets * belts

theorem roberto_outfits :
  number_of_outfits = 192 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l1443_144383


namespace NUMINAMATH_CALUDE_cristine_lemons_l1443_144376

theorem cristine_lemons : ∀ (initial_lemons : ℕ),
  (3 / 4 : ℚ) * initial_lemons = 9 →
  initial_lemons = 12 := by
  sorry

end NUMINAMATH_CALUDE_cristine_lemons_l1443_144376


namespace NUMINAMATH_CALUDE_max_sum_of_endpoints_l1443_144337

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the theorem
theorem max_sum_of_endpoints
  (m n : ℝ)
  (h1 : n > m)
  (h2 : ∀ x ∈ Set.Icc m n, -5 ≤ f x ∧ f x ≤ 4)
  (h3 : ∃ x ∈ Set.Icc m n, f x = -5)
  (h4 : ∃ x ∈ Set.Icc m n, f x = 4) :
  ∃ k : ℝ, (∀ a b : ℝ, a ≥ m ∧ b ≤ n ∧ b > a ∧
    (∀ x ∈ Set.Icc a b, -5 ≤ f x ∧ f x ≤ 4) ∧
    (∃ x ∈ Set.Icc a b, f x = -5) ∧
    (∃ x ∈ Set.Icc a b, f x = 4) →
    a + b ≤ k) ∧
  k = n + m ∧ k = 7 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_endpoints_l1443_144337


namespace NUMINAMATH_CALUDE_angle_AMD_is_45_degrees_l1443_144315

/-- A rectangle with sides 8 and 4 -/
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (is_rectangle : sorry)
  (AB_length : dist A B = 8)
  (BC_length : dist B C = 4)

/-- A point on side AB of the rectangle -/
def point_on_AB (rect : Rectangle) (M : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • rect.A + t • rect.B

/-- The angle AMD -/
def angle_AMD (rect : Rectangle) (M : ℝ × ℝ) : ℝ := sorry

/-- The angle CMD -/
def angle_CMD (rect : Rectangle) (M : ℝ × ℝ) : ℝ := sorry

/-- The theorem to be proved -/
theorem angle_AMD_is_45_degrees (rect : Rectangle) (M : ℝ × ℝ) 
  (h1 : point_on_AB rect M) 
  (h2 : angle_AMD rect M = angle_CMD rect M) : 
  angle_AMD rect M = 45 := by sorry

end NUMINAMATH_CALUDE_angle_AMD_is_45_degrees_l1443_144315


namespace NUMINAMATH_CALUDE_car_purchase_group_size_l1443_144331

theorem car_purchase_group_size :
  ∀ (initial_cost : ℕ) (car_wash_earnings : ℕ) (cost_increase : ℕ),
    initial_cost = 1700 →
    car_wash_earnings = 500 →
    cost_increase = 40 →
    ∃ (F : ℕ),
      F > 0 ∧
      (initial_cost - car_wash_earnings) / F + cost_increase = 
      (initial_cost - car_wash_earnings) / (F - 1) ∧
      F = 6 :=
by sorry

end NUMINAMATH_CALUDE_car_purchase_group_size_l1443_144331


namespace NUMINAMATH_CALUDE_three_rug_overlap_l1443_144327

theorem three_rug_overlap (A B C X Y Z : ℝ) 
  (h1 : A + B + C = 200) 
  (h2 : X + Y + Z = 140) 
  (h3 : Y = 22) 
  (h4 : X + 2*Y + 3*Z = A + B + C) : 
  Z = 19 := by
sorry

end NUMINAMATH_CALUDE_three_rug_overlap_l1443_144327


namespace NUMINAMATH_CALUDE_divisors_18_product_and_sum_l1443_144391

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0)

theorem divisors_18_product_and_sum :
  (divisors 18).prod = 5832 ∧ (divisors 18).sum = 39 := by
  sorry

end NUMINAMATH_CALUDE_divisors_18_product_and_sum_l1443_144391


namespace NUMINAMATH_CALUDE_parabola_directrix_l1443_144399

/-- Given a parabola with equation x = 4y², prove that its directrix has equation x = -1/16 -/
theorem parabola_directrix (y : ℝ) : 
  (∃ x, x = 4 * y^2) → 
  (∃ x, x = -1/16 ∧ ∀ y, x = -1/16 → (x + 1/16)^2 = y^2/4) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1443_144399


namespace NUMINAMATH_CALUDE_principal_calculation_l1443_144311

theorem principal_calculation (P r : ℝ) 
  (h1 : P * (1 + 2 * r) = 720)
  (h2 : P * (1 + 7 * r) = 1020) : 
  P = 600 := by
sorry

end NUMINAMATH_CALUDE_principal_calculation_l1443_144311


namespace NUMINAMATH_CALUDE_target_walmart_knife_ratio_l1443_144301

/-- Represents the number of tools in a multitool -/
structure Multitool where
  screwdrivers : ℕ
  knives : ℕ
  other_tools : ℕ

/-- The Walmart multitool -/
def walmart : Multitool :=
  { screwdrivers := 1
    knives := 3
    other_tools := 2 }

/-- The Target multitool -/
def target (k : ℕ) : Multitool :=
  { screwdrivers := 1
    knives := k
    other_tools := 4 }  -- 3 files + 1 pair of scissors

/-- Total number of tools in a multitool -/
def total_tools (m : Multitool) : ℕ :=
  m.screwdrivers + m.knives + m.other_tools

/-- The theorem to prove -/
theorem target_walmart_knife_ratio :
    ∃ k : ℕ, 
      total_tools (target k) = total_tools walmart + 5 ∧ 
      k = 2 * walmart.knives := by
  sorry


end NUMINAMATH_CALUDE_target_walmart_knife_ratio_l1443_144301


namespace NUMINAMATH_CALUDE_seven_eighths_of_64_l1443_144356

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_64_l1443_144356


namespace NUMINAMATH_CALUDE_division_problem_l1443_144335

theorem division_problem (y : ℚ) : y ≠ 0 → (6 / y) * 12 = 12 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1443_144335


namespace NUMINAMATH_CALUDE_constant_rate_walking_l1443_144318

/-- Given a constant walking rate where 600 metres are covered in 4 minutes,
    prove that the distance covered in 6 minutes is 900 metres. -/
theorem constant_rate_walking (rate : ℝ) (h1 : rate > 0) (h2 : rate * 4 = 600) :
  rate * 6 = 900 := by
  sorry

end NUMINAMATH_CALUDE_constant_rate_walking_l1443_144318


namespace NUMINAMATH_CALUDE_min_value_theorem_l1443_144307

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (4 / x + 1 / (y + 1) ≥ 9 / 4) ∧
  (4 / x + 1 / (y + 1) = 9 / 4 ↔ x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1443_144307


namespace NUMINAMATH_CALUDE_circle_intersection_distance_ellipse_standard_form_collinearity_condition_l1443_144308

noncomputable section

-- Define the curve C
def C (t : ℝ) (x y : ℝ) : Prop := (4 - t) * x^2 + t * y^2 = 12

-- Part 1
theorem circle_intersection_distance (t : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, C t x₁ y₁ ∧ C t x₂ y₂ ∧ 
   y₁ = x₁ - 2 ∧ y₂ = x₂ - 2 ∧ 
   ∀ x y : ℝ, C t x y → ∃ r : ℝ, x^2 + y^2 = r^2) →
  ∃ A B : ℝ × ℝ, C t A.1 A.2 ∧ C t B.1 B.2 ∧ 
            A.2 = A.1 - 2 ∧ B.2 = B.1 - 2 ∧
            (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16 :=
sorry

-- Part 2
theorem ellipse_standard_form (t : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧
   ∀ x y : ℝ, C t x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (1 - b^2 / a^2 = 2/3 ∨ 1 - a^2 / b^2 = 2/3) →
  (∀ x y : ℝ, C t x y ↔ x^2 / 12 + y^2 / 4 = 1) ∨
  (∀ x y : ℝ, C t x y ↔ x^2 / 4 + y^2 / 12 = 1) :=
sorry

-- Part 3
theorem collinearity_condition (k m s : ℝ) :
  let P := {p : ℝ × ℝ | p.1^2 + 3 * p.2^2 = 12 ∧ p.2 = k * p.1 + m}
  let Q := {q : ℝ × ℝ | q.1^2 + 3 * q.2^2 = 12 ∧ q.2 = k * q.1 + m ∧ q ≠ (0, 2) ∧ q ≠ (0, -2)}
  let G := {g : ℝ × ℝ | g.2 = s ∧ ∃ q ∈ Q, g.2 - (-2) = (g.1 - 0) * (q.2 - (-2)) / (q.1 - 0)}
  s * m = 4 →
  ∃ p ∈ P, ∃ g ∈ G, (2 - g.2) * (p.1 - g.1) = (p.2 - g.2) * (0 - g.1) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_distance_ellipse_standard_form_collinearity_condition_l1443_144308


namespace NUMINAMATH_CALUDE_girls_in_class_correct_number_of_girls_l1443_144371

theorem girls_in_class (total_books : ℕ) (boys : ℕ) (girls_books : ℕ) : ℕ :=
  let boys_books := total_books - girls_books
  let books_per_student := boys_books / boys
  girls_books / books_per_student

theorem correct_number_of_girls :
  girls_in_class 375 10 225 = 15 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_correct_number_of_girls_l1443_144371


namespace NUMINAMATH_CALUDE_family_bought_three_soft_tacos_l1443_144349

/-- Represents the taco truck's sales during lunch rush -/
structure TacoSales where
  soft_taco_price : ℕ
  hard_taco_price : ℕ
  family_hard_tacos : ℕ
  other_customers : ℕ
  soft_tacos_per_customer : ℕ
  total_revenue : ℕ

/-- Calculates the number of soft tacos bought by the family -/
def family_soft_tacos (sales : TacoSales) : ℕ :=
  (sales.total_revenue -
   sales.family_hard_tacos * sales.hard_taco_price -
   sales.other_customers * sales.soft_tacos_per_customer * sales.soft_taco_price) /
  sales.soft_taco_price

/-- Theorem stating that the family bought 3 soft tacos -/
theorem family_bought_three_soft_tacos (sales : TacoSales)
  (h1 : sales.soft_taco_price = 2)
  (h2 : sales.hard_taco_price = 5)
  (h3 : sales.family_hard_tacos = 4)
  (h4 : sales.other_customers = 10)
  (h5 : sales.soft_tacos_per_customer = 2)
  (h6 : sales.total_revenue = 66) :
  family_soft_tacos sales = 3 := by
  sorry

end NUMINAMATH_CALUDE_family_bought_three_soft_tacos_l1443_144349


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l1443_144341

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of boys in the arrangement. -/
def num_boys : ℕ := 4

/-- The number of girls in the arrangement. -/
def num_girls : ℕ := 2

/-- The total number of people in the arrangement. -/
def total_people : ℕ := num_boys + num_girls

theorem photo_arrangement_count :
  arrangements total_people -
  arrangements (total_people - 1) -
  arrangements (total_people - 1) +
  arrangements (total_people - 2) = 504 := by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l1443_144341


namespace NUMINAMATH_CALUDE_largest_fraction_l1443_144387

theorem largest_fraction (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b = c) (h4 : c < d) :
  let f1 := (a + b) / (c + d)
  let f2 := (a + d) / (b + c)
  let f3 := (b + c) / (a + d)
  let f4 := (b + d) / (a + c)
  let f5 := (c + d) / (a + b)
  (f4 = f5) ∧ (f4 ≥ f1) ∧ (f4 ≥ f2) ∧ (f4 ≥ f3) :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l1443_144387


namespace NUMINAMATH_CALUDE_bottom_row_is_2143_l1443_144397

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Fin 4

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  -- Each number appears exactly once per row and column
  (∀ i j k, i ≠ k → g i j ≠ g k j) ∧
  (∀ i j k, j ≠ k → g i j ≠ g i k) ∧
  -- L-shaped sum constraints
  g 0 0 + g 0 1 = 3 ∧
  g 0 3 + g 1 3 = 6 ∧
  g 2 1 + g 3 1 = 5

-- Define the bottom row
def bottom_row (g : Grid) : Fin 4 → Fin 4 := g 3

-- Theorem stating the bottom row forms 2143
theorem bottom_row_is_2143 (g : Grid) (h : is_valid_grid g) :
  (bottom_row g 0, bottom_row g 1, bottom_row g 2, bottom_row g 3) = (2, 1, 4, 3) :=
sorry

end NUMINAMATH_CALUDE_bottom_row_is_2143_l1443_144397


namespace NUMINAMATH_CALUDE_emma_bank_account_l1443_144386

theorem emma_bank_account (initial_amount : ℝ) : 
  let withdrawal := 60
  let deposit := 2 * withdrawal
  let final_balance := 290
  (initial_amount - withdrawal + deposit = final_balance) → initial_amount = 230 := by
sorry

end NUMINAMATH_CALUDE_emma_bank_account_l1443_144386


namespace NUMINAMATH_CALUDE_point_on_axis_l1443_144367

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of a point being on the x-axis -/
def onXAxis (p : Point2D) : Prop := p.y = 0

/-- Definition of a point being on the y-axis -/
def onYAxis (p : Point2D) : Prop := p.x = 0

/-- Theorem: If xy = 0, then the point is on the x-axis or y-axis -/
theorem point_on_axis (p : Point2D) (h : p.x * p.y = 0) :
  onXAxis p ∨ onYAxis p := by
  sorry

end NUMINAMATH_CALUDE_point_on_axis_l1443_144367


namespace NUMINAMATH_CALUDE_square_side_length_l1443_144323

theorem square_side_length (area : ℚ) (h : area = 9 / 16) :
  ∃ side : ℚ, side * side = area ∧ side = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1443_144323


namespace NUMINAMATH_CALUDE_speaking_orders_eq_720_l1443_144394

/-- The number of different speaking orders for selecting 4 students from a group of 7 students,
    including students A and B, with the requirement that at least one of A or B must participate. -/
def speakingOrders : ℕ :=
  Nat.descFactorial 7 4 - Nat.descFactorial 5 4

/-- Theorem stating that the number of speaking orders is 720. -/
theorem speaking_orders_eq_720 : speakingOrders = 720 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_eq_720_l1443_144394


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l1443_144333

theorem ice_cream_combinations (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l1443_144333


namespace NUMINAMATH_CALUDE_union_M_N_l1443_144328

def M : Set ℕ := {1, 2}

def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_M_N_l1443_144328


namespace NUMINAMATH_CALUDE_x_squared_over_y_squared_equals_two_l1443_144395

theorem x_squared_over_y_squared_equals_two
  (x y z : ℝ) 
  (x_pos : 0 < x) 
  (y_pos : 0 < y) 
  (z_pos : 0 < z) 
  (all_different : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h : y^2 / (x^2 - z^2) = (x^2 + y^2) / z^2)
  (h2 : (x^2 + y^2) / z^2 = x^2 / y^2) :
  x^2 / y^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_x_squared_over_y_squared_equals_two_l1443_144395


namespace NUMINAMATH_CALUDE_no_solution_exponential_equation_l1443_144355

theorem no_solution_exponential_equation :
  ¬ ∃ z : ℝ, (16 : ℝ) ^ (3 * z) = (64 : ℝ) ^ (2 * z + 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exponential_equation_l1443_144355


namespace NUMINAMATH_CALUDE_jesse_room_area_l1443_144303

/-- The area of a rectangular room -/
def room_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of Jesse's room is 96 square feet -/
theorem jesse_room_area : room_area 12 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_jesse_room_area_l1443_144303


namespace NUMINAMATH_CALUDE_scalene_triangle_c_equals_four_l1443_144372

/-- A scalene triangle with integer side lengths satisfying a specific equation -/
structure ScaleneTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  equation : a^2 + b^2 - 6*a - 4*b + 13 = 0

/-- Theorem: If a scalene triangle satisfies the given equation, then c = 4 -/
theorem scalene_triangle_c_equals_four (t : ScaleneTriangle) : t.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_scalene_triangle_c_equals_four_l1443_144372


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1443_144365

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (large_cube_edge : ℝ) (sphere_diameter : ℝ) (small_cube_diagonal : ℝ) :
  large_cube_edge = 12 →
  sphere_diameter = large_cube_edge →
  small_cube_diagonal = sphere_diameter / 2 →
  (small_cube_diagonal / Real.sqrt 3) ^ 3 = 24 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1443_144365


namespace NUMINAMATH_CALUDE_rectangle_diagonal_estimate_l1443_144385

theorem rectangle_diagonal_estimate (length width diagonal : ℝ) : 
  length = 3 → width = 2 → diagonal^2 = length^2 + width^2 → 
  3.6 < diagonal ∧ diagonal < 3.7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_estimate_l1443_144385


namespace NUMINAMATH_CALUDE_job_completion_time_l1443_144396

theorem job_completion_time (x : ℝ) : 
  x > 0 → -- A's completion time is positive
  4 * (1/x + 1/20) = 1 - 0.5333333333333333 → -- Condition from working together
  x = 15 := by
    sorry

end NUMINAMATH_CALUDE_job_completion_time_l1443_144396


namespace NUMINAMATH_CALUDE_magician_performances_l1443_144388

/-- The number of performances a magician has put on --/
def num_performances : ℕ := 100

/-- The probability that an audience member never reappears --/
def prob_never_reappear : ℚ := 1/10

/-- The probability that two people reappear instead of one --/
def prob_two_reappear : ℚ := 1/5

/-- The total number of people who have reappeared --/
def total_reappeared : ℕ := 110

theorem magician_performances :
  (1 - prob_never_reappear - prob_two_reappear) * num_performances +
  2 * prob_two_reappear * num_performances = total_reappeared :=
by sorry

end NUMINAMATH_CALUDE_magician_performances_l1443_144388


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1155_l1443_144336

def is_prime_factor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ n % p = 0

theorem sum_of_largest_and_smallest_prime_factors_of_1155 :
  ∃ (smallest largest : ℕ),
    is_prime_factor smallest 1155 ∧
    is_prime_factor largest 1155 ∧
    (∀ p, is_prime_factor p 1155 → smallest ≤ p) ∧
    (∀ p, is_prime_factor p 1155 → p ≤ largest) ∧
    smallest + largest = 14 :=
sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1155_l1443_144336


namespace NUMINAMATH_CALUDE_intersection_range_l1443_144359

theorem intersection_range (k₁ k₂ t p q m n : ℝ) : 
  k₁ > 0 →
  k₂ > 0 →
  k₁ = k₂ →
  t ≠ 0 →
  t ≠ -2 →
  p = k₁ * t →
  q = k₁ * (t + 2) →
  m = k₂ / t →
  n = k₂ / (t + 2) →
  (p - m) * (q - n) < 0 ↔ (-3 < t ∧ t < -2) ∨ (0 < t ∧ t < 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l1443_144359


namespace NUMINAMATH_CALUDE_percentage_5_years_plus_is_30_percent_l1443_144366

/-- Represents the number of employees for each year group -/
def employee_distribution : List ℕ := [5, 5, 8, 3, 2, 2, 2, 1, 1, 1]

/-- Calculates the total number of employees -/
def total_employees : ℕ := employee_distribution.sum

/-- Calculates the number of employees who have worked for 5 years or more -/
def employees_5_years_plus : ℕ := (employee_distribution.drop 5).sum

/-- Theorem: The percentage of employees who have worked at the Gauss company for 5 years or more is 30% -/
theorem percentage_5_years_plus_is_30_percent :
  (employees_5_years_plus : ℚ) / total_employees * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_percentage_5_years_plus_is_30_percent_l1443_144366


namespace NUMINAMATH_CALUDE_three_digit_subtraction_result_l1443_144373

theorem three_digit_subtraction_result :
  ∃ (a b c d e f : ℕ),
    100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c ≤ 999 ∧
    100 ≤ d * 100 + e * 10 + f ∧ d * 100 + e * 10 + f ≤ 999 ∧
    (∃ (g : ℕ), 0 ≤ g ∧ g ≤ 9 ∧ (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = g) ∧
    (∃ (h i : ℕ), 10 ≤ h * 10 + i ∧ h * 10 + i ≤ 99 ∧ (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = h * 10 + i) ∧
    (∃ (j k l : ℕ), 100 ≤ j * 100 + k * 10 + l ∧ j * 100 + k * 10 + l ≤ 999 ∧ (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = j * 100 + k * 10 + l) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_result_l1443_144373


namespace NUMINAMATH_CALUDE_mike_ride_mileage_l1443_144361

/-- Represents the cost of a taxi ride -/
structure TaxiRide where
  startFee : ℝ
  tollFee : ℝ
  mileage : ℝ
  costPerMile : ℝ

/-- Calculates the total cost of a taxi ride -/
def totalCost (ride : TaxiRide) : ℝ :=
  ride.startFee + ride.tollFee + ride.mileage * ride.costPerMile

theorem mike_ride_mileage :
  let mikeRide : TaxiRide := {
    startFee := 2.5,
    tollFee := 0,
    mileage := m,
    costPerMile := 0.25
  }
  let annieRide : TaxiRide := {
    startFee := 2.5,
    tollFee := 5,
    mileage := 26,
    costPerMile := 0.25
  }
  totalCost mikeRide = totalCost annieRide → m = 36 := by
  sorry

#check mike_ride_mileage

end NUMINAMATH_CALUDE_mike_ride_mileage_l1443_144361


namespace NUMINAMATH_CALUDE_library_visitors_on_sunday_l1443_144305

/-- Proves that the average number of visitors on Sundays is 660 given the specified conditions --/
theorem library_visitors_on_sunday (total_days : Nat) (non_sunday_avg : Nat) (overall_avg : Nat) : 
  total_days = 30 →
  non_sunday_avg = 240 →
  overall_avg = 310 →
  (5 * (total_days * overall_avg - 25 * non_sunday_avg)) / 5 = 660 := by
  sorry

#check library_visitors_on_sunday

end NUMINAMATH_CALUDE_library_visitors_on_sunday_l1443_144305


namespace NUMINAMATH_CALUDE_nine_integer_chords_l1443_144309

/-- Represents a circle with a given radius and a point P at a distance from the center -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Counts the number of integer-length chords containing P in the given circle -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem stating that for a circle with radius 20 and P at distance 12,
    there are exactly 9 integer-length chords containing P -/
theorem nine_integer_chords :
  let c := CircleWithPoint.mk 20 12
  count_integer_chords c = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_integer_chords_l1443_144309


namespace NUMINAMATH_CALUDE_f_properties_l1443_144313

/-- The function f(x) = tan(3x + φ) + 1 -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.tan (3 * x + φ) + 1

/-- Theorem stating the properties of the function f -/
theorem f_properties (φ : ℝ) (h1 : |φ| < π / 2) (h2 : f φ (π / 9) = 1) :
  (∃ (T : ℝ), T > 0 ∧ T = π / 3 ∧ ∀ (x : ℝ), f φ (x + T) = f φ x) ∧
  (∀ (x : ℝ), f φ x < 2 ↔ ∃ (k : ℤ), -π / 18 + k * π / 3 < x ∧ x < 7 * π / 36 + k * π / 3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1443_144313


namespace NUMINAMATH_CALUDE_protest_jail_time_protest_jail_time_result_l1443_144379

/-- Calculate the total combined weeks of jail time given protest and arrest conditions -/
theorem protest_jail_time (days_of_protest : ℕ) (num_cities : ℕ) (arrests_per_day : ℕ) 
  (pre_trial_days : ℕ) (sentence_weeks : ℕ) : ℕ :=
  let total_arrests := days_of_protest * num_cities * arrests_per_day
  let jail_days_per_person := pre_trial_days + (sentence_weeks / 2) * 7
  let total_jail_days := total_arrests * jail_days_per_person
  total_jail_days / 7

/-- The total combined weeks of jail time is 9900 weeks -/
theorem protest_jail_time_result : 
  protest_jail_time 30 21 10 4 2 = 9900 := by sorry

end NUMINAMATH_CALUDE_protest_jail_time_protest_jail_time_result_l1443_144379


namespace NUMINAMATH_CALUDE_exists_equal_digit_sum_l1443_144326

-- Define an arithmetic progression
def arithmeticProgression (a₀ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₀ + n * d

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_equal_digit_sum (a₀ : ℕ) (d : ℕ) (h : d ≠ 0) :
  ∃ (m n : ℕ), m ≠ n ∧ 
    sumOfDigits (arithmeticProgression a₀ d m) = sumOfDigits (arithmeticProgression a₀ d n) := by
  sorry


end NUMINAMATH_CALUDE_exists_equal_digit_sum_l1443_144326


namespace NUMINAMATH_CALUDE_solution_a_l1443_144340

-- Define the function F
def F (a b c : ℝ) : ℝ := a * b^3 + c

-- State the theorem
theorem solution_a : ∃ (a : ℝ), F a 2 3 = F a 3 10 ∧ a = -7/19 := by
  sorry

end NUMINAMATH_CALUDE_solution_a_l1443_144340


namespace NUMINAMATH_CALUDE_shenile_score_theorem_l1443_144368

/-- Represents the number of points Shenille scored in a basketball game -/
def shenilesScore (threePointAttempts twoPointAttempts : ℕ) : ℝ :=
  0.6 * (threePointAttempts + twoPointAttempts)

theorem shenile_score_theorem :
  ∀ threePointAttempts twoPointAttempts : ℕ,
  threePointAttempts + twoPointAttempts = 30 →
  shenilesScore threePointAttempts twoPointAttempts = 18 :=
by
  sorry

#check shenile_score_theorem

end NUMINAMATH_CALUDE_shenile_score_theorem_l1443_144368


namespace NUMINAMATH_CALUDE_initial_experiment_range_is_appropriate_l1443_144375

-- Define the types of microorganisms
inductive Microorganism
| Bacteria
| Actinomycetes
| Fungi
| Unknown

-- Define a function to represent the typical dilution range for each microorganism
def typicalDilutionRange (m : Microorganism) : Set ℕ :=
  match m with
  | Microorganism.Bacteria => {4, 5, 6}
  | Microorganism.Actinomycetes => {3, 4, 5}
  | Microorganism.Fungi => {2, 3, 4}
  | Microorganism.Unknown => {}

-- Define the general dilution range for initial experiments
def initialExperimentRange : Set ℕ := {n | 1 ≤ n ∧ n ≤ 7}

-- Theorem statement
theorem initial_experiment_range_is_appropriate :
  ∀ m : Microorganism, (typicalDilutionRange m).Subset initialExperimentRange :=
sorry

end NUMINAMATH_CALUDE_initial_experiment_range_is_appropriate_l1443_144375


namespace NUMINAMATH_CALUDE_lily_reads_28_books_l1443_144382

/-- Represents Lily's reading habits and goals over two months -/
structure LilyReading where
  last_month_weekday : Nat
  last_month_weekend : Nat
  this_month_weekday_factor : Nat
  this_month_weekend_factor : Nat

/-- Calculates the total number of books Lily reads in two months -/
def total_books_read (r : LilyReading) : Nat :=
  let last_month_total := r.last_month_weekday + r.last_month_weekend
  let this_month_weekday := r.last_month_weekday * r.this_month_weekday_factor
  let this_month_weekend := r.last_month_weekend * r.this_month_weekend_factor
  let this_month_total := this_month_weekday + this_month_weekend
  last_month_total + this_month_total

/-- Theorem stating that Lily reads 28 books in total over two months -/
theorem lily_reads_28_books :
  ∀ (r : LilyReading),
    r.last_month_weekday = 4 →
    r.last_month_weekend = 4 →
    r.this_month_weekday_factor = 2 →
    r.this_month_weekend_factor = 3 →
    total_books_read r = 28 :=
  sorry


end NUMINAMATH_CALUDE_lily_reads_28_books_l1443_144382


namespace NUMINAMATH_CALUDE_largest_integer_with_3_digit_square_base7_l1443_144370

/-- The largest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 7) :: aux (m / 7)
    aux n |>.reverse

theorem largest_integer_with_3_digit_square_base7 :
  (M ^ 2 ≥ 7^2) ∧ 
  (M ^ 2 < 7^3) ∧ 
  (∀ n : ℕ, n > M → n ^ 2 ≥ 7^3) ∧
  (toBase7 M = [6, 6]) := by
  sorry

#eval M
#eval toBase7 M

end NUMINAMATH_CALUDE_largest_integer_with_3_digit_square_base7_l1443_144370


namespace NUMINAMATH_CALUDE_binomial_variance_four_third_l1443_144354

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  (p_nonneg : 0 ≤ p)
  (p_le_one : p ≤ 1)

/-- The variance of a binomial random variable -/
def variance (n : ℕ) (p : ℝ) (ξ : BinomialRV n p) : ℝ :=
  n * p * (1 - p)

theorem binomial_variance_four_third (ξ : BinomialRV 4 (1/3)) :
  variance 4 (1/3) ξ = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_four_third_l1443_144354


namespace NUMINAMATH_CALUDE_circus_tent_capacity_l1443_144320

theorem circus_tent_capacity (total_capacity : ℕ) (num_sections : ℕ) (section_capacity : ℕ) : 
  total_capacity = 984 → 
  num_sections = 4 → 
  total_capacity = num_sections * section_capacity → 
  section_capacity = 246 := by
sorry

end NUMINAMATH_CALUDE_circus_tent_capacity_l1443_144320


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_point_five_l1443_144306

theorem reciprocal_of_negative_one_point_five :
  let x : ℚ := -3/2  -- -1.5 as a rational number
  let y : ℚ := -2/3  -- The proposed reciprocal
  (∀ z : ℚ, z ≠ 0 → ∃ w : ℚ, z * w = 1) →  -- Definition of reciprocal
  x * y = 1 ∧ y * x = 1 :=  -- Proving y is the reciprocal of x
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_point_five_l1443_144306


namespace NUMINAMATH_CALUDE_correct_parentheses_removal_l1443_144319

theorem correct_parentheses_removal (x : ℝ) :
  2 - 4 * ((1/4) * x + 1) = 2 - x - 4 := by
sorry

end NUMINAMATH_CALUDE_correct_parentheses_removal_l1443_144319


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l1443_144304

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ (c d : ℕ+), c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l1443_144304


namespace NUMINAMATH_CALUDE_least_expensive_route_cost_l1443_144339

/-- Represents the cost of travel between two cities -/
structure TravelCost where
  car : ℝ
  train : ℝ

/-- Calculates the travel cost between two cities given the distance -/
def calculateTravelCost (distance : ℝ) : TravelCost :=
  { car := 0.20 * distance,
    train := 150 + 0.15 * distance }

/-- Theorem: The least expensive route for Dereven's trip costs $37106.25 -/
theorem least_expensive_route_cost :
  let xz : ℝ := 5000
  let xy : ℝ := 5500
  let yz : ℝ := Real.sqrt (xy^2 - xz^2)
  let costXY := calculateTravelCost xy
  let costYZ := calculateTravelCost yz
  let costZX := calculateTravelCost xz
  min costXY.car costXY.train + min costYZ.car costYZ.train + min costZX.car costZX.train = 37106.25 := by
  sorry


end NUMINAMATH_CALUDE_least_expensive_route_cost_l1443_144339


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1443_144344

def A : Set ℤ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}

def B : Set ℤ := {y | ∃ x ∈ A, y = 2*x - 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1443_144344


namespace NUMINAMATH_CALUDE_f_difference_180_90_l1443_144362

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ+) : ℕ := sorry

-- Define the function f
def f (n : ℕ+) : ℚ := (sum_of_divisors n : ℚ) / n

-- Theorem statement
theorem f_difference_180_90 : f 180 - f 90 = 13 / 30 := by sorry

end NUMINAMATH_CALUDE_f_difference_180_90_l1443_144362


namespace NUMINAMATH_CALUDE_ali_ate_four_times_more_l1443_144316

def total_apples : ℕ := 80
def sara_apples : ℕ := 16

def ali_apples : ℕ := total_apples - sara_apples

theorem ali_ate_four_times_more :
  ali_apples / sara_apples = 4 :=
by sorry

end NUMINAMATH_CALUDE_ali_ate_four_times_more_l1443_144316


namespace NUMINAMATH_CALUDE_day_1500_is_sunday_l1443_144390

/-- Given that the first day is a Friday, prove that the 1500th day is a Sunday -/
theorem day_1500_is_sunday (first_day : Nat) (h : first_day % 7 = 5) : 
  (first_day + 1499) % 7 = 0 := by
  sorry

#check day_1500_is_sunday

end NUMINAMATH_CALUDE_day_1500_is_sunday_l1443_144390


namespace NUMINAMATH_CALUDE_motion_equation_l1443_144314

/-- Given a point's rectilinear motion with velocity v(t) = t^2 - 8t + 3,
    prove that its displacement function s(t) satisfies
    s(t) = t^3/3 - 4t^2 + 3t + C for some constant C. -/
theorem motion_equation (v : ℝ → ℝ) (s : ℝ → ℝ) :
  (∀ t, v t = t^2 - 8*t + 3) →
  (∀ t, (deriv s) t = v t) →
  ∃ C, ∀ t, s t = t^3/3 - 4*t^2 + 3*t + C :=
sorry

end NUMINAMATH_CALUDE_motion_equation_l1443_144314


namespace NUMINAMATH_CALUDE_angelas_height_l1443_144398

/-- Given the heights of Amy, Helen, and Angela, prove Angela's height. -/
theorem angelas_height (amy_height helen_height angela_height : ℕ) 
  (helen_taller : helen_height = amy_height + 3)
  (angela_taller : angela_height = helen_height + 4)
  (amy_is_150 : amy_height = 150) :
  angela_height = 157 := by
  sorry

end NUMINAMATH_CALUDE_angelas_height_l1443_144398


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1443_144310

def original_expression (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 8 * (x^3 - 2*x^2 + 4*x - 1)

theorem sum_of_squared_coefficients :
  ∃ a b c d : ℝ, 
    (∀ x : ℝ, original_expression x = a * x^3 + b * x^2 + c * x + d) ∧
    a^2 + b^2 + c^2 + d^2 = 2395 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1443_144310


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1443_144393

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 9
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -3 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1443_144393


namespace NUMINAMATH_CALUDE_train_length_l1443_144351

/-- Calculates the length of a train given the time it takes to cross a bridge and a lamp post -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ) :
  bridge_length = 1500 →
  bridge_time = 70 →
  post_time = 20 →
  ∃ (train_length : ℝ),
    train_length / post_time = (train_length + bridge_length) / bridge_time ∧
    train_length = 600 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1443_144351


namespace NUMINAMATH_CALUDE_odd_cube_minus_n_div_24_l1443_144353

theorem odd_cube_minus_n_div_24 (n : ℤ) (h : Odd n) : ∃ k : ℤ, n^3 - n = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_odd_cube_minus_n_div_24_l1443_144353


namespace NUMINAMATH_CALUDE_consecutive_odd_product_equality_l1443_144343

/-- The product of consecutive integers from (n+1) to (n+n) -/
def consecutiveProduct (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => n + i + 1)

/-- The product of odd numbers from 1 to (2n-1) -/
def oddProduct (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => 2 * i + 1)

/-- The main theorem stating the equality -/
theorem consecutive_odd_product_equality (n : ℕ) :
  n > 0 → consecutiveProduct n = 2^n * oddProduct n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_equality_l1443_144343


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_216_l1443_144317

theorem closest_integer_to_cube_root_216 : 
  ∀ n : ℤ, |n - (216 : ℝ)^(1/3)| ≥ |6 - (216 : ℝ)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_216_l1443_144317


namespace NUMINAMATH_CALUDE_diamond_six_three_l1443_144350

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

-- Theorem statement
theorem diamond_six_three : diamond 6 3 = 18 := by sorry

end NUMINAMATH_CALUDE_diamond_six_three_l1443_144350


namespace NUMINAMATH_CALUDE_negation_equivalence_l1443_144378

theorem negation_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1443_144378


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l1443_144345

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 6

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 5

/-- The minimum area requirement for the rectangle -/
def min_area : ℝ := 1

/-- The function to calculate the number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating the number of ways to choose four lines to form a rectangle with area ≥ 1 -/
theorem rectangle_formation_count :
  choose_two num_horizontal_lines * choose_two num_vertical_lines = 150 :=
sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l1443_144345


namespace NUMINAMATH_CALUDE_function_and_triangle_properties_l1443_144302

theorem function_and_triangle_properties 
  (ω : ℝ) 
  (h_ω_pos : ω > 0)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = 2 * Real.sin (ω * x) * Real.cos (ω * x) + 1)
  (h_period : ∀ x, f (x + 4 * Real.pi) = f x)
  (A B C : ℝ)
  (a b c : ℝ)
  (h_triangle : 2 * b * Real.cos A = a * Real.cos C + c * Real.cos A)
  (h_positive : 0 < A ∧ A < Real.pi) :
  ω = 1/2 ∧ 
  Real.cos A = 1/2 ∧ 
  A = Real.pi/3 ∧ 
  f A = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_function_and_triangle_properties_l1443_144302


namespace NUMINAMATH_CALUDE_cone_volume_l1443_144347

/-- The volume of a cone with base radius 1 and slant height 2√7 is √3π. -/
theorem cone_volume (r h s : ℝ) : 
  r = 1 → s = 2 * Real.sqrt 7 → h^2 + r^2 = s^2 → (1/3) * π * r^2 * h = Real.sqrt 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l1443_144347


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1443_144392

theorem sqrt_expression_equality : 
  Real.sqrt 8 - (1/3)⁻¹ / Real.sqrt 3 + (1 - Real.sqrt 2)^2 = 3 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1443_144392


namespace NUMINAMATH_CALUDE_light_2004_is_yellow_l1443_144363

def light_sequence : ℕ → Fin 4
  | n => match n % 7 with
    | 0 => 0  -- green
    | 1 => 1  -- yellow
    | 2 => 1  -- yellow
    | 3 => 2  -- red
    | 4 => 3  -- blue
    | 5 => 2  -- red
    | _ => 2  -- red

theorem light_2004_is_yellow : light_sequence 2003 = 1 := by
  sorry

end NUMINAMATH_CALUDE_light_2004_is_yellow_l1443_144363


namespace NUMINAMATH_CALUDE_probability_reach_bottom_is_one_fifth_l1443_144358

/-- Represents a dodecahedron -/
structure Dodecahedron where
  top_vertex : Vertex
  bottom_vertex : Vertex
  middle_vertices : Finset Vertex
  adjacent : Vertex → Finset Vertex

/-- The probability of an ant reaching the bottom vertex in two steps -/
def probability_reach_bottom (d : Dodecahedron) : ℚ :=
  1 / 5

/-- Theorem stating the probability of reaching the bottom vertex in two steps -/
theorem probability_reach_bottom_is_one_fifth (d : Dodecahedron) :
  probability_reach_bottom d = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_reach_bottom_is_one_fifth_l1443_144358


namespace NUMINAMATH_CALUDE_f_properties_l1443_144357

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

-- Main theorem
theorem f_properties (a : ℝ) (h : a > 1) :
  -- 1. Explicit formula for f
  (∀ x, f a x = (a / (a^2 - 1)) * (a^x - a^(-x))) ∧
  -- 2. f is odd and increasing
  (∀ x, f a (-x) = -(f a x)) ∧
  (∀ x y, x < y → f a x < f a y) ∧
  -- 3. Range of m
  (∀ m, (∀ x ∈ Set.Ioo (-1) 1, f a (1 - m) + f a (1 - m^2) < 0) →
    1 < m ∧ m < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1443_144357


namespace NUMINAMATH_CALUDE_expression_factorization_l1443_144332

theorem expression_factorization (x : ℝ) :
  (3 * x^3 - 67 * x^2 - 14) - (-8 * x^3 + 3 * x^2 - 14) = x^2 * (11 * x - 70) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1443_144332


namespace NUMINAMATH_CALUDE_four_digit_number_satisfies_schemes_l1443_144360

def is_valid_division (n : ℕ) (d : ℕ) : Prop :=
  d > 0 ∧ d < 10 ∧ n % d < 10

def satisfies_scheme1 (n : ℕ) : Prop :=
  ∃ d : ℕ, is_valid_division n d

def satisfies_scheme2 (n : ℕ) : Prop :=
  ∃ d : ℕ, is_valid_division n d ∧ d ≠ 1

theorem four_digit_number_satisfies_schemes :
  ∃ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧ 
    satisfies_scheme1 n ∧ 
    satisfies_scheme2 n ∧
    n = 1512 :=
sorry

end NUMINAMATH_CALUDE_four_digit_number_satisfies_schemes_l1443_144360


namespace NUMINAMATH_CALUDE_intersection_points_differ_from_roots_l1443_144342

-- Define the original quadratic equation
def original_equation (x : ℝ) : Prop := x^2 - 4*x + 3 = 0

-- Define the roots of the original equation
def roots : Set ℝ := {1, 3}

-- Define the intersection points of y = x and y = x^2 - 4x + 4
def intersection_points : Set ℝ := {x : ℝ | x = x^2 - 4*x + 4}

-- Theorem statement
theorem intersection_points_differ_from_roots : intersection_points ≠ roots := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_differ_from_roots_l1443_144342


namespace NUMINAMATH_CALUDE_initial_typists_count_l1443_144381

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 20

/-- The number of letters typed by the initial group in 20 minutes -/
def letters_20min : ℕ := 44

/-- The number of typists in the second group -/
def second_group_typists : ℕ := 30

/-- The number of letters typed by the second group in 1 hour -/
def letters_1hour : ℕ := 198

/-- The ratio of 1 hour to 20 minutes -/
def time_ratio : ℕ := 3

theorem initial_typists_count :
  initial_typists * second_group_typists * letters_20min * time_ratio = letters_1hour * initial_typists * initial_typists :=
sorry

end NUMINAMATH_CALUDE_initial_typists_count_l1443_144381


namespace NUMINAMATH_CALUDE_same_color_sock_probability_l1443_144321

def total_socks : ℕ := 30
def blue_socks : ℕ := 16
def green_socks : ℕ := 10
def red_socks : ℕ := 4

theorem same_color_sock_probability :
  let total_combinations := total_socks.choose 2
  let blue_combinations := blue_socks.choose 2
  let green_combinations := green_socks.choose 2
  let red_combinations := red_socks.choose 2
  let same_color_combinations := blue_combinations + green_combinations + red_combinations
  (same_color_combinations : ℚ) / total_combinations = 19 / 45 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_probability_l1443_144321


namespace NUMINAMATH_CALUDE_blue_face_prob_is_half_l1443_144312

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  blue_faces : ℕ
  red_faces : ℕ
  green_faces : ℕ
  face_sum : blue_faces + red_faces + green_faces = total_faces

/-- The probability of rolling a blue face on a colored cube -/
def blue_face_probability (cube : ColoredCube) : ℚ :=
  cube.blue_faces / cube.total_faces

/-- Theorem: The probability of rolling a blue face on a cube with 3 blue faces out of 6 total faces is 1/2 -/
theorem blue_face_prob_is_half (cube : ColoredCube) 
    (h1 : cube.total_faces = 6)
    (h2 : cube.blue_faces = 3) : 
    blue_face_probability cube = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_prob_is_half_l1443_144312


namespace NUMINAMATH_CALUDE_function_value_at_one_l1443_144330

/-- Given a function f where f(x-3) = 2x^2 - 3x + 1, prove that f(1) = 21 -/
theorem function_value_at_one (f : ℝ → ℝ) 
  (h : ∀ x, f (x - 3) = 2 * x^2 - 3 * x + 1) : 
  f 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_one_l1443_144330


namespace NUMINAMATH_CALUDE_cosine_sine_power_equation_l1443_144300

theorem cosine_sine_power_equation (x : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  (Real.cos x)^8 + (Real.sin x)^8 = 97 / 128 ↔ x = Real.pi / 12 ∨ x = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_power_equation_l1443_144300


namespace NUMINAMATH_CALUDE_integer_solution_l1443_144380

theorem integer_solution (n : ℤ) : n + 5 > 7 ∧ -3*n > -15 → n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_l1443_144380


namespace NUMINAMATH_CALUDE_system_solution_l1443_144329

theorem system_solution (x y : ℝ) (h1 : 3 * x + 2 * y = 2) (h2 : 2 * x + 3 * y = 8) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1443_144329


namespace NUMINAMATH_CALUDE_sequence_contains_30_l1443_144346

theorem sequence_contains_30 : ∃ n : ℕ+, n * (n + 1) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sequence_contains_30_l1443_144346


namespace NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l1443_144348

theorem cos_pi_sixth_minus_alpha (α : ℝ) 
  (h : Real.sin (α + π / 6) + Real.cos α = -Real.sqrt 3 / 3) : 
  Real.cos (π / 6 - α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l1443_144348


namespace NUMINAMATH_CALUDE_factor_theorem_application_l1443_144364

theorem factor_theorem_application (x t : ℝ) : 
  (∃ k : ℝ, 4 * x^2 + 9 * x - 2 = (x - t) * k) ↔ (t = -1/4 ∨ t = -2) :=
by sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l1443_144364


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1443_144389

theorem fraction_equivalence : 
  ∀ (n : ℚ), n = 1/2 → (4 - n) / (7 - n) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1443_144389


namespace NUMINAMATH_CALUDE_percentage_problem_l1443_144322

theorem percentage_problem (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1443_144322


namespace NUMINAMATH_CALUDE_books_lost_l1443_144338

/-- Given that Sandy has 10 books, Tim has 33 books, and they now have 19 books together,
    prove that Benny lost 24 books. -/
theorem books_lost (sandy_books tim_books total_books_now : ℕ)
  (h1 : sandy_books = 10)
  (h2 : tim_books = 33)
  (h3 : total_books_now = 19) :
  sandy_books + tim_books - total_books_now = 24 := by
  sorry

end NUMINAMATH_CALUDE_books_lost_l1443_144338


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l1443_144377

theorem pirate_treasure_distribution (x : ℕ) : x > 0 → (x * (x + 1)) / 2 = 5 * x → x + 5 * x = 54 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l1443_144377


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1443_144324

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : A = 64 * Real.pi → d = 16 → A = Real.pi * (d / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1443_144324


namespace NUMINAMATH_CALUDE_betty_pays_nothing_l1443_144384

-- Define the ages and cost
def doug_age : ℕ := 40
def alice_age : ℕ := doug_age / 2
def total_age_sum : ℕ := 130
def cost_decrease_per_year : ℕ := 5

-- Define Betty's age
def betty_age : ℕ := total_age_sum - doug_age - alice_age

-- Define the original cost of a pack of nuts
def original_nut_cost : ℕ := 2 * betty_age

-- Define the age difference between Betty and Alice
def age_difference : ℕ := betty_age - alice_age

-- Define the total cost decrease
def total_cost_decrease : ℕ := age_difference * cost_decrease_per_year

-- Define the new cost of a pack of nuts
def new_nut_cost : ℕ := max 0 (original_nut_cost - total_cost_decrease)

-- Theorem to prove
theorem betty_pays_nothing : new_nut_cost * 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_betty_pays_nothing_l1443_144384


namespace NUMINAMATH_CALUDE_f_intersects_iff_m_le_one_l1443_144325

/-- The quadratic function f(x) = mx^2 + (m-3)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 3) * x + 1

/-- The condition that f intersects the x-axis with at least one point to the right of the origin -/
def intersects_positive_x (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ f m x = 0

theorem f_intersects_iff_m_le_one :
  ∀ m : ℝ, intersects_positive_x m ↔ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_intersects_iff_m_le_one_l1443_144325


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1443_144334

theorem quadratic_equation_equivalence (x : ℝ) : 
  (x + 1)^2 + (x - 2) * (x + 2) = 1 ↔ 2 * x^2 + 2 * x - 4 = 0 :=
by sorry

-- Definitions for the components of the quadratic equation
def quadratic_term (x : ℝ) : ℝ := 2 * x^2
def quadratic_coefficient : ℝ := 2
def linear_term (x : ℝ) : ℝ := 2 * x
def linear_coefficient : ℝ := 2
def constant_term : ℝ := -4

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1443_144334


namespace NUMINAMATH_CALUDE_sold_below_cost_price_l1443_144352

def cost_price : ℚ := 5625
def profit_percentage : ℚ := 16 / 100
def additional_amount : ℚ := 1800

def selling_price_with_profit : ℚ := cost_price * (1 + profit_percentage)
def actual_selling_price : ℚ := selling_price_with_profit - additional_amount

def percentage_below_cost : ℚ := (cost_price - actual_selling_price) / cost_price * 100

theorem sold_below_cost_price : percentage_below_cost = 16 := by sorry

end NUMINAMATH_CALUDE_sold_below_cost_price_l1443_144352


namespace NUMINAMATH_CALUDE_evan_future_books_l1443_144369

/-- Calculates the number of books Evan will have in ten years given the initial conditions --/
def books_in_ten_years (initial_books : ℕ) (reduction : ℕ) (multiplier : ℕ) (addition : ℕ) : ℕ :=
  let current_books := initial_books - reduction
  let books_after_halving := current_books / 2
  multiplier * books_after_halving + addition

/-- Theorem stating that Evan will have 1080 books in ten years --/
theorem evan_future_books :
  books_in_ten_years 400 80 6 120 = 1080 := by
  sorry

#eval books_in_ten_years 400 80 6 120

end NUMINAMATH_CALUDE_evan_future_books_l1443_144369


namespace NUMINAMATH_CALUDE_bakers_new_cakes_l1443_144374

/-- Baker's cake problem -/
theorem bakers_new_cakes 
  (initial_cakes : ℕ) 
  (sold_initial : ℕ) 
  (sold_difference : ℕ) 
  (h1 : initial_cakes = 170)
  (h2 : sold_initial = 78)
  (h3 : sold_difference = 47)
  : ∃ (new_cakes : ℕ), 
    sold_initial + sold_difference = new_cakes + sold_difference ∧ 
    new_cakes = 78 :=
by sorry

end NUMINAMATH_CALUDE_bakers_new_cakes_l1443_144374
