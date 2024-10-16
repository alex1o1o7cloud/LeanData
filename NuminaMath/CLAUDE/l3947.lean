import Mathlib

namespace NUMINAMATH_CALUDE_parabola_symmetry_axis_part1_parabola_symmetry_axis_part2_l3947_394739

-- Define the parabola and its properties
def Parabola (a b c : ℝ) (h : a > 0) :=
  {f : ℝ → ℝ | ∀ x, f x = a * x^2 + b * x + c}

def AxisOfSymmetry (t : ℝ) (p : Parabola a b c h) :=
  t = -b / (2 * a)

-- Theorem for part (1)
theorem parabola_symmetry_axis_part1
  (a b c : ℝ) (h : a > 0) (p : Parabola a b c h) (t : ℝ) :
  AxisOfSymmetry t p →
  (a * 1^2 + b * 1 + c = a * 2^2 + b * 2 + c) →
  t = 3/2 := by sorry

-- Theorem for part (2)
theorem parabola_symmetry_axis_part2
  (a b c : ℝ) (h : a > 0) (p : Parabola a b c h) (t : ℝ) :
  AxisOfSymmetry t p →
  (∀ x₁ x₂, 0 < x₁ → x₁ < 1 → 1 < x₂ → x₂ < 2 →
    a * x₁^2 + b * x₁ + c < a * x₂^2 + b * x₂ + c) →
  t ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_axis_part1_parabola_symmetry_axis_part2_l3947_394739


namespace NUMINAMATH_CALUDE_p_arithmetic_fibonacci_property_l3947_394784

/-- Definition of p-arithmetic Fibonacci sequence -/
def PArithmeticFibonacci (p : ℕ) (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + p) = v n + v (n + 1)

/-- Theorem: For any p-arithmetic Fibonacci sequence, vₖ + vₖ₊ₚ = vₖ₊₂ₚ holds for all k -/
theorem p_arithmetic_fibonacci_property {p : ℕ} {v : ℕ → ℝ} 
  (hv : PArithmeticFibonacci p v) :
  ∀ k, v k + v (k + p) = v (k + 2 * p) := by
  sorry

end NUMINAMATH_CALUDE_p_arithmetic_fibonacci_property_l3947_394784


namespace NUMINAMATH_CALUDE_cube_stacking_height_l3947_394746

/-- The edge length of the large cube in meters -/
def large_cube_edge : ℝ := 1

/-- The edge length of the small cubes in millimeters -/
def small_cube_edge : ℝ := 1

/-- Conversion factor from meters to millimeters -/
def m_to_mm : ℝ := 1000

/-- Conversion factor from kilometers to millimeters -/
def km_to_mm : ℝ := 1000000

/-- The height of the column formed by stacking all small cubes in kilometers -/
def column_height : ℝ := 1000

theorem cube_stacking_height :
  (large_cube_edge * m_to_mm)^3 / small_cube_edge^3 * small_cube_edge / km_to_mm = column_height := by
  sorry

end NUMINAMATH_CALUDE_cube_stacking_height_l3947_394746


namespace NUMINAMATH_CALUDE_log_equation_solution_l3947_394710

theorem log_equation_solution (x : ℝ) :
  (Real.log x / Real.log 4) + (Real.log (1/6) / Real.log 4) = 1/2 → x = 12 := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3947_394710


namespace NUMINAMATH_CALUDE_cars_remaining_l3947_394753

theorem cars_remaining (initial : Nat) (first_group : Nat) (second_group : Nat)
  (h1 : initial = 24)
  (h2 : first_group = 8)
  (h3 : second_group = 6) :
  initial - first_group - second_group = 10 := by
  sorry

end NUMINAMATH_CALUDE_cars_remaining_l3947_394753


namespace NUMINAMATH_CALUDE_engagement_ring_saving_time_l3947_394736

/-- Proves the time required to save for an engagement ring based on annual salary and monthly savings -/
theorem engagement_ring_saving_time 
  (annual_salary : ℕ) 
  (monthly_savings : ℕ) 
  (h1 : annual_salary = 60000)
  (h2 : monthly_savings = 1000) : 
  (2 * (annual_salary / 12)) / monthly_savings = 10 := by
  sorry

end NUMINAMATH_CALUDE_engagement_ring_saving_time_l3947_394736


namespace NUMINAMATH_CALUDE_problem_1_l3947_394799

theorem problem_1 : -20 - (-14) - |(-18)| - 13 = -37 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3947_394799


namespace NUMINAMATH_CALUDE_ru_length_is_8_25_l3947_394700

/-- Triangle PQR with given side lengths and specific geometric constructions -/
structure SpecialTriangle where
  /-- Side length PQ -/
  pq : ℝ
  /-- Side length QR -/
  qr : ℝ
  /-- Side length RP -/
  rp : ℝ
  /-- Point S on QR where the angle bisector of ∠PQR intersects QR -/
  s : ℝ × ℝ
  /-- Point T on the circumcircle of PQR where the angle bisector of ∠PQR intersects (T ≠ P) -/
  t : ℝ × ℝ
  /-- Point U on PQ where the circumcircle of PST intersects (U ≠ P) -/
  u : ℝ × ℝ
  /-- PQ = 13 -/
  h_pq : pq = 13
  /-- QR = 30 -/
  h_qr : qr = 30
  /-- RP = 26 -/
  h_rp : rp = 26
  /-- S is on QR -/
  h_s_on_qr : s.1 + s.2 = qr
  /-- T is on the circumcircle of PQR -/
  h_t_on_circumcircle : True  -- placeholder
  /-- U is on PQ -/
  h_u_on_pq : u.1 + u.2 = pq
  /-- T ≠ P -/
  h_t_ne_p : t ≠ (0, 0)
  /-- U ≠ P -/
  h_u_ne_p : u ≠ (0, 0)

/-- The length of RU in the special triangle construction -/
def ruLength (tri : SpecialTriangle) : ℝ := sorry

/-- Theorem stating that RU = 8.25 in the special triangle construction -/
theorem ru_length_is_8_25 (tri : SpecialTriangle) : ruLength tri = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_ru_length_is_8_25_l3947_394700


namespace NUMINAMATH_CALUDE_triangle_base_length_l3947_394707

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 13.5 → height = 6 → area = (base * height) / 2 → base = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3947_394707


namespace NUMINAMATH_CALUDE_unattainable_value_of_function_l3947_394743

theorem unattainable_value_of_function (x : ℝ) (y : ℝ) : 
  x ≠ -4/3 → 
  y = (2-x) / (3*x+4) → 
  y ≠ -1/3 := by
sorry

end NUMINAMATH_CALUDE_unattainable_value_of_function_l3947_394743


namespace NUMINAMATH_CALUDE_intersection_slope_l3947_394793

/-- Given two lines that intersect at a point, find the slope of one line -/
theorem intersection_slope (k : ℝ) : 
  (∃ (x y : ℝ), y = 2*x + 3 ∧ y = k*x + 4 ∧ x = 1 ∧ y = 5) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l3947_394793


namespace NUMINAMATH_CALUDE_work_completion_time_l3947_394737

/-- Given that two workers A and B together complete a work in a certain number of days,
    and one worker alone can complete the work in a different number of days,
    we can determine how long it takes for both workers together to complete the work. -/
theorem work_completion_time
  (days_together : ℝ)
  (days_a_alone : ℝ)
  (h1 : days_together > 0)
  (h2 : days_a_alone > 0)
  (h3 : days_together < days_a_alone)
  (h4 : (1 / days_together) = (1 / days_a_alone) + (1 / days_b_alone))
  (h5 : days_together = 6)
  (h6 : days_a_alone = 10) :
  days_together = 6 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3947_394737


namespace NUMINAMATH_CALUDE_inequality_proof_l3947_394794

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : b^2 / a ≥ 2*b - a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3947_394794


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_eq_two_thirds_l3947_394761

/-- The value of the repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- Theorem stating that 1 minus the repeating decimal 0.333... equals 2/3 --/
theorem one_minus_repeating_third_eq_two_thirds :
  1 - repeating_third = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_eq_two_thirds_l3947_394761


namespace NUMINAMATH_CALUDE_race_head_start_l3947_394754

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (22 / 19) * Vb) :
  ∃ H : ℝ, H / L = 3 / 22 ∧ L / Va = (L - H) / Vb :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l3947_394754


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3947_394744

/-- The y-intercept of the line x/a² - y/b² = 1 is -b², where a and b are non-zero real numbers. -/
theorem y_intercept_of_line (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ y : ℝ, (0 : ℝ) / a^2 - y / b^2 = 1 ∧ y = -b^2 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3947_394744


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l3947_394701

theorem arctan_tan_difference (θ₁ θ₂ : Real) (h₁ : θ₁ = 75 * π / 180) (h₂ : θ₂ = 35 * π / 180) :
  Real.arctan (Real.tan θ₁ - 2 * Real.tan θ₂) = 15 * π / 180 := by
  sorry

#check arctan_tan_difference

end NUMINAMATH_CALUDE_arctan_tan_difference_l3947_394701


namespace NUMINAMATH_CALUDE_smallest_square_multiplier_l3947_394776

def y : ℕ := 2^4 * 3^3 * 5^4 * 7^2 * 6^7 * 8^3 * 9^10

theorem smallest_square_multiplier (n : ℕ) : 
  (∃ m : ℕ, n * y = m^2) ∧ (∀ k : ℕ, 0 < k ∧ k < n → ¬∃ m : ℕ, k * y = m^2) → n = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_multiplier_l3947_394776


namespace NUMINAMATH_CALUDE_union_A_B_when_m_1_intersection_A_B_empty_union_A_B_equals_A_l3947_394767

-- Define sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ 2 * m + 1}

-- Theorem 1: Union of A and B when m = 1
theorem union_A_B_when_m_1 :
  A ∪ B 1 = {x | -2 < x ∧ x ≤ 3} := by sorry

-- Theorem 2: Condition for A ∩ B = ∅
theorem intersection_A_B_empty (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≤ -3/2 ∨ m ≥ 4 := by sorry

-- Theorem 3: Condition for A ∪ B = A
theorem union_A_B_equals_A (m : ℝ) :
  A ∪ B m = A ↔ m < -3 ∨ (0 < m ∧ m < 1/2) := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_1_intersection_A_B_empty_union_A_B_equals_A_l3947_394767


namespace NUMINAMATH_CALUDE_remainder_problem_l3947_394731

theorem remainder_problem (n : ℕ) (h : n > 0) (h1 : (n + 1) % 6 = 4) : n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3947_394731


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l3947_394703

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 74 ∧ n % 7 = 3 ∧ ∀ m : ℕ, m < 74 ∧ m % 7 = 3 → m ≤ n → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l3947_394703


namespace NUMINAMATH_CALUDE_inverse_proportion_l3947_394726

/-- Given that y is inversely proportional to x and when x = 2, y = -3,
    this theorem proves the relationship between y and x, and the value of x when y = 2. -/
theorem inverse_proportion (x y : ℝ) : 
  (∃ k : ℝ, ∀ x ≠ 0, y = k / x) →  -- y is inversely proportional to x
  (2 : ℝ) * (-3 : ℝ) = y * x →     -- when x = 2, y = -3
  y = -6 / x ∧                     -- the function relationship
  (y = 2 → x = -3)                 -- when y = 2, x = -3
  := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_l3947_394726


namespace NUMINAMATH_CALUDE_inequality_proof_l3947_394783

theorem inequality_proof (a b : ℝ) (h : a - |b| > 0) : b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3947_394783


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l3947_394711

theorem min_value_sum_of_squares (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l3947_394711


namespace NUMINAMATH_CALUDE_trailing_zeros_of_square_l3947_394774

theorem trailing_zeros_of_square : ∃ n : ℕ, (10^11 - 2)^2 = n * 10^10 ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_square_l3947_394774


namespace NUMINAMATH_CALUDE_count_of_special_numbers_l3947_394741

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def multiples_of_25_in_range : ℕ := 36

def multiples_of_25_and_60_in_range : ℕ := 3

theorem count_of_special_numbers : 
  (multiples_of_25_in_range - multiples_of_25_and_60_in_range) = 33 := by sorry

end NUMINAMATH_CALUDE_count_of_special_numbers_l3947_394741


namespace NUMINAMATH_CALUDE_leo_laundry_problem_l3947_394718

theorem leo_laundry_problem (total_bill : ℕ) (shirt_cost : ℕ) (trouser_cost : ℕ) (total_shirts : ℕ) :
  total_bill = 140 →
  shirt_cost = 5 →
  trouser_cost = 9 →
  total_shirts = 10 →
  ∃ (num_trousers : ℕ), num_trousers = 10 ∧ total_bill = shirt_cost * total_shirts + trouser_cost * num_trousers :=
by
  sorry

end NUMINAMATH_CALUDE_leo_laundry_problem_l3947_394718


namespace NUMINAMATH_CALUDE_peaches_theorem_l3947_394725

def peaches_problem (peaches_per_basket : ℕ) (num_baskets : ℕ) (peaches_eaten : ℕ) (peaches_per_box : ℕ) : Prop :=
  let total_peaches := peaches_per_basket * num_baskets
  let remaining_peaches := total_peaches - peaches_eaten
  remaining_peaches / peaches_per_box = 8

theorem peaches_theorem : 
  peaches_problem 25 5 5 15 := by sorry

end NUMINAMATH_CALUDE_peaches_theorem_l3947_394725


namespace NUMINAMATH_CALUDE_combined_squares_perimeter_l3947_394738

/-- The perimeter of the resulting figure when combining two squares -/
theorem combined_squares_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  p1 + p2 - 2 * (p1 / 4) = 120 :=
by sorry

end NUMINAMATH_CALUDE_combined_squares_perimeter_l3947_394738


namespace NUMINAMATH_CALUDE_vector_on_line_l3947_394769

/-- Given distinct vectors a and b in a real vector space, 
    prove that the vector (1/4)*a + (3/4)*b lies on the line passing through a and b. -/
theorem vector_on_line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (h : a ≠ b) :
  ∃ t : ℝ, (1/4 : ℝ) • a + (3/4 : ℝ) • b = a + t • (b - a) :=
sorry

end NUMINAMATH_CALUDE_vector_on_line_l3947_394769


namespace NUMINAMATH_CALUDE_jennifer_grooming_time_l3947_394732

/-- The time it takes Jennifer to groom one dog, in minutes. -/
def grooming_time : ℕ := 20

/-- The number of dogs Jennifer has. -/
def num_dogs : ℕ := 2

/-- The number of days in the given period. -/
def num_days : ℕ := 30

/-- The total time Jennifer spends grooming her dogs in the given period, in hours. -/
def total_grooming_time : ℕ := 20

theorem jennifer_grooming_time :
  grooming_time * num_dogs * num_days = total_grooming_time * 60 :=
sorry

end NUMINAMATH_CALUDE_jennifer_grooming_time_l3947_394732


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3947_394719

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- The side length of the inscribed square -/
noncomputable def s : ℝ := -2 + 2 * Real.sqrt 2

/-- Theorem: The area of the inscribed square is 12 - 8√2 -/
theorem inscribed_square_area :
  let square_area := s^2
  square_area = 12 - 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3947_394719


namespace NUMINAMATH_CALUDE_leila_cake_count_l3947_394758

/-- The number of cakes Leila ate on Monday -/
def monday_cakes : ℕ := 6

/-- The number of cakes Leila ate on Friday -/
def friday_cakes : ℕ := 9

/-- The number of cakes Leila ate on Saturday -/
def saturday_cakes : ℕ := 3 * monday_cakes

/-- The total number of cakes Leila ate -/
def total_cakes : ℕ := monday_cakes + friday_cakes + saturday_cakes

theorem leila_cake_count : total_cakes = 33 := by
  sorry

end NUMINAMATH_CALUDE_leila_cake_count_l3947_394758


namespace NUMINAMATH_CALUDE_firefighter_solution_l3947_394723

/-- Represents the problem of calculating the number of firefighters needed to put out a fire. -/
def FirefighterProblem (hose_rate : ℚ) (water_needed : ℚ) (time_taken : ℚ) : Prop :=
  ∃ (num_firefighters : ℚ),
    num_firefighters * hose_rate * time_taken = water_needed ∧
    num_firefighters = 5

/-- Theorem stating that given the specific conditions of the problem, 
    the number of firefighters required is 5. -/
theorem firefighter_solution :
  FirefighterProblem 20 4000 40 := by
  sorry

end NUMINAMATH_CALUDE_firefighter_solution_l3947_394723


namespace NUMINAMATH_CALUDE_inequality_proof_l3947_394752

theorem inequality_proof (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x > deriv f x) (a b : ℝ) (hab : a > b) : 
  Real.exp a * f b > Real.exp b * f a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3947_394752


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l3947_394722

theorem max_value_of_trigonometric_expression :
  let f : ℝ → ℝ := λ x => Real.sin (x + 3 * Real.pi / 4) + Real.cos (x + Real.pi / 3) + Real.cos (x + Real.pi / 4)
  let max_value := 2 * Real.cos (-Real.pi / 24)
  ∀ x ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 4), f x ≤ max_value ∧ ∃ x₀ ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 4), f x₀ = max_value :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l3947_394722


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3947_394759

theorem triangle_abc_properties (A B C : Real) (m n : Real × Real) :
  -- Given conditions
  m = (Real.sin A, Real.sin B) →
  n = (Real.cos B, Real.cos A) →
  m.1 * n.1 + m.2 * n.2 = Real.sin (2 * C) →
  A + B + C = π →
  2 * Real.sin C = Real.sin A + Real.sin B →
  Real.sin A * Real.sin C * (Real.sin B - Real.sin A) = 18 →
  -- Conclusions
  C = π / 3 ∧ 
  2 * Real.sin A * Real.sin B * Real.cos C = 18 ∧
  Real.sin A * Real.sin B = 16 ∧
  Real.sin C = Real.sin A + Real.sin B - Real.sin A * Real.sin B / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3947_394759


namespace NUMINAMATH_CALUDE_afternoon_fliers_fraction_l3947_394787

theorem afternoon_fliers_fraction (total : ℕ) (morning_fraction : ℚ) (left_over : ℕ) 
  (h_total : total = 2000)
  (h_morning : morning_fraction = 1 / 10)
  (h_left : left_over = 1350) :
  (total - left_over - (morning_fraction * total)) / (total - (morning_fraction * total)) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_afternoon_fliers_fraction_l3947_394787


namespace NUMINAMATH_CALUDE_other_number_is_twenty_l3947_394735

theorem other_number_is_twenty (a b : ℕ) (h1 : a + b = 30) (h2 : a = 10 ∨ b = 10) : 
  (a = 20 ∨ b = 20) :=
by sorry

end NUMINAMATH_CALUDE_other_number_is_twenty_l3947_394735


namespace NUMINAMATH_CALUDE_cinema_entry_cost_l3947_394706

def totalEntryCost (totalStudents : ℕ) (regularPrice : ℕ) (discountInterval : ℕ) (freeInterval : ℕ) : ℕ :=
  let discountedStudents := totalStudents / discountInterval
  let freeStudents := totalStudents / freeInterval
  let fullPriceStudents := totalStudents - discountedStudents - freeStudents
  let fullPriceCost := fullPriceStudents * regularPrice
  let discountedCost := discountedStudents * (regularPrice / 2)
  fullPriceCost + discountedCost

theorem cinema_entry_cost :
  totalEntryCost 84 50 12 35 = 3925 := by
  sorry

end NUMINAMATH_CALUDE_cinema_entry_cost_l3947_394706


namespace NUMINAMATH_CALUDE_probability_one_heads_three_coins_l3947_394729

theorem probability_one_heads_three_coins :
  let n : ℕ := 3  -- number of coins
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  let k : ℕ := 1  -- number of heads we want
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k) = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_heads_three_coins_l3947_394729


namespace NUMINAMATH_CALUDE_prime_square_plus_eight_l3947_394762

theorem prime_square_plus_eight (p : ℕ) : 
  Nat.Prime p → (Nat.Prime (p^2 + 8) ↔ p = 3) := by sorry

end NUMINAMATH_CALUDE_prime_square_plus_eight_l3947_394762


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_l3947_394796

/-- Represents the position of the cat -/
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the position of the mouse -/
inductive MousePosition
  | TopLeft
  | TopMiddle
  | TopRight
  | RightMiddle
  | BottomRight
  | BottomMiddle
  | BottomLeft
  | LeftMiddle

/-- The number of squares in the cat's cycle -/
def catCycleLength : Nat := 4

/-- The number of segments in the mouse's cycle -/
def mouseCycleLength : Nat := 8

/-- The total number of moves -/
def totalMoves : Nat := 317

/-- Function to determine the cat's position after a given number of moves -/
def catPositionAfterMoves (moves : Nat) : CatPosition :=
  match moves % catCycleLength with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | 3 => CatPosition.BottomRight
  | _ => CatPosition.TopLeft  -- This case should never occur due to the modulo operation

/-- Function to determine the mouse's position after a given number of moves -/
def mousePositionAfterMoves (moves : Nat) : MousePosition :=
  match moves % mouseCycleLength with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.LeftMiddle
  | 2 => MousePosition.BottomLeft
  | 3 => MousePosition.BottomMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.RightMiddle
  | 6 => MousePosition.TopRight
  | 7 => MousePosition.TopMiddle
  | _ => MousePosition.TopLeft  -- This case should never occur due to the modulo operation

theorem cat_and_mouse_positions :
  catPositionAfterMoves totalMoves = CatPosition.TopLeft ∧
  mousePositionAfterMoves totalMoves = MousePosition.BottomMiddle := by
  sorry

end NUMINAMATH_CALUDE_cat_and_mouse_positions_l3947_394796


namespace NUMINAMATH_CALUDE_perfect_match_of_parts_l3947_394742

/-- Represents the number of workers assigned to produce type A parts -/
def workers_A : ℕ := 60

/-- Represents the number of workers assigned to produce type B parts -/
def workers_B : ℕ := 25

/-- The total number of workers -/
def total_workers : ℕ := 85

/-- The number of type A parts one worker can produce per day -/
def parts_A_per_worker : ℕ := 10

/-- The number of type B parts one worker can produce per day -/
def parts_B_per_worker : ℕ := 16

/-- The number of type A parts in a complete set -/
def parts_A_per_set : ℕ := 3

/-- The number of type B parts in a complete set -/
def parts_B_per_set : ℕ := 2

theorem perfect_match_of_parts :
  workers_A + workers_B = total_workers ∧
  parts_A_per_worker * workers_A * parts_B_per_set = parts_B_per_worker * workers_B * parts_A_per_set :=
by sorry

end NUMINAMATH_CALUDE_perfect_match_of_parts_l3947_394742


namespace NUMINAMATH_CALUDE_platform_length_platform_length_is_210_l3947_394755

/-- Given a train's speed and time to pass a platform and a man, calculate the platform length -/
theorem platform_length 
  (train_speed : ℝ) 
  (time_platform : ℝ) 
  (time_man : ℝ) : ℝ :=
  let train_speed_ms := train_speed * (1000 / 3600)
  let train_length := train_speed_ms * time_man
  let platform_length := train_speed_ms * time_platform - train_length
  platform_length

/-- The length of the platform is 210 meters -/
theorem platform_length_is_210 :
  platform_length 54 34 20 = 210 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_is_210_l3947_394755


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l3947_394734

theorem point_on_terminal_side (y : ℝ) (β : ℝ) : 
  (- Real.sqrt 3 : ℝ) ^ 2 + y ^ 2 > 0 →  -- Point P is not at the origin
  Real.sin β = Real.sqrt 13 / 13 →      -- Given condition for sin β
  y > 0 →                               -- y is positive (terminal side in first quadrant)
  y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l3947_394734


namespace NUMINAMATH_CALUDE_insurance_percentage_l3947_394766

theorem insurance_percentage (salary tax_rate utility_rate remaining_amount : ℝ) 
  (h1 : salary = 2000)
  (h2 : tax_rate = 0.2)
  (h3 : utility_rate = 0.25)
  (h4 : remaining_amount = 1125)
  (h5 : ∃ insurance_rate : ℝ, 
    remaining_amount = salary * (1 - tax_rate - insurance_rate) * (1 - utility_rate)) :
  ∃ insurance_rate : ℝ, insurance_rate = 0.05 := by
sorry

end NUMINAMATH_CALUDE_insurance_percentage_l3947_394766


namespace NUMINAMATH_CALUDE_coeff_x6_q_squared_is_16_l3947_394785

/-- The polynomial q(x) -/
def q (x : ℝ) : ℝ := x^5 - 4*x^3 + 3

/-- The coefficient of x^6 in (q(x))^2 -/
def coeff_x6_q_squared : ℝ := 16

/-- Theorem: The coefficient of x^6 in (q(x))^2 is 16 -/
theorem coeff_x6_q_squared_is_16 : coeff_x6_q_squared = 16 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x6_q_squared_is_16_l3947_394785


namespace NUMINAMATH_CALUDE_work_distance_calculation_l3947_394790

/-- The distance to Tim's work in miles -/
def work_distance : ℝ := 20

/-- The number of workdays Tim rides his bike -/
def workdays : ℕ := 5

/-- The distance of Tim's weekend bike ride in miles -/
def weekend_ride : ℝ := 200

/-- Tim's biking speed in miles per hour -/
def biking_speed : ℝ := 25

/-- The total time Tim spends biking in a week in hours -/
def total_biking_time : ℝ := 16

theorem work_distance_calculation : 
  2 * workdays * work_distance + weekend_ride = biking_speed * total_biking_time := by
  sorry

end NUMINAMATH_CALUDE_work_distance_calculation_l3947_394790


namespace NUMINAMATH_CALUDE_multiply_divide_sqrt_equation_l3947_394704

theorem multiply_divide_sqrt_equation (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 1.3333333333333333) :
  (x * y) / 3 = x^2 ↔ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_sqrt_equation_l3947_394704


namespace NUMINAMATH_CALUDE_dans_balloons_l3947_394779

theorem dans_balloons (fred_balloons sam_balloons total_balloons : ℕ) 
  (h1 : fred_balloons = 10)
  (h2 : sam_balloons = 46)
  (h3 : total_balloons = 72) :
  total_balloons - (fred_balloons + sam_balloons) = 16 := by
  sorry

end NUMINAMATH_CALUDE_dans_balloons_l3947_394779


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3947_394773

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (n : ℝ) / 2 * (a 1 + a n)) →  -- sum formula
  3 * a 5 - a 1 = 10 →  -- given condition
  S 13 = 117 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3947_394773


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3947_394720

theorem quadratic_root_relation (p q : ℝ) : 
  (∀ x : ℝ, x^2 - p^2*x + p*q = 0 ↔ (∃ y : ℝ, y^2 + p*y + q = 0 ∧ x = y + 1)) →
  (p = 1 ∨ (p = -2 ∧ q = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3947_394720


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l3947_394702

theorem lcm_hcf_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 47 → 
  b = 517 → 
  a = 210 := by sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l3947_394702


namespace NUMINAMATH_CALUDE_wendy_recycling_points_l3947_394780

def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def unrecycled_bags : ℕ := 2

theorem wendy_recycling_points : 
  (total_bags - unrecycled_bags) * points_per_bag = 45 := by
  sorry

end NUMINAMATH_CALUDE_wendy_recycling_points_l3947_394780


namespace NUMINAMATH_CALUDE_fraction_addition_l3947_394728

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3947_394728


namespace NUMINAMATH_CALUDE_calculate_savings_l3947_394733

/-- Given a person's income and expenditure ratio, and their income, calculate their savings -/
theorem calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) 
  (h1 : income_ratio = 7)
  (h2 : expenditure_ratio = 6)
  (h3 : income = 14000) :
  income - (expenditure_ratio * income / income_ratio) = 2000 := by
  sorry

#check calculate_savings

end NUMINAMATH_CALUDE_calculate_savings_l3947_394733


namespace NUMINAMATH_CALUDE_mean_of_sequence_mean_of_sequence_is_17_75_l3947_394745

theorem mean_of_sequence : Real → Prop :=
  fun mean =>
    let sequence := [1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 7^2, 2]
    mean = (sequence.sum : Real) / sequence.length ∧ mean = 17.75

-- The proof is omitted
theorem mean_of_sequence_is_17_75 : ∃ mean, mean_of_sequence mean :=
  sorry

end NUMINAMATH_CALUDE_mean_of_sequence_mean_of_sequence_is_17_75_l3947_394745


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l3947_394716

/-- Given a quadratic expression of the form bx^2 + 16x + 16,
    if it is the square of a binomial, then b = 4. -/
theorem quadratic_square_of_binomial (b : ℝ) :
  (∃ t u : ℝ, ∀ x : ℝ, bx^2 + 16*x + 16 = (t*x + u)^2) →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l3947_394716


namespace NUMINAMATH_CALUDE_brick_length_is_20_l3947_394772

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 20

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 10

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 7.5

/-- The length of the wall in meters -/
def wall_length : ℝ := 26

/-- The width of the wall in meters -/
def wall_width : ℝ := 2

/-- The height of the wall in meters -/
def wall_height : ℝ := 0.75

/-- The number of bricks required to build the wall -/
def num_bricks : ℕ := 26000

/-- Theorem stating that the length of the brick is 20 cm given the conditions -/
theorem brick_length_is_20 :
  brick_length = 20 ∧
  brick_width * brick_height * brick_length * num_bricks = 
  wall_length * wall_width * wall_height * 1000000 :=
by sorry

end NUMINAMATH_CALUDE_brick_length_is_20_l3947_394772


namespace NUMINAMATH_CALUDE_solve_sticker_price_l3947_394760

def sticker_price_problem (p : ℝ) : Prop :=
  let store_a_price := 1.08 * (0.8 * p - 120)
  let store_b_price := 1.08 * (0.7 * p + 50)
  store_b_price - store_a_price = 27 ∧ p = 1450

theorem solve_sticker_price : ∃ p : ℝ, sticker_price_problem p := by
  sorry

end NUMINAMATH_CALUDE_solve_sticker_price_l3947_394760


namespace NUMINAMATH_CALUDE_popped_kernels_in_first_bag_l3947_394748

/-- Represents the number of kernels in a bag -/
structure BagOfKernels where
  total : ℕ
  popped : ℕ

/-- Given information about three bags of popcorn kernels, proves that
    the number of popped kernels in the first bag is 61. -/
theorem popped_kernels_in_first_bag
  (bag1 : BagOfKernels)
  (bag2 : BagOfKernels)
  (bag3 : BagOfKernels)
  (h1 : bag1.total = 75)
  (h2 : bag2.total = 50 ∧ bag2.popped = 42)
  (h3 : bag3.total = 100 ∧ bag3.popped = 82)
  (h_avg : (bag1.popped + bag2.popped + bag3.popped) / (bag1.total + bag2.total + bag3.total) = 82 / 100) :
  bag1.popped = 61 := by
  sorry

#check popped_kernels_in_first_bag

end NUMINAMATH_CALUDE_popped_kernels_in_first_bag_l3947_394748


namespace NUMINAMATH_CALUDE_remainder_problem_l3947_394792

theorem remainder_problem (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3947_394792


namespace NUMINAMATH_CALUDE_initial_savings_theorem_l3947_394717

def calculate_initial_savings (repair_fee : ℕ) (remaining_savings : ℕ) : ℕ :=
  let corner_light := 2 * repair_fee
  let brake_disk := 3 * corner_light
  let floor_mats := brake_disk
  let steering_wheel_cover := corner_light / 2
  let seat_covers := 2 * floor_mats
  let total_expenses := repair_fee + corner_light + 2 * brake_disk + floor_mats + steering_wheel_cover + seat_covers
  remaining_savings + total_expenses

theorem initial_savings_theorem (repair_fee : ℕ) (remaining_savings : ℕ) :
  repair_fee = 10 ∧ remaining_savings = 480 →
  calculate_initial_savings repair_fee remaining_savings = 820 :=
by sorry

end NUMINAMATH_CALUDE_initial_savings_theorem_l3947_394717


namespace NUMINAMATH_CALUDE_nine_digit_palindromes_l3947_394770

/-- A function that returns the number of n-digit palindromic integers using only the digits 1, 2, and 3 -/
def count_palindromes (n : ℕ) : ℕ :=
  if n % 2 = 0 then 3^(n/2) else 3^((n+1)/2)

/-- The number of positive nine-digit palindromic integers using only the digits 1, 2, and 3 is 243 -/
theorem nine_digit_palindromes : count_palindromes 9 = 243 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_palindromes_l3947_394770


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3947_394765

open Set

def A : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_of_A_and_B : A ∪ B = Ioc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3947_394765


namespace NUMINAMATH_CALUDE_milkman_profit_l3947_394797

/-- Calculates the profit of a milkman selling three mixtures of milk and water -/
theorem milkman_profit (total_milk : ℝ) (total_water : ℝ) 
  (milk1 : ℝ) (water1 : ℝ) (price1 : ℝ)
  (milk2 : ℝ) (water2 : ℝ) (price2 : ℝ)
  (water3 : ℝ) (price3 : ℝ)
  (milk_cost : ℝ) :
  total_milk = 80 ∧
  total_water = 20 ∧
  milk1 = 40 ∧
  water1 = 5 ∧
  price1 = 19 ∧
  milk2 = 25 ∧
  water2 = 10 ∧
  price2 = 18 ∧
  water3 = 5 ∧
  price3 = 21 ∧
  milk_cost = 22 →
  let milk3 := total_milk - milk1 - milk2
  let revenue1 := (milk1 + water1) * price1
  let revenue2 := (milk2 + water2) * price2
  let revenue3 := (milk3 + water3) * price3
  let total_revenue := revenue1 + revenue2 + revenue3
  let total_cost := total_milk * milk_cost
  let profit := total_revenue - total_cost
  profit = 50 := by
sorry

end NUMINAMATH_CALUDE_milkman_profit_l3947_394797


namespace NUMINAMATH_CALUDE_rectangle_area_l3947_394721

theorem rectangle_area (width length perimeter area : ℝ) : 
  length = 4 * width →
  perimeter = 2 * (length + width) →
  perimeter = 200 →
  area = length * width →
  area = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3947_394721


namespace NUMINAMATH_CALUDE_benny_seashells_l3947_394788

theorem benny_seashells (initial_seashells : Real) (percentage_given : Real) 
  (h1 : initial_seashells = 66.5)
  (h2 : percentage_given = 75) :
  initial_seashells - (percentage_given / 100) * initial_seashells = 16.625 := by
  sorry

end NUMINAMATH_CALUDE_benny_seashells_l3947_394788


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l3947_394724

/-- Calculates the total surface area of a cube with square holes on each face. -/
def totalSurfaceArea (cubeEdge : ℝ) (holeEdge : ℝ) (holeDepth : ℝ) : ℝ :=
  let originalSurface := 6 * cubeEdge^2
  let holeArea := 6 * holeEdge^2
  let newSurfaceInHoles := 6 * 4 * holeEdge * holeDepth
  originalSurface - holeArea + newSurfaceInHoles

/-- Theorem: The total surface area of a cube with edge length 4 meters and
    square holes (side 1 meter, depth 1 meter) centered on each face is 114 square meters. -/
theorem cube_with_holes_surface_area :
  totalSurfaceArea 4 1 1 = 114 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l3947_394724


namespace NUMINAMATH_CALUDE_unique_factorization_1870_l3947_394714

/-- A function that returns true if a number is composed of only prime factors -/
def isPrimeComposite (n : Nat) : Bool :=
  sorry

/-- A function that returns true if a number is composed of a prime factor multiplied by a one-digit non-prime number -/
def isPrimeTimesNonPrime (n : Nat) : Bool :=
  sorry

/-- A function that counts the number of valid factorizations of n according to the given conditions -/
def countValidFactorizations (n : Nat) : Nat :=
  sorry

theorem unique_factorization_1870 :
  countValidFactorizations 1870 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_factorization_1870_l3947_394714


namespace NUMINAMATH_CALUDE_masud_siblings_count_l3947_394740

theorem masud_siblings_count :
  ∀ (janet_siblings masud_siblings carlos_siblings : ℕ),
    janet_siblings = 4 * masud_siblings - 60 →
    carlos_siblings = 3 * masud_siblings / 4 →
    janet_siblings = carlos_siblings + 135 →
    masud_siblings = 60 := by
  sorry

end NUMINAMATH_CALUDE_masud_siblings_count_l3947_394740


namespace NUMINAMATH_CALUDE_remainder_of_sum_l3947_394789

theorem remainder_of_sum (d : ℕ) (h1 : 242 % d = 8) (h2 : 698 % d = 9) (h3 : d = 13) :
  (242 + 698) % d = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l3947_394789


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l3947_394749

theorem max_product_sum_2000 : 
  ∃ (x : ℤ), ∀ (y : ℤ), y * (2000 - y) ≤ x * (2000 - x) ∧ x * (2000 - x) = 1000000 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l3947_394749


namespace NUMINAMATH_CALUDE_length_of_mn_l3947_394795

/-- Given four collinear points A, B, C, D in order on a line,
    with M as the midpoint of AC and N as the midpoint of BD,
    prove that the length of MN is 24 when AD = 68 and BC = 20. -/
theorem length_of_mn (A B C D M N : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order
  (M = (A + C) / 2) →            -- M is midpoint of AC
  (N = (B + D) / 2) →            -- N is midpoint of BD
  (D - A = 68) →                 -- AD = 68
  (C - B = 20) →                 -- BC = 20
  (N - M = 24) :=                -- MN = 24
by sorry

end NUMINAMATH_CALUDE_length_of_mn_l3947_394795


namespace NUMINAMATH_CALUDE_limit_f_derivative_at_one_l3947_394747

noncomputable def f (x : ℝ) : ℝ := (x^3 - 2*x) * Real.exp x

theorem limit_f_derivative_at_one :
  (deriv f) 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_limit_f_derivative_at_one_l3947_394747


namespace NUMINAMATH_CALUDE_base_for_125_with_4_digits_l3947_394791

theorem base_for_125_with_4_digits : ∃! b : ℕ, b > 1 ∧ b^3 ≤ 125 ∧ 125 < b^4 := by
  sorry

end NUMINAMATH_CALUDE_base_for_125_with_4_digits_l3947_394791


namespace NUMINAMATH_CALUDE_race_speed_ratio_l3947_394730

/-- Proves that A runs 4 times faster than B given the race conditions --/
theorem race_speed_ratio (v_B : ℝ) (k : ℝ) : 
  (k > 0) →  -- A is faster than B
  (88 / (k * v_B) = (88 - 66) / v_B) →  -- They finish at the same time
  (k = 4) :=
by sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l3947_394730


namespace NUMINAMATH_CALUDE_inequality_proof_l3947_394708

theorem inequality_proof (x : ℝ) : 2 * (5 * x + 3) ≤ x - 3 * (1 - 2 * x) → x ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3947_394708


namespace NUMINAMATH_CALUDE_existence_of_n_good_not_n_plus_1_good_l3947_394756

def sum_of_digits (k : ℕ+) : ℕ := sorry

def is_n_good (a n : ℕ+) : Prop :=
  ∃ (seq : Fin (n + 1) → ℕ+),
    seq (Fin.last n) = a ∧
    ∀ i : Fin n, seq i.succ = seq i - sum_of_digits (seq i)

theorem existence_of_n_good_not_n_plus_1_good :
  ∀ n : ℕ+, ∃ b : ℕ+, is_n_good b n ∧ ¬is_n_good b (n + 1) :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_good_not_n_plus_1_good_l3947_394756


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_properties_l3947_394705

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

def arithmetic_sequence (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := b₁ + (n - 1) * d

theorem geometric_arithmetic_sequence_properties
  (a₁ : ℝ) (b₁ : ℝ) (q : ℝ) (d : ℝ) 
  (h1 : q = -2/3)
  (h2 : b₁ = 12)
  (h3 : geometric_sequence a₁ q 9 > arithmetic_sequence b₁ d 9)
  (h4 : geometric_sequence a₁ q 10 > arithmetic_sequence b₁ d 10) :
  (geometric_sequence a₁ q 9 * geometric_sequence a₁ q 10 < 0) ∧
  (arithmetic_sequence b₁ d 9 > arithmetic_sequence b₁ d 10) :=
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_properties_l3947_394705


namespace NUMINAMATH_CALUDE_sum_of_even_and_odd_is_odd_l3947_394713

def P : Set ℤ := {x | ∃ k, x = 2 * k}
def Q : Set ℤ := {x | ∃ k, x = 2 * k + 1}
def R : Set ℤ := {x | ∃ k, x = 4 * k + 1}

theorem sum_of_even_and_odd_is_odd (a b : ℤ) (ha : a ∈ P) (hb : b ∈ Q) : 
  a + b ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_and_odd_is_odd_l3947_394713


namespace NUMINAMATH_CALUDE_fundraising_shortfall_l3947_394751

def goal : ℚ := 500
def pizza_price : ℚ := 12
def fries_price : ℚ := 0.3
def soda_price : ℚ := 2
def pizza_sold : ℕ := 15
def fries_sold : ℕ := 40
def soda_sold : ℕ := 25

theorem fundraising_shortfall :
  goal - (pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold) = 258 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_shortfall_l3947_394751


namespace NUMINAMATH_CALUDE_triangle_area_special_case_l3947_394771

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove that under the given conditions, the area of the triangle is √15.
-/
theorem triangle_area_special_case (A B C : ℝ) (a b c : ℝ) : 
  a = 2 →
  2 * Real.sin A = Real.sin C →
  π / 2 < B → B < π →
  Real.cos (2 * C) = -1/4 →
  (1/2) * a * c * Real.sin B = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_special_case_l3947_394771


namespace NUMINAMATH_CALUDE_school_attendance_l3947_394764

/-- The number of students who came to school given the number of female students,
    the difference between female and male students, and the number of absent students. -/
def students_who_came_to_school (female_students : ℕ) (female_male_difference : ℕ) (absent_students : ℕ) : ℕ :=
  female_students + (female_students - female_male_difference) - absent_students

/-- Theorem stating that given the specific conditions, 1261 students came to school. -/
theorem school_attendance : students_who_came_to_school 658 38 17 = 1261 := by
  sorry

end NUMINAMATH_CALUDE_school_attendance_l3947_394764


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_l3947_394715

def f (x : ℝ) : ℝ := |x - 2| + |x + 1| + 2 * |x + 2|

theorem f_lower_bound : ∀ x : ℝ, f x ≥ 5 := by sorry

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, 15 - 2 * (f x) < a^2 + 9 / (a^2 + 1)) → 
  a ≠ Real.sqrt 2 ∧ a ≠ -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_l3947_394715


namespace NUMINAMATH_CALUDE_equation_equality_l3947_394763

theorem equation_equality : 2 * 18 * 14 = 6 * 12 * 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l3947_394763


namespace NUMINAMATH_CALUDE_girls_in_college_l3947_394757

theorem girls_in_college (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : boys + girls = 520) : girls = 200 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_college_l3947_394757


namespace NUMINAMATH_CALUDE_definite_integral_3x_plus_sin_x_l3947_394798

theorem definite_integral_3x_plus_sin_x : 
  ∫ x in (0)..(π/2), (3*x + Real.sin x) = (3*π^2)/8 + 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_3x_plus_sin_x_l3947_394798


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3947_394777

theorem right_triangle_shorter_leg (a b c m : ℝ) : 
  a > 0 → b > 0 → c > 0 → m > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle
  m = c / 2 →        -- Median to hypotenuse
  m = 15 →           -- Median length
  b = a + 9 →        -- One leg 9 units longer
  a = (-9 + Real.sqrt 1719) / 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3947_394777


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3947_394709

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (6 * x + 3) * (4 : ℝ) ^ (3 * x + 6) = (8 : ℝ) ^ (-4 * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3947_394709


namespace NUMINAMATH_CALUDE_blocks_placement_probability_l3947_394781

/-- Number of people --/
def num_people : ℕ := 3

/-- Number of blocks each person has --/
def blocks_per_person : ℕ := 6

/-- Number of empty boxes --/
def num_boxes : ℕ := 5

/-- Maximum number of blocks a person can place in a box --/
def max_blocks_per_person_per_box : ℕ := 2

/-- Maximum total number of blocks in a box --/
def max_blocks_per_box : ℕ := 4

/-- Number of ways each person can distribute their blocks --/
def distribution_ways : ℕ := Nat.choose (num_boxes + blocks_per_person - 1) (blocks_per_person - 1)

/-- Number of favorable distributions for a specific box getting all blocks of the same color --/
def favorable_distributions : ℕ := blocks_per_person

/-- Probability of blocks placement --/
theorem blocks_placement_probability :
  let p := 1 - num_boxes * (favorable_distributions : ℚ) / (distribution_ways ^ num_people : ℚ)
  ∃ ε > 0, abs (p - (1 - 1.86891e-6)) < ε :=
sorry

end NUMINAMATH_CALUDE_blocks_placement_probability_l3947_394781


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3947_394750

theorem square_area_from_perimeter (p : ℝ) (p_pos : p > 0) : 
  let perimeter := 12 * p
  let side_length := perimeter / 4
  let area := side_length ^ 2
  area = 9 * p ^ 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3947_394750


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3947_394712

/-- If x^2 + mx + n is a perfect square, then n = (|m| / 2)^2 -/
theorem perfect_square_condition (m n : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + m*x + n = k^2) → n = (|m| / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3947_394712


namespace NUMINAMATH_CALUDE_negation_of_exists_exp_leq_zero_l3947_394768

theorem negation_of_exists_exp_leq_zero :
  (¬ ∃ x : ℝ, Real.exp x ≤ 0) ↔ (∀ x : ℝ, Real.exp x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_exp_leq_zero_l3947_394768


namespace NUMINAMATH_CALUDE_equation_solutions_l3947_394727

theorem equation_solutions :
  (∃ y₁ y₂ : ℝ, (2 * y₁^2 + 3 * y₁ - 1 = 0 ∧ 
                 2 * y₂^2 + 3 * y₂ - 1 = 0 ∧
                 y₁ = (-3 + Real.sqrt 17) / 4 ∧
                 y₂ = (-3 - Real.sqrt 17) / 4)) ∧
  (∃ x : ℝ, x * (x - 4) = -4 ∧ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3947_394727


namespace NUMINAMATH_CALUDE_equation_solutions_l3947_394786

def equation (x : ℝ) : Prop :=
  x ≠ 1 ∧ x ≠ -6 → (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 3 ∨ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3947_394786


namespace NUMINAMATH_CALUDE_sixth_power_sum_l3947_394775

theorem sixth_power_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 511 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l3947_394775


namespace NUMINAMATH_CALUDE_cloth_selling_price_l3947_394782

/-- Calculates the total selling price of cloth given its length, cost price per meter, and profit per meter. -/
def total_selling_price (length : ℝ) (cost_price_per_meter : ℝ) (profit_per_meter : ℝ) : ℝ :=
  length * (cost_price_per_meter + profit_per_meter)

/-- The total selling price of 78 meters of cloth with a cost price of Rs. 58.02564102564102 per meter
    and a profit of Rs. 29 per meter is approximately Rs. 6788.00. -/
theorem cloth_selling_price :
  let length : ℝ := 78
  let cost_price_per_meter : ℝ := 58.02564102564102
  let profit_per_meter : ℝ := 29
  abs (total_selling_price length cost_price_per_meter profit_per_meter - 6788) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l3947_394782


namespace NUMINAMATH_CALUDE_committee_probability_l3947_394778

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 5

def probability_at_least_one_of_each : ℚ := 1705 / 1771

theorem committee_probability :
  let total_committees := Nat.choose total_members committee_size
  let all_one_gender := Nat.choose boys committee_size + Nat.choose girls committee_size
  (1 : ℚ) - (all_one_gender : ℚ) / (total_committees : ℚ) = probability_at_least_one_of_each :=
sorry

end NUMINAMATH_CALUDE_committee_probability_l3947_394778
