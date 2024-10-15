import Mathlib

namespace NUMINAMATH_CALUDE_tan_22_5_deg_sum_l119_11909

theorem tan_22_5_deg_sum (a b c d : ℕ+) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - (b : ℝ).sqrt + (c : ℝ).sqrt - (d : ℝ)) :
  a + b + c + d = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_sum_l119_11909


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_three_b_plus_c_range_l119_11915

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  t.c * (t.a * Real.cos t.B - t.b / 2) = t.a^2 - t.b^2

-- Theorem for part I
theorem angle_A_is_pi_over_three (t : Triangle) 
  (h : satisfies_condition t) : t.A = π / 3 := by
  sorry

-- Theorem for part II
theorem b_plus_c_range (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.a = Real.sqrt 3) : 
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_three_b_plus_c_range_l119_11915


namespace NUMINAMATH_CALUDE_oliver_bill_denomination_l119_11917

/-- The denomination of Oliver's unknown bills -/
def x : ℕ := sorry

/-- Oliver's total money -/
def oliver_money : ℕ := 10 * x + 3 * 5

/-- William's total money -/
def william_money : ℕ := 15 * 10 + 4 * 5

theorem oliver_bill_denomination :
  (oliver_money = william_money + 45) → x = 20 := by sorry

end NUMINAMATH_CALUDE_oliver_bill_denomination_l119_11917


namespace NUMINAMATH_CALUDE_count_negative_numbers_l119_11995

theorem count_negative_numbers : ∃ (negative_count : ℕ), 
  negative_count = 2 ∧ 
  negative_count = (if (-1 : ℚ)^2007 < 0 then 1 else 0) + 
                   (if (|(-1 : ℚ)|^3 : ℚ) < 0 then 1 else 0) + 
                   (if (-1 : ℚ)^18 > 0 then 1 else 0) + 
                   (if (18 : ℚ) < 0 then 1 else 0) := by
  sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l119_11995


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l119_11997

/-- Represents an ellipse with equation x²/(2m) + y²/m = 1, where m > 0 -/
structure Ellipse (m : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / (2*m) + y^2 / m = 1
  m_pos : m > 0

/-- Represents a point on the ellipse -/
structure EllipsePoint (m : ℝ) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / (2*m) + y^2 / m = 1

/-- The theorem stating that if an ellipse with equation x²/(2m) + y²/m = 1 (m > 0)
    is intersected by the line x = √m at two points with distance 2 between them,
    then the length of the major axis of the ellipse is 4 -/
theorem ellipse_major_axis_length 
  (m : ℝ) 
  (e : Ellipse m) 
  (A B : EllipsePoint m) 
  (h1 : A.x = Real.sqrt m) 
  (h2 : B.x = Real.sqrt m) 
  (h3 : (A.y - B.y)^2 = 4) : 
  ∃ (a : ℝ), a = 2 ∧ 2*a = 4 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l119_11997


namespace NUMINAMATH_CALUDE_house_wall_planks_l119_11910

/-- Given the total number of nails, nails per plank, and additional nails used,
    calculate the number of planks needed. -/
def planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  (total_nails - additional_nails) / nails_per_plank

/-- Theorem stating that given the specific conditions, the number of planks needed is 1. -/
theorem house_wall_planks :
  planks_needed 11 3 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_planks_l119_11910


namespace NUMINAMATH_CALUDE_equation_equivalence_l119_11992

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0

-- Define the equivalent quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  9 * x^2 - 26 * x - 12 = 0

-- Theorem stating the equivalence of the two equations
theorem equation_equivalence :
  ∀ x : ℝ, original_equation x ↔ quadratic_equation x :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l119_11992


namespace NUMINAMATH_CALUDE_probability_both_selected_l119_11922

theorem probability_both_selected (p_ram p_ravi : ℚ) 
  (h1 : p_ram = 6/7) (h2 : p_ravi = 1/5) : 
  p_ram * p_ravi = 6/35 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l119_11922


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l119_11986

/-- The surface area of a cylinder with base radius 2 and lateral surface length
    equal to the diameter of the base is 24π. -/
theorem cylinder_surface_area : 
  let r : ℝ := 2
  let l : ℝ := 2 * r
  let surface_area : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * l
  surface_area = 24 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l119_11986


namespace NUMINAMATH_CALUDE_max_value_range_l119_11939

/-- The function f(x) = -x^2 + 4x - 3 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 3

/-- The theorem stating the range of m for the maximum value of f(x) -/
theorem max_value_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ f m) →
  0 < m ∧ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_range_l119_11939


namespace NUMINAMATH_CALUDE_triangle_problem_l119_11927

theorem triangle_problem (a b c A B C : Real) (h1 : b * (Real.sin B - Real.sin C) = a * Real.sin A - c * Real.sin C)
  (h2 : a = 2 * Real.sqrt 3) (h3 : (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3) :
  A = π/3 ∧ a + b + c = 2 * Real.sqrt 3 + 6 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l119_11927


namespace NUMINAMATH_CALUDE_hotel_loss_calculation_l119_11918

/-- Calculates the loss incurred by a hotel given its operations expenses and the fraction of expenses covered by client payments. -/
def hotel_loss (expenses : ℝ) (payment_fraction : ℝ) : ℝ :=
  expenses - (payment_fraction * expenses)

/-- Theorem stating that a hotel with $100 in expenses and client payments covering 3/4 of expenses incurs a $25 loss. -/
theorem hotel_loss_calculation :
  hotel_loss 100 (3/4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_hotel_loss_calculation_l119_11918


namespace NUMINAMATH_CALUDE_max_value_sin_function_l119_11900

theorem max_value_sin_function :
  ∀ x : ℝ, -π/2 ≤ x ∧ x ≤ 0 →
  ∃ y_max : ℝ, y_max = 5 ∧
  ∀ y : ℝ, y = 3 * Real.sin x + 5 → y ≤ y_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_function_l119_11900


namespace NUMINAMATH_CALUDE_financial_equation_balance_l119_11919

theorem financial_equation_balance (f w p : ℂ) : 
  f = 10 → w = -10 + 250 * I → f * p - w = 8000 → p = 799 + 25 * I := by
  sorry

end NUMINAMATH_CALUDE_financial_equation_balance_l119_11919


namespace NUMINAMATH_CALUDE_x_minus_q_upper_bound_l119_11923

theorem x_minus_q_upper_bound (x q : ℝ) (h1 : |x - 3| > q) (h2 : x < 3) :
  x - q < 3 - 2*q := by
sorry

end NUMINAMATH_CALUDE_x_minus_q_upper_bound_l119_11923


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l119_11994

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S = (√3/4)(a² + b² - c²), then the measure of angle C is π/3. -/
theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := Real.sqrt 3 / 4 * (a^2 + b^2 - c^2)
  S = 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) →
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_measure_l119_11994


namespace NUMINAMATH_CALUDE_bill_difference_l119_11902

theorem bill_difference (anna_tip bob_tip cindy_tip : ℝ)
  (anna_percent bob_percent cindy_percent : ℝ)
  (h_anna : anna_tip = 3 ∧ anna_percent = 0.15)
  (h_bob : bob_tip = 4 ∧ bob_percent = 0.10)
  (h_cindy : cindy_tip = 5 ∧ cindy_percent = 0.25)
  (h_anna_bill : anna_tip = anna_percent * (anna_tip / anna_percent))
  (h_bob_bill : bob_tip = bob_percent * (bob_tip / bob_percent))
  (h_cindy_bill : cindy_tip = cindy_percent * (cindy_tip / cindy_percent)) :
  max (anna_tip / anna_percent) (max (bob_tip / bob_percent) (cindy_tip / cindy_percent)) -
  min (anna_tip / anna_percent) (min (bob_tip / bob_percent) (cindy_tip / cindy_percent)) = 20 :=
by sorry

end NUMINAMATH_CALUDE_bill_difference_l119_11902


namespace NUMINAMATH_CALUDE_sqrt_one_sixty_four_l119_11906

theorem sqrt_one_sixty_four : Real.sqrt (1 / 64) = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_one_sixty_four_l119_11906


namespace NUMINAMATH_CALUDE_average_students_count_l119_11947

theorem average_students_count (total : ℕ) (honor average poor : ℕ)
  (first_yes second_yes third_yes : ℕ) :
  total = 30 →
  total = honor + average + poor →
  first_yes = 19 →
  second_yes = 12 →
  third_yes = 9 →
  first_yes = honor + average / 2 →
  second_yes = average →
  third_yes = poor + average / 2 →
  average = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_students_count_l119_11947


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l119_11959

/-- The value of m for which an ellipse and hyperbola with given equations have the same foci -/
theorem ellipse_hyperbola_same_foci (m : ℝ) : m > 0 →
  (∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1) →
  (∀ x y : ℝ, x^2 / m^2 - y^2 / 2 = 1) →
  (∀ c : ℝ, c^2 = 4 - m^2 ↔ c^2 = m^2 + 2) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l119_11959


namespace NUMINAMATH_CALUDE_food_distributor_comparison_l119_11964

theorem food_distributor_comparison (p₁ p₂ : ℝ) 
  (h₁ : 0 < p₁) (h₂ : 0 < p₂) (h₃ : p₁ < p₂) :
  (2 * p₁ * p₂) / (p₁ + p₂) < (p₁ + p₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_food_distributor_comparison_l119_11964


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l119_11921

/-- A parabola with equation y = -x^2 + 2x + m has its vertex on the x-axis if and only if m = -1 -/
theorem parabola_vertex_on_x_axis (m : ℝ) : 
  (∃ x, -x^2 + 2*x + m = 0 ∧ ∀ y, y = -x^2 + 2*x + m → y ≤ 0) ↔ m = -1 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l119_11921


namespace NUMINAMATH_CALUDE_y_value_l119_11916

-- Define the property for y
def satisfies_condition (y : ℝ) : Prop :=
  y = (1 / y) * (-y) - 3

-- Theorem statement
theorem y_value : ∃ y : ℝ, satisfies_condition y ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l119_11916


namespace NUMINAMATH_CALUDE_x_value_l119_11926

theorem x_value : ∃ x : ℝ, (0.4 * x = (1/3) * x + 110) ∧ (x = 1650) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l119_11926


namespace NUMINAMATH_CALUDE_facebook_group_members_l119_11941

/-- Proves that the original number of members in a Facebook group was 150 -/
theorem facebook_group_members : 
  ∀ (original_members removed_members remaining_messages_per_week messages_per_member_per_day : ℕ),
  removed_members = 20 →
  messages_per_member_per_day = 50 →
  remaining_messages_per_week = 45500 →
  original_members = 
    (remaining_messages_per_week / (messages_per_member_per_day * 7)) + removed_members →
  original_members = 150 := by
sorry

end NUMINAMATH_CALUDE_facebook_group_members_l119_11941


namespace NUMINAMATH_CALUDE_smallest_degree_poly_div_by_30_l119_11972

/-- A polynomial with coefficients in {-1, 0, 1} -/
def RestrictedPoly (k : ℕ) := {f : Polynomial ℤ // ∀ i, i < k → f.coeff i ∈ ({-1, 0, 1} : Set ℤ)}

/-- A polynomial is divisible by 30 for all positive integers -/
def DivisibleBy30 (f : Polynomial ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → (30 : ℤ) ∣ f.eval n

/-- The theorem stating the smallest degree of a polynomial satisfying the conditions -/
theorem smallest_degree_poly_div_by_30 :
  ∃ (k : ℕ) (f : RestrictedPoly k),
    DivisibleBy30 f.val ∧
    (∀ (j : ℕ) (g : RestrictedPoly j), DivisibleBy30 g.val → k ≤ j) ∧
    k = 10 := by sorry

end NUMINAMATH_CALUDE_smallest_degree_poly_div_by_30_l119_11972


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_A_l119_11938

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for the complement of A
theorem complement_A : Aᶜ = {x | x < -1 ∨ 2 ≤ x} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_A_l119_11938


namespace NUMINAMATH_CALUDE_program_output_l119_11955

theorem program_output (A : ℕ) (h : A = 1) : (((A * 2) * 3) * 4) * 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_program_output_l119_11955


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l119_11966

theorem sufficient_not_necessary : 
  (∀ X Y : ℝ, X > 2 ∧ Y > 3 → X + Y > 5 ∧ X * Y > 6) ∧ 
  (∃ X Y : ℝ, X + Y > 5 ∧ X * Y > 6 ∧ ¬(X > 2 ∧ Y > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l119_11966


namespace NUMINAMATH_CALUDE_expression_value_l119_11952

theorem expression_value (a b : ℝ) (h : a + 3*b = 0) : 
  a^3 + 3*a^2*b - 2*a - 6*b - 5 = -5 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l119_11952


namespace NUMINAMATH_CALUDE_football_playtime_l119_11949

/-- Given a total playtime of 1.5 hours and a basketball playtime of 30 minutes,
    prove that the football playtime is 60 minutes. -/
theorem football_playtime
  (total_time : ℝ)
  (basketball_time : ℕ)
  (h1 : total_time = 1.5)
  (h2 : basketball_time = 30)
  : ↑basketball_time + 60 = total_time * 60 := by
  sorry

end NUMINAMATH_CALUDE_football_playtime_l119_11949


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_b_proper_subset_of_a_iff_a_in_range_l119_11940

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 2}

-- Theorem for part (1)
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part (2)
theorem b_proper_subset_of_a_iff_a_in_range (a : ℝ) :
  B a ⊂ A ↔ (0 ≤ a ∧ a ≤ 1) ∨ a > 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_b_proper_subset_of_a_iff_a_in_range_l119_11940


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l119_11934

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_odd_sum : a 1 + a 3 + a 5 + a 7 + a 9 = 10)
  (h_even_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 20) :
  a 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l119_11934


namespace NUMINAMATH_CALUDE_gain_represents_12_meters_l119_11971

-- Define the total meters of cloth sold
def total_meters : ℝ := 60

-- Define the gain percentage
def gain_percentage : ℝ := 0.20

-- Define the cost price per meter (as a variable)
variable (cost_price : ℝ)

-- Define the selling price per meter
def selling_price (cost_price : ℝ) : ℝ := cost_price * (1 + gain_percentage)

-- Define the total gain
def total_gain (cost_price : ℝ) : ℝ := 
  total_meters * selling_price cost_price - total_meters * cost_price

-- Theorem: The gain represents 12 meters of cloth
theorem gain_represents_12_meters (cost_price : ℝ) (cost_price_pos : cost_price > 0) :
  total_gain cost_price = 12 * cost_price := by
  sorry

end NUMINAMATH_CALUDE_gain_represents_12_meters_l119_11971


namespace NUMINAMATH_CALUDE_color_one_third_square_l119_11937

theorem color_one_third_square (n : ℕ) (k : ℕ) : n = 18 → k = 6 → Nat.choose n k = 18564 := by
  sorry

end NUMINAMATH_CALUDE_color_one_third_square_l119_11937


namespace NUMINAMATH_CALUDE_symmetry_implies_phi_value_l119_11960

/-- Given a function f(x) = sin(x) + √3 * cos(x), prove that if y = f(x + φ) is symmetric about x = 0, then φ = π/6 -/
theorem symmetry_implies_phi_value (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin x + Real.sqrt 3 * Real.cos x) →
  (∀ x, f (x + φ) = f (-x + φ)) →
  φ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_phi_value_l119_11960


namespace NUMINAMATH_CALUDE_smallest_k_for_64k_gt_4_20_l119_11946

theorem smallest_k_for_64k_gt_4_20 : ∃ k : ℕ, k = 7 ∧ 64^k > 4^20 ∧ ∀ m : ℕ, m < k → 64^m ≤ 4^20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64k_gt_4_20_l119_11946


namespace NUMINAMATH_CALUDE_number_of_pieces_l119_11962

-- Define the rod length in meters
def rod_length_meters : ℝ := 42.5

-- Define the piece length in centimeters
def piece_length_cm : ℝ := 85

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Theorem to prove
theorem number_of_pieces : 
  ⌊(rod_length_meters * meters_to_cm) / piece_length_cm⌋ = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pieces_l119_11962


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l119_11956

/-- A hyperbola with foci F₁ and F₂, and a point P on the hyperbola. -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points in ℝ² -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) 
  (angle_condition : angle h.F₁ h.P h.F₂ = π / 3)
  (distance_condition : distance h.P h.F₁ = 3 * distance h.P h.F₂) :
  eccentricity h = Real.sqrt 7 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l119_11956


namespace NUMINAMATH_CALUDE_cube_sum_of_symmetric_relations_l119_11968

theorem cube_sum_of_symmetric_relations (a b c : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a * b + a * c + b * c = 2)
  (h3 : a * b * c = 1) :
  a^3 + b^3 + c^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_symmetric_relations_l119_11968


namespace NUMINAMATH_CALUDE_sequences_properties_l119_11911

def sequence1 (n : ℕ) : ℤ := (-3)^n
def sequence2 (n : ℕ) : ℤ := -2 * (-3)^n
def sequence3 (n : ℕ) : ℤ := (-3)^n + 2

theorem sequences_properties :
  (∃ k : ℕ, sequence2 k + sequence2 (k+1) + sequence2 (k+2) = 378) ∧
  (sequence1 2024 + sequence2 2024 + sequence3 2024 = 2) := by
  sorry

end NUMINAMATH_CALUDE_sequences_properties_l119_11911


namespace NUMINAMATH_CALUDE_inequality_proof_l119_11978

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^3 + b^3 = c^3) :
  a^2 + b^2 - c^2 > 6*(c - a)*(c - b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l119_11978


namespace NUMINAMATH_CALUDE_weight_sum_l119_11988

theorem weight_sum (m n o p : ℕ) 
  (h1 : m + n = 320)
  (h2 : n + o = 295)
  (h3 : o + p = 310) :
  m + p = 335 := by
  sorry

end NUMINAMATH_CALUDE_weight_sum_l119_11988


namespace NUMINAMATH_CALUDE_roots_of_unity_quadratic_count_l119_11920

/-- A complex number z is a root of unity if there exists a positive integer n such that z^n = 1 -/
def is_root_of_unity (z : ℂ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ z^n = 1

/-- The quadratic equation z^2 + az - 1 = 0 for some integer a -/
def quadratic_equation (z : ℂ) : Prop :=
  ∃ (a : ℤ), z^2 + a*z - 1 = 0

/-- The number of roots of unity that are also roots of the quadratic equation is exactly two -/
theorem roots_of_unity_quadratic_count :
  ∃! (S : Finset ℂ), (∀ z ∈ S, is_root_of_unity z ∧ quadratic_equation z) ∧ S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_roots_of_unity_quadratic_count_l119_11920


namespace NUMINAMATH_CALUDE_tony_water_consumption_l119_11903

/-- 
Given that Tony drank 48 ounces of water yesterday, which is 4% less than 
what he drank two days ago, prove that he drank 50 ounces of water two days ago.
-/
theorem tony_water_consumption (yesterday : ℝ) (two_days_ago : ℝ) 
  (h1 : yesterday = 48)
  (h2 : yesterday = two_days_ago * (1 - 0.04)) : 
  two_days_ago = 50 := by
  sorry

end NUMINAMATH_CALUDE_tony_water_consumption_l119_11903


namespace NUMINAMATH_CALUDE_minimum_requirement_proof_l119_11912

/-- The minimum pound requirement for purchasing peanuts -/
def minimum_requirement : ℕ := 15

/-- The cost of peanuts per pound in dollars -/
def cost_per_pound : ℕ := 3

/-- The amount spent by the customer in dollars -/
def amount_spent : ℕ := 105

/-- The number of pounds purchased over the minimum requirement -/
def extra_pounds : ℕ := 20

/-- Theorem stating that the minimum requirement is correct given the conditions -/
theorem minimum_requirement_proof :
  cost_per_pound * (minimum_requirement + extra_pounds) = amount_spent :=
by sorry

end NUMINAMATH_CALUDE_minimum_requirement_proof_l119_11912


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l119_11982

theorem quadratic_form_sum (x : ℝ) : 
  ∃ (a b c : ℝ), (6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) ∧ (a + b + c = 171) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l119_11982


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_stock_price_calculation_l119_11943

theorem stock_price_after_two_years 
  (initial_price : ℝ) 
  (first_year_increase : ℝ) 
  (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price

theorem stock_price_calculation : 
  stock_price_after_two_years 120 1 0.3 = 168 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_after_two_years_stock_price_calculation_l119_11943


namespace NUMINAMATH_CALUDE_error_percentage_calculation_l119_11905

theorem error_percentage_calculation (x : ℝ) (h : x > 0) :
  let correct_result := x + 5
  let erroneous_result := x - 5
  let error := abs (correct_result - erroneous_result)
  let error_percentage := (error / correct_result) * 100
  error_percentage = (10 / (x + 5)) * 100 := by sorry

end NUMINAMATH_CALUDE_error_percentage_calculation_l119_11905


namespace NUMINAMATH_CALUDE_hyperbola_circle_tangency_l119_11950

/-- Given a hyperbola and a circle, if one asymptote of the hyperbola is tangent to the circle,
    then the ratio of the hyperbola's parameters is 3/4 -/
theorem hyperbola_circle_tangency (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), (y^2 / a^2) - (x^2 / b^2) = 1 ∧ 
   (∃ (t : ℝ), y = (a/b) * x + t) ∧
   (x - 2)^2 + (y - 1)^2 = 1) →
  b / a = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_circle_tangency_l119_11950


namespace NUMINAMATH_CALUDE_max_min_product_l119_11933

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (sum_eq : x + y + z = 20) (prod_sum_eq : x*y + y*z + z*x = 78) :
  ∃ (M : ℝ), M = min (x*y) (min (y*z) (z*x)) ∧ M ≤ 400/9 ∧
  ∀ (M' : ℝ), (∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x' + y' + z' = 20 ∧ x'*y' + y'*z' + z'*x' = 78 ∧
    M' = min (x'*y') (min (y'*z') (z'*x'))) → M' ≤ 400/9 :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l119_11933


namespace NUMINAMATH_CALUDE_marbles_lost_l119_11969

theorem marbles_lost (initial : ℕ) (current : ℕ) (lost : ℕ) : 
  initial = 16 → current = 9 → lost = initial - current → lost = 7 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l119_11969


namespace NUMINAMATH_CALUDE_movie_collection_size_l119_11999

theorem movie_collection_size :
  ∀ (dvd_count blu_count : ℕ),
  (dvd_count : ℚ) / blu_count = 17 / 4 →
  (dvd_count : ℚ) / (blu_count - 4) = 9 / 2 →
  dvd_count + blu_count = 378 :=
by
  sorry

end NUMINAMATH_CALUDE_movie_collection_size_l119_11999


namespace NUMINAMATH_CALUDE_last_digit_of_35_power_last_digit_of_35_to_large_power_l119_11913

theorem last_digit_of_35_power (n : ℕ) : 35^n ≡ 5 [MOD 10] := by sorry

theorem last_digit_of_35_to_large_power :
  35^(18 * (13^33)) ≡ 5 [MOD 10] := by sorry

end NUMINAMATH_CALUDE_last_digit_of_35_power_last_digit_of_35_to_large_power_l119_11913


namespace NUMINAMATH_CALUDE_final_worker_count_l119_11979

/-- Represents the number of bees in a hive -/
structure BeeHive where
  workers : ℕ
  drones : ℕ
  queen : ℕ

def initial_hive : BeeHive := { workers := 400, drones := 75, queen := 1 }

def bees_leave (hive : BeeHive) (workers_leaving : ℕ) (drones_leaving : ℕ) : BeeHive :=
  { workers := hive.workers - workers_leaving,
    drones := hive.drones - drones_leaving,
    queen := hive.queen }

def workers_return (hive : BeeHive) (returning_workers : ℕ) : BeeHive :=
  { workers := hive.workers + returning_workers,
    drones := hive.drones,
    queen := hive.queen }

theorem final_worker_count :
  let hive1 := bees_leave initial_hive 28 12
  let hive2 := workers_return hive1 15
  hive2.workers = 387 := by sorry

end NUMINAMATH_CALUDE_final_worker_count_l119_11979


namespace NUMINAMATH_CALUDE_burger_share_l119_11981

theorem burger_share (burger_length : ℕ) (inches_per_foot : ℕ) : 
  burger_length = 1 → 
  inches_per_foot = 12 → 
  (burger_length * inches_per_foot) / 2 = 6 :=
by
  sorry

#check burger_share

end NUMINAMATH_CALUDE_burger_share_l119_11981


namespace NUMINAMATH_CALUDE_journey_time_is_41_hours_l119_11957

-- Define the flight and layover times
def flight_NO_ATL : ℝ := 2
def layover_ATL : ℝ := 4
def flight_ATL_CHI : ℝ := 5
def layover_CHI : ℝ := 3
def flight_CHI_NY : ℝ := 3
def layover_NY : ℝ := 16
def flight_NY_SF : ℝ := 24

-- Define the total time from New Orleans to New York
def time_NO_NY : ℝ := flight_NO_ATL + layover_ATL + flight_ATL_CHI + layover_CHI + flight_CHI_NY

-- Define the total journey time
def total_journey_time : ℝ := time_NO_NY + layover_NY + flight_NY_SF

-- Theorem to prove
theorem journey_time_is_41_hours : total_journey_time = 41 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_is_41_hours_l119_11957


namespace NUMINAMATH_CALUDE_unique_solution_l119_11928

theorem unique_solution (x y z : ℝ) 
  (h1 : x + y^2 + z^3 = 3)
  (h2 : y + z^2 + x^3 = 3)
  (h3 : z + x^2 + y^3 = 3)
  (px : x > 0)
  (py : y > 0)
  (pz : z > 0) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l119_11928


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l119_11967

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l119_11967


namespace NUMINAMATH_CALUDE_seventeen_is_possible_result_l119_11942

def expression (op1 op2 op3 : ℕ → ℕ → ℕ) : ℕ :=
  op1 7 (op2 2 (op3 5 8))

def is_valid_operation (op : ℕ → ℕ → ℕ) : Prop :=
  (op = (·+·)) ∨ (op = (·-·)) ∨ (op = (·*·))

theorem seventeen_is_possible_result :
  ∃ (op1 op2 op3 : ℕ → ℕ → ℕ),
    is_valid_operation op1 ∧
    is_valid_operation op2 ∧
    is_valid_operation op3 ∧
    op1 ≠ op2 ∧ op2 ≠ op3 ∧ op1 ≠ op3 ∧
    expression op1 op2 op3 = 17 :=
by
  sorry

#check seventeen_is_possible_result

end NUMINAMATH_CALUDE_seventeen_is_possible_result_l119_11942


namespace NUMINAMATH_CALUDE_angle_solution_l119_11977

def angle_coincides (α : ℝ) : Prop :=
  ∃ k : ℤ, 9 * α = k * 360 + α

theorem angle_solution :
  ∀ α : ℝ, 0 < α → α < 180 → angle_coincides α → (α = 45 ∨ α = 90) :=
by sorry

end NUMINAMATH_CALUDE_angle_solution_l119_11977


namespace NUMINAMATH_CALUDE_rotation_equivalence_l119_11930

theorem rotation_equivalence (y : ℝ) : 
  (330 : ℝ) = (360 - y) → y < 360 → y = 30 := by sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l119_11930


namespace NUMINAMATH_CALUDE_calculation_proof_l119_11948

theorem calculation_proof : (π - 1) ^ 0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l119_11948


namespace NUMINAMATH_CALUDE_counterexample_exists_l119_11945

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem counterexample_exists : ∃ n : ℕ, 
  (sum_of_digits n % 27 = 0) ∧ 
  (n % 27 ≠ 0) ∧ 
  (n = 81 ∨ n = 999 ∨ n = 9918 ∨ n = 18) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l119_11945


namespace NUMINAMATH_CALUDE_max_expression_value_l119_11975

/-- Represents the count of integers equal to each value from 1 to 2003 -/
def IntegerCounts := Fin 2003 → ℕ

/-- The sum of all integers is 2003 -/
def SumConstraint (counts : IntegerCounts) : Prop :=
  (Finset.range 2003).sum (fun i => (i + 1) * counts i) = 2003

/-- The expression to be maximized -/
def ExpressionToMaximize (counts : IntegerCounts) : ℕ :=
  (Finset.range 2002).sum (fun i => i * counts (i + 1))

/-- There are at least two integers in the set -/
def AtLeastTwoIntegers (counts : IntegerCounts) : Prop :=
  (Finset.range 2003).sum (fun i => counts i) ≥ 2

theorem max_expression_value (counts : IntegerCounts) 
  (h1 : SumConstraint counts) (h2 : AtLeastTwoIntegers counts) :
  ExpressionToMaximize counts ≤ 2001 := by
  sorry

end NUMINAMATH_CALUDE_max_expression_value_l119_11975


namespace NUMINAMATH_CALUDE_supplier_payment_proof_l119_11984

/-- Calculates the amount paid to a supplier given initial funds, received payment, expenses, and final amount -/
def amount_paid_to_supplier (initial_funds : ℤ) (received_payment : ℤ) (expenses : ℤ) (final_amount : ℤ) : ℤ :=
  initial_funds + received_payment - expenses - final_amount

/-- Proves that the amount paid to the supplier is 600 given the problem conditions -/
theorem supplier_payment_proof (initial_funds : ℤ) (received_payment : ℤ) (expenses : ℤ) (final_amount : ℤ)
  (h1 : initial_funds = 2000)
  (h2 : received_payment = 800)
  (h3 : expenses = 1200)
  (h4 : final_amount = 1000) :
  amount_paid_to_supplier initial_funds received_payment expenses final_amount = 600 := by
  sorry

#eval amount_paid_to_supplier 2000 800 1200 1000

end NUMINAMATH_CALUDE_supplier_payment_proof_l119_11984


namespace NUMINAMATH_CALUDE_equilateral_triangle_circle_radii_l119_11985

/-- For an equilateral triangle with side length a, prove the radii of circumscribed and inscribed circles. -/
theorem equilateral_triangle_circle_radii (a : ℝ) (h : a > 0) :
  ∃ (R r : ℝ),
    R = a * Real.sqrt 3 / 3 ∧
    r = a * Real.sqrt 3 / 6 ∧
    R = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circle_radii_l119_11985


namespace NUMINAMATH_CALUDE_inequality_proof_l119_11963

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * d) - Real.sqrt (b * c)| ≤ 1) : 
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l119_11963


namespace NUMINAMATH_CALUDE_octagon_area_l119_11965

/-- The area of a regular octagon inscribed in a circle with area 400π -/
theorem octagon_area (circle_area : ℝ) (h : circle_area = 400 * Real.pi) :
  let r := (circle_area / Real.pi).sqrt
  let triangle_area := (1 / 2) * r^2 * Real.sin (Real.pi / 4)
  8 * triangle_area = 800 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l119_11965


namespace NUMINAMATH_CALUDE_job_completion_time_l119_11990

/-- The time it takes for three workers to complete a job together, given their individual efficiencies -/
theorem job_completion_time 
  (sakshi_time : ℝ) 
  (tanya_efficiency : ℝ) 
  (rahul_efficiency : ℝ) 
  (h1 : sakshi_time = 5) 
  (h2 : tanya_efficiency = 1.25) 
  (h3 : rahul_efficiency = 0.6) : 
  (1 / (1 / sakshi_time + tanya_efficiency / sakshi_time + rahul_efficiency / sakshi_time)) = 100 / 57 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l119_11990


namespace NUMINAMATH_CALUDE_max_value_of_f_l119_11970

noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x)

theorem max_value_of_f :
  ∃ (M : ℝ), M = (16 * Real.sqrt 3) / 9 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l119_11970


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l119_11936

theorem sum_of_specific_numbers : 3 + 33 + 333 + 33.3 = 402.3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l119_11936


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l119_11989

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l119_11989


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l119_11908

/-- Given a parallelogram with area 128 sq m and base 8 m, prove the ratio of altitude to base is 2 -/
theorem parallelogram_altitude_base_ratio :
  ∀ (area base altitude : ℝ),
  area = 128 ∧ base = 8 ∧ area = base * altitude →
  altitude / base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l119_11908


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l119_11931

theorem fraction_sum_simplification : 
  5 / (1/(1*2) + 1/(2*3) + 1/(3*4) + 1/(4*5) + 1/(5*6)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l119_11931


namespace NUMINAMATH_CALUDE_joanne_weekly_earnings_l119_11932

def main_job_hours : ℝ := 8
def main_job_rate : ℝ := 16
def part_time_hours : ℝ := 2
def part_time_rate : ℝ := 13.5
def days_per_week : ℝ := 5

def weekly_earnings : ℝ := (main_job_hours * main_job_rate + part_time_hours * part_time_rate) * days_per_week

theorem joanne_weekly_earnings : weekly_earnings = 775 := by
  sorry

end NUMINAMATH_CALUDE_joanne_weekly_earnings_l119_11932


namespace NUMINAMATH_CALUDE_right_angled_triangle_l119_11924

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem right_angled_triangle (t : Triangle) 
  (h : (Real.cos (t.A / 2))^2 = (t.b + t.c) / (2 * t.c)) : 
  t.C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l119_11924


namespace NUMINAMATH_CALUDE_ab_value_l119_11958

theorem ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a^2 + b^2 = 1) (h2 : a^4 + b^4 = 5/8) : a * b = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l119_11958


namespace NUMINAMATH_CALUDE_tree_shadow_length_l119_11991

/-- Given a person and a tree casting shadows, this theorem calculates the length of the tree's shadow. -/
theorem tree_shadow_length 
  (person_height : ℝ) 
  (person_shadow : ℝ) 
  (tree_height : ℝ) 
  (h1 : person_height = 1.5)
  (h2 : person_shadow = 0.5)
  (h3 : tree_height = 30) :
  ∃ (tree_shadow : ℝ), tree_shadow = 10 ∧ 
    person_height / person_shadow = tree_height / tree_shadow :=
by sorry

end NUMINAMATH_CALUDE_tree_shadow_length_l119_11991


namespace NUMINAMATH_CALUDE_intersection_in_second_quadrant_l119_11935

/-- The intersection point of two lines is in the second quadrant if and only if k is in the open interval (0, 1/2) -/
theorem intersection_in_second_quadrant (k : ℝ) :
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y - x = 2 * k ∧ x < 0 ∧ y > 0) ↔ 0 < k ∧ k < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_in_second_quadrant_l119_11935


namespace NUMINAMATH_CALUDE_arcadia_population_growth_l119_11953

/-- Represents the population of Arcadia at a given year -/
def population (year : ℕ) : ℕ :=
  if year ≤ 2020 then 250
  else 250 * (3 ^ ((year - 2020) / 25))

/-- The year we're trying to prove -/
def target_year : ℕ := 2095

/-- The population threshold we're trying to exceed -/
def population_threshold : ℕ := 6000

theorem arcadia_population_growth :
  (population target_year > population_threshold) ∧
  (∀ y : ℕ, y < target_year → population y ≤ population_threshold) :=
by sorry

end NUMINAMATH_CALUDE_arcadia_population_growth_l119_11953


namespace NUMINAMATH_CALUDE_parallelogram_area_l119_11954

/-- The area of a parallelogram with vertices at (0, 0), (4, 0), (1, 5), and (5, 5) is 20 square units. -/
theorem parallelogram_area : ℝ := by
  -- Define the vertices of the parallelogram
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (4, 0)
  let v3 : ℝ × ℝ := (1, 5)
  let v4 : ℝ × ℝ := (5, 5)

  -- Calculate the area of the parallelogram
  have area : ℝ := 20

  -- Assert that the calculated area is correct
  exact area

end NUMINAMATH_CALUDE_parallelogram_area_l119_11954


namespace NUMINAMATH_CALUDE_concrete_cost_theorem_l119_11951

/-- Calculates the cost of concrete for home foundations -/
theorem concrete_cost_theorem 
  (num_homes : ℕ) 
  (length width height : ℝ) 
  (density : ℝ) 
  (cost_per_pound : ℝ) : 
  num_homes * length * width * height * density * cost_per_pound = 45000 :=
by
  sorry

#check concrete_cost_theorem 3 100 100 0.5 150 0.02

end NUMINAMATH_CALUDE_concrete_cost_theorem_l119_11951


namespace NUMINAMATH_CALUDE_jersey_revenue_proof_l119_11961

/-- The amount of money made from selling jerseys -/
def jersey_revenue (price_per_jersey : ℕ) (jerseys_sold : ℕ) : ℕ :=
  price_per_jersey * jerseys_sold

/-- Proof that the jersey revenue is $25,740 -/
theorem jersey_revenue_proof :
  jersey_revenue 165 156 = 25740 := by
  sorry

end NUMINAMATH_CALUDE_jersey_revenue_proof_l119_11961


namespace NUMINAMATH_CALUDE_sequence_conditions_l119_11993

theorem sequence_conditions (a : ℝ) : 
  let a₁ : ℝ := 1
  let a₂ : ℝ := 1
  let a₃ : ℝ := 1
  let a₄ : ℝ := a
  let a₅ : ℝ := a
  (a₁ = a₂ * a₃) ∧ 
  (a₂ = a₁ * a₃) ∧ 
  (a₃ = a₁ * a₂) ∧ 
  (a₄ = a₁ * a₅) ∧ 
  (a₅ = a₁ * a₄) := by
sorry

end NUMINAMATH_CALUDE_sequence_conditions_l119_11993


namespace NUMINAMATH_CALUDE_smallest_natural_with_remainders_l119_11996

theorem smallest_natural_with_remainders : ∃ N : ℕ,
  (N % 9 = 8) ∧
  (N % 8 = 7) ∧
  (N % 7 = 6) ∧
  (N % 6 = 5) ∧
  (N % 5 = 4) ∧
  (N % 4 = 3) ∧
  (N % 3 = 2) ∧
  (N % 2 = 1) ∧
  (∀ M : ℕ, M < N →
    ¬((M % 9 = 8) ∧
      (M % 8 = 7) ∧
      (M % 7 = 6) ∧
      (M % 6 = 5) ∧
      (M % 5 = 4) ∧
      (M % 4 = 3) ∧
      (M % 3 = 2) ∧
      (M % 2 = 1))) ∧
  N = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_natural_with_remainders_l119_11996


namespace NUMINAMATH_CALUDE_point_inside_circle_implies_a_range_l119_11944

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  (x + a)^2 + (y - a)^2 = 4

-- Define what it means for a point to be inside the circle
def point_inside_circle (x y a : ℝ) : Prop :=
  (x + a)^2 + (y - a)^2 < 4

-- Theorem statement
theorem point_inside_circle_implies_a_range :
  ∀ a : ℝ, point_inside_circle (-1) (-1) a → -1 < a ∧ a < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_point_inside_circle_implies_a_range_l119_11944


namespace NUMINAMATH_CALUDE_rectangle_area_l119_11998

/-- Given a rectangle where the length is 15% more than the breadth and the breadth is 20 meters,
    prove that its area is 460 square meters. -/
theorem rectangle_area (b l a : ℝ) : 
  b = 20 →                  -- The breadth is 20 meters
  l = b * 1.15 →            -- The length is 15% more than the breadth
  a = l * b →               -- Area formula
  a = 460 := by sorry       -- The area is 460 square meters

end NUMINAMATH_CALUDE_rectangle_area_l119_11998


namespace NUMINAMATH_CALUDE_rationalize_denominator_l119_11980

theorem rationalize_denominator :
  ∃ (x : ℝ), x = (Real.sqrt 6 - 1) ∧
  x = (Real.sqrt 8 + Real.sqrt 3) / (Real.sqrt 2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l119_11980


namespace NUMINAMATH_CALUDE_fish_sales_revenue_l119_11983

theorem fish_sales_revenue : 
  let first_week_quantity : ℕ := 50
  let first_week_price : ℚ := 10
  let second_week_quantity_multiplier : ℕ := 3
  let second_week_discount_percentage : ℚ := 25 / 100

  let first_week_revenue := first_week_quantity * first_week_price
  let second_week_quantity := first_week_quantity * second_week_quantity_multiplier
  let second_week_price := first_week_price * (1 - second_week_discount_percentage)
  let second_week_revenue := second_week_quantity * second_week_price
  let total_revenue := first_week_revenue + second_week_revenue

  total_revenue = 1625 := by
sorry

end NUMINAMATH_CALUDE_fish_sales_revenue_l119_11983


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l119_11914

theorem absolute_value_simplification : |(-5^2 - 6 * 2)| = 37 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l119_11914


namespace NUMINAMATH_CALUDE_remaining_balance_is_correct_l119_11974

-- Define the problem parameters
def initial_balance : ℚ := 100
def daily_spending : ℚ := 8
def exchange_fee_rate : ℚ := 0.03
def days_in_week : ℕ := 7
def flat_fee : ℚ := 2
def bill_denomination : ℚ := 5

-- Define the function to calculate the remaining balance
def calculate_remaining_balance : ℚ := 
  let total_daily_spend := daily_spending * (1 + exchange_fee_rate)
  let weekly_spend := total_daily_spend * days_in_week
  let balance_after_spending := initial_balance - weekly_spend
  let balance_after_fee := balance_after_spending - flat_fee
  let bills_taken := (balance_after_fee / bill_denomination).floor * bill_denomination
  balance_after_fee - bills_taken

-- Theorem statement
theorem remaining_balance_is_correct : 
  calculate_remaining_balance = 0.32 := by sorry

end NUMINAMATH_CALUDE_remaining_balance_is_correct_l119_11974


namespace NUMINAMATH_CALUDE_product_of_numbers_l119_11907

theorem product_of_numbers (x y : ℝ) 
  (h1 : (x + y) / (x - y) = 7)
  (h2 : (x * y) / (x - y) = 24) : 
  x * y = 48 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l119_11907


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l119_11987

theorem multiplication_puzzle (a b : ℕ) : 
  a < 10 → b < 10 → (20 + a) * (10 * b + 3) = 989 → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l119_11987


namespace NUMINAMATH_CALUDE_equation_solution_l119_11976

theorem equation_solution : 
  ∃! x : ℝ, x ≠ (1/2) ∧ (5*x + 1) / (2*x^2 + 5*x - 3) = 2*x / (2*x - 1) ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l119_11976


namespace NUMINAMATH_CALUDE_fixed_point_on_line_fixed_point_unique_l119_11901

/-- The line equation passing through a fixed point -/
def line_equation (k x y : ℝ) : Prop :=
  y = k * (x - 2) + 3

/-- The fixed point through which the line always passes -/
def fixed_point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the fixed point satisfies the line equation for all k -/
theorem fixed_point_on_line :
  ∀ k : ℝ, line_equation k (fixed_point.1) (fixed_point.2) :=
sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_unique :
  ∀ x y : ℝ, (∀ k : ℝ, line_equation k x y) → (x, y) = fixed_point :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_fixed_point_unique_l119_11901


namespace NUMINAMATH_CALUDE_triangle_base_difference_l119_11973

theorem triangle_base_difference (b h : ℝ) (hb : b > 0) (hh : h > 0) : 
  let b_A := (0.99 * b * h) / (0.9 * h)
  let h_A := 0.9 * h
  b_A = 1.1 * b := by sorry

end NUMINAMATH_CALUDE_triangle_base_difference_l119_11973


namespace NUMINAMATH_CALUDE_parallel_not_coincident_condition_l119_11929

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- Two lines are coincident if they are parallel and have the same y-intercept -/
def coincident (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop := 
  parallel a₁ b₁ a₂ b₂ ∧ (c₁ * b₂ = c₂ * b₁)

/-- The necessary and sufficient condition for the given lines to be parallel and not coincident -/
theorem parallel_not_coincident_condition : 
  ∀ a : ℝ, (parallel a 2 3 (a-1) ∧ 
            ¬coincident a 2 (-3*a) 3 (a-1) (7-a)) ↔ 
           (a = 3) := by sorry

end NUMINAMATH_CALUDE_parallel_not_coincident_condition_l119_11929


namespace NUMINAMATH_CALUDE_weight_of_b_l119_11925

-- Define the variables
variable (wa wb wc : ℝ)
variable (ha hb hc : ℝ)

-- Define the conditions
def average_weight_abc : Prop := (wa + wb + wc) / 3 = 45
def average_weight_ab : Prop := (wa + wb) / 2 = 40
def average_weight_bc : Prop := (wb + wc) / 2 = 43
def average_height_ac : Prop := (ha + hc) / 2 = 155

-- Define the quadratic relationship
def weight_height_relation (w h : ℝ) : Prop := w = 2 * h^2 + 3 * h - 5

-- Theorem statement
theorem weight_of_b (hwabc : average_weight_abc wa wb wc)
                    (hwab : average_weight_ab wa wb)
                    (hwbc : average_weight_bc wb wc)
                    (hhac : average_height_ac ha hc)
                    (hwa : weight_height_relation wa ha)
                    (hwb : weight_height_relation wb hb)
                    (hwc : weight_height_relation wc hc) :
  wb = 31 := by sorry

end NUMINAMATH_CALUDE_weight_of_b_l119_11925


namespace NUMINAMATH_CALUDE_simplify_expression_l119_11904

theorem simplify_expression (x y : ℝ) : -x + y - 2*x - 3*y = -3*x - 2*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l119_11904
