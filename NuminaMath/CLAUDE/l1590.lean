import Mathlib

namespace NUMINAMATH_CALUDE_horner_v₁_value_l1590_159025

def horner_polynomial (x : ℝ) : ℝ := 12 + 3*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def v₀ : ℝ := 3

def a_n_minus_1 : ℝ := 5

def x : ℝ := -4

def v₁ : ℝ := v₀ * x + a_n_minus_1

theorem horner_v₁_value : v₁ = -7 := by sorry

end NUMINAMATH_CALUDE_horner_v₁_value_l1590_159025


namespace NUMINAMATH_CALUDE_scooter_purchase_price_l1590_159095

/-- Proves that given the conditions of the scooter purchase, repair, sale, and profit,
    the original purchase price must be $4700. -/
theorem scooter_purchase_price (P : ℝ) : 
  P > 0 →
  5800 - (P + 600) = (9.433962264150944 / 100) * (P + 600) →
  P = 4700 := by
sorry

end NUMINAMATH_CALUDE_scooter_purchase_price_l1590_159095


namespace NUMINAMATH_CALUDE_log_properties_l1590_159009

-- Define approximate values for log₁₀ 2 and log₁₀ 3
def log10_2 : ℝ := 0.3010
def log10_3 : ℝ := 0.4771

-- Define the properties to be proved
theorem log_properties :
  let log10_27 := 3 * log10_3
  let log10_100_div_9 := 2 - 2 * log10_3
  let log10_sqrt_10 := (1 : ℝ) / 2
  (log10_27 = 3 * log10_3) ∧
  (log10_100_div_9 = 2 - 2 * log10_3) ∧
  (log10_sqrt_10 = (1 : ℝ) / 2) := by
  sorry


end NUMINAMATH_CALUDE_log_properties_l1590_159009


namespace NUMINAMATH_CALUDE_min_value_m_plus_2n_l1590_159060

/-- The function f(x) = |x-a| where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

/-- The theorem stating the minimum value of m + 2n -/
theorem min_value_m_plus_2n (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f 2 x ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3) →
  1/m + 1/(2*n) = 2 →
  ∀ k l, k > 0 → l > 0 → 1/k + 1/(2*l) = 2 → m + 2*n ≤ k + 2*l :=
by sorry

end NUMINAMATH_CALUDE_min_value_m_plus_2n_l1590_159060


namespace NUMINAMATH_CALUDE_function_transformation_l1590_159040

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_transformation (x : ℝ) : 
  (∀ y, f (y + 1) = 3 * y + 2) → f x = 3 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l1590_159040


namespace NUMINAMATH_CALUDE_school_election_votes_l1590_159089

theorem school_election_votes (total_votes : ℕ) 
  (h1 : 45 = (3 : ℕ) * total_votes / 8)
  (h2 : (1 : ℕ) * total_votes / 4 + (3 : ℕ) * total_votes / 8 ≤ total_votes) : 
  total_votes = 120 := by
sorry

end NUMINAMATH_CALUDE_school_election_votes_l1590_159089


namespace NUMINAMATH_CALUDE_max_sin_A_in_triangle_l1590_159026

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi

-- Define the theorem
theorem max_sin_A_in_triangle (t : Triangle) 
  (h : Real.tan t.A / Real.tan t.B + Real.tan t.A / Real.tan t.C = 3) :
  Real.sin t.A ≤ Real.sqrt 21 / 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sin_A_in_triangle_l1590_159026


namespace NUMINAMATH_CALUDE_girl_scout_cookies_l1590_159070

theorem girl_scout_cookies (total_goal : ℕ) (boxes_left : ℕ) (first_customer : ℕ) : 
  total_goal = 150 →
  boxes_left = 75 →
  first_customer = 5 →
  let second_customer := 4 * first_customer
  let third_customer := second_customer / 2
  let fourth_customer := 3 * third_customer
  let sold_to_first_four := first_customer + second_customer + third_customer + fourth_customer
  total_goal - boxes_left - sold_to_first_four = 10 := by
sorry


end NUMINAMATH_CALUDE_girl_scout_cookies_l1590_159070


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_two_l1590_159029

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ b₁ m₂ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + b₁ = y ↔ m₂ * x + b₂ = y) ↔ m₁ = m₂

/-- Definition of line l₁ -/
def l₁ (x y : ℝ) : Prop := 2 * x - y - 1 = 0

/-- Definition of line l₂ -/
def l₂ (a x y : ℝ) : Prop := 2 * x + (a + 1) * y + 2 = 0

/-- Theorem: If l₁ is parallel to l₂, then a = -2 -/
theorem parallel_lines_imply_a_eq_neg_two :
  (∀ x y : ℝ, l₁ x y ↔ l₂ a x y) → a = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_two_l1590_159029


namespace NUMINAMATH_CALUDE_solve_candy_bar_problem_l1590_159043

def candy_bar_problem (initial_amount : ℚ) (num_candy_bars : ℕ) (remaining_amount : ℚ) : Prop :=
  ∃ (price_per_bar : ℚ),
    initial_amount - num_candy_bars * price_per_bar = remaining_amount ∧
    price_per_bar > 0

theorem solve_candy_bar_problem :
  candy_bar_problem 4 10 1 → (4 : ℚ) - 1 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_candy_bar_problem_l1590_159043


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l1590_159034

theorem rectangle_areas_sum : 
  let width : ℝ := 2
  let lengths : List ℝ := [1, 8, 27]
  let areas : List ℝ := lengths.map (λ l => width * l)
  areas.sum = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l1590_159034


namespace NUMINAMATH_CALUDE_greatest_b_quadratic_inequality_l1590_159080

theorem greatest_b_quadratic_inequality :
  ∃ b : ℝ, b^2 - 14*b + 45 ≤ 0 ∧
  ∀ x : ℝ, x^2 - 14*x + 45 ≤ 0 → x ≤ b ∧
  b = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_b_quadratic_inequality_l1590_159080


namespace NUMINAMATH_CALUDE_selina_pants_sold_l1590_159054

/-- Represents the number of pants Selina sold -/
def pants_sold : ℕ := sorry

/-- The price of each pair of pants -/
def pants_price : ℕ := 5

/-- The price of each pair of shorts -/
def shorts_price : ℕ := 3

/-- The price of each shirt -/
def shirt_price : ℕ := 4

/-- The number of shorts Selina sold -/
def shorts_sold : ℕ := 5

/-- The number of shirts Selina sold -/
def shirts_sold : ℕ := 5

/-- The price of each new shirt Selina bought -/
def new_shirt_price : ℕ := 10

/-- The number of new shirts Selina bought -/
def new_shirts_bought : ℕ := 2

/-- The amount of money Selina left the store with -/
def money_left : ℕ := 30

theorem selina_pants_sold : 
  pants_sold * pants_price + 
  shorts_sold * shorts_price + 
  shirts_sold * shirt_price = 
  money_left + new_shirts_bought * new_shirt_price ∧ 
  pants_sold = 3 := by sorry

end NUMINAMATH_CALUDE_selina_pants_sold_l1590_159054


namespace NUMINAMATH_CALUDE_triangle_inequality_variant_l1590_159044

theorem triangle_inequality_variant (x y z : ℝ) :
  (|x| < |y - z| ∧ |y| < |z - x|) → |z| ≥ |x - y| := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_variant_l1590_159044


namespace NUMINAMATH_CALUDE_radio_cost_price_l1590_159052

theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1275)
  (h2 : loss_percentage = 15) : 
  ∃ (cost_price : ℝ), 
    cost_price = 1500 ∧ 
    selling_price = cost_price * (1 - loss_percentage / 100) := by
sorry

end NUMINAMATH_CALUDE_radio_cost_price_l1590_159052


namespace NUMINAMATH_CALUDE_cylinder_in_sphere_volume_l1590_159068

theorem cylinder_in_sphere_volume (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 6) (h_cylinder : r_cylinder = 4) :
  let h_cylinder := 2 * (r_sphere ^ 2 - r_cylinder ^ 2).sqrt
  let v_sphere := (4 / 3) * π * r_sphere ^ 3
  let v_cylinder := π * r_cylinder ^ 2 * h_cylinder
  (v_sphere - v_cylinder) / π = 288 - 64 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_cylinder_in_sphere_volume_l1590_159068


namespace NUMINAMATH_CALUDE_vector_decomposition_l1590_159049

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![13, 2, 7]
def p : Fin 3 → ℝ := ![5, 1, 0]
def q : Fin 3 → ℝ := ![2, -1, 3]
def r : Fin 3 → ℝ := ![1, 0, -1]

/-- Theorem stating the decomposition of x in terms of p, q, and r -/
theorem vector_decomposition :
  x = fun i => 3 * p i + q i - 4 * r i := by
  sorry

end NUMINAMATH_CALUDE_vector_decomposition_l1590_159049


namespace NUMINAMATH_CALUDE_solution_for_a_l1590_159074

theorem solution_for_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (eq1 : a + 2 / b = 17) (eq2 : b + 2 / a = 1 / 3) :
  a = 6 ∨ a = 17 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_a_l1590_159074


namespace NUMINAMATH_CALUDE_sneaker_coupon_value_l1590_159085

/-- Proves that the coupon value is $10 given the conditions of the sneaker purchase problem -/
theorem sneaker_coupon_value (original_price : ℝ) (membership_discount : ℝ) (final_price : ℝ)
  (h1 : original_price = 120)
  (h2 : membership_discount = 0.1)
  (h3 : final_price = 99) :
  ∃ (coupon_value : ℝ), 
    (1 - membership_discount) * (original_price - coupon_value) = final_price ∧
    coupon_value = 10 :=
by sorry

end NUMINAMATH_CALUDE_sneaker_coupon_value_l1590_159085


namespace NUMINAMATH_CALUDE_smallest_sum_consecutive_integers_l1590_159022

theorem smallest_sum_consecutive_integers : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (l : ℤ), n = (9 : ℤ) * (l + 4)) ∧ 
  (∃ (m : ℤ), n = (5 : ℤ) * (2 * m + 9)) ∧ 
  (∃ (k : ℤ), n = (11 : ℤ) * (k + 5)) ∧ 
  (∀ (n' : ℕ), n' > 0 → 
    (∃ (l : ℤ), n' = (9 : ℤ) * (l + 4)) → 
    (∃ (m : ℤ), n' = (5 : ℤ) * (2 * m + 9)) → 
    (∃ (k : ℤ), n' = (11 : ℤ) * (k + 5)) → 
    n ≤ n') ∧ 
  n = 495 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_consecutive_integers_l1590_159022


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l1590_159069

-- Define the divisibility relation
def divides (m n : ℕ) : Prop := ∃ k : ℕ, n = m * k

-- Define an infinite set of natural numbers
def InfiniteSet (S : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃ m ∈ S, m > n

theorem divisibility_implies_equality (a b : ℕ) 
  (h : ∃ S : Set ℕ, InfiniteSet S ∧ ∀ n ∈ S, divides (a^n + b^n) (a^(n+1) + b^(n+1))) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l1590_159069


namespace NUMINAMATH_CALUDE_track_length_l1590_159017

theorem track_length : ∀ (x : ℝ), 
  (x > 0) →  -- track length is positive
  (120 / (x/2 - 120) = (x/2 + 50) / (3*x/2 - 170)) →  -- ratio of distances is constant
  x = 418 := by
sorry

end NUMINAMATH_CALUDE_track_length_l1590_159017


namespace NUMINAMATH_CALUDE_joe_age_proof_l1590_159087

theorem joe_age_proof (joe james : ℕ) : 
  joe = james + 10 →
  2 * (joe + 8) = 3 * (james + 8) →
  joe = 22 := by
sorry

end NUMINAMATH_CALUDE_joe_age_proof_l1590_159087


namespace NUMINAMATH_CALUDE_magnitude_comparison_l1590_159076

theorem magnitude_comparison (a b c : ℝ) 
  (ha : a > 0) 
  (hbc : b * c > a^2) 
  (heq : a^2 - 2*a*b + c^2 = 0) : 
  b > c ∧ c > a :=
by sorry

end NUMINAMATH_CALUDE_magnitude_comparison_l1590_159076


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l1590_159013

/-- The trajectory C of point P in the Cartesian coordinate system xOy,
    where the sum of distances from P to (0, -√3) and (0, √3) equals 4 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 / 4 + p.1^2 = 1}

/-- The line that intersects C -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- Theorem stating the properties of the trajectory C and its intersection with the line -/
theorem trajectory_and_intersection :
  ∀ k : ℝ,
  (∀ p : ℝ × ℝ, p ∈ C → (Real.sqrt ((p.1)^2 + (p.2 + Real.sqrt 3)^2) +
                         Real.sqrt ((p.1)^2 + (p.2 - Real.sqrt 3)^2) = 4)) ∧
  (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ line k ∧ B ∈ line k ∧
    (k = 1/2 ∨ k = -1/2) ↔ (A.1 * B.1 + A.2 * B.2 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l1590_159013


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_one_inequality_holds_iff_m_in_range_l1590_159000

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| - |x + 3*m|

-- Theorem for part I
theorem solution_set_when_m_is_one :
  {x : ℝ | f x 1 ≥ 1} = {x : ℝ | x ≤ -3/2} := by sorry

-- Theorem for part II
theorem inequality_holds_iff_m_in_range :
  (∀ (x t : ℝ), f x m < |2 + t| + |t - 1|) ↔ (0 < m ∧ m < 3/4) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_one_inequality_holds_iff_m_in_range_l1590_159000


namespace NUMINAMATH_CALUDE_cost_price_is_640_l1590_159033

/-- The cost price of an article given its selling price and profit percentage -/
def costPrice (sellingPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  sellingPrice / (1 + profitPercentage / 100)

/-- Theorem stating that the cost price is 640 given the conditions -/
theorem cost_price_is_640 (sellingPrice : ℚ) (profitPercentage : ℚ) 
  (h1 : sellingPrice = 800)
  (h2 : profitPercentage = 25) : 
  costPrice sellingPrice profitPercentage = 640 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_640_l1590_159033


namespace NUMINAMATH_CALUDE_exam_average_l1590_159077

theorem exam_average (students_group1 : ℕ) (average1 : ℚ) 
  (students_group2 : ℕ) (average2 : ℚ) : 
  students_group1 = 15 → 
  average1 = 70/100 → 
  students_group2 = 10 → 
  average2 = 90/100 → 
  (students_group1 * average1 + students_group2 * average2) / (students_group1 + students_group2) = 78/100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l1590_159077


namespace NUMINAMATH_CALUDE_line_AB_equation_l1590_159062

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 2.5)^2 + (y - 0.5)^2 = 2.5

-- Define point P
def P : ℝ × ℝ := (4, 1)

-- Define the line AB
def lineAB (x y : ℝ) : Prop := 3*x + y - 4 = 0

-- Theorem statement
theorem line_AB_equation :
  ∀ x y : ℝ,
  (circle1 x y ∧ circle2 x y) →
  lineAB x y :=
sorry

end NUMINAMATH_CALUDE_line_AB_equation_l1590_159062


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1590_159028

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (2 * x - 6) / (x + 1)) ↔ x ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1590_159028


namespace NUMINAMATH_CALUDE_x_squared_less_than_abs_x_plus_two_l1590_159045

theorem x_squared_less_than_abs_x_plus_two (x : ℝ) :
  x^2 < |x| + 2 ↔ -2 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_x_squared_less_than_abs_x_plus_two_l1590_159045


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1590_159021

theorem triangle_angle_c (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π → 
  a > 0 → b > 0 → c > 0 →
  a = 2 →
  b + c = 2 * a →
  3 * Real.sin A = 5 * Real.sin B →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  c^2 = a^2 + b^2 - 2*a*b * Real.cos C →
  C = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1590_159021


namespace NUMINAMATH_CALUDE_statement_is_universal_l1590_159051

-- Define the concept of a line
def Line : Type := sorry

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the property of two lines intersecting
def intersect (l1 l2 : Line) : Prop := sorry

-- Define the property of a plane passing through two lines
def passes_through (p : Plane) (l1 l2 : Line) : Prop := sorry

-- Define the statement as a proposition
def statement : Prop :=
  ∀ l1 l2 : Line, intersect l1 l2 → ∃! p : Plane, passes_through p l1 l2

-- Theorem to prove that the statement is a universal proposition
theorem statement_is_universal : 
  (∀ l1 l2 : Line, intersect l1 l2 → ∃! p : Plane, passes_through p l1 l2) ↔ statement :=
sorry

end NUMINAMATH_CALUDE_statement_is_universal_l1590_159051


namespace NUMINAMATH_CALUDE_sqrt_132_plus_46_sqrt_11_l1590_159010

theorem sqrt_132_plus_46_sqrt_11 :
  ∃ (a b c : ℤ), 
    (132 + 46 * Real.sqrt 11 : ℝ).sqrt = a + b * Real.sqrt c ∧
    ¬ ∃ (d : ℤ), c = d * d ∧
    ∃ (e f : ℤ), c = e * f ∧ (∀ (g : ℤ), g * g ∣ e → g = 1 ∨ g = -1) ∧
                             (∀ (h : ℤ), h * h ∣ f → h = 1 ∨ h = -1) :=
sorry

end NUMINAMATH_CALUDE_sqrt_132_plus_46_sqrt_11_l1590_159010


namespace NUMINAMATH_CALUDE_impurity_reduction_proof_l1590_159001

/-- Represents the reduction factor of impurities after each filtration -/
def reduction_factor : ℝ := 0.8

/-- Represents the target impurity level as a fraction of the original -/
def target_impurity : ℝ := 0.05

/-- The minimum number of filtrations required to reduce impurities below the target level -/
def min_filtrations : ℕ := 14

theorem impurity_reduction_proof :
  (reduction_factor ^ min_filtrations : ℝ) < target_impurity ∧
  ∀ n : ℕ, n < min_filtrations → (reduction_factor ^ n : ℝ) ≥ target_impurity :=
sorry

end NUMINAMATH_CALUDE_impurity_reduction_proof_l1590_159001


namespace NUMINAMATH_CALUDE_student_average_weight_l1590_159064

theorem student_average_weight 
  (n : ℕ) 
  (teacher_weight : ℝ) 
  (weight_increase : ℝ) : 
  n = 24 → 
  teacher_weight = 45 → 
  weight_increase = 0.4 → 
  (n * 35 + teacher_weight) / (n + 1) = 35 + weight_increase :=
by sorry

end NUMINAMATH_CALUDE_student_average_weight_l1590_159064


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l1590_159090

theorem no_solution_absolute_value_equation :
  ¬∃ x : ℝ, |(-2 * x + 1)| + 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l1590_159090


namespace NUMINAMATH_CALUDE_water_bottle_drinking_time_l1590_159011

/-- Proves that drinking a 2-liter bottle of water with 40 ml sips every 5 minutes takes 250 minutes -/
theorem water_bottle_drinking_time :
  let bottle_capacity_liters : ℝ := 2
  let ml_per_liter : ℝ := 1000
  let sip_volume_ml : ℝ := 40
  let minutes_per_sip : ℝ := 5
  
  bottle_capacity_liters * ml_per_liter / sip_volume_ml * minutes_per_sip = 250 := by
  sorry


end NUMINAMATH_CALUDE_water_bottle_drinking_time_l1590_159011


namespace NUMINAMATH_CALUDE_negation_equivalence_l1590_159065

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1590_159065


namespace NUMINAMATH_CALUDE_problem_statement_l1590_159020

-- Define the sets A and B
def A : Set ℝ := Set.Ioo (-2) 2
def B (a : ℝ) : Set ℝ := Set.Ioo a (1 - a)

-- State the theorem
theorem problem_statement (a : ℝ) (h : a < 0) :
  (A ∪ B a = B a → a ≤ -2) ∧
  (A ∩ B a = B a → a ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1590_159020


namespace NUMINAMATH_CALUDE_customers_who_tipped_l1590_159086

theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) : 
  initial_customers = 39 →
  additional_customers = 12 →
  non_tipping_customers = 49 →
  initial_customers + additional_customers - non_tipping_customers = 2 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tipped_l1590_159086


namespace NUMINAMATH_CALUDE_six_arts_competition_l1590_159048

theorem six_arts_competition (a b c : ℕ) (h_abc : a > b ∧ b > c) :
  (∃ (x y z : ℕ),
    x + y + z = 6 ∧
    a * x + b * y + c * z = 26 ∧
    (∃ (p q r : ℕ),
      p + q + r = 6 ∧
      a * p + b * q + c * r = 11 ∧
      p = 1 ∧
      (∃ (u v w : ℕ),
        u + v + w = 6 ∧
        a * u + b * v + c * w = 11 ∧
        a + b + c = 8))) →
  (∃ (p q r : ℕ),
    p + q + r = 6 ∧
    a * p + b * q + c * r = 11 ∧
    p = 1 ∧
    r = 4) :=
by sorry

end NUMINAMATH_CALUDE_six_arts_competition_l1590_159048


namespace NUMINAMATH_CALUDE_probability_of_27_l1590_159067

/-- Represents a die with numbered and blank faces -/
structure Die :=
  (total_faces : ℕ)
  (numbered_faces : ℕ)
  (min_number : ℕ)
  (max_number : ℕ)

/-- Calculates the number of ways to get a sum with two dice -/
def waysToGetSum (d1 d2 : Die) (sum : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of possible outcomes when rolling two dice -/
def totalOutcomes (d1 d2 : Die) : ℕ :=
  d1.total_faces * d2.total_faces

/-- Theorem: Probability of rolling 27 with given dice is 3/100 -/
theorem probability_of_27 :
  let die1 : Die := ⟨20, 18, 1, 18⟩
  let die2 : Die := ⟨20, 17, 3, 20⟩
  (waysToGetSum die1 die2 27 : ℚ) / (totalOutcomes die1 die2 : ℚ) = 3 / 100 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_27_l1590_159067


namespace NUMINAMATH_CALUDE_parabola_parameter_distance_l1590_159081

/-- Parabola type representing y = ax^2 -/
structure Parabola where
  a : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate the distance from a point to the directrix of a parabola -/
noncomputable def distance_to_directrix (p : Parabola) (pt : Point) : ℝ :=
  if p.a > 0 then
    abs (pt.y + 1 / (4 * p.a))
  else
    abs (pt.y - 1 / (4 * p.a))

/-- Theorem stating the relationship between the parabola parameter and the distance to directrix -/
theorem parabola_parameter_distance (p : Parabola) :
  let m : Point := ⟨2, 1⟩
  distance_to_directrix p m = 2 →
  p.a = 1/4 ∨ p.a = -1/12 :=
sorry

end NUMINAMATH_CALUDE_parabola_parameter_distance_l1590_159081


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1590_159078

theorem complex_fraction_sum (a b : ℝ) :
  (3 + b * Complex.I) / (1 - Complex.I) = a + b * Complex.I →
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1590_159078


namespace NUMINAMATH_CALUDE_same_color_probability_value_l1590_159024

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 5

def same_color_probability : ℚ :=
  (Nat.choose white_balls drawn_balls + Nat.choose black_balls drawn_balls) /
  Nat.choose total_balls drawn_balls

theorem same_color_probability_value :
  same_color_probability = 77 / 3003 := by sorry

end NUMINAMATH_CALUDE_same_color_probability_value_l1590_159024


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_to_height_ratio_l1590_159035

/-- A regular tetrahedron with an inscribed sphere -/
structure RegularTetrahedronWithInscribedSphere where
  /-- The height of the regular tetrahedron -/
  height : ℝ
  /-- The radius of the inscribed sphere -/
  sphereRadius : ℝ
  /-- The area of one face of the regular tetrahedron -/
  faceArea : ℝ
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The sphere radius is positive -/
  sphereRadius_pos : 0 < sphereRadius
  /-- The face area is positive -/
  faceArea_pos : 0 < faceArea
  /-- Volume relation between the tetrahedron and the four pyramids formed by the inscribed sphere -/
  volume_relation : 4 * (1/3 * faceArea * sphereRadius) = 1/3 * faceArea * height

/-- The ratio of the radius of the inscribed sphere to the height of the regular tetrahedron is 1/4 -/
theorem inscribed_sphere_radius_to_height_ratio 
  (t : RegularTetrahedronWithInscribedSphere) : t.sphereRadius = 1/4 * t.height := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_to_height_ratio_l1590_159035


namespace NUMINAMATH_CALUDE_songs_added_l1590_159057

theorem songs_added (initial : ℕ) (deleted : ℕ) (final : ℕ) : 
  initial = 11 → deleted = 7 → final = 28 → final - (initial - deleted) = 24 :=
by sorry

end NUMINAMATH_CALUDE_songs_added_l1590_159057


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1590_159084

/-- The area of a square with perimeter 40 feet is 100 square feet -/
theorem square_area_from_perimeter :
  ∀ (s : ℝ), s > 0 → 4 * s = 40 → s^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1590_159084


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1590_159004

/-- Given two vectors OA and OB in 2D space, where OA is perpendicular to AB, prove that m = 4 -/
theorem perpendicular_vectors (OA OB : ℝ × ℝ) (m : ℝ) : 
  OA = (-1, 2) → 
  OB = (3, m) → 
  OA.1 * (OB.1 - OA.1) + OA.2 * (OB.2 - OA.2) = 0 → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1590_159004


namespace NUMINAMATH_CALUDE_smallest_shift_l1590_159007

open Real

theorem smallest_shift (n : ℝ) : n > 0 ∧ 
  (∀ x, cos (2 * π * x - π / 3) = sin (2 * π * (x - n) + π / 3)) → 
  n ≥ 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l1590_159007


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1590_159005

theorem isosceles_triangle (A B C : Real) (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  ∃ (s : Real), s > 0 ∧ Real.sin A = s ∧ Real.sin B = s := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1590_159005


namespace NUMINAMATH_CALUDE_fiona_owns_three_hoodies_l1590_159047

/-- The number of hoodies Fiona owns -/
def fiona_hoodies : ℕ := sorry

/-- The number of hoodies Casey owns -/
def casey_hoodies : ℕ := sorry

/-- The total number of hoodies Fiona and Casey own -/
def total_hoodies : ℕ := 8

theorem fiona_owns_three_hoodies :
  fiona_hoodies = 3 ∧ casey_hoodies = fiona_hoodies + 2 ∧ fiona_hoodies + casey_hoodies = total_hoodies :=
sorry

end NUMINAMATH_CALUDE_fiona_owns_three_hoodies_l1590_159047


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l1590_159063

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l1590_159063


namespace NUMINAMATH_CALUDE_parabola_curve_intersection_l1590_159061

/-- A parabola with equation y² = 4x and focus at (1, 0) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- A curve with equation y = k/x where k > 0 -/
def Curve (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k / p.1 ∧ k > 0}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- A point P is perpendicular to the x-axis if its x-coordinate is 1 -/
def isPerpendicular (P : ℝ × ℝ) : Prop :=
  P.1 = 1

theorem parabola_curve_intersection (k : ℝ) :
  ∃ P : ℝ × ℝ, P ∈ Parabola ∧ P ∈ Curve k ∧ isPerpendicular P → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_curve_intersection_l1590_159061


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1590_159050

theorem quadratic_roots_property (m : ℝ) (r s : ℝ) : 
  (∀ x, x^2 - (m+1)*x + m = 0 ↔ x = r ∨ x = s) →
  |r + s - 2*r*s| = |1 - m| := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1590_159050


namespace NUMINAMATH_CALUDE_problem_statement_l1590_159083

theorem problem_statement (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 440 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1590_159083


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1590_159092

def A : Set ℝ := {x | x ≥ -1}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1590_159092


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l1590_159012

theorem sum_of_x_and_y_on_circle (x y : ℝ) 
  (h : 4 * x^2 + 4 * y^2 = 40 * x - 24 * y + 64) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l1590_159012


namespace NUMINAMATH_CALUDE_inequalities_proof_l1590_159056

theorem inequalities_proof :
  (Real.log (Real.sqrt 2) < Real.sqrt 2 / 2) ∧
  (2 * Real.log (Real.sin (1/8) + Real.cos (1/8)) < 1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1590_159056


namespace NUMINAMATH_CALUDE_river_speed_proof_l1590_159093

theorem river_speed_proof (rowing_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) 
  (h1 : rowing_speed = 6)
  (h2 : total_time = 1)
  (h3 : total_distance = 5.76) :
  ∃ (river_speed : ℝ),
    river_speed = 1.2 ∧
    (total_distance / 2) / (rowing_speed - river_speed) +
    (total_distance / 2) / (rowing_speed + river_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_river_speed_proof_l1590_159093


namespace NUMINAMATH_CALUDE_two_pump_filling_time_l1590_159002

/-- Given two pumps with filling rates of 1/3 tank per hour and 4 tanks per hour respectively,
    the time taken to fill a tank when both pumps work together is 3/13 hours. -/
theorem two_pump_filling_time :
  let small_pump_rate : ℚ := 1/3  -- Rate of small pump in tanks per hour
  let large_pump_rate : ℚ := 4    -- Rate of large pump in tanks per hour
  let combined_rate : ℚ := small_pump_rate + large_pump_rate
  let filling_time : ℚ := 1 / combined_rate
  filling_time = 3/13 := by
  sorry

end NUMINAMATH_CALUDE_two_pump_filling_time_l1590_159002


namespace NUMINAMATH_CALUDE_range_of_a_l1590_159091

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Condition: For all x ∈ ℝ, f'(x) < x
axiom f'_less_than_x : ∀ x, f' x < x

-- Condition: f(1-a) - f(a) ≤ 1/2 - a
axiom inequality_condition : ∀ a, f (1 - a) - f a ≤ 1/2 - a

-- Theorem: The range of values for a is a ≤ 1/2
theorem range_of_a : ∀ a, (∀ x, f (1 - x) - f x ≤ 1/2 - x) → a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1590_159091


namespace NUMINAMATH_CALUDE_no_eulerian_or_hamiltonian_path_l1590_159019

/-- A graph representing the science museum layout. -/
structure MuseumGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  bipartite : Finset Nat × Finset Nat
  degree_three : Finset Nat

/-- Predicate for the existence of an Eulerian path in the graph. -/
def has_eulerian_path (g : MuseumGraph) : Prop :=
  ∃ (path : List (Nat × Nat)), path.Nodup ∧ path.length = g.edges.card

/-- Predicate for the existence of a Hamiltonian path in the graph. -/
def has_hamiltonian_path (g : MuseumGraph) : Prop :=
  ∃ (path : List Nat), path.Nodup ∧ path.length = g.vertices.card

/-- The main theorem stating the non-existence of Eulerian and Hamiltonian paths. -/
theorem no_eulerian_or_hamiltonian_path (g : MuseumGraph)
  (h1 : g.vertices.card = 19)
  (h2 : g.edges.card = 30)
  (h3 : g.bipartite.1.card = 7 ∧ g.bipartite.2.card = 12)
  (h4 : g.degree_three.card ≥ 6) :
  ¬(has_eulerian_path g) ∧ ¬(has_hamiltonian_path g) := by
  sorry

#check no_eulerian_or_hamiltonian_path

end NUMINAMATH_CALUDE_no_eulerian_or_hamiltonian_path_l1590_159019


namespace NUMINAMATH_CALUDE_original_solution_concentration_l1590_159088

/-- Proves that given the conditions, the original solution's concentration is 50% -/
theorem original_solution_concentration
  (replaced_portion : ℝ)
  (h_replaced : replaced_portion = 0.8181818181818182)
  (x : ℝ)
  (h_result : x / 100 * (1 - replaced_portion) + 30 / 100 * replaced_portion = 40 / 100) :
  x = 50 :=
sorry

end NUMINAMATH_CALUDE_original_solution_concentration_l1590_159088


namespace NUMINAMATH_CALUDE_missing_number_is_five_l1590_159036

/-- Represents the sum of two adjacent children's favorite numbers -/
structure AdjacentSum :=
  (value : ℕ)

/-- Represents a circle of children with their favorite numbers -/
structure ChildrenCircle :=
  (size : ℕ)
  (sums : List AdjacentSum)

/-- Calculates the missing number in the circle -/
def calculateMissingNumber (circle : ChildrenCircle) : ℕ :=
  sorry

/-- Theorem stating that the missing number is 5 -/
theorem missing_number_is_five (circle : ChildrenCircle) 
  (h1 : circle.size = 6)
  (h2 : circle.sums = [⟨8⟩, ⟨14⟩, ⟨12⟩])
  : calculateMissingNumber circle = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_five_l1590_159036


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1590_159032

theorem line_passes_through_point (A B C : ℝ) :
  A - B + C = 0 →
  ∀ (x y : ℝ), A * x + B * y + C = 0 ↔ (x = 1 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1590_159032


namespace NUMINAMATH_CALUDE_exists_natural_not_in_five_gp_l1590_159031

/-- A geometric progression with integer terms -/
structure GeometricProgression where
  first_term : ℤ
  common_ratio : ℤ
  common_ratio_nonzero : common_ratio ≠ 0

/-- The nth term of a geometric progression -/
def GeometricProgression.nth_term (gp : GeometricProgression) (n : ℕ) : ℤ :=
  gp.first_term * gp.common_ratio ^ n

/-- Theorem: There exists a natural number not in any of five given geometric progressions -/
theorem exists_natural_not_in_five_gp (gp1 gp2 gp3 gp4 gp5 : GeometricProgression) :
  ∃ (k : ℕ), (∀ n : ℕ, gp1.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp2.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp3.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp4.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp5.nth_term n ≠ k) :=
  sorry

end NUMINAMATH_CALUDE_exists_natural_not_in_five_gp_l1590_159031


namespace NUMINAMATH_CALUDE_ruler_cost_l1590_159023

theorem ruler_cost (total_spent : ℕ) (notebook_cost : ℕ) (num_pencils : ℕ) (pencil_cost : ℕ) 
  (h1 : total_spent = 74)
  (h2 : notebook_cost = 35)
  (h3 : num_pencils = 3)
  (h4 : pencil_cost = 7) :
  total_spent - (notebook_cost + num_pencils * pencil_cost) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ruler_cost_l1590_159023


namespace NUMINAMATH_CALUDE_rearrangement_time_proof_l1590_159042

/-- The number of hours required to write all rearrangements of a 12-letter name -/
def rearrangement_hours : ℕ := 798336

/-- The number of letters in the name -/
def name_length : ℕ := 12

/-- The number of arrangements written per minute -/
def arrangements_per_minute : ℕ := 10

/-- Theorem stating the time required to write all rearrangements -/
theorem rearrangement_time_proof :
  rearrangement_hours = (name_length.factorial / arrangements_per_minute) / 60 := by
  sorry


end NUMINAMATH_CALUDE_rearrangement_time_proof_l1590_159042


namespace NUMINAMATH_CALUDE_red_to_blue_bead_ratio_l1590_159039

theorem red_to_blue_bead_ratio :
  let red_beads : ℕ := 30
  let blue_beads : ℕ := 20
  (red_beads : ℚ) / blue_beads = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_red_to_blue_bead_ratio_l1590_159039


namespace NUMINAMATH_CALUDE_y_bounds_for_n_4_l1590_159066

/-- The function y(t) = (n-1)t² - 10t + 10 -/
def y (n : ℕ) (t : ℝ) : ℝ := (n - 1) * t^2 - 10 * t + 10

/-- The theorem stating that for n = 4, y(t) is always between 0 and 30 for t in (0,4] -/
theorem y_bounds_for_n_4 :
  ∀ t : ℝ, t > 0 → t ≤ 4 → 0 < y 4 t ∧ y 4 t ≤ 30 := by sorry

end NUMINAMATH_CALUDE_y_bounds_for_n_4_l1590_159066


namespace NUMINAMATH_CALUDE_time_after_1007_hours_l1590_159058

def clock_add (current_time hours_elapsed : ℕ) : ℕ :=
  (current_time + hours_elapsed) % 12

theorem time_after_1007_hours :
  let current_time := 5
  let hours_elapsed := 1007
  clock_add current_time hours_elapsed = 4 := by
sorry

end NUMINAMATH_CALUDE_time_after_1007_hours_l1590_159058


namespace NUMINAMATH_CALUDE_inequality_solutions_range_l1590_159037

theorem inequality_solutions_range (a : ℝ) : 
  (∀ x : ℕ+, x < a ↔ x ≤ 5) → (5 < a ∧ a < 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_range_l1590_159037


namespace NUMINAMATH_CALUDE_smallest_w_l1590_159014

theorem smallest_w (w : ℕ+) 
  (h1 : (2^5 : ℕ) ∣ (936 * w))
  (h2 : (3^3 : ℕ) ∣ (936 * w))
  (h3 : (10^2 : ℕ) ∣ (936 * w)) :
  w ≥ 900 ∧ ∃ (v : ℕ+), v = 900 ∧ 
    (2^5 : ℕ) ∣ (936 * v) ∧ 
    (3^3 : ℕ) ∣ (936 * v) ∧ 
    (10^2 : ℕ) ∣ (936 * v) :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l1590_159014


namespace NUMINAMATH_CALUDE_age_ratio_l1590_159082

def kul_age : ℕ := 22
def saras_age : ℕ := 33

theorem age_ratio : 
  (saras_age : ℚ) / (kul_age : ℚ) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_age_ratio_l1590_159082


namespace NUMINAMATH_CALUDE_glass_to_sand_ratio_l1590_159003

/-- Represents the number of items in each container --/
structure BeachTreasures where
  bucket : ℕ  -- number of seashells in the bucket
  jar : ℕ     -- number of glass pieces in the jar
  bag : ℕ     -- number of sand dollars in the bag

/-- The conditions of Simon's beach treasure collection --/
def simons_treasures : BeachTreasures → Prop
  | t => t.bucket = 5 * t.jar ∧ 
         t.jar = t.bag ∧ 
         t.bag = 10 ∧ 
         t.bucket + t.jar + t.bag = 190

/-- The theorem stating the ratio of glass pieces to sand dollars --/
theorem glass_to_sand_ratio (t : BeachTreasures) 
  (h : simons_treasures t) : t.jar / t.bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_glass_to_sand_ratio_l1590_159003


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1590_159098

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) ≤ 3 * (a * b + b * c + c * a) / (2 * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1590_159098


namespace NUMINAMATH_CALUDE_cube_root_of_1331_l1590_159015

theorem cube_root_of_1331 (y : ℝ) (h1 : y > 0) (h2 : y^3 = 1331) : y = 11 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_1331_l1590_159015


namespace NUMINAMATH_CALUDE_s_of_one_eq_394_div_25_l1590_159038

/-- Given functions t and s, prove that s(1) = 394/25 -/
theorem s_of_one_eq_394_div_25 
  (t : ℝ → ℝ)
  (s : ℝ → ℝ)
  (h1 : ∀ x, t x = 5 * x - 12)
  (h2 : ∀ x, s (t x) = x^2 + 5 * x - 4) :
  s 1 = 394 / 25 := by
  sorry

end NUMINAMATH_CALUDE_s_of_one_eq_394_div_25_l1590_159038


namespace NUMINAMATH_CALUDE_painter_problem_l1590_159041

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculates the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Proves that for the given scenario, the time to paint the remaining rooms is 49 hours. -/
theorem painter_problem :
  let total_rooms : ℕ := 12
  let time_per_room : ℕ := 7
  let painted_rooms : ℕ := 5
  time_to_paint_remaining total_rooms time_per_room painted_rooms = 49 := by
  sorry


end NUMINAMATH_CALUDE_painter_problem_l1590_159041


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1590_159030

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 2 + a 5 + a 8 = 39) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 117 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1590_159030


namespace NUMINAMATH_CALUDE_combined_figure_area_l1590_159046

/-- The area of a figure consisting of a twelve-sided polygon and a rhombus -/
theorem combined_figure_area (polygon_area : ℝ) (rhombus_diagonal1 : ℝ) (rhombus_diagonal2 : ℝ) :
  polygon_area = 13 →
  rhombus_diagonal1 = 2 →
  rhombus_diagonal2 = 1 →
  polygon_area + (rhombus_diagonal1 * rhombus_diagonal2) / 2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_combined_figure_area_l1590_159046


namespace NUMINAMATH_CALUDE_range_of_p_l1590_159016

/-- The function p(x) = x^4 + 6x^2 + 9 -/
def p (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

/-- The domain of the function -/
def domain : Set ℝ := { x | x ≥ 0 }

/-- The range of the function -/
def range : Set ℝ := { y | ∃ x ∈ domain, p x = y }

theorem range_of_p : range = { y | y ≥ 9 } := by sorry

end NUMINAMATH_CALUDE_range_of_p_l1590_159016


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1590_159075

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 5) (hy : |y| = 3) (hxy : x - y > 0) :
  x + y = 8 ∨ x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1590_159075


namespace NUMINAMATH_CALUDE_museum_group_time_l1590_159094

/-- Proves that the time taken for each group to go through the museum is 24 minutes -/
theorem museum_group_time (total_students : ℕ) (num_groups : ℕ) (time_per_student : ℕ) : 
  total_students = 18 → num_groups = 3 → time_per_student = 4 → 
  (total_students / num_groups) * time_per_student = 24 := by
  sorry

end NUMINAMATH_CALUDE_museum_group_time_l1590_159094


namespace NUMINAMATH_CALUDE_probability_A_selected_l1590_159099

/-- The number of people in the group -/
def group_size : ℕ := 4

/-- The number of representatives to be selected -/
def representatives : ℕ := 2

/-- The probability of person A being selected as a representative -/
def prob_A_selected : ℚ := 1/2

/-- Theorem stating the probability of person A being selected as a representative -/
theorem probability_A_selected :
  prob_A_selected = (representatives : ℚ) / group_size :=
by sorry

end NUMINAMATH_CALUDE_probability_A_selected_l1590_159099


namespace NUMINAMATH_CALUDE_not_divisible_by_three_times_sum_of_products_l1590_159053

theorem not_divisible_by_three_times_sum_of_products (x y z : ℕ+) :
  ¬ (3 * (x * y + y * z + z * x) ∣ x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_three_times_sum_of_products_l1590_159053


namespace NUMINAMATH_CALUDE_red_balls_count_l1590_159097

theorem red_balls_count (total white green yellow purple : ℕ) (prob : ℚ) : 
  total = 60 ∧ 
  white = 22 ∧ 
  green = 10 ∧ 
  yellow = 7 ∧ 
  purple = 6 ∧ 
  prob = 65 / 100 ∧ 
  (white + green + yellow : ℚ) / total = prob →
  total - (white + green + yellow + purple) = 0 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1590_159097


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1590_159059

-- Define the equation
def ellipse_equation (x y k : ℝ) : Prop := x^2 + k*y^2 = 2

-- Define what it means for the equation to represent an ellipse with foci on the x-axis
def is_ellipse_with_foci_on_x_axis (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  ∀ (x y : ℝ), ellipse_equation x y k ↔ (x^2 / (a^2) + y^2 / (b^2) = 1)

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse_with_foci_on_x_axis k ↔ k > 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1590_159059


namespace NUMINAMATH_CALUDE_carols_peanuts_l1590_159018

/-- Represents the number of peanuts Carol's father gave her -/
def peanuts_from_father (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem carols_peanuts : peanuts_from_father 2 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_carols_peanuts_l1590_159018


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1590_159096

-- Define the hyperbola and its properties
def Hyperbola (m : ℝ) : Prop :=
  m > 0 ∧ ∃ x y : ℝ, x^2 / m - y^2 = 1

-- Define the asymptotic line
def AsymptoticLine (x y : ℝ) : Prop :=
  x + 3 * y = 0

-- Theorem statement
theorem hyperbola_asymptote (m : ℝ) :
  Hyperbola m → (∃ x y : ℝ, AsymptoticLine x y) → m = 9 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1590_159096


namespace NUMINAMATH_CALUDE_swim_team_girls_count_l1590_159055

theorem swim_team_girls_count :
  ∀ (boys girls coaches managers : ℕ),
  girls = 5 * boys →
  coaches = 4 →
  managers = 4 →
  boys + girls + coaches + managers = 104 →
  girls = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_swim_team_girls_count_l1590_159055


namespace NUMINAMATH_CALUDE_video_game_time_l1590_159072

/-- 
Proves that given the conditions of the problem, 
the time spent playing video games is 9 hours.
-/
theorem video_game_time 
  (study_rate : ℝ)  -- Rate at which grade increases per hour of studying
  (final_grade : ℝ)  -- Final grade achieved
  (study_ratio : ℝ)  -- Ratio of study time to gaming time
  (h_study_rate : study_rate = 15)  -- Grade increases by 15 points per hour of studying
  (h_final_grade : final_grade = 45)  -- Final grade is 45 points
  (h_study_ratio : study_ratio = 1/3)  -- Study time is 1/3 of gaming time
  : ∃ (game_time : ℝ), game_time = 9 := by
  sorry

end NUMINAMATH_CALUDE_video_game_time_l1590_159072


namespace NUMINAMATH_CALUDE_quartic_equation_solution_l1590_159027

theorem quartic_equation_solution :
  ∀ x : ℂ, x^4 - 16*x^2 + 256 = 0 ↔ x = 4 ∨ x = -4 :=
by sorry

end NUMINAMATH_CALUDE_quartic_equation_solution_l1590_159027


namespace NUMINAMATH_CALUDE_inscribed_circle_existence_l1590_159071

-- Define a convex polygon type
structure ConvexPolygon where
  -- Add necessary fields (this is a simplified representation)
  vertices : List (ℝ × ℝ)
  is_convex : Bool

-- Define a function to represent the outward translation of polygon sides
def translate_sides (p : ConvexPolygon) (distance : ℝ) : ConvexPolygon :=
  sorry

-- Define a similarity relation between polygons
def is_similar (p1 p2 : ConvexPolygon) : Prop :=
  sorry

-- Define the property of having parallel and proportional sides
def has_parallel_proportional_sides (p1 p2 : ConvexPolygon) : Prop :=
  sorry

-- Define what it means for a circle to be inscribed in a polygon
def has_inscribed_circle (p : ConvexPolygon) : Prop :=
  sorry

-- The main theorem
theorem inscribed_circle_existence 
  (p : ConvexPolygon) 
  (h_convex : p.is_convex)
  (h_similar : is_similar p (translate_sides p 1))
  (h_parallel_prop : has_parallel_proportional_sides p (translate_sides p 1)) :
  has_inscribed_circle p :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_existence_l1590_159071


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l1590_159079

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis,
    prove that its semi-minor axis has length √3 -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ) 
  (focus : ℝ × ℝ) 
  (semi_major_endpoint : ℝ × ℝ) 
  (h1 : center = (-3, 1)) 
  (h2 : focus = (-3, 0)) 
  (h3 : semi_major_endpoint = (-3, 3)) : 
  Real.sqrt ((center.2 - semi_major_endpoint.2)^2 - (center.2 - focus.2)^2) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l1590_159079


namespace NUMINAMATH_CALUDE_collectible_figure_price_l1590_159073

theorem collectible_figure_price (sneaker_cost lawn_count lawn_price job_hours job_rate figure_count : ℕ) 
  (h1 : sneaker_cost = 92)
  (h2 : lawn_count = 3)
  (h3 : lawn_price = 8)
  (h4 : job_hours = 10)
  (h5 : job_rate = 5)
  (h6 : figure_count = 2) :
  let lawn_earnings := lawn_count * lawn_price
  let job_earnings := job_hours * job_rate
  let total_earnings := lawn_earnings + job_earnings
  let remaining_amount := sneaker_cost - total_earnings
  (remaining_amount / figure_count : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_collectible_figure_price_l1590_159073


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1590_159006

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x^2 * |x| = 3*x + 4 ∧ 
  ∀ (y : ℝ), y^2 * |y| = 3*y + 4 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1590_159006


namespace NUMINAMATH_CALUDE_eliminate_denominators_l1590_159008

theorem eliminate_denominators (x : ℝ) :
  (2*x - 1) / 3 - (3*x - 4) / 4 = 1 ↔ 4*(2*x - 1) - 3*(3*x - 4) = 12 :=
by sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l1590_159008
