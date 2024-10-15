import Mathlib

namespace NUMINAMATH_CALUDE_set_intersection_problem_l968_96802

/-- Given sets M and N, prove their intersection -/
theorem set_intersection_problem (M N : Set ℝ) 
  (hM : M = {x : ℝ | -2 < x ∧ x < 2})
  (hN : N = {x : ℝ | |x - 1| ≤ 2}) :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l968_96802


namespace NUMINAMATH_CALUDE_circle_f_value_l968_96872

def Circle (d e f : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + d * p.1 + e * p.2 + f = 0}

def isDiameter (c : ℝ × ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∀ p : ℝ × ℝ, c p → 
    (p.1 - midpoint.1)^2 + (p.2 - midpoint.2)^2 ≤ ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4

theorem circle_f_value (d e f : ℝ) :
  isDiameter (Circle d e f) (20, 22) (10, 30) → f = 860 := by
  sorry

end NUMINAMATH_CALUDE_circle_f_value_l968_96872


namespace NUMINAMATH_CALUDE_six_students_three_colleges_l968_96878

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins,
    where each bin must contain at least one object. -/
def distributeWithMinimum (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n

/-- The specific case for 6 students and 3 colleges -/
theorem six_students_three_colleges :
  distributeWithMinimum 6 3 = 540 := by
  sorry

end NUMINAMATH_CALUDE_six_students_three_colleges_l968_96878


namespace NUMINAMATH_CALUDE_triangle_properties_l968_96868

-- Define the triangle
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b)
  (h2 : Real.cos t.B = 1/4)
  (h3 : t.b = 2) :
  Real.sin t.C / Real.sin t.A = 2 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l968_96868


namespace NUMINAMATH_CALUDE_new_number_properties_l968_96899

def new_number (a b : ℕ) : ℕ := a * b + a + b

def is_new_number (n : ℕ) : Prop :=
  ∃ a b, new_number a b = n

theorem new_number_properties :
  (¬ is_new_number 2008) ∧
  (∀ a b : ℕ, 2 ∣ (new_number a b + 1)) ∧
  (∀ a b : ℕ, 10 ∣ (new_number a b + 1)) :=
sorry

end NUMINAMATH_CALUDE_new_number_properties_l968_96899


namespace NUMINAMATH_CALUDE_unique_polygon_pair_l968_96836

/-- The interior angle of a regular polygon with n sides --/
def interior_angle (n : ℕ) : ℚ :=
  180 - 360 / n

/-- The condition for the ratio of interior angles to be 5:3 --/
def angle_ratio_condition (a b : ℕ) : Prop :=
  interior_angle a / interior_angle b = 5 / 3

/-- The main theorem --/
theorem unique_polygon_pair :
  ∃! (pair : ℕ × ℕ), 
    pair.1 > 2 ∧ 
    pair.2 > 2 ∧ 
    angle_ratio_condition pair.1 pair.2 :=
sorry

end NUMINAMATH_CALUDE_unique_polygon_pair_l968_96836


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l968_96839

/-- The speed of a canoe rowing downstream, given its upstream speed against a stream -/
theorem canoe_downstream_speed
  (upstream_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : upstream_speed = 4)
  (h2 : stream_speed = 4) :
  upstream_speed + 2 * stream_speed = 12 :=
by sorry

end NUMINAMATH_CALUDE_canoe_downstream_speed_l968_96839


namespace NUMINAMATH_CALUDE_quiz_win_probability_l968_96825

def num_questions : ℕ := 4
def num_choices : ℕ := 4
def min_correct : ℕ := 3

def prob_correct_one : ℚ := 1 / num_choices

def prob_all_correct : ℚ := prob_correct_one ^ num_questions

def prob_three_correct : ℚ := num_questions * (prob_correct_one ^ 3) * (1 - prob_correct_one)

theorem quiz_win_probability :
  prob_all_correct + prob_three_correct = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_quiz_win_probability_l968_96825


namespace NUMINAMATH_CALUDE_tan_alpha_3_implies_sin_2alpha_over_cos_alpha_squared_6_l968_96807

theorem tan_alpha_3_implies_sin_2alpha_over_cos_alpha_squared_6 (α : Real) 
  (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (Real.cos α)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_implies_sin_2alpha_over_cos_alpha_squared_6_l968_96807


namespace NUMINAMATH_CALUDE_eighteenth_term_of_sequence_l968_96875

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem eighteenth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 6) :
  arithmetic_sequence a₁ (a₂ - a₁) 18 = 105 :=
by
  sorry

#check eighteenth_term_of_sequence 3 9 (by norm_num)

end NUMINAMATH_CALUDE_eighteenth_term_of_sequence_l968_96875


namespace NUMINAMATH_CALUDE_second_throw_difference_l968_96883

/-- Represents the number of skips for each throw -/
structure Throws :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)
  (fifth : ℕ)

/-- Conditions for the stone skipping problem -/
def StoneSkippingProblem (t : Throws) : Prop :=
  t.third = 2 * t.second ∧
  t.fourth = t.third - 3 ∧
  t.fifth = t.fourth + 1 ∧
  t.fifth = 8 ∧
  t.first + t.second + t.third + t.fourth + t.fifth = 33

theorem second_throw_difference (t : Throws) 
  (h : StoneSkippingProblem t) : t.second - t.first = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_throw_difference_l968_96883


namespace NUMINAMATH_CALUDE_max_colored_cells_l968_96867

/-- Represents a cell in the 8x8 square --/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Predicate to check if four cells form a rectangle with sides parallel to the edges --/
def formsRectangle (c1 c2 c3 c4 : Cell) : Prop :=
  (c1.row = c2.row ∧ c3.row = c4.row ∧ c1.col = c3.col ∧ c2.col = c4.col) ∨
  (c1.row = c3.row ∧ c2.row = c4.row ∧ c1.col = c2.col ∧ c3.col = c4.col)

/-- The main theorem --/
theorem max_colored_cells :
  ∃ (S : Finset Cell),
    S.card = 24 ∧
    (∀ (c1 c2 c3 c4 : Cell),
      c1 ∈ S → c2 ∈ S → c3 ∈ S → c4 ∈ S →
      c1 ≠ c2 → c1 ≠ c3 → c1 ≠ c4 → c2 ≠ c3 → c2 ≠ c4 → c3 ≠ c4 →
      ¬formsRectangle c1 c2 c3 c4) ∧
    (∀ (T : Finset Cell),
      T.card > 24 →
      ∃ (c1 c2 c3 c4 : Cell),
        c1 ∈ T ∧ c2 ∈ T ∧ c3 ∈ T ∧ c4 ∈ T ∧
        c1 ≠ c2 ∧ c1 ≠ c3 ∧ c1 ≠ c4 ∧ c2 ≠ c3 ∧ c2 ≠ c4 ∧ c3 ≠ c4 ∧
        formsRectangle c1 c2 c3 c4) :=
by sorry


end NUMINAMATH_CALUDE_max_colored_cells_l968_96867


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l968_96877

/-- The volume of a sphere inscribed in a cube with a given diagonal -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 10) :
  let s := d / Real.sqrt 3
  let r := s / 2
  (4 / 3) * Real.pi * r ^ 3 = (500 * Real.sqrt 3 * Real.pi) / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l968_96877


namespace NUMINAMATH_CALUDE_card_flipping_theorem_l968_96897

/-- Represents the sum of visible numbers on cards after i flips -/
def sum_after_flips (n : ℕ) (initial_config : Fin n → Bool) (i : Fin (n + 1)) : ℕ :=
  sorry

/-- The statement to be proved -/
theorem card_flipping_theorem (n : ℕ) (h : 0 < n) :
  (∀ initial_config : Fin n → Bool,
    ∃ i j : Fin (n + 1), i ≠ j ∧ sum_after_flips n initial_config i ≠ sum_after_flips n initial_config j) ∧
  (∀ initial_config : Fin n → Bool,
    ∃ r : Fin (n + 1), sum_after_flips n initial_config r = n / 2 ∨ sum_after_flips n initial_config r = (n + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_card_flipping_theorem_l968_96897


namespace NUMINAMATH_CALUDE_closest_to_sqrt_65_minus_sqrt_63_l968_96881

theorem closest_to_sqrt_65_minus_sqrt_63 :
  let options : List ℝ := [0.12, 0.13, 0.14, 0.15, 0.16]
  ∀ x ∈ options, x ≠ 0.13 →
    |Real.sqrt 65 - Real.sqrt 63 - 0.13| < |Real.sqrt 65 - Real.sqrt 63 - x| := by
  sorry

end NUMINAMATH_CALUDE_closest_to_sqrt_65_minus_sqrt_63_l968_96881


namespace NUMINAMATH_CALUDE_f_properties_l968_96845

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x =>
  if x < 0 then (x - 1)^2
  else if x = 0 then 0
  else -(x + 1)^2

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = -(x + 1)^2) →  -- given condition
  (∀ x < 0, f x = (x - 1)^2) ∧  -- part of the analytic expression
  (f 0 = 0) ∧  -- part of the analytic expression
  (∀ m, f (m^2 + 2*m) + f m > 0 ↔ -3 < m ∧ m < 0) :=  -- range of m
by sorry

end NUMINAMATH_CALUDE_f_properties_l968_96845


namespace NUMINAMATH_CALUDE_quadratic_integer_values_l968_96892

theorem quadratic_integer_values (a b c : ℝ) :
  (∀ x : ℤ, ∃ n : ℤ, a * x^2 + b * x + c = n) ↔
  (∃ m : ℤ, 2 * a = m) ∧ (∃ n : ℤ, a + b = n) ∧ (∃ p : ℤ, c = p) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_values_l968_96892


namespace NUMINAMATH_CALUDE_children_who_got_off_bus_l968_96847

theorem children_who_got_off_bus (initial_children : ℕ) (remaining_children : ℕ) 
  (h1 : initial_children = 43) 
  (h2 : remaining_children = 21) : 
  initial_children - remaining_children = 22 := by
  sorry

end NUMINAMATH_CALUDE_children_who_got_off_bus_l968_96847


namespace NUMINAMATH_CALUDE_inequality_equivalence_l968_96876

theorem inequality_equivalence (x : ℝ) : (x - 2) / 2 ≥ (7 - x) / 3 ↔ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l968_96876


namespace NUMINAMATH_CALUDE_hemisphere_volume_calculation_l968_96884

/-- Given a total volume of water and the number of hemisphere containers,
    calculate the volume of each hemisphere container. -/
def hemisphere_volume (total_volume : ℚ) (num_containers : ℕ) : ℚ :=
  total_volume / num_containers

/-- Theorem stating that the volume of each hemisphere container is 4 L
    when 2735 containers are used to hold 10940 L of water. -/
theorem hemisphere_volume_calculation :
  hemisphere_volume 10940 2735 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_volume_calculation_l968_96884


namespace NUMINAMATH_CALUDE_ellipse_circle_intersection_l968_96894

/-- The ellipse C defined by x^2 + 16y^2 = 16 -/
def ellipse_C (x y : ℝ) : Prop := x^2 + 16 * y^2 = 16

/-- The circle Γ with center (0, h) and radius r -/
def circle_Γ (h r : ℝ) (x y : ℝ) : Prop := x^2 + (y - h)^2 = r^2

/-- The foci of ellipse C -/
def foci : Set (ℝ × ℝ) := {(-Real.sqrt 15, 0), (Real.sqrt 15, 0)}

theorem ellipse_circle_intersection (a b : ℝ) :
  (∃ r h, r ∈ Set.Icc a b ∧
    (∀ (f : ℝ × ℝ), f ∈ foci → circle_Γ h r f.1 f.2) ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧ ellipse_C x₃ y₃ ∧ ellipse_C x₄ y₄ ∧
      circle_Γ h r x₁ y₁ ∧ circle_Γ h r x₂ y₂ ∧ circle_Γ h r x₃ y₃ ∧ circle_Γ h r x₄ y₄ ∧
      (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
      (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄))) →
  a + b = Real.sqrt 15 + 8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_circle_intersection_l968_96894


namespace NUMINAMATH_CALUDE_donation_difference_l968_96862

def total_donation : ℕ := 1000
def treetown_forest_donation : ℕ := 570

theorem donation_difference : 
  treetown_forest_donation - (total_donation - treetown_forest_donation) = 140 := by
  sorry

end NUMINAMATH_CALUDE_donation_difference_l968_96862


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l968_96886

theorem binomial_expansion_problem (m n : ℕ) (hm : m ≠ 0) (hn : n ≥ 2) :
  (∀ k, 0 ≤ k ∧ k ≤ n → (n.choose k) * m^k ≤ (n.choose 5) * m^5) ∧
  (n.choose 2) * m^2 = 9 * (n.choose 1) * m →
  m = 2 ∧ n = 10 ∧ (1 - 2 * 9)^10 % 6 = 1 :=
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l968_96886


namespace NUMINAMATH_CALUDE_alpha_cheaper_at_min_shirts_l968_96860

/-- Alpha T-Shirt Company's pricing model -/
def alpha_cost (n : ℕ) : ℚ := 80 + 12 * n

/-- Omega T-Shirt Company's pricing model -/
def omega_cost (n : ℕ) : ℚ := 10 + 18 * n

/-- The minimum number of shirts for which Alpha becomes cheaper -/
def min_shirts_for_alpha : ℕ := 12

theorem alpha_cheaper_at_min_shirts :
  alpha_cost min_shirts_for_alpha < omega_cost min_shirts_for_alpha ∧
  ∀ m : ℕ, m < min_shirts_for_alpha → alpha_cost m ≥ omega_cost m :=
by sorry

end NUMINAMATH_CALUDE_alpha_cheaper_at_min_shirts_l968_96860


namespace NUMINAMATH_CALUDE_prime_from_divisibility_condition_l968_96842

-- Define the divisibility condition
def divisibility_condition (n : ℤ) : Prop :=
  ∀ d : ℤ, d ∣ n → (d + 1) ∣ (n + 1)

-- Theorem statement
theorem prime_from_divisibility_condition (n : ℤ) :
  divisibility_condition n → Nat.Prime (Int.natAbs n) :=
by
  sorry

end NUMINAMATH_CALUDE_prime_from_divisibility_condition_l968_96842


namespace NUMINAMATH_CALUDE_root_conditions_imply_inequalities_l968_96849

theorem root_conditions_imply_inequalities (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b > 0) (hc : c ≠ 0)
  (h_distinct : ∃ x y : ℝ, x ≠ y ∧ 
    a * x^2 + b * x - c = 0 ∧ 
    a * y^2 + b * y - c = 0)
  (h_cubic : ∀ x : ℝ, a * x^2 + b * x - c = 0 → 
    x^3 + b * x^2 + a * x - c = 0) :
  a * b * c > 16 ∧ a * b * c ≥ 3125 / 108 := by
  sorry

end NUMINAMATH_CALUDE_root_conditions_imply_inequalities_l968_96849


namespace NUMINAMATH_CALUDE_frank_problems_per_type_l968_96880

/-- The number of math problems composed by Bill -/
def bill_problems : ℕ := 20

/-- The number of math problems composed by Ryan -/
def ryan_problems : ℕ := 2 * bill_problems

/-- The number of math problems composed by Frank -/
def frank_problems : ℕ := 3 * ryan_problems

/-- The number of different types of math problems -/
def problem_types : ℕ := 4

theorem frank_problems_per_type :
  frank_problems / problem_types = 30 := by sorry

end NUMINAMATH_CALUDE_frank_problems_per_type_l968_96880


namespace NUMINAMATH_CALUDE_train_speed_l968_96890

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length time : ℝ) (h1 : length = 160) (h2 : time = 8) :
  length / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l968_96890


namespace NUMINAMATH_CALUDE_bagel_cut_theorem_l968_96817

/-- The number of pieces resulting from cutting a bagel -/
def bagel_pieces (n : ℕ) : ℕ := n + 1

/-- Theorem: Cutting a bagel with 10 cuts results in 11 pieces -/
theorem bagel_cut_theorem :
  bagel_pieces 10 = 11 :=
by sorry

end NUMINAMATH_CALUDE_bagel_cut_theorem_l968_96817


namespace NUMINAMATH_CALUDE_power_expression_evaluation_l968_96840

theorem power_expression_evaluation (b : ℕ) (h : b = 4) : b^3 * b^6 / b^2 = 16384 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_evaluation_l968_96840


namespace NUMINAMATH_CALUDE_solve_equation_l968_96887

theorem solve_equation (x : ℝ) (h : Real.sqrt ((2 / x) + 2) = 4 / 3) : x = -9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l968_96887


namespace NUMINAMATH_CALUDE_disjunction_true_l968_96812

theorem disjunction_true : 
  (∀ x > 0, ∃ y, y = x + 1/(2*x) ∧ y ≥ 1 ∧ ∀ z, z = x + 1/(2*x) → z ≥ y) ∨ 
  (∀ x > 1, x^2 + 2*x - 3 > 0) := by
sorry

end NUMINAMATH_CALUDE_disjunction_true_l968_96812


namespace NUMINAMATH_CALUDE_function_range_contained_in_unit_interval_l968_96864

/-- Given a function f: ℝ → ℝ satisfying (f x)^2 ≤ f y for all x > y,
    prove that the range of f is contained in [0, 1]. -/
theorem function_range_contained_in_unit_interval
  (f : ℝ → ℝ) (h : ∀ x y, x > y → (f x)^2 ≤ f y) :
  ∀ x, 0 ≤ f x ∧ f x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_contained_in_unit_interval_l968_96864


namespace NUMINAMATH_CALUDE_f_symmetric_f_upper_bound_f_solution_range_l968_96801

noncomputable section

def f (x : ℝ) : ℝ := Real.log ((1 + x) / (x - 1))

theorem f_symmetric : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem f_upper_bound : ∀ x : ℝ, x > 1 → f x + Real.log (0.5 * (x - 1)) < -1 := by sorry

theorem f_solution_range (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 3 ∧ f x = Real.log (0.5 * (x + k))) ↔ k ∈ Set.Icc (-1) 1 := by sorry

end

end NUMINAMATH_CALUDE_f_symmetric_f_upper_bound_f_solution_range_l968_96801


namespace NUMINAMATH_CALUDE_sector_area_l968_96815

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 24) (h2 : θ = 110 * π / 180) :
  r^2 * θ / 2 = 176 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l968_96815


namespace NUMINAMATH_CALUDE_student_scores_l968_96832

theorem student_scores (M P C : ℕ) : 
  M + P = 50 →
  (M + C) / 2 = 35 →
  C > P →
  C - P = 20 := by
sorry

end NUMINAMATH_CALUDE_student_scores_l968_96832


namespace NUMINAMATH_CALUDE_odd_increasing_function_inequality_l968_96828

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem odd_increasing_function_inequality 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_incr : is_increasing f) 
  (m : ℝ) 
  (h_ineq : ∀ θ : ℝ, f (Real.cos (2 * θ) - 5) + f (2 * m + 4 * Real.sin θ) > 0) :
  m > 5 := by
sorry

end NUMINAMATH_CALUDE_odd_increasing_function_inequality_l968_96828


namespace NUMINAMATH_CALUDE_main_line_probability_l968_96809

/-- Represents a train schedule -/
structure TrainSchedule where
  start_time : ℕ
  frequency : ℕ

/-- Calculates the probability of getting the main line train -/
def probability_main_line (main : TrainSchedule) (harbor : TrainSchedule) : ℚ :=
  1 / 2

/-- Theorem stating that the probability of getting the main line train is 1/2 -/
theorem main_line_probability 
  (main : TrainSchedule) 
  (harbor : TrainSchedule) 
  (h1 : main.start_time = 0)
  (h2 : harbor.start_time = 2)
  (h3 : main.frequency = 10)
  (h4 : harbor.frequency = 10) :
  probability_main_line main harbor = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_main_line_probability_l968_96809


namespace NUMINAMATH_CALUDE_equation_solutions_l968_96855

theorem equation_solutions : 
  ∀ x : ℝ, x ≠ 3 ∧ x ≠ 5 →
  ((x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 3) * (x - 2) * (x - 1)) / 
  ((x - 3) * (x - 5) * (x - 3)) = 1 ↔ 
  x = 1 ∨ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l968_96855


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l968_96822

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + 2*a - 3) (a + 3)
  (z.re = 0 ∧ z.im ≠ 0) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l968_96822


namespace NUMINAMATH_CALUDE_max_followers_count_l968_96827

/-- Represents the types of islanders --/
inductive IslanderType
  | Knight
  | Liar
  | Follower

/-- Represents an answer to the question --/
inductive Answer
  | Yes
  | No

/-- Defines the properties of the island and its inhabitants --/
structure Island where
  totalPopulation : Nat
  knightCount : Nat
  liarCount : Nat
  followerCount : Nat
  yesAnswers : Nat
  noAnswers : Nat

/-- Defines the conditions of the problem --/
def isValidIsland (i : Island) : Prop :=
  i.totalPopulation = 2018 ∧
  i.knightCount + i.liarCount + i.followerCount = i.totalPopulation ∧
  i.yesAnswers = 1009 ∧
  i.noAnswers = i.totalPopulation - i.yesAnswers

/-- The main theorem to prove --/
theorem max_followers_count (i : Island) (h : isValidIsland i) :
  i.followerCount ≤ 1009 ∧ ∃ (j : Island), isValidIsland j ∧ j.followerCount = 1009 :=
sorry

end NUMINAMATH_CALUDE_max_followers_count_l968_96827


namespace NUMINAMATH_CALUDE_second_worker_time_l968_96858

/-- The time it takes for two workers to load a truck together -/
def combined_time : ℚ := 30 / 11

/-- The time it takes for the first worker to load a truck alone -/
def worker1_time : ℚ := 6

/-- Theorem stating that the second worker's time to load a truck alone is 5 hours -/
theorem second_worker_time :
  ∃ (worker2_time : ℚ),
    worker2_time = 5 ∧
    1 / worker1_time + 1 / worker2_time = 1 / combined_time :=
by sorry

end NUMINAMATH_CALUDE_second_worker_time_l968_96858


namespace NUMINAMATH_CALUDE_complex_sum_powers_l968_96850

theorem complex_sum_powers (z : ℂ) (h : z = (1 - Complex.I) / (1 + Complex.I)) :
  z^2 + z^4 + z^6 + z^8 + z^10 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l968_96850


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l968_96823

theorem negation_of_universal_statement :
  (¬∀ x : ℝ, x^2 - 3*x + 2 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l968_96823


namespace NUMINAMATH_CALUDE_cube_root_of_cube_l968_96866

theorem cube_root_of_cube (x : ℝ) : x^(1/3)^3 = x := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_cube_l968_96866


namespace NUMINAMATH_CALUDE_betty_nuts_purchase_l968_96821

/-- The number of packs of nuts Betty wants to buy -/
def num_packs : ℕ := 20

/-- Betty's age -/
def betty_age : ℕ := 50

/-- Doug's age -/
def doug_age : ℕ := 40

/-- Cost of one pack of nuts -/
def pack_cost : ℕ := 100

/-- Total cost Betty wants to spend on nuts -/
def total_cost : ℕ := 2000

theorem betty_nuts_purchase :
  (2 * betty_age = pack_cost) ∧
  (betty_age + doug_age = 90) ∧
  (num_packs * pack_cost = total_cost) →
  num_packs = 20 := by sorry

end NUMINAMATH_CALUDE_betty_nuts_purchase_l968_96821


namespace NUMINAMATH_CALUDE_problem_solution_l968_96810

theorem problem_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 4) (h3 : c^2 / a = 4) :
  a = 64^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l968_96810


namespace NUMINAMATH_CALUDE_tim_children_treats_l968_96857

/-- Calculates the total number of treats Tim's children get while trick-or-treating --/
def total_treats (num_children : ℕ) 
                 (houses_hour1 houses_hour2 houses_hour3 houses_hour4 : ℕ)
                 (treats_hour1 treats_hour2 treats_hour3 treats_hour4 : ℕ) : ℕ :=
  (houses_hour1 * treats_hour1 * num_children) +
  (houses_hour2 * treats_hour2 * num_children) +
  (houses_hour3 * treats_hour3 * num_children) +
  (houses_hour4 * treats_hour4 * num_children)

/-- Theorem stating that Tim's children get 237 treats in total --/
theorem tim_children_treats : 
  total_treats 3 4 6 5 7 3 4 3 4 = 237 := by
  sorry


end NUMINAMATH_CALUDE_tim_children_treats_l968_96857


namespace NUMINAMATH_CALUDE_three_number_problem_l968_96859

theorem three_number_problem (x y z : ℝ) : 
  x + y + z = 19 → 
  y^2 = x * z → 
  y = (2/3) * z → 
  x = 4 ∧ y = 6 ∧ z = 9 := by
sorry

end NUMINAMATH_CALUDE_three_number_problem_l968_96859


namespace NUMINAMATH_CALUDE_tank_leak_consistency_l968_96819

/-- Proves that a leak emptying a tank in 12 hours without an inlet pipe is consistent with the given scenario. -/
theorem tank_leak_consistency 
  (tank_capacity : ℝ) 
  (inlet_rate : ℝ) 
  (emptying_time_with_inlet : ℝ) 
  (emptying_time_without_inlet : ℝ) : 
  tank_capacity = 5760 ∧ 
  inlet_rate = 4 ∧ 
  emptying_time_with_inlet = 8 * 60 ∧ 
  emptying_time_without_inlet = 12 * 60 → 
  ∃ (leak_rate : ℝ), 
    leak_rate > 0 ∧
    tank_capacity / leak_rate = emptying_time_without_inlet ∧
    tank_capacity / (leak_rate - inlet_rate) = emptying_time_with_inlet :=
by sorry

#check tank_leak_consistency

end NUMINAMATH_CALUDE_tank_leak_consistency_l968_96819


namespace NUMINAMATH_CALUDE_expedition_duration_l968_96808

theorem expedition_duration (total_time : ℝ) (ratio : ℝ) (h1 : total_time = 10) (h2 : ratio = 3) :
  let first_expedition := total_time / (1 + ratio)
  first_expedition = 2.5 := by
sorry

end NUMINAMATH_CALUDE_expedition_duration_l968_96808


namespace NUMINAMATH_CALUDE_nancy_games_this_month_l968_96826

/-- Represents the number of football games Nancy attended or plans to attend -/
structure FootballGames where
  lastMonth : ℕ
  thisMonth : ℕ
  nextMonth : ℕ
  total : ℕ

/-- Theorem stating that Nancy attended 9 games this month -/
theorem nancy_games_this_month (g : FootballGames)
  (h1 : g.lastMonth = 8)
  (h2 : g.nextMonth = 7)
  (h3 : g.total = 24)
  (h4 : g.total = g.lastMonth + g.thisMonth + g.nextMonth) :
  g.thisMonth = 9 := by
  sorry


end NUMINAMATH_CALUDE_nancy_games_this_month_l968_96826


namespace NUMINAMATH_CALUDE_set_union_problem_l968_96889

theorem set_union_problem (a b : ℕ) :
  let A : Set ℕ := {5, a + 1}
  let B : Set ℕ := {a, b}
  A ∩ B = {2} → A ∪ B = {1, 2, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l968_96889


namespace NUMINAMATH_CALUDE_evaluate_expression_l968_96882

theorem evaluate_expression : (-3)^6 / 3^4 + 2^5 - 7^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l968_96882


namespace NUMINAMATH_CALUDE_missing_number_proof_l968_96846

theorem missing_number_proof (n : ℕ) (sum_with_missing : ℕ) : 
  (n = 63) → 
  (sum_with_missing = 2012) → 
  (n * (n + 1) / 2 - sum_with_missing = 4) :=
by sorry

end NUMINAMATH_CALUDE_missing_number_proof_l968_96846


namespace NUMINAMATH_CALUDE_peters_extra_pictures_l968_96853

theorem peters_extra_pictures (randy_pictures : ℕ) (peter_pictures : ℕ) (quincy_pictures : ℕ) :
  randy_pictures = 5 →
  quincy_pictures = peter_pictures + 20 →
  randy_pictures + peter_pictures + quincy_pictures = 41 →
  peter_pictures - randy_pictures = 3 := by
  sorry

end NUMINAMATH_CALUDE_peters_extra_pictures_l968_96853


namespace NUMINAMATH_CALUDE_min_sum_of_product_1020_l968_96848

theorem min_sum_of_product_1020 (a b c : ℕ+) (h : a * b * c = 1020) :
  ∃ (x y z : ℕ+), x * y * z = 1020 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 33 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1020_l968_96848


namespace NUMINAMATH_CALUDE_tonys_normal_temp_l968_96804

/-- Tony's normal body temperature -/
def normal_temp : ℝ := 95

/-- The fever threshold temperature -/
def fever_threshold : ℝ := 100

/-- Tony's current temperature -/
def current_temp : ℝ := normal_temp + 10

theorem tonys_normal_temp :
  (current_temp = fever_threshold + 5) →
  (fever_threshold = 100) →
  (normal_temp = 95) := by sorry

end NUMINAMATH_CALUDE_tonys_normal_temp_l968_96804


namespace NUMINAMATH_CALUDE_three_digit_square_ends_with_self_l968_96863

theorem three_digit_square_ends_with_self (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) ∧ (A^2 % 1000 = A) ↔ (A = 376 ∨ A = 625) := by
sorry

end NUMINAMATH_CALUDE_three_digit_square_ends_with_self_l968_96863


namespace NUMINAMATH_CALUDE_robinson_family_has_six_children_l968_96896

/-- Represents the Robinson family -/
structure RobinsonFamily where
  num_children : ℕ
  father_age : ℕ
  mother_age : ℕ
  children_ages : List ℕ

/-- The average age of a list of ages -/
def average_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

/-- The properties of the Robinson family -/
def is_robinson_family (family : RobinsonFamily) : Prop :=
  let total_ages := family.mother_age :: family.father_age :: family.children_ages
  average_age total_ages = 22 ∧
  family.father_age = 50 ∧
  average_age (family.mother_age :: family.children_ages) = 18

theorem robinson_family_has_six_children :
  ∀ family : RobinsonFamily, is_robinson_family family → family.num_children = 6 :=
by sorry

end NUMINAMATH_CALUDE_robinson_family_has_six_children_l968_96896


namespace NUMINAMATH_CALUDE_interest_difference_approx_l968_96898

def principal : ℝ := 147.69
def rate : ℝ := 0.15
def time1 : ℝ := 3.5
def time2 : ℝ := 10

def interest (p r t : ℝ) : ℝ := p * r * t

theorem interest_difference_approx :
  ∃ ε > 0, ε < 0.001 ∧ 
  |interest principal rate time2 - interest principal rate time1 - 143.998| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_difference_approx_l968_96898


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l968_96814

theorem reciprocal_of_negative_half : ((-1/2)⁻¹ : ℚ) = -2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l968_96814


namespace NUMINAMATH_CALUDE_sphere_packing_radius_l968_96895

/-- A configuration of spheres packed in a cube. -/
structure SpherePacking where
  cube_side : ℝ
  num_spheres : ℕ
  sphere_radius : ℝ

/-- The specific sphere packing configuration described in the problem. -/
def problem_packing : SpherePacking where
  cube_side := 2
  num_spheres := 16
  sphere_radius := 1  -- This is what we want to prove

/-- Predicate to check if a sphere packing configuration is valid according to the problem description. -/
def is_valid_packing (p : SpherePacking) : Prop :=
  p.cube_side = 2 ∧
  p.num_spheres = 16 ∧
  -- One sphere at the center, others tangent to it and three faces
  2 * p.sphere_radius = p.cube_side / 2

theorem sphere_packing_radius : 
  is_valid_packing problem_packing ∧ 
  problem_packing.sphere_radius = 1 :=
by sorry

end NUMINAMATH_CALUDE_sphere_packing_radius_l968_96895


namespace NUMINAMATH_CALUDE_box_height_l968_96891

/-- Proves that a rectangular box with given dimensions has a height of 3 cm -/
theorem box_height (base_length base_width volume : ℝ) 
  (h1 : base_length = 2)
  (h2 : base_width = 5)
  (h3 : volume = 30) :
  volume / (base_length * base_width) = 3 := by
  sorry

end NUMINAMATH_CALUDE_box_height_l968_96891


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017th_term_l968_96843

/-- An arithmetic sequence is monotonically increasing if its common difference is positive -/
def is_monotonically_increasing_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence if the ratio between consecutive terms is constant -/
def is_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

theorem arithmetic_sequence_2017th_term
  (a : ℕ → ℝ)
  (h_incr : is_monotonically_increasing_arithmetic a)
  (h_first : a 1 = 2)
  (h_geom : is_geometric_sequence (a 1 - 1) (a 3) (a 5 + 5)) :
  a 2017 = 1010 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017th_term_l968_96843


namespace NUMINAMATH_CALUDE_pythagorean_fraction_bound_l968_96820

theorem pythagorean_fraction_bound (m n t : ℝ) (h1 : m^2 + n^2 = t^2) (h2 : t ≠ 0) :
  -Real.sqrt 3 / 3 ≤ n / (m - 2 * t) ∧ n / (m - 2 * t) ≤ Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_fraction_bound_l968_96820


namespace NUMINAMATH_CALUDE_solve_money_problem_l968_96865

def money_problem (mildred_spent candice_spent amount_left : ℕ) : Prop :=
  let total_spent := mildred_spent + candice_spent
  let mom_gave := total_spent + amount_left
  mom_gave = mildred_spent + candice_spent + amount_left

theorem solve_money_problem :
  ∀ (mildred_spent candice_spent amount_left : ℕ),
  money_problem mildred_spent candice_spent amount_left :=
by
  sorry

end NUMINAMATH_CALUDE_solve_money_problem_l968_96865


namespace NUMINAMATH_CALUDE_min_p_plus_q_l968_96871

theorem min_p_plus_q (p q : ℕ) : 
  p > 1 → q > 1 → 17 * (p + 1) = 21 * (q + 1) → 
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 21 * (q' + 1) → 
  p + q ≤ p' + q' :=
by
  sorry

end NUMINAMATH_CALUDE_min_p_plus_q_l968_96871


namespace NUMINAMATH_CALUDE_tiles_required_to_cover_floor_l968_96854

-- Define the dimensions
def floor_length : ℚ := 10
def floor_width : ℚ := 15
def tile_length : ℚ := 5 / 12  -- 5 inches in feet
def tile_width : ℚ := 2 / 3    -- 8 inches in feet

-- Theorem statement
theorem tiles_required_to_cover_floor :
  (floor_length * floor_width) / (tile_length * tile_width) = 540 := by
  sorry

end NUMINAMATH_CALUDE_tiles_required_to_cover_floor_l968_96854


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_divided_by_11_l968_96831

theorem units_digit_of_sum_of_powers_divided_by_11 : 
  (3^2018 + 7^2018) % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_divided_by_11_l968_96831


namespace NUMINAMATH_CALUDE_clock_strike_theorem_l968_96856

/-- Calculates the time taken for a clock to strike a given number of times,
    given the time it takes to strike 3 times. -/
def strike_time (time_for_three : ℕ) (num_strikes : ℕ) : ℕ :=
  let interval_time := time_for_three / 2
  interval_time * (num_strikes - 1)

/-- Theorem stating that if a clock takes 6 seconds to strike 3 times,
    it will take 33 seconds to strike 12 times. -/
theorem clock_strike_theorem :
  strike_time 6 12 = 33 := by
  sorry

end NUMINAMATH_CALUDE_clock_strike_theorem_l968_96856


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l968_96824

def M : Set ℝ := {y | ∃ x, y = 3 - x^2}
def N : Set ℝ := {y | ∃ x, y = 2*x^2 - 1}

theorem intersection_of_M_and_N : M ∩ N = Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l968_96824


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l968_96818

theorem perfect_square_trinomial (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l968_96818


namespace NUMINAMATH_CALUDE_greening_project_optimization_l968_96879

/-- The optimization problem for greening project --/
theorem greening_project_optimization (total_area : ℝ) (team_a_rate : ℝ) (team_b_rate : ℝ)
  (team_a_wage : ℝ) (team_b_wage : ℝ) (h1 : total_area = 1200)
  (h2 : team_a_rate = 100) (h3 : team_b_rate = 50) (h4 : team_a_wage = 4000) (h5 : team_b_wage = 3000) :
  ∃ (days_a days_b : ℝ),
    days_a ≥ 3 ∧ days_b ≥ days_a ∧
    team_a_rate * days_a + team_b_rate * days_b = total_area ∧
    ∀ (x y : ℝ),
      x ≥ 3 → y ≥ x →
      team_a_rate * x + team_b_rate * y = total_area →
      team_a_wage * days_a + team_b_wage * days_b ≤ team_a_wage * x + team_b_wage * y ∧
      team_a_wage * days_a + team_b_wage * days_b = 56000 :=
by
  sorry

end NUMINAMATH_CALUDE_greening_project_optimization_l968_96879


namespace NUMINAMATH_CALUDE_least_possible_bananas_l968_96870

/-- Represents the distribution of bananas among three monkeys. -/
structure BananaDistribution where
  b₁ : ℕ  -- bananas taken by first monkey
  b₂ : ℕ  -- bananas taken by second monkey
  b₃ : ℕ  -- bananas taken by third monkey

/-- Checks if the given distribution satisfies all conditions of the problem. -/
def isValidDistribution (d : BananaDistribution) : Prop :=
  let m₁ := (2 * d.b₁) / 3 + d.b₂ / 3 + (7 * d.b₃) / 16
  let m₂ := d.b₁ / 6 + d.b₂ / 3 + (7 * d.b₃) / 16
  let m₃ := d.b₁ / 6 + d.b₂ / 3 + d.b₃ / 8
  (∀ n : ℕ, n ∈ [m₁, m₂, m₃] → n > 0) ∧  -- whole number condition
  5 * m₂ = 3 * m₁ ∧ 5 * m₃ = 2 * m₁       -- ratio condition

/-- The theorem stating the least possible total number of bananas. -/
theorem least_possible_bananas :
  ∃ (d : BananaDistribution),
    isValidDistribution d ∧
    d.b₁ + d.b₂ + d.b₃ = 336 ∧
    (∀ d' : BananaDistribution, isValidDistribution d' → d'.b₁ + d'.b₂ + d'.b₃ ≥ 336) :=
  sorry

end NUMINAMATH_CALUDE_least_possible_bananas_l968_96870


namespace NUMINAMATH_CALUDE_choose_three_from_ten_l968_96834

theorem choose_three_from_ten : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_ten_l968_96834


namespace NUMINAMATH_CALUDE_cake_sharing_percentage_l968_96829

theorem cake_sharing_percentage (total : ℝ) (rich_portion : ℝ) (ben_portion : ℝ) : 
  total > 0 →
  rich_portion > 0 →
  ben_portion > 0 →
  rich_portion + ben_portion = total →
  rich_portion / ben_portion = 3 →
  ben_portion / total = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_cake_sharing_percentage_l968_96829


namespace NUMINAMATH_CALUDE_wade_hot_dog_truck_l968_96811

theorem wade_hot_dog_truck (tips_per_customer : ℚ) (friday_customers : ℕ) (total_tips : ℚ) :
  tips_per_customer = 2 →
  friday_customers = 28 →
  total_tips = 296 →
  let saturday_customers := 3 * friday_customers
  let sunday_customers := (total_tips - tips_per_customer * (friday_customers + saturday_customers)) / tips_per_customer
  sunday_customers = 36 := by
sorry


end NUMINAMATH_CALUDE_wade_hot_dog_truck_l968_96811


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l968_96830

def A : Set ℤ := {1, 3, 5}
def B : Set ℤ := {-1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l968_96830


namespace NUMINAMATH_CALUDE_rectangle_perimeter_rectangle_perimeter_400_l968_96803

/-- A rectangle divided into four identical squares with a given area has a specific perimeter -/
theorem rectangle_perimeter (area : ℝ) (h_area : area > 0) : 
  ∃ (side : ℝ), 
    side > 0 ∧ 
    4 * side^2 = area ∧ 
    8 * side = 80 :=
by
  sorry

/-- The perimeter of a rectangle with area 400 square centimeters, 
    divided into four identical squares, is 80 centimeters -/
theorem rectangle_perimeter_400 : 
  ∃ (side : ℝ), 
    side > 0 ∧ 
    4 * side^2 = 400 ∧ 
    8 * side = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_rectangle_perimeter_400_l968_96803


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l968_96861

theorem arithmetic_expression_equality : 60 + 5 * 12 / (180 / 3) = 61 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l968_96861


namespace NUMINAMATH_CALUDE_birthday_paradox_l968_96805

theorem birthday_paradox (n : ℕ) (h : n = 367) :
  ∃ (f : Fin n → Fin 366), ¬Function.Injective f :=
sorry

end NUMINAMATH_CALUDE_birthday_paradox_l968_96805


namespace NUMINAMATH_CALUDE_meaningful_expression_l968_96833

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 2)) / (x - 1)) ↔ x ≥ -2 ∧ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l968_96833


namespace NUMINAMATH_CALUDE_calculate_expression_l968_96806

theorem calculate_expression : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 + 9000000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l968_96806


namespace NUMINAMATH_CALUDE_intersection_M_N_l968_96816

def M : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l968_96816


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l968_96844

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- Theorem: 15! ends with 5 zeroes when written in base 8 -/
theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes 15 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l968_96844


namespace NUMINAMATH_CALUDE_exists_divisible_term_l968_96852

/-- Sequence defined by a₀ = 5 and aₙ₊₁ = 2aₙ + 1 -/
def a : ℕ → ℕ
  | 0 => 5
  | n + 1 => 2 * a n + 1

/-- For every natural number n, there exists a different k such that a_n divides a_k -/
theorem exists_divisible_term (n : ℕ) : ∃ k : ℕ, k ≠ n ∧ a n ∣ a k := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_term_l968_96852


namespace NUMINAMATH_CALUDE_cube_edge_increase_l968_96851

theorem cube_edge_increase (surface_area_increase : Real) 
  (h : surface_area_increase = 69.00000000000001) : 
  ∃ edge_increase : Real, 
    edge_increase = 30 ∧ 
    (1 + edge_increase / 100)^2 = 1 + surface_area_increase / 100 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_increase_l968_96851


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l968_96873

/-- The price of an imported car after 5 years of annual reduction -/
def price_after_five_years (initial_price : ℝ) (annual_reduction_rate : ℝ) : ℝ :=
  initial_price * (1 - annual_reduction_rate)^5

/-- Theorem stating the relationship between the initial price, 
    annual reduction rate, and final price after 5 years -/
theorem price_reduction_theorem (x : ℝ) :
  price_after_five_years 300000 (x / 100) = 30000 * (1 - x / 100)^5 := by
  sorry

#check price_reduction_theorem

end NUMINAMATH_CALUDE_price_reduction_theorem_l968_96873


namespace NUMINAMATH_CALUDE_rose_count_l968_96869

theorem rose_count : ∃ (n : ℕ), 
  300 ≤ n ∧ n ≤ 400 ∧ 
  ∃ (x y : ℕ), n = 21 * x + 13 ∧ n = 15 * y - 8 ∧
  n = 307 := by
  sorry

end NUMINAMATH_CALUDE_rose_count_l968_96869


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l968_96874

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the point of interest
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_slope_angle :
  let slope := (deriv f) point.1
  Real.arctan slope = π/4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l968_96874


namespace NUMINAMATH_CALUDE_function_always_negative_l968_96837

theorem function_always_negative (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ m ∈ Set.Ioc (-4) 0 := by
sorry

end NUMINAMATH_CALUDE_function_always_negative_l968_96837


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_multiplier_l968_96893

def y : ℕ := 2^3^3^4^4^5^5^6^6^7^7^8^8^9

theorem smallest_perfect_cube_multiplier :
  (∃ k : ℕ, k > 0 ∧ ∃ n : ℕ, k * y = n^3) ∧
  (∀ k : ℕ, k > 0 → (∃ n : ℕ, k * y = n^3) → k ≥ 1500) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_multiplier_l968_96893


namespace NUMINAMATH_CALUDE_product_xyz_is_negative_one_l968_96838

theorem product_xyz_is_negative_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 1) 
  (h2 : y + 1/z = 1) : 
  x * y * z = -1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_negative_one_l968_96838


namespace NUMINAMATH_CALUDE_triangle_area_change_l968_96835

theorem triangle_area_change (b h : ℝ) (h_pos : 0 < h) (b_pos : 0 < b) :
  let new_height := 0.9 * h
  let new_base := 1.2 * b
  let original_area := (b * h) / 2
  let new_area := (new_base * new_height) / 2
  new_area = 1.08 * original_area := by
sorry

end NUMINAMATH_CALUDE_triangle_area_change_l968_96835


namespace NUMINAMATH_CALUDE_phone_production_ratio_l968_96800

/-- Proves that the ratio of this year's production to last year's production is 2:1 --/
theorem phone_production_ratio :
  ∀ (this_year last_year : ℕ),
  last_year = 5000 →
  (3 * this_year) / 4 = 7500 →
  (this_year : ℚ) / last_year = 2 := by
sorry

end NUMINAMATH_CALUDE_phone_production_ratio_l968_96800


namespace NUMINAMATH_CALUDE_vincent_book_purchase_l968_96813

/-- The number of books about outer space Vincent bought -/
def books_outer_space : ℕ := 1

/-- The number of books about animals Vincent bought -/
def books_animals : ℕ := 10

/-- The number of books about trains Vincent bought -/
def books_trains : ℕ := 3

/-- The cost of each book in dollars -/
def cost_per_book : ℕ := 16

/-- The total amount spent on books in dollars -/
def total_spent : ℕ := 224

theorem vincent_book_purchase :
  books_outer_space = 1 ∧
  books_animals = 10 ∧
  books_trains = 3 ∧
  cost_per_book = 16 ∧
  total_spent = 224 →
  books_outer_space = 1 :=
by sorry

end NUMINAMATH_CALUDE_vincent_book_purchase_l968_96813


namespace NUMINAMATH_CALUDE_bicycle_speeds_l968_96888

/-- Represents a bicycle with front and rear gears -/
structure Bicycle where
  front_gears : Nat
  rear_gears : Nat

/-- Calculates the number of unique speeds for a bicycle -/
def unique_speeds (b : Bicycle) : Nat :=
  b.front_gears * b.rear_gears - b.rear_gears

/-- Theorem stating that a bicycle with 3 front gears and 4 rear gears has 8 unique speeds -/
theorem bicycle_speeds :
  ∃ (b : Bicycle), b.front_gears = 3 ∧ b.rear_gears = 4 ∧ unique_speeds b = 8 :=
by
  sorry

#eval unique_speeds ⟨3, 4⟩

end NUMINAMATH_CALUDE_bicycle_speeds_l968_96888


namespace NUMINAMATH_CALUDE_binary_of_25_l968_96885

/-- Represents a binary number as a list of bits (least significant bit first) -/
def BinaryRepr := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryRepr :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Theorem: The binary representation of 25 is 11001 -/
theorem binary_of_25 :
  toBinary 25 = [true, false, false, true, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_of_25_l968_96885


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l968_96841

theorem sum_of_coefficients (a b c d : ℤ) :
  (∀ x, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + x^3 - 2*x^2 + 17*x + 15) →
  a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l968_96841
