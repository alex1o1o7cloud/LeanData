import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2597_259719

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | x^2 - (a + 2)*x + 2*a > 0}
  (a > 2 → solution_set = {x : ℝ | x < 2 ∨ x > a}) ∧
  (a = 2 → solution_set = {x : ℝ | x ≠ 2}) ∧
  (a < 2 → solution_set = {x : ℝ | x < a ∨ x > 2}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2597_259719


namespace NUMINAMATH_CALUDE_f_max_min_in_interval_l2597_259721

-- Define the function f
def f : ℝ → ℝ := sorry

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- f satisfies f(1 + x) = f(1 - x) for all x
axiom f_symmetry : ∀ x, f (1 + x) = f (1 - x)

-- f is monotonically increasing in [-1, 1]
axiom f_monotone : ∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y

-- Theorem statement
theorem f_max_min_in_interval :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ f 1) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f 3 ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_in_interval_l2597_259721


namespace NUMINAMATH_CALUDE_domain_exclusion_sum_l2597_259711

theorem domain_exclusion_sum (C D : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 8 * x + 6 = 0 ↔ (x = C ∨ x = D)) →
  C + D = 4 := by
  sorry

end NUMINAMATH_CALUDE_domain_exclusion_sum_l2597_259711


namespace NUMINAMATH_CALUDE_maximize_x5y2_l2597_259755

theorem maximize_x5y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 35) :
  x^5 * y^2 ≤ 25^5 * 10^2 ∧ 
  (x^5 * y^2 = 25^5 * 10^2 ↔ x = 25 ∧ y = 10) :=
by sorry

end NUMINAMATH_CALUDE_maximize_x5y2_l2597_259755


namespace NUMINAMATH_CALUDE_half_of_number_is_315_l2597_259708

theorem half_of_number_is_315 (x : ℝ) : 
  (4/15 : ℝ) * (5/7 : ℝ) * x - (4/9 : ℝ) * (2/5 : ℝ) * x = 8 → x/2 = 315 := by
sorry

end NUMINAMATH_CALUDE_half_of_number_is_315_l2597_259708


namespace NUMINAMATH_CALUDE_stating_speed_ratio_equals_one_plus_head_start_l2597_259770

/-- The ratio of runner A's speed to runner B's speed in a race where A gives B a head start -/
def speed_ratio : ℝ := 1.11764705882352941

/-- The fraction of the race length that runner A gives as a head start to runner B -/
def head_start : ℝ := 0.11764705882352941

/-- 
Theorem stating that the speed ratio of runner A to runner B is equal to 1 plus the head start fraction,
given that the race ends in a dead heat when A gives B the specified head start.
-/
theorem speed_ratio_equals_one_plus_head_start : 
  speed_ratio = 1 + head_start := by sorry

end NUMINAMATH_CALUDE_stating_speed_ratio_equals_one_plus_head_start_l2597_259770


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2597_259732

theorem inequality_system_solution (x : ℝ) :
  (5 * x + 1 ≥ 3 * (x - 1)) →
  (1 - (x + 3) / 3 ≤ x) →
  x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2597_259732


namespace NUMINAMATH_CALUDE_cable_length_equals_scientific_notation_l2597_259742

/-- The total length of fiber optic cable routes in kilometers -/
def cable_length : ℝ := 59580000

/-- The scientific notation representation of the cable length -/
def cable_length_scientific : ℝ := 5.958 * (10 ^ 7)

/-- Theorem stating that the cable length is equal to its scientific notation representation -/
theorem cable_length_equals_scientific_notation : cable_length = cable_length_scientific := by
  sorry

end NUMINAMATH_CALUDE_cable_length_equals_scientific_notation_l2597_259742


namespace NUMINAMATH_CALUDE_gcf_lcm_360_270_l2597_259749

theorem gcf_lcm_360_270 :
  (Nat.gcd 360 270 = 90) ∧ (Nat.lcm 360 270 = 1080) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_360_270_l2597_259749


namespace NUMINAMATH_CALUDE_eight_books_distribution_l2597_259752

/-- The number of ways to distribute indistinguishable books between two locations --/
def distribute_books (total : ℕ) : ℕ := 
  if total ≥ 2 then total - 1 else 0

/-- Theorem: Distributing 8 indistinguishable books between two locations, 
    with at least one book in each location, results in 7 different ways --/
theorem eight_books_distribution : distribute_books 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eight_books_distribution_l2597_259752


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2597_259712

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, -1/2 < x ∧ x < 1/3 ↔ a*x^2 + 2*x + 20 > 0) → a = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2597_259712


namespace NUMINAMATH_CALUDE_regular_triangle_on_hyperbola_coordinates_l2597_259766

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define a point on the hyperbola
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

-- Define the branches of the hyperbola
def on_positive_branch (p : PointOnHyperbola) : Prop := p.x > 0
def on_negative_branch (p : PointOnHyperbola) : Prop := p.x < 0

-- Define a regular triangle on the hyperbola
structure RegularTriangleOnHyperbola where
  P : PointOnHyperbola
  Q : PointOnHyperbola
  R : PointOnHyperbola
  is_regular : True  -- We assume this property without proving it

-- Theorem statement
theorem regular_triangle_on_hyperbola_coordinates 
  (t : RegularTriangleOnHyperbola)
  (h_P : t.P.x = -1 ∧ t.P.y = 1)
  (h_P_branch : on_negative_branch t.P)
  (h_Q_branch : on_positive_branch t.Q)
  (h_R_branch : on_positive_branch t.R) :
  ((t.Q.x = 2 - Real.sqrt 3 ∧ t.Q.y = 2 + Real.sqrt 3) ∧
   (t.R.x = 2 + Real.sqrt 3 ∧ t.R.y = 2 - Real.sqrt 3)) ∨
  ((t.Q.x = 2 + Real.sqrt 3 ∧ t.Q.y = 2 - Real.sqrt 3) ∧
   (t.R.x = 2 - Real.sqrt 3 ∧ t.R.y = 2 + Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_regular_triangle_on_hyperbola_coordinates_l2597_259766


namespace NUMINAMATH_CALUDE_cube_root_problem_l2597_259764

theorem cube_root_problem :
  ∃ (a b : ℤ) (c : ℚ),
    (5 * a - 2 : ℚ) = -27 ∧
    b = Int.floor (Real.sqrt 22) ∧
    c = -(4 : ℚ)/25 ∧
    a = -5 ∧
    b = 4 ∧
    c = -(2 : ℚ)/5 ∧
    Real.sqrt ((4 : ℚ) * a * c + 7 * b) = 6 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_problem_l2597_259764


namespace NUMINAMATH_CALUDE_ascending_order_abc_l2597_259743

theorem ascending_order_abc :
  let a := Real.sin (17 * π / 180) * Real.cos (45 * π / 180) + Real.cos (17 * π / 180) * Real.sin (45 * π / 180)
  let b := 2 * (Real.cos (13 * π / 180))^2 - 1
  let c := Real.sqrt 3 / 2
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l2597_259743


namespace NUMINAMATH_CALUDE_defective_units_shipped_percentage_l2597_259760

theorem defective_units_shipped_percentage
  (total_units : ℝ)
  (defective_percentage : ℝ)
  (defective_shipped_percentage : ℝ)
  (h1 : defective_percentage = 5)
  (h2 : defective_shipped_percentage = 0.2)
  : (defective_shipped_percentage * total_units) / (defective_percentage * total_units) * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_percentage_l2597_259760


namespace NUMINAMATH_CALUDE_line_equation_correct_l2597_259710

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The equation 3x - y + 4 = 0 represents a line with slope 3 and y-intercept 4 -/
theorem line_equation_correct (l : Line) (eq : LineEquation) :
  l.slope = 3 ∧ l.intercept = 4 ∧ 
  eq.a = 3 ∧ eq.b = -1 ∧ eq.c = 4 →
  eq.a * x + eq.b * y + eq.c = 0 ↔ y = l.slope * x + l.intercept :=
by sorry

end NUMINAMATH_CALUDE_line_equation_correct_l2597_259710


namespace NUMINAMATH_CALUDE_peggy_stamps_to_add_l2597_259773

/-- Given the number of stamps each person has, calculates how many stamps Peggy needs to add to have as many as Bert. -/
def stamps_to_add (peggy_stamps : ℕ) (ernie_multiplier : ℕ) (bert_multiplier : ℕ) : ℕ :=
  bert_multiplier * (ernie_multiplier * peggy_stamps) - peggy_stamps

/-- Proves that Peggy needs to add 825 stamps to have as many as Bert. -/
theorem peggy_stamps_to_add : 
  stamps_to_add 75 3 4 = 825 := by sorry

end NUMINAMATH_CALUDE_peggy_stamps_to_add_l2597_259773


namespace NUMINAMATH_CALUDE_new_average_age_with_teacher_l2597_259735

theorem new_average_age_with_teacher (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℕ) :
  num_students = 40 →
  student_avg_age = 15 →
  teacher_age = 56 →
  (num_students : ℝ) * student_avg_age + teacher_age = 16 * (num_students + 1) := by
  sorry


end NUMINAMATH_CALUDE_new_average_age_with_teacher_l2597_259735


namespace NUMINAMATH_CALUDE_variance_of_scores_l2597_259783

def scores : List ℝ := [9, 10, 9, 7, 10]

theorem variance_of_scores : 
  let n : ℕ := scores.length
  let mean : ℝ := (scores.sum) / n
  let variance : ℝ := (scores.map (λ x => (x - mean)^2)).sum / n
  variance = 6/5 := by sorry

end NUMINAMATH_CALUDE_variance_of_scores_l2597_259783


namespace NUMINAMATH_CALUDE_f_extrema_l2597_259771

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

theorem f_extrema :
  let a : ℝ := -3
  let b : ℝ := 3
  (∀ x ∈ Set.Icc a b, f x ≤ 48) ∧
  (∃ x ∈ Set.Icc a b, f x = 48) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ -4) ∧
  (∃ x ∈ Set.Icc a b, f x = -4) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l2597_259771


namespace NUMINAMATH_CALUDE_min_value_zero_iff_k_eq_one_l2597_259727

/-- The quadratic expression in x and y with parameter k -/
def f (k x y : ℝ) : ℝ := 3*x^2 - 4*k*x*y + (2*k^2 + 1)*y^2 - 6*x - 2*y + 4

/-- The theorem stating that the minimum value of f is 0 iff k = 1 -/
theorem min_value_zero_iff_k_eq_one :
  (∃ (m : ℝ), m = 0 ∧ ∀ x y : ℝ, f 1 x y ≥ m) ∧
  (∀ k : ℝ, k ≠ 1 → ¬∃ (m : ℝ), m = 0 ∧ ∀ x y : ℝ, f k x y ≥ m) :=
sorry

end NUMINAMATH_CALUDE_min_value_zero_iff_k_eq_one_l2597_259727


namespace NUMINAMATH_CALUDE_domain_sum_l2597_259794

theorem domain_sum (y : ℝ → ℝ) (A B : ℝ) : 
  (∀ x, y x = 5 * x / (3 * x^2 - 9 * x + 6)) →
  (3 * A^2 - 9 * A + 6 = 0) →
  (3 * B^2 - 9 * B + 6 = 0) →
  A + B = 3 := by
sorry

end NUMINAMATH_CALUDE_domain_sum_l2597_259794


namespace NUMINAMATH_CALUDE_find_number_l2597_259756

theorem find_number (x : ℝ) : (0.62 * 150 - 0.20 * x = 43) → x = 250 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2597_259756


namespace NUMINAMATH_CALUDE_z_value_l2597_259765

theorem z_value (x y z : ℝ) (h : (x + 1)⁻¹ + (y + 1)⁻¹ = z⁻¹) : 
  z = ((x + 1) * (y + 1)) / (x + y + 2) := by
  sorry

end NUMINAMATH_CALUDE_z_value_l2597_259765


namespace NUMINAMATH_CALUDE_number_puzzle_l2597_259799

theorem number_puzzle : ∃ x : ℚ, (x / 6) * 12 = 8 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2597_259799


namespace NUMINAMATH_CALUDE_market_equilibrium_and_max_revenue_l2597_259726

-- Define the demand function
def demand_function (P : ℝ) : ℝ := 688 - 4 * P

-- Define the supply function (to be proven)
def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Define the tax revenue function
def tax_revenue (t : ℝ) (Q : ℝ) : ℝ := t * Q

-- Theorem statement
theorem market_equilibrium_and_max_revenue :
  -- Conditions
  let change_ratio : ℝ := 1.5
  let production_tax : ℝ := 90
  let producer_price : ℝ := 64

  -- Prove that the supply function is correct
  ∀ P, supply_function P = 6 * P - 312 ∧
  
  -- Prove that the maximum tax revenue is 8640
  ∃ t_optimal, 
    let Q_optimal := demand_function (producer_price + t_optimal)
    tax_revenue t_optimal Q_optimal = 8640 ∧
    ∀ t, tax_revenue t (demand_function (producer_price + t)) ≤ 8640 :=
by sorry

end NUMINAMATH_CALUDE_market_equilibrium_and_max_revenue_l2597_259726


namespace NUMINAMATH_CALUDE_oil_depth_in_specific_tank_l2597_259722

/-- Represents a horizontal cylindrical tank --/
structure HorizontalCylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the depth of oil in a horizontal cylindrical tank --/
def oilDepth (tank : HorizontalCylindricalTank) (surfaceArea : ℝ) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The depth of oil in the specified tank with given surface area --/
theorem oil_depth_in_specific_tank :
  let tank : HorizontalCylindricalTank := ⟨12, 8⟩
  let surfaceArea : ℝ := 48
  oilDepth tank surfaceArea = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_oil_depth_in_specific_tank_l2597_259722


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l2597_259774

-- Define the parabola
def parabola (x : ℝ) : ℝ := (x - 5)^2 + 4

-- State the theorem
theorem axis_of_symmetry :
  ∀ x : ℝ, parabola (5 + x) = parabola (5 - x) := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l2597_259774


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2597_259754

theorem absolute_value_equality (x y : ℝ) : 
  |x - Real.sqrt y| = x + Real.sqrt y → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2597_259754


namespace NUMINAMATH_CALUDE_arrangement_count_proof_l2597_259798

/-- The number of ways to arrange 4 men and 4 women into two indistinguishable groups
    of two (each containing one man and one woman) and one group of four
    (containing the remaining two men and two women) -/
def arrangement_count : ℕ := 72

/-- The number of ways to choose one man from 4 men -/
def choose_man : ℕ := 4

/-- The number of ways to choose one woman from 4 women -/
def choose_woman : ℕ := 4

/-- The number of ways to choose one man from 3 remaining men -/
def choose_remaining_man : ℕ := 3

/-- The number of ways to choose one woman from 3 remaining women -/
def choose_remaining_woman : ℕ := 3

/-- The number of ways to arrange two indistinguishable groups -/
def indistinguishable_groups : ℕ := 2

theorem arrangement_count_proof :
  arrangement_count = (choose_man * choose_woman * choose_remaining_man * choose_remaining_woman) / indistinguishable_groups :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_proof_l2597_259798


namespace NUMINAMATH_CALUDE_smallest_share_amount_l2597_259720

def total_amount : ℝ := 500
def give_away_percentage : ℝ := 0.60
def friend_count : ℕ := 5
def shares : List ℝ := [0.30, 0.25, 0.20, 0.15, 0.10]

theorem smallest_share_amount :
  let amount_to_distribute := total_amount * give_away_percentage
  let smallest_share := shares.minimum?
  smallest_share.map (λ s => s * amount_to_distribute) = some 30 := by sorry

end NUMINAMATH_CALUDE_smallest_share_amount_l2597_259720


namespace NUMINAMATH_CALUDE_class_composition_l2597_259728

theorem class_composition (total : ℕ) (swimmers : ℕ) : 
  (total / 4 : ℚ) = (total - swimmers) ∧  -- A quarter of students are non-swimmers
  (total / 8 : ℚ) = ((total - swimmers) / 2 : ℚ) ∧  -- Half of non-swimmers signed up
  (total - swimmers) / 2 = 4 →  -- Four non-swimmers did not sign up
  total = 32 ∧ swimmers = 24 := by
sorry

end NUMINAMATH_CALUDE_class_composition_l2597_259728


namespace NUMINAMATH_CALUDE_wallace_existing_bags_l2597_259779

/- Define the problem parameters -/
def batch_size : ℕ := 10
def order_size : ℕ := 60
def days_to_fulfill : ℕ := 4

/- Define the function to calculate the number of bags Wallace can make in given days -/
def bags_made_in_days (days : ℕ) : ℕ := days * batch_size

/- Theorem: Wallace has already made 20 bags of jerky -/
theorem wallace_existing_bags : 
  order_size - bags_made_in_days days_to_fulfill = 20 := by
  sorry

#eval order_size - bags_made_in_days days_to_fulfill

end NUMINAMATH_CALUDE_wallace_existing_bags_l2597_259779


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2597_259725

theorem algebraic_expression_value (m n : ℝ) (h : m ≠ n) 
  (h_equal : m^2 - 2*m + 3 = n^2 - 2*n + 3) : 
  let x := m + n
  (x^2 - 2*x + 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2597_259725


namespace NUMINAMATH_CALUDE_largest_y_satisfies_equation_forms_triangle_largest_y_forms_triangle_l2597_259782

def largest_y : ℝ := 23

theorem largest_y_satisfies_equation :
  |largest_y - 8| = 15 ∧
  ∀ y : ℝ, |y - 8| = 15 → y ≤ largest_y :=
sorry

theorem forms_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem largest_y_forms_triangle :
  forms_triangle largest_y 20 9 :=
sorry

end NUMINAMATH_CALUDE_largest_y_satisfies_equation_forms_triangle_largest_y_forms_triangle_l2597_259782


namespace NUMINAMATH_CALUDE_sally_balloons_l2597_259737

def initial_orange_balloons : ℕ := sorry

def lost_balloons : ℕ := 2

def current_orange_balloons : ℕ := 7

theorem sally_balloons : initial_orange_balloons = current_orange_balloons + lost_balloons :=
by sorry

end NUMINAMATH_CALUDE_sally_balloons_l2597_259737


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2597_259796

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2597_259796


namespace NUMINAMATH_CALUDE_dogs_food_consumption_l2597_259780

/-- The amount of dog food eaten by one dog per day -/
def dog_food_per_day : ℝ := 0.12

/-- The number of dogs -/
def num_dogs : ℕ := 2

/-- The total amount of dog food eaten by all dogs per day -/
def total_dog_food : ℝ := dog_food_per_day * num_dogs

theorem dogs_food_consumption :
  total_dog_food = 0.24 := by sorry

end NUMINAMATH_CALUDE_dogs_food_consumption_l2597_259780


namespace NUMINAMATH_CALUDE_four_dice_same_number_probability_l2597_259787

/-- The number of sides on a standard die -/
def standardDieSides : ℕ := 6

/-- The number of dice being tossed -/
def numberOfDice : ℕ := 4

/-- The probability of all dice showing the same number -/
def probabilitySameNumber : ℚ := 1 / (standardDieSides ^ (numberOfDice - 1))

/-- Theorem: The probability of four standard six-sided dice showing the same number when tossed simultaneously is 1/216 -/
theorem four_dice_same_number_probability : 
  probabilitySameNumber = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_four_dice_same_number_probability_l2597_259787


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2597_259768

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 2, 3}

-- Define set B
def B : Finset Nat := {2, 4}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2597_259768


namespace NUMINAMATH_CALUDE_binomial_remainder_l2597_259734

theorem binomial_remainder (x : ℕ) : x = 2000 → (1 - x)^1999 % 2000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_remainder_l2597_259734


namespace NUMINAMATH_CALUDE_gwen_recycling_points_l2597_259738

/-- Calculates the points earned by recycling bags of cans. -/
def points_earned (total_bags : ℕ) (unrecycled_bags : ℕ) (points_per_bag : ℕ) : ℕ :=
  (total_bags - unrecycled_bags) * points_per_bag

/-- Proves that Gwen earns 16 points given the problem conditions. -/
theorem gwen_recycling_points :
  points_earned 4 2 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_gwen_recycling_points_l2597_259738


namespace NUMINAMATH_CALUDE_anthony_painting_time_l2597_259716

/-- The time it takes Kathleen and Anthony to paint two rooms together -/
def joint_time : ℝ := 3.428571428571429

/-- The time it takes Kathleen to paint one room -/
def kathleen_time : ℝ := 3

/-- Anthony's painting time for one room -/
def anthony_time : ℝ := 4

/-- Theorem stating that given Kathleen's painting time and their joint time for two rooms, 
    Anthony's individual painting time for one room is 4 hours -/
theorem anthony_painting_time : 
  (1 / kathleen_time + 1 / anthony_time) * joint_time = 2 :=
sorry

end NUMINAMATH_CALUDE_anthony_painting_time_l2597_259716


namespace NUMINAMATH_CALUDE_min_overlap_percentage_l2597_259707

theorem min_overlap_percentage (computer_users smartphone_users : ℝ) :
  computer_users ≥ 0 ∧ computer_users ≤ 100 ∧
  smartphone_users ≥ 0 ∧ smartphone_users ≤ 100 →
  let min_overlap := max 0 (computer_users + smartphone_users - 100)
  ∀ overlap, 
    overlap ≥ 0 ∧ 
    overlap ≤ min computer_users smartphone_users ∧
    overlap ≤ computer_users ∧
    overlap ≤ smartphone_users ∧
    computer_users + smartphone_users - overlap ≤ 100 →
    overlap ≥ min_overlap := by
sorry

end NUMINAMATH_CALUDE_min_overlap_percentage_l2597_259707


namespace NUMINAMATH_CALUDE_midpoint_on_number_line_l2597_259714

theorem midpoint_on_number_line (A B C : ℝ) : 
  A = -7 → 
  |B - A| = 5 → 
  C = (A + B) / 2 → 
  C = -9/2 ∨ C = -19/2 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_on_number_line_l2597_259714


namespace NUMINAMATH_CALUDE_rogers_initial_money_l2597_259788

theorem rogers_initial_money (initial_money : ℕ) : 
  (initial_money - 47 = 3 * 7) → initial_money = 68 := by
  sorry

end NUMINAMATH_CALUDE_rogers_initial_money_l2597_259788


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l2597_259745

/-- 
Given an arithmetic progression where a₁₂, a₁₃, a₁₅ are the 12th, 13th, and 15th terms respectively,
and their squares form a geometric progression with common ratio q,
prove that q must be one of: 4, 4 - 2√3, 4 + 2√3, or 9/25.
-/
theorem arithmetic_geometric_progression (a₁₂ a₁₃ a₁₅ d q : ℝ) : 
  (a₁₃ = a₁₂ + d ∧ a₁₅ = a₁₃ + 2*d) →  -- arithmetic progression condition
  (a₁₃^2)^2 = a₁₂^2 * a₁₅^2 →  -- geometric progression condition
  (q = (a₁₃^2 / a₁₂^2)) →  -- definition of q
  (q = 4 ∨ q = 4 - 2*Real.sqrt 3 ∨ q = 4 + 2*Real.sqrt 3 ∨ q = 9/25) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l2597_259745


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_inequality_l2597_259789

/-- Given a tetrahedron ABCD with inscribed sphere radius r and exinscribed sphere radii r_A, r_B, r_C, r_D,
    the sum of the reciprocals of the square roots of the sums of squares minus products of adjacent radii
    is less than or equal to 2/r. -/
theorem tetrahedron_sphere_inequality (r r_A r_B r_C r_D : ℝ) 
  (hr : r > 0) (hr_A : r_A > 0) (hr_B : r_B > 0) (hr_C : r_C > 0) (hr_D : r_D > 0) :
  1 / Real.sqrt (r_A^2 - r_A*r_B + r_B^2) + 
  1 / Real.sqrt (r_B^2 - r_B*r_C + r_C^2) + 
  1 / Real.sqrt (r_C^2 - r_C*r_D + r_D^2) + 
  1 / Real.sqrt (r_D^2 - r_D*r_A + r_A^2) ≤ 2 / r :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_inequality_l2597_259789


namespace NUMINAMATH_CALUDE_integer_solution_problem_l2597_259758

theorem integer_solution_problem :
  ∀ (a b c d : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
    a > b ∧ b > c ∧ c > d →
    a * b + c * d = 34 →
    a * c - b * d = 19 →
    ((a = 1 ∧ b = 4 ∧ c = -5 ∧ d = -6) ∨
     (a = -1 ∧ b = -4 ∧ c = 5 ∧ d = 6)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_problem_l2597_259758


namespace NUMINAMATH_CALUDE_bella_max_number_l2597_259729

theorem bella_max_number : 
  ∃ (m : ℕ), m = 720 ∧ 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → 3 * (250 - n) ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_bella_max_number_l2597_259729


namespace NUMINAMATH_CALUDE_order_of_numbers_l2597_259793

/-- Converts a number from base b to base 10 --/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

theorem order_of_numbers :
  let a := toBase10 0x12 16
  let b := toBase10 25 7
  let c := toBase10 33 4
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l2597_259793


namespace NUMINAMATH_CALUDE_book_distribution_l2597_259763

theorem book_distribution (x : ℕ) : 
  (3 * x + 20 = 4 * x - 25) ↔ 
  (∃ (total_books : ℕ), 
    (total_books = 3 * x + 20) ∧ 
    (total_books = 4 * x - 25)) :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_l2597_259763


namespace NUMINAMATH_CALUDE_cash_count_correction_l2597_259736

/-- Represents the correction needed for a cash count error -/
def correction_needed (q d n c x : ℕ) : ℤ :=
  let initial_count := 25 * q + 10 * d + 5 * n + c
  let corrected_count := 25 * (q - x) + 10 * (d - x) + 5 * (n + x) + (c + x)
  corrected_count - initial_count

/-- 
Theorem: Given a cash count with q quarters, d dimes, n nickels, c cents,
and x nickels mistakenly counted as quarters and x dimes as cents,
the correction needed is to add 11x cents.
-/
theorem cash_count_correction (q d n c x : ℕ) :
  correction_needed q d n c x = 11 * x := by
  sorry

end NUMINAMATH_CALUDE_cash_count_correction_l2597_259736


namespace NUMINAMATH_CALUDE_bakers_cakes_l2597_259748

/-- The number of cakes Baker made is equal to the sum of cakes sold and cakes left. -/
theorem bakers_cakes (total sold left : ℕ) (h1 : sold = 145) (h2 : left = 72) (h3 : total = sold + left) :
  total = 217 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l2597_259748


namespace NUMINAMATH_CALUDE_sum_of_seventh_eighth_ninth_l2597_259750

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  sum_first_three : a 1 + a 2 + a 3 = 30
  sum_next_three : a 4 + a 5 + a 6 = 120

/-- The sum of the 7th, 8th, and 9th terms equals 480 -/
theorem sum_of_seventh_eighth_ninth (seq : GeometricSequence) : 
  seq.a 7 + seq.a 8 + seq.a 9 = 480 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_eighth_ninth_l2597_259750


namespace NUMINAMATH_CALUDE_sum_of_four_digit_even_and_multiples_of_three_l2597_259704

/-- The number of four-digit even numbers -/
def C : ℕ := 4500

/-- The number of four-digit multiples of 3 -/
def D : ℕ := 3000

/-- Theorem stating that the sum of four-digit even numbers and four-digit multiples of 3 is 7500 -/
theorem sum_of_four_digit_even_and_multiples_of_three :
  C + D = 7500 := by sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_even_and_multiples_of_three_l2597_259704


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2597_259759

theorem algebraic_expression_value (a b : ℝ) 
  (sum_eq : a + b = 5) 
  (product_eq : a * b = 2) : 
  a^2 - a*b + b^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2597_259759


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2597_259718

theorem unique_positive_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + y^2 + z^3 = 3)
  (eq2 : y + z^2 + x^3 = 3)
  (eq3 : z + x^2 + y^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2597_259718


namespace NUMINAMATH_CALUDE_stratified_sample_proportion_l2597_259705

/-- Calculates the number of teachers under 40 in a stratified sample -/
def teachersUnder40InSample (totalTeachers : ℕ) (under40Teachers : ℕ) (sampleSize : ℕ) : ℕ :=
  (under40Teachers * sampleSize) / totalTeachers

theorem stratified_sample_proportion 
  (totalTeachers : ℕ) 
  (under40Teachers : ℕ) 
  (over40Teachers : ℕ) 
  (sampleSize : ℕ) :
  totalTeachers = 490 →
  under40Teachers = 350 →
  over40Teachers = 140 →
  sampleSize = 70 →
  totalTeachers = under40Teachers + over40Teachers →
  teachersUnder40InSample totalTeachers under40Teachers sampleSize = 50 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_proportion_l2597_259705


namespace NUMINAMATH_CALUDE_hexagon_wire_problem_l2597_259769

/-- Calculates the remaining wire length after creating a regular hexagon. -/
def remaining_wire_length (total_wire : ℝ) (hexagon_side : ℝ) : ℝ :=
  total_wire - 6 * hexagon_side

/-- Proves that given a wire of 50 cm and a regular hexagon with side length 8 cm, 
    the remaining wire length is 2 cm. -/
theorem hexagon_wire_problem :
  remaining_wire_length 50 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_wire_problem_l2597_259769


namespace NUMINAMATH_CALUDE_gcf_75_90_l2597_259775

theorem gcf_75_90 : Nat.gcd 75 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_75_90_l2597_259775


namespace NUMINAMATH_CALUDE_kids_difference_l2597_259795

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 11) 
  (h2 : tuesday = 12) : 
  tuesday - monday = 1 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l2597_259795


namespace NUMINAMATH_CALUDE_appetizer_cost_l2597_259706

/-- Proves that the cost of the appetizer is $10 given the conditions of the restaurant bill --/
theorem appetizer_cost (entree_cost : ℝ) (entree_count : ℕ) (tip_rate : ℝ) (total_spent : ℝ) :
  entree_cost = 20 →
  entree_count = 4 →
  tip_rate = 0.2 →
  total_spent = 108 →
  ∃ (appetizer_cost : ℝ),
    appetizer_cost + entree_cost * entree_count + tip_rate * (appetizer_cost + entree_cost * entree_count) = total_spent ∧
    appetizer_cost = 10 := by
  sorry


end NUMINAMATH_CALUDE_appetizer_cost_l2597_259706


namespace NUMINAMATH_CALUDE_smallest_number_l2597_259747

theorem smallest_number (S : Set ℤ) (h : S = {0, -1, 1, 2}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2597_259747


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l2597_259724

/-- Given a circle with radius 6 cm tangent to three sides of a rectangle,
    prove that the longer side of the rectangle is 9π cm when the rectangle's
    area is three times the circle's area. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_width rectangle_length : ℝ) :
  circle_radius = 6 →
  rectangle_width = 2 * circle_radius →
  rectangle_length * rectangle_width = 3 * Real.pi * circle_radius^2 →
  rectangle_length = 9 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l2597_259724


namespace NUMINAMATH_CALUDE_womens_haircut_cost_l2597_259772

theorem womens_haircut_cost :
  let childrens_haircut_cost : ℝ := 36
  let num_children : ℕ := 2
  let tip_percentage : ℝ := 0.20
  let tip_amount : ℝ := 24
  let womens_haircut_cost : ℝ := 48
  tip_amount = tip_percentage * (womens_haircut_cost + num_children * childrens_haircut_cost) :=
by
  sorry

end NUMINAMATH_CALUDE_womens_haircut_cost_l2597_259772


namespace NUMINAMATH_CALUDE_class_selection_probabilities_l2597_259786

/-- Represents the total number of classes -/
def total_classes : ℕ := 10

/-- Represents the number of classes to be selected -/
def selected_classes : ℕ := 3

/-- Represents the class number we're interested in -/
def target_class : ℕ := 4

/-- Probability of the target class being drawn first -/
def prob_first : ℝ := sorry

/-- Probability of the target class being drawn second -/
def prob_second : ℝ := sorry

/-- Theorem stating the probabilities of the target class being drawn first and second -/
theorem class_selection_probabilities :
  prob_first = sorry ∧ prob_second = sorry :=
sorry

end NUMINAMATH_CALUDE_class_selection_probabilities_l2597_259786


namespace NUMINAMATH_CALUDE_line_through_points_sum_m_b_l2597_259785

/-- Given a line passing through points (2,8) and (5,2) with equation y = mx + b, prove that m + b = 10 -/
theorem line_through_points_sum_m_b (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → (x = 2 ∧ y = 8) ∨ (x = 5 ∧ y = 2)) →
  m + b = 10 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_sum_m_b_l2597_259785


namespace NUMINAMATH_CALUDE_production_increase_l2597_259715

def planned_daily_production : ℕ := 500

def daily_changes : List ℤ := [40, -30, 90, -50, -20, -10, 20]

def actual_daily_production : List ℕ := 
  List.scanl (λ acc change => (acc : ℤ) + change |>.toNat) planned_daily_production daily_changes

def total_actual_production : ℕ := actual_daily_production.sum

def total_planned_production : ℕ := planned_daily_production * 7

theorem production_increase :
  total_actual_production = 3790 ∧ total_actual_production > total_planned_production :=
by sorry

end NUMINAMATH_CALUDE_production_increase_l2597_259715


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2597_259753

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (45 - b) + c / (54 - c) = 8) : 
  4 / (36 - a) + 5 / (45 - b) + 6 / (54 - c) = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2597_259753


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l2597_259761

/-- Represents a quadratic equation in one variable x -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- Checks if an equation is quadratic in one variable x -/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (q : QuadraticEquation), ∀ x, f x = q.a * x^2 + q.b * x + q.c

/-- The equation x² = 1 -/
def equation (x : ℝ) : ℝ := x^2 - 1

theorem equation_is_quadratic : is_quadratic_in_x equation := by sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l2597_259761


namespace NUMINAMATH_CALUDE_josh_marbles_l2597_259746

/-- The number of marbles Josh had earlier -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Josh lost -/
def lost_marbles : ℕ := 11

/-- The number of marbles Josh has now -/
def current_marbles : ℕ := 8

/-- Theorem stating that the initial number of marbles is 19 -/
theorem josh_marbles : initial_marbles = lost_marbles + current_marbles := by sorry

end NUMINAMATH_CALUDE_josh_marbles_l2597_259746


namespace NUMINAMATH_CALUDE_inequality_proof_l2597_259791

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hne : ¬(a = b ∧ b = c)) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2597_259791


namespace NUMINAMATH_CALUDE_discount_comparison_l2597_259767

theorem discount_comparison (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_price : x = 2 * y) :
  x + y = (3/2) * (0.6 * x + 0.8 * y) :=
by sorry

#check discount_comparison

end NUMINAMATH_CALUDE_discount_comparison_l2597_259767


namespace NUMINAMATH_CALUDE_jeff_scores_mean_l2597_259739

def jeff_scores : List ℝ := [85, 94, 87, 93, 95, 88, 90]

theorem jeff_scores_mean : 
  (jeff_scores.sum / jeff_scores.length : ℝ) = 90.2857142857 := by
  sorry

end NUMINAMATH_CALUDE_jeff_scores_mean_l2597_259739


namespace NUMINAMATH_CALUDE_base10_89_equals_base5_324_l2597_259777

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Converts a list of digits in base 5 to a natural number --/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 5 * acc + d) 0

theorem base10_89_equals_base5_324 : fromBase5 [4, 2, 3] = 89 := by
  sorry

end NUMINAMATH_CALUDE_base10_89_equals_base5_324_l2597_259777


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2597_259744

theorem cubic_sum_theorem (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 2)
  (eq2 : b^2 + 5*c = 3)
  (eq3 : c^2 + 7*a = 6) :
  a^3 + b^3 + c^3 = -0.875 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2597_259744


namespace NUMINAMATH_CALUDE_pool_volume_l2597_259700

/-- Represents a pool with given parameters -/
structure Pool where
  diameter : ℝ
  fill_time : ℝ
  hose_rates : List ℝ

/-- Calculates the volume of water delivered by hoses over a given time -/
def water_volume (p : Pool) : ℝ :=
  (p.hose_rates.sum * p.fill_time * 60)

/-- The theorem states that a pool with given parameters has a volume of 15000 gallons -/
theorem pool_volume (p : Pool) 
  (h1 : p.diameter = 24)
  (h2 : p.fill_time = 25)
  (h3 : p.hose_rates = [2, 2, 3, 3]) :
  water_volume p = 15000 := by
  sorry

#check pool_volume

end NUMINAMATH_CALUDE_pool_volume_l2597_259700


namespace NUMINAMATH_CALUDE_mountain_trail_length_l2597_259709

/-- Represents the hike of Phoenix on the Mountain Trail --/
structure MountainTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of Phoenix's hike --/
def HikeConditions (hike : MountainTrail) : Prop :=
  hike.day1 + hike.day2 = 28 ∧
  (hike.day2 + hike.day3) / 2 = 15 ∧
  hike.day4 + hike.day5 = 34 ∧
  hike.day1 + hike.day3 = 32

/-- Theorem: The total length of the Mountain Trail is 94 miles --/
theorem mountain_trail_length (hike : MountainTrail) 
  (h : HikeConditions hike) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 94 := by
  sorry


end NUMINAMATH_CALUDE_mountain_trail_length_l2597_259709


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l2597_259781

theorem quiz_competition_participants (total : ℕ) 
  (h1 : (total : ℝ) * (1 - 0.6) * 0.25 = 16) : total = 160 := by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l2597_259781


namespace NUMINAMATH_CALUDE_part_one_part_two_l2597_259751

-- Define the universe set U
def U : Set ℕ := {0, 1, 2, 3}

-- Define the set A as a function of m
def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m*x = 0}

-- Part 1
theorem part_one (m : ℝ) : (Aᶜ m = {1, 2}) → m = -3 := by sorry

-- Part 2
theorem part_two (m : ℝ) : (∃! x, x ∈ A m) → m = 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2597_259751


namespace NUMINAMATH_CALUDE_carolyn_practice_time_l2597_259731

/-- Calculates the total practice time for Carolyn in a month --/
def total_practice_time (piano_time : ℕ) (violin_multiplier : ℕ) (practice_days_per_week : ℕ) (weeks_in_month : ℕ) : ℕ :=
  let violin_time := piano_time * violin_multiplier
  let daily_practice_time := piano_time + violin_time
  let weekly_practice_time := daily_practice_time * practice_days_per_week
  weekly_practice_time * weeks_in_month

/-- Proves that Carolyn's total practice time in a month with 4 weeks is 1920 minutes --/
theorem carolyn_practice_time :
  total_practice_time 20 3 6 4 = 1920 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_practice_time_l2597_259731


namespace NUMINAMATH_CALUDE_tree_growth_fraction_l2597_259733

/-- Represents the height of a tree over time -/
def tree_height (initial_height : ℕ) (growth_rate : ℕ) (years : ℕ) : ℕ :=
  initial_height + growth_rate * years

/-- The fraction representing the increase in height from year a to year b -/
def height_increase_fraction (initial_height : ℕ) (growth_rate : ℕ) (a b : ℕ) : ℚ :=
  (tree_height initial_height growth_rate b - tree_height initial_height growth_rate a) /
  tree_height initial_height growth_rate a

theorem tree_growth_fraction :
  height_increase_fraction 4 1 4 6 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_fraction_l2597_259733


namespace NUMINAMATH_CALUDE_complex_equality_problem_l2597_259723

theorem complex_equality_problem : ∃! (z : ℂ), 
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
  Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_problem_l2597_259723


namespace NUMINAMATH_CALUDE_balls_sold_l2597_259792

theorem balls_sold (selling_price : ℕ) (cost_price : ℕ) (loss : ℕ) : 
  selling_price = 720 →
  loss = 5 * cost_price →
  cost_price = 120 →
  selling_price + loss = 11 * cost_price :=
by
  sorry

end NUMINAMATH_CALUDE_balls_sold_l2597_259792


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l2597_259730

theorem largest_prime_factor_of_1729 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l2597_259730


namespace NUMINAMATH_CALUDE_face_mask_selling_price_l2597_259757

/-- Proves that the selling price of each face mask is $0.50 given the conditions --/
theorem face_mask_selling_price 
  (num_boxes : ℕ)
  (masks_per_box : ℕ)
  (total_cost : ℚ)
  (total_profit : ℚ)
  (h1 : num_boxes = 3)
  (h2 : masks_per_box = 20)
  (h3 : total_cost = 15)
  (h4 : total_profit = 15) :
  (total_cost + total_profit) / (num_boxes * masks_per_box : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_face_mask_selling_price_l2597_259757


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_ge_sum_over_sqrt2_l2597_259740

theorem sqrt_sum_squares_ge_sum_over_sqrt2 (a b : ℝ) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_ge_sum_over_sqrt2_l2597_259740


namespace NUMINAMATH_CALUDE_cos_two_beta_l2597_259741

theorem cos_two_beta (α β : Real) 
  (h1 : Real.sin α = Real.cos β) 
  (h2 : Real.sin α * Real.cos β - 2 * Real.cos α * Real.sin β = 1/2) : 
  Real.cos (2 * β) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cos_two_beta_l2597_259741


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2597_259713

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.I * (m + 1) : ℂ).re = 0 ∧ (Complex.I * (m + 1) : ℂ).im ≠ 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2597_259713


namespace NUMINAMATH_CALUDE_power_of_two_equality_l2597_259778

theorem power_of_two_equality (x : ℕ) : (1 / 16 : ℝ) * (2 ^ 50) = 2 ^ x → x = 46 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l2597_259778


namespace NUMINAMATH_CALUDE_shop_owner_gain_l2597_259762

/-- Represents the problem of calculating the gain in terms of cloth meters for a shop owner. -/
theorem shop_owner_gain (total_meters : ℝ) (gain_percentage : ℝ) (gain_meters : ℝ) : 
  total_meters = 30 ∧ 
  gain_percentage = 50 / 100 → 
  gain_meters = 10 := by
  sorry


end NUMINAMATH_CALUDE_shop_owner_gain_l2597_259762


namespace NUMINAMATH_CALUDE_solution_set_is_open_unit_interval_l2597_259717

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f x < f y

-- Define the set of x values satisfying f(2x-1) < f(1)
def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f (2*x - 1) < f 1}

-- State the theorem
theorem solution_set_is_open_unit_interval (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) : 
  solution_set f = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_unit_interval_l2597_259717


namespace NUMINAMATH_CALUDE_child_growth_l2597_259784

theorem child_growth (current_height previous_height : ℝ) 
  (h1 : current_height = 41.5)
  (h2 : previous_height = 38.5) :
  current_height - previous_height = 3 := by
  sorry

end NUMINAMATH_CALUDE_child_growth_l2597_259784


namespace NUMINAMATH_CALUDE_quadratic_completion_l2597_259701

theorem quadratic_completion (x : ℝ) : ∃ (a b c : ℤ), 
  a > 0 ∧ 
  (a * x + b : ℝ)^2 = 64 * x^2 + 96 * x + c ∧
  a + b + c = 131 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l2597_259701


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_order_l2597_259703

theorem monotonic_decreasing_implies_order (f : ℝ → ℝ) 
  (h : ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) :
  f 3 < f 2 ∧ f 2 < f 1 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_order_l2597_259703


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2597_259776

/-- The number of games in a chess tournament --/
def tournament_games (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * k

/-- Theorem: In a chess tournament with 50 players, where each player plays
    four times with each opponent, the total number of games is 4900 --/
theorem chess_tournament_games :
  tournament_games 50 4 = 4900 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2597_259776


namespace NUMINAMATH_CALUDE_cyclic_triples_count_l2597_259702

/-- Represents a round-robin tournament. -/
structure Tournament where
  n : ℕ  -- number of teams
  wins : ℕ  -- number of wins per team
  losses : ℕ  -- number of losses per team

/-- Calculates the number of cyclic triples in a tournament. -/
def cyclic_triples (t : Tournament) : ℕ :=
  if t.n * (t.n - 1) = 2 * (t.wins + t.losses) ∧ t.wins = 12 ∧ t.losses = 8
  then 665
  else 0

theorem cyclic_triples_count (t : Tournament) :
  t.n * (t.n - 1) = 2 * (t.wins + t.losses) →
  t.wins = 12 →
  t.losses = 8 →
  cyclic_triples t = 665 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_triples_count_l2597_259702


namespace NUMINAMATH_CALUDE_smallest_area_of_P_l2597_259797

/-- Represents a point on the grid --/
structure GridPoint where
  x : Nat
  y : Nat
  label : Nat
  deriving Repr

/-- Defines the properties of the grid --/
def grid : List GridPoint := sorry

/-- Checks if a label is divisible by 7 --/
def isDivisibleBySeven (n : Nat) : Bool :=
  n % 7 == 0

/-- Defines the convex polygon P --/
def P : Set GridPoint := sorry

/-- Calculates the area of a convex polygon --/
noncomputable def areaOfConvexPolygon (polygon : Set GridPoint) : Real := sorry

/-- States that P contains all points with labels divisible by 7 --/
axiom P_contains_divisible_by_seven :
  ∀ p : GridPoint, p ∈ grid → isDivisibleBySeven p.label → p ∈ P

/-- Theorem: The smallest possible area of P is 60.5 square units --/
theorem smallest_area_of_P :
  ∀ Q : Set GridPoint,
    (∀ p : GridPoint, p ∈ grid → isDivisibleBySeven p.label → p ∈ Q) →
    areaOfConvexPolygon P ≤ areaOfConvexPolygon Q ∧
    areaOfConvexPolygon P = 60.5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_area_of_P_l2597_259797


namespace NUMINAMATH_CALUDE_no_positive_solution_l2597_259790

theorem no_positive_solution :
  ¬ ∃ (x y z : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^4 + y^4 + z^4 = 13 ∧
    x^3 * y^3 * z + y^3 * z^3 * x + z^3 * x^3 * y = 6 * Real.sqrt 3 ∧
    x^3 * y * z + y^3 * z * x + z^3 * x * y = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_solution_l2597_259790
